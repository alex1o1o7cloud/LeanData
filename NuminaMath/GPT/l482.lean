import Mathlib

namespace carolyn_practice_time_l482_482011

variable (P : ℕ) -- Number of minutes Carolyn practices the piano each day

theorem carolyn_practice_time (h1 : ∀ (P : ℕ), ∃ P, P * (6 + 18) * 4 = 1920) : P = 20 :=
by
  obtain ⟨P, hP⟩ := h1 P
  have : 24 * 4 * P = 1920 := by rw hP
  have : 96 * P = 1920 := by norm_num
  have : P = 20 := by linarith
  exact this

end carolyn_practice_time_l482_482011


namespace largest_and_smallest_l482_482217

variables {x y : ℝ} (hx : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y)

def Q : ℝ := Real.sqrt ((x^2 + y^2) / 2)
def A : ℝ := (x + y) / 2
def G : ℝ := Real.sqrt (x * y)
def H : ℝ := (2 * x * y) / (x + y)

theorem largest_and_smallest : 
  A - G > Q - A ∧ Q - A > G - H :=
sorry

end largest_and_smallest_l482_482217


namespace complex_number_count_l482_482893

def is_unit_circle (z : ℂ) : Prop := abs z = 1
def is_real (x : ℂ) : Prop := ∃ r : ℝ, x = r

theorem complex_number_count :
  {z : ℂ | is_unit_circle z ∧ is_real (z^(7!) - z^(6!))}.to_finset.card = 5600 := sorry

end complex_number_count_l482_482893


namespace sqrt_72_plus_sqrt_32_l482_482709

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l482_482709


namespace inequality_implies_product_l482_482123

theorem inequality_implies_product (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hineq : 4 * log x + 2 * log y ≥ x^2 + 4 * y - 4) : 
  x * y = Real.sqrt 2 / 2 :=
sorry

end inequality_implies_product_l482_482123


namespace men_with_ac_at_least_12_l482_482598

-- Define the variables and conditions
variable (total_men : ℕ) (married_men : ℕ) (tv_men : ℕ) (radio_men : ℕ) (men_with_all_four : ℕ)

-- Assume the given conditions
axiom h1 : total_men = 100
axiom h2 : married_men = 82
axiom h3 : tv_men = 75
axiom h4 : radio_men = 85
axiom h5 : men_with_all_four = 12

-- Define the number of men with AC
variable (ac_men : ℕ)

-- State the proposition that the number of men with AC is at least 12
theorem men_with_ac_at_least_12 : ac_men ≥ 12 := sorry

end men_with_ac_at_least_12_l482_482598


namespace find_M_of_ratios_l482_482989

variables (Y M B : ℕ)
variables (total_fans : ℕ) (ratio_1_2 : ℕ) (ratio_2_3 : ℕ)

noncomputable def ratio_Y_M := 3 / 2
noncomputable def ratio_M_B := 4 / 5
noncomputable def total_fans := 360

theorem find_M_of_ratios (h1 : ratio_Y_M = 3 / 2) (h2 : ratio_M_B = 4 / 5) (h3 : Y + M + B = total_fans) : M = 96 := by
  sorry

end find_M_of_ratios_l482_482989


namespace correct_derivative_operations_l482_482379

theorem correct_derivative_operations :
  let cond1 := (derivative (λ x : ℝ, 3^x) = λ x, 3^x * Real.log 3)
  let cond2 := (derivative (λ x : ℝ, Real.log x / Real.log 2) = λ x, 1 / (x * Real.log 2))
  let cond3 := (derivative (λ x : ℝ, Real.exp x) = λ x, Real.exp x)
  let cond4 := (derivative (λ x : ℝ, 1 / Real.log x) = λ x, -1 / (x * (Real.log x)^2))
  let cond5 := (derivative (λ x : ℝ, x * Real.exp x) = λ x, Real.exp x + x * Real.exp x)
  cond1 = false ∧
  cond2 = true ∧
  cond3 = true ∧
  cond4 = false ∧
  cond5 = false →
-- Combining all the conditions, we need to prove that we have exactly 2 correct ones
  2 = 2 := sorry

end correct_derivative_operations_l482_482379


namespace equilateral_triangle_division_possible_equilateral_triangle_side_lengths_possible_l482_482328

-- Part (a)
theorem equilateral_triangle_division_possible (n : ℤ) : ∃ k : ℕ, n = 4^k := sorry

-- Part (b)
theorem equilateral_triangle_side_lengths_possible (n : ℕ) : 
  (∃ a b : ℕ, a ≠ b) ∧ 
  (∀ t : ℕ, t ≤ n → (side_length t ∈ {a, b})) := sorry

end equilateral_triangle_division_possible_equilateral_triangle_side_lengths_possible_l482_482328


namespace tap_b_fill_time_l482_482293

theorem tap_b_fill_time (t : ℝ) (h1 : t > 0) : 
  (∀ (A_fill B_fill together_fill : ℝ), 
    A_fill = 1/45 ∧ 
    B_fill = 1/t ∧ 
    together_fill = A_fill + B_fill ∧ 
    (9 * A_fill) + (23 * B_fill) = 1) → 
    t = 115 / 4 :=
by
  sorry

end tap_b_fill_time_l482_482293


namespace quadratic_eq_two_distinct_real_roots_isosceles_triangle_value_of_k_l482_482917

/-- Proof that the quadratic equation x^2 - (2k + 1)x + k^2 + k = 0 has two distinct real roots -/
theorem quadratic_eq_two_distinct_real_roots (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
  (x1 * x2 = k^2 + k ∧ x1 + x2 = 2*k + 1) :=
by
  sorry

/-- For triangle ΔABC with sides AB, AC as roots of x^2 - (2k + 1)x + k^2 + k = 0 and BC = 4, find k when ΔABC is isosceles -/
theorem isosceles_triangle_value_of_k (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 * x2 = k^2 + k ∧ x1 + x2 = 2*k + 1 ∧ 
    ((x1 = 4 ∨ x2 = 4) ∧ (x1 + x2 - 4 isosceles))) →
  (k = 3 ∨ k = 4) :=
by
  sorry

end quadratic_eq_two_distinct_real_roots_isosceles_triangle_value_of_k_l482_482917


namespace fraction_sum_identity_l482_482053

theorem fraction_sum_identity :
  (∑ n in Finset.range 336, (n + 1) * (2 * (n + 1)) * (3 * (n + 1))) / 
  (∑ n in Finset.range 336, (n + 1) * (3 * (n + 1)) * (6 * (n + 1))) = 1 / 3 := 
sorry

end fraction_sum_identity_l482_482053


namespace remainder_of_16_pow_2048_mod_11_l482_482767

theorem remainder_of_16_pow_2048_mod_11 : (16^2048) % 11 = 4 := by
  sorry

end remainder_of_16_pow_2048_mod_11_l482_482767


namespace equilateral_triangle_coloring_l482_482921

-- Context: Definition of an equilateral triangle and coloring properties

structure Triangle :=
  (A B C : Point)
  (equilateral : EquilateralTriangle A B C)
  (color : Point → Color)

def exists_monochromatic_right_triangle (T : Triangle) : Prop :=
  ∃ (P Q R : Point), 
    is_right_triangle P Q R ∧ 
    T.color P = T.color Q ∧ 
    T.color Q = T.color R ∧ 
    on_side P T ∧ 
    on_side Q T ∧ 
    on_side R T

theorem equilateral_triangle_coloring (T : Triangle) : exists_monochromatic_right_triangle T := sorry

end equilateral_triangle_coloring_l482_482921


namespace find_coords_of_P_l482_482740

-- Definitions from the conditions
def line_eq (x y : ℝ) : Prop := x - y - 7 = 0
def is_midpoint (P Q M : ℝ × ℝ) : Prop := 
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Coordinates given in the problem
def P : ℝ × ℝ := (-2, 1)

-- The proof goal
theorem find_coords_of_P : ∃ Q : ℝ × ℝ,
  is_midpoint P Q (1, -1) ∧ 
  line_eq Q.1 Q.2 :=
sorry

end find_coords_of_P_l482_482740


namespace books_leftover_l482_482990

/-- In a certain warehouse, there are 1200 boxes, each containing 35 books. Before repacking, Melvin discovers that 100 books are damaged and discards them. 
He is then instructed to repack the remaining books so that there are 45 books in each new box. After packing as many such boxes as possible, this 
proves how many books Melvin has left over. -/
theorem books_leftover (initial_boxes : ℕ) (books_per_box : ℕ) (damaged_books : ℕ) (books_per_new_box : ℕ) :
  initial_boxes = 1200 →
  books_per_box = 35 →
  damaged_books = 100 →
  books_per_new_box = 45 →
  let total_books := initial_boxes * books_per_box in
  let remaining_books := total_books - damaged_books in
  remaining_books % books_per_new_box = 5 :=
by
  intros
  rw [this, this_1, this_2, this_3]
  let total_books := 1200 * 35
  let remaining_books := total_books - 100
  sorry

end books_leftover_l482_482990


namespace lambda_constant_l482_482517

noncomputable def a (n : ℕ) : ℤ := 2 * n - 5

def b (n : ℕ) : ℤ := 3^(a n + 4)

def T (n : ℕ) : ℤ := (8/3) * (9^n - 1)

theorem lambda_constant {λ : ℤ} :
  (λ : ℚ) = 8 → 
  (∀ n : ℕ, λ * (T n : ℚ) - (b (n + 1) : ℚ) = -3) :=
by
  sorry

end lambda_constant_l482_482517


namespace sqrt_sum_simplify_l482_482670

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l482_482670


namespace tan_sum_angle_l482_482081

open Real

theorem tan_sum_angle {m : ℝ} (h : m ≠ 0) :
  let α := atan (-2) in
  tan (α + (π / 4)) = -1 / 3 :=
by
  have α := atan (-2)
  have tan_α : tan α = -2 := by
    rw [tan_atan]
  have tan_sum_formula : tan (α + (π / 4)) = (tan α + tan (π / 4)) / (1 - tan α * tan (π / 4)) :=
    tan_add α (π / 4)
  simp [tan_α, tan_pi_div_four, tan_sum_formula] at *
  linarith

end tan_sum_angle_l482_482081


namespace probability_is_0_4_l482_482389

def coin_values : List ℕ := [10, 10, 5, 5, 2]

def valid_combination (comb : List ℕ) : Prop :=
  comb.sum ≥ 19

def favorable_outcomes : Finset (Finset ℕ) :=
  {s ∈ coin_values.to_finset.powerset.filter (λ s, s.card = 3) | valid_combination s.val.to_list}

def total_outcomes : Finset (Finset ℕ) :=
  coin_values.to_finset.powerset.filter (λ s, s.card = 3)

def probability : ℚ :=
  favorable_outcomes.card / total_outcomes.card

theorem probability_is_0_4 : probability = 2 / 5 :=
by
  -- Proof will go here
  sorry

end probability_is_0_4_l482_482389


namespace length_of_platform_l482_482795

theorem length_of_platform
  (length_of_train time_crossing_platform time_crossing_pole : ℝ) 
  (length_of_train_eq : length_of_train = 400)
  (time_crossing_platform_eq : time_crossing_platform = 45)
  (time_crossing_pole_eq : time_crossing_pole = 30) :
  ∃ (L : ℝ), (400 + L) / time_crossing_platform = length_of_train / time_crossing_pole :=
by {
  use 200,
  sorry
}

end length_of_platform_l482_482795


namespace radical_axis_fixed_point_l482_482492

/-- 
  Given a triangle ABC and a variable point X on the line BC such that C lies between B and X,
  the radical axis of the incircles of the triangles ABX and ACX passes through a fixed point
  independent of X.
-/
theorem radical_axis_fixed_point (A B C X : Point) (h_line : lies_on_line X B C) (h_between : is_between C B X) :
  ∃ P : Point, ∀ X' : Point, lies_on_line X' B C → is_between C B X' → 
  passes_through_fixed_point (radical_axis (incircle_triangle (A B X')) (incircle_triangle (A C X'))) P := 
sorry

definition lies_on_line (X B C : Point) : Prop := sorry
definition is_between (C B X : Point) : Prop := sorry
definition passes_through_fixed_point (radical_axis P : Prop) := sorry
definition incircle_triangle (A B X : Triangle) : Circle := sorry
definition radical_axis (circle1 circle2 : Circle) : Line := sorry

end radical_axis_fixed_point_l482_482492


namespace count_increasing_decreasing_triples_l482_482970

theorem count_increasing_decreasing_triples : 
  let count := (nat.choose 9 3) + (nat.choose 9 3) in
  168 = count :=
by
  -- Defining the range and properties
  have h1: ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → 
           ∃ (a b c : ℕ), a < b ∧ b < c ∨ a > b ∧ b > c :=
  sorry
  -- Calculating the number of increasing triples
  have h2: (nat.choose 9 3) = 84 :=
  by
    rw [nat.choose_eq_factorial_div_factorial],
    norm_num,
  -- Calculating the number of decreasing triples
  have h3: (nat.choose 9 3) = 84 :=
  by
    rw [nat.choose_eq_factorial_div_factorial],
    norm_num,
  
  -- Summing them up to get the final count
  exact Eq.symm (Nat.add_eq_of_eq_add_eq (Eq.refl 168)),
  
  sorry

end count_increasing_decreasing_triples_l482_482970


namespace exists_special_set_l482_482916

theorem exists_special_set (n : ℕ) (h : n ≥ 3) : 
  ∃ (S : Finset ℕ), S.card = 2 * n ∧ 
    (∀ m : ℕ, 2 ≤ m ∧ m ≤ n → ∃ A : Finset ℕ, A ⊆ S ∧ A.card = m ∧ A.sum = (S.sum / 2) :=
sorry

end exists_special_set_l482_482916


namespace total_pies_sold_l482_482856

theorem total_pies_sold :
  let shepherd_slices := 52
  let chicken_slices := 80
  let shepherd_pieces_per_pie := 4
  let chicken_pieces_per_pie := 5
  let shepherd_pies := shepherd_slices / shepherd_pieces_per_pie
  let chicken_pies := chicken_slices / chicken_pieces_per_pie
  shepherd_pies + chicken_pies = 29 :=
by
  sorry

end total_pies_sold_l482_482856


namespace interest_rate_proof_l482_482263

variable (P : ℝ) (n : ℕ) (CI SI : ℝ → ℝ → ℕ → ℝ) (diff : ℝ → ℝ → ℝ)

def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n
def simple_interest (P r : ℝ) (n : ℕ) : ℝ := P * r * n

theorem interest_rate_proof (r : ℝ) :
  diff (compound_interest 5400 r 2) (simple_interest 5400 r 2) = 216 → r = 0.2 :=
by sorry

end interest_rate_proof_l482_482263


namespace sally_pokemon_cards_bought_equals_20_l482_482665

-- Condition definitions
def initial_sally_cards : ℕ := 27
def initial_dan_cards : ℕ := 41
def additional_sally_cards (x : ℕ) : ℕ := initial_sally_cards + x
def sally_cards_greater_than_dan : Prop := ∀ (x : ℕ), additional_sally_cards(x) = initial_dan_cards + 6 

-- Lean statement for the proof problem
theorem sally_pokemon_cards_bought_equals_20 (x : ℕ) 
  (h : additional_sally_cards x = initial_dan_cards + 6) : x = 20 :=
by 
  sorry

end sally_pokemon_cards_bought_equals_20_l482_482665


namespace solve_for_y_l482_482715

theorem solve_for_y (y : ℤ) (h : 7 - y = 10) : y = -3 :=
sorry

end solve_for_y_l482_482715


namespace sum_of_tangency_points_l482_482018

def g (x : ℝ) : ℝ := max (-7 * x - 21) (max (2 * x - 6) (4 * x + 12))

noncomputable def q (x : ℝ) : ℝ := sorry
-- we assume that q(x) is some quadratic polynomial tangent to g(x) at three distinct points x1, x2, and x3

axiom tangency_points (x1 x2 x3 : ℝ) (q : ℝ → ℝ) :
  (∀ x, q x - (-7 * x - 21) = b * (x - x1)^2) ∧
  (∀ x, q x - (2 * x - 6) = b * (x - x2)^2) ∧
  (∀ x, q x - (4 * x + 12) = b * (x - x3)^2)

theorem sum_of_tangency_points (x1 x2 x3 : ℝ) (q : ℝ → ℝ) (b : ℝ) :
  tangency_points x1 x2 x3 q →
  x1 + x2 + x3 = -19 / 6 :=
sorry

end sum_of_tangency_points_l482_482018


namespace expand_and_simplify_l482_482443

theorem expand_and_simplify :
  ∀ (x : ℝ), 2 * x * (3 * x ^ 2 - 4 * x + 5) - (x ^ 2 - 3 * x) * (4 * x + 5) = 2 * x ^ 3 - x ^ 2 + 25 * x :=
by
  intro x
  sorry

end expand_and_simplify_l482_482443


namespace total_pies_sold_l482_482859

def shepherds_pie_slices_per_pie : Nat := 4
def chicken_pot_pie_slices_per_pie : Nat := 5
def shepherds_pie_slices_ordered : Nat := 52
def chicken_pot_pie_slices_ordered : Nat := 80

theorem total_pies_sold :
  shepherds_pie_slices_ordered / shepherds_pie_slices_per_pie +
  chicken_pot_pie_slices_ordered / chicken_pot_pie_slices_per_pie = 29 := by
sorry

end total_pies_sold_l482_482859


namespace laura_owes_amount_l482_482331

noncomputable def calculate_amount_owed (P R T : ℝ) : ℝ :=
  let I := P * R * T 
  P + I

theorem laura_owes_amount (P : ℝ) (R : ℝ) (T : ℝ) (hP : P = 35) (hR : R = 0.09) (hT : T = 1) :
  calculate_amount_owed P R T = 38.15 := by
  -- Prove that the total amount owed calculated by the formula matches the correct answer
  sorry

end laura_owes_amount_l482_482331


namespace inequality_solution_sets_l482_482586

theorem inequality_solution_sets (a : ℝ)
  (h1 : ∀ x : ℝ, (1/2) < x ∧ x < 2 ↔ ax^2 + 5*x - 2 > 0) :
  a = -2 ∧ (∀ x : ℝ, -3 < x ∧ x < (1/2) ↔ ax^2 - 5*x + a^2 - 1 > 0) :=
by {
  sorry
}

end inequality_solution_sets_l482_482586


namespace committee_vowel_first_rearrangements_count_l482_482967

theorem committee_vowel_first_rearrangements_count :
  let vowels := ['O', 'I', 'E', 'E']
  let consonants := ['C', 'M', 'M', 'T', 'T']
  let vowel_count := Multiset.card vowels
  let consonant_count := Multiset.card consonants
  let vowel_permutations := Nat.fact vowel_count / (Nat.fact (Multiset.count 'E' vowels))
  let consonant_permutations := Nat.fact consonant_count / (Nat.fact (Multiset.count 'M' consonants) * Nat.fact (Multiset.count 'T' consonants))
  vowel_permutations * consonant_permutations = 360 :=
by
  sorry

end committee_vowel_first_rearrangements_count_l482_482967


namespace maxLShapesIn7x7_l482_482766

-- Define a structure for an L-shaped figure
structure LShape :=
  (squares : Set (ℕ × ℕ)) -- represents the coordinates of the 5 squares

-- Example configuration for LShape:
def exampleLShape : LShape :=
  { squares := {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)} }

-- Definition of the 7x7 grid
def grid7x7 : Set (ℕ × ℕ) :=
  { (i, j) | i < 7 ∧ j < 7 }

-- Function to check if L-shaped can be placed in grid without overlapping
def canPlace (ls : LShape) (grid : Set (ℕ × ℕ)) (placement : ℕ × ℕ) : Prop :=
  ls.squares ⊆ grid.map (λ p, (p.1 + placement.1, p.2 + placement.2))

-- Claiming there exist 9 non-overlapping placements in the 7x7 grid
theorem maxLShapesIn7x7 : 
  ∃ (placements : List (ℕ × ℕ)), 
    placements.length = 9 ∧ 
    ∀ placement ∈ placements, canPlace exampleLShape grid7x7 placement ∧
    (∀ p1 p2 ∈ placements, p1 ≠ p2 → 
       (map (λ p, (p.1 + p1.1, p.2 + p1.2)) exampleLShape.squares ∩ 
        map (λ p, (p.1 + p2.1, p.2 + p2.2)) exampleLShape.squares = ∅))
:=
sorry

end maxLShapesIn7x7_l482_482766


namespace problem_I_problem_II_l482_482072

variable (a b c : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := |x - a| + |x - 1|

-- Problem I
theorem problem_I (h : a = 3) : ∀ x : ℝ, f x < 6 ↔ (x > -1 ∧ x < 5) := sorry

-- Problem II
theorem problem_II (h₁ : a + b + c = 1)
    (h₂ : ∀ x : ℝ, f x ≥ (a^2 + b^2 + c^2) / (b + c))
    (h₃ : 0 < a) : a ≤ sqrt 2 - 1 := sorry

end problem_I_problem_II_l482_482072


namespace sum_distinct_prime_factors_of_7pow7_minus_7pow4_l482_482452

noncomputable def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  let factors := (Nat.factors n).erase_dup
  factors.sum

theorem sum_distinct_prime_factors_of_7pow7_minus_7pow4 :
  sum_of_distinct_prime_factors (7 ^ 7 - 7 ^ 4) = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7pow7_minus_7pow4_l482_482452


namespace indistinguishable_distributions_l482_482975

def ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if boxes = 2 && balls = 6 then 4 else 0

theorem indistinguishable_distributions : ways_to_distribute_balls 6 2 = 4 :=
by sorry

end indistinguishable_distributions_l482_482975


namespace sally_bought_cards_l482_482664

def sally_initial : ℕ := 27
def dan_gave : ℕ := 41
def sally_now : ℕ := 88

theorem sally_bought_cards : ∃ (sally_bought : ℕ), sally_bought = sally_now - (sally_initial + dan_gave) :=
by
  let sally_bought := sally_now - (sally_initial + dan_gave)
  use sally_bought
  sorry

end sally_bought_cards_l482_482664


namespace inverse_proportion_function_eq_range_of_m_l482_482071

-- Definitions
def point (x y : ℝ) := (x, y)
def inverse_proportion_function (k : ℝ) (x : ℝ) := k / x

-- Given conditions
def A := point 2 6
def B := point 3 4

-- Questions and answers in Lean 4 statement
theorem inverse_proportion_function_eq (k : ℝ) (A B : ℝ × ℝ) (hA : A = (2, 6)) (hB : B = (3, 4)) :
  ∃ k, inverse_proportion_function k 2 = 6 ∧ inverse_proportion_function k 3 = 4 :=
by sorry

theorem range_of_m (A B : ℝ × ℝ) (hA : A = (2, 6)) (hB : B = (3, 4)) :
  ∃ m : set.Icc (4 / 3) 3, ∀ P ∈ set.Icc (fst A) (fst B), m = snd P / fst P :=
by sorry

end inverse_proportion_function_eq_range_of_m_l482_482071


namespace train_length_is_correct_l482_482377

-- Defining necessary variables and constants
def train_speed_kmph := 45 -- km/hr
def bridge_length := 215 -- meters
def crossing_time := 30 -- seconds

noncomputable def train_speed_mps := (train_speed_kmph * 1000) / 3600 -- m/s

def distance_travelled := train_speed_mps * crossing_time

-- Theorem stating that the length of the train equals 160 meters
theorem train_length_is_correct : 
  ∃ l_t : ℝ, l_t = distance_travelled - bridge_length ∧ l_t = 160 :=
by { 
  unfold train_speed_mps,
  unfold distance_travelled,
  unfold train_speed_kmph,
  unfold bridge_length,
  unfold crossing_time,
  sorry
}

end train_length_is_correct_l482_482377


namespace find_k_l482_482269

-- Define the points and their coordinates
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P₁ : Point := ⟨2, -3⟩
def P₂ : Point := ⟨4, 3⟩
def P₃ (k : ℝ) : Point := ⟨5, k / 2⟩

-- Define the condition that three points are collinear
def collinear (P Q R : Point) : Prop :=
  (Q.y - P.y) * (R.x - P.x) = (R.y - P.y) * (Q.x - P.x)

-- The actual theorem statement we want to prove
theorem find_k : ∃ k : ℝ, collinear P₁ P₂ (P₃ k) ∧ k = 12 :=
by
  sorry

end find_k_l482_482269


namespace probability_of_picking_combination_is_0_4_l482_482386

noncomputable def probability_at_least_19_rubles (total_coins total_value: ℕ) :=
  let coins := [10, 10, 5, 5, 2] in
  let all_combinations := (Finset.powersetLen 3 (coins.to_finset)).to_list in
  let favorable_combinations := all_combinations.filter (fun c => c.sum ≥ total_value) in
  (favorable_combinations.length : ℚ) / (all_combinations.length : ℚ)

theorem probability_of_picking_combination_is_0_4 :
  probability_at_least_19_rubles 5 19 = 0.4 :=
by
  sorry

end probability_of_picking_combination_is_0_4_l482_482386


namespace lateral_surface_area_of_cone_l482_482730

noncomputable def cone_lateral_surface_area (r l : ℝ) : ℝ :=
  π * r * l

theorem lateral_surface_area_of_cone : cone_lateral_surface_area 2 4 = 8 * π :=
by
  -- conditions from a)
  let base_radius := 2
  let slant_height := 4
  -- Proof left as sorry as per task instructions
  sorry

end lateral_surface_area_of_cone_l482_482730


namespace largest_area_of_rotating_triangle_l482_482164

def Point := (ℝ × ℝ)

def A : Point := (0, 0)
def B : Point := (13, 0)
def C : Point := (21, 0)

def line (P : Point) (slope : ℝ) (x : ℝ) : ℝ := P.2 + slope * (x - P.1)

def l_A (x : ℝ) : ℝ := line A 1 x
def l_B (x : ℝ) : ℝ := x
def l_C (x : ℝ) : ℝ := line C (-1) x

def rotating_triangle_max_area (l_A l_B l_C : ℝ → ℝ) : ℝ := 116.5

theorem largest_area_of_rotating_triangle :
  rotating_triangle_max_area l_A l_B l_C = 116.5 :=
sorry

end largest_area_of_rotating_triangle_l482_482164


namespace number_of_male_alligators_l482_482198

def total_alligators (female_alligators : ℕ) : ℕ := 2 * female_alligators
def female_juveniles (female_alligators : ℕ) : ℕ := 2 * female_alligators / 5 
def adult_females (female_alligators : ℕ) : ℕ := (3 * female_alligators) / 5

def male_alligators (total_alligators : ℕ) := total_alligators / 2

theorem number_of_male_alligators
    (half_male : ∀ total, male_alligators total = total / 2) 
    (adult_female_count : ∀ female, adult_females female = 15) 
    (female_count : ∃ female, adult_females female = 15) :
  (2 * classical.some female_count) / 2 = 25 :=
by
  have female_count := classical.some female_count
  rw [(adult_female_count female_count)]
  have total := 2 * female_count
  rw [← half_male total]
  sorry

end number_of_male_alligators_l482_482198


namespace find_f2_l482_482944

-- Definition of the function f
def f (x : ℝ) (a : ℤ) : ℝ := x ^ (a^2 - 2*a - 3)

-- Statement of the proof problem
theorem find_f2 (a : ℤ) (h1 : ∀ x > 0, (x:ℝ) ^ (a^2 - 2*a - 3) < x ^ (a^2 - 2*a - 3))
    (h2 : ∀ x : ℝ, f x a = f (-x) a ∨ a^2 - 2*a - 3 < 0)
    (h3 : a = 1) :
  f 2 a = 1 / 16 := 
sorry

end find_f2_l482_482944


namespace balance_difference_is_7292_83_l482_482383

noncomputable def angela_balance : ℝ := 7000 * (1 + 0.05)^15
noncomputable def bob_balance : ℝ := 9000 * (1 + 0.03)^30
noncomputable def balance_difference : ℝ := bob_balance - angela_balance

theorem balance_difference_is_7292_83 : balance_difference = 7292.83 := by
  sorry

end balance_difference_is_7292_83_l482_482383


namespace complement_of_A_in_B_l482_482547

-- Define the sets A and B based on the given conditions
def setA (x : ℝ) : Prop := x^2 - 2*x - 3 < 0
def setB (x : ℝ) : Prop := 2^(x + 1) > 1

-- Define the complement of set A in set B
def complementB_A (x : ℝ) : Prop := setB x ∧ ¬setA x

-- The theorem we need to prove stating that the complement of set A in set B equals [3, +∞)
theorem complement_of_A_in_B : {x : ℝ | complementB_A x} = {x | 3 ≤ x} :=
by
  sorry

end complement_of_A_in_B_l482_482547


namespace triangle_ABC_a_squared_eq_bb_plus_bc_triangle_area_magnitude_B_l482_482147
-- Proof Problem 1: 


theorem triangle_ABC_a_squared_eq_bb_plus_bc (A B C a b c : ℝ) 
  (h1 : a = 2*b)
  (h2 : B + C = π / 2)
  : a^2 = b * (b + c) := sorry

-- Proof Problem 2: 


theorem triangle_area_magnitude_B (A B C a b c : ℝ) 
  (h1 : A = 2*B) 
  (h2 : (1/2) * a * b * sin(B) = (1/4) * a^2)
  : B = π / 4 ∨ B = π / 8 := sorry

end triangle_ABC_a_squared_eq_bb_plus_bc_triangle_area_magnitude_B_l482_482147


namespace merchant_discount_l482_482779

theorem merchant_discount (C : ℝ) (D : ℝ) :
  let M := 1.75 * C,
      S := 1.575 * C in
  S = M - (D / 100) * M → D = 10 :=
by
  sorry

end merchant_discount_l482_482779


namespace arrangement_impossible_l482_482622

-- Define the bounded sequence
def sequence : List ℕ := List.range' 1 1980

-- Define the required condition
def condition (a : List ℕ) : Prop :=
  ∀ i : ℕ, i + 2 < a.length → (a.get! i + a.get! (i + 2)) % 3 = 0

-- The proof problem
theorem arrangement_impossible : ¬ ∃ a : List ℕ, a = sequence ∧ condition a :=
by
  -- Proof goes here
  sorry

end arrangement_impossible_l482_482622


namespace board_cut_ratio_l482_482794

theorem board_cut_ratio (L S : ℝ) (h1 : S + L = 20) (h2 : S = L + 4) (h3 : S = 8.0) : S / L = 1 := by
  sorry

end board_cut_ratio_l482_482794


namespace charlyn_visible_area_approx_l482_482012

noncomputable def visible_area (side_length km_view : ℝ) : ℝ := 
  let inner_area := side_length^2 - (side_length - 2 * km_view)^2 in
  let outer_rect_area := 4 * side_length * km_view in
  let outer_circle_area := 4 * (real.pi * km_view^2 / 4) in
  inner_area + outer_rect_area + outer_circle_area

theorem charlyn_visible_area_approx (side_length km_view : ℝ) 
  (h1 : side_length = 7) (h2 : km_view = 2) :
  abs (visible_area side_length km_view - 109) < 1 :=
  by 
    unfold visible_area
    rw [h1, h2]
    norm_num
    sorry

end charlyn_visible_area_approx_l482_482012


namespace find_a17_a18_a19_a20_l482_482618

variable {α : Type*} [Field α]

-- Definitions based on the given conditions:
def geometric_sequence (a : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, a n = a 0 * r ^ n

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n : ℕ, S n = (Finset.range n).sum a

-- Problem statement based on the question and conditions:
theorem find_a17_a18_a19_a20 (a S : ℕ → α) (h_geom : geometric_sequence a)
  (h_sum : sum_of_first_n_terms a S) (hS4 : S 4 = 1) (hS8 : S 8 = 3) :
  a 17 + a 18 + a 19 + a 20 = 16 :=
sorry

end find_a17_a18_a19_a20_l482_482618


namespace no_positive_integer_solutions_l482_482024

theorem no_positive_integer_solutions :
  ∀ (A : ℕ), 1 ≤ A ∧ A ≤ 9 → ¬∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * y = A * 10 + A ∧ x + y = 10 * A + 1 := by
  sorry

end no_positive_integer_solutions_l482_482024


namespace distance_between_A_and_B_l482_482294

-- Let d be the unknown distance we need to find
variable (d : ℚ)

-- Condition when Jia reaches the midpoint of AB
def jia_midpoint : Prop := d / 2 + (d / 2 - 5) = d - 5

-- Condition when Yi reaches the midpoint of AB
def yi_midpoint : Prop := d - (d / 2 - 45 / 8) = 45 / 8

-- The theorem stating that under given conditions, the distance d is 90 km
theorem distance_between_A_and_B :
  jia_midpoint d ∧ yi_midpoint d → d = 90 := 
sorry -- Proof is omitted

end distance_between_A_and_B_l482_482294


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482461

def seven_pow_seven_minus_seven_pow_four : ℤ := 7^7 - 7^4
def prime_factors_of_three_hundred_forty_two : List ℤ := [2, 3, 19]

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let distinct_prime_factors := prime_factors_of_three_hundred_forty_two.head!
  + prime_factors_of_three_hundred_forty_two.tail!.head!
  + prime_factors_of_three_hundred_forty_two.tail!.tail!.head!
  seven_pow_seven_minus_seven_pow_four = 7^4 * (7^3 - 1) ∧
  7^3 - 1 = 342 ∧
  prime_factors_of_three_hundred_forty_two = [2, 3, 19] ∧
  distinct_prime_factors = 24 := 
sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482461


namespace arithmetic_sequence_of_sides_minimum_cos_C_l482_482146

variable {A B C a b c : ℝ}
variable {α β γ : Type*} [metric_space α] [metric_space β] [metric_space γ]

-- Given conditions
def triangle_conditions : Prop :=
  -- sides opposite to angles A, B, C are a, b, c respectively
  True -- simply meaning the triangle ABC with sides a, b, c exists,

def tan_identity_condition (A B : ℝ) : Prop :=
  2 * (Real.tan A + Real.tan B) = (Real.sin A + Real.sin B) / (Real.cos A * Real.cos B)

-- Proof goals
theorem arithmetic_sequence_of_sides (h1: triangle_conditions) (h2: tan_identity_condition A B) : a + b = 2 * c :=
sorry

theorem minimum_cos_C (h1: triangle_conditions) (h2: tan_identity_condition A B) : Real.cos C >= 1/2 :=
sorry

end arithmetic_sequence_of_sides_minimum_cos_C_l482_482146


namespace area_triangle_APB_l482_482367

theorem area_triangle_APB (A B C D P : ℝ × ℝ)
  (h_square : (A.1 = 0 ∧ A.2 = 0) ∧ (B.1 = 8 ∧ B.2 = 0) ∧ (C.1 = 8 ∧ C.2 = 8) ∧ (D.1 = 0 ∧ D.2 = 8))
  (h_eq : dist P A = dist P B ∧ dist P B = dist P D)
  (h_perp : P.1 = 4 ∧ P.2 = 4)
  : area (triangle A P B) = 16 :=
sorry

end area_triangle_APB_l482_482367


namespace parabola_equation_and_no_ellipse_l482_482957

-- Definitions of the premises
def parabola (p : ℝ) : Set (ℝ × ℝ) := { P | ∃ y x, P = (x, y) ∧ y^2 = 2 * p * x }

def E (p : ℝ) : ℝ × ℝ := (p^2 / 4, 0)

def line_intersects_parabola (l : ℝ → ℝ) (p : ℝ) : Prop :=
  ∃ x1 y1 x2 y2, parabola p (x1, y1) ∧ parabola p (x2, y2) ∧
  l y1 = x1 ∧ l y2 = x2 ∧ x1 + x2 + p = abs (x1 - x2)

def perpendicular_line (l l' : ℝ → ℝ) : Prop :=
  ∀ x, l' x = - (1 / (l x))

def symmetric_point (E C : ℝ × ℝ) : ℝ × ℝ := (2 * E.1 - C.1, 2 * E.2 - C.2)

-- Main theorem statement
theorem parabola_equation_and_no_ellipse (p : ℝ) (l l' : ℝ → ℝ)
  (h_pos : 0 < p) (h_line : line_intersects_parabola l p)
  (h_perpendicular : perpendicular_line l l')
  (h_slope : ∀ x, l x > 0) :
  (∀ x y, parabola p (x, y) ↔ y^2 = 4 * x) ∧
  ¬(∃ e : ℝ, e = sqrt 3 / 2 ∧
    ∃ f, ∀ (A B C' D' : ℝ × ℝ),
      parabola p A ∧ parabola p B ∧
      parabola p C ∧ parabola p D ∧
      symmetric_point (E p) C = C' ∧
      symmetric_point (E p) D = D' ∧
      (x1, y1) = A ∧ (x2, y2) = B ∧ 
      (x3, y3) = C ∧ (x4, y4) = D ∧ 
      ((x1^2) / (4*f^2) + (y1^2) / (f^2) = 1) ∧ 
      ((x2^2) / (4*f^2) + (y2^2) / (f^2) = 1) ∧ 
      ((x3^2) / (4*f^2) + (y3^2) / (f^2) = 1) ∧ 
      ((x4^2) / (4*f^2) + (y4^2) / (f^2) = 1)) :=
by sorry

end parabola_equation_and_no_ellipse_l482_482957


namespace find_MO_minus_MT_l482_482372

noncomputable def hyperbola_focus_left (a b : ℝ) : ℝ :=
  - sqrt (a^2 + b^2)

noncomputable def hyperbola_focus_right (a b : ℝ) : ℝ :=
  sqrt (a^2 + b^2)

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def midpoint (x1 y1 x2 y2 : ℝ) : (ℝ, ℝ) :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

theorem find_MO_minus_MT :
  (m : ℝ) (O : ℝ × ℝ := (0, 0)) (T : ℝ × ℝ) (F : ℝ × ℝ := (hyperbola_focus_left 3 4, 0)) 
  (hyperbola_point : ℝ × ℝ) (M : ℝ × ℝ := midpoint F.1 F.2 hyperbola_point.1 hyperbola_point.2) 
  (c : ℝ := 1) (ht1 : T = (-3, ?T_y))
  (ht2 : ∀ x y : ℝ, (x^2 + y^2 = 9) ∧ y^2 = 9 - x^2 ∧ (x = F.1 - 3 * y) ∧ (distance F.1 F.2 x y = sqrt 5)) :
  abs (distance M.1 M.2 O.1 O.2 - distance M.1 M.2 T.1 T.2) = 1 :=
by
  sorry

end find_MO_minus_MT_l482_482372


namespace find_n_l482_482067

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = 2 * a n

def sum_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = Σ k in finset.range n, a k

theorem find_n (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :
  sequence a ∧ sum_sequence a S ∧ S n = 126 → n = 6 :=
by
  sorry

end find_n_l482_482067


namespace sqrt_sum_simplify_l482_482673

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l482_482673


namespace middle_number_is_11_l482_482735

variables {a b c d e : ℕ}

theorem middle_number_is_11 (h1 : {a, b, c, d, e} = {7, 8, 9, 10, 11})
  (h2 : a + b + c = 26)
  (h3 : c + d + e = 30)
  (h_sum : a + b + c + d + e = 45) :
  c = 11 :=
by
  sorry

end middle_number_is_11_l482_482735


namespace shortest_distance_line_segment_l482_482316

noncomputable theory

open Real

-- Definition of a point in 2D space
structure Point (ℝ : Type*) :=
(x : ℝ)
(y : ℝ)

-- Definition of a line segment distance function
def line_segment_distance (A B : Point ℝ) : ℝ :=
real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

-- Proving that the line segment between points A and B is the shortest distance
theorem shortest_distance_line_segment (A B : Point ℝ) :
  ∀ P : Point ℝ, (real.sqrt ((P.x - A.x)^2 + (P.y - A.y)^2) + real.sqrt ((B.x - P.x)^2 + (B.y - P.y)^2)) 
  ≥ line_segment_distance A B :=
sorry

end shortest_distance_line_segment_l482_482316


namespace quadruplet_zero_solution_l482_482037

theorem quadruplet_zero_solution (a b c d : ℝ)
  (h1 : (a + b) * (a^2 + b^2) = (c + d) * (c^2 + d^2))
  (h2 : (a + c) * (a^2 + c^2) = (b + d) * (b^2 + d^2))
  (h3 : (a + d) * (a^2 + d^2) = (b + c) * (b^2 + c^2)) :
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := 
sorry

end quadruplet_zero_solution_l482_482037


namespace mass_percentage_of_C_in_CCl4_l482_482892

theorem mass_percentage_of_C_in_CCl4 :
  let mass_carbon : ℝ := 12.01
  let mass_chlorine : ℝ := 35.45
  let molar_mass_CCl4 : ℝ := mass_carbon + 4 * mass_chlorine
  let mass_percentage_C : ℝ := (mass_carbon / molar_mass_CCl4) * 100
  mass_percentage_C = 7.81 := 
by
  sorry

end mass_percentage_of_C_in_CCl4_l482_482892


namespace eccentricity_range_l482_482101

open Real

variable (a b x_o c e : ℝ)
variable (h1 : a > 0) (h2 : b > 0)
variable (h3 : x_o > a)
variable (h4 : c = sqrt (a^2 + b^2))
variable (h5 : ∀ P : ℝ × ℝ, 
  ∃ P : ℝ × ℝ, 
    P.1^2 / a^2 - P.2^2 / b^2 = 1 ∧
    (sin (arcsin ((P.1 + c) / sqrt ((P.1 - c)^2 + P.2^2)) / arcsin ((P.1 - c) / sqrt ((P.1 + c)^2 + P.2^2))) = a / c))

theorem eccentricity_range : 1 < e ∧ e < sqrt 2 + 1 :=
sorry

end eccentricity_range_l482_482101


namespace probability_is_0_4_l482_482390

def coin_values : List ℕ := [10, 10, 5, 5, 2]

def valid_combination (comb : List ℕ) : Prop :=
  comb.sum ≥ 19

def favorable_outcomes : Finset (Finset ℕ) :=
  {s ∈ coin_values.to_finset.powerset.filter (λ s, s.card = 3) | valid_combination s.val.to_list}

def total_outcomes : Finset (Finset ℕ) :=
  coin_values.to_finset.powerset.filter (λ s, s.card = 3)

def probability : ℚ :=
  favorable_outcomes.card / total_outcomes.card

theorem probability_is_0_4 : probability = 2 / 5 :=
by
  -- Proof will go here
  sorry

end probability_is_0_4_l482_482390


namespace simplify_sum_of_square_roots_l482_482678

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l482_482678


namespace paint_32_cells_possible_paint_33_cells_possible_l482_482844

def cell := (ℕ × ℕ)

def board := fin 7 × fin 7

def adjacent (c1 c2 : cell) : Prop :=
  (c1.1 = c2.1 ∧ (c1.2 + 1 = c2.2 ∨ c1.2 = c2.2 + 1)) ∨
  (c1.2 = c2.2 ∧ (c1.1 + 1 = c2.1 ∨ c1.1 = c2.1 + 1))

-- Define the rule of painting cells
def valid_painting (cells : list cell) : Prop :=
  ∀ (i j : ℕ) (h1 : i < cells.length) (h2 : j < cells.length), 
    i ≠ j → adjacent (cells.nth_le i h1) (cells.nth_le j h2) →
    ∃ k, k ≠ i ∧ k ≠ j ∧ adjacent (cells.nth_le i h1) (cells.nth_le k sorry) ∨ adjacent (cells.nth_le j h2) (cells.nth_le k sorry)

-- We need to prove that we can paint 32 and 33 cells following the rules.
theorem paint_32_cells_possible : ∃ (cells : list cell), cells.length = 32 ∧ valid_painting cells := sorry

theorem paint_33_cells_possible : ∃ (cells : list cell), cells.length = 33 ∧ valid_painting cells := sorry

end paint_32_cells_possible_paint_33_cells_possible_l482_482844


namespace children_total_savings_l482_482195

theorem children_total_savings :
  let josiah_savings := 0.25 * 24
  let leah_savings := 0.50 * 20
  let megan_savings := (2 * 0.50) * 12
  josiah_savings + leah_savings + megan_savings = 28 := by
{
  -- lean proof goes here
  sorry
}

end children_total_savings_l482_482195


namespace p_100_bound_l482_482154

-- Definitions
def p (n : ℕ) : ℝ := sorry -- Probability function definition

-- Initial conditions
axiom p0 : p 0 = 1
axiom p_minus1 : p (-1) = 0
axiom p_minus2 : p (-2) = 0
axiom p_minus3 : p (-3) = 0

-- Recurrence relation for n ≥ 1
axiom recurrence (n : ℕ) (h : n ≥ 1) : 
  p n = (1/4) * p (n - 1) + (1/4) * p (n - 2) + (1/4) * p (n - 3) + (1/4) * p (n - 4)

-- Theorem to prove
theorem p_100_bound : 0.399999 < p 100 ∧ p 100 < 0.400001 :=
  sorry

end p_100_bound_l482_482154


namespace isosceles_trapezoid_circumcircle_radius_l482_482450

theorem isosceles_trapezoid_circumcircle_radius :
  (radius_circumscribed_circle
    (isosceles_trapezoid
      { base1 := 2, base2 := 14, side := 10 })) = 5 * sqrt 2 := sorry

end isosceles_trapezoid_circumcircle_radius_l482_482450


namespace total_pies_sold_l482_482861

def shepherds_pie_slices_per_pie : Nat := 4
def chicken_pot_pie_slices_per_pie : Nat := 5
def shepherds_pie_slices_ordered : Nat := 52
def chicken_pot_pie_slices_ordered : Nat := 80

theorem total_pies_sold :
  shepherds_pie_slices_ordered / shepherds_pie_slices_per_pie +
  chicken_pot_pie_slices_ordered / chicken_pot_pie_slices_per_pie = 29 := by
sorry

end total_pies_sold_l482_482861


namespace simplify_radicals_l482_482696

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l482_482696


namespace a_sequence_b_sequence_sum_T_n_l482_482961

-- Define the sequences a_n and b_n
def a (n : ℕ) : ℝ :=
  if n = 1 then 2 else (1 / 2^(n-1))

def b (n : ℕ) : ℝ :=
  if n = 1 then 1 else n

-- Define the sum T_n
def T_n (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a(i+1) * b(i+1)

-- The problem statements as Lean 4 declarations
theorem a_sequence (n : ℕ) : a (n + 1) = 2 * a n := sorry

theorem b_sequence (n : ℕ) : b (n + 1) = (1/2) * b 2 + (1/3) * b 3 + ... + (1/n) * b n + b (n + 1) - 1 := sorry

theorem sum_T_n (n : ℕ) : T_n n = 8 - (n+2)/(2^(n-2)) := sorry

end a_sequence_b_sequence_sum_T_n_l482_482961


namespace general_term_sequence_l482_482068

theorem general_term_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (S_def : ∀ n, S n = ∑ i in Finset.range (n + 1), a i)
  (rec_def : ∀ n, a n = 3 * S n - 2) :
  ∀ n, a n = (-1/2)^ (n - 1) :=
sorry

end general_term_sequence_l482_482068


namespace number_of_twos_l482_482562

theorem number_of_twos (x : Fin 25 → ℕ) (h_sum : (∑ i, x i) = 70) (h_bounds : ∀ i, 1 ≤ x i ∧ x i ≤ 20) : 
  ∑ i in (Finset.filter (λ i, x i = 2) Finset.univ), 1 = 5 :=
by
  sorry

end number_of_twos_l482_482562


namespace other_train_length_l482_482759

theorem other_train_length
  (speed_train1_kmph : ℝ)
  (speed_train2_kmph : ℝ)
  (length_train1_m : ℝ)
  (cross_time_s : ℝ) :
  speed_train1_kmph = 60 →
  speed_train2_kmph = 40 →
  length_train1_m = 180 →
  cross_time_s = 11.519078473722104 →
  let relative_speed_mps := ((speed_train1_kmph + speed_train2_kmph) * 1000 / 3600 : ℝ) in
  let total_distance_m := relative_speed_mps * cross_time_s in
  total_distance_m - length_train1_m ≈ 140 :=
by
  intros h1 h2 h3 h4
  let relative_speed_mps := ((speed_train1_kmph + speed_train2_kmph) * 1000 / 3600 : ℝ)
  let total_distance_m := relative_speed_mps * cross_time_s
  calc total_distance_m - length_train1_m ≈ 140 : sorry

end other_train_length_l482_482759


namespace binomial_square_l482_482002

theorem binomial_square (a b : ℝ) : (2 * a - 3 * b)^2 = 4 * a^2 - 12 * a * b + 9 * b^2 :=
by
  sorry

end binomial_square_l482_482002


namespace solve_problem_l482_482640

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the function f(n)
def f (n : ℕ) : ℕ :=
  sum_of_digits (n ^ 2 + 1)

-- Define the recursive sequence f_k(n)
noncomputable def f_k : ℕ → ℕ → ℕ
| 0, n := n
| (k+1), n := f (f_k k n)

-- Prove that f_100(1990) = 11
theorem solve_problem : f_k 100 1990 = 11 :=
  sorry

end solve_problem_l482_482640


namespace total_number_of_cows_l482_482993

variable (D C : ℕ) -- D is the number of ducks and C is the number of cows

-- Define the condition given in the problem
def legs_eq : Prop := 2 * D + 4 * C = 2 * (D + C) + 28

theorem total_number_of_cows (h : legs_eq D C) : C = 14 := by
  sorry

end total_number_of_cows_l482_482993


namespace amy_bike_miles_l482_482837

theorem amy_bike_miles (yesterday_miles : ℕ) (total_miles : ℕ) (today_miles_less : ℕ) :
  yesterday_miles = 12 → total_miles = 33 → 12 + (24 - today_miles_less) = 33 → today_miles_less = 3 := by
  intros h1 h2 h3
  rw [←h1, ←h2] at h3
  linarith

end amy_bike_miles_l482_482837


namespace max_val_l482_482247

def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)
def g (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3) + 1

theorem max_val (x1 x2 : ℝ) (H1 : g x1 * g x2 = 9) (H2 : x1 ∈ Icc (-2 * π) (2 * π)) (H3 : x2 ∈ Icc (-2 * π) (2 * π)) :
  2 * x1 - x2 ≤ 49 * π / 12 :=
sorry

end max_val_l482_482247


namespace num_black_circles_in_first_120_circles_l482_482825

theorem num_black_circles_in_first_120_circles : 
  let S := λ n : ℕ, n * (n + 1) / 2 in
  ∃ n : ℕ, S n < 120 ∧ 120 ≤ S (n + 1) := 
by
  sorry

end num_black_circles_in_first_120_circles_l482_482825


namespace true_proposition_l482_482924

def f (x : ℝ) : ℝ := (2017 ^ x - 1) / (2017 ^ x + 1)
def g (x : ℝ) : ℝ := x^3 - x^2

def p : Prop := ∀ x : ℝ, f (-x) = -f x
def q : Prop := ∀ x > 0, g x > g (x)

theorem true_proposition : p ∨ q :=
by
  -- Here the proof is omitted
  -- sorry

end true_proposition_l482_482924


namespace part1_part2_l482_482513

def sequence_a (n : ℕ) : ℕ := 2^n - 1

def sequence_b (n : ℕ) : ℕ := (sequence_a n) * (Nat.log2 (sequence_a n + 1))

def sum_T (n : ℕ) : ℕ := (Finset.range (n + 1)).sum (λ i, sequence_b (i + 1))

theorem part1 : ∀ (n : ℕ), ∃ r : ℕ, (sequence_a n + 1) = r^(n + 1) := 
sorry

theorem part2 : ∃ n : ℕ, (n > 0) ∧ (sum_T n + (n*(n+1))/2 > 2015) :=
sorry

end part1_part2_l482_482513


namespace find_amount_l482_482038

theorem find_amount (amount : ℝ) (h : 0.25 * amount = 75) : amount = 300 :=
sorry

end find_amount_l482_482038


namespace problem1_problem2_l482_482077

-- Problem 1
theorem problem1 (α : ℝ) (h : (Real.tan α) / (Real.tan α - 1) = -1) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -5 / 3 :=
by sorry

-- Problem 2
theorem problem2 (α : ℝ) (h : (Real.tan α) / (Real.tan α - 1) = -1) (h_quad : π < α ∧ α < 3 * π / 2) :
  Real.cos (-π + α) + Real.cos (π / 2 + α) = 3 * Real.sqrt 5 / 5 :=
by sorry

end problem1_problem2_l482_482077


namespace greatest_possible_value_of_sum_l482_482564

theorem greatest_possible_value_of_sum (x : ℝ) (h : 13 = x^2 + 1/x^2) : 
  max (x + 1/x) (- (x + 1/x)) = sqrt 15 :=
sorry

end greatest_possible_value_of_sum_l482_482564


namespace problem_statement_l482_482542

noncomputable def f (x k : ℝ) := x^3 / (2^x + k * 2^(-x))

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def k2_eq_1_is_nec_but_not_suff (f : ℝ → ℝ) (k : ℝ) : Prop :=
  (k^2 = 1) → (is_even_function f → k = -1 ∧ ¬(k = 1))

theorem problem_statement (k : ℝ) :
  k2_eq_1_is_nec_but_not_suff (λ x => f x k) k :=
by
  sorry

end problem_statement_l482_482542


namespace max_area_of_triangle_l482_482145

noncomputable def max_area_triangle_ABC (BC AB AC : ℝ) (hBC : BC = 3) (hAB : AB = 2 * AC) : ℝ :=
  let x := AC in (3 / 2) * x * Real.sqrt (1 - (Real.cos C)^2)

theorem max_area_of_triangle (BC AC : ℝ) (hBC : BC = 3) (hAB : 2 * AC = AB) : 
  ∃ x, AC = x ∧ AB = 2 * x ∧ 1 < x ∧ x < 3 ∧ max_area_triangle_ABC BC AB AC hBC hAB = 3 :=
sorry

end max_area_of_triangle_l482_482145


namespace problem_proof_l482_482200

noncomputable def bounded (A : set ℝ) : Prop :=
  ∃ M : ℝ, ∀ x ∈ A, abs x < M

def closed (C : set ℝ) : Prop := is_closed C

def convex (C : set ℝ) : Prop :=
  ∀ (x y : ℝ) (t : ℝ), x ∈ C → y ∈ C → 0 ≤ t → t ≤ 1 → (t * x + (1 - t) * y ∈ C)

theorem problem_proof
  {A B C : set ℝ}
  (hA_nonempty : A.nonempty)
  (hB_nonempty : B.nonempty)
  (hC_nonempty : C.nonempty)
  (hA_bounded : bounded A)
  (hC_closed : closed C)
  (hC_convex : convex C)
  (h_subset : (λ (x : ℝ), ∃ a ∈ A, ∃ b ∈ B, x = a + b) ⊆ (λ (x : ℝ), ∃ a ∈ A, ∃ c ∈ C, x = a + c)) :
  B ⊆ C := sorry

end problem_proof_l482_482200


namespace length_of_BC_as_fraction_of_AD_l482_482238

theorem length_of_BC_as_fraction_of_AD 
  (A B C D : Point)
  (hADB : B ∈ line_segment A D)
  (hADC : C ∈ line_segment A D)
  (hAB_3BD : ∃ x : ℝ, length (A, B) = 3 * x ∧ length (B, D) = x)
  (hAC_8CD : ∃ y : ℝ, length (A, C) = 8 * y ∧ length (C, D) = y)
  : length (B, C) = (5 / 36) * length (A, D) :=
by
  sorry

end length_of_BC_as_fraction_of_AD_l482_482238


namespace determine_age_l482_482319

def David_age (D Y : ℕ) : Prop := Y = 2 * D ∧ Y = D + 7

theorem determine_age (D : ℕ) (h : David_age D (D + 7)) : D = 7 :=
by
  sorry

end determine_age_l482_482319


namespace smallest_integer_congruent_l482_482311

noncomputable def smallest_four_digit_negative_integer_congr_1_mod_37 : ℤ :=
-1034

theorem smallest_integer_congruent (n : ℤ) (h1 : 37 * n + 1 < -999) : 
  ∃ (k : ℤ), (37 * k + 1) = -1034 ∧ 
  -10000 < (37 * k + 1) ∧ (37 * k + 1 = 1 % 37) := 
by {
  use -28,
  split,
  { refl },
  split,
  { sorry },
  { sorry }
}

end smallest_integer_congruent_l482_482311


namespace problem1_solution_set_problem2_range_of_a_l482_482088

section Problem1

def f1 (x : ℝ) : ℝ := |x - 4| + |x - 2|

theorem problem1_solution_set (a : ℝ) (h : a = 2) :
  { x : ℝ | f1 x > 10 } = { x : ℝ | x > 8 ∨ x < -2 } := sorry

end Problem1


section Problem2

def f2 (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem problem2_range_of_a (f_geq : ∀ x : ℝ, f2 x a ≥ 1) :
  a ≥ 5 ∨ a ≤ 3 := sorry

end Problem2

end problem1_solution_set_problem2_range_of_a_l482_482088


namespace tower_construction_l482_482346

-- Define the number of cubes the child has
def red_cubes : Nat := 3
def blue_cubes : Nat := 3
def green_cubes : Nat := 4

-- Define the total number of cubes
def total_cubes : Nat := red_cubes + blue_cubes + green_cubes

-- Define the height of the tower and the number of cubes left out
def tower_height : Nat := 8
def cubes_left_out : Nat := 2

-- Prove that the number of different towers that can be constructed is 980
theorem tower_construction : 
  (∑ k in {0,1}, (Nat.factorial tower_height) / 
    (Nat.factorial (red_cubes - k) * Nat.factorial (blue_cubes - k) * 
     Nat.factorial (green_cubes - 2*k))) +
  (∑ k in {0,1}, (Nat.factorial total_cubes) / 
    (Nat.factorial (red_cubes - k) * Nat.factorial (blue_cubes - k) * 
     Nat.factorial (green_cubes - 2*k) * Nat.factorial (cubes_left_out - k))) = 980 := 
by 
  sorry

end tower_construction_l482_482346


namespace lambda_range_l482_482552

def vector (α : Type*) := α × α

def dot_product {α : Type*} [has_mul α] [has_add α] (v1 v2 : vector α) : α :=
v1.1 * v2.1 + v1.2 * v2.2

def is_not_collinear {α : Type*} [decidable_eq α] [ring α] (v1 v2 : vector α) : Prop :=
¬ (v1.1 * v2.2 = v1.2 * v2.1)

theorem lambda_range (λ : ℝ) :
  let a := (1 : ℝ, 2 : ℝ)
      b := (1 : ℝ, 1 : ℝ)
  in dot_product a (a.1 + λ * b.1, a.2 + λ * b.2) > 0 
  ∧ is_not_collinear a (a.1 + λ * b.1, a.2 + λ * b.2) ↔ 
  λ > -5 / 3 ∧ λ ≠ 0 := 
by 
  let a : vector ℝ := (1, 2)
  let b : vector ℝ := (1, 1)
  sorry

end lambda_range_l482_482552


namespace cats_remaining_l482_482363

theorem cats_remaining 
  (siamese_cats : ℕ) 
  (house_cats : ℕ) 
  (cats_sold : ℕ) 
  (h1 : siamese_cats = 13) 
  (h2 : house_cats = 5) 
  (h3 : cats_sold = 10) : 
  siamese_cats + house_cats - cats_sold = 8 := 
by
  sorry

end cats_remaining_l482_482363


namespace stack_of_logs_total_l482_482371

-- Define the given conditions as variables and constants in Lean
def bottom_row : Nat := 15
def top_row : Nat := 4
def rows : Nat := bottom_row - top_row + 1
def sum_arithmetic_series (a l n : Nat) : Nat := n * (a + l) / 2

-- Define the main theorem to prove
theorem stack_of_logs_total : sum_arithmetic_series top_row bottom_row rows = 114 :=
by
  -- Here you will normally provide the proof
  sorry

end stack_of_logs_total_l482_482371


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l482_482485

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31 :
  let n := 7^7 - 7^4 in
  let prime_factors := {2, 3, 7, 19} in
  finset.sum prime_factors id = 31 :=
by
  sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l482_482485


namespace sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l482_482479

theorem sum_distinct_prime_factors_of_7_to_7_minus_7_to_4 : 
  let pfs := primeFactors (7 ^ 7 - 7 ^ 4)
  in (pfs = {2, 3, 19}) → sum pfs = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l482_482479


namespace negation_necessary_not_sufficient_l482_482925

theorem negation_necessary_not_sufficient (p q : Prop) : 
  ((¬ p) → ¬ (p ∨ q)) := 
sorry

end negation_necessary_not_sufficient_l482_482925


namespace dante_final_coconuts_l482_482655

theorem dante_final_coconuts
  (Paolo_coconuts : ℕ) (Dante_init_coconuts : ℝ)
  (Bianca_coconuts : ℕ) (Dante_final_coconuts : ℕ):
  Paolo_coconuts = 14 →
  Dante_init_coconuts = 1.5 * Real.sqrt Paolo_coconuts →
  Bianca_coconuts = 2 * (Paolo_coconuts + Int.floor Dante_init_coconuts) →
  Dante_final_coconuts = (Int.floor (Dante_init_coconuts) - (Int.floor (Dante_init_coconuts) / 3)) - 
    (25 * (Int.floor (Dante_init_coconuts) - (Int.floor (Dante_init_coconuts) / 3)) / 100) →
  Dante_final_coconuts = 3 :=
by
  sorry

end dante_final_coconuts_l482_482655


namespace angle_A_measure_l482_482589

variable {a b c A : ℝ}

def vector_m (b c a : ℝ) : ℝ × ℝ := (b, c - a)
def vector_n (b c a : ℝ) : ℝ × ℝ := (b - c, c + a)

theorem angle_A_measure (h_perpendicular : (vector_m b c a).1 * (vector_n b c a).1 + (vector_m b c a).2 * (vector_n b c a).2 = 0) :
  A = 2 * π / 3 := sorry

end angle_A_measure_l482_482589


namespace sqrt_sum_simplify_l482_482693

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l482_482693


namespace binomial_defective_products_l482_482835

-- Define the conditions
def total_products : ℕ := 100
def defective_products : ℕ := 5
def selection_count : ℕ := 10
def p_defective : ℝ := defective_products / total_products

-- Define the random variable X
def X : ProbDistrib ℝ := Distrib.binomial selection_count p_defective

-- State the theorem
theorem binomial_defective_products :
  X = Distrib.binomial selection_count p_defective :=
by sorry

end binomial_defective_products_l482_482835


namespace sum_f_eq_neg_one_third_l482_482770

def f (x : ℚ) : ℚ := (x^2 - 1) / (3 * x^2 + 3)

theorem sum_f_eq_neg_one_third : 
    (finset.univ.image (λ n, (2020 : ℚ) / n)).sum f + 
    (finset.range (2021 + 1)).sum f = ⟨-1, 3, by norm_num⟩ :=
by
  sorry

end sum_f_eq_neg_one_third_l482_482770


namespace product_of_odd_numbers_not_greater_than_9_with_greatest_difference_l482_482449

theorem product_of_odd_numbers_not_greater_than_9_with_greatest_difference :
  ∃ (a b : ℕ), (a % 2 = 1) ∧ (b % 2 = 1) ∧ (a ≤ 9) ∧ (b ≤ 9) ∧ |a - b| = 8 ∧ a * b = 9 :=
by
  sorry

end product_of_odd_numbers_not_greater_than_9_with_greatest_difference_l482_482449


namespace inequality_solution_set_l482_482137

theorem inequality_solution_set 
  (m n : ℤ)
  (h1 : ∀ x : ℤ, mx - n > 0 → x < 1 / 3)
  (h2 : ∀ x : ℤ, (m + n) x < n - m) :
  ∀ x : ℤ, x > -1 / 2 := 
sorry

end inequality_solution_set_l482_482137


namespace speed_of_train_l482_482326

-- Define the conditions
def length_of_train : ℕ := 240
def length_of_bridge : ℕ := 150
def time_to_cross : ℕ := 20

-- Compute the expected speed of the train
def expected_speed : ℝ := 19.5

-- The statement that needs to be proven
theorem speed_of_train : (length_of_train + length_of_bridge) / time_to_cross = expected_speed := by
  -- sorry is used to skip the actual proof
  sorry

end speed_of_train_l482_482326


namespace tan_phi_l482_482569

theorem tan_phi (φ : ℝ) (h1 : sin (π / 2 + φ) = sqrt 3 / 2) (h2 : 0 < φ ∧ φ < π) :
  tan φ = sqrt 3 / 3 := by
  sorry

end tan_phi_l482_482569


namespace sally_last_10_shots_l482_482830

-- Definitions of the conditions
def initial_shots : ℕ := 30
def initial_percentage : ℝ := 0.60
def additional_shots : ℕ := 10
def final_percentage : ℝ := 0.65

-- The mathematical proof problem translation
theorem sally_last_10_shots (initial_made: ℕ) (final_made: ℕ) :
  initial_made = (initial_percentage * initial_shots).to_nat →
  final_made = (final_percentage * (initial_shots + additional_shots)).to_nat →
  final_made - initial_made = 8 :=
begin
  sorry
end

end sally_last_10_shots_l482_482830


namespace range_of_x_l482_482274

theorem range_of_x (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x - 3 ≠ 0) ↔ (x ≥ -1) ∧ (x ≠ 3) :=
by sorry

end range_of_x_l482_482274


namespace candy_total_l482_482174

theorem candy_total (x : ℕ) : 216 + 137 + x = 353 + x :=
by {
  rw add_assoc,
  rw add_comm 216 137,
  norm_num,
  rw add_comm,
}

end candy_total_l482_482174


namespace dave_books_about_outer_space_l482_482425

theorem dave_books_about_outer_space (x : ℕ) 
  (H1 : 8 + 3 = 11) 
  (H2 : 11 * 6 = 66) 
  (H3 : 102 - 66 = 36) 
  (H4 : 36 / 6 = x) : 
  x = 6 := 
by
  sorry

end dave_books_about_outer_space_l482_482425


namespace exists_line_through_A_intersecting_segment_l482_482630

theorem exists_line_through_A_intersecting_segment (d : ℝ) (h : 0 ≤ d ∧ d ≤ 1) :
  ∃ x : ℝ, x = (-d + Real.sqrt(d^2 + 8)) / 2 :=
by
  sorry

end exists_line_through_A_intersecting_segment_l482_482630


namespace number_of_male_alligators_l482_482197

-- Define the Conditions
def is_population_evenly_divided (total : ℕ) (males : ℕ) (females : ℕ) : Prop :=
  males = females ∧ total = males + females

def is_females_composition (total_females : ℕ) (adult_females : ℕ) (juvenile_ratio adult_ratio : ℝ) : Prop :=
  adult_ratio = 0.60 ∧ juvenile_ratio = 0.40 ∧ adult_females = (total_females * adult_ratio).to_nat

def given_adult_females (adult_females : ℕ) : Prop :=
  adult_females = 15

-- Translate to a mathematically equivalent proof problem
theorem number_of_male_alligators (total_alligators males females total_females : ℕ) 
  (juvenile_ratio adult_ratio : ℝ) :
  is_population_evenly_divided total_alligators males females →
  is_females_composition total_females 15 juvenile_ratio adult_ratio →
  given_adult_females 15 →
  males = 25 :=
by 
  intros h_population h_females h_adults
  sorry

end number_of_male_alligators_l482_482197


namespace cara_younger_than_mom_l482_482413

noncomputable def cara_grandmothers_age : ℤ := 75
noncomputable def cara_moms_age := cara_grandmothers_age - 15
noncomputable def cara_age : ℤ := 40

theorem cara_younger_than_mom :
  cara_moms_age - cara_age = 20 := by
  sorry

end cara_younger_than_mom_l482_482413


namespace solution_set_l482_482914

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set (f : ℝ → ℝ) (h1 : ∀ x, f(x) - f'(x) > 0) (h2 : f(2) = 2) : 
  (∀ x, f(x) > 2 * exp(x - 2) ↔ x < 2) := 
by
  sorry

end solution_set_l482_482914


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482467

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let expr := 7 ^ 7 - 7 ^ 4 in
  (7^4 * (7^3 - 1) = expr) ∧ (7^3 - 1 = 342) ∧ (Prime 2) ∧ (Prime 3) ∧ (Prime 7) ∧ (Prime 19) ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ 342 → p = 2 ∨ p = 3 ∨ p = 19) → 
  (∀ p : ℕ, Nat.Prime p → p ∣ expr → p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19) → 
  (2 + 3 + 7 + 19 = 31) := 
by
  intro expr fact1 fact2 prime2 prime3 prime7 prime19 factors342 factorsExpr
  sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482467


namespace angle_ratio_l482_482612

-- Define the angles and their properties
variables (A B C P Q M : Type)
variables (mABQ mMBQ mPBQ : ℝ)

-- Define the conditions from the problem:
-- 1. BP and BQ bisect ∠ABC
-- 2. BM bisects ∠PBQ
def conditions (h1 : 2 * mPBQ = mABQ)
               (h2 : 2 * mMBQ = mPBQ) : Prop :=
  true

-- Translate the question and correct answer into a Lean definition.
def find_ratio (h1 : 2 * mPBQ = mABQ) 
               (h2 : 2 * mMBQ = mPBQ) : Prop :=
  mMBQ / mABQ = 1 / 4

-- Now define the theorem that encapsulates the problem statement
theorem angle_ratio (h1 : 2 * mPBQ = mABQ) 
                    (h2 : 2 * mMBQ = mPBQ) :
  find_ratio A B C P Q M mABQ mMBQ mPBQ h1 h2 :=
by
  -- Proof to be provided.
  sorry

end angle_ratio_l482_482612


namespace positive_integers_satisfying_inequality_l482_482558

def satisfies_inequality (n : ℕ) : Prop :=
  (5 * n + 10) * (2 * n - 6) * (n - 20) < 0

theorem positive_integers_satisfying_inequality :
  {n : ℕ // satisfies_inequality n}.card = 16 := 
sorry

end positive_integers_satisfying_inequality_l482_482558


namespace find_stream_speed_l482_482279

/-- Variables to represent speeds in kmph. -/
variables {s_stream : ℝ} {s_still : ℝ}

/-- Conditions given in the problem. -/
def boat_conditions (s_stream s_still : ℝ) : Prop :=
  ∀ d : ℝ, d > 0 → (d / (s_still - s_stream)) = 2 * (d / (s_still + s_stream))

/-- The speed of the boat in still water is 36 kmph. -/
def boat_still_water_speed : ℝ := 36

theorem find_stream_speed : boat_conditions s_stream boat_still_water_speed → s_stream = 12 :=
sorry

end find_stream_speed_l482_482279


namespace minimum_value_of_abs_z_l482_482641

open Complex

theorem minimum_value_of_abs_z (z : ℂ) (H : |z + 2 - (3 : ℂ)*I| + |z - (2 : ℂ)*I| = 7) : |z| = 0 := 
  sorry

end minimum_value_of_abs_z_l482_482641


namespace bob_needs_10_fills_l482_482847

theorem bob_needs_10_fills (flour_needed : ℚ) (cup_capacity : ℚ)
  (h1 : flour_needed = 19 / 4) (h2 : cup_capacity = 1 / 2) :
  (flour_needed / cup_capacity).ceil = 10 :=
by
  have h3 : flour_needed / cup_capacity = 9.5 := by sorry
  exact h3.ceil_eq 10

end bob_needs_10_fills_l482_482847


namespace solve_for_y_l482_482714

theorem solve_for_y (y : ℤ) (h : 7 - y = 10) : y = -3 :=
sorry

end solve_for_y_l482_482714


namespace min_cubes_needed_proof_l482_482760

noncomputable def min_cubes_needed_to_form_30_digit_number : ℕ :=
  sorry

theorem min_cubes_needed_proof : min_cubes_needed_to_form_30_digit_number = 50 :=
  sorry

end min_cubes_needed_proof_l482_482760


namespace TomTotalWeight_l482_482753

def TomWeight : ℝ := 150
def HandWeight (personWeight: ℝ) : ℝ := 1.5 * personWeight
def VestWeight (personWeight: ℝ) : ℝ := 0.5 * personWeight
def TotalHandWeight (handWeight: ℝ) : ℝ := 2 * handWeight
def TotalWeight (totalHandWeight vestWeight: ℝ) : ℝ := totalHandWeight + vestWeight

theorem TomTotalWeight : TotalWeight (TotalHandWeight (HandWeight TomWeight)) (VestWeight TomWeight) = 525 := 
by
  sorry

end TomTotalWeight_l482_482753


namespace complementary_event_of_hitting_at_least_once_is_missing_both_l482_482818

-- Definition: A soldier fires two consecutive shots
def fires_two_consecutive_shots := true

-- Question: What is the complementary event of "hitting the target at least once" given the condition?
def hitting_at_least_once := true

-- Correct Answer: The complementary event of "hitting the target at least once"
-- is "missing both attempts"
def missing_both_attempts (event : Prop) : Prop := event = (true = false)

theorem complementary_event_of_hitting_at_least_once_is_missing_both:
  fires_two_consecutive_shots → (hitting_at_least_once = missing_both_attempts false) :=
by
  intro h
  simp [hitting_at_least_once, missing_both_attempts]
  sorry

end complementary_event_of_hitting_at_least_once_is_missing_both_l482_482818


namespace compute_100a_plus_b_l482_482639

theorem compute_100a_plus_b :
  ∀ (q r : ℕ) (h_r : 0 ≤ r ∧ r < 11),
  let k := 11 * q + r,
      gcd_q := Nat.gcd q (11 * q + r),
      a := q / gcd_q,
      b := (11 * q + r) / gcd_q in
  Nat.coprime a b ∧ 100 * a + b = 1000 :=
by
  intros q r h_r,
  let k := 11 * q + r,
  let gcd_q := Nat.gcd q (11 * q + r),
  let a := q / gcd_q,
  let b := (11 * q + r) / gcd_q,
  
  sorry

end compute_100a_plus_b_l482_482639


namespace smallest_number_l482_482768

theorem smallest_number (x : ℕ) : (∃ y : ℕ, y = x - 16 ∧ (y % 4 = 0) ∧ (y % 6 = 0) ∧ (y % 8 = 0) ∧ (y % 10 = 0)) → x = 136 := by
  sorry

end smallest_number_l482_482768


namespace molecular_weight_X_l482_482350

theorem molecular_weight_X (Ba_weight : ℝ) (total_molecular_weight : ℝ) (X_weight : ℝ) 
  (h1 : Ba_weight = 137) 
  (h2 : total_molecular_weight = 171) 
  (h3 : total_molecular_weight - Ba_weight * 1 = 2 * X_weight) : 
  X_weight = 17 :=
by
  sorry

end molecular_weight_X_l482_482350


namespace ellipse_equation_l482_482076

theorem ellipse_equation (b : ℝ) (c : ℝ) :
  0 < b ∧ b < 1 →
  c^2 = 1 - b^2 →
  x, y ∈ ℝ →
  ellipse_eq : ∀ {x y : ℝ}, x^2 + (y^2 / b^2) = 1 →
  |af1| = 2 * |bf1| →
  af2_perpendicular_x : af2 ⟂ x_axis →
  (x^2 + (5 * y^2) / 4 = 1) :=
by {
  sorry
}

end ellipse_equation_l482_482076


namespace sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l482_482478

theorem sum_distinct_prime_factors_of_7_to_7_minus_7_to_4 : 
  let pfs := primeFactors (7 ^ 7 - 7 ^ 4)
  in (pfs = {2, 3, 19}) → sum pfs = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l482_482478


namespace lucy_total_cookies_l482_482644

/-- Lucy's total money in dollars --/
def lucy_money : ℝ := 20.75

/-- Price per pack of cookies in dollars --/
def pack_price : ℝ := 1.75

/-- Number of cookies per pack --/
def cookies_per_pack : ℕ := 2

/-- Calculate the maximum number of packs Lucy can buy --/
def max_packs : ℕ := (lucy_money / pack_price).floor

/-- Calculate the total number of cookies Lucy can buy --/
def total_cookies : ℕ := max_packs * cookies_per_pack

/-- The main theorem stating the total number of cookies Lucy can buy --/
theorem lucy_total_cookies : total_cookies = 22 := by
  sorry

end lucy_total_cookies_l482_482644


namespace option_B_option_C_option_D_l482_482317

variable {A B C : ℝ}
variable {AB AC : ℝ}
variable {B_deg : ℝ}
variable (triangleABC : Type) [triangleABC ∈ Triangle ABC]

-- Option B: A > B <-> sin A > sin B
theorem option_B (H : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2) : 
  (A > B ↔ Real.sin A > Real.sin B) := sorry

-- Option C: Shifting the graph of y = sin 2x to the left by π/4 units results in y = cos 2x
theorem option_C : 
  (∀ x, Real.sin (2 * (x + π/4)) = Real.cos (2 * x)) := sorry

-- Option D: Given AB = √3, AC = 1, B = 30°, the area of triangle ABC is √3/4 or √3/2
theorem option_D (H1 : AB = Real.sqrt 3) (H2 : AC = 1) (H3 : B_deg = 30) :
  let B := Real.pi / 6 in
  let s₁ := 1 / 2 * AB * AC * Real.sin (Real.pi / 2) in
  let s₂ := 1 / 2 * AB * AC * Real.sin (Real.pi / 6) in
  (s₁ = Real.sqrt 3 / 2 ∨ s₂ = Real.sqrt 3 / 4) := sorry

end option_B_option_C_option_D_l482_482317


namespace number_of_logs_in_stack_l482_482369

theorem number_of_logs_in_stack :
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  total_logs = 114 := by
{
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  sorry
}

end number_of_logs_in_stack_l482_482369


namespace simplify_radicals_l482_482704

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l482_482704


namespace vector_dot_product_zero_l482_482239

-- Given conditions
variables {A C B D : Type} [MetricSpace A] [MetricSpace C] [MetricSpace B] [MetricSpace D]
variable {t : ℝ}
variable (circle : Circle A C)
variable (AB_len : dist A B = sqrt (t + 1))
variable (AD_len : dist A D = sqrt (t + 2))
variable (B_on_circle : B ∈ circle)
variable (D_on_circle : D ∈ circle)

-- The final statement to prove
theorem vector_dot_product_zero :
  ∀ A C B D (circle : Circle A C), 
  dist A B = sqrt (t + 1) → 
  dist A D = sqrt (t + 2) → 
  B ∈ circle → 
  D ∈ circle →
  dist A C * dist B D * cos (angle A C B D) = 0 :=
sorry

end vector_dot_product_zero_l482_482239


namespace wife_catch_up_l482_482359

/-- A man drives at a speed of 40 miles/hr.
His wife left 30 minutes late with a speed of 50 miles/hr.
Prove that they will meet 2 hours after the wife starts driving. -/
theorem wife_catch_up (t : ℝ) (speed_man speed_wife : ℝ) (late_time : ℝ) :
  speed_man = 40 →
  speed_wife = 50 →
  late_time = 0.5 →
  50 * t = 40 * (t + 0.5) →
  t = 2 :=
by
  intros h_man h_wife h_late h_eq
  -- Actual proof goes here. 
  -- (Skipping the proof as requested, leaving it as a placeholder)
  sorry

end wife_catch_up_l482_482359


namespace a_eq_zero_of_sine_equality_l482_482498

theorem a_eq_zero_of_sine_equality (a x : ℝ) 
    (hx_nonneg : x ≥ 0) 
    (ha_nonneg : a ≥ 0) 
    (h_sine_eq : ∀ x : ℝ, x ≥ 0 → sin (sqrt (x + a)) = sin (sqrt x)) : 
    a = 0 :=
begin
  sorry
end

end a_eq_zero_of_sine_equality_l482_482498


namespace collinear_MFC_l482_482250

theorem collinear_MFC 
  {A D E F C M : Point}  -- define the points involved
  (h1 : isosceles_triangle_congruence A D E F C M)
  (h2 : EF = AD)  -- EF equals AD
  (h3 : ED = AC)  -- ED equals AC
  (h4 : ∆ FDC is_isosceles) :  -- triangle FDC is isosceles
  are_collinear M F C :=  -- conclusion to prove
sorry

end collinear_MFC_l482_482250


namespace total_wire_length_l482_482345

theorem total_wire_length (S : ℕ) (L : ℕ)
  (hS : S = 20) 
  (hL : L = 2 * S) : S + L = 60 :=
by
  sorry

end total_wire_length_l482_482345


namespace probability_point_between_l_and_m_l482_482143

def l (x : ℝ) : ℝ := -2 * x + 8
def m (x : ℝ) : ℝ := -3 * x + 9

def area_under_l : ℝ := 0.5 * 4 * 8
def area_under_m : ℝ := 0.5 * 3 * 9

theorem probability_point_between_l_and_m : 
  (area_under_l - area_under_m) / area_under_l = 0.16 :=
by
  -- Variables to store areas for clarity
  have area_l : ℝ := 0.5 * 4 * 8
  have area_m : ℝ := 0.5 * 3 * 9

  -- Probability calculation
  calc (area_l - area_m) / area_l = 2.5 / 16 : by sorry
  ... = 0.15625 : by sorry
  ... ≈ 0.16 : by sorry

end probability_point_between_l_and_m_l482_482143


namespace angles_cosine_condition_l482_482170

theorem angles_cosine_condition {A B : ℝ} (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) :
  (A > B) ↔ (Real.cos A < Real.cos B) :=
by
sorry

end angles_cosine_condition_l482_482170


namespace sqrt_72_plus_sqrt_32_l482_482705

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l482_482705


namespace area_triangle_ABC_l482_482148

-- Definitions corresponding to the conditions in the problem.
variables {A B C D E F : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]

-- Hypotheses about the triangles and their properties
axiom triangle_DEF_isosceles_right_triangle : is_isosceles_right_triangle D E F
axiom triangle_ABC_isosceles_right_triangle : is_isosceles_right_triangle A B C
axiom hypotenuses_aligned : aligned_hypotenuses (hypotenuse D E F) (hypotenuse A B C)
axiom point_E_interior_abc : lies_within_triangle E A B C
axiom point_D_interior_abc : lies_within_triangle D A B C
axiom point_F_interior_abc : lies_within_triangle F A B C

-- Given area of triangle DEF
axiom area_DEF : area (triangle D E F) = 1

-- Theorem to prove the area of triangle ABC
theorem area_triangle_ABC : area (triangle A B C) = 36 := 
sorry

end area_triangle_ABC_l482_482148


namespace sum_eq_constant_x_l482_482563

theorem sum_eq_constant_x :
  (∑ n in finset.range 1991 + 1, n * (1992 - n)) = 1991 * 996 * 664 :=
by
  sorry

end sum_eq_constant_x_l482_482563


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l482_482469

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4 :
  let n := 7^7 - 7^4 in 
  (∑ p in (nat.factors n).to_finset, p) = 31 :=
by sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l482_482469


namespace anya_probability_l482_482400

open Finset

def possible_coins := {10, 10, 5, 5, 2}
def target := 19

noncomputable def combinations := (possible_coins.vals.ctype_power 3).val.filter (λ s, Finset.sum s >= target)

noncomputable def probability : ℝ :=
  (combinations.card : ℝ) / (possible_coins.vals.ctype_power 3).card

theorem anya_probability : probability = 0.4 := sorry

end anya_probability_l482_482400


namespace flowers_bloom_l482_482284

theorem flowers_bloom (num_unicorns : ℕ) (flowers_per_step : ℕ) (distance_km : ℕ) (step_length_m : ℕ) 
  (h1 : num_unicorns = 6) (h2 : flowers_per_step = 4) (h3 : distance_km = 9) (h4 : step_length_m = 3) : 
  num_unicorns * (distance_km * 1000 / step_length_m) * flowers_per_step = 72000 :=
by
  sorry

end flowers_bloom_l482_482284


namespace dark_squares_exceed_light_l482_482813

-- Define the problem conditions.
def grid_rows : ℕ := 7
def grid_cols : ℕ := 8

def start_with_dark (r : ℕ) (c : ℕ) : bool := (c % 2 = 0)

def dark_count_per_row (cols: ℕ) : ℕ :=
  (cols + 1) / 2

def light_count_per_row (cols: ℕ) : ℕ :=
  cols / 2

def total_dark_squares (rows cols : ℕ) : ℕ :=
  rows * dark_count_per_row cols

def total_light_squares (rows cols : ℕ) : ℕ :=
  rows * light_count_per_row cols

def excess_dark_squares (rows cols : ℕ) : ℕ :=
  total_dark_squares rows cols - total_light_squares rows cols

-- State the proof problem.
theorem dark_squares_exceed_light :
  excess_dark_squares grid_rows grid_cols = 14 :=
by
  sorry

end dark_squares_exceed_light_l482_482813


namespace john_initial_diamonds_l482_482790

theorem john_initial_diamonds :
  let bill_initial := 12
  let sam_initial := 12
  let john_initial := x
  let bill_final := bill_initial
  let sam_final := sam_initial
  let john_final := john_initial
  (average_decrease bill_initial 1) ∧
  (average_decrease sam_initial 2) ∧
  (average_increase john_initial 4) →
  x = 9 :=
begin
  sorry
end

end john_initial_diamonds_l482_482790


namespace lattice_points_in_5_sphere_l482_482555

theorem lattice_points_in_5_sphere :
  (Finset.card {p : ℤ × ℤ × ℤ × ℤ × ℤ | p.1^2 + p.2.1^2 + p.2.2.1^2 + p.2.2.2.1^2 + p.2.2.2.2^2 ≤ 9}) = 1343 := sorry

end lattice_points_in_5_sphere_l482_482555


namespace possible_values_of_a_l482_482952

noncomputable def f (x : ℝ) : ℝ := abs (sin x ^ 2 - (sqrt 3) * cos x * cos (3 * pi / 2 - x) - 1 / 2)

theorem possible_values_of_a
  (a : ℝ)
  (h : 0 < a ∧ a < pi)
  (symmetry : ∀ x, f (x + a) = f (-x)) :
  a = pi / 12 ∨ a = pi / 3 ∨ a = 7 * pi / 12 ∨ a = 5 * pi / 6 :=
sorry

end possible_values_of_a_l482_482952


namespace proof_problem_l482_482774

def statement_1 : Prop := ∀ (l1 l2 : Line) (t : Transversal), 
  CorrespondingAngles l1 l2 t → AnglesEq (CorrespondingAngle l1 t) (CorrespondingAngle l2 t)

def statement_2 : Prop := ∀ (P : Point) (l : Line), 
  PerpendicularSegment P l → ShortestSegment P l

def statement_3 : Prop := ∀ (P : Point) (l : Line), 
  P ∉ l → ∃! (l' : Line), Parallel l l' ∧ P ∈ l'

def statement_4 : Prop := ∀ r : ℝ, 
  RationalNumber r → CorrespondsOneToOne r (PointOnNumberLine r)

def statement_5 : Prop := IntegerPart (sqrt 63 - 1) = 7

theorem proof_problem : ∃ n : ℕ, n = 2 ∧
  ([statement_2, statement_3].count (λ P, P = True) = n) := by
  sorry

end proof_problem_l482_482774


namespace number_of_scenarios_l482_482846

theorem number_of_scenarios :
  ∃ (count : ℕ), count = 42244 ∧
  (∃ (x1 x2 x3 x4 x5 x6 x7 : ℕ),
    x1 % 7 = 0 ∧ x2 % 7 = 0 ∧ x3 % 7 = 0 ∧ x4 % 7 = 0 ∧
    x5 % 13 = 0 ∧ x6 % 13 = 0 ∧ x7 % 13 = 0 ∧
    x1 + x2 + x3 + x4 + x5 + x6 + x7 = 270) :=
sorry

end number_of_scenarios_l482_482846


namespace calculate_savings_l482_482226

theorem calculate_savings :
  let plane_cost : ℕ := 600
  let boat_cost : ℕ := 254
  plane_cost - boat_cost = 346 := by
    let plane_cost : ℕ := 600
    let boat_cost : ℕ := 254
    sorry

end calculate_savings_l482_482226


namespace parallel_projection_of_equilateral_triangle_l482_482421

-- Define the equilateral triangle with side length 4 cm
structure Triangle :=
  (A B C : Point)
  (sides : dist A B = dist B C ∧ dist B C = dist C A)
  (side_length : dist A B = 4)

-- Define the parallel projection preserving the triangle's properties
def parallel_projection (T : Triangle) : Triangle :=
  ⟨T.A, T.B, T.C, T.sides, T.side_length⟩

-- Prove the parallel projection is the same equilateral triangle
theorem parallel_projection_of_equilateral_triangle :
  ∀ T : Triangle, T.side_length = 4 → T.sides →
  parallel_projection T = T :=
by 
  intro T hlen hsides
  sorry


end parallel_projection_of_equilateral_triangle_l482_482421


namespace merchant_profit_l482_482128

variable {C S : ℚ}

-- Given condition
def cost_eq_sell : Prop := 18 * C = 16 * S

-- The theorem to prove
theorem merchant_profit (h : cost_eq_sell) : (S - C) / C * 100 = 12.5 := 
by
  sorry

end merchant_profit_l482_482128


namespace sqrt_fraction_difference_l482_482003

theorem sqrt_fraction_difference : 
  (Real.sqrt (16 / 9) - Real.sqrt (9 / 16)) = 7 / 12 :=
by
  sorry

end sqrt_fraction_difference_l482_482003


namespace power_of_exponents_l482_482565

theorem power_of_exponents (m n : ℝ) (h1 : 2^m = 3) (h2 : 4^n = 8) : 2^(3 * m - 2 * n + 3) = 27 :=
by
  sorry

end power_of_exponents_l482_482565


namespace sqrt_sum_simplify_l482_482671

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l482_482671


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482465

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let expr := 7 ^ 7 - 7 ^ 4 in
  (7^4 * (7^3 - 1) = expr) ∧ (7^3 - 1 = 342) ∧ (Prime 2) ∧ (Prime 3) ∧ (Prime 7) ∧ (Prime 19) ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ 342 → p = 2 ∨ p = 3 ∨ p = 19) → 
  (∀ p : ℕ, Nat.Prime p → p ∣ expr → p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19) → 
  (2 + 3 + 7 + 19 = 31) := 
by
  intro expr fact1 fact2 prime2 prime3 prime7 prime19 factors342 factorsExpr
  sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482465


namespace sqrt_sum_simplify_l482_482677

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l482_482677


namespace inscribed_circle_radius_l482_482305

theorem inscribed_circle_radius (A B C : ℝ) (hAB : AB = 6) (hAC : AC = 8) (hBC : BC = 10) : r = 2 :=
by
  -- definitions and conditions 
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt(s * (s - AB) * (s - AC) * (s - BC))
  have h_s : s = 12 
    by sorry -- calculated the semiperimeter
  have h_K : K = 24 
    by sorry -- calculated the area using Heron's formula
  have relation_area_radius : K = r * s 
    by sorry -- relation between area and radius of inscribed circle
  show r = 2
    from sorry -- final step to solve for r

end inscribed_circle_radius_l482_482305


namespace faster_car_distance_l482_482292

theorem faster_car_distance (d v : ℝ) (h_dist: d + 2 * d = 4) (h_faster: 2 * v = 2 * (d / v)) : 
  d = 4 / 3 → 2 * d = 8 / 3 :=
by sorry

end faster_car_distance_l482_482292


namespace total_matches_l482_482814

noncomputable def matches_in_tournament (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem total_matches :
  matches_in_tournament 5 + matches_in_tournament 7 + matches_in_tournament 4 = 37 := 
by 
  sorry

end total_matches_l482_482814


namespace angle_ratio_l482_482609

theorem angle_ratio (A B C P Q M : Type) (θ : ℝ)
  (h1 : ∠B P B = ∠A B C / 2)
  (h2 : ∠B Q B = ∠A B C / 2)
  (h3 : ∠M B P = ∠M B Q)
  (h4 : ∠A B C = 2 * θ) :
  ∠M B Q / ∠A B Q = 1 / 3 := by
  sorry

end angle_ratio_l482_482609


namespace not_in_set_l482_482926

def A : Set ℕ := {0, 2}

theorem not_in_set (h : A = {0, 2}) : {2} ∉ A :=
by
  sorry

end not_in_set_l482_482926


namespace part_one_part_two_l482_482097

noncomputable def f (x a : ℝ) : ℝ :=
  |x + a| + 2 * |x - 1|

theorem part_one (a : ℝ) (h : a = 1) : 
  ∃ x : ℝ, f x 1 = 2 :=
sorry

theorem part_two (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : ∀ x : ℝ, 1 ≤ x → x ≤ 2 → f x a > x^2 - b + 1) : 
  (a + 1 / 2)^2 + (b + 1 / 2)^2 > 2 :=
sorry

end part_one_part_two_l482_482097


namespace parallelogram_area_example_l482_482302

noncomputable def area_parallelogram (A B C D : (ℝ × ℝ)) : ℝ := 
  0.5 * |(A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * D.1 + D.2 * A.1)|

theorem parallelogram_area_example : 
  let A := (0, 0)
  let B := (20, 0)
  let C := (25, 7)
  let D := (5, 7)
  area_parallelogram A B C D = 140 := 
by
  sorry

end parallelogram_area_example_l482_482302


namespace students_without_scholarships_l482_482235

theorem students_without_scholarships :
  let total_students := 300
  let full_merit_percent := 0.05
  let half_merit_percent := 0.10
  let sports_percent := 0.03
  let need_based_percent := 0.07
  let full_merit_and_sports_percent := 0.01
  let half_merit_and_need_based_percent := 0.02
  let full_merit := full_merit_percent * total_students
  let half_merit := half_merit_percent * total_students
  let sports := sports_percent * total_students
  let need_based := need_based_percent * total_students
  let full_merit_and_sports := full_merit_and_sports_percent * total_students
  let half_merit_and_need_based := half_merit_and_need_based_percent * total_students
  let total_with_scholarships := (full_merit + half_merit + sports + need_based) - (full_merit_and_sports + half_merit_and_need_based)
  let students_without_scholarships := total_students - total_with_scholarships
  students_without_scholarships = 234 := 
by
  sorry

end students_without_scholarships_l482_482235


namespace two_x_plus_two_y_value_l482_482911

theorem two_x_plus_two_y_value (x y : ℝ) (h1 : x^2 - y^2 = 8) (h2 : x - y = 6) : 2 * x + 2 * y = 8 / 3 := 
by sorry

end two_x_plus_two_y_value_l482_482911


namespace total_profit_for_year_l482_482663

structure HorseshoeSet (initial_outlay per_set_cost high_price low_price : ℕ)

def typeA : HorseshoeSet := {
  initial_outlay := 10000,
  per_set_cost := 20,
  high_price := 60,
  low_price := 50
}

def typeB : HorseshoeSet := {
  initial_outlay := 6000,
  per_set_cost := 15,
  high_price := 40,  -- consistent price, use the same field
  low_price := 40    -- consistent price, use the same field
}

def revenue (sets_high sets_low : ℕ) (hs : HorseshoeSet) : ℕ :=
  (sets_high * hs.high_price) + (sets_low * hs.low_price)

def cost (total_sets : ℕ) (hs : HorseshoeSet) : ℕ :=
  hs.initial_outlay + (total_sets * hs.per_set_cost)

def profit (sets_high sets_low : ℕ) (hs : HorseshoeSet) : ℕ :=
  revenue sets_high sets_low hs - cost (sets_high + sets_low) hs

theorem total_profit_for_year :
  let profitA := profit 300 200 typeA in
  let profitB := profit 0 800 typeB in
  profitA + profitB = 22000 :=
by
  sorry

end total_profit_for_year_l482_482663


namespace probability_correct_l482_482141

noncomputable def probability_point_between_lines : ℝ :=
  let intersection_x_l := 4    -- x-intercept of line l
  let intersection_x_m := 3    -- x-intercept of line m
  let area_under_l := (1 / 2) * intersection_x_l * 8 -- area under line l
  let area_under_m := (1 / 2) * intersection_x_m * 9 -- area under line m
  let area_between := area_under_l - area_under_m    -- area between lines
  (area_between / area_under_l : ℝ)

theorem probability_correct : probability_point_between_lines = 0.16 :=
by
  simp only [probability_point_between_lines]
  sorry

end probability_correct_l482_482141


namespace geometric_sum_S15_l482_482532

noncomputable def S (n : ℕ) : ℝ := sorry  -- Assume S is defined for the sequence sum

theorem geometric_sum_S15 (S_5 S_10 : ℝ) (h1 : S_5 = 5) (h2 : S_10 = 30) : 
    S 15 = 155 := 
by 
  -- Placeholder for geometric sequence proof
  sorry

end geometric_sum_S15_l482_482532


namespace find_remainder_of_n_l482_482511

theorem find_remainder_of_n (n k d : ℕ) (hn_pos : n > 0) (hk_pos : k > 0) (hd_pos_digits : d < 10^k) 
  (h : n * 10^k + d = n * (n + 1) / 2) : n % 9 = 1 :=
sorry

end find_remainder_of_n_l482_482511


namespace cow_cost_calculation_l482_482358

theorem cow_cost_calculation (C cow calf : ℝ) 
  (h1 : cow = 8 * calf) 
  (h2 : cow + calf = 990) : 
  cow = 880 :=
by
  sorry

end cow_cost_calculation_l482_482358


namespace simplify_sum_of_square_roots_l482_482686

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l482_482686


namespace probability_both_segments_successful_expected_number_of_successful_segments_probability_given_3_successful_l482_482489

-- Definitions and conditions from the problem
def success_probability_each_segment : ℚ := 3 / 4
def num_segments : ℕ := 4

-- Correct answers from the solution
def prob_both_success : ℚ := 9 / 16
def expected_successful_segments : ℚ := 3
def cond_prob_given_3_successful : ℚ := 3 / 4

theorem probability_both_segments_successful :
  (success_probability_each_segment * success_probability_each_segment) = prob_both_success :=
by
  sorry

theorem expected_number_of_successful_segments :
  (num_segments * success_probability_each_segment) = expected_successful_segments :=
by
  sorry

theorem probability_given_3_successful :
  let prob_M := 4 * (success_probability_each_segment^3 * (1 - success_probability_each_segment))
  let prob_NM := 3 * (success_probability_each_segment^3 * (1 - success_probability_each_segment))
  (prob_NM / prob_M) = cond_prob_given_3_successful :=
by
  sorry

end probability_both_segments_successful_expected_number_of_successful_segments_probability_given_3_successful_l482_482489


namespace arithmetic_seq_S11_l482_482568

variable {a1 d : ℤ}
def S (n : ℤ) := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_seq_S11 :
  S 8 - S 3 = 20 →
  S 11 = 44 := by
  intros hS
  sorry

end arithmetic_seq_S11_l482_482568


namespace relationship_of_values_l482_482091

def f (x : ℝ) : ℝ := x * Real.sin x 

theorem relationship_of_values :
  f (-π / 4) < f 1 ∧ f 1 < f (π / 3) :=
by
  sorry

end relationship_of_values_l482_482091


namespace total_savings_correct_l482_482190

-- Definitions of savings per day and days saved for Josiah, Leah, and Megan
def josiah_saving_per_day : ℝ := 0.25
def josiah_days : ℕ := 24

def leah_saving_per_day : ℝ := 0.50
def leah_days : ℕ := 20

def megan_saving_per_day : ℝ := 1.00
def megan_days : ℕ := 12

-- Definition to calculate total savings for each child
def total_saving (saving_per_day : ℝ) (days : ℕ) : ℝ :=
  saving_per_day * days

-- Total amount saved by Josiah, Leah, and Megan
def total_savings : ℝ :=
  total_saving josiah_saving_per_day josiah_days +
  total_saving leah_saving_per_day leah_days +
  total_saving megan_saving_per_day megan_days

-- Theorem to prove the total savings is $28
theorem total_savings_correct : total_savings = 28 := by
  sorry

end total_savings_correct_l482_482190


namespace part1_part2_l482_482538

noncomputable def f (x k : ℝ) : ℝ := (x - 1) * Real.exp x - k * x^2 + 2

theorem part1 {x : ℝ} (hx : x = 0) : 
    f x 0 = 1 :=
by
  sorry

theorem part2 {x k : ℝ} (hx : 0 ≤ x) (hxf : f x k ≥ 1) : 
    k ≤ 1 / 2 :=
by
  sorry

end part1_part2_l482_482538


namespace sum_17_20_l482_482616

-- Definitions for the conditions given in the problem
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ) 
variable (r : ℝ)

-- The sequence is geometric and sums are defined accordingly
axiom geo_seq : ∀ n, a (n + 1) = r * a n
axiom sum_def : ∀ n, S n = ∑ i in finset.range n, a (i + 1)

-- Given conditions
axiom S4 : S 4 = 1
axiom S8 : S 8 = 3

-- The value we need to prove
theorem sum_17_20 : a 17 + a 18 + a 19 + a 20 = 16 :=
sorry

end sum_17_20_l482_482616


namespace max_value_of_sin_l482_482267

theorem max_value_of_sin (x : ℝ) : 
  set_exists! (λ y, (∃ x, y = 3 * real.sin (2 * x)) ∧ 
    (∀ x, y ≤ 3 )) := 
by {
  use 3,
  split,
  { use 0, 
    rw real.sin_zero, 
    norm_num },
  { intro x,
    have sin_bound := real.sin_le_abs (2 * x),
    simp at sin_bound,
    linarith }
}

end max_value_of_sin_l482_482267


namespace range_of_b_a_l482_482099

theorem range_of_b_a (a b : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) :
  (∀ x : ℝ, 0 < x → (x^2 + b * x - 4) * Real.log x a ≤ 0) → 1 < b^a ∧ b^a < 3 :=
begin
  sorry
end

end range_of_b_a_l482_482099


namespace probability_correct_l482_482140

noncomputable def probability_point_between_lines : ℝ :=
  let intersection_x_l := 4    -- x-intercept of line l
  let intersection_x_m := 3    -- x-intercept of line m
  let area_under_l := (1 / 2) * intersection_x_l * 8 -- area under line l
  let area_under_m := (1 / 2) * intersection_x_m * 9 -- area under line m
  let area_between := area_under_l - area_under_m    -- area between lines
  (area_between / area_under_l : ℝ)

theorem probability_correct : probability_point_between_lines = 0.16 :=
by
  simp only [probability_point_between_lines]
  sorry

end probability_correct_l482_482140


namespace num_factors_more_than_three_l482_482430

theorem num_factors_more_than_three (n : ℕ) (h : n = 2550) : 
  (∃ k : ℕ, 2550 = (2^if 2 > 1 then 1 else 0) * (3^if 3 > 1 then 1 else 0) * (5^2) * (17^if 17 > 1 then 1 else 0)) → 
  (card {d : ℕ | d ∣ 2550 ∧ 3 < card {m : ℕ | m ∣ d}} = 9) :=
sorry

end num_factors_more_than_three_l482_482430


namespace arrangement_schemes_l482_482355

def numClasses : ℕ := 6
def numStudents : ℕ := 4

theorem arrangement_schemes :
  (∑ (i in (Finset.range numClasses).powerset.filter(λ x, x.card = 2)), 1) *
  (∑ (i in (Finset.range numStudents).powerset.filter(λ x, x.card = 2)), 1) / 2 =
  1/2 * (nat.choose numClasses 2) * (nat.choose numStudents 2) :=
sorry

end arrangement_schemes_l482_482355


namespace pure_imaginary_iff_real_part_zero_l482_482801

theorem pure_imaginary_iff_real_part_zero (a b : ℝ) : (∃ z : ℂ, z = a + bi ∧ z.im ≠ 0) ↔ (a = 0 ∧ b ≠ 0) :=
sorry

end pure_imaginary_iff_real_part_zero_l482_482801


namespace power_means_inequality_l482_482240

-- Definitions of power means
noncomputable def power_mean (p : ℝ) (x : List ℝ) : ℝ :=
if p = 0 then (List.prod x) ^ (1 / (x.length : ℝ))
else (List.sum (x.map (λ xi => xi ^ p)) / (x.length : ℝ)) ^ (1 / p)

-- Statement of the problem in Lean
theorem power_means_inequality {α β : ℝ} (h : α < β) (x : List ℝ) :
  power_mean α x ≤ power_mean β x ∧ 
  (power_mean α x = power_mean β x ↔ List.all (λ xi => xi = x.head) x) :=
by sorry

end power_means_inequality_l482_482240


namespace min_box_height_l482_482021

noncomputable def height_of_box (x : ℝ) := x + 4

def surface_area (x : ℝ) : ℝ := 2 * x^2 + 4 * x * (x + 4)

theorem min_box_height (x h : ℝ) (h₁ : h = height_of_box x) (h₂ : surface_area x ≥ 130) : h ≥ 25 / 3 :=
by sorry

end min_box_height_l482_482021


namespace smallest_7_heavy_three_digit_number_l482_482828

def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem smallest_7_heavy_three_digit_number :
  ∃ n : ℕ, is_three_digit n ∧ is_7_heavy n ∧ (∀ m : ℕ, is_three_digit m ∧ is_7_heavy m → n ≤ m) ∧
  n = 103 := 
by
  sorry

end smallest_7_heavy_three_digit_number_l482_482828


namespace average_yield_per_tree_l482_482991

theorem average_yield_per_tree (x : ℕ) (h_x : x = 6) :
  let nuts_per_tree_per_year := (60 * (x + 3) + 120 * x + 180 * (x - 3)) / ((x + 3) + x + (x - 3))
  in nuts_per_tree_per_year = 100 :=
by
  sorry

end average_yield_per_tree_l482_482991


namespace maximum_value_exists_l482_482254

theorem maximum_value_exists (x y : ℝ) (h_positivity_x : 0 < x) (h_positivity_y : 0 < y) (h_condition : x^2 - 3 * x * y + 4 * y^2 = 12) :
  ∃ m : ℝ, m = x^2 + 3 * x * y + 4 * y^2 ∧ m ≤ 84 :=
begin
  sorry
end

end maximum_value_exists_l482_482254


namespace ellipse_properties_l482_482085

theorem ellipse_properties 
  (a : ℝ) (ha : a > 1) 
  (l : ℝ → ℝ) (F1 F2 A B : ℝ × ℝ)
  (hleq : ∀ x : ℝ, (x+1)^2/a^2 + (l x)^2 = 1) 
  (p : ℝ × ℝ) (hp_range : p.1 ∈ Ico (-1/4) 0) 
  (ab_eq : |(A.1 - B.1, A.2 - B.2)| = 2 * real.sqrt 2 * ((3/2) - (1 / (2 * (2 * (k^2)) + 1))) = 2 * real.sqrt 2 * ((a.1 + b.1)/(2 * (a.2 + b.2)) - 1/(2 * (a.2 + b.2))) )  :
    (1 : ℝ) / 2 + 1 ≤ 1 :=
sorry

-- Note: The theorem statements are illustrative and assume that importing 
-- Mathlib brings all necessary dependencies, and that the manually written 
-- math equations translate correctly within Lean's syntax and semantics.

end ellipse_properties_l482_482085


namespace proof_problem_l482_482499

theorem proof_problem (α : ℝ) (hα1 : π / 4 < α) (hα2 : α < π / 2) :
  let a := Real.cos α ^ Real.cos α,
      b := Real.sin α ^ Real.cos α,
      c := Real.cos α ^ Real.sin α
  in c < a ∧ a < b :=
by
  sorry

end proof_problem_l482_482499


namespace perimeter_triangle_cos_A_minus_C_l482_482255

-- Define the problem and conditions
variables (a b c : ℝ) (A B C : ℝ)

-- Conditions: a = 1, b = 2, cos C = 1/2
axiom h1 : a = 1
axiom h2 : b = 2
axiom h3 : real.cos C = 1 / 2

-- Problem 1: Prove the perimeter of triangle ABC is 3 + sqrt 3
theorem perimeter_triangle (h1 : a = 1) (h2 : b = 2) (h3 : real.cos C = 1 / 2) :
    a + b + real.sqrt (a^2 + b^2 - 2 * a * b * real.cos C) = 3 + real.sqrt 3 :=
sorry

-- Problem 2: Prove cos(A - C) = (3 + sqrt 3) / 4
theorem cos_A_minus_C (h1 : a = 1) (h2 : b = 2) (h3 : real.cos C = 1 / 2) :
    real.cos (A - C) = (3 + real.sqrt 3) / 4 :=
sorry

end perimeter_triangle_cos_A_minus_C_l482_482255


namespace infertile_eggs_percentage_l482_482804

variables (total_eggs eggs_hatch : ℕ) (P : ℝ)

-- Condition: A gecko lays 30 eggs per year
def total_eggs := 30

-- Condition: Some percentage P of them are infertile
def infertile_eggs (P : ℝ) := (P * total_eggs) / 100

-- Condition: A third of the remaining eggs will not hatch due to calcification issues
def fertile_eggs (P : ℝ) : ℝ := total_eggs - infertile_eggs P

def hatching_eggs (P : ℝ) : ℝ := (2 / 3) * fertile_eggs P

-- Condition: 16 eggs actually hatch
def eggs_hatch := 16

theorem infertile_eggs_percentage : hatching_eggs P = eggs_hatch → P = 20 :=
begin
  assume h,  -- assume hatching_eggs P = 16
  sorry
end

end infertile_eggs_percentage_l482_482804


namespace cot_225_eq_1_l482_482035

theorem cot_225_eq_1 : Real.cot (225 * Real.pi / 180) = 1 := 
by
  -- Import necessary properties and convert angles from degrees to radians
  -- Define the degrees and their conversions
  have tan_45_eq_1 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_pi_div_four
  have tan_225_eq_tan_45 : Real.tan (225 * Real.pi / 180) = Real.tan (45 * Real.pi / 180) := 
    by
      rw [Real.tan_add_pi (Real.pi + Real.pi / 4)]
  have tan_225_eq_1 : Real.tan (225 * Real.pi / 180) = 1 := by
    rw [tan_225_eq_tan_45, tan_45_eq_1]
  show Real.cot (225 * Real.pi / 180) = 1 
  rw [Real.cot_eq (225 * Real.pi / 180)]
  rw [tan_225_eq_1, inv_one]
  rfl

end cot_225_eq_1_l482_482035


namespace number_of_dots_in_120_circles_l482_482824

theorem number_of_dots_in_120_circles :
  ∃ n : ℕ, (n = 14) ∧ (∀ m : ℕ, m * (m + 1) / 2 + m ≤ 120 → m ≤ n) :=
by
  sorry

end number_of_dots_in_120_circles_l482_482824


namespace spaceship_distance_l482_482819

-- Define the distance variables and conditions
variables (D : ℝ) -- Distance from Earth to Planet X
variable (T : ℝ) -- Total distance traveled by the spaceship

-- Conditions
variables (hx : T = 0.7) -- Total distance traveled is 0.7 light-years
variables (hy : D + 0.1 + 0.1 = T) -- Sum of distances along the path

-- Theorem statement to prove the distance from Earth to Planet X
theorem spaceship_distance (h1 : T = 0.7) (h2 : D + 0.1 + 0.1 = T) : D = 0.5 :=
by
  -- Proof steps would go here
  sorry

end spaceship_distance_l482_482819


namespace probability_2_le_ξ_lt_4_l482_482530

-- Let ξ be a random variable following the normal distribution N(μ, σ^2)
variables {μ σ : ℝ}

def ξ : MeasureTheory.ProbabilityTheory.ProbabilityMonadic.random_variable nnreal ℝ :=
  MeasureTheory.ProbabilityTheory.ProbabilityMonadic.normal μ σ

-- Given conditions
axiom h1 : (MeasureTheory.ProbabilityTheory.Probability (ξ < 2)) = 0.15
axiom h2 : (MeasureTheory.ProbabilityTheory.Probability (ξ > 6)) = 0.15

-- We want to prove that P(2 ≤ ξ < 4) = 0.35
theorem probability_2_le_ξ_lt_4 : (MeasureTheory.ProbabilityTheory.Probability (2 ≤ ξ ∧ ξ < 4)) = 0.35 :=
sorry

end probability_2_le_ξ_lt_4_l482_482530


namespace intersection_A_B_l482_482548

theorem intersection_A_B :
  let A := {1, 3, 5, 7}
  let B := {x | x^2 - 2 * x - 5 ≤ 0}
  A ∩ B = {1, 3} := by
sorry

end intersection_A_B_l482_482548


namespace first_set_matches_l482_482259

theorem first_set_matches 
  (avg_runs_set1 : ℝ)
  (matches_set2 : ℝ)
  (avg_runs_set2 : ℝ)
  (total_avg_runs : ℝ)
  (total_matches : ℝ)
  (avg_set1 : avg_runs_set1 = 40)
  (avg_set2 : avg_runs_set2 = 20)
  (total_avg : total_avg_runs = 33.333333333333336)
  (next_matches : matches_set2 = 10)
  (total_num_matches : total_matches = 30)
  : ∃ x : ℝ, x = 20 := 
by
  have_eq : 40 * x + 200 = 1000 := sorry
  use 20
  sorry

end first_set_matches_l482_482259


namespace accumulated_capital_exponential_accumulated_capital_linear_accumulated_capital_polynomial_l482_482329

-- Part (a): Define the function of accumulated capital as a function of time being exponential.
theorem accumulated_capital_exponential (P : ℝ) (r : ℝ) (t : ℕ) :
  ∃ f : ℕ → ℝ, ∀ t, f t = P * (1 + r) ^ t :=
begin
  use λ t, P * (1 + r) ^ t,
  intro t,
  refl,
end

-- Part (b): Define the function of accumulated capital as a function of initial principal being linear.
theorem accumulated_capital_linear (r : ℝ) (t : ℕ) :
  ∃ f : ℝ → ℝ, ∀ P, f P = P * (1 + r) ^ t :=
begin
  use λ P, P * (1 + r) ^ t,
  intro P,
  refl,
end

-- Part (c): Define the function of accumulated capital as a function of interest rate being polynomial.
theorem accumulated_capital_polynomial (P : ℝ) (t : ℕ) :
  ∃ f : ℝ → ℝ, ∀ r, f r = P * (1 + r) ^ t :=
begin
  use λ r, P * (1 + r) ^ t,
  intro r,
  refl,
end

end accumulated_capital_exponential_accumulated_capital_linear_accumulated_capital_polynomial_l482_482329


namespace find_number_l482_482898

-- Let's define the condition
def condition (x : ℝ) : Prop := x * 99999 = 58293485180

-- Statement to be proved
theorem find_number : ∃ x : ℝ, condition x ∧ x = 582.935 := 
by
  sorry

end find_number_l482_482898


namespace basic_computer_price_l482_482280

theorem basic_computer_price :
  ∃ C P : ℝ,
    C + P = 2500 ∧
    (C + 800) + (1 / 5) * (C + 800 + P) = 2500 ∧
    (C + 1100) + (1 / 8) * (C + 1100 + P) = 2500 ∧
    (C + 1500) + (1 / 10) * (C + 1500 + P) = 2500 ∧
    C = 1040 :=
by
  sorry

end basic_computer_price_l482_482280


namespace find_a1_l482_482815

noncomputable def sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 0 then a 0 else 1 / (1 - a (n - 1))

theorem find_a1 (a : ℕ → ℝ) (h0 : a 8 = 2) :
  (a 7 = 1/2) ∧ (a 6 = -1) ∧ (a 5 = 2) ∧ a 1 = 1/2 :=
  by
  sorry

end find_a1_l482_482815


namespace max_value_of_f_l482_482215

noncomputable def f (z : ℝ) : ℝ := ∫ x in 0..z, real.sqrt (x^4 + (z - z^2)^2)

theorem max_value_of_f : (∀ z : ℝ, 0 ≤ z ∧ z ≤ 1 → f z ≤ (1 / 3)) ∧ (f 1 = 1 / 3) := by
  sorry

end max_value_of_f_l482_482215


namespace sqrt_72_plus_sqrt_32_l482_482708

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l482_482708


namespace equilateral_triangle_side_length_same_perimeter_as_square_l482_482891

theorem equilateral_triangle_side_length_same_perimeter_as_square
  (side_length_square : ℝ) (h : side_length_square = 21) :
  let perimeter_square := 4 * side_length_square
  let perimeter_equilateral_triangle := perimeter_square
  let side_length_equilateral := perimeter_equilateral_triangle / 3
  in side_length_equilateral = 28 :=
by
  sorry

end equilateral_triangle_side_length_same_perimeter_as_square_l482_482891


namespace area_ratio_l482_482152

-- Given definitions
variables (A1 A2 A3 A4 A5 A6 B1 B2 B3 B4 B5 B6: Point) -- Points on the plane
variables (convex_A : convex_hexagon A1 A2 A3 A4 A5 A6) -- A1 A2 A3 A4 A5 A6 forms a convex hexagon
variables (midpoint_A6A2 : midpoint B1 A6 A2)  -- B1 is the midpoint of A6A2
variables (midpoint_A1A3 : midpoint B2 A1 A3)  -- B2 is the midpoint of A1A3
variables (midpoint_A2A4 : midpoint B3 A2 A4)  -- B3 is the midpoint of A2A4
variables (midpoint_A3A5 : midpoint B4 A3 A5)  -- B4 is the midpoint of A3A5
variables (midpoint_A4A6 : midpoint B5 A4 A6)  -- B5 is the midpoint of A4A6
variables (midpoint_A5A1 : midpoint B6 A5 A1)  -- B6 is the midpoint of A5A1
variables (convex_B : convex_hexagon B1 B2 B3 B4 B5 B6) -- B1 B2 B3 B4 B5 B6 forms a convex hexagon

-- Property to prove
theorem area_ratio :
  area (hexagon B1 B2 B3 B4 B5 B6) = (1 / 4) * area (hexagon A1 A2 A3 A4 A5 A6) :=
sorry -- proof omitted

end area_ratio_l482_482152


namespace consecutive_fours_l482_482300

theorem consecutive_fours (n : ℕ) 
  (h : ∃ a : ℕ, ∏ i in finset.range 5, (a + i) % n ≠ 0 ∧ n ∣ ∏ i in finset.range 5, (a + i)) :
  ∃ b : ℕ, ∏ i in finset.range 4, (b + i) % n ≠ 0 ∧ n ∣ ∏ i in finset.range 4, (b + i) :=
sorry

end consecutive_fours_l482_482300


namespace simplify_radicals_l482_482699

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l482_482699


namespace sum_of_integers_75_to_100_l482_482312

theorem sum_of_integers_75_to_100 : ∑ i in finset.range (100 - 75 + 1), (75 + i) = 2275 := by
  sorry

end sum_of_integers_75_to_100_l482_482312


namespace sqrt_sum_simplify_l482_482688

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l482_482688


namespace opposite_of_two_is_negative_two_l482_482268

theorem opposite_of_two_is_negative_two : -2 = -2 :=
by
  sorry

end opposite_of_two_is_negative_two_l482_482268


namespace triangle_coordinates_proves_l482_482757

/-- Define the initial conditions for the triangle OPQ and the rotation -/
def triangle_coordinates : Prop :=
  let O := (0 : ℝ, 0 : ℝ) in
  let Q := (8 : ℝ, 0 : ℝ) in
  ∃ (P : ℝ × ℝ), P.1 > 0 ∧ P.2 > 0 ∧
    let θ₁ := 90 * (Real.pi / 180) in
    let θ₂ := 45 * (Real.pi / 180) in
    let new_angle := 120 * (Real.pi / 180) in
    (Real.angle P Q O = θ₁) ∧ (Real.angle O P Q = θ₂) ∧
    let rotated_P :=
      (cos new_angle * P.1 - sin new_angle * P.2, sin new_angle * P.1 + cos new_angle * P.2) in
    rotated_P = (-4 * Real.sqrt 2 - 4 * Real.sqrt 6, 4 * Real.sqrt 6 - 4 * Real.sqrt 2)

theorem triangle_coordinates_proves :
  ∃ (P : ℝ × ℝ), triangle_coordinates :=
sorry

end triangle_coordinates_proves_l482_482757


namespace max_expr_on_circle_l482_482525

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x - 6 * y + 4 = 0

noncomputable def expr (x y : ℝ) : ℝ :=
  3 * x - 4 * y

theorem max_expr_on_circle : 
  ∃ (x y : ℝ), circle_eq x y ∧ ∀ (x' y' : ℝ), circle_eq x' y' → expr x y ≤ expr x' y' :=
sorry

end max_expr_on_circle_l482_482525


namespace vector_perpendicular_x_value_l482_482906

open Real

-- Define vectors a and b
def a : ℝ × ℝ := (-sqrt 3, 1)
def b (x : ℝ) : ℝ × ℝ := (1, x)

-- Define perpendicularity condition
def perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem vector_perpendicular_x_value (x : ℝ) : 
  perpendicular a (b x) → x = sqrt 3 := by
  sorry

end vector_perpendicular_x_value_l482_482906


namespace train_length_l482_482374

-- Definitions for the conditions
def train_speed_kmh : ℝ := 45
def passing_time_seconds : ℝ := 52
def bridge_length_meters : ℝ := 140

-- Speed conversion from km/h to m/s
def speed_m_s : ℝ := train_speed_kmh * (1000 / 3600)

-- Total distance calculated using speed * time
def total_distance : ℝ := speed_m_s * passing_time_seconds

-- Given conditions and target conclusion
theorem train_length : total_distance = 510 + bridge_length_meters :=
by
  -- Convert speeds and calculate
  let L := 510
  have h1 : speed_m_s = 12.5 := by
    sorry -- insert correct calculations here
  have h2 : total_distance = speed_m_s * passing_time_seconds := by
    sorry 
  show total_distance = 510 + bridge_length_meters from
  sorry

end train_length_l482_482374


namespace integral_evaluation_l482_482004

noncomputable def integrand (x : ℝ) : ℝ := 12 / ((6 + 5 * Real.tan x) * Real.sin (2 * x))

def lower_bound : ℝ := Real.arccos (1 / Real.sqrt 10)
def upper_bound : ℝ := Real.arccos (1 / Real.sqrt 26)

theorem integral_evaluation :
  ∫ x in lower_bound..upper_bound, integrand x = Real.log (105 / 93) :=
by
  sorry

end integral_evaluation_l482_482004


namespace number_of_true_propositions_l482_482432

noncomputable def original_proposition (a : ℝ) : Prop := a > 0 → a > 1
noncomputable def converse_proposition (a : ℝ) : Prop := a > 1 → a > 0
noncomputable def inverse_proposition (a : ℝ) : Prop := a ≤ 0 → a ≤ 1
noncomputable def contrapositive_proposition (a : ℝ) : Prop := a ≤ 1 → a ≤ 0

theorem number_of_true_propositions : 
  (¬ (∀ a, original_proposition a) ∧ (∀ a, converse_proposition a) ∧ (∀ a, inverse_proposition a) ∧ (¬ (∀ a, contrapositive_proposition a))) →
  2 := 
by
  sorry

end number_of_true_propositions_l482_482432


namespace find_m_if_no_extreme_values_l482_482727

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 - (1 / 2) * (m + 1) * x^2 + 2 * (m - 1) * x

-- Define the derivative of the function f
def f_prime (m : ℝ) (x : ℝ) : ℝ := x^2 - (m + 1) * x + 2 * (m - 1)

-- State that the function f(x) has no extreme values on (0, 4)
def has_no_extreme_values (m : ℝ) : Prop :=
  ∀ x ∈ Ioo 0 4, f_prime m x ≠ 0

-- Proof statement
theorem find_m_if_no_extreme_values (m : ℝ) (h : has_no_extreme_values m) : m = 3 :=
sorry

end find_m_if_no_extreme_values_l482_482727


namespace arithmetic_expression_equals_47_l482_482014

-- Define the arithmetic expression
def arithmetic_expression : ℕ :=
  2 + 5 * 3^2 - 4 + 6 * 2 / 3

-- The proof goal: arithmetic_expression equals 47
theorem arithmetic_expression_equals_47 : arithmetic_expression = 47 := 
by
  sorry

end arithmetic_expression_equals_47_l482_482014


namespace range_of_a_l482_482504

noncomputable def f (a x : ℝ) := (Real.exp x - a * x^2) 

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), 0 ≤ x → f a x ≥ x + 1) ↔ a ∈ Set.Iic (1/2) :=
by
  sorry

end range_of_a_l482_482504


namespace fractional_sum_l482_482078

-- Definitions based on conditions
def greatest_integer (x : ℚ) : ℤ :=
  if x ≥ 0 then
    int.floor x
  else
    int.ceil x

def fractional_part (x : ℚ) : ℚ :=
  greatest_integer x - x

-- Theorem statement based on the question and correct answer
theorem fractional_sum : fractional_part 2.9 + fractional_part (-5/3) = -37 / 30 :=
by sorry

end fractional_sum_l482_482078


namespace probability_of_at_least_19_l482_482396

-- Defining the possible coins in Anya's pocket
def coins : list ℕ := [10, 10, 5, 5, 2]

-- Function to calculate the sum of chosen coins
def sum_coins (l : list ℕ) := list.sum l

-- Function to check if the sum of chosen coins is at least 19 rubles
def at_least_19 (l : list ℕ) := (sum_coins l) ≥ 19

-- Extract all possible combinations of 3 coins from the list
def combinations (l : list ℕ) (n : ℕ) := 
  if h : n ≤ l.length then 
    (list.permutations l).dedup.map (λ p, p.take n).dedup
  else
    []

-- Specific combinations of 3 coins out of 5
def three_coin_combinations := combinations coins 3 

-- Count the number of favorable outcomes (combinations that sum to at least 19)
def favorable_combinations := list.filter at_least_19 three_coin_combinations

-- Calculate the probability
def probability := (favorable_combinations.length : ℚ) / (three_coin_combinations.length : ℚ)

-- Prove that the probability is 0.4
theorem probability_of_at_least_19 : probability = 0.4 :=
  sorry

end probability_of_at_least_19_l482_482396


namespace sin_geq_tan_minus_half_tan_cubed_l482_482659

theorem sin_geq_tan_minus_half_tan_cubed (x : ℝ) (hx : 0 ≤ x ∧ x < π / 2) :
  Real.sin x ≥ Real.tan x - 1/2 * (Real.tan x) ^ 3 := 
sorry

end sin_geq_tan_minus_half_tan_cubed_l482_482659


namespace find_triangle_sides_l482_482173

-- Define the variables and conditions
noncomputable def k := 5
noncomputable def c := 12
noncomputable def d := 10

-- Assume the perimeters of the figures
def P1 : ℕ := 74
def P2 : ℕ := 84
def P3 : ℕ := 82

-- Define the equations based on the perimeters
def Equation1 := P2 = P1 + 2 * k
def Equation2 := P3 = P1 + 6 * c - 2 * k

-- The lean theorem proving that the sides of the triangle are as given
theorem find_triangle_sides : 
  (Equation1 ∧ Equation2) →
  (k = 5 ∧ c = 12 ∧ d = 10) :=
by
  sorry

end find_triangle_sides_l482_482173


namespace smallest_digit_divisible_by_9_l482_482049

theorem smallest_digit_divisible_by_9 :
  ∃ (d : ℕ), (25 + d) % 9 = 0 ∧ (∀ e : ℕ, (25 + e) % 9 = 0 → e ≥ d) :=
by
  sorry

end smallest_digit_divisible_by_9_l482_482049


namespace total_savings_correct_l482_482191

-- Definitions of savings per day and days saved for Josiah, Leah, and Megan
def josiah_saving_per_day : ℝ := 0.25
def josiah_days : ℕ := 24

def leah_saving_per_day : ℝ := 0.50
def leah_days : ℕ := 20

def megan_saving_per_day : ℝ := 1.00
def megan_days : ℕ := 12

-- Definition to calculate total savings for each child
def total_saving (saving_per_day : ℝ) (days : ℕ) : ℝ :=
  saving_per_day * days

-- Total amount saved by Josiah, Leah, and Megan
def total_savings : ℝ :=
  total_saving josiah_saving_per_day josiah_days +
  total_saving leah_saving_per_day leah_days +
  total_saving megan_saving_per_day megan_days

-- Theorem to prove the total savings is $28
theorem total_savings_correct : total_savings = 28 := by
  sorry

end total_savings_correct_l482_482191


namespace meeting_day_correct_l482_482256

noncomputable def smallest_meeting_day :=
  ∀ (players courts : ℕ)
    (initial_reimu_court initial_marisa_court : ℕ),
    players = 2016 →
    courts = 1008 →
    initial_reimu_court = 123 →
    initial_marisa_court = 876 →
    ∀ (winner_moves_to court : ℕ → ℕ),
      (∀ (i : ℕ), 2 ≤ i ∧ i ≤ courts → winner_moves_to i = i - 1) →
      (winner_moves_to 1 = 1) →
      ∀ (loser_moves_to court : ℕ → ℕ),
        (∀ (j : ℕ), 1 ≤ j ∧ j ≤ courts - 1 → loser_moves_to j = j + 1) →
        (loser_moves_to courts = courts) →
        ∃ (n : ℕ), n = 1139

theorem meeting_day_correct : smallest_meeting_day :=
  sorry

end meeting_day_correct_l482_482256


namespace find_q_l482_482959

theorem find_q (x y : ℝ) (h1 : x + 1 / x = 5) (h2 : y - x = 2) : 
  let q := x^2 + (1 / x)^2 
  in q = 23 := 
by 
  sorry

end find_q_l482_482959


namespace find_pq_expression_monotonic_decreasing_find_m_range_l482_482535

noncomputable def f (p q x : ℝ) : ℝ := p * x + q / x

theorem find_pq_expression (p q : ℝ) (h1 : f p q 1 = 5 / 2) (h2 : f p q 2 = 17 / 4) : p = 2 ∧ q = 1 / 2 :=
sorry

noncomputable def f_explicit (x : ℝ) : ℝ := 2 * x + 1 / (2 * x)

theorem monotonic_decreasing (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : x1 < x2) (hx2_lt : x2 ≤ 1 / 2) :
  f_explicit x1 > f_explicit x2 :=
sorry

theorem find_m_range (m : ℝ) (hx : ∀ x ∈ Ioo 0 (1 / 2), f_explicit x ≥ 2 - m) : 0 ≤ m :=
sorry

end find_pq_expression_monotonic_decreasing_find_m_range_l482_482535


namespace sum_distinct_prime_factors_of_7pow7_minus_7pow4_l482_482456

noncomputable def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  let factors := (Nat.factors n).erase_dup
  factors.sum

theorem sum_distinct_prime_factors_of_7pow7_minus_7pow4 :
  sum_of_distinct_prime_factors (7 ^ 7 - 7 ^ 4) = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7pow7_minus_7pow4_l482_482456


namespace average_score_l482_482827

variable (num_students_B : ℤ)
variable (average_A : ℚ := 84)
variable (average_B : ℚ := 70)

-- Define the number of students in Class A and Class B
def num_students_A : ℤ := 3 * (num_students_B / 4)
def total_students : ℤ := num_students_A + num_students_B

-- Define the weighted average calculation
def weighted_sum : ℚ := (num_students_A : ℚ) * average_A + (num_students_B : ℚ) * average_B
def overall_average : ℚ := weighted_sum / (total_students : ℚ)

-- Prove the overall average
theorem average_score (h : num_students_B ≠ 0) : overall_average = 76 := by
  -- assume here num_students_B = 4 * x
  have h1 : num_students_A = 3 * (num_students_B / 4) := rfl
  have h2 : total_students = 7 * (num_students_B / 4) := by sorry
  have h3 : weighted_sum = 532 * (num_students_B / 4) := by sorry
  have h4 : overall_average = 532 / 7 := by sorry
  exact sorry

end average_score_l482_482827


namespace angle_ratio_l482_482606

-- Conditions and setup
variables {α β γ : Type} [linear_ordered_field α]
variables {P Q B M : β}
variables (θ ψ : α)

-- Condition 1: BP and BQ bisect ∠ABC
axiom BQ_bisects_ABC : 
  ∀ (A B C : γ), ∠(A, B, P) = ∠(P, B, C)

axiom BP_bisects_ABC : 
  ∀ (A B C : γ), ∠(A, B, Q) = ∠(Q, B, C)

-- Condition 2: BM bisects ∠PBQ
axiom BM_bisects_PBQ : 
  ∠(P, B, M) = ∠(M, B, Q)

-- Prove the desired ratio
theorem angle_ratio (h1 : BQ_bisects_ABC) (h2 : BP_bisects_ABC) (h3 : BM_bisects_PBQ) : 
  θ / ψ = 1 / 4 :=
sorry

end angle_ratio_l482_482606


namespace ratio_of_A_and_B_l482_482734

theorem ratio_of_A_and_B :
  let seq := (List.range' 1 50).map (Nat.lt_succ_self 50),  -- numbers from 1 to 50
      F := λ (a b c : Nat), (a + b) * (b + c) * (c + a),
      operations := 24,
      final_sum := 1275,
      identities_sum := (1 + 1274^3 - final_sum^2) / 3,
      max_sum := identities_sum,
      min_sum := (637^3 + 638^3 - final_sum^2) / 3,
      ratio := max_sum / min_sum
  in
  ratio = 4 :=
begin
  sorry  -- proof goes here
end

end ratio_of_A_and_B_l482_482734


namespace ratio_of_inverse_l482_482729

theorem ratio_of_inverse (a b c d : ℝ) (h : ∀ x, (3 * (a * x + b) / (c * x + d) - 2) / ((a * x + b) / (c * x + d) + 4) = x) : 
  a / c = -4 :=
sorry

end ratio_of_inverse_l482_482729


namespace min_max_SX_SY_l482_482629

theorem min_max_SX_SY (n : ℕ) (hn : 2 ≤ n) (a : Finset ℕ) 
  (ha_sum : Finset.sum a id = 2 * n - 1) :
  ∃ (min_val max_val : ℕ), 
    (min_val = 2 * n - 2) ∧ 
    (max_val = n * (n - 1)) :=
sorry

end min_max_SX_SY_l482_482629


namespace petunia_fertilizer_problem_l482_482246

theorem petunia_fertilizer_problem
  (P : ℕ)
  (h1 : 4 * P * 8 + 3 * 6 * 3 + 2 * 2 = 314) :
  P = 8 :=
by
  sorry

end petunia_fertilizer_problem_l482_482246


namespace flowers_bloom_l482_482285

theorem flowers_bloom (num_unicorns : ℕ) (flowers_per_step : ℕ) (distance_km : ℕ) (step_length_m : ℕ) 
  (h1 : num_unicorns = 6) (h2 : flowers_per_step = 4) (h3 : distance_km = 9) (h4 : step_length_m = 3) : 
  num_unicorns * (distance_km * 1000 / step_length_m) * flowers_per_step = 72000 :=
by
  sorry

end flowers_bloom_l482_482285


namespace angle_ratio_l482_482610

theorem angle_ratio (A B C P Q M : Type) (θ : ℝ)
  (h1 : ∠B P B = ∠A B C / 2)
  (h2 : ∠B Q B = ∠A B C / 2)
  (h3 : ∠M B P = ∠M B Q)
  (h4 : ∠A B C = 2 * θ) :
  ∠M B Q / ∠A B Q = 1 / 3 := by
  sorry

end angle_ratio_l482_482610


namespace total_volume_l482_482414

theorem total_volume (n_cubes : ℕ) (side_length_c : ℕ) (n_spheres : ℕ) (radius_s : ℕ) :
  n_cubes = 3 → side_length_c = 3 → n_spheres = 4 → radius_s = 2 →
  let volume_cubes := n_cubes * side_length_c^3 in
  let volume_spheres := n_spheres * (4/3 * Real.pi * radius_s^3) in
  volume_cubes + volume_spheres = 81 + 128*Real.pi/3 :=
by {
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  let volume_cubes := 3 * 3^3,
  let volume_spheres := 4 * (4/3 * Real.pi * 2^3),
  rw [Nat.cast_mul, Nat.cast_pow, Nat.cast_mul, Nat.cast_add],
  norm_num,
  rw [mul_assoc, mul_comm (4/3)],
  norm_num,
  sorry
}

end total_volume_l482_482414


namespace cats_awake_l482_482742

theorem cats_awake (total_cats : ℕ) (percent_asleep : ℝ) 
  (h_total_cats : total_cats = 1270) 
  (h_percent_asleep : percent_asleep = 0.734) : 
  (total_cats - Int.ofNat (Int.ofReal (total_cats * percent_asleep).round)) = 337 := 
by
  sorry

end cats_awake_l482_482742


namespace range_f_area_under_f_l482_482943

-- Define the function
def f (x : ℝ) : ℝ := x + Real.sin x

-- Define the given interval
def a : ℝ := Real.pi / 2
def b : ℝ := Real.pi

noncomputable def f_min : ℝ := a + 1
noncomputable def f_max : ℝ := b

theorem range_f : Set.range f = Set.Icc (a + 1) b :=
by sorry

theorem area_under_f : ∫ x in Icc a b, f x = (3 * Real.pi ^ 2) / 8 + 1 :=
by sorry

end range_f_area_under_f_l482_482943


namespace nat_numbers_condition_l482_482429

open Nat

theorem nat_numbers_condition (n : ℕ) (h : n ≥ 2) : 
  (∃ a b : ℕ, a^2 + b^2 = n ∧ a > 1 ∧ ∀ d : ℕ, d ∣ n → d = a ∨ d = 1 ∨ a = ∃ m : ℕ, n = m * a) → 
  (n = 8 ∨ n = 20) :=
sorry

end nat_numbers_condition_l482_482429


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l482_482483

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31 :
  let n := 7^7 - 7^4 in
  let prime_factors := {2, 3, 7, 19} in
  finset.sum prime_factors id = 31 :=
by
  sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l482_482483


namespace acute_angle_of_rhombus_l482_482039

theorem acute_angle_of_rhombus (a α : ℝ) (V1 V2 : ℝ) (OA BD AN AB : ℝ) 
  (h_volumes : V1 / V2 = 1 / (2 * Real.sqrt 5)) 
  (h_V1 : V1 = (1 / 3) * Real.pi * (OA^2) * BD)
  (h_V2 : V2 = Real.pi * (AN^2) * AB)
  (h_OA : OA = a * Real.sin (α / 2))
  (h_BD : BD = 2 * a * Real.cos (α / 2))
  (h_AN : AN = a * Real.sin α)
  (h_AB : AB = a)
  : α = Real.arccos (1 / 9) :=
sorry

end acute_angle_of_rhombus_l482_482039


namespace sum_of_roots_l482_482932

def function_f (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 + x) = f (2 - x)

def has_four_distinct_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 

theorem sum_of_roots (f : ℝ → ℝ) 
  (h_f_symm : function_f f) 
  (h_four_roots : has_four_distinct_real_roots f) : 
  ∃ a b c d : ℝ, a + b + c + d = 8 :=
begin
  sorry
end

end sum_of_roots_l482_482932


namespace problem_solution_l482_482945

noncomputable def f (x : ℝ) := 2 * x / (x + 1)

def symmetric_about_x2 (f : ℝ → ℝ) : ℝ → ℝ := 
  λ x, f (4 - x)

noncomputable def g (x : ℝ) := if x ≠ 5 then (2 * x - 8) / (x - 5) else 0

axiom Phi_periodic : ∀ x : ℝ, -2 < x ∧ x < 0 → (Φ (x + 2) = 1 / Φ x)

lemma Phi_period (x : ℝ) (h : -2 < x ∧ x < 0) : 
  ∀ n : ℤ, Φ (x + 4 * n) = Φ x := sorry

lemma Phi2005_value : Φ 2005 = 3 / 5 := sorry

theorem problem_solution :
  (∀ x : ℝ, g x = symmetric_about_x2 f x) →
  (Φ = g) → 
  (∀ x : ℝ, -2 < x ∧ x < 0 → Phi_periodic x) →
  Φ 2005 = 3 / 5 :=
begin
  intros h_sym h_Phi h_periodic,
  rw [Phi2005_value],
  sorry
end

end problem_solution_l482_482945


namespace find_number_l482_482881

theorem find_number: ∃ x: ℝ, 0.6667 * x - 10 = 0.25 * x ∧ x ≈ 24 :=
by
  sorry

end find_number_l482_482881


namespace x_eq_sum_of_squares_of_two_consecutive_integers_l482_482739

noncomputable def x_seq (n : ℕ) : ℝ :=
  1 / 4 * ((2 + Real.sqrt 3) ^ (2 * n - 1) + (2 - Real.sqrt 3) ^ (2 * n - 1))

theorem x_eq_sum_of_squares_of_two_consecutive_integers (n : ℕ) : 
  ∃ y : ℤ, x_seq n = (y:ℝ)^2 + (y + 1)^2 :=
sorry

end x_eq_sum_of_squares_of_two_consecutive_integers_l482_482739


namespace log_interval_l482_482910

theorem log_interval (x : ℝ)
  (h : x = log 6 / log 5 * log 7 / log 6 * log 8 / log 7) :
  1 < x ∧ x < 2 :=
by
  sorry

end log_interval_l482_482910


namespace difference_between_largest_and_third_largest_l482_482761

theorem difference_between_largest_and_third_largest : 
  let digits := [1, 6, 8] in
  let permutations := [(8,6,1), (8,1,6), (6,8,1), (6,1,8), (1,8,6), (1,6,8)] in
  let largest := 861 in
  let third_largest := 681 in
  largest - third_largest = 180 :=
by
  let digits := [1, 6, 8]
  let permutations := [(8,6,1), (8,1,6), (6,8,1), (6,1,8), (1,8,6), (1,6,8)]
  let largest := 861
  let third_largest := 681
  show largest - third_largest = 180
  sorry

end difference_between_largest_and_third_largest_l482_482761


namespace probability_is_0_4_l482_482391

def coin_values : List ℕ := [10, 10, 5, 5, 2]

def valid_combination (comb : List ℕ) : Prop :=
  comb.sum ≥ 19

def favorable_outcomes : Finset (Finset ℕ) :=
  {s ∈ coin_values.to_finset.powerset.filter (λ s, s.card = 3) | valid_combination s.val.to_list}

def total_outcomes : Finset (Finset ℕ) :=
  coin_values.to_finset.powerset.filter (λ s, s.card = 3)

def probability : ℚ :=
  favorable_outcomes.card / total_outcomes.card

theorem probability_is_0_4 : probability = 2 / 5 :=
by
  -- Proof will go here
  sorry

end probability_is_0_4_l482_482391


namespace jelly_bean_probability_l482_482807

theorem jelly_bean_probability :
  ∀ (p_red p_orange p_green p_yellow : ℝ),
  p_red = 0.15 →
  p_orange = 0.4 →
  p_green = 0.1 →
  p_red + p_orange + p_green + p_yellow = 1 →
  p_yellow = 0.35 :=
by
  intros p_red p_orange p_green p_yellow h_red h_orange h_green h_total 
  have h_sum : p_red + p_orange + p_green + p_yellow = 1 := h_total
  linarith only [h_red, h_orange, h_green, h_total]

end jelly_bean_probability_l482_482807


namespace simplest_quadratic_radical_l482_482315

theorem simplest_quadratic_radical :
  ∀ (r1 r2 r3 r4 : ℝ), 
    r1 = sqrt 27 → 
    r2 = sqrt 9 → 
    r3 = sqrt (1/4) → 
    r4 = sqrt 6 → 
    (r4 < r1 ∧ r4 < r2 ∧ r4 < r3) :=
begin
  sorry
end

end simplest_quadratic_radical_l482_482315


namespace slope_and_intercept_l482_482277

-- Define the linear function
def linear_function (x : ℝ) : ℝ := 3 * x + 2

-- State the problem to prove
theorem slope_and_intercept :
  let slope := 3 in
  let intercept := 2 in
  (∃ m b : ℝ, (∀ x : ℝ, linear_function x = m * x + b) ∧ m = slope ∧ b = intercept) :=
by
  sorry

end slope_and_intercept_l482_482277


namespace total_rainfall_2005_l482_482590

theorem total_rainfall_2005
  (rainfall_2003 : ℕ)
  (increase_per_year : ℕ)
  (years_passed : ℕ)
  (months_in_year : ℕ)
  (total_rainfall : ℕ) :
  rainfall_2003 = 30 →
  increase_per_year = 3 →
  years_passed = 2 →
  months_in_year = 12 →
  total_rainfall = (rainfall_2003 + increase_per_year * years_passed) * months_in_year →
  total_rainfall = 432 :=
begin
  sorry
end

end total_rainfall_2005_l482_482590


namespace trigonometric_identity_l482_482853

theorem trigonometric_identity (α : ℝ) :
  (sin (2 * Real.pi + α / 4) * (cos (α / 8) / sin (α / 8))
  - cos (2 * Real.pi + α / 4))
  / (cos (α / 4 - 3 * Real.pi) * (cos (α / 8) / sin (α / 8))
  + cos (7 / 2 * Real.pi - α / 4)) 
  = -tan (α / 8) :=
by
  sorry

end trigonometric_identity_l482_482853


namespace evaluate_expression_l482_482032

theorem evaluate_expression (x : Real) (hx : x = -52.7) : 
  ⌈(⌊|x|⌋ + ⌈|x|⌉)⌉ = 105 := by
  sorry

end evaluate_expression_l482_482032


namespace sequence_properties_l482_482069

open BigOperators

/-- Given a sequence {a_n} that satisfies a_{n+1} = a_n + 2^n for n ∈ ℕ* and a₁ = 2,
    and b_n = log_2(a_n), prove that:
    1. The general formula for the sequence is a_n = 2^n.
    2. The sum of the first n terms of the sequence {a_n * b_n} is 
       (n - 1) * 2^(n + 1) + 2. -/
theorem sequence_properties (a : ℕ+ → ℕ) (b : ℕ+ → ℕ)
  (h1 : ∀ n : ℕ+, a (n + 1) = a n + 2^n) 
  (h2 : a 1 = 2)
  (h3 : ∀ n : ℕ+, b n = Nat.log 2 (a n)) :
  (∀ n : ℕ+, a n = 2^n) ∧ 
  (∀ n : ℕ+, ∑ i in Finset.range n, a i.succ * b i.succ = (n - 1) * 2 ^ (n + 1) + 2) :=
by
  sorry

end sequence_properties_l482_482069


namespace sqrt_72_plus_sqrt_32_l482_482712

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l482_482712


namespace simplify_sum_of_square_roots_l482_482680

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l482_482680


namespace center_of_symmetry_g_x_l482_482537

theorem center_of_symmetry_g_x 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h₁ : ∀ x, f(x) = 1/2 * sin (2 * x) + sqrt 3 / 2 * cos (2 * x))
  (h₂ : ∀ x, g(x) = f(2 * (x - π / 6))) : 
  ∃ k : ℤ, ∀ x, g(x) = cos (x) ∧ (g(x) = cos (x) → (2 * k * π + π / 2, 0) = (k * π + π / 2, 0)) := 
sorry

end center_of_symmetry_g_x_l482_482537


namespace part1_part2i_part2ii_l482_482539

-- Part (1)
theorem part1 (f : ℝ → ℝ) (x1 x2 : ℝ) (h_f : ∀ x, f x = -3 * x^2 + 1) :
  f ((x1 + x2) / 2) ≥ (f x1 + f x2) / 2 :=
sorry

-- Part (2)(i)
theorem part2i (h : ℝ → ℝ) (g : ℝ → ℝ) (a b : ℝ)
  (h_h : ∀ x, h (2 * a - x) + h x = 2 * b)
  (h_g : ∀ x, g x = -3 * x^2 + x^3 - 1)
  (h_center : a = 1 ∧ b = -2) : true :=
sorry

-- Part (2)(ii)
theorem part2ii
  (f g : ℝ → ℝ) (h_f : ∀ x, f x = -3 * x^2 + 1) (h_g : ∀ x, g x = f x + x^3 - 1)
  (h_symm : ∀ x, g (2 - x) + g x = -4) :
  S = ∑ i in finset.range 2024, g (i / 2023) = -8090 :=
sorry

end part1_part2i_part2ii_l482_482539


namespace distance_uniquely_determined_l482_482384

-- Define the conditions
variables (a b d : ℝ) (h_speed_ana_pos : a > 0) (h_speed_bor_pos : b > 0) (h_ana_meet_b : 2 * a * b / (2 * (a + b)) = 2)

-- Define the Lean statement
theorem distance_uniquely_determined :
  let dist_moved_towards_B := 2 * a * b / (2 * (a + b)) in
  let dist_moved_towards_A := 2 * a * b / (2 * (a + b)) in
  dist_moved_towards_B = 2 ∧ dist_moved_towards_A = 2 :=
by {
  rw [←h_ana_meet_b],
  exact ⟨h_ana_meet_b, h_ana_meet_b⟩,
  sorry
}

end distance_uniquely_determined_l482_482384


namespace optimal_initial_moves_l482_482738

-- Define the chessboard and the queen's move rules
inductive Square : Type
| a1 | a2 | a3 | a4 | a5 | a6 | a7 | a8
| b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8
| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8
| d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8
| e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8
| f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8
| g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8
| h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8
deriving DecidableEq

def is_diagonal : Square → Square → Prop
| Square.a1, Square.b2 | Square.b2, Square.a1 => true
| ... -- continue defining all the acceptable diagonal pairs
| _, _ => false

def is_right : Square → Square → Prop
| Square.a1, Square.a2 | Square.a2, Square.a3 => true
| ... -- continue defining all the acceptable right pairs
| _, _ => false

def is_up : Square → Square → Prop
| Square.a1, Square.b1 | Square.b1, Square.c1 => true
| ... -- continue defining all the acceptable up pairs
| _, _ => false

def move : Square → Square → Prop :=
λ sq1 sq2, is_diagonal sq1 sq2 ∨ is_right sq1 sq2 ∨ is_up sq1 sq2

theorem optimal_initial_moves :
  (move Square.c1 Square.c5 ∨ move Square.c1 Square.e3 ∨ move Square.c1 Square.d1) →
  ∃ sq : Square, sq = Square.h8 :=
by
  sorry -- Proof omitted

end optimal_initial_moves_l482_482738


namespace collinear_A2_B2_C2_l482_482515

-- Definitions of points and lines
variables (A B C l : Type) [Point A] [Point B] [Point C] [Line l]

-- Definitions of midpoints of segments on line l
variables (A1 B1 C1 : Type) [Midpoint A1] [Midpoint B1] [Midpoint C1]

-- Definitions of intersection points of corresponding lines
variables (A2 B2 C2 : Type) [Intersection A2] [Intersection B2] [Intersection C2]

-- Theorem: Proving the collinearity of points A2, B2, and C2
theorem collinear_A2_B2_C2
    (hA1 : is_midpoint A l A1)
    (hB1 : is_midpoint B l B1)
    (hC1 : is_midpoint C l C1)
    (hA2 : is_intersection (line_through A A1) (line_through B C) A2)
    (hB2 : is_intersection (line_through B B1) (line_through A C) B2)
    (hC2 : is_intersection (line_through C C1) (line_through A B) C2) :
    collinear {A2, B2, C2} :=
by
  sorry -- Proof is omitted

end collinear_A2_B2_C2_l482_482515


namespace total_votes_cast_l482_482332

theorem total_votes_cast (F A T : ℕ) (h1 : F = A + 70) (h2 : A = 2 * T / 5) (h3 : T = F + A) : T = 350 :=
by
  sorry

end total_votes_cast_l482_482332


namespace elena_bread_max_flour_l482_482441

variable (butter_per_cup_flour butter sugar_per_cup_flour sugar : ℕ)
variable (available_butter available_sugar : ℕ)

def max_flour (butter_per_cup_flour butter sugar_per_cup_flour sugar : ℕ)
  (available_butter available_sugar : ℕ) : ℕ :=
  min (available_butter * sugar / butter_per_cup_flour) (available_sugar * butter / sugar_per_cup_flour)

theorem elena_bread_max_flour : 
  max_flour 3 4 2 5 24 30 = 32 := sorry

end elena_bread_max_flour_l482_482441


namespace ratio_sheep_horses_l482_482406

theorem ratio_sheep_horses (amount_food_per_horse : ℕ) (total_food_per_day : ℕ) (num_sheep : ℕ) (num_horses : ℕ) :
  amount_food_per_horse = 230 ∧ total_food_per_day = 12880 ∧ num_sheep = 24 ∧ num_horses = total_food_per_day / amount_food_per_horse →
  num_sheep / num_horses = 3 / 7 :=
by
  sorry

end ratio_sheep_horses_l482_482406


namespace total_miles_walked_l482_482183

-- Definitions of the conditions
def num_ladies := 5
def daily_miles_together := 3
def jamie_additional_daily_miles := 2
def sue_additional_daily_miles := (jamie_additional_daily_miles / 2)

-- The theorem statement proving the total miles walked
theorem total_miles_walked (d : ℕ) : 
  let total_miles := num_ladies * daily_miles_together * d + (jamie_additional_daily_miles * d) + (sue_additional_daily_miles * d)
  in total_miles = 18 * d :=
by
  -- Proof will be here
  sorry

end total_miles_walked_l482_482183


namespace apple_cost_l482_482840

theorem apple_cost (l q : ℕ)
  (h1 : 30 * l + 6 * q = 366)
  (h2 : 15 * l = 150)
  (h3 : 30 * l + (333 - 30 * l) / q * q = 333) :
  30 + (333 - 30 * l) / q = 33 := 
sorry

end apple_cost_l482_482840


namespace smallest_four_digit_negative_congruent_one_mod_37_l482_482309

theorem smallest_four_digit_negative_congruent_one_mod_37 :
  ∃ (x : ℤ), x < -999 ∧ x >= -10000 ∧ x ≡ 1 [MOD 37] ∧ ∀ y, y < -999 ∧ y >= -10000 ∧ y ≡ 1 [MOD 37] → y ≥ x :=
sorry

end smallest_four_digit_negative_congruent_one_mod_37_l482_482309


namespace net_effect_on_sale_l482_482335

theorem net_effect_on_sale 
  (P S : ℝ) :
  let new_price := 0.85 * P,
      new_sales := 1.80 * S,
      original_revenue := P * S,
      new_revenue := new_price * new_sales in
  (new_revenue - original_revenue) / original_revenue = 0.53 :=
by
  let new_price := 0.85 * P
  let new_sales := 1.80 * S
  let original_revenue := P * S
  let new_revenue := new_price * new_sales
  sorry

end net_effect_on_sale_l482_482335


namespace determine_age_l482_482320

def David_age (D Y : ℕ) : Prop := Y = 2 * D ∧ Y = D + 7

theorem determine_age (D : ℕ) (h : David_age D (D + 7)) : D = 7 :=
by
  sorry

end determine_age_l482_482320


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482468

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let expr := 7 ^ 7 - 7 ^ 4 in
  (7^4 * (7^3 - 1) = expr) ∧ (7^3 - 1 = 342) ∧ (Prime 2) ∧ (Prime 3) ∧ (Prime 7) ∧ (Prime 19) ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ 342 → p = 2 ∨ p = 3 ∨ p = 19) → 
  (∀ p : ℕ, Nat.Prime p → p ∣ expr → p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19) → 
  (2 + 3 + 7 + 19 = 31) := 
by
  intro expr fact1 fact2 prime2 prime3 prime7 prime19 factors342 factorsExpr
  sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482468


namespace cats_favorite_number_is_13_l482_482416

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_prime (n : ℕ) : Prop :=
  is_prime n ∧ 10 ≤ n ∧ n < 100

def first_digit_lt_second (n : ℕ) : Prop :=
  let d1 := n / 10 in
  let d2 := n % 10 in
  d1 < d2

def reverse_is_prime (n : ℕ) : Prop :=
  let rev := (n % 10) * 10 + (n / 10) in
  is_prime rev

def random_digit_not_unique (n : ℕ) : Prop :=
  ∀ d : ℕ, (d = n / 10 ∨ d = n % 10) → d ∈ { (p / 10) | p ∈ eligible_numbers } ∨ d ∈ { (p % 10) | p ∈ eligible_numbers }

def units_digit_unique (n : ℕ) : Prop :=
  ∃ u : ℕ, u = n % 10 ∧ ∀ m : ℕ, m ≠ n → ¬(m % 10 = u)


def eligible_numbers : set ℕ := {13, 17, 37, 79, 97}

theorem cats_favorite_number_is_13 :
  ∃ ! n : ℕ, two_digit_prime n ∧ first_digit_lt_second n ∧ reverse_is_prime n ∧ random_digit_not_unique n ∧ units_digit_unique n :=
by
  sorry

end cats_favorite_number_is_13_l482_482416


namespace joint_mean_approx_l482_482020

-- Define the standard deviations and means as variables
variables (σ1 σ2 σ3 μ1 μ2 μ3 : ℝ)

-- Assuming the conditions given in the problem
theorem joint_mean_approx :
  σ1 = 2 ∧ σ2 = 4 ∧ σ3 = 6 ∧
  (μ1 - 3 * σ1 > 48) ∧ 
  (μ2 = μ1 + σ2) ∧ 
  (μ3 = μ2 + σ3) →
  ((μ1 + μ2 + μ3) / 3 ≈ 58.68) :=
  by
    sorry

end joint_mean_approx_l482_482020


namespace seq_fifth_element_is_31_l482_482342

-- Define the sequence function
def seq : ℕ → ℕ
| 0 := 1
| 1 := 3
| 2 := 7
| 3 := 15
| 4 := 31
| 5 := 63
| _ := 0  -- We define others as 0 because we don't need them in this proof.

-- Prove that the fifth element of the sequence is 31
theorem seq_fifth_element_is_31 : seq 4 = 31 :=
by {
  -- The proof is omitted here, marked with sorry
  sorry
}

end seq_fifth_element_is_31_l482_482342


namespace sequence_solution_l482_482276

variables {α β λ : ℝ}
variables (x : ℕ → ℝ)

noncomputable def recur_seq (x : ℕ → ℝ) (α β λ : ℝ) : Prop :=
  x 0 = 1 ∧
  x 1 = λ ∧
  ∀ n > 1, (α + β) ^ n * x n = 
    (Finset.range (n + 1)).sum (λ k, (α ^ (n - k)) * (β ^ k) * (x (n - k)) * (x k))

theorem sequence_solution {α β λ : ℝ} (hα : 0 < α) (hβ : 0 < β) (hλ : 0 < λ) :
  recur_seq x α β λ ↔ (∀ n : ℕ, x n = λ ^ n / n!) ∧ 
    (x (nat.floor λ) ≥ x n ∀ n, n ≠ (nat.floor λ)) :=
sorry

end sequence_solution_l482_482276


namespace simplify_sum_of_square_roots_l482_482679

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l482_482679


namespace trapezoid_area_sum_l482_482550

-- Definition of side lengths
def side1 : ℕ := 4
def side2 : ℕ := 6
def side3 : ℕ := 8
def side4 : ℕ := 10

-- Main theorem statement to prove
theorem trapezoid_area_sum : 
  greatest_integer_le_sum_area side1 side2 side3 side4 = 96 := 
sorry

end trapezoid_area_sum_l482_482550


namespace total_volume_in_liters_l482_482821

/-- Definition of the conditions for the given problem -/
variables (n250 n300 n350 : ℕ)
variables (vol250 vol300 vol350 : ℕ)
variables (totalBottles totalVolume : ℕ)
variable (totalVolumeLiters : ℚ)

-- Quantity of each type of bottle
def num250 := 20
def num300 := 25
def numTotal := 60
def num350 := numTotal - num250 - num300

-- Volume of each type of bottle in mL
def vol250 := 250
def vol300 := 300
def vol350 := 350

-- Total volume in mL
def totalVolume := num250 * vol250 + num300 * vol300 + num350 * vol350

-- Conversion from mL to liters
def totalVolumeLiters := (totalVolume : ℚ) / 1000

-- The theorem to prove
theorem total_volume_in_liters :
  totalVolumeLiters = 17.75 := by
  sorry

end total_volume_in_liters_l482_482821


namespace arithmetic_sequence_and_sum_l482_482516

-- Define the arithmetic sequence {a_n} and its sum S_n
variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Given conditions
axiom sum_first_two : S 2 = 16
axiom geometric_sequence_cond : ∀ d n, a (n+1) - a n = d
axiom initial_terms_geom_seq : ∃ k: ℕ, a 1 * (a 1 + 2 * k - 8) = (a 1 + k - 4)^2

-- Definitions for general term of a_n
def a_n := ∀ n, a n = 4*n + 2

-- Finding sum T_n of sequence {b_n}
def b : ℕ → ℕ := λ n, ((S n / (2 * n)) * (a n - 2) / (2 * n))^n
def T : ℕ → ℕ
| 0             := 0
| (n + 1) := T n + b (n + 1)

-- Correct answer for the sum of the first n terms (T_n)
def T_n_correct (n : ℕ) : Prop := T n = (n + 1) * 2^(n + 1) - 2

-- The statement to prove
theorem arithmetic_sequence_and_sum : (∀ n, S n = n * (4 * n + 2)) ∧ (∀ n, a n = 4 * n + 2) ∧ (∀ n, T n = (n + 1) * 2^(n + 1) - 2) :=
by
  -- Insert the proofs here
  sorry

end arithmetic_sequence_and_sum_l482_482516


namespace total_pies_sold_l482_482858

theorem total_pies_sold :
  let shepherd_slices := 52
  let chicken_slices := 80
  let shepherd_pieces_per_pie := 4
  let chicken_pieces_per_pie := 5
  let shepherd_pies := shepherd_slices / shepherd_pieces_per_pie
  let chicken_pies := chicken_slices / chicken_pieces_per_pie
  shepherd_pies + chicken_pies = 29 :=
by
  sorry

end total_pies_sold_l482_482858


namespace evaluate_f_3_minus_f_neg3_l482_482870

def f (x : ℝ) : ℝ := x^6 + x^4 + 3*x^3 + 4*x^2 + 8*x

theorem evaluate_f_3_minus_f_neg3 : f 3 - f (-3) = 210 := by
  sorry

end evaluate_f_3_minus_f_neg3_l482_482870


namespace gift_distribution_l482_482666

def total_people : ℕ := 12 + 6 + 9
def budget : ℝ := 100
def budget_per_person : ℝ := budget / total_people
def paper_star_cost : ℝ := 1
def cookie_cost : ℝ := 1.50
def candle_cost : ℝ := 2.50

def num_paper_stars : ℕ := ⌊budget_per_person / paper_star_cost⌋.toNat
def num_cookies : ℕ := ⌊budget_per_person / cookie_cost⌋.toNat
def num_candles : ℕ := ⌊budget_per_person / candle_cost⌋.toNat

theorem gift_distribution :
  num_paper_stars = 3 ∧
  num_cookies = 2 ∧
  num_candles = 1 :=
by
  sorry

end gift_distribution_l482_482666


namespace honeydews_initially_l482_482023

def initial_cantaloupes : ℕ := 30
def initial_honeydews : ℕ -- unknown
def price_cantaloupe : ℕ := 2
def price_honeydew : ℕ := 3
def dropped_cantaloupes : ℕ := 2
def rotten_honeydews : ℕ := 3
def remaining_cantaloupes : ℕ := 8
def remaining_honeydews : ℕ := 9
def total_revenue : ℕ := 85

theorem honeydews_initially (initial_honeydews : ℕ) :
  let sold_cantaloupes := initial_cantaloupes - dropped_cantaloupes - remaining_cantaloupes,
      revenue_cantaloupes := sold_cantaloupes * price_cantaloupe,
      revenue_honeydews := total_revenue - revenue_cantaloupes,
      sold_honeydews := revenue_honeydews / price_honeydew
  in initial_honeydews = sold_honeydews + remaining_honeydews + rotten_honeydews :=
by
  let sold_cantaloupes := initial_cantaloupes - dropped_cantaloupes - remaining_cantaloupes
  let revenue_cantaloupes := sold_cantaloupes * price_cantaloupe
  let revenue_honeydews := total_revenue - revenue_cantaloupes
  let sold_honeydews := revenue_honeydews / price_honeydew
  have h : initial_honeydews = sold_honeydews + remaining_honeydews + rotten_honeydews := sorry
  exact h

end honeydews_initially_l482_482023


namespace population_decreases_by_2015_l482_482877

noncomputable def year_population_decreases (P_0 : ℝ) : ℤ :=
  let target_population := 0.2 * P_0
  in let decrease_rate := 0.7
  in let n := Real.log target_population / Real.log decrease_rate
  in Int.ceil (n.to_real)

theorem population_decreases_by_2015 (P_0 : ℝ) : year_population_decreases P_0 - 2010 = 5 := by
  sorry

end population_decreases_by_2015_l482_482877


namespace area_triangle_l482_482040

-- Definition of the lines and their intercepts
def line1 (x : ℝ) : ℝ := 2 * x + 4
def line2 (x : ℝ) : ℝ := (6 + x) / 2

-- Definitions to be used in the proof
def y_intercept_line1 : ℝ := line1 0
def y_intercept_line2 : ℝ := line2 0
def x_intercept_line1 : ℝ := -2
def x_intercept_line2 : ℝ := -6
def height : ℝ := y_intercept_line1 - y_intercept_line2
def base : ℝ := abs (x_intercept_line1 - x_intercept_line2)
def area_of_triangle : ℝ := (1 / 2) * base * height

-- Lean statement
theorem area_triangle : area_of_triangle = 2 :=
by
  sorry

end area_triangle_l482_482040


namespace problem_statement_l482_482878

-- Define all the conditions from the problem
def card := (shape : Fin 3) × (color : Fin 3) × (shade : Fin 3) × (size : Fin 3)

def isComplementary (c1 c2 c3 : card) : Prop :=
  ((c1.1 = c2.1 ∧ c2.1 = c3.1) ∨ (c1.1 ≠ c2.1 ∧ c2.1 ≠ c3.1 ∧ c1.1 ≠ c3.1)) ∧
  ((c1.2 = c2.2 ∧ c2.2 = c3.2) ∨ (c1.2 ≠ c2.2 ∧ c2.2 ≠ c3.2 ∧ c1.2 ≠ c3.2)) ∧
  ((c1.3 = c2.3 ∧ c2.3 = c3.3) ∨ (c1.3 ≠ c2.3 ∧ c2.3 ≠ c3.3 ∧ c1.3 ≠ c3.3)) ∧
  ((c1.4 = c2.4 ∧ c2.4 = c3.4) ∨ (c1.4 ≠ c2.4 ∧ c2.4 ≠ c3.4 ∧ c1.4 ≠ c3.4))

-- The deck of cards contains all combinations of attributes
def deck : List card := 
  List.product (List.product (List.product (List.range 3) (List.range 3)) (List.range 3)) (List.range 3)

-- Proposition stating the problem's question
def numberOfComplementarySets : Nat := 
  (deck.product deck).product deck |>.count (λ ((c1, c2), c3) => isComplementary c1 c2 c3)

theorem problem_statement : numberOfComplementarySets = 2400 :=
  sorry

end problem_statement_l482_482878


namespace goals_by_P15_to_P24_l482_482366

theorem goals_by_P15_to_P24
  (total_goals : ℕ)
  (games_played : ℕ)
  (goals_P1 : ℕ) (goals_P2_P8 : ℕ) (goals_P9 : ℕ) (goals_P10 : ℕ)
  (goals_P11 : ℕ) (goals_P12 : ℕ) (goals_P13 : ℕ) (goals_P14 : ℕ)
  (goals_total : ℕ) :
  total_goals = 150 →
  games_played = 15 →
  goals_P1 = 10 →
  goals_P2_P8 = 52.5 → -- this will need to be adjusted to handle 7.5 in Lean (probably use rational numbers)
  goals_P9 = 9 →
  goals_P10 = 9 →
  goals_P11 = 6 →
  goals_P12 = 6 →
  goals_P13 = 3.75 → -- this will need to be adjusted to handle 3.75 in Lean
  goals_P14 = 3.75 → -- this will need to be adjusted to handle 3.75 in Lean
  goals_total = goals_P1 + goals_P2_P8 + goals_P9 + goals_P10 + goals_P11 + goals_P12 + goals_P13 + goals_P14 →
  (150 - goals_total) = 50 :=
by {
  intros,
  rw [H, H_1, H_2, H_3, ←H_4, ←H_5, ←H_6, ←H_7, ←H_8, ←H_9],
  sorry
}

end goals_by_P15_to_P24_l482_482366


namespace solve_y_l482_482718

theorem solve_y (y : ℤ) (h : 7 - y = 10) : y = -3 := by
  sorry

end solve_y_l482_482718


namespace vector_subtraction_proof_l482_482964

theorem vector_subtraction_proof (a b : ℝ × ℝ) (ha : a = (3, 2)) (hb : b = (0, -1)) :
    3 • b - a = (-3, -5) := by
  sorry

end vector_subtraction_proof_l482_482964


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482466

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let expr := 7 ^ 7 - 7 ^ 4 in
  (7^4 * (7^3 - 1) = expr) ∧ (7^3 - 1 = 342) ∧ (Prime 2) ∧ (Prime 3) ∧ (Prime 7) ∧ (Prime 19) ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ 342 → p = 2 ∨ p = 3 ∨ p = 19) → 
  (∀ p : ℕ, Nat.Prime p → p ∣ expr → p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19) → 
  (2 + 3 + 7 + 19 = 31) := 
by
  intro expr fact1 fact2 prime2 prime3 prime7 prime19 factors342 factorsExpr
  sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482466


namespace inscribed_circle_radius_l482_482307

theorem inscribed_circle_radius (AB AC BC : ℝ) (h1 : AB = 6) (h2 : AC = 8) (h3 : BC = 10) : 
  (let s := (AB + AC + BC) / 2 in
   let K := Math.sqrt (s * (s - AB) * (s - AC) * (s - BC)) in
   let r := K / s in
   r = 2) :=
  sorry

end inscribed_circle_radius_l482_482307


namespace minimum_possible_length_of_third_side_l482_482130

theorem minimum_possible_length_of_third_side (a b : ℝ) (h : a = 8 ∧ b = 15 ∨ a = 15 ∧ b = 8) : 
  ∃ c : ℝ, (c * c = a * a + b * b ∨ c * c = a * a - b * b ∨ c * c = b * b - a * a) ∧ c = Real.sqrt 161 :=
by
  sorry

end minimum_possible_length_of_third_side_l482_482130


namespace solve_for_n_l482_482016

theorem solve_for_n (n : ℤ) (h : (5/4 : ℚ) * n + (5/4 : ℚ) = n) : n = -5 := by
    sorry

end solve_for_n_l482_482016


namespace find_percentage_decrease_l482_482271

noncomputable def initialPrice : ℝ := 100
noncomputable def priceAfterJanuary : ℝ := initialPrice * 1.30
noncomputable def priceAfterFebruary : ℝ := priceAfterJanuary * 0.85
noncomputable def priceAfterMarch : ℝ := priceAfterFebruary * 1.10

theorem find_percentage_decrease :
  ∃ (y : ℝ), (priceAfterMarch * (1 - y / 100) = initialPrice) ∧ abs (y - 18) < 1 := 
sorry

end find_percentage_decrease_l482_482271


namespace probability_both_segments_successful_expected_number_of_successful_segments_probability_given_3_successful_l482_482488

-- Definitions and conditions from the problem
def success_probability_each_segment : ℚ := 3 / 4
def num_segments : ℕ := 4

-- Correct answers from the solution
def prob_both_success : ℚ := 9 / 16
def expected_successful_segments : ℚ := 3
def cond_prob_given_3_successful : ℚ := 3 / 4

theorem probability_both_segments_successful :
  (success_probability_each_segment * success_probability_each_segment) = prob_both_success :=
by
  sorry

theorem expected_number_of_successful_segments :
  (num_segments * success_probability_each_segment) = expected_successful_segments :=
by
  sorry

theorem probability_given_3_successful :
  let prob_M := 4 * (success_probability_each_segment^3 * (1 - success_probability_each_segment))
  let prob_NM := 3 * (success_probability_each_segment^3 * (1 - success_probability_each_segment))
  (prob_NM / prob_M) = cond_prob_given_3_successful :=
by
  sorry

end probability_both_segments_successful_expected_number_of_successful_segments_probability_given_3_successful_l482_482488


namespace find_other_diagonal_l482_482153

noncomputable def convex_quadrilateral (A B C D : ℝ) (area : ℝ) (sum_of_sides_diagonal : ℝ) (other_diagonal : ℝ) : Prop :=
  A + B + D = sum_of_sides_diagonal ∧ 
  (∃ S S' : ℝ, S + S' = area ∧ S = 1/2 * A * D * sin (0.5) ∧
   S' = 1/2 * C * D * sin (0.5) ∧ 32 ≤ 1/2 * (A + C) * D ∧ 
  A + C = D ∧ 2 * D = sum_of_sides_diagonal → 
  other_diagonal = sqrt 2 * D → other_diagonal = 8 * sqrt 2)

theorem find_other_diagonal :
  ∀ (A B C D other_diagonal : ℝ),
  convex_quadrilateral A B C D 32 (A + B + D) other_diagonal → other_diagonal = 8 * sqrt 2 :=
begin
  intros A B C D other_diagonal h,
  rcases h with ⟨h1, h2, h3, h4, h5, h6, h7⟩,
  sorry
end

end find_other_diagonal_l482_482153


namespace triangle_area_scaled_l482_482578

theorem triangle_area_scaled (a b : ℝ) (θ : ℝ) :
  let A := 1/2 * a * b * Real.sin θ
  let a' := 3 * a
  let b' := 2 * b
  let A' := 1/2 * a' * b' * Real.sin θ
  A' = 6 * A := by
  sorry

end triangle_area_scaled_l482_482578


namespace multiplication_factor_correct_l482_482808

theorem multiplication_factor_correct (N X : ℝ) (h1 : 98 = abs ((N * X - N / 10) / (N * X)) * 100) : X = 5 := by
  sorry

end multiplication_factor_correct_l482_482808


namespace number_of_merchants_l482_482789

theorem number_of_merchants (x : ℕ) (h : 2 * x^3 = 2662) : x = 11 :=
  sorry

end number_of_merchants_l482_482789


namespace minimum_value_l482_482631

noncomputable def min_expression (a b : ℝ) : ℝ :=
  a^2 + b^2 + 1 / (a + b)^2 + 1 / (a^2 * b^2)

theorem minimum_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 + 3 ∧ min_expression a b ≥ c :=
by
  use 2 * Real.sqrt 2 + 3
  sorry

end minimum_value_l482_482631


namespace find_n_l482_482838

/-- Define the sequence a_n where the first term is 1,
    followed by two 2s, then three 3s, then four 4s, and so on. -/
noncomputable def a : ℕ → ℕ
| n := let m := (1 + (Math.sqrt (1 + 8 * n) - 1) / 2) in (m.floor)

/-- Prove that the value of n for which a_{n-1} = 20 and a_n = 21 is 211. -/
theorem find_n (n : ℕ) (hn1 : a (n - 1) = 20) (hn2 : a n = 21) : n = 211 := by
  sorry

end find_n_l482_482838


namespace trig_identity_l482_482502

theorem trig_identity (α : ℝ) (h : Real.tan α = 1/3) :
  Real.cos α ^ 2 + Real.cos (Real.pi / 2 + 2 * α) = 3 / 10 := 
sorry

end trig_identity_l482_482502


namespace arithmetic_sequence_length_l482_482939

theorem arithmetic_sequence_length 
  (a₁ : ℕ) (d : ℤ) (x : ℤ) (n : ℕ) 
  (h_start : a₁ = 20)
  (h_diff : d = -2)
  (h_eq : x = 10)
  (h_term : x = a₁ + (n - 1) * d) :
  n = 6 :=
by
  sorry

end arithmetic_sequence_length_l482_482939


namespace inequality_solution_set_l482_482136

theorem inequality_solution_set 
  (m n : ℤ)
  (h1 : ∀ x : ℤ, mx - n > 0 → x < 1 / 3)
  (h2 : ∀ x : ℤ, (m + n) x < n - m) :
  ∀ x : ℤ, x > -1 / 2 := 
sorry

end inequality_solution_set_l482_482136


namespace number_of_participants_eq_14_l482_482996

theorem number_of_participants_eq_14 (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 :=
by
  sorry

end number_of_participants_eq_14_l482_482996


namespace parallelogram_with_right_angle_is_rectangle_l482_482577

-- Definitions and conditions for the problem
def is_parallelogram (ABC: Type) (A B C D : ABC) : Prop :=
  ∠A + ∠B = 180 ∧ ∠B + ∠C = 180 ∧ ∠C + ∠D = 180 ∧ ∠D + ∠A = 180 ∧
  ∠A = ∠C ∧ ∠B = ∠D

def is_rectangle (ABC: Type) (A B C D : ABC) : Prop :=
  is_parallelogram ABC A B C D ∧ ∠A = 90 ∧ ∠B = 90 ∧ ∠C = 90 ∧ ∠D = 90

theorem parallelogram_with_right_angle_is_rectangle
  {ABC : Type} {A B C D : ABC} (h_parallelogram : is_parallelogram ABC A B C D) (h_right_angle : ∠A = 90) :
  is_rectangle ABC A B C D :=
by sorry

end parallelogram_with_right_angle_is_rectangle_l482_482577


namespace monotone_f_iff_l482_482521

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if h : x < 1 then a^x
  else x^2 + 4 / x + a * Real.log x

theorem monotone_f_iff (a : ℝ) :
  (∀ x₁ x₂, x₁ ≤ x₂ → f a x₁ ≤ f a x₂) ↔ 2 ≤ a ∧ a ≤ 5 :=
by
  sorry

end monotone_f_iff_l482_482521


namespace towels_given_l482_482776

theorem towels_given (green_towels white_towels remaining_towels : ℕ) 
  (h_green : green_towels = 40) 
  (h_white : white_towels = 44) 
  (h_remaining : remaining_towels = 19) : 
  (green_towels + white_towels - remaining_towels = 65) :=
by
  rw [h_green, h_white, h_remaining]
  exact 65

#lint -- Expect no linting issues

end towels_given_l482_482776


namespace curve_C2_equation_distance_AB_eq_one_l482_482161

-- Definition of curve C1
def curve_C1 (α : ℝ) (h : 0 < α ∧ α < π) : ℝ × ℝ := (2 + 2 * cos α, 2 * sin α)

-- Definition of curve C2
def curve_C2 (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 1 ∧ 0 < y ∧ y ≤ 1

-- Theorem Ⅰ: Prove the standard equation of C2
theorem curve_C2_equation {α : ℝ} (hα : 0 < α ∧ α < π) :
  ∃ (x y : ℝ), curve_C1 α hα = (1 + cos α, sin α) → curve_C2 x y :=
sorry

-- Definition of polar coordinates for C1
def polar_curve_C1 (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) : ℝ := 4 * cos θ

-- Definition of polar coordinates for C2
def polar_curve_C2 (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) : ℝ := 2 * cos θ

-- Coordinates of A on C1
def A (h : θ = π / 3) : ℝ × ℝ := (2, π / 3)

-- Coordinates of B on C2
def B (h : θ = π / 3) : ℝ × ℝ := (1, π / 3)

-- Theorem Ⅱ: Prove the distance |AB| is 1
theorem distance_AB_eq_one {A B : ℝ × ℝ} : |(A.1 - B.1) * (A.2 - B.2)| = 1 :=
sorry

end curve_C2_equation_distance_AB_eq_one_l482_482161


namespace g_20_minus_g_3_l482_482634

noncomputable def g : ℝ → ℝ := sorry  -- This will be defined as a linear function in the actual proof

-- Given conditions
axiom lin_cond : ∀ x y, g(x) - g(y) = (x - y) * 3
axiom cond1 : g(8) - g(3) = 15

-- Goal
theorem g_20_minus_g_3 : g(20) - g(3) = 51 :=
by
  have slope : ∀ x y, (g(x) - g(y)) / (x - y) = 3,
  from λ x y, by rw [lin_cond x y, sub_eq_mul_div],
  have h1 : g(20) - g(3) = 3 * (20 - 3),
  from slope 20 3,
  rw [sub_comm, sub_self, mul_zero, cond1] at h1,
  rw [sub_eq_zero, ite.not] at h1,
  exact h1

end g_20_minus_g_3_l482_482634


namespace area_overlap_of_triangles_l482_482728

structure Point where
  x : ℝ
  y : ℝ

def Triangle (p1 p2 p3 : Point) : Set Point :=
  { q | ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ (a * p1.x + b * p2.x + c * p3.x = q.x) ∧ (a * p1.y + b * p2.y + c * p3.y = q.y) }

def area_of_overlap (t1 t2 : Set Point) : ℝ :=
  -- Assume we have a function that calculates the overlap area
  sorry

def point1 : Point := ⟨0, 2⟩
def point2 : Point := ⟨2, 1⟩
def point3 : Point := ⟨0, 0⟩
def point4 : Point := ⟨2, 2⟩
def point5 : Point := ⟨0, 1⟩
def point6 : Point := ⟨2, 0⟩

def triangle1 : Set Point := Triangle point1 point2 point3
def triangle2 : Set Point := Triangle point4 point5 point6

theorem area_overlap_of_triangles :
  area_of_overlap triangle1 triangle2 = 1 :=
by
  -- Proof goes here, replacing sorry with actual proof steps
  sorry

end area_overlap_of_triangles_l482_482728


namespace haley_lives_l482_482553

theorem haley_lives : ∀ (initial_lives lost_lives gained_lives : ℕ), initial_lives = 14 → lost_lives = 4 → gained_lives = 36 → (initial_lives - lost_lives + gained_lives) = 46 :=
by
  intros initial_lives lost_lives gained_lives h1 h2 h3
  rw [h1, h2, h3]
  sorry

end haley_lives_l482_482553


namespace find_value_of_expression_l482_482519

noncomputable def roots_g : Set ℂ := { x | x^2 - 3*x - 2 = 0 }

theorem find_value_of_expression:
  ∀ γ δ : ℂ, γ ∈ roots_g → δ ∈ roots_g →
  (γ + δ = 3) → (7 * γ^4 + 10 * δ^3 = 1363) :=
by
  intros γ δ hγ hδ hsum
  -- Proof skipped
  sorry

end find_value_of_expression_l482_482519


namespace part1_part2i_part2ii_l482_482633

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1) * Real.exp x

theorem part1 (x : ℝ) (hx : 0 < x) : (f x (-3)).derivative > 0 := sorry

theorem part2i (a : ℝ) (ha : 0 < a ∧ a < 1 / Real.exp 1) : 
  ∃! x0 : ℝ, 0 < x0 ∧ f x0 a = 0 :=
sorry

theorem part2ii (a : ℝ) (x0 x1 : ℝ) (ha : 0 < a ∧ a < 1 / Real.exp 1) 
 (hx0 : 0 < x0 ∧ f x0 a = 0) (hx1 : f x1 a = 0 ∧ x1 > x0) : 
  3 * x0 - x1 > 2 :=
sorry

end part1_part2i_part2ii_l482_482633


namespace angle_ratio_l482_482614

-- Define the angles and their properties
variables (A B C P Q M : Type)
variables (mABQ mMBQ mPBQ : ℝ)

-- Define the conditions from the problem:
-- 1. BP and BQ bisect ∠ABC
-- 2. BM bisects ∠PBQ
def conditions (h1 : 2 * mPBQ = mABQ)
               (h2 : 2 * mMBQ = mPBQ) : Prop :=
  true

-- Translate the question and correct answer into a Lean definition.
def find_ratio (h1 : 2 * mPBQ = mABQ) 
               (h2 : 2 * mMBQ = mPBQ) : Prop :=
  mMBQ / mABQ = 1 / 4

-- Now define the theorem that encapsulates the problem statement
theorem angle_ratio (h1 : 2 * mPBQ = mABQ) 
                    (h2 : 2 * mMBQ = mPBQ) :
  find_ratio A B C P Q M mABQ mMBQ mPBQ h1 h2 :=
by
  -- Proof to be provided.
  sorry

end angle_ratio_l482_482614


namespace compare_a_b_c_l482_482922

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (a := 1 / 2 * f(1 / 2))
variable (b := -2 * f(-2))
variable (c := -real.log 2 * f(real.log (1 / 2)))

-- Assertions about the function f
axiom odd_function : ∀ x : ℝ, f(-x) = -f(x)
axiom second_deriv_positivity : ∀ x : ℝ, x ≠ 0 → f'' x + f x / x > 0

-- Goal: to prove the relationship among a, b, and c
theorem compare_a_b_c : b > c ∧ c > a :=
by
  -- Assertions, properties, and steps skipped (since proofs are not required)
  sorry

end compare_a_b_c_l482_482922


namespace sum_distinct_prime_factors_of_7pow7_minus_7pow4_l482_482454

noncomputable def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  let factors := (Nat.factors n).erase_dup
  factors.sum

theorem sum_distinct_prime_factors_of_7pow7_minus_7pow4 :
  sum_of_distinct_prime_factors (7 ^ 7 - 7 ^ 4) = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7pow7_minus_7pow4_l482_482454


namespace distinct_arrangements_round_table_l482_482160

theorem distinct_arrangements_round_table (n : ℕ) (h : n = 7) : 
  nat.factorial (n - 1) = 720 :=
by 
  rw h
  have : nat.factorial 6 = 720 := rfl
  exact this

end distinct_arrangements_round_table_l482_482160


namespace probability_of_picking_combination_is_0_4_l482_482387

noncomputable def probability_at_least_19_rubles (total_coins total_value: ℕ) :=
  let coins := [10, 10, 5, 5, 2] in
  let all_combinations := (Finset.powersetLen 3 (coins.to_finset)).to_list in
  let favorable_combinations := all_combinations.filter (fun c => c.sum ≥ total_value) in
  (favorable_combinations.length : ℚ) / (all_combinations.length : ℚ)

theorem probability_of_picking_combination_is_0_4 :
  probability_at_least_19_rubles 5 19 = 0.4 :=
by
  sorry

end probability_of_picking_combination_is_0_4_l482_482387


namespace number_of_n_for_perfect_square_l482_482901

theorem number_of_n_for_perfect_square :
  {n : ℤ | 0 ≤ n ∧ n < 24 ∧ ∃ k : ℤ, n / (24 - n) = k^2}.to_finset.card = 4 :=
by
  sorry

end number_of_n_for_perfect_square_l482_482901


namespace correct_operations_result_l482_482236

theorem correct_operations_result {n : ℕ} (h₁ : n / 8 - 20 = 12) :
  (n * 8 + 20) = 2068 ∧ 1800 < 2068 ∧ 2068 < 2200 :=
by
  sorry

end correct_operations_result_l482_482236


namespace number_of_two_digit_factors_of_3_pow_18_minus_1_l482_482114

theorem number_of_two_digit_factors_of_3_pow_18_minus_1 :
  let n := 3^18 - 1,
  let factors := [28, 26, 91],
  ∀ k ∈ factors, 10 ≤ k ∧ k < 100 →
  ∃ s : Finset ℕ, s = {k | k ∣ n ∧ 10 ≤ k ∧ k < 100} ∧ s.card = 3 :=
by
  let n := 3^18 - 1
  let factors := [28, 26, 91]
  have two_digit_factors : ∀ k ∈ factors, 10 ≤ k ∧ k < 100, from sorry
  use factors.to_finset
  have hn : factors.to_finset.card = 3, from sorry
  split
  { intro k,
    split
    { intro hk,
      exact and.intro (by linarith) (by linarith) },
    { intro hnk, linarith } },
  { exact hn }

end number_of_two_digit_factors_of_3_pow_18_minus_1_l482_482114


namespace sum_first_13_terms_l482_482524

namespace ArithmeticSequence
variables {aₙ : ℕ → ℝ}

-- Given conditions
def is_arithmetic_sequence (aₙ : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, aₙ (n + 1) - aₙ n = aₙ 1 - aₙ 0

-- Given specific values
def a₇ := 12

-- Sum of the first 13 terms of the sequence
def S₁₃ (aₙ : ℕ → ℝ) := (13 * (aₙ 0 + aₙ 12)) / 2

axiom a₇_is_12 : aₙ 6 = a₇

-- Prove that given a₇ = 12, the sum of the first 13 terms is 156
theorem sum_first_13_terms (h_arith : is_arithmetic_sequence aₙ) : S₁₃ aₙ = 156 :=
  sorry

end ArithmeticSequence

end sum_first_13_terms_l482_482524


namespace harmonic_sum_inequality_l482_482902

theorem harmonic_sum_inequality (n k : ℕ) (hn : 1 < n) (hk : 1 < k) :
  (∑ i in Finset.range (n ^ k).succ \ Finset.range 2, 1 / (i : ℝ)) >
  k * (∑ i in Finset.range n.succ \ Finset.range 2, 1 / (i : ℝ)) :=
sorry

end harmonic_sum_inequality_l482_482902


namespace find_f_zero_l482_482575

-- Definitions of the polynomial constraints
def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d, f(x) = x^4 + a * x^3 + b * x^2 + c * x + d

def valid_conditions (f : ℝ → ℝ) : Prop :=
  is_monic_quartic f ∧ f(-2) = -4 ∧ f(1) = -1 ∧ f(-4) = -16 ∧ f(5) = -25

-- Main statement
theorem find_f_zero (f : ℝ → ℝ) (h : valid_conditions f) : f(0) = 40 :=
sorry

end find_f_zero_l482_482575


namespace abs_diff_real_imag_eq_one_l482_482427

def w : ℕ → ℂ
| 0 := 1
| 1 := Complex.I
| n+2 := 2 * w (n + 1) + 3 * w n

theorem abs_diff_real_imag_eq_one (n : ℕ) (hn : n ≥ 1) :
  |(w n).re - (w n).im| = 1 :=
sorry

end abs_diff_real_imag_eq_one_l482_482427


namespace largest_percentage_increase_between_2018_2019_l482_482227

def participation : ℕ → ℕ 
| 2015 := 110
| 2016 := 125
| 2017 := 130
| 2018 := 140
| 2019 := 160
| 2020 := 165
| _ := 0 -- For all other years, assuming 0 participants (not required for this proof)

noncomputable def percentage_increase (p1 p2 : ℕ) := 
  ((p2 - p1).to_rat / p1.to_rat) * 100

theorem largest_percentage_increase_between_2018_2019 :
  ∀ n m, (n, m) ∈ [(2015, 2016), (2016, 2017), (2017, 2018), (2018, 2019), (2019, 2020)] → 
  percentage_increase (participation n) (participation m) ≤ percentage_increase (participation 2018) (participation 2019) :=
by
  sorry -- proof goes here

end largest_percentage_increase_between_2018_2019_l482_482227


namespace solve_exponential_equation_l482_482435

theorem solve_exponential_equation :
  {x : ℝ | 2^(2*x) - 6 * 2^x + 8 = 0} = {1, 2} :=
by
  sorry -- Proof is skipped, only the statement is provided

end solve_exponential_equation_l482_482435


namespace number_of_ways_no_consecutive_l482_482057

-- Defining the set of natural numbers from 1 to 8
def numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Predicate to check if a set of numbers has no consecutive numbers
def no_consecutive (s : Finset ℕ) : Prop :=
  ∀ (x ∈ s) (y ∈ s), x < y → y ≠ x + 1

-- The main theorem to prove: there are 300 ways to select 3 different non-consecutive numbers from 1 to 8
theorem number_of_ways_no_consecutive : 
  (Finset.card (numbers.powerset.filter (λ s, s.card = 3 ∧ no_consecutive s))) = 300 := by
  sorry

end number_of_ways_no_consecutive_l482_482057


namespace triangle_area_correct_l482_482756

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let r := 7
    let R := 20
    let cos_rel := 3 * (b / (2 * R)) = 2 * (a / (2 * R)) + c / (2 * R)
    let s := (a + b + c) / 2
  in r * s

theorem triangle_area_correct (a b c : ℝ) (h1 : triangle_area a b c = 7 * (a + c + 2 * sqrt 319) / 2) :
  ∀ (ΔABC : triangle), 
  ΔABC.inradius = 7 ∧ ΔABC.circumradius = 20 ∧ (3 * cos ΔABC.B = 2 * cos ΔABC.A + cos ΔABC.C) →
  ΔABC.area = 7 * (a + c + 2 * sqrt 319) / 2 :=
begin
  sorry
end

end triangle_area_correct_l482_482756


namespace glass_bowls_sold_l482_482810

theorem glass_bowls_sold
  (BowlsBought : ℕ) (CostPricePerBowl SellingPricePerBowl : ℝ) (PercentageGain : ℝ)
  (CostPrice := BowlsBought * CostPricePerBowl)
  (SellingPrice : ℝ := (102 : ℝ) * SellingPricePerBowl)
  (gain := (SellingPrice - CostPrice) / CostPrice * 100) :
  PercentageGain = 8.050847457627118 →
  BowlsBought = 118 →
  CostPricePerBowl = 12 →
  SellingPricePerBowl = 15 →
  PercentageGain = gain →
  102 = 102 := by
  intro h1 h2 h3 h4 h5
  sorry

end glass_bowls_sold_l482_482810


namespace work_completed_together_in_4_days_l482_482778

/-- A can do the work in 6 days. -/
def A_work_rate : ℚ := 1 / 6

/-- B can do the work in 12 days. -/
def B_work_rate : ℚ := 1 / 12

/-- Combined work rate of A and B working together. -/
def combined_work_rate : ℚ := A_work_rate + B_work_rate

/-- Number of days for A and B to complete the work together. -/
def days_to_complete : ℚ := 1 / combined_work_rate

theorem work_completed_together_in_4_days : days_to_complete = 4 := by
  sorry

end work_completed_together_in_4_days_l482_482778


namespace tenth_graders_science_only_l482_482842

theorem tenth_graders_science_only (total_students science_students art_students : ℕ) 
  (h1 : total_students = 140) 
  (h2 : science_students = 100) 
  (h3 : art_students = 75) : 
  (science_students - (science_students + art_students - total_students)) = 65 :=
by
  sorry

end tenth_graders_science_only_l482_482842


namespace seq_sum_correct_l482_482851

noncomputable def seq_term (n : ℕ) : ℝ :=
  n * (1 - 1 / n)

noncomputable def sum_seq : ℝ :=
  ∑ n in finset.range (11 - 3 + 1), seq_term (n + 3)

theorem seq_sum_correct : sum_seq = 54 := by
  -- Calculation based on the problem steps leads to sum_seq = 54
  sorry

end seq_sum_correct_l482_482851


namespace find_ratio_l482_482065

open Real

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (q : ℝ)

-- The geometric sequence conditions
def geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q

-- Sum of the first n terms for the geometric sequence
def sum_of_first_n_terms := ∀ n : ℕ, S n = (a 0) * (1 - q ^ n) / (1 - q)

-- Given conditions
def given_conditions :=
  a 0 + a 2 = 5 / 2 ∧
  a 1 + a 3 = 5 / 4

-- The goal to prove
theorem find_ratio (geo_seq : geometric_sequence a q) (sum_terms : sum_of_first_n_terms a S q) (cond : given_conditions a) :
  S 4 / a 4 = 31 :=
  sorry

end find_ratio_l482_482065


namespace price_per_pound_of_peanuts_l482_482855

-- Definitions based on conditions in the problem
def price_per_pound_of_cashews : ℝ := 5.00
def total_weight_mixture : ℝ := 25
def total_price_mixture : ℝ := 92
def weight_cashews : ℝ := 11

-- Define the main theorem to prove the price per pound of peanuts
theorem price_per_pound_of_peanuts : 
  let remaining_cost := total_price_mixture - (weight_cashews * price_per_pound_of_cashews),
  let weight_peanuts := total_weight_mixture - weight_cashews,
  remaining_cost / weight_peanuts = 2.64 :=
by
  sorry

end price_per_pound_of_peanuts_l482_482855


namespace bisecting_line_of_circle_l482_482266

theorem bisecting_line_of_circle : 
  (∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y + 1 = 0 → x - y + 1 = 0) := 
sorry

end bisecting_line_of_circle_l482_482266


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482462

def seven_pow_seven_minus_seven_pow_four : ℤ := 7^7 - 7^4
def prime_factors_of_three_hundred_forty_two : List ℤ := [2, 3, 19]

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let distinct_prime_factors := prime_factors_of_three_hundred_forty_two.head!
  + prime_factors_of_three_hundred_forty_two.tail!.head!
  + prime_factors_of_three_hundred_forty_two.tail!.tail!.head!
  seven_pow_seven_minus_seven_pow_four = 7^4 * (7^3 - 1) ∧
  7^3 - 1 = 342 ∧
  prime_factors_of_three_hundred_forty_two = [2, 3, 19] ∧
  distinct_prime_factors = 24 := 
sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482462


namespace sole_mart_meals_l482_482013

theorem sole_mart_meals (c_c_meals : ℕ) (meals_given_away : ℕ) (meals_left : ℕ)
  (h1 : c_c_meals = 113) (h2 : meals_givenAway = 85) (h3 : meals_left = 78)  :
  ∃ m : ℕ, m + c_c_meals = meals_givenAway + meals_left ∧ m = 50 := 
by
  sorry

end sole_mart_meals_l482_482013


namespace number_of_diagonals_is_correct_sum_of_interior_angles_is_correct_l482_482026

-- Definition for the number of sides in the polygon
def n : ℕ := 150

-- Definition of the formula for the number of diagonals
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Definition of the formula for the sum of interior angles
def sum_of_interior_angles (n : ℕ) : ℕ :=
  180 * (n - 2)

-- Theorem statements to be proved
theorem number_of_diagonals_is_correct : number_of_diagonals n = 11025 := sorry

theorem sum_of_interior_angles_is_correct : sum_of_interior_angles n = 26640 := sorry

end number_of_diagonals_is_correct_sum_of_interior_angles_is_correct_l482_482026


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482460

def seven_pow_seven_minus_seven_pow_four : ℤ := 7^7 - 7^4
def prime_factors_of_three_hundred_forty_two : List ℤ := [2, 3, 19]

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let distinct_prime_factors := prime_factors_of_three_hundred_forty_two.head!
  + prime_factors_of_three_hundred_forty_two.tail!.head!
  + prime_factors_of_three_hundred_forty_two.tail!.tail!.head!
  seven_pow_seven_minus_seven_pow_four = 7^4 * (7^3 - 1) ∧
  7^3 - 1 = 342 ∧
  prime_factors_of_three_hundred_forty_two = [2, 3, 19] ∧
  distinct_prime_factors = 24 := 
sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482460


namespace sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l482_482477

theorem sum_distinct_prime_factors_of_7_to_7_minus_7_to_4 : 
  let pfs := primeFactors (7 ^ 7 - 7 ^ 4)
  in (pfs = {2, 3, 19}) → sum pfs = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l482_482477


namespace red_ball_expectation_variance_l482_482343

noncomputable def red_ball_problem (n : ℕ) (p : ℝ) := sorry

theorem red_ball_expectation_variance :
  let ξ_1 := binomial 2 (1/3)
  let ξ_2 := { 0 := 1/3, 1 := 2/3 }
  E ξ_1 = E ξ_2 ∧ D ξ_1 > D ξ_2 := by
  sorry

end red_ball_expectation_variance_l482_482343


namespace power_function_const_coeff_l482_482527

theorem power_function_const_coeff (m : ℝ) (h1 : m^2 + 2 * m - 2 = 1) (h2 : m ≠ 1) : m = -3 :=
  sorry

end power_function_const_coeff_l482_482527


namespace cube_identity_l482_482980

variable {x : ℝ}

theorem cube_identity (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 :=
by
  sorry

end cube_identity_l482_482980


namespace new_person_weight_l482_482782

theorem new_person_weight (avg_increase : ℝ) (num_persons : ℕ) (old_weight : ℝ) : 
    avg_increase = 2.5 ∧ num_persons = 8 ∧ old_weight = 65 → 
    (old_weight + num_persons * avg_increase = 85) :=
by
  intro h
  sorry

end new_person_weight_l482_482782


namespace intersection_complement_l482_482223

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable [DecidableEq ℕ] [DecidablePred (∈ U)]

def universal_set := {1, 2, 3, 4, 5, 6}
def set_A := {1, 2}
def set_B := {2, 3, 4}

-- Prove that A ∩ (complement of B in U) = {1}
theorem intersection_complement (A B U : Set ℕ) (hU : U = universal_set) (hA : A = set_A) (hB : B = set_B) :
  A ∩ (U \ B) = {1} :=
by 
  rw [hU, hA, hB]
  -- skipping proof steps
  sorry

end intersection_complement_l482_482223


namespace range_of_f_l482_482897

noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / (x - 5)

theorem range_of_f : set.range f = set.Ioo (-∞ : ℝ) 3 ∪ set.Ioo 3 ∞ :=
by sorry

end range_of_f_l482_482897


namespace abs_triangle_inequality_l482_482129

theorem abs_triangle_inequality {a : ℝ} (h : ∀ x : ℝ, |x - 3| + |x + 1| > a) : a < 4 :=
sorry

end abs_triangle_inequality_l482_482129


namespace indistinguishable_distributions_l482_482974

def ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if boxes = 2 && balls = 6 then 4 else 0

theorem indistinguishable_distributions : ways_to_distribute_balls 6 2 = 4 :=
by sorry

end indistinguishable_distributions_l482_482974


namespace complex_number_in_second_quadrant_l482_482581

-- Define the condition for second quadrant
def is_second_quadrant (a b : ℝ) : Prop := a < 0 ∧ b > 0

-- The main statement we need to prove:
theorem complex_number_in_second_quadrant (a b : ℝ) (h : a + b * I ∈ {z : ℂ | z.re < 0 ∧ 0 < z.im}) :
  is_second_quadrant a b :=
by
  sorry

end complex_number_in_second_quadrant_l482_482581


namespace probability_is_0_4_l482_482392

def coin_values : List ℕ := [10, 10, 5, 5, 2]

def valid_combination (comb : List ℕ) : Prop :=
  comb.sum ≥ 19

def favorable_outcomes : Finset (Finset ℕ) :=
  {s ∈ coin_values.to_finset.powerset.filter (λ s, s.card = 3) | valid_combination s.val.to_list}

def total_outcomes : Finset (Finset ℕ) :=
  coin_values.to_finset.powerset.filter (λ s, s.card = 3)

def probability : ℚ :=
  favorable_outcomes.card / total_outcomes.card

theorem probability_is_0_4 : probability = 2 / 5 :=
by
  -- Proof will go here
  sorry

end probability_is_0_4_l482_482392


namespace power_function_value_at_9_l482_482545

theorem power_function_value_at_9 :
  (∃ a : ℝ, 3^a = (sqrt 3)/3) →
  ∃ f : ℝ → ℝ, (∀ x, f x = x^(-1/2)) ∧ (f 9 = 1/3) :=
by
  sorry

end power_function_value_at_9_l482_482545


namespace min_value_expr_l482_482062

/-- Given x > y > 0 and x^2 - y^2 = 1, we need to prove that the minimum value of 2x^2 + 3y^2 - 4xy is 1. -/
theorem min_value_expr {x y : ℝ} (h1 : x > y) (h2 : y > 0) (h3 : x^2 - y^2 = 1) :
  2 * x^2 + 3 * y^2 - 4 * x * y = 1 :=
sorry

end min_value_expr_l482_482062


namespace sum_of_squares_l482_482661

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 110) : x^2 + y^2 = 1380 := 
by sorry

end sum_of_squares_l482_482661


namespace proof_problem_l482_482116

noncomputable def problem_statement (α : ℝ) : Prop :=
  sin (π / 4 - α) = 5 / 13 ∧ 0 < α ∧ α < π / 2 → cos (2 * α) / cos (π / 4 + α) = 24 / 13

theorem proof_problem (α : ℝ) : problem_statement α :=
sorry

end proof_problem_l482_482116


namespace quadratic_eq_two_distinct_real_roots_isosceles_triangle_value_of_k_l482_482918

/-- Proof that the quadratic equation x^2 - (2k + 1)x + k^2 + k = 0 has two distinct real roots -/
theorem quadratic_eq_two_distinct_real_roots (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
  (x1 * x2 = k^2 + k ∧ x1 + x2 = 2*k + 1) :=
by
  sorry

/-- For triangle ΔABC with sides AB, AC as roots of x^2 - (2k + 1)x + k^2 + k = 0 and BC = 4, find k when ΔABC is isosceles -/
theorem isosceles_triangle_value_of_k (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 * x2 = k^2 + k ∧ x1 + x2 = 2*k + 1 ∧ 
    ((x1 = 4 ∨ x2 = 4) ∧ (x1 + x2 - 4 isosceles))) →
  (k = 3 ∨ k = 4) :=
by
  sorry

end quadratic_eq_two_distinct_real_roots_isosceles_triangle_value_of_k_l482_482918


namespace part1_part2_l482_482946

def f (a x : ℝ) : ℝ := - (2 * a) / x + Real.ln x - 2

theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a 1 = -1 → (derivative (f a) 1 = -1)) → 
  a = -1 :=
by
  intros h
  have h_perp := h 1
  -- condition for the tangent being perpendicular
  sorry

theorem part2 (a : ℝ) : 
  (∀ x : ℝ, 0 < x → f a x > 2 * a) → 
  a < -1 / 2 :=
by
  intros h
  -- analyze the behavior of f(x) to find the range of 'a'
  sorry

end part1_part2_l482_482946


namespace num_values_of_N_divisor_of_48_is_integer_l482_482055

theorem num_values_of_N_divisor_of_48_is_integer :
  {N : ℕ | 0 < N ∧ ∃ k, 48 = k * (N + 3)}.to_finset.card = 7 := 
by sorry

end num_values_of_N_divisor_of_48_is_integer_l482_482055


namespace intervals_of_monotonicity_range_of_k_sum_of_sequence_l482_482540

-- Definition of the function f and its derivative
def f (x : ℝ) := Real.exp x * Real.sin x
def f' (x : ℝ) := Real.sqrt 2 * Real.exp x * Real.sin (x + Real.pi / 4)

-- Statement for Problem 1
theorem intervals_of_monotonicity (k : ℤ) :
  ( ∀ x ∈ Set.Icc (2 * k * Real.pi - Real.pi / 4) (2 * k * Real.pi + 3 * Real.pi / 4), f' x > 0 ) ∧
  ( ∀ x ∈ Set.Icc (2 * k * Real.pi + 3 * Real.pi / 4) (2 * k * Real.pi + 7 * Real.pi / 4), f' x < 0 ) :=
sorry

-- Definition of the function g for Problem 2
def g (x k : ℝ) := f x - k * x
def g' (x k : ℝ) := Real.exp x * (Real.sin x + Real.cos x) - k

-- Statement for Problem 2
theorem range_of_k :
  ( ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ k * x ) ↔ k ≤ 1 :=
sorry

-- Definition of the function F and its derivative for Problem 3
def F (x : ℝ) := f x + Real.exp x * Real.cos x
def F' (x : ℝ) := 2 * Real.exp x * Real.cos x

-- Statement for Problem 3
theorem sum_of_sequence :
  let M := (Real.pi - 1) / 2
  let interval := Set.Icc (-2015 * Real.pi / 2) (2017 * Real.pi / 2)
  let abscissas := { x : ℝ | x ∈ interval ∧ F (M, 0) = 0 }
  ∑ (x_n ∈ abscissas), x_n = 1008 * Real.pi :=
sorry

end intervals_of_monotonicity_range_of_k_sum_of_sequence_l482_482540


namespace hyperbola_line_intersection_range_vector_condition_value_a_l482_482543

-- Problem 1: Establish the range of values for 'a' given the intersection conditions
theorem hyperbola_line_intersection_range (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  ((0 < a ∧ a < 1) ∨ (1 < a ∧ a < Real.sqrt 2)) :=
by {
  sorry
}

-- Problem 2: Verify the value of 'a' given the vector condition PA == (5/12)PB
theorem vector_condition_value_a (a : ℝ) (h₀ : a > 0) (h₁ : ∀ A B P : ℝ × ℝ, 
  let PA := (A.1 - P.1, A.2 - P.2),
      PB := (B.1 - P.1, B.2 - P.2) in
  PA = ((5 : ℝ) / 12) • PB) :
  a = 17 / 13 :=
by {
  sorry
}

end hyperbola_line_intersection_range_vector_condition_value_a_l482_482543


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l482_482482

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31 :
  let n := 7^7 - 7^4 in
  let prime_factors := {2, 3, 7, 19} in
  finset.sum prime_factors id = 31 :=
by
  sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l482_482482


namespace spherical_to_rectangular_conversion_l482_482424

-- Define the spherical to rectangular conversion
def sphericalToRectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

-- Given conditions
def ρ := 10
def θ := 3 * Real.pi / 4
def φ := Real.pi / 4

-- Expected result
def expectedResult : ℝ × ℝ × ℝ :=
  (-5, 5, 5 * Real.sqrt 2)

-- Theorem statement
theorem spherical_to_rectangular_conversion :
  sphericalToRectangular ρ θ φ = expectedResult :=
  by sorry

end spherical_to_rectangular_conversion_l482_482424


namespace jeremy_oranges_l482_482650

theorem jeremy_oranges (M : ℕ) (h : M + 3 * M + 70 = 470) : M = 100 := 
by
  sorry

end jeremy_oranges_l482_482650


namespace probability_greater_difficulty_probability_same_difficulty_l482_482149

/-- A datatype representing the difficulty levels of questions. -/
inductive Difficulty
| easy : Difficulty
| medium : Difficulty
| difficult : Difficulty

/-- A datatype representing the four questions with their difficulties. -/
inductive Question
| A1 : Question
| A2 : Question
| B : Question
| C : Question

/-- The function to get the difficulty of a question. -/
def difficulty (q : Question) : Difficulty :=
  match q with
  | Question.A1 => Difficulty.easy
  | Question.A2 => Difficulty.easy
  | Question.B  => Difficulty.medium
  | Question.C  => Difficulty.difficult

/-- The set of all possible pairings of questions selected by two students A and B. -/
def all_pairs : List (Question × Question) :=
  [ (Question.A1, Question.A1), (Question.A1, Question.A2), (Question.A1, Question.B), (Question.A1, Question.C),
    (Question.A2, Question.A1), (Question.A2, Question.A2), (Question.A2, Question.B), (Question.A2, Question.C),
    (Question.B, Question.A1), (Question.B, Question.A2), (Question.B, Question.B), (Question.B, Question.C),
    (Question.C, Question.A1), (Question.C, Question.A2), (Question.C, Question.B), (Question.C, Question.C) ]

/-- The event that the difficulty of the question selected by student A is greater than that selected by student B. -/
def event_N : List (Question × Question) :=
  [ (Question.B, Question.A1), (Question.B, Question.A2), (Question.C, Question.A1), (Question.C, Question.A2), (Question.C, Question.B) ]

/-- The event that the difficulties of the questions selected by both students are the same. -/
def event_M : List (Question × Question) :=
  [ (Question.A1, Question.A1), (Question.A1, Question.A2), (Question.A2, Question.A1), (Question.A2, Question.A2), 
    (Question.B, Question.B), (Question.C, Question.C) ]

/-- The probabilities of the events. -/
noncomputable def probability_event_N : ℚ := (event_N.length : ℚ) / (all_pairs.length : ℚ)
noncomputable def probability_event_M : ℚ := (event_M.length : ℚ) / (all_pairs.length : ℚ)

/-- The theorem statements -/
theorem probability_greater_difficulty : probability_event_N = 5 / 16 := sorry
theorem probability_same_difficulty : probability_event_M = 3 / 8 := sorry

end probability_greater_difficulty_probability_same_difficulty_l482_482149


namespace find_imaginary_part_of_z_l482_482936

theorem find_imaginary_part_of_z :
  let i : ℂ := complex.I,
  let z : ℂ := (1 + i) / i
  in z.im = -1 :=
by
  sorry

end find_imaginary_part_of_z_l482_482936


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l482_482470

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4 :
  let n := 7^7 - 7^4 in 
  (∑ p in (nat.factors n).to_finset, p) = 31 :=
by sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l482_482470


namespace determine_phi_l482_482583

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

noncomputable def is_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y

theorem determine_phi (φ : ℝ) :
  (∀ x, (2 * sin (2 * x + φ + π / 3)) = -(2 * sin (2 * (-x) + φ + π / 3))) ∧
  (∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ π / 4 → (2 * sin (2 * x + φ + π / 3)) ≥ (2 * sin (2 * y + φ + π / 3)))
  → φ = 2 * π / 3 := by
  sorry

end determine_phi_l482_482583


namespace percent_increase_to_restore_price_l482_482333

variable (p : ℝ)

def first_reduction (p : ℝ) : ℝ := 0.65 * p
def second_reduction (p : ℝ) : ℝ := 0.9 * first_reduction p
def required_percent_increase (p : ℝ) : ℝ := (p / second_reduction p - 1) * 100

theorem percent_increase_to_restore_price :
  required_percent_increase p ≈ 70.94 :=
by
  sorry

end percent_increase_to_restore_price_l482_482333


namespace count_equilateral_triangles_l482_482439

-- Define the hexagonal lattice condition
noncomputable def hexagonal_lattice := sorry

-- Define the condition for lattice points at a distance of 2 units
noncomputable def points_at_two_units (lattice : Type) := sorry

-- Theorem stating the total number of equilateral triangles is 14
theorem count_equilateral_triangles (lattice : Type) [hexagonal_lattice lattice] [points_at_two_units lattice] : 
  ∃ n : ℕ, n = 14 :=
begin
  sorry
end

end count_equilateral_triangles_l482_482439


namespace range_of_t_l482_482948

theorem range_of_t (t : ℝ) :
  (∀ α ∈ Icc (-π/4) (π/3), ∃ β ∈ Iio t, 
    sin (2 * α + π / 6) + sin (2 * β + π / 6) = 0) ↔ t ∈ Ioi (π / 12) :=
sorry

end range_of_t_l482_482948


namespace find_p_q_l482_482875

noncomputable def quadratic_sum_product (p q : ℝ) : Prop :=
  let a := 3 in
  let sum_of_roots := p / a in
  let product_of_roots := q / a in
  sum_of_roots = 4 ∧ product_of_roots = 6

theorem find_p_q : ∃ p q : ℝ, quadratic_sum_product p q ∧ p = 12 ∧ q = 18 := by
  sorry

end find_p_q_l482_482875


namespace solve_system_of_equations_l482_482720

theorem solve_system_of_equations :
  ∃ (x y : ℝ), 
    (x^2 - 9 * y^2 = 0 ∧ x^2 + y^2 = 9 ∧ 
    ((x = 9 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10) ∨ 
     (x = -9 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10) ∨ 
     (x = 9 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10) ∨ 
     (x = -9 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10))) :=
begin
  sorry
end

end solve_system_of_equations_l482_482720


namespace painting_ways_l482_482746

theorem painting_ways :
  ∃ (p : Fin 8 → Prop), (∀ i, p i → (i ≤ 4 ∧ i + 1 ≤ 4 ∧ i + 2 ≤ 4)) →
  (∀ i, ¬p i → (finsupp.card (fun x => ¬p x) = 3 ∧ finsupp.card p = 5)) →
  (p ∈ {f | ∃ i, f i ∧ f (i+1) ∧ f (i+2)}) → 
  (finsupp.card (fun x => p x) = 24) :=
sorry

end painting_ways_l482_482746


namespace correct_propositions_l482_482931

noncomputable theory

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- Define the properties of the function f
axiom f_prop1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y) + 1/2
axiom f_prop2 : f (1/2) = 0
axiom f_prop3 : ∀ x : ℝ, x > 1/2 → f(x) > 0

-- Prove the propositions
theorem correct_propositions :
  f(0) = -1/2 ∧ (∀ x₁ x₂ : ℝ, x₁ < x₂ → f(x₁) < f(x₂)) ∧ (∀ x : ℝ, f(x) + 1/2 = - (f(-x) + 1/2)) :=
by
  sorry

end correct_propositions_l482_482931


namespace coefficient_X4_in_expression_l482_482408

-- Definitions from conditions
def term1 := 5 * (X^3 - X^4)
def term2 := -2 * (3 * X^2 - 2 * X^4 + X^6)
def term3 := 3 * (2 * X^4 - X^10)

-- The expression given by combining the terms
def expression := term1 + term2 + term3

-- The theorem to prove the coefficient of X^4 is 5
theorem coefficient_X4_in_expression :
  coefficient X^4 (expand expression) = 5 :=
sorry

end coefficient_X4_in_expression_l482_482408


namespace quadrant_of_complex_number_l482_482082

theorem quadrant_of_complex_number (z : ℂ) (h : (1 - complex.I) / (z - 2) = 1 + complex.I) : 
  z = 2 - complex.I ∧ (complex.re z > 0 ∧ complex.im z < 0) :=
by
  sorry

end quadrant_of_complex_number_l482_482082


namespace cube_sum_of_edges_corners_faces_eq_26_l482_482624

theorem cube_sum_of_edges_corners_faces_eq_26 :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 :=
by
  let edges := 12
  let corners := 8
  let faces := 6
  sorry

end cube_sum_of_edges_corners_faces_eq_26_l482_482624


namespace cos_alpha_beta_value_l482_482074

theorem cos_alpha_beta_value
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : Real.cos (π / 4 + α) = 1 / 3)
  (h4 : Real.cos (π / 4 - β) = Real.sqrt 3 / 3) :
  Real.cos (α + β) = (5 * Real.sqrt 3) / 9 := 
by
  sorry

end cos_alpha_beta_value_l482_482074


namespace part_one_part_two_l482_482096

noncomputable def f (x a : ℝ) : ℝ :=
  |x + a| + 2 * |x - 1|

theorem part_one (a : ℝ) (h : a = 1) : 
  ∃ x : ℝ, f x 1 = 2 :=
sorry

theorem part_two (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : ∀ x : ℝ, 1 ≤ x → x ≤ 2 → f x a > x^2 - b + 1) : 
  (a + 1 / 2)^2 + (b + 1 / 2)^2 > 2 :=
sorry

end part_one_part_two_l482_482096


namespace solve_equation_l482_482252

theorem solve_equation (x : ℝ) : (x + 4)^2 = 5 * (x + 4) ↔ (x = -4 ∨ x = 1) :=
by sorry

end solve_equation_l482_482252


namespace find_A_B_l482_482487

theorem find_A_B (A B : ℝ) (h : ∀ x : ℝ, x ≠ 5 ∧ x ≠ -2 → 
  (A / (x - 5) + B / (x + 2) = (5 * x - 4) / (x^2 - 3 * x - 10))) :
  A = 3 ∧ B = 2 :=
sorry

end find_A_B_l482_482487


namespace T_lt_one_fourth_l482_482102

-- Define the arithmetic sequence with common difference 2
def arithmetic_seq (n : ℕ) : ℕ := 2 * n + 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 1 / ((arithmetic_seq n)^2 - 1)

-- Define T_n as the sum of the first n terms of sequence b
noncomputable def T (n : ℕ) : ℚ := ∑ i in Finset.range n, b i

-- Prove the inequalities
theorem T_lt_one_fourth (n : ℕ) : T n < 1 / 4 :=
by
  sorry

end T_lt_one_fourth_l482_482102


namespace problem1_problem2_problem3_l482_482792

-- Problem 1
theorem problem1 : sqrt 3 + sqrt 27 - sqrt 12 = 2 * sqrt 3 :=
sorry

-- Problem 2
theorem problem2 : (sqrt 3 + sqrt 2) * (sqrt 3 - sqrt 2) - (sqrt 20 - sqrt 15) / sqrt 5 = sqrt 3 - 1 :=
sorry

-- Problem 3
theorem problem3 (x y : ℝ) (h1 : 2 * (x + 1) - y = 6) (h2 : x = y - 1) : x = 5 ∧ y = 6 :=
sorry

end problem1_problem2_problem3_l482_482792


namespace projection_vector_l482_482273

theorem projection_vector (b : ℝ) (h : (   (
   (-12 : ℝ) * (3 : ℝ) + b * (2 : ℝ)
) / (3 * 3 + 2 * 2)) * (⟨3, 2⟩ : ℝ × ℝ) = (-18 / 13 : ℝ) * (⟨3, 2⟩ : ℝ × ℝ)) : b = 9 :=
by
   sorry

end projection_vector_l482_482273


namespace ellipse_properties_l482_482084

theorem ellipse_properties :
  (∀ a b : ℝ, a > 0 → b > 0 →
    (let E : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ (x^2 / a^2 + y^2 / b^2 = 1)} in
      (2, sqrt 2) ∈ E ∧ (sqrt 6, 1) ∈ E →
      (∃ a b : ℝ, a^2 = 8 ∧ b^2 = 4 ∧ E = {p | ∃ x y, p = (x, y) ∧ (x^2 / 8 + y^2 / 4 = 1)}))) ∧
  (∃ c : ℝ, c > 0 ∧ ∃ F : set (ℝ × ℝ), F = {p | ∃ x y, p = (x, y) ∧ x^2 + y^2 = c} ∧ c = 8 / 3) ∧
  (∀ k m : ℝ, (∃ A B : ℝ × ℝ, let line := λ p : ℝ × ℝ, ∃ x, p = (x, k * x + m) in
    line A ∧ line B ∧
    ∃ t : ℝ, (A.1^2 + (k * A.1 + m)^2 - 1) * (B.1^2 + (k * B.1 + m)^2 - 1) = 0 ∧
    (A.1 * B.1 + (k * A.1 + m) * (k * B.1 + m) = 0) →
    (sqrt 6 / 3) ^ 2 + F = (mul_self (2 * (sqrt 6 / 3)))))) sorry

end ellipse_properties_l482_482084


namespace simplify_radicals_l482_482697

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l482_482697


namespace joan_has_10_books_l482_482185

def toms_books := 38
def together_books := 48
def joans_books := together_books - toms_books

theorem joan_has_10_books : joans_books = 10 :=
by
  -- The proof goes here, but we'll add "sorry" to indicate it's a placeholder.
  sorry

end joan_has_10_books_l482_482185


namespace James_pays_6_dollars_l482_482182

-- Defining the conditions
def packs : ℕ := 4
def stickers_per_pack : ℕ := 30
def cost_per_sticker : ℚ := 0.10
def friend_share : ℚ := 0.5

-- Total number of stickers
def total_stickers : ℕ := packs * stickers_per_pack

-- Total cost calculation
def total_cost : ℚ := total_stickers * cost_per_sticker

-- James' payment calculation
def james_payment : ℚ := total_cost * friend_share

-- Theorem statement to be proven
theorem James_pays_6_dollars : james_payment = 6 := by
  sorry

end James_pays_6_dollars_l482_482182


namespace projection_of_b_onto_a_l482_482111

open Real EuclideanGeometry

def vector := ℝ × ℝ

def dot_product (u v : vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (u : vector) : ℝ :=
  Real.sqrt (u.1 * u.1 + u.2 * u.2)

def projection (u v : vector) : vector :=
  let c := dot_product u v / (u.1 * u.1 + u.2 * u.2)
  in (c * u.1, c * u.2)

noncomputable def a : vector := (1, 2)
noncomputable def b : vector := (0, 3)

theorem projection_of_b_onto_a :
  magnitude (projection a b) = 6 * Real.sqrt 5 / 5 :=
by
  sorry

end projection_of_b_onto_a_l482_482111


namespace problemStatement_l482_482092

-- Given function definition
def f (x m : ℝ) := x^2 - m * x + m - 1

-- Conditions: For all x in [2, 4], f(x, m) >= -1
def condition1 (m : ℝ) := ∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → f x m ≥ -1

-- Define the main theorem
def mainTheorem1 : Prop :=
  ∀ m : ℝ, condition1 m → m ≤ 4

def mainTheorem2 : Prop :=
  ∃ a b : ℤ, a < b ∧ (a + b = m.toInt + 1) ∧ (a * b = m.toInt - 1)

-- Combined theorem statement for range of m and existence of a, b
theorem problemStatement : mainTheorem1 ∧ mainTheorem2 :=
by
  sorry

end problemStatement_l482_482092


namespace exists_alpha_divisible_by_2014_l482_482643

theorem exists_alpha_divisible_by_2014 (a : Fin 11 → ℤ) :
  ∃ (α : Fin 11 → ℤ), (∀ i, α i = -1 ∨ α i = 0 ∨ α i = 1) ∧ (∃ i, α i ≠ 0) ∧ 
    2014 ∣ ∑ i, α i * a i :=
by sorry

end exists_alpha_divisible_by_2014_l482_482643


namespace range_of_m_l482_482956

-- Define the piecewise function f
def f (x : ℝ) (m : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (x + 4) + Real.exp (-x) - m
  else x^2 - 2 * x

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∃ (x₀ : ℝ), f x₀ m = 0 ∧ ∑_positions x₀ = -2) →
  ((2 * Real.exp 2) < m ∧ m ≤ (Real.exp 4) + 1) := sorry

end range_of_m_l482_482956


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l482_482474

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4 :
  let n := 7^7 - 7^4 in 
  (∑ p in (nat.factors n).to_finset, p) = 31 :=
by sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l482_482474


namespace point_in_quadrant_l482_482122

theorem point_in_quadrant (m n : ℝ) (h₁ : 2 * (m - 1)^2 - 7 = -5) (h₂ : n > 3) :
  (m = 0 → 2*m - 3 < 0 ∧ (3*n - m)/2 > 0) ∧ 
  (m = 2 → 2*m - 3 > 0 ∧ (3*n - m)/2 > 0) :=
by 
  sorry

end point_in_quadrant_l482_482122


namespace minimum_value_expression_l482_482871

theorem minimum_value_expression
  (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) :
  ∃ θ₀ : ℝ, θ₀ = (Real.arctan (3/4)) ∧
  (3 * Real.cos θ₀ + 1 / Real.sin θ₀ + 4 * Real.tan θ₀) = 3 * Real.sqrt(6)^(1/3) :=
by
  sorry

end minimum_value_expression_l482_482871


namespace sqrt_sum_simplify_l482_482695

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l482_482695


namespace honey_bees_honey_production_l482_482574

theorem honey_bees_honey_production :
  (n : ℕ) (t : ℕ) (h : ℕ) (m : ℕ),
  n = 50 ∧ t = 50 ∧ m = 1 ∧ (h = n * m) →
  h = 50 :=
by
  -- Sorry to skip the proof
  sorry

end honey_bees_honey_production_l482_482574


namespace cube_surface_area_l482_482747

-- Three given vertices of the cube
def A : (ℝ × ℝ × ℝ) := (5, 7, 15)
def B : (ℝ × ℝ × ℝ) := (6, 3, 6)
def C : (ℝ × ℝ × ℝ) := (9, -2, 14)

def distance (P Q : (ℝ × ℝ × ℝ)) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2 + (Q.3 - P.3)^2)

noncomputable def side_length (d : ℝ) : ℝ := d / real.sqrt 2

-- Prove the side length yields the surface area of 294
theorem cube_surface_area : 
  distance A B = real.sqrt 98 ∧ 
  distance A C = real.sqrt 98 ∧ 
  distance B C = real.sqrt 98 ∧ 
  6 * (side_length (real.sqrt 98))^2 = 294 :=
by
  sorry

end cube_surface_area_l482_482747


namespace indefinite_integral_l482_482005

open Real

theorem indefinite_integral : 
  ∃ C : ℝ, ∫ (x : ℝ) in Ioi 0, (3 * x^3 - x^2 - 12 * x - 2) / (x * (x + 1) * (x - 2)) = 
  (λ x, 3 * x + log (abs x) + 2 * log (abs (x + 1)) - log (abs (x - 2)) + C) := 
sorry

end indefinite_integral_l482_482005


namespace problem1_problem2_l482_482248

-- Proof of Problem 1
theorem problem1 (x y : ℤ) (h1 : x = -2) (h2 : y = -3) : (6 * x - 5 * y + 3 * y - 2 * x) = -2 :=
by
  sorry

-- Proof of Problem 2
theorem problem2 (a : ℚ) (h : a = -1 / 2) : (1 / 4 * (-4 * a^2 + 2 * a - 8) - (1 / 2 * a - 2)) = -1 / 4 :=
by
  sorry

end problem1_problem2_l482_482248


namespace tangent_line_at_p_l482_482981

open Real

-- Definition of points and the circle
def point_p : ℝ × ℝ := (1, 2)

def is_center (c : ℝ × ℝ) : Prop := c = (0, 0)

-- The slope of the line OP where O is the origin and P is the point (1,2)
def slope_op (p : ℝ × ℝ) : ℝ := (p.snd) / (p.fst)

-- Given that point P(1, 2) is on a circle centered at the origin
theorem tangent_line_at_p (c : ℝ × ℝ) (p : ℝ × ℝ) (h1 : is_center c) (h2 : p = point_p) : 
  ∃ m b : ℝ, (m = -1/slope_op p) ∧ (b = p.snd - m * p.fst) ∧ 
  (x y : ℝ → x + 2*y - 5 = 0) :=
by
  sorry

end tangent_line_at_p_l482_482981


namespace new_temperature_l482_482278

-- Define the initial temperature
variable (t : ℝ)

-- Define the temperature drop
def temperature_drop : ℝ := 2

-- State the theorem
theorem new_temperature (t : ℝ) (temperature_drop : ℝ) : t - temperature_drop = t - 2 :=
by
  sorry

end new_temperature_l482_482278


namespace fourth_derivative_at_0_l482_482638

noncomputable def f : ℝ → ℝ := sorry

axiom f_at_0 : f 0 = 1
axiom f_prime_at_0 : deriv f 0 = 2
axiom f_double_prime : ∀ t, deriv (deriv f) t = 4 * deriv f t - 3 * f t + 1

-- We want to prove that the fourth derivative of f at 0 equals 54
theorem fourth_derivative_at_0 : deriv (deriv (deriv (deriv f))) 0 = 54 :=
sorry

end fourth_derivative_at_0_l482_482638


namespace find_divisor_l482_482652

theorem find_divisor (N D k : ℤ) (h1 : N = 5 * D) (h2 : N % 11 = 2) : D = 7 :=
by
  sorry

end find_divisor_l482_482652


namespace angle_ratio_l482_482611

theorem angle_ratio (A B C P Q M : Type) (θ : ℝ)
  (h1 : ∠B P B = ∠A B C / 2)
  (h2 : ∠B Q B = ∠A B C / 2)
  (h3 : ∠M B P = ∠M B Q)
  (h4 : ∠A B C = 2 * θ) :
  ∠M B Q / ∠A B Q = 1 / 3 := by
  sorry

end angle_ratio_l482_482611


namespace path_to_B_is_correct_path_to_V_is_correct_l482_482750

-- Definitions representing the tree and paths:
inductive Move
| left : Move
| right : Move

def Path := List Move

-- Tree data structure
inductive Tree
| leaf : String → Tree
| branch : Tree → Tree → Tree

open Move Tree

-- Example tree for the problem:
def exampleTree : Tree :=
  branch 
    (branch 
      (branch 
        (leaf "A") 
        (leaf "-")) 
      (leaf "-")) 
    (branch 
      (leaf "-") 
      (branch 
        (leaf "-") 
        (leaf "-")))

-- Function to follow a path on a tree
def followPath : Path → Tree → Option String
| [], leaf name      => some name
| (left :: ps), branch l _ => followPath ps l
| (right :: ps), branch _ r => followPath ps r
| _, _ => none

-- Example paths
def pathB : Path := [left, right, left, right]
def pathV : Path := [right, right, left, right, left, left]

-- Problem statements:
-- Prove that the path to 'B' is "лплп"
theorem path_to_B_is_correct : followPath pathB exampleTree = some "B" := sorry

-- After adding 'V', prove that the path to 'V' is correct
def updatedExampleTree : Tree :=
  branch 
    (branch 
      (branch 
        (leaf "A") 
        (leaf "-")) 
      (leaf "-")) 
    (branch 
      (leaf "-") 
      (branch 
        (leaf "-") 
        (branch 
          (branch 
            (leaf "-") 
            (branch 
              (leaf "V") 
              (leaf "-"))) 
          (leaf "-"))))

theorem path_to_V_is_correct : followPath pathV updatedExampleTree = some "V" := sorry

end path_to_B_is_correct_path_to_V_is_correct_l482_482750


namespace pentagon_cannot_cover_ground_completely_l482_482773

def interior_angle (n : ℕ) : ℝ :=
  180 - 360 / n

theorem pentagon_cannot_cover_ground_completely :
  ¬(360 % (180 - 360 / 5).nat_abs = 0) :=
by
  sorry

end pentagon_cannot_cover_ground_completely_l482_482773


namespace count_neither_6_nice_nor_9_nice_less_500_l482_482903

-- Define k-nice property
def k_nice (k N : ℕ) : Prop :=
  ∃ b : ℕ, b > 0 ∧ Nat.totient (b^(2 * k)) = N

-- The theorem we want to prove
theorem count_neither_6_nice_nor_9_nice_less_500 : 
  (finset.range 500).filter (λ n, ¬(k_nice 6 n) ∧ ¬(k_nice 9 n)).card = 443 :=
by 
  sorry


end count_neither_6_nice_nor_9_nice_less_500_l482_482903


namespace integer_pairs_satisfy_equation_l482_482882

theorem integer_pairs_satisfy_equation :
  ∀ (x y : ℤ), (x^2 * y + y^2 = x^3) → (x = 0 ∧ y = 0) ∨ (x = -4 ∧ y = -8) :=
by
  sorry

end integer_pairs_satisfy_equation_l482_482882


namespace circumcircle_pqr_l482_482158

variables {A B C D E F Q R P M : Type*}

-- Assume we have points A, B, C forming an acute triangle
-- Assume D, E, F are the feet of the altitudes from A, B, and C respectively
-- Assume M is the midpoint of BC
-- Assume line through D parallel to EF intersects AC at Q and AB at R
-- Assume EF intersects BC at P

-- The goal is to prove that the circumcircle of triangle PQR passes through M

theorem circumcircle_pqr (h1 : acute_triangle A B C) 
                        (h2 : altitude A B C D) (h3 : altitude B C A E)
                        (h4 : altitude C A B F) (h5 : midpoint B C M)
                        (h6 : parallel_line_through D EF Q)
                        (h7 : intersects AB R)
                        (h8 : intersects BC P) :
  concyclic Q R M P := 
sorry

end circumcircle_pqr_l482_482158


namespace find_function_expression_point_on_function_graph_l482_482175

-- Problem setup
def y_minus_2_is_directly_proportional_to_x (y x : ℝ) : Prop :=
  ∃ k : ℝ, y - 2 = k * x

-- Conditions
def specific_condition : Prop :=
  y_minus_2_is_directly_proportional_to_x 6 1

-- Function expression derivation
theorem find_function_expression : ∃ k, ∀ x, 6 - 2 = k * 1 ∧ ∀ y, y = k * x + 2 :=
sorry

-- Given point P belongs to the function graph
theorem point_on_function_graph (a : ℝ) : (∀ x y, y = 4 * x + 2) → ∃ a, 4 * a + 2 = -1 :=
sorry

end find_function_expression_point_on_function_graph_l482_482175


namespace part_a_part_b_l482_482494

def product_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n.digits 10.foldl (λ acc d, if acc = 0 then d else acc * d) 1

theorem part_a (n : ℕ) (h : n > 0) : n ≥ product_of_digits n :=
  sorry

theorem part_b (n : ℕ) (h : n ^ 2 - 17 * n + 56 = product_of_digits n) : n = 4 :=
  sorry

end part_a_part_b_l482_482494


namespace similar_polygon_area_sum_l482_482422

theorem similar_polygon_area_sum 
  (t1 t2 a1 a2 b : ℝ)
  (h_ratio: t1 / t2 = a1^2 / a2^2)
  (t3 : ℝ := t1 + t2)
  (h_area_eq : t3 = b^2 * a1^2 / a2^2): 
  b = Real.sqrt (a1^2 + a2^2) :=
by
  sorry

end similar_polygon_area_sum_l482_482422


namespace puzzle_permutations_l482_482966

/--
The number of distinct arrangements of the letters in the word "puzzle",
where the letter "z" appears twice, is 360.
-/
theorem puzzle_permutations : 
  ∀ (word : list Char),
  (word = ['p', 'u', 'z', 'z', 'l', 'e']) →
  (Nat.factorial 6) / (Nat.factorial 2) = 360 :=
by
  intros word h_word
  sorry

end puzzle_permutations_l482_482966


namespace knight_tour_impossible_l482_482411

-- Define the chessboard and knight's move properties
def knight_move (start end : ℕ × ℕ) : Prop :=
  let (x1, y1) := start
  let (x2, y2) := end
  (abs (x2 - x1) = 2 ∧ abs (y2 - y1) = 1) ∨ (abs (x2 - x1) = 1 ∧ abs (y2 - y1) = 2)

def is_black (square : ℕ × ℕ) : Bool :=
  let (x, y) := square
  (x + y) % 2 = 0

def knight_tour (start end : ℕ × ℕ) (n : ℕ) : Prop :=
  ∀ (move_list : List (ℕ × ℕ)),
  move_list.head = some start ∧ move_list.reverse.head = some end
  ∧ (List.length move_list = n)
  ∧ (∀ i < n, knight_move (move_list.nth i) (move_list.nth (i + 1))) -- Valid knight moves
  ∧ (List.Nodup move_list) -- Each square visited exactly once

-- Formal problem statement in Lean: Prove that no such knight's tour exists
theorem knight_tour_impossible : ¬ knight_tour (1, 1) (8, 8) 64 :=
by
  sorry

end knight_tour_impossible_l482_482411


namespace coloring_edges_satisfies_inequality_l482_482593

theorem coloring_edges_satisfies_inequality 
  (n : ℕ) 
  (G : Type) [graph G] 
  (connected : ∀ (u v : G), ∃ path : list G, graph.is_path u v path) 
  (vertices_count : @fintype.card G _ = 2 * n) :
  ∃ (color : G → bool), 
  (∃ (k m : ℕ), k - m ≥ n ∧ k = count_multicolored_edges G color ∧ m = count_monochromatic_edges G color) := 
sorry

end coloring_edges_satisfies_inequality_l482_482593


namespace expected_value_is_correct_l482_482361

-- Define the probabilities of heads and tails
def P_H := 2 / 5
def P_T := 3 / 5

-- Define the winnings for heads and the loss for tails
def W_H := 5
def L_T := -4

-- Calculate the expected value
def expected_value := P_H * W_H + P_T * L_T

-- Prove that the expected value is -2/5
theorem expected_value_is_correct : expected_value = -2 / 5 := by
  sorry

end expected_value_is_correct_l482_482361


namespace sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l482_482475

theorem sum_distinct_prime_factors_of_7_to_7_minus_7_to_4 : 
  let pfs := primeFactors (7 ^ 7 - 7 ^ 4)
  in (pfs = {2, 3, 19}) → sum pfs = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l482_482475


namespace area_triangle_BDE_eq_sqrt2_l482_482601

-- Define the isosceles right triangle ABC with AB = BC = 2
variables (A B C D E : Type)
variables [euclidean_geometry : euclidean_space ℝ]
include euclidean_geometry

-- Given conditions
noncomputable def isosceles_right_triangle (A B C : point ℝ) : Prop := 
  (distance A B = 2) ∧ (distance B C = 2) ∧ (angle B = π/2)

noncomputable def circle_tangent_to_legs (A B C D E : point ℝ) : Prop := 
  tangent_to_legs (AB_midpoint B C) (BC_midpoint A C) 
  ∧ (circle_passes_through_points D E)
  ∧ (D E : segment intersects (AC : hypotenuse))

-- Question to be proved
theorem area_triangle_BDE_eq_sqrt2 
  {A B C D E : point ℝ}
  (h1 : isosceles_right_triangle A B C)
  (h2 : circle_tangent_to_legs A B C D E) : 
  triangle_area B D E = sqrt 2 :=
sorry

end area_triangle_BDE_eq_sqrt2_l482_482601


namespace total_savings_l482_482188

theorem total_savings :
  let J := 0.25 in
  let D_J := 24 in
  let L := 0.50 in
  let D_L := 20 in
  let M := 2 * L in
  let D_M := 12 in
  J * D_J + L * D_L + M * D_M = 28.00 :=
by 
  sorry

end total_savings_l482_482188


namespace expected_value_of_X_l482_482041

noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if x ≤ 1 then x^2
  else 1

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if x ≤ 1 then 2 * x
  else 0

theorem expected_value_of_X : ∫ x in 0..1, x * (f x) = 2 / 3 :=
by
  sorry

end expected_value_of_X_l482_482041


namespace arccos_sqrt_two_over_two_l482_482015

theorem arccos_sqrt_two_over_two :
  real.arccos (real.sqrt 2 / 2) = real.pi / 4 :=
sorry

end arccos_sqrt_two_over_two_l482_482015


namespace estimate_proportion_of_households_owning_3_or_more_houses_l482_482988

theorem estimate_proportion_of_households_owning_3_or_more_houses :
  ∀ (total_households ordinary_households high_income_households 
     sampled_ordinary sampled_high_income
     owning_3_or_more_ordinary owning_3_or_more_high_income: ℕ),
    total_households = 100000 →
    ordinary_households = 99000 →
    high_income_households = 1000 →
    sampled_ordinary = 990 →
    sampled_high_income = 100 →
    owning_3_or_more_ordinary = 50 →
    owning_3_or_more_high_income = 70 →
    let estimate_ordinary := (owning_3_or_more_ordinary * ordinary_households / sampled_ordinary) in
    let estimate_high_income := (owning_3_or_more_high_income * high_income_households / sampled_high_income) in
    estimate_ordinary + estimate_high_income = 5700 :=
sorry

end estimate_proportion_of_households_owning_3_or_more_houses_l482_482988


namespace sum_distinct_prime_factors_of_7pow7_minus_7pow4_l482_482451

noncomputable def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  let factors := (Nat.factors n).erase_dup
  factors.sum

theorem sum_distinct_prime_factors_of_7pow7_minus_7pow4 :
  sum_of_distinct_prime_factors (7 ^ 7 - 7 ^ 4) = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7pow7_minus_7pow4_l482_482451


namespace y_intercept_is_2_l482_482822

def y_intercept_of_line (m : ℝ) (x₁ y₁ : ℝ) : ℝ :=
  y₁ - m * x₁

theorem y_intercept_is_2 (slope point_x point_y : ℝ) (h_slope : slope = 2) (h_point_x : point_x = 269) (h_point_y : point_y = 540) :
  y_intercept_of_line slope point_x point_y = 2 :=
by
  rw [h_slope, h_point_x, h_point_y]
  exact eq.refl 2

end y_intercept_is_2_l482_482822


namespace sum_b_n_l482_482920

noncomputable def a_n (n : ℕ) : ℝ :=
(n + 1 : ℝ)  -- Based on arithmetic sequence deduction in the solution

def b_n (n : ℕ) : ℝ :=
a_n (n + 2) - a_n n + (1 / (a_n (n + 2) * a_n n))

def T_n (n : ℕ) : ℝ :=
2 * n + 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3))

theorem sum_b_n (n : ℕ) : 
  (finset.range n).sum b_n = T_n n := 
by sorry

end sum_b_n_l482_482920


namespace ants_proximity_part1_ants_proximity_part2_l482_482784

-- Definition of some preliminary structures
structure Cube (α : Type) :=
(edge_length : α)

-- Ant placement on edges of the cube
axiom ants_on_edges (n : ℕ) (cube : Cube ℝ) : Prop

-- Part 1 Lean statement: 13 ants on edges of cube with edge length 1
theorem ants_proximity_part1 (ants : ℕ) (cube : Cube ℝ)
  (h1 : ants = 13)
  (h2 : cube.edge_length = 1)
  (h3 : ants_on_edges ants cube) :
  ∃ (a₁ a₂ : ℕ), a₁ < a₂ ∧ distance a₁ a₂ ≤ 1 := sorry

-- Part 2 Lean statement: 9 ants on edges of cube with edge length 1
theorem ants_proximity_part2 (ants : ℕ) (cube : Cube ℝ)
  (h1 : ants = 9)
  (h2 : cube.edge_length = 1)
  (h3 : ants_on_edges ants cube) :
  ∃ (a₁ a₂ : ℕ), a₁ < a₂ ∧ distance a₁ a₂ ≤ 1 := sorry

end ants_proximity_part1_ants_proximity_part2_l482_482784


namespace unique_real_root_of_f_eq_zero_f_is_increasing_f_inequality_l482_482508

variable {f : ℝ → ℝ}
variable {x a b c : ℝ}

-- Condition: f is defined for x in (0, +∞) and f is not identically zero
axiom not_id_zero : ∃ x > 0, f x ≠ 0

-- Condition: For any x > 0 and any y, f(x^y) = y * f(x)
axiom f_property : ∀ {x : ℝ} {y : ℝ}, 0 < x → f (x ^ y) = y * f x

-- Question (a): Prove that the equation f(x) = 0 has exactly one real root
theorem unique_real_root_of_f_eq_zero :
  (∃! x > 0, f x = 0) :=
begin
  sorry
end

-- Question (b): Given a > 1 and f(a) > 0, prove that f(x) is increasing on (0, +∞)
theorem f_is_increasing (a : ℝ) (h₁ : a > 1) (h₂ : f a > 0) :
  ∀ {x₁ x₂ : ℝ}, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ :=
begin
  sorry
end

-- Question (c): Given a > b > c > 1 and 2 * b = a + c, prove that f(a)f(c) < [f(b)]^2
theorem f_inequality (a b c : ℝ) (h₁ : a > b) (h₂ : b > c) (h₃ : c > 1) (h₄ : 2 * b = a + c) :
  f a * f c < (f b) ^ 2 :=
begin
  sorry
end

end unique_real_root_of_f_eq_zero_f_is_increasing_f_inequality_l482_482508


namespace count_integers_abs_inequality_l482_482969

theorem count_integers_abs_inequality : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℤ, |(x: ℝ) - 3| ≤ 7.2 ↔ x ∈ {i : ℤ | -4 ≤ i ∧ i ≤ 10} := 
by 
  sorry

end count_integers_abs_inequality_l482_482969


namespace total_savings_l482_482187

theorem total_savings :
  let J := 0.25 in
  let D_J := 24 in
  let L := 0.50 in
  let D_L := 20 in
  let M := 2 * L in
  let D_M := 12 in
  J * D_J + L * D_L + M * D_M = 28.00 :=
by 
  sorry

end total_savings_l482_482187


namespace geometric_sequence_third_term_l482_482404

theorem geometric_sequence_third_term :
  ∀ (a_1 a_5 : ℚ) (r : ℚ), 
    a_1 = 1 / 2 →
    (a_1 * r^4) = a_5 →
    a_5 = 16 →
    (a_1 * r^2) = 2 := 
by
  intros a_1 a_5 r h1 h2 h3
  sorry

end geometric_sequence_third_term_l482_482404


namespace Tom_spends_375_dollars_l482_482290

noncomputable def totalCost (numBricks : ℕ) (halfDiscount : ℚ) (fullPrice : ℚ) : ℚ :=
  let halfBricks := numBricks / 2
  let discountedPrice := fullPrice * halfDiscount
  (halfBricks * discountedPrice) + (halfBricks * fullPrice)

theorem Tom_spends_375_dollars : 
  ∀ (numBricks : ℕ) (halfDiscount fullPrice : ℚ), 
  numBricks = 1000 → halfDiscount = 0.5 → fullPrice = 0.5 → totalCost numBricks halfDiscount fullPrice = 375 := 
by
  intros numBricks halfDiscount fullPrice hnumBricks hhalfDiscount hfullPrice
  rw [hnumBricks, hhalfDiscount, hfullPrice]
  sorry

end Tom_spends_375_dollars_l482_482290


namespace prove_inequality_l482_482094

-- Given conditions
variables {a b : ℝ}
variable {x : ℝ}
variable h : 0 < a
variable k : 0 < b
variable l : ∀ (x : ℝ), (1 ≤ x ∧ x ≤ 2) → (abs(x + a) + 2 * abs(x - 1) > x^2 - b + 1)

-- To prove (a + 1/2)^2 + (b + 1/2)^2 > 2
theorem prove_inequality (h : 0 < a) (k : 0 < b) (l : ∀ (x : ℝ), (1 ≤ x ∧ x ≤ 2) → (abs(x + a) + 2 * abs(x - 1) > x^2 - b + 1)) :
  (a + 1/2)^2 + (b + 1/2)^2 > 2 :=
sorry

end prove_inequality_l482_482094


namespace max_value_le_3_sqrt_2_div_4_l482_482214

noncomputable def max_value_a_sqrt_1_plus_b2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 + b^2 / 2 = 1) : ℝ := 
  a * real.sqrt (1 + b^2)

theorem max_value_le_3_sqrt_2_div_4 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 + b^2 / 2 = 1) :
  max_value_a_sqrt_1_plus_b2 a b h1 h2 h3 ≤ (3 * real.sqrt 2) / 4 :=
sorry

end max_value_le_3_sqrt_2_div_4_l482_482214


namespace sum_last_three_coefficients_l482_482314

theorem sum_last_three_coefficients (a : ℕ) (h : a ≠ 0) :
  let exp := (1 - (1 : ℚ) / a) ^ 7 in
  ∑ i in {5, 6, 7}, coeff (exp.expand) i = 15 :=
sorry

end sum_last_three_coefficients_l482_482314


namespace sphere_radius_and_volume_l482_482526

theorem sphere_radius_and_volume (A : ℝ) (d : ℝ) (π : ℝ) (r : ℝ) (R : ℝ) (V : ℝ) 
  (h_cross_section : A = π) (h_distance : d = 1) (h_radius : r = 1) :
  R = Real.sqrt (r^2 + d^2) ∧ V = (4 / 3) * π * R^3 := 
by
  sorry

end sphere_radius_and_volume_l482_482526


namespace problem_solution_l482_482001

noncomputable def negThreePower25 : Real := (-3) ^ 25
noncomputable def twoPowerExpression : Real := 2 ^ (4^2 + 5^2 - 7^2)
noncomputable def threeCubed : Real := 3^3

theorem problem_solution :
  negThreePower25 + twoPowerExpression + threeCubed = -3^25 + 27 + (1 / 256) :=
by
  -- proof omitted
  sorry

end problem_solution_l482_482001


namespace value_of_k_l482_482119

theorem value_of_k (k : ℤ) : (1/2)^(22) * (1/(81 : ℝ))^k = 1/(18 : ℝ)^(22) → k = 11 :=
by
  sorry

end value_of_k_l482_482119


namespace solve_y_l482_482717

theorem solve_y (y : ℤ) (h : 7 - y = 10) : y = -3 := by
  sorry

end solve_y_l482_482717


namespace train_speed_in_kph_l482_482805

def train_length : ℝ := 230
def platform_length : ℝ := 290
def time : ℝ := 26
def total_distance : ℝ := train_length + platform_length
def speed_mps : ℝ := total_distance / time
def conversion_factor : ℝ := 3.6
def speed_kph : ℝ := speed_mps * conversion_factor

theorem train_speed_in_kph : speed_kph = 72 :=
by
  have h1 : total_distance = 520 := rfl
  have h2 : speed_mps = 20 := by
    calc
      total_distance / time = 520 / 26 := by rw [h1]
      ... = 20 : by norm_num
  show speed_kph = 72 from
    calc
      speed_mps * conversion_factor = 20 * 3.6 := by rw [h2]
      ... = 72 : by norm_num

end train_speed_in_kph_l482_482805


namespace distance_Owlford_Highcastle_l482_482843

open Complex

theorem distance_Owlford_Highcastle :
  let Highcastle := (0 : ℂ)
  let Owlford := (900 + 1200 * I : ℂ)
  dist Highcastle Owlford = 1500 := by
  sorry

end distance_Owlford_Highcastle_l482_482843


namespace sqrt_72_plus_sqrt_32_l482_482710

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l482_482710


namespace matrix_condition_l482_482510

open Matrix

variable {R : Type*} [Field R]

noncomputable def matrix_B := λ (p q r s : R), matrix![
  [p, q],
  [r, s]
]

theorem matrix_condition 
(p q r s : R) (h : (matrix_B p q r s)ᵀ = 2 * (matrix_B p q r s)⁻¹) :
  p^2 + q^2 + r^2 + s^2 = 1 :=
sorry

end matrix_condition_l482_482510


namespace negation_of_p_l482_482132

variable {x : ℝ}

def proposition_p : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

theorem negation_of_p :
  ¬ (∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) :=
sorry

end negation_of_p_l482_482132


namespace modulo_arithmetic_l482_482007

theorem modulo_arithmetic :
  (222 * 15 - 35 * 9 + 2^3) % 18 = 17 :=
by
  sorry

end modulo_arithmetic_l482_482007


namespace position_relationships_two_lines_l482_482270

-- Define the space and the concept of lines positions
universe u
variables {α : Type u} [EuclideanSpace α] -- Assume α to be a Euclidean space

def LinesPosition (l1 l2 : α → Prop) : Prop := 
(intersect l1 l2) ∨ (parallel l1 l2) ∨ (skew l1 l2)

-- Proof statement: possible position relationships between two lines
theorem position_relationships_two_lines (l1 l2 : α → Prop) :
  (∃ P : α, l1 P ∧ l2 P) ∨
  (∀ P Q : α, l1 P → l2 Q → P ≠ Q ∧ distance P Q = constant) ∨ 
  (skew l1 l2) :=
by {
  -- Start proof
  sorry -- Replace this with the actual proof
}

end position_relationships_two_lines_l482_482270


namespace longest_side_of_triangle_l482_482783

theorem longest_side_of_triangle (x : ℕ) (h1 : 5 * x + 6 * x + 7 * x = 720) : 
  7 * (720 / 18) = 280 :=
by
  have x_eq : x = 720 / 18 := sorry
  rw [x_eq]
  norm_num

end longest_side_of_triangle_l482_482783


namespace three_painters_three_rooms_l482_482587

theorem three_painters_three_rooms :
  ∃ h : ℕ, (∀ t : ℕ, t ≥ 0 ∧ 3 * t / h = 3 * (t / h) -> (if 9 * t / h then t / 9 else 3 = 9) 
∧ if 9 * t / h = 27 then h = 3 : 
sorry

end three_painters_three_rooms_l482_482587


namespace sqrt_sum_simplify_l482_482676

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l482_482676


namespace sum_expression_l482_482009

theorem sum_expression : 3 * 501 + 2 * 501 + 4 * 501 + 500 = 5009 := by
  sorry

end sum_expression_l482_482009


namespace cumulative_discount_difference_from_claim_l482_482820

noncomputable def actual_discount (P : ℝ) : ℝ := 0.6375 * P

theorem cumulative_discount (P : ℝ) :
  let P' := 0.75 * P in
  let P'' := 0.85 * P' in
  1 - P'' / P = 0.3625 :=
by
  sorry

theorem difference_from_claim :
  |0.40 - 0.3625| = 0.0375 :=
by
  sorry

end cumulative_discount_difference_from_claim_l482_482820


namespace polynomial_real_root_l482_482883

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^5 + a * x^4 - x^3 + a * x^2 + x + 1 = 0) ↔
  (a ∈ (Set.Iic (-1/2)) ∨ a ∈ (Set.Ici (1/2))) :=
by
  sorry

end polynomial_real_root_l482_482883


namespace simplify_radicals_l482_482701

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l482_482701


namespace combined_perimeter_is_140_l482_482365

-- Definitions of given conditions
def r : ℝ := 14  -- radius of the semicircle
def L : ℝ := 20  -- length of the rectangle
def W : ℝ := 14  -- width of the rectangle

-- Definition of the combined perimeter calculation
def combined_perimeter (r L W : ℝ) : ℝ :=
  let P_rectangle := 2 * (L + W)
  let C_semicircle := π * r + 2 * r
  P_rectangle + C_semicircle

-- Theorem statement to prove the combined perimeter is 140 units
theorem combined_perimeter_is_140 : combined_perimeter r L W ≈ 140 := 
  sorry

end combined_perimeter_is_140_l482_482365


namespace solve_for_y_l482_482716

theorem solve_for_y (y : ℤ) (h : 7 - y = 10) : y = -3 :=
sorry

end solve_for_y_l482_482716


namespace no_extreme_values_f_prime_solution_range_a_l482_482503

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x ^ 2

noncomputable def f_prime (a x : ℝ) : ℝ := Real.exp x - 2 * a * x

noncomputable def g (a : ℝ) : ℝ := 2 * a * (1 - Real.log (2 * a))

noncomputable def F (a x : ℝ) : ℝ := Real.exp x - a * x ^ 2 - x - 1

-- 1. Prove that f'(x) has no extreme values if a ≤ 0.
theorem no_extreme_values_f_prime (a : ℝ) (h : a ≤ 0) : ¬∃ x, has_extreme_value_at (f_prime a) x :=
sorry

-- 2. Prove that if the equation f(x) = x + 1 has only one real solution, then a ∈ (-∞, 0] ∪ {1/2}.
theorem solution_range_a (a : ℝ) (h : ∃! x : ℝ, f a x = x + 1) : a ∈ Set.Iic 0 ∪ {1 / 2} :=
sorry

end no_extreme_values_f_prime_solution_range_a_l482_482503


namespace num_black_circles_in_first_120_circles_l482_482826

theorem num_black_circles_in_first_120_circles : 
  let S := λ n : ℕ, n * (n + 1) / 2 in
  ∃ n : ℕ, S n < 120 ∧ 120 ≤ S (n + 1) := 
by
  sorry

end num_black_circles_in_first_120_circles_l482_482826


namespace min_distance_PQ_l482_482930

-- Point P lies on the line y = 2
def point_on_line (P : ℝ × ℝ) : Prop := P.2 = 2

-- Point Q lies on the circle (x - 1)^2 + y^2 = 1
def point_on_circle (Q : ℝ × ℝ) : Prop := (Q.1 - 1) ^ 2 + Q.2 ^ 2 = 1

-- Define the distance between two points (P and Q)
def dist (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- The Lean proof statement
theorem min_distance_PQ : ∃ (P Q : ℝ × ℝ), point_on_line P ∧ point_on_circle Q ∧ dist P Q = 1 := 
by
  sorry

end min_distance_PQ_l482_482930


namespace next_palindrome_year_after_2002_l482_482833

def is_palindrome_year (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def product_of_digits (n : ℕ) : ℕ :=
  n.toString.foldl (λ acc c, acc * c.toNat) 1

theorem next_palindrome_year_after_2002 (y : ℕ) : 
  is_palindrome_year y ∧ y > 2002 ∧ product_of_digits y > 15 → y = 2222 :=
sorry

end next_palindrome_year_after_2002_l482_482833


namespace normal_distribution_problem_l482_482221
noncomputable def X : MeasureTheory.MeasurableSpace ℝ := sorry

theorem normal_distribution_problem
  (X : ℝ → MeasureTheory.ProbabilityMeasure ℝ)
  (hX : X ~[ℝ] MeasureTheory.ProbabilityMeasure.normal 3 6)
  (hP : ∀ m, MeasureTheory.ProbabilityMeasure.prob (X > m) = MeasureTheory.ProbabilityMeasure.prob (X < m - 2))
  : ∃ m = 4, true :=
begin
  sorry
end

end normal_distribution_problem_l482_482221


namespace david_age_l482_482321

theorem david_age (x : ℕ) (y : ℕ) (h1 : y = x + 7) (h2 : y = 2 * x) : x = 7 :=
by
  sorry

end david_age_l482_482321


namespace cost_of_dinner_l482_482186

theorem cost_of_dinner (x : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (total_cost : ℝ) : 
  tax_rate = 0.09 → tip_rate = 0.18 → total_cost = 36.90 → 
  1.27 * x = 36.90 → x = 29 :=
by
  intros htr htt htc heq
  rw [←heq] at htc
  sorry

end cost_of_dinner_l482_482186


namespace proof_problem_l482_482202

variables (p q r u v w : ℝ)

def condition1 := 17 * u + q * v + r * w = 0
def condition2 := p * u + 29 * v + r * w = 0
def condition3 := p * u + q * v + 56 * w = 0
def condition4 := p ≠ 17
def condition5 := u ≠ 0

theorem proof_problem (hp : condition1 p q r u v w) (hq : condition2 p q r u v w) (hr : condition3 p q r u v w) (h4 : condition4 p q r u v w) (h5 : condition5 p q r u v w):
  (p / (p - 17)) + (q / (q - 29)) + (r / (r - 56)) = 1 := 
sorry

end proof_problem_l482_482202


namespace children_total_savings_l482_482194

theorem children_total_savings :
  let josiah_savings := 0.25 * 24
  let leah_savings := 0.50 * 20
  let megan_savings := (2 * 0.50) * 12
  josiah_savings + leah_savings + megan_savings = 28 := by
{
  -- lean proof goes here
  sorry
}

end children_total_savings_l482_482194


namespace simplify_expression_l482_482977

theorem simplify_expression (x y z : ℝ) : 
  (x + y + z)⁻² * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ * (x + y + z)⁻¹ :=
by
  sorry

end simplify_expression_l482_482977


namespace probability_two_slate_rocks_l482_482281

theorem probability_two_slate_rocks :
  let slate := 12
  let pumice := 16
  let granite := 8
  let total := slate + pumice + granite
  (slate / total) * ((slate - 1) / (total - 1)) = 11 / 105 :=
by 
  sorry

end probability_two_slate_rocks_l482_482281


namespace focus_of_parabola_l482_482887

variable (a : ℝ)

theorem focus_of_parabola (h : a < 0) : (0, 1 / (4 * a)) = (0, real.abs (1 / (4 * a))) :=
by
  sorry

end focus_of_parabola_l482_482887


namespace log_value_of_arithmetic_sequence_l482_482163

theorem log_value_of_arithmetic_sequence :
  ∀ (a_n : ℕ → ℝ), 
  (∀ n, a_n = a_1 + (n - 1) * d) ∧
  (∃ c₁ c₂, f(0) = 0 ∧ is_global_min_on f (set.Icc 0 1) c₁ ∧ 
             is_global_max_on f (set.Icc 0 1) c₂ ∧
             a_1 = c₁ ∧ a_4025 = c₂) →
  ∃ f : ℝ → ℝ, 
    (∀ x, f x = (1/3) * x ^ 3 - 4 * x ^ 2 + 6 * x - 1) ∧ 
    log 2 (a 2013) = 2 :=
begin
  sorry
end

end log_value_of_arithmetic_sequence_l482_482163


namespace value_of_x2_plus_1_div_x2_l482_482566

theorem value_of_x2_plus_1_div_x2 (x : ℝ) (h : 49 = x^6 + 1 / x^6) : x^2 + 1 / x^2 = Real.cbrt 51 := 
by 
  sorry

end value_of_x2_plus_1_div_x2_l482_482566


namespace simplify_radicals_l482_482702

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l482_482702


namespace find_a_plus_c_l482_482934

-- Define the conditions
def condition1 (a : ℤ) : Prop := (sqrt (2*a - 1) = 3) ∨ (sqrt (2*a - 1) = -3)
def c_value : ℤ := int.floor (Real.sqrt 17)

-- Define the theorem
theorem find_a_plus_c (a : ℤ) (c : ℤ) (h₁ : condition1 a) (h₂ : c = c_value) : a + c = 9 :=
by
  sorry

end find_a_plus_c_l482_482934


namespace Jerry_reaches_first_l482_482234

def distance_AB : ℝ := 32
def distance_BD : ℝ := 12
def distance_AC : ℝ := 13
def distance_CD : ℝ := 27
def speed_Jerry : ℝ := 4
def speed_Tom : ℝ := 5
def head_start : ℝ := 5 -- seconds

def time_Jerry : ℝ := (distance_AB + distance_BD) / speed_Jerry
def time_Tom : ℝ := (distance_AC + distance_CD) / speed_Tom
def effective_time_Jerry : ℝ := time_Jerry - head_start

theorem Jerry_reaches_first : effective_time_Jerry < time_Tom := by
  sorry

end Jerry_reaches_first_l482_482234


namespace expected_value_is_correct_l482_482362

noncomputable def expected_value_of_heads : ℝ :=
  let penny := 1 / 2 * 1
  let nickel := 1 / 2 * 5
  let dime := 1 / 2 * 10
  let quarter := 1 / 2 * 25
  let half_dollar := 1 / 2 * 50
  (penny + nickel + dime + quarter + half_dollar : ℝ)

theorem expected_value_is_correct : expected_value_of_heads = 45.5 := by
  sorry

end expected_value_is_correct_l482_482362


namespace convex_quadrilateral_lower_bound_l482_482156

theorem convex_quadrilateral_lower_bound
  (n : ℕ)
  (h_n_gt_four : n > 4)
  (no_three_collinear : ∀ p1 p2 p3 : ℝ × ℝ, 
    (p1 ≠ p2) → (p2 ≠ p3) → (p1 ≠ p3) → ¬ collinear ℝ {p1, p2, p3}) :
  ∃ k, k ≥ binom (n - 3) 2 ∧ (∃ convex_quadrilaterals : set (set (ℝ × ℝ)),
    convex_quadrilaterals ⊆ (powerset_univ n).filter(λ s, s.card = 4 ∧ convex_hull_convex s) ∧
    convex_quadrilaterals.card = k) :=
sorry

end convex_quadrilateral_lower_bound_l482_482156


namespace probability_two_black_balls_probability_black_ball_second_draw_l482_482796

/-- A bag contains 6 black balls and 4 white balls. We take two balls without replacement. -/
def total_balls := 10
def black_balls := 6
def white_balls := 4

/-- Calculation of combinations -/
def comb (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

/-- The probability of drawing 2 black balls according to the conditions. -/
theorem probability_two_black_balls :
  let total_draws := comb total_balls 2 in
  let black_draws := comb black_balls 2 in
  black_draws / total_draws = 1 / 3 :=
  by sorry

/-- The probability of drawing a black ball on the second draw given that a black ball was drawn on the first draw. -/
theorem probability_black_ball_second_draw :
  let total_draws_after_first := total_balls - 1 in
  let black_draws_after_first := black_balls - 1 in
  black_draws_after_first / total_draws_after_first = 5 / 9 :=
  by sorry

end probability_two_black_balls_probability_black_ball_second_draw_l482_482796


namespace office_chair_legs_l482_482159

theorem office_chair_legs :
  ∀ (x : ℕ),
  (80 * x) + (20 * 3) -
  ((40 * 80) / 100 * x) = 300 → x = 5 :=
by
  intros x h
  have h' : 48 * x + 60 = 300, from h
  sorry

end office_chair_legs_l482_482159


namespace triangle_perimeter_range_l482_482529

variable {A C : ℝ}
variable {P : Set ℝ}

-- Condition 1: Length of side AC is 2
def length_ac_eq_two := (AC : ℝ) = 2

-- Condition 2: Equation involving tangents of angles
def tangent_condition := (sqrt 3) * (tan A) * (tan C) = (tan A) + (tan C) + (sqrt 3)

-- Definition of the perimeter range based on the earlier conditions
def perimeter_range := P = Set.union (Set.Ioo 4 (2 + 2 * sqrt 3)) (Set.Icc (2 + 2 * sqrt 3) 6)

-- Main theorem statement
theorem triangle_perimeter_range (h1 : length_ac_eq_two) (h2 : tangent_condition) : 
  perimeter_range :=
sorry

end triangle_perimeter_range_l482_482529


namespace explicit_expression_of_f_l482_482928

open Function

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + x + 1
  else if x = 0 then 0
  else -x^2 + x - 1

variable (x : ℝ)

theorem explicit_expression_of_f :
  (f(x) = x^2 + x + 1 ∧ x < 0) ∨ 
  (f(x) = 0 ∧ x = 0) ∨ 
  (f(x) = -x^2 + x - 1 ∧ x > 0) :=
by
  sorry

end explicit_expression_of_f_l482_482928


namespace color_triangles_l482_482744

-- Definitions based on the conditions
def vertices : ℕ := 54
def points : ℕ := vertices + 1

-- Given conditions
axiom center_and_polygon (p : ℕ) : p = points

-- Proof statement
theorem color_triangles :
  (∃ t : ℕ, t = 72) → (center_and_polygon points) → (t = 72) :=
by
  sorry

end color_triangles_l482_482744


namespace mass_determination_l482_482791

theorem mass_determination : 
  ∃ (a b c : ℕ), 
    a ∈ {1, 2, 3, 4, 5} ∧
    b ∈ {1, 2, 3, 4, 5} ∧
    c ∈ {1, 2, 3, 4, 5} ∧
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
    2 * a > 3 * b ∧
    b > 2 * c ∧
    a = 5 ∧ b = 3 ∧ c = 1 :=
by {
  use [5, 3, 1],
  simp, 
  split, 
  {simp},
  split, 
  {simp},
  split,
  {simp},
  split,
  {simp},
  split,
  {simp},
  split,
  {calc
    2 * 5 = 10 : by norm_num
    3 * 3 = 9 : by norm_num
    10 > 9 : by norm_num,},
  split,
  {calc
    3 > 2 * 1 : by norm_num},
  simp,
  simp,
}

end mass_determination_l482_482791


namespace price_difference_is_correct_l482_482025

-- Definitions from the problem conditions
def list_price : ℝ := 58.80
def tech_shop_discount : ℝ := 12.00
def value_mart_discount_rate : ℝ := 0.20

-- Calculating the sale prices from definitions
def tech_shop_sale_price : ℝ := list_price - tech_shop_discount
def value_mart_sale_price : ℝ := list_price * (1 - value_mart_discount_rate)

-- The proof problem statement
theorem price_difference_is_correct :
  value_mart_sale_price - tech_shop_sale_price = 0.24 :=
by
  sorry

end price_difference_is_correct_l482_482025


namespace purely_imaginary_implies_alpha_condition_l482_482127

theorem purely_imaginary_implies_alpha_condition
  (α : ℝ) (k : ℤ) :
  (∃ (z : ℂ), z = complex.of_real (sin α) - complex.I * complex.of_real (1 - cos α) 
    ∧ (z.im = complex.abs z))
  → (∃ (k : ℤ), α = (2 * k + 1) * π) :=
by
  sorry

end purely_imaginary_implies_alpha_condition_l482_482127


namespace find_prime_n_l482_482929

def is_prime (p : ℕ) : Prop := 
  p > 1 ∧ (∀ n, n ∣ p → n = 1 ∨ n = p)

def prime_candidates : List ℕ := [11, 17, 23, 29, 41, 47, 53, 59, 61, 71, 83, 89]

theorem find_prime_n (n : ℕ) 
  (h1 : n ∈ prime_candidates) 
  (h2 : is_prime (n)) 
  (h3 : is_prime (n + 20180500)) : 
  n = 61 :=
by sorry

end find_prime_n_l482_482929


namespace min_value_of_f_in_D_l482_482447

noncomputable def f (x y : ℝ) : ℝ := 6 * (x^2 + y^2) * (x + y) - 4 * (x^2 + x * y + y^2) - 3 * (x + y) + 5

def D (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem min_value_of_f_in_D : ∃ (x y : ℝ), D x y ∧ f x y = 2 ∧ (∀ (u v : ℝ), D u v → f u v ≥ 2) :=
by
  sorry

end min_value_of_f_in_D_l482_482447


namespace average_age_in_terms_of_m_l482_482994

-- Define the ages based on given conditions
noncomputable def Mary's_age (m : ℕ) := m
noncomputable def John's_age (m : ℕ) := 2 * m
noncomputable def Tonya_age : ℕ := 60
noncomputable def Sam_age : ℕ := Tonya_age - 4
noncomputable def Carol_age (m : ℕ) := 3 * m

-- Define the average age function
noncomputable def average_age (m : ℕ) : ℚ := 
  (Mary's_age m + John's_age m + Tonya_age + Sam_age + Carol_age m) / 5

-- State the theorem
theorem average_age_in_terms_of_m (m : ℕ) : average_age m = (6 * m + 116) / 5 :=
by
  unfold Mary's_age John's_age Tonya_age Sam_age Carol_age average_age
  sorry

end average_age_in_terms_of_m_l482_482994


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l482_482486

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31 :
  let n := 7^7 - 7^4 in
  let prime_factors := {2, 3, 7, 19} in
  finset.sum prime_factors id = 31 :=
by
  sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l482_482486


namespace smaller_odd_number_l482_482048

theorem smaller_odd_number (n : ℤ) (h : n + (n + 2) = 48) : n = 23 :=
by
  sorry

end smaller_odd_number_l482_482048


namespace PropositionA_is_not_axiom_l482_482341

definition PropositionA : Prop := 
  ∀ (P Q R : plane), (P ∥ R ∧ Q ∥ R) → P ∥ Q

definition PropositionB : Prop := 
  ∀ (A B C : point), (¬ collinear A B C) → ∃! (P : plane), {A, B, C} ⊆ P

definition PropositionC : Prop :=
  ∀ (A B : point) (L : line) (P : plane), (A ∈ L ∧ B ∈ L ∧ A ∈ P ∧ B ∈ P) → ∀ (X : point), X ∈ L → X ∈ P

definition PropositionD : Prop := 
  ∀ (P Q : plane) (A : point), (A ∈ P ∧ A ∈ Q) → ∃! (L : line), (A ∈ L) ∧ (L ⊆ P) ∧ (L ⊆ Q)

axiom PropB_is_axiom : PropositionB
axiom PropC_is_axiom : PropositionC
axiom PropD_is_axiom : PropositionD

theorem PropositionA_is_not_axiom : ¬PropositionA := 
by
  -- Using the conditions that PropositionB, PropositionC, and PropositionD are axioms,
  -- we are to prove that PropositionA is not an axiom.
  sorry

end PropositionA_is_not_axiom_l482_482341


namespace fixed_point_value_of_a_unique_minimum_and_range_l482_482061

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x

-- The coordinates of the fixed point
theorem fixed_point {a : ℝ} : f 0 a = 1 :=
by
  unfold f
  simp

-- Prove that the value of a should be 1
theorem value_of_a {a : ℝ} : (∀ x : ℝ, f' x a ≥ -a * x - 1) → a = 1 :=
by
  sorry

-- Prove the existence of a unique minimum point and its value range
theorem unique_minimum_and_range 
  (a : ℝ) (h : a = 1) : ∃ (x₀ : ℝ), is_minimum (λ x, f x a) x₀ ∧ -2 < f x₀ a ∧ f x₀ a < -1/4 :=
by
  sorry

end fixed_point_value_of_a_unique_minimum_and_range_l482_482061


namespace sum_alternate_terms_l482_482620

noncomputable def sequence (n : ℕ) : ℕ :=
if n = 1 then 1 else if n = 2 then 2 else sequence (n - 1) + sequence (n - 2)

theorem sum_alternate_terms (k : ℕ) (h : sequence 2022 = k) :
  (Finset.sum (Finset.range 1010) (λ i, sequence (2*i + 3))) = k - 2 :=
sorry

end sum_alternate_terms_l482_482620


namespace complement_M_N_correct_l482_482549

-- Define the sets M and N
def M := {0, 1, 2, 3, 4, 5} : Set ℕ
def N := {0, 2, 3} : Set ℕ

-- Define the complement function
def complement (M N : Set ℕ) : Set ℕ := { x ∈ M | x ∉ N }

-- State the theorem
theorem complement_M_N_correct : complement M N = {1, 4, 5} :=
by
  sorry

end complement_M_N_correct_l482_482549


namespace area_pentagon_AFDCB_l482_482242
noncomputable theory

-- Define the problem conditions
def AF : ℝ := 12
def DF : ℝ := 9
def AD : ℝ := real.sqrt (AF^2 + DF^2)
def area_square_ABCD : ℝ := AD^2
def area_triangle_AFD : ℝ := 1 / 2 * AF * DF

-- Define what needs to be proven
theorem area_pentagon_AFDCB : 
    area_square_ABCD - area_triangle_AFD = 171 := by 
simp only [AF, DF, AD, area_square_ABCD, area_triangle_AFD, real.sqrt_eq_rpow, real.sqrt, rpow_two, mul_assoc, add_comm ]
  sorry

end area_pentagon_AFDCB_l482_482242


namespace problem_solution_l482_482046

-- Definitions based on the conditions from the problem
def a : ℕ := 2
def n : ℕ := 1999
def p : ℕ := 17

-- Fermat's Little Theorem applied
lemma fermat_little_theorem_mod_17 
  (a : ℕ) (ha : a ≠ 0) (p : ℕ) (hp : Nat.Prime p) (hpa : ¬ p ∣ a) : 
  a^(p - 1) % p = 1 := sorry

-- Given conditions of the problem using fermat_little_theorem_mod_17
theorem problem_solution : (a^(n) + 1) % p = 10 :=
  by
    have h_prime : Nat.Prime p := by norm_num
    have h_non_zero : a ≠ 0 := by norm_num
    have h_not_divisible : ¬ p ∣ a := by norm_num
    have h_fermat := fermat_little_theorem_mod_17 a h_non_zero p h_prime h_not_divisible
    sorry

end problem_solution_l482_482046


namespace trapezoid_ABCD_ratio_l482_482201

variable (AB CD AD BC a x : ℝ)
variable (E H : Point)
variable [AB_parallel_CD : parallel AB CD]
variable [angle_D_90 : angle D = 90]
variable [E_on_CD : E ∈ Line CD]
variable [AE_eq_BE : distance A E = distance B E]
variable [triangles_similar : ∼similar AED CEB]
variable [CD_divided_by_AB : CD / AB = 2014]
variable [BC_AD_eq_sqrt3 : BC / AD = sqrt 3]

theorem trapezoid_ABCD_ratio {AB CD AD BC : ℝ} 
  (h_parallel : parallel AB CD)
  (h_D_90 : angle D = 90)
  (h_E_on_CD : E ∈ Line CD)
  (h_AE_eq_BE : distance A E = distance B E)
  (h_triangles_similar : similar AED CEB ∧ ¬(congruent AED CEB))
  (h_CD_by_AB : CD / AB = 2014) :
  BC / AD = sqrt 3 := 
by
  sorry

end trapezoid_ABCD_ratio_l482_482201


namespace maximum_abs_value_of_z4_l482_482649

noncomputable def z1 := -1 - Complex.i
noncomputable def z2 := -2 * Complex.i
noncomputable def z3 := Real.sqrt 3 - Complex.i
noncomputable def z4 := 1 - 2 * Complex.i
noncomputable def z5 := (3 : ℂ)

theorem maximum_abs_value_of_z4 :
  ∀ z ∈ {z1, z2, z3, z4, z5}, abs (z^4) ≤ abs (z5^4) := by
  sorry

end maximum_abs_value_of_z4_l482_482649


namespace number_of_male_alligators_l482_482199

def total_alligators (female_alligators : ℕ) : ℕ := 2 * female_alligators
def female_juveniles (female_alligators : ℕ) : ℕ := 2 * female_alligators / 5 
def adult_females (female_alligators : ℕ) : ℕ := (3 * female_alligators) / 5

def male_alligators (total_alligators : ℕ) := total_alligators / 2

theorem number_of_male_alligators
    (half_male : ∀ total, male_alligators total = total / 2) 
    (adult_female_count : ∀ female, adult_females female = 15) 
    (female_count : ∃ female, adult_females female = 15) :
  (2 * classical.some female_count) / 2 = 25 :=
by
  have female_count := classical.some female_count
  rw [(adult_female_count female_count)]
  have total := 2 * female_count
  rw [← half_male total]
  sorry

end number_of_male_alligators_l482_482199


namespace option_B_correct_l482_482523

def is_same_function (f g : ℝ → ℝ) (domain : set ℝ) : Prop :=
  ∀ x ∈ domain, f x = g x

def f_B (x : ℝ) : ℝ := (sqrt x)^2 / x
def g_B (x : ℝ) : ℝ := x / (sqrt x)^2

theorem option_B_correct : is_same_function f_B g_B {x : ℝ | x > 0} := sorry

end option_B_correct_l482_482523


namespace math_problem_l482_482571

theorem math_problem (x y : ℕ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end math_problem_l482_482571


namespace exists_trapezoid_in_selected_vertices_l482_482244

theorem exists_trapezoid_in_selected_vertices :
  ∀ (n : ℕ) (k : ℕ), (even n) ∧ (3 ≤ k) →
  ∃ (v : Fin k → Fin n) (e : ∀ i j : Fin k, i ≠ j → ∃ (p : Fin n) (q : Fin n), v i = p ∧ v j = q),
    ∃ (A B C D : Fin n), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
      v A = v B ∧ v C = v D →
        ∃ (i j m l : Fin k),
        (i ≠ j ∧ j ≠ m ∧ m ≠ l ∧ l ≠ i ∧ i ≠ m ∧ j ≠ l) ∧
        (v i = v j) ∧ (v m = v l) :=
sorry

end exists_trapezoid_in_selected_vertices_l482_482244


namespace math_problem_l482_482544

noncomputable def problem_statement : Prop :=
  let line_l := { x | ∃ t : ℝ, x = (t, (1 / 2 * 2.sqrt) + t * 3.sqrt) }
  let curve_C := { (x, y) | (x - 1 / 2 * 2.sqrt)^2 + (y - 1 / 2 * 2.sqrt)^2 = 1 }
  let P := (0, (1 / 2 * 2.sqrt))
  ∃ A B ∈ curve_C, A ∈ line_l ∧ B ∈ line_l ∧ |P - A| + |P - B| = (5.sqrt) / 2

theorem math_problem : problem_statement :=
sorry

end math_problem_l482_482544


namespace sqrt_sum_simplify_l482_482669

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l482_482669


namespace smallest_positive_period_monotonic_intervals_find_zeros_l482_482949

noncomputable def f (x : ℝ) : ℝ :=
  √3 * Real.sin (2 * x - π / 6) + 2 * (Real.sin (x - π / 12))^2

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ ε > 0, ε < T → ∃ x, f (x + ε) ≠ f x) := sorry

theorem monotonic_intervals :
  ∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π / 12) (k * π + 5 * π / 12), (f x)' > 0 := sorry

theorem find_zeros :
  ∀ x ∈ Set.Icc 0 (2 * π), f x = 0 → x = π / 12 ∨ x = 13 * π / 12 ∨ x = 3 * π / 4 ∨ x = 7 * π / 4 := sorry

end smallest_positive_period_monotonic_intervals_find_zeros_l482_482949


namespace oldest_child_age_l482_482723

theorem oldest_child_age
  (avg_age : ℝ)
  (n : ℕ)
  (child1 child2 child3 child4 : ℕ)
  (oldest : ℕ) :
  avg_age = 8.2 → n = 5 → child1 = 6 → child2 = 10 → child3 = 12 → child4 = 7 →
  (n * avg_age).to_nat - (child1 + child2 + child3 + child4) = oldest →
  oldest = 6 :=
by
  intros h_avg h_n h_c1 h_c2 h_c3 h_c4 h_oldest
  sorry

end oldest_child_age_l482_482723


namespace sum_of_g_values_at_33_l482_482209

def f (x : ℝ) : ℝ := 4 * x^2 - 3
def g (x : ℝ) : ℝ := x^2 - x + 2

noncomputable def values_of_g (a : ℝ) := {x : ℝ | f x = a}.image g

theorem sum_of_g_values_at_33 : ∑ x in values_of_g 33, x = 22 :=
by sorry

end sum_of_g_values_at_33_l482_482209


namespace intersecting_lines_and_brocard_point_l482_482339

-- Definitions and conditions
variables (A B C A1 B1 C1 O P : Point)
variables (triangle_ABC : Triangle A B C)
variables (triangle_CA1B : SimilarTriangle C A1 B)
variables (triangle_CAB1 : SimilarTriangle C A B1)
variables (triangle_C1AB : SimilarTriangle C1 A B)

theorem intersecting_lines_and_brocard_point :
  (LineThrough A A1 ∩ LineThrough B B1 ∩ LineThrough C C1 = O) ∧
  (∠ ABP = ∠ CAP) ∧ (∠ CAP = ∠ BCP) ∧ (P = O) :=
begin
  sorry
end

end intersecting_lines_and_brocard_point_l482_482339


namespace smaller_root_of_equation_l482_482873

theorem smaller_root_of_equation :
  ∀ x : ℚ, (x - 7 / 8)^2 + (x - 1/4) * (x - 7 / 8) = 0 → x = 9 / 16 :=
by
  intro x
  intro h
  sorry

end smaller_root_of_equation_l482_482873


namespace count_valid_3_digit_numbers_l482_482554

-- Define the set of valid digits
def is_valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d < 10

-- Define the condition that the sum of the digits is divisible by a prime number
def is_divisible_by_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, nat.prime p ∧ n % p = 0

-- Define the set of 3-digit numbers satisfying the conditions
def valid_3_digit_numbers_count : ℕ :=
  (set.univ.filter (λ x : ℕ × ℕ × ℕ,
    let H := x.1 in
    let T := x.2.1 in
    let U := x.2.2 in
    is_valid_digit H ∧ is_valid_digit T ∧ is_valid_digit U ∧
    H ≥ 1 ∧ T > H ∧ U < H ∧ is_divisible_by_prime (H + T + U))).card

-- Statement of the problem
theorem count_valid_3_digit_numbers : 
  valid_3_digit_numbers_count = <known_value> := sorry

end count_valid_3_digit_numbers_l482_482554


namespace transistors_in_2010_l482_482648

-- Define initial conditions
def initial_transistors : ℕ := 500000
def years_passed : ℕ := 15
def tripling_period : ℕ := 3
def tripling_factor : ℕ := 3

-- Define the function to compute the number of transistors after a number of years
noncomputable def final_transistors (initial : ℕ) (years : ℕ) (period : ℕ) (factor : ℕ) : ℕ :=
  initial * factor ^ (years / period)

-- State the proposition we aim to prove
theorem transistors_in_2010 : final_transistors initial_transistors years_passed tripling_period tripling_factor = 121500000 := 
by 
  sorry

end transistors_in_2010_l482_482648


namespace probability_of_at_least_19_l482_482393

-- Defining the possible coins in Anya's pocket
def coins : list ℕ := [10, 10, 5, 5, 2]

-- Function to calculate the sum of chosen coins
def sum_coins (l : list ℕ) := list.sum l

-- Function to check if the sum of chosen coins is at least 19 rubles
def at_least_19 (l : list ℕ) := (sum_coins l) ≥ 19

-- Extract all possible combinations of 3 coins from the list
def combinations (l : list ℕ) (n : ℕ) := 
  if h : n ≤ l.length then 
    (list.permutations l).dedup.map (λ p, p.take n).dedup
  else
    []

-- Specific combinations of 3 coins out of 5
def three_coin_combinations := combinations coins 3 

-- Count the number of favorable outcomes (combinations that sum to at least 19)
def favorable_combinations := list.filter at_least_19 three_coin_combinations

-- Calculate the probability
def probability := (favorable_combinations.length : ℚ) / (three_coin_combinations.length : ℚ)

-- Prove that the probability is 0.4
theorem probability_of_at_least_19 : probability = 0.4 :=
  sorry

end probability_of_at_least_19_l482_482393


namespace matrix_proof_l482_482213
open Matrix Complex

variable {n : Type u} [Fintype n] [DecidableEq n]
variable (A B C D : Matrix n n ℂ) (k : ℝ)

def AC_kBD_eq_I (A B C D : Matrix n n ℂ) (k : ℝ) : Prop := A ⬝ C + (k: ℂ) • (B ⬝ D) = 1
def AD_eq_BC (A B C D : Matrix n n ℂ) : Prop := A ⬝ D = B ⬝ C

theorem matrix_proof (A B C D : Matrix n n ℂ) (k : ℝ) (h1 : AC_kBD_eq_I A B C D k) (h2 : AD_eq_BC A B C D) :
  (C ⬝ A + (k: ℂ) • (D ⬝ B) = 1) ∧ (D ⬝ A = C ⬝ B) := by
  sorry

end matrix_proof_l482_482213


namespace f_periodic_f_on_2_4_f_sum_2014_l482_482210

noncomputable def f : ℝ → ℝ := sorry

-- Define the conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_shift : ∀ x, f (x + 2) = -f x
axiom f_on_0_2 : ∀ x, (0 ≤ x ∧ x ≤ 2) → f x = 2 * x - x^2

-- 1. Prove f is periodic with a period of 4
theorem f_periodic : ∀ x, f (x + 4) = f x := sorry

-- 2. Find the explicit expression for f(x) when x ∈ [2, 4]
theorem f_on_2_4 (x : ℝ) (h : 2 ≤ x ∧ x ≤ 4) : f x = x^2 - 6 * x + 8 := sorry

-- 3. Calculate the value of f(0) + f(1) + ... + f(2014)
theorem f_sum_2014 : ∑ i in finset.range 2015, f i = 1 := sorry

end f_periodic_f_on_2_4_f_sum_2014_l482_482210


namespace avg_salary_of_non_officers_l482_482260

theorem avg_salary_of_non_officers 
  (avg_total_salary : ℝ) (avg_officer_salary : ℝ) (num_officers : ℕ) (num_non_officers : ℕ) : 
  avg_total_salary = 120 → avg_officer_salary = 430 → num_officers = 15 → num_non_officers = 465 → 
  ∃ avg_non_officer_salary, avg_non_officer_salary = 110 :=
by 
  assume h1 : avg_total_salary = 120
  assume h2 : avg_officer_salary = 430
  assume h3 : num_officers = 15
  assume h4 : num_non_officers = 465
  sorry

end avg_salary_of_non_officers_l482_482260


namespace min_value_of_f_is_neg1_l482_482028

def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 2

theorem min_value_of_f_is_neg1 : ∃ x : ℝ, f x = -1 ∧ ∀ y : ℝ, f y ≥ f x :=
begin
  sorry
end

end min_value_of_f_is_neg1_l482_482028


namespace paraboloid_area_first_octant_bounded_plane_y6_l482_482409

open RealMeasureTheory Set Filter

noncomputable def paraboloid_surface_area : ℝ :=
  let f x z := sqrt (1 + (4 * (x^2 + z^2)) / 9)
  let region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ p.1^2 + p.2^2 ≤ 18}
  (∫ region (λ p, f p.1 p.2)) * π / 2
-- Use improper integral calculus to derive the result
theorem paraboloid_area_first_octant_bounded_plane_y6 :
  paraboloid_surface_area = (39 * π) / 4 := by
  -- The proof is lengthy and involves proper change of variables and evaluation.
  sorry

end paraboloid_area_first_octant_bounded_plane_y6_l482_482409


namespace license_plate_count_l482_482968

theorem license_plate_count :
  let consonants := 21
  let vowels := 5
  let digits := 10
  consonants * vowels * consonants * digits = 22050 :=
by
  let consonants := 21
  let vowels := 5
  let digits := 10
  have h1 : consonants = 21 := rfl
  have h2 : vowels = 5 := rfl
  have h3 : digits = 10 := rfl
  have h_total : consonants * vowels * consonants * digits = 21 * 5 * 21 * 10 := by rw [h1, h2, h3]
  have h_result : 21 * 5 * 21 * 10 = 22050 := by norm_num
  rw h_total at h_result
  exact h_result

end license_plate_count_l482_482968


namespace solve_diamond_eq_l482_482064

-- Define the binary operation
def diamond (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : ℝ := a / b

-- State the theorem
theorem solve_diamond_eq :
  ∀ (x : ℝ) (hx : x ≠ 0), 
  diamond 504 (diamond 7 x sorry sorry) sorry = 150 → x = 175 / 84 :=
begin
  intros x hx h,
  sorry
end

end solve_diamond_eq_l482_482064


namespace purely_imaginary_implies_alpha_condition_l482_482126

theorem purely_imaginary_implies_alpha_condition
  (α : ℝ) (k : ℤ) :
  (∃ (z : ℂ), z = complex.of_real (sin α) - complex.I * complex.of_real (1 - cos α) 
    ∧ (z.im = complex.abs z))
  → (∃ (k : ℤ), α = (2 * k + 1) * π) :=
by
  sorry

end purely_imaginary_implies_alpha_condition_l482_482126


namespace simplify_sqrt_l482_482249

theorem simplify_sqrt (a b c d : ℝ) (h1 : a * b = d) (h2 : a + b = c) : 
  Real.sqrt (c - 2 * Real.sqrt d) = Real.sqrt a - Real.sqrt b := 
sorry

example : Real.sqrt (12 - 4 * Real.sqrt 5) = Real.sqrt 10 - Real.sqrt 2 :=
simplify_sqrt 10 2 12 20
(by norm_num)
(by norm_num)

end simplify_sqrt_l482_482249


namespace perfect_apples_count_l482_482150

def total_apples : ℕ := 60

def proportion_small : ℚ := 1/4
def proportion_medium : ℚ := 1/2
def proportion_large : ℚ := 1/4

def proportion_unripe : ℚ := 1/3
def proportion_partly_ripe : ℚ := 1/6
def proportion_fully_ripe : ℚ := 1/2

def small_apples : ℕ := (total_apples * proportion_small).to_nat
def medium_apples : ℕ := (total_apples * proportion_medium).to_nat
def large_apples : ℕ := (total_apples * proportion_large).to_nat

def fully_ripe_apples : ℕ := (total_apples * proportion_fully_ripe).to_nat

def fully_ripe_small_apples : ℕ := (fully_ripe_apples * proportion_small).to_nat
def fully_ripe_medium_apples : ℕ := (fully_ripe_apples * proportion_medium).to_nat
def fully_ripe_large_apples : ℕ := (fully_ripe_apples * proportion_large).to_nat

def perfect_apples : ℕ := fully_ripe_medium_apples + fully_ripe_large_apples

theorem perfect_apples_count : perfect_apples = 22 :=
by
  rw [perfect_apples, fully_ripe_medium_apples, fully_ripe_large_apples]
  -- We'll further expand the definitions and verify the calculations in the proof.
  sorry

end perfect_apples_count_l482_482150


namespace howard_rewards_l482_482405

theorem howard_rewards (initial_bowls : ℕ) (customers : ℕ) (customers_bought_20 : ℕ) 
                       (bowls_remaining : ℕ) (rewards_per_bowl : ℕ) :
  initial_bowls = 70 → 
  customers = 20 → 
  customers_bought_20 = 10 → 
  bowls_remaining = 30 → 
  rewards_per_bowl = 2 →
  ∀ (bowls_bought_per_customer : ℕ), bowls_bought_per_customer = 20 → 
  2 * (200 / 20) = 10 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end howard_rewards_l482_482405


namespace B_joined_amount_l482_482344

theorem B_joined_amount (T : ℝ)
  (A_investment : ℝ := 45000)
  (B_time : ℝ := 2)
  (profit_ratio : ℝ := 2 / 1)
  (investment_ratio_rule : (A_investment * T) / (B_investment_amount * B_time) = profit_ratio) :
  B_investment_amount = 22500 :=
by
  sorry

end B_joined_amount_l482_482344


namespace factor_64_minus_16y_squared_l482_482879

theorem factor_64_minus_16y_squared (y : ℝ) : 
  64 - 16 * y^2 = 16 * (2 - y) * (2 + y) :=
by
  -- skipping the actual proof steps
  sorry

end factor_64_minus_16y_squared_l482_482879


namespace gcd_1234_2047_l482_482889

theorem gcd_1234_2047 : Nat.gcd 1234 2047 = 1 :=
by sorry

end gcd_1234_2047_l482_482889


namespace evaluate_fraction_sqrt_l482_482442

theorem evaluate_fraction_sqrt :
  (Real.sqrt ((1 / 8) + (1 / 18)) = (Real.sqrt 26) / 12) :=
by
  sorry

end evaluate_fraction_sqrt_l482_482442


namespace min_units_l482_482351

theorem min_units (x : ℕ) (h1 : 5500 * 60 + 5000 * (x - 60) > 550000) : x ≥ 105 := 
by {
  sorry
}

end min_units_l482_482351


namespace f_eq_l482_482546

noncomputable def a (n : ℕ) : ℚ := 1 / ((n + 1) ^ 2)

noncomputable def f : ℕ → ℚ
| 0     => 1
| (n+1) => f n * (1 - a (n+1))

theorem f_eq : ∀ n : ℕ, f n = (n + 2) / (2 * (n + 1)) :=
by
  sorry

end f_eq_l482_482546


namespace cube_neighbors_l482_482741

-- Define a cube and its labeling
def cube := fin 8 → ℤ  -- A function from vertex indices (0 to 7) to labels (1 to 8)

-- A predicate for a "beautiful" face on the cube
def is_beautiful_face (f : fin 4 → fin 8) (c : cube) : Prop :=
  (c (f 0) = c (f 1) + c (f 2) + c (f 3)) ∨
  (c (f 1) = c (f 0) + c (f 2) + c (f 3)) ∨
  (c (f 2) = c (f 0) + c (f 1) + c (f 3)) ∨
  (c (f 3) = c (f 0) + c (f 1) + c (f 2))

-- Check that all faces of the cube meet the beautiful condition
def all_beautiful_faces (faces : list (fin 4 → fin 8)) (c : cube) : Prop :=
  ∀ f ∈ faces, is_beautiful_face f c

-- The main theorem statement to be proven
theorem cube_neighbors (faces : list (fin 4 → fin 8)) (c : cube)
  (h1 : all_beautiful_faces faces c)
  (h2 : c 0 = 6) :
  {c 1, c 2, c 3} = {2, 3, 5} :=
sorry

end cube_neighbors_l482_482741


namespace determine_x_l482_482029

variable {x y : ℝ}

theorem determine_x (h : (x - 1) / x = (y^3 + 3 * y^2 - 4) / (y^3 + 3 * y^2 - 5)) : 
  x = y^3 + 3 * y^2 - 5 := 
sorry

end determine_x_l482_482029


namespace sin_double_angle_of_line_inclination_l482_482079

open Real

theorem sin_double_angle_of_line_inclination : 
  (∃ α : ℝ, ∀ x y : ℝ, 2 * x - 4 * y + 5 = 0 → tan α = 1 / 2) → sin (2 * α) = 4 / 5 :=
by 
  intros h α ⟨ hx hy h_eq ⟩
  sorry

end sin_double_angle_of_line_inclination_l482_482079


namespace calculate_fraction_l482_482852

theorem calculate_fraction : 
  let a := 0.3
  let b := 0.03
  a = 3 * 10^(-1) ∧ b = 3 * 10^(-2)
  → (a^4 / b^3) = 300 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end calculate_fraction_l482_482852


namespace binomial_sum_identity_l482_482338

theorem binomial_sum_identity (n m k : ℕ) (h1 : n ≥ m) (h2 : m ≥ k) :
  (∑ i in Finset.range(n+1), Nat.choose i k * Nat.choose (n - i) (m - k)) = 
  (∑ t in Finset.range(m+2), Nat.choose k t * Nat.choose (n + 1 - k) (m + 1 - t)) := 
by 
  sorry

end binomial_sum_identity_l482_482338


namespace fx1_positive_l482_482950

theorem fx1_positive (f : ℝ → ℝ) (x0 x1 : ℝ)
  (h1 : f = λ x, (1 / 2) ^ x - Real.log x / Real.log 3)
  (h2 : f x0 = 0)
  (h3 : 0 < x1 ∧ x1 < x0) :
  f x1 > 0 := 
sorry

end fx1_positive_l482_482950


namespace player_5_shots_l482_482592

theorem player_5_shots
  (p1 : 15)
  (p2 : 20)
  (p3 : 14)
  (p4 : 12)
  (team_a_total : 75)
  (team_b_total : 68)
  (player_5_total : 14)
  (player_5_three_pointers : Nat)
  (player_5_two_point_shot : Nat)
  (player_5_free_throws : Nat)
  (player_5_three_pointers >= 2)
  (player_5_two_point_shot >= 1)
  (player_5_free_throws <= 4)
  (6 * player_5_three_pointers + 2 * player_5_two_point_shot + player_5_free_throws = player_5_total) :
  player_5_three_pointers = 2 ∧ player_5_two_point_shot = 2 ∧ player_5_free_throws = 4 :=
sorry

end player_5_shots_l482_482592


namespace min_score_needed_l482_482626

-- Definitions of the conditions
def current_scores : List ℤ := [88, 92, 75, 81, 68, 70]
def desired_increase : ℤ := 5
def number_of_tests := current_scores.length
def current_total : ℤ := current_scores.sum
def current_average : ℤ := current_total / number_of_tests
def desired_average : ℤ := current_average + desired_increase 
def new_number_of_tests : ℤ := number_of_tests + 1
def total_required_score : ℤ := desired_average * new_number_of_tests

-- Lean 4 statement (theorem) to prove
theorem min_score_needed : total_required_score - current_total = 114 := by
  sorry

end min_score_needed_l482_482626


namespace impossible_to_place_10_squares_l482_482667

def unit_square (S : Type*) := sorry -- Define what a unit square is (this is a placeholder)

def interior (S : unit_square) : set (S : Type*) := sorry -- Define the interior of a square (this is a placeholder)

def point_on_boundary_or_corner (S T : unit_square) : Prop := sorry -- Define when a point is on a boundary or corner (this is a placeholder)

theorem impossible_to_place_10_squares :
  ∀ (S : Type*) [finset.unit_square S], ∀ n : ℕ, n = 10 →
  (∀ i j ∈ finset.range n, i ≠ j → interior(S i) ∩ interior(S j) = ∅) →
  ¬ (∃ T, ∀ i ≠ j, point_on_boundary_or_corner(T, S i)) :=
begin
  sorry
end

end impossible_to_place_10_squares_l482_482667


namespace compare_flows_l482_482336

-- Let's define the constants and assumptions first
def water_flow (n : Nat) : Nat → ℝ
| 0 => 1  -- Initial flow rate F is normalized to 1 unit for simplicity
| (n+1) => water_flow n n / 2  -- Flow rate divided equally at each branching

-- Jeníček's specific points' total flow in the 5th row
def Jenicek_flow : ℝ :=
  let f := water_flow 5
  (f 0) + (f 2) + (f 8) + (f 10)

-- Mařenka's total flow in the 2019th row
def Marenka_flow : ℝ :=
  water_flow 2019 0  -- It's enough to check just one node as the total flow for the complete row would be the same as the initial flow

-- Theorem stating that the two total flows are equal
theorem compare_flows : Jenicek_flow = water_flow 0 0 := by
  sorry

end compare_flows_l482_482336


namespace candle_burning_problem_l482_482010

theorem candle_burning_problem (burn_time_per_night_1h : ∀ n : ℕ, n = 8) 
                                (nightly_burn_rate : ∀ h : ℕ, h / 2 = 4) 
                                (total_nights : ℕ) 
                                (two_hour_nightly_burn : ∀ t : ℕ, t = 24) 
                                : ∃ candles : ℕ, candles = 6 := 
by {
  sorry
}

end candle_burning_problem_l482_482010


namespace find_a8_l482_482596

variable (a : ℕ → ℝ)
variable (q : ℝ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_a8 
  (hq : is_geometric_sequence a q)
  (h1 : a 1 * a 3 = 4)
  (h2 : a 9 = 256) : 
  a 8 = 128 ∨ a 8 = -128 :=
by
  sorry

end find_a8_l482_482596


namespace triangle_perimeter_l482_482169

-- Definitions and conditions.
variables (A B C : Type) [triangle A B C]

axiom len_opposite (side_a side_b side_c : ℝ) : 
(triangle A B C) (len_opposite a A) (len_opposite a A) (len_opposite b B)
(len_opposite c C)

axiom perimeter_condition : 
(b = 9) ∧ (a = 2*c) ∧ (B = π/3)

theorem triangle_perimeter (a b c : ℝ) (perimeter : ℝ) :
  (len_opposite a A ∧ len_opposite b B ∧ len_opposite c C) →
  (perimeter_condition) →
  perimeter = 9 + 9 * sin((a + b) * B/c) := 
by sorry

end triangle_perimeter_l482_482169


namespace angle_ratio_l482_482608

-- Conditions and setup
variables {α β γ : Type} [linear_ordered_field α]
variables {P Q B M : β}
variables (θ ψ : α)

-- Condition 1: BP and BQ bisect ∠ABC
axiom BQ_bisects_ABC : 
  ∀ (A B C : γ), ∠(A, B, P) = ∠(P, B, C)

axiom BP_bisects_ABC : 
  ∀ (A B C : γ), ∠(A, B, Q) = ∠(Q, B, C)

-- Condition 2: BM bisects ∠PBQ
axiom BM_bisects_PBQ : 
  ∠(P, B, M) = ∠(M, B, Q)

-- Prove the desired ratio
theorem angle_ratio (h1 : BQ_bisects_ABC) (h2 : BP_bisects_ABC) (h3 : BM_bisects_PBQ) : 
  θ / ψ = 1 / 4 :=
sorry

end angle_ratio_l482_482608


namespace nested_square_root_eval_l482_482572

theorem nested_square_root_eval (x : ℝ) (hx : x ≥ 0) : (sqrt (x * sqrt (x * sqrt (x * sqrt x)))) = x^(11 / 8) :=
  sorry

end nested_square_root_eval_l482_482572


namespace probability_one_absent_other_present_l482_482591

noncomputable def prob_absent_present (absent_rate : ℚ) : ℚ :=
  let present_rate := 1 - absent_rate in
  (present_rate * absent_rate + absent_rate * present_rate) * 100

theorem probability_one_absent_other_present :
  prob_absent_present (1/20) = 9.5 :=
by
  sorry

end probability_one_absent_other_present_l482_482591


namespace last_digit_inverse_power_two_l482_482763

theorem last_digit_inverse_power_two :
  let n := 12
  let x := 5^n
  let y := 10^n
  (x % 10 = 5) →
  ((1 / (2^n)) * (5^n) / (5^n) == (5^n) / (10^n)) →
  (y % 10 = 0) →
  ((1 / (2^n)) % 10 = 5) :=
by
  intros n x y h1 h2 h3
  sorry

end last_digit_inverse_power_two_l482_482763


namespace average_temperature_l482_482829

theorem average_temperature (temps : List ℕ) (temps_eq : temps = [40, 47, 45, 41, 39]) :
  (temps.sum : ℚ) / temps.length = 42.4 :=
by
  sorry

end average_temperature_l482_482829


namespace purely_imaginary_sin_cos_l482_482125

theorem purely_imaginary_sin_cos (α : ℝ) (k : ℤ) :
  (∃ z : ℂ, z = complex.sin α - complex.I * (1 - complex.cos α) ∧ z.im = 0) ↔ α = (2 * k + 1) * real.pi :=
sorry

end purely_imaginary_sin_cos_l482_482125


namespace triangle_perimeter_l482_482850

-- Define the points A, B, and C
def A := (2, 3)
def B := (2, 10)
def C := (8, 6)

-- Define the distance formula between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Calculate the distances between the points
def AB := distance A B
def BC := distance B C
def CA := distance C A

-- Prove that the perimeter is as expected
theorem triangle_perimeter :
  AB + BC + CA = 7 + 2 * real.sqrt 13 + 3 * real.sqrt 5 :=
by
  -- The proof will go here
  sorry

end triangle_perimeter_l482_482850


namespace david_age_l482_482322

theorem david_age (x : ℕ) (y : ℕ) (h1 : y = x + 7) (h2 : y = 2 * x) : x = 7 :=
by
  sorry

end david_age_l482_482322


namespace average_of_numbers_divisible_by_13_between_200_and_10000_is_5102_l482_482885

noncomputable def first_number : ℕ := (Nat.ceil (200 / 13) : ℕ) * 13
noncomputable def last_number : ℕ := (Nat.floor (10000 / 13) : ℕ) * 13
noncomputable def num_terms : ℕ := ((last_number - first_number) / 13) + 1
noncomputable def average : ℚ := (first_number + last_number) / 2

theorem average_of_numbers_divisible_by_13_between_200_and_10000_is_5102.5 :
  average = 5102.5 := 
sorry

end average_of_numbers_divisible_by_13_between_200_and_10000_is_5102_l482_482885


namespace simplify_sum_of_square_roots_l482_482684

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l482_482684


namespace quadratic_decreasing_interval_l482_482080

theorem quadratic_decreasing_interval (a : ℝ) :
  (∀ x : ℝ, x ∈ Iio (6 : ℝ) → deriv (λ x, x^2 + 4*a*x + 2) x ≤ 0) ↔ (a ≤ -3) :=
by
  sorry

end quadratic_decreasing_interval_l482_482080


namespace distinct_floor_squares_2007_l482_482872

noncomputable theory

open Int

def distinctIntegers (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).card

theorem distinct_floor_squares_2007 :
  distinctIntegers (2007) = 1506 := by
  sorry

end distinct_floor_squares_2007_l482_482872


namespace locus_of_points_equidistant_l482_482605

-- Define the given point F and line l using complex number representation
def F : ℂ := -1 / 3 + 3 * Complex.I
def l (z : ℂ) : Prop := 3 * z + 3 * Complex.conj z + 2 = 0

-- The statement asserting the locus of points equidistant from F and l is a straight line y=3
theorem locus_of_points_equidistant (z : ℂ) :
  (Complex.abs (z - F) = Complex.abs (z - Complex.mpr (λ x, (0, 3))).re) ↔ (Complex.im z = 3) :=
sorry

end locus_of_points_equidistant_l482_482605


namespace repeating_decimal_to_fraction_product_l482_482303

theorem repeating_decimal_to_fraction_product :
  let x := 0.037 in
  let frac := 1 / 27 in
  x = frac → (1 * 27 = 27) :=
sorry

end repeating_decimal_to_fraction_product_l482_482303


namespace indistinguishable_balls_boxes_l482_482972

theorem indistinguishable_balls_boxes (n m : ℕ) (h : n = 6) (k : m = 2) : 
  (finset.card (finset.filter (λ x : finset ℕ, x.card ≤ n / 2) 
    (finset.powerset (finset.range (n + 1)))) = 4) :=
by
  sorry

end indistinguishable_balls_boxes_l482_482972


namespace isosceles_triangle_largest_angle_l482_482999

theorem isosceles_triangle_largest_angle 
  (T : Triangle)
  (h_iso : IsIsosceles T)
  (h_angle : ∃ (A : T.angle), T.angle A = 50) : 
  ∃ (A : T.angle), T.angle A = 80 := 
sorry

end isosceles_triangle_largest_angle_l482_482999


namespace quadratic_polynomial_properties_l482_482042

theorem quadratic_polynomial_properties :
  ∃ k : ℝ, (k * (3 - (3+4*I)) = 8 ∧ 
            (∀ x : ℂ, (x = (3 + 4 * I) → polynomial.eval x (k * (X - (3+4*I)) * (X - (3-4*I))) = 0)) ∧ 
            polynomial.coeff (k * (X - (3+4*I)) * (X - (3-4*I))) 1 = 8) :=
sorry

end quadratic_polynomial_properties_l482_482042


namespace five_circles_common_point_l482_482912

variables (P : Type) [metric_space P]

variables (a b c d e : set P)

def is_circle (C : set P) : Prop := sorry  -- define what it means to be a circle

axiom four_circles_common_point (C1 C2 C3 C4 : set P) : is_circle C1 → is_circle C2 → is_circle C3 → is_circle C4 → ∃ p : P, p ∈ C1 ∧ p ∈ C2 ∧ p ∈ C3 ∧ p ∈ C4

theorem five_circles_common_point
  (h_a : is_circle a) 
  (h_b : is_circle b) 
  (h_c : is_circle c) 
  (h_d : is_circle d) 
  (h_e : is_circle e)
  (h_abcd : ∃ p : P, p ∈ a ∧ p ∈ b ∧ p ∈ c ∧ p ∈ d)
  (h_abce : ∃ p : P, p ∈ a ∧ p ∈ b ∧ p ∈ c ∧ p ∈ e)
  (h_abde : ∃ p : P, p ∈ a ∧ p ∈ b ∧ p ∈ d ∧ p ∈ e)
  (h_acde : ∃ p : P, p ∈ a ∧ p ∈ c ∧ p ∈ d ∧ p ∈ e)
  (h_bcde : ∃ p : P, p ∈ b ∧ p ∈ c ∧ p ∈ d ∧ p ∈ e) :
  ∃ p : P, p ∈ a ∧ p ∈ b ∧ p ∈ c ∧ p ∈ d ∧ p ∈ e :=
by sorry

end five_circles_common_point_l482_482912


namespace chef_michel_total_pies_l482_482862

theorem chef_michel_total_pies 
  (shepherd_pie_pieces : ℕ) 
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ) 
  (chicken_pot_pie_customers : ℕ) 
  (h1 : shepherd_pie_pieces = 4)
  (h2 : chicken_pot_pie_pieces = 5)
  (h3 : shepherd_pie_customers = 52)
  (h4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) +
  (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by {
  sorry
}

end chef_michel_total_pies_l482_482862


namespace op_recursive_evaluation_l482_482426

-- Define the operation ⊕
def op (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Prove the desired expression evaluates to 1
theorem op_recursive_evaluation : op 10 (op 9 (op 8 (op 7 (op 6 (op 5 (op 4 (op 3 (op 2 1)))))))) = 1 := by
  sorry

end op_recursive_evaluation_l482_482426


namespace find_a_b_l482_482985

theorem find_a_b (a b : ℝ) : 
  let y := λ x : ℝ, x ^ 2 + a * x + b,
      tangent_eq := λ (x y : ℝ), x - y + 1 = 0,
      pt := (⟨0, b⟩ : ℝ × ℝ) in
  (tangent_eq pt.1 (y pt.1)) ∧ 
  (∃ f' : ℝ → ℝ, 
     (f' = λ x, 2 * x + a) ∧ 
     (f' pt.1 = a)) → 
  (a = 1 ∧ b = 1) :=
by injective sorry

end find_a_b_l482_482985


namespace value_of_f_neg_a_l482_482909

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + x^3 + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 3) : f (-a) = -1 := 
by
  sorry

end value_of_f_neg_a_l482_482909


namespace James_pays_6_dollars_l482_482181

-- Defining the conditions
def packs : ℕ := 4
def stickers_per_pack : ℕ := 30
def cost_per_sticker : ℚ := 0.10
def friend_share : ℚ := 0.5

-- Total number of stickers
def total_stickers : ℕ := packs * stickers_per_pack

-- Total cost calculation
def total_cost : ℚ := total_stickers * cost_per_sticker

-- James' payment calculation
def james_payment : ℚ := total_cost * friend_share

-- Theorem statement to be proven
theorem James_pays_6_dollars : james_payment = 6 := by
  sorry

end James_pays_6_dollars_l482_482181


namespace factorial_division_l482_482867

theorem factorial_division :
  (12! / (7! * 5!) = 792) :=
by
  sorry

end factorial_division_l482_482867


namespace game_expected_value_l482_482798

theorem game_expected_value (m : ℕ) (h : (7 * 3 + m * (-1))/(7 + m) = 1) : m = 7 :=
by
  sorry

end game_expected_value_l482_482798


namespace sin_gamma_delta_l482_482978

theorem sin_gamma_delta (γ δ : ℝ)
  (hγ : Complex.exp (Complex.I * γ) = Complex.ofReal 4 / 5 + Complex.I * (3 / 5))
  (hδ : Complex.exp (Complex.I * δ) = Complex.ofReal (-5 / 13) + Complex.I * (12 / 13)) :
  Real.sin (γ + δ) = 21 / 65 :=
by
  sorry

end sin_gamma_delta_l482_482978


namespace tie_distribution_impossible_l482_482356

-- Definitions based on the identified conditions
def youth_knows_girls (youth : Type) (girl : Type) (knows : youth → girl → Prop) : Prop :=
  ∀ (y : youth), ∀ (s : set girl), (finite s) → card s ≥ 2015 → ∃ (g1 g2 : girl), g1 ∈ s ∧ g2 ∈ s ∧ g1 ≠ g2 ∧ ∃ (color : Type) (tie : girl → color), tie g1 ≠ tie g2

def girl_knows_youths (girl : Type) (youth : Type) (knows : girl → youth → Prop) : Prop :=
  ∀ (g : girl), ∀ (s : set youth), (finite s) → card s ≥ 2015 → ∃ (y1 y2 : youth), y1 ∈ s ∧ y2 ∈ s ∧ y1 ≠ y2 ∧ ∃ (color : Type) (tie : youth → color), tie y1 ≠ tie y2

theorem tie_distribution_impossible (youth girl : Type) (knows_yg : youth → girl → Prop) (knows_gy : girl → youth → Prop) :
  ¬ (youth_knows_girls youth girl knows_yg ∧ girl_knows_youths girl youth knows_gy) :=
sorry

end tie_distribution_impossible_l482_482356


namespace cameron_list_count_l482_482410

theorem cameron_list_count :
  let lower := 100
  let upper := 1000
  let step := 20
  let n_min := lower / step
  let n_max := upper / step
  lower % step = 0 ∧ upper % step = 0 →
  upper ≥ lower →
  n_max - n_min + 1 = 46 :=
by
  sorry

end cameron_list_count_l482_482410


namespace find_quadratic_polynomial_l482_482044

def quadratic_polynomial (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem find_quadratic_polynomial : 
  ∃ a b c: ℝ, (∀ x : ℂ, quadratic_polynomial a b c x.re = 0 → (x = 3 + 4*I) ∨ (x = 3 - 4*I)) 
  ∧ (b = 8) 
  ∧ (a = -4/3) 
  ∧ (c = -50/3) :=
by
  sorry

end find_quadratic_polynomial_l482_482044


namespace determine_length_BD_l482_482165

open Real

noncomputable def length_BD (AB AC AD BC BD : ℝ) : Prop :=
  AB = 45 ∧ AC = 120 ∧
  (BC = sqrt (AB^2 + AC^2)) ∧
  (AD = (2 * sqrt (AB^2 + AC^2) * (1 / 2) * AB * AC) / sqrt (AB^2 + AC^2)) ∧ 
  (BD = sqrt (AB^2 - AD^2)) 
  → BD = 5 * sqrt 17

theorem determine_length_BD : length_BD 45 120 40 135 (5 * sqrt 17) :=
by sorry

end determine_length_BD_l482_482165


namespace floor_factorial_even_l482_482054

theorem floor_factorial_even (n : ℕ) (hn : n > 0) : 
  Nat.floor ((Nat.factorial (n - 1) : ℝ) / (n * (n + 1))) % 2 = 0 := 
sorry

end floor_factorial_even_l482_482054


namespace total_weight_moved_l482_482752

theorem total_weight_moved (tom_weight : ℝ) (vest_fraction : ℝ) (hold_fraction : ℝ) :
  tom_weight = 150 → vest_fraction = 0.5 → hold_fraction = 1.5 →
  let vest_weight := vest_fraction * tom_weight,
      hand_weight := hold_fraction * tom_weight,
      total_hand_weight := 2 * hand_weight,
      total_weight := tom_weight + vest_weight + total_hand_weight in
  total_weight = 675 :=
by
  sorry

end total_weight_moved_l482_482752


namespace find_a17_a18_a19_a20_l482_482619

variable {α : Type*} [Field α]

-- Definitions based on the given conditions:
def geometric_sequence (a : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, a n = a 0 * r ^ n

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n : ℕ, S n = (Finset.range n).sum a

-- Problem statement based on the question and conditions:
theorem find_a17_a18_a19_a20 (a S : ℕ → α) (h_geom : geometric_sequence a)
  (h_sum : sum_of_first_n_terms a S) (hS4 : S 4 = 1) (hS8 : S 8 = 3) :
  a 17 + a 18 + a 19 + a 20 = 16 :=
sorry

end find_a17_a18_a19_a20_l482_482619


namespace pavan_speed_l482_482656

theorem pavan_speed (total_distance : ℕ) (total_time : ℕ) (second_half_speed : ℕ) :
  total_distance = 300 → total_time = 11 → second_half_speed = 25 →
  let first_half_distance := total_distance / 2 in
  let second_half_distance := total_distance / 2 in
  let second_half_time := second_half_distance / second_half_speed in
  let first_half_time := total_time - second_half_time in
  let first_half_speed := first_half_distance / first_half_time in
  first_half_speed = 30 :=
by
  intros
  sorry

end pavan_speed_l482_482656


namespace planes_parallel_implies_intersections_parallel_l482_482500

-- Definitions for the planes and their intersections
variables (α β γ : Plane)
variables (m n : Line)

-- Conditions
axiom α_inter_γ : (α ∩ γ) = m
axiom β_inter_γ : (β ∩ γ) = n

-- The statement to be proven
theorem planes_parallel_implies_intersections_parallel (h : α ∥ β) : m ∥ n :=
sorry

end planes_parallel_implies_intersections_parallel_l482_482500


namespace find_area_triangle_l482_482514

variables {A B C : Type} [triangle A B C]
variables {a b c : ℝ}
variables (area_B1 area_B2 : ℝ)

-- Condition for angle B
axiom angle_condition :
  a^2 + c^2 - b^2 = sqrt 3 * a * c → angle B = π / 6

-- Conditions for area calculations
axiom side_length_b : b = 2
axiom side_length_c : c = 2 * sqrt 3
axiom angle_B : ∠ B = π / 6

-- Definitions for the area based on angle B
noncomputable def area_triangle_ABC :=
  1 / 2 * a * c * sin (∠ B)

-- Theorem statement: area of triangle ABC with specified side lengths and angle B can be either sqrt 3 or 2 sqrt 3
theorem find_area_triangle :
  a = 2 → area_triangle_ABC = sqrt 3 → a = 4 → area_triangle_ABC = 2 * sqrt 3 → ∃ area_B1 area_B2, 
    area_B1 = sqrt 3 ∧ area_B2 = 2 * sqrt 3 := 
  sorry

end find_area_triangle_l482_482514


namespace area_of_isosceles_triangle_l482_482237

open Real

noncomputable def isosceles_triangle_area (AO OM : ℝ) : ℝ :=
  let x_sqrd := 20 / 121 in
  let sin_ACB := ((4 : ℝ) * sqrt 5) / 9 in
  let area := (1 / 2) * 11 * sqrt x_sqrd * (99 * sqrt x_sqrd / 2) * sin_ACB in
  area

theorem area_of_isosceles_triangle :
  ∀ (AO OM : ℝ), AO = 3 → OM = 27 / 11 → isosceles_triangle_area AO OM = 20 * sqrt 5 :=
begin
  intros AO OM hAO hOM,
  sorry
end

end area_of_isosceles_triangle_l482_482237


namespace evaluate_expression_l482_482219

noncomputable def a : ℝ := Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 15
noncomputable def b : ℝ := -Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 15
noncomputable def c : ℝ := Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 15
noncomputable def d : ℝ := -Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 15

theorem evaluate_expression (ha : a = Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 15)
                            (hb : b = -Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 15)
                            (hc : c = Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 15)
                            (hd : d = -Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 15) :
   (1 / a + 1 / b + 1 / c + 1 / d) ^ 2 = 960 / 3481 := 
begin
  -- Proof goes here
  sorry
end

end evaluate_expression_l482_482219


namespace find_a_l482_482075

noncomputable def f (t : ℝ) (a : ℝ) : ℝ := (1 / (Real.cos t)) + (a / (1 - (Real.cos t)))

theorem find_a (t : ℝ) (a : ℝ) (h1 : 0 < t) (h2 : t < (Real.pi / 2)) (h3 : 0 < a) (h4 : ∀ t, 0 < t ∧ t < (Real.pi / 2) → f t a = 16) :
  a = 9 :=
sorry

end find_a_l482_482075


namespace similar_triangle_leg_length_l482_482364

theorem similar_triangle_leg_length (a b c : ℝ) (h0 : a = 12) (h1 : b = 9) (h2 : c = 7.5) :
  ∃ y : ℝ, ((12 / 7.5) = (9 / y) → y = 5.625) :=
by
  use 5.625
  intro h
  linarith

end similar_triangle_leg_length_l482_482364


namespace area_of_triangle_ABC_l482_482347

theorem area_of_triangle_ABC 
  (r : ℝ) (R : ℝ) (ACB : ℝ) 
  (hr : r = 2) 
  (hR : R = 4) 
  (hACB : ACB = 120) : 
  let s := (2 * (2 + 4 * Real.sqrt 3)) / Real.sqrt 3 
  let S := s * r 
  S = 56 / Real.sqrt 3 :=
sorry

end area_of_triangle_ABC_l482_482347


namespace flag_design_count_l482_482869

theorem flag_design_count :
  let colors := {1, 2, 3}  -- 1 represents purple, 2 represents gold, 3 represents silver
  ∃ (flag : Fin 3 → Σ n : ℕ, n ∈ colors), (∀ i : Fin 2, flag i.1 ≠ flag i.2) ∧ (card colors * card (colors \ {flag 0.2}) * card (colors \ {flag 1.2}) = 12) :=
by
  sorry

end flag_design_count_l482_482869


namespace train_crossing_time_l482_482330

def train_length : ℝ := 110
def train_speed_kmph : ℝ := 60
def bridge_length : ℝ := 170
def kmph_to_mps (speed : ℝ) : ℝ := speed * 1000 / 3600

theorem train_crossing_time : 
  let total_distance := train_length + bridge_length
  let train_speed_mps := kmph_to_mps train_speed_kmph
  total_distance / train_speed_mps ≈ 16.79 := 
by
  sorry

end train_crossing_time_l482_482330


namespace purely_imaginary_sin_cos_l482_482124

theorem purely_imaginary_sin_cos (α : ℝ) (k : ℤ) :
  (∃ z : ℂ, z = complex.sin α - complex.I * (1 - complex.cos α) ∧ z.im = 0) ↔ α = (2 * k + 1) * real.pi :=
sorry

end purely_imaginary_sin_cos_l482_482124


namespace meeting_point_l482_482834

/-- Along a straight alley with 400 streetlights placed at equal intervals, numbered consecutively from 1 to 400,
    Alla and Boris set out towards each other from opposite ends of the alley with different constant speeds.
    Alla starts at streetlight number 1 and Boris starts at streetlight number 400. When Alla is at the 55th streetlight,
    Boris is at the 321st streetlight. The goal is to prove that they will meet at the 163rd streetlight.
-/
theorem meeting_point (n : ℕ) (h1 : n = 400) (h2 : ∀ i j k l : ℕ, i = 55 → j = 321 → k = 1 → l = 400) : 
  ∃ m, m = 163 := 
by
  sorry

end meeting_point_l482_482834


namespace math_problem_l482_482419

theorem math_problem :
  (8 / 125 : ℝ) ^ (-2 / 3) - log 10 (sqrt 2) - log 10 (sqrt 5) = 23 / 4 :=
by
  sorry

end math_problem_l482_482419


namespace find_probability_l482_482357

noncomputable def is_lattice_point_visible_from_origin (x y z : ℤ) : Prop :=
  Int.gcd (Int.gcd x y) z = 1

noncomputable def probability_visible_from_origin : ℝ :=
  1 / (∏ p in filter Nat.prime (Icc 2 1000000), (1 - (1:ℝ)/p)^3)

theorem find_probability : 
  ∃ (i k s : ℤ), i = 1 ∧ k = 1 ∧ s = 3 ∧ 
  (1 / probability_visible_from_origin = ∑' (n : ℕ) in (Icc 1 1000000), (1:ℝ) / (n^3)) :=
begin
  use [1, 1, 3],
  simp,
  sorry
end

end find_probability_l482_482357


namespace probability_page_multiple_of_7_l482_482724

theorem probability_page_multiple_of_7 (total_pages : ℕ) (probability : ℚ)
  (h_total_pages : total_pages = 500) 
  (h_probability : probability = 71 / 500) :
  probability = 0.142 := 
sorry

end probability_page_multiple_of_7_l482_482724


namespace pappus_theorem_l482_482653

theorem pappus_theorem
  {A1 B1 C1 A2 B2 C2 A B C : Type}
  [has_collinear A1 B1 C1]
  [has_collinear A2 B2 C2]
  (hC : is_intersect A1 B2 A2 B1 C)
  (hA : is_intersect B1 C2 B2 C1 A)
  (hB : is_intersect C1 A2 C2 A1 B) :
  are_collinear A B C := 
sorry

end pappus_theorem_l482_482653


namespace minimum_volume_ratio_l482_482512

theorem minimum_volume_ratio (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  let V1 := x * y * z,
      V2 := 8 * (x + y) * (y + z) * (z + x) in
  V1 > 0 ∧ V2 > 0 →
  (V1 = x * y * z) →
  (V2 = 8 * (x + y) * (y + z) * (z + x)) →
  (8 * (x + y) * (y + z) * (z + x)) / (x * y * z) ≥ 64 := sorry

end minimum_volume_ratio_l482_482512


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l482_482481

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31 :
  let n := 7^7 - 7^4 in
  let prime_factors := {2, 3, 7, 19} in
  finset.sum prime_factors id = 31 :=
by
  sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l482_482481


namespace digit_7_count_from_10_to_99_l482_482172

theorem digit_7_count_from_10_to_99 : (List.range' 10 90).countp (fun n => n / 10 = 7 ∨ n % 10 = 7) = 19 := by
  sorry

end digit_7_count_from_10_to_99_l482_482172


namespace math_problem_l482_482865

theorem math_problem (x : ℤ) (h : x = 9) :
  (x^6 - 27*x^3 + 729) / (x^3 - 27) = 702 :=
by
  sorry

end math_problem_l482_482865


namespace three_letter_initials_l482_482965

theorem three_letter_initials : 
  let letters := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'} in
  let choices_for_first := 10 in
  let choices_for_third := 9 in
  let arrangements := 3 in
  choices_for_first * choices_for_third * arrangements = 270 :=
by
  sorry

end three_letter_initials_l482_482965


namespace sum_reciprocals_divisors_of_144_l482_482052

theorem sum_reciprocals_divisors_of_144 :
  let σ := 403 in
  144 = 2^4 * 3^2 →
  σ = ∑ d in (Finset.filter (λ d, 144 % d = 0) (Finset.range (144+1))), d →
  (∑ d in (Finset.filter (λ d, 144 % d = 0) (Finset.range (144+1))), (1 : ℚ) / d) = 403 / 144 :=
by
  intros h1 h2
  sorry

end sum_reciprocals_divisors_of_144_l482_482052


namespace inequality_solution_l482_482222

theorem inequality_solution (f g : ℝ → ℝ) :
  (∀ x, f(x) ≥ 0 ↔ x ∈ Set.Icc 1 2) →
  (∀ x, g(x) ≥ 0 ↔ False) →
  {x | (f x) / (g x) > 0} = {x | x < 1} ∪ {x | x > 2} :=
by
  intros hf hg
  sorry

end inequality_solution_l482_482222


namespace solve_problem_l482_482505

variable (x y : ℝ)

def A (x : ℝ) : Set ℝ := {x^2 + x + 1, -x, -x - 1}
def B (y : ℝ) : Set ℝ := {-y, -y / 2, y + 1}

theorem solve_problem
  (hx : x ∈ ℝ)
  (hy : y ∈ {y : ℝ | y > 0})
  (hAB : A x = B y) :
  x^2 + y^2 = 5 :=
begin
  sorry
end

end solve_problem_l482_482505


namespace find_a_eq_neg14_l482_482211

def h (x : ℝ) : ℝ :=
if x ≤ 0 then -x else 3*x - 30

theorem find_a_eq_neg14 (a : ℝ) (ha : a < 0) : h (h (h 6)) = h (h (h a)) ↔ a = -14 := 
by 
  sorry

end find_a_eq_neg14_l482_482211


namespace number_divisible_by_5_probability_l482_482726

theorem number_divisible_by_5_probability : 
  let digits := {0, 3, 5, 7}
  let numbers := permutations digits
  (count (λ n, n % 5 = 0) numbers) / (count (λ n, true) numbers) = 1 / 2 :=
sorry

end number_divisible_by_5_probability_l482_482726


namespace max_ab_eq_one_quarter_l482_482570

theorem max_ab_eq_one_quarter (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : ab ≤ 1 / 4 :=
by
  sorry

end max_ab_eq_one_quarter_l482_482570


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l482_482472

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4 :
  let n := 7^7 - 7^4 in 
  (∑ p in (nat.factors n).to_finset, p) = 31 :=
by sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l482_482472


namespace problem_solution_l482_482937

section
variable (x y t θ ρ α : ℝ)

-- Curve C: x^2/4 + y^2 = 1
def curve_C (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

-- Line l: parametric form x = t, y = 2 - sqrt(3) * t
def line_l (x y t : ℝ) : Prop := (x = t) ∧ (y = 2 - real.sqrt 3 * t)

-- Parametric equation of curve C
def parametric_curve (θ : ℝ) : ℝ × ℝ := (2 * real.cos θ, real.sin θ)

-- Polar coordinate equation of line l
def polar_line_l (θ ρ : ℝ) : Prop := ρ * real.sin(θ + real.pi / 3) = 1

-- Distance from point P on curve C to line l
def dist_PA (θ α : ℝ) : ℝ := |real.sqrt 13 * real.sin(θ + α) - 2|

-- Maximum value of |PA|
def max_PA (α : ℝ) : ℝ := real.sqrt 13 + 2

-- Minimum value of |PA|
def min_PA (α : ℝ) : ℝ := 0

-- Prove the conditions
theorem problem_solution : 
  (∀ (θ ρ : ℝ), polar_line_l θ ρ) ∧
  (∀ (α : ℝ), ∃ θ, dist_PA θ α = max_PA α) ∧
  (∀ (α : ℝ), ∃ θ, dist_PA θ α = min_PA α) :=
by
  sorry
end

end problem_solution_l482_482937


namespace range_g_l482_482433

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arcsin (x / 3))^2 - Real.pi * Real.arccos (x / 3) + 
  (Real.arccos (x / 3))^2 + 
  (Real.pi^2 / 18) * (x^2 - 3 * x + 9)

theorem range_g : 
  set.range (g) ⊆ set.Icc (Real.pi^2 / 4) (3 * Real.pi^2 / 2) :=
by {
  sorry
}

end range_g_l482_482433


namespace intervals_strictly_decreasing_triangle_abc_a_value_l482_482933

noncomputable def f (ω x : ℝ) := -cos(ω/2 * x)^2 + (sqrt 3 / 2) * sin(ω * x)

theorem intervals_strictly_decreasing (ω : ℝ) (hω : ω > 0)
  (h : ∀ x, f ω (x + π/2/ω) = f ω x) :
  ∀ k : ℤ, ∀ x : ℝ, k * π + π / 3 ≤ x ∧ x ≤ k * π + 5 * π / 6 → is_strict_decreasing (f ω) x :=
sorry

theorem triangle_abc_a_value (A : ℝ) (a b c S : ℝ)
  (h1 : f 2 A = 1 / 2) 
  (h2 : c = 3) (h3 : S = 3 * sqrt 3)
  (h_area : S = 1 / 2 * b * c * sin A)
  (h_cos : cos A = 1 / 2) :
  a = sqrt 13 := 
sorry

end intervals_strictly_decreasing_triangle_abc_a_value_l482_482933


namespace average_velocity_mass_flow_rate_available_horsepower_l482_482133

/-- Average velocity of water flowing out of the sluice gate. -/
theorem average_velocity (g h₁ h₂ : ℝ) (h1_5m : h₁ = 5) (h2_5_4m : h₂ = 5.4) (g_9_81 : g = 9.81) :
    (1 / 2) * (Real.sqrt (2 * g * h₁) + Real.sqrt (2 * g * h₂)) = 10.1 :=
by
  sorry

/-- Mass flow rate of water per second when given average velocity and opening dimensions. -/
theorem mass_flow_rate (v A : ℝ) (v_10_1 : v = 10.1) (A_0_6 : A = 0.4 * 1.5) (rho : ℝ) (rho_1000 : rho = 1000) :
    ρ * A * v = 6060 :=
by
  sorry

/-- Available horsepower through turbines given mass flow rate and average velocity. -/
theorem available_horsepower (m v : ℝ) (m_6060 : m = 6060) (v_10_1 : v = 10.1 ) (hp : ℝ)
    (hp_735_5 : hp = 735.5 ) :
    (1 / 2) * m * v^2 / hp = 420 :=
by
  sorry

end average_velocity_mass_flow_rate_available_horsepower_l482_482133


namespace james_paid_amount_l482_482180

def total_stickers (packs : ℕ) (stickers_per_pack : ℕ) : ℕ :=
  packs * stickers_per_pack

def total_cost (num_stickers : ℕ) (cost_per_sticker : ℕ) : ℕ :=
  num_stickers * cost_per_sticker

def half_cost (total_cost : ℕ) : ℕ :=
  total_cost / 2

theorem james_paid_amount :
  let packs : ℕ := 4,
      stickers_per_pack : ℕ := 30,
      cost_per_sticker : ℕ := 10,  -- Using cents for simplicity to avoid decimals
      friend_share : ℕ := 2,
      num_stickers := total_stickers packs stickers_per_pack,
      total_amt := total_cost num_stickers cost_per_sticker,
      james_amt := half_cost total_amt
  in
  james_amt = 600 :=
by
  sorry

end james_paid_amount_l482_482180


namespace infinite_solutions_of_diophantine_l482_482642

theorem infinite_solutions_of_diophantine 
  (n : ℕ) 
  (A : fin (n + 1) → ℕ)
  (hA_pos : ∀ i, 0 < A i)
  (hGCD : ∀ i : fin n, Nat.gcd (A i) (A (fin.last n)) = 1) :
  ∃∞ (x : fin (n + 1) → ℕ), ∀ i, 0 < x i ∧ 
  (∑ i : fin n, (x i) ^ A i) = (x (fin.last n)) ^ A (fin.last n) :=
sorry

end infinite_solutions_of_diophantine_l482_482642


namespace last_digit_of_one_over_two_pow_twelve_l482_482764

theorem last_digit_of_one_over_two_pow_twelve : 
  let x : ℚ := 1 / 2^12 in (x * 10^12).den = 244140625 → (x.toReal - floor x.toReal) * 10 ^ 12 = 244140625 :=
by
  sorry

end last_digit_of_one_over_two_pow_twelve_l482_482764


namespace geometric_sequence_an_plus_half_sum_of_cn_l482_482531

noncomputable def an (n : ℕ) : ℕ :=
if h : n > 0 then 3^(n-1) / 2 else 1  -- given a_1 = 1 and recurrence relation

noncomputable def bn (n : ℕ) : ℕ :=
1 + (n - 1)  -- arithmetic sequence with initial term 1 and common difference 1

noncomputable def cn (n : ℕ) : ℕ :=
an n * bn n  -- definition of c_n = a_n * b_n

noncomputable def Tn (n : ℕ) : ℕ :=
∑ i in Finset.range n, cn (i + 1)  -- sum of the first n terms of the sequence {c_n}

theorem geometric_sequence_an_plus_half :
  ∃ r a₀, ∀ n, an n + 1/2 = a₀ * r ^ n :=
sorry

theorem sum_of_cn :
  ∀ n, Tn n = (2*(n-1)/8) * 3^(n+1) + 3/8 - (n*(n+1)/4) :=
sorry

end geometric_sequence_an_plus_half_sum_of_cn_l482_482531


namespace not_mutually_exclusive_probability_both_segments_success_expectation_successful_segments_conditional_prob_three_segments_given_l482_482491

open Classical

-- Condition definitions
variable (p : ℝ) (n : ℕ)
def segment_success_prob : ℝ := 3 / 4
def num_segments : ℕ := 4

noncomputable def prob_of_two_segments_success := (segment_success_prob * segment_success_prob : ℝ)
noncomputable def expected_successful_segments := num_segments * segment_success_prob
noncomputable def prob_three_successful_and_one_specific :=
  (3 / 4) ^ 3 * (1 / 4) * (3 choose 2)
noncomputable def exactly_three_successful :=
  (4 choose 3) * (3 / 4) ^ 3 * (1 / 4)
noncomputable def conditional_prob_welcoming_success :=
  prob_three_successful_and_one_specific / exactly_three_successful

theorem not_mutually_exclusive : ¬(prob_of_two_segments_success = 0) :=
sorry

theorem probability_both_segments_success :
  prob_of_two_segments_success = 9 / 16 :=
sorry

theorem expectation_successful_segments :
  expected_successful_segments = 3 :=
sorry

theorem conditional_prob_three_segments_given :
  conditional_prob_welcoming_success = 3 / 4 :=
sorry

end not_mutually_exclusive_probability_both_segments_success_expectation_successful_segments_conditional_prob_three_segments_given_l482_482491


namespace solve_y_l482_482719

theorem solve_y (y : ℤ) (h : 7 - y = 10) : y = -3 := by
  sorry

end solve_y_l482_482719


namespace anya_probability_l482_482398

open Finset

def possible_coins := {10, 10, 5, 5, 2}
def target := 19

noncomputable def combinations := (possible_coins.vals.ctype_power 3).val.filter (λ s, Finset.sum s >= target)

noncomputable def probability : ℝ :=
  (combinations.card : ℝ) / (possible_coins.vals.ctype_power 3).card

theorem anya_probability : probability = 0.4 := sorry

end anya_probability_l482_482398


namespace part1_min_max_part2_value_of_a_l482_482340

-- Part 1: Maximum and Minimum Values of the Quadratic Function
theorem part1_min_max :
  let y := λ x : ℝ, 2 * x^2 - 3 * x + 5 in
  (∀ x ∈ Icc (-2 : ℝ) (2 : ℝ), y x ≥ (31/8 : ℝ)) ∧
  (∃ x ∈ Icc (-2 : ℝ) (2 : ℝ), y x = (31/8 : ℝ)) ∧
  (∀ x ∈ Icc (-2 : ℝ) (2 : ℝ), y x ≤ 19) ∧
  (∃ x ∈ Icc (-2 : ℝ) (2 : ℝ), y x = 19) :=
by unfold Icc; sorry

-- Part 2: Value of 'a' for the given maximum value of the function
theorem part2_value_of_a :
  ∀ (a : ℝ),
  let y := λ x : ℝ, x^2 + 2 * a * x + 1 in
  (∀ x ∈ Icc (-1 : ℝ) (2 : ℝ), y x ≤ 4) →
  (∃ x ∈ Icc (-1 : ℝ) (2 : ℝ), y x = 4) →
  (a = -1 ∨ a = -1/4) :=
by unfold Icc; sorry

end part1_min_max_part2_value_of_a_l482_482340


namespace b_is_zero_if_on_real_axis_l482_482604

-- Defining the complex number as 1 + b*i where b is a real number
def complex_number (b : ℝ) : ℂ := 1 + b * complex.I

-- The main theorem to prove: if the complex_number lies on the real axis, then b = 0
theorem b_is_zero_if_on_real_axis (b : ℝ) :
  (complex_number b).im = 0 → b = 0 :=
by
    sorry

end b_is_zero_if_on_real_axis_l482_482604


namespace initial_candies_l482_482831

theorem initial_candies (eaten left : ℕ) (h1 : eaten = 15) (h2 : left = 13) : 
  eaten + left = 28 := by
  rw [h1, h2]
  exact rfl

end initial_candies_l482_482831


namespace sum_smallest_largest_primes_l482_482176

theorem sum_smallest_largest_primes : 
  let primes : Set ℕ := { p ∈ Set.range 50 | Nat.prime p } in
  (2 ∈ primes) → (47 ∈ primes) → (∀ p ∈ primes, p ≥ 2) → (∀ p ∈ primes, p ≤ 47) → 2 + 47 = 49 :=
by
  intros primes h1 h2 h3 h4
  sorry

end sum_smallest_largest_primes_l482_482176


namespace geometric_mean_triangle_areas_l482_482352

variable {V : Type*} [inner_product_space ℝ V]

-- Definitions of the points and the cube vertices
variable (A B C D P Q R S : V)

-- Conditions: P, Q, and R are points on the edges of the cube defined by A, B, C, D
def on_edges_of_cube (A B C D P Q R : V) : Prop :=
  -- P is on edge AB
  P ∈ segment ℝ A B ∧
  -- Q is on edge AC
  Q ∈ segment ℝ A C ∧
  -- R is on edge AD
  R ∈ segment ℝ A D

-- S is the orthogonal projection of A onto the plane PQR
def orthogonal_projection (plane : set V) (A S : V) :=
  orthogonal_projection_fn plane A = S

-- Definition of the area of a triangle PQR
def triangle_area (P Q R : V) : ℝ :=
  1/2 * (P - Q).norm * (P - R).norm * sin (angle_between (P - Q) (P - R))

-- The statement to be proven
theorem geometric_mean_triangle_areas
  (h1 : on_edges_of_cube A B C D P Q R)
  (h2 : orthogonal_projection (affine_span ℝ ({P, Q, R} : set V)) A S) :
  triangle_area P Q A = real.sqrt (triangle_area P Q S * triangle_area P Q R) :=
sorry

end geometric_mean_triangle_areas_l482_482352


namespace chef_michel_total_pies_l482_482863

theorem chef_michel_total_pies 
  (shepherd_pie_pieces : ℕ) 
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ) 
  (chicken_pot_pie_customers : ℕ) 
  (h1 : shepherd_pie_pieces = 4)
  (h2 : chicken_pot_pie_pieces = 5)
  (h3 : shepherd_pie_customers = 52)
  (h4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) +
  (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by {
  sorry
}

end chef_michel_total_pies_l482_482863


namespace book_club_meeting_days_l482_482184

theorem book_club_meeting_days :
  Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 (Nat.lcm 9 10)) = 360 := 
by sorry

end book_club_meeting_days_l482_482184


namespace sequence_limit_l482_482866

theorem sequence_limit : 
  (∃ l : ℝ, tendsto (λ n, ((∑ k in (finset.range n).filter (λ x, odd x).val) / (∑ k in finset.range n)) at_top (𝓝 l)) ∧ l = 2 :=
by 
suffices H₀ : ∑ k in (finset.range n).filter (λ x, odd x).val = n^2, sorry,
suffices H₁ : ∑ k in finset.range n = n * (n + 1) / 2, sorry,
exact ⟨2, by simpa using sequence_limit⟩

end sequence_limit_l482_482866


namespace inscribed_circle_radius_l482_482306

theorem inscribed_circle_radius (AB AC BC : ℝ) (h1 : AB = 6) (h2 : AC = 8) (h3 : BC = 10) : 
  (let s := (AB + AC + BC) / 2 in
   let K := Math.sqrt (s * (s - AB) * (s - AC) * (s - BC)) in
   let r := K / s in
   r = 2) :=
  sorry

end inscribed_circle_radius_l482_482306


namespace area_of_quadrilateral_AMDN_eq_area_of_triangle_ABC_l482_482401

-- Definitions

-- Type of points used in geometry
variable {Point : Type}

-- Functions for the existence of specific points and geometric properties
variables {A B C : Point} -- vertices of the triangle
variable {E F M N D : Point} -- specific points mentioned in the problem

-- Lines and angles
variable BC : Line -- Line BC
variable AE : Line -- Line AE
variable circumcircle_of_ABC : Circle -- Circumcircle of triangle ABC
variable perp_AB : Perpendicular F M A B -- Perpendicular from F to AB with foot M
variable perp_AC : Perpendicular F N A C -- Perpendicular from F to AC with foot N

-- Definitions for geometric properties
variable acute_triangle_ABC : AcuteTriangle A B C
variable angle_BAE_eq_angle_CAF : Angle B A E = Angle C A F

-- Definition and intersection properties
variable intersection_AE_circumcircle : IntersectionPoints AE circumcircle_of_ABC D

-- Areas of quadrilateral and triangle
variable area_AMDN : ℝ -- Area of quadrilateral AMDN
variable area_ABC : ℝ -- Area of triangle ABC

-- Theorem statement
theorem area_of_quadrilateral_AMDN_eq_area_of_triangle_ABC
  (acute_triangle_ABC : AcuteTriangle A B C)
  (on_BC_EF : E ∈ BC ∧ F ∈ BC)
  (angle_BAE_eq_angle_CAF : Angle B A E = Angle C A F)
  (perpendicular_FM : Perpendicular F M A B)
  (perpendicular_FN : Perpendicular F N A C)
  (intersection_AE_D : D ∈ circumcircle_of_ABC ∧ AE = line A E)
  : area_AMDN = area_ABC := sorry

end area_of_quadrilateral_AMDN_eq_area_of_triangle_ABC_l482_482401


namespace area_of_ABCD_l482_482668

noncomputable def AB := 6
noncomputable def BC := 8
noncomputable def CD := 15
noncomputable def DA := 17
def right_angle_BCD := true
def convex_ABCD := true

theorem area_of_ABCD : ∃ area : ℝ, area = 110 := by
  -- Given conditions
  have hAB : AB = 6 := rfl
  have hBC : BC = 8 := rfl
  have hCD : CD = 15 := rfl
  have hDA : DA = 17 := rfl
  have hAngle : right_angle_BCD = true := rfl
  have hConvex : convex_ABCD = true := rfl

  -- skip the proof
  sorry

end area_of_ABCD_l482_482668


namespace units_digit_sum_is_9_l482_482313

-- Define the units function
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions
def x := 42 ^ 2
def y := 25 ^ 3

-- Define variables for the units digits of x and y
def units_digit_x := units_digit x
def units_digit_y := units_digit y

-- Define the problem statement to be proven
theorem units_digit_sum_is_9 : units_digit (x + y) = 9 :=
by sorry

end units_digit_sum_is_9_l482_482313


namespace cannot_be_covered_by_dominoes_l482_482876

-- Definitions for each board
def board_3x4_squares : ℕ := 3 * 4
def board_3x5_squares : ℕ := 3 * 5
def board_4x4_one_removed_squares : ℕ := 4 * 4 - 1
def board_5x5_squares : ℕ := 5 * 5
def board_6x3_squares : ℕ := 6 * 3

-- Parity check
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Mathematical proof problem statement
theorem cannot_be_covered_by_dominoes :
  ¬ is_even board_3x5_squares ∧
  ¬ is_even board_4x4_one_removed_squares ∧
  ¬ is_even board_5x5_squares :=
by
  -- Checking the conditions that must hold
  sorry

end cannot_be_covered_by_dominoes_l482_482876


namespace tim_balloons_proof_l482_482022

-- Define the number of balloons Dan has
def dan_balloons : ℕ := 29

-- Define the relationship between Tim's and Dan's balloons
def balloons_ratio : ℕ := 7

-- Define the number of balloons Tim has
def tim_balloons : ℕ := balloons_ratio * dan_balloons

-- Prove that the number of balloons Tim has is 203
theorem tim_balloons_proof : tim_balloons = 203 :=
sorry

end tim_balloons_proof_l482_482022


namespace solve_y_determinant_l482_482908

theorem solve_y_determinant (b y : ℝ) (hb : b ≠ 0) :
  Matrix.det ![
    ![y + b, y, y], 
    ![y, y + b, y], 
    ![y, y, y + b]
  ] = 0 ↔ y = -b / 3 :=
by
  sorry

end solve_y_determinant_l482_482908


namespace probability_density_function_l482_482941
open Real

/-- Define the function f(x) as given in the conditions. -/
def f (c α x : ℝ) : ℝ :=
  if x ≤ 0 then 0 else c * exp (-α * x)

/-- Define α > 0 as a condition. -/
axiom alpha_positive (α : ℝ) : α > 0

/--
  Prove that the function f(x) is a probability density function when c = α.
  Specifically, prove that the integral of f(x) over its entire range equals 1.
-/
theorem probability_density_function (c α : ℝ) (hα : α > 0) : (∫ x in -∞..∞, f c α x) = 1 ↔ c = α :=
  sorry

end probability_density_function_l482_482941


namespace number_of_students_scored_at_least_120_l482_482602

noncomputable theory
open_locale classical

variables {a : ℝ} {N : ℕ}
-- conditions
def normal_distribution (μ σ : ℝ) (x : ℝ) : ℝ :=
(1 / (σ * real.sqrt (2 * real.pi))) * real.exp (-((x - μ)^2) / (2 * σ^2))

-- The scores xi follow a normal distribution N(100, a^2) 
def scores_distribution : Prop := ∀ x, normal_distribution 100 a x

-- Approximately 60% of students scored between 80 and 120
def score_probability : Prop := real.integral (λ x, normal_distribution 100 a x) 80 120 = 0.6

-- Total number of students
def total_students : ℕ := 1000

-- Question: How many students scored at least 120?
def students_at_least_120 : ℕ := (0.2 * total_students).to_nat

theorem number_of_students_scored_at_least_120 :
  scores_distribution → score_probability → 
  students_at_least_120 = 200 :=
sorry

end number_of_students_scored_at_least_120_l482_482602


namespace p_is_necessary_but_not_sufficient_for_q_l482_482073

variable (x : ℝ)

def p : Prop := -1 ≤ x ∧ x ≤ 5
def q : Prop := (x - 5) * (x + 1) < 0

theorem p_is_necessary_but_not_sufficient_for_q : (∀ x, p x → q x) ∧ ¬ (∀ x, q x → p x) := 
sorry

end p_is_necessary_but_not_sufficient_for_q_l482_482073


namespace determine_x_l482_482874

theorem determine_x (x : ℚ) (h : ∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - (9 / 2) = 0) : x = 3 / 2 :=
sorry

end determine_x_l482_482874


namespace area_triangle_ABC_specific_values_l482_482171

/-- Triangle ABC with specified angles and side lengths -/
def triangle_ABC_area (a b : ℝ) (A B C : ℝ) (sin : ℝ → ℝ) : ℝ :=
  (1/2) * b * a * sin A

theorem area_triangle_ABC_specific_values :
  triangle_ABC_area 15 7 (80 / 180 * Real.pi) (60 / 180 * Real.pi) (40 / 180 * Real.pi) Real.sin ≈ 51.702 :=
by
  -- Specifications:
  -- Angle A = 80 degrees (converted to radians as 80 / 180 * π)
  -- Angle B = 60 degrees (converted to radians as 60 / 180 * π)
  -- Angle C = 40 degrees (converted to radians as 40 / 180 * π)
  -- Side a = 15 cm
  -- Side b = 7 cm
  -- Using the sine of 80 degrees

  -- The converted radians and sin are already used within the theorem call.
  sorry

end area_triangle_ABC_specific_values_l482_482171


namespace max_side_parallel_to_barn_correct_l482_482802

noncomputable def maximizeArea : Real :=
let cost_per_foot := 5
let total_cost := 1500
let total_length := total_cost / cost_per_foot
let area (x : ℝ) := x * (total_length - 2 * x)
let x := deriv area x
let critical_point := 75
let side_parallel_to_barn := total_length - 2 * critical_point
side_parallel_to_barn

theorem max_side_parallel_to_barn_correct :
  maximizeArea = 150 :=
by
  sorry

end max_side_parallel_to_barn_correct_l482_482802


namespace sqrt_sum_simplify_l482_482674

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l482_482674


namespace n_is_even_l482_482337

variable (n : ℕ) (B : Type*) (A : Fin (2 * n + 1) → Set B)

-- Condition 1: Each A_i has exactly 2n elements
def each_has_2n_elements := ∀ i, (A i).card = 2 * n

-- Condition 2: The intersection of every two distinct A_i contains exactly one element
def intersection_contains_one_element := 
  ∀ i j, i ≠ j → (A i ∩ A j).card = 1

-- Condition 3: Every element of B belongs to at least two of the A_i
def element_belongs_to_at_least_two := 
  ∀ b : B, (Finset.card {i : Fin (2 * n + 1) | b ∈ A i}) ≥ 2

theorem n_is_even
  (h1 : each_has_2n_elements n B A)
  (h2 : intersection_contains_one_element n B A)
  (h3 : element_belongs_to_at_least_two n B A) :
  Even n :=
sorry

end n_is_even_l482_482337


namespace number_of_students_know_secret_on_sunday_l482_482651

noncomputable def S (n : ℕ) : ℕ :=
  (3^(n + 1) - 1) / 2

theorem number_of_students_know_secret_on_sunday :
  ∃ n : ℕ, S n = 3280 ∧ n = 7 :=
by
  use 7
  split
  · exact congr_arg (λ n, (3^(n + 1) - 1) / 2) (nat.succ_eq_add_one 7)
  · rfl

end number_of_students_know_secret_on_sunday_l482_482651


namespace age_difference_l482_482832

variables (O N A : ℕ)

theorem age_difference (avg_age_stable : 10 * A = 10 * A + 50 - O + N) :
  O - N = 50 :=
by
  -- proof would go here
  sorry

end age_difference_l482_482832


namespace axis_of_symmetry_real_solutions_condition_l482_482086

-- Definitions given the function f(x)
def f (x : ℝ) : ℝ := 2 * real.cos x * real.sin (x - real.pi / 6) + 1 / 2

-- Theorem for part (1)
theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, f(x) = f(2 * x - (k * real.pi + real.pi / 3)) :=
sorry

-- Theorem for part (2)
theorem real_solutions_condition (m : ℝ) :
  (∃ x : ℝ, x ∈ set.Icc (-real.pi / 3) (real.pi / 2) ∧ 
    real.sin (2 * x) + 2 * abs (f (x + real.pi / 12)) - m + 1 = 0) →
  m = 2 ∨ (1 < m ∧ m < 1 + real.sqrt 3 / 2) :=
sorry

end axis_of_symmetry_real_solutions_condition_l482_482086


namespace choose_most_suitable_l482_482775

def Survey := ℕ → Bool
structure Surveys :=
  (A B C D : Survey)
  (census_suitable : Survey)

theorem choose_most_suitable (s : Surveys) :
  s.census_suitable = s.C :=
sorry

end choose_most_suitable_l482_482775


namespace number_of_points_on_parabola_l482_482448
noncomputable theory

def parabola_points : ℕ → ℕ → Prop := 
  λ x y, y = 33 - x^2 / 9

theorem number_of_points_on_parabola :
  (finset.univ.filter (λ (xy : ℕ × ℕ), parabola_points xy.1 xy.2)).card = 5 := 
sorry

end number_of_points_on_parabola_l482_482448


namespace min_value_f_on_interval_l482_482982

noncomputable def f : ℝ → ℝ := λ x, x^2 - 2 * x

theorem min_value_f_on_interval : 
  ∀ x ∈ set.Icc 0 3, f x ≥ -1 ∧ (∃ y ∈ set.Icc 0 3, f y = -1) :=
by
  sorry

end min_value_f_on_interval_l482_482982


namespace triangle_lattice_points_l482_482245

-- Given lengths of the legs of the right triangle
def DE : Nat := 15
def EF : Nat := 20

-- Calculate the hypotenuse using the Pythagorean theorem
def DF : Nat := Nat.sqrt (DE ^ 2 + EF ^ 2)

-- Calculate the area of the triangle
def Area : Nat := (DE * EF) / 2

-- Calculate the number of boundary points
def B : Nat :=
  let points_DE := DE + 1
  let points_EF := EF + 1
  let points_DF := DF + 1
  points_DE + points_EF + points_DF - 3

-- Calculate the number of interior points using Pick's Theorem
def I : Int := Area - (B / 2 - 1)

-- Calculate the total number of lattice points
def total_lattice_points : Int := I + Int.ofNat B

-- The theorem statement
theorem triangle_lattice_points : total_lattice_points = 181 := by
  -- The actual proof goes here
  sorry

end triangle_lattice_points_l482_482245


namespace polynomial_equality_l482_482900

theorem polynomial_equality (x : ℝ) : 
  x * (x * (x * (3 - x) - 3) + 5) + 1 = -x^4 + 3*x^3 - 3*x^2 + 5*x + 1 :=
by 
  sorry

end polynomial_equality_l482_482900


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482463

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let expr := 7 ^ 7 - 7 ^ 4 in
  (7^4 * (7^3 - 1) = expr) ∧ (7^3 - 1 = 342) ∧ (Prime 2) ∧ (Prime 3) ∧ (Prime 7) ∧ (Prime 19) ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ 342 → p = 2 ∨ p = 3 ∨ p = 19) → 
  (∀ p : ℕ, Nat.Prime p → p ∣ expr → p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19) → 
  (2 + 3 + 7 + 19 = 31) := 
by
  intro expr fact1 fact2 prime2 prime3 prime7 prime19 factors342 factorsExpr
  sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482463


namespace domain_of_function_l482_482264

theorem domain_of_function {x : ℝ} :
  (x ≥ 0) → (sqrt x ≠ 1) → (x ∈ (Set.Ici 0 \ {1})) :=
by
  intros hx hneq1
  sorry

end domain_of_function_l482_482264


namespace arrangement_of_representatives_l482_482297

theorem arrangement_of_representatives:
  (A B : Type) (a_reps : Finset A) (b_reps : Finset B) (h₁ : a_reps.card = 7) (h₂ : b_reps.card = 3) :
  ∃ (n : Nat), n = 8! * 3! ∧ n = 241920 :=
by
  sorry

end arrangement_of_representatives_l482_482297


namespace find_k_for_one_real_solution_l482_482497

theorem find_k_for_one_real_solution : ∃ k : ℚ, (∀ x : ℝ, (x + 3) * (x + 2) = k + 3 * x) → k = 5 := 
by
  -- We begin by assuming the condition and then deriving the necessary conclusions.
  intro h,
  -- Establish the proof that k = 5 ensures the equation has exactly one real solution.
  sorry

end find_k_for_one_real_solution_l482_482497


namespace neg_p_false_sufficient_but_not_necessary_for_p_or_q_l482_482958

variable (p q : Prop)

theorem neg_p_false_sufficient_but_not_necessary_for_p_or_q :
  (¬ p = false) → (p ∨ q) ∧ ¬((p ∨ q) → (¬ p = false)) :=
by
  sorry

end neg_p_false_sufficient_but_not_necessary_for_p_or_q_l482_482958


namespace part_A_part_B_l482_482623

-- Definitions for the setup
variables (d : ℝ) (n : ℕ) (d_ne_0 : d ≠ 0)

-- Part (A): Specific distance 5d
theorem part_A (d : ℝ) (d_ne_0 : d ≠ 0) : 
  (∀ (x y : ℝ), x^2 + y^2 = 25 * d^2 ∧ |y - d| = 5 * d → 
  (x = 3 * d ∧ y = -4 * d) ∨ (x = -3 * d ∧ y = -4 * d)) :=
sorry

-- Part (B): General distance nd
theorem part_B (d : ℝ) (n : ℕ) (d_ne_0 : d ≠ 0) : 
  (∀ (x y : ℝ), x^2 + y^2 = (n * d)^2 ∧ |y - d| = n * d → ∃ x y, (x^2 + y^2 = (n * d)^2 ∧ |y - d| = n * d)) :=
sorry

end part_A_part_B_l482_482623


namespace yellow_3x3_exclusion_probability_l482_482440

theorem yellow_3x3_exclusion_probability :
  ∃ (m n : ℕ), m + n = 130562 ∧ 
  Nat.coprime m n ∧
  let total_ways := 2^16,
      exclusion_ways := 65536 - 510 in
  m = exclusion_ways ∧ n = total_ways :=
begin
  sorry
end

end yellow_3x3_exclusion_probability_l482_482440


namespace problem1_problem2_l482_482083

-- Problem (1)
-- Given ellipse 5x^2 + 9y^2 = 45
-- and a line passing through its right focus point F with a slope of 1
def ellipse : Prop := ∀ x y : ℝ, 5 * x^2 + 9 * y^2 = 45
def line_through_focus (F : ℝ × ℝ) (slope : ℝ) : Prop := ∀ x y : ℝ, y = x - 2
def chord_length_correct : Prop := 
  ∃ F : ℝ × ℝ, length_of_chord ellipse (line_through_focus F 1) = 30 / 7

-- Problem (2)
-- Given point A(1,1), the point is on the midpoint of the chord
def point_A : (ℝ × ℝ) := (1, 1)
def midpoint_chord (A : ℝ × ℝ) (ellipse : Prop) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, 
    ellipse x1 y1 → ellipse x2 y2 →
    2 * A.1 = x1 + x2 ∧ 2 * A.2 = y1 + y2 →
    (5 * x + 9 * y = 14)

theorem problem1 (ellipse : ∀x y : ℝ, 5 * x^2 + 9 * y^2 = 45) (F : ℝ × ℝ) (slope : ℝ) : 
  line_through_focus F 1 → chord_length_correct :=
begin
  sorry
end

theorem problem2 (ellipse : ∀ x1 y1 x2 y2 : ℝ, 5 * x1^2 + 9 * y1^2 = 45 ∧ 5 * x2^2 + 9 * y2^2 = 45) : 
  midpoint_chord point_A ellipse :=
begin
  sorry
end

end problem1_problem2_l482_482083


namespace james_paid_amount_l482_482179

def total_stickers (packs : ℕ) (stickers_per_pack : ℕ) : ℕ :=
  packs * stickers_per_pack

def total_cost (num_stickers : ℕ) (cost_per_sticker : ℕ) : ℕ :=
  num_stickers * cost_per_sticker

def half_cost (total_cost : ℕ) : ℕ :=
  total_cost / 2

theorem james_paid_amount :
  let packs : ℕ := 4,
      stickers_per_pack : ℕ := 30,
      cost_per_sticker : ℕ := 10,  -- Using cents for simplicity to avoid decimals
      friend_share : ℕ := 2,
      num_stickers := total_stickers packs stickers_per_pack,
      total_amt := total_cost num_stickers cost_per_sticker,
      james_amt := half_cost total_amt
  in
  james_amt = 600 :=
by
  sorry

end james_paid_amount_l482_482179


namespace solve_y_determinant_l482_482907

theorem solve_y_determinant (b y : ℝ) (hb : b ≠ 0) :
  Matrix.det ![
    ![y + b, y, y], 
    ![y, y + b, y], 
    ![y, y, y + b]
  ] = 0 ↔ y = -b / 3 :=
by
  sorry

end solve_y_determinant_l482_482907


namespace part1_part2_l482_482536

-- We define the function f and its derivative f' in terms of the unknowns a and b.
def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b * x
def f' (x : ℝ) (a b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

-- g is defined as f - f'.
def g (x : ℝ) (a b : ℝ) : ℝ := f x a b - f' x a b

-- condition that g is an odd function
def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = - (h x)

-- Given a specific f, and x in [1, 3], find max and min of g(x) = x^3 - 6x
def f_specific (x : ℝ) : ℝ := x^3 + 3 * x^2
def g_specific (x : ℝ) : ℝ := x^3 - 6 * x

-- Statement of the mathematical problems
theorem part1 (a b : ℝ) (h : is_odd (g x a b)) : f x a b = x^3 + 3 * x^2 :=
sorry

theorem part2 : (1 : ℝ) ≤ x ∧ x ≤ 3 → ∃ (max min : ℝ),
  max = g_specific 3 ∧ min = g_specific (sqrt 2) :=
sorry

end part1_part2_l482_482536


namespace not_mutually_exclusive_probability_both_segments_success_expectation_successful_segments_conditional_prob_three_segments_given_l482_482490

open Classical

-- Condition definitions
variable (p : ℝ) (n : ℕ)
def segment_success_prob : ℝ := 3 / 4
def num_segments : ℕ := 4

noncomputable def prob_of_two_segments_success := (segment_success_prob * segment_success_prob : ℝ)
noncomputable def expected_successful_segments := num_segments * segment_success_prob
noncomputable def prob_three_successful_and_one_specific :=
  (3 / 4) ^ 3 * (1 / 4) * (3 choose 2)
noncomputable def exactly_three_successful :=
  (4 choose 3) * (3 / 4) ^ 3 * (1 / 4)
noncomputable def conditional_prob_welcoming_success :=
  prob_three_successful_and_one_specific / exactly_three_successful

theorem not_mutually_exclusive : ¬(prob_of_two_segments_success = 0) :=
sorry

theorem probability_both_segments_success :
  prob_of_two_segments_success = 9 / 16 :=
sorry

theorem expectation_successful_segments :
  expected_successful_segments = 3 :=
sorry

theorem conditional_prob_three_segments_given :
  conditional_prob_welcoming_success = 3 / 4 :=
sorry

end not_mutually_exclusive_probability_both_segments_success_expectation_successful_segments_conditional_prob_three_segments_given_l482_482490


namespace cannot_change_all_signs_l482_482157

open Set

-- Define conditions for a regular decagon with diagonals and intersection points.
def regular_decagon : Type := sorry  -- This would usually be a formal geometric definition

-- Assume each vertex and intersection point of diagonals has a +1 placed
def has_pos_1 (d : regular_decagon) : Prop := sorry  -- Formal definition needed

-- Define operation allowed: changing signs on one side or along one diagonal
inductive Operation
| side (s : regular_decagon) : Operation
| diagonal (d : regular_decagon) : Operation

def apply_operation (op : Operation) (places : regular_decagon → ℤ) : regular_decagon → ℤ :=
  sorry  -- would define how the operation changes signs

-- The main theorem
theorem cannot_change_all_signs :
  ¬ (∃ ops : list Operation, ∀ d, has_pos_1 (apply_operations ops d) = -has_pos_1 d) :=
  sorry

end cannot_change_all_signs_l482_482157


namespace Vasya_Tolya_cannot_win_l482_482282

theorem Vasya_Tolya_cannot_win (total_coins : ℕ) (Petya_moves Vasya_moves Tolya_moves : ℕ → ℕ → Prop)
  (h_initial_coins : total_coins = 300)
  (h_Petya_moves : ∀ n (h : 1 ≤ n ∧ n ≤ 4), Petya_moves n)
  (h_Vasya_moves : ∀ n (h : 1 ≤ n ∧ n ≤ 2), Vasya_moves n)
  (h_Tolya_moves : ∀ n (h : 1 ≤ n ∧ n ≤ 2), Tolya_moves n) :
  ¬ (∀ Petya_strategy Vasya_strategy Tolya_strategy, 
        (∀ turn coins_left,
          (turn % 3 = 0 → Petya_strategy turn coins_left ∈ Petya_moves ∧
           turn % 3 = 1 → Vasya_strategy turn coins_left ∈ Vasya_moves ∧
           turn % 3 = 2 → Tolya_strategy turn coins_left ∈ Tolya_moves) →
          ∃ turn_last, turn_last % 3 = 1 ∨ turn_last % 3 = 2 ∧
            Petya_strategy turn_last coins_left = 1)) := sorry

end Vasya_Tolya_cannot_win_l482_482282


namespace simplify_radicals_l482_482700

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l482_482700


namespace sqrt_sum_simplify_l482_482692

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l482_482692


namespace math_problem_l482_482058

variable (a b c : ℤ)

theorem math_problem
  (h₁ : 3 * a + 4 * b + 5 * c = 0)
  (h₂ : |a| = 1)
  (h₃ : |b| = 1)
  (h₄ : |c| = 1) :
  a * (b + c) = - (3 / 5) :=
sorry

end math_problem_l482_482058


namespace probability_of_at_least_19_l482_482395

-- Defining the possible coins in Anya's pocket
def coins : list ℕ := [10, 10, 5, 5, 2]

-- Function to calculate the sum of chosen coins
def sum_coins (l : list ℕ) := list.sum l

-- Function to check if the sum of chosen coins is at least 19 rubles
def at_least_19 (l : list ℕ) := (sum_coins l) ≥ 19

-- Extract all possible combinations of 3 coins from the list
def combinations (l : list ℕ) (n : ℕ) := 
  if h : n ≤ l.length then 
    (list.permutations l).dedup.map (λ p, p.take n).dedup
  else
    []

-- Specific combinations of 3 coins out of 5
def three_coin_combinations := combinations coins 3 

-- Count the number of favorable outcomes (combinations that sum to at least 19)
def favorable_combinations := list.filter at_least_19 three_coin_combinations

-- Calculate the probability
def probability := (favorable_combinations.length : ℚ) / (three_coin_combinations.length : ℚ)

-- Prove that the probability is 0.4
theorem probability_of_at_least_19 : probability = 0.4 :=
  sorry

end probability_of_at_least_19_l482_482395


namespace solve_for_x_l482_482769

theorem solve_for_x :
  ∃ x : ℝ, (24 / 36) = Real.sqrt (x / 36) ∧ x = 16 :=
by
  use 16
  sorry

end solve_for_x_l482_482769


namespace geom_equality_l482_482588

-- Define the geometric entities and conditions
variables (A B C P E H F I G J Q : Type)
variables [is_triangle A B C] [right_angle C] (P : on_altitude A B C) (E H F I G J Q : point)
variables [pe_parallel_BC : parallel_line E P B C]
          [ph_parallel_AB : parallel_line H P A B]
          [pg_parallel_CA : parallel_line G P C A]
          [qe_eq_qf : circle Q E Q F] [qf_eq_qg : circle Q F Q G]
          
-- The theorem statement we want to prove
theorem geom_equality : QG = GH ∧ QI = QJ ∧ QH = QI ∧ QI = QJ :=
by
  -- Skip the proof with sorry
  sorry

end geom_equality_l482_482588


namespace range_of_ab_l482_482098

theorem range_of_ab (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : (2:ℝ)^a * (2:ℝ)^b = 2) : ab ∈ set.Ioo (0:ℝ) ((1:ℝ)/4) ∨ ab = (1:ℝ)/4 :=
by
  sorry

end range_of_ab_l482_482098


namespace Bethany_total_riding_hours_l482_482000

-- Define daily riding hours
def Monday_hours : Nat := 1
def Wednesday_hours : Nat := 1
def Friday_hours : Nat := 1
def Tuesday_hours : Nat := 1 / 2
def Thursday_hours : Nat := 1 / 2
def Saturday_hours : Nat := 2

-- Define total weekly hours
def weekly_hours : Nat :=
  Monday_hours + Wednesday_hours + Friday_hours + (Tuesday_hours + Thursday_hours) + Saturday_hours

-- Definition to account for the 2-week period
def total_hours (weeks : Nat) : Nat := weeks * weekly_hours

-- Prove that Bethany rode 12 hours over 2 weeks
theorem Bethany_total_riding_hours : total_hours 2 = 12 := by
  sorry

end Bethany_total_riding_hours_l482_482000


namespace justin_received_10_fewer_l482_482417

theorem justin_received_10_fewer (total_stickers : ℕ)
    (gave_to_each_friend : ℕ) (number_of_friends : ℕ) (extra_for_mandy : ℕ)
    (leftover_stickers : ℕ) 
    (Ht : total_stickers = 72)
    (Hgf : gave_to_each_friend = 4) 
    (Hnf : number_of_friends = 3)
    (Hem : extra_for_mandy = 2)
    (Hl : leftover_stickers = 42) :
  let stickers_given_to_friends := gave_to_each_friend * number_of_friends in
  let stickers_given_to_mandy := stickers_given_to_friends + extra_for_mandy in
  let stickers_given_total := stickers_given_to_friends + stickers_given_to_mandy in
  let stickers_left_computed := total_stickers - stickers_given_total in
  let stickers_given_to_justin := stickers_left_computed - leftover_stickers in
  stickers_given_to_mandy - stickers_given_to_justin = 10 :=
by
  sorry

end justin_received_10_fewer_l482_482417


namespace part1_part2_l482_482090

-- Define the functions and requirements
noncomputable def f (x : ℝ) (a : ℝ) := x + a * real.log x
noncomputable def g (x : ℝ) (b : ℝ) := real.log x + (1 / 2) * x^2 - (b - 1) * x

-- Define the conditions and statements
theorem part1 (a : ℝ) (h : deriv (λ x, f x a) 1 = -2) : a = 1 := by 
  sorry

theorem part2 (b x1 x2 : ℝ) (hx1x2 : 0 < x1 ∧ x1 < x2) (hx1x2_eq : x1 * x2 = 1) 
    (h_gx1_gx2 : abs (g x1 b - g x2 b) ≥ 3 / 4 - real.log 2) : 
    b > 1 + 3 * real.sqrt 2 / 2 := by
  sorry

end part1_part2_l482_482090


namespace concave_function_minus_x_exp_neg_x_l482_482938

def is_concave_on (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  ∀ x ∈ D, ∃ f'' : ℝ → ℝ, (∀ x ∈ D, has_deriv_at (deriv f x) (f'' x) x) ∧ (∀ x ∈ D, f'' x > 0)

theorem concave_function_minus_x_exp_neg_x :
  is_concave_on (λ x : ℝ, -x * exp (-x)) (set.Ioo 0 (real.pi / 2)) :=
sorry

end concave_function_minus_x_exp_neg_x_l482_482938


namespace find_a2015_l482_482104

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ 
  (a 2 = 1 / 2) ∧ 
  (∀ n : ℕ, n > 0 → 2 / a (n + 1) = 1 / a n + 1 / a (n + 2))

theorem find_a2015 (a : ℕ → ℝ) (h : sequence a) : a 2015 = 1 / 2015 :=
by
  -- sequence conditions
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2
  sorry

end find_a2015_l482_482104


namespace vehicle_power_consumption_correct_l482_482229

-- Assume we introduce the variables and conditions as per the problem description
variable (xi : ℝ)
variable (n : ℕ) (σ : ℝ)
variable (h1 : probability_density_function xi (normal_distribution 13 σ) (12 <:= xi <:= 14) = 0.7)
variable (h2 : n = 1200)

-- The proof of the statement (only the theorem definition with "sorry" as the proof placeholder)
theorem vehicle_power_consumption_correct : 
  (∑ xi ∈ filter (λ x, x ≥ 14) (samples xi n), xi) = 180 :=
sorry

end vehicle_power_consumption_correct_l482_482229


namespace ratio_of_areas_l482_482121

-- Define the points
variables (A B C M : Point)

-- Define the vectors
variables (MA MB MC : Vect)

-- Assume the condition provided
axiom vector_sum_zero : MA + MB + MC = 0

-- Area function definitions
noncomputable def area (A B C : Point) : Real := sorry

-- The Lean statement for the problem
theorem ratio_of_areas (A B C M : Point) (MA MB MC : Vect)
  (h1 : MA + MB + MC = 0)
  (h2 : MA = (M - A))
  (h3 : MB = (M - B))
  (h4 : MC = (M - C))
  : (area A B M) / (area A B C) = 1 / 3 :=
sorry

end ratio_of_areas_l482_482121


namespace num_positive_divisors_of_720_multiples_of_5_l482_482556

theorem num_positive_divisors_of_720_multiples_of_5 :
  (∃ (a b c : ℕ), 0 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 2 ∧ c = 1) →
  ∃ (n : ℕ), n = 15 :=
by
  -- Proof will go here
  sorry

end num_positive_divisors_of_720_multiples_of_5_l482_482556


namespace number_of_male_alligators_l482_482196

-- Define the Conditions
def is_population_evenly_divided (total : ℕ) (males : ℕ) (females : ℕ) : Prop :=
  males = females ∧ total = males + females

def is_females_composition (total_females : ℕ) (adult_females : ℕ) (juvenile_ratio adult_ratio : ℝ) : Prop :=
  adult_ratio = 0.60 ∧ juvenile_ratio = 0.40 ∧ adult_females = (total_females * adult_ratio).to_nat

def given_adult_females (adult_females : ℕ) : Prop :=
  adult_females = 15

-- Translate to a mathematically equivalent proof problem
theorem number_of_male_alligators (total_alligators males females total_females : ℕ) 
  (juvenile_ratio adult_ratio : ℝ) :
  is_population_evenly_divided total_alligators males females →
  is_females_composition total_females 15 juvenile_ratio adult_ratio →
  given_adult_females 15 →
  males = 25 :=
by 
  intros h_population h_females h_adults
  sorry

end number_of_male_alligators_l482_482196


namespace min_value_4a_plus_b_l482_482131

theorem min_value_4a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) : 4*a + b = 9 :=
sorry

end min_value_4a_plus_b_l482_482131


namespace sum_17_20_l482_482617

-- Definitions for the conditions given in the problem
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ) 
variable (r : ℝ)

-- The sequence is geometric and sums are defined accordingly
axiom geo_seq : ∀ n, a (n + 1) = r * a n
axiom sum_def : ∀ n, S n = ∑ i in finset.range n, a (i + 1)

-- Given conditions
axiom S4 : S 4 = 1
axiom S8 : S 8 = 3

-- The value we need to prove
theorem sum_17_20 : a 17 + a 18 + a 19 + a 20 = 16 :=
sorry

end sum_17_20_l482_482617


namespace incorrect_slope_angle_relationship_l482_482380

theorem incorrect_slope_angle_relationship (α : ℝ) :
  (∀ (slope : ℝ), ∃ (theta : ℝ), theta = α) ∧
  (∀ (line : Type), ∃! (theta : ℝ), theta = α) ∧
  (α = 0 ∨ α = π / 2 → α = 0 ∨ α = 90) ∧
  (α = π / 2 → ¬ ∃ (slope : ℝ), slope = Mathlib.Real.tan α) → 
  ¬ ∀ (θ : ℝ), θ = α → ∃ (slope : ℝ), slope = Mathlib.Real.tan θ :=
by
  sorry

end incorrect_slope_angle_relationship_l482_482380


namespace prism_height_l482_482262

variables (l α β : ℝ)

-- Define conditions:
-- Base of the pyramid is an equilateral triangle
-- One lateral edge SA is perpendicular to the base and has length l
-- Other two lateral edges form angle α with the base
-- Diagonal of the lateral face of the prism forms an angle β with the base

-- Convert the given mathematical solution to a Lean statement
theorem prism_height (α β : ℝ) (l : ℝ) : 
  0 < β ∧ 0 < α → height_of_inscribed_prism = (l * cos α * sin β) / (sin (α + β)) := 
sorry

end prism_height_l482_482262


namespace last_digit_of_one_over_two_pow_twelve_l482_482765

theorem last_digit_of_one_over_two_pow_twelve : 
  let x : ℚ := 1 / 2^12 in (x * 10^12).den = 244140625 → (x.toReal - floor x.toReal) * 10 ^ 12 = 244140625 :=
by
  sorry

end last_digit_of_one_over_two_pow_twelve_l482_482765


namespace dinner_cost_l482_482178

variables 
  (ticket_cost : ℝ) (num_tickets : ℕ) (limo_cost_per_hour : ℝ) 
  (num_hours : ℕ) (total_cost : ℝ) (tip_percentage : ℝ)

-- Define the conditions from the problem
def conditions : Prop := 
  ticket_cost = 100 ∧ 
  num_tickets = 2 ∧ 
  limo_cost_per_hour = 80 ∧ 
  num_hours = 6 ∧ 
  total_cost = 836 ∧ 
  tip_percentage = 0.30

-- The theorem stating the cost of dinner before tip
theorem dinner_cost (D : ℝ) (h : conditions) : 
  let ticket_total := ticket_cost * num_tickets in
  let limo_total := limo_cost_per_hour * num_hours in
  let other_cost := ticket_total + limo_total in
  let dinner_with_tip := total_cost - other_cost in
  let dinner_cost := D in
  dinner_with_tip = 1.30 * dinner_cost → 
  D = 120 := sorry

end dinner_cost_l482_482178


namespace factor_coeff_l482_482576

theorem factor_coeff (c : ℤ) : (∃ p : polynomial ℤ, (9 * X^3 + 27 * X^2 + C * X + 48) = (X + (4/3)) * p) → c = 8 :=
by
  sorry

end factor_coeff_l482_482576


namespace most_people_can_attend_l482_482915

def attendees_on_day (day : String) : Finset String :=
  if day = "Mon" then {"Anna", "Carl"}
  else if day = "Tues" then {"Bill", "Carl", "Dave"}
  else if day = "Wed" then {"Anna", "Dave"}
  else if day = "Thurs" then {"Bill", "Carl"}
  else if day = "Fri" then {"Bill", "Carl"}
  else ∅

def max_attendees := {Mon, Wed, Thurs, Fri}

theorem most_people_can_attend :
  ∀ day, day ∈ max_attendees → set.card (attendees_on_day day) = 2 := 
by
  sorry

end most_people_can_attend_l482_482915


namespace anya_probability_l482_482399

open Finset

def possible_coins := {10, 10, 5, 5, 2}
def target := 19

noncomputable def combinations := (possible_coins.vals.ctype_power 3).val.filter (λ s, Finset.sum s >= target)

noncomputable def probability : ℝ :=
  (combinations.card : ℝ) / (possible_coins.vals.ctype_power 3).card

theorem anya_probability : probability = 0.4 := sorry

end anya_probability_l482_482399


namespace bob_hair_length_l482_482848

-- Define the current length of Bob's hair
def current_length : ℝ := 36

-- Define the growth rate in inches per month
def growth_rate : ℝ := 0.5

-- Define the duration in years
def duration_years : ℕ := 5

-- Define the total growth over the duration in years
def total_growth : ℝ := growth_rate * 12 * duration_years

-- Define the length of Bob's hair when he last cut it
def initial_length : ℝ := current_length - total_growth

-- Theorem stating that the length of Bob's hair when he last cut it was 6 inches
theorem bob_hair_length :
  initial_length = 6 :=
by
  -- Proof omitted
  sorry

end bob_hair_length_l482_482848


namespace largest_possible_s_l482_482637

theorem largest_possible_s (r s : ℕ) (h1 : r ≥ s) (h2 : s ≥ 3) 
  (h3 : ((r - 2) * 180 : ℚ) / r = (29 / 28) * ((s - 2) * 180 / s)) :
    s = 114 := by sorry

end largest_possible_s_l482_482637


namespace solve_for_y_l482_482506
-- The imports below encompass all necessary libraries in Lean 4

-- Define the given relationship between x and y
def xy_relationship (y x : ℝ) : Prop := x = (2 * y + 1) / (y - 2)

-- Define the function f(x) derived from the original problem
def f (x : ℝ) : ℝ := (2 * x + 1) / (x - 2)

-- State the theorem that carves out the solution and necessary conditions
theorem solve_for_y (x y : ℝ) (h : xy_relationship y x) : y = f x ∧ x ≠ 2 :=
by
  sorry

end solve_for_y_l482_482506


namespace compute_54_mul_46_l482_482418

theorem compute_54_mul_46 : (54 * 46 = 2484) :=
by sorry

end compute_54_mul_46_l482_482418


namespace percentage_of_percentage_l482_482334

theorem percentage_of_percentage (a b : ℝ) (h_a : a = 0.03) (h_b : b = 0.05) : (a / b) * 100 = 60 :=
by
  sorry

end percentage_of_percentage_l482_482334


namespace slope_of_line_l482_482047

theorem slope_of_line : ∀ (x y : ℝ), (x / 4 + y / 3 = 1) → (∃ m : ℝ, m = -3 / 4) :=
by
  intros x y h_eq
  exists -3 / 4
  sorry

end slope_of_line_l482_482047


namespace cistern_wet_surface_area_l482_482800

theorem cistern_wet_surface_area
  (length : ℝ) (width : ℝ) (breadth : ℝ)
  (h_length : length = 9)
  (h_width : width = 6)
  (h_breadth : breadth = 2.25) :
  (length * width + 2 * (length * breadth) + 2 * (width * breadth)) = 121.5 :=
by
  -- Proof goes here
  sorry

end cistern_wet_surface_area_l482_482800


namespace sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l482_482476

theorem sum_distinct_prime_factors_of_7_to_7_minus_7_to_4 : 
  let pfs := primeFactors (7 ^ 7 - 7 ^ 4)
  in (pfs = {2, 3, 19}) → sum pfs = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l482_482476


namespace odd_and_decreasing_l482_482836

open Real

-- Definitions used in the problem:
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

-- Statement of the proof problem:
theorem odd_and_decreasing (-sin : ℝ → ℝ) :
  is_odd (λ x, -sin x) ∧ is_decreasing_on (λ x, -sin x) 0 (π/2) :=
by
  sorry

end odd_and_decreasing_l482_482836


namespace sum_distinct_prime_factors_of_7pow7_minus_7pow4_l482_482453

noncomputable def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  let factors := (Nat.factors n).erase_dup
  factors.sum

theorem sum_distinct_prime_factors_of_7pow7_minus_7pow4 :
  sum_of_distinct_prime_factors (7 ^ 7 - 7 ^ 4) = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7pow7_minus_7pow4_l482_482453


namespace keys_in_boxes_l482_482745

theorem keys_in_boxes :
  (∑ (σ : Finset.perm (fin 8)), 1) - (∑ (σ in {s : Finset.perm (fin 8) | s.is_cycle}, 1)) = 35280 :=
by
  sorry

end keys_in_boxes_l482_482745


namespace infinite_sorted_subsequence_l482_482295

theorem infinite_sorted_subsequence : 
  ∀ (warriors : ℕ → ℕ), (∀ n, ∃ m, m > n ∧ warriors m < warriors n) 
  ∨ (∃ k, warriors k = 0) → 
  ∃ (remaining : ℕ → ℕ), (∀ i j, i < j → remaining i > remaining j) :=
by
  intros warriors h
  sorry

end infinite_sorted_subsequence_l482_482295


namespace calc_c15_l482_482220

noncomputable def seq : ℕ → ℕ
| 0     := 0   -- dummy, sequence is 1-based in the statement
| 1     := 3
| 2     := 1
| (n+1) := 2 * (seq n + seq (n-1))

theorem calc_c15 : seq 15 = 1187008 := 
by 
  sorry

end calc_c15_l482_482220


namespace minimum_colors_needed_for_tessellation_coloring_l482_482725

-- Defining a quadrilateral tessellation with conditions
def quadrilateral_tessellation (Q : Type) [fintype Q] :=
  ∀ (v : Q), (∃! q1 q2 q3 q4 : Q, meet_at_point q1 q2 ∧ meet_at_point q2 q3 ∧ meet_at_point q3 q4 ∧ meet_at_point q4 q1)

-- Noncomputable logic for quadrilateral tessellation coloring
noncomputable def quadrilateral_coloring (Q : Type) [fintype Q] :=
  ∀ (coloring : Q → ℕ), ∀ (q1 q2 : Q), meet_at_point q1 q2 → coloring q1 ≠ coloring q2

-- Main theorem statement
theorem minimum_colors_needed_for_tessellation_coloring :
  ∀ (Q : Type) [fintype Q], quadrilateral_tessellation Q → ∃ (k : ℕ), (k = 4) ∧ quadrilateral_coloring Q :=
  by
    intro Q h tess
    use 4
    sorry

end minimum_colors_needed_for_tessellation_coloring_l482_482725


namespace inradii_sum_l482_482243

theorem inradii_sum (ABCD : Type) (r_a r_b r_c r_d : ℝ) 
  (inscribed_quadrilateral : Prop) 
  (inradius_BCD : Prop) 
  (inradius_ACD : Prop) 
  (inradius_ABD : Prop) 
  (inradius_ABC : Prop) 
  (Tebo_theorem : Prop) :
  r_a + r_c = r_b + r_d := 
by
  sorry

end inradii_sum_l482_482243


namespace noah_jelly_beans_equals_50_point_4_l482_482845

noncomputable def total_jelly_beans : ℝ := 600
noncomputable def thomas_percentage : ℝ := 0.06
noncomputable def sarah_percentage : ℝ := 0.10
noncomputable def barry_ratio : ℕ := 4
noncomputable def emmanuel_ratio : ℕ := 5
noncomputable def miguel_ratio : ℕ := 6
noncomputable def chloe_percentage : ℝ := 0.40
noncomputable def noah_percentage : ℝ := 0.30

theorem noah_jelly_beans_equals_50_point_4 :
  let thomas_share := thomas_percentage * total_jelly_beans,
      sarah_share := sarah_percentage * total_jelly_beans,
      remaining_jelly_beans := total_jelly_beans - thomas_share - sarah_share,
      total_ratio := barry_ratio + emmanuel_ratio + miguel_ratio,
      jelly_beans_per_ratio := remaining_jelly_beans / total_ratio,
      barry_share := barry_ratio * jelly_beans_per_ratio,
      emmanuel_share := emmanuel_ratio * jelly_beans_per_ratio,
      noah_share := noah_percentage * emmanuel_share
  in noah_share = 50.4 := 
by {
  sorry
}

end noah_jelly_beans_equals_50_point_4_l482_482845


namespace simplify_sum_of_square_roots_l482_482683

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l482_482683


namespace floor_lambda_is_square_l482_482786

theorem floor_lambda_is_square (λ : ℝ) (n : ℕ)
  (hλ : λ ≥ 1)
  (h : ∀ k, n + 1 ≤ k → k ≤ 4 * n → ∃ m : ℕ, m^2 = ⌊λ^k⌋) : ∃ a : ℕ, a^2 = ⌊λ⌋ :=
by
sory

end floor_lambda_is_square_l482_482786


namespace mira_result_l482_482647

def round_to_nearest_hundred (n : ℤ) : ℤ :=
  if n % 100 >= 50 then n / 100 * 100 + 100 else n / 100 * 100

theorem mira_result :
  round_to_nearest_hundred ((63 + 48) - 21) = 100 :=
by
  sorry

end mira_result_l482_482647


namespace clock_angle_at_9_oclock_l482_482113

theorem clock_angle_at_9_oclock :
  (∀ h m : ℕ, 
     h = 9 ∧ m = 0 →
     let minute_hand_angle := m * 6 in
     let hour_hand_angle := h * 30 in
     let angle := (hour_hand_angle - minute_hand_angle) % 360 in
     let smaller_angle := if angle > 180 then 360 - angle else angle in
     smaller_angle = 90) :=
by
  intros h m h_eq.
  cases h_eq with h9 m0.
  rw [h9, m0].
  let minute_hand_angle := 0 * 6,
  let hour_hand_angle := 9 * 30,
  let angle := (hour_hand_angle - minute_hand_angle) % 360,
  let smaller_angle := if angle > 180 then 360 - angle else angle,
  have : minute_hand_angle = 0 := rfl,
  have : hour_hand_angle = 270 := rfl,
  have : angle = (270 - 0) % 360 := rfl,
  have : angle = 270 := rfl,
  have : smaller_angle = 360 - 270 := rfl,
  have : smaller_angle = 90 := rfl,
  sorry

end clock_angle_at_9_oclock_l482_482113


namespace infinite_lines_in_plane_perpendicular_to_l_l482_482579

variable (l : Type) (α : Type)

-- assuming the perpendicular relation between a line and a plane or a line exists
variable [perpendicular : HasPerp l α] [perpendicular : HasPerp l l]

-- stating the essential condition
axiom l_not_perpendicular_to_alpha : ¬(perpendicular.perp l α)

-- essential property needed for proof
axiom infinite_lines_perpendicular_in_plane : 
  ∀ (l : Type) (α : Type) [perpendicular : HasPerp l α] [perpendicular : HasPerp l l], 
  ¬(perpendicular.perp l α) → ∃ (lines : Set l), ∞ ∈ lines ∧ ∀ l' ∈ lines, perpendicular.perp l' l

theorem infinite_lines_in_plane_perpendicular_to_l : 
  ∃ (lines : Set l), ∞ ∈ lines ∧ ∀ l' ∈ lines, perpendicular.perp l' l :=
  sorry

end infinite_lines_in_plane_perpendicular_to_l_l482_482579


namespace probability_of_picking_combination_is_0_4_l482_482385

noncomputable def probability_at_least_19_rubles (total_coins total_value: ℕ) :=
  let coins := [10, 10, 5, 5, 2] in
  let all_combinations := (Finset.powersetLen 3 (coins.to_finset)).to_list in
  let favorable_combinations := all_combinations.filter (fun c => c.sum ≥ total_value) in
  (favorable_combinations.length : ℚ) / (all_combinations.length : ℚ)

theorem probability_of_picking_combination_is_0_4 :
  probability_at_least_19_rubles 5 19 = 0.4 :=
by
  sorry

end probability_of_picking_combination_is_0_4_l482_482385


namespace simplify_sum_of_square_roots_l482_482681

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l482_482681


namespace part1_part2_l482_482963

variables (a b : ℝ → E)
variables (k : ℝ)
variables [inner_product_space ℝ E]

-- Given conditions
def norm_a : real := 2
def norm_b : real := 1
def angle_ab : real := 60
def dot_ab : real := norm_a * norm_b * real.cos(angle_ab * real.pi / 180)

def c : E := 2 • a + 3 • b
def d : E := 3 • a + k • b

-- Part 1: Proving k for orthogonality
theorem part1 (h1 : inner c d = 0) : k = -33/5 :=
by sorry

-- Part 2: Proving k for magnitude of d
theorem part2 (h2 : ∥d∥ = 2 * sqrt 13) : k = 2 ∨ k = -8 :=
by sorry

end part1_part2_l482_482963


namespace translate_sin_function_l482_482755

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)
def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - 5 * Real.pi / 6)

theorem translate_sin_function :
    (∀ x : ℝ, f (x - Real.pi / 3) = g x) :=
by 
  intros x
  dsimp [f, g]
  rw [←Real.sin_sub, ←sub_sub, Real.sub_left_inj]
  sorry

end translate_sin_function_l482_482755


namespace autumn_sales_l482_482600

theorem autumn_sales (T : ℝ) (spring summer winter autumn : ℝ) 
    (h1 : spring = 3)
    (h2 : summer = 6)
    (h3 : winter = 5)
    (h4 : T = (3 / 0.2)) :
    autumn = 1 :=
by 
  -- Proof goes here
  sorry

end autumn_sales_l482_482600


namespace smaug_gold_coins_l482_482251

theorem smaug_gold_coins :
  ∃ G : ℕ, let silver_value := 60 * 8,
             gold_value := G * 3 * 8,
             total_value := gold_value + silver_value + 33 in
           total_value = 2913 ∧ G = 100 :=
by
  sorry

end smaug_gold_coins_l482_482251


namespace bushes_needed_l482_482580

-- Definitions for the problem conditions
def side_length : ℕ := 20
def spacing : ℕ := 2

-- Statement of the problem
theorem bushes_needed (s : ℕ) (d : ℕ) (hs : s = 20) (hd : d = 2) : (4 * s) / d = 40 :=
by {
  rw [hs, hd],
  norm_num,
  sorry
}

end bushes_needed_l482_482580


namespace sqrt_sum_simplify_l482_482675

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l482_482675


namespace total_onions_grown_l482_482228

-- Given conditions
def onions_grown_by_Nancy : ℕ := 2
def onions_grown_by_Dan : ℕ := 9
def onions_grown_by_Mike : ℕ := 4
def days_worked : ℕ := 6

-- Statement we need to prove
theorem total_onions_grown : onions_grown_by_Nancy + onions_grown_by_Dan + onions_grown_by_Mike = 15 :=
by sorry

end total_onions_grown_l482_482228


namespace constant_term_expansion_l482_482615

theorem constant_term_expansion :
  let expr := (Real.sqrt x - 2 / x) ^ 3 in
  ∃ c : ℝ, c = -6 ∧ is_constant_term expr c :=
sorry

end constant_term_expansion_l482_482615


namespace increasing_function_range_l482_482942

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 1) * x - 1 else x + 1

theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (1 / 2 < a ∧ a ≤ 2) :=
sorry

end increasing_function_range_l482_482942


namespace factor_roots_l482_482036

noncomputable def checkRoots (a b c t : ℚ) : Prop :=
  a * t^2 + b * t + c = 0

theorem factor_roots (t : ℚ) :
  checkRoots 8 17 (-10) t ↔ t = 5/8 ∨ t = -2 := by
sorry

end factor_roots_l482_482036


namespace distance_skew_lines_correct_l482_482203

open Real EuclideanGeometry

noncomputable def distance_skew_lines (A B : Point) (a b : Line) (n : Vector) (hA : A ∈ a) (hB : B ∈ b)
(hna : n ⊥ a) (hnb : n ⊥ b) : Real :=
|over A B ⬝ n| / |n|

theorem distance_skew_lines_correct : 
  ∀ (A B : Point) (a b : Line) (n : Vector), 
  (A ∈ a) → 
  (B ∈ b) → 
  (n ⊥ a) → 
  (n ⊥ b) → 
  distance_skew_lines A B a b n _ _ _ _ = |over(A, B) ⬝ n| / |n| :=
by
  intros A B a b n hA hB hna hnb
  sorry

end distance_skew_lines_correct_l482_482203


namespace part1_part2_l482_482087

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - a * x^2 + x

-- Part 1
theorem part1 (a : ℝ) : (∀ x : ℝ, x > 0 → f' x a ≤ 0) → a ≥ Real.exp 1 / 2 := 
sorry

-- Part 2
theorem part2 (a x1 x2 : ℝ) (h_zero : f x1 a = 0 ∧ f x2 a = 0) (h_cond : x2 > 2 * x1) : 
  x1 * x2 > 8 / Real.exp 2 :=
sorry

end part1_part2_l482_482087


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l482_482471

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4 :
  let n := 7^7 - 7^4 in 
  (∑ p in (nat.factors n).to_finset, p) = 31 :=
by sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l482_482471


namespace dominoes_games_unique_l482_482899

theorem dominoes_games_unique (players : Fin 5 → Fin 5 → Fin 5 → Prop)
  (h_unique : ∀ (i j : Fin 5), i ≠ j → ∃! k, players i j k)
  (h_partners : ∀ (i : Fin 5), ∃! j, ∃! k, j ≠ k ∧ players i j k ∧ players i j k)
  (h_opponents : ∀ (i j : Fin 5), i ≠ j → ∃! k, players i j k ∧ players j i k) :
  ∃ (games : Fin 5), ∀ (arrangements : Fin 5 → Fin 5 → Prop), arrangements = players :=
sorry

end dominoes_games_unique_l482_482899


namespace children_total_savings_l482_482193

theorem children_total_savings :
  let josiah_savings := 0.25 * 24
  let leah_savings := 0.50 * 20
  let megan_savings := (2 * 0.50) * 12
  josiah_savings + leah_savings + megan_savings = 28 := by
{
  -- lean proof goes here
  sorry
}

end children_total_savings_l482_482193


namespace find_polynomial_parameters_and_minimum_value_l482_482093

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_polynomial_parameters_and_minimum_value 
  (a b c : ℝ)
  (h1 : f (-1) a b c = 7)
  (h2 : 3 * (-1)^2 + 2 * a * (-1) + b = 0)
  (h3 : 3 * 3^2 + 2 * a * 3 + b = 0)
  (h4 : a = -3)
  (h5 : b = -9)
  (h6 : c = 2) :
  f 3 (-3) (-9) 2 = -25 :=
by
  sorry

end find_polynomial_parameters_and_minimum_value_l482_482093


namespace meet_at_correct_time_l482_482415

-- Define the main problem constants and variables
def meeting_time := 11 + (15 / 60)

-- Define conditions
def route_distance := 78
def cassie_start_time := 8.5 -- 8:30 AM
def cassie_speed := 14 -- miles per hour
def brian_start_time := 9 -- 9:00 AM
def brian_speed := 18 -- miles per hour

-- Definition of the time they'll meet
def time_to_meet (x: ℝ) : ℝ := 
  cassie_speed * x + brian_speed * (x - 0.5) = route_distance

-- Prove that the meeting time is 11:15 AM
theorem meet_at_correct_time (x: ℝ) (h: time_to_meet x) : 
  cassie_start_time + x = meeting_time :=
  by sorry

end meet_at_correct_time_l482_482415


namespace tom_paid_total_amount_l482_482291

def cost_apples := 8 * 70
def cost_mangoes := 9 * 75
def cost_bananas := 6 * 40
def cost_grapes := 4 * 120
def cost_cherries := 3 * 180
def total_cost := cost_apples + cost_mangoes + cost_bananas + cost_grapes + cost_cherries

theorem tom_paid_total_amount :
  total_cost = 2495 :=
by
  unfold cost_apples cost_mangoes cost_bananas cost_grapes cost_cherries total_cost
  rfl

end tom_paid_total_amount_l482_482291


namespace problem_x_y_z_l482_482118

theorem problem_x_y_z (x y z : ℕ) (h1 : xy + z = 47) (h2 : yz + x = 47) (h3 : xz + y = 47) : x + y + z = 48 :=
sorry

end problem_x_y_z_l482_482118


namespace sqrt_72_plus_sqrt_32_l482_482713

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l482_482713


namespace num_ordered_pairs_eq_1728_l482_482272

theorem num_ordered_pairs_eq_1728 (x y : ℕ) (h1 : 1728 = 2^6 * 3^3) (h2 : x * y = 1728) : 
  ∃ (n : ℕ), n = 28 := 
sorry

end num_ordered_pairs_eq_1728_l482_482272


namespace math_problem_l482_482962

noncomputable def alpha_condition (α : ℝ) : Prop :=
  4 * Real.cos α - 2 * Real.sin α = 0

theorem math_problem (α : ℝ) (h : alpha_condition α) :
  (Real.sin α)^3 + (Real.cos α)^3 / (Real.sin α - Real.cos α) = 9 / 5 :=
  sorry

end math_problem_l482_482962


namespace tower_mod_500_is_152_l482_482348

-- Definitions of the conditions
def isValidTower (tower : List ℕ) : Prop :=
  ∀ i < tower.length - 1, tower[i+1] <= tower[i] + 2

def allTowers : List (List ℕ) :=
  List.permutations [2, 3, 4, 5, 6, 7, 8, 9]

noncomputable def totalValidTowers : ℕ :=
  (allTowers.filter isValidTower).length

def remainderWhenDividedBy500 (n : ℕ) : ℕ := n % 500

-- The statement we need to prove
theorem tower_mod_500_is_152 : remainderWhenDividedBy500 totalValidTowers = 152 :=
by
  sorry

end tower_mod_500_is_152_l482_482348


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482464

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let expr := 7 ^ 7 - 7 ^ 4 in
  (7^4 * (7^3 - 1) = expr) ∧ (7^3 - 1 = 342) ∧ (Prime 2) ∧ (Prime 3) ∧ (Prime 7) ∧ (Prime 19) ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ 342 → p = 2 ∨ p = 3 ∨ p = 19) → 
  (∀ p : ℕ, Nat.Prime p → p ∣ expr → p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19) → 
  (2 + 3 + 7 + 19 = 31) := 
by
  intro expr fact1 fact2 prime2 prime3 prime7 prime19 factors342 factorsExpr
  sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482464


namespace inscribed_circle_radius_l482_482304

theorem inscribed_circle_radius (A B C : ℝ) (hAB : AB = 6) (hAC : AC = 8) (hBC : BC = 10) : r = 2 :=
by
  -- definitions and conditions 
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt(s * (s - AB) * (s - AC) * (s - BC))
  have h_s : s = 12 
    by sorry -- calculated the semiperimeter
  have h_K : K = 24 
    by sorry -- calculated the area using Heron's formula
  have relation_area_radius : K = r * s 
    by sorry -- relation between area and radius of inscribed circle
  show r = 2
    from sorry -- final step to solve for r

end inscribed_circle_radius_l482_482304


namespace triangle_dot_product_range_l482_482603

theorem triangle_dot_product_range :
  ∀ (A B C M N : ℝ × ℝ),
    ∃ CA CB : ℝ,
    (triangle_right A B C) →
    (CA = 2) →
    (CB = 2) →
    (point_on_line M A B) →
    (point_on_line N A B) →
    (dist M N = √2) →
    (range (λb : ℝ, 2 * (b^2 - b + 1)) = set.interval (3/2) 2) :=
begin
  sorry
end

end triangle_dot_product_range_l482_482603


namespace train_length_correct_l482_482327

-- Definitions of conditions
def speed_kmh : ℝ := 90
def time_sec : ℝ := 9

-- Define conversion factors
def km_to_m : ℝ := 1000
def hr_to_sec : ℝ := 3600

-- Convert speed from km/hr to m/s
def speed_mps : ℝ := speed_kmh * (km_to_m / hr_to_sec)

-- Define the expected length of the train
def expected_length : ℝ := 225

-- Statement to prove that the length of the train is 225 meters given the speed and time
theorem train_length_correct :
  (speed_mps * time_sec) = expected_length :=
by
  unfold speed_mps
  unfold expected_length
  simp
  sorry

end train_length_correct_l482_482327


namespace number_of_dots_in_120_circles_l482_482823

theorem number_of_dots_in_120_circles :
  ∃ n : ℕ, (n = 14) ∧ (∀ m : ℕ, m * (m + 1) / 2 + m ≤ 120 → m ≤ n) :=
by
  sorry

end number_of_dots_in_120_circles_l482_482823


namespace M_is_positive_rationals_l482_482493

axiom M : set ℚ

-- Condition 1: if a, b ∈ M, then a+b ∈ M and a*b ∈ M
axiom cond1 (a b : ℚ) : a ∈ M → b ∈ M → a + b ∈ M ∧ a * b ∈ M

-- Condition 2: For any r in ℚ, exactly one of r ∈ M, -r ∈ M, or r = 0
axiom cond2 (r : ℚ) : (r ∈ M ∨ -r ∈ M ∨ r = 0) ∧ ¬(r ∈ M ∧ -r ∈ M) ∧ ¬(r = 0 ∧ (r ∈ M ∨ -r ∈ M))

theorem M_is_positive_rationals : M = { x : ℚ | 0 < x } :=
by sorry

end M_is_positive_rationals_l482_482493


namespace correct_A_correct_B_intersection_A_B_complement_B_l482_482089

noncomputable def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

theorem correct_A : A = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
by
  sorry

theorem correct_B : B = {x : ℝ | 1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem intersection_A_B : (A ∩ B) = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
by
  sorry

theorem complement_B : (Bᶜ) = {x : ℝ | x < 1 ∨ x > 4} :=
by
  sorry

end correct_A_correct_B_intersection_A_B_complement_B_l482_482089


namespace greatest_area_triangle_ABD_l482_482323

theorem greatest_area_triangle_ABD (A B C D E F : Point) 
  (Hconv : convex_quadrilateral A B C D) 
  (HmidE : midpoint E B C) 
  (HmidF : midpoint F C D)
  (A_triangle_areas : ∃ (n : ℕ), triangle_area A E B = n ∧ 
                                  triangle_area A F B = n+1 ∧ 
                                  triangle_area A E F = n+2 ∧ 
                                  triangle_area A F C = n+3) :
  ∃ (area : ℕ), is_greatest_area_of_triangle_ABD area 6 :=
  sorry

end greatest_area_triangle_ABD_l482_482323


namespace total_weight_moved_l482_482751

theorem total_weight_moved (tom_weight : ℝ) (vest_fraction : ℝ) (hold_fraction : ℝ) :
  tom_weight = 150 → vest_fraction = 0.5 → hold_fraction = 1.5 →
  let vest_weight := vest_fraction * tom_weight,
      hand_weight := hold_fraction * tom_weight,
      total_hand_weight := 2 * hand_weight,
      total_weight := tom_weight + vest_weight + total_hand_weight in
  total_weight = 675 :=
by
  sorry

end total_weight_moved_l482_482751


namespace tim_gave_away_l482_482749

def winnings_total : ℕ := 100
def amount_kept : ℕ := 80
def amount_given : ℕ := winnings_total - amount_kept
def percentage_given : ℚ := (amount_given / winnings_total) * 100

theorem tim_gave_away (h_winnings : winnings_total = 100) (h_kept : amount_kept = 80) : percentage_given = 20 := 
by
  unfold winnings_total amount_kept amount_given percentage_given
  rw [h_winnings, h_kept]
  norm_num
  sorry

end tim_gave_away_l482_482749


namespace value_of_x_when_y_is_20_l482_482841

-- Definitions from conditions
variables (x y k : ℕ)
hypothesis h1 : x * y = k
hypothesis h2 : 40 * 8 = k
hypothesis target_y : y = 20

-- Statement of the proof problem
theorem value_of_x_when_y_is_20 : x = 16 :=
by 
  sorry

end value_of_x_when_y_is_20_l482_482841


namespace sandwich_meat_cost_l482_482721

theorem sandwich_meat_cost (bread_cost cheese_cost meat_cost_coupon cheese_coupon meat_coupon per_sandwich_cost total_sandwiches : ℝ) 
  (h_bread_cost : bread_cost = 4) 
  (h_cheese_cost : cheese_cost = 4)
  (h_cheese_coupon : cheese_coupon = 1)
  (h_meat_coupon : meat_coupon = 1)
  (h_per_sandwich_cost : per_sandwich_cost = 2) 
  (h_total_sandwiches : total_sandwiches = 10) :
  let total_cost := total_sandwiches * per_sandwich_cost in
  let total_cheese_cost := cheese_cost + (cheese_cost - cheese_coupon) in
  let eqn := bread_cost + total_cheese_cost + (meat_cost_coupon + (meat_cost_coupon - meat_coupon)) in
  total_cost = 20 → meat_cost_coupon = 5 := 
by 
  intros total_cost total_cheese_cost eqn h_total_cost;
  sorry

end sandwich_meat_cost_l482_482721


namespace indistinguishable_balls_boxes_l482_482973

theorem indistinguishable_balls_boxes (n m : ℕ) (h : n = 6) (k : m = 2) : 
  (finset.card (finset.filter (λ x : finset ℕ, x.card ≤ n / 2) 
    (finset.powerset (finset.range (n + 1)))) = 4) :=
by
  sorry

end indistinguishable_balls_boxes_l482_482973


namespace tetrahedron_volume_l482_482582

noncomputable def volume_of_tetrahedron (SC A B C : ℝ) (AB : ℝ) (angle_SCA angle_SCB : ℝ) :=
  -- Conditions for the problem
  SC = 2 ∧
  AB = (Real.sqrt 3) / 2 ∧
  angle_SCA = 60 ∧
  angle_SCB = 60 →
  -- The volume of the tetrahedron S-ABC.
  (volume_of_tetrahedron S A B C = (Real.sqrt 3) / 8)

theorem tetrahedron_volume (S A B C : ℝ) (AB : ℝ) (angle_SCA angle_SCB : ℝ) :
  volume_of_tetrahedron SC A B C AB angle_SCA angle_SCB :=
begin
  sorry
end

end tetrahedron_volume_l482_482582


namespace Anita_should_buy_more_cartons_l482_482839

def Anita_needs (total_needed : ℕ) : Prop :=
total_needed = 26

def Anita_has (strawberries blueberries : ℕ) : Prop :=
strawberries = 10 ∧ blueberries = 9

def additional_cartons (total_needed strawberries blueberries : ℕ) : ℕ :=
total_needed - (strawberries + blueberries)

theorem Anita_should_buy_more_cartons :
  ∀ (total_needed strawberries blueberries : ℕ),
    Anita_needs total_needed →
    Anita_has strawberries blueberries →
    additional_cartons total_needed strawberries blueberries = 7 :=
by
  intros total_needed strawberries blueberries Hneeds Hhas
  sorry

end Anita_should_buy_more_cartons_l482_482839


namespace finite_operations_for_any_configuration_l482_482257

noncomputable def stopsAfterFiniteOperations (n : ℕ) : Prop :=
  ∀ (coins : List Bool), List.length coins = n → ∃ m : ℕ, (∀ i < m, (∃ k, k ≥ 1 ∧ (List.filter id coins).length = k ∧ coins.get? (k - 1) = some true ∧ (coins := coins.set (k - 1) !coins.get (k - 1).get)) ∧ (List.filter id coins).length = 0

theorem finite_operations_for_any_configuration (n : ℕ) : stopsAfterFiniteOperations n :=
by
  sorry

end finite_operations_for_any_configuration_l482_482257


namespace expression_approx_five_l482_482854

def expression := (10^2005 + 10^2007) / (10^2006 + 10^2006)

theorem expression_approx_five : (10^2005 + 10^2007) / (10^2006 + 10^2006) ≈ 5 := 
by sorry

end expression_approx_five_l482_482854


namespace angle_BAC_eq_30_degrees_l482_482758

/--
Given a triangle ABC inscribed in a circle O with AB > AC > BC.
Let D be a point on the arc BC. Perpendiculars from O to AB and AC
intersect AD at points E and F, respectively. Let rays BE and CF
intersect at point P. Given BP = PC + PO, then the angle BAC is 30 degrees.
-/

theorem angle_BAC_eq_30_degrees
  (O A B C D E F P : Point)
  (h1 : triangle_in_circle A B C O)
  (h2 : on_arc D B C O)
  (h3 : perp_from_center O A B E)
  (h4 : perp_from_center O A C F)
  (h5 : AD_intersects E F)
  (h6 : rays_intersect P B E)
  (h7 : rays_intersect P C F)
  (h8 : BP = PC + PO) :
  ∠BAC = 30 :=
sorry

end angle_BAC_eq_30_degrees_l482_482758


namespace smallest_integer_congruent_l482_482310

noncomputable def smallest_four_digit_negative_integer_congr_1_mod_37 : ℤ :=
-1034

theorem smallest_integer_congruent (n : ℤ) (h1 : 37 * n + 1 < -999) : 
  ∃ (k : ℤ), (37 * k + 1) = -1034 ∧ 
  -10000 < (37 * k + 1) ∧ (37 * k + 1 = 1 % 37) := 
by {
  use -28,
  split,
  { refl },
  split,
  { sorry },
  { sorry }
}

end smallest_integer_congruent_l482_482310


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482457

def seven_pow_seven_minus_seven_pow_four : ℤ := 7^7 - 7^4
def prime_factors_of_three_hundred_forty_two : List ℤ := [2, 3, 19]

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let distinct_prime_factors := prime_factors_of_three_hundred_forty_two.head!
  + prime_factors_of_three_hundred_forty_two.tail!.head!
  + prime_factors_of_three_hundred_forty_two.tail!.tail!.head!
  seven_pow_seven_minus_seven_pow_four = 7^4 * (7^3 - 1) ∧
  7^3 - 1 = 342 ∧
  prime_factors_of_three_hundred_forty_two = [2, 3, 19] ∧
  distinct_prime_factors = 24 := 
sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482457


namespace solution_l482_482884

noncomputable theory

def f (x : ℚ) : ℚ := sorry  -- To be defined

lemma functional_equation1 (x y : ℚ) :
  f(x + y) - y * f(x) - x * f(y) = f(x) * f(y) - x - y + x * y := sorry

lemma functional_equation2 (x : ℚ) :
  f(x) = 2 * f(x + 1) + 2 + x := sorry

lemma positivity_condition :
  f(1) + 1 > 0 := sorry

theorem solution (x : ℚ) :
  f(x) = - x / 2 :=
begin
  -- The proof goes here
  sorry
end

end solution_l482_482884


namespace required_tents_l482_482645

def numberOfPeopleInMattFamily : ℕ := 1 + 2
def numberOfPeopleInBrotherFamily : ℕ := 1 + 1 + 4
def numberOfPeopleInUncleJoeFamily : ℕ := 1 + 1 + 3
def totalNumberOfPeople : ℕ := numberOfPeopleInMattFamily + numberOfPeopleInBrotherFamily + numberOfPeopleInUncleJoeFamily
def numberOfPeopleSleepingInHouse : ℕ := 4
def numberOfPeopleSleepingInTents : ℕ := totalNumberOfPeople - numberOfPeopleSleepingInHouse
def peoplePerTent : ℕ := 2

def numberOfTentsNeeded : ℕ :=
  numberOfPeopleSleepingInTents / peoplePerTent

theorem required_tents : numberOfTentsNeeded = 5 := by
  sorry

end required_tents_l482_482645


namespace uncle_jerry_tomatoes_l482_482296

variable (yesterdayReap todayReap totalReap : ℕ)

-- Conditions
def condition1 : yesterdayReap = 120 := by sorry
def condition2 : todayReap = yesterdayReap + 50 := by sorry

-- Problem statement
theorem uncle_jerry_tomatoes (yesterdayReap todayReap totalReap : ℕ) 
    (h1 : condition1) 
    (h2 : condition2) : 
    totalReap = yesterdayReap + todayReap := by
  rw [condition1, condition2]
  unfold totalReap
  sorry

end uncle_jerry_tomatoes_l482_482296


namespace intersection_points_count_l482_482953

def f (x : ℝ) : ℝ := if x ∈ set.Icc (-1 : ℝ) (1 : ℝ) then x^2 else sorry

theorem intersection_points_count (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x + 1) = f (x - 1))
  (h2 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = x^2) :
  ∃ n : ℕ, n = 4 ∧ ∀ x : ℝ, (f x = real.log x / real.log 3 → sorry) ∧ (f x = real.log (-x) / real.log 3 → sorry) :=
begin
  sorry
end

end intersection_points_count_l482_482953


namespace teena_distance_behind_poe_l482_482722

theorem teena_distance_behind_poe (D : ℝ)
    (teena_speed : ℝ) (poe_speed : ℝ)
    (time_hours : ℝ) (teena_ahead : ℝ) :
    teena_speed = 55 
    → poe_speed = 40 
    → time_hours = 1.5 
    → teena_ahead = 15 
    → D + teena_ahead = (teena_speed - poe_speed) * time_hours 
    → D = 7.5 := 
by 
    intros 
    sorry

end teena_distance_behind_poe_l482_482722


namespace triangle_CEF_perimeter_l482_482657

noncomputable def point := (ℝ, ℝ)
structure square (A B C D E F: point) :=
  (side_length : ℝ)
  (E_on_BC : ∃ x, E = (B.1 + x, C.2))
  (F_on_CD : ∃ y, F = (C.1, D.2 - y))
  (angle_EAF : ∃ θ, θ = 45)

def perimeter (C E F : point) : ℝ := 
  (Real.dist C E) + (Real.dist E F) + (Real.dist F C)

theorem triangle_CEF_perimeter (A B C D E F : point) : 
  square A B C D E F ∧ A.1 = 0 ∧ A.2 = 1 ∧ B.1 = 1 ∧ B.2 = 1 ∧ C.1 = 1 ∧ C.2 = 0 ∧ D.1 = 0 ∧ D.2 = 0 → 
  perimeter C E F = 2 :=
by
  sorry

end triangle_CEF_perimeter_l482_482657


namespace exists_adjacent_pair_with_large_diff_l482_482438

-- Define the graph structure
structure Graph :=
  (vertices : Finset ℕ)
  (edges : Finset (ℕ × ℕ))
  (adj : (ℕ × ℕ) → Prop)
  
-- Define the specific graph as given in the problem
noncomputable def specific_graph : Graph :=
{ vertices := Finset.range 20,
  edges := sorry, -- Representing the connections as given in the problem
  adj := sorry -- Define adjacency based on the graph structure
}

-- Define the numbers placed on vertices ranging from 1 to 20
noncomputable def numbers_on_vertices : Fin 20 → ℕ := sorry

-- Define the property that checks if the difference is greater than 3
def diff_greater_than_three (n1 n2 : ℕ) : Prop := |n1 - n2| > 3

-- The main theorem to be proved
theorem exists_adjacent_pair_with_large_diff :
  ∃ (v1 v2 : Fin 20), specific_graph.adj (v1, v2) ∧ diff_greater_than_three (numbers_on_vertices v1) (numbers_on_vertices v2) :=
sorry

end exists_adjacent_pair_with_large_diff_l482_482438


namespace find_a_is_8_l482_482265

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 1

def is_even_function_on_interval (a : ℝ) (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x ∈ I, f (-x) = f x

def interval (a : ℝ) : set ℝ := set.Icc (3 - a) 5

theorem find_a_is_8 :
  ∃ a : ℝ, is_even_function_on_interval a (f a) (interval a) ∧ a = 8 :=
sorry

end find_a_is_8_l482_482265


namespace number_of_logs_in_stack_l482_482368

theorem number_of_logs_in_stack :
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  total_logs = 114 := by
{
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  sorry
}

end number_of_logs_in_stack_l482_482368


namespace no_fractions_meet_condition_l482_482420

def relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem no_fractions_meet_condition :
  ∀ x y : ℕ,
    0 < x → 0 < y →
    relatively_prime x y →
    (↑(x + 1) / ↑(y + 1) = 1.2 * (↑x / ↑y)) →
    false :=
by
  intros x y hx hy hrel hcond
  sorry

end no_fractions_meet_condition_l482_482420


namespace problem_201_l482_482105

theorem problem_201 (a b c : ℕ) 
  (h₀ : {a, b, c} = {0, 1, 2}) 
  (h₁ : (a ≠ 2 ∧ b ≠ 2 ∧ c = 0) 
        ∨ (a = 0 ∧ b = 2 ∧ c ≠ 0) 
        ∨ (a = 2 ∧ b = 0 ∧ c = 1)) :
  100 * a + 10 * b + c = 201 := 
by sorry

end problem_201_l482_482105


namespace perp_dot_product_square_l482_482772

variable {V : Type*} [InnerProductSpace ℝ V]

theorem perp_dot_product_square (a b : V) (h : ⟪a, b⟫ = 0) : ⟪a, b⟫ = ⟨⟪a, b⟫, λ⟫ :=
by
  rw [h, mul_zero]
  -- here \(\⟪a, b⟫\) represents the inner product, alternatively \(\dot\cdot\) can be used
  exact h

end perp_dot_product_square_l482_482772


namespace value_of_f_minus_a_l482_482541

noncomputable def f (x : ℝ) : ℝ := x^3 + x + 1

theorem value_of_f_minus_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by sorry

end value_of_f_minus_a_l482_482541


namespace range_of_k_for_inequality_l482_482896

theorem range_of_k_for_inequality :
  { k : ℝ // ∃ x : ℝ, |x + 1| + k < x } = { k : ℝ // k < -1 } :=
by
  ext k
  simp only [set_of_eq, set.mem_set_of_eq]
  split
  · intro h
    obtain ⟨x, hx⟩ := h
    by_cases h1 : x + 1 > 0
    · have : k < -1 := by linarith [hx, h1]
      exact this
    by_cases h2 : x + 1 < 0
    · have : k < -1 := by linarith [hx, h2]
      exact this
    · have x_eq : x + 1 = 0 := by linarith [not_lt.mp (not_or.mp h)]
      have : k < -1 := by linarith [hx, x_eq]
      exact this
  · intro h
    use -2 -- any x < -1 would work
    linarith

end range_of_k_for_inequality_l482_482896


namespace solution_exists_l482_482436

def divide_sum_of_squares_and_quotient_eq_seventy_two (x : ℝ) : Prop :=
  (10 - x)^2 + x^2 + (10 - x) / x = 72

theorem solution_exists (x : ℝ) : divide_sum_of_squares_and_quotient_eq_seventy_two x → x = 2 := sorry

end solution_exists_l482_482436


namespace maximum_possible_value_l482_482940

def max_expr_value : ℕ :=
  let S := {0, 1, 2, 3, 4}
  Finset.sup (Set.to_finset
    {c * (a + b) - d | a b c d ∈ S ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c}) id

theorem maximum_possible_value :
  max_expr_value = 14 :=
by
  sorry

end maximum_possible_value_l482_482940


namespace exists_block_with_five_primes_l482_482287

-- Define the function f(n) which counts the number of primes in {n, n+1, ..., n+999}
def f (n : ℕ) : ℕ :=
  (Finset.filter (λ x, Nat.Prime x) (Finset.range' n 1000)).card

-- State the theorem that there exists a block of 1000 consecutive integers with exactly 5 primes
theorem exists_block_with_five_primes :
  ∃ m : ℕ, f m = 5 :=
begin
  -- Conditions given in the problem
  have block_with_no_primes : ∀ k, 2 ≤ k ∧ k ≤ 1001 → ¬Nat.Prime (1001! + k),
  { sorry },

  have f_initial : f 1 > 5,
  { sorry },

  have f_final : f (1001! + 2) = 0,
  { sorry },

  -- Use the intermediate value theorem to find the block with exactly 5 primes
  obtain ⟨m, h_m⟩ := Intermediate_value_theorem f_initial f_final,
  use m,
  exact h_m,
end

end exists_block_with_five_primes_l482_482287


namespace union_of_sets_l482_482927

def A : Set ℤ := {0, 1}
def B : Set ℤ := {1, 2}

theorem union_of_sets :
  A ∪ B = {0, 1, 2} :=
by
  sorry

end union_of_sets_l482_482927


namespace sqrt_sum_simplify_l482_482690

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l482_482690


namespace can_form_square_by_cutting_shape_l482_482412

theorem can_form_square_by_cutting_shape :
  ∃ parts : list (set (ℤ × ℤ)), 
  (∀ part ∈ parts, part.card = 4) ∧ 
  (∀ part1 part2 ∈ parts, part1 ≠ part2 → part1 ∩ part2 = ∅) ∧ 
  (∀ part ∈ parts, ∃ (x y : ℤ), 
    part = {(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)}) :=
sorry

end can_form_square_by_cutting_shape_l482_482412


namespace complement_degree_correct_l482_482381

-- Define the measure of the given angle
def given_angle : ℝ := 28 + 39 / 60

-- Define the measure of the complement angle
def complement_angle : ℝ := 90 - given_angle

-- Define the expected complement measure based on the solution provided
def expected_complement : ℝ := 61 + 21 / 60

-- The theorem to prove
theorem complement_degree_correct :
  complement_angle = expected_complement :=
sorry

end complement_degree_correct_l482_482381


namespace construct_triangle_l482_482423

-- Definitions for the conditions
def angle_bisector (A B C D : Point) (alpha : ℝ) : Prop := 
  ∠BAC = 2 * alpha ∧ LineSeg A D ∧ ∠BAD = alpha ∧ ∠CAD = alpha

def bisect_segment (B C D : Point) (BD DC : ℝ) : Prop :=
  dist B D = BD ∧ dist C D = DC

def feasibility_condition (alpha : ℝ) : Prop :=
  2 * alpha < 180

-- Main statement
theorem construct_triangle (A B C D : Point) (alpha BD DC : ℝ)
  (h_angle : angle_bisector A B C D alpha) 
  (h_segments : bisect_segment B C D BD DC)
  (h_feasible : feasibility_condition alpha) : 
  ∃ A B C : Point, 
    triangle A B C ∧ 
    ∠BAC = 2 * alpha ∧ 
    dist B D = BD ∧ 
    dist C D = DC := 
sorry

end construct_triangle_l482_482423


namespace quadratic_roots_abs_less_than_one_l482_482298

noncomputable def quadratic_roots (a b : ℝ) : ℝ × ℝ :=
if h : (a^2 - 4 * b) ≥ 0 then 
  let d := real.sqrt (a^2 - 4 * b) in
  ((-a + d) / 2, (-a - d) / 2)
else
  (0, 0)  -- placeholder in case there are no real roots

open real 

theorem quadratic_roots_abs_less_than_one (a b : ℝ) 
  (h1 : |a| + |b| < 1) 
  (h2 : a^2 - 4*b ≥ 0) : 
  let (r1, r2) := quadratic_roots a b in 
  |r1| < 1 ∧ |r2| < 1 :=
  sorry

end quadratic_roots_abs_less_than_one_l482_482298


namespace production_days_l482_482056

theorem production_days (n : ℕ) (P : ℕ) (H1 : P = n * 50) (H2 : (P + 90) / (n + 1) = 52) : n = 19 :=
by
  sorry

end production_days_l482_482056


namespace inequality_minus_x_plus_3_l482_482979

variable (x y : ℝ)

theorem inequality_minus_x_plus_3 (h : x < y) : -x + 3 > -y + 3 :=
by {
  sorry
}

end inequality_minus_x_plus_3_l482_482979


namespace sum_of_roots_of_polynomial_l482_482434

-- Definition of the polynomial
def polynomial (x : ℝ) := 5 * x^3 - 10 * x^2 - 72 * x - 15

-- Statement that the sum of roots is 2
theorem sum_of_roots_of_polynomial : 
  (let roots := multiset.map (λ x, 5 * x^3 - 10 * x^2 - 72 * x - 15) multiset.univ in
   multiset.sum roots) = 2 :=
sorry

end sum_of_roots_of_polynomial_l482_482434


namespace sum_of_a_and_b_l482_482731

variable a b : ℝ

-- condition based definitions
def line1 (y : ℝ) : ℝ := (1 / 3) * y + a
def line2 (x : ℝ) : ℝ := (1 / 3) * x + b
def point := (2, 3)

-- function stating the intersection point
def intersects_at (x y : ℝ) : Prop := line1 y = x ∧ line2 x = y

-- stating the main theorem to prove
theorem sum_of_a_and_b : 
  (intersects_at 2 3) → 
  (a + b = 10 / 3) :=
by 
  sorry

end sum_of_a_and_b_l482_482731


namespace fraction_inequality_l482_482117

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
  (b / a) < ((b + m) / (a + m)) :=
sorry

end fraction_inequality_l482_482117


namespace fraction_square_of_integer_l482_482923

theorem fraction_square_of_integer (a b : ℕ) (ha : a > 0) (hb : b > 0) (hdiv : (ab + 1) ∣ (a^2 + b^2)) :
  ∃ k : ℕ, \frac{a^2 + b^2}{1 + ab} = k^2 :=
sorry

end fraction_square_of_integer_l482_482923


namespace monic_poly_p0_p4_l482_482635

theorem monic_poly_p0_p4 (p : ℚ[X])
  (h_mon : p.monic)
  (deg_p : p.nat_degree = 4)
  (h_p1 : p.eval 1 = 17)
  (h_p2 : p.eval 2 = 34)
  (h_p3 : p.eval 3 = 51) :
  p.eval 0 + p.eval 4 = 92 :=
by sorry

end monic_poly_p0_p4_l482_482635


namespace probability_of_at_least_19_l482_482394

-- Defining the possible coins in Anya's pocket
def coins : list ℕ := [10, 10, 5, 5, 2]

-- Function to calculate the sum of chosen coins
def sum_coins (l : list ℕ) := list.sum l

-- Function to check if the sum of chosen coins is at least 19 rubles
def at_least_19 (l : list ℕ) := (sum_coins l) ≥ 19

-- Extract all possible combinations of 3 coins from the list
def combinations (l : list ℕ) (n : ℕ) := 
  if h : n ≤ l.length then 
    (list.permutations l).dedup.map (λ p, p.take n).dedup
  else
    []

-- Specific combinations of 3 coins out of 5
def three_coin_combinations := combinations coins 3 

-- Count the number of favorable outcomes (combinations that sum to at least 19)
def favorable_combinations := list.filter at_least_19 three_coin_combinations

-- Calculate the probability
def probability := (favorable_combinations.length : ℚ) / (three_coin_combinations.length : ℚ)

-- Prove that the probability is 0.4
theorem probability_of_at_least_19 : probability = 0.4 :=
  sorry

end probability_of_at_least_19_l482_482394


namespace point_not_in_transformed_plane_l482_482787

theorem point_not_in_transformed_plane :
  let A := (1, 1, 1)
  let plane := λ x y z, 7 * x - 6 * y + z - 5 = 0
  let k := -2
  let transformed_plane := λ x y z, 7 * x - 6 * y + z + 10 = 0
  ¬ transformed_plane 1 1 1 :=
by
  let A : ℝ × ℝ × ℝ := (1, 1, 1)
  let plane := λ x y z, 7 * x - 6 * y + z - 5 = 0
  let k : ℝ := -2
  let transformed_plane := λ x y z, 7 * x - 6 * y + z + 10 = 0
  show ¬ transformed_plane 1 1 1
  rw transformed_plane
  rw not_iff_not
  simp
  exact dec_trivial

end point_not_in_transformed_plane_l482_482787


namespace problem_proof_l482_482019

-- Let f be the polynomial defined as follows:
def f (x : ℝ) : ℝ := x^2010 + 18 * x^2009 + 2

-- Let the distinct roots of f be r_1, r_2, ..., r_{2010}
def roots : List ℝ := -- A list of distinct real numbers which are the roots of f
sorry

-- Define the polynomial P such that P(r_j + 1/r_j) = 0 for each j from 1 to 2010
def P (z : ℝ) : ℝ := sorry

-- The value we need to calculate:
def result : ℝ := (P 2) / (P (-2))

-- Prove that the result is 1
theorem problem_proof : result = 1 := 
by 
  sorry

end problem_proof_l482_482019


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482459

def seven_pow_seven_minus_seven_pow_four : ℤ := 7^7 - 7^4
def prime_factors_of_three_hundred_forty_two : List ℤ := [2, 3, 19]

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let distinct_prime_factors := prime_factors_of_three_hundred_forty_two.head!
  + prime_factors_of_three_hundred_forty_two.tail!.head!
  + prime_factors_of_three_hundred_forty_two.tail!.tail!.head!
  seven_pow_seven_minus_seven_pow_four = 7^4 * (7^3 - 1) ∧
  7^3 - 1 = 342 ∧
  prime_factors_of_three_hundred_forty_two = [2, 3, 19] ∧
  distinct_prime_factors = 24 := 
sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482459


namespace divisible_by_56_l482_482658

theorem divisible_by_56 (n : ℕ) (h1 : ∃ k, 3 * n + 1 = k * k) (h2 : ∃ m, 4 * n + 1 = m * m) : 56 ∣ n := 
sorry

end divisible_by_56_l482_482658


namespace simplify_sum_of_square_roots_l482_482685

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l482_482685


namespace limit_sum_infinite_geometric_series_l482_482518

noncomputable def infinite_geometric_series_limit (a_1 q : ℝ) :=
  if |q| < 1 then (a_1 / (1 - q)) else 0

theorem limit_sum_infinite_geometric_series :
  infinite_geometric_series_limit 1 (1 / 3) = 3 / 2 :=
by
  sorry

end limit_sum_infinite_geometric_series_l482_482518


namespace infinite_triples_exist_l482_482971

theorem infinite_triples_exist :
  ∃ᶠ (a b c : ℕ) in at_top, a < b ∧ b < c ∧ (∃ k1 k2 k3 : ℕ, ab + 1 = k1^2 ∧ bc + 1 = k2^2 ∧ ca + 1 = k3^2) := 
sorry

end infinite_triples_exist_l482_482971


namespace problem_l482_482534

def f (x : ℝ) (a : ℝ) := real.sqrt 3 * real.sin x * real.cos x + real.cos x ^ 2 + a

theorem problem (a : ℝ) :
  (∃ T > 0, ∀ x : ℝ, f x a = f (x + T) a) ∧ 
  (∀ k : ℤ, ∃ I, I = set.Icc (k * real.pi + real.pi / 6) (k * real.pi + 2 * real.pi / 3) ∧ ∀ x ∈ I, monotone_decreasing_on f x) ∧ 
  (a = 0) := by
  sorry

end problem_l482_482534


namespace stack_of_logs_total_l482_482370

-- Define the given conditions as variables and constants in Lean
def bottom_row : Nat := 15
def top_row : Nat := 4
def rows : Nat := bottom_row - top_row + 1
def sum_arithmetic_series (a l n : Nat) : Nat := n * (a + l) / 2

-- Define the main theorem to prove
theorem stack_of_logs_total : sum_arithmetic_series top_row bottom_row rows = 114 :=
by
  -- Here you will normally provide the proof
  sorry

end stack_of_logs_total_l482_482370


namespace angle_ratio_l482_482607

-- Conditions and setup
variables {α β γ : Type} [linear_ordered_field α]
variables {P Q B M : β}
variables (θ ψ : α)

-- Condition 1: BP and BQ bisect ∠ABC
axiom BQ_bisects_ABC : 
  ∀ (A B C : γ), ∠(A, B, P) = ∠(P, B, C)

axiom BP_bisects_ABC : 
  ∀ (A B C : γ), ∠(A, B, Q) = ∠(Q, B, C)

-- Condition 2: BM bisects ∠PBQ
axiom BM_bisects_PBQ : 
  ∠(P, B, M) = ∠(M, B, Q)

-- Prove the desired ratio
theorem angle_ratio (h1 : BQ_bisects_ABC) (h2 : BP_bisects_ABC) (h3 : BM_bisects_PBQ) : 
  θ / ψ = 1 / 4 :=
sorry

end angle_ratio_l482_482607


namespace xinxin_nights_at_seaside_l482_482318

-- Definitions from conditions
def arrival_day : ℕ := 30
def may_days : ℕ := 31
def departure_day : ℕ := 4
def nights_spent : ℕ := (departure_day + (may_days - arrival_day))

-- Theorem to prove the number of nights spent
theorem xinxin_nights_at_seaside : nights_spent = 5 := 
by
  -- Include proof steps here in actual Lean proof
  sorry

end xinxin_nights_at_seaside_l482_482318


namespace sqrt_sum_simplify_l482_482694

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l482_482694


namespace cody_final_tickets_l482_482407

def initial_tickets : ℝ := 56.5
def lost_tickets : ℝ := 6.3
def spent_tickets : ℝ := 25.75
def won_tickets : ℝ := 10.25
def dropped_tickets : ℝ := 3.1

theorem cody_final_tickets : 
  initial_tickets - lost_tickets - spent_tickets + won_tickets - dropped_tickets = 31.6 :=
by
  sorry

end cody_final_tickets_l482_482407


namespace next_terms_correct_l482_482895

def first_subsequence (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

def second_subsequence (n : ℕ) : ℕ :=
  2 ^ n

noncomputable def next_five_terms (seq : List ℕ) : List ℕ :=
  seq ++ [32, 11, 64, 13, 128]

theorem next_terms_correct (seq : List ℕ) :
  seq = [1, 2, 3, 4, 5, 8, 7, 16, 9] →
  next_five_terms seq = [1, 2, 3, 4, 5, 8, 7, 16, 9, 32, 11, 64, 13, 128] :=
by
  intro h
  simp [h, next_five_terms]
  sorry

end next_terms_correct_l482_482895


namespace determine_abs_r_l482_482253

theorem determine_abs_r (p q r : ℤ) (h1 : p * (1 + Complex.i)^4 + q * (1 + Complex.i)^3 + r * (1 + Complex.i)^2 + q * (1 + Complex.i) + p = 0)
(h2 : Int.gcd (Int.gcd p q) r = 1) : |r| = 7 :=
sorry

end determine_abs_r_l482_482253


namespace arthur_has_winning_strategy_l482_482627

theorem arthur_has_winning_strategy :
  (∀ (n: ℕ) (arthur_moves merlin_moves : fin n → ℝ × ℝ),
    (∀ i, inside_table (arthur_moves i) ∧ inside_table (merlin_moves i) ∧ 
      (∀ j, j ≠ i → distance (arthur_moves j) (arthur_moves i) > coin_radius ∧ 
             distance (merlin_moves j) (merlin_moves i) > coin_radius ∧
             distance (arthur_moves i) (merlin_moves j) > coin_radius
    )) →
    (∃ (arthur_final_move : ℝ × ℝ), inside_table (arthur_final_move) ∧ 
      ∀ j, distance arthur_final_move (arthur_moves j) > coin_radius ∧ 
           distance arthur_final_move (merlin_moves j) > coin_radius)) :=
begin
  sorry -- proof not required 
end

end arthur_has_winning_strategy_l482_482627


namespace sqrt_72_plus_sqrt_32_l482_482711

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l482_482711


namespace count_sets_without_perfect_squares_l482_482206

theorem count_sets_without_perfect_squares:
  let S'_i (i : ℕ) := { n : ℤ | 150 * i ≤ n ∧ n < 150 * (i + 1) } in
  ∃ p : ℕ, (p = 391) ∧
  let sets := finset.range 667 in
  (sets.filter (λ i, (∀ n ∈ S'_i i, ∀ m : ℕ, m*m ≠ n))).card = p :=
by
  sorry

end count_sets_without_perfect_squares_l482_482206


namespace count_values_cos2x_plus_3sin2x_eq_1_l482_482560

theorem count_values_cos2x_plus_3sin2x_eq_1 :
  ∃ n : ℕ, n = 48 ∧ ∀ x : ℝ, -30 < x ∧ x < 120 → cos x ^ 2 + 3 * sin x ^ 2 = 1 → x = n :=
sorry

end count_values_cos2x_plus_3sin2x_eq_1_l482_482560


namespace min_value_geometric_sequence_l482_482595

theorem min_value_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h : 0 < q ∧ 0 < a 0) 
  (H : 2 * a 3 + a 2 - 2 * a 1 - a 0 = 8) 
  (h_geom : ∀ n, a (n+1) = a n * q) : 
  2 * a 4 + a 3 = 12 * Real.sqrt 3 :=
sorry

end min_value_geometric_sequence_l482_482595


namespace quadratic_polynomial_properties_l482_482043

theorem quadratic_polynomial_properties :
  ∃ k : ℝ, (k * (3 - (3+4*I)) = 8 ∧ 
            (∀ x : ℂ, (x = (3 + 4 * I) → polynomial.eval x (k * (X - (3+4*I)) * (X - (3-4*I))) = 0)) ∧ 
            polynomial.coeff (k * (X - (3+4*I)) * (X - (3-4*I))) 1 = 8) :=
sorry

end quadratic_polynomial_properties_l482_482043


namespace sum_distinct_prime_factors_of_7pow7_minus_7pow4_l482_482455

noncomputable def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  let factors := (Nat.factors n).erase_dup
  factors.sum

theorem sum_distinct_prime_factors_of_7pow7_minus_7pow4 :
  sum_of_distinct_prime_factors (7 ^ 7 - 7 ^ 4) = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7pow7_minus_7pow4_l482_482455


namespace sqrt_72_plus_sqrt_32_l482_482706

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l482_482706


namespace general_equation_of_line_cartesian_equation_of_curve_range_of_OA_OB_l482_482162

-- Definitions for conditions
def parametric_eq_line (t α : ℝ) := (x = t * cos α) ∧ (y = t * sin α)
def polar_eq_curve (ρ θ : ℝ) := (ρ^2 - 4 * ρ * cos θ + 3 = 0)
def general_eq_line (x y α : ℝ) := (x * sin α - y * cos α = 0)
def cartesian_eq_curve (x y : ℝ) := (x^2 + y^2 - 4 * x + 3 = 0)

-- Main statements to prove
theorem general_equation_of_line (t α x y : ℝ) :
  parametric_eq_line t α → general_eq_line x y α :=
sorry

theorem cartesian_equation_of_curve (ρ θ x y : ℝ) :
  polar_eq_curve ρ θ → cartesian_eq_curve x y :=
sorry

theorem range_of_OA_OB (t1 t2 α : ℝ) (h : α ∈ (0, π/6)) :
  t1 + t2 = 4 * cos α → t1 * t2 = 3 → (|t1| + |t2|) ∈ (2 * sqrt 3, 4) :=
sorry

end general_equation_of_line_cartesian_equation_of_curve_range_of_OA_OB_l482_482162


namespace probability_of_picking_combination_is_0_4_l482_482388

noncomputable def probability_at_least_19_rubles (total_coins total_value: ℕ) :=
  let coins := [10, 10, 5, 5, 2] in
  let all_combinations := (Finset.powersetLen 3 (coins.to_finset)).to_list in
  let favorable_combinations := all_combinations.filter (fun c => c.sum ≥ total_value) in
  (favorable_combinations.length : ℚ) / (all_combinations.length : ℚ)

theorem probability_of_picking_combination_is_0_4 :
  probability_at_least_19_rubles 5 19 = 0.4 :=
by
  sorry

end probability_of_picking_combination_is_0_4_l482_482388


namespace TomTotalWeight_l482_482754

def TomWeight : ℝ := 150
def HandWeight (personWeight: ℝ) : ℝ := 1.5 * personWeight
def VestWeight (personWeight: ℝ) : ℝ := 0.5 * personWeight
def TotalHandWeight (handWeight: ℝ) : ℝ := 2 * handWeight
def TotalWeight (totalHandWeight vestWeight: ℝ) : ℝ := totalHandWeight + vestWeight

theorem TomTotalWeight : TotalWeight (TotalHandWeight (HandWeight TomWeight)) (VestWeight TomWeight) = 525 := 
by
  sorry

end TomTotalWeight_l482_482754


namespace find_a_l482_482520

noncomputable def f (x : ℝ) : ℝ := x^2 + 12
noncomputable def g (x : ℝ) : ℝ := x^2 - x - 4

theorem find_a (a : ℝ) (h_pos : a > 0) (h_fga : f (g a) = 12) : a = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end find_a_l482_482520


namespace quadratic_function_expression_range_of_f_log2x_l482_482737

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 1

theorem quadratic_function_expression :
  (∀ x, f(x+1) - f(x) = 2*x) ∧ (f 0 = 1) :=
begin
  split,
  { intro x,
    rw [f, f],
    suffices : ((x + 1)^2 - (x + 1) + 1) - (x^2 - x + 1) = 2 * x,
    exact this,
    norm_num [sq, mul_add, add_assoc, add_comm, add_left_comm],
  },
  { rw [f],
    simp, },
end

theorem range_of_f_log2x :
  (∀ x, 4^(x- (1 / 2)) - 3 * 2^x + 4 < 0) → (range (λ x, f(log x / log 2)) = Icc (3/4) 1) :=
begin
  intro h,
  sorry, -- Proof to be filled in.
end

end quadratic_function_expression_range_of_f_log2x_l482_482737


namespace math_problem_l482_482008

def calc_expr : ℝ := (1 / 8) ^ (-2 / 3) - Real.pi ^ 0 + Real.log10 100

theorem math_problem : calc_expr = 5 := by
  have term1 : (1 / 8) ^ (-2 / 3) = 4 := by -- condition 1
    sorry
  have term2 : Real.pi ^ 0 = 1 := by -- condition 2
    sorry
  have term3 : Real.log10 100 = 2 := by -- condition 3
    sorry
  rw [term1, term2, term3]
  linarith

end math_problem_l482_482008


namespace last_number_remaining_l482_482301

theorem last_number_remaining (n : ℕ) (h : n ≥ 1) : 
  ∃ C : ℚ, (∀ (s : finset ℚ), (∀ x ∈ s, x = 1 ∨ x = (1:ℚ) / (k + 1)) → 
    (∃ a b ∈ s, a ≠ b ∧ s.card = 1 → C = n)) :=
sorry

end last_number_remaining_l482_482301


namespace total_savings_correct_l482_482192

-- Definitions of savings per day and days saved for Josiah, Leah, and Megan
def josiah_saving_per_day : ℝ := 0.25
def josiah_days : ℕ := 24

def leah_saving_per_day : ℝ := 0.50
def leah_days : ℕ := 20

def megan_saving_per_day : ℝ := 1.00
def megan_days : ℕ := 12

-- Definition to calculate total savings for each child
def total_saving (saving_per_day : ℝ) (days : ℕ) : ℝ :=
  saving_per_day * days

-- Total amount saved by Josiah, Leah, and Megan
def total_savings : ℝ :=
  total_saving josiah_saving_per_day josiah_days +
  total_saving leah_saving_per_day leah_days +
  total_saving megan_saving_per_day megan_days

-- Theorem to prove the total savings is $28
theorem total_savings_correct : total_savings = 28 := by
  sorry

end total_savings_correct_l482_482192


namespace num_factors_36288_l482_482557

theorem num_factors_36288 : 
  (let n := 36288 in
   let prime_factors := (2, 6), (3, 4), (7, 1) in
   (prime_factors, (2^6 * 3^4 * 7) = n) →
   ((prime_factors.1.2 + 1) * (prime_factors.2.2 + 1) * (prime_factors.3.2 + 1)) = 70) :=
by
  sorry

end num_factors_36288_l482_482557


namespace total_water_volume_sum_l482_482289

def flow_rate_in_m_per_min (rate_in_kmph : ℕ) : ℝ :=
  (rate_in_kmph * 1000) / 60

def river_volume_per_minute (depth width flow_rate_kmph : ℕ) : ℝ :=
  let flow_rate_m_per_min := flow_rate_in_m_per_min(flow_rate_kmph)
  depth * width * flow_rate_m_per_min

theorem total_water_volume_sum (depth1 width1 fr1 depth2 width2 fr2 depth3 width3 fr3 : ℕ)
  (h1 : depth1 = 2) (h2 : width1 = 45) (h3 : fr1 = 5)
  (h4 : depth2 = 3.5) (h5 : width2 = 60) (h6 : fr2 = 4)
  (h7 : depth3 = 4) (h8 : width3 = 75) (h9 : fr3 = 3) :
  river_volume_per_minute depth1 width1 fr1 +
  river_volume_per_minute depth2 width2 fr2 +
  river_volume_per_minute depth3 width3 fr3 = 36500 :=
by
  sorry

end total_water_volume_sum_l482_482289


namespace part_I_part_II_l482_482951

noncomputable def f (x a : ℝ) := |2 * x - a| + |2 * x - 1|

theorem part_I (x : ℝ) : 
  (f x 3 ≤ 6) ↔ (-1/2 ≤ x ∧ x ≤ 5/2) :=
by {
  sorry
}

theorem part_II (a : ℝ) :
  (∀ x : ℝ, f x a ≥ a^2 - a - 13) ↔ (-Real.sqrt 14 ≤ a ∧ a ≤ 1 + Real.sqrt 13) :=
by {
  sorry
}

end part_I_part_II_l482_482951


namespace triangleProblem_correct_l482_482070

noncomputable def triangleProblem : Prop :=
  ∃ (a b c A B C : ℝ),
    A = 60 * Real.pi / 180 ∧
    b = 1 ∧
    (1 / 2) * b * c * Real.sin A = Real.sqrt 3 ∧
    Real.cos A = 1 / 2 ∧
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A ∧
    (a / Real.sin A) = (b / Real.sin B) ∧ (b / Real.sin B) = (c / Real.sin C) ∧
    (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3

theorem triangleProblem_correct : triangleProblem :=
  sorry

end triangleProblem_correct_l482_482070


namespace fifth_lucky_of_2005_l482_482139

def sum_of_digits (a : ℕ) : ℕ :=
  a.digits.sum

def is_lucky_number (a : ℕ) : Prop :=
  sum_of_digits a = 7

def lucky_numbers : List ℕ :=
  List.filter (λ x, is_lucky_number x) (List.range 100000)

def nth_lucky_number (n : ℕ) : ℕ :=
  lucky_numbers.getOrElse n 0

theorem fifth_lucky_of_2005 :
  ∃ n : ℕ, nth_lucky_number n = 2005 → nth_lucky_number (5 * n) = 52000 :=
sorry

end fifth_lucky_of_2005_l482_482139


namespace probability_different_plants_l482_482360

-- Define the types of plants as an enum
inductive Plant
| Pothos
| LuckyBamboo
| Jade
| Aloe

open Plant

def all_pairs (pl1 pl2 : Plant) :=
  [(Pothos, Pothos), (Pothos, LuckyBamboo), (Pothos, Jade), (Pothos, Aloe),
   (LuckyBamboo, Pothos), (LuckyBamboo, LuckyBamboo), (LuckyBamboo, Jade), (LuckyBamboo, Aloe),
   (Jade, Pothos), (Jade, LuckyBamboo), (Jade, Jade), (Jade, Aloe),
   (Aloe, Pothos), (Aloe, LuckyBamboo), (Aloe, Jade), (Aloe, Aloe)]

-- Condition: total number of pairs
def total_pairs : ℕ := 16

-- Condition: same plant pairs
def same_plant_pairs : List (Plant × Plant) :=
  [ (Pothos, Pothos), (LuckyBamboo, LuckyBamboo), (Jade, Jade), (Aloe, Aloe) ]

-- Theorem statement (proof omitted)
theorem probability_different_plants: 
  (total_pairs - List.length same_plant_pairs) / total_pairs = 13 / 16 := by
  sorry

end probability_different_plants_l482_482360


namespace prove_smallest_x_l482_482799

noncomputable def f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 3 then 1 - |x - 2| else sorry

theorem prove_smallest_x (x_min : ℝ) :
  (∀ x : ℝ, 0 < x → f(3 * x) = 3 * f(x)) →
  (∀ y : ℝ, 1 ≤ y ∧ y ≤ 3 → f(y) = 1 - |y - 2|) →
  (f 2001 = f x_min ∧ (∀ x' < x_min, f x' ≠ f 2001)) :=
sorry

end prove_smallest_x_l482_482799


namespace incoming_students_count_l482_482258

theorem incoming_students_count :
  ∃ (n : ℕ), n < 600 ∧ n % 26 = 25 ∧ n % 24 = 15 ∧ n = 519 :=
begin
  sorry
end

end incoming_students_count_l482_482258


namespace appropriate_word_count_l482_482230

-- Define the conditions of the problem
def min_minutes := 40
def max_minutes := 55
def words_per_minute := 120

-- Define the bounds for the number of words
def min_words := min_minutes * words_per_minute
def max_words := max_minutes * words_per_minute

-- Define the appropriate number of words
def appropriate_words (words : ℕ) : Prop :=
  words >= min_words ∧ words <= max_words

-- The specific numbers to test
def words1 := 5000
def words2 := 6200

-- The main proof statement
theorem appropriate_word_count : 
  appropriate_words words1 ∧ appropriate_words words2 :=
by
  -- We do not need to provide the proof steps, just state the theorem
  sorry

end appropriate_word_count_l482_482230


namespace megan_correct_number_prob_l482_482224

theorem megan_correct_number_prob :
  let first_choices := {295, 296, 299}
  ∧ let fourth_digit_choices := {6, 7}
  ∧ let num_of_last_four_permutations := 12 + 12
  ∧ let total_numbers := 3 * 24
  ∧ let correct_number_in_total := 1 in
  (1 / total_numbers) = (1 / 72) :=
by
  unfold first_choices fourth_digit_choices num_of_last_four_permutations total_numbers correct_number_in_total
  sorry

end megan_correct_number_prob_l482_482224


namespace number_of_correct_propositions_is_zero_l482_482733

-- Declare the propositions as Lean definitions
def proposition_1 : Prop := ∀ (pyramid : Type) (section : Type), ¬is_frustum (cut_with_plane pyramid section)
def proposition_2 : Prop := ∀ (polyhedron : Type), ¬(has_parallel_similar_bases polyhedron ∧ all_other_faces_trapezoids polyhedron → is_frustum polyhedron)
def proposition_3 : Prop := ∀ (hexahedron : Type), ¬(has_parallel_faces hexahedron ∧ other_faces_isosceles_trapezoids hexahedron → is_frustum hexahedron)

-- Problem statement
theorem number_of_correct_propositions_is_zero :
  ∀ (correct_prop_count : ℕ), correct_prop_count = 0 :=
by
  -- Correct proof follows based on given conditions
  sorry

end number_of_correct_propositions_is_zero_l482_482733


namespace vector_calculation_l482_482501

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (2, 1)
def vec_b : ℝ × ℝ := (-3, 4)

-- Define scalar multiplication and vector addition functions
def smul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vadd (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

-- State the theorem to be proved
theorem vector_calculation : 
  3 • vec_a + 4 • vec_b = (-6, 19) :=
by
  conv_rhs { rw [smul, vadd] } -- Use the smul and vadd definitions
  sorry

end vector_calculation_l482_482501


namespace prove_inequality_l482_482095

-- Given conditions
variables {a b : ℝ}
variable {x : ℝ}
variable h : 0 < a
variable k : 0 < b
variable l : ∀ (x : ℝ), (1 ≤ x ∧ x ≤ 2) → (abs(x + a) + 2 * abs(x - 1) > x^2 - b + 1)

-- To prove (a + 1/2)^2 + (b + 1/2)^2 > 2
theorem prove_inequality (h : 0 < a) (k : 0 < b) (l : ∀ (x : ℝ), (1 ≤ x ∧ x ≤ 2) → (abs(x + a) + 2 * abs(x - 1) > x^2 - b + 1)) :
  (a + 1/2)^2 + (b + 1/2)^2 > 2 :=
sorry

end prove_inequality_l482_482095


namespace four_digit_number_count_l482_482112

theorem four_digit_number_count :
  let choices_first_two := {1, 4, 5, 6}
  let choices_last_two := {0, 5, 7}
  let choices_third_digit := {x | x ∈ choices_first_two ∧ (x % 2 = 1)}
  ∃ (n : ℕ), n = 192 ∧ 
    count (λ (w : Fin 4 → Fin 10),
      w 0 ∈ choices_first_two ∧
      w 1 ∈ choices_first_two ∧
      w 2 ∈ choices_third_digit ∧
      w 3 ≠ w 2 ∧
      w 3 ∈ choices_last_two ∧
      (∃ y z, {w 3, w 2} = {y, z}))
    = n :=
begin
  let choices_first_two := {1, 4, 5, 6},
  let choices_last_two := {0, 5, 7},
  let choices_third_digit := {x | x ∈ choices_first_two ∧ (x % 2 = 1)},
  have : count (λ (w : Fin 4 → Fin 10),
      w 0 ∈ choices_first_two ∧
      w 1 ∈ choices_first_two ∧
      w 2 ∈ choices_third_digit ∧
      w 3 ≠ w 2 ∧
      w 3 ∈ choices_last_two ∧
      (∃ y z, {w 3, w 2} = {y, z})) = 192,
  { sorry, },
  exact ⟨192, rfl, this⟩,
end

end four_digit_number_count_l482_482112


namespace calculate_sum_of_cubes_l482_482017

theorem calculate_sum_of_cubes :
  let u := Root (x - 3) (x - real.cbrt(45)) (x - real.cbrt(81));
  let v := Root (x - 3) (x - real.cbrt(45)) (x - real.cbrt(81));
  let w := Root (x - 3) (x - real.cbrt(45)) (x - real.cbrt(81));
  u * v * w = real.cbrt(27) * real.cbrt(45) * real.cbrt(81) + 2 / 5 →
  u^3 + v^3 + w^3 = 153.6 + 118.6025 * (10.8836 - 3 * (real.cbrt(27) * real.cbrt(45) + real.cbrt(27) * real.cbrt(81) + real.cbrt(45) * real.cbrt(81))) - sorry :=
sorry

end calculate_sum_of_cubes_l482_482017


namespace average_bike_speed_l482_482031

noncomputable def swimSpeed : ℝ := 1
noncomputable def swimDistance : ℝ := 0.5
noncomputable def runSpeed : ℝ := 8
noncomputable def runDistance : ℝ := 4
noncomputable def totalTriathlonTime : ℝ := 3
noncomputable def bikeDistance : ℝ := 20

theorem average_bike_speed :
  let t_swim := swimDistance / swimSpeed in
  let t_run := runDistance / runSpeed in
  let t_total := t_swim + t_run in
  let t_bike := totalTriathlonTime - t_total in
  t_bike ≠ 0 → bikeDistance / t_bike = 10 :=
by
  intro h_nonzero_t_bike
  -- proof skipped
  sorry

end average_bike_speed_l482_482031


namespace sum_first_2501_terms_l482_482816

-- Given conditions
def seq (b : ℕ → ℤ) : Prop :=
  ∀ n, n ≥ 3 → b n = b (n - 1) - b (n - 2)

def sum_first_n_terms (b : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, b i

def condition_1850 (b : ℕ → ℤ) : Prop :=
  sum_first_n_terms b 1850 = 2022

def condition_2022 (b : ℕ → ℤ) : Prop :=
  sum_first_n_terms b 2022 = 1850

-- Proof goal
theorem sum_first_2501_terms (b : ℕ → ℤ) :
  seq b →
  condition_1850 b →
  condition_2022 b →
  sum_first_n_terms b 2501 = -172 :=
by {
  intros,
  -- Proof task, skipped using sorry
  sorry
}

end sum_first_2501_terms_l482_482816


namespace coefficient_x_term_binomial_l482_482886

noncomputable def binomial_coefficient (n : ℕ) (k : ℕ) : ℤ :=
  if h : k ≤ n then (nat.choose n k).toNat else 0

theorem coefficient_x_term_binomial :
  let a := (5 : ℕ)
  let x := ?m_1
  let coeff := (-2) ^ 3 * binomial_coefficient 5 3
  (x^2 - ((2 : ℚ) / x))^a = (-80 : ℤ) * x :=
sorry

end coefficient_x_term_binomial_l482_482886


namespace correct_statements_l482_482212

variable (m n : ℝ) (a : ℝ)

-- Statement 2 Condition
def statement2 (m n a : ℝ) := ma^2 < na^2 → m < n

-- Statement 4 Condition
def statement4 (m n : ℝ) := (m < n ∧ n < 0) → (n / m < 1)

-- The theorem to prove which statements are correct
theorem correct_statements (h2 : statement2 m n a) (h4 : statement4 m n) : 
  ∀ (m n a : ℝ), (statement2 m n a ∧ statement4 m n) :=
begin
  sorry,
end

end correct_statements_l482_482212


namespace sum_of_squares_of_sequence_l482_482103

theorem sum_of_squares_of_sequence (a : ℕ → ℕ) (n : ℕ) (h : ∑ i in finset.range n.succ, a i = 2^n.succ - 1) : 
  ∑ i in finset.range n.succ, (a i)^2 = (1/3) * (4^n.succ - 1) := 
by
  sorry

end sum_of_squares_of_sequence_l482_482103


namespace number_picked_by_person_10_is_13_l482_482444

-- Define the setting of 15 people
variable (a : Fin 15 → ℕ)

-- Define the conditions
def condition_1 := (a 4 + a 6) = 24
def condition_2 := (a 9 + a 11) = 14
def condition_3 := (Finset.univ.sum a) = 120

-- Define the main statement
theorem number_picked_by_person_10_is_13 (a : Fin 15 → ℕ) 
  (h1 : condition_1 a) 
  (h2 : condition_2 a) 
  (h3 : condition_3 a) : 
  a 10 = 13 :=
begin
  sorry
end

end number_picked_by_person_10_is_13_l482_482444


namespace bisector_varies_l482_482507

-- Define the conditions
variable (circle : Type) [CircularGeometry circle]
variable (A B O C D P : circle)
variable (diameter_AB : CircleDiameter A B circle)
variable (center_O : CircleCenter O circle)
variable (on_circle : OnCircle C circle)
variable (chord_CD : Chord C D circle)
variable (bisector_P : AngleBisector P (Angle_OCD O C D) circle)

-- Define the theorem to prove that P varies as C moves
theorem bisector_varies
  (H : varies (λ C, bisector_P C)) :
  True
:= sorry

noncomputable theory

end bisector_varies_l482_482507


namespace smallest_root_of_quadratic_l482_482050

theorem smallest_root_of_quadratic (y : ℝ) (h : 4 * y^2 - 7 * y + 3 = 0) : y = 3 / 4 :=
sorry

end smallest_root_of_quadratic_l482_482050


namespace angle_APB_third_AOB_l482_482628

theorem angle_APB_third_AOB {O A B C D C' D' P : Point}
  (h_circle : Circle O)
  (h_chord : Chord A B)
  (h_arc : SmallestArc A B)
  (h_congruent_arcs : Arc A C ≃ Arc C D ∧ Arc C D ≃ Arc D B)
  (h_equal_segments : Segment A C' = Segment C' D' ∧ Segment C' D' = Segment D' B)
  (h_point_P : P = intersect (Line C C') (Line D D')) :
  Angle A P B = 1 / 3 * Angle A O B :=
  sorry

end angle_APB_third_AOB_l482_482628


namespace sequence_bn_arithmetic_sum_first_n_terms_l482_482919

variable (a : ℕ → ℝ)
variable {n : ℕ}
variable (h_pos : ∀ n, 0 < a n)
variable (h_a1 : a 1 = 1)
variable (h_rec : ∀ n, a (n+1) * a n + a (n+1) - a n = 0)

theorem sequence_bn_arithmetic (b : ℕ → ℝ) (hb_def : ∀ n, b n = 1 / a n) :
  ∃ d : ℝ, ∀ n, b (n + 1) - b n = d := sorry

theorem sum_first_n_terms (c : ℕ → ℝ) (hc_def : ∀ n, c n = a n / (n + 1)) :
  ∀ n, (∑ i in Finset.range n, c (i + 1)) = n / (n + 1) := sorry

end sequence_bn_arithmetic_sum_first_n_terms_l482_482919


namespace points_distance_leq_half_l482_482233

theorem points_distance_leq_half 
  (T : Type) [regular_tetrahedron T]
  (points : finset T) 
  (h1 : points.card = 9)
  (h2 : ∀ (x y : T), x ∈ points → y ∈ points → x ≠ y → dist x y ≤ 1) :
  ∃ (x y : T), x ∈ points ∧ y ∈ points ∧ x ≠ y ∧ dist x y ≤ 0.5 :=
by
  sorry

end points_distance_leq_half_l482_482233


namespace chef_michel_total_pies_l482_482864

theorem chef_michel_total_pies 
  (shepherd_pie_pieces : ℕ) 
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ) 
  (chicken_pot_pie_customers : ℕ) 
  (h1 : shepherd_pie_pieces = 4)
  (h2 : chicken_pot_pie_pieces = 5)
  (h3 : shepherd_pie_customers = 52)
  (h4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) +
  (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by {
  sorry
}

end chef_michel_total_pies_l482_482864


namespace cloth_sale_total_amount_l482_482817

theorem cloth_sale_total_amount :
  let CP := 70 -- Cost Price per metre in Rs.
  let Loss := 10 -- Loss per metre in Rs.
  let SP := CP - Loss -- Selling Price per metre in Rs.
  let total_metres := 600 -- Total metres sold
  let total_amount := SP * total_metres -- Total amount from the sale
  total_amount = 36000 := by
  sorry

end cloth_sale_total_amount_l482_482817


namespace tangent_line_equation_l482_482446

theorem tangent_line_equation :
  ∀ (line : ℝ → ℝ → Prop),
    (∀ x y, line x y ↔ x - 2)^2 + y^2 = 2 ∧
    ∃ a. ∀ b, line a b ↔ a = b ∨ a = -4 :=
  sorry

end tangent_line_equation_l482_482446


namespace sqrt_sum_simplify_l482_482691

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l482_482691


namespace series_diverges_l482_482030

noncomputable def sequence (n : ℕ) : ℝ := (1 / n) * (1 / (1 + 1 / n))

theorem series_diverges : ¬ summable (λ n, sequence n) := by
  sorry

end series_diverges_l482_482030


namespace books_loaned_out_eq_80_l482_482325

open_locale classical
noncomputable theory

-- Definitions for our problem:
def initial_books := 150
def returned_percent := 0.65
def end_books := 122
def missing_books := initial_books - end_books
def not_returned_percent := 1 - returned_percent

-- Theorem statement:
theorem books_loaned_out_eq_80 (x : ℝ) 
  (h1 : not_returned_percent * x = missing_books) : 
  x = 80 := 
by 
  sorry

end books_loaned_out_eq_80_l482_482325


namespace sin_theta_is_sqrt3_div2_l482_482218

open Real

variables {a b c : ℝ × ℝ × ℝ} 

-- Conditions: nonzero vectors and no two are parallel
axiom a_nonzero : a ≠ (0, 0, 0)
axiom b_nonzero : b ≠ (0, 0, 0)
axiom c_nonzero : c ≠ (0, 0, 0)
axiom a_b_not_parallel : a ≠ b ∧ a ≠ -b
axiom b_c_not_parallel : b ≠ c ∧ b ≠ -c
axiom a_c_not_parallel : a ≠ c ∧ a ≠ -c

-- Given condition
axiom given_condition : (a ×ₗ b) ×ₗ c = (1/2) * ∥b∥ * ∥c∥ • a

-- Definition of angle and its sine
noncomputable def angle_sin (u v : ℝ × ℝ × ℝ) : ℝ := sin (angle u v)

-- Goal to prove
theorem sin_theta_is_sqrt3_div2 : angle_sin b c = (sqrt 3) / 2 :=
sorry

end sin_theta_is_sqrt3_div2_l482_482218


namespace minimum_sum_l482_482027

theorem minimum_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a)) ≥ (3 / Real.cbrt 162) :=
  sorry

end minimum_sum_l482_482027


namespace cubic_polynomial_roots_ratio_l482_482913

theorem cubic_polynomial_roots_ratio (a b c d x1 x2 x3 : ℝ) 
  (h1 : p(x) = a * x^3 + b * x^2 + c * x + d)
  (h2 : p(1/2) + p(-1/2) = 1000 * p(0))
  (hx1 : p(x1) = 0)
  (hx2 : p(x2) = 0)
  (hx3 : p(x3) = 0) :
  (1 / (x1 * x2)) + (1 / (x2 * x3)) + (1 / (x1 * x3)) = 1996 :=
sorry

end cubic_polynomial_roots_ratio_l482_482913


namespace inverse_value_l482_482528

-- Assuming f is a function from ℝ to ℝ and has an inverse.
variable {f : ℝ → ℝ}
variable {f_inv : ℝ → ℝ}
variable {h_inv : Function.Inverse f f_inv}

-- The symmetry condition about the point M(1, -2).
def symmetric_about_M (f : ℝ → ℝ) := ∀ x, f(x + 1) + f(1 - x) = -4

-- Given values for the function f
variable {h_fval : f 2011 = 2008}

theorem inverse_value :
  symmetric_about_M f →
  Function.Bijective f →
  f 2011 = 2008 →
  f_inv (-2012) = -2009 :=
by
  intro h_symm h_fbij h_f2011
  -- proof goes here
  sorry

end inverse_value_l482_482528


namespace find_k_l482_482983

theorem find_k (x y k : ℝ) (hx1 : x - 4 * y + 3 ≤ 0) (hx2 : 3 * x + 5 * y - 25 ≤ 0) (hx3 : x ≥ 1)
  (hmax : ∃ (z : ℝ), z = 12 ∧ z = k * x + y) (hmin : ∃ (z : ℝ), z = 3 ∧ z = k * x + y) :
  k = 2 :=
sorry

end find_k_l482_482983


namespace integral_inequality_l482_482495

variable {a b c : ℝ} {f : ℝ → ℝ}

def f (x : ℝ) : ℝ := a * x^2 + b * x + c
def f' (x : ℝ) : ℝ := 2 * a * x + b

theorem integral_inequality 
  (h1 : ∫ x in -1..1, (1 - x^2) * (f'(x))^2 ≤ 6 * ∫ x in -1..1, (f(x))^2) : 
  ∫ x in -1..1, (1 - x^2) * (f'(x))^2 ≤ 6 * ∫ x in -1..1, (f(x))^2 :=
by
  sorry

end integral_inequality_l482_482495


namespace division_remainder_l482_482992

theorem division_remainder (dividend divisor quotient remainder : ℕ)
  (h₁ : dividend = 689)
  (h₂ : divisor = 36)
  (h₃ : quotient = 19)
  (h₄ : dividend = divisor * quotient + remainder) :
  remainder = 5 :=
by
  sorry

end division_remainder_l482_482992


namespace intersection_M_N_l482_482107

open Set

theorem intersection_M_N : 
  let M := { x : ℝ | log10 x > 0 }
  let N := { x : ℝ | x^2 ≤ 4 }
  M ∩ N = { x : ℝ | 1 < x ∧ x ≤ 2 } := 
by
  intro M N
  sorry

end intersection_M_N_l482_482107


namespace sqrt_72_plus_sqrt_32_l482_482707

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l482_482707


namespace last_digit_inverse_power_two_l482_482762

theorem last_digit_inverse_power_two :
  let n := 12
  let x := 5^n
  let y := 10^n
  (x % 10 = 5) →
  ((1 / (2^n)) * (5^n) / (5^n) == (5^n) / (10^n)) →
  (y % 10 = 0) →
  ((1 / (2^n)) % 10 = 5) :=
by
  intros n x y h1 h2 h3
  sorry

end last_digit_inverse_power_two_l482_482762


namespace trajectory_midpoint_l482_482771

theorem trajectory_midpoint (P Q M : ℝ × ℝ)
  (hP : P.1^2 + P.2^2 = 1)
  (hQ : Q.1 = 3 ∧ Q.2 = 0)
  (hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (2 * M.1 - 3)^2 + 4 * M.2^2 = 1 :=
sorry

end trajectory_midpoint_l482_482771


namespace total_pies_sold_l482_482860

def shepherds_pie_slices_per_pie : Nat := 4
def chicken_pot_pie_slices_per_pie : Nat := 5
def shepherds_pie_slices_ordered : Nat := 52
def chicken_pot_pie_slices_ordered : Nat := 80

theorem total_pies_sold :
  shepherds_pie_slices_ordered / shepherds_pie_slices_per_pie +
  chicken_pot_pie_slices_ordered / chicken_pot_pie_slices_per_pie = 29 := by
sorry

end total_pies_sold_l482_482860


namespace point_inside_polygon_odd_marked_vertices_l482_482299

/-- Given a polygon, a line l, and a point P on l in general position such that all lines containing a side of the polygon meet l at distinct points different from P, and each vertex of the polygon is marked if the sides meeting there, when extended, intersect the line l on opposite sides of P. Show that P lies inside the polygon if and only if on each side of l there are an odd number of marked vertices. -/
theorem point_inside_polygon_odd_marked_vertices (polygon : Type)
  (vertices : list polygon)
  (l : Type)
  (P : l) 
  (general_position : ∀ (side : polygon), side.intersection_points l ≠ P)
  (mark_vertex : ∀ (v : polygon), (v.side_intersects l on_opposite_side_of P) → marked v) :
  (P inside polygon ↔ (∀ side_of_l : l, odd_count (marked_vertices side_of_l))) :=
sorry

end point_inside_polygon_odd_marked_vertices_l482_482299


namespace shaded_square_ensures_all_columns_l482_482812

def first_shaded_square_in_each_column (n : ℕ) : Prop :=
  let board := list.range (12 * 12 + 1) in
  ∀ i, i < 12 → ∃ x, x ∈ board ∧ (x^2 % 12 = i)

theorem shaded_square_ensures_all_columns :
  first_shaded_square_in_each_column 12 = 144 :=
by
  sorry

end shaded_square_ensures_all_columns_l482_482812


namespace tomatoes_left_after_yesterday_correct_l482_482353

def farmer_initial_tomatoes := 160
def tomatoes_picked_yesterday := 56
def tomatoes_left_after_yesterday : ℕ := farmer_initial_tomatoes - tomatoes_picked_yesterday

theorem tomatoes_left_after_yesterday_correct :
  tomatoes_left_after_yesterday = 104 :=
by
  unfold tomatoes_left_after_yesterday
  -- Proof goes here
  sorry

end tomatoes_left_after_yesterday_correct_l482_482353


namespace slope_of_line_AB_is_minus2_l482_482059

theorem slope_of_line_AB_is_minus2 :
  ∀ (A B : ℝ × ℝ),
  A = (-1, 2) →
  B = (-2, 4) →
  let slope := (B.2 - A.2) / (B.1 - A.1)
  in slope = -2 :=
by
  intros A B hA hB
  rw [hA, hB]
  simp
  sorry

end slope_of_line_AB_is_minus2_l482_482059


namespace g_neg6_eq_43_over_16_l482_482208

def f (x : ℝ) : ℝ := 4 * x - 9
def g (y : ℝ) : ℝ := 3 * (y / 4)d^2 + 4 * (y / 4) - 2

theorem g_neg6_eq_43_over_16 :
  g (-6) = 43 / 16 :=
sorry

end g_neg6_eq_43_over_16_l482_482208


namespace first_year_after_2020_sum_4_l482_482987

theorem first_year_after_2020_sum_4 : 
  ∃ (y : ℕ), y > 2020 ∧ (∑ d in (y.digits 10), d) = 4 ∧ ∀ z, 2020 < z < y → (∑ d in (z.digits 10), d) ≠ 4 :=
sorry

end first_year_after_2020_sum_4_l482_482987


namespace angle_ratio_l482_482613

-- Define the angles and their properties
variables (A B C P Q M : Type)
variables (mABQ mMBQ mPBQ : ℝ)

-- Define the conditions from the problem:
-- 1. BP and BQ bisect ∠ABC
-- 2. BM bisects ∠PBQ
def conditions (h1 : 2 * mPBQ = mABQ)
               (h2 : 2 * mMBQ = mPBQ) : Prop :=
  true

-- Translate the question and correct answer into a Lean definition.
def find_ratio (h1 : 2 * mPBQ = mABQ) 
               (h2 : 2 * mMBQ = mPBQ) : Prop :=
  mMBQ / mABQ = 1 / 4

-- Now define the theorem that encapsulates the problem statement
theorem angle_ratio (h1 : 2 * mPBQ = mABQ) 
                    (h2 : 2 * mMBQ = mPBQ) :
  find_ratio A B C P Q M mABQ mMBQ mPBQ h1 h2 :=
by
  -- Proof to be provided.
  sorry

end angle_ratio_l482_482613


namespace domain_g_l482_482888

noncomputable def g (x : ℝ) := Real.tan (Real.arcsin (x ^ 3))

theorem domain_g : { x : ℝ | -1 < x ∧ x < 1 } = { x : ℝ | g x ≠ ⊥ } := 
by
  sorry

end domain_g_l482_482888


namespace trailing_zeros_of_square_l482_482976

theorem trailing_zeros_of_square :
  ∀ (n : ℕ), n = 899999999999 →
  ∃ k : ℕ, k = 17 ∧
  (∃ m : ℕ, n = 9 * 10^11 - 1 ∧
  ∃ p : ℕ, n^2 = 81 * 10^22 - 18 * 10^11 + 1 ∧
  (m = (n^2 % 10^(k+1)) / 10^(k) ∧ m = 0)) :=
begin
  sorry
end

end trailing_zeros_of_square_l482_482976


namespace inequality_proof_l482_482241

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a + b + c ≤ (a^2 + b^2) / (2 * c) + (a^2 + c^2) / (2 * b) + (b^2 + c^2) / (2 * a) ∧ 
    (a^2 + b^2) / (2 * c) + (a^2 + c^2) / (2 * b) + (b^2 + c^2) / (2 * a) ≤ (a^3 / (b * c)) + (b^3 / (a * c)) + (c^3 / (a * b)) := 
by
  sorry

end inequality_proof_l482_482241


namespace simplify_radicals_l482_482703

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l482_482703


namespace proof_problem_l482_482960

open Set Real

def M : Set ℝ := { x : ℝ | ∃ y : ℝ, y = log (1 - 2 / x) }
def N : Set ℝ := { x : ℝ | ∃ y : ℝ, y = sqrt (x - 1) }

theorem proof_problem : N ∩ (U \ M) = Icc 1 2 := by
  sorry

end proof_problem_l482_482960


namespace range_of_a_l482_482106

open Set

theorem range_of_a : {a : ℝ | (2a-3, a+1) ∪ {x | x ≤ -2 ∨ x > 1} = univ} = Ioo 0 (1 / 2) ∪ {1 / 2} := sorry

end range_of_a_l482_482106


namespace same_remainder_division_l482_482904

theorem same_remainder_division (k r a b c d : ℕ) 
  (h_k_pos : 0 < k)
  (h_nonzero_r : 0 < r)
  (h_r_lt_k : r < k)
  (a_def : a = 2613)
  (b_def : b = 2243)
  (c_def : c = 1503)
  (d_def : d = 985)
  (h_a : a % k = r)
  (h_b : b % k = r)
  (h_c : c % k = r)
  (h_d : d % k = r) : 
  k = 74 ∧ r = 23 := 
by
  sorry

end same_remainder_division_l482_482904


namespace ark5_ensures_metabolic_energy_l482_482662

-- Define conditions
def inhibits_ark5_activity (inhibits: Bool) (balance: Bool): Prop :=
  if inhibits then ¬balance else balance

def cancer_cells_proliferate_without_energy (proliferate: Bool) (die_due_to_insufficient_energy: Bool) : Prop :=
  proliferate → die_due_to_insufficient_energy

-- Define the hypothesis based on conditions
def hypothesis (inhibits: Bool) (balance: Bool) (proliferate: Bool) (die_due_to_insufficient_energy: Bool): Prop :=
  inhibits_ark5_activity inhibits balance ∧ cancer_cells_proliferate_without_energy proliferate die_due_to_insufficient_energy

-- Define the theorem to be proved
theorem ark5_ensures_metabolic_energy
  (inhibits : Bool)
  (balance : Bool)
  (proliferate : Bool)
  (die_due_to_insufficient_energy : Bool)
  (h : hypothesis inhibits balance proliferate die_due_to_insufficient_energy) :
  ensures_metabolic_energy :=
  sorry

end ark5_ensures_metabolic_energy_l482_482662


namespace algebra_comparison_l482_482275

theorem algebra_comparison (a : ℝ) : 
  let x := (a + 3) * (a - 5) in
  let y := (a + 2) * (a - 4) in
  x < y :=
by
  sorry

end algebra_comparison_l482_482275


namespace find_x_l482_482109

noncomputable def a (x : ℝ) : ℝ × ℝ × ℝ := (1, 1, x)
def b : ℝ × ℝ × ℝ := (1, 2, 1)
def c : ℝ × ℝ × ℝ := (1, 1, 1)
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_x (x : ℝ) :
  dot_product ((c.1 - a x.1, c.2 - a x.2, c.3 - a x.3) : ℝ × ℝ × ℝ) (2*b.1, 2*b.2, 2*b.3) = -2 ↔ x = 2 :=
by sorry

end find_x_l482_482109


namespace max_lucky_numbers_l482_482998

def is_lucky (k : ℕ) (n : ℕ) (grid : ℕ → ℕ → Prop) : Prop :=
  k ≤ n ∧ ∀ i j, i + k ≤ n ∧ j + k ≤ n →
  (grid i j).card (λ (i', j'), i ≤ i' ∧ i' < i + k ∧ j ≤ j' ∧ j' < j + k ∧ grid i' j') = k

theorem max_lucky_numbers (n : ℕ) (grid : ℕ → ℕ → Prop) (h1 : n = 2016)
  (h2 : ∀ i j k, k = is_lucky k n grid) : ∃ l, l ≤ n ∧ ∀ k, is_lucky k n grid → k ≤ l :=
begin
  sorry
end

end max_lucky_numbers_l482_482998


namespace smallest_four_digit_negative_congruent_one_mod_37_l482_482308

theorem smallest_four_digit_negative_congruent_one_mod_37 :
  ∃ (x : ℤ), x < -999 ∧ x >= -10000 ∧ x ≡ 1 [MOD 37] ∧ ∀ y, y < -999 ∧ y >= -10000 ∧ y ≡ 1 [MOD 37] → y ≥ x :=
sorry

end smallest_four_digit_negative_congruent_one_mod_37_l482_482308


namespace exterior_angle_bisector_ratio_l482_482986

theorem exterior_angle_bisector_ratio (A B C P : Point) (h1: SegmentRatio A C B C = 1 / 2) (h2: IsExteriorAngleBisector C P A B) : 
  SegmentRatio P B A B = 2 / 1 := by
  sorry

end exterior_angle_bisector_ratio_l482_482986


namespace inequality_solution_set_l482_482135

theorem inequality_solution_set (m n : ℝ) 
    (h₁ : ∀ x : ℝ, mx - n > 0 ↔ x < 1 / 3) 
    (h₂ : m + n < 0) 
    (h₃ : m = 3 * n) 
    (h₄ : n < 0) : 
    ∀ x : ℝ, (m + n) * x < n - m ↔ x > -1 / 2 :=
by
  sorry

end inequality_solution_set_l482_482135


namespace FBCDE_area_ratio_FBCDE_m_n_sum_l482_482205

noncomputable def FBCDE_triangle_ratio (FB BC CE BD : ℕ) (angleFBC : ℝ) : ℚ :=
  if h : (FB = 4 ∧ BC = 7 ∧ CE = 21 ∧ BD = real.sqrt 93 ∧ angleFBC = real.pi * (2/3)) then
    ⟨7, real.to_rat(93)⟩
  else
    0

theorem FBCDE_area_ratio (FB BC CE BD : ℕ) 
  (h1 : FB = 4) 
  (h2 : BC = 7) 
  (h3 : CE = 21) 
  (h4 : BD = real.sqrt 93) 
  (h5 : ∠(FB, BC) = 120) : 
  FBCDE_triangle_ratio FB BC CE BD 120 = 7 / real.sqrt 93 :=
by 
  sorry

theorem FBCDE_m_n_sum : m n : ℕ, (m = 7 ∧ n = 93) → m + n = 100 :=
by 
  sorry

end FBCDE_area_ratio_FBCDE_m_n_sum_l482_482205


namespace non_intersecting_chords_sum_l482_482216

theorem non_intersecting_chords_sum {n : ℕ} (h : 0 < n) (numbers : Finset ℕ) (h_numbers : numbers.card = 2 * n)
  (circle : Finset ℕ) (h_circle : circle = numbers) :
  ∃ (chords : Finset (ℕ × ℕ)), (chords.card = n) ∧ 
  (∀ (c1 c2 : ℕ × ℕ), c1 ∈ chords → c2 ∈ chords → c1 ≠ c2 → 
  ∀ (x1 y1 x2 y2: ℕ), c1 = (x1, y1) → c2 = (x2, y2) → 
  x1 ≠ x2 → y1 ≠ y2 → x1 ≠ y2 → x2 ≠ y1) ∧
  ∑ (c : ℕ × ℕ) in chords, abs (c.1 - c.2) = n^2 := by
  sorry

end non_intersecting_chords_sum_l482_482216


namespace eli_glazed_donuts_l482_482155

theorem eli_glazed_donuts (rEli rMia rCarlos : ℕ) (rateEli rateMia rateCarlos : ℕ)
  (hEliRadius : rEli = 5) (hMiaRadius : rMia = 9) (hCarlosRadius : rCarlos = 12)
  (hEliRate : rateEli = 2) (hMiaRate : rateMia = 1) (hCarlosRate : rateCarlos = 3) :
  let surfaceArea (r : ℕ) := 4 * real.pi * (r : ℕ) ^ 2
      glazeTime (area : real) (rate : ℕ) := area / rate
  in
  let tEli := glazeTime (surfaceArea rEli) rateEli
      tMia := glazeTime (surfaceArea rMia) rateMia
      tCarlos := glazeTime (surfaceArea rCarlos) rateCarlos
  in
  let lcm_main := nat.lcm (nat.lcm (50 : ℕ) 324) 192
  in
  (lcm_main * real.pi) / (50 * real.pi) = 115.2 :=
by
  let surfaceArea := λ r : ℕ => 4 * real.pi * r ^ 2
  let glazeTime := λ area rate : ℕ => area / rate
  let tEli := glazeTime (surfaceArea rEli) rateEli
  let tMia := glazeTime (surfaceArea rMia) rateMia
  let tCarlos := glazeTime (surfaceArea rCarlos) rateCarlos
  let lcm_main := nat.lcm (nat.lcm (50 : ℕ) 324) 192
  have h := (lcm_main * real.pi) / (50 * real.pi) 
  exact sorry

end eli_glazed_donuts_l482_482155


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l482_482473

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4 :
  let n := 7^7 - 7^4 in 
  (∑ p in (nat.factors n).to_finset, p) = 31 :=
by sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l482_482473


namespace length_of_train_l482_482376

theorem length_of_train (speed_km_per_hr : ℝ) (bridge_length : ℝ) (crossing_time_seconds : ℝ) :
  speed_km_per_hr = 45 → bridge_length = 215 → crossing_time_seconds = 30 → 
  (let speed_m_per_s := speed_km_per_hr * 1000 / 3600 in
  let total_distance := speed_m_per_s * crossing_time_seconds in
  let train_length := total_distance - bridge_length in
  train_length = 160) :=
by 
  intros h_speed h_bridge h_time
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600
  have h_speed_m_per_s : speed_m_per_s = 12.5 := by sorry
  let total_distance := speed_m_per_s * crossing_time_seconds
  have h_total_distance : total_distance = 375 := by sorry
  let train_length := total_distance - bridge_length
  have h_train_length : train_length = 160 := by sorry
  exact h_train_length

end length_of_train_l482_482376


namespace parallel_vectors_solution_l482_482110

theorem parallel_vectors_solution 
  (x : ℝ) 
  (a : ℝ × ℝ := (-1, 3)) 
  (b : ℝ × ℝ := (x, 1)) 
  (h : ∃ k : ℝ, a = k • b) :
  x = -1 / 3 :=
by
  sorry

end parallel_vectors_solution_l482_482110


namespace ratio_of_fruit_salads_l482_482382

theorem ratio_of_fruit_salads 
  (salads_Alaya : ℕ) 
  (total_salads : ℕ) 
  (h1 : salads_Alaya = 200) 
  (h2 : total_salads = 600) : 
  (total_salads - salads_Alaya) / salads_Alaya = 2 :=
by 
  sorry

end ratio_of_fruit_salads_l482_482382


namespace red_flower_ratio_l482_482803

theorem red_flower_ratio
  (total : ℕ)
  (O : ℕ)
  (P Pu : ℕ)
  (R Y : ℕ)
  (h_total : total = 105)
  (h_orange : O = 10)
  (h_pink_purple : P + Pu = 30)
  (h_equal_pink_purple : P = Pu)
  (h_yellow : Y = R - 5)
  (h_sum : R + Y + O + P + Pu = total) :
  (R / O) = 7 / 2 :=
by
  sorry

end red_flower_ratio_l482_482803


namespace proof_problem_l482_482533

noncomputable def f (a b : ℝ) (x : ℝ) :=
  (a - 3 * b + 9) * Real.log (x + 3) + 0.5 * x^2 + (b - 3) * x

def f' (a b : ℝ) (x : ℝ) :=
  (x^2 + b * x + a) / (x + 3)

theorem proof_problem 
  (a b : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : f' a b 1 = 0)
  (h₃ : f' a b 3 ≤ 1 / 6)
  (h₄ : ∀ x : ℝ, x > -3 → abs x ≥ 2 → f' a b x ≥ 0) :
  (b = -a - 1 ∧ 0 < a ∧ a < 1 → 
    (∀ x, -3 < x ∧ x < a ∨ 1 < x → f' a b x > 0) ∧ 
    (∀ x, a < x ∧ x < 1 → f' a b x < 0)) ∧ 
  (1 < a → 
    (∀ x, -3 < x ∧ x < 1 ∨ a < x → f' a b x > 0) ∧ 
    (∀ x, 1 < x ∧ x < a → f' a b x < 0)) ∧
  (b = -4 ∧ a = 4 → 
    f a b = λ x, 25 * Real.log (x + 3) + 0.5 * x^2 - 7 * x ∧
    (∀ x, -3 < x ∧ x < 2 → (∃ y, f a b y = 16 ∧ y = -2))) := 
sorry

end proof_problem_l482_482533


namespace percent_area_square_in_rectangle_l482_482811

theorem percent_area_square_in_rectangle
  (s : ℝ) 
  (w : ℝ) 
  (l : ℝ)
  (h1 : w = 3 * s) 
  (h2 : l = (9 / 2) * s) 
  : (s^2 / (l * w)) * 100 = 7.41 :=
by
  sorry

end percent_area_square_in_rectangle_l482_482811


namespace total_pies_sold_l482_482857

theorem total_pies_sold :
  let shepherd_slices := 52
  let chicken_slices := 80
  let shepherd_pieces_per_pie := 4
  let chicken_pieces_per_pie := 5
  let shepherd_pies := shepherd_slices / shepherd_pieces_per_pie
  let chicken_pies := chicken_slices / chicken_pieces_per_pie
  shepherd_pies + chicken_pies = 29 :=
by
  sorry

end total_pies_sold_l482_482857


namespace remainder_142_to_14_l482_482809

theorem remainder_142_to_14 (N k : ℤ) 
  (h : N = 142 * k + 110) : N % 14 = 8 :=
sorry

end remainder_142_to_14_l482_482809


namespace congruence_problem_l482_482567

theorem congruence_problem (x : ℤ) (h : 5 * x + 9 ≡ 4 [ZMOD 18]) : 3 * x + 15 ≡ 12 [ZMOD 18] :=
sorry

end congruence_problem_l482_482567


namespace basketball_travel_distance_l482_482797

noncomputable def total_distance (initial_height : ℝ) (rebound_factor : ℝ) (bounces : ℕ) : ℝ :=
  let descent_distances := (List.range (bounces + 1)).map (λ n, initial_height * (rebound_factor ^ n))
  let ascent_distances := (List.range bounces).map (λ n, initial_height * (rebound_factor ^ (n + 1)))
  descent_distances.sum + ascent_distances.sum

theorem basketball_travel_distance :
  total_distance 80 0.75 5 = 408.125 :=
by
  sorry

end basketball_travel_distance_l482_482797


namespace tetrahedron_angle_l482_482997

noncomputable def angle_between_edge_and_opposite_face (a : ℝ) : ℝ := 
  real.arccos (1 / real.sqrt 3)

theorem tetrahedron_angle (a : ℝ) (h : a > 0) : 
  ∃ α : ℝ, α = angle_between_edge_and_opposite_face a :=
begin
  use real.arccos (1 / real.sqrt 3),
  exact rfl,
end

end tetrahedron_angle_l482_482997


namespace find_geometric_progression_l482_482288

theorem find_geometric_progression (a b c : ℚ)
  (h1 : a * c = b * b)
  (h2 : a + c = 2 * (b + 8))
  (h3 : a * (c + 64) = (b + 8) * (b + 8)) :
  (a = 4/9 ∧ b = -20/9 ∧ c = 100/9) ∨ (a = 4 ∧ b = 12 ∧ c = 36) :=
sorry

end find_geometric_progression_l482_482288


namespace sqrt_sum_simplify_l482_482687

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l482_482687


namespace percentage_of_boy_scouts_with_signed_permission_slips_l482_482806

noncomputable def total_scouts : ℕ := 100 -- assume 100 scouts
noncomputable def total_signed_permission_slips : ℕ := 70 -- 70% of 100
noncomputable def boy_scouts : ℕ := 60 -- 60% of 100
noncomputable def girl_scouts : ℕ := 40 -- total_scouts - boy_scouts 

noncomputable def girl_scouts_signed_permission_slips : ℕ := girl_scouts * 625 / 1000 

theorem percentage_of_boy_scouts_with_signed_permission_slips :
  (boy_scouts * 75 / 100) = (total_signed_permission_slips - girl_scouts_signed_permission_slips) :=
by
  sorry

end percentage_of_boy_scouts_with_signed_permission_slips_l482_482806


namespace cannot_rearrange_rooks_l482_482788

section
variables (rook A1 B1 C1 A' B' C' : Type)

-- Definition of protecting rook.
def protects (r1 r2 : rook) : Prop := sorry

-- Initial positions.
def initial_positions (r : rook) : Prop :=
  r = A1 ∨ r = B1 ∨ r = C1

-- Final positions.
def final_positions (r : rook) : Prop :=
  r = A' ∨ r = B' ∨ r = C'

-- Proposition: It's impossible to rearrange rooks subject to the given conditions.
theorem cannot_rearrange_rooks :
  (∀ r1 r2 : rook, protects r1 r2 → r1 ≠ r2) →
  (∀ r : rook, initial_positions r) →
  (∀ r : rook, ¬ final_positions r) →
  false :=
sorry
end

end cannot_rearrange_rooks_l482_482788


namespace four_points_form_convex_quadrilateral_l482_482063

theorem four_points_form_convex_quadrilateral 
(points : Set (ℝ × ℝ))
(h_card : points.size = 5)
(h_non_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), {p1, p2, p3} ⊆ points → ¬ collinear ℝ p1.1 p2.1 p3.1)
: ∃ (s : Set (ℝ × ℝ)), s ⊆ points ∧ s.size = 4 ∧ convex_hull ℝ s = s :=
by
  sorry

end four_points_form_convex_quadrilateral_l482_482063


namespace S_sum_l482_482060

-- Define the sequence S_n
def S (n : ℕ) : ℤ :=
  (List.range (n + 1)).sumBy (fun i => (-1 : ℤ) ^ (i + 1) * (i : ℤ))

-- Prove S_100 + S_200 + S_301 = 51
theorem S_sum : S 100 + S 200 + S 301 = 51 :=
  sorry

end S_sum_l482_482060


namespace ten_pow_condition_l482_482905

theorem ten_pow_condition {a b : ℝ} (h1 : 10^a = 2) (h2 : 10^b = 6) : 
  10^(2*a - 3*b) = 1 / 54 := 
  by 
    sorry

end ten_pow_condition_l482_482905


namespace arrangement_count_l482_482437

theorem arrangement_count :
  let students := {A, B, C, D}
  let attractions := {T1, T2, T3}
  ∃ (arrangement : students → attractions),
    (∀ t ∈ attractions, ∃ s ∈ students, arrangement s = t) ∧
    arrangement A ≠ arrangement B ∧
    (finset.card (finset.univ.image arrangement) = 3) :=
sorry

end arrangement_count_l482_482437


namespace abs_diff_roots_quad_eq_one_l482_482445

theorem abs_diff_roots_quad_eq_one : 
  ∀ r1 r2 : ℝ, (r1 + r2 = 7) ∧ (r1 * r2 = 12) → |r1 - r2| = 1 :=
by
  intro r1 r2 h
  have h_sum := h.1
  have h_prod := h.2
  sorry

end abs_diff_roots_quad_eq_one_l482_482445


namespace find_quadratic_polynomial_l482_482045

def quadratic_polynomial (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem find_quadratic_polynomial : 
  ∃ a b c: ℝ, (∀ x : ℂ, quadratic_polynomial a b c x.re = 0 → (x = 3 + 4*I) ∨ (x = 3 - 4*I)) 
  ∧ (b = 8) 
  ∧ (a = -4/3) 
  ∧ (c = -50/3) :=
by
  sorry

end find_quadratic_polynomial_l482_482045


namespace integer_sequence_l482_482428

/-- Define the sequence {a_n} such that
  a₁ = 1,
  a₂ = 2,
  a₃ = 3,
  and for any n ≥ 3,
  aₙ₊₁ = aₙ - aₙ₋₁ + aₙ^2 / aₙ₋₂.
  Prove that a_n is an integer sequence. -/
theorem integer_sequence (a : ℕ → ℚ) 
  (h₁ : a 1 = 1)
  (h₂ : a 2 = 2)
  (h₃ : a 3 = 3)
  (h_step : ∀ n ≥ 3, a (n + 1) = a n - a (n - 1) + a n ^ 2 / a (n - 2)) :
  ∀ n, a n ∈ ℤ := 
sorry

end integer_sequence_l482_482428


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482458

def seven_pow_seven_minus_seven_pow_four : ℤ := 7^7 - 7^4
def prime_factors_of_three_hundred_forty_two : List ℤ := [2, 3, 19]

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let distinct_prime_factors := prime_factors_of_three_hundred_forty_two.head!
  + prime_factors_of_three_hundred_forty_two.tail!.head!
  + prime_factors_of_three_hundred_forty_two.tail!.tail!.head!
  seven_pow_seven_minus_seven_pow_four = 7^4 * (7^3 - 1) ∧
  7^3 - 1 = 342 ∧
  prime_factors_of_three_hundred_forty_two = [2, 3, 19] ∧
  distinct_prime_factors = 24 := 
sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l482_482458


namespace black_stones_count_l482_482144

theorem black_stones_count (T W B : ℕ) (hT : T = 48) (hW1 : 4 * W = 37 * 2 + 26) (hB : B = T - W) : B = 23 :=
by
  sorry

end black_stones_count_l482_482144


namespace score_difference_l482_482496

theorem score_difference (chuck_score red_score : ℕ) (h1 : chuck_score = 95) (h2 : red_score = 76) : chuck_score - red_score = 19 := by
  sorry

end score_difference_l482_482496


namespace total_pencils_crayons_l482_482880

theorem total_pencils_crayons (r : ℕ) (p : ℕ) (c : ℕ) 
  (hp : p = 31) (hc : c = 27) (hr : r = 11) : 
  r * p + r * c = 638 := 
  by
  sorry

end total_pencils_crayons_l482_482880


namespace simplify_sum_of_square_roots_l482_482682

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l482_482682


namespace time_first_platform_15s_l482_482373

variables (L_t L_p2 L_p1 time2 speed t1 : ℕ)

def length_of_train := L_t = 270
def length_of_second_platform := L_p2 = 250
def time_to_cross_second_platform := time2 = 20
def length_of_first_platform := L_p1 = 120

def speed := speed = (L_t + L_p2) / time2
def time_to_cross_first_platform := t1 = (L_t + L_p1) / speed

theorem time_first_platform_15s :
  length_of_train ∧ length_of_second_platform ∧ time_to_cross_second_platform ∧ length_of_first_platform →
  t1 = 15 :=
by
  intros h
  sorry

end time_first_platform_15s_l482_482373


namespace length_of_train_l482_482375

theorem length_of_train (speed_km_per_hr : ℝ) (bridge_length : ℝ) (crossing_time_seconds : ℝ) :
  speed_km_per_hr = 45 → bridge_length = 215 → crossing_time_seconds = 30 → 
  (let speed_m_per_s := speed_km_per_hr * 1000 / 3600 in
  let total_distance := speed_m_per_s * crossing_time_seconds in
  let train_length := total_distance - bridge_length in
  train_length = 160) :=
by 
  intros h_speed h_bridge h_time
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600
  have h_speed_m_per_s : speed_m_per_s = 12.5 := by sorry
  let total_distance := speed_m_per_s * crossing_time_seconds
  have h_total_distance : total_distance = 375 := by sorry
  let train_length := total_distance - bridge_length
  have h_train_length : train_length = 160 := by sorry
  exact h_train_length

end length_of_train_l482_482375


namespace find_line_equation_l482_482736

-- Definitions for the conditions
structure Point where
  x : ℝ
  y : ℝ

def origin : Point := Point.mk 0 0
def proj_origin_on_line : Point := Point.mk (-2) 1

-- Definition of the line in slope-intercept form
def line_equation (x y : ℝ) : Prop := 2*x - y + 5 = 0

-- Proof statement: Given the projection of the origin onto line l is P(-2, 1), the equation of the line is 2x - y + 5 = 0.
theorem find_line_equation :
  let P := proj_origin_on_line in
  line_equation P.x P.y :=
by
  let P := proj_origin_on_line
  show line_equation P.x P.y
  sorry

end find_line_equation_l482_482736


namespace train_length_is_correct_l482_482378

-- Defining necessary variables and constants
def train_speed_kmph := 45 -- km/hr
def bridge_length := 215 -- meters
def crossing_time := 30 -- seconds

noncomputable def train_speed_mps := (train_speed_kmph * 1000) / 3600 -- m/s

def distance_travelled := train_speed_mps * crossing_time

-- Theorem stating that the length of the train equals 160 meters
theorem train_length_is_correct : 
  ∃ l_t : ℝ, l_t = distance_travelled - bridge_length ∧ l_t = 160 :=
by { 
  unfold train_speed_mps,
  unfold distance_travelled,
  unfold train_speed_kmph,
  unfold bridge_length,
  unfold crossing_time,
  sorry
}

end train_length_is_correct_l482_482378


namespace largest_number_among_ten_l482_482743

-- Define digit sum function
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else digit_sum (n / 10) + (n % 10)

-- Define the condition that the sum of a list of numbers
def list_sum (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

-- Lean 4 statement for the given problem
theorem largest_number_among_ten :
  ∃ (numbers : list ℕ), 
    numbers.length = 10 ∧ 
    list_sum numbers = 604 ∧ 
    (∀ n ∈ numbers, digit_sum n = digit_sum 109) ∧ 
    list.all numbers (λ n, 604 > n) ∧ 
    (∀ n ∈ numbers, n ≠ 109) → 
    109 ∈ numbers :=
sorry

end largest_number_among_ten_l482_482743


namespace seq_sum_difference_l482_482849

-- Define the sequences
def seq1 : List ℕ := List.range 93 |> List.map (λ n => 2001 + n)
def seq2 : List ℕ := List.range 93 |> List.map (λ n => 301 + n)

-- Define the sum of the sequences
def sum_seq1 : ℕ := seq1.sum
def sum_seq2 : ℕ := seq2.sum

-- Define the difference between the sums of the sequences
def diff_seq_sum : ℕ := sum_seq1 - sum_seq2

-- Lean statement to prove the difference equals 158100
theorem seq_sum_difference : diff_seq_sum = 158100 := by
  sorry

end seq_sum_difference_l482_482849


namespace anya_probability_l482_482397

open Finset

def possible_coins := {10, 10, 5, 5, 2}
def target := 19

noncomputable def combinations := (possible_coins.vals.ctype_power 3).val.filter (λ s, Finset.sum s >= target)

noncomputable def probability : ℝ :=
  (combinations.card : ℝ) / (possible_coins.vals.ctype_power 3).card

theorem anya_probability : probability = 0.4 := sorry

end anya_probability_l482_482397


namespace knocks_to_knicks_l482_482573
open_locale classical

noncomputable def knicks : Type := ℕ
noncomputable def knacks : Type := ℕ
noncomputable def knocks : Type := ℕ

axiom knicks_to_knacks : ∃ (k : ℕ) (n : ℕ), 5 * k = 3 * n
axiom knacks_to_knocks : ∃ (n : ℕ) (c : ℕ), 2 * n = 5 * c

theorem knocks_to_knicks : ∀ (c : ℕ), (30 * c = 20 * k) :=
by {
  sorry,
}

end knocks_to_knicks_l482_482573


namespace correct_inferences_l482_482100

-- Define the function f(x)
def f (x : ℝ) : ℝ := x + Real.cos x

-- Statement of the problem
theorem correct_inferences (x : ℝ) :
  -- Proposition 1: The domain of f(x) is \mathbb{R}
  (∀ x, (x ∈ ℝ)) →
  -- Proposition 2: The range of f(x) is \mathbb{R}
  (∃ y : ℝ, ∀ x : ℝ, f(x) = y) →
  -- Proposition 3: f(x) is an odd function
  (f(-x) = -f(x)) →
  -- Proposition 4: The graph of f(x) intersects the line y = x at x = k\pi + \frac{\pi}{2}
  (∃ k : ℤ, f(k * Real.pi + Real.pi / 2) = k * Real.pi + Real.pi / 2) →
  -- The total number of correct propositions
  2 = 2 :=
sorry -- Proof to be completed.

end correct_inferences_l482_482100


namespace max_sum_composite_2013_l482_482732

def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem max_sum_composite_2013 :
  ∃ m : ℕ, (∀ k : ℕ, 2013 = (∑ _ in (finset.range m).erase 0, if k = 1 then 9 else 4) → m = 502)
  sorry

end max_sum_composite_2013_l482_482732


namespace b_2016_value_l482_482551

noncomputable def a : ℕ → ℚ
| 1 := 1/2
| n := 1 - b n

noncomputable def b : ℕ → ℚ
| 1 := 1/2
| (n+1) := b n / (1 - (a n)^2)

theorem b_2016_value : b 2016 = 2016 / 2017 :=
by sorry

end b_2016_value_l482_482551


namespace intersect_lines_l482_482636

open EuclideanGeometry

-- Definitions:
variables {A B C L P O M : Point}

-- Conditions:
def angle_bisector (BL : Line) (A B C : Point) : Prop :=
  bisects_angle BL A B C

def circumcenter (O : Point) (L P B : Triangle) : Prop :=
  is_circumcenter O L P B

def perpendicular_angle (B P C : Point) : Prop :=
  ∠ B P C = 90

def sum_of_angles (L P C B LBC : Point) : Prop :=
  ∠ L P C + ∠ L B C = 180

def midpoint (M B C : Point) : Prop :=
  is_midpoint M B C

-- Theorem to prove:
theorem intersect_lines
  (h1 : angle_bisector (line_through B L) A B C)
  (h2 : circumcenter O (triangle L P B))
  (h3 : perpendicular_angle B P C)
  (h4 : sum_of_angles L P C B LBC)
  (h5 : midpoint M B C) :
  exists P, collinear {P, line_through C O, line_through B L, line_through A M} :=
sorry

end intersect_lines_l482_482636


namespace minimum_of_f_on_interval_l482_482522

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 3*x + 2 else -(x^2 - 3*x + 2)

theorem minimum_of_f_on_interval :
  is_odd_function f →
  (∀ x, x < 0 → f x = x^2 + 3*x + 2) →
  ∃ y ∈ set.Icc 1 3, ∀ z ∈ set.Icc 1 3, f y ≤ f z ∧ f y = -2 :=
by
  sorry

end minimum_of_f_on_interval_l482_482522


namespace possible_values_of_angle_F_l482_482167

-- Define angle F conditions in a triangle DEF
def triangle_angle_F_conditions (D E : ℝ) : Prop :=
  5 * Real.sin D + 2 * Real.cos E = 8 ∧ 3 * Real.sin E + 5 * Real.cos D = 2

-- The main statement: proving the possible values of ∠F
theorem possible_values_of_angle_F (D E : ℝ) (h : triangle_angle_F_conditions D E) : 
  ∃ F : ℝ, F = Real.arcsin (43 / 50) ∨ F = 180 - Real.arcsin (43 / 50) :=
by
  sorry

end possible_values_of_angle_F_l482_482167


namespace round_trip_time_l482_482231

theorem round_trip_time 
  (d1 d2 d3 : ℝ) 
  (s1 s2 s3 t : ℝ) 
  (h1 : d1 = 18) 
  (h2 : d2 = 18) 
  (h3 : d3 = 36) 
  (h4 : s1 = 12) 
  (h5 : s2 = 10) 
  (h6 : s3 = 9) 
  (h7 : t = (d1 / s1) + (d2 / s2) + (d3 / s3)) :
  t = 7.3 :=
by
  sorry

end round_trip_time_l482_482231


namespace angle_ACB_right_angle_l482_482793

open Real
open Set

-- Let's define the points and properties required
variable (A B C O : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
variable (circle : Metric.Circle B ℝ)
variable (diameter : B × B) -- to denote AB being the diameter

-- Point C is on the circle
variable (C_on_circle : C ∈ circle)

-- O is the midpoint of diameter AB and also the center of circle
variable (O_center : Center diameter O)
variable (AB_diameter : ∀ (x ∈ circle),  diameter.1 = diameter.2)

theorem angle_ACB_right_angle :
  ∀ (A B C : Point circle) (O : Point diameter),
    (diameter.1 = A) → (diameter.2 = B) → 
    (O_center diameter O) ∧ (C_on_circle C) → (AB_diameter A B) → 
    ∠ A C B = 90 :=
by sorry

end angle_ACB_right_angle_l482_482793


namespace probability_largest_number_l482_482780

theorem probability_largest_number (n : ℕ) : 
  let total_numbers := (n * (n + 1)) / 2 in
  let p : ℕ → ℚ := 
    λ n, if n = 0 then 1 else (2 / (n + 1) : ℚ) * p (n - 1) in
  p n = (2^n : ℚ) / (n + 1)! := 
sorry

end probability_largest_number_l482_482780


namespace transform_graphs_l482_482954

noncomputable def f : ℝ → ℝ := sorry

def passes_through (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop := f p.1 = p.2

theorem transform_graphs (h₁ : passes_through (λ x, f (x + 1)) (3, 2)) :
  passes_through (λ x, -f x) (4, -2) :=
by {
  sorry
}

end transform_graphs_l482_482954


namespace find_k_l482_482984

theorem find_k (x y k : ℝ) 
  (h1 : 4 * x + 2 * y = 5 * k - 4) 
  (h2 : 2 * x + 4 * y = -1) 
  (h3 : x - y = 1) : 
  k = 1 := 
by sorry

end find_k_l482_482984


namespace football_team_total_players_l482_482286

variable (P : ℕ)
variable (throwers : ℕ := 52)
variable (total_right_handed : ℕ := 64)
variable (remaining := P - throwers)
variable (left_handed := remaining / 3)
variable (right_handed_non_throwers := 2 * remaining / 3)

theorem football_team_total_players:
  right_handed_non_throwers + throwers = total_right_handed →
  P = 70 :=
by
  sorry

end football_team_total_players_l482_482286


namespace three_digit_powers_of_two_l482_482559

theorem three_digit_powers_of_two : 
  ∃ (N : ℕ), N = 3 ∧ ∀ (n : ℕ), (100 ≤ 2^n ∧ 2^n < 1000) ↔ (n = 7 ∨ n = 8 ∨ n = 9) :=
by
  sorry

end three_digit_powers_of_two_l482_482559


namespace tickets_sold_l482_482748

theorem tickets_sold (S G : ℕ) (hG : G = 388) (h_total : 4 * S + 6 * G = 2876) :
  S + G = 525 := by
  sorry

end tickets_sold_l482_482748


namespace no_bird_gathering_six_trees_bird_gathering_seven_trees_l482_482654

-- (a) Prove that 6 birds cannot gather on one tree given the conditions
theorem no_bird_gathering_six_trees :
  ∀ (f : Fin 6 → ℕ), (∀ i j m (h1 : f i = f j + m) (h2 : f j = f i - m), true) →
  ¬ (∃ x, ∑ i, f i = 6 * x) :=
by
  sorry

-- (b) Prove that 7 birds can gather on one tree given the conditions
theorem bird_gathering_seven_trees :
  ∀ (f : Fin 7 → ℕ), (∀ i j m (h1 : f i = f j + m) (h2 : f j = f i - m), true) →
  ∃ y, ∑ i, f i = 7 * y :=
by
  use 4
  sorry

end no_bird_gathering_six_trees_bird_gathering_seven_trees_l482_482654


namespace seventh_observation_l482_482781

-- Declare the conditions with their definitions
def average_of_six (sum6 : ℕ) : Prop := sum6 = 6 * 14
def new_average_decreased (sum6 sum7 : ℕ) : Prop := sum7 = sum6 + 7 ∧ 13 = (sum6 + 7) / 7

-- The main statement to prove that the seventh observation is 7
theorem seventh_observation (sum6 sum7 : ℕ) (h_avg6 : average_of_six sum6) (h_new_avg : new_average_decreased sum6 sum7) :
  sum7 - sum6 = 7 := 
  sorry

end seventh_observation_l482_482781


namespace total_chickens_l482_482660

open Nat

theorem total_chickens 
  (Q S C : ℕ) 
  (h1 : Q = 2 * S + 25) 
  (h2 : S = 3 * C - 4) 
  (h3 : C = 37) : 
  Q + S + C = 383 := by
  sorry

end total_chickens_l482_482660


namespace coterminal_angle_neg_60_eq_300_l482_482777

theorem coterminal_angle_neg_60_eq_300 :
  ∃ k : ℤ, 0 ≤ k * 360 - 60 ∧ k * 360 - 60 < 360 ∧ (k * 360 - 60 = 300) := by
  sorry

end coterminal_angle_neg_60_eq_300_l482_482777


namespace domain_f_l482_482431

noncomputable def f (x : ℝ) : ℝ := -2 / (Real.sqrt (x + 5)) + Real.log (2^x + 1)

theorem domain_f :
  {x : ℝ | (-5 ≤ x)} = {x : ℝ | f x ∈ Set.univ} := sorry

end domain_f_l482_482431


namespace total_savings_l482_482189

theorem total_savings :
  let J := 0.25 in
  let D_J := 24 in
  let L := 0.50 in
  let D_L := 20 in
  let M := 2 * L in
  let D_M := 12 in
  J * D_J + L * D_L + M * D_M = 28.00 :=
by 
  sorry

end total_savings_l482_482189


namespace find_n_l482_482890

theorem find_n : ∃ n : ℤ, 100 ≤ n ∧ n ≤ 280 ∧ Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180) ∧ n = 317 := 
by
  sorry

end find_n_l482_482890


namespace g_of_neg_2_l482_482632

def f (x : ℚ) : ℚ := 4 * x - 9

def g (y : ℚ) : ℚ :=
  3 * ((y + 9) / 4)^2 - 4 * ((y + 9) / 4) + 2

theorem g_of_neg_2 : g (-2) = 67 / 16 :=
by
  sorry

end g_of_neg_2_l482_482632


namespace inequality_solution_set_l482_482134

theorem inequality_solution_set (m n : ℝ) 
    (h₁ : ∀ x : ℝ, mx - n > 0 ↔ x < 1 / 3) 
    (h₂ : m + n < 0) 
    (h₃ : m = 3 * n) 
    (h₄ : n < 0) : 
    ∀ x : ℝ, (m + n) * x < n - m ↔ x > -1 / 2 :=
by
  sorry

end inequality_solution_set_l482_482134


namespace problem_conditions_l482_482947

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.tan x

def domain (x : ℝ) : Prop := ∀ k : ℤ, x ≠ (Int.cast k) * Real.pi + Real.pi / 2

theorem problem_conditions (x : ℝ) :
  domain x →
  (f(-x) = f(x)) ∧
  (¬(∀ (a b : ℝ), -Real.pi / 2 < a ∧ a < b ∧ b < 0 → a < b → f(a) < f(b))) ∧
  ((∀ y, f(x + 2 * Real.pi) = f(x))) ∧
  ((f(Real.pi - x) = -f(x)) ∧ (f(Real.pi + x) = -f(x)) → f(Real.pi - x) = f(Real.pi + x)) :=
by
  assume h : domain x
  split
  -- Various apologies for proofs
  sorry
  sorry
  sorry
  sorry

end problem_conditions_l482_482947


namespace total_cost_of_phone_l482_482349

theorem total_cost_of_phone (cost_per_phone : ℕ) (monthly_cost : ℕ) (months : ℕ) (phone_count : ℕ) :
  cost_per_phone = 2 → monthly_cost = 7 → months = 4 → phone_count = 1 →
  (cost_per_phone * phone_count + monthly_cost * months) = 30 :=
by
  intros h1 h2 h3 h4
  sorry

end total_cost_of_phone_l482_482349


namespace mod_inv_3_197_l482_482034

theorem mod_inv_3_197 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 196 ∧ (3 * x ≡ 1 [MOD 197]) ∧ x = 66 := by
  sorry

end mod_inv_3_197_l482_482034


namespace sqrt_sum_simplify_l482_482672

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l482_482672


namespace probability_point_between_l_and_m_l482_482142

def l (x : ℝ) : ℝ := -2 * x + 8
def m (x : ℝ) : ℝ := -3 * x + 9

def area_under_l : ℝ := 0.5 * 4 * 8
def area_under_m : ℝ := 0.5 * 3 * 9

theorem probability_point_between_l_and_m : 
  (area_under_l - area_under_m) / area_under_l = 0.16 :=
by
  -- Variables to store areas for clarity
  have area_l : ℝ := 0.5 * 4 * 8
  have area_m : ℝ := 0.5 * 3 * 9

  -- Probability calculation
  calc (area_l - area_m) / area_l = 2.5 / 16 : by sorry
  ... = 0.15625 : by sorry
  ... ≈ 0.16 : by sorry

end probability_point_between_l_and_m_l482_482142


namespace num_ways_sum_1620_l482_482561

theorem num_ways_sum_1620 (n : ℕ) (h : 1620 = 2 * m + 3 * n) : 
  { m | 2 * m + 3 * n = 1620 }.card = 271 :=
sorry

end num_ways_sum_1620_l482_482561


namespace F_is_midpoint_CD_l482_482403

variable {P : Type} [MetricSpace P]
variables {A B C D E F : P}
variables (circumcircle : Set P) (tangent_at_A tangent_at_C : Set P)

-- Conditions
variable (h_AC_BC : dist AC = dist BC)
variable (h_tangents_intersect_at_D : tangent_at_A ∩ tangent_at_C = {D})
variable (h_BD_intersects_circumcircle_at_E : E ∈ circumcircle ∧ collinear B D E)
variable (h_AE_intersects_CD_at_F : collinear A E F ∧ collinear C D F)

-- Goal
theorem F_is_midpoint_CD :
  midpoint C D F :=
sorry

end F_is_midpoint_CD_l482_482403


namespace circle_center_and_tangent_line_l482_482935

noncomputable def center_of_circle (x y : ℝ) := (x - 1)^2 + y^2 = 1

theorem circle_center_and_tangent_line :
  (∀ x y : ℝ, x^2 - 2 * x + y^2 = 0 → center_of_circle x y) ∧ 
  (∀ k : ℝ, k = sqrt(3) / 3 ∨ k = - (sqrt(3) / 3) →
          ∀ x y : ℝ, (-1, 0) ∈ line_through (1, 0) k → 
          ∃ x y : ℝ, line_tangent_to_circle (1, 0) k) :=
sorry

end circle_center_and_tangent_line_l482_482935


namespace condition_sufficient_not_necessary_l482_482621

-- Define the triangle ABC
variables {A B C : Point ℝ}
variables {t : ℝ} (ht : t ≠ 1)

-- Define the vectors BA, BC, and AC
def BA := vector (B - A)
def BC := vector (B - C)
def AC := vector (A - C)

-- Condition
axiom h : ∀ t ≠ 1, ∥BA - t • BC∥ > ∥AC∥

-- Theorem: The given condition is sufficient but not necessary for a right-angled triangle at C
theorem condition_sufficient_not_necessary : 
  (∀ t ≠ 1, ∥BA - t • BC∥ > ∥AC∥) → ∃ (R : ℝ), right_angle_triangle R A B C :=
  sorry

end condition_sufficient_not_necessary_l482_482621


namespace protective_additive_increase_l482_482324

def percentIncrease (old_val new_val : ℕ) : ℚ :=
  (new_val - old_val) / old_val * 100

theorem protective_additive_increase :
  percentIncrease 45 60 = 33.33 := 
sorry

end protective_additive_increase_l482_482324


namespace number_of_people_in_group_l482_482261

/-- The number of people in the group N is such that when one of the people weighing 65 kg is replaced
by a new person weighing 100 kg, the average weight of the group increases by 3.5 kg. -/
theorem number_of_people_in_group (N : ℕ) (W : ℝ) 
  (h1 : (W + 35) / N = W / N + 3.5) 
  (h2 : W + 35 = W - 65 + 100) : 
  N = 10 :=
sorry

end number_of_people_in_group_l482_482261


namespace proof_problem_l482_482066

-- Define the given problem
def problem_statement (n : ℕ) (h1 : 3 < n) : Prop :=
  ∀ (k : ℕ) (x : Fin k → ℕ),
    (1 ≤ k) →
    (∑ i in Finset.range k, x i) ≤ n →
    Nat.lcm_list (Multiset.map (λ (j : Fin k), Multiset.prod (Multiset.map (λ (i : Fin 1), x i) (Multiset.range k))) (Multiset.range k)).to_list < n!

theorem proof_problem (n : ℕ) (h1 : 3 < n) : problem_statement n h1 :=
sorry

end proof_problem_l482_482066


namespace total_pairs_of_corresponding_interior_angles_is_16_l482_482585

-- Definitions related to the problem conditions
def are_parallel (l1 l2 : Prop) : Prop := sorry
def is_intersecting (l1 l2 : Prop) : Prop := sorry
def is_consecutive_interior_angle (a1 a2 : Prop) : Prop := sorry

-- Given conditions
variable (EF MN AB CD : Prop)
axiom EF_parallel_MN : are_parallel EF MN
axiom AB_intersects_EF : is_intersecting AB EF
axiom AB_intersects_MN : is_intersecting AB MN
axiom CD_intersects_EF : is_intersecting CD EF
axiom CD_intersects_MN : is_intersecting CD MN

-- The statement we're proving
theorem total_pairs_of_corresponding_interior_angles_is_16 :
  ∃ pairs: Nat, pairs = 16 ∧
    ((are_parallel EF MN) ∧ 
    (is_intersecting AB EF) ∧
    (is_intersecting AB MN) ∧
    (is_intersecting CD EF) ∧
    (is_intersecting CD MN)) :=
begin
  use 16,
  split,
  { refl },
  { split; assumption }
end

end total_pairs_of_corresponding_interior_angles_is_16_l482_482585


namespace sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l482_482480

theorem sum_distinct_prime_factors_of_7_to_7_minus_7_to_4 : 
  let pfs := primeFactors (7 ^ 7 - 7 ^ 4)
  in (pfs = {2, 3, 19}) → sum pfs = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l482_482480


namespace surface_area_of_solid_l482_482108

-- Defining the conditions described in the problem
def radius : ℝ := 1
def slant_height : ℝ := 3

-- Defining the surface area of the cone’s slant surface
def cone_lateral_surface_area (r l : ℝ) : ℝ :=
  π * r * l

-- Stating the proof problem: Prove that the surface area of the object is 3π
theorem surface_area_of_solid ::
  cone_lateral_surface_area radius slant_height = 3 * π :=
by
  sorry

end surface_area_of_solid_l482_482108


namespace uniform_b_interval_l482_482207

noncomputable def b (b1 : ℝ) := 3 * (b1 - 2)

theorem uniform_b_interval (b1 : ℝ) (h : 0 ≤ b1 ∧ b1 ≤ 1) :
  ∃ (a c : ℝ), a ≤ b1 ∧ b1 ≤ c ∧ b ∈ set.Icc (-6) (-3) :=
sorry

end uniform_b_interval_l482_482207


namespace smallest_sector_angle_24_l482_482625

theorem smallest_sector_angle_24
  (a : ℕ) (d : ℕ)
  (h1 : ∀ i, i < 8 → ((a + i * d) : ℤ) > 0)
  (h2 : (2 * a + 7 * d = 90)) : a = 24 :=
by
  sorry

end smallest_sector_angle_24_l482_482625


namespace number_of_members_l482_482225

theorem number_of_members
  (headband_cost : ℕ := 3)
  (jersey_cost : ℕ := 10)
  (total_cost : ℕ := 2700)
  (cost_per_member : ℕ := 26) :
  total_cost / cost_per_member = 103 := by
  sorry

end number_of_members_l482_482225


namespace triangle_perimeter_is_30_square_area_is_75_l482_482597

/-- Perimeter of an equilateral triangle with a given side length --/
def triangle_perimeter (s : ℝ) : ℝ := 3 * s

/-- Height of an equilateral triangle with a given side length --/
def triangle_height (s : ℝ) : ℝ := (s * Real.sqrt 3) / 2

/-- Area of a square with a given side length --/
def square_area (a : ℝ) : ℝ := a * a

/-- Given an equilateral triangle with a side length of 10 m, calculate the perimeter of the triangle --/
theorem triangle_perimeter_is_30 : triangle_perimeter 10 = 30 :=
by
  sorry

/-- Given an equilateral triangle with a side length of 10 m, calculate the area of the square with sides equal to the height of the triangle --/
theorem square_area_is_75 :
  square_area (triangle_height 10) = 75 :=
by
  sorry

end triangle_perimeter_is_30_square_area_is_75_l482_482597


namespace systematic_sampling_correct_l482_482283

theorem systematic_sampling_correct :
  ∀ (products : Finset ℕ) (n : ℕ) (k : ℕ),
    products = Finset.range n.succ ∧ n = 40 ∧ k = 4 →
    (∃ seq : list ℕ, seq = [2, 12, 22, 32]) :=
by
  intros products n k h,
  cases h with h_prod h_rest,
  cases h_rest with h_n h_k,
  use [2, 12, 22, 32],
  sorry

end systematic_sampling_correct_l482_482283


namespace ratio_DO_BO_equals_half_m_plus_n_l482_482232

noncomputable theory

variables (A B C M N O D : Type) 
variables [isTriangle A B C]
variables [isosceles A B C]
variables (m n : ℝ)
variables (AM : ℝ) (BM : ℝ) (AN : ℝ) (BN : ℝ)
variables (BD : ℝ)

def ratio_AM_BM := AM / BM = m
def ratio_CN_BN := CN / BN = n
def line_MN_intersects_BD := ∃ O, line_MN M N ∧ altitude_BD B D O

theorem ratio_DO_BO_equals_half_m_plus_n :
  ratio_AM_BM AM BM m →
  ratio_CN_BN CN BN n → 
  line_MN_intersects_BD M N O D →
  let DO := dist_coords D O in
  let BO := dist_coords B O in
  DO / BO = (m + n) / 2 := 
sorry

end ratio_DO_BO_equals_half_m_plus_n_l482_482232


namespace sqrt_sum_simplify_l482_482689

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l482_482689


namespace class_ranking_l482_482166

variables {a b c d : ℝ}

theorem class_ranking (h1 : a > b + c) (h2 : a + b = c + d) (h3 : b + d > a + c) :
  -- We need to formally represent the conclusion regarding the ranking
  sorry

end class_ranking_l482_482166


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l482_482484

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31 :
  let n := 7^7 - 7^4 in
  let prime_factors := {2, 3, 7, 19} in
  finset.sum prime_factors id = 31 :=
by
  sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l482_482484


namespace geometric_progression_common_ratio_l482_482594

-- Define the problem conditions in Lean 4
theorem geometric_progression_common_ratio (a : ℕ → ℝ) (r : ℝ) (n : ℕ)
  (h_pos : ∀ n, a n > 0) 
  (h_rel : ∀ n, a n = (a (n + 1) + a (n + 2)) / 2 + 2 ) : 
  r = 1 :=
sorry

end geometric_progression_common_ratio_l482_482594


namespace meaningful_sqrt_range_l482_482138

theorem meaningful_sqrt_range (x : ℝ) (h : x - 1 ≥ 0) : x ≥ 1 :=
by
  sorry

end meaningful_sqrt_range_l482_482138


namespace integer_nearest_to_telescope_sum_l482_482006

def sum1 := (1000 : ℝ) * (Finset.sum (Finset.range 10006 \ Finset.range 4) (λ (n : ℕ), 1 / (n^2 - 4 : ℝ)))
def nearestInteger := Real.toNearestInt sum1

theorem integer_nearest_to_telescope_sum :
  nearestInteger = 321 :=
by
  sorry

end integer_nearest_to_telescope_sum_l482_482006


namespace sin_eq_exponential_solution_count_l482_482894

open Real

noncomputable def number_of_solutions (f g : ℝ → ℝ) (a b : ℝ) : ℕ := sorry

theorem sin_eq_exponential_solution_count :
  number_of_solutions sin (λ x, (1 / 3) ^ x) 0 (50 * π) = 50 := sorry

end sin_eq_exponential_solution_count_l482_482894


namespace length_BC_l482_482151

noncomputable def center (O : Type) : Prop := sorry   -- Center of the circle.

noncomputable def diameter (AD : Type) : Prop := sorry   -- AD is a diameter.

noncomputable def chord (ABC : Type) : Prop := sorry   -- ABC is a chord.

noncomputable def radius_equal (BO : ℝ) : Prop := BO = 8   -- BO = 8.

noncomputable def angle_ABO (α : ℝ) : Prop := α = 45   -- ∠ABO = 45°.

noncomputable def arc_CD (β : ℝ) : Prop := β = 90   -- Arc CD subtended by ∠AOD = 90°.

theorem length_BC (O AD ABC : Type) (BO : ℝ) (α β γ : ℝ)
  (h1 : center O)
  (h2 : diameter AD)
  (h3 : chord ABC)
  (h4 : radius_equal BO)
  (h5 : angle_ABO α)
  (h6 : arc_CD β)
  : γ = 8 := 
sorry

end length_BC_l482_482151


namespace range_of_f_neg2_l482_482955

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem range_of_f_neg2 (a b : ℝ) :
  let f_val := f a b,
      f_neg1 := f_val (-1),
      f_1 := f_val 1 in
  (1 ≤ f_neg1 ∧ f_neg1 ≤ 2) →
  (2 ≤ f_1 ∧ f_1 ≤ 4) →
  let f_neg2 := f a b (-2) in
  5 ≤ f_neg2 ∧ f_neg2 ≤ 10 :=
begin
  let f_val := f a b,
  let f_neg1 := f_val (-1),
  let f_1 := f_val 1,
  intros h_neg1 h_1,
  let lb_neg1 := h_neg1.left,
  let ub_neg1 := h_neg1.right,
  let lb_1 := h_1.left,
  let ub_1 := h_1.right,
  let f_val_neg2 := f a b (-2),
  -- Explaining that f(-2) = f(1) + 3 * f(-1)
  have h_eq_f_neg2 : f_val (-2) = f_val (1) + 3 * f_val (-1), from sorry,
  -- Using the equalities to derive the bounds
  have ineq_lower : 5 ≤ f_val (-2), from sorry,
  have ineq_upper : f_val (-2) ≤ 10, from sorry,
  exact ⟨ineq_lower, ineq_upper⟩
end

end range_of_f_neg2_l482_482955


namespace quiz_of_riches_smallest_increase_l482_482599

def percent_increase (prev current : ℕ) : ℚ :=
  ((current - prev).to_rat / prev.to_rat) * 100

def quiz_of_riches : String :=
  let q1 := 200
  let q2 := 400
  let q3 := 700
  let q4 := 1000
  let q5 := 1500
  let increases := [
    percent_increase q1 q2,
    percent_increase q2 q3,
    percent_increase q3 q4,
    percent_increase q4 q5
  ]
  let min_increase_index := increases.enum.min_by (λ (_, v) => v).fst
  if min_increase_index == 2 then "Between Question 3 and 4"
  else "Other"

theorem quiz_of_riches_smallest_increase :
  quiz_of_riches = "Between Question 3 and 4" := sorry

end quiz_of_riches_smallest_increase_l482_482599


namespace proof_problem_l482_482509

section Geometry

variable {t : ℝ} {x y : ℝ} {θ ρ π : ℝ}
variable (x y t : ℝ)
noncomputable def parametric_to_standard := x = (sqrt 2 * t) / 2 + 1 ∧ y = - (sqrt 2 * t) / 2 → x + y - 1 = 0

noncomputable def polar_to_cartesian (p : ℝ) (θ : ℝ) := 
  p = 2 * sqrt 2 * cos (θ + π / 4) → 
  ∃ (x y : ℝ), p^2 = 2 * p * cos θ - 2 * p * sin θ ∧ x^2 + y^2 = 2 * x - 2 * y ∧ x^2 - 2 * x + y^2 + 2 * y = 0

variable (P A B : ℝ × ℝ)

noncomputable def length_PA_PB := (P = (1, 0) ∧ 
  ∃ t1 t2 : ℝ, (x = sqrt 2 * t / 2 + 1 ∧ y = - sqrt 2 * t / 2) ∧ 
  t^2 - sqrt 2 * t - 1 = 0 ∧ (t1 + t2 = sqrt 2 ∧ t1 * t2 = -1)) → 
  abs (dist P A) + abs (dist P B) = sqrt 6

theorem proof_problem (l_eq_c : parametric_to_standard) 
  (c_eq_p : polar_to_cartesian ρ θ) 
  (pa_pb_length : length_PA_PB) :
  ∃ (l : parametric_to_standard (x y t)), 
    ∃ (c : polar_to_cartesian ρ θ), 
      ∃ (pa_pb : length_PA_PB) := by
        sorry

end Geometry

end proof_problem_l482_482509


namespace eccentricity_of_ellipse_l482_482868

theorem eccentricity_of_ellipse :
  let z : ℂ → ℂ := λ z, (z - 2) * (z^2 + 4z + 8) * (z^2 + 6z + 10)
  let solutions := {z | z = 2 ∨ z = -2 + 2 * I ∨ z = -2 - 2 * I ∨ z = -3 + I ∨ z = -3 - I}
  let points := { (Re z, Im z) | z ∈ solutions }
  let h, a, b : ℝ := some_proof_specifics_skipped.h a b 
  let a2 := a^2
  let b2 := b^2
  let c2 := a2 - b2
  let e := real.sqrt (c2 / a2)
  in e = real.sqrt (1 / 7) → (1 + 7 = 8) := 
sorry

end eccentricity_of_ellipse_l482_482868


namespace root_exists_l482_482584

def f (x : ℝ) : ℝ := x - Real.log x - 2

theorem root_exists (k : ℕ) (hk : k ∈ {1, 2, 3, ...}) : 
  ∃ x ∈ (k : ℝ), f x = 0 := 
by
  sorry

end root_exists_l482_482584


namespace areas_equal_of_diagonals_and_angle_equal_l482_482402

theorem areas_equal_of_diagonals_and_angle_equal 
  (A B C D E G F O : Type*) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace G] [MetricSpace F] [MetricSpace O]
  (AC GF : ℝ) (BD EG : ℝ) (BOC EGF : ℝ) 
  (h1 : AC = GF) 
  (h2 : BD = EG) 
  (h3 : BOC = EGF) : 
  area A B C D = area E G F :=
  sorry

end areas_equal_of_diagonals_and_angle_equal_l482_482402


namespace votes_cast_46800_l482_482995

-- Define the election context
noncomputable def total_votes (v : ℕ) : Prop :=
  let percentage_a := 0.35
  let percentage_b := 0.40
  let vote_diff := 2340
  (percentage_b - percentage_a) * (v : ℝ) = (vote_diff : ℝ)

-- Theorem stating the total number of votes cast in the election
theorem votes_cast_46800 : total_votes 46800 :=
by
  sorry

end votes_cast_46800_l482_482995


namespace simplify_radicals_l482_482698

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l482_482698


namespace log_x_squared_y_squared_l482_482115

theorem log_x_squared_y_squared (x y : ℝ) (h1 : Real.log (x * y^2) = 2) (h2 : Real.log (x^3 * y) = 2) : 
  Real.log (x^2 * y^2) = 12 / 5 := 
by
  sorry

end log_x_squared_y_squared_l482_482115


namespace faye_money_left_l482_482033
noncomputable theory

open Real

/-- Define all initial conditions and costs --/
def originalMoney : ℝ := 20
def fatherGift : ℝ := 3 * originalMoney
def motherGift : ℝ := 2 * fatherGift
def grandfatherGift : ℝ := 4 * originalMoney
def muffinsCost : ℝ := 15 * 1.75
def cookiesCost : ℝ := 10 * 2.50
def juiceCost : ℝ := 2 * 4
def candyCost : ℝ := 25 * 0.25
def totalTip : ℝ := 0.15 * (cookiesCost + muffinsCost)

/-- Calculate total money before shopping and total amount spent, finally the remaining money --/
def totalMoneyBeforeShopping : ℝ := originalMoney + fatherGift + motherGift + grandfatherGift
def totalSpentIncludingTip : ℝ := muffinsCost + cookiesCost + juiceCost + candyCost + totalTip 
def moneyLeft : ℝ := totalMoneyBeforeShopping - totalSpentIncludingTip

/-- Prove that Faye has $206.81 left after her shopping spree --/
theorem faye_money_left : moneyLeft = 206.81 := by
  sorry

end faye_money_left_l482_482033


namespace max_y_value_l482_482120

noncomputable def y (x : ℝ) : ℝ :=
  Math.tan (x + 2/3 * Real.pi) - Math.tan (x + Real.pi / 6) + Math.cos (x + Real.pi / 6)

theorem max_y_value :
  ∃ x ∈ set.Icc (-5/12 * Real.pi) (-Real.pi / 3), y x = 11/6 * Real.sqrt 3 :=
sorry

end max_y_value_l482_482120


namespace max_value_AC_AB_l482_482168

-- Given a triangle ABC with angle A = 60 degrees and BC = sqrt(3)
def triangle_ABC (A B C : ℝ) (angleA : A = 60) (sideBC : B = sqrt(3)) : Prop :=
  ∃ AB AC, AB + AC ≤ 2 * sqrt(3)

-- Statement capturing the original problem
theorem max_value_AC_AB {A B C AB AC : ℝ} (hA : A = 60) (hBC : B = sqrt(3)) :
  AC + AB ≤ 2 * sqrt(3) :=
sorry

end max_value_AC_AB_l482_482168


namespace max_yellow_stamps_l482_482646

theorem max_yellow_stamps (N_y : ℕ) (h1 : 20 * 1.1 + 80 * 0.8 + N_y * 2 = 100) : N_y = 7 :=
by 
  sorry

end max_yellow_stamps_l482_482646


namespace solution_inequality_1_solution_inequality_2_l482_482051

theorem solution_inequality_1 (x : ℝ) : -x^2 + 4*x + 5 < 0 ↔ (x < -1 ∨ x > 5) :=
by sorry

theorem solution_inequality_2 (x : ℝ) : 2*x^2 - 5*x + 2 ≤ 0 ↔ (1/2 ≤ x ∧ x ≤ 2) :=
by sorry

end solution_inequality_1_solution_inequality_2_l482_482051


namespace bounded_sequence_max_lambda_l482_482785

noncomputable def sequence (λ : ℝ) (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = λ * (a n) * (a n) + 2

theorem bounded_sequence_max_lambda :
  (∃ (a : ℕ → ℝ) (M : ℝ), sequence λ a ∧ (∀ n : ℕ, a n ≤ M))
  ↔ (0 < λ ∧ λ ≤ 1 / 8) := sorry

end bounded_sequence_max_lambda_l482_482785


namespace trapezoid_distance_ef_l482_482204

noncomputable def isosceles_trapezoid_distance (AD BC AE DE EB : ℝ) (angle : ℝ) : ℝ :=
  if h : (AD = AE + DE) ∧ (angle = π / 4) then 60 * sqrt 2
  else 0

theorem trapezoid_distance_ef : isosceles_trapezoid_distance (40 * sqrt 2) (20 * sqrt 10) (20 * sqrt 2) (20 * sqrt 2) (20 * sqrt 10) (π / 4) = 60 * sqrt 2 :=
by {
  sorry
}

end trapezoid_distance_ef_l482_482204


namespace s_mores_graham_crackers_l482_482177

def graham_crackers_per_smore (total_graham_crackers total_marshmallows : ℕ) : ℕ :=
total_graham_crackers / total_marshmallows

theorem s_mores_graham_crackers :
  let total_graham_crackers := 48
  let available_marshmallows := 6
  let additional_marshmallows := 18
  let total_marshmallows := available_marshmallows + additional_marshmallows
  graham_crackers_per_smore total_graham_crackers total_marshallows = 2 := sorry

end s_mores_graham_crackers_l482_482177


namespace find_m_l482_482354

def g (n : Int) : Int :=
  if n % 2 ≠ 0 then n + 5 else 
  if n % 3 = 0 then n / 3 else n

theorem find_m (m : Int) 
  (h_odd : m % 2 ≠ 0) 
  (h_ggg : g (g (g m)) = 35) : 
  m = 85 := 
by
  sorry

end find_m_l482_482354
