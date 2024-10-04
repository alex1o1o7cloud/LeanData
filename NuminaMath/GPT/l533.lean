import Mathlib

namespace find_set_of_x_l533_533614

theorem find_set_of_x :
  {x : ℝ | ∀ n : ℕ, cos (2^n * x) < 0} = {x | ∃ k : ℤ, x = 2 * k * π + 2 * π / 3 ∨ x = 2 * k * π - 2 * π / 3} :=
sorry

end find_set_of_x_l533_533614


namespace even_weeks_count_l533_533386

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def week_sum_is_even (start_day : ℕ) : Prop :=
  is_even ((7 * start_day + 21) % 2)

def count_even_weeks (start_week : ℕ) (weeks : ℕ) (months_with_31_days : ℕ) : ℕ :=
  if even start_week then
    ((weeks / 2) + months_with_31_days + 1) % weeks
  else
    ((weeks / 2) + months_with_31_days) % weeks

theorem even_weeks_count :
  count_even_weeks 1 52 7 = 30 :=
by
  sorry

end even_weeks_count_l533_533386


namespace value_of_y_l533_533446

theorem value_of_y (x y z : ℚ) (h1 : x + y + z = 120) (h2 : x + 10 = y - 5 ∧ y - 5 = 4 * z) : y = 545 / 9 := 
by 
  have h3 : y - 5 = 4 * z := h2.right,
  have h4 : y = 4 * z + 5 := by linarith,
  have h5 : x = 4 * z - 10 := by linarith using [←h2.left, h3],
  have h6 : (4 * z - 10) + (4 * z + 5) + z = 120,
  { rw [h4, h5] at h1,
    exact h1, },
  have h7 : 9 * z - 5 = 120 := by linarith using h6,
  have h8 : 9 * z = 125 := by linarith,
  have h9 : z = 125 / 9 := by norm_num,
  rw h9 at h4,
  have hy : y = 4 * (125 / 9) + 5 := h4,
  have hy_result : y = (500 / 9) + 5 := by linarith,
  have hy_final : y = 500 / 9 + 45 / 9 := by linarith,
  have hy_overall : y = 545 / 9 := by linarith,
  exact hy_overall,

end value_of_y_l533_533446


namespace angle_between_uv_l533_533245

def u : ℝ × ℝ := (3, 4)
def v : ℝ × ℝ := (4, -3)

theorem angle_between_uv : 
  let dot_product := u.1 * v.1 + u.2 * v.2 in
  let mag_u := Real.sqrt (u.1 ^ 2 + u.2 ^ 2) in
  let mag_v := Real.sqrt (v.1 ^ 2 + v.2 ^ 2) in
  let cos_theta := dot_product / (mag_u * mag_v) in
  Real.arccos cos_theta = π / 2 :=
by
  sorry

end angle_between_uv_l533_533245


namespace three_digit_square_palindromes_count_l533_533677

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem three_digit_square_palindromes_count : ∃ count : ℕ, count = 3 ∧
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ is_square n ∧ is_palindrome n ↔ n = 121 ∨ n = 484 ∨ n = 676 :=
by
  sorry

end three_digit_square_palindromes_count_l533_533677


namespace num_blue_balls_l533_533537

theorem num_blue_balls (total_balls blue_balls : ℕ) 
  (prob_all_blue : ℚ)
  (h_total : total_balls = 12)
  (h_prob : prob_all_blue = 1 / 55)
  (h_prob_eq : (blue_balls / 12) * ((blue_balls - 1) / 11) * ((blue_balls - 2) / 10) = prob_all_blue) :
  blue_balls = 4 :=
by
  -- Placeholder for proof
  sorry

end num_blue_balls_l533_533537


namespace range_of_x_l533_533652

noncomputable def f : ℝ → ℝ := sorry -- f is not explicitly defined, provided it's increasing and satisfies given conditions

theorem range_of_x (h_inc : ∀ {x y : ℝ}, 0 < x → 0 < y → x < y → f(x) < f(y)) 
                   (h_add : ∀ {x y : ℝ}, 0 < x → 0 < y → f(x * y) = f(x) + f(y))
                   (h_value : f(3) = 1) 
                   (hx : 0 < x) (hx8 : 0 < x - 8) (h_ineq : f(x) + f(x - 8) ≤ 2) : 
                     8 < x ∧ x ≤ 9 :=
by sorry

end range_of_x_l533_533652


namespace ratio_of_areas_of_triangles_is_one_l533_533542

-- Define the problem statement in Lean 4
theorem ratio_of_areas_of_triangles_is_one
  (rect : Type) [Rect rect] (P Q : Triangle)
  (O : Point) 
  (h1 : IntersectsAtDiagonals rect O) 
  (h2 : TriangleFromDiagonals rect P Q O) :
  area P / area Q = 1 :=
  sorry

end ratio_of_areas_of_triangles_is_one_l533_533542


namespace math_problem_equivalence_l533_533515

section

variable (x y z : ℝ) (w : String)

theorem math_problem_equivalence (h₀ : x / 15 = 4 / 5) (h₁ : y = 80) (h₂ : z = 0.8) (h₃ : w = "八折"):
  x = 12 ∧ y = 80 ∧ z = 0.8 ∧ w = "八折" :=
by
  sorry

end

end math_problem_equivalence_l533_533515


namespace sum_alternating_binom_eq_l533_533491

theorem sum_alternating_binom_eq {
  S := ∑ k in range 51, (-1 : ℝ) ^ k * (nat.choose 101 (2 * k))
} : S = 2 ^ 50 := 
sorry

end sum_alternating_binom_eq_l533_533491


namespace parallelogram_rhombus_l533_533284

section ParallelogramRhombus

variables {A B C D : Type} [AffineSpace ℝ A B C D]

-- Define a parallelogram
def is_parallelogram (A B C D : A) : Prop :=
  vector.parallel (A - B) (C - D) ∧ vector.parallel (A - D) (B - C)

-- Define perpendicular diagonals
def perpendicular_diagonals (A B C D : A) : Prop :=
  (A - C) ⬝ (B - D) = 0

-- Define rhombus
def is_rhombus (A B C D : A) : Prop :=
  (A - B).norm = (B - C).norm ∧ (B - C).norm = (C - D).norm ∧ (C - D).norm = (D - A).norm

variables {A B C D : A}

-- The Lean theorem statement
theorem parallelogram_rhombus (h_parallelogram : is_parallelogram A B C D)
  (h_perpendicular_diagonals : perpendicular_diagonals A B C D) : 
  is_rhombus A B C D := 
sorry

end ParallelogramRhombus

end parallelogram_rhombus_l533_533284


namespace total_amount_to_pay_l533_533888

namespace StorePromotion

def shoePrices : List ℕ := [74, 92]
def sockPrices : List ℕ := [2, 3, 4, 5]
def bagPrices : List ℕ := [42, 58]
def discount_10_percent (x : ℕ) := x * 10 / 100
def discount_5_percent (x : ℕ) := x * 5 / 100
def sales_tax_percent (x : ℕ) := x * 8 / 100

theorem total_amount_to_pay : 
  let total_before_discounts := 74 + 92 + 2 + 3 + 4 + 5 + 42 + 58,
  let shoes_discount := min 74 92 / 2,
  let socks_discount := min 2 3 4 5,
  let total_after_promotions := total_before_discounts - shoes_discount - socks_discount,
  let discount_for_hundreds := 2 * discount_10_percent 100,
  let discount_for_fifties := discount_5_percent 150,
  let total_after_discounts := total_after_promotions - discount_for_hundreds - discount_for_fifties,
  let tax := sales_tax_percent total_after_discounts in
  total_after_discounts + tax = 208.98 :=
by
  let total_before_discounts := 74 + 92 + 2 + 3 + 4 + 5 + 42 + 58,
  let shoes_discount := min 74 92 / 2,
  let socks_discount := min 2 3 4 5,
  let total_after_promotions := total_before_discounts - shoes_discount - socks_discount,
  let discount_for_hundreds := 2 * discount_10_percent total_after_promotions,
  let discount_for_fifties := discount_5_percent total_after_promotions,
  let total_after_discounts := total_after_promotions - discount_for_hundreds - discount_for_fifties,
  let tax := sales_tax_percent total_after_discounts in
  have eq1 : total_before_discounts = 280 := by rfl,
  have eq2 : shoes_discount = 37 := by rfl,
  have eq3 : socks_discount = 2 := by rfl,
  have eq4 : total_after_promotions = 280 - 37 - 2 := by rfl,
  have eq5 : total_after_promotions = 241 := by rfl,
  have eq6 : discount_for_hundreds = 40 := by rfl,
  have eq7 : discount_for_fifties = 7.5 := by rfl,
  have eq8 : total_after_discounts = 241 - 40 - 7.5 := by rfl,
  have eq9 : total_after_discounts = 193.5 := by rfl,
  have eq10 : tax = 15.48 := by rfl,
  have eq11 : total_after_discounts + tax = 193.5 + 15.48 := by rfl,
  show 193.5 + 15.48 = 208.98 from rfl

end total_amount_to_pay_l533_533888


namespace sum_of_real_solutions_l533_533973

theorem sum_of_real_solutions (x : ℝ) (h : (x^2 + 2*x + 3)^( (x^2 + 2*x + 3)^( (x^2 + 2*x + 3) )) = 2012) : 
  ∃ (x1 x2 : ℝ), (x1 + x2 = -2) ∧ (x1^2 + 2*x1 + 3 = x2^2 + 2*x2 + 3 ∧ x2^2 + 2*x2 + 3 = x^2 + 2*x + 3) := 
by
  sorry

end sum_of_real_solutions_l533_533973


namespace corn_acres_l533_533504

/-- Definition of the acre allocation for beans, wheat, and corn -/
def total_acres : ℕ := 1034
def ratio_beans : ℕ := 5
def ratio_wheat : ℕ := 2
def ratio_corn : ℕ := 4

/-- Proof statement -/
theorem corn_acres :
  let total_parts := ratio_beans + ratio_wheat + ratio_corn in
  let acres_per_part := total_acres / total_parts in
  let corn_parts := ratio_corn in
  acres_per_part * corn_parts = 376 :=
by
  sorry

end corn_acres_l533_533504


namespace janet_initial_action_figures_l533_533723

theorem janet_initial_action_figures (x : ℕ) :
  (x - 2 + 2 * (x - 2) = 24) -> x = 10 := 
by
  sorry

end janet_initial_action_figures_l533_533723


namespace trapezoid_area_calculation_l533_533131

noncomputable def trapezoid_area : ℝ :=
  let y1 := 20
  let y2 := 10
  let x1 := y1 / 2
  let x2 := y2 / 2
  let base1 := x1
  let base2 := x2
  let height := y1 - y2
  (base1 + base2) * height / 2

theorem trapezoid_area_calculation :
  let y1 := 20
  let y2 := 10
  let x1 := y1 / 2
  let x2 := y2 / 2
  let base1 := x1
  let base2 := x2
  let height := y1 - y2
  (base1 + base2) * height / 2 = 75 := 
by
  -- Validation of the translation to Lean 4. Proof steps are omitted.
  sorry

end trapezoid_area_calculation_l533_533131


namespace volume_of_gravel_pile_l533_533886

-- Define the given conditions: diameter and the relationship between height and diameter.
def diameter := 10
def height := 0.6 * diameter

-- Define the radius as half of the diameter.
def radius := diameter / 2

-- The formula for the volume of a cone.
def volume_of_cone (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

-- State the proof problem.
theorem volume_of_gravel_pile : 
  volume_of_cone radius height = 50 * Real.pi := 
by
  sorry

end volume_of_gravel_pile_l533_533886


namespace cindy_smallest_number_of_coins_l533_533938

theorem cindy_smallest_number_of_coins : ∃ n : ℕ, (∀ y : ℕ, (y ∣ n ∧ y > 2 ∧ y < n) → y ∈ {d | d ∣ n} \ {1, n}) ∧ fintype.card {d | d ∣ n} = 18 ∧ n = 972 :=
by
  sorry

end cindy_smallest_number_of_coins_l533_533938


namespace triple_composition_even_l533_533740

variable {α β : Type} [AddGroup α] [AddGroup β]

def even_function (f : α → β) : Prop :=
∀ x : α, f (-x) = f x

theorem triple_composition_even (f : α → α) (h : even_function f) : even_function (λ x, f (f (f x))) :=
by
  unfold even_function at *
  intro x
  specialize h x
  -- Apply the steps mentioned
  sorry

end triple_composition_even_l533_533740


namespace I_eq_2H_l533_533035

def H (x : ℝ) : ℝ :=
  log ((3 + x) / (3 - x))

def I (x : ℝ) : ℝ :=
  H ((2 * x + x ^ 2) / (2 - x ^ 2))

theorem I_eq_2H (x : ℝ) : 
  I x = 2 * H x :=
  sorry

end I_eq_2H_l533_533035


namespace arithmetic_common_difference_l533_533027

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533027


namespace count_paths_l533_533731

-- Define the lattice points and paths
def isLatticePoint (P : ℤ × ℤ) : Prop := true
def isLatticePath (P : ℕ → ℤ × ℤ) (n : ℕ) : Prop :=
  (∀ i, 0 < i → i ≤ n → abs ((P i).1 - (P (i - 1)).1) + abs ((P i).2 - (P (i - 1)).2) = 1)

-- Define F(n) with the given constraints
def numberOfPaths (n : ℕ) : ℕ :=
  -- Placeholder for the actual complex counting logic, which is not detailed here
  sorry

-- Identify F(n) from the initial conditions and the correct result
theorem count_paths (n : ℕ) :
  numberOfPaths n = Nat.choose (2 * n) n :=
sorry

end count_paths_l533_533731


namespace range_of_function_l533_533099

theorem range_of_function :
  let f := λ x : ℝ, x^2 - 4x + 6 in
  let I := set.Ico 1 5 in
  set.range (λ x, f x) ∩ set.Icc (f 2) (f 5) = set.Ico 2 11 :=
by
  let f := λ x : ℝ, x^2 - 4x + 6
  let I := set.Ico 1 5
  have : ∀ x ∈ I, 2 ≤ f x ∧ f x < 11, sorry
  have : ∃ y ∈ I, f y = 2, sorry
  have : ∀ x ∈ I, f x ∈ set.Ico 2 11, sorry
  exact ⟨this⟩

end range_of_function_l533_533099


namespace min_tables_needed_l533_533171

theorem min_tables_needed 
  (n : ℕ) (m : ℕ) (seats_per_table : ℕ) (total_guests : ℕ)
  (h1 : n = 44) (h2 : m = 15) (h3 : seats_per_table = 4) (h4 : total_guests = 44) :
  ∃ k : ℕ, k = 11 ∧ k * seats_per_table = total_guests := 
by {
  use 11,
  split,
  { refl },
  { rw [h3, nat.mul_comm],
    exact h4 }
}

end min_tables_needed_l533_533171


namespace proper_subsets_of_A_is_7_l533_533671

noncomputable def universal_set : Set ℕ := {1, 2, 3, 4}

noncomputable def complement_A : Set ℕ := {2}

open Set

theorem proper_subsets_of_A_is_7 
  (U : Set ℕ := universal_set) 
  (complement_A : Set ℕ := complement_A)
  (A : Set ℕ := U \ complement_A) :
  (2 ^ A.card - 1) = 7 :=
by
  sorry

end proper_subsets_of_A_is_7_l533_533671


namespace solve_for_x_l533_533414

theorem solve_for_x :
  let x := (sqrt (3^2 + 4^2)) / (sqrt (25 - 1))
  in x = 5 * sqrt 6 / 12 :=
by
  sorry

end solve_for_x_l533_533414


namespace square_perimeter_l533_533196

theorem square_perimeter (s : ℕ) (h : 5 * s / 2 = 40) : 4 * s = 64 := by
  sorry

end square_perimeter_l533_533196


namespace option_D_not_random_l533_533850

-- Definitions
def face_probs (faces: List ℕ) : List ℚ :=
  (faces.groupBy id).map (λ g => (g.length : ℕ) / faces.length)

def is_random (probs: List ℚ) : Prop :=
  probs.all (λ p => p = 1 / (probs.length : ℚ))

-- Conditions
def option_D_faces : List ℕ := [1, 2, 2, 3, 4, 5]
def option_D_probs := face_probs option_D_faces

-- Theorem to prove
theorem option_D_not_random : ¬ is_random option_D_probs := by
  sorry

end option_D_not_random_l533_533850


namespace equal_vertex_diagonal_l533_533966

-- Define the theorem statement
theorem equal_vertex_diagonal {n : ℕ} (h : n ≥ 2) :
  (∀ (P : Type) [simple_polygon P] (v : set P), 
    card v = 2 * n → 
    (∃ d : P, is_diagonal d ∧ splits_into_equal_parts P d n)) ↔ n = 2 :=
sorry

end equal_vertex_diagonal_l533_533966


namespace problem_statement_l533_533989

theorem problem_statement (a b : ℝ) (h : a^2 + |b + 1| = 0) : (a + b)^2015 = -1 := by
  sorry

end problem_statement_l533_533989


namespace train_crossing_time_l533_533551

noncomputable def relative_speed_kmh (speed_train : ℕ) (speed_man : ℕ) : ℕ := speed_train + speed_man

noncomputable def kmh_to_mps (speed_kmh : ℕ) : ℝ := speed_kmh * 1000 / 3600

noncomputable def crossing_time (length_train : ℕ) (speed_train_kmh : ℕ) (speed_man_kmh : ℕ) : ℝ :=
  let relative_speed_kmh := relative_speed_kmh speed_train_kmh speed_man_kmh
  let relative_speed_mps := kmh_to_mps relative_speed_kmh
  length_train / relative_speed_mps

theorem train_crossing_time :
  crossing_time 210 25 2 = 28 :=
  by
  sorry

end train_crossing_time_l533_533551


namespace triple_composition_even_l533_533739

variable {α β : Type} [AddGroup α] [AddGroup β]

def even_function (f : α → β) : Prop :=
∀ x : α, f (-x) = f x

theorem triple_composition_even (f : α → α) (h : even_function f) : even_function (λ x, f (f (f x))) :=
by
  unfold even_function at *
  intro x
  specialize h x
  -- Apply the steps mentioned
  sorry

end triple_composition_even_l533_533739


namespace smallest_class_number_l533_533536

-- Define the conditions
def num_classes : Nat := 24
def num_selected_classes : Nat := 4
def total_sum : Nat := 52
def sampling_interval : Nat := num_classes / num_selected_classes

-- The core theorem to be proved
theorem smallest_class_number :
  ∃ x : Nat, x + (x + sampling_interval) + (x + 2 * sampling_interval) + (x + 3 * sampling_interval) = total_sum ∧ x = 4 := by
  sorry

end smallest_class_number_l533_533536


namespace older_brother_is_14_l533_533829

theorem older_brother_is_14 {Y O : ℕ} (h1 : Y + O = 26) (h2 : O = Y + 2) : O = 14 :=
by
  sorry

end older_brother_is_14_l533_533829


namespace books_pairs_count_l533_533679

theorem books_pairs_count 
  (mystery_books : Fin 4 → ℕ)
  (fantasy_books : Fin 4 → ℕ)
  (biography_books : Fin 4 → ℕ)
  (sci_fi_books : Fin 4 → ℕ) :
  (∑ i j k, (choose 4 2) * (4 * 4)) = 96 := 
  by 
    sorry

end books_pairs_count_l533_533679


namespace cargo_to_cruise_ratio_l533_533173

theorem cargo_to_cruise_ratio
  (C S F : ℕ)
  (hS1 : S = C + 6)
  (hS2 : S = 7 * F)
  (hCruise : 4 = 4)
  (hTotal : 4 + C + S + F = 28) :
  C / 4 = 2 :=
by
  sorry

end cargo_to_cruise_ratio_l533_533173


namespace sum_of_interior_angles_pentagon_l533_533106

theorem sum_of_interior_angles_pentagon : 
  let n := 5 in
  180 * (n - 2) = 540 :=
  by
  -- Introducing the variable n for the number of sides
  let n := 5
  -- Using the given formula to calculate the sum of the interior angles
  have h : 180 * (n - 2) = 540 := by sorry
  exact h

end sum_of_interior_angles_pentagon_l533_533106


namespace sum_alternating_binom_eq_l533_533492

theorem sum_alternating_binom_eq {
  S := ∑ k in range 51, (-1 : ℝ) ^ k * (nat.choose 101 (2 * k))
} : S = 2 ^ 50 := 
sorry

end sum_alternating_binom_eq_l533_533492


namespace Olive_can_reach_one_l533_533335

noncomputable def can_reach_one (N : ℕ) : Prop :=
  ∃(f : ℕ → ℕ) (n : ℕ), f n = 1 ∧ ∀i < n, (∃m : ℕ, f i = m * N) ∨ (∃k : ℕ, rearrangement f i k)

-- Function to check whether k is a rearrangement of n
def rearrangement (n k : ℕ) : Prop :=
  n.digits 10 ~ k.digits 10  -- List permutation of digits in base 10

theorem Olive_can_reach_one (N : ℕ) :
  N % 3 ≠ 0 ↔ can_reach_one N :=
begin
  sorry
end

end Olive_can_reach_one_l533_533335


namespace line_perpendicular_to_plane_l533_533742

open EuclideanGeometry

-- Definitions for the main structures
variables {P : Type*} [EuclideanSpace P]
variables {l : Line P} {a : Plane P} {A B C : P}

-- Assumptions about the conditions
axiom l_outside_plane_a : ¬ (l ⊆ a)
axiom a_contains_triangle : A ∈ a ∧ B ∈ a ∧ C ∈ a
axiom l_perpendicular_AB : l ⊥ (line_through A B)
axiom l_perpendicular_AC : l ⊥ (line_through A C)

-- Statement of the problem in Lean
theorem line_perpendicular_to_plane :
  l ⊥ a :=
sorry

end line_perpendicular_to_plane_l533_533742


namespace financial_outcome_l533_533398

-- Define the selling prices and conditions
def selling_price (pipe : ℕ) : ℝ := 1.50
def profit_percentage (pipe : ℕ) : ℝ := if pipe = 1 then 0.25 else -0.15
def cost_price (pipe : ℕ) : ℝ := selling_price pipe / (1 + profit_percentage pipe)

-- Calculate the total cost and total revenue
def total_cost : ℝ := (cost_price 1) + (cost_price 2)
def total_revenue : ℝ := selling_price 1 + selling_price 2

-- Define the net result
def net_result : ℝ := total_revenue - total_cost

theorem financial_outcome : net_result ≈ 0.0353 :=
by
  -- Insert proof steps here
  sorry

end financial_outcome_l533_533398


namespace purchase_plan_count_l533_533548

theorem purchase_plan_count (B1 B2 B3 : Type) : 
  let books := {B1, B2, B3}
  ∃ plans, plans = 7 := 
by
  sorry

end purchase_plan_count_l533_533548


namespace find_x_of_arithmetic_mean_l533_533597

theorem find_x_of_arithmetic_mean :
  ∃ x : ℚ, ((x + 6) + 18 + 3x + 12 + (x + 9) + (3x - 5)) / 6 = 19 ∧ x = 37 / 4 :=
begin
  sorry
end

end find_x_of_arithmetic_mean_l533_533597


namespace product_of_two_numbers_l533_533119

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y ≠ 0) 
  (h2 : (x + y) / (x - y) = 7)
  (h3 : xy = 24 * (x - y)) : xy = 48 := 
sorry

end product_of_two_numbers_l533_533119


namespace ab_cd_value_l533_533684

theorem ab_cd_value (a b c d : ℝ) 
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = 0) : 
  a * b + c * d = -31 :=
by
  sorry

end ab_cd_value_l533_533684


namespace find_x_given_y_l533_533448

-- Given that x and y are always positive and x^2 and y vary inversely.
-- i.e., we have a relationship x^2 * y = k for a constant k,
-- and given that y = 8 when x = 3, find the value of x when y = 648.

theorem find_x_given_y
  (x y : ℝ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_inv : ∀ x y, x^2 * y = 72)
  (h_y : y = 648) : x = 1 / 3 :=
by
  sorry

end find_x_given_y_l533_533448


namespace correct_operation_only_l533_533158

theorem correct_operation_only (a b x y : ℝ) : 
  (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) ∧ 
  (4 * x^2 * y - x^2 * y ≠ 3) ∧ 
  ((a + b)^2 ≠ a^2 + b^2) ∧
  ((ab)^3 = a^3 * b^3) := 
by 
  sorry

end correct_operation_only_l533_533158


namespace percentage_died_by_bombardment_l533_533529

theorem percentage_died_by_bombardment (P_initial : ℝ) (P_remaining : ℝ) (died_percentage : ℝ) (fear_percentage : ℝ) :
  P_initial = 3161 → P_remaining = 2553 → fear_percentage = 0.15 → 
  P_initial - (died_percentage/100) * P_initial - fear_percentage * (P_initial - (died_percentage/100) * P_initial) = P_remaining → 
  abs (died_percentage - 4.98) < 0.01 :=
by
  intros h1 h2 h3 h4
  sorry

end percentage_died_by_bombardment_l533_533529


namespace ratio_of_centroid_and_sides_l533_533734

variables (A B C O : Type) [AddCommGroup A] [AffineSpace A O]

-- Conditions
variables (AB BC AC : ℝ)
variables (x y : ℝ)
variables (cosA sinA cosC sinC : ℝ)

-- Given properties of the triangle and circumcenter
variable (h1 : AB + BC = (2 * Real.sqrt 3 / 3) * AC)
variable (h2 : sinC * (cosA - Real.sqrt 3) + cosC * sinA = 0)
variable (h3 : ∀ (AO : A), AO = x • (A - B) + y • (A - C))

-- Objective to Prove
theorem ratio_of_centroid_and_sides 
  (AO_eq : ∀ (AO : A), AO = x • (A - B) + y • (A - C))
  (hxy : x = 1 / 3 ∧ y = 1 / 3) :
  x / y = 1 := 
sorry

end ratio_of_centroid_and_sides_l533_533734


namespace convert_235_to_base_five_l533_533843

theorem convert_235_to_base_five : nat_to_base 235 5 = [1, 4, 2, 0] :=
sorry

end convert_235_to_base_five_l533_533843


namespace vacation_costs_l533_533917

theorem vacation_costs :
  let a := 15
  let b := 22.5
  let c := 22.5
  a + b + c = 45 → b - a = 7.5 := by
sorry

end vacation_costs_l533_533917


namespace probability_dd_l533_533834

theorem probability_dd (P_DD P_Dd P_dd : ℝ)
  (H1 : P_DD = 1/4) (H2 : P_Dd = 1/2) (H3 : P_dd = 1/4) : 
  let P_draw_dd := 
    (P_Dd * (1/2) * P_Dd * (1/2)) + 
    (P_Dd * (1/2) * P_dd) + 
    (P_dd * P_Dd * (1/2)) + 
    (P_dd * P_dd)
  in P_draw_dd = 1/4 :=
sorry

end probability_dd_l533_533834


namespace sum_of_real_roots_l533_533616

theorem sum_of_real_roots :
  let f (x : Real) := sin (π * (x^2 - x + 1)) 
  let g (x : Real) := sin (π * (x - 1))
  sum_roots : Real :=
  f x = g x ∧ 0 ≤ x ∧ x ≤ 2 
    sum_roots = 3 + Real.sqrt 3 :=
by
  sorry

end sum_of_real_roots_l533_533616


namespace find_g_neg5_l533_533043

def g : ℝ → ℝ :=
λ x, if x ≤ -3 then 3 * x - 4 else 7 - 3 * x

theorem find_g_neg5 : g (-5) = -19 := 
by {
  -- Proof goes here
  sorry
}

end find_g_neg5_l533_533043


namespace min_marked_necessary_l533_533146

-- A 12x12 board represented as a set of coordinates
def board_12x12 := fin 12 × fin 12

-- A four-cell figure's possible placements, given by its upper-left corner position.
def figure_placements : list (set (fin 12 × fin 12)) :=
  [(0,0), (0,1), (1,0), (1,1), (1,2), (2,0), (2,1)].map (λ c, {(0,0), (0,1), (1,0), (1,1)}.image (prod.add c))

-- The marking condition function
def well_marked (marked : set (fin 12 × fin 12)) :=
  ∀ p ∈ board_12x12, ¬(figure_placements.map (λ s, disjoint marked s)).all id

-- The core hypothesis and the final theorem
theorem min_marked_necessary : ∃ k : ℕ, k = 48 ∧ ∃ marked : set (fin 12 × fin 12), well_marked marked ∧ marked.card = k :=
by sorry

end min_marked_necessary_l533_533146


namespace smallest_x_gx_eq_g2023_l533_533753

def g (x : ℝ) : ℝ :=
  if 2 ≤ x ∧ x ≤ 4 then 2 - |x - 3| else sorry

theorem smallest_x_gx_eq_g2023
  (h1 : ∀ x > 0, g (4 * x) = 4 * g x)
  (h2 : ∀ x, 2 ≤ x ∧ x ≤ 4 → g x = 2 - |x - 3|) :
  ∃ x : ℝ, g x = g 2023 ∧ ∀ y : ℝ, g y = g 2023 → x ≤ y := 
sorry

end smallest_x_gx_eq_g2023_l533_533753


namespace orange_slices_l533_533422

theorem orange_slices (x : ℕ) (hx1 : 5 * x = x + 8) : x + 2 * x + 5 * x = 16 :=
by {
  sorry
}

end orange_slices_l533_533422


namespace k_starts_at_10_l533_533163

variable (V_k V_l : ℝ)
variable (t_k t_l : ℝ)

-- Conditions
axiom k_faster_than_l : V_k = 1.5 * V_l
axiom l_speed : V_l = 50
axiom l_start_time : t_l = 9
axiom meet_time : t_k + 3 = 12
axiom distance_apart : V_l * 3 + V_k * (12 - t_k) = 300

-- Proof goal
theorem k_starts_at_10 : t_k = 10 :=
by
  sorry

end k_starts_at_10_l533_533163


namespace exists_thousand_triples_l533_533780

theorem exists_thousand_triples :
  ∃ (a b c : ℕ), (∃ n : ℕ, n > 1000 ∧ (∀ i : fin n, (∃ (a b c : ℕ), a^15 + b^15 = c^16))) :=
sorry

end exists_thousand_triples_l533_533780


namespace polygon_interior_angle_sum_l533_533109

theorem polygon_interior_angle_sum (n : ℕ) (hn : 3 ≤ n) :
  (n - 2) * 180 + 180 = 2007 → n = 13 := by
  sorry

end polygon_interior_angle_sum_l533_533109


namespace prime_divisors_sum_2018_l533_533439

theorem prime_divisors_sum_2018 :
  (∃ p q : ℕ, p.prime ∧ q.prime ∧ 2018 = p * q ∧ p + q = 1011) :=
sorry

end prime_divisors_sum_2018_l533_533439


namespace sum_y_coordinates_of_other_two_vertices_l533_533305

theorem sum_y_coordinates_of_other_two_vertices (A B : ℝ × ℝ) (hA : A = (5, 22)) (hB : B = (12, -3)) :
    let y_sum := 19 in
    ∃ C D : ℝ × ℝ, C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B ∧ 
        (A.1 = C.1 ∨ A.1 = D.1) ∧ (B.1 = C.1 ∨ B.1 = D.1) ∧ 
        (A.2 = y_sum / 2 ∨ B.2 = y_sum / 2) ∧ 
        C.2 + D.2 = y_sum := 
sorry

end sum_y_coordinates_of_other_two_vertices_l533_533305


namespace minimum_value_is_16_l533_533228

noncomputable def minimum_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) : ℝ :=
  (x^3 / (y - 1) + y^3 / (x - 1))

theorem minimum_value_is_16 (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  minimum_value_expression x y hx hy ≥ 16 :=
sorry

end minimum_value_is_16_l533_533228


namespace final_price_is_9_80_l533_533533

def original_price : ℝ := 10.00
def increase_percentage : ℝ := 0.40
def decrease_percentage : ℝ := 0.30

theorem final_price_is_9_80 :
  let increased_price := original_price * (1 + increase_percentage) in
  let final_price := increased_price * (1 - decrease_percentage) in
  final_price = 9.80 := 
by
  sorry

end final_price_is_9_80_l533_533533


namespace plane_PAC_perpendicular_plane_PBC_l533_533714

open EuclideanGeometry

variables 
  (O : Point)
  (P A B C : Point)
  (h1 : diameter O A B)
  (h2 : ∃ plane : Plane, plane.contains_circle O ∧ PA ⟂ plane)
  (h3 : on_circumference O C ∧ C ≠ A ∧ C ≠ B)

theorem plane_PAC_perpendicular_plane_PBC
  (plane_PAC plane_PBC : Plane)
  (hPAC : plane_PAC.contains_points [P, A, C])
  (hPBC : plane_PBC.contains_points [P, B, C]) :
  plane_PAC ⟂ plane_PBC :=
sorry

end plane_PAC_perpendicular_plane_PBC_l533_533714


namespace max_three_digit_is_931_l533_533474

def greatest_product_three_digit : Prop :=
  ∃ (a b c d e : ℕ), {a, b, c, d, e} = {1, 3, 5, 8, 9} ∧
  1 ≤ a ∧ a ≤ 9 ∧
  1 ≤ b ∧ b ≤ 9 ∧
  1 ≤ c ∧ c ≤ 9 ∧
  1 ≤ d ∧ d ≤ 9 ∧
  1 ≤ e ∧ e ≤ 9 ∧
  (100 * a + 10 * b + c) * (10 * d + e) = 81685 ∧
  (100 * a + 10 * b + c) = 931

theorem max_three_digit_is_931 : greatest_product_three_digit :=
begin
  sorry
end

end max_three_digit_is_931_l533_533474


namespace differential_solution_correct_l533_533972

noncomputable def y (x : ℝ) : ℝ := (x + 1)^2

theorem differential_solution_correct : 
  (∀ x : ℝ, deriv (deriv y) x = 2) ∧ y 0 = 1 ∧ (deriv y 0) = 2 := 
by
  sorry

end differential_solution_correct_l533_533972


namespace casper_candy_problem_l533_533584

theorem casper_candy_problem (o y gr : ℕ) (n : ℕ) (h1 : 10 * o = 16 * y) (h2 : 16 * y = 18 * gr) (h3 : 18 * gr = 18 * n) :
    n = 40 :=
by
  sorry

end casper_candy_problem_l533_533584


namespace part_a_l533_533114

theorem part_a 
  (R : ℝ) 
  (cylinder1 cylinder2 cylinder3 : ℝ × ℝ → Prop)
  (axes_mutually_perpendicular : cylinder1 = cylinder2 = cylinder3 = λ p : ℝ × ℝ, p.1 ^ 2 + p.2 ^ 2 = R^2)
  (cylinders_touch_each_other : ∀ x y z : ℝ, R^2 = x^2 + y^2 → R^2 = y^2 + z^2 → R^2 = z^2 + x^2): 
  smallest_sphere_radius = R * (Real.sqrt 2 - 1) :=
by
  sorry

end part_a_l533_533114


namespace f_periodic_4_l533_533319

noncomputable def f : ℝ → ℝ := sorry -- f is some function ℝ → ℝ

theorem f_periodic_4 (h : ∀ x, f x = -f (x + 2)) : f 100 = f 4 := 
by
  sorry

end f_periodic_4_l533_533319


namespace find_x_value_l533_533630

open Nat

theorem find_x_value : ∃ x : ℕ, x > 0 ∧ x ≤ 5 ∧ 
  (fact 5 / fact (5 - x) = 2 * (fact 6 / fact (7 - x))) ∧ x = 3 :=
by
  sorry

end find_x_value_l533_533630


namespace num_conditions_imply_neg_rs_false_exactly_three_imply_neg_rs_false_l533_533078

variables (r s : Prop)

theorem num_conditions_imply_neg_rs_false :
  (r ∧ s) ∨ (r ∧ ¬s) ∨ (¬r ∧ s) → (r ∨ s) :=
by {
  sorry
}

theorem exactly_three_imply_neg_rs_false :
  -- Define the conditions
  let conditions : List Prop := [
    (r ∧ s),
    (r ∧ ¬s),
    (¬r ∧ s),
    (¬r ∧ ¬s)
  ] in
  -- Count the ones that imply the negation of "r and s are both false"
  let num_implications := List.count (λ c, c → (r ∨ s)) conditions in
  num_implications = 3 :=
by {
  sorry
}

end num_conditions_imply_neg_rs_false_exactly_three_imply_neg_rs_false_l533_533078


namespace integral_calculation_l533_533511

noncomputable def definite_integral : ℝ :=
  ∫ x in 1..(Real.exp 2), (Real.log x)^2 / Real.sqrt x

theorem integral_calculation :
  definite_integral = 24 * Real.exp 1 - 32 :=
by sorry

end integral_calculation_l533_533511


namespace save_after_increase_l533_533898

def monthly_saving_initial (salary : ℕ) (saving_percentage : ℕ) : ℕ :=
  salary * saving_percentage / 100

def monthly_expense_initial (salary : ℕ) (saving : ℕ) : ℕ :=
  salary - saving

def increase_by_percentage (amount : ℕ) (percentage : ℕ) : ℕ :=
  amount * percentage / 100

def new_expense (initial_expense : ℕ) (increase : ℕ) : ℕ :=
  initial_expense + increase

def new_saving (salary : ℕ) (expense : ℕ) : ℕ :=
  salary - expense

theorem save_after_increase (salary saving_percentage increase_percentage : ℕ) 
  (H_salary : salary = 5500) 
  (H_saving_percentage : saving_percentage = 20) 
  (H_increase_percentage : increase_percentage = 20) :
  new_saving salary (new_expense (monthly_expense_initial salary (monthly_saving_initial salary saving_percentage)) (increase_by_percentage (monthly_expense_initial salary (monthly_saving_initial salary saving_percentage)) increase_percentage)) = 220 := 
by
  sorry

end save_after_increase_l533_533898


namespace verify_root_l533_533254

-- Define the quadratic equation and the additional condition as hypotheses
variable {a b c : ℝ}

-- Declare the main proof stating that x = -2 is a root given the conditions
theorem verify_root : (4 * a - 2 * b + c = 0) → (a * (-2)^2 + b * (-2) + c = 0) :=
by
  -- We provide the condition as hypothesis
  intro h
  -- We state the goal that we want to prove
  -- Substitute x = -2
  have h1 : (a * 4 - 2 * b + c = 4 * a - 2 * b + c) := by
  sorry
  -- Use the given condition to prove the goal
  rw h at h1
  exact h1

end verify_root_l533_533254


namespace andrew_total_stickers_l533_533050

theorem andrew_total_stickers
  (total_stickers : ℕ)
  (ratio_susan: ℕ)
  (ratio_andrew: ℕ)
  (ratio_sam: ℕ)
  (sam_share_ratio: 2/3)
  (initial_andrew_share: ℕ)
  (initial_sam_share: ℕ) :
  total_stickers = 1500 →
  ratio_susan = 1 →
  ratio_andrew = 1 →
  ratio_sam = 3 →
  initial_andrew_share = (total_stickers / (ratio_susan + ratio_andrew + ratio_sam)) * ratio_andrew →
  initial_sam_share = (total_stickers / (ratio_susan + ratio_andrew + ratio_sam)) * ratio_sam →
  (ratio_andrew = (initial_andrew_share + initial_sam_share * sam_share_ratio)) →
  initial_andrew_share + initial_sam_share * sam_share_ratio = 900 :=
by
  intros _ _ _ _ _ _
  sorry

end andrew_total_stickers_l533_533050


namespace find_a_l533_533666

noncomputable def f (a x : ℝ) : ℝ :=
  a * (1 / x - 1) + log x

theorem find_a (a : ℝ) : (∀ x : ℝ, x > 0 → f a x ≥ 0) → a = 1 :=
by
  sorry

end find_a_l533_533666


namespace grasshoppers_can_jump_left_l533_533770

noncomputable def grasshoppers_initial_positions (n : ℕ) : list ℕ := sorry
noncomputable def right_jump (pos : ℕ) (dist : ℕ) : ℕ := pos + dist
noncomputable def left_jump (pos : ℕ) (dist : ℕ) : ℕ := pos - dist

theorem grasshoppers_can_jump_left (n : ℕ) (h : n = 2019) 
    (initial_positions : list ℕ) :
    (∃ i j, i ≠ j ∧ abs (initial_positions[i] - initial_positions[j]) = 1) →
    (∃ k l, k ≠ l ∧ abs (left_jump (initial_positions[k]) 2019 - 
                         left_jump (initial_positions[l]) 2019) = 1) :=
sorry

end grasshoppers_can_jump_left_l533_533770


namespace prove_common_difference_l533_533006

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533006


namespace max_value_b_h_derivative_inequality_l533_533635

-- Part I
theorem max_value_b (a b : ℝ) (h1 : a > 0) 
  (h2 : ∃ x₀ > 0, (x₀^2 + 4 * a * x₀ + 1 = 6 * a^2 * Real.log x₀ + 2 * b + 1) 
                  ∧ (2 * x₀ + 4 * a = 6 * a^2 / x₀)) :
  b = (5 / 2) * a^2 - 3 * a^2 * Real.log a 
  ∧ b ≤ (3 / 2) * Real.exp (2 / 3) := sorry

-- Part II
theorem h_derivative_inequality 
  (a : ℝ) (h : a ≥ Real.sqrt 3 - 1) 
  (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hx₁ₓ₂ : x₁ ≠ x₂) :
  let f (x : ℝ) := x^2 + 4 * a * x + 1;
      g (x : ℝ) := 6 * a^2 * Real.log x + 2 * b + 1;
      h (x : ℝ) := f x + g x
  in (h x₂ - h x₁) / (x₂ - x₁) > 8 := sorry

end max_value_b_h_derivative_inequality_l533_533635


namespace max_min_sum_l533_533334

/-- In a 4x4 grid filled with numbers 1 and 2, if any 3x3 subgrid's sum is divisible by 4 
and the sum of all 16 numbers is not divisible by 4, then the max sum is 30 
and the min sum is 19. -/
theorem max_min_sum (f : Fin 4 ⊕ Fin 4 → Nat)
  (h1 : ∀ i j : Fin 2, ∃ n : Nat, (∑ x in (Finset.range 3).image (λ x, f (Fin.castAdd 1 x, Fin.castAdd 1 x)), n + i + j) % 4 = 0)
  (h2 : (∑ x in (Finset.finRange 16), f x) % 4 ≠ 0) :
  19 ≤ (∑ x in (Finset.finRange 16), f x) ∧ (∑ x in (Finset.finRange 16), f x) ≤ 30 :=
sorry

end max_min_sum_l533_533334


namespace sum_in_range_l533_533104

theorem sum_in_range :
  let a := (27 : ℚ) / 8
  let b := (22 : ℚ) / 5
  let c := (67 : ℚ) / 11
  13 < a + b + c ∧ a + b + c < 14 :=
by
  sorry

end sum_in_range_l533_533104


namespace growth_rate_is_20_percent_verify_target_yield_l533_533863

-- Definitions based on the conditions
def initial_yield : ℝ := 700
def third_phase_yield : ℝ := 1008
def target_fourth_phase_yield : ℝ := 1200

-- Define the statement that proves the growth rate is 20%
theorem growth_rate_is_20_percent : 
  ∃ x : ℝ, 0 < x ∧ (initial_yield * (1 + x)^2 = third_phase_yield) ∧ x = 0.2 := 
by 
  existsi (0.2 : ℝ)
  split
  sorry

-- Define the statement that verifies the target yield can be reached
theorem verify_target_yield : 
  ∃ x : ℝ, 0 < x ∧ (initial_yield * (1 + x)^2 = third_phase_yield) ∧ (third_phase_yield * (1 + x) ≥ target_fourth_phase_yield) := 
by 
  existsi (0.2 : ℝ)
  split
  sorry

end growth_rate_is_20_percent_verify_target_yield_l533_533863


namespace find_m_l533_533646

theorem find_m
  (θ : ℝ)
  (m : ℝ)
  (h1 : sin θ = (m - 3) / (m + 5))
  (h2 : cos θ = (4 - 2m) / (m + 5))
  (fourth_quad : θ ∈ Icc (-2 * π) (0)) :
  m = 0 :=
sorry

end find_m_l533_533646


namespace sum_of_numbers_l533_533263

theorem sum_of_numbers : 
  (87 + 91 + 94 + 88 + 93 + 91 + 89 + 87 + 92 + 86 + 90 + 92 + 88 + 90 + 91 + 86 + 89 + 92 + 95 + 88) = 1799 := 
by 
  sorry

end sum_of_numbers_l533_533263


namespace find_a_33_l533_533799

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (d : ℤ)
variable (a1 : ℤ)

noncomputable def a_8 : ℤ := a 7
noncomputable def a_5 : ℤ := a 4
noncomputable def S_8 : ℤ := S 8
noncomputable def S_5 : ℤ := S 5

axiom h1 : a_8 - a_5 = 9
axiom h2 : S_8 - S_5 = 66
axiom h3 : ∀ n, a n = a1 + n * d
axiom h4 : ∀ n, S n = n * (2 * a1 + (n - 1) * d) / 2

theorem find_a_33 : a 32 = 100 := 
by {
  sorry
}

end find_a_33_l533_533799


namespace inequality_ab_leq_a_b_l533_533407

theorem inequality_ab_leq_a_b (a b : ℝ) (x : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  a * b ≤ (a * (Real.sin x) ^ 2 + b * (Real.cos x) ^ 2) * (a * (Real.cos x) ^ 2 + b * (Real.sin x) ^ 2)
  ∧ (a * (Real.sin x) ^ 2 + b * (Real.cos x) ^ 2) * (a * (Real.cos x) ^ 2 + b * (Real.sin x) ^ 2) ≤ (a + b)^2 / 4 := 
sorry

end inequality_ab_leq_a_b_l533_533407


namespace MrSami_sold_20_shares_of_stock_x_l533_533830

theorem MrSami_sold_20_shares_of_stock_x
    (shares_v : ℕ := 68)
    (shares_w : ℕ := 112)
    (shares_x : ℕ := 56)
    (shares_y : ℕ := 94)
    (shares_z : ℕ := 45)
    (additional_shares_y : ℕ := 23)
    (increase_in_range : ℕ := 14)
    : (shares_x - (shares_y + additional_shares_y - ((shares_w - shares_z + increase_in_range) - shares_y - additional_shares_y)) = 20) :=
by
  sorry

end MrSami_sold_20_shares_of_stock_x_l533_533830


namespace tangent_slope_angle_l533_533283

theorem tangent_slope_angle (m : ℝ) (h : (1 : ℝ)^2 + (√3 : ℝ)^2 = m) :
  ∃ k : ℝ, k = - (1 / (√3)) ∧ k = -√3 / 3 ∧ (π - (acos (k / (1 + k^2)).sqrt)) = 150 * (π / 180) :=
by
  sorry

end tangent_slope_angle_l533_533283


namespace ratio_circumradius_legs_only_sometimes_l533_533462

noncomputable def TriangleI (L B P K R : ℝ) :=
  P = 2 * L + B ∧
  K = sqrt ((L + B / 2) * (L + B / 2 - L) * (L + B / 2 - L) * (L + B / 2 - B)) ∧
  R = L^2 / (4 * sqrt (L^2 - B^2 / 4))

noncomputable def TriangleII (l b p k r : ℝ) :=
  p = 2 * l + b ∧
  k = sqrt ((l + b / 2) * (l + b / 2 - l) * (l + b / 2 - l) * (l + b / 2 - b)) ∧
  r = l^2 / (4 * sqrt (l^2 - b^2 / 4))

theorem ratio_circumradius_legs_only_sometimes
  (L B P K R l b p k r : ℝ) 
  (hL : L ≠ l) :
  TriangleI L B P K R →
  TriangleII l b p k r →
  (R / r = L / l) ↔ false :=
sorry

end ratio_circumradius_legs_only_sometimes_l533_533462


namespace max_product_of_functions_is_35_l533_533789

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem max_product_of_functions_is_35 
  (R_f : ∀ x : ℝ, 1 ≤ f(x) ∧ f(x) ≤ 7)
  (R_g : ∀ x : ℝ, -3 ≤ g(x) ∧ g(x) ≤ 5)
  (max_condition: ∃ x, f(x) = 7 ∧ g(x) = 5) :
  ∃ b, b = 35 ∧ ∀ x, f(x) * g(x) ≤ b :=
by {
  use 35,
  sorry
}

end max_product_of_functions_is_35_l533_533789


namespace average_speed_correct_l533_533174

-- Define the speeds for each hour
def speed_hour1 := 90 -- km/h
def speed_hour2 := 40 -- km/h
def speed_hour3 := 60 -- km/h
def speed_hour4 := 80 -- km/h
def speed_hour5 := 50 -- km/h

-- Define the total time of the journey
def total_time := 5 -- hours

-- Calculate the sum of distances
def total_distance := speed_hour1 + speed_hour2 + speed_hour3 + speed_hour4 + speed_hour5

-- Define the average speed calculation
def average_speed := total_distance / total_time

-- The proof problem: average speed is 64 km/h
theorem average_speed_correct : average_speed = 64 := by
  sorry

end average_speed_correct_l533_533174


namespace proof_probability_events_l533_533111

-- Definitions
def total_books := 7
def total_ways_to_select_2_books := Nat.choose total_books 2

def chinese_books := 4
def math_books := 3

def ways_to_select_2_math_books := Nat.choose math_books 2
def ways_to_select_1_chinese_and_1_math_book := chinese_books * math_books

-- Propositions
def probability_event_A : Prop :=
  (ways_to_select_2_math_books / total_ways_to_select_2_books.toReal) = (1 / 7)

def probability_event_B : Prop :=
  (ways_to_select_1_chinese_and_1_math_book / total_ways_to_select_2_books.toReal) = (4 / 7)

-- Statement to be proven
theorem proof_probability_events :
  probability_event_A ∧ probability_event_B :=
by
  sorry

end proof_probability_events_l533_533111


namespace count_four_digit_integers_with_thousands_digit_three_l533_533678

theorem count_four_digit_integers_with_thousands_digit_three :
  ∃ n : ℕ, n = 1000 ∧ 
    ∀ (x : ℕ), 3000 ≤ x ∧ x < 4000 → x - 3000 < 1000 :=
begin
  -- proof statement only
  sorry
end

end count_four_digit_integers_with_thousands_digit_three_l533_533678


namespace problem_l533_533306

theorem problem (x : ℕ) (h1 : x > 0) (h2 : ∃ k : ℕ, 7 - x = k^2) : x = 3 ∨ x = 6 ∨ x = 7 :=
by
  sorry

end problem_l533_533306


namespace return_kittens_due_to_rehoming_problems_l533_533046

def num_breeding_rabbits : Nat := 10
def kittens_first_spring : Nat := num_breeding_rabbits * num_breeding_rabbits
def kittens_adopted_first_spring : Nat := kittens_first_spring / 2
def kittens_second_spring : Nat := 60
def kittens_adopted_second_spring : Nat := 4
def total_rabbits : Nat := 121

def non_breeding_rabbits_from_first_spring : Nat :=
  total_rabbits - num_breeding_rabbits - kittens_second_spring

def kittens_returned_to_lola : Prop :=
  non_breeding_rabbits_from_first_spring - kittens_adopted_first_spring = 1

theorem return_kittens_due_to_rehoming_problems : kittens_returned_to_lola :=
sorry

end return_kittens_due_to_rehoming_problems_l533_533046


namespace problem_statement_l533_533101

noncomputable def a (n : ℕ) : ℚ :=
  if n = 0 then 1/4
  else if n = 1 then 1/5
  else if h : n > 1 then 1 / (n + 3)
  else 0  -- For completeness, but this case is not actually used.

lemma problem_conditions (n : ℕ) (hn : n > 0) :
  ∑ i in finset.range n, a i * a (i + 1) = n * a 0 * a n := sorry

theorem problem_statement : 
  (∑ i in finset.range 97, 1 / a i) = 5044 := sorry

end problem_statement_l533_533101


namespace sum_alternating_binomial_coeff_l533_533490

theorem sum_alternating_binomial_coeff (S : ℕ) : 
  (S = ∑ k in Finset.range 51, (-1)^k * Nat.choose 101 (2*k)) → 
  S = 2^50 := by
  intro h
  sorry

end sum_alternating_binomial_coeff_l533_533490


namespace new_perimeter_of_rectangle_l533_533451

theorem new_perimeter_of_rectangle (w : ℝ) (A : ℝ) (new_area_factor : ℝ) (L : ℝ) (L' : ℝ) (P' : ℝ) 
  (h_w : w = 10) (h_A : A = 150) (h_new_area_factor: new_area_factor = 4 / 3)
  (h_orig_length : L = A / w) (h_new_area: A' = new_area_factor * A) (h_A' : A' = 200)
  (h_new_length : L' = A' / w) (h_perimeter : P' = 2 * (L' + w)) 
  : P' = 60 :=
sorry

end new_perimeter_of_rectangle_l533_533451


namespace sum_ab_l533_533664

noncomputable def log_base (a x : ℝ) := Real.log x / Real.log a

def f (a b x : ℝ) : ℝ := log_base a ((2 - x) / (b + x))

def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f(x)

def conditions (a b : ℝ) : Prop :=
  (0 < a) ∧ (a < 1) ∧ (is_odd_function (f a b)) ∧ (∀ x, x ∈ Ioo (-2:ℝ) (2 * a)) ∧ (∀ y, y ∈ Ioo (-∞:ℝ) 1)

theorem sum_ab (a b : ℝ) (h : conditions a b) : a + b = Real.sqrt 2 + 1 :=
sorry

end sum_ab_l533_533664


namespace boat_speed_problem_l533_533522

noncomputable def upstream_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

noncomputable def downstream_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

noncomputable def speed_of_current (ds_speed : ℝ) (us_speed : ℝ) : ℝ := (ds_speed - us_speed) / 2

theorem boat_speed_problem :
  let distance := 1 in
  let upstream_time_minutes := 20 in
  let downstream_time_minutes := 9 in
  let minutes_to_hours := 60 in
  let us_speed := upstream_speed distance (upstream_time_minutes / minutes_to_hours) in
  let ds_speed := downstream_speed distance (downstream_time_minutes / minutes_to_hours) in
  speed_of_current ds_speed us_speed = 1.835 := 
by
  sorry

end boat_speed_problem_l533_533522


namespace range_of_eccentricity_l533_533274

theorem range_of_eccentricity (a b : ℝ) (F : ℝ × ℝ) (A B : ℝ × ℝ) (α : ℝ) (e : ℝ) :
  (a > b) → (b > 0) →
  (A.fst^2 / a^2 + A.snd^2 / b^2 = 1) →
  (B = (-A.fst, -A.snd)) →
  (F = (c, 0)) → -- Assuming F lies on the x-axis, with c being the distance from origin to focus
  (angle A F B = α) →
  (α ∈ set.Icc (real.pi/6) (real.pi/4)) →
  (AF ⊥ BF) → -- using a predicate for perpendicular vectors
  e = (1 / (real.sqrt 2 * real.sin (α + real.pi/4))) →
  (set.Icc (real.sqrt 2 / 2) (real.sqrt 3 - 1)).mem e := sorry

end range_of_eccentricity_l533_533274


namespace simplify_expression_l533_533413

theorem simplify_expression (x : ℝ) : 5 * x + 6 - x + 12 = 4 * x + 18 :=
by sorry

end simplify_expression_l533_533413


namespace collinear_points_l533_533915

open EuclideanGeometry

variable {A B C D E F : Point}

theorem collinear_points
  (h_triangle : Triangle A B C)
  (h_tangent : Tangent (circumcircle A B C) A D)
  (h_perp_bisector_AB : Perpendicular (Line B D) (perpendicular_bisector A B E))
  (h_perp_bisector_AC : Perpendicular (Line C D) (perpendicular_bisector A C F)) :
  Collinear D E F :=      
sorry

end collinear_points_l533_533915


namespace total_animals_l533_533568

-- Definitions of the initial conditions
def initial_beavers := 20
def initial_chipmunks := 40
def doubled_beavers := 2 * initial_beavers
def decreased_chipmunks := initial_chipmunks - 10

theorem total_animals (initial_beavers initial_chipmunks doubled_beavers decreased_chipmunks : ℕ)
    (h1 : doubled_beavers = 2 * initial_beavers)
    (h2 : decreased_chipmunks = initial_chipmunks - 10) :
    (initial_beavers + initial_chipmunks) + (doubled_beavers + decreased_chipmunks) = 130 :=
by 
  sorry

end total_animals_l533_533568


namespace distance_between_points_is_6sqrt10_l533_533609

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points_is_6sqrt10 :
  distance (0, 18) (6, 0) = 6 * real.sqrt 10 :=
by
  sorry

end distance_between_points_is_6sqrt10_l533_533609


namespace exists_good_set_l533_533265

open Set Function Finset

variable (n : ℕ) (hn : n ≥ 2)

noncomputable def M : Finset ℕ := finset.range (2 * n + 1)

def f (S : {S' : Finset ℕ // S'.card = n - 1 ∧ S' ⊆ M n}) : ℕ := sorry

theorem exists_good_set (n : ℕ) (hn : n ≥ 2) :
  ∃ T : Finset ℕ, T.card = n ∧ T ⊆ M n ∧ ∀ k ∈ T, f n ⟨T.erase k, sorry⟩ ≠ k :=
sorry

end exists_good_set_l533_533265


namespace total_leaves_l533_533727

def fernTypeA_fronds := 15
def fernTypeA_leaves_per_frond := 45
def fernTypeB_fronds := 20
def fernTypeB_leaves_per_frond := 30
def fernTypeC_fronds := 25
def fernTypeC_leaves_per_frond := 40

def fernTypeA_count := 4
def fernTypeB_count := 5
def fernTypeC_count := 3

theorem total_leaves : 
  fernTypeA_count * (fernTypeA_fronds * fernTypeA_leaves_per_frond) + 
  fernTypeB_count * (fernTypeB_fronds * fernTypeB_leaves_per_frond) + 
  fernTypeC_count * (fernTypeC_fronds * fernTypeC_leaves_per_frond) = 
  8700 := 
sorry

end total_leaves_l533_533727


namespace probability_dist_geq_side_l533_533097

noncomputable def num_points := 5
noncomputable def num_pairs_total := (num_points.choose 2 = 10)
noncomputable def num_pairs_min_dist := (4.choose 2 = 6)

theorem probability_dist_geq_side : (6 / 10) = (3 / 5) :=
by
  sorry

end probability_dist_geq_side_l533_533097


namespace tax_base_amount_l533_533878

theorem tax_base_amount (tax_amount : ℝ) (tax_rate : ℝ) (base_amount : ℝ) : 
  tax_amount = (tax_rate / 100) * base_amount → 
  tax_amount = 65 → 
  tax_rate = 65 → 
  base_amount = 100 := 
begin
  intros h1 h2 h3,
  rw [h2, h3] at h1,
  simp at h1,
  assumption,
end

end tax_base_amount_l533_533878


namespace gcd_incorrect_l533_533496

theorem gcd_incorrect (a b c : ℕ) (h : a * b * c = 3000) : gcd (gcd a b) c ≠ 15 := 
sorry

end gcd_incorrect_l533_533496


namespace total_animals_l533_533570

-- Definitions of the initial conditions
def initial_beavers := 20
def initial_chipmunks := 40
def doubled_beavers := 2 * initial_beavers
def decreased_chipmunks := initial_chipmunks - 10

theorem total_animals (initial_beavers initial_chipmunks doubled_beavers decreased_chipmunks : ℕ)
    (h1 : doubled_beavers = 2 * initial_beavers)
    (h2 : decreased_chipmunks = initial_chipmunks - 10) :
    (initial_beavers + initial_chipmunks) + (doubled_beavers + decreased_chipmunks) = 130 :=
by 
  sorry

end total_animals_l533_533570


namespace shift_to_obtain_transformed_function_l533_533117

-- Define the original function
def original_function (x : ℝ) : ℝ := sin (2 * x)

-- Define the transformed function
def transformed_function (x : ℝ) : ℝ := sin (2 * x + π / 3)

-- Define the function after shifting
def shifted_function (x : ℝ) : ℝ := sin (2 * (x + π / 6))

-- The theorem to be proved: the transformed function can be obtained by shifting the original function
theorem shift_to_obtain_transformed_function :
  ∀ (x : ℝ), transformed_function x = shifted_function x :=
by
  sorry

end shift_to_obtain_transformed_function_l533_533117


namespace find_angle_BCA_l533_533361

noncomputable def angle_BCA : ℝ := 50

theorem find_angle_BCA 
  (ΔABC : Type) [triangle ΔABC]
  (A B C E : ΔABC)
  (h1 : ∠BAC = 30)
  (h2 : 3 * AE = EC)
  (h3 : ∠EBC = 10) :
  ∠BCA = angle_BCA :=
begin
  sorry
end

end find_angle_BCA_l533_533361


namespace cone_lsa_15pi_l533_533812

variable (r l : ℝ)

def cone_lateral_surface_area (r l : ℝ) : ℝ :=
  Real.pi * r * l

theorem cone_lsa_15pi (h₁ : r = 3) (h₂ : l = 5) :
  cone_lateral_surface_area r l = 15 * Real.pi := by
  rw [h₁, h₂, cone_lateral_surface_area]
  sorry

end cone_lsa_15pi_l533_533812


namespace sum_reciprocal_roots_between_1_and_2_l533_533224

def cubic_polynomial (x : ℝ) : ℝ := 15 * x^3 - 25 * x^2 + 8 * x - 1

theorem sum_reciprocal_roots_between_1_and_2
  (a b c : ℝ)
  (h_roots : cubic_polynomial a = 0 ∧ cubic_polynomial b = 0 ∧ cubic_polynomial c = 0)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_bounds : 1 < a ∧ a < 2 ∧ 1 < b ∧ b < 2 ∧ 1 < c ∧ c < 2) :
  \(\frac{1}{2-a} + \frac{1}{2-b} + \frac{1}{2-c} = \frac{13}{3}\) :=
sorry

end sum_reciprocal_roots_between_1_and_2_l533_533224


namespace minimum_fm_fn_dot_product_l533_533656

noncomputable def parabola_focus : ℝ × ℝ := (2, 0)

def parabola (x y : ℝ) : Prop := y^2 = -8 * x

def circle (x y : ℝ) (radius : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = radius^2

def line_tangent_circle (x y : ℝ) : Prop :=
-- (equation for line passing through F and tangent to circle at N)

def vector_dot_product (v1 v2 : (ℝ × ℝ)) : ℝ :=
-- definition for dot product of vectors

theorem minimum_fm_fn_dot_product :
  ∀ (F M N : ℝ × ℝ),
  parabola_focus = F →
  parabola M.1 M.2 →
  circle M.1 M.2 1 →
  line_tangent_circle (N.1 N.2) →
  (vector_dot_product (F - M) (F - N)) = 3 :=
sorry

end minimum_fm_fn_dot_product_l533_533656


namespace problem_statement_l533_533945

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 4) + cos (2 * x + π / 4)

theorem problem_statement :
  (∀ x : ℝ, 0 < x ∧ x < π / 2 → f x < f (π / 2 - x)) ∧ 
  (∀ x : ℝ, f x = f (π / 2 - (x - π / 2))) :=
sorry

end problem_statement_l533_533945


namespace no_integer_roots_p_eq_2016_l533_533632

noncomputable def p (a b c d : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots_p_eq_2016 
  (a b c d : ℤ)
  (h₁ : p a b c d 1 = 2015)
  (h₂ : p a b c d 2 = 2017) :
  ¬ ∃ x : ℤ, p a b c d x = 2016 :=
sorry

end no_integer_roots_p_eq_2016_l533_533632


namespace day_of_20th_is_Thursday_l533_533801

noncomputable def day_of_week (d : ℕ) : String :=
  match d % 7 with
  | 0 => "Saturday"
  | 1 => "Sunday"
  | 2 => "Monday"
  | 3 => "Tuesday"
  | 4 => "Wednesday"
  | 5 => "Thursday"
  | 6 => "Friday"
  | _ => "Unknown"

theorem day_of_20th_is_Thursday (s1 s2 s3: ℕ) (h1: 2 ≤ s1) (h2: s1 ≤ 30) (h3: s2 = s1 + 14) (h4: s3 = s2 + 14) (h5: s3 ≤ 30) (h6: day_of_week s1 = "Sunday") : 
  day_of_week 20 = "Thursday" :=
by
  sorry

end day_of_20th_is_Thursday_l533_533801


namespace distance_between_circle_centers_l533_533031

noncomputable def triangle_PQR : Type :=
let PQ := 13
let PR := 15
let QR := 14

-- Definitions of the sides of the triangle
structure triangle :=
(PQ PR QR : ℝ)

-- Definitions of the inradius and exradius
noncomputable def inradius (K s : ℝ) : ℝ :=
K / s

noncomputable def exradius (K s a : ℝ) : ℝ :=
K / (s - a)

/-- Proof problem: The distance between the centers of the incircle and the excircle opposite QR -/
theorem distance_between_circle_centers :
  let s := (PQ + PR + QR) / 2 in
  let K := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR)) in
  let r := inradius K s in
  let ra := exradius K s (s - PQ) in
  let I := r in
  let E := ra in
  let dist := E - I in
  dist = 5 * Real.sqrt 13 :=
by
  sorry

end distance_between_circle_centers_l533_533031


namespace angle_BXD_l533_533716

open Real EuclideanGeometry

def rectangle (A B C D : Point) : Prop :=
  (∃ (a b : ℝ), b / a = sqrt 2 ∧
                B = A + (a, 0) ∧
                C = B + (0, b) ∧
                D = A + (0, b))

def equidistant_point (X B D : Point) (a : ℝ) : Prop :=
  dist X B = a ∧ dist X D = a

theorem angle_BXD (A B C D X : Point) (a : ℝ) (h1 : rectangle A B C D)
  (h2 : B = A + (a, 0)) (h3 : equidistant_point X B D a) :
  ∠ B X D = 2 * π / 3 := by {
  sorry
}

end angle_BXD_l533_533716


namespace negation_of_proposition_l533_533298

theorem negation_of_proposition (x : ℝ) (h : 2 * x + 1 ≤ 0) : ¬ (2 * x + 1 ≤ 0) ↔ 2 * x + 1 > 0 := 
by
  sorry

end negation_of_proposition_l533_533298


namespace max_product_931_l533_533467

open Nat

theorem max_product_931 : ∀ a b c d e : ℕ,
  -- Digits must be between 1 and 9 and all distinct.
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
  {1, 3, 5, 8, 9} = {a, b, c, d, e} ∧ 
  -- Three-digit and two-digit numbers.
  100 * a + 10 * b + c ≤ 999 ∧ 100 * a + 10 * b + c ≥ 100 ∧
  10 * d + e ≤ 99 ∧ 10 * d + e ≥ 10
  -- Then the resulting product is maximized when abc = 931.
  → 100 * a + 10 * b + c = 931 := 
by
  intros a b c d e h_distinct h_digits h_3digit_range h_2digit_range
  sorry

end max_product_931_l533_533467


namespace min_cuts_to_one_meter_pieces_l533_533110

theorem min_cuts_to_one_meter_pieces (x y : ℕ) (hx : x + y = 30) (hl : 3 * x + 4 * y = 100) : (2 * x + 3 * y) = 70 := 
by sorry

end min_cuts_to_one_meter_pieces_l533_533110


namespace gcd_84_294_315_l533_533808

def gcd_3_integers : ℕ := Nat.gcd (Nat.gcd 84 294) 315

theorem gcd_84_294_315 : gcd_3_integers = 21 :=
by
  sorry

end gcd_84_294_315_l533_533808


namespace probability_of_bug9_is_zero_l533_533352

-- Definitions based on conditions provided
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def non_vowels : List Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
def digits_or_vowels : List Char := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'E', 'I', 'O', 'U']

-- Defining the number of choices for each position
def first_symbol_choices : Nat := 5
def second_symbol_choices : Nat := 21
def third_symbol_choices : Nat := 20
def fourth_symbol_choices : Nat := 15

-- Total number of possible license plates
def total_plates : Nat := first_symbol_choices * second_symbol_choices * third_symbol_choices * fourth_symbol_choices

-- Probability calculation for the specific license plate "BUG9"
def probability_bug9 : Nat := 0

theorem probability_of_bug9_is_zero : probability_bug9 = 0 := by sorry

end probability_of_bug9_is_zero_l533_533352


namespace lulu_final_cash_l533_533388

-- Definitions of the problem conditions
def initial_amount : ℕ := 65
def spent_on_ice_cream : ℕ := 5
def spent_on_tshirt (remaining : ℕ) : ℕ := remaining / 2
def deposit_in_bank (remaining : ℕ) : ℕ := remaining / 5

-- The proof problem statement
theorem lulu_final_cash :
  ∃ final_cash : ℕ,
    final_cash = initial_amount - spent_on_ice_cream - spent_on_tshirt (initial_amount - spent_on_ice_cream) - 
                      deposit_in_bank (spent_on_tshirt (initial_amount - spent_on_ice_cream)) ∧
    final_cash = 24 :=
by {
  sorry
}

end lulu_final_cash_l533_533388


namespace number_with_square_starting_with_100_nines_l533_533068

noncomputable def starts_with_n_nines (a : ℝ) (n : ℕ) : Prop :=
  let threshold := 1 - 10^(-n)
  0 < a ∧ a < 1 ∧ (threshold ≤ a ∨ threshold ≤ a*a)

theorem number_with_square_starting_with_100_nines (a : ℝ) :
  starts_with_n_nines (a*a) 100 → starts_with_n_nines a 100 :=
by
  sorry

end number_with_square_starting_with_100_nines_l533_533068


namespace number_of_true_propositions_l533_533094

-- Define the propositions
def original (a b c : ℝ) : Prop := a > b → c ≠ 0 ∧ ac^2 > bc^2
def converse (a b c : ℝ) : Prop := c ≠ 0 ∧ ac^2 > bc^2 → a > b
def inverse (a b c : ℝ) : Prop := a ≤ b → c ≠ 0 ∧ ac^2 ≤ bc^2
def contrapositive (a b c : ℝ) : Prop := c ≠ 0 ∧ ac^2 ≤ bc^2 → a ≤ b

-- Prove the number of true propositions is 2
theorem number_of_true_propositions (a b c : ℝ) : 
  (¬ original a b c) ∧ (¬ contrapositive a b c) ∧ converse a b c ∧ inverse a b c :=
sorry

end number_of_true_propositions_l533_533094


namespace f_monotonically_decreasing_l533_533294

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 5 * x + 2 * Real.log x

-- State the theorem that f(x) is monotonically decreasing on the interval (1/2, 2)
theorem f_monotonically_decreasing : 
  ∀ x ∈ Ioo (1 / 2 : ℝ) 2, deriv f x < 0 :=
by
  sorry

end f_monotonically_decreasing_l533_533294


namespace count_inequalities_l533_533237

theorem count_inequalities :
  let expressions := [
    x - y,
    x ≤ y,
    x + y,
    x^2 - 3 * y,
    x ≥ 0,
    (1 : Rat) / 2 * x ≠ 3
  ] in
  (card (set_of (λ e, ∃ x y : ℝ, e = (x - y) ∨ e = (x ≤ y) ∨ e = (x + y) ∨ e = (x^2 - 3*y) ∨ e = (x ≥ 0) ∨ e = ((1 : Rat) / 2 * x ≠ 3))) = 3 :=
begin
  sorry
end

end count_inequalities_l533_533237


namespace find_p_q_sum_l533_533376

noncomputable def roots (r1 r2 r3 : ℝ) := (r1 + r2 + r3 = 11 ∧ r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3) ∧ 
                                         (∀ x : ℝ, x^3 - 11*x^2 + (r1 * r2 + r2 * r3 + r3 * r1) * x - r1 * r2 * r3 = 0)

theorem find_p_q_sum : ∃ (p q : ℝ), roots 2 4 5 → p + q = 78 :=
by
  sorry

end find_p_q_sum_l533_533376


namespace ice_cream_not_sold_total_l533_533210

theorem ice_cream_not_sold_total :
  let chocolate_initial := 50
  let mango_initial := 54
  let vanilla_initial := 80
  let strawberry_initial := 40
  let chocolate_sold := (3 / 5 : ℚ) * chocolate_initial
  let mango_sold := (2 / 3 : ℚ) * mango_initial
  let vanilla_sold := (75 / 100 : ℚ) * vanilla_initial
  let strawberry_sold := (5 / 8 : ℚ) * strawberry_initial
  let chocolate_not_sold := chocolate_initial - chocolate_sold
  let mango_not_sold := mango_initial - mango_sold
  let vanilla_not_sold := vanilla_initial - vanilla_sold
  let strawberry_not_sold := strawberry_initial - strawberry_sold
  chocolate_not_sold + mango_not_sold + vanilla_not_sold + strawberry_not_sold = 73 :=
by sorry

end ice_cream_not_sold_total_l533_533210


namespace area_of_ABCD_is_correct_l533_533122

theorem area_of_ABCD_is_correct :
  let r := 15
  let theta := real.pi / 2
  let full_circle_area := real.pi * r^2
  let sector_area := (theta / (2 * real.pi)) * full_circle_area
  let total_area := 2 * sector_area
  total_area = 112.5 * real.pi :=
by
  let r := 15
  let theta := real.pi / 2
  let full_circle_area := real.pi * r^2
  let sector_area := (theta / (2 * real.pi)) * full_circle_area
  let total_area := 2 * sector_area
  sorry

end area_of_ABCD_is_correct_l533_533122


namespace ellipse_scalar_product_l533_533273

theorem ellipse_scalar_product :
  ∀ (M P : ℝ × ℝ) (y₀ : ℝ),
  let C := (-2, 0), D := (2, 0) in
  let OM := (2, y₀) in
  let OP := ( (-2 * y₀^2 + 16) / (y₀^2 + 8), (8 * y₀) / (y₀^2 + 8) ) in
  ∃ (a b : ℝ) (h₁ : a > b) (h₂ : b > 0),
  (a = 2) ∧ (b = sqrt 2) ∧
  ( ∀ (M P : ℝ × ℝ), (OM • OP) = 4 ) := sorry

end ellipse_scalar_product_l533_533273


namespace F_midpoint_BC_l533_533513

-- Define all the necessary geometric objects and conditions.
variables {O A B C D E P F : Type} [metric_space O]
variables [affine_space ℝ O] -- An affine space over the reals.
variables (circle_O : metric.sphere O 1) -- Represents the circle with center O and radius 1.

-- Conditions
def diameter (A B O : O) : Prop := metric.dist A B = 2 * metric.dist O A
def midpoint (M A B : O) : Prop := metric.dist A M = metric.dist M B ∧ metric.dist A B = 2 * metric.dist A M
def perpendicular (l1 l2 : set O) : Prop := ∃ P ∈ l1 ∩ l2, ∀ (A ∈ l1) (B ∈ l2), ∠ A P B = π / 2
def non_diameter_chord (C D : O) : Prop := metric.dist C D ≠ 2 * metric.dist O D

-- Definitions of the geometric configurations.
def configuration :=
  diameter A B O ∧
  non_diameter_chord C D ∧
  perp ↔ perpendicular {A B} {C D} ∧
  is_midpoint E O C ↔ midpoint E O C ∧
  extends_to_circle P A E ↔ (∃ P, metric.dist A P ∈ circle_O ∧ ∃ l, ∀ x ∈ l, x = A ∨ x = P) ∧
  meets_at_point F D P B C

-- Prove that F is the midpoint of BC
theorem F_midpoint_BC (h : configuration) : midpoint F B C :=
sorry -- The proof itself is not required based on the given instructions.

end F_midpoint_BC_l533_533513


namespace greatest_mondays_in_45_days_l533_533482

/-- 
Given that a week consists of 7 days, 
and each week has exactly 1 Monday, 
prove that the greatest number of Mondays 
that can occur in the first 45 days of a year is 7.
-/
theorem greatest_mondays_in_45_days : 
  (∃ (n : ℕ), n = 45) → 
  (∀ (week : ℕ), week * 7 ≤ 45 → week.mondays = 1) → 
  ∃ (max_mondays : ℕ), max_mondays = 7 :=
sorry

end greatest_mondays_in_45_days_l533_533482


namespace assignment_schemes_count_l533_533977

theorem assignment_schemes_count (a b c d e : Prop) 
  (ClassA ClassB ClassC : { set Prop }) 
  (h_class_size_constraints : ∀ class_set ∈ {ClassA, ClassB, ClassC}, 1 ≤ class_set.size ∧ class_set.size ≤ 2) 
  (h_a_not_in_ClassA : a ∉ ClassA) : 
  (card {asmt | (ClassA ∪ ClassB ∪ ClassC = {a, b, c, d, e}) ∧
    (a ∉ ClassA) ∧
    (∀ class_set ∈ {ClassA, ClassB, ClassC}, 1 ≤ class_set.size ∧ class_set.size ≤ 2)} = 90) := sorry

end assignment_schemes_count_l533_533977


namespace ed_did_not_lose_marbles_l533_533233

variables (ed_initial ed_now doug : ℕ)

-- Define the conditions
def condition1 : Prop := ed_initial = doug + 12
def condition2 : Prop := doug = 5
def condition3 : Prop := ed_now = 17

-- Define the target to prove
def ed_lost_no_marbles : Prop := ed_now = ed_initial - 0

theorem ed_did_not_lose_marbles (h1 : condition1) (h2 : condition2) (h3 : condition3) : ed_lost_no_marbles :=
by {
  -- We will provide explicit proof steps if necessary, but for now we use sorry to skip the proof
  sorry
}

end ed_did_not_lose_marbles_l533_533233


namespace NineChaptersProblem_l533_533351

-- Conditions: Assign the given conditions to variables
variables (x y : Int)
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Proof problem: Prove that the system of equations is consistent with the given conditions
theorem NineChaptersProblem : condition1 x y ∧ condition2 x y := sorry

end NineChaptersProblem_l533_533351


namespace prove_common_difference_l533_533014

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533014


namespace width_of_playground_is_250_l533_533452

noncomputable def total_area_km2 : ℝ := 0.6
def num_playgrounds : ℕ := 8
def length_of_playground_m : ℝ := 300

theorem width_of_playground_is_250 :
  let total_area_m2 := total_area_km2 * 1000000
  let area_of_one_playground := total_area_m2 / num_playgrounds
  let width_of_playground := area_of_one_playground / length_of_playground_m
  width_of_playground = 250 := by
  sorry

end width_of_playground_is_250_l533_533452


namespace Mike_height_l533_533396

theorem Mike_height (h_mark: 5 * 12 + 3 = 63) (h_mark_mike:  63 + 10 = 73) (h_foot: 12 = 12)
: 73 / 12 = 6 ∧ 73 % 12 = 1 := 
sorry

end Mike_height_l533_533396


namespace cone_lsa_15pi_l533_533813

variable (r l : ℝ)

def cone_lateral_surface_area (r l : ℝ) : ℝ :=
  Real.pi * r * l

theorem cone_lsa_15pi (h₁ : r = 3) (h₂ : l = 5) :
  cone_lateral_surface_area r l = 15 * Real.pi := by
  rw [h₁, h₂, cone_lateral_surface_area]
  sorry

end cone_lsa_15pi_l533_533813


namespace tan_A_tan_C_l533_533362

theorem tan_A_tan_C (A B C H : Type) [Triangle ABC] [Orthocenter H] [Altitude BD] (HD : Real) (HB : Real)
  (h1: HD = 3)
  (h2: HB = 10) :
  tanA * tanC = 3 / Real.sqrt 178 := 
by
  sorry

end tan_A_tan_C_l533_533362


namespace compute_g3_pow4_l533_533790

noncomputable def f (x : ℝ) : ℝ := sorry  -- Placeholder for function f
noncomputable def g (x : ℝ) : ℝ := sorry  -- Placeholder for function g

theorem compute_g3_pow4 :
  (∀ x, x ≥ 1 → f(g(x)) = x^3) →
  (∀ x, x ≥ 1 → g(f(x)) = x^4) →
  g(81) = 81 →
  [g(3)]^4 = 81 :=
by
  intros h1 h2 h3
  sorry

end compute_g3_pow4_l533_533790


namespace min_sum_of_primes_divisible_by_30_l533_533752

open Nat

def is_prime (n : ℕ) : Prop := Prime n

theorem min_sum_of_primes_divisible_by_30 (p q r s : ℕ) 
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (hs : is_prime s)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ p ≠ r ∧ p ≠ s ∧ q ≠ s)
  (h_divisible : 30 ∣ (p * q - r * s)) : p + q + r + s ≥ 54 := 
sorry

end min_sum_of_primes_divisible_by_30_l533_533752


namespace uniformly_increasing_intervals_l533_533689

def uniformly_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  (∀ x ∈ I, ∀ y ∈ I, x < y → f x < f y) ∧ (∀ x ∈ I, ∀ y ∈ I, x < y → (f x / x) < (f y / y))

theorem uniformly_increasing_intervals (I : Set ℝ) :
  (uniformly_increasing (λ x, x + (Real.exp x) / x) I) →
  I ⊆ Set.Iio (-2) ∪ Set.Ioi 2 :=
by
  sorry

end uniformly_increasing_intervals_l533_533689


namespace cube_sum_to_zero_l533_533846

theorem cube_sum_to_zero :
  (∑ i in Finset.range (50+1), i^3 + ∑ i in Finset.range (50), (101 + i)^3) +
  (∑ i in Finset.range (50+1), (-i)^3 + ∑ i in Finset.range (50), (-(101 + i))^3) = 0 :=
by
  sorry

end cube_sum_to_zero_l533_533846


namespace radius_of_sphere_l533_533599

-- Define the problem parameters and conditions
def edge_length := 2
def center_to_face := (edge_length : ℝ) / 2
def tangent_sphere_distance (r : ℝ) := center_to_face - r
def position_of_tangent_sphere_center (r : ℝ) := (r, r, r)
def distance_from_center_to_tangent (r : ℝ) : ℝ := Real.sqrt 3 * r

-- Main theorem stating the radius of the sphere
theorem radius_of_sphere :
  ∃ r : ℝ, r = (Real.sqrt 3 - 1) / 2 :=
sorry

end radius_of_sphere_l533_533599


namespace ratio_of_visible_spots_l533_533554

theorem ratio_of_visible_spots (S S1 : ℝ) (h1 : ∀ (fold_type : ℕ), 
  (fold_type = 1 ∨ fold_type = 2 ∨ fold_type = 3) → 
  (if fold_type = 1 ∨ fold_type = 2 then S1 else S) = S1) : S1 / S = 2 / 3 := 
sorry

end ratio_of_visible_spots_l533_533554


namespace street_length_l533_533188

variable (time_in_minutes : ℕ) (speed_in_kmph : ℝ)

theorem street_length (h1 : time_in_minutes = 14) (h2 : speed_in_kmph = 4.628571428571429) : 
  let speed_in_m_per_min := speed_in_kmph * 1000 / 60 in
  let length_of_street := speed_in_m_per_min * time_in_minutes in
  length_of_street ≈ 1080 :=
begin
  sorry -- Proof to be provided
end

end street_length_l533_533188


namespace partition_Q_plus_problem_l533_533867

-- Part (a): Definitions, Relations, and Partition Existence
def partition_Q_plus (A B C : set ℚ) : Prop :=
  (∀ x ∈ B, ∀ y ∈ A, (x * y) ∈ B) ∧
  (∀ x ∈ B, ∀ y ∈ B, (x * y) ∈ C) ∧
  (∀ x ∈ B, ∀ y ∈ C, (x * y) ∈ A) ∧
  (A ∪ B ∪ C = set.univ \ {0}) ∧
  (∀ x, x ∈ A → x ∉ B ∧ x ∉ C)

-- Part (b): Showing all positive rational cubes are in A
def all_positive_rational_cubes_in_A (A : set ℚ) : Prop :=
  ∀ (q : ℚ), (∃ (n : ℚ), q = n^3) → q ∈ A

-- Part (c): No n and n + 1 in A
def no_consecutive_integers_in_A_up_to_34 (A : set ℚ) : Prop :=
  ∀ n : ℕ, n ≤ 34 → ¬ (n ∈ A ∧ (n + 1) ∈ A)

-- Combining the statements (a) (b) and (c) into one proof problem
theorem partition_Q_plus_problem : 
  ∃ (A B C : set ℚ), 
    partition_Q_plus A B C ∧
    all_positive_rational_cubes_in_A A ∧
    no_consecutive_integers_in_A_up_to_34 A :=
sorry

end partition_Q_plus_problem_l533_533867


namespace cross_shape_rectangle_count_l533_533510

def original_side_length := 30
def smallest_square_side_length := 1
def cut_corner_length := 10
def N : ℕ := sorry  -- total number of rectangles in the resultant graph paper
def result : ℕ := 14413

theorem cross_shape_rectangle_count :
  (1/10 : ℚ) * N = result := 
sorry

end cross_shape_rectangle_count_l533_533510


namespace positive_difference_mean_median_l533_533450

noncomputable def mean (values : List ℝ) : ℝ :=
  values.sum / values.length

noncomputable def median (values : List ℝ) : ℝ :=
  let sorted := values.qsort (≤)
  if sorted.length % 2 = 1 then
    sorted.nthLe (sorted.length / 2) sorry
  else
    (sorted.nthLe (sorted.length / 2 - 1) sorry + sorted.nthLe (sorted.length / 2) sorry) / 2

theorem positive_difference_mean_median 
  (h : [170, 120, 140, 305, 200, 150] = values) :
  Nat.abs (Int.ofNat ⟨mean values - median values⟩) = 21 :=
by
  sorry

end positive_difference_mean_median_l533_533450


namespace max_edges_no_quadrilateral_sg_l533_533612

-- Define a simple graph with 8 vertices and no quadrilateral
structure SimpleGraph (V : Type) :=
(adj : V → V → Prop)
(sym : symmetric adj)
(loopless : irreflexive adj)

def no_quadrilateral (G : SimpleGraph (Fin 8)) :=
  ∀ a b c d : Fin 8, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d → 
  ¬(G.adj a b ∧ G.adj b c ∧ G.adj c d ∧ G.adj d a)

theorem max_edges_no_quadrilateral_sg (G : SimpleGraph (Fin 8)) (h_no_quad : no_quadrilateral G) :
  (∃ E : Finset (Fin 8 × Fin 8), E.card ≤ 11 ∧ ∀ x y, (x, y) ∈ E → G.adj x y) :=
sorry

end max_edges_no_quadrilateral_sg_l533_533612


namespace convert_to_scientific_notation_l533_533123

-- Define the number to be converted
def original_number : ℕ := 58000000000

-- Define the expected scientific notation
def scientific_notation (n : ℕ) : Prop :=
  n = 5.8 * real.pow 10 10

-- State the main theorem
theorem convert_to_scientific_notation : scientific_notation original_number :=
by
  sorry

end convert_to_scientific_notation_l533_533123


namespace two_b_squared_eq_a_squared_plus_c_squared_l533_533170

theorem two_b_squared_eq_a_squared_plus_c_squared (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 2 / (c + a)) : 
  2 * b^2 = a^2 + c^2 := 
sorry

end two_b_squared_eq_a_squared_plus_c_squared_l533_533170


namespace ratio_shaded_area_to_circle_area_l533_533794

noncomputable def segmentAB : ℝ := 10
noncomputable def segmentAC : ℝ := 6
noncomputable def segmentCB : ℝ := 4

noncomputable def radius_big_semicircle : ℝ := segmentAB / 2
noncomputable def radius_mid_semicircle : ℝ := segmentAC / 2
noncomputable def radius_small_semicircle : ℝ := segmentCB / 2

noncomputable def area_big_semicircle : ℝ := (1 / 2) * real.pi * radius_big_semicircle ^ 2
noncomputable def area_mid_semicircle : ℝ := (1 / 2) * real.pi * radius_mid_semicircle ^ 2
noncomputable def area_small_semicircle : ℝ := (1 / 2) * real.pi * radius_small_semicircle ^ 2

noncomputable def total_shaded_area : ℝ := 
  area_big_semicircle - area_mid_semicircle - area_small_semicircle

noncomputable def area_circle_diameterCB : ℝ := real.pi * radius_small_semicircle ^ 2

theorem ratio_shaded_area_to_circle_area : 
  (total_shaded_area / area_circle_diameterCB) = 1.5 :=
by
  sorry

end ratio_shaded_area_to_circle_area_l533_533794


namespace complex_division_correct_l533_533653

noncomputable def complex_number_example : Prop :=
  let i := Complex.I in
  Complex.div (2 + 4 * i) (1 + i) = 3 + i

theorem complex_division_correct : complex_number_example :=
by
  sorry

end complex_division_correct_l533_533653


namespace angle_EHC_in_parallelogram_l533_533706

theorem angle_EHC_in_parallelogram 
  (EFGH : Type) 
  [parallelogram E F G H : EFGH]
  (EFG_eq_4FGH : ∠EFG = 4 * ∠FGH) :
  ∠EHC = 144 :=
by
  -- Proof steps would go here
  sorry

end angle_EHC_in_parallelogram_l533_533706


namespace sum_of_square_areas_eq_144_l533_533558

noncomputable def right_triangle_area_sum (A C F : Type) [MetricSpace A] [MetricSpace C] [MetricSpace F]
  (right_angle : ∃ (ACF : ℕ), ∀ (A C F : ℕ), A = C ∧ F = A + 90)
  (CF_eq : ∃ (CF : ℕ), CF = 12) : Prop :=
  ∃ (AC AF : ℕ), (AC^2 + AF^2) = 144

-- The problem statement rewritten as a Lean 4 theorem
theorem sum_of_square_areas_eq_144 (A C F : Type) [MetricSpace A] [MetricSpace C] [MetricSpace F]
  (right_angle : ∃ (ACF : ℕ), ∀ (A C F : ℕ), right_angle)
  (CF_eq : CF =  12) : (AC^2 + AF^2) = 144 :=
begin
  sorry
end

end sum_of_square_areas_eq_144_l533_533558


namespace triangle_angle_and_side_relation_l533_533424

theorem triangle_angle_and_side_relation 
  (A B C : Type) [metric_space A] 
  [metric_space B] [metric_space C] 
  (triangle : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (angle_A_is_twice_angle_B : ∠A = 2 * ∠B) 
  (side_AB side_AC side_BC : ℝ) 
  (h1 : side_AB = dist A B) 
  (h2 : side_AC = dist A C) 
  (h3 : side_BC = dist B C) 
  : side_BC^2 = (side_AC + side_AB) * side_AC := 
  sorry

end triangle_angle_and_side_relation_l533_533424


namespace purchasing_methods_count_is_seven_l533_533763

noncomputable def countPurchasingMethods : ℕ :=
  let validMethods := {xy : ℕ × ℕ | 60 * xy.1 + 70 * xy.2 ≤ 500 ∧ xy.1 ≥ 3 ∧ xy.2 ≥ 2}
  validMethods.toFinset.card

theorem purchasing_methods_count_is_seven :
  countPurchasingMethods = 7 :=
  sorry

end purchasing_methods_count_is_seven_l533_533763


namespace problem1_problem2_problem3_problem4_l533_533221

theorem problem1 : 
  (3 / 5 : ℚ) - ((2 / 15) + (1 / 3)) = (2 / 15) := 
  by 
  sorry

theorem problem2 : 
  (-2 : ℤ) - 12 * ((1 / 3 : ℚ) - (1 / 4 : ℚ) + (1 / 2 : ℚ)) = -8 := 
  by 
  sorry

theorem problem3 : 
  (2 : ℤ) * (-3) ^ 2 - (6 / (-2) : ℚ) * (-1 / 3) = 17 := 
  by 
  sorry

theorem problem4 : 
  (-1 ^ 4 : ℤ) + ((abs (2 ^ 3 - 10)) : ℤ) - ((-3 : ℤ) / (-1) ^ 2019) = -2 := 
  by 
  sorry

end problem1_problem2_problem3_problem4_l533_533221


namespace problem_b_is_proposition_l533_533497

def is_proposition (s : String) : Prop :=
  s = "sin 45° = 1" ∨ s = "x^2 + 2x - 1 > 0"

theorem problem_b_is_proposition : is_proposition "sin 45° = 1" :=
by
  -- insert proof steps to establish that "sin 45° = 1" is a proposition
  sorry

end problem_b_is_proposition_l533_533497


namespace area_enclosed_shape_is_correct_l533_533427

noncomputable def area_enclosed_by_curves : ℝ :=
  ∫ x in 0..1, (x^2 - x^3)

theorem area_enclosed_shape_is_correct :
  area_enclosed_by_curves = 1 / 12 :=
by
  sorry

end area_enclosed_shape_is_correct_l533_533427


namespace grade12_sample_size_correct_l533_533183

-- Given conditions
def grade10_students : ℕ := 1200
def grade11_students : ℕ := 900
def grade12_students : ℕ := 1500
def total_sample_size : ℕ := 720
def total_students : ℕ := grade10_students + grade11_students + grade12_students

-- Stratified sampling calculation
def fraction_grade12 : ℚ := grade12_students / total_students
def number_grade12_in_sample : ℚ := fraction_grade12 * total_sample_size

-- Main theorem
theorem grade12_sample_size_correct :
  number_grade12_in_sample = 300 := by
  sorry

end grade12_sample_size_correct_l533_533183


namespace surface_area_of_second_cube_volume_of_third_sphere_l533_533195

noncomputable def surface_area_second_cube (surface_area_first_cube : ℝ) : ℝ := 
  let side_length_first_cube := real.sqrt (surface_area_first_cube / 6)
  let diagonal_second_cube := side_length_first_cube
  let side_length_second_cube := diagonal_second_cube / real.sqrt 3
  6 * (side_length_second_cube ^ 2)

noncomputable def volume_third_sphere (surface_area_first_cube : ℝ) : ℝ := 
  let side_length_first_cube := real.sqrt (surface_area_first_cube / 6)
  let diagonal_second_cube := side_length_first_cube
  let side_length_second_cube := diagonal_second_cube / real.sqrt 3
  let radius_third_sphere := side_length_second_cube / 2
  (4 / 3) * real.pi * (radius_third_sphere ^ 3)

theorem surface_area_of_second_cube 
  (h1 : surface_area_second_cube 54 = 18) : surface_area_second_cube 54 = 18 := 
by {
  exact h1
  }

theorem volume_of_third_sphere 
  (h2 : volume_third_sphere 54 = (real.sqrt 3 * real.pi) / 2) : 
  volume_third_sphere 54 = (real.sqrt 3 * real.pi) / 2 := 
by {
  exact h2
  }

end surface_area_of_second_cube_volume_of_third_sphere_l533_533195


namespace no_valid_prime_pairs_l533_533622

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_valid_prime_pairs :
  ∀ x y : ℕ, is_prime x → is_prime y → y < x → x ≤ 200 → (x % y = 0) → ((x +1) % (y +1) = 0) → false :=
by
  sorry

end no_valid_prime_pairs_l533_533622


namespace franks_daily_reading_l533_533623

-- Define the conditions
def total_pages : ℕ := 612
def days_to_finish : ℕ := 6

-- State the theorem we want to prove
theorem franks_daily_reading : (total_pages / days_to_finish) = 102 :=
by
  sorry

end franks_daily_reading_l533_533623


namespace f_even_and_period_l533_533754

/-- Define the function f(x) -/
def f (x : ℝ) : ℝ := Math.sin (3 * x - (Real.pi / 2))

/-- The mathematical problem: Prove that f(x) is an even function with the smallest positive period of 2π/3 -/
theorem f_even_and_period :
  (∀ x : ℝ, f(-x) = f(x)) ∧ (∃ T > 0, T = 2 * Real.pi / 3 ∧ ∀ x : ℝ, f (x + T) = f x) :=
sorry

end f_even_and_period_l533_533754


namespace circle_equation_tangent_line_equation_l533_533169

-- Problem 1: Proof of the circle equation
theorem circle_equation (h_pass: (2 - 1)^2 + ((-1) + 2)^2 = 2) (h_tangent: abs (1 - 2*(-1)-1)/(sqrt 2)) (h_center: -1 = -2 * 1) :
  forall x y, ((x - 1)^2 + (y + 2)^2 = 2) :=
sorry

-- Problem 2: Proof of the line equation from tangents
theorem tangent_line_equation :
  let A := (2, -2)
  let C := (0, 2)
  let R := 2
  ∀ (x1 y1 x2 y2 : ℝ), 
    let T1 := (x1, y1)
    let T2 := (x2, y2)
    (x1^2 + (y1-2)^2 = 4 ∧  x2^2 + (y2-2)^2 = 4 ∧
    T1.dist A = sqrt ((C.1 - T1.1)^2 + (C.2 - T1.2)^2) ∧
    T2.dist A = sqrt ((C.1 - T2.1)^2 + (C.2 - T2.2)^2) ∧
    T1.1 * T2.1 + (T1.2 - 2) * (T2.2 - 2) = 4 ∧
    2 * T1.1 - 4 * (T1.2 - 2) = 4 ∧
    2 * T1.1 - 4 * (T2.2 - 2) = 4) → 
  (x - 2*y + 2 = 0) :=
sorry

end circle_equation_tangent_line_equation_l533_533169


namespace harmonic_series_induction_number_of_terms_added_l533_533124

-- Define the hypotheses and goals for the mathematical induction proof
theorem harmonic_series_induction (n : ℕ) (h₁ : 2 ≤ n) : 
  (∑ i in finset.range (2^n), 1 / (i + 1) < n) := by
  sorry

-- Define the hypotheses and goals for the number of terms added
theorem number_of_terms_added (k : ℕ) (h₁ : 1 ≤ k) : 
  (2^(k+1) - 1 - 2^k + 1 = 2^k) := by
  sorry

end harmonic_series_induction_number_of_terms_added_l533_533124


namespace no_real_roots_of_quadratic_l533_533260

theorem no_real_roots_of_quadratic 
  (a b c : ℝ) 
  (h1 : b - a + c > 0) 
  (h2 : b + a - c > 0) 
  (h3 : b - a - c < 0) 
  (h4 : b + a + c > 0) 
  (x : ℝ) : ¬ ∃ x : ℝ, a^2 * x^2 + (b^2 - a^2 - c^2) * x + c^2 = 0 := 
by
  sorry

end no_real_roots_of_quadratic_l533_533260


namespace probability_even_number_four_dice_l533_533140

noncomputable def probability_same_even_number_facing_up_four_dice : ℚ :=
  1 / 432

theorem probability_even_number_four_dice :
  ∀ (P : ℚ), 
    (∀ (DieResult1 DieResult2 DieResult3 DieResult4 : ℕ), 
      DieResult1 ∈ {2, 4, 6} ∧ DieResult2 = DieResult1 ∧ DieResult3 = DieResult1 ∧ DieResult4 = DieResult1 → 
        P = 1 / 432) →
      P = probability_same_even_number_facing_up_four_dice :=
by
  intros P h
  exact sorry

end probability_even_number_four_dice_l533_533140


namespace simplify_fraction_l533_533579

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 4) (h2 : a ≠ -4) : 
  (2 * a / (a^2 - 16) - 1 / (a - 4) = 1 / (a + 4)) := 
by 
  sorry 

end simplify_fraction_l533_533579


namespace number_of_correct_propositions_l533_533992

variables {α β : Plane} {l m : Line}

-- Given conditions
axiom line_perpendicular_to_plane (l : Line) (α : Plane) : l ⊥ α
axiom line_in_plane (m : Line) (β : Plane) : m ∈ β

-- Propositions
def prop1 (α β : Plane) (l m : Line) : Prop := (α ∥ β) → (l ⊥ m)
def prop2 (α β : Plane) (l m : Line) : Prop := (α ⊥ β) → (l ∥ m)
def prop3 (α β : Plane) (l m : Line) : Prop := (l ∥ m) → (α ⊥ β)
def prop4 (α β : Plane) (l m : Line) : Prop := (l ⊥ m) → (α ∥ β)

-- Proof problem statement
theorem number_of_correct_propositions : 
  (∃ (count : ℕ), count = 2 ∧ (
    (prop1 α β l m) ∧ 
    ¬(prop2 α β l m) ∧ 
    (prop3 α β l m) ∧ 
    ¬(prop4 α β l m)
  )) :=
sorry

end number_of_correct_propositions_l533_533992


namespace log_value_l533_533382

-- Definitions for the conditions

variables (z1 z2 : ℂ)

def condition1 : Prop := abs z1 = 3
def condition2 : Prop := abs (z1 + z2) = 3
def condition3 : Prop := abs (z1 - z2) = 3 * real.sqrt 3

-- The proof problem statement
theorem log_value (h1 : condition1 z1) (h2 : condition2 z1 z2) (h3 : condition3 z1 z2) :
  real.logb 3 (abs ((z1 * conj z2)^2000 + (conj z1 * z2)^2000)) = 4000 :=
sorry

end log_value_l533_533382


namespace average_infections_per_round_infections_after_three_rounds_l533_533961

-- Define the average number of infections per round such that the total after two rounds is 36 and x > 0
theorem average_infections_per_round :
  ∃ x : ℤ, (1 + x)^2 = 36 ∧ x > 0 :=
by
  sorry

-- Given x = 5, prove that the total number of infections after three rounds exceeds 200
theorem infections_after_three_rounds (x : ℤ) (H : x = 5) :
  (1 + x)^3 > 200 :=
by
  sorry

end average_infections_per_round_infections_after_three_rounds_l533_533961


namespace lulu_cash_left_l533_533391

-- Define the initial amount
def initial_amount : ℕ := 65

-- Define the amount spent on ice cream
def spent_on_ice_cream : ℕ := 5

-- Define the amount spent on a t-shirt
def spent_on_tshirt (remaining_after_ice_cream : ℕ) : ℕ := remaining_after_ice_cream / 2

-- Define the amount deposited in the bank
def deposited_in_bank (remaining_after_tshirt : ℕ) : ℕ := remaining_after_tshirt / 5

-- Define the remaining cash after all transactions
def remaining_cash (initial : ℕ) (spent_ice_cream : ℕ) (spent_tshirt: ℕ) (deposited: ℕ) :ℕ :=
  initial - spent_ice_cream - spent_tshirt - deposited

-- Theorem statement to prove
theorem lulu_cash_left : remaining_cash initial_amount spent_on_ice_cream (spent_on_tshirt (initial_amount - spent_on_ice_cream)) 
(deposited_in_bank ((initial_amount - spent_on_ice_cream) - (spent_on_tshirt (initial_amount - spent_on_ice_cream)))) = 24 :=
by
  sorry

end lulu_cash_left_l533_533391


namespace Lisa_goal_l533_533561

theorem Lisa_goal
  (total_quizzes : ℕ)
  (goal_percentage : ℕ)
  (achieved_A_initial : ℕ)
  (initial_quizzes : ℕ)
  (remaining_quizzes : ℕ)
  (min_quizzes_A : ℕ)
  (required_A_remaining : ℕ) :
  total_quizzes = 60 →
  goal_percentage = 85 →
  achieved_A_initial = 28 →
  initial_quizzes = 40 →
  remaining_quizzes = total_quizzes - initial_quizzes →
  min_quizzes_A = (goal_percentage * total_quizzes) / 100 →
  required_A_remaining = min_quizzes_A - achieved_A_initial →
  required_A_remaining <= remaining_quizzes →
  required_A_remaining = remaining_quizzes →
  true :=
by sorry

lemma determine_Lisa_goal :
  Lisa_goal 60 85 28 40 20 51 23 :=
by sorry

end Lisa_goal_l533_533561


namespace infinitely_many_n_pairwise_coprime_l533_533512

open Nat

theorem infinitely_many_n_pairwise_coprime (a b c : ℤ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ᶠ n in at_top, ∀ i j, pairwise_coprime (finset.map (λ x, x + n) (finset {a, b, c})) i j :=
sorry

end infinitely_many_n_pairwise_coprime_l533_533512


namespace john_buys_spools_l533_533368

theorem john_buys_spools (spool_length necklace_length : ℕ) 
  (necklaces : ℕ) 
  (total_length := necklaces * necklace_length) 
  (spools := total_length / spool_length) :
  spool_length = 20 → 
  necklace_length = 4 → 
  necklaces = 15 → 
  spools = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end john_buys_spools_l533_533368


namespace solve_system_of_inequalities_l533_533787

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x + 3 ≤ x + 2) ∧ ((x + 1) / 3 > x - 1) → x ≤ -1 := by
  sorry

end solve_system_of_inequalities_l533_533787


namespace prove_common_difference_l533_533008

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533008


namespace evaluate_expression_l533_533958

theorem evaluate_expression (a b : ℕ) (h_a : a = 15) (h_b : b = 7) :
  (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 210 :=
by 
  rw [h_a, h_b]
  sorry

end evaluate_expression_l533_533958


namespace area_of_rectangle_of_roots_l533_533433

open Complex Polynomial

noncomputable def polynomial : Polynomial ℂ :=
  (Polynomial.mk [(-5 - 20i), (10 - 2i), (5 + 5i), 4i, 1])

theorem area_of_rectangle_of_roots :
  let roots := polynomial.roots.toFinset in
  (roots.card = 4 ∧
   isRectangle roots ∧
   ∀ z ∈ roots, polynomial.eval z = 0) →
   area_of_rectangle roots = sqrt 634 := sorry

end area_of_rectangle_of_roots_l533_533433


namespace selection_methods_l533_533985

theorem selection_methods (boys girls : ℕ) (hboys : boys = 4) (hgirls : girls = 3) :
  ∃ n, n = 30 ∧ 
  (∑ b in finset.range (boys + 1), ∑ g in finset.range (girls + 1), if b + g = 3 ∧ b > 0 ∧ g > 0 then nat.choose boys b * nat.choose girls g else 0) = n := by
simp [hboys, hgirls]
sorry

end selection_methods_l533_533985


namespace book_loss_percentage_l533_533318

theorem book_loss_percentage (CP SP_profit SP_loss : ℝ) (L : ℝ) 
  (h1 : CP = 50) 
  (h2 : SP_profit = CP + 0.09 * CP) 
  (h3 : SP_loss = CP - L / 100 * CP) 
  (h4 : SP_profit - SP_loss = 9) : 
  L = 9 :=
by
  sorry

end book_loss_percentage_l533_533318


namespace fraction_savings_on_makeup_l533_533729

theorem fraction_savings_on_makeup (savings : ℝ) (sweater_cost : ℝ) (makeup_cost : ℝ) (h_savings : savings = 80) (h_sweater : sweater_cost = 20) (h_makeup : makeup_cost = savings - sweater_cost) : makeup_cost / savings = 3 / 4 := by
  sorry

end fraction_savings_on_makeup_l533_533729


namespace rectangle_perimeter_area_change_l533_533325

theorem rectangle_perimeter_area_change
  (a b : ℝ) :
  let new_length := 1.1 * a,
      new_width := 0.9 * b,
      original_perimeter := 2 * (a + b),
      new_perimeter := 2 * (new_length + new_width),
      original_area := a * b,
      new_area := new_length * new_width in
  new_perimeter > original_perimeter ∧ new_area < original_area :=
by
  sorry

end rectangle_perimeter_area_change_l533_533325


namespace convert_to_slope_intercept_find_slope_intercept_l533_533538

theorem convert_to_slope_intercept (x y : ℝ) :
  (\begin{pmatrix} 3 \\ -4 \end{pmatrix} : ℝ × ℝ) • 
  (⟨x, y⟩ - ⟨2, 8⟩) = 0 → 
  y = (3 / 4) * x + 6.5 := 
by 
  sorry

theorem find_slope_intercept (x y : ℝ) :
  (\begin{pmatrix} 3 \\ -4 \end{pmatrix} : ℝ × ℝ) • 
  (⟨x, y⟩ - ⟨2, 8⟩) = 0 → 
  (3 / 4, 6.5) = (3 / 4, 26 / 4) := 
by 
  sorry

end convert_to_slope_intercept_find_slope_intercept_l533_533538


namespace chess_tournament_games_l533_533336

theorem chess_tournament_games (n : ℕ) (h : n = 15) :
  nat.choose n 2 = 105 := by
  rw h
  exact nat.choose_self_sub 15 2

end chess_tournament_games_l533_533336


namespace data_a_value_l533_533084

theorem data_a_value (a b c : ℕ) (h1 : a + b = c) (h2 : b = 3 * a) (h3 : a + b + c = 96) : a = 12 :=
by
  sorry

end data_a_value_l533_533084


namespace arithmetic_sequence_inequality_l533_533272

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1

noncomputable def b_n (n : ℕ) : ℝ := 1 / (a_n n ^ 2 - 1)

noncomputable def T_n (n : ℕ) : ℝ := ∑ i in finset.range n, b_n (i + 1)

theorem arithmetic_sequence_inequality (n : ℕ) : T_n n < 1 / 4 :=
by {
  -- Proof to be provided
  sorry
}

end arithmetic_sequence_inequality_l533_533272


namespace arithmetic_common_difference_l533_533022

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533022


namespace sum_of_real_roots_l533_533615

theorem sum_of_real_roots :
  let f (x : Real) := sin (π * (x^2 - x + 1)) 
  let g (x : Real) := sin (π * (x - 1))
  sum_roots : Real :=
  f x = g x ∧ 0 ≤ x ∧ x ≤ 2 
    sum_roots = 3 + Real.sqrt 3 :=
by
  sorry

end sum_of_real_roots_l533_533615


namespace gazprom_rnd_costs_calc_l533_533932

theorem gazprom_rnd_costs_calc (R_D_t ΔAPL_t1 : ℝ) (h1 : R_D_t = 3157.61) (h2 : ΔAPL_t1 = 0.69) :
  R_D_t / ΔAPL_t1 = 4576 :=
by
  sorry

end gazprom_rnd_costs_calc_l533_533932


namespace find_m_minus_n_l533_533327

theorem find_m_minus_n (m n : ℝ) (h1 : -5 + 1 = m) (h2 : -5 * 1 = n) : m - n = 1 :=
sorry

end find_m_minus_n_l533_533327


namespace smallest_k_inequality_l533_533971

theorem smallest_k_inequality :
  ∃ k : ℝ, (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ k * (x^4 + y^4 + z^4)) ∧ k = 3 :=
by
  let k := 3
  use k
  split
  { intro x y z
    sorry },
  { simp }

end smallest_k_inequality_l533_533971


namespace base_n_quadratic_solution_l533_533090

noncomputable def base_n_b_rep (n : ℕ) (a : ℕ) (b : ℕ) : Prop :=
  n > 9 ∧ a = n + 9 ∧ (∃ (m : ℕ), m = 9 ∧ b = n * m) ∧ 
  (let bn := 90 in b = bn * n / 10)

theorem base_n_quadratic_solution (n a b : ℕ) (h : base_n_b_rep n a b) : 
  b = 9 * n ∧ (b = 9 * n) := by 
  sorry

end base_n_quadratic_solution_l533_533090


namespace number_of_sine_law_numbers_equals_l533_533534

def satisfies_sine_law (a b c d e : ℕ) : Prop :=
  a < b ∧ b > c ∧ c > d ∧ d < e ∧ a > d ∧ b > e

def count_sine_law_numbers : ℕ :=
  Nat.card { x // ∃ (a b c d e : ℕ), x = (a * 10000 + b * 1000 + c * 100 + d * 10 + e) ∧ satisfies_sine_law a b c d e }

theorem number_of_sine_law_numbers_equals :
  count_sine_law_numbers = 2892 :=
sorry

end number_of_sine_law_numbers_equals_l533_533534


namespace common_difference_l533_533001

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l533_533001


namespace seq_formula_correct_l533_533088

/-- Define the sequence 4, 3, 2, 1, ... -/
def seq : ℕ → ℕ
| 1 := 4
| 2 := 3
| 3 := 2
| 4 := 1
| _ := 0  -- Assuming the sequence thereafter is not relevant for this proof.

/-- Define the formula a_n = 5 - n -/
def a_n (n : ℕ) : ℕ := 5 - n

/-- Theorem that the formula a_n = 5 - n matches the sequence 4, 3, 2, 1, ... -/
theorem seq_formula_correct : ∀ n, n ∈ {1, 2, 3, 4} → seq n = a_n n :=
by
  intros n h
  cases h
  repeat { cases n; sorry }

end seq_formula_correct_l533_533088


namespace product_of_two_numbers_less_than_the_smaller_of_the_two_factors_l533_533583

theorem product_of_two_numbers_less_than_the_smaller_of_the_two_factors
    (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) : 
  a * b < min a b := 
sorry

end product_of_two_numbers_less_than_the_smaller_of_the_two_factors_l533_533583


namespace laura_remaining_pay_is_correct_l533_533371

-- Define constants for the problem conditions
def hourly_wage : ℕ := 10
def hours_per_day : ℕ := 8
def days_worked : ℕ := 10
def percentage_spent_on_food_and_clothing : ℚ := 0.25
def rent_paid : ℕ := 350

-- Define total earnings
def total_earnings : ℕ := hours_per_day * days_worked * hourly_wage

-- Amount spent on food and clothing
def amount_spent_on_food_and_clothing : ℕ := (percentage_spent_on_food_and_clothing * total_earnings).to_nat

-- Remaining amount after food and clothing expenses
def remaining_after_food_and_clothing : ℕ := total_earnings - amount_spent_on_food_and_clothing

-- Remaining amount after paying rent
def remaining_after_all_expenses : ℕ := remaining_after_food_and_clothing - rent_paid

-- Lean 4 statement to be proven
theorem laura_remaining_pay_is_correct : remaining_after_all_expenses = 250 := by
  sorry

end laura_remaining_pay_is_correct_l533_533371


namespace roots_of_quadratic_function_l533_533993

variable (a b x : ℝ)

theorem roots_of_quadratic_function (h : a + b = 0) : (b * x * x + a * x = 0) → (x = 0 ∨ x = 1) :=
by {sorry}

end roots_of_quadratic_function_l533_533993


namespace distance_fall_l533_533660

-- Given conditions as definitions
def velocity (g : ℝ) (t : ℝ) := g * t

-- The theorem stating the relationship between time t0 and distance S
theorem distance_fall (g : ℝ) (t0 : ℝ) : 
  (∫ t in (0 : ℝ)..t0, velocity g t) = (1/2) * g * t0^2 :=
by 
  sorry

end distance_fall_l533_533660


namespace sum_of_interior_angles_of_pentagon_l533_533108

theorem sum_of_interior_angles_of_pentagon :
  let n := 5
  let angleSum := 180 * (n - 2)
  angleSum = 540 :=
by
  sorry

end sum_of_interior_angles_of_pentagon_l533_533108


namespace magnitude_z_l533_533661

-- Defining the complex number z
def z : ℂ := (2 - complex.I) / (1 + 2 * complex.I)

-- The theorem stating that the magnitude of z is 1
theorem magnitude_z : complex.abs z = 1 := 
by 
  -- the proof steps are omitted based on instructions; adding sorry
  sorry

end magnitude_z_l533_533661


namespace find_W_l533_533814

def digit_sum_eq (X Y Z W : ℕ) : Prop := X * 10 + Y + Z * 10 + X = W * 10 + X
def digit_diff_eq (X Y Z : ℕ) : Prop := X * 10 + Y - (Z * 10 + X) = X
def is_digit (n : ℕ) : Prop := n < 10

theorem find_W (X Y Z W : ℕ) (h1 : digit_sum_eq X Y Z W) (h2 : digit_diff_eq X Y Z) 
  (hX : is_digit X) (hY : is_digit Y) (hZ : is_digit Z) (hW : is_digit W) : W = 0 := 
sorry

end find_W_l533_533814


namespace recipe_sugar_amount_l533_533051

theorem recipe_sugar_amount (F_total F_added F_additional F_needed S : ℕ)
  (h1 : F_total = 9)
  (h2 : F_added = 2)
  (h3 : F_additional = S + 1)
  (h4 : F_needed = F_total - F_added)
  (h5 : F_needed = F_additional) :
  S = 6 := 
sorry

end recipe_sugar_amount_l533_533051


namespace prove_common_difference_l533_533003

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533003


namespace find_days_l533_533856

-- Define the work rates of A and B
def work_rate_B := (1:ℝ) / 21
def work_rate_A := 2 * work_rate_B

-- Define the combined work rate of A and B
def combined_work_rate := work_rate_A + work_rate_B

-- Define the number of days taken to complete the work together
def days_taken := (1:ℝ) / combined_work_rate

theorem find_days (h1: work_rate_A = 2 * work_rate_B) (h2: work_rate_B = (1:ℝ)/21) : days_taken = 7 := by
  -- Proof to be filled in
  sorry

end find_days_l533_533856


namespace probability_B_receives_once_after_three_passes_l533_533796

-- Definitions
def ball_passing_sequence : Type := List (List Char)
def total_sequences (n : Nat) (choices : Nat) := choices^n
def valid_sequences (sequences : ball_passing_sequence) (n : Nat) : ball_passing_sequence :=
  (sequences.filter (λ seq => seq.count ('B') = 1))

-- The main theorem to prove
theorem probability_B_receives_once_after_three_passes :
  let sequences := (['A', 'B', 'C', 'D']).bind (λ x => ['A', 'B', 'C', 'D']) ++ (['A', 'B', 'C', 'D'])
  let relevant_sequences := valid_sequences sequences 3
  (relevant_sequences.length : ℚ) / (total_sequences 3 3 : ℚ) = 16 / 27 :=
sorry

end probability_B_receives_once_after_three_passes_l533_533796


namespace measure_of_angle_C_l533_533692

variable (A B C : Real)

theorem measure_of_angle_C (h1 : 4 * Real.sin A + 2 * Real.cos B = 4) 
                           (h2 : (1/2) * Real.sin B + Real.cos A = Real.sqrt 3 / 2) :
                           C = Real.pi / 6 :=
by
  sorry

end measure_of_angle_C_l533_533692


namespace largest_angle_of_triangle_l533_533912

theorem largest_angle_of_triangle
  (t : ℝ) (h_t : t > 1)
  (a b c : ℝ)
  (h_a : a = t^2 + t + 1)
  (h_b : b = t^2 - 1)
  (h_c : c = 2t + 1)
  (triangle_inequality : b + c > a ∧ a + c > b ∧ a + b > c) :
  ∃ A : ℝ, A = 120 ∧ ∃ (cos_A : ℝ), cos_A = -1 / 2 :=
  sorry

end largest_angle_of_triangle_l533_533912


namespace angle_ABC_in_hexagon_with_equilateral_triangle_shared_side_l533_533076

theorem angle_ABC_in_hexagon_with_equilateral_triangle_shared_side (A B C G : Type)
  (hex : geometric_configuration.hexagon) (tri : geometric_configuration.equilateral_triangle)
  (common_side: geometric_configuration.common_side hex tri B C):
  ∠ABC = 120 :=
sorry

end angle_ABC_in_hexagon_with_equilateral_triangle_shared_side_l533_533076


namespace sum_of_excluded_solutions_l533_533373

noncomputable def P : ℚ := 3
noncomputable def Q : ℚ := 5 / 3
noncomputable def R : ℚ := 25 / 3

theorem sum_of_excluded_solutions :
    (P = 3) ∧
    (Q = 5 / 3) ∧
    (R = 25 / 3) ∧
    (∀ x, (x ≠ -R ∧ x ≠ -10) →
    ((x + Q) * (P * x + 50) / ((x + R) * (x + 10)) = 3)) →
    (-R + -10 = -55 / 3) :=
by
  sorry

end sum_of_excluded_solutions_l533_533373


namespace solution_set_inequality_l533_533267

section SolutionSetInequality

variable (f : ℝ → ℝ)
variable [Differentiable ℝ f]
variable h1 : ∀ x : ℝ, f(x) + (deriv^[2] f) x > 1
variable h2 : f 1 = 0

theorem solution_set_inequality :
  { x : ℝ | f(x) - 1 + 1 / (Real.exp (x - 1)) ≤ 0 } = Set.Iic 1 :=
by
  sorry

end SolutionSetInequality

end solution_set_inequality_l533_533267


namespace children_ages_correct_l533_533185

-- Defining the given conditions as Lean structures and propositions
variables {parent_age : ℕ} {first_child_age second_child_age third_child_age : ℕ}

def conditions (parent_age first_child_age second_child_age third_child_age : ℕ) : Prop :=
  let total_family_age_when_first_child_born := 2 * parent_age in
  let total_family_age_one_year_ago := 2 * (parent_age + 7) + (first_child_age - 1) + (second_child_age - 1) + (14 - (first_child_age - 1) - (second_child_age - 1)) in
  let total_age_of_children_now := first_child_age + second_child_age + third_child_age = 14 in
  total_family_age_when_first_child_born = 45 ∧ total_family_age_one_year_ago = 70 ∧ total_age_of_children_now ∧ third_child_age = 1 ∧ first_child_age = 8 ∧ second_child_age = 5

theorem children_ages_correct :
  ∃ parent_age first_child_age second_child_age third_child_age, 
  conditions parent_age first_child_age second_child_age third_child_age :=
begin
  -- These values satisfy the conditions as per the problem's conclusion
  use [21, 8, 5, 1],
  rw conditions,
  simp,
  split,
  {
    simp,
    rfl,
  },
  {
    simp,
    rfl,
  },
  {
    simp,
  }
end

end children_ages_correct_l533_533185


namespace shopkeeper_bananas_l533_533907

theorem shopkeeper_bananas :
  ∃ B : ℕ,
    let good_oranges := 0.85 * 600 in
    let total_fruits := 600 + B in
    let good_fruits := good_oranges + 0.94 * B in
    (good_fruits / total_fruits) = 0.886 ∧ B = 400 :=
by
  sorry

end shopkeeper_bananas_l533_533907


namespace ordered_pair_f_0_f_1_l533_533892

noncomputable def f : ℤ → ℤ := sorry

theorem ordered_pair_f_0_f_1 :
  (f 0, f 1) = (-1, 1) :=
  have h1 : ∀ x : ℤ, f (x + 4) - f x = 8 * x + 20, from sorry,
  have h2 : ∀ x : ℤ, f (x^2 - 1) = (f x - x)^2 + x^2 - 2, from sorry,
  sorry

end ordered_pair_f_0_f_1_l533_533892


namespace unique_integer_solution_l533_533041

theorem unique_integer_solution (a b : ℤ) : 
  ∀ x₁ x₂ : ℤ, (x₁ - a) * (x₁ - b) * (x₁ - 3) + 1 = 0 ∧ (x₂ - a) * (x₂ - b) * (x₂ - 3) + 1 = 0 → x₁ = x₂ :=
by
  sorry

end unique_integer_solution_l533_533041


namespace prove_common_difference_l533_533010

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533010


namespace ratio_area_proof_l533_533880

noncomputable def ratio_area_circle_hexagon (r : ℝ) : ℝ :=
  let area_smaller_circle := π * r ^ 2
  let side_length_larger_hexagon := (2 * r) / (Real.sqrt 3)
  let area_larger_hexagon := (3 * Real.sqrt 3 / 2) * (side_length_larger_hexagon ^ 2)
  area_smaller_circle / area_larger_hexagon

theorem ratio_area_proof (r : ℝ) (h : r > 0) : ratio_area_circle_hexagon r = π / (2 * Real.sqrt 3) := by
  sorry

end ratio_area_proof_l533_533880


namespace scallops_cost_calculation_l533_533871

def scallops_per_pound : ℕ := 8
def cost_per_pound : ℝ := 24.00
def scallops_per_person : ℕ := 2
def number_of_people : ℕ := 8

def total_cost : ℝ := 
  let total_scallops := number_of_people * scallops_per_person
  let total_pounds := total_scallops / scallops_per_pound
  total_pounds * cost_per_pound

theorem scallops_cost_calculation :
  total_cost = 48.00 :=
by sorry

end scallops_cost_calculation_l533_533871


namespace dice_even_probability_l533_533138

theorem dice_even_probability :
  (Pr (λ (outcome : Fin 6 × Fin 6 × Fin 6 × Fin 6),
    (∃ k : Fin 6, k % 2 = 0 ∧ outcome.1 = k ∧ outcome.2 = k ∧ outcome.3 = k ∧ outcome.4 = k) = 
    1/432) :=
sorry

end dice_even_probability_l533_533138


namespace tangent_line_eqn_l533_533610

theorem tangent_line_eqn (f : ℝ → ℝ) (f' : ℝ → ℝ) (A : ℝ × ℝ) (h₁ : ∀ x, f x = -x^3 + 3 * x) 
  (h₂ : ∀ x, f' x = -3 * x^2 + 3) (h₃ : A = (2, -2) ∧ f 2 = -2) :
  ( ∃ m b, f' 2 = -9 ∧ b = A.2 + 9 * A.1 ∧ A.2 = -9 * A.1 + b) ∨ (A.2 = -2) :=
by
  sorry

end tangent_line_eqn_l533_533610


namespace monomial_properties_l533_533800

theorem monomial_properties (a b : ℕ) (h : a = 2 ∧ b = 1) : 
  (2 * a ^ 2 * b = 2 * (a ^ 2) * b) ∧ (2 = 2) ∧ ((2 + 1) = 3) :=
by
  sorry

end monomial_properties_l533_533800


namespace binomial_expansion_l533_533777

theorem binomial_expansion (k n : ℕ) (x : Fin k → ℝ) :
  (∑ i in Finset.univ, x i) ^ n = 
  ∑ l in {l : Fin k → ℕ | ∑ i, l i = n}, 
    (Nat.factorial n / ∏ i, Nat.factorial (l i)) * ∏ i, (x i) ^ (l i) := 
sorry

end binomial_expansion_l533_533777


namespace side_length_of_square_equal_to_triangle_area_l533_533426

theorem side_length_of_square_equal_to_triangle_area :
  ∀ (b h s: ℕ),
  b = 6 ∧ h = 12 ∧ (1/2 * b * h = s * s) → s = 6 :=
by
  intros b h s H
  cases H with Hb Hh
  cases Hh with Hbh Hsq
  sorry

end side_length_of_square_equal_to_triangle_area_l533_533426


namespace max_product_931_l533_533468

open Nat

theorem max_product_931 : ∀ a b c d e : ℕ,
  -- Digits must be between 1 and 9 and all distinct.
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
  {1, 3, 5, 8, 9} = {a, b, c, d, e} ∧ 
  -- Three-digit and two-digit numbers.
  100 * a + 10 * b + c ≤ 999 ∧ 100 * a + 10 * b + c ≥ 100 ∧
  10 * d + e ≤ 99 ∧ 10 * d + e ≥ 10
  -- Then the resulting product is maximized when abc = 931.
  → 100 * a + 10 * b + c = 931 := 
by
  intros a b c d e h_distinct h_digits h_3digit_range h_2digit_range
  sorry

end max_product_931_l533_533468


namespace f_fixed_points_l533_533375

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2

theorem f_fixed_points :
  { x : ℝ | f(f(x)) = f(x) } = {0, 3} :=
by
  sorry

end f_fixed_points_l533_533375


namespace street_length_in_meters_l533_533189

-- Definitions of the conditions
def time_in_minutes : ℕ := 5
def speed_in_kmph : ℝ := 7.2

-- Converting speed from km/h to m/min
def speed_in_mpm : ℝ := (speed_in_kmph * 1000) / 60

-- The main theorem statement without the proof
theorem street_length_in_meters :
  (speed_in_mpm * time_in_minutes) = 600 :=
by
  -- No proof needed, using sorry to signify the omitted proof
  sorry

end street_length_in_meters_l533_533189


namespace clara_current_age_l533_533922

theorem clara_current_age (a c : ℕ) (h1 : a = 54) (h2 : (c - 41) = 3 * (a - 41)) : c = 80 :=
by
  -- This is where the proof would be constructed.
  sorry

end clara_current_age_l533_533922


namespace arithmetic_mean_of_reciprocals_of_first_three_primes_l533_533608

def first_three_primes: List ℕ := [2, 3, 5]

def reciprocals (nums: List ℕ): List ℚ :=
nums.map (λ n => 1 / n)

def arithmetic_mean (fractions: List ℚ): ℚ :=
(fractions.foldr (+) 0) / fractions.length

theorem arithmetic_mean_of_reciprocals_of_first_three_primes:
  arithmetic_mean (reciprocals first_three_primes) = 31 / 90 := by
  sorry

end arithmetic_mean_of_reciprocals_of_first_three_primes_l533_533608


namespace rhombus_area_l533_533802

def diagonal1 : ℝ := 24
def diagonal2 : ℝ := 16

theorem rhombus_area : 0.5 * diagonal1 * diagonal2 = 192 :=
by
  sorry

end rhombus_area_l533_533802


namespace sausage_left_l533_533179

variables (S x y : ℝ)

-- Conditions
axiom dog_bites : y = x + 300
axiom cat_bites : x = y + 500

-- Theorem Statement
theorem sausage_left {S x y : ℝ}
  (h1 : y = x + 300)
  (h2 : x = y + 500) : S - x - y = 400 :=
by
  sorry

end sausage_left_l533_533179


namespace log_prod_eq_neg_twelve_l533_533582

theorem log_prod_eq_neg_twelve : 
  log 2 (1 / 25) * log 3 (1 / 8) * log 5 (1 / 9) = -12 :=
sorry

end log_prod_eq_neg_twelve_l533_533582


namespace smallest_sum_remainder_l533_533430

theorem smallest_sum_remainder (m n : ℕ) (h1 : 1 < m) (h2 : fdomain_length : ((m + n) % 1000) = 371 :=
by
  sorry

end smallest_sum_remainder_l533_533430


namespace find_number_l533_533772

theorem find_number (x : ℝ) (h : (5/4) * x = 40) : x = 32 := 
sorry

end find_number_l533_533772


namespace probability_sunglasses_and_cap_l533_533766

-- We denote the number of people wearing sunglasses, caps, and the probability condition.
def numSunglasses : ℕ := 80
def numCaps : ℕ := 60
def probCapThenSunglasses : ℚ := 1 / 3

-- Definition of the number of people wearing both sunglasses and caps.
def numBothSunglassesAndCaps : ℕ := numCaps * probCapThenSunglasses

-- The main theorem: Finding the probability of wearing sunglasses and also a cap.
theorem probability_sunglasses_and_cap :
  numBothSunglassesAndCaps / numSunglasses = 1 / 4 :=
by
  sorry

end probability_sunglasses_and_cap_l533_533766


namespace system_of_eqs_solution_l533_533075

theorem system_of_eqs_solution :
  ∃ (A B C D : ℚ), 
    (A = B * C * D) ∧ 
    (A + B = C * D) ∧ 
    (A + B + C = D) ∧ 
    (A + B + C + D = 1) ∧ 
    (A = 1/42) ∧ 
    (B = 1/7) ∧ 
    (C = 1/3) ∧ 
    (D = 1/2) :=
by
  exists 1/42, 1/7, 1/3, 1/2
  simp
  -- skip the proof for this exercise
  sorry

end system_of_eqs_solution_l533_533075


namespace alice_cuboids_l533_533555

theorem alice_cuboids (total_faces cuboid_faces : ℕ) (h1 : total_faces = 36) (h2 : cuboid_faces = 6) : total_faces / cuboid_faces = 6 := by
  rw [h1, h2]
  simp
  sorry

end alice_cuboids_l533_533555


namespace grasshoppers_jump_l533_533769

theorem grasshoppers_jump (grasshoppers : Fin 2019 → ℝ) :
  (∀ pos : Fin 2019 → ℝ, (∀ i j, i < j → (pos j - pos i) ∈ (finset.image (λ i, 2 * i) (finset.range 2019))) →
    ∃ i j, i < j ∧ pos j - pos i = 1)
    → 
  (∀ pos : Fin 2019 → ℝ, (∀ i j, i < j → (pos j - pos i) ∈ (finset.image (λ i, 2 * i) (finset.range 2019))) →
    ∃ i j, i < j ∧ pos j - pos i = 1) :=
begin
  sorry
end

end grasshoppers_jump_l533_533769


namespace coplanar_square_area_l533_533459

theorem coplanar_square_area :
  ∀ (side1 side2 side3 : ℝ) (length_occupied : ℝ) (slope : ℝ)
  (height_smallest height_middle height_largest : ℝ),
  side1 = 3 ∧ 
  side2 = 6 ∧ 
  side3 = 9 ∧ 
  length_occupied = side1 + side2 + side3 ∧
  slope = side3 / length_occupied ∧
  height_smallest = side1 * slope ∧
  height_middle = side2 * slope ∧
  height_largest = side3 * slope ∧   
  (shaded_trapezoid_area = 1/2 * (height_smallest + height_middle + height_largest) * (length_occupied / 3)) = 18 ∧
  (largest_triangle_area = 1/2 * side3 * height_largest) = 20.25 := 
by
  intros
  sorry

end coplanar_square_area_l533_533459


namespace chess_tile_covering_proof_l533_533222

noncomputable def tile_covering_impossible : Prop := ∀ (board : Fin 6 × Fin 6 → ℕ)
  (tiles : ℕ) (color_A color_B color_C color_D : Fin 6 × Fin 6 → Bool),
  let A := 9,
  let B := 10,
  let C := 9,
  let D := 8 in
  (tiles = 9) →
  (∀ pos, board pos < 36) →
  (∀ pos, color_A pos = 9 ∧ color_B pos = 10 ∧ color_C pos = 9 ∧ color_D pos = 8) →
  (∀ tile, ∃ (r : Fin 6) (c : Fin 6), (r ≠ r ∨ c ≠ c) →
    (∃ k, tile = 1 * k + 4 * k + board (⟨r, c⟩))) →
  false

theorem chess_tile_covering_proof : tile_covering_impossible := by
  intro board tiles color_A color_B color_C color_D A B C D h_tiles h_board h_colors h_tile
  sorry

end chess_tile_covering_proof_l533_533222


namespace monotonicity_f_l533_533950

open Set

noncomputable def f (a x : ℝ) : ℝ := a * x / (x - 1)

theorem monotonicity_f (a : ℝ) (h : a ≠ 0) :
  (∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → (if a > 0 then f a x1 > f a x2 else if a < 0 then f a x1 < f a x2 else False)) :=
by
  sorry

end monotonicity_f_l533_533950


namespace adrianna_gum_pieces_l533_533199

-- Definitions based on conditions
def initial_gum_pieces : ℕ := 10
def additional_gum_pieces : ℕ := 3
def friends_count : ℕ := 11

-- Expression to calculate the final pieces of gum
def total_gum_pieces : ℕ := initial_gum_pieces + additional_gum_pieces
def gum_left : ℕ := total_gum_pieces - friends_count

-- Lean statement we want to prove
theorem adrianna_gum_pieces: gum_left = 2 := 
by 
  sorry

end adrianna_gum_pieces_l533_533199


namespace Gwen_money_left_l533_533251

theorem Gwen_money_left (received spent : ℕ) (h_received : received = 14) (h_spent : spent = 8) : 
  received - spent = 6 := 
by 
  sorry

end Gwen_money_left_l533_533251


namespace calculate_percentage_l533_533527

/-- A candidate got a certain percentage of the votes polled and he lost to his rival by 2000 votes.
There were 10,000.000000000002 votes cast. What percentage of the votes did the candidate get? --/

def candidate_vote_percentage (P : ℝ) (total_votes : ℝ) (rival_margin : ℝ) : Prop :=
  (P / 100 * total_votes = total_votes - rival_margin) → P = 80

theorem calculate_percentage:
  candidate_vote_percentage P 10000.000000000002 2000 := 
by 
  sorry

end calculate_percentage_l533_533527


namespace uniform_transform_l533_533841

theorem uniform_transform :
  ∀ (a_1 : ℝ), (0 ≤ a_1 ∧ a_1 ≤ 1) →
  let a := a_1 * 5 - 2 in
  -2 ≤ a ∧ a ≤ 3 := 
by {
  intros a_1 ha1,
  let a := a_1 * 5 - 2,
  sorry
}

end uniform_transform_l533_533841


namespace object_reaches_3_2_in_8_steps_or_less_l533_533788

theorem object_reaches_3_2_in_8_steps_or_less :
  ∃ q : ℚ, (q = 195 / 8192) ∧
           (q = (probability_reaches_3_2 8)) :=
by
  sorry

def probability_reaches_3_2 (steps : ℕ) : ℚ :=
sorry

end object_reaches_3_2_in_8_steps_or_less_l533_533788


namespace intercept_sum_equation_l533_533974

theorem intercept_sum_equation (c : ℝ) (h₀ : 3 * x + 4 * y + c = 0)
  (h₁ : (-(c / 3)) + (-(c / 4)) = 28) : c = -48 := 
by
  sorry

end intercept_sum_equation_l533_533974


namespace percentage_profit_is_24_l533_533187

variables (cp sp profit percentage_profit : ℝ)

def cost_price : ℝ := 480
def selling_price : ℝ := 595.2
def profit : ℝ := selling_price - cost_price
def percentage_profit : ℝ := (profit / cost_price) * 100

theorem percentage_profit_is_24 :
  percentage_profit = 24 := sorry

end percentage_profit_is_24_l533_533187


namespace semicircle_area_l533_533747

theorem semicircle_area (A B C : Type) [MetricSpace A] [InnerProductSpace ℝ A]
  [MetricSpace B] [InnerProductSpace ℝ B] [MetricSpace C] [InnerProductSpace ℝ C]
  {AB AC BC : ℝ} (h1 : AB = 2) (h2 : AC = 3) (h3 : BC = 4) : 
  ∃ R : ℝ, R = 27 * Real.pi / 40 :=
by
  sorry

end semicircle_area_l533_533747


namespace mask_production_l533_533177

theorem mask_production (M : ℕ) (h : 16 * M = 48000) : M = 3000 :=
by
  sorry

end mask_production_l533_533177


namespace limit_sequence_is_5_l533_533577

noncomputable def sequence (n : ℕ) : ℝ :=
  (real.sqrt (3 * n - 1) - real.cbrt (125 * n ^ 3 + n)) / (real.rpow n (1 / 5) - n)

theorem limit_sequence_is_5 : 
  filter.tendsto (sequence) at_top (nhds 5) :=
sorry

end limit_sequence_is_5_l533_533577


namespace polynomial_remainder_l533_533596

theorem polynomial_remainder (p : ℚ[X]) (h1 : p.eval 2 = 4) (h2 : p.eval 5 = 10) :
  ∃ (r : ℚ[X]), r.degree < ↓2 ∧ r = 2 * X :=
by
  sorry

end polynomial_remainder_l533_533596


namespace product_of_roots_eq_neg_four_l533_533588

theorem product_of_roots_eq_neg_four : 
  ∀ (x : ℝ), ∃ (a b c : ℝ), 
  (2 * x^3 - 3 * x^2 - 10 * x + 8 = 0) ∧ 
  a = 2 ∧ b = -3 ∧ c = -10 ∧ d = 8 
  → 
  (xyz := (2 : ℝ), (-3 : ℝ), (-10 : ℝ), (8 : ℝ)),
  ({rootsa rootsb rootsc rootsd} : Polynomial ℝ) roots (xyz) :=
begin
  -- Definitions
  let a := 2,
  let d := 8,
  -- Statement to Prove
  show product_of_roots (a, b, c, d) = -4,
  sorry,
end.

end product_of_roots_eq_neg_four_l533_533588


namespace cubic_roots_expression_l533_533380

theorem cubic_roots_expression (p q r : ℝ) 
  (h1 : p + q + r = 4) 
  (h2 : pq + pr + qr = 6) 
  (h3 : pqr = 3) : 
  p / (qr + 2) + q / (pr + 2) + r / (pq + 2) = 4 / 5 := 
by 
  sorry

end cubic_roots_expression_l533_533380


namespace arithmetic_common_difference_l533_533017

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533017


namespace girls_to_total_ratio_l533_533215

def total_campers : ℕ := 96
def boys_ratio : ℚ := 2 / 3
def girls_ratio : ℚ := 1 - boys_ratio
def boys_count : ℕ := (boys_ratio * total_campers).toNat
def girls_count : ℕ := total_campers - boys_count

theorem girls_to_total_ratio : (girls_count : ℚ) / total_campers = 1 / 3 := by
  sorry

end girls_to_total_ratio_l533_533215


namespace range_of_a_l533_533688

theorem range_of_a (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 2*a*x - 2*y + 3 - a = 0 ∧ (a < 0 ∧ a ≠ -2 ∧ a ≠ 1)) ↔ a ∈ Iio (-2) :=
by sorry

end range_of_a_l533_533688


namespace polar_eqn_circle_segment_length_PQ_solve_inequality_min_val_2a_b_l533_533868

-- Problem 1a
theorem polar_eqn_circle (θ : ℝ) : (∃ φ : ℝ, 1 + cos φ = ρ * cos θ ∧ sin φ = ρ * sin θ) → ρ = 2 * cos θ :=
sorry

-- Problem 1b
theorem segment_length_PQ (Q P : ℝ × ℝ) :
  (Q = (1/2, sqrt(3)/2)) ∧
  (P = (3/2, 3*sqrt(3)/2)) →
  dist P Q = 2 :=
sorry

-- Problem 2a
theorem solve_inequality (x : ℝ) : (x + 1 + |3 - x| ≤ 6 ∧ x ≥ -1) → -1 ≤ x ∧ x ≤ 4 :=
sorry

-- Problem 2b
theorem min_val_2a_b (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 2 * 4 * a * b = a + 2 * b → 
  2 * a + b = 9 / 8 :=
sorry

end polar_eqn_circle_segment_length_PQ_solve_inequality_min_val_2a_b_l533_533868


namespace shapes_remaining_after_turns_l533_533859

-- Define the initial conditions for the problem
def initial_shapes : ℕ := 12
def triangles : ℕ := 3
def squares : ℕ := 4
def pentagons : ℕ := 5
def turns_per_player : ℕ := 5

-- Define the main question to prove
theorem shapes_remaining_after_turns 
  (initial_shapes = 12)
  (turns_per_player = 5)
  (petya_starts : Prop)
  (no_shared_sides : Prop)
  (petya_strategy : Prop)
  (vasya_strategy : Prop) :
  ∃ remaining_shapes : ℕ, remaining_shapes = 6 := 
sorry

end shapes_remaining_after_turns_l533_533859


namespace total_cost_for_doughnuts_l533_533928

theorem total_cost_for_doughnuts
  (num_students : ℕ)
  (num_chocolate : ℕ)
  (num_glazed : ℕ)
  (price_chocolate : ℕ)
  (price_glazed : ℕ)
  (H1 : num_students = 25)
  (H2 : num_chocolate = 10)
  (H3 : num_glazed = 15)
  (H4 : price_chocolate = 2)
  (H5 : price_glazed = 1) :
  num_chocolate * price_chocolate + num_glazed * price_glazed = 35 :=
by
  -- Proof steps would go here
  sorry

end total_cost_for_doughnuts_l533_533928


namespace handshake_even_acquaintance_l533_533924

theorem handshake_even_acquaintance (n : ℕ) (hn : n = 225) : 
  ∃ (k : ℕ), k < n ∧ (∀ m < n, k ≠ m) :=
by sorry

end handshake_even_acquaintance_l533_533924


namespace exists_k6_l533_533358

open Finset

variable {α : Type*}

theorem exists_k6 (E : Finset α) [Fintype α] (h_size : E.card = 1991)
  (h_deg : ∀ x ∈ E, (E.filter (λ y, x ≠ y ∧ (∃ p, path p x y))).card ≥ 1593) :
  ∃ (K : Finset α), K.card = 6 ∧ (∀ x ∈ K, ∀ y ∈ K, x ≠ y → (∃ p, path p x y)) := sorry

end exists_k6_l533_533358


namespace circle_center_l533_533531

theorem circle_center (h1 : circle_passes_through (0, 2))
                     (h2 : circle_tangent_to (parabola y = x^2) (1, 1)) :
                     center_of_circle = (0, 1) :=
sorry

end circle_center_l533_533531


namespace arithmetic_common_difference_l533_533030

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533030


namespace sum_quotient_is_integer_l533_533621

theorem sum_quotient_is_integer
  (n : ℕ) 
  (h_n : n ≥ 2) 
  (a : Fin n → ℕ) 
  (distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j) 
  (k : ℕ) :
  ∃ (P : Fin n → ℤ), 
    (∀ i : Fin n, P i = ∏ j in (Finset.univ.filter (λ j, j ≠ i)), (a i - a j)) ∧
    (∑ i in Finset.univ, ((a i : ℤ)^k / P i)) ∈ ℤ :=
sorry

end sum_quotient_is_integer_l533_533621


namespace smallest_N_divisible_by_p_l533_533744

theorem smallest_N_divisible_by_p (p : ℕ) (hp : Nat.Prime p)
    (N1 : ℕ) (N2 : ℕ) :
  (∃ N1 N2, 
    (N1 % p = 0 ∧ ∀ n, 1 ≤ n ∧ n < p → N1 % n = 1) ∧
    (N2 % p = 0 ∧ ∀ n, 1 ≤ n ∧ n < p → N2 % n = n - 1)
  ) :=
sorry

end smallest_N_divisible_by_p_l533_533744


namespace find_angle_EOM_l533_533719

-- Given definitions
structure Trapezoid (A B C D E O M : Type) :=
  (perpendicular_AB_AD : ∀ (a : A) (b : B) (d : D), (perp a b d))
  (perpendicular_AB_BC : ∀ (a : A) (b : B) (c : C), (perp a b c))
  (AB_eq_sqrt_AD_BC : ∀ (ab ad bc : ℝ), ab = Real.sqrt (ad * bc))

-- Main statement to prove
theorem find_angle_EOM
  {A B C D E O M : Type}
  [trapezoid_structure : Trapezoid A B C D E O M]
  (O_is_intersection : ∀ (o : O) (e : E) (d1 d2 : Type), intersects d1 d2 o)
  (E_is_intersection : ∀ (e : E) (p1 p2 : Type), intersects p1 p2 e)
  (M_is_midpoint : ∀ (m : M) (ab : ℝ), midpoint m ab) :
  ∃ α, angle O E M α ∧ α = 90 :=
sorry

end find_angle_EOM_l533_533719


namespace largest_root_l533_533483

theorem largest_root (a b c : ℝ) (ha : a = 9) (hb : b = -74) (hc : c = 9)
  (hΔ : Δ = (b^2) - 4 * a * c)
  (hroot1 : x1 = (-b + real.sqrt Δ) / (2 * a))
  (hroot2 : x2 = (-b - real.sqrt Δ) / (2 * a))
  (hΔpos : Δ > 0)
  (hlargest : ∀ x, x = max x1 x2):
  hlargest ≈ 8.099 := sorry

end largest_root_l533_533483


namespace categorize_numbers_l533_533963

def numbers : List ℚ := [-16/10, -5/6, 89/10, -7, 1/12, 0, 25]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x.den ≠ 1
def is_negative_integer (x : ℚ) : Prop := x < 0 ∧ x.den = 1

theorem categorize_numbers :
  { x | x ∈ numbers ∧ is_positive x } = { 89 / 10, 1 / 12, 25 } ∧
  { x | x ∈ numbers ∧ is_negative_fraction x } = { -5 / 6 } ∧
  { x | x ∈ numbers ∧ is_negative_integer x } = { -7 } := by
  sorry

end categorize_numbers_l533_533963


namespace how_many_dozen_oatmeal_raisin_given_away_l533_533921

theorem how_many_dozen_oatmeal_raisin_given_away
  (oatmeal_cookie_baked : ℕ) (sugar_cookie_baked : ℕ) (chocolate_chip_cookie_baked : ℕ)
  (sugar_cookie_given : ℕ) (chocolate_chip_cookie_given : ℕ) (cookies_kept : ℕ)
  (H1 : oatmeal_cookie_baked = 3 * 12)  -- 3 dozen oatmeal raisin cookies
  (H2 : sugar_cookie_baked = 2 * 12)  -- 2 dozen sugar cookies
  (H3 : chocolate_chip_cookie_baked = 4 * 12)  -- 4 dozen chocolate chip cookies
  (H4 : sugar_cookie_given = 1.5 * 12)  -- 1.5 dozen sugar cookies given away
  (H5 : chocolate_chip_cookie_given = 2.5 * 12)  -- 2.5 dozen chocolate chip cookies given away
  (H6 : cookies_kept = 36) :  -- 36 cookies kept in total
  ∃ (oatmeal_cookie_given : ℕ), oatmeal_cookie_given / 12 = 2 :=  -- 2 dozen oatmeal raisin cookies given away

begin
  sorry
end

end how_many_dozen_oatmeal_raisin_given_away_l533_533921


namespace general_term_arithmetic_sum_of_first_n_terms_geometric_l533_533643

def arithmetic_sequence (a : ℕ → ℕ) := ∀ n, ∃ d, a n = a 1 + (n - 1) * d

def geometric_sequence (b : ℕ → ℕ) := ∀ n, ∃ q, b n = b 1 * q ^ (n - 1)

theorem general_term_arithmetic (a : ℕ → ℕ) (h1 : a 1 = 1) (h3 : a 3 = 5) :
  arithmetic_sequence a → ∀ n, a n = 2 * n - 1 :=
by sorry

theorem sum_of_first_n_terms_geometric (a b : ℕ → ℕ) (h_a1 : a 1 = 1) (h_a3 : a 3 = 5)
  (h_b1 : b 1 = a 2) (h_b2 : b 2 = a 1 + a 2 + a 3) :
  geometric_sequence b → ∀ n, ∑ i in finset.range n, b i = (3^(n+1) - 3) / 2 :=
by sorry

end general_term_arithmetic_sum_of_first_n_terms_geometric_l533_533643


namespace correct_statements_count_l533_533345

theorem correct_statements_count :
  ∀ (correlation_deterministic : false)
    (regression_model : true)
    (regression_effect : true)
    (independence_test : false)
    (residual_distribution : true),
    [regression_model, regression_effect, residual_distribution].count (λ b, b = true) = 3 := 
by
  intros
  sorry

end correct_statements_count_l533_533345


namespace correct_operation_l533_533156

theorem correct_operation : 
  ¬(2 + sqrt 2 = 2 * sqrt 2) ∧
  ¬(4 * x^2 * y - x^2 * y = 3) ∧
  ¬((a + b)^2 = a^2 + b^2) ∧
  ((ab)^3 = a^3 * b^3) :=
by sorry

end correct_operation_l533_533156


namespace smallest_marked_cells_l533_533152

theorem smallest_marked_cells (k : ℕ) : ∃ k = 48, ∀ (marking : ℕ × ℕ → Prop), 
  (∀ i j, 0 ≤ i < 12 → 0 ≤ j < 12 → marking (i, j)) → 
  (∃ i j, 0 ≤ i < 12 → 0 ≤ j < 12 → 
    (marking (i, j)) → 
    (∀ r s, 0 ≤ r < 2 → 0 ≤ s < 2 → 
      (i + r, j + s) ∈ { (i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1) }) → 
    ∃ marked_cell (i + r, j + s) ) := 
sorry

end smallest_marked_cells_l533_533152


namespace sequence_a_10_value_l533_533359

theorem sequence_a_10_value : 
  ∀ (a : ℕ → ℤ), a 1 = 3 → (∀ n : ℕ, 0 < n → a (n + 1) - a n = 2) → a 10 = 21 := 
by 
  intros a h1 hdiff
  sorry

end sequence_a_10_value_l533_533359


namespace binomial_probability_example_l533_533994

noncomputable def binomial_distribution (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (nat.choose n k : ℚ) * (p^k) * ((1 - p)^(n - k))

theorem binomial_probability_example :
  let n := 6
  let p := (1 : ℚ) / 3
  let k := 2
  binomial_distribution n p k = 80 / 243 :=
by 
  sorry

end binomial_probability_example_l533_533994


namespace exists_point_K_l533_533477

theorem exists_point_K (A B C K : Point)
  (h_triangle_ABC : acute_triangle A B C)
  (condition1 : angle K B A = 2 * angle K A B)
  (condition2 : angle K B C = 2 * angle K C B) :
  ∃ K : Point, angle K B A = 2 * angle K A B ∧ angle K B C = 2 * angle K C B :=
sorry

end exists_point_K_l533_533477


namespace no_equal_area_parallelograms_l533_533066

-- Define the area of the regular octagon
def octagon_area : ℝ := 2 + 2 * Real.sqrt 2

-- Define the conditions that need to be met for decomposition
def parallelogram_conditions (n : ℕ) (t : ℝ) : Prop :=
  (n * t = 2 + 2 * Real.sqrt 2) ∧ (1 = k * t) for some integer k

-- The main theorem statement
theorem no_equal_area_parallelograms : ¬ ∃ n t, parallelogram_conditions n t :=
by
  -- Assuming proof steps are omitted
  sorry

end no_equal_area_parallelograms_l533_533066


namespace height_enlarged_by_factor_of_four_l533_533330

variables (a b c h : ℝ)
def enlarged_sides (factor : ℝ) := (factor * a, factor * b, factor * c)
def original_height := h

theorem height_enlarged_by_factor_of_four :
  ∀ (factor : ℝ), factor = 4 → original_height = h → ∃ h', h' = factor * h := by
  intros factor hf heq
  use factor * h
  rw [hf, heq]
  sorry

end height_enlarged_by_factor_of_four_l533_533330


namespace prove_common_difference_l533_533011

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533011


namespace ratio_of_smaller_circle_to_larger_hexagon_l533_533881

-- Definitions based on the conditions
def larger_circle_radius (r : ℝ) : ℝ := r / sqrt 3
def larger_hexagon_side_length (r : ℝ) : ℝ := 2 * larger_circle_radius r
def area_of_larger_hexagon (r : ℝ) : ℝ := (3 * sqrt 3 / 2) * (larger_hexagon_side_length r) ^ 2
def area_of_smaller_circle (r : ℝ) : ℝ := π * r ^ 2
def ratio_of_areas (r : ℝ) : ℝ := area_of_smaller_circle r / area_of_larger_hexagon r

-- The theorem we need to prove
theorem ratio_of_smaller_circle_to_larger_hexagon (r : ℝ) (hr : r > 0) :
  ratio_of_areas r = π / (2 * sqrt 3) :=
by
  sorry

end ratio_of_smaller_circle_to_larger_hexagon_l533_533881


namespace peony_total_count_l533_533073

theorem peony_total_count (n : ℕ) (x : ℕ) (total_sample : ℕ) (single_sample : ℕ) (double_sample : ℕ) (thousand_sample : ℕ) (extra_thousand : ℕ)
    (h1 : thousand_sample > single_sample)
    (h2 : thousand_sample - single_sample = extra_thousand)
    (h3 : total_sample = single_sample + double_sample + thousand_sample)
    (h4 : total_sample = 12)
    (h5 : single_sample = 4)
    (h6 : double_sample = 2)
    (h7 : thousand_sample = 6)
    (h8 : extra_thousand = 30) :
    n = 180 :=
by 
  sorry

end peony_total_count_l533_533073


namespace melissa_games_played_l533_533052

theorem melissa_games_played (total_points : ℕ) (points_per_game : ℕ) (num_games : ℕ) 
  (h1 : total_points = 81) 
  (h2 : points_per_game = 27) 
  (h3 : num_games = total_points / points_per_game) : 
  num_games = 3 :=
by
  -- Proof goes here
  sorry

end melissa_games_played_l533_533052


namespace points_coincide_l533_533939

-- Define the circles and their intersection points
variables {S1 S2 S3 : Type} -- placeholder for actual circle definitions
variables {A1 A2 A3 A4 A5 A6 : Type} -- intersection points

-- Define the sequence of points M1, M2, M3, M4, M5, M6, M7
variables (M1 M2 M3 M4 M5 M6 M7 : Type)

-- Define the conditions
axiom intersect_1 : ∃ M1 M2, M1 ≠ M2 ∧ M1 ∈ S1 ∧ M1 ∈ S2 ∧ M2 ∈ S1 ∧ M2 ∈ S2 ∧ A1 ∈ S1 ∧ A1 ∈ S2
axiom intersect_2 : ∃ M2 M3, M2 ≠ M3 ∧ M2 ∈ S2 ∧ M2 ∈ S3 ∧ M3 ∈ S2 ∧ M3 ∈ S3 ∧ A2 ∈ S2 ∧ A2 ∈ S3
axiom intersect_3 : ∃ M3 M4, M3 ≠ M4 ∧ M3 ∈ S3 ∧ M3 ∈ S1 ∧ M4 ∈ S3 ∧ M4 ∈ S1 ∧ A3 ∈ S3 ∧ A3 ∈ S1
axiom intersect_4 : ∃ M4 M5, M4 ≠ M5 ∧ M4 ∈ S1 ∧ M4 ∈ S2 ∧ M5 ∈ S1 ∧ M5 ∈ S2 ∧ A4 ∈ S1 ∧ A4 ∈ S2
axiom intersect_5 : ∃ M5 M6, M5 ≠ M6 ∧ M5 ∈ S2 ∧ M5 ∈ S3 ∧ M6 ∈ S2 ∧ M6 ∈ S3 ∧ A5 ∈ S2 ∧ A5 ∈ S3
axiom intersect_6 : ∃ M6 M7, M6 ≠ M7 ∧ M6 ∈ S3 ∧ M6 ∈ S1 ∧ M7 ∈ S3 ∧ M7 ∈ S1 ∧ A6 ∈ S3 ∧ A6 ∈ S1

-- Define the theorem to be proven
theorem points_coincide : M1 = M7 :=
sorry 

end points_coincide_l533_533939


namespace melted_mixture_weight_l533_533166

-- Let Zinc and Copper be real numbers representing their respective weights in kilograms.
variables (Zinc Copper: ℝ)
-- Assume the ratio of Zinc to Copper is 9:11.
axiom ratio_zinc_copper : Zinc / Copper = 9 / 11
-- Assume 26.1kg of Zinc has been used.
axiom zinc_value : Zinc = 26.1

-- Define the total weight of the melted mixture.
def total_weight := Zinc + Copper

-- We state the theorem to prove that the total weight of the mixture equals 58kg.
theorem melted_mixture_weight : total_weight Zinc Copper = 58 :=
by
  sorry

end melted_mixture_weight_l533_533166


namespace cos_C_value_l533_533333

-- Define the given conditions
def triangle_ABC : Type := 
  { A B C : ℝ // A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ C < π }

-- Define the given sin and cos values
def given_sin_cos (A B C : ℝ) (h : A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ C < π) : Prop :=
  sin A = 3 / 5 ∧ cos B = 5 / 13

-- The theorem to be proved
theorem cos_C_value {A B C : ℝ} (h : A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ C < π)
    (hc : given_sin_cos A B C h) : cos C = 16 / 65 :=
sorry

end cos_C_value_l533_533333


namespace house_number_count_l533_533954

def is_two_digit_prime (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100 ∧ Prime n

def house_number_conditions (AB CD : ℕ) : Prop :=
  is_two_digit_prime AB ∧ is_two_digit_prime CD ∧ AB ≠ CD

theorem house_number_count : 
  ∃ (count : ℕ), count = 110 ∧ 
  (∀ (AB CD : ℕ), house_number_conditions AB CD → AB < 50 → CD < 50) :=
by 
  use 110
  intros AB CD hABCD hAB50 hCD50
  have h1 : ∃ primes : ℕ, primes = 11 := sorry
  sorry

end house_number_count_l533_533954


namespace incorrect_geometric_locus_definitions_l533_533590

theorem incorrect_geometric_locus_definitions
  (locus_condition : Point → Prop)
  (locus : Set Point)
  (A : ∀ p, p ∈ locus ↔ locus_condition p)
  (B : ∀ p, ¬locus_condition p → ¬(p ∈ locus) ∧ ∃ p, locus_condition p ∧ ¬(p ∈ locus))
  (C : ∀ p, p ∈ locus ↔ locus_condition p)
  (D : ∀ p, ¬(p ∈ locus) → ¬locus_condition p ∧ (locus_condition p → p ∈ locus))
  (E : ∀ p, p ∈ locus → locus_condition p ∧ ∃ p, locus_condition p ∧ ¬(p ∈ locus))
: B ∧ E := by {
  sorry
}

end incorrect_geometric_locus_definitions_l533_533590


namespace ellipse_standard_eq_and_eccentricity_line_eq_area_condition_l533_533644

noncomputable theory

def ellipse_eq (x y : ℝ) := (x^2 / 3) + (y^2 / 2) = 1

def f1 := (-1, 0)
def f2 := (1, 0)

def point_p : ℝ × ℝ := (-1, 2*real.sqrt(3) / 3)
def distance_pf2 : ℝ := 4*real.sqrt(3) / 3

theorem ellipse_standard_eq_and_eccentricity :
  ellipse_eq (point_p.1) (point_p.2) ∧
  ecc (sqrt(3)) (sqrt(2)) = (sqrt(3) / 3) :=
sorry

def line_eq (x k : ℝ) := k * (x + 1)

theorem line_eq_area_condition (l : ℝ → ℝ) :
  area (0, 0) (M.1, M.2) (N.1, N.2) = 12 / 11 → 
  ∀ (x : ℝ) (k : ℝ), (line_eq x k = l x) :=
sorry

end ellipse_standard_eq_and_eccentricity_line_eq_area_condition_l533_533644


namespace R_and_D_expenditure_per_unit_increase_l533_533931

theorem R_and_D_expenditure_per_unit_increase :
  (R_t : ℝ) (delta_APL_t_plus_1 : ℝ) (R_t = 3157.61) (delta_APL_t_plus_1 = 0.69) :
  R_t / delta_APL_t_plus_1 = 4576 :=
by
  sorry

end R_and_D_expenditure_per_unit_increase_l533_533931


namespace compute_R_at_3_l533_533374

def R (x : ℝ) := 3 * x ^ 4 + x ^ 3 + x ^ 2 + x + 1

theorem compute_R_at_3 : R 3 = 283 := by
  sorry

end compute_R_at_3_l533_533374


namespace area_of_each_triangle_l533_533838

-- Conditions
def is_rhombus (a b : ℝ) : Prop :=
  a * b / 2 = 150

-- Define the area of each triangle
def each_triangle_area (a b : ℝ) : ℝ :=
  (a * b / 2) / 2

theorem area_of_each_triangle (a b : ℝ) (h : is_rhombus a b) : each_triangle_area a b = 75 := by
  sorry

end area_of_each_triangle_l533_533838


namespace translation_even_function_l533_533461

def f (x : ℝ) : ℝ := sin x * cos x - 1 + sin x ^ 2

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem translation_even_function : 
  is_even (λ x, f (x - π / 8)) :=
sorry

end translation_even_function_l533_533461


namespace pine_tree_count_l533_533339

theorem pine_tree_count (P R : ℕ) 
  (h1 : R = 1.20 * P)
  (h2 : P + R = 1320) : 
  P = 600 :=
sorry

end pine_tree_count_l533_533339


namespace max_product_yields_831_l533_533471

open Nat

noncomputable def max_product_3digit_2digit : ℕ :=
  let digits : List ℕ := [1, 3, 5, 8, 9]
  let products := digits.permutations.filter (λ l, l.length ≥ 5).map (λ l, 
    let a := l.get? 0 |>.getD 0
    let b := l.get? 1 |>.getD 0
    let c := l.get? 2 |>.getD 0
    let d := l.get? 3 |>.getD 0
    let e := l.get? 4 |>.getD 0
    (100 * a + 10 * b + c) * (10 * d + e)
  )
  products.maximum.getD 0

theorem max_product_yields_831 : 
  ∃ (a b c d e : ℕ), 
    List.permutations [1, 3, 5, 8, 9].any 
      (λ l, l = [a, b, c, d, e] ∧ 
            100 * a + 10 * b + c = 831 ∧
            ∀ (x y z w v : ℕ), 
              List.permutations [1, 3, 5, 8, 9].any 
                (λ l, l = [x, y, z, w, v] ∧ 
                      (100 * x + 10 * y + z) * (10 * w + v) ≤ (100 * a + 10 * b + c) * (10 * d + e)
                )
      )
:= by
  sorry

end max_product_yields_831_l533_533471


namespace max_three_digit_is_931_l533_533475

def greatest_product_three_digit : Prop :=
  ∃ (a b c d e : ℕ), {a, b, c, d, e} = {1, 3, 5, 8, 9} ∧
  1 ≤ a ∧ a ≤ 9 ∧
  1 ≤ b ∧ b ≤ 9 ∧
  1 ≤ c ∧ c ≤ 9 ∧
  1 ≤ d ∧ d ≤ 9 ∧
  1 ≤ e ∧ e ≤ 9 ∧
  (100 * a + 10 * b + c) * (10 * d + e) = 81685 ∧
  (100 * a + 10 * b + c) = 931

theorem max_three_digit_is_931 : greatest_product_three_digit :=
begin
  sorry
end

end max_three_digit_is_931_l533_533475


namespace scallops_cost_calculation_l533_533870

def scallops_per_pound : ℕ := 8
def cost_per_pound : ℝ := 24.00
def scallops_per_person : ℕ := 2
def number_of_people : ℕ := 8

def total_cost : ℝ := 
  let total_scallops := number_of_people * scallops_per_person
  let total_pounds := total_scallops / scallops_per_pound
  total_pounds * cost_per_pound

theorem scallops_cost_calculation :
  total_cost = 48.00 :=
by sorry

end scallops_cost_calculation_l533_533870


namespace function_characterization_l533_533995

noncomputable def f : ℝ → ℝ := sorry

theorem function_characterization (f : ℝ → ℝ) (k : ℝ) :
  (∀ x y : ℝ, f (x^2 + 2*x*y + y^2) = (x + y) * (f x + f y)) →
  (∀ x : ℝ, |f x - k * x| ≤ |x^2 - x|) →
  ∀ x : ℝ, f x = k * x :=
by
  sorry

end function_characterization_l533_533995


namespace prime_divisor_not_dividing_2_pow_m_minus_1_l533_533067

theorem prime_divisor_not_dividing_2_pow_m_minus_1 {n m : ℤ} (hnm : n > m) (hm : m > 0) :
  ∃ p : ℕ, Prime p ∧ p ∣ (2^n - 1) ∧ ¬ p ∣ (2^m - 1) := 
sorry

end prime_divisor_not_dividing_2_pow_m_minus_1_l533_533067


namespace line_parallel_to_one_of_two_parallel_planes_l533_533320

theorem line_parallel_to_one_of_two_parallel_planes {line : affine_space ℝ} {plane1 plane2 : affine_space ℝ} 
  (h_parallel : line ∥ plane1)
  (h_planes_parallel : plane1 ∥ plane2) : 
  (line ∥ plane2) ∨ (line ⊂ plane2) :=
sorry

end line_parallel_to_one_of_two_parallel_planes_l533_533320


namespace fractional_part_inequality_l533_533167

theorem fractional_part_inequality (p s : ℤ) (hp : p.prime) (hs : 0 < s) (hsp : s < p) :
  (∃ m n : ℤ, 0 < m ∧ m < n ∧ n < p ∧ 
  (s * m / p - (s * m / p).floor : ℚ) < (s * n / p - (s * n / p).floor : ℚ) ∧
  (s * n / p - (s * n / p).floor : ℚ) < s / p) ↔ ¬ s ∣ (p - 1) :=
by
  sorry

end fractional_part_inequality_l533_533167


namespace grasshoppers_jump_l533_533768

theorem grasshoppers_jump (grasshoppers : Fin 2019 → ℝ) :
  (∀ pos : Fin 2019 → ℝ, (∀ i j, i < j → (pos j - pos i) ∈ (finset.image (λ i, 2 * i) (finset.range 2019))) →
    ∃ i j, i < j ∧ pos j - pos i = 1)
    → 
  (∀ pos : Fin 2019 → ℝ, (∀ i j, i < j → (pos j - pos i) ∈ (finset.image (λ i, 2 * i) (finset.range 2019))) →
    ∃ i j, i < j ∧ pos j - pos i = 1) :=
begin
  sorry
end

end grasshoppers_jump_l533_533768


namespace initial_savings_l533_533591

-- Definitions based on problem conditions
def weekly_savings := 10
def weeks := 10
def vacuum_cost := 120
def total_savings := weekly_savings * weeks

-- Lean statement to prove the equivalence
theorem initial_savings : total_savings = 100 → (vacuum_cost - total_savings = 20) :=
by
  intro h
  calc
    vacuum_cost - total_savings = 120 - 100 := by rw [h]
                                 ...         = 20        := by rfl

end initial_savings_l533_533591


namespace coaches_average_age_is_23_l533_533702

noncomputable def average_age_of_coaches (total_members men women coaches : ℕ) 
  (total_avg_age men_avg_age women_avg_age : ℝ)
  (h_total_members : total_members = 50)
  (h_total_avg_age : total_avg_age = 20)
  (h_men : men = 30)
  (h_women : women = 15)
  (h_coaches : coaches = 5)
  (h_men_avg_age : men_avg_age = 19)
  (h_women_avg_age : women_avg_age = 21) : ℝ :=
  let total_age := total_members * total_avg_age in
  let sum_men := men * men_avg_age in
  let sum_women := women * women_avg_age in
  let sum_coaches := total_age - (sum_men + sum_women) in
  let avg_coaches := sum_coaches / coaches in
  avg_coaches

theorem coaches_average_age_is_23 : average_age_of_coaches 50 30 15 5 20 19 21 
  50 rfl 20 rfl 30 rfl 15 rfl 5 rfl 19 rfl 21 rfl = 23 := 
by 
  sorry

end coaches_average_age_is_23_l533_533702


namespace symmetric_intervals_equal_l533_533252

noncomputable def normal_distribution (μ σ : ℝ) := sorry

def X := normal_distribution 1 9

def probability_interval (X : Type) (a b : ℝ) := sorry

def m := probability_interval X 2 3
def n := probability_interval X (-1) 0

theorem symmetric_intervals_equal (X : Type) (m n : ℝ) :
  m = n :=
sorry

end symmetric_intervals_equal_l533_533252


namespace fraction_multiplication_l533_533481

theorem fraction_multiplication :
  (2 / 3) * (3 / 8) = (1 / 4) :=
sorry

end fraction_multiplication_l533_533481


namespace product_of_real_roots_eq_one_l533_533969

theorem product_of_real_roots_eq_one:
  ∀ x : ℝ, (x ^ (Real.log x / Real.log 5) = 25) → (∀ x1 x2 : ℝ, (x1 ^ (Real.log x1 / Real.log 5) = 25) → (x2 ^ (Real.log x2 / Real.log 5) = 25) → x1 * x2 = 1) :=
by
  sorry

end product_of_real_roots_eq_one_l533_533969


namespace sum_of_roots_quadratic_polynomial_l533_533098

theorem sum_of_roots_quadratic_polynomial (Q : ℝ → ℝ)
  (hQ : ∀ x : ℝ, Q(x^3 - x) ≥ Q(x^2 - 1))
  (quadratic : ∃ a b c : ℝ, (∀ x : ℝ, Q x = a * x^2 + b * x + c)) :
  (∃ a b c : ℝ, Q x = a * x^2 + b * x + c → -b / a = 0) :=
sorry

end sum_of_roots_quadratic_polynomial_l533_533098


namespace speed_of_boat_in_still_water_l533_533103

variable (x : ℝ)

theorem speed_of_boat_in_still_water (h : 10 = (x + 5) * 0.4) : x = 20 :=
sorry

end speed_of_boat_in_still_water_l533_533103


namespace trigonometric_identity_l533_533949

theorem trigonometric_identity :
  cos (36 * Real.pi / 180) * cos (24 * Real.pi / 180) - sin (36 * Real.pi / 180) * sin (24 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trigonometric_identity_l533_533949


namespace shapes_remaining_after_turns_l533_533860

-- Define the initial conditions for the problem
def initial_shapes : ℕ := 12
def triangles : ℕ := 3
def squares : ℕ := 4
def pentagons : ℕ := 5
def turns_per_player : ℕ := 5

-- Define the main question to prove
theorem shapes_remaining_after_turns 
  (initial_shapes = 12)
  (turns_per_player = 5)
  (petya_starts : Prop)
  (no_shared_sides : Prop)
  (petya_strategy : Prop)
  (vasya_strategy : Prop) :
  ∃ remaining_shapes : ℕ, remaining_shapes = 6 := 
sorry

end shapes_remaining_after_turns_l533_533860


namespace new_barbell_cost_l533_533722

theorem new_barbell_cost (old_cost : ℝ) (increase_percentage : ℝ) (increase : ℝ): 
  old_cost = 250 ∧ increase_percentage = 0.30 ∧ increase = old_cost * increase_percentage → 
  old_cost + increase = 325 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3] at h4
  sorry

end new_barbell_cost_l533_533722


namespace probability_of_selected_member_l533_533341

section Probability

variables {N : ℕ} -- Total number of members in the group

-- Conditions
-- Probabilities of selecting individuals by gender
def P_woman : ℝ := 0.70
def P_man : ℝ := 0.20
def P_non_binary : ℝ := 0.10

-- Conditional probabilities of occupations given gender
def P_engineer_given_woman : ℝ := 0.20
def P_doctor_given_man : ℝ := 0.20
def P_translator_given_non_binary : ℝ := 0.20

-- The main proof statement
theorem probability_of_selected_member :
  (P_woman * P_engineer_given_woman) + (P_man * P_doctor_given_man) + (P_non_binary * P_translator_given_non_binary) = 0.20 :=
by
  sorry

end Probability

end probability_of_selected_member_l533_533341


namespace range_of_m_l533_533227

-- Define function f on ℝ satisfying the conditions
def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x < 1 then 
    (1 / 2) - 2 * x^2 
  else if 1 ≤ x ∧ x < 2 then 
    -2^(1 - abs(x - 3 / 2)) 
  else 
    sorry  -- Assuming the general definition for all other cases

-- Define the periodic condition of function f
axiom f_periodic : ∀ x : ℝ, f (x + 2) = (1 / 2) * f x

-- Define function g
def g (x : ℝ) (m : ℝ) : ℝ := x^3 + 3 * x^2 + m

-- Theorem to demonstrate the range of m
theorem range_of_m : ∀ (m : ℝ), 
  (∀ s ∈ set.Ico(-4 : ℝ, -2), ∃ t ∈ set.Ico(-4 : ℝ, -2), f s - g t m ≥ 0) ↔ m ≤ 8 :=
by
  sorry

end range_of_m_l533_533227


namespace smallest_integral_l533_533038

open Real

noncomputable def p (x : ℝ) : ℝ := 2 * (x^6 + 1) + 4 * (x^5 + x) + 3 * (x^4 + x^2) + 5 * x^3

noncomputable def a : ℝ := ∫ x in 0..∞, x / p x
noncomputable def b : ℝ := ∫ x in 0..∞, x^2 / p x
noncomputable def c : ℝ := ∫ x in 0..∞, x^3 / p x
noncomputable def d : ℝ := ∫ x in 0..∞, x^4 / p x

theorem smallest_integral : b = min (min a b) (min c d) := 
sorry

end smallest_integral_l533_533038


namespace min_value_of_expression_l533_533262

theorem min_value_of_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : 4 * x + 3 * y = 1) :
  1 / (2 * x - y) + 2 / (x + 2 * y) = 9 :=
sorry

end min_value_of_expression_l533_533262


namespace ellipse_properties_l533_533208

-- Define the conditions for the ellipse
def foci_F1 : ℝ × ℝ := (0, -1)
def foci_F2 : ℝ × ℝ := (0, 1)
def point_P : ℝ × ℝ := (Real.sqrt 2 / 2, 1)
def point_S : ℝ × ℝ := (-1 / 3, 0)

-- The equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop := x^2 + y^2 / 2 = 1

-- Define the points on the ellipse with a line passing through S intersecting the ellipse at A and B
def intersects_at_A_B (l : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ, 
    (y₁ = l x₁ ∧ y₂ = l x₂) ∧ 
    ellipse_equation x₁ y₁ ∧ ellipse_equation x₂ y₂ ∧ 
    A = (x₁, y₁) ∧ B = (x₂, y₂)

-- The fixed point on the coordinate plane
def fixed_point_T : ℝ × ℝ := (1, 0)

-- Statement to be proven
theorem ellipse_properties :
  (∀ x y : ℝ, ellipse_equation x y → (abs (x - point_P.1) + abs (y - point_P.2)) < 2 * Real.sqrt 2) ∧
  (∀ (l : ℝ → ℝ) A B, intersects_at_A_B l A B → 
    ∃ T : ℝ × ℝ, T = fixed_point_T ∧ 
    ∀ (A B : ℝ × ℝ), 
    let circle_diameter_AB := (dist A T = dist B T) in 
    circle_diameter_AB = true) :=
by 
  sorry

end ellipse_properties_l533_533208


namespace problem1_problem2_l533_533673

variables (x : ℝ)
def a : ℝ × ℝ := (1, x)
def b : ℝ × ℝ := (2 * x + 3, -x)

-- Prove that if the vectors are parallel, then the norm of their difference is either 2 or 2√5
theorem problem1 (h_parallel: ∃ k : ℝ, b = (k * fst a, k * snd a)) :
  ‖a - b‖ = 2 ∨ ‖a - b‖ = 2 * Real.sqrt 5 := by
  sorry

-- Prove that if the angle between the vectors is acute, then x is in the interval (-1, 0) ∪ (0, 3)
theorem problem2 (h_acute : 0 < (fst a) * (fst b) + (snd a) * (snd b)) :
  x ∈ Ioo (-1 : ℝ) 0 ∨ x ∈ Ioo 0 3 := by
  sorry

end problem1_problem2_l533_533673


namespace derivative_sin_2x_l533_533864

variable (x : ℝ)

theorem derivative_sin_2x : (derivative (λ x, Real.sin (2 * x))) x = 2 * Real.cos (2 * x) :=
by sorry

end derivative_sin_2x_l533_533864


namespace athlete_B_has_highest_speed_l533_533458

-- Define the basic conditions as assumptions
variables (d_A t_A d_B t_B d_C t_C : ℝ)
variables (dA_eq : d_A = 400) (tA_eq : t_A = 56)
variables (dB_eq : d_B = 600) (tB_eq : t_B = 80)
variables (dC_eq : d_C = 800) (tC_eq : t_C = 112)

-- Define the speeds based on the conditions
noncomputable def speed (d t : ℝ) : ℝ := d / t

def speed_A := speed d_A t_A
def speed_B := speed d_B t_B
def speed_C := speed d_C t_C

-- Set the speeds for comparison
theorem athlete_B_has_highest_speed :
  d_A = 400 ∧ t_A = 56 ∧
  d_B = 600 ∧ t_B = 80 ∧
  d_C = 800 ∧ t_C = 112 →
  speed_B > speed_A ∧ speed_B > speed_C :=
by
  intros h
  rw [dA_eq, tA_eq, dB_eq, tB_eq, dC_eq, tC_eq] at h
  have h_speed_eq : speed_A = 400 / 56 ∧ speed_B = 600 / 80 ∧ speed_C = 800 / 112 := 
      by simp [speed, dA_eq, tA_eq, dB_eq, tB_eq, dC_eq, tC_eq];
  have h_speed_values : speed_A ≈ 7.14 ∧ speed_B = 7.5 ∧ speed_C ≈ 7.14 := 
      by norm_num;
  exact h


end athlete_B_has_highest_speed_l533_533458


namespace smallest_marked_cells_l533_533151

theorem smallest_marked_cells (k : ℕ) : ∃ k = 48, ∀ (marking : ℕ × ℕ → Prop), 
  (∀ i j, 0 ≤ i < 12 → 0 ≤ j < 12 → marking (i, j)) → 
  (∃ i j, 0 ≤ i < 12 → 0 ≤ j < 12 → 
    (marking (i, j)) → 
    (∀ r s, 0 ≤ r < 2 → 0 ≤ s < 2 → 
      (i + r, j + s) ∈ { (i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1) }) → 
    ∃ marked_cell (i + r, j + s) ) := 
sorry

end smallest_marked_cells_l533_533151


namespace max_projection_area_of_rectangular_prism_l533_533845

theorem max_projection_area_of_rectangular_prism 
  (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 12) : 
  ∃ max_area : ℝ, max_area = 12 * Real.sqrt 26 :=
by
  use 12 * Real.sqrt 26
  sorry

end max_projection_area_of_rectangular_prism_l533_533845


namespace slices_with_only_mushrooms_l533_533874

theorem slices_with_only_mushrooms :
  ∀ (T P M n : ℕ),
    T = 16 →
    P = 9 →
    M = 12 →
    (9 - n) + (12 - n) + n = 16 →
    M - n = 7 :=
by
  intros T P M n hT hP hM h_eq
  sorry

end slices_with_only_mushrooms_l533_533874


namespace basketball_distribution_count_l533_533231

-- Definitions related to the problem
def basketballs : List Nat := [1, 2, 3, 4]

def kids : List Nat := [1, 2, 3]

def all_distributions : List (Nat × Nat × Nat × Nat) := 
  List.product (List.product kids kids) (List.product kids kids)

def valid_distribution (d: Nat × Nat × Nat × Nat) : Bool :=
  let kid1, kid2, kid3, kid4 := d
  (kid1 ≠ kid2) && -- Condition that basketballs 1 and 2 aren't with the same kid
  List.count kids kid1 + List.count kids kid2 + List.count kids kid3 + List.count kids kid4 = 4 && -- Ensure a basketball is distributed to each kid
  kid1 ≠ kid3 && kid1 ≠ kid4 && kid2 ≠ kid3 && kid2 ≠ kid4 -- Sideline to ensure the uniqeness among given basket balls to kids

def valid_distributions : List (Nat × Nat × Nat × Nat) :=
  all_distributions.filter valid_distribution

-- The theorem to prove
theorem basketball_distribution_count: valid_distributions.length = 30 := by
  sorry

end basketball_distribution_count_l533_533231


namespace complex_conjugate_and_modulus_l533_533629

theorem complex_conjugate_and_modulus (z : ℂ) (h : z = 4 + 3 * complex.I) : 
  (conj z / abs z) = (4 / 5 : ℂ) - (3 / 5) * complex.I :=
by
  rw h
  sorry

end complex_conjugate_and_modulus_l533_533629


namespace circle_midpoint_distance_sum_l533_533255

theorem circle_midpoint_distance_sum 
  (A B C D P Q R : Type)
  (rA rB rC rD : Real)
  (h1 : ¬(∃ x, rA = rB x))
  (h2 : ¬(∃ y, rB = rC y))
  (h3 : ¬(∃ z, rC = rD z))
  (h4 : P ∈ Circle rA ∧ P ∈ Circle rB ∧ P ∈ Circle rC ∧ P ∈ Circle rD)
  (h5 : Q ∈ Circle rA ∧ Q ∈ Circle rB ∧ Q ∈ Circle rC ∧ Q ∈ Circle rD)
  (h6 : rA = (4 / 7) * rB)
  (h7 : rC = (4 / 7) * rD)
  (h8 : dist A B = 42)
  (h9 : dist C D = 42)
  (h10 : dist P Q = 56)
  (h11 : R = midpoint P Q) :
  dist A R + dist B R + dist C R + dist D R = 168 :=
sorry

end circle_midpoint_distance_sum_l533_533255


namespace simplest_quadratic_radical_is_sqrt5_l533_533851

def is_simplest_quadratic_radical (x : ℚ) : Prop :=
  (∀ (y : ℚ), (y = sqrt 1.5 ∨ y = sqrt (1/3) ∨ y = sqrt 8 ∨ y = sqrt 5) → (sqrt x) ≤ (sqrt y)) ∧ 
  (x = 5)

theorem simplest_quadratic_radical_is_sqrt5 :
  is_simplest_quadratic_radical 5 :=
by
  /- The proof will go here -/
  sorry

end simplest_quadratic_radical_is_sqrt5_l533_533851


namespace second_hand_distance_l533_533824

variable (r : Real) (t : Real)

def linear_distance (r : Real) (t : Real) : Real :=
  2 * Real.pi * r * t

def angular_distance (t : Real) : Real :=
  360 * t

theorem second_hand_distance
  (h_r : r = 8)
  (h_t : t = 45) :
  linear_distance r t = 720 * Real.pi ∧ angular_distance t = 16200 :=
by
  sorry

end second_hand_distance_l533_533824


namespace solution_set_of_inequality_l533_533102

theorem solution_set_of_inequality (x : ℝ) : (3 / x < 1) ↔ (x ∈ set.Iio 0 ∪ set.Ioi 3) :=
sorry

end solution_set_of_inequality_l533_533102


namespace sum_of_all_positive_integer_solutions_l533_533486

theorem sum_of_all_positive_integer_solutions :
  (∑ x in {x | 7 * (5 * x - 3) % 10 = 4 ∧ x ≤ 30 ∧ 0 < x}.toFinset) = 225 :=
by
  sorry

end sum_of_all_positive_integer_solutions_l533_533486


namespace find_x_l533_533270

noncomputable def S (x : ℝ) : list ℝ := 
  (list.range 51).map (λ n, x^n)

noncomputable def A : list ℝ → list ℝ
| [] := []
| [a] := []
| (a1 :: a2 :: rest) := ((a1 + a2) / 2) :: A (a2 :: rest)

noncomputable def A_m (m : ℕ) (S : list ℝ) : list ℝ :=
nat.rec_on m S (λ m IH, A IH)

theorem find_x (x : ℝ) (h1 : x > 0) 
  (h2 : A_m 50 (S x) = [1 / 2^25]) : 
  x = real.sqrt 2 - 1 := 
sorry

end find_x_l533_533270


namespace limit_sequence_l533_533575

open Filter
open Real

noncomputable def sequence_limit := 
  filter.tendsto (λ n : ℕ, (sqrt (3 * n - 1) - real.cbrt (125 * n^3 + n)) / (real.rpow n (1 / 5) - n)) at_top (nhds 5)

theorem limit_sequence: sequence_limit :=
  sorry

end limit_sequence_l533_533575


namespace disjoint_subsets_mod_l533_533736

open Finset

def S : Finset ℕ := (range 12).map (λ x, x + 1)

noncomputable def number_of_disjoint_subsets (S : Finset ℕ) : ℕ :=
  (3^12 - 2 * (2^12) + 1) / 2

theorem disjoint_subsets_mod :
  (number_of_disjoint_subsets S) % 1000 = 625 := by
  sorry

end disjoint_subsets_mod_l533_533736


namespace total_price_of_houses_l533_533724

theorem total_price_of_houses (price_first price_second total_price : ℝ)
    (h1 : price_first = 200000)
    (h2 : price_second = 2 * price_first)
    (h3 : total_price = price_first + price_second) :
  total_price = 600000 := by
  sorry

end total_price_of_houses_l533_533724


namespace solve_for_a_l533_533687

theorem solve_for_a (a : ℂ) (h : (2 - complex.i) * (a + 2 * complex.i)).re = 0) : a = -1 :=
sorry

end solve_for_a_l533_533687


namespace probability_line_circle_intersection_l533_533638

def line_intersects_circle (a : ℝ) : Prop :=
  ∃ x y : ℝ, (x + y + a = 0) ∧ ((x - 1) ^ 2 + (y + 2) ^ 2 = 2)

def probability_interval (a : ℝ) (I : set ℝ) : ℚ :=
  if I a then (4 : ℚ) / 10 else 0

theorem probability_line_circle_intersection :
  (measure_of (set_of (λ (a : ℝ), line_intersects_circle a)) (set.Icc (-5 : ℝ) (5 : ℝ)) = (2 : ℚ) / 5) :=
sorry


end probability_line_circle_intersection_l533_533638


namespace song_configuration_count_l533_533206

theorem song_configuration_count :
  let S := {1, 2, 3, 4, 5} -- the set of five different songs
  let AB := {s ∈ S | liked_by_Amy s ∧ liked_by_Beth s ∧ ¬liked_by_Jo s}
  let BC := {s ∈ S | liked_by_Beth s ∧ liked_by_Jo s ∧ ¬liked_by_Amy s}
  let CA := {s ∈ S | liked_by_Jo s ∧ liked_by_Amy s ∧ ¬liked_by_Beth s}
  let A := {s ∈ S | liked_by_Amy s ∧ ¬liked_by_Beth s ∧ ¬liked_by_Jo s}
  let B := {s ∈ S | liked_by_Beth s ∧ ¬liked_by_Amy s ∧ ¬liked_by_Jo s}
  let C := {s ∈ S | liked_by_Jo s ∧ ¬liked_by_Amy s ∧ ¬liked_by_Beth s}
  let N := {s ∈ S | ¬liked_by_Amy s ∧ ¬liked_by_Beth s ∧ ¬liked_by_Jo s}
  ∀ s₁ s₂ s₃ s₄, s₁ ∈ AB ∧ s₂ ∈ BC ∧ s₃ ∈ CA ∧ s₄ ∈ {s ∈ S | s ∉ AB ∧ s ∉ BC ∧ s ∉ CA}
  ∃ s₅ ∈ S, s₅ ∈ S :=
  168 := sorry

end song_configuration_count_l533_533206


namespace domain_f_log_base_one_half_l533_533655

-- Definitions directly from the problem conditions
def domain_f (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 4

-- The mathematically equivalent proof statement
theorem domain_f_log_base_one_half (x : ℝ) (hf : ∀ x, domain_f x → (domain_f x)) :
  domain_f (log (1/2) x) ↔ (1/16 ≤ x ∧ x ≤ 1/4) :=
sorry

end domain_f_log_base_one_half_l533_533655


namespace relationship_a_plus_b_greater_c_relationship_a_squared_plus_b_squared_equals_c_squared_relationship_a_n_plus_b_n_less_than_c_n_l533_533738

-- Let a, b, and c be the sides of a right triangle with c as the hypotenuse.
variables (a b c : ℝ) (n : ℕ)

-- Assume the triangle is a right triangle
-- and assume n is a positive integer.
axiom right_triangle : a^2 + b^2 = c^2
axiom positive_integer : n > 0 

-- The relationships we need to prove:
theorem relationship_a_plus_b_greater_c : n = 1 → a + b > c := sorry
theorem relationship_a_squared_plus_b_squared_equals_c_squared : n = 2 → a^2 + b^2 = c^2 := sorry
theorem relationship_a_n_plus_b_n_less_than_c_n : n ≥ 3 → a^n + b^n < c^n := sorry

end relationship_a_plus_b_greater_c_relationship_a_squared_plus_b_squared_equals_c_squared_relationship_a_n_plus_b_n_less_than_c_n_l533_533738


namespace find_numbers_with_property_l533_533541

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, d > 1 ∧ d < n ∧ n % d = 0) (List.range n)

lemma satisfies_condition (n : ℕ) : Prop :=
  ∀ (d1 d2 ∈ proper_divisors n), n % (d1 - d2).natAbs = 0

theorem find_numbers_with_property :
  {n | satisfies_condition n ∧ 2 < (proper_divisors n).length} = {6, 8, 12} :=
by
  sorry

end find_numbers_with_property_l533_533541


namespace problem_condition_necessary_and_sufficient_l533_533311

theorem problem_condition_necessary_and_sufficient (a b : ℝ) (h : a * b > 0) :
  (a > b) ↔ (1 / a < 1 / b) :=
sorry

end problem_condition_necessary_and_sufficient_l533_533311


namespace outfit_combinations_l533_533420

section
variables (shirts ties pants belts : ℕ)
variables (wearable_belts_with_tie : ℕ)
variables (shirt_choices pants_choices : ℕ)
variables (tie_choices_no_tie tie_choices_with_tie : ℕ)
variables (belt_choices_with_no_tie belt_choices_with_tie : ℕ)

def total_outfits : ℕ :=
  shirts * pants * belt_choices_with_no_tie + shirts * pants * ties * belt_choices_with_tie

theorem outfit_combinations : 
  shirts = 8 → 
  ties = 5 → 
  pants = 3 → 
  belts = 4 → 
  wearable_belts_with_tie = 2 → 
  belt_choices_with_no_tie = 5 → 
  belt_choices_with_tie = 3 → 
  ties + 1 = 6 → 
  total_outfits shirts ties pants belt_choices_with_no_tie belts belt_choices_with_tie = 480 :=
by sorry
end

end outfit_combinations_l533_533420


namespace best_fit_line_slope_l533_533502

theorem best_fit_line_slope (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) (d : ℝ) 
  (h1 : x2 - x1 = 2 * d) (h2 : x3 - x2 = 3 * d) (h3 : x4 - x3 = d) : 
  ((y4 - y1) / (x4 - x1)) = (y4 - y1) / (x4 - x1) :=
by
  sorry

end best_fit_line_slope_l533_533502


namespace find_phi_l533_533324

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x)
noncomputable def g (x ϕ : ℝ) : ℝ := 2 * Real.sin (2 * (x - ϕ))

theorem find_phi 
  (ϕ : ℝ) 
  (hϕ1 : 0 < ϕ) 
  (hϕ2 : ϕ < (Real.pi / 2))
  (min_abs_diff_x1_x2 : ∀ x₁ x₂ : ℝ, |f x₁ - g x₂ ϕ| = 4 → |x₁ - x₂| = Real.pi / 6) :
  ϕ = Real.pi / 3 :=
begin
  sorry
end

end find_phi_l533_533324


namespace triangle_sides_inequality_l533_533625

theorem triangle_sides_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^2 - 2 * a * b + b^2 - c^2 < 0 :=
by
  sorry

end triangle_sides_inequality_l533_533625


namespace find_second_divisor_l533_533519

theorem find_second_divisor :
  ∃ x : ℕ, 377 / 13 / x * (1/4 : ℚ) / 2 = 0.125 ∧ x = 29 :=
by
  use 29
  -- Proof steps would go here
  sorry

end find_second_divisor_l533_533519


namespace gazprom_rnd_costs_calc_l533_533933

theorem gazprom_rnd_costs_calc (R_D_t ΔAPL_t1 : ℝ) (h1 : R_D_t = 3157.61) (h2 : ΔAPL_t1 = 0.69) :
  R_D_t / ΔAPL_t1 = 4576 :=
by
  sorry

end gazprom_rnd_costs_calc_l533_533933


namespace sum_common_ratios_l533_533741

variable (k p r : ℝ)
variable (hp : p ≠ r)

theorem sum_common_ratios (h : k * p ^ 2 - k * r ^ 2 = 2 * (k * p - k * r)) : 
  p + r = 2 := by
  have hk : k ≠ 0 := sorry -- From the nonconstancy condition
  sorry

end sum_common_ratios_l533_533741


namespace total_candies_options_l533_533598

def numberOfCandies (vitya : ℕ) (masha : ℕ) (sasha : ℕ) : ℕ :=
  vitya + masha + sasha

theorem total_candies_options :
  ∃ (masha : ℕ), masha < 5 ∧ 
  ∃ (total : ℕ), total = numberOfCandies 5 masha (5 + masha) ∧ 
    (total = 18 ∨ total = 16 ∨ total = 14 ∨ total = 12) :=
begin
  sorry
end

end total_candies_options_l533_533598


namespace sphere_radius_in_truncated_cone_l533_533913

theorem sphere_radius_in_truncated_cone (r1 r2 : ℝ) (cone : TruncatedCone r1 r2) (h_tangent : TangentSphere cone) :
  r1 = 12 ∧ r2 = 3 → sphere_radius h_tangent = 6 :=
by
  sorry

end sphere_radius_in_truncated_cone_l533_533913


namespace grasshoppers_can_jump_left_l533_533771

noncomputable def grasshoppers_initial_positions (n : ℕ) : list ℕ := sorry
noncomputable def right_jump (pos : ℕ) (dist : ℕ) : ℕ := pos + dist
noncomputable def left_jump (pos : ℕ) (dist : ℕ) : ℕ := pos - dist

theorem grasshoppers_can_jump_left (n : ℕ) (h : n = 2019) 
    (initial_positions : list ℕ) :
    (∃ i j, i ≠ j ∧ abs (initial_positions[i] - initial_positions[j]) = 1) →
    (∃ k l, k ≠ l ∧ abs (left_jump (initial_positions[k]) 2019 - 
                         left_jump (initial_positions[l]) 2019) = 1) :=
sorry

end grasshoppers_can_jump_left_l533_533771


namespace f_neg4_plus_f_0_range_of_a_l533_533657

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else if x < 0 then -log (-x) / log 2 else 0

/- Prove that f(-4) + f(0) = -2 given the function properties -/
theorem f_neg4_plus_f_0 : f (-4) + f 0 = -2 :=
sorry

/- Prove the range of a such that f(a) > f(-a) is a > 1 or -1 < a < 0 given the function properties -/
theorem range_of_a (a : ℝ) : f a > f (-a) ↔ a > 1 ∨ (-1 < a ∧ a < 0) :=
sorry

end f_neg4_plus_f_0_range_of_a_l533_533657


namespace subsequence_geometric_progression_iff_l533_533986

theorem subsequence_geometric_progression_iff (a d : ℤ) (h : d ≠ 0) : 
  (∃ (f : ℕ → ℕ), ∀ n, a + f(n) * d = a * (1 + f(1)) ^ n) ↔ ∃ (s r : ℕ), a = s * d ∧ d = r * b := 
sorry

end subsequence_geometric_progression_iff_l533_533986


namespace Geoff_can_align_windmills_l533_533553

-- Definitions for our problem
def Windmill (Point : Type) := { l : set Point × set Point | (l.1.card = 1 ∧ l.2.card = 1) ∧ dist l.1 l.2 = 1 }
def Configuration (Point : Type) := { w : set (Windmill Point) | ∀ w1 w2 ∈ w, w1 ≠ w2 → disjoint w1.2 w2.2 }
def PivotsMoreThanSqrtTwoApart (Point : Type) (ps : set Point) :=
  ∀ (p1 p2 ∈ ps), p1 ≠ p2 → dist p1 p2 > real.sqrt 2

-- Statement of the problem in Lean
theorem Geoff_can_align_windmills (Point : Type) (ps : set Point) (w : set (Windmill Point)) :
  finite w ∧ Configuration w ∧ PivotsMoreThanSqrtTwoApart ps → ∃ n ∈ ℕ, ∃ rotations : fin n → Windmill Point, 
  (∀ i < n, Configuration (rotate w (rotations i))) :=
sorry

end Geoff_can_align_windmills_l533_533553


namespace geometric_sequence_fifth_term_l533_533844

theorem geometric_sequence_fifth_term (a₁ r : ℤ) (n : ℕ) (h_a₁ : a₁ = 5) (h_r : r = -2) (h_n : n = 5) :
  (a₁ * r^(n-1) = 80) :=
by
  rw [h_a₁, h_r, h_n]
  sorry

end geometric_sequence_fifth_term_l533_533844


namespace distinct_pos_numbers_implies_not_zero_at_least_one_of_abc_impossible_for_all_neq_l533_533250

noncomputable section

variables (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) (h1 : 0 < a) 
(h2 : 0 < b) (h3 : 0 < c)

theorem distinct_pos_numbers_implies_not_zero :
  (a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2 ≠ 0 :=
sorry

theorem at_least_one_of_abc :
  a > b ∨ a < b ∨ a = b :=
sorry

theorem impossible_for_all_neq :
  ¬(a ≠ c ∧ b ≠ c ∧ a ≠ b) :=
sorry

end distinct_pos_numbers_implies_not_zero_at_least_one_of_abc_impossible_for_all_neq_l533_533250


namespace original_cost_price_l533_533540

noncomputable def original_cost (final_price : ℝ) : ℝ :=
  final_price / 1.134

theorem original_cost_price (final_price : ℝ) (h : final_price = 67320) :
  original_cost final_price = 59366.14 :=
by
  rw [h]
  norm_num

#eval original_cost 67320 -- This should evaluate to 59366.14

end original_cost_price_l533_533540


namespace total_cost_for_doughnuts_l533_533929

theorem total_cost_for_doughnuts
  (num_students : ℕ)
  (num_chocolate : ℕ)
  (num_glazed : ℕ)
  (price_chocolate : ℕ)
  (price_glazed : ℕ)
  (H1 : num_students = 25)
  (H2 : num_chocolate = 10)
  (H3 : num_glazed = 15)
  (H4 : price_chocolate = 2)
  (H5 : price_glazed = 1) :
  num_chocolate * price_chocolate + num_glazed * price_glazed = 35 :=
by
  -- Proof steps would go here
  sorry

end total_cost_for_doughnuts_l533_533929


namespace horner_evaluation_l533_533125

def f (x : ℝ) := x^5 + 3 * x^4 - 5 * x^3 + 7 * x^2 - 9 * x + 11

theorem horner_evaluation : f 4 = 1559 := by
  sorry

end horner_evaluation_l533_533125


namespace smallest_marked_cells_l533_533150

theorem smallest_marked_cells (k : ℕ) : ∃ k = 48, ∀ (marking : ℕ × ℕ → Prop), 
  (∀ i j, 0 ≤ i < 12 → 0 ≤ j < 12 → marking (i, j)) → 
  (∃ i j, 0 ≤ i < 12 → 0 ≤ j < 12 → 
    (marking (i, j)) → 
    (∀ r s, 0 ≤ r < 2 → 0 ≤ s < 2 → 
      (i + r, j + s) ∈ { (i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1) }) → 
    ∃ marked_cell (i + r, j + s) ) := 
sorry

end smallest_marked_cells_l533_533150


namespace probability_of_exactly_6_heads_in_12_flips_l533_533129

theorem probability_of_exactly_6_heads_in_12_flips :
  (number_of_ways : ℕ // number_of_ways = Nat.choose 12 6) /
  (total_outcomes : ℕ // total_outcomes = 2 ^ 12) = (231 : ℚ) / 1024 := by
sorry

end probability_of_exactly_6_heads_in_12_flips_l533_533129


namespace worm_length_difference_l533_533795

theorem worm_length_difference
  (worm1 worm2: ℝ)
  (h_worm1: worm1 = 0.8)
  (h_worm2: worm2 = 0.1) :
  worm1 - worm2 = 0.7 :=
by
  -- starting the proof
  sorry

end worm_length_difference_l533_533795


namespace perimeter_of_star_is_160_l533_533983

-- Define the radius of the circles
def radius := 5 -- in cm

-- Define the diameter based on radius
def diameter := 2 * radius

-- Define the side length of the square
def side_length_square := 2 * diameter

-- Define the side length of each equilateral triangle
def side_length_triangle := side_length_square

-- Define the perimeter of the four-pointed star
def perimeter_star := 8 * side_length_triangle

-- Statement: The perimeter of the star is 160 cm
theorem perimeter_of_star_is_160 :
  perimeter_star = 160 := by
    sorry

end perimeter_of_star_is_160_l533_533983


namespace general_formula_sum_of_na_n_l533_533978

-- Definition of a positive geometric sequence
def is_positive_geometric_sequence {α : Type*} [linear_ordered_field α] (a : ℕ → α) (q : α) : Prop :=
  (∀ n, a n = a 0 * q ^ n) ∧ (a 0 > 0) ∧ (q > 0)

-- Given conditions and problem statement
def problem_statement (a : ℕ → ℝ) (S : ℕ → ℝ): Prop :=
  let q := 2 in
  let a_5 := 32 in
  S 6 - S 3 = 7 * a 4 ∧ 
  a 5 = a 0 * (q^4) ∧
  a 0 * (q ^ 4) = a_5 ∧
  a 0 > 0 ∧
  q > 0

-- Proving the general formula for the sequence is correct
theorem general_formula (a : ℕ → ℝ) (S : ℕ → ℝ): 
  problem_statement a S →
  ∀ n, a n = 2 ^ n :=
by
  sorry

-- Definition of sum T_n
def T (n : ℕ) : ℝ := Σ (i : ℕ) in finset.range n, (i + 1 : ℝ) * 2^(i + 1)

-- Proving T_n equals the given formula
theorem sum_of_na_n (n : ℕ) : T n = (n - 1 : ℝ) * 2^(n + 1) + 2 :=
by
  sorry

end general_formula_sum_of_na_n_l533_533978


namespace sum_of_products_of_non_empty_subsets_l533_533442

def M : Finset ℤ := {1, 99, -1, 0, 25, -36, -91, 19, -2, 11}

def non_empty_subsets (s : Finset ℤ) : Finset (Finset ℤ) :=
  s.powerset.filter (λ x => x ≠ ∅)

def product (s : Finset ℤ) : ℤ :=
  s.fold (*) 1 id

theorem sum_of_products_of_non_empty_subsets :
  ∑ s in non_empty_subsets M, product s = -1 :=
by
  sorry

end sum_of_products_of_non_empty_subsets_l533_533442


namespace no_three_common_tangents_l533_533463

-- Definitions and conditions based on the problem statement
def in_same_plane (C1 C2 : Circle) : Prop := 
  C1.center.z = 0 ∧ C2.center.z = 0

def different_radii (C1 C2 : Circle) : Prop := 
  C1.radius ≠ C2.radius

-- Main theorem statement translating the problem's solution
theorem no_three_common_tangents (C1 C2 : Circle) (h1 : in_same_plane C1 C2) (h2 : different_radii C1 C2) : 
  ¬ (number_of_common_tangents C1 C2 = 3) :=
  sorry

end no_three_common_tangents_l533_533463


namespace log_a_domain_range_l533_533323

def f (a : ℝ) (x : ℝ) := log a (x + 1)

theorem log_a_domain_range (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) 
  (h_domain : ∀ x, 0 ≤ x ∧ x ≤ 1 → f a x ≥ 0 ∧ f a x ≤ 1)
  (h_range : ∀ y, 0 ≤ y ∧ y ≤ 1 → ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f a x = y ) :
  a = 2 :=
by
  sorry

end log_a_domain_range_l533_533323


namespace age_is_nine_l533_533836

-- Define the conditions
def current_age (X : ℕ) :=
  X = 3 * (X - 6)

-- The theorem: Prove that the age X is equal to 9 under the conditions given
theorem age_is_nine (X : ℕ) (h : current_age X) : X = 9 :=
by
  -- The proof is omitted
  sorry

end age_is_nine_l533_533836


namespace tangent_line_at_point_pi_l533_533085

noncomputable def tangent_line_equation (x : ℝ) : ℝ :=
  -π * x + π^2

theorem tangent_line_at_point_pi (y : ℝ) :
  y = x * Real.sin x → ∀ x = π, y = 0 → 
  tangent_line_equation x = -π * x + π^2 :=
by
  intro h₁ h₂
  rw [tangent_line_equation]
  sorry

end tangent_line_at_point_pi_l533_533085


namespace min_value_expr_l533_533100

   theorem min_value_expr (x y : ℝ) 
     (h : x^2 - 2*x*y + 2*y^2 = 2) : x^2 + 2*y^2 = 4 - 2 * sqrt 2 :=
   sorry
   
end min_value_expr_l533_533100


namespace tournament_total_rankings_l533_533055

theorem tournament_total_rankings : 
  (∃ E F A B C : Prop, 
    (E ∨ F) ∧  -- Friday match winner
    ((A ∨ B) ∧ (C ∨ (E ∨ F))) ∧  -- Saturday matches
    (((A ∨ C) ∨ (A ∨ (E ∨ F)) ∨ (B ∨ C) ∨ (B ∨ (E ∨ F))) ∧  -- Sunday matches for 1st and 2nd place
     ((B ∧ (E ∨ F)) ∨ 3rd and 4th place loser scenario)) ∧ -- Sunday matches for 3rd and 4th place
    total_possible_rankings = 16)  -- Total number of ranking sequences
  sorry

end tournament_total_rankings_l533_533055


namespace minimum_marked_cells_k_l533_533141

def cell := (ℕ × ℕ)

def is_marked (board : set cell) (cell : cell) : Prop :=
  cell ∈ board

def touches_marked_cell (board : set cell) (fig : set cell) : Prop :=
  ∃ (c : cell), c ∈ fig ∧ is_marked(board, c)

def four_cell_figures (fig : set cell) : Prop :=
  (fig = { (0, 0), (0, 1), (1, 0), (1, 1) }) ∨
  (fig = { (0, 0), (1, 0), (2, 0), (3, 0) }) ∨
  (fig = { (0, 0), (0, 1), (0, 2), (0, 3) }) ∨
  -- Add other rotations and flipped configurations if needed

def board12x12 :=
  { (i, j) | i < 12 ∧ j < 12 }

theorem minimum_marked_cells_k :
  ∃ (k : ℕ) (marked_cells : (fin 144) → bool),
  k = 48 ∧
  (∀ (fig : set cell),
    four_cell_figures fig →
    ∀ (x y : ℕ), x < 12 ∧ y < 12 →
      touches_marked_cell
        { cell | marked_cells (fin.mk (cell.1 * 12 + cell.2) _) }
        (image (λ (rc : cell), (rc.1 + x, rc.2 + y)) fig)
  ) :=
begin
  sorry
end

end minimum_marked_cells_k_l533_533141


namespace slower_plane_speed_l533_533120

-- Let's define the initial conditions and state the theorem in Lean 4
theorem slower_plane_speed 
    (x : ℕ) -- speed of the slower plane
    (h1 : x + 2*x = 900) : -- based on the total distance after 3 hours
    x = 300 :=
by
    -- Proof goes here
    sorry

end slower_plane_speed_l533_533120


namespace total_shaded_area_l533_533354

variable (r : ℝ) (R : ℝ)

def area_of_circle (r : ℝ) : ℝ := π * r^2

theorem total_shaded_area 
  (h1 : area_of_circle R = 100 * π)
  (h2 : r = (1 / 2) * R / 2)
  : area_of_circle R / 2 + area_of_circle r / 2 = 53.125 * π := 
by 
  -- Verify the input conditions and solve the proof here
  sorry

end total_shaded_area_l533_533354


namespace log_expression_evaluation_l533_533516

noncomputable def log2 : ℝ := Real.log 2
noncomputable def log5 : ℝ := Real.log 5

theorem log_expression_evaluation (condition : log2 + log5 = 1) :
  log2^2 + log2 * log5 + log5 - (Real.sqrt 2 - 1)^0 = 0 :=
by
  sorry

end log_expression_evaluation_l533_533516


namespace symmetric_point_of_P_l533_533247

-- Let P be a point with coordinates (5, -3)
def P : ℝ × ℝ := (5, -3)

-- Definition of the symmetric point with respect to the x-axis
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Theorem stating that the symmetric point to P with respect to the x-axis is (5, 3)
theorem symmetric_point_of_P : symmetric_point P = (5, 3) := 
  sorry

end symmetric_point_of_P_l533_533247


namespace largest_square_side_length_largest_rectangle_dimensions_l533_533996

variable (a b : ℝ) (h : a > 0) (k : b > 0)

-- Part (a): Side length of the largest possible square
theorem largest_square_side_length (h : a > 0) (k : b > 0) :
  ∃ (s : ℝ), s = (a * b) / (a + b) := sorry

-- Part (b): Dimensions of the largest possible rectangle
theorem largest_rectangle_dimensions (h : a > 0) (k : b > 0) :
  ∃ (x y : ℝ), x = a / 2 ∧ y = b / 2 := sorry

end largest_square_side_length_largest_rectangle_dimensions_l533_533996


namespace find_f_of_lg_lg_3_l533_533651

noncomputable def f (x : Real) (a b : Real) := a * Real.sin x + b * (Real.cbrt x) + 4

theorem find_f_of_lg_lg_3 (a b : Real) (h : f (Real.log10 (Real.log 3 10)) a b = 5) :
  f (Real.log10 (Real.log10 3)) a b = 3 :=
sorry

end find_f_of_lg_lg_3_l533_533651


namespace correct_understanding_l533_533797

-- Definitions based on conditions
def probability_of_rain (p : ℚ) := p = 0.7
def high_possibility_of_rain_if_no_gear (p : ℚ) := p > 0.5

-- Statement to prove
theorem correct_understanding (p : ℚ) (h : probability_of_rain p) : high_possibility_of_rain_if_no_gear p :=
by {
  unfold probability_of_rain at h,
  unfold high_possibility_of_rain_if_no_gear,
  rw h,
  norm_num,
}

end correct_understanding_l533_533797


namespace parabola_and_bisector_y_intercept_l533_533654

-- Definitions for the problem conditions
def line_passing_point_with_slope (x₁ y₁: ℝ) (m: ℝ) := ∃ b: ℝ, ∀ x, y, y = m * x + b ↔ y = m * x + (y₁ - m * x₁)

def parabola (p: ℝ) (h: p > 0) := ∀ x y, x^2 = 2 * p * y

def midpoint_x_coordinate (x₁ x₂: ℝ) (m: ℝ) := (x₁ + x₂) / 2 = m

-- The main theorem to prove
theorem parabola_and_bisector_y_intercept (p: ℝ) (hp: p > 0) 
  (A : ℝ × ℝ) (hA : A = (-4, 0))
  (h_slope : line_passing_point_with_slope (-4) 0 (1/2))
  (h_intersection : ∃ B C : ℝ × ℝ, ∀ l, line_passing_point_with_slope (-4) 0 (1/2) → 
    parabola p hp → l = B ∧ l = C)
  (h_midpoint : midpoint_x_coordinate (B : ℝ × ℝ) (C : ℝ × ℝ) 1) :
  parabola 2 (by norm_num) ↔ 
  ∃ b : ℝ, (∃ MBC : ℝ × ℝ, MBC = (1, 5/2)) ∧ 
  ∀ y₀, y₀ = 5/2 → line_passing_point_with_slope 1 y₀ (-2) → b = 9/2 := sorry

end parabola_and_bisector_y_intercept_l533_533654


namespace rectangle_area_unchanged_l533_533425

theorem rectangle_area_unchanged (x y : ℕ) (h1 : x * y = (x + 5/2) * (y - 2/3)) (h2 : x * y = (x - 5/2) * (y + 4/3)) : x * y = 20 :=
by
  sorry

end rectangle_area_unchanged_l533_533425


namespace fraction_subtraction_l533_533935

theorem fraction_subtraction (x y : ℝ) (h : x ≠ y) : (x + y) / (x - y) - (2 * y) / (x - y) = 1 := by
  sorry

end fraction_subtraction_l533_533935


namespace number_of_correct_judgments_is_2_l533_533919

theorem number_of_correct_judgments_is_2 :
  ¬ (∀ (a m n : ℕ), (a^m)^n = a^(m+n)) ∧
  (∀ x : ℝ, (1 + Real.exp x) > (1 + Real.exp (x - 1))) ∧
  ¬ (∀ a b c : ℝ, (b^2 = 4 * a * c) ↔ (a * x^2 + b * x + c = 0 ∧ ∃ x : ℝ, ax^2 + bx + c = 0)) ∧
  (∀ x : ℝ, Real.log x = -Real.log x → x = 1) ↔ (∃ x : ℝ, Real.log x = -Real.log x) :=
begin
 sorry
end

#eval number_of_correct_judgments_is_2 -- this evaluates to true

end number_of_correct_judgments_is_2_l533_533919


namespace suit_cost_l533_533601

theorem suit_cost (S : ℝ) 
  (h_commission : ∀ (x : ℝ), 0.15 * x) 
  (h_sales : 2 * 0.15 * S + 6 * 0.15 * 50 + 2 * 0.15 * 150 = 300) : 
  S = 700 := 
sorry

end suit_cost_l533_533601


namespace base_conversion_sum_correct_l533_533941

theorem base_conversion_sum_correct :
  (253 / 8 / 13 / 3 + 245 / 7 / 35 / 6 : ℚ) = 339 / 23 := sorry

end base_conversion_sum_correct_l533_533941


namespace find_k_l533_533675

def vector := (ℝ × ℝ)

def a : vector := (3, 1)
def b : vector := (1, 3)
def c (k : ℝ) : vector := (k, 2)

def subtract (v1 v2 : vector) : vector :=
  (v1.1 - v2.1, v1.2 - v2.2)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_k (k : ℝ) (h : dot_product (subtract a (c k)) b = 0) : k = 0 := by
  sorry

end find_k_l533_533675


namespace monomial_2015_l533_533765

def a (n : ℕ) : ℤ := (-1 : ℤ)^n * (2 * n - 1)

theorem monomial_2015 :
  a 2015 * (x : ℤ) ^ 2015 = -4029 * (x : ℤ) ^ 2015 :=
by
  sorry

end monomial_2015_l533_533765


namespace rahul_work_rate_l533_533782

variable (R : ℝ) 

def works_individually (rahul_rate : ℝ) (meena_rate : ℝ) (combined_rate : ℝ) : Prop :=
  rahul_rate = R ∧ meena_rate = 1/10 ∧ combined_rate = 3/10

theorem rahul_work_rate (R : ℝ) :
  works_individually (1 / R) (1 / 10) (3 / 10) → R = 5 :=
by
  intro h
  sorry

end rahul_work_rate_l533_533782


namespace shortest_path_equilateral_triangle_l533_533383
noncomputable def shortestPathLength (ABC : Triangle) (Γ : Circle) (O : Point) : Real :=
  sqrt (28 / 3) - 1

theorem shortest_path_equilateral_triangle :
  ∀ (ABC : Triangle) (Γ : Circle) (O : Point),
    ABC.is_equilateral → ABC.has_side_length 2 → Γ.center = O → Γ.radius = 1 / 2 →
    shortestPathLength ABC Γ O = sqrt (28 / 3) - 1 :=
begin
  intros ABC Γ O h1 h2 h3 h4,
  sorry,
end

end shortest_path_equilateral_triangle_l533_533383


namespace part1_part2_l533_533934

-- Problem part (1)
theorem part1 : (Real.sqrt 12 + Real.sqrt (4 / 3)) * Real.sqrt 3 = 8 := 
  sorry

-- Problem part (2)
theorem part2 : Real.sqrt 48 - Real.sqrt 54 / Real.sqrt 2 + (3 - Real.sqrt 3) * (3 + Real.sqrt 3) = Real.sqrt 3 + 6 := 
  sorry

end part1_part2_l533_533934


namespace bridget_profit_is_correct_l533_533573

noncomputable section

def bridget_profit : ℕ :=
  let loaves := 60
  let morning_loaves := loaves / 3
  let morning_revenue := morning_loaves * 3
  let afternoon_loaves := (loaves - morning_loaves) / 2
  let afternoon_revenue := afternoon_loaves * 1.5
  let remaining_afternoon_loaves := loaves - morning_loaves - afternoon_loaves
  let late_afternoon_loaves := (2 * remaining_afternoon_loaves) / 3
  let late_afternoon_revenue := late_afternoon_loaves * 1
  let evening_loaves := remaining_afternoon_loaves - late_afternoon_loaves
  let evening_revenue := evening_loaves * 1
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue + evening_revenue
  let total_cost := loaves * 1
  total_revenue - total_cost

theorem bridget_profit_is_correct : bridget_profit = 50 := by
  sorry

end bridget_profit_is_correct_l533_533573


namespace sin_HIO_value_l533_533695

noncomputable def sin_angle_HIO (A B C H I O: Point) (angleHIO: ℝ) : ℝ :=
  if (dist A B = 8) ∧ (dist B C = 13) ∧ (dist C A = 15)
  ∧ (is_orthocenter H A B C) ∧ (is_incenter I A B C) ∧ (is_circumcenter O A B C) then
    sin angleHIO
  else
    0

theorem sin_HIO_value (A B C H I O : Point) (angleHIO : ℝ) :
  dist A B = 8 → dist B C = 13 → dist C A = 15 →
  is_orthocenter H A B C → is_incenter I A B C → is_circumcenter O A B C →
  angleHIO = ∠ H I O → 
  sin_angle_HIO A B C H I O angleHIO = \frac{7 \sqrt{3}}{26} :=
begin
  intros hAB hBC hCA hH hI hO hAngleHIO,
  rw [sin_angle_HIO, if_pos _],
  { -- prove the result here
    sorry,
  },
  { -- prove the conditions
    exact ⟨hAB, hBC, hCA, hH, hI, hO⟩,
  },
end

end sin_HIO_value_l533_533695


namespace pirate_15th_l533_533894

def smallest_initial_coins (init_coins : ℕ) : Prop :=
  ∃ (k : ℕ), (k = 15) ∧
  ∀ (i : ℕ), (1 ≤ i ∧ i ≤ 15) → ((15-i).factorial * init_coins) % (15^14) = 0

theorem pirate_15th ({init_coins : ℕ}) (h : smallest_initial_coins init_coins) :
  (init_coins / (15^14 / (14.factorial : ℕ))) = 1001 :=
sorry

end pirate_15th_l533_533894


namespace solve_for_ab_l533_533307

theorem solve_for_ab (a b : ℤ) 
  (h1 : a + 3 * b = 27) 
  (h2 : 5 * a + 4 * b = 47) : 
  a + b = 11 :=
sorry

end solve_for_ab_l533_533307


namespace ellipse_eccentricity_is_sqrt2_div_2_l533_533431

def ellipse_eccentricity (a b : ℝ) : ℝ := 
  let c := Real.sqrt (a^2 - b^2)
  c / a

-- Condition
def a : ℝ := Real.sqrt 16
def b : ℝ := Real.sqrt 8

-- Question with the correct answer
theorem ellipse_eccentricity_is_sqrt2_div_2 : ellipse_eccentricity a b = Real.sqrt 2 / 2 :=
by 
  sorry

end ellipse_eccentricity_is_sqrt2_div_2_l533_533431


namespace arithmetic_common_difference_l533_533021

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533021


namespace birch_trees_probability_l533_533535

theorem birch_trees_probability :
  let non_birch_trees := 4 + 5,
      total_trees := 4 + 5 + 3,
      birch_trees := 3,
      slots := non_birch_trees + 1,
      successful_arrangements := Nat.choose slots birch_trees,
      total_arrangements := Nat.choose total_trees birch_trees,
      probability := successful_arrangements / total_arrangements
  in Rat.num_den (success_probability : ℚ) = (6, 11) ∧ 6 + 11 = 17 := 
by
  sorry

end birch_trees_probability_l533_533535


namespace max_radius_of_spheres_in_cone_l533_533509

theorem max_radius_of_spheres_in_cone :
  ∃ r : ℝ, let h := 4 in let s := 8 in let R := \(\frac{12}{5 + 2 \sqrt{3}}\) in
  (h = 4 ∧ s = 8 ∧ 
  ∀ r : ℝ, spheres_fit_in_cone r h s ∧ spheres_touch_each_other_externally r ∧ 
  spheres_touch_lateral_surface r ∧ spheres_touch_base r) → r = R :=
sorry

end max_radius_of_spheres_in_cone_l533_533509


namespace lens_area_equals_triangle_area_l533_533783

theorem lens_area_equals_triangle_area (a b c : ℝ) (h : c^2 = a^2 + b^2) : 
  let area_triangle := 2 * a * b in
  let area_semi_legs := π * a^2 + π * b^2 in
  let area_semi_hyp := π * c^2 in
  area_semi_legs + area_triangle - area_semi_hyp = 2 * a * b :=
sorry

end lens_area_equals_triangle_area_l533_533783


namespace no_integer_roots_p_eq_2016_l533_533631

noncomputable def p (a b c d : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots_p_eq_2016 
  (a b c d : ℤ)
  (h₁ : p a b c d 1 = 2015)
  (h₂ : p a b c d 2 = 2017) :
  ¬ ∃ x : ℤ, p a b c d x = 2016 :=
sorry

end no_integer_roots_p_eq_2016_l533_533631


namespace arithmetic_common_difference_l533_533019

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533019


namespace proof_calc_l533_533220

noncomputable def calc_result := (243 : ℝ)^(3/5)

theorem proof_calc : calc_result = 27 := by
  have h : (243 : ℝ) = (3 : ℝ)^5 := by norm_num
  rw [h]
  have h1 : (3^5)^(3/5) = 3^((5 : ℝ) * (3/5)) := by norm_num
  rw [h1]
  have h2 : (5 : ℝ) * (3/5) = 3 := by norm_num
  rw [h2]
  norm_num
  sorry

end proof_calc_l533_533220


namespace problem1_problem2_l533_533667

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * Real.exp x

theorem problem1 (a : ℝ) : 
  (∀ x ∈ Set.Icc a - 1 x, f a x ≤ f a (x + 1)) ∧ (∀ x < a - 1,  f a x ≥ f a (x - 1)) := 
sorry

noncomputable def F (x : ℝ) : ℝ := (x - 2) * Real.exp x - x + Real.log x

theorem problem2 : 
  ∃ m, ∃ x ∈ Set.Icc (1 / 4:ℝ) 1, f 2 x - x + @Real.log x ∈ -4 < m < - 3 := 
sorry

end problem1_problem2_l533_533667


namespace johns_average_speed_l533_533725

-- Definitions of conditions
def total_time_hours : ℝ := 6.5
def total_distance_miles : ℝ := 255

-- Stating the problem to be proven
theorem johns_average_speed :
  (total_distance_miles / total_time_hours) = 39.23 := 
sorry

end johns_average_speed_l533_533725


namespace doughnut_cost_is_35_l533_533927

variable (students : ℕ) (choco_demand : ℕ) (glaze_demand : ℕ)
variable (choco_cost : ℕ) (glaze_cost : ℕ)

def total_doughnut_cost (choco_demand choco_cost glaze_demand glaze_cost : ℕ) : ℕ :=
  (choco_demand * choco_cost) + (glaze_demand * glaze_cost)

theorem doughnut_cost_is_35 : 
  students = 25 → 
  choco_demand = 10 → 
  glaze_demand = 15 → 
  choco_cost = 2 → 
  glaze_cost = 1 → 
  total_doughnut_cost choco_demand choco_cost glaze_demand glaze_cost = 35 :=
by
  intros h1 h2 h3 h4 h5
  subst h2
  subst h3
  subst h4
  subst h5
  unfold total_doughnut_cost
  norm_num
  rfl

end doughnut_cost_is_35_l533_533927


namespace cost_per_ton_ice_correct_l533_533920

variables {a p n s : ℝ}

-- Define the cost per ton of ice received by enterprise A
noncomputable def cost_per_ton_ice_received (a p n s : ℝ) : ℝ :=
  (2.5 * a + p * s) * 1000 / (2000 - n * s)

-- The statement of the theorem
theorem cost_per_ton_ice_correct :
  ∀ a p n s : ℝ,
  2000 - n * s ≠ 0 →
  cost_per_ton_ice_received a p n s = (2.5 * a + p * s) * 1000 / (2000 - n * s) := by
  intros a p n s h
  unfold cost_per_ton_ice_received
  sorry

end cost_per_ton_ice_correct_l533_533920


namespace M_union_N_eq_M_l533_533037

def M : Set (ℝ × ℝ) := {p | p.1 * p.2 = 1 ∧ p.1 > 0}
def N : Set (ℝ × ℝ) := {p | Real.arctan p.1 + Real.arccot p.2 = Real.pi}

theorem M_union_N_eq_M : M ∪ N = M := 
sorry

end M_union_N_eq_M_l533_533037


namespace eval_expression_l533_533976

theorem eval_expression : 
  (∛ ((-4: ℝ)^3)) - (1/2: ℝ)^0 + (0.25: ℝ)^ (1/2: ℝ) * ((-1 / real.sqrt 2): ℝ)^(-4: ℝ) + 2^(real.log 2 3) = 0 :=
by {
  rw [real.cbrt_pow (by norm_num : (-4: ℝ)) 3],
  norm_num,
  rfl
}

end eval_expression_l533_533976


namespace number_of_pencils_per_box_l533_533909

theorem number_of_pencils_per_box (
  P : ℕ, -- number of pencils in each box
  total_boxes_pencils : ℕ := 15,
  cost_per_pencil : ℕ := 4,
  cost_per_pen : ℕ := 5,
  total_cost : ℕ := 18300,
  extra_pens : ℕ := 300,
  more_pens_twice_pencils : 2 * (total_boxes_pencils * P) + extra_pens -- number of pens ordered
) : 
  total_boxes_pencils * P * cost_per_pencil + (more_pens_twice_pencils) * cost_per_pen = total_cost → P = 80 :=
begin
  sorry
end

end number_of_pencils_per_box_l533_533909


namespace max_possible_salary_l533_533192

theorem max_possible_salary
  (num_players : ℕ)
  (min_salary total_salary_cap : ℝ) 
  (h_num_players : num_players = 15) 
  (h_min_salary : min_salary = 20000)
  (h_total_salary_cap : total_salary_cap = 800000) :
  ∃ max_salary : ℝ, max_salary = 520000 :=
by
  use 520000
  sorry

end max_possible_salary_l533_533192


namespace sqrt_neg_8a3_eq_neg_2a_sqrt_neg_2a_l533_533786

variable (a : ℝ)
variable (i : ℂ) [ComplexField]

noncomputable def sqrt_of_neg_8a3 : ℂ :=
  sqrt (-8 * a ^ 3)

theorem sqrt_neg_8a3_eq_neg_2a_sqrt_neg_2a (h1 : a ≤ 0) (h2 : i^2 = -1) :
  sqrt_of_neg_8a3 a i = -2 * a * sqrt (-2 * a) :=
sorry

end sqrt_neg_8a3_eq_neg_2a_sqrt_neg_2a_l533_533786


namespace triplet_A_not_sum_7_triplet_E_not_sum_7_l533_533230

def triplet_A : ℚ × ℚ × ℚ := (3/2, 4/3, 13/6)
def triplet_B : ℝ × ℝ × ℝ := (4, -3, 6)
def triplet_C : ℝ × ℝ × ℝ := (2.5, 3.1, 1.4)
def triplet_D : ℝ × ℝ × ℝ := (7.4, -9.4, 9.0)
def triplet_E : ℚ × ℚ × ℚ := (-3/4, -9/4, 8)

theorem triplet_A_not_sum_7 : triplet_A.1 + triplet_A.2 + triplet_A.3 ≠ 7 := by
  sorry

theorem triplet_E_not_sum_7 : triplet_E.1 + triplet_E.2 + triplet_E.3 ≠ 7 := by
  sorry

-- Note that you might need to include additional helps or imports to benefit from rational number arithmetic.

end triplet_A_not_sum_7_triplet_E_not_sum_7_l533_533230


namespace prime_implies_form_l533_533999

-- Given conditions
def sequence (n : ℕ) : ℕ
| 1 := 3
| 2 := 7
| n + 2 := (sequence (n + 1))^2 + 5 - sequence n

-- Prime number function
def is_prime (n : ℕ) : Prop :=
  ∀ m < n, m > 1 → n % m ≠ 0

-- The theorem to prove
theorem prime_implies_form (n : ℕ) (h : is_prime (sequence n + (-1)^n)) :
  ∃ m : ℕ, n = 3^m := by
  sorry

end prime_implies_form_l533_533999


namespace arithmetic_common_difference_l533_533024

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533024


namespace compute_g3_power4_l533_533792

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom h1 : ∀ x, x ≥ 1 → f(g(x)) = x^3
axiom h2 : ∀ x, x ≥ 1 → g(f(x)) = x^4
axiom h3 : g(81) = 81

theorem compute_g3_power4 : [g 3]^4 = 81 := sorry

end compute_g3_power4_l533_533792


namespace negation_of_P_is_exists_Q_l533_533818

def P (x : ℝ) : Prop := x^2 - x + 3 > 0

theorem negation_of_P_is_exists_Q :
  (¬ (∀ x : ℝ, P x)) ↔ (∃ x : ℝ, ¬ P x) :=
sorry

end negation_of_P_is_exists_Q_l533_533818


namespace volume_relation_l533_533449

variables (A B C D : ℝ^3)
def V (a b c d : ℝ^3) : ℝ := (1/6) * (b - a) ⬝ (c - a) ⨯ (d - a)

def reflect (p q : ℝ^3) : ℝ^3 := 2 • q - p

noncomputable def A' := reflect A B
noncomputable def B' := reflect B C
noncomputable def C' := reflect C D
noncomputable def D' := reflect D A

theorem volume_relation :
  V A' B' C' D' = 15 * V A B C D :=
sorry

end volume_relation_l533_533449


namespace correct_subtraction_l533_533501

-- Define the given conditions
variable (A B C : ℕ)
variable h1 : A * 1000 + B * 100 + C * 10 + 6 - 57 = 1819

-- Prove the correct subtraction
theorem correct_subtraction (A B C : ℕ) (h1 : A * 1000 + B * 100 + C * 10 + 6 - 57 = 1819) : 
  A * 1000 + B * 100 + C * 10 + 9 - 57 = 1822 :=
by
  sorry

end correct_subtraction_l533_533501


namespace problem1_problem2_l533_533676

-- Definitions for given vectors and the function
def vec_m : ℝ × ℝ := (1, real.sqrt 3)
def vec_n (x : ℝ) : ℝ × ℝ := (real.sin x, real.cos x)
def f (x : ℝ) : ℝ := vec_m.1 * vec_n x.1 + vec_m.2 * vec_n x.2

-- Problem 1: Prove that the smallest positive period of f(x) is 2π and the maximum value of f(x) is 2
theorem problem1 :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ 
  (∀ x, f x ≤ f 0 + real.abs (vec_m.1) * real.abs (vec_n 0.1) + 
    real.abs (vec_m.2) * real.abs (vec_n 0.2)) := sorry

-- Definitions for triangle problem
variables (a b c : ℝ)
def angle_B : ℝ := real.arccos (1 / 3)
def angle_C : ℝ := real.arcsin (real.sqrt 3 / 2) - π / 3

-- Conditions given in the problem
axiom condition1 : c = real.sqrt 6
axiom condition2 : f angle_C = real.sqrt 3

-- Applying the law of sines and cosine conditions
theorem problem2 :
  ∀ (b : ℝ), (∃ b, (b / real.sin angle_B) = (c / real.sin angle_C)) → b = 8 / 3 := sorry

end problem1_problem2_l533_533676


namespace domain_proof_l533_533594

def domain_of_function : Set ℝ := {x : ℝ | x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x}

theorem domain_proof :
  (∀ x : ℝ, (x ≠ 7) → (x^2 - 16 ≥ 0) → (x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x)) ∧
  (∀ x : ℝ, (x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x) → (x ≠ 7) ∧ (x^2 - 16 ≥ 0)) :=
by
  sorry

end domain_proof_l533_533594


namespace complex_number_solution_trajectory_equation_l533_533044

theorem complex_number_solution (z : ℂ) :
  (1 + 2 * Complex.I) * Complex.conj z = 4 + 3 * Complex.I →
  z = 2 + Complex.I ∧ z / Complex.conj z = (3/5) + (4/5) * Complex.I :=
by sorry

theorem trajectory_equation (x y z : ℝ) :
  z = Complex.abs (2 + Complex.I) →
  abs (complex.mk (x - 1) y) = z →
  (x - 1)^2 + y^2 = 5 :=
by sorry

end complex_number_solution_trajectory_equation_l533_533044


namespace projection_is_correct_l533_533970

noncomputable def vector_projection_onto_plane
  (v : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
let dot_product := v.1 * n.1 + v.2 * n.2 + v.3 * n.3 in
let norm_squared := n.1 * n.1 + n.2 * n.2 + n.3 * n.3 in
let scalar := dot_product / norm_squared in
(n.1 * scalar, n.2 * scalar, n.3 * scalar)

theorem projection_is_correct :
  let v := (2:ℝ, -1, 4)
  let n := (2:ℝ, 3, -1)
  let projection := vector_projection_onto_plane v n
  let p := (v.1 - projection.1, v.2 - projection.2, v.3 - projection.3)
  p = (17 / 7 : ℝ, -5 / 14, 53 / 14) :=
by
  sorry

end projection_is_correct_l533_533970


namespace tan_angle_sum_angle_sum_double_l533_533064

variable {α β : ℝ} (h0 : 1 / (1^2 + 7^2) ≤ 1)
variable (h1 : sin β = sqrt 5 / 5)
variable (h2 : tan α = 7)
variable (h3 : tan β = 1 / 2)

theorem tan_angle_sum : tan (α + β) = -3 :=
by {
  sorry
}

theorem angle_sum_double : α + 2 * β = 3 * π / 4 :=
by {
  sorry
}

end tan_angle_sum_angle_sum_double_l533_533064


namespace dot_product_evaluation_l533_533674

variables {V : Type*} [InnerProductSpace ℝ V]

def vector_a (a : V) := ∥a∥ = 2
def vector_b (b : V) := ∥b∥ = 3
def angle_60 (a b : V) := (real.angle_between_vectors a b = real.pi/3)

theorem dot_product_evaluation
  (a b : V)
  (ha : vector_a a)
  (hb : vector_b b)
  (hab : angle_60 a b) :
  (inner_product a (a + b) = 7) := by
  sorry

end dot_product_evaluation_l533_533674


namespace arithmetic_seq_zeros_l533_533293

noncomputable def f (a x : ℝ) : ℝ := (|x - a| - (3 / x) + a - 2)

theorem arithmetic_seq_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧
        f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧
        2 * x2 = x1 + x3) →
  (a ∈ { (5 + 3 * Real.sqrt 33) / 8, -9 / 5 }) :=
sorry

end arithmetic_seq_zeros_l533_533293


namespace solve_for_z_l533_533241

theorem solve_for_z (z : ℝ) (h : sqrt (5 - 4 * z) = 7) : z = -11 := by
  sorry

end solve_for_z_l533_533241


namespace solve_system_l533_533419

open Complex

def equation1 (x y z : ℂ) : Prop := x^2 + 2*y*z = x
def equation2 (x y z : ℂ) : Prop := y^2 + 2*z*x = z
def equation3 (x y z : ℂ) : Prop := z^2 + 2*x*y = y

theorem solve_system :
  (equation1 0 0 0 ∧ equation2 0 0 0 ∧ equation3 0 0 0)
  ∨ (equation1 (2/3 : ℂ) (-1/3 : ℂ) (-1/3 : ℂ) ∧ equation2 (2/3 : ℂ) (-1/3 : ℂ) (-1/3 : ℂ) ∧ equation3 (2/3 : ℂ) (-1/3 : ℂ) (-1/3 : ℂ))
  ∨ (equation1 1 0 0 ∧ equation2 1 0 0 ∧ equation3 1 0 0)
  ∨ (equation1 (1/3 : ℂ) (1/3 : ℂ) (1/3 : ℂ) ∧ equation2 (1/3 : ℂ) (1/3 : ℂ) (1/3 : ℂ) ∧ equation3 (1/3 : ℂ) (1/3 : ℂ) (1/3 : ℂ))
  ∨ (∃ y z : ℂ, y = (-1 + complex.sqrt 3 * complex.I) / 6 ∧ z = (-1 - complex.sqrt 3 * complex.I) / 6 ∧ equation1 ((1 - y - z) : ℂ) y z ∧ equation2 ((1 - y - z) : ℂ) y z ∧ equation3 ((1 - y - z) : ℂ) y z)
  ∨ (∃ y z : ℂ, y = (-1 - complex.sqrt 3 * complex.I) / 6 ∧ z = (-1 + complex.sqrt 3 * complex.I) / 6 ∧ equation1 ((1 - y - z) : ℂ) y z ∧ equation2 ((1 - y - z) : ℂ) y z ∧ equation3 ((1 - y - z) : ℂ) y z)
  sorry

end solve_system_l533_533419


namespace angle_CPD_eq_126_l533_533356

-- Definitions for points and semicircles
variables (S M T C D P : Point)
variables (SUM MVT : Semicircle)
variables (O1 O2 : Point) -- Centers of the semicircles

-- Conditions: 
-- PC is tangent to semicircle SUM 
def tangent_PC_SUM : Tangent PC SUM := sorry

-- PD is tangent to semicircle MVT 
def tangent_PD_MVT : Tangent PD MVT := sorry

-- SMT is a straight line 
def straight_SMT : Collinear S M T := sorry

-- Arc SU is 72 degrees
def arc_SU_72 : Arc SUM S U = 72 := sorry

-- Arc VT is 54 degrees
def arc_VT_54 : Arc MVT V T = 54 := sorry

-- Proof statement
theorem angle_CPD_eq_126 
(tangent1 : Tangent PC SUM)
(tangent2 : Tangent PD MVT)
(collinear : Collinear S M T)
(arc1 : Arc SUM S U = 72)
(arc2 : Arc MVT V T = 54) 
: Angle C P D = 126 := sorry

end angle_CPD_eq_126_l533_533356


namespace fixed_monthly_charge_for_100_GB_l533_533223

theorem fixed_monthly_charge_for_100_GB
  (fixed_charge M : ℝ)
  (extra_charge_per_GB : ℝ := 0.25)
  (total_bill : ℝ := 65)
  (GB_over : ℝ := 80)
  (extra_charge : ℝ := GB_over * extra_charge_per_GB) :
  total_bill = M + extra_charge → M = 45 :=
by sorry

end fixed_monthly_charge_for_100_GB_l533_533223


namespace inequality_3var_l533_533039

theorem inequality_3var (x y z : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y) (h₃ : 0 ≤ z) (h₄ : x * y + y * z + z * x = 1) : 
    1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 5 / 2 :=
sorry

end inequality_3var_l533_533039


namespace right_triangle_AB_length_l533_533707

noncomputable def length_AB (BC : ℝ) (A : ℝ) : ℝ :=
  BC / Real.tan A

theorem right_triangle_AB_length :
  ∀ A B C : Type,
    ∀ a b c : A,
    ∀ h : ∠ a b c = 90,
    ∀ k : ∠ a c b = 40,
    ∀ l : length b c = 12,
  abs (length_AB 12 (40 * Real.pi / 180) - 14.3) < 0.1 :=
by
  sorry

end right_triangle_AB_length_l533_533707


namespace common_difference_l533_533000

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l533_533000


namespace sequence_increases_starting_from_some_term_l533_533857

noncomputable def a : ℕ → ℕ
| 0     => 1 
| (n+1) => if a n % 5 = 0 then a n / 5 else floor (real.sqrt 5 * a n)

theorem sequence_increases_starting_from_some_term :
  ∃ k : ℕ, ∀ n : ℕ, n ≥ k -> a (n + 1) ≥ a n :=
sorry

end sequence_increases_starting_from_some_term_l533_533857


namespace geometric_series_abs_sum_range_l533_533658

open Function

variable {α : Type*} [LinearOrderedField α]

-- Definition of geometric series sum convergence
def geometric_series_sum (a q : α) (h : |q| < 1) : α :=
  a / (1 - q)

-- The problem statement
theorem geometric_series_abs_sum_range (a : α) (q : α) (h : |q| < 1) 
  (h_sum : geometric_series_sum a q (abs_lt.mpr h) = -2) : 
  (∑' n : ℕ, |a * q^n|) ∈ set.Ici (2 : α) :=
by
  sorry

end geometric_series_abs_sum_range_l533_533658


namespace aubree_animals_total_l533_533563

theorem aubree_animals_total (b_go c_go b_return c_return : ℕ) 
    (h1 : b_go = 20) (h2 : c_go = 40) 
    (h3 : b_return = b_go * 2) 
    (h4 : c_return = c_go - 10) : 
    b_go + c_go + b_return + c_return = 130 := by 
  sorry

end aubree_animals_total_l533_533563


namespace line_parallel_plane_theorem_l533_533849

-- Definitions based on the conditions
variables {ℝ : Type*} [euclidean_space ℝ] 
variables (P : plane ℝ) (l1 l2 : line ℝ)

-- Definition of "parallel" between lines and planes
def line_parallel_to_plane (l1 : line ℝ) (P : plane ℝ) : Prop :=
  ∃ l2 : line ℝ, l2 ⊆ P ∧ l1 ∥ l2

-- Statement of the theorem
theorem line_parallel_plane_theorem
  (h₁ : ∃ l2 : line ℝ, l2 ⊆ P ∧ l1 ∥ l2) :
  line_parallel_to_plane l1 P :=
begin
    sorry
end

end line_parallel_plane_theorem_l533_533849


namespace compute_g3_power4_l533_533793

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom h1 : ∀ x, x ≥ 1 → f(g(x)) = x^3
axiom h2 : ∀ x, x ≥ 1 → g(f(x)) = x^4
axiom h3 : g(81) = 81

theorem compute_g3_power4 : [g 3]^4 = 81 := sorry

end compute_g3_power4_l533_533793


namespace central_angle_eq_pi_over_3_l533_533530

-- Definition of the problem:
def radius := let r : ℝ := r

def chord_length_eq_radius (r : ℝ) := radius r = r

-- Proof statement:
theorem central_angle_eq_pi_over_3 (r : ℝ) (h : chord_length_eq_radius r) : angle_of_chord r r = π / 3 :=
sorry

end central_angle_eq_pi_over_3_l533_533530


namespace conical_funnel_max_volume_height_l533_533116

theorem conical_funnel_max_volume_height (l : ℝ) (h_pos : 0 < l) (h_val : l = 20) :
  ∃ h : ℝ, 0 < h ∧ h < l ∧ 
    (∀ x : ℝ, 0 < x ∧ x < l → volume l x ≤ volume l h) ∧ 
    h = 20 * Real.sqrt 3 / 3 :=
sorry

noncomputable def volume (l : ℝ) (h : ℝ) : ℝ := 
  (1 / 3) * Real.pi * h * (l^2 - h^2) 

end conical_funnel_max_volume_height_l533_533116


namespace equivalence_a_gt_b_and_inv_a_lt_inv_b_l533_533312

variable {a b : ℝ}

theorem equivalence_a_gt_b_and_inv_a_lt_inv_b (h : a * b > 0) : 
  (a > b) ↔ (1 / a < 1 / b) := 
sorry

end equivalence_a_gt_b_and_inv_a_lt_inv_b_l533_533312


namespace smaller_of_two_numbers_l533_533441

theorem smaller_of_two_numbers (a b : ℕ) (h1 : a * b = 4761) (h2 : 10 ≤ a ∧ a < 100) (h3 : 10 ≤ b ∧ b < 100) : min a b = 53 :=
by {
  sorry -- proof skips as directed
}

end smaller_of_two_numbers_l533_533441


namespace fruit_falling_days_l533_533457

noncomputable def total_fruits : Nat := 123

def fruit_fall_rate (day : Nat) : Nat :=
  if day = 1 then 1 else day

def cumulative_fruit_fall (days : Nat) : Nat :=
  (days * (days + 1)) / 2

def remaining_fruits (days : Nat) : Nat :=
  total_fruits - cumulative_fruit_fall (days - 1)

def total_days (total_fruits : Nat) : Nat :=
  Nat.recOn total_fruits
    0
    (fun days fruits =>
      if remaining_fruits days < fruit_fall_rate (days + 1) then
        days + remaining_fruits days
      else
        days + 1)

theorem fruit_falling_days :
  total_days total_fruits = 17 :=
by
  sorry

end fruit_falling_days_l533_533457


namespace doughnut_cost_is_35_l533_533926

variable (students : ℕ) (choco_demand : ℕ) (glaze_demand : ℕ)
variable (choco_cost : ℕ) (glaze_cost : ℕ)

def total_doughnut_cost (choco_demand choco_cost glaze_demand glaze_cost : ℕ) : ℕ :=
  (choco_demand * choco_cost) + (glaze_demand * glaze_cost)

theorem doughnut_cost_is_35 : 
  students = 25 → 
  choco_demand = 10 → 
  glaze_demand = 15 → 
  choco_cost = 2 → 
  glaze_cost = 1 → 
  total_doughnut_cost choco_demand choco_cost glaze_demand glaze_cost = 35 :=
by
  intros h1 h2 h3 h4 h5
  subst h2
  subst h3
  subst h4
  subst h5
  unfold total_doughnut_cost
  norm_num
  rfl

end doughnut_cost_is_35_l533_533926


namespace sin_is_even_and_has_period_pi_l533_533087

/-- Prove that the function y = sin(2x + 5π/2) is an even function and has a period of π. -/
theorem sin_is_even_and_has_period_pi : 
  (∀ x : ℝ, sin(2 * x + 5 * π / 2) = sin(2 * x + π) ∧ sin(2 * x + π) = sin (2 * (x + π / 2))) ∧ 
  (∀ x : ℝ, sin(2 * (-x) + 5 * π / 2) = sin(2 * x + 5 * π / 2)) :=
sorry

end sin_is_even_and_has_period_pi_l533_533087


namespace prove_common_difference_l533_533016

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533016


namespace prove_common_difference_l533_533005

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533005


namespace intersection_point_y_axis_l533_533091

theorem intersection_point_y_axis : 
  ∃ y, (∃ c : ℝ, y = 2 * (0 : ℝ)^2 + c) ∧ y = 3 :=
begin
  use 3,
  split,
  { use (0 : ℝ),
    simp, },
  { refl, },
end

end intersection_point_y_axis_l533_533091


namespace nth_monomial_monomial_sequence_l533_533906

noncomputable def monomial_sequence : ℕ → ℕ × ℕ
| 0       := (2, 1)
| (n + 1) := let (a_, b_) := monomial_sequence n 
             in (a_ * 2, b_ + 2)

theorem nth_monomial_monomial_sequence (n : ℕ) : 
  let (a, b) := monomial_sequence n in
  ∃ k, a = 2^k ∧ b = 2 * k - 1 := by 
{
  sorry
}

end nth_monomial_monomial_sequence_l533_533906


namespace angle_less_than_or_equal_30_l533_533746

theorem angle_less_than_or_equal_30
  (A B C M : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  [MetricSpace M]
  (hM_inside_ABC : M ∈ triangle ABC) :
  ∃ θ : ℝ, θ ∈ {∠ (M, A, B), ∠ (M, B, C), ∠ (M, C, A)} ∧ θ ≤ 30 :=
sorry

end angle_less_than_or_equal_30_l533_533746


namespace calc_g_8_12_13_l533_533979

def g (n : ℕ) : ℝ := real.logb 1001 ((n + 2)^2)

theorem calc_g_8_12_13 :
  g 8 + g 12 + g 13 = 2 + real.logb 1001 (300^2) := sorry

end calc_g_8_12_13_l533_533979


namespace max_product_yields_831_l533_533472

open Nat

noncomputable def max_product_3digit_2digit : ℕ :=
  let digits : List ℕ := [1, 3, 5, 8, 9]
  let products := digits.permutations.filter (λ l, l.length ≥ 5).map (λ l, 
    let a := l.get? 0 |>.getD 0
    let b := l.get? 1 |>.getD 0
    let c := l.get? 2 |>.getD 0
    let d := l.get? 3 |>.getD 0
    let e := l.get? 4 |>.getD 0
    (100 * a + 10 * b + c) * (10 * d + e)
  )
  products.maximum.getD 0

theorem max_product_yields_831 : 
  ∃ (a b c d e : ℕ), 
    List.permutations [1, 3, 5, 8, 9].any 
      (λ l, l = [a, b, c, d, e] ∧ 
            100 * a + 10 * b + c = 831 ∧
            ∀ (x y z w v : ℕ), 
              List.permutations [1, 3, 5, 8, 9].any 
                (λ l, l = [x, y, z, w, v] ∧ 
                      (100 * x + 10 * y + z) * (10 * w + v) ≤ (100 * a + 10 * b + c) * (10 * d + e)
                )
      )
:= by
  sorry

end max_product_yields_831_l533_533472


namespace number_of_fixed_points_l533_533268

def S (n : ℕ) := Fin n → Bool

def triangularArray (n : ℕ) : Type :=
  { a : { i // 1 ≤ i ∧ i ≤ n } → { j // 1 ≤ j ∧ j ≤ n - i + 1 } → Bool //
    ∀ (i j : ℕ)
      (hi : 1 < i ∧ i ≤ n)
      (hj : 1 ≤ j ∧ j ≤ n - i + 1),
      a ⟨i, hi.1, hi.2⟩ ⟨j, hj.1, hj.2⟩ = tt ↔
        ((a ⟨i - 1, Nat.sub_lt_succ_iff.mpr (Nat.lt_trans hi.1 (Nat.succ_le.mpr hi.2))⟩ ⟨j, hj.1, Nat.le_trans hj.2 (Nat.sub_le n i)⟩ ≠ 
         a ⟨i - 1, Nat.sub_lt_succ_iff.mpr (Nat.lt_trans hi.1 (Nat.succ_le.mpr hi.2))⟩ ⟨j + 1, Nat.le_of_lt_add_one (Nat.lt_of_le_of_lt hj.2 (Nat.sub_lt (Nat.lt_succ_self i) (Nat.succ_pos 0)))⟩)) } 

noncomputable def f {n : ℕ} (x : S n) : S n :=
  λ k, x ⟨n - k.1, Nat.sub_lt_succ k.1 1⟩

theorem number_of_fixed_points {n : ℕ} :
  ∃ (k : ℕ), k = 2^((n + 1) / 2) :=
sorry

end number_of_fixed_points_l533_533268


namespace area_of_triangle_l533_533253

variables (a b c : ℝ) (x y : ℝ)

-- Condition: defining the ellipse equation
def is_ellipse (x y a b : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Condition: focal length
def focal_length (a b c : ℝ) : Prop :=
  c = Real.sqrt (a^2 - b^2)

-- Proving the area of the triangle
theorem area_of_triangle (a b c : ℝ) (h1 : is_ellipse 0 b a b) (h2 : focal_length a b c) :
  ∃ (A : ℝ), A = c * b :=
begin
  use c * b,
  sorry
end

end area_of_triangle_l533_533253


namespace MG_eq_MH_l533_533649

variable {O : Type*} [MetricSpace O]
variables (A B M C D E F G H : O)
variables [Circle O] {AB : ClosedSegment O} {CD EF : OpenSegment O}
variables (OM : MetricBall O)

-- Definitions and conditions
def IsMidpoint (M A B : O) : Prop :=
  dist A M = dist M B

def IsChord (O : Type*) [Circle O] (CD : OpenSegment O) : Prop :=
  ∃ P Q : O, P ∈ Circle O ∧ Q ∈ Circle O ∧ CD = segment P Q

def OnCircle (O : Type*) [Circle O] (X : O) : Prop :=
  X ∈ Circle O

-- The proof problem statement
theorem MG_eq_MH
  (h_mid : IsMidpoint M A B)
  (h_chords_thru_M : IsChord O CD ∧ IsChord O EF)
  (h_CF_DE : ∃ G', ∃ H', ∃ CGF DEF, CF = ClosedSegment C G' ∨ CF = ClosedSegment F G' 
                                  ∧ DE = ClosedSegment D H' ∨ DE = ClosedSegment E H'):
  (dist M G = dist M H) :=
sorry

end MG_eq_MH_l533_533649


namespace single_colony_reaches_limit_in_24_days_l533_533175

/-- A bacteria colony doubles in size every day. -/
def double (n : ℕ) : ℕ := 2 ^ n

/-- Two bacteria colonies growing simultaneously will take 24 days to reach the habitat's limit. -/
axiom two_colonies_24_days : ∀ k : ℕ, double k + double k = double 24

/-- Prove that it takes 24 days for a single bacteria colony to reach the habitat's limit. -/
theorem single_colony_reaches_limit_in_24_days : ∃ x : ℕ, double x = double 24 :=
sorry

end single_colony_reaches_limit_in_24_days_l533_533175


namespace house_number_count_l533_533953

def is_two_digit_prime (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100 ∧ Prime n

def house_number_conditions (AB CD : ℕ) : Prop :=
  is_two_digit_prime AB ∧ is_two_digit_prime CD ∧ AB ≠ CD

theorem house_number_count : 
  ∃ (count : ℕ), count = 110 ∧ 
  (∀ (AB CD : ℕ), house_number_conditions AB CD → AB < 50 → CD < 50) :=
by 
  use 110
  intros AB CD hABCD hAB50 hCD50
  have h1 : ∃ primes : ℕ, primes = 11 := sorry
  sorry

end house_number_count_l533_533953


namespace blue_lipstick_count_l533_533060

def total_students : Nat := 200

def colored_lipstick_students (total : Nat) : Nat :=
  total / 2

def red_lipstick_students (colored : Nat) : Nat :=
  colored / 4

def blue_lipstick_students (red : Nat) : Nat :=
  red / 5

theorem blue_lipstick_count :
  blue_lipstick_students (red_lipstick_students (colored_lipstick_students total_students)) = 5 := 
sorry

end blue_lipstick_count_l533_533060


namespace area_of_fifteen_sided_figure_l533_533225

def vertices : List (ℚ × ℚ) :=
  [(1,3), (2,4), (2,5), (3,6), (4,6), (5,6), (6,5), (6,4), (5,3), (5,2), (4,1), (3,1), (2,2), (1,2), (1,3)]

theorem area_of_fifteen_sided_figure : 
  let area := 15
  in figure_area vertices = area := 
by
  -- Proof here
  sorry

end area_of_fifteen_sided_figure_l533_533225


namespace mack_writing_time_tuesday_l533_533401

variable (minutes_per_page_mon : ℕ := 30)
variable (time_mon : ℕ := 60)
variable (pages_wed : ℕ := 5)
variable (total_pages : ℕ := 10)
variable (minutes_per_page_tue : ℕ := 15)

theorem mack_writing_time_tuesday :
  (time_mon / minutes_per_page_mon) + pages_wed + (3 * minutes_per_page_tue / minutes_per_page_tue) = total_pages →
  (3 * minutes_per_page_tue) = 45 := by
  intros h
  sorry

end mack_writing_time_tuesday_l533_533401


namespace sum_of_digits_of_smallest_prime_palindrome_greater_than_300_l533_533899

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  n.toString.foldl (λ acc d, acc + (d.toNat - '0'.toNat)) 0

def smallest_prime_palindrome_greater_than_300 : ℕ :=
  Nat.find (λ n => n > 300 ∧ Nat.Prime n ∧ is_palindrome n)

theorem sum_of_digits_of_smallest_prime_palindrome_greater_than_300 :
  sum_of_digits smallest_prime_palindrome_greater_than_300 = 7 :=
sorry

end sum_of_digits_of_smallest_prime_palindrome_greater_than_300_l533_533899


namespace croissant_machine_completion_time_l533_533521

noncomputable def start_time : ℕ := 9 -- Representing 9:00 AM
noncomputable def fraction_completed_time : ℕ := 11 * 60 + 45 - 9 * 60 -- Minutes from 9:00 AM to 11:45 AM
noncomputable def fraction_job : ℚ := 1 / 4 -- Fraction of job completed by 11:45 AM

theorem croissant_machine_completion_time :
  let total_minutes_in_day := 24 * 60 in
  let completion_minutes := start_time * 60 + (fraction_completed_time / fraction_job).toNat in
  completion_minutes = 20 * 60 :=
by
  sorry

end croissant_machine_completion_time_l533_533521


namespace expression_simplification_l533_533219

theorem expression_simplification : (4^2 * 7 / (8 * 9^2) * (8 * 9 * 11^2) / (4 * 7 * 11)) = 44 / 9 :=
by
  sorry

end expression_simplification_l533_533219


namespace factor_3m2n_12mn_12n_factor_abx2_4y2ba_calculate_result_l533_533604

-- Proof 1: Factorize 3m^2 n - 12mn + 12n
theorem factor_3m2n_12mn_12n (m n : ℤ) : 3 * m^2 * n - 12 * m * n + 12 * n = 3 * n * (m - 2)^2 :=
by sorry

-- Proof 2: Factorize (a-b)x^2 + 4y^2(b-a)
theorem factor_abx2_4y2ba (a b x y : ℤ) : (a - b) * x^2 + 4 * y^2 * (b - a) = (a - b) * (x + 2 * y) * (x - 2 * y) :=
by sorry

-- Proof 3: Calculate 2023 * 51^2 - 2023 * 49^2
theorem calculate_result : 2023 * 51^2 - 2023 * 49^2 = 404600 :=
by sorry

end factor_3m2n_12mn_12n_factor_abx2_4y2ba_calculate_result_l533_533604


namespace probability_winning_pair_l533_533178

theorem probability_winning_pair :
  let deck : Finset (Fin 12) := Finset.univ in
  let pairs := deck.product deck
  let same_color (c1 c2 : Fin 12) : Prop :=
    (c1 < 6 ∧ c2 < 6) ∨ (6 ≤ c1 ∧ 6 ≤ c2)
  let same_label (c1 c2 : Fin 12) : Prop :=
    c1 % 6 = c2 % 6
  let winning_pair (c1 c2 : Fin 12) : Prop :=
    same_color c1 c2 ∨ same_label c1 c2
  let total_pairs := pairs.card / 2
  let winning_pairs :=
    (pairs.filter (λ (p : Fin 12 × Fin 12), winning_pair p.1 p.2)).card / 2
  in
  (winning_pairs : ℚ) / total_pairs = 6 / 11 :=
by
  sorry

end probability_winning_pair_l533_533178


namespace fruit_seller_price_l533_533891

noncomputable def CP : ℝ := 12.35 / 1.05

def PricePerKG15Loss : ℝ := 0.85 * CP

theorem fruit_seller_price (h : CP = 12.35 / 1.05) : PricePerKG15Loss ≈ 9.99 :=
by
  rw [h, PricePerKG15Loss]
  sorry

end fruit_seller_price_l533_533891


namespace triangle_perimeter_l533_533848

-- Define the three sides of the triangle such that one side is 7,
-- another side is at least 14, and the third side respects the Triangle Inequality
def side1 : ℕ := 7
def side2_min : ℕ := 2 * side1  -- the minimum length of the second side, which is 14
noncomputable def smallest_whole_larger_perimeter : ℕ := 42

theorem triangle_perimeter :
  ∀ (a b c : ℕ), a = side1 → b = side2_min → 
  (a + b > c ∧ a + c > b ∧ b + c > a) → 
  smallest_whole_larger_perimeter = 42 :=
by
  sorry

end triangle_perimeter_l533_533848


namespace theorem_problems_l533_533718

open Int
open Real

noncomputable def rectilinear_coordinates : Type :=
  {p : ℝ × ℝ // p.2 ≠ 0}

-- Defining the ellipse condition
def is_on_ellipse (P : rectilinear_coordinates) : Prop :=
  let x := P.val.1 in
  let y := P.val.2 in
  (x^2 / 2) + y^2 = 1

-- Defining the line (l) passing through point P
def line_passing_through_P (P : rectilinear_coordinates) (x y : ℝ) : Prop :=
  let x0 := P.val.1 in
  let y0 := P.val.2 in
  ((x0 * x) / 2) + y0 * y = 1

-- Define the conditions and compute required results
theorem theorem_problems (P : rectilinear_coordinates) 
(h_ellipse : is_on_ellipse P)
(h_line : ∀ x y, line_passing_through_P P x y):
  ∃ e : ℝ, e = (Real.sqrt 2) / 2 ∧
  ∃ S_min : ℝ, S_min = Real.sqrt 2 ∧
  ∃ Q F2 : ℝ × ℝ, collinear Q P.val F2 :=
begin
  sorry
end

end theorem_problems_l533_533718


namespace arithmetic_common_difference_l533_533023

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533023


namespace smallest_positive_debt_resolved_l533_533464

theorem smallest_positive_debt_resolved : ∃ (D : ℕ), D > 0 ∧ (∃ (p g : ℤ), D = 400 * p + 240 * g) ∧ D = 80 := by
  sorry

end smallest_positive_debt_resolved_l533_533464


namespace pentagon_cannot_tile_plane_l533_533705

def interior_angle (n : ℕ) : ℝ :=
  if n >= 3 then (180.0 * (n - 2)) / n
  else 0

def tiles_plane (n : ℕ) : Prop :=
  let angle := interior_angle n
  ∃ k : ℕ, 360.0 = k * angle

theorem pentagon_cannot_tile_plane :
  ¬ tiles_plane 5 :=
  sorry

end pentagon_cannot_tile_plane_l533_533705


namespace system_inequalities_1_system_inequalities_2_l533_533074

theorem system_inequalities_1 (x: ℝ):
  (4 * (x + 1) ≤ 7 * x + 10) → (x - 5 < (x - 8)/3) → (-2 ≤ x ∧ x < 7 / 2) :=
by
  intros h1 h2
  sorry

theorem system_inequalities_2 (x: ℝ):
  (x - 3 * (x - 2) ≥ 4) → ((2 * x - 1) / 5 ≥ (x + 1) / 2) → (x ≤ -7) :=
by
  intros h1 h2
  sorry

end system_inequalities_1_system_inequalities_2_l533_533074


namespace problem1_problem2_problem3_l533_533418

-- Problem 1: Prove that 1/(sqrt(2) + 1) = sqrt(2) - 1
theorem problem1 : 1 / (Real.sqrt 2 + 1) = Real.sqrt 2 - 1 :=
sorry

-- Problem 2: Prove the sum of a series = 2sqrt(10) - 1
theorem problem2 : (∑ n in Finset.range (40-2+1) + 2, 1 / (Real.sqrt n + Real.sqrt (n - 1))) = 2 * Real.sqrt 10 - 1 :=
sorry

-- Problem 3: Given a = 1/(sqrt(5) - 2), prove a^2 - 4a + 1 = 2
theorem problem3 (a : Real) (h : a = 1 / (Real.sqrt 5 - 2)) : a^2 - 4 * a + 1 = 2 :=
sorry

end problem1_problem2_problem3_l533_533418


namespace inv_f_neg_7_eq_2_l533_533663

def f (x : ℝ) := -2^(x + 1) + 1

theorem inv_f_neg_7_eq_2 : f⁻¹' {-7} = {2} :=
by sorry

end inv_f_neg_7_eq_2_l533_533663


namespace sequence_1729th_term_is_370_l533_533807

def digit_cubes_sum (n : ℕ) :=
  (n.digits 10).map (λ x => x ^ 3).sum

noncomputable def sequence_terms (n : ℕ) : ℕ → ℕ
| 0     => 1729
| (k+1) => digit_cubes_sum (sequence_terms k)

theorem sequence_1729th_term_is_370 : sequence_terms 1729 = 370 :=
  sorry

end sequence_1729th_term_is_370_l533_533807


namespace angle_AO_AB_is_30_degrees_l533_533639

-- Define a nonagon, its circumcircle center O, and points P, Q, and R
noncomputable def nonagon (N : Type) [grddiff N] : Type := sorry
noncomputable def center (O : Type) [point O] : Type := sorry
noncomputable def segment (PQ QR : Type) [line PQ QR] : Type := sorry

-- Define midpoint A of segment PQ and midpoint B of the radius perpendicular to QR
noncomputable def midpoint_PQ (A : Type) [point A] : Type := sorry
noncomputable def perpendicular_radius_QR (B : Type) [point B] : Type := sorry

-- Define the angle between AO and AB
noncomputable def angle (P Q R : Type) [point P Q R] : ℝ := sorry

-- The theorem statement
theorem angle_AO_AB_is_30_degrees
  (N : Type) [nonagon N] (O : Type) [center O] 
  (P Q R : Point)
  (PQ QR : Line) [segment PQ QR]
  (A : Point) [midpoint_PQ A]
  (B : Point) [perpendicular_radius_QR B] :
  angle A O B = 30 :=
sorry

end angle_AO_AB_is_30_degrees_l533_533639


namespace find_func_l533_533965

noncomputable def f : ℕ → ℕ := sorry

theorem find_func (f : ℕ → ℕ) :
  (∀ n m : ℕ, f (3 * n + 2 * m) = f n * f m) →
  (f = λ n, 0) ∨ (f = λ n, 1) ∨ (f = λ n, if n = 0 then 1 else 0) :=
begin
  sorry
end

end find_func_l533_533965


namespace combined_salaries_of_A_B_C_E_is_correct_l533_533823

-- Given conditions
def D_salary : ℕ := 7000
def average_salary : ℕ := 8800
def n_individuals : ℕ := 5

-- Combined salary of A, B, C, and E
def combined_salaries : ℕ := 37000

theorem combined_salaries_of_A_B_C_E_is_correct :
  (average_salary * n_individuals - D_salary) = combined_salaries :=
by
  sorry

end combined_salaries_of_A_B_C_E_is_correct_l533_533823


namespace graph_shift_right_and_up_l533_533089

noncomputable def g (x : ℝ) : ℝ :=
if h₁ : -2 ≤ x ∧ x ≤ 1 then 2 - x
else if h₂ : 1 ≤ x ∧ x ≤ 3 then sqrt (4 - (x - 1) ^ 2)
else if h₃ : 3 ≤ x ∧ x ≤ 5 then x - 3
else 0

noncomputable def g_shifted (x : ℝ) : ℝ :=
g (x - 1) + 3

theorem graph_shift_right_and_up :
  (∀ x, g (x - 1) + 3 = g_shifted x) ∧
  (∀ x, -1 ≤ x - 1 ∧ x - 1 ≤ 4 → g (x - 1) + 3 = g_shifted x) :=
by
  sorry

end graph_shift_right_and_up_l533_533089


namespace sum_reciprocal_triangular_l533_533603

def t (n : ℕ) : ℚ := n * (n + 1) / 2

theorem sum_reciprocal_triangular :
  ∑ n in Finset.range 2003 | n > 0, (1 / t n) = 2003 / 1002 :=
by
  sorry

end sum_reciprocal_triangular_l533_533603


namespace identify_worst_player_l533_533893

structure Player :=
  (name : String)
  (sex : String)
  (generation : Nat)
  (twin : Option Player)

def grandfather : Player := ⟨"grandfather", "male", 0, none⟩
def mother : Player := ⟨"mother", "female", 1, none⟩
def son : Player := ⟨"son", "male", 2, none⟩
def daughter : Player := ⟨"daughter", "female", 2, some son⟩
noncomputable def son'' : Player := { son with twin := some daughter }

def is_twin (p1 p2 : Player) : Prop := p1.twin = some p2 ∧ p2.twin = some p1
def same_generation (p1 p2 : Player) : Prop := p1.generation = p2.generation
def same_sex (p1 p2 : Player) : Prop := p1.sex = p2.sex

def worst_player : Player := daughter
def best_player : Player := daughter  -- Making a placeholder assignment to ensure condition consistency

theorem identify_worst_player
  (h1 : same_sex (worst_player.twin.getD grandfather) best_player)
  (h2 : same_generation worst_player best_player)
  : worst_player = daughter :=
sorry

end identify_worst_player_l533_533893


namespace cups_of_sugar_already_put_in_l533_533761

-- Defining the given conditions
variable (f s x : ℕ)

-- The total flour and sugar required
def total_flour_required := 9
def total_sugar_required := 6

-- Mary needs to add 7 more cups of flour than cups of sugar
def remaining_flour_to_sugar_difference := 7

-- Proof goal: to find how many cups of sugar Mary has already put in
theorem cups_of_sugar_already_put_in (total_flour_remaining : ℕ := 9 - 7)
    (remaining_sugar : ℕ := 9 - 7) 
    (already_added_sugar : ℕ := 6 - 2) : already_added_sugar = 4 :=
by sorry

end cups_of_sugar_already_put_in_l533_533761


namespace smallest_geq_06_l533_533833

theorem smallest_geq_06 : 
  ∀ (a b c d : ℝ), 
  a = 0.8 → b = 1/2 → c = 0.9 → d = 1/3 →
  (∀ x, (x = a ∨ x = b ∨ x = c ∨ x = d) → x ≥ 0.6) → 
  ∃ y, (y = a ∨ y = b ∨ y = c ∨ y = d) ∧ y ≥ 0.6 ∧ ∀ z, (z = a ∨ z = b ∨ z = c ∨ z = d) → z ≥ 0.6 → y ≤ z :=
by
  intros a b c d ha hb hc hd h
  have h₀ : b = 0.5, from by rw hb,
  have h₁ : d ≈ 0.333, from by rw hd,

  -- Use decimals directly for ease of setup, prove the main result
  sorry

end smallest_geq_06_l533_533833


namespace arithmetic_common_difference_l533_533029

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533029


namespace smallest_value_of_N_l533_533908

noncomputable theory
open_locale classical

theorem smallest_value_of_N : ∃ (l m n : ℕ), (l - 1) * (m - 1) * (n - 1) = 378 ∧ l * m * n = 560 :=
by {
  -- The actual proof goes here
  sorry
}

end smallest_value_of_N_l533_533908


namespace arithmetic_common_difference_l533_533025

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533025


namespace equivalence_a_gt_b_and_inv_a_lt_inv_b_l533_533313

variable {a b : ℝ}

theorem equivalence_a_gt_b_and_inv_a_lt_inv_b (h : a * b > 0) : 
  (a > b) ↔ (1 / a < 1 / b) := 
sorry

end equivalence_a_gt_b_and_inv_a_lt_inv_b_l533_533313


namespace plane_through_A_perpendicular_to_BC_l533_533500

noncomputable def point_A : (ℝ × ℝ × ℝ) := (-4, -2, 5)
noncomputable def point_B : (ℝ × ℝ × ℝ) := (3, -3, -7)
noncomputable def point_C : (ℝ × ℝ × ℝ) := (9, 3, -7)

noncomputable def vector_BC : (ℝ × ℝ × ℝ) :=
  (point_C.1 - point_B.1, point_C.2 - point_B.2, point_C.3 - point_B.3)

theorem plane_through_A_perpendicular_to_BC :
  let normal_vector := vector_BC in
  let plane_eq : ℝ × ℝ × ℝ → ℝ := λ p, normal_vector.1 * (p.1 + 4) + normal_vector.2 * (p.2 + 2) + normal_vector.3 * (p.3 - 5) 
  ∀ (p : ℝ × ℝ × ℝ), plane_eq p = 0 ↔ p.1 + p.2 + 6 = 0 := 
by 
  sorry

end plane_through_A_perpendicular_to_BC_l533_533500


namespace scallop_cost_equivalence_l533_533873

-- Define given conditions
def scallops_per_pound := 8
def cost_per_pound := 24  -- in dollars
def scallops_per_person := 2
def number_of_people := 8

-- Define and prove the total cost of scallops for serving 8 people
theorem scallop_cost_equivalence (h1 : scallops_per_pound = 8) (h2 : cost_per_pound = 24)
(h3 : scallops_per_person = 2) (h4 : number_of_people = 8) :
  let total_scallops := number_of_people * scallops_per_person in
  let weight_in_pounds := total_scallops / scallops_per_pound in
  let total_cost := weight_in_pounds * cost_per_pound in
  total_cost = 48 :=
by
  sorry

end scallop_cost_equivalence_l533_533873


namespace perpendicular_slope_l533_533815

theorem perpendicular_slope (k : ℝ) : (∀ x, y = k*x) ∧ (∀ x, y = 2*x + 1) → k = -1 / 2 :=
by
  intro h
  sorry

end perpendicular_slope_l533_533815


namespace aubree_animals_total_l533_533562

theorem aubree_animals_total (b_go c_go b_return c_return : ℕ) 
    (h1 : b_go = 20) (h2 : c_go = 40) 
    (h3 : b_return = b_go * 2) 
    (h4 : c_return = c_go - 10) : 
    b_go + c_go + b_return + c_return = 130 := by 
  sorry

end aubree_animals_total_l533_533562


namespace weight_on_planets_l533_533831

theorem weight_on_planets
  (weight_on_earth : ℝ)
  (weight_ratio_moon : ℝ)
  (weight_ratio_mars : ℝ)
  (weight_ratio_jupiter : ℝ)
  (weight_ratio_venus : ℝ)
  (known_weight_on_earth : ℝ)
  (known_weight_on_moon : ℝ)
  (h_moon : known_weight_on_moon = known_weight_on_earth * weight_ratio_moon)
  (new_weight_on_earth : ℝ := 136) : 
  weight_on_earth = 136 → 
  weight_ratio_moon = 1 / 5 → 
  weight_ratio_mars = 3 / 8 → 
  weight_ratio_jupiter = 25 / 3 → 
  weight_ratio_venus = 8 / 9 → 
  known_weight_on_earth = 133 → 
  known_weight_on_moon = 26.6 →
  let new_weight_on_moon := new_weight_on_earth * weight_ratio_moon in
  let new_weight_on_mars := new_weight_on_earth * weight_ratio_mars in
  let new_weight_on_jupiter := new_weight_on_earth * weight_ratio_jupiter in
  let new_weight_on_venus := new_weight_on_earth * weight_ratio_venus in
  new_weight_on_moon ≈ 27.08 ∧ 
  new_weight_on_mars = 51 ∧ 
  new_weight_on_jupiter ≈ 1133.33 ∧ 
  new_weight_on_venus ≈ 121.78 :=
by {
  sorry
}

end weight_on_planets_l533_533831


namespace perpendicular_planes_l533_533279

variables {m n : Line}
variables {α β : Plane}

def perpendicular (m : Line) (α : Plane) : Prop := sorry
def parallel (m n : Line) : Prop := sorry
def in_plane (m : Line) (α : Plane) : Prop := sorry

theorem perpendicular_planes
  (h1 : perpendicular m α)
  (h2 : parallel m n)
  (h3 : parallel n β) :
  perpendicular α β :=
sorry

end perpendicular_planes_l533_533279


namespace ratio_of_smaller_circle_to_larger_hexagon_l533_533882

-- Definitions based on the conditions
def larger_circle_radius (r : ℝ) : ℝ := r / sqrt 3
def larger_hexagon_side_length (r : ℝ) : ℝ := 2 * larger_circle_radius r
def area_of_larger_hexagon (r : ℝ) : ℝ := (3 * sqrt 3 / 2) * (larger_hexagon_side_length r) ^ 2
def area_of_smaller_circle (r : ℝ) : ℝ := π * r ^ 2
def ratio_of_areas (r : ℝ) : ℝ := area_of_smaller_circle r / area_of_larger_hexagon r

-- The theorem we need to prove
theorem ratio_of_smaller_circle_to_larger_hexagon (r : ℝ) (hr : r > 0) :
  ratio_of_areas r = π / (2 * sqrt 3) :=
by
  sorry

end ratio_of_smaller_circle_to_larger_hexagon_l533_533882


namespace count_triangles_in_expanded_grid_l533_533302

theorem count_triangles_in_expanded_grid :
  let base_layers := 3 in
  let small_triangles (n : ℕ) := (n * (n + 1) / 2) in
  let medium_triangles (n : ℕ) := (n * (n - 1) / 2) in
  let large_triangles (n : ℕ) := if n ≥ 2 then 1 else 0 in
  small_triangles base_layers + medium_triangles (base_layers - 1) + large_triangles (base_layers - 2) = 17 :=
begin
  sorry
end

end count_triangles_in_expanded_grid_l533_533302


namespace all_seven_are_brothers_l533_533697

variable (Boy : Type) [Fintype Boy] [DecidableEq Boy]

variable (B1 B2 B3 B4 B5 B6 B7 : Boy)

-- Each boy has at least three brothers among the others.
def has_at_least_three_brothers (b : Boy) : Prop :=
  Fintype.card {b' // b' ≠ b ∧ b' ≠ B1 ∧ b' ≠ B2 ∧ b' ≠ B3 ∧ b' ≠ B4 ∧ b' ≠ B5 ∧ b' ≠ B6 ∧ b' ≠ B7} ≥ 3

-- Prove all seven are brothers
theorem all_seven_are_brothers (h1 : has_at_least_three_brothers B1)
                               (h2 : has_at_least_three_brothers B2)
                               (h3 : has_at_least_three_brothers B3)
                               (h4 : has_at_least_three_brothers B4)
                               (h5 : has_at_least_three_brothers B5)
                               (h6 : has_at_least_three_brothers B6)
                               (h7 : has_at_least_three_brothers B7) :
  ∀ (i j : Boy), i ≠ j → has_at_least_three_brothers i → has_at_least_three_brothers j → i = j :=
begin
  -- proof needed here
  sorry
end

end all_seven_are_brothers_l533_533697


namespace probability_even_number_four_dice_l533_533139

noncomputable def probability_same_even_number_facing_up_four_dice : ℚ :=
  1 / 432

theorem probability_even_number_four_dice :
  ∀ (P : ℚ), 
    (∀ (DieResult1 DieResult2 DieResult3 DieResult4 : ℕ), 
      DieResult1 ∈ {2, 4, 6} ∧ DieResult2 = DieResult1 ∧ DieResult3 = DieResult1 ∧ DieResult4 = DieResult1 → 
        P = 1 / 432) →
      P = probability_same_even_number_facing_up_four_dice :=
by
  intros P h
  exact sorry

end probability_even_number_four_dice_l533_533139


namespace triangle_side_relation_l533_533619

theorem triangle_side_relation (a b c : ℝ) (α β γ : ℝ)
  (h1 : 3 * α + 2 * β = 180)
  (h2 : α + β + γ = 180) :
  a^2 + b * c - c^2 = 0 :=
sorry

end triangle_side_relation_l533_533619


namespace prove_common_difference_l533_533007

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533007


namespace lulu_cash_left_l533_533395

theorem lulu_cash_left :
  ∀ (initial money spentIceCream spentTshirt deposited finalCash: ℝ),
    initial = 65 →
    spentIceCream = 5 →
    spentTshirt = 0.5 * (initial - spentIceCream) →
    deposited = (initial - spentIceCream - spentTshirt) / 5 →
    finalCash = initial - spentIceCream - spentTshirt - deposited →
    finalCash = 24 :=
by
  intros
  sorry

end lulu_cash_left_l533_533395


namespace B_joined_after_11_months_l533_533547

theorem B_joined_after_11_months :
  ∀ (A_inv : ℝ) (B_inv : ℝ) (A_time : ℝ) (profit_ratio : ℝ × ℝ),
    A_inv = 75000 →
    B_inv = 105000 →
    A_time = 18 →
    profit_ratio = (7, 4) →
    let B_join_time := 18 - (75000 * 18 / 105000 * (profit_ratio.2 / profit_ratio.1)) in
    B_join_time ≈ 11 :=
begin
  intros A_inv B_inv A_time profit_ratio A_inv_def B_inv_def A_time_def profit_ratio_def,
  have B_join_time_calc : B_join_time =
    18 - (75000 * 18 / 105000 * (4 / 7)), from sorry,
  show B_join_time ≈ 11, from sorry,
end

end B_joined_after_11_months_l533_533547


namespace triangle_count_in_square_configuration_l533_533589

/-- Problem configuration:
    Consider a square ABCD with points E, F, G, and H being the midpoints of sides AB, BC, CD, and DA, respectively.
    Additional line segments are drawn from each vertex of the square to the midpoint of the adjacent sides (not opposite sides),
    i.e., segments AE, BF, CG, and DH. Including these segments and the square's sides,
    prove the total number of triangles is 24.
-/
theorem triangle_count_in_square_configuration : 
    ∃ (A B C D E F G H : Type)
    (is_square_config: is_square A B C D)
    (midpoints: midpoint A B E ∧ midpoint B C F ∧ midpoint C D G ∧ midpoint D A H)
    (segments: segment A E ∧ segment B F ∧ segment C G ∧ segment D H),
    ∑ T in (triangles_in_configuration is_square_config midpoints segments), 1 = 24 :=
by sorry

end triangle_count_in_square_configuration_l533_533589


namespace cost_proof_l533_533960

-- Given conditions
def total_cost : Int := 190
def working_days : Int := 19
def trips_per_day : Int := 2
def total_trips : Int := working_days * trips_per_day

-- Define the problem to prove
def cost_per_trip : Int := 5

theorem cost_proof : (total_cost / total_trips = cost_per_trip) := 
by 
  -- This is a placeholder to indicate that we're skipping the proof
  sorry

end cost_proof_l533_533960


namespace find_a_l533_533363

open Real

theorem find_a
    (sin_B cos_A cos_C sin_C : ℝ)
    (a b c : ℝ)
    (h1 : sin_B ^ 2 + cos_A ^ 2 - cos_C ^ 2 = sqrt 3 * sin_B * sin_C)
    (h2 : π * (2 : ℝ)^2 = 4 * π)
    (R : ℝ)
    (h3 : 2 = R) :
  a = 2 :=
sorry

end find_a_l533_533363


namespace fraction_expression_value_find_m_roots_and_theta_l533_533670

noncomputable def sin_cos_sum (θ : ℝ) : Prop :=
  θ ∈ (0, 2 * Real.pi) ∧ sin θ + cos θ = (Real.sqrt 3 + 1) / 2

noncomputable def sin_cos_product (m θ : ℝ) : Prop :=
  θ ∈ (0, 2 * Real.pi) ∧ sin θ * cos θ = m / 2

theorem fraction_expression_value (θ : ℝ) (h1 : sin_cos_sum θ) :
  (sin θ / (1 - cot θ) + cos θ / (1 - tan θ)) = (Real.sqrt 3 + 1) / 2 :=
sorry

theorem find_m (m θ : ℝ) (h1 : sin_cos_sum θ) (h2 : sin_cos_product m θ) :
  m = Real.sqrt 3 / 2 :=
sorry

theorem roots_and_theta (θ : ℝ) (m : ℝ) (h : 2 * x ^ 2 - (Real.sqrt 3 + 1) * x + m = 0) :
  ((θ = Real.pi / 3 ∨ θ = Real.pi / 6) ∧ (x = sin θ ∨ x = cos θ)) :=
sorry

end fraction_expression_value_find_m_roots_and_theta_l533_533670


namespace motorcycles_count_l533_533340

/-- 
Prove that the number of motorcycles in the parking lot is 28 given the conditions:
1. Each car has 5 wheels (including one spare).
2. Each motorcycle has 2 wheels.
3. Each tricycle has 3 wheels.
4. There are 19 cars in the parking lot.
5. There are 11 tricycles in the parking lot.
6. Altogether all vehicles have 184 wheels.
-/
theorem motorcycles_count 
  (cars := 19) 
  (tricycles := 11) 
  (total_wheels := 184) 
  (wheels_per_car := 5) 
  (wheels_per_tricycle := 3) 
  (wheels_per_motorcycle := 2) :
  (184 - (19 * 5 + 11 * 3)) / 2 = 28 :=
by 
  sorry

end motorcycles_count_l533_533340


namespace find_pairs_satisfying_condition_l533_533130

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

namespace FactorialProof

theorem find_pairs_satisfying_condition :
  ∀ n p : ℕ, prime p → (factorial (p - 1) + 1 = p ^ n) ↔ (n = 1 ∧ p = 2) ∨ (n = 1 ∧ p = 3) ∨ (n = 2 ∧ p = 5) :=
by sorry

end FactorialProof

end find_pairs_satisfying_condition_l533_533130


namespace repeating_decimal_as_fraction_l533_533236

theorem repeating_decimal_as_fraction :
  let x := 0.135135135135135 -- Representing .\overline{135} in a repeating decimal form
  in x = 5 / 37 := by
  let x := 0.135135135135135
  have h : 1000 * x - x = 135 := sorry
  have h2 : 999 * x = 135 := by
    exact h.symm
  have hx : x = 135 / 999 := by
    exact h2.symm
  have simpl_frac : 135 / 999 = 5 / 37 := sorry
  rw [hx, simpl_frac]
  rfl

end repeating_decimal_as_fraction_l533_533236


namespace erdos_ginzburg_ziv_l533_533321

noncomputable def least_integer_set {α : Type*} (n : ℕ) (S : Finset α) : Prop :=
∃ (T : Finset α), T ⊆ S ∧ T.card = 6 ∧ (∑ t in T, t) % 6 = 0

theorem erdos_ginzburg_ziv (S : Finset ℤ) :
  (∀ n : ℕ, n ≥ 11 → least_integer_set 6 S) :=
by
  sorry

end erdos_ginzburg_ziv_l533_533321


namespace treasures_first_level_is_4_l533_533842

-- Definitions based on conditions
def points_per_treasure : ℕ := 5
def treasures_second_level : ℕ := 3
def score_second_level : ℕ := treasures_second_level * points_per_treasure
def total_score : ℕ := 35
def points_first_level : ℕ := total_score - score_second_level

-- Main statement to prove
theorem treasures_first_level_is_4 : points_first_level / points_per_treasure = 4 := 
by
  -- We are skipping the proof here and using sorry.
  sorry

end treasures_first_level_is_4_l533_533842


namespace noodles_given_to_William_l533_533226

def initial_noodles : ℝ := 54.0
def noodles_left : ℝ := 42.0
def noodles_given : ℝ := initial_noodles - noodles_left

theorem noodles_given_to_William : noodles_given = 12.0 := 
by
  sorry -- Proof to be filled in

end noodles_given_to_William_l533_533226


namespace ratio_AB_CD_over_AC_BD_sqrt2_l533_533749

-- We first define points and their geometrical relationships in the plane.
variables {A B C D : Type}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]

-- Conditions on the triangle ABC and point D
variables (u v w x : A)
variables [AcuteTriangleABC : AcuteTriangle u v w]
variables [PointDInsideTriangle : IsInside x (triangle u v w)]
variables (h1 : ∠ u x v = ∠ u v w + 90)
variables (h2 : dist u w * dist x v = dist x u * dist w v)

-- The problem to prove
theorem ratio_AB_CD_over_AC_BD_sqrt2 : (dist u v * dist x w) / (dist u w * dist x v) = sqrt 2 := sorry

end ratio_AB_CD_over_AC_BD_sqrt2_l533_533749


namespace angle_between_vectors_l533_533242

-- Declare the vectors as constants
def u : ℝ × ℝ := (3, 4)
def v : ℝ × ℝ := (4, -3)

-- Define functions to calculate dot product and vector magnitude
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def magnitude (a : ℝ × ℝ) : ℝ :=
  Real.sqrt (a.1 ^ 2 + a.2 ^ 2)

-- Statement of the theorem
theorem angle_between_vectors :
  let θ := Real.arccos ((dot_product u v) / ((magnitude u) * (magnitude v))) in
  θ = Real.pi / 2 :=
by
  sorry

end angle_between_vectors_l533_533242


namespace prove_common_difference_l533_533012

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533012


namespace problem1_problem2_l533_533936

-- Definition and proof statement for Problem 1
theorem problem1 (y : ℝ) : 
  (y + 2) * (y - 2) + (y - 1) * (y + 3) = 2 * y^2 + 2 * y - 7 := 
by sorry

-- Definition and proof statement for Problem 2
theorem problem2 (x : ℝ) (h : x ≠ -1) :
  (1 + 2 / (x + 1)) / ((x^2 + 6 * x + 9) / (x + 1)) = 1 / (x + 3) :=
by sorry

end problem1_problem2_l533_533936


namespace move_point_right_3_units_from_neg_2_l533_533203

noncomputable def move_point_to_right (start : ℤ) (units : ℤ) : ℤ :=
start + units

theorem move_point_right_3_units_from_neg_2 : move_point_to_right (-2) 3 = 1 :=
by
  sorry

end move_point_right_3_units_from_neg_2_l533_533203


namespace sequence_general_term_l533_533360

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 2) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end sequence_general_term_l533_533360


namespace equilateral_triangle_tangent_area_l533_533209

def equilateral_triangle_area_cutoff_by_tangent (a b : ℝ) : ℝ :=
  (a * (a - 2 * b) * Real.sqrt 3) / 12

theorem equilateral_triangle_tangent_area
  (a b : ℝ)
  (h_triangle : a > 0)
  (h_b : 0 < b ∧ b < a) :
  equilateral_triangle_area_cutoff_by_tangent a b = (a * (a - 2 * b) * Real.sqrt 3) / 12 :=
by
  sorry

end equilateral_triangle_tangent_area_l533_533209


namespace emily_can_see_emerson_for_5_minutes_l533_533559

def timeCanSeeEmerson : ℕ := 5

def Emily_speed : ℝ := 15
def Emerson_speed : ℝ := 9
def initial_distance : ℝ := 1 / 4
def billboard_distance : ℝ := 3 / 4
def emily_losing_sight_distance : ℝ := 1 / 2

theorem emily_can_see_emerson_for_5_minutes :
  let relative_speed := Emily_speed - Emerson_speed,
      time_to_billboard := initial_distance + billboard_distance / relative_speed,
      time_until_lost_sight := emily_losing_sight_distance / relative_speed in
      timeToBillboard min time_to_billboard = 5 / 60 :=
  by
  sorry

end emily_can_see_emerson_for_5_minutes_l533_533559


namespace probability_both_in_picture_l533_533071

-- Label the noncomputable if necessary
noncomputable def rachel_lap_time : ℝ := 100
noncomputable def ryan_lap_time : ℝ := 75
noncomputable def time_of_picture : ℝ := 10 * 60 + 30  -- 10 minutes and 30 seconds in seconds
noncomputable def fraction_of_track_visible : ℝ := 1 / 3

theorem probability_both_in_picture :
  ∀ (rachel_lap_time ryan_lap_time time_of_picture fraction_of_track_visible : ℝ),
  rachel_lap_time = 100 → ryan_lap_time = 75 → time_of_picture = 630 → fraction_of_track_visible = 1 / 3 →
  let rachel_position := (time_of_picture % rachel_lap_time) / rachel_lap_time in
  let ryan_position := (time_of_picture % ryan_lap_time) / ryan_lap_time in
  (∃ θ : ℝ, θ ∈ Icc (-fraction_of_track_visible / 2) (fraction_of_track_visible / 2) → 
    (rachel_position ∈ Icc (θ - fraction_of_track_visible / 2) (θ + fraction_of_track_visible / 2) ∧ 
    ryan_position ∈ Icc (θ - fraction_of_track_visible / 2) (θ + fraction_of_track_visible / 2))) :=
by
  intros
  sorry

end probability_both_in_picture_l533_533071


namespace short_pencil_cost_l533_533703

theorem short_pencil_cost (x : ℝ)
  (h1 : 200 * 0.8 + 40 * 0.5 + 35 * x = 194) : x = 0.4 :=
by {
  sorry
}

end short_pencil_cost_l533_533703


namespace prove_common_difference_l533_533009

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533009


namespace number_of_correct_statements_is_zero_l533_533093

-- Definitions for each statement as conditions
def statement1 (a b : Line) : Prop := a ∩ b = ∅ → skew a b
def statement2 (q : Quadrilateral) : Prop := (oppositeSidesEqual q) → parallelogram q
def statement3 (l1 l2 l3 : Line) : Prop := pairwiseIntersect l1 l2 l3 → coplanar l1 l2 l3
def statement4 (p : Polyhedron) : Prop := (twoParallelFacesAndParallelograms p) → prism p
def statement5 (l : Line) (π : Plane) : Prop := (infinitelyManyPointsNotInPlane l π) → parallel l π

-- Proof problem statement
theorem number_of_correct_statements_is_zero : 
  ∀ a b l1 l2 l3 (q : Quadrilateral) (p : Polyhedron) (l : Line) (π : Plane), 
    ¬statement1 a b ∧ ¬statement2 q ∧ ¬statement3 l1 l2 l3 ∧ ¬statement4 p ∧ ¬statement5 l π → 
    numberOfCorrectStatements = 0 :=
by 
  intros 
  sorry

end number_of_correct_statements_is_zero_l533_533093


namespace sum_of_interior_angles_pentagon_l533_533105

theorem sum_of_interior_angles_pentagon : 
  let n := 5 in
  180 * (n - 2) = 540 :=
  by
  -- Introducing the variable n for the number of sides
  let n := 5
  -- Using the given formula to calculate the sum of the interior angles
  have h : 180 * (n - 2) = 540 := by sorry
  exact h

end sum_of_interior_angles_pentagon_l533_533105


namespace no_two_acute_angles_l533_533277

variable (α β γ : ℝ)
variable (x y z : ℝ)

-- Assume the conditions given
axiom interior_angles_sum : α + β + γ = 180
axiom def_x : x = α + β
axiom def_y : y = β + γ
axiom def_z : z = γ + α

-- The theorem to be proven
theorem no_two_acute_angles (h : x < 90 ∧ y < 90) : false :=
by
  have sum_x_y_z : x + y + z = 360 := 
    calc
      x + y + z = (α + β) + (β + γ) + (γ + α) : by rw [def_x, def_y, def_z]
      ... = 2 * (α + β + γ) : by ring
      ... = 360 : by rw [interior_angles_sum, mul_comm, mul_assoc]
  have x_y_range : 0 < x + y ∧ x + y < 180 :=
    by apply And.intro
       . exact (lt_trans (add_pos zero_lt_two (by linarith)) h.1)
       . exact (add_lt_of_lt_sub' (show x + y < 180 from sorry))
  have z_value : z = 360 - (x + y) := eq_sub_of_add_eq sum_x_y_z
  have : z > 180 := calc
    z = 360 - (x + y) : by rw [z_value]
    ... > 180 : by linarith
  -- This contradicts the property of triangle angles sums to 180 degrees
  exact lt_irrefl 𝕌 this

end no_two_acute_angles_l533_533277


namespace lower_selling_price_l533_533316

theorem lower_selling_price (C P : ℝ) (h1 : C = 200) (h2 : 350 - C = (P - C) + 0.05 * (P - C)) : P = 368.42 :=
by
  -- Define the cost of the book
  have hC : C = 200 := h1
  -- Define the gain equation
  have hGain : 350 - hC = (P - hC) * (1 + 0.05) := by rw [h2]
  -- Solve for P
  sorry

end lower_selling_price_l533_533316


namespace length_CF_is_26_area_ACF_is_119_l533_533161

open Classical

noncomputable theory

-- Definitions for the given conditions
def sameRadius (r : ℝ) := 
  let R := 13 in R = r

def intersects (A B : Point) (C1 C2 : Circle) := 
  C1.contains A ∧ C1.contains B ∧ C2.contains A ∧ C2.contains B

def angleCADIs90 (C A D : Point) := 
  ∠CAD = 90

def BFEqualsBD (B F D : Point) := 
  dist B F = dist B D

def pointBOnLineCD (C D B : Point) := 
  B ∈ lineOfPoints C D

def pointFOnSameSideAsA (C D A F : Point) := 
  sameSideOfLine A F (lineOfPoints C D)

def BCGiven (B C : Point) := 
  dist B C = 10

-- Problem statement
theorem length_CF_is_26 (r : ℝ) (A B C D F: Point) (C1 C2: Circle) (h1: sameRadius r)
                        (h2: intersects A B C1 C2) (h3: angleCADIs90 C A D) 
                        (h4: BFEqualsBD B F D) (h5: pointBOnLineCD C D B) 
                        (h6: pointFOnSameSideAsA C D A F) :
                        dist C F = 26 :=
sorry

theorem area_ACF_is_119 (r : ℝ) (A B C D F : Point) (C1 C2 : Circle) (h1 : sameRadius r)
                        (h2 : intersects A B C1 C2) (h3 : angleCADIs90 C A D)
                        (h4 : BFEqualsBD B F D) (h5 : pointBOnLineCD C D B)
                        (h6 : pointFOnSameSideAsA C D A F) (h7 : BCGiven B C) :
                        area A C F = 119 :=
sorry

end length_CF_is_26_area_ACF_is_119_l533_533161


namespace sampling_interval_is_100_l533_533257

-- Define the total number of numbers (N), the number of samples to be taken (k), and the condition for systematic sampling.
def N : ℕ := 2005
def k : ℕ := 20

-- Define the concept of systematic sampling interval
def sampling_interval (N k : ℕ) : ℕ := N / k

-- The proof that the sampling interval is 100 when 2005 numbers are sampled as per the systematic sampling method.
theorem sampling_interval_is_100 (N k : ℕ) 
  (hN : N = 2005) 
  (hk : k = 20) 
  (h1 : N % k ≠ 0) : 
  sampling_interval (N - (N % k)) k = 100 :=
by
  -- Initialization
  sorry

end sampling_interval_is_100_l533_533257


namespace factorization_problem_l533_533805

theorem factorization_problem (a b : ℤ) : 
  (∀ y : ℤ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b)) → a - b = 1 := 
by
  sorry

end factorization_problem_l533_533805


namespace car_mpg_difference_l533_533528

theorem car_mpg_difference 
(h_highway : ∀ (tank_size : ℝ), 462 / tank_size)
(h_city_miles : 336)
(h_city_mpg : 40) : 
  let tank_size := h_city_miles / h_city_mpg in
  (462 / tank_size) - h_city_mpg = 15 :=
by
  sorry

end car_mpg_difference_l533_533528


namespace odd_n_colorable_convex_polygon_l533_533698

theorem odd_n_colorable_convex_polygon (n : ℕ) (h : ∃ c : Fin (n+1) → Fin (n) → Fin (n), 
  ∀ (i j k : Fin (n+1)), i ≠ j → j ≠ k → k ≠ i →
  ∃ (a b c : Fin (n)), c i j = a ∧ c j k = b ∧ c k i = c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  n % 2 = 1 :=
by
  sorry

end odd_n_colorable_convex_polygon_l533_533698


namespace blue_lipstick_count_l533_533061

def total_students : Nat := 200

def colored_lipstick_students (total : Nat) : Nat :=
  total / 2

def red_lipstick_students (colored : Nat) : Nat :=
  colored / 4

def blue_lipstick_students (red : Nat) : Nat :=
  red / 5

theorem blue_lipstick_count :
  blue_lipstick_students (red_lipstick_students (colored_lipstick_students total_students)) = 5 := 
sorry

end blue_lipstick_count_l533_533061


namespace sum_of_every_third_term_l533_533193

theorem sum_of_every_third_term (y : ℕ → ℕ) (h_seq_length : ∀ n < 3003, y n = y 0 + 2 * n) 
  (h_sum_seq : Finset.sum (Finset.range 3003) y = 902002) : 
  Finset.sum (Finset.range 1001) (λ n, y (3 * n)) = 300667 := 
sorry

end sum_of_every_third_term_l533_533193


namespace valerie_not_enough_money_l533_533453

def num_small_bulbs := 3
def num_large_bulbs := 1
def num_medium_bulbs := 2
def num_extra_small_bulbs := 4

def cost_small_bulb := 8.50
def cost_large_bulb := 14.25
def cost_medium_bulb := 10.75
def cost_extra_small_bulb := 6.25

def valerie_total_money := 80.0

noncomputable def total_cost :=
  (num_small_bulbs * cost_small_bulb) + 
  (num_large_bulbs * cost_large_bulb) + 
  (num_medium_bulbs * cost_medium_bulb) + 
  (num_extra_small_bulbs * cost_extra_small_bulb)

theorem valerie_not_enough_money : total_cost > valerie_total_money :=
by sorry

end valerie_not_enough_money_l533_533453


namespace trig_fraction_identity_l533_533412

theorem trig_fraction_identity :
  (sin 20 + sin 40 + sin 60 + sin 80 + sin 100 + sin 120 + sin 140 + sin 160) / (cos 15 * cos 30 * cos 45)
  = 16 * (sin 50 * sin 70 * cos 10) / (cos 15 * cos 30 * real.sqrt 2) :=
sorry

end trig_fraction_identity_l533_533412


namespace sum_of_positive_integer_solutions_l533_533487

theorem sum_of_positive_integer_solutions (n : ℕ) :
  (sum (filter (λ x : ℕ, x > 0 ∧ x ≤ 30 ∧ 7 * (5 * x - 3) % 10 = 14 % 10) (finset.range (30 + 1)))) = 225 :=
by
  sorry

end sum_of_positive_integer_solutions_l533_533487


namespace interestDifference_l533_533493

noncomputable def simpleInterest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

noncomputable def compoundInterest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

theorem interestDifference (P R T : ℝ) (hP : P = 500) (hR : R = 20) (hT : T = 2) :
  compoundInterest P R T - simpleInterest P R T = 120 := by
  sorry

end interestDifference_l533_533493


namespace area_of_shaded_region_l533_533353

-- Definitions from conditions
def radius : ℝ := 6

def perpendicular_diameters (x y : circle) : Prop := 
  is_diameter x ∧ is_diameter y ∧ angle x.center y.center = π/2

-- The theorem we want to prove
theorem area_of_shaded_region (radius : ℝ)
    (h : perpendicular_diameters PQ RS) : 
    area_shaded_region (PQ : circle, RS : circle radius) = 36 + 18 * π :=
sorry

end area_of_shaded_region_l533_533353


namespace min_marked_necessary_l533_533145

-- A 12x12 board represented as a set of coordinates
def board_12x12 := fin 12 × fin 12

-- A four-cell figure's possible placements, given by its upper-left corner position.
def figure_placements : list (set (fin 12 × fin 12)) :=
  [(0,0), (0,1), (1,0), (1,1), (1,2), (2,0), (2,1)].map (λ c, {(0,0), (0,1), (1,0), (1,1)}.image (prod.add c))

-- The marking condition function
def well_marked (marked : set (fin 12 × fin 12)) :=
  ∀ p ∈ board_12x12, ¬(figure_placements.map (λ s, disjoint marked s)).all id

-- The core hypothesis and the final theorem
theorem min_marked_necessary : ∃ k : ℕ, k = 48 ∧ ∃ marked : set (fin 12 × fin 12), well_marked marked ∧ marked.card = k :=
by sorry

end min_marked_necessary_l533_533145


namespace probability_allison_wins_l533_533556

open ProbabilityTheory

noncomputable def ADice : Pmf ℕ := ⟨λ n, if n = 4 then 1 else 0, by simp [sum_ite_eq _ 4]⟩
noncomputable def CDice : Pmf ℕ := uniformOfFin 6
noncomputable def EDice : Pmf ℕ := ⟨λ n, if n = 3 then 1/3 else if n = 4 then 1/2 else if n = 5 then 1/6 else 0, by simp [finset.sum_ite, finset.sum_const, mul_inv_cancel, ne_of_gt]⟩

theorem probability_allison_wins :
  ADice.prob (λ a, a > 3) * CDice.prob (λ c, c < 4) * EDice.prob (λ e, e < 4) = 1/6 := sorry

end probability_allison_wins_l533_533556


namespace sine_product_identity_l533_533587

open Real

theorem sine_product_identity :
  sin 12 * sin 36 * sin 54 * sin 72 = 1 / 16 := by
  have h1 : sin 72 = cos 18 := by sorry
  have h2 : sin 54 = cos 36 := by sorry
  have h3 : ∀ θ, sin θ * cos θ = 1 / 2 * sin (2 * θ) := by sorry
  have h4 : ∀ θ, cos (2 * θ) = 2 * cos θ ^ 2 - 1 := by sorry
  have h5 : cos 36 = 1 - 2 * (sin 18) ^ 2 := by sorry
  have h6 : ∀ θ, sin (180 - θ) = sin θ := by sorry
  sorry

end sine_product_identity_l533_533587


namespace ratio_of_PR_to_QS_l533_533776

-- Define the points P, Q, R, S and their respective distances.
def P : Point := sorry
def Q : Point := sorry
def R : Point := sorry
def S : Point := sorry

-- Define the distances PQ, QR, and PS.
def PQ : ℝ := 3
def QR : ℝ := 6
def PS : ℝ := 20

-- Calculate PR and QS.
def PR : ℝ := PQ + QR
def QS : ℝ := PS - PQ

-- Define the ratio of PR to QS.
def ratio_PR_QS : ℝ := PR / QS

-- State the theorem to be proved.
theorem ratio_of_PR_to_QS : ratio_PR_QS = 9 / 17 := by
  -- Proof placeholder
  sorry

end ratio_of_PR_to_QS_l533_533776


namespace no_primes_in_sequence_l533_533853

-- Define the product of all primes less than or equal to 61
def P : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53 * 59 * 61

-- Define the set of terms in the sequence
def sequence : Finset ℕ := (Finset.range' 2 58).image (λ n, P + n)

-- Define the proof statement
theorem no_primes_in_sequence : (sequence.filter Nat.prime).card = 0 := 
by
  sorry

end no_primes_in_sequence_l533_533853


namespace cube_surface_area_increase_l533_533494

theorem cube_surface_area_increase (L : ℝ) (h1 : L > 0) :
    let SA_original := 6 * L^2
    let L_new := 1.10 * L
    let SA_new := 6 * (L_new)^2
    let percentage_increase := ((SA_new - SA_original) / SA_original) * 100 
in percentage_increase = 21 := by
  let SA_original := 6 * L^2
  let L_new := 1.10 * L
  let SA_new := 6 * (L_new)^2
  let percentage_increase := ((SA_new - SA_original) / SA_original) * 100 
  sorry

end cube_surface_area_increase_l533_533494


namespace Samantha_apples_count_l533_533757

theorem Samantha_apples_count :
  ∃ A : ℕ, let L_oranges := 5 in
           let L_apples := 3 in
           let S_oranges := 8 in
           let M_oranges := 2 * L_oranges in
           let M_apples := 3 * A in
           (M_oranges + M_apples = 31) ∧ (A = 7) :=
by
  sorry

end Samantha_apples_count_l533_533757


namespace scallop_cost_equivalence_l533_533872

-- Define given conditions
def scallops_per_pound := 8
def cost_per_pound := 24  -- in dollars
def scallops_per_person := 2
def number_of_people := 8

-- Define and prove the total cost of scallops for serving 8 people
theorem scallop_cost_equivalence (h1 : scallops_per_pound = 8) (h2 : cost_per_pound = 24)
(h3 : scallops_per_person = 2) (h4 : number_of_people = 8) :
  let total_scallops := number_of_people * scallops_per_person in
  let weight_in_pounds := total_scallops / scallops_per_pound in
  let total_cost := weight_in_pounds * cost_per_pound in
  total_cost = 48 :=
by
  sorry

end scallop_cost_equivalence_l533_533872


namespace organizingCommitteeWays_l533_533701

-- Define the problem context
def numberOfTeams : Nat := 5
def membersPerTeam : Nat := 8
def hostTeamSelection : Nat := 4
def otherTeamsSelection : Nat := 2

-- Define binomial coefficient
def binom (n k : Nat) : Nat := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the number of ways to select committee members
def totalCommitteeWays : Nat := numberOfTeams * 
                                 (binom membersPerTeam hostTeamSelection) * 
                                 ((binom membersPerTeam otherTeamsSelection) ^ (numberOfTeams - 1))

-- The theorem to prove
theorem organizingCommitteeWays : 
  totalCommitteeWays = 215134600 := 
    sorry

end organizingCommitteeWays_l533_533701


namespace perpendicular_planes_normal_vector_l533_533286

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

theorem perpendicular_planes_normal_vector {m : ℝ} 
  (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) 
  (h₁ : a = (1, 2, -2)) 
  (h₂ : b = (-2, 1, m)) 
  (h₃ : dot_product a b = 0) : 
  m = 0 := 
sorry

end perpendicular_planes_normal_vector_l533_533286


namespace sum_S_20_sum_S_2017_l533_533825

-- Define the sequence and sum properties
def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1 else if n = 2 then 2 else
  let k := (n+1) / 3 in
    if n = 3 * k - 2 then 1 else 2

def sum_sequence (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, sequence (i + 1)

theorem sum_S_20 : sum_sequence 20 = 36 := by
  sorry

theorem sum_S_2017 : sum_sequence 2017 = 3989 := by
  sorry

end sum_S_20_sum_S_2017_l533_533825


namespace measure_of_angle_l533_533282

noncomputable def complementary (α : ℝ) := 90 - α
noncomputable def supplementary (α : ℝ) := 180 - α
def sum_complementary_supplementary (α : ℝ) : Prop := complementary α + supplementary α = 180

theorem measure_of_angle : ∃ α : ℝ, sum_complementary_supplementary α ∧ α = 45 :=
begin
  existsi 45,
  split,
  { sorry }, -- This part proves the sum_complementary_supplementary definition
  { refl } -- Reflexivity directly proves that α = 45
end

end measure_of_angle_l533_533282


namespace divisible_by_p_squared_l533_533743

theorem divisible_by_p_squared (p : ℕ) (hp_prime : p.prime) (hp_gt_3 : p > 3) :
  ((p + 1)^(p - 1) - 1) % (p^2) = 0 :=
by
  sorry

end divisible_by_p_squared_l533_533743


namespace geometric_expressions_l533_533755

variables (a b c x y z : ℝ) (α β γ : ℝ)

theorem geometric_expressions :
  (a^2 + b * c) = units.area ∧
  (a * b^2 / (x * y)) = units.segment ∧
  (((a + b)^3 + c^3) / x) = units.volume ∧
  (a * b * (real.sqrt (x * y))) = units.volume ∧
  (real.sqrt (a^2 + b^2 + c^2) * (real.sin α + real.sin β + real.sin γ)) = units.segment ∧
  ((a^3 + b * c * x) / y * (real.tan α) * (real.tan β) * (real.tan γ)) = units.area :=
sorry

end geometric_expressions_l533_533755


namespace product_maximized_equal_numbers_l533_533069

theorem product_maximized_equal_numbers (n : ℕ) (x : Fin n → ℝ) (S : ℝ) (h_sum : (∑ i, x i) = S) (h_pos : ∀ i, 0 < x i) : 
  (∃ v, (∀ i, x i = v) ∧ (∏ i, x i) ≤ v^n) :=
by sorry

end product_maximized_equal_numbers_l533_533069


namespace maximize_revenue_l533_533172

-- Define the revenue function
def revenue (p : ℝ) : ℝ :=
  p * (150 - 4 * p)

-- Define the price constraints
def price_constraint (p : ℝ) : Prop :=
  0 ≤ p ∧ p ≤ 30

-- The theorem statement to prove that p = 19 maximizes the revenue
theorem maximize_revenue : ∀ p: ℕ, price_constraint p → revenue p ≤ revenue 19 :=
by
  sorry

end maximize_revenue_l533_533172


namespace find_tan_A_l533_533637

noncomputable theory

variables {A B C O : Type*}
variables [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ O]

variables (a b c : ℝ) (angle_A : ℝ) (O : ℝ)
variables {pA pB pC pO : ℝ} -- points of triangle vertices and circumcenter

-- triangle ABC is non-right-angled
axiom non_right_triangle (h : ¬ (angle_A = π / 2))

-- circumcenter of triangle ABC is O
axiom circumcenter (hO : is_circumcenter pA pB pC pO)

-- given condition involving vectors
axiom given_condition (h1 : | ⟪ pC - pA, pO - pB ⟫ | = 3)

-- given relationship between side lengths and trigonometric identities
theorem find_tan_A (h : a = sqrt 3 * b) (h2 : a = sqrt 5 / 2 * b) : 
  tan angle_A = sqrt 39 / 3 := 
sorry

end find_tan_A_l533_533637


namespace num_lines_through_point_with_intercepts_l533_533346

theorem num_lines_through_point_with_intercepts 
  (point : ℝ × ℝ) 
  (is_prime : ℕ → Prop) 
  (passing_through : (ℝ × ℝ) → (ℝ × ℝ) → Prop) :
  point = (4, 3) →
  (∀ b : ℕ, b > 0 → ∃ a : ℕ, a > 0 ∧ is_prime a ∧ passing_through (a : ℝ, b : ℝ) point) →
  (∃ l : ℕ, l = 2) :=
begin 
  intro h_point,
  sorry -- proof to be provided
end

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def passing_through (intercepts : ℝ × ℝ) (point : ℝ × ℝ) : Prop :=
  let (a, b) := intercepts in
  let (x, y) := point in
  let line_eq := x / a + y / b in
  line_eq = 1

end num_lines_through_point_with_intercepts_l533_533346


namespace angle_between_lines_l533_533699

theorem angle_between_lines (n : ℕ) (h : n > 0)
  (lines : Fin n → ℝ × ℝ → Prop)
  (h_lines : ∀ i j, i ≠ j → ¬ parallel (lines i) (lines j)) : 
  ∃ i j, i ≠ j ∧ angle_between (lines i) (lines j) ≤ 180 / n :=
by
  sorry

end angle_between_lines_l533_533699


namespace blue_lipstick_count_l533_533062

def total_students : Nat := 200

def colored_lipstick_students (total : Nat) : Nat :=
  total / 2

def red_lipstick_students (colored : Nat) : Nat :=
  colored / 4

def blue_lipstick_students (red : Nat) : Nat :=
  red / 5

theorem blue_lipstick_count :
  blue_lipstick_students (red_lipstick_students (colored_lipstick_students total_students)) = 5 := 
sorry

end blue_lipstick_count_l533_533062


namespace limit_a_limit_b_limit_c_l533_533578

open Filter Real

-- Part (a)
theorem limit_a :
  tendsto (λ p : ℝ × ℝ, (tan (p.1^2 + p.2^2)) / (p.1^2 + p.2^2)) (𝓝 (0,0)) (𝓝 1) :=
sorry

-- Part (b)
theorem limit_b :
  ¬tendsto (λ p : ℝ × ℝ, (p.1^2 - p.2^2) / (p.1^2 + p.2^2)) (𝓝 (0,0)) at_top :=
sorry

-- Part (c)
theorem limit_c :
  tendsto (λ p : ℝ × ℝ, (p.1 * p.2) / (sqrt (4 - p.1 * p.2) - 2)) (𝓝 (0,0)) (𝓝 (-4)) :=
sorry

end limit_a_limit_b_limit_c_l533_533578


namespace ella_percentage_video_games_l533_533957

theorem ella_percentage_video_games :
  ∃ S : ℝ, S + 0.1 * S = 275 ∧ (100 / S) * 100 = 40 :=
by
  let S := 250
  use S
  split
  · simp [S]
  · simp [S]
  sorry

end ella_percentage_video_games_l533_533957


namespace speed_of_second_train_l533_533875

/-- Given:
1. The first train has a length of 220 meters.
2. The speed of the first train is 120 kilometers per hour.
3. The time taken to cross each other is 9 seconds.
4. The length of the second train is 280.04 meters.

Prove the speed of the second train is 80 kilometers per hour. -/
theorem speed_of_second_train
    (len_first_train : ℝ := 220)
    (speed_first_train_kmph : ℝ := 120)
    (time_to_cross : ℝ := 9)
    (len_second_train : ℝ := 280.04) 
  : (len_first_train / time_to_cross + len_second_train / time_to_cross - (speed_first_train_kmph * 1000 / 3600)) * (3600 / 1000) = 80 := 
by
  sorry

end speed_of_second_train_l533_533875


namespace color_grid_ways_l533_533455

def count_colorings : ℕ :=
  48

theorem color_grid_ways : 
  ∃ (f : (Fin 3) × (Fin 3) → Fin 3), 
  (∀ (c1 c2 : Fin 3) (H : (c1, c2) ∈ {(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)}), f (c1, c2) ≠ f (c1 + H, c2 + H)) ∧ 
  (∀ (col : Fin 3), ∃ (i j : Fin 3), f (i, j) = col) ∧
  (∑ (i j : Fin 3), if f (i, j) = 0 then 1 else 0) >= 1 ∧
  (∑ (i j : Fin 3), if f (i, j) = 1 then 1 else 0) >= 1 ∧
  (∑ (i j : Fin 3), if f (i, j) = 2 then 1 else 0) >= 1 :=
by {
  use 48,
  sorry
}

end color_grid_ways_l533_533455


namespace people_per_table_l533_533552

theorem people_per_table (initial_customers left_customers tables remaining_customers : ℕ) 
  (h1 : initial_customers = 21) 
  (h2 : left_customers = 12) 
  (h3 : tables = 3) 
  (h4 : remaining_customers = initial_customers - left_customers) 
  : remaining_customers / tables = 3 :=
by
  sorry

end people_per_table_l533_533552


namespace transformed_function_monotonically_decreasing_l533_533837

noncomputable def transformed_function (x : ℝ) : ℝ :=
  3 * sin (2 * x + 4 * real.pi / 3)

-- Define the interval
def interval_left : ℝ := real.pi / 12
def interval_right : ℝ := 7 * real.pi / 12

theorem transformed_function_monotonically_decreasing :
  ∀ x₁ x₂ : ℝ,
    interval_left ≤ x₁ ∧ x₁ ≤ interval_right →
    interval_left ≤ x₂ ∧ x₂ ≤ interval_right →
    x₁ ≤ x₂ →
    transformed_function x₂ ≤ transformed_function x₁ :=
sorry

end transformed_function_monotonically_decreasing_l533_533837


namespace num_ways_replace_ast_divisible_by_75_l533_533715

theorem num_ways_replace_ast_divisible_by_75 :
  let digits := {0, 2, 4, 5, 7, 9}
  let num_asterisks := 6
  let num := "2*0*1*6*0*2*"
  let target_sum := 16
d):
  ∃ (ways : ℕ), ways = 2592 := 
begin
  let is_valid := λ (num : String), ("subs with digits" && "last two digits as Fixed (50)") 
       && (digits_sum(num) % 3 == (target_sum(num) == 0(mod 3))).    sorry     
end  
}

end num_ways_replace_ast_divisible_by_75_l533_533715


namespace profit_share_difference_correct_l533_533855

noncomputable def profit_share_difference (a_capital b_capital c_capital b_profit : ℕ) : ℕ :=
  let total_parts := 4 + 5 + 6
  let part_size := b_profit / 5
  let a_profit := 4 * part_size
  let c_profit := 6 * part_size
  c_profit - a_profit

theorem profit_share_difference_correct :
  profit_share_difference 8000 10000 12000 1600 = 640 :=
by
  sorry

end profit_share_difference_correct_l533_533855


namespace proof_students_scored_above_120_l533_533696

open ProbabilityMeasure

noncomputable def estimate_students_scored_above_120 (total_students : ℕ) (μ : ℝ) (σ : ℝ) (p_interval : ℝ) (score_threshold : ℝ) : ℕ :=
  let P := 0.5 - p_interval in
  let num_students := total_students * P in
  num_students.to_nat
  
theorem proof_students_scored_above_120 :
  estimate_students_scored_above_120 50 110 10 0.36 120 = 7 :=
by
  sorry

end proof_students_scored_above_120_l533_533696


namespace sum_of_middle_numbers_l533_533126

open List

-- Given digits used to form the four-digit numbers
def digits := [1, 2, 3, 4]

-- Define a function to generate all unique four-digit numbers from the given digits
def fourDigitNumbers := permutations digits

-- Convert list of digits to a number (assuming digits are in little-endian order)
def digitsToNumber (ds : List ℕ) : ℕ :=
  ds.foldl (λ acc d => acc * 10 + d) 0

-- Generate the actual list of four-digit numbers sorted in ascending order
def sortedNumbers : List ℕ :=
  (fourDigitNumbers.map digitsToNumber).qsort (≤)

-- Define the proof that the sum of the 12th and 13th smallest numbers is 4844
theorem sum_of_middle_numbers : 
  let mid1 := sortedNumbers.get! 11;  -- 12th element (index 11)
  let mid2 := sortedNumbers.get! 12;  -- 13th element (index 12)
  mid1 + mid2 = 4844
:= by
  sorry

end sum_of_middle_numbers_l533_533126


namespace distance_between_foci_is_six_l533_533984

-- Define the points as given in the conditions
def p1 : ℝ × ℝ := (1, 5)
def p2 : ℝ × ℝ := (-3, 9)
def p3 : ℝ × ℝ := (1, -3)
def p4 : ℝ × ℝ := (9, 5)

-- Define the distance between two points in the plane
def distance (p q : ℝ × ℝ) :=
  ( (p.1 - q.1)^2 + (p.2 - q.2)^2 )^.5

-- Hypothesize the semi-major axis (a) and semi-minor axis (b)
noncomputable def semi_major_axis : ℝ := 5
noncomputable def semi_minor_axis : ℝ := 4

-- Calculate the focal distance
noncomputable def c : ℝ := (semi_major_axis^2 - semi_minor_axis^2)^.5

-- The desired proof statement
theorem distance_between_foci_is_six :
  2 * c = 6 :=
by
  -- Definitions of semi_major_axis and semi_minor_axis are directly based on the problem given
  unfold semi_major_axis
  unfold semi_minor_axis
  unfold c
  sorry -- Proof here is skipped; the statement is structurally in place

end distance_between_foci_is_six_l533_533984


namespace coordinates_of_B_l533_533276

theorem coordinates_of_B (x y : ℝ) (A : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (2, 4) ∧ a = (3, 4) ∧ (x - 2, y - 4) = (2 * a.1, 2 * a.2) → (x, y) = (8, 12) :=
by
  intros h
  sorry

end coordinates_of_B_l533_533276


namespace birth_rate_is_correct_l533_533096

-- Define the given constants
def death_rate_per_1000 : ℤ := 11
def population_increase_percentage : ℤ := 21

-- The constant we're trying to prove
def birth_rate_per_1000 : ℤ := 32

theorem birth_rate_is_correct :
  let B := death_rate_per_1000 + population_increase_percentage
  in B = birth_rate_per_1000 :=
by
  sorry

end birth_rate_is_correct_l533_533096


namespace tap_fill_time_l533_533532

theorem tap_fill_time (T : ℝ) :
  let rate_fill_tap := 1 / T
  let rate_empty_tap := 1 / 7
  let combined_rate := 3 / 28
  rate_fill_tap - rate_empty_tap = combined_rate →
  T = 4 :=
begin
  assume h,
  sorry
end

end tap_fill_time_l533_533532


namespace stratified_sampling_technicians_in_sample_l533_533883

theorem stratified_sampling_technicians_in_sample 
  (num_engineers : ℕ)
  (num_technicians : ℕ)
  (num_workers : ℕ)
  (sample_size : ℕ)
  (total_population : ℕ := num_engineers + num_technicians + num_workers)
  (fraction_technicians : ℚ := num_technicians / total_population) :
  num_engineers = 20 → num_technicians = 100 → num_workers = 280 → sample_size = 20 →
  num_technicians_in_sample : ℕ := nat.ceil (fraction_technicians * sample_size) :=
  by
  intros h_eng h_tech h_work h_sample
  rw [h_eng, h_tech, h_work, h_sample]
  have h_total_population : total_population = 400 := by norm_num [total_population, h_eng, h_tech, h_work]
  have h_fraction_technicians : fraction_technicians = 1 / 4 := by norm_num [fraction_technicians, h_total_population, h_tech]
  have h_num_technicians_in_sample : num_technicians_in_sample = nat.ceil ( (1 / 4) * 20) := by norm_num [fraction_technicians, sample_size]
  norm_num at h_num_technicians_in_sample
  exact h_num_technicians_in_sample

end stratified_sampling_technicians_in_sample_l533_533883


namespace gallery_pieces_total_l533_533207

noncomputable def TotalArtGalleryPieces (A : ℕ) : Prop :=
  let D := (1 : ℚ) / 3 * A
  let N := A - D
  let notDisplayedSculptures := (2 : ℚ) / 3 * N
  let totalSculpturesNotDisplayed := 800
  (4 : ℚ) / 9 * A = 800

theorem gallery_pieces_total (A : ℕ) (h : (TotalArtGalleryPieces A)) : A = 1800 :=
by sorry

end gallery_pieces_total_l533_533207


namespace find_g2_l533_533317

variable (g : ℝ → ℝ)

theorem find_g2 (h : ∀ x : ℝ, g (3 * x - 7) = 5 * x + 11) : g 2 = 26 := by
  sorry

end find_g2_l533_533317


namespace students_who_wore_blue_lipstick_l533_533058

theorem students_who_wore_blue_lipstick (total_students : ℕ) (h1 : total_students = 200) : 
  ∃ blue_lipstick_students : ℕ, blue_lipstick_students = 5 :=
by
  have colored_lipstick_students := total_students / 2
  have red_lipstick_students := colored_lipstick_students / 4
  let blue_lipstick_students := red_lipstick_students / 5
  have h2 : blue_lipstick_students = 5 :=
    calc blue_lipstick_students
          = (total_students / 2) / 4 / 5 : by sorry -- detailed calculation steps omitted
  use blue_lipstick_students
  exact h2

end students_who_wore_blue_lipstick_l533_533058


namespace andrew_stickers_now_l533_533048

-- Defining the conditions
def total_stickers : Nat := 1500
def ratio_susan : Nat := 1
def ratio_andrew : Nat := 1
def ratio_sam : Nat := 3
def total_ratio : Nat := ratio_susan + ratio_andrew + ratio_sam
def part : Nat := total_stickers / total_ratio
def susan_share : Nat := ratio_susan * part
def andrew_share_initial : Nat := ratio_andrew * part
def sam_share : Nat := ratio_sam * part
def sam_to_andrew : Nat := (2 * sam_share) / 3

-- Andrew's final stickers count
def andrew_share_final : Nat :=
  andrew_share_initial + sam_to_andrew

-- The theorem to prove
theorem andrew_stickers_now : andrew_share_final = 900 :=
by
  -- Proof would go here
  sorry

end andrew_stickers_now_l533_533048


namespace train_cars_count_l533_533499

theorem train_cars_count
  (cars_in_first_15_seconds : ℕ)
  (time_for_first_5_cars : ℕ)
  (total_time_to_pass : ℕ)
  (h_cars_in_first_15_seconds : cars_in_first_15_seconds = 5)
  (h_time_for_first_5_cars : time_for_first_5_cars = 15)
  (h_total_time_to_pass : total_time_to_pass = 210) :
  (total_time_to_pass / time_for_first_5_cars) * cars_in_first_15_seconds = 70 := 
by 
  sorry

end train_cars_count_l533_533499


namespace work_completion_days_l533_533914

-- Definitions based on the conditions
def A_work_days : ℕ := 20
def B_work_days : ℕ := 30
def C_work_days : ℕ := 10  -- Twice as fast as A, and A can do it in 20 days, hence 10 days.
def together_work_days : ℕ := 12
def B_C_half_day_rate : ℚ := (1 / B_work_days) / 2 + (1 / C_work_days) / 2  -- rate per half day for both B and C
def A_full_day_rate : ℚ := 1 / A_work_days  -- rate per full day for A

-- Converting to rate per day when B and C work only half day daily
def combined_rate_per_day_with_BC_half : ℚ := A_full_day_rate + B_C_half_day_rate

-- The main theorem to prove
theorem work_completion_days 
  (A_work_days B_work_days C_work_days together_work_days : ℕ)
  (C_work_days_def : C_work_days = A_work_days / 2) 
  (total_days_def : 1 / combined_rate_per_day_with_BC_half = 60 / 7) :
  (1 / combined_rate_per_day_with_BC_half) = 60 / 7 :=
sorry

end work_completion_days_l533_533914


namespace total_number_of_animals_l533_533182

-- Define the given conditions as hypotheses
def num_horses (T : ℕ) : Prop :=
  ∃ (H x z : ℕ), H + x + z = 75

def cows_vs_horses (T : ℕ) : Prop :=
  ∃ (w z : ℕ),  w = z + 10

-- Define the final conclusion we need to prove
def total_animals (T : ℕ) : Prop :=
  T = 170

-- The main theorem which states the conditions imply the conclusion
theorem total_number_of_animals (T : ℕ) (h1 : num_horses T) (h2 : cows_vs_horses T) : total_animals T :=
by
  -- Proof to be filled in later
  sorry

end total_number_of_animals_l533_533182


namespace series_bound_l533_533779

theorem series_bound (n : ℕ) : 
  (∑ k in Finset.range (n + 1), 1 / (3 + k^2 : ℝ)) < 4 / 5 := 
sorry

end series_bound_l533_533779


namespace sum_first_25_terms_of_bn_l533_533326

def is_class_periodic_geometric (a : ℕ → ℕ) (m q : ℕ) : Prop :=
  ∀ n : ℕ, a (n + m) = a n * q

theorem sum_first_25_terms_of_bn :
  let b : ℕ → ℕ := λ n, if n % 4 = 0 then 3 * 3^(n / 4) else if n % 4 = 1 then 3^(n / 4) else if n % 4 = 2 then 2 * 3^(n / 4) else 3 * 3^(n / 4)
  let m := 4
  let q := 3
  let b1 := 1
  let b2 := 1
  let b3 := 2
  let b4 := 3
  (is_class_periodic_geometric b m q) →
  (b 1 = b1) →
  (b 2 = b2) →
  (b 3 = b3) →
  (b 4 = b4) →
  ∑ i in finRange 25, b i = 3277 :=
begin
  sorry
end

end sum_first_25_terms_of_bn_l533_533326


namespace max_valid_words_for_AU_language_l533_533709

noncomputable def maxValidWords : ℕ :=
  2^14 - 128

theorem max_valid_words_for_AU_language 
  (letters : Finset (String)) (validLengths : Set ℕ) (noConcatenation : Prop) :
  letters = {"a", "u"} ∧ validLengths = {n | 1 ≤ n ∧ n ≤ 13} ∧ noConcatenation →
  maxValidWords = 16256 :=
by
  sorry

end max_valid_words_for_AU_language_l533_533709


namespace max_three_digit_is_931_l533_533473

def greatest_product_three_digit : Prop :=
  ∃ (a b c d e : ℕ), {a, b, c, d, e} = {1, 3, 5, 8, 9} ∧
  1 ≤ a ∧ a ≤ 9 ∧
  1 ≤ b ∧ b ≤ 9 ∧
  1 ≤ c ∧ c ≤ 9 ∧
  1 ≤ d ∧ d ≤ 9 ∧
  1 ≤ e ∧ e ≤ 9 ∧
  (100 * a + 10 * b + c) * (10 * d + e) = 81685 ∧
  (100 * a + 10 * b + c) = 931

theorem max_three_digit_is_931 : greatest_product_three_digit :=
begin
  sorry
end

end max_three_digit_is_931_l533_533473


namespace tree_height_l533_533911

theorem tree_height (BR MH MB MR TB : ℝ)
  (h_cond1 : BR = 5)
  (h_cond2 : MH = 1.8)
  (h_cond3 : MB = 1)
  (h_cond4 : MR = BR - MB)
  (h_sim : TB / BR = MH / MR)
  : TB = 2.25 :=
by sorry

end tree_height_l533_533911


namespace first_nonzero_digit_fraction_l533_533133

theorem first_nonzero_digit_fraction (n : ℤ) (hn : n = 271) : 
    ∃ d : ℕ, (d = 3 ∧ (d = first_nonzero_digit (1 / (n : ℝ))) := 
by
  sorry

end first_nonzero_digit_fraction_l533_533133


namespace solve_equation_l533_533416

def equation_params (a x : ℝ) : Prop :=
  a * (1 / (Real.cos x) - Real.tan x) = 1

def valid_solutions (a x : ℝ) (k : ℤ) : Prop :=
  (a ≠ 0) ∧ (Real.cos x ≠ 0) ∧ (
    (|a| ≥ 1 ∧ x = Real.arccos (a / Real.sqrt (a * a + 1)) + 2 * Real.pi * k) ∨
    ((-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1) ∧ x = - Real.arccos (a / Real.sqrt (a * a + 1)) + 2 * Real.pi * k)
  )

theorem solve_equation (a x : ℝ) (k : ℤ) :
  equation_params a x → valid_solutions a x k := by
  sorry

end solve_equation_l533_533416


namespace lateral_surface_area_of_cone_l533_533810

-- Definitions of the given conditions
def base_radius_cm : ℝ := 3
def slant_height_cm : ℝ := 5

-- The theorem to prove
theorem lateral_surface_area_of_cone :
  let r := base_radius_cm
  let l := slant_height_cm
  π * r * l = 15 * π := 
by
  sorry

end lateral_surface_area_of_cone_l533_533810


namespace sum_alternating_binomial_coeff_l533_533489

theorem sum_alternating_binomial_coeff (S : ℕ) : 
  (S = ∑ k in Finset.range 51, (-1)^k * Nat.choose 101 (2*k)) → 
  S = 2^50 := by
  intro h
  sorry

end sum_alternating_binomial_coeff_l533_533489


namespace max_square_plots_l533_533902

theorem max_square_plots (l w : ℕ) (f : ℕ) (n : ℕ) (n_col : ℕ) (n_row : ℕ) (k : ℕ) : 
  l = 30 →
  w = 60 →
  f = 3000 →
  n ≥ 4 →
  n_col = n → 
  n_row = 2 * n →
  (30 * (2 * n - 1) + 60 * (n - 1)) ≤ f →
  k = 2 * n ^ 2 →
  k = 1250 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  subst h1
  subst h2
  subst h3
  subst h5
  subst h6
  calc
    2 * 25 ^ 2 = 1250 : sorry

end max_square_plots_l533_533902


namespace CaitlinIs24_l533_533925

-- Definition using the given conditions
def AuntAnnaAge : ℕ := 45
def BriannaAge : ℕ := (2 * AuntAnnaAge) / 3
def CaitlinAge : ℕ := BriannaAge - 6

-- Statement to be proved
theorem CaitlinIs24 : CaitlinAge = 24 :=
by
  sorry

end CaitlinIs24_l533_533925


namespace probability_at_least_60_cents_l533_533523

open Function
open nat


theorem probability_at_least_60_cents :
  let total_ways := nat.choose 15 7,
      successful_ways := (nat.choose 5 2) * (nat.choose 7 5) + (nat.choose 5 1 * (nat.choose 7 6)) + nat.choose 7 7 in
  total_ways = 6435 → successful_ways = 246 → (successful_ways / total_ways : Real) = 246 / 6435 :=
by
  intros total_ways successful_ways h_total h_successful
  sorry

end probability_at_least_60_cents_l533_533523


namespace widget_difference_l533_533400

variable (w t : ℕ)

def monday_widgets (w t : ℕ) : ℕ := w * t
def tuesday_widgets (w t : ℕ) : ℕ := (w + 5) * (t - 3)

theorem widget_difference (h : w = 3 * t) :
  monday_widgets w t - tuesday_widgets w t = 4 * t + 15 :=
by
  sorry

end widget_difference_l533_533400


namespace sin_cos_half_sum_l533_533605

theorem sin_cos_half_sum (α β : ℝ)
  (h1 : sin α + sin β = -21 / 65)
  (h2 : cos α + cos β = -27 / 65)
  (hα : 5 / 2 * real.pi < α ∧ α < 3 * real.pi)
  (hβ : -1 / 2 * real.pi < β ∧ β < 0) :
  sin ((α + β) / 2) = -7 / real.sqrt 130 ∧ cos ((α + β) / 2) = -9 / real.sqrt 130 :=
  sorry

end sin_cos_half_sum_l533_533605


namespace inequality_proof_l533_533988

theorem inequality_proof (a b x : ℝ) (h : a > b) : a * 2 ^ x > b * 2 ^ x :=
sorry

end inequality_proof_l533_533988


namespace find_T5_l533_533806

variable (x y : ℝ)

-- Define the sequence terms
def T1 := x + y^2
def T2 := x - 2y
def T3 := 2x - 3y^2
def T4 := 3x - 4y

-- Placeholder definition for T5 to state the theorem
def T5 := 4x - 5y^2

-- The theorem stating that T5 is the fifth term of the sequence based on the pattern
theorem find_T5 : T5 x y = 4x - 5y^2 := by
  sorry

end find_T5_l533_533806


namespace orthocenter_midpoint_incircle_l533_533036

theorem orthocenter_midpoint_incircle 
  {A B C H M N K L F J : Point} 
  (hH : orthocenter H (Triangle.mk A B C))
  (hM : midpoint M A B)
  (hN : midpoint N A C)
  (hH_in_BMNC : inside_quadrilateral H (Quadrilateral.mk B M N C))
  (hBMH_tangent : tangent_circumcircle (Triangle.mk B M H))
  (hCNH_tangent : tangent_circumcircle (Triangle.mk C N H))
  (h_parallel_KL : parallel (line_through H K) (line_through B C))
  (h_parallel_LH : parallel (line_through H L) (line_through B C))
  (hF_intersection : intersects_at (line_through M K) (line_through N L) F)
  (hJ_incenter : incenter J (Triangle.mk M H N)) :
  dist F J = dist F A := by
sorry

end orthocenter_midpoint_incircle_l533_533036


namespace polynomial_divisibility_l533_533982

open Polynomial

def P (n : ℕ) : Polynomial ℤ :=
  ∑ i in Finset.range n, x^(2 * i)

def Q (n : ℕ) : Polynomial ℤ :=
  ∑ i in Finset.range n, x^i

theorem polynomial_divisibility (n : ℕ) (h : 0 < n) :
  (Q n ∣ P n) ↔ odd n :=
sorry

end polynomial_divisibility_l533_533982


namespace angle_BAC_is_30_degrees_l533_533337

variables {A B C D : Type*}
variables [planar_geometrical_space : plane_geometry A]
open planar_geometrical_space

def regular_hexagon (P : A) : Prop := -- Regular hexagon definition
sorry

def square_inside_hexagon (Q : A) : Prop := -- Square inside hexagon touching at vertex and parallel side definition
sorry

def is_isosceles_triangle (Δ : triangle A) : Prop := -- Isosceles triangle definition
sorry

theorem angle_BAC_is_30_degrees 
  (P : A) (Q : A) (hex : regular_hexagon P) (square : square_inside_hexagon Q) : 
  ∃θ : ℝ, θ = 30 := 
by
  -- Geometric proof would go here
  sorry

end angle_BAC_is_30_degrees_l533_533337


namespace victor_wins_ratio_l533_533127

theorem victor_wins_ratio (victor_wins friend_wins : ℕ) (hvw : victor_wins = 36) (fw : friend_wins = 20) : (victor_wins : ℚ) / friend_wins = 9 / 5 :=
by
  sorry

end victor_wins_ratio_l533_533127


namespace find_height_CD_l533_533923

variable (p t1 t2 : ℝ)
variable (h_par : 0 < p)
variable (A : ℝ × ℝ := (2 * p * t1, 2 * p * t1^2))
variable (C : ℝ × ℝ := (2 * p * t2, 2 * p * t2^2))
variable (B : ℝ × ℝ := (-2 * p * t1, 2 * p * t1^2))

noncomputable def height_CD : ℝ := 2 * p

theorem find_height_CD
  (h_AC_perp_BC : (2 * p * (t2 - t1)) * (2 * p * (t2 + t1)) +
                  ((2 * p * t2^2 - 2 * p * t1^2) * (2 * p * t2^2 - 2 * p * t1^2)) = 0) :
  height_CD p t1 t2 = 2 * p := sorry

end find_height_CD_l533_533923


namespace curtain_price_l533_533600

theorem curtain_price
  (C : ℝ)
  (h1 : 2 * C + 9 * 15 + 50 = 245) :
  C = 30 :=
sorry

end curtain_price_l533_533600


namespace length_bisector_C_l533_533366

noncomputable theory

variables (A B C D : Type*) [real_euclidean α]
variables {alpha beta : ℝ}

-- Definition of a triangle with given angles
def triangle_ABC (A B C : Type*) [real_euclidean α] :=
  (angle A B C = 40 * π / 180) ∧
  (angle B C A = 20 * π / 180) ∧
  (AB - BC = 4)

-- The theorem we want to prove
theorem length_bisector_C
  (A B C D : Type*) [real_euclidean α]
  (hABC : triangle_ABC A B C) :
  length (bisector (angle A B C)) = 4 :=
sorry

end length_bisector_C_l533_533366


namespace length_of_plot_l533_533903

theorem length_of_plot (W P C r : ℝ) (hW : W = 65) (hP : P = 2.5) (hC : C = 340) (hr : r = 0.4) :
  let L := (C / r - (W + 2 * P) * P) / (W - 2 * P)
  L = 100 :=
by
  sorry

end length_of_plot_l533_533903


namespace avg_speed_from_x_to_y_l533_533508

-- Given conditions
variables (D : ℝ) (S : ℝ)

-- Conditions in the problem
def speed_y_to_x (D : ℝ) : ℝ := 34 -- Speed from y to x
def avg_speed_total (D : ℝ) (S : ℝ) : ℝ := 38 -- Average speed of the whole journey

-- Proof problem statement: Prove that the speed from x to y is approximately 43.07
theorem avg_speed_from_x_to_y (D : ℝ) (S : ℝ) :
  (avg_speed_total D S) = 2 * D / (D / S + D / speed_y_to_x D) → S ≈ 43.07 := sorry

end avg_speed_from_x_to_y_l533_533508


namespace inequality_holds_l533_533650

theorem inequality_holds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  (a + (1 / b))^2 + (b + (1 / c))^2 + (c + (1 / a))^2 ≥ 3 * (a + b + c + 1) := by
  sorry

end inequality_holds_l533_533650


namespace finalStoresAtEndOf2020_l533_533081

def initialStores : ℕ := 23
def storesOpened2019 : ℕ := 5
def storesClosed2019 : ℕ := 2
def storesOpened2020 : ℕ := 10
def storesClosed2020 : ℕ := 6

theorem finalStoresAtEndOf2020 : initialStores + (storesOpened2019 - storesClosed2019) + (storesOpened2020 - storesClosed2020) = 30 :=
by
  sorry

end finalStoresAtEndOf2020_l533_533081


namespace minimum_marked_cells_k_l533_533144

def cell := (ℕ × ℕ)

def is_marked (board : set cell) (cell : cell) : Prop :=
  cell ∈ board

def touches_marked_cell (board : set cell) (fig : set cell) : Prop :=
  ∃ (c : cell), c ∈ fig ∧ is_marked(board, c)

def four_cell_figures (fig : set cell) : Prop :=
  (fig = { (0, 0), (0, 1), (1, 0), (1, 1) }) ∨
  (fig = { (0, 0), (1, 0), (2, 0), (3, 0) }) ∨
  (fig = { (0, 0), (0, 1), (0, 2), (0, 3) }) ∨
  -- Add other rotations and flipped configurations if needed

def board12x12 :=
  { (i, j) | i < 12 ∧ j < 12 }

theorem minimum_marked_cells_k :
  ∃ (k : ℕ) (marked_cells : (fin 144) → bool),
  k = 48 ∧
  (∀ (fig : set cell),
    four_cell_figures fig →
    ∀ (x y : ℕ), x < 12 ∧ y < 12 →
      touches_marked_cell
        { cell | marked_cells (fin.mk (cell.1 * 12 + cell.2) _) }
        (image (λ (rc : cell), (rc.1 + x, rc.2 + y)) fig)
  ) :=
begin
  sorry
end

end minimum_marked_cells_k_l533_533144


namespace father_l533_533962

variable (R F M : ℕ)
variable (h1 : F = 4 * R)
variable (h2 : 4 * R + 8 = M * (R + 8))
variable (h3 : 4 * R + 16 = 2 * (R + 16))

theorem father's_age_ratio (hR : R = 8) : (F + 8) / (R + 8) = 5 / 2 := by
  sorry

end father_l533_533962


namespace diagonal_divides_parallelogram_l533_533778

theorem diagonal_divides_parallelogram (ABCD : Type*) [parallelogram ABCD] (A B C D : ABCD)
  (BD : diagonal A B D) : triangle_congruent ABD BDC :=
begin
  sorry
end

end diagonal_divides_parallelogram_l533_533778


namespace correct_number_of_six_letter_words_l533_533595

def number_of_six_letter_words (alphabet_size : ℕ) : ℕ :=
  alphabet_size ^ 4

theorem correct_number_of_six_letter_words :
  number_of_six_letter_words 26 = 456976 :=
by
  -- We write 'sorry' to omit the detailed proof.
  sorry

end correct_number_of_six_letter_words_l533_533595


namespace ratio_of_boys_to_girls_l533_533216

theorem ratio_of_boys_to_girls (G B : ℕ) (hg : G = 30) (hb : B = G + 18) : B / G = 8 / 5 :=
by
  sorry

end ratio_of_boys_to_girls_l533_533216


namespace landscape_avoid_repetition_l533_533480

theorem landscape_avoid_repetition :
  let frames : ℕ := 5
  let days_per_month : ℕ := 30
  (Nat.factorial frames) / days_per_month = 4 := by
  sorry

end landscape_avoid_repetition_l533_533480


namespace find_x_l533_533713

theorem find_x (x : ℕ) (h : 5 * x + 4 * x + x + 2 * x = 360) : x = 30 :=
by
  sorry

end find_x_l533_533713


namespace range_of_a_l533_533328

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 :=
sorry

end range_of_a_l533_533328


namespace sum_of_real_roots_interval_l533_533617

theorem sum_of_real_roots_interval :
  (∑ x in {x | sin (π * (x^2 - x + 1)) = sin (π * (x - 1)) ∧ x ∈ Icc 0 2}, x) = 3 + Real.sqrt 3 :=
sorry

end sum_of_real_roots_interval_l533_533617


namespace area_shaded_region_concentrics_l533_533432

noncomputable def area_of_shaded_region (r R : ℝ) : ℝ :=
  π * (R^2 - r^2)

theorem area_shaded_region_concentrics (r R : ℝ) (h : r^2 + 2500 = R^2) 
  (h1 : ∀ C D : ℝ, C = 100 → C = D) :
  area_of_shaded_region r R = 2500 * π :=
by
  sorry

end area_shaded_region_concentrics_l533_533432


namespace cake_cost_is_20_l533_533053

-- Define the given conditions
def total_budget : ℕ := 50
def additional_needed : ℕ := 11
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5

-- Define the derived conditions
def total_cost : ℕ := total_budget + additional_needed
def combined_bouquet_balloons_cost : ℕ := bouquet_cost + balloons_cost
def cake_cost : ℕ := total_cost - combined_bouquet_balloons_cost

-- The theorem to be proved
theorem cake_cost_is_20 : cake_cost = 20 :=
by
  -- proof steps are not required
  sorry

end cake_cost_is_20_l533_533053


namespace imaginary_part_of_z_l533_533434

-- Define the complex number z
def z : Complex := Complex.mk 3 (-4)

-- State the proof goal
theorem imaginary_part_of_z : z.im = -4 :=
by
  sorry

end imaginary_part_of_z_l533_533434


namespace trapezoid_DBCE_area_l533_533357

noncomputable def trapezoid_area (ABC_area smallest_triangle_area : ℝ) (num_smallest_triangles : ℕ) : ℝ :=
  let total_smallest_area := num_smallest_triangles * smallest_triangle_area
  let ADE_area := 8 * smallest_triangle_area
  ABC_area - ADE_area

theorem trapezoid_DBCE_area
  (ABC : Type)
  (similar_triangles : ∀ (T : Type), T → Prop)
  (is_equilateral : ∀ (T : Type) (a b c : ℝ), T ∧ (a = b ∧ b = c) → Prop)
  (ABC_properties : is_equilateral ABC 1 1 1)
  (area_ABC : real)
  (area_smallest_triangle : ℝ)
  (num_smallest_triangles : ℕ)
  (h1 : similar_triangles ABC)
  (h2 : area_smallest_triangle = 2)
  (h3 : area_ABC = 96)
  (h4 : num_smallest_triangles = 12) : 
  trapezoid_area area_ABC area_smallest_triangle num_smallest_triangles = 80 :=
by
  sorry

end trapezoid_DBCE_area_l533_533357


namespace total_games_played_l533_533218

def games_attended : ℕ := 14
def games_missed : ℕ := 25

theorem total_games_played : games_attended + games_missed = 39 :=
by
  sorry

end total_games_played_l533_533218


namespace triangle_AB_equal_BE_l533_533403

/-- Given a triangle ABC with E on side BC, F on the angle bisector BD, 
and EF parallel to AC with AF = AD, prove that AB = BE. -/
theorem triangle_AB_equal_BE 
  {A B C E D F : Point} 
  (hE : E ∈ line_segment B C)
  (hF : F ∈ line_segment B D)
  (h_parallel : parallel (line_through E F) (line_through A C))
  (h_AF_AD : distance A F = distance A D) 
  : distance A B = distance B E := 
  sorry

end triangle_AB_equal_BE_l533_533403


namespace small_sphere_radius_in_unit_cube_l533_533721

theorem small_sphere_radius_in_unit_cube
    (r : ℝ)  -- r is the radius of a small sphere
    (unit_cube : set (ℝ × ℝ × ℝ))  -- Representing the unit cube
    (inscribed_sphere : ℝ × ℝ × ℝ → Prop)  -- Inscribed sphere predicate
    (small_sphere1 : ℝ × ℝ × ℝ → Prop)  -- Small sphere 1 predicate
    (small_sphere2 : ℝ × ℝ × ℝ → Prop)  -- Small sphere 2 predicate
    (small_sphere3 : ℝ × ℝ × ℝ → Prop)  -- Small sphere 3 predicate
    (small_sphere4 : ℝ × ℝ × ℝ → Prop)  -- Small sphere 4 predicate
    (small_sphere5 : ℝ × ℝ × ℝ → Prop)  -- Small sphere 5 predicate
    (small_sphere6 : ℝ × ℝ × ℝ → Prop)  -- Small sphere 6 predicate
    (small_sphere7 : ℝ × ℝ × ℝ → Prop)  -- Small sphere 7 predicate
    (small_sphere8 : ℝ × ℝ × ℝ → Prop)  -- Small sphere 8 predicate
    (tangent_condition : ∀ s, s ∈ { small_sphere1, small_sphere2, small_sphere3, small_sphere4, small_sphere5, small_sphere6, small_sphere7, small_sphere8 } → r = 1/2 * (sqrt(3) - 1))
    (unit_cube_condition : unit_cube = set.univ)  -- The unit cube is the whole universe in this context
    (inscribed_sphere_condition : ∀ p, inscribed_sphere p → dist p (0.5, 0.5, 0.5) = 1 / 2)
    (small_spheres_tangent_to_faces :
        (dist (0, 0, 0) (0, 0, 1) = (1 / 2 - r)) ∧
        (dist (0, 0, 0) (0, 1, 0) = (1 / 2 - r)) ∧
        (dist (0, 0, 0) (1, 0, 0) = (1 / 2 - r)) ∧
        -- Other seven tangency conditions omitted for brevity.
        (dist (1, 1, 1) (1, 1, 0) = (1 / 2 - r))
    ) :
    r = 1/2 * (sqrt(3) - 1) :=
by
  sorry

end small_sphere_radius_in_unit_cube_l533_533721


namespace find_circle_radius_l533_533289

theorem find_circle_radius (C D : Circle) (O P Q R L : Point) 
  (h1 : O ∈ C.circumference) 
  (h2 : P ∈ (circle_intersect C D O))
  (h3 : Q ∈ (circle_intersect C D O))
  (h4 : R ∈ (circle_intersect (Circle.mk Q D.radius) D))
  (h5 : L ∈ (line_intersect PR C.circumference)) :
  QL = C.radius :=
  sorry

end find_circle_radius_l533_533289


namespace gum_left_after_sharing_l533_533202

-- Define the initial state of Adrianna's gum and the changes to it
def initial_gum : Nat := 10
def additional_gum : Nat := 3
def given_out_gum : Nat := 11

-- Define the final state of Adrianna's gum
def final_gum : Nat := initial_gum + additional_gum - given_out_gum

-- Prove that Adrianna ends up with 2 pieces of gum under the given conditions
theorem gum_left_after_sharing :
  final_gum = 2 :=
by 
  -- Since this is just the statement and not the proof, we end with sorry.
  sorry

end gum_left_after_sharing_l533_533202


namespace maximize_profit_l533_533884

-- Statement of the problem
def fixed_cost : ℝ := 20000
def cost_per_unit : ℝ := 100
def revenue (x : ℝ) : ℝ := -x^3 / 9000 + 400 * x

def cost (x : ℝ) : ℝ := fixed_cost + cost_per_unit * x

def profit (x : ℝ) : ℝ := revenue(x) - cost(x)

theorem maximize_profit : ∀ x : ℝ, (0 ≤ x ∧ x ≤ 390) → x = 300 → ∀ y : ℝ, (0 ≤ y ∧ y ≤ 390) → profit(x) ≥ profit(y) :=
by
  sorry

end maximize_profit_l533_533884


namespace sqrt_36_eq_6_l533_533153

theorem sqrt_36_eq_6 : real.sqrt 36 = 6 :=
sorry

end sqrt_36_eq_6_l533_533153


namespace sum_largest_odd_divisors_eq_square_l533_533620

-- Define the function that gives the largest odd divisor of a number
def largest_odd_divisor (k : ℕ) : ℕ :=
  if h : k = 0 then 0 else
  let rec aux (m : ℕ) : ℕ :=
    if m % 2 = 1 then m else aux (m / 2)
  in aux k

-- Define the sum of the largest odd divisors from n+1 to 2n
def sum_largest_odd_divisors (n : ℕ) : ℕ :=
  (finset.range (2 * n + 1)).filter (λ i, n < i).sum largest_odd_divisor

-- Prove that the sum of the largest odd divisors from n+1 to 2n is n^2
theorem sum_largest_odd_divisors_eq_square (n : ℕ) : sum_largest_odd_divisors n = n^2 := sorry

end sum_largest_odd_divisors_eq_square_l533_533620


namespace general_formula_a_n_sum_first_n_terms_b_l533_533997

open Nat

-- Definitions from conditions.
def S (n : ℕ+) : ℕ := 2^(n + 1) - 2

-- Prove the general formula of a_n
theorem general_formula_a_n (n : ℕ+) : ∃ a : ℕ → ℕ, (∀ n ≥ 1, a n = 2^n) ∧ (S n = ∑ i in finset.range n.val + 1, a i) :=
by
  sorry

-- Prove the sum of the first n terms of b_n is (n-1)2^{n+1} + 2
theorem sum_first_n_terms_b (n : ℕ+) : ∃ b : ℕ → ℕ, (∀ n ≥ 1, b n = n * 2^n) ∧ (∑ i in finset.range n.val + 1, b i = (n.val - 1) * 2^(n + 1) + 2) :=
by
  sorry

end general_formula_a_n_sum_first_n_terms_b_l533_533997


namespace problem_l533_533280

theorem problem (a b c d : ℝ) (h1 : b + c = 7) (h2 : c + d = 5) (h3 : a + d = 2) : a + b = 4 :=
sorry

end problem_l533_533280


namespace company_production_average_l533_533162

theorem company_production_average (n : ℕ) 
  (h1 : (50 * n) / n = 50) 
  (h2 : (50 * n + 105) / (n + 1) = 55) :
  n = 10 :=
sorry

end company_production_average_l533_533162


namespace ice_cream_cones_sold_l533_533402

theorem ice_cream_cones_sold (T W : ℕ) (h1 : W = 2 * T) (h2 : T + W = 36000) : T = 12000 :=
by
  sorry

end ice_cream_cones_sold_l533_533402


namespace no_integer_roots_l533_533633

def cubic_polynomial (a b c d x : ℤ) : ℤ :=
  a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots (a b c d : ℤ) (h1 : cubic_polynomial a b c d 1 = 2015) (h2 : cubic_polynomial a b c d 2 = 2017) :
  ∀ x : ℤ, cubic_polynomial a b c d x ≠ 2016 :=
by
  sorry

end no_integer_roots_l533_533633


namespace students_who_wore_blue_lipstick_l533_533059

theorem students_who_wore_blue_lipstick (total_students : ℕ) (h1 : total_students = 200) : 
  ∃ blue_lipstick_students : ℕ, blue_lipstick_students = 5 :=
by
  have colored_lipstick_students := total_students / 2
  have red_lipstick_students := colored_lipstick_students / 4
  let blue_lipstick_students := red_lipstick_students / 5
  have h2 : blue_lipstick_students = 5 :=
    calc blue_lipstick_students
          = (total_students / 2) / 4 / 5 : by sorry -- detailed calculation steps omitted
  use blue_lipstick_students
  exact h2

end students_who_wore_blue_lipstick_l533_533059


namespace find_upper_number_l533_533213

-- Definitions from conditions
def three_sections_with_ten_beads (s1 s2 s3 : ℕ) (b1 b2 b3 : ℕ) : Prop :=
  s1 = 10 ∧ s2 = 10 ∧ s3 = 10 ∧ b1 = 10 ∧ b2 = 10 ∧ b3 = 10

def distinct_digits (n : ℕ) : Prop :=
  let digits := nat.digits 10 n in
  list.nodup digits

def upper_is_multiple_of_lower (upper lower : ℕ) : Prop :=
  ∃ k : ℕ, upper = k * lower

def sum_equals_1110 (upper lower : ℕ) : Prop :=
  upper + lower = 1110

-- Final theorem statement
theorem find_upper_number (abc upper : ℕ) : 
  three_sections_with_ten_beads 10 10 10 10 10 10 → 
  distinct_digits upper → 
  upper_is_multiple_of_lower upper abc → 
  sum_equals_1110 upper abc → 
  upper = 925 :=
by 
  intros 
  sorry

end find_upper_number_l533_533213


namespace trigonometric_identity_l533_533662

open Real

theorem trigonometric_identity :
  (sin (20 * π / 180) * sin (80 * π / 180) - cos (160 * π / 180) * sin (10 * π / 180) = 1 / 2) :=
by
  -- Trigonometric calculations
  sorry

end trigonometric_identity_l533_533662


namespace min_value_of_parabola_l533_533136

theorem min_value_of_parabola : ∃ x : ℝ, ∀ y : ℝ, y = 3 * x^2 - 18 * x + 244 → y = 217 := by
  sorry

end min_value_of_parabola_l533_533136


namespace average_score_for_first_6_matches_l533_533082

theorem average_score_for_first_6_matches:
  (A : ℚ) (avg_10_matches : ℚ) (avg_last_4_matches : ℚ) :
  avg_10_matches = 38.9 →
  avg_last_4_matches = 34.25 →
  (10 * avg_10_matches - 4 * avg_last_4_matches) / 6 = 42 :=
by
  intros A avg_10_matches avg_last_4_matches h1 h2
  sorry

end average_score_for_first_6_matches_l533_533082


namespace number_of_circles_l533_533259

noncomputable def equation (a : ℝ) (x y : ℝ) :=
  x^2 + y^2 + a * x + 2 * a * y + 2 * a^2 + a - 1

theorem number_of_circles :
  let possible_a := [-2, 0, 1, 3/4] in
  (∃! a ∈ possible_a, ∃ x y : ℝ, equation a x y = 0 ∧ (∃ r : ℝ, r > 0 ∧ ∀ (xc yc : ℝ), (x - xc)^2 + (y - yc)^2 = r^2)) :=
sorry

end number_of_circles_l533_533259


namespace total_ways_to_turn_on_lights_l533_533832

theorem total_ways_to_turn_on_lights (n : ℕ) (hswitch : n = 4) : (2 ^ n) - 1 = 15 :=
by
  sorry
end

end total_ways_to_turn_on_lights_l533_533832


namespace maximal_value_ratio_l533_533732

theorem maximal_value_ratio (a b c h : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_altitude : h = (a * b) / c) :
  ∃ θ : ℝ, a = c * Real.cos θ ∧ b = c * Real.sin θ ∧ (1 < Real.cos θ + Real.sin θ ∧ Real.cos θ + Real.sin θ ≤ Real.sqrt 2) ∧
  ( Real.cos θ * Real.sin θ = (1 + 2 * Real.cos θ * Real.sin θ - 1) / 2 ) → 
  (c + h) / (a + b) ≤ 3 * Real.sqrt 2 / 4 :=
sorry

end maximal_value_ratio_l533_533732


namespace farmer_land_acres_l533_533890

theorem farmer_land_acres
  (initial_ratio_corn : Nat)
  (initial_ratio_sugar_cane : Nat)
  (initial_ratio_tobacco : Nat)
  (new_ratio_corn : Nat)
  (new_ratio_sugar_cane : Nat)
  (new_ratio_tobacco : Nat)
  (additional_tobacco_acres : Nat)
  (total_land_acres : Nat) :
  initial_ratio_corn = 5 →
  initial_ratio_sugar_cane = 2 →
  initial_ratio_tobacco = 2 →
  new_ratio_corn = 2 →
  new_ratio_sugar_cane = 2 →
  new_ratio_tobacco = 5 →
  additional_tobacco_acres = 450 →
  total_land_acres = 1350 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end farmer_land_acres_l533_533890


namespace ratio_area_proof_l533_533879

noncomputable def ratio_area_circle_hexagon (r : ℝ) : ℝ :=
  let area_smaller_circle := π * r ^ 2
  let side_length_larger_hexagon := (2 * r) / (Real.sqrt 3)
  let area_larger_hexagon := (3 * Real.sqrt 3 / 2) * (side_length_larger_hexagon ^ 2)
  area_smaller_circle / area_larger_hexagon

theorem ratio_area_proof (r : ℝ) (h : r > 0) : ratio_area_circle_hexagon r = π / (2 * Real.sqrt 3) := by
  sorry

end ratio_area_proof_l533_533879


namespace fill_time_l533_533774

theorem fill_time
    (num_barrels : ℕ) (gallons_per_barrel : ℕ) (flow_rate : ℕ) (total_time : ℕ)
    (h1 : num_barrels = 4)
    (h2 : gallons_per_barrel = 7)
    (h3 : flow_rate = 3.5)
    (h4 : total_time = 8) :
    (num_barrels * gallons_per_barrel) / flow_rate = total_time :=
by
    sorry

end fill_time_l533_533774


namespace problem1_problem2_problem3_l533_533344

noncomputable def arrangement_adjacent (n : ℕ) (A B : ℕ) : ℕ :=
  if A = B then nat.factorial (n - 1) else 2 * nat.factorial (n - 1)

theorem problem1 : arrangement_adjacent 5 1 2 = 48 := 
  by 
    -- This is a placeholder to demonstrate the concept, actual proof is not included.
    sorry

noncomputable def arrangement_non_adjacent (n : ℕ) (A B : ℕ) : ℕ :=
  if A = B then 0 else let m := n - 2 in nat.factorial (n - 2) * (n - 2).choose 2

theorem problem2 : arrangement_non_adjacent 5 1 2 = 72 :=
  by 
    -- This is a placeholder to demonstrate the concept, actual proof is not included.
    sorry

noncomputable def arrangement_not_middle_and_not_ends (n : ℕ) (A B : ℕ) : ℕ :=
  let middle_position := if A = (n + 1) / 2 then 0 else nat.factorial (n - 1)
  let end_positions := if B = 1 ∨ B = n then 0 else 2 * nat.factorial (n - 1)
  middle_position + end_positions - nat.factorial (n - 2)

theorem problem3 : arrangement_not_middle_and_not_ends 5 1 2 = 60 :=
  by 
    -- This is a placeholder to demonstrate the concept, actual proof is not included.
    sorry

end problem1_problem2_problem3_l533_533344


namespace min_marked_necessary_l533_533148

-- A 12x12 board represented as a set of coordinates
def board_12x12 := fin 12 × fin 12

-- A four-cell figure's possible placements, given by its upper-left corner position.
def figure_placements : list (set (fin 12 × fin 12)) :=
  [(0,0), (0,1), (1,0), (1,1), (1,2), (2,0), (2,1)].map (λ c, {(0,0), (0,1), (1,0), (1,1)}.image (prod.add c))

-- The marking condition function
def well_marked (marked : set (fin 12 × fin 12)) :=
  ∀ p ∈ board_12x12, ¬(figure_placements.map (λ s, disjoint marked s)).all id

-- The core hypothesis and the final theorem
theorem min_marked_necessary : ∃ k : ℕ, k = 48 ∧ ∃ marked : set (fin 12 × fin 12), well_marked marked ∧ marked.card = k :=
by sorry

end min_marked_necessary_l533_533148


namespace a_arithmetic_sequence_sum_b_sequence_l533_533445

-- Sequence a_n
def S (n : ℕ) : ℕ := n^2 + n
def a (n : ℕ) : ℕ := if n = 1 then S 1 else S n - S (n - 1)

-- Sequence b_n
def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- Theorem that a_n is arithmetic sequence
theorem a_arithmetic_sequence : ∃ d : ℕ, ∀ n ≥ 2, a (n + 1) - a n = d := 
sorry

-- Theorem for the sum of first n terms of b_n
def T (n : ℕ) : ℚ := (1/4) * (1 - 1 / (n + 1))

theorem sum_b_sequence (n : ℕ) : ∑ i in Finset.range n, b (i + 1) = T n :=
sorry

end a_arithmetic_sequence_sum_b_sequence_l533_533445


namespace solve_system_of_equations_l533_533415

-- Defining the system of quadratic equations
def eq1 (x y z : ℝ) := x^2 - (y + z + yz) * x + (y + z) * yz = 0
def eq2 (x y z : ℝ) := y^2 - (z + x + zx) * y + (z + x) * zx = 0
def eq3 (x y z : ℝ) := z^2 - (x + y + xy) * z + (x + y) * xy = 0

-- Main theorem statement of solutions
theorem solve_system_of_equations (x y z : ℝ) :
  (eq1 x y z) ∧ (eq2 x y z) ∧ (eq3 x y z) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 1 ∧ y = -1 ∧ z = -1) ∨
  (x = -1 ∧ y = 1 ∧ z = -1) ∨
  (x = -1 ∧ y = -1 ∧ z = 1) ∨
  (x = 1 ∧ y = 1/2 ∧ z = 1/2) ∨
  (x = 1/2 ∧ y = 1 ∧ z = 1/2) ∨
  (x = 1/2 ∧ y = 1/2 ∧ z = 1) :=
by sorry

end solve_system_of_equations_l533_533415


namespace eval_expression_l533_533602

noncomputable def complex_i : ℂ := complex.I

theorem eval_expression : 
  let i := complex_i in
  (i^255 + i^256 + i^257 + i^258 + i^259) = -i :=
by
  let i : ℂ := complex_i
  have h : i^4 = 1 := by simp [complex.pow_nat, complex.I_mul_I, complex.I_cos, complex.I_sin, complex_exp_pi_div_four, complex.one]
  sorry

end eval_expression_l533_533602


namespace simplify_fractions_l533_533785

theorem simplify_fractions :
  (270 / 18) * (7 / 210) * (9 / 4) = 9 / 8 :=
by sorry

end simplify_fractions_l533_533785


namespace number_of_paths_l533_533079

noncomputable def paths_from_C_to_D : ℕ :=
  nat.choose 8 2

theorem number_of_paths (C D : ℕ) (h : D = C + 6 + 2) : paths_from_C_to_D = 28 :=
by {
  -- The proof will go here, using the properties of binomial coefficients.
  sorry,
}

end number_of_paths_l533_533079


namespace ceil_neg_sqrt_equiv_l533_533234

theorem ceil_neg_sqrt_equiv : ⌈-real.sqrt (100 / 9)⌉ = -3 := by
  have sqrt_val : real.sqrt (100 / 9) = 10 / 3 := by
    sorry
  rw [sqrt_val]
  sorry

end ceil_neg_sqrt_equiv_l533_533234


namespace no_integer_solution_l533_533135

theorem no_integer_solution :
  ¬(∃ x : ℤ, 7 - 3 * (x^2 - 2) > 19) :=
by
  sorry

end no_integer_solution_l533_533135


namespace max_distance_z_squared_z_four_l533_533745

noncomputable def z : ℂ := sorry -- complex number such that |z| = 3

theorem max_distance_z_squared_z_four (z : ℂ) (hz : |z| = 3) :
  ∃ w : ℝ, w = (36 * Real.sqrt 26) / 5 ∧
  ∀ z : ℂ, |(2 + complex.I) * z^2 - z^4| ≤ w := sorry

end max_distance_z_squared_z_four_l533_533745


namespace shopkeeper_net_loss_percent_l533_533546

theorem shopkeeper_net_loss_percent (cp : ℝ)
  (sp1 sp2 sp3 sp4 : ℝ)
  (h_cp : cp = 1000)
  (h_sp1 : sp1 = cp * 1.1)
  (h_sp2 : sp2 = cp * 0.9)
  (h_sp3 : sp3 = cp * 1.2)
  (h_sp4 : sp4 = cp * 0.75) :
  ((cp + cp + cp + cp) - (sp1 + sp2 + sp3 + sp4)) / (cp + cp + cp + cp) * 100 = 1.25 :=
by sorry

end shopkeeper_net_loss_percent_l533_533546


namespace sum_positive_132_l533_533711

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem sum_positive_132 {a: ℕ → ℝ}
  (h1: a 66 < 0)
  (h2: a 67 > 0)
  (h3: a 67 > |a 66|):
  ∃ n, ∀ k < n, S k > 0 :=
by
  have h4 : (a 67 - a 66) > 0 := sorry
  have h5 : a 67 + a 66 > 0 := sorry
  have h6 : 66 * (a 67 + a 66) > 0 := sorry
  have h7 : S 132 = 66 * (a 67 + a 66) := sorry
  existsi 132
  intro k hk
  sorry

end sum_positive_132_l533_533711


namespace gcd_pow_diff_l533_533134

theorem gcd_pow_diff :
  gcd (2 ^ 2100 - 1) (2 ^ 2091 - 1) = 511 := 
sorry

end gcd_pow_diff_l533_533134


namespace volume_units_l533_533238

theorem volume_units :
  ∃ U_oj U_wb, (U_oj = "milliliters") ∧ (U_wb = "liters") ∧ (V_oj = 500) ∧ (V_wb = 3) :=
by
  use "milliliters"
  use "liters"
  exact ⟨rfl, rfl, rfl, rfl⟩

end volume_units_l533_533238


namespace segment_construction_l533_533300

-- Assume a and b are given segments
variables {a b : ℝ}

-- The segment c is defined as:
def c : ℝ := (a^4 + b^4)^(1/4)

-- The theorem to be proved
theorem segment_construction : c = (a^4 + b^4)^(1/4) :=
sorry

end segment_construction_l533_533300


namespace weight_after_one_year_l533_533937

def initial_weight : ℕ := 250
def training_weight_loss : List ℕ := [8, 5, 7, 6, 8, 7, 5, 7, 4, 6, 5, 7]
def diet_weight_loss : ℕ := 3

theorem weight_after_one_year : 
  let total_training_loss := (training_weight_loss.foldl (· + ·) 0)
  let total_diet_loss := diet_weight_loss * 12
  let total_loss := total_training_loss + total_diet_loss
  let final_weight := initial_weight - total_loss
  final_weight = 139 :=
by
  have h1 : total_training_loss = 8 + 5 + 7 + 6 + 8 + 7 + 5 + 7 + 4 + 6 + 5 + 7 := rfl
  have h2 : total_training_loss = 75 := by simp [h1]
  have h3 : total_diet_loss = 3 * 12 := rfl
  have h4 : total_diet_loss = 36 := by simp [h3]
  have h5 : total_loss = 75 + 36 := rfl
  have h6 : total_loss = 111 := by simp [h5]
  have h7 : final_weight = 250 - 111 := rfl
  have h8 : final_weight = 139 := by simp [h7]
  exact h8

end weight_after_one_year_l533_533937


namespace avg_new_students_l533_533428

-- Definitions for conditions
def orig_strength : ℕ := 17
def orig_avg_age : ℕ := 40
def new_students_count : ℕ := 17
def decreased_avg_age : ℕ := 36 -- given that average decreases by 4 years, i.e., 40 - 4

-- Definition for the original total age
def total_age_orig : ℕ := orig_strength * orig_avg_age

-- Definition for the total number of students after new students join
def total_students : ℕ := orig_strength + new_students_count

-- Definition for the total age after new students join
def total_age_new : ℕ := total_students * decreased_avg_age

-- Definition for the total age of new students
def total_age_new_students : ℕ := total_age_new - total_age_orig

-- Definition for the average age of new students
def avg_age_new_students : ℕ := total_age_new_students / new_students_count

-- Lean theorem stating the proof problem
theorem avg_new_students : 
  avg_age_new_students = 32 := 
by sorry

end avg_new_students_l533_533428


namespace unique_solution_4a_5b_minus_3c_11d_eq_1_l533_533239

theorem unique_solution_4a_5b_minus_3c_11d_eq_1 (a b c d : ℕ) (h : 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d):
  4^a * 5^b - 3^c * 11^d = 1 → a = 1 ∧ b = 2 ∧ c = 2 ∧ d = 1 :=
by
  intros h
  linarith
  sorry

end unique_solution_4a_5b_minus_3c_11d_eq_1_l533_533239


namespace problem_f_sin_pi_over_6_l533_533987
noncomputable def f : ℝ → ℝ := λ t, (t^2 - 1) / 2

theorem problem_f_sin_pi_over_6 :
  f (Real.sin (Real.pi / 6)) = -3/8 :=
by
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  rw [h1]
  have h2 : f (1 / 2) = -3 / 8 := by sorry
  rw [h2]
  exact rfl

end problem_f_sin_pi_over_6_l533_533987


namespace fruit_basket_combinations_l533_533304

namespace FruitBasket

def apples := 3
def oranges := 8
def min_apples := 1
def min_oranges := 1

theorem fruit_basket_combinations : 
  (apples + 1 - min_apples) * (oranges + 1 - min_oranges) = 36 := by
  sorry

end FruitBasket

end fruit_basket_combinations_l533_533304


namespace lambda_inequality_l533_533998

noncomputable def a (n : ℕ) : ℝ :=
1 / (2 * n * (n + 1))

noncomputable def T (n : ℕ) : ℝ :=
(∑ i in Finset.range n, a (i + 1))

theorem lambda_inequality (λ : ℝ) :
  (∀ n : ℕ, 0 < n → T n > n * λ / (n^2 + 4 * n + 19)) →
  λ < 5 :=
by
  sorry

end lambda_inequality_l533_533998


namespace messages_after_noon_l533_533592

theorem messages_after_noon (t n : ℕ) (h1 : t = 39) (h2 : n = 21) : t - n = 18 := by
  sorry

end messages_after_noon_l533_533592


namespace part1_part2_l533_533045

def f (x : ℝ) (k : ℝ) : ℝ := k * (x - 1) - 2 * log x

def g (x : ℝ) : ℝ := x * exp (1 - x)

theorem part1 (k : ℝ) (h : (f 1 k = 0) ∧ (∀ x, f x k = 0 → x = 1)) : k = 2 :=
by
  sorry

theorem part2 (k : ℝ) (h : ∀ s ∈ Ioo 0 real.exp 1, ∃ t1 ∈ Ioo ((1 / real.exp 2) : ℝ) real.exp 1, ∃ t2 ∈ Ioo ((1 / real.exp 2) : ℝ) real.exp 1, f t1 k = g s ∧ f t2 k = g s ∧ t1 ≠ t2) : (3 / (real.exp 1 - 1) ≤ k) ∧ (k ≤ 3 * (real.exp 2 ^ 2 / ((real.exp 2 ^ 2) - 1))) :=
by
  sorry

end part1_part2_l533_533045


namespace ellie_investment_percentage_change_correct_l533_533693

def initial_investment : ℝ := 200

def first_year_loss (amount : ℝ) : ℝ := amount * 0.90
def second_year_gain (amount : ℝ) : ℝ := amount * 1.15
def third_year_gain (amount : ℝ) : ℝ := amount * 1.25

def final_amount : ℝ := 
  third_year_gain (second_year_gain (first_year_loss initial_investment))

def percentage_change : ℝ :=
  ((final_amount - initial_investment) / initial_investment) * 100

theorem ellie_investment_percentage_change_correct :
  percentage_change = 29.38 := by
  sorry

end ellie_investment_percentage_change_correct_l533_533693


namespace number_of_matrices_l533_533613

-- Define the set of matrices with given form and condition
def setMatrices := { M : Matrix (Fin 3) (Fin 3) (Fin 2) // 
  M 0 0 = 1 ∧
  M 1 1 = 1 ∧
  M 2 2 = 1 ∧
  M 0 1 ∈ {0,1} ∧ M 0 2 ∈ {0,1} ∧
  M 1 0 ∈ {0,1} ∧ M 1 2 ∈ {0,1} ∧
  M 2 0 ∈ {0,1} ∧ M 2 1 ∈ {0,1} ∧
  ∀ i j : Fin 3, (i ≠ j → M i = M j -> False) }

-- The proof statement
theorem number_of_matrices : Fintype.card setMatrices = 42 :=
by 
  -- Skipping the proof with sorry
  sorry

end number_of_matrices_l533_533613


namespace circle_S_radius_in_triangle_DEF_l533_533364

theorem circle_S_radius_in_triangle_DEF 
(DE DF EF : ℝ) -- Sides of the triangle
(hDE : DE = 120) (hDF : DF = 120) (hEF : EF = 80) -- Conditions/Lengths of the sides of the triangle
(radius_R : ℝ) (hr : radius_R = 20) -- Radius of circle R
(radius_S : ℝ) -- Radius of circle S
(S_form : ∃ p q l : ℕ, l.prime_factors_prod = ∏₀ l and radius_S = p - q * Real.sqrt l) -- Desired form of radius S
(RS_is_tangent : RS = radius_S + 20) (tangent_to_DE_EF : circle S is tangent to DE and EF) (inside_triangle : ∀ x ∈ ∂ circle S, x ∈ interior (triangle DEF)) :
  let s := (54 : ℝ) - 8 * Real.sqrt 41 in
  p + q * l = 338 := by
  sorry

end circle_S_radius_in_triangle_DEF_l533_533364


namespace real_solution_unique_l533_533248

theorem real_solution_unique:
  ∀ x : ℝ, (x ^ 2006 + 1) * (∑ k in finset.range (1003), x ^ (2 * (1003 - k)) + 1) = 2006 * x ^ 2005 → x = 1 :=
by
  intros x hx
  sorry

end real_solution_unique_l533_533248


namespace circle_equation_and_distances_l533_533347

noncomputable def circle_cartesian (ρ θ : ℝ) : Prop :=
  ∃ (x y : ℝ), ρ = real.sqrt (x^2 + y^2) ∧ θ = real.arctan2 y x ∧ ρ = 6 * real.sin θ

noncomputable def intersection_points (x y t : ℝ) : Prop :=
  x = 1 + (real.sqrt 2 / 2) * t ∧ y = 2 + (real.sqrt 2 / 2) * t

theorem circle_equation_and_distances:
  (∃ x y, circle_cartesian (real.sqrt (x^2 + y^2)) (real.arctan2 y x) ∧ x^2 + (y - 3)^2 = 9) ∧
  (let P := (1, 2) in
   let A := (1 + (real.sqrt 2 / 2) * real.sqrt 7, 2 + (real.sqrt 2 / 2) * real.sqrt 7) in
   let B := (1 + (real.sqrt 2 / 2) * -real.sqrt 7, 2 + (real.sqrt 2 / 2) * -real.sqrt 7) in
   |real.sqrt 7 - -real.sqrt 7| = 2 * real.sqrt 7) :=
by
  sorry

end circle_equation_and_distances_l533_533347


namespace star_4_3_l533_533682

def star (a b : ℤ) : ℤ := a^2 - a * b + b^2

theorem star_4_3 : star 4 3 = 13 :=
by
  sorry

end star_4_3_l533_533682


namespace collinearity_equiv_l533_533343

-- Additional imports may be needed depending on the library content
-- Define a structure for an acute triangle with required points and properties
structure AcuteAngledTriangle :=
  (A B C D E P Q I O : Type)
  (triangle_is_acute : ∀ (a b c : ℝ), ∠A < 90 ∧ ∠B < 90 ∧ ∠C < 90)
  (AD_is_altitude : ∃ (D : Type), height A D B = perpendicular A D B) -- AD is an altitude
  (BE_is_altitude : ∃ (E : Type), height B E C = perpendicular B E C) -- BE is an altitude
  (AP_is_bisector : ∃ (P : Type), bisector A P B ∧ P ≠ I)  -- AP is a bisector
  (BQ_is_bisector : ∃ (Q : Type), bisector B Q C ∧ Q ≠ I)  -- BQ is a bisector
  (I_is_incenter : ∀ (I : Type), incenter I A B C)  -- I is incenter
  (O_is_circumcenter : ∀ (O : Type), circumcenter O A B C)  -- O is circumcenter
  (D_E_I_collinear : collinear D E I)  -- D, E, I collinear
  (P_Q_O_collinear : collinear P Q O)  -- P, Q, O collinear

-- The main theorem
theorem collinearity_equiv {T : AcuteAngledTriangle}
  (cos_angle_sum : ∀ (α β γ : ℝ), cos α + cos β = cos γ)
  : T.D_E_I_collinear ↔ T.P_Q_O_collinear :=
begin
  sorry
end

end collinearity_equiv_l533_533343


namespace sum_of_digits_10_pow_100_minus_57_l533_533580

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  let digits := (n.toString.toList.map (λ c => c.toString.toInt)).filterMap id
  digits.sum

theorem sum_of_digits_10_pow_100_minus_57 :
  sum_of_digits (10^100 - 57) = 889 := by
  sorry

end sum_of_digits_10_pow_100_minus_57_l533_533580


namespace area_of_base_of_cone_l533_533285

theorem area_of_base_of_cone (semicircle_area : ℝ) (h1 : semicircle_area = 2 * Real.pi) : 
  ∃ (base_area : ℝ), base_area = Real.pi :=
by
  sorry

end area_of_base_of_cone_l533_533285


namespace gum_left_after_sharing_l533_533201

-- Define the initial state of Adrianna's gum and the changes to it
def initial_gum : Nat := 10
def additional_gum : Nat := 3
def given_out_gum : Nat := 11

-- Define the final state of Adrianna's gum
def final_gum : Nat := initial_gum + additional_gum - given_out_gum

-- Prove that Adrianna ends up with 2 pieces of gum under the given conditions
theorem gum_left_after_sharing :
  final_gum = 2 :=
by 
  -- Since this is just the statement and not the proof, we end with sorry.
  sorry

end gum_left_after_sharing_l533_533201


namespace count_terminating_decimals_l533_533948

def has_terminating_decimal (n: ℕ) : Prop :=
  ∃ m k : ℕ, n * m = 2^k * 5^k

theorem count_terminating_decimals :
  (∑ n in Finset.range 181, if has_terminating_decimal (n / 180) then 1 else 0) = 60 :=
  sorry

end count_terminating_decimals_l533_533948


namespace gcd_positive_ints_l533_533372

theorem gcd_positive_ints (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (hdiv : (a^2 + b^2) ∣ (a * c + b * d)) : 
  Nat.gcd (c^2 + d^2) (a^2 + b^2) > 1 := 
sorry

end gcd_positive_ints_l533_533372


namespace minimize_AC_CB_l533_533645

/-
Given points A(-3, -4) and B(6, 3) in the xy-plane, point C(1, m) is taken such that AC + CB is minimized.
Prove that m = -8/9.
-/

theorem minimize_AC_CB (m : ℝ) :
  let A := (-3, -4)
  let B := (6, 3)
  let C := (1, m)
  let slope := (3 - (-4)) / (6 - (-3))
  let line_eq := slope * (1 - (-3)) - 4 -- equation of line AB
  AC + CB = AB -> m = -8/9 := by
  sorry

end minimize_AC_CB_l533_533645


namespace smallest_positive_period_of_f_is_pi_range_of_f_in_interval_l533_533665

def f (x : ℝ) := sin (2 * x) - 2 * real.sqrt 3 * (sin x) ^ 2 + real.sqrt 3 + 1

theorem smallest_positive_period_of_f_is_pi :
  ∃ (T : ℝ), (∀ x, f (x + T) = f x) ∧ (∀ T', 0 < T' ∧ T' < T → ¬∀ x, f (x + T') = f x) :=
sorry

theorem range_of_f_in_interval : 
  ∀ x, x ∈ (set.Icc (-real.pi / 6) (real.pi / 6)) → f x ∈ (set.Icc 1 3) :=
sorry

end smallest_positive_period_of_f_is_pi_range_of_f_in_interval_l533_533665


namespace minimum_value_of_f_l533_533816

noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 4) / (x - 1)

theorem minimum_value_of_f :
  ∀ x : ℝ, x > 1 → f x ≥ 5 ∧ (f x = 5 ↔ x = 3) :=
by {
  intro x,
  intro x_gt_1,
  sorry
}

end minimum_value_of_f_l533_533816


namespace cost_per_gumball_l533_533397

theorem cost_per_gumball (total_money : ℕ) (num_gumballs : ℕ) (cost_each : ℕ) 
  (h1 : total_money = 32) (h2 : num_gumballs = 4) : cost_each = 8 :=
by
  sorry -- Proof omitted

end cost_per_gumball_l533_533397


namespace lulu_cash_left_l533_533392

-- Define the initial amount
def initial_amount : ℕ := 65

-- Define the amount spent on ice cream
def spent_on_ice_cream : ℕ := 5

-- Define the amount spent on a t-shirt
def spent_on_tshirt (remaining_after_ice_cream : ℕ) : ℕ := remaining_after_ice_cream / 2

-- Define the amount deposited in the bank
def deposited_in_bank (remaining_after_tshirt : ℕ) : ℕ := remaining_after_tshirt / 5

-- Define the remaining cash after all transactions
def remaining_cash (initial : ℕ) (spent_ice_cream : ℕ) (spent_tshirt: ℕ) (deposited: ℕ) :ℕ :=
  initial - spent_ice_cream - spent_tshirt - deposited

-- Theorem statement to prove
theorem lulu_cash_left : remaining_cash initial_amount spent_on_ice_cream (spent_on_tshirt (initial_amount - spent_on_ice_cream)) 
(deposited_in_bank ((initial_amount - spent_on_ice_cream) - (spent_on_tshirt (initial_amount - spent_on_ice_cream)))) = 24 :=
by
  sorry

end lulu_cash_left_l533_533392


namespace lulu_cash_left_l533_533394

theorem lulu_cash_left :
  ∀ (initial money spentIceCream spentTshirt deposited finalCash: ℝ),
    initial = 65 →
    spentIceCream = 5 →
    spentTshirt = 0.5 * (initial - spentIceCream) →
    deposited = (initial - spentIceCream - spentTshirt) / 5 →
    finalCash = initial - spentIceCream - spentTshirt - deposited →
    finalCash = 24 :=
by
  intros
  sorry

end lulu_cash_left_l533_533394


namespace sine_product_identity_l533_533586

open Real

theorem sine_product_identity :
  sin 12 * sin 36 * sin 54 * sin 72 = 1 / 16 := by
  have h1 : sin 72 = cos 18 := by sorry
  have h2 : sin 54 = cos 36 := by sorry
  have h3 : ∀ θ, sin θ * cos θ = 1 / 2 * sin (2 * θ) := by sorry
  have h4 : ∀ θ, cos (2 * θ) = 2 * cos θ ^ 2 - 1 := by sorry
  have h5 : cos 36 = 1 - 2 * (sin 18) ^ 2 := by sorry
  have h6 : ∀ θ, sin (180 - θ) = sin θ := by sorry
  sorry

end sine_product_identity_l533_533586


namespace incorrect_median_l533_533404

def data_points : List ℝ := [6.5, 6.3, 7.8, 9.2, 5.7, 7, 7.9, 8.1, 7.2, 5.8, 8.3]

def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (≤)
  if h : l.length % 2 = 1 then
    sorted.get ⟨l.length / 2, by simp [Nat.div_lt_of_lt]; exact h⟩
  else
    let m := l.length / 2
    (sorted.get ⟨m - 1, by simp [Nat.sub_lt_of_pos, Nat.lt_of_le_and_ne, Nat.div_pos]; exact ⟨Nat.div_nonneg (Nat.le_of_lt h), Nat.neq_of_gt h⟩⟩ + sorted.get ⟨m, Nat.div_lt_of_lt h⟩) / 2

-- We need to prove that the median of the data points is not 6.8

theorem incorrect_median : median data_points ≠ 6.8 := by
  sorry

end incorrect_median_l533_533404


namespace correct_statements_l533_533314

variable {R : Type*} [LinearOrderedField R]

noncomputable def f : R → R := sorry

def is_odd (f : R → R) : Prop := ∀ x, f (-x) = -f x

def satisfies_eq (f : R → R) : Prop := ∀ x, f (x - 2) = -f x

theorem correct_statements
  (h_odd : is_odd f)
  (h_satisfies_eq : satisfies_eq f) :
  (f 2 = 0) ∧
  (∀ x, f (x + 4) = f x) ∧
  (∀ x, f (x + 2) = f (-x)) ∧
  (¬∀ x, f (-x) = f x) :=
by {
  sorry
}

end correct_statements_l533_533314


namespace Beth_initial_crayons_l533_533571

def initial_crayons (x : ℕ) (given_away : ℕ) (remaining : ℕ) : Prop :=
  x - given_away = remaining

theorem Beth_initial_crayons : ∃ x : ℕ, initial_crayons x 54 52 ∧ x = 106 :=
begin
  use 106,
  unfold initial_crayons,
  sorry
end

end Beth_initial_crayons_l533_533571


namespace minimum_pencil_strokes_for_figure_l533_533232

def figure (A B : Type) : Prop :=
  -- placeholder for the actual structure/description of the figure
  sorry

theorem minimum_pencil_strokes_for_figure :
  ∃ (s : set (list (list point))), figure ∧ (∀ l ∈ s, isValidStroke l) → (min_strokes s = 14) :=
  sorry

end minimum_pencil_strokes_for_figure_l533_533232


namespace water_evaporation_problem_l533_533181

theorem water_evaporation_problem 
  (W : ℝ) 
  (evaporation_rate : ℝ := 0.01) 
  (evaporation_days : ℝ := 20) 
  (total_evaporation : ℝ := evaporation_rate * evaporation_days) 
  (evaporation_percentage : ℝ := 0.02) 
  (evaporation_amount : ℝ := evaporation_percentage * W) :
  evaporation_amount = total_evaporation → W = 10 :=
by
  sorry

end water_evaporation_problem_l533_533181


namespace distance_between_points_l533_533121

theorem distance_between_points (a b c d : ℝ) 
  (h1 : b = 2 * a + 3) 
  (h2 : c = 5) : 
  real.sqrt ((5 - a)^2 + (d - (2 * a + 3))^2) = real.dist (5, d) (a, 2 * a + 3) :=
sorry

end distance_between_points_l533_533121


namespace arithmetic_common_difference_l533_533028

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533028


namespace sum_of_next_17_consecutive_integers_l533_533444

theorem sum_of_next_17_consecutive_integers (x : ℤ) (h₁ : (List.range 17).sum + 17 * x = 306) :
  (List.range 17).sum + 17 * (x + 17)  = 595 := 
sorry

end sum_of_next_17_consecutive_integers_l533_533444


namespace hair_color_cost_l533_533758

theorem hair_color_cost :
  let palette_cost := 15
  let lipstick_cost := 2.5
  let total_cost := 67
  let palettes := 3
  let lipsticks := 4
  let hair_boxes := 3
  let cost_per_box := 4 in
  palettes * palette_cost + lipsticks * lipstick_cost + hair_boxes * cost_per_box = total_cost :=
by
  sorry

end hair_color_cost_l533_533758


namespace range_m_two_distinct_real_roots_l533_533290

def f (x : ℝ) : ℝ :=
  if x < 2 then 2^(2 - x)
  else Real.logBase 3 (x + 1)

theorem range_m_two_distinct_real_roots :
  {m : ℝ | ∃ x1 x2 : ℝ, f x1 = m ∧ f x2 = m ∧ x1 ≠ x2} = set.Ioi 1 :=
sorry

end range_m_two_distinct_real_roots_l533_533290


namespace total_supervisors_l533_533809

def buses : ℕ := 7
def supervisors_per_bus : ℕ := 3

theorem total_supervisors : buses * supervisors_per_bus = 21 := 
by
  have h : buses * supervisors_per_bus = 21 := by sorry
  exact h

end total_supervisors_l533_533809


namespace min_marked_necessary_l533_533147

-- A 12x12 board represented as a set of coordinates
def board_12x12 := fin 12 × fin 12

-- A four-cell figure's possible placements, given by its upper-left corner position.
def figure_placements : list (set (fin 12 × fin 12)) :=
  [(0,0), (0,1), (1,0), (1,1), (1,2), (2,0), (2,1)].map (λ c, {(0,0), (0,1), (1,0), (1,1)}.image (prod.add c))

-- The marking condition function
def well_marked (marked : set (fin 12 × fin 12)) :=
  ∀ p ∈ board_12x12, ¬(figure_placements.map (λ s, disjoint marked s)).all id

-- The core hypothesis and the final theorem
theorem min_marked_necessary : ∃ k : ℕ, k = 48 ∧ ∃ marked : set (fin 12 × fin 12), well_marked marked ∧ marked.card = k :=
by sorry

end min_marked_necessary_l533_533147


namespace hexagon_side_equalities_l533_533636

variable {Point : Type} [AddCommGroup Point] [Module ℝ Point]

-- Definitions for the points of the hexagon
variables (A B C D E F : Point)

-- Conditions given in the problem
variables (h1 : ∥A - B∥ = ∥D - E∥) -- AB = DE
variables (h2 : ∥B - C∥ = ∥E - F∥) -- Opposite sides parallel

theorem hexagon_side_equalities : ∥B - C∥ = ∥E - F∥ ∧ ∥C - D∥ = ∥F - A∥ :=
by
  -- Proof goes here
  sorry

end hexagon_side_equalities_l533_533636


namespace smallest_five_digit_int_congruent_to_2_mod_37_l533_533484

theorem smallest_five_digit_int_congruent_to_2_mod_37 :
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ n % 37 = 2 ∧ ∀ m, (10000 ≤ m ∧ m < 100000 ∧ m % 37 = 2) → 10027 ≤ m :=
by
  use 10027
  split
  repeat { sorry }

end smallest_five_digit_int_congruent_to_2_mod_37_l533_533484


namespace matrix_power_sub_l533_533735

section 
variable (A : Matrix (Fin 2) (Fin 2) ℝ)
variable (hA : A = ![![2, 3], ![0, 1]])

theorem matrix_power_sub (A : Matrix (Fin 2) (Fin 2) ℝ)
  (h: A = ![![2, 3], ![0, 1]]) :
  A ^ 20 - 2 * A ^ 19 = ![![0, 3], ![0, -1]] :=
by
  sorry
end

end matrix_power_sub_l533_533735


namespace convert_to_dms_l533_533518

-- Define the conversion factors
def degrees_to_minutes (d : ℝ) : ℝ := d * 60
def minutes_to_seconds (m : ℝ) : ℝ := m * 60

-- The main proof statement
theorem convert_to_dms (d : ℝ) :
  d = 24.29 →
  (24, 17, 24) = (24, degrees_to_minutes (0.29), minutes_to_seconds 0.4) :=
by
  sorry

end convert_to_dms_l533_533518


namespace angie_carlos_seated_two_seats_apart_l533_533557

   noncomputable def probability_angie_carlos_two_seats_apart : ℚ :=
     let total_arrangements := 4! -- Total possible arrangements for remaining people
     let favorable_arrangements := 2 * 3! -- Favorable ways where Carlos is exactly two seats apart from Angie
     favorable_arrangements / total_arrangements

   theorem angie_carlos_seated_two_seats_apart :
     probability_angie_carlos_two_seats_apart = 1 / 2 :=
   sorry
   
end angie_carlos_seated_two_seats_apart_l533_533557


namespace count_4_edge_trips_from_C_to_D_l533_533437

-- Define vertices and edges of a cube
inductive CubeVertex : Type
| C : CubeVertex
| other : Nat → CubeVertex -- Assuming there are 7 other vertices, numbered

-- Define conditions: C and D are diagonal corners on adjacent faces
constant C : CubeVertex := CubeVertex.C
constant D : CubeVertex := CubeVertex.other 1 -- Assign D as some other vertex

-- Define adjacency relation on the cube vertices
constant is_adjacent : CubeVertex → CubeVertex → Prop

noncomputable def shortest_trip_count (v₁ v₂ : CubeVertex) (edges : Nat) : Nat :=
  -- This function is a placeholder for the actual trip counting logic.
  if v₁ = C ∧ v₂ = D ∧ edges = 4 then 12 else sorry

-- The final Lean statement to prove
theorem count_4_edge_trips_from_C_to_D : shortest_trip_count C D 4 = 12 :=
by
  -- Proof omitted, represent with sorry
  sorry

end count_4_edge_trips_from_C_to_D_l533_533437


namespace martin_less_than_43_l533_533056

variable (C K M : ℕ)

-- Conditions
def campbell_correct := C = 35
def kelsey_correct := K = C + 8
def martin_fewer := M < K

-- Conclusion we want to prove
theorem martin_less_than_43 (h1 : campbell_correct C) (h2 : kelsey_correct C K) (h3 : martin_fewer K M) : M < 43 := 
by {
  sorry
}

end martin_less_than_43_l533_533056


namespace aubree_total_animals_l533_533567

noncomputable def total_animals_seen : Nat :=
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10
  morning_total + new_beavers + new_chipmunks

theorem aubree_total_animals :
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10
  morning_total + new_beavers + new_chipmunks = 130 := by
  -- Define all constants and conditions
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10

  -- State the equation
  show morning_total + new_beavers + new_chipmunks = 130 from sorry

end aubree_total_animals_l533_533567


namespace nike_function_max_min_nike_function_range_a_l533_533981

noncomputable def f (a x : ℝ) : ℝ := (x^2 + x + a) / x

theorem nike_function_max_min (x : ℝ) (a : ℝ) (h₁ : a = 4) (h₂ : 1 / 2 ≤ x) (h₃ : x ≤ 3) : 
  (f a 2 = 5) ∧ (f a (1 / 2) ≤ f a 3) ∧ (f a (1 / 2) = 19 / 2) :=
by
  have hx₁ : 0 < x := sorry
  have fa_def : f a x = x + a / x + 1 := sorry
  have fa_mono_decreasing : ∀ x, 0 < x ∧ x ≤ 2 → f a x ≤ f a 2 := sorry
  have fa_mono_increasing : ∀ x, 2 ≤ x → f a 2 ≤ f a x := sorry
  sorry

theorem nike_function_range_a (a : ℝ) (h₁ : ∀ x, 1 ≤ x ∧ x ≤ 2 → monotone (f a)) :
  (0 < a ∧ a ≤ 1) ∨ (4 ≤ a) :=
by
  sorry

end nike_function_max_min_nike_function_range_a_l533_533981


namespace tangent_line_eqn_l533_533968

noncomputable def f : ℝ → ℝ :=
  fun x => x^3 + x

theorem tangent_line_eqn : ∀ (x y : ℝ),
    f 1 = 2 → (∂ f ' 1) = 4 → y = 4x - 2 → 4 * x - y - 2 = 0 :=
by
  intro x y hf1 hf'_1 h_line_eq
  sorry

end tangent_line_eqn_l533_533968


namespace arithmetic_sequence_sum_l533_533032

variable {a : ℕ → ℝ}  -- Define the sequence a_n
variable {d : ℝ}      -- Define the common difference d

-- Define arithmetic sequence property 
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define S_n as the sum of first n terms of the sequence
def sum_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

-- The main theorem to be proved
theorem arithmetic_sequence_sum (h : is_arithmetic_sequence a d) (S11 : sum_n a 11 = 44) :
  a 4 + a 6 + a 8 = 12 :=
sorry

end arithmetic_sequence_sum_l533_533032


namespace negation_of_universal_statement_l533_533092

variable (R : Type _) [Real R]

theorem negation_of_universal_statement :
  ¬ (∀ x : R, x^2 + x + 1 < 0) ↔ ∃ x : R, x^2 + x + 1 ≥ 0 :=
by sorry

end negation_of_universal_statement_l533_533092


namespace work_efficiency_ratio_l533_533184
noncomputable section

variable (A_eff B_eff : ℚ)

-- Conditions
def efficient_together (A_eff B_eff : ℚ) : Prop := A_eff + B_eff = 1 / 12
def efficient_alone (A_eff : ℚ) : Prop := A_eff = 1 / 16

-- Theorem to prove
theorem work_efficiency_ratio (A_eff B_eff : ℚ) (h1 : efficient_together A_eff B_eff) (h2 : efficient_alone A_eff) : A_eff / B_eff = 3 := by
  sorry

end work_efficiency_ratio_l533_533184


namespace correct_propositions_l533_533454

-- Define the conditions and corresponding propositions
def proposition_1 := ∀ (x : ℝ), x^2 + x + 1 ≠ 0

def setA := {x : ℝ | x > 0}
def setB := {x : ℝ | x ≤ -1}
def complement_B := {x : ℝ | x > -1}

def proposition_2 := setA = setA ∩ complement_B

def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)
def proposition_3 := ∀ (ω > 0) (k : ℤ) (φ : ℝ), φ = k * Real.pi + Real.pi / 2 → 
    (∀ (x : ℝ), f x ω φ = f (-x) ω φ)

def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def proposition_4 (a b : ℝ × ℝ) :=
    vector_magnitude a = vector_magnitude b ∧
    vector_magnitude (a - b) = vector_magnitude b →
    angle_between b (a - b) = 60

-- The four propositions
def prop1 : Prop := proposition_1
def prop2 : Prop := proposition_2
def prop3 : Prop := proposition_3
def prop4 : Prop := proposition_4

-- Main theorem: prove that Proposition 2 and Proposition 3 are true
theorem correct_propositions : prop2 ∧ prop3 :=
by {
    sorry
}

end correct_propositions_l533_533454


namespace dice_even_probability_l533_533137

theorem dice_even_probability :
  (Pr (λ (outcome : Fin 6 × Fin 6 × Fin 6 × Fin 6),
    (∃ k : Fin 6, k % 2 = 0 ∧ outcome.1 = k ∧ outcome.2 = k ∧ outcome.3 = k ∧ outcome.4 = k) = 
    1/432) :=
sorry

end dice_even_probability_l533_533137


namespace least_positive_value_prime_expression_l533_533385

open Nat

def is_least_positive_value (e : ℕ → ℕ → ℕ → ℕ → ℤ) (x y z w : ℕ) (p : ℤ) : Prop :=
  e x y z w = p ∧ ∀ m, (m > 0) → (∃ a b c d, e a b c d = m) → (m ≥ p)

theorem least_positive_value_prime_expression :
  ∃ x y z w : ℕ, prime x ∧ prime y ∧ prime z ∧ prime w ∧ is_least_positive_value (λ x y z w => 24 * x + 16 * y - 7 * z + 5 * w) x y z w 13 :=
by
  sorry

end least_positive_value_prime_expression_l533_533385


namespace intervals_of_increase_area_of_triangle_abc_l533_533291

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

-- Definitions for the conditions in problem (Ⅱ)
def side_a : ℝ := 1
def sum_bc (b c : ℝ) : Prop := b + c = 2
def angle_A : ℝ := Real.pi / 3  -- from the conditions derived in the solution
def sin_property (A : ℝ) : Prop := Real.sin (2 * A + Real.pi / 6) = 1 / 2

-- Proof statement for problem (Ⅰ)
theorem intervals_of_increase :
  (∀ (k : ℤ), ∃ (x : ℝ), k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 → 
    (∃ ε > 0, ∀ d : ℝ, 0 < d ∧ d < ε → f (x + d) > f x)) := by sorry

-- Proof statement for problem (Ⅱ)
theorem area_of_triangle_abc (b c : ℝ) (h_sum : sum_bc b c) :
  sin_property angle_A → side_a = 1 → 
  (b^2 + c^2 - 2 * b * c * Real.cos angle_A = 1) →
  (1 / 2 * b * c * Real.sin angle_A = √3 / 4) := by sorry

end intervals_of_increase_area_of_triangle_abc_l533_533291


namespace andrew_stickers_now_l533_533047

-- Defining the conditions
def total_stickers : Nat := 1500
def ratio_susan : Nat := 1
def ratio_andrew : Nat := 1
def ratio_sam : Nat := 3
def total_ratio : Nat := ratio_susan + ratio_andrew + ratio_sam
def part : Nat := total_stickers / total_ratio
def susan_share : Nat := ratio_susan * part
def andrew_share_initial : Nat := ratio_andrew * part
def sam_share : Nat := ratio_sam * part
def sam_to_andrew : Nat := (2 * sam_share) / 3

-- Andrew's final stickers count
def andrew_share_final : Nat :=
  andrew_share_initial + sam_to_andrew

-- The theorem to prove
theorem andrew_stickers_now : andrew_share_final = 900 :=
by
  -- Proof would go here
  sorry

end andrew_stickers_now_l533_533047


namespace slope_of_tangent_line_at_A_l533_533443

noncomputable def f (x : ℝ) := x^2 + 3 * x

def derivative_at (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (sorry : ℝ)  -- Placeholder for the definition of the derivative

theorem slope_of_tangent_line_at_A : 
  derivative_at f 1 = 5 := 
sorry

end slope_of_tangent_line_at_A_l533_533443


namespace max_product_931_l533_533469

open Nat

theorem max_product_931 : ∀ a b c d e : ℕ,
  -- Digits must be between 1 and 9 and all distinct.
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
  {1, 3, 5, 8, 9} = {a, b, c, d, e} ∧ 
  -- Three-digit and two-digit numbers.
  100 * a + 10 * b + c ≤ 999 ∧ 100 * a + 10 * b + c ≥ 100 ∧
  10 * d + e ≤ 99 ∧ 10 * d + e ≥ 10
  -- Then the resulting product is maximized when abc = 931.
  → 100 * a + 10 * b + c = 931 := 
by
  intros a b c d e h_distinct h_digits h_3digit_range h_2digit_range
  sorry

end max_product_931_l533_533469


namespace probability_neither_square_nor_fourth_power_l533_533820

noncomputable def count_perfect_squares (n : ℕ) : ℕ :=
(nat.floor (real.sqrt n))

noncomputable def count_perfect_fourth_powers (n : ℕ) : ℕ :=
(nat.floor (real.sqrt (real.sqrt n)))

theorem probability_neither_square_nor_fourth_power :
  let total_numbers := 200 in
  let perfect_squares := count_perfect_squares total_numbers in
  let perfect_fourth_powers := count_perfect_fourth_powers total_numbers in
  let overlap := count_perfect_fourth_powers total_numbers in
  let desirable_numbers := total_numbers - (perfect_squares + perfect_fourth_powers - overlap) in
  (desirable_numbers : ℝ) / total_numbers = 93 / 100 :=
sorry

end probability_neither_square_nor_fourth_power_l533_533820


namespace intersection_point_varies_l533_533514

theorem intersection_point_varies
  (O : Point)
  (A B : Point)
  (circle : Circle O)
  (AB_diameter : diameter circle A B)
  (C : Point)
  (C_on_circle : on_circle C circle)
  (D : Point)
  (angle_ACD : angle A C D = 30) :
  ∃ P, (P ≠ C ∧ P ≠ D ∧ on_circle P circle ∧ perpendicular_bisector_intersection C D P) -> P varies :=
by
  sorry

end intersection_point_varies_l533_533514


namespace number_of_solutions_mod_n_l533_533264

open Nat

theorem number_of_solutions_mod_n (n : ℕ) (h : n ≥ 2) :
  ∃ xs : Finset ℕ, (∀ x ∈ xs, x^2 % n = x % n) ∧ xs.card = nat.divisors n :=
sorry

end number_of_solutions_mod_n_l533_533264


namespace fill_digits_subtraction_correct_l533_533964

theorem fill_digits_subtraction_correct :
  ∀ (A B : ℕ), A236 - (B*100 + 97) = 5439 → A = 6 ∧ B = 7 :=
by
  sorry

end fill_digits_subtraction_correct_l533_533964


namespace horse_travel_distance_on_seventh_day_l533_533423

noncomputable def geometric_series_sum (a r n : ℕ) : ℝ :=
a * (1 - r^n) / (1 - r)

noncomputable def geometric_nth_term (a r n : ℕ) : ℝ :=
a * r^(n - 1)

theorem horse_travel_distance_on_seventh_day :
  let a₁ := real.of_rat (1400 * 128 / 127)
  let d7 := geometric_nth_term a₁ (1/2) 7
  (geometric_series_sum a₁ (1/2) 7 = 700) → 
  d7 = real.of_rat (700 / 127) :=
by
  sorry

end horse_travel_distance_on_seventh_day_l533_533423


namespace remainder_of_division_l533_533803

theorem remainder_of_division (L S R : ℕ) (hL : L = 1620) (h_diff : L - S = 1365) (h_div : L = 6 * S + R) : R = 90 :=
by {
  -- Since we are not providing the proof, we use sorry
  sorry
}

end remainder_of_division_l533_533803


namespace smallest_number_l533_533205

open Nat

theorem smallest_number :
  let num1 : ℕ := 8 * (9 : ℕ)^1 + 5 * (9 : ℕ)^0,
      num2 : ℕ := 1 * (4 : ℕ)^3,
      num3 : ℕ := 1 * (2 : ℕ)^5 + 1 * (2 : ℕ)^4 + 1 * (2 : ℕ)^3 + 1 * (2 : ℕ)^2 + 1 * (2 : ℕ)^1 + 1 * (2 : ℕ)^0
  in num3 < num2 ∧ num3 < num1 :=
by
  sorry

end smallest_number_l533_533205


namespace Alfred_spent_on_repairs_l533_533204

noncomputable def AlfredRepairCost (purchase_price selling_price gain_percent : ℚ) : ℚ :=
  let R := (selling_price - purchase_price * (1 + gain_percent)) / (1 + gain_percent)
  R

theorem Alfred_spent_on_repairs :
  AlfredRepairCost 4700 5800 0.017543859649122806 = 1000 := by
  sorry

end Alfred_spent_on_repairs_l533_533204


namespace count_valid_house_numbers_l533_533951

def is_two_digit_prime (n : ℕ) : Prop :=
  n > 9 ∧ n < 100 ∧ Nat.Prime n

def valid_house_number (A B C D : ℕ) : Prop :=
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
  is_two_digit_prime (10 * A + B) ∧
  is_two_digit_prime (10 * C + D) ∧
  (10 * A + B ≠ 10 * C + D)

theorem count_valid_house_numbers : 
  (finset.univ : finset (ℕ × ℕ × ℕ × ℕ)).filter (λ abcd, valid_house_number abcd.1 abcd.2 abcd.3 abcd.4).card = 110 := 
sorry

end count_valid_house_numbers_l533_533951


namespace completing_square_16x2_32x_512_eq_33_l533_533728

theorem completing_square_16x2_32x_512_eq_33:
  (∃ p q : ℝ, (16 * x ^ 2 + 32 * x - 512 = 0) → (x + p) ^ 2 = q ∧ q = 33) :=
by
  sorry

end completing_square_16x2_32x_512_eq_33_l533_533728


namespace ruth_weekly_class_hours_l533_533409

def hours_in_a_day : ℕ := 8
def days_in_a_week : ℕ := 5
def weekly_school_hours := hours_in_a_day * days_in_a_week

def math_class_percentage : ℚ := 0.25
def language_class_percentage : ℚ := 0.30
def science_class_percentage : ℚ := 0.20
def history_class_percentage : ℚ := 0.10

def math_hours := math_class_percentage * weekly_school_hours
def language_hours := language_class_percentage * weekly_school_hours
def science_hours := science_class_percentage * weekly_school_hours
def history_hours := history_class_percentage * weekly_school_hours

def total_class_hours := math_hours + language_hours + science_hours + history_hours

theorem ruth_weekly_class_hours : total_class_hours = 34 := by
  -- Calculation proof logic will go here
  sorry

end ruth_weekly_class_hours_l533_533409


namespace find_a_l533_533607

noncomputable def fractionalPart (a : ℝ) : ℝ :=
  a - ⌊a⌋

theorem find_a (a : ℝ) (h1 : 3 < a ∧ a < 4) (h2 : (a * (a - 3 * fractionalPart a)).integer? ≠ none) :
  a = 3 + (Real.sqrt 17 - 3) / 4 ∨
  a = 7 / 2 ∨
  a = 3 + (Real.sqrt 33 - 3) / 4 ∨
  a = 3 + (Real.sqrt 41 - 3) / 4 := 
sorry

end find_a_l533_533607


namespace range_of_k_l533_533647

variable (a k : ℝ)

def f (x : ℝ) : ℝ := a * Real.log x
def g (x : ℝ) : ℝ := x - 1

theorem range_of_k :
  (∀ x, x ≥ 0 → g x ≥ k * x) →
  f' 0 = g' 0 →
  a = 1 →
  k ≤ 1 := 
sorry

end range_of_k_l533_533647


namespace math_problem_l533_533309

theorem math_problem
  (a b c d m : ℝ)
  (h1 : a = -b)
  (h2 : c = (1 / d) ∨ d = (1 / c))
  (h3 : |m| = 4) :
  (a + b = 0) ∧ (c * d = 1) ∧ (m = 4 ∨ m = -4) ∧
  ((a + b) / 3 + m^2 - 5 * (c * d) = 11) := by
  sorry

end math_problem_l533_533309


namespace fish_weight_l533_533524

variables (H T X : ℝ)
-- Given conditions
def tail_weight : Prop := X = 1
def head_weight : Prop := H = X + 0.5 * T
def torso_weight : Prop := T = H + X

theorem fish_weight (H T X : ℝ) 
  (h_tail : tail_weight X)
  (h_head : head_weight H T X)
  (h_torso : torso_weight H T X) : 
  H + T + X = 8 :=
sorry

end fish_weight_l533_533524


namespace M_A_C_l533_533378

def A := {B | B ⊆ {1, 2, 3} ∧ (1 ∈ B ∨ 3 ∈ B)}
def C := {x | x^2 - 4 * |x| + 3 = 0}

def M (a b : ℝ) : ℝ :=
if a >= b then a else b

theorem M_A_C : M (|A|) (|C|) = 4 :=
sorry

end M_A_C_l533_533378


namespace value_of_fraction_l533_533685

theorem value_of_fraction (a b c : ℝ) (h₁ : a = 5) (h₂ : b = -3) (h₃ : c = 2) : 3 / (a + b + c) = 3 / 4 :=
by {
  rw [h₁, h₂, h₃],
  norm_num,
  }
  

end value_of_fraction_l533_533685


namespace triangle_probability_in_regular_15gon_l533_533835

noncomputable def num_segments := Nat.choose 15 2

noncomputable def segment_length (k : ℕ) : ℝ := 2 * Real.sin (k * Real.pi / 15)

def valid_triangle_probability : ℝ :=
  let num_valid_triangles := (some calculation here) -- replace with the computation of valid triangles (to be detailed)
  let total_triangles := Nat.choose num_segments 3
  (num_valid_triangles : ℝ) / total_triangles

theorem triangle_probability_in_regular_15gon :
  valid_triangle_probability = 713 / 780 := 
sorry

end triangle_probability_in_regular_15gon_l533_533835


namespace hydrogen_atoms_in_compound_l533_533885

theorem hydrogen_atoms_in_compound : 
  ∀ (Al_weight O_weight H_weight : ℕ) (total_weight : ℕ) (num_Al num_O num_H : ℕ),
  Al_weight = 27 →
  O_weight = 16 →
  H_weight = 1 →
  total_weight = 78 →
  num_Al = 1 →
  num_O = 3 →
  (num_Al * Al_weight + num_O * O_weight + num_H * H_weight = total_weight) →
  num_H = 3 := 
by
  intros
  sorry

end hydrogen_atoms_in_compound_l533_533885


namespace ancient_chinese_problem_l533_533349

theorem ancient_chinese_problem (x y : ℤ) 
  (h1 : y = 8 * x - 3) 
  (h2 : y = 7 * x + 4) : 
  (y = 8 * x - 3) ∧ (y = 7 * x + 4) :=
by
  exact ⟨h1, h2⟩

end ancient_chinese_problem_l533_533349


namespace set_intersection_complement_l533_533331

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 5, 8}
def B : Set ℕ := {1, 3, 5, 7}

theorem set_intersection_complement :
  ((U \ A) ∩ B) = {1, 3, 7} :=
by
  sorry

end set_intersection_complement_l533_533331


namespace factorable_polynomial_l533_533229

theorem factorable_polynomial (n : ℤ) :
  ∃ (a b c d e f : ℤ), 
    (a = 1) ∧ (d = 1) ∧ 
    (b + e = 2) ∧ 
    (f = b * e) ∧ 
    (c + f + b * e = 2) ∧ 
    (c * f + b * e = -n^2) ↔ 
    (n = 0 ∨ n = 2 ∨ n = -2) :=
by
  sorry

end factorable_polynomial_l533_533229


namespace sufficient_but_not_necessary_condition_l533_533672

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1) → (∀ x y : ℝ, l₁ x y = ax + y - 1 ∧ l₂ x y = 2x + (a + 1)y + 3 → (l₁ x y = 0 ∧ l₂ x y = 0) → (a = -2 ∨ a = 1 )) :=
by
  intro ha
  intro x y
  intro h₁ h₂
  sorry

end sufficient_but_not_necessary_condition_l533_533672


namespace num_4digit_numbers_with_three_identical_digits_l533_533095

theorem num_4digit_numbers_with_three_identical_digits :
  (∃ (n : ℕ), n = 18 ∧
    ∀ m : ℕ, 
      (1000 ≤ m ∧ 
      m < 10000 ∧ 
      ∃ d : ℕ, d ≠ 10 ∧
      count (m.digits 10) d = 3 ∧
      (m.digits 10).head = 1) ↔
      m ∈ (filter (λ x, (∃ k : ℕ, (x.digits 10).head = 1 ∧ count (x.digits 10) k = 3 ∧ k ≠ 1) ∨ 
        (∃ l : ℕ, (count (x.digits 10) 1 = 3 ∧ (x.digits 10).head = 1) ∧ l ≠ 1))
        (range (1000, 10000)))) :=
sorry

end num_4digit_numbers_with_three_identical_digits_l533_533095


namespace find_number_l533_533086

-- Define the factorial function
def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the problem conditions and target statement
theorem find_number : ∃ (n : ℕ), (factorial n) / (factorial (n - 3)) = 24 ∧ n = 4 := by
  sorry

end find_number_l533_533086


namespace main_theorem_l533_533379

variable {n : ℕ}
variable {a b : ℕ → ℝ}
variable {δ : ℝ}

noncomputable def S : ℕ → ℝ := sorry

axiom a_pos (i : ℕ) : 1 ≤ i ∧ i ≤ n + 1 → a i > 0
axiom b_pos (i : ℕ) : 1 ≤ i ∧ i ≤ n + 1 → b i > 0
axiom b_diff (i : ℕ) : 1 ≤ i ∧ i ≤ n → b (i + 1) - b i ≥ δ
axiom delta_pos : δ > 0
axiom sum_a : ∑ i in Finset.range n, a (i + 1) = 1
axiom S_ge (i : ℕ) : 1 ≤ i ∧ i ≤ n → S i ≥ i * Real.sqrt i (Finset.prod (Finset.range i) (λ j, a (j + 1) * b (j + 1)))

theorem main_theorem : (∑ i in Finset.range n, i * Real.sqrt i (Finset.prod (Finset.range i) (λ j, a (j + 1) * b (j + 1))) / (b (i + 1) * b i)) < 1 / δ := sorry

end main_theorem_l533_533379


namespace blue_pieces_of_candy_l533_533113

theorem blue_pieces_of_candy (total_pieces : ℕ) (red_pieces : ℕ) (h1 : total_pieces = 3409) (h2 : red_pieces = 145) :
  total_pieces - red_pieces = 3264 :=
by
  rw [h1, h2]
  sorry

end blue_pieces_of_candy_l533_533113


namespace groupD_can_form_triangle_l533_533438

def groupA := (5, 7, 12)
def groupB := (7, 7, 15)
def groupC := (6, 9, 16)
def groupD := (6, 8, 12)

def canFormTriangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem groupD_can_form_triangle : canFormTriangle 6 8 12 :=
by
  -- Proof of the above theorem will follow the example from the solution.
  sorry

end groupD_can_form_triangle_l533_533438


namespace number_of_possible_values_of_n_l533_533828

theorem number_of_possible_values_of_n 
  (S_n : ℕ)
  (d : ℕ)
  (Sn_eq : S_n = 225)
  (d_eq : d = 2)
  (n_gt_one : ∀ n, n > 1 → ∃ a: ℤ, a = (225 - n^2 + n) / n)
  : ∃! n, (n > 1 ∧ ∃ a: ℤ, a = (225 - n^2 + n) / n ∧ n ∈ {3, 5, 9, 15, 25, 45, 75, 225}) 
    := sorry

end number_of_possible_values_of_n_l533_533828


namespace jack_customs_wait_time_l533_533495

-- Define the conditions as constants
constant total_wait_time : ℕ := 356
constant quarantine_days : ℕ := 14
constant hours_per_day : ℕ := 24

-- Define the result of converting days to hours
noncomputable def quarantine_hours : ℕ := quarantine_days * hours_per_day

-- Define the proof problem statement
theorem jack_customs_wait_time : total_wait_time - quarantine_hours = 20 := by
  sorry

end jack_customs_wait_time_l533_533495


namespace compute_g3_pow4_l533_533791

noncomputable def f (x : ℝ) : ℝ := sorry  -- Placeholder for function f
noncomputable def g (x : ℝ) : ℝ := sorry  -- Placeholder for function g

theorem compute_g3_pow4 :
  (∀ x, x ≥ 1 → f(g(x)) = x^3) →
  (∀ x, x ≥ 1 → g(f(x)) = x^4) →
  g(81) = 81 →
  [g(3)]^4 = 81 :=
by
  intros h1 h2 h3
  sorry

end compute_g3_pow4_l533_533791


namespace hotel_rooms_total_l533_533756

theorem hotel_rooms_total
  (floors_wing1 : ℕ)
  (halls_floor_wing1 : ℕ)
  (rooms_hall_wing1 : ℕ)
  (floors_wing2 : ℕ)
  (halls_floor_wing2 : ℕ)
  (rooms_hall_wing2 : ℕ)
  (h_wing1 : floors_wing1 = 9)
  (h_halls_wing1 : halls_floor_wing1 = 6)
  (h_rooms_hall_wing1 : rooms_hall_wing1 = 32)
  (h_wing2 : floors_wing2 = 7)
  (h_halls_wing2 : halls_floor_wing2 = 9)
  (h_rooms_hall_wing2 : rooms_hall_wing2 = 40) :
  floors_wing1 * halls_floor_wing1 * rooms_hall_wing1 + floors_wing2 * halls_floor_wing2 * rooms_hall_wing2 = 4248 :=
by {
  rw [h_wing1, h_halls_wing1, h_rooms_hall_wing1, h_wing2, h_halls_wing2, h_rooms_hall_wing2],
  norm_num,
  sorry
}

end hotel_rooms_total_l533_533756


namespace seed_space_per_row_l533_533572

theorem seed_space_per_row (feet_per_row : ℕ) (seeds_per_row : ℕ) (inches_per_foot : ℕ)
  (H1 : feet_per_row = 120) (H2 : seeds_per_row = 80) (H3 : inches_per_foot = 12) :
  (feet_per_row * inches_per_foot) / seeds_per_row = 18 :=
by {
  rw [H1, H2, H3],
  norm_num,
}

end seed_space_per_row_l533_533572


namespace diagonals_of_polygon_l533_533077

theorem diagonals_of_polygon (f : ℕ → ℕ) (k : ℕ) (h_k : k ≥ 3) : f (k + 1) = f k + (k - 1) :=
sorry

end diagonals_of_polygon_l533_533077


namespace lulu_final_cash_l533_533389

-- Definitions of the problem conditions
def initial_amount : ℕ := 65
def spent_on_ice_cream : ℕ := 5
def spent_on_tshirt (remaining : ℕ) : ℕ := remaining / 2
def deposit_in_bank (remaining : ℕ) : ℕ := remaining / 5

-- The proof problem statement
theorem lulu_final_cash :
  ∃ final_cash : ℕ,
    final_cash = initial_amount - spent_on_ice_cream - spent_on_tshirt (initial_amount - spent_on_ice_cream) - 
                      deposit_in_bank (spent_on_tshirt (initial_amount - spent_on_ice_cream)) ∧
    final_cash = 24 :=
by {
  sorry
}

end lulu_final_cash_l533_533389


namespace negative_reciprocal_slopes_minimum_area_conjecture_l533_533669

noncomputable def parabola : set (ℝ × ℝ) :=
  {p | ∃ x y, p = ((4 * x), y) ∧ y^2 = 4 * x}

def M (m : ℝ) := (m, 0)
def N (m : ℝ) := (-m, 0)

theorem negative_reciprocal_slopes (x₁ x₂ y₁ y₂ : ℝ) (k : ℝ) 
  (hp1 : y₁ = k*(x₁ - 1)) (hp2 : y₂ = k*(x₂ - 1))
  (hp3 : y₁^2 = 4 * x₁) (hp4 : y₂^2 = 4 * x₂) :
  let k₁ := (y₁ / (x₁ + 1)) in let k₂ := (y₂ / (x₂ + 1)) in
  k₁ = -k₂ := sorry

theorem minimum_area (x₁ x₂ y₁ y₂ : ℝ) (k : ℝ)
  (hp1 : y₁ = k*(x₁ - 1)) (hp2 : y₂ = k*(x₂ - 1))
  (hp3 : y₁^2 = 4 * x₁) (hp4 : y₂^2 = 4 * x₂) :
  let area := |y₁ - y₂| in
  area ≥ 4 := sorry

theorem conjecture (m : ℝ) (hm : m > 0 ∧ m ≠ 1)
  (x₁ x₂ y₁ y₂ : ℝ) (k : ℝ) (hp1 : y₁ = k*(x₁ - m)) 
  (hp2 : y₂ = k*(x₂ - m)) (hp3 : y₁^2 = 4 * x₁) 
  (hp4 : y₂^2 = 4 * x₂) :
  let k₁ := (y₁ / (x₁ + m)) in let k₂ := (y₂ / (x₂ + m)) in 
  k₁ = -k₂ ∧ let area := 4 * m * sqrt m in 
  area ≥ 4 * m * sqrt m := sorry

end negative_reciprocal_slopes_minimum_area_conjecture_l533_533669


namespace properSubsets_31_l533_533297

open Nat

def A : Set ℕ := {x ∈ ℕ | ∃ k ∈ ℕ, 12 = k * (6 - x)}

def properSubsetCount (s : Finset ℕ) : ℕ :=
  2^s.card - 1

theorem properSubsets_31 : properSubsetCount (A.to_finset) = 31 := by
  sorry

end properSubsets_31_l533_533297


namespace smallest_integer_to_make_y_perfect_square_l533_533896

-- Define y as given in the problem
def y : ℕ :=
  2^33 * 3^54 * 4^45 * 5^76 * 6^57 * 7^38 * 8^69 * 9^10

-- Smallest integer n such that (y * n) is a perfect square
theorem smallest_integer_to_make_y_perfect_square
  : ∃ n : ℕ, (∀ k : ℕ, y * n = k * k) ∧ (∀ m : ℕ, (∀ k : ℕ, y * m = k * k) → n ≤ m) := 
sorry

end smallest_integer_to_make_y_perfect_square_l533_533896


namespace sum_m_n_eq_53_l533_533980

theorem sum_m_n_eq_53 (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m + 8 < n + 3) 
(h4 : (m + 8 + (n + 3))/2 = n) 
(h5 : (m + (m+3) + (m+8) + (n+3) + (n+4) + 2n)/6 = n) :
  m + n = 53 := 
sorry

end sum_m_n_eq_53_l533_533980


namespace student_tickets_sold_l533_533869

variable (S N : ℕ)

theorem student_tickets_sold (h1 : S + N = 193) (h2 : 0.50 * S + 1.50 * N = 206.50) : S = 83 :=
by 
  sorry

end student_tickets_sold_l533_533869


namespace set_subset_implies_zero_intersection_sets_l533_533278

-- Definition for first proof problem
theorem set_subset_implies_zero (a : ℝ) (S : set ℝ) :
  (∀ S, {x | a * x = 1} ⊆ S) → a = 0 :=
sorry

-- Definitions for second proof problem
def A := {x : ℝ | -1 < x ∧ x < 4}
def B := {x : ℝ | 2 < x ∧ x < 6}
def C := {x : ℝ | 2 < x ∧ x < 4}

-- Theorem for second proof problem
theorem intersection_sets : A ∩ B = C :=
sorry

end set_subset_implies_zero_intersection_sets_l533_533278


namespace find_k_l533_533641

-- Define the arithmetic sequence and the sum of the first n terms
def a (n : ℕ) : ℤ := 2 * n + 2
def S (n : ℕ) : ℤ := n^2 + 3 * n

-- The main assertion
theorem find_k : ∃ (k : ℕ), k > 0 ∧ (S k - a (k + 5) = 44) ∧ k = 7 :=
by
  sorry

end find_k_l533_533641


namespace function_satisfies_differential_eq_l533_533411

theorem function_satisfies_differential_eq (x : ℝ) (y : ℝ → ℝ) : 
  y = λ x, x * exp (-x^2 / 2) →
  (x * (deriv y x) = (1 - x^2) * y) :=
by
  intro hyp
  rw hyp
  sorry

end function_satisfies_differential_eq_l533_533411


namespace even_three_digit_numbers_count_l533_533478

open Nat

theorem even_three_digit_numbers_count : 
  ∃ (count : ℕ), count = 9 ∧ (∀ (n : ℕ), 
  100 ≤ n ∧ n < 400 ∧ (∀ (d : ℤ), d ∈ [1, 2, 3] → toDigits 10 n ≠ [] → d ∈ (list.map toDigits 10 [n.mod 10, (n / 10).mod 10, (n / 100).mod 10])) ∧ (n.mod 10 ≠ 0 → n.mod 10 = 2) → count = 9) :=
begin
  -- start of proof (you can replace this "sorry" with the actual proof)
  sorry
end

end even_three_digit_numbers_count_l533_533478


namespace periodic_function_value_l533_533180

-- Definitions
def isOddFunction (f : ℝ → ℝ) := ∀ x ∈ ℝ, f(x) = -f(-x)
def satisfiesCondition (f : ℝ → ℝ) := ∀ x ∈ ℝ, f(x) = f(1 - x)
def specificCondition (f : ℝ → ℝ) := ∀ x ∈ ℝ, (0 < x ∧ x ≤ 1/2) → f(x) = 2 * x^2

-- Proof statement
theorem periodic_function_value (f : ℝ → ℝ) 
  (Hodd : isOddFunction f)
  (Hcond : satisfiesCondition f)
  (Hspec : specificCondition f) : 
  f 3 + f (-5/2) = -0.5 :=
by sorry

end periodic_function_value_l533_533180


namespace time_distribution_l533_533854

noncomputable def total_hours_at_work (hours_task1 day : ℕ) (hours_task2 day : ℕ) (work_days : ℕ) (reduce_per_week : ℕ) : ℕ :=
  (hours_task1 + hours_task2) * work_days

theorem time_distribution (h1 : 5 = 5) (h2 : 3 = 3) (days : 5 = 5) (reduction : 5 = 5) :
  total_hours_at_work 5 3 5 5 = 40 :=
by
  sorry

end time_distribution_l533_533854


namespace log_base_c_lt_l533_533308

theorem log_base_c_lt {a b c : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : 0 < c) (h4 : c < 1) : 
  Real.log c a < Real.log c b := 
sorry

end log_base_c_lt_l533_533308


namespace lulu_final_cash_l533_533387

-- Definitions of the problem conditions
def initial_amount : ℕ := 65
def spent_on_ice_cream : ℕ := 5
def spent_on_tshirt (remaining : ℕ) : ℕ := remaining / 2
def deposit_in_bank (remaining : ℕ) : ℕ := remaining / 5

-- The proof problem statement
theorem lulu_final_cash :
  ∃ final_cash : ℕ,
    final_cash = initial_amount - spent_on_ice_cream - spent_on_tshirt (initial_amount - spent_on_ice_cream) - 
                      deposit_in_bank (spent_on_tshirt (initial_amount - spent_on_ice_cream)) ∧
    final_cash = 24 :=
by {
  sorry
}

end lulu_final_cash_l533_533387


namespace smallest_base_10_integer_l533_533847

theorem smallest_base_10_integer :
  ∃ (c d : ℕ), 3 < c ∧ 3 < d ∧ (3 * c + 4 = 4 * d + 3) ∧ (3 * c + 4 = 19) :=
by {
 sorry
}

end smallest_base_10_integer_l533_533847


namespace major_axis_length_l533_533190

-- Define the radius of the cylinder
def cylinder_radius : ℝ := 2

-- Define the relationship given in the problem
def major_axis_ratio : ℝ := 1.6

-- Define the calculation for minor axis
def minor_axis : ℝ := 2 * cylinder_radius

-- Define the calculation for major axis
def major_axis : ℝ := major_axis_ratio * minor_axis

-- The theorem statement
theorem major_axis_length:
  major_axis = 6.4 :=
by 
  sorry -- Proof to be provided later

end major_axis_length_l533_533190


namespace dress_costs_l533_533063

-- Define the costs of the dresses for each person
variables (P J I Pa : ℝ)
variable (X : ℝ)

-- Conditions
def condition1 : Prop := Pa = 30
def condition2 : Prop := J = Pa - 10
def condition3 : Prop := I = J + X
def condition4 : Prop := P = I + 10
def condition5 : Prop := Pa + J + I + P = 160

-- Question to prove
def question : Prop := X = 30

-- The theorem combining all conditions to the question
theorem dress_costs (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) : question :=
by sorry

end dress_costs_l533_533063


namespace angle_between_vectors_l533_533243

-- Declare the vectors as constants
def u : ℝ × ℝ := (3, 4)
def v : ℝ × ℝ := (4, -3)

-- Define functions to calculate dot product and vector magnitude
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def magnitude (a : ℝ × ℝ) : ℝ :=
  Real.sqrt (a.1 ^ 2 + a.2 ^ 2)

-- Statement of the theorem
theorem angle_between_vectors :
  let θ := Real.arccos ((dot_product u v) / ((magnitude u) * (magnitude v))) in
  θ = Real.pi / 2 :=
by
  sorry

end angle_between_vectors_l533_533243


namespace andrew_total_stickers_l533_533049

theorem andrew_total_stickers
  (total_stickers : ℕ)
  (ratio_susan: ℕ)
  (ratio_andrew: ℕ)
  (ratio_sam: ℕ)
  (sam_share_ratio: 2/3)
  (initial_andrew_share: ℕ)
  (initial_sam_share: ℕ) :
  total_stickers = 1500 →
  ratio_susan = 1 →
  ratio_andrew = 1 →
  ratio_sam = 3 →
  initial_andrew_share = (total_stickers / (ratio_susan + ratio_andrew + ratio_sam)) * ratio_andrew →
  initial_sam_share = (total_stickers / (ratio_susan + ratio_andrew + ratio_sam)) * ratio_sam →
  (ratio_andrew = (initial_andrew_share + initial_sam_share * sam_share_ratio)) →
  initial_andrew_share + initial_sam_share * sam_share_ratio = 900 :=
by
  intros _ _ _ _ _ _
  sorry

end andrew_total_stickers_l533_533049


namespace only_n_eq_1_solution_l533_533947

theorem only_n_eq_1_solution (n : ℕ) (h : n > 0): 
  (∃ x : ℤ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 :=
by
  sorry

end only_n_eq_1_solution_l533_533947


namespace marbles_in_larger_container_l533_533887

theorem marbles_in_larger_container :
  (∀ (v n : ℕ), v = 36 → n = 120 → (∃ m : ℕ, v * 2 = 72 ∧ m = n * 2)) →
  (∃ y : ℕ, y = 240) :=
by
  intro h
  obtain ⟨m, hm1, hm2⟩ := h 36 120 rfl rfl
  use m
  rw [←hm2, Nat.mul_comm, Nat.mul_comm 120 2, Nat.mul_assoc (14: ℕ) 2 2, Nat.mul_comm 14 4]
  sorry

end marbles_in_larger_container_l533_533887


namespace part_a_part_b_l533_533866

-- Part (a)
theorem part_a (A B C D P : Point) (h1 : B ∈ segment A D)  (h2 : C ∈ segment A D) (h3 : dist A B = dist C D) : 
  dist P A + dist P D ≥ dist P B + dist P C := sorry

-- Part (b)
theorem part_b (A B C D : Point) (h : ∀ P : Point, dist P A + dist P D ≥ dist P B + dist P C) :
  B ∈ segment A D ∧ C ∈ segment A D ∧ dist A B = dist C D := sorry

end part_a_part_b_l533_533866


namespace lulu_cash_left_l533_533390

-- Define the initial amount
def initial_amount : ℕ := 65

-- Define the amount spent on ice cream
def spent_on_ice_cream : ℕ := 5

-- Define the amount spent on a t-shirt
def spent_on_tshirt (remaining_after_ice_cream : ℕ) : ℕ := remaining_after_ice_cream / 2

-- Define the amount deposited in the bank
def deposited_in_bank (remaining_after_tshirt : ℕ) : ℕ := remaining_after_tshirt / 5

-- Define the remaining cash after all transactions
def remaining_cash (initial : ℕ) (spent_ice_cream : ℕ) (spent_tshirt: ℕ) (deposited: ℕ) :ℕ :=
  initial - spent_ice_cream - spent_tshirt - deposited

-- Theorem statement to prove
theorem lulu_cash_left : remaining_cash initial_amount spent_on_ice_cream (spent_on_tshirt (initial_amount - spent_on_ice_cream)) 
(deposited_in_bank ((initial_amount - spent_on_ice_cream) - (spent_on_tshirt (initial_amount - spent_on_ice_cream)))) = 24 :=
by
  sorry

end lulu_cash_left_l533_533390


namespace friends_recycled_pounds_l533_533503

-- Definitions of given conditions
def points_earned : ℕ := 6
def pounds_per_point : ℕ := 8
def zoe_pounds : ℕ := 25

-- Calculation based on given conditions
def total_pounds := points_earned * pounds_per_point
def friends_pounds := total_pounds - zoe_pounds

-- Statement of the proof problem
theorem friends_recycled_pounds : friends_pounds = 23 := by
  sorry

end friends_recycled_pounds_l533_533503


namespace camera_value_l533_533726

variables (V : ℝ)

def rental_fee_per_week (V : ℝ) := 0.1 * V
def total_rental_fee(V : ℝ) := 4 * rental_fee_per_week V
def johns_share_of_fee(V : ℝ) := 0.6 * (0.4 * total_rental_fee V)

theorem camera_value (h : johns_share_of_fee V = 1200): 
  V = 5000 :=
by
  sorry

end camera_value_l533_533726


namespace largest_k_inequality_l533_533611

theorem largest_k_inequality :
  ∃ k : ℝ, (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → a + b + c = 3 → a^3 + b^3 + c^3 - 3 ≥ k * (3 - a * b - b * c - c * a)) ∧ k = 5 :=
sorry

end largest_k_inequality_l533_533611


namespace find_rate_compound_interest_l533_533826

noncomputable def sum_si : ℝ := 1833.33
noncomputable def rate_si : ℝ := 16
noncomputable def time_si : ℝ := 6
noncomputable def si : ℝ := (sum_si * rate_si * time_si) / 100
noncomputable def ci : ℝ := 2 * si
noncomputable def principal_ci : ℝ := 8000
noncomputable def time_ci : ℝ := 2

theorem find_rate_compound_interest (R : ℝ) :
  let compound_interest := principal_ci * ((1 + R/100)^2 - 1) in
  compound_interest = ci →
  R ≈ 20 := sorry

end find_rate_compound_interest_l533_533826


namespace probability_white_ball_initial_probability_yellow_or_red_ball_initial_probability_white_ball_after_removal_l533_533704

def total_balls := 20
def red_balls := 10
def yellow_balls := 6
def white_balls := 4
def initial_white_balls_probability := (white_balls : ℚ) / total_balls
def initial_yellow_or_red_balls_probability := (yellow_balls + red_balls : ℚ) / total_balls

def removed_red_balls := 2
def removed_white_balls := 2
def remaining_balls := total_balls - (removed_red_balls + removed_white_balls)
def remaining_white_balls := white_balls - removed_white_balls
def remaining_white_balls_probability := (remaining_white_balls : ℚ) / remaining_balls

theorem probability_white_ball_initial : initial_white_balls_probability = 1 / 5 := by sorry
theorem probability_yellow_or_red_ball_initial : initial_yellow_or_red_balls_probability = 4 / 5 := by sorry
theorem probability_white_ball_after_removal : remaining_white_balls_probability = 1 / 8 := by sorry

end probability_white_ball_initial_probability_yellow_or_red_ball_initial_probability_white_ball_after_removal_l533_533704


namespace area_QCA_area_ACD_l533_533944

-- Define the points and their coordinates
def Q : ℝ × ℝ := (0, 15)
def A : ℝ × ℝ := (3, 15)
def C (p : ℝ) : ℝ × ℝ := (0, p)
def D : ℝ × ℝ := (3, 0)

-- Define a function to calculate the area of a triangle given 3 points in the plane
def area_triangle (P1 P2 P3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((P1.1 * (P2.2 - P3.2)) + (P2.1 * (P3.2 - P1.2)) + (P3.1 * (P1.2 - P2.2)))

-- Statement for triangle QCA area
theorem area_QCA (p : ℝ) : area_triangle Q (C p) A = (45 - 3 * p) / 2 := by
  sorry

-- Statement for triangle ACD area
theorem area_ACD : area_triangle A C D = 22.5 := by
  sorry

end area_QCA_area_ACD_l533_533944


namespace number_of_buns_l533_533256

-- Define the necessary constants related to the problem
def cost_per_bun : ℝ := 0.1
def total_cost_milk : ℝ := 4
def cost_eggs : ℝ := 6
def total_cost : ℝ := 11

-- Define the final property we need to prove
theorem number_of_buns (B : ℕ) :
  B * cost_per_bun + total_cost_milk + cost_eggs = total_cost → B = 10 :=
begin
  intro h,
  -- Proof can be continued here
  sorry
end

end number_of_buns_l533_533256


namespace correct_calculation_l533_533154

variable (a : ℝ)

theorem correct_calculation : (2 * a ^ 3) ^ 3 = 8 * a ^ 9 :=
by sorry

end correct_calculation_l533_533154


namespace limit_sequence_is_5_l533_533576

noncomputable def sequence (n : ℕ) : ℝ :=
  (real.sqrt (3 * n - 1) - real.cbrt (125 * n ^ 3 + n)) / (real.rpow n (1 / 5) - n)

theorem limit_sequence_is_5 : 
  filter.tendsto (sequence) at_top (nhds 5) :=
sorry

end limit_sequence_is_5_l533_533576


namespace chess_tournament_games_l533_533517

theorem chess_tournament_games (n : ℕ) (h : n = 10) : 2 * n * (n - 1) = 180 :=
by {
  rw h,
  norm_num,
  sorry,
}

end chess_tournament_games_l533_533517


namespace aubree_total_animals_l533_533566

noncomputable def total_animals_seen : Nat :=
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10
  morning_total + new_beavers + new_chipmunks

theorem aubree_total_animals :
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10
  morning_total + new_beavers + new_chipmunks = 130 := by
  -- Define all constants and conditions
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10

  -- State the equation
  show morning_total + new_beavers + new_chipmunks = 130 from sorry

end aubree_total_animals_l533_533566


namespace tangent_line_AB_l533_533288

noncomputable def circle := λ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 2

def point_P := (2, -1 : ℝ × ℝ)

theorem tangent_line_AB :
  ∃ A B : ℝ × ℝ, ((A.1 - 1)^2 + (A.2 - 2)^2 = 2)
  ∧ ((B.1 - 1)^2 + (B.2 - 2)^2 = 2)
  ∧ (line_through P A)
  ∧ (line_through P B)
  → ∀ x y : ℝ, x - 3 * y + 3 = 0 :=
begin
  sorry
end

end tangent_line_AB_l533_533288


namespace octopus_shoes_needed_l533_533889

-- Defining the basic context: number of legs and current shod legs
def num_legs : ℕ := 8

-- Conditions based on the number of already shod legs for each member
def father_shod_legs : ℕ := num_legs / 2       -- Father-octopus has half of his legs shod
def mother_shod_legs : ℕ := 3                  -- Mother-octopus has 3 legs shod
def son_shod_legs : ℕ := 6                     -- Each son-octopus has 6 legs shod
def num_sons : ℕ := 2                          -- There are 2 sons

-- Calculate unshod legs for each 
def father_unshod_legs : ℕ := num_legs - father_shod_legs
def mother_unshod_legs : ℕ := num_legs - mother_shod_legs
def son_unshod_legs : ℕ := num_legs - son_shod_legs

-- Aggregate the total shoes needed based on unshod legs
def total_shoes_needed : ℕ :=
  father_unshod_legs + 
  mother_unshod_legs + 
  (son_unshod_legs * num_sons)

-- The theorem to prove
theorem octopus_shoes_needed : total_shoes_needed = 13 := 
  by 
    sorry

end octopus_shoes_needed_l533_533889


namespace student_A_more_stable_l533_533465

-- Given conditions
def average_score (n : ℕ) (score : ℕ) := score = 110
def variance_A := 3.6
def variance_B := 4.4

-- Prove that student A has more stable scores than student B
theorem student_A_more_stable : variance_A < variance_B :=
by
  -- Skipping the actual proof
  sorry

end student_A_more_stable_l533_533465


namespace incorrect_statements_l533_533498

-- Define the conditions
def empty_subset (A : Set) : Prop := ∅ ⊆ A

def odd_increasing (f : ℝ → ℝ) : Prop := 
  ∀ x y, 0 < x → x < y → f(x) < f(y) ∧ f(-x) = -f(x)

def min_value (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f(x) = (x^2 + 3) / (sqrt(x^2 + 2)) → 2 ≤ f(x)

def quad_roots_within (m : ℝ) : Prop := 
  ∀ x, x ∈ (1 : ℝ, +∞) →
  ∃ a b : ℝ, a > 1 ∧ b > 1 ∧ 
  (a - b)^2 = m^2 - 8 ∧ 
  a + b = m ∧
  a * b = 2

-- Prove the incorrect statements given the conditions
theorem incorrect_statements:
  ¬ (∀ (f : ℝ → ℝ), odd_increasing f → ∀ x y : ℝ, x < y → f(x) < f(y)) ∧
  ¬ (min_value (λ x, (x^2 + 3) / (sqrt(x^2 + 2)))) ∧
  ¬ quad_roots_within (2*sqrt(2)) →
  "BCD" = "BCD" :=
by { sorry }

end incorrect_statements_l533_533498


namespace correct_operation_only_l533_533157

theorem correct_operation_only (a b x y : ℝ) : 
  (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) ∧ 
  (4 * x^2 * y - x^2 * y ≠ 3) ∧ 
  ((a + b)^2 ≠ a^2 + b^2) ∧
  ((ab)^3 = a^3 * b^3) := 
by 
  sorry

end correct_operation_only_l533_533157


namespace collinear_A_l533_533269

-- Auxiliary definitions, such as the definition of a scalene triangle and the properties of angle bisectors
def scalene_triangle (A B C : Type _) := sorry
def angle_bisector (A B C P : Type _) : Prop := sorry
def perpendicular_bisector (X Y Z : Type _) : Prop := sorry
def is_collinear (P Q R : Type _) : Prop := sorry

theorem collinear_A''_B''_C'' 
  {A B C A' B' C' A'' B'' C'' : Type _}
  (h_scalene : scalene_triangle A B C)
  (h_A' : angle_bisector A B C A')
  (h_B' : angle_bisector B A C B')
  (h_C' : angle_bisector C A B C')
  (h_A'' : perpendicular_bisector A A' A'')
  (h_B'' : perpendicular_bisector B B' B'')
  (h_C'' : perpendicular_bisector C C' C'') :
  is_collinear A'' B'' C'' := sorry

end collinear_A_l533_533269


namespace angle_between_AD_and_BE_is_60_deg_l533_533839

open EuclideanGeometry

-- Definitions for equilateral triangles and shared vertex condition
variables {A B C D E : Point}

-- Lean statement for the given proof problem
theorem angle_between_AD_and_BE_is_60_deg
  (h1 : EquilateralTriangle A B C)
  (h2 : EquilateralTriangle C D E)
  (h3 : SharedVertex C): 
  ∠(A, D, CommonVertex P AD BE) = 60 := 
sorry

end angle_between_AD_and_BE_is_60_deg_l533_533839


namespace find_counterfeit_coin_in_two_weighings_l533_533281

-- Define the conditions for the problem
def num_coins : ℕ := 5
def genuine_coin_weight : ℤ := 5
def has_balance_scale : Prop := true
def has_reference_weight : ℤ := 5

-- Define the main theorem
theorem find_counterfeit_coin_in_two_weighings :
  ∃ is_counterfeit : (fin num_coins → bool),
  (exists_counterfeit : ∃ i : fin num_coins, is_counterfeit i = true) →
  (∀ c : fin num_coins, c ≠ i → is_counterfeit c = false) →
  -- We can find the counterfeit coin in exactly two weighings
  (number_of_weighings : ℕ = 2) :=
by
  sorry

end find_counterfeit_coin_in_two_weighings_l533_533281


namespace train_length_correct_l533_533910

noncomputable def train_length (speed_kmph : ℝ) (time_s : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let speed_mps := (speed_kmph * 1000) / 3600
  let total_distance := speed_mps * time_s
  total_distance - bridge_length_m

theorem train_length_correct :
  train_length 36 82.49340052795776 660 = 164.9340052795776 :=
by
  unfold train_length
  norm_num
  sorry

end train_length_correct_l533_533910


namespace sufficient_condition_for_m_l533_533275

variable (x m : ℝ)

def p (x : ℝ) : Prop := abs (x - 4) ≤ 6
def q (x m : ℝ) : Prop := x ≤ 1 + m

theorem sufficient_condition_for_m (h : ∀ x, p x → q x m ∧ ∃ x, ¬p x ∧ q x m) : m ≥ 9 :=
sorry

end sufficient_condition_for_m_l533_533275


namespace percentage_reduction_l533_533164

noncomputable def P : ℝ := 1  -- price
noncomputable def C : ℝ := 1  -- consumption
def increase_25_percent (price : ℝ) : ℝ := 1.25 * price
def increase_10_percent (price : ℝ) : ℝ := 1.10 * price
def new_price := increase_10_percent (increase_25_percent P)
def original_expenditure := P * C
def new_expenditure (C_new : ℝ) := new_price * C_new

theorem percentage_reduction :
  ∃ (C_new : ℝ), original_expenditure = new_expenditure C_new →
  ((1 - C_new / C) * 100) = 27.27 :=
by
  sorry

end percentage_reduction_l533_533164


namespace sum_of_real_roots_interval_l533_533618

theorem sum_of_real_roots_interval :
  (∑ x in {x | sin (π * (x^2 - x + 1)) = sin (π * (x - 1)) ∧ x ∈ Icc 0 2}, x) = 3 + Real.sqrt 3 :=
sorry

end sum_of_real_roots_interval_l533_533618


namespace num_students_wearing_short_sleeves_not_short_pants_l533_533421

def students_wearing_short_sleeves := {1, 3, 7, 10, 23, 27}
def students_wearing_short_pants := {1, 9, 11, 20, 23}

theorem num_students_wearing_short_sleeves_not_short_pants :
  (students_wearing_short_sleeves \ students_wearing_short_pants).size = 4 :=
by
  sorry

end num_students_wearing_short_sleeves_not_short_pants_l533_533421


namespace find_denominator_l533_533549

noncomputable def original_denominator (d : ℝ) : Prop :=
  (7 / (d + 3)) = 2 / 3

theorem find_denominator : ∃ d : ℝ, original_denominator d ∧ d = 7.5 :=
by
  use 7.5
  unfold original_denominator
  sorry

end find_denominator_l533_533549


namespace arithmetic_common_difference_l533_533026

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533026


namespace max_product_yields_831_l533_533470

open Nat

noncomputable def max_product_3digit_2digit : ℕ :=
  let digits : List ℕ := [1, 3, 5, 8, 9]
  let products := digits.permutations.filter (λ l, l.length ≥ 5).map (λ l, 
    let a := l.get? 0 |>.getD 0
    let b := l.get? 1 |>.getD 0
    let c := l.get? 2 |>.getD 0
    let d := l.get? 3 |>.getD 0
    let e := l.get? 4 |>.getD 0
    (100 * a + 10 * b + c) * (10 * d + e)
  )
  products.maximum.getD 0

theorem max_product_yields_831 : 
  ∃ (a b c d e : ℕ), 
    List.permutations [1, 3, 5, 8, 9].any 
      (λ l, l = [a, b, c, d, e] ∧ 
            100 * a + 10 * b + c = 831 ∧
            ∀ (x y z w v : ℕ), 
              List.permutations [1, 3, 5, 8, 9].any 
                (λ l, l = [x, y, z, w, v] ∧ 
                      (100 * x + 10 * y + z) * (10 * w + v) ≤ (100 * a + 10 * b + c) * (10 * d + e)
                )
      )
:= by
  sorry

end max_product_yields_831_l533_533470


namespace tina_record_l533_533115

theorem tina_record :
  let initial_wins := 10 in
  let wins_after_first_series := initial_wins + 5 in
  let wins_after_second_series := wins_after_first_series * 3 in
  let wins_after_third_series := wins_after_second_series + 7 in
  let total_wins := wins_after_third_series * 2 in
  let total_losses :=  3 in
  (total_wins - total_losses) = 131 :=
  by
  sorry

end tina_record_l533_533115


namespace part_a_not_single_minimum_part_b_d_c_le_d_m_l533_533624

def d (t : ℝ) (x : List ℝ) : ℝ :=
  (min (List.minimum (List.map (λ x_i, abs (x_i - t)) x)) +
   max (List.maximum (List.map (λ x_i, abs (x_i - t)) x))) / 2

def c (x : List ℝ) : ℝ :=
  (List.minimum x + List.maximum x) / 2

def median (x : List ℝ) : ℝ := -- This needs to be defined properly, skipped for brevity
  sorry

theorem part_a_not_single_minimum (x : List ℝ) : 
  ¬ ∃ t : ℝ, ∀ t' : ℝ, d t x < d t' x :=
sorry

theorem part_b_d_c_le_d_m (x : List ℝ) :
  d (c x) x ≤ d (median x) x :=
sorry

end part_a_not_single_minimum_part_b_d_c_le_d_m_l533_533624


namespace candles_from_24_hives_l533_533367

def candles_per_hive (candles hives : ℕ) : ℕ :=
  candles / hives

def total_candles (candles_per_hive hives : ℕ) : ℕ :=
  candles_per_hive * hives

theorem candles_from_24_hives :
  ∀ (candles : ℕ) (hives : ℕ) (candles_per_hive: ℕ),
    candles = 12 → hives = 3 → candles_per_hive = candles_per_hive candles hives → 
    total_candles candles_per_hive 24 = 96 :=
begin
  intros,
  rw [candles_per_hive, total_candles] at *,
  sorry -- to be proven
end

end candles_from_24_hives_l533_533367


namespace adrianna_gum_pieces_l533_533200

-- Definitions based on conditions
def initial_gum_pieces : ℕ := 10
def additional_gum_pieces : ℕ := 3
def friends_count : ℕ := 11

-- Expression to calculate the final pieces of gum
def total_gum_pieces : ℕ := initial_gum_pieces + additional_gum_pieces
def gum_left : ℕ := total_gum_pieces - friends_count

-- Lean statement we want to prove
theorem adrianna_gum_pieces: gum_left = 2 := 
by 
  sorry

end adrianna_gum_pieces_l533_533200


namespace equivalent_problem_l533_533266

def equation_of_circle : (ℝ → ℝ → Prop) :=
  λ x y, (x + 1)^2 + (y - 2)^2 = 25

def equation_of_line (m : ℝ) : (ℝ → ℝ → Prop) :=
  λ x y, (3 * m + 1) * x + (m + 1) * y - 5 * m - 3 = 0

def line_intersects_circle (m x y : ℝ) : Prop :=
  equation_of_line m x y ∧ equation_of_circle x y

def line_through_fixed_point (m x y : ℝ) : Prop :=
  equation_of_line m x y ∧ x = 2 ∧ y = 1

def chord_length_on_y_axis : ℝ :=
  4 * Real.sqrt 6

def shortest_chord_length_eqn (m : ℝ) : (ℝ → ℝ → Prop) :=
  λ x y, x = 1

theorem equivalent_problem (m x y : ℝ) :
  (line_intersects_circle m x y ∧
  ¬line_through_fixed_point m x y ∧
  chord_length_on_y_axis = 4 * Real.sqrt 6 ∧
  shortest_chord_length_eqn m x y) :=
sorry

end equivalent_problem_l533_533266


namespace am_plus_an_eq_ab_l533_533694

theorem am_plus_an_eq_ab
  (r a : ℝ)
  (h1 : 2 * a < r / 2)
  (h2 : 2 * r = |AB|)
  (M N : Point) -- assuming Point is defined elsewhere to represent points
  (hM : OnSemicircle M r) -- Assuming OnSemicircle is a predicate indicating the point is on the semicircle of radius r
  (hN : OnSemicircle N r)
  (h3 : ∃ l : Line, Perpendicular l (extension BA) ∧ distance(A, l) = 2*a)
  (h4 : distance_from_line M l / distance A M = 1)
  (h5 : distance_from_line N l / distance A N = 1) :
  distance A M + distance A N = distance A B := sorry

end am_plus_an_eq_ab_l533_533694


namespace untouched_shapes_after_moves_l533_533861

-- Definitions
def num_shapes : ℕ := 12
def num_triangles : ℕ := 3
def num_squares : ℕ := 4
def num_pentagons : ℕ := 5
def total_moves : ℕ := 10
def petya_moves_first : Prop := True
def vasya_strategy : Prop := True  -- Vasya's strategy to minimize untouched shapes
def petya_strategy : Prop := True  -- Petya's strategy to maximize untouched shapes

-- Theorem
theorem untouched_shapes_after_moves : num_shapes = 12 ∧ num_triangles = 3 ∧ num_squares = 4 ∧ num_pentagons = 5 ∧
                                        total_moves = 10 ∧ petya_moves_first ∧ vasya_strategy ∧ petya_strategy → 
                                        num_shapes - 5 = 6 :=
by
  sorry

end untouched_shapes_after_moves_l533_533861


namespace larger_page_of_opened_book_l533_533329

theorem larger_page_of_opened_book (x : ℕ) (h : x + (x + 1) = 137) : x + 1 = 69 :=
sorry

end larger_page_of_opened_book_l533_533329


namespace minimum_marked_cells_k_l533_533142

def cell := (ℕ × ℕ)

def is_marked (board : set cell) (cell : cell) : Prop :=
  cell ∈ board

def touches_marked_cell (board : set cell) (fig : set cell) : Prop :=
  ∃ (c : cell), c ∈ fig ∧ is_marked(board, c)

def four_cell_figures (fig : set cell) : Prop :=
  (fig = { (0, 0), (0, 1), (1, 0), (1, 1) }) ∨
  (fig = { (0, 0), (1, 0), (2, 0), (3, 0) }) ∨
  (fig = { (0, 0), (0, 1), (0, 2), (0, 3) }) ∨
  -- Add other rotations and flipped configurations if needed

def board12x12 :=
  { (i, j) | i < 12 ∧ j < 12 }

theorem minimum_marked_cells_k :
  ∃ (k : ℕ) (marked_cells : (fin 144) → bool),
  k = 48 ∧
  (∀ (fig : set cell),
    four_cell_figures fig →
    ∀ (x y : ℕ), x < 12 ∧ y < 12 →
      touches_marked_cell
        { cell | marked_cells (fin.mk (cell.1 * 12 + cell.2) _) }
        (image (λ (rc : cell), (rc.1 + x, rc.2 + y)) fig)
  ) :=
begin
  sorry
end

end minimum_marked_cells_k_l533_533142


namespace container_capacity_l533_533877

variable (C : ℝ) -- Let C be a real number representing the capacity of the container
variable (h1 : 0.3 * C + 18 = 0.75 * C) -- The given condition when 18 liters is added it becomes 3/4 full

theorem container_capacity : C = 40 :=
by
  have eq1 : 0.75 * C - 0.3 * C = 18 := h1
  have eq2 : 0.45 * C = 18 := by linarith
  have result : C = 18 / 0.45 := by linarith
  have simpl_result : 18 / 0.45 = 40 := by norm_num
  exact calc
    C = 18 / 0.45 : result
    ... = 40 : simpl_result

end container_capacity_l533_533877


namespace algebraic_expression_value_l533_533332

-- Given condition
def cond (a : ℝ) : Prop := a^2 + a = 3

-- The expression whose value we need to determine
def expr (a : ℝ) : ℝ := 2 * a^2 + 2 * a - 1

-- The theorem stating our problem
theorem algebraic_expression_value (a : ℝ) (h : cond a) : expr a = 5 :=
sorry

end algebraic_expression_value_l533_533332


namespace tennis_game_probability_l533_533520

theorem tennis_game_probability (p : ℝ) (hp : p ≤ 1/2) :
  (prob_win_A : ℝ) (hprob_win_A: prob_win_A = probability_A_wins p) :
  prob_win_A ≤ 2 * p^2 := by
  sorry

def probability_A_wins (p : ℝ) : ℝ :=
  -- Definitions and computation of probability based on given conditions
  sorry

end tennis_game_probability_l533_533520


namespace greatest_gcd_of_factors_of_7200_l533_533479

open Nat

theorem greatest_gcd_of_factors_of_7200 (a b : ℕ) (h : a * b = 7200) : 
  ∃ d, d ∣ a ∧ d ∣ b ∧ ∀ d', (d' ∣ a ∧ d' ∣ b → d' ≤ d) ∧ d = 60 :=
begin
  sorry
end

end greatest_gcd_of_factors_of_7200_l533_533479


namespace minimum_odd_integers_l533_533118

theorem minimum_odd_integers (x y a b m n : ℤ) 
  (h1 : x + y = 28)
  (h2 : x + y + a + b = 45)
  (h3 : x + y + a + b + m + n = 60) :
  ∃ (min_odd_count : ℕ), min_odd_count = 2 :=
begin
  sorry
end

end minimum_odd_integers_l533_533118


namespace find_geometric_sequence_l533_533606

def geometric_sequence : Prop :=
  ∃ (a r : ℂ), 
  (r^2 - 1 ≠ 0) ∧
  (r^4 - 1 ≠ 0) ∧
  (ar^2 - a = 48) ∧
  ((a * r^2)^2 - a^2) / (a^2 + (a * r)^2 + (a * r^2)^2) = 208 / 217

theorem find_geometric_sequence :
  geometric_sequence :=
sorry

end find_geometric_sequence_l533_533606


namespace present_age_of_son_l533_533505

theorem present_age_of_son
  (S M : ℕ)
  (h1 : M = S + 24)
  (h2 : M + 2 = 2 * (S + 2)) :
  S = 22 :=
by {
  sorry
}

end present_age_of_son_l533_533505


namespace circle_tangents_radius_l533_533355

theorem circle_tangents_radius (
  (AB CD : ℝ)
  (h_AB : AB = 12)
  (h_CD : CD = 20)
  (EF : ℝ)
  (h_EF : EF = 8)) :
  ∃ r : ℝ, r = 6 :=
by
  sorry

end circle_tangents_radius_l533_533355


namespace sqrt_of_square_of_neg_eleven_l533_533827

-- Define the problem statement as a theorem in Lean
theorem sqrt_of_square_of_neg_eleven : real.sqrt ((-11 : ℝ) ^ 2) = 11 := by
  sorry

end sqrt_of_square_of_neg_eleven_l533_533827


namespace tanner_saved_in_november_l533_533080

-- Defining the conditions as constants.
constant saved_september : ℝ := 17
constant saved_october : ℝ := 48
constant spent_video_game : ℝ := 49
constant money_left : ℝ := 41

-- The money Tanner saved in November
def saved_november (N : ℝ) : Prop :=
  saved_september + saved_october + N - spent_video_game = money_left

-- The goal is to prove that Tanner saved 25 in November.
theorem tanner_saved_in_november : saved_november 25 :=
by
  sorry

end tanner_saved_in_november_l533_533080


namespace R_and_D_expenditure_per_unit_increase_l533_533930

theorem R_and_D_expenditure_per_unit_increase :
  (R_t : ℝ) (delta_APL_t_plus_1 : ℝ) (R_t = 3157.61) (delta_APL_t_plus_1 = 0.69) :
  R_t / delta_APL_t_plus_1 = 4576 :=
by
  sorry

end R_and_D_expenditure_per_unit_increase_l533_533930


namespace equal_segments_l533_533767

open real

/-- Given points A, B, and C on a line with C between A and B,
and circles constructed on diameters AB, AC, and BC respectively,
a chord MQ of the circle with diameter AB intersects the circles
with diameters AC and BC at points N and P respectively.
Prove that the segments MN and PQ are equal. -/
theorem equal_segments
  (A B C : ℝ)
  (hABC : A < C ∧ C < B)
  (M Q N P : ℝ)
  (hMQ : chord_of_circle M Q (circle_diameter AB))
  (hN : N = intersection_point with diameter AC and chord MQ)
  (hP : P = intersection_point with diameter BC and chord MQ) :
  segment_length M N = segment_length P Q :=
sorry

end equal_segments_l533_533767


namespace product_of_points_l533_533918

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def total_points (rolls : List ℕ) : ℕ :=
  List.sum (rolls.map g)

def allie_rolls := [3, 6, 5, 2, 4]
def betty_rolls := [2, 6, 1, 4]

theorem product_of_points : total_points allie_rolls * total_points betty_rolls = 308 :=
by
  have h_allie : total_points allie_rolls = 22 := by admit -- Prove allie's total points here
  have h_betty : total_points betty_rolls = 14 := by admit -- Prove betty's total points here
  rw [h_allie, h_betty]
  exact Nat.mul_eq_308_of_22_and_14 -- Assuming multiplication lemma for 22*14=308

end product_of_points_l533_533918


namespace steel_allocation_l533_533211

theorem steel_allocation (steel_per_A : ℝ) (steel_per_B : ℝ) (total_steel : ℝ) (num_A_parts : ℕ) (num_B_parts : ℕ) :
  (num_A_parts = 1) → (num_B_parts = 3) → (steel_per_A * 40 = 1) → (steel_per_B * 240 = 1) → (total_steel = 6) →
    let x := total_steel / (steel_per_A + 3 * steel_per_B) in
    (x = 4) ∧ (total_steel - x = 2) :=
by
  intros hA hB hSteelA hSteelB hTotal
  let x := total_steel / (steel_per_A + 3 * steel_per_B)
  have hx : x = 4 := sorry
  have hx_remaining : total_steel - x = 2 := sorry
  exact ⟨hx, hx_remaining⟩

end steel_allocation_l533_533211


namespace sum_of_interior_angles_of_pentagon_l533_533107

theorem sum_of_interior_angles_of_pentagon :
  let n := 5
  let angleSum := 180 * (n - 2)
  angleSum = 540 :=
by
  sorry

end sum_of_interior_angles_of_pentagon_l533_533107


namespace find_a_from_roots_l533_533659

theorem find_a_from_roots (a : ℝ) :
  let A := {x | (x = a) ∨ (x = a - 1)}
  2 ∈ A → a = 2 ∨ a = 3 :=
by
  intros A h
  sorry

end find_a_from_roots_l533_533659


namespace parking_probability_l533_533186

-- Define the parking procedure
def is_valid_parking (n : ℕ) (parking : list ℕ) : Prop :=
  ∀ k, k < n → k ∉ parking → 
    (∀ d1 d2, d1 ≠ d2 → 
      (abs (k - d1) ≠ abs (k - d2) ∨ d1 ∉ parking ∨ d2 ∉ parking))

-- Define the function to calculate the number of spots left (f)
def f : ℕ → ℕ
| 1 := 1
| 2 := 2
| n := if 2^nat.floor_log2 (n - 1) ≤ n ∧ n < 3 / 2 * 2^nat.floor_log2 (n - 1) then
  n - 2^(nat.floor_log2 (n - 1) - 1) + 1
else
  2^nat.floor_log2 (n - 1)

theorem parking_probability :
  let spots := 2012,
      remaining := f (spots - 3) + 1 in
  (1 / spots : ℚ) * (1 / remaining : ℚ) = 1 / 2062300 := by
  sorry

end parking_probability_l533_533186


namespace angle_measure_l533_533686

-- Define the complement function
def complement (α : ℝ) : ℝ := 180 - α

-- Given condition
variable (α : ℝ)
variable (h : complement α = 120)

-- Theorem to prove
theorem angle_measure : α = 60 :=
by sorry

end angle_measure_l533_533686


namespace sufficient_but_not_necessary_condition_of_p_q_l533_533628

variable (x m : ℝ)

def p := (x-2)*(x-6) ≤ 32
def q := x^2 - 2*x + 1 - m^2 ≤ 0
def not_p := ¬p
def not_q := ¬q

theorem sufficient_but_not_necessary_condition_of_p_q :
  (¬p → ¬q) →
  0 < m ∧ m ≤ 3 :=
by
  sorry

end sufficient_but_not_necessary_condition_of_p_q_l533_533628


namespace case_one_ellipses_foci_xaxis_case_two_ellipses_foci_exact_l533_533249

theorem case_one_ellipses_foci_xaxis :
  ∀ (a : ℝ) (e : ℝ), a = 6 ∧ e = 2 / 3 → (∃ (b : ℝ), (b^2 = (a^2 - (e * a)^2) ∧ (a > 0) → ((∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)) ∨ (y^2 / a^2 + x^2 / b^2 = 1)))) :=
by
  sorry

theorem case_two_ellipses_foci_exact :
  ∀ (F1 F2 : ℝ × ℝ), F1 = (-4,0) ∧ F2 = (4,0) ∧ ∀ P : ℝ × ℝ, ((dist P F1) + (dist P F2) = 10) →
  ∃ (a : ℝ) (b : ℝ), a = 5 ∧ b^2 = a^2 - 4^2 → ((∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1))) :=
by
  sorry

end case_one_ellipses_foci_xaxis_case_two_ellipses_foci_exact_l533_533249


namespace integral_inequality_l533_533781

noncomputable def fn (n : ℕ) : ℝ → ℝ
| 0     := id
| (n+1) := λ x, tan (fn n x)

theorem integral_inequality :
  (∫ x in 0..(π/4), x / (cos x) ^ 2 / (cos (tan x)) ^ 2 / (cos (tan (tan x))) ^ 2 / (cos (tan (tan (tan x)))) ^ 2) ^ (1/3) < 4 / π := 
sorry

end integral_inequality_l533_533781


namespace problem_condition_necessary_and_sufficient_l533_533310

theorem problem_condition_necessary_and_sufficient (a b : ℝ) (h : a * b > 0) :
  (a > b) ↔ (1 / a < 1 / b) :=
sorry

end problem_condition_necessary_and_sufficient_l533_533310


namespace john_pays_8000_l533_533369

theorem john_pays_8000 (
  upfront_fee : ℕ := 1000,
  hourly_rate : ℕ := 100,
  court_time : ℕ := 50,
  prep_time_factor : ℕ := 2,
  brother_contribution_factor : ℕ := 2
) : 
  let prep_time := prep_time_factor * court_time
  let total_hours := court_time + prep_time
  let hourly_fee := hourly_rate * total_hours
  let total_fee := upfront_fee + hourly_fee
  let brother_share := total_fee / brother_contribution_factor
  let john_pays := total_fee - brother_share
  john_pays = 8000 :=
by
  sorry

end john_pays_8000_l533_533369


namespace similar_triangle_legs_l533_533904

theorem similar_triangle_legs {y : ℝ} 
  (h₁ : 12 / y = 9 / 6) : y = 8 :=
sorry

end similar_triangle_legs_l533_533904


namespace volume_conversion_l533_533543

theorem volume_conversion (V_inches : ℕ) (V_feet : ℕ) (h1 : V_inches = 1728) (h2 : 12^3 = 1728) : V_inches / 1728 = V_feet → V_feet = 1 :=
by
  intros
  rw h1 at *
  rw h2 at *
  norm_num
  assumption

end volume_conversion_l533_533543


namespace arithmetic_common_difference_l533_533020

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533020


namespace contradiction_method_at_most_one_positive_l533_533476

theorem contradiction_method_at_most_one_positive :
  (∃ a b c : ℝ, (a > 0 → (b ≤ 0 ∧ c ≤ 0)) ∧ (b > 0 → (a ≤ 0 ∧ c ≤ 0)) ∧ (c > 0 → (a ≤ 0 ∧ b ≤ 0))) → 
  (¬(∃ a b c : ℝ, (a > 0 ∧ b > 0) ∨ (b > 0 ∧ c > 0) ∨ (a > 0 ∧ c > 0))) :=
by sorry

end contradiction_method_at_most_one_positive_l533_533476


namespace non_adjacent_ball_arrangements_l533_533212

-- Statement only, proof is omitted
theorem non_adjacent_ball_arrangements :
  let n := (3: ℕ) -- Number of identical yellow balls
  let white_red_positions := (4: ℕ) -- Positions around the yellow unit
  let choose_positions := Nat.choose white_red_positions 2
  let arrange_balls := (2: ℕ) -- Ways to arrange the white and red balls in the chosen positions
  let total_arrangements := choose_positions * arrange_balls
  total_arrangements = 12 := 
by
  sorry

end non_adjacent_ball_arrangements_l533_533212


namespace recommendation_plans_count_l533_533700

def num_male : ℕ := 3
def num_female : ℕ := 2
def num_recommendations : ℕ := 5

def num_spots_russian : ℕ := 2
def num_spots_japanese : ℕ := 2
def num_spots_spanish : ℕ := 1

def condition_russian (males : ℕ) : Prop := males > 0
def condition_japanese (males : ℕ) : Prop := males > 0

theorem recommendation_plans_count : 
  (∃ (males_r : ℕ) (males_j : ℕ), condition_russian males_r ∧ condition_japanese males_j ∧ 
  num_male - males_r - males_j >= 0 ∧ males_r + males_j ≤ num_male ∧ 
  num_female + (num_male - males_r - males_j) >= num_recommendations - (num_spots_russian + num_spots_japanese + num_spots_spanish)) →
  (∃ (x : ℕ), x = 24) := by
  sorry

end recommendation_plans_count_l533_533700


namespace matrix_multiplication_correct_l533_533585

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, -2], ![4, 5]]

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -6], ![-1, 3]]

def C : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![8, -24], ![3, -9]]

theorem matrix_multiplication_correct :
  A ⬝ B = C :=
by 
  sorry

end matrix_multiplication_correct_l533_533585


namespace triangle_shortest_side_length_l533_533773

noncomputable def shortest_side_length (a b c : ℕ) (h1 : a + b + c = 48) (h2 : a = 21) : ℕ :=
  if h3 : b ≤ c then b else c

theorem triangle_shortest_side_length : 
  ∃ (a b c : ℕ), a = 21 ∧ a + b + c = 48 ∧ b.is_integer ∧ c.is_integer ∧ 
  (shortest_side_length a b c (by sorry) (by sorry)) = 10 :=
begin
  use 21,
  use 10,
  use 17,
  split,
  { refl, },
  split,
  { refl, },
  split,
  { apply int.of_nat_is_integer, },
  split,
  { apply int.of_nat_is_integer, },
  {
    unfold shortest_side_length,
    split_ifs,
    { exact h_1, },
    { exact h_1, },
  }
end

end triangle_shortest_side_length_l533_533773


namespace composite_sum_l533_533436

theorem composite_sum (a b : ℤ) (h : 56 * a = 65 * b) : ∃ m n : ℤ,  m > 1 ∧ n > 1 ∧ a + b = m * n :=
sorry

end composite_sum_l533_533436


namespace _l533_533560

-- Let I be the incenter of △ABC with sides a, b, and c opposite to ∠A, ∠B, and ∠C, respectively.
variable (I A B C : Point)
variable (a b c : ℝ)
variable (IA IB IC : ℝ)
variable [fintype Point]

axiom incenter_of_triangle : (is_incenter I A B C)

def problem_statement : Prop := 
  (IA ^ 2 / (b * c)) + (IB ^ 2 / (a * c)) + (IC ^ 2 / (a * b)) = 1

lemma math_theorem : problem_statement :=
by {
  sorry -- proof goes here
}

end _l533_533560


namespace triangle_cosine_rule_example_l533_533691

theorem triangle_cosine_rule_example 
  (a b : ℝ) (C : ℝ) (ha : a = 2) (hb : b = sqrt 3 - 1) (hC : C = Real.pi / 6) : 
  let c := sqrt (a^2 + b^2 - 2 * a * b * Real.cos (C)) in
  c = sqrt 2 := 
by
  sorry

end triangle_cosine_rule_example_l533_533691


namespace range_of_a_l533_533287

variables {x : ℝ}

def f (x : ℝ) : ℝ := x^2 + log x
def g (x : ℝ) : ℝ := 2*x^2 - a*x

theorem range_of_a (a : ℝ) (h : ∃ x, f x = g (-x)) : a ≤ -1 :=
by
  sorry

end range_of_a_l533_533287


namespace exists_two_points_same_color_l533_533821

theorem exists_two_points_same_color :
  ∀ (x : ℝ), ∀ (color : ℝ × ℝ → Prop),
  (∀ (p : ℝ × ℝ), color p = red ∨ color p = blue) →
  (∃ (p1 p2 : ℝ × ℝ), dist p1 p2 = x ∧ color p1 = color p2) :=
by
  intro x color color_prop
  sorry

end exists_two_points_same_color_l533_533821


namespace aubree_total_animals_l533_533565

noncomputable def total_animals_seen : Nat :=
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10
  morning_total + new_beavers + new_chipmunks

theorem aubree_total_animals :
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10
  morning_total + new_beavers + new_chipmunks = 130 := by
  -- Define all constants and conditions
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10

  -- State the equation
  show morning_total + new_beavers + new_chipmunks = 130 from sorry

end aubree_total_animals_l533_533565


namespace sequence_an_general_formula_sequence_bn_sum_l533_533271

noncomputable def Sn (n : ℕ+) : ℕ := 2^n - 1

noncomputable def an (n : ℕ+) : ℕ := 2^(n-1)

noncomputable def b_n (n : ℕ+) : ℕ := Nat.log 4 (an n) + 1

noncomputable def Tn (n : ℕ+) : ℕ := (n * (1 + (n + 1) / 2)) / 2

theorem sequence_an_general_formula (n : ℕ+) : 
  Sn n = 2^n - 1 → an n = 2^(n-1) := 
sorry

theorem sequence_bn_sum (n : ℕ+) :
  b_n n = Nat.log 4 (an n) + 1 → ∑ i in Finset.range n.succ, b_n i = (n^2 + 3 * n) / 4 :=
sorry

end sequence_an_general_formula_sequence_bn_sum_l533_533271


namespace find_k_l533_533315

theorem find_k 
  (k : ℝ)
  (h : (1/2)^25 * (1/81)^k = 1/(18^25)) :
  k = 12.5 :=
sorry

end find_k_l533_533315


namespace wheat_flour_used_l533_533159

-- Conditions and definitions
def total_flour_used : ℝ := 0.3
def white_flour_used : ℝ := 0.1

-- Statement of the problem
theorem wheat_flour_used : 
  (total_flour_used - white_flour_used) = 0.2 :=
by
  sorry

end wheat_flour_used_l533_533159


namespace proof_complex_log_proof_l533_533384

noncomputable def complex_log_proof : Prop :=
  ∃ (z1 z2 : ℂ), 
    abs z1 = 3 ∧
    abs (z1 + z2) = 3 ∧
    abs (z1 - z2) = 3 * Real.sqrt 3 ∧
    Real.log 3 (abs ((z1 * conj z2)^2000 + (conj z1 * z2)^2000)) = 4000

theorem proof_complex_log_proof : complex_log_proof :=
  sorry

end proof_complex_log_proof_l533_533384


namespace DZ_tangent_to_circle_AXY_l533_533748

-- Definitions of points and relevant settings
variables {A B C O D W X Y Z : Type*}
variables [geometry EuclideanSpace ℝ]  -- Assuming geometry context in a Euclidean space

-- Conditions
variables (h1 : acute_angle A B C)        -- Triangle ABC is acute-angled
variables (h2 : AC > AB)                  -- Side AC is greater than AB
variables (h3 : circumcenter A B C = O)   -- O is the circumcentre of triangle ABC
variables (h4 : segment BC contains D)    -- D lies on segment BC
variables (h5 : perpendicular_to D BC intersects AO at W)  -- D-perpendicular to BC intersects AO at W
variables (h6 : perpendicular_to D BC intersects AC at X)  -- D-perpendicular to BC intersects AC at X
variables (h7 : perpendicular_to D BC intersects AB at Y)  -- D-perpendicular to BC intersects AB at Y
variables (h8 : circumcircle A X Y intersects circumcircle A B C again at Z)  -- Circles intersect at Z ≠ A
variables (h9 : OW = OD)  -- OW is equal to OD

-- Theorem to be proved
theorem DZ_tangent_to_circle_AXY :
  tangent DZ (circumcircle A X Y) :=
sorry -- proof goes here

end DZ_tangent_to_circle_AXY_l533_533748


namespace count_valid_house_numbers_l533_533952

def is_two_digit_prime (n : ℕ) : Prop :=
  n > 9 ∧ n < 100 ∧ Nat.Prime n

def valid_house_number (A B C D : ℕ) : Prop :=
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
  is_two_digit_prime (10 * A + B) ∧
  is_two_digit_prime (10 * C + D) ∧
  (10 * A + B ≠ 10 * C + D)

theorem count_valid_house_numbers : 
  (finset.univ : finset (ℕ × ℕ × ℕ × ℕ)).filter (λ abcd, valid_house_number abcd.1 abcd.2 abcd.3 abcd.4).card = 110 := 
sorry

end count_valid_house_numbers_l533_533952


namespace ellipse_focal_length_l533_533322

theorem ellipse_focal_length (a : ℝ) :
  (∃ x y : ℝ, (x^2 / (10 - a) + y^2 / (a - 2) = 1) ∧ (2 * real.sqrt((abs ((10 - a) - (a - 2))))  = 4))
  → (a = 4 ∨ a = 8) :=
by
  sorry

end ellipse_focal_length_l533_533322


namespace find_y_coordinate_l533_533733

-- Define points A, B, C, and D
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (-2, 2)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (4, 0)

-- Define the property that a point P satisfies PA + PD = PB + PC = 10
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PD := Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2)
  let PB := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  let PC := Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)
  PA + PD = 10 ∧ PB + PC = 10

-- Lean statement to prove the y-coordinate of P that satisfies the condition
theorem find_y_coordinate :
  ∃ (P : ℝ × ℝ), satisfies_condition P ∧ ∃ (a b c d : ℕ), a = 0 ∧ b = 1 ∧ c = 21 ∧ d = 3 ∧ P.2 = (14 + Real.sqrt 21) / 3 ∧ a + b + c + d = 25 :=
by
  sorry

end find_y_coordinate_l533_533733


namespace common_difference_l533_533002

variable (a1 d : ℤ)
variable (S : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

-- Condition: 2 * S 3 = 3 * S 2 + 6
axiom cond : 2 * sum_first_n_terms 3 = 3 * sum_first_n_terms 2 + 6

-- Theorem: The common difference d of the arithmetic sequence is 2
theorem common_difference : d = 2 :=
by
  sorry

end common_difference_l533_533002


namespace euler_lines_concurrent_l533_533730

theorem euler_lines_concurrent 
  {A B C P : Type}
  [euclidean_geometry A B C P]
  (h_interior : interior_point_of_triangle P A B C)
  (h_angles : angle P B C = 120 ∧ angle P C A = 120 ∧ angle P A B = 120) :
  concurrent (euler_line_of_triangle A P B) 
             (euler_line_of_triangle B P C) 
             (euler_line_of_triangle C P A) :=
by sorry

end euler_lines_concurrent_l533_533730


namespace definite_integral_exp_2x_derivative_quotient_x2_sin2x_exp_l533_533168

-- Problem 1
theorem definite_integral_exp_2x :
  ∫ x in 0..1, (Real.exp x + 2 * x) = Real.exp 1 := 
sorry

-- Problem 2
theorem derivative_quotient_x2_sin2x_exp :
  ∀ (x : ℝ), 
  HasDerivAt (λ x, (x^2 + Real.sin (2 * x)) / (Real.exp x))
             ((2 * x + 2 * Real.cos (2 * x) - x^2 - Real.sin (2 * x)) / Real.exp x)
             x :=
sorry

end definite_integral_exp_2x_derivative_quotient_x2_sin2x_exp_l533_533168


namespace bottles_per_person_l533_533762

theorem bottles_per_person (initial_bottles : ℕ) (drank_bottles : ℕ) (friends : ℕ) :
  initial_bottles = 120 →
  drank_bottles = 15 →
  friends = 5 →
  (initial_bottles - drank_bottles) / (friends + 1) = 17 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact Nat.div_eq_of_lt (show 105 < 6 * 18 from by norm_num)

end bottles_per_person_l533_533762


namespace question_I_question_II_l533_533642

theorem question_I (a : ℝ) (a1₀ a2₀ a3₀ : ℝ) (h₁ : a1₀ = a - 1) (h₂ : a2₀ = 4) (h₃ : a3₀ = 2 * a) : 
  a = 3 ∧ ∀ n : ℕ, n > 0 → (a_n n = 2 * n) :=
by 
  sorry

theorem question_II (S_n : ℕ → ℝ) (b_n : ℕ → ℝ) (c_n : ℕ → ℝ) (h₁ : ∀ n, S_n n = n^2 + n) (h₂ : ∀ n, b_n n = S_n n / n) (h₃ : ∀ n, c_n n = 2^(b_n n)) :
  ∀ n, T_n n = 2^(n + 2) - 4 :=
by 
  sorry

end question_I_question_II_l533_533642


namespace find_m_n_inequality_solution_part1_inequality_solution_part2_inequality_solution_part3_l533_533295

-- Definition of the inequality conditions
def inequality_condition (m n : ℝ) : Prop :=
  ∀ x : ℝ, (mx^2 + nx - (1 / m)) < 0 ↔ (x < - (1/2) ∨ x > 2)

-- Theorem statement for the first part of the problem
theorem find_m_n (m n : ℝ) (h : inequality_condition m n) : m = -1 ∧ n = 3 / 2 :=
sorry

-- Theorem statements for the second part of the problem
theorem inequality_solution_part1 (m : ℝ) (a x : ℝ) (h1 : m = -1) (h2 : 2 * a - 1 < 1) :
  (2 * a - 1 - x) * (x + m) > 0 ↔ 2 * a - 1 < x ∧ x < 1 :=
sorry

theorem inequality_solution_part2 (m : ℝ) (a x : ℝ) (h1 : m = -1) (h2 : 2 * a - 1 = 1) :
  (2 * a - 1 - x) * (x + m) > 0 ↔ false :=
sorry

theorem inequality_solution_part3 (m : ℝ) (a x : ℝ) (h1 : m = -1) (h2 : 2 * a - 1 > 1) :
  (2 * a - 1 - x) * (x + m) > 0 ↔ 1 < x ∧ x < 2 * a - 1 :=
sorry

end find_m_n_inequality_solution_part1_inequality_solution_part2_inequality_solution_part3_l533_533295


namespace prove_common_difference_l533_533004

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533004


namespace value_of_a_l533_533683

variable (a : ℝ)

-- Condition: 9x² + 18x + a is the square of a binomial
def is_square_of_binomial (a : ℝ) : Prop :=
  ∃ b : ℝ, ∀ x : ℝ, 9 * x^2 + 18 * x + a = (3 * x + b) ^ 2

-- The proof problem: Prove a = 9 given the condition
theorem value_of_a (h : is_square_of_binomial a) : a = 9 :=
begin
  sorry
end

end value_of_a_l533_533683


namespace error_percent_is_0_point_8_l533_533507

-- Define the variables and conditions
variables (L W : ℝ)
def actual_area := L * W
def measured_length := 1.05 * L
def measured_width := 0.96 * W
def measured_area := measured_length * measured_width
def error := measured_area - actual_area

-- Define the error percent
def error_percent := (error / actual_area) * 100

-- Prove the statement
theorem error_percent_is_0_point_8 :
  error_percent = 0.8 := by
  sorry

end error_percent_is_0_point_8_l533_533507


namespace probability_largest_segment_exceeds_three_times_smallest_l533_533840

theorem probability_largest_segment_exceeds_three_times_smallest :
  let S := {p : ℝ × ℝ | 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ p.1 + p.2 ≤ 1}
  let countable := {p : ℝ × ℝ | 3 * (1 - p.1 - p.2) ≤ p.1}
  (countable.finiteOn S) → 
  (measure.countable.countableOn S).measure countable = (13 / 21) :=
begin
  sorry
end

end probability_largest_segment_exceeds_three_times_smallest_l533_533840


namespace limit_sequence_l533_533574

open Filter
open Real

noncomputable def sequence_limit := 
  filter.tendsto (λ n : ℕ, (sqrt (3 * n - 1) - real.cbrt (125 * n^3 + n)) / (real.rpow n (1 / 5) - n)) at_top (nhds 5)

theorem limit_sequence: sequence_limit :=
  sorry

end limit_sequence_l533_533574


namespace construct_angles_l533_533112

theorem construct_angles (α : ℝ) (h : α = 40) : 
  (2 * α = 80) ∧ (180 - α / 2 = 160) ∧ (α / 2 = 20) :=
by
  -- Ensuring the proof steps to demonstrate the equalities here
  split ; sorry
  split ; sorry
  sorry

end construct_angles_l533_533112


namespace magnitude_five_minus_twelve_i_l533_533235

-- Define the magnitude of a complex number:
def magnitude (a b : ℝ) : ℝ := (a^2 + b^2).sqrt

-- Define the specific complex number components:
def a : ℝ := 5
def b : ℝ := -12

-- State the theorem:
theorem magnitude_five_minus_twelve_i : magnitude a b = 13 :=
by 
  -- Proof goes here
  sorry

end magnitude_five_minus_twelve_i_l533_533235


namespace sum_difference_l533_533132

noncomputable def sum_arith_seq (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_difference :
  let S_even := sum_arith_seq 2 2 1001
  let S_odd := sum_arith_seq 1 2 1002
  S_odd - S_even = 1002 :=
by
  sorry

end sum_difference_l533_533132


namespace downstream_distance_l533_533905

-- Definitions based on identified conditions
def Vs : ℝ := sorry -- speed of the sailor in still water (unknown value)
def Vc : ℝ := sorry -- speed of the current (unknown value)
def V_total := 10 -- combined speed of sailor and current
def time_downstream := 2 / 3 -- time taken to go downstream in hours
def time_upstream := 1 -- time taken to come back upstream in hours

-- Main proof statement
theorem downstream_distance :
  V_total = Vs + Vc →
  (Vs + Vc) * time_downstream = (Vs - Vc) * time_upstream →
  D = 20 / 3 :=
by
  intro hV_total,
  intro hEquality,
  have hVs_Vc := hV_total,
  have hDistanceEq := hEquality,
  sorry -- placeholder for the proof

end downstream_distance_l533_533905


namespace even_multiples_of_5_count_l533_533301

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

def even_multiples_of_5_between (a b : ℕ) : ℕ :=
  Nat.count (λ x => a ≤ x ∧ x ≤ b ∧ is_even x ∧ is_multiple_of_5 x) (List.range (b + 1))

theorem even_multiples_of_5_count : even_multiples_of_5_between 1 99 = 9 := 
  sorry

end even_multiples_of_5_count_l533_533301


namespace total_students_in_school_buses_l533_533545

def bus_seats_total (columns rows broken : Nat) : Nat :=
  (columns * rows) - broken

theorem total_students_in_school_buses :
  let total := 
    (bus_seats_total 4 10 2) + 
    (bus_seats_total 5 8 4) + 
    (bus_seats_total 3 12 3) + 
    (bus_seats_total 4 12 1) + 
    (bus_seats_total 6 8 5) + 
    (bus_seats_total 5 10 2)
  in total = 245 :=
by
  let total := 
    (bus_seats_total 4 10 2) + 
    (bus_seats_total 5 8 4) + 
    (bus_seats_total 3 12 3) + 
    (bus_seats_total 4 12 1) + 
    (bus_seats_total 6 8 5) + 
    (bus_seats_total 5 10 2)
  show total = 245
  sorry

end total_students_in_school_buses_l533_533545


namespace part1_part2_l533_533070

variable {x t : Real}

theorem part1 (h1 : -1 ≤ sin x ∧ sin x ≤ 1) (h2 : -1 ≤ cos x ∧ cos x ≤ 1) : 
  1 / 3 * (4 - Real.sqrt 7) ≤ (2 - sin x) / (2 - cos x) ∧ 
  (2 - sin x) / (2 - cos x) ≤ 1 / 3 * (4 + Real.sqrt 7) := by
  sorry

theorem part2 (h3 : -1 ≤ sin t ∧ sin t ≤ 1) (h4 : -1 ≤ cos t ∧ cos t ≤ 1) :
  1 / 3 * (4 - Real.sqrt 7) ≤ (t ^ 2 + t * sin t + 1) / (t ^ 2 + t * cos t + 1) ∧ 
  (t ^ 2 + t * sin t + 1) / (t ^ 2 + t * cos t + 1) ≤ 1 / 3 * (4 + Real.sqrt 7) := by
  sorry

end part1_part2_l533_533070


namespace last_locker_opened_l533_533197

theorem last_locker_opened : ∀ (n : ℕ), n ∈ {i | 1 ≤ i ∧ i ≤ 2048} →
  (∀ (k : ℕ), k ∈ {i | 1 ≤ i ∧ i ≤ 2048} → is_open_after_passes k (2048)) :=
begin
  sorry
end

end last_locker_opened_l533_533197


namespace lateral_surface_area_of_cone_l533_533811

-- Definitions of the given conditions
def base_radius_cm : ℝ := 3
def slant_height_cm : ℝ := 5

-- The theorem to prove
theorem lateral_surface_area_of_cone :
  let r := base_radius_cm
  let l := slant_height_cm
  π * r * l = 15 * π := 
by
  sorry

end lateral_surface_area_of_cone_l533_533811


namespace range_of_a_l533_533668

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 4 → x^2 - 2 * x + 1 - a^2 < 0) ↔ a > 3 ∨ a < -3 :=
by
  sorry

end range_of_a_l533_533668


namespace chord_length_polar_to_cartesian_l533_533710

theorem chord_length_polar_to_cartesian {ρ θ : ℝ}
  (line_eq : ρ * real.sin θ = 1)
  (circle_eq : ρ = 4 * real.sin θ) :
  2 * real.sqrt 3 = 2 * real.sqrt 3 := 
by 
  -- The proof would show converting polar to Cartesian and finding the intersections, but is omitted here.
  -- sorry is used as the proof is omitted in the problem statement.
  sorry

end chord_length_polar_to_cartesian_l533_533710


namespace ratio_x_y_l533_533690

theorem ratio_x_y (x y : ℝ) (h : x * y = 25) (hy : y = 0.8333333333333334) : x / y ≈ 36 := by
  sorry

end ratio_x_y_l533_533690


namespace total_distance_travelled_l533_533539

def walking_distance_flat_surface (speed_flat : ℝ) (time_flat : ℝ) : ℝ := speed_flat * time_flat
def running_distance_downhill (speed_downhill : ℝ) (time_downhill : ℝ) : ℝ := speed_downhill * time_downhill
def walking_distance_hilly (speed_hilly_walk : ℝ) (time_hilly_walk : ℝ) : ℝ := speed_hilly_walk * time_hilly_walk
def running_distance_hilly (speed_hilly_run : ℝ) (time_hilly_run : ℝ) : ℝ := speed_hilly_run * time_hilly_run

def total_distance (ds1 ds2 ds3 ds4 : ℝ) : ℝ := ds1 + ds2 + ds3 + ds4

theorem total_distance_travelled :
  let speed_flat := 8
  let time_flat := 3
  let speed_downhill := 24
  let time_downhill := 1.5
  let speed_hilly_walk := 6
  let time_hilly_walk := 2
  let speed_hilly_run := 18
  let time_hilly_run := 1
  total_distance (walking_distance_flat_surface speed_flat time_flat) (running_distance_downhill speed_downhill time_downhill)
                            (walking_distance_hilly speed_hilly_walk time_hilly_walk) (running_distance_hilly speed_hilly_run time_hilly_run) = 90 := 
by
  sorry

end total_distance_travelled_l533_533539


namespace determinant_solve_l533_533417

theorem determinant_solve (a b c x : ℝ) :
  det ![
    #[a - x, c, b],
    #[c, b - x, a],
    #[b, a, c - x]
  ] = 0 →
  x = a + b + c ∨
  x = real.sqrt (a^2 + b^2 + c^2 - ab - ac - bc) ∨
  x = -real.sqrt (a^2 + b^2 + c^2 - ab - ac - bc) := 
sorry

end determinant_solve_l533_533417


namespace fencing_required_l533_533191

theorem fencing_required
  (L : ℝ) (W : ℝ) (A : ℝ) (F : ℝ)
  (hL : L = 25)
  (hA : A = 880)
  (hArea : A = L * W)
  (hF : F = L + 2 * W) :
  F = 95.4 :=
by
  sorry

end fencing_required_l533_533191


namespace max_value_special_ratio_l533_533338

theorem max_value_special_ratio (a b c : ℝ) (h1 : a + b + c = 0) (h2 : ¬(a = 0 ∧ b = 0 ∧ c = 0)) : 
  (∃ M : ℝ, M = real.sqrt 2 ∧ ∀ a b c : ℝ, a + b + c = 0 → ¬(a = 0 ∧ b = 0 ∧ c = 0) → abs (a + 2 * b + 3 * c) / real.sqrt (a^2 + b^2 + c^2) ≤ M) := sorry

end max_value_special_ratio_l533_533338


namespace distance_from_center_to_point_l533_533967

theorem distance_from_center_to_point :
  let circle_center := (5, -7)
  let point := (3, -4)
  let distance := Real.sqrt ((3 - 5)^2 + (-4 + 7)^2)
  distance = Real.sqrt 13 := sorry

end distance_from_center_to_point_l533_533967


namespace number_of_rational_terms_is_4_l533_533819

-- Define the expansion term
def binomial_term (r : ℕ) : expr :=
  (nat.choose 6 r) * (-1)^r * 3^(6-r) * x^(6 - (3*r)/2)

-- Define a predicate to check if the term has an integer exponent of x
def integer_exponent (r : ℕ) : Prop :=
  ∃ n : ℤ, 6 - (3*r)/2 = n

-- The theorem to prove the number of rational terms in the expansion
theorem number_of_rational_terms_is_4 :
  {r : ℕ | r ∈ finset.range 7 ∧ integer_exponent r}.card = 4 :=
by
  sorry

end number_of_rational_terms_is_4_l533_533819


namespace find_c_l533_533377

variable {r s b c : ℚ}

-- Conditions based on roots of the original quadratic equation
def roots_of_original_quadratic (r s : ℚ) := 
  (5 * r ^ 2 - 8 * r + 2 = 0) ∧ (5 * s ^ 2 - 8 * s + 2 = 0)

-- New quadratic equation with roots shifted by 3
def new_quadratic_roots (r s b c : ℚ) :=
  (r - 3) + (s - 3) = -b ∧ (r - 3) * (s - 3) = c 

theorem find_c (r s : ℚ) (hb : b = 22/5) : 
  (roots_of_original_quadratic r s) → 
  (new_quadratic_roots r s b c) → 
  c = 23/5 := 
by
  intros h1 h2
  sorry

end find_c_l533_533377


namespace person_speed_l533_533900

theorem person_speed (distance : ℝ) (time_minutes : ℝ) (speed : ℝ)
  (h_distance : distance = 1.8) (h_time : time_minutes = 3) :
  let time_hours := (time_minutes / 60) in
  let calculated_speed := (distance / time_hours) in
  speed = calculated_speed := 
by sorry

end person_speed_l533_533900


namespace count_values_a_l533_533750

theorem count_values_a : 
  { a : ℕ // 0 < a ∧ a < 100 ∧ (a^3 + 23) % 24 = 0 }.card = 5 := 
sorry

end count_values_a_l533_533750


namespace butterfly_distance_l533_533526

noncomputable def omega : ℂ := Complex.exp (-(Real.pi * Complex.I) / 4)
noncomputable def z : ℂ := 1 + 3 * omega + 5 * omega^2 + 7 * omega^3 + 9 * omega^4 + 11 * omega^5 + 13 * omega^6 + 15 * omega^7 + 17 * omega^8

theorem butterfly_distance :
  Complex.abs z = (9 * Real.sqrt(2 + Real.sqrt 2)) / 2 :=
sorry

end butterfly_distance_l533_533526


namespace acute_angles_of_right_triangle_l533_533342

theorem acute_angles_of_right_triangle (x : ℝ) (hx_pos : x > 0) 
  (ratio_condition : 5 * x + 4 * x = 90) :
  (5 * x = 50) ∧ (4 * x = 40) :=
by
  have h1 : 9 * x = 90, by rwa [add_comm, ← add_assoc],
  have h2 : x = 10, from (eq_div_iff (by norm_num : 9 ≠ 0)).mp h1,
  have h3 : 5 * 10 = 50, by norm_num,
  have h4 : 4 * 10 = 40, by norm_num,
  exact ⟨h3, h4⟩

end acute_angles_of_right_triangle_l533_533342


namespace concave_quad_perimeter_less_than_rectangle_l533_533460

universe u

-- Definitions representing the geometric objects
structure Point := (x : ℝ) (y : ℝ)
structure Rectangle := (A B C D : Point)

-- Function that measures the perimeter of a quadrilateral
def perimeter (p1 p2 p3 p4 : Point) : ℝ :=
  dist p1 p2 + dist p2 p3 + dist p3 p4 + dist p4 p1

-- The theorem statement 
theorem concave_quad_perimeter_less_than_rectangle
  (A B C D X Y Z : Point)
  (hA : A ∈ rectangle (A B C D))
  (hX : X ∈ rectangle (A B C D))
  (hY : Y ∈ rectangle (A B C D))
  (hZ : Z ∈ triangle (A X Y))
: ∃ (Q1 Q2 Q3 : Point), 
    perimeter A X Z Y < perimeter A B C D ∨ 
    perimeter A Y Z X < perimeter A B C D ∨ 
    perimeter A Z X Y < perimeter A B C D :=
sorry

end concave_quad_perimeter_less_than_rectangle_l533_533460


namespace arithmetic_seq_max_term_is_S9_a9_l533_533640

noncomputable def arithmetic_seq_max_term (a : ℕ → ℝ) (S : ℕ → ℝ) : ℝ :=
  have h1 : S 17 > 0 := sorry,
  have h2 : S 18 < 0 := sorry,
  let maximum_term := max_seq_term (λ n, S (n + 1) / a (n + 1)) (fin_seq 17)
  in (S 9 / a 9)

def max_seq_term {α : Type*} [linear_order α] (f : ℕ → α) (n : ℕ) :=
  fin.max seq.iterate n f

def fin_seq (n : ℕ) : ℕ → ℕ 
  | i => if i < n then i else 0

theorem arithmetic_seq_max_term_is_S9_a9 (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 17 > 0) (h2 : S 18 < 0) :
  arithmetic_seq_max_term a S = (S 9 / a 9) := 
sorry

end arithmetic_seq_max_term_is_S9_a9_l533_533640


namespace savings_calculation_l533_533435

-- Definitions based on conditions
def ratio_income_expenditure : ℚ := 5 / 4
def income : ℕ := 16000

-- Main statement
theorem savings_calculation (ratio_income_expenditure: ℚ) (income: ℕ) : ℕ :=
  let expenditure := ((4/5 : ℚ) * income.toRat).toInt; 
  let savings := income - expenditure;
  savings = 3200 :=
by
  -- the actual proof would follow here
  sorry

end savings_calculation_l533_533435


namespace lulu_cash_left_l533_533393

theorem lulu_cash_left :
  ∀ (initial money spentIceCream spentTshirt deposited finalCash: ℝ),
    initial = 65 →
    spentIceCream = 5 →
    spentTshirt = 0.5 * (initial - spentIceCream) →
    deposited = (initial - spentIceCream - spentTshirt) / 5 →
    finalCash = initial - spentIceCream - spentTshirt - deposited →
    finalCash = 24 :=
by
  intros
  sorry

end lulu_cash_left_l533_533393


namespace equal_chord_lengths_l533_533990

theorem equal_chord_lengths (t : ℝ) :
  (∀ x y : ℝ, (x - t)^2 + (y + t - 1)^2 = 8 
  → (sqrt (8 - (t - 1)^2) = sqrt (8 - t^2))) → t = 1 / 2 :=
by
  -- Proof implementation will go here 
  -- Placeholder for proof
  sorry

end equal_chord_lengths_l533_533990


namespace jacob_has_more_money_l533_533764

def exchange_rate : ℝ := 1.11
def Mrs_Hilt_total_in_dollars : ℝ := 
  3 * 0.01 + 2 * 0.10 + 2 * 0.05 + 5 * 0.25 + 1 * 1.00

def Jacob_total_in_euros : ℝ := 
  4 * 0.01 + 1 * 0.05 + 1 * 0.10 + 3 * 0.20 + 2 * 0.50 + 2 * 1.00

def Jacob_total_in_dollars : ℝ := Jacob_total_in_euros * exchange_rate

def difference : ℝ := Jacob_total_in_dollars - Mrs_Hilt_total_in_dollars

theorem jacob_has_more_money : difference = 1.63 :=
by sorry

end jacob_has_more_money_l533_533764


namespace prove_common_difference_l533_533015

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533015


namespace change_digit_correct_sum_l533_533798

theorem change_digit_correct_sum :
  ∃ d e, 
  d = 2 ∧ e = 8 ∧ 
  653479 + 938521 ≠ 1616200 ∧
  (658479 + 938581 = 1616200) ∧ 
  d + e = 10 := 
by {
  -- our proof goes here
  sorry
}

end change_digit_correct_sum_l533_533798


namespace non_overlapping_segments_total_length_l533_533410

open Set

theorem non_overlapping_segments_total_length {n : ℕ} (a b : vector ℝ (n+1)) (h : ∀ i, 0 ≤ a i ∧ a i ≤ b i ∧ b i ≤ 1) :
(∃ s : finset ℕ, (∀ i j ∈ s, i ≠ j → disjoint (i.size) (j.size)) ∧ ∑ i in s, (b i - a i) ≥ 1 / 2) :=
begin
  sorry
end

end non_overlapping_segments_total_length_l533_533410


namespace eccentricity_of_ellipse_l533_533955

variables (a b c : ℝ)
def ellipse (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def eccentricity (a c : ℝ) := c / a

theorem eccentricity_of_ellipse {a b c : ℝ} (ha : a > 0) (hb : b > 0) (h_ellipse : ellipse a b)
  (h_angle : ∀ x y : ℝ, 
     let P := (-c, b^2 / a) in
     let F1 := (-c, 0) in
     let F2 := (c, 0) in
     (x, y) = P ∧ ∠ F1 P F2 = 60) :
  eccentricity a c = sqrt 3 / 3 := 
sorry

end eccentricity_of_ellipse_l533_533955


namespace average_pages_per_book_deshaun_l533_533946

-- Definitions related to the conditions
def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def person_closest_percentage : ℚ := 0.75
def second_person_daily_pages : ℕ := 180

-- Derived definitions
def second_person_total_pages : ℕ := second_person_daily_pages * summer_days
def deshaun_total_pages : ℚ := second_person_total_pages / person_closest_percentage

-- The final proof statement
theorem average_pages_per_book_deshaun : 
  deshaun_total_pages / deshaun_books = 320 := 
by
  -- We would provide the proof here
  sorry

end average_pages_per_book_deshaun_l533_533946


namespace pairs_satisfy_equation_l533_533240

theorem pairs_satisfy_equation :
  ∀ (x n : ℕ), (x > 0 ∧ n > 0) ∧ 3 * 2 ^ x + 4 = n ^ 2 → (x, n) = (2, 4) ∨ (x, n) = (5, 10) ∨ (x, n) = (6, 14) :=
by
  sorry

end pairs_satisfy_equation_l533_533240


namespace sum_of_all_positive_integer_solutions_l533_533485

theorem sum_of_all_positive_integer_solutions :
  (∑ x in {x | 7 * (5 * x - 3) % 10 = 4 ∧ x ≤ 30 ∧ 0 < x}.toFinset) = 225 :=
by
  sorry

end sum_of_all_positive_integer_solutions_l533_533485


namespace cylinder_volume_l533_533447

theorem cylinder_volume (r : ℝ) (h : ℝ) (A : ℝ) (V : ℝ) 
  (sphere_surface_area : A = 256 * Real.pi)
  (cylinder_height : h = 2 * r) 
  (sphere_surface_formula : A = 4 * Real.pi * r^2) 
  (cylinder_volume_formula : V = Real.pi * r^2 * h) : V = 1024 * Real.pi := 
by
  -- Definitions provided as conditions
  sorry

end cylinder_volume_l533_533447


namespace shekar_average_is_81_9_l533_533784

def shekar_average_marks (marks : List ℕ) : ℚ :=
  (marks.sum : ℚ) / marks.length

theorem shekar_average_is_81_9 :
  shekar_average_marks [92, 78, 85, 67, 89, 74, 81, 95, 70, 88] = 81.9 :=
by
  sorry

end shekar_average_is_81_9_l533_533784


namespace sum_of_possible_values_l533_533440

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 4) = -21) :
    ∃ N₁ N₂ : ℝ, (N₁ + N₂ = 4 ∧ N₁ * (N₁ - 4) = -21 ∧ N₂ * (N₂ - 4) = -21) :=
sorry

end sum_of_possible_values_l533_533440


namespace problem1_problem2_l533_533627

-- Given function f with parameter k
def f (x : ℝ) (k : ℝ) : ℝ := Real.log (4^x + 1) / Real.log 2 - k * x

-- First question: If f is even, then k = 1
theorem problem1 (k : ℝ) (h : ∀ x : ℝ, f x k = f (-x) k) : k = 1 := 
sorry

-- Given function g with parameter a
def g (x : ℝ) (a : ℝ) : ℝ := f x 2 - a + 1

-- Second question: If g has a zero, then a > 1
theorem problem2 (a : ℝ) (h : ∃ x : ℝ, g x a = 0) : 1 < a :=
sorry

end problem1_problem2_l533_533627


namespace min_m_value_for_sequences_l533_533737

theorem min_m_value_for_sequences :
  ∀ (a b : ℕ → ℕ) (S : ℕ → ℕ),
    (∀ d, a 1 + a 2 = 4) →
    (a 3 = 5) →
    (∀ n, a n = a 1 + (n - 1) * d) →
    (S n = n * a 1 + n * (n - 1) / 2 * d) →
    (b 1 = 2) →
    (∀ q, b 3 = a 4 + 1) →
    (∀ n, b n = b 1 * q ^ (n - 1)) →
    (∀ n, S n ≤ b n) →
    ∃ m, m = 4 :=
by
  intros a b S h1 h2 h3 h4 h5 h6 h7 h8
  existsi 4
  sorry

end min_m_value_for_sequences_l533_533737


namespace complex_sum_correct_l533_533680

-- Definitions based on conditions
def B : Complex := 3 + 2 * Complex.i
def Q : Complex := -5
def R : Complex := 0 + 3 * Complex.i
def T : Complex := 1 + 5 * Complex.i

-- Theorem statement
theorem complex_sum_correct : B - Q + R + T = -1 + 10 * Complex.i := by
  sorry

end complex_sum_correct_l533_533680


namespace correct_operation_l533_533155

theorem correct_operation : 
  ¬(2 + sqrt 2 = 2 * sqrt 2) ∧
  ¬(4 * x^2 * y - x^2 * y = 3) ∧
  ¬((a + b)^2 = a^2 + b^2) ∧
  ((ab)^3 = a^3 * b^3) :=
by sorry

end correct_operation_l533_533155


namespace males_listen_l533_533550

theorem males_listen (total_listen : ℕ) (females_listen : ℕ) (known_total_listen : total_listen = 160)
  (known_females_listen : females_listen = 75) : (total_listen - females_listen) = 85 :=
by 
  sorry

end males_listen_l533_533550


namespace determine_quantitative_description_l533_533194

variables
  (mass_solute : ℝ) -- mass of solute
  (mass_solvent : ℝ) -- mass of solvent
  (molar_mass_solute : ℝ) -- molar mass of solute
  (molar_mass_solvent : ℝ) -- molar mass of solvent

theorem determine_quantitative_description : 
  (∃(m : ℝ), m = mass_solute / (molar_mass_solute * mass_solvent)) ∧
  ¬(∃(M : ℝ), M = mass_solute / (molar_mass_solute * ?volume_solution)) ∧
  ¬(∃(ρ : ℝ), ρ = (mass_solute + mass_solvent) / ?volume_solution) :=
by
  sorry

end determine_quantitative_description_l533_533194


namespace sum_of_BCs_in_ABC_l533_533720

noncomputable def law_of_cosines := sorry -- Define and prove this if necessary

theorem sum_of_BCs_in_ABC (B C : ℝ) (B_pos : 0 < B) (B_lt_180 : B < 180)
  (B_eq_45 : B = 45) (A B C_sum_180 : B + C + A = 180)
  (AB AC : ℝ) (AB_eq_80 : AB = 80) (AC_eq_80_sqrt_2 : AC = 80 * Real.sqrt 2) :
  ∃ BC : ℝ, 
    (A = 180 - B - C ∧ C = 30) → 
      BC = Real.sqrt (80^2 + (80 * Real.sqrt 2)^2 - 2 * 80 * (80 * Real.sqrt 2) * Real.cos 105) ∧
    (∃ sum_BC : ℝ, sum_BC = 147.42) :=
sorry

end sum_of_BCs_in_ABC_l533_533720


namespace sum_series_eq_three_l533_533940

theorem sum_series_eq_three :
  (∑' k : ℕ, (9^k) / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 3 :=
by 
  sorry

end sum_series_eq_three_l533_533940


namespace complex_number_z_eq_zero_l533_533975

theorem complex_number_z_eq_zero (i : ℂ) (h : i^2 = -1) : 
  let z := i + i^2 + i^3 + i^4
  in z = 0 :=
by
  sorry

end complex_number_z_eq_zero_l533_533975


namespace perfect_square_proof_l533_533466

noncomputable def isPerfectSquare (f : ℚ[X]) : Prop :=
  ∃ g : ℚ[X], f = g^2

variable {a a' b b' c c' : ℚ}

def condition1 : Prop := isPerfectSquare ((a + b * X)^2 + (a' + b' * X)^2)
def condition2 : Prop := isPerfectSquare ((a + c * X)^2 + (a' + c' * X)^2)

theorem perfect_square_proof (h1 : condition1) (h2 : condition2) :
  isPerfectSquare ((b + c * X)^2 + (b' + c' * X)^2) := 
sorry

end perfect_square_proof_l533_533466


namespace sum_of_rel_prime_prob_l533_533901

noncomputable def factorial_prime_factors : ℕ := 10!

def prime_factorization_of_ten_factorial := (2^8 * 3^4 * 5^2 * 7^1 : ℕ)
def perfect_square_prob_m := 1
def perfect_square_prob_n := 9

theorem sum_of_rel_prime_prob : perfect_square_prob_m + perfect_square_prob_n = 10 := 
by sorry

end sum_of_rel_prime_prob_l533_533901


namespace prove_common_difference_l533_533013

variable {a1 d : ℤ}
variable (S : ℕ → ℤ)
variable (arith_seq : ℕ → ℤ)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the condition given in the problem
def problem_condition : Prop := 2 * sum_first_n 3 = 3 * sum_first_n 2 + 6

-- Define that the common difference d is 2
def d_value_is_2 : Prop := d = 2

theorem prove_common_difference (h : problem_condition) : d_value_is_2 :=
by
  sorry

end prove_common_difference_l533_533013


namespace maximum_value_l533_533034

def expression (A B C : ℕ) : ℕ := A * B * C + A * B + B * C + C * A

theorem maximum_value (A B C : ℕ) 
  (h1 : A + B + C = 15) : 
  expression A B C ≤ 200 :=
sorry

end maximum_value_l533_533034


namespace max_pies_without_any_ingredients_l533_533916

theorem max_pies_without_any_ingredients (total_pies : ℕ) (nuts_fraction berries_fraction cream_fraction choco_fraction : ℚ) 
  (h_total_pies : total_pies = 48)
  (h_nuts_fraction : nuts_fraction = 1/3)
  (h_berries_fraction : berries_fraction = 1/2)
  (h_cream_fraction : cream_fraction = 3/5)
  (h_choco_fraction : choco_fraction = 1/4) :
  let nuts_pies := (nuts_fraction * total_pies).ceil.to_nat,
      berries_pies := (berries_fraction * total_pies).ceil.to_nat,
      cream_pies := (cream_fraction * total_pies).ceil.to_nat,
      choco_pies := (choco_fraction * total_pies).ceil.to_nat,
      max_pies_without_cream := total_pies - cream_pies
  in max_pies_without_cream = 19 :=
by 
  have h1 : total_pies = 48 := h_total_pies
  have h2 : cream_fraction = 3/5 := h_cream_fraction
  have h_cream_pies : cream_pies = (3/5 * 48).ceil.to_nat := sorry 
  have h_cream_pies_rounded : cream_pies = 29 := sorry
  have h_max_pies_without_cream : max_pies_without_cream = 48 - 29 := sorry
  exact (h_total_pies ▸ h_cream_pies_rounded ▸ rfl)

end max_pies_without_any_ingredients_l533_533916


namespace mike_found_2_unbroken_seashells_l533_533054

variable (total_seashells : ℕ)
variable (broken_seashells : ℕ)

def found_unbroken_seashells := total_seashells - broken_seashells

theorem mike_found_2_unbroken_seashells
  (h1 : total_seashells = 6)
  (h2 : broken_seashells = 4) :
  found_unbroken_seashells total_seashells broken_seashells = 2 :=
by {
    rw [h1, h2],
    exact Nat.sub_eq_of_eq_add (refl 6),
    sorry
}

end mike_found_2_unbroken_seashells_l533_533054


namespace correct_statement_D_l533_533852

def is_correct_option (n : ℕ) := n = 4

theorem correct_statement_D : is_correct_option 4 :=
  sorry

end correct_statement_D_l533_533852


namespace students_who_wore_blue_lipstick_l533_533057

theorem students_who_wore_blue_lipstick (total_students : ℕ) (h1 : total_students = 200) : 
  ∃ blue_lipstick_students : ℕ, blue_lipstick_students = 5 :=
by
  have colored_lipstick_students := total_students / 2
  have red_lipstick_students := colored_lipstick_students / 4
  let blue_lipstick_students := red_lipstick_students / 5
  have h2 : blue_lipstick_students = 5 :=
    calc blue_lipstick_students
          = (total_students / 2) / 4 / 5 : by sorry -- detailed calculation steps omitted
  use blue_lipstick_students
  exact h2

end students_who_wore_blue_lipstick_l533_533057


namespace work_completion_days_l533_533176

theorem work_completion_days (D : ℕ) 
  (h : 40 * D = 48 * (D - 10)) : D = 60 := 
sorry

end work_completion_days_l533_533176


namespace function_graph_passes_through_point_l533_533991

variable {α β : Type} (f : α → β) (f_inv : β → α)

def has_inverse (f : α → β) (f_inv : β → α) : Prop :=
  ∀ x : α, f_inv (f x) = x ∧ ∀ y : β, f (f_inv y) = y

theorem function_graph_passes_through_point (h_inv : has_inverse f f_inv) (h_point : f_inv 1 = 2) : f 2 = 1 :=
  by
  sorry

end function_graph_passes_through_point_l533_533991


namespace needed_packs_for_project_l533_533525

theorem needed_packs_for_project (boards_total screws_total brackets_total : ℕ)
    (boards_per_pack screws_per_pack brackets_per_pack : ℕ)
    (H_boards : boards_total = 154) 
    (H_screws : screws_total = 78) 
    (H_brackets : brackets_total = 42) 
    (H_boards_pack : boards_per_pack = 3) 
    (H_screws_pack : screws_per_pack = 20) 
    (H_brackets_pack : brackets_per_pack = 5) :
    let packs_boards := Nat.ceil (154 / 3 : ℚ)
    let packs_screws := Nat.ceil (78 / 20 : ℚ)
    let packs_brackets := Nat.ceil (42 / 5 : ℚ) in
    packs_boards = 52 ∧ packs_screws = 4 ∧ packs_brackets = 9 := by
    simp only [H_boards, H_screws, H_brackets, H_boards_pack, H_screws_pack, H_brackets_pack]
    have pboards : Nat.ceil (154 / 3 : ℚ) = 52 := by
        norm_num
    have pscrews : Nat.ceil (78 / 20 : ℚ) = 4 := by
        norm_num
    have pbrackets : Nat.ceil (42 / 5 : ℚ) = 9 := by
        norm_num
    exact ⟨pboards, pscrews, pbrackets⟩

end needed_packs_for_project_l533_533525


namespace num_branches_eq_catalan_l533_533858

def binom (n k : ℕ) : ℕ := Nat.choose n k

def catalan (n : ℕ) : ℕ :=
  binom (2*n) n - binom (2*n) (n+1)

theorem num_branches_eq_catalan (n : ℕ) : 
  ∃ (num_branches : ℕ), 
    num_branches = binom (2*(n + 1)) (n + 1) - binom (2*(n + 1)) n :=
sorry

end num_branches_eq_catalan_l533_533858


namespace ways_to_change_12_dollars_into_nickels_and_quarters_l533_533033

theorem ways_to_change_12_dollars_into_nickels_and_quarters :
  ∃ n q : ℕ, 5 * n + 25 * q = 1200 ∧ n > 0 ∧ q > 0 ∧ ∀ q', (q' ≥ 1 ∧ q' ≤ 47) ↔ (n = 240 - 5 * q') :=
by
  sorry

end ways_to_change_12_dollars_into_nickels_and_quarters_l533_533033


namespace arrangement_of_people_l533_533456

noncomputable def total_unique_arrangements : ℕ := 28

theorem arrangement_of_people :  
  ∃ (arrangements : ℕ), arrangements = total_unique_arrangements ∧
  (∀ (R Y B : Type) (people : List (Sum R (Sum Y B))),
    people.length = 5 ∧
    (∃ R_count Y_count B_count : ℕ, R_count = 2 ∧ Y_count = 2 ∧ B_count = 1 ∧ 
    (∀ i, (people.nth i ≠ people.nth (i + 1)))) →
    ∃ (arrangings : ℕ), arrangings = arrangements) :=
by
  use 28
  split
  case left => refl
  case right => sorry

end arrangement_of_people_l533_533456


namespace translation_result_l533_533292

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

def g (x : ℝ) : ℝ := -Real.cos (2 * x)

theorem translation_result :
    (f (x - Real.pi / 6) = g x) :=
sorry

end translation_result_l533_533292


namespace sum_of_positive_integer_solutions_l533_533488

theorem sum_of_positive_integer_solutions (n : ℕ) :
  (sum (filter (λ x : ℕ, x > 0 ∧ x ≤ 30 ∧ 7 * (5 * x - 3) % 10 = 14 % 10) (finset.range (30 + 1)))) = 225 :=
by
  sorry

end sum_of_positive_integer_solutions_l533_533488


namespace john_rejection_percentage_l533_533370

-- Define the conditions
def total_products (P : ℝ) := P > 0
def inspected_by_jane (P : ℝ) := (5 / 6) * P
def inspected_by_john (P : ℝ) := (1 / 6) * P
def rejected_by_jane (P : ℝ) := (0.008 * 5 / 6) * P
def total_rejected (P : ℝ) := 0.0075 * P

-- Prove that John rejected 0.5% of products given the conditions
theorem john_rejection_percentage (P : ℝ) (J : ℝ)
  (h_total : total_products P)
  (h_jane_inspect : inspected_by_jane P = (5 / 6) * P)
  (h_john_inspect : inspected_by_john P = (1 / 6) * P)
  (h_jane_reject : rejected_by_jane P = (0.008 * 5 / 6) * P)
  (h_total_reject : total_rejected P = 0.0075 * P) :
  J = 0.5 :=
by {
  sorry
}

end john_rejection_percentage_l533_533370


namespace preferred_apples_percentage_l533_533429

theorem preferred_apples_percentage (A B C O G : ℕ) (total freq_apples : ℕ)
  (hA : A = 70) (hB : B = 50) (hC: C = 30) (hO: O = 50) (hG: G = 40)
  (htotal : total = A + B + C + O + G)
  (hfa : freq_apples = A) :
  (freq_apples / total : ℚ) * 100 = 29 :=
by sorry

end preferred_apples_percentage_l533_533429


namespace product_sequence_l533_533959

theorem product_sequence : 
  let seq := [1/3, 9/1, 1/27, 81/1, 1/243, 729/1, 1/2187, 6561/1, 1/19683, 59049/1]
  ((seq[0] * seq[1]) * (seq[2] * seq[3]) * (seq[4] * seq[5]) * (seq[6] * seq[7]) * (seq[8] * seq[9])) = 243 :=
by
  sorry

end product_sequence_l533_533959


namespace count_triangles_in_figure_l533_533943

-- Define the problem as a Lean statement
theorem count_triangles_in_figure :
  ∃ (figure : Type) (ABCD : figure) (diagonals : figure) (MNPQ : figure),
  let num_triangles := -- The total number of triangles formed by the described configuration
    36 in
  num_triangles = 36 :=
begin
  -- Definitions according to the problem
  -- larger square vertices A, B, C, D form square ABCD
  -- diagonals AC, BD
  -- smaller square connecting midpoints of ABCD form square MNPQ
  -- total triangle count based on conditions = 36
  
  -- This is just a placeholder to indicate the configuration
  -- and expected number justification through sorry
  sorry,
end

end count_triangles_in_figure_l533_533943


namespace unique_point_proof_l533_533299

-- We state the problem using Lean constructs
theorem unique_point_proof (A1 A2 B1 B2 : ℝ) (l : set ℝ)
  (hA1 : A1 ∈ l) (hA2 : A2 ∈ l) (hB1 : B1 ∈ l) (hB2 : B2 ∈ l) :
  ∃! P ∈ l, (abs (P - A1) * abs (P - A2) = abs (P - B1) * abs (P - B2)) :=
sorry

end unique_point_proof_l533_533299


namespace trajectory_of_M_l533_533804

theorem trajectory_of_M (M : ℝ × ℝ) (h : (M.2 < 0 → M.1 = 0) ∧ (M.2 ≥ 0 → M.1 ^ 2 = 8 * M.2)) :
  (M.2 < 0 → M.1 = 0) ∧ (M.2 ≥ 0 → M.1 ^ 2 = 8 * M.2) :=
by
  sorry

end trajectory_of_M_l533_533804


namespace min_N_to_determine_PIN_l533_533876

theorem min_N_to_determine_PIN (num_clients : ℕ) (pin_length : ℕ) 
    (total_combinations : ℕ := 10 ^ pin_length)
    (clients : fin num_clients → fin total_combinations)
    (unique_PINs : ∀ i j : fin num_clients, i ≠ j → clients i ≠ clients j) :
    ∃ (N : ℕ), N = 3 := 
by {
    have num_clients := 1000000,   -- one million clients
    have pin_length := 6,          -- six-digit PIN code
    have total_combinations := 10 ^ 6, -- 1,000,000 possible combinations
    exact ⟨3, rfl⟩                  -- the solution shows N = 3 is correct
}

end min_N_to_determine_PIN_l533_533876


namespace find_15th_number_in_sequence_l533_533399

theorem find_15th_number_in_sequence : 
    (∀ n : ℕ, (n % 4 = 1 → a n = (-3) ^ ((n + 2) / 4)) ∧ 
                  (n % 4 = 3 → a n = 3 ^ (2 * ((n - 3) / 4 + 1))) ∧ 
                  ((n % 4 ≠ 1 ∧ n % 4 ≠ 3) → a n = 1))
  → a 15 = 3 ^ 8 := 
  sorry

end find_15th_number_in_sequence_l533_533399


namespace untouched_shapes_after_moves_l533_533862

-- Definitions
def num_shapes : ℕ := 12
def num_triangles : ℕ := 3
def num_squares : ℕ := 4
def num_pentagons : ℕ := 5
def total_moves : ℕ := 10
def petya_moves_first : Prop := True
def vasya_strategy : Prop := True  -- Vasya's strategy to minimize untouched shapes
def petya_strategy : Prop := True  -- Petya's strategy to maximize untouched shapes

-- Theorem
theorem untouched_shapes_after_moves : num_shapes = 12 ∧ num_triangles = 3 ∧ num_squares = 4 ∧ num_pentagons = 5 ∧
                                        total_moves = 10 ∧ petya_moves_first ∧ vasya_strategy ∧ petya_strategy → 
                                        num_shapes - 5 = 6 :=
by
  sorry

end untouched_shapes_after_moves_l533_533862


namespace no_integer_roots_l533_533634

def cubic_polynomial (a b c d x : ℤ) : ℤ :=
  a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots (a b c d : ℤ) (h1 : cubic_polynomial a b c d 1 = 2015) (h2 : cubic_polynomial a b c d 2 = 2017) :
  ∀ x : ℤ, cubic_polynomial a b c d x ≠ 2016 :=
by
  sorry

end no_integer_roots_l533_533634


namespace gcd_282_470_l533_533865

theorem gcd_282_470 : Int.gcd 282 470 = 94 :=
by
  sorry

end gcd_282_470_l533_533865


namespace arithmetic_common_difference_l533_533018

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l533_533018


namespace no_real_solutions_l533_533593

/-- Proof problem: Prove that there are no real solutions to the equation (2x - 6)^2 + 4 = -2 * |x| -/
theorem no_real_solutions : ∀ x : ℝ, ¬ ((2 * x - 6) ^ 2 + 4 = -2 * |x|) :=
by
  intro x
  -- introduce that (2x - 6)^2 is non-negative and 4 is positive
  have h1 : (2 * x - 6) ^ 2 ≥ 0 := pow_two_nonneg (2 * x - 6)
  -- introduce the positivity of the left-hand side
  have h2 : (2 * x - 6) ^ 2 + 4 > 0 := by linarith
  -- introduce the non-positivity of the right-hand side
  have h3 : -2 * |x| ≤ 0 := by nlinarith
  -- contradiction since a positive number cannot equal a non-positive number
  sorry

end no_real_solutions_l533_533593


namespace total_profit_is_120_l533_533895

-- Defining the conditions
def A_investment_amount : ℝ := 400
def A_investment_duration : ℝ := 12
def B_investment_amount : ℝ := 200
def B_investment_duration : ℝ := 6
def A_share_profit : ℝ := 80

-- Defining the total money-months investments
def A_money_months := A_investment_amount * A_investment_duration
def B_money_months := B_investment_amount * B_investment_duration

-- Defining the total profit
noncomputable def P := 120

-- Stating the theorem to prove the total profit
theorem total_profit_is_120 : 
  let total_investment := A_money_months + B_money_months,
      A_ratio := A_money_months / total_investment in
  A_share_profit = A_ratio * P → P = 120 :=
by sorry

end total_profit_is_120_l533_533895


namespace find_n_l533_533040

noncomputable def a (n : ℕ) : ℝ :=
1 / (Real.sqrt (n + 1) + Real.sqrt (n + 2))

noncomputable def S (n : ℕ) : ℝ :=
∑ i in Finset.range (n + 1), a i

theorem find_n :
  (S 30 = 3 * Real.sqrt 2) → (∑ i in Finset.range (30 + 1), a i = 3 * Real.sqrt 2) :=
by sorry

end find_n_l533_533040


namespace triangle_area_proof_l533_533246

-- Define the conditions
def side_a : ℝ := 7
def side_b : ℝ := 11
def angle_C : ℝ := 60 -- in degrees

-- Define the conversion from degrees to radians
def degrees_to_radians (d : ℝ) : ℝ := d * (Real.pi / 180)

-- Define the sine of 60 degrees
def sin_60 : ℝ := Real.sin (degrees_to_radians angle_C)

-- Define the area formula for two sides and included angle
def triangle_area (a b : ℝ) (C_degrees : ℝ) : ℝ :=
  (1 / 2) * a * b * Real.sin (degrees_to_radians C_degrees)

-- State the proof problem
theorem triangle_area_proof :
  triangle_area side_a side_b angle_C = (77 * Real.sqrt 3 / 4) :=
by
  sorry

end triangle_area_proof_l533_533246


namespace Cornelia_three_times_Kilee_l533_533708

variable (x : ℕ)

def Kilee_current_age : ℕ := 20
def Cornelia_current_age : ℕ := 80

theorem Cornelia_three_times_Kilee (x : ℕ) :
  Cornelia_current_age + x = 3 * (Kilee_current_age + x) ↔ x = 10 :=
by
  sorry

end Cornelia_three_times_Kilee_l533_533708


namespace worker_weekly_pay_l533_533198

theorem worker_weekly_pay : 
  let regular_rate := 30
  let total_surveys := 100
  let cellphone_surveys := 50
  let regular_surveys := total_surveys - cellphone_surveys
  let cellphone_rate := regular_rate + (20 * regular_rate / 100)
  let regular_pay := regular_surveys * regular_rate
  let cellphone_pay := cellphone_surveys * cellphone_rate
in regular_pay + cellphone_pay = 3300 :=
by 
  let regular_rate := 30
  let total_surveys := 100
  let cellphone_surveys := 50
  let regular_surveys := total_surveys - cellphone_surveys
  let cellphone_rate := regular_rate + (20 * regular_rate / 100)
  let regular_pay := regular_surveys * regular_rate
  let cellphone_pay := cellphone_surveys * cellphone_rate
  have h1 : regular_surveys = 50 := by sorry
  have h2 : cellphone_rate = 36 := by sorry
  have h3 : regular_pay = 1500 := by sorry
  have h4 : cellphone_pay = 1800 := by sorry
  have h5 : regular_pay + cellphone_pay = 3300 := by sorry
  exact h5 

end worker_weekly_pay_l533_533198


namespace part1_part2_part3_l533_533408

def folklore {a b m n : ℤ} (h1 : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : Prop :=
  a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n

theorem part1 (a b m n : ℤ) (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n :=
by sorry

theorem part2 : 13 + 4 * Real.sqrt 3 = (1 + 2 * Real.sqrt 3) ^ 2 :=
by sorry

theorem part3 (a m n : ℤ) (h1 : 4 = 2 * m * n) (h2 : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : a = 7 ∨ a = 13 :=
by sorry

end part1_part2_part3_l533_533408


namespace trigonometric_identity_l533_533258

theorem trigonometric_identity 
  (α : Real)
  (h : (sin (11 * Real.pi - α) - cos (-α)) / cos (7 * Real.pi / 2 + α) = 3) :
  tan α = -1/2 ∧ sin (2 * α) + cos (2 * α) = -1/5 :=
by
  sorry

end trigonometric_identity_l533_533258


namespace sum_sin_squared_angles_l533_533942

theorem sum_sin_squared_angles :
  (∑ k in finset.range 36, real.sin (5 * (k + 1) * real.pi / 180) ^ 2) = 18.5 :=
begin
  sorry
end

end sum_sin_squared_angles_l533_533942


namespace width_of_shop_l533_533817

theorem width_of_shop 
  (monthly_rent : ℕ) 
  (length : ℕ) 
  (annual_rent_per_sqft : ℕ) 
  (h1 : monthly_rent = 3600) 
  (h2 : length = 18) 
  (h3 : annual_rent_per_sqft = 120) :
  ∃ width : ℕ, width = 20 :=
by
  sorry

end width_of_shop_l533_533817


namespace NineChaptersProblem_l533_533350

-- Conditions: Assign the given conditions to variables
variables (x y : Int)
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Proof problem: Prove that the system of equations is consistent with the given conditions
theorem NineChaptersProblem : condition1 x y ∧ condition2 x y := sorry

end NineChaptersProblem_l533_533350


namespace logM_NM_eq_one_and_addition_l533_533681

theorem logM_NM_eq_one_and_addition (M N : ℝ) (h1 : log M N * log N M = 1) (h2 : M ≠ N) (h3 : M * N > 0) (h4 : M ≠ 1) (h5 : N ≠ 1) : M + N = 2 :=
by sorry

end logM_NM_eq_one_and_addition_l533_533681


namespace smallest_marked_cells_l533_533149

theorem smallest_marked_cells (k : ℕ) : ∃ k = 48, ∀ (marking : ℕ × ℕ → Prop), 
  (∀ i j, 0 ≤ i < 12 → 0 ≤ j < 12 → marking (i, j)) → 
  (∃ i j, 0 ≤ i < 12 → 0 ≤ j < 12 → 
    (marking (i, j)) → 
    (∀ r s, 0 ≤ r < 2 → 0 ≤ s < 2 → 
      (i + r, j + s) ∈ { (i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1) }) → 
    ∃ marked_cell (i + r, j + s) ) := 
sorry

end smallest_marked_cells_l533_533149


namespace area_of_triangle_BOC_l533_533296

-- Definitions for the line and circle
def line_parametric (t : ℝ) : ℝ × ℝ :=
  (2 - t * Real.sin (Real.pi / 6), -1 + t * Real.sin (Real.pi / 6))

def circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 8

-- Coordinates of B and C found by solving the intersection problem
def B : ℝ × ℝ := (3/2 - Real.sqrt 3 / 2, -1/2 - Real.sqrt 3 / 2)
def C : ℝ × ℝ := (3/2 + Real.sqrt 3 / 2, -1/2 + Real.sqrt 3 / 2)

-- Function to calculate the area of the triangle OBC
def area_triangle (O B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * Real.abs ((B.1 * C.2) - (C.1 * B.2))

-- Main theorem stating the area of triangle BOC
theorem area_of_triangle_BOC :
  area_triangle (0, 0) B C = Real.sqrt 15 / 2 :=
sorry

end area_of_triangle_BOC_l533_533296


namespace max_distance_AB_l533_533717

-- Define curve C1 in Cartesian coordinates
def C1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define curve C2 in Cartesian coordinates
def C2 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the problem to prove the maximum value of distance AB is 8
theorem max_distance_AB :
  ∀ (Ax Ay Bx By : ℝ),
    C1 Ax Ay →
    C2 Bx By →
    dist (Ax, Ay) (Bx, By) ≤ 8 :=
sorry

end max_distance_AB_l533_533717


namespace sum_of_reversed_base_5_numbers_eq_l533_533083

def sum_of_base_5_numbers : ℕ :=
  let num1 := 201
  let num2 := 402
  let expected_sum := 1103
  expected_sum

theorem sum_of_reversed_base_5_numbers_eq :
  let A₁ := 2, B₁ := 0, C₁ := 1
  let A₂ := 4, B₂ := 0, C₂ := 2
  let num1 := 25 * A₁ + 5 * B₁ + C₁
  let num2 := 25 * A₂ + 5 * B₂ + C₂
  let sum_in_base_10 := num1 + num2
  let sum_in_base_5 := 125 * 1 + 25 * 1 + 5 * 0 + 3
  sum_in_base_10 = 153 :=
by
  assume A₁ B₁ C₁ A₂ B₂ C₂ num1 num2 sum_in_base_10 sum_in_base_5
  sorry

end sum_of_reversed_base_5_numbers_eq_l533_533083


namespace compare_abc_l533_533626

def a : ℝ := Real.log Real.pi
def b : ℝ := Real.log 2 / Real.log 3
def c : ℝ := 5^(-1 / 2 : ℝ)

theorem compare_abc : c < b ∧ b < a :=
by {
  have h1 : a = Real.log Real.pi := sorry,
  have h2 : b = Real.log 2 / Real.log 3 := sorry,
  have h3 : c = 5^(-1 / 2 : ℝ) := sorry,
  
  -- Insert intermediate steps using the solution if needed for proving the theorem.
  
  sorry} 

end compare_abc_l533_533626


namespace number_of_valid_angles_l533_533303

-- Define the condition that describes the values of x between 0 and 360 degrees
def is_valid_angle (x : ℝ) : Prop :=
  0 ≤ x ∧ x < 360

-- Define the condition that describes the cosine value equal to -0.5
def cos_condition (x : ℝ) : Prop :=
  Real.cos (x * Real.pi / 180) = -0.5

-- Define the condition that describes the sine value less than 0
def sin_condition (x : ℝ) : Prop :=
  Real.sin (x * Real.pi / 180) < 0

-- Combine all conditions to form the final statement
theorem number_of_valid_angles : 
  {x : ℝ | is_valid_angle x ∧ cos_condition x ∧ sin_condition x}.toFinset.card = 1 :=
by 
  sorry

end number_of_valid_angles_l533_533303


namespace solution_10_digit_divisible_by_72_l533_533217

def attach_digits_to_divisible_72 : Prop :=
  ∃ (a d : ℕ), (a < 10) ∧ (d < 10) ∧ a * 10^9 + 20222023 * 10 + d = 3202220232 ∧ (3202220232 % 72 = 0)

theorem solution_10_digit_divisible_by_72 : attach_digits_to_divisible_72 :=
  sorry

end solution_10_digit_divisible_by_72_l533_533217


namespace weather_conditions_on_Dec_15_l533_533214

-- Definitions
def temperature_below_32 (T : ℝ) : Prop := T < 32
def temperature_at_least_32 (T : ℝ) : Prop := T ≥ 32
def snowing : Prop := sorry
def closed : Prop := sorry
def open : Prop := sorry

-- Conditions
axiom temp_and_snow_impl_closed : (temperature_below_32 T ∧ snowing) → closed
axiom pass_open : open

-- Theorem to Prove
theorem weather_conditions_on_Dec_15 (T : ℝ) : open → (temperature_at_least_32 T ∨ ¬ snowing) :=
by
  sorry

end weather_conditions_on_Dec_15_l533_533214


namespace certain_number_is_14_l533_533160

theorem certain_number_is_14 (a b : ℕ) (q : set ℕ) (n : ℕ) 
  (h1 : a % n = 0) (h2 : b % n = 0) 
  (h3 : q = set.Icc a b)
  (h4 : ∃ k, set.count (λ x, x % n = 0) q = 9) 
  (h5 : ∃ k, set.count (λ x, x % 7 = 0) q = 17) : 
  n = 14 :=
begin
  sorry
end

end certain_number_is_14_l533_533160


namespace y_investment_l533_533165

theorem y_investment (x_investment : ℝ) (y_ratio : ℝ) (x_ratio : ℝ) : x_investment = 5000 → x_ratio = 1 → y_ratio = 3 → y_investment = 15000 :=
by
  intros x_investment_eq x_ratio_eq y_ratio_eq
  sorry

end y_investment_l533_533165


namespace rectangular_area_eq_l533_533544

variables (e f g h : ℝ)
variables (he : 0 < e) (hf : 0 < f) (hg : 0 < g) (hh : 0 < h)

theorem rectangular_area_eq : 
  let horizontal_length := (2 * e) + (3 * f)
  let vertical_length := (4 * g) + (5 * h)
  let area := horizontal_length * vertical_length
  in area = 8 * e * g + 10 * e * h + 12 * f * g + 15 * f * h := 
by 
  sorry

end rectangular_area_eq_l533_533544


namespace viviana_vanilla_chips_l533_533128

variable (V_c S_c V_v S_v : ℕ)

-- Conditions
axiom h1 : S_c = 25
axiom h2 : V_c = S_c + 5
axiom h3 : S_v = (3 / 4 : ℚ) * V_v
axiom h4 : V_c + S_c + V_v + S_v = 90

theorem viviana_vanilla_chips : V_v = 20 :=
by
  rw [h1] at h2 h4
  sorry

end viviana_vanilla_chips_l533_533128


namespace aubree_animals_total_l533_533564

theorem aubree_animals_total (b_go c_go b_return c_return : ℕ) 
    (h1 : b_go = 20) (h2 : c_go = 40) 
    (h3 : b_return = b_go * 2) 
    (h4 : c_return = c_go - 10) : 
    b_go + c_go + b_return + c_return = 130 := by 
  sorry

end aubree_animals_total_l533_533564


namespace total_animals_l533_533569

-- Definitions of the initial conditions
def initial_beavers := 20
def initial_chipmunks := 40
def doubled_beavers := 2 * initial_beavers
def decreased_chipmunks := initial_chipmunks - 10

theorem total_animals (initial_beavers initial_chipmunks doubled_beavers decreased_chipmunks : ℕ)
    (h1 : doubled_beavers = 2 * initial_beavers)
    (h2 : decreased_chipmunks = initial_chipmunks - 10) :
    (initial_beavers + initial_chipmunks) + (doubled_beavers + decreased_chipmunks) = 130 :=
by 
  sorry

end total_animals_l533_533569


namespace max_value_of_xy_expression_l533_533042

theorem max_value_of_xy_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x + 3 * y < 60) : 
  xy * (60 - 4 * x - 3 * y) ≤ 2000 / 3 := 
sorry

end max_value_of_xy_expression_l533_533042


namespace find_limpet_shells_l533_533956

variable (L L_shells E_shells J_shells totalShells : ℕ)

def Ed_and_Jacob_initial_shells := 2
def Ed_oyster_shells := 2
def Ed_conch_shells := 4
def Jacob_more_shells := 2
def total_shells := 30

def Ed_total_shells := L + Ed_oyster_shells + Ed_conch_shells
def Jacob_total_shells := Ed_total_shells + Jacob_more_shells

theorem find_limpet_shells
  (H : Ed_and_Jacob_initial_shells + Ed_total_shells + Jacob_total_shells = total_shells) :
  L = 7 :=
by
  sorry

end find_limpet_shells_l533_533956


namespace minimum_marked_cells_k_l533_533143

def cell := (ℕ × ℕ)

def is_marked (board : set cell) (cell : cell) : Prop :=
  cell ∈ board

def touches_marked_cell (board : set cell) (fig : set cell) : Prop :=
  ∃ (c : cell), c ∈ fig ∧ is_marked(board, c)

def four_cell_figures (fig : set cell) : Prop :=
  (fig = { (0, 0), (0, 1), (1, 0), (1, 1) }) ∨
  (fig = { (0, 0), (1, 0), (2, 0), (3, 0) }) ∨
  (fig = { (0, 0), (0, 1), (0, 2), (0, 3) }) ∨
  -- Add other rotations and flipped configurations if needed

def board12x12 :=
  { (i, j) | i < 12 ∧ j < 12 }

theorem minimum_marked_cells_k :
  ∃ (k : ℕ) (marked_cells : (fin 144) → bool),
  k = 48 ∧
  (∀ (fig : set cell),
    four_cell_figures fig →
    ∀ (x y : ℕ), x < 12 ∧ y < 12 →
      touches_marked_cell
        { cell | marked_cells (fin.mk (cell.1 * 12 + cell.2) _) }
        (image (λ (rc : cell), (rc.1 + x, rc.2 + y)) fig)
  ) :=
begin
  sorry
end

end minimum_marked_cells_k_l533_533143


namespace angle_A_equal_pi_div_3_y_in_range_area_of_triangle_l533_533365

-- Question 1
theorem angle_A_equal_pi_div_3 
  (a b c : ℝ) (A B C : ℝ) 
  (h₁ : ∠ A + ∠ B + ∠ C = π)
  (h₂ : tan A / tan B = 2 * c / b) :
  A = π / 3 :=
sorry

-- Question 2
theorem y_in_range
  (A B C : ℝ)
  (h₁ : ∠ A + ∠ B + ∠ C = π)
  (h₂ : A = π / 3) 
  (h₃ : ∠ A < π / 2 ∧ ∠ B < π / 2 ∧ ∠ C < π / 2) 
  (y : ℝ)
  (h₄ : y = 2 * sin B ^ 2 - 2 * cos B * cos C) :
  y ∈ Ioo (1 / 2) 2 :=
sorry

-- Question 3
theorem area_of_triangle
  (a b c : ℝ) (A B C : ℝ)
  (h₁ : ∠ A = π / 3)
  (h₂ : ∠ B = π / 4 ∨ (∃ b : ℝ, (sqrt 3 + 1) * b = 0)) :
    area a b c = (sqrt 3 + 3) / 12 :=
sorry

end angle_A_equal_pi_div_3_y_in_range_area_of_triangle_l533_533365


namespace coplanar_points_l533_533775

open_locale classical

variables {V : Type*} [inner_product_space ℝ V]
variables (O A B C P : V)

-- Given condition
def coplanar_condition (O A B C P : V) : Prop :=
  P - O = (3/4) • (A - O) + (1/8) • (B - O) + (1/8) • (C - O)

-- The problem: proving the points are coplanar
theorem coplanar_points (h : coplanar_condition O A B C P) : 
  ∃ (u v w : ℝ), u + v + w = 1 ∧ P = u • A + v • B + w • C :=
sorry

end coplanar_points_l533_533775


namespace coin_difference_l533_533405

theorem coin_difference (h : ∃ (n5 n10 n25 n50 : ℕ), n5 * 5 + n10 * 10 + n25 * 25 + n50 * 50 = 75) :
  (∀ x5 x10 x25 x50, x5 * 5 + x10 * 10 + x25 * 25 + x50 * 50 = 75 → 2 ≤ x5 + x10 + x25 + x50 ≤ 15) →
  (∃ a b, (a = 2 ∧ b = 15) ∧ (b - a = 13)) :=
by sorry

end coin_difference_l533_533405


namespace octagon_area_difference_l533_533712

theorem octagon_area_difference :
  let A := (1 + 1 * has_sqrt.sqrt 2) in
  let smaller_octagon_area := A in
  let larger_octagon_area := 4 * A in
  (larger_octagon_area - smaller_octagon_area) = 6 + 6 * has_sqrt.sqrt 2 :=
sorry

end octagon_area_difference_l533_533712


namespace sum_of_divisors_lt_square_l533_533751

theorem sum_of_divisors_lt_square (n : ℕ) (hpos : 0 < n) :
  ∑ k in (finset.filter (λ k : ℕ, n % k = 0) (finset.range (n + 1))), k < n ^ 2 :=
by
  sorry

end sum_of_divisors_lt_square_l533_533751


namespace sum_first_100_odd_l533_533581

theorem sum_first_100_odd :
  (Finset.sum (Finset.range 100) (λ x => 2 * (x + 1) - 1)) = 10000 := by
  sorry

end sum_first_100_odd_l533_533581


namespace rational_iff_geometric_progression_l533_533072

theorem rational_iff_geometric_progression (x : ℚ) :
  (∃ a b c : ℕ, a < b ∧ b < c ∧ (x + a) * (x + c) = (x + b) * (x + b))
  ↔ x ∈ ℚ :=
sorry

end rational_iff_geometric_progression_l533_533072


namespace negation_of_zero_product_l533_533406

theorem negation_of_zero_product (x y : ℝ) : (xy ≠ 0) → (x ≠ 0) ∧ (y ≠ 0) :=
sorry

end negation_of_zero_product_l533_533406


namespace no_integers_a_b_existence_no_positive_integers_a_b_c_existence_l533_533506

-- Part (a)
theorem no_integers_a_b_existence (a b : ℤ) :
  ¬(a^2 - 3 * (b^2) = 8) :=
sorry

-- Part (b)
theorem no_positive_integers_a_b_c_existence (a b c : ℕ) (ha: a > 0) (hb: b > 0) (hc: c > 0 ) :
  ¬(a^2 + b^2 = 3 * (c^2)) :=
sorry

end no_integers_a_b_existence_no_positive_integers_a_b_c_existence_l533_533506


namespace exists_multiple_with_sum_divisible_l533_533381

-- Define the sum of the digits
def sum_of_digits (n : ℕ) : ℕ := -- Implementation of sum_of_digits function is omitted here
sorry

-- Main theorem statement
theorem exists_multiple_with_sum_divisible (n : ℕ) (hn : n > 0) : 
  ∃ k, k % n = 0 ∧ sum_of_digits k ∣ k :=
sorry

end exists_multiple_with_sum_divisible_l533_533381


namespace complex_quadrant_proof_l533_533648

noncomputable def i : ℂ := complex.I

-- Define the complex number z
noncomputable def z : ℂ := i + 2 * (i^2) + 3 * (i^3)

-- Define a predicate for "in the third quadrant"
def in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

-- The theorem we need to prove
theorem complex_quadrant_proof (i := complex.I) (z := i + 2*i^2 + 3*i^3) :
  in_third_quadrant z :=
sorry

end complex_quadrant_proof_l533_533648


namespace ancient_chinese_problem_l533_533348

theorem ancient_chinese_problem (x y : ℤ) 
  (h1 : y = 8 * x - 3) 
  (h2 : y = 7 * x + 4) : 
  (y = 8 * x - 3) ∧ (y = 7 * x + 4) :=
by
  exact ⟨h1, h2⟩

end ancient_chinese_problem_l533_533348


namespace paint_problem_l533_533760

theorem paint_problem :
  let mary_paint : ℕ := 3
  let mike_paint : ℕ := mary_paint + 2
  let total_paint : ℕ := 13
  let used_paint : ℕ := mary_paint + mike_paint
  total_paint - used_paint = 5 := 
by
  let mary_paint : ℕ := 3
  let mike_paint : ℕ := mary_paint + 2
  let total_paint : ℕ := 13
  let used_paint : ℕ := mary_paint + mike_paint
  show total_paint - used_paint = 5 from sorry

end paint_problem_l533_533760


namespace angle_between_uv_l533_533244

def u : ℝ × ℝ := (3, 4)
def v : ℝ × ℝ := (4, -3)

theorem angle_between_uv : 
  let dot_product := u.1 * v.1 + u.2 * v.2 in
  let mag_u := Real.sqrt (u.1 ^ 2 + u.2 ^ 2) in
  let mag_v := Real.sqrt (v.1 ^ 2 + v.2 ^ 2) in
  let cos_theta := dot_product / (mag_u * mag_v) in
  Real.arccos cos_theta = π / 2 :=
by
  sorry

end angle_between_uv_l533_533244


namespace ratio_of_AC_to_BD_l533_533065

noncomputable def length_of_AC_and_BD (AB BC AD : ℝ) : ℝ × ℝ :=
  let AC := AB + BC
  let BD := AD - AB
  (AC, BD)

theorem ratio_of_AC_to_BD
  (AB BC AD : ℝ)
  (h1 : AB = 4)
  (h2 : BC = 3)
  (h3 : AD = 20) :
  let (AC, BD) := length_of_AC_and_BD AB BC AD in
  AC / BD = 7 / 16 := by
  sorry

end ratio_of_AC_to_BD_l533_533065


namespace composite_function_l533_533261

def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x + 1

theorem composite_function : ∀ (x : ℝ), f (g x) = 2 * x + 1 :=
by
  intro x
  sorry

end composite_function_l533_533261


namespace payment_difference_correct_l533_533759

noncomputable def initial_debt : ℝ := 12000

noncomputable def planA_interest_rate : ℝ := 0.08
noncomputable def planA_compounding_periods : ℕ := 2

noncomputable def planB_interest_rate : ℝ := 0.08

noncomputable def planA_payment_years : ℕ := 4
noncomputable def planA_remaining_years : ℕ := 4

noncomputable def planB_years : ℕ := 8

-- Amount accrued in Plan A after 4 years
noncomputable def planA_amount_after_first_period : ℝ :=
  initial_debt * (1 + planA_interest_rate / planA_compounding_periods) ^ (planA_compounding_periods * planA_payment_years)

-- Amount paid at the end of first period (two-thirds of total)
noncomputable def planA_first_payment : ℝ :=
  (2/3) * planA_amount_after_first_period

-- Remaining debt after first payment
noncomputable def planA_remaining_debt : ℝ :=
  planA_amount_after_first_period - planA_first_payment

-- Amount accrued on remaining debt after 8 years (second 4-year period)
noncomputable def planA_second_payment : ℝ :=
  planA_remaining_debt * (1 + planA_interest_rate / planA_compounding_periods) ^ (planA_compounding_periods * planA_remaining_years)

-- Total payment under Plan A
noncomputable def total_payment_planA : ℝ :=
  planA_first_payment + planA_second_payment

-- Total payment under Plan B
noncomputable def total_payment_planB : ℝ :=
  initial_debt * (1 + planB_interest_rate * planB_years)

-- Positive difference between payments
noncomputable def payment_difference : ℝ :=
  total_payment_planB - total_payment_planA

theorem payment_difference_correct :
  payment_difference = 458.52 :=
by
  sorry

end payment_difference_correct_l533_533759


namespace interest_rate_B_lent_to_C_l533_533897

noncomputable def principal : ℝ := 1500
noncomputable def rate_A : ℝ := 10
noncomputable def time : ℝ := 3
noncomputable def gain_B : ℝ := 67.5
noncomputable def interest_paid_by_B_to_A : ℝ := principal * rate_A * time / 100
noncomputable def interest_received_by_B_from_C : ℝ := interest_paid_by_B_to_A + gain_B
noncomputable def expected_rate : ℝ := 11.5

theorem interest_rate_B_lent_to_C :
  interest_received_by_B_from_C = principal * (expected_rate) * time / 100 := 
by
  -- the proof will go here
  sorry

end interest_rate_B_lent_to_C_l533_533897


namespace range_of_quadratic_expression_l533_533822

theorem range_of_quadratic_expression (x : ℝ) (h : x^2 - 7*x + 12 < 0) :
  42 < x^2 + 7*x + 12 ∧ x^2 + 7*x + 12 < 56 :=
begin
  sorry
end

end range_of_quadratic_expression_l533_533822
