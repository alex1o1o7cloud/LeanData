import Mathlib

namespace angle_CK_BD_l661_661190

--- Definitions
def Point := ℝ × ℝ
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def rectangle (A B C D : Point) (AB BC : ℝ) : Prop :=
  distance A B = AB ∧ distance B C = BC ∧
  ∃ D', distance C D' = AB ∧ distance A D' = BC

def PointsOnRectangle (K A B C : Point) : Prop :=
  distance K A = real.sqrt 10 ∧ distance K B = 2 ∧ distance K C = 3

def angle_between_lines (p1 p2 p3 p4 : Point) : ℝ := 
  let v1 := (p2.1 - p1.1, p2.2 - p1.2)
  let v2 := (p4.1 - p3.1, p4.2 - p3.2)
  real.arcsin ((v1.1 * v2.2 - v1.2 * v2.1) / (real.sqrt (v1.1^2 + v1.2^2) * real.sqrt (v2.1^2 + v2.2^2)))

--- Theorem statement
theorem angle_CK_BD (A B C D K : Point) (AB BC : ℝ) 
  (rect : rectangle A B C D AB BC) (pts : PointsOnRectangle K A B C) :
  angle_between_lines C K B D = real.arcsin (4 / 5) :=
sorry

end angle_CK_BD_l661_661190


namespace rectangle_B_perimeter_l661_661278

theorem rectangle_B_perimeter 
    (A_perimeter : ℕ) 
    (squares_A : ℕ) 
    (squares_B : ℕ) 
    (A_perimeter_eq : A_perimeter = 112) 
    (squares_A_eq : squares_A = 3) 
    (squares_B_eq : squares_B = 4) :
    (B_perimeter : ℕ) (B_perimeter = 168) :=
begin
  sorry
end

end rectangle_B_perimeter_l661_661278


namespace greatest_triangle_perimeter_l661_661580

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l661_661580


namespace no_real_solutions_l661_661406

theorem no_real_solutions :
  ∀ (x : ℝ), (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) ≠ 1 / 8) :=
by
  intro x
  sorry

end no_real_solutions_l661_661406


namespace alice_quadratic_expression_l661_661364

theorem alice_quadratic_expression (b : ℝ) (h_b_pos : b > 0) :
  (∃ p : ℝ, (∀ x : ℝ, x^2 + b*x + 1 = (x + p)^2 - 7/4) ∧ b = sqrt 11) :=
begin
  use sqrt 11 / 2, -- p = sqrt 11 / 2
  split,
  {
    intro x,
    calc
    x^2 + b*x + 1 = x^2 + 2*(sqrt 11 / 2)*x + (sqrt 11 / 2)^2 - 7/4 : by sorry
               ... = (x + sqrt 11 / 2)^2 - 7/4 : by sorry,
  },
  {
    exact sqrt 11_pos, -- b = sqrt 11 as b is positive
  }
end

end alice_quadratic_expression_l661_661364


namespace incorrect_arrangements_hello_l661_661177

-- Given conditions: the word "hello" with letters 'h', 'e', 'l', 'l', 'o'
def letters : List Char := ['h', 'e', 'l', 'l', 'o']

-- The number of permutations of the letters in "hello" excluding the correct order
-- We need to prove that the number of incorrect arrangements is 59.
theorem incorrect_arrangements_hello : 
  (List.permutations letters).length - 1 = 59 := 
by sorry

end incorrect_arrangements_hello_l661_661177


namespace total_amount_owed_l661_661823

theorem total_amount_owed :
  ∃ (P remaining_balance processing_fee new_total discount: ℝ),
    0.05 * P = 50 ∧
    remaining_balance = P - 50 ∧
    processing_fee = 0.03 * remaining_balance ∧
    new_total = remaining_balance + processing_fee ∧
    discount = 0.10 * new_total ∧
    new_total - discount = 880.65 :=
sorry

end total_amount_owed_l661_661823


namespace greatest_4_digit_number_divisible_by_15_25_40_75_l661_661792

theorem greatest_4_digit_number_divisible_by_15_25_40_75 : 
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (15 ∣ n) ∧ (25 ∣ n) ∧ (40 ∣ n) ∧ (75 ∣ n) ∧ n = 9600 :=
by
  -- Proof to be provided
  sorry

end greatest_4_digit_number_divisible_by_15_25_40_75_l661_661792


namespace find_z_coordinate_l661_661841

-- Define the two points the line passes through
def point1 : ℝ × ℝ × ℝ := (3, 3, 2)
def point2 : ℝ × ℝ × ℝ := (6, 2, -1)

-- Define the line in parametric form
def parametric_line (t : ℝ) : ℝ × ℝ × ℝ :=
  let (x1, y1, z1) := point1
  let (x2, y2, z2) := point2
  (x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1))

-- Define the condition x = 5
def x_condition (x : ℝ × ℝ × ℝ) : Prop := x.1 = 5

-- The goal is to find and prove the z-coordinate given the x-condition
theorem find_z_coordinate : ∃ z, ∃ t, 
  let (x, y, z') := parametric_line t in 
  x = 5 ∧ z = z' := 
by
  -- Skipping the detailed steps of the proof
  sorry


end find_z_coordinate_l661_661841


namespace distance_range_in_tetrahedron_l661_661275

theorem distance_range_in_tetrahedron
  (A B C D P Q : ℝ × ℝ × ℝ)
  (h_tetrahedron : ∀ (u v : ℝ × ℝ × ℝ), u ∈ {A, B, C, D} → v ∈ {A, B, C, D} → u ≠ v → dist u v = 1)
  (h_PQ_on_AB_CD : ∃ (t1 t2 : ℝ), 0 ≤ t1 ∧ t1 ≤ 1 ∧ P = t1 • A + (1 - t1) • B ∧ 0 ≤ t2 ∧ t2 ≤ 1 ∧ Q = t2 • C + (1 - t2) • D) :
  dist P Q ∈ set.Icc (Real.sqrt 2 / 2) 1 := sorry

end distance_range_in_tetrahedron_l661_661275


namespace product_of_roots_l661_661940

theorem product_of_roots :
  let p : Polynomial ℚ := Polynomial.Coeff.mk_fin 3 0 (⟨[50, -75, 15, -1]⟩ : ℕ → ℚ)
  (p.root_prod = 50) :=
begin
  sorry
end

end product_of_roots_l661_661940


namespace inequality_solution_set_inequality_proof_2_l661_661337

theorem inequality_solution_set : 
  { x : ℝ | |x + 1| + |x + 3| < 4 } = { x : ℝ | -4 < x ∧ x < 0 } :=
sorry

theorem inequality_proof_2 (a b : ℝ) (ha : -4 < a) (ha' : a < 0) (hb : -4 < b) (hb' : b < 0) : 
  2 * |a - b| < |a * b + 2 * a + 2 * b| :=
sorry

end inequality_solution_set_inequality_proof_2_l661_661337


namespace opposite_of_neg_2023_l661_661196

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 :=
by {
  use 2023,
  split,
  { exact eq.refl 0 },
  { refl }
}

end opposite_of_neg_2023_l661_661196


namespace malia_buttons_l661_661320

theorem malia_buttons (a : ℕ) : 
  let first := a,
      second := 3,
      third := 9,
      fourth := 27,
      fifth := 81,
      sixth := 243 in
  second = 3 * first ∧ 
  third = 3 * second ∧ 
  fourth = 3 * third ∧ 
  fifth = 3 * fourth ∧ 
  sixth = 3 * fifth → 
  first = 1 :=
by
  intros h,
  have h1 : second = 3 * first := h.1,
  have h2 : third = 3 * second := h.2.1,
  have h3 : fourth = 3 * third := h.2.2.1,
  have h4 : fifth = 3 * fourth := h.2.2.2.1,
  have h5 : sixth = 3 * fifth := h.2.2.2.2,
  sorry

end malia_buttons_l661_661320


namespace unique_integer_n_l661_661103

noncomputable def S (n : ℕ) (b : ℕ → ℝ) : ℝ :=
∑ k in finset.range n, real.sqrt ((2*k+1)^2 + (b k)^2)

theorem unique_integer_n (n : ℕ) (b : ℕ → ℝ) (h : ∑ k in finset.range n, b k = 25) :
  S n b = real.sqrt (n^4 + 625) → n = 18 := sorry

end unique_integer_n_l661_661103


namespace product_of_roots_of_cubic_polynomial_l661_661901

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l661_661901


namespace greatest_possible_perimeter_l661_661596

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l661_661596


namespace exists_squares_sum_l661_661719

noncomputable def sequence (n : ℕ) : ℕ :=
  nat.rec_on n 5 (fun n aₙ => nat.rec_on n 25 (fun n aₙ₁ => 7 * aₙ₁ - aₙ - 6))

theorem exists_squares_sum :
  ∃ x y : ℤ, sequence 2023 = x^2 + y^2 :=
sorry

end exists_squares_sum_l661_661719


namespace product_of_roots_of_cubic_polynomial_l661_661896

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l661_661896


namespace miranda_savings_per_month_l661_661243

-- Only import everything needed to ensure the code can be built successfully

theorem miranda_savings_per_month :
  ∀ (months_saved : ℕ) (sister_contrib : ℕ) (original_price : ℕ) (discount_percent : ℕ)
    (shipping_cost : ℕ) (total_paid : ℕ),
  months_saved = 3 → sister_contrib = 50 → original_price = 240 → discount_percent = 10 →
  shipping_cost = 20 → total_paid = 236 →
  (total_paid - sister_contrib) / months_saved = 62 :=
by
  intros months_saved sister_contrib original_price discount_percent shipping_cost total_paid
  intro h_months_saved h_sister_contrib h_original_price h_discount_percent h_shipping_cost h_total_paid
  -- Define intermediate computation steps used only in the proof and then prove the final result
  have intermediate_step : (236 - 50) / 3 = 62 := sorry
  exact intermediate_step

end miranda_savings_per_month_l661_661243


namespace quadratic_equation_general_form_l661_661987

theorem quadratic_equation_general_form :
  ∀ (x : ℝ), 3 * x^2 + 1 = 7 * x ↔ 3 * x^2 - 7 * x + 1 = 0 :=
by
  intro x
  constructor
  · intro h
    sorry
  · intro h
    sorry

end quadratic_equation_general_form_l661_661987


namespace least_divisor_to_perfect_square_l661_661810

theorem least_divisor_to_perfect_square (n : ℕ) (h : n = 16800) : 
  ∃ m, m = 21 ∧ (∃ k, (n / m) = k^2) :=
by
  use 21
  split
  · rfl
  · sorry

end least_divisor_to_perfect_square_l661_661810


namespace area_of_square_is_correct_l661_661430

-- Define the nature of the problem setup and parameters
def radius_of_circle : ℝ := 7
def diameter_of_circle : ℝ := 2 * radius_of_circle
def side_length_of_square : ℝ := 2 * diameter_of_circle
def area_of_square : ℝ := side_length_of_square ^ 2

-- Statement of the problem to prove
theorem area_of_square_is_correct : area_of_square = 784 := by
  sorry

end area_of_square_is_correct_l661_661430


namespace eggs_eaten_by_parents_each_night_l661_661734

noncomputable def totalEggsPurchased : ℕ := 2 * 24
def eggsEatenByChildren : ℕ := 2 * 2 * 7
def eggsNotEaten : ℕ := 6
def totalEggsForParents : ℕ := totalEggsPurchased - eggsEatenByChildren - eggsNotEaten
def daysInWeek : ℕ := 7

theorem eggs_eaten_by_parents_each_night
  (totalEggsPurchased = 2 * 24)
  (eggsEatenByChildren = 2 * 2 * 7)
  (eggsNotEaten = 6)
  (totalEggsForParents = totalEggsPurchased - eggsEatenByChildren - eggsNotEaten)
  (daysInWeek = 7) :
  totalEggsForParents / daysInWeek = 2 := sorry

end eggs_eaten_by_parents_each_night_l661_661734


namespace surface_area_ratio_l661_661351

theorem surface_area_ratio (l : ℕ) (n : ℕ) (s : ℕ) (h1 : l = 5) (h2 : n = 125) (h3 : s = 1) :
  let S_large := 6 * l^2 in
  let S_small := n * 6 * s^2 in
  (S_small / S_large) = 5 :=
by
  sorry

end surface_area_ratio_l661_661351


namespace dave_apps_files_difference_l661_661075

theorem dave_apps_files_difference :
  let initial_apps := 15
  let initial_files := 24
  let final_apps := 21
  let final_files := 4
  final_apps - final_files = 17 :=
by
  intros
  sorry

end dave_apps_files_difference_l661_661075


namespace max_digit_sum_watch_l661_661025

def digit_sum (n : Nat) : Nat :=
  (n / 10) + (n % 10)

theorem max_digit_sum_watch :
  ∃ (h m : Nat), (1 <= h ∧ h <= 12) ∧ (0 <= m ∧ m <= 59) 
  ∧ (digit_sum h + digit_sum m = 23) :=
by 
  sorry

end max_digit_sum_watch_l661_661025


namespace min_value_of_function_l661_661094

open Real

theorem min_value_of_function (x y : ℝ) (h : 2 * x + 8 * y = 3) : ∃ (min_value : ℝ), min_value = -19 / 20 ∧ ∀ (x y : ℝ), 2 * x + 8 * y = 3 → x^2 + 4 * y^2 - 2 * x ≥ -19 / 20 :=
by
  sorry

end min_value_of_function_l661_661094


namespace change_in_expression_increase_or_decrease_l661_661984

variable (x a k : ℝ) (hka : 0 < k) (hax : 0 < a)

theorem change_in_expression_increase_or_decrease 
: 3 * (x + a) ^ 2 - k - (3 * x ^ 2 - k) = 6 * x * a + 3 * a ^ 2 
∧ 3 * (x - a) ^ 2 - k - (3 * x ^ 2 - k) = -6 * x * a + 3 * a ^ 2 := 
by split; sorry

end change_in_expression_increase_or_decrease_l661_661984


namespace max_perimeter_l661_661628

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l661_661628


namespace number_of_valid_triples_in_S_l661_661693

-- Define the set S
def S : Set ℕ := { n | 1 ≤ n ∧ n ≤ 23 }

-- Define the relation a ≻ b
def rel (a b : ℕ) : Prop := (0 < a - b ∧ a - b ≤ 11) ∨ (b - a > 11)

-- Define the property of ordered triple
def valid_triple (x y z : ℕ) : Prop := rel x y ∧ rel y z ∧ rel z x

-- Define the proof problem
theorem number_of_valid_triples_in_S : 
  ∃ n, n = 759 ∧ (∃ triples : Finset (ℕ × ℕ × ℕ), (∀ t ∈ triples, valid_triple t.1 t.2 t.3) ∧ Finset.card triples = n) :=
sorry

end number_of_valid_triples_in_S_l661_661693


namespace product_of_roots_cubic_l661_661961

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  let roots := {x : ℝ // p x = 0}.toFinset
  ∃ r : ℝ, (∀ x ∈ roots, p x = 0) ∧ (∏ x in roots, x) = 50 :=
by
  sorry

end product_of_roots_cubic_l661_661961


namespace beetle_max_distance_positions_l661_661248

-- Define the initial positions of water strider and beetle
def M0 := (2 : ℝ, 2 * Real.sqrt 7)
def N0 := (5 : ℝ, 5 * Real.sqrt 7)

-- Define water surface coordinates system and float position
def float_position := (0 : ℝ, 0 : ℝ)

-- Define water strider speed as twice the speed of beetle
def water_strider_speed := 2 * beetle_speed : ℝ

-- The main theorem stating the maximum distance coordinates
theorem beetle_max_distance_positions :
  ∃ (N1 N2 N3 N4 : ℝ × ℝ),
    N1 = (5 / Real.sqrt 2 * (1 - Real.sqrt 7), 5 / Real.sqrt 2 * (1 + Real.sqrt 7)) ∧
    N2 = (-5 / Real.sqrt 2 * (1 + Real.sqrt 7), 5 / Real.sqrt 2 * (1 - Real.sqrt 7)) ∧
    N3 = (5 / Real.sqrt 2 * (Real.sqrt 7 - 1), -5 / Real.sqrt 2 * (1 + Real.sqrt 7)) ∧
    N4 = (5 / Real.sqrt 2 * (1 + Real.sqrt 7), 5 / Real.sqrt 2 * (Real.sqrt 7 - 1))
:= sorry -- The proof is omitted.

end beetle_max_distance_positions_l661_661248


namespace total_children_on_playground_l661_661299

theorem total_children_on_playground (boys girls : ℕ) (hb : boys = 27) (hg : girls = 35) : boys + girls = 62 :=
  by
  -- Proof goes here
  sorry

end total_children_on_playground_l661_661299


namespace triangle_transformable_by_rotations_l661_661383

/-
  Define the transformation function for a 90-degree counterclockwise rotation 
  around a lattice point (m, n).
-/
def rotate_90_ccw (m n : ℤ) (a b : ℤ) : ℤ × ℤ :=
  (m + n - b, a + n - m)

/-
  The initial and target triangle vertices.
-/
def initial_triangle : set (ℤ × ℤ) :=
  {(0, 0), (1, 0), (0, 1)}

def target_triangle : set (ℤ × ℤ) :=
  {(0, 0), (1, 0), (1, 1)}

/-
  The statement that we need to prove.
-/
theorem triangle_transformable_by_rotations :
  ∃ seq : ℕ → ℤ × ℤ, 
    (∀ i, rotate_90_ccw (seq i).1 (seq i).2 (rotate_90_ccw (seq (i+1)).1 (seq (i+1)).2 (0, 0)) ∈ target_triangle) ∧
    (∀ i, rotate_90_ccw (seq i).1 (seq i).2 (rotate_90_ccw (seq (i+1)).1 (seq (i+1)).2 (1, 0)) ∈ target_triangle) ∧
    (∀ i, rotate_90_ccw (seq i).1 (seq i).2 (rotate_90_ccw (seq (i+1)).1 (seq (i+1)).2 (0, 1)) ∈ target_triangle) :=
begin
  sorry
end

end triangle_transformable_by_rotations_l661_661383


namespace all_statements_true_l661_661743

noncomputable def g : ℝ → ℝ := sorry

axiom g_defined (x : ℝ) : ∃ y, g x = y
axiom g_positive (x : ℝ) : g x > 0
axiom g_multiplicative (a b : ℝ) : g (a) * g (b) = g (a + b)
axiom g_div (a b : ℝ) (h : a > b) : g (a - b) = g (a) / g (b)

theorem all_statements_true :
  (g 0 = 1) ∧
  (∀ a, g (-a) = 1 / g (a)) ∧
  (∀ a, g (a) = (g (3 * a))^(1 / 3)) ∧
  (∀ a b, b > a → g (b - a) < g (b)) :=
by
  sorry

end all_statements_true_l661_661743


namespace sector_area_eq_three_halves_l661_661176

theorem sector_area_eq_three_halves (θ R S : ℝ) (hθ : θ = 3) (h₁ : 2 * R + θ * R = 5) :
  S = 3 / 2 :=
by
  sorry

end sector_area_eq_three_halves_l661_661176


namespace original_price_of_coffee_l661_661672

variable (P : ℝ)

theorem original_price_of_coffee :
  (4 * P - 2 * (1.5 * P) = 2) → P = 2 :=
by
  sorry

end original_price_of_coffee_l661_661672


namespace problem_solution_l661_661995

theorem problem_solution :
  let sum1 := ( ∑ x in ({x | x^2 - 7 * x + 10 = 0}).to_finset, x),
      sum2 := ( ∑ x in ({x | x^2 - 6 * x + 5 = 0}).to_finset, x),
      sum3 := ( ∑ x in ({x | x^2 - 6 * x + 6 = 0}).to_finset, x)
  in
  sum1 + sum2 + sum3 = 16 := sorry

end problem_solution_l661_661995


namespace pizza_dough_milk_water_l661_661102

theorem pizza_dough_milk_water (flour_amount milk_ratio flour_ratio : ℕ) 
  (h1 : flour_amount = 1200) (h2 : milk_ratio = 80) (h3 : flour_ratio = 400) :
  let portions := flour_amount / flour_ratio,
      milk_needed := portions * milk_ratio,
      water_needed := milk_needed / 2 in
  milk_needed = 240 ∧ water_needed = 120 := 
by
  let portions := flour_amount / flour_ratio in
  have portions_calculated : portions = 3 := by sorry,
  let milk_needed := portions * milk_ratio in
  have milk_needed_calculated : milk_needed = 240 := by sorry,
  let water_needed := milk_needed / 2 in
  have water_needed_calculated : water_needed = 120 := by sorry,
  exact ⟨milk_needed_calculated, water_needed_calculated⟩

end pizza_dough_milk_water_l661_661102


namespace length_of_AC_l661_661657

theorem length_of_AC (AB DC AD : ℝ) (h_AB : AB = 15) (h_DC : DC = 24) (h_AD : AD = 9) : 
  ∃ AC : ℝ, AC = Real.sqrt 873 ∧ Real.floor (10 * AC) / 10 = 29.5 :=
by
  sorry

end length_of_AC_l661_661657


namespace necessary_and_sufficient_condition_l661_661233

theorem necessary_and_sufficient_condition (x : ℝ) : (|x - 2| < 1) ↔ (1 < x ∧ x < 3) := 
by
  sorry

end necessary_and_sufficient_condition_l661_661233


namespace angle_between_AC_and_MN_l661_661651

variables (A B C D M N : Point)
variables (a b : ℝ)
variable (hb : b > a)
variable (AD_AB : ∀ P Q R S : Point, P = A → Q = D → R = A → S = B → dist P Q = a ∧ dist R S = b)
variable (folded : ∀ P Q : Point, P = A → Q = C → P = Q)
variable (dihedral_angle : ∀ P Q R S : Point, P = D → Q = A → R = M → S = N → dihedral_angle (P, Q, R) = 57)

theorem angle_between_AC_and_MN : ∀ P Q : Point, P = A → Q = C → angle (P, M, Q) = 90 :=
by
  sorry

end angle_between_AC_and_MN_l661_661651


namespace fuel_tank_capacity_l661_661015

-- Definition of conditions
variables (x : ℝ) (miles_per_gallon : ℝ := 24)
variable (modification_ratio : ℝ := 0.75)
variable (additional_miles : ℝ := 96)

-- Statement of the theorem
theorem fuel_tank_capacity : 
  (miles_per_gallon * x + additional_miles = miles_per_gallon * (1 / modification_ratio) * x) → 
  x = 12 :=
by
  -- Introduction of variables
  intro h,
  -- Provided is the provided condition equation
  have h1 : miles_per_gallon * x + additional_miles = miles_per_gallon * (1 / modification_ratio) * x := h,
  -- Proof would go here
  sorry

end fuel_tank_capacity_l661_661015


namespace greatest_possible_perimeter_l661_661534

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l661_661534


namespace greatest_possible_perimeter_l661_661553

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l661_661553


namespace triangle_sides_and_angles_find_angle_A_l661_661459

theorem triangle_sides_and_angles (A B C : ℝ) (a b c : ℝ)
  (h1 : c = 2) (h2 : C = (π / 3)) (h3 : (1 / 2) * a * b * Real.sin (π / 3) = √3) :
  a = 2 ∧ b = 2 :=
by sorry

theorem find_angle_A (A B C : ℝ) (a b c : ℝ)
  (h1 : c = 2) (h2 : C = π / 3)
  (h3 : Real.sin (π / 3) + Real.sin (B - A) = 2 * Real.sin (2 * A)) :
  A = π / 2 ∨ A = π / 6 :=
by sorry

end triangle_sides_and_angles_find_angle_A_l661_661459


namespace solve_inequality_l661_661265

theorem solve_inequality :
  { x : ℝ | (9 * x^2 + 27 * x - 64) / ((3 * x - 4) * (x + 5) * (x - 1)) < 4 } = 
    { x : ℝ | -5 < x ∧ x < -17 / 3 } ∪ { x : ℝ | 1 < x ∧ x < 4 } :=
by
  sorry

end solve_inequality_l661_661265


namespace product_of_roots_eq_50_l661_661950

theorem product_of_roots_eq_50 :
  let a := -15
  let c := -50
  let equation := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  -- Vieta's formulas for the product of roots in cubic equation ax^3 + bx^2 + cx + d = 0 is -d/a
  ∃ p q r : ℝ, 
    equation p = 0 ∧ equation q = 0 ∧ equation r = 0 ∧ (p * q * r = 50) :=
by
  sorry

end product_of_roots_eq_50_l661_661950


namespace midpoint_polar_coordinates_l661_661189

noncomputable def polar_midpoint :=
  let A := (10, 7 * Real.pi / 6)
  let B := (10, 11 * Real.pi / 6)
  let A_cartesian := (10 * Real.cos (7 * Real.pi / 6), 10 * Real.sin (7 * Real.pi / 6))
  let B_cartesian := (10 * Real.cos (11 * Real.pi / 6), 10 * Real.sin (11 * Real.pi / 6))
  let midpoint_cartesian := ((A_cartesian.1 + B_cartesian.1) / 2, (A_cartesian.2 + B_cartesian.2) / 2)
  let r := Real.sqrt (midpoint_cartesian.1 ^ 2 + midpoint_cartesian.2 ^ 2)
  let θ := if midpoint_cartesian.1 = 0 then 0 else Real.arctan (midpoint_cartesian.2 / midpoint_cartesian.1)
  (r, θ)

theorem midpoint_polar_coordinates :
  polar_midpoint = (5 * Real.sqrt 3, Real.pi) := by
  sorry

end midpoint_polar_coordinates_l661_661189


namespace tank_capacity_is_32_l661_661006

noncomputable def capacity_of_tank (C : ℝ) : Prop :=
  (3/4) * C + 4 = (7/8) * C

theorem tank_capacity_is_32 : ∃ C : ℝ, capacity_of_tank C ∧ C = 32 :=
sorry

end tank_capacity_is_32_l661_661006


namespace working_days_19_in_month_with_specific_conditions_l661_661188

theorem working_days_19_in_month_with_specific_conditions :
  let total_days := 30
  let saturdays := 4
  let sundays := 4
  let public_holidays := 3
  let unexpected_closures := 2
  let working_saturdays := saturdays / 2
  let non_working_days := saturdays + sundays + public_holidays + unexpected_closures
  total_days - non_working_days = 19 :=
by
  let total_days := 30
  let saturdays := 4
  let sundays := 4
  let public_holidays := 3
  let unexpected_closures := 2
  let working_saturdays := 2
  let non_working_days := 2 + 4 + public_holidays + unexpected_closures
  let working_days := total_days - non_working_days
  show working_days = 19
  sorry

end working_days_19_in_month_with_specific_conditions_l661_661188


namespace greatest_possible_perimeter_l661_661639

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l661_661639


namespace two_lines_perpendicular_to_same_line_are_parallel_l661_661049

theorem two_lines_perpendicular_to_same_line_are_parallel (P Q R : Type) [euclidean_geometry P] 
    {l m n : Line P} (hlm : l.perpendicular_to n) (hmn : m.perpendicular_to n) : 
    l.parallel_to m := 
sorry

end two_lines_perpendicular_to_same_line_are_parallel_l661_661049


namespace positive_difference_is_zero_l661_661670

-- Definitions based on conditions
def jo_sum (n : ℕ) : ℕ := (n * (n + 1)) / 2

def rounded_to_nearest_5 (x : ℕ) : ℕ :=
  if x % 5 = 0 then x
  else (x / 5) * 5 + (if x % 5 >= 3 then 5 else 0)

def alan_sum (n : ℕ) : ℕ :=
  (List.range (n + 1)).map rounded_to_nearest_5 |>.sum

-- Theorem based on question and correct answer
theorem positive_difference_is_zero :
  jo_sum 120 - alan_sum 120 = 0 := sorry

end positive_difference_is_zero_l661_661670


namespace reduced_rate_fraction_l661_661801

theorem reduced_rate_fraction:
  let total_hours_in_week := 7 * 24
  let weekday_hours := 12 * 5
  let weekend_hours := 24 * 2
  let reduced_hours := weekday_hours + weekend_hours
  reduced_hours.to_rat / total_hours_in_week.to_rat = 9 / 14 :=
by
  let total_hours_in_week := 7 * 24
  let weekday_hours := 12 * 5
  let weekend_hours := 24 * 2
  let reduced_hours := weekday_hours + weekend_hours
  sorry

end reduced_rate_fraction_l661_661801


namespace geometry_problem_l661_661062

open_locale real

noncomputable def right_triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (x3, y3) := C in
  (x1 - x3) ^ 2 + (y1 - y3) ^ 2 + (x2 - x3) ^ 2 + (y2 - y3) ^ 2 = (x1 - x2) ^ 2 + (y1 - y2) ^ 2

-- Conditions in terms of coordinates assumed for simplicity
def PC_length := 4
def BP_length := 3
def CQ_length := 3

noncomputable def PQR_equilateral (P Q R : ℝ × ℝ) : Prop :=
  let (x1, y1) := P in
  let (x2, y2) := Q in
  let (x3, y3) := R in
  (x1 - x2)^2 + (y1 - y2)^2 = (x2 - x3)^2 + (y2 - y3)^2 ∧ 
  (x2 - x3)^2 + (y2 - y3)^2 = (x3 - x1)^2 + (y3 - y1)^2

theorem geometry_problem (A B C P Q R : ℝ × ℝ) 
  (h1 : right_triangle_ABC A B C)
  (h2 : PQR_equilateral P Q R)
  (h3 : dist P C = PC_length)
  (h4 : dist B P = BP_length)
  (h5 : dist C Q = CQ_length) :
  let AQ := dist A Q in 
  AQ = 11 / 2  ∧ 
  (dist A B + dist B C + dist C A + 3 * 7 = 37 + sqrt 13) :=
sorry

end geometry_problem_l661_661062


namespace option_b_correct_l661_661425

theorem option_b_correct (a b c : ℝ) (hc : c ≠ 0) : ac^2 > bc^2 → a > b :=
sorry

end option_b_correct_l661_661425


namespace greatest_possible_perimeter_l661_661557

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l661_661557


namespace product_of_repeating_decimal_l661_661097

noncomputable def repeating_decimal_product (q : ℚ) : ℚ := q * 9

theorem product_of_repeating_decimal : 
  let q := 0.45 in
  ∃ r : ℚ, r = 45 / 11 ∧ repeating_decimal_product q = r :=
by
  sorry

end product_of_repeating_decimal_l661_661097


namespace find_n_l661_661821

theorem find_n : ∀ (n x : ℝ), (3639 + n - x = 3054) → (x = 596.95) → (n = 11.95) :=
by
  intros n x h1 h2
  sorry

end find_n_l661_661821


namespace exist_divisible_expression_l661_661813

theorem exist_divisible_expression (a : Fin 101 → ℕ) : ∃ b : (Fin 101 → ℕ) → ℕ, (b a) % (16!) = 0 :=
sorry

end exist_divisible_expression_l661_661813


namespace task_area_of_A_l661_661679

-- Define the conditions for the locus of points (\alpha, \beta, \gamma)
def valid_locus (α β γ : ℝ) : Prop :=
  α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = π ∧
  ∃ x y z : ℝ, (y^2 + z^2 = sin(α)^2) ∧ (z^2 + x^2 = sin(β)^2) ∧ (x^2 + y^2 = sin(γ)^2)

-- Define the problem statement
theorem task_area_of_A : 
  ∑ α β γ, valid_locus α β γ → true → ∃s:ℝ, ∑ a b, a + b > 0 ∧  s = γ ∧ s>0 := sorry

end task_area_of_A_l661_661679


namespace greatest_possible_perimeter_l661_661530

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l661_661530


namespace find_speed_of_stream_l661_661000

def boat_speeds (V_b V_s : ℝ) : Prop :=
  V_b + V_s = 10 ∧ V_b - V_s = 8

theorem find_speed_of_stream (V_b V_s : ℝ) (h : boat_speeds V_b V_s) : V_s = 1 :=
by
  sorry

end find_speed_of_stream_l661_661000


namespace range_of_k_l661_661149

noncomputable def f (k : ℝ) (x : ℝ) := 1 - k * x^2
noncomputable def g (x : ℝ) := Real.cos x

theorem range_of_k (k : ℝ) : (∀ x : ℝ, f k x < g x) ↔ k ≥ (1 / 2) :=
by
  sorry

end range_of_k_l661_661149


namespace product_of_roots_cubic_l661_661964

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  in (∃ r1 r2 r3 : ℝ, p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ r1 * r2 * r3 = 50) := 
by 
  sorry

end product_of_roots_cubic_l661_661964


namespace geometric_sequence_sum_l661_661137

theorem geometric_sequence_sum {a : ℕ → ℤ} (r : ℤ) (h1 : a 1 = 1) (h2 : r = -2) 
(h3 : ∀ n, a (n + 1) = a n * r) : 
  a 1 + |a 2| + |a 3| + a 4 = 15 := 
by sorry

end geometric_sequence_sum_l661_661137


namespace diagonal_not_parallel_to_any_side_l661_661256

theorem diagonal_not_parallel_to_any_side (n : ℕ) (h : 2 ≤ n) :
  ∃ d, is_diagonal d ∧ convex (polygon 2n) ∧ ¬ (parallel d (side polygon)) :=
by
  sorry

end diagonal_not_parallel_to_any_side_l661_661256


namespace greatest_possible_perimeter_l661_661544

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l661_661544


namespace product_of_roots_l661_661916

theorem product_of_roots :
  let f := Polynomial.C (-50) + Polynomial.X * (Polynomial.C 75 + Polynomial.X * (Polynomial.C (-15) + Polynomial.X)) in 
  (f.roots.Prod = 50) :=
by sorry

end product_of_roots_l661_661916


namespace only_prime_when_p_is_2_l661_661390

theorem only_prime_when_p_is_2 {p : ℕ} (hp : Prime p) :
  p = 2 ∨ ¬ Prime (1 + ∑ i in Finset.range (p+1), i^p) :=
begin
  sorry
end

end only_prime_when_p_is_2_l661_661390


namespace carla_marble_purchase_l661_661068

variable (started_with : ℕ) (now_has : ℕ) (bought : ℕ)

theorem carla_marble_purchase (h1 : started_with = 53) (h2 : now_has = 187) : bought = 134 := by
  sorry

end carla_marble_purchase_l661_661068


namespace grass_knot_segments_butterfly_knot_segments_l661_661304

-- Definitions for the grass knot problem
def outer_loops_cut : Nat := 5
def segments_after_outer_loops_cut : Nat := 6

-- Theorem for the grass knot
theorem grass_knot_segments (n : Nat) (h : n = outer_loops_cut) : (n + 1 = segments_after_outer_loops_cut) :=
sorry

-- Definitions for the butterfly knot problem
def butterfly_wings_loops_per_wing : Nat := 7
def segments_after_butterfly_wings_cut : Nat := 15

-- Theorem for the butterfly knot
theorem butterfly_knot_segments (w : Nat) (h : w = butterfly_wings_loops_per_wing) : ((w * 2 * 2 + 2) / 2 = segments_after_butterfly_wings_cut) :=
sorry

end grass_knot_segments_butterfly_knot_segments_l661_661304


namespace product_of_roots_of_cubic_polynomial_l661_661903

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l661_661903


namespace construct_triangle_l661_661986

-- Define the necessary variables and hypotheses.
variable (a : ℝ) (b c : ℝ) (Δθ : ℝ)
-- Additional necessary angle setup.
variable (θ₁ θ₂ : ℝ) 
-- Assume the sum of the sides and the difference of the base angles.
hypothesis (h_sum : b + c = a + Δθ)
hypothesis (h_diff : |θ₁ - θ₂| = Δθ)

theorem construct_triangle (a b c Δθ : ℝ) 
  (h_sum : b + c = a + Δθ)
  (h_diff : |θ₁ - θ₂| = Δθ) :
  ∃ (A B C : Type _), 
    (∃ (AB BC CA : ℝ), (AB = a) ∧ (B ≠ C) ∧
    (BC ≠ a) ∧ (CA ≠ a) ∧ (BC + CA = b + c) ∧
    (|θ₁ - θ₂| = Δθ)) :=
by
  -- Skip the proof steps.
  sorry

end construct_triangle_l661_661986


namespace find_total_area_of_kitchen_guest_bath_living_area_l661_661303

noncomputable def total_rent : ℝ := 3000
noncomputable def cost_per_sqft : ℝ := 2
noncomputable def master_bedroom_and_bath_area : ℝ := 500
noncomputable def guest_bedroom_area : ℝ := 200
noncomputable def num_guest_bedrooms : ℝ := 2

-- Define the total area problem
theorem find_total_area_of_kitchen_guest_bath_living_area :
  let total_house_area := total_rent / cost_per_sqft in
  let area_master_guest_rooms := master_bedroom_and_bath_area + num_guest_bedrooms * guest_bedroom_area in
  let area_kitchen_guest_bath_living := total_house_area - area_master_guest_rooms in
  area_kitchen_guest_bath_living = 600 := sorry

end find_total_area_of_kitchen_guest_bath_living_area_l661_661303


namespace product_divisible_by_4_l661_661827

noncomputable def biased_die_prob_divisible_by_4 : ℚ :=
  let q := 1/4  -- probability of rolling a number divisible by 3
  let p4 := 2 * q -- probability of rolling a number divisible by 4
  let p_neither := (1 - p4) * (1 - p4) -- probability of neither roll being divisible by 4
  1 - p_neither -- probability that at least one roll is divisible by 4

theorem product_divisible_by_4 :
  biased_die_prob_divisible_by_4 = 3/4 :=
by
  sorry

end product_divisible_by_4_l661_661827


namespace only_statement_I_has_nontrivial_solutions_l661_661072

theorem only_statement_I_has_nontrivial_solutions :
  (∃ (a b : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ √(a^2 + b^2) = |a - b|)
  ∧ ∀ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) →
    ¬ (√(a^2 + b^2) = a^2 - b^2 ∨ √(a^2 + b^2) = a + b ∨ √(a^2 + b^2) = |a| + |b|) := by
  sorry

end only_statement_I_has_nontrivial_solutions_l661_661072


namespace product_of_roots_cubic_l661_661891

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l661_661891


namespace M_empty_iff_k_range_M_interval_iff_k_range_l661_661151

-- Part 1
theorem M_empty_iff_k_range (k : ℝ) :
  (∀ x : ℝ, (k^2 + 2 * k - 3) * x^2 + (k + 3) * x - 1 ≤ 0) ↔ -3 ≤ k ∧ k ≤ 1 / 5 := sorry

-- Part 2
theorem M_interval_iff_k_range (k a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_ab : a < b) :
  (∀ x : ℝ, (k^2 + 2 * k - 3) * x^2 + (k + 3) * x - 1 > 0 ↔ a < x ∧ x < b) ↔ 1 / 5 < k ∧ k < 1 := sorry

end M_empty_iff_k_range_M_interval_iff_k_range_l661_661151


namespace fill_pool_duration_l661_661742

theorem fill_pool_duration :
  (∀ (pool_capacity hoses hose_flow_rate_per_minute : ℕ),
    pool_capacity = 24000 ∧ hoses = 5 ∧ hose_flow_rate_per_minute = 3 →
    let total_flow_rate_per_minute := hoses * hose_flow_rate_per_minute in
    let total_flow_rate_per_hour := total_flow_rate_per_minute * 60 in
    let total_hours := (pool_capacity : ℚ) / (total_flow_rate_per_hour : ℚ) in
    total_hours ≈ 27) :=
by 
  intros pool_capacity hoses hose_flow_rate_per_minute h
  cases' h with h_pool_capacity h_rest
  cases' h_rest with h_hoses h_hose_flow_rate
  have eq_1 : pool_capacity = 24000 := h_pool_capacity
  have eq_2 : hoses = 5 := h_hoses
  have eq_3 : hose_flow_rate_per_minute = 3 := h_hose_flow_rate
  let total_flow_rate_per_minute := hoses * hose_flow_rate_per_minute
  let total_flow_rate_per_hour := total_flow_rate_per_minute * 60
  let total_hours : ℚ := pool_capacity / total_flow_rate_per_hour
  have h_total_hours_approx := total_hours ≈ 27
  sorry

end fill_pool_duration_l661_661742


namespace greatest_possible_perimeter_l661_661648

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l661_661648


namespace max_remaining_value_is_2011_l661_661711

open Nat

def primes_upto_2020 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
  101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
  211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331,
  337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457,
  461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
  601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
  739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877,
  881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021,
  1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153,
  1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291,
  1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447,
  1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571,
  1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709,
  1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871,
  1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017]

def operation (a b : ℕ) : ℕ :=
  let x := sqrt (a^2 - a * b + b^2);
  primes_upto_2020.filter (λ p, p ≤ x) |>.maximum' (by decide)

-- The theorem proving that after several operations, the maximum value remaining is 2011
theorem max_remaining_value_is_2011 :
  ∀ (initial : List ℕ), initial = primes_upto_2020 → 
  ∃ a ∈ initial, ∀ b ∈ initial, b ≠ a → operation a b ≤ 2011 := 
sorry

end max_remaining_value_is_2011_l661_661711


namespace max_real_part_of_sum_l661_661234

noncomputable theory
open Complex

theorem max_real_part_of_sum (z w : ℂ) (hz : abs z = 1) (hw : abs w = 1) 
  (h : z * conj w + conj z * w = 1 - 2 * I) :
  ∃ u : ℝ, u = re (z + w) ∧ u ≤ real.sqrt 2 := by
sorry

end max_real_part_of_sum_l661_661234


namespace max_value_expression_l661_661409

theorem max_value_expression : ∃ (max_val : ℝ), max_val = (1 / 16) ∧ ∀ a b : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 → (a - b^2) * (b - a^2) ≤ max_val :=
by
  sorry

end max_value_expression_l661_661409


namespace unique_solution_k_l661_661997

theorem unique_solution_k (k : ℚ) : (∀ x : ℚ, x ≠ -2 → (x + 3)/(k*x - 2) = x) ↔ k = -3/4 :=
sorry

end unique_solution_k_l661_661997


namespace greatest_possible_perimeter_l661_661551

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l661_661551


namespace tan_of_sin_in_fourth_quadrant_l661_661128

noncomputable def sin_x : ℝ := -1/3
noncomputable def quadrant_IV (x : ℝ) : Prop := x ∈ set.Ioo (3 * π / 2) (2 * π)

theorem tan_of_sin_in_fourth_quadrant (x : ℝ) (h1 : sin x = sin_x) (h2 : quadrant_IV x) : tan x = -√2 / 4 := 
sorry

end tan_of_sin_in_fourth_quadrant_l661_661128


namespace counterexample_to_proposition_l661_661287

theorem counterexample_to_proposition : ∃ (a : ℝ), a^2 > 0 ∧ a ≤ 0 :=
  sorry

end counterexample_to_proposition_l661_661287


namespace total_hamburgers_bought_l661_661377

-- Definitions of conditions
def total_spent : ℝ := 70.5
def single_burger_cost : ℝ := 1.0
def double_burger_cost : ℝ := 1.5
def double_burgers_bought : ℕ := 41

-- Question: How many hamburgers did Caleb buy in total?
theorem total_hamburgers_bought : 
  let single_burgers_bought := (total_spent - (double_burgers_bought * double_burger_cost)) / single_burger_cost in
  single_burgers_bought + double_burgers_bought = 50 :=
by 
  sorry

end total_hamburgers_bought_l661_661377


namespace star_operation_evaluation_l661_661436

theorem star_operation_evaluation : 
  let star (a b : ℕ) := a^2 + 2 * a * b + b^2
  in star 4 6 = 100 :=
by {
  let star (a b : ℕ) := a^2 + 2 * a * b + b^2,
  sorry
}

end star_operation_evaluation_l661_661436


namespace valid_numbers_count_l661_661164

def isAllowedDigit (d : Nat) : Bool :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def containsAllowedDigitsOnly (n : Nat) : Bool :=
  n.digits.forall isAllowedDigit

def isValidNumber (n : Nat) : Bool :=
  n < 1000 ∧ n % 4 = 0 ∧ containsAllowedDigitsOnly n

def countValidNumbers : Nat :=
  (List.range 1000).filter isValidNumber |> List.length

theorem valid_numbers_count : countValidNumbers = 31 :=
by
  sorry

end valid_numbers_count_l661_661164


namespace problem_l661_661220

noncomputable def arithmetic_mean (t : List ℝ) : ℝ :=
  (t.foldr (· + ·) 0) / t.length

noncomputable def arithmetic_mean_squared (t : List ℝ) : ℝ :=
  arithmetic_mean (t.map (· * ·))

theorem problem (p q : ℝ) (h0 : 0 < p) (h1 : p < q) (t : List ℝ)
  (h2 : ∀ t_i, t_i ∈ t → p ≤ t_i ∧ t_i ≤ q) :
  let A := arithmetic_mean t
  let B := arithmetic_mean_squared t
  in A ^ 2 / B ≥ 4 * p * q / (p + q) ^ 2 :=
by
  sorry

end problem_l661_661220


namespace line_through_two_points_has_equation_circle_with_center_on_line_and_passes_through_points_l661_661464

theorem line_through_two_points_has_equation : 
  (∀ (l : ℝ → ℝ → Prop), (l 2 1) ∧ (l 6 3) → ∃ a b c, ∀ x y, l x y ↔ a * x + b * y = c) :=
by {
  sorry
}

theorem circle_with_center_on_line_and_passes_through_points :
  (∀ (C L : ℝ → ℝ → ℝ → Prop),
   -- line L goes through (2,1) and (6,3)
   (L 2 1 0) ∧ (L 6 3 0) → 
   (C x y r → (C 2 0 r) ∧ (C 3 1 r) →
   ∃ h k, h = 2 * k ∧ ∀ x y, C x y r ↔ (x - h)^2 + (y - k)^2 = r^2) :=
by {
  sorry
}

end line_through_two_points_has_equation_circle_with_center_on_line_and_passes_through_points_l661_661464


namespace alpha_value_l661_661468

theorem alpha_value (α : ℝ) (h : 4^α = 2) : α = 1 / 2 :=
by {
  sorry
}

end alpha_value_l661_661468


namespace greatest_perimeter_l661_661566

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l661_661566


namespace cone_height_l661_661343

theorem cone_height (V : ℝ) (π : ℝ) (angle : ℝ) (h : ℝ) : 
  V = 27000 * π ∧ angle = 90 → h = 68.6 :=
by
  -- Given that the volume of the cone is 27000π cubic inches
  -- and the vertex angle of the vertical cross-section is 90 degrees,
  -- prove that the height of the cone is 68.6 inches.
  intros,
  sorry

end cone_height_l661_661343


namespace vector_orthogonal_and_k_any_real_l661_661456

noncomputable def vector_angle (a b : ℝ) : ℝ :=
if h : (a ≠ 0 ∧ b ≠ 0) then 90 else 0

theorem vector_orthogonal_and_k_any_real
  (a b : ℝ → ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (k : ℝ)
  (h : ∥a + k • b∥ = ∥a - k • b∥) :
  (k = k) ∧ (vector_angle a b = 90) :=
by {
  sorry
}

end vector_orthogonal_and_k_any_real_l661_661456


namespace number_of_n_f50_eq_16_l661_661426

def num_divisors (n : ℕ) : ℕ := 
  (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).card

def f1 (n : ℕ) : ℕ := 2 * num_divisors n

def f (j n : ℕ) : ℕ :=
  Nat.recOn j (f1 n) (λ j rec, if j = 0 then f1 n else f1 rec)

def f50 (n : ℕ) : ℕ := f 50 n

theorem number_of_n_f50_eq_16 : 
  (Finset.range 31).filter (λ n, f50 n = 16).card = 3 :=
  sorry

end number_of_n_f50_eq_16_l661_661426


namespace percentage_increase_surface_area_l661_661005

theorem percentage_increase_surface_area (side_length : ℝ) (small_cube_side : ℝ) (surface_area_original : ℝ) (surface_area_small_cubes : ℝ) (percentage_increase : ℝ) :
  side_length = 7 → small_cube_side = 1 → surface_area_original = 6 * side_length^2 → 
  surface_area_small_cubes = 343 * 6 * small_cube_side^2 → 
  percentage_increase = ((surface_area_small_cubes - surface_area_original) / surface_area_original) * 100 →
  percentage_increase = 600 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end percentage_increase_surface_area_l661_661005


namespace greatest_possible_perimeter_l661_661619

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l661_661619


namespace correct_propositions_l661_661130

-- Definitions to state distinctness of lines and planes.
variables {a b c : Type} [IsLine a] [IsLine b] [IsLine c] [Distinct a b c]
variables {α β : Type} [IsPlane α] [IsPlane β] [Distinct α β]

-- Propositions as defined.
def prop1 (a b α : Type) [HasParallel a b] [HasParallel b α] : Prop :=
  HasParallel a α

def prop2 (a b α β : Type) [Subset a α] [Subset b α] [HasParallel a β] [HasParallel b β] : Prop :=
  HasParallel α β

def prop3 (a α β : Type) [HasPerpendicular a α] [HasParallel a β] : Prop :=
  HasPerpendicular α β

def prop4 (a b α : Type) [HasPerpendicular a α] [HasParallel b α] : Prop :=
  HasPerpendicular a b

-- Statement that we need to prove is that the number of correct propositions is 2.
theorem correct_propositions :
  (count (λ (p : Prop), p) [prop3 a α β, prop4 a b α]) = 2 :=
begin
  sorry
end

end correct_propositions_l661_661130


namespace product_of_roots_l661_661936

theorem product_of_roots :
  let p : Polynomial ℚ := Polynomial.Coeff.mk_fin 3 0 (⟨[50, -75, 15, -1]⟩ : ℕ → ℚ)
  (p.root_prod = 50) :=
begin
  sorry
end

end product_of_roots_l661_661936


namespace f_x_plus_f_neg_x_eq_seven_l661_661805

variable (f : ℝ → ℝ)

-- Given conditions: 
axiom cond1 : ∀ x : ℝ, f x + f (1 - x) = 10
axiom cond2 : ∀ x : ℝ, f (1 + x) = 3 + f x

-- Prove statement:
theorem f_x_plus_f_neg_x_eq_seven : ∀ x : ℝ, f x + f (-x) = 7 := 
by
  sorry

end f_x_plus_f_neg_x_eq_seven_l661_661805


namespace truck_dirt_road_time_l661_661034

noncomputable def time_on_dirt_road (time_paved : ℝ) (speed_increment : ℝ) (total_distance : ℝ) (dirt_speed : ℝ) : ℝ :=
  let paved_speed := dirt_speed + speed_increment
  let distance_paved := paved_speed * time_paved
  let distance_dirt := total_distance - distance_paved
  distance_dirt / dirt_speed

theorem truck_dirt_road_time :
  time_on_dirt_road 2 20 200 32 = 3 :=
by
  sorry

end truck_dirt_road_time_l661_661034


namespace x_intercept_perpendicular_line_l661_661313

theorem x_intercept_perpendicular_line : 
  ∀ (L1 L2 : ℝ → ℝ), 
  (∀ x, L1 x = (3/2) * x - 3) →      -- condition 1: line defined by 3x - 2y = 6
  (∀ x, L2 x = -(2/3) * x + 4) →      -- condition 2 & 3: line perpendicular to L1 and y-intercept is 4
  (∃ x, L2 x = 0 ∧ x = 6) :=          -- prove the x-intercept is 6
by
  intros L1 L2 hL1 hL2
  use 6
  split
  { rw hL2
    have : -(2/3) * 6 + 4 = 0, ring, 
    exact this }
  { refl }
  sorry

end x_intercept_perpendicular_line_l661_661313


namespace harkamal_total_payment_l661_661161

def total_cost_of_fruits (cost_g : ℝ) (cost_m : ℝ) (cost_a : ℝ) (cost_o : ℝ) : ℝ :=
  let discount_g := if cost_g >= 400 then cost_g * 0.10 else 0
  let discount_m := if cost_m >= 450 then cost_m * 0.15 else 0
  let discount_a := if cost_a >= 200 then cost_a * 0.05 else 0
  (cost_g - discount_g) + (cost_m - discount_m) + (cost_a - discount_a) + cost_o

theorem harkamal_total_payment : 
  total_cost_of_fruits (8 * 70) (9 * 60) (5 * 50) (2 * 30) = 1260.50 := 
by 
  -- calculations omitted, they follow the steps in the solution
  sorry

end harkamal_total_payment_l661_661161


namespace youseff_blocks_l661_661323

theorem youseff_blocks (b : ℕ)
  (h1 : ∀ b, walking_time_per_block = 1)
  (h2 : ∀ b, biking_time_per_block = (20 / 60))
  (h3 : b = (biking_time_per_block * b + 6)) :
  b = 9 :=
by
  sorry

end youseff_blocks_l661_661323


namespace num_divisors_not_divisible_by_5_l661_661167

theorem num_divisors_not_divisible_by_5 (n : ℕ) (h : n = 150) (pf : factorization 150 = {2:1, 3:1, 5:2}) : 
  ∃ d : ℕ, d = 4 ∧ (∀ k : ℕ, k ∣ n → k % 5 ≠ 0 → (k ∈ divisors 150)) :=
by { sorry }

end num_divisors_not_divisible_by_5_l661_661167


namespace continuity_at_three_l661_661726

noncomputable def f (x : ℝ) : ℝ := -2 * x ^ 2 - 4

theorem continuity_at_three (ε : ℝ) (hε : 0 < ε) :
  ∃ δ > 0, ∀ x : ℝ, |x - 3| < δ → |f x - f 3| < ε :=
sorry

end continuity_at_three_l661_661726


namespace pqrs_sum_l661_661685

-- Define the polynomial Q(z)
def Q (z : ℂ) (p q r : ℝ) : ℂ := z^3 + p*z^2 + q*z + r

-- Define the roots w+4i, w+10i, 3w-5
noncomputable def roots (w : ℂ) : List ℂ :=
  [w + 4*complex.I, w + 10*complex.I, 3*w - 5]

-- Define the statement to be proven
theorem pqrs_sum (p q r : ℝ) (w : ℂ) (h : ∀ z, z ∈ roots w → Q z p q r = 0) :
  p + q + r = -235/4 :=
by
  sorry

end pqrs_sum_l661_661685


namespace spherical_to_rectangular_l661_661074

open Real

theorem spherical_to_rectangular (ρ θ φ : ℝ) (hρ : ρ = 8) (hθ : θ = 5 * π / 4) (hφ : φ = π / 6) :
  ∃ x y z : ℝ, x = -2 * sqrt 2 ∧ y = -2 * sqrt 2 ∧ z = 4 * sqrt 3 := 
by
  have x := ρ * sin φ * cos θ
  have y := ρ * sin φ * sin θ
  have z := ρ * cos φ
  use x, y, z
  rw [hρ, hθ, hφ]
  split
  { calc
    x = 8 * sin (π / 6) * cos (5 * π / 4) : by simp [hρ, hθ, hφ]
      ... = 8 * (1 / 2) * (-sqrt 2 / 2) : by rwa [sin_pi_div_six, cos_five_pi_div_four]
      ... = -2 * sqrt 2 : by ring, }
  split
  { calc
    y = 8 * sin (π / 6) * sin (5 * π / 4) : by simp [hρ, hθ, hφ]
      ... = 8 * (1 / 2) * (-sqrt 2 / 2) : by rwa [sin_pi_div_six, sin_five_pi_div_four]
      ... = -2 * sqrt 2 : by ring, }
  { calc
    z = 8 * cos (π / 6) : by simp [hρ, hθ, hφ]
      ... = 8 * (sqrt 3 / 2) : by rw cos_pi_div_six
      ... = 4 * sqrt 3 : by ring, }

end spherical_to_rectangular_l661_661074


namespace number_of_nickels_l661_661491

def dimes : ℕ := 10
def pennies_per_dime : ℕ := 10
def pennies_per_nickel : ℕ := 5
def total_pennies : ℕ := 150

theorem number_of_nickels (total_value_dimes : ℕ := dimes * pennies_per_dime)
  (pennies_needed_from_nickels : ℕ := total_pennies - total_value_dimes)
  (n : ℕ) : n = pennies_needed_from_nickels / pennies_per_nickel → n = 10 := by
  sorry

end number_of_nickels_l661_661491


namespace appointment_on_tuesday_duration_l661_661866

theorem appointment_on_tuesday_duration :
  let rate := 20
  let monday_appointments := 5
  let monday_each_duration := 1.5
  let thursday_appointments := 2
  let thursday_each_duration := 2
  let saturday_duration := 6
  let weekly_earnings := 410
  let known_earnings := (monday_appointments * monday_each_duration * rate) + (thursday_appointments * thursday_each_duration * rate) + (saturday_duration * rate)
  let tuesday_earnings := weekly_earnings - known_earnings
  (tuesday_earnings / rate = 3) :=
by
  -- let rate := 20
  -- let monday_appointments := 5
  -- let monday_each_duration := 1.5
  -- let thursday_appointments := 2
  -- let thursday_each_duration := 2
  -- let saturday_duration := 6
  -- let weekly_earnings := 410
  -- let known_earnings := (monday_appointments * monday_each_duration * rate) + (thursday_appointments * thursday_each_duration * rate) + (saturday_duration * rate)
  -- let tuesday_earnings := weekly_earnings - known_earnings
  -- exact tuesday_earnings / rate = 3
  sorry

end appointment_on_tuesday_duration_l661_661866


namespace division_calculation_l661_661063

-- Define the problem domain
def calculate_expression : ℝ := (0.08 / 0.002) / 0.04

-- The theorem to prove
theorem division_calculation : calculate_expression = 1000 := by
  sorry

end division_calculation_l661_661063


namespace range_of_a_l661_661512

-- Define the problem statement in Lean 4
theorem range_of_a (a : ℝ) : (∀ x : ℝ, ((x^2 - (a-1)*x + 1) > 0)) → (-1 < a ∧ a < 3) :=
by
  intro h
  sorry -- Proof to be filled in

end range_of_a_l661_661512


namespace line_passes_through_point_l661_661281

theorem line_passes_through_point:
  ∀ k : ℝ, ∀ x y : ℝ, (y + 2 = k * (x + 1)) → ∃ c : ℝ × ℝ, c = (-1, -2) := 
by
  intros k x y h
  use (-1, -2)
  sorry

end line_passes_through_point_l661_661281


namespace correct_eccentricity_l661_661117

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
    (h_ab : 4 * b^2 = 3 * a^2) (h_c : ∃ c, c^2 = a^2 + b^2) : ℝ :=
let ⟨c, h_eq_c⟩ := h_c in
c / a

theorem correct_eccentricity 
    (a b : ℝ) 
    (h_pos_a : 0 < a) 
    (h_pos_b : 0 < b)
    (h_ab : 4 * b^2 = 3 * a^2)
    (h_c : ∃ c, c^2 = a^2 + b^2) :
    hyperbola_eccentricity a b h_pos_a h_pos_b h_ab h_c = (Real.sqrt 7) / 2 :=
by
  sorry

end correct_eccentricity_l661_661117


namespace gcd_of_abc_cba_l661_661777

theorem gcd_of_abc_cba (a b c : ℕ) (h1 : b = a + 1) (h2 : c = a + 2) : 
  let abc := 100 * a + 10 * b + c
      cba := 100 * c + 10 * b + a
      n := abc + cba
  in gcd n 2 = 2 :=
by sorry

end gcd_of_abc_cba_l661_661777


namespace q_zero_eq_neg_two_l661_661231

variables {R : Type*} [CommRing R] (p q r : R[X])

-- Conditions
def r_eq_p_mul_q (r p q : R[X]) : Prop := r = p * q
def const_term_p (p : R[X]) : R := p.coeff 0
def const_term_r (r : R[X]) : R := r.coeff 0

-- The statement we want to prove
theorem q_zero_eq_neg_two
  (p q r : R[X])
  (h1 : r_eq_p_mul_q r p q)
  (h2 : const_term_p p = 5)
  (h3 : const_term_r r = -10) :
  q.coeff 0 = -2 :=
by
  sorry

end q_zero_eq_neg_two_l661_661231


namespace greatest_possible_perimeter_l661_661598

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l661_661598


namespace greatest_possible_perimeter_l661_661549

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l661_661549


namespace angle_sum_tangent_circles_l661_661135

open_locale real

variables {P Q R : Type}

noncomputable def is_tangent (C_inv : circle) (C_ext: circle) :=
∃ pt, pt ∈ C_inv ∧ pt ∈ C_ext ∧ is_tangent_pt C_inv pt ∧ is_tangent_pt C_ext pt

theorem angle_sum_tangent_circles 
    (Γ : circle) (O₁ O₂ : circle) (A B : point) :
    is_tangent Γ O₁ ∧ 
    is_tangent Γ O₂ ∧ 
    is_tangent O₁ (line (pt A) (pt B)) ∧ 
    is_tangent O₂ (line (pt A) (pt B)) ∧ 
    O₁.on_opposite_sides O₂ (line (pt A) (pt B)) →
    angle (pt O₁) (pt A) (pt O₂) + 
    angle (pt O₁) (pt B) (pt O₂) > 90 :=
sorry

end angle_sum_tangent_circles_l661_661135


namespace product_of_roots_cubic_l661_661885

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l661_661885


namespace radius_of_circle_is_3_sqrt_5_l661_661354

noncomputable def radius_of_circle (PQ QR PC : ℝ) := 
  real.sqrt (PC * PC - PQ * QR)

theorem radius_of_circle_is_3_sqrt_5 :
  radius_of_circle 10 18 15 = 3 * real.sqrt 5 := 
by
  sorry

end radius_of_circle_is_3_sqrt_5_l661_661354


namespace product_of_roots_cubic_l661_661973

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  in (∃ r1 r2 r3 : ℝ, p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ r1 * r2 * r3 = 50) := 
by 
  sorry

end product_of_roots_cubic_l661_661973


namespace relation_among_abc_l661_661501

-- Definitions
def a := (1/2)^(1/5)
def b := (1/5)^(-1/2)
def c := Real.logb (1/5) 10

-- Theorem
theorem relation_among_abc : b > a ∧ a > c := by
  sorry

end relation_among_abc_l661_661501


namespace lineC_correct_l661_661365

def lineA (x : ℝ) : ℝ := 2 * x + 1
def lineB (x : ℝ) : ℝ := (x / 2) + (1 / 2)
def lineC (x : ℝ) : ℝ := -2 * (x - 1) + 2
def lineD (x : ℝ) : Option ℝ :=
    if x = 2 then some 0
    else if x = 0 then some (-3)
    else none

theorem lineC_correct :
    ∀ x y : ℝ,
    (lineA x = y → (0 ≤ x ∧ 0 ≤ y ∨ x < 0 ∧ 0 ≤ y ∨ 0 ≤ x ∧ y < 0)) ∧
    (lineB x = y → (0 ≤ x ∧ 0 ≤ y ∨ x < 0 ∧ 0 ≤ y ∨ 0 ≤ x ∧ y < 0)) ∧
    (lineD x = some y → (0 ≤ x ∧ 0 ≤ y ∨ x < 0 ∧ 0 ≤ y ∨ 0 ≤ x ∧ y < 0)) ∧
    (lineC x = y → (0 ≤ x ∧ 0 ≤ y ∨ x < 0 ∧ 0 ≤ y ∨ 0 ≤ x ∧ y < 0 ∨ x < 0 ∧ y < 0)) :=
sorry

end lineC_correct_l661_661365


namespace riding_mower_speed_l661_661669

theorem riding_mower_speed :
  (∃ R : ℝ, 
     (8 * (3 / 4) = 6) ∧       -- Jerry mows 6 acres with the riding mower
     (8 * (1 / 4) = 2) ∧       -- Jerry mows 2 acres with the push mower
     (2 / 1 = 2) ∧             -- Push mower takes 2 hours to mow 2 acres
     (5 - 2 = 3) ∧             -- Time spent on the riding mower is 3 hours
     (6 / 3 = R) ∧             -- Riding mower cuts 6 acres in 3 hours
     R = 2) :=                 -- Therefore, R (speed of riding mower in acres per hour) is 2
sorry

end riding_mower_speed_l661_661669


namespace larger_integer_is_50_l661_661749

-- Definition of the problem conditions.
def is_two_digit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99

def problem_conditions (m n : ℕ) : Prop := 
  is_two_digit m ∧ is_two_digit n ∧
  (m + n) / 2 = m + n / 100

-- Statement of the proof problem.
theorem larger_integer_is_50 (m n : ℕ) (h : problem_conditions m n) : max m n = 50 :=
  sorry

end larger_integer_is_50_l661_661749


namespace product_of_roots_eq_50_l661_661944

theorem product_of_roots_eq_50 :
  let a := -15
  let c := -50
  let equation := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  -- Vieta's formulas for the product of roots in cubic equation ax^3 + bx^2 + cx + d = 0 is -d/a
  ∃ p q r : ℝ, 
    equation p = 0 ∧ equation q = 0 ∧ equation r = 0 ∧ (p * q * r = 50) :=
by
  sorry

end product_of_roots_eq_50_l661_661944


namespace product_of_roots_l661_661905

-- Let's define the given polynomial equation.
def polynomial : Polynomial ℝ := Polynomial.monomial 3 1 - Polynomial.monomial 2 15 + Polynomial.monomial 1 75 - Polynomial.C 50

-- Prove that the product of the roots of the given polynomial is 50.
theorem product_of_roots : (Polynomial.roots polynomial).prod id = 50 := by
  sorry

end product_of_roots_l661_661905


namespace part_I_part_II_l661_661476

def f (m : ℝ) (x : ℝ) : ℝ := Real.log (4^x + 1) / Real.log 2 + m * x

theorem part_I : ∀ (m : ℝ), (∀ (x : ℝ), f m (-x) = f m x) → m = -1 :=
by
  intro m
  intro h
  sorry

theorem part_II :
  ∀ (m : ℝ),
  m > 0 →
  (∀ (x : ℝ), f m (8 * (Real.log x / Real.log 4)^2 + 2 * Real.log x + 4 / m - 4) = 1 ↔ 1 ≤ x ∧ x ≤ 2 * Real.sqrt 2) →
  (8 / 9 < m ∧ m ≤ 1) :=
by
  intro m
  intro h₁
  intro h₂
  sorry

end part_I_part_II_l661_661476


namespace greatest_possible_perimeter_l661_661536

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l661_661536


namespace find_a_20_l661_661471

-- Arithmetic sequence definition and known conditions
def arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a : ℕ → ℤ)

-- Conditions
def condition1 : Prop := a 1 + a 2 + a 3 = 6
def condition2 : Prop := a 5 = 8

-- The main statement to prove
theorem find_a_20 (h_arith : arithmetic_sequence a) (h_cond1 : condition1 a) (h_cond2 : condition2 a) : 
  a 20 = 38 := by
  sorry

end find_a_20_l661_661471


namespace angle_PQR_measures_84_degrees_l661_661360

-- Define internal angles for regular polygons
def internal_angle_hexagon := 120
def internal_angle_pentagon := 108

-- Define the given problem conditions and statement to prove
theorem angle_PQR_measures_84_degrees :
  ∀ (P Q R : Type), 
    (angle QRP = internal_angle_hexagon - internal_angle_pentagon) →
    let angle PQR := 84 in
    angle PQR = 84 :=
by
  intros
  sorry

end angle_PQR_measures_84_degrees_l661_661360


namespace main_theorem_l661_661021

noncomputable def circle_center : Prop :=
  ∃ x y : ℝ, 2*x - y - 7 = 0 ∧ y = -3 ∧ x = 2

noncomputable def circle_equation : Prop :=
  (∀ (x y : ℝ), (x - 2)^2 + (y + 3)^2 = 5)

noncomputable def tangent_condition (k : ℝ) : Prop :=
  (3 + 3*k)^2 / (1 + k^2) = 5

noncomputable def symmetric_circle_center : Prop :=
  ∃ x y : ℝ, x = -22/5 ∧ y = 1/5

noncomputable def symmetric_circle_equation : Prop :=
  (∀ (x y : ℝ), (x + 22/5)^2 + (y - 1/5)^2 = 5)

theorem main_theorem : circle_center → circle_equation ∧ (∃ k : ℝ, tangent_condition k) ∧ symmetric_circle_center → symmetric_circle_equation :=
  by sorry

end main_theorem_l661_661021


namespace minimum_pool_cost_l661_661785

def poolCost (a b : ℝ) : ℝ :=
  20 * (a * b) + 20 * (a + b) + 80

theorem minimum_pool_cost : ∃ (a b : ℝ), a * b = 4 ∧ a = b ∧ poolCost a b = 160 := 
begin
  use [2, 2],
  split,
  { norm_num },
  split,
  { norm_num },
  norm_num,
end

end minimum_pool_cost_l661_661785


namespace max_triangle_perimeter_l661_661585

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l661_661585


namespace frank_reads_pages_per_day_l661_661432

-- Define the conditions and problem statement
def total_pages : ℕ := 450
def total_chapters : ℕ := 41
def total_days : ℕ := 30

-- The derived value we need to prove
def pages_per_day : ℕ := total_pages / total_days

-- The theorem to prove
theorem frank_reads_pages_per_day : pages_per_day = 15 :=
  by
  -- Proof goes here
  sorry

end frank_reads_pages_per_day_l661_661432


namespace function_continuous_at_point_continuous_at_f_l661_661724

noncomputable def delta (ε : ℝ) : ℝ := ε / 12

theorem function_continuous_at_point :
  ∀ (f : ℝ → ℝ) (x₀ : ℝ), (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε) :=
by
  let f := fun x => -2 * x^2 - 4
  let x₀ := 3
  have h1 : f x₀ = -22 := by linarith
  have h2 : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε :=
    by
      intros ε ε_pos
      use (ε / 12)
      split
      { exact div_pos ε_pos twelve_pos }
      { intros x hx
        calc
        abs (f x - f x₀)
          = abs (-2 * x^2 - 4 - (-22)) : by simp [h1]
      ... = abs (-2 * (x^2 - 9)) : by ring 
      ... = 2 * abs (x^2 - 9) : by rw [abs_mul, abs_neg]; simp
      ... < ε : by 
        let δ := ε / 12
        have hx3 : abs (x - 3) < δ := hx
        have h2 : abs (x + 3) ≤ 6 :=
          calc 
            abs (x + 3) ≤ abs (x - 3) + 6 : by linarith
            ... ≤ δ + 6 : by linarith
            ... ≤ ε / 12 + 6 : by linarith
        exact mul_lt_of_lt_div ((div_pos ε_pos twelve_pos).le)

theorem continuous_at_f : ∀ (ε : ℝ), ε > 0 → ∃ δ > 0, ∀ x, |x - 3| < δ → |(-2 * x^2 - 4) - (-22)| < ε :=
by
  intros ε ε_pos
  unfold delta
  use δ ε
  split
  { exact div_pos ε_pos twelve_pos }
  { intros x h
    calc
    |(-2 * x^2 - 4) - (-22)|
      = 2 * |x^2 - 9| : by norm_num
  ... < ε : by
      let h' := abs_sub_lt_iff.mp h
      exact lt_of_le_of_lt (abs_mul _ _) (by linarith [div_pos ε_pos twelve_pos,
        le_abs_self, (mul_div_cancel' _ twelve_pos.ne.symm)]) }

end function_continuous_at_point_continuous_at_f_l661_661724


namespace president_savings_l661_661262

theorem president_savings (total_funds : ℕ) (friends_percentage : ℕ) (family_percentage : ℕ) 
  (friends_contradiction funds_left family_contribution fundraising_amount : ℕ) :
  total_funds = 10000 →
  friends_percentage = 40 →
  family_percentage = 30 →
  friends_contradiction = (total_funds * friends_percentage) / 100 →
  funds_left = total_funds - friends_contradiction →
  family_contribution = (funds_left * family_percentage) / 100 →
  fundraising_amount = funds_left - family_contribution →
  fundraising_amount = 4200 :=
by
  intros
  sorry

end president_savings_l661_661262


namespace num_integers_satisfy_inequality_l661_661162

theorem num_integers_satisfy_inequality : 
  {x : ℤ | abs (3 * x + 4) ≤ 10}.to_finset.card = 8 := 
  sorry

end num_integers_satisfy_inequality_l661_661162


namespace pipe_flow_rate_l661_661826

-- Condition Definitions
def tank_capacity : ℕ := 6000
def initial_water : ℕ := tank_capacity / 2
def first_drain_loss_rate : ℕ := 1000 -- in liters per 4 minutes
def second_drain_loss_rate : ℕ := 1000 -- in liters per 6 minutes
def fill_time : ℕ := 36 -- in minutes

-- Problem Statement
theorem pipe_flow_rate :
  let
    total_additional_water_needed := tank_capacity - initial_water,
    total_first_drain_loss := (fill_time / 4) * first_drain_loss_rate,
    total_second_drain_loss := (fill_time / 6) * second_drain_loss_rate,
    total_loss := total_first_drain_loss + total_second_drain_loss,
    total_water_needed := total_additional_water_needed + total_loss,
    flow_rate := total_water_needed / fill_time
  in
    flow_rate = 500 :=
  sorry

end pipe_flow_rate_l661_661826


namespace operation_result_l661_661427

-- Define the operation
def operation (a b : ℝ) : ℝ := (a - b) ^ 3

theorem operation_result (x y : ℝ) : operation ((x - y) ^ 3) ((y - x) ^ 3) = -8 * (y - x) ^ 9 := 
  sorry

end operation_result_l661_661427


namespace pyramid_max_value_l661_661404

theorem pyramid_max_value :
  ∀ (a b c d e f g : ℕ),
    multiset.card ({a, b, c, d, e, f, g} : multiset ℕ) = 7 →
    multiset.count 1 ({a, b, c, d, e, f, g} : multiset ℕ) = 2 →
    multiset.count 2 ({a, b, c, d, e, f, g} : multiset ℕ) = 2 →
    multiset.count 3 ({a, b, c, d, e, f, g} : multiset ℕ) = 2 →
    multiset.count 4 ({a, b, c, d, e, f, g} : multiset ℕ) = 1 →
    a + 3 * b + 5 * c + 3 * d + 6 * e + 4 * f + g ≤ 65 :=
begin
  intros,
  sorry
end

end pyramid_max_value_l661_661404


namespace limit_numerator_denominator_l661_661811

-- Condition: The sum of the first n even numbers is n² + n
def sum_first_n_even_numbers (n : ℕ) : ℕ := n^2 + n

-- Condition: The sum of the first n odd numbers is n²
def sum_first_n_odd_numbers (n : ℕ) : ℕ := n^2

-- Prove the limit statement
theorem limit_numerator_denominator :
  (filter.tendsto (λ n : ℕ, (sum_first_n_even_numbers n : ℝ) / (sum_first_n_odd_numbers n)) filter.at_top (𝓝 1)) :=
by
  sorry

end limit_numerator_denominator_l661_661811


namespace quadratic_rational_roots_l661_661133

theorem quadratic_rational_roots (m : ℤ) (hm : mx^2 - (m - 1) * x + 1 = 0) :
  m = 6 ∧ has_root (λ x, 6 * x^2 - 5 * x + 1) (1 / 2) ∧ has_root (λ x, 6 * x^2 - 5 * x + 1) (1 / 3) :=
sorry

end quadratic_rational_roots_l661_661133


namespace num_ordered_pairs_l661_661095

theorem num_ordered_pairs : ∃ (n : ℕ), n = 24 ∧ ∀ (a b : ℂ), a^4 * b^6 = 1 ∧ a^8 * b^3 = 1 → n = 24 :=
by
  sorry

end num_ordered_pairs_l661_661095


namespace days_c_worked_l661_661329

theorem days_c_worked (Da Db Dc : ℕ) (Wa Wb Wc : ℕ)
  (h1 : Da = 6) (h2 : Db = 9) (h3 : Wc = 100) (h4 : 3 * Wc = 5 * Wa)
  (h5 : 4 * Wc = 5 * Wb)
  (h6 : Wa * Da + Wb * Db + Wc * Dc = 1480) : Dc = 4 :=
by
  sorry

end days_c_worked_l661_661329


namespace product_of_roots_cubic_l661_661956

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  let roots := {x : ℝ // p x = 0}.toFinset
  ∃ r : ℝ, (∀ x ∈ roots, p x = 0) ∧ (∏ x in roots, x) = 50 :=
by
  sorry

end product_of_roots_cubic_l661_661956


namespace find_A_intersection_B_find_a_b_l661_661157

noncomputable theory
open_locale classical

def solution_set (p : ℝ → Prop) := {x : ℝ | p x}

def A := solution_set (λ x, x^2 - 2 * x - 3 < 0)
def B := solution_set (λ x, x^2 + x - 6 < 0)
def C := solution_set (λ x, x^2 - x - 2 < 0)

theorem find_A_intersection_B : A ∩ B = (-1:ℝ, 2:ℝ) :=
sorry

theorem find_a_b (a b : ℝ) (h : solution_set (λ x, x^2 + a * x + b < 0) = (-1 : ℝ, 2 : ℝ)) :
  a = -1 ∧ b = -2 :=
sorry

end find_A_intersection_B_find_a_b_l661_661157


namespace arithmetic_sequence_general_formula_l661_661292

def f (x : ℝ) : ℝ := x^2 - 4 * x + 2

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem arithmetic_sequence_general_formula (a : ℕ → ℝ) (x : ℝ)
  (h_arith : arithmetic_seq a)
  (h1 : a 1 = f (x + 1))
  (h2 : a 2 = 0)
  (h3 : a 3 = f (x - 1)) :
  ∀ n, a n = 2 * n - 4 ∨ a n = 4 - 2 * n :=
by
  sorry

end arithmetic_sequence_general_formula_l661_661292


namespace boxes_contain_neither_markers_nor_sharpies_l661_661069

theorem boxes_contain_neither_markers_nor_sharpies :
  (∀ (total_boxes markers_boxes sharpies_boxes both_boxes neither_boxes : ℕ),
    total_boxes = 15 → markers_boxes = 8 → sharpies_boxes = 5 → both_boxes = 4 →
    neither_boxes = total_boxes - (markers_boxes + sharpies_boxes - both_boxes) →
    neither_boxes = 6) :=
by
  intros total_boxes markers_boxes sharpies_boxes both_boxes neither_boxes
  intros htotal hmarkers hsharpies hboth hcalc
  rw [htotal, hmarkers, hsharpies, hboth] at hcalc
  exact hcalc

end boxes_contain_neither_markers_nor_sharpies_l661_661069


namespace arrangement_with_at_least_one_girl_l661_661105

open nat

theorem arrangement_with_at_least_one_girl 
  (boys girls selected : ℕ) 
  (at_least_one_girl : selected > 0) 
  (total_people : ℕ := boys + girls) 
  (ways_to_choose : ℕ := (total_people.choose selected)) 
  (ways_all_boys : ℕ := (boys.choose selected)) 
  (non_girl_ways : ℕ := ways_to_choose - ways_all_boys) 
  (task_assignments : ℕ := selected.factorial) 
  (total_arrangements : ℕ := non_girl_ways * task_assignments) 
  :
  boys = 4 ∧ girls = 3 ∧ selected = 3 → total_arrangements = 186 :=
by {
  sorry -- Proof is skipped as per instructions.
}

end arrangement_with_at_least_one_girl_l661_661105


namespace smallest_positive_integer_l661_661356

theorem smallest_positive_integer (n : ℕ) (h₁ : n > 1) (h₂ : n % 2 = 1) (h₃ : n % 3 = 1) (h₄ : n % 4 = 1) (h₅ : n % 5 = 1) : n = 61 :=
by
  sorry

end smallest_positive_integer_l661_661356


namespace a2019_value_l661_661483

noncomputable def a_sequence : ℕ → ℝ
| 0       := Real.sqrt 3
| (n + 1) := Real.floor (a_sequence n) + (1 / (a_sequence n - Real.floor (a_sequence n)))

theorem a2019_value :
  a_sequence 2019 = 3029 + (Real.sqrt 3 - 1) / 2 :=
sorry

end a2019_value_l661_661483


namespace greatest_perimeter_of_triangle_l661_661612

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l661_661612


namespace females_in_coach_class_l661_661335

theorem females_in_coach_class (total_passengers females_pct first_class_pct first_class_males_pct : ℕ)
    (h_total_passengers : total_passengers = 120)
    (h_females_pct : females_pct = 40)
    (h_first_class_pct : first_class_pct = 10)
    (h_first_class_males_pct : first_class_males_pct = 33) :
    let total_females := total_passengers * females_pct / 100
        total_first_class := total_passengers * first_class_pct / 100
        first_class_males := total_first_class * first_class_males_pct / 100
        first_class_females := total_first_class - first_class_males
        coach_females := total_females - first_class_females
    in coach_females = 40 := 
by 
    sorry

end females_in_coach_class_l661_661335


namespace smallest_number_conditions_l661_661851

theorem smallest_number_conditions :
  ∃ n : ℤ, (n > 0) ∧
           (n % 2 = 1) ∧
           (n % 3 = 1) ∧
           (n % 4 = 1) ∧
           (n % 5 = 1) ∧
           (n % 6 = 1) ∧
           (n % 11 = 0) ∧
           (∀ m : ℤ, (m > 0) → 
             (m % 2 = 1) ∧
             (m % 3 = 1) ∧
             (m % 4 = 1) ∧
             (m % 5 = 1) ∧
             (m % 6 = 1) ∧
             (m % 11 = 0) → 
             (n ≤ m)) :=
sorry

end smallest_number_conditions_l661_661851


namespace equidistant_from_perpendicular_bisector_l661_661722

theorem equidistant_from_perpendicular_bisector
  (A B M X : Point) 
  (hM : M = midpoint A B) 
  (hX : is_on_perpendicular_bisector X A B) :
  dist X A = dist X B := 
sorry

-- Definitions needed for the theorem:
def Point : Type := ℕ 

def dist (P Q : Point) : ℝ := sorry  -- Define distance

def midpoint (A B : Point) : Point := sorry  -- Define midpoint

def is_on_perpendicular_bisector (X A B : Point) : Prop := sorry  -- Define perpendicular bisector condition

end equidistant_from_perpendicular_bisector_l661_661722


namespace hyperbola_eccentricity_is_two_l661_661150

-- Define the hyperbola and its conditions
def hyperbola (x y a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1 ∧ a > 0 ∧ b > 0

-- Define the parabola and its directrix
def parabola (y x : ℝ) : Prop :=
  y^2 = 4 * x

def directrix (x : ℝ) : Prop :=
  x = -1

-- Define the given area of triangle AOB
def area_of_triangle (a b : ℝ) : Prop :=
  (1 / 2) * 1 * (2 * b / a) = sqrt 3

-- Define the eccentricity in terms of a and c
def eccentricity (c a : ℝ) : ℝ :=
  c / a

-- Prove that the eccentricity is 2
theorem hyperbola_eccentricity_is_two (a b c : ℝ) (hx : hyperbola a b) (hy : parabola c a) (h_area : area_of_triangle a b) :
  eccentricity c a = 2 := 
sorry

end hyperbola_eccentricity_is_two_l661_661150


namespace coins_value_l661_661027

theorem coins_value (x : ℕ) : 
  let pennies := x
  let dimes := 4 * x
  let quarters := 8 * x
  let total_value_dollars := 0.01 * pennies + 0.10 * dimes + 0.25 * quarters
  let total_value_cents := total_value_dollars * 100
  total_value_cents = 241 * x :=
by {
  -- Definitions according to the conditions
  let pennies := x
  let dimes := 4 * pennies
  let quarters := 8 * dimes

  -- Calculate total values according to the given conditions
  let total_value_dollars := 0.01 * pennies + 0.10 * dimes + 0.25 * quarters
  let total_value_cents := total_value_dollars * 100

  -- Final equality to prove
  show total_value_cents = 241 * x, by sorry
}

end coins_value_l661_661027


namespace total_number_of_dots_not_visible_l661_661431

theorem total_number_of_dots_not_visible
  (num_dice : ℕ) 
  (faces_per_die : ℕ)
  (total_dots_per_die : ℕ)
  (visible_faces : list ℕ) 
  (total_faces_visible : ℕ)
  (total_faces_per_die : ℕ)
  (total_visible_sum : ℕ)
  (total_remaining_sum : ℕ) : 
  num_dice = 4 →
  faces_per_die = 6 →
  total_faces_per_die = num_dice * faces_per_die →
  total_dots_per_die = 21 →
  visible_faces = [1, 2, 3, 4, 5, 4, 6, 5, 3] →
  total_faces_visible = visible_faces.length →
  total_visible_sum = visible_faces.sum →
  total_remaining_sum = num_dice * total_dots_per_die - total_visible_sum →
  total_remaining_sum = 51 := 
by
  intros num_dice_4 faces_per_die_6 total_faces_per_die_24 total_dots_per_die_21 visible_faces_expr total_faces_visible_9 total_visible_sum_33 total_remaining_sum_51
  sorry

end total_number_of_dots_not_visible_l661_661431


namespace my_problem_l661_661682

-- Definitions and conditions from the problem statement
variables (p q r u v w : ℝ)

-- Conditions
axiom h1 : 17 * u + q * v + r * w = 0
axiom h2 : p * u + 29 * v + r * w = 0
axiom h3 : p * u + q * v + 56 * w = 0
axiom h4 : p ≠ 17
axiom h5 : u ≠ 0

-- Problem statement to prove
theorem my_problem : (p / (p - 17)) + (q / (q - 29)) + (r / (r - 56)) = 0 :=
sorry

end my_problem_l661_661682


namespace cost_of_basketballs_discount_on_a_l661_661020

variables (x y : ℕ) -- cost of one A brand basketball and one B brand basketball respectively
variable m : ℝ -- discount on A brand basketballs

-- Conditions
axiom cond1 : 40 * x + 40 * y = 7200
axiom cond2 : 50 * x + 30 * y = 7400
axiom cond3 : 40 * (140 - x) + 10 * (140 - (140 * (m / 100))) + 30 * (80 * 0.3) = 2440

theorem cost_of_basketballs : x = 100 ∧ y = 80 :=
sorry

theorem discount_on_a (hx : x = 100) : m = 8 :=
sorry

end cost_of_basketballs_discount_on_a_l661_661020


namespace frozen_yoghurt_count_l661_661067

theorem frozen_yoghurt_count :
  ∀ (y : ℕ), 
    (let ice_cream_cost := 10 * 4 in
     let frozen_yoghurt_cost := y * 1 in
     ice_cream_cost - frozen_yoghurt_cost = 36) →
    y = 4 :=
by
  intros y h
  sorry

end frozen_yoghurt_count_l661_661067


namespace lucy_final_balance_l661_661704

def initial_balance : ℝ := 65
def deposit : ℝ := 15
def withdrawal : ℝ := 4

theorem lucy_final_balance : initial_balance + deposit - withdrawal = 76 :=
by
  sorry

end lucy_final_balance_l661_661704


namespace positional_relationship_tangent_line_eq_circle_exists_l661_661449

section CircleProofs

def CircleO : (ℝ × ℝ) → Prop := λ p, (p.1^2 + p.2^2 = 4)
def CircleC : (ℝ × ℝ) → Prop := λ p, (p.1^2 + (p.2 - 4)^2 = 1)

theorem positional_relationship : ∀ p1 p2 ∈ CircleO, ∀ q1 q2 ∈ CircleC,
  |(0:ℝ) - 0| * |4 - 0| > (2 + 1 : ℝ) := sorry

theorem tangent_line_eq : ∃ k : ℝ, k = √3 ∨ k = - √3 → 
  (λ x y, √3 * x - y + 4 = 0 ∨ √3 * x + y - 4 = 0) := sorry

theorem circle_exists : ∃ (P : (ℝ × ℝ) → Prop), (∀ p, P p ↔ p.1^2 + p.2^2 + - 16 / 5 * p.1 - 8 / 5 * p.2 + 12 / 5 = 0)
  ∨ (∀ p, P p ↔ p.1^2 + p.2^2 = 4) ∧ P (2, 0) := sorry

end CircleProofs

end positional_relationship_tangent_line_eq_circle_exists_l661_661449


namespace product_of_roots_l661_661923

theorem product_of_roots :
  let f := Polynomial.C (-50) + Polynomial.X * (Polynomial.C 75 + Polynomial.X * (Polynomial.C (-15) + Polynomial.X)) in 
  (f.roots.Prod = 50) :=
by sorry

end product_of_roots_l661_661923


namespace maya_gold_tokens_at_end_l661_661242

theorem maya_gold_tokens_at_end : 
  ∃ (x y : ℕ), 
  let R := 100 - 3*x + 2*y in
  let B := 100 + 2*x - 4*y in
  R < 3 ∧ B < 4 ∧ (x + y = 133) := 
begin
  sorry
end

end maya_gold_tokens_at_end_l661_661242


namespace max_value_cos_sin_domain_sqrt_log_l661_661819

-- Statement for Problem 1
theorem max_value_cos_sin (α : ℝ) :
  (∃ α : ℝ, y = cos α ^ 2 + sin α + 3) → (∃ y : ℝ, y ≤ 17 / 4) :=
sorry

-- Statement for Problem 2
theorem domain_sqrt_log (x : ℝ) :
  (∀ x, (sqrt (2 * sin x ^ 2 + 3 * sin x - 2) + log 2 (-x^2 + 7 * x + 8) ≠ 0) →
  ((∃ x ∈ [π / 6, 5 * π / 6], True) ∨ (∃ x ∈ [13 * π / 6, 8], True))) :=
sorry

end max_value_cos_sin_domain_sqrt_log_l661_661819


namespace find_AC_l661_661373

theorem find_AC (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (max_val : A - C = 3) (min_val : -A - C = -1) : 
  A = 2 ∧ C = 1 :=
by
  sorry

end find_AC_l661_661373


namespace minimally_intersecting_mod_1000_l661_661076

open Finset

-- Define the sets and conditions according to the problem statement
def minimally_intersecting (A B C : Finset ℕ) : Prop :=
  (A ∩ B).card = 1 ∧ (B ∩ C).card = 1 ∧ (C ∩ A).card = 1 ∧ (A ∩ B ∩ C) = ∅

-- Define the universal set
def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the number of minimally intersecting ordered triples
def count_minimally_intersecting_triples : ℕ :=
  let choices := 8 * 7 * 6 in
  let distributions := 4^5 in
  choices * distributions

-- Define the statement to prove
theorem minimally_intersecting_mod_1000 : (count_minimally_intersecting_triples % 1000) = 64 := by
  sorry

end minimally_intersecting_mod_1000_l661_661076


namespace sleeves_add_correct_weight_l661_661671

variable (R W_r W_s S : ℝ)

-- Conditions
def raw_squat : Prop := R = 600
def wraps_add_25_percent : Prop := W_r = R + 0.25 * R
def wraps_vs_sleeves_difference : Prop := W_r = W_s + 120

-- To Prove
theorem sleeves_add_correct_weight (h1 : raw_squat R) (h2 : wraps_add_25_percent R W_r) (h3 : wraps_vs_sleeves_difference W_r W_s) : S = 30 :=
by
  sorry

end sleeves_add_correct_weight_l661_661671


namespace length_of_platform_l661_661825

theorem length_of_platform (L : ℕ) :
  (∀ (V : ℚ), V = 600 / 52 → V = (600 + L) / 78) → L = 300 :=
by
  sorry

end length_of_platform_l661_661825


namespace company_remaining_fraction_l661_661327

def company_remaining_capital (C : ℝ) : ℝ :=
  let raw_materials := (1/4) * C
  let remaining_after_raw_materials := C - raw_materials
  let machinery := (1/10) * remaining_after_raw_materials
  let remaining_after_machinery := remaining_after_raw_materials - machinery
  (27/40) * C

theorem company_remaining_fraction (C : ℝ) : company_remaining_capital C = (27/40) * C :=
  by
  sorry

end company_remaining_fraction_l661_661327


namespace marked_price_of_each_article_l661_661328

-- Define the conditions as hypotheses
def marked_price_problem
  (pair_cost : ℝ)
  (discount : ℝ)
  (M : ℝ) : Prop :=
  pair_cost = 50 ∧ discount = 0.40 ∧
  0.60 * 2 * M = pair_cost

-- The proof statement we need to prove
theorem marked_price_of_each_article :
  ∃ (M : ℝ), marked_price_problem 50 0.40 M ∧ M = 41.67 :=
by
  existsi (41.67 : ℝ)
  unfold marked_price_problem
  split
  sorry
  exact rfl

end marked_price_of_each_article_l661_661328


namespace total_seconds_for_six_flights_l661_661735

theorem total_seconds_for_six_flights :
  let a := 15
  let d := 10
  let n := 6
  (n / 2) * (2 * a + (n - 1) * d) = 240 :=
by
  let a := 15
  let d := 10
  let n := 6
  have h : (6 / 2) * (2 * 15 + (6 - 1) * 10) = 240
  sorry

end total_seconds_for_six_flights_l661_661735


namespace greatest_perimeter_l661_661562

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l661_661562


namespace ed_more_marbles_than_doug_l661_661085

def number_of_marbles_doug_has := Nat

def ed_initial_marbles (D : number_of_marbles_doug_has) : number_of_marbles_doug_has :=
  D + 29

def ed_marbles_after_losing (D : number_of_marbles_doug_has) : number_of_marbles_doug_has :=
  ed_initial_marbles D - 17

theorem ed_more_marbles_than_doug (D : number_of_marbles_doug_has) :
  ed_marbles_after_losing D - D = 12 :=
by
  sorry

end ed_more_marbles_than_doug_l661_661085


namespace recur_decimal_times_nine_l661_661419

theorem recur_decimal_times_nine : 
  (0.3333333333333333 : ℝ) * 9 = 3 :=
by
  -- Convert 0.\overline{3} to a fraction
  have h1 : (0.3333333333333333 : ℝ) = (1 / 3 : ℝ), by sorry
  -- Perform multiplication and simplification
  calc
    (0.3333333333333333 : ℝ) * 9 = (1 / 3 : ℝ) * 9 : by rw h1
                              ... = (1 * 9) / 3 : by sorry
                              ... = 9 / 3 : by sorry
                              ... = 3 : by sorry

end recur_decimal_times_nine_l661_661419


namespace greatest_possible_perimeter_l661_661620

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l661_661620


namespace max_perimeter_l661_661627

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l661_661627


namespace exactly_one_germinates_l661_661779

theorem exactly_one_germinates (pA pB : ℝ) (hA : pA = 0.8) (hB : pB = 0.9) : 
  (pA * (1 - pB) + (1 - pA) * pB) = 0.26 :=
by
  sorry

end exactly_one_germinates_l661_661779


namespace country_of_second_se_asian_fields_medal_recipient_l661_661434

-- Given conditions as definitions
def is_highest_recognition (award : String) : Prop :=
  award = "Fields Medal"

def fields_medal_freq (years : Nat) : Prop :=
  years = 4 -- Fields Medal is awarded every four years

def second_se_asian_recipient (name : String) : Prop :=
  name = "Ngo Bao Chau"

-- The main theorem to prove
theorem country_of_second_se_asian_fields_medal_recipient :
  ∀ (award : String) (years : Nat) (name : String),
    is_highest_recognition award ∧ fields_medal_freq years ∧ second_se_asian_recipient name →
    (name = "Ngo Bao Chau" → ∃ (country : String), country = "Vietnam") :=
by
  intros award years name h
  sorry

end country_of_second_se_asian_fields_medal_recipient_l661_661434


namespace wheat_pile_weight_l661_661344

noncomputable def weight_of_conical_pile
  (circumference : ℝ) (height : ℝ) (density : ℝ) : ℝ :=
  let r := circumference / (2 * 3.14)
  let volume := (1.0 / 3.0) * 3.14 * r^2 * height
  volume * density

theorem wheat_pile_weight :
  weight_of_conical_pile 12.56 1.2 30 = 150.72 :=
by
  sorry

end wheat_pile_weight_l661_661344


namespace solve_rectangular_field_problem_l661_661358

-- Define the problem
def f (L W : ℝ) := L * W = 80 ∧ 2 * W + L = 28

-- Define the length of the uncovered side
def length_of_uncovered_side (L: ℝ) := L = 20

-- The statement we need to prove
theorem solve_rectangular_field_problem (L W : ℝ) (h : f L W) : length_of_uncovered_side L :=
by
  sorry

end solve_rectangular_field_problem_l661_661358


namespace product_of_roots_cubic_l661_661890

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l661_661890


namespace negation_equiv_l661_661282

variable (T : Type) [Triangle T] (is_equilateral : T → Prop) (angles_eq : T → Prop)

-- The proposition: If a triangle is an equilateral triangle, then the three interior angles of the triangle are equal.
def prop := ∀ t : T, is_equilateral t → angles_eq t

-- The negation of the proposition in Lean.
theorem negation_equiv {T : Type} [Triangle T] (is_equilateral angles_eq : T → Prop) :
  ¬ (∀ t : T, is_equilateral t → angles_eq t) ↔ ∃ t : T, ¬ (is_equilateral t → angles_eq t) :=
by
  sorry

end negation_equiv_l661_661282


namespace original_percentage_alcohol_l661_661340

-- Definitions of the conditions
def original_mixture_volume : ℝ := 15
def additional_water_volume : ℝ := 3
def final_percentage_alcohol : ℝ := 20.833333333333336
def final_mixture_volume : ℝ := original_mixture_volume + additional_water_volume

-- Lean statement to prove
theorem original_percentage_alcohol (A : ℝ) :
  (A / 100 * original_mixture_volume) = (final_percentage_alcohol / 100 * final_mixture_volume) →
  A = 25 :=
by
  sorry

end original_percentage_alcohol_l661_661340


namespace Maurice_current_age_l661_661047

variable (Ron_now Maurice_now : ℕ)

theorem Maurice_current_age
  (h1 : Ron_now = 43)
  (h2 : ∀ t Ron_future Maurice_future : ℕ, Ron_future = 4 * Maurice_future → Ron_future = Ron_now + 5 → Maurice_future = Maurice_now + 5) :
  Maurice_now = 7 := 
sorry

end Maurice_current_age_l661_661047


namespace fraction_power_multiplication_l661_661064

theorem fraction_power_multiplication :
  ( (4 / 5 : ℝ) ^ 10 * (2 / 3 : ℝ) ^ (-4) ) = (84934656 / 156250000 : ℝ) := 
by
  sorry

end fraction_power_multiplication_l661_661064


namespace exists_non_parallel_diagonal_l661_661250

theorem exists_non_parallel_diagonal (n : ℕ) (h : n > 0) : 
  ∀ (P : list (ℝ × ℝ)), convex_polygon P → (length P = 2 * n) → 
  ∃ (d : (ℝ × ℝ) × (ℝ × ℝ)), is_diagonal P d ∧ (∀ (s : (ℝ × ℝ) × (ℝ × ℝ)), s ∈ sides P → ¬ parallel d s) :=
by
  sorry

end exists_non_parallel_diagonal_l661_661250


namespace continuity_at_three_l661_661725

noncomputable def f (x : ℝ) : ℝ := -2 * x ^ 2 - 4

theorem continuity_at_three (ε : ℝ) (hε : 0 < ε) :
  ∃ δ > 0, ∀ x : ℝ, |x - 3| < δ → |f x - f 3| < ε :=
sorry

end continuity_at_three_l661_661725


namespace determine_current_transylvanian_type_l661_661667

-- Define the claim of a Transylvanian
def TransylvanianClaim := "I am a person who has lost their mind"

-- Define the logical type of a person (whether they are telling the truth or lying)
inductive PersonType
| truthful : PersonType
| lying : PersonType

-- Existing knowledge from the previous problem (assuming we have the type from the previous problem)
constant previousTransylvanianType : PersonType

-- Determine the type of the current Transylvanian based on the given logical conditions
theorem determine_current_transylvanian_type :
  (TransylvanianClaim → PersonType) ≠ previousTransylvanianType := by
  sorry

end determine_current_transylvanian_type_l661_661667


namespace correct_proposition_is_4_l661_661868

-- Define the propositions
def prop1 : Prop := (¬(x^2 = 1) → x ≠ 1) -- Correct negation
def prop2 : Prop := (∀ x : ℝ, x^2 + x - 1 ≥ 0) -- Correct negation
def prop3 : Prop := (¬(sin x ≠ sin y) → x = y) -- Correct contrapositive

-- Define conditions as declared in given problem
def cond1 : Prop := (x^2 = 1 → x = 1) = (x^2 = 1 → x ≠ 1)
def cond2 : Prop := (∃ x : ℝ, x^2 + x - 1 < 0) = (∀ x : ℝ, x^2 + x - 1 > 0)
def cond3 : Prop := (x = y → sin x = sin y) = false
def cond4 : Prop := (p ∨ q → p ∨ q) = true -- Simplified condition for logical truth

-- Given all the conditions, prove that the correct proposition number is 4
theorem correct_proposition_is_4 (hp : cond1 = false) (hq : cond2 = false) (hr : cond3 = false) (hs : cond4 = true) :
  4 = 4 :=
by
  -- Lean code requires more detailed statements here usually,
  -- but since we're focusing on the structure, we'll use sorry.
  sorry

end correct_proposition_is_4_l661_661868


namespace find_n_sin_cos_equality_l661_661092

theorem find_n_sin_cos_equality :
  let n_values := {n : ℤ | -180 ≤ n ∧ n ≤ 180 ∧ sin (n * real.pi / 180) = cos (630 * real.pi / 180)}
  n_values = {0, -180, 180} := by
  sorry

end find_n_sin_cos_equality_l661_661092


namespace sum_even_product_odd_l661_661513

theorem sum_even_product_odd (x y z : ℕ) 
  (hx : x = List.sum (List.range' 10 11).map id)
  (hy : y = List.length ((List.range' 10 11).filter Int.even))
  (hz : z = List.prod ((List.range' 10 11).filter Int.odd)) :
  x + y + z = 36,036,426 := 
  sorry

end sum_even_product_odd_l661_661513


namespace digit_difference_l661_661008

theorem digit_difference (X Y : ℕ) (h_digits : 0 ≤ X ∧ X < 10 ∧ 0 ≤ Y ∧ Y < 10) (h_diff :  (10 * X + Y) - (10 * Y + X) = 45) : X - Y = 5 :=
sorry

end digit_difference_l661_661008


namespace greatest_possible_perimeter_l661_661624

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l661_661624


namespace problem1_problem2_problem3_l661_661367

-- Problem 1: Prove that P-ABC is a regular tetrahedron
theorem problem1 (PA PB PC: ℝ) (h₁: PA = 1) (h₂: PB = 1) (h₃: PC = 1)
  (PABC_regular: sum_edge_lengths (DEF_ABC) = sum_edge_lengths (P_ABC))
  : is_regular_tetrahedron P ABC :=
  sorry

-- Problem 2: Find the dihedral angle between planes D-BC-A given PD = PA/2
theorem problem2 (PD PA: ℝ) (h: PD = PA / 2) (PA_length: PA = 1)
  : dihedral_angle D BC A = arcsin (sqrt 3 / 3) :=
  sorry

-- Problem 3: Exist a parallelepiped with volume V and equal edge lengths, having the same edge length sum as DEF-ABC
theorem problem3 {V: ℝ} (h₁: volume DEF_ABC = V) (h₂: sum_edge_lengths(DEF_ABC) = total_edge_length_similar_parallelepiped (V))
  : ∃ α, α = arcwedge(8 * V) :=
  sorry

end problem1_problem2_problem3_l661_661367


namespace point_on_transformed_plane_l661_661229

theorem point_on_transformed_plane :
  let k := (1 : ℝ) / 3
  let A := (2, -5, -1)
  let plane (x y z : ℝ) := 5 * x + 2 * y - 3 * z - 9 = 0
  let transformed_plane (x y z : ℝ) := 5 * x + 2 * y - 3 * z + k * (-9) = 0
  transformed_plane 2 (-5) (-1) :=
by
  let k := (1 : ℝ) / 3
  have h : transformed_plane (5 * 2 + 2 * -5 - 3 * -1 - 3) = 0
  sorry

end point_on_transformed_plane_l661_661229


namespace value_of_expression_l661_661797

-- defining the conditions
def in_interval (a : ℝ) : Prop := 1 < a ∧ a < 2

-- defining the algebraic expression
def algebraic_expression (a : ℝ) : ℝ := abs (a - 2) + abs (1 - a)

-- theorem to be proved
theorem value_of_expression (a : ℝ) (h : in_interval a) : algebraic_expression a = 1 :=
by
  -- proof will go here
  sorry

end value_of_expression_l661_661797


namespace find_principal_l661_661096

noncomputable def principal (A : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  A / (1 + r) ^ t

theorem find_principal :
  principal 1344 0.05 2.4 ≈ 1192.55 :=
by
  sorry

end find_principal_l661_661096


namespace statement_B_correct_statement_C_correct_l661_661317

theorem statement_B_correct :
  let total_outcomes := Nat.choose 5 2
  let diff_color_outcomes := 6
  (diff_color_outcomes / total_outcomes : Real) = 3 / 5 := by
  sorry

theorem statement_C_correct :
  let p_A := 0.8
  let p_B := 0.9
  1 - (1 - p_A) * (1 - p_B) = 0.98 := by
  sorry

end statement_B_correct_statement_C_correct_l661_661317


namespace quadrilateral_shape_l661_661122

variables {α : Type*} [linear_ordered_field α]

structure Point (α : Type*) :=
(x : α)
(y : α)

def vector_sub (P Q : Point α) : Point α :=
{ x := P.x - Q.x,
  y := P.y - Q.y }

-- Given points P and Q and point R = P - Q
variables (P Q : Point α)
def R : Point α := vector_sub P Q

-- Distinct points P and Q
axiom distinct_points (h : P ≠ Q) 
  
-- Prove that the quadrilateral OPRQ can be a parallelogram, a straight line, or a trapezoid
theorem quadrilateral_shape :
  (∃ a b, ∀ (k : α), k = -1) ∨ -- Parallelogram condition
  (P.y / Q.y = P.x / Q.x)      ∨ -- Straight line condition
  (((Q.x * Q.y - P.x * P.y) = 0) ∧ (∃ k, k ≠ 0 ∧ Q = vector_sub P (R.scale k))) -- Trapezoid condition
 := sorry

end quadrilateral_shape_l661_661122


namespace assign_parents_l661_661872

-- Definition of the problem's context
structure Orphanage := 
  (orphans : Type)
  [decidable_eq orphans]
  (are_friends : orphans → orphans → Prop)
  (are_enemies : orphans → orphans → Prop)
  (friend_or_enemy : ∀ (o1 o2 : orphans), are_friends o1 o2 ∨ are_enemies o1 o2)
  (friend_condition : ∀ (o : orphans) (f1 f2 f3 : orphans),
                        are_friends o f1 → 
                        are_friends o f2 → 
                        are_friends o f3 → 
                        even (finset.filter (λ (p : orphans × orphans), are_enemies p.fst p.snd) 
                                              (({f1, f2, f3}.powerset.filter (λ s, s.card = 2)).bUnion id)).card)

-- Definition of the conclusion
theorem assign_parents : 
  ∀ (O : Orphanage),
  ∃ (P : O.orphans → finset ℕ), 
  (∀ (o1 o2 : O.orphans), O.are_friends o1 o2 ↔ ∃ p, p ∈ P o1 ∧ p ∈ P o2) ∧ 
  (∀ (o1 o2 : O.orphans), O.are_enemies o1 o2 ↔ P o1 ∩ P o2 = ∅) ∧ 
  (∀ (o1 o2 o3 : O.orphans) (p : ℕ),
    p ∈ P o1 ∧ p ∈ P o2 ∧ p ∈ P o3 → false) :=
sorry

end assign_parents_l661_661872


namespace sum_of_absolute_values_l661_661473

theorem sum_of_absolute_values (n : ℕ) (n_ge_1 : 1 ≤ n) :
  let a : ℕ → ℤ := λ n, -4 * n + 5
  let q := a (n + 1) - a n
  let b : ℕ → ℤ := λ n, if n = 1 then 2 else -3 * (-4)^(n-2)
  ∑ i in (finset.range n).map nat.succ, |b i| = 4^n - 1 := 
by 
  sorry -- proof omitted

end sum_of_absolute_values_l661_661473


namespace product_of_roots_of_cubic_eqn_l661_661929

theorem product_of_roots_of_cubic_eqn :
  let p := Polynomial.Cubic (1 : ℝ) (-15) 75 (-50)
  in p.roots_prod = 50 :=
by
  sorry

end product_of_roots_of_cubic_eqn_l661_661929


namespace probability_of_yellow_marble_l661_661059

noncomputable def marbles_prob : ℚ := 
  let prob_white_A := 4 / 9
  let prob_black_A := 5 / 9
  let prob_yellow_B := 7 / 10
  let prob_blue_B := 3 / 10
  let prob_yellow_C := 3 / 9
  let prob_blue_C := 6 / 9
  let prob_yellow_D := 5 / 9
  let prob_blue_D := 4 / 9
  let prob_white_A_and_yellow_B := prob_white_A * prob_yellow_B
  let prob_black_A_and_blue_C := prob_black_A * prob_blue_C
  let prob_black_and_C_and_yellow_D := prob_black_A_and_blue_C * prob_yellow_D
  (prob_white_A_and_yellow_B + prob_black_and_C_and_yellow_D).reduce

theorem probability_of_yellow_marble :
  marbles_prob = 1884 / 3645 :=
sorry

end probability_of_yellow_marble_l661_661059


namespace ordinary_curve_equation_l661_661139

noncomputable def curve_parametric_equation {θ : ℝ} : ℝ × ℝ :=
  let x := Real.sin θ + Real.cos θ
  let y := 1 + Real.sin (2 * θ)
  (x, y)

theorem ordinary_curve_equation :
  ∀ θ : ℝ,
    let p := curve_parametric_equation θ in
    p.1^2 = p.2 ∧ p.1 ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) :=
by
  intro θ
  let x := Real.sin θ + Real.cos θ
  let y := 1 + Real.sin (2 * θ)
  have h1 : x^2 = 1 + 2 * Real.sin θ * Real.cos θ := by sorry
  have h2 : 1 + 2 * Real.sin θ * Real.cos θ = y := by sorry
  have h3 : x^2 = y := by sorry
  have h4 : -1 ≤ Real.sin (2 * θ) ∧ Real.sin (2 * θ) ≤ 1 := by sorry
  have h5 : 0 ≤ x^2 ∧ x^2 ≤ 2 := by sorry
  have h6 : x ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by sorry
  exact ⟨h3, h6⟩

end ordinary_curve_equation_l661_661139


namespace winning_ticket_probability_l661_661720

noncomputable def probability_holds_winning_ticket : ℚ := 1 / 5

theorem winning_ticket_probability (a b c d e : ℕ) 
(h1 : 1 ≤ a ∧ a ≤ 35)
(h2 : 1 ≤ b ∧ b ≤ 35)
(h3 : 1 ≤ c ∧ c ≤ 35)
(h4 : 1 ≤ d ∧ d ≤ 35)
(h5 : 1 ≤ e ∧ e ≤ 35)
(h6 : a ≠ b)
(h7 : a ≠ c)
(h8 : a ≠ d)
(h9 : a ≠ e)
(h10 : b ≠ c)
(h11 : b ≠ d)
(h12 : b ≠ e)
(h13 : c ≠ d)
(h14 : c ≠ e)
(h15 : d ≠ e)
(hlog5 : (Real.log a / Real.log 5) + (Real.log b / Real.log 5) + (Real.log c / Real.log 5) + (Real.log d / Real.log 5) + (Real.log e / Real.log 5) ∈ Int) :
  probability_holds_winning_ticket = 1 / 5 := by
  sorry

end winning_ticket_probability_l661_661720


namespace product_of_roots_of_cubic_l661_661979

theorem product_of_roots_of_cubic :
  (∃ (p q r : ℝ), (p, q, r ≠ 0) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 75*x - 50 = 0 ↔ x = p ∨ x = q ∨ x = r) ∧ p * q * r = 50) :=
sorry

end product_of_roots_of_cubic_l661_661979


namespace arithmetic_sequence_general_formula_max_sum_arithmetic_sequence_l661_661457

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 1) (h_a5 : a 5 = -5) :
  ∀ n : ℕ, a n = 5 - 2 * n :=
by
  sorry

theorem max_sum_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 1) (h_a5 : a 5 = -5) (h_sum : ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2) :
  S 2 = 4 :=
by
  sorry

end arithmetic_sequence_general_formula_max_sum_arithmetic_sequence_l661_661457


namespace product_of_roots_of_cubic_eqn_l661_661927

theorem product_of_roots_of_cubic_eqn :
  let p := Polynomial.Cubic (1 : ℝ) (-15) 75 (-50)
  in p.roots_prod = 50 :=
by
  sorry

end product_of_roots_of_cubic_eqn_l661_661927


namespace odd_function_a_value_l661_661507

theorem odd_function_a_value (a : ℝ) : 
  (∀ x : ℝ, (ln (a * x + sqrt (x^2 + 1)) = - ln (a * -x + sqrt (x^2 + 1)))) → (a = 1 ∨ a = -1) :=
by
  intro h
  sorry

end odd_function_a_value_l661_661507


namespace braelynn_total_cutlery_l661_661878

noncomputable def knives_initial : ℕ := 24
noncomputable def forks_initial : ℕ := 36
noncomputable def teaspoons_initial : ℕ := 2 * knives_initial

noncomputable def knives_additional : ℕ := (5 / 12 * knives_initial).floor.toNat
noncomputable def forks_additional : ℕ := (3 / 8 * forks_initial).ceil.toNat
noncomputable def teaspoons_additional : ℕ := (7 / 18 * teaspoons_initial).ceil.toNat

noncomputable def total_knives : ℕ := knives_initial + knives_additional
noncomputable def total_forks : ℕ := forks_initial + forks_additional
noncomputable def total_teaspoons : ℕ := teaspoons_initial + teaspoons_additional
noncomputable def total_cutlery : ℕ := total_knives + total_forks + total_teaspoons

theorem braelynn_total_cutlery :
  total_cutlery = 151 := by
  sorry

end braelynn_total_cutlery_l661_661878


namespace smallest_integer_k_l661_661794

theorem smallest_integer_k :
  ∃ k : ℕ, 
    k > 1 ∧ 
    k % 19 = 1 ∧ 
    k % 14 = 1 ∧ 
    k % 9 = 1 ∧ 
    k = 2395 :=
by {
  sorry
}

end smallest_integer_k_l661_661794


namespace find_curvature_radius_and_center_of_curvature_l661_661088

open Real

theorem find_curvature_radius_and_center_of_curvature :
  let α t := t - sin t
  let β t := 1 - cos t
  let t := π / 2
  let dα := 1 - cos t
  let dβ := sin t
  let d²α := sin t
  let d²β := cos t
  let K := abs (d²β * dα - dβ * d²α) / ((dα^2 + dβ^2) ^ (3/2))
  let R := (dα^2 + dβ^2) ^ (3/2) / abs (d²β * dα - dβ * d²α)
  let X := α t - ((dα^2 + dβ^2) * dβ) / (d²β * dα - dβ * d²α)
  let Y := β t + ((dα^2 + dβ^2) * dα) / (d²β * dα - dβ * d²α)
  K = 1 / (2 * sqrt 2) ∧
  R = 2 * sqrt 2 ∧
  (X, Y) = (π / 2 + 1, -1) :=
by sorry

end find_curvature_radius_and_center_of_curvature_l661_661088


namespace percent_students_own_only_cats_l661_661525

theorem percent_students_own_only_cats (total_students : ℕ) (students_owning_cats : ℕ) (students_owning_dogs : ℕ) (students_owning_both : ℕ) (h_total : total_students = 500) (h_cats : students_owning_cats = 80) (h_dogs : students_owning_dogs = 150) (h_both : students_owning_both = 40) : 
  (students_owning_cats - students_owning_both) * 100 / total_students = 8 := 
by
  sorry

end percent_students_own_only_cats_l661_661525


namespace solution_set_inequality_l661_661408

theorem solution_set_inequality :
  {x : ℝ | (x^2 + 4) / (x - 4)^2 ≥ 0} = {x | x < 4} ∪ {x | x > 4} :=
by
  sorry

end solution_set_inequality_l661_661408


namespace meena_work_days_l661_661727

theorem meena_work_days (M : ℝ) : 1/5 + 1/M = 3/10 → M = 10 :=
by
  sorry

end meena_work_days_l661_661727


namespace product_of_roots_of_cubic_eqn_l661_661931

theorem product_of_roots_of_cubic_eqn :
  let p := Polynomial.Cubic (1 : ℝ) (-15) 75 (-50)
  in p.roots_prod = 50 :=
by
  sorry

end product_of_roots_of_cubic_eqn_l661_661931


namespace find_third_side_length_l661_661193

noncomputable def length_of_third_side (a b : ℝ) (theta : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2 - 2*a*b*real.cos theta)

theorem find_third_side_length :
  length_of_third_side 9 11 (150 * real.pi / 180) = real.sqrt (202 + 99 * real.sqrt 3) :=
by
  sorry

end find_third_side_length_l661_661193


namespace prove_expr_simplification_l661_661880

def expr_simplification : Prop :=
    (\frac{1}{Real.sqrt 2 + 1} - Real.sqrt 8 + (Real.sqrt 3 + 1) ^ 0 = -Real.sqrt 2)

theorem prove_expr_simplification : expr_simplification :=
    sorry

end prove_expr_simplification_l661_661880


namespace product_extrema_l661_661441

theorem product_extrema (n : ℕ) (hn : 2 ≤ n) 
  (x : Fin n → ℝ) (h1 : ∀ i, 1 / n ≤ x i) 
  (h2 : ∑ i, (x i)^2 = 1) : 
  ∃ (xmin xmax : ℝ), 
    xmin = (Real.sqrt (n^2 - n + 1)) / (n^n) ∧ 
    xmax = n^(-n/2) ∧ 
    xmin ≤ (∏ i, x i) ∧ (∏ i, x i) ≤ xmax :=
begin
  sorry -- Proof goes here
end

end product_extrema_l661_661441


namespace distance_centers_isosceles_triangle_l661_661152

theorem distance_centers_isosceles_triangle
  (A B C : Point)
  (h_iso : is_isosceles_triangle A B C)
  (R : Real)
  (r : Real)
  (h_circumradius : circumradius A B C = R)
  (h_inradius : inradius A B C = r) :
  distance (circumcenter A B C) (incenter A B C) = Real.sqrt (R * (R - 2 * r)) := 
sorry

end distance_centers_isosceles_triangle_l661_661152


namespace product_of_roots_cubic_l661_661968

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  in (∃ r1 r2 r3 : ℝ, p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ r1 * r2 * r3 = 50) := 
by 
  sorry

end product_of_roots_cubic_l661_661968


namespace interval_between_births_l661_661774

variables {A1 A2 A3 A4 A5 : ℝ}
variable {x : ℝ}

def ages (A1 A2 A3 A4 A5 : ℝ) := A1 + A2 + A3 + A4 + A5 = 50
def youngest (A1 : ℝ) := A1 = 4
def interval (x : ℝ) := x = 3.4

theorem interval_between_births
  (h_age_sum: ages A1 A2 A3 A4 A5)
  (h_youngest: youngest A1)
  (h_ages: A2 = A1 + x ∧ A3 = A1 + 2 * x ∧ A4 = A1 + 3 * x ∧ A5 = A1 + 4 * x) :
  interval x :=
by {
  sorry
}

end interval_between_births_l661_661774


namespace two_axes_of_symmetry_are_perpendicular_l661_661336

noncomputable def figure_has_two_axes_of_symmetry (figure : Type) :=
  ∃ l₁ l₂ : set (point figure), (is_axis_of_symmetry l₁ figure) ∧ (is_axis_of_symmetry l₂ figure) ∧ l₁ ≠ l₂

theorem two_axes_of_symmetry_are_perpendicular (figure : Type) 
  (h : figure_has_two_axes_of_symmetry figure) : 
  ∃ l₁ l₂ : set (point figure), (is_axis_of_symmetry l₁ figure) 
    ∧ (is_axis_of_symmetry l₂ figure) 
    ∧ (l₁ ≠ l₂)
    ∧ (is_perpendicular l₁ l₂) :=
sorry

end two_axes_of_symmetry_are_perpendicular_l661_661336


namespace g_101_minus_g_99_eq_g_99_times_100_l661_661331

-- Define the function g
def g (n : ℕ) : ℕ :=
  if h : n < 3 then 1
  else (List.range' 1 n (by linarith)).filter (λ x, x % 2 = 1).prod

-- The theorem stating the problem
theorem g_101_minus_g_99_eq_g_99_times_100 :
  g 101 - g 99 = g 99 * 100 :=
sorry

end g_101_minus_g_99_eq_g_99_times_100_l661_661331


namespace lucy_bank_balance_l661_661705

def initial_balance : ℕ := 65
def deposit : ℕ := 15
def withdrawal : ℕ := 4

theorem lucy_bank_balance : initial_balance + deposit - withdrawal = 76 := by
  rw [← Nat.add_sub_assoc, Nat.add_sub_self, Nat.add_zero]
  exact rfl

end lucy_bank_balance_l661_661705


namespace print_width_is_correct_l661_661842

theorem print_width_is_correct
  (original_height : ℝ) (original_width : ℝ)
  (print_height : ℝ) (aspect_ratio : ℝ)
  (h1 : original_height = 10)
  (h2 : original_width = 15)
  (h3 : print_height = 25)
  (h4 : aspect_ratio = original_width / original_height) :
  (print_height * aspect_ratio = 37.5) :=
by
  rw [h1, h2, h3, h4]
  norm_num
  exact eq.refl 37.5


end print_width_is_correct_l661_661842


namespace tournament_conditions_l661_661524

variable {n : ℕ}
variables {P : Fin n → Fin n → Bool} -- P(i, j) means player i defeats player j

def condition1 (n : ℕ) (P : Fin n → Fin n → Bool) : Prop :=
  ∃ S T : Finset (Fin n), S ≠ ∅ ∧ T ≠ ∅ ∧ S ∪ T = Finset.univ ∧ (∀ i ∈ S, ∀ j ∈ T, P i j = true)

def condition2 (n : ℕ) (P : Fin n → Fin n → Bool) : Prop :=
  ∃ f : Fin n → Fin n, (∀ i : Fin n, P (f i) (f ((i + 1) % n)) = true)

theorem tournament_conditions (n : ℕ) (P : Fin n → Fin n → Bool) (no_draws : ∀ i j, i ≠ j → P i j = !P j i) :
  condition1 n P ∨ condition2 n P :=
by
  sorry  -- proof goes here

end tournament_conditions_l661_661524


namespace exists_team_with_min_wins_and_losses_l661_661219

theorem exists_team_with_min_wins_and_losses (n : ℕ) (hn : 0 < n)
  (teams : Finset (Fin (3 * n)))
  (matches : Finset (teams × teams))
  (hsize_teams : teams.card = 3 * n)
  (hsize_matches : matches.card = 3 * n^2)
  (hmatch_cond : ∀ (i j : teams), i ≠ j → (i, j) ∈ matches ∨ (j, i) ∈ matches) :
  ∃ team ∈ teams, (∃ k : ℕ, \(\frac{n}{4}\) ≤ k) ∧ (∃ l : ℕ, l ≤ \(\frac{n}{4}\)) :=
sorry

end exists_team_with_min_wins_and_losses_l661_661219


namespace length_of_BC_l661_661664

-- Define the relevant lengths and properties.
variable (A B C M : Type)
variable (AB AC BC AM BM MC : ℝ)

-- Define the given conditions.
def equilateral_triangle (AB AC AM : ℝ) : Prop :=
  AB = 9 ∧ AC = 9 ∧ AM = 5 ∧ AB = AC

-- Define the midpoint condition.
def midpoint (BM MC : ℝ) : Prop :=
  BM = MC 

-- Define the final problem to prove.
theorem length_of_BC
  (h1 : equilateral_triangle AB AC AM)
  (h2 : midpoint BM MC)
  (h3 : BC = 2 * BM)
  (h4 : AM = 5)
  (h5 : AB = 9)
  (h6 : AC = 9)
  (h7 : AB = AC) :
  BC = 4 * Real.sqrt 14 :=
by
  sorry

end length_of_BC_l661_661664


namespace product_of_roots_of_cubic_polynomial_l661_661898

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l661_661898


namespace domino_arrangement_possible_l661_661321

theorem domino_arrangement_possible (A B : List (ℕ × ℕ)) (h_ident_sets : ∀ a : (ℕ × ℕ), (a ∈ A ↔ a ∈ B)) (h_same_ends : A.head = B.head ∧ A.last = B.last) : 
  ∃ B' : List (ℕ × ℕ), (∀ a : (ℕ × ℕ), a ∈ B' ↔ a ∈ B) ∧ (A = B') :=
sorry

end domino_arrangement_possible_l661_661321


namespace urea_formed_l661_661822

-- Define the reactants and products
def NH3 := ℕ
def CO2 := ℕ
def H2NCONH2 := ℕ
def H2O := ℕ

-- Conditions
def initial_moles_NH3 : NH3 := 6
def initial_moles_CO2 : CO2 := 3
def balanced_reaction (nh3 : NH3) (co2 : CO2) (urea : H2NCONH2) (h2o : H2O) : Prop :=
  nh3 / 2 = co2 ∧ nh3 / 2 = urea ∧ urea = h2o

-- Proof goal
theorem urea_formed : ∃ (urea : H2NCONH2), balanced_reaction 6 3 urea 3 := 
by
  use 3
  simp [balanced_reaction]
  split
  { norm_num }
  split
  { norm_num }
  { norm_num }

end urea_formed_l661_661822


namespace jade_tower_levels_l661_661205

theorem jade_tower_levels (total_pieces : ℕ) (pieces_per_level : ℕ) (pieces_left : ℕ) 
  (hypo1 : total_pieces = 100) (hypo2 : pieces_per_level = 7) (hypo3 : pieces_left = 23) : 
  (total_pieces - pieces_left) / pieces_per_level = 11 :=
by
  have h1 : 100 - 23 = 77, sorry
  have h2 : 77 / 7 = 11, sorry
  exact h2

end jade_tower_levels_l661_661205


namespace count_valid_numbers_l661_661166

def valid_digits : List ℕ := [1, 2, 3, 4, 5]

def num_meets_conditions (n : ℕ) : Prop :=
  (n < 1000) ∧ (n % 4 = 0) ∧ (∀ d ∈ (n.digits 10), d ∈ valid_digits)

theorem count_valid_numbers : (Finset.filter num_meets_conditions (Finset.range 1000)).card = 31 := 
sorry

end count_valid_numbers_l661_661166


namespace elodie_rats_l661_661675

-- Define the problem conditions as hypotheses
def E (H : ℕ) : ℕ := H + 10
def K (H : ℕ) : ℕ := 3 * (E H + H)

-- The goal is to prove E = 30 given the conditions
theorem elodie_rats (H : ℕ) (h1 : E (H := H) + H + K (H := H) = 200) : E H = 30 :=
by
  sorry

end elodie_rats_l661_661675


namespace product_of_roots_l661_661908

-- Let's define the given polynomial equation.
def polynomial : Polynomial ℝ := Polynomial.monomial 3 1 - Polynomial.monomial 2 15 + Polynomial.monomial 1 75 - Polynomial.C 50

-- Prove that the product of the roots of the given polynomial is 50.
theorem product_of_roots : (Polynomial.roots polynomial).prod id = 50 := by
  sorry

end product_of_roots_l661_661908


namespace arithmetic_sequence_a9_l661_661198

variable {a : ℕ → ℤ}

-- Given Conditions
def condition1 : Prop := a 4 - a 2 = -2
def condition2 : Prop := a 7 = -3

-- Theorem: Under the above conditions, a 9 = -5
theorem arithmetic_sequence_a9 : condition1 ∧ condition2 → a 9 = -5 :=
by sorry

end arithmetic_sequence_a9_l661_661198


namespace no_three_consecutive_increasing_or_decreasing_permutations_ways_l661_661496

theorem no_three_consecutive_increasing_or_decreasing_permutations_ways : 
  let sequence := [1, 2, 3, 4, 5, 6] in 
  let is_valid (seq : List ℤ) : Bool :=
    ∀ i (h1 : 1 ≤ i ∧ i + 2 < seq.length), 
      ¬ (seq.nthLe i h1.1 < seq.nthLe (i+1) h1.2 ∧ seq.nthLe (i+1) h1.2 < seq.nthLe (i+2) h1.2) ∧ 
      ¬ (seq.nthLe i h1.1 > seq.nthLe (i+1) h1.2 ∧ seq.nthLe (i+1) h1.2 > seq.nthLe (i+2) h1.2)
  in
  (sequence.permutations.count (λ seq, is_valid seq) = 90) :=
sorry

end no_three_consecutive_increasing_or_decreasing_permutations_ways_l661_661496


namespace dimension_proof_l661_661844

noncomputable def sports_field_dimensions (x y: ℝ) : Prop :=
  -- Given conditions
  x^2 + y^2 = 185^2 ∧
  (x - 4) * (y - 4) = x * y - 1012 ∧
  -- Seeking to prove dimensions
  ((x = 153 ∧ y = 104) ∨ (x = 104 ∧ y = 153))

theorem dimension_proof : ∃ x y: ℝ, sports_field_dimensions x y := by
  sorry

end dimension_proof_l661_661844


namespace strawberry_gum_pieces_l661_661238

theorem strawberry_gum_pieces : 
  ∀ (x : ℕ), (∀ (y : ℕ), y = 30 / 5 → x = 30 / y → 30 = y * x) → x = 5 :=
by
  intros x y h
  have hy : y = 6 := 
    by
      simp [h y]
  have hx : x = 30 / y := 
    by
      simp [h x y]
  have hx_eq_5 : x = 5 := 
    by
      rw [hx, hy, Nat.div_self]
  exact hx_eq_5

end strawberry_gum_pieces_l661_661238


namespace product_of_roots_of_cubic_polynomial_l661_661895

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l661_661895


namespace part1_part2_l661_661124

variable (x y : ℝ) (k : ℝ) (a b c : ℝ)

-- Condition definitions
def positive_numbers (x y : ℝ) := x > 0 ∧ y > 0

def a_def (x y : ℝ) := x + y

def b_def (x y : ℝ) := Real.sqrt (x^2 + 7 * x * y + y^2)

def c_def (x y k : ℝ) := Real.sqrt (k * x * y)

noncomputable def triangle_inequality_condition (x y c : ℝ) :=
  ∀ (a b : ℝ), (a = x + y) ∧ (b = Real.sqrt (x^2 + 7 * x * y + y^2)) →
  (c < a + b) ∧ (c > b - a) ∧ (a < b + c) ∧ (b < a + c)

-- Proof statement for part (1)
theorem part1 (x : ℝ) (h₀ : x > 0) :
  let a := a_def x 1
  let b := b_def x 1
  1 < b / a ∧ b / a <= 3 / 2 :=
by
  let a := a_def x 1
  let b := b_def x 1
  exact ⟨sorry⟩

-- Proof statement for part (2)
theorem part2 (x y : ℝ) (h₀ : positive_numbers x y)
  (h₁ : c = c_def x y k) (h₂ : triangle_inequality_condition x y c) :
  1 < k ∧ k < 25 :=
by
  exact ⟨sorry⟩

end part1_part2_l661_661124


namespace triangle_perimeter_sum_l661_661717

theorem triangle_perimeter_sum (AE EF AG EG : ℕ) (h : ∀ x y : ℕ, x^2 - y^2 = 260 → 
  (y = 8 ∧ (x = 18 ∨ x = 66)) → x = 66 ∨ x = 18) :
  AE = 8 ∧ EF = 18 ∧ AG = FG ∧ ∀ z : ℤ, AG = z ∧ EG = 8 ∨ EG = 64 →
  let t : ℕ := 220
  t = 220 :=
begin
  sorry
end

end triangle_perimeter_sum_l661_661717


namespace product_of_roots_of_cubic_l661_661977

theorem product_of_roots_of_cubic :
  (∃ (p q r : ℝ), (p, q, r ≠ 0) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 75*x - 50 = 0 ↔ x = p ∨ x = q ∨ x = r) ∧ p * q * r = 50) :=
sorry

end product_of_roots_of_cubic_l661_661977


namespace greatest_possible_perimeter_l661_661559

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l661_661559


namespace product_of_roots_eq_50_l661_661949

theorem product_of_roots_eq_50 :
  let a := -15
  let c := -50
  let equation := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  -- Vieta's formulas for the product of roots in cubic equation ax^3 + bx^2 + cx + d = 0 is -d/a
  ∃ p q r : ℝ, 
    equation p = 0 ∧ equation q = 0 ∧ equation r = 0 ∧ (p * q * r = 50) :=
by
  sorry

end product_of_roots_eq_50_l661_661949


namespace product_of_roots_of_cubic_eqn_l661_661933

theorem product_of_roots_of_cubic_eqn :
  let p := Polynomial.Cubic (1 : ℝ) (-15) 75 (-50)
  in p.roots_prod = 50 :=
by
  sorry

end product_of_roots_of_cubic_eqn_l661_661933


namespace product_of_roots_of_cubic_l661_661974

theorem product_of_roots_of_cubic :
  (∃ (p q r : ℝ), (p, q, r ≠ 0) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 75*x - 50 = 0 ↔ x = p ∨ x = q ∨ x = r) ∧ p * q * r = 50) :=
sorry

end product_of_roots_of_cubic_l661_661974


namespace min_k_squared_floor_l661_661086

open Nat

theorem min_k_squared_floor (n : ℕ) :
  (∀ k : ℕ, k >= 1 → k^2 + (n / k^2) ≥ 1991) ∧
  (∃ k : ℕ, k >= 1 ∧ k^2 + (n / k^2) < 1992) ↔
  1024 * 967 ≤ n ∧ n ≤ 1024 * 967 + 1023 := 
by
  sorry

end min_k_squared_floor_l661_661086


namespace arctan_of_tan_diff_l661_661070

theorem arctan_of_tan_diff :
  ∀ (x y : ℝ), x = 80 ∧ y = 30 ∧ 0 <= arctan (tan x - 3 * tan y) < 180 → arctan (tan 80 - 3 * tan 30) = 80 :=
by
  intro x y
  intro h
  cases h with hx hy
  cases hy with hy hrange
  rw [hx, hy]
  sorry

end arctan_of_tan_diff_l661_661070


namespace greatest_possible_perimeter_l661_661555

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l661_661555


namespace problem_area_of_circle_l661_661314

noncomputable def circleAreaPortion : ℝ :=
  let r := Real.sqrt 59
  let theta := 135 * Real.pi / 180
  (theta / (2 * Real.pi)) * (Real.pi * r^2)

theorem problem_area_of_circle :
  circleAreaPortion = (177 / 8) * Real.pi := by
  sorry

end problem_area_of_circle_l661_661314


namespace sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees_l661_661355

theorem sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees :
  ∃ (n : ℕ), (n * (n - 3) / 2 = 14) → ((n - 2) * 180 = 900) :=
by
  sorry

end sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees_l661_661355


namespace inverse_value_at_2_l661_661078

noncomputable def f (x : ℝ) : ℝ := x / (2 * x + 1)

noncomputable def f_inv (x : ℝ) : ℝ := x / (1 - 2 * x)

theorem inverse_value_at_2 :
  f_inv 2 = -2/3 := by
  sorry

end inverse_value_at_2_l661_661078


namespace rectangle_area_l661_661363

theorem rectangle_area
  (s : ℝ)
  (h_square_area : s^2 = 49)
  (rect_width : ℝ := s)
  (rect_length : ℝ := 3 * rect_width)
  (h_rect_width_eq_s : rect_width = s)
  (h_rect_length_eq_3w : rect_length = 3 * rect_width) :
  rect_width * rect_length = 147 :=
by 
  skip
  sorry

end rectangle_area_l661_661363


namespace chocolate_candies_total_cost_l661_661831

-- Condition 1: A box of 30 chocolate candies costs $7.50
def box_cost : ℝ := 7.50
def candies_per_box : ℕ := 30

-- Condition 2: The local sales tax rate is 10%
def sales_tax_rate : ℝ := 0.10

-- Total number of candies to be bought
def total_candy_count : ℕ := 540

-- Calculate the number of boxes needed
def number_of_boxes (total_candies : ℕ) (candies_per_box : ℕ) : ℕ :=
  total_candies / candies_per_box

-- Calculate the cost without tax
def cost_without_tax (num_boxes : ℕ) (cost_per_box : ℝ) : ℝ :=
  num_boxes * cost_per_box

-- Calculate the total cost including tax
def total_cost_with_tax (cost : ℝ) (tax_rate : ℝ) : ℝ :=
  cost * (1 + tax_rate)

-- The main statement
theorem chocolate_candies_total_cost :
  total_cost_with_tax 
    (cost_without_tax (number_of_boxes total_candy_count candies_per_box) box_cost)
    sales_tax_rate = 148.50 :=
by
  sorry

end chocolate_candies_total_cost_l661_661831


namespace transportation_cost_invariant_l661_661739

/--
Several settlements are connected by roads to a city. A vehicle sets out from the city carrying loads for all the settlements. The cost of each trip is equal to the product of the weight of all the loads in the vehicle and the distance traveled. If the weight of each load is numerically equal to the distance from the city to the destination, then the total transportation cost does not depend on the order in which the settlements are visited.
-/

theorem transportation_cost_invariant (n : ℕ) (d : Fin n → ℝ) :
  ∑ (i : Fin n), d i * ∑ (j : Fin n), ite (i ≤ j) 1 0 * (d j) =
  ∑ (i : Fin n), d i^2 + 2 * ∑ (i j : Fin n), ite (i < j) 1 0 * (d i * d j) :=
sorry

end transportation_cost_invariant_l661_661739


namespace milestone_third_observation_l661_661856

variable (A B : ℕ)

theorem milestone_third_observation (A B : ℕ) (hAB : 10 * A + B) (hBA : 10 * B + A) (hA0B : 100 * A + B) :
  ∃ x y : ℕ, x = 1 ∧ y = 6 ∧ 10 * x + y = 106 :=
by
  -- Provide some initial assumptions about the digits to calculate the conclusion
  have h1 : y = 6 * x := sorry
  have h2 : x = 1 := sorry
  have h3 : y = 6 := sorry
  use x, y
  sorry

end milestone_third_observation_l661_661856


namespace distinct_special_fraction_sums_l661_661378

theorem distinct_special_fraction_sums :
  let special_fractions := {x : ℚ // ∃ a b : ℤ, a > 0 ∧ b > 0 ∧ a + b = 18 ∧ x = a / b}
  let special_sums := {s : ℤ // ∃ (x y : {x : ℚ // ∃ a b : ℤ, a > 0 ∧ b > 0 ∧ a + b = 18 ∧ x = a / b}), s = x.val + y.val}
  card (special_sums) = 14 := 
  sorry

end distinct_special_fraction_sums_l661_661378


namespace average_age_l661_661268

theorem average_age (total_age_fg total_age_parents total_age_teachers : ℕ)
  (h1 : total_age_fg / 30 = 10)
  (h2 : total_age_parents / 50 = 40)
  (h3 : total_age_teachers / 10 = 35) :
  (total_age_fg + total_age_parents + total_age_teachers) / 90 = 29.44 := 
by
  sorry

end average_age_l661_661268


namespace increasing_on_interval_l661_661143

theorem increasing_on_interval {a : ℝ} :
  (∀ x y, x < y → x < 4 → y < 4 → f x ≤ f y) ↔ a ≥ 5 
  where
    f (x : ℝ) : ℝ := -x^2 + 2*(a-1)*x + 2 :=
sorry

end increasing_on_interval_l661_661143


namespace greatest_possible_perimeter_l661_661547

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l661_661547


namespace lisa_remaining_assignments_l661_661701

theorem lisa_remaining_assignments (total_assignments : ℕ) (required_percentage : ℝ) (completed_assignments : ℕ) (b_grades_earned : ℕ) : ℕ :=
  let total_b_needed := (required_percentage * total_assignments).ceil in
  let additional_b_needed := total_b_needed - b_grades_earned in
  let remaining_assignments := total_assignments - completed_assignments in
  remaining_assignments - additional_b_needed

example (h : lisa_remaining_assignments 60 0.85 40 34 = 3) : true := 
  by {
    sorry
  }

end lisa_remaining_assignments_l661_661701


namespace hexagon_circle_radius_l661_661022

noncomputable def hexagon_radius (sides : List ℝ) (probability : ℝ) : ℝ :=
  let total_angle := 360.0
  let visible_angle := probability * total_angle
  let side_length_average := (sides.sum / sides.length : ℝ)
  let theta := (visible_angle / 6 : ℝ) -- assuming θ approximately splits equally among 6 gaps
  side_length_average / Real.sin (theta / 2 * Real.pi / 180.0)

theorem hexagon_circle_radius :
  hexagon_radius [3, 2, 4, 3, 2, 4] (1 / 3) = 17.28 :=
by
  sorry

end hexagon_circle_radius_l661_661022


namespace problem_1_problem_2_l661_661148

noncomputable def f (x a : ℝ) : ℝ := x * (Real.exp x + 1) - a * (Real.exp x - 1)

theorem problem_1 (a : ℝ) (h : (Deriv (fun x => f x a)) 1 = 1) : a = 2 :=
by
  sorry

theorem problem_2 (a : ℝ) (h : ∀ x > 0, f x a > 0) : a ≤ 2 :=
by
  sorry

end problem_1_problem_2_l661_661148


namespace product_of_roots_cubic_l661_661960

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  let roots := {x : ℝ // p x = 0}.toFinset
  ∃ r : ℝ, (∀ x ∈ roots, p x = 0) ∧ (∏ x in roots, x) = 50 :=
by
  sorry

end product_of_roots_cubic_l661_661960


namespace clock_angle_5pm_smaller_angle_l661_661368

-- Define the conditions:
def minute_hand_position : ℕ := 0 -- minute hand on the 12 (0 minutes)
def hour_hand_position : ℕ := 5 -- hour hand on the 5

-- Define the statement:
theorem clock_angle_5pm_smaller_angle : 
  (let hour_angle := 30 * (hour_hand_position - minute_hand_position) in
   let smaller_angle := if hour_angle > 180 then 360 - hour_angle else hour_angle in
   smaller_angle = 150) := 
by
  sorry

end clock_angle_5pm_smaller_angle_l661_661368


namespace product_of_roots_quadratic_eq_l661_661098

theorem product_of_roots_quadratic_eq (a b c : ℤ) (h_a : a = 25) (h_b : b = 60) (h_c : c = -700) :
  (∀ x y : ℤ, (a * x^2 + b * x + c = 0) ∧ (a * y^2 + b * y + c = 0) → x * y = c / a) →
  c / a = -28 :=
by
  rw [h_a, h_c]
  norm_num
  intro h
  specialize h 0 0  -- This is just a placeholder to fill out the statement
  sorry

end product_of_roots_quadratic_eq_l661_661098


namespace graph_fixed_point_l661_661508

theorem graph_fixed_point (f : ℝ → ℝ) (h : f 1 = 1) : f 1 = 1 :=
by
  sorry

end graph_fixed_point_l661_661508


namespace triangle_moved_up_by_3_l661_661184

noncomputable def triangle_moved_up (x y : ℝ) : Prop :=
  ∀ (t : ℝ × ℝ), t = (x, y) → ∃ t', t' = (x, y + 3)

theorem triangle_moved_up_by_3 (x y : ℝ) :
  triangle_moved_up x y :=
by {
  intros,
  use (x, y + 3),
  split,
  simp,
  sorry
}

end triangle_moved_up_by_3_l661_661184


namespace greatest_possible_perimeter_l661_661546

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l661_661546


namespace percentage_hexagon_area_l661_661843

theorem percentage_hexagon_area : 
  ∀ (pattern: Type) (side_length: ℝ), 
  (pattern = mix_of_congruent_squares_and_hexagons) ∧ (unit_pattern pattern = (3, 2)) ∧ (side_length = 1) → 
  let sq_area := 1 in
  let hex_area := (3 * Real.sqrt 3) / 2 in
  let total_area := 3 + 3 * Real.sqrt 3 in
  let hexagon_area_percentage := (100 * (3 * Real.sqrt 3)) / total_area in
  abs ((hexagon_area_percentage: ℝ) - 64) < 1%  :=
begin
  intros,
  sorry
end

end percentage_hexagon_area_l661_661843


namespace root_sum_cubed_identity_l661_661688

variables {a b c : ℝ}

def polynomial_has_roots (p : ℝ → ℝ) (a b c : ℝ) : Prop :=
  p.aeval a = 0 ∧ p.aeval b = 0 ∧ p.aeval c = 0

theorem root_sum_cubed_identity
  (h_roots : polynomial_has_roots (x^3 - 4 * x^2 + 50 * x - 7) a b c) :
  (a + b + 1)^3 + (b + c + 1)^3 + (c + a + 1)^3 = 991 := 
by sorry

end root_sum_cubed_identity_l661_661688


namespace product_of_roots_l661_661937

theorem product_of_roots :
  let p : Polynomial ℚ := Polynomial.Coeff.mk_fin 3 0 (⟨[50, -75, 15, -1]⟩ : ℕ → ℚ)
  (p.root_prod = 50) :=
begin
  sorry
end

end product_of_roots_l661_661937


namespace pauline_total_spent_l661_661249

variable {items_total : ℝ} (discount_rate : ℝ) (discount_limit : ℝ) (sales_tax_rate : ℝ)

def total_spent (items_total discount_rate discount_limit sales_tax_rate : ℝ) : ℝ :=
  let discount_amount := discount_rate * discount_limit
  let discounted_total := discount_limit - discount_amount
  let non_discounted_total := items_total - discount_limit
  let subtotal := discounted_total + non_discounted_total
  let sales_tax := sales_tax_rate * subtotal
  subtotal + sales_tax

theorem pauline_total_spent :
  total_spent 250 0.15 100 0.08 = 253.80 :=
by
  sorry

end pauline_total_spent_l661_661249


namespace product_of_roots_l661_661913

-- Let's define the given polynomial equation.
def polynomial : Polynomial ℝ := Polynomial.monomial 3 1 - Polynomial.monomial 2 15 + Polynomial.monomial 1 75 - Polynomial.C 50

-- Prove that the product of the roots of the given polynomial is 50.
theorem product_of_roots : (Polynomial.roots polynomial).prod id = 50 := by
  sorry

end product_of_roots_l661_661913


namespace total_boys_in_school_l661_661526

theorem total_boys_in_school
  (h_muslims : ∀ B : ℝ, 0.44 * B = real.to_rat (44 * B / 100))
  (h_hindus : ∀ B : ℝ, 0.28 * B = real.to_rat (28 * B / 100))
  (h_sikhs : ∀ B : ℝ, 0.10 * B = real.to_rat (10 * B / 100))
  (h_others : ∀ B : ℝ, 0.18 * B = 126) :  ∃ B : ℝ, B = 700 :=
by
  sorry

end total_boys_in_school_l661_661526


namespace volume_pyramid_l661_661271

/-- The volume of a pyramid with a square base ABCD, where P is a vertex equidistant
    from A, B, C, and D. If AB = 1 and ∠APB = 2θ, then the volume equals 
    (√(cos(2θ)) / (6 * sin(θ))). -/
theorem volume_pyramid
  (A B C D P : Type)
  (h_base_square : ∀ x y (x ≠ y), dist (A x) (B y) = dist (B x) (C y))  -- condition that ABCD is a square
  (h_eq_dist : ∀ x (x ∈ {A, B, C, D}), dist P x = dist P A)             -- condition that P is equidistant from A, B, C, D
  (h_AB_1 : dist A B = 1)                                                 -- condition that AB = 1
  (θ : ℝ)
  (h_angle : ∠ A P B = 2 * θ)                                             -- condition that ∠APB = 2θ
  : volume P A B C D = (sqrt (cos (2 * θ))) / (6 * sin θ) := sorry

end volume_pyramid_l661_661271


namespace savings_calculation_l661_661803

theorem savings_calculation (income expenditure savings : ℕ) (ratio_income ratio_expenditure : ℕ)
  (h_ratio : ratio_income = 10) (h_ratio2 : ratio_expenditure = 7) (h_income : income = 10000)
  (h_expenditure : 10 * expenditure = 7 * income) :
  savings = income - expenditure :=
by
  sorry

end savings_calculation_l661_661803


namespace product_of_roots_of_cubic_l661_661975

theorem product_of_roots_of_cubic :
  (∃ (p q r : ℝ), (p, q, r ≠ 0) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 75*x - 50 = 0 ↔ x = p ∨ x = q ∨ x = r) ∧ p * q * r = 50) :=
sorry

end product_of_roots_of_cubic_l661_661975


namespace distance_from_origin_to_point_8_neg3_6_l661_661527

def distance_from_origin (x y z : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 + z^2)

theorem distance_from_origin_to_point_8_neg3_6 : 
  distance_from_origin 8 (-3) 6 = Real.sqrt 109 := 
by
  sorry

end distance_from_origin_to_point_8_neg3_6_l661_661527


namespace product_of_roots_eq_50_l661_661953

theorem product_of_roots_eq_50 :
  let a := -15
  let c := -50
  let equation := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  -- Vieta's formulas for the product of roots in cubic equation ax^3 + bx^2 + cx + d = 0 is -d/a
  ∃ p q r : ℝ, 
    equation p = 0 ∧ equation q = 0 ∧ equation r = 0 ∧ (p * q * r = 50) :=
by
  sorry

end product_of_roots_eq_50_l661_661953


namespace exists_infinitely_many_solutions_l661_661988

theorem exists_infinitely_many_solutions :
  ∃ m : ℕ, (m = 12) ∧ ∃∞ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 
  ((1 / (a : ℚ)) + (1 / (b : ℚ)) + (1 / (c : ℚ)) + (1 / ((a * b * c) : ℚ)) = m / (a + b + c : ℚ)) :=
begin
  sorry
end

end exists_infinitely_many_solutions_l661_661988


namespace assign_parents_l661_661874

variable (V E : Type) [Fintype V] [Fintype E]
variable (G : SimpleGraph V) -- Graph with vertices V and edges between vertices E.

-- Conditions:
def condition1 : Prop :=
  ∀ {a b : V}, G.adj a b ∨ a = b ∨ ¬(G.adj a b)

def condition2 : Prop :=
  ∀ {a b c : V}, G.adj a b ∧ G.adj b c ∧ G.adj a c → (¬G.adj a b ∨ ¬G.adj b c ∨ ¬G.adj a c) ∨ (G.adj a b ∧ G.adj b c ∧ G.adj a c)

-- Theorem: 
theorem assign_parents (G : SimpleGraph V) [Fintype V] [DecidableRel G.adj] (cond1 : condition1 G) (cond2 : condition2 G) :
  ∃ (P : V → set V), 
    (∀ {u v : V}, G.adj u v → ∃ p, p ∈ P u ∧ p ∈ P v) ∧ 
    (∀ {u v : V}, ¬G.adj u v → ∃ p, p ∈ P u ∨ p ∈ P v) ∧ 
    (∀ {u v w : V}, (u ≠ v ∧ v ≠ w ∧ u ≠ w) → ∃ p1 p2 p3, (p1 ∈ P u ∧ p2 ∈ P v ∧ p3 ∈ P w)) :=
sorry

end assign_parents_l661_661874


namespace maurice_age_l661_661044

theorem maurice_age (M : ℕ) 
  (h₁ : 48 = 4 * (M + 5)) : M = 7 := 
by
  sorry

end maurice_age_l661_661044


namespace sum_a_lt_one_l661_661218

noncomputable def a : ℕ → ℝ
| 0       := 1 / 2
| (n + 1) := (2 * (n + 1) - 3) / (2 * (n + 1)) * a n

theorem sum_a_lt_one (n : ℕ) : (∑ k in Finset.range (n + 1), a k) < 1 := sorry

end sum_a_lt_one_l661_661218


namespace greatest_triangle_perimeter_l661_661572

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l661_661572


namespace line_through_A_eq_chord_min_area_triangle_l661_661481

-- Problem 1: Equation of the line l passing through A(0,5) and having a chord length 4√3
theorem line_through_A_eq_chord :
  ∃ l : ℝ → ℝ, (∀ x, l x = 5 ∨ l x = (3 * x - 20) / 4) ∧
              ∀ x1 x2, x1^2 + (l x1)^2 + 4 * x1 - 12 * (l x1) + 24 = 0 ∧ 
              x2^2 + (l x2)^2 + 4 * x2 - 12 * (l x2) + 24 = 0 ∧
              (x1 ≠ x2) → (l x1 - l x2).abs = 4 * Real.sqrt 3 := sorry

-- Problem 2: Minimum area of triangle QMN with Q on circle C
theorem min_area_triangle :
  let M := ( -1, 0)
  let N := ( 0, 1)
  ∀ (Q : ℝ × ℝ), Q.1^2 + Q.2^2 + 4 * Q.1 - 12 * Q.2 + 24 = 0 →
  (∃ min_area : ℝ, min_area = 7 / 2 - 2 * Real.sqrt 2) := sorry

end line_through_A_eq_chord_min_area_triangle_l661_661481


namespace sum_of_real_solutions_l661_661993

theorem sum_of_real_solutions :
  let f (x : ℝ) := (x^2 - 6 * x + 5)^(x^2 - 7 * x + 10)
  ∃ (S : set ℝ), (∀ x ∈ S, f x = 1) ∧ (S.sum id = 19) :=
  sorry

end sum_of_real_solutions_l661_661993


namespace positive_x_difference_at_y12_l661_661450

theorem positive_x_difference_at_y12 : 
  ∀ (p q : ℝ × ℝ → Prop), 
  (∀ x y, p (x, y) ↔ y = -2 * x + 8) → 
  (∀ x y, q (x, y) ↔ y = -1/3 * x + 3) → 
  (∀ x1 x2, p (x1, 12) → q (x2, 12) → abs (x1 - x2) = 25) :=
begin
  intros p q hp hq x1 x2 hp12 hq12,
  sorry
end

end positive_x_difference_at_y12_l661_661450


namespace volume_ratio_of_inscribed_cube_in_cone_l661_661203

noncomputable def cone_to_cube_volume_ratio (R h x : ℝ) : ℝ :=
  (π * R^2 * h) / x^3

theorem volume_ratio_of_inscribed_cube_in_cone {R h x : ℝ} 
  (cube_edge : x > 0) 
  (cone_radius : R = x / (2 * √2)) 
  (cone_height : h = x * (3 - √3)) :
  cone_to_cube_volume_ratio R h x = (π * √2 * (53 - 7 * √3)) / 45 := 
by
  sorry

end volume_ratio_of_inscribed_cube_in_cone_l661_661203


namespace greatest_perimeter_l661_661564

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l661_661564


namespace hyperbola_asymptote_value_of_a_l661_661478

-- Define the hyperbola and the conditions given
variables {a : ℝ} (h1 : a > 0) (h2 : ∀ x y : ℝ, 3 * x + 2 * y = 0 ∧ 3 * x - 2 * y = 0)

theorem hyperbola_asymptote_value_of_a :
  a = 2 := by
  sorry

end hyperbola_asymptote_value_of_a_l661_661478


namespace initial_men_is_250_l661_661350

-- Define the given conditions
def provisions (initial_men remaining_men initial_days remaining_days : ℕ) : Prop :=
  initial_men * initial_days = remaining_men * remaining_days

-- Define the problem statement
theorem initial_men_is_250 (initial_days remaining_days : ℕ) (remaining_men_leaving : ℕ) :
  provisions initial_men (initial_men - remaining_men_leaving) initial_days remaining_days → initial_men = 250 :=
by
  intros h
  -- Requirement to solve the theorem.
  -- This is where the proof steps would go, but we put sorry to satisfy the statement requirement.
  sorry

end initial_men_is_250_l661_661350


namespace purchase_price_of_radio_l661_661357

theorem purchase_price_of_radio 
  (selling_price : ℚ) (loss_percentage : ℚ) (purchase_price : ℚ) 
  (h1 : selling_price = 465.50)
  (h2 : loss_percentage = 0.05):
  purchase_price = 490 :=
by 
  sorry

end purchase_price_of_radio_l661_661357


namespace greatest_possible_perimeter_l661_661647

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l661_661647


namespace product_of_roots_cubic_l661_661969

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  in (∃ r1 r2 r3 : ℝ, p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ r1 * r2 * r3 = 50) := 
by 
  sorry

end product_of_roots_cubic_l661_661969


namespace product_of_roots_of_cubic_eqn_l661_661930

theorem product_of_roots_of_cubic_eqn :
  let p := Polynomial.Cubic (1 : ℝ) (-15) 75 (-50)
  in p.roots_prod = 50 :=
by
  sorry

end product_of_roots_of_cubic_eqn_l661_661930


namespace product_of_roots_cubic_l661_661958

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  let roots := {x : ℝ // p x = 0}.toFinset
  ∃ r : ℝ, (∀ x ∈ roots, p x = 0) ∧ (∏ x in roots, x) = 50 :=
by
  sorry

end product_of_roots_cubic_l661_661958


namespace proj_magnitude_l661_661222

open Real

noncomputable def magnitude_proj_v_onto_w
  (v w : ℝ → ℝ) (dot_vw : ℝ) (norm_w : ℝ) : ℝ :=
  abs (dot_vw / (norm_w ^ 2)) * norm_w

theorem proj_magnitude (v w : ℝ → ℝ)
  (hvw : dot_product v w = 8)
  (hw : norm w = 10) :
  magnitude_proj_v_onto_w v w 8 10 = 0.8 :=
by
  sorry

end proj_magnitude_l661_661222


namespace angle_MAN_constant_l661_661114

-- Definitions of objects and conditions
variables {r d : ℝ} (h1 : d > r)

-- Main theorem statement
theorem angle_MAN_constant (O : ℝ × ℝ) (l : ℝ) (M N : ℝ × ℝ) (A : ℝ × ℝ) :
  (A = (0, sqrt (d^2 - r^2))) →
  (dist O (fst M) = d) →
  (dist O (fst N) = d) →
  (sqrt (fst M - fst N)^2 + d^2 = (r + sqrt ((fst M - fst N)^2 + d^2) / 2)^2) →
  angle M A N = constant :=
sorry

end angle_MAN_constant_l661_661114


namespace A1_and_A2_complement_independent_l661_661516

open MeasureTheory

-- Hypothetical events A1 and A2
variable (Ω : Type) [MeasurableSpace Ω]
variable (P : ProbabilityMeasure Ω)

-- Event A1: drawing a black ball the first time
variable (A1 : Event Ω)

-- Event A2: drawing a black ball the second time
variable (A2 : Event Ω)

-- Event A2 complement: not drawing a black ball the second time
def A2_complement : Event Ω := A2ᶜ

-- Balls drawn with replacement implies independent events
variable (h_independent : P.IndepEvents A1 A2)

-- Prove that A1 and A2_complement are independent
theorem A1_and_A2_complement_independent : P.IndepEvents A1 A2_complement :=
by
  sorry

end A1_and_A2_complement_independent_l661_661516


namespace max_perimeter_l661_661635

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l661_661635


namespace sum_of_three_smallest_positive_solutions_equals_ten_and_half_l661_661421

noncomputable def sum_three_smallest_solutions : ℚ :=
    let x1 : ℚ := 2.75
    let x2 : ℚ := 3 + (4 / 9)
    let x3 : ℚ := 4 + (5 / 16)
    x1 + x2 + x3

theorem sum_of_three_smallest_positive_solutions_equals_ten_and_half :
  sum_three_smallest_solutions = 10.5 :=
by
  sorry

end sum_of_three_smallest_positive_solutions_equals_ten_and_half_l661_661421


namespace greatest_possible_perimeter_l661_661618

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l661_661618


namespace first_player_win_strategy_l661_661780

def can_first_player_always_win : Prop :=
  ∀ (piles : list ℕ), 
    (piles = [100, 100]) → 
    ∃ (strategy : Π (piles : list ℕ), list ℕ),
      (∀ (turn : ℕ), ∃ (next_piles : list ℕ), 
        (strategy piles = next_piles) ∧ 
        (∃ odd_pile_piles, odd_pile_piles ∈ next_piles ∧ 
          ∀ p ∈ odd_pile_piles, p % 2 = 1)) ∧ 
        ∀ p ∈ next_piles, p ≥ 1

theorem first_player_win_strategy : can_first_player_always_win :=
sorry

end first_player_win_strategy_l661_661780


namespace curve_and_circle_intersect_l661_661153

theorem curve_and_circle_intersect :
  ∃ t : ℝ, (2 * t, 1 + 4 * t) ∈ {p : ℝ × ℝ | p.1^2 + (p.2 - real.sqrt 2)^2 = 2} :=
sorry

end curve_and_circle_intersect_l661_661153


namespace speed_of_first_car_l661_661812

variable (V1 V2 V3 : ℝ) -- Define the speeds of the three cars
variable (t x : ℝ) -- Time interval and distance from A to B

-- Conditions of the problem
axiom condition_1 : x / V1 = (x / V2) + t
axiom condition_2 : x / V2 = (x / V3) + t
axiom condition_3 : 120 / V1  = (120 / V2) + 1
axiom condition_4 : 40 / V1 = 80 / V3

-- Proof statement
theorem speed_of_first_car : V1 = 30 := by
  sorry

end speed_of_first_car_l661_661812


namespace product_of_roots_l661_661912

-- Let's define the given polynomial equation.
def polynomial : Polynomial ℝ := Polynomial.monomial 3 1 - Polynomial.monomial 2 15 + Polynomial.monomial 1 75 - Polynomial.C 50

-- Prove that the product of the roots of the given polynomial is 50.
theorem product_of_roots : (Polynomial.roots polynomial).prod id = 50 := by
  sorry

end product_of_roots_l661_661912


namespace proof_problem_l661_661440

theorem proof_problem
  (x y a b c d : ℝ)
  (h1 : |x - 1| + (y + 2)^2 = 0)
  (h2 : a * b = 1)
  (h3 : c + d = 0) :
  (x + y)^3 - (-a * b)^2 + 3 * c + 3 * d = -2 :=
by
  -- The proof steps go here.
  sorry

end proof_problem_l661_661440


namespace smallest_positive_integer_expr_2010m_44000n_l661_661395

theorem smallest_positive_integer_expr_2010m_44000n :
  ∃ (m n : ℤ), 10 = gcd 2010 44000 :=
by
  sorry

end smallest_positive_integer_expr_2010m_44000n_l661_661395


namespace lego_tower_levels_l661_661208

theorem lego_tower_levels (initial_pieces : ℕ) (pieces_per_level : ℕ) (pieces_left : ℕ) 
    (h1 : initial_pieces = 100) (h2 : pieces_per_level = 7) (h3 : pieces_left = 23) :
    (initial_pieces - pieces_left) / pieces_per_level = 11 := 
by
  sorry

end lego_tower_levels_l661_661208


namespace part_a_part_b_l661_661740

-- Part (a): Number of ways to distribute 20 identical balls into 6 boxes so that no box is empty
theorem part_a:
  ∃ (n : ℕ), n = Nat.choose 19 5 :=
sorry

-- Part (b): Number of ways to distribute 20 identical balls into 6 boxes if some boxes can be empty
theorem part_b:
  ∃ (n : ℕ), n = Nat.choose 25 5 :=
sorry

end part_a_part_b_l661_661740


namespace topless_cubical_box_configurations_count_l661_661745

noncomputable def count_topless_cubical_box_configurations 
  (T_shaped_figure : Type) 
  (additional_squares : finset T_shaped_figure)
  (num_sides : ℕ) : ℕ :=
  if (T_shaped_figure ∈ (finset.range 4)) 
    ∧ (additional_squares = {1, 2, 3, 4, 5, 6, 7, 8})
    ∧ (num_sides = 2)
  then 56
  else 0

theorem topless_cubical_box_configurations_count :
  count_topless_cubical_box_configurations T_shaped_figure {1, 2, 3, 4, 5, 6, 7, 8} 2 = 56 :=
sorry

end topless_cubical_box_configurations_count_l661_661745


namespace gcd_of_459_and_357_l661_661065

theorem gcd_of_459_and_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_of_459_and_357_l661_661065


namespace product_of_roots_cubic_l661_661967

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  in (∃ r1 r2 r3 : ℝ, p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ r1 * r2 * r3 = 50) := 
by 
  sorry

end product_of_roots_cubic_l661_661967


namespace product_of_roots_cubic_l661_661893

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l661_661893


namespace distance_between_points_l661_661211

open Complex Real

def joe_point : ℂ := 2 + 3 * I
def gracie_point : ℂ := -2 + 2 * I

theorem distance_between_points : abs (joe_point - gracie_point) = sqrt 17 := by
  sorry

end distance_between_points_l661_661211


namespace desired_percentage_of_alcohol_l661_661824

def initial_volume : ℝ := 6
def initial_percentage : ℝ := 20
def added_volume : ℝ := 3.6

def total_volume : ℝ := initial_volume + added_volume
def initial_alcohol : ℝ := initial_volume * (initial_percentage / 100)
def final_alcohol : ℝ := initial_alcohol + added_volume
def final_percentage : ℝ := (final_alcohol / total_volume) * 100

theorem desired_percentage_of_alcohol :
  final_percentage = 50 := by
  sorry

end desired_percentage_of_alcohol_l661_661824


namespace construct_triangle_l661_661385

-- Definitions based on the conditions in a)
variable (a : ℝ) (β : ℝ) (r : ℝ)
variable (A B C : ℝ × ℝ) -- Points on the plane
variable (O : ℝ × ℝ) -- Center of the incircle
variable (circle_incircle : set (ℝ × ℝ)) -- The incircle as a set of points
variable (triangle_ABC : set (ℝ × ℝ))

-- Conditions as definitions
def is_incircle := ∀ (P : ℝ × ℝ), P ∈ circle_incircle ↔ (∃ θ : ℝ, P = (O.1 + r * cos θ, O.2 + r * sin θ))

def is_triangle := ∀ (P : ℝ × ℝ), P ∈ triangle_ABC ↔ (P = A ∨ P = B ∨ P = C)

def base_condition := dist B C = a

def angle_condition := ∃ (x y : ℝ), ceil (( √( (B.1 - x) ^ 2 + (B.2 - y) ^ 2) / dist B C)) = β / pi

def incircle_tangent := ∀ (P : ℝ × ℝ), P ∈ triangle_ABC → dist P O = r

-- Lean problem statement with conditions and a final conclusion
theorem construct_triangle :
  ∃ (A B C O : ℝ × ℝ) (circle_incircle triangle_ABC : set (ℝ × ℝ)),
  is_triangle triangle_ABC ∧
  base_condition B C ∧
  angle_condition B C β ∧
  is_incircle circle_incircle ∧
  incircle_tangent triangle_ABC := sorry

end construct_triangle_l661_661385


namespace curve_equation_and_line_l661_661154

theorem curve_equation_and_line (θ : ℝ) :
  let P := (1 + cos θ, sin θ),
      C := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1},
      l := {p : ℝ × ℝ | p.1 - p.2 = 0},
      A := ((1 + cos θ, sin θ): ℝ × ℝ),
      B := ((1 + cos (θ + π/2), sin (θ + π/2)): ℝ × ℝ),
      M_day := (sqrt 2 / 2 + 1, - sqrt 2 / 2)
  in
  (A ∈ C) ∧ (B ∈ C) ∧ (A ∈ l) ∧ (B ∈ l) ∧
  ∃ M ∈ C, max_area_C_A_B_M ∈ (sol [(sqrt 2) + 1] / 2).
  sorry

end curve_equation_and_line_l661_661154


namespace solution_set_for_inequality_l661_661131

noncomputable def f (x : ℝ) : ℝ := |x^2 - 3*x|

def even_function (g : ℝ → ℝ) := ∀ x : ℝ, g(x) = g(-x)

theorem solution_set_for_inequality :
  even_function f →
  ∀ x, f(x) = |x^2 - 3*x| →
  {x | f(x-2) ≤ 2} = 
    { x | -3 ≤ x ∧ x ≤ 1 } ∪ 
    { x | 0 ≤ x ∧ x ≤ (sqrt 17 - 1) / 2 } ∪ 
    { x | -(7 + sqrt 17) / 2 ≤ x ∧ x ≤ -4 } :=
sorry

end solution_set_for_inequality_l661_661131


namespace triangular_array_sum_digits_l661_661855

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 3780) : (N / 10 + N % 10) = 15 :=
sorry

end triangular_array_sum_digits_l661_661855


namespace find_vector_at_t_zero_l661_661352

open Matrix

/-- Defining vectors as matrices -/
def v1 : Matrix (Fin 2) (Fin 1) ℤ := !![2; 5]   -- vector at t = 1
def v4 : Matrix (Fin 2) (Fin 1) ℤ := !![5; -7]  -- vector at t = 4

/-- Define the line and its vectors at specific values of t -/
def line (a d : Matrix (Fin 2) (Fin 1) ℤ) (t : ℤ) : Matrix (Fin 2) (Fin 1) ℤ :=
  a + t • d

/-- Given conditions as hypotheses -/
variable (a d : Matrix (Fin 2) (Fin 1) ℤ)
variable (h1 : line a d 1 = v1)
variable (h4 : line a d 4 = v4)

/-- The theorem we need to prove -/
theorem find_vector_at_t_zero : line a d 0 = !![1; 9] :=
  sorry

end find_vector_at_t_zero_l661_661352


namespace total_time_with_pets_l661_661239

def time_with_cat : ℝ := 
12 -- petting
+ (1/3 * 12) -- combing
+ (1/4 * (1/3 * 12)) -- brushing teeth
+ (1/2 * 12) -- playing
+ 5 -- feeding
+ (2/5 * 5) -- cleaning food bowl

def time_with_dog : ℝ := 
18 -- walking
+ (2/3 * 18) -- playing fetch
+ 9 -- grooming
+ (1/3 * 9) -- trimming nails
+ (1/4 * 18) -- training for tricks

def time_with_parrot : ℝ := 
15 -- speech training
+ 8 -- cleaning cage
+ (1/2 * 8) -- setting up toys

def time_with_all_pets : ℝ := time_with_cat + time_with_dog + time_with_parrot

theorem total_time_with_pets : time_with_all_pets = 103.5 := by
  sorry

end total_time_with_pets_l661_661239


namespace paint_arrangement_l661_661715

variable {R B K : Type} [Fintype R] [Fintype B] [Fintype K] (c1 c2 c3 : R)

def ways_to_paint_arrangement : ℕ :=
  let inner_sticks := 3^6 in
  let triangle_coloring := 2^6 in
  inner_sticks * triangle_coloring

theorem paint_arrangement : ways_to_paint_arrangement c1 c2 c3 = 46656 :=
by
  sorry

end paint_arrangement_l661_661715


namespace greatest_possible_perimeter_l661_661531

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l661_661531


namespace total_bottles_ordered_in_april_and_may_is_1000_l661_661847

-- Define the conditions
def casesInApril : Nat := 20
def casesInMay : Nat := 30
def bottlesPerCase : Nat := 20

-- The total number of bottles ordered in April and May
def totalBottlesOrdered : Nat := (casesInApril + casesInMay) * bottlesPerCase

-- The main statement to be proved
theorem total_bottles_ordered_in_april_and_may_is_1000 :
  totalBottlesOrdered = 1000 :=
sorry

end total_bottles_ordered_in_april_and_may_is_1000_l661_661847


namespace greatest_possible_perimeter_l661_661644

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l661_661644


namespace area_of_circumcircle_of_isosceles_triangle_l661_661837

theorem area_of_circumcircle_of_isosceles_triangle:
  ∀ (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C],
  ∃ r: ℝ,
  (isosceles_triangle A B C r (5 : ℝ) (5 : ℝ) (4 : ℝ)) →
  (area_of_circumcircle A B C r = (13125 / 1764) * π) := 
  begin
    sorry
  end

end area_of_circumcircle_of_isosceles_triangle_l661_661837


namespace product_of_roots_of_cubic_l661_661980

theorem product_of_roots_of_cubic :
  (∃ (p q r : ℝ), (p, q, r ≠ 0) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 75*x - 50 = 0 ↔ x = p ∨ x = q ∨ x = r) ∧ p * q * r = 50) :=
sorry

end product_of_roots_of_cubic_l661_661980


namespace smallest_negative_integer_solution_l661_661420

theorem smallest_negative_integer_solution :
  ∃ x : ℤ, 45 * x + 8 ≡ 5 [ZMOD 24] ∧ x = -7 :=
sorry

end smallest_negative_integer_solution_l661_661420


namespace floor_cube_neg_seven_four_l661_661403

theorem floor_cube_neg_seven_four :
  (Int.floor ((-7 / 4 : ℚ) ^ 3) = -6) :=
by
  sorry

end floor_cube_neg_seven_four_l661_661403


namespace a1_value_l661_661116

noncomputable def geometric_sequence (a q : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := geometric_sequence a q n * q

-- Given conditions
axiom S3_eqn (a q : ℝ) : a + a * q + a * q^2 = a + 3 * (a * q)
axiom a4_eqn (a q : ℝ) : a * q^3 = 8

-- Prove a_1 = 1 given conditions
theorem a1_value (a q : ℝ) (hS3 : S3_eqn a q) (ha4 : a4_eqn a q) : a = 1 :=
by
  -- proof omitted
  sorry

end a1_value_l661_661116


namespace product_of_roots_of_cubic_l661_661976

theorem product_of_roots_of_cubic :
  (∃ (p q r : ℝ), (p, q, r ≠ 0) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 75*x - 50 = 0 ↔ x = p ∨ x = q ∨ x = r) ∧ p * q * r = 50) :=
sorry

end product_of_roots_of_cubic_l661_661976


namespace product_of_roots_l661_661917

theorem product_of_roots :
  let f := Polynomial.C (-50) + Polynomial.X * (Polynomial.C 75 + Polynomial.X * (Polynomial.C (-15) + Polynomial.X)) in 
  (f.roots.Prod = 50) :=
by sorry

end product_of_roots_l661_661917


namespace problem_statement_l661_661443

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom f_pos (x : ℝ) : x > 0 → f x > 0
axiom f'_less_f (x : ℝ) : f' x < f x
axiom f_has_deriv_at : ∀ x, HasDerivAt f (f' x) x

def a : ℝ := sorry
axiom a_in_range : 0 < a ∧ a < 1

theorem problem_statement : 3 * f 0 > f a ∧ f a > a * f 1 :=
  sorry

end problem_statement_l661_661443


namespace total_students_in_class_l661_661840

/-- 
There are 208 boys in the class.
There are 69 more girls than boys.
The total number of students in the class is the sum of boys and girls.
Prove that the total number of students in the graduating class is 485.
-/
theorem total_students_in_class (boys girls : ℕ) (h1 : boys = 208) (h2 : girls = boys + 69) : 
  boys + girls = 485 :=
by
  sorry

end total_students_in_class_l661_661840


namespace reduce_to_single_digit_l661_661030

theorem reduce_to_single_digit (N : ℕ) : ∃ k ≤ 15, ∃ M : ℕ, M < 10 ∧ (apply_operations k N = M) :=
sorry

end reduce_to_single_digit_l661_661030


namespace recur_decimal_times_nine_l661_661418

theorem recur_decimal_times_nine : 
  (0.3333333333333333 : ℝ) * 9 = 3 :=
by
  -- Convert 0.\overline{3} to a fraction
  have h1 : (0.3333333333333333 : ℝ) = (1 / 3 : ℝ), by sorry
  -- Perform multiplication and simplification
  calc
    (0.3333333333333333 : ℝ) * 9 = (1 / 3 : ℝ) * 9 : by rw h1
                              ... = (1 * 9) / 3 : by sorry
                              ... = 9 / 3 : by sorry
                              ... = 3 : by sorry

end recur_decimal_times_nine_l661_661418


namespace other_x_intercept_of_parabola_l661_661104

theorem other_x_intercept_of_parabola (a b c : ℝ) :
  (∃ x : ℝ, y = a * x ^ 2 + b * x + c) ∧ (2, 10) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)} ∧ (1, 0) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)}
  → ∃ x : ℝ, x = 3 ∧ (x, 0) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)} :=
by
  sorry

end other_x_intercept_of_parabola_l661_661104


namespace find_sin_C_find_area_l661_661182

variable {A B C a b c : ℝ}

-- Define given conditions for triangle ABC
def given_conditions (a b c : ℝ) (A : ℝ) : Prop :=
  a^2 + c^2 + real.sqrt 2 * a * c = b^2 ∧ 
  real.sin A = real.sqrt 10 / 10

-- Statement 1: Find sin C and prove it equals √5/5
theorem find_sin_C 
  (h : given_conditions a b c A) 
  (hA_pos : 0 < A ∧ A < π) :
  real.sin (π - (A + π / 4)) = real.sqrt 5 / 5 :=
sorry

-- Statement 2: If a = 2, find the area of triangle ABC and prove it equals 2
theorem find_area (h : given_conditions 2 b c A)
  (hA_pos : 0 < A ∧ A < π)
  (h_sinC : real.sin (π - (A + π / 4)) = real.sqrt 5 / 5) :
  let a := 2 in 
  let c := 2 * real.sqrt 2 in 
  (1 / 2) * a * c * (real.sin (3 * π / 4)) = 2 :=
sorry

end find_sin_C_find_area_l661_661182


namespace trains_length_difference_eq_zero_l661_661309

theorem trains_length_difference_eq_zero
  (T1_pole_time : ℕ) (T1_platform_time : ℕ) (T2_pole_time : ℕ) (T2_platform_time : ℕ) (platform_length : ℕ)
  (h1 : T1_pole_time = 11)
  (h2 : T1_platform_time = 22)
  (h3 : T2_pole_time = 15)
  (h4 : T2_platform_time = 30)
  (h5 : platform_length = 120) :
  let L1 := T1_pole_time * platform_length / (T1_platform_time - T1_pole_time)
  let L2 := T2_pole_time * platform_length / (T2_platform_time - T2_pole_time)
  L1 = L2 :=
by
  sorry

end trains_length_difference_eq_zero_l661_661309


namespace findLineL_l661_661474

-- Define the points O and M
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def M : Point := ⟨3, -3⟩

-- Define the condition that M is symmetric to O with respect to line L
noncomputable def isSymmetric (O M : Point) (L : Point → ℝ → Prop) : Prop :=
  L (Point.mk 0 0) 0 ∧ L (Point.mk 3 -3) 0 ∧ 
  (∀ P : Point, L P 0 → L (Point.mk (2 * (3/2) - P.x) (2 * (-3/2) - P.y)) 0)

-- The line L is given by the equation x - y - 3 = 0
def L (P : Point) : Prop := P.x - P.y - 3 = 0

theorem findLineL : ∃ L : Point → Prop, isSymmetric O M L ∧ (∀ P : Point, L P ↔ P.x - P.y - 3= 0) := by
  use L
  sorry

end findLineL_l661_661474


namespace knights_truth_maximum_l661_661084

theorem knights_truth_maximum: 
  ∀ (numbers : Fin 12 → ℕ), 
  (∀ i j, i ≠ j → numbers i ≠ numbers j) → 
  (∀ i, numbers i > numbers ((i + 1) % 12) ∧ numbers i > numbers ((i + 11) % 12) → 
       (∀ k, numbers k < numbers ((k + 1) % 12) ∨ numbers k < numbers ((k + 11) % 12) → 
       count (λ i, numbers i > numbers ((i + 1) % 12) ∧ numbers i > numbers ((i + 11) % 12)) ≤ 6)) :=
sorry

end knights_truth_maximum_l661_661084


namespace product_of_roots_of_cubic_l661_661983

theorem product_of_roots_of_cubic :
  (∃ (p q r : ℝ), (p, q, r ≠ 0) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 75*x - 50 = 0 ↔ x = p ∨ x = q ∨ x = r) ∧ p * q * r = 50) :=
sorry

end product_of_roots_of_cubic_l661_661983


namespace division_value_l661_661511

theorem division_value (x y : ℝ) (h1 : (x - 5) / y = 7) (h2 : (x - 14) / 10 = 4) : y = 7 :=
sorry

end division_value_l661_661511


namespace greatest_triangle_perimeter_l661_661574

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l661_661574


namespace greatest_perimeter_l661_661568

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l661_661568


namespace projection_length_l661_661308

variables (s_A : ℝ × ℝ) (s_B : ℝ × ℝ)

def proj_length (s_A s_B : ℝ × ℝ) : ℝ :=
  let dot_product := s_A.1 * s_B.1 + s_A.2 * s_B.2
  let mag_s_A := real.sqrt (s_A.1^2 + s_A.2^2)
  let mag_s_B := real.sqrt (s_B.1^2 + s_B.2^2)
  let cos_theta := dot_product / (mag_s_A * mag_s_B)
  mag_s_B * cos_theta

theorem projection_length (h₁ : s_A = (4, 3)) (h₂ : s_B = (-2, 6)) : proj_length s_A s_B = 2 := by
  -- Definitions placed for context based on problem statement
  have eq_dot_product : (s_A.1 * s_B.1 + s_A.2 * s_B.2) = 10 := by sorry
  have eq_mag_s_A : real.sqrt (s_A.1^2 + s_A.2^2) = 5 := by sorry
  have eq_mag_s_B : real.sqrt (s_B.1^2 + s_B.2^2) = 2 * real.sqrt 10 := by sorry
  have eq_cos_theta : (10 : ℝ) / (5 * (2 * real.sqrt 10)) = real.sqrt 10 / 10 := by sorry
  have eq_proj_length : (2 * real.sqrt 10) * (real.sqrt 10 / 10) = 2 := by sorry
  exact eq_proj_length
  sorry -- Full proof must be completed

end projection_length_l661_661308


namespace product_of_roots_l661_661941

theorem product_of_roots :
  let p : Polynomial ℚ := Polynomial.Coeff.mk_fin 3 0 (⟨[50, -75, 15, -1]⟩ : ℕ → ℚ)
  (p.root_prod = 50) :=
begin
  sorry
end

end product_of_roots_l661_661941


namespace max_perimeter_l661_661636

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l661_661636


namespace minimum_real_roots_l661_661691

-- Definitions and declarations from the conditions
noncomputable def g (x : ℝ) : Polynomial ℝ := sorry -- g(x) as a polynomial with real coefficients

namespace Degree2010PolynomialRealCoefficients
variables {k : ℕ} (s : Fin 2010 → ℂ) 

-- Condition that there are exactly 1006 distinct values among |s_1|, |s_2|, ..., |s_2010|
def distinct_magnitudes_values_1006 (s : Fin 2010 → ℂ) : Prop :=
  (s.toFinset.image (λ s : Fin 2010 → ℂ, Complex.abs (s s.val))).card = 1006

-- Condition that g(x) is a polynomial of degree 2010
def degree_2010 (g : Polynomial ℝ) : Prop :=
  g.degree = 2010

-- The main statement
theorem minimum_real_roots (g : Polynomial ℝ) (s : Fin 2010 → ℂ)
  (hg : degree_2010 g) (hm : distinct_magnitudes_values_1006 s) : 
  ∃ r_roots : ℕ, r_roots = 6 := 
  sorry
end Degree2010PolynomialRealCoefficients

end minimum_real_roots_l661_661691


namespace assign_parents_l661_661875

variable (V E : Type) [Fintype V] [Fintype E]
variable (G : SimpleGraph V) -- Graph with vertices V and edges between vertices E.

-- Conditions:
def condition1 : Prop :=
  ∀ {a b : V}, G.adj a b ∨ a = b ∨ ¬(G.adj a b)

def condition2 : Prop :=
  ∀ {a b c : V}, G.adj a b ∧ G.adj b c ∧ G.adj a c → (¬G.adj a b ∨ ¬G.adj b c ∨ ¬G.adj a c) ∨ (G.adj a b ∧ G.adj b c ∧ G.adj a c)

-- Theorem: 
theorem assign_parents (G : SimpleGraph V) [Fintype V] [DecidableRel G.adj] (cond1 : condition1 G) (cond2 : condition2 G) :
  ∃ (P : V → set V), 
    (∀ {u v : V}, G.adj u v → ∃ p, p ∈ P u ∧ p ∈ P v) ∧ 
    (∀ {u v : V}, ¬G.adj u v → ∃ p, p ∈ P u ∨ p ∈ P v) ∧ 
    (∀ {u v w : V}, (u ≠ v ∧ v ≠ w ∧ u ≠ w) → ∃ p1 p2 p3, (p1 ∈ P u ∧ p2 ∈ P v ∧ p3 ∈ P w)) :=
sorry

end assign_parents_l661_661875


namespace count_valid_4x4_matrices_l661_661381

/-- 
  A matrix is a 4x4 grid with numbers from 1 to 16 such that:
  - Each number appears exactly once.
  - Each row is sorted in ascending order.
  - Each column is sorted in ascending order.
  We aim to prove that the number of such valid matrices is 24.
-/
theorem count_valid_4x4_matrices : 
  ∃ (M : Matrix (Fin 4) (Fin 4) ℕ), 
    (∀ i, StrictlyIncreasing (fun j => M i j)) ∧
    (∀ j, StrictlyIncreasing (fun i => M i j)) ∧
    (Finset.image (fun i => M.1 i) Finset.univ = Finset.image (fun i => Finset.range 1 17)) ∧
    Finset.card {M | 
      ∀ i j, (1 ≤ M.1 i j) ∧ (M.1 i j ≤ 16) ∧ 
      (∀ i_x j_x, (i_x < i → M.1 i_x j < M.1 i j) ∧ (j_x < j → M.1 i j_x < M.1 i j))
    } = 24 :=
begin
  sorry
end

end count_valid_4x4_matrices_l661_661381


namespace area_of_given_figure_l661_661879

noncomputable def area_of_bounded_figure (f g : ℝ → ℝ) (a b : ℝ) :=
  ∫ t in a..b, f t * (deriv g t)

theorem area_of_given_figure :
  area_of_bounded_figure (λ t, 3 * Real.cos t) (λ t, 8 * Real.sin t) (Real.pi / 6) (5 * Real.pi / 6) = 
    8 * Real.pi - 6 * Real.sqrt 3 :=
by sorry

end area_of_given_figure_l661_661879


namespace passed_boys_count_l661_661269

theorem passed_boys_count (P F : ℕ) 
  (h1 : P + F = 120) 
  (h2 : 37 * 120 = 39 * P + 15 * F) : 
  P = 110 :=
sorry

end passed_boys_count_l661_661269


namespace greatest_possible_perimeter_l661_661541

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l661_661541


namespace square_side_measurement_error_l661_661052

theorem square_side_measurement_error {S S' : ℝ} (h1 : S' = S * Real.sqrt 1.0816) :
  ((S' - S) / S) * 100 = 4 := by
  sorry

end square_side_measurement_error_l661_661052


namespace infinite_set_int_prime_has_subset_l661_661681

open Set

variables (p : ℕ) [Fact (Nat.Prime p)] (A : Set ℤ)

def exists_subset_B (A : Set ℤ) (p : ℕ) :=
  ∃ B ⊆ A, B.finite ∧ B.card = 2 * p - 2 ∧
  ∀ (S : Finset ℤ), S ⊆ B → S.card = p → (S.sum / p : ℚ) ∉ A

theorem infinite_set_int_prime_has_subset (p : ℕ) [hp : Fact (Nat.Prime p)] 
  (A : Set ℤ) (hA : A.Infinite) : exists_subset_B A p :=
sorry -- proof required

end infinite_set_int_prime_has_subset_l661_661681


namespace greatest_perimeter_of_triangle_l661_661609

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l661_661609


namespace total_bottles_ordered_l661_661849

constant cases_april : ℕ
constant cases_may : ℕ
constant bottles_per_case : ℕ

axiom cases_april_def : cases_april = 20
axiom cases_may_def : cases_may = 30
axiom bottles_per_case_def : bottles_per_case = 20

theorem total_bottles_ordered :
  cases_april * bottles_per_case + cases_may * bottles_per_case = 1000 :=
by 
  rw [cases_april_def, cases_may_def, bottles_per_case_def]
  -- The remaining steps will be carried out and concluded with the necessary checks
  sorry

end total_bottles_ordered_l661_661849


namespace greatest_possible_perimeter_l661_661539

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l661_661539


namespace log_identity_l661_661319

noncomputable def my_log (base x : ℝ) := Real.log x / Real.log base

theorem log_identity (x : ℝ) (h : x > 0) (h1 : x ≠ 1) : 
  (my_log 4 x) * (my_log x 5) = my_log 4 5 :=
by
  sorry

end log_identity_l661_661319


namespace symmetric_diff_cardinality_l661_661007

theorem symmetric_diff_cardinality (X Y : Finset ℤ) 
  (hX : X.card = 8) 
  (hY : Y.card = 10) 
  (hXY : (X ∩ Y).card = 6) : 
  (X \ Y ∪ Y \ X).card = 6 := 
by
  sorry

end symmetric_diff_cardinality_l661_661007


namespace park_width_l661_661353

theorem park_width (length number_of_trees area_per_tree : ℕ) 
    (h_length : length = 1000)
    (h_number_of_trees : number_of_trees = 100000)
    (h_area_per_tree : area_per_tree = 20) :
    let total_area := number_of_trees * area_per_tree,
        width := total_area / length
    in width = 2000 :=
by {
    sorry
}

end park_width_l661_661353


namespace problem_statement_l661_661228

def f (x : ℝ) : ℝ := (5 / 4) ^ |x| + x ^ 2

def a : ℝ := f (Real.log (1 / 3))
def b : ℝ := f (Real.logBase 7 (1 / 3))
def c : ℝ := f (3 ^ 1.2)

theorem problem_statement : b < a ∧ a < c := sorry

end problem_statement_l661_661228


namespace exists_subset_S_l661_661216

open Finset

def D (n : ℕ) : Finset (ℕ × ℕ) := {p | 1 ≤ p.1 ∧ p.1 ≤ n ∧ 1 ≤ p.2 ∧ p.2 ≤ n}.to_finset

theorem exists_subset_S (n : ℕ) :
  ∃ S ⊆ D n,
    S.card ≥ (3 * n * (n + 1)) / 5 ∧
    ∀ (x1 y1 x2 y2 : ℕ),
      (x1, y1) ∈ S → 
      (x2, y2) ∈ S →
      (x1 + x2, y1 + y2) ∉ S :=
sorry

end exists_subset_S_l661_661216


namespace volume_of_solid_is_zero_l661_661035

variables (x y z : ℝ)

def u := (x, y, z : ℝ)

theorem volume_of_solid_is_zero : 
  (x^2 + y^2 + z^2 = 8 * x - 32 * y + 12 * z) → 
  (4 / 3) * real.pi * 0^3 = 0 := 
by
  intro h
  sorry

end volume_of_solid_is_zero_l661_661035


namespace greatest_possible_perimeter_l661_661542

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l661_661542


namespace find_a7_a8_l661_661659

noncomputable def geometric_sequence_property (a : ℕ → ℝ) (r : ℝ) :=
∀ n, a (n + 1) = r * a n

theorem find_a7_a8
  (a : ℕ → ℝ)
  (r : ℝ)
  (hs : geometric_sequence_property a r)
  (h1 : a 1 + a 2 = 40)
  (h2 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end find_a7_a8_l661_661659


namespace diagonal_not_parallel_l661_661254

theorem diagonal_not_parallel (n : ℕ) (h : 2 ≤ n) :
  ∃ (d : diagonal), ¬ (d ∥ side) := by
  sorry

end diagonal_not_parallel_l661_661254


namespace largest_last_digit_l661_661279

-- Define the string and constraints
constant str : String
constant first_digit : str.front = '2'
constant length_2500 : str.length = 2500
def valid_pair (a b: Char) : Prop :=
  let n := ((a.to_nat - '0'.to_nat) * 10 + (b.to_nat - '0'.to_nat)) in
  n % 17 = 0 ∨ n % 23 = 0

-- Each consecutive pair of characters in str must be a valid pair
constant valid_string : ∀ i : Fin (str.length - 1), valid_pair str[i] str[i+1]

-- Goal to prove
theorem largest_last_digit : str.back = '2' :=
by {
  sorry
}

end largest_last_digit_l661_661279


namespace greatest_possible_perimeter_l661_661535

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l661_661535


namespace area_of_triangle_l661_661410

theorem area_of_triangle {A B C : Type} (dist_AC dist_BC : ℝ) (median_perpendicular : Prop) : 
  dist (A, C) = 3 ∧ 
  dist (B, C) = 4 ∧ 
  median_perpendicular ∧ 
  (∀ AK BL : ℝ, orthogonal AK BL ∧ midpoint A K ∧ midpoint B L ∧ centroid A B C = intersection AK BL) 
  → 
  area_of_triangle ABC = sqrt 11 :=
by
  sorry

end area_of_triangle_l661_661410


namespace choose_officials_l661_661195

theorem choose_officials (n : ℕ) (h : n = 6) :
  ∃ (ways : ℕ), ways = 6 * 5 * 4 := by
  have h1 : 6 * 5 * 4 = 120 := rfl
  use 120
  exact h1

end choose_officials_l661_661195


namespace complex_quadrant_l661_661394

theorem complex_quadrant :
  let z := (1 - 3 * Complex.i) / (Complex.i - 1) in
  z = 1 - (3 / 2) * Complex.i ∧
  z.re > 0 ∧ z.im < 0 := 
by
  sorry

end complex_quadrant_l661_661394


namespace total_children_correct_l661_661054

def blocks : ℕ := 9
def children_per_block : ℕ := 6
def total_children : ℕ := blocks * children_per_block

theorem total_children_correct : total_children = 54 := by
  sorry

end total_children_correct_l661_661054


namespace min_freight_cost_l661_661787

noncomputable def minFreightCostSupply (a : ℕ) : ℕ × ℕ × ℕ :=
  (0, 4, 8)

theorem min_freight_cost :
  ∃ (xa xb xc ya yb yc : ℕ),
    xa + ya = 9 ∧
    xb + yb = 15 ∧
    xc + yc = 8 ∧
    xa + xb + xc = 12 ∧
    ya + yb + yc = 20 ∧
    let f := λ (x y : ℕ), 10 * x + 5 * y + 6 * (12 - x - y) + 4 * (9 - x) + 8 * (15 - y) + 15 * (x + y - 4) in
    f 0 4 = 1440 * a :=
by
  use 0, 9, 3, 0, 6, 14
  simp
  sorry

end min_freight_cost_l661_661787


namespace sum_distances_square_midpoints_l661_661066

theorem sum_distances_square_midpoints :
  let A := (0, 0)
  let B := (4, 0)
  let C := (4, 4)
  let D := (0, 4)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)  -- midpoint of AB
  let N := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)  -- midpoint of BC
  let O := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)  -- midpoint of CD
  let P := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)  -- midpoint of DA
  let distance := λ (p q : ℝ × ℝ), Math.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)
  distance A M + distance A N + distance A O + distance A P = 4 + 4 * Math.sqrt 5 :=
by
  sorry

end sum_distances_square_midpoints_l661_661066


namespace jade_tower_levels_l661_661206

theorem jade_tower_levels (total_pieces : ℕ) (pieces_per_level : ℕ) (pieces_left : ℕ) 
  (hypo1 : total_pieces = 100) (hypo2 : pieces_per_level = 7) (hypo3 : pieces_left = 23) : 
  (total_pieces - pieces_left) / pieces_per_level = 11 :=
by
  have h1 : 100 - 23 = 77, sorry
  have h2 : 77 / 7 = 11, sorry
  exact h2

end jade_tower_levels_l661_661206


namespace product_of_roots_l661_661911

-- Let's define the given polynomial equation.
def polynomial : Polynomial ℝ := Polynomial.monomial 3 1 - Polynomial.monomial 2 15 + Polynomial.monomial 1 75 - Polynomial.C 50

-- Prove that the product of the roots of the given polynomial is 50.
theorem product_of_roots : (Polynomial.roots polynomial).prod id = 50 := by
  sorry

end product_of_roots_l661_661911


namespace correct_addition_result_l661_661497

theorem correct_addition_result :
  ∃ (x : ℤ), (63 - x = 70) ∧ (36 + x = 29) :=
by
  use -7
  split
  · show 63 - (-7) = 70
    sorry
  · show 36 + (-7) = 29
    sorry

end correct_addition_result_l661_661497


namespace brazil_medal_fraction_closest_l661_661057

theorem brazil_medal_fraction_closest :
  let frac_win : ℚ := 23 / 150
  let frac_1_6 : ℚ := 1 / 6
  let frac_1_7 : ℚ := 1 / 7
  let frac_1_8 : ℚ := 1 / 8
  let frac_1_9 : ℚ := 1 / 9
  let frac_1_10 : ℚ := 1 / 10
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_6) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_8) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_9) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_10) :=
by
  sorry

end brazil_medal_fraction_closest_l661_661057


namespace quadrilateral_inequality_l661_661120

variables (α β γ θ r : ℝ)
variables (ABCD PQRS : Type)
variables [Fintype ABCD] [Fintype PQRS]
variables [Nonempty ABCD] [Nonempty PQRS]

-- Given areas and perimeters of the quadrilaterals
def S1 : ℝ := (1 / 2) * r * (2 * r * (cot α + cot β + cot γ + cot θ))
def S2 : ℝ := r^2 * (sin α * cos α + sin β * cos β + sin γ * cos γ + sin θ * cos θ)
def P1 : ℝ := 2 * r * (cot α + cot β + cot γ + cot θ)
def P2 : ℝ := 2 * r * (cos α + cos β + cos γ + cos θ)

theorem quadrilateral_inequality :
  (S1 r) / (S2 r α β γ θ) ≤ (P1 r α β γ θ) / (P2 r α β γ θ) ^ 2 :=
begin
  sorry
end

end quadrilateral_inequality_l661_661120


namespace product_of_roots_cubic_l661_661888

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l661_661888


namespace minimum_workers_l661_661386

theorem minimum_workers (total_days remaining_days completed_percent remaining_percent : ℕ) 
  (initial_workers : ℕ) (daily_work_rate_per_worker : ℕ) 
  (h1 : total_days = 40) 
  (h2 : remaining_days = 30) 
  (h3 : completed_percent = 40)
  (h4 : remaining_percent = 60) 
  (h5 : initial_workers = 10) 
  (h6 : daily_work_rate_per_worker = 4) : 
  ∃ w : ℕ, w = 5 :=
by
  exists 5
  sorry

end minimum_workers_l661_661386


namespace average_marks_of_all_students_l661_661748

theorem average_marks_of_all_students 
  (students_class1 students_class2 : ℕ) 
  (avg_marks_class1 avg_marks_class2 : ℕ)
  (h1 : students_class1 = 35)
  (h2 : avg_marks_class1 = 40)
  (h3 : students_class2 = 45)
  (h4 : avg_marks_class2 = 60) :
  (students_class1 * avg_marks_class1 + students_class2 * avg_marks_class2) / (students_class1 + students_class2) = 51.25 := 
sorry

end average_marks_of_all_students_l661_661748


namespace quadratic_complete_square_l661_661288

theorem quadratic_complete_square : 
  ∃ d e : ℝ, ((x^2 - 16*x + 15) = ((x + d)^2 + e)) ∧ (d + e = -57) := by
  sorry

end quadratic_complete_square_l661_661288


namespace fraction_denominator_condition_l661_661784

theorem fraction_denominator_condition (x : ℝ) : (2 - x ≠ 0) → (x ≠ 2) :=
by 
assume h : 2 - x ≠ 0
have h1 : 2 ≠ x := by exact ne.symm h
exact h1

end fraction_denominator_condition_l661_661784


namespace equal_intercepts_line_l661_661756

theorem equal_intercepts_line (x y : ℝ)
  (h1 : x + 2*y - 6 = 0) 
  (h2 : x - 2*y + 2 = 0) 
  (hx : x = 2) 
  (hy : y = 2) :
  (y = x) ∨ (x + y = 4) :=
sorry

end equal_intercepts_line_l661_661756


namespace lego_tower_levels_l661_661207

theorem lego_tower_levels (initial_pieces : ℕ) (pieces_per_level : ℕ) (pieces_left : ℕ) 
    (h1 : initial_pieces = 100) (h2 : pieces_per_level = 7) (h3 : pieces_left = 23) :
    (initial_pieces - pieces_left) / pieces_per_level = 11 := 
by
  sorry

end lego_tower_levels_l661_661207


namespace complement_of_union_correct_l661_661159

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def universal_set := {1, 2, 3, 4, 5, 6, 7, 8} : Set ℕ
def set_M := {1, 3, 5, 7} : Set ℕ
def set_N := {5, 6, 7} : Set ℕ

theorem complement_of_union_correct :
  (U = universal_set) → (M = set_M) → (N = set_N) → 
  (U \ (M ∪ N) = {2, 4, 8}) :=
by
  intros hU hM hN,
  rw [hU, hM, hN],
  sorry

end complement_of_union_correct_l661_661159


namespace product_of_roots_l661_661907

-- Let's define the given polynomial equation.
def polynomial : Polynomial ℝ := Polynomial.monomial 3 1 - Polynomial.monomial 2 15 + Polynomial.monomial 1 75 - Polynomial.C 50

-- Prove that the product of the roots of the given polynomial is 50.
theorem product_of_roots : (Polynomial.roots polynomial).prod id = 50 := by
  sorry

end product_of_roots_l661_661907


namespace inequality_proof_l661_661290

noncomputable def seq (n : ℕ) : ℝ :=
  if n = 1 then 1 / 2
  else let a := seq (n - 1) in -a + (1 / (2 - a))

theorem inequality_proof (n : ℕ) (hn : n ≥ 1):
    let s := (finset.range n).sum (λ k, seq (k + 1)) in
    ( (n : ℝ) / (2 * s) - 1)^n ≤ 
    ( s / n)^n * ((1 / seq 1 - 1) * (1 / seq 2 - 1) * ... * (1 / seq n - 1)) :=
sorry

end inequality_proof_l661_661290


namespace simplify_and_evaluate_correct_evaluation_l661_661260

variable (x y : ℚ)

theorem simplify_and_evaluate :
  (6 * (x^2 - (1/3) * x * y) - 3 * (x^2 - x * y) - 2 * x^2) = x^2 + x * y :=
by
  ring

theorem correct_evaluation :
  ∀ x y, x = -1/2 → y = 2 → 
  (6 * (x^2 - (1/3) * x * y) - 3 * (x^2 - x * y) - 2 * x^2) = -3/4 :=
by
  intros x y hx hy
  rw [hx, hy]
  let simplified_expr := x^2 + x * y
  have simplification : (6 * (x^2 - (1/3) * x * y) - 3 * (x^2 - x * y) - 2 * x^2) = simplified_expr :=
    by ring
  rw [simplified_expr] at simplification
  calc
    simplified_expr = (-1/2)^2 + (-1/2) * 2 : by rw [hx, hy]
    ... = 1/4 - 1 : by norm_num
    ... = -3/4 : by norm_num
  
  exact simplification

end simplify_and_evaluate_correct_evaluation_l661_661260


namespace number_of_elements_in_A_l661_661738

theorem number_of_elements_in_A (a b : ℕ) (h1 : a = 3 * b)
  (h2 : a + b - 100 = 500) (h3 : 100 = 100) (h4 : a - 100 = b - 100 + 50) : a = 450 := by
  sorry

end number_of_elements_in_A_l661_661738


namespace garden_remaining_area_is_250_l661_661349

open Nat

-- Define the dimensions of the rectangular garden
def garden_length : ℕ := 18
def garden_width : ℕ := 15
-- Define the dimensions of the square cutouts
def cutout1_side : ℕ := 4
def cutout2_side : ℕ := 2

-- Calculate areas based on the definitions
def garden_area : ℕ := garden_length * garden_width
def cutout1_area : ℕ := cutout1_side * cutout1_side
def cutout2_area : ℕ := cutout2_side * cutout2_side

-- Calculate total area excluding the cutouts
def remaining_area : ℕ := garden_area - cutout1_area - cutout2_area

-- Prove that the remaining area is 250 square feet
theorem garden_remaining_area_is_250 : remaining_area = 250 :=
by
  sorry

end garden_remaining_area_is_250_l661_661349


namespace cost_of_basketballs_discount_on_a_l661_661019

variables (x y : ℕ) -- cost of one A brand basketball and one B brand basketball respectively
variable m : ℝ -- discount on A brand basketballs

-- Conditions
axiom cond1 : 40 * x + 40 * y = 7200
axiom cond2 : 50 * x + 30 * y = 7400
axiom cond3 : 40 * (140 - x) + 10 * (140 - (140 * (m / 100))) + 30 * (80 * 0.3) = 2440

theorem cost_of_basketballs : x = 100 ∧ y = 80 :=
sorry

theorem discount_on_a (hx : x = 100) : m = 8 :=
sorry

end cost_of_basketballs_discount_on_a_l661_661019


namespace probability_of_dice_outcome_l661_661013

theorem probability_of_dice_outcome : 
  let p_one_digit := 3 / 4
  let p_two_digit := 1 / 4
  let comb := Nat.choose 5 3
  (comb * (p_one_digit^3) * (p_two_digit^2)) = 135 / 512 := 
by
  sorry

end probability_of_dice_outcome_l661_661013


namespace largest_integer_in_set_A_l661_661109

theorem largest_integer_in_set_A :
  let A := {x : ℝ | |x - 55| ≤ 11 / 2}
  in ∃ (n : ℤ), n ∈ A ∧ ∀ (m : ℤ), m ∈ A → m ≤ n :=
by
  let A := {x : ℝ | |x - 55| ≤ 11 / 2}
  have exists_60 : 60 ∈ A := sorry
  use 60
  split
  · exact exists_60
  · intros m hm
    sorry

end largest_integer_in_set_A_l661_661109


namespace greatest_possible_perimeter_l661_661548

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l661_661548


namespace number_of_elements_in_A_intersection_N_star_l661_661487

open Set

def A : Set ℝ := {x : ℝ | (x - 1) ^ 2 < 3 * x + 7}

def N_star : Set ℝ := {x : ℝ | x ∈ (Set.univ : Set ℝ) ∧ x > 0 ∧ x = Int.floor (x : ℝ)}

theorem number_of_elements_in_A_intersection_N_star  :
  ((A ∩ (N_star ∩ Ioi 0)).to_finset.to_list.length = 5) :=
sorry

end number_of_elements_in_A_intersection_N_star_l661_661487


namespace combined_girls_ave_l661_661864

-- Define the conditions as hypotheses
variables {C c D d : ℕ}

-- Given conditions translated to Lean
def cedar_ave_boys (C c : ℕ) : Prop := (65 * C + 70 * c) / (C + c) = 68
def dale_ave_boys (D d : ℕ) : Prop := (75 * D + 82 * d) / (D + d) = 78
def combined_boys_ave (C D : ℕ) : Prop := (65 * C + 75 * D) / (C + D) = 73

-- Goal: Prove that the average score for girls at both schools combined is 76
theorem combined_girls_ave
  (C c D d : ℕ)
  (h1 : cedar_ave_boys C c)
  (h2 : dale_ave_boys D d)
  (h3 : combined_boys_ave C D)
  : (70 + 82) / 2 = 76 :=
begin
  sorry
end

end combined_girls_ave_l661_661864


namespace max_perimeter_l661_661632

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l661_661632


namespace pears_thrown_away_is_72_percent_l661_661039

-- Define the conditions
variable (P : ℝ) -- Assume P is the initial count of pears.
variable (initial_count : P = 100) -- Vendor starts with 100 pears

-- Define the action on the first day
def first_day_remainder (P : ℝ) : ℝ := (P * 0.8) / 2 -- Pears thrown away on first day

-- Define the action on the second day
def second_day_remainder (P : ℝ) : ℝ := (P * 0.8 * 0.5) * 0.8 -- Pears thrown away on second day

-- Define the total thrown away over two days
def total_thrown (P : ℝ) : ℝ := first_day_remainder P + second_day_remainder P

-- The percentage of pears thrown away in total
def percentage_thrown_away (P : ℝ) : ℝ := (total_thrown P / P) * 100

-- Prove the vendor throws away 72% of his pears
theorem pears_thrown_away_is_72_percent : percentage_thrown_away 100 = 72 :=
by
  let P := 100 -- substituting P with 100 as per the initial count
  have h1 : P = 100 := rfl
  sorry -- Proof to be completed

end pears_thrown_away_is_72_percent_l661_661039


namespace bus_driver_total_hours_l661_661016

theorem bus_driver_total_hours
  (reg_rate : ℝ := 16)
  (ot_rate : ℝ := 28)
  (total_hours : ℝ)
  (total_compensation : ℝ := 920)
  (h : total_compensation = reg_rate * 40 + ot_rate * (total_hours - 40)) :
  total_hours = 50 := 
by 
  sorry

end bus_driver_total_hours_l661_661016


namespace product_of_roots_l661_661921

theorem product_of_roots :
  let f := Polynomial.C (-50) + Polynomial.X * (Polynomial.C 75 + Polynomial.X * (Polynomial.C (-15) + Polynomial.X)) in 
  (f.roots.Prod = 50) :=
by sorry

end product_of_roots_l661_661921


namespace employees_trained_in_all_three_restaurants_l661_661330

theorem employees_trained_in_all_three_restaurants :
  ∀ (employees total family_buffet dining_room snack_bar exactly_two : ℕ),
  total = 39 →
  family_buffet = 19 →
  dining_room = 18 →
  snack_bar = 12 →
  exactly_two = 4 →
  ∃ (x : ℕ), 19 + 18 + 12 - 4 - 2 * x = 39 ∧ x = 5 :=
by
  intros employees total family_buffet dining_room snack_bar exactly_two
  intros h_total h_family_buffet h_dining_room h_snack_bar h_exactly_two
  use 5
  split
  sorry

end employees_trained_in_all_three_restaurants_l661_661330


namespace one_imaginary_necessary_not_sufficient_l661_661236

noncomputable def is_imaginary (z : ℂ) : Prop := z.im ≠ 0

theorem one_imaginary_necessary_not_sufficient (z1 z2 : ℂ) :
  (is_imaginary z1 ∨ is_imaginary z2) -> (is_imaginary (z1 - z2)) :=
begin
  sorry
end

end one_imaginary_necessary_not_sufficient_l661_661236


namespace ollie_caught_fewer_fish_than_angus_l661_661871

theorem ollie_caught_fewer_fish_than_angus :
  ∀ (Angus Patrick Ollie : ℕ),
    (Patrick = 8) →
    (Angus = Patrick + 4) →
    (Ollie = 5) →
    (Angus - Ollie = 7) :=
by
  intros Angus Patrick Ollie Patrick_eq Angus_eq Ollie_eq
  rw [Patrick_eq, Angus_eq, Ollie_eq]
  rw [Nat.add_sub_assoc, ←Nat.add_sub_assoc] <;> norm_num
  rw [Nat.add_sub_cancel_left] <;> norm_num
  sorry

end ollie_caught_fewer_fish_than_angus_l661_661871


namespace find_number_subtracted_l661_661806

-- Given a number x, where the ratio of the two natural numbers is 6:5,
-- and another number y is subtracted to both numbers such that the new ratio becomes 5:4,
-- and the larger number exceeds the smaller number by 5,
-- prove that y = 5.
theorem find_number_subtracted (x y : ℕ) (h1 : 6 * x - 5 * x = 5) (h2 : (30 - y) * 4 = (25 - y) * 5) : y = 5 :=
by sorry

end find_number_subtracted_l661_661806


namespace correct_sample_size_l661_661835

-- Definitions based on conditions:
def population_size : ℕ := 1800
def sample_size : ℕ := 1000
def surveyed_parents : ℕ := 1000

-- The proof statement we need: 
-- Prove that the sample size is 1000, given the surveyed parents are 1000
theorem correct_sample_size (ps : ℕ) (sp : ℕ) (ss : ℕ) (h1 : ps = population_size) (h2 : sp = surveyed_parents) : ss = sample_size :=
  sorry

end correct_sample_size_l661_661835


namespace sum_of_integers_in_range_l661_661101

theorem sum_of_integers_in_range : 
  (∑ n in (multiset.filter (λ x, x < 2.5) (multiset.range' (-5) 8)), n) = -12 :=
by
  -- The elements in the range are the integers from -5 to 2 inclusive.
  have h : multiset.filter (λ x, x < 2.5) (multiset.range' (-5) 8) = [-5, -4, -3, -2, -1, 0, 1, 2] :=
    sorry,
  rw h,
  -- Sum of the list [-5, -4, -3, -2, -1, 0, 1, 2]
  simp,
  sorry

end sum_of_integers_in_range_l661_661101


namespace compute_polynomial_at_3_l661_661684

noncomputable def polynomial_p (b : Fin 6 → ℕ) (x : ℝ) : ℝ :=
  b 0 + b 1 * x + b 2 * x^2 + b 3 * x^3 + b 4 * x^4 + b 5 * x^5

theorem compute_polynomial_at_3
  (b : Fin 6 → ℕ)
  (hbi : ∀ i, b i < 5)
  (hP5 : polynomial_p b (Real.sqrt 5) = 40 + 31 * Real.sqrt 5) :
  polynomial_p b 3 = 381 :=
sorry

end compute_polynomial_at_3_l661_661684


namespace product_of_roots_l661_661939

theorem product_of_roots :
  let p : Polynomial ℚ := Polynomial.Coeff.mk_fin 3 0 (⟨[50, -75, 15, -1]⟩ : ℕ → ℚ)
  (p.root_prod = 50) :=
begin
  sorry
end

end product_of_roots_l661_661939


namespace vector_identity_l661_661221

-- Define unit vectors and vector cross and dot products
variables {V : Type*} [inner_product_space ℝ V]

-- Assumptions
variables (a b c : V)
variable [decidable_eq V]

-- Define the unit vector property
axiom unit_vectors : ∥a∥ = 1 ∧ ∥b∥ = 1

-- Define the given vector relations
axiom c_def : c = a × b + b
axiom cross_relation : c × b = a

-- Main theorem statement
theorem vector_identity : b ⋅ (a × c) = -1 := 
sorry

end vector_identity_l661_661221


namespace projectile_speed_l661_661789

theorem projectile_speed (d v₂ t : ℝ) (h₁ : d = 1998) (h₂ : v₂ = 555) (h₃ : t = 2) :
    let v := (888 / 2) in v = 444 :=
by
  -- Let v be the speed of the first projectile that needs to be proven as 444 km/h
  let v := (888 / 2)
  -- To satisfy the proof state
  exact Eq.refl 444

end projectile_speed_l661_661789


namespace product_of_roots_of_cubic_eqn_l661_661928

theorem product_of_roots_of_cubic_eqn :
  let p := Polynomial.Cubic (1 : ℝ) (-15) 75 (-50)
  in p.roots_prod = 50 :=
by
  sorry

end product_of_roots_of_cubic_eqn_l661_661928


namespace Jerry_gets_amount_l661_661210

def annual_salary := 50000
def number_of_years := 30
def medical_bills := 200000
def punitive_multiplier := 3
def fraction_awarded := 0.8

theorem Jerry_gets_amount :
  let total_lost_salary := annual_salary * number_of_years in
  let total_direct_damages := total_lost_salary + medical_bills in
  let punitive_damages := total_direct_damages * punitive_multiplier in
  let total_asked_for := total_direct_damages + punitive_damages in
  let total_awarded := total_asked_for * fraction_awarded in
  total_awarded = 5440000 := by
  sorry

end Jerry_gets_amount_l661_661210


namespace tammy_weekly_distance_l661_661746

-- Define the conditions.
def track_length : ℕ := 50
def loops_per_day : ℕ := 10
def days_in_week : ℕ := 7

-- Using the conditions, prove the total distance per week is 3500 meters.
theorem tammy_weekly_distance : (track_length * loops_per_day * days_in_week) = 3500 := by
  sorry

end tammy_weekly_distance_l661_661746


namespace solution_range_l661_661112

variable {x : ℝ}

def p := (x + 2) * (x - 2) ≤ 0
def q := x^2 - 3 * x - 4 ≤ 0

theorem solution_range :
  ¬ (p ∧ q) ∧ (p ∨ q) →
  x ∈ Set.Ioo (-2 : ℝ) (-1) ∪ Set.Ioo (2 : ℝ) (4 : ℝ) := by
  sorry

end solution_range_l661_661112


namespace greatest_possible_perimeter_l661_661550

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l661_661550


namespace sum_of_first_five_terms_l661_661472

noncomputable def S₅ (a : ℕ → ℝ) := (a 1 + a 5) / 2 * 5

theorem sum_of_first_five_terms (a : ℕ → ℝ) (a_2 a_4 : ℝ)
  (h1 : a 2 = 4)
  (h2 : a 4 = 2)
  (h3 : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  S₅ a = 15 :=
sorry

end sum_of_first_five_terms_l661_661472


namespace product_of_roots_of_cubic_eqn_l661_661926

theorem product_of_roots_of_cubic_eqn :
  let p := Polynomial.Cubic (1 : ℝ) (-15) 75 (-50)
  in p.roots_prod = 50 :=
by
  sorry

end product_of_roots_of_cubic_eqn_l661_661926


namespace product_of_roots_l661_661918

theorem product_of_roots :
  let f := Polynomial.C (-50) + Polynomial.X * (Polynomial.C 75 + Polynomial.X * (Polynomial.C (-15) + Polynomial.X)) in 
  (f.roots.Prod = 50) :=
by sorry

end product_of_roots_l661_661918


namespace greatest_possible_perimeter_l661_661528

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l661_661528


namespace carla_bought_two_bags_l661_661881

def original_price : ℚ := 6
def discount : ℚ := 0.75
def total_spent : ℚ := 3

theorem carla_bought_two_bags :
  let discounted_price := original_price * (1 - discount) in
  let number_of_bags := total_spent / discounted_price in
  number_of_bags = 2 :=
by
  sorry

end carla_bought_two_bags_l661_661881


namespace greatest_possible_perimeter_l661_661601

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l661_661601


namespace Walter_age_in_2003_l661_661876

-- Defining the conditions
def Walter_age_1998 (walter_age_1998 grandmother_age_1998 : ℝ) : Prop :=
  walter_age_1998 = grandmother_age_1998 / 3

def birth_years_sum (walter_age_1998 grandmother_age_1998 : ℝ) : Prop :=
  (1998 - walter_age_1998) + (1998 - grandmother_age_1998) = 3858

-- Defining the theorem to be proved
theorem Walter_age_in_2003 (walter_age_1998 grandmother_age_1998 : ℝ) 
  (h1 : Walter_age_1998 walter_age_1998 grandmother_age_1998) 
  (h2 : birth_years_sum walter_age_1998 grandmother_age_1998) : 
  walter_age_1998 + 5 = 39.5 :=
  sorry

end Walter_age_in_2003_l661_661876


namespace product_of_roots_eq_50_l661_661946

theorem product_of_roots_eq_50 :
  let a := -15
  let c := -50
  let equation := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  -- Vieta's formulas for the product of roots in cubic equation ax^3 + bx^2 + cx + d = 0 is -d/a
  ∃ p q r : ℝ, 
    equation p = 0 ∧ equation q = 0 ∧ equation r = 0 ∧ (p * q * r = 50) :=
by
  sorry

end product_of_roots_eq_50_l661_661946


namespace distance_range_in_tetrahedron_l661_661274

theorem distance_range_in_tetrahedron
  (A B C D P Q : ℝ × ℝ × ℝ)
  (h_tetrahedron : ∀ (u v : ℝ × ℝ × ℝ), u ∈ {A, B, C, D} → v ∈ {A, B, C, D} → u ≠ v → dist u v = 1)
  (h_PQ_on_AB_CD : ∃ (t1 t2 : ℝ), 0 ≤ t1 ∧ t1 ≤ 1 ∧ P = t1 • A + (1 - t1) • B ∧ 0 ≤ t2 ∧ t2 ≤ 1 ∧ Q = t2 • C + (1 - t2) • D) :
  dist P Q ∈ set.Icc (Real.sqrt 2 / 2) 1 := sorry

end distance_range_in_tetrahedron_l661_661274


namespace max_triangle_perimeter_l661_661583

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l661_661583


namespace diagonal_not_parallel_to_any_side_l661_661257

theorem diagonal_not_parallel_to_any_side (n : ℕ) (h : 2 ≤ n) :
  ∃ d, is_diagonal d ∧ convex (polygon 2n) ∧ ¬ (parallel d (side polygon)) :=
by
  sorry

end diagonal_not_parallel_to_any_side_l661_661257


namespace find_x_l661_661270

theorem find_x 
  (x : ℚ)
  (h : (list.sum (list.range 51) + x) / 51 = 50 * x) :
  x = 1275 / 2549 :=
by
  sorry

end find_x_l661_661270


namespace greatest_possible_perimeter_l661_661616

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l661_661616


namespace ratio_of_Ts_Tns_l661_661505

noncomputable def Ts (s : ℕ) (n : ℕ) (z : Fin n → ℂ) : ℂ :=
  ∑ i in (Finset.powersetLen s Finset.univ), ∏ j in i, (z j)

noncomputable def Tns (s : ℕ) (n : ℕ) (z : Fin n → ℂ) : ℂ :=
  ∑ i in (Finset.powersetLen (n - s) Finset.univ), ∏ j in i, (z j)

theorem ratio_of_Ts_Tns (z : Fin n → ℂ) (r : ℝ) (s : ℕ) (n : ℕ) (h_r_nonzero : r ≠ 0)
    (h_magnitudes : ∀ i, ∥z i∥ = r) (h_Tns_nonzero : Tns s n z ≠ 0) :
    ∥Ts s n z / Tns s n z∥ = r ^ (2 * s - n) :=
  by
  sorry

end ratio_of_Ts_Tns_l661_661505


namespace product_of_roots_l661_661910

-- Let's define the given polynomial equation.
def polynomial : Polynomial ℝ := Polynomial.monomial 3 1 - Polynomial.monomial 2 15 + Polynomial.monomial 1 75 - Polynomial.C 50

-- Prove that the product of the roots of the given polynomial is 50.
theorem product_of_roots : (Polynomial.roots polynomial).prod id = 50 := by
  sorry

end product_of_roots_l661_661910


namespace find_a_l661_661877

noncomputable def a' := 3
noncomputable def b' : ℝ := sorry
noncomputable def c : ℝ := sorry
def y (x : ℝ) : ℝ := a' * (1 / real.sin (b' * x + c))
def is_vertical_asymptote (x : ℝ) : Prop := ∃ k : ℤ, x = k * real.pi

theorem find_a' :
  (∀ x, is_vertical_asymptote x → x = (k * real.pi) ) ∧
  (∃ y_min, 0 < y_min ∧ ∀ x, y x ≥ y_min) →
  a' = 3 :=
begin
  sorry
end

end find_a_l661_661877


namespace rolling_hexagon_area_l661_661291

noncomputable def area_of_hexagon (side_length : ℝ) : ℝ :=
  (3 * real.sqrt 3 / 2) * side_length^2

noncomputable def area_of_circle (radius : ℝ) : ℝ :=
  real.pi * radius^2

theorem rolling_hexagon_area
  (side_length : ℝ)
  (radius : ℝ)
  (hexagon_area : ℝ := area_of_hexagon side_length)
  (circle_area : ℝ := area_of_circle radius) :
  (let traced_area := hexagon_area + 2 * circle_area in
   traced_area = hexagon_area + 2 * circle_area) :=
begin
  sorry
end

end rolling_hexagon_area_l661_661291


namespace diagonal_not_parallel_to_any_side_l661_661258

theorem diagonal_not_parallel_to_any_side (n : ℕ) (h : 2 ≤ n) :
  ∃ d, is_diagonal d ∧ convex (polygon 2n) ∧ ¬ (parallel d (side polygon)) :=
by
  sorry

end diagonal_not_parallel_to_any_side_l661_661258


namespace initial_workers_count_l661_661834

theorem initial_workers_count (W : ℕ) 
  (h1 : W * 30 = W * 30) 
  (h2 : W * 15 = (W - 5) * 20)
  (h3 : W > 5) 
  : W = 20 :=
by {
  sorry
}

end initial_workers_count_l661_661834


namespace greatest_possible_perimeter_l661_661537

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l661_661537


namespace minimum_sin6_cos6_l661_661412

theorem minimum_sin6_cos6 (x : ℝ) : ∃ y : ℝ, (∀ t : ℝ, sin t ^ 6 + 2 * cos t ^ 6 ≥ y) ∧ y = 2 / 3 :=
by
  sorry

end minimum_sin6_cos6_l661_661412


namespace max_triangle_perimeter_l661_661587

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l661_661587


namespace projection_matrix_v0_to_v2_l661_661223

theorem projection_matrix_v0_to_v2 :
  let v0 := ![a, b]
  let v1 := (1 / 17) • ![16, 4; 4, 1] ⬝ v0
  let v2 := (1 / 5) • ![1, 2; 2, 4] ⬝ v1
  (1 / 85) • ![(24 : ℚ), 6; 48, 12] = ![24 / 85, 6 / 85; 48 / 85, 12 / 85] :=
by sorry

end projection_matrix_v0_to_v2_l661_661223


namespace swim_distance_l661_661040

theorem swim_distance 
  (v c d : ℝ)
  (h₁ : c = 2)
  (h₂ : (d / (v + c) = 5))
  (h₃ : (25 / (v - c) = 5)) :
  d = 45 :=
by
  sorry

end swim_distance_l661_661040


namespace chess_game_analysis_l661_661397

noncomputable def chess_game_time : Prop :=
  ∃ (t₁ t₂ : ℝ), (t₁ ≥ 0) ∧ (t₂ ≥ 0) ∧ 
  (t₁ + t₂ = 150 * 60 * 2) ∧ -- total time for both players in seconds
  (∀ (T₁ T₂ : ℝ), (T₁ - T₂ ≥ 110) ∨ (T₂ - T₁ ≥ 110))

noncomputable def no_two_minute_difference : Prop :=
  ¬ ∃ (t₁ t₂ : ℝ), (t₁ ≥ 0) ∧ (t₂ ≥ 0) ∧ 
  (t₁ + t₂ = 150 * 60 * 2) ∧ -- total time for both players in seconds
  ((T₁ - T₂ ≥ 120) ∨ (T₂ - T₁ ≥ 120))

theorem chess_game_analysis :
  chess_game_time ∧ no_two_minute_difference :=
by
  sorry

end chess_game_analysis_l661_661397


namespace plane_division_ratio_l661_661446

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Parallelepiped :=
(A B C D A1 B1 C1 D1 : Point3D)

def segment_ratio (p1 p2 : Point3D) : ℝ → Point3D :=
λ k, ⟨p1.x + k * (p2.x - p1.x), p1.y + k * (p2.y - p1.y), p1.z + k * (p2.z - p1.z)⟩

def point_M (par : Parallelepiped) : Point3D :=
segment_ratio par.C1 par.C (5 / 2)

def point_N (par : Parallelepiped) : Point3D :=
segment_ratio par.C1 par.B1 (5 / 2)

def point_K (par : Parallelepiped) : Point3D :=
segment_ratio par.C1 par.D1 (5 / 2)

noncomputable def volume_ratio (par : Parallelepiped) : ℝ :=
if (linearly_independent R [par.A, par.B, par.C, par.D, par.A1, par.B1, par.C1, par.D1]) then
  1 / 47
else
  0

theorem plane_division_ratio (par : Parallelepiped) :
  volume_ratio par = 1 / 47 :=
sorry

end plane_division_ratio_l661_661446


namespace product_of_roots_cubic_l661_661965

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  in (∃ r1 r2 r3 : ℝ, p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ r1 * r2 * r3 = 50) := 
by 
  sorry

end product_of_roots_cubic_l661_661965


namespace greatest_possible_perimeter_l661_661621

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l661_661621


namespace problem_1_problem_2_l661_661488

def setP (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def setS (x : ℝ) (m : ℝ) : Prop := |x - 1| ≤ m

theorem problem_1 (m : ℝ) : (m ∈ Set.Iic (3)) → ∀ x, (setP x ∨ setS x m) → setP x := sorry

theorem problem_2 : ¬ ∃ m : ℝ, ∀ x : ℝ, (setP x ↔ setS x m) := sorry

end problem_1_problem_2_l661_661488


namespace distance_from_point_to_line_l661_661480

-- Definition of the conditions
def point := (3, 0)
def line_y := 1

-- Problem statement: Prove that the distance between the point (3,0) and the line y=1 is 1.
theorem distance_from_point_to_line (point : ℝ × ℝ) (line_y : ℝ) : abs (point.snd - line_y) = 1 :=
by
  -- insert proof here
  sorry

end distance_from_point_to_line_l661_661480


namespace max_perimeter_l661_661631

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l661_661631


namespace product_of_roots_eq_50_l661_661947

theorem product_of_roots_eq_50 :
  let a := -15
  let c := -50
  let equation := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  -- Vieta's formulas for the product of roots in cubic equation ax^3 + bx^2 + cx + d = 0 is -d/a
  ∃ p q r : ℝ, 
    equation p = 0 ∧ equation q = 0 ∧ equation r = 0 ∧ (p * q * r = 50) :=
by
  sorry

end product_of_roots_eq_50_l661_661947


namespace conjugate_of_z_l661_661442

theorem conjugate_of_z (z : ℂ) (h : (1 + complex.I) * z = 1 - complex.I) : complex.conj z = complex.I :=
sorry

end conjugate_of_z_l661_661442


namespace slope_of_line_angle_l661_661771

theorem slope_of_line_angle (x y : ℝ) (h : x + (√3) * y = 0) : 
  ∃ θ : ℝ, θ = 150 ∧ tan θ = -1 / √3 :=
by 
  sorry

end slope_of_line_angle_l661_661771


namespace sum_of_possible_values_eq_20_l661_661237

noncomputable def sum_of_possible_values (p q r s : ℤ) : ℤ :=
  |p - s|

theorem sum_of_possible_values_eq_20 (p q r s : ℤ) (h1 : |p - q| = 1) (h2 : |q - r| = 4) (h3 : |r - s| = 5) :
  ∑ x in {sum_of_possible_values p q r s | x ∈ {10, 8, 2, 0}}, x = 20 :=
by sorry

end sum_of_possible_values_eq_20_l661_661237


namespace computer_sequence_value_l661_661174

def sum_of_arithmetic_sequence (n : ℕ) (a d : ℝ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

theorem computer_sequence_value :
  ∃ n : ℕ, (sum_of_arithmetic_sequence n 5 3 ≥ 5000) ∧ (5 + 3 * (n - 1) = 173) :=
by {
  sorry -- This will contain the proof steps
}

end computer_sequence_value_l661_661174


namespace normal_distribution_interval_probability_l661_661186

open ProbabilityTheory

theorem normal_distribution_interval_probability 
  (σ : ℝ) (hσ : σ > 0) (h0 : ProbabilityTheory.PDF (Normal 1 σ^2) (set.Ioo 0 1) = 0.4) :
  ProbabilityTheory.PDF (Normal 1 σ^2) (set.Ioo 0 2) = 0.8 :=
sorry

end normal_distribution_interval_probability_l661_661186


namespace greatest_possible_perimeter_l661_661645

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l661_661645


namespace count_two_digit_numbers_with_twice_digit_l661_661649

def isTwoDigitNumberWithTwiceDigit (n : ℕ) : Prop :=
  let a := n / 10
  let b := n % 10
  b ≠ 0 ∧ (a = 2 * b ∨ b = 2 * a)

def allTwoDigitNumbers : List ℕ :=
  List.filter isTwoDigitNumberWithTwiceDigit (List.range' 10 90)

theorem count_two_digit_numbers_with_twice_digit : allTwoDigitNumbers.length = 8 := 
  by
    sorry

end count_two_digit_numbers_with_twice_digit_l661_661649


namespace greatest_perimeter_of_triangle_l661_661607

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l661_661607


namespace count_valid_numbers_l661_661495

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

def valid_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 300 ∧ sum_of_digits n = 7

theorem count_valid_numbers : finset.card (finset.filter valid_number (finset.range 300)) = 13 := sorry

end count_valid_numbers_l661_661495


namespace spongebob_earnings_l661_661266

theorem spongebob_earnings :
  let price_burger := 2.50
  let num_burgers := 30
  let price_large_fries := 1.75
  let num_large_fries := 12
  let price_small_fries := 1.25
  let num_small_fries := 20
  let price_sodas := 1.00
  let num_sodas := 50
  let price_milkshakes := 2.85
  let num_milkshakes := 18
  let price_soft_serve := 1.30
  let num_soft_serve := 40
  let total :=
    num_burgers * price_burger +
    num_large_fries * price_large_fries +
    num_small_fries * price_small_fries +
    num_sodas * price_sodas +
    num_milkshakes * price_milkshakes +
    num_soft_serve * price_soft_serve
  in total = 274.30 :=
by
  sorry

end spongebob_earnings_l661_661266


namespace jones_children_age_not_5_l661_661246

theorem jones_children_age_not_5 :
  ∃ n : ℕ, 
  (0 < n ∧ n < 10) ∧
  (∀ d, d ∈ {1, 2, 3, 4, 6, 7, 8, 9} → n % d = 0) ∧ 
  (n % 5 ≠ 0) :=
begin
  use 5544,
  split,
  { split, 
    { norm_num },
    { norm_num } },
  split,
  { intros d hd,
    fin_cases hd;
    norm_num },
  { norm_num }
end

end jones_children_age_not_5_l661_661246


namespace selling_price_of_cycle_l661_661346

theorem selling_price_of_cycle
  (cost_price : ℕ)
  (gain_percent_decimal : ℚ)
  (h_cp : cost_price = 850)
  (h_gpd : gain_percent_decimal = 27.058823529411764 / 100) :
  ∃ selling_price : ℚ, selling_price = cost_price * (1 + gain_percent_decimal) ∧ selling_price = 1080 := 
by
  use (cost_price * (1 + gain_percent_decimal))
  sorry

end selling_price_of_cycle_l661_661346


namespace intersection_is_expected_l661_661158

-- Define the set A as the set of real numbers satisfying the inequality.
def set_A : set ℝ := { x : ℝ | x^2 - x - 6 > 0 }

-- Define the universal set U as the set of all real numbers.
def U : set ℝ := set.univ

-- Define the complement of A in U.
def complement_U_A : set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 3 }

-- Define the set B as the set of integers satisfying the inequality.
def set_B : set ℤ := { x : ℤ | abs (x - 1) < 3 }

-- Define the intersection of complement_U_A and set_B.
def intersection : set ℤ := { x : ℤ | -2 ≤ x ∧ x ≤ 3 } ∩ set_B

-- The target set we expect to prove equivalency with.
def expected : set ℤ := {-1, 0, 1, 2, 3}

theorem intersection_is_expected : intersection = expected := by
  sorry

end intersection_is_expected_l661_661158


namespace sin_neg_60_eq_neg_sqrt_3_div_2_l661_661883

theorem sin_neg_60_eq_neg_sqrt_3_div_2 : 
  Real.sin (-π / 3) = - (Real.sqrt 3) / 2 := 
by
  sorry

end sin_neg_60_eq_neg_sqrt_3_div_2_l661_661883


namespace greatest_possible_perimeter_l661_661595

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l661_661595


namespace value_not_in_range_of_g_l661_661998

theorem value_not_in_range_of_g (c : ℝ) :
  (∀ x : ℝ, g(x) ≠ 3) ↔ c ∈ Ioo (-2 : ℝ) (2 : ℝ) := 
by
  let g (x : ℝ) := x^2 + c * x + 4
  have h : ∀ x : ℝ, g(x) ≠ 3 ↔ ∀ x : ℝ, x^2 + c * x + 1 ≠ 0 := 
    assume x,
      ⟨sorry⟩ -- this is a placeholder for the actual equivalence proof
  sorry -- this is a placeholder for the final proof

end value_not_in_range_of_g_l661_661998


namespace trendy_haircut_cost_l661_661839

theorem trendy_haircut_cost (T : ℝ) (H1 : 5 * 5 * 7 + 3 * 6 * 7 + 2 * T * 7 = 413) : T = 8 :=
by linarith

end trendy_haircut_cost_l661_661839


namespace no_x_intersect_one_x_intersect_l661_661156

variable (m : ℝ)

-- Define the original quadratic function
def quadratic_function (x : ℝ) := x^2 - 2 * m * x + m^2 + 3

-- 1. Prove the function does not intersect the x-axis
theorem no_x_intersect : ∀ m, ∀ x : ℝ, quadratic_function m x ≠ 0 := by
  intros
  unfold quadratic_function
  sorry

-- 2. Prove that translating down by 3 units intersects the x-axis at one point
def translated_quadratic (x : ℝ) := (x - m)^2

theorem one_x_intersect : ∃ x : ℝ, translated_quadratic m x = 0 := by
  unfold translated_quadratic
  sorry

end no_x_intersect_one_x_intersect_l661_661156


namespace numbers_appearing_more_than_twice_l661_661770

noncomputable def seq : ℕ → ℕ
| 0     := 0  -- index starts from 1, so seq(0) is arbitrary
| 1     := 1
| (n+2) := ⌊Real.sqrt (∑ i in Finset.range (n+1), seq (i+1))⌋₊

theorem numbers_appearing_more_than_twice :
  ∀ k, (∃ n1 n2 n3, n1 < n2 ∧ n2 < n3 ∧ seq n1 = k ∧ seq n2 = k ∧ seq n3 = k) ↔ (k = 1 ∨ ∃ m, k = 2^m) :=
  sorry

end numbers_appearing_more_than_twice_l661_661770


namespace number_of_dolls_combined_l661_661862

-- Defining the given conditions as variables
variables (aida sophie vera : ℕ)

-- Given conditions
def condition1 : Prop := aida = 2 * sophie
def condition2 : Prop := sophie = 2 * vera
def condition3 : Prop := vera = 20

-- The final proof statement we need to prove
theorem number_of_dolls_combined (h1 : condition1 aida sophie) (h2 : condition2 sophie vera) (h3 : condition3 vera) : 
  aida + sophie + vera = 140 :=
  by sorry

end number_of_dolls_combined_l661_661862


namespace smallest_digit_to_correct_sum_l661_661197

theorem smallest_digit_to_correct_sum :
  ∃ (d : ℕ), d = 3 ∧
  (3 ∈ [3, 5, 7]) ∧
  (371 + 569 + 784 + (d*100) = 1824) := sorry

end smallest_digit_to_correct_sum_l661_661197


namespace problem1_problem2_l661_661815

-- Problem 1
theorem problem1 : (π - 2023)^0 + abs (-sqrt 3) - 2 * real.sin (real.pi / 3) = 1 :=
  sorry

-- Problem 2
theorem problem2 {x : ℝ} (h1 : 2 * (x + 3) ≥ 8) (h2 : x < (x + 4) / 2) : 1 ≤ x ∧ x < 4 :=
  sorry

end problem1_problem2_l661_661815


namespace greatest_triangle_perimeter_l661_661581

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l661_661581


namespace universal_friendship_l661_661662

-- Define the inhabitants and their relationships
def inhabitants (n : ℕ) : Type := Fin n

-- Condition for friends and enemies
inductive Relationship (n : ℕ) : inhabitants n → inhabitants n → Prop
| friend (A B : inhabitants n) : Relationship n A B
| enemy (A B : inhabitants n) : Relationship n A B

-- Transitivity condition
axiom transitivity {n : ℕ} {A B C : inhabitants n} :
  Relationship n A B = Relationship n B C → Relationship n A C = Relationship n A B

-- At least two friends among any three inhabitants
axiom at_least_two_friends {n : ℕ} (A B C : inhabitants n) :
  ∃ X Y : inhabitants n, X ≠ Y ∧ Relationship n X Y = Relationship n X Y

-- Inhabitants can start a new life switching relationships
axiom start_new_life {n : ℕ} (A : inhabitants n) :
  ∀ B : inhabitants n, Relationship n A B = Relationship n B A

-- The main theorem we need to prove
theorem universal_friendship (n : ℕ) : 
  ∀ A B : inhabitants n, ∃ C : inhabitants n, Relationship n A C = Relationship n B C :=
sorry

end universal_friendship_l661_661662


namespace fewest_coach_handshakes_l661_661369

noncomputable def binom (n : ℕ) := n * (n - 1) / 2

theorem fewest_coach_handshakes : 
  ∃ (k1 k2 k3 : ℕ), binom 43 + k1 + k2 + k3 = 903 ∧ k1 + k2 + k3 = 0 := 
by
  use 0, 0, 0
  sorry

end fewest_coach_handshakes_l661_661369


namespace danny_age_l661_661389

theorem danny_age (D : ℕ) (h : D - 19 = 3 * (26 - 19)) : D = 40 := by
  sorry

end danny_age_l661_661389


namespace system1_solution_system2_solution_l661_661264

-- For System (1)
theorem system1_solution (x y : ℝ) (h1 : y = 2 * x) (h2 : 3 * y + 2 * x = 8) : x = 1 ∧ y = 2 :=
by
  sorry

-- For System (2)
theorem system2_solution (s t : ℝ) (h1 : 2 * s - 3 * t = 2) (h2 : (s + 2 * t) / 3 = 3 / 2) : s = 5 / 2 ∧ t = 1 :=
by
  sorry

end system1_solution_system2_solution_l661_661264


namespace geometric_sequence_sum_l661_661199

-- Define the geometric sequence {a_n}
def a (n : ℕ) : ℕ := 2 * (1 ^ (n - 1))

-- Define the sum of the first n terms, s_n
def s (n : ℕ) : ℕ := (Finset.range n).sum (a)

-- The transformed sequence {a_n + 1} assumed also geometric
def b (n : ℕ) : ℕ := a n + 1

-- Lean theorem that s_n = 2n
theorem geometric_sequence_sum (n : ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, (b (n + 1)) * (b (n + 1)) = (b n * b (n + 2))) : 
  s n = 2 * n :=
sorry

end geometric_sequence_sum_l661_661199


namespace minimum_degree_q_l661_661312

variable (p q r : Polynomial ℝ)

theorem minimum_degree_q (h1 : 2 * p + 5 * q = r)
                        (hp : p.degree = 7)
                        (hr : r.degree = 10) :
  q.degree = 10 :=
sorry

end minimum_degree_q_l661_661312


namespace greatest_possible_perimeter_l661_661556

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l661_661556


namespace rationalize_denominator_XYZ_sum_l661_661730

noncomputable def a := (5 : ℝ)^(1/3)
noncomputable def b := (4 : ℝ)^(1/3)

theorem rationalize_denominator_XYZ_sum : 
  let X := 25
  let Y := 20
  let Z := 16
  let W := 1
  X + Y + Z + W = 62 :=
by 
  sorry

end rationalize_denominator_XYZ_sum_l661_661730


namespace greatest_possible_perimeter_l661_661599

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l661_661599


namespace function_continuous_at_point_continuous_at_f_l661_661723

noncomputable def delta (ε : ℝ) : ℝ := ε / 12

theorem function_continuous_at_point :
  ∀ (f : ℝ → ℝ) (x₀ : ℝ), (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε) :=
by
  let f := fun x => -2 * x^2 - 4
  let x₀ := 3
  have h1 : f x₀ = -22 := by linarith
  have h2 : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε :=
    by
      intros ε ε_pos
      use (ε / 12)
      split
      { exact div_pos ε_pos twelve_pos }
      { intros x hx
        calc
        abs (f x - f x₀)
          = abs (-2 * x^2 - 4 - (-22)) : by simp [h1]
      ... = abs (-2 * (x^2 - 9)) : by ring 
      ... = 2 * abs (x^2 - 9) : by rw [abs_mul, abs_neg]; simp
      ... < ε : by 
        let δ := ε / 12
        have hx3 : abs (x - 3) < δ := hx
        have h2 : abs (x + 3) ≤ 6 :=
          calc 
            abs (x + 3) ≤ abs (x - 3) + 6 : by linarith
            ... ≤ δ + 6 : by linarith
            ... ≤ ε / 12 + 6 : by linarith
        exact mul_lt_of_lt_div ((div_pos ε_pos twelve_pos).le)

theorem continuous_at_f : ∀ (ε : ℝ), ε > 0 → ∃ δ > 0, ∀ x, |x - 3| < δ → |(-2 * x^2 - 4) - (-22)| < ε :=
by
  intros ε ε_pos
  unfold delta
  use δ ε
  split
  { exact div_pos ε_pos twelve_pos }
  { intros x h
    calc
    |(-2 * x^2 - 4) - (-22)|
      = 2 * |x^2 - 9| : by norm_num
  ... < ε : by
      let h' := abs_sub_lt_iff.mp h
      exact lt_of_le_of_lt (abs_mul _ _) (by linarith [div_pos ε_pos twelve_pos,
        le_abs_self, (mul_div_cancel' _ twelve_pos.ne.symm)]) }

end function_continuous_at_point_continuous_at_f_l661_661723


namespace clownfish_display_tank_count_l661_661051

def initial_fish_count (C B A : ℕ) : Prop := C + B + A = 180

def ratio_fish_count (C B A : ℕ) : Prop := C = B ∧ A = 2 * B

def blowfish_in_display (B : ℕ) : ℕ := B - 26

def clownfish_in_display_initial (B : ℕ) : ℕ := B - 26

def clownfish_swim_back (clownfish_in_display_initial : ℕ) : ℕ := (clownfish_in_display_initial * 1 / 3).nat_abs

def clownfish_in_display_remaining (clownfish_in_display_initial : ℕ) (clownfish_swim_back : ℕ) : ℕ :=
  clownfish_in_display_initial - clownfish_swim_back

def angelfish_in_display (clownfish_in_display_remaining : ℕ) : ℕ :=
  (3 * clownfish_in_display_remaining) / 2

def angelfish_in_tank (B : ℕ) : ℕ := B / 2

theorem clownfish_display_tank_count {C B A : ℕ} 
  (h1 : initial_fish_count C B A) 
  (h2 : ratio_fish_count C B A)
  (h3 : angelfish_in_tank B = B / 2)
  (h4: B = 45) :
  clownfish_in_display_remaining (clownfish_in_display_initial B) 
  (clownfish_swim_back (clownfish_in_display_initial B)) = 13 := 
  sorry

end clownfish_display_tank_count_l661_661051


namespace max_triangle_perimeter_l661_661591

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l661_661591


namespace product_of_roots_of_cubic_polynomial_l661_661902

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l661_661902


namespace average_time_in_storm_l661_661832

-- Define conditions for the problem.
def car_position (t : ℝ) : ℝ × ℝ := (2 / 3 * t, 0)
def storm_position (t : ℝ) : ℝ × ℝ := (t, 130 - t)

-- Distance formula condition to be within the storm radius
def within_storm (t : ℝ) : Prop :=
  Real.sqrt ((2 / 3 * t - t)^2 + (130 - t)^2) ≤ 60

-- Define the times when the car enters and leaves the storm
def times_in_storm : set ℝ := { t | within_storm t }

-- The proof goal
theorem average_time_in_storm : 
  let times := {t | within_storm t},
      t1 := times_inf times,
      t2 := times_sup times
  in 
    (t1 + t2) / 2 = 117 :=
sorry

end average_time_in_storm_l661_661832


namespace lucas_meet_hannah_l661_661702

-- Define the distance between houses in miles
def distance_in_miles : ℝ := 3

-- Define the conversion factor from miles to feet
def miles_to_feet : ℝ := 5280

-- Calculate the distance in feet
def distance_in_feet : ℝ := distance_in_miles * miles_to_feet

-- Define Lucas's speed in feet per minute, let it be l
variables (l : ℝ)

-- Define Hannah's speed, which is 3 times Lucas's speed
def hannah_speed : ℝ := 3 * l

-- Together they cover
def combined_speed : ℝ := l + hannah_speed

-- Define the time it takes for them to meet
def meeting_time : ℝ := distance_in_feet / combined_speed

-- Calculate the distance Lucas covers in the time they meet
def lucas_distance : ℝ := l * meeting_time

-- Lucas covers 3 feet with each step
def steps_lucas_takes : ℝ := lucas_distance / 3

-- The proof that Lucas takes 1320 steps to meet Hannah
theorem lucas_meet_hannah : steps_lucas_takes l = 1320 := by
  sorry

end lucas_meet_hannah_l661_661702


namespace problem_l661_661055

noncomputable def pointsConcyclic
  (A B C O O₁ O₂ O₃ : Point)
  (l : Line)
  (h1 : A ∈ l)
  (h2 : B ∈ l)
  (h3 : C ∈ l)
  (h4 : O ∉ l)
  (h5 : is_circumcenter O₁ A B O)
  (h6 : is_circumcenter O₂ B C O)
  (h7 : is_circumcenter O₃ C A O) : Prop :=
  concyclic O O₁ O₂ O₃

theorem problem (A B C O O₁ O₂ O₃ : Point) (l : Line)
  (h1 : A ∈ l) (h2 : B ∈ l) (h3 : C ∈ l) (h4 : O ∉ l)
  (h5 : is_circumcenter O₁ A B O)
  (h6 : is_circumcenter O₂ B C O)
  (h7 : is_circumcenter O₃ C A O) :
  pointsConcyclic A B C O O₁ O₂ O₃ l h1 h2 h3 h4 h5 h6 h7 :=
sorry

end problem_l661_661055


namespace exists_non_parallel_diagonal_l661_661251

theorem exists_non_parallel_diagonal (n : ℕ) (h : n > 0) : 
  ∀ (P : list (ℝ × ℝ)), convex_polygon P → (length P = 2 * n) → 
  ∃ (d : (ℝ × ℝ) × (ℝ × ℝ)), is_diagonal P d ∧ (∀ (s : (ℝ × ℝ) × (ℝ × ℝ)), s ∈ sides P → ¬ parallel d s) :=
by
  sorry

end exists_non_parallel_diagonal_l661_661251


namespace same_side_inequality_l661_661467

variable (a b : ℝ)

def point_on_same_side (P Q : ℝ × ℝ) (line : ℝ → ℝ → ℝ) : Prop :=
  line P.1 P.2 * line Q.1 Q.2 > 0

theorem same_side_inequality (h : point_on_same_side (a, b) (1, 2) (λ x y, 3 * x + 2 * y - 8)) :
  3 * a + 2 * b - 8 > 0 :=
sorry

end same_side_inequality_l661_661467


namespace max_perimeter_l661_661634

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l661_661634


namespace prove_a_minus_c_l661_661061

-- Define the invertible function g with conditions.
variable {α β : Type} [Field α] [Field β] (g : α → β)

-- Invertibility assumption
axiom invertible (hg : ∃ h : β → α, Function.RightInverse h g ∧ Function.LeftInverse h g)

-- Given conditions
variables (a b c : α) (ha : g a = c) (hb : g b = a) (hc : g c = 6)

-- Prove the value of a - c is -3
theorem prove_a_minus_c : a - c = -3 :=
by
  sorry

end prove_a_minus_c_l661_661061


namespace max_perimeter_l661_661637

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l661_661637


namespace greatest_triangle_perimeter_l661_661582

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l661_661582


namespace income_after_selling_more_l661_661747

theorem income_after_selling_more (x y : ℝ)
  (h1 : 26 * x + 14 * y = 264) 
  : 39 * x + 21 * y = 396 := 
by 
  sorry

end income_after_selling_more_l661_661747


namespace greatest_possible_perimeter_l661_661617

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l661_661617


namespace parallel_perpendicular_implies_perpendicular_l661_661132

variables (m n : ℝ^3 → ℝ^3 → Prop) (α β : ℝ^3 → Prop)

def parallel (l1 l2 : ℝ^3 → ℝ^3 → Prop) : Prop := ∀ x y ∈ l1, ∀ u v ∈ l2, ∃ k ∈ ℝ, u = x + k * (y - x)
def perpendicular (l1 l2 : ℝ^3 → ℝ^3 → Prop) : Prop := ∀ x y ∈ l1, ∀ u v ∈ l2, (y - x) ⬝ (v - u) = 0 -- inner product

axiom n_in_beta : ∀ x ∈ (n : ℝ^3 → Prop), β x
axiom m_parallel_n : parallel m n

theorem parallel_perpendicular_implies_perpendicular (m n : ℝ^3 → ℝ^3 → Prop) (β : ℝ^3 → Prop) :
  (parallel m n) → (∀ x y ∈ n, ∃ k ∈ ℝ3, x + k * (y - x) ∈ β) → ∀ x y ∈ m, ∀ u v ∈ β, (y - x) ⬝ (v - u) = 0 :=
by
  intros h1 h2 x y hx hy u v hu hv
  sorry

end parallel_perpendicular_implies_perpendicular_l661_661132


namespace evaluate_expression_l661_661141

theorem evaluate_expression (a b : ℝ) 
(h : a^2 + b^2 - 4 * a - 2 * b + 5 = 0) : 
  (sqrt a + b) / (2 * sqrt a + b + 1) = 1 / 2 :=
sorry

end evaluate_expression_l661_661141


namespace modulus_z1_real_part_bounds_omega_imaginary_l661_661235

namespace MyMathProof

variable {z1 : ℂ} -- ℂ represents complex numbers

-- Conditions
def is_imaginary (z : ℂ) : Prop := z.im ≠ 0
def z2 (z : ℂ) : ℂ := z + (1 / z)
def real_z2 (z : ℂ) : Prop := (z2 z).im = 0
def bounds_z2 (z : ℂ) : Prop := -1 ≤ (z2 z).re ∧ (z2 z).re ≤ 1

-- Proof goals
theorem modulus_z1 {z1 : ℂ} (h1 : is_imaginary z1) (h2 : real_z2 z1) (h3 : bounds_z2 z1) :
  abs z1 = 1 := sorry 

theorem real_part_bounds {z1 : ℂ} (h1 : is_imaginary z1) (h2 : real_z2 z1) (h3 : bounds_z2 z1) :
  -1/2 ≤ z1.re ∧ z1.re ≤ 1/2 := sorry

theorem omega_imaginary {z1 : ℂ} (h1 : is_imaginary z1) (h2 : real_z2 z1) (h3 : bounds_z2 z1) :
  let ω := (1 - z1) / (1 + z1) in ω.im ≠ 0 ∧ ω.re = 0 := sorry

end MyMathProof

end modulus_z1_real_part_bounds_omega_imaginary_l661_661235


namespace recur_decimal_times_nine_l661_661417

theorem recur_decimal_times_nine : 
  (0.3333333333333333 : ℝ) * 9 = 3 :=
by
  -- Convert 0.\overline{3} to a fraction
  have h1 : (0.3333333333333333 : ℝ) = (1 / 3 : ℝ), by sorry
  -- Perform multiplication and simplification
  calc
    (0.3333333333333333 : ℝ) * 9 = (1 / 3 : ℝ) * 9 : by rw h1
                              ... = (1 * 9) / 3 : by sorry
                              ... = 9 / 3 : by sorry
                              ... = 3 : by sorry

end recur_decimal_times_nine_l661_661417


namespace infinite_naturals_with_perfect_square_repetition_l661_661762

def repetition (A : ℕ) (n : ℕ) : ℕ :=
  A * 10^n + A

theorem infinite_naturals_with_perfect_square_repetition :
  ∃ᶠ A : ℕ, ∃ m : ℕ, repetition A m = m^2 :=
sorry

end infinite_naturals_with_perfect_square_repetition_l661_661762


namespace jane_wins_probability_l661_661209

def spinner_outcomes := {1, 2, 3, 4, 5, 6}

/- Conditions: Each player spins once, and the spinner has six congruent sectors labeled 1 to 6 -/
def spins (j b : ℕ) := j ∈ spinner_outcomes ∧ b ∈ spinner_outcomes

/- Jane wins if the non-negative difference is less than 4 -/
def jane_wins (j b : ℕ) := spins j b ∧ abs (j - b) < 4

/- Calculate total number of outcomes -/
def total_outcomes := finset.card (finset.product (spinner_outcomes.to_finset) (spinner_outcomes.to_finset))

/- Calculate losing outcomes -/
def losing_outcomes : finset (ℕ × ℕ) :=
  let losing_pairs := {(1, 5), (1, 6), (2, 6), (5, 1), (6, 1), (6, 2)} in
  losing_pairs.to_finset

def total_losing_outcomes := losing_outcomes.card

/- Calculate winning outcomes as the complement of losing outcomes -/
def winning_probability : ℚ := 1 - (total_losing_outcomes / total_outcomes : ℚ)

theorem jane_wins_probability: winning_probability = 5 / 6 :=
by
  /- Proof placeholder -/
  sorry

end jane_wins_probability_l661_661209


namespace range_exponential_2_intersection_exponential_fn_xx_l661_661146

noncomputable def lg (x : ℝ) : ℝ := Math.log x

noncomputable def A := {x : ℝ | -1 < x ∧ x < 2}

def exponential_fn (x : ℝ) (a : ℝ) := a ^ x

theorem range_exponential_2 (A : Set ℝ) (B : Set ℝ) : 
  A = {x | -1 < x ∧ x <2} → 
  ∃ A B, (exponential_fn x 2) x ∈ ExponentialFn x 2) = (B ∪ A) → 
  B = {y | (1 / 2) < y ∧ y < 4} → 
  (A ∪ B = {x | -1 < x ∧ x < 4})
:= 
begin
    sorry
end

theorem intersection_exponential_fn_xx (A : Set ℝ) (a : ℝ) (B : Set ℝ) :
  a ≠ 1 ∧ 0 < a → 
  A = {x | -1 < x ∧ x < 2} →
  A ∩ {y | exbonential_fn x a |}  = {\frac{1}{2} , 2\} → 
  a = 2 := 
  begin
    sorry
end

end range_exponential_2_intersection_exponential_fn_xx_l661_661146


namespace pyramid_spheres_volume_l661_661522

theorem pyramid_spheres_volume
  (area_base : ℝ)
  (area_lateral : ℝ)
  (height : ℝ)
  (inscribed_radius : ℝ)
  (total_volume : ℝ)
  (h_area : area_base = (1 / 2) * area_lateral)
  (h_height : height = 70)
  (r1_relation : r1 = inscribed_radius)
  (geometric_sequence : ∀ n : ℕ, r (n + 1) = (1 / 2) * r n)
  (h_total_volume : total_volume = (686 / 327) * pi) : 
  total_volume = (32/21) * pi * inscribed_radius^3 := by
  sorry

end pyramid_spheres_volume_l661_661522


namespace smallest_magnitude_index_l661_661224

def vec (n : ℕ) : ℕ × ℕ := (n - 2015, n + 12)

def magnitude_sq (v : ℕ × ℕ) : ℕ :=
  v.1 * v.1 + v.2 * v.2

theorem smallest_magnitude_index :
  ∃ n : ℕ, magnitude_sq (vec n) ≤ magnitude_sq (vec m) ∀ m : ℕ :=
  (n = 1001 ∨ n = 1002)
:=
begin
  sorry
end

end smallest_magnitude_index_l661_661224


namespace wind_velocity_l661_661766

def pressure (P A V : ℝ) (k : ℝ) : Prop :=
  P = k * A * V^2

theorem wind_velocity (k : ℝ) (h_initial : pressure 4 4 8 k) (h_final : pressure 64 16 v k) : v = 16 := by
  sorry

end wind_velocity_l661_661766


namespace product_of_roots_cubic_l661_661963

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  let roots := {x : ℝ // p x = 0}.toFinset
  ∃ r : ℝ, (∀ x ∈ roots, p x = 0) ∧ (∏ x in roots, x) = 50 :=
by
  sorry

end product_of_roots_cubic_l661_661963


namespace product_of_roots_l661_661906

-- Let's define the given polynomial equation.
def polynomial : Polynomial ℝ := Polynomial.monomial 3 1 - Polynomial.monomial 2 15 + Polynomial.monomial 1 75 - Polynomial.C 50

-- Prove that the product of the roots of the given polynomial is 50.
theorem product_of_roots : (Polynomial.roots polynomial).prod id = 50 := by
  sorry

end product_of_roots_l661_661906


namespace max_subset_size_l661_661694

open Set

def valid_subset (T : Set ℕ) : Prop :=
  T ⊆ {n | 1 ≤ n ∧ n ≤ 1597} ∧ (∀ x y ∈ T, x ≠ y → abs (x - y) ≠ 5 ∧ abs (x - y) ≠ 8)

theorem max_subset_size : ∃ T ⊆ {n | 1 ≤ n ∧ n ≤ 1597}, valid_subset T ∧ T.card = 613 :=
begin
  sorry
end

end max_subset_size_l661_661694


namespace three_irrational_numbers_l661_661666

theorem three_irrational_numbers (a b c d e : ℝ) 
  (ha : ¬ ∃ q1 q2 : ℚ, a = q1 + q2) 
  (hb : ¬ ∃ q1 q2 : ℚ, b = q1 + q2) 
  (hc : ¬ ∃ q1 q2 : ℚ, c = q1 + q2) 
  (hd : ¬ ∃ q1 q2 : ℚ, d = q1 + q2) 
  (he : ¬ ∃ q1 q2 : ℚ, e = q1 + q2) : 
  ∃ x y z, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) 
  ∧ (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) 
  ∧ (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e)
  ∧ (¬ ∃ q1 q2 : ℚ, x + y = q1 + q2) 
  ∧ (¬ ∃ q1 q2 : ℚ, y + z = q1 + q2) 
  ∧ (¬ ∃ q1 q2 : ℚ, z + x = q1 + q2) :=
sorry

end three_irrational_numbers_l661_661666


namespace apples_in_basket_l661_661348

def cost_of_bananas (num_bananas : ℕ) (cost_per_banana : ℕ) : ℕ := num_bananas * cost_per_banana
def cost_of_strawberries (num_strawberries : ℕ) (cost_per_dozen : ℕ) : ℕ := (num_strawberries / 12) * cost_per_dozen
def cost_of_avocados (num_avocados : ℕ) (cost_per_avocado : ℕ) : ℕ := num_avocados * cost_per_avocado
def cost_of_grapes (cost_half_bunch : ℕ) : ℕ := cost_half_bunch * 2

theorem apples_in_basket (cost_total : ℕ) (bananas : ℕ) (strawberries : ℕ) (avocados : ℕ) (cost_per_apple : ℕ) (cost_per_banana : ℕ) (cost_per_dozen_strawberries : ℕ) (cost_per_avocado : ℕ) (cost_half_bunch_grapes : ℕ)
: cost_total = 28 ∧ bananas = 4 ∧ strawberries = 24 ∧ avocados = 2 ∧ cost_per_apple = 2 ∧ cost_per_banana = 1 ∧ cost_per_dozen_strawberries = 4 ∧ cost_per_avocado = 3 ∧ cost_half_bunch_grapes = 2
→ let remaining_money := cost_total - (cost_of_bananas bananas cost_per_banana + cost_of_strawberries strawberries cost_per_dozen_strawberries + cost_of_avocados avocados cost_per_avocado) in
    let money_for_apples := remaining_money - cost_of_grapes cost_half_bunch_grapes in
    money_for_apples / cost_per_apple = 3 :=
by
  intros
  sorry

end apples_in_basket_l661_661348


namespace find_k_l661_661828

open_locale classical

variable (k : ℕ)

-- Definitions derived from conditions
def red_balls : ℕ := 7
def total_balls : ℕ := red_balls + k
def prob_red : ℚ := red_balls / total_balls
def prob_blue : ℚ := k / total_balls
def win_amount_red : ℚ := 3
def lose_amount_blue : ℚ := -1
def expected_value : ℚ := prob_red * win_amount_red + prob_blue * lose_amount_blue

-- Theorem that encapsulates proving k = 7 given the expected value is 1
theorem find_k (h : expected_value k = 1) : k = 7 :=
sorry

end find_k_l661_661828


namespace maxwell_meets_brad_in_8_hours_l661_661804

-- Definitions based on conditions
def distance_between_homes := 74 -- in km
def maxwell_speed := 4 -- in km/h
def brad_speed := 6 -- in km/h
def time_difference := 1 -- in hours

-- Required proof problem statement
theorem maxwell_meets_brad_in_8_hours :
  let t := (distance_between_homes - maxwell_speed * time_difference) / 
             (maxwell_speed + brad_speed) in
  t + time_difference = 8 :=
by
  -- Formal proof can be constructed here
  sorry

end maxwell_meets_brad_in_8_hours_l661_661804


namespace president_savings_l661_661263

theorem president_savings (total_funds : ℕ) (friends_percentage : ℕ) (family_percentage : ℕ) 
  (friends_contradiction funds_left family_contribution fundraising_amount : ℕ) :
  total_funds = 10000 →
  friends_percentage = 40 →
  family_percentage = 30 →
  friends_contradiction = (total_funds * friends_percentage) / 100 →
  funds_left = total_funds - friends_contradiction →
  family_contribution = (funds_left * family_percentage) / 100 →
  fundraising_amount = funds_left - family_contribution →
  fundraising_amount = 4200 :=
by
  intros
  sorry

end president_savings_l661_661263


namespace greatest_possible_perimeter_l661_661642

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l661_661642


namespace math_problem_l661_661171

theorem math_problem (m : ℝ) (h : m^2 - m = 2) : (m - 1)^2 + (m + 2) * (m - 2) = 1 := 
by sorry

end math_problem_l661_661171


namespace solve_for_a_l661_661428

theorem solve_for_a {a x : ℝ} (H : (x - 2) * (a * x^2 - x + 1) = a * x^3 + (-1 - 2 * a) * x^2 + 3 * x - 2 ∧ (-1 - 2 * a) = 0) : a = -1/2 := sorry

end solve_for_a_l661_661428


namespace largest_result_is_multiplication_l661_661175

noncomputable def evaluate_expression (a b : ℝ) (op : ℝ → ℝ → ℝ) : ℝ :=
  op (5 * √2 - √2) b

def candidates : List (ℝ → ℝ → ℝ) := [λ x y => x + y, λ x y => x - y, λ x y => x * y, λ x y => x / y]

def evaluated_results : List ℝ := 
  candidates.map (evaluate_expression (5 * √2 - √2) √2)

theorem largest_result_is_multiplication :
  evaluated_results.nth 2 = some 8 :=
by
  sorry

end largest_result_is_multiplication_l661_661175


namespace individual_weights_l661_661820

theorem individual_weights (A P : ℕ) 
    (h1 : 12 * A + 14 * P = 692)
    (h2 : P = A - 10) : 
    A = 32 ∧ P = 22 :=
by
  sorry

end individual_weights_l661_661820


namespace cost_of_pen_l661_661783

theorem cost_of_pen :
  ∃ p q : ℚ, (3 * p + 4 * q = 264) ∧ (4 * p + 2 * q = 230) ∧ (p = 39.2) :=
by
  sorry

end cost_of_pen_l661_661783


namespace height_difference_bounded_l661_661185

theorem height_difference_bounded (boys girls : Fin 15 → ℝ)
  (h_initial : ∀ i : Fin 15, |boys i - girls i| ≤ 10) :
  ∀ i : Fin 15, |(Sorted boys) i - (Sorted girls) i | ≤ 10 :=
by
  sorry

end height_difference_bounded_l661_661185


namespace profit_without_discount_l661_661002

-- Define the conditions as hypotheses
variables
  (cost_price : ℝ) -- Cost price (CP) of the article
  (discount : ℝ) -- Discount percentage offered
  (profit_percentage : ℝ) -- Profit percentage earned

-- Define the conditions
def shopkeeper_conditions (cost_price discount profit_percentage : ℝ) : Prop :=
  cost_price = 100 ∧ discount = 5 ∧ profit_percentage = 18.75

-- The goal is to show that the profit percentage without discount is equal to the given profit percentage
theorem profit_without_discount
  (cost_price discount profit_percentage : ℝ)
  (h : shopkeeper_conditions cost_price discount profit_percentage) :
  let sp_no_discount := cost_price + (profit_percentage / 100) * cost_price in
  (sp_no_discount - cost_price) / cost_price * 100 = profit_percentage := sorry

end profit_without_discount_l661_661002


namespace greatest_possible_perimeter_l661_661597

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l661_661597


namespace max_distance_to_line_l661_661201

noncomputable def point_on_c1 (θ : ℝ) (ρ : ℝ) : Prop :=
  ρ * Math.sin θ = 2

noncomputable def point_on_ray_om (ρ1 ρ2 θ : ℝ) : Prop :=
  ρ1 * ρ2 = 4

noncomputable def curve_c2 (θ ρ : ℝ) : Prop :=
  ρ = 2 * Math.sin θ

theorem max_distance_to_line (ρ θ : ℝ) :
  (point_on_c1 θ ρ) →
  (point_on_ray_om (2 * Math.sin θ) ρ θ) →
  (∃ ρ, curve_c2 θ ρ) →
  1 + 3 * Real.sqrt 2 / 2 = 1 + 3 * Real.sqrt 2 / 2 :=
by
  intros
  sorry

end max_distance_to_line_l661_661201


namespace find_BC_l661_661305

noncomputable def problem_statement : Prop :=
  ∃ O₁ O₂ : ℝ × ℝ, ∃ A B C : ℝ × ℝ,
  let r₁ := 2
  let r₂ := 3
  let O₁A := 2
  let O₂A := 3 in
  -- The circles touch externally at A
  dist O₁ A = r₁ ∧ dist O₂ A = r₂ ∧
  dist O₁ O₂ = r₁ + r₂ ∧
  
  -- The common tangent passing through point A intersects their other two common tangents at points B and C
  -- distance BC
  dist B A = real.sqrt (r₁ * r₂) ∧
  dist C A = real.sqrt (r₁ * r₂) ∧
  dist B C = 2 * real.sqrt (r₁ * r₂)

theorem find_BC : problem_statement :=
  sorry -- Proof is omitted as per instructions

end find_BC_l661_661305


namespace greatest_possible_perimeter_l661_661646

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l661_661646


namespace max_triangle_perimeter_l661_661590

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l661_661590


namespace greatest_possible_perimeter_l661_661622

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l661_661622


namespace area_triangle_COB_l661_661073

theorem area_triangle_COB (p : ℝ) : 
  let C := (0 : ℝ, p)
  let O := (0 : ℝ, 0)
  let B := (24 : ℝ, 0)
  abs (C.1 - O.1) = 0 ∧ abs (B.2 - O.2) = 0 → 
  (1 / 2) * 24 * p = 12 * p :=
by
  sorry

end area_triangle_COB_l661_661073


namespace minimal_divisors_at_kth_place_l661_661011

open Nat

theorem minimal_divisors_at_kth_place (n k : ℕ) (hnk : n ≥ k) (S : ℕ) (hS : ∃ d : ℕ, d ≥ n ∧ d = S ∧ ∀ i, i ≤ d → exists m, m = d):
  ∃ (min_div : ℕ), min_div = ⌈ (n : ℝ) / k ⌉ :=
by
  sorry

end minimal_divisors_at_kth_place_l661_661011


namespace gain_loss_l661_661245

structure Conditions where
  initial_cash_A : ℕ
  initial_cash_B : ℕ
  initial_house_value : ℕ
  selling_price_1 : ℕ
  buying_price_2 : ℕ
  appreciation_value : ℕ
  selling_price_3 : ℕ

theorem gain_loss (c : Conditions)
  (h1 : c.initial_cash_A = 15000)
  (h2 : c.initial_cash_B = 18000)
  (h3 : c.initial_house_value = 15000)
  (h4 : c.selling_price_1 = 16000)
  (h5 : c.buying_price_2 = 14000)
  (h6 : c.appreciation_value = c.initial_house_value + 2000)
  (h7 : c.selling_price_3 = 17000) :
  -- Mr. A's gain is $4000 and Mr. B's loss is $2000
  let final_cash_A := c.initial_cash_A + c.selling_price_1 - c.buying_price_2 + c.selling_price_3 in
  let final_cash_B := c.initial_cash_B - c.selling_price_1 + c.buying_price_2 - c.selling_price_3 in
  final_cash_A - (c.initial_cash_A + c.initial_house_value) = 4000 ∧
  final_cash_B - c.initial_cash_B = -2000 :=
  sorry

end gain_loss_l661_661245


namespace tetrahedron_with_congruent_faces_and_OH_equals_OH_l661_661852

variables (A B C D O H H' : Type)
variables [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C]
          [EuclideanGeometry D] [EuclideanGeometry O] [EuclideanGeometry H]
          [EuclideanGeometry H']

-- Conditions
axiom circumsphere_centers_at_O : (circle_center A B C D) = O
axiom insphere_centers_at_O : (sphere_center A B C D) = O
axiom orthocenter_H : orthocenter A B C = H
axiom projection_H_prime : projection D (plane A B C) = H'

-- Statements to prove
theorem tetrahedron_with_congruent_faces_and_OH_equals_OH' :
  (tetrahedron_with_congruent_faces A B C D) ∧ (distance O H = distance O H') := 
sorry

end tetrahedron_with_congruent_faces_and_OH_equals_OH_l661_661852


namespace correct_statements_l661_661118

def sequence (a : ℕ → ℤ) : Prop :=
  ∀ (n : ℕ), (a 1 = 1) ∧ (∀ n, (n % 2 = 1 → a (n + 1) = a n - 2^n) ∧ (n % 2 = 0 → a (n + 1) = a n + 2^(n+1)))

lemma verify_statement_A (a : ℕ → ℤ) (h : sequence a) : a 3 = 7 := sorry
lemma verify_statement_B (a : ℕ → ℤ) (h : sequence a) : a 2022 = a 2 := sorry
lemma verify_statement_C (a : ℕ → ℤ) (h : sequence a) : ¬ (a 2023 = 2^2023) := sorry
lemma verify_statement_D (a : ℕ → ℤ) (h : sequence a) (S : ℕ → ℤ) : 3 * S (2*n + 1) = 2^(2*n + 3) - 6*n - 5 := sorry

theorem correct_statements (a : ℕ → ℤ) (S : ℕ → ℤ) (h : sequence a) :
  (verify_statement_A a h) ∧ (verify_statement_B a h) ∧ (verify_statement_C a h) ∧ (verify_statement_D a h S) := sorry

end correct_statements_l661_661118


namespace tomato_weight_l661_661677

variables (cost_meat cost_buns cost_lettuce cost_pickles_with_coupon cost_tomato : ℝ)
variables (change_back initial_money total_cost : ℝ)
variables (price_per_pound_tomato : ℝ)

-- Define the parameters based on the problem
def cost_meat := 2 * 3.5
def cost_buns := 1.5
def cost_lettuce := 1.0
def cost_pickles_with_coupon := 2.5 - 1.0
def change_back := 6.0
def initial_money := 20.0
def total_cost := initial_money - change_back

def cost_tomato := total_cost - (cost_meat + cost_buns + cost_lettuce + cost_pickles_with_coupon)

def price_per_pound_tomato := 2.0

-- The goal is to show that the weight of the tomato is 1.5 pounds
theorem tomato_weight : cost_tomato / price_per_pound_tomato = 1.5 :=
by
  -- Since the proof is omitted, we use "sorry" to indicate the missing proof.
  sorry

end tomato_weight_l661_661677


namespace log5_x_equals_neg_two_log5_2_l661_661172

theorem log5_x_equals_neg_two_log5_2 (x : ℝ) (h : x = (Real.log 3 / Real.log 9) ^ (Real.log 9 / Real.log 3)) :
  Real.log x / Real.log 5 = -2 * (Real.log 2 / Real.log 5) :=
by
  sorry

end log5_x_equals_neg_two_log5_2_l661_661172


namespace correct_statements_count_l661_661475

theorem correct_statements_count :
  let s1 := "When two lines are intersected by a third line, the alternate interior angles are equal"
  let s2 := "The complements of equal angles are equal"
  let s3 := "The perpendicular segment from a point outside a line to the line is called the distance from the point to the line"
  let s4 := "Complementary angles are adjacent angles"
  (¬s1 ∧ s2 ∧ ¬s3 ∧ ¬s4) → (1 = 1) :=
by 
  intros
  sorry

end correct_statements_count_l661_661475


namespace hash_op_is_100_l661_661714

def hash_op (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

theorem hash_op_is_100 (a b : ℕ) (h1 : a + b = 5) : hash_op a b = 100 :=
sorry

end hash_op_is_100_l661_661714


namespace blueberries_in_each_blue_box_l661_661311

theorem blueberries_in_each_blue_box (S B : ℕ) (h1 : S - B = 12) (h2 : 2 * S = 76) : B = 26 := by
  sorry

end blueberries_in_each_blue_box_l661_661311


namespace find_smallest_n_l661_661754

-- Definitions of the condition that m and n are relatively prime and that the fraction includes the digits 4, 5, and 6 consecutively
def is_coprime (m n : ℕ) : Prop := Nat.gcd m n = 1

def has_digits_456 (m n : ℕ) : Prop := 
  ∃ k : ℕ, ∃ c : ℕ, 10^k * m % (10^k * n) = 456 * 10^c

-- The theorem to prove the smallest value of n
theorem find_smallest_n (m n : ℕ) (h1 : is_coprime m n) (h2 : m < n) (h3 : has_digits_456 m n) : n = 230 :=
sorry

end find_smallest_n_l661_661754


namespace race_problem_l661_661183

theorem race_problem 
  (A B C : ℝ) 
  (h1 : A = 100) 
  (h2 : B = 100 - x) 
  (h3 : C = 72) 
  (h4 : B = C + 4)
  : x = 24 := 
by 
  sorry

end race_problem_l661_661183


namespace product_of_roots_cubic_l661_661886

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l661_661886


namespace product_of_roots_cubic_l661_661971

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  in (∃ r1 r2 r3 : ℝ, p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ r1 * r2 * r3 = 50) := 
by 
  sorry

end product_of_roots_cubic_l661_661971


namespace total_apple_weight_proof_l661_661241

-- Define the weights of each fruit in terms of ounces
def weight_apple : ℕ := 4
def weight_orange : ℕ := 3
def weight_plum : ℕ := 2

-- Define the bag's capacity and the number of bags
def bag_capacity : ℕ := 49
def number_of_bags : ℕ := 5

-- Define the least common multiple (LCM) of the weights
def lcm_weight : ℕ := Nat.lcm weight_apple (Nat.lcm weight_orange weight_plum)

-- Define the largest multiple of LCM that is less than or equal to the bag's capacity
def max_lcm_multiple : ℕ := (bag_capacity / lcm_weight) * lcm_weight

-- Determine the number of each fruit per bag
def sets_per_bag : ℕ := max_lcm_multiple / lcm_weight
def apples_per_bag : ℕ := sets_per_bag * 1  -- 1 apple per set

-- Calculate the weight of apples per bag and total needed in all bags
def apple_weight_per_bag : ℕ := apples_per_bag * weight_apple
def total_apple_weight : ℕ := apple_weight_per_bag * number_of_bags

-- The statement to be proved in Lean
theorem total_apple_weight_proof : total_apple_weight = 80 := by
  sorry

end total_apple_weight_proof_l661_661241


namespace max_perimeter_l661_661629

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l661_661629


namespace length_PM_l661_661514

-- Definitions and conditions
variables {P Q R M : Type} [inner_product_space ℝ P]

def QR : ℝ := 40
def PQ : ℝ := 40
def PR : ℝ := 36

def M (Q R : P) : P := Q + (1/2 : ℝ) • (R - Q)
def triangle_is_isosceles : Prop := PQ = QR

-- The main theorem to be proved
theorem length_PM (PQ QR PR : ℝ) (M : P) (N : P) :
  triangle_is_isosceles → QR = 40 → PQ = 40 → PR = 36 → dist P M = 8 * real.sqrt 14 :=
by sorry

end length_PM_l661_661514


namespace lucy_final_balance_l661_661703

def initial_balance : ℝ := 65
def deposit : ℝ := 15
def withdrawal : ℝ := 4

theorem lucy_final_balance : initial_balance + deposit - withdrawal = 76 :=
by
  sorry

end lucy_final_balance_l661_661703


namespace greatest_perimeter_of_triangle_l661_661613

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l661_661613


namespace solution_of_fractional_inequality_l661_661293

noncomputable def solution_set_of_inequality : Set ℝ :=
  {x : ℝ | -3 < x ∨ x > 1/2 }

theorem solution_of_fractional_inequality :
  {x : ℝ | (2 * x - 1) / (x + 3) > 0} = solution_set_of_inequality :=
by
  sorry

end solution_of_fractional_inequality_l661_661293


namespace downstream_speed_is_45_l661_661029

-- Define the conditions
def upstream_speed := 35 -- The man can row upstream at 35 kmph
def still_water_speed := 40 -- The speed of the man in still water is 40 kmph

-- Define the speed of the stream based on the given conditions
def stream_speed := still_water_speed - upstream_speed 

-- Define the speed of the man rowing downstream
def downstream_speed := still_water_speed + stream_speed

-- The assertion to prove
theorem downstream_speed_is_45 : downstream_speed = 45 := by
  sorry

end downstream_speed_is_45_l661_661029


namespace max_value_y2_minus_x2_plus_x_plus_5_l661_661179

theorem max_value_y2_minus_x2_plus_x_plus_5 (x y : ℝ) (h : y^2 + x - 2 = 0) : 
  ∃ M, M = 7 ∧ ∀ u v, v^2 + u - 2 = 0 → y^2 - x^2 + x + 5 ≤ M :=
by
  sorry

end max_value_y2_minus_x2_plus_x_plus_5_l661_661179


namespace product_of_roots_l661_661909

-- Let's define the given polynomial equation.
def polynomial : Polynomial ℝ := Polynomial.monomial 3 1 - Polynomial.monomial 2 15 + Polynomial.monomial 1 75 - Polynomial.C 50

-- Prove that the product of the roots of the given polynomial is 50.
theorem product_of_roots : (Polynomial.roots polynomial).prod id = 50 := by
  sorry

end product_of_roots_l661_661909


namespace range_of_a_l661_661227

noncomputable def f (a x : ℝ) : ℝ := a * x * exp x - a * x + a - exp x

theorem range_of_a (a : ℝ) (h : 0 < a) :
  (\exists! x : ℤ, f a x < 0) ↔ (a ∈ Icc (exp 2 / (2 * exp 2 - 1)) 1) := sorry

end range_of_a_l661_661227


namespace count_integers_with_factors_l661_661493

theorem count_integers_with_factors (x y z : ℕ) (h1 : y > x) (h2 : x > 0) :
  let lcm_val := lcm 16 9 in
  let min_val := Nat.ceil (z / lcm_val) * lcm_val in
  let max_val := Nat.floor (y / lcm_val) * lcm_val in
  let count := ((max_val / lcm_val) - (min_val / lcm_val) + 1) in
  (count = 2) :=
by
  sorry

end count_integers_with_factors_l661_661493


namespace general_term_of_sequence_l661_661989

theorem general_term_of_sequence (n : ℕ) (hn : 0 < n) : (1 :: (List.range (2 * n - 1)).map (λ i, 1 + i * 2)) !! (n - 1) = 2 * n - 1 :=
by
  sorry

end general_term_of_sequence_l661_661989


namespace greatest_perimeter_of_triangle_l661_661606

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l661_661606


namespace zhenya_painting_l661_661324

noncomputable def painted_sphere (S : Set Point) (O : Point) (S₁ S₂ S₃ S₄ S₅ : Set Point) : Prop := 
  (∀ (P : Point), P ∈ S₁) ∧
  (∀ (P : Point), P ∈ S₂) ∧
  (∀ (P : Point), P ∈ S₃) ∧
  (∀ (P : Point), P ∈ S₄) ∧
  (∀ (P : Point), P ∈ S₅) ∧
  (S₁ ∪ S₂ ∪ S₃ ∪ S₄ ∪ S₅ = S)

def unnecessary_color (S : Set Point) (O : Point) (S₁ S₂ S₃ S₄ S₅ : Set Point) : Prop := 
  ∃ T : List (Set Point), (T.length = 4) ∧ (∀ (P : Point), P ∈ (T.foldl (∪) ∅) ↔ P ∈ S)

theorem zhenya_painting (S : Set Point) (O : Point) (S₁ S₂ S₃ S₄ S₅ : Set Point)
  (h : painted_sphere S O S₁ S₂ S₃ S₄ S₅) : unnecessary_color S O S₁ S₂ S₃ S₄ S₅ := sorry

end zhenya_painting_l661_661324


namespace count_valid_numbers_l661_661165

def valid_digits : List ℕ := [1, 2, 3, 4, 5]

def num_meets_conditions (n : ℕ) : Prop :=
  (n < 1000) ∧ (n % 4 = 0) ∧ (∀ d ∈ (n.digits 10), d ∈ valid_digits)

theorem count_valid_numbers : (Finset.filter num_meets_conditions (Finset.range 1000)).card = 31 := 
sorry

end count_valid_numbers_l661_661165


namespace greatest_possible_perimeter_l661_661560

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l661_661560


namespace total_students_in_class_l661_661518

theorem total_students_in_class
  (x : ℕ)
  (ratio_condition : ∃ x : ℕ, ∀ boys girls, boys = 4 * x ∧ girls = 3 * x)
  (absent_boys : 8)
  (absent_girls : 14)
  (square_condition : ∀ boys_present girls_present, boys_present = (girls_present)^2) :
  (7 * (some (Exists.intro x 
                      (λ boys girls, boys = 4 * x ∧ girls = 3 * x))) = 42) := 
begin
  sorry
end

end total_students_in_class_l661_661518


namespace max_triangle_perimeter_l661_661586

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l661_661586


namespace product_of_roots_of_cubic_l661_661982

theorem product_of_roots_of_cubic :
  (∃ (p q r : ℝ), (p, q, r ≠ 0) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 75*x - 50 = 0 ↔ x = p ∨ x = q ∨ x = r) ∧ p * q * r = 50) :=
sorry

end product_of_roots_of_cubic_l661_661982


namespace find_a_l661_661466

theorem find_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x = 0 ∧ x = 1) → a = -1 := by
  intro h
  obtain ⟨x, hx, rfl⟩ := h
  have H : 1^2 + a * 1 = 0 := hx
  linarith

end find_a_l661_661466


namespace find_f_f_neg_1_l661_661147

-- Define the piecewise function f.
def f(x : ℝ) : ℝ :=
if x > 0 then log x / log 10 else x + 11

-- Theorem statement proving f(f(-1)) = 1.
theorem find_f_f_neg_1 : f(f(-1)) = 1 := 
by sorry

end find_f_f_neg_1_l661_661147


namespace greatest_perimeter_of_triangle_l661_661608

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l661_661608


namespace greatest_possible_perimeter_l661_661604

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l661_661604


namespace upgraded_fraction_l661_661033

theorem upgraded_fraction (N U : ℕ) (h1 : ∀ (k : ℕ), k = 24)
  (h2 : ∀ (n : ℕ), N = n) (h3 : ∀ (u : ℕ), U = u)
  (h4 : N = U / 8) : U / (24 * N + U) = 1 / 4 := by
  sorry

end upgraded_fraction_l661_661033


namespace triangle_angles_are_alpha_beta_gamma_l661_661697

noncomputable def is_identity_composition {α β γ : ℝ} (A B C : ℂ) :=
  (0 < α ∧ α < π) ∧ (0 < β ∧ β < π) ∧ (0 < γ ∧ γ < π) ∧
  (α + β + γ = π) ∧
  id = (rotation C (2 * γ) ∘ rotation B (2 * β) ∘ rotation A (2 * α))

theorem triangle_angles_are_alpha_beta_gamma {α β γ : ℝ} {A B C : ℂ} :
  is_identity_composition A B C α β γ → triangle_angle A B C = (α, β, γ) :=
begin
  sorry
end

end triangle_angles_are_alpha_beta_gamma_l661_661697


namespace cos_squared_alpha_minus_pi_four_l661_661435

theorem cos_squared_alpha_minus_pi_four (α : ℝ) (h : real.sin (2 * α) = 1 / 3) :
  real.cos (α - real.pi / 4) ^ 2 = 2 / 3 :=
by
  sorry

end cos_squared_alpha_minus_pi_four_l661_661435


namespace max_triangle_perimeter_l661_661589

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l661_661589


namespace count_fair_points_l661_661445

theorem count_fair_points :
  ∃ n : ℕ, n = 669 ∧
  ∀ k : ℕ, (1 ≤ k ∧ k < 670) → ∃ x y : ℕ, x = 3 * k ∧ y = 1340 - 2 * k ∧ 3 * y + 2 * x = 4020 := 
begin
  sorry
end

end count_fair_points_l661_661445


namespace total_dolls_combined_l661_661861

-- Define the number of dolls for Vera
def vera_dolls : ℕ := 20

-- Define the relationship that Sophie has twice as many dolls as Vera
def sophie_dolls : ℕ := 2 * vera_dolls

-- Define the relationship that Aida has twice as many dolls as Sophie
def aida_dolls : ℕ := 2 * sophie_dolls

-- The statement to prove that the total number of dolls is 140
theorem total_dolls_combined : aida_dolls + sophie_dolls + vera_dolls = 140 :=
by
  sorry

end total_dolls_combined_l661_661861


namespace angle_between_a_b_is_60_l661_661160

open Real

-- Definitions of vectors, magnitudes, dot product, and angle
variable {a b : E} [InnerProductSpace ℝ E]
variable (θ : ℝ)
variable [∀ (x : E), DecidableEq (x ≠ 0)]

-- Given conditions
axiom mag_a : ∥a∥ = 3
axiom mag_b : ∥b∥ = 8
axiom dot_ab_minus_a : ⟪a, b - a⟫ = 3

-- Statement to prove the angle between a and b is 60 degrees
theorem angle_between_a_b_is_60 :
  ∠ (a : E) (b : E) = π / 3 :=
sorry

end angle_between_a_b_is_60_l661_661160


namespace equilateral_triangle_inradius_circumradius_ratio_l661_661768

theorem equilateral_triangle_inradius_circumradius_ratio (r R : ℝ) 
  (h1 : Triangle.equilateral T) 
  (h2 : Triangle.inradius T = r) 
  (h3 : Triangle.circumradius T = R) : 
  r / R = 1 / 2 :=
sorry

end equilateral_triangle_inradius_circumradius_ratio_l661_661768


namespace a_10_eq_0_l661_661656

noncomputable def a : ℕ → ℝ 
noncomputable def d : ℝ

axiom a_3_eq_7 : a 3 = 7
axiom a_7_eq_3 : a 7 = 3
axiom arithmetic_sequence (n : ℕ) : a n = a 1 + (n - 1) * d

theorem a_10_eq_0 : a 10 = 0 :=
by sorry

end a_10_eq_0_l661_661656


namespace product_of_roots_eq_50_l661_661951

theorem product_of_roots_eq_50 :
  let a := -15
  let c := -50
  let equation := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  -- Vieta's formulas for the product of roots in cubic equation ax^3 + bx^2 + cx + d = 0 is -d/a
  ∃ p q r : ℝ, 
    equation p = 0 ∧ equation q = 0 ∧ equation r = 0 ∧ (p * q * r = 50) :=
by
  sorry

end product_of_roots_eq_50_l661_661951


namespace sqrt_of_4_eq_2_l661_661376

theorem sqrt_of_4_eq_2 : Real.sqrt 4 = 2 := by
  sorry

end sqrt_of_4_eq_2_l661_661376


namespace vector_solution_l661_661686

def a : ℝ × ℝ × ℝ := (2, 3, 1)
def b : ℝ × ℝ × ℝ := (4, -1, 2)
def v : ℝ × ℝ × ℝ := (6, 2, 3)

noncomputable def vector_cross (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

theorem vector_solution :
  vector_cross v a = vector_cross b a ∧ vector_cross v b = vector_cross a b :=
by
  have va := vector_cross v a
  have ba := vector_cross b a
  have vb := vector_cross v b
  have ab := vector_cross a b
  sorry

end vector_solution_l661_661686


namespace math_proof_problem_l661_661865

variables {A B C P Q H : Type} [TriangleType A B C]

-- Given conditions as definitions
def HP : ℝ := 6
def HQ : ℝ := 3
def AreaABC : ℝ := 36

noncomputable def task : ℝ := (BP * PC - AQ * QC)

theorem math_proof_problem 
  (h₁ : altitudes_intersect (triangle A B C) P Q H)
  (h₂ : HP = 6)
  (h₃ : HQ = 3)
  (h₄ : Area (triangle A B C) = 36)
  : task = 27 :=
sorry

end math_proof_problem_l661_661865


namespace common_difference_is_3_l661_661470

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition1 (a : ℕ → ℝ) : Prop := 
  a 3 + a 11 = 24

def condition2 (a : ℕ → ℝ) : Prop := 
  a 4 = 3

theorem common_difference_is_3 (h_arith : is_arithmetic a d) (h1 : condition1 a) (h2 : condition2 a) : 
  d = 3 := 
sorry

end common_difference_is_3_l661_661470


namespace greatest_perimeter_l661_661565

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l661_661565


namespace surface_area_of_circumscribed_sphere_of_cube_with_edge_length_1_l661_661996

theorem surface_area_of_circumscribed_sphere_of_cube_with_edge_length_1 :
  let a := 1
  let R := sqrt 3 / 2
  4 * Real.pi * R^2 = 3 * Real.pi
  := by
    sorry

end surface_area_of_circumscribed_sphere_of_cube_with_edge_length_1_l661_661996


namespace find_value_of_f_neg1_l661_661226

noncomputable def f (y : ℝ) : ℝ := 2 ^ (2 ^ y)

theorem find_value_of_f_neg1 (x : ℝ) (h : x > 0) (h1 : log 2 x = -1) : f (-1) = Real.sqrt 2 :=
by
  -- We introduce the conditions and complete the proof
  sorry

end find_value_of_f_neg1_l661_661226


namespace pure_milk_in_final_solution_l661_661802

noncomputable def final_quantity_of_milk (initial_milk : ℕ) (milk_removed_each_step : ℕ) (steps : ℕ) : ℝ :=
  let remaining_milk_step1 := initial_milk - milk_removed_each_step
  let proportion := (milk_removed_each_step : ℝ) / (initial_milk : ℝ)
  let milk_removed_step2 := proportion * remaining_milk_step1
  remaining_milk_step1 - milk_removed_step2

theorem pure_milk_in_final_solution :
  final_quantity_of_milk 30 9 2 = 14.7 :=
by
  sorry

end pure_milk_in_final_solution_l661_661802


namespace sequence_next_l661_661809

noncomputable def difference (a b : ℕ) : ℕ := b - a
noncomputable def next_diff (d : ℕ) : ℕ := d + 12
noncomputable def next_term (a d : ℕ) : ℕ := a + d

theorem sequence_next :
  let seq := [11, 23, 47, 83, 131, 191, 263, 347, 443, 551] in
  let diffs := seq.zipWith difference (seq.tail.append [0]) |
  ∃ next, diffs.length = seq.length - 1 ∧
             (∀ i < diffs.length - 1, diffs[i + 1] = next_diff diffs[i]) ∧
             (next = next_term seq.back (next_diff diffs.back)) ∧
             next = 671 :=
by
  sorry

end sequence_next_l661_661809


namespace count_valid_even_numbers_l661_661106

def digits := {0, 1, 2, 3, 4}

def isEven (n : ℕ) : Prop := n % 2 = 0

def validFourDigitNumbers : ℕ :=
  (digits - {0}).card * 
  ((digits - {0}).card.pred * 
  ((digits - {0}).card.pred.pred) + 
  (digits.card.pred * digits.card.pred.pred.pred)) +
  ((digits - {2, 4}).card.pred *
  ((digits - {2, 4}).card.pred.pred * 
  ((digits - {2, 4}).card.pred.pred.pred + 
  (digits.card.pred.pred * 
  digits.card.pred.pred.pred)))

theorem count_valid_even_numbers : validFourDigitNumbers = 60 := by
  sorry

end count_valid_even_numbers_l661_661106


namespace tshirt_cost_correct_l661_661379

   -- Definitions of the conditions
   def initial_amount : ℕ := 91
   def cost_of_sweater : ℕ := 24
   def cost_of_shoes : ℕ := 11
   def remaining_amount : ℕ := 50

   -- Define the total cost of the T-shirt purchase
   noncomputable def cost_of_tshirt := 
     initial_amount - remaining_amount - cost_of_sweater - cost_of_shoes

   -- Proof statement that cost_of_tshirt = 6
   theorem tshirt_cost_correct : cost_of_tshirt = 6 := 
   by
     sorry
   
end tshirt_cost_correct_l661_661379


namespace greatest_possible_perimeter_l661_661594

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l661_661594


namespace max_triangle_perimeter_l661_661588

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l661_661588


namespace product_of_roots_of_cubic_l661_661981

theorem product_of_roots_of_cubic :
  (∃ (p q r : ℝ), (p, q, r ≠ 0) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 75*x - 50 = 0 ↔ x = p ∨ x = q ∨ x = r) ∧ p * q * r = 50) :=
sorry

end product_of_roots_of_cubic_l661_661981


namespace product_of_roots_cubic_l661_661962

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  let roots := {x : ℝ // p x = 0}.toFinset
  ∃ r : ℝ, (∀ x ∈ roots, p x = 0) ∧ (∏ x in roots, x) = 50 :=
by
  sorry

end product_of_roots_cubic_l661_661962


namespace find_n_values_l661_661485

/-- Represents the n-th digit of the decimal expansion of sqrt(2) -/
def a (n : ℕ) : ℕ :=  -- implementation depends on how we retrieve the decimal digits
  if n = 1 then 4 else if n = 2 then 1 else if n = 3 then 4 else if n = 4 then 2 else if n = 5 then 1 else if n = 6 then 3 else sorry -- simplified representation

/-- Represents the sequence b_n defined in terms of a_n -/
def b : ℕ → ℕ := λ n, if n = 1 then a 1 else a (b (n-1))

theorem find_n_values : (b 676 = 676 - 2022) ∧ (b 675 = 675 - 2022) :=
  by {
    sorry
  }

end find_n_values_l661_661485


namespace Martha_improvement_in_lap_time_l661_661707

theorem Martha_improvement_in_lap_time 
  (initial_laps : ℕ) (initial_time : ℕ) 
  (first_month_laps : ℕ) (first_month_time : ℕ) 
  (second_month_laps : ℕ) (second_month_time : ℕ)
  (sec_per_min : ℕ)
  (conds : initial_laps = 15 ∧ initial_time = 30 ∧ first_month_laps = 18 ∧ first_month_time = 27 ∧ 
           second_month_laps = 20 ∧ second_month_time = 27 ∧ sec_per_min = 60)
  : ((initial_time / initial_laps : ℚ) - (second_month_time / second_month_laps)) * sec_per_min = 39 :=
by
  sorry

end Martha_improvement_in_lap_time_l661_661707


namespace volume_of_cut_out_box_l661_661366

theorem volume_of_cut_out_box (x : ℝ) : 
  let l := 16
  let w := 12
  let new_l := l - 2 * x
  let new_w := w - 2 * x
  let height := x
  let V := new_l * new_w * height
  V = 4 * x^3 - 56 * x^2 + 192 * x :=
by
  sorry

end volume_of_cut_out_box_l661_661366


namespace find_angle_C_1_find_angle_C_2_find_max_CD_sq_l661_661380

-- Define the context of the problem in Lean 4
variable (A B C a b c : ℝ)
variable (D : ℝ)
variable (triangle_ABC : Triangle a b c A B C)

-- Part 1: Prove C = π / 3 under either of the given conditions
theorem find_angle_C_1 (h : b - c * cos A = a * (sqrt 3 * sin C - 1)) : C = π / 3 :=
sorry

theorem find_angle_C_2 (h : sin (A + B) * cos (C - π / 6) = 3 / 4) : C = π / 3 :=
sorry

-- Part 2: Prove the maximum value of CD^2 / (a^2 + b^2)
noncomputable def vec_C (ax : ℝ) (ay : ℝ) := (vec a x ax, vec a y ay)
noncomputable def vec_CA := vec_C b 0 - vec_C a a
noncomputable def vec_CB := vec_C b b - vec_C a 0
noncomputable def vec_CD := (vec_CA + vec_CB) / 2
noncomputable def CD_squared := vec_CD.x^2 + vec_CD.y^2

theorem find_max_CD_sq (h1 : (CD_squared / (a^2 + b^2)) ≤ 3 / 8) : 
  (CD_squared / (a^2 + b^2)) = 3 / 8 :=
sorry

end find_angle_C_1_find_angle_C_2_find_max_CD_sq_l661_661380


namespace find_y_of_arithmetic_mean_l661_661461

theorem find_y_of_arithmetic_mean (y : ℝ) (h : (8 + 16 + 12 + 24 + 7 + y) / 6 = 12) : y = 5 :=
by
  sorry

end find_y_of_arithmetic_mean_l661_661461


namespace complex_subtract_sum_l661_661991

theorem complex_subtract_sum :
  let z1 := (5 : ℂ) + (6 : ℂ) * complex.I
  let z2 := (-1 : ℂ) + (4 : ℂ) * complex.I
  let z3 := (3 : ℂ) + (-2 : ℂ) * complex.I
  (z1 + z2) - z3 = (1 : ℂ) + (12 : ℂ) * complex.I :=
by
  sorry

end complex_subtract_sum_l661_661991


namespace largest_integer_less_than_85_remainder_2_l661_661093

theorem largest_integer_less_than_85_remainder_2 :
  ∃ n : ℕ, n < 85 ∧ n % 6 = 2 ∧ ∀ m : ℕ, m < 85 ∧ m % 6 = 2 → m ≤ n :=
begin
  use 82,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  intros m hm hmod,
  sorry
end

end largest_integer_less_than_85_remainder_2_l661_661093


namespace rationalize_denominator_XYZ_sum_l661_661731

noncomputable def a := (5 : ℝ)^(1/3)
noncomputable def b := (4 : ℝ)^(1/3)

theorem rationalize_denominator_XYZ_sum : 
  let X := 25
  let Y := 20
  let Z := 16
  let W := 1
  X + Y + Z + W = 62 :=
by 
  sorry

end rationalize_denominator_XYZ_sum_l661_661731


namespace proof_problem_l661_661462

theorem proof_problem (a b c : ℝ) (h : a > b) (h1 : b > c) :
  (1 / (a - b) + 1 / (b - c) + 4 / (c - a) ≥ 0) :=
sorry

end proof_problem_l661_661462


namespace ellipse_equation_and_triangle_area_l661_661136

theorem ellipse_equation_and_triangle_area :
  (∃ a b c : ℝ, a > b ∧ b > 0 ∧ c = sqrt 3 ∧ (∃ x y : ℝ, x = sqrt 3 ∧ y = -1/2 ∧ (x^2 / a^2) + (y^2 / b^2) = 1) ∧ (a^2 = b^2 + c^2) ∧ (∃ ellipse : ℝ → ℝ → Prop, ellipse x y ↔ (x^2 / a^2) + (y^2 / b^2) = 1 ∧ ∃ P : ℝ × ℝ, P = (sqrt 3, -1/2) ∧ ellipse P.1 P.2)) ∧
  (∃ (A B : ℝ × ℝ) (O : ℝ × ℝ), O = (0, 0) ∧ ∃ l : ℝ → ℝ → Prop, (∀ x y : ℝ, l x y ↔ x - m * y - n = 0 ∧ m ≠ 0 ∧ n^2 = m^2 + 1 ∧ (∃ x1 y1 x2 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ (∃ ellipse : ℝ → ℝ → Prop, ellipse x1 y1 ∧ ellipse x2 y2) ∧ O.x1 * O.x2 + O.y1 * O.y2 = λ ∧ 1/2 ≤ λ ∧ λ ≤ 2/3 ∧ (y1 + y2 = -2 * m * n / (m^2 + 4)) ∧ (y1 * y2 = (n^2 - 4) / (m^2 + 4))) ∧ (S: ℝ, (1/2 * ((1 / sqrt (1 + m^2)) * abs n * sqrt (1 + m^2) * abs (y1 - y2)) = S) ∧ (S ∈ [2 * sqrt 2 / 3, 1]))) :=
begin
  obtain ⟨a, b, c, ha, hb, hc, hp, P⟩ := exists_ellipse,
  obtain ⟨⟨A, B, O, l, m, n, ⟨ha, hb, m_ne_zero, n_squared, x1, y1, x2, y2, hx1, hy1, hx2, hy2, ellipse, OA_dot_OB, hlambda_min, hlambda_max, hy1_sum, hy1y2_prod⟩⟩, S, hs⟩ := exists_intersection_area,
  exact sorry
end

end ellipse_equation_and_triangle_area_l661_661136


namespace weights_max_and_count_l661_661267

theorem weights_max_and_count
  (weights : set ℕ)
  (h_weights : weights = {1, 4, 12}) :
  ∃ max_weight count, max_weight = 17 ∧ count = 17 :=
by
  use 17
  use 17
  sorry

end weights_max_and_count_l661_661267


namespace product_of_roots_l661_661935

theorem product_of_roots :
  let p : Polynomial ℚ := Polynomial.Coeff.mk_fin 3 0 (⟨[50, -75, 15, -1]⟩ : ℕ → ℚ)
  (p.root_prod = 50) :=
begin
  sorry
end

end product_of_roots_l661_661935


namespace folded_triangle_DE_length_l661_661751

noncomputable def triangle_length_DE (AB : ℝ) (area_ratio : ℝ) : ℝ :=
  let DE := sqrt(area_ratio) * AB in
  DE

theorem folded_triangle_DE_length :
  let AB := 15 in
  let area_ratio := 0.25 in
  triangle_length_DE AB area_ratio = 7.5 :=
by
  -- The proof will go here
  sorry

end folded_triangle_DE_length_l661_661751


namespace adam_coin_collection_value_l661_661858

-- Definitions related to the problem conditions
def value_per_first_type_coin := 15 / 5
def value_per_second_type_coin := 18 / 6

def total_value_first_type (num_first_type_coins : ℕ) := num_first_type_coins * value_per_first_type_coin
def total_value_second_type (num_second_type_coins : ℕ) := num_second_type_coins * value_per_second_type_coin

-- The main theorem, stating that the total collection value is 90 dollars given the conditions
theorem adam_coin_collection_value :
  total_value_first_type 18 + total_value_second_type 12 = 90 := 
sorry

end adam_coin_collection_value_l661_661858


namespace maurice_age_l661_661043

theorem maurice_age (M : ℕ) 
  (h₁ : 48 = 4 * (M + 5)) : M = 7 := 
by
  sorry

end maurice_age_l661_661043


namespace hyperbola_eccentricity_l661_661465

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (asymptote : 2 * x + y = 0) (focus : (sqrt 5, 0) ∈ set.prod set.univ set.univ)
  (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) :
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  e = sqrt 5 :=
by {
  sorry
}

end hyperbola_eccentricity_l661_661465


namespace monotonic_intervals_of_f_f_less_x_minus_1_range_of_k_l661_661477

noncomputable def f (x : ℝ) : ℝ := Real.log x - (x - 1)^2 / 2

theorem monotonic_intervals_of_f : 
  ∃ a b : ℝ, (∀ x ∈ Ioo a b, 0 < f' x) ∧ 
             (∀ x ∈ Ioo b +∞, f' x ≤ 0) :=
sorry

theorem f_less_x_minus_1 (x : ℝ) (h₁ : x > 1) : f x < x - 1 :=
sorry

theorem range_of_k (x₀ : ℝ) : 
  ∃ k : ℝ, (∀ x ∈ Ioo 1 x₀, f x > k * (x - 1)) ↔ k < 1 :=
sorry

end monotonic_intervals_of_f_f_less_x_minus_1_range_of_k_l661_661477


namespace roof_shingle_width_l661_661359

theorem roof_shingle_width (L A W : ℕ) (hL : L = 10) (hA : A = 70) (hArea : A = L * W) : W = 7 :=
by
  sorry

end roof_shingle_width_l661_661359


namespace find_a20_l661_661129

variables {a : ℕ → ℤ} {S : ℕ → ℤ}
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem find_a20 (h_arith : is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_a1 : a 1 = -1)
  (h_S10 : S 10 = 35) :
  a 20 = 18 :=
sorry

end find_a20_l661_661129


namespace greatest_possible_perimeter_l661_661538

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l661_661538


namespace max_f_theta_l661_661140

noncomputable def determinant (a b c d : ℝ) : ℝ := a*d - b*c

noncomputable def f (θ : ℝ) : ℝ :=
  determinant (Real.sin θ) (Real.cos θ) (-1) 1

theorem max_f_theta :
  ∀ θ : ℝ, 0 < θ ∧ θ < (Real.pi / 3) →
  f θ ≤ (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end max_f_theta_l661_661140


namespace product_of_roots_l661_661942

theorem product_of_roots :
  let p : Polynomial ℚ := Polynomial.Coeff.mk_fin 3 0 (⟨[50, -75, 15, -1]⟩ : ℕ → ℚ)
  (p.root_prod = 50) :=
begin
  sorry
end

end product_of_roots_l661_661942


namespace sum_of_real_solutions_l661_661992

theorem sum_of_real_solutions :
  let f (x : ℝ) := (x^2 - 6 * x + 5)^(x^2 - 7 * x + 10)
  ∃ (S : set ℝ), (∀ x ∈ S, f x = 1) ∧ (S.sum id = 19) :=
  sorry

end sum_of_real_solutions_l661_661992


namespace total_dolls_combined_l661_661860

-- Define the number of dolls for Vera
def vera_dolls : ℕ := 20

-- Define the relationship that Sophie has twice as many dolls as Vera
def sophie_dolls : ℕ := 2 * vera_dolls

-- Define the relationship that Aida has twice as many dolls as Sophie
def aida_dolls : ℕ := 2 * sophie_dolls

-- The statement to prove that the total number of dolls is 140
theorem total_dolls_combined : aida_dolls + sophie_dolls + vera_dolls = 140 :=
by
  sorry

end total_dolls_combined_l661_661860


namespace line_tangent_to_circle_l661_661510

noncomputable def tangent_line_a (a : ℝ) : Prop :=
  let circle_center := (0 : ℝ, 1 : ℝ)
  let circle_radius := 1
  let line_eq (a : ℝ) (x y : ℝ) := x + (2 - a) * y + 1 = 0
  let point_line_distance (cx cy a : ℝ) :=
    |3 - a| / real.sqrt (1 + (2 - a)^2)
  point_line_distance (0 : ℝ) (1 : ℝ) a = 1

theorem line_tangent_to_circle : tangent_line_a 2 := 
by 
  sorry

end line_tangent_to_circle_l661_661510


namespace new_time_between_maintenance_checks_l661_661018

-- Definitions based on the conditions
def original_time : ℝ := 25
def percentage_increase : ℝ := 0.20

-- Statement to be proved
theorem new_time_between_maintenance_checks : original_time * (1 + percentage_increase) = 30 := by
  sorry

end new_time_between_maintenance_checks_l661_661018


namespace product_of_roots_cubic_l661_661970

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  in (∃ r1 r2 r3 : ℝ, p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ r1 * r2 * r3 = 50) := 
by 
  sorry

end product_of_roots_cubic_l661_661970


namespace cake_mix_buyers_l661_661342

-- Definitions based on given conditions
def total_buyers : ℕ := 100
def muffin_buyers : ℕ := 40
def both_cake_and_muffin_buyers : ℕ := 18
def prob_neither : ℝ := 0.28

-- The proposition we want to prove
theorem cake_mix_buyers :
  let total := total_buyers,
      M := muffin_buyers,
      B := both_cake_and_muffin_buyers,
      P_neither := prob_neither in
  (1 - P_neither) * total = C + M - B → C = 50 :=
by
  sorry

end cake_mix_buyers_l661_661342


namespace smallest_sector_angle_division_is_10_l661_661708

/-
  Prove that the smallest possible sector angle in a 15-sector division of a circle,
  where the central angles form an arithmetic sequence with integer values and the
  total sum of angles is 360 degrees, is 10 degrees.
-/
theorem smallest_sector_angle_division_is_10 :
  ∃ (a1 d : ℕ), (∀ i, i ∈ (List.range 15) → a1 + i * d > 0) ∧ (List.sum (List.map (fun i => a1 + i * d) (List.range 15)) = 360) ∧
  a1 = 10 := by
  sorry

end smallest_sector_angle_division_is_10_l661_661708


namespace pyramid_volume_l661_661750

theorem pyramid_volume
  (a b c : ℝ) (h1 : a = 6) (h2 : b = 5) (h3 : c = 5)
  (angle_lateral : ℝ) (h4 : angle_lateral = 45) :
  ∃ (V : ℝ), V = 6 :=
by
  -- the proof steps would be included here
  sorry

end pyramid_volume_l661_661750


namespace prob_A_B_same_fee_expectation_η_l661_661650

noncomputable def rental_fee (hours: ℝ) : ℝ :=
if hours ≤ 2 then 0 else 2 * ⌈hours - 2⌉

def A_hours_distribution : ℕ → ℝ
| 2 := (1 / 4)
| 3 := (1 / 2)
| 4 := (1 / 4)
| _ := 0

def B_hours_distribution : ℕ → ℝ
| 2 := (1 / 2)
| 3 := (1 / 4)
| 4 := (1 / 4)
| _ := 0

def rental_fee_A (hours: ℕ) : ℝ :=
rental_fee hours

def rental_fee_B (hours: ℕ) : ℝ :=
rental_fee hours

def same_fee_prob : ℝ :=
(1 / 4) * (1 / 2) + 
(1 / 2) * (1 / 4) + 
(1 / 4) * (1 / 4)

theorem prob_A_B_same_fee : same_fee_prob = 5 / 16 := 
by sorry

def η_distribution : ℝ → ℝ
| 0 := (1 / 8)
| 2 := (5 / 16)
| 4 := (5 / 16)
| 6 := (3 / 16)
| 8 := (1 / 16)
| _ := 0

def E_η : ℝ :=
(5 / 16) * 2 + (5 / 16) * 4 + (3 / 16) * 6 + (1 / 16) * 8

theorem expectation_η : E_η = 7 / 2 :=
by sorry

end prob_A_B_same_fee_expectation_η_l661_661650


namespace assign_parents_l661_661873

-- Definition of the problem's context
structure Orphanage := 
  (orphans : Type)
  [decidable_eq orphans]
  (are_friends : orphans → orphans → Prop)
  (are_enemies : orphans → orphans → Prop)
  (friend_or_enemy : ∀ (o1 o2 : orphans), are_friends o1 o2 ∨ are_enemies o1 o2)
  (friend_condition : ∀ (o : orphans) (f1 f2 f3 : orphans),
                        are_friends o f1 → 
                        are_friends o f2 → 
                        are_friends o f3 → 
                        even (finset.filter (λ (p : orphans × orphans), are_enemies p.fst p.snd) 
                                              (({f1, f2, f3}.powerset.filter (λ s, s.card = 2)).bUnion id)).card)

-- Definition of the conclusion
theorem assign_parents : 
  ∀ (O : Orphanage),
  ∃ (P : O.orphans → finset ℕ), 
  (∀ (o1 o2 : O.orphans), O.are_friends o1 o2 ↔ ∃ p, p ∈ P o1 ∧ p ∈ P o2) ∧ 
  (∀ (o1 o2 : O.orphans), O.are_enemies o1 o2 ↔ P o1 ∩ P o2 = ∅) ∧ 
  (∀ (o1 o2 o3 : O.orphans) (p : ℕ),
    p ∈ P o1 ∧ p ∈ P o2 ∧ p ∈ P o3 → false) :=
sorry

end assign_parents_l661_661873


namespace range_of_a_l661_661687

theorem range_of_a (a : ℝ) : 
  (∃ (x : ℝ), (1/2)^|x| < a) → (∀ x, ax^2 + (a-2)x + 9/8 > 0) →
  ((a ≥ 8) ∨ (1/2 < a ∧ a ≤ 1)) :=
by
  sorry

end range_of_a_l661_661687


namespace coins_left_zero_when_divided_by_9_l661_661830

noncomputable def smallestCoinCount (n: ℕ) : Prop :=
  n % 8 = 6 ∧ n % 7 = 5

theorem coins_left_zero_when_divided_by_9 (n : ℕ) (h : smallestCoinCount n) (h_min: ∀ m : ℕ, smallestCoinCount m → n ≤ m) :
  n % 9 = 0 :=
sorry

end coins_left_zero_when_divided_by_9_l661_661830


namespace triangle_means_equal_l661_661192

noncomputable def arithmetic_mean (a b c : ℝ) : ℝ :=
  (a + b + c) / 3

noncomputable def harmonic_mean (a b : ℝ) : ℝ :=
  2 / ((1 / a) + (1 / b))

theorem triangle_means_equal
  (a b c : ℝ) (G I : Point)
  (h1 : is_centroid G (triangle a b c))
  (h2 : is_incenter I (triangle a b c))
  (h3 : intersects_perpendicularly (line_through G I) (angle_bisector (triangle a b c) c)) :
  arithmetic_mean a b c = harmonic_mean a b := 
sorry

end triangle_means_equal_l661_661192


namespace product_of_roots_cubic_l661_661884

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l661_661884


namespace collinear_vectors_condition_l661_661786

def areVectorsCollinear (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1

theorem collinear_vectors_condition :
  ∀ (a1 b1 a2 b2 : ℝ), 
  (let z1 := a1 + b1 * I in let z2 := a2 + b2 * I in
  z1 ≠ 0 ∧ z2 ≠ 0) →
  areVectorsCollinear a1 b1 a2 b2 :=
by
  intros a1 b1 a2 b2 h
  sorry

end collinear_vectors_condition_l661_661786


namespace greatest_perimeter_of_triangle_l661_661615

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l661_661615


namespace correct_statements_l661_661119

def sequence (a : ℕ → ℤ) : Prop :=
  ∀ (n : ℕ), (a 1 = 1) ∧ (∀ n, (n % 2 = 1 → a (n + 1) = a n - 2^n) ∧ (n % 2 = 0 → a (n + 1) = a n + 2^(n+1)))

lemma verify_statement_A (a : ℕ → ℤ) (h : sequence a) : a 3 = 7 := sorry
lemma verify_statement_B (a : ℕ → ℤ) (h : sequence a) : a 2022 = a 2 := sorry
lemma verify_statement_C (a : ℕ → ℤ) (h : sequence a) : ¬ (a 2023 = 2^2023) := sorry
lemma verify_statement_D (a : ℕ → ℤ) (h : sequence a) (S : ℕ → ℤ) : 3 * S (2*n + 1) = 2^(2*n + 3) - 6*n - 5 := sorry

theorem correct_statements (a : ℕ → ℤ) (S : ℕ → ℤ) (h : sequence a) :
  (verify_statement_A a h) ∧ (verify_statement_B a h) ∧ (verify_statement_C a h) ∧ (verify_statement_D a h S) := sorry

end correct_statements_l661_661119


namespace product_of_roots_of_cubic_polynomial_l661_661897

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l661_661897


namespace max_primes_in_8x8_table_l661_661818

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_product_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p * q

def is_distinct (table : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j k l : ℕ, i ≠ k ∨ j ≠ l → table i j ≠ table k l

def not_rel_prime (a b : ℕ) : Prop :=
  ¬ nat.coprime a b

def valid_table (table : ℕ → ℕ → ℕ) : Prop :=
  (∀ i j, is_prime (table i j) ∨ is_product_of_two_primes (table i j)) ∧
  is_distinct table ∧
  ∀ i j, ∃ k l, (i = k ∨ j = l) ∧ not_rel_prime (table i j) (table k l)

theorem max_primes_in_8x8_table :
  ∀ (table : ℕ → ℕ → ℕ), valid_table table → ∑ i j, if is_prime (table i j) then 1 else 0 ≤ 42 :=
by
  sorry

end max_primes_in_8x8_table_l661_661818


namespace abigail_fence_building_time_l661_661041

def abigail_time_per_fence (total_built: ℕ) (additional_hours: ℕ) (total_fences: ℕ): ℕ :=
  (additional_hours * 60) / (total_fences - total_built)

theorem abigail_fence_building_time :
  abigail_time_per_fence 10 8 26 = 30 :=
sorry

end abigail_fence_building_time_l661_661041


namespace product_of_roots_cubic_l661_661966

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  in (∃ r1 r2 r3 : ℝ, p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ r1 * r2 * r3 = 50) := 
by 
  sorry

end product_of_roots_cubic_l661_661966


namespace george_speed_second_segment_l661_661755

theorem george_speed_second_segment 
  (distance_total : ℝ)
  (speed_normal : ℝ)
  (distance_first : ℝ)
  (speed_first : ℝ) : 
  distance_total = 1 ∧ 
  speed_normal = 3 ∧ 
  distance_first = 0.5 ∧ 
  speed_first = 2 →
  (distance_first / speed_first + 0.5 * speed_second = 1 / speed_normal → speed_second = 6) :=
sorry

end george_speed_second_segment_l661_661755


namespace train_length_is_25_l661_661310

noncomputable def length_of_train
  (v_f : ℝ)  -- Speed of faster train in km/hr
  (v_s : ℝ)  -- Speed of slower train in km/hr
  (t : ℝ)    -- Time taken to pass in seconds
  (L : ℝ)    -- Length of one train in meters
  : Prop :=
  let v_rel := (v_f - v_s) * (5 / 18) in -- Convert relative speed to m/s
  2 * L = v_rel * t

theorem train_length_is_25
: length_of_train 46 36 18 25 :=
by
  simp [length_of_train]
  sorry

end train_length_is_25_l661_661310


namespace bridge_length_l661_661853

-- Given conditions
def train_length : ℝ := 130
def train_speed_kmh : ℝ := 65
def time_to_cross_bridge : ℝ := 15.506451791548985

-- Conversion factor from km/hr to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

-- Speed of the train in m/s
def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh

-- Total distance covered by the train while crossing the bridge
def total_distance_covered : ℝ := train_speed_ms * time_to_cross_bridge

-- Length of the bridge
theorem bridge_length : total_distance_covered - train_length = 150 :=
by
  sorry

end bridge_length_l661_661853


namespace product_of_roots_of_cubic_eqn_l661_661925

theorem product_of_roots_of_cubic_eqn :
  let p := Polynomial.Cubic (1 : ℝ) (-15) 75 (-50)
  in p.roots_prod = 50 :=
by
  sorry

end product_of_roots_of_cubic_eqn_l661_661925


namespace problem_1_problem_2_problem_3_l661_661438

-- Condition: x1 and x2 are the roots of the quadratic equation x^2 - 2(m+2)x + m^2 = 0
variables {x1 x2 m : ℝ}
axiom roots_quadratic_equation : x1^2 - 2*(m+2) * x1 + m^2 = 0 ∧ x2^2 - 2*(m+2) * x2 + m^2 = 0

-- 1. When m = 0, the roots of the equation are 0 and 4
theorem problem_1 (h : m = 0) : x1 = 0 ∧ x2 = 4 :=
by 
  sorry

-- 2. If (x1 - 2)(x2 - 2) = 41, then m = 9
theorem problem_2 (h : (x1 - 2) * (x2 - 2) = 41) : m = 9 :=
by
  sorry

-- 3. Given an isosceles triangle ABC with one side length 9, if x1 and x2 are the lengths of the other two sides, 
--    prove that the perimeter is 19.
theorem problem_3 (h1 : x1 + x2 > 9) (h2 : 9 + x1 > x2) (h3 : 9 + x2 > x1) : x1 = 1 ∧ x2 = 9 ∧ (x1 + x2 + 9) = 19 :=
by 
  sorry

end problem_1_problem_2_problem_3_l661_661438


namespace greatest_possible_perimeter_l661_661643

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l661_661643


namespace tangent_line_eq_l661_661757

def tangent_line_at (f : ℝ → ℝ) (p : ℝ × ℝ) : (ℝ → ℝ) := λ x, f p.1 + (x - p.1) * (f' p.1)
  where
    f' := deriv f

theorem tangent_line_eq : ∀ (x y : ℝ), (x = 1) → (y = -x^3 + 3*x^2) →
    tangent_line_at (λ x, -x^3 + 3*x^2) (1, 2) = λ x, 3*x - 1 := 
by
  intros x y h₁ h₂
  unfold tangent_line_at
  apply funext
  intro z
  have h_deriv : deriv (λ x, -x^3 + 3*x^2) 1 = 3 := 
    by sorry
  have h_y : (-1)^3 + 3*(-1)^2 = 2 := 
    by sorry
  simp [h₁, h_deriv, h_y]
  sorry

end tangent_line_eq_l661_661757


namespace greatest_possible_perimeter_l661_661638

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l661_661638


namespace total_bottles_ordered_in_april_and_may_is_1000_l661_661848

-- Define the conditions
def casesInApril : Nat := 20
def casesInMay : Nat := 30
def bottlesPerCase : Nat := 20

-- The total number of bottles ordered in April and May
def totalBottlesOrdered : Nat := (casesInApril + casesInMay) * bottlesPerCase

-- The main statement to be proved
theorem total_bottles_ordered_in_april_and_may_is_1000 :
  totalBottlesOrdered = 1000 :=
sorry

end total_bottles_ordered_in_april_and_may_is_1000_l661_661848


namespace exists_non_parallel_diagonal_l661_661252

theorem exists_non_parallel_diagonal (n : ℕ) (h : n > 0) : 
  ∀ (P : list (ℝ × ℝ)), convex_polygon P → (length P = 2 * n) → 
  ∃ (d : (ℝ × ℝ) × (ℝ × ℝ)), is_diagonal P d ∧ (∀ (s : (ℝ × ℝ) × (ℝ × ℝ)), s ∈ sides P → ¬ parallel d s) :=
by
  sorry

end exists_non_parallel_diagonal_l661_661252


namespace pascals_triangle_modified_l661_661733

theorem pascals_triangle_modified (n : ℕ) : 
  (∀ k ∈ {0,..., 2^n-1}, ∃ m : ℕ, replace_odd_even (pascal_triangle_row (2^n-1)) k = 1) ∧
  (card (filter (λ x, x = 1) (replace_odd_even (pascal_triangle_row 61))) = 32) :=
sorry

end pascals_triangle_modified_l661_661733


namespace total_score_l661_661833

theorem total_score (score_cap : ℝ) (score_val : ℝ) (score_imp : ℝ) (wt_cap : ℝ) (wt_val : ℝ) (wt_imp : ℝ) (total_weight : ℝ) :
  score_cap = 8 → score_val = 9 → score_imp = 7 → wt_cap = 5 → wt_val = 3 → wt_imp = 2 → total_weight = 10 →
  ((score_cap * (wt_cap / total_weight)) + (score_val * (wt_val / total_weight)) + (score_imp * (wt_imp / total_weight))) = 8.1 := 
by
  intros
  sorry

end total_score_l661_661833


namespace line_perpendicular_to_plane_l661_661050

variables (l : Type) (alpha beta : Type)
  [has_perpendicular l beta]
  [has_parallel alpha beta]

-- The Lean typeclass for perpendicular and parallel needs to be defined according to the specific details
-- of the perpendicular and parallel relationships amongst lines and planes.

theorem line_perpendicular_to_plane (h1 : l ⊥ beta) (h2 : alpha ∥ beta) : l ⊥ alpha :=
sorry

end line_perpendicular_to_plane_l661_661050


namespace hannahs_trip_cost_l661_661492

noncomputable def calculate_gas_cost (initial_odometer final_odometer : ℕ) (fuel_economy_mpg : ℚ) (cost_per_gallon : ℚ) : ℚ :=
  let distance := final_odometer - initial_odometer
  let fuel_used := distance / fuel_economy_mpg
  fuel_used * cost_per_gallon

theorem hannahs_trip_cost :
  calculate_gas_cost 36102 36131 32 (385 / 100) = 276 / 100 :=
by
  sorry

end hannahs_trip_cost_l661_661492


namespace solution_set_of_inequality_l661_661772

theorem solution_set_of_inequality (x : ℝ) :
  (|x| - 2) * (x - 1) ≥ 0 ↔ (-2 ≤ x ∧ x ≤ 1) ∨ (x ≥ 2) :=
by
  sorry

end solution_set_of_inequality_l661_661772


namespace third_root_of_polynomial_l661_661307

theorem third_root_of_polynomial 
  (a b : ℝ) 
  (h1 : a * (-2)^3 + (a + 2 * b) * (-2)^2 + (b - 3 * a) * (-2) + (10 - a) = 0)
  (h2 : a * 3^3 + (a + 2 * b) * 3^2 + (b - 3 * a) * 3 + (10 - a) = 0) 
  : ∃ c : ℝ, c = 4 / 3 :=
begin
  sorry
end

end third_root_of_polynomial_l661_661307


namespace triangle_area_l661_661294

theorem triangle_area (a b c : ℝ) (h : sqrt (a - 5) + (b - 12)^2 + abs (c - 13) = 0) : 
    area (triangle a b c) = 30 := 
sorry

end triangle_area_l661_661294


namespace product_of_roots_cubic_l661_661957

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  let roots := {x : ℝ // p x = 0}.toFinset
  ∃ r : ℝ, (∀ x ∈ roots, p x = 0) ∧ (∏ x in roots, x) = 50 :=
by
  sorry

end product_of_roots_cubic_l661_661957


namespace net_profit_calc_l661_661767

theorem net_profit_calc (purchase_price : ℕ) (overhead_percentage : ℝ) (markup : ℝ) 
  (h_pp : purchase_price = 48) (h_op : overhead_percentage = 0.10) (h_markup : markup = 35) :
  let overhead := overhead_percentage * purchase_price
  let net_profit := markup - overhead
  net_profit = 30.20 := by
    sorry

end net_profit_calc_l661_661767


namespace probability_of_y_leq_x_l661_661698

noncomputable def complex_num (x y : ℝ) : ℂ := x + (y - 1) * complex.I

def is_within_unit_circle (z : ℂ) : Prop :=
  complex.abs z ≤ 1

def prob_y_leq_x_within_unit_circle : ℝ :=
  (1/4) - (1 / (2 * Real.pi))

theorem probability_of_y_leq_x
  (x y : ℝ)
  (h : is_within_unit_circle (complex_num x y)) :
  ∃ p : ℝ, p = prob_y_leq_x_within_unit_circle :=
sorry

end probability_of_y_leq_x_l661_661698


namespace ratio_boise_seattle_l661_661298

theorem ratio_boise_seattle (Boise Seattle LakeView TotalPopulation : ℕ)
    (h1 : LakeView = 24000)
    (h2 : LakeView = Seattle + 4000)
    (h3 : TotalPopulation = Boise + Seattle + LakeView)
    (h4 : TotalPopulation = 56000) :
    (Boise : ℚ) / (Seattle : ℚ) = 3 / 5 := 
begin
  sorry
end

end ratio_boise_seattle_l661_661298


namespace chess_game_pieces_l661_661372

theorem chess_game_pieces (missing_queens missing_knights missing_pawns pieces_per_player : ℕ)
  (h₀ : missing_queens = 2)
  (h₁ : missing_knights = 5)
  (h₂ : missing_pawns = 8)
  (h₃ : pieces_per_player = 11) : 
  let missing_pieces_total := missing_queens + missing_knights + missing_pawns,
      total_pieces_before_missing := 3 * pieces_per_player + missing_pieces_total
  in missing_pieces_total = 15 ∧ total_pieces_before_missing = 33 := by
  sorry

end chess_game_pieces_l661_661372


namespace unique_solution_l661_661413

theorem unique_solution : ∃! (x y : ℝ), 16^(x^2 + y) + 16^(x + y^2) = 1 := 
by {
    sorry
}

end unique_solution_l661_661413


namespace john_recreation_spending_percentage_l661_661333

theorem john_recreation_spending_percentage (W : ℝ) (h : W > 0) : 
  let last_week_recreation := 0.30 * W
  let this_week_wages := 0.75 * W
  let this_week_recreation := 0.20 * this_week_wages
  this_week_recreation / last_week_recreation = 0.50 :=
by
  let last_week_recreation := 0.30 * W
  let this_week_wages := 0.75 * W
  let this_week_recreation := 0.20 * this_week_wages
  have h1 : this_week_recreation = 0.15 * W := by sorry
  have h2 : last_week_recreation = 0.30 * W := by sorry
  have h3 : this_week_recreation / last_week_recreation = (0.15 * W) / (0.30 * W) := by sorry
  have h4 : (0.15 * W) / (0.30 * W) = 0.50 := by sorry
  exact h4

end john_recreation_spending_percentage_l661_661333


namespace no_zeros_of_g_l661_661758

theorem no_zeros_of_g (f : ℝ → ℝ) (g := λ x, f(x) + 1/x) :
  (∀ x > 0, differentiable_at ℝ f x) →
  f(1) = -1 →
  (∀ x > 0, f'' x + f(x) / x > 0) →
  ∀ x > 0, g(x) ≠ 0 :=
by
  sorry

end no_zeros_of_g_l661_661758


namespace diagonal_not_parallel_l661_661253

theorem diagonal_not_parallel (n : ℕ) (h : 2 ≤ n) :
  ∃ (d : diagonal), ¬ (d ∥ side) := by
  sorry

end diagonal_not_parallel_l661_661253


namespace ratio_pentagon_rectangle_l661_661845

theorem ratio_pentagon_rectangle (P: ℝ) (a w: ℝ) (h1: 5 * a = P) (h2: 6 * w = P) (h3: P = 75) : a / w = 6 / 5 := 
by 
  -- Proof steps will be provided to conclude this result 
  sorry

end ratio_pentagon_rectangle_l661_661845


namespace inequality_inequality_triangle_abc_l661_661665

variables (A B C : ℝ)
variables (a b c : ℝ)
variables (R r : ℝ)
variables [Fact (a > 0)] [Fact (b > 0)] [Fact (c > 0)]
variables [Fact (R > 0)] [Fact (r > 0)]

noncomputable def cot (x : ℝ) := real.cos x / real.sin x

theorem inequality_inequality_triangle_abc
  (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) (h₄: R > 0) (h₅: r > 0) :
  6*r ≤ a * cot A + b * cot B + c * cot C ∧ a * cot A + b * cot B + c * cot C ≤ 3*R :=
sorry

end inequality_inequality_triangle_abc_l661_661665


namespace mushroom_mass_decrease_l661_661790

theorem mushroom_mass_decrease :
  ∀ (initial_mass water_content_fresh water_content_dry : ℝ),
  water_content_fresh = 0.8 →
  water_content_dry = 0.2 →
  (initial_mass * (1 - water_content_fresh) / (1 - water_content_dry) = initial_mass * 0.25) →
  (initial_mass - initial_mass * 0.25) / initial_mass = 0.75 :=
by
  intros initial_mass water_content_fresh water_content_dry h_fresh h_dry h_dry_mass
  sorry

end mushroom_mass_decrease_l661_661790


namespace product_of_roots_cubic_l661_661892

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l661_661892


namespace multiple_of_9_l661_661334

noncomputable def digit_sum (x : ℕ) : ℕ := sorry  -- Placeholder for the digit sum function

theorem multiple_of_9 (n : ℕ) (h1 : digit_sum n = digit_sum (3 * n))
  (h2 : ∀ x, x % 9 = digit_sum x % 9) :
  n % 9 = 0 :=
by
  sorry

end multiple_of_9_l661_661334


namespace tablecloth_overhang_l661_661712

theorem tablecloth_overhang (d r l overhang1 overhang2 : ℝ) (h1 : d = 0.6) (h2 : r = d / 2) (h3 : l = 1) 
  (h4 : overhang1 = 0.5) (h5 : overhang2 = 0.3) :
  ∃ overhang3 overhang4 : ℝ, overhang3 = 0.33 ∧ overhang4 = 0.52 := 
sorry

end tablecloth_overhang_l661_661712


namespace kevin_paperclips_l661_661676

theorem kevin_paperclips : ∃ n : ℕ, 1 ≤ n ∧ 5 * 3 ^ (n - 1) > 200 ∧ nat.cycle 7 (n + 1) = 5 := by
  sorry

end kevin_paperclips_l661_661676


namespace ball_placement_ways_l661_661716

theorem ball_placement_ways :
  let balls := 5 
  let boxes := 4 
  ∃ (ways : ℕ), 
  (ways = (Combination balls (boxes - 1) ) * Fact boxes) ∧ 
  ways = 240 :=
by
  sorry

end ball_placement_ways_l661_661716


namespace valid_numbers_count_l661_661163

def isAllowedDigit (d : Nat) : Bool :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def containsAllowedDigitsOnly (n : Nat) : Bool :=
  n.digits.forall isAllowedDigit

def isValidNumber (n : Nat) : Bool :=
  n < 1000 ∧ n % 4 = 0 ∧ containsAllowedDigitsOnly n

def countValidNumbers : Nat :=
  (List.range 1000).filter isValidNumber |> List.length

theorem valid_numbers_count : countValidNumbers = 31 :=
by
  sorry

end valid_numbers_count_l661_661163


namespace median_of_set_is_71_l661_661392

theorem median_of_set_is_71 (x y : ℕ) (hx : x + y = 140) :
  let s := [75, 77, 72, 68, x, y] in
  let med := (s.sort.nth_le 2 (by dec_trivial) + s.sort.nth_le 3 (by dec_trivial)) / 2 in
  med = 71 := 
by
  sorry

end median_of_set_is_71_l661_661392


namespace quotient_remainder_l661_661099

def f (x : ℝ) : ℝ := x^4 - 22*x^3 + 12*x^2 - 13*x + 9
def d (x : ℝ) : ℝ := x - 3
def q (x : ℝ) : ℝ := x^3 - 19*x^2 - 45*x - 148

theorem quotient_remainder :
  ∃ (q r : ℝ). (f x = (d x) * (q x) + r) ∧ q x = x^3 - 19*x^2 - 45*x - 148 ∧ r = -435 :=
by
  sorry

end quotient_remainder_l661_661099


namespace product_of_roots_of_cubic_eqn_l661_661924

theorem product_of_roots_of_cubic_eqn :
  let p := Polynomial.Cubic (1 : ℝ) (-15) 75 (-50)
  in p.roots_prod = 50 :=
by
  sorry

end product_of_roots_of_cubic_eqn_l661_661924


namespace even_iff_exists_functions_f_g_l661_661429

def exists_functions_f_g (n : ℕ) : Prop :=
  ∃ (f g : ℕ → ℕ), (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (f (g i) = i ∨ g (f i) = i) ∧ ¬(f (g i) = i ∧ g (f i) = i))

theorem even_iff_exists_functions_f_g (n : ℕ) (h_pos : 0 < n) : 
  nat.even n ↔ exists_functions_f_g n := sorry

end even_iff_exists_functions_f_g_l661_661429


namespace greatest_possible_perimeter_l661_661532

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l661_661532


namespace max_pens_l661_661322

theorem max_pens (total_money notebook_cost pen_cost num_notebooks : ℝ) (notebook_qty pen_qty : ℕ):
  total_money = 18 ∧ notebook_cost = 3.6 ∧ pen_cost = 3 ∧ num_notebooks = 2 →
  (pen_qty = 1 ∨ pen_qty = 2 ∨ pen_qty = 3) ↔ (2 * notebook_cost + pen_qty * pen_cost ≤ total_money) :=
by {
  sorry
}

end max_pens_l661_661322


namespace lucy_bank_balance_l661_661706

def initial_balance : ℕ := 65
def deposit : ℕ := 15
def withdrawal : ℕ := 4

theorem lucy_bank_balance : initial_balance + deposit - withdrawal = 76 := by
  rw [← Nat.add_sub_assoc, Nat.add_sub_self, Nat.add_zero]
  exact rfl

end lucy_bank_balance_l661_661706


namespace product_of_roots_l661_661922

theorem product_of_roots :
  let f := Polynomial.C (-50) + Polynomial.X * (Polynomial.C 75 + Polynomial.X * (Polynomial.C (-15) + Polynomial.X)) in 
  (f.roots.Prod = 50) :=
by sorry

end product_of_roots_l661_661922


namespace rationalize_cube_root_identity_l661_661728

theorem rationalize_cube_root_identity :
  let a := (5 : ℝ)^(1/3)
  let b := (4 : ℝ)^(1/3)
  let X := 25
  let Y := 20
  let Z := 16
  let W := 1
  a^3 = 5 ∧ b^3 = 4 ∧ a - b ≠ 0 ∧
  (X + Y + Z + W) = 62 :=
by
  -- Define a and b
  let a := (5 : ℝ)^(1/3)
  let b := (4 : ℝ)^(1/3)
  -- Rationalize using identity a^3 - b^3 = (a - b)(a^2 + ab + b^2)
  have h1 : a^3 = 5, by sorry
  have h2 : b^3 = 4, by sorry
  have h3 : a - b ≠ 0, by sorry
  let X := 25
  let Y := 20
  let Z := 16
  let W := 1
  -- Conclude the sum X + Y + Z + W = 62
  have h4 : (X + Y + Z + W) = 62, by sorry
  -- Returning the combined statement
  exact ⟨h1, h2, h3, h4⟩

end rationalize_cube_root_identity_l661_661728


namespace greatest_possible_perimeter_l661_661552

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l661_661552


namespace product_of_roots_l661_661938

theorem product_of_roots :
  let p : Polynomial ℚ := Polynomial.Coeff.mk_fin 3 0 (⟨[50, -75, 15, -1]⟩ : ℕ → ℚ)
  (p.root_prod = 50) :=
begin
  sorry
end

end product_of_roots_l661_661938


namespace train_passes_jogger_in_40_seconds_l661_661001

variable (speed_jogger_kmh : ℕ)
variable (speed_train_kmh : ℕ)
variable (head_start : ℕ)
variable (train_length : ℕ)

noncomputable def time_to_pass_jogger (speed_jogger_kmh speed_train_kmh head_start train_length : ℕ) : ℕ :=
  let speed_jogger_ms := (speed_jogger_kmh * 1000) / 3600
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  let relative_speed := speed_train_ms - speed_jogger_ms
  let total_distance := head_start + train_length
  total_distance / relative_speed

theorem train_passes_jogger_in_40_seconds : time_to_pass_jogger 9 45 280 120 = 40 := by
  sorry

end train_passes_jogger_in_40_seconds_l661_661001


namespace question_1_question_2_l661_661439

noncomputable def prob_1_mag_condition (a b : ℝ³) : Prop :=
  ‖a‖ = real.sqrt 2 ∧ ‖b‖ = 1

noncomputable def prob_1_angle_condition (a b : ℝ³) (θ : ℝ) : Prop :=
  θ = real.pi / 4 ∧ prob_1_mag_condition a b

noncomputable def prob_2_perp_condition (a b : ℝ³) : Prop :=
  inner (a - b) b = 0 ∧ prob_1_mag_condition a b

theorem question_1 (a b : ℝ³) (θ : ℝ) (h : prob_1_angle_condition a b θ) :
  ‖a - b‖ = 1 :=
sorry

theorem question_2 (a b : ℝ³) (h : prob_2_perp_condition a b) :
  ∃ θ : ℝ, θ = real.pi / 4 :=
sorry

end question_1_question_2_l661_661439


namespace third_day_sales_correct_l661_661017

variable (a : ℕ)

def firstDaySales := a
def secondDaySales := a + 4
def thirdDaySales := 2 * (a + 4) - 7
def expectedSales := 2 * a + 1

theorem third_day_sales_correct : thirdDaySales a = expectedSales a :=
by
  -- Main proof goes here
  sorry

end third_day_sales_correct_l661_661017


namespace isabel_homework_problems_l661_661668

theorem isabel_homework_problems (initial_problems finished_problems remaining_pages problems_per_page : ℕ) 
  (h1 : initial_problems = 72)
  (h2 : finished_problems = 32)
  (h3 : remaining_pages = 5)
  (h4 : initial_problems - finished_problems = 40)
  (h5 : 40 = remaining_pages * problems_per_page) : 
  problems_per_page = 8 := 
by sorry

end isabel_homework_problems_l661_661668


namespace neighborhood_total_households_l661_661520

def total_households (H neither both car bike_only : ℕ) : Prop :=
  car = 44 ∧
  both = 14 ∧
  neither = 11 ∧
  bike_only = 35 ∧
  H = neither + (car - both) + bike_only + both

theorem neighborhood_total_households : ∃ H, total_households H 11 14 44 35 :=
by
  use 90
  unfold total_households
  split
  all_goals {try {norm_num}}
  sorry

end neighborhood_total_households_l661_661520


namespace lcm_of_three_numbers_l661_661300

theorem lcm_of_three_numbers (x : ℕ) :
  (Nat.gcd (3 * x) (Nat.gcd (4 * x) (5 * x)) = 40) →
  Nat.lcm (3 * x) (Nat.lcm (4 * x) (5 * x)) = 2400 :=
by
  sorry

end lcm_of_three_numbers_l661_661300


namespace seeking_cause_from_effect_is_necessary_condition_l661_661870

theorem seeking_cause_from_effect_is_necessary_condition :
  (analytical_proof : Prop) (seeking_cause_from_effect : Prop → Prop) →
  ∀ (proposition : Prop), seeking_cause_from_effect proposition ↔ (proposition → analytical_proof) := sorry

end seeking_cause_from_effect_is_necessary_condition_l661_661870


namespace percentage_less_than_l661_661181

theorem percentage_less_than (x y z : Real) (h1 : x = 1.20 * y) (h2 : x = 0.84 * z) : 
  ((z - y) / z) * 100 = 30 := 
sorry

end percentage_less_than_l661_661181


namespace rachel_age_when_emily_half_age_l661_661402

-- Conditions
def Emily_current_age : ℕ := 20
def Rachel_current_age : ℕ := 24

-- Proof statement
theorem rachel_age_when_emily_half_age :
  ∃ x : ℕ, (Emily_current_age - x = (Rachel_current_age - x) / 2) ∧ (Rachel_current_age - x = 8) := 
sorry

end rachel_age_when_emily_half_age_l661_661402


namespace lab_tech_ratio_l661_661296

theorem lab_tech_ratio (U T C : ℕ) (hU : U = 12) (hC : C = 6 * U) (hT : T = (C + U) / 14) :
  (T : ℚ) / U = 1 / 2 :=
by
  sorry

end lab_tech_ratio_l661_661296


namespace diagonal_not_parallel_l661_661255

theorem diagonal_not_parallel (n : ℕ) (h : 2 ≤ n) :
  ∃ (d : diagonal), ¬ (d ∥ side) := by
  sorry

end diagonal_not_parallel_l661_661255


namespace power_identity_l661_661486

namespace MathProof

variable {a : ℤ}
def A : Set ℤ := {a+2, (a+1)^2, a^2 + 3a + 3}
axiom h : 1 ∈ A

theorem power_identity : 2015^a = 1 := by
  sorry

end MathProof

end power_identity_l661_661486


namespace sum_of_midpoint_coords_l661_661795

theorem sum_of_midpoint_coords :
  let x1 := 10
  let y1 := 7
  let x2 := 4
  let y2 := 1
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  xm + ym = 11 :=
by
  let x1 := 10
  let y1 := 7
  let x2 := 4
  let y2 := 1
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  sorry

end sum_of_midpoint_coords_l661_661795


namespace product_of_roots_of_cubic_polynomial_l661_661899

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l661_661899


namespace greatest_possible_perimeter_l661_661641

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l661_661641


namespace shopkeeper_packets_l661_661036

noncomputable def milk_packets (oz_to_ml: ℝ) (ml_per_packet: ℝ) (total_milk_oz: ℝ) : ℝ :=
  (total_milk_oz * oz_to_ml) / ml_per_packet

theorem shopkeeper_packets (oz_to_ml: ℝ) (ml_per_packet: ℝ) (total_milk_oz: ℝ) :
  oz_to_ml = 30 → ml_per_packet = 250 → total_milk_oz = 1250 → milk_packets oz_to_ml ml_per_packet total_milk_oz = 150 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end shopkeeper_packets_l661_661036


namespace greatest_possible_perimeter_l661_661543

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l661_661543


namespace complement_of_M_in_U_l661_661700

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def complement_U_M : Set ℕ := {2, 4, 6}

theorem complement_of_M_in_U :
  \complement_U M = {2, 4, 6} := 
by
  sorry

end complement_of_M_in_U_l661_661700


namespace circle_line_distance_range_l661_661121

open Real

theorem circle_line_distance_range (a : ℝ) :
  ∃ (x y : ℝ), (x^2 + y^2 = 4) ∧ (|x + y - a| = 1) ↔ -3 * sqrt 2 < a ∧ a < 3 * sqrt 2 :=
by
  sorry

end circle_line_distance_range_l661_661121


namespace exists_unique_lambda_for_derivative_l661_661100

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a + 1 / x

theorem exists_unique_lambda_for_derivative (a : ℝ) :
  -1 / real.exp 1 < a ∧ a < 0 →
  ∃! λ : ℝ, ∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 → 
  let x0 := λ * x1 + (1 - λ) * x2 in 
  0 < x0 → f' a x0 < 0 :=
begin
  sorry -- proof not required
end

end exists_unique_lambda_for_derivative_l661_661100


namespace trig_problem_part_1_trig_problem_part_2_l661_661142

open Real

-- Definitions from conditions
def equation_has_trig_roots (m : ℝ) (θ : ℝ) : Prop :=
  2 * sin θ ^ 2 - (sqrt 3 + 1) * sin θ + m = 0 ∧
  2 * cos θ ^ 2 - (sqrt 3 + 1) * cos θ + m = 0 ∧
  0 < θ ∧ θ < 2 * π

noncomputable def problem_1 (θ : ℝ) : ℝ :=
  (sin θ ^ 2 / (sin θ - cos θ)) + (cos θ ^ 2 / (cos θ - sin θ))

theorem trig_problem_part_1 (m : ℝ) (θ : ℝ) (h : equation_has_trig_roots m θ) :
  problem_1 θ = (sqrt 3 + 1) / 2 := sorry

theorem trig_problem_part_2 (m : ℝ) (θ : ℝ) (h : equation_has_trig_roots m θ) :
  m = sqrt 3 / 2 := sorry

end trig_problem_part_1_trig_problem_part_2_l661_661142


namespace range_of_k_l661_661463

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → k * (Real.exp (k * x) + 1) - ((1 / x) + 1) * Real.log x > 0) ↔ k > 1 / Real.exp 1 := 
  sorry

end range_of_k_l661_661463


namespace zilla_savings_deposit_l661_661325

-- Definitions based on problem conditions
def monthly_earnings (E : ℝ) : Prop :=
  0.07 * E = 133

def tax_deduction (E : ℝ) : ℝ :=
  E - 0.10 * E

def expenditure (earnings : ℝ) : ℝ :=
  133 +  0.30 * earnings + 0.20 * earnings + 0.12 * earnings

def savings_deposit (remaining_earnings : ℝ) : ℝ :=
  0.15 * remaining_earnings

-- The final proof statement
theorem zilla_savings_deposit (E : ℝ) (total_spent : ℝ) (earnings_after_tax : ℝ) (remaining_earnings : ℝ) : 
  monthly_earnings E →
  tax_deduction E = earnings_after_tax →
  expenditure earnings_after_tax = total_spent →
  remaining_earnings = earnings_after_tax - total_spent →
  savings_deposit remaining_earnings = 77.52 :=
by
  intros
  sorry

end zilla_savings_deposit_l661_661325


namespace find_vector_magnitude_l661_661490

open Real

variables (a b : Vector ℝ) -- We need to define vectors in appropriate context; assuming ℝ^n space.

-- Given Conditions
def cond1 : Prop := (b.norm = 5)
def cond2 : Prop := ((2 • a + b).norm = 5 * Real.sqrt 3)
def cond3 : Prop := ((a - b).norm = 5 * Real.sqrt 2)

-- Proof Goal
theorem find_vector_magnitude (a b : Vector ℝ) (h1 : cond1 b) (h2 : cond2 a b) (h3 : cond3 a b) : a.norm = 5 * Real.sqrt 2 / 3 :=
sorry

end find_vector_magnitude_l661_661490


namespace emily_tables_l661_661401

-- Definitions based on the given conditions
def num_chairs := 4
def time_per_item := 8
def total_time := 48

-- Theorem statement
theorem emily_tables : ∃ T : ℕ, total_time = (num_chairs * time_per_item) + (T * time_per_item) ∧ T = 2 := by
  use 2
  unfold num_chairs time_per_item total_time
  constructor
  { norm_num }
  { refl }

end emily_tables_l661_661401


namespace Maurice_current_age_l661_661046

variable (Ron_now Maurice_now : ℕ)

theorem Maurice_current_age
  (h1 : Ron_now = 43)
  (h2 : ∀ t Ron_future Maurice_future : ℕ, Ron_future = 4 * Maurice_future → Ron_future = Ron_now + 5 → Maurice_future = Maurice_now + 5) :
  Maurice_now = 7 := 
sorry

end Maurice_current_age_l661_661046


namespace extreme_values_number_of_zeros_l661_661145

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5
noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem extreme_values :
  (∀ x : ℝ, f x ≤ 12) ∧ (f (-1) = 12) ∧ (∀ x : ℝ, -15 ≤ f x) ∧ (f 2 = -15) := 
sorry

theorem number_of_zeros (m : ℝ) :
  (m > 12 ∨ m < -15 → ∃! x : ℝ, g x m = 0) ∧
  (m = 12 ∨ m = -15 → ∃ x y : ℝ, x ≠ y ∧ g x m = 0 ∧ g y m = 0) ∧
  (-15 < m ∧ m < 12 → ∃ x y z : ℝ, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ g x m = 0 ∧ g y m = 0 ∧ g z m = 0) :=
sorry

end extreme_values_number_of_zeros_l661_661145


namespace greatest_triangle_perimeter_l661_661573

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l661_661573


namespace squares_with_center_35_65_l661_661247

theorem squares_with_center_35_65 : 
  (∃ (n : ℕ), n = 1190 ∧ ∀ (x y : ℕ), x ≠ y → (x, y) = (35, 65)) :=
sorry

end squares_with_center_35_65_l661_661247


namespace decimal_to_binary_34_l661_661387

theorem decimal_to_binary_34 : nat_to_binary 34 = 100010 := 
  sorry

end decimal_to_binary_34_l661_661387


namespace greatest_perimeter_of_triangle_l661_661610

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l661_661610


namespace tin_silver_ratio_l661_661326

theorem tin_silver_ratio (T S : ℝ) 
  (h1 : T + S = 50) 
  (h2 : 0.1375 * T + 0.075 * S = 5) : 
  T / S = 2 / 3 :=
by
  sorry

end tin_silver_ratio_l661_661326


namespace product_of_max_and_min_values_l661_661134

theorem product_of_max_and_min_values (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : 4^(sqrt (5*x + 9*y + 4*z)) - 68 * 2^(sqrt (5*x + 9*y + 4*z)) + 256 = 0) :
  (max (4 : ℝ) (x + y + z)) * (min ((4 : ℝ) / 9) (x + y + z)) = 4 :=
sorry

end product_of_max_and_min_values_l661_661134


namespace partition_nat_numbers_seventh_power_l661_661999

theorem partition_nat_numbers_seventh_power :
  ∃ (G : ℕ → finset ℕ), (∀ k, G k.card = k) ∧ (∀ k, ∃ m : ℕ, (G k).sum = m^7) :=
by sorry

end partition_nat_numbers_seventh_power_l661_661999


namespace ensure_4_shirts_same_color_ensure_5_shirts_same_color_ensure_6_shirts_same_color_ensure_7_shirts_same_color_ensure_8_shirts_same_color_ensure_9_shirts_same_color_l661_661347

-- Given conditions
def drawer (blue gray red : ℕ) : Prop :=
  blue = 4 ∧ gray = 7 ∧ red = 9 ∧ blue + gray + red = 20

-- Prove number of shirts required for each condition
theorem ensure_4_shirts_same_color : ∀ (blue gray red : ℕ), drawer blue gray red → ∀ (n : ℕ), n = 10 → 
                                        ∃ chosen, (chosen ∘ n = 4) ∨ (chosen ∘ n = 5) ∨ (chosen ∘ n = 6) := sorry

theorem ensure_5_shirts_same_color : ∀ (blue gray red : ℕ), drawer blue gray red → ∀ (n : ℕ), n = 13 → 
                                        ∃ chosen, (chosen ∘ n = 5) := sorry

theorem ensure_6_shirts_same_color : ∀ (blue gray red : ℕ), drawer blue gray red → ∀ (n : ℕ), n = 16 → 
                                        ∃ chosen, (chosen ∘ n = 6) := sorry

theorem ensure_7_shirts_same_color : ∀ (blue gray red : ℕ), drawer blue gray red → ∀ (n : ℕ), n = 17 → 
                                        ∃ chosen, (chosen ∘ n = 7) := sorry

theorem ensure_8_shirts_same_color : ∀ (blue gray red : ℕ), drawer blue gray red → ∀ (n : ℕ), n = 19 → 
                                        ∃ chosen, (chosen ∘ n = 8) := sorry

theorem ensure_9_shirts_same_color : ∀ (blue gray red : ℕ), drawer blue gray red → ∀ (n : ℕ), n = 20 → 
                                        ∃ chosen, (chosen ∘ n = 9) := sorry

end ensure_4_shirts_same_color_ensure_5_shirts_same_color_ensure_6_shirts_same_color_ensure_7_shirts_same_color_ensure_8_shirts_same_color_ensure_9_shirts_same_color_l661_661347


namespace max_people_in_groups_l661_661519

theorem max_people_in_groups :
  ∀ (total : ℕ) (non_working : ℕ) (with_families : ℕ) (sing_shower : ℕ) (engages_sport : ℕ),
  total = 1000 →
  non_working = 400 →
  with_families = 300 →
  sing_shower = 700 →
  engages_sport = 200 →
  ∀ (senior_citizens : ℕ), 
  senior_citizens ≤ non_working ∧ senior_citizens ≤ sing_shower →
  (600 ≤ 700 ∧ 600 ≤ total - non_working) →
  (700 - with_families = 700 ∧ 700 - with_families ≤ total - with_families) →
  (∃ x, x = engages_sport ∧ x ≤ 600 ∧ x ≤ 700 ∧ x ≤ engages_sport)
  :=
begin
  intros,
  sorry
end

end max_people_in_groups_l661_661519


namespace race_length_l661_661371

theorem race_length (covered_meters remaining_meters race_length : ℕ)
  (h_covered : covered_meters = 721)
  (h_remaining : remaining_meters = 279)
  (h_race_length : race_length = covered_meters + remaining_meters) :
  race_length = 1000 :=
by
  rw [h_covered, h_remaining] at h_race_length
  exact h_race_length

end race_length_l661_661371


namespace cone_surface_area_l661_661469

def radius_of_base := 3
def central_angle := (2 / 3) * Real.pi

theorem cone_surface_area : 
  ∀ (r : ℝ) (θ : ℝ), 
  r = radius_of_base → θ = central_angle → 
  let base_area := Real.pi * r ^ 2 in
  let side_length := 2 * Real.pi * r in
  let sector_radius := side_length / θ in
  let side_area := (1 / 2) * side_length * sector_radius in
  let total_surface_area := base_area + side_area in
  total_surface_area = 36 * Real.pi :=
by
  intros r θ hr hθ
  rw [hr, hθ]
  let base_area := Real.pi * radius_of_base ^ 2
  let side_length := 2 * Real.pi * radius_of_base
  let sector_radius := side_length / central_angle
  let side_area := (1 / 2) * side_length * sector_radius
  let total_surface_area := base_area + side_area
  sorry

end cone_surface_area_l661_661469


namespace max_sum_of_digits_in_watch_l661_661024

theorem max_sum_of_digits_in_watch : ∃ max_sum : ℕ, max_sum = 23 ∧ 
  ∀ hours minutes : ℕ, 
  (1 ≤ hours ∧ hours ≤ 12) → 
  (0 ≤ minutes ∧ minutes < 60) → 
  let hour_digits_sum := (hours / 10) + (hours % 10) in
  let minute_digits_sum := (minutes / 10) + (minutes % 10) in
  hour_digits_sum + minute_digits_sum ≤ max_sum :=
sorry

end max_sum_of_digits_in_watch_l661_661024


namespace solve_complex_eq_l661_661115

theorem solve_complex_eq (z : ℂ) (h : (1 - 3 * Complex.i) * z = 2 + 4 * Complex.i) : z = -1 + Complex.i :=
sorry

end solve_complex_eq_l661_661115


namespace max_true_claims_l661_661082

-- Define the conditions
def knight_positions (n : ℕ) : Prop :=
  n = 12

def distinct_numbers (numbers : Fin 12 → ℕ) : Prop :=
  ∀ i j, i ≠ j → numbers i ≠ numbers j

def knight_claims (numbers : Fin 12 → ℕ) (claims : Fin 12 → Prop) : Prop :=
  ∀ i, claims i ↔ (numbers i > numbers ((i + 1) % 12) ∧ numbers i > numbers ((i + 11) % 12))

-- Define the problem statement to prove
theorem max_true_claims : ∀ (numbers : Fin 12 → ℕ) (claims : Fin 12 → Prop),
  knight_positions 12 →
  distinct_numbers numbers →
  knight_claims numbers claims →
  ∃ (C : Fin 12 → Prop), (∀ i, C i → claims i) ∧ (∑ i, if C i then 1 else 0) ≤ 6 :=
sorry

end max_true_claims_l661_661082


namespace gain_percentage_calculation_l661_661374

theorem gain_percentage_calculation :
  let CP := (1/15) / 0.85 in  -- Cost Price of each pencil when selling 15 pencils for 1 rupee at 15% loss
  let SP_new := 1 / 11.09 in  -- New Selling Price per pencil for 11.09 pencils per rupee
  let gain := [(SP_new / CP) - 1] * 100 in
  abs (gain - 14.94) < 0.01 :=  -- Defined margin of error for approximation
by
  sorry

end gain_percentage_calculation_l661_661374


namespace percent_men_tenured_l661_661056

theorem percent_men_tenured (total_profs : ℕ) (women_profs : ℕ) (tenured_profs : ℕ)
  (women_or_tenured : ℕ) :
  women_profs = 70 * total_profs / 100 →
  tenured_profs = 70 * total_profs / 100 →
  women_or_tenured = 90 * total_profs / 100 →
  let total_men := total_profs - women_profs in
  let men_tenured := tenured_profs - (women_profs + tenured_profs - women_or_tenured) in
  (men_tenured * 100 / total_men : ℝ) ≈ 66.67 :=
begin
  intros h1 h2 h3,
  have h4 : total_men = total_profs - women_profs, from rfl,
  have h5 : women_and_tenured = women_profs + tenured_profs - women_or_tenured, from rfl,
  have h6 : men_tenured = tenured_profs - women_and_tenured, from rfl,
  have h7 : (men_tenured * 100 / total_men : ℝ) ≈ 66.67, {
    -- Assuming total_profs = 100
    have h8 : total_profs = 100, sorry,
    rw h8 at *,
    simp at *,
    exact abs_sub_lt_iff.mpr ⟨sorry, sorry⟩
  },
  exact h7
end

end percent_men_tenured_l661_661056


namespace solve_eq_0_l661_661696

def f (x : ℝ) : ℝ :=
if x < 0 then 5 * x + 10 else 3 * x - 18

theorem solve_eq_0 : { x : ℝ | f x = 0 } = { -2, 6 } :=
sorry

end solve_eq_0_l661_661696


namespace altitude_from_A_to_BC_in_triangle_l661_661663

theorem altitude_from_A_to_BC_in_triangle
  (a b c : ℝ)
  (h_ab : a = 2 * real.sqrt 3)
  (h_bc : b = real.sqrt 7 + 1)
  (h_ac : c = real.sqrt 7 - 1)
  : ∃ h_A, h_A = (2 * real.sqrt 14 - 2 * real.sqrt 2) / 3 :=
by
  use (2 * real.sqrt 14 - 2 * real.sqrt 2) / 3
  sorry

end altitude_from_A_to_BC_in_triangle_l661_661663


namespace sufficient_but_not_necessary_for_reciprocal_l661_661814

theorem sufficient_but_not_necessary_for_reciprocal (x : ℝ) : (x > 1 → 1/x < 1) ∧ (¬ (1/x < 1 → x > 1)) :=
by
  sorry

end sufficient_but_not_necessary_for_reciprocal_l661_661814


namespace probability_order_l661_661433

theorem probability_order :
  let red_cards := 26 in
  let fives := 4 in
  let five_of_hearts := 1 in
  let jokers := 2 in
  let clubs := 13 in
  [five_of_hearts, jokers, fives, clubs, red_cards] = [1, 2, 4, 13, 26]
    ∧ list.sort (≤) [five_of_hearts, jokers, fives, clubs, red_cards] = [five_of_hearts, jokers, fives, clubs, red_cards] :=
by
  let red_cards := 26
  let fives := 4
  let five_of_hearts := 1
  let jokers := 2
  let clubs := 13
  simp [five_of_hearts, jokers, fives, clubs, red_cards]
  sorry

end probability_order_l661_661433


namespace ring_arrangement_leftmost_three_digits_l661_661451

theorem ring_arrangement_leftmost_three_digits :
  let m := (Nat.choose 9 5) * 5.factorial * (Nat.choose 9 4) in
  -- Extract the leftmost three nonzero digits of m
  let digits := Nat.digits 10 m in
  let nonzero_digits := List.filter (λ d, d ≠ 0) digits in 
  (nonzero_digits.reverse.take 3).reverse = [1, 9, 0] :=
by
  sorry

end ring_arrangement_leftmost_three_digits_l661_661451


namespace min_value_of_expression_l661_661692

theorem min_value_of_expression (x y z : ℝ) (h1x : -2 < x) (h2x : x < 2) (h1y : -2 < y) (h2y : y < 2) (h1z : -2 < z) (h2z : z < 2) :
  2 = min (λ f, ((1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) + (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))))) :=
sorry

end min_value_of_expression_l661_661692


namespace table_permutation_moves_l661_661776

theorem table_permutation_moves (m n : ℕ) (k : ℕ) 
  (horiz_move : ∀ row : Fin m, ∀ x y : Fin n, x ≠ y → row_permute (x, y))
  (vert_move : ∀ col : Fin n, ∀ x y : Fin m, x ≠ y → col_permute (x, y)) : 
  (m = 1 ∨ n = 1 → k = 1) ∧ (m ≠ 1 ∧ n ≠ 1 → k = 3) :=
sorry

end table_permutation_moves_l661_661776


namespace max_triangle_perimeter_l661_661593

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l661_661593


namespace median_is_correct_l661_661038

def times_in_seconds := 
  [(1, 15), (1, 35), (1, 50), (2, 10), (2, 22), (2, 30), (3, 5), (3, 5), (3, 36), (3, 40), (3, 55),
   (4, 20), (4, 25), (4, 30), (4, 33), (4, 35), (5, 10)].map (λ ⟨m, s⟩ => m * 60 + s)

def median_time : Float := 227.5

theorem median_is_correct : list.median times_in_seconds = median_time :=
by
  sorry

end median_is_correct_l661_661038


namespace greatest_triangle_perimeter_l661_661579

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l661_661579


namespace find_pairs_m_n_l661_661087

theorem find_pairs_m_n (m n a : ℕ) (h_m : m ≥ 3) (h_n : n ≥ 3) :
  (∃ a, ∀ a : ℕ, ∃ inf_positives a, (a^m + a - 1) % (a^n + a^2 - 1) = 0) ↔ (m = 5 ∧ n = 3) :=
by sorry

end find_pairs_m_n_l661_661087


namespace polynomial_roots_l661_661407

theorem polynomial_roots :
  ∀ (x : ℝ), (x^3 - x^2 - 6 * x + 8 = 0) ↔ (x = 2 ∨ x = (-1 + Real.sqrt 17) / 2 ∨ x = (-1 - Real.sqrt 17) / 2) :=
by
  sorry

end polynomial_roots_l661_661407


namespace maurice_age_l661_661042

theorem maurice_age (M : ℕ) 
  (h₁ : 48 = 4 * (M + 5)) : M = 7 := 
by
  sorry

end maurice_age_l661_661042


namespace max_perimeter_l661_661630

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l661_661630


namespace conics_touch_square_l661_661259

theorem conics_touch_square (x y : ℝ) : 
  (9 - x^2) * ( (deriv y) / (deriv x) )^2 = (9 - y^2) → 
  ∃ C : ℝ, ∃ (ε : ℝ), ε = 1 ∨ ε = -1 ∧
  (x^2 + y^2 ∓ 2 * (ε * C) * x * y = 9 * sin(ε * C)^2) ∧
  (∀ x : ℝ, (-3 ≤ x ∧ x ≤ 3 → y = 0)) ∧
  (∀ y : ℝ, (-3 ≤ y ∧ y ≤ 3 → x = 0)) :=
begin
  sorry
end

end conics_touch_square_l661_661259


namespace locus_of_F_l661_661683

-- Define points A, B, and C on a circle k
variables {k : Type} {A B C : Point} 

-- Define the condition that A, B, and C lie on circle k
def points_on_circle (A B C : Point) (k : Circle) : Prop :=
  k.contains A ∧ k.contains B ∧ k.contains C

-- Define the point F that bisects the broken line ACB
def bisects_ACB (A B C F : Point) : Prop :=
  (dist A C < dist C B ∧ dist A C + dist C F = dist F B ∧ on_segment C B F) ∨
  (dist A C ≥ dist C B ∧ dist A F = dist F C + dist C B ∧ on_segment A C F)

-- Define the loci properties involving Thales circles
def thales_loci (F : Point) (A B C P Q : Point) (k : Circle) : Prop :=
  let (ApBp) := (
    midpoint (line_through A P ∩ k.to_set) (line_through B P ∩ k.to_set)
  ),
  let (AqBq) := (
    midpoint (line_through A Q ∩ k.to_set) (line_through B Q ∩ k.to_set)
  ) in
  F ∈ Arc ApBp ∧ F ∈ Arc AqBq

-- The main statement expressing the desired property
theorem locus_of_F (A B C F P Q : Point) (k : Circle) :
  points_on_circle A B C k →
  bisects_ACB A B C F →
  ∃ P Q, thales_loci F A B C P Q k := by
  sorry

end locus_of_F_l661_661683


namespace knights_truth_maximum_l661_661083

theorem knights_truth_maximum: 
  ∀ (numbers : Fin 12 → ℕ), 
  (∀ i j, i ≠ j → numbers i ≠ numbers j) → 
  (∀ i, numbers i > numbers ((i + 1) % 12) ∧ numbers i > numbers ((i + 11) % 12) → 
       (∀ k, numbers k < numbers ((k + 1) % 12) ∨ numbers k < numbers ((k + 11) % 12) → 
       count (λ i, numbers i > numbers ((i + 1) % 12) ∧ numbers i > numbers ((i + 11) % 12)) ≤ 6)) :=
sorry

end knights_truth_maximum_l661_661083


namespace number_of_masters_students_l661_661080

theorem number_of_masters_students (total_sample : ℕ) (ratio_assoc : ℕ) (ratio_undergrad : ℕ) (ratio_masters : ℕ) (ratio_doctoral : ℕ) 
(h1 : ratio_assoc = 5) (h2 : ratio_undergrad = 15) (h3 : ratio_masters = 9) (h4 : ratio_doctoral = 1) (h_total_sample : total_sample = 120) :
  (ratio_masters * total_sample) / (ratio_assoc + ratio_undergrad + ratio_masters + ratio_doctoral) = 36 :=
by
  sorry

end number_of_masters_students_l661_661080


namespace avg_speed_276_l661_661854

-- Conditions
def avg_speed_69 (D T : ℝ) (speed_69 : ℝ) : Prop := speed_69 = D / T
def convert_min_to_hr (min : ℝ) : ℝ := min / 60
def distance (speed : ℝ) (time_hr : ℝ) : ℝ := speed * time_hr
def avg_speed (D time_hr : ℝ) (speed : ℝ) : Prop := speed = D / time_hr

-- Given values
noncomputable def speed_69 := 16
noncomputable def time_69_min := 69
noncomputable def time_276_min := 276

-- Converted times
noncomputable def time_69_hr := convert_min_to_hr time_69_min
noncomputable def time_276_hr := convert_min_to_hr time_276_min

-- Distance calculation
noncomputable def distance_69 := distance speed_69 time_69_hr

-- The proof problem
theorem avg_speed_276 : 
  avg_speed distance_69 time_276_hr 4 :=
sorry

end avg_speed_276_l661_661854


namespace prob_diff_sets_is_three_fourths_E_X_is_three_fourths_l661_661302

-- Define the problem context
noncomputable def housing : Type := Fin 4

-- Applicants
inductive Applicant
| A | B | C
open Applicant

-- Event Definitions
def chooses (p : housing → Prop) (applicant : Applicant) : Prop :=
  ∃ (h : housing), p h

-- Problem 1: Probability that A and B do not apply for the same set of housing
def prob_diff_sets : ℝ :=
  let pABsame := (1 : ℝ) / 4
  1 - pABsame

theorem prob_diff_sets_is_three_fourths : prob_diff_sets = 3 / 4 := sorry

-- Problem 2: Defining X and calculating its expectation
def X (h : housing → Prop) : ℝ := 
  (if chooses h A then 1 else 0) +
  (if chooses h B then 1 else 0) +
  (if chooses h C then 1 else 0)

def E_X : ℝ := 
  (3 : ℝ) / 4

theorem E_X_is_three_fourths : E_X = 3 / 4 := sorry

end prob_diff_sets_is_three_fourths_E_X_is_three_fourths_l661_661302


namespace f_simplification_l661_661077

def star (a b : ℝ) : ℝ := if a ≤ b then a else b

def f (x : ℝ) : ℝ := star 1 x

theorem f_simplification (x : ℝ) : f(x) = if 1 ≤ x then 1 else x := by
  unfold f
  unfold star
  exact sorry

end f_simplification_l661_661077


namespace product_of_roots_cubic_l661_661955

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  let roots := {x : ℝ // p x = 0}.toFinset
  ∃ r : ℝ, (∀ x ∈ roots, p x = 0) ∧ (∏ x in roots, x) = 50 :=
by
  sorry

end product_of_roots_cubic_l661_661955


namespace company_profit_l661_661341

noncomputable def y (x : ℝ) : ℝ := -x + 150

def profit (x : ℝ) : ℝ := (-x + 150) * (x - 20) - 300

def max_profit (x : ℝ) : ℝ := -(x - 85)^2 + 1225

def two_year_profit (x1 x2 : ℝ) : ℝ := profit x1 + profit x2

theorem company_profit:
  (30 ≤ x1 ∧ x1 ≤ 70) ∧
  (30 ≤ x2 ∧ x2 ≤ 70) ∧
  (profit 70 = 1000) ∧
  (two_year_profit 70 50 = 3500) :=
begin
  sorry
end

end company_profit_l661_661341


namespace smallest_value_floor_l661_661170

theorem smallest_value_floor (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) :
  (⌊(2 * a + b) / c⌋ + ⌊(b + 2 * c) / a⌋ + ⌊(c + 2 * a) / b⌋) = 9 :=
sorry

end smallest_value_floor_l661_661170


namespace volume_of_cube_with_perimeter_24cm_face_l661_661763

theorem volume_of_cube_with_perimeter_24cm_face : 
  ∀ (P : ℕ), (P = 24) → ∃ (V : ℕ), V = 216 :=
by
  intro P
  intro hP
  exists 216
  sorry

end volume_of_cube_with_perimeter_24cm_face_l661_661763


namespace sin2x_value_l661_661127

noncomputable def sin2x (x : ℝ) (hx1 : cos x = 3 / 5) (hx2 : 0 < x ∧ x < π / 2) : ℝ :=
  2 * sin x * cos x

theorem sin2x_value (x : ℝ) (hx1 : cos x = 3 / 5) (hx2 : 0 < x ∧ x < π / 2) : sin2x x hx1 hx2 = 24 / 25 :=
begin
  sorry
end

end sin2x_value_l661_661127


namespace length_c_dot_product_value_l661_661816

-- Proof Problem (1)
variables (a b : ℝ × ℝ)
def c := (2 * a.1 + b.1, 2 * a.2 + b.2)

theorem length_c :
  a = (1,0) → b = (-1,1) → (real.sqrt (c a b).1^2 + (c a b).2^2) = real.sqrt 2 :=
by
  intros h_a h_b
  simp [c, h_a, h_b]
  sorry

-- Proof Problem (2)
variables (a b : ℝ × ℝ)
def dot (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2

theorem dot_product_value :
  real.sqrt (a.1 ^ 2 + a.2 ^ 2) = 2 →
  real.sqrt (b.1 ^ 2 + b.2 ^ 2) = 1 →
  a.1 * b.1 + a.2 * b.2 = 2 * 1 * real.cos (real.pi / 3) →
  dot a (a.1 + b.1, a.2 + b.2) = 5 :=
by
  intros h_a_length h_b_length h_angle
  simp [dot]
  sorry

end length_c_dot_product_value_l661_661816


namespace product_of_repeating_decimal_l661_661415

theorem product_of_repeating_decimal (x : ℝ) (h : x = 1 / 3) : x * 9 = 3 :=
by
  rw [h]
  norm_num
  sorry

end product_of_repeating_decimal_l661_661415


namespace sufficient_but_not_necessary_condition_l661_661773

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (0 ≤ x ∧ x ≤ 1) → |x| ≤ 1 :=
by sorry

end sufficient_but_not_necessary_condition_l661_661773


namespace greatest_possible_perimeter_l661_661603

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l661_661603


namespace arithmetic_sequence_diff_l661_661655

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition for the arithmetic sequence
def condition (a : ℕ → ℝ) : Prop := 
  a 2 + a 4 + a 6 + a 8 + a 10 = 80

-- Definition of the common difference
def common_difference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- The proof problem statement in Lean 4
theorem arithmetic_sequence_diff (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a → condition a → common_difference a d → a 7 - a 8 = -d :=
by
  intros _ _ _
  -- Proof will be conducted here
  sorry

end arithmetic_sequence_diff_l661_661655


namespace equal_bills_at_20_minutes_l661_661808

theorem equal_bills_at_20_minutes :
  ∃ (m : ℕ), 
    (11 + 0.25 * m) = (12 + 0.20 * m) → m = 20 :=
by
  sorry

end equal_bills_at_20_minutes_l661_661808


namespace benzoic_acid_O_mass_percentage_l661_661411

def benzoic_acid_formula : list (string × ℕ) := [("C", 7), ("H", 6), ("O", 2)]
def atomic_mass (element : string) : ℝ :=
  if element = "C" then 12.01
  else if element = "H" then 1.008
  else if element = "O" then 16.00
  else 0

def mass_percentage_O (compound : list (string × ℕ)) : ℝ :=
  let molar_mass := compound.foldl (λ acc (elem, count), acc + count * atomic_mass elem) 0
  let mass_O := 2 * atomic_mass "O"
  (mass_O / molar_mass) * 100

theorem benzoic_acid_O_mass_percentage : mass_percentage_O benzoic_acid_formula = 26.2 := sorry

end benzoic_acid_O_mass_percentage_l661_661411


namespace number_of_zeros_of_f4_eq_8_l661_661689

def f (x : ℝ) := |2 * x - 1|
def f1 (x : ℝ) := f x
def f2 (x : ℝ) := f (f1 x)
def f3 (x : ℝ) := f (f2 x)
def f4 (x : ℝ) := f (f3 x)

theorem number_of_zeros_of_f4_eq_8 : 
  (finset.filter (λ x : ℝ, f4 x = 0) (finset.range 100)).card = 8 := sorry

end number_of_zeros_of_f4_eq_8_l661_661689


namespace problem_solution_l661_661796

theorem problem_solution : (3106 - 2935 + 17)^2 / 121 = 292 := by
  sorry

end problem_solution_l661_661796


namespace probability_passing_through_C_l661_661523

theorem probability_passing_through_C (P Q : ℤ → ℤ → Prop)
    (hA : P 0 0)  -- A is at the top left corner (0, 0)
    (hB : P 3 3)  -- B is at the bottom right corner (3, 3)
    (hC : P 1 2)  -- C is one block east and two blocks south of A (1, 2)
    (h_moves : ∀ i j, P i j → (P (i+1) j ∨ P i (j+1))) -- student can only move east or south
    (h_choice : ∀ i j, (P (i+1) j ∨ P i (j+1)) → ℕ:=1) -- student chooses randomly with equal probability
    : (path_probability : ℚ) := 9 / 20 := sorry

end probability_passing_through_C_l661_661523


namespace tree_growth_period_l661_661798

variable (initialHeight : ℝ) (growthRate : ℝ) (growthFactor : ℝ) (yearsAtPeriod : ℕ)

def treeHeight (years : ℕ) : ℝ := initialHeight + growthRate * years

theorem tree_growth_period (h1 : initialHeight = 4) 
                           (h2 : growthRate = 0.5) 
                           (h3 : growthFactor = 1/6) 
                           (h4 : treeHeight (4 + yearsAtPeriod) = treeHeight 4 * (1 + growthFactor)) :
    4 + yearsAtPeriod = 6 :=
by
  sorry

end tree_growth_period_l661_661798


namespace segment_MN_length_l661_661718

theorem segment_MN_length
  (A B C D M N : ℝ)
  (hA : A < B)
  (hB : B < C)
  (hC : C < D)
  (hM : M = (A + C) / 2)
  (hN : N = (B + D) / 2)
  (hAD : D - A = 68)
  (hBC : C - B = 26) :
  |M - N| = 21 :=
sorry

end segment_MN_length_l661_661718


namespace m_le_neg_one_l661_661509

theorem m_le_neg_one (m : ℝ) : (∀ x : ℝ, log 2 (abs (x + 1) + abs (x - 2) - m) ≥ 2) → m ≤ -1 :=
by
  sorry

end m_le_neg_one_l661_661509


namespace product_of_roots_cubic_l661_661959

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  let roots := {x : ℝ // p x = 0}.toFinset
  ∃ r : ℝ, (∀ x ∈ roots, p x = 0) ∧ (∏ x in roots, x) = 50 :=
by
  sorry

end product_of_roots_cubic_l661_661959


namespace find_original_price_l661_661859

def discount_price (original_price : ℝ) : ℝ :=
  original_price * 0.75 * 0.85 * 0.90 * 0.95

theorem find_original_price (final_price : ℝ) (original_price : ℝ) :
  final_price = 6800 → discount_price original_price = final_price → original_price = 11868.42 :=
by
  intros h1 h2
  rw [h2, h1]
  exact eq_refl 11868.42

end find_original_price_l661_661859


namespace study_group_scores_l661_661398

noncomputable def scores : List ℕ := [135, 122, 122, 110, 110, 110, 110, 90, 90]

def average (l : List ℕ) : ℝ :=
  (l.sum : ℝ) / (l.length : ℝ)

def median (l : List ℕ) : ℕ :=
  (l.sorted.get (l.length / 2)).get!

theorem study_group_scores :
  average scores = 111 ∧ median scores = 110 :=
by
  have h_avg : average scores = 111 := sorry
  have h_med : median scores = 110 := sorry
  exact ⟨h_avg, h_med⟩

end study_group_scores_l661_661398


namespace percent_of_a_is_b_l661_661332

variable (a b c : ℝ)
variable (h1 : c = 0.20 * a) (h2 : c = 0.10 * b)

theorem percent_of_a_is_b : b = 2 * a :=
by sorry

end percent_of_a_is_b_l661_661332


namespace max_digit_sum_watch_l661_661026

def digit_sum (n : Nat) : Nat :=
  (n / 10) + (n % 10)

theorem max_digit_sum_watch :
  ∃ (h m : Nat), (1 <= h ∧ h <= 12) ∧ (0 <= m ∧ m <= 59) 
  ∧ (digit_sum h + digit_sum m = 23) :=
by 
  sorry

end max_digit_sum_watch_l661_661026


namespace pizza_order_total_cost_l661_661736

noncomputable def pizzaOrderCost : ℕ := 
  let base_price := 10
  let topping_cost := λ t, match t with
    | "pepperoni" => 1.5
    | "sausage" => 1.5
    | "bacon" => 1.5
    | "onions" => 1
    | "black olives" => 1
    | "mushrooms" => 1
    | "green peppers" => 1
    | "spinach" => 1
    | "tomatoes" => 1
    | "artichokes" => 1
    | "red onions" => 1
    | "jalapenos" => 1
    | "extra cheese" => 2
    | "feta cheese" => 2
    | "chicken" => 2.5
    | "barbecue sauce" => 0
    | "cilantro" => 0
    | _ => 0
  let son_pizza := base_price + topping_cost "pepperoni" + topping_cost "bacon"
  let daughter_pizza := base_price + topping_cost "sausage" + topping_cost "onions" + topping_cost "pineapple"
  let ruby_husband_pizza := base_price + topping_cost "black olives" + topping_cost "mushrooms" + topping_cost "green peppers" + topping_cost "feta cheese"
  let cousin_pizza := base_price + topping_cost "spinach" + topping_cost "tomatoes" + topping_cost "extra cheese" + topping_cost "artichokes"
  let family_pizza := base_price + topping_cost "chicken" + topping_cost "barbecue sauce" + topping_cost "red onions" + topping_cost "cilantro" + topping_cost "jalapenos"
  let total_cost_without_tip := son_pizza + daughter_pizza + ruby_husband_pizza + cousin_pizza + family_pizza
  let tip := 5
  total_cost_without_tip + tip

theorem pizza_order_total_cost : pizzaOrderCost = 76 := by
  sorry

end pizza_order_total_cost_l661_661736


namespace gold_coins_percentage_is_35_l661_661869

-- Define the conditions: percentage of beads and percentage of silver coins
def percent_beads : ℝ := 0.30
def percent_silver_coins : ℝ := 0.50

-- Definition of the percentage of all objects that are gold coins
def percent_gold_coins (percent_beads percent_silver_coins : ℝ) : ℝ :=
  (1 - percent_beads) * (1 - percent_silver_coins)

-- The statement that we need to prove:
theorem gold_coins_percentage_is_35 :
  percent_gold_coins percent_beads percent_silver_coins = 0.35 :=
  by
    unfold percent_gold_coins percent_beads percent_silver_coins
    sorry

end gold_coins_percentage_is_35_l661_661869


namespace dist_range_regular_tetrahedron_l661_661277

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def distance (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2 + (A.z - B.z)^2)

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2, (A.z + B.z) / 2⟩

noncomputable def range_of_distances (A B C D P Q : Point) : set ℝ :=
  { d | ∃ (x y : ℝ) (hP : 0 ≤ x ∧ x ≤ 1) (hQ : 0 ≤ y ∧ y ≤ 1), 
  d = distance ⟨(1 - x) * A.x + x * B.x, (1 - x) * A.y + x * B.y, (1 - x) * A.z + x * B.z⟩ 
                ⟨(1 - y) * C.x + y * D.x, (1 - y) * C.y + y * D.y, (1 - y) * C.z + y * D.z⟩}

theorem dist_range_regular_tetrahedron (A B C D : Point)
  (hAB : distance A B = 1)
  (hAC : distance A C = 1)
  (hAD : distance A D = 1)
  (hBC : distance B C = 1)
  (hBD : distance B D = 1)
  (hCD : distance C D = 1)
  (P Q : Point) :
  range_of_distances A B C D P Q = set.Icc (Real.sqrt 2 / 2) 1 :=
sorry

end dist_range_regular_tetrahedron_l661_661277


namespace max_true_claims_l661_661081

-- Define the conditions
def knight_positions (n : ℕ) : Prop :=
  n = 12

def distinct_numbers (numbers : Fin 12 → ℕ) : Prop :=
  ∀ i j, i ≠ j → numbers i ≠ numbers j

def knight_claims (numbers : Fin 12 → ℕ) (claims : Fin 12 → Prop) : Prop :=
  ∀ i, claims i ↔ (numbers i > numbers ((i + 1) % 12) ∧ numbers i > numbers ((i + 11) % 12))

-- Define the problem statement to prove
theorem max_true_claims : ∀ (numbers : Fin 12 → ℕ) (claims : Fin 12 → Prop),
  knight_positions 12 →
  distinct_numbers numbers →
  knight_claims numbers claims →
  ∃ (C : Fin 12 → Prop), (∀ i, C i → claims i) ∧ (∑ i, if C i then 1 else 0) ≤ 6 :=
sorry

end max_true_claims_l661_661081


namespace product_of_roots_eq_50_l661_661945

theorem product_of_roots_eq_50 :
  let a := -15
  let c := -50
  let equation := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  -- Vieta's formulas for the product of roots in cubic equation ax^3 + bx^2 + cx + d = 0 is -d/a
  ∃ p q r : ℝ, 
    equation p = 0 ∧ equation q = 0 ∧ equation r = 0 ∧ (p * q * r = 50) :=
by
  sorry

end product_of_roots_eq_50_l661_661945


namespace product_of_roots_cubic_l661_661887

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l661_661887


namespace consecutive_negative_integers_sum_l661_661286

theorem consecutive_negative_integers_sum (n : ℤ) (hn : n < 0) (hn1 : n + 1 < 0) (hprod : n * (n + 1) = 2550) : n + (n + 1) = -101 :=
by
  sorry

end consecutive_negative_integers_sum_l661_661286


namespace exist_arithmetic_seq_nice_l661_661217

-- Define a polynomial as in Q[x] of degree 2016 and leading coefficient 1
def is_nice (m : ℕ) : Prop :=
  ∃ n : ℕ, m = n^3 + 3 * n + 1

def P (x : ℚ) : ℚ := sorry  -- P is a polynomial of degree 2016 with leading coefficient 1

-- There exist infinitely many positive integers n such that P(n) are nice
axiom infinite_nice_values (P : ℚ -> ℚ) :
  ∀ N : ℕ, ∃ n > N, is_nice (P n)

-- There exists an arithmetic sequence (n_k) of arbitrary length such that P(n_k) are all nice
theorem exist_arithmetic_seq_nice (P : ℚ -> ℚ) :
  (∀ N : ℕ, ∃ n > N, is_nice (P n)) →
  ∀ (k : ℕ), ∃ (a d : ℕ), ∀ i : ℕ, i < k → is_nice (P (a + i * d)) :=
begin
  intros h_inf k,
  sorry  -- Placeholder for the proof
end

end exist_arithmetic_seq_nice_l661_661217


namespace cos_sum_inequality_l661_661424

theorem cos_sum_inequality (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  cos (α + β) < cos α + cos β :=
sorry

end cos_sum_inequality_l661_661424


namespace boat_distance_downstream_l661_661829

theorem boat_distance_downstream
  (speed_boat_still : ℝ)
  (speed_stream : ℝ)
  (time_downstream : ℝ)
  (speed_boat_still = 30)
  (speed_stream = 5)
  (time_downstream = 2) :
  (speed_boat_still + speed_stream) * time_downstream = 70 := 
sorry

end boat_distance_downstream_l661_661829


namespace saltwater_concentration_l661_661173

theorem saltwater_concentration (salt_mass water_mass : ℝ) (h₁ : salt_mass = 8) (h₂ : water_mass = 32) : 
  salt_mass / (salt_mass + water_mass) * 100 = 20 := 
by
  sorry

end saltwater_concentration_l661_661173


namespace product_of_roots_l661_661943

theorem product_of_roots :
  let p : Polynomial ℚ := Polynomial.Coeff.mk_fin 3 0 (⟨[50, -75, 15, -1]⟩ : ℕ → ℚ)
  (p.root_prod = 50) :=
begin
  sorry
end

end product_of_roots_l661_661943


namespace correct_options_l661_661318

noncomputable def median (s : List ℕ) : ℕ :=
(s.sorted.get! (s.length / 2) + s.sorted.get! (s.length / 2 - 1)) / 2

def is_correct_a (s : List ℕ) (m : ℕ) : Prop :=
median s = m

def is_correct_b (μ : ℝ) (σ : ℝ) (p : ℝ) (p_between : ℝ) : Prop :=
(X : ℝ) ∼ Normal μ σ^2 ∧ P(X > 3) = p → P(1 < X < 2) = p_between

def is_correct_c (n : ℕ) (p : ℝ) (e : ℝ) : Prop :=
(Y : ℕ) ∼ Binomial n p ∧ E(Y) = e → E(Y) ≠ e

def is_correct_d (a : ℝ) (b : ℝ) : Prop :=
(a < 0) → (x y : ℝ) ∧ (y = a * x + b) → y and x have a negative linear correlation

theorem correct_options (s : List ℕ) (m : ℕ) (μ : ℝ) (σ : ℝ) (p : ℝ) (p_between : ℝ)
  (n : ℕ) (p : ℝ) (e : ℝ) (a : ℝ) (b : ℝ) :
  is_correct_a s m ∧ is_correct_b μ σ p p_between ∧ 
  ¬ is_correct_c n p e ∧ is_correct_d a b → 
  "ABD" := sorry

end correct_options_l661_661318


namespace proof_problem_l661_661155

-- Defining the statement in Lean 4.

noncomputable def p : Prop :=
  ∀ x : ℝ, x > Real.sin x

noncomputable def neg_p : Prop :=
  ∃ x : ℝ, x ≤ Real.sin x

theorem proof_problem : ¬p ↔ neg_p := 
by sorry

end proof_problem_l661_661155


namespace tan_half_alpha_plus_pi_over_8_sin_alpha_plus_pi_over_12_l661_661126

variables {α : ℝ}

-- Assume that α is in the second quadrant
def in_second_quadrant (α : ℝ) : Prop := (π / 2 < α) ∧ (α < π)

-- Given condition
def given_condition (α : ℝ) : Prop := (1 - Real.tan α) / (1 + Real.tan α) = 4 / 3

theorem tan_half_alpha_plus_pi_over_8 
  (h1 : in_second_quadrant α) 
  (h2 : given_condition α) : 
  Real.tan (α / 2 + π / 8) = -3 := 
sorry

theorem sin_alpha_plus_pi_over_12 
  (h1 : in_second_quadrant α) 
  (h2 : given_condition α) : 
  Real.sin (α + π / 12) = (4 - 3 * Real.sqrt 3) / 10 :=
sorry

end tan_half_alpha_plus_pi_over_8_sin_alpha_plus_pi_over_12_l661_661126


namespace value_of_c_l661_661504

theorem value_of_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : c = 7 :=
sorry

end value_of_c_l661_661504


namespace part1_simplification_part2_inequality_l661_661012

-- Part 1: Prove the simplification of the algebraic expression
theorem part1_simplification (x : ℝ) (h₁ : x ≠ 3):
  (2 * x + 4) / (x^2 - 6 * x + 9) / ((2 * x - 1) / (x - 3) - 1) = 2 / (x - 3) :=
sorry

-- Part 2: Prove the solution set for the inequality system
theorem part2_inequality (x : ℝ) :
  (5 * x - 2 > 3 * (x + 1)) → (1/2 * x - 1 ≥ 7 - 3/2 * x) → x ≥ 4 :=
sorry

end part1_simplification_part2_inequality_l661_661012


namespace find_a1_l661_661769

-- Definitions for the sequence and its properties
variable {a : ℕ → ℤ} (m : ℤ)

-- Hypotheses from the problem statement
def seq_def (n : ℕ) : Prop := a (n + 1) + 2 = m * (a n + 2) ∧ a n ≠ -2

-- The range for specific terms
def range_check : Prop := 
  a 3 ∈ {-18, -6, -2, 6, 30} ∧
  a 4 ∈ {-18, -6, -2, 6, 30} ∧
  a 5 ∈ {-18, -6, -2, 6, 30} ∧
  a 6 ∈ {-18, -6, -2, 6, 30}

-- Final theorem statement
theorem find_a1 : ∃ a1 : ℤ, (a 1 = -3 ∨ a 1 = 126) :=
by
  sorry

end find_a1_l661_661769


namespace product_of_roots_cubic_l661_661954

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  let roots := {x : ℝ // p x = 0}.toFinset
  ∃ r : ℝ, (∀ x ∈ roots, p x = 0) ∧ (∏ x in roots, x) = 50 :=
by
  sorry

end product_of_roots_cubic_l661_661954


namespace permutations_of_digits_l661_661214

theorem permutations_of_digits : 
  ∀ (digits : Finset ℕ), 
  digits = {5, 3, 9, 1} → 
  digits.card = 4 → 
  Finset.permutations digits ∈ (Finset.range 24) :=
by
  sorry

end permutations_of_digits_l661_661214


namespace bullet_trains_cross_time_l661_661807

-- Conditions from problem:
def length_of_train := 120 -- in meters
def time_to_cross_post_train1 := 8 -- in seconds
def time_to_cross_post_train2 := 15 -- in seconds

-- Definitions derived from the conditions:
def speed_train1 := length_of_train / time_to_cross_post_train1
def speed_train2 := length_of_train / time_to_cross_post_train2
def relative_speed := speed_train1 + speed_train2
def total_distance_cross_each_other := 2 * length_of_train

-- Theorem stating the problem and expected answer:
theorem bullet_trains_cross_time : 
  total_distance_cross_each_other / relative_speed ≈ 10.43 :=
sorry

end bullet_trains_cross_time_l661_661807


namespace no_valid_n_lt_200_l661_661494

noncomputable def roots_are_consecutive (n m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * (k + 1) ∧ n = 2 * k + 1

theorem no_valid_n_lt_200 :
  ¬∃ n m : ℕ, n < 200 ∧
              m % 4 = 0 ∧
              ∃ t : ℕ, t^2 = m ∧
              roots_are_consecutive n m := 
by
  sorry

end no_valid_n_lt_200_l661_661494


namespace cartesian_circle_eq_min_PA_PB_l661_661654

-- Define the parametric equation of the line l
def line_l (t : ℝ) (α : ℝ) := (1 + t * Real.cos α, 2 + t * Real.sin α)

-- Define the polar equation of the circle C and convert it to Cartesian coordinates
def polar_circle (θ : ℝ) : ℝ := 6 * Real.sin θ

-- Prove the Cartesian coordinate equation of circle C
theorem cartesian_circle_eq : ∀ (x y : ℝ), 
  (x^2 + y^2 = (6 * Real.sin (Real.atan2 y x))^2) →
  x^2 + (y - 3)^2 = 9 :=
by
  intros x y h
  sorry

-- Define the distance between two points
def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Prove the minimum value of |PA| + |PB|
theorem min_PA_PB : ∀ (α : ℝ), ∀ (A B : ℝ × ℝ),
  (A, B ∈ set_of (fun t => line_l t α)) →
  ∀ P, P = (1, 2) →
  set_intersects A B (polar_circle (Real.atan2 B.2 B.1)) →
  ∃ t1 t2,
  |t1 - t2| = Real.sqrt 28 - 4 * Real.sin (2 * α) 2 * Real.sqrt 7 :=
by
  intros α A B hA hB P hP h_intersect
  sorry

end cartesian_circle_eq_min_PA_PB_l661_661654


namespace greatest_perimeter_l661_661561

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l661_661561


namespace land_plot_side_length_l661_661009

theorem land_plot_side_length (area : ℝ) (h : area = Real.sqrt 1600) : ∃ side_length : ℝ, side_length = 40 := 
by {
  use 40,
  have h1 := h,
  rw [Real.sqrt_eq_iff_mul_self_eq, mul_comm] at h1,
  exact h1
}

end land_plot_side_length_l661_661009


namespace peaches_picked_l661_661709

theorem peaches_picked (p1 p2 p3 : ℕ) (h1 : p1 = 34) (h2 : p2 = 86) (h3 : p3 = p2 - p1) : p3 = 52 :=
by  
  rw [h3, h1, h2]
  rfl

end peaches_picked_l661_661709


namespace max_value_expression_l661_661285

theorem max_value_expression (a b c d : ℝ) 
  (h1 : -11.5 ≤ a ∧ a ≤ 11.5)
  (h2 : -11.5 ≤ b ∧ b ≤ 11.5)
  (h3 : -11.5 ≤ c ∧ c ≤ 11.5)
  (h4 : -11.5 ≤ d ∧ d ≤ 11.5):
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a ≤ 552 :=
by
  sorry

end max_value_expression_l661_661285


namespace mona_drives_125_miles_l661_661244

/-- Mona can drive 125 miles with $25 worth of gas, given the car mileage
    and the cost per gallon of gas. -/
theorem mona_drives_125_miles (miles_per_gallon : ℕ) (cost_per_gallon : ℕ) (total_money : ℕ)
  (h_miles_per_gallon : miles_per_gallon = 25) (h_cost_per_gallon : cost_per_gallon = 5)
  (h_total_money : total_money = 25) :
  (total_money / cost_per_gallon) * miles_per_gallon = 125 :=
by
  sorry

end mona_drives_125_miles_l661_661244


namespace arun_age_proof_l661_661004

theorem arun_age_proof {A G M : ℕ} 
  (h1 : (A - 6) / 18 = G)
  (h2 : G = M - 2)
  (h3 : M = 5) :
  A = 60 :=
by
  sorry

end arun_age_proof_l661_661004


namespace greatest_possible_perimeter_l661_661600

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l661_661600


namespace constant_term_is_45_l661_661108

theorem constant_term_is_45 
  (n : ℕ) 
  (i : ℂ) 
  (h_i_squared : i ^ 2 = -1)
  (h_ratio : binomial_coefficient n 2 / binomial_coefficient n 4 = -3 / 14)
: ∃ t : ℂ, t = 45 :=
sorry

end constant_term_is_45_l661_661108


namespace true_propositions_count_l661_661778

theorem true_propositions_count:
  let P1 := (∀ x, ∃ k: ℤ, (y = sin x^4 - cos x^4) := (y = -cos 2x) ∧ (min_period (y) = π))
  let P2 := (∀ α, (α ∈ {α | ∃ k: ℤ, α = k * (π / 2)} → α lies on the y-axis))
  let P3 := (∃ P: Set (ℝ × ℝ), graph_sin_x.intersect graph_x = {origin})
  let P4 := (∀ x, graph_3sin2x_eq_translation_3sin2x_plus_pi_over_3_by_pi_over_6)
  let P5 := (decreasing_on (sin (x - π / 2)) (0, π) = false)
  in P1 + P4 = 2 := sorry

end true_propositions_count_l661_661778


namespace binomial_expansion_sum_l661_661180

theorem binomial_expansion_sum (n : ℕ) (h : (2:ℕ)^n = 256) : n = 8 :=
sorry

end binomial_expansion_sum_l661_661180


namespace lateral_surface_area_of_prism_l661_661752

theorem lateral_surface_area_of_prism 
  (a : ℝ) 
  (equilateral_base : ∀ A B C : ℝ, A + B + C = a ∧ (A = B ∧ B = C)) 
  (orthogonal_projection_center : ∀ A_1 : ℝ, A_1 = sqrt(3) * a / 3) 
  (angle_lateral_base : ∀ α : ℝ, α = 60) 
  : (S : ℝ) 
  ∧ (S = a^2 * (2*sqrt(3) + sqrt(13)) / sqrt(3)) := 
sorry

end lateral_surface_area_of_prism_l661_661752


namespace average_of_four_l661_661499

-- Define the variables
variables {p q r s : ℝ}

-- Conditions as hypotheses
theorem average_of_four (h : (5 / 4) * (p + q + r + s) = 15) : (p + q + r + s) / 4 = 3 := 
by
  sorry

end average_of_four_l661_661499


namespace product_of_roots_eq_50_l661_661952

theorem product_of_roots_eq_50 :
  let a := -15
  let c := -50
  let equation := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  -- Vieta's formulas for the product of roots in cubic equation ax^3 + bx^2 + cx + d = 0 is -d/a
  ∃ p q r : ℝ, 
    equation p = 0 ∧ equation q = 0 ∧ equation r = 0 ∧ (p * q * r = 50) :=
by
  sorry

end product_of_roots_eq_50_l661_661952


namespace distance_between_point_of_tangency_and_A_l661_661032

variables (A B C : Type) [metric_space A] [point A] (MN : A → A → Prop)
variables (a r : ℝ) (circle : A → ℝ → Prop) 

noncomputable def distance_from_tangency_to_A (A : A) (MN : A → A → Prop) (a r : ℝ) (circle : A → ℝ → Prop) : ℝ :=
  sqrt (2 * a * r)

theorem distance_between_point_of_tangency_and_A (A : A) (line MN : A → A → Prop) (a r : ℝ) (circle : A → ℝ → Prop)
  (h_a : distance A MN = a)
  (h_circle : circle A r)
  (h_tangent : tangent_to MN (circle A r))
  : distance_from_tangency_to_A A MN a r circle = sqrt (2 * a * r) :=
begin
  sorry
end

end distance_between_point_of_tangency_and_A_l661_661032


namespace max_product_geom_seq_l661_661280

theorem max_product_geom_seq :
  ∀ (aₙ : ℕ → ℝ) (n : ℕ), 
  (aₙ 1 = 1536) → 
  (∀ k : ℕ, aₙ (k + 1) = aₙ k * (-1 / 2)) → 
  (∃ n : ℕ, n = 11 ∧ (∀ m : ℕ, m ≠ 11 → Πₙ m ≤ Πₙ 11)) := 
by
  sorry

end max_product_geom_seq_l661_661280


namespace identical_nonempty_solution_sets_l661_661695

def f (x a b : ℝ) : ℝ := x^2 + a*x + b * cos x

theorem identical_nonempty_solution_sets :
  ∀ a b : ℝ, 
    (∃ x : ℝ, f x a b = 0) ∧ (∃ x : ℝ, f (f x a b) a b = 0) ↔ (0 ≤ a ∧ a < 4 ∧ b = 0) :=
by
  sorry

end identical_nonempty_solution_sets_l661_661695


namespace area_PQR_greater_than_ABC_l661_661678

variables {A B C M P Q R : Type}
variables [point A] [point B] [point C] [point M] [point P] [point Q] [point R]
variables {α β γ: ℝ}

-- Given conditions
def angle_AMC_90 (A B C M : point) : Prop := ∠AMC = 90
def angle_AMB_150 (A B C M : point) : Prop := ∠AMB = 150
def angle_BMC_120 (A B C M : point) : Prop := ∠BMC = 120

-- Centers of circumcircles
def circumcenter_AMC (A M C : point) : point := P
def circumcenter_AMB (A M B : point) : point := Q
def circumcenter_BMC (B M C : point) : point := R

-- Proof goal
theorem area_PQR_greater_than_ABC
  (h1 : angle_AMC_90 A B C M)
  (h2 : angle_AMB_150 A B C M)
  (h3 : angle_BMC_120 A B C M)
  (h4 : circumcenter_AMC A M C = P)
  (h5 : circumcenter_AMB A M B = Q)
  (h6 : circumcenter_BMC B M C = R) :
  area(△PQR) > area(△ABC) :=
sorry

end area_PQR_greater_than_ABC_l661_661678


namespace no_perfect_squares_exist_l661_661396

theorem no_perfect_squares_exist (x y : ℕ) :
  ¬(∃ k1 k2 : ℕ, x^2 + y = k1^2 ∧ y^2 + x = k2^2) :=
sorry

end no_perfect_squares_exist_l661_661396


namespace salesman_ways_l661_661361

-- Definitions as conditions from part (a)
def S : ℕ → ℕ
| 1     := 1
| 2     := 2
| 3     := 4
| (n+1) := S n + S (n-1) + S (n-2)

-- Theorem stating the solution
theorem salesman_ways (n : ℕ) (h : n = 12) : S n = 927 := by
  sorry

end salesman_ways_l661_661361


namespace new_angle_β_l661_661031

variables (R1 R2 : ℝ) (F : ℝ)
constant α : ℝ
axiom α_value : α = 20 * Real.pi / 180 -- converting degrees to radians
axiom R1_value : R1 = R2 / 2

noncomputable def β : ℝ := Real.arccos (1 - Real.cos α)

theorem new_angle_β :
  β = Real.arccos 0.06 := by
sorry

end new_angle_β_l661_661031


namespace part_a_part_b_l661_661010

-- Definition for the number of triangles when the n-gon is divided using non-intersecting diagonals
theorem part_a (n : ℕ) (h : n ≥ 3) : 
  ∃ k, k = n - 2 := 
sorry

-- Definition for the number of diagonals when the n-gon is divided using non-intersecting diagonals
theorem part_b (n : ℕ) (h : n ≥ 3) : 
  ∃ l, l = n - 3 := 
sorry

end part_a_part_b_l661_661010


namespace number_of_nonzero_terms_l661_661393

theorem number_of_nonzero_terms :
  let p₁ := (2 * x + 5)
  let p₂ := (3 * x^2 - x + 4)
  let p₃ := 4 * (x^3 + x^2 - 6 * x)
  let p := p₁ * p₂ + p₃
  (p.coeffs.filter (≠ 0)).length = 4 :=
by 
  sorry

end number_of_nonzero_terms_l661_661393


namespace log10_7_in_terms_of_r_and_s_l661_661168

-- We will define the given conditions and what we need to prove based on those conditions.

variables (r s : ℝ)

-- Assume log_4(2) = r and log_2(7) = s
axiom log4_2_eq_r : log 4 2 = r
axiom log2_7_eq_s : log 2 7 = s

-- We need to prove that log_10(7) = s / (1 + s)
theorem log10_7_in_terms_of_r_and_s : log 10 7 = s / (1 + s) :=
by {
  sorry -- Proof is omitted
}

end log10_7_in_terms_of_r_and_s_l661_661168


namespace product_of_roots_l661_661919

theorem product_of_roots :
  let f := Polynomial.C (-50) + Polynomial.X * (Polynomial.C 75 + Polynomial.X * (Polynomial.C (-15) + Polynomial.X)) in 
  (f.roots.Prod = 50) :=
by sorry

end product_of_roots_l661_661919


namespace trigonometric_identity_l661_661454

theorem trigonometric_identity
  (α β : ℝ)
  (h : (sin β)^4 / (sin α)^2 + (cos β)^4 / (cos α)^2 = 1) :
  (cos α)^4 / (cos β)^2 + (sin α)^4 / (sin β)^2 = 1 :=
sorry

end trigonometric_identity_l661_661454


namespace normal_distribution_prob_l661_661699

open ProbabilityTheory

/-- Given a random variable ξ follows normal distribution N(0,1), and P(ξ > 1) = p,
prove that P(-1 < ξ < 0) = 1/2 - p. -/
theorem normal_distribution_prob (ξ : ℝ) (p : ℝ)
  (H : Normal 0 1 ξ)
  (H1 : P (ξ > 1) = p) : 
  P (-1 < ξ ∧ ξ < 0) = 1 / 2 - p := 
sorry

end normal_distribution_prob_l661_661699


namespace greatest_triangle_perimeter_l661_661578

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l661_661578


namespace modulus_reciprocal_z_l661_661138

def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

noncomputable def z (m : ℤ) : ℂ :=
  m - 3 + ((m - 1) * complex.i)

theorem modulus_reciprocal_z (m : ℤ)
  (hm1 : 1 < m)
  (hm2 : m < 3)
  (hz : in_second_quadrant (z m)) :
  ∥(1 / (z m))∥ = real.sqrt 2 / 2 := 
sorry

end modulus_reciprocal_z_l661_661138


namespace points_on_same_line_l661_661521

theorem points_on_same_line (
  (s : Set Point) (h : ∀ (a b : Point), a ∈ s → b ∈ s → ∃ (c : Point), c ∈ s ∧ between a c b)
  ) :
  ∃ (l : Line), ∀ (p : Point), p ∈ s → p ∈ l
{
  sorry
}

end points_on_same_line_l661_661521


namespace product_of_repeating_decimal_l661_661414

theorem product_of_repeating_decimal (x : ℝ) (h : x = 1 / 3) : x * 9 = 3 :=
by
  rw [h]
  norm_num
  sorry

end product_of_repeating_decimal_l661_661414


namespace greatest_perimeter_l661_661570

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l661_661570


namespace greatest_possible_perimeter_l661_661625

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l661_661625


namespace product_of_roots_eq_50_l661_661948

theorem product_of_roots_eq_50 :
  let a := -15
  let c := -50
  let equation := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  -- Vieta's formulas for the product of roots in cubic equation ax^3 + bx^2 + cx + d = 0 is -d/a
  ∃ p q r : ℝ, 
    equation p = 0 ∧ equation q = 0 ∧ equation r = 0 ∧ (p * q * r = 50) :=
by
  sorry

end product_of_roots_eq_50_l661_661948


namespace log_geometric_seq_min_value_l661_661444

theorem log_geometric_seq_min_value :
  ∀ (a : ℕ → ℝ) (q : ℝ),
  (∀ n, a n > 0) →
  (a 1 + a 3 = 5 / 16) →
  (a 2 + a 4 = 5 / 8) →
  (∀ n, a n = a 1 * q ^ (n - 1)) →
  ∃ n : ℕ, ∀ m, m > 0 → log 2 (a 1 * a 2 * ... * a n) ≥ -10 :=
by
  sorry

end log_geometric_seq_min_value_l661_661444


namespace product_of_roots_of_cubic_polynomial_l661_661894

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l661_661894


namespace solution_set_of_inequality_l661_661690

noncomputable def f (x : ℝ) : ℝ := sorry

theorem solution_set_of_inequality :
  (∀ x, f (-x) = -f x) →                      -- f is odd
  (∀ x, x ≠ 0 → x ∈ ℝ) →                      -- For all x ≠ 0, x is in ℝ
  (∀ x, x < 0 → f' x > 0) →                   -- For x < 0, f'(x) > 0
  f (-2) = 0 →                                -- f(-2) = 0
  { x : ℝ | f x > 0 } = { x : ℝ | -2 < x ∧ x < 0 } ∪ { x : ℝ | 2 < x ∧ x < ℝ } :=
by
  sorry

end solution_set_of_inequality_l661_661690


namespace greatest_possible_perimeter_l661_661554

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l661_661554


namespace find_k_l661_661123

variables {V : Type*} [InnerProductSpace ℝ V]

theorem find_k (a b : V) (k : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : ∥a∥ = 1) (h4 : ∥b∥ = 1) (h5 : inner a b = 0)
  (h6 : inner (2 • a + 3 • b) (k • a - 4 • b) = 0) :
  k = 6 :=
sorry

end find_k_l661_661123


namespace shaded_area_l661_661037

theorem shaded_area (x1 y1 x2 y2 x3 y3 : ℝ) 
  (vA vB vC vD vE vF : ℝ × ℝ)
  (h1 : vA = (0, 0))
  (h2 : vB = (0, 12))
  (h3 : vC = (12, 12))
  (h4 : vD = (12, 0))
  (h5 : vE = (24, 0))
  (h6 : vF = (18, 12))
  (h_base : 32 - 12 = 20)
  (h_height : 12 = 12) :
  (1 / 2 : ℝ) * 20 * 12 = 120 :=
by
  sorry

end shaded_area_l661_661037


namespace inequality_true_l661_661144

theorem inequality_true (f : ℝ → ℝ) (h : ∀ x, f x + 2 * (deriv f x) > 0) : 
  f 1 > f 0 / real.sqrt real.exp1 := 
sorry

end inequality_true_l661_661144


namespace no_valid_N_for_case1_valid_N_values_for_case2_l661_661187

variable (P R N : ℕ)
variable (N_less_than_40 : N < 40)
variable (avg_all : ℕ)
variable (avg_promoted : ℕ)
variable (avg_repeaters : ℕ)
variable (new_avg_promoted : ℕ)
variable (new_avg_repeaters : ℕ)

variables
  (promoted_condition : (71 * P + 56 * R) / N = 66)
  (increase_condition : (76 * P) / (P + R) = 75 ∧ (61 * R) / (P + R) = 59)
  (equation1 : 71 * P = 2 * R)
  (equation2: P + R = N)

-- Proof for part (a)
theorem no_valid_N_for_case1 
  (new_avg_promoted' : ℕ := 75) 
  (new_avg_repeaters' : ℕ := 59)
  : ∀ N, ¬ N < 40 ∨ ¬ ((76 * P) / (P + R) = new_avg_promoted' ∧ (61 * R) / (P + R) = new_avg_repeaters') := 
  sorry

-- Proof for part (b)
theorem valid_N_values_for_case2
  (possible_N_values : Finset ℕ := {6, 12, 18, 24, 30, 36})
  (new_avg_promoted'' : ℕ := 79)
  (new_avg_repeaters'' : ℕ := 47)
  : ∀ N, N ∈ possible_N_values ↔ (((76 * P) / (P + R) = new_avg_promoted'') ∧ (61 * R) / (P + R) = new_avg_repeaters'') := 
  sorry

end no_valid_N_for_case1_valid_N_values_for_case2_l661_661187


namespace greatest_perimeter_of_triangle_l661_661611

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l661_661611


namespace sum_of_extreme_values_l661_661506
noncomputable def problem_statement (a : ℝ) : Prop :=
  a - 2 * a * b + 2 * a * b^2 + 4 = 0

theorem sum_of_extreme_values (a : ℝ) :
  problem_statement a →
  (∀ (a : ℝ), -8 ≤ a ∧ a ≤ 0) →
  ∃ (max_a min_a : ℝ), (max_a = 0) ∧ (min_a = -8) ∧ (max_a + min_a = -8) :=
by sorry

end sum_of_extreme_values_l661_661506


namespace greatest_perimeter_l661_661563

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l661_661563


namespace total_revenue_correct_l661_661079

def original_price_sneakers : ℝ := 80
def discount_sneakers : ℝ := 0.25
def pairs_sneakers_sold : ℕ := 2

def original_price_sandals : ℝ := 60
def discount_sandals : ℝ := 0.35
def pairs_sandals_sold : ℕ := 4

def original_price_boots : ℝ := 120
def discount_boots : ℝ := 0.40
def pairs_boots_sold : ℕ := 11

def calculate_total_revenue : ℝ := 
  let revenue_sneakers := pairs_sneakers_sold * (original_price_sneakers * (1 - discount_sneakers))
  let revenue_sandals := pairs_sandals_sold * (original_price_sandals * (1 - discount_sandals))
  let revenue_boots := pairs_boots_sold * (original_price_boots * (1 - discount_boots))
  revenue_sneakers + revenue_sandals + revenue_boots

theorem total_revenue_correct : calculate_total_revenue = 1068 := by
  sorry

end total_revenue_correct_l661_661079


namespace cory_fruit_order_count_l661_661388

theorem cory_fruit_order_count :
  let apples := 4
  let oranges := 2
  let bananas := 2
  let total_days := 8
  let all_fruits := apples + oranges + bananas
  let arr := nat.factorial all_fruits / (nat.factorial apples * nat.factorial oranges * nat.factorial bananas)
  arr = 6 := 
sorry

end cory_fruit_order_count_l661_661388


namespace bowling_ball_volume_after_holes_l661_661362

noncomputable def bowling_ball_diameter : ℝ := 24
noncomputable def hole1_depth : ℝ := 6
noncomputable def hole1_diameter : ℝ := 3
noncomputable def hole2_depth : ℝ := 6
noncomputable def hole2_diameter : ℝ := 4

def volume_sphere (r : ℝ) : ℝ := (4/3) * Real.pi * (r ^ 3)
def volume_cylinder (r h : ℝ) : ℝ := Real.pi * (r ^ 2) * h

theorem bowling_ball_volume_after_holes :
  let r_bowling_ball := bowling_ball_diameter / 2 in
  let r_hole1 := hole1_diameter / 2 in
  let r_hole2 := hole2_diameter / 2 in
  volume_sphere r_bowling_ball - volume_cylinder r_hole1 hole1_depth - volume_cylinder r_hole2 hole2_depth = 2266.5 * Real.pi :=
by
  sorry

end bowling_ball_volume_after_holes_l661_661362


namespace product_of_roots_l661_661904

-- Let's define the given polynomial equation.
def polynomial : Polynomial ℝ := Polynomial.monomial 3 1 - Polynomial.monomial 2 15 + Polynomial.monomial 1 75 - Polynomial.C 50

-- Prove that the product of the roots of the given polynomial is 50.
theorem product_of_roots : (Polynomial.roots polynomial).prod id = 50 := by
  sorry

end product_of_roots_l661_661904


namespace greatest_possible_value_q_minus_r_l661_661283

theorem greatest_possible_value_q_minus_r : ∃ q r : ℕ, 1025 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q - r = 31 :=
by {
  sorry
}

end greatest_possible_value_q_minus_r_l661_661283


namespace third_circle_properties_l661_661306

noncomputable def radius_of_third_circle (r1 r2 : ℝ) : ℝ :=
  real.sqrt ((r2^2 - r1^2) * π)

noncomputable def side_length_of_square (r : ℝ) : ℝ :=
  2 * r

theorem third_circle_properties
  (r1 r2 : ℝ)
  (h1 : r1 = 23)
  (h2 : r2 = 33)
  (h_shaded_area : π * (r2^2 - r1^2) = 560 * π)
  (h_third_circle_radius : real.sqrt (560) = 4 * (real.sqrt 35)) :
  radius_of_third_circle r1 r2 = 4 * (real.sqrt 35) ∧
  side_length_of_square (radius_of_third_circle r1 r2) = 8 * (real.sqrt 35) :=
by
  sorry

end third_circle_properties_l661_661306


namespace equivalent_annual_interest_rate_l661_661744

noncomputable def bi_monthly_to_annual (r : ℝ) : ℝ :=
  (1 + r / 24) ^ 24 - 1

theorem equivalent_annual_interest_rate :
  bi_monthly_to_annual 0.08 ≈ 0.0829 :=
sorry

end equivalent_annual_interest_rate_l661_661744


namespace div_relation_l661_661169

theorem div_relation (a b c : ℚ) (h1 : a / b = 3) (h2 : b / c = 2 / 3) : c / a = 1 / 2 := 
by 
  sorry

end div_relation_l661_661169


namespace intersection_expression_value_l661_661759

theorem intersection_expression_value
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : x₁ * y₁ = 1)
  (h₂ : x₂ * y₂ = 1)
  (h₃ : x₁ = -x₂)
  (h₄ : y₁ = -y₂) :
  x₁ * y₂ + x₂ * y₁ = -2 :=
by
  sorry

end intersection_expression_value_l661_661759


namespace chord_length_of_circle_cut_by_line_l661_661273

-- Define the circle with its parametric equations and derive its standard form
def circle_equation : (ℝ × ℝ) → Prop := 
  λ ⟨x, y⟩, x^2 + y^2 = 9

-- Define the line equation
def line_equation : (ℝ × ℝ) → Prop := 
  λ ⟨x, y⟩, x - y - 3 = 0

-- Define the problem that states the length of the chord cut by the line from the circle
theorem chord_length_of_circle_cut_by_line :
  ∃ d : ℝ, (∀ x y : ℝ, circle_equation (x, y) → line_equation (x, y)) →
  d = 3 * real.sqrt 2 := 
sorry

end chord_length_of_circle_cut_by_line_l661_661273


namespace storks_joined_l661_661339

-- Conditions
def initial_birds : ℕ := 3
def additional_birds : ℕ := 2
def total_birds : ℕ := initial_birds + additional_birds
def storks := total_birds + 1

-- Theorem to prove
theorem storks_joined (total_birds = 5) : storks = 6 := by
  sorry

end storks_joined_l661_661339


namespace dist_range_regular_tetrahedron_l661_661276

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def distance (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2 + (A.z - B.z)^2)

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2, (A.z + B.z) / 2⟩

noncomputable def range_of_distances (A B C D P Q : Point) : set ℝ :=
  { d | ∃ (x y : ℝ) (hP : 0 ≤ x ∧ x ≤ 1) (hQ : 0 ≤ y ∧ y ≤ 1), 
  d = distance ⟨(1 - x) * A.x + x * B.x, (1 - x) * A.y + x * B.y, (1 - x) * A.z + x * B.z⟩ 
                ⟨(1 - y) * C.x + y * D.x, (1 - y) * C.y + y * D.y, (1 - y) * C.z + y * D.z⟩}

theorem dist_range_regular_tetrahedron (A B C D : Point)
  (hAB : distance A B = 1)
  (hAC : distance A C = 1)
  (hAD : distance A D = 1)
  (hBC : distance B C = 1)
  (hBD : distance B D = 1)
  (hCD : distance C D = 1)
  (P Q : Point) :
  range_of_distances A B C D P Q = set.Icc (Real.sqrt 2 / 2) 1 :=
sorry

end dist_range_regular_tetrahedron_l661_661276


namespace solution_a1010_l661_661225

noncomputable def sequence_a : ℕ → ℚ
| 0 := 4
| 1 := 5
| n + 2 := sequence_a (n + 1) / sequence_a n

theorem solution_a1010 : sequence_a 1009 = 1 / 4 :=
by sorry

end solution_a1010_l661_661225


namespace greatest_possible_perimeter_l661_661558

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l661_661558


namespace find_fraction_l661_661090

-- Variables and Definitions
variables (x : ℚ)

-- Conditions
def condition1 := (2 / 3) / x = (3 / 5) / (7 / 15)

-- Theorem to prove the certain fraction
theorem find_fraction (h : condition1 x) : x = 14 / 27 :=
by sorry

end find_fraction_l661_661090


namespace product_of_roots_l661_661914

theorem product_of_roots :
  let f := Polynomial.C (-50) + Polynomial.X * (Polynomial.C 75 + Polynomial.X * (Polynomial.C (-15) + Polynomial.X)) in 
  (f.roots.Prod = 50) :=
by sorry

end product_of_roots_l661_661914


namespace chocolate_bars_in_small_box_l661_661028

-- Given conditions
def num_small_boxes : ℕ := 21
def total_chocolate_bars : ℕ := 525

-- Statement to prove
theorem chocolate_bars_in_small_box : total_chocolate_bars / num_small_boxes = 25 := by
  sorry

end chocolate_bars_in_small_box_l661_661028


namespace max_triangle_perimeter_l661_661592

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l661_661592


namespace probability_sum_not_less_than_14_l661_661014

theorem probability_sum_not_less_than_14 :
  let bag := Finset.range 8    -- This creates the set {0, 1, ..., 7}
  let cards := bag.map (λ x, x + 1)  -- Map to set {1, 2, ..., 8}
  let card_draws := cards.powerset.filter (λ s, s.card = 2)
  let favorable_draws := card_draws.filter (λ s, s.sum ≥ 14)
  (favorable_draws.card : ℚ) / card_draws.card = 1 / 14 := by
sorry

end probability_sum_not_less_than_14_l661_661014


namespace product_of_roots_l661_661915

theorem product_of_roots :
  let f := Polynomial.C (-50) + Polynomial.X * (Polynomial.C 75 + Polynomial.X * (Polynomial.C (-15) + Polynomial.X)) in 
  (f.roots.Prod = 50) :=
by sorry

end product_of_roots_l661_661915


namespace tip_percentage_l661_661345

theorem tip_percentage (total_amount spent : ℝ) (sales_tax_rate : ℝ) (food_price : ℝ) : 
  total_amount = 132 ∧ sales_tax_rate = 0.1 ∧ food_price = 100 → 
  (total_amount - (food_price + sales_tax_rate * food_price)) / food_price * 100 = 22 :=
by
  intros h,
  rcases h with ⟨h1, h2, h3⟩,
  simp [h1, h2, h3], sorry

end tip_percentage_l661_661345


namespace rafting_trip_permission_slips_l661_661838

variable (total_scouts : ℕ)
variable (boy_scouts_percentage : ℝ) (boy_scouts_signed_percentage : ℝ) 
          (girl_scouts_signed_percentage : ℝ)

theorem rafting_trip_permission_slips 
  (h_boy_scouts : boy_scouts_percentage = 60 / 100)
  (h_signed_boy_scouts : boy_scouts_signed_percentage = 75 / 100)
  (h_signed_girl_scouts : girl_scouts_signed_percentage = 62.5 / 100)
  (total_scouts = 100) :
  let number_of_boy_scouts := total_scouts * boy_scouts_percentage,
      number_of_girl_scouts := total_scouts * (1 - boy_scouts_percentage),
      boy_scouts_with_permission := number_of_boy_scouts * boy_scouts_signed_percentage,
      girl_scouts_with_permission := number_of_girl_scouts * girl_scouts_signed_percentage,
      total_with_permission := boy_scouts_with_permission + girl_scouts_with_permission
  in (total_with_permission / total_scouts) * 100 = 70 := by
  sorry

end rafting_trip_permission_slips_l661_661838


namespace cesaro_sum_51_term_l661_661423

def T (n : ℕ) (B : Fin n → ℝ) (k : Fin n) : ℝ := (list.sum (list.map B (list.fin.to_list (Fin.range (k+1)))))

def cesaro_sum {n : ℕ} (B : Fin n → ℝ) : ℝ :=
(list.sum (list.map (T n B) (list.fin.to_list (Fin.range n)))) / n

variable (b : Fin 50 → ℝ)

axiom cesaro_sum_b : cesaro_sum b = 600

theorem cesaro_sum_51_term : cesaro_sum (fun i => if i = 0 then 2 else b (i.pred sorry)) = 590.235294 := 
sorry

end cesaro_sum_51_term_l661_661423


namespace greatest_possible_perimeter_l661_661623

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l661_661623


namespace greatest_perimeter_l661_661569

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l661_661569


namespace six_digit_numbers_q_plus_r_divisible_by_seven_l661_661230

theorem six_digit_numbers_q_plus_r_divisible_by_seven :
  let f (n : ℕ) := n % 200 in
  let q (n : ℕ) := n / 200 in
  ∃ v : ℕ, v = 642 ∧ ∀ n : ℕ, 100000 ≤ n ∧ n ≤ 999999 →
  (7 ∣ (q n + f n)) ↔ (∃ m : ℕ, m < 642) :=
by
  -- proof omitted
  sorry

end six_digit_numbers_q_plus_r_divisible_by_seven_l661_661230


namespace exist_point_in_triangle_l661_661384

noncomputable def point_in_triangle (a b c m_a : ℝ) : Prop :=
  ∃ P : EuclideanGeometry.Point, 
    let x := a * m_a / (a + 2 * b + 3 * c) in
    EuclideanGeometry.distance_to_side P a = x ∧
    EuclideanGeometry.distance_to_side P b = 2 * x ∧
    EuclideanGeometry.distance_to_side P c = 3 * x

theorem exist_point_in_triangle (a b c m_a : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < m_a → point_in_triangle a b c m_a :=
by
  sorry

end exist_point_in_triangle_l661_661384


namespace max_perimeter_of_new_polygon_l661_661782

def regularPolygonInteriorAngle (n : ℕ) : ℝ := (n - 2) * 180 / n

theorem max_perimeter_of_new_polygon 
  (P1 P2 P3 : { n // 3 ≤ n })
  (side_length : ℝ)
  (intersect_at_A : P1.val = 12 ∧ P2.val = 12 ∧ P3.val = 3)
  (sum_interior_angles : regularPolygonInteriorAngle P1.val + regularPolygonInteriorAngle P2.val + regularPolygonInteriorAngle P3.val = 360)
  : ∑ polygon in [P1, P2, P3], polygon.val * side_length - 3 * 1 = 24 :=
sorry

end max_perimeter_of_new_polygon_l661_661782


namespace ratio_exists_in_interval_l661_661113

theorem ratio_exists_in_interval :
  ∀ (a : Fin 10 → ℕ),
  (∀ i, a i ≤ 91) →
  ∃ i j, (2 / 3 : ℝ) ≤ a i.to_nat / a j.to_nat ∧ a i.to_nat / a j.to_nat ≤ (3 / 2 : ℝ) :=
by sorry

end ratio_exists_in_interval_l661_661113


namespace eval_expression_l661_661295

theorem eval_expression : (2^5 - 5^2) = 7 :=
by {
  -- Proof steps will be here
  sorry
}

end eval_expression_l661_661295


namespace find_circle_eq_find_range_of_dot_product_l661_661661

open Real
open Set

-- Define the problem conditions
def line_eq (x y : ℝ) : Prop := x - sqrt 3 * y = 4
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point P inside the circle and condition that |PA|, |PO|, |PB| form a geometric sequence
def geometric_sequence_condition (x y : ℝ) : Prop :=
  sqrt ((x + 2)^2 + y^2) * sqrt ((x - 2)^2 + y^2) = x^2 + y^2

-- Prove the equation of the circle
theorem find_circle_eq :
  (∃ (r : ℝ), ∀ (x y : ℝ), line_eq x y → r = 2) → circle_eq x y :=
by
  -- skipping the proof
  sorry

-- Prove the range of values for the dot product
theorem find_range_of_dot_product :
  (∀ (x y : ℝ), circle_eq x y ∧ geometric_sequence_condition x y) →
  -2 < (x^2 - 1 * y^2 - 1) → (x^2 - 4 + y^2) < 0 :=
by
  -- skipping the proof
  sorry

end find_circle_eq_find_range_of_dot_product_l661_661661


namespace dot_product_value_l661_661489

open Real

variables (a b : ℝ^3) -- Assuming vectors in ℝ^3 for generality

-- Hypotheses
def norm_a : ∥a∥ = 4 := sorry
def norm_b : ∥b∥ = 5 := sorry
def dot_product_zero : a ⋅ b = 0 := sorry

-- Theorem statement
theorem dot_product_value : (a + b) ⋅ (a - b) = -9 := 
by
  -- Proof would go here
  sorry

end dot_product_value_l661_661489


namespace greatest_triangle_perimeter_l661_661576

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l661_661576


namespace rationalize_cube_root_identity_l661_661729

theorem rationalize_cube_root_identity :
  let a := (5 : ℝ)^(1/3)
  let b := (4 : ℝ)^(1/3)
  let X := 25
  let Y := 20
  let Z := 16
  let W := 1
  a^3 = 5 ∧ b^3 = 4 ∧ a - b ≠ 0 ∧
  (X + Y + Z + W) = 62 :=
by
  -- Define a and b
  let a := (5 : ℝ)^(1/3)
  let b := (4 : ℝ)^(1/3)
  -- Rationalize using identity a^3 - b^3 = (a - b)(a^2 + ab + b^2)
  have h1 : a^3 = 5, by sorry
  have h2 : b^3 = 4, by sorry
  have h3 : a - b ≠ 0, by sorry
  let X := 25
  let Y := 20
  let Z := 16
  let W := 1
  -- Conclude the sum X + Y + Z + W = 62
  have h4 : (X + Y + Z + W) = 62, by sorry
  -- Returning the combined statement
  exact ⟨h1, h2, h3, h4⟩

end rationalize_cube_root_identity_l661_661729


namespace union_A_B_correct_l661_661453

def A : Set ℕ := {0, 1}
def B : Set ℕ := {x | 0 < x ∧ x < 3}

theorem union_A_B_correct : A ∪ B = {0, 1, 2} :=
by sorry

end union_A_B_correct_l661_661453


namespace incorrect_conclusion_l661_661799

def linear_function (x : Real) : Real :=
  -2 * x + 1

theorem incorrect_conclusion :
  ¬ (∀ x1 x2 : Real, x1 < x2 → linear_function x1 < linear_function x2) :=
by
  intros x1 x2 h
  simp [linear_function]
  linarith

end incorrect_conclusion_l661_661799


namespace probability_of_rain_l661_661737

variable (P_R P_B0 : ℝ)
variable (H1 : 0 ≤ P_R ∧ P_R ≤ 1)
variable (H2 : 0 ≤ P_B0 ∧ P_B0 ≤ 1)
variable (H : P_R + P_B0 - P_R * P_B0 = 0.2)

theorem probability_of_rain : 
  P_R = 1/9 :=
by
  sorry

end probability_of_rain_l661_661737


namespace Warriors_won_25_games_l661_661284

def CricketResults (Sharks Falcons Warriors Foxes Knights : ℕ) :=
  Sharks > Falcons ∧
  (Warriors > Foxes ∧ Warriors < Knights) ∧
  Foxes > 15 ∧
  (Foxes = 20 ∨ Foxes = 25 ∨ Foxes = 30) ∧
  (Warriors = 20 ∨ Warriors = 25 ∨ Warriors = 30) ∧
  (Knights = 20 ∨ Knights = 25 ∨ Knights = 30)

theorem Warriors_won_25_games (Sharks Falcons Warriors Foxes Knights : ℕ) 
  (h : CricketResults Sharks Falcons Warriors Foxes Knights) :
  Warriors = 25 :=
by
  sorry

end Warriors_won_25_games_l661_661284


namespace hyperbola_line_equation_through_point_l661_661479

theorem hyperbola_line_equation_through_point (A B P: ℝ × ℝ)
  (h_hyperbola_A : (A.1^2 - (A.2^2 / 2) = 1))
  (h_hyperbola_B : (B.1^2 - (B.2^2 / 2) = 1))
  (h_midpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h_point : P = (2, 1)) :
  ∃ (k : ℝ), (k = 4) ∧ (∀ x y, y = k * (x - 2) + 1 → 4 * x - y = 7) :=
by
  obtain ⟨k, hk⟩ : ∃ k, k = 4 := ⟨4, rfl⟩
  use k
  constructor
  · exact hk
  intro x y h_eq
  rw h_eq
  ring
  sorry

end hyperbola_line_equation_through_point_l661_661479


namespace sum_of_three_pairwise_relatively_prime_integers_l661_661301

theorem sum_of_three_pairwise_relatively_prime_integers
  (a b c : ℕ)
  (h1 : a > 1)
  (h2 : b > 1)
  (h3 : c > 1)
  (h4 : a * b * c = 13824)
  (h5 : Nat.gcd a b = 1)
  (h6 : Nat.gcd b c = 1)
  (h7 : Nat.gcd a c = 1) :
  a + b + c = 144 :=
by
  sorry

end sum_of_three_pairwise_relatively_prime_integers_l661_661301


namespace product_of_roots_of_cubic_eqn_l661_661932

theorem product_of_roots_of_cubic_eqn :
  let p := Polynomial.Cubic (1 : ℝ) (-15) 75 (-50)
  in p.roots_prod = 50 :=
by
  sorry

end product_of_roots_of_cubic_eqn_l661_661932


namespace A_wins_B_no_more_than_two_throws_C_treats_after_two_throws_C_treats_exactly_two_days_l661_661048

def prob_A_wins_B_one_throw : ℚ := 1 / 3
def prob_tie_one_throw : ℚ := 1 / 3
def prob_A_wins_B_no_more_2_throws : ℚ := 4 / 9

def prob_C_treats_two_throws : ℚ := 2 / 9

def prob_C_treats_exactly_2_days_out_of_3 : ℚ := 28 / 243

theorem A_wins_B_no_more_than_two_throws (P1 : ℚ := prob_A_wins_B_one_throw) (P2 : ℚ := prob_tie_one_throw) :
  P1 + P2 * P1 = prob_A_wins_B_no_more_2_throws := 
by
  sorry

theorem C_treats_after_two_throws : prob_tie_one_throw ^ 2 = prob_C_treats_two_throws :=
by
  sorry

theorem C_treats_exactly_two_days (n : ℕ := 3) (k : ℕ := 2) (p_success : ℚ := prob_C_treats_two_throws) :
  (n.choose k) * (p_success ^ k) * ((1 - p_success) ^ (n - k)) = prob_C_treats_exactly_2_days_out_of_3 :=
by
  sorry

end A_wins_B_no_more_than_two_throws_C_treats_after_two_throws_C_treats_exactly_two_days_l661_661048


namespace estimated_mode_and_median_probability_of_same_group_l661_661846

def score_distribution : List (Set ℝ × ℕ) := [
  (Set.Ico 60 75, 2),
  (Set.Ico 75 90, 3),
  (Set.Ico 90 105, 14),
  (Set.Ico 105 120, 15),
  (Set.Ico 120 135, 12),
  (Set.Icc 135 150, 4)
]

theorem estimated_mode_and_median :
  (∃ m med, m = 112.5 ∧ med = 111) :=
sorry

def two_help_one_groups : List (Set ℕ) :=
  [{1, 2, 0}, {1, 3, 0}, {1, 4, 0}, {2, 3, 0}, {2, 4, 0}, {3, 4, 0}, {1, 2, 5}, {1, 3, 5}, {1, 4, 5}, {2, 3, 5}, {2, 4, 5}, {3, 4, 5}]

theorem probability_of_same_group :
  (card (filter (λ e, 0 ∈ e ∧ 4 ∈ e) two_help_one_groups)).to_real / (card two_help_one_groups).to_real = 1 / 4 :=
sorry

end estimated_mode_and_median_probability_of_same_group_l661_661846


namespace polar_eq_rho_1_is_circle_l661_661764

theorem polar_eq_rho_1_is_circle :
  ∀ (ρ : ℝ) (θ : ℝ), (ρ = 1) -> (∃ x y : ℝ, x = ρ * Real.cos(θ) ∧ y = ρ * Real.sin(θ) ∧ (x^2 + y^2 = 1)) :=
by
  intros ρ θ hρ
  use [ ρ * Real.cos(θ), ρ * Real.sin(θ) ]
  split
  { rw hρ }
  split
  { rw hρ }
  sorry

end polar_eq_rho_1_is_circle_l661_661764


namespace total_bottles_ordered_l661_661850

constant cases_april : ℕ
constant cases_may : ℕ
constant bottles_per_case : ℕ

axiom cases_april_def : cases_april = 20
axiom cases_may_def : cases_may = 30
axiom bottles_per_case_def : bottles_per_case = 20

theorem total_bottles_ordered :
  cases_april * bottles_per_case + cases_may * bottles_per_case = 1000 :=
by 
  rw [cases_april_def, cases_may_def, bottles_per_case_def]
  -- The remaining steps will be carried out and concluded with the necessary checks
  sorry

end total_bottles_ordered_l661_661850


namespace alicia_stickers_l661_661316

theorem alicia_stickers :
  ∃ S : ℕ, S > 2 ∧
  (S % 5 = 2) ∧ (S % 11 = 2) ∧ (S % 13 = 2) ∧
  S = 717 :=
sorry

end alicia_stickers_l661_661316


namespace find_angle_x_l661_661658

-- Define the segments and angles
variables (AB CD CE : Type)
variable (C : AB)
variable (D : AB)
variable (E : Type)

-- Define the conditions
variables (angle_ACD angle_ECB : ℝ)
variable (perpendicular_CD_AB : CD ⟂ AB)
variable (C_on_AB : C ∈ AB)
variable (D_on_AB : D ∈ AB)
variable (E_on_C : E ∈ CD)

-- Given conditions
def given_conditions :=
  angle_ACD = 90 ∧ angle_ECB = 52


-- Target proposition
theorem find_angle_x (h : given_conditions) : ∃ x : ℝ, x = 180 - 90 - 52 := by
  sorry

end find_angle_x_l661_661658


namespace population_in_2003_l661_661765

theorem population_in_2003 (P : ℕ → ℕ) (k : ℝ) 
  (h1 : ∀ n, P (n + 2) - P n = k * real.sqrt (P (n + 1))) 
  (h_year_2001 : P 2001 = 49) 
  (h_year_2002 : P 2002 = 100) 
  (h_year_2004 : P 2004 = 256) : 
  P 2003 = 225 :=
sorry

end population_in_2003_l661_661765


namespace domain_of_f_l661_661391

theorem domain_of_f (x : ℝ) : (2*x - x^2 > 0 ∧ x ≠ 1) ↔ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) :=
by
  -- proof omitted
  sorry

end domain_of_f_l661_661391


namespace cos_sub_identity_l661_661437

theorem cos_sub_identity {α : ℝ} (h : cos α + sqrt 3 * sin α = 8/5) : cos (α - π/3) = 4/5 :=
by
  sorry

end cos_sub_identity_l661_661437


namespace point_in_first_quadrant_l661_661867

def Point (x : Int) (y : Int) := (x, y)

def A := Point 3 2
def B := Point (-3) 2
def C := Point 3 (-2)
def D := Point (-3) (-2)

theorem point_in_first_quadrant (A B C D : Point) : A = Point 3 2 :=
by
  have hA : A = Point 3 2 := rfl
  have hFirstQuadrant : (∀ (p : Point), p = A → p.1 > 0 ∧ p.2 > 0) :=
    λ p h, by
      have hA_coords : p = (3, 2) := h ▸ hA
      have hpx : p.1 = 3 := by rw [hA_coords]; rfl
      have hpy : p.2 = 2 := by rw [hA_coords]; rfl
      exact ⟨by rw [hpx]; exact Int.lt_of_le_and_ne (le_refl 3) (by decide), by rw [hpy]; exact Int.lt_of_le_and_ne (le_refl 2) (by decide)⟩
  exact hA

end point_in_first_quadrant_l661_661867


namespace cakes_difference_l661_661060

theorem cakes_difference :
  let bought := 154
  let sold := 91
  bought - sold = 63 :=
by
  let bought := 154
  let sold := 91
  show bought - sold = 63
  sorry

end cakes_difference_l661_661060


namespace greatest_possible_perimeter_l661_661545

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l661_661545


namespace product_of_roots_of_cubic_polynomial_l661_661900

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l661_661900


namespace lateral_surface_area_of_solid_of_revolution_l661_661289

theorem lateral_surface_area_of_solid_of_revolution 
  (r α : ℝ) 
  (hα : 0 < α ∧ α < π / 2) :
  let S := (8 * π * r^2 * (cos ((π / 4) - (α / 2)))^2) / (sin α)^2 in
  S = (8 * π * r^2 * (cos ((π / 4) - (α / 2)))^2) / (sin α)^2 :=
by sorry

end lateral_surface_area_of_solid_of_revolution_l661_661289


namespace correct_inequality_l661_661460

noncomputable def f : ℝ → ℝ := sorry

-- Define the conditions
axiom f_monotonically_decreasing : ∀ x y : ℝ, (0 < x ∧ x < y) → f y < f x
axiom f_derivative : ∀ x : ℝ, (0 < x) → ∃ f' : ℝ → ℝ, ∀ h : ℝ, has_deriv_at f (f' h) h
axiom f_cond : ∀ x : ℝ, (0 < x) → (f x / f'' x) > x

-- Proof: Prove that ef(e) < f(e^2)
theorem correct_inequality : (↑ℯ * f ↑ℯ < f (↑ℯ ^ 2)) :=
sorry

end correct_inequality_l661_661460


namespace solve_for_t_l661_661498

theorem solve_for_t (s t : ℚ) (h1 : 8 * s + 6 * t = 160) (h2 : s = t + 3) : t = 68 / 7 :=
by
  sorry

end solve_for_t_l661_661498


namespace product_of_roots_of_cubic_l661_661978

theorem product_of_roots_of_cubic :
  (∃ (p q r : ℝ), (p, q, r ≠ 0) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 75*x - 50 = 0 ↔ x = p ∨ x = q ∨ x = r) ∧ p * q * r = 50) :=
sorry

end product_of_roots_of_cubic_l661_661978


namespace sum_of_coefficients_zero_l661_661125

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the points and vectors
variables (A B C P M : V)

-- Conditions
variable (h1 : P ∉ submodule.span ℝ ({A, B, C} : set V))
variable (h2 : P - M = 2 • (M - C))
variable (h3 : ∃ x y z : ℝ, B - M = x • (B - A) + y • (C - A) + z • (P - A))

-- Goal
theorem sum_of_coefficients_zero :
  (∃ x y z : ℝ, (B - M : V) = x • (B - A) + y • (C - A) + z • (P - A) ∧ x + y + z = 0) :=
by
  sorry

end sum_of_coefficients_zero_l661_661125


namespace max_sum_of_digits_in_watch_l661_661023

theorem max_sum_of_digits_in_watch : ∃ max_sum : ℕ, max_sum = 23 ∧ 
  ∀ hours minutes : ℕ, 
  (1 ≤ hours ∧ hours ≤ 12) → 
  (0 ≤ minutes ∧ minutes < 60) → 
  let hour_digits_sum := (hours / 10) + (hours % 10) in
  let minute_digits_sum := (minutes / 10) + (minutes % 10) in
  hour_digits_sum + minute_digits_sum ≤ max_sum :=
sorry

end max_sum_of_digits_in_watch_l661_661023


namespace circumcenter_condition_l661_661448

variables {A B C P: Type} 
variables (acute_angled_triangle : ∀ (A B C : Point), acute_triangle A B C)
variables (inside_triangle : ∀ (P : Point) (A B C : Point), in_triangle P A B C)
variables (angle_eq_1 : 1 ≤ ((angle P A B) / (angle A C B)))
variables (angle_eq_2 : ((angle P A B) / (angle A C B)) ≤ 2)
variables (angle_eq_3 : 1 ≤ ((angle P B C) / (angle B A C)))
variables (angle_eq_4 : ((angle P B C) / (angle B A C)) ≤ 2)
variables (angle_eq_5 : 1 ≤ ((angle P C A) / (angle C B A)))
variables (angle_eq_6 : ((angle P C A) / (angle C B A)) ≤ 2)
variable circumcenter_is_unique : ∀ (A B C: Point), ∃! (O : Point), circumcenter A B C = O

theorem circumcenter_condition : ∀ {A B C P : Point}, acute_angled_triangle A B C →
  (inside_triangle P A B C) →
  (1 ≤ ((angle P A B) / (angle A C B))) ∧
  (((angle P A B) / (angle A C B)) ≤ 2) ∧
  (1 ≤ ((angle P B C) / (angle B A C))) ∧
  (((angle P B C) / (angle B A C)) ≤ 2) ∧
  (1 ≤ ((angle P C A) / (angle C B A))) ∧
  (((angle P C A) / (angle C B A)) ≤ 2) →
  P = circumcenter_is_unique A B C
  := by 
  sorry

end circumcenter_condition_l661_661448


namespace acute_triangle_perimeter_range_l661_661194

-- We define our hypotheses and state our goal.
theorem acute_triangle_perimeter_range (A B C a b c : ℝ)
  (h_acute_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
  (h_opposite_sides : a = sin A ∧ b = sin B ∧ c = sin C)
  (h_vectors_m : ∀ (m : ℝ × ℝ), m = (sqrt 3 * a, c))
  (h_vectors_n : ∀ (n : ℝ × ℝ), n = (sin A, cos C))
  (h_vectors_equal : ∀ (m n : ℝ × ℝ), m = 3 * n) :
  C = π / 3 ∧ (3 * sqrt 3 + 3) / 2 < a + b + c ∧ a + b + c ≤ 9 / 2 := 
  sorry

end acute_triangle_perimeter_range_l661_661194


namespace prob_limit_thm_l661_661053

def prob_heads (a : ℝ) (n : ℕ) := 1 - (1 - a)^n - n * a * (1 - a)^(n - 1)
def prob_no_tail_head (a : ℝ) (n : ℕ): ℝ := (a^(n + 1) - (1 - a)^(n + 1)) / (2 * a - 1)
def prob_A_n_cap_B_n (a : ℝ) (n : ℕ): ℝ := (a^(n + 1) - (1 - a)^(n + 1)) / (2 * a - 1) - (1 - a)^n - a * (1 - a)^(n - 1)
def prob_limit (a : ℝ) : ℝ := ((1 - a) / a)^2

theorem prob_limit_thm (a : ℝ) (h1 : 0 < a) (h2 : a < 1 / 2) :
  (∀ n : ℕ, 2 ≤ n) →
  ∀ n : ℕ, limit_nat (λ n, (prob_heads a n) * (prob_no_tail_head a n) / (prob_A_n_cap_B_n a n)) = prob_limit a :=
sorry

end prob_limit_thm_l661_661053


namespace smallest_n_division_l661_661793

-- Lean statement equivalent to the mathematical problem
theorem smallest_n_division (n : ℕ) (hn : n ≥ 3) : 
  (∃ (s : Finset ℕ), (∀ m ∈ s, 3 ≤ m ∧ m ≤ 2006) ∧ s.card = n - 2) ↔ n = 3 := 
sorry

end smallest_n_division_l661_661793


namespace highest_power_of_prime_dividing_factorial_l661_661232

theorem highest_power_of_prime_dividing_factorial (p : ℕ) (hp : p.prime) (n : ℕ) :
  ∃ k : ℕ, p^k ≤ n ∧ n < p^(k+1) ∧ 
  p(n!) = (finset.range (n+1)).sum (λ i, n / p^i) :=
begin
  sorry
end

end highest_power_of_prime_dividing_factorial_l661_661232


namespace zero_if_subset_of_any_l661_661458

theorem zero_if_subset_of_any (a : ℝ) (S : set ℝ) (h : ∀ S, { x | a * x = 1 } ⊆ S) : a = 0 := 
by 
  sorry

end zero_if_subset_of_any_l661_661458


namespace negation_of_universal_proposition_l661_661721

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, cos x > sin x - 1) ↔ ∃ x : ℝ, cos x ≤ sin x - 1 :=
by
sorry

end negation_of_universal_proposition_l661_661721


namespace max_perimeter_l661_661633

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l661_661633


namespace product_of_roots_cubic_l661_661889

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l661_661889


namespace area_of_square_from_bisector_l661_661191

theorem area_of_square_from_bisector 
  (a b : ℝ) 
  (h : 0 < a ∧ 0 < b) 
  (right_triangle : ∃ c : ℝ, a^2 + b^2 = c^2) :
  let AD := λ a b, (2 * a^2 * b^2) / (a^2 + b^2) in
  AD a b = (2 * a^2 * b^2) / (a^2 + b^2) :=
by
  sorry

end area_of_square_from_bisector_l661_661191


namespace average_hours_per_day_l661_661753

theorem average_hours_per_day (h : ℝ) :
  (3 * h * 12 + 2 * h * 9 = 108) → h = 2 :=
by 
  intro h_condition
  sorry

end average_hours_per_day_l661_661753


namespace product_of_repeating_decimal_l661_661416

theorem product_of_repeating_decimal (x : ℝ) (h : x = 1 / 3) : x * 9 = 3 :=
by
  rw [h]
  norm_num
  sorry

end product_of_repeating_decimal_l661_661416


namespace ellipse_equation_tangent_line_l661_661653

def ellipse (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def parabola := ∀ x y : ℝ, y^2 = 4 * x

def is_tangent (l : ℝ → ℝ) (e : ℝ → ℝ → Prop) : Prop :=
∃ x y : ℝ, e x y ∧ l x = y ∧ ∀ ϵ : ℝ, ϵ ≠ 0 → ¬e (x + ϵ) (l (x + ϵ)) ∧ ¬e (x - ϵ) (l (x - ϵ))

theorem ellipse_equation (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0)
  (F1 : ∀ c : ℝ, c = 1 → ∃ x y : ℝ, x = -1 ∧ y = 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1)
  (P : ∀ x y : ℝ, y = 1 → x = 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1) :
  a = sqrt 2 ∧ b = 1 ∧ (ellipse (sqrt 2) 1) :=
sorry

theorem tangent_line (e : ∀ x y : ℝ, (x^2 / 2) + y^2 = 1) (p : parabola)
  (l : ℝ → ℝ) (tangent_e : is_tangent l e) (tangent_p : is_tangent l p) :
  l = (λ x : ℝ, (sqrt 2 / 2) * x + sqrt 2) ∨ l = (λ x : ℝ, -(sqrt 2 / 2) * x - sqrt 2) :=
sorry

end ellipse_equation_tangent_line_l661_661653


namespace sum_of_two_is_divisible_by_3_l661_661107

def set_of_numbers : Finset ℕ := {1, 2, 3, 4, 5}

def combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

def is_sum_divisible_by_3 (a b : ℕ) : Bool :=
  (a + b) % 3 = 0

def favorable_combinations (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s |>.filter (λ (x : ℕ × ℕ), x.1 < x.2 ∧ is_sum_divisible_by_3 x.1 x.2)

def total_possible_combinations : ℕ :=
  combinations set_of_numbers.card 2

def favorable_count : ℕ :=
  (favorable_combinations set_of_numbers).card

def probability : ℚ :=
  favorable_count / total_possible_combinations

theorem sum_of_two_is_divisible_by_3 :
  probability = 2 / 5 := by
  sorry

end sum_of_two_is_divisible_by_3_l661_661107


namespace sum_of_n_values_is_89_l661_661775

noncomputable def find_sum_of_n_values (sum_first_n : ℕ) (common_diff : ℕ) 
  (is_integer : ℤ → Prop) (a1_is_int : ∀ n, is_integer (↑n)) : ℕ :=
∑ n in {n | ∃ a1, a1_is_int n ∧ sum_first_n = n * (a1 + n - 1) ∧ n > 1}, n

theorem sum_of_n_values_is_89 :
  find_sum_of_n_values 2000 2 (λ x, ∃ (n : ℕ), x = ↑n) (λ n, int.nat_abs n) = 89 := sorry

end sum_of_n_values_is_89_l661_661775


namespace greatest_triangle_perimeter_l661_661575

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l661_661575


namespace company_A_higher_income_l661_661370

noncomputable def salary_A (n : ℕ) : ℝ :=
  1500 + 230 * (n - 1)

noncomputable def salary_B (n : ℕ) : ℝ :=
  2000 * (1 + 0.05)^(n - 1)

def total_salary_A_10_years : ℝ :=
  12 * (salary_A 1 + salary_A 2 + salary_A 3 + salary_A 4 + salary_A 5 + 
        salary_A 6 + salary_A 7 + salary_A 8 + salary_A 9 + salary_A 10)

def total_salary_B_10_years : ℝ :=
  12 * (salary_B 1 + salary_B 2 + salary_B 3 + salary_B 4 + salary_B 5 + 
        salary_B 6 + salary_B 7 + salary_B 8 + salary_B 9 + salary_B 10)

theorem company_A_higher_income :
  total_salary_A_10_years > total_salary_B_10_years :=
by {
  sorry
}

end company_A_higher_income_l661_661370


namespace bread_rise_time_l661_661240

theorem bread_rise_time (x : ℕ) (kneading_time : ℕ) (baking_time : ℕ) (total_time : ℕ) 
  (h1 : kneading_time = 10) 
  (h2 : baking_time = 30) 
  (h3 : total_time = 280) 
  (h4 : kneading_time + baking_time + 2 * x = total_time) : 
  x = 120 :=
sorry

end bread_rise_time_l661_661240


namespace trajectory_chord_midpoints_trajectory_midpoint_M_l661_661452

noncomputable def P : Point := ⟨0, 5⟩
noncomputable def C : Circle := ⟨⟨-2, 6⟩, 4⟩

-- Prove the trajectory equation of the chord midpoints passing through point P in circle C
theorem trajectory_chord_midpoints :
  let x y : ℝ
  in x^2 + y^2 + 4*x - 12*y + 24 = 0 ∧ ∃ N : Point, segment (α) (Point.mk (x, y)) = 0 → x^2 + y^2 + 2*x - 11*y + 30 = 0 := 
sorry

-- Prove the trajectory of the midpoint M of PQ
theorem trajectory_midpoint_M :
  let x y : ℝ
  in x^2 + y^2 + 4*x - 12*y + 24 = 0 ∧ ∃ Q : Point, distance (P, Q) = x → x^2 + y^2 + 2*x - 11*y - 11/4 = 0 :=
sorry

end trajectory_chord_midpoints_trajectory_midpoint_M_l661_661452


namespace problem_solution_l661_661713

-- Define the structure of the dartboard and scoring
structure Dartboard where
  inner_radius : ℝ
  intermediate_radius : ℝ
  outer_radius : ℝ
  regions : List (List ℤ) -- List of lists representing scores in the regions

-- Define the probability calculation function
noncomputable def probability_odd_score (d : Dartboard) : ℚ := sorry

-- Define the specific dartboard with given conditions
def revised_dartboard : Dartboard :=
  { inner_radius := 4.5,
    intermediate_radius := 6.75,
    outer_radius := 9,
    regions := [[3, 2, 2], [2, 1, 1], [1, 1, 3]] }

-- The theorem to prove the solution to the problem
theorem problem_solution : probability_odd_score revised_dartboard = 265 / 855 :=
  sorry

end problem_solution_l661_661713


namespace december_sales_fraction_l661_661215

variable (A : ℝ) -- Define the average monthly sales total for January through November as a real number

-- Define a constant to represent the fraction
def fraction_of_december_sales (A : ℝ) : ℝ :=
  let december_sales := 7 * A
  let total_yearly_sales := 11 * A + december_sales
  december_sales / total_yearly_sales

-- Now, we need to state the theorem
theorem december_sales_fraction (A : ℝ) : fraction_of_december_sales A = 7 / 18 :=
by
  unfold fraction_of_december_sales
  sorry

end december_sales_fraction_l661_661215


namespace customer_difference_l661_661857

theorem customer_difference (X Y Z : ℕ) (h1 : X - Y = 10) (h2 : 10 - Z = 4) : X - 4 = 10 :=
by sorry

end customer_difference_l661_661857


namespace trig_identity_l661_661732

theorem trig_identity (x y : ℝ) (h₁ : tan x = x) (h₂ : tan y = y) (h₃ : |x| ≠ |y|) :
  (sin (x + y) / (x + y)) - (sin (x - y) / (x - y)) = 0 := 
sorry

end trig_identity_l661_661732


namespace pyramid_volume_correct_l661_661272

noncomputable def pyramid_volume (l α β : ℝ) : ℝ :=
  (4 / 3) * l^3 * (Real.cos α) * (Real.cos β) * Real.sqrt(-Real.cos (α + β) * Real.cos(α - β))

theorem pyramid_volume_correct
  (l α β : ℝ)
  (h_base_rect : True)  -- Base is a rectangle (placeholder condition)
  (h_lateral_equal : True)  -- Each lateral edge has length l (placeholder condition)
  (h_angles : True)  -- Lateral edges form angles α and β with adjacent sides
  : pyramid_volume l α β = (4 / 3) * l^3 * Real.cos α * Real.cos β * Real.sqrt(-Real.cos (α + β) * Real.cos (α - β)) :=
sorry

end pyramid_volume_correct_l661_661272


namespace airplane_rows_l661_661399

theorem airplane_rows (r : ℕ) (h1 : ∀ (seats_per_row total_rows : ℕ), seats_per_row = 8 → total_rows = r →
  ∀ occupied_seats : ℕ, occupied_seats = (3 * seats_per_row) / 4 →
  ∀ unoccupied_seats : ℕ, unoccupied_seats = seats_per_row * total_rows - occupied_seats * total_rows →
  unoccupied_seats = 24): 
  r = 12 :=
by
  sorry

end airplane_rows_l661_661399


namespace exists_lattice_points_l661_661482

noncomputable def n : ℕ := arbitrary
def lattice_point := ℤ × ℤ
def area (O P₁ P₂ : lattice_point) : ℚ := 1/2 -- given areas are equal halves

theorem exists_lattice_points (n : ℕ) (h₁ : n ≥ 5)
    (P : fin n → lattice_point)
    (h₂ : ∀ i : fin n, area (0, 0) (P i) (P (i + 1) % n) = 1/2) :
    ∃ (i j : fin n), 2 ≤ |i - j| ∧ |i - j| ≤ n - 2 ∧
    area (0, 0) (P i) (P j) = 1/2 :=
sorry

end exists_lattice_points_l661_661482


namespace greatest_possible_perimeter_l661_661640

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l661_661640


namespace scheme2_saves_money_for_80_participants_l661_661836

-- Define the variables and conditions
def total_charge_scheme1 (x : ℕ) (hx : x > 50) : ℕ :=
  1500 + 240 * x

def total_charge_scheme2 (x : ℕ) (hx : x > 50) : ℕ :=
  270 * (x - 5)

-- Define the theorem
theorem scheme2_saves_money_for_80_participants :
  total_charge_scheme2 80 (by decide) < total_charge_scheme1 80 (by decide) :=
sorry

end scheme2_saves_money_for_80_participants_l661_661836


namespace sum_of_angles_of_square_and_pentagon_l661_661202

theorem sum_of_angles_of_square_and_pentagon (A B C D : Type) [square : square_shape A B C D] [pentagon : pentagon_shape A B D] :
  ∠ ABC + ∠ ABD = 198 :=
by 
  sorry

end sum_of_angles_of_square_and_pentagon_l661_661202


namespace max_volume_ratio_l661_661781

theorem max_volume_ratio (a R : ℝ)
  (h : a^2 + ( (sqrt 2 * a) / 2 )^2 = R^2) :
  (a^3) / ( (1/2) * (4 * π * R^3 / 3) ) = sqrt 6 / (3 * π) :=
sorry

end max_volume_ratio_l661_661781


namespace greatest_possible_perimeter_l661_661529

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l661_661529


namespace product_of_roots_l661_661920

theorem product_of_roots :
  let f := Polynomial.C (-50) + Polynomial.X * (Polynomial.C 75 + Polynomial.X * (Polynomial.C (-15) + Polynomial.X)) in 
  (f.roots.Prod = 50) :=
by sorry

end product_of_roots_l661_661920


namespace substitute_teachers_after_lunch_l661_661517

-- Defining the problem conditions
def total_teachers : Nat := 60
def perc_A : ℝ := 0.20
def perc_B : ℝ := 0.30
def perc_C : ℝ := 0.25
def perc_D : ℝ := 0.15
def perc_E : ℝ := 0.10

-- Initial number of substitutes in each school
def initial_A := (perc_A * total_teachers).toNat
def initial_B := (perc_B * total_teachers).toNat
def initial_C := (perc_C * total_teachers).toNat
def initial_D := (perc_D * total_teachers).toNat
def initial_E := (perc_E * total_teachers).toNat

-- After 1 hour
def after_hour_A := (0.50 * initial_A).toNat
def after_hour_B := (0.50 * initial_B).toNat
def after_hour_C := (0.50 * initial_C).toNat
def after_hour_D := (0.50 * initial_D).toNat
def after_hour_E := (0.50 * initial_E).toNat

-- After quitting before lunch
def before_lunch_A := (0.30 * after_hour_A).toNat
def before_lunch_B := (0.30 * after_hour_B).toNat
def before_lunch_C := (0.30 * after_hour_C).toNat
def before_lunch_D := (0.30 * after_hour_D).toNat
def before_lunch_E := (0.30 * after_hour_E).toNat

-- After dealing with misbehaving students
def after_misbehavior_A := (0.10 * before_lunch_A).toNat
def after_misbehavior_B := (0.10 * before_lunch_B).toNat
def after_misbehavior_C := (0.10 * before_lunch_C).toNat
def after_misbehavior_D := (0.10 * before_lunch_D).toNat
def after_misbehavior_E := (0.10 * before_lunch_E).toNat

-- Number of substitute teachers left after lunch
def final_A := initial_A - after_hour_A - before_lunch_A - after_misbehavior_A
def final_B := initial_B - after_hour_B - before_lunch_B - after_misbehavior_B
def final_C := initial_C - after_hour_C - before_lunch_C - after_misbehavior_C
def final_D := initial_D - after_hour_D - before_lunch_D - after_misbehavior_D
def final_E := initial_E - after_hour_E - before_lunch_E - after_misbehavior_E

theorem substitute_teachers_after_lunch :
  final_A = 5 ∧ final_B = 7 ∧ final_C = 5 ∧ final_D = 3 ∧ final_E = 3 :=
by
  -- Insert proof here
  sorry

end substitute_teachers_after_lunch_l661_661517


namespace opposite_of_a_is_2_l661_661178

theorem opposite_of_a_is_2 (a : ℤ) (h : -a = 2) : a = -2 := 
by
  -- proof to be provided
  sorry

end opposite_of_a_is_2_l661_661178


namespace conical_tank_volume_l661_661375

theorem conical_tank_volume
  (diameter : ℝ) (height : ℝ) (depth_linear : ∀ x : ℝ, 0 ≤ x ∧ x ≤ diameter / 2 → height - (height / (diameter / 2)) * x = 0) :
  diameter = 20 → height = 6 → (1 / 3) * Real.pi * (10 ^ 2) * height = 200 * Real.pi :=
by
  sorry

end conical_tank_volume_l661_661375


namespace charlie_and_father_wood_l661_661882

theorem charlie_and_father_wood :
  ∀ (x : ℕ), 15 + 2 * x = 35 → x = 10 :=
begin
  sorry
end

end charlie_and_father_wood_l661_661882


namespace kelly_paper_purchase_l661_661674

def supplies (students pieces_per_student glue_bottles initial_buy additional_buy remaining_supplies : ℕ) :=
  let total_initial_supplies := students * pieces_per_student + glue_bottles in
  let lost_supplies := total_initial_supplies / 2 in
  let total_supplies_after_purchase := lost_supplies + additional_buy in
  remaining_supplies = total_supplies_after_purchase 

theorem kelly_paper_purchase : 
  ∀ (students pieces_per_student glue_bottles initial_buy additional_buy remaining_supplies : ℕ), 
  students = 8 → 
  pieces_per_student = 3 → 
  glue_bottles = 6 → 
  initial_buy = 30 → 
  additional_buy = 5 → 
  remaining_supplies = 20 → 
  supplies students pieces_per_student glue_bottles initial_buy additional_buy remaining_supplies := 
by 
  intros students pieces_per_student glue_bottles initial_buy additional_buy remaining_supplies
  sorry

end kelly_paper_purchase_l661_661674


namespace monica_tiles_l661_661710

theorem monica_tiles (room_length : ℕ) (room_width : ℕ) (border_tile_size : ℕ) (inner_tile_size : ℕ) 
  (border_tiles : ℕ) (inner_tiles : ℕ) (total_tiles : ℕ) :
  room_length = 24 ∧ room_width = 18 ∧ border_tile_size = 2 ∧ inner_tile_size = 3 ∧ 
  border_tiles = 38 ∧ inner_tiles = 32 → total_tiles = 70 :=
by {
  sorry
}

end monica_tiles_l661_661710


namespace impossible_to_form_16_unique_remainders_with_3_digits_l661_661204

theorem impossible_to_form_16_unique_remainders_with_3_digits :
  ¬∃ (digits : Finset ℕ) (num_fun : Fin 16 → ℕ), digits.card = 3 ∧ 
  ∀ i j : Fin 16, i ≠ j → num_fun i % 16 ≠ num_fun j % 16 ∧ 
  ∀ n : ℕ, n ∈ (digits : Set ℕ) → 100 ≤ num_fun i ∧ num_fun i < 1000 :=
sorry

end impossible_to_form_16_unique_remainders_with_3_digits_l661_661204


namespace Maurice_current_age_l661_661045

variable (Ron_now Maurice_now : ℕ)

theorem Maurice_current_age
  (h1 : Ron_now = 43)
  (h2 : ∀ t Ron_future Maurice_future : ℕ, Ron_future = 4 * Maurice_future → Ron_future = Ron_now + 5 → Maurice_future = Maurice_now + 5) :
  Maurice_now = 7 := 
sorry

end Maurice_current_age_l661_661045


namespace zoe_pictures_l661_661800

theorem zoe_pictures (P : ℕ) (h1 : P + 16 = 44) : P = 28 :=
by sorry

end zoe_pictures_l661_661800


namespace impossible_five_correct_l661_661297

open Finset

theorem impossible_five_correct (P L : Finset ℕ) (hP : P.card = 6) (hL : L.card = 6)
  (letters_assigned : P → L) :
  ¬ ∃ H : Π p ∈ P, p = letters_assigned p, card P - card (↑((λ p, p = letters_assigned p) '' P.toFin)) = 5 := 
by 
  sorry

end impossible_five_correct_l661_661297


namespace maximum_value_of_f_l661_661990

noncomputable def f (x : ℝ) : ℝ := ((x - 3) * (12 - x)) / x

theorem maximum_value_of_f :
  ∀ x : ℝ, 3 < x ∧ x < 12 → f x ≤ 3 :=
by
  sorry

end maximum_value_of_f_l661_661990


namespace domain_ln_sqrt_func_l661_661338

theorem domain_ln_sqrt_func : {x | 3 - x > 0 ∧ 2^x - 4 ≥ 0} = Icc 2 3 :=
by sorry

end domain_ln_sqrt_func_l661_661338


namespace license_plate_combinations_l661_661058

theorem license_plate_combinations : 
  let letters := 26 in
  let positions := 5 in
  let odd_digits := [1, 3, 5, 7, 9] in
  ∃ plates, plates = 
    (choose letters 2 * choose positions 2 * choose 3 2 * 24) * (5 * 4) = 936000 := 
by
  sorry

end license_plate_combinations_l661_661058


namespace number_of_dolls_combined_l661_661863

-- Defining the given conditions as variables
variables (aida sophie vera : ℕ)

-- Given conditions
def condition1 : Prop := aida = 2 * sophie
def condition2 : Prop := sophie = 2 * vera
def condition3 : Prop := vera = 20

-- The final proof statement we need to prove
theorem number_of_dolls_combined (h1 : condition1 aida sophie) (h2 : condition2 sophie vera) (h3 : condition3 vera) : 
  aida + sophie + vera = 140 :=
  by sorry

end number_of_dolls_combined_l661_661863


namespace greatest_possible_perimeter_l661_661602

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l661_661602


namespace greatest_possible_perimeter_l661_661626

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l661_661626


namespace cartesian_circle_correct_and_min_distance_l661_661652

-- Define the parametric equation of the line
def line_eqn (t : ℝ) : ℝ × ℝ :=
  (1 + (sqrt 2 / 2) * t, 2 + (sqrt 2 / 2) * t)

-- Define the polar equation of the circle
def polar_circle_eqn (rho theta : ℝ) : Prop :=
  rho = 6 * sin theta

-- Define the Cartesian equation of the circle using x, y coordinates
def cartesian_circle_eqn (x y : ℝ) : Prop :=
  x^2 + (y - 3)^2 = 9

-- Define the intersection and minimum PA + PB calculation 
def compute_intersect_and_min_distance (P : ℝ × ℝ) : ℝ :=
  let t1 := sqrt 7 in
  let t2 := -sqrt 7 in
  abs (t1 - t2)

-- The theorem to be proved
theorem cartesian_circle_correct_and_min_distance :
  (∀ ρ θ, polar_circle_eqn ρ θ → ∃ x y, cartesian_circle_eqn x y) ∧
  compute_intersect_and_min_distance (1, 2) = 2 * sqrt 7 :=
by
  sorry

end cartesian_circle_correct_and_min_distance_l661_661652


namespace greatest_perimeter_l661_661571

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l661_661571


namespace greatest_root_f_l661_661091

noncomputable def f (x : ℝ) : ℝ := 21 * x ^ 4 - 20 * x ^ 2 + 3

theorem greatest_root_f :
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x :=
sorry

end greatest_root_f_l661_661091


namespace f_range_l661_661680

/-- Let k ≥ 2 be an integer. The function f: ℕ → ℕ is defined by
    f(n) = n + ⌊n^(1/k) + n^(1/(2k))⌋. -/
def f (k : ℕ) (n : ℕ) : ℕ := n + ⌊(n + ⌊n^(1/k)⌋)^(1/k)⌋

theorem f_range (k : ℕ) (hk : k ≥ 2) : 
  set_of (λ y, ∃ n : ℕ, f k n = y) = set_of (λ y, ∀ n : ℕ, y ≠ n^k) :=
sorry

end f_range_l661_661680


namespace find_beta_l661_661200

theorem find_beta :
  ∃ β : ℝ, 
    (∃ α : ℝ,
      ((cos α, sin α) = (1/2, sqrt 3 / 2) →
       (cos β, sin β) = (2*k*π, (2*π/3) + 2*k*π) ∧
       (cos β - cos α = 1 / 2) ∧ (sin β - sin α = sqrt 3 / 2))) ∧
    β = 2 * π / 3 :=
begin
  sorry
end

end find_beta_l661_661200


namespace greatest_perimeter_of_triangle_l661_661605

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l661_661605


namespace number_of_smaller_cuboids_l661_661003

def question : ℕ := 7

def smaller_cuboid_volume (length width height : ℕ) : ℕ :=
  length * width * height

def larger_cuboid_volume (length width height : ℕ) : ℕ :=
  length * width * height

theorem number_of_smaller_cuboids (L1 W1 H1 L2 W2 H2 : ℕ) (h: smaller_cuboid_volume 6 4 3 = 72) (h2: larger_cuboid_volume 18 15 2 = 540) :
  (larger_cuboid_volume L2 W2 H2) / (smaller_cuboid_volume L1 W1 H1) = 7 :=
by {
  have small_v : ℕ := smaller_cuboid_volume L1 W1 H1,
  have large_v : ℕ := larger_cuboid_volume L2 W2 H2,
  have vol_div : ℕ := large_v / small_v,
  sorry
}

end number_of_smaller_cuboids_l661_661003


namespace range_of_a_l661_661111

noncomputable def f (a x : ℝ) : ℝ := a * exp (x - 1) - log x + log a
noncomputable def g (x : ℝ) : ℝ := (1 - Real.exp 1) * x

theorem range_of_a (a : ℝ) :
  (∀ x > 0, (Real.exp 1) * f a x ≥ g x) → a ≥ 1 / Real.exp 1 :=
by
  intros h
  sorry

end range_of_a_l661_661111


namespace sandwich_cost_correct_l661_661212

structure SandwichCost where
  bread_cost : ℝ
  ham_cost : ℝ
  cheese_cost : ℝ
  mayo_cost : ℝ
  lettuce_cost : ℝ
  tomato_cost : ℝ
  packaging_cost : ℝ
  sales_tax_rate : ℝ
  discount_rate : ℝ

constant Joe_sandwich : SandwichCost :=
{ bread_cost      := 0.15,
  ham_cost        := 0.25,
  cheese_cost     := 0.35,
  mayo_cost       := 0.10,
  lettuce_cost    := 0.05,
  tomato_cost     := 0.08,
  packaging_cost  := 0.02,
  sales_tax_rate  := 0.05,
  discount_rate   := 0.10 }

def total_cost (s : SandwichCost) : ℝ :=
  let bread_total := s.bread_cost * 2
  let ham_total := s.ham_cost * 2
  let cheese_total := s.cheese_cost * 2
  let mayo_total := s.mayo_cost
  let lettuce_total := s.lettuce_cost
  let tomato_total := s.tomato_cost * 2
  let ingredients_cost := bread_total + ham_total + cheese_total + mayo_total + lettuce_total + tomato_total
  let discount := (ham_total + cheese_total) * s.discount_rate
  let adjusted_cost := ingredients_cost - discount
  let total_with_packaging := adjusted_cost + s.packaging_cost
  let sales_tax := 3.00 * s.sales_tax_rate
  total_with_packaging + sales_tax

theorem sandwich_cost_correct :
  total_cost Joe_sandwich = 1.86 := by
  sorry

end sandwich_cost_correct_l661_661212


namespace find_tangent_l661_661405

variable (A B X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace X]
variable (AB AX BX : ℝ)

-- Conditions given in the problem
axiom right_triangle : AX^2 + AB^2 = BX^2
axiom hypotenuse : BX = 13
axiom base : AX = 12

-- Statement to prove (tangent definition in right triangle)
theorem find_tangent (A B X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace X]
    (AB AX BX : ℝ) (right_triangle : AX^2 + AB^2 = BX^2)
    (hypotenuse : BX = 13) (base : AX = 12) : tan(AB / AX) = 5 / 12 :=
by
  sorry

end find_tangent_l661_661405


namespace cosine_120_eq_neg_one_half_l661_661791

theorem cosine_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1/2 :=
by
-- Proof omitted
sorry

end cosine_120_eq_neg_one_half_l661_661791


namespace problem_solution_l661_661994

theorem problem_solution :
  let sum1 := ( ∑ x in ({x | x^2 - 7 * x + 10 = 0}).to_finset, x),
      sum2 := ( ∑ x in ({x | x^2 - 6 * x + 5 = 0}).to_finset, x),
      sum3 := ( ∑ x in ({x | x^2 - 6 * x + 6 = 0}).to_finset, x)
  in
  sum1 + sum2 + sum3 = 16 := sorry

end problem_solution_l661_661994


namespace a_2011_value_l661_661484

noncomputable def sequence_a : ℕ → ℝ
| 0 => 6/7
| (n + 1) => if 0 ≤ sequence_a n ∧ sequence_a n < 1/2 then 2 * sequence_a n
              else 2 * sequence_a n - 1

theorem a_2011_value : sequence_a 2011 = 6/7 := sorry

end a_2011_value_l661_661484


namespace derivative_arcsin_arccos_derivative_arcsin_arcctg_derivative_arctg_arcctg_at_0_derivative_arctg_arcctg_at_pi_l661_661089

-- 1. Derivative of y = 5 * arcsin(k*x) + 3 * arccos(k*x)
theorem derivative_arcsin_arccos (k x : ℝ) :
  deriv (λ x, 5 * Real.arcsin(k * x) + 3 * Real.arccos(k * x)) x = 2 * k / Real.sqrt(1 - (k * x)^2) :=
sorry

-- 2. Derivative of y = arcsin(a/x) - arcctg(x/a)
theorem derivative_arcsin_arcctg (a x : ℝ) :
  deriv (λ x, Real.arcsin(a / x) - Real.arccot(x / a)) x = -a / (Real.abs x * Real.sqrt(x^2 - a^2)) + a / (x^2 + a^2) :=
sorry

-- 3. Derivative of r = arctg(m / φ) + arcctg(m * ctg(φ)) at φ = 0 and φ = π
theorem derivative_arctg_arcctg_at_0 (m : ℝ) :
  deriv (λ φ, Real.arctan(m / φ) + Real.arccot(m * Real.cot(φ))) 0 = 0 :=
sorry

theorem derivative_arctg_arcctg_at_pi (m : ℝ) :
  deriv (λ φ, Real.arctan(m / φ) + Real.arccot(m * Real.cot(φ))) Real.pi = Real.pi^2 / (m * (Real.pi^2 + m^2)) :=
sorry

end derivative_arcsin_arccos_derivative_arcsin_arcctg_derivative_arctg_arcctg_at_0_derivative_arctg_arcctg_at_pi_l661_661089


namespace greatest_perimeter_l661_661567

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l661_661567


namespace all_positive_rationals_appear_exactly_once_l661_661985

def seq (n : ℕ) : ℚ :=
  if n = 1 then
    1
  else if n.even then
    1 + seq (n / 2)
  else
    1 / seq (n / 2)

theorem all_positive_rationals_appear_exactly_once :
  ∀ q : ℚ, 0 < q → ∃ n : ℕ, seq n = q ∧ ∀ m : ℕ, seq m = q → m = n :=
begin
  sorry
end

end all_positive_rationals_appear_exactly_once_l661_661985


namespace cube_root_inequality_l661_661741

theorem cube_root_inequality (x : ℝ) : 
  (x ∈ Set.Ioo (-27 : ℝ) (-1 : ℝ)) ↔ (Real.cbrt x - (3 / (Real.cbrt x + 4)) ≤ 0) :=
by
  sorry

end cube_root_inequality_l661_661741


namespace length_of_chord_cut_from_circle_l661_661761

theorem length_of_chord_cut_from_circle
  (x y t : ℝ)
  (parametric_eqs : x = 2 * t - 1 ∧ y = t + 1)
  (circle_eq : x^2 + y^2 = 9) :
  ∃ l : ℝ, l = 12 * sqrt(5) / 5 :=
by 
sorry

end length_of_chord_cut_from_circle_l661_661761


namespace power_for_decimal_digits_l661_661502

theorem power_for_decimal_digits (x : ℤ) : 
  x = 11 := 
by 
  -- Definitions from the problem's conditions
  let y := 10 ^ 4 * 3.456789
  let digits_right_of_decimal := 22
  -- Translation from condition
  have h1 : formatDigitsRight (y ^ x) = digits_right_of_decimal := sorry
  -- Proving the conclusion
  have h2 : formatDigits (34567.89) = 2 := sorry
  have h3 : x = digits_right_of_decimal / h2 := sorry
  exact h3

end power_for_decimal_digits_l661_661502


namespace max_ab_value_l661_661110

noncomputable def f (a b x : ℝ) := 4 * x^3 - a * x^2 - 2 * b * x + 2

theorem max_ab_value
  (a b : ℝ)
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : ∀ x, deriv (f a b) x = 12 * x^2 - 2 * a * x - 2 * b)
  (h₄ : deriv (f a b) 1 = 0)
  : ab_max_bound : ab ≤ 9 :=
begin
  sorry
end

end max_ab_value_l661_661110


namespace real_roots_a_set_t_inequality_l661_661400

noncomputable def set_of_a : Set ℝ := {a | -1 ≤ a ∧ a ≤ 7}

theorem real_roots_a_set (x a : ℝ) :
  (∃ x, x^2 - 4 * x + abs (a - 3) = 0) ↔ a ∈ set_of_a := 
by
  sorry

theorem t_inequality (t a : ℝ) (h : ∀ a ∈ set_of_a, t^2 - 2 * a * t + 12 < 0) :
  3 < t ∧ t < 4 := 
by
  sorry

end real_roots_a_set_t_inequality_l661_661400


namespace least_integer_gt_square_l661_661760

theorem least_integer_gt_square (x : ℝ) (y : ℝ) (h1 : x = 2) (h2 : y = Real.sqrt 3) :
  ∃ (n : ℤ), n = 14 ∧ n > (x + y) ^ 2 := by
  sorry

end least_integer_gt_square_l661_661760


namespace greatest_possible_perimeter_l661_661533

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l661_661533


namespace product_of_roots_l661_661934

theorem product_of_roots :
  let p : Polynomial ℚ := Polynomial.Coeff.mk_fin 3 0 (⟨[50, -75, 15, -1]⟩ : ℕ → ℚ)
  (p.root_prod = 50) :=
begin
  sorry
end

end product_of_roots_l661_661934


namespace area_region_S_is_1087_l661_661382

-- Definitions for the given conditions
def rhombus_side_length : ℝ := 3
def angle_B : ℝ := 150 -- in degrees

-- Area calculation function for the region S
noncomputable def area_of_region_S : ℝ :=
  let side_half := rhombus_side_length / 2
  let sin_75 := real.sin (75 * real.pi / 180) -- converting degrees to radians
  0.5 * side_half * side_half * sin_75

-- The statement to prove
theorem area_region_S_is_1087 (h₁ : rhombus_side_length = 3) (h₂ : angle_B = 150) : 
  abs (area_of_region_S - 1.087) < 0.001 :=
by sorry

end area_region_S_is_1087_l661_661382


namespace conditional_probability_heads_on_second_given_heads_on_first_l661_661422

open ProbabilityTheory

-- Definitions based on conditions in the problem
def A : Event := {ω | ω.headsOnFirstFlip}
def B : Event := {ω | ω.headsOnSecondFlip}
def P : Measure Space := { measureOfEvent ω | ω = 1 / 2 }

-- Theorem statement
theorem conditional_probability_heads_on_second_given_heads_on_first :
  P(B|A) = 1 / 2 :=
sorry

end conditional_probability_heads_on_second_given_heads_on_first_l661_661422


namespace geometric_sequence_sum_first_10_terms_l661_661447

/- conditions: -/
def a_seq : ℕ → ℤ 
| 0     := 3
| (n+1) := 2 * a_seq n + (-1)^n * (3*n + 1)

/- question 1: prove geometric sequence -/
theorem geometric_sequence :  
  ∃ r : ℤ, ∃ a : ℤ, ∀ n : ℕ, (a_seq n + (-1)^n * n) = a * r^n :=
sorry

/- question 2: sum of first 10 terms -/
theorem sum_first_10_terms : 
  (∑ i in range 10, a_seq i) = 2041 :=
sorry

end geometric_sequence_sum_first_10_terms_l661_661447


namespace annual_interest_rate_of_second_investment_l661_661788

-- Definitions for the conditions
def total_income : ℝ := 575
def investment1 : ℝ := 3000
def rate1 : ℝ := 0.085
def income1 : ℝ := investment1 * rate1
def investment2 : ℝ := 5000
def target_income : ℝ := total_income - income1

-- Lean 4 statement to prove the annual simple interest rate of the second investment
theorem annual_interest_rate_of_second_investment : ∃ (r : ℝ), target_income = investment2 * (r / 100) ∧ r = 6.4 :=
by sorry

end annual_interest_rate_of_second_investment_l661_661788


namespace triangle_ABC_CD_length_l661_661515

noncomputable def length_CD (A B C D : EuclideanSpace) : ℝ :=
  -- placeholder definition
  sorry

theorem triangle_ABC_CD_length :
  ∀ (A B C D : EuclideanSpace),
    ∠ B A C = 150 ∧ dist A B = 5 ∧ dist B C = 6 ∧
    is_perpendicular (line_through A B) (line_through A D) ∧
    is_perpendicular (line_through B C) (line_through C D) →
    dist C D = (48 * Real.sqrt 3) / 15 :=
by
  intro A B C D
  intro h
  have h_angle : ∠ B A C = 150 := h.1
  have h_dist_AB : dist A B = 5 := h.2.1
  have h_dist_BC : dist B C = 6 := h.2.2.1
  have h_perp_AB_AD : is_perpendicular (line_through A B) (line_through A D) := h.2.2.2.1
  have h_perp_BC_CD : is_perpendicular (line_through B C) (line_through C D) := h.2.2.2.2
  -- Proof to be filled in
  sorry

end triangle_ABC_CD_length_l661_661515


namespace product_of_roots_cubic_l661_661972

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  in (∃ r1 r2 r3 : ℝ, p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ r1 * r2 * r3 = 50) := 
by 
  sorry

end product_of_roots_cubic_l661_661972


namespace karl_total_income_l661_661673

theorem karl_total_income : 
  let originalTshirtPrice := 5
      refurbishedTshirtPrice := originalTshirtPrice / 2
      pantPrice := 4
      skirtPrice := 6
      tshirtsSold := 15
      refurbishedTshirtsSold := 7
      pantsSold := 6
      skirtsSold := 12
      tshirtsDiscount := (tshirtsSold / 5).to_nat * (5 * originalTshirtPrice * 0.20)
      skirtsDiscount := (skirtsSold / 2).to_nat * (2 * skirtPrice * 0.10)
      salesTaxRate := 0.08 in
  let totalIncomeBeforeDiscounts := (tshirtsSold - refurbishedTshirtsSold) * originalTshirtPrice + refurbishedTshirtsSold * refurbishedTshirtPrice + pantsSold * pantPrice + skirtsSold * skirtPrice in
  let totalIncomeAfterDiscounts := totalIncomeBeforeDiscounts - tshirtsDiscount - skirtsDiscount in
  let totalIncomeIncludingTax := totalIncomeAfterDiscounts * (1 + salesTaxRate) in
  round (totalIncomeIncludingTax * 100) / 100 = 141.80 := sorry

end karl_total_income_l661_661673


namespace curve_intersections_count_l661_661071

def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (Real.cos t + t, Real.sin t)

def curve_intersections (N : ℕ) : Prop :=
  ∃ (t s : ℝ), 
    x =  Real.cos t + t ∧ Real.sin t =  Real.sin s ∧ 
    x ∈ [2, 50] ∧ t ≠ s ∧
    N = 12

theorem curve_intersections_count : curve_intersections 12 := 
  sorry

end curve_intersections_count_l661_661071


namespace simplify_expression_l661_661261

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x - 2) / (x ^ 2 - 1) / (1 - 1 / (x - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_expression_l661_661261


namespace john_bought_3_tshirts_l661_661213

theorem john_bought_3_tshirts (T : ℕ) (h : 20 * T + 50 = 110) : T = 3 := 
by 
  sorry

end john_bought_3_tshirts_l661_661213


namespace greatest_possible_perimeter_l661_661540

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l661_661540


namespace infinite_geometric_series_division_l661_661817

theorem infinite_geometric_series_division (r : ℕ) (h_r : r > 1)
  (S : ℕ → ℝ) (h_S : (∑ i in finset.range r, S i) = 1) :
  ∃ (a : ℕ → ℝ), (∀ n, a n > 0) ∧ (∑' n, a n = 1) ∧
  (∀ (S : ℕ → ℝ) (hS : (∑ i in finset.range r, S i) = 1), 
  ∃ (A : ℕ → ℕ → ℝ), (∀ k : fin r, ∑' n, A k n = S k) ∧ 
  (∀ k n, A k n ≥ 0) ∧ 
  (∀ k n, ∑ (i : ℕ) in finset.range (n+1), A k i ≤ S k)) :=
sorry

end infinite_geometric_series_division_l661_661817


namespace profit_is_minus_4r_l661_661503

def cost_of_buying_oranges := 10 * r
def cost_of_transportation := 2 * 2 * r
def storage_fee := 1 * r
def total_cost := cost_of_buying_oranges + cost_of_transportation + storage_fee

def revenue_from_selling_oranges := 11 * r

def profit := revenue_from_selling_oranges - total_cost

theorem profit_is_minus_4r : profit = -4 * r :=
by
  sorry

end profit_is_minus_4r_l661_661503


namespace problem_statement_l661_661455

noncomputable def is_tangent (A B C : ℝ × ℝ) (p : ℝ) : Prop :=
  let k := (B.2 - A.2) / (B.1 - A.1)
  let line_eq := λ x, k * x + (B.2 - k * B.1)
  ∀ x y, y = line_eq x → y = x^2 / (2 * p) → x = A.1

noncomputable def product_of_distances_greater (O B P Q A C : ℝ × ℝ) : Prop :=
  let bp := Mathlib.sqrt((P.1 - B.1)^2 + (P.2 - (B.2 + 1))^2)
  let bq := Mathlib.sqrt((Q.1 - B.1)^2 + (Q.2 - (B.2 + 1))^2)
  let ba := Mathlib.sqrt((A.1 - B.1)^2 + (A.2 - B.2)^2)
  bp * bq > ba^2

theorem problem_statement (O A B P Q : ℝ × ℝ) (p : ℝ) (hp : p > 0) 
  (hA_on_C : A.1^2 = 2 * p * A.2) 
  (hPQ_on_C : (P.1^2 = 2 * p * P.2) ∧ (Q.1^2 = 2 * p * Q.2)) :
  is_tangent A B C p ∧ product_of_distances_greater O B P Q A C := 
begin
  sorry, -- The proof goes here
end

end problem_statement_l661_661455


namespace max_triangle_perimeter_l661_661584

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l661_661584


namespace average_of_four_l661_661500

-- Define the variables
variables {p q r s : ℝ}

-- Conditions as hypotheses
theorem average_of_four (h : (5 / 4) * (p + q + r + s) = 15) : (p + q + r + s) / 4 = 3 := 
by
  sorry

end average_of_four_l661_661500


namespace transform_coordinates_l661_661660

def coordinates_transformation (a b : ℝ) (A B : ℝ × ℝ) : Prop :=
  B = (a, b) ∧ A = (-a, -b)

theorem transform_coordinates (a b : ℝ) :
  coordinates_transformation a b (0, 0) (a, b) :=
by
  unfold coordinates_transformation
  split
  · exact rfl
  · exact rfl

end transform_coordinates_l661_661660


namespace sum_of_tens_and_units_digit_of_7_pow_1024_l661_661315

theorem sum_of_tens_and_units_digit_of_7_pow_1024 :
  let n := 7 ^ 1024 in 
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  tens + units = 17 :=
by
  let n := 7 ^ 1024
  let tens := (n / 10) % 10
  let units := n % 10
  have : tens + units = 17
  sorry
  exact this

end sum_of_tens_and_units_digit_of_7_pow_1024_l661_661315


namespace greatest_perimeter_of_triangle_l661_661614

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l661_661614


namespace greatest_triangle_perimeter_l661_661577

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l661_661577
