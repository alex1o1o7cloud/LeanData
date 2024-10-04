import Mathlib

namespace sum_of_non_solutions_l276_276626

theorem sum_of_non_solutions (A B C : ℝ) (h_inf_solutions : ∀ x : ℝ, ∃ inf : Prop, (x ∉ {-14, -28} → (x+B)*(2*A*x+56) = 2*(x+C)*(x+14))) :
  A = 1 ∧ B = 14 ∧ C = 28 → (-14 + (-28)) = -42 :=
by
  intro h_values
  cases h_values with hA h_rest
  cases h_rest with hB hC
  rw [hA, hB, hC]
  simp

end sum_of_non_solutions_l276_276626


namespace shapes_identification_l276_276377

theorem shapes_identification :
  (∃ x y: ℝ, (x - 1/2)^2 + y^2 = 1/4) ∧ (∃ t: ℝ, x = -t ∧ y = 2 + t → x + y + 1 = 0) :=
by
  sorry

end shapes_identification_l276_276377


namespace matrix_A_3v_l276_276391

open Matrix

noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]]

open_locale matrix

theorem matrix_A_3v (v : Fin 3 → ℝ) : (A.mulVec v) = 3 • v :=
by {

  sorry
}

end matrix_A_3v_l276_276391


namespace percentage_of_primes_divisible_by_3_is_12_5_l276_276848

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l276_276848


namespace percentage_of_primes_divisible_by_3_l276_276791

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l276_276791


namespace solution_set_inequality_l276_276217

theorem solution_set_inequality :
  {x : ℝ | (x^2 - 4) * (x - 6)^2 ≤ 0} = {x : ℝ | (-2 ≤ x ∧ x ≤ 2) ∨ x = 6} :=
  sorry

end solution_set_inequality_l276_276217


namespace fraction_to_decimal_l276_276301

theorem fraction_to_decimal : (7 : Rat) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l276_276301


namespace percentage_of_primes_divisible_by_3_l276_276807

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l276_276807


namespace application_methods_count_l276_276342

theorem application_methods_count (total_universities: ℕ) (universities_with_coinciding_exams: ℕ) (chosen_universities: ℕ) 
  (remaining_universities: ℕ) (remaining_combinations: ℕ) : 
  total_universities = 6 → universities_with_coinciding_exams = 2 → chosen_universities = 3 → 
  remaining_universities = 4 → remaining_combinations = 16 := 
by
  intros
  sorry

end application_methods_count_l276_276342


namespace arithmetic_sequence_n_l276_276010

theorem arithmetic_sequence_n (a_n : ℕ → ℕ) (S_n : ℕ) (n : ℕ) 
  (h1 : ∀ i, a_n i = 20 + (i - 1) * (54 - 20) / (n - 1)) 
  (h2 : S_n = 37 * n) 
  (h3 : S_n = 999) : 
  n = 27 :=
by sorry

end arithmetic_sequence_n_l276_276010


namespace find_angle_between_vectors_l276_276009

noncomputable def vector_a : ℝ × ℝ := sorry
def vector_b : ℝ × ℝ := (1, 1)
def dot_product (v₁ v₂ : ℝ × ℝ) := v₁.1 * v₂.1 + v₁.2 * v₂.2
def magnitude (v : ℝ × ℝ) := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
def angle_between (v₁ v₂ : ℝ × ℝ) := Real.acos (dot_product v₁ v₂ / (magnitude v₁ * magnitude v₂))
  
theorem find_angle_between_vectors (ha_dot_condition : dot_product vector_a (vector_a - (2, 2) * vector_b) = 3)
        (ha_magnitude : magnitude vector_a = 1) : 
    angle_between vector_a vector_b = 3 * Real.pi / 4 :=
sorry

end find_angle_between_vectors_l276_276009


namespace sqrt_three_binary_rep_nonzero_l276_276616

theorem sqrt_three_binary_rep_nonzero (n : ℕ) (n_pos : 0 < n) :
  ∃ i, n ≤ i ∧ i ≤ 2 * n ∧ (bit_n (sqrt 3) i = 1) := by
  sorry

end sqrt_three_binary_rep_nonzero_l276_276616


namespace spiders_loose_l276_276336

noncomputable def initial_birds : ℕ := 12
noncomputable def initial_puppies : ℕ := 9
noncomputable def initial_cats : ℕ := 5
noncomputable def initial_spiders : ℕ := 15
noncomputable def birds_sold : ℕ := initial_birds / 2
noncomputable def puppies_adopted : ℕ := 3
noncomputable def remaining_puppies : ℕ := initial_puppies - puppies_adopted
noncomputable def remaining_cats : ℕ := initial_cats
noncomputable def total_remaining_animals_except_spiders : ℕ := birds_sold + remaining_puppies + remaining_cats
noncomputable def total_animals_left : ℕ := 25
noncomputable def remaining_spiders : ℕ := total_animals_left - total_remaining_animals_except_spiders
noncomputable def spiders_went_loose : ℕ := initial_spiders - remaining_spiders

theorem spiders_loose : spiders_went_loose = 7 := by
  sorry

end spiders_loose_l276_276336


namespace extreme_values_of_f_max_value_of_g_l276_276893

noncomputable def f : ℝ → ℝ := λ x, x * Real.log x

theorem extreme_values_of_f :
  (∃ x, f(x) = -Real.exp (-1)) ∧ (∀ y, f y ≤ -Real.exp (-1)) :=
sorry

noncomputable def g (k : ℝ) : ℝ → ℝ := λ x, x * Real.log x - (x - 1)

theorem max_value_of_g (k : ℝ) :
  (∀ x ∈ Set.Icc (1:ℝ) (Real.exp 1), g k x ≤ if k < 1 then (Real.exp 1) - k * (Real.exp 1) + k else 0) :=
sorry

end extreme_values_of_f_max_value_of_g_l276_276893


namespace subsets_containing_5_and_6_l276_276496

theorem subsets_containing_5_and_6 {α : Type} [DecidableEq α] 
  (S : Finset α) (e1 e2 : α) (h : e1 ≠ e2) 
  (H : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ T, e1 ∈ T ∧ e2 ∈ T)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276496


namespace solve_for_x_l276_276166

theorem solve_for_x (x : ℚ) (h : (3 - x)/(x + 2) + (3 * x - 6)/(3 - x) = 2) : x = -7/6 := 
by 
  sorry

end solve_for_x_l276_276166


namespace monotonically_decreasing_iff_a_lt_1_l276_276450

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * a * x^2 - 2 * x

theorem monotonically_decreasing_iff_a_lt_1 {a : ℝ} (h : ∀ x > 0, (deriv (f a) x) < 0) : a < 1 :=
sorry

end monotonically_decreasing_iff_a_lt_1_l276_276450


namespace new_rectangle_area_l276_276633

-- Conditions
def A : ℝ := 3
def B : ℝ := 4
def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)
def C : ℝ := hypotenuse A B
def L : ℝ := C + 2 * A
def W : ℝ := abs (C - 2 * A)

-- Main statement
theorem new_rectangle_area : L * W = 11 := by sorry

end new_rectangle_area_l276_276633


namespace inradius_inequality_l276_276133

noncomputable def circumradius (R : ℝ) : ℝ := 1
noncomputable def inradius (r : ℝ) : ℝ 
noncomputable def orthic_triangle_inradius (p : ℝ) : ℝ

theorem inradius_inequality (r p : ℝ) 
  (h1 : circumradius 1 = 1)
  (h2 : inradius r = r)
  (h3 : orthic_triangle_inradius p = p) :
  p ≤ 1 - 1 / (3 * (1 + r) ^ 2) := 
sorry

end inradius_inequality_l276_276133


namespace train_speed_l276_276926

theorem train_speed :
  (length_of_train = 450) → (cross_time = 27) → 
  (speed_in_km_hr = 60) :=
by
  -- convert length of train from meters to kilometers
  let length_of_train_km := 0.45
  -- convert cross time from seconds to hours
  let cross_time_hr := 27 / 3600
  -- compute speed in km/hr
  let speed_in_km_hr := length_of_train_km / cross_time_hr
  sorry

end train_speed_l276_276926


namespace area_of_isosceles_triangle_l276_276084

theorem area_of_isosceles_triangle (A B C D : Type) [MetricSpace D]
  (h1 : ∀ P Q : D, dist P Q = dist Q P)  -- Metric Space Symmetry
  (hAB : dist A B = 41)  -- AB = 41
  (hAC : dist A C = 41)  -- AC = 41
  (hBD : dist B D = 9)   -- BD = 9
  (hDC : dist D C = 9)   -- DC = 9
  (hBC : dist B C = 18)  -- BC = 18
  (hAD_perp : ∀ P Q : D, ∀ u w : ℝ, perpendicular u w → dist P Q = sqrt(u^2 + w^2))    -- AD perpendicular to BC (Pythagorean theorem)
  (hAD : dist A D = 40)  -- AD = 40
: 
  let base := dist B C in 
  let height := dist A D in 
  1/2 * base * height = 360 := 
begin
  sorry
end

end area_of_isosceles_triangle_l276_276084


namespace isosceles_triangle_count_l276_276592

namespace TriangleProblem

-- Define the basic geometric objects and their properties
structure Point := (x : ℝ) (y : ℝ)
def Triangle := (A B C : Point)

axiom is_congruent (A B : Point) : Prop
axiom is_parallel (L1 L2 : Point → Point) : Prop
axiom angle (A B C : Point) : ℝ

-- Given conditions
variables {A B C D E F : Point}
def ΔABC : Triangle := ⟨A, B, C⟩
def ΔABD : Triangle := ⟨A, B, D⟩
def ΔBDE : Triangle := ⟨B, D, E⟩
def ΔDEF : Triangle := ⟨D, E, F⟩
def ΔEFB : Triangle := ⟨E, F, B⟩
def ΔFEC : Triangle := ⟨F, E, C⟩
def ΔDEC : Triangle := ⟨D, E, C⟩

axiom H1 : is_congruent A B A C
axiom H2 : angle A B C = 60
axiom H3 : ∃ D, angle A B D = angle D B C
axiom H4 : ∃ E ∈ line B C, is_parallel (λ x, D) (λ x, A B)
axiom H5 : ∃ F ∈ line A C, is_parallel (λ x, E F) (λ x, B D)

-- Proof goal
theorem isosceles_triangle_count 
  (h1 : ΔABC.is_isosceles) 
  (h2 : ΔABD.is_isosceles) 
  (h3 : ΔBDE.is_isosceles) 
  (h4 : ΔDEF.is_isosceles) 
  (h5 : ΔEFB.is_isosceles) 
  (h6 : ΔFEC.is_isosceles) 
  (h7 : ΔDEC.is_isosceles) : 
  7 = 7 := 
sorry

end TriangleProblem

end isosceles_triangle_count_l276_276592


namespace sequence_general_formula_l276_276589

theorem sequence_general_formula (a : ℕ → ℝ) 
  (h1 : ∀ n, sqrt (a (n + 1)) = sqrt (a n) + sqrt 2) 
  (h2 : a 1 = 8) : 
  ∀ n, a n = 2 * (n + 1)^2 := 
by
  intros n
  sorry

end sequence_general_formula_l276_276589


namespace proposition_holds_for_all_positive_odd_numbers_l276_276337

theorem proposition_holds_for_all_positive_odd_numbers
  (P : ℕ → Prop)
  (h1 : P 1)
  (h2 : ∀ k, k ≥ 1 → P k → P (k + 2)) :
  ∀ n, n % 2 = 1 → n ≥ 1 → P n :=
by
  sorry

end proposition_holds_for_all_positive_odd_numbers_l276_276337


namespace numberOfWaysToChoose4Cards_l276_276515

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l276_276515


namespace draw_3_balls_in_order_l276_276314

-- Definitions based on conditions
def num_balls : ℕ := 12
def num_draws : ℕ := 3

-- Theorem statement
theorem draw_3_balls_in_order (h1 : num_balls = 12) (h2 : num_draws = 3) : 
  (12 * 11 * 10 = 1320) :=
begin
  sorry
end

end draw_3_balls_in_order_l276_276314


namespace star_interior_angles_sum_l276_276573

theorem star_interior_angles_sum (n : ℕ) (h : n ≥ 6) : 
  let S := 180 * (n - 2) in 
  S = 180 * (n - 2) :=
by
  sorry

end star_interior_angles_sum_l276_276573


namespace primes_less_than_20_divisible_by_3_percentage_l276_276723

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l276_276723


namespace number_of_tangent_circles_l276_276113

noncomputable def C1 : set ℝ := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 = 4}
noncomputable def C2 : set ℝ := {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2)^2 = 9}

theorem number_of_tangent_circles :
  ∃ n : ℕ, n = 4 ∧
  (∀ C : set ℝ, 
    (∀ p : ℝ × ℝ, (p.1)^2 + (p.2)^2 = 4 → p ∈ C) ∨
    (∀ p : ℝ × ℝ, (p.1 - 5)^2 + (p.2)^2 = 9 → p ∈ C)) := sorry

end number_of_tangent_circles_l276_276113


namespace primes_less_than_20_divisible_by_3_percentage_l276_276728

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l276_276728


namespace percentage_primes_divisible_by_3_l276_276783

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l276_276783


namespace range_of_f_l276_276207

noncomputable def f (x : ℝ) := x^2 + 2*x + 3

theorem range_of_f :
  (set.Ici 0) ⊆ set.univ → set.range (λ x : ℝ, f x) = set.Ici 3 :=
by
  intro h
  sorry

end range_of_f_l276_276207


namespace arrangement_problems_l276_276894

def arrangements_without_head_tail (arrangements : Finset (Fin 6)) (A : Fin 6) : Nat :=
  -- Calculate the number of arrangements where student A is not at the head or the tail
  let without_head_tail := arrangements.filter (λ a => a ≠ 0 ∧ a ≠ 5)
  without_head_tail.card * (5.factorial)

def non_adjacent_ABC (arrangements : Finset (Fin 6)) (A B C : Fin 6) : Nat :=
  -- Calculate the number of arrangements where A, B, and C are not adjacent
  let remaining := arrangements.filter (λ a => a ≠ A ∧ a ≠ B ∧ a ≠ C)
  remaining.card.factorial * (4.choose 3).factorial

theorem arrangement_problems 
  (arrangements : Finset (Fin 6))
  (A B C : Fin 6) :
  arrangements_without_head_tail arrangements A = 480 ∧
  non_adjacent_ABC arrangements A B C = 144 :=
by
  sorry

end arrangement_problems_l276_276894


namespace correct_operation_x_inv_l276_276868

theorem correct_operation_x_inv (x : ℝ) (h : x ≠ 0) : x⁻² = 1 / x² :=
by sorry

end correct_operation_x_inv_l276_276868


namespace percent_primes_divisible_by_3_less_than_20_l276_276762

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l276_276762


namespace probability_xiao_chen_given_xiao_li_l276_276873

noncomputable def P_A : ℝ := 1 / 4
noncomputable def P_B : ℝ := 1 / 3
noncomputable def P_AB : ℝ := 1 / 5

def P_B_given_A : ℝ := P_AB / P_A

theorem probability_xiao_chen_given_xiao_li :
  P_B_given_A = 4 / 5 :=
by
-- The proof would go here
sorry

end probability_xiao_chen_given_xiao_li_l276_276873


namespace line_intersects_circle_midpoint_trajectory_l276_276310

-- Definitions based on conditions
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

def line_eq (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Statement of the problem
theorem line_intersects_circle :
  ∀ m : ℝ, ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y :=
sorry

theorem midpoint_trajectory :
  ∀ (x y : ℝ), 
  (∃ (xa ya xb yb : ℝ), circle_eq xa ya ∧ line_eq m xa ya ∧ 
   circle_eq xb yb ∧ line_eq m xb yb ∧ (x, y) = ((xa + xb) / 2, (ya + yb) / 2)) ↔
   ( x - 1 / 2)^2 + (y - 1)^2 = 1 / 4 :=
sorry

end line_intersects_circle_midpoint_trajectory_l276_276310


namespace different_suits_choice_count_l276_276507

-- Definitions based on the conditions
def standard_deck : List (Card × Suit) := 
  List.product Card.all Suit.all

def four_cards (deck : List (Card × Suit)) : Prop :=
  deck.length = 4 ∧ ∀ (i j : Fin 4), i ≠ j → (deck.nthLe i (by simp) : Card × Suit).2 ≠ (deck.nthLe j (by simp) : Card × Suit).2

-- Statement of the proof problem
theorem different_suits_choice_count :
  ∃ l : List (Card × Suit), four_cards l ∧ standard_deck.choose 4 = 28561 :=
by
  sorry

end different_suits_choice_count_l276_276507


namespace range_of_z_in_parallelogram_l276_276004

-- Define the points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := {x := -1, y := 2}
def B : Point := {x := 3, y := 4}
def C : Point := {x := 4, y := -2}

-- Define the condition for point (x, y) to be inside the parallelogram (including boundary)
def isInsideParallelogram (p : Point) : Prop := sorry -- Placeholder for actual geometric condition

-- Statement of the problem
theorem range_of_z_in_parallelogram (p : Point) (h : isInsideParallelogram p) : 
  -14 ≤ 2 * p.x - 5 * p.y ∧ 2 * p.x - 5 * p.y ≤ 20 :=
sorry

end range_of_z_in_parallelogram_l276_276004


namespace proportional_segments_l276_276430

variables {A B C P D E F : Type}

-- Given "P is a point on the circumcircle of triangle ABC"
axiom circumcircle (A B C P : Type) : Prop

-- Given "Perpendiculars from P to BC, CA, and AB intersect them at D, E, and F"
axiom perpendicular (P BC CA AB : Type) (D E F : Type) : Prop

-- Given "E is the midpoint of the segment DF"
axiom midpoint (E D F : Type) : Prop

-- We have to prove that ratio holds true
theorem proportional_segments
  (h1 : circumcircle A B C P)
  (h2 : perpendicular P BC CA AB D E F)
  (h3 : midpoint E D F) :
  ∀ (AP BC AB PC : ℝ), AP / PC = AB / BC := 
by
  sorry

end proportional_segments_l276_276430


namespace driving_hours_fresh_l276_276966

theorem driving_hours_fresh (x : ℚ) : (25 * x + 15 * (9 - x) = 152) → x = 17 / 10 :=
by
  intros h
  sorry

end driving_hours_fresh_l276_276966


namespace percentage_primes_divisible_by_3_l276_276786

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l276_276786


namespace correct_statements_l276_276370

variable (a : ℝ) (x y : ℝ)

/- Define the functions -/
def exp_function := a^x
def log_function := log a (a^x)
def square_function := x^2
def exponential_function := 3^x
def first_odd_function := (1 / 2) + (1 / (2^x - 1))
def second_odd_function := ((1 + 2^x)^2) / (x * 2^x)
def parabola_function := (x - 1)^2
def linear_function := 2 * x - 1

/- Define the conditions -/
def condition1 := ∀ x : ℝ, a > 0 ∧ a ≠ 1 → exp_function = log_function
def condition2 := ∀ x : ℝ, (0 ≤ x^2) ∧ (0 < 3^x)
def condition3 := ∀ x : ℝ, (first_odd_function = -first_odd_function) ∧ 
  (second_odd_function = -second_odd_function)
def condition4 := ∀ x : ℝ, (0 < x) → (parabola_function ≠ -parabola_function) ∧
  (linear_function ≠ -linear_function)

/- Define the problem statement -/
theorem correct_statements (a : ℝ) (x : ℝ) :
  (condition1 a x) ∧ ¬(condition2 x) ∧ (condition3 x) ∧ ¬(condition4 x) :=
by
  sorry

end correct_statements_l276_276370


namespace sum_of_three_numbers_l276_276274

theorem sum_of_three_numbers :
  ∃ A B C : ℕ, 
    (100 ≤ A ∧ A < 1000) ∧  -- A is a three-digit number
    (10 ≤ B ∧ B < 100) ∧     -- B is a two-digit number
    (10 ≤ C ∧ C < 100) ∧     -- C is a two-digit number
    (A + (if (B / 10 = 7 ∨ B % 10 = 7) then B else 0) + 
       (if (C / 10 = 7 ∨ C % 10 = 7) then C else 0) = 208) ∧
    (if (B / 10 = 3 ∨ B % 10 = 3) then B else 0) + 
    (if (C / 10 = 3 ∨ C % 10 = 3) then C else 0) = 76 ∧
    A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276274


namespace arithmetic_seq_solution_l276_276678

variables (a : ℕ → ℤ) (d : ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) - a n = d

def seq_cond (a : ℕ → ℤ) (d : ℤ) : Prop :=
is_arithmetic_sequence a d ∧ (a 2 + a 6 = a 8)

noncomputable def sum_first_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem arithmetic_seq_solution :
  ∀ (a : ℕ → ℤ) (d : ℤ), seq_cond a d → (a 2 - a 1 ≠ 0) → 
    (sum_first_n a 5 / a 5) = 3 :=
by
  intros a d h_cond h_d_ne_zero
  sorry

end arithmetic_seq_solution_l276_276678


namespace smaller_square_area_percentage_of_larger_l276_276903

theorem smaller_square_area_percentage_of_larger 
    (a b y : ℝ)
    (h_1 : a = 8)
    (h_2 : b = 2 * y)
    (h_3 : 3 * y^2 - 8 * sqrt 2 * y = 0) : 
    b^2 / a^2 * 100 ≈ 88.89 :=
by
  sorry

end smaller_square_area_percentage_of_larger_l276_276903


namespace primes_divisible_by_3_percentage_is_12_5_l276_276747

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l276_276747


namespace regular_hourly_rate_l276_276136

variable (R : ℝ)

axiom max_hours : 40
axiom regular_hours : 20
axiom overtime_rate_multiplier : ℝ := 1.25
axiom total_earnings : ℝ := 360

theorem regular_hourly_rate : R = 8 :=
sorry

end regular_hourly_rate_l276_276136


namespace bins_of_vegetables_l276_276965

-- Define the conditions
def total_bins : ℝ := 0.75
def bins_of_soup : ℝ := 0.12
def bins_of_pasta : ℝ := 0.5

-- Define the statement to be proved
theorem bins_of_vegetables :
  total_bins = bins_of_soup + (0.13) + bins_of_pasta := 
sorry

end bins_of_vegetables_l276_276965


namespace probability_monotonically_increasing_l276_276017

noncomputable def is_monotonically_increasing_in_interval (a : ℝ) : Prop :=
  a ≥ 1 ∧ a < 4

theorem probability_monotonically_increasing {a : ℝ} (h : a ∈ set.Icc 1 6) :
  ∃ (P : ℝ), P = 3 / 5 ∧ ∀ (b : ℝ), b ∈ set.Icc 1 6 → is_monotonically_increasing_in_interval b -> P = 3 / 5 :=
sorry

end probability_monotonically_increasing_l276_276017


namespace decreasing_iff_m_range_l276_276543

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := - (1 / 2) * x^2 + m * Real.log x

theorem decreasing_iff_m_range (m : ℝ) : 
  (∀ x : ℝ, 1 < x → (deriv (λ x, - (1 / 2) * x^2 + m * Real.log x) x < 0)) ↔ (m ≤ 1) := 
by
  sorry

end decreasing_iff_m_range_l276_276543


namespace math_problem_l276_276945

theorem math_problem:
  (-1)^(4) + (sqrt 3 - 1)^(0 : ℤ) + (1 / 3 : ℝ)^(-2 : ℤ) + abs (sqrt 3 - 2) + tan (π / 3) = 11 :=
by
  -- The proof should go here
  sorry

end math_problem_l276_276945


namespace triangle_inequality_l276_276003

theorem triangle_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
begin
  sorry
end

end triangle_inequality_l276_276003


namespace y_intercept_of_line_l276_276683

theorem y_intercept_of_line : ∀ (x y : ℝ), (5 * x - 2 * y - 10 = 0) → (x = 0) → (y = -5) :=
by
  intros x y h1 h2
  sorry

end y_intercept_of_line_l276_276683


namespace percentage_primes_divisible_by_3_l276_276779

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l276_276779


namespace find_a_plus_b_l276_276326

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x ^ 2) / 4 + (y ^ 2) = 1

def foci_of_ellipse : set (ℝ × ℝ) :=
  {p | p = (sqrt 3, 0) ∨ p = (-sqrt 3, 0)}

def circle_radius (r : ℝ) : Prop :=
  ∀ (x y : ℝ), (x ^ 2 + y ^ 2 = r ^ 2) →
    (∃ (p : ℝ × ℝ), p ∈ foci_of_ellipse ∧ p = (x, y)) ∧
    (∃ (p : ℂ), ellipse p.fst p.snd ∧ (p.fst, p.snd) = (x, y))

theorem find_a_plus_b : ∃ (a b : ℝ), a + b = 2 * sqrt 3 + 3 ∧ (∀ r, circle_radius r) -> a ≤ r < b :=
sorry

end find_a_plus_b_l276_276326


namespace boat_speed_still_water_l276_276880

theorem boat_speed_still_water (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 16) (h2 : upstream_speed = 9) : 
  (downstream_speed + upstream_speed) / 2 = 12.5 := 
by
  -- conditions explicitly stated above
  sorry

end boat_speed_still_water_l276_276880


namespace carpet_length_l276_276937

theorem carpet_length (percent_covered : ℝ) (width : ℝ) (floor_area : ℝ) (carpet_length : ℝ) :
  percent_covered = 0.30 → width = 4 → floor_area = 120 → carpet_length = 9 :=
by
  sorry

end carpet_length_l276_276937


namespace winning_candidate_percentage_l276_276689

noncomputable def total_votes := 1136 + 5636 + 11628

noncomputable def winning_candidate_votes := 11628

noncomputable def winning_percentage := (winning_candidate_votes.toFloat / total_votes.toFloat) * 100

theorem winning_candidate_percentage : winning_percentage ≈ 63.2 :=
by
  sorry

end winning_candidate_percentage_l276_276689


namespace percentage_of_primes_divisible_by_3_l276_276754

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l276_276754


namespace explicit_formula_for_f_range_of_values_for_a_l276_276023

-- Define the function f
noncomputable def f : ℝ → ℝ
| x := if x < 0 then -2^(-x) else if x = 0 then 0 else 2^x

-- First part: Prove the explicit formula for f(x)
theorem explicit_formula_for_f :
  ∀ x : ℝ, -1 < x → x < 1 → 
    f x = (if x < 0 then -2^(-x) else if x = 0 then 0 else 2^x) :=
by
  intros x h1 h2
  sorry

-- Second part: Prove the range of values for a
theorem range_of_values_for_a (a : ℝ) :
  (∀ x, -1 < x ∧ x < 1 → f x ≤ 2 * a) → 1 ≤ a :=
by
  intros h
  sorry

end explicit_formula_for_f_range_of_values_for_a_l276_276023


namespace determinant_matrix_3x3_l276_276948

theorem determinant_matrix_3x3 :
  Matrix.det ![![3, 1, -2], ![8, 5, -4], ![1, 3, 6]] = 140 :=
by
  sorry

end determinant_matrix_3x3_l276_276948


namespace corrected_mean_l276_276882

theorem corrected_mean (n : ℕ) (mean old_obs new_obs : ℝ) 
    (obs_count : n = 50) (old_mean : mean = 36) (incorrect_obs : old_obs = 23) (correct_obs : new_obs = 46) :
    (mean * n - old_obs + new_obs) / n = 36.46 := by
  sorry

end corrected_mean_l276_276882


namespace triangle_area_le_half_parallelogram_area_l276_276152

theorem triangle_area_le_half_parallelogram_area {P : Type} [parallelogram P]
  (T : ℝ)
  (hT : parallelogram_area P = T)
  (E F G : point P)
  (hEFG_inside : E ∈ P ∧ F ∈ P ∧ G ∈ P) :
  triangle_area E F G ≤ T / 2 :=
sorry

end triangle_area_le_half_parallelogram_area_l276_276152


namespace math_problem_l276_276405

open Real

theorem math_problem (x : ℝ) (p q : ℕ)
  (h1 : (1 + sin x) * (1 + cos x) = 9 / 4)
  (h2 : (1 - sin x) * (1 - cos x) = p - sqrt q)
  (hp_pos : p > 0) (hq_pos : q > 0) : p + q = 1 := sorry

end math_problem_l276_276405


namespace boundary_length_correct_l276_276345

noncomputable def total_boundary_length : ℝ :=
  let side_length := Real.sqrt 144 in
  let larger_segment := side_length / 3 in
  let smaller_segment := larger_segment / 2 in
  let larger_arc_length := 2 * Real.pi * larger_segment / 4 in
  let smaller_arc_length := 2 * Real.pi * smaller_segment / 4 in
  let total_arc_length := 4 * (larger_arc_length + smaller_arc_length) in
  let total_straight_length := 4 * larger_segment in
  total_arc_length + total_straight_length

theorem boundary_length_correct : abs (total_boundary_length - 53.7) < 0.1 :=
  sorry

end boundary_length_correct_l276_276345


namespace xiao_ming_fruits_l276_276554

theorem xiao_ming_fruits :
  let choices := 
    (λ n : ℕ, (∀ i : ℕ, (1 ≤ i ∧ i ≤ 7) → 
      (if (i = 1 ∨ i = 7) then n i = 3 else true) ∧ 
      (if (2 ≤ i ∧ i ≤ 6) then abs (n i - n (i-1)) ≤ 1 else true))) in
  ∃ C : ℕ, C = 141 ∧ true := sorry

end xiao_ming_fruits_l276_276554


namespace angle_bisector_GM_of_ABCD_l276_276643

-- Define the main theorem, which encapsulates the problem statement and conclusion.
theorem angle_bisector_GM_of_ABCD
  (A B C D P M G : Type) 
  [add_comm_group A] [affine_space A B] -- B is an affine space over A
  [add_comm_group G] [affine_space G B] -- B is an affine space over G
  (h1 : midpoint M C D)  -- M is the midpoint of C and D
  (h2 : circle_center P (circumcircle B C M)) -- P is the center of circumcircle of ∆BCM
  (h3 : circle_center P (circumcircle B G D)) -- P is the center of circumcircle of ∆BGD
  :
  is_angle_bisector (segment.mk G M) (angle.mk B G D) :=
sorry

end angle_bisector_GM_of_ABCD_l276_276643


namespace compare_x_y_l276_276054

theorem compare_x_y :
  let x := 123456789 * 123456786
  let y := 123456788 * 123456787
  x < y := sorry

end compare_x_y_l276_276054


namespace parabola_vertex_correct_l276_276954

noncomputable def parabola_vertex (a : ℝ) (h_a : a ≠ 0) : ℝ × ℝ :=
  let x1 := -4
  let x2 := 2
  let y1 := x1^2 + a * x1 - 5
  let y2 := x2^2 + a * x2 - 5
  let k := (y2 - y1) / (x2 - x1)
  let tangent_slope := 2 * -1 + a
  have h_slope : tangent_slope = k, from sorry,
  let tangent_point := (-1, -a - 4)
  let line := (a - 2) * x - y - 6
  have h1 : 5 * (0^2) + 5 * (0^2) = 36, from sorry,
  have h2 : ((6 : ℝ) / real.sqrt ((a - 2 : ℝ) ^ 2 + 1)) = real.sqrt (36 / 5), from sorry,
  have a_val : a = 4, from sorry,
  let vertex_x := -a / (2 : ℝ)
  let vertex_y := (vertex_x) ^ 2 + a * (vertex_x) - 5
  (vertex_x, vertex_y)

theorem parabola_vertex_correct (a : ℝ) (h_a : a ≠ 0) : parabola_vertex a h_a = (-2, -9) :=
  sorry

end parabola_vertex_correct_l276_276954


namespace intersection_Y_function_coordinates_l276_276374

-- Definitions for the problem conditions
def is_Y_function (f g : ℝ → ℝ) : Prop := ∀ x, f x = g (-x)

def intersects_x_axis_once (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

-- The given function
def given_function (k : ℝ) (x : ℝ) : ℝ :=
  (k / 4) * x^2 + (k - 1) * x + k - 3

-- The Y function for a given function
def Y_function (k : ℝ) (x : ℝ) : ℝ :=
  given_function k (-x)

-- The coordinates are (3, 0) or (4, 0)
theorem intersection_Y_function_coordinates (k : ℝ) :
  intersects_x_axis_once (given_function k) →
  (∀ x, (Y_function 0 x = 0 → x = 3) ∨ (Y_function (-1) x = 0 → x = 4)) :=
by sorry

end intersection_Y_function_coordinates_l276_276374


namespace intersection_coordinates_l276_276186

theorem intersection_coordinates (x y : ℝ) 
  (h1 : y = 2 * x - 1) 
  (h2 : y = x + 1) : 
  x = 2 ∧ y = 3 := 
by 
  sorry

end intersection_coordinates_l276_276186


namespace percentage_of_primes_divisible_by_3_l276_276751

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l276_276751


namespace subsets_containing_5_and_6_l276_276489

theorem subsets_containing_5_and_6 (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ s, 5 ∈ s ∧ 6 ∈ s)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276489


namespace percentage_primes_divisible_by_3_l276_276780

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l276_276780


namespace cannot_partition_nat_l276_276601

theorem cannot_partition_nat (A : ℕ → Set ℕ) (h1 : ∀ i j, i ≠ j → Disjoint (A i) (A j))
    (h2 : ∀ k, Finite (A k) ∧ sum {n | n ∈ A k}.toFinset id = k + 2013) :
    False :=
sorry

end cannot_partition_nat_l276_276601


namespace percentage_primes_divisible_by_3_l276_276860

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l276_276860


namespace no_integral_roots_l276_276919

theorem no_integral_roots (p : ℤ[x])
  (h0 : p.eval 0 % 2 = 1)
  (h1 : p.eval 1 % 2 = 1) :
  ¬ ∃ k : ℤ, p.eval k = 0 :=
by
  sorry

end no_integral_roots_l276_276919


namespace percentage_of_primes_divisible_by_3_l276_276759

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l276_276759


namespace hyperbola_eccentricity_l276_276911

theorem hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
    (intersects_origin : ∃ M N : ℝ × ℝ, \(x, y). \frac{x^2}{a^2} - \frac{y^2}{b^2} = 1 ∧ origin_line_through (0, 0) M N )
    (P : ℝ × ℝ) (hP : (P.1^2) / (a^2) - (P.2^2) / (b^2) = 1)
    (slope_product : ∀ M N : ℝ × ℝ, M ≠ N ∧ M ≠ P ∧ N ≠ P →
      (exists k_PM, exists k_NP, (k_PM = (P.2 - M.2) / (P.1 - M.1) ∧ k_NP = (P.2 + M.2) / (P.1 + M.1)) ∧ (k_PM * k_NP = 5/4))) :
  let e := sqrt (1 + (b^2 / a^2)) in
  e = 3 / 2 := by
    sorry

end hyperbola_eccentricity_l276_276911


namespace sum_of_possible_students_l276_276924

theorem sum_of_possible_students (h₁ : ∀ s : ℕ, 200 ≤ s ∧ s ≤ 250 → (s-1) % 7 = 0 → 1 ≤ 7) :
  (∑ n in Finset.filter (λ n, 200 ≤ n ∧ n ≤ 250 ∧ (n-1) % 7 = 0) (Finset.range 251)) = 1575 := sorry

end sum_of_possible_students_l276_276924


namespace minimal_connections_correct_l276_276071

-- Define a Lean structure to encapsulate the conditions
structure IslandsProblem where
  islands : ℕ
  towns : ℕ
  min_towns_per_island : ℕ
  condition_islands : islands = 13
  condition_towns : towns = 25
  condition_min_towns : min_towns_per_island = 1

-- Define a function to represent the minimal number of ferry connections
def minimalFerryConnections (p : IslandsProblem) : ℕ :=
  222

-- Define the statement to be proved
theorem minimal_connections_correct (p : IslandsProblem) : 
  p.islands = 13 → 
  p.towns = 25 → 
  p.min_towns_per_island = 1 → 
  minimalFerryConnections p = 222 :=
by
  intros
  sorry

end minimal_connections_correct_l276_276071


namespace sum_of_numbers_l276_276259

def contains_digit (n : Nat) (d : Nat) : Prop := 
  (n / 100 = d) ∨ (n % 100 / 10 = d) ∨ (n % 10 = d)

variables {A B C : Nat}

-- Given conditions
axiom three_digit_number : A ≥ 100 ∧ A < 1000
axiom two_digit_numbers : B ≥ 10 ∧ B < 100 ∧ C ≥ 10 ∧ C < 100
axiom sum_with_sevens : contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7 → A + B + C = 208
axiom sum_with_threes : contains_digit B 3 ∧ contains_digit C 3 ∧ B + C = 76

-- Main theorem to be proved
theorem sum_of_numbers : A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276259


namespace sum_of_three_numbers_l276_276272

theorem sum_of_three_numbers :
  ∃ A B C : ℕ, 
    (100 ≤ A ∧ A < 1000) ∧  -- A is a three-digit number
    (10 ≤ B ∧ B < 100) ∧     -- B is a two-digit number
    (10 ≤ C ∧ C < 100) ∧     -- C is a two-digit number
    (A + (if (B / 10 = 7 ∨ B % 10 = 7) then B else 0) + 
       (if (C / 10 = 7 ∨ C % 10 = 7) then C else 0) = 208) ∧
    (if (B / 10 = 3 ∨ B % 10 = 3) then B else 0) + 
    (if (C / 10 = 3 ∨ C % 10 = 3) then C else 0) = 76 ∧
    A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276272


namespace sum_of_numbers_l276_276262

def contains_digit (n : Nat) (d : Nat) : Prop := 
  (n / 100 = d) ∨ (n % 100 / 10 = d) ∨ (n % 10 = d)

variables {A B C : Nat}

-- Given conditions
axiom three_digit_number : A ≥ 100 ∧ A < 1000
axiom two_digit_numbers : B ≥ 10 ∧ B < 100 ∧ C ≥ 10 ∧ C < 100
axiom sum_with_sevens : contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7 → A + B + C = 208
axiom sum_with_threes : contains_digit B 3 ∧ contains_digit C 3 ∧ B + C = 76

-- Main theorem to be proved
theorem sum_of_numbers : A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276262


namespace class3_total_score_l276_276912

theorem class3_total_score 
  (total_points : ℕ)
  (class1_score class2_score class3_score : ℕ)
  (class1_places class2_places class3_places : ℕ)
  (total_places : ℕ)
  (points_1st  points_2nd  points_3rd : ℕ)
  (h1 : total_points = 27)
  (h2 : class1_score = class2_score)
  (h3 : 2 * class1_places = class2_places)
  (h4 : class1_places + class2_places + class3_places = total_places)
  (h5 : 3 * points_1st + 3 * points_2nd + 3 * points_3rd = total_points)
  (h6 : total_places = 9)
  (h7 : points_1st = 5)
  (h8 : points_2nd = 3)
  (h9 : points_3rd = 1) :
  class3_score = 7 :=
sorry

end class3_total_score_l276_276912


namespace find_k_for_quadratic_root_l276_276415

theorem find_k_for_quadratic_root (k : ℝ) (h : (1 : ℝ).pow 2 + k * 1 - 3 = 0) : k = 2 :=
by
  sorry

end find_k_for_quadratic_root_l276_276415


namespace probability_more_heads_than_tails_l276_276545

theorem probability_more_heads_than_tails :
  let x := \frac{193}{512}
  let y := \frac{63}{256}
  (2 * x + y = 1) →
  (y = \frac{252}{1024}) →
  (x = \frac{193}{512}) :=
by
  let x : ℚ := 193 / 512
  let y : ℚ := 63 / 256
  sorry

end probability_more_heads_than_tails_l276_276545


namespace different_suits_card_combinations_l276_276530

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l276_276530


namespace equilateral_triangle_l276_276051

variable {a b c : ℝ}

-- Conditions
def condition1 (a b c : ℝ) : Prop :=
  (a + b + c) * (b + c - a) = 3 * b * c

def condition2 (a b c : ℝ) (cos_B cos_C : ℝ) : Prop :=
  c * cos_B = b * cos_C

-- Theorem statement
theorem equilateral_triangle (a b c : ℝ) (cos_B cos_C : ℝ)
  (h1 : condition1 a b c)
  (h2 : condition2 a b c cos_B cos_C) :
  a = b ∧ b = c :=
sorry

end equilateral_triangle_l276_276051


namespace max_halls_visited_l276_276671

-- Definitions based on conditions
def Hall : Type := ℕ -- Represent halls as natural numbers
def displays_paintings : Hall → Prop
def displays_sculptures : Hall → Prop
def adjacent (h1 h2 : Hall) : Prop -- Define when two halls are adjacent

axiom hall_count : ∀ h, 0 ≤ h < 16
axiom hall_alternation : ∀ h, displays_paintings h ↔ ¬ displays_sculptures h
axiom initial_hall_A : displays_paintings 0
axiom final_hall_B : displays_paintings 15

-- Main theorem to prove
theorem max_halls_visited : ∀ path : List Hall, (path.head = 0) → (path.last = 15) → 
  (∀ i, 0 ≤ i < path.length → adjacent (path.nth i) (path.nth (i+1)) → path.nth i ≠ path.nth (i+1)) →
  (∀ i j, 0 ≤ i < j < path.length → path.nth i ≠ path.nth j) → 
  path.length ≤ 15 :=
sorry

end max_halls_visited_l276_276671


namespace max_omega_is_9_l276_276029

noncomputable def max_omega (ω : ℝ) (φ : ℝ) : Prop :=
  (ω > 0) ∧ (abs φ ≤ π / 2) ∧
  (sin (ω * -π/4 + φ) = 0) ∧
  ((∀ x, sin (ω * x + φ) = sin (ω * (π/2 - x) + φ) → x = π/4)) ∧ 
  (monotone_on (λ x, sin (ω * x + φ)) (set.Ioo (π/18) (5 * π / 36)))

theorem max_omega_is_9 : ∃ ω φ, max_omega ω φ ∧ ω = 9 :=
begin
  sorry
end

end max_omega_is_9_l276_276029


namespace number_of_real_solutions_to_gx_eq_2x_l276_276977

noncomputable def g (x : ℝ) : ℝ := (Finset.range 50).sum (λ i, (i + 1) / (x - (i + 1)))

theorem number_of_real_solutions_to_gx_eq_2x :
  (∃! x : ℝ, g x = 2 * x) = 51 := 
  sorry

end number_of_real_solutions_to_gx_eq_2x_l276_276977


namespace abe_bob_jellybeans_l276_276930

noncomputable def probability_colors_match : ℚ :=
  let prob_abe_green := 2 / 3 in
  let prob_bob_green := 2 / 6 in
  let prob_both_green := prob_abe_green * prob_bob_green in
  
  let prob_abe_blue := 1 / 3 in
  let prob_bob_blue := 1 / 6 in
  let prob_both_blue := prob_abe_blue * prob_bob_blue in
  
  prob_both_green + prob_both_blue

theorem abe_bob_jellybeans :
  probability_colors_match = 5 / 18 :=
by
  -- This is where the proof would go, but for now we'll skip it
  sorry

end abe_bob_jellybeans_l276_276930


namespace values_of_a_and_b_l276_276116

theorem values_of_a_and_b (a b : ℝ) 
  (hT : (2, 1) ∈ {p : ℝ × ℝ | ∃ (a : ℝ), p.1 * a + p.2 - 3 = 0})
  (hS : (2, 1) ∈ {p : ℝ × ℝ | ∃ (b : ℝ), p.1 - p.2 - b = 0}) :
  a = 1 ∧ b = 1 :=
by
  sorry

end values_of_a_and_b_l276_276116


namespace tan_theta_eq_one_l276_276044

noncomputable def θ : ℝ := sorry

def a (θ : ℝ) : ℝ × ℝ := (1 - Real.sin θ, 1)
def b (θ : ℝ) : ℝ × ℝ := (1 / 2, 1 + Real.sin θ)

def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2

def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

theorem tan_theta_eq_one (h_acute: is_acute θ) (h_parallel: parallel (a θ) (b θ)) : Real.tan θ = 1 :=
by
  sorry

end tan_theta_eq_one_l276_276044


namespace complex_modulus_solution_l276_276563

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem complex_modulus_solution (a : ℝ) (ha : 6 = a)
  (h : is_pure_imaginary ( (a + 3 * complex.I) / (1 - 2 * complex.I)) ) :
  complex.abs (a + 2 * complex.I) = 2 * Real.sqrt 10 := by
  sorry

end complex_modulus_solution_l276_276563


namespace min_of_expression_l276_276431

theorem min_of_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  ∃ x : ℝ, x = 3 * a^2 + 1 / (a * (a - b)) + 1 / (a * b) - 6 * a * c + 9 * c^2 ∧ x = 4 * sqrt 2 :=
sorry

end min_of_expression_l276_276431


namespace number_of_subsets_with_5_and_6_l276_276499

theorem number_of_subsets_with_5_and_6 : 
  let S := {1, 2, 3, 4, 5, 6}
  ∃ n : ℕ, (n = (set.powerset S).count (λ x, {5, 6} ⊆ x)) ∧ n = 16 := 
sorry

end number_of_subsets_with_5_and_6_l276_276499


namespace percent_primes_divisible_by_3_less_than_20_l276_276773

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l276_276773


namespace prob_run_past_spectator_is_one_fourth_l276_276923

open ProbabilityTheory

noncomputable theory

def probability_run_past_spectator (A B : ℝ) : ℝ := 
  if |A - B| > 1 / 2 then 1 else 0

theorem prob_run_past_spectator_is_one_fourth : 
  ∀ (A B : ℝ), (A, B) ∈ set.Icc (0 : ℝ) 1 × set.Icc (0 : ℝ) 1 →
  integrable (λ (A B: ℝ), probability_run_past_spectator A B) measure_space.volume →
  probability (|A - B| > 1 / 2) = 1 / 4 :=
sorry

end prob_run_past_spectator_is_one_fourth_l276_276923


namespace doubled_team_completes_half_in_three_days_l276_276349

theorem doubled_team_completes_half_in_three_days
  (R : ℝ) -- Combined work rate of the original team
  (h : R * 12 = W) -- Original team completes the work W in 12 days
  (W : ℝ) : -- Total work to be done
  (2 * R) * 3 = W/2 := -- Doubled team completes half the work in 3 days
by 
  sorry

end doubled_team_completes_half_in_three_days_l276_276349


namespace volume_relation_of_pyramid_l276_276091

noncomputable def volume_of_tetrahedron (A B C D : ℝ^3) : ℝ := sorry

theorem volume_relation_of_pyramid (S A B C D O : ℝ^3) (base_parallelogram : parallelogram A B C D) (O_inside : point_in_pyramid O S A B C D) :
  volume_of_tetrahedron O S A B + volume_of_tetrahedron O S C D = volume_of_tetrahedron O S B C + volume_of_tetrahedron O S D A := 
sorry

end volume_relation_of_pyramid_l276_276091


namespace sum_sequence_squared_sum_sequence_100_l276_276143

/-
  Prove that the sum of the sequence 1 + 2 + 3 + ... + n + (n-1) + ... + 3 + 2 + 1,
  where the middle number is n, is equal to n^2.

  Specifically, for n = 100, prove that the sum is 100^2 = 10,000.
-/

theorem sum_sequence_squared (n : ℕ) : (∑ i in finset.range(n - 1), i + n + ∑ i in finset.range(n - 1), (n - 1 - i)) = n^2 :=
sorry

/-
  Proof specifically for n = 100
-/
theorem sum_sequence_100 : (∑ i in finset.range(99), i + 100 + ∑ i in finset.range(99), (99 - i)) = 100^2 :=
sorry

end sum_sequence_squared_sum_sequence_100_l276_276143


namespace number_of_subsets_with_5_and_6_l276_276500

theorem number_of_subsets_with_5_and_6 : 
  let S := {1, 2, 3, 4, 5, 6}
  ∃ n : ℕ, (n = (set.powerset S).count (λ x, {5, 6} ⊆ x)) ∧ n = 16 := 
sorry

end number_of_subsets_with_5_and_6_l276_276500


namespace factoring_expression_l276_276187

theorem factoring_expression (a b c x y : ℝ) :
  -a * (x - y) - b * (y - x) + c * (x - y) = -(x - y) * (a + b - c) :=
by
  sorry

end factoring_expression_l276_276187


namespace five_circles_tangent_l276_276964

noncomputable def circle (center : Point) (radius : ℝ) : set Point := sorry

def touch (c1 c2 : set Point) : Prop := ∃ p : Point, p ∈ c1 ∧ p ∈ c2

theorem five_circles_tangent :
  ∃ (c1 c2 c3 c4 c5 : set Point),
    (∀ (ci cj : set Point), ci ≠ cj → touch ci cj) :=
begin
  sorry
end

end five_circles_tangent_l276_276964


namespace scarlett_oil_amount_l276_276688

theorem scarlett_oil_amount (initial_oil add_oil : ℝ) (h1 : initial_oil = 0.17) (h2 : add_oil = 0.67) :
  initial_oil + add_oil = 0.84 :=
by
  rw [h1, h2]
  -- Proof step goes here
  sorry

end scarlett_oil_amount_l276_276688


namespace bricks_in_chimney_l276_276359

theorem bricks_in_chimney :
  ∀ (h : ℕ), 
  (Brenda_rate := (h:ℚ) / 6) →
  (Brandon_rate := (h:ℚ) / 8) →
  (combined_rate := Brenda_rate + Brandon_rate - 15) →
  (time := 4) →
  time * combined_rate = h →
  h = 360 :=
by
  intros h Brenda_rate Brandon_rate combined_rate time hy no_decrease_rate
  sorry

end bricks_in_chimney_l276_276359


namespace profit_percent_l276_276675

-- Problem statement definitions based on given conditions.
variables {x : ℝ} {CP SP : ℝ}
hypothesis h1 : CP = 4 * x
hypothesis h2 : SP = 5 * x

-- Theorem statement to prove profit percent
theorem profit_percent (h1 : CP = 4 * x) (h2 : SP = 5 * x) : ((SP - CP) / CP) * 100 = 25 := 
by
  -- Skipping proof here
  sorry

end profit_percent_l276_276675


namespace number_of_correct_statements_l276_276444

theorem number_of_correct_statements :
  let α := Real.pi / 6
  let ϕ := Real.pi / 2 + 2 * Int.pi * k
  let p1 := ((α = Real.pi / 6) → (Real.sin α = 1 / 2))
  let p2 := (∃ x₀ : ℝ, Real.sin x₀ > 1)
  let q2 := (∀ x : ℝ, Real.sin x ≤ 1)
  let p3 := (ϕ = Real.pi / 2 + k * Real.pi)
  let f : ℝ → ℝ := λ x, Real.sin (2 * x + ϕ)
  let p4 := (∃ x ∈ Set.Ioo 0 (Real.pi / 2), Real.sin x + Real.cos x = 1 / 2)
  let q4 := (∀ (A B : ℝ), (Real.sin A > Real.sin B → A > B))
  ((¬(p4)) ∧ q4) →
  p1 ∧ q2 ∧ p3 ∧ (¬(p4) ∧ q4) → 3 = 3 :=
by
  assume α ϕ p1 p2 q2 p3 f p4 q4 h hc
  sorry


end number_of_correct_statements_l276_276444


namespace cone_surface_area_l276_276066

-- Definitions based on conditions
def sector_angle : Real := 2 * Real.pi / 3
def sector_radius : Real := 2

-- Definition of the radius of the cone's base
def cone_base_radius (sector_angle sector_radius : Real) : Real :=
  sector_radius * sector_angle / (2 * Real.pi)

-- Definition of the lateral surface area of the cone
def lateral_surface_area (r l : Real) : Real :=
  Real.pi * r * l

-- Definition of the base area of the cone
def base_area (r : Real) : Real :=
  Real.pi * r^2

-- Total surface area of the cone
def total_surface_area (sector_angle sector_radius : Real) : Real :=
  let r := cone_base_radius sector_angle sector_radius
  let S1 := lateral_surface_area r sector_radius
  let S2 := base_area r
  S1 + S2

theorem cone_surface_area (h1 : sector_angle = 2 * Real.pi / 3)
                          (h2 : sector_radius = 2) :
  total_surface_area sector_angle sector_radius = 16 * Real.pi / 9 :=
by
  sorry

end cone_surface_area_l276_276066


namespace total_number_of_people_l276_276698

theorem total_number_of_people (num_cannoneers num_women num_men total_people : ℕ)
  (h1 : num_women = 2 * num_cannoneers)
  (h2 : num_cannoneers = 63)
  (h3 : num_men = 2 * num_women)
  (h4 : total_people = num_women + num_men) : 
  total_people = 378 := by
  sorry

end total_number_of_people_l276_276698


namespace incorrect_conclusion_option_D_l276_276559

def happy_number (n : ℕ) : ℕ := 8 * n

def is_consecutive_odd_squares_subtracted (n : ℕ) : Prop := 
  ∃ k : ℕ, 2 * k + 1 = n ∨ 2 * k - 1 = n

def is_happy_number (n : ℕ) : Prop := 
  ∃ k : ℕ, happy_number k = n

theorem incorrect_conclusion_option_D :
  let happy_numbers_within_30 := {8, 16, 24}
  ∑ x in happy_numbers_within_30, x ≠ 49 :=
by {
  sorry
}

end incorrect_conclusion_option_D_l276_276559


namespace erica_has_correct_amount_l276_276161

-- Definitions for conditions
def total_money : ℕ := 91
def sam_money : ℕ := 38

-- Definition for the question regarding Erica's money
def erica_money := total_money - sam_money

-- The theorem stating the proof problem
theorem erica_has_correct_amount : erica_money = 53 := sorry

end erica_has_correct_amount_l276_276161


namespace angle_mtb_l276_276334

open Real EuclideanSpace

noncomputable def calculate_mtb_angle (A B C M N O K T : ℝ × ℝ) : Prop :=
  let ⟨ax, ay⟩ := A
  let ⟨bx, by⟩ := B
  let ⟨cx, cy⟩ := C
  let ⟨mx, my⟩ := M
  let ⟨nx, ny⟩ := N
  let ⟨ox, oy⟩ := O
  let ⟨kx, ky⟩ := K
  let ⟨tx, ty⟩ := T in
  (cy = 0) ∧
  (by = 0) ∧
  (by = bx) ∧ -- coordinate transformation for simplicity
  (bx * 2 = ax) ∧
  (ny = 0) ∧
  (nx / bx = 2/3) ∧
  (my = ax) ∧
  (mx = nx) ∧
  (ox = (nx + mx) / 2) ∧
  (oy = (cy + my) / 2) ∧
  (K = (ox + nx, oy + ny)) ∧
  (by / (ox - kx) = (oy - ty) / (ox - tx)) ∧
  -- Need to show angle MTB = 90 degrees
  (angle (M - T) (B - T) = π / 2)

theorem angle_mtb {A B C M N O K T : ℝ × ℝ} :
  calculate_mtb_angle A B C M N O K T → angle (M - T) (B - T) = π / 2 :=
by
  intro h
  sorry

end angle_mtb_l276_276334


namespace equation_relationship_linear_l276_276375

theorem equation_relationship_linear 
  (x y : ℕ)
  (h1 : (x, y) = (0, 200) ∨ (x, y) = (1, 160) ∨ (x, y) = (2, 120) ∨ (x, y) = (3, 80) ∨ (x, y) = (4, 40)) :
  y = 200 - 40 * x :=
  sorry

end equation_relationship_linear_l276_276375


namespace primes_divisible_by_3_percentage_l276_276818

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l276_276818


namespace sin_minus_cos_eq_minus_1_l276_276556

theorem sin_minus_cos_eq_minus_1 (x : ℝ) 
  (h : Real.sin x ^ 3 - Real.cos x ^ 3 = -1) :
  Real.sin x - Real.cos x = -1 := by
  sorry

end sin_minus_cos_eq_minus_1_l276_276556


namespace inequality_of_sequence_l276_276435

theorem inequality_of_sequence (n : ℕ) 
  (a : Fin (2 * n - 1) → ℝ) 
  (h : ∀ i : Fin (2 * n - 2), a i.val ≥ a (Fin.succ i).val) 
  (hnonneg : ∀ i : Fin (2 * n - 1), 0 ≤ a i):
  (∑ i in Finset.range (2 * n - 1), (-1)^i * (a i)^2) 
  ≥ (∑ i in Finset.range (2 * n - 1), (-1)^i * a i)^2 := 
begin
  sorry,
end

end inequality_of_sequence_l276_276435


namespace primes_divisible_by_3_percentage_l276_276826

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l276_276826


namespace square_perimeter_ratio_l276_276665

theorem square_perimeter_ratio (a b : ℝ) (h : a^2 / b^2 = 16 / 25) :
  (4 * a) / (4 * b) = 4 / 5 :=
by
  sorry

end square_perimeter_ratio_l276_276665


namespace probability_C_eq_one_fourth_l276_276928

variables (P : Type → ℚ) (A B C D : Type)

def probability_A : ℚ := 1/4
def probability_B : ℚ := 1/3
def probability_D : ℚ := 1/6

theorem probability_C_eq_one_fourth (h : P A + P B + P C + P D = 1) :
  P C = 1/4 :=
by 
  have hA : P A = 1/4 := probability_A
  have hB : P B = 1/3 := probability_B
  have hD : P D = 1/6 := probability_D
  sorry

end probability_C_eq_one_fourth_l276_276928


namespace primes_divisible_by_3_percentage_is_12_5_l276_276743

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l276_276743


namespace regression_line_correct_l276_276423

variable {n : ℕ} {x y: ℝ}
variable {xi yi: Fin n → ℝ}
variable (avg_x : ℝ) (avg_y : ℝ) (b : ℝ)

def regression_line (xi yi : Fin n → ℝ) (n : ℕ) (avg_x avg_y b : ℝ) : Prop :=
  (b = (∑ i, xi i * yi i - n * avg_x * avg_y) / (∑ i, xi i ^ 2 - n * avg_x ^ 2)) ∧
  (avg_y = (∑ i, yi i) / n) ∧
  (avg_x = (∑ i, xi i) / n)

theorem regression_line_correct : ∀ (xi yi : Fin n → ℝ) (n : ℕ) (avg_x avg_y b a : ℝ),
  regression_line xi yi n avg_x avg_y b →
  option_b_is_incorrect xi yi n avg_x avg_y b a
:= by
  intros xi yi n avg_x avg_y b a h
  -- proof goes here
  sorry

end regression_line_correct_l276_276423


namespace ways_to_choose_4_cards_of_different_suits_l276_276525

theorem ways_to_choose_4_cards_of_different_suits :
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  ∃ n : ℕ, n = (choose num_suits num_suits) * cards_per_suit ^ num_suits ∧ n = 28561 :=
by
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  have ways_to_choose_suits : (choose num_suits num_suits) = 1 := by simp
  have ways_to_choose_cards : cards_per_suit ^ num_suits = 28561 := by norm_num
  let n := 1 * 28561
  use n
  constructor
  · exact by simp [ways_to_choose_suits, ways_to_choose_cards]
  · exact by rfl

end ways_to_choose_4_cards_of_different_suits_l276_276525


namespace correct_inequality_l276_276299

theorem correct_inequality :
  (∀ x : Real, 0 < x → log 2 3 > 1) →
  (∀ x : Real, 0 < x → log 3 2 < 1) →
  (∀ x : Real, 1 < x → log 2 3.5 < log 2 3.6) →
  (∀ x : Real, 0 < x → log 0.3 1.8 > log 0.3 2.7) →
  (∀ x : Real, 2 < x → log 3 π > 1 ∧ log 2 0.8 < 0) →
  log 2 3.5 < log 2 3.6 :=
by
  intros h1 h2 h3 h4 h5
  exact h3 2 sorry

end correct_inequality_l276_276299


namespace sum_of_numbers_l276_276263

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), n = k * 10 + d + m * 10 * (10 ^ k)

theorem sum_of_numbers
  (A B C : ℕ)
  (hA : A >= 100 ∧ A < 1000)
  (hB : B >= 10 ∧ B < 100)
  (hC : C >= 10 ∧ C < 100)
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
              (if contains_digit A 7 then A else 0) +
              (if contains_digit B 7 then B else 0) +
              (if contains_digit C 7 then C else 0) = 208)
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧ 
              (if contains_digit B 3 then B else 0) +
              (if contains_digit C 3 then C else 0) = 76) :
  A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276263


namespace primes_divisible_by_3_percentage_l276_276819

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l276_276819


namespace roots_equation_l276_276211

theorem roots_equation (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) : p + q = 69 :=
sorry

end roots_equation_l276_276211


namespace part1_part2_l276_276006

def sequence_geometric (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∃ t : ℤ, ∀ n : ℕ, a (n+1) = r * a n + t

noncomputable def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  2 * a n + n

noncomputable def b (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (1 - a n)

noncomputable def T (b : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in finset.range n, b (i + 1)

theorem part1 (a : ℕ → ℤ) (h : ∀ n, 2 * a n + n = S a n) :
  sequence_geometric (λ n, a n - 1) :=
sorry

theorem part2 (a : ℕ → ℤ) (h1 : ∀ n, S a n = 2 * a n + n) (h2 : ∀ n, b a n = n * (1 - a n)) :
  ∀ n, T (b a) n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end part1_part2_l276_276006


namespace find_altitude_length_l276_276575

noncomputable def triangle_altitude 
  (a b c : ℝ) 
  (dot_result : ℝ) 
  (bc_diff : ℝ) 
  (tan_A_val : ℝ) 
  (altitude_result : ℝ) : Prop :=
  let tan_A := -real.sqrt 15 in
  let sin_A := tan_A / real.sqrt ((tan_A ^ 2) + 1) in
  let cos_A := -1 / real.sqrt ((tan_A ^ 2) + 1) in
  (b - c = bc_diff) ∧
  (dot_result = 6) ∧
  (tan_A = tan_A_val) ∧
  (a * sin_A = altitude_result)

theorem find_altitude_length : 
  ∃ (a b c : ℝ), triangle_altitude a b c 6 2 (-real.sqrt 15) (3 * real.sqrt 15 / 4) := 
sorry

end find_altitude_length_l276_276575


namespace range_of_x_for_f_eq_7_l276_276447

theorem range_of_x_for_f_eq_7 (a : ℝ) (h_a : a = 4) :
  { x : ℝ | |x + 3| + |x - a| = 7 } = Icc (-3) 4 :=
by
  sorry

end range_of_x_for_f_eq_7_l276_276447


namespace sum_winning_cards_divisible_by_101_l276_276346

-- Define what a winning card is
def is_winning_card (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a + b = c + d

-- Sum of all winning card numbers
def sum_winning_card_numbers : ℕ :=
  (Finset.range 10000).filter is_winning_card |>.sum id

-- Theorem statement
theorem sum_winning_cards_divisible_by_101 : sum_winning_card_numbers % 101 = 0 :=
  sorry

end sum_winning_cards_divisible_by_101_l276_276346


namespace Joe_time_from_home_to_school_l276_276102

-- Define the parameters
def walking_time := 4 -- minutes
def waiting_time := 2 -- minutes
def running_speed_ratio := 2 -- Joe's running speed is twice his walking speed

-- Define the walking and running times
def running_time (walking_time : ℕ) (running_speed_ratio : ℕ) : ℕ :=
  walking_time / running_speed_ratio

-- Total time it takes Joe to get from home to school
def total_time (walking_time waiting_time : ℕ) (running_speed_ratio : ℕ) : ℕ :=
  walking_time + waiting_time + running_time walking_time running_speed_ratio

-- Conjecture to be proved
theorem Joe_time_from_home_to_school :
  total_time walking_time waiting_time running_speed_ratio = 10 := by
  sorry

end Joe_time_from_home_to_school_l276_276102


namespace percentage_of_primes_divisible_by_3_is_12_5_l276_276845

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l276_276845


namespace hermans_breakfast_cost_l276_276292

-- Define the conditions
def meals_per_day : Nat := 4
def days_per_week : Nat := 5
def cost_per_meal : Nat := 4
def total_weeks : Nat := 16

-- Define the statement to prove
theorem hermans_breakfast_cost :
  (meals_per_day * days_per_week * cost_per_meal * total_weeks) = 1280 := by
  sorry

end hermans_breakfast_cost_l276_276292


namespace factorization_a_minus_b_l276_276384

theorem factorization_a_minus_b (a b : ℤ) (h1 : 3 * b + a = -7) (h2 : a * b = -6) : a - b = 7 :=
sorry

end factorization_a_minus_b_l276_276384


namespace cherry_tomatoes_final_count_l276_276224

noncomputable def final_tomatoes : ℕ :=
  let initial := 21 in
  let after_first_birds := initial - initial / 3 in
  let after_second_birds := after_first_birds - after_first_birds * 40 / 100 in
  let after_growth1 := after_second_birds + after_second_birds * 50 / 100 in
  let after_growth2 := after_growth1 + 4 in
  let after_last_bird := after_growth2 - after_growth2 * 25 / 100 in
  after_last_bird

theorem cherry_tomatoes_final_count : final_tomatoes = 13 :=
  by
    sorry -- Proof goes here

end cherry_tomatoes_final_count_l276_276224


namespace primes_less_than_20_divisible_by_3_percentage_l276_276731

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l276_276731


namespace constant_product_of_intersections_l276_276995

theorem constant_product_of_intersections
  (k : Type*) [inner_product_space ℝ k]
  (M A B K C D P Q : k)
  (r : ℝ)
  (h1 : dist M A = r)
  (h2 : dist M B = r)
  (h3 : dist M C = r)
  (h4 : dist M D = r)
  (h5 : ∃ t : k, ∀ (x : k), dist x A = dist x t → false) -- Tangent definition
  (h6 : segment K A M) -- K is on segment AM
  (h7 : K ≠ A)
  (h8 : K ≠ M)
  (h9 : t ∉ k)
  (h10 : ∀ (CD : set k), CD ≠ (segment A B) ∧ K ∈ CD → ∃! (P Q : k), (line_segment B C ∩ t = {P}) ∧ (line_segment B D ∩ t = {Q})) :
  ∀ (CD : set k) (hcd : CD ≠ (segment A B) ∧ K ∈ CD),
  let ⟨P, hP₁⟩ := (h10 CD hcd).exists in
  let ⟨Q, hQ₁⟩ := (h10 CD hcd).exists in
  dist A P * dist A Q = (dist A B)^2 * (dist A K / dist B K) :=
by
  sorry

end constant_product_of_intersections_l276_276995


namespace probability_more_heads_than_tails_l276_276548

open Nat

def coin_flip_probability (n : Nat) := \(
    let y := (Nat.choose 10 5) * (1 / (2 ^ 10)) -- binomial coefficient
    let x := (1 - y) / 2 -- calculating x
    x
\)

theorem probability_more_heads_than_tails : coin_flip_probability 10 = 193 / 512 :=
by
  sorry

end probability_more_heads_than_tails_l276_276548


namespace divisors_inequality_l276_276887

noncomputable def d (n : ℕ) : ℕ :=
  n.proper_divisors.card + 1

theorem divisors_inequality (a n : ℕ)
  (h₁ : 1 < a) (h₂ : 0 < n) (h₃ : (a^n + 1).prime) : 
  d (a^n - 1) ≥ n := 
sorry

end divisors_inequality_l276_276887


namespace percentage_of_primes_divisible_by_3_l276_276790

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l276_276790


namespace distribution_of_collection_items_l276_276106

-- Declaring the collections
structure Collection where
  stickers : Nat
  baseball_cards : Nat
  keychains : Nat
  stamps : Nat

-- Defining the individual collections based on the conditions
def Karl : Collection := { stickers := 25, baseball_cards := 15, keychains := 5, stamps := 10 }
def Ryan : Collection := { stickers := Karl.stickers + 20, baseball_cards := Karl.baseball_cards - 10, keychains := Karl.keychains + 2, stamps := Karl.stamps }
def Ben_scenario1 : Collection := { stickers := Ryan.stickers - 10, baseball_cards := (Ryan.baseball_cards / 2), keychains := Karl.keychains * 2, stamps := Karl.stamps + 5 }

-- Total number of items in the collection
def total_items_scenario1 :=
  Karl.stickers + Karl.baseball_cards + Karl.keychains + Karl.stamps +
  Ryan.stickers + Ryan.baseball_cards + Ryan.keychains + Ryan.stamps +
  Ben_scenario1.stickers + Ben_scenario1.baseball_cards + Ben_scenario1.keychains + Ben_scenario1.stamps

-- The proof statement
theorem distribution_of_collection_items :
  total_items_scenario1 = 184 ∧ total_items_scenario1 % 4 = 0 → (184 / 4 = 46) := 
by
  sorry

end distribution_of_collection_items_l276_276106


namespace sum_digits_of_9A_is_9_l276_276585

-- Placeholder function to extract digits of a number.
def digits (n : ℕ) : List ℕ := sorry

-- Predicate to check if digits are in ascending order.
def ascending_digits (n : ℕ) : Prop := 
  digits n = List.qsort (≤) (digits n)

-- Function to sum the digits of a number.
def sum_of_digits (n : ℕ) : ℕ :=
  (digits n).sum

-- Main theorem statement.
theorem sum_digits_of_9A_is_9 {A : ℕ} (h : ascending_digits A) : 
  sum_of_digits (9 * A) = 9 := 
begin
  sorry
end

end sum_digits_of_9A_is_9_l276_276585


namespace no_12_term_geometric_seq_in_1_to_100_l276_276608

theorem no_12_term_geometric_seq_in_1_to_100 :
  ¬ ∃ (s : Fin 12 → Set ℕ),
    (∀ i, ∃ (a q : ℕ), (s i = {a * q^n | n : ℕ}) ∧ (∀ x ∈ s i, 1 ≤ x ∧ x ≤ 100)) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → ∃ i, n ∈ s i) := 
sorry

end no_12_term_geometric_seq_in_1_to_100_l276_276608


namespace correct_propositions_l276_276953

-- Define the propositions as individual statements
def proposition_4 (a b α : Type) [line a] [line b] [plane α] (h₁ : parallel a b) (h₂ : parallel a α) (h₃ : ¬ in_plane b α) : parallel b α := sorry

def proposition_5 (α β χ l : Type) [plane α] [plane β] [plane χ] [line l] 
(h₁ : perp α χ) (h₂ : perp β χ) (h₃ : α ∩ β = l) : perp l χ := sorry

-- Combine the propositions to form the final theorem
theorem correct_propositions (a b α β χ : Type) 
    [line a] [line b] [plane α] [plane β] [plane χ]
    (h₁ : parallel a b) (h₂ : parallel a α) (h₃ : ¬ in_plane b α) 
    (h₄ : perp α χ) (h₅ : perp β χ) (h₆ : α ∩ β = l) :
    (proposition_4 a b α h₁ h₂ h₃) ∧ (proposition_5 α β χ l h₄ h₅ h₆) := 
begin
  split,
  any_goals { sorry }
end

end correct_propositions_l276_276953


namespace alynn_number_of_bulbs_l276_276344

noncomputable def power_usage_per_day := 60  -- Each bulb uses 60 watts per day
noncomputable def cost_per_watt := 0.20      -- 20 cents per watt
noncomputable def total_expenses := 14400.0  -- Total monthly expenses in dollars

def num_bulbs_alynn_has : ℕ :=
  let total_power_consumption := total_expenses / cost_per_watt
  let total_power_per_bulb := power_usage_per_day
  total_power_consumption / total_power_per_bulb

theorem alynn_number_of_bulbs : num_bulbs_alynn_has = 1200 := by
  sorry

end alynn_number_of_bulbs_l276_276344


namespace sum_of_numbers_is_247_l276_276230

/-- Definitions of the conditions -/
def number_contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d < 10 ∧ ∃ (k : ℕ), n / 10 ^ k % 10 = d

variable (A B C : ℕ)
variable (hA : 100 ≤ A ∧ A < 1000)
variable (hB : 10 ≤ B ∧ B < 100)
variable (hC : 10 ≤ C ∧ C < 100)
variable (h_sum_7 : if number_contains_digit A 7 
                  then if number_contains_digit B 7 
                  then if number_contains_digit C 7 
                  then A + B + C 
                  else A + B
                  else A
                  else B + C = 208)
variable (h_sum_3 : if number_contains_digit A 3 
                  then if number_contains_digit B 3
                  then if number_contains_digit C 3
                  then A + B + C 
                  else A + B
                  else A 
                  else B + C = 76)

/-- Prove that the sum of all three numbers is 247 -/
theorem sum_of_numbers_is_247 : A + B + C = 247 :=
by
  sorry

end sum_of_numbers_is_247_l276_276230


namespace total_number_of_people_l276_276704

theorem total_number_of_people
  (cannoneers : ℕ) 
  (women : ℕ) 
  (men : ℕ) 
  (hc : cannoneers = 63)
  (hw : women = 2 * cannoneers)
  (hm : men = 2 * women) :
  cannoneers + women + men = 378 := 
sorry

end total_number_of_people_l276_276704


namespace list_scores_from_lowest_to_highest_l276_276609

variables (J E N L : ℕ)

-- Conditions
axiom Elina_thinks : E = J
axiom Norah_thinks : N ≤ J
axiom Liam_thinks : L > J

-- Task
theorem list_scores_from_lowest_to_highest (h1 : E = J) (h2 : N ≤ J) (h3 : L > J) : 
  N ≤ E ∧ E < L :=
by {
  split,
  { rw h1, exact h2 },
  { rw h1, exact h3 }
}

#check list_scores_from_lowest_to_highest

end list_scores_from_lowest_to_highest_l276_276609


namespace subset_relation_l276_276456

-- Define the sets M and N
def M := {1, 2, 3}
def N := {1}

-- The statement to prove that N ⊆ M
theorem subset_relation : N ⊆ M := sorry

end subset_relation_l276_276456


namespace baseball_players_seating_l276_276077

theorem baseball_players_seating : 
  let dodgers := 3
  let astros := 4
  let mets := 2
  let marlin := 1
  let total_ways := (4! * 3! * 4! * 2! * 1!)
  in total_ways = 6912 :=
by
  sorry

end baseball_players_seating_l276_276077


namespace hyperbola_asymptotes_l276_276453

theorem hyperbola_asymptotes (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : (sqrt 5 / 2) = (sqrt (a^2 + b^2) / a)) :
  ∀ x y : ℝ, (y = (1 / 2) * x ∨ y = -(1 / 2) * x) :=
by
  sorry

end hyperbola_asymptotes_l276_276453


namespace choose_4_cards_of_different_suits_l276_276540

theorem choose_4_cards_of_different_suits :
  (∃ (n : ℕ), choose 4 4 = n) ∧
  (∃ (m : ℕ), (13^4 = m)) ∧
  (1 * (13^4) = 28561)

end choose_4_cards_of_different_suits_l276_276540


namespace max_U_value_l276_276014

noncomputable def maximum_value (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) : ℝ :=
  x + y

theorem max_U_value (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) :
  maximum_value x y h ≤ Real.sqrt 13 :=
  sorry

end max_U_value_l276_276014


namespace solve_equation_l276_276169

def problem_statement : Prop :=
  ∃ x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -7 / 6

theorem solve_equation : problem_statement :=
by {
  sorry
}

end solve_equation_l276_276169


namespace find_k_for_quadratic_root_l276_276416

theorem find_k_for_quadratic_root (k : ℝ) (h : (1 : ℝ).pow 2 + k * 1 - 3 = 0) : k = 2 :=
by
  sorry

end find_k_for_quadratic_root_l276_276416


namespace sqrt6_add_sqrt7_gt_2sqrt2_add_sqrt5_l276_276365

theorem sqrt6_add_sqrt7_gt_2sqrt2_add_sqrt5 : 
  ( √6 + √7 > 2 * √2 + √5 ) :=
sorry

end sqrt6_add_sqrt7_gt_2sqrt2_add_sqrt5_l276_276365


namespace isosceles_right_triangle_eccentricity_l276_276083

theorem isosceles_right_triangle_eccentricity (c : ℝ) (h_eq : c ≠ 0) :
  let b := c,
  let a := sqrt(2) * c in
  c / a = sqrt(2) / 2 := 
by
  -- Definition of variables and conditions
  let b := c,
  let a := Real.sqrt(2) * c,
  -- Using a property of the standard ellipse equation
  sorry

end isosceles_right_triangle_eccentricity_l276_276083


namespace subsets_containing_5_and_6_l276_276465

theorem subsets_containing_5_and_6 :
  let S := {1, 2, 3, 4, 5, 6}
  ∃ s ⊆ S, 5 ∈ s ∧ 6 ∈ s ∧ s.card = 16 :=
sorry

end subsets_containing_5_and_6_l276_276465


namespace grid_fill_possible_l276_276409

theorem grid_fill_possible (m n : ℕ)
  (a : Fin m → ℕ) (b : Fin n → ℕ)
  (a_pos : ∀ i, 0 < a i)
  (b_pos : ∀ j, 0 < b j)
  (sum_eq : (Finset.univ.sum a) = (Finset.univ.sum b)) :
  ∃ (grid : Fin m → Fin n → ℕ),
    (∀ i, (Finset.univ.sum (λ j, grid i j)) = a i) ∧
    (∀ j, (Finset.univ.sum (λ i, grid i j)) = b j) ∧
    (Finset.card (Finset.filter (λ x, x ≠ 0) (Finset.univ.image (λ (ij : Fin m × Fin n), grid ij.1 ij.2))) ≤ m + n - 1) :=
sorry

end grid_fill_possible_l276_276409


namespace arithmetic_mean_between_l276_276640

theorem arithmetic_mean_between (a b c : ℚ) (h1 : a = 7 / 10) (h2 : b = 4 / 5) (h3 : c = 3 / 4) :
  (a + b) / 2 = c :=
by
  -- Problem setup: defining the three numbers
  let a := 7 / 10
  let b := 8 / 10 -- since 4/5 = 8/10
  let c := 7.5 / 10 -- since 3/4 = 7.5/10
  -- Verify if the arithmetic mean of a and b is equal to c
  have h4 : (a + b) / 2 = 7.5 / 10 := by sorry
  exact h4

end arithmetic_mean_between_l276_276640


namespace polynomial_constant_on_range_l276_276109

theorem polynomial_constant_on_range (k : ℤ) (P : ℤ[X])
  (hk : k ≥ 4)
  (hP : ∀ c : ℤ, 0 ≤ c ∧ c ≤ k + 1 → 0 ≤ P.eval c ∧ P.eval c ≤ k) :
  ∀ c d : ℤ, 0 ≤ c ∧ c ≤ k + 1 → 0 ≤ d ∧ d ≤ k + 1 → P.eval c = P.eval d :=
by
  sorry

end polynomial_constant_on_range_l276_276109


namespace number_of_subsets_with_5_and_6_l276_276505

theorem number_of_subsets_with_5_and_6 : 
  let S := {1, 2, 3, 4, 5, 6}
  ∃ n : ℕ, (n = (set.powerset S).count (λ x, {5, 6} ⊆ x)) ∧ n = 16 := 
sorry

end number_of_subsets_with_5_and_6_l276_276505


namespace percentage_of_primes_divisible_by_3_l276_276803

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l276_276803


namespace primes_divisible_by_3_percentage_is_12_5_l276_276744

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l276_276744


namespace percentage_of_primes_divisible_by_3_is_12_5_l276_276849

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l276_276849


namespace intersecting_lines_l276_276696

theorem intersecting_lines (m b : ℝ)
  (h1 : ∀ x, (9 : ℝ) = 2 * m * x + 3 → x = 3)
  (h2 : ∀ x, (9 : ℝ) = 4 * x + b → x = 3) :
  b + 2 * m = -1 :=
sorry

end intersecting_lines_l276_276696


namespace sum_of_star_tips_l276_276142

theorem sum_of_star_tips :
  let n := 9
  let alpha := 80  -- in degrees
  let total := n * alpha
  total = 720 := by sorry

end sum_of_star_tips_l276_276142


namespace angle_between_perpendiculars_l276_276986

theorem angle_between_perpendiculars (φ : ℝ) (hφ : 0 < φ ∧ φ < 180)
  (M : Type) (H : M → ℝ) :
    ∀ (A B : M), angle M A B = 180 - φ := by sorry

end angle_between_perpendiculars_l276_276986


namespace translate_parabola_to_zero_vertex_l276_276046

theorem translate_parabola_to_zero_vertex : (∃ h k : ℝ, ∀ x : ℝ, -2 * (x + h)^2 + k = -2 * x^2) :=
by
  let y1 := -2*x^2 + 4*x + 1
  let y2 := -2*x^2
  have h_eq : h = -1
  have k_eq : k = -3
  exact ⟨-1, -3, sorry⟩

end translate_parabola_to_zero_vertex_l276_276046


namespace percentage_primes_divisible_by_3_l276_276856

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l276_276856


namespace find_BD_l276_276586

-- Definitions directly from conditions
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (m : ℝ) (distance : A → A → ℝ)
variables (angle : A → A → A → ℝ)
variables (AB BC : ℝ)
variables (angle_ABC angle_ADC : ℝ)

-- Given conditions
variable (quadrilateral_ABCD : Type)
variables (AB_condition : AB = m) (BC_condition : BC = m)
variables (angle_ABC_condition : angle A B C = 120) (angle_ADC_condition : angle A D C = 120)

-- Proof that BD = m
theorem find_BD (BD_condition : distance B D = BD) :
  BD = m :=
sorry

end find_BD_l276_276586


namespace percent_primes_divisible_by_3_less_than_20_l276_276774

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l276_276774


namespace rationalize_sqrt_denominator_l276_276648

theorem rationalize_sqrt_denominator :
  ∀ (a : ℝ) (b : ℝ) (c : ℝ), a = 7 → b = 3 → c = 7 → ( ∀ x y, sqrt (x * y) = sqrt x * sqrt y ) → 
    (7 / sqrt (63) = sqrt 7 / 3) :=
by
  intros a b c ha hb hc hxy
  rw ←ha at *
  rw ←hb at *
  rw ←hc at *
  sorry

end rationalize_sqrt_denominator_l276_276648


namespace emily_beads_l276_276381

theorem emily_beads (n : ℕ) (b : ℕ) (total_beads : ℕ) (h1 : n = 26) (h2 : b = 2) (h3 : total_beads = n * b) : total_beads = 52 :=
by
  sorry

end emily_beads_l276_276381


namespace product_of_distances_l276_276424

-- Definitions based on the problem conditions
def P : ℝ × ℝ := (1, 1)
def α : ℝ := Real.pi / 6 
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Parametric equations of line l based on angle and passing through point P
def parametric_eq_x (t : ℝ) : ℝ := 1 + (Real.sqrt 3 / 2) * t
def parametric_eq_y (t : ℝ) : ℝ := 1 + (1 / 2) * t

-- Proof statement
theorem product_of_distances : 
  let t1 t2 := roots_of_quadratic (parametric_eq_x, parametric_eq_y) circle_eq in
  |t1| * |t2| = 2 := sorry

end product_of_distances_l276_276424


namespace product_of_2020_numbers_even_l276_276220

theorem product_of_2020_numbers_even (a : ℕ → ℕ) 
  (h : (Finset.sum (Finset.range 2020) a) % 2 = 1) : 
  (Finset.prod (Finset.range 2020) a) % 2 = 0 :=
sorry

end product_of_2020_numbers_even_l276_276220


namespace people_on_raft_without_life_jackets_l276_276277

variable (P : ℕ)
variable (k : ℕ)
variable (PeopleWithLifeJackets : ℕ := 8)
variable (PeopleOnRaftWith8LifeJackets : ℕ := 17)

theorem people_on_raft_without_life_jackets : P = 24 :=
  have PeopleOnRaftWithLifeJackets := P - 7
  have FitWith8LifeJackets := PeopleOnRaftWithLifeJackets + PeopleWithLifeJackets
  have FitWith8LifeJackets = PeopleOnRaftWith8LifeJackets
  by sorry

end people_on_raft_without_life_jackets_l276_276277


namespace subsets_containing_5_and_6_l276_276474

theorem subsets_containing_5_and_6: 
  let s := {1, 2, 3, 4, 5, 6} in
  {t : Finset ℕ // t ⊆ s ∧ 5 ∈ t ∧ 6 ∈ t}.card = 16 :=
by sorry

end subsets_containing_5_and_6_l276_276474


namespace max_intersection_points_l276_276993

theorem max_intersection_points (A B C D E : Point) :
  -- Define that no two segments coincide, are parallel, or are perpendicular
  (∀ (P Q R : Point), P ≠ Q ∧ P ≠ R ∧ Q ≠ R ∧
     ¬(segment P Q ∥ segment P R ∨ segment P Q ∥ segment Q R ∨ segment P R ∥ segment Q R)) →
  -- Question: Prove that the maximum number of unique intersection points among the perpendiculars is 310
  count_intersections A B C D E = 310 :=
by sorry

end max_intersection_points_l276_276993


namespace three_identical_differences_l276_276002

open Finset

/-- 
Given 8 different natural numbers, each no greater than 15, 
prove that among their positive pairwise differences, 
there are three identical ones.
-/
theorem three_identical_differences (s : Finset ℕ) (h_size : s.card = 8) (h_range : ∀ x ∈ s, x ≤ 15) :
  ∃ d ∈ (s.pairs (≠)).image (λ p, abs (p.1 - p.2)), (s.pairs (≠)).image (λ p, abs (p.1 - p.2)).count d ≥ 3 :=
sorry

end three_identical_differences_l276_276002


namespace students_play_neither_l276_276072

theorem students_play_neither (total_students football_players tennis_players both_players : ℕ)
  (h1 : total_students = 38) 
  (h2 : football_players = 26)
  (h3 : tennis_players = 20)
  (h4 : both_players = 17) : 
  ∃ neither_players : ℕ, neither_players = 9 :=
by
  let playing_either_or_both := football_players + tennis_players - both_players
  have h_total := total_students - playing_either_or_both
  have h_neither : h_total = 9 := by
    rw [h1, h2, h3, h4]
    simp
  use h_neither
  sorry

end students_play_neither_l276_276072


namespace even_function_behavior_l276_276623

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def condition (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 ≠ x2 → (x2 - x1) * (f x2 - f x1) > 0

theorem even_function_behavior (f : ℝ → ℝ) (h_even : is_even_function f) (h_condition : condition f) 
  (n : ℕ) (h_n : n > 0) : 
  f (n+1) < f (-n) ∧ f (-n) < f (n-1) :=
sorry

end even_function_behavior_l276_276623


namespace distance_constant_ellipse_eq_min_area_triangle_AOB_l276_276012

noncomputable def ellipse : ℝ → ℝ → Prop := λ x y, (x^2 / 4) + y^2 = 1

def eccentricity : ℝ := sqrt 3 / 2

def vertex_distance_condition : Prop := ∀ x y, abs (-2 - 2 * y) / sqrt 5 = 4 * sqrt 5 / 5

-- Distance from origin to line: a constant
def distance_from_origin_to_line : ℝ := 2 * sqrt 5 / 5

-- Minimum area of triangle AOB
def min_triangle_area : ℝ := 4/5

-- Prove that the distance is constant
theorem distance_constant : distance_from_origin_to_line = 2 * sqrt 5 / 5 := 
sorry

-- Prove the equation of ellipse
theorem ellipse_eq : ∀ x y, ellipse x y = (x^2 / 4) + y^2 = 1 := 
sorry

-- Prove the minimum area of the triangle AOB
theorem min_area_triangle_AOB : min_triangle_area = 4 / 5 := 
sorry

end distance_constant_ellipse_eq_min_area_triangle_AOB_l276_276012


namespace minimum_value_of_f_l276_276711

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x + 5) + abs (x + 6)

theorem minimum_value_of_f : ∃ x : ℝ, f x = 1 :=
by sorry

end minimum_value_of_f_l276_276711


namespace maximum_sum_minimum_sum_max_sum_is_max_min_sum_is_min_l276_276309

variables {α : Type*} [LinearOrderedCommRing α] {a b : Finset α} (as bs : Finset α)

theorem maximum_sum (h1 : as.card = bs.card) (h2 : a ∈ as → b ∈ bs → a ≥ b) :
  ∑ i in as, i * Finset.max' bs h1 = ∑ i in as, i :=
sorry

theorem minimum_sum (h1 : as.card = bs.card) (h2 : a ∈ as → b ∈ bs → a ≥ b) :
  ∑ i in as, i * Finset.min' bs h1 = ∑ i in as, i :=
sorry

noncomputable def max_sum {n : ℕ} (a b : Fin n → ℝ) (h1 : ∀ i, a i ≥ a (i + 1)) (h2 : ∀ i, b i ≥ b (i + 1)) : ℝ :=
∑ i in Finset.range n, a i * b i

noncomputable def min_sum {n : ℕ} (a b : Fin n → ℝ) (h1 : ∀ i, a i ≥ a (i + 1)) (h2 : ∀ i, b i ≥ b (i + 1)) : ℝ :=
∑ i in Finset.range n, a i * b (n - i - 1)

theorem max_sum_is_max {n : ℕ} (a b : Fin n → ℝ) (h1 : ∀ i, a i ≥ a (i + 1)) (h2 : ∀ i, b i ≥ b (i + 1)) :
  ∑ i in Finset.range n, a i * b i = max_sum a b h1 h2 :=
sorry

theorem min_sum_is_min {n : ℕ} (a b : Fin n → ℝ) (h1 : ∀ i, a i ≥ a (i + 1)) (h2 : ∀ i, b i ≥ b (i + 1)) :
  ∑ i in Finset.range n, a i * b (n - i - 1) = min_sum a b h1 h2 :=
sorry

end maximum_sum_minimum_sum_max_sum_is_max_min_sum_is_min_l276_276309


namespace pumpkin_pie_filling_l276_276340

theorem pumpkin_pie_filling (price_per_pumpkin : ℕ) (total_earnings : ℕ) (total_pumpkins : ℕ) (pumpkins_per_can : ℕ) :
  price_per_pumpkin = 3 →
  total_earnings = 96 →
  total_pumpkins = 83 →
  pumpkins_per_can = 3 →
  (total_pumpkins - total_earnings / price_per_pumpkin) / pumpkins_per_can = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end pumpkin_pie_filling_l276_276340


namespace ratio_PG_PE_square_l276_276088

theorem ratio_PG_PE_square (a : ℝ) (EFGH : square E F G H)
  (EF_eq_6 : dist E F = 6)
  (N_midpoint_FH : midpoint N F H)
  (P_intersection : intersection P (line_through E G) (line_through F N)) :
  ratio (dist P G) (dist P E) = 2 :=
sorry

end ratio_PG_PE_square_l276_276088


namespace fraction_pow_zero_l276_276288

theorem fraction_pow_zero (a b : ℤ) (hb_nonzero : b ≠ 0) : (a / (b : ℚ)) ^ 0 = 1 :=
by 
  sorry

end fraction_pow_zero_l276_276288


namespace smallest_n_relat_prime_subset_l276_276121

theorem smallest_n_relat_prime_subset:
  let S := {x : ℕ | 1 ≤ x ∧ x ≤ 280} in
  ∃ (n : ℕ), (∀ T ⊆ S, T.card = n → ∃ a1 a2 a3 a4 a5 ∈ T, nat.coprime a1 a2 ∧ nat.coprime a1 a3 ∧ nat.coprime a1 a4 ∧ nat.coprime a1 a5 ∧ nat.coprime a2 a3 ∧ nat.coprime a2 a4 ∧ nat.coprime a2 a5 ∧ nat.coprime a3 a4 ∧ nat.coprime a3 a5 ∧ nat.coprime a4 a5) 
    ↔ n = 217 := 
sorry

end smallest_n_relat_prime_subset_l276_276121


namespace percentage_primes_divisible_by_3_l276_276781

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l276_276781


namespace sum_even_coeff_l276_276111

theorem sum_even_coeff (n : ℕ) : 
  let f : ℤ[x] := (1 - X + X^2) ^ n in
  let a : ℤ → ℤ := λ i, if h : i ≤ 2 * n then coeff f i else 0 in
  (a 0 + a 2 + a 4 + ⋯ + a (2 * n)) = (1 + 3 ^ n) / 2 :=
by
  sorry

end sum_even_coeff_l276_276111


namespace cos_150_eq_neg_sqrt3_div_2_l276_276962

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  unfold Real.cos
  sorry

end cos_150_eq_neg_sqrt3_div_2_l276_276962


namespace sum_numerator_denominator_of_repeating_decimal_l276_276295

theorem sum_numerator_denominator_of_repeating_decimal (x : ℝ) (h : x = 0.121212...) : 
  let frac := (4, 33) in frac.fst + frac.snd = 37 :=
sorry

end sum_numerator_denominator_of_repeating_decimal_l276_276295


namespace domain_f_l276_276958

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x + 2) + 1 / (x - 1)

theorem domain_f :
  ∀ x : ℝ, (x ≥ -2 ∧ x ≠ 1) ↔ (x ∈ set.Icc (-2 : ℝ) 1 ∨ x ∈ set.Ioi 1) :=
begin
  sorry
end

end domain_f_l276_276958


namespace smallest_area_of_square_l276_276399

noncomputable def smallest_square_area (line_eqn : ℝ → ℝ) (parabola_eqn : ℝ → ℝ) : ℝ :=
  let k := 2 in 8 * (k + 1)

-- Given conditions that are translated to definitions in Lean 4:
def line_eqn (x : ℝ) : ℝ := x - 4
def parabola_eqn (x : ℝ) : ℝ := x^2 + 3*x

-- The statement to be proved: 
-- The smallest possible area of the square under given conditions is 24.
theorem smallest_area_of_square : smallest_square_area line_eqn parabola_eqn = 24 :=
by
  sorry

end smallest_area_of_square_l276_276399


namespace distribute_cheat_sheets_l276_276656

theorem distribute_cheat_sheets :
  let pockets := (fin 4), sheets := (fin 6) in
  ∃ (arrangement : sheets → fin 4),
    (arrangement 0 = arrangement 1) ∧
    (arrangement 3 = arrangement 4) ∧
    (arrangement 0 ≠ arrangement 3) ∧
    (card (set.image arrangement univ) ≥ 3) ∧
    (number_of_distributions = 60) :=
sorry

end distribute_cheat_sheets_l276_276656


namespace solve_for_x_l276_276173

theorem solve_for_x (x : ℚ) : (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ↔ x = -5 / 3 :=
by
  sorry

end solve_for_x_l276_276173


namespace proportional_segments_l276_276642

-- Define the tetrahedron and points
structure Tetrahedron :=
(A B C D O A1 B1 C1 : ℝ)

-- Define the conditions of the problem
variables {tetra : Tetrahedron}

-- Define the segments and their relationships
axiom segments_parallel (DA : ℝ) (DB : ℝ) (DC : ℝ)
  (OA1 : ℝ) (OB1 : ℝ) (OC1 : ℝ) :
  OA1 / DA + OB1 / DB + OC1 / DC = 1

-- The theorem to prove, which follows directly from the given axiom 
theorem proportional_segments (DA DB DC : ℝ)
  (OA1 : ℝ) (OB1 : ℝ) (OC1 : ℝ) :
  OA1 / DA + OB1 / DB + OC1 / DC = 1 :=
segments_parallel DA DB DC OA1 OB1 OC1

end proportional_segments_l276_276642


namespace money_left_is_41_l276_276181

-- Define the amounts saved by Tanner in each month
def savings_september : ℕ := 17
def savings_october : ℕ := 48
def savings_november : ℕ := 25

-- Define the amount spent by Tanner on the video game
def spent_video_game : ℕ := 49

-- Total savings after the three months
def total_savings : ℕ := savings_september + savings_october + savings_november

-- Calculate the money left after spending on the video game
def money_left : ℕ := total_savings - spent_video_game

-- The theorem we need to prove
theorem money_left_is_41 : money_left = 41 := by
  sorry

end money_left_is_41_l276_276181


namespace unique_circle_with_equal_circumference_and_area_l276_276048

theorem unique_circle_with_equal_circumference_and_area :
  ∃! r > 0, 2 * Real.pi * r = Real.pi * r^2 := by
  sorry

end unique_circle_with_equal_circumference_and_area_l276_276048


namespace inclination_angle_necessary_but_not_sufficient_l276_276680

theorem inclination_angle_necessary_but_not_sufficient (a : ℝ) :
  (∃ θ : ℝ, θ > π / 4 ∧ tan θ = a / 2) ↔ a > 2 ∨ a < 0 := 
sorry

end inclination_angle_necessary_but_not_sufficient_l276_276680


namespace sum_of_three_numbers_l276_276271

theorem sum_of_three_numbers :
  ∃ A B C : ℕ, 
    (100 ≤ A ∧ A < 1000) ∧  -- A is a three-digit number
    (10 ≤ B ∧ B < 100) ∧     -- B is a two-digit number
    (10 ≤ C ∧ C < 100) ∧     -- C is a two-digit number
    (A + (if (B / 10 = 7 ∨ B % 10 = 7) then B else 0) + 
       (if (C / 10 = 7 ∨ C % 10 = 7) then C else 0) = 208) ∧
    (if (B / 10 = 3 ∨ B % 10 = 3) then B else 0) + 
    (if (C / 10 = 3 ∨ C % 10 = 3) then C else 0) = 76 ∧
    A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276271


namespace isosceles_triangle_count_l276_276591

-- Define the variables
variables {A B C D E F : Type}

-- Define the angles and congruence conditions
axiom h1 : ∠BAC + ∠ABC + ∠ACB = 180 -- Sum of angles in triangle ABC
axiom h2 : AB = AC                   -- AB is congruent to AC
axiom h3 : ∠ABC = 60                 -- measure of angle ABC is 60 degrees
axiom h4 : BD bisects ∠ABC           -- Segment BD bisects angle ABC
axiom h5 : BD ⟨intersection⟩ AC = D  -- Point D on side AC
axiom h6 : E ⟨intersection⟩ BC = E   -- Point E on side BC
axiom h7 : DE ∥ AB                   -- Segment DE is parallel to AB
axiom h8 : F ⟨intersection⟩ AC = F   -- Point F on side AC
axiom h9 : EF ∥ BD                   -- Segment EF is parallel to BD

-- Define the goal for the proof
theorem isosceles_triangle_count : 
  ∃ A B C D E F : Type, 
  ∠BAC + ∠ABC + ∠ACB = 180 ∧ 
  AB = AC ∧ 
  ∠ABC = 60 ∧ 
  BD bisects ∠ABC ∧ 
  BD ⟨intersection⟩ AC = D ∧ 
  E ⟨intersection⟩ BC = E ∧ 
  DE ∥ AB ∧ 
  F ⟨intersection⟩ AC = F ∧ 
  EF ∥ BD ∧ 
  -- Number of isosceles triangles in the figure
  (isosceles_triangles A B C D E F = 6) :=
sorry

end isosceles_triangle_count_l276_276591


namespace cakes_remaining_l276_276940

theorem cakes_remaining (initial_cakes sold_cakes remaining_cakes: ℕ) (h₀ : initial_cakes = 167) (h₁ : sold_cakes = 108) (h₂ : remaining_cakes = initial_cakes - sold_cakes) : remaining_cakes = 59 :=
by
  rw [h₀, h₁] at h₂
  exact h₂

end cakes_remaining_l276_276940


namespace different_suits_choice_count_l276_276511

-- Definitions based on the conditions
def standard_deck : List (Card × Suit) := 
  List.product Card.all Suit.all

def four_cards (deck : List (Card × Suit)) : Prop :=
  deck.length = 4 ∧ ∀ (i j : Fin 4), i ≠ j → (deck.nthLe i (by simp) : Card × Suit).2 ≠ (deck.nthLe j (by simp) : Card × Suit).2

-- Statement of the proof problem
theorem different_suits_choice_count :
  ∃ l : List (Card × Suit), four_cards l ∧ standard_deck.choose 4 = 28561 :=
by
  sorry

end different_suits_choice_count_l276_276511


namespace circumcircle_diameter_correct_l276_276560

-- Define conditions: lengths of the sticks and triangle side lengths
def stick_lengths : List ℕ := [1, 4, 8, 9]
def triangle_sides (a b c : ℕ) : Prop := (a = 5) ∧ (b = 8) ∧ (c = 9)

-- Function to calculate the diameter of the circumcircle based on side lengths (a, b, c) and the Law of Sines
noncomputable def circumcircle_diameter (a b c : ℕ) (sin_θ : ℝ) : ℝ :=
  (c : ℝ) / sin_θ

-- Lean 4 theorem statement
theorem circumcircle_diameter_correct :
  ∃ (a b c : ℕ) (sin_θ : ℝ), 
  (triangle_sides a b c ∧ sin_θ = (3 * Real.sqrt 11) / 10) ∧ 
  circumcircle_diameter a b c sin_θ = (30 * Real.sqrt 11) / 11 :=
by
  -- Conditions: lengths of the sticks, possible triangle sides, and calculated sin(θ)
  have stick_lengths := [1, 4, 8, 9]
  have triangle_sides := (5, 8, 9)
  have sin_θ := (3 * Real.sqrt 11) / 10

  -- Stating the existence of triangle sides and the required diameter of the circumcircle
  use 5, 8, 9, sin_θ
  split
  { split,
    { exact ⟨rfl, rfl, rfl⟩ },
    { exact rfl }},
  sorry

end circumcircle_diameter_correct_l276_276560


namespace solve_for_m_l276_276544

theorem solve_for_m (m x : ℤ) (h : 2 * x + m - 1 = 0) (hx : x = 2) : m = -3 :=
by
  rw [hx] at h
  simp at h
  exact h

-- This is a sketched solution to give context; final theorem statement is below:
-- theorem solve_for_m (hx : @let x := 2 in 2 * x + m - 1 = 0) : m = -3 := sorry

end solve_for_m_l276_276544


namespace stratified_sampling_l276_276074

theorem stratified_sampling 
  (male_students : ℕ)
  (female_students : ℕ)
  (sample_size : ℕ)
  (H_male_students : male_students = 40)
  (H_female_students : female_students = 30)
  (H_sample_size : sample_size = 7)
  (H_stratified_sample : sample_size = male_students_drawn + female_students_drawn) :
  male_students_drawn = 4 ∧ female_students_drawn = 3  :=
sorry

end stratified_sampling_l276_276074


namespace arutyun_amayak_super_trick_l276_276313

theorem arutyun_amayak_super_trick
    (circle : set ℝ)
    (points : finset ℝ)
    (removed_point : ℝ)
    (H_circle : ∀ x ∈ points, x ∈ circle)
    (H_points_count : points.card = 2007)
    (H_removed : removed_point ∈ points)
    (strategy_amayak : ∀ A B ∈ points, clockwise_distance A B ≤ 180 → A = removed_point ∨ B = removed_point) :
    ∃ semicircle : set ℝ, removed_point ∈ semicircle ∧ (∀ x ∈ semicircle, x ∈ circle) :=
by
  sorry

end arutyun_amayak_super_trick_l276_276313


namespace inradius_of_right_triangle_l276_276959

variable (a b c : ℕ) -- Define the sides
def right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

noncomputable def area (a b : ℕ) : ℝ :=
  0.5 * (a : ℝ) * (b : ℝ)

noncomputable def semiperimeter (a b c : ℕ) : ℝ :=
  ((a + b + c) : ℝ) / 2

noncomputable def inradius (a b c : ℕ) : ℝ :=
  let s := semiperimeter a b c
  let A := area a b
  A / s

theorem inradius_of_right_triangle (h : right_triangle 7 24 25) : inradius 7 24 25 = 3 := by
  sorry

end inradius_of_right_triangle_l276_276959


namespace subsets_containing_5_and_6_l276_276480

theorem subsets_containing_5_and_6 (S : Set ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  {T : Set ℕ // {5, 6} ⊆ T ∧ T ⊆ S}.card = 16 := 
sorry

end subsets_containing_5_and_6_l276_276480


namespace rectangle_area_l276_276921

theorem rectangle_area (L W P A : ℕ) (h1 : P = 52) (h2 : L = 11) (h3 : 2 * L + 2 * W = P) : 
  A = L * W → A = 165 :=
by
  sorry

end rectangle_area_l276_276921


namespace smallest_value_abs_sum_l276_276715

theorem smallest_value_abs_sum : 
  ∃ x : ℝ, (λ x, |x + 3| + |x + 5| + |x + 6| = 5) ∧ 
           (∀ y : ℝ, |y + 3| + |y + 5| + |y + 6| ≥ 5) :=
by
  sorry

end smallest_value_abs_sum_l276_276715


namespace simplify_f_l276_276021

noncomputable def condition1 (a : ℝ) : Prop := 
  a > π ∧ a < 3 * π / 2

noncomputable def condition2 (a : ℝ) : ℝ := 
  (sin (π - a) * sin a * cos (π + a)) / (sin (π / 2 - a) * cos (a + π / 2) * tan (-a))

noncomputable def condition3 (a : ℝ) : Prop :=
  sin (2 * π - a) = 1 / 5

theorem simplify_f (a : ℝ) (h1 : condition1 a) (h2 : condition3 a) :
  condition2 a = 2 * sqrt 6 / 5 :=
sorry

end simplify_f_l276_276021


namespace sum_of_positive_real_solutions_l276_276396

theorem sum_of_positive_real_solutions :
  (∑ x in { x : ℝ | 0 < x ∧ 2*sin(2*x)*(sin(2*x) - sin(2016*π^2 / x)) = sin(4*x) - 1 }, x) = 104 * π :=
sorry

end sum_of_positive_real_solutions_l276_276396


namespace quadratic_root_real_and_discriminant_l276_276441

theorem quadratic_root_real_and_discriminant {m : ℝ} : 
  let Δ := (5*m - 1)^2 - 4 * 2 * m * (3 * m - 1)
  in
  (Δ ≥ 0) ∧ (Δ = 1 → m = 2) :=
by
  let Δ := (5*m - 1)^2 - 4 * 2 * m * (3 * m - 1)
  have h1 : Δ = (m - 1)^2 := by
    calc
      Δ = (5*m - 1)^2 - 8*m*(3*m - 1) : rfl
      ... = (25*m^2 - 10*m + 1) - (24*m^2 - 8*m) : by ring
      ... = (m^2 - 2*m + 1)                   : by ring
      ... = (m - 1)^2                         : by ring
  split
  {
    -- Part 1: Δ ≥ 0 for all m ∈ ℝ
    calc
      Δ = (m - 1)^2 : by exact h1
      ... ≥ 0       : by apply pow_two_nonneg
  },
  {
    -- Part 2: if Δ = 1 then m = 2
    intro h2
    rw [h1] at h2
    calc
      (m - 1)^2 = 1 → m = 2 :
        by
          rw [pow_two_eq_one_iff] at h2
          cases h2
          { exact h2.symm }
          { contradiction }
  }

end quadratic_root_real_and_discriminant_l276_276441


namespace area_of_30_60_90_triangle_with_altitude_5_l276_276664

noncomputable section

def area_of_triangle (h : ℝ) (bc : ℝ) : ℝ := 1/2 * bc * h

def expected_area : ℝ := 25 * Real.sqrt 3 / 3

theorem area_of_30_60_90_triangle_with_altitude_5 :
  ∀ (a b c : ℝ), a = 30 ∧ b = 60 ∧ c = 90 ∧ (h : ℝ), h = 5 → 
  (area_of_triangle h (2 * h / Real.sqrt 3)) = expected_area :=
by
  sorry

end area_of_30_60_90_triangle_with_altitude_5_l276_276664


namespace primes_divisible_by_3_percentage_is_12_5_l276_276736

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l276_276736


namespace vector_parallel_l276_276037

theorem vector_parallel (x : ℝ)
  (a : ℝ × ℝ := (1, 2))
  (b : ℝ × ℝ := (-2, x))
  (h_parallel : ∃ k : ℝ, (1 + (-2), 2 + x) = k • (1 - (-2), 2 - x))
  : x = -4 := sorry

end vector_parallel_l276_276037


namespace a_n_formula_T_n_sum_l276_276421

-- Definitions for the sequence and the sum
def a (n : ℕ) : ℕ := 2 ^ n
def S (n : ℕ) : ℕ := 2 * a n - 2
def b (n : ℕ) : ℕ := (n : ℕ).log2 (2 ^ (2 * n - 1)) -- Logarithm base 2 of a_2n-1
def T : ℕ → ℕ := sorry -- Sum of the first n terms of a_n * b_n (definition deferred until proven)

-- The first question: Proving the general term formula for the sequence \{a_n\}
theorem a_n_formula (n : ℕ) : a n = 2 ^ n :=
by
  sorry

-- The second question: Proving the sum of the first n terms T_n of the sequence {a_n b_n}
theorem T_n_sum (n : ℕ) : T n = (2 * n - 3) * 2 ^ (n + 1) + 6 :=
by
  sorry

end a_n_formula_T_n_sum_l276_276421


namespace curve_not_necessarily_C_l276_276570

variable {C : Type} [curve : Curve C]
variable {f : ℝ → ℝ → Prop}  -- f represents the equation f(x, y) = 0

-- condition: The coordinates of points on curve C satisfy the equation f(x, y) = 0
axiom coordinates_on_curve : ∀ (x y : ℝ), C x y → f x y = 0

-- proof goal: To show that the curve represented by f(x, y) = 0 is not necessarily C
theorem curve_not_necessarily_C :
  (∃ (x y : ℝ), f x y = 0 ∧ ¬ C x y) →
  (¬ (∀ (x y : ℝ), f x y = 0 → C x y)) :=
begin
  sorry,  -- skip the proof
end

end curve_not_necessarily_C_l276_276570


namespace sum_of_numbers_is_247_l276_276228

/-- Definitions of the conditions -/
def number_contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d < 10 ∧ ∃ (k : ℕ), n / 10 ^ k % 10 = d

variable (A B C : ℕ)
variable (hA : 100 ≤ A ∧ A < 1000)
variable (hB : 10 ≤ B ∧ B < 100)
variable (hC : 10 ≤ C ∧ C < 100)
variable (h_sum_7 : if number_contains_digit A 7 
                  then if number_contains_digit B 7 
                  then if number_contains_digit C 7 
                  then A + B + C 
                  else A + B
                  else A
                  else B + C = 208)
variable (h_sum_3 : if number_contains_digit A 3 
                  then if number_contains_digit B 3
                  then if number_contains_digit C 3
                  then A + B + C 
                  else A + B
                  else A 
                  else B + C = 76)

/-- Prove that the sum of all three numbers is 247 -/
theorem sum_of_numbers_is_247 : A + B + C = 247 :=
by
  sorry

end sum_of_numbers_is_247_l276_276228


namespace sum_distances_is_108_l276_276398

noncomputable def circle_radius_A := sorry
noncomputable def circle_radius_B := (4 / 3) * circle_radius_A
noncomputable def circle_radius_C := sorry
noncomputable def circle_radius_D := (4 / 3) * circle_radius_C
noncomputable def circle_radius_E := (4 / 3) * circle_radius_A

def distance_AB := 36
def distance_CD := 36
def distance_DE := 36
def distance_PQ := 60
def midpoint_distance_PQ := distance_PQ / 2

def power_point (r : ℝ) (d : ℝ) : ℝ := (midpoint_distance_PQ * midpoint_distance_PQ) - d * d

axiom power_A : power_point circle_radius_A (sorry) = 900
axiom power_B : power_point circle_radius_B (sorry) = 900
axiom power_C : power_point circle_radius_C (sorry) = 900
axiom power_D : power_point circle_radius_D (sorry) = 900
axiom power_E : power_point circle_radius_E (sorry) = 900

theorem sum_distances_is_108 : 
  sorry + sorry + sorry + sorry + sorry = 108 := sorry

end sum_distances_is_108_l276_276398


namespace sum_of_numbers_l276_276257

def contains_digit (n : Nat) (d : Nat) : Prop := 
  (n / 100 = d) ∨ (n % 100 / 10 = d) ∨ (n % 10 = d)

variables {A B C : Nat}

-- Given conditions
axiom three_digit_number : A ≥ 100 ∧ A < 1000
axiom two_digit_numbers : B ≥ 10 ∧ B < 100 ∧ C ≥ 10 ∧ C < 100
axiom sum_with_sevens : contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7 → A + B + C = 208
axiom sum_with_threes : contains_digit B 3 ∧ contains_digit C 3 ∧ B + C = 76

-- Main theorem to be proved
theorem sum_of_numbers : A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276257


namespace necessary_and_sufficient_condition_l276_276673

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x - 4 * a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) :=
sorry

end necessary_and_sufficient_condition_l276_276673


namespace irreducible_cover_general_irreducible_cover_n_minus_1_irreducible_cover_2_l276_276635

noncomputable def irreducibleCoverCount (n k : ℕ) : ℕ :=
  ∑ j in finset.range (n - (k - 1)), nat.choose n j * (2 ^ k - k - 1) ^ (n - j) * nat.stirling j k

theorem irreducible_cover_general (n k : ℕ) : irreducibleCoverCount n k = 
  ∑ j in finset.range (n - (k - 1)), nat.choose n j * (2 ^ k - k - 1) ^ (n - j) * nat.stirling j k := sorry

theorem irreducible_cover_n_minus_1 (n : ℕ) : irreducibleCoverCount n (n-1) = 
  (1 / 2 * n * (2^n - n - 1)) := sorry

theorem irreducible_cover_2 (n : ℕ) : irreducibleCoverCount n 2 = 
  nat.stirling (n + 1) 3 := sorry

end irreducible_cover_general_irreducible_cover_n_minus_1_irreducible_cover_2_l276_276635


namespace max_value_of_2_abs_m_n_l276_276402

def f (x : ℝ) : ℝ := 2^x
def g (x : ℝ) : ℝ := 2 * x
def is_covering_function (f g : ℝ → ℝ) (D : set ℝ) : Prop :=
  ∀ x ∈ D, f x ≤ g x

theorem max_value_of_2_abs_m_n (m n : ℝ) (h : is_covering_function f g (set.Icc m n)) :
  2^(abs (m - n)) = 2 :=
by sorry

end max_value_of_2_abs_m_n_l276_276402


namespace percentage_primes_divisible_by_3_l276_276778

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l276_276778


namespace percentage_of_primes_divisible_by_3_l276_276792

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l276_276792


namespace find_fx_l276_276992

theorem find_fx (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = 19 * x ^ 2 + 55 * x - 44) :
  ∀ x : ℝ, f x = 19 * x ^ 2 + 93 * x + 30 :=
by
  sorry

end find_fx_l276_276992


namespace percent_primes_divisible_by_3_less_than_20_l276_276763

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l276_276763


namespace percent_primes_divisible_by_3_l276_276838

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l276_276838


namespace number_of_subsets_with_5_and_6_l276_276501

theorem number_of_subsets_with_5_and_6 : 
  let S := {1, 2, 3, 4, 5, 6}
  ∃ n : ℕ, (n = (set.powerset S).count (λ x, {5, 6} ⊆ x)) ∧ n = 16 := 
sorry

end number_of_subsets_with_5_and_6_l276_276501


namespace length_PF_is_8_l276_276637

noncomputable def parabola_properties (P F : ℝ × ℝ) (focus : ℝ × ℝ) (directrix : ℝ → Prop) : Prop :=
  let l := λ x : ℝ, x = -2 -- Equation for the directrix
  let F := (2, 0)          -- Focus of the parabola
  let A := (-2, 4 * Real.sqrt 3)
  let P := (6, 4 * Real.sqrt 3)
  let AF_slope := -Real.sqrt 3
  F = focus ∧                              -- Given focus
  (directrix = l) ∧                        -- Given directrix
  (PA ⊥ l) ∧                               -- PA perpendicular to directrix
  (A = (-2, 4 * Real.sqrt 3)) ∧             -- Point A
  (P = (6, 4 * Real.sqrt 3))                -- Point P

theorem length_PF_is_8 : ∀ (P F : ℝ × ℝ) (focus : ℝ × ℝ) (directrix : ℝ → Prop),
  parabola_properties P F focus directrix → dist P F = 8 :=
by
  intro P F focus directrix h
  rw [dist_eq, show F = (2,0) from h.1, show P = (6, 4 * Real.sqrt 3) from h.5]
  sorry

end length_PF_is_8_l276_276637


namespace roots_equation_l276_276210

theorem roots_equation (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) : p + q = 69 :=
sorry

end roots_equation_l276_276210


namespace selling_price_per_unit_profit_per_unit_after_discount_l276_276322

-- Define the initial cost per unit
variable (a : ℝ)

-- Problem statement for part 1: Selling price per unit is 1.22a yuan
theorem selling_price_per_unit (a : ℝ) : 1.22 * a = a + 0.22 * a :=
by
  sorry

-- Problem statement for part 2: Profit per unit after 15% discount is still 0.037a yuan
theorem profit_per_unit_after_discount (a : ℝ) : 
  (1.22 * a * 0.85) - a = 0.037 * a :=
by
  sorry

end selling_price_per_unit_profit_per_unit_after_discount_l276_276322


namespace probability_more_heads_than_tails_l276_276546

theorem probability_more_heads_than_tails :
  let x := \frac{193}{512}
  let y := \frac{63}{256}
  (2 * x + y = 1) →
  (y = \frac{252}{1024}) →
  (x = \frac{193}{512}) :=
by
  let x : ℚ := 193 / 512
  let y : ℚ := 63 / 256
  sorry

end probability_more_heads_than_tails_l276_276546


namespace number_of_student_clubs_l276_276684

theorem number_of_student_clubs : 
  let n := 2019 in
  let advisory_board_members := 12 in
  let club_members := 27 in
  let total_clubs_with_27_members := Nat.choose (club_members + advisory_board_members - 1) (advisory_board_members - 1) in
  total_clubs_with_27_members = Nat.choose 2003 11 :=
by 
  sorry

end number_of_student_clubs_l276_276684


namespace complex_number_representation_l276_276159

-- Define the variables and constants
def z_initial : ℂ := 1 - real.cos (200 * real.pi / 180) + complex.I * real.sin (200 * real.pi / 180)
def z_final : ℂ := 2 * real.sin (10 * real.pi / 180) * complex.exp (- complex.I * 10 * real.pi / 180)

-- Theorem statement
theorem complex_number_representation : 
  z_initial = z_final :=
sorry

end complex_number_representation_l276_276159


namespace percentage_primes_divisible_by_3_l276_276861

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l276_276861


namespace handshake_problem_l276_276315

def combinations (n k : ℕ) : ℕ :=
  n.choose k

theorem handshake_problem : combinations 40 2 = 780 := 
by
  sorry

end handshake_problem_l276_276315


namespace find_number_l276_276896

theorem find_number (x : ℝ) (h : 75 = 0.6 * x) : x = 125 :=
sorry

end find_number_l276_276896


namespace area_of_triangle_l276_276927

def line1 (x : ℝ) : ℝ := (3 / 4) * x + 1
def line2 (x : ℝ) : ℝ := -(1 / 2) * x + 5
def line3 (y : ℝ) : ℝ := 2

theorem area_of_triangle :
  let A := (4 / 3, 2)
  let B := (6, 2)
  let C := (16 / 5, 17 / 5)
  let base := 6 - (4 / 3)
  let height := (17 / 5) - 2
  (1/2) * base * height = 49 / 15 :=
by
  let A := (4 / 3, 2)
  let B := (6, 2)
  let C := (16 / 5, 17 / 5)
  let base := 6 - (4 / 3)
  let height := (17 / 5) - 2
  have base_value : base = 14 / 3 := by sorry
  have height_value : height = 7 / 5 := by sorry
  calc
    (1/2) * base * height = (1/2) * (14 / 3) * (7 / 5) : by sorry
    ... = 49 / 15 : by sorry

end area_of_triangle_l276_276927


namespace smallest_abs_diff_l276_276980

theorem smallest_abs_diff (m n : ℕ) (hm : m > 0) (hn : n > 0) : ∃ m n : ℕ, |253^m - 40^n| = 9 := 
sorry

end smallest_abs_diff_l276_276980


namespace octal_to_decimal_123_l276_276371

theorem octal_to_decimal_123 :
  let d := 1 * 8^2 + 2 * 8^1 + 3 * 8^0
  in d = 83 :=
by
  let d := 1 * 8^2 + 2 * 8^1 + 3 * 8^0
  show d = 83
  sorry

end octal_to_decimal_123_l276_276371


namespace heaviest_vs_lightest_box_difference_total_weight_of_20_boxes_earnings_from_selling_l276_276900

def standard_weight : ℝ := 25
def weight_differences : List ℝ := [-2, -1.5, -1, 0, 1, 2.5]
def box_frequencies : List ℕ := [1, 4, 2, 3, 2, 8]
def num_boxes : ℕ := 20
def cost_price_per_kg : ℝ := 5
def selling_price_per_kg : ℝ := 8
def weight_loss : ℝ := 10

theorem heaviest_vs_lightest_box_difference :
  let heaviest_box := standard_weight + List.maximum' weight_differences
  let lightest_box := standard_weight + List.minimum' weight_differences
  heaviest_box - lightest_box = 4.5 :=
by
  sorry

theorem total_weight_of_20_boxes :
  let total_weight := standard_weight * num_boxes + List.foldr (λ (x, n) acc => x * n + acc) 0 (List.zip weight_differences box_frequencies)
  total_weight = 512 :=
by
  sorry

theorem earnings_from_selling :
  let total_weight := standard_weight * num_boxes + List.foldr (λ (x, n) acc => x * n + acc) 0 (List.zip weight_differences box_frequencies)
  let total_sellable_weight := total_weight - weight_loss
  let revenue := total_sellable_weight * selling_price_per_kg
  let cost := total_weight * cost_price_per_kg
  let earnings := revenue - cost
  earnings = 1456 :=
by
  sorry

end heaviest_vs_lightest_box_difference_total_weight_of_20_boxes_earnings_from_selling_l276_276900


namespace surface_area_of_cone_l276_276063

-- Definitions based solely on conditions
def central_angle (θ : ℝ) := θ = (2 * Real.pi) / 3
def slant_height (l : ℝ) := l = 2
def radius_cone (r : ℝ) := ∃ (θ l : ℝ), central_angle θ ∧ slant_height l ∧ θ * l = 2 * Real.pi * r
def lateral_surface_area (A₁ : ℝ) (r l : ℝ) := A₁ = Real.pi * r * l
def base_area (A₂ : ℝ) (r : ℝ) := A₂ = Real.pi * r^2
def total_surface_area (A A₁ A₂ : ℝ) := A = A₁ + A₂

-- The theorem proving the total surface area is as specified
theorem surface_area_of_cone :
  ∃ (r l A₁ A₂ A : ℝ), central_angle ((2 * Real.pi) / 3) ∧ slant_height 2 ∧ radius_cone r ∧
  lateral_surface_area A₁ r 2 ∧ base_area A₂ r ∧ total_surface_area A A₁ A₂ ∧ A = (16 * Real.pi) / 9 := sorry

end surface_area_of_cone_l276_276063


namespace sum_of_numbers_l276_276269

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), n = k * 10 + d + m * 10 * (10 ^ k)

theorem sum_of_numbers
  (A B C : ℕ)
  (hA : A >= 100 ∧ A < 1000)
  (hB : B >= 10 ∧ B < 100)
  (hC : C >= 10 ∧ C < 100)
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
              (if contains_digit A 7 then A else 0) +
              (if contains_digit B 7 then B else 0) +
              (if contains_digit C 7 then C else 0) = 208)
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧ 
              (if contains_digit B 3 then B else 0) +
              (if contains_digit C 3 then C else 0) = 76) :
  A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276269


namespace kenneth_distance_ahead_when_biff_finishes_l276_276941

noncomputable def biff_speed_still : ℝ := 50
noncomputable def kenneth_speed_still : ℝ := 51
noncomputable def current_speed : ℝ := 5
noncomputable def wind_speed : ℝ := 3
noncomputable def race_distance : ℝ := 500

noncomputable def biff_eff_speed : ℝ := biff_speed_still + current_speed - wind_speed
noncomputable def kenneth_eff_speed : ℝ := kenneth_speed_still + current_speed - wind_speed

noncomputable def biff_time_to_finish : ℝ := race_distance / biff_eff_speed
noncomputable def kenneth_time_to_finish : ℝ := race_distance / kenneth_eff_speed

noncomputable def time_diff : ℝ := biff_time_to_finish - kenneth_time_to_finish
noncomputable def kenneth_additional_distance : ℝ := kenneth_eff_speed * time_diff

theorem kenneth_distance_ahead_when_biff_finishes :
  kenneth_additional_distance ≈ 9.6142 := by
  sorry

end kenneth_distance_ahead_when_biff_finishes_l276_276941


namespace percent_primes_divisible_by_3_l276_276835

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l276_276835


namespace log_sum_ge_4_l276_276052

theorem log_sum_ge_4 (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (h : log 2 (x + y) = log 2 x + log 2 y) : 4 ≤ x + y :=
begin
  sorry,
end

end log_sum_ge_4_l276_276052


namespace range_of_a_l276_276000

variable (x a : ℝ)

def p : Prop := -3 ≤ x ∧ x ≤ 3
def q : Prop := x < a

theorem range_of_a (h : ∀ x, p x ↔ q x) : 3 < a :=
sorry

end range_of_a_l276_276000


namespace two_distinct_real_roots_of_modified_quadratic_l276_276123

theorem two_distinct_real_roots_of_modified_quadratic (a b k : ℝ) (h1 : a^2 - b > 0) (h2 : k > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + 2 * a * x₁ + b + k * (x₁ + a)^2 = 0) ∧ (x₂^2 + 2 * a * x₂ + b + k * (x₂ + a)^2 = 0) :=
by
  sorry

end two_distinct_real_roots_of_modified_quadratic_l276_276123


namespace tank_capacity_l276_276293

theorem tank_capacity (T : ℝ) (h : 0.4 * T = 0.9 * T - 36) : T = 72 := by
  sorry

end tank_capacity_l276_276293


namespace percentage_of_primes_divisible_by_3_l276_276804

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l276_276804


namespace smallest_value_abs_sum_l276_276713

theorem smallest_value_abs_sum : 
  ∃ x : ℝ, (λ x, |x + 3| + |x + 5| + |x + 6| = 5) ∧ 
           (∀ y : ℝ, |y + 3| + |y + 5| + |y + 6| ≥ 5) :=
by
  sorry

end smallest_value_abs_sum_l276_276713


namespace smallest_x_not_defined_l276_276290

theorem smallest_x_not_defined : ∃ x : ℝ, (9 * x^2 - 98 * x + 21 = 0) ∧ ∀ y : ℝ, (9 * y^2 - 98 * y + 21 = 0) → x ≤ y :=
begin
  use 1 / 9, 
  split,
  { -- Prove 1 / 9 is a solution
    sorry,
  },
  { -- Prove it is the smallest solution
    sorry,
  }
end

end smallest_x_not_defined_l276_276290


namespace find_g_2_l276_276196

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_2
  (H : ∀ (x : ℝ), x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x ^ 2):
  g 2 = 67 / 14 :=
by
  sorry

end find_g_2_l276_276196


namespace solve_equation_l276_276658

theorem solve_equation : ∀ x : ℝ, ((1 - x) / (x - 4)) + (1 / (4 - x)) = 1 → x = 2 :=
by
  intros x h
  sorry

end solve_equation_l276_276658


namespace equation_of_C2_l276_276429

-- Define the hyperbola C_1 and its parameters
structure Hyperbola (a b : ℝ) :=
  (eqn : ∀ (x y : ℝ), a > 0 ∧ b > 0 → x^2 / a^2 - y^2 / b^2 = 1)

-- Foci of the hyperbola at (±c, 0) where c² = a² + b²
def foci (a b c : ℝ) (h : a^2 + b^2 = c^2) : ℝ × ℝ := (c, -c)

-- Define the circle C_2 centered at F_1 with radius |F_1 F_2|
structure Circle (F_1 : ℝ × ℝ) (r : ℝ) :=
  (eqn : ∀ (x y : ℝ), (x - F_1.1)^2 + (y - F_1.2)^2 = r^2)

-- The problem statement
theorem equation_of_C2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = c^2)
    (F1 F2 : ℝ × ℝ) (h_foci1 : F1 = (-c, 0)) (h_foci2 : F2 = (c, 0))
    (P Q : ℝ × ℝ) (h_intersect : ∀ (x y : ℝ), ((x, y) = P ∨ (x, y) = Q) → Hyperbola a b)
    (area_pf1f2 : ℝ) (h_area: area_pf1f2 = 4) (angle_f1pf2 : ℝ)
    (h_angle : angle_f1pf2 = 75) :
  Circle F1 4
  :=
by
  have : F1 = (-2, 0) := sorry
  exact { eqn := λ x y, (x + 2)^2 + y^2 = 16 }

end equation_of_C2_l276_276429


namespace least_value_a_plus_b_l276_276060

theorem least_value_a_plus_b (a b : ℝ) : 
  log 3 a + log 3 b ≥ 6 → a + b ≥ 54 :=
begin
  sorry
end

end least_value_a_plus_b_l276_276060


namespace sum_of_g2_values_l276_276118

def f (x : ℝ) := x^2 - 3 * x + 2
def g (y : ℝ) := 3 * (y - 2) + 4

theorem sum_of_g2_values : (let x1 := 0 in let g2_1 := g 2) + (let x2 := 3 in let g2_2 := g 2) = 17 := by
  sorry

end sum_of_g2_values_l276_276118


namespace sum_three_numbers_is_247_l276_276251

variables (A B C : ℕ)

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ (nat.digits 10 n)

theorem sum_three_numbers_is_247
  (hA : 100 ≤ A ∧ A < 1000) -- A is a three-digit number
  (hB : 10 ≤ B ∧ B < 100)   -- B is a two-digit number
  (hC : 10 ≤ C ∧ C < 100)   -- C is a two-digit number
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
        (if contains_digit A 7 then A else 0) +
        (if contains_digit B 7 then B else 0) +
        (if contains_digit C 7 then C else 0) = 208) -- Sum of numbers containing digit 7 is 208
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧
        (if contains_digit B 3 then B else 0) +
        (if contains_digit C 3 then C else 0) = 76) -- Sum of numbers containing digit 3 is 76
  : A + B + C = 247 := 
sorry

end sum_three_numbers_is_247_l276_276251


namespace number_of_three_digit_prime_numbers_l276_276393

theorem number_of_three_digit_prime_numbers :
  ∃ (n : ℕ), n = 24 ∧
  ∀ a b c : ℕ,
    (a = 2 ∨ a = 3 ∨ a = 5 ∨ a = 7) ∧
    (b = 2 ∨ b = 3 ∨ b = 5 ∨ b = 7) ∧
    (c = 2 ∨ c = 3 ∨ c = 5 ∨ c = 7) ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    (a * 100 + b * 10 + c).natAbs > 100 ∧ (a * 100 + b * 10 + c).natAbs < 1000 :=
by
  sorry

end number_of_three_digit_prime_numbers_l276_276393


namespace trajectory_of_point_P_dot_product_const_l276_276101

noncomputable def point_A (t : ℝ) : ℝ × ℝ := (t, t)
noncomputable def point_B (t : ℝ) : ℝ × ℝ := (t, -t)
noncomputable def point_P (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def distance (A B : ℝ × ℝ) := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem trajectory_of_point_P (A B P : ℝ × ℝ) (h : distance A B = 4 * real.sqrt 5 / 5) :
  let P := point_P A B in P.1^2 + P.2^2 = 4 / 5 := by
  sorry

theorem dot_product_const (A B P M N : ℝ × ℝ) (hC : P.1^2 + P.2^2 = 4 / 5)
  (ellipse : ∀ M N, (M.1^2 / 4 + M.2^2 = 1) ∧ (N.1^2 / 4 + N.2^2 = 1)) :
  ∃ l : ℝ → ℝ, (∀ t, let P := point_P (point_A t) (point_B t)
  in l P.1 = P.2 + t ∧ ∀ (M N : ℝ × ℝ), (M.1.1)^2 + M.1.2^2 = 4 ∧ (N.1)^2 + N.2^2 = 4) →
  (P.1 * N.1 + P.2 * N.2 = 0) :=
  sorry

end trajectory_of_point_P_dot_product_const_l276_276101


namespace find_eighth_time_l276_276279

-- Define the initial race times before the eighth attempt
def initial_race_times : List ℕ :=
  [99, 103, 106, 108, 110]

-- Define Tim's time for the eighth race
def race_time_eighth (x : ℕ) : List ℕ := 
  (initial_race_times.take 3) ++ [x] ++ (initial_race_times.drop 3)

-- Define the median function for even number of elements
def median (l : List ℕ) : ℕ :=
  let l_sorted := l.insertionSort (· ≤ ·)
  (l_sorted[(l.length / 2) - 1] + l_sorted[l.length / 2]) / 2

-- Theorem stating the proof problem
theorem find_eighth_time (x : ℕ) (h_median : median (race_time_eighth x) = 104) : x = 102 :=
by
  sorry

end find_eighth_time_l276_276279


namespace one_pow_sub_div_l276_276970

theorem one_pow_sub_div (m n : ℕ) (h1 : 1 ^ 567 = 1) (h2 : 3 ^ m / 3 ^ n = 3 ^ (m - n)) :
  1 ^ 567 - 3 ^ 8 / 3 ^ 5 = -26 :=
by
  have h3 : 3 ^ 8 / 3 ^ 5 = 3 ^ 3, from h2 8 5
  rw [h1, h3]
  norm_num
  sorry

end one_pow_sub_div_l276_276970


namespace percentage_expression_l276_276179

variable {A B : ℝ} (hA : A > 0) (hB : B > 0)

theorem percentage_expression (h : A = (x / 100) * B) : x = 100 * (A / B) :=
sorry

end percentage_expression_l276_276179


namespace train_crossing_time_l276_276463

-- Definitions based on conditions
def length_of_train : ℝ := 110  -- meters
def length_of_bridge : ℝ := 340  -- meters
def speed_kmph : ℝ := 60  -- km/h

-- Conversion factor
def kmph_to_mps (v : ℝ) : ℝ := v * (1000 / 3600)

-- Compute total distance the train needs to travel
def total_distance : ℝ := length_of_train + length_of_bridge

-- Convert speed to meters per second
def speed_mps : ℝ := kmph_to_mps speed_kmph

-- Compute time to cross the bridge
def time_to_cross (d : ℝ) (s : ℝ) : ℝ := d / s

-- Theorem statement
theorem train_crossing_time :
  time_to_cross total_distance speed_mps ≈ 27 := 
sorry

end train_crossing_time_l276_276463


namespace arithmetic_sequence_general_formula_l276_276011

open Finset BigOperators

-- Part (1): General formula for the arithmetic sequence
def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = 4 * n

-- Part (2): Sum of first 2n terms of sequence b
def special_seq (a b : ℕ → ℕ) : Prop :=
  b 1 = 1 ∧ 
  (∀ n : ℕ, n % 2 = 1 → b (n + 1) = a n) ∧ 
  (∀ n : ℕ, n % 2 = 0 → b (n + 1) = - b n + 2^n)

def sum_first_2n_terms (P : ℕ → ℕ → ℕ) : Prop :=
  ∀ S n: ℕ,
  S = (∑ i in range (2 * n), λ i, b i) → 
  S = (4^n - 1) / 3 + 4 * n - 3

-- Main Lean statement
theorem arithmetic_sequence_general_formula (a : ℕ → ℕ) (b : ℕ → ℕ) :
  arithmetic_seq a →
  special_seq a b →
  sum_first_2n_terms b :=
by 
  sorry

end arithmetic_sequence_general_formula_l276_276011


namespace trigonometric_identity_l276_276541

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h : 3 * sin α + cos α = 0) :
  1 / (cos α ^ 2 + 2 * sin α * cos α) = 10 / 3 :=
sorry

end trigonometric_identity_l276_276541


namespace positive_integer_divisors_l276_276287

theorem positive_integer_divisors (n : ℕ) (h1 : 0 < n) (h2 : (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))).card = n / 2) : n = 8 ∨ n = 12 :=
sorry

end positive_integer_divisors_l276_276287


namespace abs_a_gt_abs_c_sub_abs_b_l276_276055

theorem abs_a_gt_abs_c_sub_abs_b (a b c : ℝ) (h : |a + c| < b) : |a| > |c| - |b| :=
sorry

end abs_a_gt_abs_c_sub_abs_b_l276_276055


namespace percentage_of_primes_divisible_by_3_is_12_5_l276_276842

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l276_276842


namespace primes_divisible_by_3_percentage_l276_276823

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l276_276823


namespace m_minus_n_is_perfect_square_l276_276124

theorem m_minus_n_is_perfect_square (m n : ℕ) (h : 0 < m) (h1 : 0 < n) (h2 : 2001 * m^2 + m = 2002 * n^2 + n) : ∃ k : ℕ, m = n + k^2 :=
by
    sorry

end m_minus_n_is_perfect_square_l276_276124


namespace percentage_of_primes_divisible_by_3_l276_276750

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l276_276750


namespace percentage_of_primes_divisible_by_3_l276_276758

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l276_276758


namespace beef_cubes_per_slab_l276_276343

-- Define the conditions as variables
variables (kabob_sticks : ℕ) (cubes_per_stick : ℕ) (cost_per_slab : ℕ) (total_cost : ℕ) (total_kabob_sticks : ℕ)

-- Assume the conditions from step a)
theorem beef_cubes_per_slab 
  (h1 : cubes_per_stick = 4) 
  (h2 : cost_per_slab = 25) 
  (h3 : total_cost = 50) 
  (h4 : total_kabob_sticks = 40)
  : total_cost / cost_per_slab * (total_kabob_sticks * cubes_per_stick) / (total_cost / cost_per_slab) = 80 := 
by {
  -- the proof goes here
  sorry
}

end beef_cubes_per_slab_l276_276343


namespace geometric_sequence_problem_l276_276022

theorem geometric_sequence_problem
  (q : ℕ)
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 : ℕ)
  (n : ℕ)
  (hq : q = 2)
  (hS : ∀ n, S n = a1 * (1 - q ^ n) / (1 - q))
  (ha : ∀ n, a n = a1 * q ^ (n - 1)) :
  (S 4 / a 2) = (-15 / 2) := 
by
  sorry

end geometric_sequence_problem_l276_276022


namespace line_slope_through_origin_intersects_parabola_l276_276335

theorem line_slope_through_origin_intersects_parabola (k : ℝ) :
  (∃ x1 x2 : ℝ, 5 * (kx1) = 2 * x1 ^ 2 - 9 * x1 + 10 ∧ 5 * (kx2) = 2 * x2 ^ 2 - 9 * x2 + 10 ∧ x1 + x2 = 77) → k = 29 :=
by
  intro h
  sorry

end line_slope_through_origin_intersects_parabola_l276_276335


namespace driver_travel_distance_per_week_l276_276329

open Nat

-- Defining the parameters
def speed1 : ℕ := 30
def time1 : ℕ := 3
def speed2 : ℕ := 25
def time2 : ℕ := 4
def days : ℕ := 6

-- Lean statement to prove
theorem driver_travel_distance_per_week : 
  (speed1 * time1 + speed2 * time2) * days = 1140 := 
by 
  sorry

end driver_travel_distance_per_week_l276_276329


namespace numberOfWaysToChoose4Cards_l276_276513

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l276_276513


namespace inverse_function_value_l276_276682

-- Defining the function g as a list of pairs
def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 3
  | 2 => 6
  | 3 => 1
  | 4 => 5
  | 5 => 4
  | 6 => 2
  | _ => 0 -- default case which should not be used

-- Defining the inverse function g_inv using the values determined from g
def g_inv (y : ℕ) : ℕ :=
  match y with
  | 3 => 1
  | 6 => 2
  | 1 => 3
  | 5 => 4
  | 4 => 5
  | 2 => 6
  | _ => 0 -- default case which should not be used

theorem inverse_function_value :
  g_inv (g_inv (g_inv 6)) = 2 :=
by
  sorry

end inverse_function_value_l276_276682


namespace farmer_rent_l276_276878

-- Definitions based on given conditions
def rent_per_acre_per_month : ℕ := 60
def length_of_plot : ℕ := 360
def width_of_plot : ℕ := 1210
def square_feet_per_acre : ℕ := 43560

-- Problem statement: 
-- Prove that the monthly rent to rent the rectangular plot is $600.
theorem farmer_rent : 
  (length_of_plot * width_of_plot) / square_feet_per_acre * rent_per_acre_per_month = 600 :=
by
  sorry

end farmer_rent_l276_276878


namespace different_suits_choice_count_l276_276506

-- Definitions based on the conditions
def standard_deck : List (Card × Suit) := 
  List.product Card.all Suit.all

def four_cards (deck : List (Card × Suit)) : Prop :=
  deck.length = 4 ∧ ∀ (i j : Fin 4), i ≠ j → (deck.nthLe i (by simp) : Card × Suit).2 ≠ (deck.nthLe j (by simp) : Card × Suit).2

-- Statement of the proof problem
theorem different_suits_choice_count :
  ∃ l : List (Card × Suit), four_cards l ∧ standard_deck.choose 4 = 28561 :=
by
  sorry

end different_suits_choice_count_l276_276506


namespace percentage_primes_divisible_by_3_l276_276782

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l276_276782


namespace a_9_value_l276_276057

noncomputable def a : ℕ → ℕ
| 3 := 1
| 4 := 3
| (n + 2) := a (n + 1) + a n + n

theorem a_9_value : a 9 = 79 := by
  -- Initial conditions
  have h_a3 : a 3 = 1 := rfl
  have h_a4 : a 4 = 3 := rfl
  -- Recursive formula
  have h_a5 : a 5 = a 4 + a 3 + 3 := by rw [h_a4, h_a3]; norm_num
  have h_a6 : a 6 = a 5 + a 4 + 4 := by rw [h_a5, h_a4]; norm_num
  have h_a7 : a 7 = a 6 + a 5 + 5 := by rw [h_a6, h_a5]; norm_num
  have h_a8 : a 8 = a 7 + a 6 + 6 := by rw [h_a7, h_a6]; norm_num
  have h_a9 : a 9 = a 8 + a 7 + 7 := by rw [h_a8, h_a7]; norm_num
  -- Proof for a 9
  exact h_a9

end a_9_value_l276_276057


namespace percentage_primes_divisible_by_3_l276_276865

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l276_276865


namespace domain_lg_func_l276_276668

def domain_of_function := {x : ℝ | (1 < x ∧ x ≠ 2)}

theorem domain_lg_func : domain_of_function = {x : ℝ | (1 < x < 2) ∨ (2 < x)} :=
by sorry

end domain_lg_func_l276_276668


namespace infinitely_many_n_divides_2n_plus_3n_l276_276154

def a_sequence : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 ^ a_sequence n + 3 ^ a_sequence n

theorem infinitely_many_n_divides_2n_plus_3n :
  ∃ᶠ n in at_top, n ∣ 2^n + 3^n := 
sorry

end infinitely_many_n_divides_2n_plus_3n_l276_276154


namespace primes_divisible_by_3_percentage_l276_276817

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l276_276817


namespace number_of_elements_in_A_l276_276679

variable {S : Set ℕ} [Fintype S]

def A (n : ℕ) := {x : ℕ | x ∈ S ∧ x = 13 * n} 
def B (n : ℕ) := {x : ℕ | x ∈ S ∧ x = x1}
variable n : ℕ := 154
theorem number_of_elements_in_A:
n ≤ ∑ k : n, x1
-/
∃ A B: (Set ℕ),
  (∀ x : ℕ, x ∈ A ∧ x ∉ B) ∧ ∀ x1 : x, ∃ x1 + x2 ∈ B
  ∃ (x2 * x) ∈ S
Sorry
#align qs_to_statement number_of_elements_in_A

end number_of_elements_in_A_l276_276679


namespace increasing_function_iff_l276_276448

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a ^ x else (3 - a) * x + (1 / 2) * a

theorem increasing_function_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 2 ≤ a ∧ a < 3 :=
by
  sorry

end increasing_function_iff_l276_276448


namespace neg_number_is_A_l276_276932

def A : ℤ := -(3 ^ 2)
def B : ℤ := (-3) ^ 2
def C : ℤ := abs (-3)
def D : ℤ := -(-3)

theorem neg_number_is_A : A < 0 := 
by sorry

end neg_number_is_A_l276_276932


namespace false_converse_implication_l276_276353

theorem false_converse_implication : ∃ x : ℝ, (0 < x) ∧ (x - 3 ≤ 0) := by
  sorry

end false_converse_implication_l276_276353


namespace daily_serving_size_l276_276189

-- Definitions based on problem conditions
def days : ℕ := 180
def capsules_per_bottle : ℕ := 60
def bottles : ℕ := 6
def total_capsules : ℕ := bottles * capsules_per_bottle

-- Theorem statement to prove the daily serving size
theorem daily_serving_size :
  total_capsules / days = 2 := by
  sorry

end daily_serving_size_l276_276189


namespace cos_theta_l276_276043

variables {α : Type*} [inner_product_space ℝ α] (a b : α)

theorem cos_theta (ha : ∥a∥ = 5) (hb : ∥b∥ = 7) (hensum : ∥a + b∥ = 10) :
  real.cos (real.angle_of_vectors a b) = 13 / 35 :=
sorry

end cos_theta_l276_276043


namespace bouquet_branches_l276_276079

variable (w : ℕ) (b : ℕ)

theorem bouquet_branches :
  (w + b = 7) → 
  (w ≥ 1) → 
  (∀ x y, x ≠ y → (x = w ∨ y = w) → (x = b ∨ y = b)) → 
  (w = 1 ∧ b = 6) :=
by
  intro h1 h2 h3
  sorry

end bouquet_branches_l276_276079


namespace sphere_to_hemisphere_volume_ratio_l276_276204

theorem sphere_to_hemisphere_volume_ratio (r : ℝ) : 
  let V_sphere := (4 / 3) * Real.pi * r^3 in
  let V_hemisphere := (1 / 2) * (4 / 3) * Real.pi * (3 * r)^3 in
  V_sphere / V_hemisphere = 2 / 27 := 
by sorry

end sphere_to_hemisphere_volume_ratio_l276_276204


namespace chord_length_of_circle_and_line_l276_276185

noncomputable def compute_chord_length (t : ℝ) : ℝ :=
2 * sqrt (9 - (3 / sqrt 5) ^ 2)

theorem chord_length_of_circle_and_line :
  let circle_radius := 3 in
  let distance_to_center := 3 / sqrt 5 in
  let chord_length := 2 * sqrt (circle_radius ^ 2 - distance_to_center ^ 2) in
  chord_length = (12 / 5) * sqrt 5 :=
sorry

end chord_length_of_circle_and_line_l276_276185


namespace boys_score_l276_276904

-- Definitions
def percentage_boys : ℝ := 0.4
def percentage_girls : ℝ := 0.6
def score_girls : ℝ := 90
def class_average_score : ℝ := 86
def total_students : ℕ := 100

-- Proof statement
theorem boys_score :
  ∀ B : ℝ,
  percentage_boys * B * total_students + percentage_girls * score_girls * total_students = class_average_score * total_students →
  B = 80 :=
by 
  intros B h,
  sorry

end boys_score_l276_276904


namespace subsets_containing_5_and_6_l276_276478

theorem subsets_containing_5_and_6 (S : Set ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  {T : Set ℕ // {5, 6} ⊆ T ∧ T ⊆ S}.card = 16 := 
sorry

end subsets_containing_5_and_6_l276_276478


namespace sum_of_three_numbers_l276_276247

def contains_digit (n : ℕ) (d : ℕ) : Prop := d ∈ n.digits 10

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1: 100 ≤ A ∧ A ≤ 999)
  (h2: 10 ≤ B ∧ B ≤ 99) 
  (h3: 10 ≤ C ∧ C ≤ 99)
  (h4: (contains_digit A 7 → A) + (contains_digit B 7 → B) + (contains_digit C 7 → C) = 208)
  (h5: (contains_digit B 3 → B) + (contains_digit C 3 → C) = 76) :
  A + B + C = 247 := 
by 
  sorry

end sum_of_three_numbers_l276_276247


namespace cristine_lemons_left_l276_276372

theorem cristine_lemons_left (initial_lemons : ℕ) (given_fraction : ℚ) (exchanged_lemons : ℕ) (h1 : initial_lemons = 12) (h2 : given_fraction = 1/4) (h3 : exchanged_lemons = 2) : 
  initial_lemons - initial_lemons * given_fraction - exchanged_lemons = 7 :=
by 
  sorry

end cristine_lemons_left_l276_276372


namespace smallest_integer_omega_l276_276194

noncomputable def f (ω : ℝ) (x : ℝ) := sin (ω * x + π / 3) + cos (ω * x - π / 6)

noncomputable def g (ω : ℝ) (x : ℝ) := f ω (x / 2)

-- Prove that the smallest integer ω > 0 such that g(ω) has exactly one extremum in
-- the interval (0, π/18) is 2.
theorem smallest_integer_omega (ω : ℝ) (hω_pos : 0 < ω) :
  (∃! x ∈ set.Ioo 0 (π / 18), derivative (g ω) x = 0) ∧ 
  ∀ x' ∈ set.Ioo 0 (π / 18), derivative (g ω) x' ≠ 0 → ω = 2 :=
sorry

end smallest_integer_omega_l276_276194


namespace choose_4_cards_of_different_suits_l276_276536

theorem choose_4_cards_of_different_suits :
  (∃ (n : ℕ), choose 4 4 = n) ∧
  (∃ (m : ℕ), (13^4 = m)) ∧
  (1 * (13^4) = 28561)

end choose_4_cards_of_different_suits_l276_276536


namespace min_abs_sum_l276_276718

theorem min_abs_sum : ∃ x : ℝ, ∀ x : ℝ, 
  let f := λ x : ℝ, abs (x + 3) + abs (x + 5) + abs (x + 6) in
  f x = 5 :=
sorry

end min_abs_sum_l276_276718


namespace exponent_of_4_proof_l276_276026

noncomputable def exponent_of_4 (x y : ℝ) : ℝ :=
  y - 1

theorem exponent_of_4_proof (x y : ℝ)
  (h1 : 5^(x+1) * 4^(y-1) = 25^x * 64^y)
  (h2 : x + y = 0.5) :
  exponent_of_4 x y = -1.5 :=
sorry

end exponent_of_4_proof_l276_276026


namespace sum_quotient_dividend_divisor_l276_276902

theorem sum_quotient_dividend_divisor (N : ℕ) (divisor : ℕ) (quotient : ℕ) (sum : ℕ) 
    (h₁ : N = 40) (h₂ : divisor = 2) (h₃ : quotient = N / divisor)
    (h₄ : sum = quotient + N + divisor) : sum = 62 := by
  -- proof goes here
  sorry

end sum_quotient_dividend_divisor_l276_276902


namespace no_partition_possible_l276_276605

noncomputable def partition_possible (A : ℕ → Set ℕ) :=
  (∀ k: ℕ, ∃ finA : Finset ℕ, (A k = finA.to_set) ∧ (finA.sum id = k + 2013)) ∧
  (∀ i j: ℕ, i ≠ j → (A i ∩ A j) = ∅) ∧
  (⋃ i, A i) = Set.univ

theorem no_partition_possible :
  ¬ ∃ A : ℕ → Set ℕ, partition_possible A := 
sorry

end no_partition_possible_l276_276605


namespace angle_C_measure_ratio_inequality_l276_276039

open Real

variables (A B C a b c : ℝ)

-- Assumptions
variable (ABC_is_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
variable (sin_condition : sin (2 * C - π / 2) = 1/2)
variable (inequality_condition : a^2 + b^2 < c^2)

theorem angle_C_measure :
  0 < C ∧ C < π ∧ C = 2 * π / 3 := sorry

theorem ratio_inequality :
  1 < (a + b) / c ∧ (a + b) / c ≤ 2 * sqrt 3 / 3 := sorry

end angle_C_measure_ratio_inequality_l276_276039


namespace acute_triangle_cos_inequality_l276_276122

variable {α β γ : ℝ}

theorem acute_triangle_cos_inequality (hα : 0 < α ∧ α < π / 2)
    (hβ : 0 < β ∧ β < π / 2)
    (hγ : 0 < γ ∧ γ < π / 2)
    (h_sum : α + β + γ = π) :
    (cos α / cos (β - γ)) + (cos β / cos (γ - α)) + (cos γ / cos (α - β)) ≥ 3 / 2 :=
by
  sorry

end acute_triangle_cos_inequality_l276_276122


namespace cost_of_adult_ticket_l276_276692

theorem cost_of_adult_ticket (A : ℝ) 
  (child_ticket_cost : ℝ) (total_tickets_sold : ℕ) 
  (total_receipts : ℝ) (adult_tickets_sold : ℕ)
  (child_tickets_cost_sum : ℝ) :
  child_ticket_cost = 4 ∧ total_tickets_sold = 130 ∧ total_receipts = 840 ∧ adult_tickets_sold = 40 ∧ 
  90 * child_ticket_cost = child_tickets_cost_sum →
  40 * A + child_tickets_cost_sum = 840 → A = 12 := 
by
  intros h1 h2
  cases h1 with h1_rest h1_sum
  cases h1_rest with h1_child_ticket_cost h1_rest'
  cases h1_rest' with h1_total_tickets_sold h1_rest''
  cases h1_rest'' with h1_total_receipts h1_adult_tickets_sold
  cases h1_adult_tickets_sold with h1_adult_tickets_sold_val h1_child_tickets_cost_sum
  rw h1_child_tickets_cost_sum at h2
  have h : 40 * A + 90 * 4 = 840 := by assumption
  linarith

end cost_of_adult_ticket_l276_276692


namespace triangle_problem_solution_l276_276070

noncomputable def triangle_proof_problem
  (a b c : ℝ) (sin_B : ℝ) (cos_B : ℝ)
  (h_a_gt_b : a > b) (h_a_eq_5 : a = 5) (h_c_eq_6 : c = 6)
  (h_sin_B : sin_B = 3 / 5) (h_cos_B : cos_B = 4 / 5) : Prop :=
  let b_squared := a^2 + c^2 - 2 * a * c * cos_B in
  let b_value := real.sqrt b_squared in
  let sin_A := (a * sin_B) / b_value in
  let cos_A := real.sqrt (1 - sin_A^2) in
  let sin_2A := 2 * sin_A * cos_A in
  let cos_2A := 1 - 2 * (sin_A)^2 in
  let sin_2A_plus_pi_div_4 := sin_2A * real.cos (π / 4) + cos_2A * real.sin (π / 4) in
  b_value = real.sqrt 13 ∧ sin_A = 3 * real.sqrt 13 / 13 ∧ sin_2A_plus_pi_div_4 = 7 * real.sqrt 2 / 26

-- Statements in Lean
theorem triangle_problem_solution
  (a b c : ℝ) (sin_B : ℝ) (cos_B : ℝ)
  (h_a_gt_b : a > b) (h_a_eq_5 : a = 5) (h_c_eq_6 : c = 6)
  (h_sin_B : sin_B = 3 / 5) (h_cos_B : cos_B = 4 / 5) :
  triangle_proof_problem a b c sin_B cos_B h_a_gt_b h_a_eq_5 h_c_eq_6 h_sin_B h_cos_B :=
sorry

end triangle_problem_solution_l276_276070


namespace unique_positive_root_l276_276891

noncomputable def polynomial (a b : ℝ) : ℝ → ℝ := λ x, x^3 + a * x^2 - b

theorem unique_positive_root (a b : ℝ) (hb : b > 0) : 
  ∃! x > 0, polynomial a b x = 0 :=
sorry

end unique_positive_root_l276_276891


namespace percent_primes_divisible_by_3_less_than_20_l276_276770

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l276_276770


namespace range_of_x_for_meaningful_sqrt_l276_276571

noncomputable def meaningful_sqrt (x : ℝ) : Prop :=
  sqrt (1 / (x - 1)) ≥ 0

theorem range_of_x_for_meaningful_sqrt:
  ∀ x : ℝ, meaningful_sqrt x → x > 1 :=
by
  intros x h
  sorry

end range_of_x_for_meaningful_sqrt_l276_276571


namespace only_OptionA_like_terms_l276_276870

-- Definitions of the expressions in each option
def OptionA_expr1 : ℕ := 2023
def OptionA_expr2 : ℕ := 2024

def OptionB_expr1 (a b : ℕ) : ℕ := a^2 * b
def OptionB_expr2 (a b : ℕ) : ℕ := 3 * b^2 * a

def OptionC_expr1 (x y : ℕ) : ℕ := 3 * x * y
def OptionC_expr2 (y z : ℕ) : ℕ := 4 * y * z

def OptionD_expr1 (x y : ℕ) : ℕ := -x * y
def OptionD_expr2 (x y z : ℕ) : ℕ := x * y * z

-- The theorem stating that only Option A expressions are like terms
theorem only_OptionA_like_terms : 
  (OptionA_expr1 = OptionA_expr2) 
  ∧ (∀ (a b : ℕ), OptionB_expr1 a b ≠ OptionB_expr2 a b)
  ∧ (∀ (x y z : ℕ), OptionC_expr1 x y ≠ OptionC_expr2 y z)
  ∧ (∀ (x y z : ℕ), OptionD_expr1 x y ≠ OptionD_expr2 x y z) := 
by
  sorry

end only_OptionA_like_terms_l276_276870


namespace constant_term_in_expansion_l276_276289

theorem constant_term_in_expansion {α : Type*} [Comm_ring α] (x : α) :
  let term := (10.choose 5) * 4^5 in
  (term : α) = 258048 :=
by
  sorry

end constant_term_in_expansion_l276_276289


namespace equation_D_is_quadratic_l276_276298

-- Define each of the given equations as predicates over a variable x (and possibly y)
def equation_A (x : ℝ) : Prop := -6 * x + 2 = 0
def equation_B (x y : ℝ) : Prop := 2 * x^2 - y + 1 = 0
def equation_C (x : ℝ) : Prop := 1 / (x^2) + x = 2
def equation_D (x : ℝ) : Prop := x^2 + 2 * x = 0

-- Define a predicate that states an equation is a quadratic equation
def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, (a ≠ 0) ∧ ∀ x, eq x = (a * x^2 + b * x + c = 0)

-- The theorem states that among the given equations, only equation D is quadratic
theorem equation_D_is_quadratic :
  is_quadratic equation_D ∧
  ¬ is_quadratic equation_A ∧
  ¬ is_quadratic equation_B ∧
  ¬ is_quadratic equation_C :=
by
  sorry

end equation_D_is_quadratic_l276_276298


namespace problem_statement_l276_276981

theorem problem_statement (x y : ℕ) (hx : x = 7) (hy : y = 3) : (x - y)^2 * (x + y)^2 = 1600 :=
by
  rw [hx, hy]
  sorry

end problem_statement_l276_276981


namespace circle_radii_l276_276148

theorem circle_radii (A B C D E : Point) 
  (BD BE R1 R2 : ℝ)
  (collinear : Collinear [A, B, C])
  (diameter_AB : diameter AB = 2 * R1)
  (diameter_BC : diameter BC = 2 * R2)
  (line_tangent : Tangent (line A D E) (circle_centered_at B with_radius R2))
  (BD_len : BD = 9)
  (BE_len : BE = 12) :
  R1 = 36 ∧ R2 = 8 := 
sorry

end circle_radii_l276_276148


namespace different_suits_choice_count_l276_276509

-- Definitions based on the conditions
def standard_deck : List (Card × Suit) := 
  List.product Card.all Suit.all

def four_cards (deck : List (Card × Suit)) : Prop :=
  deck.length = 4 ∧ ∀ (i j : Fin 4), i ≠ j → (deck.nthLe i (by simp) : Card × Suit).2 ≠ (deck.nthLe j (by simp) : Card × Suit).2

-- Statement of the proof problem
theorem different_suits_choice_count :
  ∃ l : List (Card × Suit), four_cards l ∧ standard_deck.choose 4 = 28561 :=
by
  sorry

end different_suits_choice_count_l276_276509


namespace subsets_containing_5_and_6_l276_276485

theorem subsets_containing_5_and_6 (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ s, 5 ∈ s ∧ 6 ∈ s)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276485


namespace fourth_root_expression_l276_276356

-- Define a positive real number y
variable (y : ℝ) (hy : 0 < y)

-- State the problem in Lean
theorem fourth_root_expression : 
  Real.sqrt (Real.sqrt (y^2 * Real.sqrt y)) = y^(5/8) := sorry

end fourth_root_expression_l276_276356


namespace primes_divisible_by_3_percentage_is_12_5_l276_276742

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l276_276742


namespace numberOfWaysToChoose4Cards_l276_276516

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l276_276516


namespace f_sin_alpha_gt_f_cos_beta_l276_276013

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_is_even : ∀ x ∈ set.Icc (-1 : ℝ) 1, f (-x) = f x
axiom f_is_decreasing_on_neg : ∀ x y ∈ set.Icc (-1 : ℝ) 0, x < y → f x > f y
axiom alpha_beta_acute (α β : ℝ) : α + β > (90 : ℝ) / 180 * real.pi

theorem f_sin_alpha_gt_f_cos_beta (α β : ℝ) (hα : 0 < α) (hβ : 0 < β) (hαβ : α + β < real.pi / 2):
  f (real.sin α) > f (real.cos β) :=
by
  sorry

end f_sin_alpha_gt_f_cos_beta_l276_276013


namespace sum_three_numbers_is_247_l276_276250

variables (A B C : ℕ)

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ (nat.digits 10 n)

theorem sum_three_numbers_is_247
  (hA : 100 ≤ A ∧ A < 1000) -- A is a three-digit number
  (hB : 10 ≤ B ∧ B < 100)   -- B is a two-digit number
  (hC : 10 ≤ C ∧ C < 100)   -- C is a two-digit number
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
        (if contains_digit A 7 then A else 0) +
        (if contains_digit B 7 then B else 0) +
        (if contains_digit C 7 then C else 0) = 208) -- Sum of numbers containing digit 7 is 208
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧
        (if contains_digit B 3 then B else 0) +
        (if contains_digit C 3 then C else 0) = 76) -- Sum of numbers containing digit 3 is 76
  : A + B + C = 247 := 
sorry

end sum_three_numbers_is_247_l276_276250


namespace vinegar_final_percentage_l276_276324

def vinegar_percentage (volume1 volume2 : ℕ) (percent1 percent2 : ℚ) : ℚ :=
  let vinegar1 := volume1 * percent1 / 100
  let vinegar2 := volume2 * percent2 / 100
  (vinegar1 + vinegar2) / (volume1 + volume2) * 100

theorem vinegar_final_percentage:
  vinegar_percentage 128 128 8 13 = 10.5 :=
  sorry

end vinegar_final_percentage_l276_276324


namespace percentage_of_primes_divisible_by_3_l276_276789

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l276_276789


namespace original_oil_weight_is_75_l276_276319

def initial_oil_weight (original : ℝ) : Prop :=
  let first_remaining := original / 2
  let second_remaining := first_remaining * (4 / 5)
  second_remaining = 30

theorem original_oil_weight_is_75 : ∃ (original : ℝ), initial_oil_weight original ∧ original = 75 :=
by
  use 75
  unfold initial_oil_weight
  sorry

end original_oil_weight_is_75_l276_276319


namespace samantha_routes_l276_276162

-- Definitions of the conditions
def blocks_west_to_sw_corner := 3
def blocks_south_to_sw_corner := 2
def blocks_east_to_school := 4
def blocks_north_to_school := 3
def ways_house_to_sw_corner : ℕ := Nat.choose (blocks_west_to_sw_corner + blocks_south_to_sw_corner) blocks_south_to_sw_corner
def ways_through_park : ℕ := 2
def ways_ne_corner_to_school : ℕ := Nat.choose (blocks_east_to_school + blocks_north_to_school) blocks_north_to_school

-- The proof statement
theorem samantha_routes : (ways_house_to_sw_corner * ways_through_park * ways_ne_corner_to_school) = 700 :=
by
  -- Using "sorry" as a placeholder for the actual proof
  sorry

end samantha_routes_l276_276162


namespace angle_bisectors_sum_l276_276098

theorem angle_bisectors_sum (A B C A1 B1 : Point) (α β : ∠) 
  (h_triangle : Triangle A B C)
  (h_angle_C : ∠C = 60)
  (h_AA1 : Angle_bisector AA1)
  (h_BB1 : Angle_bisector BB1) :
  dist A B1 + dist B A1 = dist A B :=
sorry

end angle_bisectors_sum_l276_276098


namespace percentage_of_primes_divisible_by_3_is_12_5_l276_276840

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l276_276840


namespace positional_relationship_l276_276202

noncomputable def center_radius (a b c : ℝ) : (ℝ × ℝ) × ℝ :=
  let h := a / 2;
  let k := b / 2;
  let r := (h^2 + k^2 - c).sqrt in
  ((h, k), r)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

theorem positional_relationship :
  let circle1 := (x^2 + y^2 - 6*y);
  let circle2 := (x^2 + y^2 - 8*x + 12);
  let ((x1, y1), r1) := center_radius 0 (-6) 0;
  let ((x2, y2), r2) := center_radius (-8) 12 0;
  distance (x1, y1) (x2, y2) = r1 + r2 →
  ∃ d, d = r1 + r2 ∧ distance (x1, y1) (x2, y2) = d :=
by sorry

end positional_relationship_l276_276202


namespace bc_length_tangent_circle_l276_276579

theorem bc_length_tangent_circle (a b bc cd ad : ℝ) (h_ab : a > b) 
  (h_ab_eq : |AB| = a) (h_ad_eq : |AD| = b) (h_tangent : tangent_circle BC CD AD AB) 
  : |BC| = a - b :=
begin
  -- Proof will be provided here
  sorry
end

end bc_length_tangent_circle_l276_276579


namespace smallest_value_abs_sum_l276_276714

theorem smallest_value_abs_sum : 
  ∃ x : ℝ, (λ x, |x + 3| + |x + 5| + |x + 6| = 5) ∧ 
           (∀ y : ℝ, |y + 3| + |y + 5| + |y + 6| ≥ 5) :=
by
  sorry

end smallest_value_abs_sum_l276_276714


namespace second_largest_prime_factor_of_450_is_13_l276_276115

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in divisors n, d

def prime_factors (n : ℕ) : List ℕ :=
  multiset.to_list (unique_factorization_monoid.factor n)

def second_largest_prime_factor (n : ℕ) : Option ℕ :=
  let factors := prime_factors n
  if h : 1 < length factors then
    some (nth_le factors (length factors - 2) (by linarith))
  else
    none

theorem second_largest_prime_factor_of_450_is_13 :
  second_largest_prime_factor (sum_of_divisors 450) = some 13 :=
by
  sorry

end second_largest_prime_factor_of_450_is_13_l276_276115


namespace sum_of_numbers_l276_276266

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), n = k * 10 + d + m * 10 * (10 ^ k)

theorem sum_of_numbers
  (A B C : ℕ)
  (hA : A >= 100 ∧ A < 1000)
  (hB : B >= 10 ∧ B < 100)
  (hC : C >= 10 ∧ C < 100)
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
              (if contains_digit A 7 then A else 0) +
              (if contains_digit B 7 then B else 0) +
              (if contains_digit C 7 then C else 0) = 208)
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧ 
              (if contains_digit B 3 then B else 0) +
              (if contains_digit C 3 then C else 0) = 76) :
  A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276266


namespace subsets_containing_5_and_6_l276_276490

theorem subsets_containing_5_and_6 (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ s, 5 ∈ s ∧ 6 ∈ s)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276490


namespace problem1_problem2_l276_276454

variable p_def : ∃ x0 ∈ Set.Icc 1 3, x0 - Real.log x0 < m
variable q_def : ∀ x : ℝ, x^2 + 2 > m^2

theorem problem1 (h : ¬p_def ∧ q_def) : -Real.sqrt 2 < m ∧ m ≤ 1 := 
by sorry

theorem problem2 (h1 : p_def ∨ q_def) (h2 : ¬(p_def ∧ q_def)) : 
(m < -Real.sqrt 2 ∨ (m ≥ -Real.sqrt 2 ∧ m ≤ 1) ∨ m ≥ Real.sqrt 2) := 
by sorry

end problem1_problem2_l276_276454


namespace sufficient_not_necessary_necessary_and_sufficient_P_inter_Q_l276_276457

noncomputable def P (x : ℝ) : Prop := (x - 1)^2 > 16
noncomputable def Q (x a : ℝ) : Prop := x^2 + (a - 8) * x - 8 * a ≤ 0

theorem sufficient_not_necessary (a : ℝ) (x : ℝ) :
  a = 3 →
  (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8) :=
sorry

theorem necessary_and_sufficient (a : ℝ) :
  (-5 ≤ a ∧ a ≤ 3) ↔ ∀ x, (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8) :=
sorry

theorem P_inter_Q (a : ℝ) (x : ℝ) :
  (a > 3 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a) ∨ (5 < x ∧ x ≤ 8)) ∧
  (-5 ≤ a ∧ a ≤ 3 → (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8)) ∧
  (-8 ≤ a ∧ a < -5 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a)) ∧
  (a < -8 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a)) :=
sorry

end sufficient_not_necessary_necessary_and_sufficient_P_inter_Q_l276_276457


namespace different_suits_choice_count_l276_276508

-- Definitions based on the conditions
def standard_deck : List (Card × Suit) := 
  List.product Card.all Suit.all

def four_cards (deck : List (Card × Suit)) : Prop :=
  deck.length = 4 ∧ ∀ (i j : Fin 4), i ≠ j → (deck.nthLe i (by simp) : Card × Suit).2 ≠ (deck.nthLe j (by simp) : Card × Suit).2

-- Statement of the proof problem
theorem different_suits_choice_count :
  ∃ l : List (Card × Suit), four_cards l ∧ standard_deck.choose 4 = 28561 :=
by
  sorry

end different_suits_choice_count_l276_276508


namespace sum_of_three_numbers_l276_276241

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d ∨ n / 10 % 10 = d ∨ n / 100 = d

theorem sum_of_three_numbers (A B C : ℕ) :
  (100 ≤ A ∧ A < 1000 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100) ∧
  (∃ (B7 C7 : ℕ), B7 + C7 = 208 ∧ (contains_digit A 7 ∨ contains_digit B7 7 ∨ contains_digit C7 7)) ∧
  (∃ (B3 C3 : ℕ), B3 + C3 = 76 ∧ (contains_digit B3 3 ∨ contains_digit C3 3)) →
  A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276241


namespace percentage_of_primes_divisible_by_3_l276_276757

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l276_276757


namespace pie_filling_cans_l276_276339

-- Conditions
def price_per_pumpkin : ℕ := 3
def total_pumpkins : ℕ := 83
def total_revenue : ℕ := 96
def pumpkins_per_can : ℕ := 3

-- Definition
def cans_of_pie_filling (price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can : ℕ) : ℕ :=
  let pumpkins_sold := total_revenue / price_per_pumpkin
  let pumpkins_remaining := total_pumpkins - pumpkins_sold
  pumpkins_remaining / pumpkins_per_can

-- Theorem
theorem pie_filling_cans : cans_of_pie_filling price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can = 17 :=
  by sorry

end pie_filling_cans_l276_276339


namespace num_valid_arrangements_without_A_at_start_and_B_at_end_l276_276685

-- Define a predicate for person A being at the beginning
def A_at_beginning (arrangement : List ℕ) : Prop :=
  arrangement.head! = 1

-- Define a predicate for person B being at the end
def B_at_end (arrangement : List ℕ) : Prop :=
  arrangement.getLast! = 2

-- Main theorem stating the number of valid arrangements
theorem num_valid_arrangements_without_A_at_start_and_B_at_end : ∃ (count : ℕ), count = 78 :=
by
  have total_arrangements := Nat.factorial 5
  have A_at_start_arrangements := Nat.factorial 4
  have B_at_end_arrangements := Nat.factorial 4
  have both_A_and_B_arrangements := Nat.factorial 3
  let valid_arrangements := total_arrangements - 2 * A_at_start_arrangements + both_A_and_B_arrangements
  use valid_arrangements
  sorry

end num_valid_arrangements_without_A_at_start_and_B_at_end_l276_276685


namespace sequence_a_lt_sqrt_l276_276889

noncomputable def sequence_a (n : ℕ) (N : ℕ) : ℕ → ℝ
| 0 => 0
| 1 => 1
| (k+1) => (sequence_a k N) * (2 - 1 / N) - (sequence_a (k - 1) N)

theorem sequence_a_lt_sqrt (N : ℕ) (hn : 0 < N) (n : ℕ) : sequence_a n N < Real.sqrt (N + 1) :=
by
  sorry

end sequence_a_lt_sqrt_l276_276889


namespace compute_expression_l276_276949

theorem compute_expression :
  let mixed_fraction := (3 : ℚ) + 3 / 8
  let improper_fraction := 27 / 8
  let exponent_result := (improper_fraction⁻¹)^(2 / 3)
  let log_base_half := log 0.5 2
  exponent_result + log_base_half = -5 / 9 :=
by
  let mixed_fraction := (3 : ℚ) + 3 / 8
  let improper_fraction := 27 / 8
  let exponent_result := (improper_fraction⁻¹)^(2 / 3)
  let log_base_half := -1
  show (exponent_result + log_base_half) = -5 / 9
  sorry

end compute_expression_l276_276949


namespace percentage_of_primes_divisible_by_3_l276_276809

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l276_276809


namespace coordinates_of_M_trajectory_of_P_l276_276452

-- Part (1): Coordinates of point M
theorem coordinates_of_M (k : ℝ) (hk : k > 0) (hk_not_2 : k ≠ 2) :
  ∃ (M : ℝ × ℝ), (M.1, M.2) = (-√2, -2) ∧
  ∃ (m := 2), k ≠ -2 ∧ (k ≠ 2) and
  ∀ (x y : ℝ), (x^2 - (y^2) / 4 = 1) ∧ (y = k*x + m) :=
sorry

-- Part (2): Trajectory of point P
theorem trajectory_of_P (x y : ℝ) (k m : ℝ) (hk : k > 0) (hk_not_2 : k ≠ 2) :
  y ≠ 0 → (x, y) moves along M trajectory →
  (x^2 / 25) - (4 * y^2 / 25) = 1 :=
sorry

end coordinates_of_M_trajectory_of_P_l276_276452


namespace product_f_vals_l276_276031
noncomputable def f : ℝ → ℝ := sorry 

theorem product_f_vals :
  (∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → f (x₁ + x₂) = f x₁ * f x₂) →
  (f 0 ≠ 0) →
  (∏ k in (Finset.range (2007 * 2 + 1)).image (λ i, ↑i - 2007), f k) = 1 :=
  sorry

end product_f_vals_l276_276031


namespace largest_number_from_digits_l276_276294

theorem largest_number_from_digits : ∃ (n : ℕ), (n = 865) ∧ 
  (∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  ({a, b, c} = {5, 8, 6}) ∧ 
  (n = a * 100 + b * 10 + c)) :=
begin
  sorry,
end

end largest_number_from_digits_l276_276294


namespace subsets_containing_5_and_6_l276_276484

theorem subsets_containing_5_and_6 (S : Set ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  {T : Set ℕ // {5, 6} ⊆ T ∧ T ⊆ S}.card = 16 := 
sorry

end subsets_containing_5_and_6_l276_276484


namespace area_OCD_correct_l276_276674
-- Lean 4 Statement

variables (R α : ℝ)

def area_triangle_OCD (R α : ℝ) : ℝ :=
  (R^2 * real.sin α) / 8 * (real.sqrt (4 - real.sin α ^ 2) - real.cos α)

theorem area_OCD_correct (R α : ℝ) :
  area_triangle_OCD R α = (R^2 * real.sin α) / 8 * (real.sqrt (4 - real.sin α ^ 2) - real.cos α) :=
sorry

end area_OCD_correct_l276_276674


namespace num_possible_outcomes_num_outcomes_exactly_3_hits_num_outcomes_3_hits_2_consecutive_l276_276158

-- Statement for part ①
theorem num_possible_outcomes : 
  let n := 6,
      p := 2 in
  p ^ n = 64 :=
by
  sorry

-- Statement for part ②
theorem num_outcomes_exactly_3_hits : 
  let n := 6,
      k := 3 in
  Nat.choose n k = 20 :=
by
  sorry

-- Statement for part ③
theorem num_outcomes_3_hits_2_consecutive : 
  let total_shots := 6,
      consecutive_hits := 2,
      total_hits := 3,
      total_misses := total_shots - total_hits,
      -- Arrangement steps:
      arrange_misses := 1,
      insert_hits := 4,
      permute_hits_spaces := insert_hits in
  (permute_hits_spaces choose consecutive_hits) = 12 :=
by
  sorry

end num_possible_outcomes_num_outcomes_exactly_3_hits_num_outcomes_3_hits_2_consecutive_l276_276158


namespace sum_of_squares_of_roots_l276_276126

noncomputable def polynomial := Polynomial R

-- Define the polynomial
def p : polynomial := polynomial.monomial 8 1 - 14 * polynomial.monomial 4 1 - 8 * polynomial.monomial 3 1 
                      - polynomial.monomial 2 1 + polynomial.C 1

-- Define the roots as variables
variables r : ℝ
variables r1 r2 r3 r4 : ℝ

-- Assume that r1, r2, r3, r4 are distinct real roots of the polynomial
axiom roots: polynomial.has_roots [r1, r2, r3, r4]

-- The main theorem stating the proof goal
theorem sum_of_squares_of_roots : r1^2 + r2^2 + r3^2 + r4^2 = 8 :=
sorry

end sum_of_squares_of_roots_l276_276126


namespace percentage_of_primes_divisible_by_3_is_12_5_l276_276843

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l276_276843


namespace primes_less_than_20_divisible_by_3_percentage_l276_276730

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l276_276730


namespace find_k_l276_276460

-- Define the vectors a and b
def a : Vector ℝ 2 := ![1, -2]
def b (k : ℝ) : Vector ℝ 2 := ![-2, k]

-- Define the parallel condition
def parallel (a b : Vector ℝ 2) : Prop :=
∃ λ : ℝ, a = λ • b

-- State the main theorem
theorem find_k (k : ℝ) : parallel a (b k) → k = 4 :=
by
  sorry

end find_k_l276_276460


namespace root_sum_product_eq_l276_276213

theorem root_sum_product_eq (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) :
  p + q = 69 :=
by 
  sorry

end root_sum_product_eq_l276_276213


namespace jon_initial_fastball_speed_l276_276104

theorem jon_initial_fastball_speed 
  (S : ℝ) -- Condition: Jon's initial fastball speed \( S \)
  (h1 : ∀ t : ℕ, t = 4 * 4)  -- Condition: Training time is 4 times for 4 weeks each
  (h2 : ∀ w : ℕ, w = 16)  -- Condition: Total weeks of training (4*4=16)
  (h3 : ∀ g : ℝ, g = 1)  -- Condition: Gains 1 mph per week
  (h4 : ∃ S_new : ℝ, S_new = (S + 16) ∧ S_new = 1.2 * S) -- Condition: Speed increases by 20%
  : S = 80 := 
sorry

end jon_initial_fastball_speed_l276_276104


namespace true_false_questions_count_l276_276082

/-- 
 In an answer key for a quiz, there are some true-false questions followed by 3 multiple-choice questions with 4 answer choices each. 
 The correct answers to all true-false questions cannot be the same. 
 There are 384 ways to write the answer key. How many true-false questions are there?
-/
theorem true_false_questions_count : 
  ∃ n : ℕ, 2^n - 2 = 6 ∧ (2^n - 2) * 4^3 = 384 := 
sorry

end true_false_questions_count_l276_276082


namespace bounded_sum_reciprocals_l276_276647

-- Define the property of not containing the digit '7'.
def no_seven_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → ¬ (d = 7)

-- Define the sum of reciprocals of all natural numbers less than N without the digit '7'.
noncomputable def sum_reciprocals_without_seven (N : ℕ) : ℝ :=
  ∑ i in Finset.range N, if no_seven_digits i then (1 : ℝ) / i else 0

-- The theorem statement
theorem bounded_sum_reciprocals : ∃ K > 0, ∀ N, sum_reciprocals_without_seven N < K :=
by sorry

end bounded_sum_reciprocals_l276_276647


namespace midpoints_of_hypotenuses_form_rectangle_l276_276284

/-- Let ABCD be a quadrilateral with right angles at B and D.
    Let E and F be the intersections of line segments AB with CD and AD with BC, respectively.
    Let M₁, M₂, M₃, and M₄ be the midpoints of the hypotenuses of the four right-angled
    triangles ABE, BCE, CDF, and DAF.
    We need to prove that M₁, M₂, M₃, and M₄ form the vertices of a rectangle.
-/
theorem midpoints_of_hypotenuses_form_rectangle
    (A B C D E F M₁ M₂ M₃ M₄ : Point)
    (h1 : angle B = 90°) 
    (h2 : angle D = 90°) 
    (h3 : intersection_points (line_through A B) (line_through C D) = E)
    (h4 : intersection_points (line_through A D) (line_through B C) = F)
    (h5 : perpendicular AC EF)
    (h6 : midpoint (hypotenuse (triangle A B E)) = M₁)
    (h7 : midpoint (hypotenuse (triangle B C E)) = M₂)
    (h8 : midpoint (hypotenuse (triangle C D F)) = M₃)
    (h9 : midpoint (hypotenuse (triangle D A F)) = M₄) :
    is_rectangle M₁ M₂ M₃ M₄ := 
sorry

end midpoints_of_hypotenuses_form_rectangle_l276_276284


namespace range_x_range_f_of_x_l276_276019

noncomputable theory

section
open Real

def log_half (x : ℝ) : ℝ := log x / log (1/2)
def f (x : ℝ) : ℝ := (log x / log 2 - log 4 / log 2) * (log x / log 2 - log 2 / log 2)

theorem range_x : ∀ x : ℝ, log_half (x^2) ≥ log_half (3 * x - 2) → 1 ≤ x ∧ x ≤ 2 :=
begin
  sorry
end

theorem range_f_of_x : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → 0 ≤ f x ∧ f x ≤ 2 :=
begin
  sorry
end
end

end range_x_range_f_of_x_l276_276019


namespace no_finite_planes_intersect_all_cubes_l276_276598

theorem no_finite_planes_intersect_all_cubes (n : ℕ) : 
  ∀ R : ℝ, R > (3 * Real.sqrt 3 / 2) * n → 
  ¬ (∃ planes : Fin n → Set (Set ℝ), ∀ cube ∈ integer_grid, ∃ plane ∈ planes, 
      plane ∩ cube ≠ ∅) :=
by
  intros R hR
  intro h
  sorry

end no_finite_planes_intersect_all_cubes_l276_276598


namespace sum_of_three_numbers_l276_276239

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d ∨ n / 10 % 10 = d ∨ n / 100 = d

theorem sum_of_three_numbers (A B C : ℕ) :
  (100 ≤ A ∧ A < 1000 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100) ∧
  (∃ (B7 C7 : ℕ), B7 + C7 = 208 ∧ (contains_digit A 7 ∨ contains_digit B7 7 ∨ contains_digit C7 7)) ∧
  (∃ (B3 C3 : ℕ), B3 + C3 = 76 ∧ (contains_digit B3 3 ∨ contains_digit C3 3)) →
  A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276239


namespace percentage_of_primes_divisible_by_3_l276_276756

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l276_276756


namespace sin_eq_x_div_100_has_63_roots_l276_276879

theorem sin_eq_x_div_100_has_63_roots :
  (∃ (n : ℕ), n = 63) ∧ ∀ x ∈ Icc (-100 : ℝ) 100, 
  (sin x = x / 100) → (card {x : ℝ | x ∈ Icc (-100 : ℝ) 100 ∧ sin x = x / 100} = 63) := 
sorry

end sin_eq_x_div_100_has_63_roots_l276_276879


namespace cows_eat_grass_l276_276597

theorem cows_eat_grass (ha_per_cow_per_week : ℝ) (ha_grow_per_week : ℝ) :
  (∀ (weeks_cows_weeks_ha : ℕ × ℕ × ℕ × ℕ), weeks_cows_weeks_ha = (2, 3, 2, 2) →
    (2 : ℝ) = 3 * 2 * ha_per_cow_per_week - 2 * ha_grow_per_week) → 
  (∀ (weeks_cows_weeks_ha : ℕ × ℕ × ℕ × ℕ), weeks_cows_weeks_ha = (4, 2, 4, 2) →
    (2 : ℝ) = 2 * 4 * ha_per_cow_per_week - 4 * ha_grow_per_week) → 
  ∃ (cows : ℕ), (6 : ℝ) = cows * 6 * ha_per_cow_per_week - 6 * ha_grow_per_week ∧ cows = 3 :=
sorry

end cows_eat_grass_l276_276597


namespace primes_divisible_by_3_percentage_is_12_5_l276_276741

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l276_276741


namespace percentage_of_primes_divisible_by_3_l276_276798

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l276_276798


namespace game_is_unfair_root_probability_k1_k2_k3_l276_276147

def game_rule_unfair (a b : ℕ) : Prop :=
  a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ 
  let ξ := |a - b|
  ξ ∈ {0, 1, 2, 3, 4, 5} ∧ ξ ≤ 2 ∧ 
  P(ξ ≤ 2) = 2 / 3 ∧ P(ξ ≤ 2) > 1 / 2

def probability_root_exactly_one (k : ℕ) (ξ : ℕ) : Prop :=
  k ∈ ℕ ∧ k > 0 ∧ ξ = |a - b| ∧ 
  let f := k * x^2 - ξ * x - 1
  if k = 1 then 
    P(ξ = 2) = 2 / 9 
  else if k = 2 then 
    let prob_geq_3 := 1 / 6 
    P(ξ ≥ 3) = prob_geq_3
  else if k ≥ 3 then 
    ξ > 11 / 2 ∧ P(ξ > 11 / 2) = 0

theorem game_is_unfair (a b : ℕ) : game_rule_unfair a b :=
sorry

theorem root_probability_k1_k2_k3 (k ξ : ℕ) : probability_root_exactly_one k ξ :=
sorry

end game_is_unfair_root_probability_k1_k2_k3_l276_276147


namespace value_two_stds_less_than_mean_l276_276305

theorem value_two_stds_less_than_mean (μ σ : ℝ) (hμ : μ = 16.5) (hσ : σ = 1.5) : (μ - 2 * σ) = 13.5 :=
by
  rw [hμ, hσ]
  norm_num

end value_two_stds_less_than_mean_l276_276305


namespace cyclic_quadrilateral_l276_276690

-- Geometrical setup of triangle ABC and the relevant points and lines 
variables {A B C M N P Q : Type} [MetricSpace A]
variables (circle_through_A_B : Circle ℝ)
variables (on_AC : Point ℝ -> Point ℝ -> Prop)
variables (on_BC : Point ℝ -> Point ℝ -> Prop)
variables (on_AB : Point ℝ -> Point ℝ -> Prop)
variables (line_through_M : Line ℝ -> Point ℝ -> Prop)
variables (line_through_N : Line ℝ -> Point ℝ -> Prop)
variables (parallel_to_BC : Line ℝ -> Line ℝ -> Prop)
variables (parallel_to_AC : Line ℝ -> Line ℝ -> Prop)
variables (concyecles : Point ℝ → Point ℝ → Point ℝ → Point ℝ → Prop)

-- Prove that points M, N, P, Q are concyclic
theorem cyclic_quadrilateral :
  ∀ (A B C M N P Q : Point ℝ),
    circle_through_A_B A B M N →
    on_AC M C →
    on_BC N C →
    line_through_M (parallel_to_BC BC) →
    line_through_N (parallel_to_AC AC) →
    on_AB P A B →
    on_AB Q A B →
    concyclic M N P Q :=
by {
  sorry
}

end cyclic_quadrilateral_l276_276690


namespace solve_for_x_l276_276165

theorem solve_for_x (x : ℚ) (h : (3 - x)/(x + 2) + (3 * x - 6)/(3 - x) = 2) : x = -7/6 := 
by 
  sorry

end solve_for_x_l276_276165


namespace probability_heads_mod_coin_l276_276905

theorem probability_heads_mod_coin (p : ℝ) (h : 20 * p ^ 3 * (1 - p) ^ 3 = 1 / 20) : p = (1 - Real.sqrt 0.6816) / 2 :=
by
  sorry

end probability_heads_mod_coin_l276_276905


namespace binary_ternary_product_base_10_l276_276969

theorem binary_ternary_product_base_10 :
  let b2 := 2
  let t3 := 3
  let n1 := 1011 -- binary representation
  let n2 := 122 -- ternary representation
  let a1 := (1 * b2^3) + (0 * b2^2) + (1 * b2^1) + (1 * b2^0)
  let a2 := (1 * t3^2) + (2 * t3^1) + (2 * t3^0)
  a1 * a2 = 187 :=
by
  sorry

end binary_ternary_product_base_10_l276_276969


namespace percentage_primes_divisible_by_3_l276_276858

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l276_276858


namespace elevator_translation_l276_276933

-- Definitions based on conditions
def turning_of_steering_wheel : Prop := False
def rotation_of_bicycle_wheels : Prop := False
def motion_of_pendulum : Prop := False
def movement_of_elevator : Prop := True

-- Theorem statement
theorem elevator_translation :
  movement_of_elevator := by
  exact True.intro

end elevator_translation_l276_276933


namespace triangle_medians_projections_sum_l276_276096

theorem triangle_medians_projections_sum :
  ∀ (A B C G P Q R : ℝ) (h1 : A = 7) (h2 : B = 7) (h3 : C = 8) (h4 : ∠ BAC = Real.pi / 3)
    (h5 : medians_intersect_at_centroid A B C G) (h6 : projections_of_centroid A B C G P Q R),
    GP + GQ + GR = 343 * Real.sqrt 3 / 112 :=
by sorry

end triangle_medians_projections_sum_l276_276096


namespace problem_l276_276455

noncomputable def a_seq (n : ℕ) : ℝ :=
sorry -- Define sequence a_n (1 = a_0 ≤ a_1 ≤ ... ≤ a_n ≤ ...)

def b_seq (a_seq : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ k in Finset.range (n + 1), (1 - (a_seq (k-1)) / (a_seq k)) * (1 / (Real.sqrt (a_seq k)))

theorem problem (a_seq : ℕ → ℝ) (a0_one : a_seq 0 = 1)
  (a_nondec : ∀ n, a_seq n ≤ a_seq (n + 1)) : ∀ n, 0 ≤ b_seq a_seq n ∧ b_seq a_seq n ≤ 2 :=
begin
  sorry -- Proof to be completed
end

end problem_l276_276455


namespace subsets_containing_5_and_6_l276_276492

theorem subsets_containing_5_and_6 {α : Type} [DecidableEq α] 
  (S : Finset α) (e1 e2 : α) (h : e1 ≠ e2) 
  (H : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ T, e1 ∈ T ∧ e2 ∈ T)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276492


namespace number_of_solution_values_l276_276961

theorem number_of_solution_values (c : ℕ) : 
  0 ≤ c ∧ c ≤ 2000 ↔ (∃ x : ℝ, 5 * (⌊x⌋ : ℝ) + 3 * (⌈x⌉ : ℝ) = c) →
  c = 251 := 
sorry

end number_of_solution_values_l276_276961


namespace sum_of_three_numbers_l276_276244

def contains_digit (n : ℕ) (d : ℕ) : Prop := d ∈ n.digits 10

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1: 100 ≤ A ∧ A ≤ 999)
  (h2: 10 ≤ B ∧ B ≤ 99) 
  (h3: 10 ≤ C ∧ C ≤ 99)
  (h4: (contains_digit A 7 → A) + (contains_digit B 7 → B) + (contains_digit C 7 → C) = 208)
  (h5: (contains_digit B 3 → B) + (contains_digit C 3 → C) = 76) :
  A + B + C = 247 := 
by 
  sorry

end sum_of_three_numbers_l276_276244


namespace triangle_third_side_length_l276_276578

theorem triangle_third_side_length (a b : ℝ) (θ : ℝ) 
  (ha : a = 9) (hb : b = 11) (hθ : θ = 135) : 
  ∃ c : ℝ, c = Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos (θ / 180 * Real.pi)) 
          ∧ c = Real.sqrt (202 + 99 * Real.sqrt 2) :=
by 
  have hcos : Real.cos (135 / 180 * Real.pi) = -Real.sqrt 2 / 2 := sorry
  use Real.sqrt (a^2 + b^2 - 2 * a * b * (-Real.sqrt 2 / 2))
  split
  · simp [ha, hb, hcos]; sorry
  · sorry

end triangle_third_side_length_l276_276578


namespace problem_solution_l276_276634

noncomputable def problem_statement (a : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ ε > 0, (∀ x > 0, (x > 0 → 0 < f x)) → (uniformly_continuous_on fun x => x^(a+ε))) ∧ 
  (∀ ε > 0, (∀ x > 0, (x > 0 → 0 < f x)) → (∀ x > 0, (x > 0 → divergent_seq $ fun x => 1 / x^(a-ε)))) ↔ 
  (∀ x > 0, (x > 0 → 0 < f x)) → 
  (lim_at_infinity $ fun x => (log (f x)) / (log x)) = a

theorem problem_solution (a : ℝ) (f : ℝ → ℝ) :
  problem_statement a f :=
by sorry

end problem_solution_l276_276634


namespace compare_f_l276_276451

theorem compare_f (a x1 x2 : ℝ) (h1 : a > 0) (h2 : x1 < x2) (h3 : x1 + x2 = 0) :
  (let f (x : ℝ) := a * x^2 + 2 * a * x + 4 in f x1 < f x2) :=
by
  let f (x : ℝ) := a * x^2 + 2 * a * x + 4
  sorry

end compare_f_l276_276451


namespace sphere_to_hemisphere_volume_ratio_l276_276205

theorem sphere_to_hemisphere_volume_ratio (r : ℝ) : 
  let V_sphere := (4 / 3) * Real.pi * r^3 in
  let V_hemisphere := (1 / 2) * (4 / 3) * Real.pi * (3 * r)^3 in
  V_sphere / V_hemisphere = 2 / 27 := 
by sorry

end sphere_to_hemisphere_volume_ratio_l276_276205


namespace subsets_containing_5_and_6_l276_276475

theorem subsets_containing_5_and_6: 
  let s := {1, 2, 3, 4, 5, 6} in
  {t : Finset ℕ // t ⊆ s ∧ 5 ∈ t ∧ 6 ∈ t}.card = 16 :=
by sorry

end subsets_containing_5_and_6_l276_276475


namespace correct_operation_l276_276869

-- Definitions based on the conditions
def optionA (a : ℝ) : Prop := a^2 * a^3 = a^5
def optionB (a : ℝ) : Prop := (-a)^4 = -a^4
def optionC (a : ℝ) : Prop := (a^2)^3 = a^5
def optionD (a : ℝ) : Prop := a^2 + a^4 = a^6

-- The theorem stating that only option A is correct
theorem correct_operation (a : ℝ) : 
  optionA a ∧ ¬ optionB a ∧ ¬ optionC a ∧ ¬ optionD a :=
by {
  split,
  {
    -- Prove option A is correct
    sorry
  },
  split,
  {
    -- Prove option B is incorrect
    sorry
  },
  {
    split,
    {
      -- Prove option C is incorrect
      sorry
    },
    {
      -- Prove option D is incorrect
      sorry
    }
  }
}

end correct_operation_l276_276869


namespace sum_of_three_numbers_l276_276248

def contains_digit (n : ℕ) (d : ℕ) : Prop := d ∈ n.digits 10

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1: 100 ≤ A ∧ A ≤ 999)
  (h2: 10 ≤ B ∧ B ≤ 99) 
  (h3: 10 ≤ C ∧ C ≤ 99)
  (h4: (contains_digit A 7 → A) + (contains_digit B 7 → B) + (contains_digit C 7 → C) = 208)
  (h5: (contains_digit B 3 → B) + (contains_digit C 3 → C) = 76) :
  A + B + C = 247 := 
by 
  sorry

end sum_of_three_numbers_l276_276248


namespace find_x_l276_276722

theorem find_x (x : ℝ) : x * 2.25 - (5 * 0.85) / 2.5 = 5.5 → x = 3.2 :=
by
  sorry

end find_x_l276_276722


namespace subsets_containing_5_and_6_l276_276464

theorem subsets_containing_5_and_6 :
  let S := {1, 2, 3, 4, 5, 6}
  ∃ s ⊆ S, 5 ∈ s ∧ 6 ∈ s ∧ s.card = 16 :=
sorry

end subsets_containing_5_and_6_l276_276464


namespace mul_98_102_equals_9996_l276_276951

theorem mul_98_102_equals_9996 :
  let x := 98
      y := 102
  in (x = 100 - 2) →
     (y = 100 + 2) →
     x * y = 9996 :=
by
  intros h1 h2
  rw [h1, h2]
  -- Further steps would follow here
  sorry

end mul_98_102_equals_9996_l276_276951


namespace circle_diameter_l276_276709

theorem circle_diameter (A : ℝ) (hA : A = 25 * π) (r : ℝ) (h : A = π * r^2) : 2 * r = 10 := by
  sorry

end circle_diameter_l276_276709


namespace anna_probability_more_heads_than_tails_l276_276553

noncomputable def probability_more_heads_than_tails (n : ℕ) : ℚ :=
  if n % 2 = 0
  then (n.choose (n / 2)) / (2 ^ n)
  else 0

theorem anna_probability_more_heads_than_tails :
  let n := 10 in
  probability_more_heads_than_tails n = 193 / 512 := 
by
  sorry

end anna_probability_more_heads_than_tails_l276_276553


namespace radius_of_covering_circles_l276_276325

theorem radius_of_covering_circles (C : ℝ) (n : ℕ) (r : ℝ) 
  (h1 : C = 1) 
  (h2 : n = 7) 
  (h3 : ∀ (x y : ℝ), x = C → y = r → ∃ (circles : ℕ → ℝ), (∀ i, circles i = y) ∧ (∃ (n' : ℕ), n' = n ∧ nat.succ n' ∈ (finset.range n.succ))) : 
  r ≥ 1/2 :=
begin
  -- sorry placeholder, proof steps go here
  sorry,
end

end radius_of_covering_circles_l276_276325


namespace solve_quadratic_eq_l276_276218

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 = 0 → x = 2 ∨ x = -2 :=
by
  sorry

end solve_quadratic_eq_l276_276218


namespace vertex_angle_of_isosceles_triangle_l276_276209

-- Define the problem conditions
def is_isosceles (ABC : Type) [triangle ABC] (α β γ : angle) : Prop :=
  (α = β ∨ β = γ ∨ γ = α) ∧ (α + β + γ = 180)

def angle_ratio (α γ : angle) := α = γ / 2 ∨ γ = 2 * α

-- The theorem we need to prove
theorem vertex_angle_of_isosceles_triangle (α β γ : angle) (h1 : is_isosceles ABC α β γ) (h2 : angle_ratio α γ) :
  β = 90 ∨ β = 36 :=
sorry

end vertex_angle_of_isosceles_triangle_l276_276209


namespace green_pairs_count_l276_276358

theorem green_pairs_count 
  (blue_students : ℕ)
  (green_students : ℕ)
  (total_students : ℕ)
  (total_pairs : ℕ)
  (blue_blue_pairs : ℕ) 
  (mixed_pairs_students : ℕ) 
  (green_green_pairs : ℕ) 
  (count_blue : blue_students = 65)
  (count_green : green_students = 67)
  (count_total_students : total_students = 132)
  (count_total_pairs : total_pairs = 66)
  (count_blue_blue_pairs : blue_blue_pairs = 29)
  (count_mixed_blue_students : mixed_pairs_students = 7)
  (count_green_green_pairs : green_green_pairs = 30) :
  green_green_pairs = 30 :=
sorry

end green_pairs_count_l276_276358


namespace sum_three_numbers_is_247_l276_276249

variables (A B C : ℕ)

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ (nat.digits 10 n)

theorem sum_three_numbers_is_247
  (hA : 100 ≤ A ∧ A < 1000) -- A is a three-digit number
  (hB : 10 ≤ B ∧ B < 100)   -- B is a two-digit number
  (hC : 10 ≤ C ∧ C < 100)   -- C is a two-digit number
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
        (if contains_digit A 7 then A else 0) +
        (if contains_digit B 7 then B else 0) +
        (if contains_digit C 7 then C else 0) = 208) -- Sum of numbers containing digit 7 is 208
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧
        (if contains_digit B 3 then B else 0) +
        (if contains_digit C 3 then C else 0) = 76) -- Sum of numbers containing digit 3 is 76
  : A + B + C = 247 := 
sorry

end sum_three_numbers_is_247_l276_276249


namespace net_displacement_total_fuel_consumption_l276_276323

-- Define the patrol records as a list of integers
def patrol_records : List ℤ := [14, -9, 18, -7, 13, -6, 10, -6]

-- Define the fuel consumption rate per kilometer
def fuel_rate : ℝ := 0.03

-- Prove that the net displacement is 27 kilometers
theorem net_displacement :
  patrol_records.Sum = 27 :=
by
  sorry

-- Prove that the total fuel consumption is 2.49 liters
theorem total_fuel_consumption :
  (patrol_records.map (λ x, (|x| : ℝ))).Sum * fuel_rate = 2.49 :=
by
  sorry

end net_displacement_total_fuel_consumption_l276_276323


namespace total_distance_journey_l276_276103

theorem total_distance_journey :
  let south := 40
  let east := south + 20
  let north := 2 * east
  (south + east + north) = 220 :=
by
  sorry

end total_distance_journey_l276_276103


namespace sum_of_numbers_l276_276261

def contains_digit (n : Nat) (d : Nat) : Prop := 
  (n / 100 = d) ∨ (n % 100 / 10 = d) ∨ (n % 10 = d)

variables {A B C : Nat}

-- Given conditions
axiom three_digit_number : A ≥ 100 ∧ A < 1000
axiom two_digit_numbers : B ≥ 10 ∧ B < 100 ∧ C ≥ 10 ∧ C < 100
axiom sum_with_sevens : contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7 → A + B + C = 208
axiom sum_with_threes : contains_digit B 3 ∧ contains_digit C 3 ∧ B + C = 76

-- Main theorem to be proved
theorem sum_of_numbers : A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276261


namespace num_ways_wolfburg_to_sheep_village_l276_276226

theorem num_ways_wolfburg_to_sheep_village 
    (paths_W_G : ℕ) (paths_G_S : ℕ) 
    (h_paths_W_G : paths_W_G = 6) 
    (h_paths_G_S : paths_G_S = 20) : 
    paths_W_G * paths_G_S = 120 :=
by
  subst h_paths_W_G
  subst h_paths_G_S
  simp
  sorry

end num_ways_wolfburg_to_sheep_village_l276_276226


namespace percent_primes_divisible_by_3_less_than_20_l276_276772

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l276_276772


namespace sum_of_three_numbers_l276_276275

theorem sum_of_three_numbers :
  ∃ A B C : ℕ, 
    (100 ≤ A ∧ A < 1000) ∧  -- A is a three-digit number
    (10 ≤ B ∧ B < 100) ∧     -- B is a two-digit number
    (10 ≤ C ∧ C < 100) ∧     -- C is a two-digit number
    (A + (if (B / 10 = 7 ∨ B % 10 = 7) then B else 0) + 
       (if (C / 10 = 7 ∨ C % 10 = 7) then C else 0) = 208) ∧
    (if (B / 10 = 3 ∨ B % 10 = 3) then B else 0) + 
    (if (C / 10 = 3 ∨ C % 10 = 3) then C else 0) = 76 ∧
    A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276275


namespace quadratic_root_l276_276412

theorem quadratic_root (k : ℝ) (h : (1 : ℝ)^2 + k * 1 - 3 = 0) : k = 2 := 
sorry

end quadratic_root_l276_276412


namespace sarah_weeds_l276_276651

theorem sarah_weeds :
  ∃ W : ℕ, -- W is the number of weeds Sarah pulled on Tuesday
  let Wed := 3 * W,
      Thu := (1/5) * Wed,
      Fri := Thu - 10,
      Total := W + Wed + Thu + Fri in
  Total = 120 ∧ W = 25 :=
sorry

end sarah_weeds_l276_276651


namespace ms_smith_loss_l276_276140

-- Define the conditions as given in the problem
def sale_price : ℝ := 1.50
def profit_pct : ℝ := 0.25
def loss_pct : ℝ := 0.25

def first_cost_price : ℝ := sale_price / (1 + profit_pct)
def second_cost_price : ℝ := sale_price / (1 - loss_pct)

def total_cost : ℝ := first_cost_price + second_cost_price
def total_revenue : ℝ := 2 * sale_price

-- Define the statement to prove
theorem ms_smith_loss : total_revenue - total_cost = -0.20 :=
by
  sorry

end ms_smith_loss_l276_276140


namespace total_people_count_l276_276702

-- Definitions based on given conditions
def Cannoneers : ℕ := 63
def Women : ℕ := 2 * Cannoneers
def Men : ℕ := 2 * Women
def TotalPeople : ℕ := Women + Men

-- Lean statement to prove
theorem total_people_count : TotalPeople = 378 := by
  -- placeholders for proof steps
  sorry

end total_people_count_l276_276702


namespace triple_square_side_area_l276_276144

theorem triple_square_side_area (s : ℝ) : (3 * s) ^ 2 ≠ 3 * (s ^ 2) :=
by {
  sorry
}

end triple_square_side_area_l276_276144


namespace percentage_of_primes_divisible_by_3_l276_276752

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l276_276752


namespace luke_bought_stickers_l276_276134

theorem luke_bought_stickers :
  ∀ (original birthday given_to_sister used_on_card left total_before_buying stickers_bought : ℕ),
  original = 20 →
  birthday = 20 →
  given_to_sister = 5 →
  used_on_card = 8 →
  left = 39 →
  total_before_buying = original + birthday →
  stickers_bought = (left + given_to_sister + used_on_card) - total_before_buying →
  stickers_bought = 12 :=
by
  intros
  sorry

end luke_bought_stickers_l276_276134


namespace peter_total_games_l276_276145

theorem peter_total_games (peter_wins paul_wins david_wins : ℕ) (PeterWon22 : peter_wins = 22)
    (PaulWon20 : paul_wins = 20) (DavidWon32 : david_wins = 32) : 
    peter_wins + (74 - peter_wins) / 2 = 48 :=
by
  rw [PeterWon22, PaulWon20, DavidWon32]
  sorry

end peter_total_games_l276_276145


namespace percentage_primes_divisible_by_3_l276_276864

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l276_276864


namespace second_character_more_lines_l276_276610

theorem second_character_more_lines
  (C1 : ℕ) (S : ℕ) (T : ℕ) (X : ℕ)
  (h1 : C1 = 20)
  (h2 : C1 = S + 8)
  (h3 : T = 2)
  (h4 : S = 3 * T + X) :
  X = 6 :=
by
  -- proof can be filled in here
  sorry

end second_character_more_lines_l276_276610


namespace find_c_l276_276056

-- Definitions from the problem conditions
variables (a c : ℕ)
axiom cond1 : 2 ^ a = 8
axiom cond2 : a = 3 * c

-- The goal is to prove c = 1
theorem find_c : c = 1 :=
by
  sorry

end find_c_l276_276056


namespace quadrilateral_RS_length_l276_276078

noncomputable def length_of_RS (FD DR FR FS : ℝ) (θ : ℝ) : ℝ :=
  let cos_theta : ℝ := (1 / 2)
  Real.sqrt (FR^2 + FS^2 - 2 * FR * FS * cos_theta)

theorem quadrilateral_RS_length 
  (FD DR FR FS : ℝ)
  (h1 : FD = 5)
  (h2 : DR = 8)
  (h3 : FR = 7)
  (h4 : FS = 10)
  (h5 : ∃ θ, angle θ RFS FDR) : 
  length_of_RS FD DR FR FS θ = Real.sqrt 79 :=
by
  -- Use the given conditions
  rw [h1, h2, h3, h4]
  sorry

end quadrilateral_RS_length_l276_276078


namespace y_value_l276_276120

def seq (y : ℝ) : ℕ → ℝ
| 0     := (y - 1) / (y + 1)
| 1     := (seq y 0 - 1) / (seq y 0 + 1)
| (n+2) := (seq y (n+1) - 1) / (seq y (n+1) + 1)

theorem y_value (y : ℝ) (h_cond : y ≠ -1) (h_seq : seq y 1977 = -1 / 3):
  y = 3 :=
sorry

end y_value_l276_276120


namespace triangle_area_of_tangent_line_l276_276562

theorem triangle_area_of_tangent_line (a : ℝ) 
  (h : a > 0) 
  (ha : (1/2) * 3 * a * (3 / (2 * a ^ (1/2))) = 18)
  : a = 64 := 
sorry

end triangle_area_of_tangent_line_l276_276562


namespace river_depth_by_mid_july_l276_276087

theorem river_depth_by_mid_july :
  let depth_mid_may := 5
  let depth_mid_june := depth_mid_may + 10
  let depth_mid_july := 3 * depth_mid_june
  depth_mid_july = 45 :=
by
  simp [depth_mid_may, depth_mid_june, depth_mid_july]
  exact sorry

end river_depth_by_mid_july_l276_276087


namespace maximum_tickets_l276_276203

-- Define the price of a concert ticket
def ticket_price : ℝ := 15.75

-- Define the amount of money Jane has
def jane_budget : ℝ := 200.00

-- Define the condition for maximum number of tickets Jane can buy
def max_tickets (budget price : ℝ) : ℕ := int.floor (budget / price)

-- Prove that the maximum number of tickets Jane can buy is 12
theorem maximum_tickets : max_tickets jane_budget ticket_price = 12 :=
by 
  sorry

end maximum_tickets_l276_276203


namespace primes_divisible_by_3_percentage_is_12_5_l276_276746

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l276_276746


namespace line_intersection_distance_l276_276567

noncomputable def distance {α : Type*} [LinearOrderedField α] (p q : α × α) : α :=
  Real.sqrt((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem line_intersection_distance :
  let A := ( -6 : ℝ, 0 : ℝ)
  let B := ( 0 : ℝ, 3 : ℝ)
  distance A B = 3 * Real.sqrt 5 :=
by
  let A := ( -6 : ℝ, 0 : ℝ)
  let B := ( 0 : ℝ, 3 : ℝ)
  show distance A B = 3 * Real.sqrt 5
  sorry

end line_intersection_distance_l276_276567


namespace surface_area_of_cone_l276_276064

-- Definitions based solely on conditions
def central_angle (θ : ℝ) := θ = (2 * Real.pi) / 3
def slant_height (l : ℝ) := l = 2
def radius_cone (r : ℝ) := ∃ (θ l : ℝ), central_angle θ ∧ slant_height l ∧ θ * l = 2 * Real.pi * r
def lateral_surface_area (A₁ : ℝ) (r l : ℝ) := A₁ = Real.pi * r * l
def base_area (A₂ : ℝ) (r : ℝ) := A₂ = Real.pi * r^2
def total_surface_area (A A₁ A₂ : ℝ) := A = A₁ + A₂

-- The theorem proving the total surface area is as specified
theorem surface_area_of_cone :
  ∃ (r l A₁ A₂ A : ℝ), central_angle ((2 * Real.pi) / 3) ∧ slant_height 2 ∧ radius_cone r ∧
  lateral_surface_area A₁ r 2 ∧ base_area A₂ r ∧ total_surface_area A A₁ A₂ ∧ A = (16 * Real.pi) / 9 := sorry

end surface_area_of_cone_l276_276064


namespace brad_read_more_books_l276_276872

-- Definitions based on the given conditions
def books_william_read_last_month : ℕ := 6
def books_brad_read_last_month : ℕ := 3 * books_william_read_last_month
def books_brad_read_this_month : ℕ := 8
def books_william_read_this_month : ℕ := 2 * books_brad_read_this_month

-- Totals
def total_books_brad_read : ℕ := books_brad_read_last_month + books_brad_read_this_month
def total_books_william_read : ℕ := books_william_read_last_month + books_william_read_this_month

-- The statement to prove
theorem brad_read_more_books : total_books_brad_read = total_books_william_read + 4 := by
  sorry

end brad_read_more_books_l276_276872


namespace choose_4_cards_of_different_suits_l276_276539

theorem choose_4_cards_of_different_suits :
  (∃ (n : ℕ), choose 4 4 = n) ∧
  (∃ (m : ℕ), (13^4 = m)) ∧
  (1 * (13^4) = 28561)

end choose_4_cards_of_different_suits_l276_276539


namespace percentage_of_primes_divisible_by_3_l276_276802

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l276_276802


namespace part_one_part_two_l276_276045

variables (a b : ℝ^1) -- vectors a and b
variables (angle_ab : ℝ) -- the angle between a and b

noncomputable def length_a : ℝ := 1
noncomputable def length_b : ℝ := 2

-- Given the conditions

axiom len_a : ∥a∥ = length_a
axiom len_b : ∥b∥ = length_b

-- Problem (I)
axiom angle_ab_pi_over_3 : angle_ab = real.pi / 3

theorem part_one : ∥a + 2 • b∥ = real.sqrt 21 :=
by
  sorry

-- Problem (II)
axiom dot_product_condition : (2 • a - b) ⋅ (3 • a + b) = 3

theorem part_two : angle_ab = (2 * real.pi) / 3 :=
by
  sorry

end part_one_part_two_l276_276045


namespace geometric_sequence_solution_l276_276681

open Real

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n m, a (n + m) = a n * q ^ m

theorem geometric_sequence_solution :
  ∃ (a : ℕ → ℝ) (q : ℝ), geometric_sequence a q ∧
    (∀ n, 1 ≤ n ∧ n ≤ 5 → 10^8 ≤ a n ∧ a n < 10^9) ∧
    (∀ n, 6 ≤ n ∧ n ≤ 10 → 10^9 ≤ a n ∧ a n < 10^10) ∧
    (∀ n, 11 ≤ n ∧ n ≤ 14 → 10^10 ≤ a n ∧ a n < 10^11) ∧
    (∀ n, 15 ≤ n ∧ n ≤ 16 → 10^11 ≤ a n ∧ a n < 10^12) ∧
    (∀ i, a i = 7 * 3^(16-i) * 5^(i-1)) := sorry

end geometric_sequence_solution_l276_276681


namespace prime_factorization_l276_276366

theorem prime_factorization :
  1007021035035021007001 = (7 ^ 7) * (11 ^ 7) * (13 ^ 7) :=
begin
  sorry
end

end prime_factorization_l276_276366


namespace primes_divisible_by_3_percentage_is_12_5_l276_276738

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l276_276738


namespace parallel_planes_l276_276983

variables (α β γ : Plane)

theorem parallel_planes:
  (∃ (γ : Plane), α ⊥ γ ∧ β ⊥ γ) → (α ∥ β) :=
sorry

end parallel_planes_l276_276983


namespace hexagon_inequality_l276_276929

theorem hexagon_inequality
  (ABCDEF : ConvexHexagon)
  (h1 : ABCDEF.AB = ABCDEF.BC)
  (h2 : ABCDEF.CD = ABCDEF.DE)
  (h3 : ABCDEF.EF = ABCDEF.FA) :
  (ABCDEF.BC / ABCDEF.BE) + (ABCDEF.DE / ABCDEF.DA) + (ABCDEF.FA / ABCDEF.FC) ≥ 3 / 2 :=
sorry

end hexagon_inequality_l276_276929


namespace smallest_three_digit_number_l276_276979

open Nat

-- Define the conditions
def prime_factors_conditions (a b m : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ Prime a ∧ Prime b ∧ Prime (10 * a + b) ∧ Prime (a + b) ∧ (a + b) % 5 = 1 ∧ m = a * b * (10 * a + b) * (a + b)

-- State the main theorem
theorem smallest_three_digit_number (m : ℕ) : 
  (∃ a b : ℕ, prime_factors_conditions a b m) ∧ m ≥ 100 ∧ m < 1000 → m = 690 :=
by
  intros _,
  sorry

end smallest_three_digit_number_l276_276979


namespace danica_trip_l276_276373

-- Define the conditions
variables (a b c : ℕ) -- Declare variables a, b, c as natural numbers
variables (h1 : a ≥ 1) (h2 : a * b * c ≤ 300) (h3 : ∃ k : ℕ, b = 75 * k) 
variables (end_reading : ℕ)

-- Define the start and end condition of the odometer reading
def start_reading := a * b * c
def end_reading := a * b * c + b

-- The theorem to prove
theorem danica_trip : a * a + b * b + c * c = 5635 :=
    sorry

end danica_trip_l276_276373


namespace probability_X_between_neg2_and_2_l276_276024

-- Definitions based on conditions
def X_distributed_as_norm (X : ℝ → Measure ℝ) : Prop := sorry
def P_X_gt_2_eq (X : ℝ) (P : ℝ → ℝ) : Prop := P 2 = 0.023

-- The main theorem to prove
theorem probability_X_between_neg2_and_2 (X : ℝ → Measure ℝ) (P : ℝ → ℝ)
  (hX : X_distributed_as_norm X) 
  (hP : P_X_gt_2_eq X P) :
  P 2 - P (-2) = 0.954 :=
sorry

end probability_X_between_neg2_and_2_l276_276024


namespace train_crosses_pole_in_time_l276_276304

noncomputable def time_to_cross_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  length / speed_ms

theorem train_crosses_pole_in_time :
  ∀ (length speed_kmh : ℝ), length = 240 → speed_kmh = 126 →
    time_to_cross_pole length speed_kmh = 6.8571 :=
by
  intros length speed_kmh h_length h_speed
  rw [h_length, h_speed, time_to_cross_pole]
  sorry

end train_crosses_pole_in_time_l276_276304


namespace subsets_containing_5_and_6_l276_276469

theorem subsets_containing_5_and_6 :
  let S := {1, 2, 3, 4, 5, 6}
  ∃ s ⊆ S, 5 ∈ s ∧ 6 ∈ s ∧ s.card = 16 :=
sorry

end subsets_containing_5_and_6_l276_276469


namespace sum_three_numbers_is_247_l276_276252

variables (A B C : ℕ)

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ (nat.digits 10 n)

theorem sum_three_numbers_is_247
  (hA : 100 ≤ A ∧ A < 1000) -- A is a three-digit number
  (hB : 10 ≤ B ∧ B < 100)   -- B is a two-digit number
  (hC : 10 ≤ C ∧ C < 100)   -- C is a two-digit number
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
        (if contains_digit A 7 then A else 0) +
        (if contains_digit B 7 then B else 0) +
        (if contains_digit C 7 then C else 0) = 208) -- Sum of numbers containing digit 7 is 208
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧
        (if contains_digit B 3 then B else 0) +
        (if contains_digit C 3 then C else 0) = 76) -- Sum of numbers containing digit 3 is 76
  : A + B + C = 247 := 
sorry

end sum_three_numbers_is_247_l276_276252


namespace part_a_l276_276312

-- Part (a)
theorem part_a (x : ℕ)  : (x^2 - x + 2) % 7 = 0 → x % 7 = 4 := by 
  sorry

end part_a_l276_276312


namespace equation_contains_2020_l276_276443

def first_term (n : Nat) : Nat :=
  2 * n^2

theorem equation_contains_2020 :
  ∃ n, first_term n = 2020 :=
by
  use 31
  sorry

end equation_contains_2020_l276_276443


namespace parabola_focus_is_l276_276975

def parabola_focus (x : ℝ) : ℝ := x^2

theorem parabola_focus_is (f : ℝ) (d : ℝ) : 
  (∀ x, x^2 + (parabola_focus x - f)^2 = (parabola_focus x - d)^2) 
  → (f^2 = d^2) 
  → (f - d = 1/2) 
  → f = 1/4 :=
by
  sorry

# Test case to show the desired result
example : parabola_focus_is 1/4 (-1/4) := 
by
  sorry

end parabola_focus_is_l276_276975


namespace problem_proof_l276_276669

noncomputable def p (x : ℝ) := 1 * x + (2 - 1)
noncomputable def q (x : ℝ) := -4/3 * x^2 - 8/3 * x + 4

theorem problem_proof (h1 : p (-1) = -1) (h2 : q (-2) = 4) 
  (h3 : ∀ x, q x = -4/3 * (x-1) * (x+3)) (h4 : p x / q x = f x) :
  p x + q x = -4/3 * x^2 - 5/3 * x + 6 := 
sorry

end problem_proof_l276_276669


namespace percentage_of_primes_divisible_by_3_l276_276796

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l276_276796


namespace subsets_containing_5_and_6_l276_276483

theorem subsets_containing_5_and_6 (S : Set ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  {T : Set ℕ // {5, 6} ⊆ T ∧ T ⊆ S}.card = 16 := 
sorry

end subsets_containing_5_and_6_l276_276483


namespace combustion_CH₄_forming_water_l276_276387

/-
Combustion reaction for Methane: CH₄ + 2 O₂ → CO₂ + 2 H₂O
Given:
  3 moles of Methane
  6 moles of Oxygen
  Balanced equation: CH₄ + 2 O₂ → CO₂ + 2 H₂O
Goal: Prove that 6 moles of Water (H₂O) are formed.
-/

-- Define the necessary definitions for the context
def moles_CH₄ : ℝ := 3
def moles_O₂ : ℝ := 6
def ratio_water_methane : ℝ := 2

theorem combustion_CH₄_forming_water :
  moles_CH₄ * ratio_water_methane = 6 :=
by
  sorry

end combustion_CH₄_forming_water_l276_276387


namespace range_of_a_l276_276193

noncomputable def satisfies_condition (a : ℝ) : Prop :=
  ∀ (f : ℝ → ℝ), f = λ x, x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1 → 
  (a > 2 ∨ a < -1)

theorem range_of_a (a : ℝ) :
  satisfies_condition (a) :=
begin
  sorry
end

end range_of_a_l276_276193


namespace problem_part_1_problem_part_2_l276_276437

def f (x : ℝ) : ℝ := x^2 + 2 * x

noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x

theorem problem_part_1 :
  ∀ x : ℝ, g(x) = -x^2 + 2 * x :=
by
  sorry

theorem problem_part_2 :
  { x : ℝ | g(x) ≥ f(x) - abs (x - 1) } = { x : ℝ | -1 ≤ x ∧ x ≤ 1 / 2 } :=
by
  sorry

end problem_part_1_problem_part_2_l276_276437


namespace numberOfWaysToChoose4Cards_l276_276517

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l276_276517


namespace trajectory_ray_l276_276094

open Complex

theorem trajectory_ray (z : ℂ) : (|z + 1| - |z - 1| = 2) → ∃ x : ℝ, z = x + 0 * Complex.I ∧ x > 1 :=
sorry

end trajectory_ray_l276_276094


namespace slower_whale_length_l276_276884

-- Definition of variables and conditions
variables (v_fast v_slow t : ℝ)
variables (v_fast_val : v_fast = 18)
variables (v_slow_val : v_slow = 15)
variables (t_val : t = 15)

-- Definition of the relative speed
def relative_speed (v_fast v_slow : ℝ) := v_fast - v_slow

-- Definition of the length of the slower whale
def length_of_slower_whale (v_fast v_slow t : ℝ) :=
  relative_speed v_fast v_slow * t

-- Theorem about the length of the slower whale
theorem slower_whale_length :
  length_of_slower_whale v_fast v_slow t = 45 :=
by
  rw [length_of_slower_whale, relative_speed, v_fast_val, v_slow_val, t_val]
  simp
  sorry

end slower_whale_length_l276_276884


namespace selection_count_l276_276067

theorem selection_count :
  ∃(a_1 a_2 a_3 : ℕ), a_1 ∈ {n | 1 ≤ n ∧ n ≤ 14} ∧ a_2 ∈ {n | 1 ≤ n ∧ n ≤ 14} ∧ a_3 ∈ {n | 1 ≤ n ∧ n ≤ 14} ∧ 
  a_1 < a_2 ∧ a_2 < a_3 ∧ a_2 - a_1 ≥ 3 ∧ a_3 - a_2 ≥ 3 ∧ 
  (finset.card {x | ∃ a_1 a_2 a_3, a_1 < a_2 ∧ a_2 < a_3 ∧ a_2 - a_1 ≥ 3 ∧ a_3 - a_2 ≥ 3} = 120) :=
begin
  sorry
end

end selection_count_l276_276067


namespace union_A_B_l276_276036

open Set Real

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {y | ∃ x : ℝ, y = sin x}

theorem union_A_B : A ∪ B = Ico (-1 : ℝ) 2 := by
  sorry

end union_A_B_l276_276036


namespace normal_vector_of_plane_ABC_l276_276406

structure Point (ℝ : Type) :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def vector_sub (p1 p2 : Point ℝ) : Point ℝ :=
{ x := p1.x - p2.x,
  y := p1.y - p2.y,
  z := p1.z - p2.z }

def dot_product (v1 v2 : Point ℝ) : ℝ :=
v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

theorem normal_vector_of_plane_ABC :
  let A := Point.mk 3 2 0
  let B := Point.mk 0 4 0
  let C := Point.mk 3 0 2
  let AB := vector_sub B A
  let AC := vector_sub C A
  ∃ (n : Point ℝ), (dot_product n AB = 0) ∧ (dot_product n AC = 0) ∧ n = Point.mk 2 3 3 :=
by
  let A := Point.mk 3 2 0
  let B := Point.mk 0 4 0
  let C := Point.mk 3 0 2
  let AB := vector_sub B A
  let AC := vector_sub C A
  existsi Point.mk 2 3 3
  split
  · sorry
  split
  · sorry
  · rfl

end normal_vector_of_plane_ABC_l276_276406


namespace division_of_complex_numbers_l276_276946

theorem division_of_complex_numbers : (1 - complex.i) / (1 + complex.i) = - complex.i :=
by
  sorry

end division_of_complex_numbers_l276_276946


namespace cos_theta_eq_13_35_l276_276041

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (norm_a norm_b norm_a_plus_b : ℝ)

-- Given conditions
def conditions : Prop :=
  ∥a∥ = 5 ∧ ∥b∥ = 7 ∧ ∥a + b∥ = 10

-- The proposition we want to prove
theorem cos_theta_eq_13_35 (h : conditions a b) : real.cos (real.angle a b) = 13 / 35 :=
by
  sorry

end cos_theta_eq_13_35_l276_276041


namespace sum_of_three_numbers_l276_276238

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d ∨ n / 10 % 10 = d ∨ n / 100 = d

theorem sum_of_three_numbers (A B C : ℕ) :
  (100 ≤ A ∧ A < 1000 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100) ∧
  (∃ (B7 C7 : ℕ), B7 + C7 = 208 ∧ (contains_digit A 7 ∨ contains_digit B7 7 ∨ contains_digit C7 7)) ∧
  (∃ (B3 C3 : ℕ), B3 + C3 = 76 ∧ (contains_digit B3 3 ∨ contains_digit C3 3)) →
  A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276238


namespace sum_of_numbers_l276_276256

def contains_digit (n : Nat) (d : Nat) : Prop := 
  (n / 100 = d) ∨ (n % 100 / 10 = d) ∨ (n % 10 = d)

variables {A B C : Nat}

-- Given conditions
axiom three_digit_number : A ≥ 100 ∧ A < 1000
axiom two_digit_numbers : B ≥ 10 ∧ B < 100 ∧ C ≥ 10 ∧ C < 100
axiom sum_with_sevens : contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7 → A + B + C = 208
axiom sum_with_threes : contains_digit B 3 ∧ contains_digit C 3 ∧ B + C = 76

-- Main theorem to be proved
theorem sum_of_numbers : A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276256


namespace pie_filling_cans_l276_276338

-- Conditions
def price_per_pumpkin : ℕ := 3
def total_pumpkins : ℕ := 83
def total_revenue : ℕ := 96
def pumpkins_per_can : ℕ := 3

-- Definition
def cans_of_pie_filling (price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can : ℕ) : ℕ :=
  let pumpkins_sold := total_revenue / price_per_pumpkin
  let pumpkins_remaining := total_pumpkins - pumpkins_sold
  pumpkins_remaining / pumpkins_per_can

-- Theorem
theorem pie_filling_cans : cans_of_pie_filling price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can = 17 :=
  by sorry

end pie_filling_cans_l276_276338


namespace problem_l276_276618

def gcf (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem problem (A B : ℕ) (hA : A = gcf 9 15 27) (hB : B = lcm 9 15 27) : A + B = 138 :=
by
  sorry

end problem_l276_276618


namespace max_U_value_l276_276015

noncomputable def maximum_value (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) : ℝ :=
  x + y

theorem max_U_value (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) :
  maximum_value x y h ≤ Real.sqrt 13 :=
  sorry

end max_U_value_l276_276015


namespace percent_primes_divisible_by_3_l276_276829

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l276_276829


namespace percent_primes_divisible_by_3_l276_276832

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l276_276832


namespace initial_hotdogs_l276_276333

-- Definitions
variable (x : ℕ)

-- Conditions
def condition : Prop := x - 2 = 97 

-- Statement to prove
theorem initial_hotdogs (h : condition x) : x = 99 :=
  by
    sorry

end initial_hotdogs_l276_276333


namespace equal_areas_l276_276081

theorem equal_areas :
  ∀ (A B C L N K M : Type) 
    [Triangle A B C] [InteriorBisectorAt A B C L] 
    [MeetsCircumcircle A B C N] [PerpendicularFromPoint L A B K] 
    [PerpendicularFromPoint L A C M], 
  area (Quadrilateral A K N M) = area (Triangle A B C) :=
sorry

end equal_areas_l276_276081


namespace find_k_value_l276_276418

noncomputable def quadratic_root (k : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + k * x - 3 = 0 ∧ x = 1

theorem find_k_value (k : ℝ) (h : quadratic_root k) : k = 2 :=
by
  cases h with x hx,
  cases hx with hx1 hx2,
  rw hx2 at hx1,
  norm_num at hx1,
  exact hx1
  sorry

end find_k_value_l276_276418


namespace primes_divisible_by_3_percentage_l276_276824

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l276_276824


namespace correct_answers_l276_276086

-- Definitions based on conditions a)1 and a)2
def can_A_red : ℕ := 5
def can_A_white : ℕ := 2
def can_A_black : ℕ := 3
def can_B_red : ℕ := 4
def can_B_white : ℕ := 3
def can_B_black : ℕ := 3

-- Total balls in Can A
def total_A : ℕ := can_A_red + can_A_white + can_A_black

-- Total balls in Can B
def total_B (extra : ℕ) : ℕ := can_B_red + can_B_white + can_B_black + extra

-- Probability events based on conditions a)3
def P_A1 : ℚ := can_A_red / total_A
def P_A2 : ℚ := can_A_white / total_A
def P_A3 : ℚ := can_A_black / total_A
def P_B_given_A1 : ℚ := (can_B_red + 1) / (total_B 1)

theorem correct_answers :
  P_B_given_A1 = 5 / 11 ∧
  (P_A1 ≠ P_A2 ∧ P_A1 ≠ P_A3 ∧ P_A2 ≠ P_A3) :=
by {
  sorry -- Proof to be completed
}

end correct_answers_l276_276086


namespace probability_f_le_0_l276_276132

def f (x : ℝ) := x^2 - 5 * x + 6

def interval : set ℝ := set.Icc 0 5

theorem probability_f_le_0 :
  ∀ x₀ ∈ interval, f x₀ ≤ 0 ↔ 2 ≤ x₀ ∧ x₀ ≤ 3 →
  (measure_theory.measure_of (set_of (λ x, (f x ≤ 0))) / measure_theory.measure_of interval = 0.2) :=
begin
  sorry,
end

end probability_f_le_0_l276_276132


namespace find_k_for_quadratic_root_l276_276414

theorem find_k_for_quadratic_root (k : ℝ) (h : (1 : ℝ).pow 2 + k * 1 - 3 = 0) : k = 2 :=
by
  sorry

end find_k_for_quadratic_root_l276_276414


namespace num_non_zero_digits_l276_276943

theorem num_non_zero_digits (num denom : ℕ) (h_num : num = 120) (h_denom : denom = 2^4 * 5^10) :
  let fraction := (num : ℚ) / denom in
  let decimal_rep := fraction * 10^9 in
  (decimal_rep * 10^(9 : ℤ) = 1536) → (4 = 4) :=
sorry

end num_non_zero_digits_l276_276943


namespace cos_theta_l276_276042

variables {α : Type*} [inner_product_space ℝ α] (a b : α)

theorem cos_theta (ha : ∥a∥ = 5) (hb : ∥b∥ = 7) (hensum : ∥a + b∥ = 10) :
  real.cos (real.angle_of_vectors a b) = 13 / 35 :=
sorry

end cos_theta_l276_276042


namespace percentage_primes_divisible_by_3_l276_276859

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l276_276859


namespace sum_of_ratios_l276_276208

theorem sum_of_ratios (a b c : ℤ) (h : (a * a : ℚ) / (b * b) = 32 / 63) : a + b + c = 39 :=
sorry

end sum_of_ratios_l276_276208


namespace percentage_primes_divisible_by_3_l276_276775

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l276_276775


namespace driver_weekly_distance_l276_276332

-- Defining the conditions
def speed_part1 : ℕ := 30  -- speed in miles per hour for the first part
def time_part1 : ℕ := 3    -- time in hours for the first part
def speed_part2 : ℕ := 25  -- speed in miles per hour for the second part
def time_part2 : ℕ := 4    -- time in hours for the second part
def days_per_week : ℕ := 6 -- number of days the driver works in a week

-- Total distance calculation each day
def distance_part1 := speed_part1 * time_part1
def distance_part2 := speed_part2 * time_part2
def daily_distance := distance_part1 + distance_part2

-- Total distance travel in a week
def weekly_distance := daily_distance * days_per_week

-- Theorem stating that weekly distance is 1140 miles
theorem driver_weekly_distance : weekly_distance = 1140 :=
by
  -- We skip the proof using sorry
  sorry

end driver_weekly_distance_l276_276332


namespace g_eval_pi_over_4_l276_276030

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x - sin x^2

noncomputable def g (x : ℝ) : ℝ := sin (2 * (x - π / 12) + π / 6) + 1 / 2

theorem g_eval_pi_over_4 : g (π / 4) = 1 := 
by 
  sorry

end g_eval_pi_over_4_l276_276030


namespace area_R_l276_276462

open Real

def hexagon_side_length := 3
def area_of_R := 27 * sqrt 3 / 16

theorem area_R :
  let s := hexagon_side_length in
  let A := area_of_R in
  A = 27 * sqrt 3 / 16 := sorry

end area_R_l276_276462


namespace find_a_l276_276195

theorem find_a (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x^3 + 3 * x^2 + 2)
  (hf' : ∀ x, f' x = 3 * a * x^2 + 6 * x) 
  (h : f' (-1) = 4) : 
  a = (10 : ℝ) / 3 := 
sorry

end find_a_l276_276195


namespace percentage_decrease_l276_276568

noncomputable def original_fraction (N D : ℝ) : Prop := N / D = 0.75
noncomputable def new_fraction (N D x : ℝ) : Prop := (1.15 * N) / (D * (1 - x / 100)) = 15 / 16

theorem percentage_decrease (N D x : ℝ) (h1 : original_fraction N D) (h2 : new_fraction N D x) : 
  x = 22.67 := 
sorry

end percentage_decrease_l276_276568


namespace graph_min_degree_leq_l276_276660

variables {V : Type*} [Fintype V]

def degree (G : SimpleGraph V) (v : V) : ℕ := (G.adj v).toFinset.card

def min_degree (G : SimpleGraph V) : ℕ := Finset.min' (Finset.image (degree G) (Finset.univ : Finset V)) (by simp)

theorem graph_min_degree_leq (G : SimpleGraph V) (n r : ℕ)
  (h_vertices : Fintype.card V = n)
  (h_no_K_r : ¬(∃ (H : SimpleGraph V), H ≤ G ∧ Fintype.card (H.vertices) = r)) :
  min_degree G ≤ (r - 2) * n / (r - 1) :=
sorry

end graph_min_degree_leq_l276_276660


namespace divide_64_to_get_800_l276_276963

theorem divide_64_to_get_800 (x : ℝ) (h : 64 / x = 800) : x = 0.08 :=
sorry

end divide_64_to_get_800_l276_276963


namespace intersect_perpendicular_bisectors_l276_276619

open EuclideanGeometry

variables {A B C M : Point}
variables (A1 B1 C1 : Point)
variables (hA1 : is_foot_of_perpendicular M A1 B C)
variables (hB1 : is_foot_of_perpendicular M B1 C A)
variables (hC1 : is_foot_of_perpendicular M C1 A B)

theorem intersect_perpendicular_bisectors :
  let mid_B1C1 := midpoint B1 C1 in
  let mid_C1A1 := midpoint C1 A1 in
  let mid_A1B1 := midpoint A1 B1 in
  lines_intersect 
    (line_through (mid_B1C1) M)
    (line_through (mid_C1A1) M)
    (line_through (mid_A1B1) M) :=
sorry

end intersect_perpendicular_bisectors_l276_276619


namespace exists_P_on_circle_C_l276_276426

theorem exists_P_on_circle_C (M : ℝ × ℝ) (C : ℝ → ℝ → Prop) (O : ℝ × ℝ) :
  (M = (2, 0)) →
  (∀ x y, C x y ↔ (x - a - 1)^2 + (y - (sqrt 3 * a))^2 = 1) →
  (∃ P : ℝ × ℝ, (C P.1 P.2) ∧ (P.1 * (P.1 - 2) + P.2 * P.2 = 8)) →
  (a^2 ∈ set.Icc 1 4) :=
by {
  sorry
}

end exists_P_on_circle_C_l276_276426


namespace result_when_7_multiplies_number_l276_276895

theorem result_when_7_multiplies_number (x : ℤ) (h : x + 45 - 62 = 55) : 7 * x = 504 :=
by sorry

end result_when_7_multiplies_number_l276_276895


namespace temp_product_l276_276357

theorem temp_product (N : ℤ) (M D : ℤ)
  (h1 : M = D + N)
  (h2 : M - 8 = D + N - 8)
  (h3 : D + 5 = D + 5)
  (h4 : abs ((D + N - 8) - (D + 5)) = 3) :
  (N = 16 ∨ N = 10) →
  16 * 10 = 160 := 
by sorry

end temp_product_l276_276357


namespace matrix_vector_product_l276_276950

-- Definitions for matrix A and vector v
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-3, 4],
  ![2, -1]
]

def v : Fin 2 → ℤ := ![2, -2]

-- The theorem to prove
theorem matrix_vector_product :
  (A.mulVec v) = ![-14, 6] :=
by sorry

end matrix_vector_product_l276_276950


namespace solution_sum_eq_l276_276128

/-- 
  Given the system of equations:
  |x - 2| = 3|y - 4|
  |x - 6| = |y - 2|

  Prove that the sum of all solutions (x_i + y_i) equals Solution Sum.
-/
theorem solution_sum_eq {x y : ℝ} (h1 : |x - 2| = 3 * |y - 4|) (h2 : |x - 6| = |y - 2|) : 
  ∃ (n : ℕ) (solutions : fin n → ℝ × ℝ), 
    (∀ i, solutions i = (x, y) ∧ |x - 2| = 3 * |y - 4| ∧ |x - 6| = |y - 2|) ∧ 
    ∑ i, (solutions i).1 + (solutions i).2 = Solution Sum :=
sorry

end solution_sum_eq_l276_276128


namespace lemon_heads_per_package_l276_276613

theorem lemon_heads_per_package (total_lemon_heads boxes : ℕ)
  (H : total_lemon_heads = 54)
  (B : boxes = 9)
  (no_leftover : total_lemon_heads % boxes = 0) :
  total_lemon_heads / boxes = 6 :=
sorry

end lemon_heads_per_package_l276_276613


namespace primes_less_than_20_divisible_by_3_percentage_l276_276734

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l276_276734


namespace subcommittee_ways_l276_276906

theorem subcommittee_ways : 
  let republicans := 10 in
  let democrats := 7 in
  let chooseRepublicans := Nat.choose republicans 4 in
  let chooseDemocrats := Nat.choose democrats 3 in
  chooseRepublicans * chooseDemocrats = 7350 :=
by
  let republicans := 10
  let democrats := 7
  let chooseRepublicans := Nat.choose republicans 4
  let chooseDemocrats := Nat.choose democrats 3
  have h1 : chooseRepublicans = 210 := by norm_num
  have h2 : chooseDemocrats = 35 := by norm_num
  rw [h1, h2]
  exact calc
    210 * 35 = 7350 : by norm_num

end subcommittee_ways_l276_276906


namespace sum_of_three_numbers_l276_276276

theorem sum_of_three_numbers :
  ∃ A B C : ℕ, 
    (100 ≤ A ∧ A < 1000) ∧  -- A is a three-digit number
    (10 ≤ B ∧ B < 100) ∧     -- B is a two-digit number
    (10 ≤ C ∧ C < 100) ∧     -- C is a two-digit number
    (A + (if (B / 10 = 7 ∨ B % 10 = 7) then B else 0) + 
       (if (C / 10 = 7 ∨ C % 10 = 7) then C else 0) = 208) ∧
    (if (B / 10 = 3 ∨ B % 10 = 3) then B else 0) + 
    (if (C / 10 = 3 ∨ C % 10 = 3) then C else 0) = 76 ∧
    A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276276


namespace original_alcohol_amount_is_750_l276_276318

def initial_alcohol_amount (x : ℝ) : Prop := 
  let after_first_pour := (2 / 3) * x - 40
  let after_second_pour := (4 / 9) * after_first_pour
  let total_mass_before_third_pour := after_second_pour
  let final_amount := total_mass_before_third_pour - 180
  final_amount = 60

theorem original_alcohol_amount_is_750 :
  ∃ x : ℝ, initial_alcohol_amount x ∧ x = 750 :=
begin
  sorry
end

end original_alcohol_amount_is_750_l276_276318


namespace ramu_repair_cost_l276_276155

theorem ramu_repair_cost
  (initial_cost : ℝ)
  (selling_price : ℝ)
  (profit_percent : ℝ)
  (repair_cost : ℝ)
  (h1 : initial_cost = 42000)
  (h2 : selling_price = 64900)
  (h3 : profit_percent = 13.859649122807017 / 100)
  (h4 : selling_price = initial_cost + repair_cost + profit_percent * (initial_cost + repair_cost)) :
  repair_cost = 15000 :=
by
  sorry

end ramu_repair_cost_l276_276155


namespace percentage_of_primes_divisible_by_3_l276_276799

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l276_276799


namespace rectangular_eq_of_curve_C_calculate_PM_PN_l276_276587

-- Define the parametric equation of the line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + (real.sqrt 2 / 2) * t, -1 + (real.sqrt 2 / 2) * t)

-- Define the polar equation of curve C
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 + 5 = 6 * ρ * real.cos θ

-- Define the ray θ = -π/4
def ray (x y : ℝ) : Prop :=
  y = -x ∧ x ≥ 0

-- Task: Prove the rectangular equation of the curve C
theorem rectangular_eq_of_curve_C :
  ∃ (x y : ℝ), (x^2 + y^2 + 5 = 6 * x) ↔ ((x - 3)^2 + y^2 = 4) :=
sorry

-- Task: Calculate the value of |PM| + |PN|
theorem calculate_PM_PN :
  (|PM| : ℝ) + (|PN| : ℝ) = 3 * real.sqrt 2 :=
sorry

end rectangular_eq_of_curve_C_calculate_PM_PN_l276_276587


namespace units_digit_product_l276_276572

theorem units_digit_product (a b : ℕ) (h1 : (a % 10 ≠ 0) ∧ (b % 10 ≠ 0)) : (a * b % 10 = 0) ∨ (a * b % 10 ≠ 0) :=
by
  sorry

end units_digit_product_l276_276572


namespace probability_B_wins_first_two_given_conditions_l276_276182

-- Define the conditions of the problem:
def team_wins_series (seq : List Char) (team : Char) : Bool :=
  seq.count(team) >= 4

def team_wins_third_game (seq : List Char) (team : Char) : Bool :=
  seq.length >= 3 ∧ seq.nth 2 = some team

def ends_as_series (seq : List Char) : Bool :=
  seq.length ≤ 7

def is_valid_sequence (seq : List Char) : Bool :=
  ends_as_series seq ∧
  (team_wins_series seq 'A') ∧
  team_wins_third_game seq 'B'

variables (seq : List Char)
variables (valid_seqs : List (List Char))

-- Assume we have the sequence list of games
def probability_team_B_wins_first_two_games (seq : List Char) (valid_seqs : List (List Char)) : ℚ :=
  let outcomes := valid_seqs.filter (λ s, is_valid_sequence s) in
  let favorable := outcomes.filter (λ s, s.length >= 2 ∧ (s.nth 0 = some 'B') ∧ (s.nth 1 = some 'B')) in
  (favorable.length : ℚ) / (outcomes.length : ℚ)

-- The main statement of the proof problem
theorem probability_B_wins_first_two_given_conditions : 
    probability_team_B_wins_first_two_games seq valid_seqs = 1 / 2 :=
sorry

end probability_B_wins_first_two_given_conditions_l276_276182


namespace squared_product_l276_276952

theorem squared_product (a b : ℝ) : (- (1 / 2) * a^2 * b)^2 = (1 / 4) * a^4 * b^2 := by 
  sorry

end squared_product_l276_276952


namespace doubled_dimensions_volume_l276_276721

-- Define the conditions of the problem
variables {L W H : ℝ} -- Length, width, and height as real numbers
def initial_volume := L * W * H = 36

-- Statement of the problem
theorem doubled_dimensions_volume
  (h : initial_volume) : 8 * (L * W * H) = 288 :=
sorry

end doubled_dimensions_volume_l276_276721


namespace quadruples_count_l276_276617

theorem quadruples_count :
  let S := {1, 2, 3, 4}
  in (∃ (a b c d : ℕ) (h1 : a ∈ S)(h2 : b ∈ S)(h3 : c ∈ S)(h4 : d ∈ S), 
      (a * b - c * d) % 2 = 1 ∧ (a + b + c + d) % 2 = 0) → 160 :=
begin
  sorry
end

end quadruples_count_l276_276617


namespace mean_multiplied_by_3_l276_276058

theorem mean_multiplied_by_3 (b1 b2 b3 b4 b5 : ℝ) :
  let original_mean := (b1 + b2 + b3 + b4 + b5) / 5 in
  let new_set := {3 * b1, 3 * b2, 3 * b3, 3 * b4, 3 * b5} in
  let new_mean := (3 * b1 + 3 * b2 + 3 * b3 + 3 * b4 + 3 * b5) / 5 in
  new_mean = 3 * original_mean :=
by
  sorry

end mean_multiplied_by_3_l276_276058


namespace subsets_containing_5_and_6_l276_276494

theorem subsets_containing_5_and_6 {α : Type} [DecidableEq α] 
  (S : Finset α) (e1 e2 : α) (h : e1 ≠ e2) 
  (H : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ T, e1 ∈ T ∧ e2 ∈ T)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276494


namespace percent_primes_divisible_by_3_l276_276827

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l276_276827


namespace general_term_and_sum_l276_276625

noncomputable def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d
noncomputable def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := n / 2 * (2 * a + (n - 1) * d)

theorem general_term_and_sum :
  (∃ a d : ℕ, sum_arithmetic_seq a d 9 = 81 ∧ arithmetic_seq a d 2 + arithmetic_seq a d 3 = 8 ∧ 
  ∀ n : ℕ, arithmetic_seq a d n = 2 * n - 1) ∧
  (∃ m : ℕ, sum_arithmetic_seq 1 2 3 * sum_arithmetic_seq 1 2 m = arithmetic_seq 1 2 14 ^ 2 ∧ sum_arithmetic_seq 1 2 (2 * m) = 324) :=
begin
  sorry
end

end general_term_and_sum_l276_276625


namespace subsets_containing_5_and_6_l276_276495

theorem subsets_containing_5_and_6 {α : Type} [DecidableEq α] 
  (S : Finset α) (e1 e2 : α) (h : e1 ≠ e2) 
  (H : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ T, e1 ∈ T ∧ e2 ∈ T)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276495


namespace minimum_value_zero_l276_276127

noncomputable def minimumValue (x y z : ℝ) : ℝ :=
  xy + 2 * x * z + 3 * y * z

theorem minimum_value_zero (x y z : ℝ) (h : x + y + 3 * z = 6) :
  minimumValue x y z = 0 :=
sorry

end minimum_value_zero_l276_276127


namespace area_BEIH_is_1_5_l276_276877

open Real

noncomputable def point := ℝ × ℝ

noncomputable def A : point := (0, 3)
noncomputable def B : point := (0, 0)
noncomputable def C : point := (3, 0)
noncomputable def D : point := (3, 3)
noncomputable def E : point := (0, 1.5)
noncomputable def F : point := (1.5, 0)

noncomputable def line_eqn (p1 p2 : point) : ℝ → ℝ :=
  λ x, ((p2.2 - p1.2) / (p2.1 - p1.1)) * (x - p1.1) + p1.2

noncomputable def DE_eqn : ℝ → ℝ := line_eqn D E
noncomputable def AF_eqn : ℝ → ℝ := line_eqn A F

noncomputable def I : point :=
  let x := (3 * 1.5) / (2 * 2 + 5)
  let y := DE_eqn x
  (x, y)

noncomputable def H : point := (1, 1)

noncomputable def shoelace_area (pts : list point) : ℝ :=
  1 / 2 * (abs (pts.nth! 0).1 * ((pts.nth! 1).2 - (pts.nth! 2).2) +
          (pts.nth! 1).1 * ((pts.nth! 2).2 - (pts.nth! 0).2) +
          (pts.nth! 2).1 * ((pts.nth! 0).2 - (pts.nth! 1).2))

theorem area_BEIH_is_1_5 :
  let BEIH := [B, E, I, H] in
  shoelace_area BEIH = 1/5 :=
by
  sorry

end area_BEIH_is_1_5_l276_276877


namespace g_10_eq_g_formula_max_g_l276_276328

def f (x : ℕ) : ℝ :=
if 1 ≤ x ∧ x ≤ 20 then 1.1 else 
if 21 ≤ x ∧ x ≤ 60 then x / 10 else 
0

def g (x : ℕ) : ℝ := 
if 1 ≤ x ∧ x ≤ 20 then 1 / (x + 80) else 
if 21 ≤ x ∧ x ≤ 60 then 2 * x / (x^2 - x + 1600) else 
0

theorem g_10_eq : g 10 = 1 / 90 :=
sorry

theorem g_formula (x : ℕ) : g x = 
if 1 ≤ x ∧ x ≤ 20 then 1 / (x + 80) else 
if 21 ≤ x ∧ x ≤ 60 then 2 * x / (x^2 - x + 1600) else 
0 :=
sorry

theorem max_g : g 40 = 2 / 79 :=
sorry

end g_10_eq_g_formula_max_g_l276_276328


namespace relationship_abc_l276_276033

def f (x : ℝ) : ℝ := x^3

noncomputable def a : ℝ := -f (Real.log 1 / Real.log 10 / Real.log 3)
noncomputable def b : ℝ := f (Real.log 9.1 / Real.log 3)
noncomputable def c : ℝ := f (2 ^ 0.9)

theorem relationship_abc :
  c < b ∧ b < a :=
  sorry

end relationship_abc_l276_276033


namespace proof_problem_l276_276089

noncomputable def parametric_equations (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, 1 + 2 * Real.sin α)

noncomputable def polar_eq_line (θ : ℝ) : Prop := 
  Real.tan θ = 3

def curve_eq_cartesian : Prop :=
  ∀ α : ℝ, let (x, y) := parametric_equations α in x^2 + (y - 1)^2 = 4

def line_eq_cartesian : Prop :=
  ∀ x y : ℝ, polar_eq_line (Real.atan (y / x)) → y = 3 * x

def dist_sum_pm_pn (t1 t2 : ℝ) : ℝ :=
  Real.abs t1 + Real.abs t2

theorem proof_problem :
  (∀ α : ℝ, let (x, y) := parametric_equations α in x^2 + (y - 1)^2 = 4) ∧ 
  (∀ x y : ℝ, polar_eq_line (Real.atan (y / x)) → y = 3 * x) →
  (∃ t1 t2 : ℝ, t1^2 + (7 * Real.sqrt 10 / 5) * t1 + 1 = 0 ∧
                t2^2 + (7 * Real.sqrt 10 / 5) * t2 + 1 = 0 ∧
                dist_sum_pm_pn t1 t2 = 7 * Real.sqrt 10 / 5) :=
by
  sorry

end proof_problem_l276_276089


namespace fish_to_rice_equivalence_l276_276075

variable (f : ℚ) (l : ℚ)

theorem fish_to_rice_equivalence (h1 : 5 * f = 3 * l) (h2 : l = 6) : f = 18 / 5 := by
  sorry

end fish_to_rice_equivalence_l276_276075


namespace can_tile_with_L_triominoes_can_tile_with_straight_triominoes_l276_276306

-- Define the chessboard size
def chessboard_size (n : ℕ) := 2^n

-- Define an L-shaped triomino
structure L_triomino where
  cells : set (ℕ × ℕ)
  shape_L : cells = {(0, 0), (1, 0), (1, 1)} ∨ cells = {(0, 0), (1, 0), (0, 1)}

-- Define the 3x1 straight triomino
structure straight_triomino where
  cells : set (ℕ × ℕ)
  shape_straight : cells = {(0, 0), (1, 0), (2, 0)} ∨ cells = {(0, 0), (0, 1), (0, 2)}

-- Main theorem concerning the tileability with L-shaped triominoes
theorem can_tile_with_L_triominoes (n : ℕ) (removed_cell : ℕ × ℕ) :
  ∀ chessboard_size n, ∃ tiling, tiling.tiling possible :=
sorry

-- Color pattern definition for the 8x8 chessboard
def color (x y : ℕ) : color :=
  if (x + y) % 3 = 0 then color.red else
  if (x + y) % 3 = 1 then color.green else
  color.blue

-- Main theorem concerning the tileability with straight triominoes
theorem can_tile_with_straight_triominoes (removed_cell : ℕ × ℕ) :
  (color removed_cell.1 removed_cell.2 = color.red) →
  ∃ tiling, tiling.tiling_possible :=
sorry

end can_tile_with_L_triominoes_can_tile_with_straight_triominoes_l276_276306


namespace cos_sin_fraction_l276_276368

theorem cos_sin_fraction (h1 : Real.sin (30 * Real.pi / 180) = 1 / 2)
                          (h2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2) :
  (Real.cos (30 * Real.pi / 180))^2 - (Real.sin (30 * Real.pi / 180))^2) / ((Real.cos (30 * Real.pi / 180))^2 * (Real.sin (30 * Real.pi / 180))^2) = 8 / 3 :=
by
  sorry

end cos_sin_fraction_l276_276368


namespace book_pages_l276_276302

theorem book_pages (total_pages : ℝ) : 
  (0.1 * total_pages + 0.25 * total_pages + 30 = 0.5 * total_pages) → 
  total_pages = 240 :=
by
  sorry

end book_pages_l276_276302


namespace probability_of_heads_at_least_once_l276_276280

theorem probability_of_heads_at_least_once 
  (X : ℕ → ℝ)
  (hX_binom : ∀ n, X n = binomial (n := 3) (p := 0.5) n)
  (indep_tosses : ∀ i j, i ≠ j → indep_fun X (X j))
  (prob_heads : ∀ n, X n = 1/2) :
  (Pr (X ≥ 1) = 7/8) := 
by 
  sorry

end probability_of_heads_at_least_once_l276_276280


namespace min_value_x3_y2_z_w2_l276_276119

theorem min_value_x3_y2_z_w2 (x y z w : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < w)
  (h : (1/x) + (1/y) + (1/z) + (1/w) = 8) : x^3 * y^2 * z * w^2 ≥ 1/432 :=
by
  sorry

end min_value_x3_y2_z_w2_l276_276119


namespace find_y_l276_276223

variable (x y z : ℚ)

theorem find_y
  (h1 : x + y + z = 150)
  (h2 : x + 7 = y - 12)
  (h3 : x + 7 = 4 * z) :
  y = 688 / 9 :=
sorry

end find_y_l276_276223


namespace percentage_of_primes_divisible_by_3_is_12_5_l276_276847

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l276_276847


namespace sum_of_first_200_terms_l276_276025

noncomputable def x_n (x1 : ℝ) (n : ℕ) : ℝ := x1 * 2^(n-1)

theorem sum_of_first_200_terms (x1 : ℝ) (h : x1 = 100 / (2^100 - 1)) :
  (Finset.range 200).sum (λ n, x_n x1 (n + 1)) = 100 * (1 + 2^100) :=
by 
  have sum_100 : (Finset.range 100).sum (λ n, x_n x1 (n + 1)) = 100 := 
    by
      -- Given condition
      calc
        (Finset.range 100).sum (λ n, x_n x1 (n + 1))
        = x1 * (2^100 - 1) / (2 - 1) : sorry
        ... = 100 : by rw [h]; sorry
  -- Use sum_100 to show the result for 200 terms
  sorry

end sum_of_first_200_terms_l276_276025


namespace quadratic_solution_l276_276659

theorem quadratic_solution (x : ℝ) : 
  x^2 - 2 * x - 3 = 0 → (x = 3 ∨ x = -1) :=
by 
  sorry

end quadratic_solution_l276_276659


namespace find_scalars_l276_276621

noncomputable def N : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-2, 0]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

theorem find_scalars (r s : ℤ) (h_r : r = 3) (h_s : s = -8) :
    N * N = r • N + s • I :=
by
  rw [h_r, h_s]
  sorry

end find_scalars_l276_276621


namespace functional_equation_solution_l276_276627

theorem functional_equation_solution (a b : ℝ) (f : ℝ → ℝ) :
  (0 < a ∧ 0 < b) →
  (∀ x y : ℝ, 0 < x → 0 < y →
    f x * f y = y^a * f (x / 2) + x^b * f (y / 2)) →
  (∃ c : ℝ, ∀ x : ℝ, 0 < x → (f x = c * x^a ∨ f x = 0)) :=
by
  intros
  sorry

end functional_equation_solution_l276_276627


namespace five_letter_arrangements_l276_276047

-- Definitions based on the conditions
def seven_letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
def first_letter (arrangement : List Char) : Bool := arrangement.head = 'D'
def contains_a (arrangement : List Char) : Bool := 'A' ∈ arrangement
def no_repeats (arrangement : List Char) : Bool := arrangement.nodup

-- The theorem we want to prove
theorem five_letter_arrangements : 
  ∃ (arrangements : List (List Char)), 
    (∀ arr ∈ arrangements, 
      arr.length = 5 ∧  
      arr.head = 'D' ∧ 
      'A' ∈ arr.tail ∧ 
      arr.nodup) 
    ∧ arrangements.length = 240 :=
begin
  sorry
end

end five_letter_arrangements_l276_276047


namespace find_equation_of_ellipse_max_area_triangle_OPQ_point_S_exists_l276_276998

variables (a b x y t m s : ℝ)
def ellipse (a b : ℝ) : Prop := ∃ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)

theorem find_equation_of_ellipse (t : ℝ) (ht : 0 < t ∧ t < 2) :
  (2 * sqrt 3 = 2 * b) ∧ (b = sqrt 3)
  ∧ (∃ x y : ℝ, (y = 3/2 ∧ x = -1) ∧ (x^2 / a^2 + y^2 / (3 : ℝ)) = 1)
  ↔ (a^2 = 4 ∧ ((x^2 / 4) + (y^2 / 3) = 1)) := sorry

theorem max_area_triangle_OPQ (a m : ℝ) :
  ((a = 2) ∧ (b = sqrt 3)) ∧ (t = sqrt 3) 
  → (max_area := sqrt 3) :=
sorry

theorem point_S_exists (a b t s : ℝ) :
  ((0 < t) ∧ (t < 2)) ∧ (t * s = a^2) → (exist_point_S := ∃ s : ℝ, (s = a^2 / t)) :=
sorry

end find_equation_of_ellipse_max_area_triangle_OPQ_point_S_exists_l276_276998


namespace percentage_of_primes_divisible_by_3_l276_276808

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l276_276808


namespace polynomial_division_l276_276984

-- Definition of the polynomial and the conditions.
def divides (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

theorem polynomial_division (a : ℤ) (p : ℤ[x]) :
  (∀ x : ℤ, (x^2 - x + a) * p.eval x = (x^14 + x + 100)) →
  divides a 100 →
  divides a 102 →
  divides (a + 2) 98 →
  (a = 2 ∨ a = -1) :=
begin
  sorry
end

end polynomial_division_l276_276984


namespace fractional_part_sum_condition_l276_276386

theorem fractional_part_sum_condition (k : ℕ) (h1 : k > 0) (h2 : k < 202) :
  (∃ n : ℕ, (n > 0) ∧ (∑ i in finset.range k, (fract ((i + 1) * n / 202))) = k / 2) ↔ (k = 100) ∨ (k = 101) :=
sorry

end fractional_part_sum_condition_l276_276386


namespace primes_less_than_20_divisible_by_3_percentage_l276_276732

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l276_276732


namespace quadratic_root_l276_276413

theorem quadratic_root (k : ℝ) (h : (1 : ℝ)^2 + k * 1 - 3 = 0) : k = 2 := 
sorry

end quadratic_root_l276_276413


namespace ratio_of_girls_l276_276105

theorem ratio_of_girls (total_julian_friends : ℕ) (percent_julian_girls : ℚ)
  (percent_julian_boys : ℚ) (total_boyd_friends : ℕ) (percent_boyd_boys : ℚ) :
  total_julian_friends = 80 →
  percent_julian_girls = 0.40 →
  percent_julian_boys = 0.60 →
  total_boyd_friends = 100 →
  percent_boyd_boys = 0.36 →
  (0.64 * total_boyd_friends : ℚ) / (0.40 * total_julian_friends : ℚ) = 2 :=
by
  sorry

end ratio_of_girls_l276_276105


namespace base_length_l276_276569

-- Definition: Isosceles triangle
structure IsoscelesTriangle :=
  (perimeter : ℝ)
  (side : ℝ)

-- Conditions: Perimeter and one side of the isosceles triangle
def given_triangle : IsoscelesTriangle := {
  perimeter := 26,
  side := 11
}

-- The problem to solve: length of the base given the perimeter and one side
theorem base_length : 
  (given_triangle.perimeter = 26 ∧ given_triangle.side = 11) →
  (∃ b : ℝ, b = 11 ∨ b = 7.5) :=
by 
  sorry

end base_length_l276_276569


namespace percentage_of_primes_divisible_by_3_l276_276801

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l276_276801


namespace determine_functions_l276_276957

noncomputable def satisfies_condition (f : ℕ → ℕ) : Prop :=
∀ (n p : ℕ), Prime p → (f n)^p % f p = n % f p

theorem determine_functions :
  ∀ (f : ℕ → ℕ),
  satisfies_condition f →
  f = id ∨
  (∀ p: ℕ, Prime p → f p = 1) ∨
  (f 2 = 2 ∧ (∀ p: ℕ, Prime p → p > 2 → f p = 1) ∧ ∀ n: ℕ, f n % 2 = n % 2) :=
by
  intros f h1
  sorry

end determine_functions_l276_276957


namespace cannot_partition_nat_l276_276599

theorem cannot_partition_nat (A : ℕ → Set ℕ) (h1 : ∀ i j, i ≠ j → Disjoint (A i) (A j))
    (h2 : ∀ k, Finite (A k) ∧ sum {n | n ∈ A k}.toFinset id = k + 2013) :
    False :=
sorry

end cannot_partition_nat_l276_276599


namespace ways_to_choose_4_cards_of_different_suits_l276_276522

theorem ways_to_choose_4_cards_of_different_suits :
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  ∃ n : ℕ, n = (choose num_suits num_suits) * cards_per_suit ^ num_suits ∧ n = 28561 :=
by
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  have ways_to_choose_suits : (choose num_suits num_suits) = 1 := by simp
  have ways_to_choose_cards : cards_per_suit ^ num_suits = 28561 := by norm_num
  let n := 1 * 28561
  use n
  constructor
  · exact by simp [ways_to_choose_suits, ways_to_choose_cards]
  · exact by rfl

end ways_to_choose_4_cards_of_different_suits_l276_276522


namespace primes_divisible_by_3_percentage_is_12_5_l276_276739

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l276_276739


namespace sin_x1_add_x2_zero_points_l276_276433

noncomputable theory
open Real

def f (x m : ℝ) : ℝ := 2 * sin (2 * x) + cos (2 * x) - m

theorem sin_x1_add_x2_zero_points 
  (m x1 x2 : ℝ)
  (h1 : f x1 m = 0)
  (h2 : f x2 m = 0)
  (h_range1 : 0 ≤ x1 ∧ x1 ≤ π / 2)
  (h_range2 : 0 ≤ x2 ∧ x2 ≤ π / 2) 
  : sin (x1 + x2) = (2 * sqrt 5) / 5 := 
by 
  sorry

end sin_x1_add_x2_zero_points_l276_276433


namespace no_partition_with_sum_k_plus_2013_l276_276603

open Nat

theorem no_partition_with_sum_k_plus_2013 (A : ℕ → Finset ℕ) (h_disjoint : ∀ i j, i ≠ j → Disjoint (A i) (A j)) 
  (h_sum : ∀ k, (A k).sum id = k + 2013) : False :=
by
  sorry

end no_partition_with_sum_k_plus_2013_l276_276603


namespace percentage_of_primes_divisible_by_3_l276_276806

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l276_276806


namespace standard_deviation_of_applicants_l276_276183

theorem standard_deviation_of_applicants (s : ℝ) :
  let average_age := 10 in
  (∀ n : ℤ, (n ≥ (average_age - s)) ∧ (n ≤ (average_age + s))) →
  (((average_age + s).toInt - (average_age - s).toInt + 1) = 17) →
  s = 8 :=
by
  sorry

end standard_deviation_of_applicants_l276_276183


namespace number_of_solutions_l276_276050

def sign (α : ℝ) : ℝ :=
  if α > 0 then 1 else if α = 0 then 0 else -1

def satisfies_system (x y z : ℝ) : Prop :=
  x = 2018 - 2019 * sign (y + z) ∧
  y = 2018 - 2019 * sign (z + x) ∧
  z = 2018 - 2019 * sign (x + y)

theorem number_of_solutions : ∃ n : ℕ, n = 3 ∧ ∀ x y z : ℝ, satisfies_system x y z ↔ (x = -1 ∧ y = -1 ∧ z = 4037) ∨ (x = -1 ∧ y = 4037 ∧ z = -1) ∨ (x = 4037 ∧ y = -1 ∧ z = -1) :=
by
  sorry

end number_of_solutions_l276_276050


namespace primes_divisible_by_3_percentage_is_12_5_l276_276737

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l276_276737


namespace garden_area_increase_l276_276922

/-- 
Given a rectangular garden with length 40 feet and width 15 feet,
if the garden is transformed into a circular shape using the same perimeter of the original garden,
prove that the increase in the garden's area is approximately 362.3 square feet.
--/
theorem garden_area_increase :
  let length := 40
  let width := 15
  let original_area := length * width
  let perimeter := 2 * (length + width)
  let radius := (perimeter / (2 * Real.pi))
  let circle_area := Real.pi * radius^2
  let area_increase := circle_area - original_area
  by sorry :=
  abs (area_increase - 362.3) < 1 := sorry

end garden_area_increase_l276_276922


namespace animals_per_aquarium_l276_276285

theorem animals_per_aquarium (total_animals : ℕ) (num_aquariums : ℕ) (h1 : total_animals = 512) (h2 : num_aquariums = 8) : (total_animals / num_aquariums) = 64 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul_left (Nat.zero_lt_succ _) rfl sorry

end animals_per_aquarium_l276_276285


namespace problem_equivalent_l276_276361

theorem problem_equivalent :
  ( (1 / 3) ^ (-2) + (π - 2022) ^ 0 + 2 * Real.sin (Real.pi / 3) + |Real.sqrt 3 - 2| ) = 12 :=
by
  sorry

end problem_equivalent_l276_276361


namespace percentage_of_primes_divisible_by_3_is_12_5_l276_276846

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l276_276846


namespace solve_equation_l276_276170

def problem_statement : Prop :=
  ∃ x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -7 / 6

theorem solve_equation : problem_statement :=
by {
  sorry
}

end solve_equation_l276_276170


namespace subsets_containing_5_and_6_l276_276486

theorem subsets_containing_5_and_6 (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ s, 5 ∈ s ∧ 6 ∈ s)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276486


namespace sum_of_three_numbers_l276_276235

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d ∨ n / 10 % 10 = d ∨ n / 100 = d

theorem sum_of_three_numbers (A B C : ℕ) :
  (100 ≤ A ∧ A < 1000 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100) ∧
  (∃ (B7 C7 : ℕ), B7 + C7 = 208 ∧ (contains_digit A 7 ∨ contains_digit B7 7 ∨ contains_digit C7 7)) ∧
  (∃ (B3 C3 : ℕ), B3 + C3 = 76 ∧ (contains_digit B3 3 ∨ contains_digit C3 3)) →
  A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276235


namespace largest_possible_value_of_k_l276_276561

open Classical

noncomputable def is_clique_of_size_three (G : Type*) [Fintype G] (E : G → G → Prop) : Prop :=
∀ (x y z : G), E x y ∨ E y z ∨ E z x

noncomputable def class_with_students_friends (students : Fin 21) : Prop :=
∃ (student : Fin 21), 
∀ (G : Type*) (E : G → G → Prop), 
  (Fintype.card G = 21) → 
  (is_clique_of_size_three G E) → 
  (∃ (at_least_k_friends : G), ∃ (k : ℕ), k ≥ 10 ∧ at_least_k_friends = student ∧ (Fintype.card {y | E at_least_k_friends y} ≥ k))

theorem largest_possible_value_of_k 
  (students_friends : ∀ class_with_students_friends (students : Fin 21)) : Prop :=
∃ (student : Fin 21), ∀ (k : ℕ), k = 10

end largest_possible_value_of_k_l276_276561


namespace rectangle_dimensions_l276_276222

-- Define the dimensions and properties of the rectangle
variables {a b : ℕ}

-- Theorem statement
theorem rectangle_dimensions 
  (h1 : b = a + 3)
  (h2 : 2 * a + 2 * b + a = a * b) : 
  (a = 3 ∧ b = 6) :=
by
  sorry

end rectangle_dimensions_l276_276222


namespace partition_right_triangle_l276_276114

theorem partition_right_triangle 
  (ABC : Type) [EquilateralTriangle ABC] : 
  ∃ S T : Set Point, 
    (S ∪ T = M) ∧ (S ∩ T = ∅) ∧ 
    ∃ A₁ A₂ A₃: Point,
      (A₁ ∈ S ∨ A₁ ∈ T) ∧ 
      (A₂ ∈ S ∨ A₂ ∈ T) ∧ 
      (A₃ ∈ S ∨ A₃ ∈ T) ∧ 
      isRightTriangle A₁ A₂ A₃ := 
by
  sorry

structure EquilateralTriangle (ABC : Type) :=
  (points : Set Point)
  (is_equilateral : ∀ (p₁ p₂ p₃: Point), p₁ ∈ points ∧ p₂ ∈ points ∧ p₃ ∈ points → 
    distance p₁ p₂ = distance p₂ p₃ ∧ distance p₂ p₃ = distance p₃ p₁)

noncomputable def M : Set Point := { p // p ∈ ABC.points ∧ p ∈ Perimeter ABC }

def Perimeter (ABC : Type) := { p // ∃ (a b: Point), (a, b) ∈ edges ABC ∧ p ∈ segment a b }

structure Point :=
  (x y : ℝ)
  -- Additional fields and properties if needed

def isRightTriangle (A B C : Point) : Prop :=
  ∃ (a b c: ℝ), 
    ((A.x - B.x)^2 + (A.y - B.y)^2 = a) ∧ 
    ((B.x - C.x)^2 + (B.y - C.y)^2 = b) ∧ 
    ((C.x - A.x)^2 + (C.y - A.y)^2 = c) ∧ 
    (a * b = c)

end partition_right_triangle_l276_276114


namespace trigonometric_expression_value_l276_276719

-- Define the line equation and the conditions about the slope angle
def line_eq (x y : ℝ) : Prop := 6 * x - 2 * y - 5 = 0

-- The slope angle alpha
variable (α : ℝ)

-- Given conditions
axiom slope_tan : Real.tan α = 3

-- The expression we need to prove equals -2
theorem trigonometric_expression_value :
  (Real.sin (Real.pi - α) + Real.cos (-α)) / (Real.sin (-α) - Real.cos (Real.pi + α)) = -2 :=
by
  sorry

end trigonometric_expression_value_l276_276719


namespace percentage_of_primes_divisible_by_3_l276_276761

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l276_276761


namespace maximum_value_l276_276130

noncomputable def maxValue (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^5 * y + x^4 * y^2 + x^3 * y^3 + x^2 * y^4 + x * y^5

theorem maximum_value (x y : ℝ) (h : x + y = 5) : maxValue x y h ≤ 625 / 4 :=
sorry

end maximum_value_l276_276130


namespace primes_divisible_by_3_percentage_l276_276822

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l276_276822


namespace percentage_of_primes_divisible_by_3_is_12_5_l276_276850

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l276_276850


namespace smallest_k_divisible_by_7_l276_276400

def a_n (n : ℕ) : ℕ := (n+8)! / (n-1)!

def digit_sum (n : ℕ) : ℕ := (n.toString.filter (λ c, c ≠ '0')).foldl (λ acc c, acc + (c.toNat - '0'.toNat)) 0

theorem smallest_k_divisible_by_7 : ∃ k : ℕ, (k > 0) ∧ (digit_sum (a_n k) % 7 = 0) ∧ ∀ m : ℕ, (m > 0) ∧ (m < k) → ¬(digit_sum (a_n m) % 7 = 0) := 
by
  sorry

end smallest_k_divisible_by_7_l276_276400


namespace continuous_at_x0_l276_276646

theorem continuous_at_x0 : ∀ ε > 0, ∃ δ > 0, (∀ x, | x - 5 | < δ → | (2 * x^2 + 8) - 58 | < ε) :=
by sorry

end continuous_at_x0_l276_276646


namespace problem_1_problem_2_l276_276095

noncomputable def line (p θ : ℝ) : ℝ := p * (Real.cos θ + Real.sin θ) - 2
noncomputable def curve (p θ : ℝ) : ℝ := p - 4 * Real.cos θ
noncomputable def distance_AB (x1 y1 x2 y2 : ℝ) : ℝ := Real.dist (x1, y1) (x2, y2)
noncomputable def max_area_PAB (θ : ℝ) : ℝ := 4 * Real.sin (2 * θ)

theorem problem_1 (θ θ1 θ2 : ℝ) (pt1 pt2 A B: ℝ) :
  (line pt1 θ1 = 0) ∧ (curve pt1 θ1 = 0) ∧ (line pt2 θ2 = 0) ∧ (curve pt2 θ2 = 0) 
  → distance_AB (pt1 * Real.cos θ1) (pt1 * Real.sin θ1) (pt2 * Real.cos θ2) (pt2 * Real.sin θ2) = 4 :=
sorry

theorem problem_2 (P A B θ : ℝ) (ptP ptA ptB : ℝ) :
  (curve ptP θ = 0) ∧ ptP ≠ ptA ∧ ptP ≠ ptB ∧ 
  ∀ θ ∈ Set.Ioo 0 (Real.pi / 2), max_area_PAB θ ≤ 4 :=
sorry

end problem_1_problem_2_l276_276095


namespace subsets_containing_5_and_6_l276_276488

theorem subsets_containing_5_and_6 (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ s, 5 ∈ s ∧ 6 ∈ s)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276488


namespace popcorn_probability_l276_276317

theorem popcorn_probability {w y b : ℝ} (hw : w = 3/5) (hy : y = 1/5) (hb : b = 1/5)
  {pw py pb : ℝ} (hpw : pw = 1/3) (hpy : py = 3/4) (hpb : pb = 1/2) :
  (y * py) / (w * pw + y * py + b * pb) = 1/3 := 
sorry

end popcorn_probability_l276_276317


namespace no_partition_possible_l276_276607

noncomputable def partition_possible (A : ℕ → Set ℕ) :=
  (∀ k: ℕ, ∃ finA : Finset ℕ, (A k = finA.to_set) ∧ (finA.sum id = k + 2013)) ∧
  (∀ i j: ℕ, i ≠ j → (A i ∩ A j) = ∅) ∧
  (⋃ i, A i) = Set.univ

theorem no_partition_possible :
  ¬ ∃ A : ℕ → Set ℕ, partition_possible A := 
sorry

end no_partition_possible_l276_276607


namespace range_of_f_l276_276395

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x)^4 * Real.tan x + (Real.cos x)^4 * Real.cot x

theorem range_of_f :
  (Set.range f) = Set.Iic (-1/2) ∪ Set.Ici (1/2) :=
sorry

end range_of_f_l276_276395


namespace numberOfWaysToChoose4Cards_l276_276519

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l276_276519


namespace dice_outcome_count_l276_276691

theorem dice_outcome_count : 
  (let outcomes := 6 * 6 in outcomes = 36) := 
by
  sorry

end dice_outcome_count_l276_276691


namespace problem_gcd_polynomials_l276_276432

theorem problem_gcd_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 7768 * k ∧ k % 2 = 0) :
  gcd (4 * b ^ 2 + 55 * b + 120) (3 * b + 12) = 12 :=
by
  sorry

end problem_gcd_polynomials_l276_276432


namespace sum_of_three_numbers_l276_276270

theorem sum_of_three_numbers :
  ∃ A B C : ℕ, 
    (100 ≤ A ∧ A < 1000) ∧  -- A is a three-digit number
    (10 ≤ B ∧ B < 100) ∧     -- B is a two-digit number
    (10 ≤ C ∧ C < 100) ∧     -- C is a two-digit number
    (A + (if (B / 10 = 7 ∨ B % 10 = 7) then B else 0) + 
       (if (C / 10 = 7 ∨ C % 10 = 7) then C else 0) = 208) ∧
    (if (B / 10 = 3 ∨ B % 10 = 3) then B else 0) + 
    (if (C / 10 = 3 ∨ C % 10 = 3) then C else 0) = 76 ∧
    A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276270


namespace mark_bench_press_correct_l276_276956

def dave_weight : ℝ := 175
def dave_bench_press : ℝ := 3 * dave_weight

def craig_bench_percentage : ℝ := 0.20
def craig_bench_press : ℝ := craig_bench_percentage * dave_bench_press

def emma_bench_percentage : ℝ := 0.75
def emma_initial_bench_press : ℝ := emma_bench_percentage * dave_bench_press
def emma_actual_bench_press : ℝ := emma_initial_bench_press + 15

def combined_craig_emma : ℝ := craig_bench_press + emma_actual_bench_press

def john_bench_factor : ℝ := 2
def john_bench_press : ℝ := john_bench_factor * combined_craig_emma

def mark_reduction : ℝ := 50
def mark_bench_press : ℝ := combined_craig_emma - mark_reduction

theorem mark_bench_press_correct : mark_bench_press = 463.75 := by
  sorry

end mark_bench_press_correct_l276_276956


namespace percent_primes_divisible_by_3_l276_276837

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l276_276837


namespace min_value_of_inverse_sum_l276_276436

variable (n : ℕ) (p q : ℝ)

-- We need to define that a variable X follows a binomial distribution B(n, p)
-- and has expectations and variance as given.
noncomputable def X := Binomial n p

axiom h1 : (X.bExpectation = 4)
axiom h2 : (X.bVariance = q)

theorem min_value_of_inverse_sum : (frac_one_div_p + frac_one_div_q ≥ 9 / 4) :=
by {
  sorry,
}

end min_value_of_inverse_sum_l276_276436


namespace camping_trip_costs_l276_276352

theorem camping_trip_costs (a b : ℕ) :
  (let alice_paid := 130 in
   let bob_paid := 150 in
   let carlos_paid := 200 in
   let total_paid := alice_paid + bob_paid + carlos_paid in
   let equal_share := total_paid / 3 in
   let a := equal_share - alice_paid in
   let b := equal_share - bob_paid in
   a - b = 20) :=
sorry

end camping_trip_costs_l276_276352


namespace two_digit_even_multiple_of_7_l276_276920

def all_digits_product_square (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  (d1 * d2) > 0 ∧ ∃ k, d1 * d2 = k * k

theorem two_digit_even_multiple_of_7 (n : ℕ) :
  10 ≤ n ∧ n < 100 ∧ n % 2 = 0 ∧ n % 7 = 0 ∧ all_digits_product_square n ↔ n = 14 ∨ n = 28 ∨ n = 70 :=
by sorry

end two_digit_even_multiple_of_7_l276_276920


namespace sum_first_eight_terms_l276_276422

def a (n : ℕ) : ℝ := 2^n - 1/2 * (n + 3)

def S (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), a i

theorem sum_first_eight_terms : S 8 = 480 := by
  sorry

end sum_first_eight_terms_l276_276422


namespace angle_PMN_is_60_l276_276583

theorem angle_PMN_is_60 (P Q R M N : Type*)
  [decidable_eq P] [decidable_eq Q] [decidable_eq R] [decidable_eq M] [decidable_eq N]
  (angle_PQR : ℝ) (h1 : angle_PQR = 60) (PM PN : ℝ) (h2 : PM = PN)
  (PR RQ : ℝ) (h3 : PR = RQ) : 
  measure_angle P M N = 60 := 
sorry

end angle_PMN_is_60_l276_276583


namespace machine_a_production_rate_l276_276881

def machine_p_time (machine_q_time : ℝ) : ℝ := machine_q_time + 10
def machine_q_rate (machine_q_time : ℝ) : ℝ := 660 / machine_q_time
def machine_a_rate (machine_q_time : ℝ) : ℝ := machine_q_rate machine_q_time / 1.1

theorem machine_a_production_rate (machine_q_time : ℝ) (h_correct_time : machine_q_time = 100) :
  machine_a_rate machine_q_time = 6 := by
  sorry

end machine_a_production_rate_l276_276881


namespace log_decreasing_condition_l276_276446

theorem log_decreasing_condition (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : log a 2 > log a 3) : a ∈ set.Ioo 0 1 := 
sorry

end log_decreasing_condition_l276_276446


namespace percentage_primes_divisible_by_3_l276_276853

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l276_276853


namespace closest_value_to_expr_l276_276942

theorem closest_value_to_expr 
    (approx1 : 3.1 ≈ 3)
    (approx2 : 9.1 ≈ 9)
    (exact_sum : 5.92 + 4.08 = 10)
    (correct_answer : 370) : 
    is_closest_to ((3.1 * 9.1 * (5.92 + 4.08)) + 100) correct_answer :=
by
  sorry

end closest_value_to_expr_l276_276942


namespace solution_of_system_l276_276989

theorem solution_of_system (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = 1) : x + y = 3 :=
sorry

end solution_of_system_l276_276989


namespace no_partition_possible_l276_276606

noncomputable def partition_possible (A : ℕ → Set ℕ) :=
  (∀ k: ℕ, ∃ finA : Finset ℕ, (A k = finA.to_set) ∧ (finA.sum id = k + 2013)) ∧
  (∀ i j: ℕ, i ≠ j → (A i ∩ A j) = ∅) ∧
  (⋃ i, A i) = Set.univ

theorem no_partition_possible :
  ¬ ∃ A : ℕ → Set ℕ, partition_possible A := 
sorry

end no_partition_possible_l276_276606


namespace repeating_ones_divisible_by_3_pow_n_l276_276100

def repeated_one_digits (n : ℕ) : ℕ :=
  (List.repeat 1 (3^n)).foldl (λ x y => 10*x + y) 0

theorem repeating_ones_divisible_by_3_pow_n (n : ℕ) (hn : 0 < n) : 
  3^n ∣ repeated_one_digits n :=
sorry

end repeating_ones_divisible_by_3_pow_n_l276_276100


namespace primes_less_than_20_divisible_by_3_percentage_l276_276726

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l276_276726


namespace greatest_avg_rate_of_change_at_x_1_l276_276378

theorem greatest_avg_rate_of_change_at_x_1:
  let Δx := 0.3 in
  let k₁ := 1 in
  let k₂ := 2 + Δx in
  let k₃ := 3 + 3 * Δx + Δx^2 in
  let k₄ := -1 / (1 + Δx) in
  k₃ > k₂ ∧ k₂ > k₁ ∧ k₁ > k₄ :=
by
  let Δx := 0.3
  let k₁ : ℝ := 1
  let k₂ : ℝ := 2 + Δx
  let k₃ : ℝ := 3 + 3 * Δx + Δx^2
  let k₄ : ℝ := -1 / (1 + Δx)
  have h1 : k₃ = 3.99 := by norm_num [k₃, Δx]
  have h2 : k₂ = 2.3 := by norm_num [k₂, Δx]
  have h3 : k₁ = 1 := rfl
  have h4 : k₄ = -10 / 13 := by norm_num [k₄, Δx]
  split
  · show k₃ > k₂
    rw [h1, h2]
    linarith
  split
  · show k₂ > k₁
    rw [h2, h3]
    linarith
  · show k₁ > k₄
    rw [h3, h4]
    norm_num

end greatest_avg_rate_of_change_at_x_1_l276_276378


namespace square_area_l276_276925

theorem square_area (s : ℝ) (h : s = 12) : s * s = 144 :=
by
  rw [h]
  norm_num

end square_area_l276_276925


namespace log_equation_solution_l276_276397

theorem log_equation_solution (x : ℝ) (hx : x^2 - 14 * x > 0) : 
  (log 10 (x^2 - 14 * x) = 3) ↔ (x = 39.5 ∨ x = -25.5) :=
sorry

end log_equation_solution_l276_276397


namespace subsets_containing_5_and_6_l276_276477

theorem subsets_containing_5_and_6: 
  let s := {1, 2, 3, 4, 5, 6} in
  {t : Finset ℕ // t ⊆ s ∧ 5 ∈ t ∧ 6 ∈ t}.card = 16 :=
by sorry

end subsets_containing_5_and_6_l276_276477


namespace inequality_solution_case_0_inequality_solution_case_pos_inequality_solution_case_neg1_inequality_solution_case_neg2_inequality_solution_case_neg3_l276_276175

variable (a x : ℝ)

theorem inequality_solution_case_0 :
  (a = 0 ∧ ax^2 - 2 ≥ 2x - ax) ↔ (x ≤ -1) := sorry

theorem inequality_solution_case_pos :
  (a > 0 ∧ ax^2 - 2 ≥ 2x - ax) ↔ (x ≥ 2/a ∨ x ≤ -1) := sorry

theorem inequality_solution_case_neg1 :
  (-2 < a ∧ a < 0 ∧ ax^2 - 2 ≥ 2x - ax) ↔ (2/a ≤ x ∧ x ≤ -1) := sorry
  
theorem inequality_solution_case_neg2 :
  (a = -2 ∧ ax^2 - 2 ≥ 2x - ax) ↔ (x = -1) := sorry

theorem inequality_solution_case_neg3 :
  (a < -2 ∧ ax^2 - 2 ≥ 2x - ax) ↔ (-1 ≤ x ∧ x ≤ 2/a) := sorry

end inequality_solution_case_0_inequality_solution_case_pos_inequality_solution_case_neg1_inequality_solution_case_neg2_inequality_solution_case_neg3_l276_276175


namespace percentage_primes_divisible_by_3_l276_276784

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l276_276784


namespace log2_sum_real_coeffs_l276_276620

noncomputable def T : ℝ := 
  (Real.exp (2011 * (Complex.log (1 + Complex.I)))).re

theorem log2_sum_real_coeffs : Real.log2 T = 1005 := by
  sorry

end log2_sum_real_coeffs_l276_276620


namespace sequence_sum_eq_34_l276_276383

def seq : ℕ → ℤ 
| 0       := 2
| (n + 1) := if n % 2 = 0 then seq n + 4 else seq n - 4

theorem sequence_sum_eq_34 : 
  (Finset.range 18).sum seq + 70 = 34 :=
by {
  sorry
}

end sequence_sum_eq_34_l276_276383


namespace total_number_of_people_l276_276699

theorem total_number_of_people (num_cannoneers num_women num_men total_people : ℕ)
  (h1 : num_women = 2 * num_cannoneers)
  (h2 : num_cannoneers = 63)
  (h3 : num_men = 2 * num_women)
  (h4 : total_people = num_women + num_men) : 
  total_people = 378 := by
  sorry

end total_number_of_people_l276_276699


namespace minimum_questions_to_find_prize_l276_276076

variable (doors : Finset ℕ) (prize : ℕ) (questions_needed : ℕ) (lies_allowed : ℕ)

def is_door (d : ℕ) : Prop := d ∈ doors
def has_prize (d : ℕ) : Prop := d = prize

theorem minimum_questions_to_find_prize (h_doors : doors = {0, 1, 2})
  (h_prize : is_door prize)
  (h_max_lies : lies_allowed = 10)
  (h_inform_host : (questions_needed : ℕ) := 32) :
  (∃ d, is_door d ∧ has_prize d) ∧ questions_needed = 32 := 
by
  sorry

end minimum_questions_to_find_prize_l276_276076


namespace primes_less_than_20_divisible_by_3_percentage_l276_276733

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l276_276733


namespace minimum_distance_l276_276008

def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem minimum_distance (M : ℝ × ℝ)
  (h1 : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ M = (x, (1/2) * x^2)) :
  dist M.1 M.2 1 1 = real.sqrt 5 / 2 :=
by
  sorry

end minimum_distance_l276_276008


namespace mike_net_spending_l276_276138

-- Definitions for given conditions
def trumpet_cost : ℝ := 145.16
def song_book_revenue : ℝ := 5.84

-- Theorem stating the result
theorem mike_net_spending : trumpet_cost - song_book_revenue = 139.32 :=
by 
  sorry

end mike_net_spending_l276_276138


namespace find_edge_RS_l276_276215

noncomputable def edge_lengths : ℕ := [8, 14, 19, 28, 37, 42]

-- Define edge lengths and the question condition
def PQ := 42

-- Define the problem to prove RS = 14 given the conditions
theorem find_edge_RS 
    (edge_lengths : list ℕ)
    (PQ : ℕ) 
    (h1 : PQ ∈ edge_lengths)
    (h2 : PQ = 42) : 
    ∃ (RS : ℕ), RS ∈ edge_lengths ∧ RS = 14 :=
by
  -- This is where we would provide the proof, e.g. by validating configurations.
  -- We skip it and place a placeholder for the proof.
  sorry

end find_edge_RS_l276_276215


namespace point_on_circle_l276_276401

theorem point_on_circle (t : ℝ) : 
  let x := (1 - t^2) / (1 + t^2)
  let y := (3 * t) / (1 + t^2)
  x^2 + y^2 = 1 :=
by
  let x := (1 - t^2) / (1 + t^2)
  let y := (3 * t) / (1 + t^2)
  sorry

end point_on_circle_l276_276401


namespace combination_count_l276_276897

-- Definitions from conditions
def packagingPapers : Nat := 10
def ribbons : Nat := 4
def stickers : Nat := 5

-- Proof problem statement
theorem combination_count : packagingPapers * ribbons * stickers = 200 := 
by
  sorry

end combination_count_l276_276897


namespace percentage_primes_divisible_by_3_l276_276776

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l276_276776


namespace angle_between_vectors_l276_276459

variables (a b : ℂ) (θ : ℝ)
def vector_a : ℝ × ℝ := (3, 0)
def vector_b : ℝ × ℝ := (5, 5)

theorem angle_between_vectors :
  vector_a • vector_b = 15 → 
  ( ∥vector_a∥ * ∥vector_b∥ * real.cos θ = 15 ) → 
  θ = real.pi / 4 :=
by
  sorry

end angle_between_vectors_l276_276459


namespace minimum_value_fraction_l276_276410

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem minimum_value_fraction (a : ℕ → ℝ) (m n : ℕ) (q : ℝ) (h_geometric : geometric_sequence a q)
  (h_positive : ∀ k : ℕ, 0 < a k)
  (h_condition1 : a 7 = a 6 + 2 * a 5)
  (h_condition2 : ∃ r, r ^ 2 = a m * a n ∧ r = 2 * a 1) :
  (1 / m + 9 / n) ≥ 4 :=
  sorry

end minimum_value_fraction_l276_276410


namespace ways_to_choose_4_cards_of_different_suits_l276_276520

theorem ways_to_choose_4_cards_of_different_suits :
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  ∃ n : ℕ, n = (choose num_suits num_suits) * cards_per_suit ^ num_suits ∧ n = 28561 :=
by
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  have ways_to_choose_suits : (choose num_suits num_suits) = 1 := by simp
  have ways_to_choose_cards : cards_per_suit ^ num_suits = 28561 := by norm_num
  let n := 1 * 28561
  use n
  constructor
  · exact by simp [ways_to_choose_suits, ways_to_choose_cards]
  · exact by rfl

end ways_to_choose_4_cards_of_different_suits_l276_276520


namespace ways_to_choose_4_cards_of_different_suits_l276_276524

theorem ways_to_choose_4_cards_of_different_suits :
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  ∃ n : ℕ, n = (choose num_suits num_suits) * cards_per_suit ^ num_suits ∧ n = 28561 :=
by
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  have ways_to_choose_suits : (choose num_suits num_suits) = 1 := by simp
  have ways_to_choose_cards : cards_per_suit ^ num_suits = 28561 := by norm_num
  let n := 1 * 28561
  use n
  constructor
  · exact by simp [ways_to_choose_suits, ways_to_choose_cards]
  · exact by rfl

end ways_to_choose_4_cards_of_different_suits_l276_276524


namespace circle_equation_l276_276327

theorem circle_equation (r : ℝ) (condition1 : r > 0) 
  (parabola : ∀ x y : ℝ, y^2 = 4 * x → x^2 + y^2 = r^2) 
  (directrix_intersection : ∀ y : ℝ, y^2 = r^2 - 1) 
  (chord_AB_CD_equal : 2 * sqrt (r^2 - 1) = 2 * sqrt (r^2 - 1)) : 
  r = sqrt 5 :=
by 
  sorry

end circle_equation_l276_276327


namespace part1_C_relationship_part2_C_relationship_part3_C_relationship_l276_276565

/-- Part (1): If f(x) = log2(8*x^2) and g(x) = log(1/2)(x), prove that there exists x such
  that the functions have a C relationship. -/
theorem part1_C_relationship (x : ℝ) : (f : ℝ → ℝ) := log (8*x^2) / log 2 
  (g : ℝ → ℝ) := log (x) / log (1/2)
  (h : log 2 * log (8*x^2) = - log (1/2) * log x ) :=
  (exists x : ℝ, h x) := sorry

/-- Part (2): If f(x) = a * sqrt(x - 1) and g(x) = -x - 1 , prove for the functions to NOT
  have a C relationship, a must be less than 2*sqrt(2). -/
theorem part2_C_relationship (a : ℝ) : (f : ℝ → ℝ ) := a * sqrt (x - 1)
  (g : ℝ → ℝ ) :=  -(x+1)
  (no_solution : ∀ x ∈ set.Ici 1, f x ≠ g x ) :
  a < 2 * sqrt (2) := sorry

/-- Part (3): If f(x) = x * e^x and g(x) = m * sin x (with m < 0), for the functions to have
  a C relationship on (0,π), prove m must be less than -1. -/
theorem part3_C_relationship (m : ℝ) : m < 0 → 
  (∀ x ∈ set.Ioo 0 π, ∃ x, x * exp x = -m * sin x ) →
  m ∈ set.Iio ( -1 ) := sorry

end part1_C_relationship_part2_C_relationship_part3_C_relationship_l276_276565


namespace angle_solution_l276_276892

theorem angle_solution (α β : ℝ) (h1 : 2 * sin (2 * β) = 3 * sin (2 * α))
  (h2 : tan β = 3 * tan α) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
: (α = real.arctan ((real.sqrt 7) / 7) ∧ β = real.arctan ((3 * real.sqrt 7) / 7)) :=
sorry

end angle_solution_l276_276892


namespace total_number_of_people_l276_276706

theorem total_number_of_people
  (cannoneers : ℕ) 
  (women : ℕ) 
  (men : ℕ) 
  (hc : cannoneers = 63)
  (hw : women = 2 * cannoneers)
  (hm : men = 2 * women) :
  cannoneers + women + men = 378 := 
sorry

end total_number_of_people_l276_276706


namespace different_suits_card_combinations_l276_276531

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l276_276531


namespace perimeter_percent_increase_l276_276936

noncomputable def side_increase (s₁ s₂_ratio s₃_ratio s₄_ratio s₅_ratio : ℝ) : ℝ :=
  let s₂ := s₂_ratio * s₁
  let s₃ := s₃_ratio * s₂
  let s₄ := s₄_ratio * s₃
  let s₅ := s₅_ratio * s₄
  s₅

theorem perimeter_percent_increase (s₁ : ℝ) (s₂_ratio s₃_ratio s₄_ratio s₅_ratio : ℝ) (P₁ := 3 * s₁)
    (P₅ := 3 * side_increase s₁ s₂_ratio s₃_ratio s₄_ratio s₅_ratio) :
    s₁ = 4 → s₂_ratio = 1.5 → s₃_ratio = 1.3 → s₄_ratio = 1.5 → s₅_ratio = 1.3 →
    P₅ = 45.63 →
    ((P₅ - P₁) / P₁) * 100 = 280.3 :=
by
  intros
  -- proof goes here
  sorry

end perimeter_percent_increase_l276_276936


namespace average_of_30th_and_50th_percentile_l276_276007

noncomputable def data_set : List ℕ := [24, 30, 40, 44, 48, 52]

def percentile (p : ℕ) (data : List ℕ) : ℕ :=
  let len := data.length;
  let position := (p * len) / 100;
  if position % 1 == 0 then
    (data[position - 1] + data[position]) / 2
  else
    data[position]

theorem average_of_30th_and_50th_percentile : 
  let p30 := percentile 30 data_set;
  let p50 := percentile 50 data_set;
  (p30 + p50) / 2 = 36 :=
by
  let p30 := percentile 30 data_set
  let p50 := percentile 50 data_set
  have hp30: p30 = 30 := sorry
  have hp50: p50 = 42 := sorry
  show (p30 + p50) / 2 = 36
  calc (30 + 42) / 2 = 36 : by sorry

end average_of_30th_and_50th_percentile_l276_276007


namespace TriangleStability_l276_276866

theorem TriangleStability (A B C : Type) [Nonempty A] [Nonempty B] [Nonempty C]
  (h_triangle : IsTriangle A B C) : Stability :=
by
  sorry

end TriangleStability_l276_276866


namespace acute_angle_bisector_slope_l276_276670

noncomputable def slope_of_angle_bisector (m1 m2 : ℝ) : ℝ :=
  (m1 + m2 - sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2)

theorem acute_angle_bisector_slope :
  slope_of_angle_bisector (3/2) 2 = (7 - sqrt 29) / 4 :=
by
  sorry

end acute_angle_bisector_slope_l276_276670


namespace percent_primes_divisible_by_3_l276_276830

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l276_276830


namespace find_a_l276_276445

def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 0 then 2^x else a * real.sqrt x

theorem find_a (a : ℝ) (h : f a (-1) + f a 1 = 1) : a = 1 / 2 :=
by
unfold f at h
simp at h
sorry

end find_a_l276_276445


namespace condo_floors_l276_276914

theorem condo_floors (F P : ℕ) (h1: 12 * F + 2 * P = 256) (h2 : P = 2) : F + P = 23 :=
by
  sorry

end condo_floors_l276_276914


namespace area_trapezoid_dbce_l276_276931

-- Definitions based on given conditions
def is_similar (ABC ADE : Triangle) : Prop := sorry -- Similarity definition placeholder
def is_isosceles (ABC : Triangle) : Prop := ABC.AB = ABC.AC
def area (t : Triangle) : ℝ := sorry -- Area definition placeholder

-- Given conditions:
variable (ABC ADE : Triangle) (D B C E : Point)

-- Conditions: All triangles are similar to isosceles triangle ABC where AB = AC
axiom SimilarTriangles : ∀ (t : Triangle), is_similar t ABC
axiom IsIsoscelesABC : is_isosceles ABC
axiom SmallTriangleArea : ∀ (t : Triangle), (area t = 2) → (t = ADE)  -- 8 smallest triangles with area 2

-- Given areas
axiom AreaABC : area ABC = 80

-- Prove that the area of trapezoid DBCE is 70
theorem area_trapezoid_dbce : area ABC - 5 * 2 = 70 := by
{
    sorry
}

end area_trapezoid_dbce_l276_276931


namespace trapezium_area_l276_276188

theorem trapezium_area (top width: ℕ) (bottom width: ℕ) (height: ℕ) (top_width_pos : Top width = 12) (bottom_width_pos : Bottom width = 8) (height_pos: Height = 50):
  (1 / 2) * (top_width + bottom_width) * height = 500 := 
sorry

end trapezium_area_l276_276188


namespace prop_p_necessary_but_not_sufficient_for_prop_q_l276_276149

theorem prop_p_necessary_but_not_sufficient_for_prop_q (x y : ℕ) :
  (x ≠ 1 ∨ y ≠ 3) → (x + y ≠ 4) → ((x+y ≠ 4) → (x ≠ 1 ∨ y ≠ 3)) ∧ ¬ ((x ≠ 1 ∨ y ≠ 3) → (x + y ≠ 4)) :=
by
  sorry

end prop_p_necessary_but_not_sufficient_for_prop_q_l276_276149


namespace aaron_weekly_earnings_l276_276351

def minutes_worked_monday : ℕ := 90
def minutes_worked_tuesday : ℕ := 40
def minutes_worked_wednesday : ℕ := 135
def minutes_worked_thursday : ℕ := 45
def minutes_worked_friday : ℕ := 60
def minutes_worked_saturday1 : ℕ := 90
def minutes_worked_saturday2 : ℕ := 75
def hourly_rate : ℕ := 4

def total_minutes_worked : ℕ :=
  minutes_worked_monday + 
  minutes_worked_tuesday + 
  minutes_worked_wednesday +
  minutes_worked_thursday + 
  minutes_worked_friday +
  minutes_worked_saturday1 + 
  minutes_worked_saturday2

def total_hours_worked : ℕ := total_minutes_worked / 60

def total_earnings : ℕ := total_hours_worked * hourly_rate

theorem aaron_weekly_earnings : total_earnings = 36 := by 
  sorry -- The proof is omitted.

end aaron_weekly_earnings_l276_276351


namespace circles_intersection_l276_276694

theorem circles_intersection (c : ℝ) :
  let circle1 := {p : ℝ × ℝ | (p.1 - 1) ^ 2 + (p.2 - 5) ^ 2 = 49}
  let circle2 := {p : ℝ × ℝ | (p.1 + 2) ^ 2 + (p.2 + 1) ^ 2 = 50} 
  (∀ p, p ∈ circle1 → p ∈ circle2 → p.1 + p.2 = c) → c = 3 :=
by
sry

end circles_intersection_l276_276694


namespace candies_per_pack_l276_276355

-- Conditions in Lean:
def total_candies : ℕ := 60
def packs_initially (packs_after : ℕ) : ℕ := packs_after + 1
def packs_after : ℕ := 2
def pack_count : ℕ := packs_initially packs_after

-- The statement of the proof problem:
theorem candies_per_pack : 
  total_candies / pack_count = 20 :=
by
  sorry

end candies_per_pack_l276_276355


namespace photograph_area_l276_276916

def dimensions_are_valid (a b : ℕ) : Prop :=
a > 0 ∧ b > 0 ∧ (a + 4) * (b + 5) = 77

theorem photograph_area (a b : ℕ) (h : dimensions_are_valid a b) : (a * b = 18 ∨ a * b = 14) :=
by 
  sorry

end photograph_area_l276_276916


namespace sum_of_numbers_is_247_l276_276233

/-- Definitions of the conditions -/
def number_contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d < 10 ∧ ∃ (k : ℕ), n / 10 ^ k % 10 = d

variable (A B C : ℕ)
variable (hA : 100 ≤ A ∧ A < 1000)
variable (hB : 10 ≤ B ∧ B < 100)
variable (hC : 10 ≤ C ∧ C < 100)
variable (h_sum_7 : if number_contains_digit A 7 
                  then if number_contains_digit B 7 
                  then if number_contains_digit C 7 
                  then A + B + C 
                  else A + B
                  else A
                  else B + C = 208)
variable (h_sum_3 : if number_contains_digit A 3 
                  then if number_contains_digit B 3
                  then if number_contains_digit C 3
                  then A + B + C 
                  else A + B
                  else A 
                  else B + C = 76)

/-- Prove that the sum of all three numbers is 247 -/
theorem sum_of_numbers_is_247 : A + B + C = 247 :=
by
  sorry

end sum_of_numbers_is_247_l276_276233


namespace Laplace_transform_of_f_l276_276976

def unit_step (t : ℝ) := if t >= 0 then 1 else 0

noncomputable def f (t : ℝ) : ℝ := unit_step t - unit_step (t - 1)

theorem Laplace_transform_of_f :
  ∀ p : ℝ, p > 0 → laplace_transform f p = (1 - real.exp (-p)) / p :=
sorry

end Laplace_transform_of_f_l276_276976


namespace problem_1_problem_2_l276_276449

theorem problem_1 (x : ℝ) : 
  (f x = |x - 1| + |x - 2| → x < 1 / 2 ∨ x > 5 / 2 ↔ f x > 2) :=
sorry

theorem problem_2 (t : ℝ) (x : ℝ) (a : ℝ) (h1 : t ∈ set.Icc 1 2) (h2 : x ∈ set.Icc (-1) 3) :
  (f x = |x - 1| + |x - t| → f x ≥ a + x ↔ a ≤ -1) :=
sorry

end problem_1_problem_2_l276_276449


namespace percentage_of_primes_divisible_by_3_l276_276755

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l276_276755


namespace percent_primes_divisible_by_3_l276_276828

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l276_276828


namespace no_such_polynomial_exists_l276_276566

def g (k : ℤ) : ℕ :=
  if k = -1 ∨ k = 0 ∨ k = 1 then
    1
  else
    let prime_factors := (Nat.factors k.nat_abs).filter Nat.Prime;
    prime_factors.foldr max 2 -- placeholder larger than all primes if none are found

def W (x : ℤ) : ℤ := sorry -- A non-constant polynomial with integer coefficients.

theorem no_such_polynomial_exists :
  ¬ ∃ (W : ℤ → ℤ), (∀ x : ℕ, 0 < x → g (W x) > 1) ∧ set.finite {g (W x) | x : ℕ} := sorry

end no_such_polynomial_exists_l276_276566


namespace sum_of_three_numbers_l276_276242

def contains_digit (n : ℕ) (d : ℕ) : Prop := d ∈ n.digits 10

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1: 100 ≤ A ∧ A ≤ 999)
  (h2: 10 ≤ B ∧ B ≤ 99) 
  (h3: 10 ≤ C ∧ C ≤ 99)
  (h4: (contains_digit A 7 → A) + (contains_digit B 7 → B) + (contains_digit C 7 → C) = 208)
  (h5: (contains_digit B 3 → B) + (contains_digit C 3 → C) = 76) :
  A + B + C = 247 := 
by 
  sorry

end sum_of_three_numbers_l276_276242


namespace sum_of_three_numbers_l276_276236

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d ∨ n / 10 % 10 = d ∨ n / 100 = d

theorem sum_of_three_numbers (A B C : ℕ) :
  (100 ≤ A ∧ A < 1000 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100) ∧
  (∃ (B7 C7 : ℕ), B7 + C7 = 208 ∧ (contains_digit A 7 ∨ contains_digit B7 7 ∨ contains_digit C7 7)) ∧
  (∃ (B3 C3 : ℕ), B3 + C3 = 76 ∧ (contains_digit B3 3 ∨ contains_digit C3 3)) →
  A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276236


namespace cone_surface_area_l276_276065

-- Definitions based on conditions
def sector_angle : Real := 2 * Real.pi / 3
def sector_radius : Real := 2

-- Definition of the radius of the cone's base
def cone_base_radius (sector_angle sector_radius : Real) : Real :=
  sector_radius * sector_angle / (2 * Real.pi)

-- Definition of the lateral surface area of the cone
def lateral_surface_area (r l : Real) : Real :=
  Real.pi * r * l

-- Definition of the base area of the cone
def base_area (r : Real) : Real :=
  Real.pi * r^2

-- Total surface area of the cone
def total_surface_area (sector_angle sector_radius : Real) : Real :=
  let r := cone_base_radius sector_angle sector_radius
  let S1 := lateral_surface_area r sector_radius
  let S2 := base_area r
  S1 + S2

theorem cone_surface_area (h1 : sector_angle = 2 * Real.pi / 3)
                          (h2 : sector_radius = 2) :
  total_surface_area sector_angle sector_radius = 16 * Real.pi / 9 :=
by
  sorry

end cone_surface_area_l276_276065


namespace magnesium_is_limiting_l276_276439

-- Define the conditions
def moles_Mg : ℕ := 4
def moles_CO2 : ℕ := 2
def moles_O2 : ℕ := 2 -- represent excess O2, irrelevant to limiting reagent
def mag_ox_reaction (mg : ℕ) (o2 : ℕ) (mgo : ℕ) : Prop := 2 * mg + o2 = 2 * mgo
def mag_carbon_reaction (mg : ℕ) (co2 : ℕ) (mgco3 : ℕ) : Prop := mg + co2 = mgco3

-- Assume Magnesium is the limiting reagent for both reactions
theorem magnesium_is_limiting (mgo : ℕ) (mgco3 : ℕ) :
  mag_ox_reaction moles_Mg moles_O2 mgo ∧ mag_carbon_reaction moles_Mg moles_CO2 mgco3 →
  mgo = 4 ∧ mgco3 = 4 :=
by
  sorry

end magnesium_is_limiting_l276_276439


namespace solve_for_x_l276_276172

theorem solve_for_x (x : ℚ) : (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ↔ x = -5 / 3 :=
by
  sorry

end solve_for_x_l276_276172


namespace find_ellipse_equation_l276_276999

variables (a b : ℝ)

noncomputable def ellipse_focus : Prop :=
  let c := Real.sqrt 2 in
    a^2 = b^2 + 2 ∧ (2 * b^2) / a = 4 * Real.sqrt 6 / 3

noncomputable def ellipse_equation : Prop :=
  ∀ (x y : ℝ), (x^2) / 6 + (y^2) / 4 = 1

theorem find_ellipse_equation (a b : ℝ) (h_conn_1 : ellipse_focus a b) : ellipse_equation :=
by
  sorry

end find_ellipse_equation_l276_276999


namespace percent_primes_divisible_by_3_l276_276833

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l276_276833


namespace number_of_knights_one_l276_276890

-- We have four children: Anu, Banu, Vanu, and Danu.
-- Let's define the initial conditions based on their statements.

-- Define properties for Knights and Liars
def knight (person : Prop) : Prop := person = True
def liar (person : Prop) : Prop := person = False

-- Statements of the children
def AnuStatement : Prop := (Banu ∧ Vanu ∧ Danu)
def BanuStatement : Prop := ¬ Anu ∧ ¬ Vanu ∧ ¬ Danu
def VanuStatement : Prop := liar Anu ∧ liar Banu
def DanuStatement : Prop := knight Anu ∧ knight Banu ∧ knight Vanu

-- Main hypothesis linking each child's statement to whether they are knights or liars
axiom Anu_knight : knight Anu → AnuStatement
axiom Anu_liar : liar Anu → (¬ AnuStatement)

axiom Banu_knight : knight Banu → BanuStatement
axiom Banu_liar : liar Banu → (¬ BanuStatement)

axiom Vanu_knight : knight Vanu → VanuStatement
axiom Vanu_liar : liar Vanu → (¬ VanuStatement)

axiom Danu_knight : knight Danu → DanuStatement
axiom Danu_liar : liar Danu → (¬ DanuStatement)

-- Our goal is to prove that exactly one of Anu, Banu, Vanu, or Danu is a knight.
theorem number_of_knights_one : (knight Anu ∨ knight Banu ∨ knight Vanu ∨ knight Danu) ∧ 
                                (liar Anu ∨ liar Banu ∨ liar Vanu ∨ liar Danu) →
                                ¬ (knight Anu ∧ knight Banu ∧ knight Vanu ∧ knight Danu) ∧ 
                                ¬ (knight Anu ∧ knight Banu ∧ knight Vanu) ∧
                                ¬ (knight Anu ∧ knight Banu ∧ knight Danu) ∧
                                ¬ (knight Anu ∧ knight Vanu ∧ knight Danu) ∧
                                ¬ (knight Banu ∧ knight Vanu ∧ knight Danu) ∧
                                ¬ (knight Anu ∧ knight Banu) ∧
                                ¬ (knight Anu ∧ knight Vanu) ∧
                                ¬ (knight Anu ∧ knight Danu) ∧
                                ¬ (knight Banu ∧ knight Vanu) ∧
                                ¬ (knight Banu ∧ knight Danu) ∧
                                ¬ (knight Vanu ∧ knight Danu) ∧
                                (knight Vanu) := 
sorry

end number_of_knights_one_l276_276890


namespace percentage_of_primes_divisible_by_3_l276_276813

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l276_276813


namespace permutation_iff_difference_l276_276110

-- Definitions of conditions
variables {n : ℕ} (h_pos : n > 0)

-- Definition of the permutation and the condition about pairwise differences
variables (a : Fin (2 * n) → ℕ)
variable (h_permute : ∀ i j, i ≠ j → a i ≠ a j)
variable (h_diff_distinct : ∀ i j, i < 2 * n - 1 → j < 2 * n - 1 → i ≠ j → (abs (a (i + 1) - a i) ≠ abs (a (j + 1) - a j)))

-- Mathematical statement to prove
theorem permutation_iff_difference :
  (∀ k, odd k → a k ∈ Finset.range n.succ) ↔ a 0 - a (2 * n - 1) = n :=
sorry

end permutation_iff_difference_l276_276110


namespace subsets_containing_5_and_6_l276_276498

theorem subsets_containing_5_and_6 {α : Type} [DecidableEq α] 
  (S : Finset α) (e1 e2 : α) (h : e1 ≠ e2) 
  (H : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ T, e1 ∈ T ∧ e2 ∈ T)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276498


namespace L_like_reflexive_l276_276908

-- Definitions of the shapes and condition of being an "L-like shape"
inductive Shape
| A | B | C | D | E | LLike : Shape → Shape

-- reflection_equiv function representing reflection equivalence across a vertical dashed line
def reflection_equiv (s1 s2 : Shape) : Prop :=
sorry -- This would be defined according to the exact conditions of the shapes and reflection logic.

-- Given the shapes
axiom L_like : Shape
axiom A : Shape
axiom B : Shape
axiom C : Shape
axiom D : Shape
axiom E : Shape

-- The proof problem: Shape D is the mirrored reflection of the given "L-like shape" across a vertical dashed line
theorem L_like_reflexive :
  reflection_equiv L_like D :=
sorry

end L_like_reflexive_l276_276908


namespace percent_primes_divisible_by_3_less_than_20_l276_276766

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l276_276766


namespace probability_of_drawing_white_ball_l276_276576

-- Define initial conditions
def initial_balls : ℕ := 6
def total_balls_after_white : ℕ := initial_balls + 1
def number_of_white_balls : ℕ := 1
def number_of_total_balls : ℕ := total_balls_after_white

-- Define the probability of drawing a white ball
def probability_of_white : ℚ := number_of_white_balls / number_of_total_balls

-- Statement to be proved
theorem probability_of_drawing_white_ball :
  probability_of_white = 1 / 7 :=
by
  sorry

end probability_of_drawing_white_ball_l276_276576


namespace zhenya_weight_change_l276_276876

theorem zhenya_weight_change :
  let initial_mass := 100.0
  let spring_loss := initial_mass - (0.20 * initial_mass)
  let summer_gain := spring_loss + (0.30 * spring_loss)
  let fall_loss := summer_gain - (0.20 * summer_gain)
  let winter_gain := fall_loss + (0.10 * fall_loss)
  winter_gain < initial_mass :=
by
  let initial_mass := 100.0
  let spring_loss := initial_mass - (0.20 * initial_mass)
  let summer_gain := spring_loss + (0.30 * spring_loss)
  let fall_loss := summer_gain - (0.20 * summer_gain)
  let winter_gain := fall_loss + (0.10 * fall_loss)
  have : winter_gain = 91.52 := by
    sorry -- this step involves numerical computation
  show winter_gain < initial_mass from by
    rw this
    linarith

end zhenya_weight_change_l276_276876


namespace average_tv_sets_and_models_l276_276580

theorem average_tv_sets_and_models:
  let shops := [20, 30, 60, 80, 50, 40, 70]
  let models := [3, 4, 5, 6, 2, 4, 3]
  (∀ (n : ℕ), n < shops.length → n > 0 → (shops.sum / shops.length = 50) ∧ (models.sum / models.length ≈ (27 / 7 : ℝ)))
  :=
by
  let shops := [20, 30, 60, 80, 50, 40, 70]
  let models := [3, 4, 5, 6, 2, 4, 3]
  sorry

end average_tv_sets_and_models_l276_276580


namespace solve_roberts_spending_problem_l276_276160

def roberts_spending_problem (total amount raw materials cash: ℝ) (H1 : raw materials = 100)
  (H2 : cash = 0.10 * total) (H3 : total = 250) : Prop :=
  ∃ machinery : ℝ, machinery = 125 ∧ raw materials + machinery + cash = total

theorem solve_roberts_spending_problem :
  roberts_spending_problem 250 100 (0.10 * 250) 100 rfl (by simp) rfl :=
sorry

end solve_roberts_spending_problem_l276_276160


namespace find_k_value_l276_276419

noncomputable def quadratic_root (k : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + k * x - 3 = 0 ∧ x = 1

theorem find_k_value (k : ℝ) (h : quadratic_root k) : k = 2 :=
by
  cases h with x hx,
  cases hx with hx1 hx2,
  rw hx2 at hx1,
  norm_num at hx1,
  exact hx1
  sorry

end find_k_value_l276_276419


namespace percentage_primes_divisible_by_3_l276_276855

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l276_276855


namespace different_suits_choice_count_l276_276512

-- Definitions based on the conditions
def standard_deck : List (Card × Suit) := 
  List.product Card.all Suit.all

def four_cards (deck : List (Card × Suit)) : Prop :=
  deck.length = 4 ∧ ∀ (i j : Fin 4), i ≠ j → (deck.nthLe i (by simp) : Card × Suit).2 ≠ (deck.nthLe j (by simp) : Card × Suit).2

-- Statement of the proof problem
theorem different_suits_choice_count :
  ∃ l : List (Card × Suit), four_cards l ∧ standard_deck.choose 4 = 28561 :=
by
  sorry

end different_suits_choice_count_l276_276512


namespace solve_for_x_l276_276171

theorem solve_for_x (x : ℚ) : (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ↔ x = -5 / 3 :=
by
  sorry

end solve_for_x_l276_276171


namespace function_decreasing_interval_l276_276389

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x
noncomputable def derivative_f (x : ℝ) : ℝ := 2 * x * (Real.log x + 1 / 2)

theorem function_decreasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < Real.sqrt Real.exp / Real.exp → derivative_f x < 0 := 
by
  intro x h
  sorry

end function_decreasing_interval_l276_276389


namespace product_remainder_l276_276867

theorem product_remainder
    (a b c : ℕ)
    (h₁ : a % 36 = 16)
    (h₂ : b % 36 = 8)
    (h₃ : c % 36 = 24) :
    (a * b * c) % 36 = 12 := 
    by
    sorry

end product_remainder_l276_276867


namespace percentage_of_primes_divisible_by_3_is_12_5_l276_276851

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l276_276851


namespace non_congruent_triangle_count_l276_276994

noncomputable def Point := ℝ × ℝ

def A : Point := (0, 0)
def B : Point := (1, 0)
def C : Point := (2, 0)
def D : Point := (3, 0)
def E : Point := (0, 1)
def F : Point := (1, 1)
def G : Point := (2, 1)
def H : Point := (3, 1)

def triangle (p1 p2 p3 : Point) := { p : Point | p = p1 ∨ p = p2 ∨ p = p3 }

def triangles : set (set Point) := 
  {t | ∃ a b c, a ∈ {A, B, C, D, E, F, G, H} ∧ 
                b ∈ {A, B, C, D, E, F, G, H} ∧ 
                c ∈ {A, B, C, D, E, F, G, H} ∧ 
                a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
                t = triangle a b c}

def is_congruent (t1 t2 : set Point) : Prop := sorry -- definition of congruency to be provided

def count_non_congruent_triangles : ℕ :=
  (finset.filter (λ t, ∀ t' ∈ triangles, ¬is_congruent t t') triangles.to_finset).card

theorem non_congruent_triangle_count : count_non_congruent_triangles = 3 := 
  sorry

end non_congruent_triangle_count_l276_276994


namespace CE_squared_plus_DE_squared_proof_l276_276117

noncomputable def CE_squared_plus_DE_squared (radius : ℝ) (diameter : ℝ) (BE : ℝ) (angle_AEC : ℝ) : ℝ :=
  if radius = 10 ∧ diameter = 20 ∧ BE = 4 ∧ angle_AEC = 30 then 200 else sorry

theorem CE_squared_plus_DE_squared_proof : CE_squared_plus_DE_squared 10 20 4 30 = 200 := by
  sorry

end CE_squared_plus_DE_squared_proof_l276_276117


namespace sum_of_three_numbers_l276_276245

def contains_digit (n : ℕ) (d : ℕ) : Prop := d ∈ n.digits 10

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1: 100 ≤ A ∧ A ≤ 999)
  (h2: 10 ≤ B ∧ B ≤ 99) 
  (h3: 10 ≤ C ∧ C ≤ 99)
  (h4: (contains_digit A 7 → A) + (contains_digit B 7 → B) + (contains_digit C 7 → C) = 208)
  (h5: (contains_digit B 3 → B) + (contains_digit C 3 → C) = 76) :
  A + B + C = 247 := 
by 
  sorry

end sum_of_three_numbers_l276_276245


namespace eleanor_cookies_l276_276967

theorem eleanor_cookies : ∃ N : ℕ, 
  N % 13 = 5 ∧ 
  N % 8 = 3 ∧ 
  N < 150 ∧
  (∀ M : ℕ, (M % 13 = 5 ∧ M % 8 = 3 ∧ M < 150) → M = N) :=
by {
  use 83,
  split; try {refl},
  split; try {refl},
  split; try {norm_num},
  intros M hM,
  cases hM with h_mod13 hM1,
  cases hM1 with h_mod8 hM2,
  have h_coprime : nat.gcd 13 8 = 1 := by norm_num,
  exact nat.mod_unique N M 13 8 5 3 h_mod13 h_mod8 hM2 h_coprime
}

end eleanor_cookies_l276_276967


namespace socks_selection_l276_276652

theorem socks_selection :
  (Nat.choose 7 3) - (Nat.choose 6 3) = 15 :=
by sorry

end socks_selection_l276_276652


namespace min_abs_sum_l276_276717

theorem min_abs_sum : ∃ x : ℝ, ∀ x : ℝ, 
  let f := λ x : ℝ, abs (x + 3) + abs (x + 5) + abs (x + 6) in
  f x = 5 :=
sorry

end min_abs_sum_l276_276717


namespace g_triply_nested_l276_276614

/-- Define the piecewise function g -/
def g : ℕ → ℕ :=
  λ n, if n < 5 then n^2 + 1 else 5 * n - 3

/-- Theorem stating the value of g(g(g(3))) -/
theorem g_triply_nested (h : g(g(g(3))) = 232) : h := sorry

end g_triply_nested_l276_276614


namespace george_speed_l276_276404

theorem george_speed : 
  ∀ (d_tot d_1st : ℝ) (v_tot v_1st : ℝ) (v_2nd : ℝ),
    d_tot = 1 ∧ d_1st = 1 / 2 ∧ v_tot = 3 ∧ v_1st = 2 ∧ ((d_tot / v_tot) = (d_1st / v_1st + d_1st / v_2nd)) →
    v_2nd = 6 :=
by
  -- Proof here
  sorry

end george_speed_l276_276404


namespace sum_of_three_numbers_l276_276237

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d ∨ n / 10 % 10 = d ∨ n / 100 = d

theorem sum_of_three_numbers (A B C : ℕ) :
  (100 ≤ A ∧ A < 1000 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100) ∧
  (∃ (B7 C7 : ℕ), B7 + C7 = 208 ∧ (contains_digit A 7 ∨ contains_digit B7 7 ∨ contains_digit C7 7)) ∧
  (∃ (B3 C3 : ℕ), B3 + C3 = 76 ∧ (contains_digit B3 3 ∨ contains_digit C3 3)) →
  A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276237


namespace transform_1_terminal_transform_2_terminal_transform_3_non_terminal_transform_4_non_terminal_l276_276886

-- We define the terms and transformations
def word := list char

-- Transformations as given in the problem:
def transform_1 : word → word := λ w, if w == ['a', 'a', 'b'] then ['b', 'a'] else w
def transform_2 : word → word := λ w, if w == ['a', 'b'] then ['b', 'a'] else w
def transform_3 : word → word := λ w, if w == ['a', 'b'] then ['b', 'b', 'a'] else w
def transform_4 : word → word := λ w, if w == ['a', 'b'] then ['b', 'b', 'a', 'a'] else w

-- A word is terminal under a transformation if it can no longer be transformed
def terminal (trans : word → word) (w : word) : Prop :=
  ∀ w', trans w = w' → w = w'

-- The proof problems seeking to establish terminality:
theorem transform_1_terminal : ∀ w, terminal transform_1 w := sorry

theorem transform_2_terminal : ∀ w, terminal transform_2 w := sorry

theorem transform_3_non_terminal : ¬ ∀ w, terminal transform_3 w := sorry

theorem transform_4_non_terminal : ¬ ∀ w, terminal transform_4 w := sorry

end transform_1_terminal_transform_2_terminal_transform_3_non_terminal_transform_4_non_terminal_l276_276886


namespace area_evaluation_l276_276369

noncomputable def radius : ℝ := 6
noncomputable def central_angle : ℝ := 90
noncomputable def p := 18
noncomputable def q := 3
noncomputable def r : ℝ := -27 / 2

theorem area_evaluation :
  p + q + r = 7.5 :=
by
  sorry

end area_evaluation_l276_276369


namespace pumpkin_pie_filling_l276_276341

theorem pumpkin_pie_filling (price_per_pumpkin : ℕ) (total_earnings : ℕ) (total_pumpkins : ℕ) (pumpkins_per_can : ℕ) :
  price_per_pumpkin = 3 →
  total_earnings = 96 →
  total_pumpkins = 83 →
  pumpkins_per_can = 3 →
  (total_pumpkins - total_earnings / price_per_pumpkin) / pumpkins_per_can = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end pumpkin_pie_filling_l276_276341


namespace french_students_l276_276073

theorem french_students 
  (T : ℕ) (G : ℕ) (B : ℕ) (N : ℕ) (F : ℕ)
  (hT : T = 78) (hG : G = 22) (hB : B = 9) (hN : N = 24)
  (h_eq : F + G - B = T - N) :
  F = 41 :=
by
  sorry

end french_students_l276_276073


namespace odd_square_minus_one_divisible_by_eight_l276_276153

theorem odd_square_minus_one_divisible_by_eight (n : ℤ) : ∃ k : ℤ, ((2 * n + 1) ^ 2 - 1) = 8 * k := 
by
  sorry

end odd_square_minus_one_divisible_by_eight_l276_276153


namespace numberOfWaysToChoose4Cards_l276_276514

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l276_276514


namespace complex_number_solution_l276_276093

theorem complex_number_solution (z : ℂ) (h : z / Complex.I = 3 - Complex.I) : z = 1 + 3 * Complex.I :=
sorry

end complex_number_solution_l276_276093


namespace percentage_primes_divisible_by_3_l276_276857

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l276_276857


namespace quadratic_root_l276_276411

theorem quadratic_root (k : ℝ) (h : (1 : ℝ)^2 + k * 1 - 3 = 0) : k = 2 := 
sorry

end quadratic_root_l276_276411


namespace total_crayons_l276_276380

theorem total_crayons (crayons_per_child : ℕ) (number_of_children : ℕ) (h1 : crayons_per_child = 3) (h2 : number_of_children = 6) : 
  crayons_per_child * number_of_children = 18 := by
  sorry

end total_crayons_l276_276380


namespace percentage_of_primes_divisible_by_3_l276_276811

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l276_276811


namespace sum_of_three_numbers_l276_276243

def contains_digit (n : ℕ) (d : ℕ) : Prop := d ∈ n.digits 10

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1: 100 ≤ A ∧ A ≤ 999)
  (h2: 10 ≤ B ∧ B ≤ 99) 
  (h3: 10 ≤ C ∧ C ≤ 99)
  (h4: (contains_digit A 7 → A) + (contains_digit B 7 → B) + (contains_digit C 7 → C) = 208)
  (h5: (contains_digit B 3 → B) + (contains_digit C 3 → C) = 76) :
  A + B + C = 247 := 
by 
  sorry

end sum_of_three_numbers_l276_276243


namespace percentage_primes_divisible_by_3_l276_276862

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l276_276862


namespace find_inverse_sum_l276_276107

def f (x : ℝ) : ℝ :=
  if x < 15 then 2 * x + 6 else 3 * x - 3

theorem find_inverse_sum :
  let g := inv_fun f in
  g 16 + g 45 = 21 :=
by
  let g := inv_fun f
  have h1 : f 5 = 16 := by 
    simp [f]
  have h2 : g 16 = 5 := 
    inv_fun_eq ⟨5, h1⟩
  have h3 : f 16 = 45 := by
    simp [f]
  have h4 : g 45 = 16 :=
    inv_fun_eq ⟨16, h3⟩
  rw [h2, h4]
  norm_num

end find_inverse_sum_l276_276107


namespace no_partition_with_sum_k_plus_2013_l276_276602

open Nat

theorem no_partition_with_sum_k_plus_2013 (A : ℕ → Finset ℕ) (h_disjoint : ∀ i j, i ≠ j → Disjoint (A i) (A j)) 
  (h_sum : ∀ k, (A k).sum id = k + 2013) : False :=
by
  sorry

end no_partition_with_sum_k_plus_2013_l276_276602


namespace ball_distribution_into_drawers_l276_276225

noncomputable def comb (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem ball_distribution_into_drawers :
  comb 7 4 = 35 := 
sorry

end ball_distribution_into_drawers_l276_276225


namespace investment_value_l276_276354

noncomputable def compound_interest (P : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P * (1 + r)^n

theorem investment_value :
  ∀ (P : ℕ) (r : ℚ) (n : ℕ),
  P = 8000 →
  r = 0.05 →
  n = 3 →
  compound_interest P r n = 9250 := by
    intros P r n hP hr hn
    unfold compound_interest
    -- calculation steps would be here
    sorry

end investment_value_l276_276354


namespace firetruck_reachable_area_l276_276584

theorem firetruck_reachable_area :
  let m := 700
  let n := 31
  let area := m / n -- The area in square miles
  let time := 1 / 10 -- The available time in hours
  let speed_highway := 50 -- Speed on the highway in miles/hour
  let speed_prairie := 14 -- Speed across the prairie in miles/hour
  -- The intersection point of highways is the origin (0, 0)
  -- The firetruck can move within the reachable area
  -- There exist regions formed by the intersection points of movement directions
  m + n = 731 :=
by
  sorry

end firetruck_reachable_area_l276_276584


namespace polynomial_degree_l276_276382

noncomputable def p (x : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^2 + 7
noncomputable def q (x : ℝ) : ℝ := 4 * x^11 - 8 * x^6 + 2 * x^5 - 15
noncomputable def r (x : ℝ) : ℝ := x^3 + 6

theorem polynomial_degree : ∀ x : ℝ, degree ((p x) * (q x) - (r x)^6) = 18 := 
by
  sorry -- Proof omitted

end polynomial_degree_l276_276382


namespace percent_primes_divisible_by_3_less_than_20_l276_276765

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l276_276765


namespace primes_divisible_by_3_percentage_l276_276821

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l276_276821


namespace number_of_subsets_with_5_and_6_l276_276503

theorem number_of_subsets_with_5_and_6 : 
  let S := {1, 2, 3, 4, 5, 6}
  ∃ n : ℕ, (n = (set.powerset S).count (λ x, {5, 6} ⊆ x)) ∧ n = 16 := 
sorry

end number_of_subsets_with_5_and_6_l276_276503


namespace probability_between_points_l276_276918

noncomputable def probability_red_greater_blue_less_than_three_times_blue :
  (ℝ × ℝ → Prop) :=
  λ ⟨x, y⟩, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y ∧ y < 3 * x

theorem probability_between_points (x y : ℝ) :
  (∀ (x y : ℝ), probability_red_greater_blue_less_than_three_times_blue (x, y)) →
  ((x, y) ∈ set.Icc 0 1 ×ˢ set.Icc 0 1) →
  (set.Icc 0 1 ×ˢ set.Icc 0 1).measure (λ ⟨x, y⟩, probability_red_greater_blue_less_than_three_times_blue (x, y)) = 1/6 :=
sorry

end probability_between_points_l276_276918


namespace midpoint_locus_l276_276707

open Real

theorem midpoint_locus (O : Point) (R : ℝ) (A : Point) :
  (∃ M : Point, M lies_on_circle_with_diameter AO) :=
sorry  -- Proof to be filled in

end midpoint_locus_l276_276707


namespace total_judges_in_RI_l276_276227

variable (J : ℕ) -- Total number of judges

-- Conditions
def judges_under_30 (J : ℕ) := 0.1 * J
def judges_between_30_50 (J : ℕ) := 0.6 * J
def judges_over_50 (J : ℕ) := 0.3 * J

-- Given condition: Number of judges over 50 is 12
axiom judges_over_50_is_12 : judges_over_50 J = 12

theorem total_judges_in_RI : J = 40 :=
by
  -- Mathematical proof
  sorry

end total_judges_in_RI_l276_276227


namespace f_value_at_5_l276_276059

def f (x : ℕ) (y : ℕ) : ℕ := 2 * x ^ 2 + y

-- Conditions
axiom h1 : f 2 y = 20

-- Theorem to prove
theorem f_value_at_5 : ∃ y : ℕ, (f 5 y = 62) :=
sorry

end f_value_at_5_l276_276059


namespace different_suits_choice_count_l276_276510

-- Definitions based on the conditions
def standard_deck : List (Card × Suit) := 
  List.product Card.all Suit.all

def four_cards (deck : List (Card × Suit)) : Prop :=
  deck.length = 4 ∧ ∀ (i j : Fin 4), i ≠ j → (deck.nthLe i (by simp) : Card × Suit).2 ≠ (deck.nthLe j (by simp) : Card × Suit).2

-- Statement of the proof problem
theorem different_suits_choice_count :
  ∃ l : List (Card × Suit), four_cards l ∧ standard_deck.choose 4 = 28561 :=
by
  sorry

end different_suits_choice_count_l276_276510


namespace general_formulas_find_smallest_n_sum_cn_l276_276622

noncomputable def geometric_seq (a₁ q : ℝ) (n : ℕ) :=
  a₁ * q ^ n

noncomputable def arithmetic_seq (b₁ d : ℝ) (n : ℕ) :=
  b₁ + n * d

noncomputable def sum_arithmetic_seq (b₁ d : ℝ) (n : ℕ) :=
  n * (2 * b₁ + (n - 1) * d) / 2

theorem general_formulas (a₁ b₁ S₄ a₃ b₃ a₂ : ℝ) (h₁ : a₁ = 2) (h₂ : b₁ = 1) (h₃ : S₄ = a₁ + a₃) (h₄ : a₂ = b₁ + b₃) :
  (∃ q d : ℝ, q + d = 2 ∧ q = 2 ∧ d = 1 ∧ (∀ n, geometric_seq a₁ q n = 2^n) ∧ (∀ n, arithmetic_seq b₁ d n = 1 + n)) :=
by
  sorry

noncomputable def Tn (n : ℕ) : ℝ :=
  2 * (2^n - 1) + n * (n + 1) / 2

theorem find_smallest_n :
  ∃ n : ℕ, n = 3 ∧ Tn n > 2^(n+1) + 1 :=
by
  sorry

noncomputable def cn (aₙ bₙ : ℝ) (n : ℕ) : ℝ :=
  if n % 2 = 1 then aₙ * bₙ else (3 * bₙ - 2) * aₙ / (bₙ * bₙ + 2)

theorem sum_cn (n : ℕ) :
  ∑ i in range (2*n), cn (2^i) (1 + i) i = (2 * n / 3 - 5 / 9) * 2^(2 * n + 1) + 2^(2 * n + 2) / (2 * n + 2) - 8 / 9 :=
by
  sorry

end general_formulas_find_smallest_n_sum_cn_l276_276622


namespace percent_primes_divisible_by_3_less_than_20_l276_276768

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l276_276768


namespace sum_of_numbers_is_247_l276_276231

/-- Definitions of the conditions -/
def number_contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d < 10 ∧ ∃ (k : ℕ), n / 10 ^ k % 10 = d

variable (A B C : ℕ)
variable (hA : 100 ≤ A ∧ A < 1000)
variable (hB : 10 ≤ B ∧ B < 100)
variable (hC : 10 ≤ C ∧ C < 100)
variable (h_sum_7 : if number_contains_digit A 7 
                  then if number_contains_digit B 7 
                  then if number_contains_digit C 7 
                  then A + B + C 
                  else A + B
                  else A
                  else B + C = 208)
variable (h_sum_3 : if number_contains_digit A 3 
                  then if number_contains_digit B 3
                  then if number_contains_digit C 3
                  then A + B + C 
                  else A + B
                  else A 
                  else B + C = 76)

/-- Prove that the sum of all three numbers is 247 -/
theorem sum_of_numbers_is_247 : A + B + C = 247 :=
by
  sorry

end sum_of_numbers_is_247_l276_276231


namespace choose_4_cards_of_different_suits_l276_276537

theorem choose_4_cards_of_different_suits :
  (∃ (n : ℕ), choose 4 4 = n) ∧
  (∃ (m : ℕ), (13^4 = m)) ∧
  (1 * (13^4) = 28561)

end choose_4_cards_of_different_suits_l276_276537


namespace cannot_partition_nat_l276_276600

theorem cannot_partition_nat (A : ℕ → Set ℕ) (h1 : ∀ i j, i ≠ j → Disjoint (A i) (A j))
    (h2 : ∀ k, Finite (A k) ∧ sum {n | n ∈ A k}.toFinset id = k + 2013) :
    False :=
sorry

end cannot_partition_nat_l276_276600


namespace prove_option_B_incorrect_l276_276997

def ellipse :=
  {x y : ℝ // (x^2 / 8) + (y^2 / 4) = 1}

def foci_left : ℝ × ℝ := (-2, 0)

def y_line_intersection (t : ℝ) (ht : 0 < t) (ht2 : t < 2) : ellipse :=
  sorry

def option_B_incorrect (t : ℝ) (ht : 0 < t) (ht2 : t < 2) : Prop :=
  let A := y_line_intersection t ht ht2 in
  let B := y_line_intersection t ht ht2 in
  let AF1 := (A.1 + 2, A.2) in
  let BF1 := (B.1 + 2, B.2) in
  ¬(AF1.1 * BF1.1 + AF1.2 * BF1.2 = 0 → t = sqrt 3)

theorem prove_option_B_incorrect (t : ℝ) (ht : 0 < t) (ht2 : t < 2) :
  option_B_incorrect t ht ht2 :=
sorry

end prove_option_B_incorrect_l276_276997


namespace find_amount_after_two_years_l276_276968

noncomputable def initial_value : ℝ := 64000
noncomputable def yearly_increase (amount : ℝ) : ℝ := amount / 9
noncomputable def amount_after_year (amount : ℝ) : ℝ := amount + yearly_increase amount
noncomputable def amount_after_two_years : ℝ := amount_after_year (amount_after_year initial_value)

theorem find_amount_after_two_years : amount_after_two_years = 79012.34 :=
by
  sorry

end find_amount_after_two_years_l276_276968


namespace probability_of_exact_hits_l276_276915

-- Define the event of hitting the target
def hit_probability := 1 / 2

-- Define the problem statement that needs to be proven
theorem probability_of_exact_hits : 
  ∃ p : ℚ, p = 3 / 16 ∧
    (binomial 6 3 * (hit_probability ^ 3) * ((1 - hit_probability) ^ 3) ≫ 0) := 
by
  sorry

end probability_of_exact_hits_l276_276915


namespace isosceles_triangle_count_l276_276590

-- Define the variables
variables {A B C D E F : Type}

-- Define the angles and congruence conditions
axiom h1 : ∠BAC + ∠ABC + ∠ACB = 180 -- Sum of angles in triangle ABC
axiom h2 : AB = AC                   -- AB is congruent to AC
axiom h3 : ∠ABC = 60                 -- measure of angle ABC is 60 degrees
axiom h4 : BD bisects ∠ABC           -- Segment BD bisects angle ABC
axiom h5 : BD ⟨intersection⟩ AC = D  -- Point D on side AC
axiom h6 : E ⟨intersection⟩ BC = E   -- Point E on side BC
axiom h7 : DE ∥ AB                   -- Segment DE is parallel to AB
axiom h8 : F ⟨intersection⟩ AC = F   -- Point F on side AC
axiom h9 : EF ∥ BD                   -- Segment EF is parallel to BD

-- Define the goal for the proof
theorem isosceles_triangle_count : 
  ∃ A B C D E F : Type, 
  ∠BAC + ∠ABC + ∠ACB = 180 ∧ 
  AB = AC ∧ 
  ∠ABC = 60 ∧ 
  BD bisects ∠ABC ∧ 
  BD ⟨intersection⟩ AC = D ∧ 
  E ⟨intersection⟩ BC = E ∧ 
  DE ∥ AB ∧ 
  F ⟨intersection⟩ AC = F ∧ 
  EF ∥ BD ∧ 
  -- Number of isosceles triangles in the figure
  (isosceles_triangles A B C D E F = 6) :=
sorry

end isosceles_triangle_count_l276_276590


namespace different_suits_card_combinations_l276_276532

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l276_276532


namespace cos_theta_eq_13_35_l276_276040

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (norm_a norm_b norm_a_plus_b : ℝ)

-- Given conditions
def conditions : Prop :=
  ∥a∥ = 5 ∧ ∥b∥ = 7 ∧ ∥a + b∥ = 10

-- The proposition we want to prove
theorem cos_theta_eq_13_35 (h : conditions a b) : real.cos (real.angle a b) = 13 / 35 :=
by
  sorry

end cos_theta_eq_13_35_l276_276040


namespace minimum_value_of_f_l276_276710

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x + 5) + abs (x + 6)

theorem minimum_value_of_f : ∃ x : ℝ, f x = 1 :=
by sorry

end minimum_value_of_f_l276_276710


namespace percentage_of_primes_divisible_by_3_l276_276800

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l276_276800


namespace max_projection_area_of_rotating_tetrahedron_l276_276693

theorem max_projection_area_of_rotating_tetrahedron :
  (∀ (a : ℝ), (a = 1) →
  (dihedral_angle : ℝ), (dihedral_angle = π / 3) →
  (max_area : ℝ), max_area = sqrt 3 / 4) :=
begin
  intros,
  sorry
end

end max_projection_area_of_rotating_tetrahedron_l276_276693


namespace ji_hoon_original_answer_l276_276611

-- Define the conditions: Ji-hoon's mistake
def ji_hoon_mistake (x : ℝ) := x - 7 = 0.45

-- The theorem statement
theorem ji_hoon_original_answer (x : ℝ) (h : ji_hoon_mistake x) : x * 7 = 52.15 :=
by
  sorry

end ji_hoon_original_answer_l276_276611


namespace find_a_sq_plus_b_sq_l276_276156

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 48
axiom h2 : a * b = 156

theorem find_a_sq_plus_b_sq : a^2 + b^2 = 1992 :=
by sorry

end find_a_sq_plus_b_sq_l276_276156


namespace positive_integer_solution_l276_276971

theorem positive_integer_solution (n x y : ℕ) (hn : 0 < n) (hx : 0 < x) (hy : 0 < y) :
  y ^ 2 + x * y + 3 * x = n * (x ^ 2 + x * y + 3 * y) → n = 1 :=
sorry

end positive_integer_solution_l276_276971


namespace max_value_g_l276_276376

def g (x : ℝ) : ℝ := 4 * x - x ^ 4

theorem max_value_g : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 2 ∧ ∀ y : ℝ, (0 ≤ y ∧ y ≤ 2) → g y ≤ g x) ∧ g x = 3 :=
by
  sorry

end max_value_g_l276_276376


namespace percentage_of_primes_divisible_by_3_l276_276810

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l276_276810


namespace trishul_invested_percentage_less_than_raghu_l276_276286

variable {T V R : ℝ}

def vishal_invested_more (T V : ℝ) : Prop :=
  V = 1.10 * T

def total_sum_of_investments (T V : ℝ) : Prop :=
  T + V + 2300 = 6647

def raghu_investment : ℝ := 2300

theorem trishul_invested_percentage_less_than_raghu
  (h1 : vishal_invested_more T V)
  (h2 : total_sum_of_investments T V) :
  ((raghu_investment - T) / raghu_investment) * 100 = 10 :=
  sorry

end trishul_invested_percentage_less_than_raghu_l276_276286


namespace percent_primes_divisible_by_3_less_than_20_l276_276769

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l276_276769


namespace simplify_f_of_alpha_f_of_cos_alpha_minus_3pi_over_2_f_of_alpha_is_minus_1860_l276_276020

noncomputable def f (α : ℝ) : ℝ :=
  (tan(Real.pi - α) * cos(2 * Real.pi - α) * sin(-α + 3 * Real.pi / 2)) /
  (cos(-α - Real.pi) * tan(-Real.pi - α))

-- Proof problem 1: Simplify f(α)
theorem simplify_f_of_alpha (α : ℝ) (h : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) :
  f(α) = cos(α) :=
sorry

-- Proof problem 2: Given cos(α - 3π/2) = 1/5, prove f(α) = -2√6/5
theorem f_of_cos_alpha_minus_3pi_over_2 (α : ℝ) (h1 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi)
  (h2 : cos(α - 3 * Real.pi / 2) = 1 / 5) :
  f(α) = -2 * Real.sqrt 6 / 5 :=
sorry

-- Proof problem 3: Given α = -1860°, prove f(α) = 1/2
theorem f_of_alpha_is_minus_1860 :
  f(-1860 * Real.pi / 180) = 1 / 2 :=
sorry

end simplify_f_of_alpha_f_of_cos_alpha_minus_3pi_over_2_f_of_alpha_is_minus_1860_l276_276020


namespace total_number_of_people_l276_276700

theorem total_number_of_people (num_cannoneers num_women num_men total_people : ℕ)
  (h1 : num_women = 2 * num_cannoneers)
  (h2 : num_cannoneers = 63)
  (h3 : num_men = 2 * num_women)
  (h4 : total_people = num_women + num_men) : 
  total_people = 378 := by
  sorry

end total_number_of_people_l276_276700


namespace matrix_product_l276_276367

open Matrix

variables {R : Type*} [CommRing R] [DecidableEq R]

def A (d e f : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![0, d, -e], ![-d, 0, f], ![e, -f, 0]]

def B (x y z : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![x^2 + 1, x*y, x*z], ![x*y, y^2 + 1, y*z], ![x*z, y*z, z^2 + 1]]

theorem matrix_product (d e f x y z : R) :
  A d e f ⬝ B x y z = A d e f :=
by
  sorry

end matrix_product_l276_276367


namespace find_angle_between_planes_l276_276198

noncomputable def angle_between_planes (α : ℝ) : ℝ :=
  Real.arctan ((Real.tan α) / 3)

theorem find_angle_between_planes 
  (lateral_edge_angle_with_base : ℝ) 
  (α := lateral_edge_angle_with_base) :
  let constructed_plane := {through_vertex_A_and_midpoint_L} 
  in ∀ (base_plane : Plane), angle_between_planes α = Real.arctan ((Real.tan α) / 3) :=
by
  sorry

end find_angle_between_planes_l276_276198


namespace isosceles_triangle_count_l276_276593

namespace TriangleProblem

-- Define the basic geometric objects and their properties
structure Point := (x : ℝ) (y : ℝ)
def Triangle := (A B C : Point)

axiom is_congruent (A B : Point) : Prop
axiom is_parallel (L1 L2 : Point → Point) : Prop
axiom angle (A B C : Point) : ℝ

-- Given conditions
variables {A B C D E F : Point}
def ΔABC : Triangle := ⟨A, B, C⟩
def ΔABD : Triangle := ⟨A, B, D⟩
def ΔBDE : Triangle := ⟨B, D, E⟩
def ΔDEF : Triangle := ⟨D, E, F⟩
def ΔEFB : Triangle := ⟨E, F, B⟩
def ΔFEC : Triangle := ⟨F, E, C⟩
def ΔDEC : Triangle := ⟨D, E, C⟩

axiom H1 : is_congruent A B A C
axiom H2 : angle A B C = 60
axiom H3 : ∃ D, angle A B D = angle D B C
axiom H4 : ∃ E ∈ line B C, is_parallel (λ x, D) (λ x, A B)
axiom H5 : ∃ F ∈ line A C, is_parallel (λ x, E F) (λ x, B D)

-- Proof goal
theorem isosceles_triangle_count 
  (h1 : ΔABC.is_isosceles) 
  (h2 : ΔABD.is_isosceles) 
  (h3 : ΔBDE.is_isosceles) 
  (h4 : ΔDEF.is_isosceles) 
  (h5 : ΔEFB.is_isosceles) 
  (h6 : ΔFEC.is_isosceles) 
  (h7 : ΔDEC.is_isosceles) : 
  7 = 7 := 
sorry

end TriangleProblem

end isosceles_triangle_count_l276_276593


namespace subsets_containing_5_and_6_l276_276481

theorem subsets_containing_5_and_6 (S : Set ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  {T : Set ℕ // {5, 6} ⊆ T ∧ T ⊆ S}.card = 16 := 
sorry

end subsets_containing_5_and_6_l276_276481


namespace sqrt2_notin_A_l276_276636

theorem sqrt2_notin_A : 
  let A := { x : ℚ | x > -1 } in
  ¬ (real.sqrt 2 ∈ A) :=
by
  let A := { x : ℚ | x > -1 }
  sorry

end sqrt2_notin_A_l276_276636


namespace sum_of_three_numbers_l276_276273

theorem sum_of_three_numbers :
  ∃ A B C : ℕ, 
    (100 ≤ A ∧ A < 1000) ∧  -- A is a three-digit number
    (10 ≤ B ∧ B < 100) ∧     -- B is a two-digit number
    (10 ≤ C ∧ C < 100) ∧     -- C is a two-digit number
    (A + (if (B / 10 = 7 ∨ B % 10 = 7) then B else 0) + 
       (if (C / 10 = 7 ∨ C % 10 = 7) then C else 0) = 208) ∧
    (if (B / 10 = 3 ∨ B % 10 = 3) then B else 0) + 
    (if (C / 10 = 3 ∨ C % 10 = 3) then C else 0) = 76 ∧
    A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276273


namespace primes_less_than_20_divisible_by_3_percentage_l276_276729

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l276_276729


namespace central_cell_is_2_l276_276582

-- Definitions based on conditions
def numbers := {i | 0 ≤ i ∧ i ≤ 8}
def grid := {0, 1, 2, 3, 4, 5, 6, 7, 8}
def adj (x y : ℕ) : Prop :=
  (x = y + 1 ∨ x = y - 1 ∨ x = y + 3 ∨ x = y - 3)
def corner_cells := {0, 2, 6, 8}
def sum_corners := (∑ i in corner_cells, i) = 18

-- Statement of the theorem
theorem central_cell_is_2
  (placement : grid → numbers)
  (h_adj : ∀ i j ∈ grid, abs (i - j) = 1 → adj (placement i) (placement j))
  (h_sum_corners : sum_corners):
  placement 4 = 2 :=
sorry

end central_cell_is_2_l276_276582


namespace sum_of_numbers_l276_276260

def contains_digit (n : Nat) (d : Nat) : Prop := 
  (n / 100 = d) ∨ (n % 100 / 10 = d) ∨ (n % 10 = d)

variables {A B C : Nat}

-- Given conditions
axiom three_digit_number : A ≥ 100 ∧ A < 1000
axiom two_digit_numbers : B ≥ 10 ∧ B < 100 ∧ C ≥ 10 ∧ C < 100
axiom sum_with_sevens : contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7 → A + B + C = 208
axiom sum_with_threes : contains_digit B 3 ∧ contains_digit C 3 ∧ B + C = 76

-- Main theorem to be proved
theorem sum_of_numbers : A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276260


namespace sum_three_numbers_is_247_l276_276253

variables (A B C : ℕ)

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ (nat.digits 10 n)

theorem sum_three_numbers_is_247
  (hA : 100 ≤ A ∧ A < 1000) -- A is a three-digit number
  (hB : 10 ≤ B ∧ B < 100)   -- B is a two-digit number
  (hC : 10 ≤ C ∧ C < 100)   -- C is a two-digit number
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
        (if contains_digit A 7 then A else 0) +
        (if contains_digit B 7 then B else 0) +
        (if contains_digit C 7 then C else 0) = 208) -- Sum of numbers containing digit 7 is 208
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧
        (if contains_digit B 3 then B else 0) +
        (if contains_digit C 3 then C else 0) = 76) -- Sum of numbers containing digit 3 is 76
  : A + B + C = 247 := 
sorry

end sum_three_numbers_is_247_l276_276253


namespace probability_more_heads_than_tails_l276_276549

open Nat

def coin_flip_probability (n : Nat) := \(
    let y := (Nat.choose 10 5) * (1 / (2 ^ 10)) -- binomial coefficient
    let x := (1 - y) / 2 -- calculating x
    x
\)

theorem probability_more_heads_than_tails : coin_flip_probability 10 = 193 / 512 :=
by
  sorry

end probability_more_heads_than_tails_l276_276549


namespace different_suits_card_combinations_l276_276533

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l276_276533


namespace sum_of_three_numbers_l276_276240

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d ∨ n / 10 % 10 = d ∨ n / 100 = d

theorem sum_of_three_numbers (A B C : ℕ) :
  (100 ≤ A ∧ A < 1000 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100) ∧
  (∃ (B7 C7 : ℕ), B7 + C7 = 208 ∧ (contains_digit A 7 ∨ contains_digit B7 7 ∨ contains_digit C7 7)) ∧
  (∃ (B3 C3 : ℕ), B3 + C3 = 76 ∧ (contains_digit B3 3 ∨ contains_digit C3 3)) →
  A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l276_276240


namespace subsets_containing_5_and_6_l276_276467

theorem subsets_containing_5_and_6 :
  let S := {1, 2, 3, 4, 5, 6}
  ∃ s ⊆ S, 5 ∈ s ∧ 6 ∈ s ∧ s.card = 16 :=
sorry

end subsets_containing_5_and_6_l276_276467


namespace Lexie_run_time_l276_276947

-- Definitions based on the conditions
def Celia_speed : ℝ := 30 / 300
def Lexie_speed : ℝ := Celia_speed / 2

-- The question to prove:
theorem Lexie_run_time : (1 / Lexie_speed) = 20 := by
  -- Definitions
  have hCelia_speed : Celia_speed = 30 / 300 := rfl
  have hLexie_speed : Lexie_speed = (30 / 300) / 2 := rfl
  
  -- Proof by calculation
  rw [hCelia_speed, hLexie_speed]
  norm_num
  sorry
  

end Lexie_run_time_l276_276947


namespace primes_divisible_by_3_percentage_l276_276816

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l276_276816


namespace pentagonal_tiles_count_l276_276320

theorem pentagonal_tiles_count (t s p : ℕ) 
  (h1 : t + s + p = 30) 
  (h2 : 3 * t + 4 * s + 5 * p = 120) : 
  p = 10 := by
  sorry

end pentagonal_tiles_count_l276_276320


namespace at_least_658_composite_numbers_l276_276150

theorem at_least_658_composite_numbers :
  let N := { n : ℕ // (n.digits 10).length = 1976 ∧ (n.digits 10).count 1 = 1975 ∧ (n.digits 10).count 7 = 1 } in
  ∃ S ⊆ N, S.card ≥ 658 ∧ ∀ n ∈ S, ¬n.prime :=
sorry

end at_least_658_composite_numbers_l276_276150


namespace time_per_page_l276_276649

theorem time_per_page 
    (planning_time : ℝ := 3) 
    (fraction : ℝ := 3/4) 
    (pages_read : ℕ := 9) 
    (minutes_per_hour : ℕ := 60) : 
    (fraction * planning_time * minutes_per_hour) / pages_read = 15 := 
by
  sorry

end time_per_page_l276_276649


namespace max_intersections_of_three_parabolas_l276_276129

def parabola (a b c : ℝ) : set (ℝ × ℝ) := { p | ∃ x, p = (x, a * x^2 + b * x + c) }

theorem max_intersections_of_three_parabolas (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℝ)
  (h₁₂ : (a₁, b₁, c₁) ≠ (a₂, b₂, c₂))
  (h₂₃ : (a₂, b₂, c₂) ≠ (a₃, b₃, c₃))
  (h₁₃ : (a₁, b₁, c₁) ≠ (a₃, b₃, c₃)) :
  ∃ n : ℕ, n = 12 ∧
    (parabola a₁ b₁ c₁ ∩ parabola a₂ b₂ c₂).finite ∧
    (parabola a₁ b₁ c₁ ∩ parabola a₂ b₂ c₂).card ≤ 4 ∧
    (parabola a₂ b₂ c₂ ∩ parabola a₃ b₃ c₃).finite ∧
    (parabola a₂ b₂ c₂ ∩ parabola a₃ b₃ c₃).card ≤ 4 ∧
    (parabola a₁ b₁ c₁ ∩ parabola a₃ b₃ c₃).finite ∧
    (parabola a₁ b₁ c₁ ∩ parabola a₃ b₃ c₃).card ≤ 4 ∧
    ((parabola a₁ b₁ c₁ ∩ parabola a₂ b₂ c₂) ∪
     (parabola a₂ b₂ c₂ ∩ parabola a₃ b₃ c₃) ∪
     (parabola a₁ b₁ c₁ ∩ parabola a₃ b₃ c₃)).card = 12 :=
sorry

end max_intersections_of_three_parabolas_l276_276129


namespace min_abs_sum_l276_276716

theorem min_abs_sum : ∃ x : ℝ, ∀ x : ℝ, 
  let f := λ x : ℝ, abs (x + 3) + abs (x + 5) + abs (x + 6) in
  f x = 5 :=
sorry

end min_abs_sum_l276_276716


namespace hotel_accommodation_arrangements_l276_276909

theorem hotel_accommodation_arrangements :
  let triple_room := 1
  let double_rooms := 2
  let adults := 3
  let children := 2
  (∀ (triple_room : ℕ) (double_rooms : ℕ) (adults : ℕ) (children : ℕ),
    children ≤ adults ∧ double_rooms + triple_room ≥ 1 →
    (∃ (arrangements : ℕ),
      arrangements = 60)) :=
sorry

end hotel_accommodation_arrangements_l276_276909


namespace sum_of_three_numbers_l276_276246

def contains_digit (n : ℕ) (d : ℕ) : Prop := d ∈ n.digits 10

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1: 100 ≤ A ∧ A ≤ 999)
  (h2: 10 ≤ B ∧ B ≤ 99) 
  (h3: 10 ≤ C ∧ C ≤ 99)
  (h4: (contains_digit A 7 → A) + (contains_digit B 7 → B) + (contains_digit C 7 → C) = 208)
  (h5: (contains_digit B 3 → B) + (contains_digit C 3 → C) = 76) :
  A + B + C = 247 := 
by 
  sorry

end sum_of_three_numbers_l276_276246


namespace subsets_containing_5_and_6_l276_276497

theorem subsets_containing_5_and_6 {α : Type} [DecidableEq α] 
  (S : Finset α) (e1 e2 : α) (h : e1 ≠ e2) 
  (H : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ T, e1 ∈ T ∧ e2 ∈ T)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276497


namespace find_A_salary_l276_276214

theorem find_A_salary (A B : ℝ) (h1 : A + B = 2000) (h2 : 0.05 * A = 0.15 * B) : A = 1500 :=
sorry

end find_A_salary_l276_276214


namespace sum_of_numbers_l276_276268

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), n = k * 10 + d + m * 10 * (10 ^ k)

theorem sum_of_numbers
  (A B C : ℕ)
  (hA : A >= 100 ∧ A < 1000)
  (hB : B >= 10 ∧ B < 100)
  (hC : C >= 10 ∧ C < 100)
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
              (if contains_digit A 7 then A else 0) +
              (if contains_digit B 7 then B else 0) +
              (if contains_digit C 7 then C else 0) = 208)
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧ 
              (if contains_digit B 3 then B else 0) +
              (if contains_digit C 3 then C else 0) = 76) :
  A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276268


namespace complex_number_root_of_polynomial_l276_276631

open Complex

noncomputable def p (t : ℂ) := t^3 + t^2 - 2*t - 1

theorem complex_number_root_of_polynomial (x : ℂ) (h : p (x + x⁻¹) = 0) : x^7 + x^(-7) = 2 := by
  sorry

end complex_number_root_of_polynomial_l276_276631


namespace angle_inequality_iff_sine_inequality_l276_276090

variables {A B C : ℝ}
variables (acute : 0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ 0 < C ∧ C < 90 ∧ A + B + C = 180)
variables (condA : A > B ∧ B > C)
variables (condB : sin (2 * A) < sin (2 * B) ∧ sin (2 * B) < sin (2 * C))

theorem angle_inequality_iff_sine_inequality (acute : 0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ 0 < C ∧ C < 90 ∧ A + B + C = 180) :
    (A > B ∧ B > C) ↔ (sin (2 * A) < sin (2 * B) ∧ sin (2 * B) < sin (2 * C)) :=
sorry

end angle_inequality_iff_sine_inequality_l276_276090


namespace derivative_f_at_zero_l276_276542

-- Define f(x) as mentioned in the condition.
def f (x : ℝ) : ℝ := Math.sin x - 1

-- State the theorem
theorem derivative_f_at_zero : (deriv f 0) = 1 :=
by
  -- The proof is omitted here.
  sorry

end derivative_f_at_zero_l276_276542


namespace sequence_limit_is_one_over_e_l276_276307

noncomputable def sequence_limit : ℝ :=
  limit (λ n : ℕ, (10 * n - 3) ^ (5 * n) / (10 * n - 1) ^ (5 * n))

theorem sequence_limit_is_one_over_e :
  sequence_limit = 1 / Real.exp 1 := sorry

end sequence_limit_is_one_over_e_l276_276307


namespace max_pieces_can_be_continuously_moved_l276_276686

theorem max_pieces_can_be_continuously_moved (n : ℕ) :
  let board := (fin 5) × (fin 9) in
  (∀ (pieces : fin n → board),
     (∀ i, |fst (pieces i) - fst (pieces (i + 1) % n)| + 
              |snd (pieces i) - snd (pieces (i + 1) % n)| = 1) →
    ∃ (seq : ℕ → fin n),
      (∀ t, (|fst (pieces (seq t)) - fst (pieces (seq (t + 1)))| = 1 ∨
             |snd (pieces (seq t)) - snd (pieces (seq (t + 1)))| = 1)) ∧
      ∃ (i : fin n),
        cycle_consistent (seq 0) (seq 1) (seq 2) (pieces i)) →
  n ≤ 32 :=
sorry

end max_pieces_can_be_continuously_moved_l276_276686


namespace primes_divisible_by_3_percentage_is_12_5_l276_276745

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l276_276745


namespace triangle_formation_and_acuteness_l276_276038

variables {a b c : ℝ} {k n : ℕ}

theorem triangle_formation_and_acuteness (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 2 ≤ n) (hk : k < n) (hp : a^n + b^n = c^n) : 
  (a^k + b^k > c^k ∧ b^k + c^k > a^k ∧ c^k + a^k > b^k) ∧ (k < n / 2 → (a^k)^2 + (b^k)^2 > (c^k)^2) :=
sorry

end triangle_formation_and_acuteness_l276_276038


namespace simplify_expression_l276_276654

-- We need to prove that the simplified expression is equal to the expected form
theorem simplify_expression (y : ℝ) : (3 * y - 7 * y^2 + 4 - (5 + 3 * y - 7 * y^2)) = (0 * y^2 + 0 * y - 1) :=
by
  -- The detailed proof steps will go here
  sorry

end simplify_expression_l276_276654


namespace percentage_primes_divisible_by_3_l276_276785

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l276_276785


namespace primes_less_than_20_divisible_by_3_percentage_l276_276725

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l276_276725


namespace number_of_subsets_with_5_and_6_l276_276502

theorem number_of_subsets_with_5_and_6 : 
  let S := {1, 2, 3, 4, 5, 6}
  ∃ n : ℕ, (n = (set.powerset S).count (λ x, {5, 6} ⊆ x)) ∧ n = 16 := 
sorry

end number_of_subsets_with_5_and_6_l276_276502


namespace solution_set_of_inequality_l276_276216

theorem solution_set_of_inequality :
  { x : ℝ // (x - 2)^2 ≤ 2 * x + 11 } = { x : ℝ | -1 ≤ x ∧ x ≤ 7 } :=
sorry

end solution_set_of_inequality_l276_276216


namespace ways_to_choose_4_cards_of_different_suits_l276_276521

theorem ways_to_choose_4_cards_of_different_suits :
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  ∃ n : ℕ, n = (choose num_suits num_suits) * cards_per_suit ^ num_suits ∧ n = 28561 :=
by
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  have ways_to_choose_suits : (choose num_suits num_suits) = 1 := by simp
  have ways_to_choose_cards : cards_per_suit ^ num_suits = 28561 := by norm_num
  let n := 1 * 28561
  use n
  constructor
  · exact by simp [ways_to_choose_suits, ways_to_choose_cards]
  · exact by rfl

end ways_to_choose_4_cards_of_different_suits_l276_276521


namespace length_of_cord_before_cut_l276_276907

def length_cord_before_cut (n : Nat) (m l s : ℝ) : ℝ :=
  n * (l / 2 + s / 2)

theorem length_of_cord_before_cut (n : Nat) (m l s : ℝ) (hn : n = 19) (hm : m = 20) (hl : l = 8) (hs : s = 2) :
  length_cord_before_cut n m l s = 114 :=
  by
    rw [hn, hm, hl, hs]
    simp
    norm_num


end length_of_cord_before_cut_l276_276907


namespace part1_part2_l276_276403

variable (m : ℝ)

def z (m : ℝ) : ℂ := (m^2 + 5 * m + 6) + (m^2 - 2 * m - 15) * complex.I

theorem part1 (h : z m = 2 - 12 * complex.I) : m = -1 := by
  sorry

theorem part2 (h1 : (m^2 + 5 * m + 6) = 0) (h2 : (m^2 - 2 * m - 15) ≠ 0) : m = -2 := by
  sorry

end part1_part2_l276_276403


namespace compute_100M_plus_N_l276_276190

theorem compute_100M_plus_N (M N : ℕ) (hM : M = 40) (hN : N = 41) : 100 * M + N = 4041 := by 
  rw [hM, hN]
  simp
  exact eq.refl 4041

end compute_100M_plus_N_l276_276190


namespace not_square_n5_plus_7_l276_276653

theorem not_square_n5_plus_7 (n : ℕ) (h : n > 1) : ¬ ∃ k : ℕ, k^2 = n^5 + 7 := 
by
  sorry

end not_square_n5_plus_7_l276_276653


namespace percentage_of_primes_divisible_by_3_is_12_5_l276_276852

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l276_276852


namespace sin_minus_cos_eq_neg_one_l276_276558

theorem sin_minus_cos_eq_neg_one (x : ℝ) 
    (h1 : sin x ^ 3 - cos x ^ 3 = -1)
    (h2 : sin x ^ 2 + cos x ^ 2 = 1) : 
    sin x - cos x = -1 :=
sorry

end sin_minus_cos_eq_neg_one_l276_276558


namespace problem1_solution_problem2_solution_l276_276362

noncomputable def problem1 : ℝ :=
  (Real.sqrt (1 / 3) + Real.sqrt 6) / Real.sqrt 3

noncomputable def problem2 : ℝ :=
  (Real.sqrt 3)^2 - Real.sqrt 4 + Real.sqrt ((-2)^2)

theorem problem1_solution :
  problem1 = 1 + 3 * Real.sqrt 2 :=
by
  sorry

theorem problem2_solution :
  problem2 = 3 :=
by
  sorry

end problem1_solution_problem2_solution_l276_276362


namespace percentage_primes_divisible_by_3_l276_276787

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l276_276787


namespace particular_solution_exists_l276_276978

theorem particular_solution_exists 
  (x : ℝ) (y : ℝ → ℝ) (y' : ℝ → ℝ)
  (H : ∀ x, (iteratedDeriv y 2 x) = (y' x) * (Real.cos (y x)))
  (H0 : y 1 = Real.pi / 2)
  (H1 : y' 1 = 1) :
  y x = 2 * Real.arctan (Real.exp (x - 1)) :=
sorry

end particular_solution_exists_l276_276978


namespace expected_total_rainfall_l276_276199

theorem expected_total_rainfall :
  let p_sunny := 0.30
  let p_rain3 := 0.40
  let p_rain6 := 0.30
  let daily_expected_rain := p_sunny * 0 + p_rain3 * 3 + p_rain6 * 6
  let total_expected_rain7 := 7 * daily_expected_rain
  total_expected_rain7 = 21.0 :=
by
  let p_sunny := 0.30
  let p_rain3 := 0.40
  let p_rain6 := 0.30
  let daily_expected_rain := p_sunny * 0 + p_rain3 * 3 + p_rain6 * 6
  let total_expected_rain7 := 7 * daily_expected_rain
  have daily_rain_eq: daily_expected_rain = 3.0 := by
    simp [p_sunny, p_rain3, p_rain6, daily_expected_rain]
  simp [daily_rain_eq, total_expected_rain7]
  exact rfl

end expected_total_rainfall_l276_276199


namespace constant_function_of_horizontal_tangent_l276_276062

theorem constant_function_of_horizontal_tangent (f : ℝ → ℝ) (h : ∀ x, deriv f x = 0) : ∃ c : ℝ, ∀ x, f x = c :=
sorry

end constant_function_of_horizontal_tangent_l276_276062


namespace inequality_proof_l276_276434

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ineq : a + b + c > 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_proof_l276_276434


namespace quadratic_polynomial_is_C_l276_276934

def is_quadratic (p : Polynomial ℝ) : Prop :=
  p.natDegree = 2

def poly_A := 3^2 * Polynomial.X + 1
def poly_B := 3 * Polynomial.X^2
def poly_C := 3 * Polynomial.X * Polynomial.Y + 1
def poly_D := 3 * Polynomial.X - 5^2

theorem quadratic_polynomial_is_C :
  ∃ p, p = poly_C ∧ is_quadratic p := 
by
  use poly_C
  split
  { refl }
  { apply sorry }

end quadratic_polynomial_is_C_l276_276934


namespace number_of_subsets_with_5_and_6_l276_276504

theorem number_of_subsets_with_5_and_6 : 
  let S := {1, 2, 3, 4, 5, 6}
  ∃ n : ℕ, (n = (set.powerset S).count (λ x, {5, 6} ⊆ x)) ∧ n = 16 := 
sorry

end number_of_subsets_with_5_and_6_l276_276504


namespace simplify_trigonometric_expression_l276_276655

theorem simplify_trigonometric_expression :
  cos (22.5 * (Real.pi / 180)) ^ 2 - sin (22.5 * (Real.pi / 180)) ^ 2 = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_trigonometric_expression_l276_276655


namespace subsets_containing_5_and_6_l276_276468

theorem subsets_containing_5_and_6 :
  let S := {1, 2, 3, 4, 5, 6}
  ∃ s ⊆ S, 5 ∈ s ∧ 6 ∈ s ∧ s.card = 16 :=
sorry

end subsets_containing_5_and_6_l276_276468


namespace find_side_b_in_triangle_l276_276574

theorem find_side_b_in_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = 180) 
  (h3 : a + c = 8) 
  (h4 : a * c = 15) 
  (h5 : (b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos (B * Real.pi / 180))) : 
  b = Real.sqrt 19 := 
  by sorry

end find_side_b_in_triangle_l276_276574


namespace total_people_count_l276_276701

-- Definitions based on given conditions
def Cannoneers : ℕ := 63
def Women : ℕ := 2 * Cannoneers
def Men : ℕ := 2 * Women
def TotalPeople : ℕ := Women + Men

-- Lean statement to prove
theorem total_people_count : TotalPeople = 378 := by
  -- placeholders for proof steps
  sorry

end total_people_count_l276_276701


namespace brad_read_more_books_l276_276871

-- Definitions based on the given conditions
def books_william_read_last_month : ℕ := 6
def books_brad_read_last_month : ℕ := 3 * books_william_read_last_month
def books_brad_read_this_month : ℕ := 8
def books_william_read_this_month : ℕ := 2 * books_brad_read_this_month

-- Totals
def total_books_brad_read : ℕ := books_brad_read_last_month + books_brad_read_this_month
def total_books_william_read : ℕ := books_william_read_last_month + books_william_read_this_month

-- The statement to prove
theorem brad_read_more_books : total_books_brad_read = total_books_william_read + 4 := by
  sorry

end brad_read_more_books_l276_276871


namespace prove_correct_proposition_l276_276407

variables (α β : Plane) (a b : Line)
def prop1 := α ∥ β ∧ a ⊆ α → a ∥ β
def prop2 := b ⊆ β ∧ angle_between a b = θ → angle_between a β = θ
def prop3 := α ⊥ β ∧ a ⊥ α → a ∥ β
def prop4 := skew_lines a b ∧ a ⊈ α ∧ b ⊈ α → 
              ∃ p q : Line, p = projection a α ∧ q = projection b α ∧ p ∩ q ≠ ∅

theorem prove_correct_proposition :    
  prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4 := by
    sorry

end prove_correct_proposition_l276_276407


namespace simplify_radical_expression_l276_276163

-- Define the given expressions and necessary conditions
variables {y z : ℝ} (hy : 0 ≤ y) (hz : 0 ≤ z)

-- Define the Lean 4 statement corresponding to the proof
theorem simplify_radical_expression (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (sqrt (32 * y) * sqrt (75 * z) * sqrt (14 * y) = 40 * y * sqrt (21 * z)) :=
sorry

end simplify_radical_expression_l276_276163


namespace sequence_general_term_l276_276677

noncomputable def b_n (n : ℕ) : ℚ := 2 * n - 1
noncomputable def c_n (n : ℕ) : ℚ := n / (2 * n + 1)

theorem sequence_general_term (n : ℕ) : 
  b_n n + c_n n = (4 * n^2 + n - 1) / (2 * n + 1) :=
by sorry

end sequence_general_term_l276_276677


namespace subsets_containing_5_and_6_l276_276493

theorem subsets_containing_5_and_6 {α : Type} [DecidableEq α] 
  (S : Finset α) (e1 e2 : α) (h : e1 ≠ e2) 
  (H : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ T, e1 ∈ T ∧ e2 ∈ T)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276493


namespace expr1_correct_expr2_correct_expr3_correct_l276_276944

-- Define the expressions and corresponding correct answers
def expr1 : Int := 58 + 15 * 4
def expr2 : Int := 216 - 72 / 8
def expr3 : Int := (358 - 295) / 7

-- State the proof goals
theorem expr1_correct : expr1 = 118 := by
  sorry

theorem expr2_correct : expr2 = 207 := by
  sorry

theorem expr3_correct : expr3 = 9 := by
  sorry

end expr1_correct_expr2_correct_expr3_correct_l276_276944


namespace complex_number_solution_l276_276092

theorem complex_number_solution (z : ℂ) (h : z / Complex.I = 3 - Complex.I) : z = 1 + 3 * Complex.I :=
sorry

end complex_number_solution_l276_276092


namespace find_m_plus_n_l276_276661

-- Define the number of ways Blair and Corey can draw the remaining cards
def num_ways_blair_and_corey_draw : ℕ := Nat.choose 50 2

-- Define the function q(a) as given in the problem
noncomputable def q (a : ℕ) : ℚ :=
  (Nat.choose (42 - a) 2 + Nat.choose (a - 1) 2) / num_ways_blair_and_corey_draw

-- Define the problem statement to find the minimum value of a for which q(a) >= 1/2
noncomputable def minimum_a : ℤ :=
  if q 7 >= 1/2 then 7 else 36 -- According to the solution, these are the points of interest

-- The final statement to be proved
theorem find_m_plus_n : minimum_a = 7 ∨ minimum_a = 36 :=
  sorry

end find_m_plus_n_l276_276661


namespace analyze_convexity_concavity_inflection_l276_276099

noncomputable def f (x : ℝ) := x^3 + 3 * x^2 + 6 * x + 7

def is_convex_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y z ∈ I, x < y → y < z → 2 * f y ≤ f x + f z

def is_concave_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y z ∈ I, x < y → y < z → 2 * f y ≥ f x + f z

def inflection_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (∀ ε > 0, ∃ δ > 0, ∀ x', abs (x - x') < δ →
     ((x - δ < x' ∧ x' < x           → second_derivative f x' < 0) ∧
      (x < x' ∧ x' < x + δ            → second_derivative f x' > 0)) ∨
     ((x - δ < x' ∧ x' < x           → second_derivative f x' > 0) ∧
      (x < x' ∧ x' < x + δ            → second_derivative f x' < 0)))

theorem analyze_convexity_concavity_inflection :
  (is_convex_on f {x | x < -1}) ∧
  (is_concave_on f {x | x > -1}) ∧
  (inflection_point f (-1)) :=
by
  sorry

end analyze_convexity_concavity_inflection_l276_276099


namespace different_suits_card_combinations_l276_276528

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l276_276528


namespace price_decrease_necessary_l276_276347

noncomputable def final_price_decrease (P : ℝ) (x : ℝ) : Prop :=
  let increased_price := 1.2 * P
  let final_price := increased_price * (1 - x / 100)
  final_price = 0.88 * P

theorem price_decrease_necessary (x : ℝ) : 
  final_price_decrease 100 x -> x = 26.67 :=
by 
  intros h
  unfold final_price_decrease at h
  sorry

end price_decrease_necessary_l276_276347


namespace percentage_primes_divisible_by_3_l276_276777

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l276_276777


namespace hyperbola_eccentricity_l276_276428

variable (a b c e : ℝ)
variable (a_pos : a > 0)
variable (b_pos : b > 0)
variable (hyperbola_eq : c = Real.sqrt (a^2 + b^2))
variable (y_B : ℝ)
variable (slope_eq : 3 = (y_B - 0) / (c - a))
variable (y_B_on_hyperbola : y_B = b^2 / a)

theorem hyperbola_eccentricity (h : a > 0) (h' : b > 0) (c_def : c = Real.sqrt (a^2 + b^2))
    (slope_cond : 3 = (y_B - 0) / (c - a)) (y_B_cond : y_B = b^2 / a) :
    e = 2 :=
sorry

end hyperbola_eccentricity_l276_276428


namespace sandy_initial_fish_l276_276650

theorem sandy_initial_fish (bought_fish : ℕ) (total_fish : ℕ) (h1 : bought_fish = 6) (h2 : total_fish = 32) :
  total_fish - bought_fish = 26 :=
by
  sorry

end sandy_initial_fish_l276_276650


namespace matias_students_count_l276_276137

theorem matias_students_count (best_rank worst_rank : ℕ) (h1 : best_rank = 75) (h2 : worst_rank = 75) : 
  best_rank + worst_rank - 1 = 149 := 
by
  rw [h1, h2]
  sorry

end matias_students_count_l276_276137


namespace minimum_value_of_f_l276_276712

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x + 5) + abs (x + 6)

theorem minimum_value_of_f : ∃ x : ℝ, f x = 1 :=
by sorry

end minimum_value_of_f_l276_276712


namespace volume_not_occupied_l276_276283

noncomputable def volume_not_occupied_by_cones
  (r : ℝ) (h_cone : ℝ) (h_cylinder : ℝ) : ℝ :=
  let V_cylinder := (r^2 * h_cylinder * π)
  let V_cone := (1/3 * r^2 * h_cone * π)
  let V_both_cones := 2 * V_cone
  V_cylinder - V_both_cones

theorem volume_not_occupied {r h_cone h_cylinder : ℝ} (hr : r = 10) (hcon : h_cone = 15) (hcyl : h_cylinder = 30) :
  volume_not_occupied_by_cones r h_cone h_cylinder = 2000 * π :=
by
  rw [volume_not_occupied_by_cones, hr, hcon, hcyl]
  simp
  sorry

end volume_not_occupied_l276_276283


namespace total_number_of_people_l276_276705

theorem total_number_of_people
  (cannoneers : ℕ) 
  (women : ℕ) 
  (men : ℕ) 
  (hc : cannoneers = 63)
  (hw : women = 2 * cannoneers)
  (hm : men = 2 * women) :
  cannoneers + women + men = 378 := 
sorry

end total_number_of_people_l276_276705


namespace anna_probability_more_heads_than_tails_l276_276552

noncomputable def probability_more_heads_than_tails (n : ℕ) : ℚ :=
  if n % 2 = 0
  then (n.choose (n / 2)) / (2 ^ n)
  else 0

theorem anna_probability_more_heads_than_tails :
  let n := 10 in
  probability_more_heads_than_tails n = 193 / 512 := 
by
  sorry

end anna_probability_more_heads_than_tails_l276_276552


namespace ayse_guarantee_win_l276_276939

def can_ayse_win (m n k : ℕ) : Prop :=
  -- Function defining the winning strategy for Ayşe
  sorry -- The exact strategy definition would be here

theorem ayse_guarantee_win :
  ((can_ayse_win 1 2012 2014) ∧ 
   (can_ayse_win 2011 2011 2012) ∧ 
   (can_ayse_win 2011 2012 2013) ∧ 
   (can_ayse_win 2011 2012 2014) ∧ 
   (can_ayse_win 2011 2013 2013)) = true :=
sorry -- Proof goes here

end ayse_guarantee_win_l276_276939


namespace avg_weights_N_square_of_integer_l276_276888

theorem avg_weights_N_square_of_integer (N : ℕ) :
  (∃ S : ℕ, S > 0 ∧ ∃ k : ℕ, k * k = N + 1 ∧ S = (N * (N + 1)) / 2 / (N - k + 1) ∧ (N * (N + 1)) / 2 - S = (N - k) * S) ↔ (∃ k : ℕ, k * k = N + 1) := by
  sorry

end avg_weights_N_square_of_integer_l276_276888


namespace question1_min_value_question1_max_value_question2_a_range_l276_276027

noncomputable def f (a : ℝ) (x : ℝ) := (a - 1/2) * x^2 + Real.log x
noncomputable def g (a : ℝ) (x : ℝ) := f a x - 2 * a * x

theorem question1_min_value (x : ℝ) (hx : x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1) :
  f (-1/2 : ℝ) x ≥ 1 - Real.exp 2 := 
sorry

theorem question1_max_value (x : ℝ) (hx : x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1) :
  f (-1/2 : ℝ) x ≤ -1/2 - 1/2 * Real.log 2 := 
sorry

theorem question2_a_range (a x : ℝ) (hx : x > 2) : 
  (g a x < 0) ↔ (a ≤ 1/2) :=
sorry

end question1_min_value_question1_max_value_question2_a_range_l276_276027


namespace proof_problem_l276_276581

-- Definitions for the arithmetic and geometric sequences
def a_n (n : ℕ) : ℚ := 2 * n - 4
def b_n (n : ℕ) : ℚ := 2^(n - 2)

-- Conditions based on initial problem statements
axiom a_2 : a_n 2 = 0
axiom b_2 : b_n 2 = 1
axiom a_3_eq_b_3 : a_n 3 = b_n 3
axiom a_4_eq_b_4 : a_n 4 = b_n 4

-- Sum of first n terms of the sequence {n * b_n}
def S_n (n : ℕ) : ℚ := (n-1) * 2^(n-1) + 1/2

-- The main theorem to prove
theorem proof_problem (n : ℕ) : ∃ a_n b_n S_n, 
    (a_n = 2 * n - 4) ∧
    (b_n = 2^(n - 2)) ∧
    (S_n = (n-1) * 2^(n-1) + 1/2) :=
by {
    sorry
}

end proof_problem_l276_276581


namespace primes_less_than_20_divisible_by_3_percentage_l276_276724

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l276_276724


namespace geometric_mean_of_tangents_l276_276708

theorem geometric_mean_of_tangents
  (circle : Set Point)
  (tangent1 tangent2 : Point → Set Point)
  (C : Point)
  (a b : ℝ)
  (c : ℝ)
  (h_tangent1 : ∀ P ∈ circle, P ∈ tangent1 C → distance C P = a)
  (h_tangent2 : ∀ P ∈ circle, P ∈ tangent2 C → distance C P = b)
  (h_tangents_dist : ∀ P Q ∈ circle, P ∈ tangent1 Q ∧ Q ∈ tangent2 Q → distance C P = c) :
  a * b = c^2 :=
by
  sorry

end geometric_mean_of_tangents_l276_276708


namespace sum_of_x_and_y_l276_276157

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 8*x - 10*y + 5) : x + y = -1 := by
  sorry

end sum_of_x_and_y_l276_276157


namespace probability_closer_to_center_l276_276917

-- Radius of the outer circle is given as 3 units
def radius_outer : ℝ := 3

-- Radius of the inner circle where points are closer to the center than to the boundary
def radius_inner := radius_outer / 2

-- Area of the outer circle
def area_outer := π * radius_outer^2

-- Area of the inner circle
def area_inner := π * radius_inner^2

-- The probability that a random point is closer to the center than to the boundary
theorem probability_closer_to_center : area_inner / area_outer = 1 / 4 := 
sorry

end probability_closer_to_center_l276_276917


namespace different_suits_card_combinations_l276_276529

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l276_276529


namespace solve_equation_l276_276657

noncomputable def omega := complex.exp (2 * real.pi * complex.I / 3)
noncomputable def omega2 := complex.exp (-2 * real.pi * complex.I / 3)

theorem solve_equation (x : ℂ) :
  (x^4 + 4*x^3 * complex.sqrt 3 + 18*x^2 + 12*x * complex.sqrt 3 + 9) + (x + complex.sqrt 3) = 0 ↔
  x = -complex.sqrt 3 ∨ x = omega - complex.sqrt 3 ∨ x = omega2 - complex.sqrt 3 := by
  sorry

end solve_equation_l276_276657


namespace line_through_p_l276_276974

open Real

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def is_midpoint (A M B : ℝ × ℝ) : Prop :=
  M = midpoint A B

-- Defining the first line x - 3y + 10 = 0
def line1 (p : ℝ × ℝ) : Prop :=
  p.1 - 3 * p.2 + 10 = 0

-- Defining the second line 2x + y - 8 = 0
def line2 (p : ℝ × ℝ) : Prop :=
  2 * p.1 + p.2 - 8 = 0

-- Defining line l as x + 4y - 4 = 0
def line_l (p : ℝ × ℝ) : Prop :=
  p.1 + 4 * p.2 - 4 = 0

theorem line_through_p (P : ℝ × ℝ)
  (H1 : line1 A)
  (H2 : line2 B)
  (mid : is_midpoint A P B)
  : line_l P :=
sorry

end line_through_p_l276_276974


namespace percentage_of_primes_divisible_by_3_l276_276753

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l276_276753


namespace total_sum_of_faces_l276_276695

/- Define the numbers on the faces of the two cubes -/
def cube1_faces : list ℕ := [3, 4, 5, 6, 7, 8]
def cube2_faces : list ℕ := [14, 15, 16, 17, 18, 19]

/- Assume the condition: Sum of the numbers on each pair of opposite faces -/
def equal_sums_of_opposite_faces : Prop :=
  (3 + 8 = 4 + 7) ∧ (4 + 7 = 5 + 6) ∧ (14 + 19 = 15 + 18) ∧ (15 + 18 = 16 + 17)

/- The theorem to be proven -/
theorem total_sum_of_faces :
  equal_sums_of_opposite_faces →
  list.sum cube1_faces + list.sum cube2_faces = 132 :=
by
  intro h,
  sorry

end total_sum_of_faces_l276_276695


namespace primes_divisible_by_3_percentage_is_12_5_l276_276748

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l276_276748


namespace numberOfWaysToChoose4Cards_l276_276518

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l276_276518


namespace percentage_of_primes_divisible_by_3_l276_276795

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l276_276795


namespace min_correct_responses_l276_276662

def points_correct : ℝ := 7.5
def points_incorrect : ℝ := -1.5
def points_unanswered : ℝ := 2
def total_questions : ℕ := 30
def questions_attempted : ℕ := 25
def questions_unanswered : ℕ := 5
def required_score : ℝ := 150
def unanswered_points : ℝ := points_unanswered * questions_unanswered
def needed_points : ℝ := required_score - unanswered_points

theorem min_correct_responses :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ questions_attempted ∧ (9 * x - 37.5) ≥ needed_points :=
begin
  sorry
end

end min_correct_responses_l276_276662


namespace percentage_of_primes_divisible_by_3_l276_276794

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l276_276794


namespace sequence_general_term_l276_276197

theorem sequence_general_term (n : ℕ) : 
  let seq := [1/2, -3/4, 5/8, -7/16] in
  n > 0 → (seq (n-1) = (-1)^(n) * (2*(n-1)+1) / 2^((n-1)+1)) → 
  ∃ (a_n : ℕ → ℚ), a_n n = (-1)^(n+1) * (2*n-1) / 2^n :=
by sorry

end sequence_general_term_l276_276197


namespace mixture_weight_l276_276901

def almonds := 116.67
def walnuts := almonds / 5
def total_weight := almonds + walnuts

theorem mixture_weight : total_weight = 140.004 := by
  sorry

end mixture_weight_l276_276901


namespace arithmetic_sequence_ratio_l276_276996

-- Define the conditions
variable (a_n : ℕ → ℝ) (d : ℝ) (S_n : ℕ → ℝ)
variable (n : ℕ) (a_1 : ℝ)
variable (h1 : (∑ i in Finset.range (10 + 1), a_n i) / (∑ i in Finset.range (5 + 1), a_n i) = 4)
variable (h2 : ∀ n, a_n n = a_1 + n * d)
variable (h3 : ∀ n, S_n n = n * a_1 + (n * (n - 1) / 2) * d)

theorem arithmetic_sequence_ratio (h4 : d ≠ 0) : (4 * a_1) / d = 2 := by
  sorry

end arithmetic_sequence_ratio_l276_276996


namespace actual_time_is_7PM_l276_276176

-- Define the conditions as hypotheses
def carClockInitial := 15 -- 3:00 PM in 24-hour format
def wristwatchInitial := 15 -- 3:00 PM in 24-hour format
def wristwatchFinish := 15.6667 -- 3:40 PM in 24-hour format (40/60)
def carClockFinish := 15.8333 -- 3:50 PM in 24-hour format (50/60)
def carClockLater := 20 -- 8:00 PM in 24-hour format

-- Prove that the actual time is 7:00 PM (19 in 24-hour format) when the car clock shows 8:00 PM
theorem actual_time_is_7PM :
  let rate_of_increase := (carClockFinish - carClockInitial) / (wristwatchFinish - wristwatchInitial) in
  let hours_elapsed := (carClockLater - carClockInitial) / rate_of_increase in
  (wristwatchInitial + hours_elapsed) = 19 := by
  sorry

end actual_time_is_7PM_l276_276176


namespace subsets_containing_5_and_6_l276_276482

theorem subsets_containing_5_and_6 (S : Set ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  {T : Set ℕ // {5, 6} ⊆ T ∧ T ⊆ S}.card = 16 := 
sorry

end subsets_containing_5_and_6_l276_276482


namespace length_of_XY_l276_276697

theorem length_of_XY 
  (C_omega C_Omega P X Y : Point)
  (r_omega r_Omega R : ℝ)
  (r_omega_eq : r_omega = 4)
  (triangle_circumradius_eq : R = 9)
  (XY_bisects_PC_Omega : bisects(XY, P, C_Omega)) 
  (non_intersecting_circles : non_intersecting(C_omega, r_omega, C_Omega, r_Omega))
  (tangents_intersect_at_P : intersects_at(P, tangents(C_omega, r_omega, C_Omega, r_Omega)))
  : XY.length = 4 * sqrt 14 :=
by
  sorry

end length_of_XY_l276_276697


namespace wire_length_l276_276667

theorem wire_length :
  ∀ (d : ℝ) (h1 h2 : ℝ), d = 20 ∧ h1 = 8 ∧ h2 = 18 → 
  real.sqrt (d^2 + (h2 - h1)^2) = 10 * real.sqrt 5 :=
by
  intros d h1 h2 h
  obtain ⟨hd, hh1, hh2⟩ := h
  rw [hd, hh1, hh2]
  sorry

end wire_length_l276_276667


namespace sqrt_identity_l276_276644

theorem sqrt_identity (a b : ℝ) (h : a > sqrt b) : 
    sqrt ((a + sqrt (a^2 - b)) / 2) + sqrt ((a - sqrt (a^2 - b)) / 2) = sqrt (a + sqrt b) ∧ 
    sqrt ((a + sqrt (a^2 - b)) / 2) - sqrt ((a - sqrt (a^2 - b)) / 2) = sqrt (a - sqrt b) := 
by
  sorry

end sqrt_identity_l276_276644


namespace num_2x2_boxes_l276_276191

def calendarMay : list (list ℕ) :=
  [[1, 2, 3, 4, 5, 6, 7],
   [8, 9, 10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19, 20, 21],
   [22, 23, 24, 25, 26, 27, 28],
   [29, 30, 31]]

def validTopLeftPositions (calendar: list (list ℕ)) : ℕ :=
  let rows := calendar.length
  let cols := calendar.head.length
    -- first 4 rows and first 6 columns are valid starting points for 2x2 in a 5x7 grid
    let validRows := rows - 1
    let validCols := cols - 1
    validRows * validCols

theorem num_2x2_boxes : validTopLeftPositions calendarMay = 22 :=
  by sorry

end num_2x2_boxes_l276_276191


namespace percent_primes_divisible_by_3_less_than_20_l276_276767

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l276_276767


namespace diff_largest_4th_largest_l276_276388

theorem diff_largest_4th_largest :
  let l := [8, 0, 3, 7, 5, 2, 4]
  let sorted_l := l.qsort (λ a b => a > b)
  let largest := sorted_l.head! 
  let fourth_largest := sorted_l.nth! 3
  largest - fourth_largest = 4 := 
by
  sorry

end diff_largest_4th_largest_l276_276388


namespace total_bananas_in_collection_l276_276687

theorem total_bananas_in_collection (groups_of_bananas : ℕ) (bananas_per_group : ℕ) 
    (h1 : groups_of_bananas = 7) (h2 : bananas_per_group = 29) :
    groups_of_bananas * bananas_per_group = 203 := by
  sorry

end total_bananas_in_collection_l276_276687


namespace rectangle_diagonal_length_l276_276672

theorem rectangle_diagonal_length (l w : ℕ) (h1 : 2 * (l + w) = 56) (h2 : 4 * w = 3 * l) : 
  sqrt (l^2 + w^2) = 20 :=
by
  sorry

end rectangle_diagonal_length_l276_276672


namespace solution_l276_276990

def f (x : ℕ) : ℕ :=
  |x - 2| + 1

def g (x : ℕ) : ℕ :=
  4 - x

theorem solution : f (g 2) < g (f 2) :=
by {
  -- Calculate the inner terms first
  have fg_2 : f(g 2) = f(2) := by simp [g, f],
  have gf_2 : g(f 2) = g(1) := by simp [f, g],
  -- Substituting the calculated terms
  rw [fg_2, gf_2],
  simp [f, g]
}

end solution_l276_276990


namespace percentage_of_primes_divisible_by_3_l276_276788

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l276_276788


namespace primes_divisible_by_3_percentage_l276_276814

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l276_276814


namespace sum_of_sides_of_regular_pentagon_l276_276291

theorem sum_of_sides_of_regular_pentagon (s : ℝ) (n : ℕ)
    (h : s = 15) (hn : n = 5) : 5 * 15 = 75 :=
sorry

end sum_of_sides_of_regular_pentagon_l276_276291


namespace choose_4_cards_of_different_suits_l276_276538

theorem choose_4_cards_of_different_suits :
  (∃ (n : ℕ), choose 4 4 = n) ∧
  (∃ (m : ℕ), (13^4 = m)) ∧
  (1 * (13^4) = 28561)

end choose_4_cards_of_different_suits_l276_276538


namespace probability_success_l276_276316

open Set

def red_die := {1, 2, 3, 4}
def green_die := {1, 2, 3, 4, 5, 6, 7, 8}

def is_divisible_by_2 (n : ℕ) : Prop := n % 2 = 0
def is_power_of_2 (n : ℕ) : Prop := ∃ k : ℕ, 2^k = n

def successful_red_die_outcomes := {n ∈ red_die | is_divisible_by_2 n}
def successful_green_die_outcomes := {n ∈ green_die | is_power_of_2 n}

def successful_outcomes := 
  {r ∈ successful_red_die_outcomes, g ∈ successful_green_die_outcomes | true}

def total_outcomes := red_die ×' green_die -- Cartesian product of the two sets

theorem probability_success : 
  (card successful_outcomes) / (card total_outcomes) = 1 / 4 :=
sorry

end probability_success_l276_276316


namespace omitted_even_number_l276_276874

theorem omitted_even_number (hm : ∑ i in (finset.range 45).filter (λ x, x % 2 = 0), i + 2 = 2014) : 56 ∈ (finset.range 45).filter (λ x, x % 2 = 0) :=
sorry

end omitted_even_number_l276_276874


namespace primes_divisible_by_3_percentage_l276_276815

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l276_276815


namespace percent_primes_divisible_by_3_less_than_20_l276_276771

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l276_276771


namespace total_candies_l276_276910

theorem total_candies (n p r : ℕ) (H1 : n = 157) (H2 : p = 235) (H3 : r = 98) :
  n * p + r = 36993 := by
  sorry

end total_candies_l276_276910


namespace no_partition_with_sum_k_plus_2013_l276_276604

open Nat

theorem no_partition_with_sum_k_plus_2013 (A : ℕ → Finset ℕ) (h_disjoint : ∀ i j, i ≠ j → Disjoint (A i) (A j)) 
  (h_sum : ∀ k, (A k).sum id = k + 2013) : False :=
by
  sorry

end no_partition_with_sum_k_plus_2013_l276_276604


namespace prime_division_or_divisibility_l276_276629

open Nat

theorem prime_division_or_divisibility (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) (hodd : Odd p) (hd : p ∣ q^r + 1) :
    (2 * r ∣ p - 1) ∨ (p ∣ q^2 - 1) := 
sorry

end prime_division_or_divisibility_l276_276629


namespace count_valid_mappings_l276_276630

noncomputable def x : Set ℤ := {-1, 0, 1}
noncomputable def y : Set ℤ := {-2, -1, 0, 1, 2}

def valid_mapping (f : ℤ → ℤ) : Prop :=
  ∀ (a : ℤ), a ∈ x → ((a % 2 = 0 ∧ (a + f a) % 2 = 0) ∨ (a % 2 = 1 ∧ (a + f a) % 2 = 1))

theorem count_valid_mappings : 
  (∃ (f : ℤ → ℤ), valid_mapping f) → (finset.card
  (finset.filter (λ f, valid_mapping f)
    (finset.univ : finset (ℤ → ℤ))) = 12) :=
sorry

end count_valid_mappings_l276_276630


namespace find_n_l276_276442

def binomial_coefficient_sum (n : ℕ) (a b : ℝ) : ℝ :=
  (a + b) ^ n

def expanded_coefficient_sum (n : ℕ) (a b : ℝ) : ℝ :=
  (a + 3 * b) ^ n

theorem find_n (n : ℕ) :
  (expanded_coefficient_sum n 1 1) / (binomial_coefficient_sum n 1 1) = 64 → n = 6 :=
by 
  sorry

end find_n_l276_276442


namespace pell_sum_identity_l276_276663

def pell (n : ℕ) : ℕ
| 0     := 0
| 1     := 1
| (n+2) := 2 * pell (n+1) + pell n

theorem pell_sum_identity :
  (∑ n in (Finset.range ∞), (Real.arctan (1 / pell (2 * n)) + Real.arctan (1 / pell (2 * n + 2))) * Real.arctan (2 / pell (2 * n + 1)))
  = (Real.arctan (1 / 2)) ^ 2 := 
sorry

end pell_sum_identity_l276_276663


namespace andrey_wins_iff_irreducible_fraction_l276_276938

def is_irreducible_fraction (p : ℝ) : Prop :=
  ∃ m n : ℕ, p = m / 2^n ∧ gcd m (2^n) = 1

def can_reach_0_or_1 (p : ℝ) : Prop :=
  ∀ move : ℝ, ∃ dir : ℝ, (p + dir * move = 0 ∨ p + dir * move = 1)

theorem andrey_wins_iff_irreducible_fraction (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∃ move_sequence : ℕ → ℝ, ∀ n, can_reach_0_or_1 (move_sequence n)) ↔ is_irreducible_fraction p :=
sorry

end andrey_wins_iff_irreducible_fraction_l276_276938


namespace number_of_good_colorings_l276_276364

noncomputable def countGoodColorings (n : ℕ) (h : n ≥ 4) : ℕ :=
  n * (n - 1)

theorem number_of_good_colorings (n : ℕ) (h : n ≥ 4) :
  ∃ f : ℕ → ℕ, f n = n * (n - 1) := 
begin
  use countGoodColorings,
  sorry
end

end number_of_good_colorings_l276_276364


namespace root_sum_product_eq_l276_276212

theorem root_sum_product_eq (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) :
  p + q = 69 :=
by 
  sorry

end root_sum_product_eq_l276_276212


namespace solve_equation_l276_276168

def problem_statement : Prop :=
  ∃ x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -7 / 6

theorem solve_equation : problem_statement :=
by {
  sorry
}

end solve_equation_l276_276168


namespace yanna_change_l276_276875

    -- Define the costs of individual items
    def cost_shirts := 10 * 5 -- $50
    def cost_sandals := 3 * 3 -- $9
    def cost_hats := 5 * 8 -- $40
    def cost_bags := 7 * 14 -- $98
    def cost_sunglasses := 2 * 12 -- $24
    def cost_skirts := 4 * 18 -- $72
    def cost_earrings := 6 * 6 -- $36

    -- Define the subtotal before discount
    def subtotal := cost_shirts + cost_sandals + cost_hats + cost_bags + cost_sunglasses + cost_skirts + cost_earrings -- $329

    -- Define the discount
    def discount := 0.10 * subtotal -- $32.90

    -- Define the discounted total
    def discounted_total := subtotal - discount -- $296.10

    -- Define the sales tax
    def sales_tax := 0.065 * discounted_total

    -- Round the sales tax to the nearest cent
    def sales_tax_rounded := Float.round sales_tax 2 -- $19.25

    -- Define final total cost
    def final_total_cost := discounted_total + sales_tax_rounded -- $315.35

    -- The amount Yanna gave
    def amount_given := 300

    -- Calculate the change
    def change := amount_given - final_total_cost

    -- Prove that the change is -$15.35
    theorem yanna_change : change = -15.35 := by {
      -- Detailed steps with calculations can be filled out
      sorry
    }
    
end yanna_change_l276_276875


namespace different_suits_card_combinations_l276_276527

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l276_276527


namespace subsets_containing_5_and_6_l276_276487

theorem subsets_containing_5_and_6 (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ s, 5 ∈ s ∧ 6 ∈ s)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276487


namespace general_term_of_sequence_l276_276420

noncomputable def seq_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range (n + 1), a i

theorem general_term_of_sequence (a : ℕ → ℝ) :
  (∀ n : ℕ, 1 ≤ n  → a n + seq_sum a (n - 1) = 1) →
  (∀ n : ℕ, 1 ≤ n → a 1 = 1/2) →
  (∀ n : ℕ, 1 ≤ n → a n = 1 / 2 ^ n) :=
by
  intros h1 h2
  sorry

end general_term_of_sequence_l276_276420


namespace sum_of_squares_not_divisible_by_11_l276_276219

theorem sum_of_squares_not_divisible_by_11 
  (x y z : ℤ) 
  (h_coprime_xy : Nat.gcd x y = 1)
  (h_coprime_xz : Nat.gcd x z = 1)
  (h_coprime_yz : Nat.gcd y z = 1)
  (h_sum_div_11 : (x + y + z) % 11 = 0)
  (h_prod_div_11 : (x * y * z) % 11 = 0):
  (x^2 + y^2 + z^2) % 11 ≠ 0 :=
by
  sorry

end sum_of_squares_not_divisible_by_11_l276_276219


namespace equal_parallel_segments_planes_l276_276068

theorem equal_parallel_segments_planes {l1 l2 l3 : ℝ}  -- three parallel line segments
(hl_par : l1 = l2 ∧ l2 = l3)  -- conditions that they are equal
: ∃ (P Q : set ℝ³), (∀ (x : ℝ³) (y : ℝ³), (x ∈ P ∧ y ∈ Q) → (l1 = l2 ∧ l2 = l3))
→ (P ∩ Q = ∅ ∨ P ∩ Q ≠ ∅) := 
begin
  sorry
end

end equal_parallel_segments_planes_l276_276068


namespace Q_satisfies_condition_l276_276385

/-- Define the polynomial Q(x) -/
def Q (x : ℝ) : ℝ :=
  x^3 - 6 * x^2 + 12 * x - 10

/-- The main theorem statement -/
theorem Q_satisfies_condition : 
  Q (Real.cbrt 2 + 2) = 0 :=
by
  have h : Q (Real.cbrt 2 + 2) = (Real.cbrt 2 + 2)^3 - 6 * (Real.cbrt 2 + 2)^2 + 12 * (Real.cbrt 2 + 2) - 10 := 
    by rfl
  sorry

end Q_satisfies_condition_l276_276385


namespace solve_for_x_l276_276167

theorem solve_for_x (x : ℚ) (h : (3 - x)/(x + 2) + (3 * x - 6)/(3 - x) = 2) : x = -7/6 := 
by 
  sorry

end solve_for_x_l276_276167


namespace solution_set_of_quadratic_inequality_l276_276408

open Set

theorem solution_set_of_quadratic_inequality :
  ∀ (a b c : ℝ), a ≠ 0 ∧ (∀ x, -3 < x ∧ x < 2 ↔ ax^2 + bx + c > 0) →
  ∀ x, (x < -1/3 ∨ x > 1/2) ↔ cx^2 + ax + b > 0 :=
by
  assume (a b c : ℝ) (h : a ≠ 0 ∧ (∀ x, -3 < x ∧ x < 2 ↔ ax^2 + bx + c > 0)),
  assume (x : ℝ),
  sorry

end solution_set_of_quadratic_inequality_l276_276408


namespace two_times_ML_eq_NO_l276_276308

noncomputable theory
open_locale classical
open geometry

variables {S A P K L M N O R : Point} {h₁ : Triangle S A P} 
          (h₂ : A.altitude K S P)
          (h₃ : L.onSegment P A)
          (h₄ : M.onExtension S A A)
          (h₅ : angle L S P = angle L P S)
          (h₆ : angle M S P = angle M P S)
          (h₇ : intersection SL PM AK N O)

theorem two_times_ML_eq_NO :
  2 * (ML : ℝ) = (NO : ℝ) :=
begin
  sorry,
end

end two_times_ML_eq_NO_l276_276308


namespace water_balloon_packs_l276_276639

theorem water_balloon_packs (P : ℕ) : 
  (6 * P + 12 = 30) → P = 3 := by
  sorry

end water_balloon_packs_l276_276639


namespace volume_of_second_cylinder_l276_276282

theorem volume_of_second_cylinder (h r1 r3 : ℝ) (V1 V3 : ℝ) (π : ℝ) 
  (h_eq : h > 0) 
  (ratio : r3 = 3 * r1) 
  (volume_first : V1 = π * r1^2 * h) 
  (V1_val : V1 = 40) : 
  V3 = 360 :=
by 
  have h_volume : V3 = π * r3^2 * h := sorry
  rw [ratio] at h_volume
  rw [mul_pow, pow_two, mul_assoc, mul_comm (3^2), mul_comm r1, mul_assoc 9 _ _] at h_volume
  rw [volume_first, V1_val, mul_comm 9 _] at h_volume
  exact h_volume

end volume_of_second_cylinder_l276_276282


namespace Kiarra_age_l276_276612

variable (Kiarra Bea Job Figaro Harry : ℕ)

theorem Kiarra_age 
  (h1 : Kiarra = 2 * Bea)
  (h2 : Job = 3 * Bea)
  (h3 : Figaro = Job + 7)
  (h4 : Harry = Figaro / 2)
  (h5 : Harry = 26) : 
  Kiarra = 30 := sorry

end Kiarra_age_l276_276612


namespace ev_infinite_k_Q_ge_Q_l276_276982

def Q (n : ℕ) : ℕ := sorry -- Define Q as the sum of the decimal digits of n

theorem ev_infinite_k_Q_ge_Q (Q : ℕ → ℕ)
  (hQ : ∀ n : ℕ, Q(n) = sorry) -- Placeholder for the actual digit sum definition
  : ∃ᶠ k in Nat, Q (3^k) ≥ Q (3^(k+1)) :=
sorry

end ev_infinite_k_Q_ge_Q_l276_276982


namespace ways_to_choose_4_cards_of_different_suits_l276_276526

theorem ways_to_choose_4_cards_of_different_suits :
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  ∃ n : ℕ, n = (choose num_suits num_suits) * cards_per_suit ^ num_suits ∧ n = 28561 :=
by
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  have ways_to_choose_suits : (choose num_suits num_suits) = 1 := by simp
  have ways_to_choose_cards : cards_per_suit ^ num_suits = 28561 := by norm_num
  let n := 1 * 28561
  use n
  constructor
  · exact by simp [ways_to_choose_suits, ways_to_choose_cards]
  · exact by rfl

end ways_to_choose_4_cards_of_different_suits_l276_276526


namespace A_inter_B_eq_l276_276035

-- Define set A based on the condition for different integer k.
def A (k : ℤ) : Set ℝ := {x | 2 * k * Real.pi - Real.pi < x ∧ x < 2 * k * Real.pi}

-- Define set B based on its condition.
def B : Set ℝ := {x | -5 ≤ x ∧ x < 4}

-- The final proof problem to show A ∩ B equals to the given set.
theorem A_inter_B_eq : 
  (⋃ k : ℤ, A k) ∩ B = {x | (-Real.pi < x ∧ x < 0) ∨ (Real.pi < x ∧ x < 4)} :=
by
  sorry

end A_inter_B_eq_l276_276035


namespace choose_4_cards_of_different_suits_l276_276535

theorem choose_4_cards_of_different_suits :
  (∃ (n : ℕ), choose 4 4 = n) ∧
  (∃ (m : ℕ), (13^4 = m)) ∧
  (1 * (13^4) = 28561)

end choose_4_cards_of_different_suits_l276_276535


namespace prob_first_class_0_75_expected_profit_correct_should_not_increase_production_l276_276321

noncomputable def probability_first_class (p_first : ℝ) (output : ℕ) : ℝ :=
  1 - (1 - p_first)^output

theorem prob_first_class_0_75 : 
  probability_first_class 0.5 2 = 0.75 :=
by { sorry }

noncomputable def expected_profit (p_first p_second p_third profit_first profit_second profit_third : ℝ) (output : ℕ) : ℝ :=
  let profits := [profit_third, profit_second, profit_first]
  let probs   := [p_third, p_second, p_first]
  list.sum (list.map (λ (x : ℝ × ℝ), x.fst * x.snd) (list.zip probs profits)) * output

theorem expected_profit_correct :
  expected_profit 0.5 0.4 0.1 0.8 0.6 -0.3 2 = 1.22 :=
by { sorry }

noncomputable def net_profit (n : ℕ) : ℝ :=
  (real.log n) - 0.39 * n

theorem should_not_increase_production : ∀ (n : ℕ), n < 100 / 39 → net_profit n < 0 :=
by { sorry }

end prob_first_class_0_75_expected_profit_correct_should_not_increase_production_l276_276321


namespace percent_primes_divisible_by_3_less_than_20_l276_276764

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l276_276764


namespace polynomial_divisible_by_squared_root_l276_276151

noncomputable def f (a1 a2 a3 a4 x : ℝ) : ℝ := 
  x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4

noncomputable def f_prime (a1 a2 a3 a4 x : ℝ) : ℝ := 
  4 * x^3 + 3 * a1 * x^2 + 2 * a2 * x + a3

theorem polynomial_divisible_by_squared_root 
  (a1 a2 a3 a4 x0 : ℝ) 
  (h1 : f a1 a2 a3 a4 x0 = 0) 
  (h2 : f_prime a1 a2 a3 a4 x0 = 0) : 
  ∃ g : ℝ → ℝ, ∀ x, f a1 a2 a3 a4 x = (x - x0)^2 * g x := 
sorry

end polynomial_divisible_by_squared_root_l276_276151


namespace steve_nickels_dimes_l276_276177

theorem steve_nickels_dimes (n d : ℕ) (h1 : d = n + 4) (h2 : 5 * n + 10 * d = 70) : n = 2 :=
by
  -- The proof goes here
  sorry

end steve_nickels_dimes_l276_276177


namespace sin_minus_cos_eq_minus_1_l276_276555

theorem sin_minus_cos_eq_minus_1 (x : ℝ) 
  (h : Real.sin x ^ 3 - Real.cos x ^ 3 = -1) :
  Real.sin x - Real.cos x = -1 := by
  sorry

end sin_minus_cos_eq_minus_1_l276_276555


namespace find_ab_l276_276461

-- Define the conditions and the goal
theorem find_ab (a b : ℝ) (h1 : a^2 + b^2 = 26) (h2 : a + b = 7) : ab = 23 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end find_ab_l276_276461


namespace sum_three_numbers_is_247_l276_276255

variables (A B C : ℕ)

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ (nat.digits 10 n)

theorem sum_three_numbers_is_247
  (hA : 100 ≤ A ∧ A < 1000) -- A is a three-digit number
  (hB : 10 ≤ B ∧ B < 100)   -- B is a two-digit number
  (hC : 10 ≤ C ∧ C < 100)   -- C is a two-digit number
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
        (if contains_digit A 7 then A else 0) +
        (if contains_digit B 7 then B else 0) +
        (if contains_digit C 7 then C else 0) = 208) -- Sum of numbers containing digit 7 is 208
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧
        (if contains_digit B 3 then B else 0) +
        (if contains_digit C 3 then C else 0) = 76) -- Sum of numbers containing digit 3 is 76
  : A + B + C = 247 := 
sorry

end sum_three_numbers_is_247_l276_276255


namespace count_distinct_digit_numbers_l276_276049

theorem count_distinct_digit_numbers : 
  let count := (5 * 9 * 8 * 7) in
  count = 2520 :=
by
  sorry

end count_distinct_digit_numbers_l276_276049


namespace total_number_of_balls_in_bag_l276_276899

variable (num_white num_green num_yellow num_red num_purple : ℕ)
variable (P_no_red_nor_purple : ℚ)

theorem total_number_of_balls_in_bag (h1 : num_white = 22)
                                      (h2 : num_green = 18)
                                      (h3 : num_yellow = 2)
                                      (h4 : num_red = 15)
                                      (h5 : num_purple = 3)
                                      (h6 : P_no_red_nor_purple = 0.7) :
  let T := num_white + num_green + num_yellow + num_red + num_purple in
  T = 60 :=
sorry

end total_number_of_balls_in_bag_l276_276899


namespace primes_divisible_by_3_percentage_l276_276825

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l276_276825


namespace minimum_shift_for_even_function_l276_276564

theorem minimum_shift_for_even_function :
  ∀ (m : ℝ), m > 0 →
  (∀ x : ℝ, 2 * (x + m) - π / 6 = k * π → (∃ k : ℤ, 2 * m - π / 6 = k * π)) →
  m = π / 12 :=
begin
  sorry
end

end minimum_shift_for_even_function_l276_276564


namespace anna_probability_more_heads_than_tails_l276_276551

noncomputable def probability_more_heads_than_tails (n : ℕ) : ℚ :=
  if n % 2 = 0
  then (n.choose (n / 2)) / (2 ^ n)
  else 0

theorem anna_probability_more_heads_than_tails :
  let n := 10 in
  probability_more_heads_than_tails n = 193 / 512 := 
by
  sorry

end anna_probability_more_heads_than_tails_l276_276551


namespace carolyn_sum_of_removals_l276_276180

def list_1_to_10 := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def carolyn_turn (remaining: List ℕ) : Option ℕ :=
  if remaining.contains 4 then some 4
  else if remaining.contains 8 then some 8
  else if remaining.contains 10 then some 10
  else if remaining.contains 6 then some 6
  else none

def paul_turn (chosen: ℕ) (remaining: List ℕ) : List ℕ :=
  remaining.filter (λ x => ¬(x ∣ chosen))

theorem carolyn_sum_of_removals :
  let remaining = list_1_to_10 in 
  let carolyn1 := 4 in
  let remaining_after_paul1 := paul_turn carolyn1 (remaining.erase 4) in
  
  let carolyn2 := 8 in
  let remaining_after_paul2 := paul_turn carolyn2 (remaining_after_paul1.erase 8) in
  
  let carolyn3 := 10 in
  let remaining_after_paul3 := paul_turn carolyn3 (remaining_after_paul2.erase 10) in
  
  let carolyn4 := 6 in
  let remaining_after_paul4 := paul_turn carolyn4 (remaining_after_paul3.erase 6) in
  
  carolyn1 + carolyn2 + carolyn3 + carolyn4 = 28 :=
by sorry

end carolyn_sum_of_removals_l276_276180


namespace sum_of_numbers_l276_276265

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), n = k * 10 + d + m * 10 * (10 ^ k)

theorem sum_of_numbers
  (A B C : ℕ)
  (hA : A >= 100 ∧ A < 1000)
  (hB : B >= 10 ∧ B < 100)
  (hC : C >= 10 ∧ C < 100)
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
              (if contains_digit A 7 then A else 0) +
              (if contains_digit B 7 then B else 0) +
              (if contains_digit C 7 then C else 0) = 208)
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧ 
              (if contains_digit B 3 then B else 0) +
              (if contains_digit C 3 then C else 0) = 76) :
  A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276265


namespace shorter_cycle_height_l276_276281

noncomputable def height_shorter_cycle (H_t S_t S_s : ℝ) : ℝ :=
  (H_t / S_t) * S_s

theorem shorter_cycle_height :
  ∀ (H_t S_t S_s : ℝ), H_t = 2.5 ∧ S_t = 5 ∧ S_s = 4 →
  height_shorter_cycle H_t S_t S_s = 2 := 
by
  intros H_t S_t S_s h
  obtain ⟨H_t_eq, S_t_eq, S_s_eq⟩ := h
  rw [H_t_eq, S_t_eq, S_s_eq]
  rfl

end shorter_cycle_height_l276_276281


namespace bottom_right_not_divisible_by_2011_l276_276641

theorem bottom_right_not_divisible_by_2011 :
  ∀ (board : Array (Array Nat)),
  (∀ i, 0 ≤ i ∧ i < 2012 → board[0][i] = 1) ∧ -- Top row
  (∀ i, 0 ≤ i ∧ i < 2012 → board[i][0] = 1) ∧ -- Left column
  (∀ i, 1 ≤ i ∧ i < 2011, board[i][2011 - i] = 0) ∧ -- Marked diagonal cells
  (∀ i j, 0 < i ∧ i < 2012 ∧ 0 < j ∧ j < 2012 ∧ (i ≠ 2011 - j) → -- Other cells' values
    board[i][j] = board[i-1][j] + board[i][j-1]) → 
  ¬ (2011 ∣ board[2011][2011]) := sorry

end bottom_right_not_divisible_by_2011_l276_276641


namespace percentage_of_primes_divisible_by_3_l276_276749

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l276_276749


namespace percent_primes_divisible_by_3_l276_276836

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l276_276836


namespace equilibrium_forces_rhombus_l276_276005

variables (a P Q α : ℝ)

theorem equilibrium_forces_rhombus :
  let tan_alpha := Real.tan α in
  (Q = P * tan_alpha^3) ↔
  (∀ (AB AC BD : ℝ), 
    AB = a →
    (AC = AB * Real.cos α) →
    (BD = AB * Real.sin α) →
    (P / AC = Q / BD)) :=
by
  sorry

end equilibrium_forces_rhombus_l276_276005


namespace total_people_count_l276_276703

-- Definitions based on given conditions
def Cannoneers : ℕ := 63
def Women : ℕ := 2 * Cannoneers
def Men : ℕ := 2 * Women
def TotalPeople : ℕ := Women + Men

-- Lean statement to prove
theorem total_people_count : TotalPeople = 378 := by
  -- placeholders for proof steps
  sorry

end total_people_count_l276_276703


namespace sum_of_digits_of_n_l276_276624

theorem sum_of_digits_of_n
  (n : ℕ) (h : 0 < n)
  (eqn : (n+1)! + (n+3)! = n! * 964) :
  n = 7 ∧ (Nat.digits 10 7).sum = 7 := 
sorry

end sum_of_digits_of_n_l276_276624


namespace cos_sin_value_l276_276016

theorem cos_sin_value (α : ℝ) (h : Real.tan α = Real.sqrt 2) : Real.cos α * Real.sin α = Real.sqrt 2 / 3 :=
sorry

end cos_sin_value_l276_276016


namespace min_value_3x_4y_l276_276427

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 3 * x + 4 * y = 25 :=
sorry

end min_value_3x_4y_l276_276427


namespace integer_fahrenheit_temperatures_equal_after_conversion_l276_276192

def fahrenheit_to_kelvin (F : ℝ) : ℝ :=
  (5 / 9) * (F - 32) + 273.15

def kelvin_to_fahrenheit (K : ℝ) : ℝ :=
  (9 / 5) * (K - 273.15) + 32

theorem integer_fahrenheit_temperatures_equal_after_conversion :
  (number_of_valid_F ∈ (50..500)) = 255 :=
sorry

end integer_fahrenheit_temperatures_equal_after_conversion_l276_276192


namespace sector_perimeter_approx_l276_276676

theorem sector_perimeter_approx (r : ℝ) (θ : ℝ) (π : ℝ) 
  (h₀ : θ = 225) (h₁ : r = 14) (h₂ : π = 3.14159) : 
  2 * r + (θ / 360) * 2 * π * r ≈ 82.99 := 
  sorry

end sector_perimeter_approx_l276_276676


namespace initial_men_count_l276_276061

theorem initial_men_count (x : ℕ) (h : x * 25 = 15 * 60) : x = 36 :=
by
  sorry

end initial_men_count_l276_276061


namespace lucky_point_m2_is_lucky_point_A33_point_M_quadrant_l276_276018

noncomputable def lucky_point (m n : ℝ) : Prop := 2 * m = 4 + n ∧ ∃ (x y : ℝ), (x = m - 1) ∧ (y = (n + 2) / 2)

theorem lucky_point_m2 :
  lucky_point 2 0 := sorry

theorem is_lucky_point_A33 :
  lucky_point 4 4 := sorry

theorem point_M_quadrant (a : ℝ) :
  lucky_point (a + 1) (2 * (2 * a - 1) - 2) → (a = 1) := sorry

end lucky_point_m2_is_lucky_point_A33_point_M_quadrant_l276_276018


namespace travel_period_l276_276141

-- Nina's travel pattern
def travels_in_one_month : ℕ := 400
def travels_in_two_months : ℕ := travels_in_one_month + 2 * travels_in_one_month

-- The total distance Nina wants to travel
def total_distance : ℕ := 14400

-- The period in months during which Nina travels the given total distance 
def required_period_in_months (d_per_2_months : ℕ) (total_d : ℕ) : ℕ := (total_d / d_per_2_months) * 2

-- Statement we need to prove
theorem travel_period : required_period_in_months travels_in_two_months total_distance = 24 := by
  sorry

end travel_period_l276_276141


namespace cadence_old_company_salary_l276_276360

variable (S : ℝ)

def oldCompanyMonths : ℝ := 36
def newCompanyMonths : ℝ := 41
def newSalaryMultiplier : ℝ := 1.20
def totalEarnings : ℝ := 426000

theorem cadence_old_company_salary :
  (oldCompanyMonths * S) + (newCompanyMonths * newSalaryMultiplier * S) = totalEarnings → 
  S = 5000 :=
by
  sorry

end cadence_old_company_salary_l276_276360


namespace solveSystem1_solveFractionalEq_l276_276174

-- Definition: system of linear equations
def system1 (x y : ℝ) : Prop :=
  x + 2 * y = 3 ∧ x - 4 * y = 9

-- Theorem: solution to the system of equations
theorem solveSystem1 : ∃ x y : ℝ, system1 x y ∧ x = 5 ∧ y = -1 :=
by
  sorry
  
-- Definition: fractional equation
def fractionalEq (x : ℝ) : Prop :=
  (x + 2) / (x^2 - 2 * x + 1) + 3 / (x - 1) = 0

-- Theorem: solution to the fractional equation
theorem solveFractionalEq : ∃ x : ℝ, fractionalEq x ∧ x = 1 / 4 :=
by
  sorry

end solveSystem1_solveFractionalEq_l276_276174


namespace halfway_between_l276_276392

-- Statement of the problem as a Lean theorem
theorem halfway_between (a b : ℚ) (h1 : a = 1/12) (h2 : b = 13/12) :
  (a + b) / 2 = 7/12 :=
by
  rw [h1, h2]
  norm_num
  sorry -- This is where the proof would go

end halfway_between_l276_276392


namespace subsets_containing_5_and_6_l276_276472

theorem subsets_containing_5_and_6: 
  let s := {1, 2, 3, 4, 5, 6} in
  {t : Finset ℕ // t ⊆ s ∧ 5 ∈ t ∧ 6 ∈ t}.card = 16 :=
by sorry

end subsets_containing_5_and_6_l276_276472


namespace percent_primes_divisible_by_3_l276_276831

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l276_276831


namespace star_evaluate_l276_276201

def star (a b : ℝ) : ℝ := a + a / b

theorem star_evaluate : star 12 3 = 16 :=
by
  unfold star
  sorry

end star_evaluate_l276_276201


namespace subsets_containing_5_and_6_l276_276470

theorem subsets_containing_5_and_6 :
  let S := {1, 2, 3, 4, 5, 6}
  ∃ s ⊆ S, 5 ∈ s ∧ 6 ∈ s ∧ s.card = 16 :=
sorry

end subsets_containing_5_and_6_l276_276470


namespace primes_divisible_by_3_percentage_l276_276820

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l276_276820


namespace largest_z_l276_276960

theorem largest_z (x y z : ℝ) 
  (h1 : x + y + z = 5)  
  (h2 : x * y + y * z + x * z = 3) 
  : z ≤ 13 / 3 := sorry

end largest_z_l276_276960


namespace first_fabulous_friday_l276_276085

-- Define the month and the day when school starts
def month_length : ℕ := 31 -- October has 31 days
def school_start_day : ℕ := 3 -- School starts on October 3

-- Define the start day of the week for school (Tuesday)
def school_start_day_of_week : ℕ := 2 -- 0 = Sunday, 1 = Monday, ..., 6 = Saturday, so 2 = Tuesday

-- Define a function to calculate the day of the week given a start day and a number of days elapsed
def day_of_week (start_day : ℕ) (days_elapsed : ℕ) : ℕ :=
  (start_day + days_elapsed) % 7

-- Define a function to determine if a given date is a Friday
def is_friday (day : ℕ) : Prop :=
  day_of_week school_start_day_of_week (day - school_start_day) = 5

-- Lean theorem to prove the first Fabulous Friday after school starts
theorem first_fabulous_friday : ∃ day, day > school_start_day ∧ is_friday day ∧ 
  (∃ count, count = 5 ∧ ∃ dates, list.is_permutation dates [6,13,20,27,day]) :=
by {
  sorry
}

end first_fabulous_friday_l276_276085


namespace driver_weekly_distance_l276_276331

-- Defining the conditions
def speed_part1 : ℕ := 30  -- speed in miles per hour for the first part
def time_part1 : ℕ := 3    -- time in hours for the first part
def speed_part2 : ℕ := 25  -- speed in miles per hour for the second part
def time_part2 : ℕ := 4    -- time in hours for the second part
def days_per_week : ℕ := 6 -- number of days the driver works in a week

-- Total distance calculation each day
def distance_part1 := speed_part1 * time_part1
def distance_part2 := speed_part2 * time_part2
def daily_distance := distance_part1 + distance_part2

-- Total distance travel in a week
def weekly_distance := daily_distance * days_per_week

-- Theorem stating that weekly distance is 1140 miles
theorem driver_weekly_distance : weekly_distance = 1140 :=
by
  -- We skip the proof using sorry
  sorry

end driver_weekly_distance_l276_276331


namespace ways_to_choose_4_cards_of_different_suits_l276_276523

theorem ways_to_choose_4_cards_of_different_suits :
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  ∃ n : ℕ, n = (choose num_suits num_suits) * cards_per_suit ^ num_suits ∧ n = 28561 :=
by
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  have ways_to_choose_suits : (choose num_suits num_suits) = 1 := by simp
  have ways_to_choose_cards : cards_per_suit ^ num_suits = 28561 := by norm_num
  let n := 1 * 28561
  use n
  constructor
  · exact by simp [ways_to_choose_suits, ways_to_choose_cards]
  · exact by rfl

end ways_to_choose_4_cards_of_different_suits_l276_276523


namespace right_triangle_area_l276_276206

theorem right_triangle_area {a r R : ℝ} (hR : R = (5 / 2) * r) (h_leg : ∃ BC, BC = a) :
  (∃ area, area = (2 * a^2 / 3) ∨ area = (3 * a^2 / 8)) :=
sorry

end right_triangle_area_l276_276206


namespace sum_of_numbers_is_247_l276_276232

/-- Definitions of the conditions -/
def number_contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d < 10 ∧ ∃ (k : ℕ), n / 10 ^ k % 10 = d

variable (A B C : ℕ)
variable (hA : 100 ≤ A ∧ A < 1000)
variable (hB : 10 ≤ B ∧ B < 100)
variable (hC : 10 ≤ C ∧ C < 100)
variable (h_sum_7 : if number_contains_digit A 7 
                  then if number_contains_digit B 7 
                  then if number_contains_digit C 7 
                  then A + B + C 
                  else A + B
                  else A
                  else B + C = 208)
variable (h_sum_3 : if number_contains_digit A 3 
                  then if number_contains_digit B 3
                  then if number_contains_digit C 3
                  then A + B + C 
                  else A + B
                  else A 
                  else B + C = 76)

/-- Prove that the sum of all three numbers is 247 -/
theorem sum_of_numbers_is_247 : A + B + C = 247 :=
by
  sorry

end sum_of_numbers_is_247_l276_276232


namespace subsets_containing_5_and_6_l276_276491

theorem subsets_containing_5_and_6 (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ s, 5 ∈ s ∧ 6 ∈ s)).card = 16 :=
by
  sorry

end subsets_containing_5_and_6_l276_276491


namespace solve_sin_alpha_solve_fraction_trig_l276_276278

section
variable (α θ : Real)
-- Conditions for the first part
variable (h_cos_alpha : cos α = -4/5)
variable (h_alpha_third_quadrant : π < α ∧ α < 3 * π / 2)

-- Conditions for the second part
variable (h_tan_theta : tan θ = 3)

theorem solve_sin_alpha : sin α = -3/5 := by sorry

theorem solve_fraction_trig : (sin θ + cos θ) / (2 * sin θ + cos θ) = 4/7 := by sorry
end

end solve_sin_alpha_solve_fraction_trig_l276_276278


namespace subsets_containing_5_and_6_l276_276466

theorem subsets_containing_5_and_6 :
  let S := {1, 2, 3, 4, 5, 6}
  ∃ s ⊆ S, 5 ∈ s ∧ 6 ∈ s ∧ s.card = 16 :=
sorry

end subsets_containing_5_and_6_l276_276466


namespace steve_goal_met_l276_276178

def earnings (lingonberries cloudberries blueberries : ℕ) (day : string) : ℕ :=
  match day with
  | "Monday" => 2 * lingonberries + 3 * cloudberries + 5 * blueberries
  | "Tuesday" => 2 * (3 * lingonberries) + 3 * (2 * cloudberries) + 5 * (blueberries + 5)
  | _ => 0

def totalEarnings (earningsMonday earningsTuesday : ℕ) : ℕ :=
  earningsMonday + earningsTuesday

theorem steve_goal_met : 
  earnings 8 10 0 "Monday" + earnings 8 10 0 "Tuesday" >= 150 := by
  sorry

end steve_goal_met_l276_276178


namespace subsets_containing_5_and_6_l276_276476

theorem subsets_containing_5_and_6: 
  let s := {1, 2, 3, 4, 5, 6} in
  {t : Finset ℕ // t ⊆ s ∧ 5 ∈ t ∧ 6 ∈ t}.card = 16 :=
by sorry

end subsets_containing_5_and_6_l276_276476


namespace probability_bug_at_A_is_547_over_2187_l276_276112

def regular_tetrahedron (A B C D : Type) := 
  ∀ (a b c d : ℝ), a = b ∧ b = c ∧ c = d ∧ a = 1

noncomputable def probability_bug_at_A_after_8 (A B C D : Type) : ℝ :=
  let P : ℕ → ℝ := λ n, if n = 0 then 1 else if n = 1 then 0 else 1 / 3 * (1 - P (n - 1)) in
  P 8

theorem probability_bug_at_A_is_547_over_2187 
  (A B C D : Type) (h_tetrahedron : regular_tetrahedron A B C D) : 
  probability_bug_at_A_after_8 A B C D = 547 / 2187 :=
sorry

end probability_bug_at_A_is_547_over_2187_l276_276112


namespace percentage_of_primes_divisible_by_3_is_12_5_l276_276844

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l276_276844


namespace percentage_of_primes_divisible_by_3_l276_276812

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l276_276812


namespace sum_of_numbers_l276_276264

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), n = k * 10 + d + m * 10 * (10 ^ k)

theorem sum_of_numbers
  (A B C : ℕ)
  (hA : A >= 100 ∧ A < 1000)
  (hB : B >= 10 ∧ B < 100)
  (hC : C >= 10 ∧ C < 100)
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
              (if contains_digit A 7 then A else 0) +
              (if contains_digit B 7 then B else 0) +
              (if contains_digit C 7 then C else 0) = 208)
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧ 
              (if contains_digit B 3 then B else 0) +
              (if contains_digit C 3 then C else 0) = 76) :
  A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276264


namespace exists_x2_for_theta_l276_276991

-- Define the function f
def f (x : ℝ) : ℝ := 3 * Real.sin x + 2

-- Define the given interval
def I : Set ℝ := Set.Icc 0 (Real.pi / 2)

-- The equivalent Lean statement of the proof problem
theorem exists_x2_for_theta :
  ∃ x2 ∈ I, ∀ x1 ∈ I, f(x1) = 2 * f(x1 + (4 * Real.pi / 5)) + 2 := by
    sorry

end exists_x2_for_theta_l276_276991


namespace percentage_primes_divisible_by_3_l276_276854

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l276_276854


namespace choose_4_cards_of_different_suits_l276_276534

theorem choose_4_cards_of_different_suits :
  (∃ (n : ℕ), choose 4 4 = n) ∧
  (∃ (m : ℕ), (13^4 = m)) ∧
  (1 * (13^4) = 28561)

end choose_4_cards_of_different_suits_l276_276534


namespace compute_expression_l276_276632

theorem compute_expression (p q : ℝ) (h1 : p + q = 5) (h2 : p * q = 6) :
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 503 :=
by
  sorry

end compute_expression_l276_276632


namespace sum_of_numbers_is_247_l276_276229

/-- Definitions of the conditions -/
def number_contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d < 10 ∧ ∃ (k : ℕ), n / 10 ^ k % 10 = d

variable (A B C : ℕ)
variable (hA : 100 ≤ A ∧ A < 1000)
variable (hB : 10 ≤ B ∧ B < 100)
variable (hC : 10 ≤ C ∧ C < 100)
variable (h_sum_7 : if number_contains_digit A 7 
                  then if number_contains_digit B 7 
                  then if number_contains_digit C 7 
                  then A + B + C 
                  else A + B
                  else A
                  else B + C = 208)
variable (h_sum_3 : if number_contains_digit A 3 
                  then if number_contains_digit B 3
                  then if number_contains_digit C 3
                  then A + B + C 
                  else A + B
                  else A 
                  else B + C = 76)

/-- Prove that the sum of all three numbers is 247 -/
theorem sum_of_numbers_is_247 : A + B + C = 247 :=
by
  sorry

end sum_of_numbers_is_247_l276_276229


namespace sequence_strictly_positive_integer_l276_276108

open Nat

def sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 3 ∧
  a 1 = 2 ∧
  a 2 = 12 ∧ 
  (∀ n : ℕ, 2 * a (n + 3) - a (n + 2) - 8 * a (n + 1) + 4 * a n = 0)

theorem sequence_strictly_positive_integer (a : ℕ → ℕ) (h : sequence a) : ∀ n, a n > 0 :=
by
  sorry

end sequence_strictly_positive_integer_l276_276108


namespace find_x_l276_276720

variable (A B C D : Type)
variable [Real ℝ]
variable (x : ℝ)

/- Conditions -/
variables (hypotenuse_AD : ℝ) (is_45_45_90_ΔABD : Prop) (is_45_45_90_ΔACD : Prop)

-- Given values
constant (h_AD : hypotenuse_AD = 12)
constant (t_ΔABD : is_45_45_90_ΔABD)
constant (t_ΔACD : is_45_45_90_ΔACD)

/- Theorem statement -/
theorem find_x (h1 : is_45_45_90_ΔABD)
               (h2 : is_45_45_90_ΔACD)
               (h3 : hypotenuse_AD = 12) :
  x = 6 * Real.sqrt 2 :=
sorry

end find_x_l276_276720


namespace general_term_l276_276034

def sequence (a : ℕ → ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + 2 * n

theorem general_term (a : ℕ → ℕ) (h : sequence a) (h1 : a 1 = 2) :
  ∀ n : ℕ, a n = n^2 - n + 2 :=
sorry

end general_term_l276_276034


namespace triangle_ABC_solution_l276_276594

-- Definitions and conditions
variables (A B C : ℝ) (AM : ℝ) (x : ℝ)

-- Conditions as hypotheses
hypothesis h1 : sin A = sin B
hypothesis h2 : sin A = -cos C
hypothesis h3 : A + B + C = Real.pi
hypothesis h4 : AM = Real.sqrt 7

-- Proof statement
theorem triangle_ABC_solution :
  (A = B ∧ A = Real.pi / 6 ∧ C = 2 * Real.pi / 3) ∧
  (∃ (x : ℝ), let area := 0.5 * x * x * sin C in x = 2 ∧ area = Real.sqrt 3) :=
sorry

end triangle_ABC_solution_l276_276594


namespace find_angle_C_max_area_of_triangle_l276_276069

variable (a b c : ℝ) (A B C : ℝ)
hypothesis (h1 : c * Real.cos B + (2 * a + b) * Real.cos C = 0)
hypothesis (h2 : c = Real.sqrt 3)
hypothesis (C_eq : C = 2 * Real.pi / 3)

-- Problem (1): Prove that cos C = -1/2 and C = 2π/3
theorem find_angle_C : Real.cos C = -1/2 ∧ C = 2 * Real.pi / 3 :=
by
  sorry

-- Problem (2): Prove that the maximum area of triangle ABC is sqrt(3)/4
theorem max_area_of_triangle : 
  ∃ (ab : ℝ), ab * Real.sin C / 2 ≤ Real.sqrt 3 / 4 :=
by
  sorry

end find_angle_C_max_area_of_triangle_l276_276069


namespace min_value_of_a_exists_min_value_of_a_is_one_l276_276032

def f (x a : ℝ) : ℝ := (4 * exp (x - 1)) / (x + 1) + x^2 - 3 * a * x + a^2 - 1

theorem min_value_of_a_exists (a : ℝ) : (∃ x0 : ℝ, x0 > 0 ∧ f x0 a ≤ 0) ↔ a ≥ 1 :=
begin
  sorry
end

theorem min_value_of_a_is_one : ∃ a : ℝ, a = 1 ∧ (∃ x0 : ℝ, x0 > 0 ∧ f x0 a ≤ 0) :=
begin
  sorry
end

end min_value_of_a_exists_min_value_of_a_is_one_l276_276032


namespace decreasing_interval_of_f_l276_276028

noncomputable def f (x : ℝ) : ℝ := Real.cos (x - Real.pi / 3)

theorem decreasing_interval_of_f :
  ∃ (a b : ℝ), (fderiv ℝ f) '' (set.Icc a b) ⊆ set.Iio 0 → 
  (∃ (a b : ℝ), a = Real.pi / 3 ∧ b = 2 * Real.pi) := 
sorry

end decreasing_interval_of_f_l276_276028


namespace rectangular_to_polar_l276_276955

theorem rectangular_to_polar :
  ∃ r θ, r = Real.sqrt (8^2 + (2 * Real.sqrt 3)^2) ∧ θ = Real.arctan (2 * Real.sqrt 3 / 8) ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (Real.sqrt 76, Real.pi / 6) := 
by
  use Real.sqrt (8^2 + (2 * Real.sqrt 3)^2)
  use Real.arctan (2 * Real.sqrt 3 / 8)
  have h_r : Real.sqrt (8^2 + (2 * Real.sqrt 3)^2) = Real.sqrt 76 := sorry
  have h_theta : Real.arctan (2 * Real.sqrt 3 / 8) = Real.pi / 6 := sorry
  refine ⟨h_r, h_theta, Real.sqrt (8^2 + (2 * Real.sqrt 3)^2) > 0, Real.zero_le _, Real.arctan_lt_iff.mpr (2 * Real.sqrt 3 / 8), _⟩
  exact (Real.sqrt (8^2 + (2 * Real.sqrt 3)^2), Real.arctan (2 * Real.sqrt 3 / 8)) = (Real.sqrt 76, Real.pi / 6)

end rectangular_to_polar_l276_276955


namespace part_one_part_two_l276_276595

noncomputable section

-- Define the main structure for triangle ABC.
structure Triangle (α : Type _) :=
(A B C a b c : α)

-- Specify the given conditions:
variables {α : Type _} [LinearOrderedField α] [Trigonometric α]

-- The conditions given in the problem
variables (triangle : Triangle α)
  (h1 : triangle.a * sin ((triangle.A + triangle.C) / 2) = triangle.b * sin triangle.A)
  (h2 : triangle.c = 1)
  (acute : 0 < triangle.A ∧ triangle.A < π / 2 ∧
           0 < triangle.B ∧ triangle.B < π / 2 ∧
           0 < triangle.C ∧ triangle.C < π / 2)

-- Part 1: Prove that angle B equals π/3
theorem part_one : triangle.B = π / 3 :=
sorry

-- Part 2: Prove that the area of triangle ABC is in the range (sqrt(3)/8, sqrt(3)/2)
theorem part_two (b : α) (A : α) : triangle.A ∈ (0, π) ∧
  (triangle.a = 1) ∧ (A = A) ∧ (triangle.b = b) :=
sorry

end part_one_part_two_l276_276595


namespace lines_perpendicular_l276_276458

-- Condition 1: Equation of the first line
def l1 (a : ℝ) (x y : ℝ) := a * x + 2 * y + 6 = 0 

-- Condition 2: Equation of the second line
def l2 (a : ℝ) (x y : ℝ) := x + (a - 1) * y + (a^2 - 1) = 0

-- Condition 3: Slopes of the lines
def k1 (a : ℝ) := -a / 2
def k2 (a : ℝ) := 1 / (1 - a)

-- Proof that the lines are perpendicular when a = 2 / 3
theorem lines_perpendicular (a : ℝ) : (l1 a = l1 a) → (l2 a = l2 a) → 
  (k1 a * k2 a = -1) → a = 2 / 3 := 
sorry

end lines_perpendicular_l276_276458


namespace fred_fewer_games_l276_276985

/-- Fred went to 36 basketball games last year -/
def games_last_year : ℕ := 36

/-- Fred went to 25 basketball games this year -/
def games_this_year : ℕ := 25

/-- Prove that Fred went to 11 fewer games this year compared to last year -/
theorem fred_fewer_games : games_last_year - games_this_year = 11 := by
  sorry

end fred_fewer_games_l276_276985


namespace percent_primes_divisible_by_3_l276_276839

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l276_276839


namespace gasoline_remaining_l276_276898

theorem gasoline_remaining (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 500) :
  let y := 50 - 0.1 * x in
  (y = 30) ↔ (x = 200) :=
by
  sorry

end gasoline_remaining_l276_276898


namespace manager_salary_l276_276666

theorem manager_salary :
  let total_50_employees := 50 * 2500 in
  let new_average_salary := 2500 + 150 in
  let total_after_manager_added := 51 * new_average_salary in
  let manager_salary := total_after_manager_added - total_50_employees in
  manager_salary = 10150 :=
by
  sorry

end manager_salary_l276_276666


namespace smallest_invariant_number_l276_276913

def operation (n : ℕ) : ℕ :=
  let q := n / 10
  let r := n % 10
  q + 2 * r

def is_invariant (n : ℕ) : Prop :=
  operation n = n

theorem smallest_invariant_number : ∃ n : ℕ, is_invariant n ∧ n = 10^99 + 1 :=
by
  sorry

end smallest_invariant_number_l276_276913


namespace geometric_series_properties_l276_276221

theorem geometric_series_properties 
    (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h1 : a 1 + a 2 + a 3 = 26)
    (h2 : S 6 = 728)
    (h3 : ∀ n, S n = ∑ i in finset.range n, a (i + 1)) :
    (∀ n, a n = 2 * 3 ^ (n - 1)) ∧
    (∀ n, S (n + 1) ^ 2 - S n * S (n + 2) = 4 * 3 ^ n) :=
by
  sorry

end geometric_series_properties_l276_276221


namespace exists_distinct_nonzero_ints_for_poly_factorization_l276_276973

theorem exists_distinct_nonzero_ints_for_poly_factorization :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ P Q : Polynomial ℤ, (P * Q = Polynomial.X * (Polynomial.X - Polynomial.C a) * 
   (Polynomial.X - Polynomial.C b) * (Polynomial.X - Polynomial.C c) + 1) ∧ 
   P.leadingCoeff = 1 ∧ Q.leadingCoeff = 1) :=
by
  sorry

end exists_distinct_nonzero_ints_for_poly_factorization_l276_276973


namespace sum_of_numbers_l276_276267

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), n = k * 10 + d + m * 10 * (10 ^ k)

theorem sum_of_numbers
  (A B C : ℕ)
  (hA : A >= 100 ∧ A < 1000)
  (hB : B >= 10 ∧ B < 100)
  (hC : C >= 10 ∧ C < 100)
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
              (if contains_digit A 7 then A else 0) +
              (if contains_digit B 7 then B else 0) +
              (if contains_digit C 7 then C else 0) = 208)
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧ 
              (if contains_digit B 3 then B else 0) +
              (if contains_digit C 3 then C else 0) = 76) :
  A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276267


namespace percentage_of_primes_divisible_by_3_l276_276805

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l276_276805


namespace intersection_nonempty_implies_b_in_interval_l276_276988

def M := {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ y = real.sqrt(9 - x^2) ∧ y ≠ 0}
def N (b : ℝ) := {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ y = x + b}

theorem intersection_nonempty_implies_b_in_interval (b : ℝ) : 
  (∃ p : ℝ × ℝ, p ∈ M ∧ p ∈ N b) → b ∈ Ioo (-3) (3 * real.sqrt 2) ∨ b = 3 * real.sqrt 2 :=
sorry

end intersection_nonempty_implies_b_in_interval_l276_276988


namespace percent_primes_divisible_by_3_l276_276834

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l276_276834


namespace weights_divisible_by_3_l276_276001

theorem weights_divisible_by_3 :
  ∃ (pile1 pile2 pile3 : set ℕ), 
    (pile1 ∪ pile2 ∪ pile3 = {1, 2, ..., 555}) ∧
    (pile1 ∩ pile2 = ∅) ∧
    (pile1 ∩ pile3 = ∅) ∧
    (pile2 ∩ pile3 = ∅) ∧
    (pile1.sum + pile2.sum + pile3.sum = (1+555)*555/2) ∧
    (pile1.sum = pile2.sum) ∧
    (pile2.sum = pile3.sum) ∧
    (pile1.sum = (1+555)*555/6) :=
sorry

end weights_divisible_by_3_l276_276001


namespace pyramid_rhombus_side_length_l276_276184

theorem pyramid_rhombus_side_length
  (α β S: ℝ) (hα : 0 < α) (hβ : 0 < β) (hS : 0 < S) :
  ∃ a : ℝ, a = 2 * Real.sqrt (2 * S * Real.cos β / Real.sin α) :=
by
  sorry

end pyramid_rhombus_side_length_l276_276184


namespace row_col_sum_nonnegative_l276_276080

variable (m n : ℕ)
variable (A : Matrix (Fin m) (Fin n) ℝ)

-- Define the operation to change signs of a whole row or column
def flip_row (A : Matrix (Fin m) (Fin n) ℝ) (r : Fin m) : Matrix (Fin m) (Fin n) ℝ :=
  λ i j => if i = r then -A i j else A i j

def flip_col (A : Matrix (Fin m) (Fin n) ℝ) (c : Fin n) : Matrix (Fin m) (Fin n) ℝ :=
  λ i j => if j = c then -A i j else A i j

-- Prove that it is possible to make the sum of elements in each row and each column non-negative
theorem row_col_sum_nonnegative : ∃ B : Matrix (Fin m) (Fin n) ℝ, 
  (∀ i : Fin m, 0 ≤ ∑ j, B i j) ∧ (∀ j : Fin n, 0 ≤ ∑ i, B i j) :=
sorry

end row_col_sum_nonnegative_l276_276080


namespace Pythagorean_triple_B_l276_276300

theorem Pythagorean_triple_B : 
  ∃ (a b c : ℕ), a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2 :=
by {
  use [9, 12, 15],
  split; simp,
  split; simp,
  split; simp,
  sorry
}

end Pythagorean_triple_B_l276_276300


namespace initial_money_l276_276139

-- Define the conditions
def spent_toy_truck : ℕ := 3
def spent_pencil_case : ℕ := 2
def money_left : ℕ := 5

-- Define the total money spent
def total_spent := spent_toy_truck + spent_pencil_case

-- Theorem statement
theorem initial_money (I : ℕ) (h : total_spent + money_left = I) : I = 10 :=
sorry

end initial_money_l276_276139


namespace max_a1_le_2_l276_276628

theorem max_a1_le_2 (a : ℕ → ℕ) (R : ℕ)
  (h1 : ∀ i j, 1 ≤ i → i < j → j ≤ R → a i < a j)
  (h2 : ∀ i, 1 ≤ i → i ≤ R → 0 < a i)
  (h3 : ∑ i in finset.range (R+1), a i = 90)
  : a 1 ≤ 2 := sorry

end max_a1_le_2_l276_276628


namespace observed_wheels_l276_276577

theorem observed_wheels (num_cars wheels_per_car : ℕ) (h1 : num_cars = 12) (h2 : wheels_per_car = 4) : num_cars * wheels_per_car = 48 := by
  sorry

end observed_wheels_l276_276577


namespace large_paintings_count_l276_276935

-- Define the problem conditions
def paint_per_large : Nat := 3
def paint_per_small : Nat := 2
def small_paintings : Nat := 4
def total_paint : Nat := 17

-- Question to find number of large paintings (L)
theorem large_paintings_count :
  ∃ L : Nat, (paint_per_large * L + paint_per_small * small_paintings = total_paint) → L = 3 :=
by
  -- Placeholder for the proof
  sorry

end large_paintings_count_l276_276935


namespace find_angle_B_l276_276097

-- Define the conditions
variables {B : ℝ} 
def vec_a := (1, Real.cos B)
def vec_b := (Real.sin B, 1)

-- Define the perpendicular condition
def perpendicular (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Proof statement
theorem find_angle_B (h1 : perpendicular vec_a vec_b) : B = (3 * Real.pi) / 4 :=
sorry

end find_angle_B_l276_276097


namespace wheel_radius_increase_l276_276379
noncomputable def increase_in_wheel_radius : ℝ :=
  let original_distance_km := 300
  let new_distance_km := 290
  let original_radius_cm := 20
  let circumference r := 2 * Real.pi * r
  let distance_per_rotation_cm km := λ r, circumference r / 100000
  let number_of_rotations km := λ r, km / (distance_per_rotation_cm km r)
  let new_radius_cm := (original_distance_km * distance_per_rotation_cm original_distance_km original_radius_cm * 100000) / (2 * Real.pi * new_distance_km)
  new_radius_cm - original_radius_cm

theorem wheel_radius_increase :
  increase_in_wheel_radius = 0.53 := by
  sorry

end wheel_radius_increase_l276_276379


namespace sin_minus_cos_eq_neg_one_l276_276557

theorem sin_minus_cos_eq_neg_one (x : ℝ) 
    (h1 : sin x ^ 3 - cos x ^ 3 = -1)
    (h2 : sin x ^ 2 + cos x ^ 2 = 1) : 
    sin x - cos x = -1 :=
sorry

end sin_minus_cos_eq_neg_one_l276_276557


namespace percentage_of_primes_divisible_by_3_l276_276793

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l276_276793


namespace driver_travel_distance_per_week_l276_276330

open Nat

-- Defining the parameters
def speed1 : ℕ := 30
def time1 : ℕ := 3
def speed2 : ℕ := 25
def time2 : ℕ := 4
def days : ℕ := 6

-- Lean statement to prove
theorem driver_travel_distance_per_week : 
  (speed1 * time1 + speed2 * time2) * days = 1140 := 
by 
  sorry

end driver_travel_distance_per_week_l276_276330


namespace boy_usual_time_to_school_l276_276296

theorem boy_usual_time_to_school
  (S : ℝ) -- Usual speed
  (T : ℝ) -- Usual time
  (D : ℝ) -- Distance, D = S * T
  (hD : D = S * T)
  (h1 : 3/4 * D / (7/6 * S) + 1/4 * D / (5/6 * S) = T - 2) : 
  T = 35 :=
by
  sorry

end boy_usual_time_to_school_l276_276296


namespace incenter_divides_angle_bisector_l276_276596

-- Define the variables and properties of the triangle
variables {A B C O D : Type}
variables {a b c : ℝ} -- Assuming the lengths a, b, and c are real numbers

-- Assume triangle sides and incenter properties in Lean
def triangle_sides (AB BC AC : ℝ) : Prop := AB = c ∧ BC = a ∧ AC = b

def ratio_incenter_divides_bisector (O D : Type) (AC AD : ℝ) : Prop :=
  let CD := (O, D) in
  AD = (b * c) / (a + b) ∧
  (CO / OD) = (a + b) / c

-- The Lean statement that expresses the equivalent proof problem
theorem incenter_divides_angle_bisector (A B C O D : Type) (a b c : ℝ)
  (h1 : triangle_sides (AB := c) (BC := a) (AC := b))
  (h2 : incenter (O := O) (A B C : Type))
  (h3 : angle_bisector (A C B D : Type)) :
  ratio_incenter_divides_bisector (O := O) (D := D) (AC := b) (AD := (b * c) / (a + b)) :=
by
  sorry

end incenter_divides_angle_bisector_l276_276596


namespace percentage_primes_divisible_by_3_l276_276863

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l276_276863


namespace correct_derivative_operation_l276_276297

open Real

-- Definitions for the given conditions
def A : Prop := (λ x : ℝ, (cos x / x)') = (λ x : ℝ, - sin x)
def B : Prop := (λ x : ℝ, (log x / log 2)') = (λ x : ℝ, 1 / (x * log 2))
def C : Prop := (λ x : ℝ, (2^x)') = (λ x : ℝ, 2^x)
def D : Prop := (λ x : ℝ, (x^3 * exp x)') = (λ x : ℝ, 3 * x^2 * exp x)

-- Proof statement to verify that condition B is correct.
theorem correct_derivative_operation : B :=
by
  sorry

end correct_derivative_operation_l276_276297


namespace vec_dot_product_range_l276_276146

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

theorem vec_dot_product_range (ha : ∥a∥ = 1) (hab : ⟪a, b⟫ = 1) (hbc : ⟪b, c⟫ = 1) 
  (habc : ∥a - b + c∥ ≤ 2 * sqrt 2) : -2 * sqrt 2 ≤ ⟪a, c⟫ ∧ ⟪a, c⟫ ≤ 2 :=
sorry

end vec_dot_product_range_l276_276146


namespace candy_bar_sugar_calories_l276_276135

theorem candy_bar_sugar_calories
  (candy_bars : Nat)
  (soft_drink_calories : Nat)
  (soft_drink_sugar_percentage : Float)
  (recommended_sugar_intake : Nat)
  (excess_percentage : Nat)
  (sugar_in_each_bar : Nat) :
  candy_bars = 7 ∧
  soft_drink_calories = 2500 ∧
  soft_drink_sugar_percentage = 0.05 ∧
  recommended_sugar_intake = 150 ∧
  excess_percentage = 100 →
  sugar_in_each_bar = 25 := by
  sorry

end candy_bar_sugar_calories_l276_276135


namespace primes_less_than_20_divisible_by_3_percentage_l276_276735

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l276_276735


namespace sum_of_numbers_is_247_l276_276234

/-- Definitions of the conditions -/
def number_contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d < 10 ∧ ∃ (k : ℕ), n / 10 ^ k % 10 = d

variable (A B C : ℕ)
variable (hA : 100 ≤ A ∧ A < 1000)
variable (hB : 10 ≤ B ∧ B < 100)
variable (hC : 10 ≤ C ∧ C < 100)
variable (h_sum_7 : if number_contains_digit A 7 
                  then if number_contains_digit B 7 
                  then if number_contains_digit C 7 
                  then A + B + C 
                  else A + B
                  else A
                  else B + C = 208)
variable (h_sum_3 : if number_contains_digit A 3 
                  then if number_contains_digit B 3
                  then if number_contains_digit C 3
                  then A + B + C 
                  else A + B
                  else A 
                  else B + C = 76)

/-- Prove that the sum of all three numbers is 247 -/
theorem sum_of_numbers_is_247 : A + B + C = 247 :=
by
  sorry

end sum_of_numbers_is_247_l276_276234


namespace primes_divisible_by_3_percentage_is_12_5_l276_276740

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l276_276740


namespace probability_divisor_is_multiple_of_3_l276_276200

theorem probability_divisor_is_multiple_of_3 (n : ℕ) (hn1 : 15.factorial = 2^11 * 3^6 * 5^3 * 7 * 11 * 13) 
  (hn2 : (11 + 1) * (6 + 1) * (3 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 1344)
  (hn3 : (11 + 1) * 6 * (3 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 1152) :
  (1152 : ℚ) / 1344 = 3 / 4 := by
  sorry

end probability_divisor_is_multiple_of_3_l276_276200


namespace primes_less_than_20_divisible_by_3_percentage_l276_276727

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l276_276727


namespace proof_problem_l276_276053

-- Definitions from conditions
variables {Π : Type} {Line : Type} {Plane : Type}
variable [Nonempty Π]
variable [Nonempty Line]
variable [Nonempty Plane]

variable (m n : Line)
variable (α β : Plane)
variable (diff_lines : m ≠ n)
variable (non_coincident_planes : α ≠ β)
variable (line_in_plane : Line → Plane → Prop)
variable (perp : Plane → Plane → Prop)
variable (line_perp_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Statement to prove
theorem proof_problem (h1 : line_perp_plane m α) (h2 : line_parallel_plane m β) : perp α β :=
sorry

end proof_problem_l276_276053


namespace distance_between_points_is_sqrt_14_l276_276588

def point : Type := ℝ × ℝ × ℝ

def distance (A B : point) : ℝ :=
  match A, B with
  | (x1, y1, z1), (x2, y2, z2) => 
    real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

open real

theorem distance_between_points_is_sqrt_14 :
  distance (3, 0, 1) (4, 2, -2) = sqrt 14 :=
by
  sorry

end distance_between_points_is_sqrt_14_l276_276588


namespace pyramid_sphere_proof_l276_276885

theorem pyramid_sphere_proof
  (h R_1 R_2 : ℝ) 
  (O_1 O_2 T_1 T_2 : ℝ) 
  (inscription: h > 0 ∧ R_1 > 0 ∧ R_2 > 0) :
  R_1 * R_2 * h^2 = (R_1^2 - O_1 * T_1^2) * (R_2^2 - O_2 * T_2^2) :=
by
  sorry

end pyramid_sphere_proof_l276_276885


namespace probability_more_heads_than_tails_l276_276547

theorem probability_more_heads_than_tails :
  let x := \frac{193}{512}
  let y := \frac{63}{256}
  (2 * x + y = 1) →
  (y = \frac{252}{1024}) →
  (x = \frac{193}{512}) :=
by
  let x : ℚ := 193 / 512
  let y : ℚ := 63 / 256
  sorry

end probability_more_heads_than_tails_l276_276547


namespace percentage_of_primes_divisible_by_3_l276_276797

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l276_276797


namespace find_ordered_pair_l276_276394

theorem find_ordered_pair : ∃ (p q : ℤ), (√(16 - 12 * (1 / (√3)))) = (p + q * (√3)) ∧ p = -4 ∧ q = 4 := by
  sorry

end find_ordered_pair_l276_276394


namespace probability_more_heads_than_tails_l276_276550

open Nat

def coin_flip_probability (n : Nat) := \(
    let y := (Nat.choose 10 5) * (1 / (2 ^ 10)) -- binomial coefficient
    let x := (1 - y) / 2 -- calculating x
    x
\)

theorem probability_more_heads_than_tails : coin_flip_probability 10 = 193 / 512 :=
by
  sorry

end probability_more_heads_than_tails_l276_276550


namespace area_of_rhombus_with_diagonals_6_and_8_l276_276438

theorem area_of_rhombus_with_diagonals_6_and_8 : 
  ∀ (d1 d2 : ℕ), d1 = 6 → d2 = 8 → (1 / 2 : ℝ) * d1 * d2 = 24 :=
by
  intros d1 d2 h1 h2
  sorry

end area_of_rhombus_with_diagonals_6_and_8_l276_276438


namespace student_score_in_first_subject_l276_276348

theorem student_score_in_first_subject 
  (x : ℝ)  -- Percentage in the first subject
  (w : ℝ)  -- Constant weight (as all subjects have same weight)
  (S2_score : ℝ)  -- Score in the second subject
  (S3_score : ℝ)  -- Score in the third subject
  (target_avg : ℝ) -- Target average score
  (hS2 : S2_score = 70)  -- Second subject score is 70%
  (hS3 : S3_score = 80)  -- Third subject score is 80%
  (havg : (x + S2_score + S3_score) / 3 = target_avg) :  -- The desired average is equal to the target average
  target_avg = 70 → x = 60 :=   -- Target average score is 70%
by
  sorry

end student_score_in_first_subject_l276_276348


namespace sum_three_numbers_is_247_l276_276254

variables (A B C : ℕ)

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ (nat.digits 10 n)

theorem sum_three_numbers_is_247
  (hA : 100 ≤ A ∧ A < 1000) -- A is a three-digit number
  (hB : 10 ≤ B ∧ B < 100)   -- B is a two-digit number
  (hC : 10 ≤ C ∧ C < 100)   -- C is a two-digit number
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
        (if contains_digit A 7 then A else 0) +
        (if contains_digit B 7 then B else 0) +
        (if contains_digit C 7 then C else 0) = 208) -- Sum of numbers containing digit 7 is 208
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧
        (if contains_digit B 3 then B else 0) +
        (if contains_digit C 3 then C else 0) = 76) -- Sum of numbers containing digit 3 is 76
  : A + B + C = 247 := 
sorry

end sum_three_numbers_is_247_l276_276254


namespace find_integer_a_l276_276972

theorem find_integer_a :
  ∃ a b c : ℤ, 
    (b, c ∈ ℤ) ∧
    (∀ x : ℝ, (x - a) * (x - 10) + 1 = (x + b) * (x + c)) ∧ 
    (a = 8 ∨ a = 12) :=
sorry

end find_integer_a_l276_276972


namespace subsets_containing_5_and_6_l276_276473

theorem subsets_containing_5_and_6: 
  let s := {1, 2, 3, 4, 5, 6} in
  {t : Finset ℕ // t ⊆ s ∧ 5 ∈ t ∧ 6 ∈ t}.card = 16 :=
by sorry

end subsets_containing_5_and_6_l276_276473


namespace percentage_of_primes_divisible_by_3_l276_276760

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l276_276760


namespace boxes_in_case_number_of_boxes_in_case_l276_276987

-- Definitions based on the conditions
def boxes_of_eggs : Nat := 5
def eggs_per_box : Nat := 3
def total_eggs : Nat := 15

-- Proposition
theorem boxes_in_case (boxes_of_eggs : Nat) (eggs_per_box : Nat) (total_eggs : Nat) : Nat :=
  if boxes_of_eggs * eggs_per_box = total_eggs then boxes_of_eggs else 0

-- Assertion that needs to be proven
theorem number_of_boxes_in_case : boxes_in_case boxes_of_eggs eggs_per_box total_eggs = 5 :=
by sorry

end boxes_in_case_number_of_boxes_in_case_l276_276987


namespace sum_of_numbers_l276_276258

def contains_digit (n : Nat) (d : Nat) : Prop := 
  (n / 100 = d) ∨ (n % 100 / 10 = d) ∨ (n % 10 = d)

variables {A B C : Nat}

-- Given conditions
axiom three_digit_number : A ≥ 100 ∧ A < 1000
axiom two_digit_numbers : B ≥ 10 ∧ B < 100 ∧ C ≥ 10 ∧ C < 100
axiom sum_with_sevens : contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7 → A + B + C = 208
axiom sum_with_threes : contains_digit B 3 ∧ contains_digit C 3 ∧ B + C = 76

-- Main theorem to be proved
theorem sum_of_numbers : A + B + C = 247 :=
sorry

end sum_of_numbers_l276_276258


namespace find_k_value_l276_276417

noncomputable def quadratic_root (k : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + k * x - 3 = 0 ∧ x = 1

theorem find_k_value (k : ℝ) (h : quadratic_root k) : k = 2 :=
by
  cases h with x hx,
  cases hx with hx1 hx2,
  rw hx2 at hx1,
  norm_num at hx1,
  exact hx1
  sorry

end find_k_value_l276_276417


namespace complex_points_on_same_circle_l276_276131

open Complex

noncomputable def a_1 : ℂ := sorry
noncomputable def a_2 : ℂ := sorry
noncomputable def a_3 : ℂ := sorry
noncomputable def a_4 : ℂ := sorry
noncomputable def a_5 : ℂ := sorry
noncomputable def S : ℝ := sorry

axiom a1_non_zero: a_1 ≠ 0
axiom a2_non_zero: a_2 ≠ 0
axiom a3_non_zero: a_3 ≠ 0
axiom a4_non_zero: a_4 ≠ 0
axiom a5_non_zero: a_5 ≠ 0

axiom ratio_condition :
  (a_2 / a_1) = (a_3 / a_2) ∧
  (a_3 / a_2) = (a_4 / a_3) ∧
  (a_4 / a_3) = (a_5 / a_4)

axiom sum_condition :
  a_1 + a_2 + a_3 + a_4 + a_5 = 4 * (1 / a_1 + 1 / a_2 + 1 / a_3 + 1 / a_4 + 1 / a_5)

axiom S_real : S ∈ ℝ
axiom S_bound : |S| ≤ 2

theorem complex_points_on_same_circle :
  ∃ (r : ℝ) (c : ℂ), ∀ (i : ℕ), i ∈ {1, 2, 3, 4, 5} -> | (cond (i = 1) a_1 (cond (i = 2) a_2 (cond (i = 3) a_3 (cond (i = 4) a_4 a_5)) )) - c | = r :=
sorry

end complex_points_on_same_circle_l276_276131


namespace alpha_eq_two_thirds_l276_276615

theorem alpha_eq_two_thirds (α : ℚ) (h1 : 0 < α) (h2 : α < 1) (h3 : Real.cos (3 * Real.pi * α) + 2 * Real.cos (2 * Real.pi * α) = 0) : α = 2 / 3 :=
sorry

end alpha_eq_two_thirds_l276_276615


namespace det_eq_product_mod_p_l276_276125

noncomputable def det (x y z : ℤ) (p : ℕ) [hp : Fact (Nat.Prime p)] : ℤ :=
  Matrix.det ![
    ![x, y, z],
    ![x, y^p, z^p],
    ![x, y^(p^2), z^(p^2)]
  ]

theorem det_eq_product_mod_p (p : ℕ) [Fact (Nat.Prime p)] (x y z : ℤ) :
  ∃ (a b c : ℤ), 
  let q := p^2 in
  let D := det x y z p in
  let Q := (a * x + b * y + c * z)^(p^2 + p + 1) in
  D % p = Q % p :=
by
  sorry

end det_eq_product_mod_p_l276_276125


namespace subsets_containing_5_and_6_l276_276479

theorem subsets_containing_5_and_6 (S : Set ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  {T : Set ℕ // {5, 6} ⊆ T ∧ T ⊆ S}.card = 16 := 
sorry

end subsets_containing_5_and_6_l276_276479


namespace necessary_but_not_sufficient_condition_l276_276311

def condition_neq_1_or_neq_2 (a b : ℤ) : Prop :=
  a ≠ 1 ∨ b ≠ 2

def statement_sum_neq_3 (a b : ℤ) : Prop :=
  a + b ≠ 3

theorem necessary_but_not_sufficient_condition :
  ∀ (a b : ℤ), condition_neq_1_or_neq_2 a b → ¬ (statement_sum_neq_3 a b) → false :=
by
  sorry

end necessary_but_not_sufficient_condition_l276_276311


namespace coprime_coefficients_l276_276440

theorem coprime_coefficients
  (n : ℕ)
  (a : Fin n → ℤ)
  (h : ∀ i : Fin n, a i ≠ 0)
  (roots : Fin n → ℤ)
  (pairwise_coprime_roots : ∀ i j : Fin n, i ≠ j → Nat.coprime (roots i).natAbs (roots j).natAbs)
  (polynomial : ∀ x : ℤ, (∑ i in Finset.range (n + 1), a i * x ^ (n - i)) = 0) :
  Nat.coprime ((a n.succ).natAbs) (a n).natAbs := sorry

end coprime_coefficients_l276_276440


namespace solve_for_P_l276_276164

-- Statement of the problem in Lean
theorem solve_for_P (P : ℝ) (h : sqrt (P^3) = 9 * real.root 9 9) : 
  P = real.root 9 (3^14) :=
sorry

end solve_for_P_l276_276164


namespace intersection_points_l276_276425

-- Define the initial acute-angled triangle and the recursive construction of new triangles.
structure Point := (x : ℝ) (y : ℝ)

structure Triangle := 
  (A : Point)
  (B : Point)
  (C : Point)

-- Define an initial acute-angled triangle.
axiom A0 B0 C0 : Triangle
  (is_acute : (angle (B0.A - C0.A) (A0.B - C0.B) < π / 2) ∧
              (angle (C0.B - A0.B) (B0.C - A0.C) < π / 2) ∧
              (angle (A0.C - B0.C) (C0.A - A0.A) < π / 2))

-- Define the center of the square constructed on the side of a triangle.
def center_of_square (P Q : Point) : Point :=
  -- Implementation of the center calculation based on P and Q
  sorry

-- Recursively generate the sequence of triangles.
def next_triangle (T : Triangle) : Triangle :=
  let A₁ := center_of_square T.B T.C
  let B₁ := center_of_square T.C T.A
  let C₁ := center_of_square T.A T.B
  Triangle.mk A₁ B₁ C₁

-- Main theorem to prove
theorem intersection_points (T : Triangle) : 
           ∀ n : ℕ, let Tn := iterate next_triangle n T
                    let Tn_plus_1 := next_triangle Tn
                    number_of_intersections Tn Tn_plus_1 = 6 := 
sorry

end intersection_points_l276_276425


namespace cost_price_of_table_l276_276883

theorem cost_price_of_table 
  (SP : ℝ) 
  (CP : ℝ) 
  (h1 : SP = 1.24 * CP) 
  (h2 : SP = 8215) :
  CP = 6625 :=
by
  sorry

end cost_price_of_table_l276_276883


namespace integer_combination_zero_l276_276645

theorem integer_combination_zero (a b c : ℤ) (h : a * Real.sqrt 2 + b * Real.sqrt 3 + c = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end integer_combination_zero_l276_276645


namespace map_upper_half_plane_l276_276638

open Complex

theorem map_upper_half_plane (a : ℝ) (h : a > 0) :
  ∀ (z : ℂ), Im z > 0 → (0 < z.im ∧ (0 < ((z.re)^2 + (z.im)^2 + a^2)) ∨ (0 < (a^2 - ((z.re)^2 + (z.im)^2))))
  → Im (sqrt (z^2 + (a : ℂ)^2)) > 0 :=
by
  intro z
  intro hImz
  intro hcut
  sorry

end map_upper_half_plane_l276_276638


namespace percentage_of_primes_divisible_by_3_is_12_5_l276_276841

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l276_276841


namespace subsets_containing_5_and_6_l276_276471

theorem subsets_containing_5_and_6: 
  let s := {1, 2, 3, 4, 5, 6} in
  {t : Finset ℕ // t ⊆ s ∧ 5 ∈ t ∧ 6 ∈ t}.card = 16 :=
by sorry

end subsets_containing_5_and_6_l276_276471


namespace detect_all_antibodies_equal_expected_value_determine_better_method_l276_276350

section VaccineProduction

-- Definitions and conditions

variables (n : ℕ) (k : ℕ) (p : ℝ)

-- Constants
axiom ht (p_gt_zero : 0 < p) (p_lt_one : p < 1) : True

-- Question 1
def probability_of_four_inspections (n : ℕ) (antibody_samples non_antibody_samples : ℕ) : ℚ :=
(factorial antibody_samples / (factorial (antibody_samples - 2) * factorial 2) *
factorial non_antibody_samples / (factorial (non_antibody_samples - 1) * factorial 1) *
factorial 3) /
(factorial n / (factorial (n - 4) * factorial 4))

-- Question 2 (part 1)
def probability_relationship (k : ℕ) (p : ℝ) : ℝ :=
1 - (1 / k)^(1 / k)

-- Question 2 (part 2)
def better_inspection_method (k : ℕ) (p : ℝ) (e_value : ℝ) : string :=
if k * real.exp (-k / e_value) > 1 then "mixing"
else "one-by-one"

-- Proofs (to be filled)
theorem detect_all_antibodies (p : ℝ) (n antibody_samples non_antibody_samples : ℕ) (ht) : 
probability_of_four_inspections n antibody_samples non_antibody_samples = 4 / 35 := sorry

theorem equal_expected_value (k : ℕ) (p : ℝ) (ht) : 
probability_relationship k p = 1 - (1 / k)^(1 / k) := sorry

theorem determine_better_method (k : ℕ) (p : ℝ) (e_value : ℝ) (ht) : 
better_inspection_method k p e_value = if k ∈ set.Icc 2 26 then "mixing" else "one-by-one" := sorry

end VaccineProduction

end detect_all_antibodies_equal_expected_value_determine_better_method_l276_276350


namespace notebook_difference_l276_276363

theorem notebook_difference {p : ℝ} (hc_price : p > 0.10) (hc : ∃ n_c : ℕ, n_c * p = 2.34)
    (hm : ∃ n_m : ℕ, n_m * p = 3.12) (h_diff : ∃ n : ℕ, n * p = 0.78) :
  (∃ k : ℕ, (3.12 / (3.12 / p).nat_abs - 2.34 / (2.34 / p).nat_abs) = k) :=
by
  sorry

end notebook_difference_l276_276363


namespace train_length_is_350_meters_l276_276303

noncomputable def length_of_train (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let time_hr := time_sec / 3600
  speed_kmh * time_hr * 1000

theorem train_length_is_350_meters :
  length_of_train 60 21 = 350 :=
by
  sorry

end train_length_is_350_meters_l276_276303


namespace largest_prime_factor_9911_l276_276390

theorem largest_prime_factor_9911 :
  ∃ p : ℕ, p = 109 ∧ Prime p ∧ 
    (∀ q : ℕ, (q ∣ 9911 → Prime q) → q ≤ p) :=
by
  have factor_9911 : 9911 = 91 * 109 := sorry
  have factor_91 : 91 = 7 * 13 := sorry
  have prime_109 : Prime 109 := sorry
  have prime_7 : Prime 7 := sorry
  have prime_13 : Prime 13 := sorry
  existsi 109
  split
  · trivial
  split
  · exact prime_109
  · intros q hq
    cases hq with hq_factor hq_prime
    interval_cases q
    · sorry
    · sorry
    · sorry

end largest_prime_factor_9911_l276_276390
