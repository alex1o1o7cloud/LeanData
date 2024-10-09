import Mathlib

namespace find_a_l338_33853

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * x^2 + 2 * a * x - Real.log x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x → x ≤ b → a ≤ y → y ≤ b → x ≤ y → f x ≤ f y

theorem find_a (a : ℝ) :
  is_increasing_on (f a) (1 / 3) 2 → a ≥ 4 / 3 :=
sorry

end find_a_l338_33853


namespace monotonic_increasing_iff_l338_33857

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + 1 / x

theorem monotonic_increasing_iff (a : ℝ) :
  (∀ x : ℝ, 1 < x → f a x ≥ f a 1) ↔ a ≥ 1 :=
by
  sorry

end monotonic_increasing_iff_l338_33857


namespace ripe_oranges_l338_33890

theorem ripe_oranges (U : ℕ) (hU : U = 25) (hR : R = U + 19) : R = 44 := by
  sorry

end ripe_oranges_l338_33890


namespace exists_distinct_numbers_satisfy_conditions_l338_33827

theorem exists_distinct_numbers_satisfy_conditions :
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a + b + c = 6) ∧
  (2 * b = a + c) ∧
  ((b^2 = a * c) ∨ (a^2 = b * c) ∨ (c^2 = a * b)) :=
by
  sorry

end exists_distinct_numbers_satisfy_conditions_l338_33827


namespace simplify_expr_correct_l338_33860

-- Define the expression
def simplify_expr (z : ℝ) : ℝ := (3 - 5 * z^2) - (5 + 7 * z^2)

-- Prove the simplified form
theorem simplify_expr_correct (z : ℝ) : simplify_expr z = -2 - 12 * z^2 := by
  sorry

end simplify_expr_correct_l338_33860


namespace break_25_ruble_bill_l338_33805

theorem break_25_ruble_bill (x y z : ℕ) :
  (x + y + z = 11 ∧ 1 * x + 3 * y + 5 * z = 25) ↔ 
    (x = 4 ∧ y = 7 ∧ z = 0) ∨ 
    (x = 5 ∧ y = 5 ∧ z = 1) ∨ 
    (x = 6 ∧ y = 3 ∧ z = 2) ∨ 
    (x = 7 ∧ y = 1 ∧ z = 3) :=
sorry

end break_25_ruble_bill_l338_33805


namespace pow_modulus_l338_33878

theorem pow_modulus : (5 ^ 2023) % 11 = 3 := by
  sorry

end pow_modulus_l338_33878


namespace lcm_12_21_30_l338_33833

theorem lcm_12_21_30 : Nat.lcm (Nat.lcm 12 21) 30 = 420 := by
  sorry

end lcm_12_21_30_l338_33833


namespace geometric_sequence_sum_t_value_l338_33800

theorem geometric_sequence_sum_t_value 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (t : ℝ)
  (h1 : ∀ n : ℕ, S_n n = 3^((n:ℝ)-1) + t)
  (h2 : a_n 1 = 3^0 + t)
  (geometric : ∀ n : ℕ, n ≥ 2 → a_n n = 2 * 3^(n-2)) :
  t = -1/3 :=
by
  sorry

end geometric_sequence_sum_t_value_l338_33800


namespace tino_jellybeans_l338_33895

theorem tino_jellybeans (Tino Lee Arnold Joshua : ℕ)
  (h1 : Tino = Lee + 24)
  (h2 : Arnold = Lee / 2)
  (h3 : Joshua = 3 * Arnold)
  (h4 : Arnold = 5) : Tino = 34 := by
sorry

end tino_jellybeans_l338_33895


namespace towel_area_decrease_l338_33894

theorem towel_area_decrease (L B : ℝ) :
  let A_original := L * B
  let L_new := 0.8 * L
  let B_new := 0.9 * B
  let A_new := L_new * B_new
  let percentage_decrease := ((A_original - A_new) / A_original) * 100
  percentage_decrease = 28 := 
by
  sorry

end towel_area_decrease_l338_33894


namespace kaleb_cherries_left_l338_33881

theorem kaleb_cherries_left (initial_cherries eaten_cherries remaining_cherries : ℕ) (h1 : initial_cherries = 67) (h2 : eaten_cherries = 25) : remaining_cherries = initial_cherries - eaten_cherries → remaining_cherries = 42 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

end kaleb_cherries_left_l338_33881


namespace sets_equality_l338_33862

variables {α : Type*} (A B C : Set α)

theorem sets_equality (h1 : A ∪ B ⊆ C) (h2 : A ∪ C ⊆ B) (h3 : B ∪ C ⊆ A) : A = B ∧ B = C :=
by
  sorry

end sets_equality_l338_33862


namespace abs_eq_2_iff_l338_33867

theorem abs_eq_2_iff (a : ℚ) : abs a = 2 ↔ a = 2 ∨ a = -2 :=
by
  sorry

end abs_eq_2_iff_l338_33867


namespace maximum_value_l338_33811

theorem maximum_value (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
    (h_eq : a^2 * (b + c - a) = b^2 * (a + c - b) ∧ b^2 * (a + c - b) = c^2 * (b + a - c)) :
    (2 * b + 3 * c) / a = 5 := 
sorry

end maximum_value_l338_33811


namespace brandon_skittles_loss_l338_33891

theorem brandon_skittles_loss (original final : ℕ) (H1 : original = 96) (H2 : final = 87) : original - final = 9 :=
by sorry

end brandon_skittles_loss_l338_33891


namespace work_days_l338_33873

theorem work_days (A B C : ℝ)
  (h1 : A + B = 1 / 20)
  (h2 : B + C = 1 / 30)
  (h3 : A + C = 1 / 30) :
  (1 / (A + B + C)) = 120 / 7 := 
by 
  sorry

end work_days_l338_33873


namespace find_Y_exists_l338_33825

variable {X : Finset ℕ} -- Consider a finite set X of natural numbers for generality
variable (S : Finset (Finset ℕ)) -- Set of all subsets of X with even number of elements
variable (f : Finset ℕ → ℝ) -- Real-valued function on subsets of X

-- Conditions
variable (hS : ∀ s ∈ S, s.card % 2 = 0) -- All elements in S have even number of elements
variable (h1 : ∃ A ∈ S, f A > 1990) -- f(A) > 1990 for some A ∈ S
variable (h2 : ∀ ⦃B C⦄, B ∈ S → C ∈ S → (Disjoint B C) → (f (B ∪ C) = f B + f C - 1990)) -- f respects the functional equation for disjoint subsets

theorem find_Y_exists :
  ∃ Y ⊆ X, (∀ D ∈ S, D ⊆ Y → f D > 1990) ∧ (∀ D ∈ S, D ⊆ (X \ Y) → f D ≤ 1990) :=
by
  sorry

end find_Y_exists_l338_33825


namespace find_a_find_min_difference_l338_33826

noncomputable def f (a x : ℝ) : ℝ := x + a * Real.log x
noncomputable def g (a b x : ℝ) : ℝ := f a x + (1 / 2) * x ^ 2 - b * x

theorem find_a (a : ℝ) (h_perpendicular : (1 : ℝ) + a = 2) : a = 1 := 
sorry

theorem find_min_difference (a b x1 x2 : ℝ) (h_b : b ≥ (7 / 2)) 
    (hx1_lt_hx2 : x1 < x2) (hx_sum : x1 + x2 = b - 1)
    (hx_prod : x1 * x2 = 1) :
    g a b x1 - g a b x2 = (15 / 8) - 2 * Real.log 2 :=
sorry

end find_a_find_min_difference_l338_33826


namespace three_digit_integer_one_more_than_multiple_l338_33855

theorem three_digit_integer_one_more_than_multiple :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n = 841 ∧ ∃ k : ℕ, n = 840 * k + 1 :=
by
  sorry

end three_digit_integer_one_more_than_multiple_l338_33855


namespace keiko_walking_speed_l338_33863

theorem keiko_walking_speed (r : ℝ) (t : ℝ) (width : ℝ) 
   (time_diff : ℝ) (h0 : width = 8) (h1 : time_diff = 48) 
   (h2 : t = (2 * (2 * (r + 8) * Real.pi) / (r + 8) + 2 * (0 * Real.pi))) 
   (h3 : 2 * (2 * r * Real.pi) / r + 2 * (0 * Real.pi) = t - time_diff) :
   t = 48 -> 
   (v : ℝ) →
   v = (16 * Real.pi) / time_diff →
   v = Real.pi / 3 :=
by
  sorry

end keiko_walking_speed_l338_33863


namespace strictly_monotone_function_l338_33851

open Function

-- Define the problem
theorem strictly_monotone_function (f : ℝ → ℝ) (F : ℝ → ℝ → ℝ)
  (hf_cont : Continuous f) (hf_nonconst : ¬ (∃ c, ∀ x, f x = c))
  (hf_eq : ∀ x y : ℝ, f (x + y) = F (f x) (f y)) :
  StrictMono f :=
sorry

end strictly_monotone_function_l338_33851


namespace geometric_series_sum_l338_33821

theorem geometric_series_sum :
  let a := 1
  let r := (1 / 4 : ℚ)
  (a / (1 - r)) = 4 / 3 :=
by
  sorry

end geometric_series_sum_l338_33821


namespace estimate_fish_number_l338_33886

noncomputable def numFishInLake (marked: ℕ) (caughtSecond: ℕ) (markedSecond: ℕ) : ℕ :=
  let totalFish := (caughtSecond * marked) / markedSecond
  totalFish

theorem estimate_fish_number (marked caughtSecond markedSecond : ℕ) :
  marked = 100 ∧ caughtSecond = 200 ∧ markedSecond = 25 → numFishInLake marked caughtSecond markedSecond = 800 :=
by
  intros h
  cases h
  sorry

end estimate_fish_number_l338_33886


namespace tom_weekly_fluid_intake_l338_33815

-- Definitions based on the conditions.
def soda_cans_per_day : ℕ := 5
def ounces_per_can : ℕ := 12
def water_ounces_per_day : ℕ := 64
def days_per_week : ℕ := 7

-- The mathematical proof problem statement.
theorem tom_weekly_fluid_intake :
  (soda_cans_per_day * ounces_per_can + water_ounces_per_day) * days_per_week = 868 := 
by
  sorry

end tom_weekly_fluid_intake_l338_33815


namespace inequality_proof_l338_33809

theorem inequality_proof (a b c d : ℝ) (h : a > 0) (h : b > 0) (h : c > 0) (h : d > 0)
  (h₁ : (a * b) / (c * d) = (a + b) / (c + d)) : (a + b) * (c + d) ≥ (a + c) * (b + d) :=
sorry

end inequality_proof_l338_33809


namespace double_average_l338_33848

theorem double_average (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (h1 : n = 25) (h2 : initial_avg = 70) (h3 : new_avg * n = 2 * (initial_avg * n)) : new_avg = 140 :=
sorry

end double_average_l338_33848


namespace trajectory_of_M_ellipse_trajectory_l338_33884

variable {x y : ℝ}

theorem trajectory_of_M (hx : x ≠ 5) (hnx : x ≠ -5)
  (h : (y / (x + 5)) * (y / (x - 5)) = -2) : 
  (2 * x^2 + y^2 = 50) :=
by
  -- Proof is omitted.
  sorry

theorem ellipse_trajectory (hx : x ≠ 5) (hnx : x ≠ -5) 
  (h : (y / (x + 5)) * (y / (x - 5)) = -2) : 
  (x^2 / 25 + y^2 / 50 = 1) :=
by
  -- Using the previous theorem to derive.
  have h1 : (2 * x^2 + y^2 = 50) := trajectory_of_M hx hnx h
  -- Proof of transformation is omitted.
  sorry

end trajectory_of_M_ellipse_trajectory_l338_33884


namespace oscar_marathon_training_l338_33806

theorem oscar_marathon_training :
  let initial_miles := 2
  let target_miles := 20
  let increment_per_week := (2 : ℝ) / 3
  ∃ weeks_required, target_miles - initial_miles = weeks_required * increment_per_week → weeks_required = 27 :=
by
  sorry

end oscar_marathon_training_l338_33806


namespace cube_point_problem_l338_33880
open Int

theorem cube_point_problem (n : ℤ) (x y z u : ℤ)
  (hx : x = 0 ∨ x = 8)
  (hy : y = 0 ∨ y = 12)
  (hz : z = 0 ∨ z = 6)
  (hu : 24 ∣ u)
  (hn : n = x + y + z + u) :
  (n ≠ 100) ∧ (n = 200) ↔ (n % 6 = 0 ∨ (n - 8) % 6 = 0) :=
by sorry

end cube_point_problem_l338_33880


namespace minimum_value_l338_33823

noncomputable def minValue (x y : ℝ) : ℝ := (2 / x) + (3 / y)

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 20) : minValue x y = 1 :=
sorry

end minimum_value_l338_33823


namespace find_four_numbers_l338_33814

theorem find_four_numbers (a b c d : ℕ) 
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7)
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) :
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := 
  by
    sorry

end find_four_numbers_l338_33814


namespace number_of_tens_in_sum_l338_33831

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end number_of_tens_in_sum_l338_33831


namespace boys_neither_happy_nor_sad_correct_l338_33899

def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def total_boys : ℕ := 16
def total_girls : ℕ := 44
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4

-- The number of boys who are neither happy nor sad
def boys_neither_happy_nor_sad : ℕ :=
  total_boys - happy_boys - (sad_children - sad_girls)

theorem boys_neither_happy_nor_sad_correct : boys_neither_happy_nor_sad = 4 := by
  sorry

end boys_neither_happy_nor_sad_correct_l338_33899


namespace total_packs_of_groceries_is_14_l338_33854

-- Define the number of packs of cookies
def packs_of_cookies : Nat := 2

-- Define the number of packs of cakes
def packs_of_cakes : Nat := 12

-- Define the total packs of groceries as the sum of packs of cookies and cakes
def total_packs_of_groceries : Nat := packs_of_cookies + packs_of_cakes

-- The theorem which states that the total packs of groceries is 14
theorem total_packs_of_groceries_is_14 : total_packs_of_groceries = 14 := by
  -- this is where the proof would go
  sorry

end total_packs_of_groceries_is_14_l338_33854


namespace probability_at_least_one_two_l338_33808

def num_dice := 2
def sides_dice := 8
def total_outcomes := sides_dice ^ num_dice
def num_non_favorable_outcomes := (sides_dice - 1) ^ num_dice
def num_favorable_outcomes := total_outcomes - num_non_favorable_outcomes
def probability_favorable_outcomes := (15 : ℚ) / (64 : ℚ)

theorem probability_at_least_one_two :
  probability_favorable_outcomes = 15 / 64 :=
sorry

end probability_at_least_one_two_l338_33808


namespace root_exists_in_interval_l338_33893

noncomputable def f (x : ℝ) := (1 / 2) ^ x - x + 1

theorem root_exists_in_interval :
  (0 < f 1) ∧ (f 1.5 < 0) ∧ (f 2 < 0) ∧ (f 3 < 0) → ∃ x, 1 < x ∧ x < 1.5 ∧ f x = 0 :=
by
  -- use the intermediate value theorem and bisection method here
  sorry

end root_exists_in_interval_l338_33893


namespace no_solution_fraction_eq_l338_33837

theorem no_solution_fraction_eq (m : ℝ) : 
  ¬(∃ x : ℝ, x ≠ -1 ∧ 3 * x / (x + 1) = m / (x + 1) + 2) ↔ m = -3 :=
by
  sorry

end no_solution_fraction_eq_l338_33837


namespace area_larger_sphere_l338_33835

variables {r1 r2 r : ℝ}
variables {A1 A2 : ℝ}

-- Declare constants for the problem
def radius_smaller_sphere : ℝ := 4 -- r1
def radius_larger_sphere : ℝ := 6  -- r2
def radius_ball : ℝ := 1           -- r
def area_smaller_sphere : ℝ := 27  -- A1

-- Given conditions
axiom radius_smaller_sphere_condition : r1 = radius_smaller_sphere
axiom radius_larger_sphere_condition : r2 = radius_larger_sphere
axiom radius_ball_condition : r = radius_ball
axiom area_smaller_sphere_condition : A1 = area_smaller_sphere

-- Statement to be proved
theorem area_larger_sphere :
  r1 = radius_smaller_sphere → r2 = radius_larger_sphere → r = radius_ball → A1 = area_smaller_sphere → A2 = 60.75 :=
by
  intros
  sorry

end area_larger_sphere_l338_33835


namespace binary_addition_to_hex_l338_33830

theorem binary_addition_to_hex :
  let n₁ := (0b11111111111 : ℕ)
  let n₂ := (0b11111111 : ℕ)
  n₁ + n₂ = 0x8FE :=
by {
  sorry
}

end binary_addition_to_hex_l338_33830


namespace sin_cos_of_theta_l338_33822

open Real

theorem sin_cos_of_theta (θ : ℝ) (P : ℝ × ℝ) (hx : P = (-3, 4))
  (hxθ : ∃ r, r > 0 ∧ P = (r * cos θ, r * sin θ)) :
  sin θ + cos θ = 1 / 5 := 
by
  sorry

end sin_cos_of_theta_l338_33822


namespace a_4_is_11_l338_33819

def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_4_is_11 : a 4 = 11 := by
  sorry

end a_4_is_11_l338_33819


namespace a_equals_1_or_2_l338_33816

def M (a : ℤ) : Set ℤ := {a, 0}
def N : Set ℤ := {x : ℤ | x^2 - 3 * x < 0}
def non_empty_intersection (a : ℤ) : Prop := (M a ∩ N).Nonempty

theorem a_equals_1_or_2 (a : ℤ) (h : non_empty_intersection a) : a = 1 ∨ a = 2 := by
  sorry

end a_equals_1_or_2_l338_33816


namespace combined_work_days_l338_33841

theorem combined_work_days (W D : ℕ) (h1: ∀ a b : ℕ, a + b = 4) (h2: (1/6:ℝ) = (1/6:ℝ)) :
  D = 4 :=
by
  sorry

end combined_work_days_l338_33841


namespace area_of_shaded_region_l338_33824

/-- A 4-inch by 4-inch square adjoins a 10-inch by 10-inch square. 
The bottom right corner of the smaller square touches the midpoint of the left side of the larger square. 
Prove that the area of the shaded region is 92/7 square inches. -/
theorem area_of_shaded_region : 
  let small_square_side := 4
  let large_square_side := 10 
  let midpoint := large_square_side / 2
  let height_from_midpoint := midpoint - small_square_side / 2
  let dg := (height_from_midpoint * small_square_side) / ((midpoint + height_from_midpoint))
  (small_square_side * small_square_side) - ((1/2) * dg * small_square_side) = 92 / 7 :=
by
  sorry

end area_of_shaded_region_l338_33824


namespace initial_blocks_l338_33801

theorem initial_blocks (used_blocks remaining_blocks : ℕ) (h1 : used_blocks = 25) (h2 : remaining_blocks = 72) : 
  used_blocks + remaining_blocks = 97 := by
  sorry

end initial_blocks_l338_33801


namespace tangent_parabola_points_l338_33866

theorem tangent_parabola_points (a b : ℝ) (h_circle : a^2 + b^2 = 1) (h_discriminant : a^2 - 4 * b * (b - 1) = 0) :
    (a = 0 ∧ b = 1) ∨ 
    (a = 2 * Real.sqrt 6 / 5 ∧ b = -1 / 5) ∨ 
    (a = -2 * Real.sqrt 6 / 5 ∧ b = -1 / 5) := sorry

end tangent_parabola_points_l338_33866


namespace quadratic_points_relationship_l338_33868

theorem quadratic_points_relationship (c y1 y2 y3 : ℝ) 
  (hA : y1 = (-3)^2 + 2*(-3) + c)
  (hB : y2 = (1/2)^2 + 2*(1/2) + c)
  (hC : y3 = 2^2 + 2*2 + c) : y2 < y1 ∧ y1 < y3 := 
sorry

end quadratic_points_relationship_l338_33868


namespace ten_times_average_letters_l338_33861

-- Define the number of letters Elida has
def letters_Elida : ℕ := 5

-- Define the number of letters Adrianna has
def letters_Adrianna : ℕ := 2 * letters_Elida - 2

-- Define the average number of letters in both names
def average_letters : ℕ := (letters_Elida + letters_Adrianna) / 2

-- Define the final statement for 10 times the average number of letters
theorem ten_times_average_letters : 10 * average_letters = 65 := by
  sorry

end ten_times_average_letters_l338_33861


namespace sufficient_not_necessary_example_l338_33865

lemma sufficient_but_not_necessary_condition (x y : ℝ) (hx : x >= 2) (hy : y >= 2) : x^2 + y^2 >= 4 :=
by
  -- We only need to state the lemma, so the proof is omitted.
  sorry

theorem sufficient_not_necessary_example :
  ¬(∀ x y : ℝ, (x^2 + y^2 >= 4) -> (x >= 2) ∧ (y >= 2)) :=
by 
  -- We only need to state the theorem, so the proof is omitted.
  sorry

end sufficient_not_necessary_example_l338_33865


namespace factor_difference_of_squares_l338_33858

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) :=
by
  sorry

end factor_difference_of_squares_l338_33858


namespace table_price_l338_33897

theorem table_price
  (C T : ℝ)
  (h1 : 2 * C + T = 0.6 * (C + 2 * T))
  (h2 : C + T = 60) :
  T = 52.5 :=
by
  sorry

end table_price_l338_33897


namespace dictionary_prices_and_max_A_l338_33859

-- Definitions for the problem
def price_A := 70
def price_B := 50

-- Conditions from the problem
def condition1 := (price_A + 2 * price_B = 170)
def condition2 := (2 * price_A + 3 * price_B = 290)

-- The proof problem statement
theorem dictionary_prices_and_max_A (h1 : price_A + 2 * price_B = 170) (h2 : 2 * price_A + 3 * price_B = 290) :
  price_A = 70 ∧ price_B = 50 ∧ (∀ (x y : ℕ), x + y = 30 → 70 * x + 50 * y ≤ 1600 → x ≤ 5) :=
by
  sorry

end dictionary_prices_and_max_A_l338_33859


namespace calculate_expression_l338_33843

theorem calculate_expression : 
  ((13^13 / 13^12)^3 * 3^3) / 3^6 = 27 :=
by
  sorry

end calculate_expression_l338_33843


namespace min_detectors_correct_l338_33864

noncomputable def min_detectors (M N : ℕ) : ℕ :=
  ⌈(M : ℝ) / 2⌉₊ + ⌈(N : ℝ) / 2⌉₊

theorem min_detectors_correct (M N : ℕ) (hM : 2 ≤ M) (hN : 2 ≤ N) :
  min_detectors M N = ⌈(M : ℝ) / 2⌉₊ + ⌈(N : ℝ) / 2⌉₊ :=
by {
  -- The proof goes here
  sorry
}

end min_detectors_correct_l338_33864


namespace distance_ratio_gt_9_l338_33888

theorem distance_ratio_gt_9 (points : Fin 1997 → ℝ × ℝ × ℝ) (M m : ℝ) :
  (∀ i j, i ≠ j → dist (points i) (points j) ≤ M) →
  (∀ i j, i ≠ j → dist (points i) (points j) ≥ m) →
  m ≠ 0 →
  M / m > 9 :=
by
  sorry

end distance_ratio_gt_9_l338_33888


namespace find_a_l338_33812

noncomputable def curve (x a : ℝ) : ℝ := 1/x + (Real.log x)/a
noncomputable def curve_derivative (x a : ℝ) : ℝ := 
  (-1/(x^2)) + (1/(a * x))

theorem find_a (a : ℝ) : 
  (curve_derivative 1 a = 3/2) ∧ ((∃ l : ℝ, curve 1 a = l) → ∃ m : ℝ, m * (-2/3) = -1)  → a = 2/5 :=
by
  sorry

end find_a_l338_33812


namespace mary_pays_fifteen_l338_33871

def apple_cost : ℕ := 1
def orange_cost : ℕ := 2
def banana_cost : ℕ := 3
def discount_per_5_fruits : ℕ := 1

def apples_bought : ℕ := 5
def oranges_bought : ℕ := 3
def bananas_bought : ℕ := 2

def total_cost_before_discount : ℕ :=
  apples_bought * apple_cost +
  oranges_bought * orange_cost +
  bananas_bought * banana_cost

def total_fruits : ℕ :=
  apples_bought + oranges_bought + bananas_bought

def total_discount : ℕ :=
  (total_fruits / 5) * discount_per_5_fruits

def final_amount_to_pay : ℕ :=
  total_cost_before_discount - total_discount

theorem mary_pays_fifteen : final_amount_to_pay = 15 := by
  sorry

end mary_pays_fifteen_l338_33871


namespace police_catches_thief_in_two_hours_l338_33852

noncomputable def time_to_catch (speed_thief speed_police distance_police_start lead_time : ℝ) : ℝ :=
  let distance_thief := speed_thief * lead_time
  let initial_distance := distance_police_start - distance_thief
  let relative_speed := speed_police - speed_thief
  initial_distance / relative_speed

theorem police_catches_thief_in_two_hours :
  time_to_catch 20 40 60 1 = 2 := by
  sorry

end police_catches_thief_in_two_hours_l338_33852


namespace distance_from_Idaho_to_Nevada_l338_33846

theorem distance_from_Idaho_to_Nevada (d1 d2 s1 s2 t total_time : ℝ) 
  (h1 : d1 = 640)
  (h2 : s1 = 80)
  (h3 : s2 = 50)
  (h4 : total_time = 19)
  (h5 : t = total_time - (d1 / s1)) :
  d2 = s2 * t :=
by
  sorry

end distance_from_Idaho_to_Nevada_l338_33846


namespace solution_set_of_inequality_l338_33820

theorem solution_set_of_inequality :
  ∀ x : ℝ, 3 * x^2 - 2 * x + 1 > 7 ↔ (x < -2/3 ∨ x > 3) :=
by
  sorry

end solution_set_of_inequality_l338_33820


namespace subletter_payment_correct_l338_33813

noncomputable def johns_monthly_rent : ℕ := 900
noncomputable def johns_yearly_rent : ℕ := johns_monthly_rent * 12
noncomputable def johns_profit_per_year : ℕ := 3600
noncomputable def total_rent_collected : ℕ := johns_yearly_rent + johns_profit_per_year
noncomputable def number_of_subletters : ℕ := 3
noncomputable def subletter_annual_payment : ℕ := total_rent_collected / number_of_subletters
noncomputable def subletter_monthly_payment : ℕ := subletter_annual_payment / 12

theorem subletter_payment_correct :
  subletter_monthly_payment = 400 :=
by
  sorry

end subletter_payment_correct_l338_33813


namespace roots_eq_solution_l338_33889

noncomputable def roots_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

noncomputable def quadratic_roots (m n : ℝ) : Prop :=
  roots_eq 1 (-2) (-2025) m ∧ roots_eq 1 (-2) (-2025) n

theorem roots_eq_solution (m n : ℝ) (hm : roots_eq 1 (-2) (-2025) m) (hn : roots_eq 1 (-2) (-2025) n) : 
  m^2 - 3 * m - n = 2023 := 
sorry

end roots_eq_solution_l338_33889


namespace expansion_gameplay_hours_l338_33804

theorem expansion_gameplay_hours :
  let total_gameplay := 100
  let boring_percentage := 80 / 100
  let enjoyable_percentage := 1 - boring_percentage
  let enjoyable_gameplay_original := enjoyable_percentage * total_gameplay
  let enjoyable_gameplay_total := 50
  let expansion_hours := enjoyable_gameplay_total - enjoyable_gameplay_original
  expansion_hours = 30 :=
by
  let total_gameplay := 100
  let boring_percentage := 80 / 100
  let enjoyable_percentage := 1 - boring_percentage
  let enjoyable_gameplay_original := enjoyable_percentage * total_gameplay
  let enjoyable_gameplay_total := 50
  let expansion_hours := enjoyable_gameplay_total - enjoyable_gameplay_original
  show expansion_hours = 30
  sorry

end expansion_gameplay_hours_l338_33804


namespace distance_between_cities_l338_33845

theorem distance_between_cities
    (v_bus : ℕ) (v_car : ℕ) (t_bus_meet : ℚ) (t_car_wait : ℚ)
    (d_overtake : ℚ) (s : ℚ)
    (h_vb : v_bus = 40)
    (h_vc : v_car = 50)
    (h_tbm : t_bus_meet = 0.25)
    (h_tcw : t_car_wait = 0.25)
    (h_do : d_overtake = 20)
    (h_eq : (s - 10) / 50 + t_car_wait = (s - 30) / 40) :
    s = 160 :=
by
    exact sorry

end distance_between_cities_l338_33845


namespace zero_sum_of_squares_eq_zero_l338_33892

theorem zero_sum_of_squares_eq_zero {a b : ℝ} (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end zero_sum_of_squares_eq_zero_l338_33892


namespace olivia_paper_count_l338_33872

-- State the problem conditions and the final proof statement.
theorem olivia_paper_count :
  let math_initial := 220
  let science_initial := 150
  let math_used := 95
  let science_used := 68
  let math_received := 30
  let science_given := 15
  let math_remaining := math_initial - math_used + math_received
  let science_remaining := science_initial - science_used - science_given
  let total_pieces := math_remaining + science_remaining
  total_pieces = 222 :=
by
  -- Placeholder for the proof
  sorry

end olivia_paper_count_l338_33872


namespace min_value_of_f_inequality_for_a_b_l338_33847

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 2)

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 3 := by
  intro x
  sorry

theorem inequality_for_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ab : 1/a + 1/b = Real.sqrt 3) : 
  1/a^2 + 2/b^2 ≥ 2 := by
  sorry

end min_value_of_f_inequality_for_a_b_l338_33847


namespace determinant_real_root_unique_l338_33896

theorem determinant_real_root_unique {a b c : ℝ} (ha : 0 < a ∧ a ≠ 1) (hb : 0 < b ∧ b ≠ 1) (hc : 0 < c ∧ c ≠ 1) :
  ∃! x : ℝ, (Matrix.det ![
    ![x - 1, c - 1, -(b - 1)],
    ![-(c - 1), x - 1, a - 1],
    ![b - 1, -(a - 1), x - 1]
  ]) = 0 :=
by
  sorry

end determinant_real_root_unique_l338_33896


namespace income_left_at_end_of_year_l338_33882

variable (I : ℝ) -- Monthly income at the beginning of the year
variable (food_expense : ℝ := 0.35 * I) 
variable (education_expense : ℝ := 0.25 * I)
variable (transportation_expense : ℝ := 0.15 * I)
variable (medical_expense : ℝ := 0.10 * I)
variable (initial_expenses : ℝ := food_expense + education_expense + transportation_expense + medical_expense)
variable (remaining_income : ℝ := I - initial_expenses)
variable (house_rent : ℝ := 0.80 * remaining_income)

variable (annual_income : ℝ := 12 * I)
variable (annual_expenses : ℝ := 12 * (initial_expenses + house_rent))

variable (increased_food_expense : ℝ := food_expense * 1.05)
variable (increased_education_expense : ℝ := education_expense * 1.05)
variable (increased_transportation_expense : ℝ := transportation_expense * 1.05)
variable (increased_medical_expense : ℝ := medical_expense * 1.05)
variable (total_increased_expenses : ℝ := increased_food_expense + increased_education_expense + increased_transportation_expense + increased_medical_expense)

variable (new_income : ℝ := 1.10 * I)
variable (new_remaining_income : ℝ := new_income - total_increased_expenses)

variable (new_house_rent : ℝ := 0.80 * new_remaining_income)

variable (final_remaining_income : ℝ := new_income - (total_increased_expenses + new_house_rent))

theorem income_left_at_end_of_year : 
  final_remaining_income / new_income * 100 = 2.15 := 
  sorry

end income_left_at_end_of_year_l338_33882


namespace a_can_be_any_sign_l338_33850

theorem a_can_be_any_sign (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h : (a / b)^2 < (c / d)^2) (hcd : c = -d) : True :=
by
  have := h
  subst hcd
  sorry

end a_can_be_any_sign_l338_33850


namespace maximum_daily_sales_l338_33817

def price (t : ℕ) : ℝ :=
if (0 < t ∧ t < 25) then t + 20
else if (25 ≤ t ∧ t ≤ 30) then -t + 100
else 0

def sales_volume (t : ℕ) : ℝ :=
if (0 < t ∧ t ≤ 30) then -t + 40
else 0

def daily_sales (t : ℕ) : ℝ :=
if (0 < t ∧ t < 25) then (t + 20) * (-t + 40)
else if (25 ≤ t ∧ t ≤ 30) then (-t + 100) * (-t + 40)
else 0

theorem maximum_daily_sales : ∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ daily_sales t = 1125 :=
sorry

end maximum_daily_sales_l338_33817


namespace problem_1_problem_2_l338_33842

theorem problem_1 :
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) = 2^32 - 1 :=
by
  sorry

theorem problem_2 :
  (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) = (3^32 - 1) / 2 :=
by
  sorry

end problem_1_problem_2_l338_33842


namespace triangle_inequality_half_perimeter_l338_33844

theorem triangle_inequality_half_perimeter 
  (a b c : ℝ)
  (h_a : a < b + c)
  (h_b : b < a + c)
  (h_c : c < a + b) :
  a < (a + b + c) / 2 ∧ b < (a + b + c) / 2 ∧ c < (a + b + c) / 2 := 
sorry

end triangle_inequality_half_perimeter_l338_33844


namespace binom_eq_fraction_l338_33885

open Nat

theorem binom_eq_fraction (n : ℕ) (h_pos : 0 < n) : choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_eq_fraction_l338_33885


namespace percentage_of_female_students_25_or_older_l338_33828

theorem percentage_of_female_students_25_or_older
  (T : ℝ) (M F : ℝ) (P : ℝ)
  (h1 : M = 0.40 * T)
  (h2 : F = 0.60 * T)
  (h3 : 0.56 = (0.20 * T) + (0.60 * (1 - P) * T)) :
  P = 0.40 :=
by
  sorry

end percentage_of_female_students_25_or_older_l338_33828


namespace area_of_feasible_region_l338_33807

theorem area_of_feasible_region :
  (∃ k m : ℝ, (∀ x y : ℝ,
    (kx - y + 1 ≥ 0 ∧ kx - my ≤ 0 ∧ y ≥ 0) ↔
    (x - y + 1 ≥ 0 ∧ x + y ≤ 0 ∧ y ≥ 0)) ∧
    k = 1 ∧ m = -1) →
  ∃ a : ℝ, a = 1 / 4 :=
by sorry

end area_of_feasible_region_l338_33807


namespace avg_stoppage_time_is_20_minutes_l338_33840

noncomputable def avg_stoppage_time : Real :=
let train1 := (60, 40) -- without stoppages, with stoppages (in kmph)
let train2 := (75, 50) -- without stoppages, with stoppages (in kmph)
let train3 := (90, 60) -- without stoppages, with stoppages (in kmph)
let time1 := (train1.1 - train1.2 : Real) / train1.1
let time2 := (train2.1 - train2.2 : Real) / train2.1
let time3 := (train3.1 - train3.2 : Real) / train3.1
let total_time := time1 + time2 + time3
(total_time / 3) * 60 -- convert hours to minutes

theorem avg_stoppage_time_is_20_minutes :
  avg_stoppage_time = 20 :=
sorry

end avg_stoppage_time_is_20_minutes_l338_33840


namespace chromosomal_variations_l338_33856

-- Define the conditions
def condition1 := "Plants grown from anther culture in vitro."
def condition2 := "Addition or deletion of DNA base pairs on chromosomes."
def condition3 := "Free combination of non-homologous chromosomes."
def condition4 := "Crossing over between non-sister chromatids in a tetrad."
def condition5 := "Cells of a patient with Down syndrome have three copies of chromosome 21."

-- Define a concept of belonging to chromosomal variations
def belongs_to_chromosomal_variations (condition: String) : Prop :=
  condition = condition1 ∨ condition = condition5

-- State the theorem
theorem chromosomal_variations :
  belongs_to_chromosomal_variations condition1 ∧ 
  belongs_to_chromosomal_variations condition5 ∧ 
  ¬ (belongs_to_chromosomal_variations condition2 ∨ 
     belongs_to_chromosomal_variations condition3 ∨ 
     belongs_to_chromosomal_variations condition4) :=
by
  sorry

end chromosomal_variations_l338_33856


namespace exists_x_for_log_eqn_l338_33810

theorem exists_x_for_log_eqn (a : ℝ) (ha : 0 < a) :
  ∃ (x : ℝ), (1 < x) ∧ (Real.log (a * x) / Real.log 10 = 2 * Real.log (x - 1) / Real.log 10) ∧ 
  x = (2 + a + Real.sqrt (a^2 + 4*a)) / 2 := sorry

end exists_x_for_log_eqn_l338_33810


namespace class_has_24_students_l338_33875

theorem class_has_24_students (n S : ℕ) 
  (h1 : (S - 91 + 19) / n = 87)
  (h2 : S / n = 90) : 
  n = 24 :=
by sorry

end class_has_24_students_l338_33875


namespace negation_proof_l338_33883

theorem negation_proof :
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 2 * x)) ↔ (∃ x : ℝ, x^2 + 1 < 2 * x) :=
by
  sorry

end negation_proof_l338_33883


namespace find_x_l338_33874

-- Define the percentages and multipliers as constants
def percent_47 := 47.0 / 100.0
def percent_36 := 36.0 / 100.0

-- Define the given quantities
def quantity1 := 1442.0
def quantity2 := 1412.0

-- Calculate the percentages of the quantities
def part1 := percent_47 * quantity1
def part2 := percent_36 * quantity2

-- Calculate the expression
def expression := (part1 - part2) + 63.0

-- Define the value of x given
def x := 232.42

-- Theorem stating the proof problem
theorem find_x : expression = x := by
  -- proof goes here
  sorry

end find_x_l338_33874


namespace find_y_l338_33869

-- Suppose C > A > B > 0
-- and A is y% smaller than C.
-- Also, C = 2B.
-- We need to show that y = 100 - 50 * (A / B).

variable (A B C : ℝ)
variable (y : ℝ)

-- Conditions
axiom h1 : C > A
axiom h2 : A > B
axiom h3 : B > 0
axiom h4 : C = 2 * B
axiom h5 : A = (1 - y / 100) * C

-- Goal
theorem find_y : y = 100 - 50 * (A / B) :=
by
  sorry

end find_y_l338_33869


namespace last_digit_of_4_over_3_power_5_l338_33829

noncomputable def last_digit_of_fraction (n d : ℕ) : ℕ :=
  (n * 10^5 / d) % 10

def four : ℕ := 4
def three_power_five : ℕ := 3^5

theorem last_digit_of_4_over_3_power_5 :
  last_digit_of_fraction four three_power_five = 7 :=
by
  sorry

end last_digit_of_4_over_3_power_5_l338_33829


namespace trip_duration_l338_33832

noncomputable def start_time : ℕ := 11 * 60 + 25 -- 11:25 a.m. in minutes
noncomputable def end_time : ℕ := 16 * 60 + 43 + 38 / 60 -- 4:43:38 p.m. in minutes

theorem trip_duration :
  end_time - start_time = 5 * 60 + 18 := 
sorry

end trip_duration_l338_33832


namespace percentage_decrease_l338_33877

theorem percentage_decrease (x y : ℝ) : 
  (xy^2 - (0.7 * x) * (0.6 * y)^2) / xy^2 = 0.748 :=
by
  sorry

end percentage_decrease_l338_33877


namespace negation_statement_l338_33870

theorem negation_statement (h : ∀ x : ℝ, |x - 2| + |x - 4| > 3) : 
  ∃ x0 : ℝ, |x0 - 2| + |x0 - 4| ≤ 3 :=
sorry

end negation_statement_l338_33870


namespace cyclists_meet_at_starting_point_l338_33803

-- Define the conditions: speeds of cyclists and the circumference of the circle
def speed_cyclist1 : ℝ := 7
def speed_cyclist2 : ℝ := 8
def circumference : ℝ := 300

-- Define the total speed by summing individual speeds
def relative_speed : ℝ := speed_cyclist1 + speed_cyclist2

-- Define the time required to meet at the starting point
def meeting_time : ℝ := 20

-- The theorem statement which states that given the conditions, the cyclists will meet after 20 seconds
theorem cyclists_meet_at_starting_point :
  meeting_time = circumference / relative_speed :=
sorry

end cyclists_meet_at_starting_point_l338_33803


namespace total_peaches_in_baskets_l338_33839

def total_peaches (red_peaches : ℕ) (green_peaches : ℕ) (baskets : ℕ) : ℕ :=
  (red_peaches + green_peaches) * baskets

theorem total_peaches_in_baskets :
  total_peaches 19 4 15 = 345 :=
by
  sorry

end total_peaches_in_baskets_l338_33839


namespace side_length_S2_l338_33834

theorem side_length_S2 (r s : ℕ) (h1 : 2 * r + s = 2260) (h2 : 2 * r + 3 * s = 3782) : s = 761 :=
by
  -- proof omitted
  sorry

end side_length_S2_l338_33834


namespace green_passes_blue_at_46_l338_33802

variable {t : ℕ}
variable {k1 k2 k3 k4 : ℝ}
variable {b1 b2 b3 b4 : ℝ}

def elevator_position (k : ℝ) (b : ℝ) (t : ℕ) : ℝ := k * t + b

axiom red_catches_blue_at_36 :
  elevator_position k1 b1 36 = elevator_position k2 b2 36

axiom red_passes_green_at_42 :
  elevator_position k1 b1 42 = elevator_position k3 b3 42

axiom red_passes_yellow_at_48 :
  elevator_position k1 b1 48 = elevator_position k4 b4 48

axiom yellow_passes_blue_at_51 :
  elevator_position k4 b4 51 = elevator_position k2 b2 51

axiom yellow_catches_green_at_54 :
  elevator_position k4 b4 54 = elevator_position k3 b3 54

theorem green_passes_blue_at_46 : 
  elevator_position k3 b3 46 = elevator_position k2 b2 46 := 
sorry

end green_passes_blue_at_46_l338_33802


namespace sampling_is_systematic_l338_33838

-- Define the total seats in each row and the total number of rows
def total_seats_per_row : ℕ := 25
def total_rows : ℕ := 30

-- Define a function to identify if the sampling is systematic
def is_systematic_sampling (sample_count : ℕ) (n : ℕ) (interval : ℕ) : Prop :=
  interval = total_seats_per_row ∧ sample_count = total_rows

-- Define the count and interval for the problem
def sample_count : ℕ := 30
def sampling_interval : ℕ := 25

-- Theorem statement: Given the conditions, it is systematic sampling
theorem sampling_is_systematic :
  is_systematic_sampling sample_count total_rows sampling_interval = true :=
sorry

end sampling_is_systematic_l338_33838


namespace simplest_form_expression_l338_33818

variable {b : ℝ}

theorem simplest_form_expression (h : b ≠ 1) :
  1 - (1 / (2 + (b / (1 - b)))) = 1 / (2 - b) :=
by
  sorry

end simplest_form_expression_l338_33818


namespace polygon_area_l338_33898

-- Define the vertices of the polygon
def x1 : ℝ := 0
def y1 : ℝ := 0

def x2 : ℝ := 4
def y2 : ℝ := 0

def x3 : ℝ := 2
def y3 : ℝ := 3

def x4 : ℝ := 4
def y4 : ℝ := 6

-- Define the expression for the Shoelace Theorem
def shoelace_area (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : ℝ :=
  0.5 * abs (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1 - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))

-- The theorem statement proving the area of the polygon
theorem polygon_area :
  shoelace_area x1 y1 x2 y2 x3 y3 x4 y4 = 6 := 
  by
  sorry

end polygon_area_l338_33898


namespace bill_can_buy_donuts_in_35_ways_l338_33887

def different_ways_to_buy_donuts : ℕ :=
  5 + 20 + 10  -- Number of ways to satisfy the conditions

theorem bill_can_buy_donuts_in_35_ways :
  different_ways_to_buy_donuts = 35 :=
by
  -- Proof steps
  -- The problem statement and the solution show the calculation to be correct.
  sorry

end bill_can_buy_donuts_in_35_ways_l338_33887


namespace employee_discount_percentage_l338_33849

theorem employee_discount_percentage:
  let purchase_price := 500
  let markup_percentage := 0.15
  let savings := 57.5
  let retail_price := purchase_price * (1 + markup_percentage)
  let discount_percentage := (savings / retail_price) * 100
  discount_percentage = 10 :=
by
  sorry

end employee_discount_percentage_l338_33849


namespace sum_primes_less_than_20_l338_33879

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l338_33879


namespace smallest_factorization_c_l338_33876

theorem smallest_factorization_c : ∃ (c : ℤ), (∀ (r s : ℤ), r * s = 2016 → r + s = c) ∧ c > 0 ∧ c = 108 :=
by 
  sorry

end smallest_factorization_c_l338_33876


namespace investment_calculation_l338_33836

theorem investment_calculation :
  ∃ (x : ℝ), x * (1.04 ^ 14) = 1000 := by
  use 571.75
  sorry

end investment_calculation_l338_33836
