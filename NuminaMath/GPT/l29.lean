import Mathlib

namespace smallest_n_l29_29874

theorem smallest_n (n : ℕ) (h₁ : ∃ k₁ : ℕ, 5 * n = k₁ ^ 2) (h₂ : ∃ k₂ : ℕ, 4 * n = k₂ ^ 3) : n = 1600 :=
sorry

end smallest_n_l29_29874


namespace line1_line2_line3_l29_29918

-- Line 1: Through (-1, 3), parallel to x - 2y + 3 = 0.
theorem line1 (x y : ℝ) : (x - 2 * y + 3 = 0) ∧ (x = -1) ∧ (y = 3) →
                              (x - 2 * y + 7 = 0) :=
by sorry

-- Line 2: Through (3, 4), perpendicular to 3x - y + 2 = 0.
theorem line2 (x y : ℝ) : (3 * x - y + 2 = 0) ∧ (x = 3) ∧ (y = 4) →
                              (x + 3 * y - 15 = 0) :=
by sorry

-- Line 3: Through (1, 2), with equal intercepts on both axes.
theorem line3 (x y : ℝ) : (x = y) ∧ (x = 1) ∧ (y = 2) →
                              (x + y - 3 = 0) :=
by sorry

end line1_line2_line3_l29_29918


namespace rectangular_field_perimeter_l29_29349

-- Definitions for conditions
def width : ℕ := 75
def length : ℕ := (7 * width) / 5
def perimeter (L W : ℕ) : ℕ := 2 * (L + W)

-- Statement to prove
theorem rectangular_field_perimeter : perimeter length width = 360 := by
  sorry

end rectangular_field_perimeter_l29_29349


namespace smallest_digit_not_in_units_place_of_odd_l29_29806

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l29_29806


namespace division_and_multiplication_l29_29123

theorem division_and_multiplication (x : ℝ) (h : x = 9) : (x / 6 * 12) = 18 := by
  sorry

end division_and_multiplication_l29_29123


namespace sufficient_not_necessary_l29_29648

namespace ProofExample

variable {x : ℝ}

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B : Set ℝ := {x | x < 2}

-- Theorem: "1 < x < 2" is a sufficient but not necessary condition for "x < 2" to hold.
theorem sufficient_not_necessary : 
  (∀ x, 1 < x ∧ x < 2 → x < 2) ∧ ¬(∀ x, x < 2 → 1 < x ∧ x < 2) := 
by
  sorry

end ProofExample

end sufficient_not_necessary_l29_29648


namespace pebbles_difference_l29_29480

theorem pebbles_difference :
  ∃ (blue yellow : ℕ), 
    (∃ total : ℕ, total = 40) ∧
    (∃ red : ℕ, red = 9) ∧
    (∃ blue : ℕ, blue = 13) ∧ 
    (∃ remaining : ℕ, remaining = total - red - blue) ∧
    (∃ groups : ℕ, groups = 3) ∧
    (∃ yellow : ℕ, yellow = remaining / groups) →
  blue - yellow = 7 :=
begin
  sorry
end

end pebbles_difference_l29_29480


namespace unique_root_of_increasing_l29_29436

variable {R : Type} [LinearOrderedField R] [DecidableEq R]

def increasing (f : R → R) : Prop :=
  ∀ x1 x2 : R, x1 < x2 → f x1 < f x2

theorem unique_root_of_increasing (f : R → R)
  (h_inc : increasing f) :
  ∃! x : R, f x = 0 :=
sorry

end unique_root_of_increasing_l29_29436


namespace functional_eq_zero_l29_29558

theorem functional_eq_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = x * f x + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end functional_eq_zero_l29_29558


namespace smallest_unfound_digit_in_odd_units_l29_29826

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l29_29826


namespace simplify_and_evaluate_expression_l29_29216

/-
Problem: Prove ( (a + 1) / (a - 1) + 1 ) / ( 2a / (a^2 - 1) ) = 2024 given a = 2023.
-/

theorem simplify_and_evaluate_expression (a : ℕ) (h : a = 2023) :
  ( (a + 1) / (a - 1) + 1 ) / ( 2 * a / (a^2 - 1) ) = 2024 :=
by
  sorry

end simplify_and_evaluate_expression_l29_29216


namespace equilateral_triangle_area_in_circle_l29_29906

theorem equilateral_triangle_area_in_circle (r : ℝ) (h : r = 9) :
  let s := 2 * r * Real.sin (π / 3)
  let A := (Real.sqrt 3 / 4) * s^2
  A = (243 * Real.sqrt 3) / 4 := by
  sorry

end equilateral_triangle_area_in_circle_l29_29906


namespace chocolate_distribution_l29_29740

theorem chocolate_distribution :
  let total_chocolate := 60 / 7
  let piles := 5
  let eaten_piles := 1
  let friends := 2
  let one_pile := total_chocolate / piles
  let remaining_chocolate := total_chocolate - eaten_piles * one_pile
  let chocolate_per_friend := remaining_chocolate / friends
  chocolate_per_friend = 24 / 7 :=
by
  sorry

end chocolate_distribution_l29_29740


namespace find_a_l29_29292

theorem find_a (a : ℤ) (A B : Set ℤ) (hA : A = {1, 3, a}) (hB : B = {1, a^2 - a + 1}) (h_subset : B ⊆ A) :
  a = -1 ∨ a = 2 := 
by
  sorry

end find_a_l29_29292


namespace smallest_digit_not_in_units_place_of_odd_l29_29814

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l29_29814


namespace number_of_new_books_l29_29394

-- Defining the given conditions
def adventure_books : ℕ := 24
def mystery_books : ℕ := 37
def used_books : ℕ := 18

-- Defining the total books and new books
def total_books : ℕ := adventure_books + mystery_books
def new_books : ℕ := total_books - used_books

-- Proving the number of new books
theorem number_of_new_books : new_books = 43 := by
  -- Here we need to show that the calculated number of new books equals 43
  sorry

end number_of_new_books_l29_29394


namespace only_valid_M_l29_29165

def digit_sum (n : ℕ) : ℕ :=
  -- definition of digit_sum as a function summing up digits of n
  sorry 

def is_valid_M (M : ℕ) := 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ M → digit_sum (M * k) = digit_sum M

theorem only_valid_M (M : ℕ) :
  is_valid_M M ↔ ∃ n : ℕ, ∀ m : ℕ, M = 10^n - 1 :=
by
  sorry

end only_valid_M_l29_29165


namespace find_digit_A_l29_29446

theorem find_digit_A :
  ∃ A : ℕ, 
    2 * 10^6 + A * 10^5 + 9 * 10^4 + 9 * 10^3 + 5 * 10^2 + 6 * 10^1 + 1 = (3 * (523 + A)) ^ 2 
    ∧ A = 4 :=
by
  sorry

end find_digit_A_l29_29446


namespace number_of_distinct_gardens_l29_29281

def is_adjacent (i1 j1 i2 j2 : ℕ) : Prop :=
  (i1 = i2 ∧ (j1 = j2 + 1 ∨ j1 + 1 = j2)) ∨ 
  (j1 = j2 ∧ (i1 = i2 + 1 ∨ i1 + 1 = i2))

def is_garden (M : ℕ → ℕ → ℕ) (m n : ℕ) : Prop :=
  ∀ i j i' j', (i < m ∧ j < n ∧ i' < m ∧ j' < n ∧ is_adjacent i j i' j') → 
    ((M i j = M i' j') ∨ (M i j = M i' j' + 1) ∨ (M i j + 1 = M i' j')) ∧
  ∀ i j, (i < m ∧ j < n ∧ 
    (∀ (i' j'), is_adjacent i j i' j' → (M i j ≤ M i' j'))) → M i j = 0

theorem number_of_distinct_gardens (m n : ℕ) : 
  ∃ (count : ℕ), count = 2 ^ (m * n) - 1 :=
sorry

end number_of_distinct_gardens_l29_29281


namespace pie_eating_fraction_l29_29512

theorem pie_eating_fraction :
  (1 / 3 + 1 / 3^2 + 1 / 3^3 + 1 / 3^4 + 1 / 3^5 + 1 / 3^6 + 1 / 3^7) = 1093 / 2187 := 
sorry

end pie_eating_fraction_l29_29512


namespace willowbrook_team_combinations_l29_29629

theorem willowbrook_team_combinations :
  let girls := 5
  let boys := 5
  let choose_three (n : ℕ) := n.choose 3
  let team_count := choose_three girls * choose_three boys
  team_count = 100 :=
by
  let girls := 5
  let boys := 5
  let choose_three (n : ℕ) := n.choose 3
  let team_count := choose_three girls * choose_three boys
  have h1 : choose_three girls = 10 := by sorry
  have h2 : choose_three boys = 10 := by sorry
  have h3 : team_count = 10 * 10 := by sorry
  exact h3

end willowbrook_team_combinations_l29_29629


namespace inequality_abc_l29_29758

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 2 * b + 3 * c) ^ 2 / (a ^ 2 + 2 * b ^ 2 + 3 * c ^ 2) ≤ 6 :=
sorry

end inequality_abc_l29_29758


namespace proof_a6_bounds_l29_29418

theorem proof_a6_bounds (a : ℝ) (h : a^5 - a^3 + a = 2) : 3 < a^6 ∧ a^6 < 4 :=
by
  sorry

end proof_a6_bounds_l29_29418


namespace intersection_A_B_l29_29577

open Set

def set_A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def set_B : Set ℤ := {x | 0 < x ∧ x < 5}

theorem intersection_A_B : set_A ∩ set_B = {1, 3} := 
by 
  sorry

end intersection_A_B_l29_29577


namespace smallest_digit_never_in_units_place_of_odd_number_l29_29789

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l29_29789


namespace smallest_digit_never_at_units_place_of_odd_l29_29845

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l29_29845


namespace steve_average_speed_l29_29762

theorem steve_average_speed 
  (Speed1 Time1 Speed2 Time2 : ℝ) 
  (cond1 : Speed1 = 40) 
  (cond2 : Time1 = 5)
  (cond3 : Speed2 = 80) 
  (cond4 : Time2 = 3) 
: 
(Speed1 * Time1 + Speed2 * Time2) / (Time1 + Time2) = 55 := 
sorry

end steve_average_speed_l29_29762


namespace triangle_area_l29_29054

noncomputable def area_ABC (AB BC : ℝ) (angle_B : ℝ) : ℝ :=
  1/2 * AB * BC * Real.sin angle_B

theorem triangle_area
  (A B C : Type)
  (AB : ℝ) (A_eq : ℝ) (B_eq : ℝ)
  (h_AB : AB = 6)
  (h_A : A_eq = Real.pi / 6)
  (h_B : B_eq = 2 * Real.pi / 3) :
  area_ABC AB AB (2 * Real.pi / 3) = 9 * Real.sqrt 3 :=
by
  simp [area_ABC, h_AB, h_A, h_B]
  sorry

end triangle_area_l29_29054


namespace interval_of_increase_l29_29391

noncomputable def f (x : ℝ) : ℝ :=
  -abs x

theorem interval_of_increase :
  ∀ x, f x ≤ f (x + 1) ↔ x ≤ 0 := by
  sorry

end interval_of_increase_l29_29391


namespace count_three_digit_multiples_of_35_l29_29444

theorem count_three_digit_multiples_of_35 : 
  ∃ n : ℕ, n = 26 ∧ ∀ x : ℕ, (100 ≤ x ∧ x < 1000) → (x % 35 = 0 → x = 35 * (3 + ((x / 35) - 3))) := 
sorry

end count_three_digit_multiples_of_35_l29_29444


namespace smallest_n_l29_29875

theorem smallest_n (n : ℕ) (hn1 : (5 * n) pow 2) (hn2 : (4 * n) pow 3) : n = 80 :=
begin
  -- sorry statement to skip the proof.
  sorry
end

end smallest_n_l29_29875


namespace flower_beds_fraction_l29_29386

open Real

noncomputable def parkArea (a b h : ℝ) := (a + b) / 2 * h
noncomputable def triangleArea (a : ℝ) := (1 / 2) * a ^ 2

theorem flower_beds_fraction 
  (a b h : ℝ) 
  (h_a: a = 15) 
  (h_b: b = 30) 
  (h_h: h = (b - a) / 2) :
  (2 * triangleArea h) / parkArea a b h = 1 / 4 := by 
  sorry

end flower_beds_fraction_l29_29386


namespace smallest_digit_never_in_units_place_of_odd_numbers_l29_29785

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l29_29785


namespace yule_log_surface_area_increase_l29_29650

noncomputable def yuleLogIncreaseSurfaceArea : ℝ := 
  let h := 10
  let d := 5
  let r := d / 2
  let n := 9
  let initialSurfaceArea := 2 * Real.pi * r * h + 2 * Real.pi * r^2
  let sliceHeight := h / n
  let sliceSurfaceArea := 2 * Real.pi * r * sliceHeight + 2 * Real.pi * r^2
  let totalSlicesSurfaceArea := n * sliceSurfaceArea
  let increaseSurfaceArea := totalSlicesSurfaceArea - initialSurfaceArea
  increaseSurfaceArea

theorem yule_log_surface_area_increase : yuleLogIncreaseSurfaceArea = 100 * Real.pi := by
  sorry

end yule_log_surface_area_increase_l29_29650


namespace binary_to_octal_conversion_l29_29680

-- Define the binary number 11010 in binary
def bin_value : ℕ := 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0

-- Define the octal value 32 in octal as decimal
def oct_value : ℕ := 3 * 8^1 + 2 * 8^0

-- The theorem to prove the binary equivalent of 11010 is the octal 32
theorem binary_to_octal_conversion : bin_value = oct_value :=
by
  -- Skip actual proof
  sorry

end binary_to_octal_conversion_l29_29680


namespace smallest_unfound_digit_in_odd_units_l29_29825

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l29_29825


namespace r_can_complete_work_in_R_days_l29_29644

theorem r_can_complete_work_in_R_days (W : ℝ) : 
  (∀ p q r P Q R : ℝ, 
    (P = W / 24) ∧
    (Q = W / 9) ∧
    (10.000000000000002 * (W / 24) + 3 * (W / 9 + W / R) = W) 
  -> R = 12) :=
by
  intros
  sorry

end r_can_complete_work_in_R_days_l29_29644


namespace stationery_sales_l29_29219

theorem stationery_sales :
  let pen_percentage : ℕ := 42
  let pencil_percentage : ℕ := 27
  let total_sales_percentage : ℕ := 100
  total_sales_percentage - (pen_percentage + pencil_percentage) = 31 :=
by
  sorry

end stationery_sales_l29_29219


namespace probability_greater_than_mean_l29_29737

noncomputable def size_distribution : MeasureTheory.MeasurableSpace ℝ := 
  MeasureTheory.Measure.Normal 22.5 0.1  

theorem probability_greater_than_mean :
  MeasureTheory.Measure.prob (size_distribution) {x | x > 22.5} = 0.5 :=
sorry

end probability_greater_than_mean_l29_29737


namespace a_3_equals_35_l29_29048

noncomputable def S (n : ℕ) : ℕ := 5 * n ^ 2 + 10 * n
noncomputable def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_3_equals_35 : a 3 = 35 := by
  sorry

end a_3_equals_35_l29_29048


namespace painters_needed_days_l29_29164

-- Let P be the total work required in painter-work-days
def total_painter_work_days : ℕ := 5

-- Let E be the effective number of workers with advanced tools
def effective_workers : ℕ := 4

-- Define the number of days, we need to prove this equals 1.25
def days_to_complete_work (P E : ℕ) : ℚ := P / E

-- The main theorem to prove: for total_painter_work_days and effective_workers, the days to complete the work is 1.25
theorem painters_needed_days :
  days_to_complete_work total_painter_work_days effective_workers = 5 / 4 :=
by
  sorry

end painters_needed_days_l29_29164


namespace probability_f_le_zero_is_1_over_7_l29_29574

noncomputable def f (x : ℝ) : ℝ := log x / log 2

def probability_f_le_zero : ℝ :=
  let a := 1 / 2
  let b := 4
  let interval_length := b - a
  let favourable_interval := 1 - 1 / 2
  favourable_interval / interval_length

theorem probability_f_le_zero_is_1_over_7 :
  probability_f_le_zero = 1 / 7 :=
  by
    sorry

end probability_f_le_zero_is_1_over_7_l29_29574


namespace murtha_total_items_at_day_10_l29_29612

-- Define terms and conditions
def num_pebbles (n : ℕ) : ℕ := n
def num_seashells (n : ℕ) : ℕ := 1 + 2 * (n - 1)

def total_pebbles (n : ℕ) : ℕ :=
  (n * (1 + n)) / 2

def total_seashells (n : ℕ) : ℕ :=
  (n * (1 + num_seashells n)) / 2

-- Define main proposition
theorem murtha_total_items_at_day_10 : total_pebbles 10 + total_seashells 10 = 155 := by
  -- Placeholder for proof
  sorry

end murtha_total_items_at_day_10_l29_29612


namespace correct_value_l29_29713

theorem correct_value (x : ℝ) (h : x + 2.95 = 9.28) : x - 2.95 = 3.38 :=
by
  sorry

end correct_value_l29_29713


namespace function_neither_even_nor_odd_l29_29963

noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (1 + x^3))

theorem function_neither_even_nor_odd :
  ¬(∀ x, f x = f (-x)) ∧ ¬(∀ x, f x = -f (-x)) := by
  sorry

end function_neither_even_nor_odd_l29_29963


namespace large_pizza_slices_l29_29615

-- Definitions and conditions based on the given problem
def slicesEatenByPhilAndre : ℕ := 9 * 2
def slicesLeft : ℕ := 2 * 2
def slicesOnSmallCheesePizza : ℕ := 8
def totalSlices : ℕ := slicesEatenByPhilAndre + slicesLeft

-- The theorem to be proven
theorem large_pizza_slices (slicesEatenByPhilAndre slicesLeft slicesOnSmallCheesePizza : ℕ) :
  slicesEatenByPhilAndre = 18 ∧ slicesLeft = 4 ∧ slicesOnSmallCheesePizza = 8 →
  totalSlices - slicesOnSmallCheesePizza = 14 :=
by
  intros h
  sorry

end large_pizza_slices_l29_29615


namespace problem1_problem2_problem3_problem4_l29_29273

-- Problem 1
theorem problem1 :
  -11 - (-8) + (-13) + 12 = -4 :=
  sorry

-- Problem 2
theorem problem2 :
  3 + 1 / 4 + (- (2 + 3 / 5)) + (5 + 3 / 4) - (8 + 2 / 5) = -2 :=
  sorry

-- Problem 3
theorem problem3 :
  -36 * (5 / 6 - 4 / 9 + 11 / 12) = -47 :=
  sorry

-- Problem 4
theorem problem4 :
  12 * (-1 / 6) + 27 / abs (3 ^ 2) + (-2) ^ 3 = -7 :=
  sorry

end problem1_problem2_problem3_problem4_l29_29273


namespace y_intercept_of_line_l29_29506

theorem y_intercept_of_line (x y : ℝ) (h : 3 * x - 2 * y + 7 = 0) (hx : x = 0) : y = 7 / 2 :=
by
  sorry

end y_intercept_of_line_l29_29506


namespace solve_container_capacity_l29_29366

noncomputable def container_capacity (C : ℝ) :=
  (0.75 * C - 0.35 * C = 48)

theorem solve_container_capacity : ∃ C : ℝ, container_capacity C ∧ C = 120 :=
by
  use 120
  constructor
  {
    -- Proof that 0.75 * 120 - 0.35 * 120 = 48
    sorry
  }
  -- Proof that C = 120
  sorry

end solve_container_capacity_l29_29366


namespace claire_balance_after_week_l29_29673

theorem claire_balance_after_week :
  ∀ (gift_card : ℝ) (latte_cost croissant_cost : ℝ) (days : ℕ) (cookie_cost : ℝ) (cookies : ℕ),
  gift_card = 100 ∧
  latte_cost = 3.75 ∧
  croissant_cost = 3.50 ∧
  days = 7 ∧
  cookie_cost = 1.25 ∧
  cookies = 5 →
  (gift_card - (days * (latte_cost + croissant_cost) + cookie_cost * cookies) = 43) :=
by
  -- Skipping proof details with sorry
  sorry

end claire_balance_after_week_l29_29673


namespace smallest_digit_not_in_odd_units_l29_29854

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l29_29854


namespace cone_volume_l29_29934

theorem cone_volume (l : ℝ) (S_side : ℝ) (h r V : ℝ)
  (hl : l = 10)
  (hS : S_side = 60 * Real.pi)
  (hr : S_side = π * r * l)
  (hh : h = Real.sqrt (l^2 - r^2))
  (hV : V = (1/3) * π * r^2 * h) :
  V = 96 * Real.pi := 
sorry

end cone_volume_l29_29934


namespace d_minus_r_eq_15_l29_29353

theorem d_minus_r_eq_15 (d r : ℤ) (h_d_gt_1 : d > 1)
  (h1 : 1059 % d = r)
  (h2 : 1417 % d = r)
  (h3 : 2312 % d = r) :
  d - r = 15 :=
sorry

end d_minus_r_eq_15_l29_29353


namespace problem_statement_l29_29623

noncomputable def sgn (x : ℝ) : ℝ :=
if x < 0 then -1 else if x = 0 then 0 else 1

noncomputable def a : ℝ :=
1 / Real.logb (1 / 4) (1 / 2015) + 1 / Real.logb (1 / 504) (1 / 2015)

def b : ℝ := 2017

theorem problem_statement :
  (a + b + (a - b) * sgn (a - b)) / 2 = 2017 :=
sorry

end problem_statement_l29_29623


namespace distance_between_trees_l29_29663

def yard_length : ℕ := 414
def number_of_trees : ℕ := 24

theorem distance_between_trees : yard_length / (number_of_trees - 1) = 18 := 
by sorry

end distance_between_trees_l29_29663


namespace gcd_of_repeated_three_digit_l29_29528

theorem gcd_of_repeated_three_digit : 
  ∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → ∀ m ∈ {k : ℕ | ∃ n, 100 ≤ n ∧ n < 1000 ∧ k = 1001 * n}, Nat.gcd 1001 m = 1001 :=
by
  sorry

end gcd_of_repeated_three_digit_l29_29528


namespace compute_expression_l29_29540

theorem compute_expression : 9 * (1 / 13) * 26 = 18 :=
by
  sorry

end compute_expression_l29_29540


namespace joey_speed_on_way_back_eq_six_l29_29642

theorem joey_speed_on_way_back_eq_six :
  ∃ (v : ℝ), 
    (∀ (d t : ℝ), 
      d = 2 ∧ t = 1 →  -- Joey runs a 2-mile distance in 1 hour
      (∀ (d_total t_avg : ℝ),
        d_total = 4 ∧ t_avg = 3 →  -- Round trip distance is 4 miles with average speed 3 mph
        (3 = 4 / (1 + 2 / v) → -- Given average speed equation
         v = 6))) := sorry

end joey_speed_on_way_back_eq_six_l29_29642


namespace solve_for_a_l29_29147

theorem solve_for_a (x a : ℝ) (hx_pos : 0 < x) (hx_sqrt1 : x = (a+1)^2) (hx_sqrt2 : x = (a-3)^2) : a = 1 :=
by
  sorry

end solve_for_a_l29_29147


namespace beakers_with_copper_l29_29654

theorem beakers_with_copper :
  ∀ (total_beakers no_copper_beakers beakers_with_copper drops_per_beaker total_drops_used : ℕ),
    total_beakers = 22 →
    no_copper_beakers = 7 →
    drops_per_beaker = 3 →
    total_drops_used = 45 →
    total_drops_used = drops_per_beaker * beakers_with_copper →
    total_beakers = beakers_with_copper + no_copper_beakers →
    beakers_with_copper = 15 := 
-- inserting the placeholder proof 'sorry'
sorry

end beakers_with_copper_l29_29654


namespace smallest_digit_never_in_units_place_of_odd_numbers_l29_29783

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l29_29783


namespace lowest_score_l29_29909

theorem lowest_score (max_mark : ℕ) (n_tests : ℕ) (avg_mark : ℕ) (h_avg : n_tests * avg_mark = 352) (h_max : ∀ k, k < n_tests → k ≤ max_mark) :
  ∃ x, (x ≤ max_mark ∧ (3 * max_mark + x) = 352) ∧ x = 52 :=
by
  sorry

end lowest_score_l29_29909


namespace count_solutions_congruence_l29_29432

theorem count_solutions_congruence : 
  ∃ (s : Finset ℕ), s.card = 4 ∧ ∀ x ∈ s, x + 20 ≡ 75 [MOD 45] ∧ x < 150 :=
sorry

end count_solutions_congruence_l29_29432


namespace max_children_arrangement_l29_29633

theorem max_children_arrangement (n : ℕ) (h1 : n = 49) 
  (h2 : ∀ i j, i ≠ j → 1 ≤ i ∧ i ≤ 49 → 1 ≤ j ∧ j ≤ 49 → (i * j < 100)) : 
  ∃ k, k = 18 :=
by
  sorry

end max_children_arrangement_l29_29633


namespace combined_average_yield_l29_29676

theorem combined_average_yield (yield_A : ℝ) (price_A : ℝ) (yield_B : ℝ) (price_B : ℝ) (yield_C : ℝ) (price_C : ℝ) :
  yield_A = 0.20 → price_A = 100 → yield_B = 0.12 → price_B = 200 → yield_C = 0.25 → price_C = 300 →
  (yield_A * price_A + yield_B * price_B + yield_C * price_C) / (price_A + price_B + price_C) = 0.1983 :=
by
  intros hYA hPA hYB hPB hYC hPC
  sorry

end combined_average_yield_l29_29676


namespace prove_road_length_l29_29254

-- Define variables for days taken by team A, B, and C
variables {a b c : ℕ}

-- Define the daily completion rates for teams A, B, and C
def rateA : ℕ := 300
def rateB : ℕ := 240
def rateC : ℕ := 180

-- Define the maximum length of the road
def max_length : ℕ := 3500

-- Define the total section of the road that team A completes in a days
def total_A (a : ℕ) : ℕ := a * rateA

-- Define the total section of the road that team B completes in b days and 18 hours
def total_B (a b : ℕ) : ℕ := 240 * (a + b) + 180

-- Define the total section of the road that team C completes in c days and 8 hours
def total_C (a b c : ℕ) : ℕ := 180 * (a + b + c) + 60

-- Define the constraint on the sum of days taken: a + b + c
def total_days (a b c : ℕ) : ℕ := a + b + c

-- The proof goal: Prove that (a * 300 == 3300) given the conditions
theorem prove_road_length :
  (total_A a = 3300) ∧ (total_B a b ≤ max_length) ∧ (total_C a b c ≤ max_length) ∧ (total_days a b c ≤ 19) :=
sorry

end prove_road_length_l29_29254


namespace possible_teams_count_l29_29215

-- Defining the problem
def team_group_division : Prop :=
  ∃ (g1 g2 g3 g4 : ℕ), (g1 ≥ 2) ∧ (g2 ≥ 2) ∧ (g3 ≥ 2) ∧ (g4 ≥ 2) ∧
  (66 = (g1 * (g1 - 1) / 2) + (g2 * (g2 - 1) / 2) + (g3 * (g3 - 1) / 2) + 
       (g4 * (g4 - 1) / 2)) ∧ 
  ((g1 + g2 + g3 + g4 = 21) ∨ (g1 + g2 + g3 + g4 = 22) ∨ 
   (g1 + g2 + g3 + g4 = 23) ∨ (g1 + g2 + g3 + g4 = 24) ∨ 
   (g1 + g2 + g3 + g4 = 25))

-- Theorem statement to prove
theorem possible_teams_count : team_group_division :=
sorry

end possible_teams_count_l29_29215


namespace volume_of_rectangular_prism_l29_29525

theorem volume_of_rectangular_prism {l w h : ℝ} 
  (h1 : l * w = 12) 
  (h2 : w * h = 18) 
  (h3 : l * h = 24) : 
  l * w * h = 72 :=
by
  sorry

end volume_of_rectangular_prism_l29_29525


namespace number_of_sets_count_number_of_sets_l29_29401

theorem number_of_sets (P : Set ℕ) :
  ({1, 2} ⊆ P ∧ P ⊆ {1, 2, 3, 4}) → (P = {1, 2} ∨ P = {1, 2, 3} ∨ P = {1, 2, 4}) :=
sorry

theorem count_number_of_sets :
  ∃ (Ps : Finset (Set ℕ)), 
  (∀ P ∈ Ps, {1, 2} ⊆ P ∧ P ⊆ {1, 2, 3, 4}) ∧ Ps.card = 3 :=
sorry

end number_of_sets_count_number_of_sets_l29_29401


namespace claire_gift_card_balance_l29_29671

/--
Claire has a $100 gift card to her favorite coffee shop.
A latte costs $3.75.
A croissant costs $3.50.
Claire buys one latte and one croissant every day for a week.
Claire buys 5 cookies, each costing $1.25.

Prove that the amount of money left on Claire's gift card after a week is $43.00.
-/
theorem claire_gift_card_balance :
  let initial_balance : ℝ := 100
  let latte_cost : ℝ := 3.75
  let croissant_cost : ℝ := 3.50
  let daily_expense : ℝ := latte_cost + croissant_cost
  let weekly_expense : ℝ := daily_expense * 7
  let cookie_cost : ℝ := 1.25
  let total_cookie_expense : ℝ := cookie_cost * 5
  let total_expense : ℝ := weekly_expense + total_cookie_expense
  let remaining_balance : ℝ := initial_balance - total_expense
  remaining_balance = 43 :=
by
  sorry

end claire_gift_card_balance_l29_29671


namespace maximum_value_of_f_on_interval_l29_29930

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 3

theorem maximum_value_of_f_on_interval :
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ 3) →
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 57 :=
by
  sorry

end maximum_value_of_f_on_interval_l29_29930


namespace smallest_digit_not_in_odd_units_l29_29836

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l29_29836


namespace coupon_value_l29_29655

theorem coupon_value (C : ℝ) (original_price : ℝ := 120) (final_price : ℝ := 99) 
(membership_discount : ℝ := 0.1) (reduced_price : ℝ := original_price - C) :
0.9 * reduced_price = final_price → C = 10 :=
by sorry

end coupon_value_l29_29655


namespace Davante_boys_count_l29_29275

def days_in_week := 7
def friends (days : Nat) := days * 2
def girls := 3
def boys (total_friends girls : Nat) := total_friends - girls

theorem Davante_boys_count :
  boys (friends days_in_week) girls = 11 :=
  by
    sorry

end Davante_boys_count_l29_29275


namespace roots_polynomial_product_l29_29974

theorem roots_polynomial_product (a b c : ℝ) (h₁ : Polynomial.eval a (Polynomial.Coeff (Polynomial.coeff 3) - 15 * Polynomial.coeff 2 + 22 * Polynomial.coeff 1 - 8 * Polynomial.X 0) = 0)
(h₂ : Polynomial.eval b (Polynomial.Coeff (Polynomial.coeff 3) - 15 * Polynomial.coeff 2 + 22 * Polynomial.coeff 1 - 8 * Polynomial.X 0) = 0)
(h₃ : Polynomial.eval c (Polynomial.Coeff (Polynomial.coeff 3) - 15 * Polynomial.coeff 2 + 22 * Polynomial.coeff 1 - 8 * Polynomial.X 0) = 0) :
(1 + a) * (1 + b) * (1 + c) = 46 :=
sorry

end roots_polynomial_product_l29_29974


namespace table_tennis_matches_l29_29962

def num_players : ℕ := 8

def total_matches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem table_tennis_matches : total_matches num_players = 28 := by
  sorry

end table_tennis_matches_l29_29962


namespace smallest_not_odd_unit_is_zero_l29_29803

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l29_29803


namespace ratio_of_height_to_radius_min_surface_area_l29_29041

theorem ratio_of_height_to_radius_min_surface_area 
  (r h : ℝ)
  (V : ℝ := 500)
  (volume_cond : π * r^2 * h = V)
  (surface_area : ℝ := 2 * π * r^2 + 2 * π * r * h) : 
  h / r = 2 :=
by
  sorry

end ratio_of_height_to_radius_min_surface_area_l29_29041


namespace fourth_equation_pattern_l29_29987

theorem fourth_equation_pattern :
  36^2 + 37^2 + 38^2 + 39^2 + 40^2 = 41^2 + 42^2 + 43^2 + 44^2 :=
by
  sorry

end fourth_equation_pattern_l29_29987


namespace officers_on_duty_l29_29090

theorem officers_on_duty
  (F : ℕ)                             -- Total female officers on the police force
  (on_duty_percentage : ℕ)            -- On duty percentage of female officers
  (H1 : on_duty_percentage = 18)      -- 18% of the female officers were on duty
  (H2 : F = 500)                      -- There were 500 female officers on the police force
  : ∃ T : ℕ, T = 2 * (on_duty_percentage * F) / 100 ∧ T = 180 :=
by
  sorry

end officers_on_duty_l29_29090


namespace alice_meets_john_time_l29_29079

-- Definitions according to conditions
def john_speed : ℝ := 4
def bob_speed : ℝ := 6
def alice_speed : ℝ := 3
def initial_distance_alice_john : ℝ := 2

-- Prove the required meeting time
theorem alice_meets_john_time : 2 / (john_speed + alice_speed) * 60 = 17 := 
by
  sorry

end alice_meets_john_time_l29_29079


namespace area_inequality_l29_29072

theorem area_inequality 
  (α β γ : ℝ) 
  (P Q S : ℝ) 
  (h1 : P / Q = α * β * γ) 
  (h2 : S = Q * (α + 1) * (β + 1) * (γ + 1)) : 
  (S ^ (1 / 3)) ≥ (P ^ (1 / 3)) + (Q ^ (1 / 3)) :=
by
  sorry

end area_inequality_l29_29072


namespace point_on_curve_l29_29894

-- Define the equation of the curve
def curve (x y : ℝ) := x^2 - x * y + 2 * y + 1 = 0

-- State that point (3, 10) satisfies the given curve equation
theorem point_on_curve : curve 3 10 :=
by
  -- this is where the proof would go but we will skip it for now
  sorry

end point_on_curve_l29_29894


namespace megan_eggs_per_meal_l29_29083

-- Define the initial conditions
def initial_eggs_from_store : Nat := 12
def initial_eggs_from_neighbor : Nat := 12
def eggs_used_for_omelet : Nat := 2
def eggs_used_for_cake : Nat := 4
def meals_to_divide : Nat := 3

-- Calculate various steps
def total_initial_eggs : Nat := initial_eggs_from_store + initial_eggs_from_neighbor
def eggs_after_cooking : Nat := total_initial_eggs - eggs_used_for_omelet - eggs_used_for_cake
def eggs_after_giving_away : Nat := eggs_after_cooking / 2
def eggs_per_meal : Nat := eggs_after_giving_away / meals_to_divide

-- State the theorem to prove the value of eggs_per_meal
theorem megan_eggs_per_meal : eggs_per_meal = 3 := by
  sorry

end megan_eggs_per_meal_l29_29083


namespace vector_addition_l29_29179

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 4)

-- State the problem as a theorem
theorem vector_addition : a + b = (-1, 5) := by
  -- the proof should go here
  sorry

end vector_addition_l29_29179


namespace smallest_digit_not_in_units_place_of_odd_numbers_l29_29822

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l29_29822


namespace cone_lateral_surface_area_l29_29719

theorem cone_lateral_surface_area (a : ℝ) (π : ℝ) (sqrt_3 : ℝ) 
  (h₁ : 0 < a)
  (h_area : (1 / 2) * a^2 * (sqrt_3 / 2) = sqrt_3) :
  π * 1 * 2 = 2 * π :=
by
  sorry

end cone_lateral_surface_area_l29_29719


namespace max_value_of_symmetric_function_l29_29729

def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) (h_sym : ∀ x : ℝ, f x a b = f (-4 - x) a b) : 
  ∃ x : ℝ, (∀ y : ℝ, f y a b ≤ f x a b) ∧ f x a b = 16 :=
by
  sorry

end max_value_of_symmetric_function_l29_29729


namespace smallest_digit_never_in_units_place_of_odd_number_l29_29787

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l29_29787


namespace A_union_B_l29_29428

noncomputable def A : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (1 - 2^x) ∧ x < 0}
noncomputable def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 2 ∧ x > 0}
noncomputable def union_set : Set ℝ := {x | x < 0 ∨ x > 0}

theorem A_union_B :
  A ∪ B = union_set :=
by
  sorry

end A_union_B_l29_29428


namespace manufacturer_cost_price_l29_29146

theorem manufacturer_cost_price
    (C : ℝ)
    (h1 : C > 0)
    (h2 : 1.18 * 1.20 * 1.25 * C = 30.09) :
    |C - 17| < 0.01 :=
by
    sorry

end manufacturer_cost_price_l29_29146


namespace solve_for_y_l29_29010

theorem solve_for_y (y : ℚ) (h : |(4 : ℚ) * y - 6| = 0) : y = 3 / 2 :=
sorry

end solve_for_y_l29_29010


namespace ratio_of_distances_l29_29698

-- Define the speeds and times for ferries P and Q
def speed_P : ℝ := 8
def time_P : ℝ := 3
def speed_Q : ℝ := speed_P + 1
def time_Q : ℝ := time_P + 5

-- Define the distances covered by ferries P and Q
def distance_P : ℝ := speed_P * time_P
def distance_Q : ℝ := speed_Q * time_Q

-- The statement to prove: the ratio of the distances
theorem ratio_of_distances : distance_Q / distance_P = 3 :=
sorry

end ratio_of_distances_l29_29698


namespace remaining_pencils_l29_29459

theorem remaining_pencils (j_pencils : ℝ) (v_pencils : ℝ)
  (j_initial : j_pencils = 300) 
  (j_donated_pct : ℝ := 0.30)
  (v_initial : v_pencils = 2 * 300) 
  (v_donated_pct : ℝ := 0.75) :
  (j_pencils - j_donated_pct * j_pencils) + (v_pencils - v_donated_pct * v_pencils) = 360 :=
by
  sorry

end remaining_pencils_l29_29459


namespace proper_subset_count_l29_29492

open Finset

noncomputable def setA : Finset ℕ := {1, 2, 3}

theorem proper_subset_count (A : Finset ℕ) (h : A = setA) : ((A.powerset.filter (λ s, s ≠ A)).card = 7) :=
by
  rw h
  sorry

end proper_subset_count_l29_29492


namespace max_value_of_symmetric_function_l29_29726

def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function 
  (a b : ℝ)
  (symmetric : ∀ t : ℝ, f (-2 + t) a b = f (-2 - t) a b) :
  ∃ M : ℝ, M = 16 ∧ ∀ x : ℝ, f x a b ≤ M :=
by
  use 16
  sorry

end max_value_of_symmetric_function_l29_29726


namespace sum_of_interior_angles_quadrilateral_l29_29774

-- Define the function for the sum of the interior angles
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

-- Theorem that the sum of the interior angles of a quadrilateral is 360 degrees
theorem sum_of_interior_angles_quadrilateral : sum_of_interior_angles 4 = 360 :=
by
  sorry

end sum_of_interior_angles_quadrilateral_l29_29774


namespace ascending_order_l29_29266

theorem ascending_order : (3 / 8 : ℝ) < 0.75 ∧ 
                          0.75 < (1 + 2 / 5 : ℝ) ∧ 
                          (1 + 2 / 5 : ℝ) < 1.43 ∧
                          1.43 < (13 / 8 : ℝ) :=
by
  sorry

end ascending_order_l29_29266


namespace courtyard_brick_problem_l29_29513

noncomputable def area_courtyard (length width : ℝ) : ℝ :=
  length * width

noncomputable def area_brick (length width : ℝ) : ℝ :=
  length * width

noncomputable def total_bricks_required (court_area brick_area : ℝ) : ℝ :=
  court_area / brick_area

theorem courtyard_brick_problem 
  (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_length : ℝ) (brick_width : ℝ)
  (H1 : courtyard_length = 18)
  (H2 : courtyard_width = 12)
  (H3 : brick_length = 15 / 100)
  (H4 : brick_width = 13 / 100) :
  
  total_bricks_required (area_courtyard courtyard_length courtyard_width * 10000) 
                        (area_brick brick_length brick_width) 
  = 11077 :=
by
  sorry

end courtyard_brick_problem_l29_29513


namespace range_contains_pi_div_4_l29_29028

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_contains_pi_div_4 : ∃ x : ℝ, f x = (Real.pi / 4) := by
  sorry

end range_contains_pi_div_4_l29_29028


namespace smallest_missing_digit_l29_29864

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l29_29864


namespace eggs_per_meal_l29_29084

noncomputable def initial_eggs_from_store : ℕ := 12
noncomputable def additional_eggs_from_neighbor : ℕ := 12
noncomputable def eggs_used_for_cooking : ℕ := 2 + 4
noncomputable def remaining_eggs_after_cooking : ℕ := initial_eggs_from_store + additional_eggs_from_neighbor - eggs_used_for_cooking
noncomputable def eggs_given_to_aunt : ℕ := remaining_eggs_after_cooking / 2
noncomputable def remaining_eggs_after_giving_to_aunt : ℕ := remaining_eggs_after_cooking - eggs_given_to_aunt
noncomputable def planned_meals : ℕ := 3

theorem eggs_per_meal : remaining_eggs_after_giving_to_aunt / planned_meals = 3 := 
by 
  sorry

end eggs_per_meal_l29_29084


namespace chickens_after_years_l29_29609

theorem chickens_after_years : 
  ∀ (initial_chickens annual_increase years : ℕ),
  initial_chickens = 550 →
  annual_increase = 150 →
  years = 9 →
  initial_chickens + (annual_increase * years) = 1900 :=
by
  intros initial_chickens annual_increase years h1 h2 h3
  rw [h1, h2, h3]
  rfl

end chickens_after_years_l29_29609


namespace geometric_seq_sum_S40_l29_29772

noncomputable def geometric_seq_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q ≠ 1 then a1 * (1 - q^n) / (1 - q) else a1 * n

theorem geometric_seq_sum_S40 :
  ∃ (a1 q : ℝ), (0 < q ∧ q ≠ 1) ∧ 
                geometric_seq_sum a1 q 10 = 10 ∧
                geometric_seq_sum a1 q 30 = 70 ∧
                geometric_seq_sum a1 q 40 = 150 :=
by
  sorry

end geometric_seq_sum_S40_l29_29772


namespace gcd_of_repeated_six_digit_integers_l29_29530

-- Given condition
def is_repeated_six_digit_integer (n : ℕ) : Prop :=
  100 ≤ n / 1000 ∧ n / 1000 < 1000 ∧ n = 1001 * (n / 1000)

-- Theorem to prove
theorem gcd_of_repeated_six_digit_integers :
  ∀ n : ℕ, is_repeated_six_digit_integer n → gcd n 1001 = 1001 :=
by sorry

end gcd_of_repeated_six_digit_integers_l29_29530


namespace frank_fence_l29_29367

theorem frank_fence (L W F : ℝ) (hL : L = 40) (hA : 320 = L * W) : F = 2 * W + L → F = 56 := by
  sorry

end frank_fence_l29_29367


namespace simplify_and_evaluate_expression_l29_29337

theorem simplify_and_evaluate_expression : 
  ∀ a : ℚ, a = -1/2 → (a + 3)^2 - (a + 1) * (a - 1) - 2 * (2 * a + 4) = 1 := 
by
  intro a ha
  simp only [ha]
  sorry

end simplify_and_evaluate_expression_l29_29337


namespace original_cost_price_l29_29244

theorem original_cost_price 
  (C SP SP_new C_new : ℝ)
  (h1 : SP = 1.05 * C)
  (h2 : C_new = 0.95 * C)
  (h3 : SP_new = SP - 8)
  (h4 : SP_new = 1.045 * C_new) :
  C = 1600 :=
by
  sorry

end original_cost_price_l29_29244


namespace sibling_of_5_over_7_child_unique_parent_one_over_2008_descendent_of_one_l29_29371

-- Part (a):
def sibling_of_frac (x : ℚ) : Prop :=
  x = 5/7

theorem sibling_of_5_over_7 : ∃ (y : ℚ), sibling_of_frac (y / (y + 1)) ∧ y + 1 = 7/2 :=
  sorry

-- Part (b):
def child (x y : ℚ) : Prop :=
  y = x + 1 ∨ y = x / (x + 1)

theorem child_unique_parent (x y z : ℚ) (hx : 0 < x) (hz : 0 < z) (hyx : child x y) (hyz : child z y) : x = z :=
  sorry

-- Part (c):
def descendent (x y : ℚ) : Prop :=
  ∃ n : ℕ, y = 1 / (x + n)

theorem one_over_2008_descendent_of_one : descendent 1 (1 / 2008) :=
  sorry

end sibling_of_5_over_7_child_unique_parent_one_over_2008_descendent_of_one_l29_29371


namespace pencils_total_l29_29456

def pencils_remaining (Jeff_pencils_initial : ℕ) (Jeff_donation_percent : ℕ) 
                      (Vicki_factor : ℕ) (Vicki_donation_fraction_num : ℕ) 
                      (Vicki_donation_fraction_den : ℕ) : ℕ :=
  let Jeff_donated := Jeff_pencils_initial * Jeff_donation_percent / 100
  let Jeff_remaining := Jeff_pencils_initial - Jeff_donated
  let Vicki_pencils_initial := Vicki_factor * Jeff_pencils_initial
  let Vicki_donated := Vicki_pencils_initial * Vicki_donation_fraction_num / Vicki_donation_fraction_den
  let Vicki_remaining := Vicki_pencils_initial - Vicki_donated
  Jeff_remaining + Vicki_remaining

theorem pencils_total :
  pencils_remaining 300 30 2 3 4 = 360 :=
by
  -- The proof should be inserted here
  sorry

end pencils_total_l29_29456


namespace fraction_transform_l29_29062

theorem fraction_transform (x : ℝ) (h : (1/3) * x = 12) : (1/4) * x = 9 :=
by 
  sorry

end fraction_transform_l29_29062


namespace randi_more_nickels_l29_29755

noncomputable def more_nickels (total_cents : ℕ) (to_peter_cents : ℕ) (to_randi_cents : ℕ) : ℕ := 
  (to_randi_cents / 5) - (to_peter_cents / 5)

theorem randi_more_nickels :
  ∀ (total_cents to_peter_cents : ℕ),
  total_cents = 175 →
  to_peter_cents = 30 →
  more_nickels total_cents to_peter_cents (2 * to_peter_cents) = 6 :=
by
  intros total_cents to_peter_cents h_total h_peter
  rw [h_total, h_peter]
  unfold more_nickels
  norm_num
  sorry

end randi_more_nickels_l29_29755


namespace evaluate_magnitude_product_l29_29548

-- Definitions of complex numbers
def z1 := Complex.mk 7 (-4)
def z2 := Complex.mk 3 11

-- The magnitude of z1
def magnitude_z1 := Complex.abs z1

-- The magnitude of z2
def magnitude_z2 := Complex.abs z2

-- Lean 4 statement expressing the problem and its final answer
theorem evaluate_magnitude_product : Complex.abs (z1 * z2) = Real.sqrt 8450 := by
  sorry

end evaluate_magnitude_product_l29_29548


namespace find_max_min_find_angle_C_l29_29933

open Real

noncomputable def f (x : ℝ) : ℝ :=
  12 * sin (x + π / 6) * cos x - 3

theorem find_max_min (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 4) :
  let fx := f x 
  (∀ a, a = abs (fx - 6)) -> (∀ b, b = abs (fx - 3)) -> fx = 6 ∨ fx = 3 := sorry

theorem find_angle_C (AC BC CD : ℝ) (hAC : AC = 6) (hBC : BC = 3) (hCD : CD = 2 * sqrt 2) :
  ∃ C : ℝ, C = π / 2 := sorry

end find_max_min_find_angle_C_l29_29933


namespace ordered_pairs_count_l29_29941

open Real

theorem ordered_pairs_count :
  ∃ x : ℕ, x = 597 ∧ ∀ (a : ℝ) (b : ℕ), (0 < a) → (2 ≤ b ∧ b ≤ 200) →
    (log b a ^ 2017 = log b (a ^ 2017) → True) :=
begin
  sorry
end

end ordered_pairs_count_l29_29941


namespace determine_Q_l29_29442

def P : Set ℕ := {1, 2}

def Q : Set ℕ := {z | ∃ x y, x ∈ P ∧ y ∈ P ∧ z = x + y}

theorem determine_Q : Q = {2, 3, 4} :=
by
  sorry

end determine_Q_l29_29442


namespace find_principal_l29_29660

noncomputable def principal_amount (A : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  A / Real.exp (r * t)

theorem find_principal : 
  principal_amount 5673981 0.1125 7.5 ≈ 2438971.57 :=
by 
  sorry

end find_principal_l29_29660


namespace prime_square_mod_six_l29_29063

theorem prime_square_mod_six (p : ℕ) (hp : Nat.Prime p) (h : p > 5) : p^2 % 6 = 1 :=
by
  sorry

end prime_square_mod_six_l29_29063


namespace calculate_f_sum_l29_29923

noncomputable def f (n : ℕ) := Real.log (3 * n^2) / Real.log 3003

theorem calculate_f_sum :
  f 7 + f 11 + f 13 = 2 :=
by
  sorry

end calculate_f_sum_l29_29923


namespace student_selection_l29_29067

theorem student_selection : 
  let first_year := 4
  let second_year := 5
  let third_year := 4
  (first_year * second_year) + (first_year * third_year) + (second_year * third_year) = 56 := by
  let first_year := 4
  let second_year := 5
  let third_year := 4
  sorry

end student_selection_l29_29067


namespace price_per_pot_l29_29669

-- Definitions based on conditions
def total_pots : ℕ := 80
def proportion_not_cracked : ℚ := 3 / 5
def total_revenue : ℚ := 1920

-- The Lean statement to prove she sold each clay pot for $40
theorem price_per_pot : (total_revenue / (total_pots * proportion_not_cracked)) = 40 := 
by sorry

end price_per_pot_l29_29669


namespace amy_local_calls_l29_29354

-- Define the conditions as hypotheses
variable (L I : ℕ)
variable (h1 : L = (5 / 2 : ℚ) * I)
variable (h2 : L = (5 / 3 : ℚ) * (I + 3))

-- Statement of the theorem
theorem amy_local_calls : L = 15 := by
  sorry

end amy_local_calls_l29_29354


namespace remainder_when_divided_by_6_l29_29368

theorem remainder_when_divided_by_6 (n : ℤ) (h_pos : 0 < n) (h_mod12 : n % 12 = 8) : n % 6 = 2 :=
sorry

end remainder_when_divided_by_6_l29_29368


namespace find_third_number_l29_29585

theorem find_third_number (x : ℝ) (third_number : ℝ) : 
  0.6 / 0.96 = third_number / 8 → x = 0.96 → third_number = 5 :=
by
  intro h1 h2
  sorry

end find_third_number_l29_29585


namespace binomial_coeff_sum_l29_29899

theorem binomial_coeff_sum : 
  (Nat.choose 3 2) + (Nat.choose 4 2) + (Nat.choose 5 2) + (Nat.choose 6 2) + (Nat.choose 7 2) + (Nat.choose 8 2) = 83 := by
  sorry

end binomial_coeff_sum_l29_29899


namespace lockers_count_l29_29226

theorem lockers_count 
(TotalCost : ℝ) 
(first_cents : ℝ) 
(additional_cents : ℝ) 
(locker_start : ℕ) 
(locker_end : ℕ) : 
  TotalCost = 155.94 
  → first_cents = 0 
  → additional_cents = 0.03 
  → locker_start = 2 
  → locker_end = 1825 := 
by
  -- Declare the number of lockers as a variable and use it to construct the proof
  let num_lockers := locker_end - locker_start + 1
  -- The cost for labeling can be calculated and matched with TotalCost
  sorry

end lockers_count_l29_29226


namespace fraction_sum_le_41_over_42_l29_29516

theorem fraction_sum_le_41_over_42 (a b c : ℕ) (h : 1/a + 1/b + 1/c < 1) : 1/a + 1/b + 1/c ≤ 41/42 :=
sorry

end fraction_sum_le_41_over_42_l29_29516


namespace smallest_digit_not_in_units_place_of_odd_l29_29815

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l29_29815


namespace integer_part_mod_8_l29_29053

theorem integer_part_mod_8 (n : ℕ) (h : n ≥ 2009) :
  ∃ x : ℝ, x = (3 + Real.sqrt 8)^(2 * n) ∧ Int.floor (x) % 8 = 1 := 
sorry

end integer_part_mod_8_l29_29053


namespace arithmetic_sequence_problem_l29_29959

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n m : ℕ, a (n+1) = a n + d

theorem arithmetic_sequence_problem
  (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 2 + a 3 = 32)
  (h2 : a 11 + a 12 + a 13 = 118) :
  a 4 + a 10 = 50 :=
sorry

end arithmetic_sequence_problem_l29_29959


namespace quadratic_inequality_range_l29_29696

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2 : ℝ) 2 :=
sorry

end quadratic_inequality_range_l29_29696


namespace proof_problem_l29_29668

-- Define the conditions
def a : ℤ := -3
def b : ℤ := -4
def cond1 := a^4 = 81
def cond2 := b^3 = -64

-- Define the goal in terms of the conditions
theorem proof_problem : a^4 + b^3 = 17 :=
by
  have h1 : a^4 = 81 := sorry
  have h2 : b^3 = -64 := sorry
  rw [h1, h2]
  norm_num

end proof_problem_l29_29668


namespace ceil_sqrt_225_eq_15_l29_29025

theorem ceil_sqrt_225_eq_15 : ⌈ Real.sqrt 225 ⌉ = 15 :=
by
  sorry

end ceil_sqrt_225_eq_15_l29_29025


namespace avg_temp_Brookdale_l29_29534

noncomputable def avg_temp (temps : List ℚ) : ℚ :=
  temps.sum / temps.length

theorem avg_temp_Brookdale : avg_temp [51, 67, 64, 61, 50, 65, 47] = 57.9 :=
by
  sorry

end avg_temp_Brookdale_l29_29534


namespace quadratic_has_negative_root_iff_l29_29491

theorem quadratic_has_negative_root_iff (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 4 * x + 1 = 0) ↔ a ≤ 4 :=
by
  sorry

end quadratic_has_negative_root_iff_l29_29491


namespace smallest_positive_angle_l29_29304

theorem smallest_positive_angle (α : ℝ) (h : α = 2012) : ∃ β : ℝ, 0 < β ∧ β < 360 ∧ β = α % 360 := by
  sorry

end smallest_positive_angle_l29_29304


namespace find_solution_pairs_l29_29284

theorem find_solution_pairs (m n : ℕ) (t : ℕ) (ht : t > 0) (hcond : 2 ≤ m ∧ 2 ≤ n ∧ n ∣ (1 + m^(3^n) + m^(2 * 3^n))) : 
  ∃ t : ℕ, t > 0 ∧ m = 3 * t - 2 ∧ n = 3 :=
by sorry

end find_solution_pairs_l29_29284


namespace smallest_digit_not_in_units_place_of_odd_l29_29816

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l29_29816


namespace path_count_l29_29142

def is_valid_step (p q : ℤ × ℤ) : Prop :=
  (p.1 = q.1 ∧ abs (p.2 - q.2) = 1) ∨ (p.2 = q.2 ∧ abs (p.1 - q.1) = 1)

def in_boundary (p : ℤ × ℤ) : Prop :=
  ¬(-3 ≤ p.1 ∧ p.1 ≤ 3 ∧ -3 ≤ p.2 ∧ p.2 ≤ 3)

def valid_path (path : List (ℤ × ℤ)) : Prop :=
  path.length = 21 ∧
  (path.head = some (-5, -5)) ∧
  (path.last = some (5, 5)) ∧
  (∀ (i : ℕ), i < 20 → is_valid_step (path.nth_le i sorry) (path.nth_le (i + 1) sorry)) ∧
  (∀ (p : ℤ × ℤ), p ∈ path → in_boundary p)

theorem path_count : ∀ (paths : Finset (List (ℤ × ℤ))), paths.card = 4252 :=
sorry

end path_count_l29_29142


namespace sheets_of_paper_per_week_l29_29282

theorem sheets_of_paper_per_week
  (sheets_per_class_per_day : ℕ)
  (num_classes : ℕ)
  (school_days_per_week : ℕ)
  (total_sheets_per_week : ℕ) 
  (h1 : sheets_per_class_per_day = 200)
  (h2 : num_classes = 9)
  (h3 : school_days_per_week = 5)
  (h4 : total_sheets_per_week = sheets_per_class_per_day * num_classes * school_days_per_week) :
  total_sheets_per_week = 9000 :=
sorry

end sheets_of_paper_per_week_l29_29282


namespace smallest_positive_integer_n_l29_29877

theorem smallest_positive_integer_n (n : ℕ) 
  (h1 : ∃ k : ℕ, n = 5 * k ∧ perfect_square(5 * k)) 
  (h2 : ∃ m : ℕ, n = 4 * m ∧ perfect_cube(4 * m)) : 
  n = 625000 :=
sorry

end smallest_positive_integer_n_l29_29877


namespace smallest_digit_never_in_units_place_of_odd_number_l29_29868

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l29_29868


namespace eq_a_sub_b_l29_29303

theorem eq_a_sub_b (a b : ℝ) (i : ℂ) (hi : i * i = -1) (h1 : (a + 4 * i) * i = b + i) : a - b = 5 :=
by
  have := hi
  have := h1
  sorry

end eq_a_sub_b_l29_29303


namespace find_n_l29_29731

def P_X_eq_2 (n : ℕ) : Prop :=
  (3 * n) / ((n + 3) * (n + 2)) = (7 : ℚ) / 30

theorem find_n (n : ℕ) (h : P_X_eq_2 n) : n = 7 :=
by sorry

end find_n_l29_29731


namespace value_of_n_l29_29709

theorem value_of_n (n : ℝ) : (∀ (x y : ℝ), x^2 + y^2 - 2 * n * x + 2 * n * y + 2 * n^2 - 8 = 0 → (x + 1)^2 + (y - 1)^2 = 2) → n = 1 :=
by
  sorry

end value_of_n_l29_29709


namespace fraction_august_tips_l29_29643

variable (A : ℝ) -- Define the average monthly tips A for March, April, May, June, July, and September
variable (august_tips : ℝ) -- Define the tips for August
variable (total_tips : ℝ) -- Define the total tips for all months

-- Define the conditions
def condition_average_tips : Prop := total_tips = 12 * A
def condition_august_tips : Prop := august_tips = 6 * A

-- The theorem we need to prove
theorem fraction_august_tips :
  condition_average_tips A total_tips →
  condition_august_tips A august_tips →
  (august_tips / total_tips) = (1 / 2) :=
by
  intros h_avg h_aug
  rw [condition_average_tips] at h_avg
  rw [condition_august_tips] at h_aug
  rw [h_avg, h_aug]
  simp
  sorry

end fraction_august_tips_l29_29643


namespace difference_blue_yellow_l29_29481

def total_pebbles : ℕ := 40
def red_pebbles : ℕ := 9
def blue_pebbles : ℕ := 13
def remaining_pebbles : ℕ := total_pebbles - red_pebbles - blue_pebbles
def groups : ℕ := 3
def pebbles_per_group : ℕ := remaining_pebbles / groups
def yellow_pebbles : ℕ := pebbles_per_group

theorem difference_blue_yellow : blue_pebbles - yellow_pebbles = 7 :=
by
  unfold blue_pebbles yellow_pebbles pebbles_per_group remaining_pebbles total_pebbles red_pebbles
  sorry

end difference_blue_yellow_l29_29481


namespace exist_rectangle_same_color_l29_29493

-- Define the colors.
inductive Color
| red
| green
| blue

open Color

-- Define the point and the plane.
structure Point :=
(x : ℝ) (y : ℝ)

-- Assume a coloring function that assigns colors to points on the plane.
def coloring : Point → Color := sorry

-- The theorem stating the existence of a rectangle with vertices of the same color.
theorem exist_rectangle_same_color :
  ∃ (A B C D : Point), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  coloring A = coloring B ∧ coloring B = coloring C ∧ coloring C = coloring D :=
sorry

end exist_rectangle_same_color_l29_29493


namespace domain_f_l29_29621

open Real

def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2) - 3

theorem domain_f :
  {x : ℝ | g x > 0} = {x : ℝ | x < 0 ∨ x > 3} :=
by 
  sorry

end domain_f_l29_29621


namespace evaluate_f_at_points_l29_29002

def f (x : ℝ) : ℝ :=
  3 * x ^ 2 - 6 * x + 10

theorem evaluate_f_at_points : 3 * f 2 + 2 * f (-2) = 98 :=
by
  sorry

end evaluate_f_at_points_l29_29002


namespace grid_midpoint_exists_l29_29093

theorem grid_midpoint_exists (points : Fin 5 → ℤ × ℤ) :
  ∃ i j : Fin 5, i ≠ j ∧ (points i).fst % 2 = (points j).fst % 2 ∧ (points i).snd % 2 = (points j).snd % 2 :=
by 
  sorry

end grid_midpoint_exists_l29_29093


namespace area_of_parallelogram_l29_29496

theorem area_of_parallelogram (base : ℝ) (height : ℝ)
  (h1 : base = 3.6)
  (h2 : height = 2.5 * base) :
  base * height = 32.4 :=
by
  sorry

end area_of_parallelogram_l29_29496


namespace equilateral_triangle_side_length_l29_29500
noncomputable def equilateral_triangle_side (r R : ℝ) (h : R > r) : ℝ :=
  r * R * Real.sqrt 3 / (Real.sqrt (r ^ 2 - r * R + R ^ 2))

theorem equilateral_triangle_side_length
  (r R : ℝ) (hRgr : R > r) :
  ∃ a, a = equilateral_triangle_side r R hRgr :=
sorry

end equilateral_triangle_side_length_l29_29500


namespace ceil_sqrt_225_eq_15_l29_29015

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := 
by 
  sorry

end ceil_sqrt_225_eq_15_l29_29015


namespace yard_length_l29_29314

theorem yard_length (trees : ℕ) (distance_per_gap : ℕ) (gaps : ℕ) :
  trees = 26 → distance_per_gap = 16 → gaps = trees - 1 → length_of_yard = gaps * distance_per_gap → length_of_yard = 400 :=
by 
  intros h_trees h_distance_per_gap h_gaps h_length_of_yard
  sorry

end yard_length_l29_29314


namespace star_equiv_zero_l29_29910

-- Define the new operation for real numbers a and b
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

-- Prove that (x^2 - y^2) star (y^2 - x^2) equals 0
theorem star_equiv_zero (x y : ℝ) : star (x^2 - y^2) (y^2 - x^2) = 0 := 
by sorry

end star_equiv_zero_l29_29910


namespace center_of_circle_eq_l29_29935

theorem center_of_circle_eq {x y : ℝ} : (x - 2)^2 + (y - 3)^2 = 1 → (x, y) = (2, 3) :=
by
  intro h
  sorry

end center_of_circle_eq_l29_29935


namespace ratio_of_socks_l29_29076

theorem ratio_of_socks (y p : ℝ) (h1 : 5 * p + y * 2 * p = 5 * p + 4 * y * p / 3) :
  (5 : ℝ) / y = 11 / 2 :=
by
  sorry

end ratio_of_socks_l29_29076


namespace remaining_pencils_check_l29_29461

variables (Jeff_initial : ℕ) (Jeff_donation_percentage : ℚ) (Vicki_ratio : ℚ) (Vicki_donation_fraction : ℚ)

def Jeff_donated_pencils := (Jeff_donation_percentage * Jeff_initial).toNat
def Jeff_remaining_pencils := Jeff_initial - Jeff_donated_pencils

def Vicki_initial_pencils := (Vicki_ratio * Jeff_initial).toNat
def Vicki_donated_pencils := (Vicki_donation_fraction * Vicki_initial_pencils).toNat
def Vicki_remaining_pencils := Vicki_initial_pencils - Vicki_donated_pencils

def total_remaining_pencils := Jeff_remaining_pencils + Vicki_remaining_pencils

theorem remaining_pencils_check
    (Jeff_initial : ℕ := 300)
    (Jeff_donation_percentage : ℚ := 0.3)
    (Vicki_ratio : ℚ := 2)
    (Vicki_donation_fraction : ℚ := 0.75) :
    total_remaining_pencils Jeff_initial Jeff_donation_percentage Vicki_ratio Vicki_donation_fraction = 360 :=
by
  sorry

end remaining_pencils_check_l29_29461


namespace hattie_jumps_l29_29583

theorem hattie_jumps (H : ℝ) (h1 : Lorelei_jumps1 = (3/4) * H)
  (h2 : Hattie_jumps2 = (2/3) * H)
  (h3 : Lorelei_jumps2 = (2/3) * H + 50)
  (h4 : H + Lorelei_jumps1 + Hattie_jumps2 + Lorelei_jumps2 = 605) : H = 180 :=
by
  sorry

noncomputable def Lorelei_jumps1 (H : ℝ) := (3/4) * H
noncomputable def Hattie_jumps2 (H : ℝ) := (2/3) * H
noncomputable def Lorelei_jumps2 (H : ℝ) := (2/3) * H + 50

end hattie_jumps_l29_29583


namespace evaluate_ceil_sqrt_225_l29_29020

def ceil (x : ℝ) : ℤ :=
  if h : ∃ n : ℤ, n ≤ x ∧ x < n + 1 then
    classical.some h
  else
    0

theorem evaluate_ceil_sqrt_225 : ceil (Real.sqrt 225) = 15 := 
sorry

end evaluate_ceil_sqrt_225_l29_29020


namespace smallest_digit_not_found_in_units_place_of_odd_number_l29_29852

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l29_29852


namespace regular_price_of_pony_jeans_l29_29926

-- Define the regular price of fox jeans
def fox_jeans_price := 15

-- Define the given conditions
def pony_discount_rate := 0.18
def total_savings := 9
def total_discount_rate := 0.22

-- State the problem: Prove the regular price of pony jeans
theorem regular_price_of_pony_jeans : 
  ∃ P, P * pony_discount_rate = 3.6 :=
by
  sorry

end regular_price_of_pony_jeans_l29_29926


namespace ceil_sqrt_225_l29_29018

theorem ceil_sqrt_225 : Nat.ceil (Real.sqrt 225) = 15 :=
by
  sorry

end ceil_sqrt_225_l29_29018


namespace charlie_fewer_games_than_dana_l29_29092

theorem charlie_fewer_games_than_dana
  (P D C Ph : ℕ)
  (h1 : P = D + 5)
  (h2 : C < D)
  (h3 : Ph = C + 3)
  (h4 : Ph = 12)
  (h5 : P = Ph + 4) :
  D - C = 2 :=
by
  sorry

end charlie_fewer_games_than_dana_l29_29092


namespace no_real_roots_ff_eq_x_l29_29441

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_real_roots_ff_eq_x (a b c : ℝ)
  (h : a ≠ 0)
  (discriminant_condition : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x := 
by 
  sorry

end no_real_roots_ff_eq_x_l29_29441


namespace square_of_binomial_conditions_l29_29243

variable (x a b m : ℝ)

theorem square_of_binomial_conditions :
  ∃ u v : ℝ, (x + a) * (x - a) = u^2 - v^2 ∧
             ∃ e f : ℝ, (-x - b) * (x - b) = - (e^2 - f^2) ∧
             ∃ g h : ℝ, (b + m) * (m - b) = g^2 - h^2 ∧
             ¬ ∃ p q : ℝ, (a + b) * (-a - b) = p^2 - q^2 :=
by
  sorry

end square_of_binomial_conditions_l29_29243


namespace unsatisfactory_tests_l29_29360

theorem unsatisfactory_tests {n k : ℕ} (h1 : n < 50) 
  (h2 : n % 7 = 0) 
  (h3 : n % 3 = 0) 
  (h4 : n % 2 = 0)
  (h5 : n = 7 * (n / 7) + 3 * (n / 3) + 2 * (n / 2) + k) : 
  k = 1 := 
by 
  sorry

end unsatisfactory_tests_l29_29360


namespace woman_alone_days_l29_29140

theorem woman_alone_days (M W : ℝ) (h1 : (10 * M + 15 * W) * 5 = 1) (h2 : M * 100 = 1) : W * 150 = 1 :=
by
  sorry

end woman_alone_days_l29_29140


namespace mike_age_proof_l29_29538

theorem mike_age_proof (a m : ℝ) (h1 : m = 3 * a - 20) (h2 : m + a = 70) : m = 47.5 := 
by {
  sorry
}

end mike_age_proof_l29_29538


namespace g_at_2_eq_9_l29_29400

def g (x : ℝ) : ℝ := x^2 + 3 * x - 1

theorem g_at_2_eq_9 : g 2 = 9 := by
  sorry

end g_at_2_eq_9_l29_29400


namespace smallest_digit_never_in_units_place_of_odd_l29_29833

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l29_29833


namespace find_m_n_sum_l29_29546

open Fintype

theorem find_m_n_sum : 
  ∀ (m n : ℕ), 
  (m + n).coprime -> 
  let p := (λ c : (Fin 2)^(Fin 4 × Fin 4), ¬ ∃ (i j : Fin 2), ∀ (di dj : Fin 3), c ⟨⟨i + di, di_sle⟩, ⟨j + dj, dj_sle⟩⟩ = 0) in
  p (m :ℝ) / (n :ℝ) = (65275 / 65536) -> 
  m + n = 130811 :=
sorry

end find_m_n_sum_l29_29546


namespace am_gm_inequality_l29_29326

open Real

theorem am_gm_inequality (
    a b c d e f : ℝ
) (h_nonneg_a : 0 ≤ a)
  (h_nonneg_b : 0 ≤ b)
  (h_nonneg_c : 0 ≤ c)
  (h_nonneg_d : 0 ≤ d)
  (h_nonneg_e : 0 ≤ e)
  (h_nonneg_f : 0 ≤ f)
  (h_cond_ab : a + b ≤ e)
  (h_cond_cd : c + d ≤ f) :
  sqrt (a * c) + sqrt (b * d) ≤ sqrt (e * f) := 
  by sorry

end am_gm_inequality_l29_29326


namespace possible_rectangle_configurations_l29_29101

-- Define the conditions as variables
variables (m n : ℕ)
-- Define the number of segments
def segments (m n : ℕ) : ℕ := 2 * m * n + m + n

theorem possible_rectangle_configurations : 
  (segments m n = 1997) → (m = 2 ∧ n = 399) ∨ (m = 8 ∧ n = 117) ∨ (m = 23 ∧ n = 42) :=
by
  sorry

end possible_rectangle_configurations_l29_29101


namespace intersection_M_N_is_neq_neg1_0_1_l29_29471

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N_is_neq_neg1_0_1 :
  M ∩ N = {-1, 0, 1} :=
by
  sorry

end intersection_M_N_is_neq_neg1_0_1_l29_29471


namespace min_value_inequality_l29_29600

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (Real.sqrt ((x^2 + y^2 + z^2) * (4 * x^2 + y^2 + z^2))) / (x * y * z)

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  min_value_expression x y z ≥ 3 / 2 := by
  sorry

end min_value_inequality_l29_29600


namespace sin_sum_of_acute_l29_29214

open Real

theorem sin_sum_of_acute (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  sin (α + β) ≤ sin α + sin β := 
by
  sorry

end sin_sum_of_acute_l29_29214


namespace math_club_team_selection_l29_29332

open scoped BigOperators

def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem math_club_team_selection :
  (comb 7 2 * comb 9 4) + 
  (comb 7 3 * comb 9 3) +
  (comb 7 4 * comb 9 2) +
  (comb 7 5 * comb 9 1) +
  (comb 7 6 * comb 9 0) = 7042 := 
sorry

end math_club_team_selection_l29_29332


namespace selling_price_correct_l29_29536

theorem selling_price_correct (cost_price : ℝ) (loss_percent : ℝ) (selling_price : ℝ) 
  (h_cost : cost_price = 600) 
  (h_loss : loss_percent = 25)
  (h_selling_price : selling_price = cost_price - (loss_percent / 100) * cost_price) : 
  selling_price = 450 := 
by 
  rw [h_cost, h_loss] at h_selling_price
  norm_num at h_selling_price
  exact h_selling_price

#check selling_price_correct

end selling_price_correct_l29_29536


namespace inequality_solution_set_l29_29356

theorem inequality_solution_set (x : ℝ) : (|x - 1| + 2 * x > 4) ↔ (x > 3) := 
sorry

end inequality_solution_set_l29_29356


namespace store_money_left_l29_29261

variable (total_items : Nat) (original_price : ℝ) (discount_percent : ℝ)
variable (percent_sold : ℝ) (amount_owed : ℝ)

theorem store_money_left
  (h_total_items : total_items = 2000)
  (h_original_price : original_price = 50)
  (h_discount_percent : discount_percent = 0.80)
  (h_percent_sold : percent_sold = 0.90)
  (h_amount_owed : amount_owed = 15000)
  : (total_items * original_price * (1 - discount_percent) * percent_sold - amount_owed) = 3000 := 
by 
  sorry

end store_money_left_l29_29261


namespace quadratic_equation_must_be_minus_2_l29_29768

-- Define the main problem statement
theorem quadratic_equation_must_be_minus_2 (m : ℝ) :
  (∀ x : ℝ, (m - 2) * x ^ |m| - 3 * x - 7 = 0) →
  (∀ (h : |m| = 2), m - 2 ≠ 0) →
  m = -2 :=
sorry

end quadratic_equation_must_be_minus_2_l29_29768


namespace savings_account_amount_l29_29265

noncomputable def final_amount : ℝ :=
  let initial_deposit : ℝ := 5000
  let first_quarter_rate : ℝ := 0.01
  let second_quarter_rate : ℝ := 0.0125
  let deposit_end_third_month : ℝ := 1000
  let withdrawal_end_fifth_month : ℝ := 500
  let amount_after_first_quarter := initial_deposit * (1 + first_quarter_rate)
  let amount_before_second_quarter := amount_after_first_quarter + deposit_end_third_month
  let amount_after_second_quarter := amount_before_second_quarter * (1 + second_quarter_rate)
  let final_amount := amount_after_second_quarter - withdrawal_end_fifth_month
  final_amount

theorem savings_account_amount :
  final_amount = 5625.625 :=
by
  sorry

end savings_account_amount_l29_29265


namespace chocolate_ratio_l29_29474

theorem chocolate_ratio (N A : ℕ) (h1 : N = 10) (h2 : A - 5 = N + 15) : A / N = 3 :=
by {
  sorry
}

end chocolate_ratio_l29_29474


namespace max_length_PC_l29_29958

-- Define the circle C and its properties
def Circle (x y : ℝ) : Prop := x^2 + (y-1)^2 = 4

-- The equilateral triangle condition and what we need to prove
theorem max_length_PC :
  (∃ (P A B : ℝ × ℝ), 
    (Circle A.1 A.2) ∧
    (Circle B.1 B.2) ∧
    (Circle ((A.1 + B.1) / 2) ((A.2 + B.2) / 2)) ∧
    (A ≠ B) ∧
    (∃ r : ℝ, (A.1 - B.1)^2 + (A.2 - B.2)^2 = r^2 ∧ 
               (P.1 - A.1)^2 + (P.2 - A.2)^2 = r^2 ∧ 
               (P.1 - B.1)^2 + (P.2 - B.2)^2 = r^2)) → 
  (∀ (P : ℝ × ℝ), 
     ∃ (max_val : ℝ), max_val = 4 ∧
     (¬(∃ (Q : ℝ × ℝ), (Circle P.1 P.2) ∧ ((Q.1 - 0)^2 + (Q.2 - 1)^2 > max_val^2))))
:= 
sorry

end max_length_PC_l29_29958


namespace fraction_spent_l29_29267

theorem fraction_spent (borrowed_from_brother borrowed_from_father borrowed_from_mother gift_from_granny savings remaining amount_spent : ℕ)
  (h_borrowed_from_brother : borrowed_from_brother = 20)
  (h_borrowed_from_father : borrowed_from_father = 40)
  (h_borrowed_from_mother : borrowed_from_mother = 30)
  (h_gift_from_granny : gift_from_granny = 70)
  (h_savings : savings = 100)
  (h_remaining : remaining = 65)
  (h_amount_spent : amount_spent = borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings - remaining) :
  (amount_spent : ℚ) / (borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings) = 3 / 4 :=
by
  sorry

end fraction_spent_l29_29267


namespace total_students_end_of_year_l29_29152

def M := 50
def E (M : ℕ) := 4 * M - 3
def H (E : ℕ) := 2 * E

def E_end (E : ℕ) := E + (E / 10)
def M_end (M : ℕ) := M - (M / 20)
def H_end (H : ℕ) := H + ((7 * H) / 100)

def total_end (E_end M_end H_end : ℕ) := E_end + M_end + H_end

theorem total_students_end_of_year : 
  total_end (E_end (E M)) (M_end M) (H_end (H (E M))) = 687 := sorry

end total_students_end_of_year_l29_29152


namespace gcd_of_repeated_six_digit_integers_l29_29531

-- Given condition
def is_repeated_six_digit_integer (n : ℕ) : Prop :=
  100 ≤ n / 1000 ∧ n / 1000 < 1000 ∧ n = 1001 * (n / 1000)

-- Theorem to prove
theorem gcd_of_repeated_six_digit_integers :
  ∀ n : ℕ, is_repeated_six_digit_integer n → gcd n 1001 = 1001 :=
by sorry

end gcd_of_repeated_six_digit_integers_l29_29531


namespace magnitude_product_complex_l29_29553

theorem magnitude_product_complex :
  let z1 := Complex.mk 7 (-4)
  let z2 := Complex.mk 3 11
  Complex.abs (z1 * z2) = Real.sqrt 8450 :=
by
  sorry

end magnitude_product_complex_l29_29553


namespace arithmetic_sequence_eighth_term_l29_29195

theorem arithmetic_sequence_eighth_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 2) : a 8 = 15 := by
  sorry

end arithmetic_sequence_eighth_term_l29_29195


namespace real_part_fraction_l29_29932

theorem real_part_fraction {i : ℂ} (h : i^2 = -1) : (
  let numerator := 1 - i
  let denominator := (1 + i) ^ 2
  let fraction := numerator / denominator
  let real_part := (fraction.re)
  real_part
) = -1/2 := sorry

end real_part_fraction_l29_29932


namespace time_for_six_visits_l29_29966

noncomputable def time_to_go_n_times (total_time : ℕ) (total_visits : ℕ) (n_visits : ℕ) : ℕ :=
  (total_time / total_visits) * n_visits

theorem time_for_six_visits (h : time_to_go_n_times 20 8 6 = 15) : time_to_go_n_times 20 8 6 = 15 :=
by
  exact h

end time_for_six_visits_l29_29966


namespace matches_length_l29_29224

-- Definitions and conditions
def area_shaded_figure : ℝ := 300 -- given in cm^2
def num_small_squares : ℕ := 8
def large_square_area_coefficient : ℕ := 4
def area_small_square (a : ℝ) : ℝ := num_small_squares * a + large_square_area_coefficient * a

-- Question and answer to be proven
theorem matches_length (a : ℝ) (side_length: ℝ) :
  area_shaded_figure = 300 → 
  area_small_square a = area_shaded_figure →
  (a = 25) →
  (side_length = 5) →
  4 * 7 * side_length = 140 :=
by
  intros h1 h2 h3 h4
  sorry

end matches_length_l29_29224


namespace percent_difference_l29_29898

theorem percent_difference : 0.12 * 24.2 - 0.10 * 14.2 = 1.484 := by
  sorry

end percent_difference_l29_29898


namespace find_cubic_polynomial_l29_29032

theorem find_cubic_polynomial (q : ℝ → ℝ) 
  (h1 : q 1 = -8) 
  (h2 : q 2 = -12) 
  (h3 : q 3 = -20) 
  (h4 : q 4 = -40) : 
  q = (λ x, - (4 / 3) * x^3 + 6 * x^2 - 4 * x - 2) :=
sorry

end find_cubic_polynomial_l29_29032


namespace height_of_removed_player_l29_29619

theorem height_of_removed_player (S : ℕ) (x : ℕ) (total_height_11 : S + x = 182 * 11)
  (average_height_10 : S = 181 * 10): x = 192 :=
by
  sorry

end height_of_removed_player_l29_29619


namespace point_c_third_quadrant_l29_29069

variable (a b : ℝ)

-- Definition of the conditions
def condition_1 : Prop := b = -1
def condition_2 : Prop := a = -3

-- Definition to check if a point is in the third quadrant
def is_third_quadrant (a b : ℝ) : Prop := a < 0 ∧ b < 0

-- The main statement to be proven
theorem point_c_third_quadrant (h1 : condition_1 b) (h2 : condition_2 a) :
  is_third_quadrant a b :=
by
  -- Proof of the theorem (to be completed)
  sorry

end point_c_third_quadrant_l29_29069


namespace bin_to_oct_l29_29682

theorem bin_to_oct (n : ℕ) (hn : n = 0b11010) : n = 0o32 := by
  sorry

end bin_to_oct_l29_29682


namespace probability_exactly_one_second_class_product_l29_29255

open Nat

/-- Proof problem -/
theorem probability_exactly_one_second_class_product :
  let n := 100 -- total products
  let k := 4   -- number of selected products
  let first_class := 90 -- first-class products
  let second_class := 10 -- second-class products
  let C (n k : ℕ) := Nat.choose n k
  (C second_class 1 * C first_class 3 : ℚ) / C n k = 
  (Nat.choose second_class 1 * Nat.choose first_class 3 : ℚ) / Nat.choose n k :=
by
  -- Mathematically equivalent proof
  sorry

end probability_exactly_one_second_class_product_l29_29255


namespace ceil_sqrt_225_eq_15_l29_29024

theorem ceil_sqrt_225_eq_15 : ⌈ Real.sqrt 225 ⌉ = 15 :=
by
  sorry

end ceil_sqrt_225_eq_15_l29_29024


namespace find_y_l29_29560

theorem find_y (y : ℝ) (h : (y + 10 + (5 * y) + 4 + (3 * y) + 12) / 3 = 6 * y - 8) :
  y = 50 / 9 := by
  sorry

end find_y_l29_29560


namespace express_x2_y2_z2_in_terms_of_sigma1_sigma2_l29_29915

variable (x y z : ℝ)
def sigma1 := x + y + z
def sigma2 := x * y + y * z + z * x

theorem express_x2_y2_z2_in_terms_of_sigma1_sigma2 :
  x^2 + y^2 + z^2 = sigma1 x y z ^ 2 - 2 * sigma2 x y z := by
  sorry

end express_x2_y2_z2_in_terms_of_sigma1_sigma2_l29_29915


namespace coffee_consumption_l29_29229

theorem coffee_consumption (h1 h2 g1 h3: ℕ) (k : ℕ) (g2 : ℕ) :
  (k = h1 * g1) → (h1 = 9) → (g1 = 2) → (h2 = 6) → (k / h2 = g2) → (g2 = 3) :=
by
  sorry

end coffee_consumption_l29_29229


namespace range_of_m_l29_29753

-- Defining the point P and the required conditions for it to lie in the fourth quadrant
def point_in_fourth_quadrant (m : ℝ) : Prop :=
  let P := (m + 3, m - 1)
  P.1 > 0 ∧ P.2 < 0

-- Defining the range of m for which the point lies in the fourth quadrant
theorem range_of_m (m : ℝ) : point_in_fourth_quadrant m ↔ (-3 < m ∧ m < 1) :=
by
  sorry

end range_of_m_l29_29753


namespace simplify_expression_evaluate_expression_at_neg1_evaluate_expression_at_2_l29_29761

theorem simplify_expression (a : ℤ) (h1 : -2 < a) (h2 : a ≤ 2) (h3 : a ≠ 0) (h4 : a ≠ 1) :
  (a - (2 * a - 1) / a) / ((a - 1) / a) = a - 1 :=
by
  sorry

theorem evaluate_expression_at_neg1 (h : (-1 : ℤ) ≠ 0) (h' : (-1 : ℤ) ≠ 1) : 
  (-1 - (2 * (-1) - 1) / (-1)) / ((-1 - 1) / (-1)) = -2 :=
by
  sorry

theorem evaluate_expression_at_2 (h : (2 : ℤ) ≠ 0) (h' : (2 : ℤ) ≠ 1) : 
  (2 - (2 * 2 - 1) / 2) / ((2 - 1) / 2) = 1 :=
by
  sorry

end simplify_expression_evaluate_expression_at_neg1_evaluate_expression_at_2_l29_29761


namespace xsquared_plus_5x_minus_6_condition_l29_29102

theorem xsquared_plus_5x_minus_6_condition (x : ℝ) : 
  (x^2 + 5 * x - 6 > 0) → (x > 2) ∨ (((x > 1) ∨ (x < -6)) ∧ ¬(x > 2)) := 
sorry

end xsquared_plus_5x_minus_6_condition_l29_29102


namespace solution_set_of_system_of_inequalities_l29_29233

theorem solution_set_of_system_of_inequalities :
  {x : ℝ | |x| - 1 < 0 ∧ x^2 - 3 * x < 0} = {x : ℝ | 0 < x ∧ x < 1} :=
sorry

end solution_set_of_system_of_inequalities_l29_29233


namespace total_logs_in_stack_l29_29388

theorem total_logs_in_stack : 
  ∀ (a_1 a_n : ℕ) (n : ℕ), 
  a_1 = 5 → a_n = 15 → n = a_n - a_1 + 1 → 
  (a_1 + a_n) * n / 2 = 110 :=
by
  intros a_1 a_n n h1 h2 h3
  sorry

end total_logs_in_stack_l29_29388


namespace inequality_solution_l29_29586

theorem inequality_solution (a b : ℝ)
  (h : ∀ x : ℝ, 0 ≤ x → (x^2 + 1 ≥ a * x + b ∧ a * x + b ≥ (3 / 2) * x^(2 / 3) )) :
  (2 - Real.sqrt 2) / 4 ≤ b ∧ b ≤ (2 + Real.sqrt 2) / 4 ∧
  (1 / Real.sqrt (2 * b)) ≤ a ∧ a ≤ 2 * Real.sqrt (1 - b) :=
  sorry

end inequality_solution_l29_29586


namespace solve_equation_l29_29617

theorem solve_equation : ∀ (x : ℝ), -2 * x + 3 - 2 * x + 3 = 3 * x - 6 → x = 12 / 7 :=
by 
  intro x
  intro h
  sorry

end solve_equation_l29_29617


namespace common_chord_of_circles_l29_29223

theorem common_chord_of_circles :
  ∀ (x y : ℝ), (x^2 + y^2 + 2 * x = 0) ∧ (x^2 + y^2 - 4 * y = 0) → (x + 2 * y = 0) :=
by
  sorry

end common_chord_of_circles_l29_29223


namespace sum_arithmetic_sequence_min_value_l29_29568

theorem sum_arithmetic_sequence_min_value (a d : ℤ) 
  (S : ℕ → ℤ) 
  (H1 : S 8 ≤ 6) 
  (H2 : S 11 ≥ 27)
  (H_Sn : ∀ n, S n = n * a + (n * (n - 1) / 2) * d) : 
  S 19 ≥ 133 :=
by
  sorry

end sum_arithmetic_sequence_min_value_l29_29568


namespace second_order_arithmetic_progression_a100_l29_29951

theorem second_order_arithmetic_progression_a100 :
  ∀ (a : ℕ → ℕ), 
    a 1 = 2 → 
    a 2 = 3 → 
    a 3 = 5 → 
    (∀ n, a (n + 1) - a n = n) → 
    a 100 = 4952 :=
by
  intros a h1 h2 h3 hdiff
  sorry

end second_order_arithmetic_progression_a100_l29_29951


namespace ribbon_cuts_l29_29886

theorem ribbon_cuts (rolls : ℕ) (length_per_roll : ℕ) (piece_length : ℕ) (total_rolls : rolls = 5) (roll_length : length_per_roll = 50) (piece_size : piece_length = 2) : 
  (rolls * ((length_per_roll / piece_length) - 1) = 120) :=
by
  sorry

end ribbon_cuts_l29_29886


namespace frost_time_with_sprained_wrist_l29_29475

-- Definitions
def normal_time_per_cake : ℕ := 5
def additional_time_for_10_cakes : ℕ := 30
def normal_time_for_10_cakes : ℕ := 10 * normal_time_per_cake
def sprained_time_for_10_cakes : ℕ := normal_time_for_10_cakes + additional_time_for_10_cakes

-- Theorems
theorem frost_time_with_sprained_wrist : ∀ x : ℕ, 
  (10 * x = sprained_time_for_10_cakes) ↔ (x = 8) := 
sorry

end frost_time_with_sprained_wrist_l29_29475


namespace gcd_245_1001_l29_29504

-- Definitions based on the given conditions

def fact245 : ℕ := 5 * 7^2
def fact1001 : ℕ := 7 * 11 * 13

-- Lean 4 statement of the proof problem
theorem gcd_245_1001 : Nat.gcd fact245 fact1001 = 7 :=
by
  -- Add the prime factorizations as assumptions
  have h1: fact245 = 245 := by sorry
  have h2: fact1001 = 1001 := by sorry
  -- The goal is to prove the GCD
  sorry

end gcd_245_1001_l29_29504


namespace least_value_expression_l29_29242

theorem least_value_expression : ∃ x : ℝ, ∀ (x : ℝ), (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 ≥ 2094
∧ ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 = 2094 := by
  sorry

end least_value_expression_l29_29242


namespace smallest_digit_not_in_units_place_of_odd_l29_29807

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l29_29807


namespace excluded_number_is_35_l29_29620

theorem excluded_number_is_35 (numbers : List ℝ) 
  (h_len : numbers.length = 5)
  (h_avg1 : (numbers.sum / 5) = 27)
  (h_len_excl : (numbers.length - 1) = 4)
  (avg_remaining : ℝ)
  (remaining_numbers : List ℝ)
  (remaining_condition : remaining_numbers.length = 4)
  (h_avg2 : (remaining_numbers.sum / 4) = 25) :
  numbers.sum - remaining_numbers.sum = 35 :=
by sorry

end excluded_number_is_35_l29_29620


namespace average_age_of_women_l29_29885

theorem average_age_of_women (A : ℕ) :
  (6 * (A + 2) = 6 * A - 22 + W) → (W / 2 = 17) :=
by
  intro h
  sorry

end average_age_of_women_l29_29885


namespace sum_of_roots_ln_abs_eq_l29_29975

theorem sum_of_roots_ln_abs_eq (m : ℝ) (x1 x2 : ℝ) (hx1 : Real.log (|x1|) = m) (hx2 : Real.log (|x2|) = m) : x1 + x2 = 0 :=
sorry

end sum_of_roots_ln_abs_eq_l29_29975


namespace initial_price_of_phone_l29_29269

theorem initial_price_of_phone
  (initial_price_TV : ℕ)
  (increase_TV_fraction : ℚ)
  (initial_price_phone : ℚ)
  (increase_phone_percentage : ℚ)
  (total_amount : ℚ)
  (h1 : initial_price_TV = 500)
  (h2 : increase_TV_fraction = 2/5)
  (h3 : increase_phone_percentage = 0.40)
  (h4 : total_amount = 1260) :
  initial_price_phone = 400 := by
  sorry

end initial_price_of_phone_l29_29269


namespace three_fifths_difference_products_l29_29667

theorem three_fifths_difference_products :
  (3 / 5) * ((7 * 9) - (4 * 3)) = 153 / 5 :=
by
  sorry

end three_fifths_difference_products_l29_29667


namespace largest_integer_satisfying_sin_cos_condition_proof_l29_29413

noncomputable def largest_integer_satisfying_sin_cos_condition :=
  ∀ (x : ℝ) (n : ℕ), (∀ (n' : ℕ), (∀ x : ℝ, (Real.sin x ^ n' + Real.cos x ^ n' ≥ 2 / n') → n ≤ n')) → n = 4

theorem largest_integer_satisfying_sin_cos_condition_proof :
  largest_integer_satisfying_sin_cos_condition :=
by
  sorry

end largest_integer_satisfying_sin_cos_condition_proof_l29_29413


namespace always_positive_iff_k_gt_half_l29_29110

theorem always_positive_iff_k_gt_half (k : ℝ) :
  (∀ x : ℝ, k * x^2 + x + k > 0) ↔ k > 0.5 :=
sorry

end always_positive_iff_k_gt_half_l29_29110


namespace age_difference_l29_29515

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 16) : A - C = 16 :=
sorry

end age_difference_l29_29515


namespace remainder_of_m_div_5_l29_29185

theorem remainder_of_m_div_5 (m n : ℕ) (h1 : m = 15 * n - 1) (h2 : n > 0) : m % 5 = 4 :=
sorry

end remainder_of_m_div_5_l29_29185


namespace find_share_of_b_l29_29514

variable (a b c : ℕ)
axiom h1 : a = 3 * b
axiom h2 : b = c + 25
axiom h3 : a + b + c = 645

theorem find_share_of_b : b = 134 := by
  sorry

end find_share_of_b_l29_29514


namespace galya_number_l29_29166

theorem galya_number (N k : ℤ) (h : (k - N + 1 = k - 7729)) : N = 7730 := 
by
  sorry

end galya_number_l29_29166


namespace time_for_grid_5x5_l29_29192

-- Definition for the 3x7 grid conditions
def grid_3x7_minutes := 26
def grid_3x7_total_length := 4 * 7 + 8 * 3
def time_per_unit_length := grid_3x7_minutes / grid_3x7_total_length

-- Definition for the 5x5 grid total length
def grid_5x5_total_length := 6 * 5 + 6 * 5

-- Theorem stating that the time it takes to trace all lines of a 5x5 grid is 30 minutes
theorem time_for_grid_5x5 : (time_per_unit_length * grid_5x5_total_length) = 30 := by
  sorry

end time_for_grid_5x5_l29_29192


namespace simplify_evaluate_l29_29760

theorem simplify_evaluate :
  ∀ (x : ℝ), x = Real.sqrt 2 - 1 →
  ((1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6))) = Real.sqrt 2 :=
by
  intros x hx
  sorry

end simplify_evaluate_l29_29760


namespace unique_integer_solution_l29_29561

def is_point_in_circle (x y cx cy radius : ℝ) : Prop :=
  (x - cx)^2 + (y - cy)^2 ≤ radius^2

theorem unique_integer_solution : ∃! (x : ℤ), is_point_in_circle (2 * x) (-x) 4 6 8 := by
  sorry

end unique_integer_solution_l29_29561


namespace fraction_order_l29_29126

theorem fraction_order :
  (19 / 15 < 17 / 13) ∧ (17 / 13 < 15 / 11) :=
by
  sorry

end fraction_order_l29_29126


namespace intersection_eq_l29_29186

def A : Set Int := { -1, 0, 1 }
def B : Set Int := { 0, 1, 2 }

theorem intersection_eq :
  A ∩ B = {0, 1} := 
by 
  sorry

end intersection_eq_l29_29186


namespace correct_equation_l29_29279

-- Definitions of the conditions
def january_turnover (T : ℝ) : Prop := T = 36
def march_turnover (T : ℝ) : Prop := T = 48
def average_monthly_growth_rate (x : ℝ) : Prop := True

-- The goal to be proved
theorem correct_equation (T_jan T_mar : ℝ) (x : ℝ) 
  (h_jan : january_turnover T_jan) 
  (h_mar : march_turnover T_mar) 
  (h_growth : average_monthly_growth_rate x) : 
  36 * (1 + x)^2 = 48 :=
sorry

end correct_equation_l29_29279


namespace residue_n_mod_17_l29_29343

noncomputable def satisfies_conditions (m n k : ℕ) : Prop :=
  m^2 + 1 = 2 * n^2 ∧ 2 * m^2 + 1 = 11 * k^2 

theorem residue_n_mod_17 (m n k : ℕ) (h : satisfies_conditions m n k) : n % 17 = 5 :=
  sorry

end residue_n_mod_17_l29_29343


namespace subtraction_solution_l29_29375

noncomputable def x : ℝ := 47.806

theorem subtraction_solution :
  (3889 : ℝ) + 12.808 - x = 3854.002 :=
by
  sorry

end subtraction_solution_l29_29375


namespace union_of_A_and_B_l29_29251

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} :=
by
  sorry

end union_of_A_and_B_l29_29251


namespace recommended_sleep_hours_l29_29078

theorem recommended_sleep_hours
  (R : ℝ)   -- The recommended number of hours of sleep per day
  (h1 : 2 * 3 + 5 * (0.60 * R) = 30) : R = 8 :=
sorry

end recommended_sleep_hours_l29_29078


namespace principal_amount_l29_29659

noncomputable def exponential (r t : ℝ) :=
  Real.exp (r * t)

theorem principal_amount (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) :
  A = 5673981 ∧ r = 0.1125 ∧ t = 7.5 ∧ P = 2438978.57 →
  P = A / exponential r t := 
by
  intros h
  sorry

end principal_amount_l29_29659


namespace smallest_digit_not_in_units_place_of_odd_numbers_l29_29818

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l29_29818


namespace minimum_value_of_functions_l29_29939

def linear_fn (a b c: ℝ) := a ≠ 0 
def f (a b: ℝ) (x: ℝ) := a * x + b 
def g (a c: ℝ) (x: ℝ) := a * x + c

theorem minimum_value_of_functions (a b c: ℝ) (hx: linear_fn a b c) :
  (∀ x: ℝ, 3 * (f a b x)^2 + 2 * g a c x ≥ -19 / 6) → (∀ x: ℝ, 3 * (g a c x)^2 + 2 * f a b x ≥ 5 / 2) :=
by
  sorry

end minimum_value_of_functions_l29_29939


namespace total_distance_traveled_l29_29091

theorem total_distance_traveled:
  let speed1 := 30
  let time1 := 4
  let speed2 := 35
  let time2 := 5
  let speed3 := 25
  let time3 := 6
  let total_time := 20
  let time1_3 := time1 + time2 + time3
  let time4 := total_time - time1_3
  let speed4 := 40

  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4

  let total_distance := distance1 + distance2 + distance3 + distance4

  total_distance = 645 :=
  sorry

end total_distance_traveled_l29_29091


namespace ceil_sqrt_225_l29_29019

theorem ceil_sqrt_225 : Nat.ceil (Real.sqrt 225) = 15 :=
by
  sorry

end ceil_sqrt_225_l29_29019


namespace milk_remaining_l29_29236

def initial_whole_milk := 15
def initial_low_fat_milk := 12
def initial_almond_milk := 8

def jason_buys := 5
def jason_promotion := 2 -- every 2 bottles he gets 1 free

def harry_buys_low_fat := 4
def harry_gets_free_low_fat := 1
def harry_buys_almond := 2

theorem milk_remaining : 
  (initial_whole_milk - jason_buys = 10) ∧ 
  (initial_low_fat_milk - (harry_buys_low_fat + harry_gets_free_low_fat) = 7) ∧ 
  (initial_almond_milk - harry_buys_almond = 6) :=
by
  sorry

end milk_remaining_l29_29236


namespace equivalent_proof_problem_l29_29649

variable (a b d e c f g h : ℚ)

def condition1 : Prop := 8 = (6 / 100) * a
def condition2 : Prop := 6 = (8 / 100) * b
def condition3 : Prop := 9 = (5 / 100) * d
def condition4 : Prop := 7 = (3 / 100) * e
def condition5 : Prop := c = b / a
def condition6 : Prop := f = d / a
def condition7 : Prop := g = e / b

theorem equivalent_proof_problem (hac1 : condition1 a)
                                 (hac2 : condition2 b)
                                 (hac3 : condition3 d)
                                 (hac4 : condition4 e)
                                 (hac5 : condition5 a b c)
                                 (hac6 : condition6 a d f)
                                 (hac7 : condition7 b e g) :
    h = f + g ↔ h = (803 / 20) * c := 
by sorry

end equivalent_proof_problem_l29_29649


namespace manufacturing_employees_percentage_l29_29220

theorem manufacturing_employees_percentage 
  (total_circle_deg : ℝ := 360) 
  (manufacturing_deg : ℝ := 18) 
  (sector_proportion : ∀ x y, x / y = (x/y : ℝ)) 
  (percentage : ∀ x, x * 100 = (x * 100 : ℝ)) :
  (manufacturing_deg / total_circle_deg) * 100 = 5 := 
by sorry

end manufacturing_employees_percentage_l29_29220


namespace ring_cost_l29_29940

theorem ring_cost (total_cost : ℕ) (rings : ℕ) (h1 : total_cost = 24) (h2 : rings = 2) : total_cost / rings = 12 :=
by
  sorry

end ring_cost_l29_29940


namespace mod_inverse_7_31_l29_29414

theorem mod_inverse_7_31 : ∃ a : ℕ, (7 * a) % 31 = 1 ∧ a = 9 :=
by
  use 9
  split
  by norm_num
  sorry

end mod_inverse_7_31_l29_29414


namespace smallest_missing_digit_l29_29860

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l29_29860


namespace john_must_work_10_more_days_l29_29463

-- Define the conditions as hypotheses
def total_days_worked := 10
def total_earnings := 250
def desired_total_earnings := total_earnings * 2
def daily_earnings := total_earnings / total_days_worked

-- Theorem that needs to be proved
theorem john_must_work_10_more_days:
  (desired_total_earnings / daily_earnings) - total_days_worked = 10 := by
  sorry

end john_must_work_10_more_days_l29_29463


namespace intersection_A_B_l29_29426

def A : Set ℝ := {-2, -1, 2, 3}
def B : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem intersection_A_B : A ∩ B = {-1, 2} :=
by
  sorry

end intersection_A_B_l29_29426


namespace kelvin_can_win_l29_29970

-- Defining the game conditions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Game Strategy
def kelvin_always_wins : Prop :=
  ∀ (n : ℕ), ∀ (d : ℕ), (d ∈ (List.range 10)) → 
    ∃ (k : ℕ), k ∈ [3, 7] ∧ ¬is_perfect_square (10 * n + k)

theorem kelvin_can_win : kelvin_always_wins :=
by {
  sorry -- Proof based on strategy of adding 3 or 7 modulo 10 and modulo 100 analysis
}

end kelvin_can_win_l29_29970


namespace line_passes_through_fixed_point_l29_29175

theorem line_passes_through_fixed_point (m : ℝ) : 
  (2 + m) * (-1) + (1 - 2 * m) * (-2) + 4 - 3 * m = 0 :=
by
  sorry

end line_passes_through_fixed_point_l29_29175


namespace surface_area_inequality_l29_29042

theorem surface_area_inequality
  (a b c d e f S : ℝ) :
  S ≤ (Real.sqrt 3 / 6) * (a^2 + b^2 + c^2 + d^2 + e^2 + f^2) :=
sorry

end surface_area_inequality_l29_29042


namespace find_three_digit_numbers_l29_29688
open Nat

theorem find_three_digit_numbers (n : ℕ) (h1 : 100 ≤ n) (h2 : n < 1000) (h3 : ∀ (k : ℕ), n^k % 1000 = n % 1000) : n = 625 ∨ n = 376 :=
sorry

end find_three_digit_numbers_l29_29688


namespace inequality_relationship_cannot_be_established_l29_29565

variable (a b : ℝ)
variable (h : a < b ∧ b < 0)

theorem inequality_relationship_cannot_be_established :
  ¬ (1 / (a - b) > 1 / a) :=
by sorry

end inequality_relationship_cannot_be_established_l29_29565


namespace parallel_lines_slope_l29_29587

theorem parallel_lines_slope (m : ℚ) (h : (x - y = 1) → (m + 3) * x + m * y - 8 = 0) :
  m = -3 / 2 :=
sorry

end parallel_lines_slope_l29_29587


namespace speed_downstream_l29_29144

variables (V_m V_s V_u V_d : ℕ)
variables (h1 : V_u = 12)
variables (h2 : V_m = 25)
variables (h3 : V_u = V_m - V_s)

theorem speed_downstream (h1 : V_u = 12) (h2 : V_m = 25) (h3 : V_u = V_m - V_s) :
  V_d = V_m + V_s :=
by
  -- The proof goes here
  sorry

end speed_downstream_l29_29144


namespace find_x_l29_29210

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0)
  (coins_megan : ℤ := 42)
  (coins_shana : ℤ := 35)
  (shana_win : ℕ := 2)
  (total_megan : shana_win * x + (total_races - shana_win) * y = coins_shana)
  (total_shana : (total_races - shana_win) * x + shana_win * y = coins_megan) :
  x = 4 := by
  sorry

end find_x_l29_29210


namespace leftover_value_is_5_30_l29_29259

variable (q_per_roll d_per_roll : ℕ)
variable (j_quarters j_dimes l_quarters l_dimes : ℕ)
variable (value_per_quarter value_per_dime : ℝ)

def total_leftover_value (q_per_roll d_per_roll : ℕ) 
  (j_quarters l_quarters j_dimes l_dimes : ℕ)
  (value_per_quarter value_per_dime : ℝ) : ℝ :=
  let total_quarters := j_quarters + l_quarters
  let total_dimes := j_dimes + l_dimes
  let leftover_quarters := total_quarters % q_per_roll
  let leftover_dimes := total_dimes % d_per_roll
  (leftover_quarters * value_per_quarter) + (leftover_dimes * value_per_dime)

theorem leftover_value_is_5_30 :
  total_leftover_value 45 55 95 140 173 285 0.25 0.10 = 5.3 := 
by
  sorry

end leftover_value_is_5_30_l29_29259


namespace original_average_age_l29_29766

theorem original_average_age (N : ℕ) (A : ℝ) (h1 : A = 50) (h2 : 12 * 32 + N * 50 = (N + 12) * (A - 4)) : A = 50 := by
  sorry 

end original_average_age_l29_29766


namespace find_x_l29_29718

theorem find_x (y : ℝ) (x : ℝ) (h : x / (x - 1) = (y^2 + 2 * y + 3) / (y^2 + 2 * y - 2)) :
  x = (y^2 + 2 * y + 3) / 5 := by
  sorry

end find_x_l29_29718


namespace complement_of_M_in_U_l29_29711

def universal_set : Set ℝ := {x | x > 0}
def set_M : Set ℝ := {x | x > 1}
def complement (U M : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ M}

theorem complement_of_M_in_U :
  complement universal_set set_M = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end complement_of_M_in_U_l29_29711


namespace final_game_deficit_l29_29327

-- Define the points for each scoring action
def free_throw_points := 1
def three_pointer_points := 3
def jump_shot_points := 2
def layup_points := 2
def and_one_points := layup_points + free_throw_points

-- Define the points scored by Liz
def liz_free_throws := 5 * free_throw_points
def liz_three_pointers := 4 * three_pointer_points
def liz_jump_shots := 5 * jump_shot_points
def liz_and_one := and_one_points

def liz_points := liz_free_throws + liz_three_pointers + liz_jump_shots + liz_and_one

-- Define the points scored by Taylor
def taylor_three_pointers := 2 * three_pointer_points
def taylor_jump_shots := 3 * jump_shot_points

def taylor_points := taylor_three_pointers + taylor_jump_shots

-- Define the points for Liz's team
def team_points := liz_points + taylor_points

-- Define the points scored by the opposing team players
def opponent_player1_points := 4 * three_pointer_points

def opponent_player2_jump_shots := 4 * jump_shot_points
def opponent_player2_free_throws := 2 * free_throw_points
def opponent_player2_points := opponent_player2_jump_shots + opponent_player2_free_throws

def opponent_player3_jump_shots := 2 * jump_shot_points
def opponent_player3_three_pointer := 1 * three_pointer_points
def opponent_player3_points := opponent_player3_jump_shots + opponent_player3_three_pointer

-- Define the points for the opposing team
def opponent_team_points := opponent_player1_points + opponent_player2_points + opponent_player3_points

-- Initial deficit
def initial_deficit := 25

-- Final net scoring in the final quarter
def net_quarter_scoring := team_points - opponent_team_points

-- Final deficit
def final_deficit := initial_deficit - net_quarter_scoring

theorem final_game_deficit : final_deficit = 12 := by
  sorry

end final_game_deficit_l29_29327


namespace kids_difference_l29_29969

def kidsPlayedOnMonday : Nat := 11
def kidsPlayedOnTuesday : Nat := 12

theorem kids_difference :
  kidsPlayedOnTuesday - kidsPlayedOnMonday = 1 := by
  sorry

end kids_difference_l29_29969


namespace smallest_digit_not_in_units_place_of_odd_l29_29793

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l29_29793


namespace solve_y_minus_x_l29_29630

theorem solve_y_minus_x (x y : ℝ) (h1 : x + y = 399) (h2 : x / y = 0.9) : y - x = 21 :=
sorry

end solve_y_minus_x_l29_29630


namespace smallest_digit_not_in_units_place_of_odd_l29_29810

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l29_29810


namespace polynomial_nonnegative_iff_eq_l29_29677

variable {R : Type} [LinearOrderedField R]

def polynomial_p (x a b c : R) : R :=
  (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem polynomial_nonnegative_iff_eq (a b c : R) :
  (∀ x : R, polynomial_p x a b c ≥ 0) ↔ a = b ∧ b = c :=
by
  sorry

end polynomial_nonnegative_iff_eq_l29_29677


namespace find_the_number_l29_29346

theorem find_the_number (x : ℝ) : (3 * x - 1 = 2 * x^2) ∧ (2 * x = (3 * x - 1) / x) → x = 1 := 
by sorry

end find_the_number_l29_29346


namespace top_card_is_queen_probability_l29_29532

-- Define the conditions of the problem
def standard_deck_size := 52
def number_of_queens := 4

-- Problem statement: The probability that the top card is a Queen
theorem top_card_is_queen_probability : 
  (number_of_queens : ℚ) / standard_deck_size = 1 / 13 := 
sorry

end top_card_is_queen_probability_l29_29532


namespace simplify_expression_l29_29717

variable (x y z : ℝ)

theorem simplify_expression (hxz : x > z) (hzy : z > y) (hy0 : y > 0) :
  (x^z * z^y * y^x) / (z^z * y^y * x^x) = x^(z-x) * z^(y-z) * y^(x-y) :=
sorry

end simplify_expression_l29_29717


namespace find_metal_molecular_weight_l29_29692

noncomputable def molecular_weight_of_metal (compound_mw: ℝ) (oh_mw: ℝ) : ℝ :=
  compound_mw - oh_mw

theorem find_metal_molecular_weight :
  let compound_mw := 171.00
  let oxygen_mw := 16.00
  let hydrogen_mw := 1.01
  let oh_ions := 2
  let oh_mw := oh_ions * (oxygen_mw + hydrogen_mw)
  molecular_weight_of_metal compound_mw oh_mw = 136.98 :=
by
  sorry

end find_metal_molecular_weight_l29_29692


namespace triangle_in_and_circumcircle_radius_l29_29361

noncomputable def radius_of_incircle (AC : ℝ) (BC : ℝ) (AB : ℝ) (Area : ℝ) (s : ℝ) : ℝ :=
  Area / s

noncomputable def radius_of_circumcircle (AB : ℝ) : ℝ :=
  AB / 2

theorem triangle_in_and_circumcircle_radius :
  ∀ (A B C : ℝ × ℝ) (AC : ℝ) (BC : ℝ) (AB : ℝ)
    (AngleA : ℝ) (AngleC : ℝ),
  AngleC = 90 ∧ AngleA = 60 ∧ AC = 6 ∧
  BC = AC * Real.sqrt 3 ∧ AB = 2 * AC
  → radius_of_incircle AC BC AB (18 * Real.sqrt 3) ((AC + BC + AB) / 2) = 6 * (Real.sqrt 3 - 1) / 13 ∧
    radius_of_circumcircle AB = 6 := by
  intros A B C AC BC AB AngleA AngleC h
  sorry

end triangle_in_and_circumcircle_radius_l29_29361


namespace next_tutoring_day_lcm_l29_29683

theorem next_tutoring_day_lcm :
  Nat.lcm (Nat.lcm (Nat.lcm 4 5) 6) 8 = 120 := by
  sorry

end next_tutoring_day_lcm_l29_29683


namespace maximum_ab_l29_29322

theorem maximum_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3*a + 8*b = 48) : ab ≤ 24 :=
by
  sorry

end maximum_ab_l29_29322


namespace tensor_identity_l29_29544

def tensor (a b : ℝ) : ℝ := a^3 - b

theorem tensor_identity (a : ℝ) : tensor a (tensor a (tensor a a)) = a^3 - a :=
by
  sorry

end tensor_identity_l29_29544


namespace f_of_integral_ratio_l29_29647

variable {f : ℝ → ℝ} (h_cont : ∀ x > 0, continuous_at f x)
variable (h_int : ∀ a b : ℝ, a > 0 → b > 0 → ∃ g : ℝ → ℝ, (∫ x in a..b, f x) = g (b / a))

theorem f_of_integral_ratio :
  (∃ c : ℝ, ∀ x > 0, f x = c / x) :=
sorry

end f_of_integral_ratio_l29_29647


namespace lcm_of_two_numbers_l29_29134

theorem lcm_of_two_numbers (A B : ℕ) 
  (h_prod : A * B = 987153000) 
  (h_hcf : Int.gcd A B = 440) : 
  Nat.lcm A B = 2243525 :=
by
  sorry

end lcm_of_two_numbers_l29_29134


namespace cos_30_deg_l29_29121

-- The condition implicitly includes the definition of cosine and the specific angle value

theorem cos_30_deg : cos (Real.pi / 6) = Real.sqrt 3 / 2 :=
by sorry

end cos_30_deg_l29_29121


namespace minimum_value_h_at_a_eq_2_range_of_a_l29_29297

noncomputable def f (a x : ℝ) : ℝ := a * x + (a - 1) / x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a x : ℝ) : ℝ := f a x - g x

theorem minimum_value_h_at_a_eq_2 : ∃ x, h 2 x = 3 := 
sorry

theorem range_of_a (a : ℝ) : (∀ x ≥ 1, h a x ≥ 1) ↔ a ≥ 1 :=
sorry

end minimum_value_h_at_a_eq_2_range_of_a_l29_29297


namespace clear_queue_with_three_windows_l29_29263

def time_to_clear_queue_one_window (a x y : ℕ) : Prop := a / (x - y) = 40

def time_to_clear_queue_two_windows (a x y : ℕ) : Prop := a / (2 * x - y) = 16

theorem clear_queue_with_three_windows (a x y : ℕ) 
  (h1 : time_to_clear_queue_one_window a x y) 
  (h2 : time_to_clear_queue_two_windows a x y) : 
  a / (3 * x - y) = 10 :=
by
  sorry

end clear_queue_with_three_windows_l29_29263


namespace chenny_friends_count_l29_29901

def initial_candies := 10
def additional_candies := 4
def candies_per_friend := 2

theorem chenny_friends_count :
  (initial_candies + additional_candies) / candies_per_friend = 7 := by
  sorry

end chenny_friends_count_l29_29901


namespace sheela_monthly_income_l29_29757

-- Definitions from the conditions
def deposited_amount : ℝ := 5000
def percentage_of_income : ℝ := 0.20

-- The theorem to be proven
theorem sheela_monthly_income : (deposited_amount / percentage_of_income) = 25000 := by
  sorry

end sheela_monthly_income_l29_29757


namespace bin_to_oct_l29_29681

theorem bin_to_oct (n : ℕ) (hn : n = 0b11010) : n = 0o32 := by
  sorry

end bin_to_oct_l29_29681


namespace range_of_a_in_quadratic_l29_29576

theorem range_of_a_in_quadratic :
  (∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ x1 ≠ x2 ∧ x1^2 + a * x1 - 2 = 0 ∧ x2^2 + a * x2 - 2 = 0) → -1 < a ∧ a < 1 :=
by
  sorry

end range_of_a_in_quadratic_l29_29576


namespace function_value_at_minus_two_l29_29298

theorem function_value_at_minus_two {f : ℝ → ℝ} (h : ∀ x : ℝ, x ≠ 0 → f (1/x) + (1/x) * f (-x) = 2 * x) : f (-2) = 7 / 2 :=
sorry

end function_value_at_minus_two_l29_29298


namespace find_N_l29_29449

theorem find_N (x N : ℝ) (h1 : x + 1 / x = N) (h2 : x^2 + 1 / x^2 = 2) : N = 2 :=
sorry

end find_N_l29_29449


namespace cards_selection_count_l29_29056

noncomputable def numberOfWaysToChooseCards : Nat :=
  (Nat.choose 4 3) * 3 * (Nat.choose 13 2) * (13 ^ 2)

theorem cards_selection_count :
  numberOfWaysToChooseCards = 158184 := by
  sorry

end cards_selection_count_l29_29056


namespace product_primes_less_than_20_l29_29115

theorem product_primes_less_than_20 :
  (2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 = 9699690) :=
by
  sorry

end product_primes_less_than_20_l29_29115


namespace find_a_l29_29049

noncomputable def f (a x : ℝ) : ℝ := a^x - 4 * a + 3

theorem find_a (H : ∃ (a : ℝ), ∃ (x y : ℝ), f a x = y ∧ f y x = a ∧ x = 2 ∧ y = -1): ∃ a : ℝ, a = 2 :=
by
  obtain ⟨a, x, y, hx, hy, hx2, hy1⟩ := H
  --skipped proof
  sorry

end find_a_l29_29049


namespace eggs_per_meal_l29_29085

noncomputable def initial_eggs_from_store : ℕ := 12
noncomputable def additional_eggs_from_neighbor : ℕ := 12
noncomputable def eggs_used_for_cooking : ℕ := 2 + 4
noncomputable def remaining_eggs_after_cooking : ℕ := initial_eggs_from_store + additional_eggs_from_neighbor - eggs_used_for_cooking
noncomputable def eggs_given_to_aunt : ℕ := remaining_eggs_after_cooking / 2
noncomputable def remaining_eggs_after_giving_to_aunt : ℕ := remaining_eggs_after_cooking - eggs_given_to_aunt
noncomputable def planned_meals : ℕ := 3

theorem eggs_per_meal : remaining_eggs_after_giving_to_aunt / planned_meals = 3 := 
by 
  sorry

end eggs_per_meal_l29_29085


namespace train_pass_time_l29_29892

noncomputable def train_speed_kmh := 36  -- Speed in km/hr
noncomputable def train_speed_ms := 10   -- Speed in m/s (converted)
noncomputable def platform_length := 180 -- Length of the platform in meters
noncomputable def platform_pass_time := 30 -- Time in seconds to pass platform
noncomputable def train_length := 120    -- Train length derived from conditions

theorem train_pass_time 
  (speed_in_kmh : ℕ) (speed_in_ms : ℕ) (platform_len : ℕ) (pass_platform_time : ℕ) (train_len : ℕ)
  (h1 : speed_in_kmh = 36)
  (h2 : speed_in_ms = 10)
  (h3 : platform_len = 180)
  (h4 : pass_platform_time = 30)
  (h5 : train_len = 120) :
  (train_len / speed_in_ms) = 12 := by
  sorry

end train_pass_time_l29_29892


namespace force_with_18_inch_crowbar_l29_29769

noncomputable def inverseForce (L F : ℝ) : ℝ :=
  F * L

theorem force_with_18_inch_crowbar :
  ∀ (F : ℝ), (inverseForce 12 200 = inverseForce 18 F) → F = 133.333333 :=
by
  intros
  sorry

end force_with_18_inch_crowbar_l29_29769


namespace GAUSS_1998_LCM_l29_29105

/-- The periodicity of cycling the word 'GAUSS' -/
def period_GAUSS : ℕ := 5

/-- The periodicity of cycling the number '1998' -/
def period_1998 : ℕ := 4

/-- The least common multiple (LCM) of the periodicities of 'GAUSS' and '1998' is 20 -/
theorem GAUSS_1998_LCM : Nat.lcm period_GAUSS period_1998 = 20 :=
by
  sorry

end GAUSS_1998_LCM_l29_29105


namespace smallest_digit_never_in_units_place_of_odd_number_l29_29790

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l29_29790


namespace gcf_270_108_150_l29_29639

theorem gcf_270_108_150 : Nat.gcd (Nat.gcd 270 108) 150 = 30 := 
  sorry

end gcf_270_108_150_l29_29639


namespace litter_patrol_total_pieces_l29_29994

theorem litter_patrol_total_pieces :
  let glass_bottles := 25
  let aluminum_cans := 18
  let plastic_bags := 12
  let paper_cups := 7
  let cigarette_packs := 5
  let discarded_face_masks := 3
  glass_bottles + aluminum_cans + plastic_bags + paper_cups + cigarette_packs + discarded_face_masks = 70 :=
by
  sorry

end litter_patrol_total_pieces_l29_29994


namespace inner_rectangle_length_l29_29260

theorem inner_rectangle_length 
  (a b c : ℝ)
  (h1 : ∃ a1 a2 a3 : ℝ, a2 - a1 = a3 - a2)
  (w_inner : ℝ)
  (width_inner : w_inner = 2)
  (w_shaded : ℝ)
  (width_shaded : w_shaded = 1.5)
  (ar_prog : a = 2 * w_inner ∧ b = 3 * w_inner + 15 ∧ c = 3 * w_inner + 33)
  : ∀ x : ℝ, 2 * x = a → 3 * x + 15 = b → 3 * x + 33 = c → x = 3 :=
by
  sorry

end inner_rectangle_length_l29_29260


namespace sector_angle_l29_29221

theorem sector_angle (l S : ℝ) (r α : ℝ) 
  (h_arc_length : l = 6)
  (h_area : S = 6)
  (h_area_formula : S = 1/2 * l * r)
  (h_arc_formula : l = r * α) : 
  α = 3 :=
by
  sorry

end sector_angle_l29_29221


namespace pages_with_same_units_digit_count_l29_29889

theorem pages_with_same_units_digit_count {n : ℕ} (h1 : n = 67) :
  ∃ k : ℕ, k = 13 ∧
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ n → 
    (x ≡ (n + 1 - x) [MOD 10] ↔ 
     (x % 10 = 4 ∨ x % 10 = 9))) :=
by
  sorry

end pages_with_same_units_digit_count_l29_29889


namespace corey_needs_more_golf_balls_l29_29908

-- Defining the constants based on the conditions
def goal : ℕ := 48
def found_on_saturday : ℕ := 16
def found_on_sunday : ℕ := 18

-- The number of golf balls Corey has found over the weekend
def total_found : ℕ := found_on_saturday + found_on_sunday

-- The number of golf balls Corey still needs to find to reach his goal
def remaining : ℕ := goal - total_found

-- The desired theorem statement
theorem corey_needs_more_golf_balls : remaining = 14 := 
by 
  sorry

end corey_needs_more_golf_balls_l29_29908


namespace initial_number_of_men_l29_29344

theorem initial_number_of_men (n : ℕ) (A : ℕ)
  (h1 : 2 * n = 16)
  (h2 : 60 - 44 = 16)
  (h3 : 60 = 2 * 30)
  (h4 : 44 = 21 + 23) :
  n = 8 :=
by
  sorry

end initial_number_of_men_l29_29344


namespace primes_product_less_than_20_l29_29117

-- Define the primes less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the product of a list of natural numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

theorem primes_product_less_than_20 :
  product primes_less_than_20 = 9699690 :=
by
  sorry

end primes_product_less_than_20_l29_29117


namespace range_of_m_l29_29293

theorem range_of_m (m : ℝ) (x : ℝ) 
  (h1 : ∀ x : ℝ, -1 < x ∧ x < 4 → x > 2 * m^2 - 3)
  (h2 : ¬ (∀ x : ℝ, x > 2 * m^2 - 3 → -1 < x ∧ x < 4))
  :
  -1 ≤ m ∧ m ≤ 1 :=
sorry

end range_of_m_l29_29293


namespace smallest_digit_not_in_units_place_of_odd_numbers_l29_29820

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l29_29820


namespace ceil_sqrt_225_eq_15_l29_29026

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := by
  sorry

end ceil_sqrt_225_eq_15_l29_29026


namespace identical_solutions_k_value_l29_29419

theorem identical_solutions_k_value (k : ℝ) :
  (∀ (x y : ℝ), y = x^2 ∧ y = 4 * x + k → (x - 2)^2 = 0) → k = -4 :=
by
  sorry

end identical_solutions_k_value_l29_29419


namespace sum_of_207_instances_of_33_difference_when_25_instances_of_112_are_subtracted_from_3000_difference_between_product_and_sum_of_12_and_13_l29_29271

-- 1. Prove that 33 * 207 = 6831
theorem sum_of_207_instances_of_33 : 33 * 207 = 6831 := by
    sorry

-- 2. Prove that 3000 - 112 * 25 = 200
theorem difference_when_25_instances_of_112_are_subtracted_from_3000 : 3000 - 112 * 25 = 200 := by
    sorry

-- 3. Prove that 12 * 13 - (12 + 13) = 131
theorem difference_between_product_and_sum_of_12_and_13 : 12 * 13 - (12 + 13) = 131 := by
    sorry

end sum_of_207_instances_of_33_difference_when_25_instances_of_112_are_subtracted_from_3000_difference_between_product_and_sum_of_12_and_13_l29_29271


namespace solve_equation_l29_29881

theorem solve_equation (x : ℝ) : 2 * x - 1 = 3 * x + 3 → x = -4 :=
by
  intro h
  sorry

end solve_equation_l29_29881


namespace cost_of_building_fence_eq_3944_l29_29122

def area_square : ℕ := 289
def price_per_foot : ℕ := 58

theorem cost_of_building_fence_eq_3944 : 
  let side_length := (area_square : ℝ) ^ (1/2)
  let perimeter := 4 * side_length
  let cost := perimeter * (price_per_foot : ℝ)
  cost = 3944 :=
by
  sorry

end cost_of_building_fence_eq_3944_l29_29122


namespace eval_expression_l29_29880

theorem eval_expression : 4^3 - 2 * 4^2 + 2 * 4 - 1 = 39 :=
by 
  -- Here we would write the proof, but according to the instructions we skip it with sorry.
  sorry

end eval_expression_l29_29880


namespace spherical_segment_equals_circle_area_l29_29631

noncomputable def spherical_segment_surface_area (R H : ℝ) : ℝ := 2 * Real.pi * R * H
noncomputable def circle_area (b : ℝ) : ℝ := Real.pi * (b * b)

theorem spherical_segment_equals_circle_area
  (R H b : ℝ) 
  (hb : b^2 = 2 * R * H) 
  : spherical_segment_surface_area R H = circle_area b :=
by
  sorry

end spherical_segment_equals_circle_area_l29_29631


namespace length_CD_l29_29213

theorem length_CD (AB AC BD CD : ℝ) (hAB : AB = 2) (hAC : AC = 5) (hBD : BD = 6) :
    CD = 3 :=
by
  sorry

end length_CD_l29_29213


namespace smallest_digit_never_at_units_place_of_odd_l29_29844

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l29_29844


namespace tg_pi_over_12_eq_exists_two_nums_l29_29139

noncomputable def tg (x : ℝ) := Real.tan x

theorem tg_pi_over_12_eq : tg (Real.pi / 12) = Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) :=
sorry

theorem exists_two_nums (a : Fin 13 → ℝ) (h_diff : Function.Injective a) :
  ∃ x y, 0 < (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) < Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) :=
sorry

end tg_pi_over_12_eq_exists_two_nums_l29_29139


namespace find_t_l29_29035

theorem find_t (t : ℝ) (h : (1 / (t + 2) + 2 * t / (t + 2) - 3 / (t + 2) = 3)) : t = -8 := 
by 
  sorry

end find_t_l29_29035


namespace minimum_reciprocal_sum_l29_29036

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) - 2

theorem minimum_reciprocal_sum (a m n : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (h₃ : f a (-1) = -1) (h₄ : m + n = 2) (h₅ : 0 < m) (h₆ : 0 < n) :
  (1 / m) + (1 / n) = 2 :=
by
  sorry

end minimum_reciprocal_sum_l29_29036


namespace find_A_l29_29995

variable (p q r s A : ℝ)

theorem find_A (H1 : (p + q + r + s) / 4 = 5) (H2 : (p + q + r + s + A) / 5 = 8) : A = 20 := 
by
  sorry

end find_A_l29_29995


namespace remaining_coins_denomination_l29_29632

def denomination_of_remaining_coins (total_coins : ℕ) (total_value : ℕ) (paise_20_count : ℕ) (paise_20_value : ℕ) : ℕ :=
  let remaining_coins := total_coins - paise_20_count
  let remaining_value := total_value - paise_20_count * paise_20_value
  remaining_value / remaining_coins

theorem remaining_coins_denomination :
  denomination_of_remaining_coins 334 7100 250 20 = 25 :=
by
  sorry

end remaining_coins_denomination_l29_29632


namespace triangle_side_length_l29_29735

theorem triangle_side_length (a b c x : ℕ) (A C : ℝ) (h1 : b = x) (h2 : a = x - 2) (h3 : c = x + 2)
  (h4 : C = 2 * A) (h5 : x + 2 = 10) : a = 8 :=
by
  sorry

end triangle_side_length_l29_29735


namespace simple_sampling_methods_l29_29626

theorem simple_sampling_methods :
  methods_of_implementing_simple_sampling = ["lottery method", "random number table method"] :=
sorry

end simple_sampling_methods_l29_29626


namespace sticks_per_chair_l29_29750

-- defining the necessary parameters and conditions
def sticksPerTable := 9
def sticksPerStool := 2
def sticksPerHour := 5
def chairsChopped := 18
def tablesChopped := 6
def stoolsChopped := 4
def hoursKeptWarm := 34

-- calculation of total sticks needed
def totalSticksNeeded := sticksPerHour * hoursKeptWarm

-- the main theorem to prove the number of sticks a chair makes
theorem sticks_per_chair (C : ℕ) : (chairsChopped * C) + (tablesChopped * sticksPerTable) + (stoolsChopped * sticksPerStool) = totalSticksNeeded → C = 6 := by
  sorry

end sticks_per_chair_l29_29750


namespace students_speak_both_l29_29732

theorem students_speak_both (total E T N : ℕ) (h1 : total = 150) (h2 : E = 55) (h3 : T = 85) (h4 : N = 30) :
  E + T - (total - N) = 20 := by
  -- Main proof logic
  sorry

end students_speak_both_l29_29732


namespace unique_solution_of_system_of_equations_l29_29616
open Set

variable {α : Type*} (A B X : Set α)

theorem unique_solution_of_system_of_equations :
  (X ∩ (A ∪ B) = X) ∧
  (A ∩ (B ∪ X) = A) ∧
  (B ∩ (A ∪ X) = B) ∧
  (X ∩ A ∩ B = ∅) →
  (X = (A \ B) ∪ (B \ A)) :=
by
  sorry

end unique_solution_of_system_of_equations_l29_29616


namespace board_arithmetic_impossibility_l29_29334

theorem board_arithmetic_impossibility :
  ¬ (∃ (a b : ℕ), a ≡ 0 [MOD 7] ∧ b ≡ 1 [MOD 7] ∧ (a * b + a^3 + b^3) = 2013201420152016) := 
    sorry

end board_arithmetic_impossibility_l29_29334


namespace expand_polynomial_l29_29160

theorem expand_polynomial (x : ℝ) : (5 * x + 3) * (6 * x ^ 2 + 2) = 30 * x ^ 3 + 18 * x ^ 2 + 10 * x + 6 :=
by
  sorry

end expand_polynomial_l29_29160


namespace greatest_of_consecutive_integers_l29_29249

theorem greatest_of_consecutive_integers (x y z : ℤ) (h1: y = x + 1) (h2: z = x + 2) (h3: x + y + z = 21) : z = 8 :=
by
  sorry

end greatest_of_consecutive_integers_l29_29249


namespace deposit_percentage_l29_29738

noncomputable def last_year_cost : ℝ := 250
noncomputable def increase_percentage : ℝ := 0.40
noncomputable def amount_paid_at_pickup : ℝ := 315
noncomputable def total_cost := last_year_cost * (1 + increase_percentage)
noncomputable def deposit := total_cost - amount_paid_at_pickup
noncomputable def percentage_deposit := deposit / total_cost * 100

theorem deposit_percentage :
  percentage_deposit = 10 := 
  by
    sorry

end deposit_percentage_l29_29738


namespace Maggie_earnings_l29_29980

theorem Maggie_earnings
    (price_per_subscription : ℕ)
    (subscriptions_parents : ℕ)
    (subscriptions_grandfather : ℕ)
    (subscriptions_nextdoor : ℕ)
    (subscriptions_another : ℕ)
    (total_subscriptions : ℕ)
    (total_earnings : ℕ) :
    subscriptions_parents = 4 →
    subscriptions_grandfather = 1 →
    subscriptions_nextdoor = 2 →
    subscriptions_another = 2 * subscriptions_nextdoor →
    total_subscriptions = subscriptions_parents + subscriptions_grandfather + subscriptions_nextdoor + subscriptions_another →
    price_per_subscription = 5 →
    total_earnings = price_per_subscription * total_subscriptions →
    total_earnings = 55 :=
by
  intros
  sorry

end Maggie_earnings_l29_29980


namespace smallest_digit_not_in_units_place_of_odd_l29_29813

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l29_29813


namespace find_constant_a_l29_29572

theorem find_constant_a :
  (∃ (a : ℝ), a > 0 ∧ (a + 2 * a + 3 * a + 4 * a = 1)) →
  ∃ (a : ℝ), a = 1 / 10 :=
sorry

end find_constant_a_l29_29572


namespace max_value_of_expression_l29_29202

noncomputable def max_expression_value (a b : ℝ) := a * b * (100 - 5 * a - 2 * b)

theorem max_value_of_expression :
  ∀ (a b : ℝ), 0 < a → 0 < b → 5 * a + 2 * b < 100 →
  max_expression_value a b ≤ 78125 / 36 := by
  intros a b ha hb h
  sorry

end max_value_of_expression_l29_29202


namespace negation_of_existence_statement_l29_29227

theorem negation_of_existence_statement :
  ¬ (∃ P : ℝ × ℝ, (P.1^2 + P.2^2 - 1 ≤ 0)) ↔ ∀ P : ℝ × ℝ, (P.1^2 + P.2^2 - 1 > 0) :=
by
  sorry

end negation_of_existence_statement_l29_29227


namespace smallest_n_l29_29876

theorem smallest_n (n : ℕ) (hn1 : (5 * n) pow 2) (hn2 : (4 * n) pow 3) : n = 80 :=
begin
  -- sorry statement to skip the proof.
  sorry
end

end smallest_n_l29_29876


namespace sum_of_squares_of_consecutive_integers_l29_29231

theorem sum_of_squares_of_consecutive_integers (a : ℕ) (h : (a - 1) * a * (a + 1) * (a + 2) = 12 * ((a - 1) + a + (a + 1) + (a + 2))) :
  (a - 1)^2 + a^2 + (a + 1)^2 + (a + 2)^2 = 86 :=
by
  sorry

end sum_of_squares_of_consecutive_integers_l29_29231


namespace hyperbola_eqn_correct_l29_29911

def parabola_focus : ℝ × ℝ := (1, 0)

def hyperbola_vertex := parabola_focus

def hyperbola_eccentricity : ℝ := 2

def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - (y^2 / 3) = 1

theorem hyperbola_eqn_correct (x y : ℝ) :
  hyperbola_equation x y :=
sorry

end hyperbola_eqn_correct_l29_29911


namespace remainder_correct_l29_29088

def dividend : ℝ := 13787
def divisor : ℝ := 154.75280898876406
def quotient : ℝ := 89
def remainder : ℝ := dividend - (divisor * quotient)

theorem remainder_correct: remainder = 14 := by
  -- Proof goes here
  sorry

end remainder_correct_l29_29088


namespace liam_comic_books_l29_29207

theorem liam_comic_books (cost_per_book : ℚ) (total_money : ℚ) (n : ℚ) : cost_per_book = 1.25 ∧ total_money = 10 → n = 8 :=
by
  intros h
  cases h
  have h1 : 1.25 * n ≤ 10 := by sorry
  have h2 : n ≤ 10 / 1.25 := by sorry
  have h3 : n ≤ 8 := by sorry
  have h4 : n = 8 := by sorry
  exact h4

end liam_comic_books_l29_29207


namespace polynomial_expansion_correct_l29_29408

open Polynomial

noncomputable def poly1 : Polynomial ℤ := X^2 + 3 * X - 4
noncomputable def poly2 : Polynomial ℤ := 2 * X^2 - X + 5
noncomputable def expected : Polynomial ℤ := 2 * X^4 + 5 * X^3 - 6 * X^2 + 19 * X - 20

theorem polynomial_expansion_correct :
  poly1 * poly2 = expected :=
sorry

end polynomial_expansion_correct_l29_29408


namespace verify_sum_of_fourth_powers_l29_29362

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def sum_of_fourth_powers (n : ℕ) : ℕ :=
  ((n * (n + 1) * (2 * n + 1) * (3 * n^2 + 3 * n - 1)) / 30)

noncomputable def square_of_sum (n : ℕ) : ℕ :=
  (n * (n + 1) / 2)^2

theorem verify_sum_of_fourth_powers (n : ℕ) :
  5 * sum_of_fourth_powers n = (4 * n + 2) * square_of_sum n - sum_of_squares n := 
  sorry

end verify_sum_of_fourth_powers_l29_29362


namespace time_for_six_visits_l29_29967

noncomputable def time_to_go_n_times (total_time : ℕ) (total_visits : ℕ) (n_visits : ℕ) : ℕ :=
  (total_time / total_visits) * n_visits

theorem time_for_six_visits (h : time_to_go_n_times 20 8 6 = 15) : time_to_go_n_times 20 8 6 = 15 :=
by
  exact h

end time_for_six_visits_l29_29967


namespace factorize_P_l29_29556

noncomputable def P (y : ℝ) : ℝ :=
  (16 * y ^ 7 - 36 * y ^ 5 + 8 * y) - (4 * y ^ 7 - 12 * y ^ 5 - 8 * y)

theorem factorize_P (y : ℝ) : P y = 8 * y * (3 * y ^ 6 - 6 * y ^ 4 + 4) :=
  sorry

end factorize_P_l29_29556


namespace polynomial_value_l29_29708

noncomputable def f : ℚ → ℚ := sorry  -- Definition of the polynomial function

theorem polynomial_value :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2013 → f k = 2 / k) → polynomial.degree f = 2012 →
  (2014 * f 2014 = 4) :=
begin
  intros h_keyvals h_degree,
  -- Construct the polynomial 
  let g := (λ x : ℚ, x * f x - 2),
  -- Prove that g(k) = 0 for k = 1, 2, ..., 2013
  have zeros_g : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 2013 → g k = 0,
  {
    intros k hk_range,
    specialize h_keyvals k hk_range,
    unfold g,
    rw h_keyvals,
    linarith,
  },
  -- essential properties of polynomial and degrees (to be shown in proof)
  sorry
end

end polynomial_value_l29_29708


namespace intersecting_graphs_l29_29624

theorem intersecting_graphs (a b c d : ℝ) 
  (h1 : -2 * |1 - a| + b = 4) 
  (h2 : 2 * |1 - c| + d = 4)
  (h3 : -2 * |7 - a| + b = 0) 
  (h4 : 2 * |7 - c| + d = 0) : a + c = 10 := 
sorry

end intersecting_graphs_l29_29624


namespace sum_odd_is_13_over_27_l29_29640

-- Define the probability for rolling an odd and an even number
def prob_odd := 1 / 3
def prob_even := 2 / 3

-- Define the probability that the sum of three die rolls is odd
def prob_sum_odd : ℚ :=
  3 * prob_odd * prob_even^2 + prob_odd^3

-- Statement asserting the goal to be proved
theorem sum_odd_is_13_over_27 :
  prob_sum_odd = 13 / 27 :=
by
  sorry

end sum_odd_is_13_over_27_l29_29640


namespace probability_of_multiples_of_3_or_5_l29_29113

open Finset

def is_multiple_of (a b: ℕ) : Prop := b % a == 0

def multiples_of_3_or_5 : Finset ℕ := (range 31).filter (λ n, is_multiple_of 3 n ∨ is_multiple_of 5 n)

theorem probability_of_multiples_of_3_or_5 : 
  (Finset.card (multiples_of_3_or_5).choose 2) / (Finset.card (range 31).choose 2) = (13 : ℚ) / 63 := 
by
  sorry

end probability_of_multiples_of_3_or_5_l29_29113


namespace smallest_digit_never_in_units_place_of_odd_numbers_l29_29786

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l29_29786


namespace find_number_l29_29125

theorem find_number (x : ℝ) (h : 7 * x + 21.28 = 50.68) : x = 4.2 :=
sorry

end find_number_l29_29125


namespace prob_all_pass_prob_at_least_one_pass_most_likely_event_l29_29592

noncomputable def probability_A := 2 / 5
noncomputable def probability_B := 3 / 4
noncomputable def probability_C := 1 / 3
noncomputable def prob_none_pass := (1 - probability_A) * (1 - probability_B) * (1 - probability_C)
noncomputable def prob_one_pass := 
  (probability_A * (1 - probability_B) * (1 - probability_C)) +
  ((1 - probability_A) * probability_B * (1 - probability_C)) +
  ((1 - probability_A) * (1 - probability_B) * probability_C)
noncomputable def prob_two_pass := 
  (probability_A * probability_B * (1 - probability_C)) +
  (probability_A * (1 - probability_B) * probability_C) +
  ((1 - probability_A) * probability_B * probability_C)

-- Prove that the probability that all three candidates pass is 1/10
theorem prob_all_pass : probability_A * probability_B * probability_C = 1 / 10 := by
  sorry

-- Prove that the probability that at least one candidate passes is 9/10
theorem prob_at_least_one_pass : 1 - prob_none_pass = 9 / 10 := by
  sorry

-- Prove that the most likely event of passing is exactly one candidate passing with probability 5/12
theorem most_likely_event : prob_one_pass > prob_two_pass ∧ prob_one_pass > probability_A * probability_B * probability_C ∧ prob_one_pass > prob_none_pass ∧ prob_one_pass = 5 / 12 := by
  sorry

end prob_all_pass_prob_at_least_one_pass_most_likely_event_l29_29592


namespace relationship_of_y_values_l29_29068

theorem relationship_of_y_values (m n y1 y2 y3 : ℝ) (h1 : m < 0) (h2 : n > 0) 
  (hA : y1 = m * (-2) + n) (hB : y2 = m * (-3) + n) (hC : y3 = m * 1 + n) :
  y3 < y1 ∧ y1 < y2 := 
by 
  sorry

end relationship_of_y_values_l29_29068


namespace average_after_17th_inning_l29_29143

theorem average_after_17th_inning (A : ℝ) (total_runs_16th_inning : ℝ) 
  (average_before_17th : A * 16 = total_runs_16th_inning) 
  (increased_average_by_3 : (total_runs_16th_inning + 83) / 17 = A + 3) :
  (A + 3) = 35 := 
sorry

end average_after_17th_inning_l29_29143


namespace symmetric_points_x_axis_l29_29425

theorem symmetric_points_x_axis (a b : ℝ) (h_a : a = -2) (h_b : b = -1) : a + b = -3 :=
by
  -- Skipping the proof steps and adding sorry
  sorry

end symmetric_points_x_axis_l29_29425


namespace problem_statement_l29_29578

open Set

def M : Set ℝ := {x | x^2 - 2008 * x - 2009 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}

theorem problem_statement (a b : ℝ) :
  (M ∪ N a b = univ) →
  (M ∩ N a b = {x | 2009 < x ∧ x ≤ 2010}) →
  (a = 2009 ∧ b = 2010) :=
by
  sorry

end problem_statement_l29_29578


namespace math_problem_l29_29947

theorem math_problem (x t : ℝ) (h1 : 6 * x + t = 4 * x - 9) (h2 : t = 7) : x + 4 = -4 := by
  sorry

end math_problem_l29_29947


namespace find_p_l29_29584

theorem find_p (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : p = 52 / 11 :=
by {
  -- Proof steps would go here
  sorry
}

end find_p_l29_29584


namespace correct_operation_l29_29508

theorem correct_operation (a : ℝ) :
  (2 * a^2) * a = 2 * a^3 :=
by sorry

end correct_operation_l29_29508


namespace obtain_2015_in_4_operations_obtain_2015_in_3_operations_l29_29634

-- Define what an operation is
def operation (cards : List ℕ) : List ℕ :=
  sorry  -- Implementation of this is unnecessary for the statement

-- Check if 2015 can be obtained in 4 operations
def can_obtain_2015_in_4_operations (initial_cards : List ℕ) : Prop :=
  ∃ cards, (operation^[4] initial_cards) = cards ∧ 2015 ∈ cards

-- Check if 2015 can be obtained in 3 operations
def can_obtain_2015_in_3_operations (initial_cards : List ℕ) : Prop :=
  ∃ cards, (operation^[3] initial_cards) = cards ∧ 2015 ∈ cards

theorem obtain_2015_in_4_operations :
  can_obtain_2015_in_4_operations [1, 2] :=
sorry

theorem obtain_2015_in_3_operations :
  can_obtain_2015_in_3_operations [1, 2] :=
sorry

end obtain_2015_in_4_operations_obtain_2015_in_3_operations_l29_29634


namespace find_divisor_l29_29503

theorem find_divisor : 
  ∀ (dividend quotient remainder divisor : ℕ), 
    dividend = 140 →
    quotient = 9 →
    remainder = 5 →
    dividend = (divisor * quotient) + remainder →
    divisor = 15 :=
by
  intros dividend quotient remainder divisor hd hq hr hdiv
  sorry

end find_divisor_l29_29503


namespace train_seats_count_l29_29262

theorem train_seats_count 
  (Standard Comfort Premium : ℝ)
  (Total_SEATS : ℝ)
  (hs : Standard = 36)
  (hc : Comfort = 0.20 * Total_SEATS)
  (hp : Premium = (3/5) * Total_SEATS)
  (ht : Standard + Comfort + Premium = Total_SEATS) :
  Total_SEATS = 180 := sorry

end train_seats_count_l29_29262


namespace remaining_payment_l29_29305

theorem remaining_payment (deposit_percent : ℝ) (deposit_amount : ℝ) (total_percent : ℝ) (total_price : ℝ) :
  deposit_percent = 5 ∧ deposit_amount = 50 ∧ total_percent = 100 → total_price - deposit_amount = 950 :=
by {
  sorry
}

end remaining_payment_l29_29305


namespace coeff_sum_l29_29182

theorem coeff_sum (a : ℕ → ℤ) (x : ℤ) :
    (3 - (1 + x))^7 = ∑ k in (Finset.range 8), a k * (1 + x)^k → 
    (∑ k in (Finset.range 7), a k) = 129 :=
by
  sorry

end coeff_sum_l29_29182


namespace max_value_of_symmetric_f_l29_29722

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_f :
  ∀ (a b : ℝ),
    (f 1 a b = 0) →
    (f (-1) a b = 0) →
    (f (-5) a b = 0) →
    (f (-3) a b = 0) →
    (∃ x : ℝ, f x 8 15 = 16) :=
by
  sorry

end max_value_of_symmetric_f_l29_29722


namespace num_men_scenario1_is_15_l29_29447

-- Definitions based on the conditions
def hours_per_day_scenario1 : ℕ := 9
def days_scenario1 : ℕ := 16
def men_scenario2 : ℕ := 18
def hours_per_day_scenario2 : ℕ := 8
def days_scenario2 : ℕ := 15
def total_work_done : ℕ := men_scenario2 * hours_per_day_scenario2 * days_scenario2

-- Definition of the number of men M in the first scenario
noncomputable def men_scenario1 : ℕ := total_work_done / (hours_per_day_scenario1 * days_scenario1)

-- Statement of desired proof: prove that the number of men in the first scenario is 15
theorem num_men_scenario1_is_15 :
  men_scenario1 = 15 := by
  sorry

end num_men_scenario1_is_15_l29_29447


namespace Mike_books_l29_29751

theorem Mike_books
  (initial_books : ℝ)
  (books_sold : ℝ)
  (books_gifts : ℝ) 
  (books_bought : ℝ)
  (h_initial : initial_books = 51.5)
  (h_sold : books_sold = 45.75)
  (h_gifts : books_gifts = 12.25)
  (h_bought : books_bought = 3.5):
  initial_books - books_sold + books_gifts + books_bought = 21.5 := 
sorry

end Mike_books_l29_29751


namespace Kevin_lost_cards_l29_29199

theorem Kevin_lost_cards (initial_cards final_cards : ℝ) (h1 : initial_cards = 47.0) (h2 : final_cards = 40) :
  initial_cards - final_cards = 7 :=
by
  sorry

end Kevin_lost_cards_l29_29199


namespace Xingyou_age_is_3_l29_29595

theorem Xingyou_age_is_3 (x : ℕ) (h1 : x = x) (h2 : x + 3 = 2 * x) : x = 3 :=
by
  sorry

end Xingyou_age_is_3_l29_29595


namespace area_of_rectangle_l29_29411

def length : ℕ := 4
def width : ℕ := 2

theorem area_of_rectangle : length * width = 8 :=
by
  sorry

end area_of_rectangle_l29_29411


namespace smallest_digit_never_in_units_place_of_odd_l29_29834

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l29_29834


namespace sufficient_but_not_necessary_condition_l29_29593

theorem sufficient_but_not_necessary_condition (b : ℝ) :
  (∀ x : ℝ, b * x^2 - b * x + 1 > 0) ↔ (b = 0 ∨ (0 < b ∧ b < 4)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l29_29593


namespace total_tickets_sold_l29_29524

/-
Problem: Prove that the total number of tickets sold is 65 given the conditions.
Conditions:
1. Senior citizen tickets cost 10 dollars each.
2. Regular tickets cost 15 dollars each.
3. Total sales were 855 dollars.
4. 24 senior citizen tickets were sold.
-/

def senior_tickets_sold : ℕ := 24
def senior_ticket_cost : ℕ := 10
def regular_ticket_cost : ℕ := 15
def total_sales : ℕ := 855

theorem total_tickets_sold (R : ℕ) (H : total_sales = senior_tickets_sold * senior_ticket_cost + R * regular_ticket_cost) :
  senior_tickets_sold + R = 65 :=
by
  sorry

end total_tickets_sold_l29_29524


namespace inequality_proof_l29_29429

variables (a b c d : ℝ)

theorem inequality_proof (h1 : a > b) (h2 : c = d) : a + c > b + d :=
sorry

end inequality_proof_l29_29429


namespace polynomial_root_abs_sum_eq_80_l29_29924

theorem polynomial_root_abs_sum_eq_80 (a b c : ℤ) (m : ℤ) 
  (h1 : a + b + c = 0) 
  (h2 : ab + bc + ac = -2023) 
  (h3 : ∃ m, ∀ x : ℤ, x^3 - 2023 * x + m = (x - a) * (x - b) * (x - c)) : 
  |a| + |b| + |c| = 80 := 
by {
  sorry
}

end polynomial_root_abs_sum_eq_80_l29_29924


namespace s_point_condition_l29_29603

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x)

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 2 * a * x
noncomputable def g_prime (a : ℝ) (x : ℝ) : ℝ := 1 / x

theorem s_point_condition (a : ℝ) (x₀ : ℝ) (h_f_g : f a x₀ = g a x₀) (h_f'g' : f_prime a x₀ = g_prime a x₀) :
  a = 2 / Real.exp 1 :=
by
  sorry

end s_point_condition_l29_29603


namespace rhombus_diagonal_difference_l29_29158

theorem rhombus_diagonal_difference (a d : ℝ) (h_a_pos : a > 0) (h_d_pos : d > 0):
  (∃ (e f : ℝ), e > f ∧ e - f = d ∧ a^2 = (e/2)^2 + (f/2)^2) ↔ d < 2 * a :=
sorry

end rhombus_diagonal_difference_l29_29158


namespace cross_fills_space_without_gaps_l29_29420

structure Cube :=
(x : ℤ)
(y : ℤ)
(z : ℤ)

structure Cross :=
(center : Cube)
(adjacent : List Cube)

def is_adjacent (c1 c2 : Cube) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ abs (c1.z - c2.z) = 1) ∨
  (c1.x = c2.x ∧ abs (c1.y - c2.y) = 1 ∧ c1.z = c2.z) ∨
  (abs (c1.x - c2.x) = 1 ∧ c1.y = c2.y ∧ c1.z = c2.z)

def valid_cross (c : Cross) : Prop :=
  ∀ (adj : Cube), adj ∈ c.adjacent → is_adjacent c.center adj

def fills_space (crosses : List Cross) : Prop :=
  ∀ (pos : Cube), ∃ (c : Cross), c ∈ crosses ∧ 
    (pos = c.center ∨ pos ∈ c.adjacent)

theorem cross_fills_space_without_gaps 
  (crosses : List Cross) 
  (Hcross : ∀ c ∈ crosses, valid_cross c) : 
  fills_space crosses :=
sorry

end cross_fills_space_without_gaps_l29_29420


namespace balloons_lost_l29_29198

theorem balloons_lost (initial remaining : ℕ) (h_initial : initial = 9) (h_remaining : remaining = 7) : initial - remaining = 2 := by
  sorry

end balloons_lost_l29_29198


namespace matrix_pow_C_50_l29_29319

def C : Matrix (Fin 2) (Fin 2) ℤ := 
  !![3, 1; -4, -1]

theorem matrix_pow_C_50 : C^50 = !![101, 50; -200, -99] := 
  sorry

end matrix_pow_C_50_l29_29319


namespace minimum_value_l29_29606

noncomputable theory

open_locale big_operators

theorem minimum_value (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 8) : 
  ∃ m : ℝ, m = 64 ∧ (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ m :=
by
  sorry

end minimum_value_l29_29606


namespace car_b_speed_l29_29377

/--
A car A going at 30 miles per hour set out on an 80-mile trip at 9:00 a.m.
Exactly 10 minutes later, a car B left from the same place and followed the same route.
Car B caught up with car A at 10:30 a.m.
Prove that the speed of car B is 33.75 miles per hour.
-/
theorem car_b_speed
    (v_a : ℝ) (t_start_a t_start_b t_end : ℝ) (v_b : ℝ)
    (h1 : v_a = 30) 
    (h2 : t_start_a = 9) 
    (h3 : t_start_b = 9 + (10 / 60)) 
    (h4 : t_end = 10.5) 
    (h5 : t_end - t_start_b = (4 / 3))
    (h6 : v_b * (t_end - t_start_b) = v_a * (t_end - t_start_a) + (v_a * (10 / 60))) :
  v_b = 33.75 := 
sorry

end car_b_speed_l29_29377


namespace atomic_weight_chlorine_l29_29109

-- Define the given conditions and constants
def molecular_weight_compound : ℝ := 53
def atomic_weight_nitrogen : ℝ := 14.01
def atomic_weight_hydrogen : ℝ := 1.01
def number_of_hydrogen_atoms : ℝ := 4
def number_of_nitrogen_atoms : ℝ := 1

-- Define the total weight of nitrogen and hydrogen in the compound
def total_weight_nh : ℝ := (number_of_nitrogen_atoms * atomic_weight_nitrogen) + (number_of_hydrogen_atoms * atomic_weight_hydrogen)

-- Define the statement to be proved: the atomic weight of chlorine
theorem atomic_weight_chlorine : (molecular_weight_compound - total_weight_nh) = 34.95 := by
  sorry

end atomic_weight_chlorine_l29_29109


namespace determine_m_l29_29469

theorem determine_m (a b c m : ℤ) 
  (h1 : c = -4 * a - 2 * b)
  (h2 : 70 < 4 * (8 * a + b) ∧ 4 * (8 * a + b) < 80)
  (h3 : 110 < 5 * (9 * a + b) ∧ 5 * (9 * a + b) < 120)
  (h4 : 2000 * m < (2500 * a + 50 * b + c) ∧ (2500 * a + 50 * b + c) < 2000 * (m + 1)) :
  m = 5 := sorry

end determine_m_l29_29469


namespace smallest_digit_never_in_units_place_of_odd_number_l29_29792

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l29_29792


namespace maggie_earnings_proof_l29_29984

def rate_per_subscription : ℕ := 5
def subscriptions_to_parents : ℕ := 4
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_neighbor1 : ℕ := 2
def subscriptions_to_neighbor2 : ℕ := 2 * subscriptions_to_neighbor1
def total_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_neighbor1 + subscriptions_to_neighbor2
def total_earnings : ℕ := total_subscriptions * rate_per_subscription

theorem maggie_earnings_proof : total_earnings = 55 := by
  sorry

end maggie_earnings_proof_l29_29984


namespace pi_irrational_l29_29641

def rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem pi_irrational : ¬ rational π :=
sorry

end pi_irrational_l29_29641


namespace milk_concentration_l29_29913

variable {V_initial V_removed V_total : ℝ}

theorem milk_concentration (h1 : V_initial = 20) (h2 : V_removed = 2) (h3 : V_total = 20) :
    (V_initial - V_removed) / V_total * 100 = 90 := 
by 
  sorry

end milk_concentration_l29_29913


namespace possible_denominators_count_l29_29341

theorem possible_denominators_count :
  ∀ a b c : ℕ, 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ (a ≠ 9 ∨ b ≠ 9 ∨ c ≠ 9) →
  ∃ (D : Finset ℕ), D.card = 7 ∧ 
  ∀ num denom, (num = 100*a + 10*b + c) → (denom = 999) → (gcd num denom > 1) → 
  denom ∈ D := 
sorry

end possible_denominators_count_l29_29341


namespace pear_price_is_6300_l29_29234

def price_of_pear (P : ℕ) : Prop :=
  P + (P + 2400) = 15000

theorem pear_price_is_6300 : ∃ (P : ℕ), price_of_pear P ∧ P = 6300 :=
by
  sorry

end pear_price_is_6300_l29_29234


namespace sin_theta_value_l29_29705

theorem sin_theta_value (θ : ℝ) (h1 : 10 * (Real.tan θ) = 4 * (Real.cos θ)) (h2 : 0 < θ ∧ θ < π) : Real.sin θ = 1/2 :=
by
  sorry

end sin_theta_value_l29_29705


namespace common_difference_of_arithmetic_sequence_l29_29567

/--
Given an arithmetic sequence {a_n}, the sum of the first n terms is S_n,
a_3 and a_7 are the two roots of the equation 2x^2 - 12x + c = 0,
and S_{13} = c.
Prove that the common difference of the sequence {a_n} satisfies d = -3/2 or d = -7/4.
-/
theorem common_difference_of_arithmetic_sequence 
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (c : ℚ)
  (h1 : ∃ a_3 a_7, (2 * a_3^2 - 12 * a_3 + c = 0) ∧ (2 * a_7^2 - 12 * a_7 + c = 0))
  (h2 : S 13 = c) :
  ∃ d : ℚ, d = -3/2 ∨ d = -7/4 :=
sorry

end common_difference_of_arithmetic_sequence_l29_29567


namespace train_people_count_l29_29777

theorem train_people_count :
  let initial := 48
  let after_first_stop := initial - 13 + 5
  let after_second_stop := after_first_stop - 9 + 10 - 2
  let after_third_stop := after_second_stop - 7 + 4 - 3
  let after_fourth_stop := after_third_stop - 16 + 7 - 5
  let after_fifth_stop := after_fourth_stop - 8 + 15
  after_fifth_stop = 26 := sorry

end train_people_count_l29_29777


namespace combined_height_after_1_year_l29_29749

def initial_heights : ℕ := 200 + 150 + 250
def spring_and_summer_growth_A : ℕ := (6 * 4 / 2) * 50
def spring_and_summer_growth_B : ℕ := (6 * 4 / 3) * 70
def spring_and_summer_growth_C : ℕ := (6 * 4 / 4) * 90
def autumn_and_winter_growth_A : ℕ := (6 * 4 / 2) * 25
def autumn_and_winter_growth_B : ℕ := (6 * 4 / 3) * 35
def autumn_and_winter_growth_C : ℕ := (6 * 4 / 4) * 45

def total_growth_A : ℕ := spring_and_summer_growth_A + autumn_and_winter_growth_A
def total_growth_B : ℕ := spring_and_summer_growth_B + autumn_and_winter_growth_B
def total_growth_C : ℕ := spring_and_summer_growth_C + autumn_and_winter_growth_C

def total_growth : ℕ := total_growth_A + total_growth_B + total_growth_C

def combined_height : ℕ := initial_heights + total_growth

theorem combined_height_after_1_year : combined_height = 3150 := by
  sorry

end combined_height_after_1_year_l29_29749


namespace jill_total_watch_time_l29_29596

theorem jill_total_watch_time :
  ∀ (length_first_show length_second_show total_watch_time : ℕ),
    length_first_show = 30 →
    length_second_show = 4 * length_first_show →
    total_watch_time = length_first_show + length_second_show →
    total_watch_time = 150 :=
by
  sorry

end jill_total_watch_time_l29_29596


namespace polygon_interior_angles_sum_l29_29345

theorem polygon_interior_angles_sum (n : ℕ) (hn : 180 * (n - 2) = 1980) : 180 * (n + 4 - 2) = 2700 :=
by
  sorry

end polygon_interior_angles_sum_l29_29345


namespace smallest_digit_not_in_units_place_of_odd_l29_29796

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l29_29796


namespace average_payment_l29_29883

theorem average_payment (n m : ℕ) (p1 p2 : ℕ) (h1 : n = 20) (h2 : m = 45) (h3 : p1 = 410) (h4 : p2 = 475) :
  (20 * p1 + 45 * p2) / 65 = 455 :=
by
  sorry

end average_payment_l29_29883


namespace find_d_h_l29_29494

theorem find_d_h (a b c d g h : ℂ) (h1 : b = 4) (h2 : g = -a - c) (h3 : a + c + g = 0) (h4 : b + d + h = 3) : 
  d + h = -1 := 
by
  sorry

end find_d_h_l29_29494


namespace range_of_a_l29_29307

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, (2 * (x : ℝ) - 7 < 0) ∧ ((x : ℝ) - a > 0) ↔ (x = 3)) →
  (2 ≤ a ∧ a < 3) :=
by
  intro h
  sorry

end range_of_a_l29_29307


namespace theorem1_theorem2_theorem3_l29_29470

-- Given conditions as definitions
variables {x y p q : ℝ}

-- Condition definitions
def condition1 : x + y = -p := sorry
def condition2 : x * y = q := sorry

-- Theorems to be proved
theorem theorem1 (h1 : x + y = -p) (h2 : x * y = q) : x^2 + y^2 = p^2 - 2 * q := sorry

theorem theorem2 (h1 : x + y = -p) (h2 : x * y = q) : x^3 + y^3 = -p^3 + 3 * p * q := sorry

theorem theorem3 (h1 : x + y = -p) (h2 : x * y = q) : x^4 + y^4 = p^4 - 4 * p^2 * q + 2 * q^2 := sorry

end theorem1_theorem2_theorem3_l29_29470


namespace num_white_balls_l29_29520

theorem num_white_balls (W : ℕ) (h : (W : ℝ) / (6 + W) = 0.45454545454545453) : W = 5 :=
by
  sorry

end num_white_balls_l29_29520


namespace gcd_of_repeated_three_digit_l29_29529

theorem gcd_of_repeated_three_digit : 
  ∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → ∀ m ∈ {k : ℕ | ∃ n, 100 ≤ n ∧ n < 1000 ∧ k = 1001 * n}, Nat.gcd 1001 m = 1001 :=
by
  sorry

end gcd_of_repeated_three_digit_l29_29529


namespace min_ratio_cyl_inscribed_in_sphere_l29_29151

noncomputable def min_surface_area_to_volume_ratio (R r : ℝ) : ℝ :=
  let h := 2 * Real.sqrt (R^2 - r^2)
  let A := 2 * Real.pi * r * (h + r)
  let V := Real.pi * r^2 * h
  A / V

theorem min_ratio_cyl_inscribed_in_sphere (R : ℝ) :
  ∃ r h, h = 2 * Real.sqrt (R^2 - r^2) ∧
         min_surface_area_to_volume_ratio R r = (Real.sqrt (Real.sqrt 4 + 1))^3 / R := 
by {
  sorry
}

end min_ratio_cyl_inscribed_in_sphere_l29_29151


namespace max_area_of_rectangular_garden_l29_29148

noncomputable def max_rectangle_area (x y : ℝ) (h1 : 2 * (x + y) = 36) (h2 : x > 0) (h3 : y > 0) : ℝ :=
  x * y

theorem max_area_of_rectangular_garden
  (x y : ℝ)
  (h1 : 2 * (x + y) = 36)
  (h2 : x > 0)
  (h3 : y > 0) :
  max_rectangle_area x y h1 h2 h3 = 81 :=
sorry

end max_area_of_rectangular_garden_l29_29148


namespace miles_per_gallon_city_l29_29651

theorem miles_per_gallon_city
  (T : ℝ) -- tank size
  (h c : ℝ) -- miles per gallon on highway 'h' and in the city 'c'
  (h_eq : h = (462 / T))
  (c_eq : c = (336 / T))
  (relation : c = h - 9)
  (solution : c = 24) : c = 24 := 
sorry

end miles_per_gallon_city_l29_29651


namespace find_smaller_number_l29_29373

theorem find_smaller_number (x y : ℕ) (h1 : y = 3 * x) (h2 : x + y = 124) : x = 31 := 
by 
  -- Proof would go here
  sorry

end find_smaller_number_l29_29373


namespace sum_of_digits_l29_29306

theorem sum_of_digits (a b c d : ℕ) (h1 : a + c = 11) (h2 : b + c = 9) (h3 : a + d = 10) (h_d : d - c = 1) : 
  a + b + c + d = 21 :=
sorry

end sum_of_digits_l29_29306


namespace dean_taller_than_ron_l29_29150

theorem dean_taller_than_ron (d h r : ℕ) (h1 : d = 15 * h) (h2 : r = 13) (h3 : d = 255) : h - r = 4 := 
by 
  sorry

end dean_taller_than_ron_l29_29150


namespace max_x2_y2_on_circle_l29_29423

noncomputable def max_value_on_circle : ℝ :=
  12 + 8 * Real.sqrt 2

theorem max_x2_y2_on_circle (x y : ℝ) (h : x^2 - 4 * x - 4 + y^2 = 0) : 
  x^2 + y^2 ≤ max_value_on_circle := 
by
  sorry

end max_x2_y2_on_circle_l29_29423


namespace same_terminal_side_l29_29535

theorem same_terminal_side (k : ℤ) : 
  ∃ (α : ℤ), α = k * 360 + 330 ∧ (α = 510 ∨ α = 150 ∨ α = -150 ∨ α = -390) :=
by
  sorry

end same_terminal_side_l29_29535


namespace smallest_digit_never_in_units_place_of_odd_number_l29_29867

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l29_29867


namespace pool_cannot_be_filled_l29_29658

noncomputable def pool := 48000 -- Pool capacity in gallons
noncomputable def hose_rate := 3 -- Rate of each hose in gallons per minute
noncomputable def number_of_hoses := 6 -- Number of hoses
noncomputable def leakage_rate := 18 -- Leakage rate in gallons per minute

theorem pool_cannot_be_filled : 
  (number_of_hoses * hose_rate - leakage_rate <= 0) -> False :=
by
  -- Skipping the proof with 'sorry' as per instructions
  sorry

end pool_cannot_be_filled_l29_29658


namespace josh_total_money_left_l29_29598

-- Definitions of the conditions
def profit_per_bracelet : ℝ := 1.5 - 1
def total_bracelets : ℕ := 12
def cost_of_cookies : ℝ := 3

-- The proof problem: 
theorem josh_total_money_left : total_bracelets * profit_per_bracelet - cost_of_cookies = 3 :=
by
  sorry

end josh_total_money_left_l29_29598


namespace minimize_squares_in_rectangle_l29_29365

theorem minimize_squares_in_rectangle (w h : ℕ) (hw : w = 63) (hh : h = 42) : 
  ∃ s : ℕ, s = Nat.gcd w h ∧ s = 21 :=
by
  sorry

end minimize_squares_in_rectangle_l29_29365


namespace smallest_digit_never_in_units_place_of_odd_l29_29830

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l29_29830


namespace percentage_of_full_marks_D_l29_29133

theorem percentage_of_full_marks_D (full_marks a b c d : ℝ)
  (h_full_marks : full_marks = 500)
  (h_a : a = 360)
  (h_a_b : a = b - 0.10 * b)
  (h_b_c : b = c + 0.25 * c)
  (h_c_d : c = d - 0.20 * d) :
  d / full_marks * 100 = 80 :=
by
  sorry

end percentage_of_full_marks_D_l29_29133


namespace white_pairs_coincide_l29_29405

theorem white_pairs_coincide :
  ∀ (red_triangles blue_triangles white_triangles : ℕ)
    (red_pairs blue_pairs red_blue_pairs : ℕ),
  red_triangles = 4 →
  blue_triangles = 4 →
  white_triangles = 6 →
  red_pairs = 3 →
  blue_pairs = 2 →
  red_blue_pairs = 1 →
  (2 * white_triangles - red_triangles - blue_triangles - red_blue_pairs) = white_triangles →
  6 = white_triangles :=
by
  intros red_triangles blue_triangles white_triangles
         red_pairs blue_pairs red_blue_pairs
         H_red H_blue H_white
         H_red_pairs H_blue_pairs H_red_blue_pairs
         H_pairs
  sorry

end white_pairs_coincide_l29_29405


namespace terrell_lifting_problem_l29_29764

theorem terrell_lifting_problem (w1 w2 w3 n1 n2 : ℕ) (h1 : w1 = 12) (h2 : w2 = 18) (h3 : w3 = 24) (h4 : n1 = 20) :
  60 * n2 = 3 * w1 * n1 → n2 = 12 :=
by
  intros h
  sorry

end terrell_lifting_problem_l29_29764


namespace translate_upwards_l29_29194

theorem translate_upwards (x : ℝ) : (2 * x^2) + 2 = 2 * x^2 + 2 := by
  sorry

end translate_upwards_l29_29194


namespace game_ends_in_36_rounds_l29_29381

theorem game_ends_in_36_rounds 
    (tokens_A : ℕ := 17) (tokens_B : ℕ := 16) (tokens_C : ℕ := 15)
    (rounds : ℕ) 
    (game_rule : (tokens_A tokens_B tokens_C round_num : ℕ) → Prop) 
    (extra_discard_rule : (tokens_A tokens_B tokens_C round_num : ℕ) → Prop)  
    (game_ends_when_token_zero : (tokens_A tokens_B tokens_C : ℕ) → Prop) :
    game_rule tokens_A tokens_B tokens_C rounds ∧
    extra_discard_rule tokens_A tokens_B tokens_C rounds ∧
    game_ends_when_token_zero tokens_A tokens_B tokens_C → 
    rounds = 36 := by
    sorry

end game_ends_in_36_rounds_l29_29381


namespace find_two_digit_number_l29_29955

-- Define the problem conditions and statement
theorem find_two_digit_number (a b n : ℕ) (h1 : a = 2 * b) (h2 : 10 * a + b + a^2 = n^2) : 
  10 * a + b = 21 :=
sorry

end find_two_digit_number_l29_29955


namespace sales_in_fifth_month_l29_29257

-- Define the sales figures and average target
def s1 : ℕ := 6435
def s2 : ℕ := 6927
def s3 : ℕ := 6855
def s4 : ℕ := 7230
def s6 : ℕ := 6191
def s_target : ℕ := 6700
def n_months : ℕ := 6

-- Define the total sales and the required fifth month sale
def total_sales : ℕ := s_target * n_months
def s5 : ℕ := total_sales - (s1 + s2 + s3 + s4 + s6)

-- The main theorem statement we need to prove
theorem sales_in_fifth_month :
  s5 = 6562 :=
sorry

end sales_in_fifth_month_l29_29257


namespace hyperbola_asymptote_slope_l29_29490

theorem hyperbola_asymptote_slope :
  (∃ m : ℚ, m > 0 ∧ ∀ x : ℚ, ∀ y : ℚ, ((x*x/16 - y*y/25 = 1) → (y = m * x ∨ y = -m * x))) → m = 5/4 :=
sorry

end hyperbola_asymptote_slope_l29_29490


namespace simplify_fraction_l29_29482

theorem simplify_fraction (x : ℝ) :
  ((x + 2) / 4) + ((3 - 4 * x) / 3) = (18 - 13 * x) / 12 := by
  sorry

end simplify_fraction_l29_29482


namespace tangent_line_circle_sol_l29_29060

theorem tangent_line_circle_sol (r : ℝ) (h_pos : r > 0)
  (h_tangent : ∀ x y : ℝ, x^2 + y^2 = 2 * r → x + 2 * y = r) : r = 10 := 
sorry

end tangent_line_circle_sol_l29_29060


namespace abc_eq_zero_l29_29971

variable (a b c : ℝ) (n : ℕ)

theorem abc_eq_zero
  (h1 : a^n + b^n = c^n)
  (h2 : a^(n+1) + b^(n+1) = c^(n+1))
  (h3 : a^(n+2) + b^(n+2) = c^(n+2)) :
  a * b * c = 0 :=
sorry

end abc_eq_zero_l29_29971


namespace linemen_count_l29_29522

-- Define the initial conditions
def linemen_drink := 8
def skill_position_players_drink := 6
def total_skill_position_players := 10
def cooler_capacity := 126
def skill_position_players_drink_first := 5

-- Define the number of ounces drunk by skill position players during the first break
def skill_position_players_first_break := skill_position_players_drink_first * skill_position_players_drink

-- Define the theorem stating that the number of linemen (L) is 12 given the conditions
theorem linemen_count :
  ∃ L : ℕ, linemen_drink * L + skill_position_players_first_break = cooler_capacity ∧ L = 12 :=
by {
  sorry -- Proof to be provided.
}

end linemen_count_l29_29522


namespace cost_of_article_l29_29183

-- Definitions for conditions
def gain_340 (C G : ℝ) : Prop := 340 = C + G
def gain_360 (C G : ℝ) : Prop := 360 = C + G + 0.05 * C

-- Theorem to be proven
theorem cost_of_article (C G : ℝ) (h1 : gain_340 C G) (h2 : gain_360 C G) : C = 400 :=
by sorry

end cost_of_article_l29_29183


namespace simplify_expression_l29_29156

theorem simplify_expression (a b : ℤ) : 4 * a + 5 * b - a - 7 * b = 3 * a - 2 * b :=
by
  sorry

end simplify_expression_l29_29156


namespace smallest_digit_never_at_units_place_of_odd_l29_29846

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l29_29846


namespace round_robin_highest_score_l29_29382

theorem round_robin_highest_score
  (n : ℕ) (hn : n = 16)
  (teams : Fin n → ℕ)
  (games_played : Fin n → Fin n → ℕ)
  (draws : Fin n → Fin n → ℕ)
  (win_points : ℕ := 2)
  (draw_points : ℕ := 1)
  (total_games : ℕ := (n * (n - 1)) / 2) :
  ¬ (∃ max_score : ℕ, ∀ i : Fin n, teams i ≤ max_score ∧ max_score < 16) :=
by sorry

end round_robin_highest_score_l29_29382


namespace smallest_enclosing_sphere_radius_l29_29013

-- Define the conditions
def sphere_radius : ℝ := 2

-- Define the sphere center coordinates in each octant
def sphere_centers : List (ℝ × ℝ × ℝ) :=
  [ (2, 2, 2), (2, 2, -2), (2, -2, 2), (2, -2, -2),
    (-2, 2, 2), (-2, 2, -2), (-2, -2, 2), (-2, -2, -2) ]

-- Define the theorem statement
theorem smallest_enclosing_sphere_radius :
  (∃ (r : ℝ), r = 2 * Real.sqrt 3 + 2) :=
by
  -- conditions and proof will go here
  sorry

end smallest_enclosing_sphere_radius_l29_29013


namespace smallest_digit_never_in_units_place_of_odd_numbers_l29_29781

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l29_29781


namespace smallest_possible_sum_l29_29058

theorem smallest_possible_sum :
  ∃ (B : ℕ) (c : ℕ), B + c = 34 ∧ 
    (B ≥ 0 ∧ B < 5) ∧ 
    (c > 7) ∧ 
    (31 * B = 4 * c + 4) := 
by
  sorry

end smallest_possible_sum_l29_29058


namespace smallest_digit_never_at_units_place_of_odd_l29_29843

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l29_29843


namespace value_of_c_infinite_solutions_l29_29563

theorem value_of_c_infinite_solutions (c : ℝ) :
  (∀ y : ℝ, 3 * (5 + 2 * c * y) = 18 * y + 15) ↔ (c = 3) :=
by
  sorry

end value_of_c_infinite_solutions_l29_29563


namespace sticks_predict_good_fortune_l29_29250

def good_fortune_probability := 11 / 12

theorem sticks_predict_good_fortune:
  (∃ (α β: ℝ), 0 ≤ α ∧ α ≤ π / 2 ∧ 0 ≤ β ∧ β ≤ π / 2 ∧ (0 ≤ β ∧ β < π - α) ∧ (0 ≤ α ∧ α < π - β)) → 
  good_fortune_probability = 11 / 12 :=
sorry

end sticks_predict_good_fortune_l29_29250


namespace mudit_age_l29_29211

theorem mudit_age :
    ∃ x : ℤ, x + 16 = 3 * (x - 4) ∧ x = 14 :=
by
  use 14
  sorry -- Proof goes here

end mudit_age_l29_29211


namespace smallest_unfound_digit_in_odd_units_l29_29824

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l29_29824


namespace evaluate_expression_l29_29555

theorem evaluate_expression : 3 * (3 * (3 * (3 + 2) + 2) + 2) + 2 = 161 := sorry

end evaluate_expression_l29_29555


namespace find_term_of_sequence_l29_29157

theorem find_term_of_sequence :
  ∀ (a d n : ℤ), a = -5 → d = -4 → (-4)*n + 1 = -401 → n = 100 :=
by
  intros a d n h₁ h₂ h₃
  sorry

end find_term_of_sequence_l29_29157


namespace find_unknown_number_l29_29922

-- Defining the conditions of the problem
def equation (x : ℝ) : Prop := (45 + x / 89) * 89 = 4028

-- Stating the theorem to be proved
theorem find_unknown_number : equation 23 :=
by
  -- Placeholder for the proof
  sorry

end find_unknown_number_l29_29922


namespace size_ratio_l29_29359

variable {A B C : ℝ} -- Declaring that A, B, and C are real numbers (their sizes)
variable (h1 : A = 3 * B) -- A is three times the size of B
variable (h2 : B = (1 / 2) * C) -- B is half the size of C

theorem size_ratio (h1 : A = 3 * B) (h2 : B = (1 / 2) * C) : A / C = 1.5 :=
by
  sorry -- Proof goes here, to be completed

end size_ratio_l29_29359


namespace area_of_triangle_CDE_l29_29071

theorem area_of_triangle_CDE
  (DE : ℝ) (h : ℝ)
  (hDE : DE = 12) (hh : h = 15) :
  1/2 * DE * h = 90 := by
  sorry

end area_of_triangle_CDE_l29_29071


namespace remaining_pencils_check_l29_29460

variables (Jeff_initial : ℕ) (Jeff_donation_percentage : ℚ) (Vicki_ratio : ℚ) (Vicki_donation_fraction : ℚ)

def Jeff_donated_pencils := (Jeff_donation_percentage * Jeff_initial).toNat
def Jeff_remaining_pencils := Jeff_initial - Jeff_donated_pencils

def Vicki_initial_pencils := (Vicki_ratio * Jeff_initial).toNat
def Vicki_donated_pencils := (Vicki_donation_fraction * Vicki_initial_pencils).toNat
def Vicki_remaining_pencils := Vicki_initial_pencils - Vicki_donated_pencils

def total_remaining_pencils := Jeff_remaining_pencils + Vicki_remaining_pencils

theorem remaining_pencils_check
    (Jeff_initial : ℕ := 300)
    (Jeff_donation_percentage : ℚ := 0.3)
    (Vicki_ratio : ℚ := 2)
    (Vicki_donation_fraction : ℚ := 0.75) :
    total_remaining_pencils Jeff_initial Jeff_donation_percentage Vicki_ratio Vicki_donation_fraction = 360 :=
by
  sorry

end remaining_pencils_check_l29_29460


namespace law_of_sines_proof_l29_29370

noncomputable def law_of_sines (a b c α β γ : ℝ) :=
  (a / Real.sin α = b / Real.sin β) ∧
  (b / Real.sin β = c / Real.sin γ) ∧
  (α + β + γ = Real.pi)

theorem law_of_sines_proof (a b c α β γ : ℝ) (h : law_of_sines a b c α β γ) :
  (a = b * Real.cos γ + c * Real.cos β) ∧
  (b = c * Real.cos α + a * Real.cos γ) ∧
  (c = a * Real.cos β + b * Real.cos α) :=
sorry

end law_of_sines_proof_l29_29370


namespace ellipse_a_plus_k_l29_29665

theorem ellipse_a_plus_k (f1 f2 p : Real × Real) (a b h k : Real) :
  f1 = (2, 0) →
  f2 = (-2, 0) →
  p = (5, 3) →
  (∀ x y, ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1) →
  a > 0 →
  b > 0 →
  h = 0 →
  k = 0 →
  a = (3 * Real.sqrt 2 + Real.sqrt 58) / 2 →
  a + k = (3 * Real.sqrt 2 + Real.sqrt 58) / 2 :=
by
  intros
  sorry

end ellipse_a_plus_k_l29_29665


namespace paityn_red_hats_l29_29752

theorem paityn_red_hats (R : ℕ) : 
  (R + 24 + (4 / 5) * ↑R + 48 = 108) → R = 20 :=
by
  intro h
  sorry


end paityn_red_hats_l29_29752


namespace minimize_expense_l29_29253

def price_after_first_discount (initial_price : ℕ) (discount : ℕ) : ℕ :=
  initial_price * (100 - discount) / 100

def final_price_set1 (initial_price : ℕ) : ℕ :=
  let step1 := price_after_first_discount initial_price 15
  let step2 := price_after_first_discount step1 25
  price_after_first_discount step2 10

def final_price_set2 (initial_price : ℕ) : ℕ :=
  let step1 := price_after_first_discount initial_price 25
  let step2 := price_after_first_discount step1 10
  price_after_first_discount step2 10

theorem minimize_expense (initial_price : ℕ) (h : initial_price = 12000) :
  final_price_set1 initial_price = 6885 ∧ final_price_set2 initial_price = 7290 ∧
  final_price_set1 initial_price < final_price_set2 initial_price := by
  sorry

end minimize_expense_l29_29253


namespace evaluate_binom_mul_factorial_l29_29283

theorem evaluate_binom_mul_factorial (n : ℕ) (h : n > 0) :
  (Nat.choose (n + 2) n) * n! = ((n + 2) * (n + 1) * n!) / 2 := by
  sorry

end evaluate_binom_mul_factorial_l29_29283


namespace seating_solution_l29_29950

/-- 
Imagine Abby, Bret, Carl, and Dana are seated in a row of four seats numbered from 1 to 4.
Joe observes them and declares:

- "Bret is sitting next to Dana" (False)
- "Carl is between Abby and Dana" (False)

Further, it is known that Abby is in seat #2.

Who is seated in seat #3? 
-/

def seating_problem : Prop :=
  ∃ (seats : ℕ → ℕ),
  (¬ (seats 1 = 1 ∧ seats 1 = 4 ∨ seats 4 = 1 ∧ seats 4 = 4)) ∧
  (¬ (seats 3 > seats 1 ∧ seats 3 < seats 2 ∨ seats 3 > seats 2 ∧ seats 3 < seats 1)) ∧
  (seats 2 = 2) →
  (seats 3 = 3)

theorem seating_solution : seating_problem :=
sorry

end seating_solution_l29_29950


namespace hall_area_proof_l29_29776

noncomputable def hall_length (L : ℕ) : ℕ := L
noncomputable def hall_width (L : ℕ) (W : ℕ) : ℕ := W
noncomputable def hall_area (L W : ℕ) : ℕ := L * W

theorem hall_area_proof (L W : ℕ) (h1 : W = 1 / 2 * L) (h2 : L - W = 15) :
  hall_area L W = 450 := by
  sorry

end hall_area_proof_l29_29776


namespace ratio_of_lost_diaries_to_total_diaries_l29_29613

theorem ratio_of_lost_diaries_to_total_diaries 
  (original_diaries : ℕ)
  (bought_diaries : ℕ)
  (current_diaries : ℕ)
  (h1 : original_diaries = 8)
  (h2 : bought_diaries = 2 * original_diaries)
  (h3 : current_diaries = 18) :
  (original_diaries + bought_diaries - current_diaries) / gcd (original_diaries + bought_diaries - current_diaries) (original_diaries + bought_diaries) 
  = 1 / 4 :=
by
  sorry

end ratio_of_lost_diaries_to_total_diaries_l29_29613


namespace correct_calculation_result_l29_29389

theorem correct_calculation_result 
  (P : Polynomial ℝ := -x^2 + x - 1) :
  (P + -3 * x) = (-x^2 - 2 * x - 1) :=
by
  -- Since this is just the proof statement, sorry is used to skip the proof.
  sorry

end correct_calculation_result_l29_29389


namespace smallest_digit_not_in_units_place_of_odd_l29_29811

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l29_29811


namespace part1_part2_part3_l29_29509

-- Part 1: There exists a real number a such that a + 1/a ≤ 2
theorem part1 : ∃ a : ℝ, a + 1/a ≤ 2 := sorry

-- Part 2: For all positive real numbers a and b, b/a + a/b ≥ 2
theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : b / a + a / b ≥ 2 := sorry

-- Part 3: For positive real numbers x and y such that x + 2y = 1, then 2/x + 1/y ≥ 8
theorem part3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 2 / x + 1 / y ≥ 8 := sorry

end part1_part2_part3_l29_29509


namespace p_more_than_q_l29_29230

def stamps (p q : ℕ) : Prop :=
  p / q = 7 / 4 ∧ (p - 8) / (q + 8) = 6 / 5

theorem p_more_than_q (p q : ℕ) (h : stamps p q) : p - 8 - (q + 8) = 8 :=
by {
  sorry
}

end p_more_than_q_l29_29230


namespace minimal_board_size_for_dominoes_l29_29096

def board_size_is_minimal (n: ℕ) (total_area: ℕ) (domino_size: ℕ) (num_dominoes: ℕ) : Prop :=
  ∀ m: ℕ, m < n → ¬ (total_area ≥ m * m ∧ m * m = num_dominoes * domino_size)

theorem minimal_board_size_for_dominoes (n: ℕ) :
  board_size_is_minimal 77 2008 2 1004 :=
by
  sorry

end minimal_board_size_for_dominoes_l29_29096


namespace calc1_calc2_l29_29272

theorem calc1 : (-2) * (-1/8) = 1/4 :=
by
  sorry

theorem calc2 : (-5) / (6/5) = -25/6 :=
by
  sorry

end calc1_calc2_l29_29272


namespace smallest_digit_not_in_units_place_of_odd_l29_29809

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l29_29809


namespace Tyler_has_200_puppies_l29_29238

-- Define the number of dogs
def numDogs : ℕ := 25

-- Define the number of puppies per dog
def puppiesPerDog : ℕ := 8

-- Define the total number of puppies
def totalPuppies : ℕ := numDogs * puppiesPerDog

-- State the theorem we want to prove
theorem Tyler_has_200_puppies : totalPuppies = 200 := by
  exact (by norm_num : 25 * 8 = 200)

end Tyler_has_200_puppies_l29_29238


namespace intersection_of_M_and_N_l29_29321

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := 
by sorry

end intersection_of_M_and_N_l29_29321


namespace smallest_n_condition_l29_29542

def pow_mod (a b m : ℕ) : ℕ := a^(b % m)

def n (r s : ℕ) : ℕ := 2^r - 16^s

def r_condition (r : ℕ) : Prop := ∃ k : ℕ, r = 3 * k + 1

def s_condition (s : ℕ) : Prop := ∃ h : ℕ, s = 3 * h + 2

theorem smallest_n_condition (r s : ℕ) (hr : r_condition r) (hs : s_condition s) :
  (n r s) % 7 = 5 → (n r s) = 768 := sorry

end smallest_n_condition_l29_29542


namespace max_sector_area_central_angle_l29_29779

theorem max_sector_area_central_angle (radius arc_length : ℝ) :
  (arc_length + 2 * radius = 20) ∧ (arc_length = 20 - 2 * radius) ∧
  (arc_length / radius = 2) → 
  arc_length / radius = 2 :=
by
  intros h 
  sorry

end max_sector_area_central_angle_l29_29779


namespace probability_of_two_black_balls_relationship_x_y_l29_29956

-- Conditions
def initial_black_balls : ℕ := 3
def initial_white_balls : ℕ := 2

variable (x y : ℕ)

-- Given relationship
def total_white_balls := x + 2
def total_black_balls := y + 3
def white_ball_probability := (total_white_balls x) / (total_white_balls x + total_black_balls y + 5)

-- Proof goals
theorem probability_of_two_black_balls :
  (3 / 5) * (2 / 4) = 3 / 10 := by sorry

theorem relationship_x_y :
  white_ball_probability x y = 1 / 3 → y = 2 * x + 1 := by sorry

end probability_of_two_black_balls_relationship_x_y_l29_29956


namespace annes_initial_bottle_caps_l29_29390

-- Define the conditions
def albert_bottle_caps : ℕ := 9
def annes_added_bottle_caps : ℕ := 5
def annes_total_bottle_caps : ℕ := 15

-- Question (to prove)
theorem annes_initial_bottle_caps :
  annes_total_bottle_caps - annes_added_bottle_caps = 10 :=
by sorry

end annes_initial_bottle_caps_l29_29390


namespace exp_gt_one_iff_a_gt_one_l29_29580

theorem exp_gt_one_iff_a_gt_one (a : ℝ) : 
  (∀ x : ℝ, 0 < x → a^x > 1) ↔ a > 1 :=
by
  sorry

end exp_gt_one_iff_a_gt_one_l29_29580


namespace max_value_of_symmetric_function_l29_29725

def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function 
  (a b : ℝ)
  (symmetric : ∀ t : ℝ, f (-2 + t) a b = f (-2 - t) a b) :
  ∃ M : ℝ, M = 16 ∧ ∀ x : ℝ, f x a b ≤ M :=
by
  use 16
  sorry

end max_value_of_symmetric_function_l29_29725


namespace mia_spent_total_l29_29986

theorem mia_spent_total (sibling_cost parent_cost : ℕ) (num_siblings num_parents : ℕ)
    (h1 : sibling_cost = 30)
    (h2 : parent_cost = 30)
    (h3 : num_siblings = 3)
    (h4 : num_parents = 2) :
    sibling_cost * num_siblings + parent_cost * num_parents = 150 :=
by
  sorry

end mia_spent_total_l29_29986


namespace solve_system_of_equations_l29_29338

theorem solve_system_of_equations (x y_1 y_2 y_3: ℝ) (n : ℤ) (h1 : -3 ≤ n) (h2 : n ≤ 3)
  (h_eq1 : (1 - x^2) * y_1 = 2 * x)
  (h_eq2 : (1 - y_1^2) * y_2 = 2 * y_1)
  (h_eq3 : (1 - y_2^2) * y_3 = 2 * y_2)
  (h_eq4 : y_3 = x) :
  y_1 = Real.tan (2 * n * Real.pi / 7) ∧
  y_2 = Real.tan (4 * n * Real.pi / 7) ∧
  y_3 = Real.tan (n * Real.pi / 7) ∧
  x = Real.tan (n * Real.pi / 7) :=
sorry

end solve_system_of_equations_l29_29338


namespace no_max_value_if_odd_and_symmetric_l29_29173

variable (f : ℝ → ℝ)

-- Definitions:
def domain_is_R (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f x
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_symmetric_about_1_1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 - x) = 2 - f x

-- The theorem stating that under the given conditions there is no maximum value.
theorem no_max_value_if_odd_and_symmetric :
  domain_is_R f → is_odd_function f → is_symmetric_about_1_1 f → ¬∃ M : ℝ, ∀ x : ℝ, f x ≤ M := by
  sorry

end no_max_value_if_odd_and_symmetric_l29_29173


namespace bathroom_visits_time_l29_29964

variable (t_8 : ℕ) (n8 : ℕ) (n6 : ℕ)

theorem bathroom_visits_time (h1 : t_8 = 20) (h2 : n8 = 8) (h3 : n6 = 6) :
  (t_8 / n8) * n6 = 15 := by
  sorry

end bathroom_visits_time_l29_29964


namespace normal_distribution_probability_between_l29_29730

open Probability

noncomputable def normal_dist (μ σ : ℝ) : EventSpace :=
  NormalDistribution μ (σ ^ 2)

variable {X : ℝ}

theorem normal_distribution_probability_between
  (μ σ : ℝ)
  (hX : X ∈ normal_dist 1 (σ ^ 2))
  (h0 : P(X ≤ 0) = 0.3) : P(0 < X < 2) = 0.4 :=
by
  sorry

end normal_distribution_probability_between_l29_29730


namespace k_is_odd_l29_29098

theorem k_is_odd (m n k : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_pos_k : 0 < k) (h : 3 * m * k = (m + 3)^n + 1) : Odd k :=
by {
  sorry
}

end k_is_odd_l29_29098


namespace smallest_unfound_digit_in_odd_units_l29_29823

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l29_29823


namespace train_a_distance_at_meeting_l29_29502

-- Define the problem conditions as constants
def distance := 75 -- distance between start points of Train A and B
def timeA := 3 -- time taken by Train A to complete the trip in hours
def timeB := 2 -- time taken by Train B to complete the trip in hours

-- Calculate the speeds
def speedA := distance / timeA -- speed of Train A in miles per hour
def speedB := distance / timeB -- speed of Train B in miles per hour

-- Calculate the combined speed and time to meet
def combinedSpeed := speedA + speedB
def timeToMeet := distance / combinedSpeed

-- Define the distance traveled by Train A at the time of meeting
def distanceTraveledByTrainA := speedA * timeToMeet

-- Theorem stating Train A has traveled 30 miles when it met Train B
theorem train_a_distance_at_meeting : distanceTraveledByTrainA = 30 := by
  sorry

end train_a_distance_at_meeting_l29_29502


namespace range_of_m_l29_29178

theorem range_of_m (m : ℝ) :
  (m + 4 - 4)*(2 + 2 * m - 4) < 0 → 0 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l29_29178


namespace quadratic_y_at_x_5_l29_29225

-- Define the quadratic function
noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions and question as part of a theorem
theorem quadratic_y_at_x_5 (a b c : ℝ) 
  (h1 : ∀ x, quadratic a b c x ≤ 10) -- Maximum value condition (The maximum value is 10)
  (h2 : (quadratic a b c (-2)) = 10) -- y = 10 when x = -2 (maximum point)
  (h3 : quadratic a b c 0 = -8) -- The first point (0, -8)
  (h4 : quadratic a b c 1 = 0) -- The second point (1, 0)
  : quadratic a b c 5 = -400 / 9 :=
sorry

end quadratic_y_at_x_5_l29_29225


namespace find_x_plus_y_l29_29433

theorem find_x_plus_y (x y : ℝ) 
  (h1 : x + Real.cos y = 1004)
  (h2 : x + 1004 * Real.sin y = 1003)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) : x + y = 1003 :=
sorry

end find_x_plus_y_l29_29433


namespace range_of_f_is_pi_div_four_l29_29031

noncomputable def f (x : ℝ) : ℝ := 
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f_is_pi_div_four : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y = π / 4 :=
sorry

end range_of_f_is_pi_div_four_l29_29031


namespace complex_modulus_l29_29549

theorem complex_modulus :
  abs ((7 - 4*complex.I) * (3 + 11*complex.I)) = Real.sqrt 8450 :=
by
  sorry

end complex_modulus_l29_29549


namespace alpha_pi_over_four_sufficient_not_necessary_l29_29372

theorem alpha_pi_over_four_sufficient_not_necessary :
  (∀ α : ℝ, (α = (Real.pi / 4) → Real.cos α = Real.sqrt 2 / 2)) ∧
  (∃ α : ℝ, (Real.cos α = Real.sqrt 2 / 2) ∧ α ≠ (Real.pi / 4)) :=
by
  sorry

end alpha_pi_over_four_sufficient_not_necessary_l29_29372


namespace product_primes_less_than_20_l29_29116

theorem product_primes_less_than_20 :
  (2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 = 9699690) :=
by
  sorry

end product_primes_less_than_20_l29_29116


namespace smallest_missing_digit_l29_29863

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l29_29863


namespace inequality_system_range_l29_29051

theorem inequality_system_range (a : ℝ) :
  (∃ (x : ℤ), (6 * (x : ℝ) + 2 > 3 * (x : ℝ) + 5) ∧ (2 * (x : ℝ) - a ≤ 0)) ∧
  (∀ x : ℤ, (6 * (x : ℝ) + 2 > 3 * (x : ℝ) + 5) ∧ (2 * (x : ℝ) - a ≤ 0) → (x = 2 ∨ x = 3)) →
  6 ≤ a ∧ a < 8 :=
by
  sorry

end inequality_system_range_l29_29051


namespace probability_not_within_square_B_l29_29385

theorem probability_not_within_square_B {A B : Type} 
  (area_A : ℝ) (perimeter_B : ℝ) (area_B : ℝ) (not_covered : ℝ) 
  (h1 : area_A = 30) 
  (h2 : perimeter_B = 16) 
  (h3 : area_B = 16) 
  (h4 : not_covered = area_A - area_B) :
  (not_covered / area_A) = 7 / 15 := by sorry

end probability_not_within_square_B_l29_29385


namespace primes_product_less_than_20_l29_29118

-- Define the primes less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the product of a list of natural numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

theorem primes_product_less_than_20 :
  product primes_less_than_20 = 9699690 :=
by
  sorry

end primes_product_less_than_20_l29_29118


namespace rectangle_area_192_l29_29998

variable (b l : ℝ) (A : ℝ)

-- Conditions
def length_is_thrice_breadth : Prop :=
  l = 3 * b

def perimeter_is_64 : Prop :=
  2 * (l + b) = 64

-- Area calculation
def area_of_rectangle : ℝ :=
  l * b

theorem rectangle_area_192 (h1 : length_is_thrice_breadth b l) (h2 : perimeter_is_64 b l) :
  area_of_rectangle l b = 192 := by
  sorry

end rectangle_area_192_l29_29998


namespace gcd_of_128_144_480_450_l29_29114

theorem gcd_of_128_144_480_450 : Nat.gcd (Nat.gcd 128 144) (Nat.gcd 480 450) = 6 := 
by
  sorry

end gcd_of_128_144_480_450_l29_29114


namespace geometric_sequence_constant_l29_29138

theorem geometric_sequence_constant (a : ℕ → ℝ) (q : ℝ) (h1 : q ≠ 1) (h2 : ∀ n, a (n + 1) = q * a n) (c : ℝ) :
  (∀ n, a (n + 1) + c = q * (a n + c)) → c = 0 := sorry

end geometric_sequence_constant_l29_29138


namespace problem_1_problem_2_l29_29440

noncomputable def f (ω x : ℝ) : ℝ :=
  Real.sin (ω * x + (Real.pi / 4))

theorem problem_1 (ω : ℝ) (hω : ω > 0) : f ω 0 = Real.sqrt 2 / 2 :=
by
  unfold f
  simp [Real.sin_pi_div_four]

theorem problem_2 : 
  ∃ x : ℝ, 
  0 ≤ x ∧ x ≤ Real.pi / 2 ∧ 
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ Real.pi / 2 → f 2 y ≤ f 2 x) ∧ 
  f 2 x = 1 :=
by
  sorry

end problem_1_problem_2_l29_29440


namespace pencils_total_l29_29457

def pencils_remaining (Jeff_pencils_initial : ℕ) (Jeff_donation_percent : ℕ) 
                      (Vicki_factor : ℕ) (Vicki_donation_fraction_num : ℕ) 
                      (Vicki_donation_fraction_den : ℕ) : ℕ :=
  let Jeff_donated := Jeff_pencils_initial * Jeff_donation_percent / 100
  let Jeff_remaining := Jeff_pencils_initial - Jeff_donated
  let Vicki_pencils_initial := Vicki_factor * Jeff_pencils_initial
  let Vicki_donated := Vicki_pencils_initial * Vicki_donation_fraction_num / Vicki_donation_fraction_den
  let Vicki_remaining := Vicki_pencils_initial - Vicki_donated
  Jeff_remaining + Vicki_remaining

theorem pencils_total :
  pencils_remaining 300 30 2 3 4 = 360 :=
by
  -- The proof should be inserted here
  sorry

end pencils_total_l29_29457


namespace shortest_chord_through_M_l29_29170

noncomputable def point_M : ℝ × ℝ := (1, 0)
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y = 0

theorem shortest_chord_through_M :
  (∀ x y : ℝ, circle_C x y → x + y - 1 = 0) :=
by
  sorry

end shortest_chord_through_M_l29_29170


namespace factor_expression_l29_29161

-- Problem Statement
theorem factor_expression (x y : ℝ) : 60 * x ^ 2 + 40 * y = 20 * (3 * x ^ 2 + 2 * y) :=
by
  -- Proof to be provided
  sorry

end factor_expression_l29_29161


namespace extremum_points_l29_29073

noncomputable def f (x1 x2 : ℝ) : ℝ := x1 * x2 / (1 + x1^2 * x2^2)

theorem extremum_points :
  (f 0 0 = 0) ∧
  (∀ x1 : ℝ, f x1 (-1 / x1) = -1 / 2) ∧
  (∀ x1 : ℝ, f x1 (1 / x1) = 1 / 2) ∧
  ∀ y1 y2 : ℝ, (f 0 0 < f y1 y2 → (0 < y1 ∧ 0 < y2)) ∧ 
             (f 0 0 > f y1 y2 → (0 > y1 ∧ 0 > y2)) :=
by
  sorry

end extremum_points_l29_29073


namespace intersection_unique_l29_29907

noncomputable def f (x : ℝ) := 3 * Real.log x
noncomputable def g (x : ℝ) := Real.log (x + 4)

theorem intersection_unique : ∃! x, f x = g x :=
sorry

end intersection_unique_l29_29907


namespace smallest_digit_not_in_odd_units_l29_29855

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l29_29855


namespace total_birds_correct_l29_29700

def num_female_doves := 80
def num_male_pigeons := 50
def eggs_per_dove := 6
def eggs_per_pigeon := 4
def dove_hatch_rate := 8 / 10
def pigeon_hatch_rate := 2 / 3

noncomputable def total_birds : ℕ :=
  num_female_doves + (num_female_doves * eggs_per_dove * dove_hatch_rate).toNat +
  num_male_pigeons + (num_male_pigeons * eggs_per_pigeon * pigeon_hatch_rate).toNat

theorem total_birds_correct :
  total_birds = 647 := by
  sorry

end total_birds_correct_l29_29700


namespace smallest_digit_never_in_units_place_of_odd_number_l29_29870

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l29_29870


namespace smallest_digit_not_in_odd_units_l29_29856

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l29_29856


namespace smallest_digit_not_in_units_place_of_odd_l29_29808

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l29_29808


namespace smallest_digit_not_in_odd_units_l29_29853

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l29_29853


namespace smallest_digit_not_in_units_place_of_odd_l29_29812

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l29_29812


namespace manufacturer_cost_price_l29_29145

theorem manufacturer_cost_price (final_price : ℝ) (m_profit r1 r2 r3 : ℝ) : 
  final_price = 30.09 → 
  m_profit = 0.18 → 
  r1 = 1.20 → 
  r2 = 1.25 → 
  let C := final_price / ((1 + m_profit) * r1 * r2) in 
  C ≈ 17 :=
by sorry

end manufacturer_cost_price_l29_29145


namespace calc_3_pow_6_mul_4_pow_6_l29_29539

theorem calc_3_pow_6_mul_4_pow_6 : (3^6) * (4^6) = 2985984 :=
by 
  sorry

end calc_3_pow_6_mul_4_pow_6_l29_29539


namespace smallest_digit_never_at_units_place_of_odd_l29_29841

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l29_29841


namespace A_n_squared_l29_29946

-- Define C(n-2)
def C_n_2 (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define A_n_2
def A_n_2 (n : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - 2)

theorem A_n_squared (n : ℕ) (hC : C_n_2 n = 15) : A_n_2 n = 30 := by
  sorry

end A_n_squared_l29_29946


namespace smallest_missing_digit_l29_29862

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l29_29862


namespace inequality_proof_l29_29421

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b^2 + b / c^2 + c / a^2 ≥ 1 / a + 1 / b + 1 / c := by
  sorry

end inequality_proof_l29_29421


namespace smallest_checkered_rectangle_area_l29_29378

def even (n: ℕ) : Prop := n % 2 = 0

-- Both figure types are present and areas of these types are 1 and 2 respectively
def isValidPieceComposition (a b : ℕ) : Prop :=
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m * 1 + n * 2 = a * b

theorem smallest_checkered_rectangle_area :
  ∀ a b : ℕ, even a → even b → isValidPieceComposition a b → a * b ≥ 40 := 
by
  intro a b a_even b_even h_valid
  sorry

end smallest_checkered_rectangle_area_l29_29378


namespace video_game_price_l29_29006

theorem video_game_price (total_games not_working_games : ℕ) (total_earnings : ℕ)
  (h1 : total_games = 10) (h2 : not_working_games = 2) (h3 : total_earnings = 32) :
  ((total_games - not_working_games) > 0) →
  (total_earnings / (total_games - not_working_games)) = 4 :=
by
  sorry

end video_game_price_l29_29006


namespace smallest_digit_not_in_odd_units_l29_29837

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l29_29837


namespace factorization_correct_l29_29397

def factor_expression (x : ℝ) : ℝ :=
  (12 * x^4 - 27 * x^3 + 45 * x) - (-3 * x^4 - 6 * x^3 + 9 * x)

theorem factorization_correct (x : ℝ) : 
  factor_expression x = 3 * x * (5 * x^3 - 7 * x^2 + 12) :=
by
  sorry

end factorization_correct_l29_29397


namespace smallest_unfound_digit_in_odd_units_l29_29828

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l29_29828


namespace evaluate_f_at_points_l29_29003

def f (x : ℝ) : ℝ :=
  3 * x ^ 2 - 6 * x + 10

theorem evaluate_f_at_points : 3 * f 2 + 2 * f (-2) = 98 :=
by
  sorry

end evaluate_f_at_points_l29_29003


namespace alice_prob_after_three_turns_l29_29893

/-
Definition of conditions:
 - Alice starts with the ball.
 - If Alice has the ball, there is a 1/3 chance that she will toss it to Bob and a 2/3 chance that she will keep the ball.
 - If Bob has the ball, there is a 1/4 chance that he will toss it to Alice and a 3/4 chance that he keeps the ball.
-/

def alice_to_bob : ℚ := 1/3
def alice_keeps : ℚ := 2/3
def bob_to_alice : ℚ := 1/4
def bob_keeps : ℚ := 3/4

theorem alice_prob_after_three_turns :
  alice_to_bob * bob_keeps * bob_to_alice +
  alice_keeps * alice_keeps * alice_keeps +
  alice_to_bob * bob_to_alice * alice_keeps = 179/432 :=
by
  sorry

end alice_prob_after_three_turns_l29_29893


namespace power_mod_l29_29163

theorem power_mod (h1: 5^2 % 17 = 8) (h2: 5^4 % 17 = 13) (h3: 5^8 % 17 = 16) (h4: 5^16 % 17 = 1):
  5^2024 % 17 = 16 :=
by
  sorry

end power_mod_l29_29163


namespace geometric_series_common_ratio_l29_29896

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 512) (hS : S = 2048) (h_sum : S = a / (1 - r)) : r = 3 / 4 :=
by
  rw [ha, hS] at h_sum 
  sorry

end geometric_series_common_ratio_l29_29896


namespace problem_statement_l29_29061

-- Definitions of A and B based on the given conditions
def A : ℤ := -5 * -3
def B : ℤ := 2 - 2

-- The theorem stating that A + B = 15
theorem problem_statement : A + B = 15 := 
by 
  sorry

end problem_statement_l29_29061


namespace pradeep_max_marks_l29_29478

-- conditions
variables (M : ℝ)
variable (h1 : 0.40 * M = 220)

-- question and answer
theorem pradeep_max_marks : M = 550 :=
by
  sorry

end pradeep_max_marks_l29_29478


namespace inequality_proof_l29_29430

variables (a b c d : ℝ)

theorem inequality_proof (h1 : a > b) (h2 : c = d) : a + c > b + d :=
sorry

end inequality_proof_l29_29430


namespace complementary_supplementary_angle_l29_29046

theorem complementary_supplementary_angle (x : ℝ) :
  (90 - x) * 3 = 180 - x → x = 45 :=
by 
  intro h
  sorry

end complementary_supplementary_angle_l29_29046


namespace exists_at_least_3_red_in_2x2_subgrid_l29_29590
open Finset Function Real

def GridCell := (Fin 9) × (Fin 9)

def is_red (cells : Finset GridCell) (i j : Fin 9) : Prop := (i, j) ∈ cells

theorem exists_at_least_3_red_in_2x2_subgrid (cells : Finset GridCell) (h : cells.card = 46) :
  ∃ (i j : Fin 8), 
    (is_red cells i j) + (is_red cells i (j + 1)) + (is_red cells (i + 1) j) + (is_red cells (i + 1) (j + 1)) ≥ 3 := 
by 
  sorry

end exists_at_least_3_red_in_2x2_subgrid_l29_29590


namespace least_negative_b_l29_29487

theorem least_negative_b (x b : ℤ) (h1 : x^2 + b * x = 22) (h2 : b < 0) : b = -21 :=
sorry

end least_negative_b_l29_29487


namespace find_c_deg3_l29_29678

-- Define the polynomials f and g.
def f (x : ℚ) : ℚ := 2 - 10 * x + 4 * x^2 - 5 * x^3 + 7 * x^4
def g (x : ℚ) : ℚ := 5 - 3 * x - 8 * x^3 + 11 * x^4

-- The statement that needs proof.
theorem find_c_deg3 (c : ℚ) : (∀ x : ℚ, f x + c * g x ≠ 0 → f x + c * g x = 2 - 10 * x + 4 * x^2 - 5 * x^3 - c * 8 * x^3) ↔ c = -7 / 11 :=
sorry

end find_c_deg3_l29_29678


namespace range_of_a_l29_29931

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Condition: f(x) is an increasing function on ℝ.
def is_increasing_on_ℝ (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x < f y

-- Equivalent proof problem in Lean 4:
theorem range_of_a (h : is_increasing_on_ℝ f) : 1 < a ∧ a < 6 := by
  sorry

end range_of_a_l29_29931


namespace total_dreams_correct_l29_29699

def dreams_per_day : Nat := 4
def days_in_year : Nat := 365
def current_year_dreams : Nat := dreams_per_day * days_in_year
def last_year_dreams : Nat := 2 * current_year_dreams
def total_dreams : Nat := current_year_dreams + last_year_dreams

theorem total_dreams_correct : total_dreams = 4380 :=
by
  -- prime verification needed here
  sorry

end total_dreams_correct_l29_29699


namespace sum_of_first_2m_terms_l29_29112

variable {α : Type*} [LinearOrderedField α]

noncomputable def arithmeticSum (a d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_2m_terms 
  (a d : α) (m : ℕ) 
  (h₁ : arithmeticSum a d m = 30)
  (h₂ : arithmeticSum a d (3 * m) = 90)
  : arithmeticSum a d (2 * m) = 60 :=
  sorry

end sum_of_first_2m_terms_l29_29112


namespace probability_of_given_condition_l29_29989

section
  -- Definitions of the conditions
  def total_lamps := 8
  def red_lamps := 4
  def blue_lamps := 4
  def total_turn_on := 4

  -- Binomial coefficient
  def choose (n k : ℕ) := Nat.choose n k

  -- Probability calculation
  def probability_condition := 
    -- Total ways to choose placements and turn ons
    let total_arrangements := choose total_lamps red_lamps
    let total_turning_on := choose total_lamps total_turn_on
  
    -- Condition-specific calculations
    let remaining_positions := 5
    let remaining_red := 2
    let remaining_blue := 3
    let remaining_turn_on := 3
  
    let ways_to_place_r := choose remaining_positions remaining_red
    let ways_to_turn_on := choose remaining_positions remaining_turn_on
    
    -- Probability
    (ways_to_place_r * ways_to_turn_on) / (total_arrangements * total_turning_on : ℝ)

  -- The proof problem stating that the result equals 1/49
  theorem probability_of_given_condition : probability_condition = 1 / 49 :=
  by
    sorry
end

end probability_of_given_condition_l29_29989


namespace inequality_solution_l29_29993

theorem inequality_solution :
  { x : ℝ | (x-1)/(x+4) ≤ 0 } = { x : ℝ | (-4 < x ∧ x ≤ 0) ∨ (x = 1) } :=
by 
  sorry

end inequality_solution_l29_29993


namespace fraction_value_is_one_fourth_l29_29363

theorem fraction_value_is_one_fourth (k : Nat) (hk : k ≥ 1) :
  (10^k + 6 * (10^k - 1) / 9) / (60 * (10^k - 1) / 9 + 4) = 1 / 4 :=
by
  sorry

end fraction_value_is_one_fourth_l29_29363


namespace value_of_product_of_sums_of_roots_l29_29973

theorem value_of_product_of_sums_of_roots 
    (a b c : ℂ)
    (h1 : a + b + c = 15)
    (h2 : a * b + b * c + c * a = 22)
    (h3 : a * b * c = 8) :
    (1 + a) * (1 + b) * (1 + c) = 46 := by
  sorry

end value_of_product_of_sums_of_roots_l29_29973


namespace cashback_percentage_l29_29217

theorem cashback_percentage
  (total_cost : ℝ) (rebate : ℝ) (final_cost : ℝ)
  (H1 : total_cost = 150) (H2 : rebate = 25) (H3 : final_cost = 110) :
  (total_cost - rebate - final_cost) / (total_cost - rebate) * 100 = 12 := by
  sorry

end cashback_percentage_l29_29217


namespace spoons_needed_to_fill_cup_l29_29767

-- Define necessary conditions
def spoon_capacity : Nat := 5
def liter_to_milliliters : Nat := 1000

-- State the problem
theorem spoons_needed_to_fill_cup : liter_to_milliliters / spoon_capacity = 200 := 
by 
  -- Skip the actual proof
  sorry

end spoons_needed_to_fill_cup_l29_29767


namespace units_digit_of_M_is_1_l29_29742

def Q (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  if units = 0 then 0 else tens / units

def T (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem units_digit_of_M_is_1 (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : b ≤ 9) (h₃ : 10*a + b = Q (10*a + b) + T (10*a + b)) :
  b = 1 :=
by
  sorry

end units_digit_of_M_is_1_l29_29742


namespace smallest_not_odd_unit_is_zero_l29_29800

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l29_29800


namespace problem_l29_29206

def m (x : ℝ) : ℝ := (x + 2) * (x + 3)
def n (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 9

theorem problem (x : ℝ) : m x < n x :=
by sorry

end problem_l29_29206


namespace smallest_digit_never_in_units_place_of_odd_numbers_l29_29782

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l29_29782


namespace David_Marks_in_Mathematics_are_85_l29_29007

theorem David_Marks_in_Mathematics_are_85
  (english_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (h1 : english_marks = 86)
  (h2 : physics_marks = 92)
  (h3 : chemistry_marks = 87)
  (h4 : biology_marks = 95)
  (h5 : average_marks = 89)
  (h6 : num_subjects = 5) : 
  (86 + 92 + 87 + 95 + 85) / 5 = 89 :=
by sorry

end David_Marks_in_Mathematics_are_85_l29_29007


namespace find_angle_and_sum_of_sides_l29_29309

noncomputable def triangle_conditions 
    (a b c : ℝ) (C : ℝ)
    (area : ℝ) : Prop :=
  a^2 + b^2 - c^2 = a * b ∧
  c = Real.sqrt 7 ∧
  area = (3 * Real.sqrt 3) / 2 

theorem find_angle_and_sum_of_sides
    (a b c C : ℝ)
    (area : ℝ)
    (h : triangle_conditions a b c C area) :
    C = Real.pi / 3 ∧ a + b = 5 := by
  sorry

end find_angle_and_sum_of_sides_l29_29309


namespace roses_after_trading_equals_36_l29_29466

-- Definitions of the given conditions
def initial_roses_given : ℕ := 24
def roses_after_trade (n : ℕ) : ℕ := n
def remaining_roses_after_first_wilt (roses : ℕ) : ℕ := roses / 2
def remaining_roses_after_second_wilt (roses : ℕ) : ℕ := roses / 2
def roses_remaining_second_day : ℕ := 9

-- The statement we want to prove
theorem roses_after_trading_equals_36 (n : ℕ) (h : roses_remaining_second_day = 9) :
  ( ∃ x, roses_after_trade x = n ∧ remaining_roses_after_first_wilt (remaining_roses_after_first_wilt x) = roses_remaining_second_day ) →
  n = 36 :=
by
  sorry

end roses_after_trading_equals_36_l29_29466


namespace corrected_mean_is_36_74_l29_29107

noncomputable def corrected_mean (incorrect_mean : ℝ) 
(number_of_observations : ℕ) 
(correct_value wrong_value : ℝ) : ℝ :=
(incorrect_mean * number_of_observations - wrong_value + correct_value) / number_of_observations

theorem corrected_mean_is_36_74 :
  corrected_mean 36 50 60 23 = 36.74 :=
by
  sorry

end corrected_mean_is_36_74_l29_29107


namespace smallest_digit_not_in_odd_units_l29_29835

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l29_29835


namespace similar_triangles_PQ_length_l29_29499

theorem similar_triangles_PQ_length (XY YZ QR : ℝ) (hXY : XY = 8) (hYZ : YZ = 16) (hQR : QR = 24)
  (hSimilar : ∃ (k : ℝ), XY = k * 8 ∧ YZ = k * 16 ∧ QR = k * 24) : (∃ (PQ : ℝ), PQ = 12) :=
by 
  -- Here we need to prove the theorem using similarity and given equalities
  sorry

end similar_triangles_PQ_length_l29_29499


namespace tangent_line_min_slope_l29_29296

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x

theorem tangent_line_min_slope :
  let f' := λ x, deriv f x
  let min_slope := -3
  let point_of_tangency := (0, f 0)
  let eq_of_tangent_line := (y := -3 * x)
  deriv f 0 = min_slope → tangent_line eq_of_tangent_line point_of_tangency :=
by
  sorry

end tangent_line_min_slope_l29_29296


namespace modulus_product_l29_29551

open Complex

theorem modulus_product :
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  (Complex.abs (z1 * z2)) = Real.sqrt 8450 :=
by
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  sorry

end modulus_product_l29_29551


namespace triangle_angles_in_given_ratio_l29_29108

theorem triangle_angles_in_given_ratio (x : ℝ) (y : ℝ) (z : ℝ) (h : x + y + z = 180) (r : x / 1 = y / 4 ∧ y / 4 = z / 7) : 
  x = 15 ∧ y = 60 ∧ z = 105 :=
by
  sorry

end triangle_angles_in_given_ratio_l29_29108


namespace prove_hyperbola_propositions_l29_29936

noncomputable def hyperbola_trajectory (x y : ℝ) : Prop :=
  (y / (x + 3)) * (y / (x - 3)) = (16 / 9)

theorem prove_hyperbola_propositions (M : ℝ × ℝ)
  (cond : hyperbola_trajectory M.1 M.2) :
  ((M.1 = -5 ∧ M.2 = 0) ∨ (M.1 = 5 ∧ M.2 = 0))
  ∧ (¬ x < 0 → M.1 = -3) :=
sorry

end prove_hyperbola_propositions_l29_29936


namespace smallest_not_odd_unit_is_zero_l29_29801

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l29_29801


namespace division_of_powers_of_ten_l29_29879

theorem division_of_powers_of_ten :
  (10 ^ 0.7 * 10 ^ 0.4) / (10 ^ 0.2 * 10 ^ 0.6 * 10 ^ 0.3) = 1 := by
  sorry

end division_of_powers_of_ten_l29_29879


namespace richard_more_pins_than_patrick_l29_29734

theorem richard_more_pins_than_patrick :
  ∀ (R P R2 P2 : ℕ), 
    P = 70 → 
    R > P →
    P2 = 2 * R →
    R2 = P2 - 3 → 
    (R + R2) = (P + P2) + 12 → 
    R = 70 + 15 := 
by 
  intros R P R2 P2 hP hRp hP2 hR2 hTotal
  sorry

end richard_more_pins_than_patrick_l29_29734


namespace boat_problem_l29_29141

theorem boat_problem (x y : ℕ) (h : 12 * x + 5 * y = 99) :
  (x = 2 ∧ y = 15) ∨ (x = 7 ∧ y = 3) :=
sorry

end boat_problem_l29_29141


namespace quadratic_roots_sum_l29_29916

theorem quadratic_roots_sum :
  ∃ a b c d : ℤ, (x^2 + 23 * x + 132 = (x + a) * (x + b)) ∧ (x^2 - 25 * x + 168 = (x - c) * (x - d)) ∧ (a + c + d = 42) :=
by {
  sorry
}

end quadratic_roots_sum_l29_29916


namespace number_of_juniors_in_sample_l29_29891

theorem number_of_juniors_in_sample
  (total_students : ℕ)
  (num_freshmen : ℕ)
  (num_freshmen_sampled : ℕ)
  (num_sophomores_exceeds_num_juniors_by : ℕ)
  (num_sophomores num_juniors num_juniors_sampled : ℕ)
  (h_total : total_students = 1290)
  (h_num_freshmen : num_freshmen = 480)
  (h_num_freshmen_sampled : num_freshmen_sampled = 96)
  (h_exceeds : num_sophomores_exceeds_num_juniors_by = 30)
  (h_equation : total_students - num_freshmen = num_sophomores + num_juniors)
  (h_num_sophomores : num_sophomores = num_juniors + num_sophomores_exceeds_num_juniors_by)
  (h_fraction : num_freshmen_sampled / num_freshmen = 1 / 5)
  (h_num_juniors_sampled : num_juniors_sampled = num_juniors * (num_freshmen_sampled / num_freshmen)) :
  num_juniors_sampled = 78 := by
  sorry

end number_of_juniors_in_sample_l29_29891


namespace smallest_digit_not_in_odd_units_l29_29857

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l29_29857


namespace system_has_three_real_k_with_unique_solution_l29_29302

theorem system_has_three_real_k_with_unique_solution :
  (∃ (k : ℝ), ∀ (x y : ℝ), (x^2 + y^2 = 2 * k^2 ∧ k * x - y = 2 * k) → (x, y) = (0, 0)) → 
  ∃ (k : ℝ), ∃ (x y : ℝ), (x^2 + y^2 = 2 * k^2 ∧ k * x - y = 2 * k) :=
by
  sorry

end system_has_three_real_k_with_unique_solution_l29_29302


namespace maximum_value_expression_l29_29205

theorem maximum_value_expression (x y : ℝ) (h : x + y = 5) :
  ∃ p, p = x * y ∧ (4 * p^3 - 92 * p^2 + 754 * p) = 441 / 2 :=
by {
  sorry
}

end maximum_value_expression_l29_29205


namespace max_value_of_symmetric_function_l29_29727

def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) (h_sym : ∀ x : ℝ, f x a b = f (-4 - x) a b) : 
  ∃ x : ℝ, (∀ y : ℝ, f y a b ≤ f x a b) ∧ f x a b = 16 :=
by
  sorry

end max_value_of_symmetric_function_l29_29727


namespace cost_per_pound_peanuts_l29_29697

-- Defining the conditions as needed for our problem
def one_dollar_bills := 7
def five_dollar_bills := 4
def ten_dollar_bills := 2
def twenty_dollar_bills := 1
def change := 4
def pounds_per_day := 3
def days_in_week := 7

-- Calculating the total initial amount of money Frank has
def total_initial_money := (one_dollar_bills * 1) + (five_dollar_bills * 5) + (ten_dollar_bills * 10) + (twenty_dollar_bills * 20)

-- Calculating the total amount spent on peanuts
def total_spent := total_initial_money - change

-- Calculating the total pounds of peanuts
def total_pounds := pounds_per_day * days_in_week

-- The proof statement
theorem cost_per_pound_peanuts : total_spent / total_pounds = 3 := sorry

end cost_per_pound_peanuts_l29_29697


namespace log_problem_l29_29155

open Real

noncomputable def lg (x : ℝ) := log x / log 10

theorem log_problem :
  lg 2 ^ 2 + lg 2 * lg 5 + lg 5 = 1 :=
by
  sorry

end log_problem_l29_29155


namespace geometric_sequence_a2_value_l29_29317

theorem geometric_sequence_a2_value
    (a : ℕ → ℝ)
    (h1 : a 1 = 1/5)
    (h3 : a 3 = 5)
    (geometric : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) :
    a 2 = 1 ∨ a 2 = -1 := by
  sorry

end geometric_sequence_a2_value_l29_29317


namespace parabola_directrix_l29_29285

theorem parabola_directrix (x : ℝ) :
  (y = (x^2 - 8 * x + 12) / 16) →
  (∃ y, y = -17/4) :=
by
  intro h
  sorry

end parabola_directrix_l29_29285


namespace smallest_digit_never_in_units_place_of_odd_number_l29_29865

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l29_29865


namespace polynomial_sequence_finite_functions_l29_29291

theorem polynomial_sequence_finite_functions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1) := 
by
  sorry

end polynomial_sequence_finite_functions_l29_29291


namespace Maggie_earnings_l29_29982

theorem Maggie_earnings
    (price_per_subscription : ℕ)
    (subscriptions_parents : ℕ)
    (subscriptions_grandfather : ℕ)
    (subscriptions_nextdoor : ℕ)
    (subscriptions_another : ℕ)
    (total_subscriptions : ℕ)
    (total_earnings : ℕ) :
    subscriptions_parents = 4 →
    subscriptions_grandfather = 1 →
    subscriptions_nextdoor = 2 →
    subscriptions_another = 2 * subscriptions_nextdoor →
    total_subscriptions = subscriptions_parents + subscriptions_grandfather + subscriptions_nextdoor + subscriptions_another →
    price_per_subscription = 5 →
    total_earnings = price_per_subscription * total_subscriptions →
    total_earnings = 55 :=
by
  intros
  sorry

end Maggie_earnings_l29_29982


namespace arithmetic_sequence_30th_term_l29_29505

theorem arithmetic_sequence_30th_term :
  let a := 3
  let d := 7 - 3
  ∀ n, (n = 30) → (a + (n - 1) * d) = 119 := by
  sorry

end arithmetic_sequence_30th_term_l29_29505


namespace find_arith_seq_sum_l29_29960

noncomputable def arith_seq_sum : ℕ → ℕ → ℕ
| 0, d => 2
| (n+1), d => arith_seq_sum n d + d

theorem find_arith_seq_sum :
  ∃ d : ℕ, 
    arith_seq_sum 1 d + arith_seq_sum 2 d = 13 ∧
    arith_seq_sum 3 d + arith_seq_sum 4 d + arith_seq_sum 5 d = 42 :=
by
  sorry

end find_arith_seq_sum_l29_29960


namespace percentage_of_brand_z_l29_29132

/-- Define the initial and subsequent conditions for the fuel tank -/
def initial_fuel_tank : ℕ := 1
def first_stage_z_gasoline : ℚ := 1 / 4
def first_stage_y_gasoline : ℚ := 3 / 4
def second_stage_z_gasoline : ℚ := first_stage_z_gasoline / 2 + 1 / 2
def second_stage_y_gasoline : ℚ := first_stage_y_gasoline / 2
def final_stage_z_gasoline : ℚ := second_stage_z_gasoline / 2
def final_stage_y_gasoline : ℚ := second_stage_y_gasoline / 2 + 1 / 2

/-- Formal statement of the problem: Prove the percentage of Brand Z gasoline -/
theorem percentage_of_brand_z :
  ∃ (percentage : ℚ), percentage = (final_stage_z_gasoline / (final_stage_z_gasoline + final_stage_y_gasoline)) * 100 ∧ percentage = 31.25 :=
by {
  sorry
}

end percentage_of_brand_z_l29_29132


namespace mass_percentage_Al_in_AlBr₃_l29_29691

theorem mass_percentage_Al_in_AlBr₃ :
  let Al_mass := 26.98
  let Br_mass := 79.90
  let M_AlBr₃ := Al_mass + 3 * Br_mass
  (Al_mass / M_AlBr₃ * 100) = 10.11 :=
by 
  let Al_mass := 26.98
  let Br_mass := 79.90
  let M_AlBr₃ := Al_mass + 3 * Br_mass
  have : (Al_mass / M_AlBr₃ * 100) = 10.11 := sorry
  assumption

end mass_percentage_Al_in_AlBr₃_l29_29691


namespace smallest_digit_not_found_in_units_place_of_odd_number_l29_29850

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l29_29850


namespace simplify_expression_l29_29095

variable (x : ℝ)

theorem simplify_expression :
  2 * x - 3 * (2 - x) + 4 * (1 + 3 * x) - 5 * (1 - x^2) = -5 * x^2 + 17 * x - 7 :=
by
  sorry

end simplify_expression_l29_29095


namespace smallest_digit_not_found_in_units_place_of_odd_number_l29_29848

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l29_29848


namespace smallest_digit_not_in_odd_units_l29_29858

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l29_29858


namespace megan_eggs_per_meal_l29_29082

-- Define the initial conditions
def initial_eggs_from_store : Nat := 12
def initial_eggs_from_neighbor : Nat := 12
def eggs_used_for_omelet : Nat := 2
def eggs_used_for_cake : Nat := 4
def meals_to_divide : Nat := 3

-- Calculate various steps
def total_initial_eggs : Nat := initial_eggs_from_store + initial_eggs_from_neighbor
def eggs_after_cooking : Nat := total_initial_eggs - eggs_used_for_omelet - eggs_used_for_cake
def eggs_after_giving_away : Nat := eggs_after_cooking / 2
def eggs_per_meal : Nat := eggs_after_giving_away / meals_to_divide

-- State the theorem to prove the value of eggs_per_meal
theorem megan_eggs_per_meal : eggs_per_meal = 3 := by
  sorry

end megan_eggs_per_meal_l29_29082


namespace total_students_in_high_school_l29_29450

theorem total_students_in_high_school 
  (num_freshmen : ℕ)
  (num_sample : ℕ) 
  (num_sophomores : ℕ)
  (num_seniors : ℕ)
  (freshmen_drawn : ℕ)
  (sampling_ratio : ℕ)
  (total_students : ℕ)
  (h1 : num_freshmen = 600)
  (h2 : num_sample = 45)
  (h3 : num_sophomores = 20)
  (h4 : num_seniors = 10)
  (h5 : freshmen_drawn = 15)
  (h6 : sampling_ratio = 40)
  (h7 : freshmen_drawn * sampling_ratio = num_freshmen)
  : total_students = 1800 :=
sorry

end total_students_in_high_school_l29_29450


namespace sum_of_squares_inequality_l29_29044

theorem sum_of_squares_inequality (a b c : ℝ) (h : a + 2 * b + 3 * c = 4) : a^2 + b^2 + c^2 ≥ 8 / 7 := by
  sorry

end sum_of_squares_inequality_l29_29044


namespace RS_segment_length_l29_29409

theorem RS_segment_length (P Q R S : ℝ) (r1 r2 : ℝ) (hP : P = 0) (hQ : Q = 10) (rP : r1 = 6) (rQ : r2 = 4) :
    (∃ PR QR SR : ℝ, PR = 6 ∧ QR = 4 ∧ SR = 6) → (R - S = 12) :=
by
  sorry

end RS_segment_length_l29_29409


namespace inequality_abc_l29_29325

open Real

theorem inequality_abc 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * b * c = 1) : 
  (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b))) ≥ 3 / 2 :=
sorry

end inequality_abc_l29_29325


namespace greatest_possible_lower_bound_of_sum_sq_l29_29398

-- Define the polynomial with given conditions
variables {n : ℕ} {a : ℕ → ℝ}

-- Assume the conditions given in the problem
def poly_monic (p : polynomial ℝ) : Prop :=
  p.coeff (p.nat_degree) = 1

def poly_degree_n (p : polynomial ℝ) (n : ℕ) : Prop :=
  p.nat_degree = n

def a_rel (a_(n_1) a_(n_2) : ℝ) : Prop :=
  a_(n_1) = 2 * a_(n_2)

noncomputable def sum_sq_roots (a_(n_1) a_(n_2) : ℝ) : ℝ :=
  4 * a_(n_2)^2 - 2 * a_(n_2)

-- Statement to be proved
theorem greatest_possible_lower_bound_of_sum_sq {p : polynomial ℝ} 
  (monic_p : poly_monic p) (deg_p : poly_degree_n p n) 
  (rel : a_rel (p.coeff (n-1)) (p.coeff (n-2))): 
  ∃ lb, lb = 1 / 4 ∧ 
  lb ≤ abs (sum_sq_roots (p.coeff (n-1)) (p.coeff (n-2))) :=
sorry

end greatest_possible_lower_bound_of_sum_sq_l29_29398


namespace smallest_positive_period_of_f_max_min_values_of_f_on_interval_l29_29439

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) ^ 2 + Real.cos (2 * x)

theorem smallest_positive_period_of_f : ∀ (x : ℝ), f (x + π) = f x :=
by sorry

theorem max_min_values_of_f_on_interval : ∃ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ ≤ π / 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ π / 2 ∧
  f x₁ = 0 ∧ f x₂ = 1 + Real.sqrt 2 :=
by sorry

end smallest_positive_period_of_f_max_min_values_of_f_on_interval_l29_29439


namespace num_passengers_on_second_plane_l29_29778

theorem num_passengers_on_second_plane :
  ∃ x : ℕ, 600 - (2 * 50) + 600 - (2 * x) + 600 - (2 * 40) = 1500 →
  x = 60 :=
by
  sorry

end num_passengers_on_second_plane_l29_29778


namespace cubic_identity_l29_29184

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + ac + bc = 40) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 1575 := 
by
  sorry

end cubic_identity_l29_29184


namespace probability_heads_and_3_l29_29128

noncomputable def biased_coin_heads_prob : ℝ := 0.4
def die_sides : ℕ := 8

theorem probability_heads_and_3 : biased_coin_heads_prob * (1 / die_sides) = 0.05 := sorry

end probability_heads_and_3_l29_29128


namespace high_fever_temperature_l29_29228

theorem high_fever_temperature (T t : ℝ) (h1 : T = 36) (h2 : t > 13 / 12 * T) : t > 39 :=
by
  sorry

end high_fever_temperature_l29_29228


namespace intersection_point_of_lines_PQ_RS_l29_29312

def point := ℝ × ℝ × ℝ

def P : point := (4, -3, 6)
def Q : point := (1, 10, 11)
def R : point := (3, -4, 2)
def S : point := (-1, 5, 16)

theorem intersection_point_of_lines_PQ_RS :
  let line_PQ (u : ℝ) := (4 - 3 * u, -3 + 13 * u, 6 + 5 * u)
  let line_RS (v : ℝ) := (3 - 4 * v, -4 + 9 * v, 2 + 14 * v)
  ∃ u v : ℝ,
    line_PQ u = line_RS v →
    line_PQ u = (19 / 5, 44 / 3, 23 / 3) :=
by
  sorry

end intersection_point_of_lines_PQ_RS_l29_29312


namespace employee_payment_l29_29645

theorem employee_payment 
    (total_pay : ℕ)
    (pay_A : ℕ)
    (pay_B : ℕ)
    (h1 : total_pay = 560)
    (h2 : pay_A = 3 * pay_B / 2)
    (h3 : pay_A + pay_B = total_pay) :
    pay_B = 224 :=
sorry

end employee_payment_l29_29645


namespace male_students_count_l29_29100

theorem male_students_count
  (average_all_students : ℕ → ℕ → ℚ → Prop)
  (average_male_students : ℕ → ℚ → Prop)
  (average_female_students : ℕ → ℚ → Prop)
  (F : ℕ)
  (total_average : average_all_students (F + M) (83 * M + 92 * F) 90)
  (male_average : average_male_students M 83)
  (female_average : average_female_students 28 92) :
  ∃ (M : ℕ), M = 8 :=
by {
  sorry
}

end male_students_count_l29_29100


namespace triangle_proof_l29_29454

-- Declare a structure for a triangle with given conditions
structure TriangleABC :=
  (a b c : ℝ) -- sides opposite to angles A, B, and C
  (A B C : ℝ) -- angles A, B, and C
  (R : ℝ) -- circumcircle radius
  (r : ℝ := 3) -- inradius is given as 3
  (area : ℝ := 6) -- area of the triangle is 6
  (h1 : a * Real.cos A + b * Real.cos B + c * Real.cos C = R / 3) -- given condition
  (h2 : ∀ a b c A B C, a * Real.sin A + b * Real.sin B + c * Real.sin C = 2 * area / (a+b+c)) -- implied area condition

-- Define the theorem using the above conditions
theorem triangle_proof (t : TriangleABC) :
  t.a + t.b + t.c = 4 ∧
  (Real.sin (2 * t.A) + Real.sin (2 * t.B) + Real.sin (2 * t.C)) = 1/3 ∧
  t.R = 6 :=
by
  sorry

end triangle_proof_l29_29454


namespace complex_transformation_l29_29380

open Complex

def dilation (z : ℂ) (center : ℂ) (scale : ℝ) : ℂ :=
  center + scale * (z - center)

def rotation90 (z : ℂ) : ℂ :=
  z * I

theorem complex_transformation (z : ℂ) (center : ℂ) (scale : ℝ) :
  center = -1 + 2 * I → scale = 2 → z = 3 + I →
  rotation90 (dilation z center scale) = 4 + 7 * I :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  dsimp [dilation]
  dsimp [rotation90]
  sorry

end complex_transformation_l29_29380


namespace solve_for_y_l29_29011

theorem solve_for_y (y : ℚ) (h : |(4 : ℚ) * y - 6| = 0) : y = 3 / 2 :=
sorry

end solve_for_y_l29_29011


namespace distinct_numerators_count_l29_29201

-- Define the set S as described
def S : Set ℚ := {x : ℚ | ∃ a b : ℕ, 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ x = (10 * a + b) / 99 ∧ 0 < x ∧ x < 1}

-- Define the main theorem
theorem distinct_numerators_count : (Finset.map (λ r : ℚ, r.num) (Finset.filter (λ r : ℚ, r.denom.gcd (r.num.gcd 99) = 1) (Finset.image (λ r : ℚ, Rat.mk' r.num r.denom) (Finset.filter (∈ S) Finset.univ)))).card = 53 := 
sorry

end distinct_numerators_count_l29_29201


namespace harper_water_duration_l29_29443

theorem harper_water_duration
  (half_bottle_per_day : ℝ)
  (bottles_per_case : ℕ)
  (cost_per_case : ℝ)
  (total_spending : ℝ)
  (cases_bought : ℕ)
  (days_per_case : ℕ)
  (total_days : ℕ) :
  half_bottle_per_day = 1/2 →
  bottles_per_case = 24 →
  cost_per_case = 12 →
  total_spending = 60 →
  cases_bought = total_spending / cost_per_case →
  days_per_case = bottles_per_case * 2 →
  total_days = days_per_case * cases_bought →
  total_days = 240 :=
by
  -- The proof will be added here
  sorry

end harper_water_duration_l29_29443


namespace chenny_friends_l29_29904

noncomputable def num_friends (initial_candies add_candies candies_per_friend) : ℕ :=
  (initial_candies + add_candies) / candies_per_friend

theorem chenny_friends :
  num_friends 10 4 2 = 7 :=
by
  sorry

end chenny_friends_l29_29904


namespace geom_sequence_eq_l29_29422

theorem geom_sequence_eq :
  ∀ {a : ℕ → ℝ} {q : ℝ}, (∀ n, a (n + 1) = q * a n) →
  a 1 + a 2 + a 3 + a 4 + a 5 = 3 →
  a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 = 15 →
  a 1 - a 2 + a 3 - a 4 + a 5 = 5 :=
by
  intro a q hgeom hsum hsum_sq
  sorry

end geom_sequence_eq_l29_29422


namespace one_zero_implies_a_eq_pm2_zero_in_interval_implies_a_in_open_interval_l29_29720

def f (a x : ℝ) : ℝ := x^2 - a * x + 1

theorem one_zero_implies_a_eq_pm2 (a : ℝ) : (∃! x, f a x = 0) → (a = 2 ∨ a = -2) := by
  sorry

theorem zero_in_interval_implies_a_in_open_interval (a : ℝ) : (∃ x, f a x = 0 ∧ 0 < x ∧ x < 1) → 2 < a := by
  sorry

end one_zero_implies_a_eq_pm2_zero_in_interval_implies_a_in_open_interval_l29_29720


namespace students_failed_to_get_degree_l29_29614

/-- 
Out of 1,500 senior high school students, 70% passed their English exams,
80% passed their Mathematics exams, and 65% passed their Science exams.
To get their degree, a student must pass in all three subjects.
Assume independence of passing rates. This Lean proof shows that
the number of students who failed to get their degree is 954.
-/
theorem students_failed_to_get_degree :
  let total_students := 1500
  let p_english := 0.70
  let p_math := 0.80
  let p_science := 0.65
  let p_all_pass := p_english * p_math * p_science
  let students_all_pass := p_all_pass * total_students
  total_students - students_all_pass = 954 :=
by
  sorry

end students_failed_to_get_degree_l29_29614


namespace part1_part2_l29_29575

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 2) - abs (x + 1)

theorem part1 (x : ℝ) : f x ≤ 3 ↔ -2/3 ≤ x ∧ x ≤ 6 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, f x ≤ abs (x + 1) + a^2) ↔ a ≤ -2 ∨ 2 ≤ a :=
by
  sorry

end part1_part2_l29_29575


namespace arithmetic_sequence_a8_l29_29315

variable (a : ℕ → ℝ)
variable (a2_eq : a 2 = 4)
variable (a6_eq : a 6 = 2)

theorem arithmetic_sequence_a8 :
  a 8 = 1 :=
sorry

end arithmetic_sequence_a8_l29_29315


namespace percentage_difference_l29_29884

theorem percentage_difference :
  let x := 50
  let y := 30
  let p1 := 60
  let p2 := 30
  (p1 / 100 * x) - (p2 / 100 * y) = 21 :=
by
  sorry

end percentage_difference_l29_29884


namespace smallest_digit_never_in_units_place_of_odd_number_l29_29788

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l29_29788


namespace find_m_l29_29573

-- Define the condition that the equation has a positive root
def hasPositiveRoot (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (2 / (x - 2) = 1 - (m / (x - 2)))

-- State the theorem
theorem find_m : ∀ m : ℝ, hasPositiveRoot m → m = -2 :=
by
  sorry

end find_m_l29_29573


namespace unique_solution_of_system_l29_29287

theorem unique_solution_of_system :
  ∀ (a : ℝ), (∃ (x y : ℝ), x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) →
  ((a = 1 ∧ ∃ x y : ℝ, x = -1 ∧ y = 0 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) ∨
  (a = -3 ∧ ∃ x y : ℝ, x = 1 ∧ y = 2 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0)) :=
by
  sorry

end unique_solution_of_system_l29_29287


namespace common_root_equations_l29_29190

theorem common_root_equations (a b : ℝ) 
  (h : ∃ x₀ : ℝ, (x₀ ^ 2 + a * x₀ + b = 0) ∧ (x₀ ^ 2 + b * x₀ + a = 0)) 
  (hc : ∀ x₁ x₂ : ℝ, (x₁ ^ 2 + a * x₁ + b = 0 ∧ x₂ ^ 2 + bx₀ + a = 0) → x₁ = x₂) :
  a + b = -1 :=
sorry

end common_root_equations_l29_29190


namespace sin_trig_identity_l29_29289

theorem sin_trig_identity (α : ℝ) (h : Real.sin (α - π/4) = 1/2) : Real.sin ((5 * π) / 4 - α) = 1/2 := 
by 
  sorry

end sin_trig_identity_l29_29289


namespace find_a_l29_29707

theorem find_a (a x : ℝ) (h : x = -1) (heq : -2 * (x - a) = 4) : a = 1 :=
by
  sorry

end find_a_l29_29707


namespace smallest_digit_not_in_odd_units_l29_29840

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l29_29840


namespace ratio_of_area_l29_29352

noncomputable def area_ratio (l w r : ℝ) : ℝ :=
  if h1 : 2 * l + 2 * w = 2 * Real.pi * r 
  ∧ l = 2 * w then 
    (l * w) / (Real.pi * r ^ 2) 
  else 
    0

theorem ratio_of_area (l w r : ℝ) 
  (h1 : 2 * l + 2 * w = 2 * Real.pi * r) 
  (h2 : l = 2 * w) :
  area_ratio l w r = 2 * Real.pi / 9 :=
by
  unfold area_ratio
  simp [h1, h2]
  sorry

end ratio_of_area_l29_29352


namespace situation1_correct_situation2_correct_situation3_correct_l29_29358

noncomputable def situation1 : Nat :=
  let choices_for_A := 4
  let remaining_perm := Nat.factorial 6
  choices_for_A * remaining_perm

theorem situation1_correct : situation1 = 2880 := by
  sorry

noncomputable def situation2 : Nat :=
  let permutations_A_B := Nat.factorial 2
  let remaining_perm := Nat.factorial 5
  permutations_A_B * remaining_perm

theorem situation2_correct : situation2 = 240 := by
  sorry

noncomputable def situation3 : Nat :=
  let perm_boys := Nat.factorial 3
  let perm_girls := Nat.factorial 4
  perm_boys * perm_girls

theorem situation3_correct : situation3 = 144 := by
  sorry

end situation1_correct_situation2_correct_situation3_correct_l29_29358


namespace astroid_area_l29_29666

-- Definitions coming from the conditions
noncomputable def x (t : ℝ) := 4 * (Real.cos t)^3
noncomputable def y (t : ℝ) := 4 * (Real.sin t)^3

-- The theorem stating the area of the astroid
theorem astroid_area : (∫ t in (0 : ℝ)..(Real.pi / 2), y t * (deriv x t)) * 4 = 24 * Real.pi :=
by
  sorry

end astroid_area_l29_29666


namespace calculate_P_1_lt_X_lt_3_l29_29702

variable (X : ℝ → ℝ)
variable (hX : X ~ Normal(3, σ^2))
variable (h1 : P(X < 5) = 0.8)

theorem calculate_P_1_lt_X_lt_3 : P(1 < X < 3) = 0.3 :=
sorry

end calculate_P_1_lt_X_lt_3_l29_29702


namespace find_a_l29_29172

theorem find_a (a : ℝ) (h : (1 - 2016 * a) = 2017) : a = -1 := by
  -- proof omitted
  sorry

end find_a_l29_29172


namespace find_m_l29_29438

theorem find_m (
  x : ℚ 
) (m : ℚ) 
  (h1 : 4 * x + 2 * m = 3 * x + 1) 
  (h2 : 3 * x + 2 * m = 6 * x + 1) 
: m = 1/2 := 
  sorry

end find_m_l29_29438


namespace sum_of_numbers_l29_29136

theorem sum_of_numbers : 4.75 + 0.303 + 0.432 = 5.485 :=
by
  -- The proof will be filled here
  sorry

end sum_of_numbers_l29_29136


namespace angle_BCA_measure_l29_29453

theorem angle_BCA_measure
  (A B C : Type)
  (angle_ABC : ℝ)
  (angle_BCA : ℝ)
  (angle_BAC : ℝ)
  (h1 : angle_ABC = 90)
  (h2 : angle_BAC = 2 * angle_BCA) :
  angle_BCA = 30 :=
by
  sorry

end angle_BCA_measure_l29_29453


namespace maggie_earnings_proof_l29_29983

def rate_per_subscription : ℕ := 5
def subscriptions_to_parents : ℕ := 4
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_neighbor1 : ℕ := 2
def subscriptions_to_neighbor2 : ℕ := 2 * subscriptions_to_neighbor1
def total_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_neighbor1 + subscriptions_to_neighbor2
def total_earnings : ℕ := total_subscriptions * rate_per_subscription

theorem maggie_earnings_proof : total_earnings = 55 := by
  sorry

end maggie_earnings_proof_l29_29983


namespace average_percent_score_is_77_l29_29331

def numberOfStudents : ℕ := 100

def percentage_counts : List (ℕ × ℕ) :=
[(100, 7), (90, 18), (80, 35), (70, 25), (60, 10), (50, 3), (40, 2)]

noncomputable def average_score (counts : List (ℕ × ℕ)) : ℚ :=
  (counts.foldl (λ acc p => acc + (p.1 * p.2)) 0 : ℚ) / numberOfStudents

theorem average_percent_score_is_77 : average_score percentage_counts = 77 := by
  sorry

end average_percent_score_is_77_l29_29331


namespace ott_fraction_l29_29087

/-- 
Moe, Loki, Nick, and Pat each give $2 to Ott.
Moe gave Ott one-seventh of his money.
Loki gave Ott one-fifth of his money.
Nick gave Ott one-fourth of his money.
Pat gave Ott one-sixth of his money.
-/
def fraction_of_money_ott_now_has (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) : Prop :=
  A = 14 ∧ B = 10 ∧ C = 8 ∧ D = 12 ∧ (2 * (1 / 7 : ℚ)) = 2 ∧ (2 * (1 / 5 : ℚ)) = 2 ∧ (2 * (1 / 4 : ℚ)) = 2 ∧ (2 * (1 / 6 : ℚ)) = 2

theorem ott_fraction (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) (h : fraction_of_money_ott_now_has A B C D) : 
  8 = (2 / 11 : ℚ) * (A + B + C + D) :=
by sorry

end ott_fraction_l29_29087


namespace internet_bill_is_100_l29_29476

theorem internet_bill_is_100 (initial_amount rent paycheck electricity_bill phone_bill final_amount internet_bill : ℝ)
  (h1 : initial_amount = 800)
  (h2 : rent = 450)
  (h3 : paycheck = 1500)
  (h4 : electricity_bill = 117)
  (h5 : phone_bill = 70)
  (h6 : final_amount = 1563)
  (h7 : initial_amount - rent + paycheck - electricity_bill - internet_bill - phone_bill = final_amount) :
  internet_bill = 100 :=
by
  sorry

end internet_bill_is_100_l29_29476


namespace satellite_modular_units_l29_29526

variable (U N S T : ℕ)

def condition1 : Prop := N = (1/8 : ℝ) * S
def condition2 : Prop := T = 4 * S
def condition3 : Prop := U * N = 3 * S

theorem satellite_modular_units
  (h1 : condition1 N S)
  (h2 : condition2 T S)
  (h3 : condition3 U N S) :
  U = 24 :=
sorry

end satellite_modular_units_l29_29526


namespace prime_solution_l29_29917

theorem prime_solution (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ (p = 2 ∨ p = 3 ∨ p = 7) :=
by
  sorry

end prime_solution_l29_29917


namespace new_person_weight_l29_29247

-- Define the initial conditions
def initial_average_weight (w : ℕ) := 6 * w -- The total weight of 6 persons

-- Define the scenario where the average weight increases by 2 kg
def total_weight_increase := 6 * 2 -- The total increase in weight due to an increase of 2 kg in average weight

def person_replaced := 75 -- The weight of the person being replaced

-- Define the expected condition on the weight of the new person
theorem new_person_weight (w_new : ℕ) :
  initial_average_weight person_replaced + total_weight_increase = initial_average_weight (w_new / 6) →
  w_new = 87 :=
sorry

end new_person_weight_l29_29247


namespace ceil_sqrt_225_l29_29016

theorem ceil_sqrt_225 : ⌈real.sqrt 225⌉ = 15 :=
by
  have h : real.sqrt 225 = 15 := by
    sorry
  rw [h]
  exact int.ceil_eq_self.mpr rfl

end ceil_sqrt_225_l29_29016


namespace probability_even_sum_of_selected_envelopes_l29_29237

theorem probability_even_sum_of_selected_envelopes :
  let face_values := [5, 6, 8, 10]
  let possible_sum_is_even (s : ℕ) : Prop := s % 2 = 0
  let num_combinations := Nat.choose 4 2
  let favorable_combinations := 3
  (favorable_combinations / num_combinations : ℚ) = 1 / 2 :=
by
  sorry

end probability_even_sum_of_selected_envelopes_l29_29237


namespace range_a_implies_not_purely_imaginary_l29_29187

def is_not_purely_imaginary (z : ℂ) : Prop :=
  z.re ≠ 0

theorem range_a_implies_not_purely_imaginary (a : ℝ) :
  ¬ is_not_purely_imaginary ⟨a^2 - a - 2, abs (a - 1) - 1⟩ ↔ a ≠ -1 :=
by
  sorry

end range_a_implies_not_purely_imaginary_l29_29187


namespace probability_sales_greater_than_10000_l29_29193

/-- Define the probability that the sales of new energy vehicles in a randomly selected city are greater than 10000 -/
theorem probability_sales_greater_than_10000 :
  (1 / 2) * (2 / 10) + (1 / 2) * (6 / 10) = 2 / 5 :=
by sorry

end probability_sales_greater_than_10000_l29_29193


namespace volume_related_to_area_l29_29274

theorem volume_related_to_area (x y z : ℝ) 
  (bottom_area_eq : 3 * x * y = 3 * x * y)
  (front_area_eq : 2 * y * z = 2 * y * z)
  (side_area_eq : 3 * x * z = 3 * x * z) :
  (3 * x * y) * (2 * y * z) * (3 * x * z) = 18 * (x * y * z) ^ 2 := 
by sorry

end volume_related_to_area_l29_29274


namespace average_weight_l29_29996

/-- 
Given the following conditions:
1. (A + B) / 2 = 40
2. (B + C) / 2 = 41
3. B = 27
Prove that the average weight of a, b, and c is 45 kg.
-/
theorem average_weight (A B C : ℝ) 
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 41)
  (h3 : B = 27): 
  (A + B + C) / 3 = 45 :=
by
  sorry

end average_weight_l29_29996


namespace histogram_groups_l29_29591

theorem histogram_groups 
  (max_height : ℕ)
  (min_height : ℕ)
  (class_interval : ℕ)
  (h_max : max_height = 176)
  (h_min : min_height = 136)
  (h_interval : class_interval = 6) :
  Nat.ceil ((max_height - min_height) / class_interval) = 7 :=
by
  sorry

end histogram_groups_l29_29591


namespace quotient_of_x6_plus_8_by_x_minus_1_l29_29921

theorem quotient_of_x6_plus_8_by_x_minus_1 :
  ∀ (x : ℝ), x ≠ 1 →
  (∃ Q : ℝ → ℝ, x^6 + 8 = (x - 1) * Q x + 9 ∧ Q x = x^5 + x^4 + x^3 + x^2 + x + 1) := 
  by
    intros x hx
    sorry

end quotient_of_x6_plus_8_by_x_minus_1_l29_29921


namespace paint_cost_is_200_l29_29103

-- Define the basic conditions and parameters
def side_length : ℕ := 5
def faces_of_cube : ℕ := 6
def area_per_face (side : ℕ) : ℕ := side * side
def total_surface_area (side : ℕ) (faces : ℕ) : ℕ := faces * area_per_face side
def coverage_per_kg : ℕ := 15
def cost_per_kg : ℕ := 20

-- Calculate total cost
def total_cost (side : ℕ) (faces : ℕ) (coverage : ℕ) (cost : ℕ) : ℕ :=
  let total_area := total_surface_area side faces
  let kgs_required := total_area / coverage
  kgs_required * cost

theorem paint_cost_is_200 :
  total_cost side_length faces_of_cube coverage_per_kg cost_per_kg = 200 :=
by
  sorry

end paint_cost_is_200_l29_29103


namespace square_combinations_l29_29943

theorem square_combinations (n : ℕ) (h : n * (n - 1) = 30) : n * (n - 1) = 30 :=
by sorry

end square_combinations_l29_29943


namespace simplify_expression_l29_29992

theorem simplify_expression : 
  (4 * 6 / (12 * 14)) * (8 * 12 * 14 / (4 * 6 * 8)) = 1 := by
  sorry

end simplify_expression_l29_29992


namespace find_initial_quarters_l29_29979

-- Define the initial number of dimes, nickels, and quarters (unknown)
def initial_dimes : ℕ := 2
def initial_nickels : ℕ := 5
def initial_quarters (Q : ℕ) := Q

-- Define the additional coins given by Linda’s mother
def additional_dimes : ℕ := 2
def additional_quarters : ℕ := 10
def additional_nickels : ℕ := 2 * initial_nickels

-- Define the total number of each type of coin after Linda receives the additional coins
def total_dimes : ℕ := initial_dimes + additional_dimes
def total_quarters (Q : ℕ) : ℕ := additional_quarters + initial_quarters Q
def total_nickels : ℕ := initial_nickels + additional_nickels

-- Define the total number of coins
def total_coins (Q : ℕ) : ℕ := total_dimes + total_quarters Q + total_nickels

theorem find_initial_quarters : ∃ Q : ℕ, total_coins Q = 35 ∧ Q = 6 := by
  -- Provide the corresponding proof here
  sorry

end find_initial_quarters_l29_29979


namespace divides_14_pow_n_minus_27_for_all_natural_numbers_l29_29925

theorem divides_14_pow_n_minus_27_for_all_natural_numbers :
  ∀ n : ℕ, 13 ∣ 14^n - 27 :=
by sorry

end divides_14_pow_n_minus_27_for_all_natural_numbers_l29_29925


namespace wall_width_l29_29248

theorem wall_width
  (w h l : ℝ)
  (h_eq : h = 6 * w)
  (l_eq : l = 7 * h)
  (volume_eq : w * h * l = 6804) :
  w = 3 :=
by
  sorry

end wall_width_l29_29248


namespace problem1_problem2_l29_29180

-- Definitions
def vec_a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)
def sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Problem 1: Prove the value of m such that vec_a ⊥ (vec_a - b(m))
theorem problem1 (m : ℝ) (h_perp: dot vec_a (sub vec_a (b m)) = 0) : m = -4 := sorry

-- Problem 2: Prove the value of k such that k * vec_a + b(-4) is parallel to vec_a - b(-4)
def scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def parallel (u v : ℝ × ℝ) := ∃ (k : ℝ), scale k u = v

theorem problem2 (k : ℝ) (h_parallel: parallel (add (scale k vec_a) (b (-4))) (sub vec_a (b (-4)))) : k = -1 := sorry

end problem1_problem2_l29_29180


namespace A_n_squared_l29_29945

-- Define C(n-2)
def C_n_2 (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define A_n_2
def A_n_2 (n : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - 2)

theorem A_n_squared (n : ℕ) (hC : C_n_2 n = 15) : A_n_2 n = 30 := by
  sorry

end A_n_squared_l29_29945


namespace choir_members_correct_l29_29222

def choir_members_condition (n : ℕ) : Prop :=
  150 < n ∧ n < 250 ∧ 
  n % 3 = 1 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 3

theorem choir_members_correct : ∃ n, choir_members_condition n ∧ (n = 195 ∨ n = 219) :=
by {
  sorry
}

end choir_members_correct_l29_29222


namespace smallest_not_odd_unit_is_zero_l29_29804

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l29_29804


namespace polygon_D_has_largest_area_l29_29511

noncomputable def area_A := 4 * 1 + 2 * (1 / 2) -- 5
noncomputable def area_B := 2 * 1 + 2 * (1 / 2) + Real.pi / 4 -- ≈ 3.785
noncomputable def area_C := 3 * 1 + 3 * (1 / 2) -- 4.5
noncomputable def area_D := 3 * 1 + 1 * (1 / 2) + 2 * (Real.pi / 4) -- ≈ 5.07
noncomputable def area_E := 1 * 1 + 3 * (1 / 2) + 3 * (Real.pi / 4) -- ≈ 4.855

theorem polygon_D_has_largest_area :
  area_D > area_A ∧
  area_D > area_B ∧
  area_D > area_C ∧
  area_D > area_E :=
by
  sorry

end polygon_D_has_largest_area_l29_29511


namespace midpoint_product_l29_29200

theorem midpoint_product (x y : ℝ) :
  (∃ B : ℝ × ℝ, B = (x, y) ∧ 
  (4, 6) = ( (2 + B.1) / 2, (9 + B.2) / 2 )) → x * y = 18 :=
by
  -- Placeholder for the proof
  sorry

end midpoint_product_l29_29200


namespace find_digits_of_abc_l29_29094

theorem find_digits_of_abc (a b c : ℕ) (h1 : a ≠ c) (h2 : c - a = 3) (h3 : (100 * a + 10 * b + c) - (100 * c + 10 * a + b) = 100 * (a - (c - 1)) + 0 + (b - b)) : 
  100 * a + 10 * b + c = 619 :=
by
  sorry

end find_digits_of_abc_l29_29094


namespace ellipse_range_m_l29_29191

theorem ellipse_range_m (m : ℝ) :
    (∀ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2 → 
    ∃ (c : ℝ), c = x^2 + (y + 1)^2 ∧ m > 5) :=
sorry

end ellipse_range_m_l29_29191


namespace isosceles_triangle_perimeter_l29_29929

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3 ∨ b = 3) (h2 : a = 6 ∨ b = 6) 
(h_isosceles : a = b ∨ b = a) : 
  a + b + a = 15 ∨ b + a + b = 15 :=
by sorry

end isosceles_triangle_perimeter_l29_29929


namespace smallest_missing_digit_l29_29859

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l29_29859


namespace evaluate_expression_l29_29245

noncomputable def given_expression : ℝ :=
  |8 - 8 * (3 - 12)^2| - |5 - Real.sin 11| + |2^(4 - 2 * 3) / ((3^2) - 7)|

theorem evaluate_expression : given_expression = 634.125009794 := 
  sorry

end evaluate_expression_l29_29245


namespace smallest_digit_not_in_units_place_of_odd_l29_29794

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l29_29794


namespace sum_b_div_5_pow_eq_l29_29743

namespace SequenceSumProblem

-- Define the sequence b_n
def b : ℕ → ℝ
| 0       => 2
| 1       => 3
| (n + 2) => b (n + 1) + b n

-- The infinite series sum we need to prove
noncomputable def sum_b_div_5_pow (Y : ℝ) : Prop :=
  Y = ∑' n : ℕ, (b n) / (5 ^ (n + 1))

-- The statement of the problem
theorem sum_b_div_5_pow_eq : sum_b_div_5_pow (2 / 25) :=
sorry

end SequenceSumProblem

end sum_b_div_5_pow_eq_l29_29743


namespace correct_commutative_property_usage_l29_29127

-- Definitions for the transformations
def transformA := 3 + (-2) = 2 + 3
def transformB := 4 + (-6) + 3 = (-6) + 4 + 3
def transformC := (5 + (-2)) + 4 = (5 + (-4)) + 2
def transformD := (1 / 6) + (-1) + (5 / 6) = ((1 / 6) + (5 / 6)) + 1

-- The theorem stating that transformB uses the commutative property correctly
theorem correct_commutative_property_usage : transformB :=
by
  sorry

end correct_commutative_property_usage_l29_29127


namespace largest_divisor_of_m_l29_29246

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 216 ∣ m^2) : 36 ∣ m :=
sorry

end largest_divisor_of_m_l29_29246


namespace mira_additional_stickers_l29_29329

-- Define the conditions
def mira_stickers : ℕ := 31
def row_size : ℕ := 7

-- Define the proof statement
theorem mira_additional_stickers (a : ℕ) (h : (31 + a) % 7 = 0) : 
  a = 4 := 
sorry

end mira_additional_stickers_l29_29329


namespace number_of_possible_values_for_b_l29_29342

theorem number_of_possible_values_for_b : 
  ∃ (n : ℕ), n = 2 ∧ ∀ b : ℕ, (b ≥ 2 ∧ b^3 ≤ 197 ∧ 197 < b^4) → b = 4 ∨ b = 5 :=
sorry

end number_of_possible_values_for_b_l29_29342


namespace ab_plus_b_l29_29276

theorem ab_plus_b (A B : ℤ) (h1 : A * B = 10) (h2 : 3 * A + 7 * B = 51) : A * B + B = 12 :=
by
  sorry

end ab_plus_b_l29_29276


namespace least_value_QGK_l29_29124

theorem least_value_QGK :
  ∃ (G K Q : ℕ), (10 * G + G) * G = 100 * Q + 10 * G + K ∧ G ≠ K ∧ (10 * G + G) ≥ 10 ∧ (10 * G + G) < 100 ∧  ∃ x, x = 44 ∧ 100 * G + 10 * 4 + 4 = (100 * Q + 10 * G + K) ∧ 100 * 0 + 10 * 4 + 4 = 044  :=
by
  sorry

end least_value_QGK_l29_29124


namespace combined_average_yield_l29_29675

variable (YieldA : ℝ) (PriceA : ℝ)
variable (YieldB : ℝ) (PriceB : ℝ)
variable (YieldC : ℝ) (PriceC : ℝ)

def AnnualIncome (Yield : ℝ) (Price : ℝ) : ℝ :=
  Yield * Price

def TotalAnnualIncome (IncomeA IncomeB IncomeC : ℝ) : ℝ :=
  IncomeA + IncomeB + IncomeC

def TotalInvestment (PriceA PriceB PriceC : ℝ) : ℝ :=
  PriceA + PriceB + PriceC

def CombinedAverageYield (TotalIncome TotalInvestment : ℝ) : ℝ :=
  TotalIncome / TotalInvestment

theorem combined_average_yield :
  YieldA = 0.20 → PriceA = 100 → YieldB = 0.12 → PriceB = 200 → YieldC = 0.25 → PriceC = 300 →
  CombinedAverageYield
    (TotalAnnualIncome
      (AnnualIncome YieldA PriceA)
      (AnnualIncome YieldB PriceB)
      (AnnualIncome YieldC PriceC))
    (TotalInvestment PriceA PriceB PriceC) = 0.1983 :=
by
  intro hYA hPA hYB hPB hYC hPC
  rw [hYA, hPA, hYB, hPB, hYC, hPC]
  sorry

end combined_average_yield_l29_29675


namespace customer_payment_probability_l29_29086

theorem customer_payment_probability :
  let total_customers := 100
  let age_40_50_non_mobile := 13
  let age_50_60_non_mobile := 27
  let total_40_60_non_mobile := age_40_50_non_mobile + age_50_60_non_mobile
  let probability := (total_40_60_non_mobile : ℚ) / total_customers
  probability = 2 / 5 := by
sorry

end customer_payment_probability_l29_29086


namespace set_intersection_complement_l29_29937

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 3}
noncomputable def B : Set ℕ := {2, 3}

theorem set_intersection_complement :
  A ∩ (U \ B) = {1} :=
sorry

end set_intersection_complement_l29_29937


namespace calculate_fraction_l29_29154

theorem calculate_fraction :
  let a := 7
  let b := 5
  let c := -2
  (a^3 + b^3 + c^3) / (a^2 - a * b + b^2 + c^2) = 460 / 43 :=
by
  sorry

end calculate_fraction_l29_29154


namespace train_cross_time_l29_29638

/-- Given the conditions:
1. Two trains run in opposite directions and cross a man in 17 seconds and some unknown time respectively.
2. They cross each other in 22 seconds.
3. The ratio of their speeds is 1 to 1.
Prove the time it takes for the first train to cross the man. -/
theorem train_cross_time (v_1 v_2 L_1 L_2 : ℝ) (t_2 : ℝ) (h1 : t_2 = 17) (h2 : v_1 = v_2)
  (h3 : (L_1 + L_2) / (v_1 + v_2) = 22) : (L_1 / v_1) = 27 := 
by 
  -- The actual proof will go here
  sorry

end train_cross_time_l29_29638


namespace measure_of_angle_C_l29_29392

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 180) (h2 : C = 12 * D) : C = 2160 / 13 :=
by {
  sorry
}

end measure_of_angle_C_l29_29392


namespace units_digit_of_product_of_seven_consecutive_l29_29775

theorem units_digit_of_product_of_seven_consecutive (n : ℕ) : 
  ∃ d ∈ [n, n+1, n+2, n+3, n+4, n+5, n+6], d % 10 = 0 :=
by
  sorry

end units_digit_of_product_of_seven_consecutive_l29_29775


namespace circumcircles_touch_each_other_l29_29608

open EuclideanGeometry

variables {k1 k2 : Circle} {A P Q R S : Point}

axiom common_tangent1 : tangent k1 P ∧ tangent k2 Q
axiom common_tangent2 : tangent k1 R ∧ tangent k2 S
axiom intersection_A : A ∈ k1 ∧ A ∈ k2
axiom different_points: P ≠ Q ∧ R ≠ S ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S

theorem circumcircles_touch_each_other :
  (∃ γ1 γ2 : Circle, circumcircle (Triangle.mk P A Q) γ1 ∧ circumcircle (Triangle.mk R A S) γ2 ∧
    tangent γ1 A ∧ tangent γ2 A) ∧ (γ1 = γ2) :=
by
  sorry

end circumcircles_touch_each_other_l29_29608


namespace quadratic_has_two_real_roots_l29_29065

theorem quadratic_has_two_real_roots (k : ℝ) (h1 : k ≠ 0) (h2 : 4 - 12 * k ≥ 0) : 0 < k ∧ k ≤ 1 / 3 :=
sorry

end quadratic_has_two_real_roots_l29_29065


namespace calculate_expression_l29_29270

theorem calculate_expression : (3^5 * 4^5) / 6^5 = 32 := 
by
  sorry

end calculate_expression_l29_29270


namespace smallest_digit_never_in_units_place_of_odd_l29_29832

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l29_29832


namespace problem_given_conditions_l29_29295

theorem problem_given_conditions (x y z : ℝ) 
  (h : x / 3 = y / (-4) ∧ y / (-4) = z / 7) : (3 * x + y + z) / y = -3 := 
by 
  sorry

end problem_given_conditions_l29_29295


namespace square_not_covered_by_circles_l29_29037

noncomputable def area_uncovered_by_circles : Real :=
  let side_length := 2
  let square_area := (side_length^2 : Real)
  let radius := 1
  let circle_area := Real.pi * radius^2
  let quarter_circle_area := circle_area / 4
  let total_circles_area := 4 * quarter_circle_area
  square_area - total_circles_area

theorem square_not_covered_by_circles :
  area_uncovered_by_circles = 4 - Real.pi := sorry

end square_not_covered_by_circles_l29_29037


namespace small_trucks_needed_l29_29404

-- Defining the problem's conditions
def total_flour : ℝ := 500
def large_truck_capacity : ℝ := 9.6
def num_large_trucks : ℝ := 40
def small_truck_capacity : ℝ := 4

-- Theorem statement to find the number of small trucks needed
theorem small_trucks_needed : (total_flour - (num_large_trucks * large_truck_capacity)) / small_truck_capacity = (500 - (40 * 9.6)) / 4 :=
by
  sorry

end small_trucks_needed_l29_29404


namespace perimeter_triangle_ABC_eq_18_l29_29137

theorem perimeter_triangle_ABC_eq_18 (h1 : ∀ (Δ : ℕ), Δ = 9) 
(h2 : ∀ (p : ℕ), p = 6) : 
∀ (perimeter_ABC : ℕ), perimeter_ABC = 18 := by
sorry

end perimeter_triangle_ABC_eq_18_l29_29137


namespace budget_allocation_degrees_l29_29379

theorem budget_allocation_degrees :
  let microphotonics := 12.3
  let home_electronics := 17.8
  let food_additives := 9.4
  let gmo := 21.7
  let industrial_lubricants := 6.2
  let artificial_intelligence := 4.1
  let nanotechnology := 5.3
  let basic_astrophysics := 100 - (microphotonics + home_electronics + food_additives + gmo + industrial_lubricants + artificial_intelligence + nanotechnology)
  (basic_astrophysics * 3.6) + (artificial_intelligence * 3.6) + (nanotechnology * 3.6) = 117.36 :=
by
  sorry

end budget_allocation_degrees_l29_29379


namespace ryegrass_percentage_l29_29990

theorem ryegrass_percentage (x_ryegrass_percent : ℝ) (y_ryegrass_percent : ℝ) (mixture_x_percent : ℝ)
  (hx : x_ryegrass_percent = 0.40)
  (hy : y_ryegrass_percent = 0.25)
  (hmx : mixture_x_percent = 0.8667) :
  (x_ryegrass_percent * mixture_x_percent + y_ryegrass_percent * (1 - mixture_x_percent)) * 100 = 38 :=
by
  sorry

end ryegrass_percentage_l29_29990


namespace area_of_segment_solution_max_sector_angle_solution_l29_29928
open Real

noncomputable def area_of_segment (α R : ℝ) : ℝ :=
  let l := (R * α)
  let sector := 0.5 * R * l
  let triangle := 0.5 * R^2 * sin α
  sector - triangle

theorem area_of_segment_solution : area_of_segment (π / 3) 10 = 50 * ((π / 3) - (sqrt 3 / 2)) :=
by sorry

noncomputable def max_sector_angle (c : ℝ) (hc : c > 0) : ℝ :=
  2

theorem max_sector_angle_solution (c : ℝ) (hc : c > 0) : max_sector_angle c hc = 2 :=
by sorry

end area_of_segment_solution_max_sector_angle_solution_l29_29928


namespace chenny_friends_count_l29_29900

def initial_candies := 10
def additional_candies := 4
def candies_per_friend := 2

theorem chenny_friends_count :
  (initial_candies + additional_candies) / candies_per_friend = 7 := by
  sorry

end chenny_friends_count_l29_29900


namespace range_of_a_l29_29066

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ 2 * x^2 - a * x + 2 > 0) ↔ a < 5 := sorry

end range_of_a_l29_29066


namespace area_R2_l29_29040

-- Definitions from conditions
def side_R1 : ℕ := 3
def area_R1 : ℕ := 24
def diagonal_ratio : ℤ := 2

-- Introduction of the theorem
theorem area_R2 (similar: ℤ) (a b: ℕ) :
  a * b = area_R1 ∧
  a = 3 ∧
  b * 3 = 8 * a ∧
  (a^2 + b^2 = 292) ∧
  similar * (a^2 + b^2) = 2 * 2 * 73 →
  (6 * 16 = 96) := by
sorry

end area_R2_l29_29040


namespace set_cannot_be_divided_l29_29763

theorem set_cannot_be_divided
  (p : ℕ) (prime_p : Nat.Prime p) (p_eq_3_mod_4 : p % 4 = 3)
  (S : Finset ℕ) (hS : S.card = p - 1) :
  ¬∃ A B : Finset ℕ, A ∪ B = S ∧ A ∩ B = ∅ ∧ A.prod id = B.prod id := 
by {
  sorry
}

end set_cannot_be_divided_l29_29763


namespace ceil_sqrt_225_eq_15_l29_29023

theorem ceil_sqrt_225_eq_15 : 
  ⌈Real.sqrt 225⌉ = 15 := 
by sorry

end ceil_sqrt_225_eq_15_l29_29023


namespace arctan_sum_eq_half_pi_l29_29687

theorem arctan_sum_eq_half_pi (y : ℚ) :
  2 * Real.arctan (1 / 3) + Real.arctan (1 / 10) + Real.arctan (1 / 30) + Real.arctan (1 / y) = Real.pi / 2 →
  y = 547 / 620 := by
  sorry

end arctan_sum_eq_half_pi_l29_29687


namespace caps_production_l29_29252

def caps1 : Int := 320
def caps2 : Int := 400
def caps3 : Int := 300

def avg_caps (caps1 caps2 caps3 : Int) : Int := (caps1 + caps2 + caps3) / 3

noncomputable def total_caps_after_four_weeks : Int :=
  caps1 + caps2 + caps3 + avg_caps caps1 caps2 caps3

theorem caps_production : total_caps_after_four_weeks = 1360 :=
by
  sorry

end caps_production_l29_29252


namespace totalCostOfCombinedSubscriptions_l29_29383

-- Define the given conditions
def packageACostPerMonth : ℝ := 10
def packageAMonths : ℝ := 6
def packageADiscount : ℝ := 0.10

def packageBCostPerMonth : ℝ := 12
def packageBMonths : ℝ := 9
def packageBDiscount : ℝ := 0.15

-- Define the total cost after discounts
def packageACostAfterDiscount : ℝ := packageACostPerMonth * packageAMonths * (1 - packageADiscount)
def packageBCostAfterDiscount : ℝ := packageBCostPerMonth * packageBMonths * (1 - packageBDiscount)

-- Statement to be proved
theorem totalCostOfCombinedSubscriptions :
  packageACostAfterDiscount + packageBCostAfterDiscount = 145.80 := by
  sorry

end totalCostOfCombinedSubscriptions_l29_29383


namespace perpendicular_lines_have_given_slope_l29_29064

theorem perpendicular_lines_have_given_slope (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 3 = 0 → x + (2 * k + 3) * y - 2 = 0) →
  k = -3 :=
by
  sorry

end perpendicular_lines_have_given_slope_l29_29064


namespace find_arith_seq_sum_l29_29961

noncomputable def arith_seq_sum : ℕ → ℕ → ℕ
| 0, d => 2
| (n+1), d => arith_seq_sum n d + d

theorem find_arith_seq_sum :
  ∃ d : ℕ, 
    arith_seq_sum 1 d + arith_seq_sum 2 d = 13 ∧
    arith_seq_sum 3 d + arith_seq_sum 4 d + arith_seq_sum 5 d = 42 :=
by
  sorry

end find_arith_seq_sum_l29_29961


namespace proof_x_bounds_l29_29745

noncomputable def x : ℝ :=
  1 / Real.logb (1 / 3) (1 / 2) +
  1 / Real.logb (1 / 3) (1 / 4) +
  1 / Real.logb 7 (1 / 8)

theorem proof_x_bounds : 3 < x ∧ x < 3.5 := 
by
  sorry

end proof_x_bounds_l29_29745


namespace median_perimeter_ratio_l29_29045

variables {A B C : Type*}
variables (AB BC AC AD BE CF : ℝ)
variable (l m : ℝ)

noncomputable def triangle_perimeter (AB BC AC : ℝ) : ℝ := AB + BC + AC
noncomputable def triangle_median_sum (AD BE CF : ℝ) : ℝ := AD + BE + CF

theorem median_perimeter_ratio (h1 : l = triangle_perimeter AB BC AC)
                                (h2 : m = triangle_median_sum AD BE CF) :
  m / l > 3 / 4 :=
by
  sorry

end median_perimeter_ratio_l29_29045


namespace infinite_n_square_plus_one_divides_factorial_infinite_n_square_plus_one_not_divide_factorial_l29_29759

theorem infinite_n_square_plus_one_divides_factorial :
  ∃ (infinitely_many n : ℕ), (n^2 + 1) ∣ (n!) := sorry

theorem infinite_n_square_plus_one_not_divide_factorial :
  ∃ (infinitely_many n : ℕ), ¬((n^2 + 1) ∣ (n!)) := sorry

end infinite_n_square_plus_one_divides_factorial_infinite_n_square_plus_one_not_divide_factorial_l29_29759


namespace range_of_independent_variable_l29_29111

theorem range_of_independent_variable (x : ℝ) : 
  (∃ y, y = x / (Real.sqrt (x + 4)) + 1 / (x - 1)) ↔ x > -4 ∧ x ≠ 1 := 
by
  sorry

end range_of_independent_variable_l29_29111


namespace range_estimate_of_expression_l29_29914

theorem range_estimate_of_expression : 
  6 < (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 ∧ 
       (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 < 7 :=
by
  sorry

end range_estimate_of_expression_l29_29914


namespace smallest_digit_not_found_in_units_place_of_odd_number_l29_29849

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l29_29849


namespace compare_negatives_l29_29396

theorem compare_negatives : -0.5 > -0.7 := 
by 
  exact sorry 

end compare_negatives_l29_29396


namespace percentage_increase_l29_29656

variable (presentIncome : ℝ) (newIncome : ℝ)

theorem percentage_increase (h1 : presentIncome = 12000) (h2 : newIncome = 12240) :
  ((newIncome - presentIncome) / presentIncome) * 100 = 2 := by
  sorry

end percentage_increase_l29_29656


namespace complex_modulus_l29_29550

theorem complex_modulus :
  abs ((7 - 4*complex.I) * (3 + 11*complex.I)) = Real.sqrt 8450 :=
by
  sorry

end complex_modulus_l29_29550


namespace smallest_digit_not_found_in_units_place_of_odd_number_l29_29847

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l29_29847


namespace range_of_a_l29_29569

variables (a x : ℝ) -- Define real number variables a and x

-- Define proposition p
def p : Prop := (a - 2) * x * x + 2 * (a - 2) * x - 4 < 0 -- Inequality condition for any real x

-- Define proposition q
def q : Prop := 0 < a ∧ a < 1 -- Condition for logarithmic function to be strictly decreasing

-- Lean 4 statement for the proof problem
theorem range_of_a (Hpq : (p a x ∨ q a) ∧ ¬ (p a x ∧ q a)) :
  (1 ≤ a ∧ a ≤ 2) ∨ (-2 < a ∧ a ≤ 0) :=
sorry

end range_of_a_l29_29569


namespace find_length_BD_l29_29089

theorem find_length_BD (c : ℝ) (h : c ≥ Real.sqrt 7) :
  ∃BD, BD = Real.sqrt (c^2 - 7) :=
sorry

end find_length_BD_l29_29089


namespace sum_odd_terms_a_18_l29_29703

noncomputable def a : ℕ → ℤ
| 0     := -7
| 1     := 5
| (n+2) := a n + 2

theorem sum_odd_terms_a_18 : 
  (∑ i in finset.range (9), a (2 * i)) = 114 := by
sorry

end sum_odd_terms_a_18_l29_29703


namespace number_of_students_who_bought_2_pencils_l29_29451

variable (a b c : ℕ)     -- a is the number of students buying 1 pencil, b is the number of students buying 2 pencils, c is the number of students buying 3 pencils.
variable (total_students total_pencils : ℕ) -- total_students is 36, total_pencils is 50
variable (students_condition1 students_condition2 : ℕ) -- conditions: students_condition1 for the sum of the students, students_condition2 for the sum of the pencils

theorem number_of_students_who_bought_2_pencils :
  total_students = 36 ∧
  total_pencils = 50 ∧
  total_students = a + b + c ∧
  total_pencils = a * 1 + b * 2 + c * 3 ∧
  a = 2 * (b + c) → 
  b = 10 :=
by sorry

end number_of_students_who_bought_2_pencils_l29_29451


namespace sum_of_solutions_l29_29415

def f (x : ℝ) : ℝ := 2^(|x|) + 4 * |x|

theorem sum_of_solutions : (∃ x₁ x₂ : ℝ, f x₁ = 20 ∧ f x₂ = 20 ∧ x₁ + x₂ = 0 ∧ x₁ ≠ x₂) →
  (∑ x in {solution : ℝ | f solution = 20}.to_finset, x = 0) :=
by
  sorry

end sum_of_solutions_l29_29415


namespace average_marks_110_l29_29954

def marks_problem (P C M B E : ℕ) : Prop :=
  (C = P + 90) ∧
  (M = P + 140) ∧
  (P + C + M + B + E = P + 350) ∧
  (B = E) ∧
  (P ≥ 40) ∧
  (C ≥ 40) ∧
  (M ≥ 40) ∧
  (B ≥ 40) ∧
  (E ≥ 40)

theorem average_marks_110 (P C M B E : ℕ) (h : marks_problem P C M B E) : 
    (B + C + M) / 3 = 110 := 
by
  sorry

end average_marks_110_l29_29954


namespace incorrect_statement_is_B_l29_29882

-- Define the conditions
def genotype_AaBb_meiosis_results (sperm_genotypes : List String) : Prop :=
  sperm_genotypes = ["AB", "Ab", "aB", "ab"]

def spermatogonial_cell_AaXbY (malformed_sperm_genotype : String) (other_sperm_genotypes : List String) : Prop :=
  malformed_sperm_genotype = "AAaY" ∧ other_sperm_genotypes = ["aY", "X^b", "X^b"]

def spermatogonial_secondary_spermatocyte_Y_chromosomes (contains_two_Y : Bool) : Prop :=
  ¬ contains_two_Y

def female_animal_meiosis (primary_oocyte_alleles : Nat) (max_oocyte_b_alleles : Nat) : Prop :=
  primary_oocyte_alleles = 10 ∧ max_oocyte_b_alleles ≤ 5

-- The main statement that needs to be proved
theorem incorrect_statement_is_B :
  ∃ (sperm_genotypes : List String) 
    (malformed_sperm_genotype : String) 
    (other_sperm_genotypes : List String) 
    (contains_two_Y : Bool) 
    (primary_oocyte_alleles max_oocyte_b_alleles : Nat),
    genotype_AaBb_meiosis_results sperm_genotypes ∧ 
    spermatogonial_cell_AaXbY malformed_sperm_genotype other_sperm_genotypes ∧ 
    spermatogonial_secondary_spermatocyte_Y_chromosomes contains_two_Y ∧ 
    female_animal_meiosis primary_oocyte_alleles max_oocyte_b_alleles 
    ∧ (malformed_sperm_genotype = "AAaY" → false) := 
sorry

end incorrect_statement_is_B_l29_29882


namespace equilateral_triangle_area_l29_29765

theorem equilateral_triangle_area (h : Real) (h_eq : h = Real.sqrt 12):
  ∃ A : Real, A = 12 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l29_29765


namespace number_of_friends_l29_29902

def initial_candies : ℕ := 10
def additional_candies : ℕ := 4
def total_candies : ℕ := initial_candies + additional_candies
def candies_per_friend : ℕ := 2

theorem number_of_friends : total_candies / candies_per_friend = 7 :=
by
  sorry

end number_of_friends_l29_29902


namespace smallest_digit_never_in_units_place_of_odd_l29_29831

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l29_29831


namespace average_monthly_growth_rate_l29_29278

variable (x : ℝ)

-- Conditions
def turnover_January : ℝ := 36
def turnover_March : ℝ := 48

-- Theorem statement that corresponds to the problem's conditions and question
theorem average_monthly_growth_rate :
  turnover_January * (1 + x)^2 = turnover_March :=
sorry

end average_monthly_growth_rate_l29_29278


namespace count_of_green_hats_l29_29239

-- Defining the total number of hats
def total_hats : ℕ := 85

-- Defining the costs of each hat type
def blue_cost : ℕ := 6
def green_cost : ℕ := 7
def red_cost : ℕ := 8

-- Defining the total cost
def total_cost : ℕ := 600

-- Defining the ratio as 3:2:1
def ratio_blue : ℕ := 3
def ratio_green : ℕ := 2
def ratio_red : ℕ := 1

-- Defining the multiplication factor
def x : ℕ := 14

-- Number of green hats based on the ratio
def G : ℕ := ratio_green * x

-- Proving that we bought 28 green hats
theorem count_of_green_hats : G = 28 := by
  -- proof steps intention: sorry to skip the proof
  sorry

end count_of_green_hats_l29_29239


namespace find_g_of_3_l29_29104

noncomputable def g (x : ℝ) : ℝ := sorry  -- Placeholder for the function g

theorem find_g_of_3 (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x) :
  g 3 = 26 / 7 :=
by sorry

end find_g_of_3_l29_29104


namespace john_work_more_days_l29_29464

theorem john_work_more_days (days_worked : ℕ) (amount_made : ℕ) (daily_earnings : ℕ) (h1 : days_worked = 10) (h2 : amount_made = 250) (h3 : daily_earnings = amount_made / days_worked) : 
  ∃ more_days : ℕ, more_days = (2 * amount_made / daily_earnings) - days_worked := 
by
  have h4 : daily_earnings = 25 := by {
    rw [h1, h2],
    norm_num,
  }
  have h5 : 2 * amount_made / daily_earnings = 20 := by {
    rw [h2, h4],
    norm_num,
  }
  use 10
  rw [h1, h5]
  norm_num

end john_work_more_days_l29_29464


namespace total_money_received_a_l29_29129

-- Define the partners and their capitals
structure Partner :=
  (name : String)
  (capital : ℕ)
  (isWorking : Bool)

def a : Partner := { name := "a", capital := 3500, isWorking := true }
def b : Partner := { name := "b", capital := 2500, isWorking := false }

-- Define the total profit
def totalProfit : ℕ := 9600

-- Define the managing fee as 10% of total profit
def managingFee (total : ℕ) : ℕ := (10 * total) / 100

-- Define the remaining profit after deducting the managing fee
def remainingProfit (total : ℕ) (fee : ℕ) : ℕ := total - fee

-- Calculate the share of remaining profit based on capital contribution
def share (capital totalCapital remaining : ℕ) : ℕ := (capital * remaining) / totalCapital

-- Theorem to prove the total money received by partner a
theorem total_money_received_a :
  let totalCapitals := a.capital + b.capital
  let fee := managingFee totalProfit
  let remaining := remainingProfit totalProfit fee
  let aShare := share a.capital totalCapitals remaining
  (fee + aShare) = 6000 :=
by
  sorry

end total_money_received_a_l29_29129


namespace xy_identity_l29_29714

theorem xy_identity (x y : ℝ) (h1 : (x + y)^2 = 16) (h2 : x * y = 6) : x^2 + y^2 = 4 := 
by 
  sorry

end xy_identity_l29_29714


namespace son_age_is_15_l29_29308

theorem son_age_is_15 (S F : ℕ) (h1 : 2 * S + F = 70) (h2 : 2 * F + S = 95) (h3 : F = 40) :
  S = 15 :=
by {
  sorry
}

end son_age_is_15_l29_29308


namespace range_of_ab_l29_29468

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |2 - a^2| = |2 - b^2|) : 0 < a * b ∧ a * b < 2 := by
  sorry

end range_of_ab_l29_29468


namespace smallest_digit_not_in_units_place_of_odd_numbers_l29_29817

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l29_29817


namespace smallest_digit_never_in_units_place_of_odd_l29_29829

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l29_29829


namespace number_of_pen_refills_l29_29498

-- Conditions
variable (k : ℕ) (x : ℕ) (hk : k > 0) (hx : (4 + k) * x = 6)

-- Question and conclusion as a theorem statement
theorem number_of_pen_refills (hk : k > 0) (hx : (4 + k) * x = 6) : 2 * x = 2 :=
sorry

end number_of_pen_refills_l29_29498


namespace monotonically_increasing_implies_non_negative_derivative_non_negative_derivative_not_implies_monotonically_increasing_l29_29912

variables {f : ℝ → ℝ}

-- Definition that f is monotonically increasing
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≤ f x2

-- Definition of the derivative being non-negative everywhere
def non_negative_derivative (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ (deriv f) x

theorem monotonically_increasing_implies_non_negative_derivative (f : ℝ → ℝ) :
  monotonically_increasing f → non_negative_derivative f :=
sorry

theorem non_negative_derivative_not_implies_monotonically_increasing (f : ℝ → ℝ) :
  non_negative_derivative f → ¬ monotonically_increasing f :=
sorry

end monotonically_increasing_implies_non_negative_derivative_non_negative_derivative_not_implies_monotonically_increasing_l29_29912


namespace binary_to_octal_conversion_l29_29679

-- Define the binary number 11010 in binary
def bin_value : ℕ := 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0

-- Define the octal value 32 in octal as decimal
def oct_value : ℕ := 3 * 8^1 + 2 * 8^0

-- The theorem to prove the binary equivalent of 11010 is the octal 32
theorem binary_to_octal_conversion : bin_value = oct_value :=
by
  -- Skip actual proof
  sorry

end binary_to_octal_conversion_l29_29679


namespace parallel_lines_sufficient_not_necessary_condition_l29_29566

theorem parallel_lines_sufficient_not_necessary_condition {a : ℝ} :
  (a = 4) → (∀ x y : ℝ, (a * x + 8 * y - 3 = 0) ↔ (2 * x + a * y - a = 0)) :=
by sorry

end parallel_lines_sufficient_not_necessary_condition_l29_29566


namespace primes_less_than_200_with_ones_digit_3_l29_29181

theorem primes_less_than_200_with_ones_digit_3 : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, Prime n ∧ n < 200 ∧ n % 10 = 3) ∧ S.card = 12 := 
by
  sorry

end primes_less_than_200_with_ones_digit_3_l29_29181


namespace smallest_real_solution_l29_29484

theorem smallest_real_solution (x : ℝ) : 
  (x * |x| = 3 * x + 4) → x = 4 :=
by {
  sorry -- Proof omitted as per the instructions
}

end smallest_real_solution_l29_29484


namespace meaningful_range_l29_29448

   noncomputable def isMeaningful (x : ℝ) : Prop :=
     (3 - x ≥ 0) ∧ (x + 1 ≠ 0)

   theorem meaningful_range :
     ∀ x : ℝ, isMeaningful x ↔ (x ≤ 3 ∧ x ≠ -1) :=
   by
     sorry
   
end meaningful_range_l29_29448


namespace sum_of_common_divisors_is_10_l29_29495

-- Define the list of numbers
def numbers : List ℤ := [42, 84, -14, 126, 210]

-- Define the common divisors
def common_divisors : List ℕ := [1, 2, 7]

-- Define the function that checks if a number is a common divisor of all numbers in the list
def is_common_divisor (d : ℕ) : Prop :=
  ∀ n ∈ numbers, (d : ℤ) ∣ n

-- Specify the sum of the common divisors
def sum_common_divisors : ℕ := common_divisors.sum

-- State the theorem to be proved
theorem sum_of_common_divisors_is_10 : 
  (∀ d ∈ common_divisors, is_common_divisor d) → 
  sum_common_divisors = 10 := 
by
  sorry

end sum_of_common_divisors_is_10_l29_29495


namespace smallest_not_odd_unit_is_zero_l29_29802

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l29_29802


namespace ceil_sqrt_225_eq_15_l29_29022

theorem ceil_sqrt_225_eq_15 : 
  ⌈Real.sqrt 225⌉ = 15 := 
by sorry

end ceil_sqrt_225_eq_15_l29_29022


namespace chord_length_l29_29290

noncomputable theory

variables (x1 x2 y1 y2 : ℝ)

def is_on_parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

def chord_through_focus (x1 x2 : ℝ) : Prop :=
  x1 + x2 = 6

theorem chord_length (x1 x2 : ℝ)
  (hx1 : is_on_parabola x1 y1)
  (hx2 : is_on_parabola x2 y2)
  (hchord : chord_through_focus x1 x2) :
  |x1 - x2 + sqrt (1 / 4 * (4 * x2 + y2^2))| = 8 :=
sorry

end chord_length_l29_29290


namespace parabola_equation_focus_l29_29357

theorem parabola_equation_focus (p : ℝ) (h₀ : p > 0)
  (h₁ : (p / 2 = 2)) : (y^2 = 2 * p * x) :=
  sorry

end parabola_equation_focus_l29_29357


namespace divisibility_l29_29977

theorem divisibility (a : ℤ) : (5 ∣ a^3) ↔ (5 ∣ a) := 
by sorry

end divisibility_l29_29977


namespace front_view_correct_l29_29424

section stack_problem

def column1 : List ℕ := [3, 2]
def column2 : List ℕ := [1, 4, 2]
def column3 : List ℕ := [5]
def column4 : List ℕ := [2, 1]

def tallest (l : List ℕ) : ℕ := l.foldr max 0

theorem front_view_correct :
  [tallest column1, tallest column2, tallest column3, tallest column4] = [3, 4, 5, 2] :=
sorry

end stack_problem

end front_view_correct_l29_29424


namespace min_value_expression_l29_29607

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 64 :=
by
  sorry

end min_value_expression_l29_29607


namespace quadratic_function_increases_l29_29301

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := 2 * x ^ 2 - 4 * x + 5

-- Prove that for x > 1, the function value y increases as x increases
theorem quadratic_function_increases (x : ℝ) (h : x > 1) : 
  quadratic_function x > quadratic_function 1 :=
sorry

end quadratic_function_increases_l29_29301


namespace solve_arctan_eq_pi_over_3_l29_29483

open Real

theorem solve_arctan_eq_pi_over_3 (x : ℝ) :
  arctan (1 / x) + arctan (1 / x^2) = π / 3 ↔ 
  x = (1 + sqrt (13 + 4 * sqrt 3)) / (2 * sqrt 3) ∨
  x = (1 - sqrt (13 + 4 * sqrt 3)) / (2 * sqrt 3) :=
by
  sorry

end solve_arctan_eq_pi_over_3_l29_29483


namespace resistance_at_least_2000_l29_29174

variable (U : ℝ) (I : ℝ) (R : ℝ)

-- Given conditions:
def voltage := U = 220
def max_current := I ≤ 0.11

-- Ohm's law in this context
def ohms_law := I = U / R

-- Proof problem statement:
theorem resistance_at_least_2000 (voltage : U = 220) (max_current : I ≤ 0.11) (ohms_law : I = U / R) : R ≥ 2000 :=
sorry

end resistance_at_least_2000_l29_29174


namespace concrete_volume_is_six_l29_29533

def to_yards (feet : ℕ) (inches : ℕ) : ℚ :=
  feet * (1 / 3) + inches * (1 / 36)

def sidewalk_volume (width_feet : ℕ) (length_feet : ℕ) (thickness_inches : ℕ) : ℚ :=
  to_yards width_feet 0 * to_yards length_feet 0 * to_yards 0 thickness_inches

def border_volume (border_width_feet : ℕ) (border_thickness_inches : ℕ) (sidewalk_length_feet : ℕ) : ℚ :=
  to_yards (2 * border_width_feet) 0 * to_yards sidewalk_length_feet 0 * to_yards 0 border_thickness_inches

def total_concrete_volume (sidewalk_width_feet : ℕ) (sidewalk_length_feet : ℕ) (sidewalk_thickness_inches : ℕ)
  (border_width_feet : ℕ) (border_thickness_inches : ℕ) : ℚ :=
  sidewalk_volume sidewalk_width_feet sidewalk_length_feet sidewalk_thickness_inches +
  border_volume border_width_feet border_thickness_inches sidewalk_length_feet

def volume_in_cubic_yards (w1_feet : ℕ) (l1_feet : ℕ) (t1_inches : ℕ) (w2_feet : ℕ) (t2_inches : ℕ) : ℚ :=
  total_concrete_volume w1_feet l1_feet t1_inches w2_feet t2_inches

theorem concrete_volume_is_six :
  -- conditions
  volume_in_cubic_yards 4 80 4 1 2 = 6 :=
by
  -- Proof omitted
  sorry

end concrete_volume_is_six_l29_29533


namespace T_30_is_13515_l29_29541

def sequence_first_element (n : ℕ) : ℕ := 1 + n * (n - 1) / 2

def sequence_last_element (n : ℕ) : ℕ := sequence_first_element n + n - 1

def sum_sequence_set (n : ℕ) : ℕ :=
  n * (sequence_first_element n + sequence_last_element n) / 2

theorem T_30_is_13515 : sum_sequence_set 30 = 13515 := by
  sorry

end T_30_is_13515_l29_29541


namespace no_positive_integer_solutions_l29_29646

theorem no_positive_integer_solutions (x y : ℕ) (hx : x > 0) (hy : y > 0) : x^5 ≠ y^2 + 4 := 
by sorry

end no_positive_integer_solutions_l29_29646


namespace degree_g_of_degree_f_and_h_l29_29340

noncomputable def degree (p : ℕ) := p -- definition to represent degree of polynomials

theorem degree_g_of_degree_f_and_h (f g : ℕ → ℕ) (h : ℕ → ℕ) 
  (deg_h : ℕ) (deg_f : ℕ) (deg_10 : deg_h = 10) (deg_3 : deg_f = 3) 
  (h_eq : ∀ x, degree (h x) = degree (f (g x)) + degree x ^ 5) :
  degree (g 0) = 4 :=
by
  sorry

end degree_g_of_degree_f_and_h_l29_29340


namespace range_of_a_for_monotonicity_l29_29176

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a_for_monotonicity (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ 2 < a ∧ a ≤ 3 :=
by sorry

end range_of_a_for_monotonicity_l29_29176


namespace min_value_of_reciprocals_l29_29323

theorem min_value_of_reciprocals (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 2) : 
  ∃ (x : ℝ), x = 2 * a + b ∧ ∃ (y : ℝ), y = 2 * b + c ∧ ∃ (z : ℝ), z = 2 * c + a ∧ (1 / x + 1 / y + 1 / z = 27 / 8) :=
sorry

end min_value_of_reciprocals_l29_29323


namespace total_people_on_bus_l29_29333

def students_left := 42
def students_right := 38
def students_back := 5
def students_aisle := 15
def teachers := 2
def bus_driver := 1

theorem total_people_on_bus : students_left + students_right + students_back + students_aisle + teachers + bus_driver = 103 :=
by
  sorry

end total_people_on_bus_l29_29333


namespace robie_initial_cards_l29_29479

-- Definitions of the problem conditions
def each_box_cards : ℕ := 25
def extra_cards : ℕ := 11
def given_away_boxes : ℕ := 6
def remaining_boxes : ℕ := 12

-- The final theorem we need to prove
theorem robie_initial_cards : 
  (given_away_boxes + remaining_boxes) * each_box_cards + extra_cards = 461 :=
by
  sorry

end robie_initial_cards_l29_29479


namespace angle_bisector_square_l29_29754

theorem angle_bisector_square (A B C D : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (dist : A → B → ℝ)
  (α β γ δ : A)
  (h1 : ∀ x y z : A, dist x y = dist y z → dist x z = dist x y)
  (htriangle : Triangle α β γ)
  (Dpoint : Point γ)
  (bisector_theorem : dist α δ / dist δ β = dist β γ / dist γ α):
  dist α δ ^ 2 = dist β γ * dist γ α - dist α δ * dist δ β := 
sorry

end angle_bisector_square_l29_29754


namespace max_value_of_symmetric_function_l29_29724

def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function 
  (a b : ℝ)
  (symmetric : ∀ t : ℝ, f (-2 + t) a b = f (-2 - t) a b) :
  ∃ M : ℝ, M = 16 ∧ ∀ x : ℝ, f x a b ≤ M :=
by
  use 16
  sorry

end max_value_of_symmetric_function_l29_29724


namespace range_contains_pi_div_4_l29_29029

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_contains_pi_div_4 : ∃ x : ℝ, f x = (Real.pi / 4) := by
  sorry

end range_contains_pi_div_4_l29_29029


namespace smallest_n_perfect_square_and_cube_l29_29872

theorem smallest_n_perfect_square_and_cube (n : ℕ) (h1 : ∃ k : ℕ, 5 * n = k^2) (h2 : ∃ m : ℕ, 4 * n = m^3) :
  n = 1080 :=
  sorry

end smallest_n_perfect_square_and_cube_l29_29872


namespace debate_organizing_committees_count_l29_29313

theorem debate_organizing_committees_count :
    ∃ (n : ℕ), n = 5 * (Nat.choose 8 4) * (Nat.choose 8 3)^4 ∧ n = 3442073600 :=
by
  sorry

end debate_organizing_committees_count_l29_29313


namespace claire_gift_card_balance_l29_29672

/--
Claire has a $100 gift card to her favorite coffee shop.
A latte costs $3.75.
A croissant costs $3.50.
Claire buys one latte and one croissant every day for a week.
Claire buys 5 cookies, each costing $1.25.

Prove that the amount of money left on Claire's gift card after a week is $43.00.
-/
theorem claire_gift_card_balance :
  let initial_balance : ℝ := 100
  let latte_cost : ℝ := 3.75
  let croissant_cost : ℝ := 3.50
  let daily_expense : ℝ := latte_cost + croissant_cost
  let weekly_expense : ℝ := daily_expense * 7
  let cookie_cost : ℝ := 1.25
  let total_cookie_expense : ℝ := cookie_cost * 5
  let total_expense : ℝ := weekly_expense + total_cookie_expense
  let remaining_balance : ℝ := initial_balance - total_expense
  remaining_balance = 43 :=
by
  sorry

end claire_gift_card_balance_l29_29672


namespace intersection_M_N_eq_segment_l29_29427

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N_eq_segment : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_eq_segment_l29_29427


namespace ceil_sqrt_225_l29_29017

theorem ceil_sqrt_225 : ⌈real.sqrt 225⌉ = 15 :=
by
  have h : real.sqrt 225 = 15 := by
    sorry
  rw [h]
  exact int.ceil_eq_self.mpr rfl

end ceil_sqrt_225_l29_29017


namespace find_f_2011_l29_29169

-- Definitions of given conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic_of_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- Main theorem to be proven
theorem find_f_2011 (f : ℝ → ℝ) 
  (hf_even: is_even_function f) 
  (hf_periodic: is_periodic_of_period f 4) 
  (hf_at_1: f 1 = 1) : 
  f 2011 = 1 := 
by 
  sorry

end find_f_2011_l29_29169


namespace marble_probability_l29_29497

theorem marble_probability (W G R B : ℕ) (h_total : W + G + R + B = 84) 
  (h_white : W / 84 = 1 / 4) (h_green : G / 84 = 1 / 7) :
  (R + B) / 84 = 17 / 28 :=
by
  sorry

end marble_probability_l29_29497


namespace product_of_p_r_s_l29_29715

-- Definition of conditions
def eq1 (p : ℕ) : Prop := 4^p + 4^3 = 320
def eq2 (r : ℕ) : Prop := 3^r + 27 = 108
def eq3 (s : ℕ) : Prop := 2^s + 7^4 = 2617

-- Main statement
theorem product_of_p_r_s (p r s : ℕ) (h1 : eq1 p) (h2 : eq2 r) (h3 : eq3 s) : p * r * s = 112 :=
by sorry

end product_of_p_r_s_l29_29715


namespace sum_of_interior_numbers_eighth_row_l29_29197

def sum_of_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2

theorem sum_of_interior_numbers_eighth_row : sum_of_interior_numbers 8 = 126 :=
by
  sorry

end sum_of_interior_numbers_eighth_row_l29_29197


namespace stratified_sampling_size_l29_29627

theorem stratified_sampling_size (a_ratio b_ratio c_ratio : ℕ) (total_items_A : ℕ) (h_ratio : a_ratio + b_ratio + c_ratio = 10)
  (h_A_ratio : a_ratio = 2) (h_B_ratio : b_ratio = 3) (h_C_ratio : c_ratio = 5) (items_A : total_items_A = 20) : 
  ∃ n : ℕ, n = total_items_A * 5 := 
by {
  -- The proof should go here. Since we only need the statement:
  sorry
}

end stratified_sampling_size_l29_29627


namespace great_grandson_age_is_36_l29_29888

-- Define the problem conditions and the required proof
theorem great_grandson_age_is_36 :
  ∃ n : ℕ, (∃ k : ℕ, k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧  2 * k * 111 = n * (n + 1)) ∧ n = 36 :=
by
  sorry

end great_grandson_age_is_36_l29_29888


namespace solve_equation_l29_29748

open Real

/-- Define the original equation as a function. -/
def equation (x : ℝ) : ℝ := 2 ^ (3 ^ (4 ^ x)) - 4 ^ (3 ^ (2 ^ x))

/-- Define the specific value of x that solves the equation. -/
noncomputable def x_solution : ℝ := log 1.4386 / log 2

/-- Statement of the proof problem verifying the solution. -/
theorem solve_equation : equation x_solution = 0 :=
by
  -- Proof steps will be added here
  sorry

end solve_equation_l29_29748


namespace triangle_inequality_third_side_l29_29733

theorem triangle_inequality_third_side (a b x : ℝ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : 0 < x) (h₄ : x < a + b) (h₅ : a < b + x) (h₆ : b < a + x) :
  ¬(x = 9) := by
  sorry

end triangle_inequality_third_side_l29_29733


namespace cos_30_eq_sqrt3_div_2_l29_29120

theorem cos_30_eq_sqrt3_div_2 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l29_29120


namespace root_zero_implies_m_eq_6_l29_29188

theorem root_zero_implies_m_eq_6 (m : ℝ) (h : ∃ x : ℝ, 3 * (x^2) + m * x + m - 6 = 0) : m = 6 := 
by sorry

end root_zero_implies_m_eq_6_l29_29188


namespace randi_has_6_more_nickels_than_peter_l29_29756

def ray_initial_cents : Nat := 175
def cents_given_peter : Nat := 30
def cents_given_randi : Nat := 2 * cents_given_peter
def nickel_worth : Nat := 5

def nickels (cents : Nat) : Nat :=
  cents / nickel_worth

def randi_more_nickels_than_peter : Prop :=
  nickels cents_given_randi - nickels cents_given_peter = 6

theorem randi_has_6_more_nickels_than_peter :
  randi_more_nickels_than_peter :=
sorry

end randi_has_6_more_nickels_than_peter_l29_29756


namespace prob_of_interval_l29_29047

open ProbabilityTheory

noncomputable def normal_ξ := ⁇ -- some placeholder for ξ, as Lean doesn't have direct access to named random variables from a distribution

theorem prob_of_interval 
  (σ : ℝ)
  (h₁ : normalDistribution 1 σ)
  (h₂ : P (λ x, x < 2) = 0.6) :
  P (λ x, 0 < x ∧ x < 1) = 0.1 := 
sorry

end prob_of_interval_l29_29047


namespace books_left_after_giveaways_l29_29557

def initial_books : ℝ := 48.0
def first_giveaway : ℝ := 34.0
def second_giveaway : ℝ := 3.0

theorem books_left_after_giveaways : 
  initial_books - first_giveaway - second_giveaway = 11.0 :=
by
  sorry

end books_left_after_giveaways_l29_29557


namespace gifts_needed_l29_29318

def num_teams : ℕ := 7
def num_gifts_per_team : ℕ := 2

theorem gifts_needed (h1 : num_teams = 7) (h2 : num_gifts_per_team = 2) : num_teams * num_gifts_per_team = 14 := 
by
  -- proof skipped
  sorry

end gifts_needed_l29_29318


namespace remaining_paint_fraction_l29_29256

def initial_paint : ℚ := 1

def paint_day_1 : ℚ := initial_paint - (1/2) * initial_paint
def paint_day_2 : ℚ := paint_day_1 - (1/4) * paint_day_1
def paint_day_3 : ℚ := paint_day_2 - (1/3) * paint_day_2

theorem remaining_paint_fraction : paint_day_3 = 1/4 :=
by
  sorry

end remaining_paint_fraction_l29_29256


namespace equation_has_one_real_root_l29_29347

noncomputable def f (x : ℝ) : ℝ :=
  (3 / 11)^x + (5 / 11)^x + (7 / 11)^x - 1

theorem equation_has_one_real_root :
  ∃! x : ℝ, f x = 0 := sorry

end equation_has_one_real_root_l29_29347


namespace real_solutions_equation_l29_29972

theorem real_solutions_equation :
  ∃! x : ℝ, 9 * x^2 - 90 * ⌊ x ⌋ + 99 = 0 :=
sorry

end real_solutions_equation_l29_29972


namespace value_a_squared_plus_b_squared_l29_29948

-- Defining the problem with the given conditions
theorem value_a_squared_plus_b_squared (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 6) : a^2 + b^2 = 21 :=
by
  sorry

end value_a_squared_plus_b_squared_l29_29948


namespace real_solutions_of_equation_l29_29685

theorem real_solutions_of_equation : 
  ∃! x₁ x₂ : ℝ, (3 * x₁^2 - 10 * x₁ + 7 = 0) ∧ (3 * x₂^2 - 10 * x₂ + 7 = 0) ∧ x₁ ≠ x₂ :=
sorry

end real_solutions_of_equation_l29_29685


namespace root_of_quadratic_l29_29294

theorem root_of_quadratic (x m : ℝ) (h : x = -1 ∧ x^2 + m*x - 1 = 0) : m = 0 :=
sorry

end root_of_quadratic_l29_29294


namespace consecutive_differences_equal_l29_29527

-- Define the set and the condition
def S : Set ℕ := {n : ℕ | n > 0}

-- Condition that for any two numbers a and b in S with a > b, at least one of a + b or a - b is also in S
axiom h_condition : ∀ a b : ℕ, a ∈ S → b ∈ S → a > b → (a + b ∈ S ∨ a - b ∈ S)

-- The main theorem that we want to prove
theorem consecutive_differences_equal (a : ℕ) (s : Fin 2003 → ℕ) 
  (hS : ∀ i, s i ∈ S)
  (h_ordered : ∀ i j, i < j → s i < s j) :
  ∃ (d : ℕ), ∀ i, i < 2002 → (s (i + 1)) - (s i) = d :=
sorry

end consecutive_differences_equal_l29_29527


namespace company_hired_22_additional_males_l29_29235

theorem company_hired_22_additional_males
  (E M : ℕ) 
  (initial_percentage_female : ℝ)
  (final_total_employees : ℕ)
  (final_percentage_female : ℝ)
  (initial_female_count : initial_percentage_female * E = 0.6 * E)
  (final_employee_count : E + M = 264) 
  (final_female_count : initial_percentage_female * E = final_percentage_female * (E + M)) :
  M = 22 := 
by
  sorry

end company_hired_22_additional_males_l29_29235


namespace Joe_total_income_l29_29462

theorem Joe_total_income : 
  (∃ I : ℝ, 0.1 * 1000 + 0.2 * 3000 + 0.3 * (I - 500 - 4000) = 848 ∧ I - 500 > 4000) → I = 4993.33 :=
by
  sorry

end Joe_total_income_l29_29462


namespace area_of_square_l29_29895

def side_length (x : ℕ) : ℕ := 3 * x - 12

def side_length_alt (x : ℕ) : ℕ := 18 - 2 * x

theorem area_of_square (x : ℕ) (h : 3 * x - 12 = 18 - 2 * x) : (side_length x) ^ 2 = 36 :=
by
  sorry

end area_of_square_l29_29895


namespace second_storm_duration_l29_29501

theorem second_storm_duration (x y : ℕ) 
  (h1 : x + y = 45) 
  (h2 : 30 * x + 15 * y = 975) : 
  y = 25 :=
by
  sorry

end second_storm_duration_l29_29501


namespace alpha_beta_value_l29_29706

noncomputable def alpha_beta_sum : ℝ := 75

theorem alpha_beta_value (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h : |Real.sin α - (1 / 2)| + Real.sqrt (Real.tan β - 1) = 0) :
  α + β = α_beta_sum := 
  sorry

end alpha_beta_value_l29_29706


namespace great_grandson_age_l29_29887

theorem great_grandson_age (n : ℕ) : 
  ∃ n, (n * (n + 1)) / 2 = 666 :=
by
  -- Solution steps would go here
  sorry

end great_grandson_age_l29_29887


namespace correct_equation_l29_29280

-- Definitions of the conditions
def january_turnover (T : ℝ) : Prop := T = 36
def march_turnover (T : ℝ) : Prop := T = 48
def average_monthly_growth_rate (x : ℝ) : Prop := True

-- The goal to be proved
theorem correct_equation (T_jan T_mar : ℝ) (x : ℝ) 
  (h_jan : january_turnover T_jan) 
  (h_mar : march_turnover T_mar) 
  (h_growth : average_monthly_growth_rate x) : 
  36 * (1 + x)^2 = 48 :=
sorry

end correct_equation_l29_29280


namespace returning_players_l29_29232

-- Definitions of conditions
def num_groups : Nat := 9
def players_per_group : Nat := 6
def new_players : Nat := 48

-- Definition of total number of players
def total_players : Nat := num_groups * players_per_group

-- Theorem: Find the number of returning players
theorem returning_players :
  total_players - new_players = 6 :=
by
  sorry

end returning_players_l29_29232


namespace claire_balance_after_week_l29_29674

theorem claire_balance_after_week :
  ∀ (gift_card : ℝ) (latte_cost croissant_cost : ℝ) (days : ℕ) (cookie_cost : ℝ) (cookies : ℕ),
  gift_card = 100 ∧
  latte_cost = 3.75 ∧
  croissant_cost = 3.50 ∧
  days = 7 ∧
  cookie_cost = 1.25 ∧
  cookies = 5 →
  (gift_card - (days * (latte_cost + croissant_cost) + cookie_cost * cookies) = 43) :=
by
  -- Skipping proof details with sorry
  sorry

end claire_balance_after_week_l29_29674


namespace smaller_sphere_surface_area_proof_l29_29736

-- Definitions based on conditions
noncomputable def cube_edge_length : ℝ := 1
noncomputable def larger_sphere_radius : ℝ := cube_edge_length / 2
noncomputable def diagonal_distance_from_center_to_vertex : ℝ := (↑(Real.sqrt 3)) * cube_edge_length / 2

-- Applying the result from the provided solution
noncomputable def smaller_sphere_radius : ℝ := (2 - Real.sqrt 3) / 2
noncomputable def surface_area_smaller_sphere : ℝ := 4 * Real.pi * smaller_sphere_radius^2

-- Statement to prove
theorem smaller_sphere_surface_area_proof :
  surface_area_smaller_sphere = π * (7 - 4 * Real.sqrt 3) := sorry

end smaller_sphere_surface_area_proof_l29_29736


namespace average_marbles_of_other_colors_l29_29212

theorem average_marbles_of_other_colors
  (clear_percentage : ℝ) (black_percentage : ℝ) (total_marbles_taken : ℕ)
  (h1 : clear_percentage = 0.4) (h2 : black_percentage = 0.2) :
  (total_marbles_taken : ℝ) * (1 - clear_percentage - black_percentage) = 2 :=
by
  sorry

end average_marbles_of_other_colors_l29_29212


namespace product_of_two_integers_l29_29588

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 26) (h2 : x^2 - y^2 = 52) (h3 : x > y) : x * y = 168 := by
  sorry

end product_of_two_integers_l29_29588


namespace average_monthly_growth_rate_l29_29277

variable (x : ℝ)

-- Conditions
def turnover_January : ℝ := 36
def turnover_March : ℝ := 48

-- Theorem statement that corresponds to the problem's conditions and question
theorem average_monthly_growth_rate :
  turnover_January * (1 + x)^2 = turnover_March :=
sorry

end average_monthly_growth_rate_l29_29277


namespace smallest_digit_not_in_units_place_of_odd_l29_29798

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l29_29798


namespace maggie_earnings_proof_l29_29985

def rate_per_subscription : ℕ := 5
def subscriptions_to_parents : ℕ := 4
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_neighbor1 : ℕ := 2
def subscriptions_to_neighbor2 : ℕ := 2 * subscriptions_to_neighbor1
def total_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_neighbor1 + subscriptions_to_neighbor2
def total_earnings : ℕ := total_subscriptions * rate_per_subscription

theorem maggie_earnings_proof : total_earnings = 55 := by
  sorry

end maggie_earnings_proof_l29_29985


namespace circle_equation_through_ABC_circle_equation_with_center_and_points_l29_29521

-- Define points
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 0⟩
def B : Point := ⟨4, 0⟩
def C : Point := ⟨6, -2⟩

-- First problem: proof of the circle equation given points A, B, and C
theorem circle_equation_through_ABC :
  ∃ (D E F : ℝ), 
  (∀ (P : Point), (P = A ∨ P = B ∨ P = C) → P.x^2 + P.y^2 + D * P.x + E * P.y + F = 0) 
  ↔ (D = -5 ∧ E = 7 ∧ F = 4) := sorry

-- Second problem: proof of the circle equation given the y-coordinate of the center and points A and B
theorem circle_equation_with_center_and_points :
  ∃ (h k r : ℝ), 
  (h = (A.x + B.x) / 2 ∧ k = 2) ∧
  ∀ (P : Point), (P = A ∨ P = B) → (P.x - h)^2 + (P.y - k)^2 = r^2
  ↔ (h = 5 / 2 ∧ k = 2 ∧ r = 5 / 2) := sorry

end circle_equation_through_ABC_circle_equation_with_center_and_points_l29_29521


namespace maximal_N8_value_l29_29268

noncomputable def max_permutations_of_projections (A : Fin 8 → ℝ × ℝ) : ℕ := sorry

theorem maximal_N8_value (A : Fin 8 → ℝ × ℝ) :
  max_permutations_of_projections A = 56 :=
sorry

end maximal_N8_value_l29_29268


namespace inequality_proof_l29_29043

variable {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + c^2 = 1) : 
  a + b + Real.sqrt 2 * c ≤ 2 := 
by 
  sorry

end inequality_proof_l29_29043


namespace max_value_of_symmetric_function_l29_29728

def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) (h_sym : ∀ x : ℝ, f x a b = f (-4 - x) a b) : 
  ∃ x : ℝ, (∀ y : ℝ, f y a b ≤ f x a b) ∧ f x a b = 16 :=
by
  sorry

end max_value_of_symmetric_function_l29_29728


namespace lindas_initial_candies_l29_29208

theorem lindas_initial_candies (candies_given : ℝ) (candies_left : ℝ) (initial_candies : ℝ) : 
  candies_given = 28 ∧ candies_left = 6 → initial_candies = candies_given + candies_left → initial_candies = 34 := 
by 
  sorry

end lindas_initial_candies_l29_29208


namespace smallest_b_l29_29299

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x - 2 * a

theorem smallest_b (a : ℝ) (b : ℝ) (x : ℝ) : (1 < a ∧ a < 4) → (0 < x) → (f a b x > 0) → b ≥ 11 :=
by
  -- placeholder for the proof
  sorry

end smallest_b_l29_29299


namespace smallest_digit_not_in_units_place_of_odd_l29_29795

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l29_29795


namespace normal_trip_distance_l29_29077

variable (S D : ℝ)

-- Conditions
axiom h1 : D = 3 * S
axiom h2 : D + 50 = 5 * S

theorem normal_trip_distance : D = 75 :=
by
  sorry

end normal_trip_distance_l29_29077


namespace find_a_for_exactly_two_solutions_l29_29689

theorem find_a_for_exactly_two_solutions :
  ∃ a : ℝ, (∀ x : ℝ, (|x + a| = 1/x) ↔ (a = -2) ∧ (x ≠ 0)) ∧ ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (|x1 + a| = 1/x1) ∧ (|x2 + a| = 1/x2) :=
sorry

end find_a_for_exactly_two_solutions_l29_29689


namespace range_of_a2_div_a1_l29_29773

theorem range_of_a2_div_a1 (a_1 a_2 d : ℤ) : 
  1 ≤ a_1 ∧ a_1 ≤ 3 ∧ 
  a_2 = a_1 + d ∧ 
  6 ≤ 3 * a_1 + 2 * d ∧ 
  3 * a_1 + 2 * d ≤ 15 
  → (2 / 3 : ℚ) ≤ (a_2 : ℚ) / a_1 ∧ (a_2 : ℚ) / a_1 ≤ 5 :=
sorry

end range_of_a2_div_a1_l29_29773


namespace five_color_theorem_l29_29988

-- Define the Five Color Theorem (a general formalization in terms of graph theory)
theorem five_color_theorem (G : SimpleGraph V) [Fintype V] [DecidableRel G.Adj] (G_colorable : ∀ v ∈ G.support, Fintype.card (G.neighbors v) ≤ 5) :
  ∃ (coloring : V → Fin 5), G.IsProperColoring coloring :=
sorry

end five_color_theorem_l29_29988


namespace initial_ratio_of_stamps_l29_29351

theorem initial_ratio_of_stamps (P Q : ℕ) (h1 : ((P - 8 : ℤ) : ℚ) / (Q + 8) = 6 / 5) (h2 : P - 8 = Q + 8) : P / Q = 6 / 5 :=
sorry

end initial_ratio_of_stamps_l29_29351


namespace artist_paints_37_sq_meters_l29_29664

-- Define the structure of the sculpture
def top_layer : ℕ := 1
def middle_layer : ℕ := 5
def bottom_layer : ℕ := 11
def edge_length : ℕ := 1

-- Define the exposed surface areas
def exposed_surface_top_layer := 5 * top_layer
def exposed_surface_middle_layer := 1 * 5 + 4 * 4
def exposed_surface_bottom_layer := bottom_layer

-- Calculate the total exposed surface area
def total_exposed_surface_area := exposed_surface_top_layer + exposed_surface_middle_layer + exposed_surface_bottom_layer

-- The final theorem statement
theorem artist_paints_37_sq_meters (hyp1 : top_layer = 1)
  (hyp2 : middle_layer = 5)
  (hyp3 : bottom_layer = 11)
  (hyp4 : edge_length = 1)
  : total_exposed_surface_area = 37 := 
by
  sorry

end artist_paints_37_sq_meters_l29_29664


namespace rectangle_perimeter_l29_29258

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : (a * b = 2 * (a + b))) : 2 * (a + b) = 36 :=
by sorry

end rectangle_perimeter_l29_29258


namespace cubic_polynomial_solution_l29_29033

noncomputable def q (x : ℝ) : ℝ := - (4 / 3) * x^3 + 6 * x^2 - (50 / 3) * x - (14 / 3)

theorem cubic_polynomial_solution :
  q 1 = -8 ∧ q 2 = -12 ∧ q 3 = -20 ∧ q 4 = -40 := by
  have h₁ : q 1 = -8 := by sorry
  have h₂ : q 2 = -12 := by sorry
  have h₃ : q 3 = -20 := by sorry
  have h₄ : q 4 = -40 := by sorry
  exact ⟨h₁, h₂, h₃, h₄⟩

end cubic_polynomial_solution_l29_29033


namespace luke_total_points_l29_29209

/-- Luke gained 327 points in each round of a trivia game. 
    He played 193 rounds of the game. 
    How many points did he score in total? -/
theorem luke_total_points : 327 * 193 = 63111 :=
by
  sorry

end luke_total_points_l29_29209


namespace find_13x2_22xy_13y2_l29_29168

variable (x y : ℝ)

theorem find_13x2_22xy_13y2 
  (h1 : 3 * x + 2 * y = 8) 
  (h2 : 2 * x + 3 * y = 11) 
: 13 * x^2 + 22 * x * y + 13 * y^2 = 184 := 
sorry

end find_13x2_22xy_13y2_l29_29168


namespace david_trip_distance_l29_29662

theorem david_trip_distance (t : ℝ) (d : ℝ) : 
  (40 * (t + 1) = d) →
  (d - 40 = 60 * (t - 0.75)) →
  d = 130 := 
by
  intro h1 h2
  sorry

end david_trip_distance_l29_29662


namespace problem_statement_l29_29747

noncomputable def curvature (k_A k_B : ℝ) (d : ℝ) : ℝ :=
  |k_A - k_B| / d

def is_constant_curvature (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, curvature (fderiv ℝ f x) (fderiv ℝ f y) (dist x y) = k

theorem problem_statement :
  (curvature (3 : ℝ) (3 : ℝ) (dist 1 (-1)) = 0) ∧
  (∃ f : ℝ → ℝ, is_constant_curvature f) ∧
  (∀ (x_A x_B : ℝ), curvature (2*x_A) (2*x_B) (dist x_A x_B) ≤ 2) ∧
  (∀ (x_1 x_2 : ℝ), curvature (exp x_1) (exp x_2) (dist x_1 x_2) < 1) :=
begin
  sorry,
end

end problem_statement_l29_29747


namespace max_glows_in_time_range_l29_29523

theorem max_glows_in_time_range (start_time end_time : ℤ) (interval : ℤ) (h1 : start_time = 3600 + 3420 + 58) (h2 : end_time = 10800 + 1200 + 47) (h3 : interval = 21) :
  (end_time - start_time) / interval = 236 := 
  sorry

end max_glows_in_time_range_l29_29523


namespace smallest_digit_not_in_units_place_of_odd_numbers_l29_29819

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l29_29819


namespace fiona_working_hours_l29_29952

theorem fiona_working_hours (F : ℕ) 
  (John_hours_per_week : ℕ := 30) 
  (Jeremy_hours_per_week : ℕ := 25) 
  (pay_rate : ℕ := 20) 
  (monthly_total_pay : ℕ := 7600) : 
  4 * (John_hours_per_week * pay_rate + Jeremy_hours_per_week * pay_rate + F * pay_rate) = monthly_total_pay → 
  F = 40 :=
by sorry

end fiona_working_hours_l29_29952


namespace smallest_digit_never_at_units_place_of_odd_l29_29842

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l29_29842


namespace namjoon_used_pencils_l29_29099

variable (taehyungUsed : ℕ) (namjoonUsed : ℕ)

/-- 
Statement:
Taehyung and Namjoon each initially have 10 pencils.
Taehyung gives 3 of his remaining pencils to Namjoon.
After this, Taehyung ends up with 6 pencils and Namjoon ends up with 6 pencils.
We need to prove that Namjoon used 7 pencils.
-/
theorem namjoon_used_pencils (H1 : 10 - taehyungUsed = 9 - 3)
  (H2 : 13 - namjoonUsed = 6) : namjoonUsed = 7 :=
sorry

end namjoon_used_pencils_l29_29099


namespace inequality_true_l29_29942

variable (a b : ℝ)

theorem inequality_true (h : a > b ∧ b > 0) : (b^2 / a) < (a^2 / b) := by
  sorry

end inequality_true_l29_29942


namespace eddy_travel_time_l29_29012

theorem eddy_travel_time (T : ℝ) (S_e S_f : ℝ) (Freddy_time : ℝ := 4)
  (distance_AB : ℝ := 540) (distance_AC : ℝ := 300) (speed_ratio : ℝ := 2.4) :
  (distance_AB / T = 2.4 * (distance_AC / Freddy_time)) -> T = 3 :=
by
  sorry

end eddy_travel_time_l29_29012


namespace find_z_l29_29589

theorem find_z 
  {x y z : ℕ}
  (hx : x = 4)
  (hy : y = 7)
  (h_least : x - y - z = 17) : 
  z = 14 :=
by
  sorry

end find_z_l29_29589


namespace total_earnings_correct_l29_29517

section
  -- Define the conditions
  def wage : ℕ := 8
  def hours_Monday : ℕ := 8
  def hours_Tuesday : ℕ := 2

  -- Define the calculation for the total earnings
  def earnings (hours : ℕ) (rate : ℕ) : ℕ := hours * rate

  -- State the total earnings
  def total_earnings : ℕ := earnings hours_Monday wage + earnings hours_Tuesday wage

  -- Theorem: Prove that Will's total earnings in those two days is $80
  theorem total_earnings_correct : total_earnings = 80 := by
    sorry
end

end total_earnings_correct_l29_29517


namespace tiles_walked_on_l29_29890

/-- 
A park has a rectangular shape with a width of 13 feet and a length of 19 feet.
Square-shaped tiles of dimension 1 foot by 1 foot cover the entire area.
The gardener walks in a straight line from one corner of the rectangle to the opposite corner.
One specific tile in the path is not to be stepped on. 
Prove that the number of tiles the gardener walks on is 30.
-/
theorem tiles_walked_on (width length gcd_width_length tiles_to_avoid : ℕ)
  (h_width : width = 13)
  (h_length : length = 19)
  (h_gcd : gcd width length = 1)
  (h_tiles_to_avoid : tiles_to_avoid = 1) : 
  (width + length - gcd_width_length - tiles_to_avoid = 30) := 
by
  sorry

end tiles_walked_on_l29_29890


namespace leah_probability_of_seeing_change_l29_29661

open Set

-- Define the length of each color interval
def green_duration := 45
def yellow_duration := 5
def red_duration := 35

-- Total cycle duration
def total_cycle_duration := green_duration + yellow_duration + red_duration

-- Leah's viewing intervals
def change_intervals : Set (ℕ × ℕ) :=
  {(40, 45), (45, 50), (80, 85)}

-- Probability calculation
def favorable_time := 15
def probability_of_change := (favorable_time : ℚ) / (total_cycle_duration : ℚ)

theorem leah_probability_of_seeing_change : probability_of_change = 3 / 17 :=
by
  -- We use sorry here as we are only required to state the theorem without proof.
  sorry

end leah_probability_of_seeing_change_l29_29661


namespace s_point_value_l29_29602

def has_s_point (f g : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = g x₀ ∧ deriv f x₀ = deriv g x₀

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x, a * x^2 - 1
noncomputable def g (a : ℝ) : ℝ → ℝ := λ x, log (a * x)

theorem s_point_value (a : ℝ) (x₀ : ℝ) (h : has_s_point (f a) (g a) x₀) : a = 2 / real.exp(1) :=
sorry

end s_point_value_l29_29602


namespace total_legs_in_christophers_room_l29_29670

def total_legs (num_spiders num_legs_per_spider num_ants num_butterflies num_beetles num_legs_per_insect : ℕ) : ℕ :=
  let spider_legs := num_spiders * num_legs_per_spider
  let ant_legs := num_ants * num_legs_per_insect
  let butterfly_legs := num_butterflies * num_legs_per_insect
  let beetle_legs := num_beetles * num_legs_per_insect
  spider_legs + ant_legs + butterfly_legs + beetle_legs

theorem total_legs_in_christophers_room : total_legs 12 8 10 5 5 6 = 216 := by
  -- Calculation and reasoning omitted
  sorry

end total_legs_in_christophers_room_l29_29670


namespace candle_cost_correct_l29_29741

-- Variables and conditions
def candles_per_cake : Nat := 8
def num_cakes : Nat := 3
def candles_needed : Nat := candles_per_cake * num_cakes

def candles_per_box : Nat := 12
def boxes_needed : Nat := candles_needed / candles_per_box

def cost_per_box : ℝ := 2.5
def total_cost : ℝ := boxes_needed * cost_per_box

-- Proof statement
theorem candle_cost_correct :
  total_cost = 5 := by
  sorry

end candle_cost_correct_l29_29741


namespace sine_sum_square_greater_l29_29467

variable {α β : Real} (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2)
variable (h : Real.sin α ^ 2 + Real.sin β ^ 2 < 1)

theorem sine_sum_square_greater (α β : Real) (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2) 
  (h : Real.sin α ^ 2 + Real.sin β ^ 2 < 1) : 
  Real.sin (α + β) ^ 2 > Real.sin α ^ 2 + Real.sin β ^ 2 :=
sorry

end sine_sum_square_greater_l29_29467


namespace range_of_b_in_acute_triangle_l29_29452

variable {a b c : ℝ}

theorem range_of_b_in_acute_triangle (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_acute : (a^2 + b^2 > c^2) ∧ (b^2 + c^2 > a^2) ∧ (c^2 + a^2 > b^2))
  (h_arith_seq : ∃ d : ℝ, 0 ≤ d ∧ a = b - d ∧ c = b + d)
  (h_sum_squares : a^2 + b^2 + c^2 = 21) :
  (2 * Real.sqrt 42) / 5 < b ∧ b ≤ Real.sqrt 7 :=
sorry

end range_of_b_in_acute_triangle_l29_29452


namespace find_number_l29_29693

theorem find_number (x : ℝ) (h : x - (3/5) * x = 60) : x = 150 :=
by
  sorry

end find_number_l29_29693


namespace bailing_rate_bailing_problem_l29_29364

theorem bailing_rate (distance : ℝ) (rate_in : ℝ) (sink_limit : ℝ) (speed : ℝ) : ℝ :=
  let time_to_shore := distance / speed * 60 -- convert hours to minutes
  let total_intake := rate_in * time_to_shore
  let excess_water := total_intake - sink_limit
  excess_water / time_to_shore

theorem bailing_problem : bailing_rate 2 12 40 3 = 11 := by
  sorry

end bailing_rate_bailing_problem_l29_29364


namespace max_two_integers_abs_leq_50_l29_29562

theorem max_two_integers_abs_leq_50
  (a b c : ℤ) (h_a : a > 100) :
  ∀ {x1 x2 x3 : ℤ}, (abs (a * x1^2 + b * x1 + c) ≤ 50) →
                    (abs (a * x2^2 + b * x2 + c) ≤ 50) →
                    (abs (a * x3^2 + b * x3 + c) ≤ 50) →
                    false :=
sorry

end max_two_integers_abs_leq_50_l29_29562


namespace find_natural_numbers_l29_29410

noncomputable def valid_n (n : ℕ) : Prop :=
  2 ^ n % 7 = n ^ 2 % 7

theorem find_natural_numbers :
  {n : ℕ | valid_n n} = {n : ℕ | n % 21 = 2 ∨ n % 21 = 4 ∨ n % 21 = 5 ∨ n % 21 = 6 ∨ n % 21 = 10 ∨ n % 21 = 15} :=
sorry

end find_natural_numbers_l29_29410


namespace new_edition_pages_less_l29_29657

theorem new_edition_pages_less :
  let new_edition_pages := 450
  let old_edition_pages := 340
  (2 * old_edition_pages - new_edition_pages) = 230 :=
by
  let new_edition_pages := 450
  let old_edition_pages := 340
  sorry

end new_edition_pages_less_l29_29657


namespace parabola_vertex_l29_29488

noncomputable def is_vertex (x y : ℝ) : Prop :=
  y^2 + 8 * y + 4 * x + 5 = 0 ∧ (∀ y₀, y₀^2 + 8 * y₀ + 4 * x + 5 ≥ 0)

theorem parabola_vertex : is_vertex (11 / 4) (-4) :=
by
  sorry

end parabola_vertex_l29_29488


namespace bags_production_l29_29135

def machines_bags_per_minute (n : ℕ) : ℕ :=
  if n = 15 then 45 else 0 -- this definition is constrained by given condition

def bags_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  machines * (machines_bags_per_minute 15 / 15) * minutes

theorem bags_production (machines minutes : ℕ) (h : machines = 150 ∧ minutes = 8):
  bags_produced machines minutes = 3600 :=
by
  cases h with
  | intro h_machines h_minutes =>
    sorry

end bags_production_l29_29135


namespace least_integer_x_l29_29920

theorem least_integer_x (x : ℤ) (h : 3 * |x| - 2 * x + 8 < 23) : x = -3 :=
sorry

end least_integer_x_l29_29920


namespace reconstruct_right_triangle_l29_29050

theorem reconstruct_right_triangle (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  ∃ A X Y: ℝ, (A ≠ X ∧ A ≠ Y ∧ X ≠ Y) ∧ 
  -- Right triangle with hypotenuse c
  (A - X) ^ 2 + (Y - X) ^ 2 = c ^ 2 ∧ 
  -- Difference of legs is d
  ∃ AY XY: ℝ, ((AY = abs (A - Y)) ∧ (XY = abs (Y - X)) ∧ (abs (AY - XY) = d)) := 
by
  sorry

end reconstruct_right_triangle_l29_29050


namespace monotonic_increasing_interval_l29_29350

open Real

theorem monotonic_increasing_interval (k : ℤ) : 
  ∀ x, -π / 12 + k * π ≤ x ∧ x ≤ 5 * π / 12 + k * π ↔ 
    ∀ t, -π / 2 + 2 * k * π ≤ 2 * t - π / 3 ∧ 2 * t - π / 3 ≤ π / 2 + 2 * k * π :=
sorry

end monotonic_increasing_interval_l29_29350


namespace inclination_angle_of_line_3x_sqrt3y_minus1_l29_29997

noncomputable def inclination_angle_of_line (A B C : ℝ) (h : A ≠ 0 ∧ B ≠ 0) : ℝ :=
  let m := -A / B 
  if m = Real.tan (120 * Real.pi / 180) then 120
  else 0 -- This will return 0 if the slope m does not match, for simplifying purposes

theorem inclination_angle_of_line_3x_sqrt3y_minus1 :
  inclination_angle_of_line 3 (Real.sqrt 3) (-1) (by sorry) = 120 := 
sorry

end inclination_angle_of_line_3x_sqrt3y_minus1_l29_29997


namespace ratio_of_sizes_l29_29897

-- Defining Anna's size
def anna_size : ℕ := 2

-- Defining Becky's size as three times Anna's size
def becky_size : ℕ := 3 * anna_size

-- Defining Ginger's size
def ginger_size : ℕ := 8

-- Defining the goal statement
theorem ratio_of_sizes : (ginger_size : ℕ) / (becky_size : ℕ) = 4 / 3 :=
by
  sorry

end ratio_of_sizes_l29_29897


namespace smallest_n_l29_29873

theorem smallest_n (n : ℕ) (h₁ : ∃ k₁ : ℕ, 5 * n = k₁ ^ 2) (h₂ : ∃ k₂ : ℕ, 4 * n = k₂ ^ 3) : n = 1600 :=
sorry

end smallest_n_l29_29873


namespace sum_of_solutions_eq_zero_l29_29416

noncomputable def f (x : ℝ) : ℝ := 2^(abs x) + 4 * abs x

theorem sum_of_solutions_eq_zero : 
  (∃ x : ℝ, f x = 20) ∧ (∃ y : ℝ, f y = 20 ∧ x = -y) → 
  x + y = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l29_29416


namespace find_y_in_terms_of_x_and_n_l29_29203

variable (x n y : ℝ)

theorem find_y_in_terms_of_x_and_n
  (h : n = 3 * x * y / (x - y)) :
  y = n * x / (3 * x + n) :=
  sorry

end find_y_in_terms_of_x_and_n_l29_29203


namespace impossible_to_form_11x12x13_parallelepiped_l29_29240

def is_possible_to_form_parallelepiped
  (brick_shapes_form_unit_cubes : Prop)
  (dimensions : ℕ × ℕ × ℕ) : Prop :=
  ∃ bricks : ℕ, 
    (bricks * 4 = dimensions.fst * dimensions.snd * dimensions.snd.fst)

theorem impossible_to_form_11x12x13_parallelepiped 
  (dimensions := (11, 12, 13)) 
  (brick_shapes_form_unit_cubes : Prop) : 
  ¬ is_possible_to_form_parallelepiped brick_shapes_form_unit_cubes dimensions := 
sorry

end impossible_to_form_11x12x13_parallelepiped_l29_29240


namespace Maggie_earnings_l29_29981

theorem Maggie_earnings
    (price_per_subscription : ℕ)
    (subscriptions_parents : ℕ)
    (subscriptions_grandfather : ℕ)
    (subscriptions_nextdoor : ℕ)
    (subscriptions_another : ℕ)
    (total_subscriptions : ℕ)
    (total_earnings : ℕ) :
    subscriptions_parents = 4 →
    subscriptions_grandfather = 1 →
    subscriptions_nextdoor = 2 →
    subscriptions_another = 2 * subscriptions_nextdoor →
    total_subscriptions = subscriptions_parents + subscriptions_grandfather + subscriptions_nextdoor + subscriptions_another →
    price_per_subscription = 5 →
    total_earnings = price_per_subscription * total_subscriptions →
    total_earnings = 55 :=
by
  intros
  sorry

end Maggie_earnings_l29_29981


namespace total_amount_spent_l29_29599

theorem total_amount_spent (T : ℝ) (h1 : 5000 + 200 + 0.30 * T = T) : 
  T = 7428.57 :=
by
  sorry

end total_amount_spent_l29_29599


namespace find_number_of_cows_l29_29953

-- Definitions from the conditions
def number_of_ducks : ℕ := sorry
def number_of_cows : ℕ := sorry

-- Define the number of legs and heads
def legs := 2 * number_of_ducks + 4 * number_of_cows
def heads := number_of_ducks + number_of_cows

-- Given condition from the problem
def condition := legs = 2 * heads + 32

-- Assert the number of cows
theorem find_number_of_cows (h : condition) : number_of_cows = 16 :=
sorry

end find_number_of_cows_l29_29953


namespace magnitude_product_complex_l29_29554

theorem magnitude_product_complex :
  let z1 := Complex.mk 7 (-4)
  let z2 := Complex.mk 3 11
  Complex.abs (z1 * z2) = Real.sqrt 8450 :=
by
  sorry

end magnitude_product_complex_l29_29554


namespace evaluate_magnitude_product_l29_29547

-- Definitions of complex numbers
def z1 := Complex.mk 7 (-4)
def z2 := Complex.mk 3 11

-- The magnitude of z1
def magnitude_z1 := Complex.abs z1

-- The magnitude of z2
def magnitude_z2 := Complex.abs z2

-- Lean 4 statement expressing the problem and its final answer
theorem evaluate_magnitude_product : Complex.abs (z1 * z2) = Real.sqrt 8450 := by
  sorry

end evaluate_magnitude_product_l29_29547


namespace smallest_digit_not_found_in_units_place_of_odd_number_l29_29851

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l29_29851


namespace average_perm_sum_eq_33_l29_29695

open Finset
open Function

-- Define the sum for a given permutation
def perm_sum (σ : Equiv.Perm (Fin 12)) : ℝ :=
  abs (σ 0 - σ 1) + abs (σ 2 - σ 3) + abs (σ 4 - σ 5) +
  abs (σ 6 - σ 7) + abs (σ 8 - σ 9) + abs (σ 10 - σ 11)

-- Main theorem statement
theorem average_perm_sum_eq_33 : 
  (1 / 12! : ℝ) * ∑ σ in univ, perm_sum σ = 33 :=
by sorry

end average_perm_sum_eq_33_l29_29695


namespace problem_statement_l29_29081

theorem problem_statement (a b c x y z : ℂ)
  (h1 : a = (b + c) / (x - 2))
  (h2 : b = (c + a) / (y - 2))
  (h3 : c = (a + b) / (z - 2))
  (h4 : x * y + y * z + z * x = 67)
  (h5 : x + y + z = 2010) :
  x * y * z = -5892 :=
by {
  sorry
}

end problem_statement_l29_29081


namespace circle_symmetry_line_l29_29097

theorem circle_symmetry_line :
  ∃ l: ℝ → ℝ → Prop, 
    (∀ x y, l x y → x - y + 2 = 0) ∧ 
    (∀ x y, l x y ↔ (x + 2)^2 + (y - 2)^2 = 4) :=
sorry

end circle_symmetry_line_l29_29097


namespace trig_expression_tangent_l29_29564

theorem trig_expression_tangent (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / 
  (Real.cos (3 * π / 2 + α) + Real.cos (π - α)) = 1 :=
sorry

end trig_expression_tangent_l29_29564


namespace ceil_sqrt_225_eq_15_l29_29014

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := 
by 
  sorry

end ceil_sqrt_225_eq_15_l29_29014


namespace remainder_expr_div_by_5_l29_29716

theorem remainder_expr_div_by_5 (n : ℤ) : 
  (7 - 2 * n + (n + 5)) % 5 = (-n + 2) % 5 := 
sorry

end remainder_expr_div_by_5_l29_29716


namespace sequence_diff_l29_29320

theorem sequence_diff (x : ℕ → ℕ)
  (h1 : ∀ n, x n < x (n + 1))
  (h2 : ∀ n, 2 * n + 1 ≤ x (2 * n + 1)) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
by
  sorry

end sequence_diff_l29_29320


namespace smallest_digit_not_in_units_place_of_odd_numbers_l29_29821

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l29_29821


namespace evaluate_ceil_sqrt_225_l29_29021

def ceil (x : ℝ) : ℤ :=
  if h : ∃ n : ℤ, n ≤ x ∧ x < n + 1 then
    classical.some h
  else
    0

theorem evaluate_ceil_sqrt_225 : ceil (Real.sqrt 225) = 15 := 
sorry

end evaluate_ceil_sqrt_225_l29_29021


namespace multiplication_result_l29_29485

theorem multiplication_result : 143 * 21 * 4 * 37 * 2 = 888888 := by
  sorry

end multiplication_result_l29_29485


namespace part1_part2_l29_29712

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -1)
noncomputable def vec_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 2), Real.cos (x / 2) ^ 2)
noncomputable def f (x : ℝ) : ℝ := (vec_m x).1 * (vec_n x).1 + (vec_m x).2 * (vec_n x).2 + 1

-- Part 1
theorem part1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) (hf : f x = 11 / 10) : 
  x = (Real.pi / 6) + Real.arcsin (3 / 5) :=
sorry

-- Part 2
theorem part2 {A B C a b c : ℝ} 
  (hABC : A + B + C = Real.pi) 
  (habc : 2 * b * Real.cos A ≤ 2 * c - Real.sqrt 3 * a) : 
  (0 < B ∧ B ≤ Real.pi / 6) → 
  ∃ y, (0 < y ∧ y ≤ 1 / 2 ∧ f B = y) :=
sorry

end part1_part2_l29_29712


namespace range_of_m_for_inversely_proportional_function_l29_29052

theorem range_of_m_for_inversely_proportional_function 
  (m : ℝ)
  (h : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > x₁ → (m - 1) / x₂ < (m - 1) / x₁) : 
  m > 1 :=
sorry

end range_of_m_for_inversely_proportional_function_l29_29052


namespace square_of_1023_l29_29001

theorem square_of_1023 : 1023^2 = 1045529 := by
  sorry

end square_of_1023_l29_29001


namespace smallest_digit_not_in_odd_units_l29_29838

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l29_29838


namespace intersection_point_l29_29402

-- Mathematical problem translated to Lean 4 statement

theorem intersection_point : 
  ∃ x y : ℝ, y = -3 * x + 1 ∧ y + 1 = 15 * x ∧ x = 1 / 9 ∧ y = 2 / 3 := 
by
  sorry

end intersection_point_l29_29402


namespace polynomial_power_degree_l29_29241

noncomputable def polynomial_degree (p : Polynomial ℝ) : ℕ := p.natDegree

theorem polynomial_power_degree : 
  polynomial_degree ((5 * X^3 - 4 * X + 7)^10) = 30 := by
  sorry

end polynomial_power_degree_l29_29241


namespace square_of_1023_l29_29000

theorem square_of_1023 : 1023^2 = 1045529 := by
  sorry

end square_of_1023_l29_29000


namespace simplify_expression_l29_29991

theorem simplify_expression (w : ℝ) : 2 * w + 4 * w + 6 * w + 8 * w + 10 * w + 12 = 30 * w + 12 :=
by
  sorry

end simplify_expression_l29_29991


namespace sum_of_areas_lt_side_length_square_l29_29437

variable (n : ℕ) (a : ℝ)
variable (S : Fin n → ℝ) (d : Fin n → ℝ)

-- Conditions
axiom areas_le_one : ∀ i, S i ≤ 1
axiom sum_d_le_a : (Finset.univ).sum d ≤ a
axiom areas_less_than_diameters : ∀ i, S i < d i

-- Theorem Statement
theorem sum_of_areas_lt_side_length_square :
  ((Finset.univ : Finset (Fin n)).sum S) < a :=
sorry

end sum_of_areas_lt_side_length_square_l29_29437


namespace remaining_pencils_l29_29458

theorem remaining_pencils (j_pencils : ℝ) (v_pencils : ℝ)
  (j_initial : j_pencils = 300) 
  (j_donated_pct : ℝ := 0.30)
  (v_initial : v_pencils = 2 * 300) 
  (v_donated_pct : ℝ := 0.75) :
  (j_pencils - j_donated_pct * j_pencils) + (v_pencils - v_donated_pct * v_pencils) = 360 :=
by
  sorry

end remaining_pencils_l29_29458


namespace value_of_A_l29_29771

theorem value_of_A {α : Type} [LinearOrderedSemiring α] 
  (L A D E : α) (L_value : L = 15) (LEAD DEAL DELL : α)
  (LEAD_value : LEAD = 50)
  (DEAL_value : DEAL = 55)
  (DELL_value : DELL = 60)
  (LEAD_condition : L + E + A + D = LEAD)
  (DEAL_condition : D + E + A + L = DEAL)
  (DELL_condition : D + E + L + L = DELL) :
  A = 25 :=
by
  sorry

end value_of_A_l29_29771


namespace problem_value_l29_29080

theorem problem_value (x : ℤ) (h : x = -2023) : 
  abs (abs (abs x - x) - abs x) - x = 4046 :=
by
  sorry

end problem_value_l29_29080


namespace maya_lift_increase_l29_29473

def initial_lift_America : ℕ := 240
def peak_lift_America : ℕ := 300

def initial_lift_Maya (a_lift : ℕ) : ℕ := a_lift / 4
def peak_lift_Maya (p_lift : ℕ) : ℕ := p_lift / 2

def lift_difference (initial_lift : ℕ) (peak_lift : ℕ) : ℕ := peak_lift - initial_lift

theorem maya_lift_increase :
  lift_difference (initial_lift_Maya initial_lift_America) (peak_lift_Maya peak_lift_America) = 90 :=
by
  -- Proof is skipped with sorry
  sorry

end maya_lift_increase_l29_29473


namespace evaluate_expression_l29_29406

noncomputable def x : ℚ := 4 / 8
noncomputable def y : ℚ := 5 / 6

theorem evaluate_expression : (8 * x + 6 * y) / (72 * x * y) = 3 / 10 :=
by
  sorry

end evaluate_expression_l29_29406


namespace smallest_unfound_digit_in_odd_units_l29_29827

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l29_29827


namespace polynomial_equivalence_l29_29976

-- Define the polynomial T in terms of x.
def T (x : ℝ) : ℝ := (x-2)^5 + 5 * (x-2)^4 + 10 * (x-2)^3 + 10 * (x-2)^2 + 5 * (x-2) + 1

-- Define the target polynomial.
def target (x : ℝ) : ℝ := (x-1)^5

-- State the theorem that T is equivalent to target.
theorem polynomial_equivalence (x : ℝ) : T x = target x :=
by
  sorry

end polynomial_equivalence_l29_29976


namespace number_of_solution_pairs_l29_29938

def integer_solutions_on_circle : Set (Int × Int) := {
  (1, 7), (1, -7), (-1, 7), (-1, -7),
  (5, 5), (5, -5), (-5, 5), (-5, -5),
  (7, 1), (7, -1), (-7, 1), (-7, -1) 
}

def system_of_equations_has_integer_solution (a b : ℝ) : Prop :=
  ∃ (x y : ℤ), a * ↑x + b * ↑y = 1 ∧ (↑x ^ 2 + ↑y ^ 2 = 50)

theorem number_of_solution_pairs : ∃ (n : ℕ), n = 72 ∧
  (∀ (a b : ℝ), system_of_equations_has_integer_solution a b → n = 72) := 
sorry

end number_of_solution_pairs_l29_29938


namespace inverse_value_l29_29300

def g (x : ℝ) : ℝ := 4 * x^3 + 5

theorem inverse_value (x : ℝ) (h : g (-3) = x) : (g ∘ g⁻¹) x = x := by
  sorry

end inverse_value_l29_29300


namespace Carlos_has_highest_result_l29_29264

def Alice_final_result : ℕ := 30 + 3
def Ben_final_result : ℕ := 34 + 3
def Carlos_final_result : ℕ := 13 * 3

theorem Carlos_has_highest_result : (Carlos_final_result > Alice_final_result) ∧ (Carlos_final_result > Ben_final_result) := by
  sorry

end Carlos_has_highest_result_l29_29264


namespace symmetric_points_coords_l29_29171

theorem symmetric_points_coords (a b : ℝ) :
    let N := (a, -b)
    let P := (-a, -b)
    let Q := (b, a)
    N = (a, -b) ∧ P = (-a, -b) ∧ Q = (b, a) →
    Q = (b, a) :=
by
  intro h
  sorry

end symmetric_points_coords_l29_29171


namespace necessary_condition_for_acute_angle_necessary_but_not_sufficient_condition_l29_29582

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (2, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 2)

-- Define the dot product calculation
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Conditionally state that x > -3 is necessary for an acute angle
theorem necessary_condition_for_acute_angle (x : ℝ) :
  dot_product vector_a (vector_b x) > 0 → x > -3 := by
  sorry

-- Define the theorem for necessary but not sufficient condition
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > -3) → (dot_product vector_a (vector_b x) > 0 ∧ x ≠ 4 / 3) := by
  sorry

end necessary_condition_for_acute_angle_necessary_but_not_sufficient_condition_l29_29582


namespace complex_number_in_first_quadrant_l29_29038

noncomputable def z : ℂ := Complex.ofReal 1 + Complex.I

theorem complex_number_in_first_quadrant 
  (h : Complex.ofReal 1 + Complex.I = Complex.I / z) : 
  (0 < z.re ∧ 0 < z.im) :=
  sorry

end complex_number_in_first_quadrant_l29_29038


namespace square_combinations_l29_29944

theorem square_combinations (n : ℕ) (h : n * (n - 1) = 30) : n * (n - 1) = 30 :=
by sorry

end square_combinations_l29_29944


namespace total_points_scored_l29_29328

-- Define the variables
def games : ℕ := 10
def points_per_game : ℕ := 12

-- Formulate the proposition to prove
theorem total_points_scored : games * points_per_game = 120 :=
by
  sorry

end total_points_scored_l29_29328


namespace exists_n_in_range_multiple_of_11_l29_29545

def is_multiple_of_11 (n : ℕ) : Prop :=
  (3 * n^5 + 4 * n^4 + 5 * n^3 + 7 * n^2 + 6 * n + 2) % 11 = 0

theorem exists_n_in_range_multiple_of_11 : ∃ n : ℕ, (2 ≤ n ∧ n ≤ 101) ∧ is_multiple_of_11 n :=
sorry

end exists_n_in_range_multiple_of_11_l29_29545


namespace valid_decomposition_2009_l29_29348

/-- A definition to determine whether a number can be decomposed
    into sums of distinct numbers with repeated digits representation. -/
def decomposable_2009 (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
  a = 1111 ∧ b = 777 ∧ c = 66 ∧ d = 55 ∧ a + b + c + d = n

theorem valid_decomposition_2009 :
  decomposable_2009 2009 :=
sorry

end valid_decomposition_2009_l29_29348


namespace restaurant_bill_l29_29218

theorem restaurant_bill
    (t : ℝ)
    (h1 : ∀ k : ℝ, k = 9 * (t / 10 + 3)) :
    t = 270 :=
by
    sorry

end restaurant_bill_l29_29218


namespace complex_modulus_problem_l29_29571

noncomputable def imaginary_unit : ℂ := Complex.I

theorem complex_modulus_problem (z : ℂ) (h : (1 + Real.sqrt 3 * imaginary_unit)^2 * z = 1 - imaginary_unit^3) :
  Complex.abs z = Real.sqrt 2 / 4 :=
by
  sorry

end complex_modulus_problem_l29_29571


namespace tire_price_l29_29355

theorem tire_price (payment : ℕ) (price_ratio : ℕ → ℕ → Prop)
  (h1 : payment = 345)
  (h2 : price_ratio 3 1)
  : ∃ x : ℕ, x = 99 := 
sorry

end tire_price_l29_29355


namespace steven_card_count_l29_29618

theorem steven_card_count (num_groups : ℕ) (cards_per_group : ℕ) (h_groups : num_groups = 5) (h_cards : cards_per_group = 6) : num_groups * cards_per_group = 30 := by
  sorry

end steven_card_count_l29_29618


namespace travel_time_reduction_l29_29455

theorem travel_time_reduction : 
  let t_initial := 19.5
  let factor_1998 := 1.30
  let factor_1999 := 1.25
  let factor_2000 := 1.20
  t_initial / factor_1998 / factor_1999 / factor_2000 = 10 := by
  sorry

end travel_time_reduction_l29_29455


namespace james_planted_60_percent_l29_29075

theorem james_planted_60_percent :
  let total_trees := 2
  let plants_per_tree := 20
  let seeds_per_plant := 1
  let total_seeds := total_trees * plants_per_tree * seeds_per_plant
  let planted_trees := 24
  (planted_trees / total_seeds) * 100 = 60 := 
by
  sorry

end james_planted_60_percent_l29_29075


namespace number_of_vans_needed_l29_29684

theorem number_of_vans_needed (capacity_per_van : ℕ) (students : ℕ) (adults : ℕ)
  (h_capacity : capacity_per_van = 9)
  (h_students : students = 40)
  (h_adults : adults = 14) :
  (students + adults + capacity_per_van - 1) / capacity_per_van = 6 := by
  sorry

end number_of_vans_needed_l29_29684


namespace modulus_product_l29_29552

open Complex

theorem modulus_product :
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  (Complex.abs (z1 * z2)) = Real.sqrt 8450 :=
by
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  sorry

end modulus_product_l29_29552


namespace smallest_digit_not_in_odd_units_l29_29839

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l29_29839


namespace center_of_circle_l29_29417

theorem center_of_circle (x y : ℝ) (h : x^2 + y^2 - 10 * x + 4 * y = -40) : 
  x + y = 3 := 
sorry

end center_of_circle_l29_29417


namespace ratio_norm_lisa_l29_29518

-- Define the number of photos taken by each photographer.
variable (L M N : ℕ)

-- Given conditions
def norm_photos : Prop := N = 110
def photo_sum_condition : Prop := L + M = M + N - 60

-- Prove the ratio of Norm's photos to Lisa's photos.
theorem ratio_norm_lisa (h1 : norm_photos N) (h2 : photo_sum_condition L M N) : N / L = 11 / 5 := 
by
  sorry

end ratio_norm_lisa_l29_29518


namespace solve_line_eq_l29_29999

theorem solve_line_eq (a b x : ℝ) (h1 : (0 : ℝ) * a + b = 2) (h2 : -3 * a + b = 0) : x = -3 :=
by
  sorry

end solve_line_eq_l29_29999


namespace num_boys_l29_29311

theorem num_boys (total_students : ℕ) (girls_ratio boys_ratio others_ratio : ℕ) (r : girls_ratio = 4) (b : boys_ratio = 3) (o : others_ratio = 2) (total_eq : girls_ratio * k + boys_ratio * k + others_ratio * k = total_students) (total_given : total_students = 63) : 
  boys_ratio * k = 21 :=
by
  sorry

end num_boys_l29_29311


namespace parallel_lines_a_value_l29_29625

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, ((a + 1) * x + 3 * y + 3 = 0) → (x + (a - 1) * y + 1 = 0)) → a = -2 :=
by
  sorry

end parallel_lines_a_value_l29_29625


namespace expression_value_l29_29286

theorem expression_value : (28 * 2 + (48 / 6) ^ 2 - 5) * (69 / 3) + 24 * (3 ^ 2 - 2) = 2813 := by
  sorry

end expression_value_l29_29286


namespace rectangle_area_l29_29653

theorem rectangle_area (r length width : ℝ) (h_ratio : length = 3 * width) (h_incircle : width = 2 * r) (h_r : r = 7) : length * width = 588 :=
by
  sorry

end rectangle_area_l29_29653


namespace find_value_l29_29059

theorem find_value (a b : ℝ) (h1 : 2 * a - 3 * b = 1) : 5 - 4 * a + 6 * b = 3 := 
by
  sorry

end find_value_l29_29059


namespace value_of_y_l29_29009

theorem value_of_y (y : ℚ) : |4 * y - 6| = 0 ↔ y = 3 / 2 :=
by
  sorry

end value_of_y_l29_29009


namespace sum_opposite_sign_zero_l29_29510

def opposite_sign (a b : ℝ) : Prop :=
(a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)

theorem sum_opposite_sign_zero {a b : ℝ} (h : opposite_sign a b) : a + b = 0 :=
sorry

end sum_opposite_sign_zero_l29_29510


namespace alex_sam_sum_difference_is_10875_l29_29543

noncomputable def alex_sum : ℕ :=
(150 * 151) / 2

noncomputable def sam_sum : ℕ :=
30 * 15

def alex_sam_difference : ℕ :=
(abs (alex_sum - sam_sum))

theorem alex_sam_sum_difference_is_10875 : alex_sam_difference = 10875 := by
  sorry

end alex_sam_sum_difference_is_10875_l29_29543


namespace sixth_grade_boys_l29_29387

theorem sixth_grade_boys (x : ℕ) :
    (1 / 11) * x + (147 - x) = 147 - x → 
    (152 - (x - (1 / 11) * x + (147 - x) - (152 - x - 5))) = x
    → x = 77 :=
by
  intros h1 h2
  sorry

end sixth_grade_boys_l29_29387


namespace number_of_books_is_8_l29_29635

def books_and_albums (x y p_a p_b : ℕ) : Prop :=
  (x * p_b = 1056) ∧ (p_b = p_a + 100) ∧ (x = y + 6)

theorem number_of_books_is_8 (y p_a p_b : ℕ) (h : books_and_albums 8 y p_a p_b) : 8 = 8 :=
by
  sorry

end number_of_books_is_8_l29_29635


namespace ceil_sqrt_225_eq_15_l29_29027

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := by
  sorry

end ceil_sqrt_225_eq_15_l29_29027


namespace mitchell_pizzas_l29_29611

def pizzas_bought (slices_per_goal goals_per_game games slices_per_pizza : ℕ) : ℕ :=
  (slices_per_goal * goals_per_game * games) / slices_per_pizza

theorem mitchell_pizzas : pizzas_bought 1 9 8 12 = 6 := by
  sorry

end mitchell_pizzas_l29_29611


namespace f_irreducible_l29_29604

noncomputable def f (n : ℕ) (x : ℤ) : ℤ := x^n + 5 * x^(n-1) + 3

theorem f_irreducible (n : ℕ) (hn : n > 1) : Irreducible (f n) :=
sorry

end f_irreducible_l29_29604


namespace y_value_on_line_l29_29070

theorem y_value_on_line (x y : ℝ) (k : ℝ → ℝ)
  (h1 : k 0 = 0)
  (h2 : ∀ x, k x = (1/5) * x)
  (hx1 : k x = 1)
  (hx2 : k 5 = y) :
  y = 1 :=
sorry

end y_value_on_line_l29_29070


namespace sin_2alpha_eq_f_increasing_interval_l29_29288

noncomputable def sin_alpha := 4 / 5
noncomputable def alpha := real.sin_pi_sub (4/5)

theorem sin_2alpha_eq : α \in (0, real.pi / 2) -> sin 2 * α - (cos (α / 2))^2 = 4 / 25 :=
by {
  sorry
}

theorem f_increasing_interval : α \in (0, real.pi / 2) -> sin (real.pi - α) = 4 / 5 -> 
(intervals k : \mathbb{Z}, f x = 5 / 6 * cos α * sin (2 * x) - 1 / 2 * cos (2 * x)) -> 
(x > 0, f(x) > f(y)) = (\forall k \in Int, [k real.pi - real.pi / 8, k real.pi + 3 * real.pi / 8]) :=
by {
  sorry
}

end sin_2alpha_eq_f_increasing_interval_l29_29288


namespace fraction_identity_l29_29407

variables {a b : ℝ}

theorem fraction_identity (h : a^2 + b^2 ≠ 0) :
  (a^4 + b^4) / (a^2 + b^2) = a^2 + b^2 - (2 * a^2 * b^2) / (a^2 + b^2) :=
by sorry

end fraction_identity_l29_29407


namespace area_of_square_with_perimeter_32_l29_29519

theorem area_of_square_with_perimeter_32 :
  ∀ (s : ℝ), 4 * s = 32 → s * s = 64 :=
by
  intros s h
  sorry

end area_of_square_with_perimeter_32_l29_29519


namespace smallest_digit_not_in_units_place_of_odd_l29_29805

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l29_29805


namespace minimum_value_N_div_a4_possible_values_a4_l29_29324

noncomputable def lcm_10 (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ) : ℕ := 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm a1 a2) a3) a4) a5) a6) a7) a8) a9) a10

theorem minimum_value_N_div_a4 {a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ} 
  (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6 ∧ a6 < a7 ∧ a7 < a8 ∧ a8 < a9 ∧ a9 < a10) : 
  (lcm_10 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10) / a4 = 630 := sorry

theorem possible_values_a4 {a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ} 
  (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6 ∧ a6 < a7 ∧ a7 < a8 ∧ a8 < a9 ∧ a9 < a10)
  (z: 1 ≤ a4 ∧ a4 ≤ 1300) :
  (lcm_10 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10) / a4 = 630 → a4 = 360 ∨ a4 = 720 ∨ a4 = 1080 := sorry

end minimum_value_N_div_a4_possible_values_a4_l29_29324


namespace tangent_line_through_origin_unique_real_root_when_a_neg_l29_29710

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem tangent_line_through_origin (a : ℝ) : 
  let y := (a * Real.exp 1 - 1) / Real.exp 1 * (x : ℝ) 
  in y = (a - 1 / Real.exp 1) * x :=
begin
  sorry
end

theorem unique_real_root_when_a_neg (a : ℝ) (h : a < 0) : 
  ∃ (x : ℝ), (f a x) + a * x^2 = 0 ∧ ∀ x' : ℝ, (f a x') + a * x'^2 = 0 → x = x' :=
begin
  sorry
end

end tangent_line_through_origin_unique_real_root_when_a_neg_l29_29710


namespace probability_set_A_on_Saturday_l29_29636

-- Define the probability function P for each day
def P : ℕ → ℚ
| 1 := 1             -- P_1 = 1 since set A is used on Monday
| 2 := 0             -- P_2 = 0 since set A was just used on Monday
| 3 := 1 / 3         -- P_3 = 1 / 3 since there are three equally likely sets for Wednesday
| n := (1 - P (n-1)) * 1 / 3   -- Recurrence relation for P_n for n > 3

-- Prove the probability of using set A on Saturday
theorem probability_set_A_on_Saturday : P 6 = 20 / 81 :=
by { sorry }

end probability_set_A_on_Saturday_l29_29636


namespace packet_weight_l29_29477

theorem packet_weight
  (tons_to_pounds : ℕ := 2600) -- 1 ton = 2600 pounds
  (total_tons : ℕ := 13)       -- Total capacity in tons
  (num_packets : ℕ := 2080)    -- Number of packets
  (expected_weight_per_packet : ℚ := 16.25) : 
  total_tons * tons_to_pounds / num_packets = expected_weight_per_packet := 
sorry

end packet_weight_l29_29477


namespace chenny_friends_l29_29905

noncomputable def num_friends (initial_candies add_candies candies_per_friend) : ℕ :=
  (initial_candies + add_candies) / candies_per_friend

theorem chenny_friends :
  num_friends 10 4 2 = 7 :=
by
  sorry

end chenny_friends_l29_29905


namespace line_does_not_pass_second_quadrant_l29_29106

-- Definitions of conditions
variables (k b x y : ℝ)
variable  (h₁ : k > 0) -- condition k > 0
variable  (h₂ : b < 0) -- condition b < 0


theorem line_does_not_pass_second_quadrant : 
  ¬∃ (x y : ℝ), (x < 0 ∧ y > 0) ∧ (y = k * x + b) :=
sorry

end line_does_not_pass_second_quadrant_l29_29106


namespace arithmetic_sequence_common_difference_l29_29434

theorem arithmetic_sequence_common_difference (a_1 a_5 d : ℝ) 
  (h1 : a_5 = a_1 + 4 * d) 
  (h2 : a_1 + (a_1 + d) + (a_1 + 2 * d) = 6) : 
  d = 2 := 
  sorry

end arithmetic_sequence_common_difference_l29_29434


namespace frederick_final_amount_l29_29507

-- Definitions of conditions
def P : ℝ := 2000
def r : ℝ := 0.05
def n : ℕ := 18

-- Define the compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Theorem stating the question's answer
theorem frederick_final_amount : compound_interest P r n = 4813.24 :=
by
  sorry

end frederick_final_amount_l29_29507


namespace common_chord_and_length_l29_29435

-- Define the two circles
def circle1 (x y : ℝ) := x^2 + y^2 + 2*x - 4*y - 5 = 0
def circle2 (x y : ℝ) := x^2 + y^2 + 2*x - 1 = 0

-- The theorem statement with the conditions and expected solutions
theorem common_chord_and_length :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → y = -1)
  ∧
  (∃ A B : (ℝ × ℝ), (circle1 A.1 A.2 ∧ circle2 A.1 A.2) ∧ 
                    (circle1 B.1 B.2 ∧ circle2 B.1 B.2) ∧ 
                    (|A.1 - B.1|^2 + |A.2 - B.2|^2 = 4)) :=
by
  sorry

end common_chord_and_length_l29_29435


namespace certain_number_is_four_l29_29694

theorem certain_number_is_four (k : ℕ) (h₁ : k = 16) : 64 / k = 4 :=
by
  sorry

end certain_number_is_four_l29_29694


namespace LCM_activities_l29_29393

theorem LCM_activities :
  ∃ (d : ℕ), d = Nat.lcm 6 (Nat.lcm 4 (Nat.lcm 16 (Nat.lcm 12 8))) ∧ d = 48 :=
by
  sorry

end LCM_activities_l29_29393


namespace incorrect_statement_l29_29039

-- Conditions
variable (A B C D E F : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]
variables (triangleABC : Triangle A B C) (triangleDEF : Triangle D E F)

-- Congruence of triangles
axiom congruent_triangles : triangleABC ≌ triangleDEF

-- Proving incorrect statement
theorem incorrect_statement : ¬ (AB = EF) := by
  sorry

end incorrect_statement_l29_29039


namespace diane_stamp_combinations_l29_29403

theorem diane_stamp_combinations :
  let stamps := [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
  in stamps.count_combinations(15) = 168 :=
sorry

end diane_stamp_combinations_l29_29403


namespace probability_adjacent_vertices_dodecagon_l29_29374

noncomputable def prob_adjacent_vertices_dodecagon : ℚ :=
  let total_vertices := 12
  let favorable_outcomes := 2  -- adjacent vertices per chosen vertex
  let total_outcomes := total_vertices - 1  -- choosing any other vertex
  favorable_outcomes / total_outcomes

theorem probability_adjacent_vertices_dodecagon :
  prob_adjacent_vertices_dodecagon = 2 / 11 := by
  sorry

end probability_adjacent_vertices_dodecagon_l29_29374


namespace yellow_sweets_l29_29968

-- Definitions
def green_sweets : Nat := 212
def blue_sweets : Nat := 310
def sweets_per_person : Nat := 256
def people : Nat := 4

-- Proof problem statement
theorem yellow_sweets : green_sweets + blue_sweets + x = sweets_per_person * people → x = 502 := by
  sorry

end yellow_sweets_l29_29968


namespace irreducible_f_l29_29605

def f (n : ℕ) (x : ℤ) : ℤ := x^n + 5 * x^(n - 1) + 3

theorem irreducible_f (n : ℕ) (hn : n > 1) : Irreducible (f n : ℤ[X]) :=
  sorry

end irreducible_f_l29_29605


namespace smallest_n_perfect_square_and_cube_l29_29871

theorem smallest_n_perfect_square_and_cube (n : ℕ) (h1 : ∃ k : ℕ, 5 * n = k^2) (h2 : ∃ m : ℕ, 4 * n = m^3) :
  n = 1080 :=
  sorry

end smallest_n_perfect_square_and_cube_l29_29871


namespace find_a_of_complex_roots_and_abs_sum_l29_29177

open Complex

theorem find_a_of_complex_roots_and_abs_sum (a : ℝ) (x1 x2 : ℂ)
  (h1 : x1^2 - 2 * x1 * a + (a^2 - 4 * a + 4) = 0)
  (h2 : x2^2 - 2 * x2 * a + (a^2 - 4 * a + 4) = 0)
  (h3 : x1 ≠ x2) -- ensures distinct but still conjugates automatically by being complex
  (h4 : |x1| + |x2| = 3) : a = 1 / 2 := 
sorry

end find_a_of_complex_roots_and_abs_sum_l29_29177


namespace inverse_implications_l29_29957

-- Definitions used in the conditions
def not_coplanar (points : set (set point)) : Prop :=
  ∃ p1 p2 p3 p4 ∈ points, ¬ (affine_span ℝ {p1, p2, p3, p4} ≤ affine_span ℝ {p1, p2, p3})

def not_collinear (points : set point) : Prop :=
  ∀ p1 p2 p3 ∈ points, ¬ (affine_span ℝ {p1, p2, p3} ≤ line ℝ p1 p2)

def skew_lines (l1 l2 : line ℝ) : Prop :=
  ¬ ∃ p, p ∈ l1 ∧ p ∈ l2

-- Problem statement
theorem inverse_implications :
  (¬ ∀ points : set point, not_collinear points → not_coplanar (insert points))
  ∧ 
  (∀ l1 l2 : line ℝ, skew_lines l1 l2 → ¬ ∃ p, p ∈ l1 ∧ p ∈ l2) :=
sorry

end inverse_implications_l29_29957


namespace line_equation_correct_l29_29927

-- Definitions for the conditions
def point := ℝ × ℝ
def vector := ℝ × ℝ

-- Given the line has a direction vector and passes through a point
def line_has_direction_vector (l : point → Prop) (v : vector) : Prop :=
  ∀ p₁ p₂ : point, l p₁ → l p₂ → (p₂.1 - p₁.1, p₂.2 - p₁.2) = v

def line_passes_through_point (l : point → Prop) (p : point) : Prop :=
  l p

-- The line equation in point-direction form
def line_equation (x y : ℝ) : Prop :=
  (x - 1) / 2 = y / -3

-- Main statement
theorem line_equation_correct :
  ∃ l : point → Prop, 
    line_has_direction_vector l (2, -3) ∧
    line_passes_through_point l (1, 0) ∧
    ∀ x y, l (x, y) ↔ line_equation x y := 
sorry

end line_equation_correct_l29_29927


namespace direction_cosines_l29_29412

theorem direction_cosines (x y z : ℝ) (α β γ : ℝ)
  (h1 : 2 * x - 3 * y - 3 * z - 9 = 0)
  (h2 : x - 2 * y + z + 3 = 0) :
  α = 9 / Real.sqrt 107 ∧ β = 5 / Real.sqrt 107 ∧ γ = 1 / Real.sqrt 107 :=
by
  -- Here, we will sketch out the proof to establish that these values for α, β, and γ hold.
  sorry

end direction_cosines_l29_29412


namespace div_transitivity_l29_29445

theorem div_transitivity (a b c : ℚ) : 
  (a / b = 3) → (b / c = 2 / 5) → (c / a = 5 / 6) :=
by 
  intros h1 h2
  have : c / a = (c / b) * (b / a),
  { field_simp, }
  rw h1 at this,
  rw h2 at this,
  field_simp at this,
  exact this

end div_transitivity_l29_29445


namespace unique_solution_exists_l29_29131

theorem unique_solution_exists :
  ∃ (a b c d e : ℕ),
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
  a + b = 1/7 * (c + d + e) ∧
  a + c = 1/5 * (b + d + e) ∧
  (a, b, c, d, e) = (1, 2, 3, 9, 9) :=
by {
  sorry
}

end unique_solution_exists_l29_29131


namespace smallest_positive_integer_n_l29_29878

theorem smallest_positive_integer_n (n : ℕ) 
  (h1 : ∃ k : ℕ, n = 5 * k ∧ perfect_square(5 * k)) 
  (h2 : ∃ m : ℕ, n = 4 * m ∧ perfect_cube(4 * m)) : 
  n = 625000 :=
sorry

end smallest_positive_integer_n_l29_29878


namespace eval_expression_l29_29004

def f (x : ℤ) : ℤ := 3 * x^2 - 6 * x + 10

theorem eval_expression : 3 * f 2 + 2 * f (-2) = 98 := by
  sorry

end eval_expression_l29_29004


namespace simplify_and_evaluate_expression_l29_29336

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (3 / (a - 1) + (a - 3) / (a^2 - 1)) / (a / (a + 1)) = 2 * Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_expression_l29_29336


namespace x_intercept_l29_29690

theorem x_intercept (x y : ℝ) (h : 4 * x - 3 * y = 24) (hy : y = 0) : (x, y) = (6, 0) :=
by {
  sorry
}

end x_intercept_l29_29690


namespace smallest_missing_digit_l29_29861

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l29_29861


namespace chickens_after_9_years_l29_29610

-- Definitions from the conditions
def annual_increase : ℕ := 150
def current_chickens : ℕ := 550
def years : ℕ := 9

-- Lean statement for the proof
theorem chickens_after_9_years : current_chickens + annual_increase * years = 1900 :=
by
  sorry

end chickens_after_9_years_l29_29610


namespace larger_number_of_hcf_lcm_is_322_l29_29770

theorem larger_number_of_hcf_lcm_is_322
  (A B : ℕ)
  (hcf: ℕ := 23)
  (factor1 : ℕ := 13)
  (factor2 : ℕ := 14)
  (hcf_condition : ∀ d, d ∣ A → d ∣ B → d ≤ hcf)
  (lcm_condition : ∀ m n, m * n = A * B → m = factor1 * hcf ∨ m = factor2 * hcf) :
  max A B = 322 :=
by sorry

end larger_number_of_hcf_lcm_is_322_l29_29770


namespace packets_of_chips_l29_29489

theorem packets_of_chips (x : ℕ) 
  (h1 : ∀ x, 2 * (x : ℝ) + 1.5 * (10 : ℝ) = 45) : 
  x = 15 := 
by 
  sorry

end packets_of_chips_l29_29489


namespace focal_length_of_curve_l29_29622

theorem focal_length_of_curve : 
  (∀ θ : ℝ, ∃ x y : ℝ, x = 2 * Real.cos θ ∧ y = Real.sin θ) →
  ∃ f : ℝ, f = 2 * Real.sqrt 3 :=
by sorry

end focal_length_of_curve_l29_29622


namespace inconsistent_proportion_l29_29167

theorem inconsistent_proportion (a b : ℝ) (h1 : 3 * a = 5 * b) (ha : a ≠ 0) (hb : b ≠ 0) : ¬ (a / b = 3 / 5) :=
sorry

end inconsistent_proportion_l29_29167


namespace max_value_of_symmetric_f_l29_29723

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_f :
  ∀ (a b : ℝ),
    (f 1 a b = 0) →
    (f (-1) a b = 0) →
    (f (-5) a b = 0) →
    (f (-3) a b = 0) →
    (∃ x : ℝ, f x 8 15 = 16) :=
by
  sorry

end max_value_of_symmetric_f_l29_29723


namespace normal_expectation_variance_l29_29162

theorem normal_expectation_variance (X : ℝ → ℝ) (a σ : ℝ) (hX : IsNormalDistribution X a σ²) :
  (expectation X = a) ∧ (variance X = σ²) :=
sorry

end normal_expectation_variance_l29_29162


namespace basketball_team_lineup_l29_29376

-- Define the problem conditions
def total_players : ℕ := 12
def twins : ℕ := 2
def lineup_size : ℕ := 5
def remaining_players : ℕ := total_players - twins
def positions_to_fill : ℕ := lineup_size - twins

-- Define the combination function as provided in the standard libraries
def combination (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem translating to the proof problem
theorem basketball_team_lineup : combination remaining_players positions_to_fill = 120 := 
sorry

end basketball_team_lineup_l29_29376


namespace product_of_05_and_2_3_is_1_3_l29_29153

theorem product_of_05_and_2_3_is_1_3 : (0.5 * (2 / 3) = 1 / 3) :=
by sorry

end product_of_05_and_2_3_is_1_3_l29_29153


namespace sqrt_x_minus_2_range_l29_29189

theorem sqrt_x_minus_2_range (x : ℝ) : (↑0 ≤ (x - 2)) ↔ (x ≥ 2) := sorry

end sqrt_x_minus_2_range_l29_29189


namespace range_of_a_l29_29704

theorem range_of_a (a : ℝ) (p : ∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) (q : 0 < 2 * a - 1 ∧ 2 * a - 1 < 1) : 
  (1 / 2) < a ∧ a ≤ (2 / 3) :=
sorry

end range_of_a_l29_29704


namespace ratio_of_james_to_jacob_l29_29472

noncomputable def MarkJumpHeight : ℕ := 6
noncomputable def LisaJumpHeight : ℕ := 2 * MarkJumpHeight
noncomputable def JacobJumpHeight : ℕ := 2 * LisaJumpHeight
noncomputable def JamesJumpHeight : ℕ := 16

theorem ratio_of_james_to_jacob : (JamesJumpHeight : ℚ) / (JacobJumpHeight : ℚ) = 2 / 3 :=
by
  sorry

end ratio_of_james_to_jacob_l29_29472


namespace a_range_condition_l29_29384

theorem a_range_condition (a : ℝ) : 
  (∀ x y : ℝ, ((x + a)^2 + (y - a)^2 < 4) → (x = -1 ∧ y = -1)) → 
  -1 < a ∧ a < 1 :=
by
  sorry

end a_range_condition_l29_29384


namespace curve_is_hyperbola_l29_29316

theorem curve_is_hyperbola (m n x y : ℝ) (h_eq : m * x^2 - m * y^2 = n) (h_mn : m * n < 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b^2/a^2 - x^2/a^2 = 1 := 
sorry

end curve_is_hyperbola_l29_29316


namespace expand_expression_l29_29686

theorem expand_expression (x y : ℝ) :
  (x + 3) * (4 * x - 5 * y) = 4 * x ^ 2 - 5 * x * y + 12 * x - 15 * y :=
by
  sorry

end expand_expression_l29_29686


namespace remainder_N_div_5_is_1_l29_29652

-- The statement proving the remainder of N when divided by 5 is 1
theorem remainder_N_div_5_is_1 (N : ℕ) (h1 : N % 2 = 1) (h2 : N % 35 = 1) : N % 5 = 1 :=
sorry

end remainder_N_div_5_is_1_l29_29652


namespace inequality_B_l29_29431

variable {x y : ℝ}

theorem inequality_B (hx : 0 < x) (hy : 0 < y) (hxy : x > y) : x + 1 / (2 * y) > y + 1 / x :=
sorry

end inequality_B_l29_29431


namespace smallest_digit_never_in_units_place_of_odd_number_l29_29869

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l29_29869


namespace reduce_fraction_l29_29369

-- Defining a structure for a fraction
structure Fraction where
  num : ℕ
  denom : ℕ
  deriving Repr

-- The original fraction
def originalFraction : Fraction :=
  { num := 368, denom := 598 }

-- The reduced fraction
def reducedFraction : Fraction :=
  { num := 184, denom := 299 }

-- The statement of our theorem
theorem reduce_fraction :
  ∃ (d : ℕ), d > 0 ∧ (originalFraction.num / d = reducedFraction.num) ∧ (originalFraction.denom / d = reducedFraction.denom) := by
  sorry

end reduce_fraction_l29_29369


namespace min_value_of_expression_l29_29034

theorem min_value_of_expression 
  (x y : ℝ) 
  (h : 3 * |x - y| + |2 * x - 5| = x + 1) : 
  ∃ (x y : ℝ), 2 * x + y = 4 :=
by {
  sorry
}

end min_value_of_expression_l29_29034


namespace smallest_digit_not_in_units_place_of_odd_l29_29797

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l29_29797


namespace range_of_f_is_pi_div_four_l29_29030

noncomputable def f (x : ℝ) : ℝ := 
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f_is_pi_div_four : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y = π / 4 :=
sorry

end range_of_f_is_pi_div_four_l29_29030


namespace adam_bought_dog_food_packages_l29_29149

-- Define the constants and conditions
def num_cat_food_packages : ℕ := 9
def cans_per_cat_food_package : ℕ := 10
def cans_per_dog_food_package : ℕ := 5
def additional_cat_food_cans : ℕ := 55

-- Define the variable for dog food packages and our equation
def num_dog_food_packages (d : ℕ) : Prop :=
  (num_cat_food_packages * cans_per_cat_food_package) = (d * cans_per_dog_food_package + additional_cat_food_cans)

-- The theorem statement representing the proof problem
theorem adam_bought_dog_food_packages : ∃ d : ℕ, num_dog_food_packages d ∧ d = 7 :=
sorry

end adam_bought_dog_food_packages_l29_29149


namespace min_value_l29_29570

theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) : a + 4 * b ≥ 9 :=
sorry

end min_value_l29_29570


namespace sales_worth_l29_29130

variable (S : ℝ)
def old_remuneration (S : ℝ) : ℝ := 0.05 * S
def new_remuneration (S : ℝ) : ℝ := 1300 + 0.025 * (S - 4000)

theorem sales_worth :
  new_remuneration S = old_remuneration S + 600 → S = 24000 :=
by
  intro h
  sorry

end sales_worth_l29_29130


namespace functional_ineq_solution_l29_29744

theorem functional_ineq_solution (n : ℕ) (h : n > 0) :
  (∀ x : ℝ, n = 1 → (x^n + (1 - x)^n ≤ 1)) ∧
  (∀ x : ℝ, n > 1 → ((x < 0 ∨ x > 1) → (x^n + (1 - x)^n > 1))) :=
by
  intros
  sorry

end functional_ineq_solution_l29_29744


namespace line_parallel_through_point_line_perpendicular_through_point_line_with_equal_intercepts_through_point_l29_29919

-- Define the line passing through a given point and parallel to another line
theorem line_parallel_through_point 
  (p : Point) (line : Line) (a b c : ℝ) (hp : p = (-1,3)) (hline : line = Equation (Coeff a b c) 
  (a = 1) (b = -2) (c = 3)) : Line := 
Proof
  sorry

-- Define the line passing through a given point and perpendicular to another line
theorem line_perpendicular_through_point 
  (p : Point) (line : Line) (a b c : ℝ) (hp : p = (3,4)) (hline : line = Equation (Coeff a b c) 
  (a = 3) (b = -1) (c = 2)) : Line := 
Proof
  sorry

-- Define the line passing through a given point with equal intercepts on both axes
theorem line_with_equal_intercepts_through_point 
  (p : Point) (intercept : ℝ) (hp : p = (1,2)) (hintercept : intercept = intercept): Line := 
Proof
  sorry

noncomputable def Point : Type* := ℝ × ℝ

noncomputable def Coeff (a b c : ℝ) : Type := ∀ (x y z: ℤ), a * x + b * y + c * z = 0

noncomputable def Equation (c : Coeff) : Type := Refl c

end line_parallel_through_point_line_perpendicular_through_point_line_with_equal_intercepts_through_point_l29_29919


namespace smallest_digit_never_in_units_place_of_odd_number_l29_29866

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l29_29866


namespace non_mobile_payment_probability_40_60_l29_29330

variable (total_customers : ℕ)
variable (num_non_mobile_40_50 : ℕ)
variable (num_non_mobile_50_60 : ℕ)

theorem non_mobile_payment_probability_40_60 
  (h_total_customers: total_customers = 100)
  (h_num_non_mobile_40_50: num_non_mobile_40_50 = 9)
  (h_num_non_mobile_50_60: num_non_mobile_50_60 = 5) : 
  (num_non_mobile_40_50 + num_non_mobile_50_60 : ℚ) / total_customers = 7 / 50 :=
by
  -- Placeholder for the actual proof
  sorry

end non_mobile_payment_probability_40_60_l29_29330


namespace value_of_y_l29_29008

theorem value_of_y (y : ℚ) : |4 * y - 6| = 0 ↔ y = 3 / 2 :=
by
  sorry

end value_of_y_l29_29008


namespace find_third_polygon_sides_l29_29780

def interior_angle (n : ℕ) : ℚ :=
  (n - 2) * 180 / n

theorem find_third_polygon_sides :
  let square_angle := interior_angle 4
  let pentagon_angle := interior_angle 5
  let third_polygon_angle := 360 - (square_angle + pentagon_angle)
  ∃ (m : ℕ), interior_angle m = third_polygon_angle ∧ m = 20 :=
by
  let square_angle := interior_angle 4
  let pentagon_angle := interior_angle 5
  let third_polygon_angle := 360 - (square_angle + pentagon_angle)
  use 20
  sorry

end find_third_polygon_sides_l29_29780


namespace eval_expression_l29_29005

def f (x : ℤ) : ℤ := 3 * x^2 - 6 * x + 10

theorem eval_expression : 3 * f 2 + 2 * f (-2) = 98 := by
  sorry

end eval_expression_l29_29005


namespace cost_of_tax_free_items_l29_29739

theorem cost_of_tax_free_items (total_cost : ℝ) (tax_40_percent : ℝ) 
  (tax_30_percent : ℝ) (discount : ℝ) : 
  (total_cost = 120) →
  (tax_40_percent = 0.4 * total_cost) →
  (tax_30_percent = 0.3 * total_cost) →
  (discount = 0.05 * tax_30_percent) →
  (tax-free_items = total_cost - (tax_40_percent + (tax_30_percent - discount))) → 
  tax_free_items = 36 :=
by sorry

end cost_of_tax_free_items_l29_29739


namespace parallel_vectors_l29_29055

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2, 1)) (h₂ : b = (1, m))
  (h₃ : ∃ k : ℝ, b = k • a) : m = 1 / 2 :=
by 
  sorry

end parallel_vectors_l29_29055


namespace y_coordinate_of_second_point_l29_29594

theorem y_coordinate_of_second_point
  (m n : ℝ)
  (h₁ : m = 2 * n + 3)
  (h₂ : m + 2 = 2 * (n + 1) + 3) :
  (n + 1) = n + 1 :=
by
  -- proof to be provided
  sorry

end y_coordinate_of_second_point_l29_29594


namespace color_of_last_bead_is_white_l29_29074

-- Defining the pattern of the beads
inductive BeadColor
| White
| Black
| Red

open BeadColor

-- Define the repeating pattern of the beads
def beadPattern : ℕ → BeadColor
| 0 => White
| 1 => Black
| 2 => Black
| 3 => Red
| 4 => Red
| 5 => Red
| (n + 6) => beadPattern n

-- Define the total number of beads
def totalBeads : ℕ := 85

-- Define the position of the last bead
def lastBead : ℕ := totalBeads - 1

-- Proving the color of the last bead
theorem color_of_last_bead_is_white : beadPattern lastBead = White :=
by
  sorry

end color_of_last_bead_is_white_l29_29074


namespace at_least_one_first_grade_product_l29_29637

noncomputable def probability_first_intern := 2 / 3
noncomputable def probability_second_intern := 1 / 2

theorem at_least_one_first_grade_product :
  let pA := probability_first_intern,
      pB := probability_second_intern,
      p_not_A := 1 - pA,
      p_not_B := 1 - pB,
      p_neither := p_not_A * p_not_B,
      p_at_least_one := 1 - p_neither in
  p_at_least_one = 5 / 6 :=
by
  let pA := probability_first_intern
  let pB := probability_second_intern
  let p_not_A := 1 - pA
  let p_not_B := 1 - pB
  let p_neither := p_not_A * p_not_B
  let p_at_least_one := 1 - p_neither
  show p_at_least_one = 5 / 6
  sorry

end at_least_one_first_grade_product_l29_29637


namespace range_m_for_p_range_m_for_q_range_m_for_not_p_or_q_l29_29978

-- Define the propositions p and q
def p (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + 2 + m = 0

def q (m : ℝ) : Prop :=
  1 - 2 * m < 0 ∧ m + 2 > 0 ∨ 1 - 2 * m > 0 ∧ m + 2 < 0 -- Hyperbola condition

-- Prove the ranges of m
theorem range_m_for_p {m : ℝ} (hp : p m) : m ≤ -2 ∨ m ≥ 1 :=
sorry

theorem range_m_for_q {m : ℝ} (hq : q m) : m < -2 ∨ m > (1 / 2) :=
sorry

theorem range_m_for_not_p_or_q {m : ℝ} (h_not_p : ¬ (p m)) (h_not_q : ¬ (q m)) : -2 < m ∧ m ≤ (1 / 2) :=
sorry

end range_m_for_p_range_m_for_q_range_m_for_not_p_or_q_l29_29978


namespace cos_double_angle_of_parallel_vectors_l29_29581

theorem cos_double_angle_of_parallel_vectors (α : ℝ) 
  (h_parallel : (1 / 3, Real.tan α) = (Real.cos α, 1)) : 
  Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end cos_double_angle_of_parallel_vectors_l29_29581


namespace john_must_work_10_more_days_l29_29465

theorem john_must_work_10_more_days
  (total_days : ℕ)
  (total_earnings : ℝ)
  (daily_earnings : ℝ)
  (target_earnings : ℝ)
  (additional_days : ℕ) :
  total_days = 10 →
  total_earnings = 250 →
  daily_earnings = total_earnings / total_days →
  target_earnings = 2 * total_earnings →
  additional_days = (target_earnings - total_earnings) / daily_earnings →
  additional_days = 10 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end john_must_work_10_more_days_l29_29465


namespace fourth_power_sum_l29_29057

variable (a b c : ℝ)

theorem fourth_power_sum (h1 : a + b + c = 2) 
                         (h2 : a^2 + b^2 + c^2 = 3) 
                         (h3 : a^3 + b^3 + c^3 = 4) : 
                         a^4 + b^4 + c^4 = 41 / 6 := 
by 
  sorry

end fourth_power_sum_l29_29057


namespace circle_intersection_range_l29_29949

noncomputable def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 = 25
noncomputable def circle2_eq (x y r : ℝ) : Prop := (x - 7)^2 + y^2 = r^2

theorem circle_intersection_range (r : ℝ) (h : r > 0) :
  (∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y r) ↔ 2 < r ∧ r < 12 :=
sorry

end circle_intersection_range_l29_29949


namespace smallest_not_odd_unit_is_zero_l29_29799

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l29_29799


namespace product_of_positive_c_for_rational_solutions_l29_29159

theorem product_of_positive_c_for_rational_solutions : 
  (∃ c₁ c₂ : ℕ, c₁ > 0 ∧ c₂ > 0 ∧ 
   (∀ x : ℝ, (3 * x ^ 2 + 7 * x + c₁ = 0) → ∃ r₁ r₂ : ℚ, x = r₁ ∨ x = r₂) ∧ 
   (∀ x : ℝ, (3 * x ^ 2 + 7 * x + c₂ = 0) → ∃ r₁ r₂ : ℚ, x = r₁ ∨ x = r₂) ∧ 
   c₁ * c₂ = 8) :=
sorry

end product_of_positive_c_for_rational_solutions_l29_29159


namespace compute_x_y_power_sum_l29_29204

noncomputable def pi : ℝ := Real.pi

theorem compute_x_y_power_sum
  (x y : ℝ)
  (h1 : 1 < x)
  (h2 : 1 < y)
  (h3 : (Real.log x / Real.log 2)^5 + (Real.log y / Real.log 3)^5 + 32 = 16 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^pi + y^pi = 2^(pi * (16:ℝ)^(1/5)) + 3^(pi * (16:ℝ)^(1/5)) :=
by
  sorry

end compute_x_y_power_sum_l29_29204


namespace Sam_has_correct_amount_of_dimes_l29_29335

-- Definitions for initial values and transactions
def initial_dimes := 9
def dimes_from_dad := 7
def dimes_taken_by_mom := 3
def sets_from_sister := 4
def dimes_per_set := 2

-- Definition of the total dimes Sam has now
def total_dimes_now : Nat :=
  initial_dimes + dimes_from_dad - dimes_taken_by_mom + (sets_from_sister * dimes_per_set)

-- Proof statement
theorem Sam_has_correct_amount_of_dimes : total_dimes_now = 21 := by
  sorry

end Sam_has_correct_amount_of_dimes_l29_29335


namespace find_b_in_triangle_l29_29196

-- Given conditions
variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : a = 3)
variable (h2 : c = 2 * Real.sqrt 3)
variable (h3 : b * Real.sin A = a * Real.cos (B + Real.pi / 6))

-- The proof goal
theorem find_b_in_triangle (h1 : a = 3) (h2 : c = 2 * Real.sqrt 3) (h3 : b * Real.sin A = a * Real.cos (B + Real.pi / 6)) : b = Real.sqrt 3 :=
sorry

end find_b_in_triangle_l29_29196


namespace find_25_percent_l29_29486

theorem find_25_percent (x : ℝ) (h : x - (3/4) * x = 100) : (1/4) * x = 100 :=
by
  sorry

end find_25_percent_l29_29486


namespace circle1_standard_form_circle2_standard_form_l29_29399

-- Define the first circle equation and its corresponding answer in standard form
theorem circle1_standard_form :
  ∀ x y : ℝ, (x^2 + y^2 + 2*x + 4*y - 4 = 0) ↔ ((x + 1)^2 + (y + 2)^2 = 9) :=
by
  intro x y
  sorry

-- Define the second circle equation and its corresponding answer in standard form
theorem circle2_standard_form :
  ∀ x y : ℝ, (3*x^2 + 3*y^2 + 6*x + 3*y - 15 = 0) ↔ ((x + 1)^2 + (y + 1/2)^2 = 25/4) :=
by
  intro x y
  sorry

end circle1_standard_form_circle2_standard_form_l29_29399


namespace square_in_semicircle_l29_29310

theorem square_in_semicircle (Q : ℝ) (h1 : ∃ Q : ℝ, (Q^2 / 4) + Q^2 = 4) : Q = 4 * Real.sqrt 5 / 5 := sorry

end square_in_semicircle_l29_29310


namespace smallest_digit_never_in_units_place_of_odd_number_l29_29791

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l29_29791


namespace arithmetic_sequence_a1_geometric_sequence_sum_l29_29628

-- Definition of the arithmetic sequence problem
theorem arithmetic_sequence_a1 (a_n s_n : ℕ) (d : ℕ) (h1 : a_n = 32) (h2 : s_n = 63) (h3 : d = 11) :
  ∃ a_1 : ℕ, a_1 = 10 :=
by
  sorry

-- Definition of the geometric sequence problem
theorem geometric_sequence_sum (a_1 q : ℕ) (h1 : a_1 = 1) (h2 : q = 2) (m : ℕ) :
  let a_m := a_1 * (q ^ (m - 1))
  let a_m_sq := a_m * a_m
  let sm'_sum := (1 - 4^m) / (1 - 4)
  sm'_sum = (4^m - 1) / 3 :=
by
  sorry

end arithmetic_sequence_a1_geometric_sequence_sum_l29_29628


namespace geometric_progression_sum_of_cubes_l29_29701

theorem geometric_progression_sum_of_cubes :
  ∃ (a r : ℕ) (seq : Fin 6 → ℕ), (seq 0 = a) ∧ (seq 1 = a * r) ∧ (seq 2 = a * r^2) ∧ (seq 3 = a * r^3) ∧ (seq 4 = a * r^4) ∧ (seq 5 = a * r^5) ∧
  (∀ i, 0 ≤ seq i ∧ seq i < 100) ∧
  (seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5 = 326) ∧
  (∃ T : ℕ, (∀ i, ∃ k, seq i = k^3 → k * k * k = seq i) ∧ T = 64) :=
sorry

end geometric_progression_sum_of_cubes_l29_29701


namespace family_members_l29_29537

variable (p : ℝ) (i : ℝ) (c : ℝ)

theorem family_members (h1 : p = 1.6) (h2 : i = 0.25) (h3 : c = 16) :
  (c / (2 * (p * (1 + i)))) = 4 := by
  sorry

end family_members_l29_29537


namespace max_value_of_symmetric_f_l29_29721

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_f :
  ∀ (a b : ℝ),
    (f 1 a b = 0) →
    (f (-1) a b = 0) →
    (f (-5) a b = 0) →
    (f (-3) a b = 0) →
    (∃ x : ℝ, f x 8 15 = 16) :=
by
  sorry

end max_value_of_symmetric_f_l29_29721


namespace ratio_of_x_and_y_l29_29395

theorem ratio_of_x_and_y (x y : ℝ) (h : 0.80 * x = 0.20 * y) : x / y = 0.25 :=
by
  sorry

end ratio_of_x_and_y_l29_29395


namespace system_solutions_l29_29579

theorem system_solutions (x y b : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 7 * y = 3 * b) (hx : x = -1) : 
  b = -22 :=
by 
  sorry

end system_solutions_l29_29579


namespace score_calculation_l29_29339

theorem score_calculation (N : ℕ) (C : ℕ) (hN: 1 ≤ N ∧ N ≤ 20) (hC: 1 ≤ C) : 
  ∃ (score: ℕ), score = Nat.floor (N / C) :=
by sorry

end score_calculation_l29_29339


namespace find_triplets_l29_29559

theorem find_triplets (a b p : ℕ) (ha : 0 < a) (hb : 0 < b) (hp : Nat.Prime p) (h_eq : (a + b)^p = p^a + p^b) : (a = 1 ∧ b = 1 ∧ p = 2) :=
by
  sorry

end find_triplets_l29_29559


namespace jill_show_duration_l29_29597

theorem jill_show_duration :
  let first_show_duration := 30
  let second_show_duration := 4 * first_show_duration
  first_show_duration + second_show_duration = 150 :=
by
  let first_show_duration := 30
  let second_show_duration := 4 * first_show_duration
  have h1 : second_show_duration = 120 := by rfl
  have h2 : first_show_duration + second_show_duration = 150 := by rfl
  show first_show_duration + second_show_duration = 150 from h2

end jill_show_duration_l29_29597


namespace tenth_term_arithmetic_sequence_l29_29119

def a : ℚ := 2 / 3
def d : ℚ := 2 / 3

theorem tenth_term_arithmetic_sequence : 
  let a := 2 / 3
  let d := 2 / 3
  let n := 10
  a + (n - 1) * d = 20 / 3 := by
  sorry

end tenth_term_arithmetic_sequence_l29_29119


namespace minimum_value_is_4_l29_29601

noncomputable def minimum_value (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) : ℝ :=
  real.sqrt ((x^2 + y^2 + z^2) * (4 * x^2 + y^2 + z^2)) / (x * y * z)

theorem minimum_value_is_4 (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) : (minimum_value x y z h) = 4 :=
sorry

end minimum_value_is_4_l29_29601


namespace number_of_friends_l29_29903

def initial_candies : ℕ := 10
def additional_candies : ℕ := 4
def total_candies : ℕ := initial_candies + additional_candies
def candies_per_friend : ℕ := 2

theorem number_of_friends : total_candies / candies_per_friend = 7 :=
by
  sorry

end number_of_friends_l29_29903


namespace limit_r_as_m_approaches_zero_l29_29746

open Topology

-- Define L(m) based on the conditions
def L (m : ℝ) : ℝ := -Real.sqrt (m + 4)
def L_neg (m : ℝ) : ℝ := -Real.sqrt (-m + 4)

-- Define r based on the conditions
noncomputable def r (m : ℝ) : ℝ := (L_neg m - L m) / m

-- State the theorem to prove the limit
theorem limit_r_as_m_approaches_zero : tendsto r (nhds 0) (nhds (1 / 2)) :=
by
  sorry

end limit_r_as_m_approaches_zero_l29_29746


namespace bathroom_visits_time_l29_29965

variable (t_8 : ℕ) (n8 : ℕ) (n6 : ℕ)

theorem bathroom_visits_time (h1 : t_8 = 20) (h2 : n8 = 8) (h3 : n6 = 6) :
  (t_8 / n8) * n6 = 15 := by
  sorry

end bathroom_visits_time_l29_29965


namespace smallest_digit_never_in_units_place_of_odd_numbers_l29_29784

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l29_29784
