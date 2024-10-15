import Mathlib

namespace NUMINAMATH_GPT_david_lewis_meeting_point_l1767_176727

theorem david_lewis_meeting_point :
  ∀ (D : ℝ),
  (∀ t : ℝ, t ≥ 0 →
    ∀ distance_to_meeting_point : ℝ, 
    distance_to_meeting_point = D →
    ∀ speed_david speed_lewis distance_cities : ℝ,
    speed_david = 50 →
    speed_lewis = 70 →
    distance_cities = 350 →
    ((distance_cities + distance_to_meeting_point) / speed_lewis = distance_to_meeting_point / speed_david) →
    D = 145.83) :=
by
  intros D t ht distance_to_meeting_point h_distance speed_david speed_lewis distance_cities h_speed_david h_speed_lewis h_distance_cities h_meeting_time
  -- We need to prove D = 145.83 under the given conditions
  sorry

end NUMINAMATH_GPT_david_lewis_meeting_point_l1767_176727


namespace NUMINAMATH_GPT_line_through_intersection_perpendicular_l1767_176772

theorem line_through_intersection_perpendicular (x y : ℝ) :
  (2 * x - 3 * y + 10 = 0) ∧ (3 * x + 4 * y - 2 = 0) →
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ (a = 2) ∧ (b = 3) ∧ (c = -2) ∧ (3 * a + 2 * b = 0)) :=
by
  sorry

end NUMINAMATH_GPT_line_through_intersection_perpendicular_l1767_176772


namespace NUMINAMATH_GPT_possible_dimensions_of_plot_l1767_176765

theorem possible_dimensions_of_plot (x : ℕ) :
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ 1000 * a + 100 * a + 10 * b + b = x * (x + 1)) →
  x = 33 ∨ x = 66 ∨ x = 99 :=
sorry

end NUMINAMATH_GPT_possible_dimensions_of_plot_l1767_176765


namespace NUMINAMATH_GPT_mrs_evans_class_l1767_176779

def students_enrolled_in_class (S Q1 Q2 missing both: ℕ) : Prop :=
  25 = Q1 ∧ 22 = Q2 ∧ 5 = missing ∧ 22 = both → S = Q1 + Q2 - both + missing

theorem mrs_evans_class (S : ℕ) : students_enrolled_in_class S 25 22 5 22 :=
by
  sorry

end NUMINAMATH_GPT_mrs_evans_class_l1767_176779


namespace NUMINAMATH_GPT_range_of_a_l1767_176701

theorem range_of_a {A B : Set ℝ} (hA : A = {x | x > 5}) (hB : B = {x | x > a}) 
  (h_sufficient_not_necessary : A ⊆ B ∧ ¬(B ⊆ A)) 
  : a < 5 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1767_176701


namespace NUMINAMATH_GPT_proof_problem_l1767_176738

variable {a : ℕ → ℝ} -- sequence a
variable {S : ℕ → ℝ} -- partial sums sequence S 
variable {n : ℕ} -- index

-- Define the conditions
def is_arith_seq (a : ℕ → ℝ) : Prop := 
  ∃ d, ∀ n, a (n+1) = a n + d

def S_is_partial_sum (a S : ℕ → ℝ) : Prop := 
  ∀ n, S (n+1) = S n + a (n+1)

-- The properties given in the problem
def conditions (a S : ℕ → ℝ) : Prop :=
  is_arith_seq a ∧ 
  S_is_partial_sum a S ∧ 
  S 6 < S 7 ∧ 
  S 7 > S 8

-- The conclusions that need to be proved
theorem proof_problem (a S : ℕ → ℝ) (h : conditions a S) : 
  S 9 < S 6 ∧
  (∀ n, a 1 ≥ a (n+1)) ∧
  (∀ m, S 7 ≥ S m) := by 
  sorry

end NUMINAMATH_GPT_proof_problem_l1767_176738


namespace NUMINAMATH_GPT_gas_pipe_probability_l1767_176760

-- Define the problem statement in Lean.
theorem gas_pipe_probability :
  let total_area := 400 * 400 / 2
  let usable_area := (300 - 100) * (300 - 100) / 2
  usable_area / total_area = 1 / 4 :=
by
  -- Sorry will be placeholder for the proof
  sorry

end NUMINAMATH_GPT_gas_pipe_probability_l1767_176760


namespace NUMINAMATH_GPT_average_loss_l1767_176780

theorem average_loss (cost_per_lootbox : ℝ) (average_value_per_lootbox : ℝ) (total_spent : ℝ)
                      (h1 : cost_per_lootbox = 5)
                      (h2 : average_value_per_lootbox = 3.5)
                      (h3 : total_spent = 40) :
  (total_spent - (average_value_per_lootbox * (total_spent / cost_per_lootbox))) = 12 :=
by
  sorry

end NUMINAMATH_GPT_average_loss_l1767_176780


namespace NUMINAMATH_GPT_trackball_mice_count_l1767_176798

theorem trackball_mice_count
  (total_mice : ℕ)
  (wireless_fraction : ℕ)
  (optical_fraction : ℕ)
  (h_total : total_mice = 80)
  (h_wireless : wireless_fraction = total_mice / 2)
  (h_optical : optical_fraction = total_mice / 4) :
  total_mice - (wireless_fraction + optical_fraction) = 20 :=
sorry

end NUMINAMATH_GPT_trackball_mice_count_l1767_176798


namespace NUMINAMATH_GPT_f_3_eq_4_l1767_176757

noncomputable def f : ℝ → ℝ := sorry

theorem f_3_eq_4 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x^2) : f 3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_f_3_eq_4_l1767_176757


namespace NUMINAMATH_GPT_total_insects_eaten_l1767_176702

-- Definitions from the conditions
def numGeckos : Nat := 5
def insectsPerGecko : Nat := 6
def numLizards : Nat := 3
def insectsPerLizard : Nat := insectsPerGecko * 2

-- Theorem statement, proving total insects eaten is 66
theorem total_insects_eaten : numGeckos * insectsPerGecko + numLizards * insectsPerLizard = 66 := by
  sorry

end NUMINAMATH_GPT_total_insects_eaten_l1767_176702


namespace NUMINAMATH_GPT_minimum_abs_a_l1767_176795

-- Given conditions as definitions
def has_integer_coeffs (a b c : ℤ) : Prop := true
def has_roots_in_range (a b c : ℤ) (x1 x2 : ℚ) : Prop :=
  x1 ≠ x2 ∧ 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧
  (a : ℚ) * x1^2 + (b : ℚ) * x1 + (c : ℚ) = 0 ∧
  (a : ℚ) * x2^2 + (b : ℚ) * x2 + (c : ℚ) = 0

-- Main statement (abstractly mentioning existence of x1, x2 such that they fulfill the polynomial conditions)
theorem minimum_abs_a (a b c : ℤ) (x1 x2 : ℚ) :
  has_integer_coeffs a b c →
  has_roots_in_range a b c x1 x2 →
  |a| ≥ 5 :=
by
  intros _ _
  sorry

end NUMINAMATH_GPT_minimum_abs_a_l1767_176795


namespace NUMINAMATH_GPT_Doug_money_l1767_176781

theorem Doug_money (B D : ℝ) (h1 : B + 2*B + D = 68) (h2 : 2*B = (3/4)*D) : D = 32 := by
  sorry

end NUMINAMATH_GPT_Doug_money_l1767_176781


namespace NUMINAMATH_GPT_smallest_n_solution_unique_l1767_176708

theorem smallest_n_solution_unique (a b c d : ℤ) (h : a^2 + b^2 + c^2 = 4 * d^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end NUMINAMATH_GPT_smallest_n_solution_unique_l1767_176708


namespace NUMINAMATH_GPT_train_speed_l1767_176721

noncomputable def speed_of_each_train (v : ℕ) : ℕ := 27

theorem train_speed
  (length_of_each_train : ℕ)
  (crossing_time : ℕ)
  (crossing_condition : 2 * (length_of_each_train * crossing_time) / (2 * crossing_time) = 15 / 2)
  (conversion_factor : ∀ n, 1 = 3.6 * n → ℕ) :
  speed_of_each_train 27 = 27 :=
by
  exact rfl

end NUMINAMATH_GPT_train_speed_l1767_176721


namespace NUMINAMATH_GPT_cistern_fill_time_l1767_176744

variable (C : ℝ) -- Volume of the cistern
variable (X Y Z : ℝ) -- Rates at which pipes X, Y, and Z fill the cistern

-- Pipes X and Y together, pipes X and Z together, and pipes Y and Z together conditions
def condition1 := X + Y = C / 3
def condition2 := X + Z = C / 4
def condition3 := Y + Z = C / 5

theorem cistern_fill_time (h1 : condition1 C X Y) (h2 : condition2 C X Z) (h3 : condition3 C Y Z) :
  1 / (X + Y + Z) = 120 / 47 :=
by
  sorry

end NUMINAMATH_GPT_cistern_fill_time_l1767_176744


namespace NUMINAMATH_GPT_find_x_plus_y_l1767_176759

theorem find_x_plus_y (x y : ℝ) (hx : |x| + x + y = 14) (hy : x + |y| - y = 16) : x + y = 26 / 5 := 
sorry

end NUMINAMATH_GPT_find_x_plus_y_l1767_176759


namespace NUMINAMATH_GPT_roots_equal_implies_a_eq_3_l1767_176718

theorem roots_equal_implies_a_eq_3 (x a : ℝ) (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
sorry

end NUMINAMATH_GPT_roots_equal_implies_a_eq_3_l1767_176718


namespace NUMINAMATH_GPT_necessary_condition_of_equilateral_triangle_l1767_176766

variable {A B C: ℝ}
variable {a b c: ℝ}

theorem necessary_condition_of_equilateral_triangle
  (h1 : B + C = 2 * A)
  (h2 : b + c = 2 * a)
  : (A = B ∧ B = C ∧ a = b ∧ b = c) ↔ (B + C = 2 * A ∧ b + c = 2 * a) := 
by
  sorry

end NUMINAMATH_GPT_necessary_condition_of_equilateral_triangle_l1767_176766


namespace NUMINAMATH_GPT_remainder_2456789_div_7_l1767_176712

theorem remainder_2456789_div_7 :
  2456789 % 7 = 6 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_2456789_div_7_l1767_176712


namespace NUMINAMATH_GPT_smallest_possible_positive_value_l1767_176729

theorem smallest_possible_positive_value (l w : ℕ) (hl : l > 0) (hw : w > 0) : ∃ x : ℕ, x = w - l + 1 ∧ x = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_possible_positive_value_l1767_176729


namespace NUMINAMATH_GPT_hyperbola_inequality_l1767_176713

-- Define point P on the hyperbola in terms of a and b
theorem hyperbola_inequality (a b : ℝ) (h : (3*a + 3*b)^2 / 9 - (a - b)^2 = 1) : |a + b| ≥ 1 :=
sorry

end NUMINAMATH_GPT_hyperbola_inequality_l1767_176713


namespace NUMINAMATH_GPT_avg_age_new_students_l1767_176747

theorem avg_age_new_students :
  ∀ (O A_old A_new_avg : ℕ) (A_new : ℕ),
    O = 12 ∧ A_old = 40 ∧ A_new_avg = (A_old - 4) ∧ A_new_avg = 36 →
    A_new * 12 = (24 * A_new_avg) - (O * A_old) →
    A_new = 32 :=
by
  intros O A_old A_new_avg A_new
  intro h
  rcases h with ⟨hO, hA_old, hA_new_avg, h36⟩
  sorry

end NUMINAMATH_GPT_avg_age_new_students_l1767_176747


namespace NUMINAMATH_GPT_no_intersection_range_k_l1767_176748

def problem_statement (k : ℝ) : Prop :=
  ∀ (x : ℝ),
    ¬(x > 1 ∧ x + 1 = k * x + 2) ∧ ¬(x < 1 ∧ -x - 1 = k * x + 2) ∧ 
    (x = 1 → (x + 1 ≠ k * x + 2 ∧ -x - 1 ≠ k * x + 2))

theorem no_intersection_range_k :
  ∀ (k : ℝ), problem_statement k ↔ -4 ≤ k ∧ k < -1 :=
sorry

end NUMINAMATH_GPT_no_intersection_range_k_l1767_176748


namespace NUMINAMATH_GPT_initial_books_l1767_176743

theorem initial_books (added_books : ℝ) (books_per_shelf : ℝ) (shelves : ℝ) 
  (total_books : ℝ) : total_books = shelves * books_per_shelf → 
  shelves = 14 → books_per_shelf = 4.0 → added_books = 10.0 → 
  total_books - added_books = 46.0 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_initial_books_l1767_176743


namespace NUMINAMATH_GPT_part1_part2_l1767_176745

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end NUMINAMATH_GPT_part1_part2_l1767_176745


namespace NUMINAMATH_GPT_problem_number_of_true_propositions_l1767_176735

open Set

variable {α : Type*} {A B : Set α}

def card (s : Set α) : ℕ := sorry -- The actual definition of cardinality is complex and in LF (not imperative here).

-- Statement of the problem translated into a Lean statement
theorem problem_number_of_true_propositions :
  (∀ {A B : Set ℕ}, A ∩ B = ∅ ↔ card (A ∪ B) = card A + card B) ∧
  (∀ {A B : Set ℕ}, A ⊆ B → card A ≤ card B) ∧
  (∀ {A B : Set ℕ}, A ⊂ B → card A < card B) →
   (3 = 3) :=
by 
  sorry


end NUMINAMATH_GPT_problem_number_of_true_propositions_l1767_176735


namespace NUMINAMATH_GPT_monotonicity_m_eq_zero_range_of_m_l1767_176787

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.exp x - m * x^2 - 2 * x

theorem monotonicity_m_eq_zero :
  ∀ x : ℝ, (x < Real.log 2 → f x 0 < f (x + 1) 0) ∧ (x > Real.log 2 → f x 0 > f (x - 1) 0) := 
sorry

theorem range_of_m :
  ∀ x : ℝ, x ∈ Set.Ici 0 → f x m > (Real.exp 1 / 2 - 1) → m < (Real.exp 1 / 2 - 1) := 
sorry

end NUMINAMATH_GPT_monotonicity_m_eq_zero_range_of_m_l1767_176787


namespace NUMINAMATH_GPT_giuseppe_can_cut_rectangles_l1767_176741

theorem giuseppe_can_cut_rectangles : 
  let board_length := 22
  let board_width := 15
  let rectangle_length := 3
  let rectangle_width := 5
  (board_length * board_width) / (rectangle_length * rectangle_width) = 22 :=
by
  sorry

end NUMINAMATH_GPT_giuseppe_can_cut_rectangles_l1767_176741


namespace NUMINAMATH_GPT_total_price_paid_l1767_176742

noncomputable def total_price
    (price_rose : ℝ) (qty_rose : ℕ) (discount_rose : ℝ)
    (price_lily : ℝ) (qty_lily : ℕ) (discount_lily : ℝ)
    (price_sunflower : ℝ) (qty_sunflower : ℕ)
    (store_discount : ℝ) (tax_rate : ℝ)
    : ℝ :=
  let total_rose := qty_rose * price_rose
  let total_lily := qty_lily * price_lily
  let total_sunflower := qty_sunflower * price_sunflower
  let total := total_rose + total_lily + total_sunflower
  let total_disc_rose := total_rose * discount_rose
  let total_disc_lily := total_lily * discount_lily
  let discounted_total := total - total_disc_rose - total_disc_lily
  let store_discount_amount := discounted_total * store_discount
  let after_store_discount := discounted_total - store_discount_amount
  let tax_amount := after_store_discount * tax_rate
  after_store_discount + tax_amount

theorem total_price_paid :
  total_price 20 3 0.15 15 5 0.10 10 2 0.05 0.07 = 140.79 :=
by
  apply sorry

end NUMINAMATH_GPT_total_price_paid_l1767_176742


namespace NUMINAMATH_GPT_value_of_f_f_2_l1767_176752

def f (x : ℝ) : ℝ := 2 * x^3 - 4 * x^2 + 3 * x - 1

theorem value_of_f_f_2 : f (f 2) = 164 := by
  sorry

end NUMINAMATH_GPT_value_of_f_f_2_l1767_176752


namespace NUMINAMATH_GPT_percent_not_filler_l1767_176736

theorem percent_not_filler (sandwich_weight filler_weight : ℕ) (h_sandwich : sandwich_weight = 180) (h_filler : filler_weight = 45) : 
  (sandwich_weight - filler_weight) * 100 / sandwich_weight = 75 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_percent_not_filler_l1767_176736


namespace NUMINAMATH_GPT_haylee_has_36_guppies_l1767_176790

variables (H J C N : ℝ)
variables (total_guppies : ℝ := 84)

def jose_has_half_of_haylee := J = H / 2
def charliz_has_third_of_jose := C = J / 3
def nicolai_has_four_times_charliz := N = 4 * C
def total_guppies_eq_84 := H + J + C + N = total_guppies

theorem haylee_has_36_guppies 
  (hJ : jose_has_half_of_haylee H J)
  (hC : charliz_has_third_of_jose J C)
  (hN : nicolai_has_four_times_charliz C N)
  (htotal : total_guppies_eq_84 H J C N) :
  H = 36 := 
  sorry

end NUMINAMATH_GPT_haylee_has_36_guppies_l1767_176790


namespace NUMINAMATH_GPT_smallest_sum_is_minus_half_l1767_176728

def smallest_sum (x: ℝ) : ℝ := x^2 + x

theorem smallest_sum_is_minus_half : ∃ x : ℝ, ∀ y : ℝ, smallest_sum y ≥ smallest_sum (-1/2) :=
by
  use -1/2
  intros y
  sorry

end NUMINAMATH_GPT_smallest_sum_is_minus_half_l1767_176728


namespace NUMINAMATH_GPT_james_speed_downhill_l1767_176731

theorem james_speed_downhill (T1 T2 v : ℝ) (h1 : T1 = 20 / v) (h2 : T2 = 12 / 3 + 1) (h3 : T1 = T2 - 1) : v = 5 :=
by
  -- Declare variables
  have hT2 : T2 = 5 := by linarith
  have hT1 : T1 = 4 := by linarith
  have hv : v = 20 / 4 := by sorry
  linarith

#exit

end NUMINAMATH_GPT_james_speed_downhill_l1767_176731


namespace NUMINAMATH_GPT_find_least_multiple_of_50_l1767_176770

def digits (n : ℕ) : List ℕ := n.digits 10

def product_of_digits (n : ℕ) : ℕ := (digits n).prod

theorem find_least_multiple_of_50 :
  ∃ n, (n % 50 = 0) ∧ ((product_of_digits n) % 50 = 0) ∧ (∀ m, (m % 50 = 0) ∧ ((product_of_digits m) % 50 = 0) → n ≤ m) ↔ n = 5550 :=
by sorry

end NUMINAMATH_GPT_find_least_multiple_of_50_l1767_176770


namespace NUMINAMATH_GPT_simplify_condition_l1767_176711

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  Real.sqrt (1 + x) - Real.sqrt (-1 - x)

theorem simplify_condition (x : ℝ) (h1 : 1 + x ≥ 0) (h2 : -1 - x ≥ 0) : simplify_expression x = 0 :=
by
  rw [simplify_expression]
  sorry

end NUMINAMATH_GPT_simplify_condition_l1767_176711


namespace NUMINAMATH_GPT_inequality_solution_l1767_176739

theorem inequality_solution (x : ℝ) (hx : 0 ≤ x ∧ x < 2) :
  ∀ y : ℝ, y > 0 → 4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y :=
by
  intro y hy
  sorry

end NUMINAMATH_GPT_inequality_solution_l1767_176739


namespace NUMINAMATH_GPT_value_of_m_l1767_176726
noncomputable def y (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^(m^2 - 3)

theorem value_of_m (m : ℝ) (x : ℝ) (h1 : x > 0) (h2 : ∀ x1 x2 : ℝ, x1 > x2 → y m x1 < y m x2) :
  m = 2 :=
sorry

end NUMINAMATH_GPT_value_of_m_l1767_176726


namespace NUMINAMATH_GPT_arrangement_problem_l1767_176717

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrangement_problem 
  (p1 p2 p3 p4 p5 : Type)  -- Representing the five people
  (youngest : p1)         -- Specifying the youngest
  (oldest : p5)           -- Specifying the oldest
  (unique_people : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5) -- Ensuring five unique people
  : (factorial 5) - (factorial 4 * 2) = 72 :=
by sorry

end NUMINAMATH_GPT_arrangement_problem_l1767_176717


namespace NUMINAMATH_GPT_complement_of_A_l1767_176716

variables (U : Set ℝ) (A : Set ℝ)
def universal_set : Prop := U = Set.univ
def range_of_function : Prop := A = {x : ℝ | 0 ≤ x}

theorem complement_of_A (hU : universal_set U) (hA : range_of_function A) : 
  U \ A = {x : ℝ | x < 0} :=
by 
  sorry

end NUMINAMATH_GPT_complement_of_A_l1767_176716


namespace NUMINAMATH_GPT_largest_integer_sol_l1767_176767

theorem largest_integer_sol (x : ℤ) : (3 * x + 4 < 5 * x - 2) -> x = 3 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_sol_l1767_176767


namespace NUMINAMATH_GPT_distance_from_A_to_C_correct_total_distance_traveled_correct_l1767_176763

-- Define the conditions
def distance_to_A : ℕ := 30
def distance_to_B : ℕ := 20
def distance_to_C : ℤ := -15
def times_to_C : ℕ := 3

-- Define the resulting calculated distances based on the conditions
def distance_A_to_C : ℕ := distance_to_A + distance_to_C.natAbs
def total_distance_traveled : ℕ := (distance_to_A + distance_to_B) * 2 + distance_to_C.natAbs * (times_to_C * 2)

-- The proof problems (statements) based on the problem's questions
theorem distance_from_A_to_C_correct : distance_A_to_C = 45 := by
  sorry

theorem total_distance_traveled_correct : total_distance_traveled = 190 := by
  sorry

end NUMINAMATH_GPT_distance_from_A_to_C_correct_total_distance_traveled_correct_l1767_176763


namespace NUMINAMATH_GPT_candy_partition_l1767_176709

theorem candy_partition :
  let candies := 10
  let boxes := 3
  ∃ ways : ℕ, ways = Nat.choose (candies + boxes - 1) (boxes - 1) ∧ ways = 66 :=
by
  let candies := 10
  let boxes := 3
  let ways := Nat.choose (candies + boxes - 1) (boxes - 1)
  have h : ways = 66 := sorry
  exact ⟨ways, ⟨rfl, h⟩⟩

end NUMINAMATH_GPT_candy_partition_l1767_176709


namespace NUMINAMATH_GPT_haley_fuel_consumption_ratio_l1767_176791

theorem haley_fuel_consumption_ratio (gallons: ℕ) (miles: ℕ) (h_gallons: gallons = 44) (h_miles: miles = 77) :
  (gallons / Nat.gcd gallons miles) = 4 ∧ (miles / Nat.gcd gallons miles) = 7 :=
by
  sorry

end NUMINAMATH_GPT_haley_fuel_consumption_ratio_l1767_176791


namespace NUMINAMATH_GPT_JackOfHeartsIsSane_l1767_176799

inductive Card
  | Ace
  | Two
  | Three
  | Four
  | Five
  | Six
  | Seven
  | JackOfHearts

open Card

def Sane (c : Card) : Prop := sorry

axiom Condition1 : Sane Three → ¬ Sane Ace
axiom Condition2 : Sane Four → (¬ Sane Three ∨ ¬ Sane Two)
axiom Condition3 : Sane Five → (Sane Ace ↔ Sane Four)
axiom Condition4 : Sane Six → (Sane Ace ∧ Sane Two)
axiom Condition5 : Sane Seven → ¬ Sane Five
axiom Condition6 : Sane JackOfHearts → (¬ Sane Six ∨ ¬ Sane Seven)

theorem JackOfHeartsIsSane : Sane JackOfHearts := by
  sorry

end NUMINAMATH_GPT_JackOfHeartsIsSane_l1767_176799


namespace NUMINAMATH_GPT_net_gain_loss_l1767_176794

-- Definitions of the initial conditions
structure InitialState :=
  (cash_x : ℕ) (painting_value : ℕ) (cash_y : ℕ)

-- Definitions of transactions
structure Transaction :=
  (sell_price : ℕ) (commission_rate : ℕ)

def apply_transaction (initial_cash : ℕ) (tr : Transaction) : ℕ :=
  initial_cash + (tr.sell_price - (tr.sell_price * tr.commission_rate / 100))

def revert_transaction (initial_cash : ℕ) (tr : Transaction) : ℕ :=
  initial_cash - tr.sell_price + (tr.sell_price * tr.commission_rate / 100)

def compute_final_cash (initial_states : InitialState) (trans1 : Transaction) (trans2 : Transaction) : ℕ :=
  let cash_x_after_first := apply_transaction initial_states.cash_x trans1
  let cash_y_after_first := initial_states.cash_y - trans1.sell_price
  let cash_x_after_second := revert_transaction cash_x_after_first trans2
  let cash_y_after_second := cash_y_after_first + (trans2.sell_price - (trans2.sell_price * trans2.commission_rate / 100))
  cash_x_after_second - initial_states.cash_x + (cash_y_after_second - initial_states.cash_y)

-- Statement of the theorem
theorem net_gain_loss (initial_states : InitialState) (trans1 : Transaction) (trans2 : Transaction)
  (h1 : initial_states.cash_x = 15000)
  (h2 : initial_states.painting_value = 15000)
  (h3 : initial_states.cash_y = 18000)
  (h4 : trans1.sell_price = 20000)
  (h5 : trans1.commission_rate = 5)
  (h6 : trans2.sell_price = 14000)
  (h7 : trans2.commission_rate = 5) : 
  compute_final_cash initial_states trans1 trans2 = 5000 - 6700 :=
sorry

end NUMINAMATH_GPT_net_gain_loss_l1767_176794


namespace NUMINAMATH_GPT_bicycle_cost_price_l1767_176783

theorem bicycle_cost_price (CP_A : ℝ) 
    (h1 : ∀ SP_B, SP_B = 1.20 * CP_A)
    (h2 : ∀ CP_C SP_B, CP_C = 1.40 * SP_B ∧ SP_B = 1.20 * CP_A)
    (h3 : ∀ SP_D CP_C, SP_D = 1.30 * CP_C ∧ CP_C = 1.40 * 1.20 * CP_A)
    (h4 : ∀ SP_D', SP_D' = 350 / 0.90) :
    CP_A = 350 / 1.9626 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_cost_price_l1767_176783


namespace NUMINAMATH_GPT_area_of_region_bounded_by_circle_l1767_176775

theorem area_of_region_bounded_by_circle :
  (∃ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 9 = 0) →
  ∃ (area : ℝ), area = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_region_bounded_by_circle_l1767_176775


namespace NUMINAMATH_GPT_percentage_sum_l1767_176785

theorem percentage_sum {A B : ℝ} 
  (hA : 0.40 * A = 160) 
  (hB : (2/3) * B = 160) : 
  0.60 * (A + B) = 384 :=
by
  sorry

end NUMINAMATH_GPT_percentage_sum_l1767_176785


namespace NUMINAMATH_GPT_max_blue_points_l1767_176704

theorem max_blue_points (n : ℕ) (h_n : n = 2016) :
  ∃ r : ℕ, r * (2016 - r) = 1008 * 1008 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_blue_points_l1767_176704


namespace NUMINAMATH_GPT_range_of_a_l1767_176754

noncomputable def p (a : ℝ) : Prop := 
  (1 + a)^2 + (1 - a)^2 < 4

noncomputable def q (a : ℝ) : Prop := 
  ∀ x : ℝ, x^2 + a * x + 1 ≥ 0

theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) ↔ (-2 ≤ a ∧ a ≤ -1) ∨ (1 ≤ a ∧ a ≤ 2) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1767_176754


namespace NUMINAMATH_GPT_fraction_simplification_l1767_176705

theorem fraction_simplification 
  (d e f : ℝ) 
  (h : d + e + f ≠ 0) : 
  (d^2 + e^2 - f^2 + 2 * d * e) / (d^2 + f^2 - e^2 + 3 * d * f) = (d + e - f) / (d + f - e) :=
sorry

end NUMINAMATH_GPT_fraction_simplification_l1767_176705


namespace NUMINAMATH_GPT_rational_solution_quadratic_l1767_176751

theorem rational_solution_quadratic (m : ℕ) (h_pos : m > 0) : 
  (∃ (x : ℚ), x * x * m + 25 * x + m = 0) ↔ m = 10 ∨ m = 12 :=
by sorry

end NUMINAMATH_GPT_rational_solution_quadratic_l1767_176751


namespace NUMINAMATH_GPT_find_angle_l1767_176755

theorem find_angle (a b c d e : ℝ) (sum_of_hexagon_angles : ℝ) (h_sum : a = 135 ∧ b = 120 ∧ c = 105 ∧ d = 150 ∧ e = 110 ∧ sum_of_hexagon_angles = 720) : 
  ∃ P : ℝ, a + b + c + d + e + P = sum_of_hexagon_angles ∧ P = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_l1767_176755


namespace NUMINAMATH_GPT_smallest_relatively_prime_210_l1767_176707

theorem smallest_relatively_prime_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ (∀ y : ℕ, y > 1 → y < x → Nat.gcd y 210 ≠ 1) :=
sorry

end NUMINAMATH_GPT_smallest_relatively_prime_210_l1767_176707


namespace NUMINAMATH_GPT_store_discount_percentage_l1767_176771

theorem store_discount_percentage
  (total_without_discount : ℝ := 350)
  (final_price : ℝ := 252)
  (coupon_percentage : ℝ := 0.1) :
  ∃ (x : ℝ), total_without_discount * (1 - x / 100) * (1 - coupon_percentage) = final_price ∧ x = 20 :=
by
  use 20
  sorry

end NUMINAMATH_GPT_store_discount_percentage_l1767_176771


namespace NUMINAMATH_GPT_choir_members_unique_l1767_176788

theorem choir_members_unique (n : ℕ) :
  (n % 10 = 6) ∧ 
  (n % 11 = 6) ∧ 
  (150 ≤ n) ∧ 
  (n ≤ 300) → 
  n = 226 := 
by
  sorry

end NUMINAMATH_GPT_choir_members_unique_l1767_176788


namespace NUMINAMATH_GPT_arithmetic_sequence_sixth_term_l1767_176732

variables (a d : ℤ)

theorem arithmetic_sequence_sixth_term :
  a + (a + d) + (a + 2 * d) = 12 →
  a + 3 * d = 0 →
  a + 5 * d = -4 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sixth_term_l1767_176732


namespace NUMINAMATH_GPT_time_to_carry_backpack_l1767_176778

/-- 
Given:
1. Lara takes 73 seconds to crank open the door to the obstacle course.
2. Lara traverses the obstacle course the second time in 5 minutes and 58 seconds.
3. The total time to complete the obstacle course is 874 seconds.

Prove:
The time it took Lara to carry the backpack through the obstacle course the first time is 443 seconds.
-/
theorem time_to_carry_backpack (door_time : ℕ) (second_traversal_time : ℕ) (total_time : ℕ) : 
  (door_time + second_traversal_time + 443 = total_time) :=
by
  -- Given conditions
  let door_time := 73
  let second_traversal_time := 5 * 60 + 58 -- Convert 5 minutes 58 seconds to seconds
  let total_time := 874
  -- Calculate the time to carry the backpack
  sorry

end NUMINAMATH_GPT_time_to_carry_backpack_l1767_176778


namespace NUMINAMATH_GPT_third_place_prize_correct_l1767_176750

-- Define the conditions and formulate the problem
def total_amount_in_pot : ℝ := 210
def third_place_percentage : ℝ := 0.15
def third_place_prize (P : ℝ) : ℝ := third_place_percentage * P

-- The theorem to be proved
theorem third_place_prize_correct : 
  third_place_prize total_amount_in_pot = 31.5 := 
by
  sorry

end NUMINAMATH_GPT_third_place_prize_correct_l1767_176750


namespace NUMINAMATH_GPT_average_infections_per_round_infections_after_three_rounds_l1767_176719

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

end NUMINAMATH_GPT_average_infections_per_round_infections_after_three_rounds_l1767_176719


namespace NUMINAMATH_GPT_height_of_shorter_pot_is_20_l1767_176776

-- Define the conditions as given
def height_of_taller_pot := 40
def shadow_of_taller_pot := 20
def shadow_of_shorter_pot := 10

-- Define the height of the shorter pot to be determined
def height_of_shorter_pot (h : ℝ) := h

-- Define the relationship using the concept of similar triangles
theorem height_of_shorter_pot_is_20 (h : ℝ) :
  (height_of_taller_pot / shadow_of_taller_pot = height_of_shorter_pot h / shadow_of_shorter_pot) → h = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_height_of_shorter_pot_is_20_l1767_176776


namespace NUMINAMATH_GPT_simple_interest_years_l1767_176797

variables (T R : ℝ)

def principal : ℝ := 1000
def additional_interest : ℝ := 90

theorem simple_interest_years
  (H: principal * (R + 3) * T / 100 - principal * R * T / 100 = additional_interest) :
  T = 3 :=
by sorry

end NUMINAMATH_GPT_simple_interest_years_l1767_176797


namespace NUMINAMATH_GPT_novels_in_shipment_l1767_176720

theorem novels_in_shipment (N : ℕ) (H1: 225 = (3/4:ℚ) * N) : N = 300 := 
by
  sorry

end NUMINAMATH_GPT_novels_in_shipment_l1767_176720


namespace NUMINAMATH_GPT_find_cows_l1767_176756

theorem find_cows :
  ∃ (D C : ℕ), (2 * D + 4 * C = 2 * (D + C) + 30) → C = 15 := 
sorry

end NUMINAMATH_GPT_find_cows_l1767_176756


namespace NUMINAMATH_GPT_factorization_correct_l1767_176774

-- Define noncomputable to deal with the natural arithmetic operations
noncomputable def a : ℕ := 66
noncomputable def b : ℕ := 231

-- Define the given expressions
noncomputable def lhs (x : ℕ) : ℤ := ((a : ℤ) * x^6) - ((b : ℤ) * x^12)
noncomputable def rhs (x : ℕ) : ℤ := (33 : ℤ) * x^6 * (2 - 7 * x^6)

-- The theorem to prove the equality
theorem factorization_correct (x : ℕ) : lhs x = rhs x :=
by sorry

end NUMINAMATH_GPT_factorization_correct_l1767_176774


namespace NUMINAMATH_GPT_marathon_distance_l1767_176789

theorem marathon_distance (d_1 : ℕ) (n : ℕ) (h1 : d_1 = 3) (h2 : n = 5): 
  (2 ^ (n - 1)) * d_1 = 48 :=
by
  sorry

end NUMINAMATH_GPT_marathon_distance_l1767_176789


namespace NUMINAMATH_GPT_not_right_triangle_D_right_triangle_A_right_triangle_B_right_triangle_C_l1767_176730

def right_angle_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem not_right_triangle_D (a b c : ℝ):
  ¬ (a = 3^2 ∧ b = 4^2 ∧ c = 5^2 ∧ right_angle_triangle a b c) :=
sorry

theorem right_triangle_A (a b c x : ℝ):
  a = 5 * x → b = 12 * x → c = 13 * x → x > 0 → right_angle_triangle a b c :=
sorry

theorem right_triangle_B (angleA angleB angleC : ℝ):
  angleA / angleB / angleC = 2 / 3 / 5 → angleC = 90 → angleA + angleB + angleC = 180 → right_angle_triangle angleA angleB angleC :=
sorry

theorem right_triangle_C (a b c k : ℝ):
  a = 9 * k → b = 40 * k → c = 41 * k → k > 0 → right_angle_triangle a b c :=
sorry

end NUMINAMATH_GPT_not_right_triangle_D_right_triangle_A_right_triangle_B_right_triangle_C_l1767_176730


namespace NUMINAMATH_GPT_max_m_value_l1767_176715

theorem max_m_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a + b + c = 12) (h_prod_sum : a * b + b * c + c * a = 35) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m = 3 :=
by
  sorry

end NUMINAMATH_GPT_max_m_value_l1767_176715


namespace NUMINAMATH_GPT_probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice_l1767_176723
-- Import all necessary libraries

-- Define the conditions as variables
variable (n k : ℕ) (p q : ℚ)
variable (dice_divisible_by_3_prob : ℚ)
variable (dice_not_divisible_by_3_prob : ℚ)

-- Assign values based on the problem statement
noncomputable def cond_replicate_n_fair_12_sided_dice := n = 7
noncomputable def cond_exactly_k_divisible_by_3 := k = 3
noncomputable def cond_prob_divisible_by_3 := dice_divisible_by_3_prob = 1 / 3
noncomputable def cond_prob_not_divisible_by_3 := dice_not_divisible_by_3_prob = 2 / 3

-- The theorem statement with the final answer incorporated
theorem probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice :
  cond_replicate_n_fair_12_sided_dice n →
  cond_exactly_k_divisible_by_3 k →
  cond_prob_divisible_by_3 dice_divisible_by_3_prob →
  cond_prob_not_divisible_by_3 dice_not_divisible_by_3_prob →
  p = (35 : ℚ) * ((1 / 3) ^ 3) * ((2 / 3) ^ 4) →
  q = (560 / 2187 : ℚ) →
  p = q :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice_l1767_176723


namespace NUMINAMATH_GPT_paula_cans_used_l1767_176764

/-- 
  Paula originally had enough paint to cover 42 rooms. 
  Unfortunately, she lost 4 cans of paint on her way, 
  and now she can only paint 34 rooms. 
  Prove the number of cans she used for these 34 rooms is 17.
-/
theorem paula_cans_used (R L P C : ℕ) (hR : R = 42) (hL : L = 4) (hP : P = 34)
    (hRooms : R - ((R - P) / L) * L = P) :
  C = 17 :=
by
  sorry

end NUMINAMATH_GPT_paula_cans_used_l1767_176764


namespace NUMINAMATH_GPT_problem_statement_l1767_176753

theorem problem_statement (a b : ℕ) (m n : ℕ)
  (h1 : 32 + (2 / 7 : ℝ) = 3 * (2 / 7 : ℝ))
  (h2 : 33 + (3 / 26 : ℝ) = 3 * (3 / 26 : ℝ))
  (h3 : 34 + (4 / 63 : ℝ) = 3 * (4 / 63 : ℝ))
  (h4 : 32014 + (m / n : ℝ) = 2014 * 3 * (m / n : ℝ))
  (h5 : 32016 + (a / b : ℝ) = 2016 * 3 * (a / b : ℝ)) :
  (b + 1) / (a * a) = 2016 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1767_176753


namespace NUMINAMATH_GPT_cost_expression_A_cost_expression_B_cost_comparison_10_students_cost_comparison_4_students_l1767_176769

-- Define the conditions
def ticket_full_price : ℕ := 240
def discount_A : ℕ := ticket_full_price / 2
def discount_B (x : ℕ) : ℕ := 144 * (x + 1)

-- Algebraic expressions provided in the answer
def cost_A (x : ℕ) : ℕ := discount_A * x + ticket_full_price
def cost_B (x : ℕ) : ℕ := 144 * (x + 1)

-- Proofs for the specific cases
theorem cost_expression_A (x : ℕ) : cost_A x = 120 * x + 240 := by
  sorry

theorem cost_expression_B (x : ℕ) : cost_B x = 144 * (x + 1) := by
  sorry

theorem cost_comparison_10_students : cost_A 10 < cost_B 10 := by
  sorry

theorem cost_comparison_4_students : cost_A 4 = cost_B 4 := by
  sorry

end NUMINAMATH_GPT_cost_expression_A_cost_expression_B_cost_comparison_10_students_cost_comparison_4_students_l1767_176769


namespace NUMINAMATH_GPT_find_number_l1767_176746

theorem find_number (x : ℝ) (h : 0.4 * x = 15) : x = 37.5 := by
  sorry

end NUMINAMATH_GPT_find_number_l1767_176746


namespace NUMINAMATH_GPT_exists_k_with_three_different_real_roots_exists_k_with_two_different_real_roots_l1767_176710

noncomputable def equation (x : ℝ) (k : ℝ) := x^2 - 2 * |x| - (2 * k + 1)^2

theorem exists_k_with_three_different_real_roots :
  ∃ k : ℝ, ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ equation x1 k = 0 ∧ equation x2 k = 0 ∧ equation x3 k = 0 :=
sorry

theorem exists_k_with_two_different_real_roots :
  ∃ k : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ equation x1 k = 0 ∧ equation x2 k = 0 :=
sorry

end NUMINAMATH_GPT_exists_k_with_three_different_real_roots_exists_k_with_two_different_real_roots_l1767_176710


namespace NUMINAMATH_GPT_total_people_in_house_l1767_176737

-- Define the number of people in various locations based on the given conditions.
def charlie_and_susan := 2
def sarah_and_friends := 5
def people_in_bedroom := charlie_and_susan + sarah_and_friends
def people_in_living_room := 8

-- Prove the total number of people in the house is 14.
theorem total_people_in_house : people_in_bedroom + people_in_living_room = 14 := by
  -- Here we can use Lean's proof system, but we skip with 'sorry'
  sorry

end NUMINAMATH_GPT_total_people_in_house_l1767_176737


namespace NUMINAMATH_GPT_negation_correct_l1767_176786

def original_statement (a : ℝ) : Prop :=
  a > 0 → a^2 > 0

def negated_statement (a : ℝ) : Prop :=
  a ≤ 0 → a^2 ≤ 0

theorem negation_correct (a : ℝ) : ¬ (original_statement a) ↔ negated_statement a :=
by
  sorry

end NUMINAMATH_GPT_negation_correct_l1767_176786


namespace NUMINAMATH_GPT_most_probable_sellable_samples_l1767_176761

/-- Prove that the most probable number k of sellable samples out of 24,
given each has a 0.6 probability of being sellable, is either 14 or 15. -/
theorem most_probable_sellable_samples (n : ℕ) (p : ℝ) (q : ℝ) (k₀ k₁ : ℕ) 
  (h₁ : n = 24) (h₂ : p = 0.6) (h₃ : q = 1 - p)
  (h₄ : 24 * p - q < k₀) (h₅ : k₀ < 24 * p + p) 
  (h₆ : k₀ = 14) (h₇ : k₁ = 15) :
  (k₀ = 14 ∨ k₀ = 15) :=
  sorry

end NUMINAMATH_GPT_most_probable_sellable_samples_l1767_176761


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1767_176700

open Real

theorem eccentricity_of_ellipse (a b c : ℝ) 
  (h1 : a > b ∧ b > 0)
  (h2 : c^2 = a^2 - b^2)
  (x : ℝ)
  (h3 : 3 * x = 2 * a)
  (h4 : sqrt 3 * x = 2 * c) :
  c / a = sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1767_176700


namespace NUMINAMATH_GPT_marble_probability_l1767_176758

theorem marble_probability :
  let total_marbles := 13
  let red_marbles := 5
  let white_marbles := 8
  let first_red_prob := (red_marbles:ℚ) / total_marbles
  let second_white_given_first_red_prob := (white_marbles:ℚ) / (total_marbles - 1)
  let third_red_given_first_red_and_second_white_prob := (red_marbles - 1:ℚ) / (total_marbles - 2)
  first_red_prob * second_white_given_first_red_prob * third_red_given_first_red_and_second_white_prob = (40 : ℚ) / 429 :=
by
  let total_marbles := 13
  let red_marbles := 5
  let white_marbles := 8
  let first_red_prob := (red_marbles:ℚ) / total_marbles
  let second_white_given_first_red_prob := (white_marbles:ℚ) / (total_marbles - 1)
  let third_red_given_first_red_and_second_white_prob := (red_marbles - 1:ℚ) / (total_marbles - 2)
  -- Adding sorry to skip the proof
  sorry

end NUMINAMATH_GPT_marble_probability_l1767_176758


namespace NUMINAMATH_GPT_sequence_equal_l1767_176762

variable {n : ℕ} (h1 : 2 ≤ n)
variable (a : ℕ → ℝ)
variable (h2 : ∀ i, a i ≠ -1)
variable (h3 : ∀ i, a (i + 2) = (a i ^ 2 + a i) / (a (i + 1) + 1))
variable (h4 : a n = a 0)
variable (h5 : a (n + 1) = a 1)

theorem sequence_equal 
  (h1 : 2 ≤ n)
  (h2 : ∀ i, a i ≠ -1) 
  (h3 : ∀ i, a (i + 2) = (a i ^ 2 + a i) / (a (i + 1) + 1))
  (h4 : a n = a 0)
  (h5 : a (n + 1) = a 1) :
  ∀ i, a i = a 0 := 
sorry

end NUMINAMATH_GPT_sequence_equal_l1767_176762


namespace NUMINAMATH_GPT_smallest_prime_divides_sum_l1767_176749

theorem smallest_prime_divides_sum :
  ∃ a, Prime a ∧ a ∣ (3 ^ 11 + 5 ^ 13) ∧
       ∀ b, Prime b → b ∣ (3 ^ 11 + 5 ^ 13) → a ≤ b :=
sorry

end NUMINAMATH_GPT_smallest_prime_divides_sum_l1767_176749


namespace NUMINAMATH_GPT_least_integer_value_y_l1767_176796

theorem least_integer_value_y (y : ℤ) (h : abs (3 * y - 4) ≤ 25) : y = -7 :=
sorry

end NUMINAMATH_GPT_least_integer_value_y_l1767_176796


namespace NUMINAMATH_GPT_eliminate_denominators_l1767_176793

theorem eliminate_denominators (x : ℝ) :
  (6 : ℝ) * ((x - 1) / 3) = (6 : ℝ) * (4 - (2 * x + 1) / 2) ↔ 2 * (x - 1) = 24 - 3 * (2 * x + 1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_eliminate_denominators_l1767_176793


namespace NUMINAMATH_GPT_toys_produced_each_day_l1767_176703

theorem toys_produced_each_day (total_weekly_production : ℕ) (days_per_week : ℕ) (H1 : total_weekly_production = 6500) (H2 : days_per_week = 5) : (total_weekly_production / days_per_week = 1300) :=
by {
  sorry
}

end NUMINAMATH_GPT_toys_produced_each_day_l1767_176703


namespace NUMINAMATH_GPT_vector_coordinates_l1767_176725

theorem vector_coordinates (b : ℝ × ℝ)
  (a : ℝ × ℝ := (Real.sqrt 3, 1))
  (angle : ℝ := 2 * Real.pi / 3)
  (norm_b : ℝ := 1)
  (dot_product_eq : (a.fst * b.fst + a.snd * b.snd = -1))
  (norm_b_eq : (b.fst ^ 2 + b.snd ^ 2 = 1)) :
  b = (0, -1) ∨ b = (-Real.sqrt 3 / 2, 1 / 2) :=
sorry

end NUMINAMATH_GPT_vector_coordinates_l1767_176725


namespace NUMINAMATH_GPT_weight_of_B_l1767_176706

/-- Let A, B, and C be the weights in kg of three individuals. If the average weight of A, B, and C is 45 kg,
and the average weight of A and B is 41 kg, and the average weight of B and C is 43 kg,
then the weight of B is 33 kg. -/
theorem weight_of_B (A B C : ℝ) 
  (h1 : A + B + C = 135) 
  (h2 : A + B = 82) 
  (h3 : B + C = 86) : 
  B = 33 := 
by 
  sorry

end NUMINAMATH_GPT_weight_of_B_l1767_176706


namespace NUMINAMATH_GPT_perpendicular_planes_l1767_176714

-- Definitions for lines and planes and their relationships
variable {a b : Line}
variable {α β : Plane}

-- Given conditions for the problem
axiom line_perpendicular (l1 l2 : Line) : Prop -- l1 ⊥ l2
axiom line_parallel (l1 l2 : Line) : Prop -- l1 ∥ l2
axiom line_plane_perpendicular (l : Line) (p : Plane) : Prop -- l ⊥ p
axiom line_plane_parallel (l : Line) (p : Plane) : Prop -- l ∥ p
axiom plane_perpendicular (p1 p2 : Plane) : Prop -- p1 ⊥ p2

-- Problem statement
theorem perpendicular_planes (h1 : line_perpendicular a b)
                            (h2 : line_plane_perpendicular a α)
                            (h3 : line_plane_perpendicular b β) :
                            plane_perpendicular α β :=
sorry

end NUMINAMATH_GPT_perpendicular_planes_l1767_176714


namespace NUMINAMATH_GPT_smallest_n_for_2n_3n_5n_conditions_l1767_176773

theorem smallest_n_for_2n_3n_5n_conditions : 
  ∃ n : ℕ, 
    (∀ k : ℕ, 2 * n ≠ k^2) ∧          -- 2n is a perfect square
    (∀ k : ℕ, 3 * n ≠ k^3) ∧          -- 3n is a perfect cube
    (∀ k : ℕ, 5 * n ≠ k^5) ∧          -- 5n is a perfect fifth power
    n = 11250 :=
sorry

end NUMINAMATH_GPT_smallest_n_for_2n_3n_5n_conditions_l1767_176773


namespace NUMINAMATH_GPT_lesser_fraction_l1767_176724

theorem lesser_fraction (x y : ℚ) (hx : x + y = 13 / 14) (hy : x * y = 1 / 8) : 
  x = (13 - Real.sqrt 57) / 28 ∨ y = (13 - Real.sqrt 57) / 28 :=
by
  sorry

end NUMINAMATH_GPT_lesser_fraction_l1767_176724


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1767_176733

def sufficient_condition (a : ℝ) : Prop := 
  (a > 1) → (1 / a < 1)

def necessary_condition (a : ℝ) : Prop := 
  (1 / a < 1) → (a > 1)

theorem sufficient_but_not_necessary_condition (a : ℝ) : sufficient_condition a ∧ ¬necessary_condition a := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1767_176733


namespace NUMINAMATH_GPT_cos_pi_minus_2alpha_eq_seven_over_twentyfive_l1767_176782

variable (α : ℝ)

theorem cos_pi_minus_2alpha_eq_seven_over_twentyfive 
  (h : Real.sin (π / 2 - α) = 3 / 5) :
  Real.cos (π - 2 * α) = 7 / 25 := 
by
  sorry

end NUMINAMATH_GPT_cos_pi_minus_2alpha_eq_seven_over_twentyfive_l1767_176782


namespace NUMINAMATH_GPT_park_length_l1767_176740

theorem park_length (width : ℕ) (trees_per_sqft : ℕ) (num_trees : ℕ) (total_area : ℕ) (length : ℕ)
  (hw : width = 2000)
  (ht : trees_per_sqft = 20)
  (hn : num_trees = 100000)
  (ha : total_area = num_trees * trees_per_sqft)
  (hl : length = total_area / width) :
  length = 1000 :=
by
  sorry

end NUMINAMATH_GPT_park_length_l1767_176740


namespace NUMINAMATH_GPT_max_d_minus_r_proof_l1767_176777

noncomputable def max_d_minus_r : ℕ := 35

theorem max_d_minus_r_proof (d r : ℕ) (h1 : 2017 % d = r) (h2 : 1029 % d = r) (h3 : 725 % d = r) :
  d - r ≤ max_d_minus_r :=
  sorry

end NUMINAMATH_GPT_max_d_minus_r_proof_l1767_176777


namespace NUMINAMATH_GPT_noon_temperature_l1767_176734

variable (a : ℝ)

theorem noon_temperature (h1 : ∀ (x : ℝ), x = a) (h2 : ∀ (y : ℝ), y = a + 10) :
  a + 10 = y :=
by
  sorry

end NUMINAMATH_GPT_noon_temperature_l1767_176734


namespace NUMINAMATH_GPT_point_in_first_quadrant_l1767_176722

theorem point_in_first_quadrant (x y : ℝ) (h₁ : x = 3) (h₂ : y = 2) (hx : x > 0) (hy : y > 0) :
  ∃ q : ℕ, q = 1 := 
by
  sorry

end NUMINAMATH_GPT_point_in_first_quadrant_l1767_176722


namespace NUMINAMATH_GPT_find_x_l1767_176784

theorem find_x (x y : ℕ) 
  (h1 : 3^x * 4^y = 59049) 
  (h2 : x - y = 10) : 
  x = 10 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l1767_176784


namespace NUMINAMATH_GPT_sum_of_consecutive_neg_ints_l1767_176768

theorem sum_of_consecutive_neg_ints (n : ℤ) (h : n * (n + 1) = 2720) (hn : n < 0) (hn_plus1 : n + 1 < 0) :
  n + (n + 1) = -105 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_neg_ints_l1767_176768


namespace NUMINAMATH_GPT_proof_problem_l1767_176792

def A : Set ℝ := {x | x < 4}
def B : Set ℝ := {x | x^2 - 4 * x + 3 > 0}

theorem proof_problem : {x | x ∈ A ∧ x ∉ (A ∩ B)} = {x | 1 ≤ x ∧ x ≤ 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_problem_l1767_176792
