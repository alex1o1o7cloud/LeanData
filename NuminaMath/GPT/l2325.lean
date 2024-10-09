import Mathlib

namespace bells_ring_together_l2325_232561

theorem bells_ring_together (church school day_care library noon : ℕ) :
  church = 18 ∧ school = 24 ∧ day_care = 30 ∧ library = 35 ∧ noon = 0 →
  ∃ t : ℕ, t = 2520 ∧ ∀ n, (t - noon) % n = 0 := by
  sorry

end bells_ring_together_l2325_232561


namespace average_of_solutions_l2325_232543

theorem average_of_solutions (a b : ℝ) :
  (∃ x1 x2 : ℝ, 3 * a * x1^2 - 6 * a * x1 + 2 * b = 0 ∧
                3 * a * x2^2 - 6 * a * x2 + 2 * b = 0 ∧
                x1 ≠ x2) →
  (1 + 1) / 2 = 1 :=
by
  intros
  sorry

end average_of_solutions_l2325_232543


namespace intersection_on_y_axis_l2325_232525

theorem intersection_on_y_axis (k : ℝ) (x y : ℝ) :
  (2 * x + 3 * y - k = 0) →
  (x - k * y + 12 = 0) →
  (x = 0) →
  k = 6 ∨ k = -6 :=
by
  sorry

end intersection_on_y_axis_l2325_232525


namespace halfway_fraction_between_l2325_232597

theorem halfway_fraction_between (a b : ℚ) (h_a : a = 1/6) (h_b : b = 1/4) : (a + b) / 2 = 5 / 24 :=
by
  have h1 : a = (1 : ℚ) / 6 := h_a
  have h2 : b = (1 : ℚ) / 4 := h_b
  sorry

end halfway_fraction_between_l2325_232597


namespace problem_a_problem_b_problem_c_l2325_232572

theorem problem_a : (7 * (2 / 3) + 16 * (5 / 12)) = 11.3333 := by
  sorry

theorem problem_b : (5 - (2 / (5 / 3))) = 3.8 := by
  sorry

theorem problem_c : (1 + 2 / (1 + 3 / (1 + 4))) = 2.25 := by
  sorry

end problem_a_problem_b_problem_c_l2325_232572


namespace beads_removed_l2325_232555

def total_beads (blue yellow : Nat) : Nat := blue + yellow

def beads_per_part (total : Nat) (parts : Nat) : Nat := total / parts

def beads_remaining (per_part : Nat) (removed : Nat) : Nat := per_part - removed

def doubled_beads (remaining : Nat) : Nat := 2 * remaining

theorem beads_removed {x : Nat} 
  (blue : Nat) (yellow : Nat) (parts : Nat) (final_per_part : Nat) :
  total_beads blue yellow = 39 →
  parts = 3 →
  beads_per_part 39 parts = 13 →
  doubled_beads (beads_remaining 13 x) = 6 →
  x = 10 := by
  sorry

end beads_removed_l2325_232555


namespace omega_not_real_root_l2325_232582

theorem omega_not_real_root {ω : ℂ} (h1 : ω^3 = 1) (h2 : ω ≠ 1) (h3 : ω^2 + ω + 1 = 0) :
  (2 + 3 * ω - ω^2)^3 + (2 - 3 * ω + ω^2)^3 = -68 + 96 * ω :=
by sorry

end omega_not_real_root_l2325_232582


namespace total_unbroken_seashells_l2325_232528

/-
Given:
On the first day, Tom found 7 seashells but 4 were broken.
On the second day, he found 12 seashells but 5 were broken.
On the third day, he found 15 seashells but 8 were broken.

We need to prove that Tom found 17 unbroken seashells in total over the three days.
-/

def first_day_total := 7
def first_day_broken := 4
def first_day_unbroken := first_day_total - first_day_broken

def second_day_total := 12
def second_day_broken := 5
def second_day_unbroken := second_day_total - second_day_broken

def third_day_total := 15
def third_day_broken := 8
def third_day_unbroken := third_day_total - third_day_broken

def total_unbroken := first_day_unbroken + second_day_unbroken + third_day_unbroken

theorem total_unbroken_seashells : total_unbroken = 17 := by
  sorry

end total_unbroken_seashells_l2325_232528


namespace abs_inequality_solution_l2325_232502

theorem abs_inequality_solution (x : ℝ) : 2 * |x - 1| - 1 < 0 ↔ (1 / 2 < x ∧ x < 3 / 2) :=
by
  sorry

end abs_inequality_solution_l2325_232502


namespace blueberries_in_blue_box_l2325_232535

theorem blueberries_in_blue_box (B S : ℕ) (h1 : S - B = 12) (h2 : S + B = 76) : B = 32 :=
sorry

end blueberries_in_blue_box_l2325_232535


namespace num_real_values_for_integer_roots_l2325_232584

theorem num_real_values_for_integer_roots : 
  (∃ (a : ℝ), ∀ (r s : ℤ), r + s = -a ∧ r * s = 9 * a) → ∃ (n : ℕ), n = 10 :=
by
  sorry

end num_real_values_for_integer_roots_l2325_232584


namespace goose_eggs_count_l2325_232530

theorem goose_eggs_count (E : ℕ)
    (hatch_fraction : ℚ := 1/3)
    (first_month_survival : ℚ := 4/5)
    (first_year_survival : ℚ := 2/5)
    (no_migration : ℚ := 3/4)
    (predator_survival : ℚ := 2/3)
    (final_survivors : ℕ := 140) :
    (predator_survival * no_migration * first_year_survival * first_month_survival * hatch_fraction * E : ℚ) = final_survivors → E = 1050 := by
  sorry

end goose_eggs_count_l2325_232530


namespace problem1_l2325_232504

theorem problem1 :
  0.064^(-1 / 3) - (-1 / 8)^0 + 16^(3 / 4) + 0.25^(1 / 2) = 10 :=
by
  sorry

end problem1_l2325_232504


namespace calculate_power_expression_l2325_232544

theorem calculate_power_expression : 4 ^ 2009 * (-0.25) ^ 2008 - 1 = 3 := 
by
  -- steps and intermediate calculations go here
  sorry

end calculate_power_expression_l2325_232544


namespace correct_values_correct_result_l2325_232518

theorem correct_values (a b : ℝ) :
  ((2 * x - a) * (3 * x + b) = 6 * x^2 + 11 * x - 10) ∧
  ((2 * x + a) * (x + b) = 2 * x^2 - 9 * x + 10) →
  (a = -5) ∧ (b = -2) :=
sorry

theorem correct_result :
  (2 * x - 5) * (3 * x - 2) = 6 * x^2 - 19 * x + 10 :=
sorry

end correct_values_correct_result_l2325_232518


namespace valid_assignment_l2325_232558

/-- A function to check if an expression is a valid assignment expression -/
def is_assignment (lhs : String) (rhs : String) : Prop :=
  lhs = "x" ∧ (rhs = "3" ∨ rhs = "x + 1")

theorem valid_assignment :
  (is_assignment "x" "x + 1") ∧
  ¬(is_assignment "3" "x") ∧
  ¬(is_assignment "x" "3") ∧
  ¬(is_assignment "x" "x2 + 1") :=
by
  sorry

end valid_assignment_l2325_232558


namespace cos_C_values_l2325_232580

theorem cos_C_values (sin_A : ℝ) (cos_B : ℝ) (cos_C : ℝ) 
  (h1 : sin_A = 4 / 5) 
  (h2 : cos_B = 12 / 13) 
  : cos_C = -16 / 65 ∨ cos_C = 56 / 65 :=
by
  sorry

end cos_C_values_l2325_232580


namespace cars_on_wednesday_more_than_monday_l2325_232548

theorem cars_on_wednesday_more_than_monday:
  let cars_tuesday := 25
  let cars_monday := 0.8 * cars_tuesday
  let cars_thursday := 10
  let cars_friday := 10
  let cars_saturday := 5
  let cars_sunday := 5
  let total_cars := 97
  ∃ (cars_wednesday : ℝ), cars_wednesday - cars_monday = 2 :=
by
  sorry

end cars_on_wednesday_more_than_monday_l2325_232548


namespace initial_number_2008_l2325_232539

theorem initial_number_2008 
  (numbers_on_blackboard : ℕ → Prop)
  (x : ℕ)
  (Ops : ∀ x, numbers_on_blackboard x → (numbers_on_blackboard (2 * x + 1) ∨ numbers_on_blackboard (x / (x + 2)))) 
  (initial_apearing : numbers_on_blackboard 2008) :
  numbers_on_blackboard 2008 = true :=
sorry

end initial_number_2008_l2325_232539


namespace partition_nat_set_l2325_232503

theorem partition_nat_set :
  ∃ (P : ℕ → ℕ), (∀ (n : ℕ), P n < 100) ∧ (∀ (a b c : ℕ), a + 99 * b = c → (P a = P b ∨ P b = P c ∨ P c = P a)) :=
sorry

end partition_nat_set_l2325_232503


namespace marcy_pets_cat_time_l2325_232542

theorem marcy_pets_cat_time (P : ℝ) (h1 : P + (1/3)*P = 16) : P = 12 :=
by
  sorry

end marcy_pets_cat_time_l2325_232542


namespace house_trailer_payment_difference_l2325_232568

-- Define the costs and periods
def cost_house : ℕ := 480000
def cost_trailer : ℕ := 120000
def loan_period_years : ℕ := 20
def months_per_year : ℕ := 12

-- Calculate total months
def total_months : ℕ := loan_period_years * months_per_year

-- Calculate monthly payments
def monthly_payment_house : ℕ := cost_house / total_months
def monthly_payment_trailer : ℕ := cost_trailer / total_months

-- Theorem stating the difference in monthly payments
theorem house_trailer_payment_difference :
  monthly_payment_house - monthly_payment_trailer = 1500 := by sorry

end house_trailer_payment_difference_l2325_232568


namespace current_age_l2325_232512

theorem current_age (A B S Y : ℕ) 
  (h1: Y = 4) 
  (h2: S = 2 * Y) 
  (h3: B = S + 3) 
  (h4: A + 10 = 2 * (B + 10))
  (h5: A + 10 = 3 * (S + 10))
  (h6: A + 10 = 4 * (Y + 10)) 
  (h7: (A + 10) + (B + 10) + (S + 10) + (Y + 10) = 88) : 
  A = 46 :=
sorry

end current_age_l2325_232512


namespace arithmetic_sequence_sum_l2325_232511

theorem arithmetic_sequence_sum {a_n : ℕ → ℤ} (d : ℤ) (S : ℕ → ℤ) 
  (h_seq : ∀ n, a_n (n + 1) = a_n n + d)
  (h_sum : ∀ n, S n = (n * (2 * a_n 1 + (n - 1) * d)) / 2)
  (h_condition : a_n 1 = 2 * a_n 3 - 3) : 
  S 9 = 27 :=
sorry

end arithmetic_sequence_sum_l2325_232511


namespace rectangular_prism_volume_l2325_232513

variables (a b c : ℝ)

theorem rectangular_prism_volume
  (h1 : a * b = 24)
  (h2 : b * c = 8)
  (h3 : c * a = 3) :
  a * b * c = 24 :=
by
  sorry

end rectangular_prism_volume_l2325_232513


namespace wings_per_person_l2325_232567

-- Define the number of friends
def number_of_friends : ℕ := 15

-- Define the number of wings already cooked
def wings_already_cooked : ℕ := 7

-- Define the number of additional wings cooked
def additional_wings_cooked : ℕ := 45

-- Define the number of friends who don't eat chicken
def friends_not_eating : ℕ := 2

-- Calculate the total number of chicken wings
def total_chicken_wings : ℕ := wings_already_cooked + additional_wings_cooked

-- Calculate the number of friends who will eat chicken
def friends_eating : ℕ := number_of_friends - friends_not_eating

-- Define the statement we want to prove
theorem wings_per_person : total_chicken_wings / friends_eating = 4 := by
  sorry

end wings_per_person_l2325_232567


namespace infinite_series_sum_l2325_232533

theorem infinite_series_sum :
  ∑' n : ℕ, (n + 1) * (1 / 1950)^n = 3802500 / 3802601 :=
by
  sorry

end infinite_series_sum_l2325_232533


namespace parkingGarageCharges_l2325_232521

variable (W : ℕ)

/-- 
  Conditions:
  1. Weekly rental cost is \( W \) dollars.
  2. Monthly rental cost is $24 per month.
  3. A person saves $232 in a year by renting by the month rather than by the week.
  4. There are 52 weeks in a year.
  5. There are 12 months in a year.
-/
def garageChargesPerWeek : Prop :=
  52 * W = 12 * 24 + 232

theorem parkingGarageCharges
  (h : garageChargesPerWeek W) : W = 10 :=
by
  sorry

end parkingGarageCharges_l2325_232521


namespace number_of_intersection_points_l2325_232536

noncomputable section

-- Define a type for Points in the plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Defining the five points
variables (A B C D E : Point)

-- Define the conditions that no three points are collinear
def no_three_collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) ≠ (C.x - A.x) * (B.y - A.y)

-- Define the theorem statement
theorem number_of_intersection_points (h1 : no_three_collinear A B C)
  (h2 : no_three_collinear A B D)
  (h3 : no_three_collinear A B E)
  (h4 : no_three_collinear A C D)
  (h5 : no_three_collinear A C E)
  (h6 : no_three_collinear A D E)
  (h7 : no_three_collinear B C D)
  (h8 : no_three_collinear B C E)
  (h9 : no_three_collinear B D E)
  (h10 : no_three_collinear C D E) :
  ∃ (N : ℕ), N = 40 :=
  sorry

end number_of_intersection_points_l2325_232536


namespace jonas_pairs_of_pants_l2325_232534

theorem jonas_pairs_of_pants (socks pairs_of_shoes t_shirts new_socks : Nat) (P : Nat) :
  socks = 20 → pairs_of_shoes = 5 → t_shirts = 10 → new_socks = 35 →
  2 * (2 * socks + 2 * pairs_of_shoes + t_shirts + P) = 2 * (2 * socks + 2 * pairs_of_shoes + t_shirts) + 70 →
  P = 5 :=
by
  intros hs hps ht hr htotal
  sorry

end jonas_pairs_of_pants_l2325_232534


namespace min_value_of_fraction_sum_l2325_232516

theorem min_value_of_fraction_sum (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x^2 + y^2 + z^2 = 1) :
  (2 * (1/(1-x^2) + 1/(1-y^2) + 1/(1-z^2))) = 3 * Real.sqrt 3 :=
sorry

end min_value_of_fraction_sum_l2325_232516


namespace sum_of_palindromes_l2325_232508

/-- Definition of a three-digit palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  n / 100 = n % 10

theorem sum_of_palindromes (a b : ℕ) (h1 : is_palindrome a)
  (h2 : is_palindrome b) (h3 : a * b = 334491) (h4 : 100 ≤ a)
  (h5 : a < 1000) (h6 : 100 ≤ b) (h7 : b < 1000) : a + b = 1324 :=
sorry

end sum_of_palindromes_l2325_232508


namespace factorize_expression_l2325_232556

theorem factorize_expression
  (x : ℝ) :
  ( (x^2-1)*(x^4+x^2+1)-(x^3+1)^2 ) = -2*(x + 1)*(x^2 - x + 1) :=
by
  sorry

end factorize_expression_l2325_232556


namespace necklace_ratio_l2325_232581

variable {J Q H : ℕ}

theorem necklace_ratio (h1 : H = J + 5) (h2 : H = 25) (h3 : H = Q + 15) : Q / J = 1 / 2 := by
  sorry

end necklace_ratio_l2325_232581


namespace xy_gt_xz_l2325_232515

variable {R : Type*} [LinearOrderedField R]
variables (x y z : R)

theorem xy_gt_xz (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) : x * y > x * z :=
by
  sorry

end xy_gt_xz_l2325_232515


namespace conic_sections_l2325_232570

theorem conic_sections (x y : ℝ) : 
  y^4 - 16*x^4 = 8*y^2 - 4 → 
  (y^2 - 4 * x^2 = 4 ∨ y^2 + 4 * x^2 = 4) :=
sorry

end conic_sections_l2325_232570


namespace composite_quotient_is_one_over_49_l2325_232599

/-- Define the first twelve positive composite integers --/
def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def next_six_composites : List ℕ := [14, 15, 16, 18, 20, 21]

/-- Define the product of a list of integers --/
def product (l : List ℕ) : ℕ := l.foldl (λ acc x => acc * x) 1

/-- This defines the quotient of the products of the first six and the next six composite numbers --/
def composite_quotient : ℚ := (↑(product first_six_composites)) / (↑(product next_six_composites))

/-- Main theorem stating that the composite quotient equals to 1/49 --/
theorem composite_quotient_is_one_over_49 : composite_quotient = 1 / 49 := 
  by
    sorry

end composite_quotient_is_one_over_49_l2325_232599


namespace third_median_length_l2325_232592

theorem third_median_length 
  (m_A m_B : ℝ) -- lengths of the first two medians
  (area : ℝ)   -- area of the triangle
  (h_median_A : m_A = 5) -- the first median is 5 inches
  (h_median_B : m_B = 8) -- the second median is 8 inches
  (h_area : area = 6 * Real.sqrt 15) -- the area of the triangle is 6√15 square inches
  : ∃ m_C : ℝ, m_C = Real.sqrt 31 := -- the length of the third median is √31
sorry

end third_median_length_l2325_232592


namespace area_of_rectangular_field_l2325_232527

theorem area_of_rectangular_field (W L : ℕ) (hL : L = 10) (hFencing : 2 * W + L = 146) : W * L = 680 := by
  sorry

end area_of_rectangular_field_l2325_232527


namespace distance_between_vertices_of_hyperbola_l2325_232517

theorem distance_between_vertices_of_hyperbola :
  ∀ (x y : ℝ), 16 * x^2 - 32 * x - y^2 + 10 * y + 19 = 0 → 
  2 * Real.sqrt (7 / 4) = Real.sqrt 7 :=
by
  intros x y h
  sorry

end distance_between_vertices_of_hyperbola_l2325_232517


namespace range_of_a_l2325_232537

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^2 + 2*x else -(x^2 + 2*(-x))

theorem range_of_a (a : ℝ) (h : f (3 - a^2) > f (2 * a)) : -3 < a ∧ a < 1 := sorry

end range_of_a_l2325_232537


namespace wall_width_l2325_232566

theorem wall_width (V h l w : ℝ) (h_cond : h = 6 * w) (l_cond : l = 42 * w) (vol_cond : 252 * w^3 = 129024) : w = 8 := 
by
  -- Proof is omitted; required to produce lean statement only
  sorry

end wall_width_l2325_232566


namespace convert_to_rectangular_and_find_line_l2325_232509

noncomputable def circle_eq1 (x y : ℝ) : Prop := x^2 + y^2 = 4 * x
noncomputable def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 + 4 * y = 0
noncomputable def line_eq (x y : ℝ) : Prop := y = -x

theorem convert_to_rectangular_and_find_line :
  (∀ x y : ℝ, circle_eq1 x y → x^2 + y^2 = 4 * x) →
  (∀ x y : ℝ, circle_eq2 x y → x^2 + y^2 + 4 * y = 0) →
  (∀ x y : ℝ, circle_eq1 x y ∧ circle_eq2 x y → line_eq x y)
:=
sorry

end convert_to_rectangular_and_find_line_l2325_232509


namespace gcd_153_119_l2325_232574

theorem gcd_153_119 : Nat.gcd 153 119 = 17 :=
by
  sorry

end gcd_153_119_l2325_232574


namespace rice_cake_slices_length_l2325_232569

noncomputable def slice_length (cake_length : ℝ) (num_cakes : ℕ) (overlap : ℝ) (num_slices : ℕ) : ℝ :=
  let total_original_length := num_cakes * cake_length
  let total_overlap := (num_cakes - 1) * overlap
  let actual_length := total_original_length - total_overlap
  actual_length / num_slices

theorem rice_cake_slices_length : 
  slice_length 2.7 5 0.3 6 = 2.05 :=
by
  sorry

end rice_cake_slices_length_l2325_232569


namespace cost_of_pack_of_socks_is_5_l2325_232593

-- Conditions definitions
def shirt_price : ℝ := 12.00
def short_price : ℝ := 15.00
def trunks_price : ℝ := 14.00
def shirts_count : ℕ := 3
def shorts_count : ℕ := 2
def total_bill : ℝ := 102.00
def total_known_cost : ℝ := 3 * shirt_price + 2 * short_price + trunks_price

-- Definition of the problem statement
theorem cost_of_pack_of_socks_is_5 (S : ℝ) : total_bill = total_known_cost + S + 0.2 * (total_known_cost + S) → S = 5 := 
by
  sorry

end cost_of_pack_of_socks_is_5_l2325_232593


namespace relationship_y1_y2_y3_l2325_232586

def quadratic (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variable (m : ℝ) (y1 y2 y3 : ℝ)

-- Given conditions
axiom m_gt_zero : m > 0
axiom point1_on_graph : y1 = quadratic m (-1)
axiom point2_on_graph : y2 = quadratic m (5 / 2)
axiom point3_on_graph : y3 = quadratic m 6

-- Prove the relationship between y1, y2, and y3
theorem relationship_y1_y2_y3 : y3 > y1 ∧ y1 > y2 :=
by sorry

end relationship_y1_y2_y3_l2325_232586


namespace solution_set_of_inequality_l2325_232546

theorem solution_set_of_inequality (x : ℝ) :
  2 * x ≤ -1 → x > -1 → -1 < x ∧ x ≤ -1 / 2 :=
by
  intro h1 h2
  have h3 : x ≤ -1 / 2 := by linarith
  exact ⟨h2, h3⟩

end solution_set_of_inequality_l2325_232546


namespace problem_statement_l2325_232554

-- Defining the terms x, y, and d as per the problem conditions
def x : ℕ := 2351
def y : ℕ := 2250
def d : ℕ := 121

-- Stating the proof problem in Lean
theorem problem_statement : (x - y)^2 / d = 84 := by
  sorry

end problem_statement_l2325_232554


namespace price_per_slice_is_five_l2325_232545

-- Definitions based on the given conditions
def pies_sold := 9
def slices_per_pie := 4
def total_revenue := 180

-- Definition derived from given conditions
def total_slices := pies_sold * slices_per_pie

-- The theorem to prove
theorem price_per_slice_is_five :
  total_revenue / total_slices = 5 :=
by
  sorry

end price_per_slice_is_five_l2325_232545


namespace nail_polishes_total_l2325_232573

theorem nail_polishes_total :
  let k := 25
  let h := k + 8
  let r := k - 6
  h + r = 52 :=
by
  sorry

end nail_polishes_total_l2325_232573


namespace log_equation_solution_l2325_232590

theorem log_equation_solution (a b x : ℝ) (h : 5 * (Real.log x / Real.log b) ^ 2 + 2 * (Real.log x / Real.log a) ^ 2 = 10 * (Real.log x) ^ 2 / (Real.log a * Real.log b)) :
    b = a ^ (1 + Real.sqrt 15 / 5) ∨ b = a ^ (1 - Real.sqrt 15 / 5) :=
sorry

end log_equation_solution_l2325_232590


namespace temperature_on_Monday_l2325_232522

theorem temperature_on_Monday 
  (M T W Th F : ℝ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : F = 31) : 
  M = 39 :=
by
  sorry

end temperature_on_Monday_l2325_232522


namespace correct_technology_used_l2325_232560

-- Define the condition that the program title is "Back to the Dinosaur Era"
def program_title : String := "Back to the Dinosaur Era"

-- Define the condition that the program vividly recreated various dinosaurs and their living environments
def recreated_living_environments : Bool := true

-- Define the options for digital Earth technologies
inductive DigitalEarthTechnology
| InformationSuperhighway
| HighResolutionSatelliteTechnology
| SpatialInformationTechnology
| VisualizationAndVirtualRealityTechnology

-- Define the correct answer
def correct_technology := DigitalEarthTechnology.VisualizationAndVirtualRealityTechnology

-- The proof problem: Prove that given the conditions, the technology used is the correct one
theorem correct_technology_used
  (title : program_title = "Back to the Dinosaur Era")
  (recreated : recreated_living_environments) :
  correct_technology = DigitalEarthTechnology.VisualizationAndVirtualRealityTechnology :=
by
  sorry

end correct_technology_used_l2325_232560


namespace book_arrangements_l2325_232531

theorem book_arrangements (total_books : ℕ) (at_least_in_library : ℕ) (at_least_checked_out : ℕ) 
  (h_total : total_books = 10) (h_at_least_in : at_least_in_library = 2) 
  (h_at_least_out : at_least_checked_out = 3) : 
  ∃ arrangements : ℕ, arrangements = 6 :=
by
  sorry

end book_arrangements_l2325_232531


namespace pollywog_maturation_rate_l2325_232529

theorem pollywog_maturation_rate :
  ∀ (initial_pollywogs : ℕ) (melvin_rate : ℕ) (total_days : ℕ) (melvin_days : ℕ) (remaining_pollywogs : ℕ),
  initial_pollywogs = 2400 →
  melvin_rate = 10 →
  total_days = 44 →
  melvin_days = 20 →
  remaining_pollywogs = initial_pollywogs - (melvin_rate * melvin_days) →
  (total_days * (remaining_pollywogs / (total_days - melvin_days))) = remaining_pollywogs →
  (remaining_pollywogs / (total_days - melvin_days)) = 50 := 
by
  intros initial_pollywogs melvin_rate total_days melvin_days remaining_pollywogs
  intros h_initial h_melvin h_total h_melvin_days h_remaining h_eq
  sorry

end pollywog_maturation_rate_l2325_232529


namespace find_f_105_5_l2325_232505

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom product_condition : ∀ x : ℝ, f x * f (x + 2) = -1
axiom specific_interval : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x = x

theorem find_f_105_5 : f 105.5 = 2.5 :=
by
  sorry

end find_f_105_5_l2325_232505


namespace find_speeds_l2325_232552

/--
From point A to point B, which are 40 km apart, a pedestrian set out at 4:00 AM,
and a cyclist set out at 7:20 AM. The cyclist caught up with the pedestrian exactly
halfway between A and B, after which both continued their journey. A second cyclist
with the same speed as the first cyclist set out from B to A at 8:30 AM and met the
pedestrian one hour after the pedestrian's meeting with the first cyclist. Prove that
the speed of the pedestrian is 5 km/h and the speed of the cyclists is 30 km/h.
-/
theorem find_speeds (x y : ℝ) : 
  (∀ t : ℝ, (0 <= t ∧ t < (7 + (1/3)) ∨ (7 + (1/3)) <= t ∧ t <= 20) -> (x * t + 20 = y * ((7 + (1/3)) - t))) ∧ -- Midpoint and catch-up condition
  (∀ t, (8 + (1/2) <= t) -> (40 - (x * (8 + (1/2))) = y * (t - (8 + (1/2))))) -> -- Second meeting condition
  x = 5 ∧ y = 30 := 
sorry

end find_speeds_l2325_232552


namespace max_value_of_function_l2325_232547

/-- Let y(x) = a^(2*x) + 2 * a^x - 1 for a positive real number a and x in [-1, 1].
    Prove that the maximum value of y on the interval [-1, 1] is 14 when a = 1/3 or a = 3. -/
theorem max_value_of_function (a : ℝ) (a_pos : 0 < a) (h : a = 1 / 3 ∨ a = 3) : 
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^(2*x) + 2 * a^x - 1 = 14 := 
sorry

end max_value_of_function_l2325_232547


namespace simplify_329_mul_101_simplify_54_mul_98_plus_46_mul_98_simplify_98_mul_125_simplify_37_mul_29_plus_37_l2325_232591

theorem simplify_329_mul_101 : 329 * 101 = 33229 := by
  sorry

theorem simplify_54_mul_98_plus_46_mul_98 : 54 * 98 + 46 * 98 = 9800 := by
  sorry

theorem simplify_98_mul_125 : 98 * 125 = 12250 := by
  sorry

theorem simplify_37_mul_29_plus_37 : 37 * 29 + 37 = 1110 := by
  sorry

end simplify_329_mul_101_simplify_54_mul_98_plus_46_mul_98_simplify_98_mul_125_simplify_37_mul_29_plus_37_l2325_232591


namespace equation_of_line_through_points_l2325_232564

-- Definitions for the problem conditions
def point1 : ℝ × ℝ := (-1, 2)
def point2 : ℝ × ℝ := (-3, -2)

-- The theorem stating the equation of the line passing through the given points
theorem equation_of_line_through_points :
  ∃ a b c : ℝ, (a * point1.1 + b * point1.2 + c = 0) ∧ (a * point2.1 + b * point2.2 + c = 0) ∧ 
             (a = 2) ∧ (b = -1) ∧ (c = 4) :=
by
  sorry

end equation_of_line_through_points_l2325_232564


namespace number_of_outfits_l2325_232538

def shirts : ℕ := 5
def hats : ℕ := 3

theorem number_of_outfits : shirts * hats = 15 :=
by 
  -- This part intentionally left blank since no proof required.
  sorry

end number_of_outfits_l2325_232538


namespace odd_if_and_only_if_m_even_l2325_232553

variables (o n m : ℕ)

theorem odd_if_and_only_if_m_even
  (h_o_odd : o % 2 = 1) :
  ((o^3 + n*o + m) % 2 = 1) ↔ (m % 2 = 0) :=
sorry

end odd_if_and_only_if_m_even_l2325_232553


namespace original_two_digit_number_is_52_l2325_232549

theorem original_two_digit_number_is_52 (x : ℕ) (h1 : 10 * x + 6 = x + 474) (h2 : 10 ≤ x ∧ x < 100) : x = 52 :=
sorry

end original_two_digit_number_is_52_l2325_232549


namespace actual_distance_traveled_l2325_232565

theorem actual_distance_traveled (D t : ℝ) 
  (h1 : D = 15 * t)
  (h2 : D + 50 = 35 * t) : 
  D = 37.5 :=
by
  sorry

end actual_distance_traveled_l2325_232565


namespace angle_measure_l2325_232594

-- Define the complement function
def complement (α : ℝ) : ℝ := 180 - α

-- Given condition
variable (α : ℝ)
variable (h : complement α = 120)

-- Theorem to prove
theorem angle_measure : α = 60 :=
by sorry

end angle_measure_l2325_232594


namespace sally_balloon_count_l2325_232519

theorem sally_balloon_count (n_initial : ℕ) (n_lost : ℕ) (n_final : ℕ) 
  (h_initial : n_initial = 9) 
  (h_lost : n_lost = 2) 
  (h_final : n_final = n_initial - n_lost) : 
  n_final = 7 :=
by
  sorry

end sally_balloon_count_l2325_232519


namespace joeys_age_next_multiple_l2325_232520

-- Definitions of the conditions and problem setup
def joey_age (chloe_age : ℕ) : ℕ := chloe_age + 2
def max_age : ℕ := 2
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Sum of digits function
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Main Lean statement
theorem joeys_age_next_multiple (chloe_age : ℕ) (H1 : is_prime chloe_age)
  (H2 : ∀ n : ℕ, (joey_age chloe_age + n) % (max_age + n) = 0)
  (H3 : ∀ i : ℕ, i < 11 → is_prime (chloe_age + i))
  : sum_of_digits (joey_age chloe_age + 1) = 5 :=
  sorry

end joeys_age_next_multiple_l2325_232520


namespace nested_roots_identity_l2325_232524

theorem nested_roots_identity (x : ℝ) (hx : x ≥ 0) : Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15 / 16) :=
sorry

end nested_roots_identity_l2325_232524


namespace work_finished_days_earlier_l2325_232583

theorem work_finished_days_earlier
  (D : ℕ) (M : ℕ) (A : ℕ) (Work : ℕ) (D_new : ℕ) (E : ℕ)
  (hD : D = 8)
  (hM : M = 30)
  (hA : A = 10)
  (hWork : Work = M * D)
  (hTotalWork : Work = 240)
  (hD_new : D_new = Work / (M + A))
  (hDnew_calculated : D_new = 6)
  (hE : E = D - D_new)
  (hE_calculated : E = 2) : 
  E = 2 :=
by
  sorry

end work_finished_days_earlier_l2325_232583


namespace real_part_of_z1_is_zero_l2325_232550

-- Define the imaginary unit i with its property
def i := Complex.I

-- Define z1 using the given expression
noncomputable def z1 := (1 - 2 * i) / (2 + i^5)

-- State the theorem about the real part of z1
theorem real_part_of_z1_is_zero : z1.re = 0 :=
by
  sorry

end real_part_of_z1_is_zero_l2325_232550


namespace compute_cubic_sum_l2325_232510

theorem compute_cubic_sum (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : x * y + x ^ 2 + y ^ 2 = 17) : x ^ 3 + y ^ 3 = 52 :=
sorry

end compute_cubic_sum_l2325_232510


namespace percent_greater_than_average_l2325_232587

variable (M N : ℝ)

theorem percent_greater_than_average (h : M > N) :
  (200 * (M - N)) / (M + N) = ((M - ((M + N) / 2)) / ((M + N) / 2)) * 100 :=
by 
  sorry

end percent_greater_than_average_l2325_232587


namespace rain_puddle_depth_l2325_232551

theorem rain_puddle_depth
  (rain_rate : ℝ) (wait_time : ℝ) (puddle_area : ℝ) 
  (h_rate : rain_rate = 10) (h_time : wait_time = 3) (h_area : puddle_area = 300) :
  ∃ (depth : ℝ), depth = rain_rate * wait_time :=
by
  use 30
  simp [h_rate, h_time]
  sorry

end rain_puddle_depth_l2325_232551


namespace parabola_directrix_l2325_232577

theorem parabola_directrix (p : ℝ) (h : p > 0) (h_directrix : -p / 2 = -4) : p = 8 :=
by
  sorry

end parabola_directrix_l2325_232577


namespace cylinder_height_l2325_232501

theorem cylinder_height {D r : ℝ} (hD : D = 10) (hr : r = 3) : 
  ∃ h : ℝ, h = 8 :=
by
  -- hD -> Diameter of hemisphere = 10
  -- hr -> Radius of cylinder's base = 3
  sorry

end cylinder_height_l2325_232501


namespace quadratic_root_sqrt_2010_2009_l2325_232540

theorem quadratic_root_sqrt_2010_2009 :
  (∃ (a b : ℤ), a = 0 ∧ b = -(2010 + 2 * Real.sqrt 2009) ∧
  ∀ (x : ℝ), x^2 + (a : ℝ) * x + (b : ℝ) = 0 → x = Real.sqrt (2010 + 2 * Real.sqrt 2009) ∨ x = -Real.sqrt (2010 + 2 * Real.sqrt 2009)) :=
sorry

end quadratic_root_sqrt_2010_2009_l2325_232540


namespace log_expression_defined_l2325_232559

theorem log_expression_defined (x : ℝ) : ∃ c : ℝ, (∀ x > c, (x > 7^8)) :=
by
  existsi 7^8
  intro x hx
  sorry

end log_expression_defined_l2325_232559


namespace solve_inequalities_l2325_232526

theorem solve_inequalities (x : ℝ) :
  (1 / x < 1 ∧ |4 * x - 1| > 2) →
  (x < -1/4 ∨ x > 1) :=
by
  sorry

end solve_inequalities_l2325_232526


namespace g_at_6_is_zero_l2325_232523

def g (x : ℝ) : ℝ := 3*x^4 - 18*x^3 + 31*x^2 - 29*x - 72

theorem g_at_6_is_zero : g 6 = 0 :=
by {
  sorry
}

end g_at_6_is_zero_l2325_232523


namespace John_completes_work_alone_10_days_l2325_232588

theorem John_completes_work_alone_10_days
  (R : ℕ)
  (T : ℕ)
  (W : ℕ)
  (H1 : R = 40)
  (H2 : T = 8)
  (H3 : 1/10 = (1/R) + (1/W))
  : W = 10 := sorry

end John_completes_work_alone_10_days_l2325_232588


namespace inequality_solution_set_l2325_232589

theorem inequality_solution_set (x : ℝ) : ((x - 1) * (x^2 - x + 1) > 0) ↔ (x > 1) :=
by
  sorry

end inequality_solution_set_l2325_232589


namespace abs_neg_three_l2325_232585

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l2325_232585


namespace quadratic_inequality_solution_set_l2325_232596

-- Define the necessary variables and conditions
variable (a b c α β : ℝ)
variable (h1 : 0 < α)
variable (h2 : α < β)
variable (h3 : ∀ x : ℝ, (a * x^2 + b * x + c > 0) ↔ (α < x ∧ x < β))

-- Statement to be proved
theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, ((a + c - b) * x^2 + (b - 2 * a) * x + a > 0) ↔ ((1 / (1 + β) < x) ∧ (x < 1 / (1 + α))) :=
sorry

end quadratic_inequality_solution_set_l2325_232596


namespace unit_stratified_sampling_l2325_232571

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

end unit_stratified_sampling_l2325_232571


namespace rectangle_area_l2325_232562

-- Define length and width
def width : ℕ := 6
def length : ℕ := 3 * width

-- Define area of the rectangle
def area (length width : ℕ) : ℕ := length * width

-- Statement to prove
theorem rectangle_area : area length width = 108 := by
  sorry

end rectangle_area_l2325_232562


namespace slope_of_perpendicular_line_l2325_232595

theorem slope_of_perpendicular_line (m1 m2 : ℝ) : 
  (5*x - 2*y = 10) →  ∃ m2, m2 = (-2/5) :=
by sorry

end slope_of_perpendicular_line_l2325_232595


namespace probability_points_one_unit_apart_l2325_232541

theorem probability_points_one_unit_apart :
  let total_points := 16
  let total_pairs := (total_points * (total_points - 1)) / 2
  let favorable_pairs := 12
  let probability := favorable_pairs / total_pairs
  probability = (1 : ℚ) / 10 :=
by
  sorry

end probability_points_one_unit_apart_l2325_232541


namespace no_real_solution_l2325_232500

theorem no_real_solution (x : ℝ) : 
  (¬ (x^4 + 3*x^3)/(x^2 + 3*x + 1) + x = -7) :=
sorry

end no_real_solution_l2325_232500


namespace product_of_numbers_l2325_232598

theorem product_of_numbers (x y : ℤ) (h1 : x + y = 37) (h2 : x - y = 5) : x * y = 336 := by
  sorry

end product_of_numbers_l2325_232598


namespace daughter_weight_l2325_232557

def main : IO Unit :=
  IO.println s!"The weight of the daughter is 50 kg."

theorem daughter_weight :
  ∀ (G D C : ℝ), G + D + C = 110 → D + C = 60 → C = (1/5) * G → D = 50 :=
by
  intros G D C h1 h2 h3
  sorry

end daughter_weight_l2325_232557


namespace student_ticket_price_is_2_50_l2325_232532

-- Defining the given conditions
def adult_ticket_price : ℝ := 4
def total_tickets_sold : ℕ := 59
def total_revenue : ℝ := 222.50
def student_tickets_sold : ℕ := 9

-- The number of adult tickets sold
def adult_tickets_sold : ℕ := total_tickets_sold - student_tickets_sold

-- The total revenue from adult tickets
def revenue_from_adult_tickets : ℝ := adult_tickets_sold * adult_ticket_price

-- The remaining revenue must come from student tickets and defining the price of student ticket
noncomputable def student_ticket_price : ℝ :=
  (total_revenue - revenue_from_adult_tickets) / student_tickets_sold

-- The theorem to be proved
theorem student_ticket_price_is_2_50 : student_ticket_price = 2.50 :=
by
  sorry

end student_ticket_price_is_2_50_l2325_232532


namespace calculate_value_of_expression_l2325_232563

theorem calculate_value_of_expression :
  (2523 - 2428)^2 / 121 = 75 :=
by
  -- calculation steps here
  sorry

end calculate_value_of_expression_l2325_232563


namespace good_numbers_l2325_232578

/-- Definition of a good number -/
def is_good (n : ℕ) : Prop :=
  ∃ (k_1 k_2 k_3 k_4 : ℕ), 
    (1 ≤ k_1 ∧ 1 ≤ k_2 ∧ 1 ≤ k_3 ∧ 1 ≤ k_4) ∧
    (n + k_1 ∣ n + k_1^2) ∧ 
    (n + k_2 ∣ n + k_2^2) ∧ 
    (n + k_3 ∣ n + k_3^2) ∧ 
    (n + k_4 ∣ n + k_4^2) ∧
    (k_1 ≠ k_2) ∧ (k_1 ≠ k_3) ∧ (k_1 ≠ k_4) ∧
    (k_2 ≠ k_3) ∧ (k_2 ≠ k_4) ∧ 
    (k_3 ≠ k_4)

/-- The main theorem to prove -/
theorem good_numbers : 
  is_good 58 ∧ 
  ∀ (p : ℕ), p > 2 → 
  (Prime p ∧ Prime (2 * p + 1) ↔ is_good (2 * p)) :=
by
  sorry

end good_numbers_l2325_232578


namespace smallest_four_digit_multiple_of_18_l2325_232576

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n > 999 ∧ n < 10000 ∧ 18 ∣ n ∧ (∀ m : ℕ, m > 999 ∧ m < 10000 ∧ 18 ∣ m → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l2325_232576


namespace mark_trees_total_l2325_232506

def mark_trees (current_trees new_trees : Nat) : Nat :=
  current_trees + new_trees

theorem mark_trees_total (x y : Nat) (h1 : x = 13) (h2 : y = 12) :
  mark_trees x y = 25 :=
by
  rw [h1, h2]
  sorry

end mark_trees_total_l2325_232506


namespace eval_expr_equals_1_l2325_232507

noncomputable def eval_expr (a b : ℕ) : ℚ :=
  (a + b) / (a * b) / ((a / b) - (b / a))

theorem eval_expr_equals_1 (a b : ℕ) (h₁ : a = 3) (h₂ : b = 2) : eval_expr a b = 1 :=
by
  sorry

end eval_expr_equals_1_l2325_232507


namespace crank_slider_motion_l2325_232514

def omega : ℝ := 10
def OA : ℝ := 90
def AB : ℝ := 90
def AM : ℝ := 60
def t : ℝ := sorry -- t is a variable, no specific value required

theorem crank_slider_motion :
  (∀ t : ℝ, ((90 * Real.cos (10 * t)), (90 * Real.sin (10 * t) + 60)) = (x, y)) ∧
  (∀ t : ℝ, ((-900 * Real.sin (10 * t)), (900 * Real.cos (10 * t))) = (vx, vy)) :=
sorry

end crank_slider_motion_l2325_232514


namespace max_marks_paper_I_l2325_232575

theorem max_marks_paper_I (M : ℝ) (h1 : 0.40 * M = 60) : M = 150 :=
by
  sorry

end max_marks_paper_I_l2325_232575


namespace donation_percentage_correct_l2325_232579

noncomputable def percentage_donated_to_orphan_house (income remaining : ℝ) (given_to_children_percentage : ℝ) (given_to_wife_percentage : ℝ) (remaining_after_donation : ℝ)
    (before_donation_remaining : income * (1 - given_to_children_percentage / 100 - given_to_wife_percentage / 100) = remaining)
    (after_donation_remaining : remaining - remaining_after_donation * remaining = 500) : Prop :=
    100 * (remaining - 500) / remaining = 16.67

theorem donation_percentage_correct 
    (income : ℝ) 
    (child_percentage : ℝ := 10)
    (num_children : ℕ := 2)
    (wife_percentage : ℝ := 20)
    (final_amount : ℝ := 500)
    (income_value : income = 1000 ) : 
    percentage_donated_to_orphan_house income 
    (income * (1 - (child_percentage * num_children) / 100 - wife_percentage / 100)) 
    (child_percentage * num_children)
    wife_percentage 
    final_amount 
    sorry 
    sorry :=
sorry

end donation_percentage_correct_l2325_232579
