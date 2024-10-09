import Mathlib

namespace incorrect_reasoning_C_l1046_104684

theorem incorrect_reasoning_C
  {Point : Type} {Line Plane : Type}
  (A B : Point) (l : Line) (α β : Plane)
  (in_line : Point → Line → Prop)
  (in_plane : Point → Plane → Prop)
  (line_in_plane : Line → Plane → Prop)
  (disjoint : Line → Plane → Prop) :

  ¬(line_in_plane l α) ∧ in_line A l ∧ in_plane A α :=
sorry

end incorrect_reasoning_C_l1046_104684


namespace financing_amount_correct_l1046_104672

-- Define the conditions
def monthly_payment : ℕ := 150
def years : ℕ := 5
def months_per_year : ℕ := 12

-- Define the total financed amount
def total_financed : ℕ := monthly_payment * years * months_per_year

-- The statement that we need to prove
theorem financing_amount_correct : total_financed = 9000 := 
by
  sorry

end financing_amount_correct_l1046_104672


namespace exists_range_of_real_numbers_l1046_104699

theorem exists_range_of_real_numbers (x : ℝ) :
  (x^2 - 5 * x + 7 ≠ 1) ↔ (x ≠ 3 ∧ x ≠ 2) := 
sorry

end exists_range_of_real_numbers_l1046_104699


namespace jordan_has_11_oreos_l1046_104640

-- Define the conditions
def jamesOreos (x : ℕ) : ℕ := 3 + 2 * x
def totalOreos (jordanOreos : ℕ) : ℕ := 36

-- Theorem stating the problem that Jordan has 11 Oreos given the conditions
theorem jordan_has_11_oreos (x : ℕ) (h1 : jamesOreos x + x = totalOreos x) : x = 11 :=
by
  sorry

end jordan_has_11_oreos_l1046_104640


namespace symmetric_point_exists_l1046_104602

-- Define the point P and line equation.
structure Point (α : Type*) := (x : α) (y : α)
def P : Point ℝ := ⟨5, -2⟩
def line_eq (x y : ℝ) : Prop := x - y + 5 = 0

-- Define a function for the line PQ being perpendicular to the given line.
def is_perpendicular (P Q : Point ℝ) : Prop :=
  (Q.y - P.y) / (Q.x - P.x) = -1

-- Define a function for the midpoint of PQ lying on the given line.
def midpoint_on_line (P Q : Point ℝ) : Prop :=
  line_eq ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

-- Define the symmetry function based on the provided conditions.
def is_symmetric (Q : Point ℝ) : Prop :=
  is_perpendicular P Q ∧ midpoint_on_line P Q

-- State the main theorem to be proved: there exists a point Q that satisfies the 
-- conditions and is symmetric to P with respect to the given line.
theorem symmetric_point_exists : ∃ Q : Point ℝ, is_symmetric Q ∧ Q = ⟨-7, 10⟩ :=
by
  sorry

end symmetric_point_exists_l1046_104602


namespace least_possible_sum_of_bases_l1046_104660

theorem least_possible_sum_of_bases : 
  ∃ (c d : ℕ), (2 * c + 9 = 9 * d + 2) ∧ (c + d = 13) :=
by
  sorry

end least_possible_sum_of_bases_l1046_104660


namespace medal_award_ways_l1046_104637

open Nat

theorem medal_award_ways :
  let sprinters := 10
  let italians := 4
  let medals := 3
  let gold_medal_ways := choose italians 1
  let remaining_sprinters := sprinters - 1
  let non_italians := remaining_sprinters - (italians - 1)
  let silver_medal_ways := choose non_italians 1
  let new_remaining_sprinters := remaining_sprinters - 1
  let new_non_italians := new_remaining_sprinters - (italians - 1)
  let bronze_medal_ways := choose new_non_italians 1
  gold_medal_ways * silver_medal_ways * bronze_medal_ways = 120 := by
    sorry

end medal_award_ways_l1046_104637


namespace ratio_of_ian_to_jessica_l1046_104612

/-- 
Rodney has 35 dollars more than Ian. 
Jessica has 100 dollars. 
Jessica has 15 dollars more than Rodney. 
Prove that the ratio of Ian's money to Jessica's money is 1/2.
-/
theorem ratio_of_ian_to_jessica (I R J : ℕ) (h1 : R = I + 35) (h2 : J = 100) (h3 : J = R + 15) :
  I / J = 1 / 2 :=
by
  sorry

end ratio_of_ian_to_jessica_l1046_104612


namespace square_of_sum_opposite_l1046_104686

theorem square_of_sum_opposite (a b : ℝ) : (-(a) + b)^2 = (-a + b)^2 :=
by
  sorry

end square_of_sum_opposite_l1046_104686


namespace find_volume_from_vessel_c_l1046_104604

noncomputable def concentration_vessel_a : ℝ := 0.45
noncomputable def concentration_vessel_b : ℝ := 0.30
noncomputable def concentration_vessel_c : ℝ := 0.10
noncomputable def volume_vessel_a : ℝ := 4
noncomputable def volume_vessel_b : ℝ := 5
noncomputable def resultant_concentration : ℝ := 0.26

theorem find_volume_from_vessel_c (x : ℝ) : 
    concentration_vessel_a * volume_vessel_a + concentration_vessel_b * volume_vessel_b + concentration_vessel_c * x = 
    resultant_concentration * (volume_vessel_a + volume_vessel_b + x) → 
    x = 6 :=
by
  sorry

end find_volume_from_vessel_c_l1046_104604


namespace savings_promotion_l1046_104655

theorem savings_promotion (reg_price promo_price num_pizzas : ℕ) (h1 : reg_price = 18) (h2 : promo_price = 5) (h3 : num_pizzas = 3) :
  reg_price * num_pizzas - promo_price * num_pizzas = 39 := by
  sorry

end savings_promotion_l1046_104655


namespace least_number_remainder_l1046_104669

theorem least_number_remainder (n : ℕ) :
  (n % 7 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) ∧ (n % 18 = 4) → n = 256 :=
by
  sorry

end least_number_remainder_l1046_104669


namespace smallest_int_cond_l1046_104671

theorem smallest_int_cond (b : ℕ) :
  (b % 9 = 5) ∧ (b % 11 = 7) → b = 95 :=
by
  intro h
  sorry

end smallest_int_cond_l1046_104671


namespace monotonically_increasing_interval_l1046_104653

noncomputable def f (x : ℝ) : ℝ := Real.log (-3 * x^2 + 4 * x + 4)

theorem monotonically_increasing_interval :
  ∀ x, x ∈ Set.Ioc (-2/3 : ℝ) (2/3 : ℝ) → MonotoneOn f (Set.Ioc (-2/3) (2/3)) :=
sorry

end monotonically_increasing_interval_l1046_104653


namespace arithmetic_geometric_sum_l1046_104634

noncomputable def a_n (n : ℕ) := 3 * n - 2
noncomputable def b_n (n : ℕ) := 4 ^ (n - 1)

theorem arithmetic_geometric_sum (n : ℕ) :
    a_n 1 = 1 ∧ a_n 2 = b_n 2 ∧ a_n 6 = b_n 3 ∧ S_n = 1 + (n - 1) * 4 ^ n :=
by sorry

end arithmetic_geometric_sum_l1046_104634


namespace smallest_positive_angle_l1046_104668

theorem smallest_positive_angle (k : ℤ) : ∃ α, α = 400 + k * 360 ∧ α > 0 ∧ α = 40 :=
by
  use 40
  sorry

end smallest_positive_angle_l1046_104668


namespace product_of_16_and_21_point_3_l1046_104651

theorem product_of_16_and_21_point_3 (h1 : 213 * 16 = 3408) : 16 * 21.3 = 340.8 :=
by sorry

end product_of_16_and_21_point_3_l1046_104651


namespace trisha_spending_l1046_104618

theorem trisha_spending :
  let initial_amount := 167
  let spent_meat := 17
  let spent_veggies := 43
  let spent_eggs := 5
  let spent_dog_food := 45
  let remaining_amount := 35
  let total_spent := initial_amount - remaining_amount
  let other_spending := spent_meat + spent_veggies + spent_eggs + spent_dog_food
  total_spent - other_spending = 22 :=
by
  let initial_amount := 167
  let spent_meat := 17
  let spent_veggies := 43
  let spent_eggs := 5
  let spent_dog_food := 45
  let remaining_amount := 35
  -- Calculate total spent
  let total_spent := initial_amount - remaining_amount
  -- Calculate spending on other items
  let other_spending := spent_meat + spent_veggies + spent_eggs + spent_dog_food
  -- Statement to prove
  show total_spent - other_spending = 22
  sorry

end trisha_spending_l1046_104618


namespace yogurt_production_cost_l1046_104644

-- Define the conditions
def milk_cost_per_liter : ℝ := 1.5
def fruit_cost_per_kg : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Define the theorem statement
theorem yogurt_production_cost : 
  (milk_cost_per_liter * milk_needed_per_batch + fruit_cost_per_kg * fruit_needed_per_batch) * batches = 63 := 
  by 
  sorry

end yogurt_production_cost_l1046_104644


namespace function_conditions_satisfied_l1046_104638

noncomputable def function_satisfying_conditions : ℝ → ℝ := fun x => -2 * x^2 + 3 * x

theorem function_conditions_satisfied :
  (function_satisfying_conditions 1 = 1) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ function_satisfying_conditions x = y) ∧
  (∀ x y : ℝ, x > 1 ∧ y = function_satisfying_conditions x → ∃ ε > 0, ∀ δ > 0, (x + δ > 1 → function_satisfying_conditions (x + δ) < y)) :=
by
  sorry

end function_conditions_satisfied_l1046_104638


namespace polygon_coloring_l1046_104676

theorem polygon_coloring (n m : ℕ) (h1 : n ≥ 3) (h2 : m ≥ 3) :
    ∃ b_n : ℕ, b_n = (m - 1) * ((m - 1) ^ (n - 1) + (-1 : ℤ) ^ n) :=
sorry

end polygon_coloring_l1046_104676


namespace sin_double_angle_l1046_104657

variable (θ : ℝ)

-- Given condition: tan(θ) = -3/5
def tan_theta : Prop := Real.tan θ = -3/5

-- Target to prove: sin(2θ) = -15/17
theorem sin_double_angle : tan_theta θ → Real.sin (2*θ) = -15/17 :=
by
  sorry

end sin_double_angle_l1046_104657


namespace amount_sharpened_off_l1046_104675

-- Defining the initial length of the pencil
def initial_length : ℕ := 31

-- Defining the length of the pencil after sharpening
def after_sharpening_length : ℕ := 14

-- Proving the amount sharpened off the pencil
theorem amount_sharpened_off : initial_length - after_sharpening_length = 17 := 
by 
  -- Here we would insert the proof steps, 
  -- but as instructed we leave it as sorry.
  sorry

end amount_sharpened_off_l1046_104675


namespace isosceles_triangle_side_length_l1046_104627

theorem isosceles_triangle_side_length (total_length : ℝ) (one_side_length : ℝ) (remaining_wire : ℝ) (equal_side : ℝ) :
  total_length = 20 → one_side_length = 6 → remaining_wire = total_length - one_side_length → remaining_wire / 2 = equal_side →
  equal_side = 7 :=
by
  intros h_total h_one_side h_remaining h_equal_side
  sorry

end isosceles_triangle_side_length_l1046_104627


namespace find_correct_result_l1046_104682

noncomputable def correct_result : Prop :=
  ∃ (x : ℝ), (-1.25 * x - 0.25 = 1.25 * x) ∧ (-1.25 * x = 0.125)

theorem find_correct_result : correct_result :=
  sorry

end find_correct_result_l1046_104682


namespace calc_subtract_l1046_104665

-- Define the repeating decimal
def repeating_decimal := (11 : ℚ) / 9

-- Define the problem statement
theorem calc_subtract : 3 - repeating_decimal = (16 : ℚ) / 9 := by
  sorry

end calc_subtract_l1046_104665


namespace ab_plus_cd_l1046_104673

variable (a b c d : ℝ)

theorem ab_plus_cd (h1 : a + b + c = -4)
                  (h2 : a + b + d = 2)
                  (h3 : a + c + d = 15)
                  (h4 : b + c + d = 10) :
                  a * b + c * d = 485 / 9 :=
by
  sorry

end ab_plus_cd_l1046_104673


namespace days_not_worked_correct_l1046_104643

def total_days : ℕ := 20
def earnings_for_work (days_worked : ℕ) : ℤ := 80 * days_worked
def penalty_for_no_work (days_not_worked : ℕ) : ℤ := -40 * days_not_worked
def final_earnings (days_worked days_not_worked : ℕ) : ℤ := 
  (earnings_for_work days_worked) + (penalty_for_no_work days_not_worked)
def received_amount : ℤ := 880

theorem days_not_worked_correct {y x : ℕ} 
  (h1 : x + y = total_days) 
  (h2 : final_earnings x y = received_amount) :
  y = 6 :=
sorry

end days_not_worked_correct_l1046_104643


namespace polyhedron_edges_l1046_104635

theorem polyhedron_edges (F V E : ℕ) (h1 : F = 12) (h2 : V = 20) (h3 : F + V = E + 2) : E = 30 :=
by
  -- Additional details would go here, proof omitted as instructed.
  sorry

end polyhedron_edges_l1046_104635


namespace deaths_during_operation_l1046_104697

noncomputable def initial_count : ℕ := 1000
noncomputable def first_day_remaining (n : ℕ) := 5 * n / 6
noncomputable def second_day_remaining (n : ℕ) := (35 * n / 48) - 1
noncomputable def third_day_remaining (n : ℕ) := (105 * n / 192) - 3 / 4

theorem deaths_during_operation : ∃ n : ℕ, initial_count - n = 472 ∧ n = 528 :=
  by sorry

end deaths_during_operation_l1046_104697


namespace longer_piece_length_l1046_104663

theorem longer_piece_length (x : ℝ) (h1 : x + (x + 2) = 30) : x + 2 = 16 :=
by sorry

end longer_piece_length_l1046_104663


namespace ratio_spaghetti_to_fettuccine_l1046_104648

def spg : Nat := 300
def fet : Nat := 80

theorem ratio_spaghetti_to_fettuccine : spg / gcd spg fet = 300 / 20 ∧ fet / gcd spg fet = 80 / 20 ∧ (spg / gcd spg fet) / (fet / gcd spg fet) = 15 / 4 := by
  sorry

end ratio_spaghetti_to_fettuccine_l1046_104648


namespace arithmetic_problem_l1046_104605

noncomputable def arithmetic_progression (a₁ d : ℝ) (n : ℕ) := a₁ + (n - 1) * d

noncomputable def sum_terms (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_problem (a₁ d : ℝ)
  (h₁ : a₁ + (a₁ + 2 * d) = 5)
  (h₂ : 4 * (2 * a₁ + 3 * d) / 2 = 20) :
  (sum_terms a₁ d 8 - 2 * sum_terms a₁ d 4) / (sum_terms a₁ d 6 - sum_terms a₁ d 4 - sum_terms a₁ d 2) = 10 := by
  sorry

end arithmetic_problem_l1046_104605


namespace distinct_diagonals_in_convex_nonagon_l1046_104698

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l1046_104698


namespace number_of_schools_is_8_l1046_104603

-- Define the number of students trying out and not picked per school
def students_trying_out := 65.0
def students_not_picked := 17.0
def students_picked := students_trying_out - students_not_picked

-- Define the total number of students who made the teams
def total_students_made_teams := 384.0

-- Define the number of schools
def number_of_schools := total_students_made_teams / students_picked

theorem number_of_schools_is_8 : number_of_schools = 8 := by
  -- Proof omitted
  sorry

end number_of_schools_is_8_l1046_104603


namespace negation_of_universal_statement_l1046_104664

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ^ 2 ≠ x) ↔ ∃ x : ℝ, x ^ 2 = x :=
by
  sorry

end negation_of_universal_statement_l1046_104664


namespace product_of_primes_l1046_104647

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l1046_104647


namespace A_days_to_complete_alone_l1046_104681

theorem A_days_to_complete_alone
  (work_left : ℝ := 0.41666666666666663)
  (B_days : ℝ := 20)
  (combined_days : ℝ := 5)
  : ∃ (A_days : ℝ), A_days = 15 := 
by
  sorry

end A_days_to_complete_alone_l1046_104681


namespace triangle_solutions_l1046_104621

theorem triangle_solutions :
  ∀ (a b c : ℝ) (A B C : ℝ),
  a = 7.012 ∧
  c - b = 1.753 ∧
  B = 38 + 12/60 + 48/3600 ∧
  A = 81 + 47/60 + 12.5/3600 ∧
  C = 60 ∧
  b = 4.3825 ∧
  c = 6.1355 :=
sorry -- Proof goes here

end triangle_solutions_l1046_104621


namespace alex_received_12_cookies_l1046_104613

theorem alex_received_12_cookies :
  ∃ y: ℕ, (∀ s: ℕ, y = s + 8 ∧ s = y / 3) → y = 12 := by
  sorry

end alex_received_12_cookies_l1046_104613


namespace total_number_of_pipes_l1046_104614

theorem total_number_of_pipes (bottom_layer top_layer layers : ℕ) 
  (h_bottom_layer : bottom_layer = 13) 
  (h_top_layer : top_layer = 3) 
  (h_layers : layers = 11) : 
  bottom_layer + top_layer = 16 → 
  (bottom_layer + top_layer) * layers / 2 = 88 := 
by
  intro h_sum
  sorry

end total_number_of_pipes_l1046_104614


namespace two_presses_printing_time_l1046_104662

def printing_time (presses newspapers hours : ℕ) : ℕ := sorry

theorem two_presses_printing_time :
  ∀ (presses newspapers hours : ℕ),
    (presses = 4) →
    (newspapers = 8000) →
    (hours = 6) →
    printing_time 2 6000 hours = 9 := sorry

end two_presses_printing_time_l1046_104662


namespace eggs_in_seven_boxes_l1046_104616

-- define the conditions
def eggs_per_box : Nat := 15
def number_of_boxes : Nat := 7

-- state the main theorem to prove
theorem eggs_in_seven_boxes : eggs_per_box * number_of_boxes = 105 := by
  sorry

end eggs_in_seven_boxes_l1046_104616


namespace tangent_line_eq_l1046_104650

theorem tangent_line_eq
    (f : ℝ → ℝ) (f_def : ∀ x, f x = x ^ 2)
    (tangent_point : ℝ × ℝ) (tangent_point_def : tangent_point = (1, 1))
    (f' : ℝ → ℝ) (f'_def : ∀ x, f' x = 2 * x)
    (slope_at_1 : f' 1 = 2) :
    ∃ (a b : ℝ), a = 2 ∧ b = -1 ∧ ∀ x y, y = a * x + b ↔ (2 * x - y - 1 = 0) :=
sorry

end tangent_line_eq_l1046_104650


namespace base_b_conversion_l1046_104611

theorem base_b_conversion (b : ℝ) (h₁ : 1 * 5^2 + 3 * 5^1 + 2 * 5^0 = 42) (h₂ : 2 * b^2 + 2 * b + 1 = 42) :
  b = (-1 + Real.sqrt 83) / 2 := 
  sorry

end base_b_conversion_l1046_104611


namespace range_of_a_l1046_104688

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_non_neg (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ y) → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  even_function f → increasing_on_non_neg f → f a ≤ f 2 → -2 ≤ a ∧ a ≤ 2 :=
by
  intro h_even h_increasing h_le
  sorry

end range_of_a_l1046_104688


namespace number_of_fours_is_even_l1046_104609

theorem number_of_fours_is_even (n3 n4 n5 : ℕ) 
  (h1 : n3 + n4 + n5 = 80)
  (h2 : 3 * n3 + 4 * n4 + 5 * n5 = 276) : Even n4 := 
sorry

end number_of_fours_is_even_l1046_104609


namespace average_decrease_l1046_104691

theorem average_decrease (avg_6 : ℝ) (obs_7 : ℝ) (new_avg : ℝ) (decrease : ℝ) :
  avg_6 = 11 → obs_7 = 4 → (6 * avg_6 + obs_7) / 7 = new_avg → avg_6 - new_avg = decrease → decrease = 1 :=
  by
    intros h1 h2 h3 h4
    rw [h1, h2] at *
    sorry

end average_decrease_l1046_104691


namespace quadratic_root_m_eq_neg_fourteen_l1046_104687

theorem quadratic_root_m_eq_neg_fourteen : ∀ (m : ℝ), (∃ x : ℝ, x = 2 ∧ x^2 + 5 * x + m = 0) → m = -14 :=
by
  sorry

end quadratic_root_m_eq_neg_fourteen_l1046_104687


namespace debt_calculation_correct_l1046_104652

-- Conditions
def initial_debt : ℤ := 40
def repayment : ℤ := initial_debt / 2
def additional_borrowing : ℤ := 10

-- Final Debt Calculation
def remaining_debt : ℤ := initial_debt - repayment
def final_debt : ℤ := remaining_debt + additional_borrowing

-- Proof Statement
theorem debt_calculation_correct : final_debt = 30 := 
by 
  -- Skipping the proof
  sorry

end debt_calculation_correct_l1046_104652


namespace find_n_l1046_104622

theorem find_n (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h1 : ∃ n : ℕ, n - 76 = a^3) (h2 : ∃ n : ℕ, n + 76 = b^3) : ∃ n : ℕ, n = 140 :=
by 
  sorry

end find_n_l1046_104622


namespace janet_roses_l1046_104677

def total_flowers (used_flowers extra_flowers : Nat) : Nat :=
  used_flowers + extra_flowers

def number_of_roses (total tulips : Nat) : Nat :=
  total - tulips

theorem janet_roses :
  ∀ (used_flowers extra_flowers tulips : Nat),
  used_flowers = 11 → extra_flowers = 4 → tulips = 4 →
  number_of_roses (total_flowers used_flowers extra_flowers) tulips = 11 :=
by
  intros used_flowers extra_flowers tulips h_used h_extra h_tulips
  rw [h_used, h_extra, h_tulips]
  -- proof steps skipped
  sorry

end janet_roses_l1046_104677


namespace temperature_in_quebec_city_is_negative_8_l1046_104690

def temperature_vancouver : ℝ := 22
def temperature_calgary (temperature_vancouver : ℝ) : ℝ := temperature_vancouver - 19
def temperature_quebec_city (temperature_calgary : ℝ) : ℝ := temperature_calgary - 11

theorem temperature_in_quebec_city_is_negative_8 :
  temperature_quebec_city (temperature_calgary temperature_vancouver) = -8 := by
  sorry

end temperature_in_quebec_city_is_negative_8_l1046_104690


namespace total_num_of_cars_l1046_104695

-- Define conditions
def row_from_front := 14
def row_from_left := 19
def row_from_back := 11
def row_from_right := 16

-- Compute total number of rows from front to back
def rows_front_to_back : ℕ := (row_from_front - 1) + 1 + (row_from_back - 1)

-- Compute total number of rows from left to right
def rows_left_to_right : ℕ := (row_from_left - 1) + 1 + (row_from_right - 1)

theorem total_num_of_cars :
  rows_front_to_back = 24 ∧
  rows_left_to_right = 34 ∧
  24 * 34 = 816 :=
by
  sorry

end total_num_of_cars_l1046_104695


namespace last_three_digits_of_7_pow_103_l1046_104670

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 327 :=
by
  sorry

end last_three_digits_of_7_pow_103_l1046_104670


namespace area_of_quadrilateral_l1046_104620

theorem area_of_quadrilateral (A B C : ℝ) (triangle1 triangle2 triangle3 quadrilateral : ℝ)
  (hA : A = 5) (hB : B = 9) (hC : C = 9)
  (h_sum : quadrilateral = triangle1 + triangle2 + triangle3)
  (h1 : triangle1 = A)
  (h2 : triangle2 = B)
  (h3 : triangle3 = C) :
  quadrilateral = 40 :=
by
  sorry

end area_of_quadrilateral_l1046_104620


namespace total_pumpkins_l1046_104624

-- Define the number of pumpkins grown by Sandy and Mike
def pumpkinsSandy : ℕ := 51
def pumpkinsMike : ℕ := 23

-- Prove that their total is 74
theorem total_pumpkins : pumpkinsSandy + pumpkinsMike = 74 := by
  sorry

end total_pumpkins_l1046_104624


namespace average_speed_of_bus_trip_l1046_104601

theorem average_speed_of_bus_trip 
  (v d : ℝ) 
  (h1 : d = 560)
  (h2 : ∀ v > 0, ∀ Δv > 0, (d / v) - (d / (v + Δv)) = 2)
  (h3 : Δv = 10): 
  v = 50 := 
by 
  sorry

end average_speed_of_bus_trip_l1046_104601


namespace common_ratio_geometric_series_l1046_104678

theorem common_ratio_geometric_series
  (a₁ a₂ a₃ : ℚ)
  (h₁ : a₁ = 7 / 8)
  (h₂ : a₂ = -14 / 27)
  (h₃ : a₃ = 56 / 81) :
  (a₂ / a₁ = a₃ / a₂) ∧ (a₂ / a₁ = -2 / 3) :=
by
  -- The proof will follow here
  sorry

end common_ratio_geometric_series_l1046_104678


namespace income_percentage_l1046_104629

theorem income_percentage (J T M : ℝ) 
  (h1 : T = 0.5 * J) 
  (h2 : M = 1.6 * T) : 
  M = 0.8 * J :=
by 
  sorry

end income_percentage_l1046_104629


namespace incorrect_intersections_l1046_104633

theorem incorrect_intersections :
  (∃ x, (x = x ∧ x = Real.sqrt (x + 2)) ↔ x = 1 ∨ x = 2) →
  (∃ x, (x^2 - 3 * x + 2 = 2 ∧ x = 2) ↔ x = 1 ∨ x = 2) →
  (∃ x, (Real.sin x = 3 * x - 4 ∧ x = 2) ↔ x = 1 ∨ x = 2) → False :=
by {
  sorry
}

end incorrect_intersections_l1046_104633


namespace juanita_loss_l1046_104696

theorem juanita_loss
  (entry_fee : ℝ) (hit_threshold : ℕ) (drum_payment_per_hit : ℝ) (drums_hit : ℕ) :
  entry_fee = 10 →
  hit_threshold = 200 →
  drum_payment_per_hit = 0.025 →
  drums_hit = 300 →
  - (entry_fee - ((drums_hit - hit_threshold) * drum_payment_per_hit)) = 7.50 :=
by
  intros h1 h2 h3 h4
  sorry

end juanita_loss_l1046_104696


namespace carson_seed_l1046_104692

variable (s f : ℕ) -- Define the variables s and f as nonnegative integers

-- Conditions given in the problem
axiom h1 : s = 3 * f
axiom h2 : s + f = 60

-- The theorem to prove
theorem carson_seed : s = 45 :=
by
  -- Proof would go here
  sorry

end carson_seed_l1046_104692


namespace factorize_x_squared_minus_four_l1046_104685

theorem factorize_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) :=
by
  sorry

end factorize_x_squared_minus_four_l1046_104685


namespace binom_20_4_l1046_104693

theorem binom_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end binom_20_4_l1046_104693


namespace rectangle_in_right_triangle_dimensions_l1046_104626

theorem rectangle_in_right_triangle_dimensions :
  ∀ (DE EF DF x y : ℝ),
  DE = 6 → EF = 8 → DF = 10 →
  -- Assuming isosceles right triangle (interchange sides for the proof)
  ∃ (G H I J : ℝ),
  (G = 0 ∧ H = 0 ∧ I = y ∧ J = x ∧ x * y = GH * GI) → -- Rectangle GH parallel to DE
  (x = 10 / 8 * y) →
  ∃ (GH GI : ℝ), 
  GH = 8 / 8.33 ∧ GI = 6.67 / 8.33 →
  (x = 25 / 3 ∧ y = 40 / 6) :=
by
  sorry

end rectangle_in_right_triangle_dimensions_l1046_104626


namespace jenna_stamp_division_l1046_104654

theorem jenna_stamp_division (a b c : ℕ) (h₁ : a = 945) (h₂ : b = 1260) (h₃ : c = 630) :
  Nat.gcd (Nat.gcd a b) c = 105 :=
by
  rw [h₁, h₂, h₃]
  -- Now we need to prove Nat.gcd (Nat.gcd 945 1260) 630 = 105
  sorry

end jenna_stamp_division_l1046_104654


namespace workman_B_days_l1046_104608

theorem workman_B_days (A B : ℝ) (hA : A = (1 / 2) * B) (hTogether : (A + B) * 14 = 1) :
  1 / B = 21 :=
sorry

end workman_B_days_l1046_104608


namespace irreducible_fraction_l1046_104631

theorem irreducible_fraction (n : ℤ) : Int.gcd (2 * n + 1) (3 * n + 1) = 1 :=
sorry

end irreducible_fraction_l1046_104631


namespace kamal_chemistry_marks_l1046_104642

-- Definitions of the marks
def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 72
def biology_marks : ℕ := 82
def average_marks : ℕ := 71
def num_subjects : ℕ := 5

-- Statement to be proved
theorem kamal_chemistry_marks : ∃ (chemistry_marks : ℕ), 
  76 + 60 + 72 + 82 + chemistry_marks = 71 * 5 :=
by
sorry

end kamal_chemistry_marks_l1046_104642


namespace find_k_for_parallel_vectors_l1046_104649

theorem find_k_for_parallel_vectors (k : ℝ) :
  let a := (1, k)
  let b := (9, k - 6)
  (1 * (k - 6) - 9 * k = 0) → k = -3 / 4 :=
by
  intros a b parallel_cond
  sorry

end find_k_for_parallel_vectors_l1046_104649


namespace percent_is_50_l1046_104623

variable (cats hogs percent : ℕ)
variable (hogs_eq_3cats : hogs = 3 * cats)
variable (hogs_eq_75 : hogs = 75)

theorem percent_is_50
  (cats_minus_5_percent_eq_10 : (cats - 5) * percent = 1000)
  (cats_eq_25 : cats = 25) :
  percent = 50 := by
  sorry

end percent_is_50_l1046_104623


namespace perimeter_of_C_l1046_104639

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l1046_104639


namespace sum_due_is_correct_l1046_104617

theorem sum_due_is_correct (BD TD PV : ℝ) (h1 : BD = 80) (h2 : TD = 70) (h_relation : BD = TD + (TD^2) / PV) : PV = 490 :=
by sorry

end sum_due_is_correct_l1046_104617


namespace mb_range_l1046_104615

theorem mb_range (m b : ℝ) (hm : m = 3 / 4) (hb : b = -2 / 3) :
  -1 < m * b ∧ m * b < 0 :=
by
  rw [hm, hb]
  sorry

end mb_range_l1046_104615


namespace lines_perpendicular_to_same_plane_are_parallel_l1046_104689

theorem lines_perpendicular_to_same_plane_are_parallel 
  (parallel_proj_parallel_lines : Prop)
  (planes_parallel_to_same_line : Prop)
  (planes_perpendicular_to_same_plane : Prop)
  (lines_perpendicular_to_same_plane : Prop) 
  (h1 : ¬ parallel_proj_parallel_lines)
  (h2 : ¬ planes_parallel_to_same_line)
  (h3 : ¬ planes_perpendicular_to_same_plane) :
  lines_perpendicular_to_same_plane := 
sorry

end lines_perpendicular_to_same_plane_are_parallel_l1046_104689


namespace sum_and_ratio_l1046_104645

theorem sum_and_ratio (x y : ℝ) (h1 : x + y = 480) (h2 : x / y = 0.8) : y - x = 53.34 :=
by
  sorry

end sum_and_ratio_l1046_104645


namespace prob_both_correct_l1046_104674

def prob_A : ℤ := 70
def prob_B : ℤ := 55
def prob_neither : ℤ := 20

theorem prob_both_correct : (prob_A + prob_B - (100 - prob_neither)) = 45 :=
by
  sorry

end prob_both_correct_l1046_104674


namespace unit_vector_perpendicular_l1046_104656

theorem unit_vector_perpendicular (x y : ℝ) (h : 3 * x + 4 * y = 0) (m : x^2 + y^2 = 1) : 
  (x = -4/5 ∧ y = 3/5) ∨ (x = 4/5 ∧ y = -3/5) :=
by
  sorry

end unit_vector_perpendicular_l1046_104656


namespace probability_of_selection_l1046_104683

-- Problem setup
def number_of_students : ℕ := 54
def number_of_students_eliminated : ℕ := 4
def number_of_remaining_students : ℕ := number_of_students - number_of_students_eliminated
def number_of_students_selected : ℕ := 5

-- Statement to be proved
theorem probability_of_selection :
  (number_of_students_selected : ℚ) / (number_of_students : ℚ) = 5 / 54 :=
sorry

end probability_of_selection_l1046_104683


namespace num_combinations_two_dresses_l1046_104630

def num_colors : ℕ := 4
def num_patterns : ℕ := 5

def combinations_first_dress : ℕ := num_colors * num_patterns
def combinations_second_dress : ℕ := (num_colors - 1) * (num_patterns - 1)

theorem num_combinations_two_dresses :
  (combinations_first_dress * combinations_second_dress) = 240 := by
  sorry

end num_combinations_two_dresses_l1046_104630


namespace smallest_value_a1_l1046_104694

def seq (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = 7 * a (n-1) - 2 * n

theorem smallest_value_a1 (a : ℕ → ℝ) (h1 : ∀ n, 0 < a n) (h2 : seq a) : 
  a 1 ≥ 13 / 18 :=
sorry

end smallest_value_a1_l1046_104694


namespace perimeter_of_smaller_rectangle_l1046_104679

theorem perimeter_of_smaller_rectangle :
  ∀ (L W n : ℕ), 
  L = 16 → W = 20 → n = 10 →
  (∃ (x y : ℕ), L % 2 = 0 ∧ W % 5 = 0 ∧ 2 * y = L ∧ 5 * x = W ∧ (L * W) / n = x * y ∧ 2 * (x + y) = 24) :=
by
  intros L W n H1 H2 H3
  use 4, 8
  sorry

end perimeter_of_smaller_rectangle_l1046_104679


namespace given_even_function_and_monotonic_increasing_l1046_104658

-- Define f as an even function on ℝ
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

-- Define that f is monotonically increasing on (-∞, 0)
def is_monotonically_increasing_on_negatives (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < 0 → f (x) < f (y)

-- Theorem statement
theorem given_even_function_and_monotonic_increasing {
  f : ℝ → ℝ
} (h_even : is_even_function f)
  (h_monotonic : is_monotonically_increasing_on_negatives f) :
  f (1) > f (-2) :=
sorry

end given_even_function_and_monotonic_increasing_l1046_104658


namespace possible_to_form_square_l1046_104600

def shape_covers_units : ℕ := 4

theorem possible_to_form_square (shape : ℕ) : ∃ n : ℕ, ∃ k : ℕ, n * n = shape * k :=
by
  use 4
  use 4
  sorry

end possible_to_form_square_l1046_104600


namespace calculate_sum_of_squares_l1046_104659

variables {a b : ℤ}
theorem calculate_sum_of_squares (h1 : (a + b)^2 = 17) (h2 : (a - b)^2 = 11) : a^2 + b^2 = 14 :=
by
  sorry

end calculate_sum_of_squares_l1046_104659


namespace taxi_service_charge_l1046_104606

theorem taxi_service_charge (initial_fee : ℝ) (additional_charge : ℝ) (increment : ℝ) (total_charge : ℝ) 
  (h_initial_fee : initial_fee = 2.25) 
  (h_additional_charge : additional_charge = 0.4) 
  (h_increment : increment = 2 / 5) 
  (h_total_charge : total_charge = 5.85) : 
  ∃ distance : ℝ, distance = 3.6 :=
by
  sorry

end taxi_service_charge_l1046_104606


namespace jogging_time_l1046_104628

theorem jogging_time (distance : ℝ) (speed : ℝ) (h1 : distance = 25) (h2 : speed = 5) : (distance / speed) = 5 :=
by
  rw [h1, h2]
  norm_num

end jogging_time_l1046_104628


namespace find_f_neg4_l1046_104666

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^2 - a * x + b

theorem find_f_neg4 (a b : ℝ) (h1 : f 1 a b = -1) (h2 : f 2 a b = 2) : 
  f (-4) a b = 14 :=
by
  sorry

end find_f_neg4_l1046_104666


namespace polynomial_solution_l1046_104607

variable (P : ℝ → ℝ → ℝ)

theorem polynomial_solution :
  (∀ x y : ℝ, P (x + y) (x - y) = 2 * P x y) →
  (∃ b c d : ℝ, ∀ x y : ℝ, P x y = b * x^2 + c * x * y + d * y^2) :=
by
  sorry

end polynomial_solution_l1046_104607


namespace mineral_age_possibilities_l1046_104636

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def count_permutations_with_repeats (n : ℕ) (repeats : List ℕ) : ℕ :=
  factorial n / List.foldl (· * factorial ·) 1 repeats

theorem mineral_age_possibilities : 
  let digits := [2, 2, 4, 4, 7, 9]
  let odd_digits := [7, 9]
  let remaining_digits := [2, 2, 4, 4]
  2 * count_permutations_with_repeats 5 [2,2] = 60 :=
by
  sorry

end mineral_age_possibilities_l1046_104636


namespace store_loss_90_l1046_104641

theorem store_loss_90 (x y : ℝ) (h1 : x * (1 + 0.12) = 3080) (h2 : y * (1 - 0.12) = 3080) :
  2 * 3080 - x - y = -90 :=
by
  sorry

end store_loss_90_l1046_104641


namespace inverse_of_5_mod_34_l1046_104632

theorem inverse_of_5_mod_34 : ∃ x : ℕ, (5 * x) % 34 = 1 ∧ 0 ≤ x ∧ x < 34 :=
by
  use 7
  have h : (5 * 7) % 34 = 1 := by sorry
  exact ⟨h, by norm_num, by norm_num⟩

end inverse_of_5_mod_34_l1046_104632


namespace megan_popsicles_consumed_l1046_104661

noncomputable def popsicles_consumed_in_time_period (time: ℕ) (interval: ℕ) : ℕ :=
  (time / interval)

theorem megan_popsicles_consumed:
  popsicles_consumed_in_time_period 315 30 = 10 :=
by
  sorry

end megan_popsicles_consumed_l1046_104661


namespace problem_statement_l1046_104680

theorem problem_statement (x1 x2 x3 : ℝ) 
  (h1 : x1 < x2)
  (h2 : x2 < x3)
  (h3 : (45*x1^3 - 4050*x1^2 - 4 = 0) ∧ 
        (45*x2^3 - 4050*x2^2 - 4 = 0) ∧ 
        (45*x3^3 - 4050*x3^2 - 4 = 0)) :
  x2 * (x1 + x3) = 0 :=
by
  sorry

end problem_statement_l1046_104680


namespace opposite_of_negative_2020_is_2020_l1046_104619

theorem opposite_of_negative_2020_is_2020 :
  ∃ x : ℤ, -2020 + x = 0 :=
by
  use 2020
  sorry

end opposite_of_negative_2020_is_2020_l1046_104619


namespace fermat_prime_sum_not_possible_l1046_104610

-- Definitions of the conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, (m ∣ p) → (m = 1 ∨ m = p)

-- The Lean statement
theorem fermat_prime_sum_not_possible 
  (n : ℕ) (x y z : ℤ) (p : ℕ) 
  (h_odd : is_odd n) 
  (h_gt_one : n > 1) 
  (h_prime : is_prime p)
  (h_sum: x + y = ↑p) :
  ¬ (x ^ n + y ^ n = z ^ n) :=
by
  sorry


end fermat_prime_sum_not_possible_l1046_104610


namespace sin_alpha_cos_beta_value_l1046_104667

variables {α β : ℝ}

theorem sin_alpha_cos_beta_value 
  (h1 : Real.sin (α + β) = 1/2) 
  (h2 : 2 * Real.sin (α - β) = 1/2) : 
  Real.sin α * Real.cos β = 3/8 := by
sorry

end sin_alpha_cos_beta_value_l1046_104667


namespace Joe_time_from_home_to_school_l1046_104625

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

end Joe_time_from_home_to_school_l1046_104625


namespace complete_the_square_l1046_104646

theorem complete_the_square (x : ℝ) : (x^2 - 8*x + 15 = 0) → ((x - 4)^2 = 1) :=
by
  intro h
  have eq1 : x^2 - 8*x + 15 = 0 := h
  sorry

end complete_the_square_l1046_104646
