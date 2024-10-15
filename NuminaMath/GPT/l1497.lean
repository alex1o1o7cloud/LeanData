import Mathlib

namespace NUMINAMATH_GPT_remainder_of_M_l1497_149712

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_remainder_of_M_l1497_149712


namespace NUMINAMATH_GPT_find_symmetric_point_l1497_149728

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def line_equation (t : ℝ) : Point :=
  { x := -t, y := 1.5, z := 2 + t }

def M : Point := { x := -1, y := 0, z := -1 }

def is_midpoint (M M' M0 : Point) : Prop :=
  M0.x = (M.x + M'.x) / 2 ∧
  M0.y = (M.y + M'.y) / 2 ∧
  M0.z = (M.z + M'.z) / 2

theorem find_symmetric_point (M0 : Point) (h_line : ∃ t, M0 = line_equation t) :
  ∃ M' : Point, is_midpoint M M' M0 ∧ M' = { x := 3, y := 3, z := 3 } :=
sorry

end NUMINAMATH_GPT_find_symmetric_point_l1497_149728


namespace NUMINAMATH_GPT_bicycle_weight_l1497_149767

theorem bicycle_weight (b s : ℕ) (h1 : 10 * b = 5 * s) (h2 : 5 * s = 200) : b = 20 := 
by 
  sorry

end NUMINAMATH_GPT_bicycle_weight_l1497_149767


namespace NUMINAMATH_GPT_min_people_liking_both_l1497_149756

theorem min_people_liking_both {A B U : Finset ℕ} (hU : U.card = 150) (hA : A.card = 130) (hB : B.card = 120) :
  (A ∩ B).card ≥ 100 :=
by
  -- Proof to be filled later
  sorry

end NUMINAMATH_GPT_min_people_liking_both_l1497_149756


namespace NUMINAMATH_GPT_exists_integers_for_x_squared_minus_y_squared_eq_a_fifth_l1497_149794

theorem exists_integers_for_x_squared_minus_y_squared_eq_a_fifth (a : ℤ) : 
  ∃ x y : ℤ, x^2 - y^2 = a^5 :=
sorry

end NUMINAMATH_GPT_exists_integers_for_x_squared_minus_y_squared_eq_a_fifth_l1497_149794


namespace NUMINAMATH_GPT_find_smallest_int_cube_ends_368_l1497_149792

theorem find_smallest_int_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 500 = 368 ∧ n = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_int_cube_ends_368_l1497_149792


namespace NUMINAMATH_GPT_range_of_a_l1497_149752

theorem range_of_a (f : ℝ → ℝ) (h1 : ∀ x, f (x - 3) = f (3 - (x - 3))) (h2 : ∀ x, 0 ≤ x → f x = x^2 + 2 * x) :
  {a : ℝ | f (2 - a^2) > f a} = {a | -2 < a ∧ a < 1} :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1497_149752


namespace NUMINAMATH_GPT_value_of_expression_l1497_149731

theorem value_of_expression (a b : ℝ) (h : -3 * a - b = -1) : 3 - 6 * a - 2 * b = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1497_149731


namespace NUMINAMATH_GPT_find_dividend_l1497_149757

theorem find_dividend (divisor : ℕ) (partial_quotient : ℕ) (dividend : ℕ) 
                       (h_divisor : divisor = 12)
                       (h_partial_quotient : partial_quotient = 909809) 
                       (h_calculation : dividend = divisor * partial_quotient) : 
                       dividend = 10917708 :=
by
  rw [h_divisor, h_partial_quotient] at h_calculation
  exact h_calculation


end NUMINAMATH_GPT_find_dividend_l1497_149757


namespace NUMINAMATH_GPT_problem1_problem2_l1497_149762

-- Definition for the first proof problem
theorem problem1 (a b : ℝ) (h : a ≠ b) :
  (a^2 / (a - b) - b^2 / (a - b)) = a + b :=
by
  sorry

-- Definition for the second proof problem
theorem problem2 (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  ((x^2 - 1) / ((x^2 + 2 * x + 1)) / (x^2 - x) / (x + 1)) = 1 / x :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1497_149762


namespace NUMINAMATH_GPT_amount_saved_l1497_149769

-- Initial conditions as definitions
def initial_amount : ℕ := 6000
def cost_ballpoint_pen : ℕ := 3200
def cost_eraser : ℕ := 1000
def cost_candy : ℕ := 500

-- Mathematical equivalent proof problem as a Lean theorem statement
theorem amount_saved : initial_amount - (cost_ballpoint_pen + cost_eraser + cost_candy) = 1300 := 
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_amount_saved_l1497_149769


namespace NUMINAMATH_GPT_total_rooms_l1497_149719

-- Definitions for the problem conditions
variables (x y : ℕ)

-- Given conditions
def condition1 : Prop := x = 8
def condition2 : Prop := 2 * x + 3 * y = 31

-- The theorem to prove
theorem total_rooms (h1 : condition1 x) (h2 : condition2 x y) : x + y = 13 :=
by sorry

end NUMINAMATH_GPT_total_rooms_l1497_149719


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1497_149723

variable (p q : Prop)

theorem necessary_but_not_sufficient_condition (hpq : p ∨ q) (h : p ∧ q) : p ∧ q ↔ (p ∨ q) := by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1497_149723


namespace NUMINAMATH_GPT_trapezium_side_length_l1497_149780

theorem trapezium_side_length (a b h A x : ℝ) 
  (ha : a = 20) (hh : h = 15) (hA : A = 285) 
  (h_formula : A = 1 / 2 * (a + b) * h) : 
  b = 18 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_side_length_l1497_149780


namespace NUMINAMATH_GPT_seventh_graders_trip_count_l1497_149727

theorem seventh_graders_trip_count (fifth_graders sixth_graders teachers_per_grade parents_per_grade grades buses seats_per_bus : ℕ) 
  (hf : fifth_graders = 109) 
  (hs : sixth_graders = 115)
  (ht : teachers_per_grade = 4) 
  (hp : parents_per_grade = 2) 
  (hg : grades = 3) 
  (hb : buses = 5)
  (hsb : seats_per_bus = 72) : 
  ∃ seventh_graders : ℕ, seventh_graders = 118 := 
by
  sorry

end NUMINAMATH_GPT_seventh_graders_trip_count_l1497_149727


namespace NUMINAMATH_GPT_lcm_factor_l1497_149753

theorem lcm_factor (A B : ℕ) (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) 
  (hcf_eq : hcf = 15) (factor1_eq : factor1 = 11) (A_eq : A = 225) 
  (hcf_divides_A : hcf ∣ A) (lcm_eq : Nat.lcm A B = hcf * factor1 * factor2) : 
  factor2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_lcm_factor_l1497_149753


namespace NUMINAMATH_GPT_John_days_per_week_l1497_149790

theorem John_days_per_week
    (patients_first : ℕ := 20)
    (patients_increase_rate : ℕ := 20)
    (patients_second : ℕ := (20 + (20 * 20 / 100)))
    (total_weeks_year : ℕ := 50)
    (total_patients_year : ℕ := 11000) :
    ∃ D : ℕ, (20 * D + (20 + (20 * 20 / 100)) * D) * total_weeks_year = total_patients_year ∧ D = 5 := by
  sorry

end NUMINAMATH_GPT_John_days_per_week_l1497_149790


namespace NUMINAMATH_GPT_problem_statement_l1497_149777

theorem problem_statement (m n c d a : ℝ)
  (h1 : m = -n)
  (h2 : c * d = 1)
  (h3 : a = 2) :
  Real.sqrt (c * d) + 2 * (m + n) - a = -1 :=
by
  -- Proof steps are skipped with sorry 
  sorry

end NUMINAMATH_GPT_problem_statement_l1497_149777


namespace NUMINAMATH_GPT_solve_equation_l1497_149759

theorem solve_equation : ∃ x : ℚ, (2*x + 1) / 4 - 1 = x - (10*x + 1) / 12 ∧ x = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1497_149759


namespace NUMINAMATH_GPT_find_initial_candies_l1497_149778

-- Define the initial number of candies as x
def initial_candies (x : ℕ) : ℕ :=
  let first_day := (3 * x) / 4 - 3
  let second_day := (3 * first_day) / 5 - 5
  let third_day := second_day - 7
  let final_candies := (5 * third_day) / 6
  final_candies

-- Formal statement of the theorem
theorem find_initial_candies (x : ℕ) (h : initial_candies x = 10) : x = 44 :=
  sorry

end NUMINAMATH_GPT_find_initial_candies_l1497_149778


namespace NUMINAMATH_GPT_clark_discount_l1497_149706

noncomputable def price_per_part : ℕ := 80
noncomputable def num_parts : ℕ := 7
noncomputable def total_paid : ℕ := 439

theorem clark_discount : (price_per_part * num_parts - total_paid) = 121 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_clark_discount_l1497_149706


namespace NUMINAMATH_GPT_first_part_length_l1497_149713

def total_length : ℝ := 74.5
def part_two : ℝ := 21.5
def part_three : ℝ := 21.5
def part_four : ℝ := 16

theorem first_part_length :
  total_length - (part_two + part_three + part_four) = 15.5 :=
by
  sorry

end NUMINAMATH_GPT_first_part_length_l1497_149713


namespace NUMINAMATH_GPT_original_total_price_l1497_149737

theorem original_total_price (total_selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) 
  (selling_price_with_profit : total_selling_price/2 = original_price * (1 + profit_percent))
  (selling_price_with_loss : total_selling_price/2 = original_price * (1 - loss_percent)) :
  (original_price / (1 + profit_percent) + original_price / (1 - loss_percent) = 1333 + 1 / 3) := 
by
  sorry

end NUMINAMATH_GPT_original_total_price_l1497_149737


namespace NUMINAMATH_GPT_clock_angle_at_7_oclock_l1497_149700

theorem clock_angle_at_7_oclock : 
  ∀ (hour_angle minute_angle : ℝ), 
    (12 : ℝ) * (30 : ℝ) = 360 →
    (7 : ℝ) * (30 : ℝ) = 210 →
    (210 : ℝ) > 180 →
    (360 : ℝ) - (210 : ℝ) = 150 →
    hour_angle = 7 * 30 →
    minute_angle = 0 →
    min (abs (hour_angle - minute_angle)) (abs ((360 - hour_angle) - minute_angle)) = 150 := by
  sorry

end NUMINAMATH_GPT_clock_angle_at_7_oclock_l1497_149700


namespace NUMINAMATH_GPT_dorchester_puppies_washed_l1497_149703

theorem dorchester_puppies_washed
  (total_earnings : ℝ)
  (daily_pay : ℝ)
  (earnings_per_puppy : ℝ)
  (p : ℝ)
  (h1 : total_earnings = 76)
  (h2 : daily_pay = 40)
  (h3 : earnings_per_puppy = 2.25)
  (hp : (total_earnings - daily_pay) / earnings_per_puppy = p) :
  p = 16 := sorry

end NUMINAMATH_GPT_dorchester_puppies_washed_l1497_149703


namespace NUMINAMATH_GPT_ahmed_goats_is_13_l1497_149795

def adam_goats : ℕ := 7

def andrew_goats : ℕ := 2 * adam_goats + 5

def ahmed_goats : ℕ := andrew_goats - 6

theorem ahmed_goats_is_13 : ahmed_goats = 13 :=
by
  sorry

end NUMINAMATH_GPT_ahmed_goats_is_13_l1497_149795


namespace NUMINAMATH_GPT_problem_1_system_solution_problem_2_system_solution_l1497_149772

theorem problem_1_system_solution (x y : ℝ)
  (h1 : x - 2 * y = 1)
  (h2 : 4 * x + 3 * y = 26) :
  x = 5 ∧ y = 2 :=
sorry

theorem problem_2_system_solution (x y : ℝ)
  (h1 : 2 * x + 3 * y = 3)
  (h2 : 5 * x - 3 * y = 18) :
  x = 3 ∧ y = -1 :=
sorry

end NUMINAMATH_GPT_problem_1_system_solution_problem_2_system_solution_l1497_149772


namespace NUMINAMATH_GPT_factor_polynomial_l1497_149733

theorem factor_polynomial {x : ℝ} : 4 * x^3 - 16 * x = 4 * x * (x + 2) * (x - 2) := 
sorry

end NUMINAMATH_GPT_factor_polynomial_l1497_149733


namespace NUMINAMATH_GPT_probability_of_winning_l1497_149721

-- Define the conditions
def total_tickets : ℕ := 10
def winning_tickets : ℕ := 3
def people : ℕ := 5
def losing_tickets : ℕ := total_tickets - winning_tickets

-- The probability calculation as per the conditions
def probability_at_least_one_wins : ℚ :=
  1 - ((Nat.choose losing_tickets people : ℚ) / (Nat.choose total_tickets people))

-- The statement to be proven
theorem probability_of_winning :
  probability_at_least_one_wins = 11 / 12 := 
sorry

end NUMINAMATH_GPT_probability_of_winning_l1497_149721


namespace NUMINAMATH_GPT_identity_holds_l1497_149779

noncomputable def identity_proof : Prop :=
∀ (x : ℝ), (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x

theorem identity_holds : identity_proof :=
by
  sorry

end NUMINAMATH_GPT_identity_holds_l1497_149779


namespace NUMINAMATH_GPT_probability_door_opened_second_attempt_discarded_probability_door_opened_second_attempt_not_discarded_l1497_149771

namespace ProbabilityKeys

-- Define the problem conditions and the probability computations
def keys : ℕ := 4
def successful_keys : ℕ := 2
def unsuccessful_keys : ℕ := 2

def probability_first_fail (k : ℕ) (s : ℕ) : ℚ := (s : ℚ) / (k : ℚ)
def probability_second_success_discarded (s : ℕ) (k : ℕ) : ℚ := (s : ℚ) / (s + 1 - 1: ℚ) 
def probability_second_success_not_discarded (s : ℕ) (k : ℕ) : ℚ := (s : ℚ) / (k : ℚ)

-- The statements to be proved
theorem probability_door_opened_second_attempt_discarded : 
  (probability_first_fail keys successful_keys) * (probability_second_success_discarded unsuccessful_keys keys) = (1 : ℚ) / (3 : ℚ) :=
by sorry

theorem probability_door_opened_second_attempt_not_discarded : 
  (probability_first_fail keys successful_keys) * (probability_second_success_not_discarded successful_keys keys) = (1 : ℚ) / (4 : ℚ) :=
by sorry

end ProbabilityKeys

end NUMINAMATH_GPT_probability_door_opened_second_attempt_discarded_probability_door_opened_second_attempt_not_discarded_l1497_149771


namespace NUMINAMATH_GPT_polynomial_equality_l1497_149764

def P (x : ℝ) : ℝ := x ^ 3 - 3 * x ^ 2 - 3 * x - 1

noncomputable def x1 : ℝ := 1 - Real.sqrt 2
noncomputable def x2 : ℝ := 1 + Real.sqrt 2
noncomputable def x3 : ℝ := 1 - 2 * Real.sqrt 2
noncomputable def x4 : ℝ := 1 + 2 * Real.sqrt 2

theorem polynomial_equality :
  P x1 + P x2 = P x3 + P x4 :=
sorry

end NUMINAMATH_GPT_polynomial_equality_l1497_149764


namespace NUMINAMATH_GPT_baker_earnings_l1497_149735

-- Define the number of cakes and pies sold
def cakes_sold := 453
def pies_sold := 126

-- Define the prices per cake and pie
def price_per_cake := 12
def price_per_pie := 7

-- Calculate the total earnings
def total_earnings : ℕ := (cakes_sold * price_per_cake) + (pies_sold * price_per_pie)

-- Theorem stating the baker's earnings
theorem baker_earnings : total_earnings = 6318 := by
  unfold total_earnings cakes_sold pies_sold price_per_cake price_per_pie
  sorry

end NUMINAMATH_GPT_baker_earnings_l1497_149735


namespace NUMINAMATH_GPT_fraction_of_meat_used_for_meatballs_l1497_149799

theorem fraction_of_meat_used_for_meatballs
    (initial_meat : ℕ)
    (spring_rolls_meat : ℕ)
    (remaining_meat : ℕ)
    (total_meat_used : ℕ)
    (meatballs_meat : ℕ)
    (h_initial : initial_meat = 20)
    (h_spring_rolls : spring_rolls_meat = 3)
    (h_remaining : remaining_meat = 12) :
    (initial_meat - remaining_meat) = total_meat_used ∧
    (total_meat_used - spring_rolls_meat) = meatballs_meat ∧
    (meatballs_meat / initial_meat) = (1/4 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_meat_used_for_meatballs_l1497_149799


namespace NUMINAMATH_GPT_f_seven_l1497_149741

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (h : ℝ) : f (-h) = -f (h)
axiom periodic_function (h : ℝ) : f (h + 4) = f (h)
axiom f_one : f 1 = 2

theorem f_seven : f (7) = -2 :=
by
  sorry

end NUMINAMATH_GPT_f_seven_l1497_149741


namespace NUMINAMATH_GPT_neg_proposition_equiv_l1497_149788

theorem neg_proposition_equiv (p : Prop) : (¬ (∃ n : ℕ, 2^n > 1000)) = (∀ n : ℕ, 2^n ≤ 1000) :=
by
  sorry

end NUMINAMATH_GPT_neg_proposition_equiv_l1497_149788


namespace NUMINAMATH_GPT_find_B_l1497_149789

theorem find_B : 
  ∀ (A B : ℕ), A ≤ 9 → B ≤ 9 → (600 + 10 * A + 5) + (100 + B) = 748 → B = 3 :=
by
  intros A B hA hB hEq
  sorry

end NUMINAMATH_GPT_find_B_l1497_149789


namespace NUMINAMATH_GPT_percentage_people_taking_bus_l1497_149750

-- Definitions
def population := 80
def car_pollution := 10 -- pounds of carbon per car per year
def bus_pollution := 100 -- pounds of carbon per bus per year
def bus_capacity := 40 -- people per bus
def carbon_reduction := 100 -- pounds of carbon reduced per year after the bus is introduced

-- Problem statement in Lean 4
theorem percentage_people_taking_bus :
  (10 / 80 : ℝ) = 0.125 :=
by
  sorry

end NUMINAMATH_GPT_percentage_people_taking_bus_l1497_149750


namespace NUMINAMATH_GPT_connected_graphs_bound_l1497_149758

noncomputable def num_connected_graphs (n : ℕ) : ℕ := sorry
  
theorem connected_graphs_bound (n : ℕ) : 
  num_connected_graphs n ≥ (1/2) * 2^(n*(n-1)/2) := 
sorry

end NUMINAMATH_GPT_connected_graphs_bound_l1497_149758


namespace NUMINAMATH_GPT_polygon_num_sides_l1497_149787

theorem polygon_num_sides (s : ℕ) (h : 180 * (s - 2) > 2790) : s = 18 :=
sorry

end NUMINAMATH_GPT_polygon_num_sides_l1497_149787


namespace NUMINAMATH_GPT_marie_needs_8_days_to_pay_for_cash_register_l1497_149738

-- Definitions of the conditions
def cost_of_cash_register : ℕ := 1040
def price_per_loaf : ℕ := 2
def loaves_per_day : ℕ := 40
def price_per_cake : ℕ := 12
def cakes_per_day : ℕ := 6
def daily_rent : ℕ := 20
def daily_electricity : ℕ := 2

-- Derive daily income and expenses
def daily_income : ℕ := (price_per_loaf * loaves_per_day) + (price_per_cake * cakes_per_day)
def daily_expenses : ℕ := daily_rent + daily_electricity
def daily_profit : ℕ := daily_income - daily_expenses

-- Define days needed to pay for the cash register
def days_needed : ℕ := cost_of_cash_register / daily_profit

-- Proof goal
theorem marie_needs_8_days_to_pay_for_cash_register : days_needed = 8 := by
  sorry

end NUMINAMATH_GPT_marie_needs_8_days_to_pay_for_cash_register_l1497_149738


namespace NUMINAMATH_GPT_line_symmetric_fixed_point_l1497_149705

theorem line_symmetric_fixed_point (k : ℝ) :
  (∀ x, (∃ y, y = k * (x - 4))) →
  (∃ p : ℝ × ℝ, p = (2, 1) ∧ ∀ x, (∃ y, y = k * (x - 4))) →
  (∃ p : ℝ × ℝ, p = (2, 1)) →
  (∃ q : ℝ × ℝ, q = (0, 2)) →
  True := 
by sorry

end NUMINAMATH_GPT_line_symmetric_fixed_point_l1497_149705


namespace NUMINAMATH_GPT_bird_migration_difference_correct_l1497_149754

def bird_migration_difference : ℕ := 54

/--
There are 250 bird families consisting of 3 different bird species, each with varying migration patterns.

Species A: 100 bird families; 35% fly to Africa, 65% fly to Asia
Species B: 120 bird families; 50% fly to Africa, 50% fly to Asia
Species C: 30 bird families; 10% fly to Africa, 90% fly to Asia

Prove that the difference in the number of bird families migrating to Asia and Africa is 54.
-/
theorem bird_migration_difference_correct (A_Africa_percent : ℕ := 35) (A_Asia_percent : ℕ := 65)
  (B_Africa_percent : ℕ := 50) (B_Asia_percent : ℕ := 50)
  (C_Africa_percent : ℕ := 10) (C_Asia_percent : ℕ := 90)
  (A_count : ℕ := 100) (B_count : ℕ := 120) (C_count : ℕ := 30) :
    bird_migration_difference = 
      (A_count * A_Asia_percent / 100 + B_count * B_Asia_percent / 100 + C_count * C_Asia_percent / 100) - 
      (A_count * A_Africa_percent / 100 + B_count * B_Africa_percent / 100 + C_count * C_Africa_percent / 100) :=
by sorry

end NUMINAMATH_GPT_bird_migration_difference_correct_l1497_149754


namespace NUMINAMATH_GPT_polygon_angles_change_l1497_149720

theorem polygon_angles_change (n : ℕ) :
  let initial_sum_interior := (n - 2) * 180
  let initial_sum_exterior := 360
  let new_sum_interior := (n + 2 - 2) * 180
  let new_sum_exterior := 360
  new_sum_exterior = initial_sum_exterior ∧ new_sum_interior - initial_sum_interior = 360 :=
by
  sorry

end NUMINAMATH_GPT_polygon_angles_change_l1497_149720


namespace NUMINAMATH_GPT_more_boys_than_girls_l1497_149716

theorem more_boys_than_girls (total_people : ℕ) (num_girls : ℕ) (num_boys : ℕ) (more_boys : ℕ) : 
  total_people = 133 ∧ num_girls = 50 ∧ num_boys = total_people - num_girls ∧ more_boys = num_boys - num_girls → more_boys = 33 :=
by 
  sorry

end NUMINAMATH_GPT_more_boys_than_girls_l1497_149716


namespace NUMINAMATH_GPT_maximize_Miraflores_win_l1497_149729

-- Definitions based on given conditions
def voters_count (n : ℕ) : ℕ := 2 * n
def support_Miraflores (n : ℕ) : ℕ := n + 1
def support_opponent (n : ℕ) : ℕ := n - 1

-- Theorem statement
theorem maximize_Miraflores_win (n : ℕ) (hn : n > 0) : 
  ∃ (d1 d2 : ℕ), d1 = 1 ∧ d2 = 2 * n - 1 ∧ support_Miraflores n > support_opponent n := 
sorry

end NUMINAMATH_GPT_maximize_Miraflores_win_l1497_149729


namespace NUMINAMATH_GPT_num_tables_l1497_149774

/-- Given conditions related to tables, stools, and benches, we want to prove the number of tables -/
theorem num_tables 
  (t s b : ℕ) 
  (h1 : s = 8 * t)
  (h2 : b = 2 * t)
  (h3 : 3 * s + 6 * b + 4 * t = 816) : 
  t = 20 := 
sorry

end NUMINAMATH_GPT_num_tables_l1497_149774


namespace NUMINAMATH_GPT_log_relation_l1497_149770

noncomputable def a := Real.log 3 / Real.log 4
noncomputable def b := Real.log 3 / Real.log 0.4
def c := (1 / 2) ^ 2

theorem log_relation (h1 : a = Real.log 3 / Real.log 4)
                     (h2 : b = Real.log 3 / Real.log 0.4)
                     (h3 : c = (1 / 2) ^ 2) : a > c ∧ c > b :=
by
  sorry

end NUMINAMATH_GPT_log_relation_l1497_149770


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1497_149730

variable (x : ℝ)

theorem sufficient_but_not_necessary : (x = 1) → (x^3 = x) ∧ (∀ y, y^3 = y → y = 1 → x ≠ y) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1497_149730


namespace NUMINAMATH_GPT_bird_height_l1497_149709

theorem bird_height (cat_height dog_height avg_height : ℕ) 
  (cat_height_eq : cat_height = 92)
  (dog_height_eq : dog_height = 94)
  (avg_height_eq : avg_height = 95) :
  let total_height := avg_height * 3 
  let bird_height := total_height - (cat_height + dog_height)
  bird_height = 99 := 
by
  sorry

end NUMINAMATH_GPT_bird_height_l1497_149709


namespace NUMINAMATH_GPT_cost_price_to_selling_price_ratio_l1497_149784

variable (CP SP : ℝ)
variable (profit_percent : ℝ)

theorem cost_price_to_selling_price_ratio
  (h1 : profit_percent = 0.25)
  (h2 : SP = (1 + profit_percent) * CP) :
  (CP / SP) = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_cost_price_to_selling_price_ratio_l1497_149784


namespace NUMINAMATH_GPT_piravena_total_round_trip_cost_l1497_149717

noncomputable def piravena_round_trip_cost : ℝ :=
  let distance_AB := 4000
  let bus_cost_per_km := 0.20
  let flight_cost_per_km := 0.12
  let flight_booking_fee := 120
  let flight_cost := distance_AB * flight_cost_per_km + flight_booking_fee
  let bus_cost := distance_AB * bus_cost_per_km
  flight_cost + bus_cost

theorem piravena_total_round_trip_cost : piravena_round_trip_cost = 1400 := by
  -- Problem conditions for reference:
  -- distance_AC = 3000
  -- distance_AB = 4000
  -- bus_cost_per_km = 0.20
  -- flight_cost_per_km = 0.12
  -- flight_booking_fee = 120
  -- Piravena decides to fly from A to B but returns by bus
  sorry

end NUMINAMATH_GPT_piravena_total_round_trip_cost_l1497_149717


namespace NUMINAMATH_GPT_delta_comparison_eps_based_on_gamma_l1497_149765

-- Definitions for the problem
variable {α β γ δ ε : ℝ}
variable {A B C : Type}
variable (s f m : Type)

-- Conditions from problem
variable (triangle_ABC : α ≠ β)
variable (median_s_from_C : s)
variable (angle_bisector_f : f)
variable (altitude_m : m)
variable (angle_between_f_m : δ = sorry)
variable (angle_between_f_s : ε = sorry)
variable (angle_at_vertex_C : γ = sorry)

-- Main statement to prove
theorem delta_comparison_eps_based_on_gamma (h1 : α ≠ β) (h2 : δ = sorry) (h3 : ε = sorry) (h4 : γ = sorry) :
  if γ < 90 then δ < ε else if γ = 90 then δ = ε else δ > ε :=
sorry

end NUMINAMATH_GPT_delta_comparison_eps_based_on_gamma_l1497_149765


namespace NUMINAMATH_GPT_painters_complete_three_rooms_in_three_hours_l1497_149708

theorem painters_complete_three_rooms_in_three_hours :
  ∃ P, (∀ (P : ℕ), (P * 3) = 3) ∧ (9 * 9 = 27) → P = 3 := by
  sorry

end NUMINAMATH_GPT_painters_complete_three_rooms_in_three_hours_l1497_149708


namespace NUMINAMATH_GPT_initial_horses_to_cows_ratio_l1497_149798

theorem initial_horses_to_cows_ratio (H C : ℕ) (h₁ : (H - 15) / (C + 15) = 13 / 7) (h₂ : H - 15 = C + 45) :
  H / C = 4 / 1 := 
sorry

end NUMINAMATH_GPT_initial_horses_to_cows_ratio_l1497_149798


namespace NUMINAMATH_GPT_find_smallest_n_l1497_149740

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_n_l1497_149740


namespace NUMINAMATH_GPT_total_price_correct_l1497_149783

-- Definitions of given conditions
def original_price : Float := 120
def discount_rate : Float := 0.30
def tax_rate : Float := 0.08

-- Definition of the final selling price
def sale_price : Float := original_price * (1 - discount_rate)
def total_selling_price : Float := sale_price * (1 + tax_rate)

-- Lean 4 statement to prove the total selling price is 90.72
theorem total_price_correct : total_selling_price = 90.72 := by
  sorry

end NUMINAMATH_GPT_total_price_correct_l1497_149783


namespace NUMINAMATH_GPT_no_nontrivial_solutions_in_integers_l1497_149743

theorem no_nontrivial_solutions_in_integers (a b c n : ℤ) 
  (h : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
  by
    sorry

end NUMINAMATH_GPT_no_nontrivial_solutions_in_integers_l1497_149743


namespace NUMINAMATH_GPT_river_depth_l1497_149711

theorem river_depth (width depth : ℝ) (flow_rate_kmph : ℝ) (volume_m3_per_min : ℝ) 
  (h1 : width = 75) 
  (h2 : flow_rate_kmph = 4) 
  (h3 : volume_m3_per_min = 35000) : 
  depth = 7 := 
by
  sorry

end NUMINAMATH_GPT_river_depth_l1497_149711


namespace NUMINAMATH_GPT_initial_water_amount_l1497_149746

variable (W : ℝ)
variable (evap_per_day : ℝ := 0.014)
variable (days : ℕ := 50)
variable (evap_percent : ℝ := 7.000000000000001)

theorem initial_water_amount :
  evap_per_day * (days : ℝ) = evap_percent / 100 * W → W = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_water_amount_l1497_149746


namespace NUMINAMATH_GPT_calculate_opening_price_l1497_149732

theorem calculate_opening_price (C : ℝ) (r : ℝ) (P : ℝ) 
  (h1 : C = 15)
  (h2 : r = 0.5)
  (h3 : C = P + r * P) :
  P = 10 :=
by sorry

end NUMINAMATH_GPT_calculate_opening_price_l1497_149732


namespace NUMINAMATH_GPT_highest_nitrogen_percentage_l1497_149748

-- Define molar masses for each compound
def molar_mass_NH2OH : Float := 33.0
def molar_mass_NH4NO2 : Float := 64.1 
def molar_mass_N2O3 : Float := 76.0
def molar_mass_NH4NH2CO2 : Float := 78.1

-- Define mass of nitrogen atoms
def mass_of_nitrogen : Float := 14.0

-- Define the percentage calculations
def percentage_NH2OH : Float := (mass_of_nitrogen / molar_mass_NH2OH) * 100.0
def percentage_NH4NO2 : Float := (2 * mass_of_nitrogen / molar_mass_NH4NO2) * 100.0
def percentage_N2O3 : Float := (2 * mass_of_nitrogen / molar_mass_N2O3) * 100.0
def percentage_NH4NH2CO2 : Float := (2 * mass_of_nitrogen / molar_mass_NH4NH2CO2) * 100.0

-- Define the proof problem
theorem highest_nitrogen_percentage : percentage_NH4NO2 > percentage_NH2OH ∧
                                      percentage_NH4NO2 > percentage_N2O3 ∧
                                      percentage_NH4NO2 > percentage_NH4NH2CO2 :=
by 
  sorry

end NUMINAMATH_GPT_highest_nitrogen_percentage_l1497_149748


namespace NUMINAMATH_GPT_find_m_of_odd_function_l1497_149742

theorem find_m_of_odd_function (m : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = ((x + 3) * (x + m)) / x)
  (h₂ : ∀ x, f (-x) = -f x) : m = -3 :=
sorry

end NUMINAMATH_GPT_find_m_of_odd_function_l1497_149742


namespace NUMINAMATH_GPT_sfl_entrances_l1497_149796

theorem sfl_entrances (people_per_entrance total_people entrances : ℕ) 
  (h1: people_per_entrance = 283) 
  (h2: total_people = 1415) 
  (h3: total_people = people_per_entrance * entrances) 
  : entrances = 5 := 
  by 
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_sfl_entrances_l1497_149796


namespace NUMINAMATH_GPT_opposite_event_is_at_least_one_hit_l1497_149782

def opposite_event_of_missing_both_times (hit1 hit2 : Prop) : Prop :=
  ¬(¬hit1 ∧ ¬hit2)

theorem opposite_event_is_at_least_one_hit (hit1 hit2 : Prop) :
  opposite_event_of_missing_both_times hit1 hit2 = (hit1 ∨ hit2) :=
by
  sorry

end NUMINAMATH_GPT_opposite_event_is_at_least_one_hit_l1497_149782


namespace NUMINAMATH_GPT_factory_needs_to_produce_l1497_149718

-- Define the given conditions
def weekly_production_target : ℕ := 6500
def production_mon_tue_wed : ℕ := 3 * 1200
def production_thu : ℕ := 800
def total_production_mon_thu := production_mon_tue_wed + production_thu
def required_production_fri := weekly_production_target - total_production_mon_thu

-- The theorem we need to prove
theorem factory_needs_to_produce : required_production_fri = 2100 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_factory_needs_to_produce_l1497_149718


namespace NUMINAMATH_GPT_volume_of_remaining_solid_after_removing_tetrahedra_l1497_149760

theorem volume_of_remaining_solid_after_removing_tetrahedra :
  let cube_volume := 1
  let tetrahedron_volume := (1/3) * (1/2) * (1/2) * (1/2) * (1/2)
  cube_volume - 8 * tetrahedron_volume = 5 / 6 := by
  let cube_volume := 1
  let tetrahedron_volume := (1/3) * (1/2) * (1/2) * (1/2) * (1/2)
  have h : cube_volume - 8 * tetrahedron_volume = 5 / 6 := sorry
  exact h

end NUMINAMATH_GPT_volume_of_remaining_solid_after_removing_tetrahedra_l1497_149760


namespace NUMINAMATH_GPT_find_k_l1497_149745

open Real

-- Define the operation "※"
def star (a b : ℝ) : ℝ := a * b + a + b^2

-- Define the main theorem stating the problem
theorem find_k (k : ℝ) (hk : k > 0) (h : star 1 k = 3) : k = 1 := by
  sorry

end NUMINAMATH_GPT_find_k_l1497_149745


namespace NUMINAMATH_GPT_parallel_vectors_eq_l1497_149785

theorem parallel_vectors_eq (t : ℝ) : ∀ (m n : ℝ × ℝ), m = (2, 8) → n = (-4, t) → (∃ k : ℝ, n = k • m) → t = -16 :=
by 
  intros m n hm hn h_parallel
  -- proof goes here
  sorry

end NUMINAMATH_GPT_parallel_vectors_eq_l1497_149785


namespace NUMINAMATH_GPT_no_nonzero_integer_solution_l1497_149755

theorem no_nonzero_integer_solution (x y z : ℤ) (h : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :
  x^2 + y^2 ≠ 3 * z^2 :=
by
  sorry

end NUMINAMATH_GPT_no_nonzero_integer_solution_l1497_149755


namespace NUMINAMATH_GPT_negation_of_proposition_l1497_149734

theorem negation_of_proposition :
  (∀ x y : ℝ, (x * y = 0 → x = 0 ∨ y = 0)) →
  (∃ x y : ℝ, x * y = 0 ∧ x ≠ 0 ∧ y ≠ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1497_149734


namespace NUMINAMATH_GPT_gerbil_weights_l1497_149763

theorem gerbil_weights
  (puffy muffy scruffy fluffy tuffy : ℕ)
  (h1 : puffy = 2 * muffy)
  (h2 : muffy = scruffy - 3)
  (h3 : scruffy = 12)
  (h4 : fluffy = muffy + tuffy)
  (h5 : fluffy = puffy / 2)
  (h6 : tuffy = puffy / 2) :
  puffy + muffy + tuffy = 36 := by
  sorry

end NUMINAMATH_GPT_gerbil_weights_l1497_149763


namespace NUMINAMATH_GPT_average_speed_trip_l1497_149715

theorem average_speed_trip 
  (total_distance : ℕ)
  (first_distance : ℕ)
  (first_speed : ℕ)
  (second_distance : ℕ)
  (second_speed : ℕ)
  (h1 : total_distance = 60)
  (h2 : first_distance = 30)
  (h3 : first_speed = 60)
  (h4 : second_distance = 30)
  (h5 : second_speed = 30) :
  40 = total_distance / ((first_distance / first_speed) + (second_distance / second_speed)) :=
by sorry

end NUMINAMATH_GPT_average_speed_trip_l1497_149715


namespace NUMINAMATH_GPT_fuel_a_added_l1497_149749

theorem fuel_a_added (capacity : ℝ) (ethanolA : ℝ) (ethanolB : ℝ) (total_ethanol : ℝ) (x : ℝ) : 
  capacity = 200 ∧ ethanolA = 0.12 ∧ ethanolB = 0.16 ∧ total_ethanol = 28 →
  0.12 * x + 0.16 * (200 - x) = 28 → x = 100 :=
sorry

end NUMINAMATH_GPT_fuel_a_added_l1497_149749


namespace NUMINAMATH_GPT_probability_of_hitting_target_at_least_once_l1497_149776

theorem probability_of_hitting_target_at_least_once :
  (∀ (p1 p2 : ℝ), p1 = 0.5 → p2 = 0.7 → (1 - (1 - p1) * (1 - p2)) = 0.85) :=
by
  intros p1 p2 h1 h2
  rw [h1, h2]
  -- This rw step simplifies (1 - (1 - 0.5) * (1 - 0.7)) to the desired result.
  sorry

end NUMINAMATH_GPT_probability_of_hitting_target_at_least_once_l1497_149776


namespace NUMINAMATH_GPT_total_spent_on_index_cards_l1497_149701

-- Definitions for conditions
def index_cards_per_student : ℕ := 10
def periods_per_day : ℕ := 6
def students_per_class : ℕ := 30
def cost_per_pack : ℕ := 3
def cards_per_pack : ℕ := 50

-- Theorem to be proven
theorem total_spent_on_index_cards :
  let total_students := students_per_class * periods_per_day
  let total_cards := total_students * index_cards_per_student
  let packs_needed := total_cards / cards_per_pack
  let total_cost := packs_needed * cost_per_pack
  total_cost = 108 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_on_index_cards_l1497_149701


namespace NUMINAMATH_GPT_correct_statement_C_l1497_149797

-- Define the function
def linear_function (x : ℝ) : ℝ := -3 * x + 1

-- Define the condition for statement C
def statement_C (x : ℝ) : Prop := x > 1 / 3 → linear_function x < 0

-- The theorem to be proved
theorem correct_statement_C : ∀ x : ℝ, statement_C x := by
  sorry

end NUMINAMATH_GPT_correct_statement_C_l1497_149797


namespace NUMINAMATH_GPT_james_total_points_l1497_149786

def points_per_correct_answer : ℕ := 2
def bonus_points_per_round : ℕ := 4
def total_rounds : ℕ := 5
def questions_per_round : ℕ := 5
def total_questions : ℕ := total_rounds * questions_per_round
def questions_missed_by_james : ℕ := 1
def questions_answered_by_james : ℕ := total_questions - questions_missed_by_james
def points_for_correct_answers : ℕ := questions_answered_by_james * points_per_correct_answer
def complete_rounds_by_james : ℕ := total_rounds - 1  -- Since James missed one question, he has 4 complete rounds
def bonus_points_by_james : ℕ := complete_rounds_by_james * bonus_points_per_round
def total_points : ℕ := points_for_correct_answers + bonus_points_by_james

theorem james_total_points : total_points = 64 := by
  sorry

end NUMINAMATH_GPT_james_total_points_l1497_149786


namespace NUMINAMATH_GPT_description_of_T_l1497_149725

def T : Set (ℝ × ℝ) := { p | ∃ c, (4 = p.1 + 3 ∨ 4 = p.2 - 2 ∨ p.1 + 3 = p.2 - 2) 
                           ∧ (p.1 + 3 ≤ c ∨ p.2 - 2 ≤ c ∨ 4 ≤ c) }

theorem description_of_T : 
  (∀ p ∈ T, (∃ x y : ℝ, p = (x, y) ∧ ((x = 1 ∧ y ≤ 6) ∨ (y = 6 ∧ x ≤ 1) ∨ (y = x + 5 ∧ x ≥ 1 ∧ y ≥ 6)))) :=
sorry

end NUMINAMATH_GPT_description_of_T_l1497_149725


namespace NUMINAMATH_GPT_f_zero_eq_one_positive_for_all_x_l1497_149766

variables {R : Type*} [LinearOrderedField R] (f : R → R)

-- Conditions
axiom domain (x : R) : true -- This translates that f has domain (-∞, ∞)
axiom non_constant (x1 x2 : R) (h : x1 ≠ x2) : f x1 ≠ f x2
axiom functional_eq (x y : R) : f (x + y) = f x * f y

-- Questions
theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem positive_for_all_x (x : R) : f x > 0 :=
sorry

end NUMINAMATH_GPT_f_zero_eq_one_positive_for_all_x_l1497_149766


namespace NUMINAMATH_GPT_find_number_of_children_l1497_149707

theorem find_number_of_children (C B : ℕ) (H1 : B = 2 * C) (H2 : B = 4 * (C - 360)) : C = 720 := 
by
  sorry

end NUMINAMATH_GPT_find_number_of_children_l1497_149707


namespace NUMINAMATH_GPT_repeating_decimal_sum_as_fraction_l1497_149773

theorem repeating_decimal_sum_as_fraction :
  let d1 := 1 / 9    -- Representation of 0.\overline{1}
  let d2 := 1 / 99   -- Representation of 0.\overline{01}
  d1 + d2 = (4 : ℚ) / 33 := by
{
  sorry
}

end NUMINAMATH_GPT_repeating_decimal_sum_as_fraction_l1497_149773


namespace NUMINAMATH_GPT_saly_needs_10_eggs_per_week_l1497_149751

theorem saly_needs_10_eggs_per_week :
  let Saly_needs_per_week := S
  let Ben_needs_per_week := 14
  let Ked_needs_per_week := Ben_needs_per_week / 2
  let total_eggs_in_month := 124
  let weeks_per_month := 4
  let Ben_needs_per_month := Ben_needs_per_week * weeks_per_month
  let Ked_needs_per_month := Ked_needs_per_week * weeks_per_month
  let Saly_needs_per_month := total_eggs_in_month - (Ben_needs_per_month + Ked_needs_per_month)
  let S := Saly_needs_per_month / weeks_per_month
  Saly_needs_per_week = 10 :=
by
  sorry

end NUMINAMATH_GPT_saly_needs_10_eggs_per_week_l1497_149751


namespace NUMINAMATH_GPT_marys_number_l1497_149704

theorem marys_number (j m : ℕ) (h₁ : j * m = 2002)
  (h₂ : ∃ k, k * m = 2002 ∧ k ≠ j)
  (h₃ : ∃ l, j * l = 2002 ∧ l ≠ m) :
  m = 1001 :=
sorry

end NUMINAMATH_GPT_marys_number_l1497_149704


namespace NUMINAMATH_GPT_twin_primes_divisible_by_12_l1497_149739

def isTwinPrime (p q : ℕ) : Prop :=
  p < q ∧ Nat.Prime p ∧ Nat.Prime q ∧ q - p = 2

theorem twin_primes_divisible_by_12 {p q r s : ℕ} 
  (h1 : isTwinPrime p q) 
  (h2 : p > 3) 
  (h3 : isTwinPrime r s) 
  (h4 : r > 3) :
  12 ∣ (p * r - q * s) := by
  sorry

end NUMINAMATH_GPT_twin_primes_divisible_by_12_l1497_149739


namespace NUMINAMATH_GPT_total_number_of_possible_outcomes_l1497_149768

-- Define the conditions
def num_faces_per_die : ℕ := 6
def num_dice : ℕ := 2

-- Define the question as a hypothesis and the answer as the conclusion
theorem total_number_of_possible_outcomes :
  (num_faces_per_die * num_faces_per_die) = 36 := 
by
  -- Provide a proof outline, this is used to skip the actual proof
  sorry

end NUMINAMATH_GPT_total_number_of_possible_outcomes_l1497_149768


namespace NUMINAMATH_GPT_anna_score_below_90_no_A_l1497_149722

def score_implies_grade (score : ℝ) : Prop :=
  score > 90 → true

theorem anna_score_below_90_no_A (score : ℝ) (A_grade : Prop) (h : score_implies_grade score) :
  score < 90 → ¬ A_grade :=
by sorry

end NUMINAMATH_GPT_anna_score_below_90_no_A_l1497_149722


namespace NUMINAMATH_GPT_sequence_contains_perfect_square_l1497_149761

noncomputable def f (n : ℕ) : ℕ := n + Nat.floor (Real.sqrt n)

theorem sequence_contains_perfect_square (m : ℕ) : ∃ k : ℕ, ∃ p : ℕ, f^[k] m = p * p := by
  sorry

end NUMINAMATH_GPT_sequence_contains_perfect_square_l1497_149761


namespace NUMINAMATH_GPT_real_part_fraction_l1497_149781

theorem real_part_fraction {i : ℂ} (h : i^2 = -1) : (
  let numerator := 1 - i
  let denominator := (1 + i) ^ 2
  let fraction := numerator / denominator
  let real_part := (fraction.re)
  real_part
) = -1/2 := sorry

end NUMINAMATH_GPT_real_part_fraction_l1497_149781


namespace NUMINAMATH_GPT_sequence_positive_and_divisible_l1497_149775

theorem sequence_positive_and_divisible:
  ∃ (a : ℕ → ℕ), 
    (a 1 = 2) ∧ (a 2 = 500) ∧ (a 3 = 2000) ∧ 
    (∀ n ≥ 2, (a (n + 2) + a (n + 1)) * a (n - 1) = a (n + 1) * (a (n + 1) + a (n - 1))) ∧ 
    (∀ n, a n > 0) ∧ 
    (2 ^ 2000 ∣ a 2000) := 
sorry

end NUMINAMATH_GPT_sequence_positive_and_divisible_l1497_149775


namespace NUMINAMATH_GPT_initial_blueberry_jelly_beans_l1497_149724

-- Definitions for initial numbers of jelly beans and modified quantities after eating
variables (b c : ℕ)

-- Conditions stated as Lean hypothesis
axiom initial_relation : b = 2 * c
axiom new_relation : b - 5 = 4 * (c - 5)

-- Theorem statement to prove the initial number of blueberry jelly beans is 30
theorem initial_blueberry_jelly_beans : b = 30 :=
by
  sorry

end NUMINAMATH_GPT_initial_blueberry_jelly_beans_l1497_149724


namespace NUMINAMATH_GPT_least_possible_z_minus_x_l1497_149726

theorem least_possible_z_minus_x (x y z : ℤ) (h₁ : x < y) (h₂ : y < z) (h₃ : y - x > 11) 
  (h₄ : Even x) (h₅ : Odd y) (h₆ : Odd z) : z - x = 15 :=
sorry

end NUMINAMATH_GPT_least_possible_z_minus_x_l1497_149726


namespace NUMINAMATH_GPT_john_drive_time_l1497_149791

theorem john_drive_time
  (t : ℝ)
  (h1 : 60 * t + 90 * (15 / 4 - t) = 300)
  (h2 : 1 / 4 = 15 / 60)
  (h3 : 4 = 15 / 4 + t + 1 / 4)
  :
  t = 1.25 :=
by
  -- This introduces the hypothesis and begins the Lean proof.
  sorry

end NUMINAMATH_GPT_john_drive_time_l1497_149791


namespace NUMINAMATH_GPT_first_term_correct_l1497_149710

noncomputable def first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^3 / (1 - (r^3)) = 80) : ℝ :=
a

theorem first_term_correct (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^3 / (1 - (r^3)) = 80) :
  first_term a r h1 h2 = 3.42 :=
sorry

end NUMINAMATH_GPT_first_term_correct_l1497_149710


namespace NUMINAMATH_GPT_problem_statement_l1497_149793

-- Define that the function f is even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define that the function f satisfies f(x) = f(2 - x)
def satisfies_symmetry (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

-- Define that the function f is decreasing on a given interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Define that the function f is increasing on a given interval
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Given hypotheses and the theorem to prove. We use two statements for clarity.
theorem problem_statement (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_symmetry : satisfies_symmetry f) 
  (h_decreasing_1_2 : is_decreasing_on f 1 2) : 
  is_increasing_on f (-2) (-1) ∧ is_decreasing_on f 3 4 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1497_149793


namespace NUMINAMATH_GPT_common_difference_d_l1497_149736

open Real

-- Define the arithmetic sequence and relevant conditions
variable (a : ℕ → ℝ) -- Define the sequence as a function from natural numbers to real numbers
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific conditions from our problem
def problem_conditions (a : ℕ → ℝ) (d : ℝ) : Prop :=
  is_arithmetic_sequence a d ∧
  a 1 = 1 ∧
  (a 2) ^ 2 = a 1 * a 6

-- The goal is to prove that the common difference d is either 0 or 3
theorem common_difference_d (a : ℕ → ℝ) (d : ℝ) :
  problem_conditions a d → (d = 0 ∨ d = 3) := by
  sorry

end NUMINAMATH_GPT_common_difference_d_l1497_149736


namespace NUMINAMATH_GPT_triangle_third_side_length_l1497_149714

theorem triangle_third_side_length (x: ℕ) (h1: x % 2 = 0) (h2: 2 + 14 > x) (h3: 14 - 2 < x) : x = 14 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_third_side_length_l1497_149714


namespace NUMINAMATH_GPT_trip_cost_l1497_149747

theorem trip_cost (original_price : ℕ) (discount : ℕ) (num_people : ℕ)
  (h1 : original_price = 147) (h2 : discount = 14) (h3 : num_people = 2) :
  num_people * (original_price - discount) = 266 :=
by
  sorry

end NUMINAMATH_GPT_trip_cost_l1497_149747


namespace NUMINAMATH_GPT_events_A_B_mutually_exclusive_events_A_C_independent_l1497_149744

-- Definitions for events A, B, and C
def event_A (x y : ℕ) : Prop := x + y = 7
def event_B (x y : ℕ) : Prop := (x * y) % 2 = 1
def event_C (x : ℕ) : Prop := x > 3

-- Proof problems to decide mutual exclusivity and independence
theorem events_A_B_mutually_exclusive :
  ∀ (x y : ℕ), event_A x y → ¬ event_B x y := 
by sorry

theorem events_A_C_independent :
  ∀ (x y : ℕ), (event_A x y) ↔ ∀ x y, event_C x ↔ event_A x y ∧ event_C x := 
by sorry

end NUMINAMATH_GPT_events_A_B_mutually_exclusive_events_A_C_independent_l1497_149744


namespace NUMINAMATH_GPT_time_spent_on_type_a_problems_l1497_149702

theorem time_spent_on_type_a_problems 
  (total_problems : ℕ)
  (exam_time_minutes : ℕ)
  (type_a_problems : ℕ)
  (type_b_problem_time : ℕ)
  (total_time_type_a : ℕ)
  (h1 : total_problems = 200)
  (h2 : exam_time_minutes = 180)
  (h3 : type_a_problems = 50)
  (h4 : ∀ x : ℕ, type_b_problem_time = 2 * x)
  (h5 : ∀ x : ℕ, total_time_type_a = type_a_problems * type_b_problem_time)
  : total_time_type_a = 72 := 
by
  sorry

end NUMINAMATH_GPT_time_spent_on_type_a_problems_l1497_149702
