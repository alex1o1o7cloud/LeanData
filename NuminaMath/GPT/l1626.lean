import Mathlib

namespace NUMINAMATH_GPT_units_digit_17_pow_2107_l1626_162605

theorem units_digit_17_pow_2107 : (17 ^ 2107) % 10 = 3 := by
  -- Definitions derived from conditions:
  -- 1. Powers of 17 have the same units digit as the corresponding powers of 7.
  -- 2. Units digits of powers of 7 cycle: 7, 9, 3, 1.
  -- 3. 2107 modulo 4 gives remainder 3.
  sorry

end NUMINAMATH_GPT_units_digit_17_pow_2107_l1626_162605


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l1626_162612

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |x - 1|

theorem part1_solution :
  {x : ℝ | f x > 4} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x} :=
by
  sorry

theorem part2_solution (x0 : ℝ) :
  (∃ x0 : ℝ, ∀ t : ℝ, f x0 < |(x0 + t)| + |(t - x0)|) →
  ∀ m : ℝ, (f x0 < |m + t| + |t - m|) ↔ m ≠ 0 ∧ (|m| > 5 / 4) :=
by
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l1626_162612


namespace NUMINAMATH_GPT_sqrt_of_9_eq_3_l1626_162675

theorem sqrt_of_9_eq_3 : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_of_9_eq_3_l1626_162675


namespace NUMINAMATH_GPT_a_5_eq_14_l1626_162691

def S (n : ℕ) : ℚ := (3 / 2) * n ^ 2 + (1 / 2) * n

def a (n : ℕ) : ℚ := S n - S (n - 1)

theorem a_5_eq_14 : a 5 = 14 := by {
  -- Proof steps go here
  sorry
}

end NUMINAMATH_GPT_a_5_eq_14_l1626_162691


namespace NUMINAMATH_GPT_calculate_neg4_mul_three_div_two_l1626_162650

theorem calculate_neg4_mul_three_div_two : (-4) * (3 / 2) = -6 := 
by
  sorry

end NUMINAMATH_GPT_calculate_neg4_mul_three_div_two_l1626_162650


namespace NUMINAMATH_GPT_vegetable_planting_methods_l1626_162607

theorem vegetable_planting_methods :
  let vegetables := ["cucumber", "cabbage", "rape", "lentils"]
  let cucumber := "cucumber"
  let other_vegetables := ["cabbage", "rape", "lentils"]
  let choose_2_out_of_3 := Nat.choose 3 2
  let arrangements := Nat.factorial 3
  total_methods = choose_2_out_of_3 * arrangements := by
  let total_methods := 3 * 6
  sorry

end NUMINAMATH_GPT_vegetable_planting_methods_l1626_162607


namespace NUMINAMATH_GPT_local_min_f_at_2_implies_a_eq_2_l1626_162615

theorem local_min_f_at_2_implies_a_eq_2 (a : ℝ) : 
  (∃ f : ℝ → ℝ, 
     (∀ x : ℝ, f x = x * (x - a)^2) ∧ 
     (∀ f' : ℝ → ℝ, 
       (∀ x : ℝ, f' x = 3 * x^2 - 4 * a * x + a^2) ∧ 
       f' 2 = 0 ∧ 
       (∀ f'' : ℝ → ℝ, 
         (∀ x : ℝ, f'' x = 6 * x - 4 * a) ∧ 
         f'' 2 > 0
       )
     )
  ) → a = 2 :=
sorry

end NUMINAMATH_GPT_local_min_f_at_2_implies_a_eq_2_l1626_162615


namespace NUMINAMATH_GPT_exists_nat_number_divisible_by_2019_and_its_digit_sum_also_divisible_by_2019_l1626_162640

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_nat_number_divisible_by_2019_and_its_digit_sum_also_divisible_by_2019 :
  ∃ N : ℕ, (N % 2019 = 0) ∧ ((sum_of_digits N) % 2019 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_exists_nat_number_divisible_by_2019_and_its_digit_sum_also_divisible_by_2019_l1626_162640


namespace NUMINAMATH_GPT_regular_octagon_interior_angle_l1626_162685

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end NUMINAMATH_GPT_regular_octagon_interior_angle_l1626_162685


namespace NUMINAMATH_GPT_suitable_comprehensive_survey_l1626_162668

def investigate_service_life_of_lamps : Prop := 
  -- This would typically involve checking a subset rather than every lamp
  sorry

def investigate_water_quality : Prop := 
  -- This would typically involve sampling rather than checking every point
  sorry

def investigate_sports_activities : Prop := 
  -- This would typically involve sampling rather than collecting data on every student
  sorry

def test_components_of_rocket : Prop := 
  -- Given the critical importance and manageable number of components, this requires comprehensive examination
  sorry

def most_suitable_for_comprehensive_survey : Prop :=
  test_components_of_rocket ∧ ¬investigate_service_life_of_lamps ∧ 
  ¬investigate_water_quality ∧ ¬investigate_sports_activities

theorem suitable_comprehensive_survey : most_suitable_for_comprehensive_survey :=
  sorry

end NUMINAMATH_GPT_suitable_comprehensive_survey_l1626_162668


namespace NUMINAMATH_GPT_train_length_is_225_m_l1626_162679

noncomputable def speed_kmph : ℝ := 90
noncomputable def time_s : ℝ := 9

noncomputable def speed_ms : ℝ := speed_kmph / 3.6
noncomputable def distance_m (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem train_length_is_225_m :
  distance_m speed_ms time_s = 225 := by
  sorry

end NUMINAMATH_GPT_train_length_is_225_m_l1626_162679


namespace NUMINAMATH_GPT_maria_must_earn_l1626_162603

-- Define the given conditions
def retail_price : ℕ := 600
def maria_savings : ℕ := 120
def mother_contribution : ℕ := 250

-- Total amount Maria has from savings and her mother's contribution
def total_savings : ℕ := maria_savings + mother_contribution

-- Prove that Maria must earn $230 to be able to buy the bike
theorem maria_must_earn : 600 - total_savings = 230 :=
by sorry

end NUMINAMATH_GPT_maria_must_earn_l1626_162603


namespace NUMINAMATH_GPT_f_eight_l1626_162655

noncomputable def f : ℝ → ℝ := sorry -- Defining the function without implementing it here

axiom f_x_neg {x : ℝ} (hx : x < 0) : f x = Real.log (-x) + x
axiom f_symmetric {x : ℝ} (hx : -Real.exp 1 ≤ x ∧ x ≤ Real.exp 1) : f (-x) = -f x
axiom f_periodic {x : ℝ} (hx : x > 1) : f (x + 2) = f x

theorem f_eight : f 8 = 2 - Real.log 2 := 
by
  sorry

end NUMINAMATH_GPT_f_eight_l1626_162655


namespace NUMINAMATH_GPT_ram_work_rate_l1626_162614

-- Definitions as given in the problem
variable (W : ℕ) -- Total work can be represented by some natural number W
variable (R M : ℕ) -- Raja's work rate and Ram's work rate, respectively

-- Given conditions
variable (combined_work_rate : R + M = W / 4)
variable (raja_work_rate : R = W / 12)

-- Theorem to be proven
theorem ram_work_rate (combined_work_rate : R + M = W / 4) (raja_work_rate : R = W / 12) : M = W / 6 := 
  sorry

end NUMINAMATH_GPT_ram_work_rate_l1626_162614


namespace NUMINAMATH_GPT_anne_more_drawings_l1626_162678

/-- Anne's markers problem setup. -/
structure MarkerProblem :=
  (markers : ℕ)
  (drawings_per_marker : ℚ)
  (drawings_made : ℕ)

-- Given conditions
def anne_conditions : MarkerProblem :=
  { markers := 12, drawings_per_marker := 1.5, drawings_made := 8 }

-- Equivalent proof problem statement in Lean
theorem anne_more_drawings(conditions : MarkerProblem) : 
  conditions.markers * conditions.drawings_per_marker - conditions.drawings_made = 10 :=
by
  -- The proof of this theorem is omitted
  sorry

end NUMINAMATH_GPT_anne_more_drawings_l1626_162678


namespace NUMINAMATH_GPT_min_value_l1626_162653

/-- Given x and y are positive real numbers such that x + 3y = 2,
    the minimum value of (2x + y) / (xy) is 1/2 * (7 + 2 * sqrt 6). -/
theorem min_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 3 * y = 2) :
  ∃ c : ℝ, c = (1/2) * (7 + 2 * Real.sqrt 6) ∧ ∀ (x y : ℝ), (0 < x) → (0 < y) → (x + 3 * y = 2) → ((2 * x + y) / (x * y)) ≥ c :=
sorry

end NUMINAMATH_GPT_min_value_l1626_162653


namespace NUMINAMATH_GPT_symmetric_point_coordinates_l1626_162651

theorem symmetric_point_coordinates 
  (k : ℝ) 
  (P : ℝ × ℝ) 
  (h1 : ∀ k, k * (P.1) - P.2 + k - 2 = 0) 
  (P' : ℝ × ℝ) 
  (h2 : P'.1 + P'.2 = 3) 
  (h3 : 2 * P'.1^2 + 2 * P'.2^2 + 4 * P'.1 + 8 * P'.2 + 5 = 0) 
  (hP : P = (-1, -2)): 
  P' = (2, 1) := 
sorry

end NUMINAMATH_GPT_symmetric_point_coordinates_l1626_162651


namespace NUMINAMATH_GPT_count_ways_to_complete_20160_l1626_162636

noncomputable def waysToComplete : Nat :=
  let choices_for_last_digit := 5
  let choices_for_first_three_digits := 9^3
  choices_for_last_digit * choices_for_first_three_digits

theorem count_ways_to_complete_20160 (choices : Fin 9 → Fin 9) : waysToComplete = 3645 := by
  sorry

end NUMINAMATH_GPT_count_ways_to_complete_20160_l1626_162636


namespace NUMINAMATH_GPT_sum_of_distances_l1626_162682

theorem sum_of_distances (a b : ℤ) (k : ℕ) 
  (h1 : |k - a| + |(k + 1) - a| + |(k + 2) - a| + |(k + 3) - a| + |(k + 4) - a| + |(k + 5) - a| + |(k + 6) - a| = 609)
  (h2 : |k - b| + |(k + 1) - b| + |(k + 2) - b| + |(k + 3) - b| + |(k + 4) - b| + |(k + 5) - b| + |(k + 6) - b| = 721)
  (h3 : a + b = 192) :
  a = 1 ∨ a = 104 ∨ a = 191 := 
sorry

end NUMINAMATH_GPT_sum_of_distances_l1626_162682


namespace NUMINAMATH_GPT_ants_harvest_remaining_sugar_l1626_162637

-- Define the initial conditions
def ants_removal_rate : ℕ := 4
def initial_sugar_amount : ℕ := 24
def hours_passed : ℕ := 3

-- Calculate the correct answer
def remaining_sugar (initial : ℕ) (rate : ℕ) (hours : ℕ) : ℕ :=
  initial - (rate * hours)

def additional_hours_needed (remaining_sugar : ℕ) (rate : ℕ) : ℕ :=
  remaining_sugar / rate

-- The specification of the proof problem
theorem ants_harvest_remaining_sugar :
  additional_hours_needed (remaining_sugar initial_sugar_amount ants_removal_rate hours_passed) ants_removal_rate = 3 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_ants_harvest_remaining_sugar_l1626_162637


namespace NUMINAMATH_GPT_employee_discount_percentage_l1626_162656

def wholesale_cost : ℝ := 200
def retail_markup : ℝ := 0.20
def employee_paid_price : ℝ := 228

theorem employee_discount_percentage :
  let retail_price := wholesale_cost * (1 + retail_markup)
  let discount := retail_price - employee_paid_price
  (discount / retail_price) * 100 = 5 := by
  sorry

end NUMINAMATH_GPT_employee_discount_percentage_l1626_162656


namespace NUMINAMATH_GPT_percent_receiving_speeding_tickets_l1626_162620

theorem percent_receiving_speeding_tickets
  (total_motorists : ℕ)
  (percent_exceeding_limit percent_exceeding_limit_without_ticket : ℚ)
  (h_exceeding_limit : percent_exceeding_limit = 0.5)
  (h_exceeding_limit_without_ticket : percent_exceeding_limit_without_ticket = 0.2) :
  let exceeding_limit := percent_exceeding_limit * total_motorists
  let without_tickets := percent_exceeding_limit_without_ticket * exceeding_limit
  let with_tickets := exceeding_limit - without_tickets
  (with_tickets / total_motorists) * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_percent_receiving_speeding_tickets_l1626_162620


namespace NUMINAMATH_GPT_least_positive_t_l1626_162611

theorem least_positive_t
  (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (ht : ∃ t, 0 < t ∧ (∃ r, (Real.arcsin (Real.sin α) * r = Real.arcsin (Real.sin (3 * α)) ∧ 
                            Real.arcsin (Real.sin (3 * α)) * r = Real.arcsin (Real.sin (5 * α)) ∧
                            Real.arcsin (Real.sin (5 * α)) * r = Real.arcsin (Real.sin (t * α))))) :
  t = 6 :=
sorry

end NUMINAMATH_GPT_least_positive_t_l1626_162611


namespace NUMINAMATH_GPT_no_positive_integer_n_satisfies_conditions_l1626_162625

theorem no_positive_integer_n_satisfies_conditions :
  ¬ ∃ (n : ℕ), (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_n_satisfies_conditions_l1626_162625


namespace NUMINAMATH_GPT_jack_round_trip_speed_l1626_162665

noncomputable def jack_average_speed (d1 d2 : ℕ) (t1 t2 : ℕ) : ℕ :=
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let total_time_hours := total_time / 60
  total_distance / total_time_hours

theorem jack_round_trip_speed : jack_average_speed 3 3 45 15 = 6 := by
  -- Import necessary library
  sorry

end NUMINAMATH_GPT_jack_round_trip_speed_l1626_162665


namespace NUMINAMATH_GPT_number_of_dogs_l1626_162670

variable (D C : ℕ)
variable (x : ℚ)

-- Conditions
def ratio_dogs_to_cats := D = (x * (C: ℚ) / 7)
def new_ratio_dogs_to_cats := D = (15 / 11) * (C + 8)

theorem number_of_dogs (h1 : ratio_dogs_to_cats D C x) (h2 : new_ratio_dogs_to_cats D C) : D = 77 := 
by sorry

end NUMINAMATH_GPT_number_of_dogs_l1626_162670


namespace NUMINAMATH_GPT_departure_of_30_tons_of_grain_l1626_162664

-- Define positive as an arrival of grain.
def positive_arrival (x : ℤ) : Prop := x > 0

-- Define negative as a departure of grain.
def negative_departure (x : ℤ) : Prop := x < 0

-- The given conditions and question translated to a Lean statement.
theorem departure_of_30_tons_of_grain :
  (positive_arrival 30) → (negative_departure (-30)) :=
by
  intro pos30
  sorry

end NUMINAMATH_GPT_departure_of_30_tons_of_grain_l1626_162664


namespace NUMINAMATH_GPT_problem_solution_l1626_162695

theorem problem_solution (m n : ℤ) (h : m + 1 = (n - 2) / 3) : 3 * m - n = -5 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1626_162695


namespace NUMINAMATH_GPT_Henry_age_ratio_l1626_162623

theorem Henry_age_ratio (A S H : ℕ)
  (hA : A = 15)
  (hS : S = 3 * A)
  (h_sum : A + S + H = 240) :
  H / S = 4 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_Henry_age_ratio_l1626_162623


namespace NUMINAMATH_GPT_B_N_Q_collinear_l1626_162674

/-- Define point positions -/
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨-1, 0⟩
def N : Point := ⟨1, 0⟩

/-- Define the curve C -/
def on_curve_C (P : Point) : Prop :=
  P.x^2 + P.y^2 - 6 * P.x + 1 = 0

/-- Define reflection of point A across the x-axis -/
def reflection_across_x (A : Point) : Point :=
  ⟨A.x, -A.y⟩

/-- Define the condition that line l passes through M and intersects curve C at two distinct points A and B -/
def line_l_condition (A B: Point) (k : ℝ) (hk : k ≠ 0) : Prop :=
  A.y = k * (A.x + 1) ∧ B.y = k * (B.x + 1) ∧ on_curve_C A ∧ on_curve_C B

/-- Main theorem to prove collinearity of B, N, Q -/
theorem B_N_Q_collinear (A B : Point) (k : ℝ) (hk : k ≠ 0)
  (hA : on_curve_C A) (hB : on_curve_C B)
  (h_l : line_l_condition A B k hk) :
  let Q := reflection_across_x A
  (B.x - N.x) * (Q.y - N.y) = (B.y - N.y) * (Q.x - N.x) :=
sorry

end NUMINAMATH_GPT_B_N_Q_collinear_l1626_162674


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_property_l1626_162634

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ)  -- sequence terms are real numbers
  (d : ℝ)      -- common difference
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum_condition : a 4 + a 8 = 16) :
  a 2 + a 10 = 16 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_property_l1626_162634


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1626_162601

theorem arithmetic_sequence_sum {S : ℕ → ℤ} (m : ℕ) (hm : 0 < m)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1626_162601


namespace NUMINAMATH_GPT_sahil_machine_purchase_price_l1626_162671

theorem sahil_machine_purchase_price
  (repair_cost : ℕ)
  (transportation_cost : ℕ)
  (selling_price : ℕ)
  (profit_percent : ℤ)
  (purchase_price : ℕ)
  (total_cost : ℕ)
  (profit_ratio : ℚ)
  (h1 : repair_cost = 5000)
  (h2 : transportation_cost = 1000)
  (h3 : selling_price = 30000)
  (h4 : profit_percent = 50)
  (h5 : total_cost = purchase_price + repair_cost + transportation_cost)
  (h6 : profit_ratio = profit_percent / 100)
  (h7 : selling_price = (1 + profit_ratio) * total_cost) :
  purchase_price = 14000 :=
by
  sorry

end NUMINAMATH_GPT_sahil_machine_purchase_price_l1626_162671


namespace NUMINAMATH_GPT_simplify_expression_l1626_162666

theorem simplify_expression :
  (2 * (Real.sqrt 2 + Real.sqrt 6)) / (3 * Real.sqrt (2 + Real.sqrt 3)) = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1626_162666


namespace NUMINAMATH_GPT_evaluate_f_at_minus_2_l1626_162689

def f (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem evaluate_f_at_minus_2 : f (-2) = 7 / 3 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_evaluate_f_at_minus_2_l1626_162689


namespace NUMINAMATH_GPT_intersection_S_T_eq_U_l1626_162686

def S : Set ℝ := {x | abs x < 5}
def T : Set ℝ := {x | (x + 7) * (x - 3) < 0}
def U : Set ℝ := {x | -5 < x ∧ x < 3}

theorem intersection_S_T_eq_U : (S ∩ T) = U := 
by 
  sorry

end NUMINAMATH_GPT_intersection_S_T_eq_U_l1626_162686


namespace NUMINAMATH_GPT_remainder_eq_four_l1626_162658

theorem remainder_eq_four {x : ℤ} (h : x % 61 = 24) : x % 5 = 4 :=
sorry

end NUMINAMATH_GPT_remainder_eq_four_l1626_162658


namespace NUMINAMATH_GPT_work_done_is_halved_l1626_162629

theorem work_done_is_halved
  (A₁₂ A₃₄ : ℝ)
  (isothermal_process : ∀ (p V₁₂ V₃₄ : ℝ), V₁₂ = 2 * V₃₄ → p * V₁₂ = A₁₂ → p * V₃₄ = A₃₄) :
  A₃₄ = (1 / 2) * A₁₂ :=
sorry

end NUMINAMATH_GPT_work_done_is_halved_l1626_162629


namespace NUMINAMATH_GPT_sugar_needed_for_partial_recipe_l1626_162652

theorem sugar_needed_for_partial_recipe :
  let initial_sugar := 5 + 3/4
  let part := 3/4
  let needed_sugar := 4 + 5/16
  initial_sugar * part = needed_sugar := 
by 
  sorry

end NUMINAMATH_GPT_sugar_needed_for_partial_recipe_l1626_162652


namespace NUMINAMATH_GPT_rational_sqrt_condition_l1626_162677

variable (r q n : ℚ)

theorem rational_sqrt_condition
  (h : (1 / (r + q * n) + 1 / (q + r * n) = 1 / (r + q))) : 
  ∃ x : ℚ, x^2 = (n - 3) / (n + 1) :=
sorry

end NUMINAMATH_GPT_rational_sqrt_condition_l1626_162677


namespace NUMINAMATH_GPT_number_of_customers_l1626_162661

theorem number_of_customers (total_sandwiches : ℕ) (office_orders : ℕ) (customers_half : ℕ) (num_offices : ℕ) (num_sandwiches_per_office : ℕ) 
  (sandwiches_per_customer : ℕ) (group_sandwiches : ℕ) (total_customers : ℕ) :
  total_sandwiches = 54 →
  num_offices = 3 →
  num_sandwiches_per_office = 10 →
  group_sandwiches = total_sandwiches - num_offices * num_sandwiches_per_office →
  customers_half * sandwiches_per_customer = group_sandwiches →
  sandwiches_per_customer = 4 →
  customers_half = total_customers / 2 →
  total_customers = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_customers_l1626_162661


namespace NUMINAMATH_GPT_find_a2018_l1626_162684

-- Given Conditions
def initial_condition (a : ℕ → ℤ) : Prop :=
  a 1 = -1

def absolute_difference (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → abs (a n - a (n-1)) = 2^(n-1)

def subseq_decreasing (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (2*n-1) > a (2*(n+1)-1)

def subseq_increasing (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (2*n) < a (2*(n+1))

-- Theorem to Prove
theorem find_a2018 (a : ℕ → ℤ)
  (h1 : initial_condition a)
  (h2 : absolute_difference a)
  (h3 : subseq_decreasing a)
  (h4 : subseq_increasing a) :
  a 2018 = (2^2018 - 1) / 3 :=
sorry

end NUMINAMATH_GPT_find_a2018_l1626_162684


namespace NUMINAMATH_GPT_new_person_age_l1626_162646

theorem new_person_age (T : ℕ) : 
  (T / 10) = ((T - 46 + A) / 10) + 3 → (A = 16) :=
by
  sorry

end NUMINAMATH_GPT_new_person_age_l1626_162646


namespace NUMINAMATH_GPT_derivative_at_0_l1626_162628

-- Define the function
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 2

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := deriv f x

-- State the theorem
theorem derivative_at_0 : f' 0 = 4 :=
by {
  -- Inserting sorry to skip the proof
  sorry
}

end NUMINAMATH_GPT_derivative_at_0_l1626_162628


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1626_162687

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 4}
def complement_set (U A : Set ℕ) : Set ℕ := U \ A

theorem complement_of_A_in_U :
  complement_set U A = {2, 3, 5} :=
by
  apply Set.ext
  intro x
  simp [complement_set, U, A]
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1626_162687


namespace NUMINAMATH_GPT_hexagon_six_legal_triangles_hexagon_ten_legal_triangles_hexagon_two_thousand_fourteen_legal_triangles_l1626_162621

-- Define a hexagon with legal points and triangles

structure Hexagon :=
  (A B C D E F : ℝ)

-- Legal point occurs when certain conditions on intersection between diagonals hold
def legal_point (h : Hexagon) (x : ℝ) (y : ℝ) : Prop :=
  -- Placeholder, we need to define the exact condition based on problem constraints.
  sorry

-- Function to check if a division is legal based on defined rules
def legal_triangle_division (h : Hexagon) (n : ℕ) : Prop :=
  -- Placeholder, this requires a definition based on how points and triangles are formed
  sorry

-- Prove the specific cases
theorem hexagon_six_legal_triangles (h : Hexagon) : legal_triangle_division h 6 :=
  sorry

theorem hexagon_ten_legal_triangles (h : Hexagon) : legal_triangle_division h 10 :=
  sorry

theorem hexagon_two_thousand_fourteen_legal_triangles (h : Hexagon)  : legal_triangle_division h 2014 :=
  sorry

end NUMINAMATH_GPT_hexagon_six_legal_triangles_hexagon_ten_legal_triangles_hexagon_two_thousand_fourteen_legal_triangles_l1626_162621


namespace NUMINAMATH_GPT_probability_merlin_dismissed_l1626_162649

variable (p : ℝ) (q : ℝ) (coin_flip : ℝ)

axiom h₁ : q = 1 - p
axiom h₂ : 0 ≤ p ∧ p ≤ 1
axiom h₃ : 0 ≤ q ∧ q ≤ 1
axiom h₄ : coin_flip = 0.5

theorem probability_merlin_dismissed : coin_flip = 0.5 := by
  sorry

end NUMINAMATH_GPT_probability_merlin_dismissed_l1626_162649


namespace NUMINAMATH_GPT_sum_of_three_numbers_eq_16_l1626_162694

variable {a b c : ℝ}

theorem sum_of_three_numbers_eq_16
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + c * a = 50) :
  a + b + c = 16 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_eq_16_l1626_162694


namespace NUMINAMATH_GPT_pages_per_day_l1626_162616

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (result : ℕ) :
  total_pages = 81 ∧ days = 3 → result = 27 :=
by
  sorry

end NUMINAMATH_GPT_pages_per_day_l1626_162616


namespace NUMINAMATH_GPT_sequence_term_general_sequence_sum_term_general_l1626_162622

theorem sequence_term_general (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S (n + 1) = 2 * S n + 1) →
  a 1 = 1 →
  (∀ n ≥ 1, a n = 2^(n-1)) :=
  sorry

theorem sequence_sum_term_general (na : ℕ → ℕ) (T : ℕ → ℕ) :
  (∀ k, na k = k * 2^(k-1)) →
  (∀ n, T n = (n - 1) * 2^n + 1) :=
  sorry

end NUMINAMATH_GPT_sequence_term_general_sequence_sum_term_general_l1626_162622


namespace NUMINAMATH_GPT_inequality_solution_l1626_162663

theorem inequality_solution (x : ℝ) :
  (2 / (x - 3) ≤ 5) ↔ (x < 3 ∨ x ≥ 17 / 5) := 
sorry

end NUMINAMATH_GPT_inequality_solution_l1626_162663


namespace NUMINAMATH_GPT_ball_height_25_l1626_162696

theorem ball_height_25 (t : ℝ) (h : ℝ) 
  (h_eq : h = 45 - 7 * t - 6 * t^2) : 
  h = 25 ↔ t = 4 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_ball_height_25_l1626_162696


namespace NUMINAMATH_GPT_find_value_of_y_l1626_162600

theorem find_value_of_y (y : ℝ) (h : 9 / y^2 = y / 81) : y = 9 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_value_of_y_l1626_162600


namespace NUMINAMATH_GPT_sum_of_polynomials_l1626_162698

-- Define the given polynomials f, g, and h
def f (x : ℝ) : ℝ := -6 * x^3 - 4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -7 * x^2 + 6 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 7 * x + 3

-- Prove that the sum of f(x), g(x), and h(x) is a specific polynomial
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x = -6 * x^3 - 5 * x^2 + 15 * x - 11 := 
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_sum_of_polynomials_l1626_162698


namespace NUMINAMATH_GPT_next_elements_l1626_162604

-- Define the conditions and the question
def next_elements_in_sequence (n : ℕ) : String :=
  match n with
  | 1 => "О"  -- "Один"
  | 2 => "Д"  -- "Два"
  | 3 => "Т"  -- "Три"
  | 4 => "Ч"  -- "Четыре"
  | 5 => "П"  -- "Пять"
  | 6 => "Ш"  -- "Шесть"
  | 7 => "С"  -- "Семь"
  | 8 => "В"  -- "Восемь"
  | _ => "?"

theorem next_elements (n : ℕ) :
  next_elements_in_sequence 7 = "С" ∧ next_elements_in_sequence 8 = "В" := by
  sorry

end NUMINAMATH_GPT_next_elements_l1626_162604


namespace NUMINAMATH_GPT_seq_geq_4_l1626_162633

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 5 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = (a n ^ 2 + 8 * a n + 16) / (4 * a n)

theorem seq_geq_4 (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, n ≥ 1 → a n ≥ 4 :=
sorry

end NUMINAMATH_GPT_seq_geq_4_l1626_162633


namespace NUMINAMATH_GPT_total_fencing_cost_is_correct_l1626_162627

-- Define the fencing cost per side
def costPerSide : Nat := 69

-- Define the number of sides for a square
def sidesOfSquare : Nat := 4

-- Define the total cost calculation for fencing the square
def totalCostOfFencing (costPerSide : Nat) (sidesOfSquare : Nat) := costPerSide * sidesOfSquare

-- Prove that for a given cost per side and number of sides, the total cost of fencing the square is 276 dollars
theorem total_fencing_cost_is_correct : totalCostOfFencing 69 4 = 276 :=
by
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_total_fencing_cost_is_correct_l1626_162627


namespace NUMINAMATH_GPT_john_profit_percentage_is_50_l1626_162606

noncomputable def profit_percentage
  (P : ℝ)  -- The sum of money John paid for purchasing 30 pens
  (recovered_amount : ℝ)  -- The amount John recovered when he sold 20 pens
  (condition : recovered_amount = P) -- Condition that John recovered the full amount P when he sold 20 pens
  : ℝ := 
  ((P / 20) - (P / 30)) / (P / 30) * 100

theorem john_profit_percentage_is_50
  (P : ℝ)
  (recovered_amount : ℝ)
  (condition : recovered_amount = P) :
  profit_percentage P recovered_amount condition = 50 := 
  by 
  sorry

end NUMINAMATH_GPT_john_profit_percentage_is_50_l1626_162606


namespace NUMINAMATH_GPT_largest_difference_l1626_162654

def U : ℕ := 2 * 1002 ^ 1003
def V : ℕ := 1002 ^ 1003
def W : ℕ := 1001 * 1002 ^ 1002
def X : ℕ := 2 * 1002 ^ 1002
def Y : ℕ := 1002 ^ 1002
def Z : ℕ := 1002 ^ 1001

theorem largest_difference : (U - V) = 1002 ^ 1003 ∧ 
  (V - W) = 1002 ^ 1002 ∧ 
  (W - X) = 999 * 1002 ^ 1002 ∧ 
  (X - Y) = 1002 ^ 1002 ∧ 
  (Y - Z) = 1001 * 1002 ^ 1001 ∧ 
  (1002 ^ 1003 > 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 999 * 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 1001 * 1002 ^ 1001) :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_difference_l1626_162654


namespace NUMINAMATH_GPT_jimin_has_most_candy_left_l1626_162613

-- Definitions based on conditions
def fraction_jimin_ate := 1 / 9
def fraction_taehyung_ate := 1 / 3
def fraction_hoseok_ate := 1 / 6

-- The goal to prove
theorem jimin_has_most_candy_left : 
  (1 - fraction_jimin_ate) > (1 - fraction_taehyung_ate) ∧ (1 - fraction_jimin_ate) > (1 - fraction_hoseok_ate) :=
by
  -- The actual proof steps are omitted here.
  sorry

end NUMINAMATH_GPT_jimin_has_most_candy_left_l1626_162613


namespace NUMINAMATH_GPT_mod_equiv_l1626_162657

theorem mod_equiv (a b c d e : ℤ) (n : ℤ) (h1 : a = 101)
                                    (h2 : b = 15)
                                    (h3 : c = 7)
                                    (h4 : d = 9)
                                    (h5 : e = 5)
                                    (h6 : n = 17) :
  (a * b - c * d + e) % n = 7 := by
  sorry

end NUMINAMATH_GPT_mod_equiv_l1626_162657


namespace NUMINAMATH_GPT_contrapositive_necessary_condition_l1626_162610

theorem contrapositive_necessary_condition (a b : Prop) (h : a → b) : ¬b → ¬a :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_necessary_condition_l1626_162610


namespace NUMINAMATH_GPT_inverse_of_A_cubed_l1626_162618

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -2,  3],
    ![  0,  1]]

theorem inverse_of_A_cubed :
  (A_inv ^ 3) = ![![ -8,  9],
                    ![  0,  1]] :=
by sorry

end NUMINAMATH_GPT_inverse_of_A_cubed_l1626_162618


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1626_162602

theorem isosceles_triangle_perimeter (x y : ℝ) (h : |x - 4| + (y - 8)^2 = 0) :
  4 + 8 + 8 = 20 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1626_162602


namespace NUMINAMATH_GPT_intersection_point_is_neg3_l1626_162662

def f (x : ℝ) : ℝ := x^3 + 6 * x^2 + 9 * x + 15

theorem intersection_point_is_neg3 :
  ∃ a b : ℝ, (f a = b) ∧ (f b = a) ∧ (a, b) = (-3, -3) := sorry

end NUMINAMATH_GPT_intersection_point_is_neg3_l1626_162662


namespace NUMINAMATH_GPT_area_of_QCA_l1626_162638

noncomputable def area_of_triangle (x p : ℝ) (hx_pos : 0 < x) (hp_bounds : 0 < p ∧ p < 15) : ℝ :=
  1 / 2 * x * (15 - p)

theorem area_of_QCA (x : ℝ) (p : ℝ) (hx_pos : 0 < x) (hp_bounds : 0 < p ∧ p < 15) :
  area_of_triangle x p hx_pos hp_bounds = 1 / 2 * x * (15 - p) :=
sorry

end NUMINAMATH_GPT_area_of_QCA_l1626_162638


namespace NUMINAMATH_GPT_product_of_number_and_its_digits_sum_l1626_162641

theorem product_of_number_and_its_digits_sum :
  ∃ (n : ℕ), (n = 24 ∧ (n % 10) = ((n / 10) % 10) + 2) ∧ (n * (n % 10 + (n / 10) % 10) = 144) :=
by
  sorry

end NUMINAMATH_GPT_product_of_number_and_its_digits_sum_l1626_162641


namespace NUMINAMATH_GPT_ratio_of_areas_l1626_162699

-- Defining the variables for sides of rectangles
variables {a b c d : ℝ}

-- Given conditions
axiom h1 : a / c = 4 / 5
axiom h2 : b / d = 4 / 5

-- Statement to prove the ratio of areas
theorem ratio_of_areas (h1 : a / c = 4 / 5) (h2 : b / d = 4 / 5) : (a * b) / (c * d) = 16 / 25 :=
sorry

end NUMINAMATH_GPT_ratio_of_areas_l1626_162699


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1626_162660

theorem isosceles_triangle_perimeter (m x₁ x₂ : ℝ) (h₁ : 1^2 + m * 1 + 5 = 0) 
  (hx : x₁^2 + m * x₁ + 5 = 0 ∧ x₂^2 + m * x₂ + 5 = 0)
  (isosceles : (x₁ = x₂ ∨ x₁ = 1 ∨ x₂ = 1)) : 
  ∃ (P : ℝ), P = 11 :=
by 
  -- Here, you'd prove that under these conditions, the perimeter must be 11.
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1626_162660


namespace NUMINAMATH_GPT_simplify_expression_l1626_162626

variable (x y : ℤ) -- Assume x and y are integers for simplicity

theorem simplify_expression : (5 - 2 * x) - (8 - 6 * x + 3 * y) = -3 + 4 * x - 3 * y := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1626_162626


namespace NUMINAMATH_GPT_parabola_vertex_shift_l1626_162624

theorem parabola_vertex_shift
  (vertex_initial : ℝ × ℝ)
  (h₀ : vertex_initial = (0, 0))
  (move_left : ℝ)
  (move_up : ℝ)
  (h₁ : move_left = -2)
  (h₂ : move_up = 3):
  (vertex_initial.1 + move_left, vertex_initial.2 + move_up) = (-2, 3) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_shift_l1626_162624


namespace NUMINAMATH_GPT_mode_is_necessary_characteristic_of_dataset_l1626_162688

-- Define a dataset as a finite set of elements from any type.
variable {α : Type*} [Fintype α]

-- Define a mode for a dataset as an element that occurs most frequently.
def mode (dataset : Multiset α) : α :=
sorry  -- Mode definition and computation are omitted for this high-level example.

-- Define the theorem that mode is a necessary characteristic of a dataset.
theorem mode_is_necessary_characteristic_of_dataset (dataset : Multiset α) : 
  exists mode_elm : α, mode_elm = mode dataset :=
sorry

end NUMINAMATH_GPT_mode_is_necessary_characteristic_of_dataset_l1626_162688


namespace NUMINAMATH_GPT_minyoung_money_l1626_162630

theorem minyoung_money (A M : ℕ) (h1 : M = 90 * A) (h2 : M = 60 * A + 270) : M = 810 :=
by 
  sorry

end NUMINAMATH_GPT_minyoung_money_l1626_162630


namespace NUMINAMATH_GPT_lion_king_box_office_earnings_l1626_162659

-- Definitions and conditions
def cost_lion_king : ℕ := 10  -- Lion King cost 10 million
def cost_star_wars : ℕ := 25  -- Star Wars cost 25 million
def earnings_star_wars : ℕ := 405  -- Star Wars earned 405 million

-- Calculate profit of Star Wars
def profit_star_wars : ℕ := earnings_star_wars - cost_star_wars

-- Define the profit of The Lion King, given it's half of Star Wars' profit
def profit_lion_king : ℕ := profit_star_wars / 2

-- Calculate the earnings of The Lion King
def earnings_lion_king : ℕ := cost_lion_king + profit_lion_king

-- Theorem to prove
theorem lion_king_box_office_earnings : earnings_lion_king = 200 :=
by
  sorry

end NUMINAMATH_GPT_lion_king_box_office_earnings_l1626_162659


namespace NUMINAMATH_GPT_factorial_mod_13_l1626_162676

open Nat

theorem factorial_mod_13 :
  let n := 10
  let p := 13
  n! % p = 6 := by
sorry

end NUMINAMATH_GPT_factorial_mod_13_l1626_162676


namespace NUMINAMATH_GPT_largest_unachievable_score_l1626_162609

theorem largest_unachievable_score :
  ∀ (x y : ℕ), 3 * x + 7 * y ≠ 11 :=
by
  sorry

end NUMINAMATH_GPT_largest_unachievable_score_l1626_162609


namespace NUMINAMATH_GPT_calc_4_op_3_l1626_162681

def specific_op (m n : ℕ) : ℕ := n^2 - m

theorem calc_4_op_3 :
  specific_op 4 3 = 5 :=
by
  sorry

end NUMINAMATH_GPT_calc_4_op_3_l1626_162681


namespace NUMINAMATH_GPT_unique_not_in_range_l1626_162669

noncomputable def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_not_in_range (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0)
  (h₅ : f a b c d 23 = 23) (h₆ : f a b c d 101 = 101) (h₇ : ∀ x ≠ -d / c, f a b c d (f a b c d x) = x) :
  (a / c) = 62 := 
 sorry

end NUMINAMATH_GPT_unique_not_in_range_l1626_162669


namespace NUMINAMATH_GPT_min_sum_abs_l1626_162672

theorem min_sum_abs (x : ℝ) : ∃ m, m = 4 ∧ ∀ x : ℝ, |x + 2| + |x - 2| + |x - 1| ≥ m := 
sorry

end NUMINAMATH_GPT_min_sum_abs_l1626_162672


namespace NUMINAMATH_GPT_problem1_problem2_l1626_162683

theorem problem1 (x : ℚ) (h : x ≠ -4) : (3 - x) / (x + 4) = 1 / 2 → x = 2 / 3 :=
by
  sorry

theorem problem2 (x : ℚ) (h : x ≠ 1) : x / (x - 1) - 2 * x / (3 * (x - 1)) = 1 → x = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1626_162683


namespace NUMINAMATH_GPT_perimeter_is_correct_l1626_162617

def side_length : ℕ := 2
def original_horizontal_segments : ℕ := 16
def original_vertical_segments : ℕ := 10

def horizontal_length : ℕ := original_horizontal_segments * side_length
def vertical_length : ℕ := original_vertical_segments * side_length

def perimeter : ℕ := horizontal_length + vertical_length

theorem perimeter_is_correct : perimeter = 52 :=
by 
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_perimeter_is_correct_l1626_162617


namespace NUMINAMATH_GPT_remainder_7531_mod_11_is_5_l1626_162648

theorem remainder_7531_mod_11_is_5 :
  let n := 7531
  let m := 7 + 5 + 3 + 1
  n % 11 = 5 ∧ m % 11 = 5 :=
by
  let n := 7531
  let m := 7 + 5 + 3 + 1
  have h : n % 11 = m % 11 := sorry  -- by property of digits sum mod
  have hm : m % 11 = 5 := sorry      -- calculation
  exact ⟨h, hm⟩

end NUMINAMATH_GPT_remainder_7531_mod_11_is_5_l1626_162648


namespace NUMINAMATH_GPT_jane_earnings_l1626_162608

def earnings_per_bulb : ℝ := 0.50
def tulip_bulbs : ℕ := 20
def iris_bulbs : ℕ := tulip_bulbs / 2
def daffodil_bulbs : ℕ := 30
def crocus_bulbs : ℕ := daffodil_bulbs * 3
def total_earnings : ℝ := (tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs) * earnings_per_bulb

theorem jane_earnings : total_earnings = 75.0 := by
  sorry

end NUMINAMATH_GPT_jane_earnings_l1626_162608


namespace NUMINAMATH_GPT_show_watching_days_l1626_162690

def numberOfEpisodes := 20
def lengthOfEachEpisode := 30
def dailyWatchingTime := 2

theorem show_watching_days:
  (numberOfEpisodes * lengthOfEachEpisode) / 60 / dailyWatchingTime = 5 := 
by
  sorry

end NUMINAMATH_GPT_show_watching_days_l1626_162690


namespace NUMINAMATH_GPT_total_bouncy_balls_l1626_162645

def red_packs := 4
def yellow_packs := 8
def green_packs := 4
def blue_packs := 6

def red_balls_per_pack := 12
def yellow_balls_per_pack := 10
def green_balls_per_pack := 14
def blue_balls_per_pack := 8

def total_red_balls := red_packs * red_balls_per_pack
def total_yellow_balls := yellow_packs * yellow_balls_per_pack
def total_green_balls := green_packs * green_balls_per_pack
def total_blue_balls := blue_packs * blue_balls_per_pack

def total_balls := total_red_balls + total_yellow_balls + total_green_balls + total_blue_balls

theorem total_bouncy_balls : total_balls = 232 :=
by
  -- calculation proof goes here
  sorry

end NUMINAMATH_GPT_total_bouncy_balls_l1626_162645


namespace NUMINAMATH_GPT_striped_shorts_difference_l1626_162692

variable (students : ℕ)
variable (striped_shirts checkered_shirts shorts : ℕ)

-- Conditions
variable (Hstudents : students = 81)
variable (Hstriped : striped_shirts = 2 * checkered_shirts)
variable (Hcheckered : checkered_shirts = students / 3)
variable (Hshorts : shorts = checkered_shirts + 19)

-- Goal
theorem striped_shorts_difference :
  striped_shirts - shorts = 8 :=
sorry

end NUMINAMATH_GPT_striped_shorts_difference_l1626_162692


namespace NUMINAMATH_GPT_find_b_l1626_162693

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b * x ^ 3 + b * x - 3

theorem find_b (b : ℝ) (h : g b (g b 1) = 1) : b = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1626_162693


namespace NUMINAMATH_GPT_value_of_expression_l1626_162643

theorem value_of_expression (x y : ℝ) (h₁ : x * y = 3) (h₂ : x + y = 4) : x ^ 2 + y ^ 2 - 3 * x * y = 1 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1626_162643


namespace NUMINAMATH_GPT_quadratic_inequality_l1626_162632

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, x^2 + 2 * a * x + a > 0) : 0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_l1626_162632


namespace NUMINAMATH_GPT_bus_speed_including_stoppages_l1626_162697

theorem bus_speed_including_stoppages 
  (speed_excl_stoppages : ℚ) 
  (ten_minutes_per_hour : ℚ) 
  (bus_stops_for_10_minutes : ten_minutes_per_hour = 10/60) 
  (speed_is_54_kmph : speed_excl_stoppages = 54) : 
  (speed_excl_stoppages * (1 - ten_minutes_per_hour)) = 45 := 
by 
  sorry

end NUMINAMATH_GPT_bus_speed_including_stoppages_l1626_162697


namespace NUMINAMATH_GPT_find_value_of_n_l1626_162644

theorem find_value_of_n (n : ℤ) : 
    n + (n + 1) + (n + 2) + (n + 3) = 22 → n = 4 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_find_value_of_n_l1626_162644


namespace NUMINAMATH_GPT_net_profit_start_year_better_investment_option_l1626_162619

-- Question 1: From which year does the developer start to make a net profit?
def investment_cost : ℕ := 81 -- in 10,000 yuan
def first_year_renovation_cost : ℕ := 1 -- in 10,000 yuan
def renovation_cost_increase : ℕ := 2 -- in 10,000 yuan per year
def annual_rental_income : ℕ := 30 -- in 10,000 yuan per year

theorem net_profit_start_year : ∃ n : ℕ, n ≥ 4 ∧ ∀ m < 4, ¬ (annual_rental_income * m > investment_cost + m^2) :=
by sorry

-- Question 2: Which option is better: maximizing total profit or average annual profit?
def profit_function (n : ℕ) : ℤ := 30 * n - (81 + n^2)
def average_annual_profit (n : ℕ) : ℤ := (30 * n - (81 + n^2)) / n
def max_total_profit_year : ℕ := 15
def max_total_profit : ℤ := 144 -- in 10,000 yuan
def max_average_profit_year : ℕ := 9
def max_average_profit : ℤ := 12 -- in 10,000 yuan

theorem better_investment_option : (average_annual_profit max_average_profit_year) ≥ (profit_function max_total_profit_year) / max_total_profit_year :=
by sorry

end NUMINAMATH_GPT_net_profit_start_year_better_investment_option_l1626_162619


namespace NUMINAMATH_GPT_simplest_common_denominator_l1626_162635

theorem simplest_common_denominator (x a : ℕ) :
  let d1 := 3 * x
  let d2 := 6 * x^2
  lcm d1 d2 = 6 * x^2 := 
by
  let d1 := 3 * x
  let d2 := 6 * x^2
  show lcm d1 d2 = 6 * x^2
  sorry

end NUMINAMATH_GPT_simplest_common_denominator_l1626_162635


namespace NUMINAMATH_GPT_chalk_boxes_needed_l1626_162647

theorem chalk_boxes_needed (pieces_per_box : ℕ) (total_pieces : ℕ) (pieces_per_box_pos : pieces_per_box > 0) : 
  (total_pieces + pieces_per_box - 1) / pieces_per_box = 194 :=
by 
  let boxes_needed := (total_pieces + pieces_per_box - 1) / pieces_per_box
  have h: boxes_needed = 194 := sorry
  exact h

end NUMINAMATH_GPT_chalk_boxes_needed_l1626_162647


namespace NUMINAMATH_GPT_solve_ineqs_l1626_162639

theorem solve_ineqs (a x : ℝ) (h1 : |x - 2 * a| ≤ 3) (h2 : 0 < x + a ∧ x + a ≤ 4) 
  (ha : a = 3) (hx : x = 1) : 
  (|x - 2 * a| ≤ 3) ∧ (0 < x + a ∧ x + a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_ineqs_l1626_162639


namespace NUMINAMATH_GPT_girls_with_rulers_l1626_162642

theorem girls_with_rulers 
  (total_students : ℕ) (students_with_rulers : ℕ) (boys_with_set_squares : ℕ) 
  (total_girls : ℕ) (student_count : total_students = 50) 
  (ruler_count : students_with_rulers = 28) 
  (boys_with_set_squares_count : boys_with_set_squares = 14) 
  (girl_count : total_girls = 31) 
  : total_girls - (total_students - students_with_rulers - boys_with_set_squares) = 23 := 
by
  sorry

end NUMINAMATH_GPT_girls_with_rulers_l1626_162642


namespace NUMINAMATH_GPT_complex_point_in_fourth_quadrant_l1626_162667

theorem complex_point_in_fourth_quadrant (z : ℂ) (h : z = 1 / (1 + I)) :
  z.re > 0 ∧ z.im < 0 :=
by
  -- Here we would provide the proof, but it is omitted as per the instructions.
  sorry

end NUMINAMATH_GPT_complex_point_in_fourth_quadrant_l1626_162667


namespace NUMINAMATH_GPT_harry_started_with_79_l1626_162680

-- Definitions using the conditions
def harry_initial_apples (x : ℕ) : Prop :=
  (x + 5 = 84)

-- Theorem statement proving the initial number of apples Harry started with
theorem harry_started_with_79 : ∃ x : ℕ, harry_initial_apples x ∧ x = 79 :=
by
  sorry

end NUMINAMATH_GPT_harry_started_with_79_l1626_162680


namespace NUMINAMATH_GPT_number_divided_by_3_equals_subtract_3_l1626_162673

theorem number_divided_by_3_equals_subtract_3 (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end NUMINAMATH_GPT_number_divided_by_3_equals_subtract_3_l1626_162673


namespace NUMINAMATH_GPT_number_of_persons_in_second_group_l1626_162631

-- Definitions based on conditions
def total_man_hours_first_group : ℕ := 42 * 12 * 5

def total_man_hours_second_group (X : ℕ) : ℕ := X * 14 * 6

-- Theorem stating that the number of persons in the second group is 30, given the conditions
theorem number_of_persons_in_second_group (X : ℕ) : 
  total_man_hours_first_group = total_man_hours_second_group X → X = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_persons_in_second_group_l1626_162631
