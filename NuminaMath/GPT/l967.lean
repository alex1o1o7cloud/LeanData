import Mathlib

namespace NUMINAMATH_GPT_exists_m_such_that_m_plus_one_pow_zero_eq_one_l967_96790

theorem exists_m_such_that_m_plus_one_pow_zero_eq_one : 
  ∃ m : ℤ, (m + 1)^0 = 1 ∧ m ≠ -1 :=
by
  sorry

end NUMINAMATH_GPT_exists_m_such_that_m_plus_one_pow_zero_eq_one_l967_96790


namespace NUMINAMATH_GPT_ScientificNotation_of_45400_l967_96783

theorem ScientificNotation_of_45400 :
  45400 = 4.54 * 10^4 := sorry

end NUMINAMATH_GPT_ScientificNotation_of_45400_l967_96783


namespace NUMINAMATH_GPT_Emmy_money_l967_96719

theorem Emmy_money {Gerry_money cost_per_apple number_of_apples Emmy_money : ℕ} 
    (h1 : Gerry_money = 100)
    (h2 : cost_per_apple = 2) 
    (h3 : number_of_apples = 150) 
    (h4 : number_of_apples * cost_per_apple = Gerry_money + Emmy_money) :
    Emmy_money = 200 :=
by
   sorry

end NUMINAMATH_GPT_Emmy_money_l967_96719


namespace NUMINAMATH_GPT_ceil_neg_seven_fourths_cubed_eq_neg_five_l967_96775

noncomputable def ceil_of_neg_seven_fourths_cubed : ℤ :=
  Int.ceil ((-7 / 4 : ℚ)^3)

theorem ceil_neg_seven_fourths_cubed_eq_neg_five :
  ceil_of_neg_seven_fourths_cubed = -5 := by
  sorry

end NUMINAMATH_GPT_ceil_neg_seven_fourths_cubed_eq_neg_five_l967_96775


namespace NUMINAMATH_GPT_mr_willam_land_percentage_over_taxable_land_l967_96772

def total_tax_collected : ℝ := 3840
def tax_paid_by_mr_willam : ℝ := 480
def farm_tax_percentage : ℝ := 0.45

theorem mr_willam_land_percentage_over_taxable_land :
  (tax_paid_by_mr_willam / total_tax_collected) * 100 = 5.625 :=
by
  sorry

end NUMINAMATH_GPT_mr_willam_land_percentage_over_taxable_land_l967_96772


namespace NUMINAMATH_GPT_correct_conclusions_l967_96782

-- Given function f with the specified domain and properties
variable {f : ℝ → ℝ}

-- Given conditions
axiom functional_eq (x y : ℝ) : f (x + y) + f (x - y) = 2 * f x * f y
axiom f_one_half : f (1/2) = 0
axiom f_zero_not_zero : f 0 ≠ 0

-- Proving our conclusions
theorem correct_conclusions :
  f 0 = 1 ∧ (∀ y : ℝ, f (1/2 + y) = -f (1/2 - y))
:=
by
  sorry

end NUMINAMATH_GPT_correct_conclusions_l967_96782


namespace NUMINAMATH_GPT_simplify_and_evaluate_l967_96785

theorem simplify_and_evaluate :
  ∀ (x : ℝ), x = -3 → 7 * x^2 - 3 * (2 * x^2 - 1) - 4 = 8 :=
by
  intros x hx
  rw [hx]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l967_96785


namespace NUMINAMATH_GPT_find_number_l967_96700

theorem find_number (x : ℝ) (h : (5/4) * x = 40) : x = 32 := 
sorry

end NUMINAMATH_GPT_find_number_l967_96700


namespace NUMINAMATH_GPT_x_condition_sufficient_not_necessary_l967_96726

theorem x_condition_sufficient_not_necessary (x : ℝ) : (x < -1 → x^2 - 1 > 0) ∧ (¬ (∀ x, x^2 - 1 > 0 → x < -1)) :=
by
  sorry

end NUMINAMATH_GPT_x_condition_sufficient_not_necessary_l967_96726


namespace NUMINAMATH_GPT_arithmetic_expression_l967_96744

theorem arithmetic_expression :
  (((15 - 2) + (4 / (1 / 2)) - (6 * 8)) * (100 - 24)) / 38 = -54 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_l967_96744


namespace NUMINAMATH_GPT_sample_size_is_15_l967_96704

-- Define the given conditions as constants and assumptions within the Lean environment.
def total_employees := 750
def young_workers := 350
def middle_aged_workers := 250
def elderly_workers := 150
def sample_young_workers := 7

-- Define the proposition that given these conditions, the sample size is 15.
theorem sample_size_is_15 : ∃ n : ℕ, (7 / n = 350 / 750) ∧ n = 15 := by
  sorry

end NUMINAMATH_GPT_sample_size_is_15_l967_96704


namespace NUMINAMATH_GPT_career_preference_degrees_l967_96724

variable (M F : ℕ)
variable (h1 : M / F = 2 / 3)
variable (preferred_males : ℚ := M / 4)
variable (preferred_females : ℚ := F / 2)
variable (total_students : ℚ := M + F)
variable (preferred_career_students : ℚ := preferred_males + preferred_females)
variable (career_fraction : ℚ := preferred_career_students / total_students)
variable (degrees : ℚ := 360 * career_fraction)

theorem career_preference_degrees :
  degrees = 144 :=
sorry

end NUMINAMATH_GPT_career_preference_degrees_l967_96724


namespace NUMINAMATH_GPT_circle_intersection_range_l967_96741

theorem circle_intersection_range (r : ℝ) (H : r > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ (x+3)^2 + (y-4)^2 = 36) → (1 < r ∧ r < 11) := 
by
  sorry

end NUMINAMATH_GPT_circle_intersection_range_l967_96741


namespace NUMINAMATH_GPT_domain_of_f_l967_96713

noncomputable def f (t : ℝ) : ℝ :=  1 / ((abs (t - 1))^2 + (abs (t + 1))^2)

theorem domain_of_f : ∀ t : ℝ, (abs (t - 1))^2 + (abs (t + 1))^2 ≠ 0 :=
by
  intro t
  sorry

end NUMINAMATH_GPT_domain_of_f_l967_96713


namespace NUMINAMATH_GPT_jack_finishes_book_in_13_days_l967_96792

def total_pages : ℕ := 285
def pages_per_day : ℕ := 23

theorem jack_finishes_book_in_13_days : (total_pages + pages_per_day - 1) / pages_per_day = 13 := by
  sorry

end NUMINAMATH_GPT_jack_finishes_book_in_13_days_l967_96792


namespace NUMINAMATH_GPT_probability_of_centrally_symmetric_card_l967_96723

def is_centrally_symmetric (shape : String) : Bool :=
  shape = "parallelogram" ∨ shape = "circle"

theorem probability_of_centrally_symmetric_card :
  let shapes := ["parallelogram", "isosceles_right_triangle", "regular_pentagon", "circle"]
  let total_cards := shapes.length
  let centrally_symmetric_cards := shapes.filter is_centrally_symmetric
  let num_centrally_symmetric := centrally_symmetric_cards.length
  (num_centrally_symmetric : ℚ) / (total_cards : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_centrally_symmetric_card_l967_96723


namespace NUMINAMATH_GPT_largest_n_for_positive_sum_l967_96771

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

def arithmetic_sum (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

theorem largest_n_for_positive_sum (n : ℕ) :
  ∀ (a : ℕ) (S : ℕ → ℤ), (a_1 = 9 ∧ a_5 = 1 ∧ S n > 0) → n = 9 :=
sorry

end NUMINAMATH_GPT_largest_n_for_positive_sum_l967_96771


namespace NUMINAMATH_GPT_speed_of_goods_train_l967_96768

open Real

theorem speed_of_goods_train
  (V_girl : ℝ := 100) -- The speed of the girl's train in km/h
  (t : ℝ := 6/3600)  -- The passing time in hours
  (L : ℝ := 560/1000) -- The length of the goods train in km
  (V_g : ℝ) -- The speed of the goods train in km/h
  : V_g = 236 := sorry

end NUMINAMATH_GPT_speed_of_goods_train_l967_96768


namespace NUMINAMATH_GPT_trigonometric_identity_l967_96767

theorem trigonometric_identity :
  (Real.cos (Real.pi / 3)) - (Real.tan (Real.pi / 4)) + (3 / 4) * (Real.tan (Real.pi / 6))^2 - (Real.sin (Real.pi / 6)) + (Real.cos (Real.pi / 6))^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l967_96767


namespace NUMINAMATH_GPT_calculate_overhead_cost_l967_96720

noncomputable def overhead_cost (prod_cost revenue_cost : ℕ) (num_performances : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost - num_performances * prod_cost

theorem calculate_overhead_cost :
  overhead_cost 7000 16000 9 (9 * 16000) = 81000 :=
by
  sorry

end NUMINAMATH_GPT_calculate_overhead_cost_l967_96720


namespace NUMINAMATH_GPT_modulus_of_complex_number_l967_96742

/-- Definition of the imaginary unit i defined as the square root of -1 --/
def i : ℂ := Complex.I

/-- Statement that the modulus of z = i (1 - i) equals sqrt(2) --/
theorem modulus_of_complex_number : Complex.abs (i * (1 - i)) = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_modulus_of_complex_number_l967_96742


namespace NUMINAMATH_GPT_flood_damage_in_euros_l967_96776

variable (yen_damage : ℕ) (yen_per_euro : ℕ) (tax_rate : ℝ)

theorem flood_damage_in_euros : 
  yen_damage = 4000000000 →
  yen_per_euro = 110 →
  tax_rate = 1.05 →
  (yen_damage / yen_per_euro : ℝ) * tax_rate = 38181818 :=
by {
  -- We could include necessary lean proof steps here, but we use sorry to skip the proof.
  sorry
}

end NUMINAMATH_GPT_flood_damage_in_euros_l967_96776


namespace NUMINAMATH_GPT_ellipse_eccentricity_l967_96711

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l967_96711


namespace NUMINAMATH_GPT_inequality_proof_l967_96715

theorem inequality_proof (x y z : ℝ) (hx : 2 < x) (hx4 : x < 4) (hy : 2 < y) (hy4 : y < 4) (hz : 2 < z) (hz4 : z < 4) :
  (x / (y^2 - z) + y / (z^2 - x) + z / (x^2 - y)) > 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l967_96715


namespace NUMINAMATH_GPT_stack_trays_height_l967_96759

theorem stack_trays_height
  (thickness : ℕ)
  (top_diameter : ℕ)
  (bottom_diameter : ℕ)
  (decrement_step : ℕ)
  (base_height : ℕ)
  (cond1 : thickness = 2)
  (cond2 : top_diameter = 30)
  (cond3 : bottom_diameter = 8)
  (cond4 : decrement_step = 2)
  (cond5 : base_height = 2) :
  (bottom_diameter + decrement_step * (top_diameter - bottom_diameter) / decrement_step * thickness + base_height) = 26 :=
by
  sorry

end NUMINAMATH_GPT_stack_trays_height_l967_96759


namespace NUMINAMATH_GPT_system_solution_is_unique_l967_96701

theorem system_solution_is_unique
  (a b : ℝ)
  (h1 : 2 - a * 5 = -1)
  (h2 : b + 3 * 5 = 8) :
  (∃ m n : ℝ, 2 * (m + n) - a * (m - n) = -1 ∧ b * (m + n) + 3 * (m - n) = 8 ∧ m = 3 ∧ n = -2) :=
by
  sorry

end NUMINAMATH_GPT_system_solution_is_unique_l967_96701


namespace NUMINAMATH_GPT_find_x_l967_96702

theorem find_x (x : ℝ) (h : (2012 + x)^2 = x^2) : x = -1006 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l967_96702


namespace NUMINAMATH_GPT_zoo_ticket_problem_l967_96779

def students_6A (total_cost_6A : ℕ) (saved_tickets_6A : ℕ) (ticket_price : ℕ) : ℕ :=
  let paid_tickets := (total_cost_6A / ticket_price)
  (paid_tickets + saved_tickets_6A)

def students_6B (total_cost_6B : ℕ) (total_students_6A : ℕ) (ticket_price : ℕ) : ℕ :=
  let paid_tickets := (total_cost_6B / ticket_price)
  let total_students := paid_tickets + (paid_tickets / 4)
  (total_students - total_students_6A)

theorem zoo_ticket_problem :
  (students_6A 1995 4 105 = 23) ∧
  (students_6B 4410 23 105 = 29) :=
by {
  -- The proof will follow the steps to confirm the calculations and final result
  sorry
}

end NUMINAMATH_GPT_zoo_ticket_problem_l967_96779


namespace NUMINAMATH_GPT_number_of_employees_excluding_manager_l967_96781

theorem number_of_employees_excluding_manager 
  (avg_salary : ℕ)
  (manager_salary : ℕ)
  (new_avg_salary : ℕ)
  (n : ℕ)
  (T : ℕ)
  (h1 : avg_salary = 1600)
  (h2 : manager_salary = 3700)
  (h3 : new_avg_salary = 1700)
  (h4 : T = n * avg_salary)
  (h5 : T + manager_salary = (n + 1) * new_avg_salary) :
  n = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_employees_excluding_manager_l967_96781


namespace NUMINAMATH_GPT_arcsin_eq_solution_domain_l967_96714

open Real

theorem arcsin_eq_solution_domain (x : ℝ) (hx1 : abs (x * sqrt 5 / 3) ≤ 1)
  (hx2 : abs (x * sqrt 5 / 6) ≤ 1)
  (hx3 : abs (7 * x * sqrt 5 / 18) ≤ 1) :
  arcsin (x * sqrt 5 / 3) + arcsin (x * sqrt 5 / 6) = arcsin (7 * x * sqrt 5 / 18) ↔ 
  x = 0 ∨ x = 8 / 7 ∨ x = -8 / 7 := sorry

end NUMINAMATH_GPT_arcsin_eq_solution_domain_l967_96714


namespace NUMINAMATH_GPT_least_number_with_remainder_l967_96732

variable (x : ℕ)

theorem least_number_with_remainder (x : ℕ) : 
  (x % 16 = 11) ∧ (x % 27 = 11) ∧ (x % 34 = 11) ∧ (x % 45 = 11) ∧ (x % 144 = 11) → x = 36731 := by
  sorry

end NUMINAMATH_GPT_least_number_with_remainder_l967_96732


namespace NUMINAMATH_GPT_minimum_value_of_a_l967_96740

noncomputable def inequality_valid_for_all_x (a : ℝ) : Prop :=
  ∀ (x : ℝ), 1 < x → x + a * Real.log x - x^a + 1 / Real.exp x ≥ 0

theorem minimum_value_of_a : ∃ a, inequality_valid_for_all_x a ∧ a = -Real.exp 1 := sorry

end NUMINAMATH_GPT_minimum_value_of_a_l967_96740


namespace NUMINAMATH_GPT_totalPayment_l967_96799

def totalNumberOfTrees : Nat := 850
def pricePerDouglasFir : Nat := 300
def pricePerPonderosaPine : Nat := 225
def numberOfDouglasFirPurchased : Nat := 350
def numberOfPonderosaPinePurchased := totalNumberOfTrees - numberOfDouglasFirPurchased

def costDouglasFir := numberOfDouglasFirPurchased * pricePerDouglasFir
def costPonderosaPine := numberOfPonderosaPinePurchased * pricePerPonderosaPine

def totalCost := costDouglasFir + costPonderosaPine

theorem totalPayment : totalCost = 217500 := by
  sorry

end NUMINAMATH_GPT_totalPayment_l967_96799


namespace NUMINAMATH_GPT_volume_of_rectangular_prism_l967_96737

    theorem volume_of_rectangular_prism (height base_perimeter: ℝ) (h: height = 5) (b: base_perimeter = 16) :
      ∃ volume, volume = 80 := 
    by
      -- Mathematically equivalent proof goes here
      sorry
    
end NUMINAMATH_GPT_volume_of_rectangular_prism_l967_96737


namespace NUMINAMATH_GPT_expression_of_f_l967_96764

theorem expression_of_f (f : ℤ → ℤ) (h : ∀ x, f (x - 1) = x^2 + 4 * x - 5) : ∀ x, f x = x^2 + 6 * x :=
by
  sorry

end NUMINAMATH_GPT_expression_of_f_l967_96764


namespace NUMINAMATH_GPT_inclination_angle_between_given_planes_l967_96736

noncomputable def Point (α : Type*) := α × α × α 

structure Plane (α : Type*) :=
(point : Point α)
(normal_vector : Point α)

def inclination_angle_between_planes (α : Type*) [Field α] (P1 P2 : Plane α) : α := 
  sorry

theorem inclination_angle_between_given_planes 
  (α : Type*) [Field α] 
  (A : Point α) 
  (n1 n2 : Point α) 
  (P1 : Plane α := Plane.mk A n1) 
  (P2 : Plane α := Plane.mk (1,0,0) n2) : 
  inclination_angle_between_planes α P1 P2 = sorry :=
sorry

end NUMINAMATH_GPT_inclination_angle_between_given_planes_l967_96736


namespace NUMINAMATH_GPT_slower_time_to_reach_top_l967_96751

def time_for_lola (stories : ℕ) (time_per_story : ℕ) : ℕ :=
  stories * time_per_story

def time_for_tara (stories : ℕ) (time_per_story : ℕ) (stopping_time : ℕ) (num_stops : ℕ) : ℕ :=
  (stories * time_per_story) + (num_stops * stopping_time)

theorem slower_time_to_reach_top (stories : ℕ) (lola_time_per_story : ℕ) (tara_time_per_story : ℕ) 
  (tara_stop_time : ℕ) (tara_num_stops : ℕ) : 
  stories = 20 
  → lola_time_per_story = 10 
  → tara_time_per_story = 8 
  → tara_stop_time = 3
  → tara_num_stops = 18
  → max (time_for_lola stories lola_time_per_story) (time_for_tara stories tara_time_per_story tara_stop_time tara_num_stops) = 214 :=
by sorry

end NUMINAMATH_GPT_slower_time_to_reach_top_l967_96751


namespace NUMINAMATH_GPT_housewife_money_left_l967_96793

theorem housewife_money_left (total : ℕ) (spent_fraction : ℚ) (spent : ℕ) (left : ℕ) :
  total = 150 → spent_fraction = 2 / 3 → spent = spent_fraction * total → left = total - spent → left = 50 :=
by
  intros
  sorry

end NUMINAMATH_GPT_housewife_money_left_l967_96793


namespace NUMINAMATH_GPT_profit_percentage_is_correct_l967_96730

noncomputable def shopkeeper_profit_percentage : ℚ :=
  let cost_A : ℚ := 12 * (15/16)
  let cost_B : ℚ := 18 * (47/50)
  let profit_A : ℚ := 12 - cost_A
  let profit_B : ℚ := 18 - cost_B
  let total_profit : ℚ := profit_A + profit_B
  let total_cost : ℚ := cost_A + cost_B
  (total_profit / total_cost) * 100

theorem profit_percentage_is_correct :
  shopkeeper_profit_percentage = 6.5 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_correct_l967_96730


namespace NUMINAMATH_GPT_total_number_of_drivers_l967_96777

theorem total_number_of_drivers (N : ℕ) (A_drivers : ℕ) (B_sample : ℕ) (C_sample : ℕ) (D_sample : ℕ)
  (A_sample : ℕ)
  (hA : A_drivers = 96)
  (hA_sample : A_sample = 12)
  (hB_sample : B_sample = 21)
  (hC_sample : C_sample = 25)
  (hD_sample : D_sample = 43) :
  N = 808 :=
by
  -- skipping the proof here
  sorry

end NUMINAMATH_GPT_total_number_of_drivers_l967_96777


namespace NUMINAMATH_GPT_matrix_power_15_l967_96727

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![ 0, -1,  0;
      1,  0,  0;
      0,  0,  1]

-- Define what we want to prove
theorem matrix_power_15 :
  B^15 = !![ 0,  1,  0;
            -1,  0,  0;
             0,  0,  1] :=
sorry

end NUMINAMATH_GPT_matrix_power_15_l967_96727


namespace NUMINAMATH_GPT_probability_not_finish_l967_96788

theorem probability_not_finish (p : ℝ) (h : p = 5 / 8) : 1 - p = 3 / 8 := 
by 
  rw [h]
  norm_num

end NUMINAMATH_GPT_probability_not_finish_l967_96788


namespace NUMINAMATH_GPT_eval_g_l967_96769

def g (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + x + 1

theorem eval_g : 3 * g 2 + 2 * g (-2) = -9 := 
by {
  sorry
}

end NUMINAMATH_GPT_eval_g_l967_96769


namespace NUMINAMATH_GPT_Reese_initial_savings_l967_96761

theorem Reese_initial_savings (F M A R : ℝ) (savings : ℝ) :
  F = 0.2 * savings →
  M = 0.4 * savings →
  A = 1500 →
  R = 2900 →
  savings = 11000 :=
by
  sorry

end NUMINAMATH_GPT_Reese_initial_savings_l967_96761


namespace NUMINAMATH_GPT_part1_intersection_1_part1_union_1_part2_range_a_l967_96746

open Set

def U := ℝ
def A (x : ℝ) := -1 < x ∧ x < 3
def B (a x : ℝ) := a - 1 ≤ x ∧ x ≤ a + 6

noncomputable def part1_a : ℝ → Prop := sorry
noncomputable def part1_b : ℝ → Prop := sorry

-- part (1)
theorem part1_intersection_1 (a : ℝ) : A x ∧ B a x := sorry

theorem part1_union_1 (a : ℝ) : A x ∨ B a x := sorry

-- part (2)
theorem part2_range_a : {a : ℝ | -3 ≤ a ∧ a ≤ 0} := sorry

end NUMINAMATH_GPT_part1_intersection_1_part1_union_1_part2_range_a_l967_96746


namespace NUMINAMATH_GPT_ratio_S6_S3_l967_96762

theorem ratio_S6_S3 (a : ℝ) (q : ℝ) (h : a + 8 * a * q^3 = 0) : 
  (a * (1 - q^6) / (1 - q)) / (a * (1 - q^3) / (1 - q)) = 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_S6_S3_l967_96762


namespace NUMINAMATH_GPT_proof_part_1_proof_part_2_l967_96789

variable {α : ℝ}

/-- Given tan(α) = 3, prove
  (1) (3 * sin(α) + 2 * cos(α))/(sin(α) - 4 * cos(α)) = -11 -/
theorem proof_part_1
  (h : Real.tan α = 3) :
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - 4 * Real.cos α) = -11 := 
by
  sorry

/-- Given tan(α) = 3, prove
  (2) (5 * cos^2(α) - 3 * sin^2(α))/(1 + sin^2(α)) = -11/5 -/
theorem proof_part_2
  (h : Real.tan α = 3) :
  (5 * (Real.cos α)^2 - 3 * (Real.sin α)^2) / (1 + (Real.sin α)^2) = -11 / 5 :=
by
  sorry

end NUMINAMATH_GPT_proof_part_1_proof_part_2_l967_96789


namespace NUMINAMATH_GPT_sin_beta_value_l967_96798

variable {α β : ℝ}
variable (h₁ : 0 < α ∧ α < β ∧ β < π / 2)
variable (h₂ : Real.sin α = 3 / 5)
variable (h₃ : Real.cos (β - α) = 12 / 13)

theorem sin_beta_value : Real.sin β = 56 / 65 :=
by
  sorry

end NUMINAMATH_GPT_sin_beta_value_l967_96798


namespace NUMINAMATH_GPT_alejandro_candies_l967_96722

theorem alejandro_candies (n : ℕ) (S_n : ℕ) :
  (S_n = 2^n - 1 ∧ S_n ≥ 2007) → ((2^11 - 1 - 2007 = 40) ∧ (∃ k, k = 11)) :=
  by
    sorry

end NUMINAMATH_GPT_alejandro_candies_l967_96722


namespace NUMINAMATH_GPT_initial_students_per_class_l967_96765

theorem initial_students_per_class (students_per_class initial_classes additional_classes total_students : ℕ) 
  (h1 : initial_classes = 15) 
  (h2 : additional_classes = 5) 
  (h3 : total_students = 400) 
  (h4 : students_per_class * (initial_classes + additional_classes) = total_students) : 
  students_per_class = 20 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_initial_students_per_class_l967_96765


namespace NUMINAMATH_GPT_simplify_expression_l967_96766

theorem simplify_expression (tan_60 cot_60 : ℝ) (h1 : tan_60 = Real.sqrt 3) (h2 : cot_60 = 1 / Real.sqrt 3) :
  (tan_60^3 + cot_60^3) / (tan_60 + cot_60) = 31 / 3 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_simplify_expression_l967_96766


namespace NUMINAMATH_GPT_bananas_added_l967_96786

variable (initial_bananas final_bananas added_bananas : ℕ)

-- Initial condition: There are 2 bananas initially
def initial_bananas_def : Prop := initial_bananas = 2

-- Final condition: There are 9 bananas finally
def final_bananas_def : Prop := final_bananas = 9

-- The number of bananas added to the pile
def added_bananas_def : Prop := final_bananas = initial_bananas + added_bananas

-- Proof statement: Prove that the number of bananas added is 7
theorem bananas_added (h1 : initial_bananas = 2) (h2 : final_bananas = 9) : added_bananas = 7 := by
  sorry

end NUMINAMATH_GPT_bananas_added_l967_96786


namespace NUMINAMATH_GPT_min_value_of_y_l967_96747

noncomputable def y (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2) - abs (x - 3)

theorem min_value_of_y : ∃ x : ℝ, (∀ x' : ℝ, y x' ≥ y x) ∧ y x = -1 :=
sorry

end NUMINAMATH_GPT_min_value_of_y_l967_96747


namespace NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l967_96709

noncomputable def prove_tenth_term (a d: ℤ) (h1: a + 2*d = 10) (h2: a + 7*d = 30) : Prop :=
  a + 9*d = 38

theorem arithmetic_sequence_tenth_term (a d: ℤ) (h1: a + 2*d = 10) (h2: a + 7*d = 30) : prove_tenth_term a d h1 h2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l967_96709


namespace NUMINAMATH_GPT_integer_coordinates_for_all_vertices_l967_96760

-- Define a three-dimensional vector with integer coordinates
structure Vec3 :=
  (x : ℤ)
  (y : ℤ)
  (z : ℤ)

-- Define a cube with 8 vertices in 3D space
structure Cube :=
  (A1 A2 A3 A4 A1' A2' A3' A4' : Vec3)

-- Assumption: four vertices with integer coordinates that do not lie on the same plane
def has_four_integer_vertices (cube : Cube) : Prop :=
  ∃ (A B C D : Vec3),
    A ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    B ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    C ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    D ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (C.x - A.x) * (D.y - B.y) ≠ (D.x - B.x) * (C.y - A.y) ∧  -- Ensure not co-planar
    (C.y - A.y) * (D.z - B.z) ≠ (D.y - B.y) * (C.z - A.z)

-- The proof problem: prove all vertices have integer coordinates given the condition
theorem integer_coordinates_for_all_vertices (cube : Cube) (h : has_four_integer_vertices cube) : 
  ∀ v ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'], 
    ∃ (v' : Vec3), v = v' := 
  by
  sorry

end NUMINAMATH_GPT_integer_coordinates_for_all_vertices_l967_96760


namespace NUMINAMATH_GPT_similar_triangles_side_length_l967_96734

theorem similar_triangles_side_length (A1 A2 : ℕ) (k : ℕ)
  (h1 : A1 - A2 = 32)
  (h2 : A1 = k^2 * A2)
  (h3 : A2 > 0)
  (side2 : ℕ) (h4 : side2 = 5) :
  ∃ side1 : ℕ, side1 = 3 * side2 ∧ side1 = 15 :=
by
  sorry

end NUMINAMATH_GPT_similar_triangles_side_length_l967_96734


namespace NUMINAMATH_GPT_lottery_ticket_not_necessarily_win_l967_96754

/-- Given a lottery with 1,000,000 tickets and a winning rate of 0.001, buying 1000 tickets may not necessarily win. -/
theorem lottery_ticket_not_necessarily_win (total_tickets : ℕ) (winning_rate : ℚ) (n_tickets : ℕ) :
  total_tickets = 1000000 →
  winning_rate = 1 / 1000 →
  n_tickets = 1000 →
  ∃ (p : ℚ), 0 < p ∧ p < 1 ∧ (p ^ n_tickets) < (1 / total_tickets) := 
by
  intros h_total h_rate h_n
  sorry

end NUMINAMATH_GPT_lottery_ticket_not_necessarily_win_l967_96754


namespace NUMINAMATH_GPT_contradiction_assumption_l967_96794

theorem contradiction_assumption (a : ℝ) (h : a < |a|) : ¬(a ≥ 0) :=
by 
  sorry

end NUMINAMATH_GPT_contradiction_assumption_l967_96794


namespace NUMINAMATH_GPT_trigonometric_transform_l967_96773

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := f (x - 3)
noncomputable def g (x : ℝ) : ℝ := 3 * h (x / 3)

theorem trigonometric_transform (x : ℝ) : g x = 3 * Real.sin (x / 3 - 3) := by
  sorry

end NUMINAMATH_GPT_trigonometric_transform_l967_96773


namespace NUMINAMATH_GPT_day_of_week_after_45_days_l967_96778

theorem day_of_week_after_45_days (day_of_week : ℕ → String) (birthday_is_tuesday : day_of_week 0 = "Tuesday") : day_of_week 45 = "Friday" :=
by
  sorry

end NUMINAMATH_GPT_day_of_week_after_45_days_l967_96778


namespace NUMINAMATH_GPT_find_x_minus_y_l967_96716

-- Variables and conditions
variables (x y : ℝ)
def abs_x_eq_3 := abs x = 3
def y_sq_eq_one_fourth := y^2 = 1 / 4
def x_plus_y_neg := x + y < 0

-- Proof problem stating that x - y must equal one of the two possible values
theorem find_x_minus_y (h1 : abs x = 3) (h2 : y^2 = 1 / 4) (h3 : x + y < 0) : 
  x - y = -7 / 2 ∨ x - y = -5 / 2 :=
  sorry

end NUMINAMATH_GPT_find_x_minus_y_l967_96716


namespace NUMINAMATH_GPT_cylindrical_tank_depth_l967_96752

theorem cylindrical_tank_depth (V : ℝ) (d h : ℝ) (π : ℝ) : 
  V = 1848 ∧ d = 14 ∧ π = Real.pi → h = 12 :=
by
  sorry

end NUMINAMATH_GPT_cylindrical_tank_depth_l967_96752


namespace NUMINAMATH_GPT_marlon_gift_card_balance_l967_96705

theorem marlon_gift_card_balance 
  (initial_amount : ℕ) 
  (spent_monday : initial_amount / 2 = 100)
  (spent_tuesday : (initial_amount / 2) / 4 = 25) 
  : (initial_amount / 2) - (initial_amount / 2 / 4) = 75 :=
by
  sorry

end NUMINAMATH_GPT_marlon_gift_card_balance_l967_96705


namespace NUMINAMATH_GPT_find_T_b_plus_T_neg_b_l967_96797

noncomputable def T (r : ℝ) : ℝ := 15 / (1 - r)

theorem find_T_b_plus_T_neg_b (b : ℝ) (h1 : -1 < b) (h2 : b < 1) (h3 : T b * T (-b) = 3600) :
  T b + T (-b) = 480 :=
sorry

end NUMINAMATH_GPT_find_T_b_plus_T_neg_b_l967_96797


namespace NUMINAMATH_GPT_relationship_between_x_and_y_l967_96755

theorem relationship_between_x_and_y
  (z : ℤ)
  (x : ℝ)
  (y : ℝ)
  (h1 : x = (z^4 + z^3 + z^2 + z + 1) / (z^2 + 1))
  (h2 : y = (z^3 + z^2 + z + 1) / (z^2 + 1)) :
  (y^2 - 2 * y + 2) * (x + y - y^2) - 1 = 0 := 
by
  sorry

end NUMINAMATH_GPT_relationship_between_x_and_y_l967_96755


namespace NUMINAMATH_GPT_minimum_weights_l967_96739

variable {α : Type} [LinearOrderedField α]

theorem minimum_weights (weights : Finset α)
  (h_unique : weights.card = 5)
  (h_balanced : ∀ {x y : α}, x ∈ weights → y ∈ weights → x ≠ y →
    ∃ a b : α, a ∈ weights ∧ b ∈ weights ∧ x + y = a + b) :
  ∃ (n : ℕ), n = 13 ∧ ∀ S : Finset α, S.card = n ∧
    (∀ {x y : α}, x ∈ S → y ∈ S → x ≠ y → ∃ a b : α, a ∈ S ∧ b ∈ S ∧ x + y = a + b) :=
by
  sorry

end NUMINAMATH_GPT_minimum_weights_l967_96739


namespace NUMINAMATH_GPT_james_prom_cost_l967_96708

def total_cost (ticket_cost dinner_cost tip_percent limo_cost_per_hour limo_hours tuxedo_cost persons : ℕ) : ℕ :=
  (ticket_cost * persons) +
  ((dinner_cost * persons) + (tip_percent * dinner_cost * persons) / 100) +
  (limo_cost_per_hour * limo_hours) + tuxedo_cost

theorem james_prom_cost :
  total_cost 100 120 30 80 8 150 4 = 1814 :=
by
  sorry

end NUMINAMATH_GPT_james_prom_cost_l967_96708


namespace NUMINAMATH_GPT_maximum_weight_truck_can_carry_l967_96717

-- Definitions for the conditions.
def weight_boxes : Nat := 100 * 100
def weight_crates : Nat := 10 * 60
def weight_sacks : Nat := 50 * 50
def weight_additional_bags : Nat := 10 * 40

-- Summing up all the weights.
def total_weight : Nat :=
  weight_boxes + weight_crates + weight_sacks + weight_additional_bags

-- The theorem stating the maximum weight.
theorem maximum_weight_truck_can_carry : total_weight = 13500 := by
  sorry

end NUMINAMATH_GPT_maximum_weight_truck_can_carry_l967_96717


namespace NUMINAMATH_GPT_meet_at_starting_line_l967_96706

theorem meet_at_starting_line (henry_time margo_time : ℕ) (h_henry : henry_time = 7) (h_margo : margo_time = 12) : Nat.lcm henry_time margo_time = 84 :=
by
  rw [h_henry, h_margo]
  sorry

end NUMINAMATH_GPT_meet_at_starting_line_l967_96706


namespace NUMINAMATH_GPT_length_of_segment_from_vertex_to_center_of_regular_hexagon_is_16_l967_96748

def hexagon_vertex_to_center_length (a : ℝ) (h : a = 16) (regular_hexagon : Prop) : Prop :=
∃ (O A : ℝ), (a = 16) → (regular_hexagon = true) → (O = 0) ∧ (A = a) ∧ (dist O A = 16)

theorem length_of_segment_from_vertex_to_center_of_regular_hexagon_is_16 :
  hexagon_vertex_to_center_length 16 (by rfl) true :=
sorry

end NUMINAMATH_GPT_length_of_segment_from_vertex_to_center_of_regular_hexagon_is_16_l967_96748


namespace NUMINAMATH_GPT_find_y_given_conditions_l967_96735

theorem find_y_given_conditions (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 5 * t + 9) (h3 : x = 0) : y = 33 / 2 := by
  sorry

end NUMINAMATH_GPT_find_y_given_conditions_l967_96735


namespace NUMINAMATH_GPT_equation_II_consecutive_integers_l967_96763

theorem equation_II_consecutive_integers :
  ∃ x y z w : ℕ, x + y + z + w = 46 ∧ [x, x+1, x+2, x+3] = [x, y, z, w] :=
by
  sorry

end NUMINAMATH_GPT_equation_II_consecutive_integers_l967_96763


namespace NUMINAMATH_GPT_sin_pi_plus_alpha_l967_96770

/-- Given that \(\sin \left(\frac{\pi}{2}+\alpha \right) = \frac{3}{5}\)
    and \(\alpha \in (0, \frac{\pi}{2})\),
    prove that \(\sin(\pi + \alpha) = -\frac{4}{5}\). -/
theorem sin_pi_plus_alpha (α : ℝ) (h1 : Real.sin (Real.pi / 2 + α) = 3 / 5)
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (Real.pi + α) = -4 / 5 := 
  sorry

end NUMINAMATH_GPT_sin_pi_plus_alpha_l967_96770


namespace NUMINAMATH_GPT_x0_range_l967_96710

noncomputable def f (x : ℝ) := (1 / 2) ^ x - Real.log x

theorem x0_range (x0 : ℝ) (h : f x0 > 1 / 2) : 0 < x0 ∧ x0 < 1 :=
by
  sorry

end NUMINAMATH_GPT_x0_range_l967_96710


namespace NUMINAMATH_GPT_circle_line_tangent_l967_96745

theorem circle_line_tangent (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = 4 * m ∧ x + y = 2 * m) ↔ m = 2 :=
sorry

end NUMINAMATH_GPT_circle_line_tangent_l967_96745


namespace NUMINAMATH_GPT_intersection_of_cylinders_within_sphere_l967_96725

theorem intersection_of_cylinders_within_sphere (a b c d e f : ℝ) :
    ∀ (x y z : ℝ), 
      (x - a)^2 + (y - b)^2 < 1 ∧ 
      (y - c)^2 + (z - d)^2 < 1 ∧ 
      (z - e)^2 + (x - f)^2 < 1 → 
      (x - (a + f) / 2)^2 + (y - (b + c) / 2)^2 + (z - (d + e) / 2)^2 < 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_cylinders_within_sphere_l967_96725


namespace NUMINAMATH_GPT_daria_needs_to_earn_l967_96731

variable (ticket_cost : ℕ) (current_money : ℕ) (total_tickets : ℕ)

def total_cost (ticket_cost : ℕ) (total_tickets : ℕ) : ℕ :=
  ticket_cost * total_tickets

def money_needed (total_cost : ℕ) (current_money : ℕ) : ℕ :=
  total_cost - current_money

theorem daria_needs_to_earn :
  total_cost 90 4 - 189 = 171 :=
by
  sorry

end NUMINAMATH_GPT_daria_needs_to_earn_l967_96731


namespace NUMINAMATH_GPT_smallest_n_condition_l967_96712

def pow_mod (a b m : ℕ) : ℕ := a^(b % m)

def n (r s : ℕ) : ℕ := 2^r - 16^s

def r_condition (r : ℕ) : Prop := ∃ k : ℕ, r = 3 * k + 1

def s_condition (s : ℕ) : Prop := ∃ h : ℕ, s = 3 * h + 2

theorem smallest_n_condition (r s : ℕ) (hr : r_condition r) (hs : s_condition s) :
  (n r s) % 7 = 5 → (n r s) = 768 := sorry

end NUMINAMATH_GPT_smallest_n_condition_l967_96712


namespace NUMINAMATH_GPT_speed_of_current_l967_96784

theorem speed_of_current (v_b v_c v_d : ℝ) (hd : v_d = 15) 
  (hvd1 : v_b + v_c = v_d) (hvd2 : v_b - v_c = 12) :
  v_c = 1.5 :=
by sorry

end NUMINAMATH_GPT_speed_of_current_l967_96784


namespace NUMINAMATH_GPT_find_t_l967_96758

theorem find_t (t : ℝ) :
  (2 * t - 7) * (3 * t - 4) = (3 * t - 9) * (2 * t - 6) →
  t = 26 / 7 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_find_t_l967_96758


namespace NUMINAMATH_GPT_survival_rate_is_100_percent_l967_96791

-- Definitions of conditions
def planted_trees : ℕ := 99
def survived_trees : ℕ := 99

-- Definition of survival rate
def survival_rate : ℕ := (survived_trees * 100) / planted_trees

-- Proof statement
theorem survival_rate_is_100_percent : survival_rate = 100 := by
  sorry

end NUMINAMATH_GPT_survival_rate_is_100_percent_l967_96791


namespace NUMINAMATH_GPT_ellipse_line_intersection_l967_96753

theorem ellipse_line_intersection (m : ℝ) : 
  (m > 0 ∧ m ≠ 3) →
  (∃ x y : ℝ, (x^2 / 3 + y^2 / m = 1) ∧ (x + 2 * y - 2 = 0)) ↔ 
  ((1 / 4 < m ∧ m < 3) ∨ (m > 3)) := 
by 
  sorry

end NUMINAMATH_GPT_ellipse_line_intersection_l967_96753


namespace NUMINAMATH_GPT_sad_children_count_l967_96774

theorem sad_children_count (total_children happy_children neither_happy_nor_sad children sad_children : ℕ)
  (h_total : total_children = 60)
  (h_happy : happy_children = 30)
  (h_neither : neither_happy_nor_sad = 20)
  (boys girls happy_boys sad_girls neither_boys : ℕ)
  (h_boys : boys = 17)
  (h_girls : girls = 43)
  (h_happy_boys : happy_boys = 6)
  (h_sad_girls : sad_girls = 4)
  (h_neither_boys : neither_boys = 5) :
  sad_children = total_children - happy_children - neither_happy_nor_sad :=
by sorry

end NUMINAMATH_GPT_sad_children_count_l967_96774


namespace NUMINAMATH_GPT_expression_divisible_by_10_l967_96756

theorem expression_divisible_by_10 (n : ℕ) : 10 ∣ (3 ^ (n + 2) - 2 ^ (n + 2) + 3 ^ n - 2 ^ n) :=
  sorry

end NUMINAMATH_GPT_expression_divisible_by_10_l967_96756


namespace NUMINAMATH_GPT_molecular_weight_H_of_H2CrO4_is_correct_l967_96787

-- Define the atomic weight of hydrogen
def atomic_weight_H : ℝ := 1.008

-- Define the number of hydrogen atoms in H2CrO4
def num_H_atoms_in_H2CrO4 : ℕ := 2

-- Define the molecular weight of the compound H2CrO4
def molecular_weight_H2CrO4 : ℝ := 118

-- Define the molecular weight of the hydrogen part (H2)
def molecular_weight_H2 : ℝ := atomic_weight_H * num_H_atoms_in_H2CrO4

-- The statement to prove
theorem molecular_weight_H_of_H2CrO4_is_correct : molecular_weight_H2 = 2.016 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_H_of_H2CrO4_is_correct_l967_96787


namespace NUMINAMATH_GPT_solve_for_a_l967_96703

theorem solve_for_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a < 13) (h3 : (51^2012 + a) % 13 = 0) : a = 12 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_a_l967_96703


namespace NUMINAMATH_GPT_option_D_is_correct_option_A_is_incorrect_option_B_is_incorrect_option_C_is_incorrect_l967_96780

variable (a b x : ℝ)

theorem option_D_is_correct :
  (2 * x + 1) * (x - 2) = 2 * x^2 - 3 * x - 2 :=
by sorry

theorem option_A_is_incorrect :
  2 * a^2 * b * 3 * a^2 * b^2 ≠ 6 * a^6 * b^3 :=
by sorry

theorem option_B_is_incorrect :
  0.00076 ≠ 7.6 * 10^4 :=
by sorry

theorem option_C_is_incorrect :
  -2 * a * (a + b) ≠ -2 * a^2 + 2 * a * b :=
by sorry

end NUMINAMATH_GPT_option_D_is_correct_option_A_is_incorrect_option_B_is_incorrect_option_C_is_incorrect_l967_96780


namespace NUMINAMATH_GPT_haley_seeds_l967_96718

theorem haley_seeds (total_seeds seeds_big_garden total_small_gardens seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 56)
  (h2 : seeds_big_garden = 35)
  (h3 : total_small_gardens = 7)
  (h4 : total_seeds - seeds_big_garden = 21)
  (h5 : 21 / total_small_gardens = seeds_per_small_garden) :
  seeds_per_small_garden = 3 :=
by sorry

end NUMINAMATH_GPT_haley_seeds_l967_96718


namespace NUMINAMATH_GPT_analysis_error_l967_96750

theorem analysis_error (x : ℝ) (h1 : x + 1 / x ≥ 2) : 
  x + 1 / x ≥ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_analysis_error_l967_96750


namespace NUMINAMATH_GPT_remainder_71_73_div_8_l967_96733

theorem remainder_71_73_div_8 :
  (71 * 73) % 8 = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_71_73_div_8_l967_96733


namespace NUMINAMATH_GPT_no_solution_exists_l967_96796

theorem no_solution_exists :
  ¬ ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 150 ∧ n % 8 = 0 ∧ n % 10 = 6 ∧ n % 7 = 6 := 
by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l967_96796


namespace NUMINAMATH_GPT_length_of_median_in_right_triangle_l967_96749

noncomputable def length_of_median (DE DF : ℝ) : ℝ :=
  let EF := Real.sqrt (DE^2 + DF^2)
  EF / 2

theorem length_of_median_in_right_triangle (DE DF : ℝ) (h1 : DE = 5) (h2 : DF = 12) :
  length_of_median DE DF = 6.5 :=
by
  -- Conditions
  rw [h1, h2]
  -- Proof (to be completed)
  sorry

end NUMINAMATH_GPT_length_of_median_in_right_triangle_l967_96749


namespace NUMINAMATH_GPT_abs_inequality_solution_l967_96729

theorem abs_inequality_solution (x : ℝ) : (|3 - x| < 4) ↔ (-1 < x ∧ x < 7) :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l967_96729


namespace NUMINAMATH_GPT_arithmetic_sequence_30th_term_l967_96743

theorem arithmetic_sequence_30th_term :
  let a₁ := 3
  let d := 4
  let n := 30
  a₁ + (n - 1) * d = 119 :=
by
  let a₁ := 3
  let d := 4
  let n := 30
  show a₁ + (n - 1) * d = 119
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_30th_term_l967_96743


namespace NUMINAMATH_GPT_power_of_power_l967_96728

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end NUMINAMATH_GPT_power_of_power_l967_96728


namespace NUMINAMATH_GPT_find_number_l967_96707

theorem find_number (n : ℕ) :
  (n % 12 = 11) ∧
  (n % 11 = 10) ∧
  (n % 10 = 9) ∧
  (n % 9 = 8) ∧
  (n % 8 = 7) ∧
  (n % 7 = 6) ∧
  (n % 6 = 5) ∧
  (n % 5 = 4) ∧
  (n % 4 = 3) ∧
  (n % 3 = 2) ∧
  (n % 2 = 1)
  → n = 27719 := 
sorry

end NUMINAMATH_GPT_find_number_l967_96707


namespace NUMINAMATH_GPT_sum_of_x_y_l967_96721

theorem sum_of_x_y (x y : ℕ) (h1 : 10 * x + y = 75) (h2 : 10 * y + x = 57) : x + y = 12 :=
sorry

end NUMINAMATH_GPT_sum_of_x_y_l967_96721


namespace NUMINAMATH_GPT_circumference_of_cone_base_l967_96757

theorem circumference_of_cone_base (V : ℝ) (h : ℝ) (C : ℝ) (π := Real.pi) 
  (volume_eq : V = 24 * π) (height_eq : h = 6) 
  (circumference_eq : C = 4 * Real.sqrt 3 * π) :
  ∃ r : ℝ, (V = (1 / 3) * π * r^2 * h) ∧ (C = 2 * π * r) :=
by
  sorry

end NUMINAMATH_GPT_circumference_of_cone_base_l967_96757


namespace NUMINAMATH_GPT_ceil_neg_sqrt_frac_l967_96795

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := 
sorry

end NUMINAMATH_GPT_ceil_neg_sqrt_frac_l967_96795


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l967_96738

theorem batsman_average_after_17th_inning
  (A : ℝ)
  (h1 : A + 10 = (16 * A + 200) / 17)
  : (A = 30 ∧ (A + 10) = 40) :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l967_96738
