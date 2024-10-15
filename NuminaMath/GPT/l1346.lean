import Mathlib

namespace NUMINAMATH_GPT_find_a_l1346_134635

-- Given conditions and definitions
def circle_eq (x y : ℝ) : Prop := (x^2 + y^2 - 2*x - 2*y + 1 = 0)
def line_eq (x y a : ℝ) : Prop := (x - 2*y + a = 0)
def chord_length (r : ℝ) : ℝ := 2 * r

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y) → 
  (∀ x y : ℝ, line_eq x y a) → 
  (∃ x y : ℝ, (x = 1 ∧ y = 1) ∧ (line_eq x y a ∧ chord_length 1 = 2)) → 
  a = 1 := by sorry

end NUMINAMATH_GPT_find_a_l1346_134635


namespace NUMINAMATH_GPT_expand_product_l1346_134641

-- Define the expressions (x + 3)(x + 8) and x^2 + 11x + 24
def expr1 (x : ℝ) : ℝ := (x + 3) * (x + 8)
def expr2 (x : ℝ) : ℝ := x^2 + 11 * x + 24

-- Prove that the two expressions are equal
theorem expand_product (x : ℝ) : expr1 x = expr2 x := by
  sorry

end NUMINAMATH_GPT_expand_product_l1346_134641


namespace NUMINAMATH_GPT_area_of_circle_l1346_134669

theorem area_of_circle (x y : ℝ) :
  (x^2 + y^2 - 8*x - 6*y = -9) → 
  (∃ (R : ℝ), (x - 4)^2 + (y - 3)^2 = R^2 ∧ π * R^2 = 16 * π) :=
by
  sorry

end NUMINAMATH_GPT_area_of_circle_l1346_134669


namespace NUMINAMATH_GPT_deductive_reasoning_example_l1346_134609

-- Definitions for the conditions
def Metal (x : Type) : Prop := sorry
def ConductsElectricity (x : Type) : Prop := sorry
def Iron : Type := sorry

-- The problem statement
theorem deductive_reasoning_example (H1 : ∀ x, Metal x → ConductsElectricity x) (H2 : Metal Iron) : ConductsElectricity Iron :=
by sorry

end NUMINAMATH_GPT_deductive_reasoning_example_l1346_134609


namespace NUMINAMATH_GPT_distance_to_nearest_town_l1346_134666

theorem distance_to_nearest_town (d : ℝ) :
  ¬ (d ≥ 6) → ¬ (d ≤ 5) → ¬ (d ≤ 4) → (d > 5 ∧ d < 6) :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_distance_to_nearest_town_l1346_134666


namespace NUMINAMATH_GPT_radius_increase_rate_l1346_134659

theorem radius_increase_rate (r : ℝ) (u : ℝ)
  (h : r = 20) (dS_dt : ℝ) (h_dS_dt : dS_dt = 10 * Real.pi) :
  u = 1 / 4 :=
by
  have S := Real.pi * r^2
  have dS_dt_eq : dS_dt = 2 * Real.pi * r * u := sorry
  rw [h_dS_dt, h] at dS_dt_eq
  exact sorry

end NUMINAMATH_GPT_radius_increase_rate_l1346_134659


namespace NUMINAMATH_GPT_ratio_cost_to_marked_l1346_134685

variable (m : ℝ)

def marked_price (m : ℝ) := m

def selling_price (m : ℝ) : ℝ := 0.75 * m

def cost_price (m : ℝ) : ℝ := 0.60 * selling_price m

theorem ratio_cost_to_marked (m : ℝ) : 
  cost_price m / marked_price m = 0.45 := 
by
  sorry

end NUMINAMATH_GPT_ratio_cost_to_marked_l1346_134685


namespace NUMINAMATH_GPT_Gunther_free_time_left_l1346_134651

def vacuuming_time := 45
def dusting_time := 60
def folding_laundry_time := 25
def mopping_time := 30
def cleaning_bathroom_time := 40
def wiping_windows_time := 15
def brushing_cats_time := 4 * 5
def washing_dishes_time := 20
def first_tasks_total_time := 2 * 60 + 30
def available_free_time := 5 * 60

theorem Gunther_free_time_left : 
  (available_free_time - 
   (vacuuming_time + dusting_time + folding_laundry_time + 
    mopping_time + cleaning_bathroom_time + 
    wiping_windows_time + brushing_cats_time + 
    washing_dishes_time) = 45) := 
by 
  sorry

end NUMINAMATH_GPT_Gunther_free_time_left_l1346_134651


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l1346_134661

noncomputable def a : ℝ := 3 ^ Real.cos (Real.pi / 6)
noncomputable def b : ℝ := Real.log (Real.sin (Real.pi / 6)) / Real.log (1 / 3)
noncomputable def c : ℝ := Real.log (Real.tan (Real.pi / 6)) / Real.log 2

theorem relationship_among_a_b_c : a > b ∧ b > c := 
by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l1346_134661


namespace NUMINAMATH_GPT_percent_voters_for_candidate_A_l1346_134631

theorem percent_voters_for_candidate_A (d r i u p_d p_r p_i p_u : ℝ) 
  (hd : d = 0.45) (hr : r = 0.30) (hi : i = 0.20) (hu : u = 0.05)
  (hp_d : p_d = 0.75) (hp_r : p_r = 0.25) (hp_i : p_i = 0.50) (hp_u : p_u = 0.50) :
  d * p_d + r * p_r + i * p_i + u * p_u = 0.5375 :=
by
  sorry

end NUMINAMATH_GPT_percent_voters_for_candidate_A_l1346_134631


namespace NUMINAMATH_GPT_money_left_for_lunch_and_snacks_l1346_134638

-- Definitions according to the conditions
def ticket_cost_per_person : ℝ := 5
def bus_fare_one_way_per_person : ℝ := 1.50
def total_budget : ℝ := 40
def number_of_people : ℝ := 2

-- The proposition to be proved
theorem money_left_for_lunch_and_snacks : 
  let total_zoo_cost := ticket_cost_per_person * number_of_people
  let total_bus_fare := bus_fare_one_way_per_person * number_of_people * 2
  let total_expense := total_zoo_cost + total_bus_fare
  total_budget - total_expense = 24 :=
by
  sorry

end NUMINAMATH_GPT_money_left_for_lunch_and_snacks_l1346_134638


namespace NUMINAMATH_GPT_average_last_four_numbers_l1346_134622

theorem average_last_four_numbers (numbers : List ℝ) 
  (h1 : numbers.length = 7)
  (h2 : (numbers.sum / 7) = 62)
  (h3 : (numbers.take 3).sum / 3 = 58) : 
  ((numbers.drop 3).sum / 4) = 65 :=
by
  sorry

end NUMINAMATH_GPT_average_last_four_numbers_l1346_134622


namespace NUMINAMATH_GPT_tate_initial_tickets_l1346_134644

theorem tate_initial_tickets (T : ℕ) (h1 : T + 2 + (T + 2)/2 = 51) : T = 32 := 
by
  sorry

end NUMINAMATH_GPT_tate_initial_tickets_l1346_134644


namespace NUMINAMATH_GPT_asha_remaining_money_l1346_134655

theorem asha_remaining_money :
  let brother := 20
  let father := 40
  let mother := 30
  let granny := 70
  let savings := 100
  let total_money := brother + father + mother + granny + savings
  let spent := (3 / 4) * total_money
  let remaining := total_money - spent
  remaining = 65 :=
by
  sorry

end NUMINAMATH_GPT_asha_remaining_money_l1346_134655


namespace NUMINAMATH_GPT_factor_81_minus_4y4_l1346_134660

theorem factor_81_minus_4y4 (y : ℝ) : 81 - 4 * y^4 = (9 + 2 * y^2) * (9 - 2 * y^2) := by 
    sorry

end NUMINAMATH_GPT_factor_81_minus_4y4_l1346_134660


namespace NUMINAMATH_GPT_range_of_a_l1346_134600

def p (a : ℝ) : Prop := ∀ k : ℝ, ∃ x y : ℝ, (y = k * x + 1) ∧ (x^2 + (y^2) / a = 1)
def q (a : ℝ) : Prop := ∃ x0 : ℝ, 4^x0 - 2^x0 - a ≤ 0

theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) → -1/4 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1346_134600


namespace NUMINAMATH_GPT_conic_section_eccentricity_l1346_134614

noncomputable def eccentricity (m : ℝ) : ℝ :=
if m = 2 then 1 / Real.sqrt 2 else
if m = -2 then Real.sqrt 3 else
0

theorem conic_section_eccentricity (m : ℝ) (h : 4 * 1 = m * m) :
  eccentricity m = 1 / Real.sqrt 2 ∨ eccentricity m = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_conic_section_eccentricity_l1346_134614


namespace NUMINAMATH_GPT_largest_avg_5_l1346_134603

def arithmetic_avg (a l : ℕ) : ℚ :=
  (a + l) / 2

def multiples_avg_2 (n : ℕ) : ℚ :=
  arithmetic_avg 2 (n - (n % 2))

def multiples_avg_3 (n : ℕ) : ℚ :=
  arithmetic_avg 3 (n - (n % 3))

def multiples_avg_4 (n : ℕ) : ℚ :=
  arithmetic_avg 4 (n - (n % 4))

def multiples_avg_5 (n : ℕ) : ℚ :=
  arithmetic_avg 5 (n - (n % 5))

def multiples_avg_6 (n : ℕ) : ℚ :=
  arithmetic_avg 6 (n - (n % 6))

theorem largest_avg_5 (n : ℕ) (h : n = 101) : 
  multiples_avg_5 n > multiples_avg_2 n ∧ 
  multiples_avg_5 n > multiples_avg_3 n ∧ 
  multiples_avg_5 n > multiples_avg_4 n ∧ 
  multiples_avg_5 n > multiples_avg_6 n :=
by
  sorry

end NUMINAMATH_GPT_largest_avg_5_l1346_134603


namespace NUMINAMATH_GPT_length_of_AB_is_1_l1346_134618

variables {A B C : ℝ} -- Points defining the triangle vertices
variables {a b c : ℝ} -- Lengths of triangle sides opposite to angles A, B, C respectively
variables {α β γ : ℝ} -- Angles at points A B C
variables {s₁ s₂ s₃ : ℝ} -- Sin values of the angles

noncomputable def length_of_AB (a b c : ℝ) : ℝ :=
  if a + b + c = 4 ∧ a + b = 3 * c then 1 else 0

theorem length_of_AB_is_1 : length_of_AB a b c = 1 :=
by
  have h_perimeter : a + b + c = 4 := sorry
  have h_sin_condition : a + b = 3 * c := sorry
  simp [length_of_AB, h_perimeter, h_sin_condition]
  sorry

end NUMINAMATH_GPT_length_of_AB_is_1_l1346_134618


namespace NUMINAMATH_GPT_find_shorter_parallel_side_l1346_134633

variable (x : ℝ) (a : ℝ) (b : ℝ) (h : ℝ)

def is_trapezium_area (a b h : ℝ) (area : ℝ) : Prop :=
  area = 1/2 * (a + b) * h

theorem find_shorter_parallel_side
  (h28 : a = 28)
  (h15 : h = 15)
  (hArea : area = 345)
  (hIsTrapezium : is_trapezium_area a b h area):
  b = 18 := 
sorry

end NUMINAMATH_GPT_find_shorter_parallel_side_l1346_134633


namespace NUMINAMATH_GPT_exponential_inequality_l1346_134627

-- Define the conditions for the problem
variables {x y a : ℝ}
axiom h1 : x > y
axiom h2 : y > 1
axiom h3 : 0 < a
axiom h4 : a < 1

-- State the problem to be proved
theorem exponential_inequality (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < 1) : a ^ x < a ^ y :=
sorry

end NUMINAMATH_GPT_exponential_inequality_l1346_134627


namespace NUMINAMATH_GPT_sin_pi_div_three_l1346_134695

theorem sin_pi_div_three : Real.sin (π / 3) = Real.sqrt 3 / 2 := 
sorry

end NUMINAMATH_GPT_sin_pi_div_three_l1346_134695


namespace NUMINAMATH_GPT_average_birds_seen_l1346_134677

def MarcusBirds : Nat := 7
def HumphreyBirds : Nat := 11
def DarrelBirds : Nat := 9
def IsabellaBirds : Nat := 15

def totalBirds : Nat := MarcusBirds + HumphreyBirds + DarrelBirds + IsabellaBirds
def numberOfIndividuals : Nat := 4

theorem average_birds_seen : (totalBirds / numberOfIndividuals : Real) = 10.5 := 
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_average_birds_seen_l1346_134677


namespace NUMINAMATH_GPT_find_a_l1346_134675

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 < a^2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}
def C : Set ℝ := {x | 1 < x ∧ x < 2}

theorem find_a (a : ℝ) (h : A a ∩ B = C) : a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_GPT_find_a_l1346_134675


namespace NUMINAMATH_GPT_find_distance_walker_l1346_134673

noncomputable def distance_walked (x t d : ℝ) : Prop :=
  (d = x * t) ∧
  (d = (x + 1) * (3 / 4) * t) ∧
  (d = (x - 1) * (t + 3))

theorem find_distance_walker (x t d : ℝ) (h : distance_walked x t d) : d = 18 := 
sorry

end NUMINAMATH_GPT_find_distance_walker_l1346_134673


namespace NUMINAMATH_GPT_simplify_fraction_l1346_134672

noncomputable def sin_15 := Real.sin (15 * Real.pi / 180)
noncomputable def cos_15 := Real.cos (15 * Real.pi / 180)
noncomputable def angle_15 := 15 * Real.pi / 180

theorem simplify_fraction : (1 / sin_15 - 1 / cos_15 = 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1346_134672


namespace NUMINAMATH_GPT_find_smallest_A_divisible_by_51_l1346_134679

theorem find_smallest_A_divisible_by_51 :
  ∃ (x y : ℕ), (A = 1100 * x + 11 * y) ∧ 
    (0 ≤ x) ∧ (x ≤ 9) ∧ 
    (0 ≤ y) ∧ (y ≤ 9) ∧ 
    (A % 51 = 0) ∧ 
    (A = 1122) :=
sorry

end NUMINAMATH_GPT_find_smallest_A_divisible_by_51_l1346_134679


namespace NUMINAMATH_GPT_common_points_count_l1346_134619

variable (x y : ℝ)

def curve1 : Prop := x^2 + 4 * y^2 = 4
def curve2 : Prop := 4 * x^2 + y^2 = 4
def curve3 : Prop := x^2 + y^2 = 1

theorem common_points_count : ∀ (x y : ℝ), curve1 x y ∧ curve2 x y ∧ curve3 x y → false := by
  intros
  sorry

end NUMINAMATH_GPT_common_points_count_l1346_134619


namespace NUMINAMATH_GPT_range_of_p_l1346_134608

theorem range_of_p (a b : ℝ) :
  (∀ x y p q : ℝ, p + q = 1 → (p * (x^2 + a * x + b) + q * (y^2 + a * y + b) ≥ ((p * x + q * y)^2 + a * (p * x + q * y) + b))) →
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 1) :=
sorry

end NUMINAMATH_GPT_range_of_p_l1346_134608


namespace NUMINAMATH_GPT_fathers_age_more_than_4_times_son_l1346_134653

-- Let F (Father's age) be 44 and S (Son's age) be 10 as given by solving the equations
def X_years_more_than_4_times_son_age (F S X : ℕ) : Prop :=
  F = 4 * S + X ∧ F + 4 = 2 * (S + 4) + 20

theorem fathers_age_more_than_4_times_son (F S X : ℕ) (h1 : F = 44) (h2 : F = 4 * S + X) (h3 : F + 4 = 2 * (S + 4) + 20) :
  X = 4 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_fathers_age_more_than_4_times_son_l1346_134653


namespace NUMINAMATH_GPT_number_of_women_per_table_l1346_134691

theorem number_of_women_per_table
  (tables : ℕ) (men_per_table : ℕ) 
  (total_customers : ℕ) (total_tables : tables = 9) 
  (men_at_each_table : men_per_table = 3) 
  (customers : total_customers = 90) 
  (total_men : 3 * 9 = 27) 
  (total_women : 90 - 27 = 63) :
  (63 / 9 = 7) :=
by
  sorry

end NUMINAMATH_GPT_number_of_women_per_table_l1346_134691


namespace NUMINAMATH_GPT_prove_values_of_a_l1346_134634

-- Definitions of the conditions
def condition_1 (a x y : ℝ) : Prop := (x * y)^(1/3) = a^(a^2)
def condition_2 (a x y : ℝ) : Prop := (Real.log x / Real.log a * Real.log y / Real.log a) + (Real.log y / Real.log a * Real.log x / Real.log a) = 3 * a^3

-- The proof problem
theorem prove_values_of_a (a x y : ℝ) (h1 : condition_1 a x y) (h2 : condition_2 a x y) : a > 0 ∧ a ≤ 2/3 :=
sorry

end NUMINAMATH_GPT_prove_values_of_a_l1346_134634


namespace NUMINAMATH_GPT_eliminate_duplicates_3n_2m1_l1346_134692

theorem eliminate_duplicates_3n_2m1 :
  ∀ k: ℤ, ∃ n m: ℤ, 3 * n ≠ 2 * m + 1 ↔ 2 * m + 1 = 12 * k + 1 ∨ 2 * m + 1 = 12 * k + 5 :=
by
  sorry

end NUMINAMATH_GPT_eliminate_duplicates_3n_2m1_l1346_134692


namespace NUMINAMATH_GPT_smaller_triangle_area_14_365_l1346_134688

noncomputable def smaller_triangle_area (A : ℝ) (H_reduction : ℝ) : ℝ :=
  A * (H_reduction)^2

theorem smaller_triangle_area_14_365 :
  smaller_triangle_area 34 0.65 = 14.365 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_smaller_triangle_area_14_365_l1346_134688


namespace NUMINAMATH_GPT_price_per_litre_mixed_oil_l1346_134683

-- Define the given conditions
def cost_oil1 : ℝ := 100 * 45
def cost_oil2 : ℝ := 30 * 57.50
def cost_oil3 : ℝ := 20 * 72
def total_cost : ℝ := cost_oil1 + cost_oil2 + cost_oil3
def total_volume : ℝ := 100 + 30 + 20

-- Define the statement to be proved
theorem price_per_litre_mixed_oil : (total_cost / total_volume) = 51.10 :=
by
  sorry

end NUMINAMATH_GPT_price_per_litre_mixed_oil_l1346_134683


namespace NUMINAMATH_GPT_ordered_pair_of_positive_integers_l1346_134668

theorem ordered_pair_of_positive_integers :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (x^y + 4 = y^x) ∧ (3 * x^y = y^x + 10) ∧ (x = 7 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_ordered_pair_of_positive_integers_l1346_134668


namespace NUMINAMATH_GPT_naomi_saw_wheels_l1346_134604

theorem naomi_saw_wheels :
  let regular_bikes := 7
  let children's_bikes := 11
  let wheels_per_regular_bike := 2
  let wheels_per_children_bike := 4
  let total_wheels := regular_bikes * wheels_per_regular_bike + children's_bikes * wheels_per_children_bike
  total_wheels = 58 := by
  sorry

end NUMINAMATH_GPT_naomi_saw_wheels_l1346_134604


namespace NUMINAMATH_GPT_booth_visibility_correct_l1346_134613

noncomputable def booth_visibility (L : ℝ) : ℝ × ℝ :=
  let ρ_min := L
  let ρ_max := (1 + Real.sqrt 2) / 2 * L
  (ρ_min, ρ_max)

theorem booth_visibility_correct (L : ℝ) (hL : L > 0) :
  booth_visibility L = (L, (1 + Real.sqrt 2) / 2 * L) :=
by
  sorry

end NUMINAMATH_GPT_booth_visibility_correct_l1346_134613


namespace NUMINAMATH_GPT_factor_expression_l1346_134637

theorem factor_expression (x y z : ℝ) :
  (x - y)^3 + (y - z)^3 + (z - x)^3 ≠ 0 →
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) =
    (x + y) * (y + z) * (z + x) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_factor_expression_l1346_134637


namespace NUMINAMATH_GPT_yacht_capacity_l1346_134657

theorem yacht_capacity :
  ∀ (x y : ℕ), (3 * x + 2 * y = 68) → (2 * x + 3 * y = 57) → (3 * x + 6 * y = 96) :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_yacht_capacity_l1346_134657


namespace NUMINAMATH_GPT_cosine_neg_alpha_l1346_134623

theorem cosine_neg_alpha (alpha : ℝ) (h : Real.sin (π/2 + alpha) = -3/5) : Real.cos (-alpha) = -3/5 :=
sorry

end NUMINAMATH_GPT_cosine_neg_alpha_l1346_134623


namespace NUMINAMATH_GPT_a_3_value_l1346_134690

def arithmetic_seq (a: ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n - 3

theorem a_3_value :
  ∃ a : ℕ → ℤ, a 1 = 19 ∧ arithmetic_seq a ∧ a 3 = 13 :=
by
  sorry

end NUMINAMATH_GPT_a_3_value_l1346_134690


namespace NUMINAMATH_GPT_min_value_sum_l1346_134664

def positive_real (x : ℝ) : Prop := x > 0

theorem min_value_sum (x y : ℝ) (hx : positive_real x) (hy : positive_real y)
  (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 6) : x + y ≥ 20 :=
sorry

end NUMINAMATH_GPT_min_value_sum_l1346_134664


namespace NUMINAMATH_GPT_length_of_bridge_is_correct_l1346_134663

noncomputable def length_of_bridge (length_of_train : ℕ) (time_in_seconds : ℕ) (speed_in_kmph : ℝ) : ℝ :=
  let speed_in_mps := speed_in_kmph * (1000 / 3600)
  time_in_seconds * speed_in_mps - length_of_train

theorem length_of_bridge_is_correct :
  length_of_bridge 150 40 42.3 = 320 := by
  sorry

end NUMINAMATH_GPT_length_of_bridge_is_correct_l1346_134663


namespace NUMINAMATH_GPT_andy_wrong_questions_l1346_134680

theorem andy_wrong_questions
  (a b c d : ℕ)
  (h1 : a + b = c + d + 6)
  (h2 : a + d = b + c + 4)
  (h3 : c = 10) :
  a = 15 :=
by
  sorry

end NUMINAMATH_GPT_andy_wrong_questions_l1346_134680


namespace NUMINAMATH_GPT_sandy_worked_days_l1346_134615

-- Definitions based on the conditions
def total_hours_worked : ℕ := 45
def hours_per_day : ℕ := 9

-- The theorem that we need to prove
theorem sandy_worked_days : total_hours_worked / hours_per_day = 5 :=
by sorry

end NUMINAMATH_GPT_sandy_worked_days_l1346_134615


namespace NUMINAMATH_GPT_ceil_e_add_pi_l1346_134620

theorem ceil_e_add_pi : ⌈Real.exp 1 + Real.pi⌉ = 6 := by
  sorry

end NUMINAMATH_GPT_ceil_e_add_pi_l1346_134620


namespace NUMINAMATH_GPT_hexagon_perimeter_l1346_134650

def side_length : ℝ := 4
def number_of_sides : ℕ := 6

theorem hexagon_perimeter :
  6 * side_length = 24 := by
    sorry

end NUMINAMATH_GPT_hexagon_perimeter_l1346_134650


namespace NUMINAMATH_GPT_greatest_number_of_pieces_leftover_l1346_134624

theorem greatest_number_of_pieces_leftover (y : ℕ) (q r : ℕ) 
  (h : y = 6 * q + r) (hrange : r < 6) : r = 5 := sorry

end NUMINAMATH_GPT_greatest_number_of_pieces_leftover_l1346_134624


namespace NUMINAMATH_GPT_fraction_sum_is_one_l1346_134656

theorem fraction_sum_is_one
    (a b c d w x y z : ℝ)
    (h1 : 17 * w + b * x + c * y + d * z = 0)
    (h2 : a * w + 29 * x + c * y + d * z = 0)
    (h3 : a * w + b * x + 37 * y + d * z = 0)
    (h4 : a * w + b * x + c * y + 53 * z = 0)
    (a_ne_17 : a ≠ 17)
    (b_ne_29 : b ≠ 29)
    (c_ne_37 : c ≠ 37)
    (wxyz_nonzero : w ≠ 0 ∨ x ≠ 0 ∨ y ≠ 0) :
    (a / (a - 17)) + (b / (b - 29)) + (c / (c - 37)) + (d / (d - 53)) = 1 := 
sorry

end NUMINAMATH_GPT_fraction_sum_is_one_l1346_134656


namespace NUMINAMATH_GPT_melanie_more_turnips_l1346_134693

theorem melanie_more_turnips (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) :
  melanie_turnips - benny_turnips = 26 := by
  sorry

end NUMINAMATH_GPT_melanie_more_turnips_l1346_134693


namespace NUMINAMATH_GPT_find_unknown_number_l1346_134617

theorem find_unknown_number (x : ℤ) (h : (20 + 40 + 60) / 3 = 9 + (10 + 70 + x) / 3) : x = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_unknown_number_l1346_134617


namespace NUMINAMATH_GPT_at_least_one_expression_is_leq_neg_two_l1346_134687

variable (a b c : ℝ)

theorem at_least_one_expression_is_leq_neg_two 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 1 / b ≤ -2) ∨ (b + 1 / c ≤ -2) ∨ (c + 1 / a ≤ -2) :=
sorry

end NUMINAMATH_GPT_at_least_one_expression_is_leq_neg_two_l1346_134687


namespace NUMINAMATH_GPT_find_original_number_l1346_134636

theorem find_original_number (x : ℝ) 
  (h1 : x * 16 = 3408) 
  (h2 : 1.6 * 21.3 = 34.080000000000005) : 
  x = 213 :=
sorry

end NUMINAMATH_GPT_find_original_number_l1346_134636


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l1346_134647

theorem cyclic_sum_inequality (x y z a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ 3 / (a + b) :=
  sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l1346_134647


namespace NUMINAMATH_GPT_total_pokemon_cards_l1346_134632

-- Definitions based on the problem statement

def dozen_to_cards (dozen : ℝ) : ℝ :=
  dozen * 12

def melanie_cards : ℝ :=
  dozen_to_cards 7.5

def benny_cards : ℝ :=
  dozen_to_cards 9

def sandy_cards : ℝ :=
  dozen_to_cards 5.2

def jessica_cards : ℝ :=
  dozen_to_cards 12.8

def total_cards : ℝ :=
  melanie_cards + benny_cards + sandy_cards + jessica_cards

theorem total_pokemon_cards : total_cards = 414 := 
  by sorry

end NUMINAMATH_GPT_total_pokemon_cards_l1346_134632


namespace NUMINAMATH_GPT_directrix_of_parabola_l1346_134601

-- Define the condition given in the problem
def parabola_eq (x y : ℝ) : Prop := x^2 = 2 * y

-- Define the directrix equation property we want to prove
theorem directrix_of_parabola (x : ℝ) :
  (∃ y : ℝ, parabola_eq x y) → (∃ y : ℝ, y = -1 / 2) :=
by sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1346_134601


namespace NUMINAMATH_GPT_determine_digit_l1346_134611

theorem determine_digit (Θ : ℚ) (h : 312 / Θ = 40 + 2 * Θ) : Θ = 6 :=
sorry

end NUMINAMATH_GPT_determine_digit_l1346_134611


namespace NUMINAMATH_GPT_any_nat_as_difference_or_element_l1346_134649

noncomputable def seq (q : ℕ → ℕ) : Prop :=
∀ n, q n < 2 * n

theorem any_nat_as_difference_or_element (q : ℕ → ℕ) (h_seq : seq q) (m : ℕ) :
  (∃ k, q k = m) ∨ (∃ k l, q l - q k = m) :=
sorry

end NUMINAMATH_GPT_any_nat_as_difference_or_element_l1346_134649


namespace NUMINAMATH_GPT_betty_cookies_brownies_l1346_134696

theorem betty_cookies_brownies (cookies_per_day brownies_per_day initial_cookies initial_brownies days : ℕ) :
  cookies_per_day = 3 → brownies_per_day = 1 → initial_cookies = 60 → initial_brownies = 10 → days = 7 →
  initial_cookies - days * cookies_per_day - (initial_brownies - days * brownies_per_day) = 36 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_betty_cookies_brownies_l1346_134696


namespace NUMINAMATH_GPT_number_of_correct_statements_l1346_134694

-- Define the statements
def statement_1 : Prop := ∀ (a : ℚ), |a| < |0| → a = 0
def statement_2 : Prop := ∃ (b : ℚ), ∀ (c : ℚ), b < 0 ∧ b ≥ c → c = b
def statement_3 : Prop := -4^6 = (-4) * (-4) * (-4) * (-4) * (-4) * (-4)
def statement_4 : Prop := ∀ (a b : ℚ), a + b = 0 → a ≠ 0 → b ≠ 0 → (a / b = -1)
def statement_5 : Prop := ∀ (c : ℚ), (0 / c = 0 ↔ c ≠ 0)

-- Define the overall proof problem
theorem number_of_correct_statements : (statement_1 ∧ statement_4) ∧ ¬(statement_2 ∨ statement_3 ∨ statement_5) :=
by
  sorry

end NUMINAMATH_GPT_number_of_correct_statements_l1346_134694


namespace NUMINAMATH_GPT_geometric_sequence_100th_term_l1346_134665

theorem geometric_sequence_100th_term :
  ∀ (a₁ a₂ : ℤ) (r : ℤ), a₁ = 5 → a₂ = -15 → r = a₂ / a₁ → 
  (a₁ * r ^ 99 = -5 * 3 ^ 99) :=
by
  intros a₁ a₂ r ha₁ ha₂ hr
  sorry

end NUMINAMATH_GPT_geometric_sequence_100th_term_l1346_134665


namespace NUMINAMATH_GPT_evaluate_expression_l1346_134643

/- The mathematical statement to prove:

Evaluate the expression 2/10 + 4/20 + 6/30, then multiply the result by 3
and show that it equals to 9/5.
-/

theorem evaluate_expression : 
  (2 / 10 + 4 / 20 + 6 / 30) * 3 = 9 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1346_134643


namespace NUMINAMATH_GPT_x_is_48_percent_of_z_l1346_134676

variable {x y z : ℝ}

theorem x_is_48_percent_of_z (h1 : x = 1.20 * y) (h2 : y = 0.40 * z) : x = 0.48 * z :=
by
  sorry

end NUMINAMATH_GPT_x_is_48_percent_of_z_l1346_134676


namespace NUMINAMATH_GPT_tournament_game_count_l1346_134645

/-- In a tournament with 25 players where each player plays 4 games against each other,
prove that the total number of games played is 1200. -/
theorem tournament_game_count : 
  let n := 25
  let games_per_pair := 4
  let total_games := (n * (n - 1) / 2) * games_per_pair
  total_games = 1200 :=
by
  -- Definitions based on the conditions
  let n := 25
  let games_per_pair := 4

  -- Calculating the total number of games
  let total_games := (n * (n - 1) / 2) * games_per_pair

  -- This is the main goal to prove
  have h : total_games = 1200 := sorry
  exact h

end NUMINAMATH_GPT_tournament_game_count_l1346_134645


namespace NUMINAMATH_GPT_math_proof_problem_l1346_134689
noncomputable def expr : ℤ := 3000 * (3000 ^ 3000) + 3000 ^ 2

theorem math_proof_problem : expr = 3000 ^ 3001 + 9000000 :=
by
  -- Proof
  sorry

end NUMINAMATH_GPT_math_proof_problem_l1346_134689


namespace NUMINAMATH_GPT_pow_mult_rule_l1346_134648

variable (x : ℝ)

theorem pow_mult_rule : (x^3) * (x^2) = x^5 :=
by sorry

end NUMINAMATH_GPT_pow_mult_rule_l1346_134648


namespace NUMINAMATH_GPT_revenue_fall_percentage_l1346_134678

theorem revenue_fall_percentage:
  let oldRevenue := 72.0
  let newRevenue := 48.0
  (oldRevenue - newRevenue) / oldRevenue * 100 = 33.33 :=
by
  let oldRevenue := 72.0
  let newRevenue := 48.0
  sorry

end NUMINAMATH_GPT_revenue_fall_percentage_l1346_134678


namespace NUMINAMATH_GPT_triangle_properties_l1346_134658

theorem triangle_properties :
  (∀ (α β γ : ℝ), α + β + γ = 180 → 
    (α = β ∨ α = γ ∨ β = γ ∨ 
     (α = 60 ∧ β = 60 ∧ γ = 60) ∨
     ¬(α = 90 ∧ β = 90))) :=
by
  -- Placeholder for the actual proof, ensuring the theorem can build
  intros α β γ h₁
  sorry

end NUMINAMATH_GPT_triangle_properties_l1346_134658


namespace NUMINAMATH_GPT_calc_3a2b_times_neg_a_squared_l1346_134629

variables {a b : ℝ}

theorem calc_3a2b_times_neg_a_squared : 3 * a^2 * b * (-a)^2 = 3 * a^4 * b :=
by
  sorry

end NUMINAMATH_GPT_calc_3a2b_times_neg_a_squared_l1346_134629


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l1346_134630

/--
Given a triangle \(ABC\) with points \(D\) and \(E\) on sides \(BC\) and \(AC\) respectively,
where \(BD = 4\), \(DE = 2\), \(EC = 6\), and \(BF = FC = 3\),
proves that the area of triangle \( \triangle ABC \) is \( 18\sqrt{3} \).
-/
theorem area_of_triangle_ABC :
  ∀ (ABC D E : Type) (BD DE EC BF FC : ℝ),
    BD = 4 → DE = 2 → EC = 6 → BF = 3 → FC = 3 → 
    ∃ area, area = 18 * Real.sqrt 3 :=
by
  intros ABC D E BD DE EC BF FC hBD hDE hEC hBF hFC
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l1346_134630


namespace NUMINAMATH_GPT_perimeter_trapezoid_l1346_134602

theorem perimeter_trapezoid 
(E F G H : Point)
(EF GH : ℝ)
(HJ EI FG EH : ℝ)
(h_eq1 : EF = GH)
(h_FG : FG = 10)
(h_EH : EH = 20)
(h_EI : EI = 5)
(h_HJ : HJ = 5)
(h_EF_HG : EF = Real.sqrt (EI^2 + ((EH - FG) / 2)^2)) :
  2 * EF + FG + EH = 30 + 10 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_trapezoid_l1346_134602


namespace NUMINAMATH_GPT_remaining_liquid_weight_l1346_134652

theorem remaining_liquid_weight 
  (liqX_content : ℝ := 0.20)
  (water_content : ℝ := 0.80)
  (initial_solution : ℝ := 8)
  (evaporated_water : ℝ := 2)
  (added_solution : ℝ := 2)
  (new_solution_fraction : ℝ := 0.25) :
  ∃ (remaining_liquid : ℝ), remaining_liquid = 6 := 
by
  -- Skip the proof to ensure the statement is built successfully
  sorry

end NUMINAMATH_GPT_remaining_liquid_weight_l1346_134652


namespace NUMINAMATH_GPT_find_rectangle_length_l1346_134610

-- Define the problem conditions
def length_is_three_times_breadth (l b : ℕ) : Prop := l = 3 * b
def area_of_rectangle (l b : ℕ) : Prop := l * b = 6075

-- Define the theorem to prove the length of the rectangle given the conditions
theorem find_rectangle_length (l b : ℕ) (h1 : length_is_three_times_breadth l b) (h2 : area_of_rectangle l b) : l = 135 := 
sorry

end NUMINAMATH_GPT_find_rectangle_length_l1346_134610


namespace NUMINAMATH_GPT_three_not_divide_thirtyone_l1346_134667

theorem three_not_divide_thirtyone : ¬ ∃ q : ℤ, 31 = 3 * q := sorry

end NUMINAMATH_GPT_three_not_divide_thirtyone_l1346_134667


namespace NUMINAMATH_GPT_unique_x1_exists_l1346_134674

theorem unique_x1_exists (x : ℕ → ℝ) :
  (∀ n : ℕ+, x (n+1) = x n * (x n + 1 / n)) →
  ∃! (x1 : ℝ), (∀ n : ℕ+, 0 < x n ∧ x n < x (n+1) ∧ x (n+1) < 1) :=
sorry

end NUMINAMATH_GPT_unique_x1_exists_l1346_134674


namespace NUMINAMATH_GPT_additional_time_needed_l1346_134639

theorem additional_time_needed (total_parts apprentice_first_phase remaining_parts apprentice_rate master_rate combined_rate : ℕ)
  (h1 : total_parts = 500)
  (h2 : apprentice_first_phase = 45)
  (h3 : remaining_parts = total_parts - apprentice_first_phase)
  (h4 : apprentice_rate = 15)
  (h5 : master_rate = 20)
  (h6 : combined_rate = apprentice_rate + master_rate) :
  remaining_parts / combined_rate = 13 := 
by {
  sorry
}

end NUMINAMATH_GPT_additional_time_needed_l1346_134639


namespace NUMINAMATH_GPT_annies_initial_amount_l1346_134628

theorem annies_initial_amount :
  let hamburger_cost := 4
  let cheeseburger_cost := 5
  let french_fries_cost := 3
  let milkshake_cost := 5
  let smoothie_cost := 6
  let people_count := 8
  let burger_discount := 1
  let milkshake_discount := 2
  let smoothie_discount_buy2_get1free := 6
  let sales_tax := 0.08
  let tip_rate := 0.15
  let max_single_person_cost := cheeseburger_cost + french_fries_cost + smoothie_cost
  let total_cost := people_count * max_single_person_cost
  let total_burger_discount := people_count * burger_discount
  let total_milkshake_discount := 4 * milkshake_discount
  let total_smoothie_discount := smoothie_discount_buy2_get1free
  let total_discount := total_burger_discount + total_milkshake_discount + total_smoothie_discount
  let discounted_cost := total_cost - total_discount
  let tax_amount := discounted_cost * sales_tax
  let subtotal_with_tax := discounted_cost + tax_amount
  let original_total_cost := people_count * max_single_person_cost
  let tip_amount := original_total_cost * tip_rate
  let final_amount := subtotal_with_tax + tip_amount
  let annie_has_left := 30
  let annies_initial_money := final_amount + annie_has_left
  annies_initial_money = 144 :=
by
  sorry

end NUMINAMATH_GPT_annies_initial_amount_l1346_134628


namespace NUMINAMATH_GPT_volume_box_l1346_134606

theorem volume_box (x y : ℝ) :
  (16 - 2 * x) * (12 - 2 * y) * y = 4 * x * y ^ 2 - 24 * x * y + 192 * y - 32 * y ^ 2 :=
by sorry

end NUMINAMATH_GPT_volume_box_l1346_134606


namespace NUMINAMATH_GPT_rolls_remaining_to_sell_l1346_134684

-- Conditions
def total_rolls_needed : ℕ := 45
def rolls_sold_to_grandmother : ℕ := 1
def rolls_sold_to_uncle : ℕ := 10
def rolls_sold_to_neighbor : ℕ := 6

-- Theorem statement
theorem rolls_remaining_to_sell : (total_rolls_needed - (rolls_sold_to_grandmother + rolls_sold_to_uncle + rolls_sold_to_neighbor) = 28) :=
by
  sorry

end NUMINAMATH_GPT_rolls_remaining_to_sell_l1346_134684


namespace NUMINAMATH_GPT_locus_of_M_l1346_134654

/-- Define the coordinates of points A and B, and given point M(x, y) with the 
    condition x ≠ ±1, ensure the equation of the locus of point M -/
theorem locus_of_M (x y : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) 
  (h3 : (y / (x + 1)) + (y / (x - 1)) = 2) : x^2 - x * y - 1 = 0 := 
sorry

end NUMINAMATH_GPT_locus_of_M_l1346_134654


namespace NUMINAMATH_GPT_calculate_minutes_worked_today_l1346_134671

-- Define the conditions
def production_rate := 6 -- shirts per minute
def total_shirts_today := 72 

-- The statement to prove
theorem calculate_minutes_worked_today :
  total_shirts_today / production_rate = 12 := 
by
  sorry

end NUMINAMATH_GPT_calculate_minutes_worked_today_l1346_134671


namespace NUMINAMATH_GPT_workers_in_first_group_l1346_134698

theorem workers_in_first_group
  (W D : ℕ)
  (h1 : 6 * W * D = 9450)
  (h2 : 95 * D = 9975) :
  W = 15 := 
sorry

end NUMINAMATH_GPT_workers_in_first_group_l1346_134698


namespace NUMINAMATH_GPT_factorial_fraction_simplification_l1346_134670

theorem factorial_fraction_simplification : 
  (4 * (Nat.factorial 6) + 24 * (Nat.factorial 5)) / (Nat.factorial 7) = 8 / 7 :=
by
  sorry

end NUMINAMATH_GPT_factorial_fraction_simplification_l1346_134670


namespace NUMINAMATH_GPT_Flynn_tv_minutes_weekday_l1346_134625

theorem Flynn_tv_minutes_weekday :
  ∀ (tv_hours_per_weekend : ℕ)
    (tv_hours_per_year : ℕ)
    (weeks_per_year : ℕ) 
    (weekdays_per_week : ℕ),
  tv_hours_per_weekend = 2 →
  tv_hours_per_year = 234 →
  weeks_per_year = 52 →
  weekdays_per_week = 5 →
  (tv_hours_per_year - (tv_hours_per_weekend * weeks_per_year)) / (weekdays_per_week * weeks_per_year) * 60
  = 30 :=
by
  intros tv_hours_per_weekend tv_hours_per_year weeks_per_year weekdays_per_week
        h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_Flynn_tv_minutes_weekday_l1346_134625


namespace NUMINAMATH_GPT_ordered_pairs_satisfy_equation_l1346_134697

theorem ordered_pairs_satisfy_equation :
  (∃ (a : ℝ) (b : ℤ), a > 0 ∧ 3 ≤ b ∧ b ≤ 203 ∧ (Real.log a / Real.log b) ^ 2021 = Real.log (a ^ 2021) / Real.log b) :=
sorry

end NUMINAMATH_GPT_ordered_pairs_satisfy_equation_l1346_134697


namespace NUMINAMATH_GPT_ab_cd_eq_neg190_over_9_l1346_134682

theorem ab_cd_eq_neg190_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 3)
  (h2 : a + b + d = -2)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = -1) :
  a * b + c * d = -190 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ab_cd_eq_neg190_over_9_l1346_134682


namespace NUMINAMATH_GPT_problem_statement_l1346_134607

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 3, 5}
def star (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem problem_statement : star A B = {1, 7} := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1346_134607


namespace NUMINAMATH_GPT_sum_of_money_l1346_134640

theorem sum_of_money (J C P : ℕ) 
  (h1 : P = 60)
  (h2 : P = 3 * J)
  (h3 : C + 7 = 2 * J) : 
  J + P + C = 113 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_money_l1346_134640


namespace NUMINAMATH_GPT_math_problem_l1346_134662

theorem math_problem (a b c : ℚ) 
  (h1 : a * (-2) = 1)
  (h2 : |b + 2| = 5)
  (h3 : c = 5 - 6) :
  4 * a - b + 3 * c = -8 ∨ 4 * a - b + 3 * c = 2 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1346_134662


namespace NUMINAMATH_GPT_speed_faster_train_correct_l1346_134616

noncomputable def speed_faster_train_proof
  (time_seconds : ℝ) 
  (speed_slower_train : ℝ)
  (train_length_meters : ℝ) :
  Prop :=
  let time_hours := time_seconds / 3600
  let train_length_km := train_length_meters / 1000
  let total_distance_km := train_length_km + train_length_km
  let relative_speed_km_hr := total_distance_km / time_hours
  let speed_faster_train := relative_speed_km_hr + speed_slower_train
  speed_faster_train = 46

theorem speed_faster_train_correct :
  speed_faster_train_proof 36.00001 36 50.000013888888894 :=
by 
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_speed_faster_train_correct_l1346_134616


namespace NUMINAMATH_GPT_percentage_x_of_yz_l1346_134686

theorem percentage_x_of_yz (x y z w : ℝ) (h1 : x = 0.07 * y) (h2 : y = 0.35 * z) (h3 : z = 0.60 * w) :
  (x / (y + z) * 100) = 1.8148 :=
by
  sorry

end NUMINAMATH_GPT_percentage_x_of_yz_l1346_134686


namespace NUMINAMATH_GPT_total_handshakes_l1346_134699

def gremlins := 30
def pixies := 12
def unfriendly_gremlins := 15
def friendly_gremlins := 15

def handshake_count : Nat :=
  let handshakes_friendly_gremlins := friendly_gremlins * (friendly_gremlins - 1) / 2
  let handshakes_friendly_unfriendly := friendly_gremlins * unfriendly_gremlins
  let handshakes_gremlins_pixies := gremlins * pixies
  handshakes_friendly_gremlins + handshakes_friendly_unfriendly + handshakes_gremlins_pixies

theorem total_handshakes : handshake_count = 690 := by
  sorry

end NUMINAMATH_GPT_total_handshakes_l1346_134699


namespace NUMINAMATH_GPT_integer_solutions_conditions_even_l1346_134646

theorem integer_solutions_conditions_even (n : ℕ) (x : ℕ → ℤ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 
    x i ^ 2 + x ((i % n) + 1) ^ 2 + 50 = 16 * x i + 12 * x ((i % n) + 1) ) → 
  n % 2 = 0 :=
by 
sorry

end NUMINAMATH_GPT_integer_solutions_conditions_even_l1346_134646


namespace NUMINAMATH_GPT_square_area_from_diagonal_l1346_134642

theorem square_area_from_diagonal (d : ℝ) (hd : d = 3.8) : 
  ∃ (A : ℝ), A = 7.22 ∧ (∀ s : ℝ, d^2 = 2 * (s^2) → A = s^2) :=
by
  sorry

end NUMINAMATH_GPT_square_area_from_diagonal_l1346_134642


namespace NUMINAMATH_GPT_positive_three_digit_integers_divisible_by_12_and_7_l1346_134681

theorem positive_three_digit_integers_divisible_by_12_and_7 : 
  ∃ n : ℕ, n = 11 ∧ ∀ k : ℕ, (k ∣ 12) ∧ (k ∣ 7) ∧ (100 ≤ k) ∧ (k < 1000) :=
by
  sorry

end NUMINAMATH_GPT_positive_three_digit_integers_divisible_by_12_and_7_l1346_134681


namespace NUMINAMATH_GPT_problem_l1346_134621

-- Definitions for conditions
def countMultiplesOf (n upperLimit : ℕ) : ℕ :=
  (upperLimit - 1) / n

def a : ℕ := countMultiplesOf 4 40
def b : ℕ := countMultiplesOf 4 40

-- Statement to prove
theorem problem : (a + b)^2 = 324 := by
  sorry

end NUMINAMATH_GPT_problem_l1346_134621


namespace NUMINAMATH_GPT_rice_mixing_ratio_l1346_134612

-- Definitions based on conditions
def rice_1_price : ℝ := 6
def rice_2_price : ℝ := 8.75
def mixture_price : ℝ := 7.50

-- Proof of the required ratio
theorem rice_mixing_ratio (x y : ℝ) (h : (rice_1_price * x + rice_2_price * y) / (x + y) = mixture_price) :
  y / x = 6 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_rice_mixing_ratio_l1346_134612


namespace NUMINAMATH_GPT_arithmetic_problem_l1346_134626

theorem arithmetic_problem : 72 * 1313 - 32 * 1313 = 52520 := by
  sorry

end NUMINAMATH_GPT_arithmetic_problem_l1346_134626


namespace NUMINAMATH_GPT_find_value_of_expression_l1346_134605

variable (α : ℝ)

theorem find_value_of_expression 
  (h : Real.tan α = 3) : 
  (Real.sin (2 * α) / (Real.cos α)^2) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l1346_134605
