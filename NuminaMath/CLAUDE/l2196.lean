import Mathlib

namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_difference_l2196_219660

theorem sin_cos_fourth_power_difference (α : ℝ) :
  Real.sin (π / 2 - 2 * α) = 3 / 5 →
  Real.sin α ^ 4 - Real.cos α ^ 4 = -(3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_difference_l2196_219660


namespace NUMINAMATH_CALUDE_original_rectangle_area_l2196_219681

theorem original_rectangle_area (original_area new_area : ℝ) : 
  (∀ (length width : ℝ), 
    length > 0 → width > 0 → 
    original_area = length * width → 
    new_area = (2 * length) * (2 * width)) →
  new_area = 32 →
  original_area = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_original_rectangle_area_l2196_219681


namespace NUMINAMATH_CALUDE_sum_of_five_terms_l2196_219624

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_five_terms (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 3 + a 15 = 6 →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_five_terms_l2196_219624


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2196_219621

/-- Represents the number of exam papers checked in a school -/
structure SchoolSample where
  total : ℕ
  sampled : ℕ

/-- Calculates the total number of exam papers checked across all schools -/
def totalSampled (schools : List SchoolSample) : ℕ :=
  schools.map (fun s => s.sampled) |>.sum

theorem stratified_sampling_theorem (schoolA schoolB schoolC : SchoolSample) :
  schoolA.total = 1260 →
  schoolB.total = 720 →
  schoolC.total = 900 →
  schoolC.sampled = 45 →
  schoolA.sampled = schoolA.total / (schoolC.total / schoolC.sampled) →
  schoolB.sampled = schoolB.total / (schoolC.total / schoolC.sampled) →
  totalSampled [schoolA, schoolB, schoolC] = 144 := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2196_219621


namespace NUMINAMATH_CALUDE_total_salaries_is_4000_l2196_219682

/-- The total amount of A and B's salaries is $4000 -/
theorem total_salaries_is_4000 
  (a_salary : ℝ) 
  (b_salary : ℝ) 
  (h1 : a_salary = 3000)
  (h2 : 0.05 * a_salary = 0.15 * b_salary) : 
  a_salary + b_salary = 4000 := by
  sorry

#check total_salaries_is_4000

end NUMINAMATH_CALUDE_total_salaries_is_4000_l2196_219682


namespace NUMINAMATH_CALUDE_total_amount_proof_l2196_219685

/-- Given that r has two-thirds of the total amount with p and q, and r has 1600,
    prove that the total amount T with p, q, and r is 4000. -/
theorem total_amount_proof (T : ℝ) (r : ℝ) 
  (h1 : r = 2/3 * T)
  (h2 : r = 1600) : 
  T = 4000 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_proof_l2196_219685


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_390_l2196_219627

theorem sin_n_equals_cos_390 (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) :
  Real.sin (n * Real.pi / 180) = Real.cos (390 * Real.pi / 180) → n = 60 := by
sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_390_l2196_219627


namespace NUMINAMATH_CALUDE_prob_at_least_one_girl_pair_approx_l2196_219625

/-- The number of boys in the group -/
def num_boys : ℕ := 8

/-- The number of girls in the group -/
def num_girls : ℕ := 8

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The probability of at least one pair consisting of two girls -/
noncomputable def prob_at_least_one_girl_pair : ℝ :=
  1 - (num_boys.factorial * num_girls.factorial * (2^num_pairs) * num_pairs.factorial) / total_people.factorial

/-- Theorem stating that the probability of at least one pair consisting of two girls is approximately 0.98 -/
theorem prob_at_least_one_girl_pair_approx :
  abs (prob_at_least_one_girl_pair - 0.98) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_girl_pair_approx_l2196_219625


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l2196_219671

/-- The total bill for 9 friends dining at a restaurant -/
def total_bill : ℕ := 156

/-- The number of friends dining -/
def num_friends : ℕ := 9

/-- The amount Judi paid -/
def judi_payment : ℕ := 5

/-- The extra amount each remaining friend paid -/
def extra_payment : ℕ := 3

theorem restaurant_bill_proof :
  let regular_share := total_bill / num_friends
  let tom_payment := regular_share / 2
  let remaining_friends := num_friends - 2
  total_bill = 
    (remaining_friends * (regular_share + extra_payment)) + 
    judi_payment + 
    tom_payment :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l2196_219671


namespace NUMINAMATH_CALUDE_statement_c_is_false_l2196_219694

theorem statement_c_is_false : ¬(∀ (p q : Prop), ¬(p ∧ q) → (¬p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_statement_c_is_false_l2196_219694


namespace NUMINAMATH_CALUDE_charity_ticket_revenue_l2196_219678

theorem charity_ticket_revenue :
  ∀ (full_price : ℕ) (full_count half_count : ℕ),
  full_count + half_count = 200 →
  full_count = 3 * half_count →
  full_count * full_price + half_count * (full_price / 2) = 3501 →
  full_count * full_price = 3000 :=
by
  sorry

end NUMINAMATH_CALUDE_charity_ticket_revenue_l2196_219678


namespace NUMINAMATH_CALUDE_four_b_b_two_divisible_by_seven_l2196_219650

theorem four_b_b_two_divisible_by_seven (B : ℕ) : 
  B ≤ 9 → (4000 + 110 * B + 2) % 7 = 0 ↔ B = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_b_b_two_divisible_by_seven_l2196_219650


namespace NUMINAMATH_CALUDE_prime_triplets_l2196_219603

def is_prime_triplet (a b c : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧
  Nat.Prime (a - b - 8) ∧ Nat.Prime (b - c - 8)

theorem prime_triplets :
  ∀ a b c : ℕ, is_prime_triplet a b c ↔ (a = 23 ∧ b = 13 ∧ (c = 2 ∨ c = 3)) :=
sorry

end NUMINAMATH_CALUDE_prime_triplets_l2196_219603


namespace NUMINAMATH_CALUDE_tan_3x_eq_sin_x_solutions_l2196_219608

open Real

theorem tan_3x_eq_sin_x_solutions (x : ℝ) :
  ∃ (s : Finset ℝ), s.card = 12 ∧
  (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2*π ∧ tan (3*x) = sin x) ∧
  (∀ y, 0 ≤ y ∧ y ≤ 2*π ∧ tan (3*y) = sin y → y ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_tan_3x_eq_sin_x_solutions_l2196_219608


namespace NUMINAMATH_CALUDE_find_b_l2196_219691

/-- Given two functions p and q, prove that b = 7 when p(q(5)) = 11 -/
theorem find_b (p q : ℝ → ℝ) (b : ℝ) 
  (hp : ∀ x, p x = 2 * x - 5)
  (hq : ∀ x, q x = 3 * x - b)
  (h_pq : p (q 5) = 11) : 
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_find_b_l2196_219691


namespace NUMINAMATH_CALUDE_bromine_mass_percentage_not_37_21_l2196_219649

/-- The mass percentage of bromine in HBrO3 is not 37.21% -/
theorem bromine_mass_percentage_not_37_21 (H_mass Br_mass O_mass : ℝ) 
  (h1 : H_mass = 1.01)
  (h2 : Br_mass = 79.90)
  (h3 : O_mass = 16.00) :
  let HBrO3_mass := H_mass + Br_mass + 3 * O_mass
  (Br_mass / HBrO3_mass) * 100 ≠ 37.21 := by sorry

end NUMINAMATH_CALUDE_bromine_mass_percentage_not_37_21_l2196_219649


namespace NUMINAMATH_CALUDE_june_election_win_l2196_219667

theorem june_election_win (total_students : ℕ) (boy_percentage : ℚ) 
  (june_boy_vote_percentage : ℚ) (june_girl_vote_percentage : ℚ) :
  total_students = 200 →
  boy_percentage = 60 / 100 →
  june_boy_vote_percentage = 675 / 1000 →
  june_girl_vote_percentage = 1 / 4 →
  ∃ (june_total_vote_percentage : ℚ), 
    june_total_vote_percentage = 505 / 1000 ∧ 
    june_total_vote_percentage > 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_june_election_win_l2196_219667


namespace NUMINAMATH_CALUDE_choose_starters_with_twins_l2196_219609

def total_players : ℕ := 12
def twin_players : ℕ := 2
def starters : ℕ := 5

theorem choose_starters_with_twins :
  (total_players.choose starters) = (total_players - twin_players).choose (starters - twin_players) :=
sorry

end NUMINAMATH_CALUDE_choose_starters_with_twins_l2196_219609


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2196_219642

theorem polynomial_remainder (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 7*x^3 + 9*x^2 + 16*x - 13
  f 3 = 8 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2196_219642


namespace NUMINAMATH_CALUDE_quadratic_root_property_l2196_219602

theorem quadratic_root_property (m : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ + m = 0) → 
  (x₂^2 + 2*x₂ + m = 0) → 
  (x₁ + x₂ = x₁*x₂ - 1) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l2196_219602


namespace NUMINAMATH_CALUDE_remaining_pieces_count_l2196_219670

/-- Represents the number of pieces in a standard chess set -/
def standard_set : Nat := 32

/-- Represents the total number of missing pieces -/
def missing_pieces : Nat := 12

/-- Represents the number of missing kings -/
def missing_kings : Nat := 1

/-- Represents the number of missing queens -/
def missing_queens : Nat := 2

/-- Represents the number of missing knights -/
def missing_knights : Nat := 3

/-- Represents the number of missing pawns -/
def missing_pawns : Nat := 6

/-- Theorem stating that the number of remaining pieces is 20 -/
theorem remaining_pieces_count :
  standard_set - missing_pieces = 20 :=
by
  sorry

#check remaining_pieces_count

end NUMINAMATH_CALUDE_remaining_pieces_count_l2196_219670


namespace NUMINAMATH_CALUDE_install_remaining_windows_time_l2196_219615

/-- Calculates the time needed to install remaining windows -/
def time_to_install_remaining (total : ℕ) (installed : ℕ) (time_per_window : ℕ) : ℕ :=
  (total - installed) * time_per_window

/-- Proves that the time to install the remaining windows is 20 hours -/
theorem install_remaining_windows_time :
  time_to_install_remaining 10 6 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_install_remaining_windows_time_l2196_219615


namespace NUMINAMATH_CALUDE_inequality_proof_l2196_219646

theorem inequality_proof (a b x : ℝ) (h : 0 ≤ x ∧ x < Real.pi / 2) :
  a^2 * Real.tan x * (Real.cos x)^(1/3) + b^2 * Real.sin x ≥ 2 * x * a * b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2196_219646


namespace NUMINAMATH_CALUDE_train_length_l2196_219699

/-- The length of a train given crossing times and platform length -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : platform_length = 350)
  (h2 : platform_time = 39)
  (h3 : pole_time = 18) :
  (platform_length * pole_time) / (platform_time - pole_time) = 300 := by
sorry

end NUMINAMATH_CALUDE_train_length_l2196_219699


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2196_219652

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| < 2} = Set.Ioo (-1) 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2196_219652


namespace NUMINAMATH_CALUDE_bd_length_l2196_219607

-- Define the triangle ABC
structure Triangle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :=
  (A B C : α)

-- Define the properties of the triangle
def IsoscelesTriangle {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : Prop :=
  ‖t.A - t.C‖ = ‖t.B - t.C‖

-- Define point D on AB
def PointOnLine {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (A B D : α) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • A + t • B

-- Main theorem
theorem bd_length {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) (D : α) :
  IsoscelesTriangle t →
  PointOnLine t.A t.B D →
  ‖t.A - t.C‖ = 10 →
  ‖t.A - D‖ = 12 →
  ‖t.C - D‖ = 4 →
  ‖t.B - D‖ = 7 := by
  sorry


end NUMINAMATH_CALUDE_bd_length_l2196_219607


namespace NUMINAMATH_CALUDE_base_three_20121_equals_178_l2196_219663

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base_three_20121_equals_178 :
  base_three_to_decimal [2, 0, 1, 2, 1] = 178 := by
  sorry

end NUMINAMATH_CALUDE_base_three_20121_equals_178_l2196_219663


namespace NUMINAMATH_CALUDE_temperature_difference_qianan_l2196_219693

/-- The temperature difference between two times of day -/
def temperature_difference (temp1 : Int) (temp2 : Int) : Int :=
  temp2 - temp1

/-- Proof that the temperature difference between 10 a.m. and midnight is 9°C -/
theorem temperature_difference_qianan : 
  let midnight_temp : Int := -4
  let morning_temp : Int := 5
  temperature_difference midnight_temp morning_temp = 9 := by
sorry

end NUMINAMATH_CALUDE_temperature_difference_qianan_l2196_219693


namespace NUMINAMATH_CALUDE_wage_calculation_l2196_219639

/-- A worker's wage calculation problem -/
theorem wage_calculation 
  (total_days : ℕ) 
  (absent_days : ℕ) 
  (fine_per_day : ℕ) 
  (total_pay : ℕ) 
  (h1 : total_days = 30)
  (h2 : absent_days = 7)
  (h3 : fine_per_day = 2)
  (h4 : total_pay = 216) :
  ∃ (daily_wage : ℕ),
    (total_days - absent_days) * daily_wage - absent_days * fine_per_day = total_pay ∧
    daily_wage = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_wage_calculation_l2196_219639


namespace NUMINAMATH_CALUDE_solve_for_T_l2196_219695

theorem solve_for_T : ∃ T : ℚ, (3/4) * (1/6) * T = (1/5) * (1/4) * 120 ∧ T = 48 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_T_l2196_219695


namespace NUMINAMATH_CALUDE_power_three_plus_four_mod_five_l2196_219688

theorem power_three_plus_four_mod_five : 3^75 + 4 ≡ 1 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_power_three_plus_four_mod_five_l2196_219688


namespace NUMINAMATH_CALUDE_reflection_about_y_axis_example_l2196_219686

/-- Given a point in 3D space, return its reflection about the y-axis -/
def reflect_about_y_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, -z)

/-- The reflection of point (3, -2, 1) about the y-axis is (-3, -2, -1) -/
theorem reflection_about_y_axis_example : 
  reflect_about_y_axis (3, -2, 1) = (-3, -2, -1) := by
  sorry

#check reflection_about_y_axis_example

end NUMINAMATH_CALUDE_reflection_about_y_axis_example_l2196_219686


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2196_219653

theorem negation_of_universal_statement (S : Set ℝ) :
  (¬ ∀ x ∈ S, 3 * x - 5 > 0) ↔ (∃ x ∈ S, 3 * x - 5 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2196_219653


namespace NUMINAMATH_CALUDE_fund_price_calculation_l2196_219604

theorem fund_price_calculation (initial_price : ℝ) 
  (monday_change tuesday_change wednesday_change thursday_change friday_change : ℝ) :
  initial_price = 35 →
  monday_change = 4.5 →
  tuesday_change = 4 →
  wednesday_change = -1 →
  thursday_change = -2.5 →
  friday_change = -6 →
  initial_price + monday_change + tuesday_change + wednesday_change + thursday_change + friday_change = 34 := by
  sorry

end NUMINAMATH_CALUDE_fund_price_calculation_l2196_219604


namespace NUMINAMATH_CALUDE_log_base_condition_l2196_219668

theorem log_base_condition (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ici 2 → |Real.log x / Real.log a| > 1) ↔ (a < 2 ∧ a ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_log_base_condition_l2196_219668


namespace NUMINAMATH_CALUDE_system_solution_unique_l2196_219665

theorem system_solution_unique : 
  ∃! (x y : ℝ), (x - 2*y = 0) ∧ (3*x + 2*y = 8) ∧ (x = 2) ∧ (y = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2196_219665


namespace NUMINAMATH_CALUDE_initial_amount_is_750_l2196_219610

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

/-- Final amount calculation using simple interest -/
def final_amount (principal rate time : ℝ) : ℝ := principal + simple_interest principal rate time

/-- Theorem stating that given the conditions, the initial amount must be 750 -/
theorem initial_amount_is_750 :
  ∀ (P : ℝ),
  final_amount P 0.06 5 = 975 →
  P = 750 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_is_750_l2196_219610


namespace NUMINAMATH_CALUDE_hotel_charge_comparison_l2196_219651

theorem hotel_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.55 * R) 
  (h2 : P = G - 0.1 * G) : 
  (R - G) / G = 1 := by
sorry

end NUMINAMATH_CALUDE_hotel_charge_comparison_l2196_219651


namespace NUMINAMATH_CALUDE_steven_seed_collection_l2196_219612

/-- Represents the number of seeds in different fruits -/
structure FruitSeeds where
  apple : ℕ
  pear : ℕ
  grape : ℕ

/-- Represents the number of fruits Steven has -/
structure FruitCount where
  apples : ℕ
  pears : ℕ
  grapes : ℕ

/-- Calculates the total number of seeds Steven needs to collect -/
def totalSeedsNeeded (avg : FruitSeeds) (count : FruitCount) (additional : ℕ) : ℕ :=
  avg.apple * count.apples + avg.pear * count.pears + avg.grape * count.grapes + additional

/-- Theorem: Steven needs to collect 60 seeds in total -/
theorem steven_seed_collection :
  let avg : FruitSeeds := ⟨6, 2, 3⟩
  let count : FruitCount := ⟨4, 3, 9⟩
  let additional : ℕ := 3
  totalSeedsNeeded avg count additional = 60 := by
  sorry


end NUMINAMATH_CALUDE_steven_seed_collection_l2196_219612


namespace NUMINAMATH_CALUDE_hex_to_binary_digits_l2196_219697

theorem hex_to_binary_digits : ∃ (n : ℕ), n = 20 ∧ 
  (∀ (m : ℕ), 2^m ≤ (11 * 16^4 + 1 * 16^3 + 2 * 16^2 + 3 * 16^1 + 4 * 16^0) → m ≤ n) ∧
  (2^n > 11 * 16^4 + 1 * 16^3 + 2 * 16^2 + 3 * 16^1 + 4 * 16^0) :=
by sorry

end NUMINAMATH_CALUDE_hex_to_binary_digits_l2196_219697


namespace NUMINAMATH_CALUDE_odd_function_extension_l2196_219614

-- Define an odd function on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function for x < 0
def f_neg (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x = x^2 + x

-- Theorem statement
theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_neg : f_neg f) :
  ∀ x, x > 0 → f x = -x^2 + x :=
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l2196_219614


namespace NUMINAMATH_CALUDE_base_seven_528_l2196_219656

def base_seven_representation (n : ℕ) : List ℕ :=
  sorry

theorem base_seven_528 :
  base_seven_representation 528 = [1, 3, 5, 3] := by
  sorry

end NUMINAMATH_CALUDE_base_seven_528_l2196_219656


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2196_219644

theorem remainder_divisibility (x y : ℤ) (h : 9 ∣ (x + 2*y)) :
  ∃ k : ℤ, 2*(5*x - 8*y - 4) = 9*k + (-8) ∨ 2*(5*x - 8*y - 4) = 9*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2196_219644


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2196_219616

noncomputable def f (x : ℝ) : ℝ := Real.log ((2 - x) / (2 + x))

theorem tangent_line_equation :
  let x₀ : ℝ := -1
  let y₀ : ℝ := f x₀
  let m : ℝ := -4/3
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (y = -4/3 * x + Real.log 3 - 4/3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2196_219616


namespace NUMINAMATH_CALUDE_cube_root_of_negative_64_l2196_219622

theorem cube_root_of_negative_64 : ∃ x : ℝ, x^3 = -64 ∧ x = -4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_64_l2196_219622


namespace NUMINAMATH_CALUDE_function_equation_solution_l2196_219677

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x^2 * y^2) →
  (∀ x : ℝ, f x = x^2 ∨ f x = -x^2) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2196_219677


namespace NUMINAMATH_CALUDE_hundred_mile_fare_l2196_219617

/-- Represents the cost of a taxi journey based on distance traveled -/
structure TaxiFare where
  /-- The distance traveled in miles -/
  distance : ℝ
  /-- The cost of the journey in dollars -/
  cost : ℝ

/-- Taxi fare is directly proportional to the distance traveled -/
axiom fare_proportional (d₁ d₂ c₁ c₂ : ℝ) :
  d₁ * c₂ = d₂ * c₁

theorem hundred_mile_fare (f : TaxiFare) (h : f.distance = 80 ∧ f.cost = 192) :
  ∃ (g : TaxiFare), g.distance = 100 ∧ g.cost = 240 :=
sorry

end NUMINAMATH_CALUDE_hundred_mile_fare_l2196_219617


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2196_219620

/-- Given vectors a and b in ℝ², if a is perpendicular to (a - b), then the second component of b is 4. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b.1 = -3) :
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) → b.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2196_219620


namespace NUMINAMATH_CALUDE_lisa_savings_analysis_l2196_219634

def lisa_savings : Fin 6 → ℝ
  | 0 => 100  -- January
  | 1 => 300  -- February
  | 2 => 200  -- March
  | 3 => 200  -- April
  | 4 => 100  -- May
  | 5 => 100  -- June

theorem lisa_savings_analysis :
  let total_average := (lisa_savings 0 + lisa_savings 1 + lisa_savings 2 + 
                        lisa_savings 3 + lisa_savings 4 + lisa_savings 5) / 6
  let first_trimester_average := (lisa_savings 0 + lisa_savings 1 + lisa_savings 2) / 3
  let second_trimester_average := (lisa_savings 3 + lisa_savings 4 + lisa_savings 5) / 3
  (total_average = 1000 / 6) ∧
  (first_trimester_average = 200) ∧
  (second_trimester_average = 400 / 3) ∧
  (first_trimester_average - second_trimester_average = 200 / 3) :=
by sorry

end NUMINAMATH_CALUDE_lisa_savings_analysis_l2196_219634


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2196_219683

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (1 / (x^2 + y*z)) + (1 / (y^2 + z*x)) + (1 / (z^2 + x*y)) ≤ 
  (1 / 2) * ((1 / (x*y)) + (1 / (y*z)) + (1 / (z*x))) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2196_219683


namespace NUMINAMATH_CALUDE_vector_angle_problem_l2196_219696

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem vector_angle_problem (a b : ℝ × ℝ) 
  (h1 : a.1^2 + a.2^2 = 4)
  (h2 : Real.sqrt (b.1^2 + b.2^2) = 2)
  (h3 : (a.1 + b.1) * (3 * a.1 - b.1) + (a.2 + b.2) * (3 * a.2 - b.2) = 4) :
  angle_between_vectors a b = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_problem_l2196_219696


namespace NUMINAMATH_CALUDE_correct_statements_l2196_219675

theorem correct_statements :
  (∀ x : ℝ, x^2 > 0 → x ≠ 0) ∧
  (∀ x : ℝ, x > 1 → x^2 > x) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_l2196_219675


namespace NUMINAMATH_CALUDE_simplify_fraction_l2196_219638

theorem simplify_fraction (m n : ℝ) (h : n ≠ 0) : m * n / (n ^ 2) = m / n := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2196_219638


namespace NUMINAMATH_CALUDE_jessicas_balloons_l2196_219676

/-- Given the number of blue balloons for Joan, Sally, and the total,
    prove that Jessica has 2 blue balloons. -/
theorem jessicas_balloons
  (joan_balloons : ℕ)
  (sally_balloons : ℕ)
  (total_balloons : ℕ)
  (h1 : joan_balloons = 9)
  (h2 : sally_balloons = 5)
  (h3 : total_balloons = 16)
  (h4 : ∃ (jessica_balloons : ℕ), joan_balloons + sally_balloons + jessica_balloons = total_balloons) :
  ∃ (jessica_balloons : ℕ), jessica_balloons = 2 ∧ joan_balloons + sally_balloons + jessica_balloons = total_balloons :=
by
  sorry

end NUMINAMATH_CALUDE_jessicas_balloons_l2196_219676


namespace NUMINAMATH_CALUDE_girls_fraction_is_37_75_l2196_219674

/-- Represents a school with a given number of students and boy-to-girl ratio -/
structure School where
  total_students : ℕ
  boys_ratio : ℕ
  girls_ratio : ℕ

/-- Calculates the number of girls in a school -/
def girls_count (s : School) : ℚ :=
  (s.total_students : ℚ) * s.girls_ratio / (s.boys_ratio + s.girls_ratio)

/-- Calculates the fraction of girls in a gathering of two schools -/
def girls_fraction (s1 s2 : School) : ℚ :=
  (girls_count s1 + girls_count s2) / (s1.total_students + s2.total_students)

theorem girls_fraction_is_37_75 (school_a school_b : School)
  (ha : school_a.total_students = 240 ∧ school_a.boys_ratio = 3 ∧ school_a.girls_ratio = 2)
  (hb : school_b.total_students = 210 ∧ school_b.boys_ratio = 2 ∧ school_b.girls_ratio = 3) :
  girls_fraction school_a school_b = 37 / 75 := by
  sorry

end NUMINAMATH_CALUDE_girls_fraction_is_37_75_l2196_219674


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2196_219643

theorem problem_1 : (1 - Real.sqrt 2) ^ 0 - 2 * Real.sin (π / 4) + (Real.sqrt 2) ^ 2 = 3 - Real.sqrt 2 := by
  sorry

theorem problem_2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) :
  (1 - 2 / (x - 1)) / ((x^2 - 6*x + 9) / (x^2 - 1)) = (x + 1) / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2196_219643


namespace NUMINAMATH_CALUDE_space_shuttle_speed_km_per_second_l2196_219647

def orbit_speed_km_per_hour : ℝ := 43200

theorem space_shuttle_speed_km_per_second :
  orbit_speed_km_per_hour / 3600 = 12 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_km_per_second_l2196_219647


namespace NUMINAMATH_CALUDE_v_1013_equals_5_l2196_219673

def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 4
| 4 => 1
| 5 => 2
| _ => 0  -- Default case for completeness

def v : ℕ → ℕ
| 0 => 3
| (n + 1) => g (v n)

theorem v_1013_equals_5 : v 1013 = 5 := by
  sorry

end NUMINAMATH_CALUDE_v_1013_equals_5_l2196_219673


namespace NUMINAMATH_CALUDE_triangle_translation_inconsistency_l2196_219657

structure Point where
  x : ℝ
  y : ℝ

def Triangle (A B C : Point) := Unit

def isTranslation (A B C A' B' C' : Point) : Prop :=
  ∃ dx dy : ℝ, 
    A'.x = A.x + dx ∧ A'.y = A.y + dy ∧
    B'.x = B.x + dx ∧ B'.y = B.y + dy ∧
    C'.x = C.x + dx ∧ C'.y = C.y + dy

def correctYCoordinates (A B C A' B' C' : Point) : Prop :=
  A'.y = A.y - 3 ∧ B'.y = B.y - 3 ∧ C'.y = C.y - 3

def oneCorrectXCoordinate (A B C A' B' C' : Point) : Prop :=
  (A'.x = A.x ∧ B'.x ≠ B.x - 1 ∧ C'.x ≠ C.x + 1) ∨
  (A'.x ≠ A.x ∧ B'.x = B.x - 1 ∧ C'.x ≠ C.x + 1) ∨
  (A'.x ≠ A.x ∧ B'.x ≠ B.x - 1 ∧ C'.x = C.x + 1)

theorem triangle_translation_inconsistency 
  (A B C A' B' C' : Point)
  (h1 : A = ⟨0, 3⟩)
  (h2 : B = ⟨-1, 0⟩)
  (h3 : C = ⟨1, 0⟩)
  (h4 : A' = ⟨0, 0⟩)
  (h5 : B' = ⟨-2, -3⟩)
  (h6 : C' = ⟨2, -3⟩)
  (h7 : correctYCoordinates A B C A' B' C')
  (h8 : oneCorrectXCoordinate A B C A' B' C') :
  ¬(isTranslation A B C A' B' C') ∧
  ((A' = ⟨0, 0⟩ ∧ B' = ⟨-1, -3⟩ ∧ C' = ⟨1, -3⟩) ∨
   (A' = ⟨-1, 0⟩ ∧ B' = ⟨-2, -3⟩ ∧ C' = ⟨0, -3⟩) ∨
   (A' = ⟨1, 0⟩ ∧ B' = ⟨0, -3⟩ ∧ C' = ⟨2, -3⟩)) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_translation_inconsistency_l2196_219657


namespace NUMINAMATH_CALUDE_fiveDigitNumbers_eq_ten_l2196_219679

/-- The number of five-digit natural numbers formed with digits 1 and 0, containing exactly three 1s -/
def fiveDigitNumbers : ℕ :=
  Nat.choose 5 2

/-- Theorem stating that the number of such five-digit numbers is 10 -/
theorem fiveDigitNumbers_eq_ten : fiveDigitNumbers = 10 := by
  sorry

end NUMINAMATH_CALUDE_fiveDigitNumbers_eq_ten_l2196_219679


namespace NUMINAMATH_CALUDE_power_equation_solution_l2196_219654

theorem power_equation_solution : ∃! x : ℤ, (3 : ℝ) ^ 7 * (3 : ℝ) ^ x = 81 := by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2196_219654


namespace NUMINAMATH_CALUDE_unique_solutions_l2196_219636

def system_solution (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 1 ∧
  x^4 - 18*x^2*y^2 + 81*y^4 - 20*x^2 - 180*y^2 + 100 = 0

theorem unique_solutions :
  (∀ x y : ℝ, system_solution x y →
    ((x = -1/Real.sqrt 10 ∧ y = 3/Real.sqrt 10) ∨
     (x = -1/Real.sqrt 10 ∧ y = -3/Real.sqrt 10) ∨
     (x = 1/Real.sqrt 10 ∧ y = 3/Real.sqrt 10) ∨
     (x = 1/Real.sqrt 10 ∧ y = -3/Real.sqrt 10))) ∧
  (system_solution (-1/Real.sqrt 10) (3/Real.sqrt 10)) ∧
  (system_solution (-1/Real.sqrt 10) (-3/Real.sqrt 10)) ∧
  (system_solution (1/Real.sqrt 10) (3/Real.sqrt 10)) ∧
  (system_solution (1/Real.sqrt 10) (-3/Real.sqrt 10)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_l2196_219636


namespace NUMINAMATH_CALUDE_delta_computation_l2196_219669

-- Define the custom operation
def delta (a b : ℕ) : ℕ := a^2 - b

-- State the theorem
theorem delta_computation :
  delta (5^(delta 6 2)) (4^(delta 7 3)) = 5^68 - 4^46 := by
  sorry

end NUMINAMATH_CALUDE_delta_computation_l2196_219669


namespace NUMINAMATH_CALUDE_fourth_circle_radius_l2196_219689

/-- Represents a configuration of seven circles tangent to each other and two lines -/
structure CircleConfiguration where
  radii : Fin 7 → ℝ
  is_geometric_sequence : ∃ r : ℝ, ∀ i : Fin 6, radii i.succ = radii i * r
  smallest_radius : radii 0 = 6
  largest_radius : radii 6 = 24

/-- The theorem stating that the radius of the fourth circle is 12 -/
theorem fourth_circle_radius (config : CircleConfiguration) : config.radii 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_l2196_219689


namespace NUMINAMATH_CALUDE_complex_magnitude_l2196_219672

theorem complex_magnitude (a b : ℝ) : 
  (Complex.I + a * Complex.I) * Complex.I = 1 - b * Complex.I →
  Complex.abs (a + b * Complex.I) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2196_219672


namespace NUMINAMATH_CALUDE_camel_cost_l2196_219601

-- Define the cost of each animal as a real number
variable (camel horse ox elephant lion bear : ℝ)

-- Define the relationships between animal costs
axiom camel_horse : 10 * camel = 24 * horse
axiom horse_ox : 16 * horse = 4 * ox
axiom ox_elephant : 6 * ox = 4 * elephant
axiom elephant_lion : 3 * elephant = 8 * lion
axiom lion_bear : 2 * lion = 6 * bear
axiom bear_cost : 14 * bear = 204000

-- Theorem to prove
theorem camel_cost : camel = 46542.86 := by sorry

end NUMINAMATH_CALUDE_camel_cost_l2196_219601


namespace NUMINAMATH_CALUDE_mailman_delivery_l2196_219680

theorem mailman_delivery (total_mail junk_mail : ℕ) 
  (h1 : total_mail = 11) 
  (h2 : junk_mail = 6) : 
  total_mail - junk_mail = 5 := by
  sorry

end NUMINAMATH_CALUDE_mailman_delivery_l2196_219680


namespace NUMINAMATH_CALUDE_p_false_q_true_l2196_219633

theorem p_false_q_true (h1 : p ∨ q) (h2 : ¬p) : ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_p_false_q_true_l2196_219633


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l2196_219684

theorem sqrt_product_plus_one : Real.sqrt ((43 : ℝ) * 42 * 41 * 40 + 1) = 1721 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l2196_219684


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2196_219630

theorem sum_of_three_numbers (x y z : ℝ) : 
  z / x = 18.48 / 15.4 →
  z = 0.4 * y →
  x + y = 400 →
  x + y + z = 520 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2196_219630


namespace NUMINAMATH_CALUDE_smallest_coprime_to_180_seven_coprime_to_180_seven_is_smallest_coprime_to_180_l2196_219618

theorem smallest_coprime_to_180 : ∀ x : ℕ, x > 1 ∧ x < 7 → Nat.gcd x 180 ≠ 1 :=
by
  sorry

theorem seven_coprime_to_180 : Nat.gcd 7 180 = 1 :=
by
  sorry

theorem seven_is_smallest_coprime_to_180 : ∀ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 → x ≥ 7 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_coprime_to_180_seven_coprime_to_180_seven_is_smallest_coprime_to_180_l2196_219618


namespace NUMINAMATH_CALUDE_minimum_gloves_needed_l2196_219661

theorem minimum_gloves_needed (participants : ℕ) (gloves_per_participant : ℕ) : 
  participants = 43 → gloves_per_participant = 2 → participants * gloves_per_participant = 86 := by
  sorry

end NUMINAMATH_CALUDE_minimum_gloves_needed_l2196_219661


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2196_219664

-- Problem 1
theorem problem_1 (x : ℝ) : x * (x + 6) + (x - 3)^2 = 2 * x^2 + 9 := by
  sorry

-- Problem 2
theorem problem_2 (m n : ℝ) (hm : m ≠ 0) (hmn : 3 * m ≠ n) :
  (3 + n / m) / ((9 * m^2 - n^2) / m) = 1 / (3 * m - n) := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2196_219664


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l2196_219619

theorem two_digit_number_interchange (x y : ℕ) : 
  x < 10 → y < 10 → x ≠ 0 → (10 * x + y) - (10 * y + x) = 54 → x - y = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l2196_219619


namespace NUMINAMATH_CALUDE_photo_ratio_proof_l2196_219628

def claire_photos : ℕ := 6
def robert_photos (claire : ℕ) : ℕ := claire + 12

theorem photo_ratio_proof (lisa : ℕ) (h1 : lisa = robert_photos claire_photos) :
  lisa / claire_photos = 3 := by
  sorry

end NUMINAMATH_CALUDE_photo_ratio_proof_l2196_219628


namespace NUMINAMATH_CALUDE_sum_of_B_coordinates_l2196_219645

-- Define the points
def A : ℝ × ℝ := (5, -1)
def M : ℝ × ℝ := (4, 3)

-- Define B as a variable point
variable (B : ℝ × ℝ)

-- Define the midpoint condition
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem sum_of_B_coordinates :
  is_midpoint M A B → B.1 + B.2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_B_coordinates_l2196_219645


namespace NUMINAMATH_CALUDE_starting_number_proof_l2196_219640

theorem starting_number_proof (x : ℕ) : 
  (x ≤ 26) → 
  (x % 2 = 0) → 
  ((x + 26) / 2 = 19) → 
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_starting_number_proof_l2196_219640


namespace NUMINAMATH_CALUDE_polygon_edges_l2196_219637

theorem polygon_edges (n : ℕ) : n ≥ 3 → (
  (n - 2) * 180 = 4 * 360 + 180 ↔ n = 11
) := by sorry

end NUMINAMATH_CALUDE_polygon_edges_l2196_219637


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l2196_219692

theorem smallest_solution_quadratic (x : ℝ) :
  (8 * x^2 - 38 * x + 35 = 0) → x ≥ 1.25 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l2196_219692


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l2196_219641

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.2 = a.2 * b.1)

/-- Given vectors a and b, prove that if they are parallel, then t = 9 -/
theorem parallel_vectors_t_value (t : ℝ) :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (3, t)
  are_parallel a b → t = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l2196_219641


namespace NUMINAMATH_CALUDE_factor_tree_value_l2196_219635

theorem factor_tree_value (F G H J X : ℕ) : 
  H = 2 * 5 →
  J = 3 * 7 →
  F = 7 * H →
  G = 11 * J →
  X = F * G →
  X = 16170 :=
by
  sorry

end NUMINAMATH_CALUDE_factor_tree_value_l2196_219635


namespace NUMINAMATH_CALUDE_right_triangle_among_options_l2196_219623

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_among_options : 
  is_right_triangle 1 2 3 ∧ 
  ¬is_right_triangle 3 4 5 ∧ 
  ¬is_right_triangle 6 8 10 ∧ 
  ¬is_right_triangle 5 10 12 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_among_options_l2196_219623


namespace NUMINAMATH_CALUDE_base_b_sum_theorem_l2196_219626

theorem base_b_sum_theorem (b : ℕ) : b > 0 → (
  (b^3 - b^2) / 2 = b^2 + 12*b + 5 ↔ b = 15
) := by sorry

end NUMINAMATH_CALUDE_base_b_sum_theorem_l2196_219626


namespace NUMINAMATH_CALUDE_point_M_on_x_axis_l2196_219631

-- Define a point M with coordinates (a+2, a-3)
def M (a : ℝ) : ℝ × ℝ := (a + 2, a - 3)

-- Define what it means for a point to lie on the x-axis
def lies_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

-- Theorem statement
theorem point_M_on_x_axis :
  ∀ a : ℝ, lies_on_x_axis (M a) → M a = (5, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_M_on_x_axis_l2196_219631


namespace NUMINAMATH_CALUDE_fraction_equality_l2196_219655

theorem fraction_equality : (1 + 5) / (3 + 5) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2196_219655


namespace NUMINAMATH_CALUDE_sum_x_y_equals_negative_two_l2196_219687

theorem sum_x_y_equals_negative_two (x y : ℝ) 
  (h1 : |x| + x + y = 14) 
  (h2 : x + |y| - y = 16) : 
  x + y = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_negative_two_l2196_219687


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2196_219659

theorem x_eq_2_sufficient_not_necessary :
  (∀ x : ℝ, x = 2 → (x + 1) * (x - 2) = 0) ∧
  (∃ x : ℝ, (x + 1) * (x - 2) = 0 ∧ x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2196_219659


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l2196_219662

theorem probability_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total_people = men + women →
  men = 10 →
  women = 5 →
  selected = 4 →
  (1 - (Nat.choose men selected : ℚ) / (Nat.choose total_people selected : ℚ)) = 77 / 91 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l2196_219662


namespace NUMINAMATH_CALUDE_inverse_mod_53_l2196_219611

theorem inverse_mod_53 (h : (19⁻¹ : ZMod 53) = 31) : (44⁻¹ : ZMod 53) = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l2196_219611


namespace NUMINAMATH_CALUDE_arman_work_hours_l2196_219666

/-- Proves that Arman worked 35 hours last week given the conditions of his work schedule and pay. -/
theorem arman_work_hours : ∀ (last_week_hours : ℝ),
  (last_week_hours * 10 + 40 * 10.5 = 770) →
  last_week_hours = 35 := by
  sorry

end NUMINAMATH_CALUDE_arman_work_hours_l2196_219666


namespace NUMINAMATH_CALUDE_h_is_correct_l2196_219606

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := -9*x^3 - x^2 - 4*x + 3

-- State the theorem
theorem h_is_correct : 
  ∀ x : ℝ, 9*x^3 + 6*x^2 - 3*x + 1 + h x = 5*x^2 - 7*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_h_is_correct_l2196_219606


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2196_219605

theorem complex_magnitude_product : |(7 + 6*I)*(-5 + 3*I)| = Real.sqrt 2890 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2196_219605


namespace NUMINAMATH_CALUDE_ellipse_foci_l2196_219632

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 25 + y^2 / 169 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) := {(0, 12), (0, -12)}

-- Theorem statement
theorem ellipse_foci :
  ∀ (x y : ℝ), ellipse_equation x y →
  ∃ (f₁ f₂ : ℝ × ℝ), f₁ ∈ foci ∧ f₂ ∈ foci ∧
  (x - f₁.1)^2 + (y - f₁.2)^2 + (x - f₂.1)^2 + (y - f₂.2)^2 =
  4 * Real.sqrt (13^2 * 5^2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l2196_219632


namespace NUMINAMATH_CALUDE_triangle_properties_l2196_219613

theorem triangle_properties (A B C : ℝ × ℝ) (S : ℝ) :
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  AB.1 * AC.1 + AB.2 * AC.2 = S →
  (C.1 - A.1) / Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 3/5 →
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 2 →
  Real.tan (2 * Real.arctan ((B.2 - A.2) / (B.1 - A.1))) = -4/3 ∧
  S = 8/5 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2196_219613


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2196_219648

/-- Given an arithmetic sequence with a non-zero common difference,
    if the 2nd, 3rd, and 6th terms form a geometric sequence,
    then the common ratio of these three terms is 3. -/
theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  d ≠ 0 →  -- non-zero common difference
  (a 2) * (a 6) = (a 3)^2 →  -- 2nd, 3rd, and 6th terms form a geometric sequence
  (a 3) / (a 2) = 3 :=  -- common ratio is 3
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2196_219648


namespace NUMINAMATH_CALUDE_baby_tarantula_legs_l2196_219658

/-- The number of legs a tarantula has -/
def tarantula_legs : ℕ := 8

/-- The number of tarantulas in one egg sac -/
def tarantulas_per_sac : ℕ := 1000

/-- The number of egg sacs being considered -/
def egg_sacs : ℕ := 5 - 1

/-- The total number of baby tarantula legs in one less than 5 egg sacs -/
def total_baby_legs : ℕ := egg_sacs * tarantulas_per_sac * tarantula_legs

theorem baby_tarantula_legs :
  total_baby_legs = 32000 := by sorry

end NUMINAMATH_CALUDE_baby_tarantula_legs_l2196_219658


namespace NUMINAMATH_CALUDE_sams_seashells_l2196_219629

theorem sams_seashells (mary_seashells : ℕ) (total_seashells : ℕ) 
  (h1 : mary_seashells = 47)
  (h2 : total_seashells = 65) :
  total_seashells - mary_seashells = 18 := by
  sorry

end NUMINAMATH_CALUDE_sams_seashells_l2196_219629


namespace NUMINAMATH_CALUDE_max_volume_special_tetrahedron_l2196_219600

/-- A tetrahedron with two vertices on a sphere of radius √10 and two on a concentric sphere of radius 2 -/
structure SpecialTetrahedron where
  /-- The radius of the larger sphere -/
  R : ℝ
  /-- The radius of the smaller sphere -/
  r : ℝ
  /-- Assertion that R = √10 -/
  h_R : R = Real.sqrt 10
  /-- Assertion that r = 2 -/
  h_r : r = 2

/-- The volume of a SpecialTetrahedron -/
def volume (t : SpecialTetrahedron) : ℝ :=
  sorry

/-- The maximum volume of a SpecialTetrahedron is 6√2 -/
theorem max_volume_special_tetrahedron :
  ∀ t : SpecialTetrahedron, volume t ≤ 6 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_volume_special_tetrahedron_l2196_219600


namespace NUMINAMATH_CALUDE_pine_boys_count_l2196_219690

/-- Represents a middle school in the winter program. -/
inductive School
| Maple
| Pine
| Oak

/-- Represents the gender of a student. -/
inductive Gender
| Boy
| Girl

/-- Represents the distribution of students in the winter program. -/
structure WinterProgram where
  total_students : Nat
  total_boys : Nat
  total_girls : Nat
  maple_students : Nat
  pine_students : Nat
  oak_students : Nat
  maple_girls : Nat

/-- Theorem stating that the number of boys from Pine Middle School is 20. -/
theorem pine_boys_count (wp : WinterProgram) 
  (h1 : wp.total_students = 120)
  (h2 : wp.total_boys = 68)
  (h3 : wp.total_girls = 52)
  (h4 : wp.maple_students = 50)
  (h5 : wp.pine_students = 40)
  (h6 : wp.oak_students = 30)
  (h7 : wp.maple_girls = 22)
  (h8 : wp.total_students = wp.total_boys + wp.total_girls)
  (h9 : wp.total_students = wp.maple_students + wp.pine_students + wp.oak_students) :
  ∃ (pine_boys : Nat), pine_boys = 20 ∧ 
    pine_boys + (wp.pine_students - pine_boys) = wp.pine_students :=
  sorry


end NUMINAMATH_CALUDE_pine_boys_count_l2196_219690


namespace NUMINAMATH_CALUDE_period_is_24_hours_period_in_hours_is_24_l2196_219698

/-- Represents the period in seconds --/
def period (birth_rate : ℚ) (death_rate : ℚ) (net_increase : ℕ) : ℚ :=
  net_increase / (birth_rate / 2 - death_rate / 2)

/-- Theorem stating that the period is 24 hours given the problem conditions --/
theorem period_is_24_hours :
  let birth_rate : ℚ := 10
  let death_rate : ℚ := 2
  let net_increase : ℕ := 345600
  period birth_rate death_rate net_increase = 86400 := by
  sorry

/-- Converts seconds to hours --/
def seconds_to_hours (seconds : ℚ) : ℚ :=
  seconds / 3600

/-- Theorem stating that 86400 seconds is equal to 24 hours --/
theorem period_in_hours_is_24 :
  seconds_to_hours 86400 = 24 := by
  sorry

end NUMINAMATH_CALUDE_period_is_24_hours_period_in_hours_is_24_l2196_219698
