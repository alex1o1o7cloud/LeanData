import Mathlib

namespace NUMINAMATH_GPT_symmetric_function_value_l1855_185526

theorem symmetric_function_value (f : ℝ → ℝ)
  (h : ∀ x, f (2^(x-2)) = x) : f 8 = 5 :=
sorry

end NUMINAMATH_GPT_symmetric_function_value_l1855_185526


namespace NUMINAMATH_GPT_fractional_eq_a_range_l1855_185543

theorem fractional_eq_a_range (a : ℝ) :
  (∃ x : ℝ, (a / (x + 2) = 1 - 3 / (x + 2)) ∧ (x < 0)) ↔ (a < -1 ∧ a ≠ -3) := by
  sorry

end NUMINAMATH_GPT_fractional_eq_a_range_l1855_185543


namespace NUMINAMATH_GPT_initial_students_count_eq_16_l1855_185510

variable (n T : ℕ)
variable (h1 : (T:ℝ) / n = 62.5)
variable (h2 : ((T - 70):ℝ) / (n - 1) = 62.0)

theorem initial_students_count_eq_16 :
  n = 16 :=
by
  sorry

end NUMINAMATH_GPT_initial_students_count_eq_16_l1855_185510


namespace NUMINAMATH_GPT_smallest_possible_average_l1855_185528

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def proper_digits (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 4 ∨ d = 8

theorem smallest_possible_average :
  ∃ n : ℕ, (n + 2) - n = 2 ∧ (sum_of_digits n + sum_of_digits (n + 2)) % 4 = 0 ∧ (∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 4 ∨ d = 8) ∧ ∀ (d : ℕ), d ∈ (n + 2).digits 10 → d = 0 ∨ d = 4 ∨ d = 8 
  ∧ (n + (n + 2)) / 2 = 249 :=
sorry

end NUMINAMATH_GPT_smallest_possible_average_l1855_185528


namespace NUMINAMATH_GPT_area_percent_of_smaller_rectangle_l1855_185523

-- Definitions of the main geometric elements and assumptions
def larger_rectangle (w h : ℝ) : Prop := (w > 0) ∧ (h > 0)
def radius_of_circle (w h r : ℝ) : Prop := r = Real.sqrt (w^2 + h^2)
def inscribed_smaller_rectangle (w h x y : ℝ) : Prop := 
  (0 < x) ∧ (x < 1) ∧ (0 < y) ∧ (y < 1) ∧
  ((h + 2 * y * h)^2 + (x * w)^2 = w^2 + h^2)

-- Prove the area percentage relationship
theorem area_percent_of_smaller_rectangle 
  (w h x y : ℝ) 
  (hw : w > 0) (hh : h > 0)
  (hcirc : radius_of_circle w h (Real.sqrt (w^2 + h^2)))
  (hsmall_rect : inscribed_smaller_rectangle w h x y) :
  (4 * x * y) / (4.0 * 1.0) * 100 = 8.33 := sorry

end NUMINAMATH_GPT_area_percent_of_smaller_rectangle_l1855_185523


namespace NUMINAMATH_GPT_four_digit_div_by_99_then_sum_div_by_18_l1855_185586

/-- 
If a whole number with at most four digits is divisible by 99, then 
the sum of its digits is divisible by 18. 
-/
theorem four_digit_div_by_99_then_sum_div_by_18 (n : ℕ) (h1 : n < 10000) (h2 : 99 ∣ n) : 
  18 ∣ (n.digits 10).sum := 
sorry

end NUMINAMATH_GPT_four_digit_div_by_99_then_sum_div_by_18_l1855_185586


namespace NUMINAMATH_GPT_sum_abc_l1855_185525

theorem sum_abc (A B C : ℕ) (hposA : 0 < A) (hposB : 0 < B) (hposC : 0 < C) (hgcd : Nat.gcd A (Nat.gcd B C) = 1)
  (hlog : A * Real.log 5 / Real.log 100 + B * Real.log 2 / Real.log 100 = C) : A + B + C = 5 :=
sorry

end NUMINAMATH_GPT_sum_abc_l1855_185525


namespace NUMINAMATH_GPT_arrangement_of_chairs_and_stools_l1855_185531

theorem arrangement_of_chairs_and_stools :
  (Nat.choose 10 3) = 120 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_arrangement_of_chairs_and_stools_l1855_185531


namespace NUMINAMATH_GPT_division_remainder_l1855_185592

theorem division_remainder (dividend divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 15) 
  (h_quotient : quotient = 9) 
  (h_dividend_eq : dividend = 136) 
  (h_eq : dividend = (divisor * quotient) + remainder) : 
  remainder = 1 :=
by
  sorry

end NUMINAMATH_GPT_division_remainder_l1855_185592


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1855_185511

theorem solution_set_of_inequality :
  { x : ℝ | |x^2 - 3 * x| > 4 } = { x : ℝ | x < -1 ∨ x > 4 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1855_185511


namespace NUMINAMATH_GPT_motorcycle_wheels_l1855_185574

/--
In a parking lot, there are cars and motorcycles. Each car has 5 wheels (including one spare) 
and each motorcycle has a certain number of wheels. There are 19 cars in the parking lot.
Altogether all vehicles have 117 wheels. There are 11 motorcycles at the parking lot.
--/
theorem motorcycle_wheels (num_cars num_motorcycles total_wheels wheels_per_car wheels_per_motorcycle : ℕ)
  (h1 : wheels_per_car = 5) 
  (h2 : num_cars = 19) 
  (h3 : total_wheels = 117) 
  (h4 : num_motorcycles = 11) 
  : wheels_per_motorcycle = 2 :=
by
  sorry

end NUMINAMATH_GPT_motorcycle_wheels_l1855_185574


namespace NUMINAMATH_GPT_flight_duration_l1855_185514

theorem flight_duration (h m : ℕ) (H1 : 11 * 60 + 7 < 14 * 60 + 45) (H2 : 0 < m) (H3 : m < 60) :
  h + m = 41 := 
sorry

end NUMINAMATH_GPT_flight_duration_l1855_185514


namespace NUMINAMATH_GPT_trig_identity_proof_l1855_185545

theorem trig_identity_proof :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 3 :=
by sorry

end NUMINAMATH_GPT_trig_identity_proof_l1855_185545


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1855_185593

noncomputable def f (x : ℝ) : ℝ := 0.5 * x^2 + 0.5 * x

theorem problem1 (h : ∀ x : ℝ, f (x + 1) = f x + x + 1) (h0 : f 0 = 0) : 
  ∀ x : ℝ, f x = 0.5 * x^2 + 0.5 * x := by 
  sorry

noncomputable def g (t : ℝ) : ℝ :=
  if t ≤ -1.5 then 0.5 * t^2 + 1.5 * t + 1
  else if -1.5 < t ∧ t < -0.5 then -1 / 8
  else 0.5 * t^2 + 0.5 * t

theorem problem2 (h : ∀ t : ℝ, g t = min (f (t)) (f (t + 1))) : 
  ∀ t : ℝ, g t = 
    if t ≤ -1.5 then 0.5 * t^2 + 1.5 * t + 1
    else if -1.5 < t ∧ t < -0.5 then -1 / 8
    else 0.5 * t^2 + 0.5 * t := by 
  sorry

theorem problem3 (m : ℝ) : (∀ t : ℝ, g t + m ≥ 0) → m ≥ 1 / 8 := by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1855_185593


namespace NUMINAMATH_GPT_largest_sum_of_digits_l1855_185578

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9) (h4 : 1 ≤ y ∧ y ≤ 10) (h5 : (1000 * (a * 100 + b * 10 + c)) = 1000) : 
  a + b + c = 8 :=
sorry

end NUMINAMATH_GPT_largest_sum_of_digits_l1855_185578


namespace NUMINAMATH_GPT_Jake_has_8_peaches_l1855_185564

variable (Steven Jill Jake : ℕ)

-- Conditions
axiom h1 : Steven = 15
axiom h2 : Steven = Jill + 14
axiom h3 : Jake = Steven - 7

-- Goal
theorem Jake_has_8_peaches : Jake = 8 := by
  sorry

end NUMINAMATH_GPT_Jake_has_8_peaches_l1855_185564


namespace NUMINAMATH_GPT_complement_of_union_is_neg3_l1855_185541

open Set

variable (U A B : Set Int)

def complement_union (U A B : Set Int) : Set Int :=
  U \ (A ∪ B)

theorem complement_of_union_is_neg3 (U A B : Set Int) (hU : U = {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6})
  (hA : A = {-1, 0, 1, 2, 3}) (hB : B = {-2, 3, 4, 5, 6}) :
  complement_union U A B = {-3} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_union_is_neg3_l1855_185541


namespace NUMINAMATH_GPT_petya_vasya_meet_at_lantern_64_l1855_185532

-- Define the total number of lanterns and intervals
def total_lanterns : ℕ := 100
def total_intervals : ℕ := total_lanterns - 1

-- Define the positions of Petya and Vasya at a given time
def petya_initial : ℕ := 1
def vasya_initial : ℕ := 100
def petya_position : ℕ := 22
def vasya_position : ℕ := 88

-- Define the number of intervals covered by Petya and Vasya
def petya_intervals_covered : ℕ := petya_position - petya_initial
def vasya_intervals_covered : ℕ := vasya_initial - vasya_position

-- Define the combined intervals covered
def combined_intervals_covered : ℕ := petya_intervals_covered + vasya_intervals_covered

-- Define the interval after which Petya and Vasya will meet
def meeting_intervals : ℕ := total_intervals - combined_intervals_covered

-- Define the final meeting point according to Petya's travel
def meeting_lantern : ℕ := petya_initial + (meeting_intervals / 2)

theorem petya_vasya_meet_at_lantern_64 : meeting_lantern = 64 := by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_petya_vasya_meet_at_lantern_64_l1855_185532


namespace NUMINAMATH_GPT_solve_mt_eq_l1855_185502

theorem solve_mt_eq (m n : ℤ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m^2 + n) * (m + n^2) = (m - n)^3 →
  (m = -1 ∧ n = -1) ∨ (m = 8 ∧ n = -10) ∨ (m = 9 ∧ n = -6) ∨ (m = 9 ∧ n = -21) :=
by
  sorry

end NUMINAMATH_GPT_solve_mt_eq_l1855_185502


namespace NUMINAMATH_GPT_sum_arithmetic_seq_nine_terms_l1855_185559

theorem sum_arithmetic_seq_nine_terms
  (a_n : ℕ → ℝ)
  (S_n : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, a_n n = k * n + 4 - 5 * k)
  (h2 : ∀ n, S_n n = (n / 2) * (a_n 1 + a_n n))
  : S_n 9 = 36 :=
sorry

end NUMINAMATH_GPT_sum_arithmetic_seq_nine_terms_l1855_185559


namespace NUMINAMATH_GPT_seokjin_fewer_books_l1855_185576

theorem seokjin_fewer_books (init_books : ℕ) (jungkook_initial : ℕ) (seokjin_initial : ℕ) (jungkook_bought : ℕ) (seokjin_bought : ℕ) :
  jungkook_initial = init_books → seokjin_initial = init_books → jungkook_bought = 18 → seokjin_bought = 11 →
  jungkook_initial + jungkook_bought - (seokjin_initial + seokjin_bought) = 7 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  sorry

end NUMINAMATH_GPT_seokjin_fewer_books_l1855_185576


namespace NUMINAMATH_GPT_min_rows_for_students_l1855_185535

def min_rows (total_students seats_per_row max_students_per_school : ℕ) : ℕ :=
  total_students / seats_per_row + if total_students % seats_per_row == 0 then 0 else 1

theorem min_rows_for_students :
  ∀ (total_students seats_per_row max_students_per_school : ℕ),
  (total_students = 2016) →
  (seats_per_row = 168) →
  (max_students_per_school = 40) →
  min_rows total_students seats_per_row max_students_per_school = 15 :=
by
  intros total_students seats_per_row max_students_per_school h1 h2 h3
  -- We write down the proof outline to show that 15 is the required minimum
  sorry

end NUMINAMATH_GPT_min_rows_for_students_l1855_185535


namespace NUMINAMATH_GPT_gcd_547_323_l1855_185547

theorem gcd_547_323 : Nat.gcd 547 323 = 1 := 
by
  sorry

end NUMINAMATH_GPT_gcd_547_323_l1855_185547


namespace NUMINAMATH_GPT_arithmetic_geometric_progression_l1855_185575

-- Define the arithmetic progression terms
def u (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Define the property that the squares of the 12th, 13th, and 15th terms form a geometric progression
def geometric_progression (a d : ℝ) : Prop :=
  let u12 := u a d 12
  let u13 := u a d 13
  let u15 := u a d 15
  (u13^2 / u12^2 = u15^2 / u13^2)

-- The main statement
theorem arithmetic_geometric_progression (a d : ℝ) (h : geometric_progression a d) :
  d = 0 ∨ 4 * ((a + 11 * d)^2) = (a + 12 *d)^2 * (a + 14 * d)^2 / (a + 12 * d)^2 ∨ (a + 11 * d) * ((a + 11 * d) - 2 *d) = 0 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_progression_l1855_185575


namespace NUMINAMATH_GPT_percentage_of_y_l1855_185598

theorem percentage_of_y (y : ℝ) : (0.3 * 0.6 * y = 0.18 * y) :=
by {
  sorry
}

end NUMINAMATH_GPT_percentage_of_y_l1855_185598


namespace NUMINAMATH_GPT_solution_set_of_absolute_inequality_l1855_185501

theorem solution_set_of_absolute_inequality :
  {x : ℝ | |2 * x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_absolute_inequality_l1855_185501


namespace NUMINAMATH_GPT_sqrt_0_1681_eq_0_41_l1855_185524

theorem sqrt_0_1681_eq_0_41 (h : Real.sqrt 16.81 = 4.1) : Real.sqrt 0.1681 = 0.41 := by 
  sorry

end NUMINAMATH_GPT_sqrt_0_1681_eq_0_41_l1855_185524


namespace NUMINAMATH_GPT_bride_older_than_groom_l1855_185562

-- Define the ages of the bride and groom
variables (B G : ℕ)

-- Given conditions
def groom_age : Prop := G = 83
def total_age : Prop := B + G = 185

-- Theorem to prove how much older the bride is than the groom
theorem bride_older_than_groom (h1 : groom_age G) (h2 : total_age B G) : B - G = 19 :=
sorry

end NUMINAMATH_GPT_bride_older_than_groom_l1855_185562


namespace NUMINAMATH_GPT_victor_won_games_l1855_185513

theorem victor_won_games (V : ℕ) (ratio_victor_friend : 9 * 20 = 5 * V) : V = 36 :=
sorry

end NUMINAMATH_GPT_victor_won_games_l1855_185513


namespace NUMINAMATH_GPT_am_gm_inequality_l1855_185552

theorem am_gm_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by 
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l1855_185552


namespace NUMINAMATH_GPT_oliver_shirts_not_washed_l1855_185536

theorem oliver_shirts_not_washed :
  let short_sleeve_shirts := 39
  let long_sleeve_shirts := 47
  let total_shirts := short_sleeve_shirts + long_sleeve_shirts
  let washed_shirts := 20
  let not_washed_shirts := total_shirts - washed_shirts
  not_washed_shirts = 66 := by
  sorry

end NUMINAMATH_GPT_oliver_shirts_not_washed_l1855_185536


namespace NUMINAMATH_GPT_second_plan_minutes_included_l1855_185553

theorem second_plan_minutes_included 
  (monthly_fee1 : ℝ := 50) 
  (limit1 : ℝ := 500) 
  (cost_per_minute1 : ℝ := 0.35) 
  (monthly_fee2 : ℝ := 75) 
  (cost_per_minute2 : ℝ := 0.45) 
  (M : ℝ) 
  (usage : ℝ := 2500)
  (cost1 := monthly_fee1 + cost_per_minute1 * (usage - limit1))
  (cost2 := monthly_fee2 + cost_per_minute2 * (usage - M))
  (equal_costs : cost1 = cost2) : 
  M = 1000 := 
by
  sorry 

end NUMINAMATH_GPT_second_plan_minutes_included_l1855_185553


namespace NUMINAMATH_GPT_exists_unique_continuous_extension_l1855_185599

noncomputable def F (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) : ℝ → ℝ :=
  sorry

theorem exists_unique_continuous_extension (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) :
  ∃! F : ℝ → ℝ, Continuous F ∧ ∀ x : ℚ, F x = f x :=
sorry

end NUMINAMATH_GPT_exists_unique_continuous_extension_l1855_185599


namespace NUMINAMATH_GPT_jason_books_is_21_l1855_185518

def keith_books : ℕ := 20
def total_books : ℕ := 41

theorem jason_books_is_21 (jason_books : ℕ) : 
  jason_books + keith_books = total_books → 
  jason_books = 21 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_jason_books_is_21_l1855_185518


namespace NUMINAMATH_GPT_square_of_1037_l1855_185580

theorem square_of_1037 : (1037 : ℕ)^2 = 1074369 := 
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_square_of_1037_l1855_185580


namespace NUMINAMATH_GPT_number_of_ways_to_form_team_l1855_185508

-- Defining the conditions
def total_employees : ℕ := 15
def num_men : ℕ := 10
def num_women : ℕ := 5
def team_size : ℕ := 6
def men_in_team : ℕ := 4
def women_in_team : ℕ := 2

-- Using binomial coefficient to represent combinations
noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to be proved
theorem number_of_ways_to_form_team :
  (choose num_men men_in_team) * (choose num_women women_in_team) = 
  choose 10 4 * choose 5 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_form_team_l1855_185508


namespace NUMINAMATH_GPT_correct_calculation_l1855_185569

theorem correct_calculation (a b : ℝ) : 
  (a + 2 * a = 3 * a) := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1855_185569


namespace NUMINAMATH_GPT_total_cost_verification_l1855_185565

-- Conditions given in the problem
def holstein_cost : ℕ := 260
def jersey_cost : ℕ := 170
def num_hearts_on_card : ℕ := 4
def num_cards_in_deck : ℕ := 52
def cow_ratio_holstein : ℕ := 3
def cow_ratio_jersey : ℕ := 2
def sales_tax : ℝ := 0.05
def transport_cost_per_cow : ℕ := 20

def num_hearts_in_deck := num_cards_in_deck
def total_num_cows := 2 * num_hearts_in_deck
def total_parts_ratio := cow_ratio_holstein + cow_ratio_jersey

-- Total number of cows calculated 
def num_holstein_cows : ℕ := (cow_ratio_holstein * total_num_cows) / total_parts_ratio
def num_jersey_cows : ℕ := (cow_ratio_jersey * total_num_cows) / total_parts_ratio

-- Cost calculations
def holstein_total_cost := num_holstein_cows * holstein_cost
def jersey_total_cost := num_jersey_cows * jersey_cost
def total_cost_before_tax_and_transport := holstein_total_cost + jersey_total_cost
def total_sales_tax := total_cost_before_tax_and_transport * sales_tax
def total_transport_cost := total_num_cows * transport_cost_per_cow
def final_total_cost := total_cost_before_tax_and_transport + total_sales_tax + total_transport_cost

-- Lean statement to prove the result
theorem total_cost_verification : final_total_cost = 26324.50 := by sorry

end NUMINAMATH_GPT_total_cost_verification_l1855_185565


namespace NUMINAMATH_GPT_fencing_required_l1855_185584

theorem fencing_required (length width area : ℕ) (length_eq : length = 30) (area_eq : area = 810) 
  (field_area : length * width = area) : 2 * length + width = 87 := 
by
  sorry

end NUMINAMATH_GPT_fencing_required_l1855_185584


namespace NUMINAMATH_GPT_maximize_abs_sum_solution_problem_l1855_185585

theorem maximize_abs_sum_solution :
ℤ → ℤ → Ennreal := sorry

theorem problem :
  (∃ (x y : ℤ), 6 * x^2 + 5 * x * y + y^2 = 6 * x + 2 * y + 7 ∧ 
  x = -8 ∧ y = 25 ∧ (maximize_abs_sum_solution x y = 33)) := sorry

end NUMINAMATH_GPT_maximize_abs_sum_solution_problem_l1855_185585


namespace NUMINAMATH_GPT_factor_polynomial_l1855_185530

theorem factor_polynomial (y : ℝ) : 
  y^6 - 64 = (y - 2) * (y + 2) * (y^2 + 2 * y + 4) * (y^2 - 2 * y + 4) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1855_185530


namespace NUMINAMATH_GPT_investment_recovery_l1855_185596

-- Define the conditions and the goal
theorem investment_recovery (c : ℕ) : 
  (15 * c - 5 * c) ≥ 8000 ↔ c ≥ 800 := 
sorry

end NUMINAMATH_GPT_investment_recovery_l1855_185596


namespace NUMINAMATH_GPT_fraction_division_addition_l1855_185572

theorem fraction_division_addition :
  (3 / 7 / 4) + (2 / 7) = 11 / 28 := by
  sorry

end NUMINAMATH_GPT_fraction_division_addition_l1855_185572


namespace NUMINAMATH_GPT_percentage_of_students_70_79_l1855_185538

-- Defining basic conditions
def students_in_range_90_100 := 5
def students_in_range_80_89 := 9
def students_in_range_70_79 := 7
def students_in_range_60_69 := 4
def students_below_60 := 3

-- Total number of students
def total_students := students_in_range_90_100 + students_in_range_80_89 + students_in_range_70_79 + students_in_range_60_69 + students_below_60

-- Percentage of students in the 70%-79% range
def percent_students_70_79 := (students_in_range_70_79 / total_students) * 100

theorem percentage_of_students_70_79 : percent_students_70_79 = 25 := by
  sorry

end NUMINAMATH_GPT_percentage_of_students_70_79_l1855_185538


namespace NUMINAMATH_GPT_cos_thm_l1855_185517

variable (θ : ℝ)

-- Conditions
def condition1 : Prop := 3 * Real.sin (2 * θ) = 4 * Real.tan θ
def condition2 : Prop := ∀ k : ℤ, θ ≠ k * Real.pi

-- Prove that cos 2θ = 1/3 given the conditions
theorem cos_thm (h1 : condition1 θ) (h2 : condition2 θ) : Real.cos (2 * θ) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_thm_l1855_185517


namespace NUMINAMATH_GPT_divisors_not_multiples_of_14_l1855_185582

theorem divisors_not_multiples_of_14 (m : ℕ)
  (h1 : ∃ k : ℕ, m = 2 * k ∧ (k : ℕ) * k = m / 2)  
  (h2 : ∃ k : ℕ, m = 3 * k ∧ (k : ℕ) * k * k = m / 3)  
  (h3 : ∃ k : ℕ, m = 7 * k ∧ (k : ℕ) ^ 7 = m / 7) : 
  let total_divisors := (6 + 1) * (10 + 1) * (7 + 1)
  let divisors_divisible_by_14 := (5 + 1) * (10 + 1) * (6 + 1)
  total_divisors - divisors_divisible_by_14 = 154 :=
by
  sorry

end NUMINAMATH_GPT_divisors_not_multiples_of_14_l1855_185582


namespace NUMINAMATH_GPT_total_balls_l1855_185542

def black_balls : ℕ := 8
def white_balls : ℕ := 6 * black_balls
theorem total_balls : white_balls + black_balls = 56 := 
by 
  sorry

end NUMINAMATH_GPT_total_balls_l1855_185542


namespace NUMINAMATH_GPT_ac_bc_nec_not_suff_l1855_185597

theorem ac_bc_nec_not_suff (a b c : ℝ) : 
  (a = b → a * c = b * c) ∧ (¬(a * c = b * c → a = b)) := by
  sorry

end NUMINAMATH_GPT_ac_bc_nec_not_suff_l1855_185597


namespace NUMINAMATH_GPT_find_p_l1855_185561

theorem find_p (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : p = 52 / 11 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l1855_185561


namespace NUMINAMATH_GPT_car_cost_l1855_185588

def initial_savings : ℕ := 14500
def charge_per_trip : ℚ := 1.5
def percentage_groceries_earnings : ℚ := 0.05
def number_of_trips : ℕ := 40
def total_value_of_groceries : ℕ := 800

theorem car_cost (initial_savings charge_per_trip percentage_groceries_earnings number_of_trips total_value_of_groceries : ℚ) :
  initial_savings + (charge_per_trip * number_of_trips) + (percentage_groceries_earnings * total_value_of_groceries) = 14600 := 
by
  sorry

end NUMINAMATH_GPT_car_cost_l1855_185588


namespace NUMINAMATH_GPT_expression_equals_16_l1855_185573

open Real

theorem expression_equals_16 (x : ℝ) :
  (x + 1) ^ 2 + 2 * (x + 1) * (3 - x) + (3 - x) ^ 2 = 16 :=
sorry

end NUMINAMATH_GPT_expression_equals_16_l1855_185573


namespace NUMINAMATH_GPT_xy_value_x2_y2_value_l1855_185527

noncomputable def x : ℝ := Real.sqrt 7 + Real.sqrt 3
noncomputable def y : ℝ := Real.sqrt 7 - Real.sqrt 3

theorem xy_value : x * y = 4 := by
  -- proof goes here
  sorry

theorem x2_y2_value : x^2 + y^2 = 20 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_xy_value_x2_y2_value_l1855_185527


namespace NUMINAMATH_GPT_rhombus_diagonals_l1855_185505

theorem rhombus_diagonals (p d_sum : ℝ) (h₁ : p = 100) (h₂ : d_sum = 62) :
  ∃ d₁ d₂ : ℝ, (d₁ + d₂ = d_sum) ∧ (d₁^2 + d₂^2 = (p/4)^2 * 4) ∧ ((d₁ = 48 ∧ d₂ = 14) ∨ (d₁ = 14 ∧ d₂ = 48)) :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonals_l1855_185505


namespace NUMINAMATH_GPT_city_map_representation_l1855_185556

-- Given conditions
def scale (x : ℕ) : ℕ := x * 6
def cm_represents_km(cm : ℕ) : ℕ := scale cm
def fifteen_cm := 15
def ninety_km := 90

-- Given condition: 15 centimeters represents 90 kilometers
axiom representation : cm_represents_km fifteen_cm = ninety_km

-- Proof statement: A 20-centimeter length represents 120 kilometers
def twenty_cm := 20
def correct_answer := 120

theorem city_map_representation : cm_represents_km twenty_cm = correct_answer := by
  sorry

end NUMINAMATH_GPT_city_map_representation_l1855_185556


namespace NUMINAMATH_GPT_boys_at_park_l1855_185507

theorem boys_at_park (girls parents groups people_per_group : ℕ) 
  (h_girls : girls = 14) 
  (h_parents : parents = 50)
  (h_groups : groups = 3) 
  (h_people_per_group : people_per_group = 25) : 
  (groups * people_per_group) - (girls + parents) = 11 := 
by 
  -- Not providing the proof, only the statement
  sorry

end NUMINAMATH_GPT_boys_at_park_l1855_185507


namespace NUMINAMATH_GPT_negation_proposition_iff_l1855_185587

-- Define propositions and their components
def P (x : ℝ) : Prop := x > 1
def Q (x : ℝ) : Prop := x^2 > 1

-- State the proof problem
theorem negation_proposition_iff (x : ℝ) : ¬ (P x → Q x) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by 
  sorry

end NUMINAMATH_GPT_negation_proposition_iff_l1855_185587


namespace NUMINAMATH_GPT_polynomial_factor_l1855_185566

theorem polynomial_factor (a b : ℝ) :
  (∃ (c d : ℝ), a = 4 * c ∧ b = -3 * c + 4 * d ∧ 40 = 2 * c - 3 * d + 18 ∧ -20 = 2 * d - 9 ∧ 9 = 9) →
  a = 11 ∧ b = -121 / 4 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_factor_l1855_185566


namespace NUMINAMATH_GPT_two_students_cover_all_questions_l1855_185533

-- Define the main properties
variables (students : Finset ℕ) (questions : Finset ℕ)
variable (solves : ℕ → ℕ → Prop)

-- Assume the given conditions
axiom total_students : students.card = 8
axiom total_questions : questions.card = 8
axiom each_question_solved_by_min_5_students : ∀ q, q ∈ questions → 
(∃ student_set : Finset ℕ, student_set.card ≥ 5 ∧ ∀ s ∈ student_set, solves s q)

-- The theorem to be proven
theorem two_students_cover_all_questions :
  ∃ s1 s2 : ℕ, s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧ 
  ∀ q ∈ questions, solves s1 q ∨ solves s2 q :=
sorry -- proof to be written

end NUMINAMATH_GPT_two_students_cover_all_questions_l1855_185533


namespace NUMINAMATH_GPT_derivative_f_intervals_of_monotonicity_extrema_l1855_185503

noncomputable def f (x : ℝ) := (x + 1)^2 * (x - 1)

theorem derivative_f (x : ℝ) : deriv f x = 3 * x^2 + 2 * x - 1 := sorry

theorem intervals_of_monotonicity :
  (∀ x, x < -1 → deriv f x > 0) ∧
  (∀ x, -1 < x ∧ x < -1/3 → deriv f x < 0) ∧
  (∀ x, x > -1/3 → deriv f x > 0) := sorry

theorem extrema :
  f (-1) = 0 ∧
  f (-1/3) = -(32 / 27) := sorry

end NUMINAMATH_GPT_derivative_f_intervals_of_monotonicity_extrema_l1855_185503


namespace NUMINAMATH_GPT_anthony_total_pencils_l1855_185581

theorem anthony_total_pencils (initial_pencils : ℕ) (pencils_given_by_kathryn : ℕ) (total_pencils : ℕ) :
  initial_pencils = 9 →
  pencils_given_by_kathryn = 56 →
  total_pencils = initial_pencils + pencils_given_by_kathryn →
  total_pencils = 65 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_anthony_total_pencils_l1855_185581


namespace NUMINAMATH_GPT_estimate_greater_than_exact_l1855_185520

namespace NasreenRounding

variables (a b c d a' b' c' d' : ℕ)

-- Conditions: a, b, c, and d are large positive integers.
-- Definitions for rounding up and down
def round_up (n : ℕ) : ℕ := n + 1  -- Simplified model for rounding up
def round_down (n : ℕ) : ℕ := n - 1  -- Simplified model for rounding down

-- Conditions: a', b', c', and d' are the rounded values of a, b, c, and d respectively.
variable (h_round_a_up : a' = round_up a)
variable (h_round_b_down : b' = round_down b)
variable (h_round_c_down : c' = round_down c)
variable (h_round_d_down : d' = round_down d)

-- Question: Show that the estimate is greater than the original
theorem estimate_greater_than_exact :
  (a' / b' - c' * d') > (a / b - c * d) :=
sorry

end NasreenRounding

end NUMINAMATH_GPT_estimate_greater_than_exact_l1855_185520


namespace NUMINAMATH_GPT_product_of_numbers_l1855_185548

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 :=
sorry

end NUMINAMATH_GPT_product_of_numbers_l1855_185548


namespace NUMINAMATH_GPT_range_of_x_l1855_185589

def valid_domain (x : ℝ) : Prop :=
  (3 - x ≥ 0) ∧ (x ≠ 4)

theorem range_of_x : ∀ x : ℝ, valid_domain x ↔ (x ≤ 3) :=
by sorry

end NUMINAMATH_GPT_range_of_x_l1855_185589


namespace NUMINAMATH_GPT_Cindy_crayons_l1855_185554

variable (K : ℕ) -- Karen's crayons
variable (C : ℕ) -- Cindy's crayons

-- Given conditions
def Karen_has_639_crayons : Prop := K = 639
def Karen_has_135_more_crayons_than_Cindy : Prop := K = C + 135

-- The proof problem: showing Cindy's crayons
theorem Cindy_crayons (h1 : Karen_has_639_crayons K) (h2 : Karen_has_135_more_crayons_than_Cindy K C) : C = 504 :=
by
  sorry

end NUMINAMATH_GPT_Cindy_crayons_l1855_185554


namespace NUMINAMATH_GPT_amount_spent_on_food_l1855_185583

-- We define the conditions given in the problem
def Mitzi_brought_money : ℕ := 75
def ticket_cost : ℕ := 30
def tshirt_cost : ℕ := 23
def money_left : ℕ := 9

-- Define the total amount Mitzi spent
def total_spent : ℕ := Mitzi_brought_money - money_left

-- Define the combined cost of the ticket and T-shirt
def combined_cost : ℕ := ticket_cost + tshirt_cost

-- The proof goal
theorem amount_spent_on_food : total_spent - combined_cost = 13 := by
  sorry

end NUMINAMATH_GPT_amount_spent_on_food_l1855_185583


namespace NUMINAMATH_GPT_every_positive_integer_has_good_multiple_l1855_185557

def is_good (n : ℕ) : Prop :=
  ∃ (D : Finset ℕ), (D.sum id = n) ∧ (1 ∈ D) ∧ (∀ d ∈ D, d ∣ n)

theorem every_positive_integer_has_good_multiple (n : ℕ) (hn : n > 0) : ∃ m : ℕ, (m % n = 0) ∧ is_good m :=
  sorry

end NUMINAMATH_GPT_every_positive_integer_has_good_multiple_l1855_185557


namespace NUMINAMATH_GPT_ratio_of_segments_of_hypotenuse_l1855_185558

theorem ratio_of_segments_of_hypotenuse
  (a b c r s : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_ratio : a / b = 2 / 5)
  (h_r : r = (a^2) / c) 
  (h_s : s = (b^2) / c) : 
  r / s = 4 / 25 := sorry

end NUMINAMATH_GPT_ratio_of_segments_of_hypotenuse_l1855_185558


namespace NUMINAMATH_GPT_average_spring_headcount_average_fall_headcount_l1855_185529

namespace AverageHeadcount

def springHeadcounts := [10900, 10500, 10700, 11300]
def fallHeadcounts := [11700, 11500, 11600, 11300]

def averageHeadcount (counts : List ℕ) : ℕ :=
  counts.sum / counts.length

theorem average_spring_headcount :
  averageHeadcount springHeadcounts = 10850 := by
  sorry

theorem average_fall_headcount :
  averageHeadcount fallHeadcounts = 11525 := by
  sorry

end AverageHeadcount

end NUMINAMATH_GPT_average_spring_headcount_average_fall_headcount_l1855_185529


namespace NUMINAMATH_GPT_first_pump_time_l1855_185560

-- Definitions for the conditions provided
def newer_model_rate := 1 / 6
def combined_rate := 1 / 3.6
def time_for_first_pump : ℝ := 9

-- The theorem to be proven
theorem first_pump_time (T : ℝ) (h1 : 1 / 6 + 1 / T = 1 / 3.6) : T = 9 :=
sorry

end NUMINAMATH_GPT_first_pump_time_l1855_185560


namespace NUMINAMATH_GPT_find_multiple_l1855_185540

-- Definitions of the divisor, original number, and remainders given in the problem conditions.
def D : ℕ := 367
def remainder₁ : ℕ := 241
def remainder₂ : ℕ := 115

-- Statement of the problem.
theorem find_multiple (N m k l : ℕ) :
  (N = k * D + remainder₁) →
  (m * N = l * D + remainder₂) →
  ∃ m, m > 0 ∧ 241 * m - 115 % 367 = 0 ∧ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l1855_185540


namespace NUMINAMATH_GPT_ab_equals_6_l1855_185594

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end NUMINAMATH_GPT_ab_equals_6_l1855_185594


namespace NUMINAMATH_GPT_common_difference_arithmetic_sequence_l1855_185509

theorem common_difference_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 5 = 10) (h2 : a 12 = 31) : d = 3 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_arithmetic_sequence_l1855_185509


namespace NUMINAMATH_GPT_difference_of_squares_l1855_185537

theorem difference_of_squares (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 16) : x^2 - y^2 = 960 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1855_185537


namespace NUMINAMATH_GPT_value_of_n_l1855_185519

theorem value_of_n (n : ℕ) (h1 : 0 < n) (h2 : n < Real.sqrt 65) (h3 : Real.sqrt 65 < n + 1) : n = 8 := 
sorry

end NUMINAMATH_GPT_value_of_n_l1855_185519


namespace NUMINAMATH_GPT_largest_possible_radius_tangent_circle_l1855_185549

theorem largest_possible_radius_tangent_circle :
  ∃ (r : ℝ), 0 < r ∧
    (∀ x y, (x - r)^2 + (y - r)^2 = r^2 → 
    ((x = 9 ∧ y = 2) → (r = 17))) :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_radius_tangent_circle_l1855_185549


namespace NUMINAMATH_GPT_calculation_of_expression_l1855_185555

theorem calculation_of_expression
  (w x y z : ℕ)
  (h : 2^w * 3^x * 5^y * 7^z = 13230) :
  3 * w + 2 * x + 6 * y + 4 * z = 23 :=
sorry

end NUMINAMATH_GPT_calculation_of_expression_l1855_185555


namespace NUMINAMATH_GPT_max_least_integer_l1855_185568

theorem max_least_integer (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 2160) (h_order : x ≤ y ∧ y ≤ z) : x ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_max_least_integer_l1855_185568


namespace NUMINAMATH_GPT_range_of_a3_l1855_185516

open Real

def convex_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, (a n + a (n + 2)) / 2 ≤ a (n + 1)

def sequence_condition (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, 1 ≤ n → n < 10 → abs (a n - b n) ≤ 20

def b (n : ℕ) : ℝ := n^2 - 6 * n + 10

theorem range_of_a3 (a : ℕ → ℝ) :
  convex_sequence a →
  a 1 = 1 →
  a 10 = 28 →
  sequence_condition a b →
  7 ≤ a 3 ∧ a 3 ≤ 19 :=
sorry

end NUMINAMATH_GPT_range_of_a3_l1855_185516


namespace NUMINAMATH_GPT_speed_of_man_correct_l1855_185591

noncomputable def speed_of_man_in_kmph (train_speed_kmph : ℝ) (train_length_m : ℝ) (time_pass_sec : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let relative_speed_mps := (train_length_m / time_pass_sec)
  let man_speed_mps := relative_speed_mps - train_speed_mps
  man_speed_mps * 3600 / 1000

theorem speed_of_man_correct : 
  speed_of_man_in_kmph 77.993280537557 140 6 = 6.00871946444388 := 
by simp [speed_of_man_in_kmph]; sorry

end NUMINAMATH_GPT_speed_of_man_correct_l1855_185591


namespace NUMINAMATH_GPT_population_of_village_l1855_185522

-- Define the given condition
def total_population (P : ℝ) : Prop :=
  0.4 * P = 23040

-- The theorem to prove that the total population is 57600
theorem population_of_village : ∃ P : ℝ, total_population P ∧ P = 57600 :=
by
  sorry

end NUMINAMATH_GPT_population_of_village_l1855_185522


namespace NUMINAMATH_GPT_integer_solutions_eq_l1855_185595

theorem integer_solutions_eq (x y z : ℤ) :
  (x + y + z) ^ 5 = 80 * x * y * z * (x ^ 2 + y ^ 2 + z ^ 2) ↔
  ∃ a : ℤ, (x = a ∧ y = -a ∧ z = 0) ∨ (x = -a ∧ y = a ∧ z = 0) ∨ (x = a ∧ y = 0 ∧ z = -a) ∨ (x = -a ∧ y = 0 ∧ z = a) ∨ (x = 0 ∧ y = a ∧ z = -a) ∨ (x = 0 ∧ y = -a ∧ z = a) :=
by sorry

end NUMINAMATH_GPT_integer_solutions_eq_l1855_185595


namespace NUMINAMATH_GPT_visitors_not_ill_l1855_185551

theorem visitors_not_ill (total_visitors : ℕ) (ill_percentage : ℕ) (fall_ill : ℕ) : 
  total_visitors = 500 → 
  ill_percentage = 40 → 
  fall_ill = (ill_percentage * total_visitors) / 100 →
  total_visitors - fall_ill = 300 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_visitors_not_ill_l1855_185551


namespace NUMINAMATH_GPT_cost_of_fencing_per_meter_l1855_185570

theorem cost_of_fencing_per_meter
  (length : ℕ) (breadth : ℕ) (total_cost : ℝ) (cost_per_meter : ℝ)
  (h1 : length = 64) 
  (h2 : length = breadth + 28)
  (h3 : total_cost = 5300)
  (h4 : cost_per_meter = total_cost / (2 * (length + breadth))) :
  cost_per_meter = 26.50 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_of_fencing_per_meter_l1855_185570


namespace NUMINAMATH_GPT_intersection_points_number_of_regions_l1855_185567

-- Given n lines on a plane, any two of which are not parallel
-- and no three of which intersect at the same point,
-- prove the number of intersection points of these lines

theorem intersection_points (n : ℕ) (h_n : 0 < n) : 
  ∃ a_n : ℕ, a_n = n * (n - 1) / 2 := by
  sorry

-- Given n lines on a plane, any two of which are not parallel
-- and no three of which intersect at the same point,
-- prove the number of regions these lines form

theorem number_of_regions (n : ℕ) (h_n : 0 < n) :
  ∃ R_n : ℕ, R_n = n * (n + 1) / 2 + 1 := by
  sorry

end NUMINAMATH_GPT_intersection_points_number_of_regions_l1855_185567


namespace NUMINAMATH_GPT_jason_bought_correct_dozens_l1855_185506

-- Given conditions
def cupcakes_per_cousin : Nat := 3
def cousins : Nat := 16
def cupcakes_per_dozen : Nat := 12

-- Calculated value
def total_cupcakes : Nat := cupcakes_per_cousin * cousins
def dozens_of_cupcakes_bought : Nat := total_cupcakes / cupcakes_per_dozen

-- Theorem statement
theorem jason_bought_correct_dozens : dozens_of_cupcakes_bought = 4 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_jason_bought_correct_dozens_l1855_185506


namespace NUMINAMATH_GPT_value_of_expression_l1855_185521

theorem value_of_expression (b : ℚ) (h : b = 1/3) : (3 * b⁻¹ + (b⁻¹ / 3)) / b = 30 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_value_of_expression_l1855_185521


namespace NUMINAMATH_GPT_speed_of_train_l1855_185534

open Real

-- Define the conditions as given in the problem
def length_of_bridge : ℝ := 650
def length_of_train : ℝ := 200
def time_to_pass_bridge : ℝ := 17

-- Define the problem statement which needs to be proved
theorem speed_of_train : (length_of_bridge + length_of_train) / time_to_pass_bridge = 50 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_train_l1855_185534


namespace NUMINAMATH_GPT_find_b_value_l1855_185590

theorem find_b_value (a b c : ℝ)
  (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
  (h2 : 6 * b * 7 = 1.5) : b = 15 := by
  sorry

end NUMINAMATH_GPT_find_b_value_l1855_185590


namespace NUMINAMATH_GPT_seashells_left_l1855_185579

theorem seashells_left (initial_seashells : ℕ) (given_seashells : ℕ) (remaining_seashells : ℕ) :
  initial_seashells = 75 → given_seashells = 18 → remaining_seashells = initial_seashells - given_seashells → remaining_seashells = 57 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_seashells_left_l1855_185579


namespace NUMINAMATH_GPT_computer_price_difference_l1855_185563

-- Define the conditions as stated
def basic_computer_price := 1500
def total_price := 2500
def printer_price (P : ℕ) := basic_computer_price + P = total_price

def enhanced_computer_price (P E : ℕ) := P = (E + P) / 3

-- The theorem stating the proof problem
theorem computer_price_difference (P E : ℕ) 
  (h1 : printer_price P) 
  (h2 : enhanced_computer_price P E) : E - basic_computer_price = 500 :=
sorry

end NUMINAMATH_GPT_computer_price_difference_l1855_185563


namespace NUMINAMATH_GPT_PQ_value_l1855_185544

theorem PQ_value (DE DF EF : ℕ) (CF : ℝ) (P Q : ℝ) 
  (h1 : DE = 996)
  (h2 : DF = 995)
  (h3 : EF = 994)
  (hCF :  CF = (995^2 - 4) / 1990)
  (hP : P = (1492.5 - EF))
  (hQ : Q = (s - DF)) :
  PQ = 1 ∧ m + n = 2 :=
by
  sorry

end NUMINAMATH_GPT_PQ_value_l1855_185544


namespace NUMINAMATH_GPT_power_function_half_l1855_185546

theorem power_function_half (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ (1/2)) (hx : f 4 = 2) : 
  f (1/2) = (Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_GPT_power_function_half_l1855_185546


namespace NUMINAMATH_GPT_gold_bars_per_row_l1855_185512

theorem gold_bars_per_row 
  (total_worth : ℝ)
  (total_rows : ℕ)
  (value_per_bar : ℝ)
  (h_total_worth : total_worth = 1600000)
  (h_total_rows : total_rows = 4)
  (h_value_per_bar : value_per_bar = 40000) :
  total_worth / value_per_bar / total_rows = 10 :=
by
  sorry

end NUMINAMATH_GPT_gold_bars_per_row_l1855_185512


namespace NUMINAMATH_GPT_range_of_a_l1855_185577

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then (x - a) ^ 2 + Real.exp 1 else x / Real.log x + a + 10

theorem range_of_a (a : ℝ) :
    (∀ x, f x a ≥ f 2 a) → (2 ≤ a ∧ a ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1855_185577


namespace NUMINAMATH_GPT_divides_14_pow_n_minus_27_for_all_natural_numbers_l1855_185515

theorem divides_14_pow_n_minus_27_for_all_natural_numbers :
  ∀ n : ℕ, 13 ∣ 14^n - 27 :=
by sorry

end NUMINAMATH_GPT_divides_14_pow_n_minus_27_for_all_natural_numbers_l1855_185515


namespace NUMINAMATH_GPT_ratio_of_segments_intersecting_chords_l1855_185539

open Real

variables (EQ FQ HQ GQ : ℝ)

theorem ratio_of_segments_intersecting_chords 
  (h1 : EQ = 5) 
  (h2 : GQ = 7) 
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_segments_intersecting_chords_l1855_185539


namespace NUMINAMATH_GPT_largest_n_l1855_185504

theorem largest_n : ∃ (n : ℕ), n < 1000 ∧ (∃ (m : ℕ), lcm m n = 3 * m * gcd m n) ∧ (∀ k, k < 1000 ∧ (∃ (m' : ℕ), lcm m' k = 3 * m' * gcd m' k) → k ≤ 972) := sorry

end NUMINAMATH_GPT_largest_n_l1855_185504


namespace NUMINAMATH_GPT_xy_value_l1855_185571

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 15) : x * y = 15 :=
by
  sorry

end NUMINAMATH_GPT_xy_value_l1855_185571


namespace NUMINAMATH_GPT_four_ping_pong_four_shuttlecocks_cost_l1855_185550

theorem four_ping_pong_four_shuttlecocks_cost
  (x y : ℝ)
  (h1 : 3 * x + 2 * y = 15.5)
  (h2 : 2 * x + 3 * y = 17) :
  4 * x + 4 * y = 26 :=
sorry

end NUMINAMATH_GPT_four_ping_pong_four_shuttlecocks_cost_l1855_185550


namespace NUMINAMATH_GPT_proposition_check_l1855_185500

variable (P : ℕ → Prop)

theorem proposition_check 
  (h : ∀ k : ℕ, ¬ P (k + 1) → ¬ P k)
  (h2012 : P 2012) : P 2013 :=
by
  sorry

end NUMINAMATH_GPT_proposition_check_l1855_185500
