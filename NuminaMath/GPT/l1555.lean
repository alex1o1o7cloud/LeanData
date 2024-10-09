import Mathlib

namespace solve_equation_l1555_155513

theorem solve_equation (x : ℝ) (h : (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2) : 
  x = -2/3 :=
sorry

end solve_equation_l1555_155513


namespace max_marks_400_l1555_155589

theorem max_marks_400 {M : ℝ} (h1 : 0.35 * M = 140) : M = 400 :=
by 
-- skipping the proof using sorry
sorry

end max_marks_400_l1555_155589


namespace green_flower_percentage_l1555_155523

theorem green_flower_percentage (yellow purple green total : ℕ)
  (hy : yellow = 10)
  (hp : purple = 18)
  (ht : total = 35)
  (hgreen : green = total - (yellow + purple)) :
  ((green * 100) / (yellow + purple)) = 25 := 
by {
  sorry
}

end green_flower_percentage_l1555_155523


namespace correct_option_is_C_l1555_155549

-- Definitions based on the problem conditions
def option_A : Prop := (-3 + (-3)) = 0
def option_B : Prop := (-3 - abs (-3)) = 0
def option_C (a b : ℝ) : Prop := (3 * a^2 * b - 4 * b * a^2) = - a^2 * b
def option_D (x : ℝ) : Prop := (-(5 * x - 2)) = -5 * x - 2

-- The theorem to be proved that option C is the correct calculation
theorem correct_option_is_C (a b : ℝ) : option_C a b :=
sorry

end correct_option_is_C_l1555_155549


namespace zoe_correct_percentage_l1555_155517

variable (t : ℝ) -- total number of problems

-- Conditions
variable (chloe_solved_fraction : ℝ := 0.60)
variable (zoe_solved_fraction : ℝ := 0.40)
variable (chloe_correct_percentage_alone : ℝ := 0.75)
variable (chloe_correct_percentage_total : ℝ := 0.85)
variable (zoe_correct_percentage_alone : ℝ := 0.95)

theorem zoe_correct_percentage (h1 : chloe_solved_fraction = 0.60)
                               (h2 : zoe_solved_fraction = 0.40)
                               (h3 : chloe_correct_percentage_alone = 0.75)
                               (h4 : chloe_correct_percentage_total = 0.85)
                               (h5 : zoe_correct_percentage_alone = 0.95) :
  (zoe_correct_percentage_alone * zoe_solved_fraction * 100 + (chloe_correct_percentage_total - chloe_correct_percentage_alone * chloe_solved_fraction) * 100 = 78) :=
sorry

end zoe_correct_percentage_l1555_155517


namespace smallest_value_of_f4_l1555_155543

def f (x : ℝ) : ℝ := (x + 3) ^ 2 - 2

theorem smallest_value_of_f4 : ∀ x : ℝ, f (f (f (f x))) ≥ 23 :=
by 
  sorry -- Proof goes here.

end smallest_value_of_f4_l1555_155543


namespace exponential_comparisons_l1555_155506

open Real

noncomputable def a : ℝ := 5 ^ (log 3.4 / log 2)
noncomputable def b : ℝ := 5 ^ (log 3.6 / (log 4))
noncomputable def c : ℝ := 5 ^ (log (10 / 3))

theorem exponential_comparisons :
  a > c ∧ c > b := by
  sorry

end exponential_comparisons_l1555_155506


namespace solve_system_eq_l1555_155571

theorem solve_system_eq (x y z : ℝ) 
  (h1 : x * y = 6 * (x + y))
  (h2 : x * z = 4 * (x + z))
  (h3 : y * z = 2 * (y + z)) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = -24 ∧ y = 24 / 5 ∧ z = 24 / 7) :=
  sorry

end solve_system_eq_l1555_155571


namespace second_spray_kill_percent_l1555_155577

-- Conditions
def first_spray_kill_percent : ℝ := 50
def both_spray_kill_percent : ℝ := 5
def germs_left_after_both : ℝ := 30

-- Lean 4 statement
theorem second_spray_kill_percent (x : ℝ) 
  (H : 100 - (first_spray_kill_percent + x - both_spray_kill_percent) = germs_left_after_both) :
  x = 15 :=
by
  sorry

end second_spray_kill_percent_l1555_155577


namespace quadratic_complete_square_l1555_155525

theorem quadratic_complete_square :
  ∃ a b c : ℝ, (∀ x : ℝ, 4 * x^2 - 40 * x + 100 = a * (x + b)^2 + c) ∧ a + b + c = -1 :=
sorry

end quadratic_complete_square_l1555_155525


namespace sequence_formula_l1555_155516

theorem sequence_formula (a : ℕ → ℤ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, a n - a (n + 1) + 2 = 0) :
  ∀ n : ℕ, a n = 2 * n - 1 := 
sorry

end sequence_formula_l1555_155516


namespace average_of_abc_l1555_155562

theorem average_of_abc (A B C : ℚ) 
  (h1 : 2002 * C + 4004 * A = 8008) 
  (h2 : 3003 * B - 5005 * A = 7007) : 
  (A + B + C) / 3 = 22 / 9 := 
by 
  sorry

end average_of_abc_l1555_155562


namespace ellipse_semimajor_axis_value_l1555_155530

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end ellipse_semimajor_axis_value_l1555_155530


namespace pipe_cistern_l1555_155518

theorem pipe_cistern (rate: ℚ) (duration: ℚ) (portion: ℚ) : 
  rate = (2/3) / 10 → duration = 8 → portion = 8/15 →
  portion = duration * rate := 
by 
  intros h1 h2 h3
  sorry

end pipe_cistern_l1555_155518


namespace man_l1555_155512

noncomputable def man's_rate_in_still_water (downstream upstream : ℝ) : ℝ :=
  (downstream + upstream) / 2

theorem man's_rate_correct :
  let downstream := 6
  let upstream := 3
  man's_rate_in_still_water downstream upstream = 4.5 :=
by
  sorry

end man_l1555_155512


namespace xyz_value_l1555_155510

variable {x y z : ℝ}

theorem xyz_value (h1 : (x + y + z) * (x * y + x * z + y * z) = 18) 
                  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 6) : 
                  x * y * z = 4 := 
by
  sorry

end xyz_value_l1555_155510


namespace problem_statement_l1555_155550

theorem problem_statement (n : ℕ) (hn : n > 0) : (122 ^ n - 102 ^ n - 21 ^ n) % 2020 = 2019 :=
by
  sorry

end problem_statement_l1555_155550


namespace cost_of_six_burritos_and_seven_sandwiches_l1555_155536

variable (b s : ℝ)
variable (h1 : 4 * b + 2 * s = 5.00)
variable (h2 : 3 * b + 5 * s = 6.50)

theorem cost_of_six_burritos_and_seven_sandwiches : 6 * b + 7 * s = 11.50 :=
  sorry

end cost_of_six_burritos_and_seven_sandwiches_l1555_155536


namespace fraction_subtraction_simplest_form_l1555_155534

theorem fraction_subtraction_simplest_form :
  (8 / 24 - 5 / 40 = 5 / 24) :=
by
  sorry

end fraction_subtraction_simplest_form_l1555_155534


namespace point_P_x_coordinate_l1555_155547

variable {P : Type} [LinearOrderedField P]

-- Definitions from the conditions
def line_equation (x : P) : P := 0.8 * x
def y_coordinate_P : P := 6
def x_coordinate_P : P := 7.5

-- Theorems to prove that the x-coordinate of P is 7.5.
theorem point_P_x_coordinate (x : P) :
  line_equation x = y_coordinate_P → x = x_coordinate_P :=
by
  intro h
  sorry

end point_P_x_coordinate_l1555_155547


namespace data_point_frequency_l1555_155548

theorem data_point_frequency 
  (data : Type) 
  (categories : data → Prop) 
  (group_counts : data → ℕ) :
  ∀ d, categories d → group_counts d = frequency := sorry

end data_point_frequency_l1555_155548


namespace jerrys_current_average_score_l1555_155565

theorem jerrys_current_average_score (A : ℝ) (h1 : 3 * A + 98 = 4 * (A + 2)) : A = 90 :=
by
  sorry

end jerrys_current_average_score_l1555_155565


namespace smallest_number_divisibility_l1555_155593

theorem smallest_number_divisibility :
  ∃ x, (x + 3) % 70 = 0 ∧ (x + 3) % 100 = 0 ∧ (x + 3) % 84 = 0 ∧ x = 6303 :=
sorry

end smallest_number_divisibility_l1555_155593


namespace intersection_is_3_l1555_155570

def setA : Set ℕ := {5, 2, 3}
def setB : Set ℕ := {9, 3, 6}

theorem intersection_is_3 : setA ∩ setB = {3} := by
  sorry

end intersection_is_3_l1555_155570


namespace zachary_pushups_l1555_155566

theorem zachary_pushups (david_pushups zachary_pushups : ℕ) (h₁ : david_pushups = 44) (h₂ : david_pushups = zachary_pushups + 9) :
  zachary_pushups = 35 :=
by
  sorry

end zachary_pushups_l1555_155566


namespace interval_of_decrease_l1555_155569

noncomputable def f : ℝ → ℝ := fun x => x^2 - 2 * x

theorem interval_of_decrease : 
  ∃ a b : ℝ, a = -2 ∧ b = 1 ∧ ∀ x1 x2 : ℝ, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 ≥ f x2 :=
by 
  use -2, 1
  sorry

end interval_of_decrease_l1555_155569


namespace intersection_complement_l1555_155529

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def B : Set ℝ := {x | 2 < x ∧ x < 4}
def complement_U (A : Set ℝ) : Set ℝ := {x | ¬ (A x)}

theorem intersection_complement (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) :
  B ∩ (complement_U A) = {x | 2 < x ∧ x ≤ 3} := by
  sorry

end intersection_complement_l1555_155529


namespace prime_divides_sum_l1555_155563

theorem prime_divides_sum 
  (a b c : ℕ) 
  (h1 : a^3 + 4 * b + c = a * b * c)
  (h2 : a ≥ c)
  (h3 : Prime (a^2 + 2 * a + 2)) : 
  (a^2 + 2 * a + 2) ∣ (a + 2 * b + 2) := 
sorry

end prime_divides_sum_l1555_155563


namespace f_equality_2019_l1555_155541

theorem f_equality_2019 (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), f (m + n) ≥ f m + f (f n) - 1) : 
  f 2019 = 2019 :=
sorry

end f_equality_2019_l1555_155541


namespace survey_preference_l1555_155584

theorem survey_preference (X Y : ℕ) 
  (ratio_condition : X / Y = 5)
  (total_respondents : X + Y = 180) :
  X = 150 := 
sorry

end survey_preference_l1555_155584


namespace total_commute_time_l1555_155503

theorem total_commute_time 
  (first_bus : ℕ) (delay1 : ℕ) (wait1 : ℕ) 
  (second_bus : ℕ) (delay2 : ℕ) (wait2 : ℕ) 
  (third_bus : ℕ) (delay3 : ℕ) 
  (arrival_time : ℕ) :
  first_bus = 40 →
  delay1 = 10 →
  wait1 = 10 →
  second_bus = 50 →
  delay2 = 5 →
  wait2 = 15 →
  third_bus = 95 →
  delay3 = 15 →
  arrival_time = 540 →
  first_bus + delay1 + wait1 + second_bus + delay2 + wait2 + third_bus + delay3 = 240 :=
by
  intros
  sorry

end total_commute_time_l1555_155503


namespace proposition_3_proposition_4_l1555_155544

variables {Plane : Type} {Line : Type} 
variables {α β : Plane} {a b : Line}

-- Assuming necessary properties of parallel planes and lines being subsets of planes
axiom plane_parallel (α β : Plane) : Prop
axiom line_in_plane (l : Line) (p : Plane) : Prop
axiom line_parallel (l m : Line) : Prop
axiom lines_skew (l m : Line) : Prop
axiom lines_coplanar (l m : Line) : Prop
axiom lines_do_not_intersect (l m : Line) : Prop

-- Assume the given conditions
variables (h1 : plane_parallel α β) 
variables (h2 : line_in_plane a α)
variables (h3 : line_in_plane b β)

-- State the equivalent proof problem as propositions to be proved in Lean
theorem proposition_3 (h1 : plane_parallel α β) 
                     (h2 : line_in_plane a α) 
                     (h3 : line_in_plane b β) : 
                     lines_do_not_intersect a b :=
sorry

theorem proposition_4 (h1 : plane_parallel α β) 
                     (h2 : line_in_plane a α) 
                     (h3 : line_in_plane b β) : 
                     lines_coplanar a b ∨ lines_skew a b :=
sorry

end proposition_3_proposition_4_l1555_155544


namespace product_consecutive_natural_number_square_l1555_155560

theorem product_consecutive_natural_number_square (n : ℕ) : 
  ∃ k : ℕ, 100 * (n^2 + n) + 25 = k^2 :=
by
  sorry

end product_consecutive_natural_number_square_l1555_155560


namespace Jill_llamas_count_l1555_155537

theorem Jill_llamas_count :
  let initial_pregnant_with_one_calf := 9
  let initial_pregnant_with_twins := 5
  let total_calves_born := (initial_pregnant_with_one_calf * 1) + (initial_pregnant_with_twins * 2)
  let calves_after_trade := total_calves_born - 8
  let initial_pregnant_lamas := initial_pregnant_with_one_calf + initial_pregnant_with_twins
  let total_lamas_after_birth := initial_pregnant_lamas + total_calves_born
  let lamas_after_trade := total_lamas_after_birth - 8 + 2
  let lamas_sold := lamas_after_trade / 3
  let final_lamas := lamas_after_trade - lamas_sold
  final_lamas = 18 :=
by
  sorry

end Jill_llamas_count_l1555_155537


namespace math_problem_l1555_155582

variable {x p q r : ℝ}

-- Conditions and Theorem
theorem math_problem (h1 : ∀ x, (x ≤ -5 ∨ 20 ≤ x ∧ x ≤ 30) ↔ (0 ≤ (x - p) * (x - q) / (x - r)))
  (h2 : p < q) : p + 2 * q + 3 * r = 65 := 
sorry

end math_problem_l1555_155582


namespace volume_ratio_sphere_cylinder_inscribed_l1555_155502

noncomputable def ratio_of_volumes (d : ℝ) : ℝ :=
  let Vs := (4 / 3) * Real.pi * (d / 2)^3
  let Vc := Real.pi * (d / 2)^2 * d
  Vs / Vc

theorem volume_ratio_sphere_cylinder_inscribed (d : ℝ) (h : d > 0) : 
  ratio_of_volumes d = 2 / 3 := 
by
  sorry

end volume_ratio_sphere_cylinder_inscribed_l1555_155502


namespace power_greater_than_linear_l1555_155586

theorem power_greater_than_linear (n : ℕ) (h : n ≥ 3) : 2^n > 2 * n + 1 := 
by {
  sorry
}

end power_greater_than_linear_l1555_155586


namespace broken_stick_triangle_probability_l1555_155557

noncomputable def probability_of_triangle (x y z : ℕ) : ℚ := sorry

theorem broken_stick_triangle_probability :
  ∀ x y z : ℕ, (x < y + z ∧ y < x + z ∧ z < x + y) → probability_of_triangle x y z = 1 / 4 := 
by
  sorry

end broken_stick_triangle_probability_l1555_155557


namespace diane_harvest_increase_l1555_155539

-- Define the conditions
def last_year_harvest : ℕ := 2479
def this_year_harvest : ℕ := 8564

-- Definition of the increase in honey harvest
def increase_in_harvest : ℕ := this_year_harvest - last_year_harvest

-- The theorem statement we need to prove
theorem diane_harvest_increase : increase_in_harvest = 6085 := 
by
  -- skip the proof for now
  sorry

end diane_harvest_increase_l1555_155539


namespace reporters_cover_local_politics_l1555_155528

structure Reporters :=
(total : ℕ)
(politics : ℕ)
(local_politics : ℕ)

def percentages (reporters : Reporters) : Prop :=
  reporters.politics = (40 * reporters.total) / 100 ∧
  reporters.local_politics = (75 * reporters.politics) / 100

theorem reporters_cover_local_politics (reporters : Reporters) (h : percentages reporters) :
  (reporters.local_politics * 100) / reporters.total = 30 :=
by
  -- Proof steps would be added here
  sorry

end reporters_cover_local_politics_l1555_155528


namespace profit_is_55_l1555_155555

-- Define the given conditions:
def cost_of_chocolates (bars: ℕ) (price_per_bar: ℕ) : ℕ :=
  bars * price_per_bar

def cost_of_packaging (bars: ℕ) (cost_per_bar: ℕ) : ℕ :=
  bars * cost_per_bar

def total_sales : ℕ :=
  90

def total_cost (cost_of_chocolates cost_of_packaging: ℕ) : ℕ :=
  cost_of_chocolates + cost_of_packaging

def profit (total_sales total_cost: ℕ) : ℕ :=
  total_sales - total_cost

-- Given values:
def bars: ℕ := 5
def price_per_bar: ℕ := 5
def cost_per_packaging_bar: ℕ := 2

-- Define the profit calculation theorem:
theorem profit_is_55 : 
  profit total_sales (total_cost (cost_of_chocolates bars price_per_bar) (cost_of_packaging bars cost_per_packaging_bar)) = 55 :=
by {
  -- The proof will be inserted here
  sorry
}

end profit_is_55_l1555_155555


namespace points_total_l1555_155553

/--
In a game, Samanta has 8 more points than Mark,
and Mark has 50% more points than Eric. Eric has 6 points.
How many points do Samanta, Mark, and Eric have in total?
-/
theorem points_total (Samanta Mark Eric : ℕ)
  (h1 : Samanta = Mark + 8)
  (h2 : Mark = Eric + Eric / 2)
  (h3 : Eric = 6) :
  Samanta + Mark + Eric = 32 := by
  sorry

end points_total_l1555_155553


namespace smallest_12_digit_proof_l1555_155575

def is_12_digit_number (n : ℕ) : Prop :=
  n >= 10^11 ∧ n < 10^12

def contains_each_digit_0_to_9 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] → d ∈ n.digits 10

def is_divisible_by_36 (n : ℕ) : Prop :=
  n % 36 = 0

noncomputable def smallest_12_digit_divisible_by_36_and_contains_each_digit : ℕ :=
  100023457896

theorem smallest_12_digit_proof :
  is_12_digit_number smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  contains_each_digit_0_to_9 smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  is_divisible_by_36 smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  ∀ m : ℕ, is_12_digit_number m ∧ contains_each_digit_0_to_9 m ∧ is_divisible_by_36 m →
  m >= smallest_12_digit_divisible_by_36_and_contains_each_digit :=
by
  sorry

end smallest_12_digit_proof_l1555_155575


namespace part_a_part_b_l1555_155551

def triangle := Type
def point := Type

structure TriangleInCircle (ABC : triangle) where
  A : point
  B : point
  C : point
  A1 : point
  B1 : point
  C1 : point
  M : point
  r : Real
  R : Real

theorem part_a (ABC : triangle) (t : TriangleInCircle ABC) :
  ∃ MA MC MB_1, (MA * MC) / MB_1 = 2 * t.r := sorry
  
theorem part_b (ABC : triangle) (t : TriangleInCircle ABC) :
  ∃ MA_1 MC_1 MB, ( (MA_1 * MC_1) / MB) = t.R := sorry

end part_a_part_b_l1555_155551


namespace isosceles_triangle_base_function_l1555_155533

theorem isosceles_triangle_base_function (x : ℝ) (hx : 5 < x ∧ x < 10) :
  ∃ y : ℝ, y = 20 - 2 * x := 
by
  sorry

end isosceles_triangle_base_function_l1555_155533


namespace fraction_tips_l1555_155501

theorem fraction_tips {S : ℝ} (H1 : S > 0) (H2 : tips = (7 / 3 : ℝ) * S) (H3 : bonuses = (2 / 5 : ℝ) * S) :
  (tips / (S + tips + bonuses)) = (5 / 8 : ℝ) :=
by
  sorry

end fraction_tips_l1555_155501


namespace square_of_real_not_always_positive_l1555_155527

theorem square_of_real_not_always_positive (a : ℝ) : ¬(a^2 > 0) := 
sorry

end square_of_real_not_always_positive_l1555_155527


namespace candy_from_sister_l1555_155546

variable (f : ℕ) (e : ℕ) (t : ℕ)

theorem candy_from_sister (h₁ : f = 47) (h₂ : e = 25) (h₃ : t = 62) :
  ∃ x : ℕ, x = t - (f - e) ∧ x = 40 :=
by sorry

end candy_from_sister_l1555_155546


namespace seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums_l1555_155576

theorem seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums
  (a1 a2 a3 a4 a5 a6 a7 : Nat) :
  ¬ ∃ (s : Finset Nat), (s = {a1 + a2, a1 + a3, a1 + a4, a1 + a5, a1 + a6, a1 + a7,
                             a2 + a3, a2 + a4, a2 + a5, a2 + a6, a2 + a7,
                             a3 + a4, a3 + a5, a3 + a6, a3 + a7,
                             a4 + a5, a4 + a6, a4 + a7,
                             a5 + a6, a5 + a7,
                             a6 + a7}) ∧
  (∃ (n : Nat), s = {n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8, n+9}) := 
sorry

end seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums_l1555_155576


namespace bottles_remaining_after_2_days_l1555_155594

def total_bottles := 48 

def first_day_father_consumption := total_bottles / 4
def first_day_mother_consumption := total_bottles / 6
def first_day_son_consumption := total_bottles / 8

def total_first_day_consumption := first_day_father_consumption + first_day_mother_consumption + first_day_son_consumption 
def remaining_after_first_day := total_bottles - total_first_day_consumption

def second_day_father_consumption := remaining_after_first_day / 5
def remaining_after_father := remaining_after_first_day - second_day_father_consumption
def second_day_mother_consumption := remaining_after_father / 7
def remaining_after_mother := remaining_after_father - second_day_mother_consumption
def second_day_son_consumption := remaining_after_mother / 9
def remaining_after_son := remaining_after_mother - second_day_son_consumption
def second_day_daughter_consumption := remaining_after_son / 9
def remaining_after_daughter := remaining_after_son - second_day_daughter_consumption

theorem bottles_remaining_after_2_days : ∀ (total_bottles : ℕ), remaining_after_daughter = 14 := 
by
  sorry

end bottles_remaining_after_2_days_l1555_155594


namespace symmetric_function_is_periodic_l1555_155573

theorem symmetric_function_is_periodic {f : ℝ → ℝ} {a b y0 : ℝ}
  (h1 : ∀ x, f (a + x) - y0 = y0 - f (a - x))
  (h2 : ∀ x, f (b + x) = f (b - x))
  (hb : b > a) :
  ∀ x, f (x + 4 * (b - a)) = f x := sorry

end symmetric_function_is_periodic_l1555_155573


namespace smallest_positive_z_l1555_155590

open Real

-- Definitions for the conditions
def sin_zero_condition (x : ℝ) : Prop := sin x = 0
def sin_half_condition (x z : ℝ) : Prop := sin (x + z) = 1 / 2

-- Theorem for the proof objective
theorem smallest_positive_z (x z : ℝ) (hx : sin_zero_condition x) (hz : sin_half_condition x z) : z = π / 6 := 
sorry

end smallest_positive_z_l1555_155590


namespace cylinder_area_ratio_l1555_155522

noncomputable def ratio_of_areas (r h : ℝ) (h_cond : 2 * r / h = h / (2 * Real.pi * r)) : ℝ :=
  let lateral_area := 2 * Real.pi * r * h
  let total_area := lateral_area + 2 * Real.pi * r * r
  lateral_area / total_area

theorem cylinder_area_ratio {r h : ℝ} (h_cond : 2 * r / h = h / (2 * Real.pi * r)) :
  ratio_of_areas r h h_cond = 2 * Real.sqrt Real.pi / (2 * Real.sqrt Real.pi + 1) := 
sorry

end cylinder_area_ratio_l1555_155522


namespace box_height_is_55_cm_l1555_155580

noncomputable def height_of_box 
  (ceiling_height_m : ℝ) 
  (light_fixture_below_ceiling_cm : ℝ) 
  (bob_height_m : ℝ) 
  (bob_reach_cm : ℝ) 
  : ℝ :=
  let ceiling_height_cm := ceiling_height_m * 100
  let bob_height_cm := bob_height_m * 100
  let light_fixture_from_floor := ceiling_height_cm - light_fixture_below_ceiling_cm
  let bob_total_reach := bob_height_cm + bob_reach_cm
  light_fixture_from_floor - bob_total_reach

-- Theorem statement
theorem box_height_is_55_cm 
  (ceiling_height_m : ℝ) 
  (light_fixture_below_ceiling_cm : ℝ) 
  (bob_height_m : ℝ) 
  (bob_reach_cm : ℝ) 
  (h : height_of_box ceiling_height_m light_fixture_below_ceiling_cm bob_height_m bob_reach_cm = 55) 
  : height_of_box 3 15 1.8 50 = 55 :=
by
  unfold height_of_box
  sorry

end box_height_is_55_cm_l1555_155580


namespace inequality_unique_solution_l1555_155578

theorem inequality_unique_solution (p : ℝ) :
  (∃ x : ℝ, 0 ≤ x^2 + p * x + 5 ∧ x^2 + p * x + 5 ≤ 1) →
  (∃ x : ℝ, x^2 + p * x + 4 = 0) → p = 4 ∨ p = -4 :=
sorry

end inequality_unique_solution_l1555_155578


namespace original_polygon_sides_l1555_155535

theorem original_polygon_sides {n : ℕ} 
  (h : (n - 2) * 180 = 1620) : n = 10 ∨ n = 11 ∨ n = 12 :=
sorry

end original_polygon_sides_l1555_155535


namespace cost_of_new_shoes_l1555_155559

theorem cost_of_new_shoes :
  ∃ P : ℝ, P = 32 ∧ (P / 2 = 14.50 + 0.10344827586206897 * 14.50) :=
sorry

end cost_of_new_shoes_l1555_155559


namespace f_is_odd_l1555_155505

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def f (x : ℝ) : ℝ := x * |x|

theorem f_is_odd : is_odd_function f :=
by sorry

end f_is_odd_l1555_155505


namespace total_sequins_correct_l1555_155521

def blue_rows : ℕ := 6
def blue_columns : ℕ := 8
def purple_rows : ℕ := 5
def purple_columns : ℕ := 12
def green_rows : ℕ := 9
def green_columns : ℕ := 6

def total_sequins : ℕ :=
  (blue_rows * blue_columns) + (purple_rows * purple_columns) + (green_rows * green_columns)

theorem total_sequins_correct : total_sequins = 162 := by
  sorry

end total_sequins_correct_l1555_155521


namespace find_number_l1555_155508

-- Definitions and conditions for the problem
def N_div_7 (N R_1 : ℕ) : ℕ := (N / 7) * 7 + R_1
def N_div_11 (N R_2 : ℕ) : ℕ := (N / 11) * 11 + R_2
def N_div_13 (N R_3 : ℕ) : ℕ := (N / 13) * 13 + R_3

theorem find_number 
  (N a b c R_1 R_2 R_3 : ℕ) 
  (hN7 : N = 7 * a + R_1)
  (hN11 : N = 11 * b + R_2)
  (hN13 : N = 13 * c + R_3)
  (hQ : a + b + c = 21)
  (hR : R_1 + R_2 + R_3 = 21)
  (hR1_lt : R_1 < 7)
  (hR2_lt : R_2 < 11)
  (hR3_lt : R_3 < 13) : 
  N = 74 :=
sorry

end find_number_l1555_155508


namespace abs_ineq_solution_set_l1555_155596

theorem abs_ineq_solution_set {x : ℝ} : (|x - 1| ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 3) :=
by
  sorry

end abs_ineq_solution_set_l1555_155596


namespace min_value_on_neg_infinite_l1555_155592

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def max_value_on_interval (F : ℝ → ℝ) (a b : ℝ) (max_val : ℝ) : Prop :=
∀ x, (0 < x → F x ≤ max_val) ∧ (∃ y, 0 < y ∧ F y = max_val)

theorem min_value_on_neg_infinite (f g : ℝ → ℝ) (a b : ℝ) (F : ℝ → ℝ)
  (h_odd_f : odd_function f) (h_odd_g : odd_function g)
  (h_def_F : ∀ x, F x = a * f x + b * g x + 2)
  (h_max_F_on_0_inf : max_value_on_interval F a b 8) :
  ∃ x, x < 0 ∧ F x = -4 :=
sorry

end min_value_on_neg_infinite_l1555_155592


namespace simple_interest_sum_l1555_155509

theorem simple_interest_sum (P_SI : ℕ) :
  let P_CI := 5000
  let r_CI := 12
  let t_CI := 2
  let r_SI := 10
  let t_SI := 5
  let CI := (P_CI * (1 + r_CI / 100)^t_CI - P_CI)
  let SI := CI / 2
  (P_SI * r_SI * t_SI / 100 = SI) -> 
  P_SI = 1272 := by {
  sorry
}

end simple_interest_sum_l1555_155509


namespace base_rate_of_second_company_l1555_155554

-- Define the conditions
def United_base_rate : ℝ := 8.00
def United_rate_per_minute : ℝ := 0.25
def Other_rate_per_minute : ℝ := 0.20
def minutes : ℕ := 80

-- Define the total bill equations
def United_total_bill (minutes : ℕ) : ℝ := United_base_rate + United_rate_per_minute * minutes
def Other_total_bill (minutes : ℕ) (B : ℝ) : ℝ := B + Other_rate_per_minute * minutes

-- Define the claim to prove
theorem base_rate_of_second_company : ∃ B : ℝ, Other_total_bill minutes B = United_total_bill minutes ∧ B = 12.00 := by
  sorry

end base_rate_of_second_company_l1555_155554


namespace maximal_N8_value_l1555_155598

noncomputable def max_permutations_of_projections (A : Fin 8 → ℝ × ℝ) : ℕ := sorry

theorem maximal_N8_value (A : Fin 8 → ℝ × ℝ) :
  max_permutations_of_projections A = 56 :=
sorry

end maximal_N8_value_l1555_155598


namespace find_fourth_month_sale_l1555_155520

theorem find_fourth_month_sale (s1 s2 s3 s4 s5 : ℕ) (avg_sale nL5 : ℕ)
  (h1 : s1 = 5420)
  (h2 : s2 = 5660)
  (h3 : s3 = 6200)
  (h5 : s5 = 6500)
  (havg : avg_sale = 6300)
  (hnL5 : nL5 = 5)
  (h_average : avg_sale * nL5 = s1 + s2 + s3 + s4 + s5) :
  s4 = 7720 := sorry

end find_fourth_month_sale_l1555_155520


namespace Sarah_collected_40_today_l1555_155595

noncomputable def Sarah_yesterday : ℕ := 50
noncomputable def Lara_yesterday : ℕ := Sarah_yesterday + 30
noncomputable def Lara_today : ℕ := 70
noncomputable def Total_yesterday : ℕ := Sarah_yesterday + Lara_yesterday
noncomputable def Total_today : ℕ := Total_yesterday - 20
noncomputable def Sarah_today : ℕ := Total_today - Lara_today

theorem Sarah_collected_40_today : Sarah_today = 40 := 
by
  sorry

end Sarah_collected_40_today_l1555_155595


namespace a719_divisible_by_11_l1555_155587

theorem a719_divisible_by_11 (a : ℕ) (h : a < 10) : (∃ k : ℤ, a - 15 = 11 * k) ↔ a = 4 :=
by
  sorry

end a719_divisible_by_11_l1555_155587


namespace find_difference_in_ticket_costs_l1555_155585

-- Conditions
def num_adults : ℕ := 9
def num_children : ℕ := 7
def cost_adult_ticket : ℕ := 11
def cost_child_ticket : ℕ := 7

def total_cost_adults : ℕ := num_adults * cost_adult_ticket
def total_cost_children : ℕ := num_children * cost_child_ticket
def total_tickets : ℕ := num_adults + num_children

-- Discount conditions (not needed for this proof since they don't apply)
def apply_discount (total_tickets : ℕ) (total_cost : ℕ) : ℕ :=
  if total_tickets >= 10 ∧ total_tickets <= 12 then
    total_cost * 9 / 10
  else if total_tickets >= 13 ∧ total_tickets <= 15 then
    total_cost * 85 / 100
  else
    total_cost

-- The main statement to prove
theorem find_difference_in_ticket_costs : total_cost_adults - total_cost_children = 50 := by
  sorry

end find_difference_in_ticket_costs_l1555_155585


namespace first_dilution_volume_l1555_155564

theorem first_dilution_volume (x : ℝ) (V : ℝ) (red_factor : ℝ) (p : ℝ) :
  V = 1000 →
  red_factor = 25 / 3 →
  (1000 - 2 * x) * (1000 - x) = 1000 * 1000 * (3 / 25) →
  x = 400 :=
by
  intros hV hred hf
  sorry

end first_dilution_volume_l1555_155564


namespace ribbon_left_after_wrapping_l1555_155568

def total_ribbon_needed (gifts : ℕ) (ribbon_per_gift : ℝ) : ℝ :=
  gifts * ribbon_per_gift

def remaining_ribbon (initial_ribbon : ℝ) (used_ribbon : ℝ) : ℝ :=
  initial_ribbon - used_ribbon

theorem ribbon_left_after_wrapping : 
  ∀ (gifts : ℕ) (ribbon_per_gift initial_ribbon : ℝ),
  gifts = 8 →
  ribbon_per_gift = 1.5 →
  initial_ribbon = 15 →
  remaining_ribbon initial_ribbon (total_ribbon_needed gifts ribbon_per_gift) = 3 :=
by
  intros gifts ribbon_per_gift initial_ribbon h1 h2 h3
  rw [h1, h2, h3]
  simp [total_ribbon_needed, remaining_ribbon]
  sorry

end ribbon_left_after_wrapping_l1555_155568


namespace acres_for_corn_l1555_155524

theorem acres_for_corn (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ)
  (total_ratio : beans_ratio + wheat_ratio + corn_ratio = 11)
  (land_parts : total_land / 11 = 94)
  : (corn_ratio = 4) → (total_land = 1034) → 4 * 94 = 376 :=
by
  intros
  sorry

end acres_for_corn_l1555_155524


namespace price_per_foot_l1555_155514

theorem price_per_foot (area : ℝ) (cost : ℝ) (side_length : ℝ) (perimeter : ℝ) 
  (h1 : area = 289) (h2 : cost = 3740) 
  (h3 : side_length^2 = area) (h4 : perimeter = 4 * side_length) : 
  (cost / perimeter = 55) :=
by
  sorry

end price_per_foot_l1555_155514


namespace english_teachers_count_l1555_155526

theorem english_teachers_count (E : ℕ) 
    (h_prob : 6 / ((E + 6) * (E + 5) / 2) = 1 / 12) : 
    E = 3 :=
by
  sorry

end english_teachers_count_l1555_155526


namespace successive_increases_eq_single_l1555_155504

variable (P : ℝ)

def increase_by (initial : ℝ) (pct : ℝ) : ℝ := initial * (1 + pct)
def discount_by (initial : ℝ) (pct : ℝ) : ℝ := initial * (1 - pct)

theorem successive_increases_eq_single (P : ℝ) :
  increase_by (increase_by (discount_by (increase_by P 0.30) 0.10) 0.15) 0.20 = increase_by P 0.6146 :=
  sorry

end successive_increases_eq_single_l1555_155504


namespace original_angle_measure_l1555_155572

theorem original_angle_measure : 
  ∃ x : ℝ, (90 - x) = 3 * x - 2 ∧ x = 23 :=
by
  sorry

end original_angle_measure_l1555_155572


namespace sqrt_diff_nat_l1555_155540

open Nat

theorem sqrt_diff_nat (a b : ℕ) (h : 2015 * a^2 + a = 2016 * b^2 + b) : ∃ k : ℕ, a - b = k^2 := 
by
  sorry

end sqrt_diff_nat_l1555_155540


namespace min_fuse_length_l1555_155511

theorem min_fuse_length 
  (safe_distance : ℝ := 70) 
  (personnel_speed : ℝ := 7) 
  (fuse_burning_speed : ℝ := 10.3) : 
  ∃ (x : ℝ), x ≥ 103 := 
by
  sorry

end min_fuse_length_l1555_155511


namespace initial_mat_weavers_l1555_155597

variable (num_weavers : ℕ) (rate : ℕ → ℕ → ℕ) -- rate weaver_count duration_in_days → mats_woven

-- Given Conditions
def condition1 := rate num_weavers 4 = 4
def condition2 := rate (2 * num_weavers) 8 = 16

-- Theorem to Prove
theorem initial_mat_weavers : num_weavers = 4 :=
by
  sorry

end initial_mat_weavers_l1555_155597


namespace intersection_range_l1555_155556

noncomputable def function_f (x: ℝ) : ℝ := abs (x^2 - 4 * x + 3)

theorem intersection_range (b : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ function_f x1 = b ∧ function_f x2 = b ∧ function_f x3 = b) ↔ (0 < b ∧ b ≤ 1) := 
sorry

end intersection_range_l1555_155556


namespace largest_value_l1555_155532

noncomputable def a : ℕ := 2 ^ 6
noncomputable def b : ℕ := 3 ^ 5
noncomputable def c : ℕ := 4 ^ 4
noncomputable def d : ℕ := 5 ^ 3
noncomputable def e : ℕ := 6 ^ 2

theorem largest_value : c > a ∧ c > b ∧ c > d ∧ c > e := by
  sorry

end largest_value_l1555_155532


namespace asymptotes_N_are_correct_l1555_155538

-- Given the conditions of the hyperbola M
def hyperbola_M (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / m - y^2 / 6 = 1

-- Eccentricity condition
def eccentricity (m : ℝ) (e : ℝ) : Prop :=
  e = 2 ∧ (m > 0)

-- Given hyperbola N
def hyperbola_N (x y : ℝ) (m : ℝ) : Prop :=
  x^2 - y^2 / m = 1

-- The theorem to be proved
theorem asymptotes_N_are_correct (m : ℝ) (x y : ℝ) :
  hyperbola_M x y 2 → eccentricity 2 2 → hyperbola_N x y m →
  (y = x * Real.sqrt 2 ∨ y = -x * Real.sqrt 2) :=
by
  sorry

end asymptotes_N_are_correct_l1555_155538


namespace max_handshakes_without_cycles_l1555_155581

open BigOperators

theorem max_handshakes_without_cycles :
  ∀ n : ℕ, n = 20 → ∑ i in Finset.range (n - 1), i = 190 :=
by intros;
   sorry

end max_handshakes_without_cycles_l1555_155581


namespace hexagon_equilateral_triangles_l1555_155588

theorem hexagon_equilateral_triangles (hexagon_area: ℝ) (num_hexagons : ℕ) (tri_area: ℝ) 
    (h1 : hexagon_area = 6) (h2 : num_hexagons = 4) (h3 : tri_area = 4) : 
    ∃ (num_triangles : ℕ), num_triangles = 8 := 
by
  sorry

end hexagon_equilateral_triangles_l1555_155588


namespace max_sum_of_squares_diff_l1555_155579

theorem max_sum_of_squares_diff {x y : ℕ} (h : x > 0 ∧ y > 0) (h_diff : x^2 - y^2 = 2016) :
  x + y ≤ 1008 ∧ ∃ x' y' : ℕ, x'^2 - y'^2 = 2016 ∧ x' + y' = 1008 :=
sorry

end max_sum_of_squares_diff_l1555_155579


namespace van_helsing_earnings_l1555_155567

theorem van_helsing_earnings (V W : ℕ) 
  (h1 : W = 4 * V) 
  (h2 : W = 8) :
  let E_v := 5 * (V / 2)
  let E_w := 10 * 8
  let E_total := E_v + E_w
  E_total = 85 :=
by
  sorry

end van_helsing_earnings_l1555_155567


namespace not_possible_to_form_triangle_l1555_155599

-- Define the conditions
variables (a : ℝ)

-- State the problem in Lean 4
theorem not_possible_to_form_triangle (h : a > 0) :
  ¬ (a + a > 2 * a ∧ a + 2 * a > a ∧ a + 2 * a > a) :=
by
  sorry

end not_possible_to_form_triangle_l1555_155599


namespace Walter_bus_time_l1555_155574

theorem Walter_bus_time :
  let start_time := 7 * 60 + 30 -- 7:30 a.m. in minutes
  let end_time := 16 * 60 + 15 -- 4:15 p.m. in minutes
  let away_time := end_time - start_time -- total time away from home in minutes
  let classes_time := 7 * 45 -- 7 classes 45 minutes each
  let lunch_time := 40 -- lunch time in minutes
  let additional_school_time := 1.5 * 60 -- additional time at school in minutes
  let school_time := classes_time + lunch_time + additional_school_time -- total school activities time
  (away_time - school_time) = 80 :=
by
  sorry

end Walter_bus_time_l1555_155574


namespace int_div_condition_l1555_155552

theorem int_div_condition (n : ℕ) (hn₁ : ∃ m : ℤ, 2^n - 2 = m * n) :
  ∃ k : ℤ, 2^(2^n - 1) - 2 = k * (2^n - 1) :=
by sorry

end int_div_condition_l1555_155552


namespace product_of_distinct_nonzero_real_satisfying_eq_l1555_155500

theorem product_of_distinct_nonzero_real_satisfying_eq (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
    (h : x + 3/x = y + 3/y) : x * y = 3 :=
by sorry

end product_of_distinct_nonzero_real_satisfying_eq_l1555_155500


namespace min_value_x_y_l1555_155531

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 8 / y = 1) : x + y ≥ 18 := 
sorry

end min_value_x_y_l1555_155531


namespace repeating_decimal_sum_l1555_155515

noncomputable def repeating_decimal_0_3 : ℚ := 1 / 3
noncomputable def repeating_decimal_0_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_0_2 : ℚ := 2 / 9

theorem repeating_decimal_sum :
  repeating_decimal_0_3 + repeating_decimal_0_6 - repeating_decimal_0_2 = 7 / 9 :=
by
  sorry

end repeating_decimal_sum_l1555_155515


namespace area_of_field_l1555_155558

noncomputable def area_square_field (speed_kmh : ℕ) (time_min : ℕ) : ℝ :=
  let speed_m_per_min := (speed_kmh * 1000) / 60
  let distance := speed_m_per_min * time_min
  let side_length := distance / Real.sqrt 2
  side_length ^ 2

-- Given conditions
theorem area_of_field : area_square_field 4 3 = 20000 := by
  sorry

end area_of_field_l1555_155558


namespace find_x_minus_y_l1555_155542

variables (x y z : ℝ)

theorem find_x_minus_y (h1 : x - (y + z) = 19) (h2 : x - y - z = 7): x - y = 13 :=
by {
  sorry
}

end find_x_minus_y_l1555_155542


namespace find_g3_value_l1555_155545

def g (n : ℕ) : ℕ :=
  if n < 5 then 2 * n ^ 2 + 3 else 4 * n + 1

theorem find_g3_value : g (g (g 3)) = 341 := by
  sorry

end find_g3_value_l1555_155545


namespace log5_square_simplification_l1555_155561

noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

theorem log5_square_simplification : (log5 (7 * log5 25))^2 = (log5 14)^2 :=
by
  sorry

end log5_square_simplification_l1555_155561


namespace ratio_of_larger_to_smaller_l1555_155583

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) : x / y = 2 :=
sorry

end ratio_of_larger_to_smaller_l1555_155583


namespace find_a2_l1555_155591

-- Definitions from the conditions
def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) := ∀ n, a (n + 1) = q * a n
def sum_geom_seq (a : ℕ → ℕ) (q : ℕ) (n : ℕ) := (a 0 * (1 - q^(n + 1))) / (1 - q)

-- Given conditions
def a_n : ℕ → ℕ := sorry -- Define the sequence a_n
def q : ℕ := 2
def S_4 := 60

-- The theorem to be proved
theorem find_a2 (h1: is_geometric_sequence a_n q)
                (h2: sum_geom_seq a_n q 3 = S_4) : 
                a_n 1 = 8 :=
sorry

end find_a2_l1555_155591


namespace meetings_percent_40_l1555_155507

def percent_of_workday_in_meetings (workday_hours : ℕ) (first_meeting_min : ℕ) (second_meeting_min : ℕ) (third_meeting_min : ℕ) : ℕ :=
  (first_meeting_min + second_meeting_min + third_meeting_min) * 100 / (workday_hours * 60)

theorem meetings_percent_40 (workday_hours : ℕ) (first_meeting_min : ℕ) (second_meeting_min : ℕ) (third_meeting_min : ℕ)
  (h_workday : workday_hours = 10) 
  (h_first_meeting : first_meeting_min = 40) 
  (h_second_meeting : second_meeting_min = 2 * first_meeting_min) 
  (h_third_meeting : third_meeting_min = first_meeting_min + second_meeting_min) : 
  percent_of_workday_in_meetings workday_hours first_meeting_min second_meeting_min third_meeting_min = 40 :=
by
  sorry

end meetings_percent_40_l1555_155507


namespace volume_of_prism_l1555_155519

   theorem volume_of_prism (a b c : ℝ)
     (h1 : a * b = 18) (h2 : b * c = 12) (h3 : a * c = 8) :
     a * b * c = 24 * Real.sqrt 3 :=
   sorry
   
end volume_of_prism_l1555_155519
