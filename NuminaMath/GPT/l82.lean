import Mathlib

namespace max_value_a7_a14_l82_82028

noncomputable def arithmetic_sequence_max_product (a_1 d : ℝ) : ℝ :=
  let a_7 := a_1 + 6 * d
  let a_14 := a_1 + 13 * d
  a_7 * a_14

theorem max_value_a7_a14 {a_1 d : ℝ} 
  (h : 10 = 2 * a_1 + 19 * d)
  (sum_first_20 : 100 = (10) * (a_1 + a_1 + 19 * d)) :
  arithmetic_sequence_max_product a_1 d = 25 :=
by
  sorry

end max_value_a7_a14_l82_82028


namespace totalCandlesInHouse_l82_82951

-- Definitions for the problem's conditions
def bedroomCandles : ℕ := 20
def livingRoomCandles : ℕ := bedroomCandles / 2
def donovanCandles : ℕ := 20

-- Problem to prove
theorem totalCandlesInHouse : bedroomCandles + livingRoomCandles + donovanCandles = 50 := by
  sorry

end totalCandlesInHouse_l82_82951


namespace nina_has_9_times_more_reading_homework_l82_82171

theorem nina_has_9_times_more_reading_homework
  (ruby_math_homework : ℕ)
  (ruby_reading_homework : ℕ)
  (nina_total_homework : ℕ)
  (nina_math_homework_factor : ℕ)
  (h1 : ruby_math_homework = 6)
  (h2 : ruby_reading_homework = 2)
  (h3 : nina_total_homework = 48)
  (h4 : nina_math_homework_factor = 4) :
  nina_total_homework - (ruby_math_homework * (nina_math_homework_factor + 1)) = 9 * ruby_reading_homework := by
  sorry

end nina_has_9_times_more_reading_homework_l82_82171


namespace tax_is_one_l82_82960

-- Define costs
def cost_eggs : ℕ := 3
def cost_pancakes : ℕ := 2
def cost_cocoa : ℕ := 2

-- Initial order
def initial_eggs := 1
def initial_pancakes := 1
def initial_mugs_of_cocoa := 2

-- Additional order by Ben
def additional_pancakes := 1
def additional_mugs_of_cocoa := 1

-- Calculate costs
def initial_cost : ℕ := initial_eggs * cost_eggs + initial_pancakes * cost_pancakes + initial_mugs_of_cocoa * cost_cocoa
def additional_cost : ℕ := additional_pancakes * cost_pancakes + additional_mugs_of_cocoa * cost_cocoa
def total_cost_before_tax : ℕ := initial_cost + additional_cost

-- Payment and change
def total_paid : ℕ := 15
def change : ℕ := 1
def actual_payment : ℕ := total_paid - change

-- Calculate tax
def tax : ℕ := actual_payment - total_cost_before_tax

-- Prove that the tax is $1
theorem tax_is_one : tax = 1 :=
by
  sorry

end tax_is_one_l82_82960


namespace dennis_initial_money_l82_82608

theorem dennis_initial_money :
  let cost_of_shirts := 27
  let change_bills := 2 * 10
  let change_coins := 3
  let total_change := change_bills + change_coins
  cost_of_shirts + total_change = 50 :=
by
  let cost_of_shirts := 27
  let change_bills := 2 * 10
  let change_coins := 3
  let total_change := change_bills + change_coins
  show cost_of_shirts + total_change = 50
  sorry

end dennis_initial_money_l82_82608


namespace longest_side_of_triangle_l82_82062

theorem longest_side_of_triangle (x : ℝ) (h1 : 8 + (2 * x + 5) + (3 * x + 2) = 40) : 
  max (max 8 (2 * x + 5)) (3 * x + 2) = 17 := 
by 
  -- proof goes here
  sorry

end longest_side_of_triangle_l82_82062


namespace tomatoes_ruined_and_discarded_l82_82605

theorem tomatoes_ruined_and_discarded 
  (W : ℝ)
  (C : ℝ)
  (P : ℝ)
  (S : ℝ)
  (profit_percentage : ℝ)
  (initial_cost : C = 0.80 * W)
  (remaining_tomatoes : S = 0.9956)
  (desired_profit : profit_percentage = 0.12)
  (final_cost : 0.896 = 0.80 + 0.096) :
  0.9956 * (1 - P / 100) = 0.896 :=
by
  sorry

end tomatoes_ruined_and_discarded_l82_82605


namespace asymptote_equation_l82_82733

theorem asymptote_equation {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (a + Real.sqrt (a^2 + b^2) = 2 * b) →
  (4 * x = 3 * y) ∨ (4 * x = -3 * y) :=
by
  sorry

end asymptote_equation_l82_82733


namespace Alex_failing_implies_not_all_hw_on_time_l82_82677

-- Definitions based on the conditions provided
variable (Alex_submits_all_hw_on_time : Prop)
variable (Alex_passes_course : Prop)

-- Given condition: Submitting all homework assignments implies passing the course
axiom Mrs_Thompson_statement : Alex_submits_all_hw_on_time → Alex_passes_course

-- The problem: Prove that if Alex failed the course, then he did not submit all homework assignments on time
theorem Alex_failing_implies_not_all_hw_on_time (h : ¬Alex_passes_course) : ¬Alex_submits_all_hw_on_time :=
  by
  sorry

end Alex_failing_implies_not_all_hw_on_time_l82_82677


namespace driers_drying_time_l82_82738

noncomputable def drying_time (r1 r2 r3 : ℝ) : ℝ := 1 / (r1 + r2 + r3)

theorem driers_drying_time (Q : ℝ) (r1 r2 r3 : ℝ)
  (h1 : r1 = Q / 24) 
  (h2 : r2 = Q / 2) 
  (h3 : r3 = Q / 8) : 
  drying_time r1 r2 r3 = 1.5 :=
by
  sorry

end driers_drying_time_l82_82738


namespace smallest_n_l82_82735

def n_expr (n : ℕ) : ℕ :=
  n * (2^7) * (3^2) * (7^3)

theorem smallest_n (n : ℕ) (h1: 25 ∣ n_expr n) (h2: 27 ∣ n_expr n) : n = 75 :=
sorry

end smallest_n_l82_82735


namespace average_salary_correct_l82_82877

/-- The salaries of A, B, C, D, and E. -/
def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

/-- The number of people. -/
def number_of_people : ℕ := 5

/-- The total salary is the sum of the salaries. -/
def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E

/-- The average salary is the total salary divided by the number of people. -/
def average_salary : ℕ := total_salary / number_of_people

/-- The average salary of A, B, C, D, and E is Rs. 8000. -/
theorem average_salary_correct : average_salary = 8000 := by
  sorry

end average_salary_correct_l82_82877


namespace An_nonempty_finite_l82_82389

def An (n : ℕ) : Set (ℕ × ℕ) :=
  { p : ℕ × ℕ | ∃ (k : ℕ), ∃ (a : ℕ), ∃ (b : ℕ), a = Nat.sqrt (p.1^2 + p.2 + n) ∧ b = Nat.sqrt (p.2^2 + p.1 + n) ∧ k = a + b }

theorem An_nonempty_finite (n : ℕ) (h : n ≥ 1) : Set.Nonempty (An n) ∧ Set.Finite (An n) :=
by
  sorry -- The proof goes here

end An_nonempty_finite_l82_82389


namespace dimes_difference_l82_82589

theorem dimes_difference
  (a b c d : ℕ)
  (h1 : a + b + c + d = 150)
  (h2 : 5 * a + 10 * b + 25 * c + 50 * d = 1500) :
  (b = 150 ∨ ∃ c d : ℕ, b = 0 ∧ 4 * c + 9 * d = 150) →
  ∃ b₁ b₂ : ℕ, (b₁ = 150 ∧ b₂ = 0 ∧ b₁ - b₂ = 150) :=
by
  sorry

end dimes_difference_l82_82589


namespace lyssa_fewer_correct_l82_82798

-- Define the total number of items in the exam
def total_items : ℕ := 75

-- Define the number of mistakes made by Lyssa
def lyssa_mistakes : ℕ := total_items * 20 / 100  -- 20% of 75

-- Define the number of correct answers by Lyssa
def lyssa_correct : ℕ := total_items - lyssa_mistakes

-- Define the number of mistakes made by Precious
def precious_mistakes : ℕ := 12

-- Define the number of correct answers by Precious
def precious_correct : ℕ := total_items - precious_mistakes

-- Statement to prove Lyssa got 3 fewer correct answers than Precious
theorem lyssa_fewer_correct : (precious_correct - lyssa_correct) = 3 := by
  sorry

end lyssa_fewer_correct_l82_82798


namespace cost_per_person_l82_82351

theorem cost_per_person (total_cost : ℕ) (num_people : ℕ) (h1 : total_cost = 30000) (h2 : num_people = 300) : total_cost / num_people = 100 := by
  -- No proof provided, only the theorem statement
  sorry

end cost_per_person_l82_82351


namespace stratified_sampling_third_grade_l82_82247

theorem stratified_sampling_third_grade (total_students : ℕ)
  (ratio_first_second_third : ℕ × ℕ × ℕ)
  (sample_size : ℕ) (r1 r2 r3 : ℕ) (h_ratio : ratio_first_second_third = (r1, r2, r3)) :
  total_students = 3000  ∧ ratio_first_second_third = (2, 3, 1)  ∧ sample_size = 180 →
  (sample_size * r3 / (r1 + r2 + r3) = 30) :=
sorry

end stratified_sampling_third_grade_l82_82247


namespace pyramid_volume_l82_82794

/-- Given the vertices of a triangle and its midpoints, calculate the volume of the folded triangular pyramid. -/
theorem pyramid_volume
  (A B C : ℝ × ℝ)
  (D E F : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (24, 0))
  (hC : C = (12, 16))
  (hD : D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hE : E = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
  (hF : F = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (area_ABC : ℝ)
  (h_area : area_ABC = 192)
  : (1 / 3) * area_ABC * 8 = 512 :=
by sorry

end pyramid_volume_l82_82794


namespace school_students_l82_82019

theorem school_students (x y : ℕ) (h1 : x + y = 432) (h2 : x - 16 = (y + 16) + 24) : x = 244 ∧ y = 188 := by
  sorry

end school_students_l82_82019


namespace unique_pair_not_opposite_l82_82804

def QuantumPair (a b : String): Prop := ∃ oppositeMeanings : Bool, a ≠ b ∧ oppositeMeanings

theorem unique_pair_not_opposite :
  ∃ (a b : String), 
    (a = "increase of 2 years" ∧ b = "decrease of 2 liters") ∧ 
    (¬ QuantumPair a b) :=
by 
  sorry

end unique_pair_not_opposite_l82_82804


namespace scarlet_savings_l82_82319

theorem scarlet_savings : 
  let initial_savings := 80
  let cost_earrings := 23
  let cost_necklace := 48
  let total_spent := cost_earrings + cost_necklace
  initial_savings - total_spent = 9 := 
by 
  sorry

end scarlet_savings_l82_82319


namespace solve_equation_correctly_l82_82938

theorem solve_equation_correctly : 
  ∀ x : ℝ, (x - 1) / 2 - 1 = (2 * x + 1) / 3 → x = -11 :=
by
  intro x h
  sorry

end solve_equation_correctly_l82_82938


namespace sequence_arithmetic_l82_82842

variable (a b : ℕ → ℤ)

theorem sequence_arithmetic :
  a 0 = 3 →
  (∀ n : ℕ, n > 0 → b n = a (n + 1) - a n) →
  b 3 = -2 →
  b 10 = 12 →
  a 8 = 3 :=
by
  intros h1 ha hb3 hb10
  sorry

end sequence_arithmetic_l82_82842


namespace center_of_circle_l82_82688

theorem center_of_circle (x y : ℝ) (h : x^2 + y^2 = 4 * x - 6 * y + 9) : x + y = -1 := 
by 
  sorry

end center_of_circle_l82_82688


namespace how_many_cakes_each_friend_ate_l82_82970

-- Definitions pertaining to the problem conditions
def crackers : ℕ := 29
def cakes : ℕ := 30
def friends : ℕ := 2

-- The main theorem statement we aim to prove
theorem how_many_cakes_each_friend_ate 
  (h1 : crackers = 29)
  (h2 : cakes = 30)
  (h3 : friends = 2) : 
  (cakes / friends = 15) :=
by
  sorry

end how_many_cakes_each_friend_ate_l82_82970


namespace one_cubic_foot_is_1728_cubic_inches_l82_82360

-- Define the basic equivalence of feet to inches.
def foot_to_inch : ℝ := 12

-- Define the conversion from cubic feet to cubic inches.
def cubic_foot_to_cubic_inch (cubic_feet : ℝ) : ℝ :=
  (foot_to_inch * cubic_feet) ^ 3

-- State the theorem to prove the equivalence in cubic measurement.
theorem one_cubic_foot_is_1728_cubic_inches : cubic_foot_to_cubic_inch 1 = 1728 :=
  sorry -- Proof skipped.

end one_cubic_foot_is_1728_cubic_inches_l82_82360


namespace one_positive_zero_l82_82309

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * x - 1

theorem one_positive_zero (a : ℝ) : ∃ x : ℝ, x > 0 ∧ f x a = 0 :=
sorry

end one_positive_zero_l82_82309


namespace find_p_l82_82899

variable (f w : ℂ) (p : ℂ)
variable (h1 : f = 4)
variable (h2 : w = 10 + 200 * Complex.I)
variable (h3 : f * p - w = 20000)

theorem find_p : p = 5002.5 + 50 * Complex.I := by
  sorry

end find_p_l82_82899


namespace anna_pays_total_l82_82601

-- Define the conditions
def daily_rental_cost : ℝ := 35
def cost_per_mile : ℝ := 0.25
def rental_days : ℝ := 3
def miles_driven : ℝ := 300

-- Define the total cost function
def total_cost (daily_rental_cost cost_per_mile rental_days miles_driven : ℝ) : ℝ :=
  (daily_rental_cost * rental_days) + (cost_per_mile * miles_driven)

-- The statement to be proved
theorem anna_pays_total : total_cost daily_rental_cost cost_per_mile rental_days miles_driven = 180 :=
by
  sorry

end anna_pays_total_l82_82601


namespace certain_number_is_sixteen_l82_82931

theorem certain_number_is_sixteen (x : ℝ) (h : x ^ 5 = 4 ^ 10) : x = 16 :=
by
  sorry

end certain_number_is_sixteen_l82_82931


namespace expand_polynomial_l82_82537

theorem expand_polynomial (x : ℝ) : (x - 3) * (4 * x + 12) = 4 * x ^ 2 - 36 := 
by {
  sorry
}

end expand_polynomial_l82_82537


namespace percentage_of_students_owning_cats_l82_82133

theorem percentage_of_students_owning_cats (N C : ℕ) (hN : N = 500) (hC : C = 75) :
  (C / N : ℚ) * 100 = 15 := by
  sorry

end percentage_of_students_owning_cats_l82_82133


namespace find_N_aN_bN_cN_dN_eN_l82_82824

theorem find_N_aN_bN_cN_dN_eN:
  ∃ (a b c d e : ℝ) (N : ℝ),
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧
    (a^2 + b^2 + c^2 + d^2 + e^2 = 1000) ∧
    (N = c * (a + 3 * b + 4 * d + 6 * e)) ∧
    (N + a + b + c + d + e = 150 + 250 * Real.sqrt 62 + 10 * Real.sqrt 50) := by
  sorry

end find_N_aN_bN_cN_dN_eN_l82_82824


namespace a_16_value_l82_82986

-- Define the recurrence relation
def seq (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0       => 2
  | (n + 1) => (1 + a n) / (1 - a n)

-- State the theorem
theorem a_16_value :
  seq (a : ℕ → ℚ) 16 = -1/3 := 
sorry

end a_16_value_l82_82986


namespace imaginary_part_of_complex_l82_82045

open Complex -- Opens the complex numbers namespace

theorem imaginary_part_of_complex:
  ∀ (a b c d : ℂ), (a = (2 + I) / (1 - I) - (2 - I) / (1 + I)) → (a.im = 3) :=
by
  sorry

end imaginary_part_of_complex_l82_82045


namespace average_weight_of_Arun_l82_82308

def arun_opinion (w : ℝ) : Prop := 66 < w ∧ w < 72
def brother_opinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def mother_opinion (w : ℝ) : Prop := w ≤ 69

theorem average_weight_of_Arun :
  (∀ w, arun_opinion w → brother_opinion w → mother_opinion w → 
    (w = 67 ∨ w = 68 ∨ w = 69)) →
  avg_weight = 68 :=
sorry

end average_weight_of_Arun_l82_82308


namespace arithmetic_sequence_sum_l82_82387

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the problem conditions
def problem_conditions (a : ℕ → ℝ) : Prop :=
  (a 3 + a 8 = 3) ∧ is_arithmetic_sequence a

-- State the theorem to be proved
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : problem_conditions a) : a 1 + a 10 = 3 :=
sorry

end arithmetic_sequence_sum_l82_82387


namespace valentines_count_l82_82857

theorem valentines_count (x y : ℕ) (h : x * y = x + y + 52) : x * y = 108 :=
by sorry

end valentines_count_l82_82857


namespace count_4_letter_words_with_A_l82_82059

-- Define the alphabet set and the properties
def Alphabet : Finset (Char) := {'A', 'B', 'C', 'D', 'E'}
def total_words := (Alphabet.card ^ 4 : ℕ)
def total_words_without_A := (Alphabet.erase 'A').card ^ 4
def total_words_with_A := total_words - total_words_without_A

-- The key theorem to prove
theorem count_4_letter_words_with_A : total_words_with_A = 369 := sorry

end count_4_letter_words_with_A_l82_82059


namespace johns_final_push_time_l82_82954

-- Definitions and assumptions
def john_speed : ℝ := 4.2
def steve_speed : ℝ := 3.8
def initial_gap : ℝ := 15
def final_gap : ℝ := 2

theorem johns_final_push_time :
  ∃ t : ℝ, john_speed * t = steve_speed * t + initial_gap + final_gap ∧ t = 42.5 :=
by
  sorry

end johns_final_push_time_l82_82954


namespace liquid_X_percent_in_mixed_solution_l82_82100

theorem liquid_X_percent_in_mixed_solution (wP wQ : ℝ) (xP xQ : ℝ) (mP mQ : ℝ) :
  xP = 0.005 * wP →
  xQ = 0.015 * wQ →
  wP = 200 →
  wQ = 800 →
  13 / 1000 * 100 = 1.3 :=
by
  intros h1 h2 h3 h4
  sorry

end liquid_X_percent_in_mixed_solution_l82_82100


namespace determine_n_l82_82176

-- Constants and variables
variables {a : ℕ → ℝ} {n : ℕ}

-- Definition for the condition at each vertex
def vertex_condition (a : ℕ → ℝ) (i : ℕ) : Prop :=
  a i = a (i - 1) * a (i + 1)

-- Mathematical problem statement
theorem determine_n (h : ∀ i, vertex_condition a i) (distinct_a : ∀ i j, a i ≠ a j) : n = 6 :=
sorry

end determine_n_l82_82176


namespace grid_black_probability_l82_82212

theorem grid_black_probability :
  let p_black_each_cell : ℝ := 1 / 3 
  let p_not_black : ℝ := (2 / 3) * (2 / 3)
  let p_one_black : ℝ := 1 - p_not_black
  let total_pairs : ℕ := 8
  (p_one_black ^ total_pairs) = (5 / 9) ^ 8 :=
sorry

end grid_black_probability_l82_82212


namespace probability_not_above_y_axis_l82_82650

-- Define the vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P := Point.mk (-1) 5
def Q := Point.mk 2 (-3)
def R := Point.mk (-5) (-3)
def S := Point.mk (-8) 5

-- Define predicate for being above the y-axis
def is_above_y_axis (p : Point) : Prop := p.y > 0

-- Define the parallelogram region (this is theoretical as defining a whole region 
-- can be complex, but we state the region as a property)
noncomputable def in_region_of_parallelogram (p : Point) : Prop := sorry

-- Define the probability calculation statement
theorem probability_not_above_y_axis (p : Point) :
  in_region_of_parallelogram p → ¬is_above_y_axis p := sorry

end probability_not_above_y_axis_l82_82650


namespace security_deposit_correct_l82_82359

-- Definitions (Conditions)
def daily_rate : ℝ := 125
def pet_fee_per_dog : ℝ := 100
def number_of_dogs : ℕ := 2
def tourism_tax_rate : ℝ := 0.10
def service_fee_rate : ℝ := 0.20
def activity_cost_per_person : ℝ := 45
def number_of_activities_per_person : ℕ := 3
def number_of_people : ℕ := 2
def security_deposit_rate : ℝ := 0.50
def usd_to_euro_conversion_rate : ℝ := 0.83

-- Function to calculate total cost
def total_cost_in_euros : ℝ :=
  let rental_cost := daily_rate * 14
  let pet_cost := pet_fee_per_dog * number_of_dogs
  let tourism_tax := tourism_tax_rate * rental_cost
  let service_fee := service_fee_rate * rental_cost
  let cabin_total := rental_cost + pet_cost + tourism_tax + service_fee
  let activities_total := number_of_activities_per_person * activity_cost_per_person * number_of_people
  let total_cost := cabin_total + activities_total
  let security_deposit_usd := security_deposit_rate * total_cost
  security_deposit_usd * usd_to_euro_conversion_rate

-- Theorem to prove
theorem security_deposit_correct :
  total_cost_in_euros = 1139.18 := 
sorry

end security_deposit_correct_l82_82359


namespace halfway_fraction_l82_82218

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l82_82218


namespace binomial_coefficient_middle_term_l82_82649

theorem binomial_coefficient_middle_term :
  let n := 11
  let sum_odd := 1024
  sum_odd = 2^(n-1) →
  let binom_coef := Nat.choose n (n / 2 - 1)
  binom_coef = 462 :=
by
  intro n
  let n := 11
  intro sum_odd
  let sum_odd := 1024
  intro h
  let binom_coef := Nat.choose n (n / 2 - 1)
  have : binom_coef = 462 := sorry
  exact this

end binomial_coefficient_middle_term_l82_82649


namespace value_of_f_log_20_l82_82293

variable (f : ℝ → ℝ)
variable (h₁ : ∀ x : ℝ, f (-x) = -f x)
variable (h₂ : ∀ x : ℝ, f (x - 2) = f (x + 2))
variable (h₃ : ∀ x : ℝ, x > -1 ∧ x < 0 → f x = 2^x + 1/5)

theorem value_of_f_log_20 : f (Real.log 20 / Real.log 2) = -1 := sorry

end value_of_f_log_20_l82_82293


namespace calculate_regular_rate_l82_82063

def regular_hours_per_week : ℕ := 6 * 10
def total_weeks : ℕ := 4
def total_regular_hours : ℕ := regular_hours_per_week * total_weeks
def total_worked_hours : ℕ := 245
def overtime_hours : ℕ := total_worked_hours - total_regular_hours
def overtime_rate : ℚ := 4.20
def total_earning : ℚ := 525
def total_overtime_pay : ℚ := overtime_hours * overtime_rate
def total_regular_pay : ℚ := total_earning - total_overtime_pay
def regular_rate : ℚ := total_regular_pay / total_regular_hours

theorem calculate_regular_rate : regular_rate = 2.10 :=
by
  -- The proof would go here
  sorry

end calculate_regular_rate_l82_82063


namespace find_possible_values_of_a_l82_82747

noncomputable def find_a (x y a : ℝ) : Prop :=
  (x + y = a) ∧ (x^3 + y^3 = a) ∧ (x^5 + y^5 = a)

theorem find_possible_values_of_a (a : ℝ) :
  (∃ x y : ℝ, find_a x y a) ↔ (a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2) :=
sorry

end find_possible_values_of_a_l82_82747


namespace benjamin_skating_time_l82_82943

-- Defining the conditions
def distance : ℕ := 80 -- Distance in kilometers
def speed : ℕ := 10   -- Speed in kilometers per hour

-- The main theorem statement
theorem benjamin_skating_time : ∀ (T : ℕ), T = distance / speed → T = 8 := by
  sorry

end benjamin_skating_time_l82_82943


namespace maximum_marks_l82_82755

noncomputable def passing_mark (M : ℝ) : ℝ := 0.35 * M

theorem maximum_marks (M : ℝ) (h1 : passing_mark M = 210) : M = 600 :=
  by
  sorry

end maximum_marks_l82_82755


namespace mean_score_of_seniors_l82_82410

variable (s n : ℕ)  -- Number of seniors and non-seniors
variable (m_s m_n : ℝ)  -- Mean scores of seniors and non-seniors
variable (total_mean : ℝ) -- Mean score of all students
variable (total_students : ℕ) -- Total number of students

theorem mean_score_of_seniors :
  total_students = 100 → total_mean = 100 →
  n = 3 * s / 2 →
  s * m_s + n * m_n = total_students * total_mean →
  m_s = (3 * m_n / 2) →
  m_s = 125 :=
by
  intros
  sorry

end mean_score_of_seniors_l82_82410


namespace negation_of_forall_prop_l82_82031

theorem negation_of_forall_prop :
  ¬ (∀ x : ℝ, x^2 + x > 0) ↔ ∃ x : ℝ, x^2 + x ≤ 0 :=
by
  sorry

end negation_of_forall_prop_l82_82031


namespace faye_pencils_l82_82846

theorem faye_pencils :
  ∀ (packs : ℕ) (pencils_per_pack : ℕ) (rows : ℕ) (total_pencils pencils_per_row : ℕ),
  packs = 35 →
  pencils_per_pack = 4 →
  rows = 70 →
  total_pencils = packs * pencils_per_pack →
  pencils_per_row = total_pencils / rows →
  pencils_per_row = 2 :=
by
  intros packs pencils_per_pack rows total_pencils pencils_per_row
  intros packs_eq pencils_per_pack_eq rows_eq total_pencils_eq pencils_per_row_eq
  sorry

end faye_pencils_l82_82846


namespace distinct_complex_numbers_no_solution_l82_82790

theorem distinct_complex_numbers_no_solution :
  ¬∃ (a b c d : ℂ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
  (a^3 - b * c * d = b^3 - c * d * a) ∧ 
  (b^3 - c * d * a = c^3 - d * a * b) ∧ 
  (c^3 - d * a * b = d^3 - a * b * c) := 
by {
  sorry
}

end distinct_complex_numbers_no_solution_l82_82790


namespace probability_diff_colors_l82_82289

-- Definitions based on the conditions provided.
-- Total number of chips
def total_chips := 15

-- Individual probabilities of drawing each color first
def prob_green_first := 6 / total_chips
def prob_purple_first := 5 / total_chips
def prob_orange_first := 4 / total_chips

-- Probabilities of drawing a different color second
def prob_not_green := 9 / total_chips
def prob_not_purple := 10 / total_chips
def prob_not_orange := 11 / total_chips

-- Combined probabilities for each case
def prob_green_then_diff := prob_green_first * prob_not_green
def prob_purple_then_diff := prob_purple_first * prob_not_purple
def prob_orange_then_diff := prob_orange_first * prob_not_orange

-- Total probability of drawing two chips of different colors
def total_prob_diff_colors := prob_green_then_diff + prob_purple_then_diff + prob_orange_then_diff

-- Theorem statement to be proved
theorem probability_diff_colors : total_prob_diff_colors = 148 / 225 :=
by
  -- Proof would go here
  sorry

end probability_diff_colors_l82_82289


namespace find_real_solutions_l82_82644

variable (x : ℝ)

theorem find_real_solutions :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔ 
  (x = -2 * Real.sqrt 14 ∨ x = 2 * Real.sqrt 14) := 
sorry

end find_real_solutions_l82_82644


namespace meaningful_fraction_l82_82038

theorem meaningful_fraction (x : ℝ) : (x - 1 ≠ 0) ↔ (x ≠ 1) :=
by sorry

end meaningful_fraction_l82_82038


namespace arithmetic_seq_perfect_sixth_power_l82_82740

theorem arithmetic_seq_perfect_sixth_power 
  (a h : ℤ)
  (seq : ∀ n : ℕ, ℤ)
  (h_seq : ∀ n, seq n = a + n * h)
  (h1 : ∃ s₁ x, seq s₁ = x^2)
  (h2 : ∃ s₂ y, seq s₂ = y^3) :
  ∃ k s, seq s = k^6 := 
sorry

end arithmetic_seq_perfect_sixth_power_l82_82740


namespace probability_Rachel_Robert_in_picture_l82_82135

noncomputable def Rachel_lap_time := 75
noncomputable def Robert_lap_time := 70
noncomputable def photo_time_start := 900
noncomputable def photo_time_end := 960
noncomputable def track_fraction := 1 / 5

theorem probability_Rachel_Robert_in_picture :
  let lap_time_Rachel := Rachel_lap_time
  let lap_time_Robert := Robert_lap_time
  let time_start := photo_time_start
  let time_end := photo_time_end
  let interval_Rachel := 15  -- ±15 seconds for Rachel
  let interval_Robert := 14  -- ±14 seconds for Robert
  let probability := (2 * interval_Robert) / (time_end - time_start) 
  probability = 7 / 15 :=
by
  sorry

end probability_Rachel_Robert_in_picture_l82_82135


namespace largest_angle_in_pentagon_l82_82422

def pentagon_angle_sum : ℝ := 540

def angle_A : ℝ := 70
def angle_B : ℝ := 90
def angle_C (x : ℝ) : ℝ := x
def angle_D (x : ℝ) : ℝ := x
def angle_E (x : ℝ) : ℝ := 3 * x - 10

theorem largest_angle_in_pentagon
  (x : ℝ)
  (h_sum : angle_A + angle_B + angle_C x + angle_D x + angle_E x = pentagon_angle_sum) :
  angle_E x = 224 :=
sorry

end largest_angle_in_pentagon_l82_82422


namespace position_of_2017_in_arithmetic_sequence_l82_82144

theorem position_of_2017_in_arithmetic_sequence :
  ∀ (n : ℕ), 4 + 3 * (n - 1) = 2017 → n = 672 :=
by
  intros n h
  sorry

end position_of_2017_in_arithmetic_sequence_l82_82144


namespace bus_return_trip_fraction_l82_82435

theorem bus_return_trip_fraction :
  (3 / 4 * 200 + x * 200 = 310) → (x = 4 / 5) := by
  sorry

end bus_return_trip_fraction_l82_82435


namespace arithmetic_sequence_difference_l82_82651

def arithmetic_sequence (a d n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem arithmetic_sequence_difference :
  let a := 3
  let d := 7
  let a₁₀₀₀ := arithmetic_sequence a d 1000
  let a₁₀₀₃ := arithmetic_sequence a d 1003
  abs (a₁₀₀₃ - a₁₀₀₀) = 21 :=
by
  sorry

end arithmetic_sequence_difference_l82_82651


namespace sum_between_52_and_53_l82_82892

theorem sum_between_52_and_53 (x y : ℝ) (h1 : y = 4 * (⌊x⌋ : ℝ) + 2) (h2 : y = 5 * (⌊x - 3⌋ : ℝ) + 7) (h3 : ∀ n : ℤ, x ≠ n) :
  52 < x + y ∧ x + y < 53 := 
sorry

end sum_between_52_and_53_l82_82892


namespace ratio_of_areas_ACP_BQA_l82_82867

open EuclideanGeometry

-- Define the geometric configuration
variables (A B C D P Q : Point)
  (is_square : square A B C D)
  (is_bisector_CAD : is_angle_bisector A C D P)
  (is_bisector_ABD : is_angle_bisector B A D Q)

-- Define the areas of triangles
def area_triangle (X Y Z : Point) : Real := sorry -- Placeholder for the area function

-- Lean statement for the proof problem
theorem ratio_of_areas_ACP_BQA 
  (h_square : is_square) 
  (h_bisector_CAD : is_bisector_CAD) 
  (h_bisector_ABD : is_bisector_ABD) :
  (area_triangle A C P) / (area_triangle B Q A) = 2 :=
sorry

end ratio_of_areas_ACP_BQA_l82_82867


namespace inequality_problem_l82_82201

variable {a b c d : ℝ}

theorem inequality_problem (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
    a + c > b + d ∧ ad^2 > bc^2 ∧ (1 / bc) < (1 / ad) :=
by
  sorry

end inequality_problem_l82_82201


namespace cricket_bat_profit_percentage_l82_82884

theorem cricket_bat_profit_percentage 
  (selling_price profit : ℝ) 
  (h_sp: selling_price = 850) 
  (h_p: profit = 230) : 
  (profit / (selling_price - profit) * 100) = 37.10 :=
by
  sorry

end cricket_bat_profit_percentage_l82_82884


namespace solution_to_equation_l82_82175

noncomputable def solve_equation (x : ℝ) : Prop :=
  x + 2 = 1 / (x - 2) ∧ x ≠ 2

theorem solution_to_equation (x : ℝ) (h : solve_equation x) : x = Real.sqrt 5 ∨ x = -Real.sqrt 5 :=
sorry

end solution_to_equation_l82_82175


namespace total_rope_length_l82_82668

theorem total_rope_length 
  (longer_side : ℕ) (shorter_side : ℕ) 
  (h1 : longer_side = 28) (h2 : shorter_side = 22) : 
  2 * longer_side + 2 * shorter_side = 100 := by
  sorry

end total_rope_length_l82_82668


namespace sector_area_l82_82908

theorem sector_area (arc_length radius : ℝ) (h1 : arc_length = 2) (h2 : radius = 2) : 
  (1/2) * arc_length * radius = 2 :=
by
  -- sorry placeholder for proof
  sorry

end sector_area_l82_82908


namespace circle_equation_l82_82257

theorem circle_equation (x y : ℝ) : (3 * x - 4 * y + 12 = 0) → (x^2 + 4 * x + y^2 - 3 * y = 0) :=
sorry

end circle_equation_l82_82257


namespace simplify_expression_l82_82476

def i : Complex := Complex.I

theorem simplify_expression : 7 * (4 - 2 * i) + 4 * i * (7 - 2 * i) = 36 + 14 * i := by
  sorry

end simplify_expression_l82_82476


namespace linear_relationship_selling_price_maximize_profit_l82_82828

theorem linear_relationship (k b : ℝ)
  (h₁ : 36 = 12 * k + b)
  (h₂ : 34 = 13 * k + b) :
  y = -2 * x + 60 :=
by
  sorry

theorem selling_price (p c x : ℝ)
  (h₁ : x ≥ 10)
  (h₂ : x ≤ 19)
  (h₃ : x - 10 = (192 / (y + 10))) :
  x = 18 :=
by
  sorry

theorem maximize_profit (x w : ℝ)
  (h_max : x = 19)
  (h_profit : w = -2 * x^2 + 80 * x - 600) :
  w = 198 :=
by
  sorry

end linear_relationship_selling_price_maximize_profit_l82_82828


namespace isosceles_triangle_area_l82_82104

theorem isosceles_triangle_area (PQ PR QR : ℝ) (PS : ℝ) (h1 : PQ = PR)
  (h2 : QR = 10) (h3 : PS^2 + (QR / 2)^2 = PQ^2) : 
  (1/2) * QR * PS = 60 :=
by
  sorry

end isosceles_triangle_area_l82_82104


namespace percentage_increase_l82_82129

theorem percentage_increase (old_earnings new_earnings : ℝ) (h_old : old_earnings = 50) (h_new : new_earnings = 70) :
  ((new_earnings - old_earnings) / old_earnings) * 100 = 40 :=
by
  rw [h_old, h_new]
  -- Simplification and calculation steps would go here
  sorry

end percentage_increase_l82_82129


namespace total_points_scored_l82_82669

theorem total_points_scored (n m T : ℕ) 
  (h1 : T = 2 * n + 5 * m) 
  (h2 : n = m + 3 ∨ m = n + 3)
  : T = 20 :=
sorry

end total_points_scored_l82_82669


namespace cricketer_hits_two_sixes_l82_82349

-- Definitions of the given conditions
def total_runs : ℕ := 132
def boundaries_count : ℕ := 12
def running_percent : ℚ := 54.54545454545454 / 100

-- Function to calculate runs made by running
def runs_by_running (total: ℕ) (percent: ℚ) : ℚ :=
  percent * total

-- Function to calculate runs made from boundaries
def runs_from_boundaries (count: ℕ) : ℕ :=
  count * 4

-- Function to calculate runs made from sixes
def runs_from_sixes (total: ℕ) (boundaries_runs: ℕ) (running_runs: ℚ) : ℚ :=
  total - boundaries_runs - running_runs

-- Function to calculate number of sixes hit
def number_of_sixes (sixes_runs: ℚ) : ℚ :=
  sixes_runs / 6

-- The proof statement for the cricketer hitting 2 sixes
theorem cricketer_hits_two_sixes:
  number_of_sixes (runs_from_sixes total_runs (runs_from_boundaries boundaries_count) (runs_by_running total_runs running_percent)) = 2 := by
  sorry

end cricketer_hits_two_sixes_l82_82349


namespace trapezoid_sides_l82_82990

theorem trapezoid_sides (r kl: ℝ) (h1 : r = 5) (h2 : kl = 8) :
  ∃ (ab cd bc_ad : ℝ), ab = 5 ∧ cd = 20 ∧ bc_ad = 12.5 :=
by
  sorry

end trapezoid_sides_l82_82990


namespace problem_statement_l82_82332

-- Define proposition p
def prop_p : Prop := ∃ x : ℝ, Real.exp x ≥ x + 1

-- Define proposition q
def prop_q : Prop := ∀ (a b : ℝ), a^2 < b^2 → a < b

-- The final statement we want to prove
theorem problem_statement : (prop_p ∧ ¬prop_q) :=
by
  sorry

end problem_statement_l82_82332


namespace length_EQ_l82_82491

-- Define the square EFGH with side length 8
def square_EFGH (a : ℝ) (b : ℝ): Prop := a = 8 ∧ b = 8

-- Define the rectangle IJKL with IL = 12 and JK = 8
def rectangle_IJKL (l : ℝ) (w : ℝ): Prop := l = 12 ∧ w = 8

-- Define the perpendicularity of EH and IJ
def perpendicular_EH_IJ : Prop := true

-- Define the shaded area condition
def shaded_area_condition (area_IJKL : ℝ) (shaded_area : ℝ): Prop :=
  shaded_area = (1/3) * area_IJKL

-- Theorem to prove
theorem length_EQ (a b l w area_IJKL shaded_area EH HG HQ EQ : ℝ):
  square_EFGH a b →
  rectangle_IJKL l w →
  perpendicular_EH_IJ →
  shaded_area_condition area_IJKL shaded_area →
  HQ * HG = shaded_area →
  EQ = EH - HQ →
  EQ = 4 := by
  intros hSquare hRectangle hPerpendicular hShadedArea hHQHG hEQ
  sorry

end length_EQ_l82_82491


namespace probability_at_least_one_woman_selected_l82_82330

open Classical

noncomputable def probability_of_selecting_at_least_one_woman : ℚ :=
  1 - (10 / 15) * (9 / 14) * (8 / 13) * (7 / 12) * (6 / 11)

theorem probability_at_least_one_woman_selected :
  probability_of_selecting_at_least_one_woman = 917 / 1001 :=
sorry

end probability_at_least_one_woman_selected_l82_82330


namespace compute_five_fold_application_l82_82512

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then -x^2 else x + 10

theorem compute_five_fold_application : f (f (f (f (f 2)))) = -16 :=
by
  sorry

end compute_five_fold_application_l82_82512


namespace sale_saving_percentage_l82_82720

theorem sale_saving_percentage (P : ℝ) : 
  let original_price := 8 * P
  let sale_price := 6 * P
  let amount_saved := original_price - sale_price
  let percentage_saved := (amount_saved / original_price) * 100
  percentage_saved = 25 :=
by
  sorry

end sale_saving_percentage_l82_82720


namespace isosceles_right_triangle_leg_length_l82_82220

theorem isosceles_right_triangle_leg_length (m : ℝ) (h : ℝ) (x : ℝ) 
  (h1 : m = 12) 
  (h2 : m = h / 2)
  (h3 : h = x * Real.sqrt 2) :
  x = 12 * Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_leg_length_l82_82220


namespace time_for_a_and_b_together_l82_82612

variable (R_a R_b : ℝ)
variable (T_ab : ℝ)

-- Given conditions
def condition_1 : Prop := R_a = 3 * R_b
def condition_2 : Prop := R_a * 28 = 1  -- '1' denotes the entire work

-- Proof goal
theorem time_for_a_and_b_together (h1 : condition_1 R_a R_b) (h2 : condition_2 R_a) : T_ab = 21 := 
by
  sorry

end time_for_a_and_b_together_l82_82612


namespace find_angle_ACD_l82_82801

-- Define the vertices of the quadrilateral
variables {A B C D : Type*}

-- Given angles and side equality
variables (angle_DAC : ℝ) (angle_DBC : ℝ) (angle_BCD : ℝ) (eq_BC_AD : Prop)

-- The given conditions in the problem
axiom angle_DAC_is_98 : angle_DAC = 98
axiom angle_DBC_is_82 : angle_DBC = 82
axiom angle_BCD_is_70 : angle_BCD = 70
axiom BC_eq_AD : eq_BC_AD = true

-- Target angle to be proven
def angle_ACD : ℝ := 28

-- The theorem
theorem find_angle_ACD (h1 : angle_DAC = 98)
                       (h2 : angle_DBC = 82)
                       (h3 : angle_BCD = 70)
                       (h4 : eq_BC_AD) : angle_ACD = 28 := 
by
  sorry  -- Proof of the theorem

end find_angle_ACD_l82_82801


namespace area_of_given_parallelogram_l82_82254

def parallelogram_base : ℝ := 24
def parallelogram_height : ℝ := 16
def parallelogram_area (b h : ℝ) : ℝ := b * h

theorem area_of_given_parallelogram : parallelogram_area parallelogram_base parallelogram_height = 384 := 
by sorry

end area_of_given_parallelogram_l82_82254


namespace value_of_R_l82_82369

theorem value_of_R (R : ℝ) (hR_pos : 0 < R)
  (h_line : ∀ x y : ℝ, x + y = 2 * R)
  (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = R) :
  R = (3 + Real.sqrt 5) / 4 ∨ R = (3 - Real.sqrt 5) / 4 :=
by
  sorry

end value_of_R_l82_82369


namespace cricket_player_average_increase_l82_82279

theorem cricket_player_average_increase
  (average : ℕ) (n : ℕ) (next_innings_runs : ℕ) 
  (x : ℕ) 
  (h1 : average = 32)
  (h2 : n = 20)
  (h3 : next_innings_runs = 200)
  (total_runs := average * n)
  (new_total_runs := total_runs + next_innings_runs)
  (new_average := (average + x))
  (new_total := new_average * (n + 1)):
  new_total_runs = 840 →
  new_total = 840 →
  x = 8 :=
by
  sorry

end cricket_player_average_increase_l82_82279


namespace maximum_n_l82_82237

theorem maximum_n (n : ℕ) (G : SimpleGraph (Fin n)) :
  (∃ (A : Fin n → Set (Fin 2020)),  ∀ i j, (G.Adj i j ↔ (A i ∩ A j ≠ ∅)) →
  n ≤ 89) := sorry

end maximum_n_l82_82237


namespace sum_interior_angles_of_regular_polygon_l82_82685

theorem sum_interior_angles_of_regular_polygon (exterior_angle : ℝ) (sum_exterior_angles : ℝ) (n : ℝ)
  (h1 : exterior_angle = 45)
  (h2 : sum_exterior_angles = 360)
  (h3 : n = sum_exterior_angles / exterior_angle) :
  180 * (n - 2) = 1080 :=
by
  sorry

end sum_interior_angles_of_regular_polygon_l82_82685


namespace mean_of_y_and_18_is_neg1_l82_82282

theorem mean_of_y_and_18_is_neg1 (y : ℤ) : 
  ((4 + 6 + 10 + 14) / 4) = ((y + 18) / 2) → y = -1 := 
by 
  -- Placeholder for the proof
  sorry

end mean_of_y_and_18_is_neg1_l82_82282


namespace find_angle_C_l82_82073

noncomputable def ABC_triangle (A B C a b c : ℝ) : Prop :=
b = c * Real.cos A + Real.sqrt 3 * a * Real.sin C

theorem find_angle_C (A B C a b c : ℝ) (h : ABC_triangle A B C a b c) :
  C = π / 6 :=
sorry

end find_angle_C_l82_82073


namespace max_knights_seated_next_to_two_knights_l82_82021

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l82_82021


namespace acute_angle_sum_l82_82856

open Real

theorem acute_angle_sum (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
                        (hβ : 0 < β ∧ β < π / 2)
                        (h1 : 3 * (sin α) ^ 2 + 2 * (sin β) ^ 2 = 1)
                        (h2 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0) :
  α + 2 * β = π / 2 :=
sorry

end acute_angle_sum_l82_82856


namespace quadratic_inequality_solution_l82_82388

theorem quadratic_inequality_solution (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) :=
  sorry

end quadratic_inequality_solution_l82_82388


namespace sequence_expression_l82_82759

theorem sequence_expression (a : ℕ → ℝ) (h_base : a 1 = 2)
  (h_rec : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * (n + 1) * a n / (a n + n)) :
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = (n + 1) * 2^(n + 1) / (2^(n + 1) - 1) :=
by
  sorry

end sequence_expression_l82_82759


namespace lisa_eats_one_candy_on_other_days_l82_82385

def candies_total : ℕ := 36
def candies_per_day_on_mondays_and_wednesdays : ℕ := 2
def weeks : ℕ := 4
def days_in_a_week : ℕ := 7
def mondays_and_wednesdays_in_4_weeks : ℕ := 2 * weeks
def total_candies_mondays_and_wednesdays : ℕ := mondays_and_wednesdays_in_4_weeks * candies_per_day_on_mondays_and_wednesdays
def total_other_candies : ℕ := candies_total - total_candies_mondays_and_wednesdays
def total_other_days : ℕ := weeks * (days_in_a_week - 2)
def candies_per_other_day : ℕ := total_other_candies / total_other_days

theorem lisa_eats_one_candy_on_other_days :
  candies_per_other_day = 1 :=
by
  -- Prove the theorem with conditions defined
  sorry

end lisa_eats_one_candy_on_other_days_l82_82385


namespace minimize_quadratic_l82_82428

theorem minimize_quadratic (c : ℝ) : ∃ b : ℝ, (∀ x : ℝ, 3 * x^2 + 2 * x + c ≥ 3 * b^2 + 2 * b + c) ∧ b = -1/3 :=
by
  sorry

end minimize_quadratic_l82_82428


namespace Faye_age_correct_l82_82054

def ages (C D E F G : ℕ) : Prop :=
  D = E - 2 ∧
  C = E + 3 ∧
  F = C - 1 ∧
  D = 16 ∧
  G = D - 5

theorem Faye_age_correct (C D E F G : ℕ) (h : ages C D E F G) : F = 20 :=
by {
  sorry
}

end Faye_age_correct_l82_82054


namespace math_problem_l82_82900

theorem math_problem
  (a b c x1 x2 : ℝ)
  (h1 : a > 0)
  (h2 : a^2 = 4 * b)
  (h3 : |x1 - x2| = 4)
  (h4 : x1 < x2) :
  (a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (c = 4) :=
by
  sorry

end math_problem_l82_82900


namespace find_p_q_r_l82_82923

theorem find_p_q_r : 
  ∃ (p q r : ℕ), 
  p > 0 ∧ q > 0 ∧ r > 0 ∧ 
  4 * (Real.sqrt (Real.sqrt 7) - Real.sqrt (Real.sqrt 6)) 
  = Real.sqrt (Real.sqrt p) + Real.sqrt (Real.sqrt q) - Real.sqrt (Real.sqrt r) 
  ∧ p + q + r = 99 := 
sorry

end find_p_q_r_l82_82923


namespace sum_of_coefficients_l82_82962

theorem sum_of_coefficients :
  ∃ (a b c d e : ℤ), (512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ (a + b + c + d + e = 60) :=
by
  sorry

end sum_of_coefficients_l82_82962


namespace a2_eq_1_l82_82350

-- Define the geometric sequence and the conditions
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Given conditions
variables (a : ℕ → ℝ) (q : ℝ)
axiom a1_eq_2 : a 1 = 2
axiom condition1 : geometric_sequence a q
axiom condition2 : 16 * a 3 * a 5 = 8 * a 4 - 1

-- Prove that a_2 = 1
theorem a2_eq_1 : a 2 = 1 :=
by
  -- This is where the proof would go
  sorry

end a2_eq_1_l82_82350


namespace bullseye_points_l82_82656

theorem bullseye_points (B : ℝ) (h : B + B / 2 = 75) : B = 50 :=
by
  sorry

end bullseye_points_l82_82656


namespace eval_fraction_l82_82914

theorem eval_fraction (a b : ℕ) : (40 : ℝ) = 2^3 * 5 → (10 : ℝ) = 2 * 5 → (40^56 / 10^28) = 160^28 :=
by 
  sorry

end eval_fraction_l82_82914


namespace driver_license_advantage_l82_82415

def AdvantageousReasonsForEarlyLicenseObtaining 
  (eligible : ℕ → Prop)
  (effectiveInsurance : ℕ → Prop)
  (rentalCarFlexibility : ℕ → Prop)
  (employmentOpportunity : ℕ → Prop) : Prop :=
  ∀ age1 age2 : ℕ, (eligible age1 ∧ eligible age2 ∧ age1 < age2) →
  (effectiveInsurance age1 ∧ rentalCarFlexibility age1 ∧ employmentOpportunity age1) →
  effectiveInsurance age1 ∧ rentalCarFlexibility age1 ∧ employmentOpportunity age1

theorem driver_license_advantage 
  (eligible : ℕ → Prop)
  (effectiveInsurance : ℕ → Prop)
  (rentalCarFlexibility : ℕ → Prop)
  (employmentOpportunity : ℕ → Prop) :
  AdvantageousReasonsForEarlyLicenseObtaining eligible effectiveInsurance rentalCarFlexibility employmentOpportunity :=
by
  sorry

end driver_license_advantage_l82_82415


namespace average_of_combined_samples_l82_82420

theorem average_of_combined_samples 
  (a : Fin 10 → ℝ)
  (b : Fin 10 → ℝ)
  (ave_a : ℝ := (1 / 10) * (Finset.univ.sum (fun i => a i)))
  (ave_b : ℝ := (1 / 10) * (Finset.univ.sum (fun i => b i)))
  (combined_average : ℝ := (1 / 20) * (Finset.univ.sum (fun i => a i) + Finset.univ.sum (fun i => b i))) :
  combined_average = (1 / 2) * (ave_a + ave_b) := 
  by
    sorry

end average_of_combined_samples_l82_82420


namespace set_intersection_l82_82477

def A : Set ℝ := {1, 2, 3, 4, 5}
def B : Set ℝ := {x | x * (4 - x) < 0}
def C_R_B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem set_intersection :
  A ∩ C_R_B = {1, 2, 3, 4} :=
by
  -- Proof goes here
  sorry

end set_intersection_l82_82477


namespace compute_f_g_at_2_l82_82493

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 4 * x - 1

theorem compute_f_g_at_2 :
  f (g 2) = 49 :=
by
  sorry

end compute_f_g_at_2_l82_82493


namespace shaded_area_l82_82450

theorem shaded_area (r1 r2 : ℝ) (h1 : r2 = 3 * r1) (h2 : r1 = 2) : 
  π * (r2 ^ 2) - π * (r1 ^ 2) = 32 * π :=
by
  sorry

end shaded_area_l82_82450


namespace min_abs_sum_of_x1_x2_l82_82711

open Real

theorem min_abs_sum_of_x1_x2 (x1 x2 : ℝ) (h : 1 / ((2 + sin x1) * (2 + sin (2 * x2))) = 1) : 
  abs (x1 + x2) = π / 4 :=
sorry

end min_abs_sum_of_x1_x2_l82_82711


namespace determine_y_l82_82454

variable {x y : ℝ}
variable (hx : x ≠ 0) (hy : y ≠ 0)
variable (hxy : x = 2 + (1 / y))
variable (hyx : y = 2 + (2 / x))

theorem determine_y (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 2 + (1 / y)) (hyx : y = 2 + (2 / x)) :
  y = (5 + Real.sqrt 41) / 4 ∨ y = (5 - Real.sqrt 41) / 4 := 
sorry

end determine_y_l82_82454


namespace Isabella_total_items_l82_82816

theorem Isabella_total_items (A_pants A_dresses I_pants I_dresses : ℕ) 
  (h1 : A_pants = 3 * I_pants) 
  (h2 : A_dresses = 3 * I_dresses)
  (h3 : A_pants = 21) 
  (h4 : A_dresses = 18) : 
  I_pants + I_dresses = 13 :=
by
  -- Proof goes here
  sorry

end Isabella_total_items_l82_82816


namespace minimum_pencils_l82_82250

-- Define the given conditions
def red_pencils : ℕ := 15
def blue_pencils : ℕ := 13
def green_pencils : ℕ := 8

-- Define the requirement for pencils to ensure the conditions are met
def required_red : ℕ := 1
def required_blue : ℕ := 2
def required_green : ℕ := 3

-- The minimum number of pencils Constanza should take out
noncomputable def minimum_pencils_to_ensure : ℕ := 21 + 1

theorem minimum_pencils (red_pencils blue_pencils green_pencils : ℕ)
    (required_red required_blue required_green minimum_pencils_to_ensure : ℕ) :
    red_pencils = 15 →
    blue_pencils = 13 →
    green_pencils = 8 →
    required_red = 1 →
    required_blue = 2 →
    required_green = 3 →
    minimum_pencils_to_ensure = 22 :=
by
    intros h1 h2 h3 h4 h5 h6
    sorry

end minimum_pencils_l82_82250


namespace temperature_representation_l82_82275

-- Defining the temperature representation problem
def posTemp := 10 -- $10^\circ \mathrm{C}$ above zero
def negTemp := -10 -- $10^\circ \mathrm{C}$ below zero
def aboveZero (temp : Int) : Prop := temp > 0
def belowZero (temp : Int) : Prop := temp < 0

-- The proof statement to be proved using the given conditions
theorem temperature_representation : 
  (aboveZero posTemp → posTemp = 10) ∧ (belowZero negTemp → negTemp = -10) := 
  by
    sorry -- Proof would go here

end temperature_representation_l82_82275


namespace sin_of_cos_in_third_quadrant_ratio_of_trig_functions_l82_82365

-- Proof for Problem 1
theorem sin_of_cos_in_third_quadrant (α : ℝ) 
  (hcos : Real.cos α = -4 / 5)
  (hquad : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -3 / 5 :=
by
  sorry

-- Proof for Problem 2
theorem ratio_of_trig_functions (α : ℝ) 
  (htan : Real.tan α = -3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 7 / 2 :=
by
  sorry

end sin_of_cos_in_third_quadrant_ratio_of_trig_functions_l82_82365


namespace intersection_primes_evens_l82_82200

open Set

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def evens : Set ℕ := {n | n % 2 = 0}
def primes : Set ℕ := {n | is_prime n}

theorem intersection_primes_evens :
  primes ∩ evens = {2} :=
by sorry

end intersection_primes_evens_l82_82200


namespace partnership_total_profit_l82_82052

theorem partnership_total_profit
  (total_capital : ℝ)
  (A_share : ℝ := 1/3)
  (B_share : ℝ := 1/4)
  (C_share : ℝ := 1/5)
  (D_share : ℝ := 1 - (A_share + B_share + C_share))
  (A_profit : ℝ := 805)
  (A_capital : ℝ := total_capital * A_share)
  (total_capital_positive : 0 < total_capital)
  (shares_add_up : A_share + B_share + C_share + D_share = 1) :
  (A_profit / (total_capital * A_share)) * total_capital = 2415 :=
by
  -- Proof will go here.
  sorry

end partnership_total_profit_l82_82052


namespace book_width_l82_82521

noncomputable def phi_conjugate : ℝ := (Real.sqrt 5 - 1) / 2

theorem book_width {w l : ℝ} (h_ratio : w / l = phi_conjugate) (h_length : l = 14) :
  w = 7 * Real.sqrt 5 - 7 :=
by
  sorry

end book_width_l82_82521


namespace quadratic_linear_term_l82_82673

theorem quadratic_linear_term (m : ℝ) 
  (h : 2 * m = 6) : -4 * (x : ℝ) + m * x = -x := by 
  sorry

end quadratic_linear_term_l82_82673


namespace evaluate_expression_l82_82283

theorem evaluate_expression : 
  ∀ (x y : ℕ), x = 3 → y = 2 → (5 * x^(y + 1) + 6 * y^(x + 1) = 231) := by 
  intros x y hx hy
  rw [hx, hy]
  sorry

end evaluate_expression_l82_82283


namespace poly_roots_arith_progression_l82_82607

theorem poly_roots_arith_progression (a b c : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, -- There exist roots x₁, x₂, x₃
    (x₁ + x₃ = 2 * x₂) ∧ -- Roots form an arithmetic progression
    (x₁ * x₂ * x₃ = -c) ∧ -- Roots satisfy polynomial's product condition
    (x₁ + x₂ + x₃ = -a) ∧ -- Roots satisfy polynomial's sum condition
    ((x₁ * x₂) + (x₂ * x₃) + (x₃ * x₁) = b)) -- Roots satisfy polynomial's sum of products condition
  → (2 * a^3 / 27 - a * b / 3 + c = 0) := 
sorry -- proof is not required

end poly_roots_arith_progression_l82_82607


namespace min_value_x_plus_y_l82_82506

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 2) : x + y = 8 :=
sorry

end min_value_x_plus_y_l82_82506


namespace vector_addition_result_l82_82132

-- Definitions based on problem conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (y : ℝ) : ℝ × ℝ := (2, y)

-- The condition that vectors are parallel
def parallel_vectors (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2)

-- The main theorem to prove
theorem vector_addition_result (y : ℝ) (h : parallel_vectors vector_a (vector_b y)) : 
  (vector_a.1 + 2 * (vector_b y).1, vector_a.2 + 2 * (vector_b y).2) = (5, 10) :=
sorry

end vector_addition_result_l82_82132


namespace diff_of_squares_l82_82556

variable {x y : ℝ}

theorem diff_of_squares : (x + y) * (x - y) = x^2 - y^2 := 
sorry

end diff_of_squares_l82_82556


namespace flowers_per_bouquet_l82_82748

theorem flowers_per_bouquet :
  let red_seeds := 125
  let yellow_seeds := 125
  let orange_seeds := 125
  let purple_seeds := 125
  let red_killed := 45
  let yellow_killed := 61
  let orange_killed := 30
  let purple_killed := 40
  let bouquets := 36
  let red_flowers := red_seeds - red_killed
  let yellow_flowers := yellow_seeds - yellow_killed
  let orange_flowers := orange_seeds - orange_killed
  let purple_flowers := purple_seeds - purple_killed
  let total_flowers := red_flowers + yellow_flowers + orange_flowers + purple_flowers
  let flowers_per_bouquet := total_flowers / bouquets
  flowers_per_bouquet = 9 :=
by
  sorry

end flowers_per_bouquet_l82_82748


namespace leggings_needed_l82_82997

theorem leggings_needed (dogs : ℕ) (cats : ℕ) (dogs_legs : ℕ) (cats_legs : ℕ) (pair_of_leggings : ℕ) 
                        (hd : dogs = 4) (hc : cats = 3) (hl1 : dogs_legs = 4) (hl2 : cats_legs = 4) (lp : pair_of_leggings = 2)
                        : (dogs * dogs_legs + cats * cats_legs) / pair_of_leggings = 14 :=
by
  sorry

end leggings_needed_l82_82997


namespace designed_height_correct_l82_82796
noncomputable def designed_height_of_lower_part (H : ℝ) (L : ℝ) : Prop :=
  H = 2 ∧ (H - L) / L = L / H

theorem designed_height_correct : ∃ L, designed_height_of_lower_part 2 L ∧ L = Real.sqrt 5 - 1 :=
by
  sorry

end designed_height_correct_l82_82796


namespace geometric_sequence_sum_inequality_l82_82198

open Classical

variable (a_1 q : ℝ) (h1 : a_1 > 0) (h2 : q > 0) (h3 : q ≠ 1)

theorem geometric_sequence_sum_inequality :
  a_1 + a_1 * q^3 > a_1 * q + a_1 * q^2 :=
by
  sorry

end geometric_sequence_sum_inequality_l82_82198


namespace math_problem_l82_82560

noncomputable def log_8 := Real.log 8
noncomputable def log_27 := Real.log 27
noncomputable def expr := (9 : ℝ) ^ (log_8 / log_27) + (2 : ℝ) ^ (log_27 / log_8)

theorem math_problem : expr = 7 := by
  sorry

end math_problem_l82_82560


namespace mrs_li_actual_birthdays_l82_82292
   
   def is_leap_year (year : ℕ) : Prop :=
     (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)
   
   def num_leap_years (start end_ : ℕ) : ℕ :=
     (start / 4 - start / 100 + start / 400) -
     (end_ / 4 - end_ / 100 + end_ / 400)
   
   theorem mrs_li_actual_birthdays : num_leap_years 1944 2011 = 16 :=
   by
     -- Calculation logic for the proof
     sorry
   
end mrs_li_actual_birthdays_l82_82292


namespace sqrt_mult_simplify_l82_82223

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l82_82223


namespace nathan_tokens_used_is_18_l82_82522

-- We define the conditions as variables and constants
variables (airHockeyGames basketballGames tokensPerGame : ℕ)

-- State the values for the conditions
def Nathan_plays : Prop :=
  airHockeyGames = 2 ∧ basketballGames = 4 ∧ tokensPerGame = 3

-- Calculate the total tokens used
def totalTokensUsed (airHockeyGames basketballGames tokensPerGame : ℕ) : ℕ :=
  (airHockeyGames * tokensPerGame) + (basketballGames * tokensPerGame)

-- Proof statement 
theorem nathan_tokens_used_is_18 : Nathan_plays airHockeyGames basketballGames tokensPerGame → totalTokensUsed airHockeyGames basketballGames tokensPerGame = 18 :=
by 
  sorry

end nathan_tokens_used_is_18_l82_82522


namespace perimeter_difference_l82_82034

-- Define the dimensions of the two figures
def width1 : ℕ := 6
def height1 : ℕ := 3
def width2 : ℕ := 6
def height2 : ℕ := 2

-- Define the perimeters of the two figures
def perimeter1 : ℕ := 2 * (width1 + height1)
def perimeter2 : ℕ := 2 * (width2 + height2)

-- Prove the positive difference in perimeters is 2 units
theorem perimeter_difference : (perimeter1 - perimeter2) = 2 := by
  sorry

end perimeter_difference_l82_82034


namespace perimeter_of_triangle_ABC_l82_82010

-- Define the focal points and their radius
def radius : ℝ := 2

-- Define the distances between centers of the tangent circles
def center_distance : ℝ := 2 * radius

-- Define the lengths of the sides of the triangle ABC based on the problem constraints
def AB : ℝ := 2 * radius + 2 * center_distance
def BC : ℝ := 2 * radius + center_distance
def CA : ℝ := 2 * radius + center_distance

-- Define the perimeter calculation
def perimeter : ℝ := AB + BC + CA

-- Theorem stating the actual perimeter of the triangle ABC
theorem perimeter_of_triangle_ABC : perimeter = 28 := by
  sorry

end perimeter_of_triangle_ABC_l82_82010


namespace double_even_l82_82102

-- Define even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Lean statement of the mathematically equivalent proof problem
theorem double_even (f : ℝ → ℝ) (h : is_even_function f) : is_even_function (f ∘ f) :=
by
  sorry

end double_even_l82_82102


namespace expression_value_l82_82799

def a : ℕ := 45
def b : ℕ := 18
def c : ℕ := 10

theorem expression_value :
  (a + b)^2 - (a^2 + b^2 + c) = 1610 := by
  sorry

end expression_value_l82_82799


namespace second_solution_concentration_l82_82690

def volume1 : ℝ := 5
def concentration1 : ℝ := 0.04
def volume2 : ℝ := 2.5
def concentration_final : ℝ := 0.06
def total_silver1 : ℝ := volume1 * concentration1
def total_volume : ℝ := volume1 + volume2
def total_silver_final : ℝ := total_volume * concentration_final

theorem second_solution_concentration :
  ∃ (C2 : ℝ), total_silver1 + volume2 * C2 = total_silver_final ∧ C2 = 0.1 := 
by 
  sorry

end second_solution_concentration_l82_82690


namespace min_value_hyperbola_l82_82917

open Real 

theorem min_value_hyperbola :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 = 1) → (3 * x^2 - 2 * y ≥ 143 / 12) ∧ 
                                          (∃ (y' : ℝ), y = y' ∧  3 * (2 + 2*y'^2)^2 - 2 * y' = 143 / 12) := 
by
  sorry

end min_value_hyperbola_l82_82917


namespace problem_inequality_l82_82941

variables {a b c x1 x2 x3 x4 x5 : ℝ} 

theorem problem_inequality
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_x1: 0 < x1) (h_pos_x2: 0 < x2) (h_pos_x3: 0 < x3) (h_pos_x4: 0 < x4) (h_pos_x5: 0 < x5)
  (h_sum_abc : a + b + c = 1) (h_prod_x : x1 * x2 * x3 * x4 * x5 = 1) :
  (a * x1^2 + b * x1 + c) * (a * x2^2 + b * x2 + c) * (a * x3^2 + b * x3 + c) * 
  (a * x4^2 + b * x4 + c) * (a * x5^2 + b * x5 + c) ≥ 1 :=
sorry

end problem_inequality_l82_82941


namespace JakePresentWeight_l82_82219

def JakeWeight (J S : ℕ) : Prop :=
  J - 33 = 2 * S ∧ J + S = 153

theorem JakePresentWeight : ∃ (J : ℕ), ∃ (S : ℕ), JakeWeight J S ∧ J = 113 := 
by
  sorry

end JakePresentWeight_l82_82219


namespace boys_trees_l82_82659

theorem boys_trees (avg_per_person trees_per_girl trees_per_boy : ℕ) :
  avg_per_person = 6 →
  trees_per_girl = 15 →
  (1 / trees_per_boy + 1 / trees_per_girl = 1 / avg_per_person) →
  trees_per_boy = 10 :=
by
  intros h_avg h_girl h_eq
  -- We will provide the proof here eventually
  sorry

end boys_trees_l82_82659


namespace cos_theta_value_sin_theta_plus_pi_over_3_value_l82_82058

variable (θ : ℝ)
variable (H1 : 0 < θ ∧ θ < π / 2)
variable (H2 : Real.sin θ = 4 / 5)

theorem cos_theta_value : Real.cos θ = 3 / 5 := sorry

theorem sin_theta_plus_pi_over_3_value : 
    Real.sin (θ + π / 3) = (4 + 3 * Real.sqrt 3) / 10 := sorry

end cos_theta_value_sin_theta_plus_pi_over_3_value_l82_82058


namespace part_a_l82_82482

theorem part_a : 
  ∃ (x y : ℕ → ℕ), (∀ n : ℕ, (1 + Real.sqrt 33) ^ n = x n + y n * Real.sqrt 33) :=
sorry

end part_a_l82_82482


namespace floor_inequality_solution_set_l82_82880

/-- Let ⌊x⌋ denote the greatest integer less than or equal to x.
    Prove that the solution set of the inequality ⌊x⌋² - 5⌊x⌋ - 36 ≤ 0 is {x | -4 ≤ x < 10}. -/
theorem floor_inequality_solution_set (x : ℝ) :
  (⌊x⌋^2 - 5 * ⌊x⌋ - 36 ≤ 0) ↔ -4 ≤ x ∧ x < 10 := by
    sorry

end floor_inequality_solution_set_l82_82880


namespace candy_game_win_l82_82636

def winning_player (A B : ℕ) : String :=
  if (A % B = 0 ∨ B % A = 0) then "Player with forcing checks" else "No inevitable winner"

theorem candy_game_win :
  winning_player 1000 2357 = "Player with forcing checks" :=
by
  sorry

end candy_game_win_l82_82636


namespace problem_statement_l82_82973

theorem problem_statement (x : ℤ) (h₁ : (x - 5) / 7 = 7) : (x - 24) / 10 = 3 := 
sorry

end problem_statement_l82_82973


namespace temperature_fifth_day_l82_82535

variable (T1 T2 T3 T4 T5 : ℝ)

-- Conditions
def condition1 : T1 + T2 + T3 + T4 = 4 * 58 := by sorry
def condition2 : T2 + T3 + T4 + T5 = 4 * 59 := by sorry
def condition3 : T5 = (8 / 7) * T1 := by sorry

-- The statement we need to prove
theorem temperature_fifth_day : T5 = 32 := by
  -- Using the provided conditions
  sorry

end temperature_fifth_day_l82_82535


namespace A_plus_B_eq_one_fourth_l82_82937

noncomputable def A : ℚ := 1 / 3
noncomputable def B : ℚ := -1 / 12

theorem A_plus_B_eq_one_fourth :
  A + B = 1 / 4 := by
  sorry

end A_plus_B_eq_one_fourth_l82_82937


namespace a5_a6_less_than_a4_squared_l82_82205

variable {a : ℕ → ℝ}
variable {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem a5_a6_less_than_a4_squared
  (h_geo : is_geometric_sequence a q)
  (h_cond : a 5 * a 6 < (a 4) ^ 2) :
  0 < q ∧ q < 1 :=
sorry

end a5_a6_less_than_a4_squared_l82_82205


namespace a_sequence_arithmetic_sum_of_bn_l82_82594

   noncomputable def a (n : ℕ) : ℕ := 1 + n

   def S (n : ℕ) : ℕ := n * (n + 1) / 2

   def b (n : ℕ) : ℚ := 1 / S n

   def T (n : ℕ) : ℚ := (Finset.range n).sum b

   theorem a_sequence_arithmetic (n : ℕ) (a_n_positive : ∀ n, a n > 0)
     (a₁_is_one : a 0 = 1) :
     (a (n+1)) - a n = 1 := by
     sorry

   theorem sum_of_bn (n : ℕ) :
     T n = 2 * n / (n + 1) := by
     sorry
   
end a_sequence_arithmetic_sum_of_bn_l82_82594


namespace sin_diff_l82_82142

theorem sin_diff (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1 / 3) 
  (h2 : Real.sin β - Real.cos α = 1 / 2) : 
  Real.sin (α - β) = -59 / 72 := 
sorry

end sin_diff_l82_82142


namespace expand_binomial_l82_82819

theorem expand_binomial (x : ℝ) : (x + 3) * (x - 8) = x^2 - 5 * x - 24 :=
by
  sorry

end expand_binomial_l82_82819


namespace equal_sundays_tuesdays_days_l82_82126

-- Define the problem in Lean
def num_equal_sundays_and_tuesdays_starts : ℕ :=
  3

-- Define a function that calculates the number of starting days that result in equal Sundays and Tuesdays
def calculate_sundays_tuesdays_starts (days_in_month : ℕ) : ℕ :=
  if days_in_month = 30 then 3 else 0

-- Prove that for a month of 30 days, there are 3 valid starting days for equal Sundays and Tuesdays
theorem equal_sundays_tuesdays_days :
  calculate_sundays_tuesdays_starts 30 = num_equal_sundays_and_tuesdays_starts :=
by 
  -- Proof outline here
  sorry

end equal_sundays_tuesdays_days_l82_82126


namespace find_number_of_children_l82_82726

def admission_cost_adult : ℝ := 30
def admission_cost_child : ℝ := 15
def total_people : ℕ := 10
def soda_cost : ℝ := 5
def discount_rate : ℝ := 0.8
def total_paid : ℝ := 197

def total_cost_with_discount (adults children : ℕ) : ℝ :=
  discount_rate * (adults * admission_cost_adult + children * admission_cost_child)

theorem find_number_of_children (A C : ℕ) 
  (h1 : A + C = total_people)
  (h2 : total_cost_with_discount A C + soda_cost = total_paid) :
  C = 4 :=
sorry

end find_number_of_children_l82_82726


namespace mean_value_of_pentagon_angles_l82_82621

theorem mean_value_of_pentagon_angles : 
  let n := 5 
  let interior_angle_sum := (n - 2) * 180 
  mean_angle = interior_angle_sum / n :=
  sorry

end mean_value_of_pentagon_angles_l82_82621


namespace smaller_angle_in_parallelogram_l82_82652

theorem smaller_angle_in_parallelogram 
  (opposite_angles : ∀ A B C D : ℝ, A = C ∧ B = D)
  (adjacent_angles_supplementary : ∀ A B : ℝ, A + B = π)
  (angle_diff : ∀ A B : ℝ, B = A + π/9) :
  ∃ θ : ℝ, θ = 4 * π / 9 :=
by
  sorry

end smaller_angle_in_parallelogram_l82_82652


namespace arithmetic_sequence_term_number_l82_82878

theorem arithmetic_sequence_term_number :
  ∀ (a : ℕ → ℤ) (n : ℕ),
    (a 1 = 1) →
    (∀ m, a (m + 1) = a m + 3) →
    (a n = 2014) →
    n = 672 :=
by
  -- conditions
  intro a n h1 h2 h3
  -- proof skipped
  sorry

end arithmetic_sequence_term_number_l82_82878


namespace benzene_molecular_weight_l82_82003

theorem benzene_molecular_weight (w: ℝ) (h: 4 * w = 312) : w = 78 :=
by
  sorry

end benzene_molecular_weight_l82_82003


namespace john_new_weekly_earnings_l82_82595

theorem john_new_weekly_earnings :
  let original_earnings : ℝ := 40
  let percentage_increase : ℝ := 37.5 / 100
  let raise_amount : ℝ := original_earnings * percentage_increase
  let new_weekly_earnings : ℝ := original_earnings + raise_amount
  new_weekly_earnings = 55 := 
by
  sorry

end john_new_weekly_earnings_l82_82595


namespace center_of_circle_sum_l82_82468

open Real

theorem center_of_circle_sum (x y : ℝ) (h k : ℝ) :
  (x - h)^2 + (y - k)^2 = 2 → (h = 3) → (k = 4) → h + k = 7 :=
by
  intro h_eq k_eq
  sorry

end center_of_circle_sum_l82_82468


namespace quadratic_equation_conditions_l82_82664

theorem quadratic_equation_conditions :
  ∃ (a b c : ℝ), a = 3 ∧ c = 1 ∧ (a * x^2 + b * x + c = 0 ↔ 3 * x^2 + 1 = 0) :=
by
  use 3, 0, 1
  sorry

end quadratic_equation_conditions_l82_82664


namespace benny_bought_books_l82_82749

theorem benny_bought_books :
  ∀ (initial_books sold_books remaining_books bought_books : ℕ),
    initial_books = 22 →
    sold_books = initial_books / 2 →
    remaining_books = initial_books - sold_books →
    remaining_books + bought_books = 17 →
    bought_books = 6 :=
by
  intros initial_books sold_books remaining_books bought_books
  sorry

end benny_bought_books_l82_82749


namespace david_age_l82_82788

theorem david_age (A B C D : ℕ)
  (h1 : A = B - 5)
  (h2 : B = C + 2)
  (h3 : D = C + 4)
  (h4 : A = 12) : D = 19 :=
sorry

end david_age_l82_82788


namespace find_b_l82_82555

noncomputable def Q (x : ℝ) (a b c : ℝ) := 3 * x ^ 3 + a * x ^ 2 + b * x + c

theorem find_b (a b c : ℝ) (h₀ : c = 6) 
  (h₁ : ∃ (r₁ r₂ r₃ : ℝ), Q r₁ a b c = 0 ∧ Q r₂ a b c = 0 ∧ Q r₃ a b c = 0 ∧ (r₁ + r₂ + r₃) / 3 = -(c / 3) ∧ r₁ * r₂ * r₃ = -(c / 3))
  (h₂ : 3 + a + b + c = -(c / 3)): 
  b = -29 :=
sorry

end find_b_l82_82555


namespace cylinder_volume_increase_l82_82113

theorem cylinder_volume_increase 
  (r h : ℝ) 
  (V : ℝ := π * r^2 * h) 
  (new_h : ℝ := 3 * h) 
  (new_r : ℝ := 2 * r) : 
  (π * new_r^2 * new_h) = 12 * V := 
by
  sorry

end cylinder_volume_increase_l82_82113


namespace bridge_length_l82_82870

noncomputable def train_length : ℝ := 250 -- in meters
noncomputable def train_speed_kmh : ℝ := 60 -- in km/hr
noncomputable def crossing_time : ℝ := 20 -- in seconds

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600 -- converting to m/s

noncomputable def total_distance_covered : ℝ := train_speed_ms * crossing_time -- distance covered in 20 seconds

theorem bridge_length : total_distance_covered - train_length = 83.4 :=
by
  -- The proof would go here
  sorry

end bridge_length_l82_82870


namespace polygon_sides_from_interior_angles_l82_82262

theorem polygon_sides_from_interior_angles (S : ℕ) (h : S = 1260) : S = (9 - 2) * 180 :=
by
  sorry

end polygon_sides_from_interior_angles_l82_82262


namespace intersecting_graphs_l82_82046

theorem intersecting_graphs (a b c d : ℝ) 
  (h1 : -2 * |1 - a| + b = 4) 
  (h2 : 2 * |1 - c| + d = 4)
  (h3 : -2 * |7 - a| + b = 0) 
  (h4 : 2 * |7 - c| + d = 0) : a + c = 10 := 
sorry

end intersecting_graphs_l82_82046


namespace m_value_l82_82321

theorem m_value (A : Set ℝ) (B : Set ℝ) (m : ℝ) 
                (hA : A = {0, 1, 2}) 
                (hB : B = {1, m}) 
                (h_subset : B ⊆ A) : 
                m = 0 ∨ m = 2 :=
by
  sorry

end m_value_l82_82321


namespace total_outfits_l82_82554

-- Define the number of shirts, pants, ties (including no-tie option), and shoes as given in the conditions.
def num_shirts : ℕ := 5
def num_pants : ℕ := 4
def num_ties : ℕ := 6 -- 5 ties + 1 no-tie option
def num_shoes : ℕ := 2

-- Proof statement: The total number of different outfits is 240.
theorem total_outfits : num_shirts * num_pants * num_ties * num_shoes = 240 :=
by
  sorry

end total_outfits_l82_82554


namespace arithmetic_sequence_sum_l82_82695

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (hS3 : S 3 = 12) (hS6 : S 6 = 42) 
  (h_arith_seq : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) :
  a 10 + a 11 + a 12 = 66 :=
sorry

end arithmetic_sequence_sum_l82_82695


namespace spherical_to_rectangular_coords_l82_82445

theorem spherical_to_rectangular_coords :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = 5 * Real.sin (Real.pi / 3) * Real.cos (Real.pi / 4) ∧
  y = 5 * Real.sin (Real.pi / 3) * Real.sin (Real.pi / 4) ∧
  z = 5 * Real.cos (Real.pi / 3) ∧
  x = 5 * (Real.sqrt 3) / 2 * (Real.sqrt 2) / 2 ∧
  y = 5 * (Real.sqrt 3) / 2 * (Real.sqrt 2) / 2 ∧
  z = 2.5 ∧
  (x = (5 * Real.sqrt 6) / 4 ∧ y = (5 * Real.sqrt 6) / 4 ∧ z = 2.5) :=
by {
  sorry
}

end spherical_to_rectangular_coords_l82_82445


namespace smallest_10_digit_number_with_sum_81_l82_82504

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

theorem smallest_10_digit_number_with_sum_81 {n : Nat} :
  n ≥ 1000000000 ∧ n < 10000000000 ∧ sum_of_digits n ≥ 81 → 
  n = 1899999999 :=
sorry

end smallest_10_digit_number_with_sum_81_l82_82504


namespace necessary_and_sufficient_condition_l82_82419

variable (p q : Prop)

theorem necessary_and_sufficient_condition (h1 : p → q) (h2 : q → p) : (p ↔ q) :=
by 
  sorry

end necessary_and_sufficient_condition_l82_82419


namespace triangle_area_l82_82927

noncomputable def area_of_triangle := 
  let a := 4
  let b := 5
  let c := 6
  let cosA := 3 / 4
  let sinA := Real.sqrt (1 - cosA ^ 2)
  (1 / 2) * b * c * sinA

theorem triangle_area :
  ∃ (a b c : ℝ), a = 4 ∧ b = 5 ∧ c = 6 ∧ 
  a < b ∧ b < c ∧ 
  -- Additional conditions
  (∃ A B C : ℝ, C = 2 * A ∧ 
   Real.cos A = 3 / 4 ∧ 
   Real.sin A * Real.cos A = sinA * cosA ∧ 
   0 < A ∧ A < Real.pi ∧ 
   (1 / 2) * b * c * sinA = (15 * Real.sqrt 7) / 4) :=
by
  sorry

end triangle_area_l82_82927


namespace original_money_l82_82041

theorem original_money (M : ℕ) (h1 : 3 * M / 8 ≤ M)
  (h2 : 1 * (M - 3 * M / 8) / 5 ≤ M - 3 * M / 8)
  (h3 : M - 3 * M / 8 - (1 * (M - 3 * M / 8) / 5) = 36) : M = 72 :=
sorry

end original_money_l82_82041


namespace min_value_AP_AQ_l82_82638

noncomputable def min_distance (A P Q : ℝ × ℝ) : ℝ := dist A P + dist A Q

theorem min_value_AP_AQ :
  ∀ (A P Q : ℝ × ℝ),
    (∀ (x : ℝ), A = (x, 0)) →
    ((P.1 - 1) ^ 2 + (P.2 - 3) ^ 2 = 1) →
    ((Q.1 - 7) ^ 2 + (Q.2 - 5) ^ 2 = 4) →
    min_distance A P Q = 7 :=
by
  intros A P Q hA hP hQ
  -- Proof is to be provided here
  sorry

end min_value_AP_AQ_l82_82638


namespace Lindsay_has_26_more_black_brown_dolls_than_blonde_l82_82178

def blonde_dolls : Nat := 4
def brown_dolls : Nat := 4 * blonde_dolls
def black_dolls : Nat := brown_dolls - 2
def total_black_brown_dolls : Nat := black_dolls + brown_dolls
def extra_black_brown_dolls (blonde_dolls black_dolls brown_dolls : Nat) : Nat :=
  total_black_brown_dolls - blonde_dolls

theorem Lindsay_has_26_more_black_brown_dolls_than_blonde :
  extra_black_brown_dolls blonde_dolls black_dolls brown_dolls = 26 := by
  sorry

end Lindsay_has_26_more_black_brown_dolls_than_blonde_l82_82178


namespace donuts_selection_l82_82439

def number_of_selections (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

theorem donuts_selection : number_of_selections 6 4 = 84 := by
  sorry

end donuts_selection_l82_82439


namespace polynomial_factors_sum_l82_82709

open Real

theorem polynomial_factors_sum
  (a b c : ℝ)
  (h1 : ∀ x, (x^2 + x + 2) * (a * x + b - a) + (c - a - b) * x + 5 + 2 * a - 2 * b = 0)
  (h2 : a * (1/2)^3 + b * (1/2)^2 + c * (1/2) - 25/16 = 0) :
  a + b + c = 45 / 11 :=
by
  sorry

end polynomial_factors_sum_l82_82709


namespace age_solution_l82_82082

noncomputable def age_problem : Prop :=
  ∃ (A B x : ℕ),
    A = B + 5 ∧
    A + B = 13 ∧
    3 * (A + x) = 4 * (B + x) ∧
    x = 11

theorem age_solution : age_problem :=
  sorry

end age_solution_l82_82082


namespace mika_saucer_surface_area_l82_82820

noncomputable def surface_area_saucer (r h rim_thickness : ℝ) : ℝ :=
  let A_cap := 2 * Real.pi * r * h  -- Surface area of the spherical cap
  let R_outer := r
  let R_inner := r - rim_thickness
  let A_rim := Real.pi * (R_outer^2 - R_inner^2)  -- Area of the rim
  A_cap + A_rim

theorem mika_saucer_surface_area :
  surface_area_saucer 3 1.5 1 = 14 * Real.pi :=
sorry

end mika_saucer_surface_area_l82_82820


namespace simplify_expression_l82_82105

theorem simplify_expression (a b : ℝ) (h : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b - a^2) / (b^2 - a^2) =
  (a^3 - 3 * a * b^2 + 2 * b^3) / (a * b * (b + a)) :=
by
  sorry

end simplify_expression_l82_82105


namespace david_course_hours_l82_82751

def total_course_hours (weeks : ℕ) (class_hours_per_week : ℕ) (homework_hours_per_week : ℕ) : ℕ :=
  weeks * (class_hours_per_week + homework_hours_per_week)

theorem david_course_hours :
  total_course_hours 24 (3 + 3 + 4) 4 = 336 :=
by
  sorry

end david_course_hours_l82_82751


namespace find_radioactive_balls_within_7_checks_l82_82464

theorem find_radioactive_balls_within_7_checks :
  ∃ (balls : Finset α), balls.card = 11 ∧ ∃ radioactive_balls ⊆ balls, radioactive_balls.card = 2 ∧
  (∀ (check : Finset α → Prop), (∀ S, check S ↔ (∃ b ∈ S, b ∈ radioactive_balls)) →
  ∃ checks : Finset (Finset α), checks.card ≤ 7 ∧ (∀ b ∈ radioactive_balls, ∃ S ∈ checks, b ∈ S)) :=
sorry

end find_radioactive_balls_within_7_checks_l82_82464


namespace parallel_lines_slope_eq_l82_82734

theorem parallel_lines_slope_eq (k : ℝ) : (∀ x : ℝ, 3 = 6 * k) → k = 1 / 2 :=
by
  intro h
  sorry

end parallel_lines_slope_eq_l82_82734


namespace find_a_l82_82492

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 + 4 * x^2 + 3 * x

-- Define the derivative of function f with respect to x
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 8 * x + 3

-- Define the condition for the problem
def condition (a : ℝ) : Prop := f' a 1 = 2

-- The statement to be proved
theorem find_a (a : ℝ) (h : condition a) : a = -3 :=
by {
  -- Proof is omitted
  sorry
}

end find_a_l82_82492


namespace original_ratio_l82_82756

namespace OilBill

-- Definitions based on conditions
def JanuaryBill : ℝ := 179.99999999999991

def FebruaryBillWith30More (F : ℝ) : Prop := 
  3 * (F + 30) = 900

-- Statement of the problem proving the original ratio
theorem original_ratio (F : ℝ) (hF : FebruaryBillWith30More F) : 
  F / JanuaryBill = 3 / 2 :=
by
  -- This will contain the proof steps
  sorry

end OilBill

end original_ratio_l82_82756


namespace jogged_distance_is_13_point_5_l82_82771

noncomputable def jogger_distance (x t d : ℝ) : Prop :=
  d = x * t ∧
  d = (x + 3/4) * (3 * t / 4) ∧
  d = (x - 3/4) * (t + 3)

theorem jogged_distance_is_13_point_5:
  ∃ (x t d : ℝ), jogger_distance x t d ∧ d = 13.5 :=
by
  sorry

end jogged_distance_is_13_point_5_l82_82771


namespace Kim_min_score_for_target_l82_82953

noncomputable def Kim_exam_scores : List ℚ := [86, 82, 89]

theorem Kim_min_score_for_target :
  ∃ x : ℚ, ↑((Kim_exam_scores.sum + x) / (Kim_exam_scores.length + 1) ≥ (Kim_exam_scores.sum / Kim_exam_scores.length) + 2)
  ∧ x = 94 := sorry

end Kim_min_score_for_target_l82_82953


namespace find_second_number_l82_82543

def problem (a b c d : ℚ) : Prop :=
  a + b + c + d = 280 ∧
  a = 2 * b ∧
  c = 2 / 3 * a ∧
  d = b + c

theorem find_second_number (a b c d : ℚ) (h : problem a b c d) : b = 52.5 :=
by
  -- Proof will go here.
  sorry

end find_second_number_l82_82543


namespace num_perfect_squares_mul_36_lt_10pow8_l82_82852

theorem num_perfect_squares_mul_36_lt_10pow8 : 
  ∃(n : ℕ), n = 1666 ∧ 
  ∀ (N : ℕ), (1 ≤ N) → (N^2 < 10^8) → (N^2 % 36 = 0) → 
  (N ≤ 9996 ∧ N % 6 = 0) :=
by
  sorry

end num_perfect_squares_mul_36_lt_10pow8_l82_82852


namespace cows_grazed_by_C_l82_82286

-- Define the initial conditions as constants
def cows_grazed_A : ℕ := 24
def months_grazed_A : ℕ := 3
def cows_grazed_B : ℕ := 10
def months_grazed_B : ℕ := 5
def cows_grazed_D : ℕ := 21
def months_grazed_D : ℕ := 3
def share_rent_A : ℕ := 1440
def total_rent : ℕ := 6500

-- Define the cow-months calculation for A, B, D
def cow_months_A : ℕ := cows_grazed_A * months_grazed_A
def cow_months_B : ℕ := cows_grazed_B * months_grazed_B
def cow_months_D : ℕ := cows_grazed_D * months_grazed_D

-- Let x be the number of cows grazed by C
variable (x : ℕ)

-- Define the cow-months calculation for C
def cow_months_C : ℕ := x * 4

-- Define rent per cow-month
def rent_per_cow_month : ℕ := share_rent_A / cow_months_A

-- Proof problem statement
theorem cows_grazed_by_C : 
  (6500 = (cow_months_A + cow_months_B + cow_months_C x + cow_months_D) * rent_per_cow_month) →
  x = 35 := by
  sorry

end cows_grazed_by_C_l82_82286


namespace total_price_of_hats_l82_82680

variables (total_hats : ℕ) (blue_hat_cost : ℕ) (green_hat_cost : ℕ) (green_hats : ℕ) (total_price : ℕ)

def total_number_of_hats := 85
def cost_per_blue_hat := 6
def cost_per_green_hat := 7
def number_of_green_hats := 30

theorem total_price_of_hats :
  (number_of_green_hats * cost_per_green_hat) + ((total_number_of_hats - number_of_green_hats) * cost_per_blue_hat) = 540 :=
sorry

end total_price_of_hats_l82_82680


namespace inequality_of_abc_l82_82302

theorem inequality_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 :=
sorry

end inequality_of_abc_l82_82302


namespace max_even_integers_for_odd_product_l82_82674

theorem max_even_integers_for_odd_product (a b c d e f g : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) (h7 : 0 < g) 
  (h_prod_odd : a * b * c * d * e * f * g % 2 = 1) : a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1 ∧ f % 2 = 1 ∧ g % 2 = 1 :=
sorry

end max_even_integers_for_odd_product_l82_82674


namespace num_pairs_eq_12_l82_82887

theorem num_pairs_eq_12 :
  ∃ (n : ℕ), (∀ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b ≤ 100 ∧
    (a + 1/b : ℚ) / (1/a + b : ℚ) = 7 ↔ (7 * b = a)) ∧ n = 12 :=
sorry

end num_pairs_eq_12_l82_82887


namespace slopes_product_no_circle_MN_A_l82_82721

-- Define the equation of the ellipse E and the specific points A and B
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the point P which lies on the ellipse
def P (x0 y0 : ℝ) : Prop := ellipse_eq x0 y0 ∧ x0 ≠ -2 ∧ x0 ≠ 2

-- Prove the product of the slopes of lines PA and PB
theorem slopes_product (x0 y0 : ℝ) (hP : P x0 y0) : 
  (y0 / (x0 + 2)) * (y0 / (x0 - 2)) = -1 / 4 := sorry

-- Define point Q
def Q : ℝ × ℝ := (-1, 0)

-- Define points M and N which are intersections of line and ellipse
def MN_line (t y : ℝ) : ℝ := t * y - 1

-- Prove there is no circle with diameter MN passing through A
theorem no_circle_MN_A (t : ℝ) : 
  ¬ ∃ M N : ℝ × ℝ, ellipse_eq M.1 M.2 ∧ ellipse_eq N.1 N.2 ∧
  (∃ x1 y1 x2 y2, (M = (x1, y1) ∧ N = (x2, y2)) ∧
  (MN_line t y1 = x1 ∧ MN_line t y2 = x2) ∧ 
  ((x1 + 2) * (x2 + 2) + y1 * y2 = 0)) := sorry

end slopes_product_no_circle_MN_A_l82_82721


namespace probability_is_4_over_5_l82_82372

variable (total_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ)
variable (total_balls_eq : total_balls = 60) (red_balls_eq : red_balls = 5) (purple_balls_eq : purple_balls = 7)

def probability_neither_red_nor_purple : ℚ :=
  let favorable_outcomes := total_balls - (red_balls + purple_balls)
  let total_outcomes := total_balls
  favorable_outcomes / total_outcomes

theorem probability_is_4_over_5 :
  probability_neither_red_nor_purple total_balls red_balls purple_balls = 4 / 5 :=
by
  have h1: total_balls = 60 := total_balls_eq
  have h2: red_balls = 5 := red_balls_eq
  have h3: purple_balls = 7 := purple_balls_eq
  sorry

end probability_is_4_over_5_l82_82372


namespace exists_n_for_perfect_square_l82_82789

theorem exists_n_for_perfect_square (k : ℕ) (hk_pos : k > 0) :
  ∃ n : ℕ, n > 0 ∧ ∃ a : ℕ, a^2 = n * 2^k - 7 :=
by
  sorry

end exists_n_for_perfect_square_l82_82789


namespace select_more_stable_athlete_l82_82896

-- Define the problem conditions
def athlete_average_score : ℝ := 9
def athlete_A_variance : ℝ := 1.2
def athlete_B_variance : ℝ := 2.4

-- Define what it means to have more stable performance
def more_stable (variance_A variance_B : ℝ) : Prop := variance_A < variance_B

-- The theorem to prove
theorem select_more_stable_athlete :
  more_stable athlete_A_variance athlete_B_variance →
  "A" = "A" :=
by
  sorry

end select_more_stable_athlete_l82_82896


namespace angle_K_is_72_l82_82008

variables {J K L M : ℝ}

/-- Given that $JKLM$ is a trapezoid with parallel sides $\overline{JK}$ and $\overline{LM}$,
and given $\angle J = 3\angle M$, $\angle L = 2\angle K$, $\angle J + \angle K = 180^\circ$,
and $\angle L + \angle M = 180^\circ$, prove that $\angle K = 72^\circ$. -/
theorem angle_K_is_72 {J K L M : ℝ}
  (h1 : J = 3 * M)
  (h2 : L = 2 * K)
  (h3 : J + K = 180)
  (h4 : L + M = 180) :
  K = 72 :=
by
  sorry

end angle_K_is_72_l82_82008


namespace cost_of_painting_murals_l82_82909

def first_mural_area : ℕ := 20 * 15
def second_mural_area : ℕ := 25 * 10
def third_mural_area : ℕ := 30 * 8

def first_mural_time : ℕ := first_mural_area * 20
def second_mural_time : ℕ := second_mural_area * 25
def third_mural_time : ℕ := third_mural_area * 30

def total_time : ℚ := (first_mural_time + second_mural_time + third_mural_time) / 60

def total_area : ℕ := first_mural_area + second_mural_area + third_mural_area

def cost (area : ℕ) : ℚ :=
  if area <= 100 then area * 150 else 
  if area <= 300 then 100 * 150 + (area - 100) * 175 
  else 100 * 150 + 200 * 175 + (area - 300) * 200

def total_cost : ℚ := cost total_area

theorem cost_of_painting_murals :
  total_cost = 148000 := by
  sorry

end cost_of_painting_murals_l82_82909


namespace sum_prime_factors_of_77_l82_82930

theorem sum_prime_factors_of_77 : ∃ (p1 p2 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ 77 = p1 * p2 ∧ p1 + p2 = 18 := by
  sorry

end sum_prime_factors_of_77_l82_82930


namespace find_b_l82_82174

noncomputable def triangle_b_value (a : ℝ) (C : ℝ) (area : ℝ) : ℝ :=
  let sin_C := Real.sin C
  let b := (2 * area) / (a * sin_C)
  b

theorem find_b (h₁ : a = 1)
              (h₂ : C = Real.pi / 4)
              (h₃ : area = 2 * a) :
              triangle_b_value a C area = 8 * Real.sqrt 2 :=
by
  -- Definitions imply what we need
  sorry

end find_b_l82_82174


namespace second_polygon_sides_l82_82966

-- Conditions as definitions
def perimeter_first_polygon (s : ℕ) := 50 * (3 * s)
def perimeter_second_polygon (N s : ℕ) := N * s
def same_perimeter (s N : ℕ) := perimeter_first_polygon s = perimeter_second_polygon N s

-- Theorem statement
theorem second_polygon_sides (s N : ℕ) :
  same_perimeter s N → N = 150 :=
by
  sorry

end second_polygon_sides_l82_82966


namespace part3_l82_82603

noncomputable def f (x a : ℝ) : ℝ := x^2 - (2*a + 1)*x + a * Real.log x

theorem part3 (a : ℝ) : 
  (∀ x > 1, f x a > 0) ↔ a ∈ Set.Iic 0 := 
sorry

end part3_l82_82603


namespace base7_to_base10_245_l82_82812

theorem base7_to_base10_245 : (2 * 7^2 + 4 * 7^1 + 5 * 7^0) = 131 := by
  sorry

end base7_to_base10_245_l82_82812


namespace arcsin_of_neg_one_l82_82056

theorem arcsin_of_neg_one : Real.arcsin (-1) = -Real.pi / 2 :=
by
  sorry

end arcsin_of_neg_one_l82_82056


namespace minimum_value_of_c_l82_82442

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  (Real.sqrt 3 / 12) * (a^2 + b^2 - c^2)

noncomputable def tan_formula (a b c B : ℝ) : Prop :=
  24 * (b * c - a) = b * Real.tan B

noncomputable def min_value_c (a b c : ℝ) : ℝ :=
  (2 * Real.sqrt 3) / 3

theorem minimum_value_of_c (a b c B : ℝ) (h₁ : 0 < B ∧ B < π / 2) (h₂ : 24 * (b * c - a) = b * Real.tan B)
  (h₃ : triangle_area a b c = (1/2) * a * b * Real.sin (π / 6)) :
  c ≥ min_value_c a b c :=
by
  sorry

end minimum_value_of_c_l82_82442


namespace find_g_3_l82_82458

theorem find_g_3 (p q r : ℝ) (g : ℝ → ℝ) (h1 : g x = p * x^7 + q * x^3 + r * x + 7) (h2 : g (-3) = -11) (h3 : ∀ x, g (x) + g (-x) = 14) : g 3 = 25 :=
by 
  sorry

end find_g_3_l82_82458


namespace derivative_value_at_pi_over_2_l82_82715

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem derivative_value_at_pi_over_2 : deriv f (Real.pi / 2) = -1 :=
by
  sorry

end derivative_value_at_pi_over_2_l82_82715


namespace cos_B_eq_find_b_eq_l82_82312

variable (A B C a b c : ℝ)

-- Given conditions
axiom sin_A_plus_C_eq : Real.sin (A + C) = 8 * Real.sin (B / 2) ^ 2
axiom a_plus_c : a + c = 6
axiom area_of_triangle : 1 / 2 * a * c * Real.sin B = 2

-- Proving cos B
theorem cos_B_eq :
  Real.cos B = 15 / 17 :=
sorry

-- Proving b given the area and sides condition
theorem find_b_eq :
  Real.cos B = 15 / 17 → b = 2 :=
sorry

end cos_B_eq_find_b_eq_l82_82312


namespace initial_men_l82_82575

variable (P M : ℕ) -- P represents the provisions and M represents the initial number of men.

-- Conditons
def provision_lasts_20_days : Prop := P / (M * 20) = P / ((M + 200) * 15)

-- The proof problem
theorem initial_men (h : provision_lasts_20_days P M) : M = 600 :=
sorry

end initial_men_l82_82575


namespace a5_a6_val_l82_82882

variable (a : ℕ → ℝ)
variable (r : ℝ)

axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom pos_seq (n : ℕ) : a n > 0

axiom a1_a2 : a 1 + a 2 = 1
axiom a3_a4 : a 3 + a 4 = 9

theorem a5_a6_val :
  a 5 + a 6 = 81 :=
by
  sorry

end a5_a6_val_l82_82882


namespace negation_proposition_l82_82579

theorem negation_proposition : 
  ¬(∀ x : ℝ, 0 ≤ x → 2^x > x^2) ↔ ∃ x : ℝ, 0 ≤ x ∧ 2^x ≤ x^2 := by
  sorry

end negation_proposition_l82_82579


namespace find_c_value_l82_82012

theorem find_c_value (a b : ℝ) (h1 : 12 = (6 / 100) * a) (h2 : 6 = (12 / 100) * b) : b / a = 0.25 :=
by
  sorry

end find_c_value_l82_82012


namespace find_alpha_l82_82116

theorem find_alpha (α : ℝ) (h0 : 0 ≤ α) (h1 : α < 360)
    (h_point : (Real.sin 215) = (Real.sin α) ∧ (Real.cos 215) = (Real.cos α)) :
    α = 235 :=
sorry

end find_alpha_l82_82116


namespace sum_of_integers_l82_82832

theorem sum_of_integers (n m : ℕ) (h1 : n * (n + 1) = 120) (h2 : (m - 1) * m * (m + 1) = 120) : 
  (n + (n + 1) + (m - 1) + m + (m + 1)) = 36 :=
by
  sorry

end sum_of_integers_l82_82832


namespace sphere_radius_l82_82742

theorem sphere_radius (r : ℝ) (π : ℝ)
    (h1 : Volume = (4 / 3) * π * r^3)
    (h2 : SurfaceArea = 4 * π * r^2)
    (h3 : Volume = SurfaceArea) :
    r = 3 :=
by
  -- Here starts the proof, but we use 'sorry' to skip it as per the instructions.
  sorry

end sphere_radius_l82_82742


namespace keep_oranges_per_day_l82_82437

def total_oranges_harvested (sacks_per_day : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  sacks_per_day * oranges_per_sack

def oranges_discarded (discarded_sacks : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  discarded_sacks * oranges_per_sack

def oranges_kept_per_day (total_oranges : ℕ) (discarded_oranges : ℕ) : ℕ :=
  total_oranges - discarded_oranges

theorem keep_oranges_per_day 
  (sacks_per_day : ℕ)
  (oranges_per_sack : ℕ)
  (discarded_sacks : ℕ)
  (h1 : sacks_per_day = 76)
  (h2 : oranges_per_sack = 50)
  (h3 : discarded_sacks = 64) :
  oranges_kept_per_day (total_oranges_harvested sacks_per_day oranges_per_sack) 
  (oranges_discarded discarded_sacks oranges_per_sack) = 600 :=
by
  sorry

end keep_oranges_per_day_l82_82437


namespace KimSweaterTotal_l82_82520

theorem KimSweaterTotal :
  let monday := 8
  let tuesday := monday + 2
  let wednesday := tuesday - 4
  let thursday := wednesday
  let friday := monday / 2
  monday + tuesday + wednesday + thursday + friday = 34 := by
  sorry

end KimSweaterTotal_l82_82520


namespace equivalent_problem_l82_82984

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x^2 else sorry

theorem equivalent_problem 
  (f_odd : ∀ x : ℝ, f (-x) = -f x)
  (f_periodic : ∀ x : ℝ, f (x + 2) = f x)
  (f_interval : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x^2)
  : f (-3/2) + f 1 = 3/4 :=
sorry

end equivalent_problem_l82_82984


namespace sum_of_reciprocals_of_roots_l82_82436

theorem sum_of_reciprocals_of_roots (r1 r2 : ℝ) (h1 : r1 * r2 = 7) (h2 : r1 + r2 = 16) :
  (1 / r1) + (1 / r2) = 16 / 7 :=
by
  sorry

end sum_of_reciprocals_of_roots_l82_82436


namespace smallest_value_of_x_l82_82066

theorem smallest_value_of_x : ∃ x, (2 * x^2 + 30 * x - 84 = x * (x + 15)) ∧ (∀ y, (2 * y^2 + 30 * y - 84 = y * (y + 15)) → x ≤ y) ∧ x = -28 := by
  sorry

end smallest_value_of_x_l82_82066


namespace fraction_is_three_eighths_l82_82081

theorem fraction_is_three_eighths (F N : ℝ) 
  (h1 : (4 / 5) * F * N = 24) 
  (h2 : (250 / 100) * N = 199.99999999999997) : 
  F = 3 / 8 :=
by 
  sorry

end fraction_is_three_eighths_l82_82081


namespace geometric_sequence_inserted_product_l82_82299

theorem geometric_sequence_inserted_product :
  ∃ (a b c : ℝ), a * b * c = 216 ∧
    (∃ (q : ℝ), 
      a = (8/3) * q ∧ 
      b = a * q ∧ 
      c = b * q ∧ 
      (8/3) * q^4 = 27/2) :=
sorry

end geometric_sequence_inserted_product_l82_82299


namespace line_eq_form_l82_82128

def line_equation (x y : ℝ) : Prop :=
  ((3 : ℝ) * (x - 2) - (4 : ℝ) * (y + 3) = 0)

theorem line_eq_form (x y : ℝ) (h : line_equation x y) :
  ∃ (m b : ℝ), y = m * x + b ∧ (m = 3/4 ∧ b = -9/2) :=
by
  sorry

end line_eq_form_l82_82128


namespace ac_plus_bd_eq_neg_10_l82_82596

theorem ac_plus_bd_eq_neg_10 (a b c d : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 3)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 6) :
  a * c + b * d = -10 :=
by
  sorry

end ac_plus_bd_eq_neg_10_l82_82596


namespace correct_statement_dice_roll_l82_82120

theorem correct_statement_dice_roll :
  (∃! s, s ∈ ["When flipping a coin, the head side will definitely face up.",
              "The probability of precipitation tomorrow is 80% means that 80% of the areas will have rain tomorrow.",
              "To understand the lifespan of a type of light bulb, it is appropriate to use a census method.",
              "When rolling a dice, the number will definitely not be greater than 6."] ∧
          s = "When rolling a dice, the number will definitely not be greater than 6.") :=
by {
  sorry
}

end correct_statement_dice_roll_l82_82120


namespace hexagon_ratio_l82_82746

theorem hexagon_ratio (A B : ℝ) (h₁ : A = 8) (h₂ : B = 2)
                      (A_above : ℝ) (h₃ : A_above = (3 + B))
                      (H : 3 + B = 1 / 2 * (A + B)) 
                      (XQ QY : ℝ) (h₄ : XQ + QY = 4)
                      (h₅ : 3 + B = 4 + B / 2) :
  XQ / QY = 2 := 
by
  sorry

end hexagon_ratio_l82_82746


namespace solve_for_m_l82_82975

namespace ProofProblem

def f (x m : ℝ) : ℝ := x^2 - 3*x + m
def g (x m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem solve_for_m (m : ℝ) : 3 * f 3 m = g 3 m → m = 0 := by
  sorry

end ProofProblem

end solve_for_m_l82_82975


namespace polygon_sides_l82_82005

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 4 * 360) : 
  n = 6 :=
by sorry

end polygon_sides_l82_82005


namespace find_missing_number_l82_82577

theorem find_missing_number (x : ℝ) (h : 1 / ((1 / 0.03) + (1 / x)) = 0.02775) : abs (x - 0.370) < 0.001 := by
  sorry

end find_missing_number_l82_82577


namespace largest_n_exists_unique_k_l82_82772

theorem largest_n_exists_unique_k (n k : ℕ) :
  (∃! k, (8 : ℚ) / 15 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 7 / 13) →
  n ≤ 112 :=
sorry

end largest_n_exists_unique_k_l82_82772


namespace non_congruent_rectangles_l82_82355

theorem non_congruent_rectangles (h w : ℕ) (hp : 2 * (h + w) = 80) :
  ∃ n, n = 20 := by
  sorry

end non_congruent_rectangles_l82_82355


namespace a_plus_b_eq_2_l82_82916

theorem a_plus_b_eq_2 (a b : ℝ) 
  (h₁ : 2 = a + b) 
  (h₂ : 4 = a + b / 4) : a + b = 2 :=
by
  sorry

end a_plus_b_eq_2_l82_82916


namespace range_equality_of_f_and_f_f_l82_82660

noncomputable def f (x a : ℝ) := x * Real.log x - x + 2 * a

theorem range_equality_of_f_and_f_f (a : ℝ) :
  (∀ x : ℝ, 0 < x → 1 < f x a) ∧ (∀ x : ℝ, 0 < x → f x a ≤ 1) →
  (∃ I : Set ℝ, (Set.range (λ x => f x a) = I) ∧ (Set.range (λ x => f (f x a) a) = I)) → 
  (1/2 < a ∧ a ≤ 1) :=
by 
  sorry

end range_equality_of_f_and_f_f_l82_82660


namespace stratified_sampling_first_grade_selection_l82_82516

theorem stratified_sampling_first_grade_selection
  (total_students : ℕ)
  (students_grade1 : ℕ)
  (sample_size : ℕ)
  (h_total : total_students = 2000)
  (h_grade1 : students_grade1 = 400)
  (h_sample : sample_size = 200) :
  sample_size * students_grade1 / total_students = 40 := by
  sorry

end stratified_sampling_first_grade_selection_l82_82516


namespace eighth_day_of_april_2000_is_saturday_l82_82566

noncomputable def april_2000_eight_day_is_saturday : Prop :=
  (∃ n : ℕ, (1 ≤ n ∧ n ≤ 7) ∧
            ((n + 0 * 7) = 2 ∨ (n + 1 * 7) = 2 ∨ (n + 2 * 7) = 2 ∨
             (n + 3 * 7) = 2 ∨ (n + 4 * 7) = 2) ∧
            ((n + 0 * 7) % 2 = 0 ∨ (n + 1 * 7) % 2 = 0 ∨
             (n + 2 * 7) % 2 = 0 ∨ (n + 3 * 7) % 2 = 0 ∨
             (n + 4 * 7) % 2 = 0) ∧
            (∃ k : ℕ, k ≤ 4 ∧ (n + k * 7 = 8))) ∧
            (8 % 7) = 1 ∧ (1 ≠ 0)

theorem eighth_day_of_april_2000_is_saturday :
  april_2000_eight_day_is_saturday := 
sorry

end eighth_day_of_april_2000_is_saturday_l82_82566


namespace div_fraction_eq_l82_82195

theorem div_fraction_eq :
  (5 / 3) / (1 / 4) = 20 / 3 := 
by
  sorry

end div_fraction_eq_l82_82195


namespace scientific_notation_of_190_million_l82_82591

theorem scientific_notation_of_190_million : (190000000 : ℝ) = 1.9 * 10^8 :=
sorry

end scientific_notation_of_190_million_l82_82591


namespace sector_area_l82_82356

theorem sector_area (α : ℝ) (l : ℝ) (r : ℝ) (S : ℝ) : 
  α = 1 ∧ l = 6 ∧ l = α * r → S = (1/2) * α * r ^ 2 → S = 18 :=
by
  intros h h' 
  sorry

end sector_area_l82_82356


namespace Fred_hourly_rate_l82_82561

-- Define the conditions
def hours_worked : ℝ := 8
def total_earned : ℝ := 100

-- Assert the proof goal
theorem Fred_hourly_rate : total_earned / hours_worked = 12.5 :=
by
  sorry

end Fred_hourly_rate_l82_82561


namespace find_numbers_l82_82705

theorem find_numbers (x y : ℝ) (r : ℝ) (d : ℝ) 
  (h_geom_x : x = 5 * r) 
  (h_geom_y : y = 5 * r^2)
  (h_arith_1 : y = x + d) 
  (h_arith_2 : 15 = y + d) : 
  x + y = 10 :=
by
  sorry

end find_numbers_l82_82705


namespace school_stats_l82_82427

-- Defining the conditions
def girls_grade6 := 315
def boys_grade6 := 309
def girls_grade7 := 375
def boys_grade7 := 341
def drama_club_members := 80
def drama_club_boys_percent := 30 / 100

-- Calculate the derived numbers
def students_grade6 := girls_grade6 + boys_grade6
def students_grade7 := girls_grade7 + boys_grade7
def total_students := students_grade6 + students_grade7
def drama_club_boys := drama_club_boys_percent * drama_club_members
def drama_club_girls := drama_club_members - drama_club_boys

-- Theorem
theorem school_stats :
  total_students = 1340 ∧
  drama_club_girls = 56 ∧
  boys_grade6 = 309 ∧
  boys_grade7 = 341 :=
by
  -- We provide the proof steps inline with sorry placeholders.
  -- In practice, these would be filled with appropriate proofs.
  sorry

end school_stats_l82_82427


namespace janice_weekly_earnings_l82_82226

-- define the conditions
def regular_days_per_week : Nat := 5
def regular_earnings_per_day : Nat := 30
def overtime_earnings_per_shift : Nat := 15
def overtime_shifts_per_week : Nat := 3

-- define the total earnings calculation
def total_earnings (regular_days : Nat) (regular_rate : Nat) (overtime_shifts : Nat) (overtime_rate : Nat) : Nat :=
  (regular_days * regular_rate) + (overtime_shifts * overtime_rate)

-- state the problem to be proved
theorem janice_weekly_earnings : total_earnings regular_days_per_week regular_earnings_per_day overtime_shifts_per_week overtime_earnings_per_shift = 195 :=
by
  sorry

end janice_weekly_earnings_l82_82226


namespace total_sales_correct_l82_82297

def maries_newspapers : ℝ := 275.0
def maries_magazines : ℝ := 150.0
def total_sales := maries_newspapers + maries_magazines

theorem total_sales_correct :
  total_sales = 425.0 :=
by
  -- Proof omitted
  sorry

end total_sales_correct_l82_82297


namespace triangle_angle_not_less_than_60_l82_82392

theorem triangle_angle_not_less_than_60 
  (a b c : ℝ) 
  (h1 : a + b + c = 180) 
  (h2 : a < 60) 
  (h3 : b < 60) 
  (h4 : c < 60) : 
  false := 
by
  sorry

end triangle_angle_not_less_than_60_l82_82392


namespace number_of_correct_statements_l82_82381

def input_statement (s : String) : Prop :=
  s = "INPUT a; b; c"

def output_statement (s : String) : Prop :=
  s = "A=4"

def assignment_statement1 (s : String) : Prop :=
  s = "3=B"

def assignment_statement2 (s : String) : Prop :=
  s = "A=B=-2"

theorem number_of_correct_statements :
    input_statement "INPUT a; b; c" = false ∧
    output_statement "A=4" = false ∧
    assignment_statement1 "3=B" = false ∧
    assignment_statement2 "A=B=-2" = false :=
sorry

end number_of_correct_statements_l82_82381


namespace problem_statement_l82_82710

noncomputable def seq_sub_triples: ℚ :=
  let a := (5 / 6 : ℚ)
  let b := (1 / 6 : ℚ)
  let c := (1 / 4 : ℚ)
  a - b - c

theorem problem_statement : seq_sub_triples = 5 / 12 := by
  sorry

end problem_statement_l82_82710


namespace almond_butter_servings_l82_82183

noncomputable def servings_in_container (total_tbsps : ℚ) (serving_size : ℚ) : ℚ :=
  total_tbsps / serving_size

theorem almond_butter_servings :
  servings_in_container (34 + 3/5) (5 + 1/2) = 6 + 21/55 :=
by
  sorry

end almond_butter_servings_l82_82183


namespace volleyball_match_prob_A_win_l82_82696

-- Definitions of given probabilities and conditions
def rally_scoring_system := true
def first_to_25_wins := true
def tie_at_24_24_continues_until_lead_by_2 := true
def prob_team_A_serves_win : ℚ := 2/3
def prob_team_B_serves_win : ℚ := 2/5
def outcomes_independent := true
def score_22_22_team_A_serves := true

-- The problem to prove
theorem volleyball_match_prob_A_win :
  rally_scoring_system ∧
  first_to_25_wins ∧
  tie_at_24_24_continues_until_lead_by_2 ∧
  prob_team_A_serves_win = 2/3 ∧
  prob_team_B_serves_win = 2/5 ∧
  outcomes_independent ∧
  score_22_22_team_A_serves →
  (prob_team_A_serves_win ^ 3 + (1 - prob_team_A_serves_win) * prob_team_B_serves_win * prob_team_A_serves_win ^ 2 + prob_team_A_serves_win * (1 - prob_team_A_serves_win) * prob_team_B_serves_win * prob_team_A_serves_win + prob_team_A_serves_win ^ 2 * (1 - prob_team_A_serves_win) * prob_team_B_serves_win) = 64/135 :=
by
  sorry

end volleyball_match_prob_A_win_l82_82696


namespace compute_quotient_of_q_and_r_l82_82881

theorem compute_quotient_of_q_and_r (p q r s t : ℤ) (h_eq_4 : 256 * p + 64 * q + 16 * r + 4 * s + t = 0)
                                     (h_eq_neg3 : -27 * p + 9 * q - 3 * r + s + t = 0)
                                     (h_eq_0 : t = 0)
                                     (h_p_nonzero : p ≠ 0) :
                                     (q + r) / p = -13 :=
by
  have eq1 := h_eq_4
  have eq2 := h_eq_neg3
  rw [h_eq_0] at eq1 eq2
  sorry

end compute_quotient_of_q_and_r_l82_82881


namespace determine_m_l82_82822

theorem determine_m (m : ℝ) : (∀ x : ℝ, (m * x = 1 → x = 1 ∨ x = -1)) ↔ (m = 0 ∨ m = 1 ∨ m = -1) :=
by sorry

end determine_m_l82_82822


namespace black_squares_in_35th_row_l82_82926

-- Define the condition for the starting color based on the row
def starts_with_black (n : ℕ) : Prop := n % 2 = 1
def ends_with_white (n : ℕ) : Prop := true  -- This is trivially true by the problem condition
def total_squares (n : ℕ) : ℕ := 2 * n 
-- Black squares are half of the total squares for rows starting with a black square
def black_squares (n : ℕ) : ℕ := total_squares n / 2

theorem black_squares_in_35th_row : black_squares 35 = 35 :=
sorry

end black_squares_in_35th_row_l82_82926


namespace sum_over_term_is_two_l82_82774

-- Definitions of conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n / 2 * (2 * a 0 + (n - 1) * (a 1 - a 0))

def seq_sn_over_an_arithmetic (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∃ dS : ℝ, ∀ n : ℕ, (S (n + 1)) / (a (n + 1)) = (S n) / (a n) + dS

-- The theorem to prove
theorem sum_over_term_is_two (a S : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_first_n_terms a S)
  (h3 : seq_sn_over_an_arithmetic S a) :
  S 3 / a 3 = 2 :=
sorry

end sum_over_term_is_two_l82_82774


namespace necessary_condition_transitivity_l82_82168

theorem necessary_condition_transitivity (A B C : Prop) 
  (hAB : A → B) (hBC : B → C) : A → C := 
by
  intro ha
  apply hBC
  apply hAB
  exact ha

-- sorry


end necessary_condition_transitivity_l82_82168


namespace a_eq_1_sufficient_not_necessary_l82_82641

theorem a_eq_1_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → |x - 1| ≤ |x - a|) ∧ ¬(∀ x : ℝ, x ≤ 1 → |x - 1| = |x - a|) :=
by
  sorry

end a_eq_1_sufficient_not_necessary_l82_82641


namespace find_x_intercept_l82_82646

-- Define the equation of the line
def line_eq (x y : ℝ) : Prop := 4 * x + 7 * y = 28

-- Define the x-intercept point when y = 0
def x_intercept (x : ℝ) : Prop := line_eq x 0

-- Prove that for the x-intercept, when y = 0, x = 7
theorem find_x_intercept : x_intercept 7 :=
by
  -- proof would go here
  sorry

end find_x_intercept_l82_82646


namespace find_b_of_roots_condition_l82_82618

theorem find_b_of_roots_condition
  (α β : ℝ)
  (h1 : α * β = -1)
  (h2 : α + β = -b)
  (h3 : α * β - 2 * α - 2 * β = -11) :
  b = -5 := 
  sorry

end find_b_of_roots_condition_l82_82618


namespace trip_time_40mph_l82_82333

noncomputable def trip_time_80mph : ℝ := 6.75
noncomputable def speed_80mph : ℝ := 80
noncomputable def speed_40mph : ℝ := 40

noncomputable def distance : ℝ := speed_80mph * trip_time_80mph

theorem trip_time_40mph : distance / speed_40mph = 13.50 :=
by
  sorry

end trip_time_40mph_l82_82333


namespace cost_of_paving_is_correct_l82_82043

def length_of_room : ℝ := 5.5
def width_of_room : ℝ := 4
def rate_per_square_meter : ℝ := 950
def area_of_room : ℝ := length_of_room * width_of_room
def cost_of_paving : ℝ := area_of_room * rate_per_square_meter

theorem cost_of_paving_is_correct : cost_of_paving = 20900 := 
by
  sorry

end cost_of_paving_is_correct_l82_82043


namespace students_like_basketball_or_cricket_or_both_l82_82361

theorem students_like_basketball_or_cricket_or_both {A B C : ℕ} (hA : A = 12) (hB : B = 8) (hC : C = 3) :
    A + B - C = 17 :=
by
  sorry

end students_like_basketball_or_cricket_or_both_l82_82361


namespace jungkook_age_l82_82146

theorem jungkook_age
    (J U : ℕ)
    (h1 : J = U - 12)
    (h2 : (J + 3) + (U + 3) = 38) :
    J = 10 := 
sorry

end jungkook_age_l82_82146


namespace range_of_f_l82_82121

def f (x : ℤ) : ℤ := x ^ 2 - 2 * x
def domain : Set ℤ := {0, 1, 2, 3}
def expectedRange : Set ℤ := {-1, 0, 3}

theorem range_of_f : (Set.image f domain) = expectedRange :=
  sorry

end range_of_f_l82_82121


namespace captain_and_vicecaptain_pair_boys_and_girls_l82_82831

-- Problem A
theorem captain_and_vicecaptain (n : ℕ) (h : n = 11) : ∃ ways : ℕ, ways = 110 :=
by
  sorry

-- Problem B
theorem pair_boys_and_girls (N : ℕ) : ∃ ways : ℕ, ways = Nat.factorial N :=
by
  sorry

end captain_and_vicecaptain_pair_boys_and_girls_l82_82831


namespace cubical_cake_l82_82484

noncomputable def cubical_cake_properties : Prop :=
  let a : ℝ := 3
  let top_area := (1 / 2) * 3 * 1.5
  let height := 3
  let volume := top_area * height
  let vertical_triangles_area := 2 * ((1 / 2) * 1.5 * 3)
  let vertical_rectangular_area := 3 * 3
  let iced_area := top_area + vertical_triangles_area + vertical_rectangular_area
  volume + iced_area = 22.5

theorem cubical_cake : cubical_cake_properties := sorry

end cubical_cake_l82_82484


namespace actor_A_constraints_l82_82622

-- Definitions corresponding to the conditions.
def numberOfActors : Nat := 6
def positionConstraints : Nat := 4
def permutations (n : Nat) : Nat := Nat.factorial n

-- Lean statement for the proof problem.
theorem actor_A_constraints : 
  (positionConstraints * permutations (numberOfActors - 1)) = 480 := by
sorry

end actor_A_constraints_l82_82622


namespace find_m_l82_82860

noncomputable def m_value (a b c d : Int) (Y : Int) : Int :=
  let l1_1 := a + b
  let l1_2 := b + c
  let l1_3 := c + d
  let l2_1 := l1_1 + l1_2
  let l2_2 := l1_2 + l1_3
  let l3 := l2_1 + l2_2
  if l3 = Y then a else 0

theorem find_m : m_value m 6 (-3) 4 20 = 7 := sorry

end find_m_l82_82860


namespace units_digit_of_five_consecutive_product_is_zero_l82_82645

theorem units_digit_of_five_consecutive_product_is_zero (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10 = 0 :=
by
  sorry

end units_digit_of_five_consecutive_product_is_zero_l82_82645


namespace length_of_bridge_l82_82765

theorem length_of_bridge (length_train : ℕ) (speed_train_kmh : ℕ) (crossing_time_sec : ℕ)
    (h_length_train : length_train = 125)
    (h_speed_train_kmh : speed_train_kmh = 45)
    (h_crossing_time_sec : crossing_time_sec = 30) : 
    ∃ (length_bridge : ℕ), length_bridge = 250 := by
  sorry

end length_of_bridge_l82_82765


namespace history_only_students_l82_82386

theorem history_only_students 
  (total_students : ℕ)
  (history_students stats_students physics_students chem_students : ℕ) 
  (hist_stats hist_phys hist_chem stats_phys stats_chem phys_chem all_four : ℕ) 
  (h1 : total_students = 500)
  (h2 : history_students = 150)
  (h3 : stats_students = 130)
  (h4 : physics_students = 120)
  (h5 : chem_students = 100)
  (h6 : hist_stats = 60)
  (h7 : hist_phys = 50)
  (h8 : hist_chem = 40)
  (h9 : stats_phys = 35)
  (h10 : stats_chem = 30)
  (h11 : phys_chem = 25)
  (h12 : all_four = 20) : 
  (history_students - hist_stats - hist_phys - hist_chem + all_four) = 20 := 
by 
  sorry

end history_only_students_l82_82386


namespace Diego_more_than_half_Martha_l82_82298

theorem Diego_more_than_half_Martha (M D : ℕ) (H1 : M = 90)
  (H2 : D > M / 2)
  (H3 : M + D = 145):
  D - M / 2 = 10 :=
by
  sorry

end Diego_more_than_half_Martha_l82_82298


namespace coin_value_difference_l82_82875

theorem coin_value_difference (p n d : ℕ) (h : p + n + d = 3000) (hp : p ≥ 1) (hn : n ≥ 1) (hd : d ≥ 1) : 
  (p + 5 * n + 10 * d).max - (p + 5 * n + 10 * d).min = 26973 := 
sorry

end coin_value_difference_l82_82875


namespace total_presents_l82_82921

-- Definitions based on the problem conditions
def numChristmasPresents : ℕ := 60
def numBirthdayPresents : ℕ := numChristmasPresents / 2

-- Theorem statement
theorem total_presents : numChristmasPresents + numBirthdayPresents = 90 :=
by
  -- Proof is omitted
  sorry

end total_presents_l82_82921


namespace tetrahedron_edge_length_l82_82101

-- Define the problem specifications
def mutuallyTangent (r : ℝ) (a b c d : ℝ → ℝ → ℝ → Prop) :=
  a = b ∧ a = c ∧ a = d ∧ b = c ∧ b = d ∧ c = d

noncomputable def tetrahedronEdgeLength (r : ℝ) : ℝ :=
  2 + 2 * Real.sqrt 6

-- Proof goal: edge length of tetrahedron containing four mutually tangent balls each of radius 1
theorem tetrahedron_edge_length (r : ℝ) (a b c d : ℝ → ℝ → ℝ → Prop)
  (h1 : r = 1)
  (h2 : mutuallyTangent r a b c d)
  : tetrahedronEdgeLength r = 2 + 2 * Real.sqrt 6 :=
sorry

end tetrahedron_edge_length_l82_82101


namespace even_sum_probability_l82_82411

-- Conditions
def prob_even_first_wheel : ℚ := 1 / 4
def prob_odd_first_wheel : ℚ := 3 / 4
def prob_even_second_wheel : ℚ := 2 / 3
def prob_odd_second_wheel : ℚ := 1 / 3

-- Statement: Theorem that the probability of the sum being even is 5/12
theorem even_sum_probability : 
  (prob_even_first_wheel * prob_even_second_wheel) + 
  (prob_odd_first_wheel * prob_odd_second_wheel) = 5 / 12 :=
by
  -- Proof steps would go here
  sorry

end even_sum_probability_l82_82411


namespace tom_sleep_increase_l82_82527

theorem tom_sleep_increase :
  ∀ (initial_sleep : ℕ) (increase_by : ℚ), 
  initial_sleep = 6 → 
  increase_by = 1/3 → 
  initial_sleep + increase_by * initial_sleep = 8 :=
by 
  intro initial_sleep increase_by h1 h2
  simp [*, add_mul, mul_comm]
  sorry

end tom_sleep_increase_l82_82527


namespace circle_through_A_B_C_l82_82718

-- Definitions of points A, B, and C
def A : ℝ × ℝ := (1, 12)
def B : ℝ × ℝ := (7, 10)
def C : ℝ × ℝ := (-9, 2)

-- Definition of the expected standard equation of the circle
def circle_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = 100

-- Theorem stating that the expected equation is the equation of the circle through points A, B, and C
theorem circle_through_A_B_C : 
  ∀ (x y : ℝ),
  (x, y) = A ∨ (x, y) = B ∨ (x, y) = C → 
  circle_eq x y := sorry

end circle_through_A_B_C_l82_82718


namespace cauchy_schwarz_inequality_l82_82206

theorem cauchy_schwarz_inequality (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by
  sorry

end cauchy_schwarz_inequality_l82_82206


namespace p_add_inv_p_gt_two_l82_82643

theorem p_add_inv_p_gt_two {p : ℝ} (hp_pos : p > 0) (hp_neq_one : p ≠ 1) : p + 1 / p > 2 :=
by
  sorry

end p_add_inv_p_gt_two_l82_82643


namespace solve_sum_of_digits_eq_2018_l82_82976

def s (n : ℕ) : ℕ := (Nat.digits 10 n).sum

theorem solve_sum_of_digits_eq_2018 : ∃ n : ℕ, n + s n = 2018 := by
  sorry

end solve_sum_of_digits_eq_2018_l82_82976


namespace cryptarithm_C_value_l82_82495

/--
Given digits A, B, and C where A, B, and C are distinct and non-repeating,
and the following conditions hold:
1. ABC - BC = A0A
Prove that C = 9.
-/
theorem cryptarithm_C_value (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_non_repeating: (0 <= A ∧ A <= 9) ∧ (0 <= B ∧ B <= 9) ∧ (0 <= C ∧ C <= 9))
  (h_subtraction : 100 * A + 10 * B + C - (10 * B + C) = 100 * A + 0 + A) :
  C = 9 := sorry

end cryptarithm_C_value_l82_82495


namespace multiply_powers_l82_82368

theorem multiply_powers (a : ℝ) : (a^3) * (a^3) = a^6 := by
  sorry

end multiply_powers_l82_82368


namespace simplify_polynomial_l82_82615

theorem simplify_polynomial :
  (3 * x ^ 5 - 2 * x ^ 3 + 5 * x ^ 2 - 8 * x + 6) + (7 * x ^ 4 + x ^ 3 - 3 * x ^ 2 + x - 9) =
  3 * x ^ 5 + 7 * x ^ 4 - x ^ 3 + 2 * x ^ 2 - 7 * x - 3 :=
by
  sorry

end simplify_polynomial_l82_82615


namespace zero_in_interval_l82_82202

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 3

theorem zero_in_interval : 
    (∀ x y : ℝ, 0 < x → x < y → f x < f y) → 
    (f 1 = -2) →
    (f 2 = Real.log 2 + 5) →
    (∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0) :=
by 
    sorry

end zero_in_interval_l82_82202


namespace sum_infinite_series_eq_l82_82187

theorem sum_infinite_series_eq {x : ℝ} (hx : |x| < 1) :
  (∑' n : ℕ, (n + 1) * x^n) = 1 / (1 - x)^2 :=
by
  sorry

end sum_infinite_series_eq_l82_82187


namespace socks_selection_l82_82460

theorem socks_selection :
  let red_socks := 120
  let green_socks := 90
  let blue_socks := 70
  let black_socks := 50
  let yellow_socks := 30
  let total_socks :=  red_socks + green_socks + blue_socks + black_socks + yellow_socks 
  (∀ k : ℕ, k ≥ 1 → k ≤ total_socks → (∃ p : ℕ, p = 12 → (p ≥ k / 2)) → k = 28) :=
by
  sorry

end socks_selection_l82_82460


namespace triangle_area_is_3_max_f_l82_82700

noncomputable def triangle_area :=
  let a : ℝ := 2
  let b : ℝ := 2 * Real.sqrt 3
  let c : ℝ := 2
  let A : ℝ := Real.pi / 3
  (1 / 2) * b * c * Real.sin A

theorem triangle_area_is_3 :
  triangle_area = 3 := by
  sorry

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.cos x * (Real.sin x * Real.cos (Real.pi / 3) + Real.cos x * Real.sin (Real.pi / 3))

theorem max_f :
  ∃ x ∈ Set.Icc 0 (Real.pi / 3), f x = 2 + Real.sqrt 3 ∧ x = Real.pi / 12 := by
  sorry

end triangle_area_is_3_max_f_l82_82700


namespace bears_in_shipment_l82_82064

theorem bears_in_shipment
  (initial_bears : ℕ) (shelves : ℕ) (bears_per_shelf : ℕ)
  (total_bears_after_shipment : ℕ) 
  (initial_bears_eq : initial_bears = 5)
  (shelves_eq : shelves = 2)
  (bears_per_shelf_eq : bears_per_shelf = 6)
  (total_bears_calculation : total_bears_after_shipment = shelves * bears_per_shelf)
  : total_bears_after_shipment - initial_bears = 7 :=
by
  sorry

end bears_in_shipment_l82_82064


namespace BKING_2023_reappears_at_20_l82_82099

-- Defining the basic conditions of the problem
def cycle_length_BKING : ℕ := 5
def cycle_length_2023 : ℕ := 4

-- Formulating the proof problem statement
theorem BKING_2023_reappears_at_20 :
  Nat.lcm cycle_length_BKING cycle_length_2023 = 20 :=
by
  sorry

end BKING_2023_reappears_at_20_l82_82099


namespace prop1_prop2_l82_82189

-- Proposition 1: Prove the contrapositive
theorem prop1 (q : ℝ) (h : ¬(∃ x : ℝ, x^2 + 2 * x + q = 0)) : q ≥ 1 :=
sorry

-- Proposition 2: Prove the contrapositive
theorem prop2 (x y : ℝ) (h : ¬(x = 0 ∧ y = 0)) : x^2 + y^2 ≠ 0 :=
sorry

end prop1_prop2_l82_82189


namespace quadratic_second_root_l82_82999

noncomputable def second_root (p q : ℝ) : ℝ :=
  -2 * p / (p - 2)

theorem quadratic_second_root (p q : ℝ) (h1 : (p + q) * 1^2 + (p - q) * 1 + p * q = 0) :
  ∃ r : ℝ, r = second_root p q :=
by 
  sorry

end quadratic_second_root_l82_82999


namespace initial_money_l82_82207

theorem initial_money (spent allowance total initial : ℕ) 
  (h1 : spent = 2) 
  (h2 : allowance = 26) 
  (h3 : total = 29) 
  (h4 : initial - spent + allowance = total) : 
  initial = 5 := 
by 
  sorry

end initial_money_l82_82207


namespace tony_will_have_4_dollars_in_change_l82_82634

def tony_change : ℕ :=
  let bucket_capacity : ℕ := 2
  let sandbox_depth : ℕ := 2
  let sandbox_width : ℕ := 4
  let sandbox_length : ℕ := 5
  let sand_weight_per_cubic_foot : ℕ := 3
  let water_consumed_per_drink : ℕ := 3
  let trips_per_drink : ℕ := 4
  let bottle_capacity : ℕ := 15
  let bottle_cost : ℕ := 2
  let initial_money : ℕ := 10

  let sandbox_volume := sandbox_depth * sandbox_width * sandbox_length
  let total_sand_weight := sandbox_volume * sand_weight_per_cubic_foot
  let number_of_trips := total_sand_weight / bucket_capacity
  let number_of_drinks := number_of_trips / trips_per_drink
  let total_water_consumed := number_of_drinks * water_consumed_per_drink
  let number_of_bottles := total_water_consumed / bottle_capacity
  let total_water_cost := number_of_bottles * bottle_cost
  let change := initial_money - total_water_cost

  change

theorem tony_will_have_4_dollars_in_change : tony_change = 4 := by
  sorry

end tony_will_have_4_dollars_in_change_l82_82634


namespace contrapositive_proposition_l82_82782

theorem contrapositive_proposition (x : ℝ) : (x > 10 → x > 1) ↔ (x ≤ 1 → x ≤ 10) :=
by
  sorry

end contrapositive_proposition_l82_82782


namespace marble_weight_l82_82246

theorem marble_weight (W : ℝ) (h : 2 * W + 0.08333333333333333 = 0.75) : 
  W = 0.33333333333333335 := 
by 
  -- Skipping the proof as specified
  sorry

end marble_weight_l82_82246


namespace ratio_of_weights_l82_82757

variable (x y : ℝ)

theorem ratio_of_weights (h : x + y = 7 * (x - y)) (h1 : x > y) : x / y = 4 / 3 :=
sorry

end ratio_of_weights_l82_82757


namespace total_amount_Rs20_l82_82890

theorem total_amount_Rs20 (x y z : ℕ) 
(h1 : x + y + z = 130) 
(h2 : 95 * x + 45 * y + 20 * z = 7000) : 
∃ z : ℕ, (20 * z) = (7000 - 95 * x - 45 * y) / 20 := sorry

end total_amount_Rs20_l82_82890


namespace carol_packs_l82_82112

theorem carol_packs (invitations_per_pack total_invitations packs_bought : ℕ) 
  (h1 : invitations_per_pack = 9)
  (h2 : total_invitations = 45) 
  (h3 : packs_bought = total_invitations / invitations_per_pack) : 
  packs_bought = 5 :=
by 
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end carol_packs_l82_82112


namespace sequence_geometric_sequence_general_term_l82_82564

theorem sequence_geometric (a : ℕ → ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∃ r : ℕ, (a 1 + 1) = 3 ∧ (∀ n, (a (n + 1) + 1) = r * (a n + 1)) := by
  sorry

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = 3 * 2^(n-1) - 1 := by
  sorry

end sequence_geometric_sequence_general_term_l82_82564


namespace average_age_increase_l82_82915

variable (A B C : ℕ)

theorem average_age_increase (A : ℕ) (B : ℕ) (C : ℕ) (h1 : 21 < B) (h2 : 23 < C) (h3 : A + B + C > A + 21 + 23) :
  (B + C) / 2 > 22 := by
  sorry

end average_age_increase_l82_82915


namespace midpoint_trajectory_l82_82844

-- Define the parabola and line intersection conditions
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

def line_through_focus (A B : ℝ × ℝ) (focus : ℝ × ℝ) : Prop :=
  ∃ m b : ℝ, (∀ P ∈ [A, B, focus], P.2 = m * P.1 + b)

def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem midpoint_trajectory (A B M : ℝ × ℝ) (focus : ℝ × ℝ):
  (parabola A.1 A.2) ∧ (parabola B.1 B.2) ∧ (line_through_focus A B focus) ∧ (midpoint A B M)
  → (M.1 ^ 2 = 2 * M.2 - 2) :=
by
  sorry

end midpoint_trajectory_l82_82844


namespace min_value_of_sum_of_squares_l82_82155

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x * y + y * z + x * z = 4) :
  x^2 + y^2 + z^2 ≥ 4 :=
sorry

end min_value_of_sum_of_squares_l82_82155


namespace problem_solution_l82_82807

variable (α : ℝ)

/-- If $\sin\alpha = 2\cos\alpha$, then the function $f(x) = 2^x - \tan\alpha$ satisfies $f(0) = -1$. -/
theorem problem_solution (h : Real.sin α = 2 * Real.cos α) : (2^0 - Real.tan α) = -1 := by
  sorry

end problem_solution_l82_82807


namespace distinct_real_numbers_sum_l82_82336

theorem distinct_real_numbers_sum:
  ∀ (p q r s : ℝ),
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
    (r + s = 12 * p) →
    (r * s = -13 * q) →
    (p + q = 12 * r) →
    (p * q = -13 * s) →
    p + q + r + s = 2028 :=
by
  intros p q r s h_distinct h1 h2 h3 h4
  sorry

end distinct_real_numbers_sum_l82_82336


namespace probability_p_s_multiple_of_7_l82_82838

section
variables (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 60) (h2 : 1 ≤ b ∧ b ≤ 60) (h3 : a ≠ b)

theorem probability_p_s_multiple_of_7 :
  (∃ k : ℕ, a * b + a + b = 7 * k) → (64 / 1770 : ℚ) = 32 / 885 :=
sorry
end

end probability_p_s_multiple_of_7_l82_82838


namespace sqrt_88200_simplified_l82_82249

theorem sqrt_88200_simplified : Real.sqrt 88200 = 210 * Real.sqrt 6 :=
by sorry

end sqrt_88200_simplified_l82_82249


namespace system_solution_unique_l82_82086

theorem system_solution_unique : 
  ∀ (x y z : ℝ),
  (4 * x^2) / (1 + 4 * x^2) = y ∧
  (4 * y^2) / (1 + 4 * y^2) = z ∧
  (4 * z^2) / (1 + 4 * z^2) = x 
  → (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end system_solution_unique_l82_82086


namespace gcd_exponentiation_gcd_fermat_numbers_l82_82727

-- Part (a)
theorem gcd_exponentiation (m n : ℕ) (a : ℕ) (h1 : m ≠ n) (h2 : a > 1) : 
  Nat.gcd (a^m - 1) (a^n - 1) = a^(Nat.gcd m n) - 1 :=
by
sorry

-- Part (b)
def fermat_number (k : ℕ) : ℕ := 2^(2^k) + 1

theorem gcd_fermat_numbers (m n : ℕ) (h1 : m ≠ n) : 
  Nat.gcd (fermat_number m) (fermat_number n) = 1 :=
by
sorry

end gcd_exponentiation_gcd_fermat_numbers_l82_82727


namespace cartons_in_a_case_l82_82494

-- Definitions based on problem conditions
def numberOfBoxesInCarton (c : ℕ) (b : ℕ) : ℕ := c * b * 300
def paperClipsInTwoCases (c : ℕ) (b : ℕ) : ℕ := 2 * numberOfBoxesInCarton c b

-- Condition from problem statement: paperClipsInTwoCases c b = 600
theorem cartons_in_a_case 
  (c b : ℕ) 
  (h1 : paperClipsInTwoCases c b = 600) 
  (h2 : b ≥ 1) : 
  c = 1 := 
by
  -- Proof will be provided here
  sorry

end cartons_in_a_case_l82_82494


namespace chess_club_officers_l82_82180

/-- The Chess Club with 24 members needs to choose 3 officers: president,
    secretary, and treasurer. Each person can hold at most one office. 
    Alice and Bob will only serve together as officers. Prove that 
    the number of ways to choose the officers is 9372. -/
theorem chess_club_officers : 
  let members := 24
  let num_officers := 3
  let alice_and_bob_together := true
  ∃ n : ℕ, n = 9372 := sorry

end chess_club_officers_l82_82180


namespace problem_statement_l82_82311

noncomputable def f (x k : ℝ) : ℝ :=
  (1/5) * (x - k + 4500 / x)

noncomputable def fuel_consumption_100km (x k : ℝ) : ℝ :=
  100 / x * f x k

theorem problem_statement (x k : ℝ)
  (hx1 : 60 ≤ x) (hx2 : x ≤ 120)
  (hk1 : 60 ≤ k) (hk2 : k ≤ 100)
  (H : f 120 k = 11.5) :

  (∀ x, 60 ≤ x ∧ x ≤ 100 → f x k ≤ 9 ∧ 
  (if 75 ≤ k ∧ k ≤ 100 then fuel_consumption_100km (9000 / k) k = 20 - k^2 / 900
   else fuel_consumption_100km 120 k = 105 / 4 - k / 6)) :=
  sorry

end problem_statement_l82_82311


namespace Ki_tae_pencils_l82_82632

theorem Ki_tae_pencils (P B : ℤ) (h1 : P + B = 12) (h2 : 1000 * P + 1300 * B = 15000) : P = 2 :=
sorry

end Ki_tae_pencils_l82_82632


namespace no_such_integers_l82_82662

theorem no_such_integers (a b : ℤ) : 
  ¬ (∃ a b : ℤ, ∃ k₁ k₂ : ℤ, a^5 * b + 3 = k₁^3 ∧ a * b^5 + 3 = k₂^3) :=
by 
  sorry

end no_such_integers_l82_82662


namespace solution_is_correct_l82_82396

-- Define the conditions of the problem.
variable (x y z : ℝ)

-- The system of equations given in the problem
def system_of_equations (x y z : ℝ) :=
  (1/x + 1/(y+z) = 6/5) ∧
  (1/y + 1/(x+z) = 3/4) ∧
  (1/z + 1/(x+y) = 2/3)

-- The desired solution
def solution (x y z : ℝ) := x = 2 ∧ y = 3 ∧ z = 1

-- The theorem to prove
theorem solution_is_correct (h : system_of_equations x y z) : solution x y z :=
sorry

end solution_is_correct_l82_82396


namespace jars_proof_l82_82027

def total_plums : ℕ := 240
def exchange_ratio : ℕ := 7
def mangoes_per_jar : ℕ := 5

def ripe_plums (total_plums : ℕ) := total_plums / 4
def unripe_plums (total_plums : ℕ) := 3 * total_plums / 4
def unripe_plums_kept : ℕ := 46

def plums_for_trade (total_plums unripe_plums_kept : ℕ) : ℕ :=
  ripe_plums total_plums + (unripe_plums total_plums - unripe_plums_kept)

def mangoes_received (plums_for_trade exchange_ratio : ℕ) : ℕ :=
  plums_for_trade / exchange_ratio

def jars_of_mangoes (mangoes_received mangoes_per_jar : ℕ) : ℕ :=
  mangoes_received / mangoes_per_jar

theorem jars_proof : jars_of_mangoes (mangoes_received (plums_for_trade total_plums unripe_plums_kept) exchange_ratio) mangoes_per_jar = 5 :=
by
  sorry

end jars_proof_l82_82027


namespace ratio_of_segments_l82_82814

-- Definitions and conditions as per part (a)
variables (a b c r s : ℝ)
variable (h₁ : a / b = 1 / 3)
variable (h₂ : a^2 = r * c)
variable (h₃ : b^2 = s * c)

-- The statement of the theorem directly addressing part (c)
theorem ratio_of_segments (a b c r s : ℝ) 
  (h₁ : a / b = 1 / 3)
  (h₂ : a^2 = r * c)
  (h₃ : b^2 = s * c) :
  r / s = 1 / 9 :=
  sorry

end ratio_of_segments_l82_82814


namespace number_of_days_l82_82639

noncomputable def days_to_lay_bricks (b c f : ℕ) : ℕ :=
(b * b) / f

theorem number_of_days (b c f : ℕ) (h_nonzero_f : f ≠ 0) (h_bc_pos : b > 0 ∧ c > 0) :
  days_to_lay_bricks b c f = (b * b) / f :=
by 
  sorry

end number_of_days_l82_82639


namespace worth_of_each_gold_bar_l82_82149

theorem worth_of_each_gold_bar
  (rows : ℕ) (gold_bars_per_row : ℕ) (total_worth : ℕ)
  (h1 : rows = 4) (h2 : gold_bars_per_row = 20) (h3 : total_worth = 1600000)
  (total_gold_bars : ℕ) (h4 : total_gold_bars = rows * gold_bars_per_row) :
  total_worth / total_gold_bars = 20000 :=
by sorry

end worth_of_each_gold_bar_l82_82149


namespace total_valid_votes_l82_82840

theorem total_valid_votes (V : ℝ)
  (h1 : ∃ c1 c2 : ℝ, c1 = 0.70 * V ∧ c2 = 0.30 * V)
  (h2 : ∀ c1 c2, c1 - c2 = 182) : V = 455 :=
sorry

end total_valid_votes_l82_82840


namespace symmetrical_implies_congruent_l82_82320

-- Define a structure to represent figures
structure Figure where
  segments : Set ℕ
  angles : Set ℕ

-- Define symmetry about a line
def is_symmetrical_about_line (f1 f2 : Figure) : Prop :=
  ∀ s ∈ f1.segments, s ∈ f2.segments ∧ ∀ a ∈ f1.angles, a ∈ f2.angles

-- Define congruent figures
def are_congruent (f1 f2 : Figure) : Prop :=
  f1.segments = f2.segments ∧ f1.angles = f2.angles

-- Lean 4 statement of the proof problem
theorem symmetrical_implies_congruent (f1 f2 : Figure) (h : is_symmetrical_about_line f1 f2) : are_congruent f1 f2 :=
by
  sorry

end symmetrical_implies_congruent_l82_82320


namespace numbers_combination_to_24_l82_82274

theorem numbers_combination_to_24 :
  (40 / 4) + 12 + 2 = 24 :=
by
  sorry

end numbers_combination_to_24_l82_82274


namespace intersection_is_target_set_l82_82327

-- Define sets A and B
def is_in_A (x : ℝ) : Prop := |x - 1| < 2
def is_in_B (x : ℝ) : Prop := x^2 < 4

-- Define the intersection A ∩ B
def is_in_intersection (x : ℝ) : Prop := is_in_A x ∧ is_in_B x

-- Define the target set
def is_in_target_set (x : ℝ) : Prop := -1 < x ∧ x < 2

-- Statement to prove
theorem intersection_is_target_set : 
  ∀ x : ℝ, is_in_intersection x ↔ is_in_target_set x := sorry

end intersection_is_target_set_l82_82327


namespace westbound_cyclist_speed_increase_l82_82791

def eastbound_speed : ℕ := 18
def travel_time : ℕ := 6
def total_distance : ℕ := 246

theorem westbound_cyclist_speed_increase (x : ℕ) :
  eastbound_speed * travel_time + (eastbound_speed + x) * travel_time = total_distance →
  x = 5 :=
by
  sorry

end westbound_cyclist_speed_increase_l82_82791


namespace minimum_radius_of_third_sphere_l82_82329

noncomputable def cone_height : ℝ := 4
noncomputable def cone_base_radius : ℝ := 3

noncomputable def radius_identical_spheres : ℝ := 4 / 3  -- derived from the conditions

theorem minimum_radius_of_third_sphere
    (h r1 r2 : ℝ) -- heights and radii one and two
    (R1 R2 Rb : ℝ) -- radii of the common base
    (cond_h : h = 4)
    (cond_Rb : Rb = 3)
    (cond_radii_eq : r1 = r2) 
  : r2 = 27 / 35 :=
by
  sorry

end minimum_radius_of_third_sphere_l82_82329


namespace prime_gt_3_div_24_num_form_6n_plus_minus_1_div_24_l82_82787

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_gt_3_div_24 (p : ℕ) (hp : is_prime p) (h : p > 3) : 
  24 ∣ (p^2 - 1) :=
sorry

theorem num_form_6n_plus_minus_1_div_24 (n : ℕ) : 
  24 ∣ (6 * n + 1)^2 - 1 ∧ 24 ∣ (6 * n - 1)^2 - 1 :=
sorry

end prime_gt_3_div_24_num_form_6n_plus_minus_1_div_24_l82_82787


namespace divide_area_into_squares_l82_82024

theorem divide_area_into_squares :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x / y = 4 / 3 ∧ (x^2 + y^2 = 100) ∧ x = 8 ∧ y = 6) := 
by {
  sorry
}

end divide_area_into_squares_l82_82024


namespace find_y_l82_82795

theorem find_y (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) 
  (h : (2 * a) ^ (2 * b ^ 2) = (a ^ b + y ^ b) ^ 2) : y = 4 * a ^ 2 - a := 
sorry

end find_y_l82_82795


namespace yellow_paint_amount_l82_82518

theorem yellow_paint_amount (b y : ℕ) (h_ratio : y * 7 = 3 * b) (h_blue_amount : b = 21) : y = 9 :=
by
  sorry

end yellow_paint_amount_l82_82518


namespace find_x_minus_y_l82_82590

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  -- Proof omitted
  sorry

end find_x_minus_y_l82_82590


namespace greatest_divisible_by_11_l82_82077

theorem greatest_divisible_by_11 :
  ∃ (A B C : ℕ), A ≠ C ∧ A ≠ B ∧ B ≠ C ∧ 
  (∀ n, n = 10000 * A + 1000 * B + 100 * C + 10 * B + A → n = 96569) ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 11 = 0 :=
sorry

end greatest_divisible_by_11_l82_82077


namespace transformed_circle_eq_l82_82270

theorem transformed_circle_eq (x y : ℝ) (h : x^2 + y^2 = 1) : x^2 + 9 * (y / 3)^2 = 1 := by
  sorry

end transformed_circle_eq_l82_82270


namespace total_cost_is_90_l82_82766

variable (jackets : ℕ) (shirts : ℕ) (pants : ℕ)
variable (price_jacket : ℕ) (price_shorts : ℕ) (price_pants : ℕ)

theorem total_cost_is_90 
  (h1 : jackets = 3)
  (h2 : price_jacket = 10)
  (h3 : shirts = 2)
  (h4 : price_shorts = 6)
  (h5 : pants = 4)
  (h6 : price_pants = 12) : 
  (jackets * price_jacket + shirts * price_shorts + pants * price_pants) = 90 := by 
  sorry

end total_cost_is_90_l82_82766


namespace amount_of_bill_l82_82731

theorem amount_of_bill (TD R FV T : ℝ) (hTD : TD = 270) (hR : R = 16) (hT : T = 9/12) 
(h_formula : TD = (R * T * FV) / (100 + (R * T))) : FV = 2520 :=
by
  sorry

end amount_of_bill_l82_82731


namespace pencils_per_student_l82_82084

theorem pencils_per_student
  (boxes : ℝ) (pencils_per_box : ℝ) (students : ℝ)
  (h1 : boxes = 4.0)
  (h2 : pencils_per_box = 648.0)
  (h3 : students = 36.0) :
  (boxes * pencils_per_box) / students = 72.0 :=
by
  sorry

end pencils_per_student_l82_82084


namespace minimize_sum_of_squares_if_and_only_if_l82_82188

noncomputable def minimize_sum_of_squares (a b c S : ℝ) (O : ℝ×ℝ×ℝ) (x y z : ℝ) : Prop :=
  let ax_by_cz := a * x + b * y + c * z
  ax_by_cz = 2 * S ∧
  x/y = a/b ∧
  y/z = b/c ∧
  x/z = a/c

theorem minimize_sum_of_squares_if_and_only_if (a b c S : ℝ) (O : ℝ×ℝ×ℝ) (x y z : ℝ) :
  (∃ P : ℝ, minimize_sum_of_squares a b c S O x y z) ↔ (x/y = a/b ∧ y/z = b/c ∧ x/z = a/c) := sorry

end minimize_sum_of_squares_if_and_only_if_l82_82188


namespace percentage_good_oranges_tree_A_l82_82563

theorem percentage_good_oranges_tree_A
  (total_trees : ℕ)
  (trees_A : ℕ)
  (trees_B : ℕ)
  (total_good_oranges : ℕ)
  (oranges_A_per_month : ℕ) 
  (oranges_B_per_month : ℕ)
  (good_oranges_B_ratio : ℚ)
  (good_oranges_total_B : ℕ) 
  (good_oranges_total_A : ℕ)
  (good_oranges_total : ℕ)
  (x : ℚ) 
  (total_trees_eq : total_trees = 10)
  (tree_percentage_eq : trees_A = total_trees / 2 ∧ trees_B = total_trees / 2)
  (oranges_A_per_month_eq : oranges_A_per_month = 10)
  (oranges_B_per_month_eq : oranges_B_per_month = 15)
  (good_oranges_B_ratio_eq : good_oranges_B_ratio = 1/3)
  (good_oranges_total_eq : total_good_oranges = 55)
  (good_oranges_total_B_eq : good_oranges_total_B = trees_B * oranges_B_per_month * good_oranges_B_ratio)
  (good_oranges_total_A_eq : good_oranges_total_A = total_good_oranges - good_oranges_total_B):
  trees_A * oranges_A_per_month * x = good_oranges_total_A → 
  x = 0.6 := by
  sorry

end percentage_good_oranges_tree_A_l82_82563


namespace evaluate_f_at_neg3_l82_82394

def f (x : ℝ) : ℝ := -2 * x^3 + 5 * x^2 - 3 * x + 2

theorem evaluate_f_at_neg3 : f (-3) = 110 :=
by 
  sorry

end evaluate_f_at_neg3_l82_82394


namespace cindy_dress_discount_l82_82606

theorem cindy_dress_discount (P D : ℝ) 
  (h1 : P * (1 - D) * 1.25 = 61.2) 
  (h2 : P - 61.2 = 4.5) : D = 0.255 :=
sorry

end cindy_dress_discount_l82_82606


namespace minimize_sum_first_n_terms_l82_82893

noncomputable def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

noncomputable def sum_first_n_terms (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n-1) / 2) * d

theorem minimize_sum_first_n_terms (a₁ : ℤ) (a₃_plus_a₅ : ℤ) (n_min : ℕ) :
  a₁ = -9 → a₃_plus_a₅ = -6 → n_min = 5 := by
  sorry

end minimize_sum_first_n_terms_l82_82893


namespace bounded_regions_l82_82528

noncomputable def regions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => regions n + n + 1

theorem bounded_regions (n : ℕ) :
  (regions n = n * (n + 1) / 2 + 1) := by
  sorry

end bounded_regions_l82_82528


namespace slopes_of_intersecting_line_l82_82366

theorem slopes_of_intersecting_line {m : ℝ} :
  (∃ x y : ℝ, y = m * x + 4 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ Set.Iic (-Real.sqrt 0.48) ∪ Set.Ici (Real.sqrt 0.48) :=
by
  sorry

end slopes_of_intersecting_line_l82_82366


namespace find_side_a_l82_82334

noncomputable def maximum_area (A b c : ℝ) : Prop :=
  A = 2 * Real.pi / 3 ∧ (b + 2 * c = 8) ∧ 
  ((1 / 2) * b * c * Real.sin (2 * Real.pi / 3) = (Real.sqrt 3 / 2) * c * (4 - c) ∧ 
   (∀ (c' : ℝ), (Real.sqrt 3 / 2) * c' * (4 - c') ≤ 2 * Real.sqrt 3) ∧ 
   c = 2)

theorem find_side_a (A b c a : ℝ) (h : maximum_area A b c) :
  a = 2 * Real.sqrt 7 := 
by
  sorry

end find_side_a_l82_82334


namespace james_final_sticker_count_l82_82971

-- Define the conditions
def initial_stickers := 478
def gift_stickers := 182
def given_away_stickers := 276

-- Define the correct answer
def final_stickers := 384

-- State the theorem
theorem james_final_sticker_count :
  initial_stickers + gift_stickers - given_away_stickers = final_stickers :=
by
  sorry

end james_final_sticker_count_l82_82971


namespace taimour_time_to_paint_alone_l82_82991

theorem taimour_time_to_paint_alone (T : ℝ) (h1 : Jamshid_time = T / 2)
  (h2 : (1 / T + 1 / (T / 2)) = 1 / 3) : T = 9 :=
sorry

end taimour_time_to_paint_alone_l82_82991


namespace total_distance_correct_l82_82118

def liters_U := 50
def liters_V := 50
def liters_W := 50
def liters_X := 50

def fuel_efficiency_U := 20 -- liters per 100 km
def fuel_efficiency_V := 25 -- liters per 100 km
def fuel_efficiency_W := 5 -- liters per 100 km
def fuel_efficiency_X := 10 -- liters per 100 km

def distance_U := (liters_U / fuel_efficiency_U) * 100 -- Distance for U in km
def distance_V := (liters_V / fuel_efficiency_V) * 100 -- Distance for V in km
def distance_W := (liters_W / fuel_efficiency_W) * 100 -- Distance for W in km
def distance_X := (liters_X / fuel_efficiency_X) * 100 -- Distance for X in km

def total_distance := distance_U + distance_V + distance_W + distance_X -- Total distance of all cars

theorem total_distance_correct :
  total_distance = 1950 := 
by {
  sorry
}

end total_distance_correct_l82_82118


namespace max_product_two_four_digit_numbers_l82_82182

theorem max_product_two_four_digit_numbers :
  ∃ (a b : ℕ), 
    (a * b = max (8564 * 7321) (8531 * 7642)) 
    ∧ max 8531 8564 = 8531 ∧ 
    (∀ x y : ℕ, x * y ≤ 8531 * 7642 → x * y = max (8564 * 7321) (8531 * 7642)) :=
sorry

end max_product_two_four_digit_numbers_l82_82182


namespace ratio_consequent_l82_82423

theorem ratio_consequent (a b x : ℕ) (h_ratio : a = 4) (h_b : b = 6) (h_x : x = 30) :
  (a : ℚ) / b = x / 45 := 
by 
  -- add here the necessary proof steps 
  sorry

end ratio_consequent_l82_82423


namespace problem_statement_l82_82157

open Real

noncomputable def log4 (x : ℝ) : ℝ := log x / log 4

noncomputable def a : ℝ := log4 (sqrt 5)
noncomputable def b : ℝ := log 2 / log 5
noncomputable def c : ℝ := log4 5

theorem problem_statement : b < a ∧ a < c :=
by
  sorry

end problem_statement_l82_82157


namespace unit_squares_in_50th_ring_l82_82581

-- Definitions from the conditions
def unit_squares_in_first_ring : ℕ := 12

def unit_squares_in_nth_ring (n : ℕ) : ℕ :=
  32 * n - 16

-- Prove the specific instance for the 50th ring
theorem unit_squares_in_50th_ring : unit_squares_in_nth_ring 50 = 1584 :=
by
  sorry

end unit_squares_in_50th_ring_l82_82581


namespace quadratic_root_c_l82_82002

theorem quadratic_root_c (c : ℝ) :
  (∀ x : ℝ, x^2 + 3 * x + c = (x + (3/2))^2 - 7/4) → c = 1/2 :=
by
  sorry

end quadratic_root_c_l82_82002


namespace a_perp_a_minus_b_l82_82929

noncomputable def a : ℝ × ℝ := (-2, 1)
noncomputable def b : ℝ × ℝ := (-1, 3)
noncomputable def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

theorem a_perp_a_minus_b : (a.1 * a_minus_b.1 + a.2 * a_minus_b.2) = 0 := by
  sorry

end a_perp_a_minus_b_l82_82929


namespace find_cos_value_l82_82534

theorem find_cos_value (α : Real) 
  (h : Real.cos (Real.pi / 8 - α) = 1 / 6) : 
  Real.cos (3 * Real.pi / 4 + 2 * α) = 17 / 18 :=
by
  sorry

end find_cos_value_l82_82534


namespace water_percentage_in_dried_grapes_l82_82839

noncomputable def fresh_grape_weight : ℝ := 40  -- weight of fresh grapes in kg
noncomputable def dried_grape_weight : ℝ := 5  -- weight of dried grapes in kg
noncomputable def water_percentage_fresh : ℝ := 0.90  -- percentage of water in fresh grapes

noncomputable def water_weight_fresh : ℝ := fresh_grape_weight * water_percentage_fresh
noncomputable def solid_weight_fresh : ℝ := fresh_grape_weight * (1 - water_percentage_fresh)
noncomputable def water_weight_dried : ℝ := dried_grape_weight - solid_weight_fresh
noncomputable def water_percentage_dried : ℝ := (water_weight_dried / dried_grape_weight) * 100

theorem water_percentage_in_dried_grapes : water_percentage_dried = 20 := by
  sorry

end water_percentage_in_dried_grapes_l82_82839


namespace find_initial_number_l82_82377

theorem find_initial_number (N : ℝ) (h : ∃ k : ℝ, 330 * k = N + 69.00000000008731) : 
  ∃ m : ℝ, N = 330 * m - 69.00000000008731 :=
by
  sorry

end find_initial_number_l82_82377


namespace Balaganov_made_a_mistake_l82_82505

variable (n1 n2 n3 : ℕ) (x : ℝ)
variable (average : ℝ)

def total_salary (n1 n2 : ℕ) (x : ℝ) (n3 : ℕ) : ℝ := 27 * n1 + 35 * n2 + x * n3

def number_of_employees (n1 n2 n3 : ℕ) : ℕ := n1 + n2 + n3

noncomputable def calculated_average_salary (n1 n2 : ℕ) (x : ℝ) (n3 : ℕ) : ℝ :=
 total_salary n1 n2 x n3 / number_of_employees n1 n2 n3

theorem Balaganov_made_a_mistake (h₀ : n1 > n2) 
  (h₁ : calculated_average_salary n1 n2 x n3 = average) 
  (h₂ : 31 < average) : false :=
sorry

end Balaganov_made_a_mistake_l82_82505


namespace construct_3x3x3_cube_l82_82995

theorem construct_3x3x3_cube :
  ∃ (cubes_1x2x2 : Finset (Set (Fin 3 × Fin 3 × Fin 3))),
  ∃ (cubes_1x1x1 : Finset (Fin 3 × Fin 3 × Fin 3)),
  cubes_1x2x2.card = 6 ∧ 
  cubes_1x1x1.card = 3 ∧ 
  (∀ c ∈ cubes_1x2x2, ∃ a b : Fin 3, ∀ x, x = (a, b, 0) ∨ x = (a, b, 1) ∨ x = (a, b, 2)) ∧
  (∀ c ∈ cubes_1x1x1, ∃ a b c : Fin 3, ∀ x, x = (a, b, c)) :=
sorry

end construct_3x3x3_cube_l82_82995


namespace sara_remaining_red_balloons_l82_82263

-- Given conditions
def initial_red_balloons := 31
def red_balloons_given := 24

-- Statement to prove
theorem sara_remaining_red_balloons : (initial_red_balloons - red_balloons_given = 7) :=
by
  -- Proof can be skipped
  sorry

end sara_remaining_red_balloons_l82_82263


namespace range_of_function_l82_82780

theorem range_of_function (x y z : ℝ)
  (h : x^2 + y^2 + x - y = 1) :
  ∃ a b : ℝ, (a = (3 * Real.sqrt 6 + Real.sqrt 6) / 2) ∧ (b = (-3 * Real.sqrt 2 + Real.sqrt 6) / 2) ∧
    ∀ f : ℝ, f = (x - 1) * Real.cos z + (y + 1) * Real.sin z →
              b ≤ f ∧ f ≤ a := 
by
  sorry

end range_of_function_l82_82780


namespace maximum_possible_median_l82_82301

theorem maximum_possible_median
  (total_cans : ℕ)
  (total_customers : ℕ)
  (min_cans_per_customer : ℕ)
  (alt_min_cans_per_customer : ℕ)
  (exact_min_cans_count : ℕ)
  (atleast_min_cans_count : ℕ)
  (min_cans_customers : ℕ)
  (alt_min_cans_customer: ℕ): 
  (total_cans = 300) → 
  (total_customers = 120) →
  (min_cans_per_customer = 2) →
  (alt_min_cans_per_customer = 4) →
  (min_cans_customers = 59) →
  (alt_min_cans_customer = 61) →
  (min_cans_per_customer * min_cans_customers + alt_min_cans_per_customer * (total_customers - min_cans_customers) = total_cans) →
  max (min_cans_per_customer + 1) (alt_min_cans_per_customer - 1) = 3 :=
sorry

end maximum_possible_median_l82_82301


namespace BANANA_arrangements_l82_82701

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l82_82701


namespace example_inequality_l82_82241

variable (a b c : ℝ)

theorem example_inequality 
  (h : a^6 + b^6 + c^6 = 3) : a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := 
by
  sorry

end example_inequality_l82_82241


namespace find_quadruples_l82_82524

theorem find_quadruples (a b p n : ℕ) (h_prime : Prime p) (h_eq : a^3 + b^3 = p^n) :
  ∃ k : ℕ, (a, b, p, n) = (2^k, 2^k, 2, 3*k + 1) ∨ 
           (a, b, p, n) = (3^k, 2 * 3^k, 3, 3*k + 2) ∨ 
           (a, b, p, n) = (2 * 3^k, 3^k, 3, 3*k + 2) :=
sorry

end find_quadruples_l82_82524


namespace base3_20121_to_base10_l82_82156

def base3_to_base10 (n : ℕ) : ℕ :=
  2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0

theorem base3_20121_to_base10 :
  base3_to_base10 20121 = 178 :=
by
  sorry

end base3_20121_to_base10_l82_82156


namespace sum_of_number_and_reverse_l82_82338

theorem sum_of_number_and_reverse :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → ((10 * a + b) - (10 * b + a) = 7 * (a + b)) → a = 8 * b → 
  (10 * a + b) + (10 * b + a) = 99 :=
by
  intros a b conditions eq diff
  sorry

end sum_of_number_and_reverse_l82_82338


namespace hyperbola_focus_l82_82179

theorem hyperbola_focus :
  ∃ (x y : ℝ), 2 * x^2 - y^2 - 8 * x + 4 * y - 4 = 0 ∧ (x, y) = (2 + 2 * Real.sqrt 3, 2) :=
by
  -- The proof would go here
  sorry

end hyperbola_focus_l82_82179


namespace inequality_preservation_l82_82532

theorem inequality_preservation (x y : ℝ) (h : x < y) : 2 * x < 2 * y :=
sorry

end inequality_preservation_l82_82532


namespace skill_of_passing_through_walls_l82_82619

theorem skill_of_passing_through_walls (k n : ℕ) (h : k = 8) (h_eq : k * Real.sqrt (k / (k * k - 1)) = Real.sqrt (k * k / (k * k - 1))) : n = k * k - 1 :=
by sorry

end skill_of_passing_through_walls_l82_82619


namespace sequence_term_2010_l82_82911

theorem sequence_term_2010 :
  ∀ (a : ℕ → ℤ), a 1 = 1 → a 2 = 2 → 
    (∀ n : ℕ, n ≥ 3 → a n = a (n - 1) - a (n - 2)) → 
    a 2010 = -1 :=
by
  sorry

end sequence_term_2010_l82_82911


namespace option_A_sufficient_not_necessary_l82_82443

variable (a b : ℝ)

def A : Set ℝ := { x | x^2 - x + a ≤ 0 }
def B : Set ℝ := { x | x^2 - x + b ≤ 0 }

theorem option_A_sufficient_not_necessary : (A = B → a = b) ∧ (a = b → A = B) :=
by
  sorry

end option_A_sufficient_not_necessary_l82_82443


namespace gcd_14m_21n_126_l82_82722

theorem gcd_14m_21n_126 {m n : ℕ} (hm_pos : 0 < m) (hn_pos : 0 < n) (h_gcd : Nat.gcd m n = 18) : 
  Nat.gcd (14 * m) (21 * n) = 126 :=
by
  sorry

end gcd_14m_21n_126_l82_82722


namespace polar_line_equation_l82_82874

/-- A line that passes through a given point in polar coordinates and is parallel to the polar axis
    has a specific polar coordinate equation. -/
theorem polar_line_equation (r : ℝ) (θ : ℝ) (h : r = 6 ∧ θ = π / 6) : θ = π / 6 :=
by
  /- We are given that the line passes through the point \(C(6, \frac{\pi}{6})\) which means
     \(r = 6\) and \(θ = \frac{\pi}{6}\). Since the line is parallel to the polar axis, 
     the angle \(θ\) remains the same. Therefore, the polar coordinate equation of the line 
     is simply \(θ = \frac{\pi}{6}\). -/
  sorry

end polar_line_equation_l82_82874


namespace parallel_vectors_result_l82_82866

noncomputable def a : ℝ × ℝ := (1, -2)
noncomputable def b (m : ℝ) : ℝ × ℝ := (m, 4)
noncomputable def m : ℝ := -1 / 2

theorem parallel_vectors_result :
  (b m).1 * a.2 = (b m).2 * a.1 →
  2 * a - b m = (4, -8) :=
by
  intro h
  -- Proof omitted
  sorry

end parallel_vectors_result_l82_82866


namespace sum_of_three_numbers_l82_82517

theorem sum_of_three_numbers (x y z : ℝ) (h₁ : x + y = 29) (h₂ : y + z = 46) (h₃ : z + x = 53) : x + y + z = 64 :=
by
  sorry

end sum_of_three_numbers_l82_82517


namespace blocks_from_gallery_to_work_l82_82382

theorem blocks_from_gallery_to_work (b_store b_gallery b_already_walked b_more_to_work total_blocks blocks_to_work_from_gallery : ℕ) 
  (h1 : b_store = 11)
  (h2 : b_gallery = 6)
  (h3 : b_already_walked = 5)
  (h4 : b_more_to_work = 20)
  (h5 : total_blocks = b_store + b_gallery + b_more_to_work)
  (h6 : blocks_to_work_from_gallery = total_blocks - b_already_walked - b_store - b_gallery) :
  blocks_to_work_from_gallery = 15 :=
by
  sorry

end blocks_from_gallery_to_work_l82_82382


namespace problem1_problem2_l82_82691

theorem problem1 : 2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 3 * Real.sqrt 3 := by
  sorry

theorem problem2 : (Real.sqrt 18 - Real.sqrt 3) * Real.sqrt 12 = 6 * Real.sqrt 6 - 6 := by
  sorry

end problem1_problem2_l82_82691


namespace necessary_but_not_sufficient_l82_82288

-- Define α as an interior angle of triangle ABC
def is_interior_angle_of_triangle (α : ℝ) : Prop :=
  0 < α ∧ α < 180

-- Define the sine condition
def sine_condition (α : ℝ) : Prop :=
  Real.sin α = Real.sqrt 2 / 2

-- Define the main theorem
theorem necessary_but_not_sufficient (α : ℝ) (h1 : is_interior_angle_of_triangle α) (h2 : sine_condition α) :
  (sine_condition α) ↔ (α = 45) ∨ (α = 135) := by
  sorry

end necessary_but_not_sufficient_l82_82288


namespace LittleJohnnyAnnualIncome_l82_82989

theorem LittleJohnnyAnnualIncome :
  ∀ (total_amount bank_amount bond_amount : ℝ) 
    (bank_interest bond_interest annual_income : ℝ),
    total_amount = 10000 →
    bank_amount = 6000 →
    bond_amount = 4000 →
    bank_interest = 0.05 →
    bond_interest = 0.09 →
    annual_income = bank_amount * bank_interest + bond_amount * bond_interest →
    annual_income = 660 :=
by
  intros total_amount bank_amount bond_amount bank_interest bond_interest annual_income 
  intros h_total_amount h_bank_amount h_bond_amount h_bank_interest h_bond_interest h_annual_income
  -- Proof is not required
  sorry

end LittleJohnnyAnnualIncome_l82_82989


namespace false_propositions_count_l82_82473

-- Definitions of the propositions
def proposition1 (A B : Prop) (P : Prop) : Prop :=
  P ∧ ¬ P

def proposition2 (A B : Prop) (P : Prop) : Prop :=
  P ∧ ¬ P

def proposition3 (A B : Prop) : Prop :=
  ¬ (A ∧ B)

def proposition4 (A B : Prop) : Prop :=
  A ∧ B

-- Theorem to prove the total number of false propositions
theorem false_propositions_count (A B : Prop) (P1 P2 P3 P4 : Prop) :
  ¬ (proposition1 A B P1) ∧ ¬ (proposition2 A B P2) ∧ ¬ (proposition3 A B) ∧ proposition4 A B → 3 = 3 :=
by
  intro h
  sorry

end false_propositions_count_l82_82473


namespace quadrilateral_area_offset_l82_82065

theorem quadrilateral_area_offset
  (d : ℝ) (x : ℝ) (y : ℝ) (A : ℝ)
  (h_d : d = 26)
  (h_y : y = 6)
  (h_A : A = 195) :
  A = 1/2 * (x + y) * d → x = 9 :=
by
  sorry

end quadrilateral_area_offset_l82_82065


namespace notebooks_last_days_l82_82472

-- Given conditions
def n := 5
def p := 40
def u := 4

-- Derived conditions
def total_pages := n * p
def days := total_pages / u

-- The theorem statement
theorem notebooks_last_days : days = 50 := sorry

end notebooks_last_days_l82_82472


namespace smallest_x_for_cubic_1890_l82_82895

theorem smallest_x_for_cubic_1890 (x : ℕ) (N : ℕ) (hx : 1890 * x = N ^ 3) : x = 4900 :=
sorry

end smallest_x_for_cubic_1890_l82_82895


namespace no_sum_of_cubes_eq_2002_l82_82592

theorem no_sum_of_cubes_eq_2002 :
  ¬ ∃ (a b c : ℕ), (a ^ 3 + b ^ 3 + c ^ 3 = 2002) :=
sorry

end no_sum_of_cubes_eq_2002_l82_82592


namespace least_five_digit_perfect_square_and_cube_l82_82500

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l82_82500


namespace lcm_5_6_8_18_l82_82633

/-- The least common multiple of the numbers 5, 6, 8, and 18 is 360. -/
theorem lcm_5_6_8_18 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 18) = 360 := by
  sorry

end lcm_5_6_8_18_l82_82633


namespace distinct_arrangements_balloon_l82_82965

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l82_82965


namespace coprime_integer_pairs_sum_285_l82_82593

theorem coprime_integer_pairs_sum_285 : 
  (∃ s : Finset (ℕ × ℕ), 
    ∀ p ∈ s, p.1 + p.2 = 285 ∧ Nat.gcd p.1 p.2 = 1 ∧ s.card = 72) := sorry

end coprime_integer_pairs_sum_285_l82_82593


namespace system_of_equations_correct_l82_82467

theorem system_of_equations_correct (x y : ℤ) :
  (8 * x - 3 = y) ∧ (7 * x + 4 = y) :=
sorry

end system_of_equations_correct_l82_82467


namespace quadrilateral_area_l82_82208

theorem quadrilateral_area (a b c d e f : ℝ) : 
    (a^2 + c^2 - b^2 - d^2) ^ 2 ≤ 4 * e^2 * f^2 :=
    by sorry

noncomputable def quadrilateral_area_formula (a b c d e f : ℝ) : ℝ :=
    if H : (a^2 + c^2 - b^2 - d^2) ^ 2 ≤ 4 * e^2 * f^2 then 
    (1/4) * Real.sqrt (4 * e^2 * f^2 - (a^2 + c^2 - b^2 - d^2) ^ 2)
    else 0

-- Ensure that the computed area matches the expected value
example (a b c d e f : ℝ) (H : (a^2 + c^2 - b^2 - d^2)^2 ≤ 4 * e^2 * f^2) : 
    quadrilateral_area_formula a b c d e f = 
        (1/4) * Real.sqrt (4 * e^2 * f^2 - (a^2 + c^2 - b^2 - d^2) ^ 2) :=
by simp [quadrilateral_area_formula, H]

end quadrilateral_area_l82_82208


namespace sum_and_difference_repeating_decimals_l82_82769

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_9 : ℚ := 1
noncomputable def repeating_decimal_3 : ℚ := 1 / 3

theorem sum_and_difference_repeating_decimals :
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_9 + repeating_decimal_3 = 2 / 9 := 
by 
  sorry

end sum_and_difference_repeating_decimals_l82_82769


namespace point_relationship_l82_82060

variables {m x1 x2 y1 y2 : ℝ}

def quadratic_function (x : ℝ) (m : ℝ) : ℝ :=
  (x + m - 3)*(x - m) + 3

theorem point_relationship 
  (hx1_lt_x2 : x1 < x2)
  (hA : y1 = quadratic_function x1 m)
  (hB : y2 = quadratic_function x2 m)
  (h_sum_lt : x1 + x2 < 3) :
  y1 > y2 :=
sorry

end point_relationship_l82_82060


namespace min_ab_l82_82325

theorem min_ab (a b : ℝ) (h_cond1 : a > 0) (h_cond2 : b > 0)
  (h_eq : a * b = a + b + 3) : a * b = 9 :=
sorry

end min_ab_l82_82325


namespace odd_function_property_l82_82565

theorem odd_function_property {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = - f x) (h2 : ∀ x, f (1 + x) = f (-x)) (h3 : f (-1 / 3) = 1 / 3) : f (5 / 3) = 1 / 3 := 
sorry

end odd_function_property_l82_82565


namespace angelina_speed_from_grocery_to_gym_l82_82461

theorem angelina_speed_from_grocery_to_gym
    (v : ℝ)
    (hv : v > 0)
    (home_to_grocery_distance : ℝ := 150)
    (grocery_to_gym_distance : ℝ := 200)
    (time_difference : ℝ := 10)
    (time_home_to_grocery : ℝ := home_to_grocery_distance / v)
    (time_grocery_to_gym : ℝ := grocery_to_gym_distance / (2 * v))
    (h_time_diff : time_home_to_grocery - time_grocery_to_gym = time_difference) :
    2 * v = 10 := by
  sorry

end angelina_speed_from_grocery_to_gym_l82_82461


namespace number_of_new_trailer_homes_l82_82672

-- Definitions coming from the conditions
def initial_trailers : ℕ := 30
def initial_avg_age : ℕ := 15
def years_passed : ℕ := 5
def current_avg_age : ℕ := initial_avg_age + years_passed

-- Let 'n' be the number of new trailer homes added five years ago
variable (n : ℕ)

def new_trailer_age : ℕ := years_passed
def total_trailers : ℕ := initial_trailers + n
def total_ages : ℕ := (initial_trailers * current_avg_age) + (n * new_trailer_age)
def combined_avg_age := total_ages / total_trailers

theorem number_of_new_trailer_homes (h : combined_avg_age = 12) : n = 34 := 
sorry

end number_of_new_trailer_homes_l82_82672


namespace gcd_765432_654321_l82_82569

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l82_82569


namespace factor_expr_l82_82629

def expr1 (x : ℝ) := 16 * x^6 + 49 * x^4 - 9
def expr2 (x : ℝ) := 4 * x^6 - 14 * x^4 - 9

theorem factor_expr (x : ℝ) :
  (expr1 x - expr2 x) = 3 * x^4 * (4 * x^2 + 21) := 
by
  sorry

end factor_expr_l82_82629


namespace exists_m_for_n_divides_2_pow_m_plus_m_l82_82069

theorem exists_m_for_n_divides_2_pow_m_plus_m (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, 0 < m ∧ n ∣ 2^m + m :=
sorry

end exists_m_for_n_divides_2_pow_m_plus_m_l82_82069


namespace spherical_cap_surface_area_l82_82724

theorem spherical_cap_surface_area (V : ℝ) (h : ℝ) (A : ℝ) (r : ℝ) 
  (volume_eq : V = (4 / 3) * π * r^3) 
  (cap_height : h = 2) 
  (sphere_volume : V = 288 * π) 
  (cap_surface_area : A = 2 * π * r * h) : 
  A = 24 * π := 
sorry

end spherical_cap_surface_area_l82_82724


namespace admin_staff_in_sample_l82_82173

theorem admin_staff_in_sample (total_staff : ℕ) (admin_staff : ℕ) (total_samples : ℕ)
  (probability : ℚ) (h1 : total_staff = 200) (h2 : admin_staff = 24)
  (h3 : total_samples = 50) (h4 : probability = 50 / 200) :
  admin_staff * probability = 6 :=
by
  -- Proof goes here
  sorry

end admin_staff_in_sample_l82_82173


namespace remaining_days_temperature_l82_82115

theorem remaining_days_temperature (avg_temp : ℕ) (d1 d2 d3 d4 d5 : ℕ) :
  avg_temp = 60 →
  d1 = 40 →
  d2 = 40 →
  d3 = 40 →
  d4 = 80 →
  d5 = 80 →
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  total_temp - known_temp = 140 := 
by
  intros _ _ _ _ _ _
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  sorry

end remaining_days_temperature_l82_82115


namespace initial_amount_l82_82271

theorem initial_amount 
  (spend1 spend2 left : ℝ)
  (hspend1 : spend1 = 1.75) 
  (hspend2 : spend2 = 1.25) 
  (hleft : left = 6.00) : 
  spend1 + spend2 + left = 9.00 := 
by
  -- Proof is omitted
  sorry

end initial_amount_l82_82271


namespace closest_ratio_adults_children_l82_82497

theorem closest_ratio_adults_children :
  ∃ (a c : ℕ), 25 * a + 15 * c = 1950 ∧ a ≥ 1 ∧ c ≥ 1 ∧ a / c = 24 / 25 := sorry

end closest_ratio_adults_children_l82_82497


namespace no_integer_root_l82_82287

theorem no_integer_root (q : ℤ) : ¬ ∃ x : ℤ, x^2 + 7 * x - 14 * (q^2 + 1) = 0 := sorry

end no_integer_root_l82_82287


namespace max_value_abs_x_sub_3y_l82_82405

theorem max_value_abs_x_sub_3y 
  (x y : ℝ)
  (h1 : y ≥ x)
  (h2 : x + 3 * y ≤ 4)
  (h3 : x ≥ -2) : 
  ∃ z, z = |x - 3 * y| ∧ ∀ (x y : ℝ), (y ≥ x) → (x + 3 * y ≤ 4) → (x ≥ -2) → |x - 3 * y| ≤ 4 :=
sorry

end max_value_abs_x_sub_3y_l82_82405


namespace symmetric_points_l82_82470

theorem symmetric_points (a b : ℤ) (h1 : (a, -2) = (1, -2)) (h2 : (-1, b) = (-1, -2)) :
  (a + b) ^ 2023 = -1 := by
  -- We know from the conditions:
  -- (a, -2) and (1, -2) implies a = 1
  -- (-1, b) and (-1, -2) implies b = -2
  -- Thus it follows that:
  sorry

end symmetric_points_l82_82470


namespace calculator_sum_l82_82447

theorem calculator_sum :
  let A := 2
  let B := 0
  let C := -1
  let D := 3
  let n := 47
  let A' := if n % 2 = 1 then -A else A
  let B' := B -- B remains 0 after any number of sqrt operations
  let C' := if n % 2 = 1 then -C else C
  let D' := D ^ (3 ^ n)
  A' + B' + C' + D' = 3 ^ (3 ^ 47) - 3
:= by
  sorry

end calculator_sum_l82_82447


namespace packages_ratio_l82_82400

theorem packages_ratio (packages_yesterday packages_today : ℕ)
  (h1 : packages_yesterday = 80)
  (h2 : packages_today + packages_yesterday = 240) :
  (packages_today / packages_yesterday) = 2 :=
by
  sorry

end packages_ratio_l82_82400


namespace Esha_behind_Anusha_l82_82108

/-- Define conditions for the race -/

def Anusha_speed := 100
def Banu_behind_when_Anusha_finishes := 10
def Banu_run_when_Anusha_finishes := Anusha_speed - Banu_behind_when_Anusha_finishes
def Esha_behind_when_Banu_finishes := 10
def Esha_run_when_Banu_finishes := Anusha_speed - Esha_behind_when_Banu_finishes
def Banu_speed_ratio := Banu_run_when_Anusha_finishes / Anusha_speed
def Esha_speed_ratio := Esha_run_when_Banu_finishes / Anusha_speed
def Esha_to_Anusha_speed_ratio := Esha_speed_ratio * Banu_speed_ratio
def Esha_run_when_Anusha_finishes := Anusha_speed * Esha_to_Anusha_speed_ratio

/-- Prove that Esha is 19 meters behind Anusha when Anusha finishes the race -/
theorem Esha_behind_Anusha {V_A V_B V_E : ℝ} :
  (V_B / V_A = 9 / 10) →
  (V_E / V_B = 9 / 10) →
  (Esha_run_when_Anusha_finishes = Anusha_speed * (9 / 10 * 9 / 10)) →
  Anusha_speed - Esha_run_when_Anusha_finishes = 19 := 
by
  intros h1 h2 h3
  sorry

end Esha_behind_Anusha_l82_82108


namespace quadratic_conditions_l82_82429

open Polynomial

noncomputable def exampleQuadratic (x : ℝ) : ℝ :=
-2 * x^2 + 12 * x - 10

theorem quadratic_conditions :
  (exampleQuadratic 1 = 0) ∧ (exampleQuadratic 5 = 0) ∧ (exampleQuadratic 3 = 8) :=
by
  sorry

end quadratic_conditions_l82_82429


namespace plaster_cost_correct_l82_82536

def length : ℝ := 25
def width : ℝ := 12
def depth : ℝ := 6
def cost_per_sq_meter : ℝ := 0.30

def area_longer_walls : ℝ := 2 * (length * depth)
def area_shorter_walls : ℝ := 2 * (width * depth)
def area_bottom : ℝ := length * width
def total_area : ℝ := area_longer_walls + area_shorter_walls + area_bottom

def calculated_cost : ℝ := total_area * cost_per_sq_meter
def correct_cost : ℝ := 223.2

theorem plaster_cost_correct : calculated_cost = correct_cost := by
  sorry

end plaster_cost_correct_l82_82536


namespace arithmetic_sequence_product_l82_82823

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) (h1 : ∀ n m, n < m → b n < b m) 
(h2 : ∀ n, b (n + 1) - b n = d) (h3 : b 3 * b 4 = 18) : b 2 * b 5 = -80 :=
sorry

end arithmetic_sequence_product_l82_82823


namespace logarithmic_relationship_l82_82016

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithmic_relationship
  (h1 : 0 < Real.cos 1)
  (h2 : Real.cos 1 < Real.sin 1)
  (h3 : Real.sin 1 < 1)
  (h4 : 1 < Real.tan 1) :
  log_base (Real.sin 1) (Real.tan 1) < log_base (Real.cos 1) (Real.tan 1) ∧
  log_base (Real.cos 1) (Real.tan 1) < log_base (Real.cos 1) (Real.sin 1) ∧
  log_base (Real.cos 1) (Real.sin 1) < log_base (Real.sin 1) (Real.cos 1) :=
sorry

end logarithmic_relationship_l82_82016


namespace max_area_of_equilateral_triangle_in_rectangle_l82_82161

noncomputable def maxEquilateralTriangleArea (a b : ℝ) : ℝ :=
  if h : a ≤ b then
    (a^2 * Real.sqrt 3) / 4
  else
    (b^2 * Real.sqrt 3) / 4

theorem max_area_of_equilateral_triangle_in_rectangle :
  maxEquilateralTriangleArea 12 14 = 36 * Real.sqrt 3 :=
by
  sorry

end max_area_of_equilateral_triangle_in_rectangle_l82_82161


namespace x_pow_4_plus_inv_x_pow_4_l82_82613

theorem x_pow_4_plus_inv_x_pow_4 (x : ℝ) (h : x^2 - 15 * x + 1 = 0) : x^4 + (1 / x^4) = 49727 :=
by
  sorry

end x_pow_4_plus_inv_x_pow_4_l82_82613


namespace sqrt7_minus_3_lt_sqrt5_minus_2_l82_82029

theorem sqrt7_minus_3_lt_sqrt5_minus_2:
  (2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3) ∧ (2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3) -> 
  Real.sqrt 7 - 3 < Real.sqrt 5 - 2 := by
  sorry

end sqrt7_minus_3_lt_sqrt5_minus_2_l82_82029


namespace gcd_f_x_x_l82_82525

theorem gcd_f_x_x (x : ℕ) (h : ∃ k : ℕ, x = 35622 * k) :
  Nat.gcd ((3 * x + 4) * (5 * x + 6) * (11 * x + 9) * (x + 7)) x = 378 :=
by
  sorry

end gcd_f_x_x_l82_82525


namespace center_of_circle_l82_82451

-- Defining the equation of the circle as a hypothesis
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y = 0

-- Stating the theorem about the center of the circle
theorem center_of_circle : ∀ x y : ℝ, circle_eq x y → (x = 2 ∧ y = -1) :=
by
  sorry

end center_of_circle_l82_82451


namespace arithmetic_sequence_S6_by_S4_l82_82557

-- Define the arithmetic sequence and the sum function
def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions
def S1 : ℕ := 1
def r (S2 S4 : ℕ) : Prop := S4 / S2 = 4

-- Proof statement
theorem arithmetic_sequence_S6_by_S4 :
  ∀ (a d : ℕ), 
  (sum_arithmetic_sequence a d 1 = S1) → (r (sum_arithmetic_sequence a d 2) (sum_arithmetic_sequence a d 4)) → 
  (sum_arithmetic_sequence a d 6 / sum_arithmetic_sequence a d 4 = 9 / 4) := 
by
  sorry

end arithmetic_sequence_S6_by_S4_l82_82557


namespace block_measure_is_40_l82_82548

def jony_walks (start_time : String) (start_block end_block stop_block : ℕ) (stop_time : String) (speed : ℕ) : ℕ :=
  let total_time := 40 -- walking time in minutes
  let total_distance := speed * total_time -- total distance walked in meters
  let blocks_forward := end_block - start_block -- blocks walked forward
  let blocks_backward := end_block - stop_block -- blocks walked backward
  let total_blocks := blocks_forward + blocks_backward -- total blocks walked
  total_distance / total_blocks

theorem block_measure_is_40 :
  jony_walks "07:00" 10 90 70 "07:40" 100 = 40 := by
  sorry

end block_measure_is_40_l82_82548


namespace triangle_height_l82_82778

theorem triangle_height (base height area : ℝ) (h_base : base = 3) (h_area : area = 6) (h_formula : area = (1/2) * base * height) : height = 4 :=
by
  sorry

end triangle_height_l82_82778


namespace inequality_true_l82_82762

theorem inequality_true (a b : ℝ) (h : a > b) : (2 * a - 1) > (2 * b - 1) :=
by {
  sorry
}

end inequality_true_l82_82762


namespace sum_of_first_9_terms_of_arithmetic_sequence_l82_82655

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sum_of_first_9_terms_of_arithmetic_sequence 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 2 + a 8 = 18) 
  (h3 : sum_of_first_n_terms a S) :
  S 9 = 81 :=
sorry

end sum_of_first_9_terms_of_arithmetic_sequence_l82_82655


namespace gary_asparagus_l82_82514

/-- Formalization of the problem -/
theorem gary_asparagus (A : ℝ) (ha : 700 * 0.50 = 350) (hg : 40 * 2.50 = 100) (hw : 630 = 3 * A + 350 + 100) : A = 60 :=
by
  sorry

end gary_asparagus_l82_82514


namespace average_increase_l82_82578

-- Define the conditions as Lean definitions
def runs_in_17th_inning : ℕ := 50
def average_after_17th_inning : ℕ := 18

-- The condition about the average increase can be written as follows
theorem average_increase 
  (initial_average: ℕ) -- The batsman's average after the 16th inning
  (h1: runs_in_17th_inning = 50)
  (h2: average_after_17th_inning = 18)
  (h3: 16 * initial_average + runs_in_17th_inning = 17 * average_after_17th_inning) :
  average_after_17th_inning - initial_average = 2 := 
sorry

end average_increase_l82_82578


namespace calculate_selling_price_l82_82665

theorem calculate_selling_price (cost_price : ℝ) (profit_percentage : ℝ) (selling_price : ℝ) 
  (h1 : cost_price = 83.33) 
  (h2 : profit_percentage = 20) : 
  selling_price = 100 := by
  sorry

end calculate_selling_price_l82_82665


namespace sequence_solution_l82_82980

theorem sequence_solution
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_a1 : a 1 = 10)
  (h_b1 : b 1 = 10)
  (h_recur_a : ∀ n : ℕ, a (n + 1) = 1 / (a n * b n))
  (h_recur_b : ∀ n : ℕ, b (n + 1) = (a n)^4 * b n) :
  (∀ n : ℕ, n > 0 → a n = 10^((2 - 3 * n) * (-1 : ℝ)^n) ∧ b n = 10^((6 * n - 7) * (-1 : ℝ)^n)) :=
by
  sorry

end sequence_solution_l82_82980


namespace a_range_l82_82037

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.log x - (1 / 2) * x^2 + 3 * x

def is_monotonic_on_interval (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a (a + 1), 4 / x - x + 3 > 0

theorem a_range (a : ℝ) :
  is_monotonic_on_interval a → (0 < a ∧ a ≤ 3) :=
by 
  sorry

end a_range_l82_82037


namespace local_minimum_at_1_1_l82_82169

noncomputable def function (x y : ℝ) : ℝ :=
  x^3 + y^3 - 3 * x * y

theorem local_minimum_at_1_1 : 
  ∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ (∀ (z : ℝ), z = function x y → z = -1) :=
sorry

end local_minimum_at_1_1_l82_82169


namespace alyona_final_balances_l82_82829

noncomputable def finalBalances (initialEuros initialDollars initialRubles : ℕ)
                                (interestRateEuroDollar interestRateRuble : ℚ)
                                (conversionRateEuroToRubles1 conversionRateRublesToDollars1 : ℚ)
                                (conversionRateDollarsToRubles2 conversionRateRublesToEuros2 : ℚ) :
                                ℕ × ℕ × ℕ :=
  sorry

theorem alyona_final_balances :
  finalBalances 3000 4000 240000
                (2.1 / 100) (7.9 / 100)
                60.10 58.90
                58.50 63.20 = (4040, 3286, 301504) :=
  sorry

end alyona_final_balances_l82_82829


namespace alice_flips_heads_probability_l82_82269

def prob_heads (n k : ℕ) (p q : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * (q ^ (n - k))

theorem alice_flips_heads_probability :
  prob_heads 8 3 (1/3 : ℚ) (2/3 : ℚ) = 1792 / 6561 :=
by
  sorry

end alice_flips_heads_probability_l82_82269


namespace no_such_natural_number_exists_l82_82011

theorem no_such_natural_number_exists :
  ¬ ∃ (n s : ℕ), n = 2014 * s + 2014 ∧ n % s = 2014 ∧ (n / s) = 2014 :=
by
  sorry

end no_such_natural_number_exists_l82_82011


namespace maddy_credits_to_graduate_l82_82483

theorem maddy_credits_to_graduate (semesters : ℕ) (credits_per_class : ℕ) (classes_per_semester : ℕ)
  (semesters_eq : semesters = 8)
  (credits_per_class_eq : credits_per_class = 3)
  (classes_per_semester_eq : classes_per_semester = 5) :
  semesters * (classes_per_semester * credits_per_class) = 120 :=
by
  -- Placeholder for proof
  sorry

end maddy_credits_to_graduate_l82_82483


namespace factor_expression_l82_82408

theorem factor_expression (x : ℝ) :
  80 * x ^ 5 - 250 * x ^ 9 = -10 * x ^ 5 * (25 * x ^ 4 - 8) :=
by
  sorry

end factor_expression_l82_82408


namespace probability_factor_120_less_9_l82_82134

theorem probability_factor_120_less_9 : 
  ∀ n : ℕ, n = 120 → (∃ p : ℚ, p = 7 / 16 ∧ (∃ factors_less_9 : ℕ, factors_less_9 < 16 ∧ factors_less_9 = 7)) := 
by 
  sorry

end probability_factor_120_less_9_l82_82134


namespace total_fishermen_count_l82_82744

theorem total_fishermen_count (F T F1 F2 : ℕ) (hT : T = 10000) (hF1 : F1 = 19 * 400) (hF2 : F2 = 2400) (hTotal : F1 + F2 = T) : F = 20 :=
by
  sorry

end total_fishermen_count_l82_82744


namespace medicine_duration_l82_82432

theorem medicine_duration (days_per_third_pill : ℕ) (pills : ℕ) (days_per_month : ℕ)
  (h1 : days_per_third_pill = 3)
  (h2 : pills = 90)
  (h3 : days_per_month = 30) :
  ((pills * (days_per_third_pill * 3)) / days_per_month) = 27 :=
sorry

end medicine_duration_l82_82432


namespace problem_l82_82150

theorem problem (x y z : ℕ) (hx : x < 9) (hy : y < 9) (hz : z < 9) 
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z % 9 = 0) :=
sorry

end problem_l82_82150


namespace proof_min_value_a3_and_a2b2_l82_82719

noncomputable def min_value_a3_and_a2b2 (a1 a2 a3 b1 b2 b3 : ℝ) : Prop :=
  (a1 > 0) ∧ (a2 > 0) ∧ (a3 > 0) ∧ (b1 > 0) ∧ (b2 > 0) ∧ (b3 > 0) ∧
  (a2 = a1 + b1) ∧ (a3 = a1 + 2 * b1) ∧ (b2 = b1 * a1) ∧ 
  (b3 = b1 * a1^2) ∧ (a3 = b3) ∧ 
  (a3 = 3 * Real.sqrt 6 / 2) ∧
  (a2 * b2 = 15 * Real.sqrt 6 / 8) 

theorem proof_min_value_a3_and_a2b2 : ∃ (a1 a2 a3 b1 b2 b3 : ℝ), min_value_a3_and_a2b2 a1 a2 a3 b1 b2 b3 :=
by
  use 2*Real.sqrt 6/3, 5*Real.sqrt 6/4, 3*Real.sqrt 6/2, Real.sqrt 6/4, 3/2, 3*Real.sqrt 6/2
  sorry

end proof_min_value_a3_and_a2b2_l82_82719


namespace triangle_lines_l82_82855

/-- Given a triangle with vertices A(1, 2), B(-1, 4), and C(4, 5):
  1. The equation of the line l₁ containing the altitude from A to side BC is 5x + y - 7 = 0.
  2. The equation of the line l₂ passing through C such that the distances from A and B to l₂ are equal
     is either x + y - 9 = 0 or x - 2y + 6 = 0. -/
theorem triangle_lines (A B C : ℝ × ℝ)
  (hA : A = (1, 2))
  (hB : B = (-1, 4))
  (hC : C = (4, 5)) :
  ∃ l₁ l₂ : ℝ × ℝ × ℝ,
  (l₁ = (5, 1, -7)) ∧
  ((l₂ = (1, 1, -9)) ∨ (l₂ = (1, -2, 6))) := by
  sorry

end triangle_lines_l82_82855


namespace minimum_positive_period_of_f_decreasing_intervals_of_f_maximum_value_of_f_l82_82240

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + 3 / 2

theorem minimum_positive_period_of_f : ∀ x : ℝ, f (x + Real.pi) = f x := by sorry

theorem decreasing_intervals_of_f : ∀ k : ℤ, ∀ x : ℝ,
  (Real.pi / 6 + k * Real.pi) ≤ x ∧ x ≤ (2 * Real.pi / 3 + k * Real.pi) → ∀ y : ℝ, 
  (Real.pi / 6 + k * Real.pi) ≤ y ∧ y ≤ (2 * Real.pi / 3 + k * Real.pi) → x ≤ y → f y ≤ f x := by sorry

theorem maximum_value_of_f : ∃ k : ℤ, ∃ x : ℝ, x = (Real.pi / 6 + k * Real.pi) ∧ f x = 5 / 2 := by sorry

end minimum_positive_period_of_f_decreasing_intervals_of_f_maximum_value_of_f_l82_82240


namespace max_gcd_is_2_l82_82620

-- Define the sequence
def a (n : ℕ) : ℕ := 101 + (n + 1)^2 + 3 * n

-- Define the gcd of consecutive terms
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_is_2 : ∀ n : ℕ, n > 0 → d n = 2 :=
by
  intros n hn
  dsimp [d]
  sorry

end max_gcd_is_2_l82_82620


namespace greatest_k_for_inquality_l82_82825

theorem greatest_k_for_inquality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 > b*c) :
    (a^2 - b*c)^2 > 4 * ((b^2 - c*a) * (c^2 - a*b)) :=
  sorry

end greatest_k_for_inquality_l82_82825


namespace floor_neg_seven_over_four_l82_82007

theorem floor_neg_seven_over_four : Int.floor (- 7 / 4 : ℝ) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l82_82007


namespace maximize_distance_l82_82692

noncomputable def maxTotalDistance (x : ℕ) (y : ℕ) (cityMPG highwayMPG : ℝ) (totalGallons : ℝ) : ℝ :=
  let cityDistance := cityMPG * ((x / 100.0) * totalGallons)
  let highwayDistance := highwayMPG * ((y / 100.0) * totalGallons)
  cityDistance + highwayDistance

theorem maximize_distance (x y : ℕ) (hx : x + y = 100) :
  maxTotalDistance x y 7.6 12.2 24.0 = 7.6 * (x / 100.0 * 24.0) + 12.2 * ((100.0 - x) / 100.0 * 24.0) :=
by
  sorry

end maximize_distance_l82_82692


namespace minimum_value_of_a_plus_b_l82_82486

noncomputable def f (x : ℝ) := Real.log x - (1 / x)
noncomputable def f' (x : ℝ) := 1 / x + 1 / (x^2)

theorem minimum_value_of_a_plus_b (a b m : ℝ) (h1 : a = 1 / m + 1 / (m^2)) 
  (h2 : b = Real.log m - 2 / m - 1) : a + b = -1 :=
by
  sorry

end minimum_value_of_a_plus_b_l82_82486


namespace ava_average_speed_l82_82191

noncomputable def initial_odometer : ℕ := 14941
noncomputable def final_odometer : ℕ := 15051
noncomputable def elapsed_time : ℝ := 4 -- hours

theorem ava_average_speed :
  (final_odometer - initial_odometer) / elapsed_time = 27.5 :=
by
  sorry

end ava_average_speed_l82_82191


namespace total_cost_of_phone_l82_82948

theorem total_cost_of_phone (cost_per_phone : ℕ) (monthly_cost : ℕ) (months : ℕ) (phone_count : ℕ) :
  cost_per_phone = 2 → monthly_cost = 7 → months = 4 → phone_count = 1 →
  (cost_per_phone * phone_count + monthly_cost * months) = 30 :=
by
  intros h1 h2 h3 h4
  sorry

end total_cost_of_phone_l82_82948


namespace steps_to_school_l82_82682

-- Define the conditions as assumptions
def distance : Float := 900
def step_length : Float := 0.45

-- Define the statement to be proven
theorem steps_to_school (x : Float) : step_length * x = distance → x = 2000 := by
  intro h
  sorry

end steps_to_school_l82_82682


namespace convex_polygons_count_l82_82687

def binomial (n k : ℕ) : ℕ := if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def count_convex_polygons_with_two_acute_angles (m n : ℕ) : ℕ :=
  if 4 < m ∧ m < n then
    (2 * n + 1) * (binomial (n + 1) (m - 1) + binomial n (m - 1))
  else 0

theorem convex_polygons_count (m n : ℕ) (h : 4 < m ∧ m < n) :
  count_convex_polygons_with_two_acute_angles m n = 
  (2 * n + 1) * (binomial (n + 1) (m - 1) + binomial n (m - 1)) :=
by sorry

end convex_polygons_count_l82_82687


namespace range_of_a_l82_82987

variable {x a : ℝ}

def A : Set ℝ := {x | x ≤ -3 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

theorem range_of_a (A_union_B_R : A ∪ B a = Set.univ) : a ∈ Set.Ici 3 :=
  sorry

end range_of_a_l82_82987


namespace sum_of_h_values_l82_82391

variable (f h : ℤ → ℤ)

-- Function definition for f and h
def f_def : ∀ x, 0 ≤ x → f x = f (x + 2) := sorry
def h_def : ∀ x, x < 0 → h x = f x := sorry

-- Symmetry condition for f being odd
def f_odd : ∀ x, f (-x) = -f x := sorry

-- Given value
def f_at_5 : f 5 = 1 := sorry

-- The proof statement we need:
theorem sum_of_h_values :
  h (-2022) + h (-2023) + h (-2024) = -1 :=
sorry

end sum_of_h_values_l82_82391


namespace at_least_one_nonnegative_l82_82763

theorem at_least_one_nonnegative (x : ℝ) (a b : ℝ) (h1 : a = x^2 - 1) (h2 : b = 4 * x + 5) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end at_least_one_nonnegative_l82_82763


namespace Tony_total_payment_l82_82452

-- Defining the cost of items
def lego_block_cost : ℝ := 250
def toy_sword_cost : ℝ := 120
def play_dough_cost : ℝ := 35

-- Quantities of each item
def total_lego_blocks : ℕ := 3
def total_toy_swords : ℕ := 5
def total_play_doughs : ℕ := 10

-- Quantities purchased on each day
def first_day_lego_blocks : ℕ := 2
def first_day_toy_swords : ℕ := 3
def second_day_lego_blocks : ℕ := total_lego_blocks - first_day_lego_blocks
def second_day_toy_swords : ℕ := total_toy_swords - first_day_toy_swords
def second_day_play_doughs : ℕ := total_play_doughs

-- Discounts and tax rates
def first_day_discount : ℝ := 0.20
def second_day_discount : ℝ := 0.10
def sales_tax : ℝ := 0.05

-- Calculating first day purchase amounts
def first_day_cost_before_discount : ℝ := (first_day_lego_blocks * lego_block_cost) + (first_day_toy_swords * toy_sword_cost)
def first_day_discount_amount : ℝ := first_day_cost_before_discount * first_day_discount
def first_day_cost_after_discount : ℝ := first_day_cost_before_discount - first_day_discount_amount
def first_day_sales_tax_amount : ℝ := first_day_cost_after_discount * sales_tax
def first_day_total_cost : ℝ := first_day_cost_after_discount + first_day_sales_tax_amount

-- Calculating second day purchase amounts
def second_day_cost_before_discount : ℝ := (second_day_lego_blocks * lego_block_cost) + (second_day_toy_swords * toy_sword_cost) + 
                                           (second_day_play_doughs * play_dough_cost)
def second_day_discount_amount : ℝ := second_day_cost_before_discount * second_day_discount
def second_day_cost_after_discount : ℝ := second_day_cost_before_discount - second_day_discount_amount
def second_day_sales_tax_amount : ℝ := second_day_cost_after_discount * sales_tax
def second_day_total_cost : ℝ := second_day_cost_after_discount + second_day_sales_tax_amount

-- Total cost
def total_cost : ℝ := first_day_total_cost + second_day_total_cost

-- Lean theorem statement
theorem Tony_total_payment : total_cost = 1516.20 := by
  sorry

end Tony_total_payment_l82_82452


namespace row_sum_1005_equals_20092_l82_82285

theorem row_sum_1005_equals_20092 :
  let row := 1005
  let n := row
  let first_element := n
  let num_elements := 2 * n - 1
  let last_element := first_element + (num_elements - 1)
  let sum_row := num_elements * (first_element + last_element) / 2
  sum_row = 20092 :=
by
  sorry

end row_sum_1005_equals_20092_l82_82285


namespace solve_eq_nonzero_solve_eq_zero_zero_solve_eq_zero_nonzero_l82_82704

-- Case 1: a ≠ 0
theorem solve_eq_nonzero (a b : ℝ) (h : a ≠ 0) : ∃ x : ℝ, x = -b / a ∧ a * x + b = 0 :=
by
  sorry

-- Case 2: a = 0 and b = 0
theorem solve_eq_zero_zero (a b : ℝ) (h1 : a = 0) (h2 : b = 0) : ∀ x : ℝ, a * x + b = 0 :=
by
  sorry

-- Case 3: a = 0 and b ≠ 0
theorem solve_eq_zero_nonzero (a b : ℝ) (h1 : a = 0) (h2 : b ≠ 0) : ¬ ∃ x : ℝ, a * x + b = 0 :=
by
  sorry

end solve_eq_nonzero_solve_eq_zero_zero_solve_eq_zero_nonzero_l82_82704


namespace min_max_value_in_interval_l82_82418

theorem min_max_value_in_interval : ∀ (x : ℝ),
  -2 < x ∧ x < 5 →
  ∃ (y : ℝ), (y = -1.5 ∨ y = 1.5) ∧ y = (x^2 - 4 * x + 6) / (2 * x - 4) := 
by sorry

end min_max_value_in_interval_l82_82418


namespace cos_60_eq_one_half_l82_82167

theorem cos_60_eq_one_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end cos_60_eq_one_half_l82_82167


namespace trees_distance_l82_82949

theorem trees_distance (num_trees : ℕ) (yard_length : ℕ) (trees_at_end : Prop) (tree_count : num_trees = 26) (yard_size : yard_length = 800) : 
  (yard_length / (num_trees - 1)) = 32 := 
by
  sorry

end trees_distance_l82_82949


namespace algebraic_expression_value_l82_82533

theorem algebraic_expression_value (x y : ℝ) (h : |x - 2| + (y + 3)^2 = 0) : (x + y)^2023 = -1 := by
  sorry

end algebraic_expression_value_l82_82533


namespace problem_a2_sub_b2_problem_a_mul_b_l82_82903

theorem problem_a2_sub_b2 {a b : ℝ} (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 :=
sorry

theorem problem_a_mul_b {a b : ℝ} (h1 : a + b = 8) (h2 : a - b = 4) : a * b = 12 :=
sorry

end problem_a2_sub_b2_problem_a_mul_b_l82_82903


namespace speed_is_90_l82_82403

namespace DrivingSpeedProof

/-- Given the observation times and marker numbers, prove the speed of the car is 90 km/hr. -/
theorem speed_is_90 
  (X Y : ℕ)
  (h0 : X ≥ 0) (h1 : X ≤ 9)
  (h2 : Y = 8 * X)
  (h3 : Y ≥ 0) (h4 : Y ≤ 9)
  (noon_marker : 10 * X + Y = 18)
  (second_marker : 10 * Y + X = 81)
  (third_marker : 100 * X + Y = 108)
  : 90 = 90 :=
by {
  sorry
}

end DrivingSpeedProof

end speed_is_90_l82_82403


namespace find_baseball_deck_price_l82_82982

variables (numberOfBasketballPacks : ℕ) (pricePerBasketballPack : ℝ) (numberOfBaseballDecks : ℕ)
           (totalMoney : ℝ) (changeReceived : ℝ) (totalSpent : ℝ) (spentOnBasketball : ℝ) (baseballDeckPrice : ℝ)

noncomputable def problem_conditions : Prop :=
  numberOfBasketballPacks = 2 ∧
  pricePerBasketballPack = 3 ∧
  numberOfBaseballDecks = 5 ∧
  totalMoney = 50 ∧
  changeReceived = 24 ∧
  totalSpent = totalMoney - changeReceived ∧
  spentOnBasketball = numberOfBasketballPacks * pricePerBasketballPack ∧
  totalSpent = spentOnBasketball + (numberOfBaseballDecks * baseballDeckPrice)

theorem find_baseball_deck_price (h : problem_conditions numberOfBasketballPacks pricePerBasketballPack numberOfBaseballDecks totalMoney changeReceived totalSpent spentOnBasketball baseballDeckPrice) :
  baseballDeckPrice = 4 :=
sorry

end find_baseball_deck_price_l82_82982


namespace smallest_of_x_y_z_l82_82741

variables {a b c d : ℕ}

/-- Given that x, y, and z are in the ratio a, b, c respectively, 
    and their sum x + y + z equals d, and 0 < a < b < c,
    prove that the smallest of x, y, and z is da / (a + b + c). -/
theorem smallest_of_x_y_z (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : 0 < d)
    (h_sum : ∀ k : ℚ, x = k * a → y = k * b → z = k * c → x + y + z = d) : 
    (∃ k : ℚ, x = k * a ∧ y = k * b ∧ z = k * c ∧ k = d / (a + b + c) ∧ x = da / (a + b + c)) :=
by 
  sorry

end smallest_of_x_y_z_l82_82741


namespace chord_length_l82_82940

noncomputable def circle_eq (θ : ℝ) : ℝ × ℝ :=
  (2 + 5 * Real.cos θ, 1 + 5 * Real.sin θ)

noncomputable def line_eq (t : ℝ) : ℝ × ℝ :=
  (-2 + 4 * t, -1 - 3 * t)

theorem chord_length :
  let center := (2, 1)
  let radius := 5
  let line_dist := |3 * center.1 + 4 * center.2 + 10| / Real.sqrt (3^2 + 4^2)
  let chord_len := 2 * Real.sqrt (radius^2 - line_dist^2)
  chord_len = 6 := 
by
  sorry

end chord_length_l82_82940


namespace hypotenuse_of_triangle_PQR_l82_82928

theorem hypotenuse_of_triangle_PQR (PA PB PC QR : ℝ) (h1: PA = 2) (h2: PB = 3) (h3: PC = 2)
  (h4: PA + PB + PC = QR) (h5: QR = PA + 3 + 2 * PA): QR = 5 * Real.sqrt 2 := 
sorry

end hypotenuse_of_triangle_PQR_l82_82928


namespace total_length_of_fence_l82_82466

theorem total_length_of_fence
  (x : ℝ)
  (h1 : (2 : ℝ) * x ^ 2 = 200) :
  (2 * x + 2 * x) = 40 :=
by
sorry

end total_length_of_fence_l82_82466


namespace hundred_div_point_two_five_eq_four_hundred_l82_82253

theorem hundred_div_point_two_five_eq_four_hundred : 100 / 0.25 = 400 := by
  sorry

end hundred_div_point_two_five_eq_four_hundred_l82_82253


namespace min_value_expression_l82_82266

theorem min_value_expression (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 3) (hy : 1 ≤ y ∧ y ≤ 4) : 
  ∃ z, z = (x + y) / x ∧ z = 4 / 3 := by
  sorry

end min_value_expression_l82_82266


namespace round_to_nearest_tenth_l82_82051

theorem round_to_nearest_tenth (x : Float) (h : x = 42.63518) : Float.round (x * 10) / 10 = 42.6 := by
  sorry

end round_to_nearest_tenth_l82_82051


namespace surface_area_bound_l82_82793

theorem surface_area_bound
  (a b c d : ℝ)
  (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) (h4: 0 ≤ d) 
  (h_quad: a + b + c > d) : 
  2 * (a * b + b * c + c * a) ≤ (a + b + c) ^ 2 - (d ^ 2) / 3 :=
sorry

end surface_area_bound_l82_82793


namespace find_Q_x_l82_82675

noncomputable def Q : ℝ → ℝ := sorry

variables (Q0 Q1 Q2 : ℝ)

axiom Q_def : ∀ x, Q x = Q0 + Q1 * x + Q2 * x^2
axiom Q_minus_2 : Q (-2) = -3

theorem find_Q_x : ∀ x, Q x = (3 / 5) * (1 + x - x^2) :=
by 
  -- Proof to be completed
  sorry

end find_Q_x_l82_82675


namespace matrix_inverse_eq_l82_82849

theorem matrix_inverse_eq (d k : ℚ) (A : Matrix (Fin 2) (Fin 2) ℚ) 
  (hA : A = ![![1, 4], ![6, d]]) 
  (hA_inv : A⁻¹ = k • A) :
  (d, k) = (-1, 1/25) :=
  sorry

end matrix_inverse_eq_l82_82849


namespace differentiate_and_evaluate_l82_82707

theorem differentiate_and_evaluate (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ) :
  (2*x - 1)^6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_1 + 2*a_2 + 3*a_3 + 4*a_4 + 5*a_5 + 6*a_6 = 12 :=
sorry

end differentiate_and_evaluate_l82_82707


namespace complex_number_solution_l82_82136

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i^2 = -1) (hz : i * (z - 1) = 1 - i) : z = -i :=
by sorry

end complex_number_solution_l82_82136


namespace Jessica_biking_speed_l82_82552

theorem Jessica_biking_speed
  (swim_distance swim_speed : ℝ)
  (run_distance run_speed : ℝ)
  (bike_distance total_time : ℝ)
  (h1 : swim_distance = 0.5)
  (h2 : swim_speed = 1)
  (h3 : run_distance = 5)
  (h4 : run_speed = 5)
  (h5 : bike_distance = 20)
  (h6 : total_time = 4) :
  bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed)) = 8 :=
by
  -- Proof omitted
  sorry

end Jessica_biking_speed_l82_82552


namespace min_sum_of_product_2004_l82_82328

theorem min_sum_of_product_2004 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
    (hxyz : x * y * z = 2004) : x + y + z ≥ 174 ∧ ∃ (a b c : ℕ), a * b * c = 2004 ∧ a + b + c = 174 :=
by sorry

end min_sum_of_product_2004_l82_82328


namespace reflected_ray_equation_l82_82985

theorem reflected_ray_equation (x y : ℝ) (incident_ray : y = 2 * x + 1) (reflecting_line : y = x) :
  x - 2 * y - 1 = 0 :=
sorry

end reflected_ray_equation_l82_82985


namespace max_red_balls_l82_82463

theorem max_red_balls (r w : ℕ) (h1 : r = 3 * w) (h2 : r + w ≤ 50) : r = 36 :=
sorry

end max_red_balls_l82_82463


namespace inequality_I_inequality_II_inequality_III_l82_82968

variable {a b c x y z : ℝ}

-- Assume the conditions
def conditions (a b c x y z : ℝ) : Prop :=
  x^2 < a ∧ y^2 < b ∧ z^2 < c

-- Prove the first inequality
theorem inequality_I (h : conditions a b c x y z) : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 < a * b + b * c + c * a :=
sorry

-- Prove the second inequality
theorem inequality_II (h : conditions a b c x y z) : x^4 + y^4 + z^4 < a^2 + b^2 + c^2 :=
sorry

-- Prove the third inequality
theorem inequality_III (h : conditions a b c x y z) : x^2 * y^2 * z^2 < a * b * c :=
sorry

end inequality_I_inequality_II_inequality_III_l82_82968


namespace xy_plus_y_square_l82_82529

theorem xy_plus_y_square {x y : ℝ} (h1 : x * y = 16) (h2 : x + y = 8) : x^2 + y^2 = 32 :=
sorry

end xy_plus_y_square_l82_82529


namespace squares_in_50th_ring_l82_82210

-- Define the problem using the given conditions
def centered_square_3x3 : ℕ := 3 -- Represent the 3x3 centered square

-- Define the function that computes the number of unit squares in the nth ring
def unit_squares_in_nth_ring (n : ℕ) : ℕ :=
  if n = 1 then 16
  else 24 + 8 * (n - 2)

-- Define the accumulation of unit squares up to the 50th ring
def total_squares_in_50th_ring : ℕ :=
  33 + 24 * 49

theorem squares_in_50th_ring : unit_squares_in_nth_ring 50 = 1209 :=
by
  -- Ensure that the correct value for the 50th ring can be verified
  sorry

end squares_in_50th_ring_l82_82210


namespace fly_total_distance_l82_82013

noncomputable def total_distance_traveled (r : ℝ) (d3 : ℝ) : ℝ :=
  let d1 := 2 * r
  let d2 := Real.sqrt (d1^2 - d3^2)
  d1 + d2 + d3

theorem fly_total_distance (r : ℝ) (h_r : r = 60) (d3 : ℝ) (h_d3 : d3 = 90) :
  total_distance_traveled r d3 = 289.37 :=
by
  rw [h_r, h_d3]
  simp [total_distance_traveled]
  sorry

end fly_total_distance_l82_82013


namespace square_perimeter_l82_82570

theorem square_perimeter (a : ℝ) (side : ℝ) (perimeter : ℝ) (h1 : a = 144) (h2 : side = Real.sqrt a) (h3 : perimeter = 4 * side) : perimeter = 48 := by
  sorry

end square_perimeter_l82_82570


namespace arithmetic_progression_common_difference_zero_l82_82076

theorem arithmetic_progression_common_difference_zero {a d : ℤ} (h₁ : a = 12) 
  (h₂ : ∀ n : ℕ, a + n * d = (a + (n + 1) * d + a + (n + 2) * d) / 2) : d = 0 :=
  sorry

end arithmetic_progression_common_difference_zero_l82_82076


namespace quadricycles_count_l82_82584

theorem quadricycles_count (s q : ℕ) (hsq : s + q = 9) (hw : 2 * s + 4 * q = 30) : q = 6 :=
by
  sorry

end quadricycles_count_l82_82584


namespace motorist_gallons_affordable_l82_82158

-- Definitions based on the conditions in the problem
def expected_gallons : ℕ := 12
def actual_price_per_gallon : ℕ := 150
def price_difference : ℕ := 30
def expected_price_per_gallon : ℕ := actual_price_per_gallon - price_difference
def total_initial_cents : ℕ := expected_gallons * expected_price_per_gallon

-- Theorem stating that given the conditions, the motorist can afford 9 gallons of gas
theorem motorist_gallons_affordable : 
  total_initial_cents / actual_price_per_gallon = 9 := 
by
  sorry

end motorist_gallons_affordable_l82_82158


namespace smallest_value_c_plus_d_l82_82490

noncomputable def problem1 (c d : ℝ) : Prop :=
c > 0 ∧ d > 0 ∧ (c^2 ≥ 12 * d) ∧ ((3 * d)^2 ≥ 4 * c)

theorem smallest_value_c_plus_d : ∃ c d : ℝ, problem1 c d ∧ c + d = 4 / Real.sqrt 3 + 4 / 9 :=
sorry

end smallest_value_c_plus_d_l82_82490


namespace tom_to_luke_ratio_l82_82125

theorem tom_to_luke_ratio (Tom Luke Anthony : ℕ) 
  (hAnthony : Anthony = 44) 
  (hTom : Tom = 33) 
  (hLuke : Luke = Anthony / 4) : 
  Tom / Nat.gcd Tom Luke = 3 ∧ Luke / Nat.gcd Tom Luke = 1 := 
by
  sorry

end tom_to_luke_ratio_l82_82125


namespace diameter_of_circle_A_l82_82775

theorem diameter_of_circle_A
  (diameter_B : ℝ)
  (r : ℝ)
  (h1 : diameter_B = 16)
  (h2 : r^2 = (r / 8)^2 * 4):
  2 * (r / 2) = 8 :=
by
  sorry

end diameter_of_circle_A_l82_82775


namespace inequality_C_l82_82025

theorem inequality_C (a b : ℝ) : 
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := 
by
  sorry

end inequality_C_l82_82025


namespace calculate_expression_l82_82131

theorem calculate_expression (m : ℝ) : (-m)^2 * m^5 = m^7 := 
sorry

end calculate_expression_l82_82131


namespace positive_diff_two_largest_prime_factors_l82_82376

theorem positive_diff_two_largest_prime_factors (a b c d : ℕ) (h : 178469 = a * b * c * d) 
  (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d) 
  (hle1 : a ≤ b) (hle2 : b ≤ c) (hle3 : c ≤ d):
  d - c = 2 := by sorry

end positive_diff_two_largest_prime_factors_l82_82376


namespace M_intersection_N_eq_N_l82_82109

def M := { x : ℝ | x < 4 }
def N := { x : ℝ | x ≤ -2 }

theorem M_intersection_N_eq_N : M ∩ N = N :=
by
  sorry

end M_intersection_N_eq_N_l82_82109


namespace cdf_of_Z_pdf_of_Z_l82_82224

noncomputable def f1 (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 2 then 0.5 else 0

noncomputable def f2 (y : ℝ) : ℝ :=
  if 0 < y ∧ y < 2 then 0.5 else 0

noncomputable def G (z : ℝ) : ℝ :=
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z^2 / 8
  else if 2 < z ∧ z ≤ 4 then 1 - (4 - z)^2 / 8
  else 1

noncomputable def g (z : ℝ) : ℝ :=
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z / 4
  else if 2 < z ∧ z ≤ 4 then 1 - z / 4
  else 0

theorem cdf_of_Z (z : ℝ) : G z = 
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z^2 / 8
  else if 2 < z ∧ z ≤ 4 then 1 - (4 - z)^2 / 8
  else 1 := sorry

theorem pdf_of_Z (z : ℝ) : g z = 
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z / 4
  else if 2 < z ∧ z ≤ 4 then 1 - z / 4
  else 0 := sorry

end cdf_of_Z_pdf_of_Z_l82_82224


namespace smallest_n_for_inequality_l82_82499

theorem smallest_n_for_inequality :
  ∃ (n : ℕ), n = 4003 ∧ (∀ m : ℤ, (0 < m ∧ m < 2001) →
  ∃ k : ℤ, (m / 2001 : ℚ) < (k / n : ℚ) ∧ (k / n : ℚ) < ((m + 1) / 2002 : ℚ)) :=
sorry

end smallest_n_for_inequality_l82_82499


namespace expression_simplification_l82_82268

theorem expression_simplification : (4^2 * 7 / (8 * 9^2) * (8 * 9 * 11^2) / (4 * 7 * 11)) = 44 / 9 :=
by
  sorry

end expression_simplification_l82_82268


namespace find_b10_l82_82049

def sequence_b (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = b (n + 1) + b n

theorem find_b10 (b : ℕ → ℕ) (h0 : ∀ n, b n > 0) (h1 : b 9 = 544) (h2 : sequence_b b) : b 10 = 883 :=
by
  -- We could provide steps of the proof here, but we use 'sorry' to omit the proof content
  sorry

end find_b10_l82_82049


namespace brown_eyed_brunettes_l82_82888

theorem brown_eyed_brunettes (total_girls blondes brunettes blue_eyed_blondes brown_eyed_girls : ℕ) 
    (h1 : total_girls = 60) 
    (h2 : blondes + brunettes = total_girls) 
    (h3 : blue_eyed_blondes = 20) 
    (h4 : brunettes = 35) 
    (h5 : brown_eyed_girls = 22) 
    (h6 : blondes = total_girls - brunettes) 
    (h7 : brown_eyed_blondes = blondes - blue_eyed_blondes) :
  brunettes - (brown_eyed_girls - brown_eyed_blondes) = 17 :=
by sorry  -- Proof is not required

end brown_eyed_brunettes_l82_82888


namespace cost_of_each_toy_car_l82_82658

theorem cost_of_each_toy_car (S M C A B : ℕ) (hS : S = 53) (hM : M = 7) (hA : A = 10) (hB : B = 14) 
(hTotalSpent : S - M = C + A + B) (hTotalCars : 2 * C / 2 = 11) : 
C / 2 = 11 :=
by
  rw [hS, hM, hA, hB] at hTotalSpent
  sorry

end cost_of_each_toy_car_l82_82658


namespace tan_alpha_eq_l82_82957

theorem tan_alpha_eq : ∀ (α : ℝ),
  (Real.tan (α - (5 * Real.pi / 4)) = 1 / 5) →
  Real.tan α = 3 / 2 :=
by
  intro α h
  sorry

end tan_alpha_eq_l82_82957


namespace kohen_apples_l82_82598

theorem kohen_apples (B : ℕ) (h1 : 300 * B = 4 * 750) : B = 10 :=
by
  -- proof goes here
  sorry

end kohen_apples_l82_82598


namespace bacteria_after_7_hours_l82_82969

noncomputable def bacteria_growth (initial : ℝ) (t : ℝ) (k : ℝ) : ℝ := initial * (10 * (Real.exp (k * t)))

noncomputable def solve_bacteria_problem : ℝ :=
let doubling_time := 1 / 60 -- In hours, since 60 minutes is 1 hour
-- Given that it doubles in 1 hour, we expect the growth to be such that y = initial * (2) in 1 hour.
let k := Real.log 2 -- Since when t = 1, we have 10 * e^(k * 1) = 2 * 10
bacteria_growth 10 7 k

theorem bacteria_after_7_hours :
  solve_bacteria_problem = 1280 :=
by
  sorry

end bacteria_after_7_hours_l82_82969


namespace people_per_seat_l82_82475

def ferris_wheel_seats : ℕ := 4
def total_people_riding : ℕ := 20

theorem people_per_seat : total_people_riding / ferris_wheel_seats = 5 := by
  sorry

end people_per_seat_l82_82475


namespace setB_is_correct_l82_82956

def setA : Set ℤ := {1, 0, -1, 2}
def setB : Set ℤ := { y | ∃ x ∈ setA, y = Int.natAbs x }

theorem setB_is_correct : setB = {0, 1, 2} := by
  sorry

end setB_is_correct_l82_82956


namespace part_I_solution_set_part_II_range_of_a_l82_82681

-- Given function definition
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a*x + 6

-- (I) Prove the solution set of f(x) < 0 when a = 5
theorem part_I_solution_set : 
  (∀ x : ℝ, f x 5 < 0 ↔ (-3 < x ∧ x < -2)) := by
  sorry

-- (II) Prove the range of a such that f(x) > 0 for all x ∈ ℝ 
theorem part_II_range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, f x a > 0) ↔ (-2*Real.sqrt 6 < a ∧ a < 2*Real.sqrt 6)) := by
  sorry

end part_I_solution_set_part_II_range_of_a_l82_82681


namespace division_value_l82_82883

theorem division_value (x : ℝ) (h1 : 2976 / x - 240 = 8) : x = 12 := 
by
  sorry

end division_value_l82_82883


namespace candy_probability_l82_82300

/-- 
A jar has 15 red candies, 15 blue candies, and 10 green candies. Terry picks three candies at random,
then Mary picks three of the remaining candies at random. Calculate the probability that they get 
the same color combination, irrespective of order, expressed as a fraction $m/n,$ where $m$ and $n$ 
are relatively prime positive integers. Find $m+n.$ -/
theorem candy_probability :
  let num_red := 15
  let num_blue := 15
  let num_green := 10
  let total_candies := num_red + num_blue + num_green
  let Terry_picks := 3
  let Mary_picks := 3
  let prob_equal_comb := (118545 : ℚ) / 2192991
  let m := 118545
  let n := 2192991
  m + n = 2310536 := sorry

end candy_probability_l82_82300


namespace karen_kept_cookies_l82_82170

def total_cookies : ℕ := 50
def cookies_to_grandparents : ℕ := 8
def number_of_classmates : ℕ := 16
def cookies_per_classmate : ℕ := 2

theorem karen_kept_cookies (x : ℕ) 
  (H1 : x = total_cookies - (cookies_to_grandparents + number_of_classmates * cookies_per_classmate)) :
  x = 10 :=
by
  -- proof omitted
  sorry

end karen_kept_cookies_l82_82170


namespace cos_squared_identity_l82_82087

theorem cos_squared_identity (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * Real.cos (π / 6 + α / 2) ^ 2 + 1 = 7 / 3 := 
by
    sorry

end cos_squared_identity_l82_82087


namespace range_of_a_l82_82979

theorem range_of_a (a x : ℝ) (h_p : a - 4 < x ∧ x < a + 4) (h_q : (x - 2) * (x - 3) > 0) :
  a ≤ -2 ∨ a ≥ 7 :=
sorry

end range_of_a_l82_82979


namespace b_geometric_l82_82905

def a (n : ℕ) : ℝ := sorry
def b (n : ℕ) : ℝ := sorry

axiom a1 : a 1 = 1
axiom a_n_recurrence (n : ℕ) : a n + a (n + 1) = 1 / (3^n)
axiom b_def (n : ℕ) : b n = 3^(n - 1) * a n - 1/4

theorem b_geometric (n : ℕ) : b (n + 1) = -3 * b n := sorry

end b_geometric_l82_82905


namespace problem_BD_l82_82725

variable (a b c : ℝ)

theorem problem_BD (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) :
  (c - a < c - b) ∧ (a⁻¹ * c > b⁻¹ * c) :=
by
  sorry

end problem_BD_l82_82725


namespace select_best_player_l82_82574

theorem select_best_player : 
  (average_A = 9.6 ∧ variance_A = 0.25) ∧ 
  (average_B = 9.5 ∧ variance_B = 0.27) ∧ 
  (average_C = 9.5 ∧ variance_C = 0.30) ∧ 
  (average_D = 9.6 ∧ variance_D = 0.23) → 
  best_player = D := 
by 
  sorry

end select_best_player_l82_82574


namespace verify_statements_l82_82401

theorem verify_statements (a b : ℝ) :
  ( (ab < 0 ∧ (a < 0 ∧ b > 0 ∨ a > 0 ∧ b < 0)) → (a / b = -1)) ∧
  ( (a + b < 0 ∧ ab > 0) → (|2 * a + 3 * b| = -(2 * a + 3 * b)) ) ∧
  ( (|a - b| + a - b = 0) → (b > a) = False ) ∧
  ( (|a| > |b|) → ((a + b) * (a - b) < 0) = False ) :=
by
  sorry

end verify_statements_l82_82401


namespace correct_option_d_l82_82145

theorem correct_option_d (a b c : ℝ) (h: a < b ∧ b < 0) : a^2 > ab ∧ ab > b^2 :=
by
  sorry

end correct_option_d_l82_82145


namespace fred_balloon_count_l82_82546

def sally_balloons : ℕ := 6

def fred_balloons (sally_balloons : ℕ) := 3 * sally_balloons

theorem fred_balloon_count : fred_balloons sally_balloons = 18 := by
  sorry

end fred_balloon_count_l82_82546


namespace ant_travel_finite_path_exists_l82_82326

theorem ant_travel_finite_path_exists :
  ∃ (x y z t : ℝ), |x| < |y - z + t| ∧ |y| < |x - z + t| ∧ 
                   |z| < |x - y + t| ∧ |t| < |x - y + z| :=
by
  sorry

end ant_travel_finite_path_exists_l82_82326


namespace airplane_average_speed_l82_82993

-- Define the conditions
def miles_to_kilometers (miles : ℕ) : ℝ :=
  miles * 1.60934

def distance_miles : ℕ := 1584
def time_hours : ℕ := 24

-- Define the problem to prove
theorem airplane_average_speed : 
  (miles_to_kilometers distance_miles) / (time_hours : ℝ) = 106.24 :=
by
  sorry

end airplane_average_speed_l82_82993


namespace solve_frac_difference_of_squares_l82_82067

theorem solve_frac_difference_of_squares :
  (108^2 - 99^2) / 9 = 207 := by
  sorry

end solve_frac_difference_of_squares_l82_82067


namespace intersection_P_Q_l82_82313

-- Define set P
def P : Set ℝ := {1, 2, 3, 4}

-- Define set Q
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the problem statement as a theorem
theorem intersection_P_Q : P ∩ Q = {1, 2} :=
by
  sorry

end intersection_P_Q_l82_82313


namespace range_of_m_l82_82344

-- Definitions from conditions
def p (m : ℝ) : Prop := (∃ x y : ℝ, 2 * x^2 / m + y^2 / (m - 1) = 1)
def q (m : ℝ) : Prop := ∃ x1 : ℝ, 8 * x1^2 - 8 * m * x1 + 7 * m - 6 = 0
def proposition (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬ (p m ∧ q m)

-- Proof statement
theorem range_of_m (m : ℝ) (h : proposition m) : (m ≤ 1 ∨ (3 / 2 < m ∧ m < 2)) :=
by
  sorry

end range_of_m_l82_82344


namespace simplify_expression_l82_82821

theorem simplify_expression :
  1 + (1 / (1 + Real.sqrt 2)) - (1 / (1 - Real.sqrt 5)) =
  1 + ((-Real.sqrt 2 - Real.sqrt 5) / (1 + Real.sqrt 2 - Real.sqrt 5 - Real.sqrt 10)) :=
by
  sorry

end simplify_expression_l82_82821


namespace values_of_a_and_b_l82_82074

theorem values_of_a_and_b (a b : ℝ) 
  (hT : (2, 1) ∈ {p : ℝ × ℝ | ∃ (a : ℝ), p.1 * a + p.2 - 3 = 0})
  (hS : (2, 1) ∈ {p : ℝ × ℝ | ∃ (b : ℝ), p.1 - p.2 - b = 0}) :
  a = 1 ∧ b = 1 :=
by
  sorry

end values_of_a_and_b_l82_82074


namespace total_amount_after_refunds_and_discounts_l82_82507

-- Definitions
def individual_bookings : ℤ := 12000
def group_bookings_before_discount : ℤ := 16000
def discount_rate : ℕ := 10
def refund_individual_1 : ℤ := 500
def count_refund_individual_1 : ℕ := 3
def refund_individual_2 : ℤ := 300
def count_refund_individual_2 : ℕ := 2
def total_refund_group : ℤ := 800

-- Calculation proofs
theorem total_amount_after_refunds_and_discounts : 
(individual_bookings + (group_bookings_before_discount - (discount_rate * group_bookings_before_discount / 100))) - 
((count_refund_individual_1 * refund_individual_1) + (count_refund_individual_2 * refund_individual_2) + total_refund_group) = 23500 := by
    sorry

end total_amount_after_refunds_and_discounts_l82_82507


namespace probability_same_flavor_l82_82616

theorem probability_same_flavor (num_flavors : ℕ) (num_bags : ℕ) (h1 : num_flavors = 4) (h2 : num_bags = 2) :
  let total_outcomes := num_flavors ^ num_bags
  let favorable_outcomes := num_flavors
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 4 :=
by
  sorry

end probability_same_flavor_l82_82616


namespace solution_set_of_inequality_l82_82343

theorem solution_set_of_inequality :
  {x : ℝ | -6 * x ^ 2 - x + 2 < 0} = {x : ℝ | x < -(2 / 3)} ∪ {x | x > 1 / 2} := 
sorry

end solution_set_of_inequality_l82_82343


namespace max_xyz_squared_l82_82610

theorem max_xyz_squared 
  (x y z : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (h1 : x * y * z = (14 - x) * (14 - y) * (14 - z)) 
  (h2 : x + y + z < 28) : 
  x^2 + y^2 + z^2 ≤ 219 :=
sorry

end max_xyz_squared_l82_82610


namespace sin_C_in_right_triangle_l82_82654

-- Triangle ABC with angle B = 90 degrees and tan A = 3/4
theorem sin_C_in_right_triangle (A C : ℝ) (h1 : A + C = π / 2) (h2 : Real.tan A = 3 / 4) : Real.sin C = 4 / 5 := by
  sorry

end sin_C_in_right_triangle_l82_82654


namespace square_diff_l82_82559

theorem square_diff (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by 
  sorry

end square_diff_l82_82559


namespace Robe_savings_l82_82806

-- Define the conditions and question in Lean 4
theorem Robe_savings 
  (repair_fee : ℕ)
  (corner_light_cost : ℕ)
  (brake_disk_cost : ℕ)
  (total_remaining_savings : ℕ)
  (total_savings_before : ℕ)
  (h1 : repair_fee = 10)
  (h2 : corner_light_cost = 2 * repair_fee)
  (h3 : brake_disk_cost = 3 * corner_light_cost)
  (h4 : total_remaining_savings = 480)
  (h5 : total_savings_before = total_remaining_savings + (repair_fee + corner_light_cost + 2 * brake_disk_cost)) :
  total_savings_before = 630 :=
by
  -- Proof steps to be filled
  sorry

end Robe_savings_l82_82806


namespace sum_of_cubes_decomposition_l82_82194

theorem sum_of_cubes_decomposition :
  ∃ a b c d e : ℤ, (∀ x : ℤ, 1728 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ (a + b + c + d + e = 132) :=
by
  sorry

end sum_of_cubes_decomposition_l82_82194


namespace salary_january_l82_82164

variable (J F M A May : ℝ)

theorem salary_january 
  (h1 : J + F + M + A = 32000) 
  (h2 : F + M + A + May = 33600) 
  (h3 : May = 6500) : 
  J = 4900 := 
by {
 sorry 
}

end salary_january_l82_82164


namespace seated_men_l82_82114

def passengers : Nat := 48
def fraction_of_women : Rat := 2/3
def fraction_of_men_standing : Rat := 1/8

theorem seated_men (men women standing seated : Nat) 
  (h1 : women = passengers * fraction_of_women)
  (h2 : men = passengers - women)
  (h3 : standing = men * fraction_of_men_standing)
  (h4 : seated = men - standing) :
  seated = 14 := by
  sorry

end seated_men_l82_82114


namespace part_I_part_II_part_III_l82_82243

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.log x

theorem part_I (a : ℝ) : (∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f a x ≥ f a 1) ↔ a ≥ -1/2 :=
by
  sorry

theorem part_II : ∀ x : ℝ, f (-Real.exp 1) x + 2 ≤ 0 :=
by
  sorry

theorem part_III : ¬ ∃ x : ℝ, |f (-Real.exp 1) x| = Real.log x / x + 3 / 2 :=
by
  sorry

end part_I_part_II_part_III_l82_82243


namespace true_statements_about_f_l82_82496

noncomputable def f (x : ℝ) := 2 * abs (Real.cos x) * Real.sin x + Real.sin (2 * x)

theorem true_statements_about_f :
  (∀ x y : ℝ, -π/4 ≤ x ∧ x < y ∧ y ≤ π/4 → f x < f y) ∧
  (∀ y : ℝ, -2 ≤ y ∧ y ≤ 2 → (∃ x : ℝ, f x = y)) :=
by
  sorry

end true_statements_about_f_l82_82496


namespace sarahs_trip_length_l82_82963

noncomputable def sarahsTrip (x : ℝ) : Prop :=
  x / 4 + 15 + x / 3 = x

theorem sarahs_trip_length : ∃ x : ℝ, sarahsTrip x ∧ x = 36 := by
  -- There should be a proof here, but it's omitted as per the task instructions
  sorry

end sarahs_trip_length_l82_82963


namespace four_sq_geq_prod_sum_l82_82863

variable {α : Type*} [LinearOrderedField α]

theorem four_sq_geq_prod_sum (a b c d : α) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
sorry

end four_sq_geq_prod_sum_l82_82863


namespace necessary_but_not_sufficient_condition_l82_82042

def condition_neq_1_or_neq_2 (a b : ℤ) : Prop :=
  a ≠ 1 ∨ b ≠ 2

def statement_sum_neq_3 (a b : ℤ) : Prop :=
  a + b ≠ 3

theorem necessary_but_not_sufficient_condition :
  ∀ (a b : ℤ), condition_neq_1_or_neq_2 a b → ¬ (statement_sum_neq_3 a b) → false :=
by
  sorry

end necessary_but_not_sufficient_condition_l82_82042


namespace evan_45_l82_82111

theorem evan_45 (k n : ℤ) (h1 : n + (k * (2 * k - 1)) = 60) : 60 - n = 45 :=
by sorry

end evan_45_l82_82111


namespace canal_depth_l82_82907

theorem canal_depth (A : ℝ) (w_top w_bottom : ℝ) (h : ℝ) 
    (hA : A = 10290) 
    (htop : w_top = 6) 
    (hbottom : w_bottom = 4) 
    (harea : A = 1 / 2 * (w_top + w_bottom) * h) : 
    h = 2058 :=
by
  -- here goes the proof steps
  sorry

end canal_depth_l82_82907


namespace find_chemistry_marks_l82_82278

theorem find_chemistry_marks (marks_english marks_math marks_physics marks_biology : ℤ)
    (average_marks total_subjects : ℤ)
    (h1 : marks_english = 36)
    (h2 : marks_math = 35)
    (h3 : marks_physics = 42)
    (h4 : marks_biology = 55)
    (h5 : average_marks = 45)
    (h6 : total_subjects = 5) :
    (225 - (marks_english + marks_math + marks_physics + marks_biology)) = 57 :=
by
  sorry

end find_chemistry_marks_l82_82278


namespace entree_cost_l82_82998

/-- 
Prove that if the total cost is 23 and the entree costs 5 more than the dessert, 
then the cost of the entree is 14.
-/
theorem entree_cost (D : ℝ) (H1 : D + (D + 5) = 23) : D + 5 = 14 :=
by
  -- note: no proof required as per instructions
  sorry

end entree_cost_l82_82998


namespace inequality_solution_set_l82_82211

theorem inequality_solution_set :
  (∀ x : ℝ, (3 * x - 2 < 2 * (x + 1) ∧ (x - 1) / 2 > 1) ↔ (3 < x ∧ x < 4)) :=
by
  sorry

end inequality_solution_set_l82_82211


namespace max_value_g_l82_82138

noncomputable def g (x : ℝ) := 4 * x - x ^ 4

theorem max_value_g : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 4 ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ Real.sqrt 4 → g y ≤ 3 :=
sorry

end max_value_g_l82_82138


namespace y_intercept_of_line_l82_82190

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) : y = -2 :=
by
  sorry

end y_intercept_of_line_l82_82190


namespace algebraic_expression_l82_82602

-- Definition for the problem expressed in Lean
def number_one_less_than_three_times (a : ℝ) : ℝ :=
  3 * a - 1

-- Theorem stating the proof problem
theorem algebraic_expression (a : ℝ) : number_one_less_than_three_times a = 3 * a - 1 :=
by
  -- Proof steps would go here; omitted as per instructions
  sorry

end algebraic_expression_l82_82602


namespace negation_statement_l82_82904

theorem negation_statement (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) : ¬ (x = 0 ∧ y = 0) :=
sorry

end negation_statement_l82_82904


namespace smallest_integer_odd_sequence_l82_82303

/-- Given the median of a set of consecutive odd integers is 157 and the greatest integer in the set is 171,
    prove that the smallest integer in the set is 149. -/
theorem smallest_integer_odd_sequence (median greatest : ℤ) (h_median : median = 157) (h_greatest : greatest = 171) :
  ∃ smallest : ℤ, smallest = 149 :=
by
  sorry

end smallest_integer_odd_sequence_l82_82303


namespace set_intersection_complement_l82_82277

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set ℕ := {1, 2, 3}

-- Define set B
def B : Set ℕ := {3, 4, 5}

-- State the theorem
theorem set_intersection_complement :
  (U \ A) ∩ B = {4, 5} := by
  sorry

end set_intersection_complement_l82_82277


namespace g_of_1986_l82_82373

-- Define the function g and its properties
noncomputable def g : ℕ → ℤ :=
sorry  -- Placeholder for the actual definition according to the conditions

axiom g_is_defined (x : ℕ) : x ≥ 0 → ∃ y : ℤ, g x = y
axiom g_at_1 : g 1 = 1
axiom g_add (a b : ℕ) (h_a : a ≥ 0) (h_b : b ≥ 0) : g (a + b) = g a + g b - 3 * g (a * b) + 1

-- Lean statement for the proof problem
theorem g_of_1986 : g 1986 = 0 :=
sorry

end g_of_1986_l82_82373


namespace parallel_lines_slope_l82_82033

theorem parallel_lines_slope {a : ℝ} 
    (h1 : ∀ x y : ℝ, 4 * y + 3 * x - 5 = 0 → y = -3 / 4 * x + 5 / 4)
    (h2 : ∀ x y : ℝ, 6 * y + a * x + 4 = 0 → y = -a / 6 * x - 2 / 3)
    (h_parallel : ∀ x₁ y₁ x₂ y₂ : ℝ, (4 * y₁ + 3 * x₁ - 5 = 0 ∧ 6 * y₂ + a * x₂ + 4 = 0) → -3 / 4 = -a / 6) : 
  a = 4.5 := sorry

end parallel_lines_slope_l82_82033


namespace length_of_rest_of_body_l82_82803

theorem length_of_rest_of_body (h : ℝ) (legs : ℝ) (head : ℝ) (rest_of_body : ℝ) :
  h = 60 → legs = (1 / 3) * h → head = (1 / 4) * h → rest_of_body = h - (legs + head) → rest_of_body = 25 := by
  sorry

end length_of_rest_of_body_l82_82803


namespace sum_of_rationals_eq_l82_82686

theorem sum_of_rationals_eq (a1 a2 a3 a4 : ℚ)
  (h : {x : ℚ | ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 4 ∧ x = a1 * a2 ∧ x = a1 * a3 ∧ x = a1 * a4 ∧ x = a2 * a3 ∧ x = a2 * a4 ∧ x = a3 * a4} = {-24, -2, -3/2, -1/8, 1, 3}) :
  a1 + a2 + a3 + a4 = 9/4 ∨ a1 + a2 + a3 + a4 = -9/4 :=
sorry

end sum_of_rationals_eq_l82_82686


namespace ratio_problem_l82_82944

theorem ratio_problem (a b c d : ℚ) (h1 : a / b = 5 / 4) (h2 : c / d = 4 / 1) (h3 : d / b = 1 / 8) :
  a / c = 5 / 2 := by
  sorry

end ratio_problem_l82_82944


namespace question_l82_82238

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.log x - 7

theorem question (x : ℝ) (n : ℕ) (h1 : 2 < x ∧ x < 3) (h2 : f x = 0) : n = 2 := by
  sorry

end question_l82_82238


namespace hexagon_exists_equal_sides_four_equal_angles_hexagon_exists_equal_angles_four_equal_sides_l82_82367

theorem hexagon_exists_equal_sides_four_equal_angles : 
  ∃ (A B C D E F : Type) (AB BC CD DE EF FA : ℝ) (angle_A angle_B angle_C angle_D angle_E angle_F : ℝ), 
  (AB = BC ∧ BC = CD ∧ CD = DE ∧ DE = EF ∧ EF = FA ∧ FA = AB) ∧ 
  (angle_A = angle_B ∧ angle_B = angle_E ∧ angle_E = angle_F) ∧ 
  4 * angle_A + angle_C + angle_D = 720 :=
sorry

theorem hexagon_exists_equal_angles_four_equal_sides :
  ∃ (A B C D E F : Type) (AB BC CD DA : ℝ) (angle : ℝ), 
  (angle_A = angle_B ∧ angle_B = angle_C ∧ angle_C = angle_D ∧ angle_D = angle_E ∧ angle_E = angle_F ∧ angle_F = angle_A) ∧ 
  (AB = BC ∧ BC = CD ∧ CD = DA) :=
sorry

end hexagon_exists_equal_sides_four_equal_angles_hexagon_exists_equal_angles_four_equal_sides_l82_82367


namespace basil_pots_count_l82_82433

theorem basil_pots_count (B : ℕ) (h1 : 9 * 18 + 6 * 30 + 4 * B = 354) : B = 3 := 
by 
  -- This is just the signature of the theorem. The proof is omitted.
  sorry

end basil_pots_count_l82_82433


namespace certain_number_is_84_l82_82684

/-
The least number by which 72 must be multiplied in order to produce a multiple of a certain number is 14.
What is that certain number?
-/

theorem certain_number_is_84 (x : ℕ) (h: 72 * 14 % x = 0 ∧ ∀ y : ℕ, 1 ≤ y → y < 14 → 72 * y % x ≠ 0) : x = 84 :=
sorry

end certain_number_is_84_l82_82684


namespace low_card_value_is_one_l82_82370

-- Definitions and setting up the conditions
def num_high_cards : ℕ := 26
def num_low_cards : ℕ := 26
def high_card_points : ℕ := 2
def draw_scenarios : ℕ := 4

-- The point value of a low card L
noncomputable def low_card_points : ℕ :=
  if num_high_cards = 26 ∧ num_low_cards = 26 ∧ high_card_points = 2
     ∧ draw_scenarios = 4
  then 1 else 0 

theorem low_card_value_is_one :
  low_card_points = 1 :=
by
  sorry

end low_card_value_is_one_l82_82370


namespace average_age_combined_l82_82708

-- Definitions of the given conditions
def avg_age_fifth_graders := 10
def number_fifth_graders := 40
def avg_age_parents := 40
def number_parents := 60

-- The theorem we need to prove
theorem average_age_combined : 
  (avg_age_fifth_graders * number_fifth_graders + avg_age_parents * number_parents) / (number_fifth_graders + number_parents) = 28 := 
by
  sorry

end average_age_combined_l82_82708


namespace lilith_additional_fund_l82_82545

theorem lilith_additional_fund
  (num_water_bottles : ℕ)
  (original_price : ℝ)
  (reduced_price : ℝ)
  (expected_difference : ℝ)
  (h1 : num_water_bottles = 5 * 12)
  (h2 : original_price = 2)
  (h3 : reduced_price = 1.85)
  (h4 : expected_difference = 9) :
  (num_water_bottles * original_price) - (num_water_bottles * reduced_price) = expected_difference :=
by
  sorry

end lilith_additional_fund_l82_82545


namespace probability_of_selecting_particular_girl_l82_82098

-- Define the numbers involved
def total_population : ℕ := 60
def num_girls : ℕ := 25
def num_boys : ℕ := 35
def sample_size : ℕ := 5

-- Total number of basic events
def total_combinations : ℕ := Nat.choose total_population sample_size

-- Number of basic events that include a particular girl
def girl_combinations : ℕ := Nat.choose (total_population - 1) (sample_size - 1)

-- Probability of selecting a particular girl
def probability_of_girl_selection : ℚ := girl_combinations / total_combinations

-- The theorem to be proved
theorem probability_of_selecting_particular_girl :
  probability_of_girl_selection = 1 / 12 :=
by sorry

end probability_of_selecting_particular_girl_l82_82098


namespace odd_square_minus_one_divisible_by_eight_l82_82808

theorem odd_square_minus_one_divisible_by_eight (n : ℤ) : ∃ k : ℤ, ((2 * n + 1) ^ 2 - 1) = 8 * k := 
by
  sorry

end odd_square_minus_one_divisible_by_eight_l82_82808


namespace jay_savings_first_week_l82_82358

theorem jay_savings_first_week :
  ∀ (x : ℕ), (x + (x + 10) + (x + 20) + (x + 30) = 60) → x = 0 :=
by
  intro x h
  sorry

end jay_savings_first_week_l82_82358


namespace polygon_sides_l82_82147

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l82_82147


namespace distance_between_houses_l82_82154

-- Definitions
def speed : ℝ := 2          -- Amanda's speed in miles per hour
def time : ℝ := 3           -- Time taken by Amanda in hours

-- The theorem to prove distance is 6 miles
theorem distance_between_houses : speed * time = 6 := by
  sorry

end distance_between_houses_l82_82154


namespace enjoyable_gameplay_l82_82580

theorem enjoyable_gameplay (total_hours : ℕ) (boring_percentage : ℕ) (expansion_hours : ℕ)
  (h_total : total_hours = 100)
  (h_boring : boring_percentage = 80)
  (h_expansion : expansion_hours = 30) :
  ((1 - boring_percentage / 100) * total_hours + expansion_hours) = 50 := 
by
  sorry

end enjoyable_gameplay_l82_82580


namespace permutation_value_l82_82614

theorem permutation_value : ∀ (n r : ℕ), n = 5 → r = 3 → (n.choose r) * r.factorial = 60 := 
by
  intros n r hn hr 
  rw [hn, hr]
  -- We use the permutation formula A_{n}^{r} = n! / (n-r)!
  -- A_{5}^{3} = 5! / 2!
  -- Simplifies to 5 * 4 * 3 = 60.
  sorry

end permutation_value_l82_82614


namespace sequence_increasing_range_l82_82779

theorem sequence_increasing_range (a : ℝ) (h : ∀ n : ℕ, (n - a) ^ 2 < (n + 1 - a) ^ 2) :
  a < 3 / 2 :=
by
  sorry

end sequence_increasing_range_l82_82779


namespace three_solutions_no_solutions_2891_l82_82291

theorem three_solutions (n : ℤ) (hpos : n > 0) (hx : ∃ (x y : ℤ), x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
    x1^3 - 3 * x1 * y1^2 + y1^3 = n ∧ 
    x2^3 - 3 * x2 * y2^2 + y2^3 = n ∧ 
    x3^3 - 3 * x3 * y3^2 + y3^3 = n := 
sorry

theorem no_solutions_2891 : ¬ ∃ (x y : ℤ), x^3 - 3 * x * y^2 + y^3 = 2891 :=
sorry

end three_solutions_no_solutions_2891_l82_82291


namespace sequence_sum_l82_82862

theorem sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
    (h1 : a 1 = 1)
    (h_rec : ∀ n, a (n + 2) = 1 / (a n + 1))
    (h6_2 : a 6 = a 2) :
    a 2016 + a 3 = (Real.sqrt 5) / 2 :=
by
  sorry

end sequence_sum_l82_82862


namespace no_nat_nums_x4_minus_y4_eq_x3_plus_y3_l82_82811

theorem no_nat_nums_x4_minus_y4_eq_x3_plus_y3 : ∀ (x y : ℕ), x^4 - y^4 ≠ x^3 + y^3 :=
by
  intro x y
  sorry

end no_nat_nums_x4_minus_y4_eq_x3_plus_y3_l82_82811


namespace total_cost_l82_82244

def num_professionals := 2
def hours_per_professional_per_day := 6
def days_worked := 7
def hourly_rate := 15

theorem total_cost : 
  (num_professionals * hours_per_professional_per_day * days_worked * hourly_rate) = 1260 := by
  sorry

end total_cost_l82_82244


namespace intersection_A_B_l82_82961

open Set

def A : Set ℤ := {x | ∃ n : ℤ, x = 3 * n - 1}
def B : Set ℤ := {x | 0 < x ∧ x < 6}

theorem intersection_A_B : A ∩ B = {2, 5} := by
  sorry

end intersection_A_B_l82_82961


namespace square_number_n_value_l82_82044

theorem square_number_n_value
  (n : ℕ)
  (h : ∃ k : ℕ, 2^6 + 2^9 + 2^n = k^2) :
  n = 10 :=
sorry

end square_number_n_value_l82_82044


namespace problem_solution_l82_82817

theorem problem_solution :
  ∀ (x y : ℚ), 
  4 * x + y = 20 ∧ x + 2 * y = 17 → 
  5 * x^2 + 18 * x * y + 5 * y^2 = 696 + 5/7 := 
by 
  sorry

end problem_solution_l82_82817


namespace square_difference_l82_82251

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 64) (h₂ : x * y = 12) : (x - y)^2 = 16 := by
  -- proof would go here
  sorry

end square_difference_l82_82251


namespace salesmans_profit_l82_82421

-- Define the initial conditions and given values
def backpacks_bought : ℕ := 72
def cost_price : ℕ := 1080
def swap_meet_sales : ℕ := 25
def swap_meet_price : ℕ := 20
def department_store_sales : ℕ := 18
def department_store_price : ℕ := 30
def online_sales : ℕ := 12
def online_price : ℕ := 28
def shipping_expenses : ℕ := 40
def local_market_price : ℕ := 24

-- Calculate the total revenue from each channel
def swap_meet_revenue : ℕ := swap_meet_sales * swap_meet_price
def department_store_revenue : ℕ := department_store_sales * department_store_price
def online_revenue : ℕ := (online_sales * online_price) - shipping_expenses

-- Calculate remaining backpacks and local market revenue
def backpacks_sold : ℕ := swap_meet_sales + department_store_sales + online_sales
def backpacks_left : ℕ := backpacks_bought - backpacks_sold
def local_market_revenue : ℕ := backpacks_left * local_market_price

-- Calculate total revenue and profit
def total_revenue : ℕ := swap_meet_revenue + department_store_revenue + online_revenue + local_market_revenue
def profit : ℕ := total_revenue - cost_price

-- State the theorem for the salesman's profit
theorem salesmans_profit : profit = 664 := by
  sorry

end salesmans_profit_l82_82421


namespace child_tickets_sold_l82_82469

theorem child_tickets_sold
  (A C : ℕ)
  (h1 : A + C = 130)
  (h2 : 12 * A + 4 * C = 840) : C = 90 :=
  by {
  -- Proof skipped
  sorry
}

end child_tickets_sold_l82_82469


namespace find_arc_length_of_sector_l82_82583

variable (s r p : ℝ)
variable (h_s : s = 4)
variable (h_r : r = 2)
variable (h_area : 2 * s = r * p)

theorem find_arc_length_of_sector 
  (h_s : s = 4) (h_r : r = 2) (h_area : 2 * s = r * p) :
  p = 4 :=
sorry

end find_arc_length_of_sector_l82_82583


namespace range_of_a_l82_82078

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧ 
  (∀ x, |x^3 - a * x^2| = x → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) →
  a > 2 :=
by
  -- The proof is to be provided here.
  sorry

end range_of_a_l82_82078


namespace ordered_pair_proportional_l82_82123

theorem ordered_pair_proportional (p q : ℝ) (h : (3 : ℝ) • (-4 : ℝ) = (5 : ℝ) • p ∧ (3 : ℝ) • q = (5 : ℝ) • (-4 : ℝ)) :
  (p, q) = (5 / 2, -8) :=
by
  sorry

end ordered_pair_proportional_l82_82123


namespace intersection_of_A_and_B_l82_82625

variable (A : Set ℝ)
variable (B : Set ℝ)
variable (C : Set ℝ)

theorem intersection_of_A_and_B (hA : A = { x | -1 < x ∧ x < 3 })
                                (hB : B = { -1, 1, 2 })
                                (hC : C = { 1, 2 }) :
  A ∩ B = C := by
  sorry

end intersection_of_A_and_B_l82_82625


namespace infinite_sum_computation_l82_82509

theorem infinite_sum_computation : 
  ∑' n : ℕ, (3 * (n + 1) + 2) / (n * (n + 1) * (n + 3)) = 10 / 3 :=
by sorry

end infinite_sum_computation_l82_82509


namespace minimal_disks_needed_l82_82284

-- Define the capacity of one disk
def disk_capacity : ℝ := 2.0

-- Define the number of files and their sizes
def num_files_0_9 : ℕ := 5
def size_file_0_9 : ℝ := 0.9

def num_files_0_8 : ℕ := 15
def size_file_0_8 : ℝ := 0.8

def num_files_0_5 : ℕ := 20
def size_file_0_5 : ℝ := 0.5

-- Total number of files
def total_files : ℕ := num_files_0_9 + num_files_0_8 + num_files_0_5

-- Proof statement: the minimal number of disks needed to store all files given their sizes and the disk capacity
theorem minimal_disks_needed : 
  ∀ (d : ℕ), 
    d = 18 → 
    total_files = 40 → 
    disk_capacity = 2.0 → 
    ((num_files_0_9 * size_file_0_9 + num_files_0_8 * size_file_0_8 + num_files_0_5 * size_file_0_5) / disk_capacity) ≤ d
  :=
by
  sorry

end minimal_disks_needed_l82_82284


namespace john_total_cost_l82_82363

theorem john_total_cost :
  let computer_cost := 1500
  let peripherals_cost := computer_cost / 4
  let base_video_card_cost := 300
  let upgraded_video_card_cost := 2.5 * base_video_card_cost
  let video_card_discount := 0.12 * upgraded_video_card_cost
  let upgraded_video_card_final_cost := upgraded_video_card_cost - video_card_discount
  let foreign_monitor_cost_local := 200
  let exchange_rate := 1.25
  let foreign_monitor_cost_usd := foreign_monitor_cost_local / exchange_rate
  let peripherals_sales_tax := 0.05 * peripherals_cost
  let subtotal := computer_cost + peripherals_cost + upgraded_video_card_final_cost + peripherals_sales_tax
  let store_loyalty_discount := 0.07 * (computer_cost + peripherals_cost + upgraded_video_card_final_cost)
  let final_cost := subtotal - store_loyalty_discount + foreign_monitor_cost_usd
  final_cost = 2536.30 := sorry

end john_total_cost_l82_82363


namespace probability_C_D_l82_82075

variable (P : String → ℚ)

axiom h₁ : P "A" = 1/4
axiom h₂ : P "B" = 1/3
axiom h₃ : P "A" + P "B" + P "C" + P "D" = 1

theorem probability_C_D : P "C" + P "D" = 5/12 := by
  sorry

end probability_C_D_l82_82075


namespace min_value_of_f_min_value_at_x_1_l82_82752

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - 2 * x) + 1 / (2 - 3 * x)

theorem min_value_of_f :
  ∀ x : ℝ, x > 0 → f x ≥ 35 :=
by
  sorry

-- As an additional statement, we can check the specific case at x = 1
theorem min_value_at_x_1 :
  f 1 = 35 :=
by
  sorry

end min_value_of_f_min_value_at_x_1_l82_82752


namespace solve_chimney_bricks_l82_82083

noncomputable def chimney_bricks (x : ℝ) : Prop :=
  let brenda_rate := x / 8
  let brandon_rate := x / 12
  let combined_rate := brenda_rate + brandon_rate - 15
  (combined_rate * 6) = x

theorem solve_chimney_bricks : ∃ (x : ℝ), chimney_bricks x ∧ x = 360 :=
by
  use 360
  unfold chimney_bricks
  sorry

end solve_chimney_bricks_l82_82083


namespace problem_l82_82830

open Function

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + m) + a (n - m) = 2 * a n

theorem problem (h_arith : arithmetic_sequence a) (h_eq : a 1 + 3 * a 6 + a 11 = 10) :
  a 5 + a 7 = 4 := 
sorry

end problem_l82_82830


namespace area_before_halving_l82_82716

theorem area_before_halving (A : ℝ) (h : A / 2 = 7) : A = 14 :=
sorry

end area_before_halving_l82_82716


namespace max_n_satisfying_property_l82_82836

theorem max_n_satisfying_property :
  ∃ n : ℕ, (0 < n) ∧ (∀ m : ℕ, Nat.gcd m n = 1 → m^6 % n = 1) ∧ n = 504 :=
by
  sorry

end max_n_satisfying_property_l82_82836


namespace crayons_left_l82_82585

def initial_crayons : ℕ := 253
def lost_or_given_away_crayons : ℕ := 70
def remaining_crayons : ℕ := 183

theorem crayons_left (initial_crayons : ℕ) (lost_or_given_away_crayons : ℕ) (remaining_crayons : ℕ) :
  initial_crayons - lost_or_given_away_crayons = remaining_crayons :=
by {
  sorry
}

end crayons_left_l82_82585


namespace volume_between_spheres_l82_82487

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_between_spheres :
  volume_of_sphere 10 - volume_of_sphere 4 = (3744 / 3) * Real.pi := by
  sorry

end volume_between_spheres_l82_82487


namespace tangents_intersection_perpendicular_parabola_l82_82891

theorem tangents_intersection_perpendicular_parabola :
  ∀ (C D : ℝ × ℝ), C.2 = 4 * C.1 ^ 2 → D.2 = 4 * D.1 ^ 2 → 
  (8 * C.1) * (8 * D.1) = -1 → 
  ∃ Q : ℝ × ℝ, Q.2 = -1 / 16 :=
by
  sorry

end tangents_intersection_perpendicular_parabola_l82_82891


namespace percentage_of_good_fruits_l82_82609

theorem percentage_of_good_fruits (total_oranges : ℕ) (total_bananas : ℕ) 
    (rotten_oranges_percent : ℝ) (rotten_bananas_percent : ℝ) :
    total_oranges = 600 ∧ total_bananas = 400 ∧ 
    rotten_oranges_percent = 0.15 ∧ rotten_bananas_percent = 0.03 →
    (510 + 388) / (600 + 400) * 100 = 89.8 :=
by
  intros
  sorry

end percentage_of_good_fruits_l82_82609


namespace circle_area_from_intersection_l82_82307

-- Statement of the problem
theorem circle_area_from_intersection (r : ℝ) (A B : ℝ × ℝ)
  (h_circle : ∀ x y, (x + 2) ^ 2 + y ^ 2 = r ^ 2 ↔ (x, y) = A ∨ (x, y) = B)
  (h_parabola : ∀ x y, y ^ 2 = 20 * x ↔ (x, y) = A ∨ (x, y) = B)
  (h_axis_sym : A.1 = -5 ∧ B.1 = -5)
  (h_AB_dist : |A.2 - B.2| = 8) : π * r ^ 2 = 25 * π :=
by
  sorry

end circle_area_from_intersection_l82_82307


namespace cylinder_original_radius_l82_82117

theorem cylinder_original_radius 
  (r h : ℝ) 
  (hr_eq : h = 3)
  (volume_increase_radius : Real.pi * (r + 8)^2 * 3 = Real.pi * r^2 * 11) :
  r = 8 :=
by
  -- the proof steps will be here
  sorry

end cylinder_original_radius_l82_82117


namespace unique_passenger_counts_l82_82103

def train_frequencies : Nat × Nat × Nat := (6, 4, 3)
def train_passengers_leaving : Nat × Nat × Nat := (200, 300, 150)
def train_passengers_taking : Nat × Nat × Nat := (320, 400, 280)
def trains_per_hour (freq : Nat) : Nat := 60 / freq

def total_passengers_leaving : Nat :=
  let t1 := (trains_per_hour 10) * 200
  let t2 := (trains_per_hour 15) * 300
  let t3 := (trains_per_hour 20) * 150
  t1 + t2 + t3

def total_passengers_taking : Nat :=
  let t1 := (trains_per_hour 10) * 320
  let t2 := (trains_per_hour 15) * 400
  let t3 := (trains_per_hour 20) * 280
  t1 + t2 + t3

theorem unique_passenger_counts :
  total_passengers_leaving = 2850 ∧ total_passengers_taking = 4360 := by
  sorry

end unique_passenger_counts_l82_82103


namespace even_n_ineq_l82_82637

theorem even_n_ineq (n : ℕ) (h : ∀ x : ℝ, 3 * x^n + n * (x + 2) - 3 ≥ n * x^2) : Even n :=
  sorry

end even_n_ineq_l82_82637


namespace pond_capacity_l82_82070

theorem pond_capacity :
  let normal_rate := 6 -- gallons per minute
  let restriction_rate := (2/3 : ℝ) * normal_rate -- gallons per minute
  let time := 50 -- minutes
  let capacity := restriction_rate * time -- total capacity in gallons
  capacity = 200 := sorry

end pond_capacity_l82_82070


namespace product_of_two_numbers_l82_82374

theorem product_of_two_numbers :
  ∃ (a b : ℚ), (∀ k : ℚ, a = k + b) ∧ (∀ k : ℚ, a + b = 8 * k) ∧ (∀ k : ℚ, a * b = 40 * k) ∧ (a * b = 6400 / 63) :=
by {
  sorry
}

end product_of_two_numbers_l82_82374


namespace fraction_decomposition_l82_82597

theorem fraction_decomposition :
  ∃ (A B : ℚ), 
  (A = 27 / 10) ∧ (B = -11 / 10) ∧ 
  (∀ x : ℚ, 
    7 * x - 13 = A * (3 * x - 4) + B * (x + 2)) := 
  sorry

end fraction_decomposition_l82_82597


namespace find_y_eq_54_div_23_l82_82139

open BigOperators

theorem find_y_eq_54_div_23 (y : ℚ) (h : (Real.sqrt (8 * y) / Real.sqrt (6 * (y - 2))) = 3) : y = 54 / 23 := 
by
  sorry

end find_y_eq_54_div_23_l82_82139


namespace UncleVanya_travel_time_l82_82869

-- Define the conditions
variables (x y z : ℝ)
variables (h1 : 2 * x + 3 * y + 20 * z = 66)
variables (h2 : 5 * x + 8 * y + 30 * z = 144)

-- Question: how long will it take to walk 4 km, cycle 5 km, and drive 80 km
theorem UncleVanya_travel_time : 4 * x + 5 * y + 80 * z = 174 :=
sorry

end UncleVanya_travel_time_l82_82869


namespace find_t_l82_82177

theorem find_t (c o u n t s : ℕ)
    (hc : c ≠ 0) (ho : o ≠ 0) (hn : n ≠ 0) (ht : t ≠ 0) (hs : s ≠ 0)
    (h1 : c + o = u)
    (h2 : u + n = t + 1)
    (h3 : t + c = s)
    (h4 : o + n + s = 15) :
    t = 7 := 
sorry

end find_t_l82_82177


namespace find_m_l82_82148

theorem find_m (m : ℝ) (A B C D : ℝ × ℝ)
  (h1 : A = (m, 1)) (h2 : B = (-3, 4))
  (h3 : C = (0, 2)) (h4 : D = (1, 1))
  (h_parallel : (4 - 1) / (-3 - m) = (1 - 2) / (1 - 0)) :
  m = 0 :=
  by
  sorry

end find_m_l82_82148


namespace tan_theta_correct_l82_82234

noncomputable def cos_double_angle (θ : ℝ) : ℝ := 2 * Real.cos θ ^ 2 - 1

theorem tan_theta_correct (θ : ℝ) (hθ₁ : θ > 0) (hθ₂ : θ < Real.pi / 2) 
  (h : 15 * cos_double_angle θ - 14 * Real.cos θ + 11 = 0) : Real.tan θ = Real.sqrt 5 / 2 :=
sorry

end tan_theta_correct_l82_82234


namespace Ceva_theorem_l82_82777

variables {A B C K L M P : Point}
variables {BK KC CL LA AM MB : ℝ}

-- Assume P is inside the triangle ABC and KP, LP, and MP intersect BC, CA, and AB at points K, L, and M respectively
-- We need to prove the ratio product property according to Ceva's theorem
theorem Ceva_theorem 
  (h1: BK / KC = b)
  (h2: CL / LA = c)
  (h3: AM / MB = a)
  (h4: (b * c * a = 1)): 
  (BK / KC) * (CL / LA) * (AM / MB) = 1 :=
sorry

end Ceva_theorem_l82_82777


namespace reflection_matrix_determine_l82_82519

theorem reflection_matrix_determine (a b : ℚ)
  (h1 : (a^2 - (3/4) * b) = 1)
  (h2 : (-(3/4) * b + (1/16)) = 1)
  (h3 : (a * b + (1/4) * b) = 0)
  (h4 : (-(3/4) * a - (3/16)) = 0) :
  (a, b) = (1/4, -5/4) := 
sorry

end reflection_matrix_determine_l82_82519


namespace find_a_even_function_l82_82203

theorem find_a_even_function (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f x = (x + 1) * (x + a))  
  (h2 : ∀ x, f x = f (-x)) : a = -1 :=
sorry

end find_a_even_function_l82_82203


namespace largest_satisfying_n_correct_l82_82072
noncomputable def largest_satisfying_n : ℕ := 4

theorem largest_satisfying_n_correct :
  ∀ n x, (1 < x ∧ x < 2 ∧ 2 < x^2 ∧ x^2 < 3 ∧ 3 < x^3 ∧ x^3 < 4 ∧ 4 < x^4 ∧ x^4 < 5) 
  → n = largest_satisfying_n ∧
  ¬ (∃ x, (1 < x ∧ x < 2 ∧ 2 < x^2 ∧ x^2 < 3 ∧ 3 < x^3 ∧ x^3 < 4 ∧ 4 < x^4 ∧ x^4 < 5 ∧ 5 < x^5 ∧ x^5 < 6)) := sorry

end largest_satisfying_n_correct_l82_82072


namespace histogram_height_representation_l82_82153

theorem histogram_height_representation (freq_ratio : ℝ) (frequency : ℝ) (class_interval : ℝ) 
  (H : freq_ratio = frequency / class_interval) : 
  freq_ratio = frequency / class_interval :=
by 
  sorry

end histogram_height_representation_l82_82153


namespace domain_of_f_eq_R_l82_82425

noncomputable def f (x m : ℝ) : ℝ := (x - 4) / (m * x^2 + 4 * m * x + 3)

theorem domain_of_f_eq_R (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x + 3 ≠ 0) ↔ (0 ≤ m ∧ m < 3 / 4) :=
by
  sorry

end domain_of_f_eq_R_l82_82425


namespace west_movement_is_negative_seven_l82_82342

-- Define a function to represent the movement notation
def movement_notation (direction: String) (distance: Int) : Int :=
  if direction = "east" then distance else -distance

-- Define the movement in the east direction
def east_movement := movement_notation "east" 3

-- Define the movement in the west direction
def west_movement := movement_notation "west" 7

-- Theorem statement
theorem west_movement_is_negative_seven : west_movement = -7 := by
  sorry

end west_movement_is_negative_seven_l82_82342


namespace shelves_needed_l82_82258

def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_per_shelf : ℕ := 4
def remaining_books : ℕ := total_books - books_taken_by_librarian
def needed_shelves : ℕ := 9

theorem shelves_needed : remaining_books / books_per_shelf = needed_shelves := by
  sorry

end shelves_needed_l82_82258


namespace match_piles_l82_82306

theorem match_piles (a b c : ℕ) (h : a + b + c = 96)
    (h1 : 2 * b = a + c) (h2 : 2 * c = b + a) (h3 : 2 * a = c + b) : 
    a = 44 ∧ b = 28 ∧ c = 24 :=
  sorry

end match_piles_l82_82306


namespace probability_defective_first_lathe_overall_probability_defective_conditional_probability_second_lathe_conditional_probability_third_lathe_l82_82379

noncomputable def defect_rate_first_lathe : ℝ := 0.06
noncomputable def defect_rate_second_lathe : ℝ := 0.05
noncomputable def defect_rate_third_lathe : ℝ := 0.05
noncomputable def proportion_first_lathe : ℝ := 0.25
noncomputable def proportion_second_lathe : ℝ := 0.30
noncomputable def proportion_third_lathe : ℝ := 0.45

theorem probability_defective_first_lathe :
  defect_rate_first_lathe * proportion_first_lathe = 0.015 :=
by sorry

theorem overall_probability_defective :
  defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe = 0.0525 :=
by sorry

theorem conditional_probability_second_lathe :
  (defect_rate_second_lathe * proportion_second_lathe) /
  (defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe) = 2 / 7 :=
by sorry

theorem conditional_probability_third_lathe :
  (defect_rate_third_lathe * proportion_third_lathe) /
  (defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe) = 3 / 7 :=
by sorry

end probability_defective_first_lathe_overall_probability_defective_conditional_probability_second_lathe_conditional_probability_third_lathe_l82_82379


namespace ratio_bc_cd_l82_82783

-- Definitions based on given conditions.
variable (a b c d e : ℝ)
variable (h_ab : b - a = 5)
variable (h_ac : c - a = 11)
variable (h_de : e - d = 8)
variable (h_ae : e - a = 22)

-- The theorem to prove bc : cd = 2 : 1.
theorem ratio_bc_cd (h_ab : b - a = 5) (h_ac : c - a = 11) (h_de : e - d = 8) (h_ae : e - a = 22) :
  (c - b) / (d - c) = 2 :=
by
  sorry

end ratio_bc_cd_l82_82783


namespace prob_single_trial_l82_82889

theorem prob_single_trial (P : ℝ) : 
  (1 - (1 - P)^4) = 65 / 81 → P = 1 / 3 :=
by
  intro h
  sorry

end prob_single_trial_l82_82889


namespace difference_max_min_planes_l82_82459

open Set

-- Defining the regular tetrahedron and related concepts
noncomputable def tetrahedron := Unit -- Placeholder for the tetrahedron

def union_faces (T : Unit) : Set Point := sorry -- Placeholder for union of faces definition

noncomputable def simple_trace (p : Plane) (T : Unit) : Set Point := sorry -- Placeholder for planes intersecting faces

-- Calculating number of planes
def maximum_planes (T : Unit) : Nat :=
  4 -- One for each face of the tetrahedron

def minimum_planes (T : Unit) : Nat :=
  2 -- Each plane covers traces on two adjacent faces if oriented appropriately

-- Statement of the problem
theorem difference_max_min_planes (T : Unit) :
  maximum_planes T - minimum_planes T = 2 :=
by
  -- Proof skipped
  sorry

end difference_max_min_planes_l82_82459


namespace evaluate_expression_l82_82826

-- Define the conditions
def two_pow_nine : ℕ := 2 ^ 9
def neg_one_pow_eight : ℤ := (-1) ^ 8

-- Define the proof statement
theorem evaluate_expression : two_pow_nine + neg_one_pow_eight = 513 := 
by
  sorry

end evaluate_expression_l82_82826


namespace tripod_height_l82_82689

-- Define the conditions of the problem
structure Tripod where
  leg_length : ℝ
  angle_equal : Bool
  top_height : ℝ
  broken_length : ℝ

def m : ℕ := 27
def n : ℕ := 10

noncomputable def h : ℝ := m / Real.sqrt n

theorem tripod_height :
  ∀ (t : Tripod),
  t.leg_length = 6 →
  t.angle_equal = true →
  t.top_height = 3 →
  t.broken_length = 2 →
  (h = m / Real.sqrt n) →
  (⌊m + Real.sqrt n⌋ = 30) :=
by
  intros
  sorry

end tripod_height_l82_82689


namespace number_of_integers_with_abs_val_conditions_l82_82623

theorem number_of_integers_with_abs_val_conditions : 
  (∃ n : ℕ, n = 8) :=
by sorry

end number_of_integers_with_abs_val_conditions_l82_82623


namespace minimum_value_of_f_is_15_l82_82248

noncomputable def f (x : ℝ) : ℝ := 9 * x + (1 / (x - 1))

theorem minimum_value_of_f_is_15 (h : ∀ x, x > 1) : ∃ x, x > 1 ∧ f x = 15 :=
by sorry

end minimum_value_of_f_is_15_l82_82248


namespace trajectory_eq_l82_82166

-- Define the conditions provided in the problem
def circle_eq (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * (m + 3) * x + 2 * (1 - 4 * m^2) + 16 * m^4 + 9 = 0

-- Define the required range for m based on the derivation
def m_valid (m : ℝ) : Prop :=
  -1/7 < m ∧ m < 1

-- Prove that the equation of the trajectory of the circle's center is y = 4(x-3)^2 -1 
-- and it's valid in the required range for x
theorem trajectory_eq (x y : ℝ) :
  (∃ m : ℝ, m_valid m ∧ y = 4 * (x - 3)^2 - 1 ∧ (x = m + 3) ∧ (y = 4 * m^2 - 1)) →
  y = 4 * (x - 3)^2 - 1 ∧ (20/7 < x) ∧ (x < 4) :=
by
  intro h
  cases' h with m hm
  sorry

end trajectory_eq_l82_82166


namespace tree_shadow_length_l82_82906

theorem tree_shadow_length (jane_shadow : ℝ) (jane_height : ℝ) (tree_height : ℝ) (tree_shadow : ℝ)
  (h₁ : jane_shadow = 0.5)
  (h₂ : jane_height = 1.5)
  (h₃ : tree_height = 30)
  (h₄ : jane_height / jane_shadow = tree_height / tree_shadow)
  : tree_shadow = 10 :=
by
  -- skipping the proof steps
  sorry

end tree_shadow_length_l82_82906


namespace rational_reciprocal_pow_2014_l82_82259

theorem rational_reciprocal_pow_2014 (a : ℚ) (h : a = 1 / a) : a ^ 2014 = 1 := by
  sorry

end rational_reciprocal_pow_2014_l82_82259


namespace bike_price_l82_82221

variable (p : ℝ)

def percent_upfront_payment : ℝ := 0.20
def upfront_payment : ℝ := 200

theorem bike_price (h : percent_upfront_payment * p = upfront_payment) : p = 1000 := by
  sorry

end bike_price_l82_82221


namespace neg_all_cups_full_l82_82730

variable (x : Type) (cup : x → Prop) (full : x → Prop)

theorem neg_all_cups_full :
  ¬ (∀ x, cup x → full x) = ∃ x, cup x ∧ ¬ full x := by
sorry

end neg_all_cups_full_l82_82730


namespace hillary_stops_short_of_summit_l82_82036

noncomputable def distance_to_summit_from_base_camp : ℝ := 4700
noncomputable def hillary_climb_rate : ℝ := 800
noncomputable def eddy_climb_rate : ℝ := 500
noncomputable def hillary_descent_rate : ℝ := 1000
noncomputable def time_of_departure : ℝ := 6
noncomputable def time_of_passing : ℝ := 12

theorem hillary_stops_short_of_summit :
  ∃ x : ℝ, 
    (time_of_passing - time_of_departure) * hillary_climb_rate = distance_to_summit_from_base_camp - x →
    (time_of_passing - time_of_departure) * eddy_climb_rate = x →
    x = 2900 :=
by
  sorry

end hillary_stops_short_of_summit_l82_82036


namespace minimum_value_of_y_exists_l82_82626

theorem minimum_value_of_y_exists :
  ∃ (y : ℝ), (∀ (x : ℝ), (y + x) = (y - x)^2 + 3 * (y - x) + 3) ∧ y = -1/2 :=
by sorry

end minimum_value_of_y_exists_l82_82626


namespace value_of_a3_a5_l82_82384

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = a n * r

theorem value_of_a3_a5 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : 
  a 3 + a 5 = 5 :=
  sorry

end value_of_a3_a5_l82_82384


namespace tan_angle_addition_l82_82196

theorem tan_angle_addition (y : ℝ) (hyp : Real.tan y = -3) : 
  Real.tan (y + Real.pi / 3) = - (5 * Real.sqrt 3 - 6) / 13 := 
by 
  sorry

end tan_angle_addition_l82_82196


namespace number_of_subcommittees_l82_82538

theorem number_of_subcommittees :
  ∃ (k : ℕ), ∀ (num_people num_sub_subcommittees subcommittee_size : ℕ), 
  num_people = 360 → 
  num_sub_subcommittees = 3 → 
  subcommittee_size = 6 → 
  k = (num_people * num_sub_subcommittees) / subcommittee_size :=
sorry

end number_of_subcommittees_l82_82538


namespace common_chord_length_l82_82530

noncomputable def dist_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / Real.sqrt (a^2 + b^2)

theorem common_chord_length
  (x y : ℝ)
  (h1 : (x-2)^2 + (y-1)^2 = 10)
  (h2 : (x+6)^2 + (y+3)^2 = 50) :
  (dist_to_line (2, 1) 2 1 0 = Real.sqrt 5) →
  2 * Real.sqrt 5 = 2 * Real.sqrt 5 :=
by
  sorry

end common_chord_length_l82_82530


namespace inverse_function_b_value_l82_82151

theorem inverse_function_b_value (b : ℝ) :
  (∀ x, ∃ y, 2^x + b = y) ∧ (∃ x, ∃ y, (x, y) = (2, 5)) → b = 1 :=
by
  sorry

end inverse_function_b_value_l82_82151


namespace percent_democrats_is_60_l82_82624
-- Import the necessary library

-- Define the problem conditions
variables (D R : ℝ)
variables (h1 : D + R = 100)
variables (h2 : 0.70 * D + 0.20 * R = 50)

-- State the theorem to be proved
theorem percent_democrats_is_60 (D R : ℝ) (h1 : D + R = 100) (h2 : 0.70 * D + 0.20 * R = 50) : D = 60 :=
by
  sorry

end percent_democrats_is_60_l82_82624


namespace power_function_point_l82_82239

theorem power_function_point (n : ℕ) (hn : 2^n = 8) : n = 3 := 
by
  sorry

end power_function_point_l82_82239


namespace le_condition_l82_82348

-- Given positive numbers a, b, c
variables {a b c : ℝ}
-- Assume positive values for the numbers
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)
-- Given condition a² + b² - ab = c²
axiom condition : a^2 + b^2 - a*b = c^2

-- We need to prove (a - c)(b - c) ≤ 0
theorem le_condition : (a - c) * (b - c) ≤ 0 :=
sorry

end le_condition_l82_82348


namespace sum_squares_6_to_14_l82_82456

def sum_of_squares (n : ℕ) := (n * (n + 1) * (2 * n + 1)) / 6

theorem sum_squares_6_to_14 :
  (sum_of_squares 14) - (sum_of_squares 5) = 960 :=
by
  sorry

end sum_squares_6_to_14_l82_82456


namespace linear_function_m_value_l82_82393

theorem linear_function_m_value (m : ℝ) (h : abs (m + 1) = 1) : m = -2 :=
sorry

end linear_function_m_value_l82_82393


namespace arccos_one_eq_zero_l82_82318

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l82_82318


namespace crayons_count_l82_82805

-- Define the initial number of crayons
def initial_crayons : ℕ := 1453

-- Define the number of crayons given away
def crayons_given_away : ℕ := 563

-- Define the number of crayons lost
def crayons_lost : ℕ := 558

-- Define the final number of crayons left
def final_crayons_left : ℕ := initial_crayons - crayons_given_away - crayons_lost

-- State that the final number of crayons left is 332
theorem crayons_count : final_crayons_left = 332 :=
by
    -- This is where the proof would go, which we're skipping with sorry
    sorry

end crayons_count_l82_82805


namespace max_value_of_y_l82_82800

noncomputable def maxY (x y : ℝ) : ℝ :=
  if x^2 + y^2 = 10 * x + 60 * y then y else 0

theorem max_value_of_y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 60 * y) : 
  y ≤ 30 + 5 * Real.sqrt 37 :=
sorry

end max_value_of_y_l82_82800


namespace division_remainder_l82_82712

theorem division_remainder (dividend divisor quotient : ℕ) (h_dividend : dividend = 131) (h_divisor : divisor = 14) (h_quotient : quotient = 9) :
  ∃ remainder : ℕ, dividend = divisor * quotient + remainder ∧ remainder = 5 :=
by
  sorry

end division_remainder_l82_82712


namespace restore_original_problem_l82_82761

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l82_82761


namespace total_cups_used_l82_82020

theorem total_cups_used (butter flour sugar : ℕ) (h1 : 2 * sugar = 3 * butter) (h2 : 5 * sugar = 3 * flour) (h3 : sugar = 12) : butter + flour + sugar = 40 :=
by
  sorry

end total_cups_used_l82_82020


namespace total_bottles_capped_in_10_minutes_l82_82018

-- Define the capacities per minute for the three machines
def machine_a_capacity : ℕ := 12
def machine_b_capacity : ℕ := machine_a_capacity - 2
def machine_c_capacity : ℕ := machine_b_capacity + 5

-- Define the total capping capacity for 10 minutes
def total_capacity_in_10_minutes (a b c : ℕ) : ℕ := a * 10 + b * 10 + c * 10

-- The theorem we aim to prove
theorem total_bottles_capped_in_10_minutes :
  total_capacity_in_10_minutes machine_a_capacity machine_b_capacity machine_c_capacity = 370 :=
by
  -- Directly use the capacities defined above
  sorry

end total_bottles_capped_in_10_minutes_l82_82018


namespace cos_double_angle_l82_82322

theorem cos_double_angle (α : ℝ) (h : Real.cos (α + Real.pi / 2) = 3 / 5) : Real.cos (2 * α) = 7 / 25 :=
by 
  sorry

end cos_double_angle_l82_82322


namespace difference_of_triangular_23_and_21_l82_82071

def triangular (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem difference_of_triangular_23_and_21 : triangular 23 - triangular 21 = 45 :=
sorry

end difference_of_triangular_23_and_21_l82_82071


namespace diameter_of_circumscribed_circle_l82_82162

theorem diameter_of_circumscribed_circle (a : ℝ) (A : ℝ) (D : ℝ) 
  (h1 : a = 12) (h2 : A = 30) : D = 24 :=
by
  sorry

end diameter_of_circumscribed_circle_l82_82162


namespace find_integers_l82_82781

theorem find_integers (a b m : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a + b^2) * (b + a^2) = 2^m → a = 1 ∧ b = 1 ∧ m = 2 :=
by
  sorry

end find_integers_l82_82781


namespace score_of_tenth_game_must_be_at_least_l82_82611

variable (score_5 average_9 average_10 score_10 : ℤ)
variable (H1 : average_9 > score_5 / 5)
variable (H2 : average_10 > 18)
variable (score_6 score_7 score_8 score_9 : ℤ)
variable (H3 : score_6 = 23)
variable (H4 : score_7 = 14)
variable (H5 : score_8 = 11)
variable (H6 : score_9 = 20)
variable (H7 : average_9 = (score_5 + score_6 + score_7 + score_8 + score_9) / 9)
variable (H8 : average_10 = (score_5 + score_6 + score_7 + score_8 + score_9 + score_10) / 10)

theorem score_of_tenth_game_must_be_at_least :
  score_10 ≥ 29 :=
by
  sorry

end score_of_tenth_game_must_be_at_least_l82_82611


namespace Meghan_total_money_l82_82352

theorem Meghan_total_money (h100 : ℕ) (h50 : ℕ) (h10 : ℕ) : 
  h100 = 2 → h50 = 5 → h10 = 10 → 100 * h100 + 50 * h50 + 10 * h10 = 550 :=
by
  sorry

end Meghan_total_money_l82_82352


namespace avg_payment_correct_l82_82426

def first_payment : ℕ := 410
def additional_amount : ℕ := 65
def num_first_payments : ℕ := 8
def num_remaining_payments : ℕ := 44
def total_installments : ℕ := num_first_payments + num_remaining_payments

def total_first_payments : ℕ := num_first_payments * first_payment
def remaining_payment : ℕ := first_payment + additional_amount
def total_remaining_payments : ℕ := num_remaining_payments * remaining_payment

def total_payment : ℕ := total_first_payments + total_remaining_payments
def average_payment : ℚ := total_payment / total_installments

theorem avg_payment_correct : average_payment = 465 := by
  sorry

end avg_payment_correct_l82_82426


namespace solution_set_of_abs_inequality_l82_82994

theorem solution_set_of_abs_inequality :
  { x : ℝ | |x^2 - 2| < 2 } = { x : ℝ | -2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2 } :=
sorry

end solution_set_of_abs_inequality_l82_82994


namespace distance_between_centers_l82_82588

-- Define the points P, Q, R in the plane
variable (P Q R : ℝ × ℝ)

-- Define the lengths PQ, PR, and QR
variable (PQ PR QR : ℝ)
variable (is_right_triangle : ∃ (a b c : ℝ), PQ = a ∧ PR = b ∧ QR = c ∧ a^2 + b^2 = c^2)

-- Define the inradii r1, r2, r3 for triangles PQR, RST, and QUV respectively
variable (r1 r2 r3 : ℝ)

-- Assume PQ = 90, PR = 120, and QR = 150
axiom PQ_length : PQ = 90
axiom PR_length : PR = 120
axiom QR_length : QR = 150

-- Define the centers O2 and O3 of the circles C2 and C3 respectively
variable (O2 O3 : ℝ × ℝ)

-- Assume the inradius length is 30 for the initial triangle
axiom inradius_PQR : r1 = 30

-- Assume the positions of the centers of C2 and C3
axiom O2_position : O2 = (15, 75)
axiom O3_position : O3 = (70, 10)

-- Use the distance formula to express the final result
theorem distance_between_centers : ∃ n : ℕ, dist O2 O3 = Real.sqrt (10 * n) ∧ n = 725 :=
by
  sorry

end distance_between_centers_l82_82588


namespace parallel_tangents_a3_plus_b2_plus_d_eq_seven_l82_82324

theorem parallel_tangents_a3_plus_b2_plus_d_eq_seven:
  ∃ (a b d : ℝ),
  (1, 1).snd = a * (1:ℝ)^3 + b * (1:ℝ)^2 + d ∧
  (-1, -3).snd = a * (-1:ℝ)^3 + b * (-1:ℝ)^2 + d ∧
  (3 * a * (1:ℝ)^2 + 2 * b * 1 = 3 * a * (-1:ℝ)^2 + 2 * b * -1) ∧
  a^3 + b^2 + d = 7 := 
sorry

end parallel_tangents_a3_plus_b2_plus_d_eq_seven_l82_82324


namespace find_unique_number_l82_82481

def is_three_digit_number (N : ℕ) : Prop := 100 ≤ N ∧ N < 1000

def nonzero_digits (A B C : ℕ) : Prop := A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0

def digits_of_number (N A B C : ℕ) : Prop := N = 100 * A + 10 * B + C

def product (N A B : ℕ) := N * (10 * A + B) * A

def divides (n m : ℕ) := ∃ k, n * k = m

theorem find_unique_number (N A B C : ℕ) (h1 : is_three_digit_number N)
    (h2 : nonzero_digits A B C) (h3 : digits_of_number N A B C)
    (h4 : divides 1000 (product N A B)) : N = 875 :=
sorry

end find_unique_number_l82_82481


namespace count_multiples_200_to_400_l82_82786

def count_multiples_in_range (a b n : ℕ) : ℕ :=
  (b / n) - ((a + n - 1) / n) + 1

theorem count_multiples_200_to_400 :
  count_multiples_in_range 200 400 78 = 3 :=
by
  sorry

end count_multiples_200_to_400_l82_82786


namespace slower_speed_is_correct_l82_82955

/-- 
A person walks at 14 km/hr instead of a slower speed, 
and as a result, he would have walked 20 km more. 
The actual distance travelled by him is 50 km. 
What is the slower speed he usually walks at?
-/
theorem slower_speed_is_correct :
    ∃ x : ℝ, (14 * (50 / 14) - (x * (30 / x))) = 20 ∧ x = 8.4 :=
by
  sorry

end slower_speed_is_correct_l82_82955


namespace man_rate_in_still_water_l82_82039

-- The conditions
def speed_with_stream : ℝ := 20
def speed_against_stream : ℝ := 4

-- The problem rephrased as a Lean statement
theorem man_rate_in_still_water : 
  (speed_with_stream + speed_against_stream) / 2 = 12 := 
by
  sorry

end man_rate_in_still_water_l82_82039


namespace jerry_total_logs_l82_82315

def logs_from_trees (p m w : Nat) : Nat :=
  80 * p + 60 * m + 100 * w

theorem jerry_total_logs :
  logs_from_trees 8 3 4 = 1220 :=
by
  -- Proof here
  sorry

end jerry_total_logs_l82_82315


namespace other_pencil_length_l82_82549

-- Definitions based on the conditions identified in a)
def pencil1_length : Nat := 12
def total_length : Nat := 24

-- Problem: Prove that the length of the other pencil (pencil2) is 12 cubes.
theorem other_pencil_length : total_length - pencil1_length = 12 := by 
  sorry

end other_pencil_length_l82_82549


namespace cosine_double_angle_tangent_l82_82440

theorem cosine_double_angle_tangent (θ : ℝ) (h : Real.tan θ = -1/3) : Real.cos (2 * θ) = 4/5 :=
by
  sorry

end cosine_double_angle_tangent_l82_82440


namespace minimum_stamps_satisfying_congruences_l82_82079

theorem minimum_stamps_satisfying_congruences (n : ℕ) :
  (n % 4 = 3) ∧ (n % 5 = 2) ∧ (n % 7 = 1) → n = 107 :=
by
  sorry

end minimum_stamps_satisfying_congruences_l82_82079


namespace line_equation_correct_l82_82897

-- Definitions for the conditions
def point := ℝ × ℝ
def vector := ℝ × ℝ

-- Given the line has a direction vector and passes through a point
def line_has_direction_vector (l : point → Prop) (v : vector) : Prop :=
  ∀ p₁ p₂ : point, l p₁ → l p₂ → (p₂.1 - p₁.1, p₂.2 - p₁.2) = v

def line_passes_through_point (l : point → Prop) (p : point) : Prop :=
  l p

-- The line equation in point-direction form
def line_equation (x y : ℝ) : Prop :=
  (x - 1) / 2 = y / -3

-- Main statement
theorem line_equation_correct :
  ∃ l : point → Prop, 
    line_has_direction_vector l (2, -3) ∧
    line_passes_through_point l (1, 0) ∧
    ∀ x y, l (x, y) ↔ line_equation x y := 
sorry

end line_equation_correct_l82_82897


namespace extended_ohara_triple_example_l82_82272

theorem extended_ohara_triple_example : 
  (2 * Real.sqrt 49 + Real.sqrt 64 = 22) :=
by
  -- We are stating the conditions and required proof here.
  sorry

end extended_ohara_triple_example_l82_82272


namespace mariela_cards_received_l82_82503

theorem mariela_cards_received (cards_in_hospital : ℕ) (cards_at_home : ℕ) 
  (h1 : cards_in_hospital = 403) (h2 : cards_at_home = 287) : 
  cards_in_hospital + cards_at_home = 690 := 
by 
  sorry

end mariela_cards_received_l82_82503


namespace range_of_x_l82_82837

variable {f : ℝ → ℝ}
variable (hf1 : ∀ x : ℝ, has_deriv_at f (derivative f x) x)
variable (hf2 : ∀ x : ℝ, derivative f x > - f x)

theorem range_of_x (h : f (Real.log 3) = 1/3) : 
  {x : ℝ | f x > 1 / Real.exp x} = Set.Ioi (Real.log 3) := 
by 
  sorry

end range_of_x_l82_82837


namespace total_players_on_ground_l82_82784

theorem total_players_on_ground 
  (cricket_players : ℕ) (hockey_players : ℕ) (football_players : ℕ) (softball_players : ℕ)
  (hcricket : cricket_players = 16) (hhokey : hockey_players = 12) 
  (hfootball : football_players = 18) (hsoftball : softball_players = 13) :
  cricket_players + hockey_players + football_players + softball_players = 59 :=
by
  sorry

end total_players_on_ground_l82_82784


namespace ratio_of_areas_l82_82090

theorem ratio_of_areas (OR : ℝ) (h : OR > 0) :
  let OY := (1 / 3) * OR
  let area_OY := π * OY^2
  let area_OR := π * OR^2
  (area_OY / area_OR) = (1 / 9) :=
by
  -- Definitions
  let OY := (1 / 3) * OR
  let area_OY := π * OY^2
  let area_OR := π * OR^2
  sorry

end ratio_of_areas_l82_82090


namespace factors_180_count_l82_82743

theorem factors_180_count : 
  ∃ (n : ℕ), 180 = 2^2 * 3^2 * 5^1 ∧ n = 18 ∧ 
  ∀ p a b c, 
  180 = p^a * p^b * p^c →
  (a+1) * (b+1) * (c+1) = 18 :=
by {
  sorry
}

end factors_180_count_l82_82743


namespace crease_length_l82_82876

theorem crease_length (AB : ℝ) (h₁ : AB = 15)
  (h₂ : ∀ (area : ℝ) (folded_area : ℝ), folded_area = 0.25 * area) :
  ∃ (DE : ℝ), DE = 0.5 * AB :=
by
  use 7.5 -- DE
  sorry

end crease_length_l82_82876


namespace find_number_of_girls_l82_82663

theorem find_number_of_girls (B G : ℕ) 
  (h1 : B + G = 604) 
  (h2 : 12 * B + 11 * G = 47 * 604 / 4) : 
  G = 151 :=
by
  sorry

end find_number_of_girls_l82_82663


namespace cows_black_more_than_half_l82_82809

theorem cows_black_more_than_half (t b : ℕ) (h1 : t = 18) (h2 : t - 4 = b) : b - t / 2 = 5 :=
by
  sorry

end cows_black_more_than_half_l82_82809


namespace pentomino_reflectional_count_l82_82770

def is_reflectional (p : Pentomino) : Prop := sorry -- Define reflectional symmetry property
def is_rotational (p : Pentomino) : Prop := sorry -- Define rotational symmetry property

theorem pentomino_reflectional_count :
  ∀ (P : Finset Pentomino),
  P.card = 15 →
  (∃ (R : Finset Pentomino), R.card = 2 ∧ (∀ p ∈ R, is_rotational p ∧ ¬ is_reflectional p)) →
  (∃ (S : Finset Pentomino), S.card = 7 ∧ (∀ p ∈ S, is_reflectional p)) :=
by
  sorry -- Proof not required as per instructions

end pentomino_reflectional_count_l82_82770


namespace find_square_divisible_by_four_l82_82273

/-- There exists an x such that x is a square number, x is divisible by four, 
and 39 < x < 80, and that x = 64 is such a number. --/
theorem find_square_divisible_by_four : ∃ (x : ℕ), (∃ (n : ℕ), x = n^2) ∧ (x % 4 = 0) ∧ (39 < x ∧ x < 80) ∧ x = 64 :=
  sorry

end find_square_divisible_by_four_l82_82273


namespace boxes_total_is_correct_l82_82462

def initial_boxes : ℕ := 7
def additional_boxes_per_box : ℕ := 7
def final_non_empty_boxes : ℕ := 10
def total_boxes := 77

theorem boxes_total_is_correct
  (h1 : initial_boxes = 7)
  (h2 : additional_boxes_per_box = 7)
  (h3 : final_non_empty_boxes = 10)
  : total_boxes = 77 :=
by
  -- Proof goes here
  sorry

end boxes_total_is_correct_l82_82462


namespace problem1_monotonic_and_extremum_problem2_range_of_k_problem3_inequality_l82_82925

noncomputable def f (x : ℝ) (k : ℝ) := (Real.log x - k - 1) * x

-- Problem 1: Interval of monotonicity and extremum.
theorem problem1_monotonic_and_extremum (k : ℝ):
  (k ≤ 0 → ∀ x, 1 < x → f x k = (Real.log x - k - 1) * x) ∧
  (k > 0 → (∀ x, 1 < x ∧ x < Real.exp k → f x k = (Real.log x - k - 1) * x) ∧
           (∀ x, Real.exp k < x → f x k = (Real.log x - k - 1) * x) ∧
           f (Real.exp k) k = -Real.exp k) := sorry

-- Problem 2: Range of k.
theorem problem2_range_of_k (k : ℝ):
  (∀ x, Real.exp 1 ≤ x ∧ x ≤ Real.exp 2 → f x k < 4 * Real.log x) ↔
  k > 1 - (8 / Real.exp 2) := sorry

-- Problem 3: Inequality involving product of x1 and x2.
theorem problem3_inequality (x1 x2 : ℝ) (k : ℝ):
  x1 ≠ x2 ∧ f x1 k = f x2 k → x1 * x2 < Real.exp (2 * k) := sorry

end problem1_monotonic_and_extremum_problem2_range_of_k_problem3_inequality_l82_82925


namespace find_m_l82_82209

open Classical

variable {d : ℤ} (h₁ : d ≠ 0) (a : ℕ → ℤ)

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) :=
  ∃ a₀ : ℤ, ∀ n, a n = a₀ + n * d

theorem find_m 
  (h_seq : arithmetic_sequence a d)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_am : ∃ m, a m = 8) :
  ∃ m, m = 8 :=
sorry

end find_m_l82_82209


namespace find_K_values_l82_82160

theorem find_K_values (K M : ℕ) (h1 : (K * (K + 1)) / 2 = M^2) (h2 : M < 200) (h3 : K > M) :
  K = 8 ∨ K = 49 :=
sorry

end find_K_values_l82_82160


namespace sergio_has_6_more_correct_answers_l82_82972

-- Define conditions
def total_questions : ℕ := 50
def incorrect_answers_sylvia : ℕ := total_questions / 5
def incorrect_answers_sergio : ℕ := 4

-- Calculate correct answers
def correct_answers_sylvia : ℕ := total_questions - incorrect_answers_sylvia
def correct_answers_sergio : ℕ := total_questions - incorrect_answers_sergio

-- The proof problem
theorem sergio_has_6_more_correct_answers :
  correct_answers_sergio - correct_answers_sylvia = 6 :=
by
  sorry

end sergio_has_6_more_correct_answers_l82_82972


namespace min_side_b_of_triangle_l82_82331

theorem min_side_b_of_triangle (A B C a b c : ℝ) 
  (h_arith_seq : 2 * B = A + C)
  (h_sum_angles : A + B + C = Real.pi)
  (h_sides_opposite : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B)
  (h_given_eq : 3 * a * c + b^2 = 25) :
  b ≥ 5 / 2 :=
  sorry

end min_side_b_of_triangle_l82_82331


namespace range_of_a_l82_82265

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 2| - |x - 3| ≤ a) → a ≥ -5 := by
  sorry

end range_of_a_l82_82265


namespace f_f_4_eq_1_l82_82375

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 x

theorem f_f_4_eq_1 : f (f 4) = 1 := by
  sorry

end f_f_4_eq_1_l82_82375


namespace smallest_possible_b_l82_82362

-- Definitions of conditions
variables {a b c : ℤ}

-- Conditions expressed in Lean
def is_geometric_progression (a b c : ℤ) : Prop := b^2 = a * c
def is_arithmetic_progression (a b c : ℤ) : Prop := a + b = 2 * c

-- The theorem statement
theorem smallest_possible_b (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c) 
  (hg : is_geometric_progression a b c) 
  (ha : is_arithmetic_progression a c b) : b = 2 := sorry

end smallest_possible_b_l82_82362


namespace points_and_conditions_proof_l82_82508

noncomputable def points_and_conditions (x y : ℝ) : Prop := 
|x - 3| + |y + 5| = 0

noncomputable def min_AM_BM (m : ℝ) : Prop :=
|3 - m| + |-5 - m| = 7 / 4 * |8|

noncomputable def min_PA_PB (p : ℝ) : Prop :=
|p - 3| + |p + 5| = 8

noncomputable def min_PD_PO (p : ℝ) : Prop :=
|p + 1| - |p| = -1

noncomputable def range_of_a (a : ℝ) : Prop :=
a ∈ Set.Icc (-5) (-1)

theorem points_and_conditions_proof (x y : ℝ) (m p a : ℝ) :
  points_and_conditions x y → 
  x = 3 ∧ y = -5 ∧ 
  ((m = -8 ∨ m = 6) → min_AM_BM m) ∧ 
  (min_PA_PB p) ∧ 
  (min_PD_PO p) ∧ 
  (range_of_a a) :=
by 
  sorry

end points_and_conditions_proof_l82_82508


namespace Jimin_addition_l82_82488

theorem Jimin_addition (x : ℕ) (h : 96 / x = 6) : 34 + x = 50 := 
by
  sorry

end Jimin_addition_l82_82488


namespace pq_sufficient_but_not_necessary_condition_l82_82854

theorem pq_sufficient_but_not_necessary_condition (p q : Prop) (hpq : p ∧ q) :
  ¬¬p = p :=
by
  sorry

end pq_sufficient_but_not_necessary_condition_l82_82854


namespace inscribed_sphere_radius_l82_82813

variable (a b r : ℝ)

theorem inscribed_sphere_radius (ha : 0 < a) (hb : 0 < b) (hr : 0 < r)
 (h : ∃ A B C D : ℝˣ, true) : r < (a * b) / (2 * (a + b)) := 
sorry

end inscribed_sphere_radius_l82_82813


namespace intersection_correct_l82_82339

noncomputable def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def intersection_M_N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_correct : M ∩ N = intersection_M_N :=
by
  sorry

end intersection_correct_l82_82339


namespace cost_of_fencing_is_289_l82_82022

def side_lengths : List ℕ := [10, 20, 15, 18, 12, 22]

def cost_per_meter : List ℚ := [3, 2, 4, 3.5, 2.5, 3]

def cost_of_side (length : ℕ) (rate : ℚ) : ℚ :=
  (length : ℚ) * rate

def total_cost : ℚ :=
  List.zipWith cost_of_side side_lengths cost_per_meter |>.sum

theorem cost_of_fencing_is_289 : total_cost = 289 := by
  sorry

end cost_of_fencing_is_289_l82_82022


namespace sequence_non_existence_l82_82841

variable (α β : ℝ)
variable (r : ℝ)

theorem sequence_non_existence 
  (hαβ : α * β > 0) :  
  (∃ (x : ℕ → ℝ), x 0 = r ∧ ∀ n, x (n + 1) = (x n + α) / (β * (x n) + 1) → false) ↔ 
  r = - (1 / β) :=
sorry

end sequence_non_existence_l82_82841


namespace multiply_decimals_l82_82861

noncomputable def real_num_0_7 : ℝ := 7 * 10⁻¹
noncomputable def real_num_0_3 : ℝ := 3 * 10⁻¹
noncomputable def real_num_0_21 : ℝ := 0.21

theorem multiply_decimals :
  real_num_0_7 * real_num_0_3 = real_num_0_21 :=
sorry

end multiply_decimals_l82_82861


namespace log_sum_eq_two_l82_82792

theorem log_sum_eq_two : 
  ∀ (lg : ℝ → ℝ),
  (∀ x y : ℝ, lg (x * y) = lg x + lg y) →
  (∀ x y : ℝ, lg (x ^ y) = y * lg x) →
  lg 4 + 2 * lg 5 = 2 :=
by
  intros lg h1 h2
  sorry

end log_sum_eq_two_l82_82792


namespace leak_rate_l82_82347

-- Definitions based on conditions
def initialWater : ℕ := 10   -- 10 cups
def finalWater : ℕ := 2      -- 2 cups
def firstThreeMilesWater : ℕ := 3 * 1    -- 1 cup per mile for first 3 miles
def lastMileWater : ℕ := 3               -- 3 cups during the last mile
def hikeDuration : ℕ := 2    -- 2 hours

-- Proving the leak rate
theorem leak_rate (drunkWater : ℕ) (leakedWater : ℕ) (leakRate : ℕ) :
  drunkWater = firstThreeMilesWater + lastMileWater ∧ 
  (initialWater - finalWater) = (drunkWater + leakedWater) ∧
  hikeDuration = 2 ∧ 
  leakRate = leakedWater / hikeDuration → leakRate = 1 :=
by
  intros h
  sorry

end leak_rate_l82_82347


namespace find_g3_l82_82818

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = 1

theorem find_g3 : g 3 = 0 := by
  sorry

end find_g3_l82_82818


namespace interest_rate_calculation_l82_82936

theorem interest_rate_calculation
  (SI : ℕ) (P : ℕ) (T : ℕ) (R : ℕ)
  (h1 : SI = 2100) (h2 : P = 875) (h3 : T = 20) :
  (SI * 100 = P * R * T) → R = 12 :=
by
  sorry

end interest_rate_calculation_l82_82936


namespace students_per_bus_l82_82706

theorem students_per_bus
  (total_students : ℕ)
  (buses : ℕ)
  (students_in_cars : ℕ)
  (h1 : total_students = 375)
  (h2 : buses = 7)
  (h3 : students_in_cars = 4) :
  (total_students - students_in_cars) / buses = 53 :=
by
  sorry

end students_per_bus_l82_82706


namespace find_cows_l82_82163

variable (D C : ℕ)

theorem find_cows (h1 : 2 * D + 4 * C = 2 * (D + C) + 36) : C = 18 :=
by
  -- Proof goes here
  sorry

end find_cows_l82_82163


namespace lisa_interest_correct_l82_82213

noncomputable def lisa_interest : ℝ :=
  let P := 2000
  let r := 0.035
  let n := 10
  let A := P * (1 + r) ^ n
  A - P

theorem lisa_interest_correct :
  lisa_interest = 821 := by
  sorry

end lisa_interest_correct_l82_82213


namespace problem_l82_82276

variable (a b : ℝ)

theorem problem (h₁ : a + b = 2) (h₂ : a - b = 3) : a^2 - b^2 = 6 := by
  sorry

end problem_l82_82276


namespace meal_combinations_l82_82245

def number_of_menu_items : ℕ := 15

theorem meal_combinations (different_orderings : ∀ Yann Camille : ℕ, Yann ≠ Camille → Yann ≤ number_of_menu_items ∧ Camille ≤ number_of_menu_items) : 
  (number_of_menu_items * (number_of_menu_items - 1)) = 210 :=
by sorry

end meal_combinations_l82_82245


namespace some_number_value_l82_82141

theorem some_number_value (a : ℕ) (x : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * x * 49) : x = 9 := by
  sorry

end some_number_value_l82_82141


namespace mary_final_weight_l82_82214

theorem mary_final_weight : 
  let initial_weight := 99
  let weight_loss1 := 12
  let weight_gain1 := 2 * weight_loss1
  let weight_loss2 := 3 * weight_loss1
  let weight_gain2 := 6
  initial_weight - weight_loss1 + weight_gain1 - weight_loss2 + weight_gain2 = 81 := by 
  sorry

end mary_final_weight_l82_82214


namespace compare_y_values_l82_82124

noncomputable def parabola (x : ℝ) : ℝ := -2 * (x + 1) ^ 2 - 1

theorem compare_y_values :
  ∃ y1 y2 y3, (parabola (-3) = y1) ∧ (parabola (-2) = y2) ∧ (parabola 2 = y3) ∧ (y3 < y1) ∧ (y1 < y2) :=
by
  sorry

end compare_y_values_l82_82124


namespace ages_of_residents_l82_82310

theorem ages_of_residents (a b c : ℕ)
  (h1 : a * b * c = 1296)
  (h2 : a + b + c = 91)
  (h3 : ∀ x y z : ℕ, x * y * z = 1296 → x + y + z = 91 → (x < 80 ∧ y < 80 ∧ z < 80) → (x = 1 ∧ y = 18 ∧ z = 72)) :
  (a = 1 ∧ b = 18 ∧ c = 72 ∨ a = 1 ∧ b = 72 ∧ c = 18 ∨ a = 18 ∧ b = 1 ∧ c = 72 ∨ a = 18 ∧ b = 72 ∧ c = 1 ∨ a = 72 ∧ b = 1 ∧ c = 18 ∨ a = 72 ∧ b = 18 ∧ c = 1) :=
by
  sorry

end ages_of_residents_l82_82310


namespace task_completion_time_l82_82628

theorem task_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ t : ℝ, t = (a * b) / (a + b) := 
sorry

end task_completion_time_l82_82628


namespace value_of_adams_collection_l82_82281

theorem value_of_adams_collection (num_coins : ℕ) (coins_value : ℕ) (total_value_4coins : ℕ) (h1 : num_coins = 20) (h2 : total_value_4coins = 16) (h3 : ∀ k, k = 4 → coins_value = total_value_4coins / k) : 
  num_coins * coins_value = 80 := 
by {
  sorry
}

end value_of_adams_collection_l82_82281


namespace compute_xy_l82_82061

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 198) : xy = 5 :=
by
  sorry

end compute_xy_l82_82061


namespace other_asymptote_of_hyperbola_l82_82648

theorem other_asymptote_of_hyperbola (a b : ℝ) : 
  (∀ x : ℝ, y = 2 * x + 3) → 
  (∃ y : ℝ, x = -4) → 
  (∀ x : ℝ, y = - (1 / 2) * x - 7) := 
by {
  -- The proof will go here
  sorry
}

end other_asymptote_of_hyperbola_l82_82648


namespace problem1_problem2_l82_82918

-- Define the first problem
theorem problem1 : (Real.cos (25 / 3 * Real.pi) + Real.tan (-15 / 4 * Real.pi)) = 3 / 2 :=
by
  sorry

-- Define vector operations and the problem
variables (a b : ℝ)

theorem problem2 : 2 * (a - b) - (2 * a + b) + 3 * b = 0 :=
by
  sorry

end problem1_problem2_l82_82918


namespace planes_touch_three_spheres_count_l82_82946

-- Declare the conditions as definitions
def square_side_length : ℝ := 10
def radii : Fin 4 → ℝ
| 0 => 1
| 1 => 2
| 2 => 4
| 3 => 3

-- The proof problem statement
theorem planes_touch_three_spheres_count :
    ∃ (planes_that_touch_three_spheres : ℕ) (planes_that_intersect_fourth_sphere : ℕ),
    planes_that_touch_three_spheres = 26 ∧ planes_that_intersect_fourth_sphere = 8 := 
by
  -- sorry skips the proof
  sorry

end planes_touch_three_spheres_count_l82_82946


namespace slower_speed_l82_82316

theorem slower_speed (f e d : ℕ) (h1 : f = 14) (h2 : e = 20) (h3 : d = 50) (x : ℕ) : 
  (50 / x : ℚ) = (50 / 14 : ℚ) + (20 / 14 : ℚ) → x = 10 := by
  sorry

end slower_speed_l82_82316


namespace cat_daytime_catches_l82_82094

theorem cat_daytime_catches
  (D : ℕ)
  (night_catches : ℕ := 2 * D)
  (total_catches : ℕ := D + night_catches)
  (h : total_catches = 24) :
  D = 8 := by
  sorry

end cat_daytime_catches_l82_82094


namespace quadratic_inequality_solution_l82_82204

theorem quadratic_inequality_solution:
  ∀ x : ℝ, -x^2 + 3 * x - 2 ≥ 0 ↔ (1 ≤ x ∧ x ≤ 2) :=
by
  sorry

end quadratic_inequality_solution_l82_82204


namespace range_of_a_l82_82729

theorem range_of_a (a : ℝ) : 
  ¬(∀ x : ℝ, x^2 - 5 * x + 15 / 2 * a <= 0) -> a > 5 / 6 :=
by
  sorry

end range_of_a_l82_82729


namespace totalCost_l82_82185
-- Importing the necessary library

-- Defining the conditions
def numberOfHotDogs : Nat := 6
def costPerHotDog : Nat := 50

-- Proving the total cost
theorem totalCost : numberOfHotDogs * costPerHotDog = 300 := by
  sorry

end totalCost_l82_82185


namespace largest_multiple_of_11_neg_greater_minus_210_l82_82676

theorem largest_multiple_of_11_neg_greater_minus_210 :
  ∃ (x : ℤ), x % 11 = 0 ∧ -x < -210 ∧ ∀ y, y % 11 = 0 ∧ -y < -210 → y ≤ x :=
sorry

end largest_multiple_of_11_neg_greater_minus_210_l82_82676


namespace ratio_kid_to_adult_ticket_l82_82296

theorem ratio_kid_to_adult_ticket (A : ℝ) : 
  (6 * 5 + 2 * A = 50) → (5 / A = 1 / 2) :=
by
  sorry

end ratio_kid_to_adult_ticket_l82_82296


namespace mixed_number_calculation_l82_82670

theorem mixed_number_calculation :
  47 * (4 + 3/7 - (5 + 1/3)) / (3 + 1/2 + (2 + 1/5)) = -7 - 119/171 := by
  sorry

end mixed_number_calculation_l82_82670


namespace total_students_is_2000_l82_82395

theorem total_students_is_2000
  (S : ℝ) 
  (h1 : 0.10 * S = chess_students) 
  (h2 : 0.50 * chess_students = swimming_students) 
  (h3 : swimming_students = 100) 
  (chess_students swimming_students : ℝ) 
  : S = 2000 := 
by 
  sorry

end total_students_is_2000_l82_82395


namespace chameleons_impossible_all_white_l82_82095

/--
On Easter Island, there are initial counts of blue (12), white (25), and red (8) chameleons.
When two chameleons of different colors meet, they both change to the third color.
Prove that it is impossible for all chameleons to become white.
--/
theorem chameleons_impossible_all_white :
  let n1 := 12 -- Blue chameleons
  let n2 := 25 -- White chameleons
  let n3 := 8  -- Red chameleons
  (∀ (n1 n2 n3 : ℕ), (n1 + n2 + n3 = 45) → 
   ∀ (k : ℕ), ∃ m1 m2 m3 : ℕ, (m1 - m2) % 3 = (n1 - n2) % 3 ∧ (m1 - m3) % 3 = (n1 - n3) % 3 ∧ 
   (m2 - m3) % 3 = (n2 - n3) % 3) → False := sorry

end chameleons_impossible_all_white_l82_82095


namespace parabola_translation_l82_82523

theorem parabola_translation :
  ∀(x y : ℝ), y = - (1 / 3) * (x - 5) ^ 2 + 3 →
  ∃(x' y' : ℝ), y' = -(1/3) * x'^2 + 6 := by
  sorry

end parabola_translation_l82_82523


namespace total_work_completed_in_days_l82_82586

-- Define the number of days Amit can complete the work
def amit_days : ℕ := 15

-- Define the number of days Ananthu can complete the work
def ananthu_days : ℕ := 90

-- Define the number of days Amit worked
def amit_work_days : ℕ := 3

-- Calculate the amount of work Amit can do in one day
def amit_work_day_rate : ℚ := 1 / amit_days

-- Calculate the amount of work Ananthu can do in one day
def ananthu_work_day_rate : ℚ := 1 / ananthu_days

-- Calculate the total work completed
theorem total_work_completed_in_days :
  amit_work_days * amit_work_day_rate + (1 - amit_work_days * amit_work_day_rate) / ananthu_work_day_rate = 75 :=
by
  -- Placeholder for the proof
  sorry

end total_work_completed_in_days_l82_82586


namespace solve_x_plus_Sx_eq_2001_l82_82942

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem solve_x_plus_Sx_eq_2001 (x : ℕ) (h : x + sum_of_digits x = 2001) : x = 1977 :=
  sorry

end solve_x_plus_Sx_eq_2001_l82_82942


namespace find_x_l82_82229

theorem find_x (x y : ℤ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y + x * y = 119) : x = 39 :=
sorry

end find_x_l82_82229


namespace gingerbreads_per_tray_l82_82017

-- Given conditions
def total_baked_gb (x : ℕ) : Prop := 4 * 25 + 3 * x = 160

-- The problem statement
theorem gingerbreads_per_tray (x : ℕ) (h : total_baked_gb x) : x = 20 := 
by sorry

end gingerbreads_per_tray_l82_82017


namespace competition_score_l82_82550

theorem competition_score (x : ℕ) (h : x ≥ 15) : 10 * x - 5 * (20 - x) > 120 := by
  sorry

end competition_score_l82_82550


namespace g_value_at_neg3_l82_82981

noncomputable def g : ℚ → ℚ := sorry

theorem g_value_at_neg3 (h : ∀ x : ℚ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 2 * x^2) : 
  g (-3) = 98 / 13 := 
sorry

end g_value_at_neg3_l82_82981


namespace find_b_num_days_worked_l82_82264

noncomputable def a_num_days_worked := 6
noncomputable def b_num_days_worked := 9  -- This is what we want to verify
noncomputable def c_num_days_worked := 4

noncomputable def c_daily_wage := 105
noncomputable def wage_ratio_a := 3
noncomputable def wage_ratio_b := 4
noncomputable def wage_ratio_c := 5

-- Helper to find daily wages for a and b given the ratio and c's wage
noncomputable def x := c_daily_wage / wage_ratio_c
noncomputable def a_daily_wage := wage_ratio_a * x
noncomputable def b_daily_wage := wage_ratio_b * x

-- Calculate total earnings
noncomputable def a_total_earning := a_num_days_worked * a_daily_wage
noncomputable def c_total_earning := c_num_days_worked * c_daily_wage
noncomputable def total_earning := 1554
noncomputable def b_total_earning := b_num_days_worked * b_daily_wage

theorem find_b_num_days_worked : total_earning = a_total_earning + b_total_earning + c_total_earning → b_num_days_worked = 9 := by
  sorry

end find_b_num_days_worked_l82_82264


namespace range_of_m_l82_82939

def point_P := (1, 1)
def circle_C1 (x y m : ℝ) := x^2 + y^2 + 2*x - m = 0

theorem range_of_m (m : ℝ) :
  (1 + 1)^2 + 1^2 > m + 1 → -1 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l82_82939


namespace dog_weights_l82_82294

structure DogWeightProgression where
  initial: ℕ   -- initial weight in pounds
  week_9: ℕ    -- weight at 9 weeks in pounds
  month_3: ℕ  -- weight at 3 months in pounds
  month_5: ℕ  -- weight at 5 months in pounds
  year_1: ℕ   -- weight at 1 year in pounds

theorem dog_weights :
  ∃ (golden_retriever labrador poodle : DogWeightProgression),
  golden_retriever.initial = 6 ∧
  golden_retriever.week_9 = 12 ∧
  golden_retriever.month_3 = 24 ∧
  golden_retriever.month_5 = 48 ∧
  golden_retriever.year_1 = 78 ∧
  labrador.initial = 8 ∧
  labrador.week_9 = 24 ∧
  labrador.month_3 = 36 ∧
  labrador.month_5 = 72 ∧
  labrador.year_1 = 102 ∧
  poodle.initial = 4 ∧
  poodle.week_9 = 16 ∧
  poodle.month_3 = 32 ∧
  poodle.month_5 = 32 ∧
  poodle.year_1 = 52 :=
by 
  have golden_retriever : DogWeightProgression := { initial := 6, week_9 := 12, month_3 := 24, month_5 := 48, year_1 := 78 }
  have labrador : DogWeightProgression := { initial := 8, week_9 := 24, month_3 := 36, month_5 := 72, year_1 := 102 }
  have poodle : DogWeightProgression := { initial := 4, week_9 := 16, month_3 := 32, month_5 := 32, year_1 := 52 }
  use golden_retriever, labrador, poodle
  repeat { split };
  { sorry }

end dog_weights_l82_82294


namespace a_10_is_100_l82_82390

-- Define the sequence a_n as a function from ℕ+ (the positive naturals) to ℤ
axiom a : ℕ+ → ℤ

-- Given assumptions
axiom seq_relation : ∀ m n : ℕ+, a m + a n = a (m + n) - 2 * m.val * n.val
axiom a1 : a 1 = 1

-- Goal statement
theorem a_10_is_100 : a 10 = 100 :=
by
  -- proof goes here, this is just the statement
  sorry

end a_10_is_100_l82_82390


namespace tom_initial_balloons_l82_82502

noncomputable def initial_balloons (x : ℕ) : ℕ :=
  if h₁ : x % 2 = 1 ∧ (x / 3) + 10 = 45 then x else 0

theorem tom_initial_balloons : initial_balloons 105 = 105 :=
by {
  -- Given x is an odd number and the equation (x / 3) + 10 = 45 holds, prove x = 105.
  -- These conditions follow from the problem statement directly.
  -- Proof is skipped.
  sorry
}

end tom_initial_balloons_l82_82502


namespace stratified_sampling_second_class_l82_82398

theorem stratified_sampling_second_class (total_products : ℕ) (first_class : ℕ) (second_class : ℕ) (third_class : ℕ) (sample_size : ℕ) (h_total : total_products = 200) (h_first : first_class = 40) (h_second : second_class = 60) (h_third : third_class = 100) (h_sample : sample_size = 40) :
  (second_class * sample_size) / total_products = 12 :=
by
  sorry

end stratified_sampling_second_class_l82_82398


namespace deepak_present_age_l82_82107

-- We start with the conditions translated into Lean definitions.

variables (R D : ℕ)

-- Condition 1: The ratio between Rahul's and Deepak's ages is 4:3.
def age_ratio := R * 3 = D * 4

-- Condition 2: After 6 years, Rahul's age will be 38 years.
def rahul_future_age := R + 6 = 38

-- The goal is to prove that D = 24 given the above conditions.
theorem deepak_present_age 
  (h1: age_ratio R D) 
  (h2: rahul_future_age R) : D = 24 :=
sorry

end deepak_present_age_l82_82107


namespace find_n_expansion_l82_82833

theorem find_n_expansion : 
  (∃ n : ℕ, 4^n + 2^n = 1056) → n = 5 :=
by sorry

end find_n_expansion_l82_82833


namespace computer_table_cost_price_l82_82919

theorem computer_table_cost_price (CP SP : ℝ) (h1 : SP = CP * (124 / 100)) (h2 : SP = 8091) :
  CP = 6525 :=
by
  sorry

end computer_table_cost_price_l82_82919


namespace complex_expression_evaluation_l82_82023

theorem complex_expression_evaluation (i : ℂ) (h1 : i^(4 : ℤ) = 1) (h2 : i^(1 : ℤ) = i)
   (h3 : i^(2 : ℤ) = -1) (h4 : i^(3 : ℤ) = -i) (h5 : i^(0 : ℤ) = 1) : 
   i^(245 : ℤ) + i^(246 : ℤ) + i^(247 : ℤ) + i^(248 : ℤ) + i^(249 : ℤ) = i :=
by
  sorry

end complex_expression_evaluation_l82_82023


namespace roses_in_february_l82_82489

-- Define initial counts of roses
def roses_oct : ℕ := 80
def roses_nov : ℕ := 98
def roses_dec : ℕ := 128
def roses_jan : ℕ := 170

-- Define the differences
def diff_on : ℕ := roses_nov - roses_oct -- 18
def diff_nd : ℕ := roses_dec - roses_nov -- 30
def diff_dj : ℕ := roses_jan - roses_dec -- 42

-- The increment in differences
def inc : ℕ := diff_nd - diff_on -- 12

-- Express the difference from January to February
def diff_jf : ℕ := diff_dj + inc -- 54

-- The number of roses in February
def roses_feb : ℕ := roses_jan + diff_jf -- 224

theorem roses_in_february : roses_feb = 224 := by
  -- Provide the expected value for Lean to verify
  sorry

end roses_in_february_l82_82489


namespace vectors_collinear_has_solution_l82_82280

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (x^2 - 1, 2 + x)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Collinearity condition (cross product must be zero) as a function
def collinear (x : ℝ) : Prop := (a x).1 * (b x).2 - (b x).1 * (a x).2 = 0

-- The proof statement
theorem vectors_collinear_has_solution (x : ℝ) (h : collinear x) : x = -1 / 2 :=
sorry

end vectors_collinear_has_solution_l82_82280


namespace relationship_among_m_n_k_l82_82267

theorem relationship_among_m_n_k :
  (¬ ∃ x : ℝ, |2 * x - 3| + m = 0) → 
  (∃! x: ℝ, |3 * x - 4| + n = 0) → 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |4 * x₁ - 5| + k = 0 ∧ |4 * x₂ - 5| + k = 0) →
  (m > n ∧ n > k) :=
by
  intros h1 h2 h3
  -- Proof part will be added here
  sorry

end relationship_among_m_n_k_l82_82267


namespace largest_integral_value_of_y_l82_82699

theorem largest_integral_value_of_y : 
  (1 / 4 : ℝ) < (y / 7 : ℝ) ∧ (y / 7 : ℝ) < (3 / 5 : ℝ) → y ≤ 4 :=
by
  sorry

end largest_integral_value_of_y_l82_82699


namespace prime_pair_perfect_square_l82_82767

theorem prime_pair_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : ∃ a : ℕ, p^2 + p * q + q^2 = a^2) : (p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3) := 
sorry

end prime_pair_perfect_square_l82_82767


namespace micheal_work_separately_40_days_l82_82137

-- Definitions based on the problem conditions
def work_complete_together (M A : ℕ) : Prop := (1/(M:ℝ) + 1/(A:ℝ) = 1/20)
def remaining_work_completed_by_adam (A : ℕ) : Prop := (1/(A:ℝ) = 1/40)

-- The theorem we want to prove
theorem micheal_work_separately_40_days (M A : ℕ) 
  (h1 : work_complete_together M A) 
  (h2 : remaining_work_completed_by_adam A) : 
  M = 40 := 
by 
  sorry  -- Placeholder for proof

end micheal_work_separately_40_days_l82_82137


namespace coefficient_of_x3y7_in_expansion_l82_82380

-- Definitions based on the conditions in the problem
def a : ℚ := (2 / 3)
def b : ℚ := - (3 / 4)
def n : ℕ := 10
def k1 : ℕ := 3
def k2 : ℕ := 7

-- Statement of the math proof problem
theorem coefficient_of_x3y7_in_expansion :
  (a * x ^ k1 + b * y ^ k2) ^ n = x3y7_coeff * x ^ k1 * y ^ k2  :=
sorry

end coefficient_of_x3y7_in_expansion_l82_82380


namespace quadratic_two_distinct_real_roots_l82_82040

theorem quadratic_two_distinct_real_roots (k : ℝ) (h1 : k ≠ 0) : 
  (∀ Δ > 0, Δ = (-2)^2 - 4 * k * (-1)) ↔ (k > -1) :=
by
  -- Since Δ = 4 + 4k, we need to show that (4 + 4k > 0) ↔ (k > -1)
  sorry

end quadratic_two_distinct_real_roots_l82_82040


namespace man_l82_82568

theorem man's_rowing_speed_in_still_water
  (river_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (H_river_speed : river_speed = 2)
  (H_total_time : total_time = 1)
  (H_total_distance : total_distance = 5.333333333333333) :
  ∃ (v : ℝ), 
    v = 7.333333333333333 ∧
    ∀ d,
    d = total_distance / 2 →
    d = (v - river_speed) * (total_time / 2) ∧
    d = (v + river_speed) * (total_time / 2) := 
by
  sorry

end man_l82_82568


namespace optimal_cylinder_dimensions_l82_82004

variable (R : ℝ)

noncomputable def optimal_cylinder_height : ℝ := (2 * R) / Real.sqrt 3
noncomputable def optimal_cylinder_radius : ℝ := R * Real.sqrt (2 / 3)

theorem optimal_cylinder_dimensions :
  ∃ (h r : ℝ), 
    (h = optimal_cylinder_height R ∧ r = optimal_cylinder_radius R) ∧
    ∀ (h' r' : ℝ), (4 * R^2 = 4 * r'^2 + h'^2) → 
      (h' = optimal_cylinder_height R ∧ r' = optimal_cylinder_radius R) → 
      (π * r' ^ 2 * h' ≤ π * r ^ 2 * h) :=
by
  -- Proof omitted
  sorry

end optimal_cylinder_dimensions_l82_82004


namespace smallest_four_digit_divisible_by_33_l82_82754

theorem smallest_four_digit_divisible_by_33 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 33 = 0 ∧ n = 1023 := by 
  sorry

end smallest_four_digit_divisible_by_33_l82_82754


namespace line_AB_eq_x_plus_3y_zero_l82_82873

variable (x y : ℝ)

def circle1 := x^2 + y^2 - 4*x + 6*y = 0
def circle2 := x^2 + y^2 - 6*x = 0

theorem line_AB_eq_x_plus_3y_zero : 
  (∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧ (A ≠ B)) → 
  (∀ (x y : ℝ), x + 3*y = 0) := 
by
  sorry

end line_AB_eq_x_plus_3y_zero_l82_82873


namespace integer_diff_of_two_squares_l82_82760

theorem integer_diff_of_two_squares (m : ℤ) : 
  (∃ x y : ℤ, m = x^2 - y^2) ↔ (∃ k : ℤ, m ≠ 4 * k + 2) := by
  sorry

end integer_diff_of_two_squares_l82_82760


namespace paper_thickness_after_2_folds_l82_82679

theorem paper_thickness_after_2_folds:
  ∀ (initial_thickness : ℝ) (folds : ℕ),
  initial_thickness = 0.1 →
  folds = 2 →
  (initial_thickness * 2^folds = 0.4) :=
by
  intros initial_thickness folds h_initial h_folds
  sorry

end paper_thickness_after_2_folds_l82_82679


namespace find_x_parallel_find_x_perpendicular_l82_82920

def a (x : ℝ) : ℝ × ℝ := (x, x + 2)
def b : ℝ × ℝ := (1, 2)

-- Given that a vector is proportional to another
def are_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Given that the dot product is zero
def are_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_x_parallel (x : ℝ) (h : are_parallel (a x) b) : x = 2 :=
by sorry

theorem find_x_perpendicular (x : ℝ) (h : are_perpendicular (a x - b) b) : x = (1 / 3 : ℝ) :=
by sorry

end find_x_parallel_find_x_perpendicular_l82_82920


namespace sum_factors_30_less_15_l82_82455

theorem sum_factors_30_less_15 : (1 + 2 + 3 + 5 + 6 + 10) = 27 := by
  sorry

end sum_factors_30_less_15_l82_82455


namespace balance_of_three_squares_and_two_heartsuits_l82_82723

-- Definitions
variable {x y z w : ℝ}

-- Given conditions
axiom h1 : 3 * x + 4 * y + z = 12 * w
axiom h2 : x = z + 2 * w

-- Problem to prove
theorem balance_of_three_squares_and_two_heartsuits :
  (3 * y + 2 * z) = (26 / 9) * w :=
sorry

end balance_of_three_squares_and_two_heartsuits_l82_82723


namespace rate_of_interest_first_year_l82_82666

-- Define the conditions
def principal : ℝ := 9000
def rate_second_year : ℝ := 0.05
def total_amount_after_2_years : ℝ := 9828

-- Define the problem statement which we need to prove
theorem rate_of_interest_first_year (R : ℝ) :
  (principal + (principal * R / 100)) + 
  ((principal + (principal * R / 100)) * rate_second_year) = 
  total_amount_after_2_years → 
  R = 4 := 
by
  sorry

end rate_of_interest_first_year_l82_82666


namespace exists_unique_subset_X_l82_82404

theorem exists_unique_subset_X :
  ∃ (X : Set ℤ), ∀ n : ℤ, ∃! (a b : ℤ), a ∈ X ∧ b ∈ X ∧ a + 2 * b = n :=
sorry

end exists_unique_subset_X_l82_82404


namespace perfect_square_trinomial_k_l82_82119

theorem perfect_square_trinomial_k (a k : ℝ) : (∃ b : ℝ, (a - b)^2 = a^2 - ka + 25) ↔ k = 10 ∨ k = -10 := 
sorry

end perfect_square_trinomial_k_l82_82119


namespace oranges_packed_in_a_week_l82_82872

open Nat

def oranges_per_box : Nat := 15
def boxes_per_day : Nat := 2150
def days_per_week : Nat := 7

theorem oranges_packed_in_a_week : oranges_per_box * boxes_per_day * days_per_week = 225750 :=
  sorry

end oranges_packed_in_a_week_l82_82872


namespace tan_alpha_minus_pi_over_4_eq_neg_3_l82_82935

theorem tan_alpha_minus_pi_over_4_eq_neg_3
  (α : ℝ)
  (h1 : True) -- condition to ensure we define α in ℝ, "True" is just a dummy
  (a : ℝ × ℝ := (Real.cos α, -2))
  (b : ℝ × ℝ := (Real.sin α, 1))
  (h2 : ∃ k : ℝ, a = k • b) : 
  Real.tan (α - Real.pi / 4) = -3 :=
  sorry

end tan_alpha_minus_pi_over_4_eq_neg_3_l82_82935


namespace smallest_five_digit_int_equiv_mod_l82_82642

theorem smallest_five_digit_int_equiv_mod (n : ℕ) (h1 : 10000 ≤ n) (h2 : n % 9 = 4) : n = 10003 := 
sorry

end smallest_five_digit_int_equiv_mod_l82_82642


namespace jaysons_moms_age_l82_82776

theorem jaysons_moms_age (jayson's_age dad's_age mom's_age : ℕ) 
  (h1 : jayson's_age = 10)
  (h2 : dad's_age = 4 * jayson's_age)
  (h3 : mom's_age = dad's_age - 2) :
  mom's_age - jayson's_age = 28 := 
by
  sorry

end jaysons_moms_age_l82_82776


namespace sara_gave_dan_limes_l82_82732

theorem sara_gave_dan_limes (initial_limes : ℕ) (final_limes : ℕ) (d : ℕ) 
  (h1: initial_limes = 9) (h2: final_limes = 13) (h3: final_limes = initial_limes + d) : d = 4 := 
by sorry

end sara_gave_dan_limes_l82_82732


namespace total_time_on_road_l82_82827

def driving_time_day1 (jade_time krista_time krista_delay lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + krista_delay + lunch_break

def driving_time_day2 (jade_time krista_time break_time krista_refuel lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + break_time + krista_refuel + lunch_break

def driving_time_day3 (jade_time krista_time krista_delay lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + krista_delay + lunch_break

def total_driving_time (day1 day2 day3 : ℝ) : ℝ :=
  day1 + day2 + day3

theorem total_time_on_road :
  total_driving_time 
    (driving_time_day1 8 6 1 1) 
    (driving_time_day2 7 5 0.5 (1/3) 1) 
    (driving_time_day3 6 4 1 1) 
  = 42.3333 := 
  by 
    sorry

end total_time_on_road_l82_82827


namespace greatest_partition_l82_82417

-- Define the condition on the partitions of the positive integers
def satisfies_condition (A : ℕ → Prop) (n : ℕ) : Prop :=
∃ a b : ℕ, a ≠ b ∧ A a ∧ A b ∧ a + b = n

-- Define what it means for k subsets to meet the requirements
def partition_satisfies (k : ℕ) : Prop :=
∃ A : ℕ → ℕ → Prop,
  (∀ i : ℕ, i < k → ∀ n ≥ 15, satisfies_condition (A i) n)

-- Our conjecture is that k can be at most 3 for the given condition
theorem greatest_partition (k : ℕ) : k ≤ 3 :=
sorry

end greatest_partition_l82_82417


namespace max_median_of_pos_integers_l82_82847

theorem max_median_of_pos_integers
  (k m p r s t u : ℕ)
  (h_avg : (k + m + p + r + s + t + u) / 7 = 24)
  (h_order : k < m ∧ m < p ∧ p < r ∧ r < s ∧ s < t ∧ t < u)
  (h_t : t = 54)
  (h_km_sum : k + m ≤ 20)
  : r ≤ 53 :=
sorry

end max_median_of_pos_integers_l82_82847


namespace triangle_side_BC_length_l82_82199

noncomputable def triangle_side_length
  (AB : ℝ) (angle_a : ℝ) (angle_c : ℝ) : ℝ := 
  let sin_a := Real.sin angle_a
  let sin_c := Real.sin angle_c
  (AB * sin_a) / sin_c

theorem triangle_side_BC_length (AB : ℝ) (angle_a angle_c : ℝ) :
  AB = (Real.sqrt 6) / 2 →
  angle_a = (45 * Real.pi / 180) →
  angle_c = (60 * Real.pi / 180) →
  triangle_side_length AB angle_a angle_c = 1 :=
sorry

end triangle_side_BC_length_l82_82199


namespace max_value_expression_l82_82562

theorem max_value_expression (x y : ℝ) : 
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 2) ≤ Real.sqrt 29 :=
by
  exact sorry

end max_value_expression_l82_82562


namespace length_of_each_movie_l82_82737

-- Defining the amount of time Grandpa Lou watched movies on Tuesday in minutes
def time_tuesday : ℕ := 4 * 60 + 30   -- 4 hours and 30 minutes

-- Defining the number of movies watched on Tuesday
def movies_tuesday (x : ℕ) : Prop := time_tuesday / x = 90

-- Defining the total number of movies watched in both days
def total_movies_two_days (x : ℕ) : Prop := x + 2 * x = 9

theorem length_of_each_movie (x : ℕ) (h₁ : total_movies_two_days x) (h₂ : movies_tuesday x) : time_tuesday / x = 90 :=
by
  -- Given the conditions, we can prove the statement:
  sorry

end length_of_each_movie_l82_82737


namespace boys_without_calculators_l82_82924

-- Definitions based on the conditions
def total_boys : Nat := 20
def students_with_calculators : Nat := 26
def girls_with_calculators : Nat := 15

-- We need to prove the number of boys who did not bring their calculators.
theorem boys_without_calculators : (total_boys - (students_with_calculators - girls_with_calculators)) = 9 :=
by {
    -- Proof goes here
    sorry
}

end boys_without_calculators_l82_82924


namespace apples_per_case_l82_82453

theorem apples_per_case (total_apples : ℕ) (number_of_cases : ℕ) (h1 : total_apples = 1080) (h2 : number_of_cases = 90) : total_apples / number_of_cases = 12 := by
  sorry

end apples_per_case_l82_82453


namespace correct_answer_l82_82290

-- Definition of the correctness condition
def indicates_number (phrase : String) : Prop :=
  (phrase = "Noun + Cardinal Number") ∨ (phrase = "the + Ordinal Number + Noun")

-- Example phrases to be evaluated
def class_first : String := "Class First"
def the_class_one : String := "the Class One"
def class_one : String := "Class One"
def first_class : String := "First Class"

-- The goal is to prove that "Class One" meets the condition
theorem correct_answer : indicates_number "Class One" :=
by {
  -- Insert detailed proof steps here, currently omitted
  sorry
}

end correct_answer_l82_82290


namespace half_percent_to_decimal_l82_82127

def percent_to_decimal (x : ℚ) : ℚ := x / 100

theorem half_percent_to_decimal : percent_to_decimal (1 / 2) = 0.005 :=
by
  sorry

end half_percent_to_decimal_l82_82127


namespace arithmetic_expression_evaluation_l82_82859

theorem arithmetic_expression_evaluation : 5 * (9 / 3) + 7 * 4 - 36 / 4 = 34 := 
by
  sorry

end arithmetic_expression_evaluation_l82_82859


namespace max_value_of_quadratic_l82_82934

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := -2 * x^2 + 8

-- State the problem formally in Lean
theorem max_value_of_quadratic : ∀ x : ℝ, quadratic x ≤ quadratic 0 :=
by
  -- Skipping the proof
  sorry

end max_value_of_quadratic_l82_82934


namespace vector_expression_identity_l82_82088

variables (E : Type) [AddCommGroup E] [Module ℝ E]
variables (e1 e2 : E)
variables (a b : E)
variables (cond1 : a = (3 : ℝ) • e1 - (2 : ℝ) • e2) (cond2 : b = (e2 - (2 : ℝ) • e1))

theorem vector_expression_identity :
  (1 / 3 : ℝ) • a + b + a - (3 / 2 : ℝ) • b + 2 • b - a = -2 • e1 + (5 / 6 : ℝ) • e2 :=
sorry

end vector_expression_identity_l82_82088


namespace find_a33_l82_82465

theorem find_a33 : 
  ∀ (a : ℕ → ℤ), a 1 = 3 → a 2 = 6 → (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) → a 33 = 3 :=
by
  intros a h1 h2 h_rec
  sorry

end find_a33_l82_82465


namespace multiples_of_three_l82_82110

theorem multiples_of_three (a b : ℤ) (h : 9 ∣ (a^2 + a * b + b^2)) : 3 ∣ a ∧ 3 ∣ b :=
by {
  sorry
}

end multiples_of_three_l82_82110


namespace necessary_not_sufficient_condition_l82_82678

theorem necessary_not_sufficient_condition (a : ℝ) :
  (a < 2) ∧ (a^2 - 4 < 0) ↔ (a < 2) ∧ (a > -2) :=
by
  sorry

end necessary_not_sufficient_condition_l82_82678


namespace point_B_coordinates_l82_82130

theorem point_B_coordinates :
  ∃ (B : ℝ × ℝ), (B.1 < 0) ∧ (|B.2| = 4) ∧ (|B.1| = 5) ∧ (B = (-5, 4) ∨ B = (-5, -4)) :=
sorry

end point_B_coordinates_l82_82130


namespace total_songs_l82_82413

variable (H : String) (M : String) (A : String) (T : String)

def num_songs (s : String) : ℕ :=
  if s = H then 9 else
  if s = M then 5 else
  if s = A ∨ s = T then 
    if H ≠ s ∧ M ≠ s then 6 else 7 
  else 0

theorem total_songs 
  (hH : num_songs H = 9)
  (hM : num_songs M = 5)
  (hA : 5 < num_songs A ∧ num_songs A < 9)
  (hT : 5 < num_songs T ∧ num_songs T < 9) :
  (num_songs H + num_songs M + num_songs A + num_songs T) / 3 = 10 :=
sorry

end total_songs_l82_82413


namespace ball_distributions_l82_82667

theorem ball_distributions (p q : ℚ) (h1 : p = (Nat.choose 5 1 * Nat.choose 4 1 * Nat.choose 20 2 * Nat.choose 18 6 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / Nat.choose 20 20)
                            (h2 : q = (Nat.choose 20 4 * Nat.choose 16 4 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / Nat.choose 20 20) :
  p / q = 10 :=
by
  sorry

end ball_distributions_l82_82667


namespace machine_B_fewer_bottles_l82_82035

-- Definitions and the main theorem statement
def MachineA_caps_per_minute : ℕ := 12
def MachineC_additional_capacity : ℕ := 5
def total_bottles_in_10_minutes : ℕ := 370

theorem machine_B_fewer_bottles (B : ℕ) 
  (h1 : MachineA_caps_per_minute * 10 + 10 * B + 10 * (B + MachineC_additional_capacity) = total_bottles_in_10_minutes) :
  MachineA_caps_per_minute - B = 2 :=
by
  sorry

end machine_B_fewer_bottles_l82_82035


namespace annie_budget_l82_82252

theorem annie_budget :
  let budget := 120
  let hamburger_count := 8
  let milkshake_count := 6
  let hamburgerA := 4
  let milkshakeA := 5
  let hamburgerB := 3.5
  let milkshakeB := 6
  let hamburgerC := 5
  let milkshakeC := 4
  let costA := hamburgerA * hamburger_count + milkshakeA * milkshake_count
  let costB := hamburgerB * hamburger_count + milkshakeB * milkshake_count
  let costC := hamburgerC * hamburger_count + milkshakeC * milkshake_count
  let min_cost := min costA (min costB costC)
  budget - min_cost = 58 :=
by {
  sorry
}

end annie_budget_l82_82252


namespace arithmetic_sequence_general_formula_l82_82933

theorem arithmetic_sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, 0 < n → (a n - 2 * a (n + 1) + a (n + 2) = 0)) : ∀ n : ℕ, a n = 2 * n :=
by
  sorry

end arithmetic_sequence_general_formula_l82_82933


namespace find_a_20_l82_82541

variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Definitions: The sequence is geometric: a_n = a_1 * r^(n-1)
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a 1 * r^(n-1)

-- Conditions in the problem: a_10 and a_30 satisfy the quadratic equation
def satisfies_quadratic_roots (a10 a30 : ℝ) : Prop :=
  a10 + a30 = 11 ∧ a10 * a30 = 16

-- Question: Find a_20
theorem find_a_20 (h1 : is_geometric_sequence a r)
                  (h2 : satisfies_quadratic_roots (a 10) (a 30)) :
  a 20 = 4 :=
sorry

end find_a_20_l82_82541


namespace range_of_m_l82_82717

variable (m : ℝ)

def hyperbola (m : ℝ) := (x y : ℝ) → (x^2 / (1 + m)) - (y^2 / (3 - m)) = 1

def eccentricity_condition (m : ℝ) := (2 / (Real.sqrt (1 + m)) > Real.sqrt 2)

theorem range_of_m (m : ℝ) (h1 : 1 + m > 0) (h2 : 3 - m > 0) (h3 : eccentricity_condition m) :
 -1 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l82_82717


namespace no_such_constant_l82_82604

noncomputable def f : ℚ → ℚ := sorry

theorem no_such_constant (h : ∀ x y : ℚ, ∃ k : ℤ, f (x + y) - f x - f y = k) :
  ¬ ∃ c : ℚ, ∀ x : ℚ, ∃ k : ℤ, f x - c * x = k := 
sorry

end no_such_constant_l82_82604


namespace no_zeros_sin_log_l82_82165

open Real

theorem no_zeros_sin_log (x : ℝ) (h1 : 1 < x) (h2 : x < exp 1) : ¬ (sin (log x) = 0) :=
sorry

end no_zeros_sin_log_l82_82165


namespace minimum_red_chips_l82_82879

variable (w b r : ℕ)

axiom C1 : b ≥ (1 / 3 : ℚ) * w
axiom C2 : b ≤ (1 / 4 : ℚ) * r
axiom C3 : w + b ≥ 75

theorem minimum_red_chips : r = 76 := by sorry

end minimum_red_chips_l82_82879


namespace largest_divisor_39_l82_82406

theorem largest_divisor_39 (m : ℕ) (hm : 0 < m) (h : 39 ∣ m ^ 2) : 39 ∣ m :=
by sorry

end largest_divisor_39_l82_82406


namespace sarah_shaded_area_l82_82479

theorem sarah_shaded_area (r : ℝ) (A_square : ℝ) (A_circle : ℝ) (A_circles : ℝ) (A_shaded : ℝ) :
  let side_length := 27
  let radius := side_length / (3 * 2)
  let area_square := side_length * side_length
  let area_one_circle := Real.pi * (radius * radius)
  let total_area_circles := 9 * area_one_circle
  let shaded_area := area_square - total_area_circles
  shaded_area = 729 - 182.25 * Real.pi := 
by
  sorry

end sarah_shaded_area_l82_82479


namespace combined_afternoon_burning_rate_l82_82501

theorem combined_afternoon_burning_rate 
  (morning_period_hours : ℕ)
  (afternoon_period_hours : ℕ)
  (rate_A_morning : ℕ)
  (rate_B_morning : ℕ)
  (total_morning_burn : ℕ)
  (initial_wood : ℕ)
  (remaining_wood : ℕ) :
  morning_period_hours = 4 →
  afternoon_period_hours = 4 →
  rate_A_morning = 2 →
  rate_B_morning = 1 →
  total_morning_burn = 12 →
  initial_wood = 50 →
  remaining_wood = 6 →
  ((initial_wood - remaining_wood - total_morning_burn) / afternoon_period_hours) = 8 := 
by
  intros
  -- We would continue with a proof here
  sorry

end combined_afternoon_burning_rate_l82_82501


namespace polycarp_error_l82_82000

def three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem polycarp_error (a b n : ℕ) (ha : three_digit a) (hb : three_digit b)
  (h : 10000 * a + b = n * a * b) : n = 73 :=
by
  sorry

end polycarp_error_l82_82000


namespace cost_of_bananas_l82_82030

theorem cost_of_bananas (A B : ℝ) (h1 : A + B = 5) (h2 : 2 * A + B = 7) : B = 3 :=
by
  sorry

end cost_of_bananas_l82_82030


namespace abs_neg_three_eq_three_l82_82750

theorem abs_neg_three_eq_three : abs (-3) = 3 := 
by 
  sorry

end abs_neg_three_eq_three_l82_82750


namespace solve_for_xy_l82_82835

theorem solve_for_xy (x y : ℝ) (h : 2 * x - 3 ≤ Real.log (x + y + 1) + Real.log (x - y - 2)) : x * y = -9 / 4 :=
by sorry

end solve_for_xy_l82_82835


namespace functional_relationship_and_point_l82_82260

noncomputable def directly_proportional (y x : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ y = k * x

theorem functional_relationship_and_point :
  (∀ x y, directly_proportional y x → y = 2 * x) ∧ 
  (∀ a : ℝ, (∃ (y : ℝ), y = 3 ∧ directly_proportional y a) → a = 3 / 2) :=
by
  sorry

end functional_relationship_and_point_l82_82260


namespace gcd_105_88_l82_82558

-- Define the numbers as constants
def a : ℕ := 105
def b : ℕ := 88

-- State the theorem: gcd(a, b) = 1
theorem gcd_105_88 : Nat.gcd a b = 1 := by
  sorry

end gcd_105_88_l82_82558


namespace ruffy_age_difference_l82_82885

theorem ruffy_age_difference (R O : ℕ) (hR : R = 9) (hRO : R = (3/4 : ℚ) * O) :
  (R - 4) - (1 / 2 : ℚ) * (O - 4) = 1 :=
by 
  sorry

end ruffy_age_difference_l82_82885


namespace rotated_angle_540_deg_l82_82304

theorem rotated_angle_540_deg (θ : ℝ) (h : θ = 60) : 
  (θ - 540) % 360 % 180 = 60 :=
by
  sorry

end rotated_angle_540_deg_l82_82304


namespace base5_representation_three_consecutive_digits_l82_82092

theorem base5_representation_three_consecutive_digits :
  ∃ (digits : ℕ), 
    (digits = 3) ∧ 
    (∃ (a1 a2 a3 : ℕ), 
      94 = a1 * 5^2 + a2 * 5^1 + a3 * 5^0 ∧
      a1 = 3 ∧ a2 = 3 ∧ a3 = 4 ∧
      (a1 = a3 + 1) ∧ (a2 = a3 + 2)) := 
    sorry

end base5_representation_three_consecutive_digits_l82_82092


namespace cost_equivalence_at_325_l82_82539

def cost_plan1 (x : ℕ) : ℝ := 65 + 0.40 * x
def cost_plan2 (x : ℕ) : ℝ := 0.60 * x

theorem cost_equivalence_at_325 : cost_plan1 325 = cost_plan2 325 :=
by sorry

end cost_equivalence_at_325_l82_82539


namespace common_factor_polynomials_l82_82952

-- Define the two polynomials
def poly1 (x y z : ℝ) := 3 * x^2 * y^3 * z + 9 * x^3 * y^3 * z
def poly2 (x y z : ℝ) := 6 * x^4 * y * z^2

-- Define the common factor
def common_factor (x y z : ℝ) := 3 * x^2 * y * z

-- The statement to prove that the common factor of poly1 and poly2 is 3 * x^2 * y * z
theorem common_factor_polynomials (x y z : ℝ) :
  ∃ (f : ℝ → ℝ → ℝ → ℝ), (poly1 x y z) = (f x y z) * (common_factor x y z) ∧
                          (poly2 x y z) = (f x y z) * (common_factor x y z) :=
sorry

end common_factor_polynomials_l82_82952


namespace number_of_integer_solutions_l82_82745

theorem number_of_integer_solutions : 
  (∃ (sols : List (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ sols ↔ (1 : ℚ)/x + (1 : ℚ)/y = 1/7) ∧ sols.length = 5) := 
sorry

end number_of_integer_solutions_l82_82745


namespace find_x_plus_y_l82_82091

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y = 2009) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 + Real.pi / 2 := 
by
  sorry

end find_x_plus_y_l82_82091


namespace combined_volleyball_percentage_l82_82573

theorem combined_volleyball_percentage (students_north: ℕ) (students_south: ℕ)
(percent_volleyball_north percent_volleyball_south: ℚ)
(H1: students_north = 1800) (H2: percent_volleyball_north = 0.25)
(H3: students_south = 2700) (H4: percent_volleyball_south = 0.35):
  (((students_north * percent_volleyball_north) + (students_south * percent_volleyball_south))
  / (students_north + students_south) * 100) = 31 := 
  sorry

end combined_volleyball_percentage_l82_82573


namespace prob_zhang_nings_wins_2_1_correct_prob_ξ_minus_2_correct_prob_ξ_minus_1_correct_prob_ξ_1_correct_prob_ξ_2_correct_expected_value_ξ_correct_l82_82582

noncomputable def prob_zhang_nings_wins_2_1 :=
  2 * 0.4 * 0.6 * 0.6 = 0.288

theorem prob_zhang_nings_wins_2_1_correct : prob_zhang_nings_wins_2_1 := sorry

def prob_ξ_minus_2 := 0.4 * 0.4 = 0.16
def prob_ξ_minus_1 := 2 * 0.4 * 0.6 * 0.4 = 0.192
def prob_ξ_1 := 2 * 0.4 * 0.6 * 0.6 = 0.288
def prob_ξ_2 := 0.6 * 0.6 = 0.36

theorem prob_ξ_minus_2_correct : prob_ξ_minus_2 := sorry
theorem prob_ξ_minus_1_correct : prob_ξ_minus_1 := sorry
theorem prob_ξ_1_correct : prob_ξ_1 := sorry
theorem prob_ξ_2_correct : prob_ξ_2 := sorry

noncomputable def expected_value_ξ :=
  (-2 * 0.16) + (-1 * 0.192) + (1 * 0.288) + (2 * 0.36) = 0.496

theorem expected_value_ξ_correct : expected_value_ξ := sorry

end prob_zhang_nings_wins_2_1_correct_prob_ξ_minus_2_correct_prob_ξ_minus_1_correct_prob_ξ_1_correct_prob_ξ_2_correct_expected_value_ξ_correct_l82_82582


namespace line_intersects_circle_l82_82317

theorem line_intersects_circle 
  (k : ℝ)
  (x y : ℝ)
  (h_line : x = 0 ∨ y = -2)
  (h_circle : (x - 1)^2 + (y + 2)^2 = 16) :
  (-2 - -2)^2 < 16 := by
  sorry

end line_intersects_circle_l82_82317


namespace hari_digs_well_alone_in_48_days_l82_82371

theorem hari_digs_well_alone_in_48_days :
  (1 / 16 + 1 / 24 + 1 / (Hari_days)) = 1 / 8 → Hari_days = 48 :=
by
  intro h
  sorry

end hari_digs_well_alone_in_48_days_l82_82371


namespace exists_multiple_with_sum_divisible_l82_82232

-- Define the sum of the digits
def sum_of_digits (n : ℕ) : ℕ := -- Implementation of sum_of_digits function is omitted here
sorry

-- Main theorem statement
theorem exists_multiple_with_sum_divisible (n : ℕ) (hn : n > 0) : 
  ∃ k, k % n = 0 ∧ sum_of_digits k ∣ k :=
sorry

end exists_multiple_with_sum_divisible_l82_82232


namespace find_abc_l82_82216

theorem find_abc (a b c : ℝ) (x y : ℝ) :
  (x^2 + y^2 + 2*a*x - b*y + c = 0) ∧
  ((-a, b / 2) = (2, 2)) ∧
  (4 = b^2 / 4 + a^2 - c) →
  a = -2 ∧ b = 4 ∧ c = 4 := by
  sorry

end find_abc_l82_82216


namespace total_surface_area_of_resulting_solid_is_12_square_feet_l82_82802

noncomputable def height_of_D :=
  let h_A := 1 / 4
  let h_B := 1 / 5
  let h_C := 1 / 8
  2 - (h_A + h_B + h_C)

theorem total_surface_area_of_resulting_solid_is_12_square_feet :
  let h_A := 1 / 4
  let h_B := 1 / 5
  let h_C := 1 / 8
  let h_D := 2 - (h_A + h_B + h_C)
  let top_and_bottom_area := 4 * 2
  let side_area := 2 * (h_A + h_B + h_C + h_D)
  top_and_bottom_area + side_area = 12 := by
  sorry

end total_surface_area_of_resulting_solid_is_12_square_feet_l82_82802


namespace value_of_x_plus_y_l82_82843

theorem value_of_x_plus_y (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 4) (h3 : x * y > 0) : x + y = 7 ∨ x + y = -7 :=
by
  sorry

end value_of_x_plus_y_l82_82843


namespace union_of_sets_l82_82599

def set_M : Set ℕ := {0, 1, 3}
def set_N : Set ℕ := {x | ∃ (a : ℕ), a ∈ set_M ∧ x = 3 * a}

theorem union_of_sets :
  set_M ∪ set_N = {0, 1, 3, 9} :=
by
  sorry

end union_of_sets_l82_82599


namespace average_speed_of_train_l82_82753

theorem average_speed_of_train (x : ℝ) (h1 : 0 < x) : 
  let Time1 := x / 40
  let Time2 := x / 10
  let TotalDistance := 3 * x
  let TotalTime := x / 8
  (TotalDistance / TotalTime = 24) :=
by
  sorry

end average_speed_of_train_l82_82753


namespace distance_between_trains_l82_82572

def speed_train1 : ℝ := 11 -- Speed of the first train in mph
def speed_train2 : ℝ := 31 -- Speed of the second train in mph
def time_travelled : ℝ := 8 -- Time in hours

theorem distance_between_trains : 
  (speed_train2 * time_travelled) - (speed_train1 * time_travelled) = 160 := by
  sorry

end distance_between_trains_l82_82572


namespace glasses_per_pitcher_l82_82510

theorem glasses_per_pitcher (t p g : ℕ) (ht : t = 54) (hp : p = 9) : g = t / p := by
  rw [ht, hp]
  norm_num
  sorry

end glasses_per_pitcher_l82_82510


namespace half_of_number_l82_82871

theorem half_of_number (x : ℝ) (h : (4 / 15 * 5 / 7 * x - 4 / 9 * 2 / 5 * x = 8)) : (1 / 2 * x = 315) :=
sorry

end half_of_number_l82_82871


namespace find_4a_3b_l82_82526

noncomputable def g (x : ℝ) : ℝ := 4 * x - 6

noncomputable def f_inv (x : ℝ) : ℝ := g x + 2

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem find_4a_3b (a b : ℝ) (h_inv : ∀ x : ℝ, f (f_inv x) a b = x) : 4 * a + 3 * b = 4 :=
by
  -- Proof skipped for now
  sorry

end find_4a_3b_l82_82526


namespace determine_n_l82_82006

theorem determine_n (n : ℕ) (h1 : n > 2020) (h2 : ∃ m : ℤ, (n - 2020) = m^2 * (2120 - n)) : 
  n = 2070 ∨ n = 2100 ∨ n = 2110 := 
sorry

end determine_n_l82_82006


namespace inverse_of_original_l82_82057

-- Definitions based on conditions
def original_proposition : Prop := ∀ (x y : ℝ), x = y → |x| = |y|

def inverse_proposition : Prop := ∀ (x y : ℝ), |x| = |y| → x = y

-- Lean 4 statement
theorem inverse_of_original : original_proposition → inverse_proposition :=
sorry

end inverse_of_original_l82_82057


namespace pills_supply_duration_l82_82617

open Nat

-- Definitions based on conditions
def one_third_pill_every_three_days : ℕ := 1 / 3 * 3
def pills_in_bottle : ℕ := 90
def days_per_pill : ℕ := 9
def days_per_month : ℕ := 30

-- The Lean statement to prove the question == answer given conditions
theorem pills_supply_duration : (pills_in_bottle * days_per_pill) / days_per_month = 27 := by
  sorry

end pills_supply_duration_l82_82617


namespace roots_of_equation_l82_82159

theorem roots_of_equation : ∃ x₁ x₂ : ℝ, (3 ^ x₁ = Real.log (x₁ + 9) / Real.log 3) ∧ 
                                     (3 ^ x₂ = Real.log (x₂ + 9) / Real.log 3) ∧ 
                                     (x₁ < 0) ∧ (x₂ > 0) := 
by {
  sorry
}

end roots_of_equation_l82_82159


namespace garden_length_increase_l82_82950

variable (L W : ℝ)  -- Original length and width
variable (X : ℝ)    -- Percentage increase in length

theorem garden_length_increase :
  (1 + X / 100) * 0.8 = 1.1199999999999999 → X = 40 :=
by
  sorry

end garden_length_increase_l82_82950


namespace dot_product_parallel_vectors_is_minus_ten_l82_82353

-- Definitions from the conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -4)
def are_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2) ∨ v = (k * u.1, k * u.2)

theorem dot_product_parallel_vectors_is_minus_ten (x : ℝ) (h : are_parallel vector_a (vector_b x)) : (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2) = -10 :=
by
  sorry

end dot_product_parallel_vectors_is_minus_ten_l82_82353


namespace boxes_amount_l82_82235

/-- 
  A food company has 777 kilograms of food to put into boxes. 
  If each box gets a certain amount of kilograms, they will have 388 full boxes.
  Prove that each box gets 2 kilograms of food.
-/
theorem boxes_amount (total_food : ℕ) (boxes : ℕ) (kilograms_per_box : ℕ) 
  (h_total : total_food = 777)
  (h_boxes : boxes = 388) :
  total_food / boxes = kilograms_per_box :=
by {
  -- Skipped proof
  sorry 
}

end boxes_amount_l82_82235


namespace reduced_price_is_correct_l82_82397

-- Definitions for the conditions in the problem
def original_price_per_dozen (P : ℝ) : Prop :=
∀ (X : ℝ), X * P = 40.00001

def reduced_price_per_dozen (P R : ℝ) : Prop :=
R = 0.60 * P

def bananas_purchased_additional (P R : ℝ) : Prop :=
∀ (X Y : ℝ), (Y = X + (64 / 12)) → (X * P = Y * R) 

-- Assertion of the proof problem
theorem reduced_price_is_correct : 
  ∃ (R : ℝ), 
  (∀ P, original_price_per_dozen P ∧ reduced_price_per_dozen P R ∧ bananas_purchased_additional P R) → 
  R = 3.00000075 := 
by sorry

end reduced_price_is_correct_l82_82397


namespace evens_in_triangle_l82_82093

theorem evens_in_triangle (a : ℕ → ℕ → ℕ) (h : ∀ i j, a i.succ j = (a i (j - 1) + a i j + a i (j + 1)) % 2) :
  ∀ n ≥ 2, ∃ j, a n j % 2 = 0 :=
  sorry

end evens_in_triangle_l82_82093


namespace stop_shooting_after_2nd_scoring_5_points_eq_l82_82383

/-
Define the conditions and problem statement in Lean:
- Each person can shoot up to 10 times.
- Student A's shooting probability for each shot is 2/3.
- If student A stops shooting at the nth consecutive shot, they score 12-n points.
- We need to prove the probability that student A stops shooting right after the 2nd shot and scores 5 points is 8/729.
-/
def student_shoot_probability (shots : List Bool) (p : ℚ) : ℚ :=
  shots.foldr (λ s acc => if s then p * acc else (1 - p) * acc) 1

def stop_shooting_probability : ℚ :=
  let shots : List Bool := [false, true, false, false, false, true, true] -- represents misses and hits
  student_shoot_probability shots (2/3)

theorem stop_shooting_after_2nd_scoring_5_points_eq :
  stop_shooting_probability = (8 / 729) :=
sorry

end stop_shooting_after_2nd_scoring_5_points_eq_l82_82383


namespace solve_quadratic_l82_82233

theorem solve_quadratic : ∃ x : ℚ, 3 * x^2 + 11 * x - 20 = 0 ∧ x > 0 ∧ x = 4 / 3 :=
by
  sorry

end solve_quadratic_l82_82233


namespace roots_cubic_l82_82736

theorem roots_cubic (a b c d r s t : ℂ) 
    (h1 : a ≠ 0)
    (h2 : r + s + t = -b / a)
    (h3 : r * s + r * t + s * t = c / a)
    (h4 : r * s * t = -d / a) :
    (1 / r^2) + (1 / s^2) + (1 / t^2) = (b^2 - 2 * a * c) / (d^2) :=
by
    sorry

end roots_cubic_l82_82736


namespace georgia_carnations_proof_l82_82357

-- Define the conditions
def carnation_cost : ℝ := 0.50
def dozen_cost : ℝ := 4.00
def friends_carnations : ℕ := 14
def total_spent : ℝ := 25.00

-- Define the answer
def teachers_dozen : ℕ := 4

-- Prove the main statement
theorem georgia_carnations_proof : 
  (total_spent - (friends_carnations * carnation_cost)) / dozen_cost = teachers_dozen :=
by
  sorry

end georgia_carnations_proof_l82_82357


namespace probability_not_snowing_l82_82378

theorem probability_not_snowing (p_snow : ℚ) (h : p_snow = 5 / 8) : 1 - p_snow = 3 / 8 :=
by
  rw [h]
  sorry

end probability_not_snowing_l82_82378


namespace smallest_gcd_of_lcm_eq_square_diff_l82_82758

theorem smallest_gcd_of_lcm_eq_square_diff (x y : ℕ) (h : Nat.lcm x y = (x - y) ^ 2) : Nat.gcd x y = 2 :=
sorry

end smallest_gcd_of_lcm_eq_square_diff_l82_82758


namespace joy_valid_rod_count_l82_82988

theorem joy_valid_rod_count : 
  let l := [4, 12, 21]
  let qs := [1, 2, 3, 5, 13, 20, 22, 40].filter (fun x => x != 4 ∧ x != 12 ∧ x != 21)
  (∀ d ∈ qs, 4 + 12 + 21 > d ∧ 4 + 12 + d > 21 ∧ 4 + 21 + d > 12 ∧ 12 + 21 + d > 4) → 
  ∃ n, n = 28 :=
by sorry

end joy_valid_rod_count_l82_82988


namespace fraction_of_green_balls_l82_82106

theorem fraction_of_green_balls (T G : ℝ)
    (h1 : (1 / 8) * T = 6)
    (h2 : (1 / 12) * T + (1 / 8) * T + 26 = T - G)
    (h3 : (1 / 8) * T = 6)
    (h4 : 26 ≥ 0):
  G / T = 1 / 4 :=
by
  sorry

end fraction_of_green_balls_l82_82106


namespace clairaut_equation_solution_l82_82983

open Real

noncomputable def clairaut_solution (f : ℝ → ℝ) (C : ℝ) : Prop :=
  (∀ x, f x = C * x + 1/(2 * C)) ∨ (∀ x, (f x)^2 = 2 * x)

theorem clairaut_equation_solution (y : ℝ → ℝ) :
  (∀ x, y x = x * (deriv y x) + 1/(2 * (deriv y x))) →
  ∃ C, clairaut_solution y C :=
sorry

end clairaut_equation_solution_l82_82983


namespace initial_birds_count_l82_82959

theorem initial_birds_count (B : ℕ) :
  ∃ B, B + 4 = 5 + 2 → B = 3 :=
by
  sorry

end initial_birds_count_l82_82959


namespace marker_cost_is_13_l82_82898

theorem marker_cost_is_13 :
  ∃ s m c : ℕ, (s > 20) ∧ (m ≥ 4) ∧ (c > m) ∧ (s * c * m = 3185) ∧ (c = 13) :=
by
  sorry

end marker_cost_is_13_l82_82898


namespace right_triangle_circum_inradius_sum_l82_82143

theorem right_triangle_circum_inradius_sum
  (a b : ℕ)
  (h1 : a = 16)
  (h2 : b = 30)
  (h_triangle : a^2 + b^2 = 34^2) :
  let c := 34
  let R := c / 2
  let A := a * b / 2
  let s := (a + b + c) / 2
  let r := A / s
  R + r = 23 :=
by
  sorry

end right_triangle_circum_inradius_sum_l82_82143


namespace exterior_angle_octagon_degree_l82_82531

-- Conditions
def sum_of_exterior_angles (n : ℕ) : ℕ := 360
def number_of_sides_octagon : ℕ := 8

-- Question and correct answer
theorem exterior_angle_octagon_degree :
  (sum_of_exterior_angles 8) / number_of_sides_octagon = 45 :=
by
  sorry

end exterior_angle_octagon_degree_l82_82531


namespace no_real_solution_l82_82050

theorem no_real_solution :
  ∀ x : ℝ, ((x - 4 * x + 15)^2 + 3)^2 + 1 ≠ -|x|^2 :=
by
  intro x
  sorry

end no_real_solution_l82_82050


namespace lowest_possible_price_l82_82515

theorem lowest_possible_price
  (manufacturer_suggested_price : ℝ := 45)
  (regular_discount_percentage : ℝ := 0.30)
  (sale_discount_percentage : ℝ := 0.20)
  (regular_discounted_price : ℝ := manufacturer_suggested_price * (1 - regular_discount_percentage))
  (final_price : ℝ := regular_discounted_price * (1 - sale_discount_percentage)) :
  final_price = 25.20 :=
by sorry

end lowest_possible_price_l82_82515


namespace intersection_of_A_and_B_l82_82048

-- Definitions based on conditions
def set_A : Set ℝ := {x | x ≥ 3 ∨ x ≤ -1}
def set_B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Statement of the proof problem
theorem intersection_of_A_and_B : set_A ∩ set_B = {x | -2 ≤ x ∧ x ≤ -1} :=
  sorry

end intersection_of_A_and_B_l82_82048


namespace angus_tokens_count_l82_82152

def worth_of_token : ℕ := 4
def elsa_tokens : ℕ := 60
def difference_worth : ℕ := 20

def elsa_worth : ℕ := elsa_tokens * worth_of_token
def angus_worth : ℕ := elsa_worth - difference_worth

def angus_tokens : ℕ := angus_worth / worth_of_token

theorem angus_tokens_count : angus_tokens = 55 := by
  sorry

end angus_tokens_count_l82_82152


namespace system_of_equations_l82_82217

theorem system_of_equations (x y z : ℝ) (h1 : 4 * x - 6 * y - 2 * z = 0) (h2 : 2 * x + 6 * y - 28 * z = 0) (hz : z ≠ 0) :
  (x^2 - 6 * x * y) / (y^2 + 4 * z^2) = -5 :=
by
  sorry

end system_of_equations_l82_82217


namespace find_k_from_given_solution_find_other_root_l82_82172

-- Given
def one_solution_of_first_eq_is_same_as_second (x k : ℝ) : Prop :=
  x^2 + k * x - 2 = 0 ∧ (x + 1) / (x - 1) = 3

-- To find k
theorem find_k_from_given_solution : ∃ k : ℝ, ∃ x : ℝ, one_solution_of_first_eq_is_same_as_second x k ∧ k = -1 := by
  sorry

-- To find the other root
theorem find_other_root : ∃ x2 : ℝ, (x2 = -1) := by
  sorry

end find_k_from_given_solution_find_other_root_l82_82172


namespace min_ab_value_l82_82967

theorem min_ab_value 
  (a b : ℝ) 
  (hab_pos : a * b > 0)
  (collinear_condition : 2 * a + 2 * b + a * b = 0) :
  a * b ≥ 16 := 
sorry

end min_ab_value_l82_82967


namespace factorize_expression_l82_82932

theorem factorize_expression (x : ℝ) : x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  sorry

end factorize_expression_l82_82932


namespace total_people_is_120_l82_82702

def num_children : ℕ := 80

def num_adults (num_children : ℕ) : ℕ := num_children / 2

def total_people (num_children num_adults : ℕ) : ℕ := num_children + num_adults

theorem total_people_is_120 : total_people num_children (num_adults num_children) = 120 := by
  sorry

end total_people_is_120_l82_82702


namespace smallest_square_area_l82_82236

theorem smallest_square_area :
  (∀ (x y : ℝ), (∃ (x1 x2 y1 y2 : ℝ), y1 = 3 * x1 - 4 ∧ y2 = 3 * x2 - 4 ∧ y = x^2 + 5 ∧ 
  ∀ (k : ℝ), x1 + x2 = 3 ∧ x1 * x2 = 5 - k ∧ 16 * k^2 - 332 * k + 396 = 0 ∧ 
  ((k = 1.5 ∧ 10 * (4 * k - 11) = 50) ∨ 
  (k = 16.5 ∧ 10 * (4 * k - 11) ≠ 50))) → 
  ∃ (A: Real), A = 50) :=
sorry

end smallest_square_area_l82_82236


namespace three_subsets_equal_sum_l82_82647

theorem three_subsets_equal_sum (n : ℕ) (h1 : n ≡ 0 [MOD 3] ∨ n ≡ 2 [MOD 3]) (h2 : 5 ≤ n) :
  ∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range (n + 1) ∧
                        A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅ ∧
                        A.sum id = B.sum id ∧ B.sum id = C.sum id ∧ C.sum id = A.sum id :=
sorry

end three_subsets_equal_sum_l82_82647


namespace heads_not_consecutive_probability_l82_82544

theorem heads_not_consecutive_probability :
  (∃ n m : ℕ, n = 2^4 ∧ m = 1 + Nat.choose 4 1 + Nat.choose 3 2 ∧ (m / n : ℚ) = 1 / 2) :=
by
  use 16     -- n
  use 8      -- m
  sorry

end heads_not_consecutive_probability_l82_82544


namespace find_c_l82_82553

theorem find_c (c q : ℤ) (h : ∃ (a b : ℤ), (3*x^3 + c*x + 9 = (x^2 + q*x + 1) * (a*x + b))) : c = -24 :=
sorry

end find_c_l82_82553


namespace minimum_value_of_function_l82_82657

theorem minimum_value_of_function : ∀ x : ℝ, x ≥ 0 → (4 * x^2 + 12 * x + 25) / (6 * (1 + x)) ≥ 8 / 3 := by
  sorry

end minimum_value_of_function_l82_82657


namespace fish_in_pond_l82_82540

-- Conditions
variable (N : ℕ)
variable (h₁ : 80 * 80 = 2 * N)

-- Theorem to prove 
theorem fish_in_pond (h₁ : 80 * 80 = 2 * N) : N = 3200 := 
by 
  sorry

end fish_in_pond_l82_82540


namespace geometric_sequence_exists_l82_82402

theorem geometric_sequence_exists 
  (a r : ℚ)
  (h1 : a = 3)
  (h2 : a * r = 8 / 9)
  (h3 : a * r^2 = 32 / 81) : 
  r = 8 / 27 :=
by
  sorry

end geometric_sequence_exists_l82_82402


namespace fraction_ordering_l82_82140

theorem fraction_ordering : (4 / 17) < (6 / 25) ∧ (6 / 25) < (8 / 31) :=
by
  sorry

end fraction_ordering_l82_82140


namespace length_of_each_section_25_l82_82996

theorem length_of_each_section_25 (x : ℝ) 
  (h1 : ∃ x, x > 0)
  (h2 : 1000 / x = 15 / (1 / 2 * 3 / 4))
  : x = 25 := 
  sorry

end length_of_each_section_25_l82_82996


namespace range_of_a_l82_82345

theorem range_of_a (a x y : ℝ) (h1 : 77 * a = (2 * x + 2 * y) / 2) (h2 : Real.sqrt (abs a) = Real.sqrt (x * y)) :
  a ∈ Set.Iic (-4) ∪ Set.Ici 4 :=
sorry

end range_of_a_l82_82345


namespace quadratic_solutions_l82_82314

theorem quadratic_solutions (x : ℝ) : x * (x - 1) = 1 - x ↔ x = 1 ∨ x = -1 :=
by
  sorry

end quadratic_solutions_l82_82314


namespace contractor_original_days_l82_82323

noncomputable def original_days (total_laborers absent_laborers working_laborers days_worked : ℝ) : ℝ :=
  (working_laborers * days_worked) / (total_laborers - absent_laborers)

-- Our conditions:
def total_laborers : ℝ := 21.67
def absent_laborers : ℝ := 5
def working_laborers : ℝ := 16.67
def days_worked : ℝ := 13

-- Our main theorem:
theorem contractor_original_days :
  original_days total_laborers absent_laborers working_laborers days_worked = 10 := 
by
  sorry

end contractor_original_days_l82_82323


namespace surface_area_of_rectangular_solid_is_334_l82_82446

theorem surface_area_of_rectangular_solid_is_334
  (l w h : ℕ)
  (h_l_prime : Prime l)
  (h_w_prime : Prime w)
  (h_h_prime : Prime h)
  (volume_eq_385 : l * w * h = 385) : 
  2 * (l * w + l * h + w * h) = 334 := 
sorry

end surface_area_of_rectangular_solid_is_334_l82_82446


namespace solve_for_x_l82_82032

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 5 / (x / 3)) : x = 15 ∨ x = -15 := by
  sorry

end solve_for_x_l82_82032


namespace least_positive_integer_l82_82193

theorem least_positive_integer (n : ℕ) :
  (∃ n : ℕ, 25^n + 16^n ≡ 1 [MOD 121] ∧ ∀ m : ℕ, (m < n ∧ 25^m + 16^m ≡ 1 [MOD 121]) → false) ↔ n = 32 :=
sorry

end least_positive_integer_l82_82193


namespace sequence_sum_zero_l82_82230

-- Define the sequence as a function
def seq (n : ℕ) : ℤ :=
  if (n-1) % 8 < 4
  then (n+1) / 2
  else - (n / 2)

-- Define the sum of the sequence up to a given number
def seq_sum (m : ℕ) : ℤ :=
  (Finset.range (m+1)).sum (λ n => seq n)

-- The actual problem statement
theorem sequence_sum_zero : seq_sum 2012 = 0 :=
  sorry

end sequence_sum_zero_l82_82230


namespace eq_square_sum_five_l82_82015

theorem eq_square_sum_five (a b : ℝ) (i : ℂ) (h : i * i = -1) (h_eq : (a - 2 * i) * i^2013 = b - i) : a^2 + b^2 = 5 :=
by
  -- Proof will be filled in later
  sorry

end eq_square_sum_five_l82_82015


namespace arithmetic_sequence_property_l82_82864

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : ((a 6 - 1)^3 + 2013 * (a 6 - 1)^3 = 1))
  (h2 : ((a 2008 - 1)^3 = -2013 * (a 2008 - 1)^3))
  (sum_formula : ∀ n, S n = n * a n) : 
  S 2013 = 2013 ∧ a 2008 < a 6 := 
sorry

end arithmetic_sequence_property_l82_82864


namespace compressor_stations_valid_l82_82886

def compressor_stations : Prop :=
  ∃ (x y z a : ℝ),
    x + y = 3 * z ∧  -- condition 1
    z + y = x + a ∧  -- condition 2
    x + z = 60 ∧     -- condition 3
    0 < a ∧ a < 60 ∧ -- condition 4
    a = 42 ∧         -- specific value for a
    x = 33 ∧         -- expected value for x
    y = 48 ∧         -- expected value for y
    z = 27           -- expected value for z

theorem compressor_stations_valid : compressor_stations := 
  by sorry

end compressor_stations_valid_l82_82886


namespace triangle_count_lower_bound_l82_82231

theorem triangle_count_lower_bound (n m : ℕ) (S : Finset (ℕ × ℕ))
  (hS : ∀ (a b : ℕ), (a, b) ∈ S → 1 ≤ a ∧ a < b ∧ b ≤ n) (hm : S.card = m) :
  ∃T, T ≥ 4 * m * (m - n^2 / 4) / (3 * n) := 
by 
  sorry

end triangle_count_lower_bound_l82_82231


namespace find_integers_satisfying_equation_l82_82184

theorem find_integers_satisfying_equation :
  ∃ (a b c : ℤ), (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b = 1 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c = 1) ∨
                  (a = 2 ∧ b = -1 ∧ c = -1) ∨ (a = -1 ∧ b = 2 ∧ c = -1) ∨ (a = -1 ∧ b = -1 ∧ c = 2)
  ↔ (∃ (a b c : ℤ), 1 / 2 * (a + b) * (b + c) * (c + a) + (a + b + c) ^ 3 = 1 - a * b * c) := sorry

end find_integers_satisfying_equation_l82_82184


namespace percentage_of_boys_is_60_percent_l82_82068

-- Definition of the problem conditions
def totalPlayers := 50
def juniorGirls := 10
def half (n : ℕ) := n / 2
def girls := 2 * juniorGirls
def boys := totalPlayers - girls
def percentage_of_boys := (boys * 100) / totalPlayers

-- The theorem stating the proof problem
theorem percentage_of_boys_is_60_percent : percentage_of_boys = 60 := 
by 
  -- Proof omitted
  sorry

end percentage_of_boys_is_60_percent_l82_82068


namespace circle_diameter_C_l82_82192

theorem circle_diameter_C {D C : ℝ} (hD : D = 20) (h_ratio : (π * (D/2)^2 - π * (C/2)^2) / (π * (C/2)^2) = 4) : C = 4 * Real.sqrt 5 := 
sorry

end circle_diameter_C_l82_82192


namespace maximal_partition_sets_l82_82868

theorem maximal_partition_sets : 
  ∃(n : ℕ), (∀(a : ℕ), a * n = 16657706 → (a = 5771 ∧ n = 2886)) := 
by
  sorry

end maximal_partition_sets_l82_82868


namespace gcd_odd_multiple_1187_l82_82902

theorem gcd_odd_multiple_1187 (b: ℤ) (h1: b % 2 = 1) (h2: ∃ k: ℤ, b = 1187 * k) :
  Int.gcd (3 * b ^ 2 + 34 * b + 76) (b + 16) = 1 :=
by
  sorry

end gcd_odd_multiple_1187_l82_82902


namespace problem1_problem2_problem3_l82_82228

-- Prove \(2x = 4\) is a "difference solution equation"
theorem problem1 (x : ℝ) : (2 * x = 4) → x = 4 - 2 :=
by
  sorry

-- Given \(4x = ab + a\) is a "difference solution equation", prove \(3(ab + a) = 16\)
theorem problem2 (x ab a : ℝ) : (4 * x = ab + a) → 3 * (ab + a) = 16 :=
by
  sorry

-- Given \(4x = mn + m\) and \(-2x = mn + n\) are both "difference solution equations", prove \(3(mn + m) - 9(mn + n)^2 = 0\)
theorem problem3 (x mn m n : ℝ) :
  (4 * x = mn + m) ∧ (-2 * x = mn + n) → 3 * (mn + m) - 9 * (mn + n)^2 = 0 :=
by
  sorry

end problem1_problem2_problem3_l82_82228


namespace student_solved_18_correctly_l82_82851

theorem student_solved_18_correctly (total_problems : ℕ) (correct : ℕ) (wrong : ℕ) 
  (h1 : total_problems = 54) (h2 : wrong = 2 * correct) (h3 : total_problems = correct + wrong) :
  correct = 18 :=
by
  sorry

end student_solved_18_correctly_l82_82851


namespace age_of_new_person_l82_82764

-- Definitions based on conditions
def initial_avg : ℕ := 15
def new_avg : ℕ := 17
def n : ℕ := 9

-- Statement to prove
theorem age_of_new_person : 
    ∃ (A : ℕ), (initial_avg * n + A) / (n + 1) = new_avg ∧ A = 35 := 
by {
    -- Proof steps would go here, but since they are not required, we add 'sorry' to skip the proof
    sorry
}

end age_of_new_person_l82_82764


namespace correct_operation_l82_82694

theorem correct_operation (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end correct_operation_l82_82694


namespace factor_of_polynomial_l82_82474

theorem factor_of_polynomial (x : ℝ) : 
  (x^2 - 2*x + 2) ∣ (29 * 39 * x^4 + 4) :=
sorry

end factor_of_polynomial_l82_82474


namespace solve_equation_l82_82055

theorem solve_equation (x : ℝ) (h : 3 + 1 / (2 - x) = 2 * (1 / (2 - x))) : x = 5 / 3 := 
  sorry

end solve_equation_l82_82055


namespace subtraction_contradiction_l82_82053

theorem subtraction_contradiction (k t : ℕ) (hk_non_zero : k ≠ 0) (ht_non_zero : t ≠ 0) : 
  ¬ ((8 * 100 + k * 10 + 8) - (k * 100 + 8 * 10 + 8) = 1 * 100 + 6 * 10 + t * 1) :=
by
  sorry

end subtraction_contradiction_l82_82053


namespace green_peaches_per_basket_l82_82600

-- Definitions based on given conditions
def total_peaches : ℕ := 10
def red_peaches_per_basket : ℕ := 4

-- Theorem statement based on the question and correct answer
theorem green_peaches_per_basket :
  (total_peaches - red_peaches_per_basket) = 6 := 
by
  sorry

end green_peaches_per_basket_l82_82600


namespace express_y_in_terms_of_x_l82_82478

theorem express_y_in_terms_of_x (x y : ℝ) (h : x + 2 * y = 6) : y = (-x + 6) / 2 := 
by { sorry }

end express_y_in_terms_of_x_l82_82478


namespace lesser_fraction_of_sum_and_product_l82_82485

open Real

theorem lesser_fraction_of_sum_and_product (a b : ℚ)
  (h1 : a + b = 11 / 12)
  (h2 : a * b = 1 / 6) :
  min a b = 1 / 4 :=
sorry

end lesser_fraction_of_sum_and_product_l82_82485


namespace sum_of_roots_l82_82894
-- Import Mathlib to cover all necessary functionality.

-- Define the function representing the given equation.
def equation (x : ℝ) : ℝ :=
  (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- State the theorem to be proved.
theorem sum_of_roots : (6 : ℝ) + (-4 / 3) = 14 / 3 :=
by
  sorry

end sum_of_roots_l82_82894


namespace four_ab_eq_four_l82_82635

theorem four_ab_eq_four {a b : ℝ} (h : a * b = 1) : 4 * a * b = 4 :=
by
  sorry

end four_ab_eq_four_l82_82635


namespace find_y_l82_82085

variable (x y : ℤ)

-- Conditions
def cond1 : Prop := x + y = 280
def cond2 : Prop := x - y = 200

-- Proof statement
theorem find_y (h1 : cond1 x y) (h2 : cond2 x y) : y = 40 := 
by 
  sorry

end find_y_l82_82085


namespace area_bounded_by_curve_and_line_l82_82697

theorem area_bounded_by_curve_and_line :
  let curve_x (t : ℝ) := 10 * (t - Real.sin t)
  let curve_y (t : ℝ) := 10 * (1 - Real.cos t)
  let y_line := 15
  (∫ t in (2/3) * Real.pi..(4/3) * Real.pi, 100 * (1 - Real.cos t)^2) = 100 * Real.pi + 200 * Real.sqrt 3 :=
by
  sorry

end area_bounded_by_curve_and_line_l82_82697


namespace maria_total_cost_l82_82255

variable (pencil_cost : ℕ)
variable (pen_cost : ℕ)

def total_cost (pencil_cost pen_cost : ℕ) : ℕ :=
  pencil_cost + pen_cost

theorem maria_total_cost : pencil_cost = 8 → pen_cost = pencil_cost / 2 → total_cost pencil_cost pen_cost = 12 := by
  sorry

end maria_total_cost_l82_82255


namespace good_or_bad_of_prime_divides_l82_82438

-- Define the conditions
variables (k n n' : ℕ)
variables (h1 : k ≥ 2) (h2 : n ≥ k) (h3 : n' ≥ k)
variables (prime_divides : ∀ p, prime p → p ≤ k → (p ∣ n ↔ p ∣ n'))

-- Define what it means for a number to be good or bad
def is_good (m : ℕ) : Prop := ∃ strategy : ℕ → Prop, strategy m

-- Prove that either both n and n' are good or both are bad
theorem good_or_bad_of_prime_divides :
  (is_good n ∧ is_good n') ∨ (¬is_good n ∧ ¬is_good n') :=
sorry

end good_or_bad_of_prime_divides_l82_82438


namespace work_completion_time_l82_82080

/-- q can complete the work in 9 days, r can complete the work in 12 days, they work together
for 3 days, and p completes the remaining work in 10.000000000000002 days. Prove that
p alone can complete the work in approximately 24 days. -/
theorem work_completion_time (W : ℝ) (q : ℝ) (r : ℝ) (p : ℝ) :
  q = 9 → r = 12 → (p * 10.000000000000002 = (5 / 12) * W) →
  p = 24.000000000000004 :=
by 
  intros hq hr hp
  sorry

end work_completion_time_l82_82080


namespace poly_sum_correct_l82_82547

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := -4 * x^2 + 12 * x - 12

theorem poly_sum_correct : ∀ x : ℝ, p x + q x + r x = s x :=
by
  sorry

end poly_sum_correct_l82_82547


namespace largest_number_l82_82471

def A : ℚ := 97 / 100
def B : ℚ := 979 / 1000
def C : ℚ := 9709 / 10000
def D : ℚ := 907 / 1000
def E : ℚ := 9089 / 10000

theorem largest_number : B > A ∧ B > C ∧ B > D ∧ B > E := by
  sorry

end largest_number_l82_82471


namespace real_root_in_interval_l82_82865

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 3

theorem real_root_in_interval : ∃ α : ℝ, f α = 0 ∧ 1 < α ∧ α < 2 :=
sorry

end real_root_in_interval_l82_82865


namespace solution_set_l82_82337

-- Define the function and the conditions
variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

-- Problem statement
theorem solution_set (hf_even : is_even f)
                     (hf_increasing : increasing_on f (Set.Ioi 0))
                     (hf_value : f (-2013) = 0) :
  {x | x * f x < 0} = {x | x < -2013 ∨ (0 < x ∧ x < 2013)} :=
by
  sorry

end solution_set_l82_82337


namespace distance_le_radius_l82_82513

variable (L : Line) (O : Circle)
variable (d r : ℝ)

-- Condition: Line L intersects with circle O
def intersects (L : Line) (O : Circle) : Prop := sorry -- Sketch: define what it means for a line to intersect a circle

axiom intersection_condition : intersects L O

-- Problem: Prove that if a line L intersects a circle O, then the distance d from the center of the circle to the line is less than or equal to the radius r of the circle.
theorem distance_le_radius (L : Line) (O : Circle) (d r : ℝ) :
  intersects L O → d ≤ r := by
  sorry

end distance_le_radius_l82_82513


namespace shorter_piece_length_l82_82977

theorem shorter_piece_length (total_len : ℝ) (h1 : total_len = 60)
                            (short_len long_len : ℝ) (h2 : long_len = (1 / 2) * short_len)
                            (h3 : short_len + long_len = total_len) :
  short_len = 40 := 
  sorry

end shorter_piece_length_l82_82977


namespace smallest_n_divisible_l82_82364

theorem smallest_n_divisible {n : ℕ} : 
  (∃ n : ℕ, n > 0 ∧ 18 ∣ n^2 ∧ 1152 ∣ n^3 ∧ 
    (∀ m : ℕ, m > 0 → 18 ∣ m^2 → 1152 ∣ m^3 → n ≤ m)) :=
  sorry

end smallest_n_divisible_l82_82364


namespace fraction_zero_if_abs_x_eq_one_l82_82653

theorem fraction_zero_if_abs_x_eq_one (x : ℝ) : 
  (|x| - 1) = 0 → (x^2 - 2 * x + 1 ≠ 0) → x = -1 := 
by 
  sorry

end fraction_zero_if_abs_x_eq_one_l82_82653


namespace average_discount_rate_l82_82848

theorem average_discount_rate :
  ∃ x : ℝ, (7200 * (1 - x)^2 = 3528) ∧ x = 0.3 :=
by
  sorry

end average_discount_rate_l82_82848


namespace proof_x1_x2_squared_l82_82407

theorem proof_x1_x2_squared (x1 x2 : ℝ) (h1 : (Real.exp 1 * x1)^x2 = (Real.exp 1 * x2)^x1)
  (h2 : 0 < x1) (h3 : 0 < x2) (h4 : x1 ≠ x2) : x1^2 + x2^2 > 2 :=
sorry

end proof_x1_x2_squared_l82_82407


namespace flower_beds_fraction_correct_l82_82910

noncomputable def flower_beds_fraction (yard_length : ℝ) (yard_width : ℝ) (trapezoid_parallel_side1 : ℝ) (trapezoid_parallel_side2 : ℝ) : ℝ :=
  let leg_length := (trapezoid_parallel_side2 - trapezoid_parallel_side1) / 2
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 2 * triangle_area
  let yard_area := yard_length * yard_width
  total_flower_bed_area / yard_area

theorem flower_beds_fraction_correct :
  flower_beds_fraction 30 5 20 30 = 1 / 6 :=
by
  sorry

end flower_beds_fraction_correct_l82_82910


namespace trigonometric_identity_l82_82640

open Real

theorem trigonometric_identity (α : ℝ) (h : sin (α - (π / 12)) = 1 / 3) :
  cos (α + (17 * π / 12)) = 1 / 3 :=
sorry

end trigonometric_identity_l82_82640


namespace sum_of_zeros_gt_two_l82_82047

noncomputable def f (a x : ℝ) := 2 * a * Real.log x + x ^ 2 - 2 * (a + 1) * x

theorem sum_of_zeros_gt_two (a x1 x2 : ℝ) (h_a : -0.5 < a ∧ a < 0)
  (h_fx_zeros : f a x1 = 0 ∧ f a x2 = 0) (h_x_order : x1 < x2) : x1 + x2 > 2 := 
sorry

end sum_of_zeros_gt_two_l82_82047


namespace candy_bar_sugar_calories_l82_82089

theorem candy_bar_sugar_calories
  (candy_bars : Nat)
  (soft_drink_calories : Nat)
  (soft_drink_sugar_percentage : Float)
  (recommended_sugar_intake : Nat)
  (excess_percentage : Nat)
  (sugar_in_each_bar : Nat) :
  candy_bars = 7 ∧
  soft_drink_calories = 2500 ∧
  soft_drink_sugar_percentage = 0.05 ∧
  recommended_sugar_intake = 150 ∧
  excess_percentage = 100 →
  sugar_in_each_bar = 25 := by
  sorry

end candy_bar_sugar_calories_l82_82089


namespace total_gymnasts_l82_82498

theorem total_gymnasts (n : ℕ) : 
  (∃ (t : ℕ) (c : t = 4) (h : n * (n-1) / 2 + 4 * 6 = 595), n = 34) :=
by {
  -- skipping the detailed proof here, just ensuring the problem is stated as a theorem
  sorry
}

end total_gymnasts_l82_82498


namespace profit_difference_is_50_l82_82901

-- Given conditions
def materials_cost_cars : ℕ := 100
def cars_made_per_month : ℕ := 4
def price_per_car : ℕ := 50

def materials_cost_motorcycles : ℕ := 250
def motorcycles_made_per_month : ℕ := 8
def price_per_motorcycle : ℕ := 50

-- Definitions based on the conditions
def profit_from_cars : ℕ := (cars_made_per_month * price_per_car) - materials_cost_cars
def profit_from_motorcycles : ℕ := (motorcycles_made_per_month * price_per_motorcycle) - materials_cost_motorcycles

-- The difference in profit
def profit_difference : ℕ := profit_from_motorcycles - profit_from_cars

-- The proof goal
theorem profit_difference_is_50 :
  profit_difference = 50 := by
  sorry

end profit_difference_is_50_l82_82901


namespace shorter_stick_length_l82_82945

variable (L S : ℝ)

theorem shorter_stick_length
  (h1 : L - S = 12)
  (h2 : (2 / 3) * L = S) :
  S = 24 := by
  sorry

end shorter_stick_length_l82_82945


namespace circle_intersects_y_axis_with_constraints_l82_82797

theorem circle_intersects_y_axis_with_constraints {m n : ℝ} 
    (H1 : n = m ^ 2 + 2 * m + 2) 
    (H2 : abs m <= 2) : 
    1 ≤ n ∧ n < 10 :=
sorry

end circle_intersects_y_axis_with_constraints_l82_82797


namespace min_max_value_l82_82815

theorem min_max_value
  (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h₁ : 0 ≤ x₁) (h₂ : 0 ≤ x₂) (h₃ : 0 ≤ x₃) (h₄ : 0 ≤ x₄) (h₅ : 0 ≤ x₅)
  (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 1) :
  (min (max (x₁ + x₂) (max (x₂ + x₃) (max (x₃ + x₄) (x₄ + x₅)))) = 1 / 3) :=
sorry

end min_max_value_l82_82815


namespace same_terminal_side_angle_in_range_0_to_2pi_l82_82399

theorem same_terminal_side_angle_in_range_0_to_2pi :
  ∃ k : ℤ, 0 ≤ 2 * k * π + (-4) * π / 3 ∧ 2 * k * π + (-4) * π / 3 ≤ 2 * π ∧
  2 * k * π + (-4) * π / 3 = 2 * π / 3 :=
by
  use 1
  sorry

end same_terminal_side_angle_in_range_0_to_2pi_l82_82399


namespace sum_of_distinct_products_l82_82026

theorem sum_of_distinct_products (G H : ℕ) (hG : G < 10) (hH : H < 10) :
  (3 * H + 8) % 8 = 0 ∧ ((6 + 2 + 8 + G + 4 + 0 + 9 + 3 + H + 8) % 9 = 0) →
  (G * H = 6 ∨ G * H = 48) →
  6 + 48 = 54 :=
by
  intros _ _
  sorry

end sum_of_distinct_products_l82_82026


namespace fifth_roll_six_probability_l82_82978
noncomputable def probability_fifth_roll_six : ℚ := sorry

theorem fifth_roll_six_probability :
  let fair_die_prob : ℚ := (1/6)^4
  let biased_die_6_prob : ℚ := (2/3)^3 * (1/15)
  let biased_die_3_prob : ℚ := (1/10)^3 * (1/2)
  let total_prob := (1/3) * fair_die_prob + (1/3) * biased_die_6_prob + (1/3) * biased_die_3_prob
  let normalized_biased_6_prob := (1/3) * biased_die_6_prob / total_prob
  let prob_of_fifth_six := normalized_biased_6_prob * (2/3)
  probability_fifth_roll_six = prob_of_fifth_six :=
sorry

end fifth_roll_six_probability_l82_82978


namespace oil_price_reduction_l82_82197

theorem oil_price_reduction (P P_reduced : ℝ) (h1 : P_reduced = 50) (h2 : 1000 / P_reduced - 5 = 5) :
  ((P - P_reduced) / P) * 100 = 25 := by
  sorry

end oil_price_reduction_l82_82197


namespace division_result_l82_82542

theorem division_result : (0.284973 / 29 = 0.009827) := 
by sorry

end division_result_l82_82542


namespace max_area_inscribed_triangle_l82_82845

/-- Let ΔABC be an inscribed triangle in the ellipse given by the equation
    (x^2 / 9) + (y^2 / 4) = 1, where the line segment AB passes through the 
    point (1, 0). Prove that the maximum area of ΔABC is (16 * sqrt 2) / 3. --/
theorem max_area_inscribed_triangle
  (A B C : ℝ × ℝ) 
  (hA : (A.1 ^ 2) / 9 + (A.2 ^ 2) / 4 = 1)
  (hB : (B.1 ^ 2) / 9 + (B.2 ^ 2) / 4 = 1)
  (hC : (C.1 ^ 2) / 9 + (C.2 ^ 2) / 4 = 1)
  (hAB : ∃ n : ℝ, ∀ x y : ℝ, (x, y) ∈ [A, B] → x = n * y + 1)
  : ∃ S : ℝ, S = ((16 : ℝ) * Real.sqrt 2) / 3 :=
sorry

end max_area_inscribed_triangle_l82_82845


namespace min_weight_of_lightest_l82_82448

theorem min_weight_of_lightest (m n : ℕ) (hm : m > 0) (hn : n > 0) 
  (h1 : 71 * m + m = 72 * m) 
  (h2 : 34 * n + n = 35 * n) 
  (h3 : 72 * m = 35 * n) : m = 35 := 
sorry

end min_weight_of_lightest_l82_82448


namespace inequality_solution_l82_82424

theorem inequality_solution :
  {x : ℝ // -1 < (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) ∧ (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) < 1} = 
  {x : ℝ // x > 1/6} :=
sorry

end inequality_solution_l82_82424


namespace evie_gave_2_shells_to_brother_l82_82341

def daily_shells : ℕ := 10
def days : ℕ := 6
def remaining_shells : ℕ := 58

def total_shells : ℕ := daily_shells * days
def shells_given : ℕ := total_shells - remaining_shells

theorem evie_gave_2_shells_to_brother :
  shells_given = 2 :=
by
  sorry

end evie_gave_2_shells_to_brother_l82_82341


namespace solve_quadratic_eq_l82_82222

theorem solve_quadratic_eq (x : ℝ) : x^2 + 2 * x - 1 = 0 ↔ (x = -1 + Real.sqrt 2 ∨ x = -1 - Real.sqrt 2) :=
by
  sorry

end solve_quadratic_eq_l82_82222


namespace tracy_initial_candies_l82_82457

variable (x : ℕ)
variable (b : ℕ)

theorem tracy_initial_candies : 
  (x % 6 = 0) ∧
  (34 ≤ (1 / 2 * x)) ∧
  ((1 / 2 * x) ≤ 38) ∧
  (1 ≤ b) ∧
  (b ≤ 5) ∧
  (1 / 2 * x - 30 - b = 3) →
  x = 72 := 
sorry

end tracy_initial_candies_l82_82457


namespace collections_in_bag_l82_82261

noncomputable def distinct_collections : ℕ :=
  let vowels := ['A', 'I', 'O']
  let consonants := ['M', 'H', 'C', 'N', 'T', 'T']
  let case1 := Nat.choose 3 2 * Nat.choose 6 3 -- when 0 or 1 T falls off
  let case2 := Nat.choose 3 2 * Nat.choose 5 1 -- when both T's fall off
  case1 + case2

theorem collections_in_bag : distinct_collections = 75 := 
  by
  -- proof goes here
  sorry

end collections_in_bag_l82_82261


namespace greatest_integer_gcd_l82_82256

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end greatest_integer_gcd_l82_82256


namespace value_of_y_l82_82947

variables (x y : ℝ)

theorem value_of_y (h1 : x - y = 16) (h2 : x + y = 8) : y = -4 :=
by
  sorry

end value_of_y_l82_82947


namespace gardener_hourly_wage_l82_82810

-- Conditions
def rose_bushes_count : Nat := 20
def cost_per_rose_bush : Nat := 150
def hours_per_day : Nat := 5
def days_worked : Nat := 4
def soil_volume : Nat := 100
def cost_per_cubic_foot_soil : Nat := 5
def total_cost : Nat := 4100

-- Theorem statement
theorem gardener_hourly_wage :
  let cost_of_rose_bushes := rose_bushes_count * cost_per_rose_bush
  let cost_of_soil := soil_volume * cost_per_cubic_foot_soil
  let total_material_cost := cost_of_rose_bushes + cost_of_soil
  let labor_cost := total_cost - total_material_cost
  let total_hours_worked := hours_per_day * days_worked
  (labor_cost / total_hours_worked) = 30 := 
by {
  -- Proof placeholder
  sorry
}

end gardener_hourly_wage_l82_82810


namespace smallest_integer_l82_82713

theorem smallest_integer :
  ∃ (M : ℕ), M > 0 ∧
             M % 3 = 2 ∧
             M % 4 = 3 ∧
             M % 5 = 4 ∧
             M % 6 = 5 ∧
             M % 7 = 6 ∧
             M % 11 = 10 ∧
             M = 4619 :=
by
  sorry

end smallest_integer_l82_82713


namespace intersection_at_one_point_l82_82728

theorem intersection_at_one_point (m : ℝ) :
  (∃ x : ℝ, (m - 4) * x^2 - 2 * m * x - m - 6 = 0 ∧
            ∀ x' : ℝ, (m - 4) * x'^2 - 2 * m * x' - m - 6 = 0 → x' = x) ↔
  m = -4 ∨ m = 3 ∨ m = 4 := 
by
  sorry

end intersection_at_one_point_l82_82728


namespace proof_speed_of_man_in_still_water_l82_82001

def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
  50 / 4 = v_m + v_s ∧ 30 / 6 = v_m - v_s

theorem proof_speed_of_man_in_still_water (v_m v_s : ℝ) :
  speed_of_man_in_still_water v_m v_s → v_m = 8.75 :=
by
  intro h
  sorry

end proof_speed_of_man_in_still_water_l82_82001


namespace semicircle_area_difference_l82_82181

theorem semicircle_area_difference 
  (A B C P D E F : Type) 
  (h₁ : S₅ - S₆ = 2) 
  (h₂ : S₁ - S₂ = 1) 
  : S₄ - S₃ = 3 :=
by
  -- Using Lean tactics to form the proof, place sorry for now.
  sorry

end semicircle_area_difference_l82_82181


namespace verify_sum_of_fourth_powers_l82_82974

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def sum_of_fourth_powers (n : ℕ) : ℕ :=
  ((n * (n + 1) * (2 * n + 1) * (3 * n^2 + 3 * n - 1)) / 30)

noncomputable def square_of_sum (n : ℕ) : ℕ :=
  (n * (n + 1) / 2)^2

theorem verify_sum_of_fourth_powers (n : ℕ) :
  5 * sum_of_fourth_powers n = (4 * n + 2) * square_of_sum n - sum_of_squares n := 
  sorry

end verify_sum_of_fourth_powers_l82_82974


namespace greatest_x_solution_l82_82097

theorem greatest_x_solution (x : ℝ) (h₁ : (x^2 - x - 30) / (x - 6) = 2 / (x + 4)) : x ≤ -3 :=
sorry

end greatest_x_solution_l82_82097


namespace part_a_part_b_part_c_l82_82430

-- Definitions for the problem
def hard_problem_ratio_a := 2 / 3
def unsolved_problem_ratio_a := 2 / 3
def well_performing_students_ratio_a := 2 / 3

def hard_problem_ratio_b := 3 / 4
def unsolved_problem_ratio_b := 3 / 4
def well_performing_students_ratio_b := 3 / 4

def hard_problem_ratio_c := 7 / 10
def unsolved_problem_ratio_c := 7 / 10
def well_performing_students_ratio_c := 7 / 10

-- Theorems to prove
theorem part_a : 
  ∃ (hard_problem_ratio_a unsolved_problem_ratio_a well_performing_students_ratio_a : ℚ),
  hard_problem_ratio_a == 2 / 3 ∧
  unsolved_problem_ratio_a == 2 / 3 ∧
  well_performing_students_ratio_a == 2 / 3 →
  (True) := sorry

theorem part_b : 
  ∀ (hard_problem_ratio_b : ℚ),
  hard_problem_ratio_b == 3 / 4 →
  (False) := sorry

theorem part_c : 
  ∀ (hard_problem_ratio_c : ℚ),
  hard_problem_ratio_c == 7 / 10 →
  (False) := sorry

end part_a_part_b_part_c_l82_82430


namespace fixed_point_of_line_l82_82567

theorem fixed_point_of_line (m : ℝ) : 
  ∀ (x y : ℝ), (3 * x - 2 * y + 7 = 0) ∧ (4 * x + 5 * y - 6 = 0) → x = -1 ∧ y = 2 :=
sorry

end fixed_point_of_line_l82_82567


namespace passing_percentage_correct_l82_82186

-- The given conditions
def marks_obtained : ℕ := 175
def marks_failed : ℕ := 89
def max_marks : ℕ := 800

-- The theorem to prove
theorem passing_percentage_correct :
  (
    (marks_obtained + marks_failed : ℕ) * 100 / max_marks
  ) = 33 :=
sorry

end passing_percentage_correct_l82_82186


namespace xy_squares_l82_82576

theorem xy_squares (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 4) : x^2 - y^2 = 40 := 
by 
  sorry

end xy_squares_l82_82576


namespace chemist_solution_l82_82964

theorem chemist_solution (x : ℝ) (h1 : ∃ x, 0 < x) 
  (h2 : x + 1 > 1) : 0.60 * x = 0.10 * (x + 1) → x = 0.2 := by
  sorry

end chemist_solution_l82_82964


namespace range_of_a_l82_82449

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def g (x : ℝ) : ℝ := Real.log (Real.exp x - 1) - Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x0 : ℝ, 0 < x0 ∧ f (g x0) a > f x0 a) ↔ 1 < a := sorry

end range_of_a_l82_82449


namespace cody_money_final_l82_82346

theorem cody_money_final (initial_money : ℕ) (birthday_money : ℕ) (money_spent : ℕ) (final_money : ℕ) 
  (h1 : initial_money = 45) (h2 : birthday_money = 9) (h3 : money_spent = 19) :
  final_money = initial_money + birthday_money - money_spent :=
by {
  sorry  -- The proof is not required here.
}

end cody_money_final_l82_82346


namespace longest_pencil_l82_82122

/-- Hallway dimensions and the longest pencil problem -/
theorem longest_pencil (L : ℝ) : 
    (∃ P : ℝ, P = 3 * L) :=
sorry

end longest_pencil_l82_82122


namespace largest_radius_cone_l82_82922

structure Crate :=
  (width : ℝ)
  (depth : ℝ)
  (height : ℝ)

structure Cone :=
  (radius : ℝ)
  (height : ℝ)

noncomputable def larger_fit_within_crate (c : Crate) (cone : Cone) : Prop :=
  cone.radius = min c.width c.depth / 2 ∧ cone.height = max (max c.width c.depth) c.height

theorem largest_radius_cone (c : Crate) (cone : Cone) : 
  c.width = 5 → c.depth = 8 → c.height = 12 → larger_fit_within_crate c cone → cone.radius = 2.5 :=
by
  sorry

end largest_radius_cone_l82_82922


namespace find_cost_price_l82_82671

theorem find_cost_price (SP : ℝ) (loss_percent : ℝ) (CP : ℝ) (h1 : SP = 1260) (h2 : loss_percent = 16) : CP = 1500 :=
by
  sorry

end find_cost_price_l82_82671


namespace friends_count_l82_82305

theorem friends_count (n : ℕ) (average_rent : ℝ) (new_average_rent : ℝ) (original_rent : ℝ) (increase_percent : ℝ)
  (H1 : average_rent = 800)
  (H2 : new_average_rent = 870)
  (H3 : original_rent = 1400)
  (H4 : increase_percent = 0.20) :
  n = 4 :=
by
  -- Define the initial total rent
  let initial_total_rent := n * average_rent
  -- Define the increased rent for one person
  let increased_rent := original_rent * (1 + increase_percent)
  -- Define the new total rent
  let new_total_rent := initial_total_rent - original_rent + increased_rent
  -- Set up the new average rent equation
  have rent_equation := new_total_rent = n * new_average_rent
  sorry

end friends_count_l82_82305


namespace equivalent_sets_l82_82693

-- Definitions of the condition and expected result
def condition_set : Set ℕ := { x | x - 3 < 2 }
def expected_set : Set ℕ := {0, 1, 2, 3, 4}

-- Theorem statement
theorem equivalent_sets : condition_set = expected_set := 
by
  sorry

end equivalent_sets_l82_82693


namespace remainder_of_55_power_55_plus_55_div_56_l82_82441

theorem remainder_of_55_power_55_plus_55_div_56 :
  (55 ^ 55 + 55) % 56 = 54 :=
by
  -- to be filled with the proof
  sorry

end remainder_of_55_power_55_plus_55_div_56_l82_82441


namespace quadratic_no_real_roots_l82_82227

theorem quadratic_no_real_roots (k : ℝ) : (∀ x : ℝ, x^2 - 3 * x - k ≠ 0) → k < -9 / 4 :=
by
  sorry

end quadratic_no_real_roots_l82_82227


namespace pie_difference_l82_82683

theorem pie_difference:
  ∀ (a b c d : ℚ), a = 6 / 7 → b = 3 / 4 → (a - b) = c → c = 3 / 28 :=
by
  sorry

end pie_difference_l82_82683


namespace cheaper_store_price_in_cents_l82_82414

/-- List price of Book Y -/
def list_price : ℝ := 24.95

/-- Discount at Readers' Delight -/
def readers_delight_discount : ℝ := 5

/-- Discount rate at Book Bargains -/
def book_bargains_discount_rate : ℝ := 0.2

/-- Calculate sale price at Readers' Delight -/
def sale_price_readers_delight : ℝ := list_price - readers_delight_discount

/-- Calculate sale price at Book Bargains -/
def sale_price_book_bargains : ℝ := list_price * (1 - book_bargains_discount_rate)

/-- Difference in price between Book Bargains and Readers' Delight in cents -/
theorem cheaper_store_price_in_cents :
  (sale_price_book_bargains - sale_price_readers_delight) * 100 = 1 :=
by
  sorry

end cheaper_store_price_in_cents_l82_82414


namespace number_of_children_l82_82850

theorem number_of_children (C : ℝ) 
  (h1 : 0.30 * C >= 0)
  (h2 : 0.20 * C >= 0)
  (h3 : 0.50 * C >= 0)
  (h4 : 0.70 * C = 42) : 
  C = 60 := by
  sorry

end number_of_children_l82_82850


namespace henry_added_water_l82_82511

theorem henry_added_water (initial_fraction full_capacity final_fraction : ℝ) (h_initial_fraction : initial_fraction = 3/4) (h_full_capacity : full_capacity = 56) (h_final_fraction : final_fraction = 7/8) :
  final_fraction * full_capacity - initial_fraction * full_capacity = 7 :=
by
  sorry

end henry_added_water_l82_82511


namespace b_95_mod_49_l82_82739

def b (n : ℕ) : ℕ := 5^n + 7^n + 3

theorem b_95_mod_49 : b 95 % 49 = 5 := 
by sorry

end b_95_mod_49_l82_82739


namespace area_ratio_eq_two_l82_82242

/-- 
  Given a unit square, let circle B be the inscribed circle and circle A be the circumscribed circle.
  Prove the ratio of the area of circle A to the area of circle B is 2.
--/
theorem area_ratio_eq_two (r_B r_A : ℝ) (hB : r_B = 1 / 2) (hA : r_A = Real.sqrt 2 / 2):
  (π * r_A ^ 2) / (π * r_B ^ 2) = 2 := by
  sorry

end area_ratio_eq_two_l82_82242


namespace weighted_average_remaining_two_l82_82992

theorem weighted_average_remaining_two (avg_10 : ℝ) (avg_2 : ℝ) (avg_3 : ℝ) (avg_3_next : ℝ) :
  avg_10 = 4.25 ∧ avg_2 = 3.4 ∧ avg_3 = 3.85 ∧ avg_3_next = 4.7 →
  (42.5 - (2 * 3.4 + 3 * 3.85 + 3 * 4.7)) / 2 = 5.025 :=
by
  intros
  sorry

end weighted_average_remaining_two_l82_82992


namespace flower_position_after_50_beats_l82_82768

-- Define the number of students
def num_students : Nat := 7

-- Define the initial position of the flower
def initial_position : Nat := 1

-- Define the number of drum beats
def drum_beats : Nat := 50

-- Theorem stating that after 50 drum beats, the flower will be with the 2nd student
theorem flower_position_after_50_beats : 
  (initial_position + (drum_beats % num_students)) % num_students = 2 := by
  -- Start the proof (this part usually would contain the actual proof logic)
  sorry

end flower_position_after_50_beats_l82_82768


namespace ordered_pairs_count_l82_82215

theorem ordered_pairs_count :
  ∃ (p : Finset (ℕ × ℕ)), (∀ (a b : ℕ), (a, b) ∈ p → a * b + 45 = 10 * Nat.lcm a b + 18 * Nat.gcd a b) ∧
  p.card = 4 :=
by
  sorry

end ordered_pairs_count_l82_82215


namespace difference_of_squares_division_l82_82009

theorem difference_of_squares_division :
  let a := 121
  let b := 112
  (a^2 - b^2) / 3 = 699 :=
by
  sorry

end difference_of_squares_division_l82_82009


namespace problem_statement_l82_82630

theorem problem_statement (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) : y * (y - 1) ≤ x^2 := 
sorry

end problem_statement_l82_82630


namespace find_y_l82_82096

theorem find_y
  (x y : ℝ)
  (h1 : x - y = 10)
  (h2 : x + y = 8) : y = -1 :=
by
  sorry

end find_y_l82_82096


namespace solve_absolute_value_equation_l82_82551

theorem solve_absolute_value_equation (x : ℝ) : x^2 - 3 * |x| - 4 = 0 ↔ x = 4 ∨ x = -4 :=
by
  sorry

end solve_absolute_value_equation_l82_82551


namespace three_distinct_real_solutions_l82_82431

theorem three_distinct_real_solutions (b c : ℝ):
  (∀ x : ℝ, x^2 + b * |x| + c = 0 → x = 0) ∧ (∃! x : ℝ, x^2 + b * |x| + c = 0) →
  b < 0 ∧ c = 0 :=
by {
  sorry
}

end three_distinct_real_solutions_l82_82431


namespace pears_for_36_bananas_l82_82412

theorem pears_for_36_bananas (p : ℕ) (bananas : ℕ) (pears : ℕ) (h : 9 * pears = 6 * bananas) :
  36 * pears = 9 * 24 :=
by
  sorry

end pears_for_36_bananas_l82_82412


namespace gwen_spending_l82_82958

theorem gwen_spending : 
    ∀ (initial_amount spent remaining : ℕ), 
    initial_amount = 7 → remaining = 5 → initial_amount - remaining = 2 :=
by
    sorry

end gwen_spending_l82_82958


namespace ratio_x_to_y_is_12_l82_82225

noncomputable def ratio_x_y (x y : ℝ) (h1 : y = x * (1 - 0.9166666666666666)) : ℝ := x / y

theorem ratio_x_to_y_is_12 (x y : ℝ) (h1 : y = x * (1 - 0.9166666666666666)) : ratio_x_y x y h1 = 12 :=
sorry

end ratio_x_to_y_is_12_l82_82225


namespace smallest_coin_remainder_l82_82416

theorem smallest_coin_remainder
  (c : ℕ)
  (h1 : c % 8 = 6)
  (h2 : c % 7 = 5)
  (h3 : ∀ d : ℕ, (d % 8 = 6) → (d % 7 = 5) → d ≥ c) :
  c % 9 = 2 :=
sorry

end smallest_coin_remainder_l82_82416


namespace solution_proof_l82_82335

noncomputable def proof_problem : Prop :=
  ∀ (x : ℝ), x ≠ 1 → (1 - 1 / (x - 1) = 2 * x / (1 - x)) → x = 2 / 3

theorem solution_proof : proof_problem := 
by
  sorry

end solution_proof_l82_82335


namespace range_of_a_l82_82913

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (1 + a) * x^2 + a * x + a < x^2 + 1) : a ≤ 0 := 
sorry

end range_of_a_l82_82913


namespace seating_arrangements_l82_82912

theorem seating_arrangements (n m k : Nat) (couples : Fin n -> Fin m -> Prop):
  let pairs : Nat := k
  let adjusted_pairs : Nat := pairs / 24
  adjusted_pairs = 5760 := by
  sorry

end seating_arrangements_l82_82912


namespace length_second_platform_l82_82480

-- Define the conditions
def length_train : ℕ := 100
def time_platform1 : ℕ := 15
def length_platform1 : ℕ := 350
def time_platform2 : ℕ := 20

-- Prove the length of the second platform is 500m
theorem length_second_platform : ∀ (speed_train : ℚ), 
  speed_train = (length_train + length_platform1) / time_platform1 →
  (speed_train = (length_train + L) / time_platform2) → 
  L = 500 :=
by 
  intro speed_train h1 h2
  sorry

end length_second_platform_l82_82480


namespace largest_n_for_divisibility_l82_82434

theorem largest_n_for_divisibility :
  ∃ n : ℕ, (n + 15) ∣ (n^3 + 250) ∧ ∀ m : ℕ, ((m + 15) ∣ (m^3 + 250)) → (m ≤ 10) → (n = 10) :=
by {
  sorry
}

end largest_n_for_divisibility_l82_82434


namespace inequality_solution_l82_82714

noncomputable def solution_set : Set ℝ := {x : ℝ | x < 4 ∨ x > 5}

theorem inequality_solution (x : ℝ) :
  (x - 2) / (x - 4) ≤ 3 ↔ x ∈ solution_set :=
by
  sorry

end inequality_solution_l82_82714


namespace solve_system_l82_82703

theorem solve_system :
  ∃ (x y : ℤ), (x * (1/7 : ℚ)^2 = 7^3) ∧ (x + y = 7^2) ∧ (x = 16807) ∧ (y = -16758) :=
by
  sorry

end solve_system_l82_82703


namespace cost_of_softball_l82_82631

theorem cost_of_softball 
  (original_budget : ℕ)
  (dodgeball_cost : ℕ)
  (num_dodgeballs : ℕ)
  (increase_rate : ℚ)
  (num_softballs : ℕ)
  (new_budget : ℕ)
  (softball_cost : ℕ)
  (h0 : original_budget = num_dodgeballs * dodgeball_cost)
  (h1 : increase_rate = 0.20)
  (h2 : new_budget = original_budget + increase_rate * original_budget)
  (h3 : new_budget = num_softballs * softball_cost) :
  softball_cost = 9 :=
by
  sorry

end cost_of_softball_l82_82631


namespace vector_addition_l82_82853

variable {𝕍 : Type} [AddCommGroup 𝕍] [Module ℝ 𝕍]
variable (a b : 𝕍)

theorem vector_addition : 
  (1 / 2 : ℝ) • (2 • a - 4 • b) + 2 • b = a :=
by
  sorry

end vector_addition_l82_82853


namespace range_of_half_alpha_minus_beta_l82_82409

theorem range_of_half_alpha_minus_beta (α β : ℝ) (h1 : 1 < α) (h2 : α < 3) (h3 : -4 < β) (h4 : β < 2) :
  -3/2 < (1/2) * α - β ∧ (1/2) * α - β < 11/2 :=
by
  -- sorry to skip the proof
  sorry

end range_of_half_alpha_minus_beta_l82_82409


namespace angles_on_y_axis_l82_82444

theorem angles_on_y_axis :
  {θ : ℝ | ∃ k : ℤ, (θ = 2 * k * Real.pi + Real.pi / 2) ∨ (θ = 2 * k * Real.pi + 3 * Real.pi / 2)} =
  {θ : ℝ | ∃ n : ℤ, θ = n * Real.pi + Real.pi / 2} :=
by 
  sorry

end angles_on_y_axis_l82_82444


namespace probability_X_Y_Z_problems_l82_82295

-- Define the success probabilities for Problem A
def P_X_A : ℚ := 1 / 5
def P_Y_A : ℚ := 1 / 2

-- Define the success probabilities for Problem B
def P_Y_B : ℚ := 3 / 5

-- Define the negation of success probabilities for Problem C
def P_Y_not_C : ℚ := 5 / 8
def P_X_not_C : ℚ := 3 / 4
def P_Z_not_C : ℚ := 7 / 16

-- State the final probability theorem
theorem probability_X_Y_Z_problems :
  P_X_A * P_Y_A * P_Y_B * P_Y_not_C * P_X_not_C * P_Z_not_C = 63 / 2048 := 
sorry

end probability_X_Y_Z_problems_l82_82295


namespace maximum_PM_minus_PN_l82_82587

noncomputable def x_squared_over_9_minus_y_squared_over_16_eq_1 (x y : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / 16) = 1

noncomputable def circle1 (x y : ℝ) : Prop :=
  (x + 5)^2 + y^2 = 4

noncomputable def circle2 (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = 1

theorem maximum_PM_minus_PN :
  ∀ (P M N : ℝ × ℝ),
    x_squared_over_9_minus_y_squared_over_16_eq_1 P.1 P.2 →
    circle1 M.1 M.2 →
    circle2 N.1 N.2 →
    (|dist P M - dist P N| ≤ 9) := sorry

end maximum_PM_minus_PN_l82_82587


namespace solve_for_x_l82_82834

theorem solve_for_x (x : ℕ) (h : x + 1 = 4) : x = 3 :=
by
  sorry

end solve_for_x_l82_82834


namespace simplify_fraction_144_1008_l82_82627

theorem simplify_fraction_144_1008 :
  (144 : ℤ) / (1008 : ℤ) = (1 : ℤ) / (7 : ℤ) :=
by
  sorry

end simplify_fraction_144_1008_l82_82627


namespace correct_completion_at_crossroads_l82_82858

theorem correct_completion_at_crossroads :
  (∀ (s : String), 
    s = "An accident happened at a crossroads a few meters away from a bank" → 
    (∃ (general_sense : Bool), general_sense = tt)) :=
by
  sorry

end correct_completion_at_crossroads_l82_82858


namespace bottle_caps_weight_l82_82698

theorem bottle_caps_weight :
  (∀ n : ℕ, n = 7 → 1 = 1) → -- 7 bottle caps weigh exactly 1 ounce
  (∀ m : ℕ, m = 2016 → 1 = 1) → -- Josh has 2016 bottle caps
  2016 / 7 = 288 := -- The weight of Josh's entire bottle cap collection is 288 ounces
by
  intros h1 h2
  sorry

end bottle_caps_weight_l82_82698


namespace calculate_expression_l82_82661

theorem calculate_expression : 
  let a := (-1 : Int) ^ 2023
  let b := (-8 : Int) / (-4)
  let c := abs (-5)
  a + b - c = -4 := 
by
  sorry

end calculate_expression_l82_82661


namespace minimum_small_bottles_l82_82340

-- Define the capacities of the bottles
def small_bottle_capacity : ℕ := 35
def large_bottle_capacity : ℕ := 500

-- Define the number of small bottles needed to fill a large bottle
def small_bottles_needed_to_fill_large : ℕ := 
  (large_bottle_capacity + small_bottle_capacity - 1) / small_bottle_capacity

-- Statement of the theorem
theorem minimum_small_bottles : small_bottles_needed_to_fill_large = 15 := by
  sorry

end minimum_small_bottles_l82_82340


namespace sum_of_below_avg_l82_82785

-- Define class averages
def a1 := 75
def a2 := 85
def a3 := 90
def a4 := 65

-- Define the overall average
def avg : ℚ := (a1 + a2 + a3 + a4) / 4

-- Define a predicate indicating if a class average is below the overall average
def below_avg (a : ℚ) : Prop := a < avg

-- The theorem to prove the required sum of averages below the overall average
theorem sum_of_below_avg : a1 < avg ∧ a4 < avg → a1 + a4 = 140 :=
by
  sorry

end sum_of_below_avg_l82_82785


namespace verify_conditions_l82_82571

-- Define the conditions as expressions
def condition_A (a : ℝ) : Prop := 2 * a * 3 * a = 6 * a
def condition_B (a b : ℝ) : Prop := 3 * a^2 * b - 3 * a * b^2 = 0
def condition_C (a : ℝ) : Prop := 6 * a / (2 * a) = 3
def condition_D (a : ℝ) : Prop := (-2 * a) ^ 3 = -6 * a^3

-- Prove which condition is correct
theorem verify_conditions (a b : ℝ) (h : a ≠ 0) : 
  ¬ condition_A a ∧ ¬ condition_B a b ∧ condition_C a ∧ ¬ condition_D a :=
by 
  sorry

end verify_conditions_l82_82571


namespace faye_total_books_l82_82014

def initial_books : ℕ := 34
def books_given_away : ℕ := 3
def books_bought : ℕ := 48

theorem faye_total_books : initial_books - books_given_away + books_bought = 79 :=
by
  sorry

end faye_total_books_l82_82014


namespace smallest_digits_to_append_l82_82773

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l82_82773


namespace solve_system_l82_82354

-- Define the conditions of the system of equations
def condition1 (x y : ℤ) := 4 * x - 3 * y = -13
def condition2 (x y : ℤ) := 5 * x + 3 * y = -14

-- Define the proof goal using the conditions
theorem solve_system : ∃ (x y : ℤ), condition1 x y ∧ condition2 x y ∧ x = -3 ∧ y = 1 / 3 :=
by
  sorry

end solve_system_l82_82354
