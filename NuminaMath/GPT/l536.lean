import Mathlib

namespace problem_expression_eval_l536_53604

theorem problem_expression_eval : (1 + 2 + 3) * (1 + 1/2 + 1/3) = 11 := by
  sorry

end problem_expression_eval_l536_53604


namespace solve_inequality_l536_53618

theorem solve_inequality : {x : ℝ | |x - 2| * (x - 1) < 2} = {x : ℝ | x < 3} :=
by
  sorry

end solve_inequality_l536_53618


namespace angles_at_point_l536_53627

theorem angles_at_point (x y : ℝ) 
  (h1 : x + y + 120 = 360) 
  (h2 : x = 2 * y) : 
  x = 160 ∧ y = 80 :=
by
  sorry

end angles_at_point_l536_53627


namespace cats_combined_weight_l536_53642

theorem cats_combined_weight :
  let cat1 := 2
  let cat2 := 7
  let cat3 := 4
  cat1 + cat2 + cat3 = 13 := 
by
  let cat1 := 2
  let cat2 := 7
  let cat3 := 4
  sorry

end cats_combined_weight_l536_53642


namespace Sara_pears_left_l536_53624

def Sara_has_left (initial_pears : ℕ) (given_to_Dan : ℕ) (given_to_Monica : ℕ) (given_to_Jenny : ℕ) : ℕ :=
  initial_pears - given_to_Dan - given_to_Monica - given_to_Jenny

theorem Sara_pears_left :
  Sara_has_left 35 28 4 1 = 2 :=
by
  sorry

end Sara_pears_left_l536_53624


namespace exists_n_for_pow_lt_e_l536_53616

theorem exists_n_for_pow_lt_e {p e : ℝ} (hp : 0 < p ∧ p < 1) (he : 0 < e) :
  ∃ n : ℕ, (1 - p) ^ n < e :=
sorry

end exists_n_for_pow_lt_e_l536_53616


namespace sum_of_first_60_digits_l536_53681

-- Define the repeating sequence and the number of repetitions
def repeating_sequence : List ℕ := [0, 0, 0, 1]
def repetitions : ℕ := 15

-- Define the sum of first n elements of a repeating sequence
def sum_repeating_sequence (seq : List ℕ) (n : ℕ) : ℕ :=
  let len := seq.length
  let complete_cycles := n / len
  let remaining_digits := n % len
  let sum_complete_cycles := complete_cycles * seq.sum
  let sum_remaining_digits := (seq.take remaining_digits).sum
  sum_complete_cycles + sum_remaining_digits

-- Prove the specific case for 60 digits
theorem sum_of_first_60_digits : sum_repeating_sequence repeating_sequence 60 = 15 := 
by
  sorry

end sum_of_first_60_digits_l536_53681


namespace total_students_university_l536_53677

theorem total_students_university :
  ∀ (sample_size freshmen sophomores other_sample other_total total_students : ℕ),
  sample_size = 500 →
  freshmen = 200 →
  sophomores = 100 →
  other_sample = 200 →
  other_total = 3000 →
  total_students = (other_total * sample_size) / other_sample →
  total_students = 7500 :=
by
  intros sample_size freshmen sophomores other_sample other_total total_students
  sorry

end total_students_university_l536_53677


namespace cube_of_square_is_15625_l536_53619

/-- The third smallest prime number is 5 --/
def third_smallest_prime := 5

/-- The square of 5 is 25 --/
def square_of_third_smallest_prime := third_smallest_prime ^ 2

/-- The cube of the square of the third smallest prime number is 15625 --/
def cube_of_square_of_third_smallest_prime := square_of_third_smallest_prime ^ 3

theorem cube_of_square_is_15625 : cube_of_square_of_third_smallest_prime = 15625 := by
  sorry

end cube_of_square_is_15625_l536_53619


namespace math_problem_l536_53650

theorem math_problem :
  (2^8 + 4^5) * (1^3 - (-1)^3)^2 = 5120 :=
by
  sorry

end math_problem_l536_53650


namespace exponentiation_division_l536_53688

variable {a : ℝ} (h1 : (a^2)^3 = a^6) (h2 : a^6 / a^2 = a^4)

theorem exponentiation_division : (a^2)^3 / a^2 = a^4 := 
by 
  sorry

end exponentiation_division_l536_53688


namespace mike_ride_distance_l536_53684

/-- 
Mike took a taxi to the airport and paid a starting amount plus $0.25 per mile. 
Annie took a different route to the airport and paid the same starting amount plus $5.00 in bridge toll fees plus $0.25 per mile. 
Each was charged exactly the same amount, and Annie's ride was 26 miles. 
Prove that Mike's ride was 46 miles given his starting amount was $2.50.
-/
theorem mike_ride_distance
  (S C A_miles : ℝ)                  -- S: starting amount, C: cost per mile, A_miles: Annie's ride distance
  (bridge_fee total_cost : ℝ)        -- bridge_fee: Annie's bridge toll fee, total_cost: total cost for both
  (M : ℝ)                            -- M: Mike's ride distance
  (hS : S = 2.5)
  (hC : C = 0.25)
  (hA_miles : A_miles = 26)
  (h_bridge_fee : bridge_fee = 5)
  (h_total_cost_equal : total_cost = S + bridge_fee + (C * A_miles))
  (h_total_cost_mike : total_cost = S + (C * M)) :
  M = 46 :=
by 
  sorry

end mike_ride_distance_l536_53684


namespace exists_monochromatic_triangle_in_K6_l536_53699

/-- In a complete graph with 6 vertices where each edge is colored either red or blue,
    there exists a set of 3 vertices such that the edges joining them are all the same color. -/
theorem exists_monochromatic_triangle_in_K6 (color : Fin 6 → Fin 6 → Prop)
  (h : ∀ {i j : Fin 6}, i ≠ j → (color i j ∨ ¬ color i j)) :
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
  ((color i j ∧ color j k ∧ color k i) ∨ (¬ color i j ∧ ¬ color j k ∧ ¬ color k i)) :=
by
  sorry

end exists_monochromatic_triangle_in_K6_l536_53699


namespace integer_ratio_zero_l536_53690

theorem integer_ratio_zero
  (A B : ℤ)
  (h : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -1 → (A / (x - 3 : ℝ) + B / (x ^ 2 + 2 * x + 1) = (x ^ 3 - x ^ 2 + 3 * x + 1) / (x ^ 3 - x - 3))) :
  B / A = 0 :=
sorry

end integer_ratio_zero_l536_53690


namespace number_of_boys_in_first_group_l536_53662

-- Define the daily work ratios
variables (M B : ℝ) (h_ratio : M = 2 * B)

-- Define the number of boys in the first group
variable (x : ℝ)

-- Define the conditions provided by the problem
variables (h1 : 5 * (12 * M + x * B) = 4 * (13 * M + 24 * B))

-- State the theorem and include the correct answer
theorem number_of_boys_in_first_group (M B : ℝ) (h_ratio : M = 2 * B) (x : ℝ)
    (h1 : 5 * (12 * M + x * B) = 4 * (13 * M + 24 * B)) 
    : x = 16 := 
by 
    sorry

end number_of_boys_in_first_group_l536_53662


namespace min_value_a_over_b_l536_53682

theorem min_value_a_over_b (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 2 * Real.sqrt a + b = 1) : ∃ c, c = 0 := 
by
  -- We need to show that the minimum value of a / b is 0 
  sorry

end min_value_a_over_b_l536_53682


namespace evaluate_expression_l536_53648

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem evaluate_expression :
  (4 / log_base 5 (2500^3) + 2 / log_base 2 (2500^3) = 1 / 3) := by
  sorry

end evaluate_expression_l536_53648


namespace jacob_additional_money_needed_l536_53623

/-- Jacob's total trip cost -/
def trip_cost : ℕ := 5000

/-- Jacob's hourly wage -/
def hourly_wage : ℕ := 20

/-- Jacob's working hours -/
def working_hours : ℕ := 10

/-- Income from job -/
def job_income : ℕ := hourly_wage * working_hours

/-- Price per cookie -/
def cookie_price : ℕ := 4

/-- Number of cookies sold -/
def cookies_sold : ℕ := 24

/-- Income from cookies -/
def cookie_income : ℕ := cookie_price * cookies_sold

/-- Lottery ticket cost -/
def lottery_ticket_cost : ℕ := 10

/-- Lottery win amount -/
def lottery_win : ℕ := 500

/-- Money received from each sister -/
def sister_gift : ℕ := 500

/-- Total income from job and cookies -/
def income_without_expenses : ℕ := job_income + cookie_income

/-- Income after lottery ticket purchase -/
def income_after_ticket : ℕ := income_without_expenses - lottery_ticket_cost

/-- Total income after lottery win -/
def income_with_lottery : ℕ := income_after_ticket + lottery_win

/-- Total gift from sisters -/
def total_sisters_gift : ℕ := 2 * sister_gift

/-- Total money Jacob has -/
def total_money : ℕ := income_with_lottery + total_sisters_gift

/-- Additional amount needed by Jacob -/
def additional_needed : ℕ := trip_cost - total_money

theorem jacob_additional_money_needed : additional_needed = 3214 := by
  sorry

end jacob_additional_money_needed_l536_53623


namespace hours_spent_writing_l536_53601

-- Define the rates at which Jacob and Nathan write
def Nathan_rate : ℕ := 25        -- Nathan writes 25 letters per hour
def Jacob_rate : ℕ := 2 * Nathan_rate  -- Jacob writes twice as fast as Nathan

-- Define the combined rate
def combined_rate : ℕ := Nathan_rate + Jacob_rate

-- Define the total letters written and the hours spent
def total_letters : ℕ := 750
def hours_spent : ℕ := total_letters / combined_rate

-- The theorem to prove
theorem hours_spent_writing : hours_spent = 10 :=
by 
  -- Placeholder for the proof
  sorry

end hours_spent_writing_l536_53601


namespace complement_union_l536_53637

theorem complement_union (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {2, 3, 5})
  (hN : N = {4, 5}) :
  U \ (M ∪ N) = {1, 6} :=
by
  sorry

end complement_union_l536_53637


namespace missing_angle_measure_l536_53654

theorem missing_angle_measure (n : ℕ) (h : 180 * (n - 2) = 3240 + 2 * (180 * (n - 2)) / n) : 
  (180 * (n - 2)) / n = 166 := 
by 
  sorry

end missing_angle_measure_l536_53654


namespace trains_crossing_time_l536_53622

theorem trains_crossing_time
  (L : ℕ) (t1 t2 : ℕ)
  (h_length : L = 120)
  (h_t1 : t1 = 10)
  (h_t2 : t2 = 15) :
  let V1 := L / t1
  let V2 := L / t2
  let V_relative := V1 + V2
  let D := L + L
  (D / V_relative) = 12 :=
by
  sorry

end trains_crossing_time_l536_53622


namespace length_of_common_chord_l536_53687

-- Problem conditions
variables (r : ℝ) (h : r = 15)

-- Statement to prove
theorem length_of_common_chord : 2 * (r / 2 * Real.sqrt 3) = 15 * Real.sqrt 3 :=
by
  sorry

end length_of_common_chord_l536_53687


namespace greatest_radius_l536_53686

theorem greatest_radius (r : ℕ) : (π * (r : ℝ)^2 < 75 * π) ↔ r ≤ 8 := 
by
  sorry

end greatest_radius_l536_53686


namespace sum_xy_22_l536_53653

theorem sum_xy_22 (x y : ℕ) (h1 : 0 < x) (h2 : x < 25) (h3 : 0 < y) (h4 : y < 25) 
  (h5 : x + y + x * y = 118) : x + y = 22 :=
sorry

end sum_xy_22_l536_53653


namespace Sara_spent_on_hotdog_l536_53638

-- Define the given constants
def totalCost : ℝ := 10.46
def costSalad : ℝ := 5.10

-- Define the value we need to prove
def costHotdog : ℝ := 5.36

-- Statement to prove
theorem Sara_spent_on_hotdog : totalCost - costSalad = costHotdog := by
  sorry

end Sara_spent_on_hotdog_l536_53638


namespace smallest_m_divisible_by_15_l536_53665

noncomputable def largest_prime_with_2023_digits : ℕ := sorry

theorem smallest_m_divisible_by_15 :
  ∃ m : ℕ, m > 0 ∧ (largest_prime_with_2023_digits ^ 2 - m) % 15 = 0 ∧ m = 1 :=
  sorry

end smallest_m_divisible_by_15_l536_53665


namespace vova_last_grades_l536_53649

theorem vova_last_grades (grades : Fin 19 → ℕ) 
  (first_four_2s : ∀ i : Fin 4, grades i = 2)
  (all_combinations_once : ∀ comb : Fin 4 → ℕ, 
    (∃ (start : Fin (19-3)), ∀ j : Fin 4, grades (start + j) = comb j) ∧
    (∀ i j : Fin (19-3), 
      (∀ k : Fin 4, grades (i + k) = grades (j + k)) → i = j)) :
  ∀ i : Fin 4, grades (15 + i) = if i = 0 then 3 else 2 :=
by
  sorry

end vova_last_grades_l536_53649


namespace solve_quadratic_l536_53632

theorem solve_quadratic (x : ℝ) : (x^2 + 2*x = 0) ↔ (x = 0 ∨ x = -2) :=
by
  sorry

end solve_quadratic_l536_53632


namespace metallic_sheet_width_l536_53606

-- Defining the conditions
def sheet_length := 48
def cut_square_side := 8
def box_volume := 5632

-- Main theorem statement
theorem metallic_sheet_width 
    (L : ℕ := sheet_length)
    (s : ℕ := cut_square_side)
    (V : ℕ := box_volume) :
    (32 * (w - 2 * s) * s = V) → (w = 38) := by
  intros h1
  sorry

end metallic_sheet_width_l536_53606


namespace mike_arcade_ratio_l536_53683

theorem mike_arcade_ratio :
  ∀ (weekly_pay food_cost hourly_rate play_minutes : ℕ),
    weekly_pay = 100 →
    food_cost = 10 →
    hourly_rate = 8 →
    play_minutes = 300 →
    (food_cost + (play_minutes / 60) * hourly_rate) / weekly_pay = 1 / 2 := 
by
  intros weekly_pay food_cost hourly_rate play_minutes h1 h2 h3 h4
  sorry

end mike_arcade_ratio_l536_53683


namespace logan_gas_expense_l536_53608

-- Definitions based on conditions:
def annual_salary := 65000
def rent_expense := 20000
def grocery_expense := 5000
def desired_savings := 42000
def new_income_target := annual_salary + 10000

-- The property to be proved:
theorem logan_gas_expense : 
  ∀ (gas_expense : ℕ), 
  new_income_target - desired_savings = rent_expense + grocery_expense + gas_expense → 
  gas_expense = 8000 := 
by 
  sorry

end logan_gas_expense_l536_53608


namespace third_consecutive_even_number_l536_53639

theorem third_consecutive_even_number (n : ℕ) (h : n % 2 = 0) (sum_eq : n + (n + 2) + (n + 4) = 246) : (n + 4) = 84 :=
by
  -- This statement sets up the conditions and the goal of the proof.
  sorry

end third_consecutive_even_number_l536_53639


namespace problem1_problem2_l536_53659

def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 1|

theorem problem1 (x : ℝ) : f x ≥ 4 ↔ x ≤ -4/3 ∨ x ≥ 4/3 := 
  sorry

theorem problem2 (a : ℝ) : (∀ x : ℝ, f x > a) ↔ a < 3/2 := 
  sorry

end problem1_problem2_l536_53659


namespace line_properties_l536_53651

theorem line_properties (m x_intercept : ℝ) (y_intercept point_on_line : ℝ × ℝ) :
  m = -4 → x_intercept = -3 → y_intercept = (0, -12) → point_on_line = (2, -20) → 
    (∀ x y, y = -4 * x - 12 → (y_intercept = (0, y) ∧ point_on_line = (x, y))) := 
by
  sorry

end line_properties_l536_53651


namespace solve_for_m_l536_53661

theorem solve_for_m 
  (m : ℝ) 
  (h : (m - 1) * 2^2 + 3 * 2 - 5 * m + 4 = 0) : m = 6 :=
sorry

end solve_for_m_l536_53661


namespace arithmetic_sequence_y_l536_53680

theorem arithmetic_sequence_y :
  let a := 3^3
  let c := 3^5
  let y := (a + c) / 2
  y = 135 :=
by
  let a := 27
  let c := 243
  let y := (a + c) / 2
  show y = 135
  sorry

end arithmetic_sequence_y_l536_53680


namespace y_satisfies_quadratic_l536_53600

theorem y_satisfies_quadratic (x y : ℝ) 
  (h1 : 2 * x^2 + 6 * x + 5 * y + 1 = 0)
  (h2 : 2 * x + y + 3 = 0) : y^2 + 10 * y - 7 = 0 := 
sorry

end y_satisfies_quadratic_l536_53600


namespace amy_seeds_l536_53652

-- Define the conditions
def bigGardenSeeds : Nat := 47
def smallGardens : Nat := 9
def seedsPerSmallGarden : Nat := 6

-- Define the total seeds calculation
def totalSeeds := bigGardenSeeds + smallGardens * seedsPerSmallGarden

-- The theorem to be proved
theorem amy_seeds : totalSeeds = 101 := by
  sorry

end amy_seeds_l536_53652


namespace singers_in_fifth_verse_l536_53696

theorem singers_in_fifth_verse (choir : ℕ) (absent : ℕ) (participating : ℕ) 
(half_first_verse : ℕ) (third_second_verse : ℕ) (quarter_third_verse : ℕ) 
(fifth_fourth_verse : ℕ) (late_singers : ℕ) :
  choir = 70 → 
  absent = 10 → 
  participating = choir - absent →
  half_first_verse = participating / 2 → 
  third_second_verse = (participating - half_first_verse) / 3 →
  quarter_third_verse = (participating - half_first_verse - third_second_verse) / 4 →
  fifth_fourth_verse = (participating - half_first_verse - third_second_verse - quarter_third_verse) / 5 →
  late_singers = 5 →
  participating = 60 :=
by sorry

end singers_in_fifth_verse_l536_53696


namespace inequality_holds_l536_53610

variables (a b c : ℝ)

theorem inequality_holds 
  (h1 : a > b) : 
  a / (c^2 + 1) > b / (c^2 + 1) :=
sorry

end inequality_holds_l536_53610


namespace fraction_study_only_japanese_l536_53635

variable (J : ℕ)

def seniors := 2 * J
def sophomores := (3 / 4) * J

def seniors_study_japanese := (3 / 8) * seniors J
def juniors_study_japanese := (1 / 4) * J
def sophomores_study_japanese := (2 / 5) * sophomores J

def seniors_study_both := (1 / 6) * seniors J
def juniors_study_both := (1 / 12) * J
def sophomores_study_both := (1 / 10) * sophomores J

def seniors_study_only_japanese := seniors_study_japanese J - seniors_study_both J
def juniors_study_only_japanese := juniors_study_japanese J - juniors_study_both J
def sophomores_study_only_japanese := sophomores_study_japanese J - sophomores_study_both J

def total_study_only_japanese := seniors_study_only_japanese J + juniors_study_only_japanese J + sophomores_study_only_japanese J
def total_students := J + seniors J + sophomores J

theorem fraction_study_only_japanese :
  (total_study_only_japanese J) / (total_students J) = 97 / 450 :=
by sorry

end fraction_study_only_japanese_l536_53635


namespace swap_values_l536_53698

theorem swap_values : ∀ (a b : ℕ), a = 3 → b = 2 → 
  (∃ c : ℕ, c = b ∧ (b = a ∧ (a = c ∨ a = 2 ∧ b = 3))) :=
by
  sorry

end swap_values_l536_53698


namespace inscribed_square_side_length_l536_53663

-- Define a right triangle
structure RightTriangle :=
  (PQ : ℝ)
  (QR : ℝ)
  (PR : ℝ)
  (is_right : PQ^2 + QR^2 = PR^2)

-- Define the triangle PQR
def trianglePQR : RightTriangle :=
  { PQ := 6, QR := 8, PR := 10, is_right := by norm_num }

-- Define the problem statement
theorem inscribed_square_side_length (t : ℝ) (h : RightTriangle) :
  t = 3 :=
  sorry

end inscribed_square_side_length_l536_53663


namespace Daniela_is_12_years_old_l536_53640

noncomputable def auntClaraAge : Nat := 60

noncomputable def evelinaAge : Nat := auntClaraAge / 3

noncomputable def fidelAge : Nat := evelinaAge - 6

noncomputable def caitlinAge : Nat := fidelAge / 2

noncomputable def danielaAge : Nat := evelinaAge - 8

theorem Daniela_is_12_years_old (h_auntClaraAge : auntClaraAge = 60)
                                (h_evelinaAge : evelinaAge = 60 / 3)
                                (h_fidelAge : fidelAge = (60 / 3) - 6)
                                (h_caitlinAge : caitlinAge = ((60 / 3) - 6) / 2)
                                (h_danielaAge : danielaAge = (60 / 3) - 8) :
  danielaAge = 12 := 
  sorry

end Daniela_is_12_years_old_l536_53640


namespace acetic_acid_molecular_weight_is_correct_l536_53617

def molecular_weight_acetic_acid : ℝ :=
  let carbon_weight := 12.01
  let hydrogen_weight := 1.008
  let oxygen_weight := 16.00
  let num_carbons := 2
  let num_hydrogens := 4
  let num_oxygens := 2
  num_carbons * carbon_weight + num_hydrogens * hydrogen_weight + num_oxygens * oxygen_weight

theorem acetic_acid_molecular_weight_is_correct : molecular_weight_acetic_acid = 60.052 :=
by 
  unfold molecular_weight_acetic_acid
  sorry

end acetic_acid_molecular_weight_is_correct_l536_53617


namespace octahedron_common_sum_is_39_l536_53647

-- Define the vertices of the regular octahedron with numbers from 1 to 12
def vertices : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the property that the sum of four numbers at the vertices of each triangle face is the same
def common_sum (faces : List (List ℕ)) (k : ℕ) : Prop :=
  ∀ face ∈ faces, face.sum = k

-- Define the faces of the regular octahedron
def faces : List (List ℕ) := [
  [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 5, 9, 6],
  [2, 6, 10, 7], [3, 7, 11, 8], [4, 8, 12, 5], [1, 9, 2, 10]
]

-- Prove that the common sum is 39
theorem octahedron_common_sum_is_39 : common_sum faces 39 :=
  sorry

end octahedron_common_sum_is_39_l536_53647


namespace percentage_of_fruits_in_good_condition_l536_53646

theorem percentage_of_fruits_in_good_condition :
  let total_oranges := 600
  let total_bananas := 400
  let rotten_oranges := (15 / 100.0) * total_oranges
  let rotten_bananas := (8 / 100.0) * total_bananas
  let good_condition_oranges := total_oranges - rotten_oranges
  let good_condition_bananas := total_bananas - rotten_bananas
  let total_fruits := total_oranges + total_bananas
  let total_fruits_in_good_condition := good_condition_oranges + good_condition_bananas
  let percentage_fruits_in_good_condition := (total_fruits_in_good_condition / total_fruits) * 100
  percentage_fruits_in_good_condition = 87.8 := sorry

end percentage_of_fruits_in_good_condition_l536_53646


namespace length_AB_slope_one_OA_dot_OB_const_l536_53672

open Real

def parabola (x y : ℝ) : Prop := y * y = 4 * x
def line_through_focus (x y : ℝ) (k : ℝ) : Prop := x = k * y + 1
def line_slope_one (x y : ℝ) : Prop := y = x - 1

theorem length_AB_slope_one {x1 x2 y1 y2 : ℝ} (hA : parabola x1 y1) (hB : parabola x2 y2) 
  (hL : line_slope_one x1 y1) (hL' : line_slope_one x2 y2) : abs (x1 - x2) + abs (y1 - y2) = 8 := 
by
  sorry

theorem OA_dot_OB_const {x1 x2 y1 y2 : ℝ} {k : ℝ} (hA : parabola x1 y1)
  (hB : parabola x2 y2) (hL : line_through_focus x1 y1 k) (hL' : line_through_focus x2 y2 k) :
  x1 * x2 + y1 * y2 = -3 :=
by
  sorry

end length_AB_slope_one_OA_dot_OB_const_l536_53672


namespace length_of_room_l536_53664

theorem length_of_room (L : ℝ) (w : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) (room_area : ℝ) :
  w = 12 →
  veranda_width = 2 →
  veranda_area = 144 →
  (L + 2 * veranda_width) * (w + 2 * veranda_width) - L * w = veranda_area →
  L = 20 :=
by
  intro h_w
  intro h_veranda_width
  intro h_veranda_area
  intro h_area_eq
  sorry

end length_of_room_l536_53664


namespace polygon_sides_l536_53636

theorem polygon_sides (n : ℕ) : 
  (180 * (n - 2) / 360 = 5 / 2) → n = 7 :=
by
  sorry

end polygon_sides_l536_53636


namespace cube_surface_area_proof_l536_53692

-- Conditions
def prism_volume : ℕ := 10 * 5 * 20
def cube_volume : ℕ := 1000
def edge_length_of_cube : ℕ := 10
def cube_surface_area (s : ℕ) : ℕ := 6 * s * s

-- Theorem Statement
theorem cube_surface_area_proof : cube_volume = prism_volume → cube_surface_area edge_length_of_cube = 600 := 
by
  intros h
  -- Proof goes here
  sorry

end cube_surface_area_proof_l536_53692


namespace bottle_caps_per_person_l536_53675

noncomputable def initial_caps : Nat := 150
noncomputable def rebecca_caps : Nat := 42
noncomputable def alex_caps : Nat := 2 * rebecca_caps
noncomputable def total_caps : Nat := initial_caps + rebecca_caps + alex_caps
noncomputable def number_of_people : Nat := 6

theorem bottle_caps_per_person : total_caps / number_of_people = 46 := by
  sorry

end bottle_caps_per_person_l536_53675


namespace total_tickets_sold_l536_53655

-- Definitions and conditions
def orchestra_ticket_price : ℕ := 12
def balcony_ticket_price : ℕ := 8
def total_revenue : ℕ := 3320
def ticket_difference : ℕ := 190

-- Variables
variables (x y : ℕ) -- x is the number of orchestra tickets, y is the number of balcony tickets

-- Statements of conditions
def revenue_eq : Prop := orchestra_ticket_price * x + balcony_ticket_price * y = total_revenue
def tickets_relation : Prop := y = x + ticket_difference

-- The proof problem statement
theorem total_tickets_sold (h1 : revenue_eq x y) (h2 : tickets_relation x y) : x + y = 370 :=
by
  sorry

end total_tickets_sold_l536_53655


namespace frequency_group_5_l536_53641

theorem frequency_group_5 (total_students : ℕ) (freq1 freq2 freq3 freq4 : ℕ)
  (h1 : total_students = 45)
  (h2 : freq1 = 12)
  (h3 : freq2 = 11)
  (h4 : freq3 = 9)
  (h5 : freq4 = 4) :
  ((total_students - (freq1 + freq2 + freq3 + freq4)) / total_students : ℚ) = 0.2 := 
sorry

end frequency_group_5_l536_53641


namespace find_common_ratio_of_geometric_sequence_l536_53674

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem find_common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : geometric_sequence a)
  (h_decreasing : ∀ n : ℕ, a n > a (n + 1))
  (h1 : a 1 * a 5 = 9)
  (h2 : a 2 + a 4 = 10) : 
  q = -1/3 :=
sorry

end find_common_ratio_of_geometric_sequence_l536_53674


namespace largest_tile_side_length_l536_53656

theorem largest_tile_side_length (w h : ℕ) (hw : w = 17) (hh : h = 23) : Nat.gcd w h = 1 := by
  -- Proof goes here
  sorry

end largest_tile_side_length_l536_53656


namespace find_y_l536_53660

theorem find_y (x y : ℝ) (h1 : x^2 = 2 * y - 6) (h2 : x = 7) : y = 55 / 2 :=
by
  sorry

end find_y_l536_53660


namespace vectors_parallel_l536_53634

def are_parallel (a b : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, b = k • a

theorem vectors_parallel :
  let a := (1, 2, -2)
  let b := (-2, -4, 4)
  are_parallel a b :=
by
  let a := (1, 2, -2)
  let b := (-2, -4, 4)
  -- Proof omitted
  sorry

end vectors_parallel_l536_53634


namespace meals_second_restaurant_l536_53628

theorem meals_second_restaurant (r1 r2 r3 total_weekly_meals : ℕ) 
    (H1 : r1 = 20) 
    (H3 : r3 = 50) 
    (H_total : total_weekly_meals = 770) : 
    (7 * r2) = 280 := 
by 
    sorry

example (r2 : ℕ) : (40 = r2) :=
    by sorry

end meals_second_restaurant_l536_53628


namespace road_length_l536_53633

theorem road_length (n : ℕ) (d : ℕ) (trees : ℕ) (intervals : ℕ) (L : ℕ) 
  (h1 : n = 10) 
  (h2 : d = 10) 
  (h3 : trees = 10) 
  (h4 : intervals = trees - 1) 
  (h5 : L = intervals * d) : 
  L = 90 :=
by
  sorry

end road_length_l536_53633


namespace trader_profit_percentage_l536_53621

theorem trader_profit_percentage (P : ℝ) (hP : 0 < P) :
  let bought_price := 0.90 * P
  let sold_price := 1.80 * bought_price
  let profit := sold_price - P
  let profit_percentage := (profit / P) * 100
  profit_percentage = 62 := 
by
  let bought_price := 0.90 * P
  let sold_price := 1.80 * bought_price
  let profit := sold_price - P
  let profit_percentage := (profit / P) * 100
  sorry

end trader_profit_percentage_l536_53621


namespace alex_jellybeans_l536_53658

theorem alex_jellybeans (n : ℕ) (h1 : n ≥ 200) (h2 : n % 17 = 15) : n = 202 :=
sorry

end alex_jellybeans_l536_53658


namespace recurring_decimal_of_division_l536_53620

theorem recurring_decimal_of_division (a b : ℤ) (h1 : a = 60) (h2 : b = 55) : (a : ℝ) / (b : ℝ) = 1.09090909090909090909090909090909 :=
by
  -- Import the necessary definitions and facts
  sorry

end recurring_decimal_of_division_l536_53620


namespace smallest_n_for_square_and_cube_l536_53695

theorem smallest_n_for_square_and_cube (n : ℕ) 
  (h1 : ∃ m : ℕ, 3 * n = m^2) 
  (h2 : ∃ k : ℕ, 5 * n = k^3) : 
  n = 675 :=
  sorry

end smallest_n_for_square_and_cube_l536_53695


namespace find_abc_triplet_l536_53614

theorem find_abc_triplet (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_order : a < b ∧ b < c) 
  (h_eqn : (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) = (a + b + c) / 2) :
  ∃ d : ℕ, d > 0 ∧ ((a = d ∧ b = 2 * d ∧ c = 3 * d) ∨ (a = d ∧ b = 3 * d ∧ c = 6 * d)) :=
  sorry

end find_abc_triplet_l536_53614


namespace proof_problem_l536_53678

def necessary_but_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem proof_problem (x : ℝ) :
  necessary_but_not_sufficient ((x+3)*(x-1) = 0) (x-1 = 0) :=
by
  sorry

end proof_problem_l536_53678


namespace rectangle_area_l536_53630

theorem rectangle_area
  (x y : ℝ) -- sides of the rectangle
  (h1 : 2 * x + 2 * y = 12)  -- perimeter
  (h2 : x^2 + y^2 = 25)  -- diagonal
  : x * y = 5.5 :=
sorry

end rectangle_area_l536_53630


namespace toys_produced_each_day_l536_53694

theorem toys_produced_each_day (total_weekly_production : ℕ) (days_worked_per_week : ℕ)
  (same_number_toys_each_day : Prop) : 
  total_weekly_production = 4340 → days_worked_per_week = 2 → 
  same_number_toys_each_day →
  (total_weekly_production / days_worked_per_week = 2170) :=
by
  intros h_production h_days h_same_toys
  -- proof skipped
  sorry

end toys_produced_each_day_l536_53694


namespace isosceles_triangle_area_l536_53685

theorem isosceles_triangle_area :
  ∀ (P Q R S : ℝ) (h1 : dist P Q = 26) (h2 : dist P R = 26) (h3 : dist Q R = 50),
  ∃ (area : ℝ), area = 25 * Real.sqrt 51 :=
by
  sorry

end isosceles_triangle_area_l536_53685


namespace find_A_time_l536_53644

noncomputable def work_rate_equations (W : ℝ) (A B C : ℝ) : Prop :=
  B + C = W / 2 ∧ A + B = W / 2 ∧ C = W / 3

theorem find_A_time {W A B C : ℝ} (h : work_rate_equations W A B C) :
  W / A = 3 :=
sorry

end find_A_time_l536_53644


namespace triangle_side_length_mod_l536_53666

theorem triangle_side_length_mod {a d x : ℕ} 
  (h_equilateral : ∃ (a : ℕ), 3 * a = 1 + d + x)
  (h_triangle : ∀ {a d x : ℕ}, 1 + d > x ∧ 1 + x > d ∧ d + x > 1)
  : d % 3 = 1 :=
by
  sorry

end triangle_side_length_mod_l536_53666


namespace spiders_loose_l536_53603

noncomputable def initial_birds : ℕ := 12
noncomputable def initial_puppies : ℕ := 9
noncomputable def initial_cats : ℕ := 5
noncomputable def initial_spiders : ℕ := 15
noncomputable def birds_sold : ℕ := initial_birds / 2
noncomputable def puppies_adopted : ℕ := 3
noncomputable def remaining_puppies : ℕ := initial_puppies - puppies_adopted
noncomputable def remaining_cats : ℕ := initial_cats
noncomputable def total_remaining_animals_except_spiders : ℕ := birds_sold + remaining_puppies + remaining_cats
noncomputable def total_animals_left : ℕ := 25
noncomputable def remaining_spiders : ℕ := total_animals_left - total_remaining_animals_except_spiders
noncomputable def spiders_went_loose : ℕ := initial_spiders - remaining_spiders

theorem spiders_loose : spiders_went_loose = 7 := by
  sorry

end spiders_loose_l536_53603


namespace find_k_l536_53625

def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x + y - 2 = 0
def quadrilateral_has_circumscribed_circle (k : ℝ) : Prop :=
  ∀ x y : ℝ, line1 x y → line2 k x y →
  k = -3

theorem find_k (k : ℝ) (x y : ℝ) : 
  (line1 x y) ∧ (line2 k x y) → quadrilateral_has_circumscribed_circle k :=
by 
  sorry

end find_k_l536_53625


namespace sufficient_not_necessary_condition_l536_53613

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x > 4 → x^2 - 4 * x > 0) ∧ ¬ (x^2 - 4 * x > 0 → x > 4) :=
sorry

end sufficient_not_necessary_condition_l536_53613


namespace max_ages_within_two_std_dev_l536_53697

def average_age : ℕ := 30
def std_dev : ℕ := 12
def lower_limit : ℕ := average_age - 2 * std_dev
def upper_limit : ℕ := average_age + 2 * std_dev
def max_different_ages : ℕ := upper_limit - lower_limit + 1

theorem max_ages_within_two_std_dev
  (avg : ℕ) (std : ℕ) (h_avg : avg = average_age) (h_std : std = std_dev)
  : max_different_ages = 49 :=
by
  sorry

end max_ages_within_two_std_dev_l536_53697


namespace die_face_never_touches_board_l536_53657

theorem die_face_never_touches_board : 
  ∃ (cube : Type) (roll : cube → cube) (occupied : Fin 8 × Fin 8 → cube → Prop),
    (∀ p : Fin 8 × Fin 8, ∃ c : cube, occupied p c) ∧ 
    (∃ f : cube, ¬ (∃ p : Fin 8 × Fin 8, occupied p f)) :=
by sorry

end die_face_never_touches_board_l536_53657


namespace range_of_a_l536_53689

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∧ 
  (∀ x > 1, f x = a^x) ∧ 
  (∀ x ≤ 1, f x = (4 - (a / 2)) * x + 2) → 
  4 ≤ a ∧ a < 8 :=
by
  sorry

end range_of_a_l536_53689


namespace percentage_of_tip_l536_53629

-- Given conditions
def steak_cost : ℝ := 20
def drink_cost : ℝ := 5
def total_cost_before_tip : ℝ := 2 * (steak_cost + drink_cost)
def billy_tip_payment : ℝ := 8
def billy_tip_coverage : ℝ := 0.80

-- Required to prove
theorem percentage_of_tip : ∃ P : ℝ, (P = (billy_tip_payment / (billy_tip_coverage * total_cost_before_tip)) * 100) ∧ P = 20 := 
by {
  sorry
}

end percentage_of_tip_l536_53629


namespace digit_pairs_for_divisibility_by_36_l536_53671

theorem digit_pairs_for_divisibility_by_36 (A B : ℕ) :
  (0 ≤ A) ∧ (A ≤ 9) ∧ (0 ≤ B) ∧ (B ≤ 9) ∧
  (∃ k4 k9 : ℕ, (10 * 5 + B = 4 * k4) ∧ (20 + A + B = 9 * k9)) ↔ 
  ((A = 5 ∧ B = 2) ∨ (A = 1 ∧ B = 6)) :=
by sorry

end digit_pairs_for_divisibility_by_36_l536_53671


namespace pies_sold_each_day_l536_53668

theorem pies_sold_each_day (total_pies : ℕ) (days_in_week : ℕ) (h1 : total_pies = 56) (h2 : days_in_week = 7) :
  (total_pies / days_in_week = 8) :=
by
exact sorry

end pies_sold_each_day_l536_53668


namespace final_price_of_jacket_l536_53676

noncomputable def original_price : ℝ := 240
noncomputable def initial_discount : ℝ := 0.6
noncomputable def additional_discount : ℝ := 0.25

theorem final_price_of_jacket :
  let price_after_initial_discount := original_price * (1 - initial_discount)
  let final_price := price_after_initial_discount * (1 - additional_discount)
  final_price = 72 := 
by
  sorry

end final_price_of_jacket_l536_53676


namespace max_value_of_f_l536_53602

open Real

noncomputable def f (x : ℝ) : ℝ := -x - 9 / x + 18

theorem max_value_of_f : ∀ x > 0, f x ≤ 12 :=
by
  sorry

end max_value_of_f_l536_53602


namespace inequlity_for_k_one_smallest_k_l536_53626

noncomputable def triangle_sides (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

theorem inequlity_for_k_one (a b c : ℝ) (h : triangle_sides a b c) :
  a^3 + b^3 + c^3 < (a + b + c) * (a * b + b * c + c * a) :=
sorry

theorem smallest_k (a b c k : ℝ) (h : triangle_sides a b c) (hk : k = 1) :
  a^3 + b^3 + c^3 < k * (a + b + c) * (a * b + b * c + c * a) :=
sorry

end inequlity_for_k_one_smallest_k_l536_53626


namespace angleina_speed_from_grocery_to_gym_l536_53645

variable (v : ℝ) (h1 : 720 / v - 40 = 240 / v)

theorem angleina_speed_from_grocery_to_gym : 2 * v = 24 :=
by
  sorry

end angleina_speed_from_grocery_to_gym_l536_53645


namespace equidistant_point_on_x_axis_l536_53679

theorem equidistant_point_on_x_axis (x : ℝ) (A B : ℝ × ℝ)
  (hA : A = (-3, 0)) (hB : B = (3, 5)) :
  (Real.sqrt ((x - (-3))^2)) = (Real.sqrt ((x - 3)^2 + 25)) →
  x = 25 / 12 := 
by 
  sorry

end equidistant_point_on_x_axis_l536_53679


namespace rationalize_denominator_correct_l536_53667

noncomputable def rationalize_denominator : Prop :=
  (50 + Real.sqrt 8) / (Real.sqrt 50 + Real.sqrt 8) = (50 * (Real.sqrt 50 - Real.sqrt 8) + 12) / 42

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l536_53667


namespace rationalize_denominator_l536_53691

theorem rationalize_denominator :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) := by
  sorry

end rationalize_denominator_l536_53691


namespace harry_total_cost_in_silver_l536_53670

def cost_of_spellbooks_in_gold := 5 * 5
def cost_of_potion_kits_in_silver := 3 * 20
def cost_of_owl_in_gold := 28
def gold_to_silver := 9

def cost_in_silver :=
  (cost_of_spellbooks_in_gold + cost_of_owl_in_gold) * gold_to_silver + cost_of_potion_kits_in_silver

theorem harry_total_cost_in_silver : cost_in_silver = 537 := by
  sorry

end harry_total_cost_in_silver_l536_53670


namespace find_p_q_r_sum_l536_53615

theorem find_p_q_r_sum (p q r : ℕ) (hpq_rel_prime : Nat.gcd p q = 1) (hq_nonzero : q ≠ 0) 
  (h1 : ∃ t, (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4) 
  (h2 : ∃ t, (1 - Real.sin t) * (1 - Real.cos t) = p / q - Real.sqrt r) : 
  p + q + r = 7 :=
sorry

end find_p_q_r_sum_l536_53615


namespace length_of_bridge_correct_l536_53609

open Real

noncomputable def length_of_bridge (length_of_train : ℝ) (time_to_cross : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed := speed_kmph * (1000 / 3600)
  let total_distance := speed * time_to_cross
  total_distance - length_of_train

theorem length_of_bridge_correct :
  length_of_bridge 200 34.997200223982084 36 = 149.97200223982084 := by
  sorry

end length_of_bridge_correct_l536_53609


namespace vertex_is_correct_l536_53611

-- Define the equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 10 * y + 4 * x + 9 = 0

-- The vertex of the parabola
def vertex_of_parabola : ℝ × ℝ := (4, -5)

-- The theorem stating that the given vertex satisfies the parabola equation
theorem vertex_is_correct : 
  parabola_equation vertex_of_parabola.1 vertex_of_parabola.2 :=
sorry

end vertex_is_correct_l536_53611


namespace existence_of_function_values_around_k_l536_53673

-- Define the function f(n, m) with the given properties
def is_valid_function (f : ℤ × ℤ → ℤ) : Prop :=
  ∀ n m : ℤ, f (n, m) = (f (n-1, m) + f (n+1, m) + f (n, m-1) + f (n, m+1)) / 4

-- Theorem to prove the existence of such a function
theorem existence_of_function :
  ∃ (f : ℤ × ℤ → ℤ), is_valid_function f :=
sorry

-- Theorem to prove that for any k in ℤ, f(n, m) has values both greater and less than k
theorem values_around_k (k : ℤ) :
  ∃ (f : ℤ × ℤ → ℤ), is_valid_function f ∧ (∃ n1 m1 n2 m2, f (n1, m1) > k ∧ f (n2, m2) < k) :=
sorry

end existence_of_function_values_around_k_l536_53673


namespace smallest_triangle_perimeter_consecutive_even_l536_53607

theorem smallest_triangle_perimeter_consecutive_even :
  ∃ (a b c : ℕ), a = 2 ∧ b = 4 ∧ c = 6 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (a + b + c = 12) :=
by {
  sorry
}

end smallest_triangle_perimeter_consecutive_even_l536_53607


namespace trains_meet_at_distance_360_km_l536_53612

-- Define the speeds of the trains
def speed_A : ℕ := 30 -- speed of train A in kmph
def speed_B : ℕ := 40 -- speed of train B in kmph
def speed_C : ℕ := 60 -- speed of train C in kmph

-- Define the head starts in hours for trains A and B
def head_start_A : ℕ := 9 -- head start for train A in hours
def head_start_B : ℕ := 3 -- head start for train B in hours

-- Define the distances traveled by trains A and B by the time train C starts at 6 p.m.
def distance_A_start : ℕ := speed_A * head_start_A -- distance traveled by train A by 6 p.m.
def distance_B_start : ℕ := speed_B * head_start_B -- distance traveled by train B by 6 p.m.

-- The formula to calculate the distance after t hours from 6 p.m. for each train
def distance_A (t : ℕ) : ℕ := distance_A_start + speed_A * t
def distance_B (t : ℕ) : ℕ := distance_B_start + speed_B * t
def distance_C (t : ℕ) : ℕ := speed_C * t

-- Problem statement to prove the point where all three trains meet
theorem trains_meet_at_distance_360_km : ∃ t : ℕ, distance_A t = 360 ∧ distance_B t = 360 ∧ distance_C t = 360 := by
  sorry

end trains_meet_at_distance_360_km_l536_53612


namespace find_m_for_eccentric_ellipse_l536_53631

theorem find_m_for_eccentric_ellipse (m : ℝ) : 
  (∀ x y : ℝ, (x^2)/5 + (y^2)/m = 1) ∧
  (∀ e : ℝ, e = (Real.sqrt 10)/5) → 
  (m = 25/3 ∨ m = 3) := sorry

end find_m_for_eccentric_ellipse_l536_53631


namespace fraction_simplification_l536_53669

theorem fraction_simplification :
  ( (5^1004)^4 - (5^1002)^4 ) / ( (5^1003)^4 - (5^1001)^4 ) = 25 := by
  sorry

end fraction_simplification_l536_53669


namespace quotient_of_poly_div_l536_53693

theorem quotient_of_poly_div :
  (10 * X^4 - 5 * X^3 + 3 * X^2 + 11 * X - 6) / (5 * X^2 + 7) =
  2 * X^2 - X - (11 / 5) :=
sorry

end quotient_of_poly_div_l536_53693


namespace total_balls_in_box_l536_53643

theorem total_balls_in_box (red blue yellow total : ℕ) 
  (h1 : 2 * blue = 3 * red)
  (h2 : 3 * yellow = 4 * red) 
  (h3 : yellow = 40)
  (h4 : red + blue + yellow = total) : total = 90 :=
sorry

end total_balls_in_box_l536_53643


namespace difference_not_divisible_by_1976_l536_53605

theorem difference_not_divisible_by_1976 (A B : ℕ) (hA : 100 ≤ A) (hA' : A < 1000) (hB : 100 ≤ B) (hB' : B < 1000) (h : A ≠ B) :
  ¬ (1976 ∣ (1000 * A + B - (1000 * B + A))) :=
by
  sorry

end difference_not_divisible_by_1976_l536_53605
