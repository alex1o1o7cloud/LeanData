import Mathlib

namespace cookie_sales_l2216_221685

theorem cookie_sales (n : ℕ) (h1 : 1 ≤ n - 11) (h2 : 1 ≤ n - 2) (h3 : (n - 11) + (n - 2) < n) : n = 12 :=
sorry

end cookie_sales_l2216_221685


namespace rows_seating_exactly_10_people_exists_l2216_221650

theorem rows_seating_exactly_10_people_exists :
  ∃ y x : ℕ, 73 = 10 * y + 9 * x ∧ (73 - 10 * y) % 9 = 0 := 
sorry

end rows_seating_exactly_10_people_exists_l2216_221650


namespace adam_clothing_ratio_l2216_221665

-- Define the initial amount of clothing Adam took out
def initial_clothing_adam : ℕ := 4 + 4 + 8 + 20

-- Define the number of friends donating the same amount of clothing as Adam
def number_of_friends : ℕ := 3

-- Define the total number of clothes being donated
def total_donated_clothes : ℕ := 126

-- Define the ratio of the clothes Adam is keeping to the clothes he initially took out
def ratio_kept_to_initial (initial_clothing: ℕ) (total_donated: ℕ) (kept: ℕ) : Prop :=
  kept * initial_clothing = 0

-- Theorem statement
theorem adam_clothing_ratio :
  ratio_kept_to_initial initial_clothing_adam total_donated_clothes 0 :=
by 
  sorry

end adam_clothing_ratio_l2216_221665


namespace bananas_count_l2216_221653

/-- Elias bought some bananas and ate 1 of them. 
    After eating, he has 11 bananas left.
    Prove that Elias originally bought 12 bananas. -/
theorem bananas_count (x : ℕ) (h1 : x - 1 = 11) : x = 12 := by
  sorry

end bananas_count_l2216_221653


namespace imaginary_number_m_l2216_221641

theorem imaginary_number_m (m : ℝ) : 
  (∀ Z, Z = (m + 2 * Complex.I) / (1 + Complex.I) → Z.im = 0 → Z.re = 0) → m = -2 :=
by
  sorry

end imaginary_number_m_l2216_221641


namespace initial_temperature_l2216_221600

theorem initial_temperature (T_initial : ℝ) 
  (heating_rate : ℝ) (cooling_rate : ℝ) (total_time : ℝ) 
  (T_heat : ℝ) (T_cool : ℝ) (T_target : ℝ) (T_final : ℝ) 
  (h1 : heating_rate = 5) (h2 : cooling_rate = 7)
  (h3 : T_target = 240) (h4 : T_final = 170) 
  (h5 : total_time = 46)
  (h6 : T_cool = (T_target - T_final) / cooling_rate)
  (h7: total_time = T_heat + T_cool)
  (h8 : T_heat = (T_target - T_initial) / heating_rate) :
  T_initial = 60 :=
by
  -- Proof yet to be filled in
  sorry

end initial_temperature_l2216_221600


namespace cost_of_chairs_l2216_221647

-- Given conditions
def total_spent : ℕ := 56
def cost_of_table : ℕ := 34
def number_of_chairs : ℕ := 2

-- The target definition
def cost_of_one_chair : ℕ := 11

-- Statement to prove
theorem cost_of_chairs (N : ℕ) (T : ℕ) (C : ℕ) (x : ℕ) (hN : N = total_spent) (hT : T = cost_of_table) (hC : C = number_of_chairs) : x = cost_of_one_chair ↔ N - T = C * x :=
by
  sorry

end cost_of_chairs_l2216_221647


namespace price_of_33_kgs_l2216_221636

theorem price_of_33_kgs (l q : ℝ) 
  (h1 : l * 20 = 100) 
  (h2 : l * 30 + q * 6 = 186) : 
  l * 30 + q * 3 = 168 := 
by
  sorry

end price_of_33_kgs_l2216_221636


namespace value_of_x_plus_y_l2216_221681

theorem value_of_x_plus_y (x y : ℚ) (h1 : 1 / x + 1 / y = 5) (h2 : 1 / x - 1 / y = -9) : x + y = -5 / 14 := sorry

end value_of_x_plus_y_l2216_221681


namespace ratio_volumes_of_spheres_l2216_221617

theorem ratio_volumes_of_spheres (r R : ℝ) (hratio : (4 * π * r^2) / (4 * π * R^2) = 4 / 9) :
    (4 / 3 * π * r^3) / (4 / 3 * π * R^3) = 8 / 27 := 
by {
  sorry
}

end ratio_volumes_of_spheres_l2216_221617


namespace fundraiser_total_money_l2216_221613

def number_of_items (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student : ℕ) : ℕ :=
  (students1 * brownies_per_student) + (students2 * cookies_per_student) + (students3 * donuts_per_student)

def total_money_raised (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) : ℕ :=
  number_of_items students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student * price_per_item

theorem fundraiser_total_money (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) :
  students1 = 30 → students2 = 20 → students3 = 15 → brownies_per_student = 12 → cookies_per_student = 24 → donuts_per_student = 12 → price_per_item = 2 → 
  total_money_raised students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item = 2040 :=
  by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end fundraiser_total_money_l2216_221613


namespace square_land_plot_area_l2216_221646

theorem square_land_plot_area (side_length : ℕ) (h1 : side_length = 40) : side_length * side_length = 1600 :=
by
  sorry

end square_land_plot_area_l2216_221646


namespace novel_corona_high_students_l2216_221676

theorem novel_corona_high_students (students_know_it_all students_karen_high total_students students_novel_corona : ℕ)
  (h1 : students_know_it_all = 50)
  (h2 : students_karen_high = 3 / 5 * students_know_it_all)
  (h3 : total_students = 240)
  (h4 : students_novel_corona = total_students - (students_know_it_all + students_karen_high))
  : students_novel_corona = 160 :=
sorry

end novel_corona_high_students_l2216_221676


namespace find_a5_l2216_221649

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

-- Given conditions
variable (a : ℕ → ℝ)
variable (h_arith : arithmetic_sequence a)
variable (h_a1 : a 0 = 2)
variable (h_sum : a 1 + a 3 = 8)

-- The target question
theorem find_a5 : a 4 = 6 :=
by
  sorry

end find_a5_l2216_221649


namespace find_value_of_y_l2216_221671

variable (p y : ℝ)
variable (h1 : p > 45)
variable (h2 : p * p / 100 = (2 * p / 300) * (p + y))

theorem find_value_of_y (h1 : p > 45) (h2 : p * p / 100 = (2 * p / 300) * (p + y)) : y = p / 2 :=
sorry

end find_value_of_y_l2216_221671


namespace number_of_blue_candles_l2216_221662

-- Conditions
def grandfather_age : ℕ := 79
def yellow_candles : ℕ := 27
def red_candles : ℕ := 14
def total_candles : ℕ := grandfather_age
def yellow_red_candles : ℕ := yellow_candles + red_candles
def blue_candles : ℕ := total_candles - yellow_red_candles

-- Proof statement
theorem number_of_blue_candles : blue_candles = 38 :=
by
  -- sorry indicates the proof is omitted
  sorry

end number_of_blue_candles_l2216_221662


namespace workshop_total_workers_l2216_221628

noncomputable def average_salary_of_all (W : ℕ) : ℝ := 8000
noncomputable def average_salary_of_technicians : ℝ := 12000
noncomputable def average_salary_of_non_technicians : ℝ := 6000

theorem workshop_total_workers
    (W : ℕ)
    (T : ℕ := 7)
    (N : ℕ := W - T)
    (h1 : (T + N) = W)
    (h2 : average_salary_of_all W = 8000)
    (h3 : average_salary_of_technicians = 12000)
    (h4 : average_salary_of_non_technicians = 6000)
    (h5 : (7 * 12000) + (N * 6000) = (7 + N) * 8000) :
  W = 21 :=
by
  sorry


end workshop_total_workers_l2216_221628


namespace evaluate_g_at_neg3_l2216_221607

def g (x : ℝ) : ℝ := 3 * x ^ 5 - 5 * x ^ 4 + 7 * x ^ 3 - 10 * x ^ 2 - 12 * x + 36

theorem evaluate_g_at_neg3 : g (-3) = -1341 := by
  sorry

end evaluate_g_at_neg3_l2216_221607


namespace abs_eq_necessary_but_not_sufficient_l2216_221623

theorem abs_eq_necessary_but_not_sufficient (x y : ℝ) :
  (|x| = |y|) → (¬(x = y) → x = -y) :=
by
  sorry

end abs_eq_necessary_but_not_sufficient_l2216_221623


namespace cost_per_dozen_l2216_221655

theorem cost_per_dozen (total_cost : ℝ) (total_rolls dozens : ℝ) (cost_per_dozen : ℝ) (h₁ : total_cost = 15) (h₂ : total_rolls = 36) (h₃ : dozens = total_rolls / 12) (h₄ : cost_per_dozen = total_cost / dozens) : cost_per_dozen = 5 :=
by
  sorry

end cost_per_dozen_l2216_221655


namespace madison_classes_l2216_221680

/-- Madison's classes -/
def total_bell_rings : ℕ := 9

/-- Each class requires two bell rings (one to start, one to end) -/
def bell_rings_per_class : ℕ := 2

/-- The number of classes Madison has on Monday -/
theorem madison_classes (total_bell_rings bell_rings_per_class : ℕ) (last_class_start_only : total_bell_rings % bell_rings_per_class = 1) : 
  (total_bell_rings - 1) / bell_rings_per_class + 1 = 5 :=
by
  sorry

end madison_classes_l2216_221680


namespace odd_and_periodic_function_l2216_221640

noncomputable def f : ℝ → ℝ := sorry

lemma given_conditions (x : ℝ) : 
  (f (10 + x) = f (10 - x)) ∧ (f (20 - x) = -f (20 + x)) :=
  sorry

theorem odd_and_periodic_function (x : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (x + 40) = f x) :=
  sorry

end odd_and_periodic_function_l2216_221640


namespace sid_initial_money_l2216_221699

variable (M : ℝ)
variable (spent_on_accessories : ℝ := 12)
variable (spent_on_snacks : ℝ := 8)
variable (remaining_money_condition : ℝ := (M / 2) + 4)

theorem sid_initial_money : (M = 48) → (remaining_money_condition = M - (spent_on_accessories + spent_on_snacks)) :=
by
  sorry

end sid_initial_money_l2216_221699


namespace max_coins_identifiable_l2216_221605

theorem max_coins_identifiable (n : ℕ) : exists (c : ℕ), c = 2 * n^2 + 1 :=
by
  sorry

end max_coins_identifiable_l2216_221605


namespace product_of_fractions_l2216_221675

theorem product_of_fractions :
  (2 / 3) * (5 / 8) * (1 / 4) = 5 / 48 := by
  sorry

end product_of_fractions_l2216_221675


namespace sum_c_eq_l2216_221668

-- Definitions and conditions
def a_n : ℕ → ℝ := λ n => 2 ^ n
def b_n : ℕ → ℝ := λ n => 2 * n
def c_n (n : ℕ) : ℝ := a_n n * b_n n

-- Sum of the first n terms of sequence {c_n}
def sum_c (n : ℕ) : ℝ := (Finset.range n).sum c_n

-- Theorem statement
theorem sum_c_eq (n : ℕ) : sum_c n = (n - 1) * 2 ^ (n + 2) + 4 :=
sorry

end sum_c_eq_l2216_221668


namespace range_of_ab_c2_l2216_221694

theorem range_of_ab_c2 (a b c : ℝ) (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) (h4 : -2 < c) (h5 : c < -1) :
    0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := 
sorry

end range_of_ab_c2_l2216_221694


namespace point_A_coordinates_l2216_221682

noncomputable def f (a x : ℝ) : ℝ := a * x - 1

theorem point_A_coordinates (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 1 :=
sorry

end point_A_coordinates_l2216_221682


namespace find_a_l2216_221669

-- Definitions
def parabola (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c
def vertex_property (a b c : ℤ) := 
  ∃ x y, x = 2 ∧ y = 5 ∧ y = parabola a b c x
def point_on_parabola (a b c : ℤ) := 
  ∃ x y, x = 1 ∧ y = 2 ∧ y = parabola a b c x

-- The main statement
theorem find_a {a b c : ℤ} (h_vertex : vertex_property a b c) (h_point : point_on_parabola a b c) : a = -3 :=
by {
  sorry
}

end find_a_l2216_221669


namespace parabola_points_count_l2216_221642

theorem parabola_points_count :
  ∃ n : ℕ, n = 8 ∧ 
    (∀ x y : ℕ, (y = -((x^2 : ℤ) / 3) + 7 * (x : ℤ) + 54) → 1 ≤ x ∧ x ≤ 26 ∧ x % 3 = 0) :=
by
  sorry

end parabola_points_count_l2216_221642


namespace part1_part2_l2216_221683

-- Define the predicate for the inequality
def prop (x m : ℝ) : Prop := x^2 - 2 * m * x - 3 * m^2 < 0

-- Define the set A
def A (m : ℝ) : Prop := m < -2 ∨ m > 2 / 3

-- Define the predicate for the other inequality
def prop_B (x a : ℝ) : Prop := x^2 - 2 * a * x + a^2 - 1 < 0

-- Define the set B in terms of a
def B (x a : ℝ) : Prop := a - 1 < x ∧ x < a + 1

-- Define the propositions required in the problem
theorem part1 (m : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → prop x m) ↔ A m :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, B x a → A x) ∧ (∃ x, A x ∧ ¬ B x a) ↔ (a ≤ -3 ∨ a ≥ 5 / 3) :=
sorry

end part1_part2_l2216_221683


namespace MeganSavingsExceed500_l2216_221677

theorem MeganSavingsExceed500 :
  ∃ n : ℕ, n ≥ 7 ∧ ((3^n - 1) / 2 > 500) :=
sorry

end MeganSavingsExceed500_l2216_221677


namespace correct_survey_method_l2216_221645

-- Definitions for the conditions
def visionStatusOfMiddleSchoolStudentsNationwide := "Comprehensive survey is impractical for this large population."
def batchFoodContainsPreservatives := "Comprehensive survey is unnecessary, sampling survey would suffice."
def airQualityOfCity := "Comprehensive survey is impractical due to vast area, sampling survey is appropriate."
def passengersCarryProhibitedItems := "Comprehensive survey is necessary for security reasons."

-- Theorem stating that option C is the correct and reasonable choice
theorem correct_survey_method : airQualityOfCity = "Comprehensive survey is impractical due to vast area, sampling survey is appropriate." := by
  sorry

end correct_survey_method_l2216_221645


namespace find_M_l2216_221697

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3}

-- Define the complement of M with respect to U
def complement_M : Set ℕ := {2}

-- Define M as U without the complement of M
def M : Set ℕ := U \ complement_M

-- Prove that M is {0, 1, 3}
theorem find_M : M = {0, 1, 3} := by
  sorry

end find_M_l2216_221697


namespace sum_of_digits_0_to_2012_l2216_221615

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat := 
  n.digits 10 |>.sum

-- Define the problem to calculate the sum of all digits from 0 to 2012
def sum_digits_up_to (n : Nat) : Nat := 
  (List.range (n + 1)).map sum_of_digits |>.sum

-- Lean theorem statement to prove the sum of digits from 0 to 2012 is 28077
theorem sum_of_digits_0_to_2012 : sum_digits_up_to 2012 = 28077 := 
  sorry

end sum_of_digits_0_to_2012_l2216_221615


namespace roll_two_dice_prime_sum_l2216_221622

noncomputable def prime_sum_probability : ℚ :=
  let favorable_outcomes := 15
  let total_outcomes := 36
  favorable_outcomes / total_outcomes

theorem roll_two_dice_prime_sum : prime_sum_probability = 5 / 12 :=
  sorry

end roll_two_dice_prime_sum_l2216_221622


namespace count100DigitEvenNumbers_is_correct_l2216_221695

noncomputable def count100DigitEvenNumbers : ℕ :=
  let validDigits : Finset ℕ := {0, 1, 3}
  let firstDigitChoices : ℕ := 2  -- Only 1 or 3
  let middleDigitsChoices : ℕ := 3 ^ 98  -- 3 choices for each of the 98 middle positions
  let lastDigitChoices : ℕ := 1  -- Only 0 (even number requirement)
  firstDigitChoices * middleDigitsChoices * lastDigitChoices

theorem count100DigitEvenNumbers_is_correct :
  count100DigitEvenNumbers = 2 * 3 ^ 98 := by
  sorry

end count100DigitEvenNumbers_is_correct_l2216_221695


namespace slope_of_l4_l2216_221601

open Real

def line1 (x y : ℝ) : Prop := 4 * x - 3 * y = 6
def pointD : ℝ × ℝ := (0, -2)
def line2 (y : ℝ) : Prop := y = -1
def area_triangle_DEF := 4

theorem slope_of_l4 
  (l4_slope : ℝ)
  (H1 : ∃ x, line1 x (-1))
  (H2 : ∀ x y, 
         x ≠ 0 ∧
         y ≠ -2 ∧
         y ≠ -1 →
         line2 y →
         l4_slope = (y - (-2)) / (x - 0) →
         (1/2) * |(y + 1)| * (sqrt ((x-0) * (x-0) + (y-(-2)) * (y-(-2)))) = area_triangle_DEF ) :
  l4_slope = 1 / 8 :=
sorry

end slope_of_l4_l2216_221601


namespace spherical_coordinates_standard_equivalence_l2216_221691

def std_spherical_coords (ρ θ φ: ℝ) : Prop :=
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi

theorem spherical_coordinates_standard_equivalence :
  std_spherical_coords 5 (11 * Real.pi / 6) (2 * Real.pi - 5 * Real.pi / 3) :=
by
  sorry

end spherical_coordinates_standard_equivalence_l2216_221691


namespace solve_for_x_l2216_221673

theorem solve_for_x (x y : ℝ) (h1 : 3 * x - 2 * y = 8) (h2 : 2 * x + 3 * y = 1) : x = 2 := 
by 
  sorry

end solve_for_x_l2216_221673


namespace total_participants_l2216_221663

theorem total_participants (freshmen sophomores : ℕ) (h1 : freshmen = 8) (h2 : sophomores = 5 * freshmen) : freshmen + sophomores = 48 := 
by
  sorry

end total_participants_l2216_221663


namespace exponentiation_rule_l2216_221679

theorem exponentiation_rule (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

end exponentiation_rule_l2216_221679


namespace negation_of_forall_inequality_l2216_221651

theorem negation_of_forall_inequality :
  (¬ (∀ x : ℝ, x > 0 → x * Real.sin x < 2^x - 1)) ↔ (∃ x : ℝ, x > 0 ∧ x * Real.sin x ≥ 2^x - 1) :=
by sorry

end negation_of_forall_inequality_l2216_221651


namespace price_of_pants_l2216_221606

-- Given conditions
variables (P B : ℝ)
axiom h1 : P + B = 70.93
axiom h2 : P = B - 2.93

-- Statement to prove
theorem price_of_pants : P = 34.00 :=
by
  sorry

end price_of_pants_l2216_221606


namespace lizzy_final_amount_l2216_221672

-- Define constants
def m : ℕ := 80   -- cents from mother
def f : ℕ := 40   -- cents from father
def s : ℕ := 50   -- cents spent on candy
def u : ℕ := 70   -- cents from uncle
def t : ℕ := 90   -- cents for the toy
def c : ℕ := 110  -- cents change she received

-- Define the final amount calculation
def final_amount : ℕ := m + f - s + u - t + c

-- Prove the final amount is 160
theorem lizzy_final_amount : final_amount = 160 := by
  sorry

end lizzy_final_amount_l2216_221672


namespace time_between_train_arrivals_l2216_221625

-- Define the conditions as given in the problem statement
def passengers_per_train : ℕ := 320 + 200
def total_passengers_per_hour : ℕ := 6240
def minutes_per_hour : ℕ := 60

-- Declare the statement to be proven
theorem time_between_train_arrivals: 
  (total_passengers_per_hour / passengers_per_train) = (minutes_per_hour / 5) := by 
  sorry

end time_between_train_arrivals_l2216_221625


namespace ant_impossibility_l2216_221644

-- Define the vertices and edges of a cube
structure Cube :=
(vertices : Finset ℕ) -- Representing a finite set of vertices
(edges : Finset (ℕ × ℕ)) -- Representing a finite set of edges between vertices
(valid_edge : ∀ e ∈ edges, ∃ v1 v2, (v1, v2) = e ∨ (v2, v1) = e)
(starting_vertex : ℕ)

-- Ant behavior on the cube
structure AntOnCube (C : Cube) :=
(is_path_valid : List ℕ → Prop) -- A property that checks the path is valid

-- Problem conditions translated: 
-- No retracing and specific visit numbers
noncomputable def ant_problem (C : Cube) (A : AntOnCube C) : Prop :=
  ∀ (path : List ℕ), A.is_path_valid path → ¬ (
    (path.count C.starting_vertex = 25) ∧ 
    (∀ v ∈ C.vertices, v ≠ C.starting_vertex → path.count v = 20)
  )

-- The final theorem statement
theorem ant_impossibility (C : Cube) (A : AntOnCube C) : ant_problem C A :=
by
  -- providing the theorem framework; proof omitted with sorry
  sorry

end ant_impossibility_l2216_221644


namespace min_sum_of_factors_l2216_221620

theorem min_sum_of_factors (x y z : ℕ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : x * y * z = 3920) : x + y + z = 70 :=
sorry

end min_sum_of_factors_l2216_221620


namespace max_cake_pieces_l2216_221602

theorem max_cake_pieces (m n : ℕ) (h₁ : m ≥ 4) (h₂ : n ≥ 4)
    (h : (m-4)*(n-4) = m * n) :
    m * n = 72 :=
by
  sorry

end max_cake_pieces_l2216_221602


namespace train_length_l2216_221619

theorem train_length (L : ℝ) (v : ℝ)
  (h1 : v = (L + 130) / 15)
  (h2 : v = (L + 250) / 20) : 
  L = 230 :=
sorry

end train_length_l2216_221619


namespace find_divisor_l2216_221656

theorem find_divisor (d x k j : ℤ) (h₁ : x = k * d + 5) (h₂ : 7 * x = j * d + 8) : d = 11 :=
sorry

end find_divisor_l2216_221656


namespace num_congruent_2_mod_11_l2216_221638

theorem num_congruent_2_mod_11 : 
  ∃ (n : ℕ), n = 28 ∧ ∀ k : ℤ, 1 ≤ 11 * k + 2 ∧ 11 * k + 2 ≤ 300 ↔ 0 ≤ k ∧ k ≤ 27 :=
sorry

end num_congruent_2_mod_11_l2216_221638


namespace trapezoid_area_ratio_l2216_221686

theorem trapezoid_area_ratio (AD AO OB BC AB DO OC : ℝ) (h_eq1 : AD = 15) (h_eq2 : AO = 15) (h_eq3 : OB = 15) (h_eq4 : BC = 15)
  (h_eq5 : AB = 20) (h_eq6 : DO = 20) (h_eq7 : OC = 20) (is_trapezoid : true) (OP_perp_to_AB : true) 
  (X_mid_AD : true) (Y_mid_BC : true) : (5 + 7 = 12) :=
by
  sorry

end trapezoid_area_ratio_l2216_221686


namespace egg_distribution_l2216_221635

-- Definitions of the conditions
def total_eggs := 10.0
def large_eggs := 6.0
def small_eggs := 4.0

def box_A_capacity := 5.0
def box_B_capacity := 4.0
def box_C_capacity := 6.0

def at_least_one_small_egg (box_A_small box_B_small box_C_small : Float) := 
  box_A_small >= 1.0 ∧ box_B_small >= 1.0 ∧ box_C_small >= 1.0

-- Problem statement
theorem egg_distribution : 
  ∃ (box_A_small box_A_large box_B_small box_B_large box_C_small box_C_large : Float),
  box_A_small + box_A_large <= box_A_capacity ∧
  box_B_small + box_B_large <= box_B_capacity ∧
  box_C_small + box_C_large <= box_C_capacity ∧
  box_A_small + box_B_small + box_C_small = small_eggs ∧
  box_A_large + box_B_large + box_C_large = large_eggs ∧
  at_least_one_small_egg box_A_small box_B_small box_C_small :=
sorry

end egg_distribution_l2216_221635


namespace mr_rainwater_chickens_l2216_221637

theorem mr_rainwater_chickens :
  ∃ (Ch : ℕ), (∀ (C G : ℕ), C = 9 ∧ G = 4 * C ∧ G = 2 * Ch → Ch = 18) :=
by
  sorry

end mr_rainwater_chickens_l2216_221637


namespace set_notation_nat_lt_3_l2216_221624

theorem set_notation_nat_lt_3 : {x : ℕ | x < 3} = {0, 1, 2} := 
sorry

end set_notation_nat_lt_3_l2216_221624


namespace number_of_swaps_independent_l2216_221698

theorem number_of_swaps_independent (n : ℕ) (hn : n = 20) (p : Fin n → Fin n) :
    (∀ i, p i ≠ i → ∃ j, p j ≠ j ∧ p (p j) = j) →
    ∃ s : List (Fin n × Fin n), List.length s ≤ n ∧
    (∀ σ : List (Fin n × Fin n), (∀ (i j : Fin n), (i, j) ∈ σ → p i ≠ i → ∃ p', σ = (i, p') :: (p', j) :: σ) →
     List.length σ = List.length s) :=
  sorry

end number_of_swaps_independent_l2216_221698


namespace number_of_morse_code_symbols_l2216_221633

-- Define the number of sequences for different lengths
def sequences_of_length (n : Nat) : Nat :=
  2 ^ n

theorem number_of_morse_code_symbols : 
  (sequences_of_length 1) + (sequences_of_length 2) + (sequences_of_length 3) + (sequences_of_length 4) + (sequences_of_length 5) = 62 := by
  sorry

end number_of_morse_code_symbols_l2216_221633


namespace middle_number_of_consecutive_numbers_sum_of_squares_eq_2030_l2216_221626

theorem middle_number_of_consecutive_numbers_sum_of_squares_eq_2030 :
  ∃ n : ℕ, n^2 + (n+1)^2 + (n+2)^2 = 2030 ∧ (n + 1) = 26 :=
by sorry

end middle_number_of_consecutive_numbers_sum_of_squares_eq_2030_l2216_221626


namespace greatest_a_inequality_l2216_221612

theorem greatest_a_inequality :
  ∃ a : ℝ, (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ a * (x₁ * x₂ + x₂ * x₃ + x₃ * x₄ + x₄ * x₅)) ∧
          (∀ b : ℝ, (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ b * (x₁ * x₂ + x₂ * x₃ + x₃ * x₄ + x₄ * x₅)) → b ≤ a) ∧
          a = 2 / Real.sqrt 3 :=
sorry

end greatest_a_inequality_l2216_221612


namespace sum_s_r_values_l2216_221687

def r_values : List ℤ := [-2, -1, 0, 1, 3]
def r_range : List ℤ := [-1, 0, 1, 3, 5]

def s (x : ℤ) : ℤ := if 1 ≤ x then 2 * x + 1 else 0

theorem sum_s_r_values :
  (s 1) + (s 3) + (s 5) = 21 :=
by
  sorry

end sum_s_r_values_l2216_221687


namespace min_expression_l2216_221604

theorem min_expression 
  (a b c : ℝ)
  (ha : -1 < a ∧ a < 1)
  (hb : -1 < b ∧ b < 1)
  (hc : -1 < c ∧ c < 1) :
  ∃ m, m = 2 ∧ ∀ x y z, (-1 < x ∧ x < 1) → (-1 < y ∧ y < 1) → (-1 < z ∧ z < 1) → 
  ( 1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)) + 1 / ((1 + x^2) * (1 + y^2) * (1 + z^2)) ) ≥ m :=
sorry

end min_expression_l2216_221604


namespace geometric_sequence_arithmetic_progression_l2216_221621

theorem geometric_sequence_arithmetic_progression
  (q : ℝ) (h_q : q ≠ 1)
  (a : ℕ → ℝ) (m n p : ℕ)
  (h1 : ∃ a1, ∀ k, a k = a1 * q ^ (k - 1))
  (h2 : a n ^ 2 = a m * a p) :
  2 * n = m + p := 
by
  sorry

end geometric_sequence_arithmetic_progression_l2216_221621


namespace right_triangle_x_value_l2216_221688

theorem right_triangle_x_value (x Δ : ℕ) (h₁ : x > 0) (h₂ : Δ > 0) :
  ((x + 2 * Δ)^2 = x^2 + (x + Δ)^2) → 
  x = (Δ * (-1 + 2 * Real.sqrt 7)) / 2 := 
sorry

end right_triangle_x_value_l2216_221688


namespace distance_between_points_l2216_221684

open Complex Real

def joe_point : ℂ := 2 + 3 * I
def gracie_point : ℂ := -2 + 2 * I

theorem distance_between_points : abs (joe_point - gracie_point) = sqrt 17 := by
  sorry

end distance_between_points_l2216_221684


namespace algebra_expression_evaluation_l2216_221639

theorem algebra_expression_evaluation (a b : ℝ) (h : a + 3 * b = 4) : 2 * a + 6 * b - 1 = 7 := by
  sorry

end algebra_expression_evaluation_l2216_221639


namespace stutterer_square_number_unique_l2216_221674

-- Definitions based on problem conditions
def is_stutterer (n : ℕ) : Prop :=
  (1000 ≤ n ∧ n < 10000) ∧ (n / 100 = (n % 1000) / 100) ∧ ((n % 1000) % 100 = n % 10 * 10 + n % 10)

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- The theorem statement
theorem stutterer_square_number_unique : ∃ n, is_stutterer n ∧ is_square n ∧ n = 7744 :=
by
  sorry

end stutterer_square_number_unique_l2216_221674


namespace prove_p_or_q_l2216_221689

-- Define propositions p and q
def p : Prop := ∃ n : ℕ, 0 = 2 * n
def q : Prop := ∃ m : ℕ, 3 = 2 * m

-- The Lean statement to prove
theorem prove_p_or_q : p ∨ q := by
  sorry

end prove_p_or_q_l2216_221689


namespace div_a2_plus_2_congr_mod8_l2216_221610

variable (a d : ℤ)
variable (h_odd : a % 2 = 1)
variable (h_pos : a > 0)

theorem div_a2_plus_2_congr_mod8 :
  (d ∣ (a ^ 2 + 2)) → (d % 8 = 1 ∨ d % 8 = 3) :=
by
  sorry

end div_a2_plus_2_congr_mod8_l2216_221610


namespace probability_y_gt_x_l2216_221664

-- Define the uniform distribution and the problem setup
def uniform_distribution (a b : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ b }

-- Define the variables
variables (x : ℝ) (hx : x ∈ uniform_distribution 0 3000) (y : ℝ) (hy : y ∈ uniform_distribution 0 6000)

-- Define the probability calculation function (assuming some proper definition for probability)
noncomputable def probability_event (E : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the event that Laurent's number is greater than Chloe's number
def event_y_gt_x : Set (ℝ × ℝ) := {p | p.2 > p.1}

-- State the theorem
theorem probability_y_gt_x (x : ℝ) (hx : x ∈ uniform_distribution 0 3000) (y : ℝ) (hy : y ∈ uniform_distribution 0 6000) :
  probability_event event_y_gt_x = 3/4 :=
sorry

end probability_y_gt_x_l2216_221664


namespace unique_function_l2216_221657

theorem unique_function (f : ℝ → ℝ) (hf : ∀ x : ℝ, 0 ≤ x → 0 ≤ f x)
  (cond1 : ∀ x : ℝ, 0 ≤ x → 4 * f x ≥ 3 * x)
  (cond2 : ∀ x : ℝ, 0 ≤ x → f (4 * f x - 3 * x) = x) :
  ∀ x : ℝ, 0 ≤ x → f x = x :=
by
  sorry

end unique_function_l2216_221657


namespace factor_54x5_135x9_l2216_221667

theorem factor_54x5_135x9 (x : ℝ) :
  54 * x ^ 5 - 135 * x ^ 9 = -27 * x ^ 5 * (5 * x ^ 4 - 2) :=
by 
  sorry

end factor_54x5_135x9_l2216_221667


namespace internet_usage_minutes_l2216_221632

-- Define the given conditions
variables (M P E : ℕ)

-- Problem statement
theorem internet_usage_minutes (h : P ≠ 0) : 
  (∀ M P E : ℕ, ∃ y : ℕ, y = (100 * E * M) / P) :=
by {
  sorry
}

end internet_usage_minutes_l2216_221632


namespace correct_equation_l2216_221690

def initial_count_A : ℕ := 54
def initial_count_B : ℕ := 48
def new_count_A (x : ℕ) : ℕ := initial_count_A + x
def new_count_B (x : ℕ) : ℕ := initial_count_B - x

theorem correct_equation (x : ℕ) : new_count_A x = 2 * new_count_B x := 
sorry

end correct_equation_l2216_221690


namespace div_remainder_l2216_221693

theorem div_remainder (B x : ℕ) (h1 : B = 301) (h2 : B % 7 = 0) : x = 3 :=
  sorry

end div_remainder_l2216_221693


namespace quadratic_inequality_solution_l2216_221618

theorem quadratic_inequality_solution {x : ℝ} :
  (x^2 - 6 * x - 16 > 0) ↔ (x < -2 ∨ x > 8) :=
sorry

end quadratic_inequality_solution_l2216_221618


namespace units_place_3_pow_34_l2216_221603

theorem units_place_3_pow_34 : (3^34 % 10) = 9 :=
by
  sorry

end units_place_3_pow_34_l2216_221603


namespace solve_for_x_l2216_221631

theorem solve_for_x (q r x : ℚ)
  (h1 : 5 / 6 = q / 90)
  (h2 : 5 / 6 = (q + r) / 102)
  (h3 : 5 / 6 = (x - r) / 150) :
  x = 135 :=
by sorry

end solve_for_x_l2216_221631


namespace hall_width_to_length_ratio_l2216_221654

def width (w l : ℝ) : Prop := w * l = 578
def length_width_difference (w l : ℝ) : Prop := l - w = 17

theorem hall_width_to_length_ratio (w l : ℝ) (hw : width w l) (hl : length_width_difference w l) : (w / l = 1 / 2) :=
by
  sorry

end hall_width_to_length_ratio_l2216_221654


namespace original_time_to_complete_book_l2216_221678

-- Define the problem based on the given conditions
variables (n : ℕ) (T : ℚ)

-- Define the conditions
def condition1 : Prop := 
  ∃ (n T : ℚ), 
  n / T = (n + 3) / (0.75 * T) ∧
  n / T = (n - 3) / (T + 5 / 6)

-- State the theorem with the correct answer
theorem original_time_to_complete_book : condition1 → T = 5 / 3 :=
by sorry

end original_time_to_complete_book_l2216_221678


namespace new_student_weight_l2216_221630

theorem new_student_weight : 
  ∀ (w_new : ℕ), 
    (∀ (sum_weight: ℕ), 80 + sum_weight - w_new = sum_weight - 18) → 
      w_new = 62 := 
by
  intros w_new h
  sorry

end new_student_weight_l2216_221630


namespace solve_for_a_l2216_221648

theorem solve_for_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 13) (h3 : 13 ∣ 51^2016 - a) : a = 1 :=
by {
  sorry
}

end solve_for_a_l2216_221648


namespace integer_solution_abs_lt_sqrt2_l2216_221661

theorem integer_solution_abs_lt_sqrt2 (x : ℤ) (h : |x| < Real.sqrt 2) : x = -1 ∨ x = 0 ∨ x = 1 :=
sorry

end integer_solution_abs_lt_sqrt2_l2216_221661


namespace cone_radius_correct_l2216_221660

noncomputable def cone_radius (CSA l : ℝ) : ℝ := CSA / (Real.pi * l)

theorem cone_radius_correct :
  cone_radius 1539.3804002589986 35 = 13.9 :=
by
  -- Proof omitted
  sorry

end cone_radius_correct_l2216_221660


namespace clover_walk_distance_l2216_221629

theorem clover_walk_distance (total_distance days walks_per_day : ℝ) (h1 : total_distance = 90) (h2 : days = 30) (h3 : walks_per_day = 2) :
  (total_distance / days / walks_per_day = 1.5) :=
by
  sorry

end clover_walk_distance_l2216_221629


namespace coprime_odd_sum_of_floors_l2216_221692

theorem coprime_odd_sum_of_floors (p q : ℕ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h_coprime : Nat.gcd p q = 1) : 
  (List.sum (List.map (λ i => Nat.floor ((i • q : ℚ) / p)) ((List.range (p / 2 + 1)).tail)) +
   List.sum (List.map (λ i => Nat.floor ((i • p : ℚ) / q)) ((List.range (q / 2 + 1)).tail))) =
  (p - 1) * (q - 1) / 4 :=
by
  sorry

end coprime_odd_sum_of_floors_l2216_221692


namespace average_weight_of_Arun_l2216_221659

theorem average_weight_of_Arun :
  ∃ avg_weight : Real,
    (avg_weight = (65 + 68) / 2) ∧
    ∀ w : Real, (65 < w ∧ w < 72) ∧ (60 < w ∧ w < 70) ∧ (w ≤ 68) → avg_weight = 66.5 :=
by
  -- we will fill the details of the proof here
  sorry

end average_weight_of_Arun_l2216_221659


namespace sum_of_possible_values_of_y_l2216_221611

-- Definitions of the conditions
variables (y : ℝ)
-- Angle measures in degrees
variables (a b c : ℝ)
variables (isosceles : Bool)

-- Given conditions
def is_isosceles_triangle (a b c : ℝ) (isosceles : Bool) : Prop :=
  isosceles = true ∧ (a = b ∨ b = c ∨ c = a)

-- Sum of angles in any triangle
def sum_of_angles_in_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180

-- Main statement to be proven
theorem sum_of_possible_values_of_y (y : ℝ) (a b c : ℝ) (isosceles : Bool) :
  is_isosceles_triangle a b c isosceles →
  sum_of_angles_in_triangle a b c →
  ((y = 60) → (a = y ∨ b = y ∨ c = y)) →
  isosceles = true → a = 60 ∨ b = 60 ∨ c = 60 →
  y + y + y = 180 :=
by
  intros h1 h2 h3 h4 h5
  sorry  -- Proof will be provided here

end sum_of_possible_values_of_y_l2216_221611


namespace arithmetic_and_geometric_sequence_l2216_221627

-- Definitions based on given conditions
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

-- Main statement to prove
theorem arithmetic_and_geometric_sequence :
  ∀ (x y a b c : ℝ), 
  is_arithmetic_sequence 1 x y 4 →
  is_geometric_sequence (-2) a b c (-8) →
  (y - x) / b = -1 / 4 :=
by
  sorry

end arithmetic_and_geometric_sequence_l2216_221627


namespace relationship_between_a_and_b_l2216_221658

def ellipse_touching_hyperbola (a b : ℝ) :=
  ∀ x y : ℝ, ( (x / a) ^ 2 + (y / b) ^ 2 = 1 ∧ y = 1 / x → False )

  theorem relationship_between_a_and_b (a b : ℝ) :
  ellipse_touching_hyperbola a b →
  a * b = 2 :=
by
  sorry

end relationship_between_a_and_b_l2216_221658


namespace teammates_score_l2216_221614

def Lizzie_score := 4
def Nathalie_score := Lizzie_score + 3
def combined_Lizzie_Nathalie := Lizzie_score + Nathalie_score
def Aimee_score := 2 * combined_Lizzie_Nathalie
def total_team_score := 50
def total_combined_score := Lizzie_score + Nathalie_score + Aimee_score

theorem teammates_score : total_team_score - total_combined_score = 17 :=
by
  sorry

end teammates_score_l2216_221614


namespace range_of_m_l2216_221670

theorem range_of_m (α β m : ℝ) (hαβ : 0 < α ∧ α < 1 ∧ 1 < β ∧ β < 2)
  (h_eq : ∀ x, x^2 - 2*(m-1)*x + (m-1) = 0 ↔ (x = α ∨ x = β)) :
  2 < m ∧ m < 7 / 3 := by
  sorry

end range_of_m_l2216_221670


namespace first_year_payment_l2216_221652

theorem first_year_payment (x : ℝ) 
  (second_year : ℝ := x + 2)
  (third_year : ℝ := x + 5)
  (fourth_year : ℝ := x + 9)
  (total_payment : ℝ := x + second_year + third_year + fourth_year)
  (h : total_payment = 96) : x = 20 := 
by
  sorry

end first_year_payment_l2216_221652


namespace fraction_of_product_l2216_221616

theorem fraction_of_product : (7 / 8) * 64 = 56 := by
  sorry

end fraction_of_product_l2216_221616


namespace average_speed_approx_l2216_221643

noncomputable def average_speed : ℝ :=
  let distance1 := 7
  let speed1 := 10
  let distance2 := 10
  let speed2 := 7
  let distance3 := 5
  let speed3 := 12
  let distance4 := 8
  let speed4 := 6
  let total_distance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let time4 := distance4 / speed4
  let total_time := time1 + time2 + time3 + time4
  total_distance / total_time

theorem average_speed_approx : abs (average_speed - 7.73) < 0.01 := by
  -- The necessary definitions fulfill the conditions and hence we put sorry here
  sorry

end average_speed_approx_l2216_221643


namespace minimum_people_who_like_both_l2216_221634

theorem minimum_people_who_like_both
    (total_people : ℕ)
    (vivaldi_likers : ℕ)
    (chopin_likers : ℕ)
    (people_surveyed : total_people = 150)
    (like_vivaldi : vivaldi_likers = 120)
    (like_chopin : chopin_likers = 90) :
    ∃ (both_likers : ℕ), both_likers = 60 ∧
                            vivaldi_likers + chopin_likers - both_likers ≤ total_people :=
by 
  sorry

end minimum_people_who_like_both_l2216_221634


namespace tunnel_length_l2216_221696

noncomputable def train_speed_mph : ℝ := 75
noncomputable def train_length_miles : ℝ := 1 / 4
noncomputable def passing_time_minutes : ℝ := 3

theorem tunnel_length :
  let speed_mpm := train_speed_mph / 60
  let total_distance_traveled := speed_mpm * passing_time_minutes
  let tunnel_length := total_distance_traveled - train_length_miles
  tunnel_length = 3.5 :=
by
  sorry

end tunnel_length_l2216_221696


namespace find_a6_l2216_221609

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ a 2 = 5 ∧ ∀ n : ℕ, a (n + 1) = a (n + 2) + a n

theorem find_a6 (a : ℕ → ℤ) (h : seq a) : a 6 = -3 :=
by
  sorry

end find_a6_l2216_221609


namespace sqrt_neg_sq_eq_two_l2216_221666

theorem sqrt_neg_sq_eq_two : Real.sqrt ((-2 : ℝ)^2) = 2 := by
  -- Proof intentionally omitted.
  sorry

end sqrt_neg_sq_eq_two_l2216_221666


namespace area_of_fig_between_x1_and_x2_l2216_221608

noncomputable def area_under_curve_x2 (a b : ℝ) : ℝ :=
∫ x in a..b, x^2

theorem area_of_fig_between_x1_and_x2 :
  area_under_curve_x2 1 2 = 7 / 3 := by
  sorry

end area_of_fig_between_x1_and_x2_l2216_221608
