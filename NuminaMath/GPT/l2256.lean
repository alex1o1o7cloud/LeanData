import Mathlib

namespace base_b_expression_not_divisible_l2256_225642

theorem base_b_expression_not_divisible 
  (b : ℕ) : 
  (b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 7 ∨ b = 8) →
  (2 * b^3 - 2 * b^2 + b - 1) % 5 ≠ 0 ↔ (b ≠ 6) :=
by
  sorry

end base_b_expression_not_divisible_l2256_225642


namespace right_triangle_side_length_l2256_225679

theorem right_triangle_side_length (hypotenuse : ℝ) (θ : ℝ) (sin_30 : Real.sin 30 = 1 / 2) (h : θ = 30) 
  (hyp_len : hypotenuse = 10) : 
  let opposite_side := hypotenuse * Real.sin θ
  opposite_side = 5 := by
  sorry

end right_triangle_side_length_l2256_225679


namespace vector_subtraction_l2256_225637

def a : ℝ × ℝ := (3, 5)
def b : ℝ × ℝ := (-2, 1)
def two_b : ℝ × ℝ := (2 * b.1, 2 * b.2)

theorem vector_subtraction : (a.1 - two_b.1, a.2 - two_b.2) = (7, 3) := by
  sorry

end vector_subtraction_l2256_225637


namespace bread_rolls_count_l2256_225648

theorem bread_rolls_count (total_items croissants bagels : Nat) 
  (h1 : total_items = 90) 
  (h2 : croissants = 19) 
  (h3 : bagels = 22) : 
  total_items - croissants - bagels = 49 := 
by
  sorry

end bread_rolls_count_l2256_225648


namespace sum_of_consecutive_integers_exists_l2256_225684

theorem sum_of_consecutive_integers_exists : 
  ∃ k : ℕ, 150 * k + 11325 = 5827604250 :=
by
  sorry

end sum_of_consecutive_integers_exists_l2256_225684


namespace can_form_triangle_l2256_225658

theorem can_form_triangle : Prop :=
  ∃ (a b c : ℝ), 
    (a = 8 ∧ b = 6 ∧ c = 4) ∧
    (a + b > c ∧ a + c > b ∧ b + c > a)

#check can_form_triangle

end can_form_triangle_l2256_225658


namespace path_length_l2256_225639

theorem path_length (scale_ratio : ℕ) (map_path_length : ℝ) 
  (h1 : scale_ratio = 500)
  (h2 : map_path_length = 3.5) : 
  (map_path_length * scale_ratio = 1750) :=
sorry

end path_length_l2256_225639


namespace find_smaller_number_l2256_225690

theorem find_smaller_number (x : ℕ) (h1 : ∃ y, y = 3 * x) (h2 : x + 3 * x = 124) : x = 31 :=
by
  -- Proof will be here
  sorry

end find_smaller_number_l2256_225690


namespace calculate_product_l2256_225616

theorem calculate_product (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3*x1*y1^2 = 2030)
  (h2 : y1^3 - 3*x1^2*y1 = 2029)
  (h3 : x2^3 - 3*x2*y2^2 = 2030)
  (h4 : y2^3 - 3*x2^2*y2 = 2029)
  (h5 : x3^3 - 3*x3*y3^2 = 2030)
  (h6 : y3^3 - 3*x3^2*y3 = 2029) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = -1 / 1015 :=
sorry

end calculate_product_l2256_225616


namespace solve_for_x_l2256_225624

theorem solve_for_x (x : ℝ) (h : 4 / (1 + 3 / x) = 1) : x = 1 :=
sorry

end solve_for_x_l2256_225624


namespace largest_x_FloorDiv7_eq_FloorDiv8_plus_1_l2256_225608

-- Definitions based on conditions
def floor_div_7 (x : ℕ) : ℕ := x / 7
def floor_div_8 (x : ℕ) : ℕ := x / 8

-- The statement of the problem
theorem largest_x_FloorDiv7_eq_FloorDiv8_plus_1 :
  ∃ x : ℕ, (floor_div_7 x = floor_div_8 x + 1) ∧ (∀ y : ℕ, floor_div_7 y = floor_div_8 y + 1 → y ≤ x) ∧ x = 104 :=
sorry

end largest_x_FloorDiv7_eq_FloorDiv8_plus_1_l2256_225608


namespace total_marks_l2256_225609

theorem total_marks (Keith_marks Larry_marks Danny_marks : ℕ)
  (hK : Keith_marks = 3)
  (hL : Larry_marks = 3 * Keith_marks)
  (hD : Danny_marks = Larry_marks + 5) :
  Keith_marks + Larry_marks + Danny_marks = 26 := 
by
  sorry

end total_marks_l2256_225609


namespace fifth_inequality_l2256_225686

theorem fifth_inequality :
  1 + (1 / 2^2) + (1 / 3^2) + (1 / 4^2) + (1 / 5^2) + (1 / 6^2) < 11 / 6 :=
sorry

end fifth_inequality_l2256_225686


namespace min_white_surface_area_is_five_over_ninety_six_l2256_225696

noncomputable def fraction_white_surface_area (total_surface_area white_surface_area : ℕ) :=
  (white_surface_area : ℚ) / (total_surface_area : ℚ)

theorem min_white_surface_area_is_five_over_ninety_six :
  let total_surface_area := 96
  let white_surface_area := 5
  fraction_white_surface_area total_surface_area white_surface_area = 5 / 96 :=
by
  sorry

end min_white_surface_area_is_five_over_ninety_six_l2256_225696


namespace triangle_area_arithmetic_sequence_l2256_225619

theorem triangle_area_arithmetic_sequence :
  ∃ (S_1 S_2 S_3 S_4 S_5 : ℝ) (d : ℝ),
  S_1 + S_2 + S_3 + S_4 + S_5 = 420 ∧
  S_2 = S_1 + d ∧
  S_3 = S_1 + 2 * d ∧
  S_4 = S_1 + 3 * d ∧
  S_5 = S_1 + 4 * d ∧
  S_5 = 112 :=
by
  sorry

end triangle_area_arithmetic_sequence_l2256_225619


namespace least_number_of_cars_per_work_day_l2256_225651

-- Define the conditions as constants in Lean
def paul_work_hours_per_day := 8
def jack_work_hours_per_day := 8
def paul_cars_per_hour := 2
def jack_cars_per_hour := 3

-- Define the total number of cars Paul and Jack can change in a workday
def total_cars_per_day := (paul_cars_per_hour + jack_cars_per_hour) * paul_work_hours_per_day

-- State the theorem to be proved
theorem least_number_of_cars_per_work_day : total_cars_per_day = 40 := by
  -- Proof goes here
  sorry

end least_number_of_cars_per_work_day_l2256_225651


namespace false_statement_l2256_225635

-- Define the geometrical conditions based on the problem statements
variable {A B C D: Type}

-- A rhombus with equal diagonals is a square
def rhombus_with_equal_diagonals_is_square (R : A) : Prop := 
  ∀ (a b : A), a = b → true

-- A rectangle with perpendicular diagonals is a square
def rectangle_with_perpendicular_diagonals_is_square (Rec : B) : Prop :=
  ∀ (a b : B), a = b → true

-- A parallelogram with perpendicular and equal diagonals is a square
def parallelogram_with_perpendicular_and_equal_diagonals_is_square (P : C) : Prop :=
  ∀ (a b : C), a = b → true

-- A quadrilateral with perpendicular and bisecting diagonals is a square
def quadrilateral_with_perpendicular_and_bisecting_diagonals_is_square (Q : D) : Prop :=
  ∀ (a b : D), (a = b) → true 

-- The main theorem: Statement D is false
theorem false_statement (Q : D) : ¬ (quadrilateral_with_perpendicular_and_bisecting_diagonals_is_square Q) := 
  sorry

end false_statement_l2256_225635


namespace range_of_a_l2256_225693

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, y = (a * (Real.cos x)^2 - 3) * (Real.sin x) ∧ y ≥ -3) 
  → a ∈ Set.Icc (-3/2 : ℝ) 12 :=
sorry

end range_of_a_l2256_225693


namespace max_sum_x_y_under_condition_l2256_225666

-- Define the conditions
variables (x y : ℝ)

-- State the problem and what needs to be proven
theorem max_sum_x_y_under_condition : 
  (3 * (x^2 + y^2) = x - y) → (x + y) ≤ (1 / Real.sqrt 2) :=
by
  sorry

end max_sum_x_y_under_condition_l2256_225666


namespace gcd_of_78_and_104_l2256_225620

theorem gcd_of_78_and_104 : Int.gcd 78 104 = 26 := by
  sorry

end gcd_of_78_and_104_l2256_225620


namespace solve_for_x_l2256_225699

theorem solve_for_x : ∀ x : ℝ, (x - 27) / 3 = (3 * x + 6) / 8 → x = -234 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l2256_225699


namespace common_chord_of_circles_l2256_225618

theorem common_chord_of_circles : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 + 2*x = 0 ∧ x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) := 
by 
  sorry

end common_chord_of_circles_l2256_225618


namespace max_height_piston_l2256_225685

theorem max_height_piston (M a P c_v g R: ℝ) (h : ℝ) 
  (h_pos : 0 < h) (M_pos : 0 < M) (a_pos : 0 < a) (P_pos : 0 < P)
  (c_v_pos : 0 < c_v) (g_pos : 0 < g) (R_pos : 0 < R) :
  h = (2 * P ^ 2) / (M ^ 2 * g * a ^ 2 * (1 + c_v / R) ^ 2) := sorry

end max_height_piston_l2256_225685


namespace prob_of_king_or_queen_top_l2256_225694

/-- A standard deck comprises 52 cards, with 13 ranks and 4 suits, each rank having one card per suit. -/
def standard_deck : Set (String × String) :=
Set.prod { "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King" }
          { "Hearts", "Diamonds", "Clubs", "Spades" }

/-- There are four cards of rank King and four of rank Queen in the standard deck. -/
def count_kings_and_queens : Nat := 
4 + 4

/-- The total number of cards in a standard deck is 52. -/
def total_cards : Nat := 52

/-- The probability that the top card is either a King or a Queen is 2/13. -/
theorem prob_of_king_or_queen_top :
  (count_kings_and_queens / total_cards : ℚ) = (2 / 13 : ℚ) :=
sorry

end prob_of_king_or_queen_top_l2256_225694


namespace f_monotonicity_l2256_225615

noncomputable def f (x : ℝ) : ℝ := abs (x^2 - 1)

theorem f_monotonicity :
  (∀ x y : ℝ, (-1 < x ∧ x < 0 ∧ x < y ∧ y < 0) → f x < f y) ∧
  (∀ x y : ℝ, (1 < x ∧ 1 < y ∧ x < y) → f x < f y) ∧
  (∀ x y : ℝ, (x < -1 ∧ y < -1 ∧ y < x) → f x < f y) ∧
  (∀ x y : ℝ, (0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ y < x) → f x < f y) :=
by
  sorry

end f_monotonicity_l2256_225615


namespace how_many_buns_each_student_gets_l2256_225646

theorem how_many_buns_each_student_gets 
  (packages : ℕ) 
  (buns_per_package : ℕ) 
  (classes : ℕ) 
  (students_per_class : ℕ)
  (h1 : buns_per_package = 8)
  (h2 : packages = 30)
  (h3 : classes = 4)
  (h4 : students_per_class = 30) :
  (packages * buns_per_package) / (classes * students_per_class) = 2 :=
by sorry

end how_many_buns_each_student_gets_l2256_225646


namespace exists_powers_mod_eq_l2256_225687

theorem exists_powers_mod_eq (N : ℕ) (A : ℤ) : ∃ r s : ℕ, r ≠ s ∧ (A ^ r - A ^ s) % N = 0 :=
sorry

end exists_powers_mod_eq_l2256_225687


namespace product_of_powers_eq_nine_l2256_225631

variable (a : ℕ)

theorem product_of_powers_eq_nine : a^3 * a^6 = a^9 := 
by sorry

end product_of_powers_eq_nine_l2256_225631


namespace university_diploma_percentage_l2256_225656

variables (population : ℝ)
          (U : ℝ) -- percentage of people with a university diploma
          (J : ℝ := 0.40) -- percentage of people with the job of their choice
          (S : ℝ := 0.10) -- percentage of people with a secondary school diploma pursuing further education

-- Condition 1: 18% of the people do not have a university diploma but have the job of their choice.
-- Condition 2: 25% of the people who do not have the job of their choice have a university diploma.
-- Condition 3: 10% of the people have a secondary school diploma and are pursuing further education.
-- Condition 4: 60% of the people with secondary school diploma have the job of their choice.
-- Condition 5: 30% of the people in further education have a job of their choice as well.
-- Condition 6: 40% of the people have the job of their choice.

axiom condition_1 : 0.18 * population = (0.18 * (1 - U)) * (population)
axiom condition_2 : 0.25 * (100 - J * 100) = 0.25 * (population - J * population)
axiom condition_3 : S * population = 0.10 * population
axiom condition_4 : 0.60 * S * population = (0.60 * S) * population
axiom condition_5 : 0.30 * S * population = (0.30 * S) * population
axiom condition_6 : J * population = 0.40 * population

theorem university_diploma_percentage : U * 100 = 37 :=
by sorry

end university_diploma_percentage_l2256_225656


namespace f_3_minus_f_4_l2256_225671

noncomputable def f : ℝ → ℝ := sorry
axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 2) = -f x
axiom initial_condition : f 1 = 1

theorem f_3_minus_f_4 : f 3 - f 4 = -1 :=
by
  sorry

end f_3_minus_f_4_l2256_225671


namespace jack_change_l2256_225647

theorem jack_change :
  let discountedCost1 := 4.50
  let discountedCost2 := 4.50
  let discountedCost3 := 5.10
  let cost4 := 7.00
  let totalDiscountedCost := discountedCost1 + discountedCost2 + discountedCost3 + cost4
  let tax := totalDiscountedCost * 0.05
  let taxRounded := 1.06 -- Tax rounded to nearest cent
  let totalCostWithTax := totalDiscountedCost + taxRounded
  let totalCostWithServiceFee := totalCostWithTax + 2.00
  let totalPayment := 20 + 10 + 4 * 1
  let change := totalPayment - totalCostWithServiceFee
  change = 9.84 :=
by
  sorry

end jack_change_l2256_225647


namespace probability_of_odd_divisor_l2256_225627

noncomputable def factorial_prime_factors : ℕ → List (ℕ × ℕ)
| 21 => [(2, 18), (3, 9), (5, 4), (7, 3), (11, 1), (13, 1), (17, 1), (19, 1)]
| _ => []

def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨_, exp⟩ => acc * (exp + 1)) 1

def number_of_odd_factors (factors : List (ℕ × ℕ)) : ℕ :=
  number_of_factors (factors.filter (λ ⟨p, _⟩ => p != 2))

theorem probability_of_odd_divisor : (number_of_odd_factors (factorial_prime_factors 21)) /
(number_of_factors (factorial_prime_factors 21)) = 1 / 19 := 
by
  sorry

end probability_of_odd_divisor_l2256_225627


namespace repeating_decimal_to_fraction_l2256_225634

theorem repeating_decimal_to_fraction (h : (0.0909090909 : ℝ) = 1 / 11) : (0.2727272727 : ℝ) = 3 / 11 :=
sorry

end repeating_decimal_to_fraction_l2256_225634


namespace discount_rate_l2256_225604

theorem discount_rate (marked_price selling_price discount_rate: ℝ) 
  (h₁: marked_price = 80)
  (h₂: selling_price = 68)
  (h₃: discount_rate = ((marked_price - selling_price) / marked_price) * 100) : 
  discount_rate = 15 :=
by
  sorry

end discount_rate_l2256_225604


namespace sin_pi_minus_alpha_l2256_225640

theorem sin_pi_minus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 Real.pi) (h2 : Real.cos α = 4 / 5) :
  Real.sin (Real.pi - α) = 3 / 5 := 
sorry

end sin_pi_minus_alpha_l2256_225640


namespace rationalize_denominator_l2256_225661

theorem rationalize_denominator : 
  let a := 32
  let b := 8
  let c := 2
  let d := 4
  (a / (c * Real.sqrt c) + b / (d * Real.sqrt c)) = (9 * Real.sqrt c) :=
by
  sorry

end rationalize_denominator_l2256_225661


namespace range_of_varphi_l2256_225665

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := 2 * Real.sin (ω * x + ϕ) + 1

theorem range_of_varphi (ω ϕ : ℝ) (h_ω_pos : ω > 0) (h_ϕ_bound : |ϕ| ≤ (Real.pi) / 2)
  (h_intersection : (∀ x, f x ω ϕ = -1 → (∃ k : ℤ, x = (k * Real.pi) / ω)))
  (h_f_gt_1 : (∀ x, -Real.pi / 12 < x ∧ x < Real.pi / 3 → f x ω ϕ > 1)) :
  ω = 2 → (Real.pi / 6 ≤ ϕ) ∧ (ϕ ≤ Real.pi / 3) :=
by
  sorry

end range_of_varphi_l2256_225665


namespace pairs_of_integers_l2256_225655

-- The main theorem to prove:
theorem pairs_of_integers (x y : ℤ) :
  y ^ 2 = x ^ 3 + 16 ↔ (x = 0 ∧ (y = 4 ∨ y = -4)) :=
by sorry

end pairs_of_integers_l2256_225655


namespace parabola_focus_at_centroid_l2256_225623

theorem parabola_focus_at_centroid (A B C : ℝ × ℝ) (a : ℝ) 
  (hA : A = (-1, 2))
  (hB : B = (3, 4))
  (hC : C = (4, -6))
  (h_focus : (a/4, 0) = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) :
  a = 8 :=
by
  sorry

end parabola_focus_at_centroid_l2256_225623


namespace sum_of_consecutive_integers_l2256_225672

theorem sum_of_consecutive_integers (x : ℕ) (h1 : x * (x + 1) = 930) : x + (x + 1) = 61 :=
sorry

end sum_of_consecutive_integers_l2256_225672


namespace Gumble_words_total_l2256_225652

noncomputable def num_letters := 25
noncomputable def exclude_B := 24

noncomputable def total_5_letters_or_less (n : ℕ) : ℕ :=
  if h : 1 ≤ n ∧ n ≤ 5 then num_letters^n - exclude_B^n else 0

noncomputable def total_Gumble_words : ℕ :=
  (total_5_letters_or_less 1) + (total_5_letters_or_less 2) + (total_5_letters_or_less 3) +
  (total_5_letters_or_less 4) + (total_5_letters_or_less 5)

theorem Gumble_words_total :
  total_Gumble_words = 1863701 := by
  sorry

end Gumble_words_total_l2256_225652


namespace find_speed_in_second_hour_l2256_225680

-- Define the given conditions as hypotheses
def speed_in_first_hour : ℝ := 50
def average_speed : ℝ := 55
def total_time : ℝ := 2

-- Define a function that represents the speed in the second hour
def speed_second_hour (s2 : ℝ) := 
  (speed_in_first_hour + s2) / total_time = average_speed

-- The statement to prove: the speed in the second hour is 60 km/h
theorem find_speed_in_second_hour : speed_second_hour 60 :=
by sorry

end find_speed_in_second_hour_l2256_225680


namespace find_number_l2256_225677

theorem find_number (n : ℕ) (h : 2 * 2 + n = 6) : n = 2 := by
  sorry

end find_number_l2256_225677


namespace fraction_of_income_from_tips_l2256_225681

variable (S T I : ℝ)

theorem fraction_of_income_from_tips (h1 : T = (5 / 2) * S) (h2 : I = S + T) : 
  T / I = 5 / 7 := by
  sorry

end fraction_of_income_from_tips_l2256_225681


namespace Q_contribution_l2256_225600

def P_contribution : ℕ := 4000
def P_months : ℕ := 12
def Q_months : ℕ := 8
def profit_ratio_PQ : ℚ := 2 / 3

theorem Q_contribution :
  ∃ X : ℕ, (P_contribution * P_months) / (X * Q_months) = profit_ratio_PQ → X = 9000 := 
by sorry

end Q_contribution_l2256_225600


namespace exponentiation_multiplication_identity_l2256_225614

theorem exponentiation_multiplication_identity :
  (-4)^(2010) * (-0.25)^(2011) = -0.25 :=
by
  sorry

end exponentiation_multiplication_identity_l2256_225614


namespace gcd_condition_implies_equality_l2256_225644

theorem gcd_condition_implies_equality (a b : ℤ) (h : ∀ n : ℤ, n ≥ 1 → Int.gcd (a + n) (b + n) > 1) : a = b :=
sorry

end gcd_condition_implies_equality_l2256_225644


namespace eval_f_at_800_l2256_225603

-- Given conditions in Lean 4:
def f : ℝ → ℝ := sorry -- placeholder for the function definition
axiom func_eqn (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom f_at_1000 : f 1000 = 4

-- The goal/proof statement:
theorem eval_f_at_800 : f 800 = 5 := sorry

end eval_f_at_800_l2256_225603


namespace max_cos_a_l2256_225613

theorem max_cos_a (a b : ℝ) (h : Real.cos (a - b) = Real.cos a - Real.cos b) : 
  Real.cos a ≤ 1 :=
by
  -- Proof goes here
  sorry

end max_cos_a_l2256_225613


namespace defective_chip_ratio_l2256_225641

theorem defective_chip_ratio (defective_chips total_chips : ℕ)
  (h1 : defective_chips = 15)
  (h2 : total_chips = 60000) :
  defective_chips / total_chips = 1 / 4000 :=
by
  sorry

end defective_chip_ratio_l2256_225641


namespace polynomial_comparison_l2256_225636

theorem polynomial_comparison {x : ℝ} :
  let A := (x - 3) * (x - 2)
  let B := (x + 1) * (x - 6)
  A > B :=
by 
  sorry -- Proof is omitted.

end polynomial_comparison_l2256_225636


namespace pages_written_in_a_year_l2256_225683

def pages_per_friend_per_letter : ℕ := 3
def friends : ℕ := 2
def letters_per_week : ℕ := 2
def weeks_per_year : ℕ := 52

theorem pages_written_in_a_year : 
  (pages_per_friend_per_letter * friends * letters_per_week * weeks_per_year) = 624 :=
by
  sorry

end pages_written_in_a_year_l2256_225683


namespace largest_angle_in_triangle_l2256_225667

theorem largest_angle_in_triangle (a b c : ℝ) (h1 : a + 3 * b + 3 * c = a ^ 2) (h2 : a + 3 * b - 3 * c = -4) 
  (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) (h6 : a + b > c) (h7 : a + c > b) (h8 : b + c > a) : 
  ∃ C : ℝ, C = 120 ∧ (by exact sorry) := sorry

end largest_angle_in_triangle_l2256_225667


namespace least_n_for_obtuse_triangle_l2256_225629

namespace obtuse_triangle

-- Define angles and n
def alpha (n : ℕ) : ℝ := 59 + n * 0.02
def beta : ℝ := 60
def gamma (n : ℕ) : ℝ := 61 - n * 0.02

-- Define condition for the triangle being obtuse
def is_obtuse_triangle (n : ℕ) : Prop :=
  alpha n > 90 ∨ gamma n > 90

-- Statement about the smallest n such that the triangle is obtuse
theorem least_n_for_obtuse_triangle : ∃ n : ℕ, n = 1551 ∧ is_obtuse_triangle n :=
by
  -- existence proof ends here, details for proof to be provided separately
  sorry

end obtuse_triangle

end least_n_for_obtuse_triangle_l2256_225629


namespace vasim_share_l2256_225617

theorem vasim_share (x : ℝ)
  (h_ratio : ∀ (f v r : ℝ), f = 3 * x ∧ v = 5 * x ∧ r = 6 * x)
  (h_diff : 6 * x - 3 * x = 900) :
  5 * x = 1500 :=
by
  try sorry

end vasim_share_l2256_225617


namespace seq_properties_l2256_225674

-- Conditions for the sequence a_n
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n * a n + 1

-- The statements to prove given the sequence definition
theorem seq_properties (a : ℕ → ℝ) (h : seq a) :
  (∀ n, a (n + 1) ≥ 2 * a n) ∧
  (∀ n, a (n + 1) / a n ≥ a n) ∧
  (∀ n, a n ≥ n * n - 2 * n + 2) :=
by
  sorry

end seq_properties_l2256_225674


namespace solve_inequalities_l2256_225659

theorem solve_inequalities :
  {x : ℤ | (x - 1) / 2 ≥ (x - 2) / 3 ∧ 2 * x - 5 < -3 * x} = {-1, 0} :=
by
  sorry

end solve_inequalities_l2256_225659


namespace arithmetic_problem_l2256_225643

theorem arithmetic_problem : 987 + 113 - 1000 = 100 :=
by
  sorry

end arithmetic_problem_l2256_225643


namespace range_of_x_l2256_225632

theorem range_of_x (x : ℝ) (hx1 : 1 / x ≤ 3) (hx2 : 1 / x ≥ -2) : x ≥ 1 / 3 := 
sorry

end range_of_x_l2256_225632


namespace number_half_reduction_l2256_225669

/-- Define the conditions -/
def percentage_more (percent : Float) (amount : Float) : Float := amount + (percent / 100) * amount

theorem number_half_reduction (x : Float) : percentage_more 30 75 = 97.5 → (x / 2) = 97.5 → x = 195 := by
  intros h1 h2
  sorry

end number_half_reduction_l2256_225669


namespace modified_cube_cubies_l2256_225691

structure RubiksCube :=
  (original_cubies : ℕ := 27)
  (removed_corners : ℕ := 8)
  (total_layers : ℕ := 3)
  (edges_per_layer : ℕ := 4)
  (faces_center_cubies : ℕ := 6)
  (center_cubie : ℕ := 1)

noncomputable def cubies_with_n_faces (n : ℕ) : ℕ :=
  if n = 4 then 12
  else if n = 1 then 6
  else if n = 0 then 1
  else 0

theorem modified_cube_cubies :
  (cubies_with_n_faces 4 = 12) ∧ (cubies_with_n_faces 1 = 6) ∧ (cubies_with_n_faces 0 = 1) := by
  sorry

end modified_cube_cubies_l2256_225691


namespace find_m_interval_l2256_225698

-- Define the sequence recursively
def sequence_recursive (x : ℕ → ℝ) (n : ℕ) : Prop :=
  x 0 = 5 ∧ ∀ n, x (n + 1) = (x n ^ 2 + 5 * x n + 4) / (x n + 6)

-- The left-hand side of the inequality
noncomputable def target_value : ℝ := 4 + 1 / (2 ^ 20)

-- The condition that the sequence element must satisfy
def condition (x : ℕ → ℝ) (m : ℕ) : Prop :=
  x m ≤ target_value

-- The proof problem statement, m lies within the given interval
theorem find_m_interval (x : ℕ → ℝ) (m : ℕ) :
  sequence_recursive x n →
  condition x m →
  81 ≤ m ∧ m ≤ 242 :=
sorry

end find_m_interval_l2256_225698


namespace remaining_people_l2256_225649

def initial_football_players : ℕ := 13
def initial_cheerleaders : ℕ := 16
def quitting_football_players : ℕ := 10
def quitting_cheerleaders : ℕ := 4

theorem remaining_people :
  (initial_football_players - quitting_football_players) 
  + (initial_cheerleaders - quitting_cheerleaders) = 15 := by
    -- Proof steps would go here, if required
    sorry

end remaining_people_l2256_225649


namespace isosceles_triangle_base_length_l2256_225654

theorem isosceles_triangle_base_length (a b : ℕ) (h1 : a = 7) (h2 : 2 * a + b = 24) : b = 10 := 
by 
  sorry

end isosceles_triangle_base_length_l2256_225654


namespace weight_of_replaced_person_l2256_225675

theorem weight_of_replaced_person 
  (avg_increase : ℝ) (new_person_weight : ℝ) (n : ℕ) (original_weight : ℝ) 
  (h1 : avg_increase = 2.5)
  (h2 : new_person_weight = 95)
  (h3 : n = 8)
  (h4 : original_weight = new_person_weight - n * avg_increase) : 
  original_weight = 75 := 
by
  sorry

end weight_of_replaced_person_l2256_225675


namespace find_pairs_l2256_225621

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ a b, (a, b) = (2, 2) ∨ (a, b) = (1, 3) ∨ (a, b) = (3, 3))
  ↔ (∃ a b, a > 0 ∧ b > 0 ∧ (a^3 * b - 1) % (a + 1) = 0 ∧ (b^3 * a + 1) % (b - 1) = 0) := by
  sorry

end find_pairs_l2256_225621


namespace sum_of_possible_values_of_x_l2256_225660

-- Define the concept of an isosceles triangle with specific angles
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

-- Define the angle sum property of a triangle
def angle_sum_property (a b c : ℝ) : Prop := 
  a + b + c = 180

-- State the problem using the given conditions and the required proof
theorem sum_of_possible_values_of_x :
  ∀ (x : ℝ), 
    is_isosceles_triangle 70 70 x ∨
    is_isosceles_triangle 70 x x ∨
    is_isosceles_triangle x 70 70 →
    angle_sum_property 70 70 x →
    angle_sum_property 70 x x →
    angle_sum_property x 70 70 →
    (70 + 55 + 40) = 165 :=
  by
    sorry

end sum_of_possible_values_of_x_l2256_225660


namespace squirrel_travel_distance_l2256_225676

theorem squirrel_travel_distance
  (height: ℝ)
  (circumference: ℝ)
  (vertical_rise: ℝ)
  (num_circuits: ℝ):
  height = 25 →
  circumference = 3 →
  vertical_rise = 5 →
  num_circuits = height / vertical_rise →
  (num_circuits * circumference) ^ 2 + height ^ 2 = 850 :=
by
  sorry

end squirrel_travel_distance_l2256_225676


namespace molecular_weight_of_BaF2_l2256_225602

theorem molecular_weight_of_BaF2 (mw_6_moles : ℕ → ℕ) (h : mw_6_moles 6 = 1050) : mw_6_moles 1 = 175 :=
by
  sorry

end molecular_weight_of_BaF2_l2256_225602


namespace cherry_tomatoes_ratio_l2256_225605

theorem cherry_tomatoes_ratio (T P B : ℕ) (M : ℕ := 3) (h1 : P = 4 * T) (h2 : B = 4 * P) (h3 : B / 3 = 32) :
  (T : ℚ) / M = 2 :=
by
  sorry

end cherry_tomatoes_ratio_l2256_225605


namespace compute_fraction_power_l2256_225628

theorem compute_fraction_power : (45000 ^ 3 / 15000 ^ 3) = 27 :=
by
  sorry

end compute_fraction_power_l2256_225628


namespace percent_decrease_l2256_225601

variable (OriginalPrice : ℝ) (SalePrice : ℝ)

theorem percent_decrease : 
  OriginalPrice = 100 → 
  SalePrice = 30 → 
  ((OriginalPrice - SalePrice) / OriginalPrice) * 100 = 70 :=
by
  intros h1 h2
  sorry

end percent_decrease_l2256_225601


namespace boat_speed_in_still_water_l2256_225611

theorem boat_speed_in_still_water (x : ℕ) 
  (h1 : x + 17 = 77) (h2 : x - 17 = 43) : x = 60 :=
by
  sorry

end boat_speed_in_still_water_l2256_225611


namespace sum_of_digits_l2256_225692

theorem sum_of_digits (x y z w : ℕ) 
  (hxz : z + x = 10) 
  (hyz : y + z = 9) 
  (hxw : x + w = 9) 
  (hx_ne_hy : x ≠ y)
  (hx_ne_hz : x ≠ z)
  (hx_ne_hw : x ≠ w)
  (hy_ne_hz : y ≠ z)
  (hy_ne_hw : y ≠ w)
  (hz_ne_hw : z ≠ w) :
  x + y + z + w = 19 := by
  sorry

end sum_of_digits_l2256_225692


namespace Patrick_fish_count_l2256_225625

variable (Angus Patrick Ollie : ℕ)

-- Conditions
axiom h1 : Ollie + 7 = Angus
axiom h2 : Angus = Patrick + 4
axiom h3 : Ollie = 5

-- Theorem statement
theorem Patrick_fish_count : Patrick = 8 := 
by
  sorry

end Patrick_fish_count_l2256_225625


namespace area_OBEC_is_19_5_l2256_225689

-- Definitions for the points and lines from the conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨5, 0⟩
def B : Point := ⟨0, 15⟩
def C : Point := ⟨6, 0⟩
def E : Point := ⟨3, 6⟩

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P1 P2 P3 : Point) : ℝ :=
  0.5 * |(P1.x * P2.y + P2.x * P3.y + P3.x * P1.y) - (P1.y * P2.x + P2.y * P3.x + P3.y * P1.x)|

-- Definitions of the vertices of the quadrilateral
def O : Point := ⟨0, 0⟩

-- Calculating the area of triangles OCE and OBE
def OCE_area : ℝ := triangle_area O C E
def OBE_area : ℝ := triangle_area O B E

-- Total area of quadrilateral OBEC
def OBEC_area : ℝ := OCE_area + OBE_area

-- Proof statement: The area of quadrilateral OBEC is 19.5
theorem area_OBEC_is_19_5 : OBEC_area = 19.5 := sorry

end area_OBEC_is_19_5_l2256_225689


namespace max_min_diff_c_l2256_225633

variable (a b c : ℝ)

theorem max_min_diff_c (h1 : a + b + c = 6) (h2 : a^2 + b^2 + c^2 = 18) : 
  (4 - 0) = 4 :=
by
  sorry

end max_min_diff_c_l2256_225633


namespace scientific_notation_35100_l2256_225697

theorem scientific_notation_35100 : 35100 = 3.51 * 10^4 :=
by
  sorry

end scientific_notation_35100_l2256_225697


namespace age_double_after_5_years_l2256_225662

-- Defining the current ages of the brothers
def older_brother_age := 15
def younger_brother_age := 5

-- Defining the condition
def after_x_years (x : ℕ) := older_brother_age + x = 2 * (younger_brother_age + x)

-- The main theorem with the condition
theorem age_double_after_5_years : after_x_years 5 :=
by sorry

end age_double_after_5_years_l2256_225662


namespace find_a_l2256_225682

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) 
    (h3 : 13 ∣ 53^2016 + a) : a = 12 := 
by 
  -- proof would be written here
  sorry

end find_a_l2256_225682


namespace cake_pieces_per_sister_l2256_225612

theorem cake_pieces_per_sister (total_pieces : ℕ) (percentage_eaten : ℕ) (sisters : ℕ)
  (h1 : total_pieces = 240) (h2 : percentage_eaten = 60) (h3 : sisters = 3) :
  (total_pieces * (1 - percentage_eaten / 100)) / sisters = 32 :=
by
  sorry

end cake_pieces_per_sister_l2256_225612


namespace inequality_transform_l2256_225626

theorem inequality_transform (x y : ℝ) (h : y > x) : 2 * y > 2 * x := 
  sorry

end inequality_transform_l2256_225626


namespace factorize_expression_l2256_225622

-- Defining the variables x and y as real numbers.
variable (x y : ℝ)

-- Statement of the proof problem.
theorem factorize_expression : 
  x * y^2 - x = x * (y + 1) * (y - 1) :=
sorry

end factorize_expression_l2256_225622


namespace purchase_price_of_article_l2256_225606

theorem purchase_price_of_article (P : ℝ) (h : 45 = 0.20 * P + 12) : P = 165 :=
by
  sorry

end purchase_price_of_article_l2256_225606


namespace original_number_is_10_l2256_225688

theorem original_number_is_10 (x : ℤ) (h : 2 * x + 3 = 23) : x = 10 :=
sorry

end original_number_is_10_l2256_225688


namespace value_of_expression_l2256_225673

variables {x y z w : ℝ}

theorem value_of_expression (h1 : 4 * x * z + y * w = 4) (h2 : x * w + y * z = 8) :
  (2 * x + y) * (2 * z + w) = 20 :=
by
  sorry

end value_of_expression_l2256_225673


namespace tan_alpha_eq_2_l2256_225668

theorem tan_alpha_eq_2 (α : ℝ) (h : Real.tan α = 2) : (Real.cos α + 3 * Real.sin α) / (3 * Real.cos α - Real.sin α) = 7 := by
  sorry

end tan_alpha_eq_2_l2256_225668


namespace find_first_number_l2256_225607

open Int

theorem find_first_number (A : ℕ) : 
  (Nat.lcm A 671 = 2310) ∧ (Nat.gcd A 671 = 61) → 
  A = 210 :=
by
  intro h
  sorry

end find_first_number_l2256_225607


namespace quadratic_inequality_solution_l2256_225678

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -12 * x^2 + 5 * x - 2 < 0 := by
  sorry

end quadratic_inequality_solution_l2256_225678


namespace ratio_a_to_b_l2256_225670

variable (a x c d b : ℝ)
variable (h1 : d = 3 * x + c)
variable (h2 : b = 4 * x)

theorem ratio_a_to_b : a / b = -1 / 4 := by 
  sorry

end ratio_a_to_b_l2256_225670


namespace potato_sales_l2256_225610

theorem potato_sales :
  let total_weight := 6500
  let damaged_weight := 150
  let bag_weight := 50
  let price_per_bag := 72
  let sellable_weight := total_weight - damaged_weight
  let num_bags := sellable_weight / bag_weight
  let total_revenue := num_bags * price_per_bag
  total_revenue = 9144 :=
by
  sorry

end potato_sales_l2256_225610


namespace even_function_is_a_4_l2256_225638

def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem even_function_is_a_4 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 4 := by
  sorry

end even_function_is_a_4_l2256_225638


namespace inverse_variation_solution_l2256_225653

theorem inverse_variation_solution :
  ∀ (x y k : ℝ),
    (x * y^3 = k) →
    (∃ k, x = 8 ∧ y = 1 ∧ k = 8) →
    (y = 2 → x = 1) :=
by
  intros x y k h1 h2 hy2
  sorry

end inverse_variation_solution_l2256_225653


namespace pencils_count_l2256_225664

theorem pencils_count (P L : ℕ) (h₁ : 6 * P = 5 * L) (h₂ : L = P + 4) : L = 24 :=
by sorry

end pencils_count_l2256_225664


namespace least_possible_sections_l2256_225650

theorem least_possible_sections (A C N : ℕ) (h1 : 7 * A = 11 * C) (h2 : N = A + C) : N = 18 :=
sorry

end least_possible_sections_l2256_225650


namespace sum_gcd_lcm_eight_twelve_l2256_225630

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l2256_225630


namespace vectors_parallel_iff_l2256_225663

-- Define the vectors a and b as given in the conditions
def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, m + 1)

-- Define what it means for two vectors to be parallel
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

-- The statement that we need to prove
theorem vectors_parallel_iff (m : ℝ) : parallel a (b m) ↔ m = 1 := by
  sorry

end vectors_parallel_iff_l2256_225663


namespace find_f_of_neg2_l2256_225657

theorem find_f_of_neg2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (3 * x + 1) = 9 * x ^ 2 - 6 * x + 5) : f (-2) = 20 :=
by
  sorry

end find_f_of_neg2_l2256_225657


namespace complement_U_A_l2256_225695

def U : Finset ℤ := {-2, -1, 0, 1, 2}
def A : Finset ℤ := {-2, -1, 1, 2}

theorem complement_U_A : (U \ A) = {0} := by
  sorry

end complement_U_A_l2256_225695


namespace fuel_needed_to_empty_l2256_225645

theorem fuel_needed_to_empty (x : ℝ) 
  (h1 : (3/4) * x - (1/3) * x = 15) :
  (1/3) * x = 12 :=
by 
-- Proving the result
sorry

end fuel_needed_to_empty_l2256_225645
