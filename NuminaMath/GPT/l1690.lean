import Mathlib

namespace NUMINAMATH_GPT_remainder_1234567_127_l1690_169061

theorem remainder_1234567_127 : (1234567 % 127) = 51 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_1234567_127_l1690_169061


namespace NUMINAMATH_GPT_stones_equally_distributed_l1690_169050

theorem stones_equally_distributed (n k : ℕ) 
    (h : ∃ piles : Fin n → ℕ, (∀ i j, 2 * piles i + piles j = k * n)) :
  ∃ m : ℕ, k = 2^m :=
by
  sorry

end NUMINAMATH_GPT_stones_equally_distributed_l1690_169050


namespace NUMINAMATH_GPT_american_literature_marks_l1690_169080

variable (History HomeEconomics PhysicalEducation Art AverageMarks NumberOfSubjects TotalMarks KnownMarks : ℕ)
variable (A : ℕ)

axiom marks_history : History = 75
axiom marks_home_economics : HomeEconomics = 52
axiom marks_physical_education : PhysicalEducation = 68
axiom marks_art : Art = 89
axiom average_marks : AverageMarks = 70
axiom number_of_subjects : NumberOfSubjects = 5

def total_marks (AverageMarks NumberOfSubjects : ℕ) : ℕ := AverageMarks * NumberOfSubjects

def known_marks (History HomeEconomics PhysicalEducation Art : ℕ) : ℕ := History + HomeEconomics + PhysicalEducation + Art

axiom total_marks_eq : TotalMarks = total_marks AverageMarks NumberOfSubjects
axiom known_marks_eq : KnownMarks = known_marks History HomeEconomics PhysicalEducation Art

theorem american_literature_marks :
  A = TotalMarks - KnownMarks := by
  sorry

end NUMINAMATH_GPT_american_literature_marks_l1690_169080


namespace NUMINAMATH_GPT_functional_equation_solution_l1690_169060

noncomputable def f : ℝ → ℝ := sorry 

theorem functional_equation_solution :
  (∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) →
  10 * f 2006 + f 0 = 20071 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1690_169060


namespace NUMINAMATH_GPT_solution_set_inequality_l1690_169010

theorem solution_set_inequality (m : ℝ) (x : ℝ) 
  (h : 3 - m < 0) : (2 - m) * x + m > 2 ↔ x < 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1690_169010


namespace NUMINAMATH_GPT_card_area_l1690_169001

theorem card_area (length width : ℕ) (h_length : length = 5) (h_width : width = 7)
  (h_area_after_shortening : (length - 1) * width = 24 ∨ length * (width - 1) = 24) :
  length * (width - 1) = 18 :=
by
  sorry

end NUMINAMATH_GPT_card_area_l1690_169001


namespace NUMINAMATH_GPT_time_to_cross_man_l1690_169058

-- Define the conversion from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℕ := (speed_kmh * 1000) / 3600

-- Given conditions
def length_of_train : ℕ := 150
def speed_of_train_kmh : ℕ := 180

-- Calculate speed in m/s
def speed_of_train_ms : ℕ := kmh_to_ms speed_of_train_kmh

-- Proof problem statement
theorem time_to_cross_man : (length_of_train : ℕ) / (speed_of_train_ms : ℕ) = 3 := by
  sorry

end NUMINAMATH_GPT_time_to_cross_man_l1690_169058


namespace NUMINAMATH_GPT_cookies_eq_23_l1690_169090

def total_packs : Nat := 27
def cakes : Nat := 4
def cookies : Nat := total_packs - cakes

theorem cookies_eq_23 : cookies = 23 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cookies_eq_23_l1690_169090


namespace NUMINAMATH_GPT_double_24_times_10_pow_8_l1690_169087

theorem double_24_times_10_pow_8 : 2 * (2.4 * 10^8) = 4.8 * 10^8 :=
by
  sorry

end NUMINAMATH_GPT_double_24_times_10_pow_8_l1690_169087


namespace NUMINAMATH_GPT_find_f_log2_5_l1690_169003

variable {f g : ℝ → ℝ}

-- f is an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- g is an odd function
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Given conditions
axiom f_even : is_even f
axiom g_odd : is_odd g
axiom f_g_equation : ∀ x, f x + g x = (2:ℝ)^x + x

-- Proof goal: Compute f(log_2 5) and show it equals 13/5
theorem find_f_log2_5 : f (Real.log 5 / Real.log 2) = (13:ℝ) / 5 := by
  sorry

end NUMINAMATH_GPT_find_f_log2_5_l1690_169003


namespace NUMINAMATH_GPT_parabola_directrix_eq_l1690_169095

theorem parabola_directrix_eq (a : ℝ) (h : - a / 4 = - (1 : ℝ) / 4) : a = 1 := by
  sorry

end NUMINAMATH_GPT_parabola_directrix_eq_l1690_169095


namespace NUMINAMATH_GPT_find_x_plus_y_squared_l1690_169051

variable (x y a b : ℝ)

def condition1 := x * y = b
def condition2 := (1 / (x ^ 2)) + (1 / (y ^ 2)) = a

theorem find_x_plus_y_squared (h1 : condition1 x y b) (h2 : condition2 x y a) : 
  (x + y) ^ 2 = a * b ^ 2 + 2 * b :=
by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_squared_l1690_169051


namespace NUMINAMATH_GPT_max_value_of_k_l1690_169049

theorem max_value_of_k (m : ℝ) (k : ℝ) (h1 : 0 < m) (h2 : m < 1/2) 
  (h3 : ∀ m, 0 < m → m < 1/2 → (1 / m + 2 / (1 - 2 * m) ≥ k)) : k = 8 :=
sorry

end NUMINAMATH_GPT_max_value_of_k_l1690_169049


namespace NUMINAMATH_GPT_students_catching_up_on_homework_l1690_169048

-- Definitions for the given conditions
def total_students := 120
def silent_reading_students := (2/5 : ℚ) * total_students
def board_games_students := (3/10 : ℚ) * total_students
def group_discussions_students := (1/8 : ℚ) * total_students
def other_activities_students := silent_reading_students + board_games_students + group_discussions_students
def catching_up_homework_students := total_students - other_activities_students

-- Statement of the proof problem
theorem students_catching_up_on_homework : catching_up_homework_students = 21 := by
  sorry

end NUMINAMATH_GPT_students_catching_up_on_homework_l1690_169048


namespace NUMINAMATH_GPT_sum_of_geometric_terms_l1690_169028

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q > 0, ∀ n, a (n + 1) = q * a n

theorem sum_of_geometric_terms {a : ℕ → ℝ} 
  (hseq : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_sum135 : a 1 + a 3 + a 5 = 21) :
  a 2 + a 4 + a 6 = 42 :=
sorry

end NUMINAMATH_GPT_sum_of_geometric_terms_l1690_169028


namespace NUMINAMATH_GPT_MaryHasBlueMarbles_l1690_169068

-- Define the number of blue marbles Dan has
def DanMarbles : Nat := 5

-- Define the relationship of Mary's marbles to Dan's marbles
def MaryMarbles : Nat := 2 * DanMarbles

-- State the theorem that we need to prove
theorem MaryHasBlueMarbles : MaryMarbles = 10 :=
by
  sorry

end NUMINAMATH_GPT_MaryHasBlueMarbles_l1690_169068


namespace NUMINAMATH_GPT_net_cannot_contain_2001_knots_l1690_169040

theorem net_cannot_contain_2001_knots (knots : Nat) (ropes_per_knot : Nat) (total_knots : knots = 2001) (ropes_per_knot_eq : ropes_per_knot = 3) :
  false :=
by
  sorry

end NUMINAMATH_GPT_net_cannot_contain_2001_knots_l1690_169040


namespace NUMINAMATH_GPT_even_function_has_specific_a_l1690_169016

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x ^ 2 + (2 * a ^ 2 - a) * x + 1

-- State the proof problem
theorem even_function_has_specific_a (a : ℝ) :
  (∀ x : ℝ, f a (-x) = f a x) → a = 1 / 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_even_function_has_specific_a_l1690_169016


namespace NUMINAMATH_GPT_floor_sub_y_eq_zero_l1690_169043

theorem floor_sub_y_eq_zero {y : ℝ} (h : ⌊y⌋ + ⌈y⌉ = 2 * y) : ⌊y⌋ - y = 0 :=
sorry

end NUMINAMATH_GPT_floor_sub_y_eq_zero_l1690_169043


namespace NUMINAMATH_GPT_ratio_of_B_to_C_l1690_169064

-- Definitions based on conditions
def A := 40
def C := A + 20
def total := 220
def B := total - A - C

-- Theorem statement
theorem ratio_of_B_to_C : B / C = 2 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_ratio_of_B_to_C_l1690_169064


namespace NUMINAMATH_GPT_correct_sum_after_digit_change_l1690_169065

theorem correct_sum_after_digit_change :
  let d := 7
  let e := 8
  let num1 := 935641
  let num2 := 471850
  let correct_sum := num1 + num2
  let new_sum := correct_sum + 10000
  new_sum = 1417491 := 
sorry

end NUMINAMATH_GPT_correct_sum_after_digit_change_l1690_169065


namespace NUMINAMATH_GPT_shaded_ratio_l1690_169074

theorem shaded_ratio (full_rectangles half_rectangles : ℕ) (n m : ℕ) (rectangle_area shaded_area total_area : ℝ)
  (h1 : n = 4) (h2 : m = 5) (h3 : rectangle_area = n * m) 
  (h4 : full_rectangles = 3) (h5 : half_rectangles = 4)
  (h6 : shaded_area = full_rectangles * 1 + 0.5 * half_rectangles * 1)
  (h7 : total_area = rectangle_area) :
  shaded_area / total_area = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_shaded_ratio_l1690_169074


namespace NUMINAMATH_GPT_evaluate_fg_sum_at_1_l1690_169063

def f (x : ℚ) : ℚ := (4 * x^2 + 3 * x + 6) / (x^2 + 2 * x + 5)
def g (x : ℚ) : ℚ := x + 1

theorem evaluate_fg_sum_at_1 : f (g 1) + g (f 1) = 497 / 104 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fg_sum_at_1_l1690_169063


namespace NUMINAMATH_GPT_projectile_height_35_l1690_169029

theorem projectile_height_35 (t : ℝ) : 
  (∃ t : ℝ, -4.9 * t ^ 2 + 30 * t = 35 ∧ t > 0) → t = 10 / 7 := 
sorry

end NUMINAMATH_GPT_projectile_height_35_l1690_169029


namespace NUMINAMATH_GPT_units_digit_7_pow_2023_l1690_169039

-- We start by defining a function to compute units digit of powers of 7 modulo 10.
def units_digit_of_7_pow (n : ℕ) : ℕ :=
  (7 ^ n) % 10

-- Define the problem statement: the units digit of 7^2023 is equal to 3.
theorem units_digit_7_pow_2023 : units_digit_of_7_pow 2023 = 3 := sorry

end NUMINAMATH_GPT_units_digit_7_pow_2023_l1690_169039


namespace NUMINAMATH_GPT_no_information_loss_chart_is_stem_and_leaf_l1690_169030

theorem no_information_loss_chart_is_stem_and_leaf :
  "The correct chart with no information loss" = "Stem-and-leaf plot" :=
sorry

end NUMINAMATH_GPT_no_information_loss_chart_is_stem_and_leaf_l1690_169030


namespace NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_1994_l1690_169031

theorem rightmost_three_digits_of_7_pow_1994 :
  (7 ^ 1994) % 800 = 49 :=
by
  sorry

end NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_1994_l1690_169031


namespace NUMINAMATH_GPT_graph_of_eq_hyperbola_l1690_169092

theorem graph_of_eq_hyperbola (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 1 → ∃ a b : ℝ, a * b = x * y ∧ a * b = 1/2 := by
  sorry

end NUMINAMATH_GPT_graph_of_eq_hyperbola_l1690_169092


namespace NUMINAMATH_GPT_fraction_expression_l1690_169052

theorem fraction_expression :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 9) = 531 / 322 :=
by sorry

end NUMINAMATH_GPT_fraction_expression_l1690_169052


namespace NUMINAMATH_GPT_josh_marbles_earlier_l1690_169020

-- Define the conditions
def marbles_lost : ℕ := 11
def marbles_now : ℕ := 8

-- Define the problem statement
theorem josh_marbles_earlier : marbles_lost + marbles_now = 19 :=
by
  sorry

end NUMINAMATH_GPT_josh_marbles_earlier_l1690_169020


namespace NUMINAMATH_GPT_Linda_total_sales_l1690_169096

theorem Linda_total_sales (necklaces_sold : ℕ) (rings_sold : ℕ) 
    (necklace_price : ℕ) (ring_price : ℕ) 
    (total_sales : ℕ) : 
    necklaces_sold = 4 → 
    rings_sold = 8 → 
    necklace_price = 12 → 
    ring_price = 4 → 
    total_sales = necklaces_sold * necklace_price + rings_sold * ring_price → 
    total_sales = 80 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end NUMINAMATH_GPT_Linda_total_sales_l1690_169096


namespace NUMINAMATH_GPT_two_solutions_exist_l1690_169062

theorem two_solutions_exist 
  (a b c : ℝ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_equation : (1 / a) + (1 / b) + (1 / c) = (1 / (a + b + c))) : 
  ∃ (a' b' c' : ℝ), 
    ((a' = 1/3 ∧ b' = 1/3 ∧ c' = 1/3) ∨ (a' = -1/3 ∧ b' = -1/3 ∧ c' = -1/3)) := 
sorry

end NUMINAMATH_GPT_two_solutions_exist_l1690_169062


namespace NUMINAMATH_GPT_total_wheels_l1690_169000

def cars := 2
def car_wheels := 4
def bikes_with_one_wheel := 1
def bikes_with_two_wheels := 2
def trash_can_wheels := 2
def tricycle_wheels := 3
def roller_skate_wheels := 3 -- since one is missing a wheel
def wheelchair_wheels := 6 -- 4 large + 2 small wheels
def wagon_wheels := 4

theorem total_wheels : cars * car_wheels + 
                        bikes_with_one_wheel * 1 + 
                        bikes_with_two_wheels * 2 + 
                        trash_can_wheels + 
                        tricycle_wheels + 
                        roller_skate_wheels + 
                        wheelchair_wheels + 
                        wagon_wheels = 31 :=
by
  sorry

end NUMINAMATH_GPT_total_wheels_l1690_169000


namespace NUMINAMATH_GPT_linear_in_one_variable_linear_in_two_variables_l1690_169038

namespace MathProof

-- Definition of the equation
def equation (k x y : ℝ) : ℝ := (k^2 - 1) * x^2 + (k + 1) * x + (k - 7) * y - (k + 2)

-- Theorem for linear equation in one variable
theorem linear_in_one_variable (k : ℝ) (x y : ℝ) :
  k = -1 → equation k x y = 0 → ∃ y' : ℝ, equation k 0 y' = 0 :=
by
  sorry

-- Theorem for linear equation in two variables
theorem linear_in_two_variables (k : ℝ) (x y : ℝ) :
  k = 1 → equation k x y = 0 → ∃ x' y' : ℝ, equation k x' y' = 0 :=
by
  sorry

end MathProof

end NUMINAMATH_GPT_linear_in_one_variable_linear_in_two_variables_l1690_169038


namespace NUMINAMATH_GPT_intersection_A_B_l1690_169076

def A : Set ℤ := {x | x > 0 }
def B : Set ℤ := {-1, 0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = {1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1690_169076


namespace NUMINAMATH_GPT_graveling_cost_is_3900_l1690_169042

noncomputable def cost_of_graveling_roads 
  (length : ℕ) (breadth : ℕ) (width_road : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  let area_road_length := length * width_road
  let area_road_breadth := (breadth - width_road) * width_road
  let total_area := area_road_length + area_road_breadth
  total_area * cost_per_sq_m

theorem graveling_cost_is_3900 :
  cost_of_graveling_roads 80 60 10 3 = 3900 := 
by 
  unfold cost_of_graveling_roads
  sorry

end NUMINAMATH_GPT_graveling_cost_is_3900_l1690_169042


namespace NUMINAMATH_GPT_find_a5_div_b5_l1690_169069

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ := n * (a 0 + a (n - 1)) / 2

-- Main statement
theorem find_a5_div_b5 (a b : ℕ → ℤ) (S T : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : is_arithmetic_sequence b)
  (h3 : ∀ n : ℕ, S n = sum_first_n_terms a n)
  (h4 : ∀ n : ℕ, T n = sum_first_n_terms b n)
  (h5 : ∀ n : ℕ, S n * (3 * n + 1) = 2 * n * T n) :
  (a 5 : ℚ) / b 5 = 9 / 14 :=
by
  sorry

end NUMINAMATH_GPT_find_a5_div_b5_l1690_169069


namespace NUMINAMATH_GPT_scientific_notation_of_1040000000_l1690_169024

theorem scientific_notation_of_1040000000 : (1.04 * 10^9 = 1040000000) :=
by
  -- Math proof steps can be added here
  sorry

end NUMINAMATH_GPT_scientific_notation_of_1040000000_l1690_169024


namespace NUMINAMATH_GPT_total_distance_traveled_is_960_l1690_169071

-- Definitions of conditions
def first_day_distance : ℝ := 100
def second_day_distance : ℝ := 3 * first_day_distance
def third_day_distance : ℝ := second_day_distance + 110
def fourth_day_distance : ℝ := 150

-- The total distance traveled in four days
def total_distance : ℝ := first_day_distance + second_day_distance + third_day_distance + fourth_day_distance

-- Theorem statement
theorem total_distance_traveled_is_960 :
  total_distance = 960 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_is_960_l1690_169071


namespace NUMINAMATH_GPT_point_in_quadrant_I_l1690_169046

theorem point_in_quadrant_I (x y : ℝ) (h1 : 4 * x + 6 * y = 24) (h2 : y = x + 3) : x > 0 ∧ y > 0 :=
by sorry

end NUMINAMATH_GPT_point_in_quadrant_I_l1690_169046


namespace NUMINAMATH_GPT_root_equation_l1690_169014

theorem root_equation (p q : ℝ) (hp : 3 * p^2 - 5 * p - 7 = 0)
                                  (hq : 3 * q^2 - 5 * q - 7 = 0) :
            (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := 
by sorry

end NUMINAMATH_GPT_root_equation_l1690_169014


namespace NUMINAMATH_GPT_triangle_obtuse_at_15_l1690_169078

-- Define the initial angles of the triangle
def x0 : ℝ := 59.999
def y0 : ℝ := 60
def z0 : ℝ := 60.001

-- Define the recurrence relations for the angles
def x (n : ℕ) : ℝ := (-2)^n * (x0 - 60) + 60
def y (n : ℕ) : ℝ := (-2)^n * (y0 - 60) + 60
def z (n : ℕ) : ℝ := (-2)^n * (z0 - 60) + 60

-- Define the obtuseness condition
def is_obtuse (a : ℝ) : Prop := a > 90

-- The main theorem stating the least positive integer n is 15 for which the triangle A_n B_n C_n is obtuse
theorem triangle_obtuse_at_15 : ∃ n : ℕ, n > 0 ∧ 
  (is_obtuse (x n) ∨ is_obtuse (y n) ∨ is_obtuse (z n)) ∧ n = 15 :=
sorry

end NUMINAMATH_GPT_triangle_obtuse_at_15_l1690_169078


namespace NUMINAMATH_GPT_negation_universal_proposition_l1690_169079

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, 0 < x → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, 0 < x ∧ x^2 + x + 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_universal_proposition_l1690_169079


namespace NUMINAMATH_GPT_Albert_eats_48_slices_l1690_169034

theorem Albert_eats_48_slices (large_pizzas : ℕ) (small_pizzas : ℕ) (slices_large : ℕ) (slices_small : ℕ) 
  (h1 : large_pizzas = 2) (h2 : small_pizzas = 2) (h3 : slices_large = 16) (h4 : slices_small = 8) :
  (large_pizzas * slices_large + small_pizzas * slices_small) = 48 := 
by 
  -- sorry is used to skip the proof.
  sorry

end NUMINAMATH_GPT_Albert_eats_48_slices_l1690_169034


namespace NUMINAMATH_GPT_triangle_is_isosceles_l1690_169099

variable (a b m_a m_b : ℝ)

-- Conditions: 
-- A circle touches two sides of a triangle (denoted as a and b).
-- The circle also touches the medians m_a and m_b drawn to these sides.
-- Given equations:
axiom Eq1 : (1/2) * a + (1/3) * m_b = (1/2) * b + (1/3) * m_a
axiom Eq3 : (1/2) * a + m_b = (1/2) * b + m_a

-- Question: Prove that the triangle is isosceles, i.e., a = b
theorem triangle_is_isosceles : a = b :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l1690_169099


namespace NUMINAMATH_GPT_sqrt_sum_difference_product_l1690_169082

open Real

theorem sqrt_sum_difference_product :
  (sqrt 3 + sqrt 2) * (sqrt 3 - sqrt 2) = 1 := by
  sorry

end NUMINAMATH_GPT_sqrt_sum_difference_product_l1690_169082


namespace NUMINAMATH_GPT_g_five_l1690_169098

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_multiplicative : ∀ x y : ℝ, g (x * y) = g x * g y
axiom g_zero : g 0 = 0
axiom g_one : g 1 = 1

theorem g_five : g 5 = 1 := by
  sorry

end NUMINAMATH_GPT_g_five_l1690_169098


namespace NUMINAMATH_GPT_triangle_is_obtuse_l1690_169017

theorem triangle_is_obtuse
  (A B C : ℝ)
  (h1 : 3 * A > 5 * B)
  (h2 : 3 * C < 2 * B)
  (h3 : A + B + C = 180) :
  A > 90 :=
sorry

end NUMINAMATH_GPT_triangle_is_obtuse_l1690_169017


namespace NUMINAMATH_GPT_cost_of_one_unit_each_l1690_169057

variables (x y z : ℝ)

theorem cost_of_one_unit_each
  (h1 : 2 * x + 3 * y + z = 130)
  (h2 : 3 * x + 5 * y + z = 205) :
  x + y + z = 55 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_unit_each_l1690_169057


namespace NUMINAMATH_GPT_complement_U_A_correct_l1690_169047

-- Define the universal set U and set A
def U : Set Int := {-1, 0, 2}
def A : Set Int := {-1, 0}

-- Define the complement of A in U
def complement_U_A : Set Int := {x | x ∈ U ∧ x ∉ A}

-- Theorem stating the required proof
theorem complement_U_A_correct : complement_U_A = {2} :=
by
  sorry -- Proof will be filled in

end NUMINAMATH_GPT_complement_U_A_correct_l1690_169047


namespace NUMINAMATH_GPT_contractor_laborers_l1690_169006

theorem contractor_laborers (x : ℕ) (h1 : 15 * x = 20 * (x - 5)) : x = 20 :=
by sorry

end NUMINAMATH_GPT_contractor_laborers_l1690_169006


namespace NUMINAMATH_GPT_parallel_vectors_sum_is_six_l1690_169021

theorem parallel_vectors_sum_is_six (x y : ℝ) :
  let a := (4, -1, 1)
  let b := (x, y, 2)
  (x / 4 = 2) ∧ (y / -1 = 2) →
  x + y = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_parallel_vectors_sum_is_six_l1690_169021


namespace NUMINAMATH_GPT_count_parallelograms_392_l1690_169070

-- Define the conditions in Lean
def is_lattice_point (x y : ℕ) : Prop :=
  ∃ q : ℕ, x = q ∧ y = q

def on_line_y_eq_x (x y : ℕ) : Prop :=
  y = x ∧ is_lattice_point x y

def on_line_y_eq_mx (x y : ℕ) (m : ℕ) : Prop :=
  y = m * x ∧ is_lattice_point x y ∧ m > 1

def area_parallelogram (q s m : ℕ) : ℕ :=
  (m - 1) * q * s

-- Define the target theorem
theorem count_parallelograms_392 :
  (∀ (q s m : ℕ),
    on_line_y_eq_x q q →
    on_line_y_eq_mx s (m * s) m →
    area_parallelogram q s m = 250000) →
  (∃! n : ℕ, n = 392) :=
sorry

end NUMINAMATH_GPT_count_parallelograms_392_l1690_169070


namespace NUMINAMATH_GPT_largest_pies_without_ingredients_l1690_169018

variable (total_pies : ℕ) (chocolate_pies marshmallow_pies cayenne_pies soy_nut_pies : ℕ)
variable (b : total_pies = 36)
variable (c : chocolate_pies = total_pies / 2)
variable (m : marshmallow_pies = 2 * total_pies / 3)
variable (k : cayenne_pies = 3 * total_pies / 4)
variable (s : soy_nut_pies = total_pies / 6)

theorem largest_pies_without_ingredients (total_pies chocolate_pies marshmallow_pies cayenne_pies soy_nut_pies : ℕ)
  (b : total_pies = 36)
  (c : chocolate_pies = total_pies / 2)
  (m : marshmallow_pies = 2 * total_pies / 3)
  (k : cayenne_pies = 3 * total_pies / 4)
  (s : soy_nut_pies = total_pies / 6) :
  9 = total_pies - chocolate_pies - marshmallow_pies - cayenne_pies - soy_nut_pies + 3 * cayenne_pies := 
by
  sorry

end NUMINAMATH_GPT_largest_pies_without_ingredients_l1690_169018


namespace NUMINAMATH_GPT_chipmunk_families_left_l1690_169041

theorem chipmunk_families_left (orig : ℕ) (left : ℕ) (h1 : orig = 86) (h2 : left = 65) : orig - left = 21 := by
  sorry

end NUMINAMATH_GPT_chipmunk_families_left_l1690_169041


namespace NUMINAMATH_GPT_child_ticket_price_l1690_169075

theorem child_ticket_price
    (num_people : ℕ)
    (num_adults : ℕ)
    (num_seniors : ℕ)
    (num_children : ℕ)
    (adult_ticket_cost : ℝ)
    (senior_discount : ℝ)
    (total_bill : ℝ) :
    num_people = 50 →
    num_adults = 25 →
    num_seniors = 15 →
    num_children = 10 →
    adult_ticket_cost = 15 →
    senior_discount = 0.25 →
    total_bill = 600 →
    ∃ x : ℝ, x = 5.63 :=
by {
  sorry
}

end NUMINAMATH_GPT_child_ticket_price_l1690_169075


namespace NUMINAMATH_GPT_jamesOreos_count_l1690_169023

noncomputable def jamesOreos (jordanOreos : ℕ) : ℕ := 4 * jordanOreos + 7

theorem jamesOreos_count (J : ℕ) (h1 : J + jamesOreos J = 52) : jamesOreos J = 43 :=
by
  sorry

end NUMINAMATH_GPT_jamesOreos_count_l1690_169023


namespace NUMINAMATH_GPT_sum_powers_l1690_169077

theorem sum_powers {a b : ℝ}
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^11 + b^11 = 199 :=
by
  sorry

end NUMINAMATH_GPT_sum_powers_l1690_169077


namespace NUMINAMATH_GPT_expected_value_of_N_l1690_169037

noncomputable def expected_value_N : ℝ :=
  30

theorem expected_value_of_N :
  -- Suppose Bob chooses a 4-digit binary string uniformly at random,
  -- and examines an infinite sequence of independent random binary bits.
  -- Let N be the least number of bits Bob has to examine to find his chosen string.
  -- Then the expected value of N is 30.
  expected_value_N = 30 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_N_l1690_169037


namespace NUMINAMATH_GPT_greatest_possible_n_l1690_169035

theorem greatest_possible_n (n : ℤ) (h1 : 102 * n^2 ≤ 8100) : n ≤ 8 :=
sorry

end NUMINAMATH_GPT_greatest_possible_n_l1690_169035


namespace NUMINAMATH_GPT_solution_a_l1690_169025

noncomputable def problem_a (a b c y : ℕ) : Prop :=
  a + b + c = 30 ∧ b + c + y = 30 ∧ a = 2 ∧ y = 3

theorem solution_a (a b c y x : ℕ)
  (h : problem_a a b c y)
  : x = 25 :=
by sorry

end NUMINAMATH_GPT_solution_a_l1690_169025


namespace NUMINAMATH_GPT_area_inequality_l1690_169056

theorem area_inequality 
  (α β γ : ℝ) 
  (P Q S : ℝ) 
  (h1 : P / Q = α * β * γ) 
  (h2 : S = Q * (α + 1) * (β + 1) * (γ + 1)) : 
  (S ^ (1 / 3)) ≥ (P ^ (1 / 3)) + (Q ^ (1 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_area_inequality_l1690_169056


namespace NUMINAMATH_GPT_real_part_of_z_l1690_169081

variable (z : ℂ) (a : ℝ)

noncomputable def condition1 : Prop := z / (2 + (a : ℂ) * Complex.I) = 2 / (1 + Complex.I)
noncomputable def condition2 : Prop := z.im = -3

theorem real_part_of_z (h1 : condition1 z a) (h2 : condition2 z) : z.re = 1 := sorry

end NUMINAMATH_GPT_real_part_of_z_l1690_169081


namespace NUMINAMATH_GPT_parabola_focus_distance_l1690_169073

theorem parabola_focus_distance
  (F P Q : ℝ × ℝ)
  (hF : F = (1 / 2, 0))
  (hP : ∃ y, P = (2 * y^2, y))
  (hQ : Q = (1 / 2, Q.2))
  (h_parallel : P.2 = Q.2)
  (h_distance : dist P Q = dist Q F) :
  dist P F = 2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_distance_l1690_169073


namespace NUMINAMATH_GPT_square_side_length_eq_area_and_perimeter_l1690_169059

theorem square_side_length_eq_area_and_perimeter (a : ℝ) (h : a^2 = 4 * a) : a = 4 :=
by sorry

end NUMINAMATH_GPT_square_side_length_eq_area_and_perimeter_l1690_169059


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l1690_169005

def p (x1 x2 : ℝ) : Prop := x1 > 1 ∧ x2 > 1
def q (x1 x2 : ℝ) : Prop := x1 + x2 > 2 ∧ x1 * x2 > 1

theorem p_sufficient_not_necessary_for_q : 
  (∀ x1 x2 : ℝ, p x1 x2 → q x1 x2) ∧ ¬ (∀ x1 x2 : ℝ, q x1 x2 → p x1 x2) :=
by 
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l1690_169005


namespace NUMINAMATH_GPT_range_of_a1_of_arithmetic_sequence_l1690_169007

theorem range_of_a1_of_arithmetic_sequence
  {a : ℕ → ℝ} (S : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum: ∀ n, S n = (n + 1) * (a 0 + a n) / 2)
  (h_min: ∀ n > 0, S n ≥ S 0)
  (h_S1: S 0 = 10) :
  -30 < a 0 ∧ a 0 < -27 := 
sorry

end NUMINAMATH_GPT_range_of_a1_of_arithmetic_sequence_l1690_169007


namespace NUMINAMATH_GPT_area_ratio_equilateral_triangl_l1690_169027

theorem area_ratio_equilateral_triangl (x : ℝ) :
  let sA : ℝ := x 
  let sB : ℝ := 3 * sA
  let sC : ℝ := 5 * sA
  let sD : ℝ := 4 * sA
  let area_ABC := (Real.sqrt 3 / 4) * (sA ^ 2)
  let s := (sB + sC + sD) / 2
  let area_A'B'C' := Real.sqrt (s * (s - sB) * (s - sC) * (s - sD))
  (area_A'B'C' / area_ABC) = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_area_ratio_equilateral_triangl_l1690_169027


namespace NUMINAMATH_GPT_hcf_of_numbers_is_five_l1690_169066

theorem hcf_of_numbers_is_five (a b x : ℕ) (ratio : a = 3 * x) (ratio_b : b = 4 * x)
  (lcm_ab : Nat.lcm a b = 60) (hcf_ab : Nat.gcd a b = 5) : Nat.gcd a b = 5 :=
by
  sorry

end NUMINAMATH_GPT_hcf_of_numbers_is_five_l1690_169066


namespace NUMINAMATH_GPT_union_of_S_and_T_l1690_169097

def S : Set ℕ := {1, 3, 5}
def T : Set ℕ := {3, 6}

theorem union_of_S_and_T : S ∪ T = {1, 3, 5, 6} := 
by
  sorry

end NUMINAMATH_GPT_union_of_S_and_T_l1690_169097


namespace NUMINAMATH_GPT_solve_quadratic_and_cubic_eqns_l1690_169022

-- Define the conditions as predicates
def eq1 (x : ℝ) : Prop := (x - 1)^2 = 4
def eq2 (x : ℝ) : Prop := (x - 2)^3 = -125

-- State the theorem
theorem solve_quadratic_and_cubic_eqns : 
  (∃ x : ℝ, eq1 x ∧ (x = 3 ∨ x = -1)) ∧ (∃ x : ℝ, eq2 x ∧ x = -3) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_and_cubic_eqns_l1690_169022


namespace NUMINAMATH_GPT_sector_area_l1690_169054

/--
The area of a sector with radius 6cm and central angle 15° is (3 * π / 2) cm².
-/
theorem sector_area (R : ℝ) (θ : ℝ) (h_radius : R = 6) (h_angle : θ = 15) :
    (S : ℝ) = (3 * Real.pi / 2) := by
  sorry

end NUMINAMATH_GPT_sector_area_l1690_169054


namespace NUMINAMATH_GPT_min_mn_value_l1690_169044

theorem min_mn_value (m n : ℕ) (hmn : m > n) (hn : n ≥ 1) 
  (hdiv : 1000 ∣ 1978 ^ m - 1978 ^ n) : m + n = 106 :=
sorry

end NUMINAMATH_GPT_min_mn_value_l1690_169044


namespace NUMINAMATH_GPT_common_ratio_neg_two_l1690_169067

theorem common_ratio_neg_two (a : ℕ → ℝ) (q : ℝ) 
  (h : ∀ n, a (n + 1) = a n * q)
  (H : 8 * a 2 + a 5 = 0) : 
  q = -2 :=
sorry

end NUMINAMATH_GPT_common_ratio_neg_two_l1690_169067


namespace NUMINAMATH_GPT_constant_sequence_from_conditions_l1690_169011

variable (k b : ℝ) [Nontrivial ℝ]
variable (a_n : ℕ → ℝ)

-- Define the conditions function
def cond1 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (q : ℝ), ∀ n, a_n (n + 1) = q * a_n n) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b = m * (k * a_n n + b))

def cond2 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (d : ℝ), ∀ n, a_n (n + 1) = a_n n + d) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b = m * (k * a_n n + b))

def cond3 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (q : ℝ), ∀ n, a_n (n + 1) = q * a_n n) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b - (k * a_n n + b) = m)

-- Lean statement to prove the problem
theorem constant_sequence_from_conditions (k b : ℝ) [Nontrivial ℝ] (a_n : ℕ → ℝ) :
  (cond1 k b a_n ∨ cond2 k b a_n ∨ cond3 k b a_n) → 
  ∃ c : ℝ, ∀ n, a_n n = c :=
by
  -- To be proven
  intros
  sorry

end NUMINAMATH_GPT_constant_sequence_from_conditions_l1690_169011


namespace NUMINAMATH_GPT_semesters_needed_l1690_169009

def total_credits : ℕ := 120
def credits_per_class : ℕ := 3
def classes_per_semester : ℕ := 5

theorem semesters_needed (h1 : total_credits = 120)
                         (h2 : credits_per_class = 3)
                         (h3 : classes_per_semester = 5) :
  total_credits / (credits_per_class * classes_per_semester) = 8 := 
by {
  sorry
}

end NUMINAMATH_GPT_semesters_needed_l1690_169009


namespace NUMINAMATH_GPT_kareem_has_largest_final_number_l1690_169004

def jose_final : ℕ := (15 - 2) * 4 + 5
def thuy_final : ℕ := (15 * 3 - 3) - 4
def kareem_final : ℕ := ((20 - 3) + 4) * 3

theorem kareem_has_largest_final_number :
  kareem_final > jose_final ∧ kareem_final > thuy_final := 
by 
  sorry

end NUMINAMATH_GPT_kareem_has_largest_final_number_l1690_169004


namespace NUMINAMATH_GPT_no_real_roots_range_l1690_169089

theorem no_real_roots_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 ≠ 0) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_range_l1690_169089


namespace NUMINAMATH_GPT_fourth_student_guess_l1690_169083

theorem fourth_student_guess :
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let total := first_guess + second_guess + third_guess
  let average := total / 3
  let fourth_guess := average + 25
  fourth_guess = 525 :=
by
  sorry

end NUMINAMATH_GPT_fourth_student_guess_l1690_169083


namespace NUMINAMATH_GPT_little_twelve_conference_games_l1690_169093

def teams_in_division : ℕ := 6
def divisions : ℕ :=  2

def games_within_division (t : ℕ) : ℕ := (t * (t - 1)) / 2 * 2

def games_between_divisions (d t : ℕ) : ℕ := t * t

def total_conference_games (d t : ℕ) : ℕ :=
  d * games_within_division t + games_between_divisions d t

theorem little_twelve_conference_games :
  total_conference_games divisions teams_in_division = 96 :=
by
  sorry

end NUMINAMATH_GPT_little_twelve_conference_games_l1690_169093


namespace NUMINAMATH_GPT_leak_empty_tank_time_l1690_169013

theorem leak_empty_tank_time (fill_time_A : ℝ) (fill_time_A_with_leak : ℝ) (leak_empty_time : ℝ) :
  fill_time_A = 6 → fill_time_A_with_leak = 9 → leak_empty_time = 18 :=
by
  intros hA hL
  -- Here follows the proof we skip
  sorry

end NUMINAMATH_GPT_leak_empty_tank_time_l1690_169013


namespace NUMINAMATH_GPT_horizontal_asymptote_at_3_l1690_169088

noncomputable def rational_function (x : ℝ) : ℝ :=
  (15 * x^4 + 2 * x^3 + 11 * x^2 + 6 * x + 4) / (5 * x^4 + x^3 + 10 * x^2 + 4 * x + 2)

theorem horizontal_asymptote_at_3 : 
  (∀ ε > 0, ∃ N > 0, ∀ x > N, |rational_function x - 3| < ε) := 
by
  sorry

end NUMINAMATH_GPT_horizontal_asymptote_at_3_l1690_169088


namespace NUMINAMATH_GPT_math_bonanza_2016_8_l1690_169002

def f (x : ℕ) := x^2 + x + 1

theorem math_bonanza_2016_8 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : f p = f q + 242) (hpq : p > q) :
  (p, q) = (61, 59) :=
by sorry

end NUMINAMATH_GPT_math_bonanza_2016_8_l1690_169002


namespace NUMINAMATH_GPT_compare_2_pow_n_n_sq_l1690_169026

theorem compare_2_pow_n_n_sq (n : ℕ) (h : n > 0) :
  (n = 1 → 2^n > n^2) ∧
  (n = 2 → 2^n = n^2) ∧
  (n = 3 → 2^n < n^2) ∧
  (n = 4 → 2^n = n^2) ∧
  (n ≥ 5 → 2^n > n^2) :=
by sorry

end NUMINAMATH_GPT_compare_2_pow_n_n_sq_l1690_169026


namespace NUMINAMATH_GPT_nth_equation_l1690_169019

theorem nth_equation (n : ℕ) : 
  1 + 6 * n = (3 * n + 1) ^ 2 - 9 * n ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_nth_equation_l1690_169019


namespace NUMINAMATH_GPT_both_participation_correct_l1690_169091

-- Define the number of total participants
def total_participants : ℕ := 50

-- Define the number of participants in Chinese competition
def chinese_participants : ℕ := 30

-- Define the number of participants in Mathematics competition
def math_participants : ℕ := 38

-- Define the number of people who do not participate in either competition
def neither_participants : ℕ := 2

-- Define the number of people who participate in both competitions
def both_participants : ℕ :=
  chinese_participants + math_participants - (total_participants - neither_participants)

-- The theorem we want to prove
theorem both_participation_correct : both_participants = 20 :=
by
  sorry

end NUMINAMATH_GPT_both_participation_correct_l1690_169091


namespace NUMINAMATH_GPT_sector_angle_l1690_169012

theorem sector_angle (r l θ : ℝ) (h : 2 * r + l = π * r) : θ = π - 2 :=
sorry

end NUMINAMATH_GPT_sector_angle_l1690_169012


namespace NUMINAMATH_GPT_greatest_integer_b_not_in_range_of_quadratic_l1690_169072

theorem greatest_integer_b_not_in_range_of_quadratic :
  ∀ b : ℤ, (∀ x : ℝ, x^2 + (b : ℝ) * x + 20 ≠ 5) ↔ (b^2 < 60) ∧ (b ≤ 7) := by
  sorry

end NUMINAMATH_GPT_greatest_integer_b_not_in_range_of_quadratic_l1690_169072


namespace NUMINAMATH_GPT_complex_number_z_l1690_169086

theorem complex_number_z (z : ℂ) (i : ℂ) (hz : i^2 = -1) (h : (1 - i)^2 / z = 1 + i) : z = -1 - i :=
by
  sorry

end NUMINAMATH_GPT_complex_number_z_l1690_169086


namespace NUMINAMATH_GPT_index_card_area_l1690_169045

theorem index_card_area (a b : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : (a - 2) * b = 21) : (a * (b - 1)) = 30 := by
  sorry

end NUMINAMATH_GPT_index_card_area_l1690_169045


namespace NUMINAMATH_GPT_length_of_CD_l1690_169008

theorem length_of_CD (x y u v : ℝ) (R S C D : ℝ → ℝ)
  (h1 : 5 * x = 3 * y)
  (h2 : 7 * u = 4 * v)
  (h3 : u = x + 3)
  (h4 : v = y - 3)
  (h5 : C x + D y = 1) : 
  x + y = 264 :=
by
  sorry

end NUMINAMATH_GPT_length_of_CD_l1690_169008


namespace NUMINAMATH_GPT_exists_x_l1690_169085

theorem exists_x (a b c : ℕ) (ha : 0 < a) (hc : 0 < c) :
  ∃ x : ℕ, (0 < x) ∧ (a ^ x + x) % c = b % c :=
sorry

end NUMINAMATH_GPT_exists_x_l1690_169085


namespace NUMINAMATH_GPT_min_value_of_function_l1690_169015

noncomputable def y (x : ℝ) : ℝ := (Real.cos x) * (Real.sin (2 * x))

theorem min_value_of_function :
  ∃ x ∈ Set.Icc (-Real.pi) Real.pi, y x = -4 * Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_GPT_min_value_of_function_l1690_169015


namespace NUMINAMATH_GPT_roger_cookie_price_l1690_169032

open Classical

theorem roger_cookie_price
  (art_base1 art_base2 art_height : ℕ) 
  (art_cookies_per_batch art_cookie_price roger_cookies_per_batch : ℕ)
  (art_area : ℕ := (art_base1 + art_base2) * art_height / 2)
  (total_dough : ℕ := art_cookies_per_batch * art_area)
  (roger_area : ℚ := total_dough / roger_cookies_per_batch)
  (art_total_earnings : ℚ := art_cookies_per_batch * art_cookie_price) :
  ∀ (roger_cookie_price : ℚ), roger_cookies_per_batch * roger_cookie_price = art_total_earnings →
  roger_cookie_price = 100 / 3 :=
sorry

end NUMINAMATH_GPT_roger_cookie_price_l1690_169032


namespace NUMINAMATH_GPT_reflection_across_x_axis_l1690_169055

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflection_across_x_axis :
  reflect_x_axis (-2, -3) = (-2, 3) :=
by
  sorry

end NUMINAMATH_GPT_reflection_across_x_axis_l1690_169055


namespace NUMINAMATH_GPT_positive_difference_sums_l1690_169084

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end NUMINAMATH_GPT_positive_difference_sums_l1690_169084


namespace NUMINAMATH_GPT_range_of_expression_l1690_169094

theorem range_of_expression (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) : 
  ∃ (z : Set ℝ), z = Set.Icc (2 / 3) 4 ∧ (4*x^2 + 4*y^2 + (1 - x - y)^2) ∈ z :=
by
  sorry

end NUMINAMATH_GPT_range_of_expression_l1690_169094


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1690_169036

theorem solution_set_of_inequality : 
  {x : ℝ | x < x^2} = {x | x < 0} ∪ {x | x > 1} :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1690_169036


namespace NUMINAMATH_GPT_tangent_circles_pass_through_homothety_center_l1690_169053

-- Define the necessary structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def is_tangent_to_line (ω : Circle) (L : ℝ → ℝ) : Prop :=
  sorry -- Definition of tangency to a line

def is_tangent_to_circle (ω : Circle) (C : Circle) : Prop :=
  sorry -- Definition of tangency to another circle

theorem tangent_circles_pass_through_homothety_center
  (L : ℝ → ℝ) (C : Circle) (ω : Circle)
  (H_ext H_int : ℝ × ℝ)
  (H_tangency_line : is_tangent_to_line ω L)
  (H_tangency_circle : is_tangent_to_circle ω C) :
  ∃ P Q : ℝ × ℝ, 
    (is_tangent_to_line ω L ∧ is_tangent_to_circle ω C) →
    (P = Q ∧ (P = H_ext ∨ P = H_int)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_circles_pass_through_homothety_center_l1690_169053


namespace NUMINAMATH_GPT_vet_fees_cat_result_l1690_169033

-- Given conditions
def vet_fees_dog : ℕ := 15
def families_dogs : ℕ := 8
def families_cats : ℕ := 3
def vet_donation : ℕ := 53

-- Mathematical equivalency in Lean
noncomputable def vet_fees_cat (C : ℕ) : Prop :=
  (1 / 3 : ℚ) * (families_dogs * vet_fees_dog + families_cats * C) = vet_donation

-- Prove the vet fees for cats are 13 using above conditions
theorem vet_fees_cat_result : ∃ (C : ℕ), vet_fees_cat C ∧ C = 13 :=
by {
  use 13,
  sorry
}

end NUMINAMATH_GPT_vet_fees_cat_result_l1690_169033
