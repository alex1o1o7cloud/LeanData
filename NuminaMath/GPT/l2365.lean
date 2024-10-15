import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2365_236527

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Define the conditions
def a_5 := a 5
def a_6 := a 6
def a_7 := a 7

axiom cond1 : a_5 = 11
axiom cond2 : a_6 = 17
axiom cond3 : a_7 = 23

noncomputable def sum_first_four_terms : ℤ :=
  a 1 + a 2 + a 3 + a 4

theorem arithmetic_sequence_sum :
  a_5 = 11 → a_6 = 17 → a_7 = 23 → sum_first_four_terms a = -16 :=
by
  intros h5 h6 h7
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2365_236527


namespace NUMINAMATH_GPT_escalator_steps_l2365_236521

theorem escalator_steps
  (steps_ascending : ℤ)
  (steps_descending : ℤ)
  (ascend_units_time : ℤ)
  (descend_units_time : ℤ)
  (speed_ratio : ℤ)
  (equation : ((steps_ascending : ℚ) / (1 + (ascend_units_time : ℚ))) = ((steps_descending : ℚ) / ((descend_units_time : ℚ) * speed_ratio)) )
  (solution_x : (125 * 0.6 = 75)) : 
  (steps_ascending * (1 + 0.6 : ℚ) = 120) :=
by
  sorry

end NUMINAMATH_GPT_escalator_steps_l2365_236521


namespace NUMINAMATH_GPT_performance_stability_l2365_236533

theorem performance_stability (avg_score : ℝ) (num_shots : ℕ) (S_A S_B : ℝ) 
  (h_avg : num_shots = 10)
  (h_same_avg : avg_score = avg_score) 
  (h_SA : S_A^2 = 0.4) 
  (h_SB : S_B^2 = 2) : 
  (S_A < S_B) :=
by
  sorry

end NUMINAMATH_GPT_performance_stability_l2365_236533


namespace NUMINAMATH_GPT_factor_difference_of_squares_l2365_236539

theorem factor_difference_of_squares (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := 
by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l2365_236539


namespace NUMINAMATH_GPT_loss_of_50_denoted_as_minus_50_l2365_236564

def is_profit (x : Int) : Prop :=
  x > 0

def is_loss (x : Int) : Prop :=
  x < 0

theorem loss_of_50_denoted_as_minus_50 : is_loss (-50) :=
  by
    -- proof steps would go here
    sorry

end NUMINAMATH_GPT_loss_of_50_denoted_as_minus_50_l2365_236564


namespace NUMINAMATH_GPT_digit_x_for_divisibility_by_29_l2365_236551

-- Define the base 7 number 34x1_7 in decimal form
def base7_to_decimal (x : ℕ) : ℕ := 3 * 7^3 + 4 * 7^2 + x * 7 + 1

-- State the proof problem
theorem digit_x_for_divisibility_by_29 (x : ℕ) (h : base7_to_decimal x % 29 = 0) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_digit_x_for_divisibility_by_29_l2365_236551


namespace NUMINAMATH_GPT_midpoint_coordinates_l2365_236587

theorem midpoint_coordinates (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 2) (hy1 : y1 = 10) (hx2 : x2 = 6) (hy2 : y2 = 2) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  mx = 4 ∧ my = 6 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_coordinates_l2365_236587


namespace NUMINAMATH_GPT_number_of_merchants_l2365_236508

theorem number_of_merchants (x : ℕ) (h : 2 * x^3 = 2662) : x = 11 :=
  sorry

end NUMINAMATH_GPT_number_of_merchants_l2365_236508


namespace NUMINAMATH_GPT_least_integer_with_remainders_l2365_236520

theorem least_integer_with_remainders :
  ∃ M : ℕ, 
    M % 6 = 5 ∧
    M % 7 = 6 ∧
    M % 9 = 8 ∧
    M % 10 = 9 ∧
    M % 11 = 10 ∧
    M = 6929 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_with_remainders_l2365_236520


namespace NUMINAMATH_GPT_max_difference_is_62_l2365_236588

open Real

noncomputable def max_difference_of_integers : ℝ :=
  let a (k : ℝ) := 2 * k + 1 + sqrt (8 * k)
  let b (k : ℝ) := 2 * k + 1 - sqrt (8 * k)
  let diff (k : ℝ) := a k - b k
  let max_k := 120 -- Maximum integer value k such that 2k + 1 + sqrt(8k) < 1000
  diff max_k

theorem max_difference_is_62 :
  max_difference_of_integers = 62 :=
sorry

end NUMINAMATH_GPT_max_difference_is_62_l2365_236588


namespace NUMINAMATH_GPT_power_function_inequality_l2365_236538

theorem power_function_inequality (m : ℕ) (h : m > 0)
  (h_point : (2 : ℝ) ^ (1 / (m ^ 2 + m)) = Real.sqrt 2) :
  m = 1 ∧ ∀ a : ℝ, 1 ≤ a ∧ a < (3 / 2) → 
  (2 - a : ℝ) ^ (1 / (m ^ 2 + m)) > (a - 1 : ℝ) ^ (1 / (m ^ 2 + m)) :=
by
  sorry

end NUMINAMATH_GPT_power_function_inequality_l2365_236538


namespace NUMINAMATH_GPT_discount_amount_l2365_236505

/-- Suppose Maria received a 25% discount on DVDs, and she paid $120.
    The discount she received is $40. -/
theorem discount_amount (P : ℝ) (h : 0.75 * P = 120) : P - 120 = 40 := 
sorry

end NUMINAMATH_GPT_discount_amount_l2365_236505


namespace NUMINAMATH_GPT_marbles_given_to_joan_l2365_236543

def mary_original_marbles : ℝ := 9.0
def mary_marbles_left : ℝ := 6.0

theorem marbles_given_to_joan :
  mary_original_marbles - mary_marbles_left = 3 := 
by
  sorry

end NUMINAMATH_GPT_marbles_given_to_joan_l2365_236543


namespace NUMINAMATH_GPT_books_finished_l2365_236581

theorem books_finished (miles_traveled : ℕ) (miles_per_book : ℕ) (h_travel : miles_traveled = 6760) (h_rate : miles_per_book = 450) : (miles_traveled / miles_per_book) = 15 :=
by {
  -- Proof will be inserted here
  sorry
}

end NUMINAMATH_GPT_books_finished_l2365_236581


namespace NUMINAMATH_GPT_range_of_a_l2365_236582

variables (m a x y : ℝ)

def p (m a : ℝ) : Prop := m^2 + 12 * a^2 < 7 * a * m ∧ a > 0

def ellipse (m x y : ℝ) : Prop := (x^2)/(m-1) + (y^2)/(2-m) = 1

def q (m : ℝ) (x y : ℝ) : Prop := ellipse m x y ∧ 1 < m ∧ m < 3/2

theorem range_of_a :
  (∃ m, p m a → (∀ x y, q m x y)) → (1/3 ≤ a ∧ a ≤ 3/8) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2365_236582


namespace NUMINAMATH_GPT_max_points_of_intersection_l2365_236569

-- Definitions from the conditions
def circles := 2
def lines := 3

-- Define the problem of the greatest intersection number
theorem max_points_of_intersection (c : ℕ) (l : ℕ) (h_c : c = circles) (h_l : l = lines) : 
  (2 + (l * 2 * c) + (l * (l - 1) / 2)) = 17 :=
by
  rw [h_c, h_l]
  -- We have 2 points from circle intersections
  -- 12 points from lines intersections with circles
  -- 3 points from lines intersections with lines
  -- Hence, 2 + 12 + 3 = 17
  exact Eq.refl 17

end NUMINAMATH_GPT_max_points_of_intersection_l2365_236569


namespace NUMINAMATH_GPT_jenna_eel_length_l2365_236568

theorem jenna_eel_length (J B L : ℝ)
  (h1 : J = (2 / 5) * B)
  (h2 : J = (3 / 7) * L)
  (h3 : J + B + L = 124) : 
  J = 21 := 
sorry

end NUMINAMATH_GPT_jenna_eel_length_l2365_236568


namespace NUMINAMATH_GPT_exists_pair_distinct_integers_l2365_236509

theorem exists_pair_distinct_integers :
  ∃ (a b : ℤ), a ≠ b ∧ (a / 2015 + b / 2016 = (2015 + 2016) / (2015 * 2016)) :=
by
  -- Constructing the proof or using sorry to skip it if not needed here
  sorry

end NUMINAMATH_GPT_exists_pair_distinct_integers_l2365_236509


namespace NUMINAMATH_GPT_chair_cost_l2365_236566

theorem chair_cost (T P n : ℕ) (hT : T = 135) (hP : P = 55) (hn : n = 4) : 
  (T - P) / n = 20 := by
  sorry

end NUMINAMATH_GPT_chair_cost_l2365_236566


namespace NUMINAMATH_GPT_bin_sum_sub_eq_l2365_236554

-- Define binary numbers
def b1 := 0b101110  -- binary 101110_2
def b2 := 0b10101   -- binary 10101_2
def b3 := 0b111000  -- binary 111000_2
def b4 := 0b110101  -- binary 110101_2
def b5 := 0b11101   -- binary 11101_2

-- Define the theorem
theorem bin_sum_sub_eq : ((b1 + b2) - (b3 - b4) + b5) = 0b1011101 := by
  sorry

end NUMINAMATH_GPT_bin_sum_sub_eq_l2365_236554


namespace NUMINAMATH_GPT_simplify_rationalize_denominator_l2365_236552

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_GPT_simplify_rationalize_denominator_l2365_236552


namespace NUMINAMATH_GPT_circle_parabola_intersection_l2365_236502

theorem circle_parabola_intersection (b : ℝ) :
  (∃ c r, ∀ x y : ℝ, y = (5 / 12) * x^2 → ((x - c)^2 + (y - b)^2 = r^2) ∧ 
   (y = (5 / 12) * x + b → ((x - c)^2 + (y - b)^2 = r^2))) → b = 169 / 60 :=
by
  sorry

end NUMINAMATH_GPT_circle_parabola_intersection_l2365_236502


namespace NUMINAMATH_GPT_volume_of_rectangular_box_l2365_236518

theorem volume_of_rectangular_box 
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12) : 
  l * w * h = 60 :=
sorry

end NUMINAMATH_GPT_volume_of_rectangular_box_l2365_236518


namespace NUMINAMATH_GPT_missing_number_geometric_sequence_l2365_236590

theorem missing_number_geometric_sequence : 
  ∃ (x : ℤ), (x = 162) ∧ 
  (x = 54 * 3 ∧ 
  486 = x * 3 ∧ 
  ∀ a b : ℤ, (b = 2 * 3) ∧ 
              (a = 2 * 3) ∧ 
              (18 = b * 3) ∧ 
              (54 = 18 * 3) ∧ 
              (54 * 3 = x)) := 
by sorry

end NUMINAMATH_GPT_missing_number_geometric_sequence_l2365_236590


namespace NUMINAMATH_GPT_sequence_general_term_l2365_236596

noncomputable def a (n : ℕ) : ℝ :=
if n = 1 then 1 else (n : ℝ) / (2 ^ (n - 1))

theorem sequence_general_term (n : ℕ) (hn : n ≠ 0) : 
  a n = if n = 1 then 1 else (n : ℝ) / (2 ^ (n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l2365_236596


namespace NUMINAMATH_GPT_find_sum_of_m_and_k_l2365_236579

theorem find_sum_of_m_and_k
  (d m k : ℤ)
  (h : (9 * d^2 - 5 * d + m) * (4 * d^2 + k * d - 6) = 36 * d^4 + 11 * d^3 - 59 * d^2 + 10 * d + 12) :
  m + k = -7 :=
by sorry

end NUMINAMATH_GPT_find_sum_of_m_and_k_l2365_236579


namespace NUMINAMATH_GPT_total_cost_for_doughnuts_l2365_236526

theorem total_cost_for_doughnuts
  (num_students : ℕ)
  (num_chocolate : ℕ)
  (num_glazed : ℕ)
  (price_chocolate : ℕ)
  (price_glazed : ℕ)
  (H1 : num_students = 25)
  (H2 : num_chocolate = 10)
  (H3 : num_glazed = 15)
  (H4 : price_chocolate = 2)
  (H5 : price_glazed = 1) :
  num_chocolate * price_chocolate + num_glazed * price_glazed = 35 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_total_cost_for_doughnuts_l2365_236526


namespace NUMINAMATH_GPT_sum_of_four_terms_l2365_236519

theorem sum_of_four_terms (a d : ℕ) (h1 : a + d > a) (h2 : a + 2 * d > a + d)
  (h3 : (a + 2 * d) * (a + 2 * d) = (a + d) * (a + 3 * d)) (h4 : (a + 3 * d) - a = 30) :
  a + (a + d) + (a + 2 * d) + (a + 3 * d) = 129 :=
sorry

end NUMINAMATH_GPT_sum_of_four_terms_l2365_236519


namespace NUMINAMATH_GPT_blue_paint_gallons_l2365_236589

-- Define the total gallons of paint used
def total_paint_gallons : ℕ := 6689

-- Define the gallons of white paint used
def white_paint_gallons : ℕ := 660

-- Define the corresponding proof problem
theorem blue_paint_gallons : 
  ∀ total white blue : ℕ, total = 6689 → white = 660 → blue = total - white → blue = 6029 := by
  sorry

end NUMINAMATH_GPT_blue_paint_gallons_l2365_236589


namespace NUMINAMATH_GPT_algebraic_expression_simplification_l2365_236578

theorem algebraic_expression_simplification (k x : ℝ) (h : (x - k * x) * (2 * x - k * x) - 3 * x * (2 * x - k * x) = 5 * x^2) :
  k = 3 ∨ k = -3 :=
by {
  sorry
}

end NUMINAMATH_GPT_algebraic_expression_simplification_l2365_236578


namespace NUMINAMATH_GPT_exactly_one_equals_xx_plus_xx_l2365_236501

theorem exactly_one_equals_xx_plus_xx (x : ℝ) (hx : x > 0) :
  let expr1 := 2 * x^x
  let expr2 := x^(2*x)
  let expr3 := (2*x)^x
  let expr4 := (2*x)^(2*x)
  (expr1 = x^x + x^x) ∧ (¬(expr2 = x^x + x^x)) ∧ (¬(expr3 = x^x + x^x)) ∧ (¬(expr4 = x^x + x^x)) := 
by
  sorry

end NUMINAMATH_GPT_exactly_one_equals_xx_plus_xx_l2365_236501


namespace NUMINAMATH_GPT_additional_lollipops_needed_l2365_236583

theorem additional_lollipops_needed
  (kids : ℕ) (initial_lollipops : ℕ) (min_lollipops : ℕ) (max_lollipops : ℕ)
  (total_kid_with_lollipops : ∀ k, ∃ n, min_lollipops ≤ n ∧ n ≤ max_lollipops ∧ k = n ∨ k = n + 1 )
  (divisible_by_kids : (min_lollipops + max_lollipops) % kids = 0)
  (min_lollipops_eq : min_lollipops = 42)
  (kids_eq : kids = 42)
  (initial_lollipops_eq : initial_lollipops = 650)
  : ∃ additional_lollipops, (n : ℕ) = 42 → additional_lollipops = 1975 := 
by sorry

end NUMINAMATH_GPT_additional_lollipops_needed_l2365_236583


namespace NUMINAMATH_GPT_intersection_S_T_eq_T_l2365_236558

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end NUMINAMATH_GPT_intersection_S_T_eq_T_l2365_236558


namespace NUMINAMATH_GPT_line_through_points_l2365_236541

theorem line_through_points (m b: ℝ) 
  (h1: ∃ m, ∀ x y : ℝ, ((x, y) = (1, 3) ∨ (x, y) = (3, 7)) → y = m * x + b) 
  (h2: ∀ x y : ℝ, ((x, y) = (1, 3) ∨ (x, y) = (3, 7)) → y = m * x + b):
  m + b = 3 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_line_through_points_l2365_236541


namespace NUMINAMATH_GPT_find_a_minus_b_l2365_236549

theorem find_a_minus_b (a b : ℚ)
  (h1 : 2 = a + b / 2)
  (h2 : 7 = a - b / 2)
  : a - b = 19 / 2 := 
  sorry

end NUMINAMATH_GPT_find_a_minus_b_l2365_236549


namespace NUMINAMATH_GPT_minimize_segment_sum_l2365_236537

theorem minimize_segment_sum (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ x y : ℝ, x = Real.sqrt (a * b) ∧ y = Real.sqrt (a * b) ∧ x * y = a * b ∧ x + y = 2 * Real.sqrt (a * b) := 
by
  sorry

end NUMINAMATH_GPT_minimize_segment_sum_l2365_236537


namespace NUMINAMATH_GPT_quadrilateral_identity_l2365_236504

theorem quadrilateral_identity 
  {A B C D : Type*} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) (AC : ℝ) (BD : ℝ)
  (angle_A : ℝ) (angle_C : ℝ) 
  (h_angle_sum : angle_A + angle_C = 120)
  : (AC * BD)^2 = (AB * CD)^2 + (BC * AD)^2 + AB * BC * CD * DA := 
by {
  sorry
}

end NUMINAMATH_GPT_quadrilateral_identity_l2365_236504


namespace NUMINAMATH_GPT_solve_for_a_plus_b_l2365_236565

theorem solve_for_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, (-1 < x ∧ x < 1 / 3) → ax^2 + bx + 1 > 0) →
  a * (-3) + b = -5 :=
by
  intro h
  -- Here we can use the proofs provided in the solution steps.
  sorry

end NUMINAMATH_GPT_solve_for_a_plus_b_l2365_236565


namespace NUMINAMATH_GPT_curve_cartesian_eq_correct_intersection_distances_sum_l2365_236530

noncomputable section

def curve_parametric_eqns (θ : ℝ) : ℝ × ℝ := 
  (1 + 3 * Real.cos θ, 3 + 3 * Real.sin θ)

def line_parametric_eqns (t : ℝ) : ℝ × ℝ := 
  (3 + (1/2) * t, 3 + (Real.sqrt 3 / 2) * t)

def curve_cartesian_eq (x y : ℝ) : Prop := 
  (x - 1)^2 + (y - 3)^2 = 9

def point_p : ℝ × ℝ := 
  (3, 3)

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem curve_cartesian_eq_correct (θ : ℝ) : 
  curve_cartesian_eq (curve_parametric_eqns θ).1 (curve_parametric_eqns θ).2 := 
by 
  sorry

theorem intersection_distances_sum (t1 t2 : ℝ) 
  (h1 : curve_cartesian_eq (line_parametric_eqns t1).1 (line_parametric_eqns t1).2) 
  (h2 : curve_cartesian_eq (line_parametric_eqns t2).1 (line_parametric_eqns t2).2) : 
  distance point_p (line_parametric_eqns t1) + distance point_p (line_parametric_eqns t2) = 2 * Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_curve_cartesian_eq_correct_intersection_distances_sum_l2365_236530


namespace NUMINAMATH_GPT_scientific_notation_316000000_l2365_236511

theorem scientific_notation_316000000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 316000000 = a * 10 ^ n ∧ a = 3.16 ∧ n = 8 :=
by
  -- Proof would be here
  sorry

end NUMINAMATH_GPT_scientific_notation_316000000_l2365_236511


namespace NUMINAMATH_GPT_highest_power_of_2_divides_l2365_236542

def a : ℕ := 17
def b : ℕ := 15
def n : ℕ := a^5 - b^5

def highestPowerOf2Divides (k : ℕ) : ℕ :=
  -- Function to find the highest power of 2 that divides k, implementation is omitted
  sorry

theorem highest_power_of_2_divides :
  highestPowerOf2Divides n = 2^5 := by
    sorry

end NUMINAMATH_GPT_highest_power_of_2_divides_l2365_236542


namespace NUMINAMATH_GPT_simplify_expression_l2365_236593

theorem simplify_expression (y : ℝ) (hy : y ≠ 0) : 
  (2 / y^2 - y⁻¹) = (2 - y) / y^2 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2365_236593


namespace NUMINAMATH_GPT_wrapping_paper_area_l2365_236577

variable {l w h : ℝ}

theorem wrapping_paper_area (hl : 0 < l) (hw : 0 < w) (hh : 0 < h) :
  (4 * l * h + 2 * l * h + 2 * w * h) = 6 * l * h + 2 * w * h :=
  sorry

end NUMINAMATH_GPT_wrapping_paper_area_l2365_236577


namespace NUMINAMATH_GPT_movie_attendance_l2365_236516

theorem movie_attendance (total_seats : ℕ) (empty_seats : ℕ) (h1 : total_seats = 750) (h2 : empty_seats = 218) :
  total_seats - empty_seats = 532 := by
  sorry

end NUMINAMATH_GPT_movie_attendance_l2365_236516


namespace NUMINAMATH_GPT_correct_pair_has_integer_distance_l2365_236507

-- Define the pairs of (x, y)
def pairs : List (ℕ × ℕ) :=
  [(88209, 90288), (82098, 89028), (28098, 89082), (90882, 28809)]

-- Define the property: a pair (x, y) has the distance √(x^2 + y^2) as an integer
def is_integer_distance_pair (x y : ℕ) : Prop :=
  ∃ (n : ℕ), n * n = x * x + y * y

-- Translate the problem to the proof: Prove (88209, 90288) satisfies the given property
theorem correct_pair_has_integer_distance :
  is_integer_distance_pair 88209 90288 :=
by
  sorry

end NUMINAMATH_GPT_correct_pair_has_integer_distance_l2365_236507


namespace NUMINAMATH_GPT_inverse_variation_l2365_236557

theorem inverse_variation (a : ℕ) (b : ℝ) (h : a * b = 400) (h₀ : a = 3200) : b = 0.125 :=
by sorry

end NUMINAMATH_GPT_inverse_variation_l2365_236557


namespace NUMINAMATH_GPT_problem_1_l2365_236592

theorem problem_1 (f : ℝ → ℝ) (hf_mul : ∀ x y : ℝ, f (x * y) = f x + f y) (hf_4 : f 4 = 2) : f (Real.sqrt 2) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_problem_1_l2365_236592


namespace NUMINAMATH_GPT_find_k_l2365_236560

theorem find_k (k : ℝ) : 
  (∃ c1 c2 : ℝ, (2 * c1^2 + 5 * c1 = k) ∧ 
                (2 * c2^2 + 5 * c2 = k) ∧ 
                (c1 > c2) ∧ 
                (c1 - c2 = 5.5)) → 
  k = 12 := 
by
  intros h
  obtain ⟨c1, c2, h1, h2, h3, h4⟩ := h
  sorry

end NUMINAMATH_GPT_find_k_l2365_236560


namespace NUMINAMATH_GPT_sequence_inequality_l2365_236563

theorem sequence_inequality (a : ℕ → ℕ) (strictly_increasing : ∀ n, a n < a (n + 1))
  (sum_condition : ∀ m : ℕ, ∃ i j : ℕ, m = a i + a j) :
  ∀ n, a n ≤ n^2 :=
by sorry

end NUMINAMATH_GPT_sequence_inequality_l2365_236563


namespace NUMINAMATH_GPT_simplify_expression_l2365_236571

theorem simplify_expression (x : ℕ) (h : x = 100) :
  (x + 1) * (x - 1) + x * (2 - x) + (x - 1) ^ 2 = 10000 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2365_236571


namespace NUMINAMATH_GPT_no_geometric_progression_l2365_236591

theorem no_geometric_progression (r s t : ℕ) (h1 : r < s) (h2 : s < t) :
  ¬ ∃ (b : ℂ), (3^r - 2^r) * b^(s - r) = 3^s - 2^s ∧ (3^s - 2^s) * b^(t - s) = 3^t - 2^t := by
  sorry

end NUMINAMATH_GPT_no_geometric_progression_l2365_236591


namespace NUMINAMATH_GPT_absent_children_count_l2365_236561

theorem absent_children_count (total_children : ℕ) (bananas_per_child : ℕ) (extra_bananas_per_child : ℕ)
    (absent_children : ℕ) (total_bananas : ℕ) (present_children : ℕ) :
    total_children = 640 →
    bananas_per_child = 2 →
    extra_bananas_per_child = 2 →
    total_bananas = (total_children * bananas_per_child) →
    present_children = (total_children - absent_children) →
    total_bananas = (present_children * (bananas_per_child + extra_bananas_per_child)) →
    absent_children = 320 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_absent_children_count_l2365_236561


namespace NUMINAMATH_GPT_max_sum_of_diagonals_l2365_236553

theorem max_sum_of_diagonals (a b : ℝ) (h_side : a^2 + b^2 = 25) (h_bounds1 : 2 * a ≤ 6) (h_bounds2 : 2 * b ≥ 6) : 2 * (a + b) = 14 :=
sorry

end NUMINAMATH_GPT_max_sum_of_diagonals_l2365_236553


namespace NUMINAMATH_GPT_ice_cubes_per_cup_l2365_236528

theorem ice_cubes_per_cup (total_ice_cubes number_of_cups : ℕ) (h1 : total_ice_cubes = 30) (h2 : number_of_cups = 6) : 
  total_ice_cubes / number_of_cups = 5 := 
by
  sorry

end NUMINAMATH_GPT_ice_cubes_per_cup_l2365_236528


namespace NUMINAMATH_GPT_cut_out_square_possible_l2365_236513

/-- 
Formalization of cutting out eight \(2 \times 1\) rectangles from an \(8 \times 8\) 
checkered board, and checking if it is always possible to cut out a \(2 \times 2\) square
from the remaining part of the board.
-/
theorem cut_out_square_possible :
  ∀ (board : ℕ) (rectangles : ℕ), (board = 64) ∧ (rectangles = 8) → (4 ∣ board) →
  ∃ (remaining_squares : ℕ), (remaining_squares = 48) ∧ 
  (∃ (square_size : ℕ), (square_size = 4) ∧ (remaining_squares ≥ square_size)) :=
by {
  sorry
}

end NUMINAMATH_GPT_cut_out_square_possible_l2365_236513


namespace NUMINAMATH_GPT_fraction_of_new_releases_l2365_236532

theorem fraction_of_new_releases (total_books : ℕ) (historical_fiction_percent : ℝ) (historical_new_releases_percent : ℝ) (other_new_releases_percent : ℝ)
  (h1 : total_books = 100)
  (h2 : historical_fiction_percent = 0.4)
  (h3 : historical_new_releases_percent = 0.4)
  (h4 : other_new_releases_percent = 0.2) :
  (historical_fiction_percent * historical_new_releases_percent * total_books) / 
  ((historical_fiction_percent * historical_new_releases_percent * total_books) + ((1 - historical_fiction_percent) * other_new_releases_percent * total_books)) = 4 / 7 :=
by
  have h_books : total_books = 100 := h1
  have h_fiction : historical_fiction_percent = 0.4 := h2
  have h_new_releases : historical_new_releases_percent = 0.4 := h3
  have h_other_new_releases : other_new_releases_percent = 0.2 := h4
  sorry

end NUMINAMATH_GPT_fraction_of_new_releases_l2365_236532


namespace NUMINAMATH_GPT_angle_C_in_triangle_l2365_236546

theorem angle_C_in_triangle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A + B = 115) : C = 65 := 
by 
  sorry

end NUMINAMATH_GPT_angle_C_in_triangle_l2365_236546


namespace NUMINAMATH_GPT_percentage_gain_second_week_l2365_236545

variables (initial_investment final_value after_first_week_value gain_percentage first_week_gain second_week_gain second_week_gain_percentage : ℝ)

def pima_investment (initial_investment: ℝ) (first_week_gain_percentage: ℝ) : ℝ :=
  initial_investment * (1 + first_week_gain_percentage)

def second_week_investment (initial_investment first_week_gain_percentage second_week_gain_percentage : ℝ) : ℝ :=
  initial_investment * (1 + first_week_gain_percentage) * (1 + second_week_gain_percentage)

theorem percentage_gain_second_week
  (initial_investment : ℝ)
  (first_week_gain_percentage : ℝ)
  (final_value : ℝ)
  (h1: initial_investment = 400)
  (h2: first_week_gain_percentage = 0.25)
  (h3: final_value = 750) :
  second_week_gain_percentage = 0.5 :=
by
  let after_first_week_value := pima_investment initial_investment first_week_gain_percentage
  let second_week_gain := final_value - after_first_week_value
  let second_week_gain_percentage := second_week_gain / after_first_week_value * 100
  sorry

end NUMINAMATH_GPT_percentage_gain_second_week_l2365_236545


namespace NUMINAMATH_GPT_intersect_complement_A_and_B_l2365_236515

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x + 1 < 0}
def B : Set ℝ := {x | x - 3 < 0}

theorem intersect_complement_A_and_B : (Set.compl A ∩ B) = {x | -1 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_GPT_intersect_complement_A_and_B_l2365_236515


namespace NUMINAMATH_GPT_largest_k_exists_l2365_236544

theorem largest_k_exists (n : ℕ) (h : n ≥ 4) : 
  ∃ k : ℕ, (∀ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n → (c - b) ≥ k ∧ (b - a) ≥ k ∧ (a + b ≥ c + 1)) ∧ 
  (k = (n - 1) / 3) :=
  sorry

end NUMINAMATH_GPT_largest_k_exists_l2365_236544


namespace NUMINAMATH_GPT_petyas_square_is_larger_l2365_236522

noncomputable def side_petya_square (a b : ℝ) : ℝ :=
  a * b / (a + b)

noncomputable def side_vasya_square (a b : ℝ) : ℝ :=
  a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petyas_square_is_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  side_petya_square a b > side_vasya_square a b := by
  sorry

end NUMINAMATH_GPT_petyas_square_is_larger_l2365_236522


namespace NUMINAMATH_GPT_ryan_chinese_learning_hours_l2365_236503

theorem ryan_chinese_learning_hours
    (hours_per_day : ℕ) 
    (days : ℕ) 
    (h1 : hours_per_day = 4) 
    (h2 : days = 6) : 
    hours_per_day * days = 24 := 
by 
    sorry

end NUMINAMATH_GPT_ryan_chinese_learning_hours_l2365_236503


namespace NUMINAMATH_GPT_marys_garbage_bill_is_correct_l2365_236536

noncomputable def calculate_garbage_bill :=
  let weekly_trash_bin_cost := 2 * 10
  let weekly_recycling_bin_cost := 1 * 5
  let weekly_green_waste_bin_cost := 1 * 3
  let total_weekly_cost := weekly_trash_bin_cost + weekly_recycling_bin_cost + weekly_green_waste_bin_cost
  let monthly_bin_cost := total_weekly_cost * 4
  let base_monthly_cost := monthly_bin_cost + 15
  let discount := base_monthly_cost * 0.18
  let discounted_cost := base_monthly_cost - discount
  let fines := 20 + 10
  discounted_cost + fines

theorem marys_garbage_bill_is_correct :
  calculate_garbage_bill = 134.14 := 
  by {
  sorry
  }

end NUMINAMATH_GPT_marys_garbage_bill_is_correct_l2365_236536


namespace NUMINAMATH_GPT_number_of_rel_prime_to_21_in_range_l2365_236559

def is_rel_prime (a b : ℕ) : Prop := gcd a b = 1

noncomputable def count_rel_prime_in_range (a b g : ℕ) : ℕ :=
  ((b - a + 1) : ℕ) - ((b / 3 - (a - 1) / 3) + (b / 7 - (a - 1) / 7) - (b / 21 - (a - 1) / 21))

theorem number_of_rel_prime_to_21_in_range :
  count_rel_prime_in_range 11 99 21 = 51 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_rel_prime_to_21_in_range_l2365_236559


namespace NUMINAMATH_GPT_parabola_vertex_below_x_axis_l2365_236512

theorem parabola_vertex_below_x_axis (a : ℝ) : (∀ x : ℝ, (x^2 + 2 * x + a < 0)) → a < 1 := 
by
  intro h
  -- proof step here
  sorry

end NUMINAMATH_GPT_parabola_vertex_below_x_axis_l2365_236512


namespace NUMINAMATH_GPT_added_amount_correct_l2365_236599

theorem added_amount_correct (n x : ℕ) (h1 : n = 20) (h2 : 1/2 * n + x = 15) :
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_added_amount_correct_l2365_236599


namespace NUMINAMATH_GPT_discounted_price_is_correct_l2365_236506

def marked_price : ℕ := 125
def discount_rate : ℚ := 4 / 100

def calculate_discounted_price (marked_price : ℕ) (discount_rate : ℚ) : ℚ :=
  marked_price - (discount_rate * marked_price)

theorem discounted_price_is_correct :
  calculate_discounted_price marked_price discount_rate = 120 := by
  sorry

end NUMINAMATH_GPT_discounted_price_is_correct_l2365_236506


namespace NUMINAMATH_GPT_common_tangents_l2365_236524

noncomputable def circle1 := { p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 4 }
noncomputable def circle2 := { p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 2)^2 = 9 }

theorem common_tangents (h : ∀ p : ℝ × ℝ, p ∈ circle1 → p ∈ circle2) : 
  ∃ tangents : ℕ, tangents = 2 :=
sorry

end NUMINAMATH_GPT_common_tangents_l2365_236524


namespace NUMINAMATH_GPT_edges_after_truncation_l2365_236580

-- Define a regular tetrahedron with 4 vertices and 6 edges
structure Tetrahedron :=
  (vertices : ℕ)
  (edges : ℕ)

-- Initial regular tetrahedron
def initial_tetrahedron : Tetrahedron :=
  { vertices := 4, edges := 6 }

-- Function to calculate the number of edges after truncating vertices
def truncated_edges (t : Tetrahedron) (vertex_truncations : ℕ) (new_edges_per_vertex : ℕ) : ℕ :=
  vertex_truncations * new_edges_per_vertex

-- Given a regular tetrahedron and the truncation process
def resulting_edges (t : Tetrahedron) (vertex_truncations : ℕ) :=
  truncated_edges t vertex_truncations 3

-- Problem statement: Proving the resulting figure has 12 edges
theorem edges_after_truncation :
  resulting_edges initial_tetrahedron 4 = 12 :=
  sorry

end NUMINAMATH_GPT_edges_after_truncation_l2365_236580


namespace NUMINAMATH_GPT_parallel_lines_no_intersection_l2365_236567

theorem parallel_lines_no_intersection (k : ℝ) :
  (∀ t s : ℝ, 
    ∃ (a b : ℝ), (a, b) = (1, -3) + t • (2, 5) ∧ (a, b) = (-4, 2) + s • (3, k)) → 
  k = 15 / 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_no_intersection_l2365_236567


namespace NUMINAMATH_GPT_max_value_f_on_0_4_l2365_236556

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem max_value_f_on_0_4 : ∃ (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) (4 : ℝ)), ∀ (y : ℝ), y ∈ Set.Icc (0 : ℝ) (4 : ℝ) → f y ≤ f x ∧ f x = 1 / Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_max_value_f_on_0_4_l2365_236556


namespace NUMINAMATH_GPT_graph_crosses_x_axis_at_origin_l2365_236529

-- Let g(x) be a quadratic function defined as ax^2 + bx
def g (a b x : ℝ) : ℝ := a * x^2 + b * x

-- Define the conditions a ≠ 0 and b ≠ 0
axiom a_ne_0 (a : ℝ) : a ≠ 0
axiom b_ne_0 (b : ℝ) : b ≠ 0

-- The problem statement
theorem graph_crosses_x_axis_at_origin (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  ∃ x : ℝ, g a b x = 0 ∧ ∀ x', g a b x' = 0 → x' = 0 ∨ x' = -b / a :=
sorry

end NUMINAMATH_GPT_graph_crosses_x_axis_at_origin_l2365_236529


namespace NUMINAMATH_GPT_number_of_friends_l2365_236548

def total_envelopes : ℕ := 37
def envelopes_per_friend : ℕ := 3
def envelopes_left : ℕ := 22

theorem number_of_friends :
  ((total_envelopes - envelopes_left) / envelopes_per_friend) = 5 := by
  sorry

end NUMINAMATH_GPT_number_of_friends_l2365_236548


namespace NUMINAMATH_GPT_no_non_degenerate_triangle_l2365_236534

theorem no_non_degenerate_triangle 
  (a b c : ℕ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ c) 
  (h3 : a ≠ c) 
  (h4 : Nat.gcd a (Nat.gcd b c) = 1) 
  (h5 : a ∣ (b - c) * (b - c)) 
  (h6 : b ∣ (a - c) * (a - c)) 
  (h7 : c ∣ (a - b) * (a - b)) : 
  ¬ (a < b + c ∧ b < a + c ∧ c < a + b) := 
sorry

end NUMINAMATH_GPT_no_non_degenerate_triangle_l2365_236534


namespace NUMINAMATH_GPT_equation_of_line_AB_l2365_236570

theorem equation_of_line_AB 
  (x y : ℝ)
  (passes_through_P : (4 - 1)^2 + (1 - 0)^2 = 1)     
  (circle_eq : (x - 1)^2 + y^2 = 1) :
  3 * x + y - 4 = 0 :=
sorry

end NUMINAMATH_GPT_equation_of_line_AB_l2365_236570


namespace NUMINAMATH_GPT_equation_of_line_bisecting_chord_l2365_236597

theorem equation_of_line_bisecting_chord
  (P : ℝ × ℝ) 
  (A B : ℝ × ℝ)
  (P_bisects_AB : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (P_on_ellipse : 3 * P.1^2 + 4 * P.2^2 = 24)
  (A_on_ellipse : 3 * A.1^2 + 4 * A.2^2 = 24)
  (B_on_ellipse : 3 * B.1^2 + 4 * B.2^2 = 24) :
  ∃ (a b c : ℝ), a * P.2 + b * P.1 + c = 0 ∧ a = 2 ∧ b = -3 ∧ c = 7 :=
by 
  sorry

end NUMINAMATH_GPT_equation_of_line_bisecting_chord_l2365_236597


namespace NUMINAMATH_GPT_porter_l2365_236598

def previous_sale_amount : ℕ := 9000

def recent_sale_price (previous_sale_amount : ℕ) : ℕ :=
  5 * previous_sale_amount - 1000

theorem porter's_recent_sale : recent_sale_price previous_sale_amount = 44000 :=
by
  sorry

end NUMINAMATH_GPT_porter_l2365_236598


namespace NUMINAMATH_GPT_range_of_m_l2365_236572

theorem range_of_m (m : ℝ) (h : (m^2 + m) ^ (3 / 5) ≤ (3 - m) ^ (3 / 5)) : 
  -3 ≤ m ∧ m ≤ 1 :=
by { sorry }

end NUMINAMATH_GPT_range_of_m_l2365_236572


namespace NUMINAMATH_GPT_log_equation_solution_l2365_236500

theorem log_equation_solution (x : ℝ) (h₁ : x > 0) (h₂ : x ≠ 1) (h₃ : x ≠ 1/16) (h₄ : x ≠ 1/2) 
    (h_eq : (Real.log 2 / Real.log (4 * Real.sqrt x)) / (Real.log 2 / Real.log (2 * x)) 
            + (Real.log 2 / Real.log (2 * x)) * (Real.log (2 * x) / Real.log (1 / 2)) = 0) 
    : x = 4 := 
sorry

end NUMINAMATH_GPT_log_equation_solution_l2365_236500


namespace NUMINAMATH_GPT_fraction_eq_zero_implies_x_eq_one_l2365_236523

theorem fraction_eq_zero_implies_x_eq_one (x : ℝ) (h1 : (x - 1) = 0) (h2 : (x - 5) ≠ 0) : x = 1 :=
sorry

end NUMINAMATH_GPT_fraction_eq_zero_implies_x_eq_one_l2365_236523


namespace NUMINAMATH_GPT_parabola_focus_l2365_236574

theorem parabola_focus (a b c : ℝ) (h k : ℝ) (p : ℝ) :
  (a = 4) →
  (b = -4) →
  (c = -3) →
  (h = -b / (2 * a)) →
  (k = a * h ^ 2 + b * h + c) →
  (p = 1 / (4 * a)) →
  (k + p = -4 + 1 / 16) →
  (h, k + p) = (1 / 2, -63 / 16) :=
by
  intros a_eq b_eq c_eq h_eq k_eq p_eq focus_eq
  rw [a_eq, b_eq, c_eq] at *
  sorry

end NUMINAMATH_GPT_parabola_focus_l2365_236574


namespace NUMINAMATH_GPT_crazy_silly_school_movie_count_l2365_236540

theorem crazy_silly_school_movie_count
  (books : ℕ) (read_books : ℕ) (watched_movies : ℕ) (diff_books_movies : ℕ)
  (total_books : books = 8) 
  (read_movie_count : watched_movies = 19)
  (read_book_count : read_books = 16)
  (book_movie_diff : watched_movies = read_books + diff_books_movies)
  (diff_value : diff_books_movies = 3) :
  ∃ M, M ≥ 19 :=
by
  sorry

end NUMINAMATH_GPT_crazy_silly_school_movie_count_l2365_236540


namespace NUMINAMATH_GPT_brian_breath_proof_l2365_236550

def breath_holding_time (initial_time: ℕ) (week1_factor: ℝ) (week2_factor: ℝ) 
  (missed_days: ℕ) (missed_decrease: ℝ) (week3_factor: ℝ): ℝ := by
  let week1_time := initial_time * week1_factor
  let hypothetical_week2_time := week1_time * (1 + week2_factor)
  let missed_decrease_total := week1_time * missed_decrease * missed_days
  let effective_week2_time := hypothetical_week2_time - missed_decrease_total
  let final_time := effective_week2_time * (1 + week3_factor)
  exact final_time

theorem brian_breath_proof :
  breath_holding_time 10 2 0.75 2 0.1 0.5 = 46.5 := 
by
  sorry

end NUMINAMATH_GPT_brian_breath_proof_l2365_236550


namespace NUMINAMATH_GPT_theater_ticket_sales_l2365_236514

theorem theater_ticket_sales 
  (total_tickets : ℕ) (price_adult_ticket : ℕ) (price_senior_ticket : ℕ) (senior_tickets_sold : ℕ) 
  (Total_tickets_condition : total_tickets = 510)
  (Price_adult_ticket_condition : price_adult_ticket = 21)
  (Price_senior_ticket_condition : price_senior_ticket = 15)
  (Senior_tickets_sold_condition : senior_tickets_sold = 327) : 
  (183 * 21 + 327 * 15 = 8748) :=
by
  sorry

end NUMINAMATH_GPT_theater_ticket_sales_l2365_236514


namespace NUMINAMATH_GPT_simplify_expression_l2365_236575

theorem simplify_expression (w : ℝ) :
  3 * w + 4 - 2 * w - 5 + 6 * w + 7 - 3 * w - 9 = 4 * w - 3 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2365_236575


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2365_236531

theorem geometric_sequence_sum (a : ℝ) (q : ℝ) (h1 : a * q^2 + a * q^5 = 6)
  (h2 : a * q^4 + a * q^7 = 9) : a * q^6 + a * q^9 = 27 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2365_236531


namespace NUMINAMATH_GPT_find_N_l2365_236562

theorem find_N (
    A B : ℝ) (N : ℕ) (r : ℝ) (hA : A = N * π * r^2 / 2) 
    (hB : B = (π * r^2 / 2) * (N^2 - N)) 
    (ratio : A / B = 1 / 18) : 
    N = 19 :=
by
  sorry

end NUMINAMATH_GPT_find_N_l2365_236562


namespace NUMINAMATH_GPT_min_distance_eq_sqrt2_l2365_236555

open Real

variables {P Q : ℝ × ℝ}
variables {x y : ℝ}

/-- Given that point P is on the curve y = e^x and point Q is on the curve y = ln x, prove that the minimum value of the distance |PQ| is sqrt(2). -/
theorem min_distance_eq_sqrt2 : 
  (P.2 = exp P.1) ∧ (Q.2 = log Q.1) → (dist P Q) = sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_eq_sqrt2_l2365_236555


namespace NUMINAMATH_GPT_max_difference_l2365_236584

theorem max_difference (U V W X Y Z : ℕ) (hUVW : U ≠ V ∧ V ≠ W ∧ U ≠ W)
    (hXYZ : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) (digits_UVW : 1 ≤ U ∧ U ≤ 9 ∧ 1 ≤ V ∧ V ≤ 9 ∧ 1 ≤ W ∧ W ≤ 9)
    (digits_XYZ : 1 ≤ X ∧ X ≤ 9 ∧ 1 ≤ Y ∧ Y ≤ 9 ∧ 1 ≤ Z ∧ Z ≤ 9) :
    U * 100 + V * 10 + W = 987 → X * 100 + Y * 10 + Z = 123 → (U * 100 + V * 10 + W) - (X * 100 + Y * 10 + Z) = 864 :=
by
  sorry

end NUMINAMATH_GPT_max_difference_l2365_236584


namespace NUMINAMATH_GPT_sum_of_interior_angles_l2365_236594

theorem sum_of_interior_angles (n : ℕ) (h1 : 180 * (n - 2) = 1800) (h2 : n = 12) : 
  180 * ((n + 4) - 2) = 2520 := 
by 
  { sorry }

end NUMINAMATH_GPT_sum_of_interior_angles_l2365_236594


namespace NUMINAMATH_GPT_car_speed_ratio_l2365_236573

noncomputable def speed_ratio (t_round_trip t_leaves t_returns t_walk_start t_walk_end : ℕ) (meet_time : ℕ) : ℕ :=
  let one_way_time_car := t_round_trip / 2
  let total_car_time := t_returns - t_leaves
  let meeting_time_car := total_car_time / 2
  let remaining_time_to_factory := one_way_time_car - meeting_time_car
  let total_walk_time := t_walk_end - t_walk_start
  total_walk_time / remaining_time_to_factory

theorem car_speed_ratio :
  speed_ratio 60 120 160 60 140 80 = 8 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_ratio_l2365_236573


namespace NUMINAMATH_GPT_quad_eq_diagonals_theorem_l2365_236576

noncomputable def quad_eq_diagonals (a b c d m n : ℝ) (A C : ℝ) : Prop :=
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (A + C)

theorem quad_eq_diagonals_theorem (a b c d m n A C : ℝ) :
  quad_eq_diagonals a b c d m n A C :=
by
  sorry

end NUMINAMATH_GPT_quad_eq_diagonals_theorem_l2365_236576


namespace NUMINAMATH_GPT_complex_plane_second_quadrant_l2365_236535

theorem complex_plane_second_quadrant (x : ℝ) :
  (x ^ 2 - 6 * x + 5 < 0 ∧ x - 2 > 0) ↔ (2 < x ∧ x < 5) :=
by
  -- The proof is to be completed.
  sorry

end NUMINAMATH_GPT_complex_plane_second_quadrant_l2365_236535


namespace NUMINAMATH_GPT_base10_to_base8_440_l2365_236547

theorem base10_to_base8_440 :
  ∃ k1 k2 k3,
    k1 = 6 ∧
    k2 = 7 ∧
    k3 = 0 ∧
    (440 = k1 * 64 + k2 * 8 + k3) ∧
    (64 = 8^2) ∧
    (8^3 > 440) :=
sorry

end NUMINAMATH_GPT_base10_to_base8_440_l2365_236547


namespace NUMINAMATH_GPT_determine_a_l2365_236517

theorem determine_a (r s a : ℝ) (h1 : r^2 = a) (h2 : 2 * r * s = 16) (h3 : s^2 = 16) : a = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_determine_a_l2365_236517


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l2365_236586

theorem cyclic_sum_inequality (n : ℕ) (a : Fin n.succ -> ℕ) (h : ∀ i, a i > 0) : 
  (Finset.univ.sum fun i => a i / a ((i + 1) % n)) ≥ n :=
by
  sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l2365_236586


namespace NUMINAMATH_GPT_water_pump_calculation_l2365_236510

-- Define the given initial conditions
variables (f h j g k l m : ℕ)

-- Provide the correctly calculated answer
theorem water_pump_calculation (hf : f > 0) (hg : g > 0) (hk : k > 0) (hm : m > 0) : 
  (k * l * m * j * h) / (10000 * f * g) = (k * (j * h / (f * g)) * l * m) / 10000 := 
sorry

end NUMINAMATH_GPT_water_pump_calculation_l2365_236510


namespace NUMINAMATH_GPT_pqrsum_l2365_236585

-- Given constants and conditions:
variables {p q r : ℝ} -- p, q, r are real numbers
axiom Hpq : p < q -- given condition p < q
axiom Hineq : ∀ x : ℝ, (x > 5 ∨ 7 ≤ x ∧ x ≤ 15) ↔ ( (x - p) * (x - q) / (x - r) ≥ 0) -- given inequality condition

-- Values from the solution:
axiom Hp : p = 7
axiom Hq : q = 15
axiom Hr : r = 5

-- Proof statement:
theorem pqrsum : p + 2 * q + 3 * r = 52 :=
sorry 

end NUMINAMATH_GPT_pqrsum_l2365_236585


namespace NUMINAMATH_GPT_geometric_sequence_fraction_l2365_236525

variable (a_1 : ℝ) (q : ℝ)

theorem geometric_sequence_fraction (h : q = 2) :
  (2 * a_1 + a_1 * q) / (2 * (a_1 * q^2) + a_1 * q^3) = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_fraction_l2365_236525


namespace NUMINAMATH_GPT_paige_folders_l2365_236595

-- Definitions derived from the conditions
def initial_files : Nat := 27
def deleted_files : Nat := 9
def files_per_folder : Nat := 6

-- Define the remaining files after deletion
def remaining_files : Nat := initial_files - deleted_files

-- The theorem: Prove that the number of folders is 3
theorem paige_folders : remaining_files / files_per_folder = 3 := by
  sorry

end NUMINAMATH_GPT_paige_folders_l2365_236595
