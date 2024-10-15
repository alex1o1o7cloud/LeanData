import Mathlib

namespace NUMINAMATH_GPT_totalBottleCaps_l1091_109107

-- Variables for the conditions
def bottleCapsPerBox : ℝ := 35.0
def numberOfBoxes : ℝ := 7.0

-- Theorem stating the equivalent proof problem
theorem totalBottleCaps : bottleCapsPerBox * numberOfBoxes = 245.0 := by
  sorry

end NUMINAMATH_GPT_totalBottleCaps_l1091_109107


namespace NUMINAMATH_GPT_system_of_equations_solution_l1091_109113

theorem system_of_equations_solution (x y z : ℝ) :
  (x * y + x * z = 8 - x^2) →
  (x * y + y * z = 12 - y^2) →
  (y * z + z * x = -4 - z^2) →
  (x = 2 ∧ y = 3 ∧ z = -1) ∨ (x = -2 ∧ y = -3 ∧ z = 1) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1091_109113


namespace NUMINAMATH_GPT_min_sales_required_l1091_109185

-- Definitions from conditions
def old_salary : ℝ := 75000
def new_base_salary : ℝ := 45000
def commission_rate : ℝ := 0.15
def sale_amount : ℝ := 750

-- Statement to be proven
theorem min_sales_required (n : ℕ) :
  n ≥ ⌈(old_salary - new_base_salary) / (commission_rate * sale_amount)⌉₊ :=
sorry

end NUMINAMATH_GPT_min_sales_required_l1091_109185


namespace NUMINAMATH_GPT_contrapositive_of_neg_and_inverse_l1091_109157

theorem contrapositive_of_neg_and_inverse (p r s : Prop) (h1 : r = ¬p) (h2 : s = ¬r) : s = (¬p → false) :=
by
  -- We have that r = ¬p
  have hr : r = ¬p := h1
  -- And we have that s = ¬r
  have hs : s = ¬r := h2
  -- Now we need to show that s is the contrapositive of p, which is ¬p → false
  sorry

end NUMINAMATH_GPT_contrapositive_of_neg_and_inverse_l1091_109157


namespace NUMINAMATH_GPT_possible_value_of_n_l1091_109175

theorem possible_value_of_n :
  ∃ (n : ℕ), (345564 - n) % (13 * 17 * 19) = 0 ∧ 0 < n ∧ n < 1000 ∧ n = 98 :=
sorry

end NUMINAMATH_GPT_possible_value_of_n_l1091_109175


namespace NUMINAMATH_GPT_value_of_f_at_5_l1091_109160

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_of_f_at_5 : f 5 = 15 := 
by {
  sorry
}

end NUMINAMATH_GPT_value_of_f_at_5_l1091_109160


namespace NUMINAMATH_GPT_parabola_equation_line_equation_chord_l1091_109164

section
variables (p : ℝ) (x_A y_A : ℝ) (M_x M_y : ℝ)
variable (h_p_pos : p > 0)
variable (h_A : y_A^2 = 8 * x_A)
variable (h_directrix_A : x_A + p / 2 = 5)
variable (h_M : (M_x, M_y) = (3, 2))

theorem parabola_equation (h_x_A : x_A = 3) : y_A^2 = 8 * x_A :=
sorry

theorem line_equation_chord
  (x1 x2 y1 y2 : ℝ)
  (h_parabola : y1^2 = 8 * x1 ∧ y2^2 = 8 * x2)
  (h_chord_M : (x1 + x2) / 2 = 3 ∧ (y1 + y2) / 2 = 2) :
  y_M - 2 * x_M + 4 = 0 :=
sorry
end

end NUMINAMATH_GPT_parabola_equation_line_equation_chord_l1091_109164


namespace NUMINAMATH_GPT_num_workers_l1091_109155

-- Define the number of workers (n) and the initial contribution per worker (x)
variable (n x : ℕ)

-- Condition 1: The total contribution is Rs. 3 lacs
axiom h1 : n * x = 300000

-- Condition 2: If each worker contributed Rs. 50 more, the total would be Rs. 3.75 lacs
axiom h2 : n * (x + 50) = 375000

-- Proof Problem: Prove that the number of workers (n) is 1500
theorem num_workers : n = 1500 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_num_workers_l1091_109155


namespace NUMINAMATH_GPT_fill_time_two_pipes_l1091_109189

variable (R : ℝ)
variable (c : ℝ)
variable (t1 : ℝ) (t2 : ℝ)

noncomputable def fill_time_with_pipes (num_pipes : ℝ) (time_per_tank : ℝ) : ℝ :=
  time_per_tank / num_pipes

theorem fill_time_two_pipes (h1 : fill_time_with_pipes 3 t1 = 12) 
                            (h2 : c = R)
                            : fill_time_with_pipes 2 (3 * R * t1) = 18 := 
by
  sorry

end NUMINAMATH_GPT_fill_time_two_pipes_l1091_109189


namespace NUMINAMATH_GPT_linemen_ounces_per_drink_l1091_109196

-- Definitions corresponding to the conditions.
def linemen := 12
def skill_position_drink := 6
def skill_position_before_refill := 5
def cooler_capacity := 126

-- The theorem that requires proof.
theorem linemen_ounces_per_drink (L : ℕ) (h : 12 * L + 5 * skill_position_drink = cooler_capacity) : L = 8 :=
by
  sorry

end NUMINAMATH_GPT_linemen_ounces_per_drink_l1091_109196


namespace NUMINAMATH_GPT_ineq_sqrt_two_l1091_109105

theorem ineq_sqrt_two (x y : ℝ) (h1 : x > y) (h2 : x * y = 1) : 
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_ineq_sqrt_two_l1091_109105


namespace NUMINAMATH_GPT_partition_value_l1091_109180

variable {a m n p x k l : ℝ}

theorem partition_value :
  (m * (a - n * x) = k * (a - n * x)) ∧
  (n * x = l * x) ∧
  (a - x = p * (a - m * (a - n * x)))
  → x = (a * (m * p - p + 1)) / (n * m * p + 1) :=
by
  sorry

end NUMINAMATH_GPT_partition_value_l1091_109180


namespace NUMINAMATH_GPT_amanda_more_than_average_l1091_109120

-- Conditions
def jill_peaches : ℕ := 12
def steven_peaches : ℕ := jill_peaches + 15
def jake_peaches : ℕ := steven_peaches - 16
def amanda_peaches : ℕ := jill_peaches * 2
def total_peaches : ℕ := jake_peaches + steven_peaches + jill_peaches
def average_peaches : ℚ := total_peaches / 3

-- Question: Prove that Amanda has 7.33 more peaches than the average peaches Jake, Steven, and Jill have
theorem amanda_more_than_average : amanda_peaches - average_peaches = 22 / 3 := by
  sorry

end NUMINAMATH_GPT_amanda_more_than_average_l1091_109120


namespace NUMINAMATH_GPT_range_of_a_l1091_109154

variable (a : ℝ)

theorem range_of_a (h : ¬ ∃ x : ℝ, x^2 + 2 * x + a ≤ 0) : 1 < a :=
by {
  -- Proof will go here.
  sorry
}

end NUMINAMATH_GPT_range_of_a_l1091_109154


namespace NUMINAMATH_GPT_denominator_or_divisor_cannot_be_zero_l1091_109116

theorem denominator_or_divisor_cannot_be_zero (a b c : ℝ) : b ≠ 0 ∧ c ≠ 0 → (a / b ≠ a ∨ a / c ≠ a) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_denominator_or_divisor_cannot_be_zero_l1091_109116


namespace NUMINAMATH_GPT_problem1_l1091_109111

variables (m n : ℝ)

axiom cond1 : 4 * m + n = 90
axiom cond2 : 2 * m - 3 * n = 10

theorem problem1 : (m + 2 * n) ^ 2 - (3 * m - n) ^ 2 = -900 := sorry

end NUMINAMATH_GPT_problem1_l1091_109111


namespace NUMINAMATH_GPT_triangle_is_right_l1091_109114

-- Define the side lengths of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- Define a predicate to check if a triangle is right using Pythagorean theorem
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- The proof problem statement
theorem triangle_is_right : is_right_triangle a b c :=
sorry

end NUMINAMATH_GPT_triangle_is_right_l1091_109114


namespace NUMINAMATH_GPT_largest_sum_is_sum3_l1091_109172

-- Definitions of the individual sums given in the conditions
def sum1 : ℚ := (1/4 : ℚ) + (1/5 : ℚ) * (1/2 : ℚ)
def sum2 : ℚ := (1/4 : ℚ) - (1/6 : ℚ)
def sum3 : ℚ := (1/4 : ℚ) + (1/3 : ℚ) * (1/2 : ℚ)
def sum4 : ℚ := (1/4 : ℚ) - (1/8 : ℚ)
def sum5 : ℚ := (1/4 : ℚ) + (1/7 : ℚ) * (1/2 : ℚ)

-- Theorem to prove that sum3 is the largest
theorem largest_sum_is_sum3 : sum3 = (5/12 : ℚ) ∧ sum3 > sum1 ∧ sum3 > sum2 ∧ sum3 > sum4 ∧ sum3 > sum5 := 
by 
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_largest_sum_is_sum3_l1091_109172


namespace NUMINAMATH_GPT_max_distinct_fans_l1091_109104

-- Definitions related to the problem conditions
def sectors := 6
def initial_configurations := 2 ^ sectors
def symmetrical_configurations := 8
def distinct_configurations := (initial_configurations - symmetrical_configurations) / 2 + symmetrical_configurations

-- The theorem to prove
theorem max_distinct_fans : distinct_configurations = 36 := by
  sorry

end NUMINAMATH_GPT_max_distinct_fans_l1091_109104


namespace NUMINAMATH_GPT_find_m_l1091_109140

theorem find_m (m : ℝ) (h₁: 0 < m) (h₂: ∀ p q : ℝ × ℝ, p = (m, 4) → q = (2, m) → ∃ s : ℝ, s = m^2 ∧ ((q.2 - p.2) / (q.1 - p.1)) = s) : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1091_109140


namespace NUMINAMATH_GPT_product_of_x_y_l1091_109142

variable (x y : ℝ)

-- Condition: EF = GH
def EF_eq_GH := (x^2 + 2 * x - 8 = 45)

-- Condition: FG = EH
def FG_eq_EH := (y^2 + 8 * y + 16 = 36)

-- Condition: y > 0
def y_pos := (y > 0)

theorem product_of_x_y : EF_eq_GH x ∧ FG_eq_EH y ∧ y_pos y → 
  x * y = -2 + 6 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_product_of_x_y_l1091_109142


namespace NUMINAMATH_GPT_sample_size_proof_l1091_109118

-- Define the quantities produced by each workshop
def units_A : ℕ := 120
def units_B : ℕ := 80
def units_C : ℕ := 60

-- Define the number of units sampled from Workshop C
def samples_C : ℕ := 3

-- Calculate the total sample size n
def total_sample_size : ℕ :=
  let sampling_fraction := samples_C / units_C
  let samples_A := sampling_fraction * units_A
  let samples_B := sampling_fraction * units_B
  samples_A + samples_B + samples_C

-- The theorem we want to prove
theorem sample_size_proof : total_sample_size = 13 :=
by sorry

end NUMINAMATH_GPT_sample_size_proof_l1091_109118


namespace NUMINAMATH_GPT_sum_of_remainders_mod_53_l1091_109119

theorem sum_of_remainders_mod_53 (d e f : ℕ) (hd : d % 53 = 19) (he : e % 53 = 33) (hf : f % 53 = 14) : 
  (d + e + f) % 53 = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_mod_53_l1091_109119


namespace NUMINAMATH_GPT_rachel_found_boxes_l1091_109170

theorem rachel_found_boxes (pieces_per_box total_pieces B : ℕ) 
  (h1 : pieces_per_box = 7) 
  (h2 : total_pieces = 49) 
  (h3 : B = total_pieces / pieces_per_box) : B = 7 := 
by 
  sorry

end NUMINAMATH_GPT_rachel_found_boxes_l1091_109170


namespace NUMINAMATH_GPT_greater_number_is_84_l1091_109145

theorem greater_number_is_84
  (x y : ℕ)
  (h1 : x * y = 2688)
  (h2 : (x + y) - (x - y) = 64)
  (h3 : x > y) : x = 84 :=
sorry

end NUMINAMATH_GPT_greater_number_is_84_l1091_109145


namespace NUMINAMATH_GPT_units_digit_p_plus_one_l1091_109156

theorem units_digit_p_plus_one (p : ℕ) (h1 : p % 2 = 0) (h2 : p % 10 ≠ 0)
  (h3 : (p ^ 3) % 10 = (p ^ 2) % 10) : (p + 1) % 10 = 7 :=
  sorry

end NUMINAMATH_GPT_units_digit_p_plus_one_l1091_109156


namespace NUMINAMATH_GPT_gym_cost_l1091_109136

theorem gym_cost (x : ℕ) (hx : x > 0) (h1 : 50 + 12 * x + 48 * x = 650) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_gym_cost_l1091_109136


namespace NUMINAMATH_GPT_geometric_sequence_sum_squared_l1091_109134

theorem geometric_sequence_sum_squared (a : ℕ → ℕ) (n : ℕ) (q : ℕ) 
    (h_geometric: ∀ n, a (n + 1) = a n * q)
    (h_a1 : a 1 = 2)
    (h_a3 : a 3 = 4) :
    (a 1)^2 + (a 2)^2 + (a 3)^2 + (a 4)^2 + (a 5)^2 + (a 6)^2 + (a 7)^2 + (a 8)^2 = 1020 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_squared_l1091_109134


namespace NUMINAMATH_GPT_smallest_b_value_l1091_109126

noncomputable def smallest_possible_value_of_b : ℝ :=
  (3 + Real.sqrt 5) / 2

theorem smallest_b_value
  (a b : ℝ)
  (h1 : 1 < a)
  (h2 : a < b)
  (h3 : b ≥ a + 1)
  (h4 : (1/b) + (1/a) ≤ 1) :
  b = smallest_possible_value_of_b :=
sorry

end NUMINAMATH_GPT_smallest_b_value_l1091_109126


namespace NUMINAMATH_GPT_Merrill_marbles_Vivian_marbles_l1091_109173

variable (M E S V : ℕ)

-- Conditions
axiom Merrill_twice_Elliot : M = 2 * E
axiom Merrill_Elliot_five_fewer_Selma : M + E = S - 5
axiom Selma_fifty_marbles : S = 50
axiom Vivian_35_percent_more_Elliot : V = (135 * E) / 100 -- since Lean works better with integers, use 135/100 instead of 1.35
axiom Vivian_Elliot_difference_greater_five : V - E > 5

-- Questions
theorem Merrill_marbles (M E S : ℕ) (h1: M = 2 * E) (h2: M + E = S - 5) (h3: S = 50) : M = 30 := by
  sorry

theorem Vivian_marbles (V E : ℕ) (h1: V = (135 * E) / 100) (h2: V - E > 5) (h3: E = 15) : V = 21 := by
  sorry

end NUMINAMATH_GPT_Merrill_marbles_Vivian_marbles_l1091_109173


namespace NUMINAMATH_GPT_largest_of_five_consecutive_integers_l1091_109108

theorem largest_of_five_consecutive_integers (n : ℕ) (h : n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 15120) : n + 4 = 9 :=
sorry

end NUMINAMATH_GPT_largest_of_five_consecutive_integers_l1091_109108


namespace NUMINAMATH_GPT_rank_classmates_l1091_109169

-- Definitions of the conditions
def emma_tallest (emma david fiona : ℕ) : Prop := emma > david ∧ emma > fiona
def fiona_not_shortest (david emma fiona : ℕ) : Prop := david > fiona ∧ emma > fiona
def david_not_tallest (david emma fiona : ℕ) : Prop := emma > david ∧ fiona > david

def exactly_one_true (david emma fiona : ℕ) : Prop :=
  (emma_tallest emma david fiona ∧ ¬fiona_not_shortest david emma fiona ∧ ¬david_not_tallest david emma fiona) ∨
  (¬emma_tallest emma david fiona ∧ fiona_not_shortest david emma fiona ∧ ¬david_not_tallest david emma fiona) ∨
  (¬emma_tallest emma david fiona ∧ ¬fiona_not_shortest david emma fiona ∧ david_not_tallest david emma fiona)

-- The final proof statement
theorem rank_classmates (david emma fiona : ℕ) (h : exactly_one_true david emma fiona) : david > fiona ∧ fiona > emma :=
  sorry

end NUMINAMATH_GPT_rank_classmates_l1091_109169


namespace NUMINAMATH_GPT_complex_multiplication_result_l1091_109167

-- Define the complex numbers used in the problem
def a : ℂ := 4 - 3 * Complex.I
def b : ℂ := 4 + 3 * Complex.I

-- State the theorem we want to prove
theorem complex_multiplication_result : a * b = 25 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_complex_multiplication_result_l1091_109167


namespace NUMINAMATH_GPT_solution_set_empty_iff_l1091_109127

def quadratic_no_solution (a b c : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (a * x^2 + b * x + c < 0)

theorem solution_set_empty_iff (a b c : ℝ) (h : quadratic_no_solution a b c) : a > 0 ∧ (b^2 - 4 * a * c ≤ 0) :=
sorry

end NUMINAMATH_GPT_solution_set_empty_iff_l1091_109127


namespace NUMINAMATH_GPT_cube_surface_area_l1091_109181

/-- A cube with an edge length of 10 cm has smaller cubes with edge length 2 cm 
    dug out from the middle of each face. The surface area of the new shape is 696 cm². -/
theorem cube_surface_area (original_edge : ℝ) (small_cube_edge : ℝ)
  (original_edge_eq : original_edge = 10) (small_cube_edge_eq : small_cube_edge = 2) :
  let original_surface := 6 * original_edge ^ 2
  let removed_area := 6 * small_cube_edge ^ 2
  let added_area := 6 * 5 * small_cube_edge ^ 2
  let new_surface := original_surface - removed_area + added_area
  new_surface = 696 := by
  sorry

end NUMINAMATH_GPT_cube_surface_area_l1091_109181


namespace NUMINAMATH_GPT_route_B_is_quicker_l1091_109110

theorem route_B_is_quicker : 
    let distance_A := 6 -- miles
    let speed_A := 30 -- mph
    let distance_B_total := 5 -- miles
    let distance_B_non_school := 4.5 -- miles
    let speed_B_non_school := 40 -- mph
    let distance_B_school := 0.5 -- miles
    let speed_B_school := 20 -- mph
    let time_A := (distance_A / speed_A) * 60 -- minutes
    let time_B_non_school := (distance_B_non_school / speed_B_non_school) * 60 -- minutes
    let time_B_school := (distance_B_school / speed_B_school) * 60 -- minutes
    let time_B := time_B_non_school + time_B_school -- minutes
    let time_difference := time_A - time_B -- minutes
    time_difference = 3.75 :=
sorry

end NUMINAMATH_GPT_route_B_is_quicker_l1091_109110


namespace NUMINAMATH_GPT_math_problem_l1091_109139

variable (a : ℝ)

theorem math_problem (h : a^2 + 3 * a - 2 = 0) :
  ((a^2 - 4) / (a^2 - 4 * a + 4) - 1 / (2 - a)) / (2 / (a^2 - 2 * a)) = 1 := 
sorry

end NUMINAMATH_GPT_math_problem_l1091_109139


namespace NUMINAMATH_GPT_radii_inequality_l1091_109128

variable {R1 R2 R3 r : ℝ}

/-- Given that R1, R2, and R3 are the radii of three circles passing through a vertex of a triangle 
and touching the opposite side, and r is the radius of the incircle of this triangle,
prove that 1 / R1 + 1 / R2 + 1 / R3 ≤ 1 / r. -/
theorem radii_inequality (h_ge : ∀ i : Fin 3, 0 < [R1, R2, R3][i]) (h_incircle : 0 < r) :
  (1 / R1) + (1 / R2) + (1 / R3) ≤ 1 / r :=
  sorry

end NUMINAMATH_GPT_radii_inequality_l1091_109128


namespace NUMINAMATH_GPT_old_edition_pages_l1091_109149

theorem old_edition_pages (x : ℕ) 
  (h₁ : 2 * x - 230 = 450) : x = 340 := 
by sorry

end NUMINAMATH_GPT_old_edition_pages_l1091_109149


namespace NUMINAMATH_GPT_girls_not_playing_soccer_l1091_109176

-- Define the given conditions
def students_total : Nat := 420
def boys_total : Nat := 312
def soccer_players_total : Nat := 250
def percent_boys_playing_soccer : Float := 0.78

-- Define the main goal based on the question and correct answer
theorem girls_not_playing_soccer : 
  students_total = 420 → 
  boys_total = 312 → 
  soccer_players_total = 250 → 
  percent_boys_playing_soccer = 0.78 → 
  ∃ (girls_not_playing_soccer : Nat), girls_not_playing_soccer = 53 :=
by 
  sorry

end NUMINAMATH_GPT_girls_not_playing_soccer_l1091_109176


namespace NUMINAMATH_GPT_seats_per_row_and_total_students_l1091_109179

theorem seats_per_row_and_total_students (R S : ℕ) 
  (h1 : S = 5 * R + 6) 
  (h2 : S = 12 * (R - 3)) : 
  R = 6 ∧ S = 36 := 
by 
  sorry

end NUMINAMATH_GPT_seats_per_row_and_total_students_l1091_109179


namespace NUMINAMATH_GPT_negation_of_proposition_l1091_109159

theorem negation_of_proposition (a : ℝ) : 
  ¬(a = -1 → a^2 = 1) ↔ (a ≠ -1 → a^2 ≠ 1) :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l1091_109159


namespace NUMINAMATH_GPT_equation_solution_l1091_109178

theorem equation_solution (x : ℝ) (h₁ : 2 * x - 5 ≠ 0) (h₂ : 5 - 2 * x ≠ 0) :
  (x / (2 * x - 5) + 5 / (5 - 2 * x) = 1) ↔ (x = 0) :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_l1091_109178


namespace NUMINAMATH_GPT_distinct_square_sum_100_l1091_109193

theorem distinct_square_sum_100 :
  ∀ a b c : ℕ, a ≠ b → b ≠ c → a ≠ c → 
  a^2 + b^2 + c^2 = 100 → false := by
  sorry

end NUMINAMATH_GPT_distinct_square_sum_100_l1091_109193


namespace NUMINAMATH_GPT_inequality_equivalence_l1091_109131

theorem inequality_equivalence (a : ℝ) : a < -1 ↔ a + 1 < 0 :=
by
  sorry

end NUMINAMATH_GPT_inequality_equivalence_l1091_109131


namespace NUMINAMATH_GPT_laura_change_l1091_109124

theorem laura_change : 
  let pants_cost := 2 * 54
  let shirts_cost := 4 * 33
  let total_cost := pants_cost + shirts_cost
  let amount_given := 250
  (amount_given - total_cost) = 10 :=
by
  -- definitions from conditions
  let pants_cost := 2 * 54
  let shirts_cost := 4 * 33
  let total_cost := pants_cost + shirts_cost
  let amount_given := 250

  -- the statement we are proving
  show (amount_given - total_cost) = 10
  sorry

end NUMINAMATH_GPT_laura_change_l1091_109124


namespace NUMINAMATH_GPT_intersection_M_N_l1091_109186

-- Given set M defined by the inequality
def M : Set ℝ := {x | x^2 + x - 6 < 0}

-- Given set N defined by the interval
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- The intersection M ∩ N should be equal to the interval [1, 2)
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1091_109186


namespace NUMINAMATH_GPT_total_money_found_l1091_109144

-- Define the conditions
def donna_share := 0.40
def friendA_share := 0.35
def friendB_share := 0.25
def donna_amount := 39.0

-- Define the problem statement/proof
theorem total_money_found (donna_share friendA_share friendB_share donna_amount : ℝ) 
  (h1 : donna_share = 0.40) 
  (h2 : friendA_share = 0.35) 
  (h3 : friendB_share = 0.25) 
  (h4 : donna_amount = 39.0) :
  ∃ total_money : ℝ, total_money = 97.50 := 
by
  -- The calculations and actual proof will go here
  sorry

end NUMINAMATH_GPT_total_money_found_l1091_109144


namespace NUMINAMATH_GPT_determinant_problem_l1091_109101

theorem determinant_problem 
  (x y z w : ℝ) 
  (h : x * w - y * z = 7) : 
  ((x * (8 * z + 4 * w)) - (z * (8 * x + 4 * y))) = 28 :=
by 
  sorry

end NUMINAMATH_GPT_determinant_problem_l1091_109101


namespace NUMINAMATH_GPT_part1_l1091_109123

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)
variable (h0 : ∀ x, 0 ≤ x → f x = Real.sqrt x)
variable (h1 : 0 ≤ x1)
variable (h2 : 0 ≤ x2)
variable (h3 : x1 ≠ x2)

theorem part1 : (1/2) * (f x1 + f x2) < f ((x1 + x2) / 2) :=
  sorry

end NUMINAMATH_GPT_part1_l1091_109123


namespace NUMINAMATH_GPT_find_charge_federal_return_l1091_109132

-- Definitions based on conditions
def charge_federal_return (F : ℝ) : ℝ := F
def charge_state_return : ℝ := 30
def charge_quarterly_return : ℝ := 80
def sold_federal_returns : ℝ := 60
def sold_state_returns : ℝ := 20
def sold_quarterly_returns : ℝ := 10
def total_revenue : ℝ := 4400

-- Lean proof statement to verify the value of F
theorem find_charge_federal_return (F : ℝ) (h : sold_federal_returns * charge_federal_return F + sold_state_returns * charge_state_return + sold_quarterly_returns * charge_quarterly_return = total_revenue) : 
  F = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_charge_federal_return_l1091_109132


namespace NUMINAMATH_GPT_system_solution_in_first_quadrant_l1091_109182

theorem system_solution_in_first_quadrant (c x y : ℝ)
  (h1 : x - y = 5)
  (h2 : c * x + y = 7)
  (hx : x > 3)
  (hy : y > 1) : c < 1 :=
sorry

end NUMINAMATH_GPT_system_solution_in_first_quadrant_l1091_109182


namespace NUMINAMATH_GPT_sheet_length_l1091_109163

theorem sheet_length (L : ℝ) : 
  (20 * L > 0) → 
  ((16 * (L - 6)) / (20 * L) = 0.64) → 
  L = 30 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_sheet_length_l1091_109163


namespace NUMINAMATH_GPT_spider_total_distance_l1091_109148

theorem spider_total_distance
    (radius : ℝ)
    (diameter : ℝ)
    (half_diameter : ℝ)
    (final_leg : ℝ)
    (total_distance : ℝ) :
    radius = 75 →
    diameter = 2 * radius →
    half_diameter = diameter / 2 →
    final_leg = 90 →
    (half_diameter ^ 2 + final_leg ^ 2 = diameter ^ 2) →
    total_distance = diameter + half_diameter + final_leg →
    total_distance = 315 :=
by
  intros
  sorry

end NUMINAMATH_GPT_spider_total_distance_l1091_109148


namespace NUMINAMATH_GPT_problem_equation_l1091_109103

def interest_rate : ℝ := 0.0306
def principal : ℝ := 5000
def interest_tax : ℝ := 0.20

theorem problem_equation (x : ℝ) :
  x + principal * interest_rate * interest_tax = principal * (1 + interest_rate) :=
sorry

end NUMINAMATH_GPT_problem_equation_l1091_109103


namespace NUMINAMATH_GPT_last_two_nonzero_digits_70_factorial_l1091_109162

theorem last_two_nonzero_digits_70_factorial : 
  let N := 70
  (∀ N : ℕ, 0 < N → N % 2 ≠ 0 → N % 5 ≠ 0 → ∃ x : ℕ, x % 100 = N % (N + (N! / (2 ^ 16)))) →
  (N! / 10 ^ 16) % 100 = 68 :=
by
sorry

end NUMINAMATH_GPT_last_two_nonzero_digits_70_factorial_l1091_109162


namespace NUMINAMATH_GPT_simplify_expression_l1091_109183

variable (y : ℝ)

theorem simplify_expression :
  y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8) = 4 * y^3 - 6 * y^2 + 15 * y - 48 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1091_109183


namespace NUMINAMATH_GPT_books_ratio_3_to_1_l1091_109195

-- Definitions based on the conditions
def initial_books : ℕ := 220
def books_rebecca_received : ℕ := 40
def remaining_books : ℕ := 60
def total_books_given_away := initial_books - remaining_books
def books_mara_received := total_books_given_away - books_rebecca_received

-- The proof that the ratio of the number of books Mara received to the number of books Rebecca received is 3:1
theorem books_ratio_3_to_1 : (books_mara_received : ℚ) / books_rebecca_received = 3 := by
  sorry

end NUMINAMATH_GPT_books_ratio_3_to_1_l1091_109195


namespace NUMINAMATH_GPT_should_agree_to_buy_discount_card_l1091_109194

noncomputable def total_cost_without_discount_card (cakes_cost fruits_cost : ℕ) : ℕ :=
  cakes_cost + fruits_cost

noncomputable def total_cost_with_discount_card (cakes_cost fruits_cost discount_card_cost : ℕ) : ℕ :=
  let total_cost := cakes_cost + fruits_cost
  let discount := total_cost * 3 / 100
  (total_cost - discount) + discount_card_cost

theorem should_agree_to_buy_discount_card : 
  let cakes_cost := 4 * 500
  let fruits_cost := 1600
  let discount_card_cost := 100
  total_cost_with_discount_card cakes_cost fruits_cost discount_card_cost < total_cost_without_discount_card cakes_cost fruits_cost :=
by
  sorry

end NUMINAMATH_GPT_should_agree_to_buy_discount_card_l1091_109194


namespace NUMINAMATH_GPT_value_of_a_l1091_109146

noncomputable def coefficient_of_x2_term (a : ℝ) : ℝ :=
  a^4 * Nat.choose 8 4

theorem value_of_a (a : ℝ) (h : coefficient_of_x2_term a = 70) : a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l1091_109146


namespace NUMINAMATH_GPT_algebraic_expression_value_l1091_109138

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y = 1) :
  (2 * x + 4 * y) / (x^2 + 4 * x * y + 4 * y^2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1091_109138


namespace NUMINAMATH_GPT_rate_of_discount_l1091_109168

theorem rate_of_discount (Marked_Price Selling_Price : ℝ) (h_marked : Marked_Price = 80) (h_selling : Selling_Price = 68) : 
  ((Marked_Price - Selling_Price) / Marked_Price) * 100 = 15 :=
by
  -- Definitions from conditions
  rw [h_marked, h_selling]
  -- Substitute the values and simplify
  sorry

end NUMINAMATH_GPT_rate_of_discount_l1091_109168


namespace NUMINAMATH_GPT_backpacks_weight_l1091_109174

variables (w_y w_g : ℝ)

theorem backpacks_weight :
  (2 * w_y + 3 * w_g = 44) ∧
  (w_y + w_g + w_g / 2 = w_g + w_y / 2) →
  (w_g = 4) ∧ (w_y = 12) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_backpacks_weight_l1091_109174


namespace NUMINAMATH_GPT_apples_weight_l1091_109137

theorem apples_weight (x : ℝ) (price1 : ℝ) (price2 : ℝ) (new_price_diff : ℝ) (total_revenue : ℝ)
  (h1 : price1 * x = 228)
  (h2 : price2 * (x + 5) = 180)
  (h3 : ∀ kg: ℝ, kg * (price1 - new_price_diff) = total_revenue)
  (h4 : new_price_diff = 0.9)
  (h5 : total_revenue = 408) :
  2 * x + 5 = 85 :=
by
  sorry

end NUMINAMATH_GPT_apples_weight_l1091_109137


namespace NUMINAMATH_GPT_incorrect_option_l1091_109129

theorem incorrect_option (a : ℝ) (h : a ≠ 0) : (a + 2) ^ 0 ≠ 1 ↔ a = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_incorrect_option_l1091_109129


namespace NUMINAMATH_GPT_total_number_of_coins_l1091_109171

theorem total_number_of_coins (x n : Nat) (h1 : 15 * 5 = 75) (h2 : 125 - 75 = 50)
  (h3 : x = 50 / 2) (h4 : n = x + 15) : n = 40 := by
  sorry

end NUMINAMATH_GPT_total_number_of_coins_l1091_109171


namespace NUMINAMATH_GPT_solve_system_l1091_109153

-- Define the system of equations
def system_of_equations (a b c x y z : ℝ) :=
  x ≠ y ∧
  a ≠ 0 ∧
  c ≠ 0 ∧
  (x + z) * a = x - y ∧
  (x + z) * b = x^2 - y^2 ∧
  (x + z)^2 * (b^2 / (a^2 * c)) = (x^3 + x^2 * y - x * y^2 - y^3)

-- Proof goal: establish the values of x, y, and z
theorem solve_system (a b c x y z : ℝ) (h : system_of_equations a b c x y z):
  x = (a^3 * c + b) / (2 * a) ∧
  y = (b - a^3 * c) / (2 * a) ∧
  z = (2 * a^2 * c - a^3 * c - b) / (2 * a) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1091_109153


namespace NUMINAMATH_GPT_line_intersection_equation_of_l4_find_a_l1091_109109

theorem line_intersection (P : ℝ × ℝ)
    (h1: 3 * P.1 + 4 * P.2 - 2 = 0)
    (h2: 2 * P.1 + P.2 + 2 = 0) :
  P = (-2, 2) :=
sorry

theorem equation_of_l4 (l4 : ℝ → ℝ → Prop)
    (P : ℝ × ℝ) (h1: 3 * P.1 + 4 * P.2 - 2 = 0)
    (h2: 2 * P.1 + P.2 + 2 = 0) 
    (h_parallel: ∀ x y, l4 x y ↔ y = 1/2 * x + 3)
    (x y : ℝ) :
  l4 x y ↔ y = 1/2 * x + 3 :=
sorry

theorem find_a (a : ℝ) :
    (∀ x y, 2 * x + y + 2 = 0 → y = -2 * x - 2) →
    (∀ x y, a * x - 2 * y + 1 = 0 → y = 1/2 * x - 1/2) →
    a = 1 :=
sorry

end NUMINAMATH_GPT_line_intersection_equation_of_l4_find_a_l1091_109109


namespace NUMINAMATH_GPT_Jaco_budget_for_parents_gifts_l1091_109187

theorem Jaco_budget_for_parents_gifts :
  ∃ (m n : ℕ), (m = 14 ∧ n = 14) ∧ 
  (∀ (friends gifts_friends budget : ℕ), 
   friends = 8 → gifts_friends = 9 → budget = 100 → 
   (budget - (friends * gifts_friends)) / 2 = m ∧ 
   (budget - (friends * gifts_friends)) / 2 = n) := 
sorry

end NUMINAMATH_GPT_Jaco_budget_for_parents_gifts_l1091_109187


namespace NUMINAMATH_GPT_fraction_sum_l1091_109165

theorem fraction_sum : (1/4 : ℚ) + (3/9 : ℚ) = (7/12 : ℚ) := 
  by 
  sorry

end NUMINAMATH_GPT_fraction_sum_l1091_109165


namespace NUMINAMATH_GPT_binom_eight_five_l1091_109188

theorem binom_eight_five :
  Nat.choose 8 5 = 56 :=
sorry

end NUMINAMATH_GPT_binom_eight_five_l1091_109188


namespace NUMINAMATH_GPT_pond_to_field_ratio_l1091_109122

theorem pond_to_field_ratio 
  (w l : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : l = 28)
  (side_pond : ℝ := 7) 
  (A_pond : ℝ := side_pond ^ 2) 
  (A_field : ℝ := l * w):
  (A_pond / A_field) = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_pond_to_field_ratio_l1091_109122


namespace NUMINAMATH_GPT_sequence_term_37_l1091_109199

theorem sequence_term_37 (n : ℕ) (h_pos : 0 < n) (h_eq : 3 * n + 1 = 37) : n = 12 :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_37_l1091_109199


namespace NUMINAMATH_GPT_unique_solution_l1091_109121

theorem unique_solution (x : ℝ) : 
  ∃! x, 2003^x + 2004^x = 2005^x := 
sorry

end NUMINAMATH_GPT_unique_solution_l1091_109121


namespace NUMINAMATH_GPT_probability_at_least_one_heart_l1091_109198

theorem probability_at_least_one_heart (total_cards hearts : ℕ) 
  (top_card_positions : Π n : ℕ, n = 3) 
  (non_hearts_cards : Π n : ℕ, n = total_cards - hearts) 
  (h_total_cards : total_cards = 52) (h_hearts : hearts = 13) 
  : (1 - ((39 * 38 * 37 : ℚ) / (52 * 51 * 50))) = (325 / 425) := 
by {
  sorry
}

end NUMINAMATH_GPT_probability_at_least_one_heart_l1091_109198


namespace NUMINAMATH_GPT_range_of_m_l1091_109158

theorem range_of_m (m : ℝ) (P : Prop) (Q : Prop) : 
  (P ∨ Q) ∧ ¬(P ∧ Q) →
  (P ↔ (m^2 - 4 > 0)) →
  (Q ↔ (16 * (m - 2)^2 - 16 < 0)) →
  (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_range_of_m_l1091_109158


namespace NUMINAMATH_GPT_correct_formulas_l1091_109130

theorem correct_formulas (n : ℕ) :
  ((2 * n - 1)^2 - 4 * (n * (n - 1)) / 2) = (2 * n^2 - 2 * n + 1) ∧ 
  (1 + ((n - 1) * n) / 2 * 4) = (2 * n^2 - 2 * n + 1) ∧ 
  ((n - 1)^2 + n^2) = (2 * n^2 - 2 * n + 1) := by
  sorry

end NUMINAMATH_GPT_correct_formulas_l1091_109130


namespace NUMINAMATH_GPT_count_four_digit_numbers_with_5_or_7_l1091_109102

def num_four_digit_numbers : Nat := 9000
def exclude_first_digit : Finset Nat := {1, 2, 3, 4, 6, 8, 9}  -- 7 options
def exclude_other_digits : Finset Nat := {0, 1, 2, 3, 4, 6, 8, 9}  -- 8 options
def excluded_numbers_count : Nat := exclude_first_digit.card * exclude_other_digits.card ^ 3  -- 3584
def included_numbers_count : Nat := num_four_digit_numbers - excluded_numbers_count  -- 5416

theorem count_four_digit_numbers_with_5_or_7 :
  included_numbers_count = 5416 :=
by
  sorry

end NUMINAMATH_GPT_count_four_digit_numbers_with_5_or_7_l1091_109102


namespace NUMINAMATH_GPT_contrapositive_equivalent_l1091_109184

variable {α : Type*} (A B : Set α) (x : α)

theorem contrapositive_equivalent : (x ∈ A → x ∈ B) ↔ (x ∉ B → x ∉ A) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_equivalent_l1091_109184


namespace NUMINAMATH_GPT_sequence_increasing_l1091_109151

theorem sequence_increasing (a : ℕ → ℝ) (a0 : ℝ) (h0 : a 0 = 1 / 5)
  (H : ∀ n : ℕ, a (n + 1) = 2^n - 3 * a n) :
  ∀ n : ℕ, a (n + 1) > a n :=
sorry

end NUMINAMATH_GPT_sequence_increasing_l1091_109151


namespace NUMINAMATH_GPT_find_expression_l1091_109190

theorem find_expression (x y : ℝ) (h1 : 3 * x + y = 7) (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 :=
by
  sorry

end NUMINAMATH_GPT_find_expression_l1091_109190


namespace NUMINAMATH_GPT_round_robin_games_count_l1091_109141

theorem round_robin_games_count (n : ℕ) (h : n = 6) : 
  (n * (n - 1)) / 2 = 15 := by
  sorry

end NUMINAMATH_GPT_round_robin_games_count_l1091_109141


namespace NUMINAMATH_GPT_angle_ABC_is_45_l1091_109115

theorem angle_ABC_is_45
  (x : ℝ)
  (h1 : ∀ (ABC : ℝ), x = 180 - ABC → x = 45) :
  2 * (x / 2) = (180 - x) / 6 → x = 45 :=
by
  sorry

end NUMINAMATH_GPT_angle_ABC_is_45_l1091_109115


namespace NUMINAMATH_GPT_total_fish_l1091_109112

theorem total_fish {lilly_fish rosy_fish : ℕ} (h1 : lilly_fish = 10) (h2 : rosy_fish = 11) : 
lilly_fish + rosy_fish = 21 :=
by 
  sorry

end NUMINAMATH_GPT_total_fish_l1091_109112


namespace NUMINAMATH_GPT_digit_8_appears_300_times_l1091_109117

-- Define a function that counts the occurrences of a specific digit in a list of numbers
def count_digit_occurrences (digit : Nat) (range : List Nat) : Nat :=
  range.foldl (λ acc n => acc + (Nat.digits 10 n).count digit) 0

-- Theorem statement: The digit 8 appears 300 times in the list of integers from 1 to 1000
theorem digit_8_appears_300_times : count_digit_occurrences 8 (List.range' 0 1000) = 300 :=
by
  sorry

end NUMINAMATH_GPT_digit_8_appears_300_times_l1091_109117


namespace NUMINAMATH_GPT_polynomial_square_b_value_l1091_109106

theorem polynomial_square_b_value (a b : ℚ) (h : ∃ (p q : ℚ), x^4 + 3 * x^3 + x^2 + a * x + b = (x^2 + p * x + q)^2) : 
  b = 25/64 := 
by 
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_polynomial_square_b_value_l1091_109106


namespace NUMINAMATH_GPT_daily_earning_r_l1091_109161

theorem daily_earning_r :
  exists P Q R : ℝ, 
    (P + Q + R = 220) ∧
    (P + R = 120) ∧
    (Q + R = 130) ∧
    (R = 30) := 
by
  sorry

end NUMINAMATH_GPT_daily_earning_r_l1091_109161


namespace NUMINAMATH_GPT_alice_age_proof_l1091_109135

-- Definitions derived from the conditions
def alice_pens : ℕ := 60
def clara_pens : ℕ := (2 * alice_pens) / 5
def clara_age_in_5_years : ℕ := 61
def clara_current_age : ℕ := clara_age_in_5_years - 5
def age_difference : ℕ := alice_pens - clara_pens

-- Proof statement to be proved
theorem alice_age_proof : (clara_current_age - age_difference = 20) :=
sorry

end NUMINAMATH_GPT_alice_age_proof_l1091_109135


namespace NUMINAMATH_GPT_volume_ratio_cones_l1091_109147

theorem volume_ratio_cones :
  let rC := 16.5
  let hC := 33
  let rD := 33
  let hD := 16.5
  let VC := (1 / 3) * Real.pi * rC^2 * hC
  let VD := (1 / 3) * Real.pi * rD^2 * hD
  (VC / VD) = (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_cones_l1091_109147


namespace NUMINAMATH_GPT_cost_per_toy_initially_l1091_109197

-- defining conditions
def num_toys : ℕ := 200
def percent_sold : ℝ := 0.8
def price_per_toy : ℝ := 30
def profit : ℝ := 800

-- defining the problem
theorem cost_per_toy_initially :
  ((num_toys * percent_sold) * price_per_toy - profit) / (num_toys * percent_sold) = 25 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_toy_initially_l1091_109197


namespace NUMINAMATH_GPT_THIS_code_is_2345_l1091_109100

def letterToDigit (c : Char) : Option Nat :=
  match c with
  | 'M' => some 0
  | 'A' => some 1
  | 'T' => some 2
  | 'H' => some 3
  | 'I' => some 4
  | 'S' => some 5
  | 'F' => some 6
  | 'U' => some 7
  | 'N' => some 8
  | _   => none

def codeToNumber (code : String) : Option String :=
  code.toList.mapM letterToDigit >>= fun digits => some (digits.foldl (fun acc d => acc ++ toString d) "")

theorem THIS_code_is_2345 :
  codeToNumber "THIS" = some "2345" :=
by
  sorry

end NUMINAMATH_GPT_THIS_code_is_2345_l1091_109100


namespace NUMINAMATH_GPT_price_increase_and_decrease_l1091_109150

theorem price_increase_and_decrease (P : ℝ) (x : ℝ) 
  (h1 : 0 < P) 
  (h2 : (P * (1 - (x / 100) ^ 2)) = 0.81 * P) : 
  abs (x - 44) < 1 :=
by
  sorry

end NUMINAMATH_GPT_price_increase_and_decrease_l1091_109150


namespace NUMINAMATH_GPT_zhang_hua_new_year_cards_l1091_109192

theorem zhang_hua_new_year_cards (x y z : ℕ) 
  (h1 : Nat.lcm (Nat.lcm x y) z = 60)
  (h2 : Nat.gcd x y = 4)
  (h3 : Nat.gcd y z = 3) : 
  x = 4 ∨ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_zhang_hua_new_year_cards_l1091_109192


namespace NUMINAMATH_GPT_maximal_sector_angle_l1091_109143

theorem maximal_sector_angle (a : ℝ) (r : ℝ) (l : ℝ) (α : ℝ)
  (h1 : l + 2 * r = a)
  (h2 : 0 < r ∧ r < a / 2)
  (h3 : α = l / r)
  (eval_area : ∀ (l r : ℝ), S = 1 / 2 * l * r)
  (S : ℝ) :
  α = 2 := sorry

end NUMINAMATH_GPT_maximal_sector_angle_l1091_109143


namespace NUMINAMATH_GPT_age_of_person_l1091_109191

/-- Given that Noah's age is twice someone's age and Noah will be 22 years old after 10 years, 
    this theorem states that the age of the person whose age is half of Noah's age is 6 years old. -/
theorem age_of_person (N : ℕ) (P : ℕ) (h1 : P = N / 2) (h2 : N + 10 = 22) : P = 6 := by
  sorry

end NUMINAMATH_GPT_age_of_person_l1091_109191


namespace NUMINAMATH_GPT_range_of_k_l1091_109133

def f (x : ℝ) : ℝ := x^3 - 12*x

def not_monotonic_on_I (k : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), k - 1 < x₁ ∧ x₁ < k + 1 ∧ k - 1 < x₂ ∧ x₂ < k + 1 ∧ x₁ ≠ x₂ ∧ (f x₁ - f x₂) * (x₁ - x₂) < 0

theorem range_of_k (k : ℝ) : not_monotonic_on_I k ↔ (k > -3 ∧ k < -1) ∨ (k > 1 ∧ k < 3) :=
sorry

end NUMINAMATH_GPT_range_of_k_l1091_109133


namespace NUMINAMATH_GPT_income_expenditure_ratio_l1091_109125

theorem income_expenditure_ratio
  (I E : ℕ)
  (h1 : I = 18000)
  (S : ℕ)
  (h2 : S = 2000)
  (h3 : S = I - E) :
  I.gcd E = 2000 ∧ I / I.gcd E = 9 ∧ E / I.gcd E = 8 :=
by sorry

end NUMINAMATH_GPT_income_expenditure_ratio_l1091_109125


namespace NUMINAMATH_GPT_slant_asymptote_sum_l1091_109177

theorem slant_asymptote_sum (m b : ℝ) 
  (h : ∀ x : ℝ, y = 3*x^2 + 4*x - 8 / (x - 4) → y = m*x + b) :
  m + b = 19 :=
sorry

end NUMINAMATH_GPT_slant_asymptote_sum_l1091_109177


namespace NUMINAMATH_GPT_profit_difference_l1091_109166

variable (a_capital b_capital c_capital b_profit : ℕ)

theorem profit_difference (h₁ : a_capital = 8000) (h₂ : b_capital = 10000) 
                          (h₃ : c_capital = 12000) (h₄ : b_profit = 2000) : 
  c_capital * (b_profit / b_capital) - a_capital * (b_profit / b_capital) = 800 := 
sorry

end NUMINAMATH_GPT_profit_difference_l1091_109166


namespace NUMINAMATH_GPT_cos_square_minus_sin_square_15_l1091_109152

theorem cos_square_minus_sin_square_15 (cos_30 : Real.cos (30 * Real.pi / 180) = (Real.sqrt 3) / 2) : 
  Real.cos (15 * Real.pi / 180) ^ 2 - Real.sin (15 * Real.pi / 180) ^ 2 = (Real.sqrt 3) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_cos_square_minus_sin_square_15_l1091_109152
