import Mathlib

namespace NUMINAMATH_GPT_xy_difference_l1737_173716

theorem xy_difference (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 - y^2 = 12) : x - y = 2 := by
  sorry

end NUMINAMATH_GPT_xy_difference_l1737_173716


namespace NUMINAMATH_GPT_arrangment_ways_basil_tomato_l1737_173707

theorem arrangment_ways_basil_tomato (basil_plants tomato_plants : Finset ℕ) 
  (hb : basil_plants.card = 5) 
  (ht : tomato_plants.card = 3) 
  : (∃ total_ways : ℕ, total_ways = 4320) :=
by
  sorry

end NUMINAMATH_GPT_arrangment_ways_basil_tomato_l1737_173707


namespace NUMINAMATH_GPT_root_difference_geom_prog_l1737_173719

theorem root_difference_geom_prog
  (x1 x2 x3 : ℝ)
  (h1 : 8 * x1^3 - 22 * x1^2 + 15 * x1 - 2 = 0)
  (h2 : 8 * x2^3 - 22 * x2^2 + 15 * x2 - 2 = 0)
  (h3 : 8 * x3^3 - 22 * x3^2 + 15 * x3 - 2 = 0)
  (geom_prog : ∃ (a r : ℝ), x1 = a / r ∧ x2 = a ∧ x3 = a * r) :
  |x3 - x1| = 33 / 14 :=
by
  sorry

end NUMINAMATH_GPT_root_difference_geom_prog_l1737_173719


namespace NUMINAMATH_GPT_contracting_arrangements_1680_l1737_173741

def num_contracting_arrangements (n a b c d : ℕ) : ℕ :=
  Nat.choose n a * Nat.choose (n - a) b * Nat.choose (n - a - b) c

theorem contracting_arrangements_1680 : num_contracting_arrangements 8 3 1 2 2 = 1680 := by
  unfold num_contracting_arrangements
  simp
  sorry

end NUMINAMATH_GPT_contracting_arrangements_1680_l1737_173741


namespace NUMINAMATH_GPT_disjunction_of_p_and_q_l1737_173703

-- Define the propositions p and q
variable (p q : Prop)

-- Assume that p is true and q is false
theorem disjunction_of_p_and_q (h1 : p) (h2 : ¬q) : p ∨ q := 
sorry

end NUMINAMATH_GPT_disjunction_of_p_and_q_l1737_173703


namespace NUMINAMATH_GPT_parabola_distance_to_focus_l1737_173754

theorem parabola_distance_to_focus (P : ℝ × ℝ) (y_axis_dist : ℝ) (hx : P.1 = 4) (hy : P.2 ^ 2 = 32) :
  (P.1 - 2) ^ 2 + P.2 ^ 2 = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_parabola_distance_to_focus_l1737_173754


namespace NUMINAMATH_GPT_find_b_from_conditions_l1737_173767

theorem find_b_from_conditions (x y z k : ℝ) (h1 : (x + y) / 2 = k) (h2 : (z + x) / 3 = k) (h3 : (y + z) / 4 = k) (h4 : x + y + z = 36) : x + y = 16 := 
by 
  sorry

end NUMINAMATH_GPT_find_b_from_conditions_l1737_173767


namespace NUMINAMATH_GPT_regular_polygon_sides_160_l1737_173710

theorem regular_polygon_sides_160 (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle : ℝ) = 160) : 
  n = 18 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_160_l1737_173710


namespace NUMINAMATH_GPT_decreasing_direct_proportion_l1737_173712

theorem decreasing_direct_proportion (k : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → k * x1 > k * x2) : k < 0 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_direct_proportion_l1737_173712


namespace NUMINAMATH_GPT_max_abs_f_le_f0_f1_l1737_173723

noncomputable def f (a b x : ℝ) : ℝ := 3 * a * x^2 - 2 * (a + b) * x + b

theorem max_abs_f_le_f0_f1 (a b : ℝ) (h : 0 < a) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) :
  |f a b x| ≤ max (|f a b 0|) (|f a b 1|) :=
sorry

end NUMINAMATH_GPT_max_abs_f_le_f0_f1_l1737_173723


namespace NUMINAMATH_GPT_solve_quadratic_l1737_173726

theorem solve_quadratic :
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ (x = 2 ∨ x = -1) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1737_173726


namespace NUMINAMATH_GPT_shaded_area_eq_l1737_173720

theorem shaded_area_eq : 
  let side := 8 
  let radius := 3 
  let square_area := side * side
  let sector_area := (1 / 4) * Real.pi * (radius * radius)
  let four_sectors_area := 4 * sector_area
  let triangle_area := (1 / 2) * radius * radius
  let four_triangles_area := 4 * triangle_area
  let shaded_area := square_area - four_sectors_area - four_triangles_area
  shaded_area = 64 - 9 * Real.pi - 18 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_eq_l1737_173720


namespace NUMINAMATH_GPT_triangle_height_l1737_173771

theorem triangle_height (b h : ℕ) (A : ℕ) (hA : A = 50) (hb : b = 10) :
  A = (1 / 2 : ℝ) * b * h → h = 10 := 
by
  sorry

end NUMINAMATH_GPT_triangle_height_l1737_173771


namespace NUMINAMATH_GPT_compute_value_l1737_173791

theorem compute_value : ((-120) - (-60)) / (-30) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_compute_value_l1737_173791


namespace NUMINAMATH_GPT_ellipse_condition_l1737_173708

theorem ellipse_condition (m : ℝ) :
  (m > 0) ∧ (2 * m - 1 > 0) ∧ (m ≠ 2 * m - 1) ↔ (m > 1/2) ∧ (m ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_condition_l1737_173708


namespace NUMINAMATH_GPT_carrie_bought_t_shirts_l1737_173742

theorem carrie_bought_t_shirts (total_spent : ℝ) (cost_each : ℝ) (n : ℕ) 
    (h_total : total_spent = 199) (h_cost : cost_each = 9.95) 
    (h_eq : n = total_spent / cost_each) : n = 20 := 
by
sorry

end NUMINAMATH_GPT_carrie_bought_t_shirts_l1737_173742


namespace NUMINAMATH_GPT_karen_savings_over_30_years_l1737_173789

theorem karen_savings_over_30_years 
  (P_exp : ℕ) (L_exp : ℕ) 
  (P_cheap : ℕ) (L_cheap : ℕ) 
  (T : ℕ)
  (hP_exp : P_exp = 300)
  (hL_exp : L_exp = 15)
  (hP_cheap : P_cheap = 120)
  (hL_cheap : L_cheap = 5)
  (hT : T = 30) : 
  (P_cheap * (T / L_cheap) - P_exp * (T / L_exp)) = 120 := 
by 
  sorry

end NUMINAMATH_GPT_karen_savings_over_30_years_l1737_173789


namespace NUMINAMATH_GPT_percentage_profits_to_revenues_l1737_173757

theorem percentage_profits_to_revenues (R P : ℝ) 
  (h1 : R > 0) 
  (h2 : P > 0)
  (h3 : 0.12 * R = 1.2 * P) 
  : P / R = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_percentage_profits_to_revenues_l1737_173757


namespace NUMINAMATH_GPT_solve_equation_l1737_173778

theorem solve_equation (x : ℝ) :
  ((x - 2)^2 - 4 = 0) ↔ (x = 4 ∨ x = 0) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1737_173778


namespace NUMINAMATH_GPT_ball_bounce_l1737_173721

theorem ball_bounce :
  ∃ b : ℕ, 324 * (3 / 4) ^ b < 40 ∧ b = 8 :=
by
  have : (3 / 4 : ℝ) < 1 := by norm_num
  have h40_324 : (40 : ℝ) / 324 = 10 / 81 := by norm_num
  sorry

end NUMINAMATH_GPT_ball_bounce_l1737_173721


namespace NUMINAMATH_GPT_nth_equation_pattern_l1737_173790

theorem nth_equation_pattern (n : ℕ) (hn : 0 < n) : n^2 - n = n * (n - 1) := by
  sorry

end NUMINAMATH_GPT_nth_equation_pattern_l1737_173790


namespace NUMINAMATH_GPT_smallest_a_value_l1737_173766

theorem smallest_a_value 
  (a b c : ℚ) 
  (a_pos : a > 0)
  (vertex_condition : ∃(x₀ y₀ : ℚ), x₀ = -1/3 ∧ y₀ = -4/3 ∧ y = a * (x + x₀)^2 + y₀)
  (integer_condition : ∃(n : ℤ), a + b + c = n)
  : a = 3/16 := 
sorry

end NUMINAMATH_GPT_smallest_a_value_l1737_173766


namespace NUMINAMATH_GPT_binom_600_600_l1737_173793

open Nat

theorem binom_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end NUMINAMATH_GPT_binom_600_600_l1737_173793


namespace NUMINAMATH_GPT_parallel_lines_a_eq_3_l1737_173764

theorem parallel_lines_a_eq_3
  (a : ℝ)
  (l1 : a^2 * x - y + a^2 - 3 * a = 0)
  (l2 : (4 * a - 3) * x - y - 2 = 0)
  (h : ∀ x y, a^2 * x - y + a^2 - 3 * a = (4 * a - 3) * x - y - 2) :
  a = 3 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_a_eq_3_l1737_173764


namespace NUMINAMATH_GPT_inverse_function_solution_l1737_173752

noncomputable def f (a b x : ℝ) := 2 / (a * x + b)

theorem inverse_function_solution (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : f a b 2 = 1 / 2) : b = 1 - 2 * a :=
by
  -- Assuming the inverse function condition means f(2) should be evaluated.
  sorry

end NUMINAMATH_GPT_inverse_function_solution_l1737_173752


namespace NUMINAMATH_GPT_min_value_fraction_l1737_173748

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  (4 / x + 9 / y) ≥ 25 :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l1737_173748


namespace NUMINAMATH_GPT_compute_expression_eq_162_l1737_173758

theorem compute_expression_eq_162 : 
  3 * 3^4 - 9^35 / 9^33 = 162 := 
by 
  sorry

end NUMINAMATH_GPT_compute_expression_eq_162_l1737_173758


namespace NUMINAMATH_GPT_line_segment_parametric_curve_l1737_173759

noncomputable def parametric_curve (θ : ℝ) := 
  (2 + Real.cos θ ^ 2, 1 - Real.sin θ ^ 2)

theorem line_segment_parametric_curve : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → 
    ∃ x y : ℝ, (x, y) = parametric_curve θ ∧ 2 ≤ x ∧ x ≤ 3 ∧ x - y = 2) := 
sorry

end NUMINAMATH_GPT_line_segment_parametric_curve_l1737_173759


namespace NUMINAMATH_GPT_midpoint_AB_l1737_173761

noncomputable def s (x t : ℝ) : ℝ := (x + t)^2 + (x - t)^2

noncomputable def CP (x : ℝ) : ℝ := x * Real.sqrt 3 / 2

theorem midpoint_AB (x : ℝ) (P : ℝ) : 
    (s x 0 = 2 * CP x ^ 2) ↔ P = x :=
by
    sorry

end NUMINAMATH_GPT_midpoint_AB_l1737_173761


namespace NUMINAMATH_GPT_handshakes_count_l1737_173732

-- Define the number of people
def num_people : ℕ := 10

-- Define a function to calculate the number of handshakes
noncomputable def num_handshakes (n : ℕ) : ℕ :=
  (n - 1) * n / 2

-- The main statement to be proved
theorem handshakes_count : num_handshakes num_people = 45 := by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_handshakes_count_l1737_173732


namespace NUMINAMATH_GPT_tinas_extra_earnings_l1737_173736

def price_per_candy_bar : ℕ := 2
def marvins_candy_bars_sold : ℕ := 35
def tinas_candy_bars_sold : ℕ := 3 * marvins_candy_bars_sold

def marvins_earnings : ℕ := marvins_candy_bars_sold * price_per_candy_bar
def tinas_earnings : ℕ := tinas_candy_bars_sold * price_per_candy_bar

theorem tinas_extra_earnings : tinas_earnings - marvins_earnings = 140 := by
  sorry

end NUMINAMATH_GPT_tinas_extra_earnings_l1737_173736


namespace NUMINAMATH_GPT_greatest_large_chips_l1737_173731

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, 2 ≤ a ∧ 2 ≤ b ∧ n = a * b

theorem greatest_large_chips (s l : ℕ) (c : ℕ) (hc : is_composite c) (h : s + l = 60) (hs : s = l + c) :
  l ≤ 28 :=
sorry

end NUMINAMATH_GPT_greatest_large_chips_l1737_173731


namespace NUMINAMATH_GPT_bridge_length_is_100_l1737_173776

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (wind_speed_kmh : ℝ) (crossing_time_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let wind_speed_ms := wind_speed_kmh * 1000 / 3600
  let effective_speed_ms := train_speed_ms - wind_speed_ms
  let distance_covered := effective_speed_ms * crossing_time_s
  distance_covered - train_length

theorem bridge_length_is_100 :
  length_of_bridge 150 45 15 30 = 100 :=
by
  sorry

end NUMINAMATH_GPT_bridge_length_is_100_l1737_173776


namespace NUMINAMATH_GPT_simplify_expression_l1737_173725

theorem simplify_expression : 1 - (1 / (1 + Real.sqrt 2)) + (1 / (1 - Real.sqrt 2)) = 1 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1737_173725


namespace NUMINAMATH_GPT_fgh_supermarkets_in_us_more_than_canada_l1737_173753

theorem fgh_supermarkets_in_us_more_than_canada
  (total_supermarkets : ℕ)
  (us_supermarkets : ℕ)
  (canada_supermarkets : ℕ)
  (h1 : total_supermarkets = 70)
  (h2 : us_supermarkets = 42)
  (h3 : us_supermarkets + canada_supermarkets = total_supermarkets):
  us_supermarkets - canada_supermarkets = 14 :=
by
  sorry

end NUMINAMATH_GPT_fgh_supermarkets_in_us_more_than_canada_l1737_173753


namespace NUMINAMATH_GPT_value_of_a2_b2_l1737_173792

theorem value_of_a2_b2 (a b : ℝ) (i : ℂ) (hi : i^2 = -1) (h : (a - i) * i = b - i) : a^2 + b^2 = 2 :=
by sorry

end NUMINAMATH_GPT_value_of_a2_b2_l1737_173792


namespace NUMINAMATH_GPT_MichelangeloCeilingPainting_l1737_173709

theorem MichelangeloCeilingPainting (total_ceiling week1_ceiling next_week_fraction : ℕ) 
  (a1 : total_ceiling = 28) 
  (a2 : week1_ceiling = 12) 
  (a3 : total_ceiling - (week1_ceiling + next_week_fraction * week1_ceiling) = 13) : 
  next_week_fraction = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_MichelangeloCeilingPainting_l1737_173709


namespace NUMINAMATH_GPT_differential_solution_correct_l1737_173786

noncomputable def y (x : ℝ) : ℝ := (x + 1)^2

theorem differential_solution_correct : 
  (∀ x : ℝ, deriv (deriv y) x = 2) ∧ y 0 = 1 ∧ (deriv y 0) = 2 := 
by
  sorry

end NUMINAMATH_GPT_differential_solution_correct_l1737_173786


namespace NUMINAMATH_GPT_remainder_problem_l1737_173774

theorem remainder_problem (n : ℕ) (h1 : n % 13 = 2) (h2 : n = 197) : 197 % 16 = 5 := by
  sorry

end NUMINAMATH_GPT_remainder_problem_l1737_173774


namespace NUMINAMATH_GPT_rhombus_side_length_l1737_173701

variable {L S : ℝ}

theorem rhombus_side_length (hL : 0 ≤ L) (hS : 0 ≤ S) :
  (∃ m : ℝ, m = 1 / 2 * Real.sqrt (L^2 - 4 * S)) :=
sorry

end NUMINAMATH_GPT_rhombus_side_length_l1737_173701


namespace NUMINAMATH_GPT_sunset_time_range_l1737_173763

theorem sunset_time_range (h : ℝ) :
  ¬(h ≥ 7) ∧ ¬(h ≤ 8) ∧ ¬(h ≤ 6) ↔ h ∈ Set.Ioi 8 :=
by
  sorry

end NUMINAMATH_GPT_sunset_time_range_l1737_173763


namespace NUMINAMATH_GPT_complement_intersection_l1737_173756

open Set

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def M : Set Int := {-1, 0, 1, 3}
def N : Set Int := {-2, 0, 2, 3}

theorem complement_intersection :
  ((U \ M) ∩ N = {-2, 2}) :=
by sorry

end NUMINAMATH_GPT_complement_intersection_l1737_173756


namespace NUMINAMATH_GPT_prime_squared_mod_six_l1737_173702

theorem prime_squared_mod_six (p : ℕ) (hp1 : p > 5) (hp2 : Nat.Prime p) : (p ^ 2) % 6 = 1 :=
sorry

end NUMINAMATH_GPT_prime_squared_mod_six_l1737_173702


namespace NUMINAMATH_GPT_area_PTR_l1737_173796

-- Define points P, Q, R, S, and T
variables (P Q R S T : Type)

-- Assume QR is divided by points S and T in the given ratio
variables (QS ST TR : ℕ)
axiom ratio_condition : QS = 2 ∧ ST = 5 ∧ TR = 3

-- Assume the area of triangle PQS is given as 60 square centimeters
axiom area_PQS : ℕ
axiom area_PQS_value : area_PQS = 60

-- State the problem
theorem area_PTR : ∃ (area_PTR : ℕ), area_PTR = 90 :=
by
  sorry

end NUMINAMATH_GPT_area_PTR_l1737_173796


namespace NUMINAMATH_GPT_abs_diff_of_two_numbers_l1737_173746

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 := 
by
  sorry

end NUMINAMATH_GPT_abs_diff_of_two_numbers_l1737_173746


namespace NUMINAMATH_GPT_part1_part2_l1737_173797

def op (a b : ℝ) : ℝ := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 :=
by
  -- sorry step to skip proof
  sorry

theorem part2 (x : ℝ) :
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- sorry step to skip proof
  sorry

end NUMINAMATH_GPT_part1_part2_l1737_173797


namespace NUMINAMATH_GPT_exists_person_who_knows_everyone_l1737_173795

variable {Person : Type}
variable (knows : Person → Person → Prop)
variable (n : ℕ)

-- Condition: In a company of 2n + 1 people, for any n people, there is another person different from them who knows each of them.
axiom knows_condition : ∀ (company : Finset Person) (h : company.card = 2 * n + 1), 
  (∀ (subset : Finset Person) (hs : subset.card = n), ∃ (p : Person), p ∉ subset ∧ ∀ q ∈ subset, knows p q)

-- Statement to be proven:
theorem exists_person_who_knows_everyone (company : Finset Person) (hcompany : company.card = 2 * n + 1) :
  ∃ p, ∀ q ∈ company, knows p q :=
sorry

end NUMINAMATH_GPT_exists_person_who_knows_everyone_l1737_173795


namespace NUMINAMATH_GPT_find_c_for_radius_6_l1737_173760

-- Define the circle equation and the radius condition.
theorem find_c_for_radius_6 (c : ℝ) :
  (∃ (x y : ℝ), x^2 + 8 * x + y^2 + 2 * y + c = 0) ∧ 6 = 6 -> c = -19 := 
by
  sorry

end NUMINAMATH_GPT_find_c_for_radius_6_l1737_173760


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1737_173700

theorem simplify_and_evaluate (a : ℤ) (h : a = -2) : 
  (1 - (1 / (a + 1))) / ((a^2 - 2*a + 1) / (a^2 - 1)) = (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1737_173700


namespace NUMINAMATH_GPT_negation_proposition_equiv_l1737_173733

variable (m : ℤ)

theorem negation_proposition_equiv :
  (¬ ∃ x : ℤ, x^2 + x + m < 0) ↔ (∀ x : ℤ, x^2 + x + m ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_equiv_l1737_173733


namespace NUMINAMATH_GPT_number_one_number_two_number_three_number_four_number_five_number_six_number_seven_number_eight_number_nine_number_ten_number_eleven_number_twelve_number_thirteen_number_fourteen_number_fifteen_number_sixteen_number_seventeen_l1737_173714

section FiveFives

def five : ℕ := 5

-- Definitions for each number 1 to 17 using five fives.
def one : ℕ := (five / five) * (five / five)
def two : ℕ := (five / five) + (five / five)
def three : ℕ := (five * five - five) / five
def four : ℕ := (five - five / five) * (five / five)
def five_num : ℕ := five + (five - five) * (five / five)
def six : ℕ := five + (five + five) / (five + five)
def seven : ℕ := five + (five * five - five^2) / five
def eight : ℕ := (five + five + five) / five + five
def nine : ℕ := five + (five - five / five)
def ten : ℕ := five + five
def eleven : ℕ := (55 - 55 / five) / five
def twelve : ℕ := five * (five - five / five) / five
def thirteen : ℕ := (five * five - five - five) / five + five
def fourteen : ℕ := five + five + five - (five / five)
def fifteen : ℕ := five + five + five
def sixteen : ℕ := five + five + five + (five / five)
def seventeen : ℕ := five + five + five + ((five / five) + (five / five))

-- Proof statements to be provided
theorem number_one : one = 1 := sorry
theorem number_two : two = 2 := sorry
theorem number_three : three = 3 := sorry
theorem number_four : four = 4 := sorry
theorem number_five : five_num = 5 := sorry
theorem number_six : six = 6 := sorry
theorem number_seven : seven = 7 := sorry
theorem number_eight : eight = 8 := sorry
theorem number_nine : nine = 9 := sorry
theorem number_ten : ten = 10 := sorry
theorem number_eleven : eleven = 11 := sorry
theorem number_twelve : twelve = 12 := sorry
theorem number_thirteen : thirteen = 13 := sorry
theorem number_fourteen : fourteen = 14 := sorry
theorem number_fifteen : fifteen = 15 := sorry
theorem number_sixteen : sixteen = 16 := sorry
theorem number_seventeen : seventeen = 17 := sorry

end FiveFives

end NUMINAMATH_GPT_number_one_number_two_number_three_number_four_number_five_number_six_number_seven_number_eight_number_nine_number_ten_number_eleven_number_twelve_number_thirteen_number_fourteen_number_fifteen_number_sixteen_number_seventeen_l1737_173714


namespace NUMINAMATH_GPT_maximum_value_l1737_173773

def expression (A B C : ℕ) : ℕ := A * B * C + A * B + B * C + C * A

theorem maximum_value (A B C : ℕ) 
  (h1 : A + B + C = 15) : 
  expression A B C ≤ 200 :=
sorry

end NUMINAMATH_GPT_maximum_value_l1737_173773


namespace NUMINAMATH_GPT_find_missing_part_l1737_173737

variable (x y : ℚ) -- Using rationals as the base field for generality.

theorem find_missing_part :
  2 * x * (-3 * x^2 * y) = -6 * x^3 * y := 
by
  sorry

end NUMINAMATH_GPT_find_missing_part_l1737_173737


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1737_173728

noncomputable def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem isosceles_triangle_perimeter {a b c : ℕ} (h1 : is_isosceles_triangle a b c) (h2 : a = 3 ∨ a = 6)
  (h3 : b = 3 ∨ b = 6) (h4 : c = 3 ∨ c = 6) (h5 : a + b + c = 15) : a + b + c = 15 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1737_173728


namespace NUMINAMATH_GPT_distance_between_homes_l1737_173745

def speed (name : String) : ℝ :=
  if name = "Maxwell" then 4
  else if name = "Brad" then 6
  else 0

def meeting_time : ℝ := 4

def delay : ℝ := 1

def distance_covered (name : String) : ℝ :=
  if name = "Maxwell" then speed name * meeting_time
  else if name = "Brad" then speed name * (meeting_time - delay)
  else 0

def total_distance : ℝ :=
  distance_covered "Maxwell" + distance_covered "Brad"

theorem distance_between_homes : total_distance = 34 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_distance_between_homes_l1737_173745


namespace NUMINAMATH_GPT_adam_earnings_per_lawn_l1737_173727

theorem adam_earnings_per_lawn (total_lawns : ℕ) (forgot_lawns : ℕ) (total_earnings : ℕ) :
  total_lawns = 12 →
  forgot_lawns = 8 →
  total_earnings = 36 →
  (total_earnings / (total_lawns - forgot_lawns)) = 9 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_adam_earnings_per_lawn_l1737_173727


namespace NUMINAMATH_GPT_translation_symmetric_y_axis_phi_l1737_173715

theorem translation_symmetric_y_axis_phi :
  ∀ (f : ℝ → ℝ) (φ : ℝ),
    (∀ x : ℝ, f x = Real.sin (2 * x + π / 6)) →
    (0 < φ ∧ φ ≤ π / 2) →
    (∀ x, Real.sin (2 * (x + φ) + π / 6) = Real.sin (2 * (-x + φ) + π / 6)) →
    φ = π / 6 :=
by
  intros f φ f_def φ_bounds symmetry
  sorry

end NUMINAMATH_GPT_translation_symmetric_y_axis_phi_l1737_173715


namespace NUMINAMATH_GPT_problem_l1737_173750

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem problem
  (ω : ℝ) 
  (hω : ω > 0)
  (hab : Real.sqrt (4 + (Real.pi ^ 2) / (ω ^ 2)) = 2 * Real.sqrt 2) :
  f ω 1 = Real.sqrt 3 / 2 := 
sorry

end NUMINAMATH_GPT_problem_l1737_173750


namespace NUMINAMATH_GPT_M_necessary_for_N_l1737_173762

def M (x : ℝ) : Prop := -1 < x ∧ x < 3
def N (x : ℝ) : Prop := 0 < x ∧ x < 3

theorem M_necessary_for_N : (∀ a : ℝ, N a → M a) ∧ (∃ b : ℝ, M b ∧ ¬N b) :=
by sorry

end NUMINAMATH_GPT_M_necessary_for_N_l1737_173762


namespace NUMINAMATH_GPT_find_a_in_geometric_sequence_l1737_173722

theorem find_a_in_geometric_sequence (S : ℕ → ℝ) (a : ℝ) :
  (∀ n, S n = 3^(n+1) + a) →
  (∃ a, ∀ n, S n = 3^(n+1) + a ∧ (18 : ℝ) ^ 2 = (S 1 - (S 1 - S 2)) * (S 2 - S 3) → a = -3) := 
by
  sorry

end NUMINAMATH_GPT_find_a_in_geometric_sequence_l1737_173722


namespace NUMINAMATH_GPT_range_of_a_l1737_173713

theorem range_of_a (h : ¬ ∃ x : ℝ, x^2 + (a-1) * x + 1 ≤ 0) : -1 < a ∧ a < 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1737_173713


namespace NUMINAMATH_GPT_sunglasses_and_cap_probability_l1737_173777

/-
On a beach:
  - 50 people are wearing sunglasses.
  - 35 people are wearing caps.
  - The probability that randomly selected person wearing a cap is also wearing sunglasses is 2/5.
  
Prove that the probability that a randomly selected person wearing sunglasses is also wearing a cap is 7/25.
-/

theorem sunglasses_and_cap_probability :
  let total_sunglasses := 50
  let total_caps := 35
  let cap_with_sunglasses_probability := (2 : ℚ) / 5
  let both := cap_with_sunglasses_probability * total_caps
  (both / total_sunglasses) = (7 : ℚ) / 25 :=
by
  -- definitions
  let total_sunglasses := 50
  let total_caps := 35
  let cap_with_sunglasses_probability := (2 : ℚ) / 5
  let both := cap_with_sunglasses_probability * (total_caps : ℚ)
  have prob : (both / (total_sunglasses : ℚ)) = (7 : ℚ) / 25 := sorry
  exact prob

end NUMINAMATH_GPT_sunglasses_and_cap_probability_l1737_173777


namespace NUMINAMATH_GPT_max_profit_at_one_device_l1737_173781

noncomputable def revenue (x : ℕ) : ℝ := 30 * x - 0.2 * x^2

def fixed_monthly_cost : ℝ := 40

def material_cost_per_device : ℝ := 5

noncomputable def cost (x : ℕ) : ℝ := fixed_monthly_cost + material_cost_per_device * x

noncomputable def profit_function (x : ℕ) : ℝ := (revenue x) - (cost x)

noncomputable def marginal_profit_function (x : ℕ) : ℝ :=
  profit_function (x + 1) - profit_function x

theorem max_profit_at_one_device :
  marginal_profit_function 1 = 24.4 ∧
  ∀ x : ℕ, marginal_profit_function x ≤ 24.4 := sorry

end NUMINAMATH_GPT_max_profit_at_one_device_l1737_173781


namespace NUMINAMATH_GPT_ratio_of_men_to_women_l1737_173739

theorem ratio_of_men_to_women (C W M : ℕ) 
  (hC : C = 30) 
  (hW : W = 3 * C) 
  (hTotal : M + W + C = 300) : 
  M / W = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_men_to_women_l1737_173739


namespace NUMINAMATH_GPT_min_value_expression_l1737_173779

-- Let x and y be positive integers such that x^2 + y^2 - 2017 * x * y > 0 and it is not a perfect square.
theorem min_value_expression (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_not_square : ¬ ∃ z : ℕ, (x^2 + y^2 - 2017 * x * y) = z^2) :
  x^2 + y^2 - 2017 * x * y > 0 → ∃ k : ℕ, k = 2019 ∧ ∀ m : ℕ, (m > 0 → ¬ ∃ z : ℤ, (x^2 + y^2 - 2017 * x * y) = z^2 ∧ x^2 + y^2 - 2017 * x * y < k) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1737_173779


namespace NUMINAMATH_GPT_range_of_m_l1737_173717

/-- The point (m^2, m) is within the planar region defined by x - 3y + 2 > 0. 
    Find the range of m. -/
theorem range_of_m {m : ℝ} : (m^2 - 3 * m + 2 > 0) ↔ (m < 1 ∨ m > 2) := 
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l1737_173717


namespace NUMINAMATH_GPT_man_twice_son_age_in_years_l1737_173787

theorem man_twice_son_age_in_years :
  ∀ (S M Y : ℕ),
  (M = S + 26) →
  (S = 24) →
  (M + Y = 2 * (S + Y)) →
  Y = 2 :=
by
  intros S M Y h1 h2 h3
  sorry

end NUMINAMATH_GPT_man_twice_son_age_in_years_l1737_173787


namespace NUMINAMATH_GPT_units_digit_fraction_l1737_173751

theorem units_digit_fraction :
  (30 * 31 * 32 * 33 * 34 * 35) % 10 = (2500) % 10 → 
  ((30 * 31 * 32 * 33 * 34 * 35) / 2500) % 10 = 1 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_units_digit_fraction_l1737_173751


namespace NUMINAMATH_GPT_physics_class_size_l1737_173772

theorem physics_class_size (total_students physics_only math_only both : ℕ) 
  (h1 : total_students = 100)
  (h2 : physics_only + math_only + both = total_students)
  (h3 : both = 10)
  (h4 : physics_only + both = 2 * (math_only + both)) :
  physics_only + both = 62 := 
by sorry

end NUMINAMATH_GPT_physics_class_size_l1737_173772


namespace NUMINAMATH_GPT_certain_number_l1737_173735

theorem certain_number (x : ℝ) (h : 4 * x = 200) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_l1737_173735


namespace NUMINAMATH_GPT_measure_of_angle_C_l1737_173765

theorem measure_of_angle_C (a b area : ℝ) (C : ℝ) :
  a = 5 → b = 8 → area = 10 →
  (1 / 2 * a * b * Real.sin C = area) →
  (C = Real.pi / 6 ∨ C = 5 * Real.pi / 6) := by
  intros ha hb harea hformula
  sorry

end NUMINAMATH_GPT_measure_of_angle_C_l1737_173765


namespace NUMINAMATH_GPT_solve_eq_l1737_173704

-- Defining the condition
def eq_condition (x : ℝ) : Prop := (x - 3) ^ 2 = x ^ 2 - 9

-- The statement we need to prove
theorem solve_eq (x : ℝ) (h : eq_condition x) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_eq_l1737_173704


namespace NUMINAMATH_GPT_certain_event_l1737_173747

-- Definitions of the events
def event1 : Prop := ∀ (P : ℝ), P ≠ 20.0
def event2 : Prop := ∀ (x : ℤ), x ≠ 105 ∧ x ≤ 100
def event3 : Prop := ∃ (r : ℝ), 0 ≤ r ∧ r ≤ 1 ∧ ¬(r = 0 ∨ r = 1)
def event4 (a b : ℝ) : Prop := ∃ (area : ℝ), area = a * b

-- Statement to prove that event4 is the only certain event
theorem certain_event (a b : ℝ) : (event4 a b) := 
by
  sorry

end NUMINAMATH_GPT_certain_event_l1737_173747


namespace NUMINAMATH_GPT_sum_of_midpoints_eq_15_l1737_173749

theorem sum_of_midpoints_eq_15 (a b c d : ℝ) (h : a + b + c + d = 15) :
  (a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2 = 15 :=
by sorry

end NUMINAMATH_GPT_sum_of_midpoints_eq_15_l1737_173749


namespace NUMINAMATH_GPT_find_missing_number_l1737_173740

theorem find_missing_number 
  (x : ℝ) (y : ℝ)
  (h1 : (12 + x + 42 + 78 + 104) / 5 = 62)
  (h2 : (128 + 255 + y + 1023 + x) / 5 = 398.2) :
  y = 511 := 
sorry

end NUMINAMATH_GPT_find_missing_number_l1737_173740


namespace NUMINAMATH_GPT_total_fence_used_l1737_173782

-- Definitions based on conditions
variables {L W : ℕ}
def area (L W : ℕ) := L * W

-- Provided conditions as Lean definitions
def unfenced_side := 40
def yard_area := 240

-- The proof problem statement
theorem total_fence_used (L_eq : L = unfenced_side) (A_eq : area L W = yard_area) : (2 * W + L) = 52 :=
sorry

end NUMINAMATH_GPT_total_fence_used_l1737_173782


namespace NUMINAMATH_GPT_friends_for_picnic_only_l1737_173738

theorem friends_for_picnic_only (M MP MG G PG A P : ℕ) 
(h1 : M + MP + MG + A = 10)
(h2 : G + MG + A = 5)
(h3 : MP = 4)
(h4 : MG = 2)
(h5 : PG = 0)
(h6 : A = 2)
(h7 : M + P + G + MP + MG + PG + A = 31) : 
    P = 20 := by {
  sorry
}

end NUMINAMATH_GPT_friends_for_picnic_only_l1737_173738


namespace NUMINAMATH_GPT_expression_result_l1737_173755

-- We define the mixed number fractions as conditions
def mixed_num_1 := 2 + 1 / 2         -- 2 1/2
def mixed_num_2 := 3 + 1 / 3         -- 3 1/3
def mixed_num_3 := 4 + 1 / 4         -- 4 1/4
def mixed_num_4 := 1 + 1 / 6         -- 1 1/6

-- Here are their improper fractions
def improper_fraction_1 := 5 / 2     -- (2 + 1/2) converted to improper fraction
def improper_fraction_2 := 10 / 3    -- (3 + 1/3) converted to improper fraction
def improper_fraction_3 := 17 / 4    -- (4 + 1/4) converted to improper fraction
def improper_fraction_4 := 7 / 6     -- (1 + 1/6) converted to improper fraction

-- Define the problematic expression
def expression := (improper_fraction_1 - improper_fraction_2)^2 / (improper_fraction_3 + improper_fraction_4)

-- Statement of the simplified result
theorem expression_result : expression = 5 / 39 :=
by
  sorry

end NUMINAMATH_GPT_expression_result_l1737_173755


namespace NUMINAMATH_GPT_find_f_of_half_l1737_173734

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_half : (∀ x : ℝ, f (Real.logb 4 x) = x) → f (1 / 2) = 2 :=
by
  intros h
  have h1 := h (4 ^ (1 / 2))
  sorry

end NUMINAMATH_GPT_find_f_of_half_l1737_173734


namespace NUMINAMATH_GPT_magnitude_of_complex_l1737_173785

open Complex

theorem magnitude_of_complex : abs (Complex.mk (3/4) (-5/6)) = Real.sqrt (181) / 12 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_of_complex_l1737_173785


namespace NUMINAMATH_GPT_base9_first_digit_is_4_l1737_173770

-- Define the base three representation of y
def y_base3 : Nat := 112211

-- Function to convert a given number from base 3 to base 10
def base3_to_base10 (n : Nat) : Nat :=
  let rec convert (n : Nat) (acc : Nat) (place : Nat) : Nat :=
    if n = 0 then acc
    else convert (n / 10) (acc + (n % 10) * (3 ^ place)) (place + 1)
  convert n 0 0

-- Compute the base 10 representation of y
def y_base10 : Nat := base3_to_base10 y_base3

-- Function to convert a given number from base 10 to base 9
def base10_to_base9 (n : Nat) : List Nat :=
  let rec convert (n : Nat) (acc : List Nat) : List Nat :=
    if n = 0 then acc
    else convert (n / 9) ((n % 9) :: acc)
  convert n []

-- Compute the base 9 representation of y as a list of digits
def y_base9 : List Nat := base10_to_base9 y_base10

-- Get the first digit (most significant digit) of the base 9 representation of y
def first_digit_base9 (digits : List Nat) : Nat :=
  digits.headD 0

-- The statement to prove
theorem base9_first_digit_is_4 : first_digit_base9 y_base9 = 4 := by sorry

end NUMINAMATH_GPT_base9_first_digit_is_4_l1737_173770


namespace NUMINAMATH_GPT_problem_l1737_173788

-- Define the conditions
variables (x y : ℝ)
axiom h1 : 2 * x + y = 7
axiom h2 : x + 2 * y = 5

-- Statement of the problem
theorem problem : (2 * x * y) / 3 = 2 :=
by 
  -- Proof is omitted, but you should replace 'sorry' by the actual proof
  sorry

end NUMINAMATH_GPT_problem_l1737_173788


namespace NUMINAMATH_GPT_find_number_l1737_173724

variable (x : ℝ)

theorem find_number : ((x * 5) / 2.5 - 8 * 2.25 = 5.5) -> x = 11.75 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_number_l1737_173724


namespace NUMINAMATH_GPT_sqrt_17_estimation_l1737_173780

theorem sqrt_17_estimation :
  4 < Real.sqrt 17 ∧ Real.sqrt 17 < 5 := 
sorry

end NUMINAMATH_GPT_sqrt_17_estimation_l1737_173780


namespace NUMINAMATH_GPT_total_students_in_class_l1737_173775

variable (K M Both Total : ℕ)

theorem total_students_in_class
  (hK : K = 38)
  (hM : M = 39)
  (hBoth : Both = 32)
  (hTotal : Total = K + M - Both) :
  Total = 45 := 
by
  rw [hK, hM, hBoth] at hTotal
  exact hTotal

end NUMINAMATH_GPT_total_students_in_class_l1737_173775


namespace NUMINAMATH_GPT_g_is_odd_l1737_173783

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) - (1 / 2)

theorem g_is_odd (x : ℝ) : g (-x) = -g x :=
by sorry

end NUMINAMATH_GPT_g_is_odd_l1737_173783


namespace NUMINAMATH_GPT_lacy_correct_percentage_is_80_l1737_173729

-- Define the total number of problems
def total_problems (x : ℕ) : ℕ := 5 * x + 10

-- Define the number of problems Lacy missed
def problems_missed (x : ℕ) : ℕ := x + 2

-- Define the number of problems Lacy answered correctly
def problems_answered (x : ℕ) : ℕ := total_problems x - problems_missed x

-- Define the fraction of problems Lacy answered correctly
def fraction_answered_correctly (x : ℕ) : ℚ :=
  (problems_answered x : ℚ) / (total_problems x : ℚ)

-- The main theorem to prove the percentage of problems correctly answered is 80%
theorem lacy_correct_percentage_is_80 (x : ℕ) : 
  fraction_answered_correctly x = 4 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_lacy_correct_percentage_is_80_l1737_173729


namespace NUMINAMATH_GPT_smallest_sum_a_b_l1737_173744

theorem smallest_sum_a_b :
  ∃ (a b : ℕ), (7 * b - 4 * a = 3) ∧ a > 7 ∧ b > 7 ∧ a + b = 24 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_a_b_l1737_173744


namespace NUMINAMATH_GPT_fish_population_estimation_l1737_173769

theorem fish_population_estimation (N : ℕ) (h1 : 80 ≤ N)
  (h_tagged_returned : true)
  (h_second_catch : 80 ≤ N)
  (h_tagged_in_second_catch : 2 = 80 * 80 / N) :
  N = 3200 :=
by
  sorry

end NUMINAMATH_GPT_fish_population_estimation_l1737_173769


namespace NUMINAMATH_GPT_correlation_relationships_l1737_173768

-- Let's define the relationships as conditions
def volume_cube_edge_length (v e : ℝ) : Prop := v = e^3
def yield_fertilizer (yield fertilizer : ℝ) : Prop := True -- Assume linear correlation within a certain range
def height_age (height age : ℝ) : Prop := True -- Assume linear correlation within a certain age range
def expenses_income (expenses income : ℝ) : Prop := True -- Assume linear correlation
def electricity_consumption_price (consumption price unit_price : ℝ) : Prop := price = consumption * unit_price

-- We want to prove that the answers correspond correctly to the conditions:
theorem correlation_relationships :
  ∀ (v e yield fertilizer height age expenses income consumption price unit_price : ℝ),
  ¬ volume_cube_edge_length v e ∧ yield_fertilizer yield fertilizer ∧ height_age height age ∧ expenses_income expenses income ∧ ¬ electricity_consumption_price consumption price unit_price → 
  "D" = "②③④" :=
by
  intros
  sorry

end NUMINAMATH_GPT_correlation_relationships_l1737_173768


namespace NUMINAMATH_GPT_log8_512_is_3_l1737_173743

def log_base_8_of_512 : Prop :=
  ∀ (log8 : ℝ → ℝ),
    (log8 8 = 1 / 3 * log8 2) →
    (log8 512 = 9 * log8 2) →
    log8 8 = 3 → log8 512 = 3

theorem log8_512_is_3 : log_base_8_of_512 :=
by
  intros log8 H1 H2 H3
  -- here you would normally provide the detailed steps to solve this.
  -- however, we directly proclaim the result due to the proof being non-trivial.
  sorry

end NUMINAMATH_GPT_log8_512_is_3_l1737_173743


namespace NUMINAMATH_GPT_carol_twice_as_cathy_l1737_173711

-- Define variables for the number of cars each person owns
variables (C L S Ca x : ℕ)

-- Define conditions based on the problem statement
def lindsey_cars := L = C + 4
def susan_cars := S = Ca - 2
def carol_cars := Ca = 2 * x
def total_cars := C + L + S + Ca = 32
def cathy_cars := C = 5

-- State the theorem to prove
theorem carol_twice_as_cathy : 
  lindsey_cars C L ∧ 
  susan_cars S Ca ∧ 
  carol_cars Ca x ∧ 
  total_cars C L S Ca ∧ 
  cathy_cars C
  → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_carol_twice_as_cathy_l1737_173711


namespace NUMINAMATH_GPT_fraction_equality_l1737_173798

theorem fraction_equality (a b : ℝ) (h : a / b = 2) : a / (a - b) = 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l1737_173798


namespace NUMINAMATH_GPT_books_more_than_movies_l1737_173784

-- Define the number of movies and books in the "crazy silly school" series.
def num_movies : ℕ := 14
def num_books : ℕ := 15

-- State the theorem to prove there is 1 more book than movies.
theorem books_more_than_movies : num_books - num_movies = 1 :=
by 
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_books_more_than_movies_l1737_173784


namespace NUMINAMATH_GPT_positiveDifferenceEquation_l1737_173705

noncomputable def positiveDifference (x y : ℝ) : ℝ := |y - x|

theorem positiveDifferenceEquation (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  positiveDifference x y = 60 / 7 :=
by
  sorry

end NUMINAMATH_GPT_positiveDifferenceEquation_l1737_173705


namespace NUMINAMATH_GPT_binary_10101_to_decimal_l1737_173799

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.zipWith (λ digit idx => digit * 2^idx) (List.range b.length) |>.sum

theorem binary_10101_to_decimal : binary_to_decimal [1, 0, 1, 0, 1] = 21 := by
  sorry

end NUMINAMATH_GPT_binary_10101_to_decimal_l1737_173799


namespace NUMINAMATH_GPT_turtles_on_Happy_Island_l1737_173794

theorem turtles_on_Happy_Island (L H : ℕ) (hL : L = 25) (hH : H = 2 * L + 10) : H = 60 :=
by
  sorry

end NUMINAMATH_GPT_turtles_on_Happy_Island_l1737_173794


namespace NUMINAMATH_GPT_men_l1737_173730

namespace WagesProblem

def men_women_boys_equivalence (man woman boy : ℕ) : Prop :=
  9 * man = woman ∧ woman = 7 * boy

def total_earnings (man woman boy earnings : ℕ) : Prop :=
  (9 * man + woman + woman) = earnings ∧ earnings = 216

theorem men's_wages (man woman boy : ℕ) (h1 : men_women_boys_equivalence man woman boy) (h2 : total_earnings man woman 7 216) : 9 * man = 72 :=
sorry

end WagesProblem

end NUMINAMATH_GPT_men_l1737_173730


namespace NUMINAMATH_GPT_true_discount_is_36_l1737_173706

noncomputable def calc_true_discount (BD SD : ℝ) : ℝ := BD / (1 + BD / SD)

theorem true_discount_is_36 :
  let BD := 42
  let SD := 252
  calc_true_discount BD SD = 36 := 
by
  -- proof here
  sorry

end NUMINAMATH_GPT_true_discount_is_36_l1737_173706


namespace NUMINAMATH_GPT_find_solutions_l1737_173718

theorem find_solutions (x y : Real) :
    (x = 1 ∧ y = 2) ∨
    (x = 1 ∧ y = 0) ∨
    (x = -4 ∧ y = 6) ∨
    (x = -5 ∧ y = 2) ∨
    (x = -3 ∧ y = 0) ↔
    x^2 + x*y + y^2 + 2*x - 3*y - 3 = 0 := by
  sorry

end NUMINAMATH_GPT_find_solutions_l1737_173718
