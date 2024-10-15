import Mathlib

namespace NUMINAMATH_GPT_kids_all_three_activities_l1234_123437

-- Definitions based on conditions
def total_kids : ℕ := 40
def kids_tubing : ℕ := total_kids / 4
def kids_tubing_rafting : ℕ := kids_tubing / 2
def kids_tubing_rafting_kayaking : ℕ := kids_tubing_rafting / 3

-- Theorem statement: proof of the final answer
theorem kids_all_three_activities : kids_tubing_rafting_kayaking = 1 := by
  sorry

end NUMINAMATH_GPT_kids_all_three_activities_l1234_123437


namespace NUMINAMATH_GPT_omega_value_l1234_123458

noncomputable def f (ω : ℝ) (k : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x - Real.pi / 6) + k

theorem omega_value (ω k : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, f ω k x ≤ f ω k (Real.pi / 3)) → ω = 8 :=
by sorry

end NUMINAMATH_GPT_omega_value_l1234_123458


namespace NUMINAMATH_GPT_seashells_total_l1234_123443

theorem seashells_total :
    let Sam := 35
    let Joan := 18
    let Alex := 27
    Sam + Joan + Alex = 80 :=
by
    sorry

end NUMINAMATH_GPT_seashells_total_l1234_123443


namespace NUMINAMATH_GPT_domain_of_function_l1234_123421

/-- Prove the domain of the function f(x) = log10(2 * cos x - 1) + sqrt(49 - x^2) -/
theorem domain_of_function :
  { x : ℝ | -7 ≤ x ∧ x < - (5 * Real.pi) / 3 ∨ - Real.pi / 3 < x ∧ x < Real.pi / 3 ∨ (5 * Real.pi) / 3 < x ∧ x ≤ 7 }
  = { x : ℝ | 2 * Real.cos x - 1 > 0 ∧ 49 - x^2 ≥ 0 } :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_of_function_l1234_123421


namespace NUMINAMATH_GPT_common_difference_is_3_l1234_123457

variable {a : ℕ → ℤ} {d : ℤ}

-- Definitions of conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition_1 (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 3 + a 11 = 24

def condition_2 (a : ℕ → ℤ) : Prop :=
  a 4 = 3

-- Theorem statement to prove
theorem common_difference_is_3 (a : ℕ → ℤ) (d : ℤ)
  (ha : is_arithmetic_sequence a d)
  (hc1 : condition_1 a d)
  (hc2 : condition_2 a) :
  d = 3 := by
  sorry

end NUMINAMATH_GPT_common_difference_is_3_l1234_123457


namespace NUMINAMATH_GPT_Ann_end_blocks_l1234_123465

-- Define blocks Ann initially has and finds
def initialBlocksAnn : ℕ := 9
def foundBlocksAnn : ℕ := 44

-- Define blocks Ann ends with
def finalBlocksAnn : ℕ := initialBlocksAnn + foundBlocksAnn

-- The proof goal
theorem Ann_end_blocks : finalBlocksAnn = 53 := by
  -- Use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_Ann_end_blocks_l1234_123465


namespace NUMINAMATH_GPT_total_difference_proof_l1234_123452

-- Definitions for the initial quantities
def initial_tomatoes : ℕ := 17
def initial_carrots : ℕ := 13
def initial_cucumbers : ℕ := 8

-- Definitions for the picked quantities
def picked_tomatoes : ℕ := 5
def picked_carrots : ℕ := 6

-- Definitions for the given away quantities
def given_away_tomatoes : ℕ := 3
def given_away_carrots : ℕ := 2

-- Definitions for the remaining quantities 
def remaining_tomatoes : ℕ := initial_tomatoes - (picked_tomatoes - given_away_tomatoes)
def remaining_carrots : ℕ := initial_carrots - (picked_carrots - given_away_carrots)

-- Definitions for the difference quantities
def difference_tomatoes : ℕ := initial_tomatoes - remaining_tomatoes
def difference_carrots : ℕ := initial_carrots - remaining_carrots

-- Definition for the total difference
def total_difference : ℕ := difference_tomatoes + difference_carrots

-- Lean Theorem Statement
theorem total_difference_proof : total_difference = 6 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_difference_proof_l1234_123452


namespace NUMINAMATH_GPT_rods_in_one_mile_l1234_123409

theorem rods_in_one_mile (chains_in_mile : ℕ) (rods_in_chain : ℕ) (mile_to_chain : 1 = 10 * chains_in_mile) (chain_to_rod : 1 = 22 * rods_in_chain) :
  1 * 220 = 10 * 22 :=
by sorry

end NUMINAMATH_GPT_rods_in_one_mile_l1234_123409


namespace NUMINAMATH_GPT_radius_of_smaller_circle_l1234_123436

theorem radius_of_smaller_circle (R : ℝ) (n : ℕ) (r : ℝ) 
  (hR : R = 10) 
  (hn : n = 7) 
  (condition : 2 * R = 2 * r * n) :
  r = 10 / 7 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_smaller_circle_l1234_123436


namespace NUMINAMATH_GPT_nth_odd_and_sum_first_n_odds_l1234_123467

noncomputable def nth_odd (n : ℕ) : ℕ := 2 * n - 1

noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n ^ 2

theorem nth_odd_and_sum_first_n_odds :
  nth_odd 100 = 199 ∧ sum_first_n_odds 100 = 10000 :=
by
  sorry

end NUMINAMATH_GPT_nth_odd_and_sum_first_n_odds_l1234_123467


namespace NUMINAMATH_GPT_solution_set_empty_range_l1234_123481

theorem solution_set_empty_range (a : ℝ) : 
  (∀ x : ℝ, ax^2 + ax + 3 < 0 → false) ↔ (0 ≤ a ∧ a ≤ 12) := 
sorry

end NUMINAMATH_GPT_solution_set_empty_range_l1234_123481


namespace NUMINAMATH_GPT_selection_schemes_l1234_123425

theorem selection_schemes (people : Finset ℕ) (A B C : ℕ) (h_people : people.card = 5) 
(h_A_B_individuals : A ∈ people ∧ B ∈ people) (h_A_B_C_exclusion : A ≠ C ∧ B ≠ C) :
  ∃ (number_of_schemes : ℕ), number_of_schemes = 36 :=
by
  sorry

end NUMINAMATH_GPT_selection_schemes_l1234_123425


namespace NUMINAMATH_GPT_a3_value_l1234_123420

theorem a3_value (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) (x : ℝ) :
  ( (1 + x) * (a - x) ^ 6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 ) →
  ( a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0 ) →
  a = 1 →
  a₃ = -5 :=
by
  sorry

end NUMINAMATH_GPT_a3_value_l1234_123420


namespace NUMINAMATH_GPT_arrange_in_ascending_order_l1234_123416

open Real

noncomputable def a := log 3 / log (1/2)
noncomputable def b := log 5 / log (1/2)
noncomputable def c := log (1/2) / log (1/3)

theorem arrange_in_ascending_order : b < a ∧ a < c :=
by
  sorry

end NUMINAMATH_GPT_arrange_in_ascending_order_l1234_123416


namespace NUMINAMATH_GPT_closure_of_A_range_of_a_l1234_123463

-- Definitions for sets A and B
def A (x : ℝ) : Prop := x < -1 ∨ x > -0.5
def B (x a : ℝ) : Prop := a - 1 ≤ x ∧ x ≤ a + 1

-- 1. Closure of A
theorem closure_of_A :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ -0.5) ↔ (∀ x : ℝ, A x) :=
sorry

-- 2. Range of a when A ∪ B = ℝ
theorem range_of_a (B_condition : ∀ x : ℝ, B x a) :
  (∀ a : ℝ, -1 ≤ x ∨ x ≥ -0.5) ↔ (-1.5 ≤ a ∧ a ≤ 0) :=
sorry

end NUMINAMATH_GPT_closure_of_A_range_of_a_l1234_123463


namespace NUMINAMATH_GPT_average_weight_increase_l1234_123487

theorem average_weight_increase (A : ℝ) :
  let initial_total_weight := 10 * A
  let new_total_weight := initial_total_weight - 65 + 97
  let new_average := new_total_weight / 10
  let increase := new_average - A
  increase = 3.2 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_increase_l1234_123487


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1234_123477

-- Define the given nested fraction expression
def nested_expr := 1 + 1 / (1 + 1 / (1 + 1))

-- Define the simplified form of the expression
def simplified_form : ℚ := 13 / 8

-- The greatest common divisor condition
def gcd_condition : ℕ := Nat.gcd 13 8

-- The ultimate theorem to prove
theorem value_of_a_plus_b : 
  nested_expr = simplified_form ∧ gcd_condition = 1 → 13 + 8 = 21 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1234_123477


namespace NUMINAMATH_GPT_three_point_one_two_six_as_fraction_l1234_123471

theorem three_point_one_two_six_as_fraction : (3126 / 1000 : ℚ) = 1563 / 500 := 
by 
  sorry

end NUMINAMATH_GPT_three_point_one_two_six_as_fraction_l1234_123471


namespace NUMINAMATH_GPT_geometric_product_is_geometric_l1234_123489

theorem geometric_product_is_geometric (q : ℝ) (a : ℕ → ℝ)
  (h_geo : ∀ n, a (n + 1) = q * a n) :
  ∀ n, (a n) * (a (n + 1)) = (q^2) * (a (n - 1) * a n) := by
  sorry

end NUMINAMATH_GPT_geometric_product_is_geometric_l1234_123489


namespace NUMINAMATH_GPT_toby_total_time_l1234_123490

theorem toby_total_time (d1 d2 d3 d4 : ℕ)
  (speed_loaded speed_unloaded : ℕ)
  (time1 time2 time3 time4 total_time : ℕ)
  (h1 : d1 = 180)
  (h2 : d2 = 120)
  (h3 : d3 = 80)
  (h4 : d4 = 140)
  (h5 : speed_loaded = 10)
  (h6 : speed_unloaded = 20)
  (h7 : time1 = d1 / speed_loaded)
  (h8 : time2 = d2 / speed_unloaded)
  (h9 : time3 = d3 / speed_loaded)
  (h10 : time4 = d4 / speed_unloaded)
  (h11 : total_time = time1 + time2 + time3 + time4) :
  total_time = 39 := by
  sorry

end NUMINAMATH_GPT_toby_total_time_l1234_123490


namespace NUMINAMATH_GPT_travel_time_K_l1234_123404

theorem travel_time_K (d x : ℝ) (h_pos_d : d > 0) (h_x_pos : x > 0) (h_time_diff : (d / (x - 1/2)) - (d / x) = 1/2) : d / x = 40 / x :=
by
  sorry

end NUMINAMATH_GPT_travel_time_K_l1234_123404


namespace NUMINAMATH_GPT_rainfall_on_Monday_l1234_123453

theorem rainfall_on_Monday (rain_on_Tuesday : ℝ) (difference : ℝ) (rain_on_Tuesday_eq : rain_on_Tuesday = 0.2) (difference_eq : difference = 0.7) :
  ∃ rain_on_Monday : ℝ, rain_on_Monday = rain_on_Tuesday + difference := 
sorry

end NUMINAMATH_GPT_rainfall_on_Monday_l1234_123453


namespace NUMINAMATH_GPT_gcd_of_factors_l1234_123476

theorem gcd_of_factors (a b : ℕ) (h : a * b = 360) : 
    ∃ n : ℕ, n = 19 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_factors_l1234_123476


namespace NUMINAMATH_GPT_solution_set_l1234_123400

-- Define the conditions
variables {f : ℝ → ℝ}

-- Condition: f(x) is an odd function
axiom odd_function : ∀ x : ℝ, f (-x) = -f x

-- Condition: xf'(x) + f(x) < 0 for x in (-∞, 0)
axiom condition1 : ∀ x : ℝ, x < 0 → x * (deriv f x) + f x < 0

-- Condition: f(-2) = 0
axiom f_neg2_zero : f (-2) = 0

-- Goal: Prove the solution set of the inequality xf(x) < 0 is {x | -2 < x < 0 ∨ 0 < x < 2}
theorem solution_set : ∀ x : ℝ, (x * f x < 0) ↔ (-2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_GPT_solution_set_l1234_123400


namespace NUMINAMATH_GPT_inequality_solution_l1234_123469

theorem inequality_solution (x : ℝ) : 
  (2*x - 1) / (x - 3) ≥ 1 ↔ (x > 3 ∨ x ≤ -2) :=
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_l1234_123469


namespace NUMINAMATH_GPT_find_days_l1234_123480

theorem find_days
  (wages1 : ℕ) (workers1 : ℕ) (days1 : ℕ)
  (wages2 : ℕ) (workers2 : ℕ) (days2 : ℕ)
  (h1 : wages1 = 9450) (h2 : workers1 = 15) (h3 : wages2 = 9975)
  (h4 : workers2 = 19) (h5 : days2 = 5) :
  days1 = 6 := 
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_find_days_l1234_123480


namespace NUMINAMATH_GPT_average_marks_all_students_l1234_123493

theorem average_marks_all_students
  (n1 n2 : ℕ)
  (avg1 avg2 : ℕ)
  (h1 : avg1 = 40)
  (h2 : avg2 = 80)
  (h3 : n1 = 30)
  (h4 : n2 = 50) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 65 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_all_students_l1234_123493


namespace NUMINAMATH_GPT_earnings_per_widget_l1234_123438

-- Defining the conditions as constants
def hours_per_week : ℝ := 40
def hourly_wage : ℝ := 12.50
def total_weekly_earnings : ℝ := 700
def widgets_produced : ℝ := 1250

-- We need to prove earnings per widget
theorem earnings_per_widget :
  (total_weekly_earnings - (hours_per_week * hourly_wage)) / widgets_produced = 0.16 := by
  sorry

end NUMINAMATH_GPT_earnings_per_widget_l1234_123438


namespace NUMINAMATH_GPT_necessarily_positive_b_plus_3c_l1234_123427

theorem necessarily_positive_b_plus_3c 
  (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + 3 * c > 0 := 
sorry

end NUMINAMATH_GPT_necessarily_positive_b_plus_3c_l1234_123427


namespace NUMINAMATH_GPT_muffs_bought_before_december_correct_l1234_123449

/-- Total ear muffs bought by customers in December. -/
def muffs_bought_in_december := 6444

/-- Total ear muffs bought by customers in all. -/
def total_muffs_bought := 7790

/-- Ear muffs bought before December. -/
def muffs_bought_before_december : Nat :=
  total_muffs_bought - muffs_bought_in_december

/-- Theorem stating the number of ear muffs bought before December. -/
theorem muffs_bought_before_december_correct :
  muffs_bought_before_december = 1346 :=
by
  unfold muffs_bought_before_december
  unfold total_muffs_bought
  unfold muffs_bought_in_december
  sorry

end NUMINAMATH_GPT_muffs_bought_before_december_correct_l1234_123449


namespace NUMINAMATH_GPT_total_amount_shared_l1234_123423

theorem total_amount_shared (A B C : ℕ) (h1 : A = 24) (h2 : 2 * A = 3 * B) (h3 : 8 * A = 4 * C) :
  A + B + C = 156 :=
sorry

end NUMINAMATH_GPT_total_amount_shared_l1234_123423


namespace NUMINAMATH_GPT_factorization_correct_l1234_123445

theorem factorization_correct (a b : ℝ) : 6 * a * b - a^2 - 9 * b^2 = -(a - 3 * b)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1234_123445


namespace NUMINAMATH_GPT_complex_pow_imaginary_unit_l1234_123428

theorem complex_pow_imaginary_unit (i : ℂ) (h : i^2 = -1) : i^2015 = -i :=
sorry

end NUMINAMATH_GPT_complex_pow_imaginary_unit_l1234_123428


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l1234_123447

noncomputable def eccentricity_asymptotes (a b : ℝ) (h₁ : a > 0) (h₂ : b = Real.sqrt 15 * a) : Prop :=
  ∀ (x y : ℝ), (y = (Real.sqrt 15) * x) ∨ (y = -(Real.sqrt 15) * x)

theorem hyperbola_asymptotes (a : ℝ) (h₁ : a > 0) :
  eccentricity_asymptotes a (Real.sqrt 15 * a) h₁ (by simp) :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l1234_123447


namespace NUMINAMATH_GPT_square_area_correct_l1234_123405

-- Define the length of the side of the square
def side_length : ℕ := 15

-- Define the area calculation for a square
def square_area (side : ℕ) : ℕ := side * side

-- Define the area calculation for a triangle using the square area division
def triangle_area (square_area : ℕ) : ℕ := square_area / 2

-- Theorem stating that the area of a square with given side length is 225 square units
theorem square_area_correct : square_area side_length = 225 := by
  sorry

end NUMINAMATH_GPT_square_area_correct_l1234_123405


namespace NUMINAMATH_GPT_horizontal_asymptote_l1234_123475

def numerator (x : ℝ) : ℝ :=
  15 * x^4 + 3 * x^3 + 7 * x^2 + 6 * x + 2

def denominator (x : ℝ) : ℝ :=
  5 * x^4 + x^3 + 4 * x^2 + 2 * x + 1

noncomputable def rational_function (x : ℝ) : ℝ :=
  numerator x / denominator x

theorem horizontal_asymptote :
  ∃ y : ℝ, (∀ x : ℝ, x ≠ 0 → rational_function x = y) ↔ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_horizontal_asymptote_l1234_123475


namespace NUMINAMATH_GPT_petya_wins_with_optimal_play_l1234_123448

theorem petya_wins_with_optimal_play :
  ∃ (n m : ℕ), n = 2000 ∧ m = (n * (n - 1)) / 2 ∧
  (∀ (v_cut : ℕ), ∀ (p_cut : ℕ), v_cut = 1 ∧ (p_cut = 2 ∨ p_cut = 3) ∧
  ((∃ k, m - v_cut = 4 * k) → ∃ k, m - v_cut - p_cut = 4 * k + 1) → 
  ∃ k, m - p_cut = 4 * k + 3) :=
sorry

end NUMINAMATH_GPT_petya_wins_with_optimal_play_l1234_123448


namespace NUMINAMATH_GPT_minute_hand_length_l1234_123414

theorem minute_hand_length (r : ℝ) (h : 20 * (2 * Real.pi / 60) * r = Real.pi / 3) : r = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minute_hand_length_l1234_123414


namespace NUMINAMATH_GPT_megatek_manufacturing_percentage_l1234_123491

-- Define the given conditions
def sector_deg : ℝ := 18
def full_circle_deg : ℝ := 360

-- Define the problem as a theorem statement in Lean
theorem megatek_manufacturing_percentage : 
  (sector_deg / full_circle_deg) * 100 = 5 := 
sorry

end NUMINAMATH_GPT_megatek_manufacturing_percentage_l1234_123491


namespace NUMINAMATH_GPT_minimize_surface_area_l1234_123415

theorem minimize_surface_area (V r h : ℝ) (hV : V = π * r^2 * h) (hA : 2 * π * r^2 + 2 * π * r * h = 2 * π * r^2 + 2 * π * r * h) : 
  (h / r) = 2 := 
by
  sorry

end NUMINAMATH_GPT_minimize_surface_area_l1234_123415


namespace NUMINAMATH_GPT_ae_length_l1234_123499

theorem ae_length (AB CD AC AE : ℝ) (h: 2 * AE + 3 * AE = 34): 
  AE = 34 / 5 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_ae_length_l1234_123499


namespace NUMINAMATH_GPT_range_of_a_l1234_123424

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2^x - a ≥ 0) ↔ (a ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1234_123424


namespace NUMINAMATH_GPT_find_unknown_number_l1234_123455

theorem find_unknown_number :
  (0.86 ^ 3 - 0.1 ^ 3) / (0.86 ^ 2) + x + 0.1 ^ 2 = 0.76 → 
  x = 0.115296 :=
sorry

end NUMINAMATH_GPT_find_unknown_number_l1234_123455


namespace NUMINAMATH_GPT_distance_between_foci_of_ellipse_l1234_123401

theorem distance_between_foci_of_ellipse : 
  let a := 5
  let b := 2
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 * Real.sqrt 21 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_ellipse_l1234_123401


namespace NUMINAMATH_GPT_graph_symmetry_l1234_123412

/-- Theorem:
The functions y = 2^x and y = 2^{-x} are symmetric about the y-axis.
-/
theorem graph_symmetry :
  ∀ (x : ℝ), (∃ (y : ℝ), y = 2^x) →
  (∃ (y' : ℝ), y' = 2^(-x)) →
  (∀ (y : ℝ), ∃ (x : ℝ), (y = 2^x ↔ y = 2^(-x)) → y = 2^x → y = 2^(-x)) :=
by
  intro x
  intro h1
  intro h2
  intro y
  exists x
  intro h3
  intro hy
  sorry

end NUMINAMATH_GPT_graph_symmetry_l1234_123412


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1234_123454

variables (P Q : Prop)
variables (p : P) (q : Q)

-- Propositions
def quadrilateral_has_parallel_and_equal_sides : Prop := P
def is_rectangle : Prop := Q

-- Necessary but not sufficient condition
theorem necessary_but_not_sufficient (h : P → Q) : ¬(Q → P) :=
by sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1234_123454


namespace NUMINAMATH_GPT_perpendicular_k_parallel_k_l1234_123461

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Define the scalar multiple operations and vector operations
def smul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ := (v₁.1 + v₂.1, v₂.2 + v₂.2)
def sub (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ := (v₁.1 - v₂.1, v₂.2 - v₂.2)
def dot (v₁ v₂ : ℝ × ℝ) : ℝ := (v₁.1 * v₂.1 + v₁.2 * v₂.2)

-- Problem 1: If k*a + b is perpendicular to a - 3*b, then k = 19
theorem perpendicular_k (k : ℝ) :
  let vak := add (smul k a) b
  let amb := sub a (smul 3 b)
  dot vak amb = 0 → k = 19 := sorry

-- Problem 2: If k*a + b is parallel to a - 3*b, then k = -1/3 and they are in opposite directions
theorem parallel_k (k : ℝ) :
  let vak := add (smul k a) b
  let amb := sub a (smul 3 b)
  ∃ m : ℝ, vak = smul m amb ∧ m < 0 → k = -1/3 := sorry

end NUMINAMATH_GPT_perpendicular_k_parallel_k_l1234_123461


namespace NUMINAMATH_GPT_nell_baseball_cards_l1234_123485

theorem nell_baseball_cards 
  (ace_cards_now : ℕ) 
  (extra_baseball_cards : ℕ) 
  (B : ℕ) : 
  ace_cards_now = 55 →
  extra_baseball_cards = 123 →
  B = ace_cards_now + extra_baseball_cards →
  B = 178 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end NUMINAMATH_GPT_nell_baseball_cards_l1234_123485


namespace NUMINAMATH_GPT_average_age_of_4_students_l1234_123472

theorem average_age_of_4_students (avg_age_15 : ℕ) (num_students_15 : ℕ)
    (avg_age_10 : ℕ) (num_students_10 : ℕ) (age_15th_student : ℕ) :
    avg_age_15 = 15 ∧ num_students_15 = 15 ∧ avg_age_10 = 16 ∧ num_students_10 = 10 ∧ age_15th_student = 9 → 
    (56 / 4 = 14) := by
  sorry

end NUMINAMATH_GPT_average_age_of_4_students_l1234_123472


namespace NUMINAMATH_GPT_arithmetic_mean_geom_mean_ratio_l1234_123468

theorem arithmetic_mean_geom_mean_ratio {a b : ℝ} (h1 : (a + b) / 2 = 3 * Real.sqrt (a * b)) (h2 : a > b) (h3 : b > 0) : 
  (∃ k : ℤ, k = 34 ∧ abs ((a / b) - 34) ≤ 0.5) :=
sorry

end NUMINAMATH_GPT_arithmetic_mean_geom_mean_ratio_l1234_123468


namespace NUMINAMATH_GPT_proof1_proof2_proof3_proof4_l1234_123440

noncomputable def calc1 : ℝ := 3.21 - 1.05 - 1.95
noncomputable def calc2 : ℝ := 15 - (2.95 + 8.37)
noncomputable def calc3 : ℝ := 14.6 * 2 - 0.6 * 2
noncomputable def calc4 : ℝ := 0.25 * 1.25 * 32

theorem proof1 : calc1 = 0.21 := by
  sorry

theorem proof2 : calc2 = 3.68 := by
  sorry

theorem proof3 : calc3 = 28 := by
  sorry

theorem proof4 : calc4 = 10 := by
  sorry

end NUMINAMATH_GPT_proof1_proof2_proof3_proof4_l1234_123440


namespace NUMINAMATH_GPT_distinct_sequences_count_l1234_123419

-- Define the set of available letters excluding 'M' for start and 'S' for end
def available_letters : List Char := ['A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C']

-- Define the cardinality function for the sequences under given specific conditions.
-- This will check specific prompt format; you may want to specify permutations, combinations based on calculations but in the spirit, we are sticking to detail.
def count_sequences (letters : List Char) (n : Nat) : Nat :=
  if letters = available_letters ∧ n = 5 then 
    -- based on detailed calculation in the solution
    480
  else
    0

-- Theorem statement in Lean 4 to verify the number of sequences
theorem distinct_sequences_count : count_sequences available_letters 5 = 480 := 
sorry

end NUMINAMATH_GPT_distinct_sequences_count_l1234_123419


namespace NUMINAMATH_GPT_gas_price_l1234_123483

theorem gas_price (x : ℝ) (h1 : 10 * (x + 0.30) = 12 * x) : x + 0.30 = 1.80 := by
  sorry

end NUMINAMATH_GPT_gas_price_l1234_123483


namespace NUMINAMATH_GPT_dragos_wins_l1234_123422

variable (S : Set ℕ) [Infinite S]
variable (x : ℕ → ℕ)
variable (M N : ℕ)
variable (p : ℕ)

theorem dragos_wins (h_prime_p : Nat.Prime p) (h_subset_S : p ∈ S) 
  (h_xn_distinct : ∀ i j, i ≠ j → x i ≠ x j) 
  (h_pM_div_xn : ∀ n, n ≥ N → p^M ∣ x n): 
  ∃ N, ∀ n, n ≥ N → p^M ∣ x n :=
sorry

end NUMINAMATH_GPT_dragos_wins_l1234_123422


namespace NUMINAMATH_GPT_solve_for_x_l1234_123482

theorem solve_for_x (x : ℝ) (h : (x+10) / (x-4) = (x-3) / (x+6)) : x = -48 / 23 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1234_123482


namespace NUMINAMATH_GPT_ratio_of_segments_l1234_123466

theorem ratio_of_segments (a b : ℕ) (ha : a = 200) (hb : b = 40) : a / b = 5 :=
by sorry

end NUMINAMATH_GPT_ratio_of_segments_l1234_123466


namespace NUMINAMATH_GPT_min_deg_g_correct_l1234_123431

open Polynomial

noncomputable def min_deg_g {R : Type*} [CommRing R]
  (f g h : R[X])
  (hf : f.natDegree = 10)
  (hh : h.natDegree = 11)
  (h_eq : 5 * f + 6 * g = h) :
  Nat :=
11

theorem min_deg_g_correct {R : Type*} [CommRing R]
  (f g h : R[X])
  (hf : f.natDegree = 10)
  (hh : h.natDegree = 11)
  (h_eq : 5 * f + 6 * g = h) :
  (min_deg_g f g h hf hh h_eq = 11) :=
sorry

end NUMINAMATH_GPT_min_deg_g_correct_l1234_123431


namespace NUMINAMATH_GPT_michael_total_earnings_l1234_123441

-- Define the cost of large paintings and small paintings
def large_painting_cost : ℕ := 100
def small_painting_cost : ℕ := 80

-- Define the number of large and small paintings sold
def large_paintings_sold : ℕ := 5
def small_paintings_sold : ℕ := 8

-- Calculate Michael's total earnings
def total_earnings : ℕ := (large_painting_cost * large_paintings_sold) + (small_painting_cost * small_paintings_sold)

-- Prove: Michael's total earnings are 1140 dollars
theorem michael_total_earnings : total_earnings = 1140 := by
  sorry

end NUMINAMATH_GPT_michael_total_earnings_l1234_123441


namespace NUMINAMATH_GPT_arctan_sum_eq_half_pi_l1234_123446

theorem arctan_sum_eq_half_pi (y : ℚ) :
  2 * Real.arctan (1 / 3) + Real.arctan (1 / 10) + Real.arctan (1 / 30) + Real.arctan (1 / y) = Real.pi / 2 →
  y = 547 / 620 := by
  sorry

end NUMINAMATH_GPT_arctan_sum_eq_half_pi_l1234_123446


namespace NUMINAMATH_GPT_least_number_to_add_l1234_123430

theorem least_number_to_add (a : ℕ) (b : ℕ) (n : ℕ) (h : a = 1056) (h1: b = 26) (h2 : n = 10) : 
  (a + n) % b = 0 := 
sorry

end NUMINAMATH_GPT_least_number_to_add_l1234_123430


namespace NUMINAMATH_GPT_percentage_problem_l1234_123486

theorem percentage_problem (x : ℝ) (h : 0.3 * 0.4 * x = 45) : 0.4 * 0.3 * x = 45 :=
by
  sorry

end NUMINAMATH_GPT_percentage_problem_l1234_123486


namespace NUMINAMATH_GPT_prime_iff_satisfies_condition_l1234_123433

def satisfies_condition (n : ℕ) : Prop :=
  if n = 2 then True
  else if 2 < n then ∀ k : ℕ, 2 ≤ k ∧ k < n → ¬ (k ∣ n)
  else False

theorem prime_iff_satisfies_condition (n : ℕ) : Prime n ↔ satisfies_condition n := by
  sorry

end NUMINAMATH_GPT_prime_iff_satisfies_condition_l1234_123433


namespace NUMINAMATH_GPT_simplify_fraction_l1234_123484

theorem simplify_fraction :
  (2023^3 - 3 * 2023^2 * 2024 + 4 * 2023 * 2024^2 - 2024^3 + 2) / (2023 * 2024) = 2023 := by
sorry

end NUMINAMATH_GPT_simplify_fraction_l1234_123484


namespace NUMINAMATH_GPT_count_rhombuses_in_large_triangle_l1234_123478

-- Definitions based on conditions
def large_triangle_side_length : ℕ := 10
def small_triangle_side_length : ℕ := 1
def small_triangle_count : ℕ := 100
def rhombuses_of_8_triangles := 84

-- Problem statement in Lean 4
theorem count_rhombuses_in_large_triangle :
  ∀ (large_side small_side small_count : ℕ),
  large_side = large_triangle_side_length →
  small_side = small_triangle_side_length →
  small_count = small_triangle_count →
  (∃ (rhombus_count : ℕ), rhombus_count = rhombuses_of_8_triangles) :=
by
  intros large_side small_side small_count h_large h_small h_count
  use 84
  sorry

end NUMINAMATH_GPT_count_rhombuses_in_large_triangle_l1234_123478


namespace NUMINAMATH_GPT__l1234_123479

noncomputable def probability_event_b_given_a : ℕ → ℕ → ℕ → ℕ × ℕ → ℚ
| zeros, ones, twos, (1, drawn_label) =>
  if drawn_label = 1 then
    (ones * (ones - 1)) / (zeros + ones + twos).choose 2
  else 0
| _, _, _, _ => 0

lemma probability_theorem :
  let zeros := 1
  let ones := 2
  let twos := 2
  let total := zeros + ones + twos
  (1 - 1) * (ones - 1)/(total.choose 2) = 1/7 :=
by
  let zeros := 1
  let ones := 2
  let twos := 2
  let total := zeros + ones + twos
  let draw_label := 1
  let event_b_given_a := probability_event_b_given_a zeros ones twos (1, draw_label)
  have pos_cases : (ones * (ones - 1))/(total.choose 2) = 1 / 7 := by sorry
  exact pos_cases

end NUMINAMATH_GPT__l1234_123479


namespace NUMINAMATH_GPT_smoke_diagram_total_height_l1234_123450

theorem smoke_diagram_total_height : 
  ∀ (h1 h2 h3 h4 h5 : ℕ),
    h1 < h2 ∧ h2 < h3 ∧ h3 < h4 ∧ h4 < h5 ∧ 
    (h2 - h1 = 2) ∧ (h3 - h2 = 2) ∧ (h4 - h3 = 2) ∧ (h5 - h4 = 2) ∧ 
    (h5 = h1 + h2) → 
    h1 + h2 + h3 + h4 + h5 = 50 := 
by 
  sorry

end NUMINAMATH_GPT_smoke_diagram_total_height_l1234_123450


namespace NUMINAMATH_GPT_rectangle_diagonal_length_l1234_123411

theorem rectangle_diagonal_length
  (a b : ℝ)
  (h1 : a = 40 * Real.sqrt 2)
  (h2 : b = 2 * a) :
  Real.sqrt (a^2 + b^2) = 160 := by
  sorry

end NUMINAMATH_GPT_rectangle_diagonal_length_l1234_123411


namespace NUMINAMATH_GPT_total_copies_in_half_hour_l1234_123470

-- Definitions of the machine rates and their time segments.
def machine1_rate := 35 -- copies per minute
def machine2_rate := 65 -- copies per minute
def machine3_rate1 := 50 -- copies per minute for the first 15 minutes
def machine3_rate2 := 80 -- copies per minute for the next 15 minutes
def machine4_rate1 := 90 -- copies per minute for the first 10 minutes
def machine4_rate2 := 60 -- copies per minute for the next 20 minutes

-- Time intervals for different machines
def machine3_time1 := 15 -- minutes
def machine3_time2 := 15 -- minutes
def machine4_time1 := 10 -- minutes
def machine4_time2 := 20 -- minutes

-- Proof statement
theorem total_copies_in_half_hour : 
  (machine1_rate * 30) + 
  (machine2_rate * 30) + 
  ((machine3_rate1 * machine3_time1) + (machine3_rate2 * machine3_time2)) + 
  ((machine4_rate1 * machine4_time1) + (machine4_rate2 * machine4_time2)) = 
  7050 :=
by 
  sorry

end NUMINAMATH_GPT_total_copies_in_half_hour_l1234_123470


namespace NUMINAMATH_GPT_man_finishes_work_in_100_days_l1234_123456

variable (M W : ℝ)
variable (H1 : 10 * M * 6 + 15 * W * 6 = 1)
variable (H2 : W * 225 = 1)

theorem man_finishes_work_in_100_days (M W : ℝ) (H1 : 10 * M * 6 + 15 * W * 6 = 1) (H2 : W * 225 = 1) : M = 1 / 100 :=
by
  sorry

end NUMINAMATH_GPT_man_finishes_work_in_100_days_l1234_123456


namespace NUMINAMATH_GPT_line_parallel_to_parallel_set_l1234_123406

variables {Point Line Plane : Type} 
variables (a : Line) (α : Plane)
variables (parallel : Line → Plane → Prop) (parallel_set : Line → Plane → Prop)

-- Definition for line parallel to plane
axiom line_parallel_to_plane : parallel a α

-- Goal: line a is parallel to a set of parallel lines within plane α
theorem line_parallel_to_parallel_set (h : parallel a α) : parallel_set a α := 
sorry

end NUMINAMATH_GPT_line_parallel_to_parallel_set_l1234_123406


namespace NUMINAMATH_GPT_factor_expression_l1234_123495

theorem factor_expression (x : ℝ) : 60 * x + 45 = 15 * (4 * x + 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1234_123495


namespace NUMINAMATH_GPT_multiple_optimal_solutions_for_z_l1234_123408

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 0 2
def B := Point.mk (-2) (-2)
def C := Point.mk 2 0

def z (a : ℝ) (P : Point) : ℝ := P.y - a * P.x

def maxz_mult_opt_solutions (a : ℝ) : Prop :=
  z a A = z a B ∨ z a A = z a C ∨ z a B = z a C

theorem multiple_optimal_solutions_for_z :
  (maxz_mult_opt_solutions (-1)) ∧ (maxz_mult_opt_solutions 2) :=
by
  sorry

end NUMINAMATH_GPT_multiple_optimal_solutions_for_z_l1234_123408


namespace NUMINAMATH_GPT_quadratic_inequality_l1234_123403

theorem quadratic_inequality (a x : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_l1234_123403


namespace NUMINAMATH_GPT_hydrogen_burns_oxygen_certain_l1234_123426

-- define what it means for a chemical reaction to be well-documented and known to occur
def chemical_reaction (reactants : String) (products : String) : Prop :=
  (reactants = "2H₂ + O₂") ∧ (products = "2H₂O")

-- Event description and classification
def event_is_certain (event : String) : Prop :=
  event = "Hydrogen burns in oxygen to form water"

-- Main statement
theorem hydrogen_burns_oxygen_certain :
  ∀ (reactants products : String), (chemical_reaction reactants products) → event_is_certain "Hydrogen burns in oxygen to form water" :=
by
  intros reactants products h
  have h1 : reactants = "2H₂ + O₂" := h.1
  have h2 : products = "2H₂O" := h.2
  -- proof omitted
  exact sorry

end NUMINAMATH_GPT_hydrogen_burns_oxygen_certain_l1234_123426


namespace NUMINAMATH_GPT_find_vector_c_l1234_123429

-- Definitions of the given vectors
def vector_a : ℝ × ℝ := (3, -1)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (2 * vector_a.1 + vector_b.1, 2 * vector_a.2 + vector_b.2)

-- The goal is to prove that vector_c = (5, 0)
theorem find_vector_c : vector_c = (5, 0) :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_find_vector_c_l1234_123429


namespace NUMINAMATH_GPT_range_of_a_l1234_123473

open Real

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, otimes x (x + a) < 1) ↔ -1 < a ∧ a < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1234_123473


namespace NUMINAMATH_GPT_compare_exp_sin_ln_l1234_123442

theorem compare_exp_sin_ln :
  let a := Real.exp 0.1 - 1
  let b := Real.sin 0.1
  let c := Real.log 1.1
  c < b ∧ b < a :=
by
  sorry

end NUMINAMATH_GPT_compare_exp_sin_ln_l1234_123442


namespace NUMINAMATH_GPT_difference_in_lengths_l1234_123444

def speed_of_first_train := 60 -- in km/hr
def time_to_cross_pole_first_train := 3 -- in seconds
def speed_of_second_train := 90 -- in km/hr
def time_to_cross_pole_second_train := 2 -- in seconds

noncomputable def length_of_first_train : ℝ := (speed_of_first_train * (5 / 18)) * time_to_cross_pole_first_train
noncomputable def length_of_second_train : ℝ := (speed_of_second_train * (5 / 18)) * time_to_cross_pole_second_train

theorem difference_in_lengths : abs (length_of_second_train - length_of_first_train) = 0.01 :=
by
  -- The full proof would be placed here.
  sorry

end NUMINAMATH_GPT_difference_in_lengths_l1234_123444


namespace NUMINAMATH_GPT_sum_of_two_digit_numbers_with_gcd_lcm_l1234_123459

theorem sum_of_two_digit_numbers_with_gcd_lcm (x y : ℕ) (h1 : Nat.gcd x y = 8) (h2 : Nat.lcm x y = 96)
  (h3 : 10 ≤ x ∧ x < 100) (h4 : 10 ≤ y ∧ y < 100) : x + y = 56 :=
sorry

end NUMINAMATH_GPT_sum_of_two_digit_numbers_with_gcd_lcm_l1234_123459


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l1234_123464

theorem arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h₁ : a 1 = 3)
  (h₂ : ∀ n ≥ 2, 2 * a n = S n * S (n - 1)) :
  (∃ d : ℚ, d = -1/2 ∧ ∀ n ≥ 2, (1 / S n) - (1 / S (n - 1)) = d) :=
sorry

theorem general_term (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h₁ : a 1 = 3)
  (h₂ : ∀ n ≥ 2, 2 * a n = S n * S (n - 1)) :
  ∀ n, a n = if n = 1 then 3 else 18 / ((8 - 3 * n) * (5 - 3 * n)) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l1234_123464


namespace NUMINAMATH_GPT_fraction_speed_bus_train_l1234_123474

theorem fraction_speed_bus_train :
  let speed_train := 16 * 5
  let speed_bus := 480 / 8
  let speed_train_prop := speed_train = 80
  let speed_bus_prop := speed_bus = 60
  speed_bus / speed_train = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_speed_bus_train_l1234_123474


namespace NUMINAMATH_GPT_minimum_ab_ge_four_l1234_123402

variable (a b : ℝ)
variables (ha : 0 < a) (hb : 0 < b)
variable (h : 1 / a + 4 / b = Real.sqrt (a * b))

theorem minimum_ab_ge_four : a * b ≥ 4 := by
  sorry

end NUMINAMATH_GPT_minimum_ab_ge_four_l1234_123402


namespace NUMINAMATH_GPT_de_morgan_birth_year_jenkins_birth_year_l1234_123410

open Nat

theorem de_morgan_birth_year
  (x : ℕ) (hx : x = 43) (hx_square : x * x = 1849) :
  1849 - 43 = 1806 :=
by
  sorry

theorem jenkins_birth_year
  (a b : ℕ) (ha : a = 5) (hb : b = 6) (m : ℕ) (hm : m = 31) (n : ℕ) (hn : n = 5)
  (ha_sq : a * a = 25) (hb_sq : b * b = 36) (ha4 : a * a * a * a = 625)
  (hb4 : b * b * b * b = 1296) (hm2 : m * m = 961) (hn4 : n * n * n * n = 625) :
  1921 - 61 = 1860 ∧
  1922 - 62 = 1860 ∧
  1875 - 15 = 1860 :=
by
  sorry

end NUMINAMATH_GPT_de_morgan_birth_year_jenkins_birth_year_l1234_123410


namespace NUMINAMATH_GPT_number_of_valid_n_l1234_123439

theorem number_of_valid_n : 
  ∃ (c : Nat), (∀ n : Nat, (n + 9) * (n - 4) * (n - 13) < 0 → n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11 ∨ n = 12) ∧ c = 11 :=
by
  sorry

end NUMINAMATH_GPT_number_of_valid_n_l1234_123439


namespace NUMINAMATH_GPT_no_such_function_exists_l1234_123413

noncomputable def f : ℕ → ℕ := sorry

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), (∀ n > 1, f n = f (f (n-1)) + f (f (n+1))) ∧ (∀ n, f n > 0) :=
sorry

end NUMINAMATH_GPT_no_such_function_exists_l1234_123413


namespace NUMINAMATH_GPT_missing_number_l1234_123418

theorem missing_number (x : ℝ) (h : 0.72 * 0.43 + x * 0.34 = 0.3504) : x = 0.12 :=
by sorry

end NUMINAMATH_GPT_missing_number_l1234_123418


namespace NUMINAMATH_GPT_common_ratio_of_gp_l1234_123407

theorem common_ratio_of_gp (a r : ℝ) (h1 : r ≠ 1) 
  (h2 : (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 343) : r = 6 := 
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_gp_l1234_123407


namespace NUMINAMATH_GPT_sqrt_x_minus_5_meaningful_iff_x_ge_5_l1234_123460

theorem sqrt_x_minus_5_meaningful_iff_x_ge_5 (x : ℝ) : (∃ y : ℝ, y^2 = x - 5) ↔ (x ≥ 5) :=
sorry

end NUMINAMATH_GPT_sqrt_x_minus_5_meaningful_iff_x_ge_5_l1234_123460


namespace NUMINAMATH_GPT_gcd_greatest_possible_value_l1234_123462

noncomputable def Sn (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem gcd_greatest_possible_value (n : ℕ) (hn : 0 < n) : 
  Nat.gcd (3 * Sn n) (n + 1) = 1 :=
sorry

end NUMINAMATH_GPT_gcd_greatest_possible_value_l1234_123462


namespace NUMINAMATH_GPT_ratio_a_to_c_l1234_123434

theorem ratio_a_to_c {a b c : ℚ} (h1 : a / b = 4 / 3) (h2 : b / c = 1 / 5) :
  a / c = 4 / 5 := 
sorry

end NUMINAMATH_GPT_ratio_a_to_c_l1234_123434


namespace NUMINAMATH_GPT_difference_of_numbers_l1234_123432

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 34800) (h2 : b % 25 = 0) (h3 : b / 100 = a) : b - a = 32112 := by
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l1234_123432


namespace NUMINAMATH_GPT_congruence_problem_l1234_123435

theorem congruence_problem (x : ℤ) (h : 5 * x + 9 ≡ 4 [ZMOD 18]) : 3 * x + 15 ≡ 12 [ZMOD 18] :=
sorry

end NUMINAMATH_GPT_congruence_problem_l1234_123435


namespace NUMINAMATH_GPT_average_interest_rate_l1234_123492

theorem average_interest_rate (total_investment : ℝ) (rate1 rate2 : ℝ) (annual_return1 annual_return2 : ℝ) 
  (h1 : total_investment = 6000) 
  (h2 : rate1 = 0.035) 
  (h3 : rate2 = 0.055) 
  (h4 : annual_return1 = annual_return2) :
  (annual_return1 + annual_return2) / total_investment * 100 = 4.3 :=
by
  sorry

end NUMINAMATH_GPT_average_interest_rate_l1234_123492


namespace NUMINAMATH_GPT_alloy_ratio_proof_l1234_123497

def ratio_lead_to_tin_in_alloy_a (x y : ℝ) (ha : 0 < x) (hb : 0 < y) : Prop :=
  let weight_tin_in_a := (y / (x + y)) * 170
  let weight_tin_in_b := (3 / 8) * 250
  let total_tin := weight_tin_in_a + weight_tin_in_b
  total_tin = 221.25

theorem alloy_ratio_proof (x y : ℝ) (ha : 0 < x) (hb : 0 < y) (hc : ratio_lead_to_tin_in_alloy_a x y ha hb) : y / x = 3 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_alloy_ratio_proof_l1234_123497


namespace NUMINAMATH_GPT_bags_needed_l1234_123496

-- Definitions for the condition
def total_sugar : ℝ := 35.5
def bag_capacity : ℝ := 0.5

-- Theorem statement to solve the problem
theorem bags_needed : total_sugar / bag_capacity = 71 := 
by 
  sorry

end NUMINAMATH_GPT_bags_needed_l1234_123496


namespace NUMINAMATH_GPT_tangent_position_is_six_l1234_123417

def clock_radius : ℝ := 30
def disk_radius : ℝ := 15
def initial_tangent_position := 12
def final_tangent_position := 6

theorem tangent_position_is_six :
  (∃ (clock_radius disk_radius : ℝ), clock_radius = 30 ∧ disk_radius = 15) →
  (initial_tangent_position = 12) →
  (final_tangent_position = 6) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_tangent_position_is_six_l1234_123417


namespace NUMINAMATH_GPT_second_dog_miles_per_day_l1234_123498

-- Definitions describing conditions
section DogWalk
variable (total_miles_week : ℕ)
variable (first_dog_miles_day : ℕ)
variable (days_in_week : ℕ)

-- Assert conditions given in the problem
def condition1 := total_miles_week = 70
def condition2 := first_dog_miles_day = 2
def condition3 := days_in_week = 7

-- The theorem to prove
theorem second_dog_miles_per_day
  (h1 : condition1 total_miles_week)
  (h2 : condition2 first_dog_miles_day)
  (h3 : condition3 days_in_week) :
  (total_miles_week - days_in_week * first_dog_miles_day) / days_in_week = 8 :=
sorry
end DogWalk

end NUMINAMATH_GPT_second_dog_miles_per_day_l1234_123498


namespace NUMINAMATH_GPT_A_and_B_mutually_exclusive_l1234_123451

-- Definitions of events based on conditions
def A (a : ℕ) : Prop := a = 3
def B (a : ℕ) : Prop := a = 4

-- Define mutually exclusive
def mutually_exclusive (P Q : ℕ → Prop) : Prop :=
  ∀ a, P a → Q a → false

-- Problem statement: Prove A and B are mutually exclusive.
theorem A_and_B_mutually_exclusive :
  mutually_exclusive A B :=
sorry

end NUMINAMATH_GPT_A_and_B_mutually_exclusive_l1234_123451


namespace NUMINAMATH_GPT_skillful_hands_wire_cut_l1234_123488

theorem skillful_hands_wire_cut :
  ∃ x : ℕ, (1000 = 15 * x) ∧ (1040 = 15 * x) ∧ x = 66 :=
by
  sorry

end NUMINAMATH_GPT_skillful_hands_wire_cut_l1234_123488


namespace NUMINAMATH_GPT_num_teachers_l1234_123494

-- This statement involves defining the given conditions and stating the theorem to be proved.
theorem num_teachers (parents students total_people : ℕ) (h_parents : parents = 73) (h_students : students = 724) (h_total : total_people = 1541) :
  total_people - (parents + students) = 744 :=
by
  -- Including sorry to skip the proof, as required.
  sorry

end NUMINAMATH_GPT_num_teachers_l1234_123494
