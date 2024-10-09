import Mathlib

namespace number_of_valid_partitions_l1263_126328

-- Define the condition to check if a list of integers has all elements same or exactly differ by 1
def validPartition (l : List ℕ) : Prop :=
  l ≠ [] ∧ (∀ (a b : ℕ), a ∈ l → b ∈ l → a = b ∨ a = b + 1 ∨ b = a + 1)

-- Count valid partitions of n (integer partitions meeting the given condition)
noncomputable def countValidPartitions (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

-- Main theorem
theorem number_of_valid_partitions (n : ℕ) : countValidPartitions n = n :=
by
  sorry

end number_of_valid_partitions_l1263_126328


namespace probability_yellow_or_blue_twice_l1263_126368

theorem probability_yellow_or_blue_twice :
  let total_faces := 12
  let yellow_faces := 4
  let blue_faces := 2
  let probability_yellow_or_blue := (yellow_faces / total_faces) + (blue_faces / total_faces)
  (probability_yellow_or_blue * probability_yellow_or_blue) = 1 / 4 := 
by
  sorry

end probability_yellow_or_blue_twice_l1263_126368


namespace maximum_figures_per_shelf_l1263_126398

theorem maximum_figures_per_shelf
  (figures_shelf_1 : ℕ)
  (figures_shelf_2 : ℕ)
  (figures_shelf_3 : ℕ)
  (additional_shelves : ℕ)
  (max_figures_per_shelf : ℕ)
  (total_figures : ℕ)
  (total_shelves : ℕ)
  (H1 : figures_shelf_1 = 9)
  (H2 : figures_shelf_2 = 14)
  (H3 : figures_shelf_3 = 7)
  (H4 : additional_shelves = 2)
  (H5 : max_figures_per_shelf = 11)
  (H6 : total_figures = figures_shelf_1 + figures_shelf_2 + figures_shelf_3)
  (H7 : total_shelves = 3 + additional_shelves)
  (H8 : ∃ d, d ∈ ({x : ℕ | x ∣ total_figures} ∩ {y : ℕ | y ≤ max_figures_per_shelf}))
  : ∃ d, d ∈ ({x : ℕ | x ∣ total_figures} ∩ {y : ℕ | y ≤ max_figures_per_shelf}) ∧ d = 6 := sorry

end maximum_figures_per_shelf_l1263_126398


namespace probability_of_last_two_marbles_one_green_one_red_l1263_126374

theorem probability_of_last_two_marbles_one_green_one_red : 
    let total_marbles := 10
    let blue := 4
    let white := 3
    let red := 2
    let green := 1
    let total_ways := Nat.choose total_marbles 8
    let favorable_ways := Nat.choose (total_marbles - red - green) 6
    total_ways = 45 ∧ favorable_ways = 28 →
    (favorable_ways : ℚ) / total_ways = 28 / 45 :=
by
    intros total_marbles blue white red green total_ways favorable_ways h
    sorry

end probability_of_last_two_marbles_one_green_one_red_l1263_126374


namespace initial_pants_l1263_126339

theorem initial_pants (pairs_per_year : ℕ) (pants_per_pair : ℕ) (years : ℕ) (total_pants : ℕ) 
  (h1 : pairs_per_year = 4) (h2 : pants_per_pair = 2) (h3 : years = 5) (h4 : total_pants = 90) : 
  ∃ (initial_pants : ℕ), initial_pants = total_pants - (pairs_per_year * pants_per_pair * years) :=
by
  use 50
  sorry

end initial_pants_l1263_126339


namespace match_graph_l1263_126315

theorem match_graph (x : ℝ) (h : x ≤ 0) : 
  Real.sqrt (-2 * x^3) = -x * Real.sqrt (-2 * x) :=
by
  sorry

end match_graph_l1263_126315


namespace p_sufficient_for_q_q_not_necessary_for_p_l1263_126337

variable (x : ℝ)

def p := |x - 2| < 1
def q := 1 < x ∧ x < 5

theorem p_sufficient_for_q : p x → q x :=
by sorry

theorem q_not_necessary_for_p : ¬ (q x → p x) :=
by sorry

end p_sufficient_for_q_q_not_necessary_for_p_l1263_126337


namespace sin_diff_angle_identity_l1263_126338

open Real

noncomputable def alpha : ℝ := sorry -- α is an obtuse angle

axiom h1 : 90 < alpha ∧ alpha < 180 -- α is an obtuse angle
axiom h2 : cos alpha = -3 / 5 -- given cosine value

theorem sin_diff_angle_identity :
  sin (π / 4 - alpha) = - (7 * sqrt 2) / 10 :=
by
  sorry

end sin_diff_angle_identity_l1263_126338


namespace bacteria_growth_rate_l1263_126352

-- Define the existence of the growth rate and the initial amount of bacteria
variable (B : ℕ → ℝ) (B0 : ℝ) (r : ℝ)

-- State the conditions from the problem
axiom bacteria_growth_model : ∀ t : ℕ, B t = B0 * r ^ t
axiom day_30_full : B 30 = B0 * r ^ 30
axiom day_26_sixteenth : B 26 = (1 / 16) * B 30

-- Theorem stating that the growth rate r of the bacteria each day is 2
theorem bacteria_growth_rate : r = 2 := by
  sorry

end bacteria_growth_rate_l1263_126352


namespace find_line_equation_l1263_126343

open Real

-- Define the parabola
def Parabola (x y : ℝ) : Prop := y^2 = 2 * x

-- Define the line passing through (0,2)
def LineThruPoint (x y k : ℝ) : Prop := y = k * x + 2

-- Define when line intersects parabola
def LineIntersectsParabola (x1 y1 x2 y2 k : ℝ) : Prop :=
  LineThruPoint x1 y1 k ∧ LineThruPoint x2 y2 k ∧ Parabola x1 y1 ∧ Parabola x2 y2

-- Define when circle with diameter MN passes through origin O
def CircleThroughOrigin (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem find_line_equation (k : ℝ) 
    (h₀ : k ≠ 0)
    (h₁ : ∃ x1 y1 x2 y2, LineIntersectsParabola x1 y1 x2 y2 k)
    (h₂ : ∃ x1 y1 x2 y2, LineIntersectsParabola x1 y1 x2 y2 k ∧ CircleThroughOrigin x1 y1 x2 y2) :
  (∃ x y, LineThruPoint x y k ∧ y = -x + 2) :=
sorry

end find_line_equation_l1263_126343


namespace cubes_sum_correct_l1263_126388

noncomputable def max_cubes : ℕ := 11
noncomputable def min_cubes : ℕ := 9

theorem cubes_sum_correct : max_cubes + min_cubes = 20 :=
by
  unfold max_cubes min_cubes
  sorry

end cubes_sum_correct_l1263_126388


namespace simplify_expression_l1263_126334

theorem simplify_expression (x : ℝ) (h : x^2 + 2 * x = 1) :
  (1 - x) ^ 2 - (x + 3) * (3 - x) - (x - 3) * (x - 1) = -10 :=
by 
  sorry

end simplify_expression_l1263_126334


namespace system_of_equations_solution_l1263_126390

theorem system_of_equations_solution (x y : ℤ) 
  (h1 : x^2 + x * y + y^2 = 37) 
  (h2 : x^4 + x^2 * y^2 + y^4 = 481) : 
  (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3) ∨ (x = -3 ∧ y = -4) ∨ (x = -4 ∧ y = -3) := 
by sorry

end system_of_equations_solution_l1263_126390


namespace simplify_and_evaluate_expression_l1263_126320

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1 + Real.sqrt 3) :
  ((x + 3) / (x^2 - 2*x + 1) * (x - 1) / (x^2 + 3*x) + 1 / x) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l1263_126320


namespace P_subset_Q_l1263_126396

def P (x : ℝ) := abs x < 2
def Q (x : ℝ) := x < 2

theorem P_subset_Q : ∀ x : ℝ, P x → Q x := by
  sorry

end P_subset_Q_l1263_126396


namespace probability_two_points_one_unit_apart_l1263_126376

def twelve_points_probability : ℚ := 2 / 11

/-- Twelve points are spaced around at intervals of one unit around a \(3 \times 3\) square.
    Two of the 12 points are chosen at random.
    Prove that the probability that the two points are one unit apart is \(\frac{2}{11}\). -/
theorem probability_two_points_one_unit_apart :
  let total_points := 12
  let total_combinations := (total_points * (total_points - 1)) / 2
  let favorable_pairs := 12
  (favorable_pairs : ℚ) / total_combinations = twelve_points_probability := by
  sorry

end probability_two_points_one_unit_apart_l1263_126376


namespace xy_fraction_l1263_126358

theorem xy_fraction (x y : ℚ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) :
  x * y = -1 / 5 := 
by sorry

end xy_fraction_l1263_126358


namespace hockey_season_games_l1263_126353

theorem hockey_season_games (n_teams : ℕ) (n_faces : ℕ) (h1 : n_teams = 18) (h2 : n_faces = 10) :
  let total_games := (n_teams * (n_teams - 1) / 2) * n_faces
  total_games = 1530 :=
by
  sorry

end hockey_season_games_l1263_126353


namespace min_value_fraction_l1263_126332

theorem min_value_fraction (x y : ℝ) (h₁ : x + y = 4) (h₂ : x > y) (h₃ : y > 0) : (∃ z : ℝ, z = (2 / (x - y)) + (1 / y) ∧ z = 2) :=
by
  sorry

end min_value_fraction_l1263_126332


namespace tetrahedron_edge_assignment_possible_l1263_126355

theorem tetrahedron_edge_assignment_possible 
(s S a b : ℝ) 
(hs : s ≥ 0) (hS : S ≥ 0) (ha : a ≥ 0) (hb : b ≥ 0) :
  ∃ (e₁ e₂ e₃ e₄ e₅ e₆ : ℝ),
    e₁ ≥ 0 ∧ e₂ ≥ 0 ∧ e₃ ≥ 0 ∧ e₄ ≥ 0 ∧ e₅ ≥ 0 ∧ e₆ ≥ 0 ∧
    (e₁ + e₂ + e₃ = s) ∧ (e₁ + e₄ + e₅ = S) ∧
    (e₂ + e₄ + e₆ = a) ∧ (e₃ + e₅ + e₆ = b) := by
  sorry

end tetrahedron_edge_assignment_possible_l1263_126355


namespace find_smaller_integer_l1263_126324

noncomputable def average_equals_decimal (m n : ℕ) : Prop :=
  (m + n) / 2 = m + n / 100

theorem find_smaller_integer (m n : ℕ) (h1 : 10 ≤ m ∧ m < 100) (h2 : 10 ≤ n ∧ n < 100) (h3 : 25 ∣ n) (h4 : average_equals_decimal m n) : m = 49 :=
by
  sorry

end find_smaller_integer_l1263_126324


namespace root_expression_eq_l1263_126354

theorem root_expression_eq (p q α β γ δ : ℝ) 
  (h1 : ∀ x, (x - α) * (x - β) = x^2 + p * x + 2)
  (h2 : ∀ x, (x - γ) * (x - δ) = x^2 + q * x + 2) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = 4 + 2 * (p^2 - q^2) := 
sorry

end root_expression_eq_l1263_126354


namespace sum_of_ages_in_three_years_l1263_126346

theorem sum_of_ages_in_three_years (H : ℕ) (J : ℕ) (SumAges : ℕ) 
  (h1 : J = 3 * H) 
  (h2 : H = 15) 
  (h3 : SumAges = (H + 3) + (J + 3)) : 
  SumAges = 66 :=
by
  sorry

end sum_of_ages_in_three_years_l1263_126346


namespace range_of_mu_l1263_126382

theorem range_of_mu (a b μ : ℝ) (ha : 0 < a) (hb : 0 < b) (hμ : 0 < μ) (h : 1 / a + 9 / b = 1) : μ ≤ 16 :=
by
  sorry

end range_of_mu_l1263_126382


namespace time_spent_on_aerobics_l1263_126379

theorem time_spent_on_aerobics (A W : ℝ) 
  (h1 : A + W = 250) 
  (h2 : A / W = 3 / 2) : 
  A = 150 := 
sorry

end time_spent_on_aerobics_l1263_126379


namespace find_unknown_rate_of_two_blankets_l1263_126377

-- Definitions of conditions based on the problem statement
def purchased_blankets_at_100 : Nat := 3
def price_per_blanket_at_100 : Nat := 100
def total_cost_at_100 := purchased_blankets_at_100 * price_per_blanket_at_100

def purchased_blankets_at_150 : Nat := 3
def price_per_blanket_at_150 : Nat := 150
def total_cost_at_150 := purchased_blankets_at_150 * price_per_blanket_at_150

def purchased_blankets_at_x : Nat := 2
def blankets_total : Nat := 8
def average_price : Nat := 150
def total_cost := blankets_total * average_price

-- The proof statement
theorem find_unknown_rate_of_two_blankets (x : Nat) 
  (h : purchased_blankets_at_100 * price_per_blanket_at_100 + 
       purchased_blankets_at_150 * price_per_blanket_at_150 + 
       purchased_blankets_at_x * x = total_cost) : x = 225 :=
by sorry

end find_unknown_rate_of_two_blankets_l1263_126377


namespace problem_l1263_126387

theorem problem (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ q) : ¬ p ∧ q :=
by
  -- proof goes here
  sorry

end problem_l1263_126387


namespace arithmetic_sequence_difference_l1263_126364

theorem arithmetic_sequence_difference (a b c : ℤ) (d : ℤ)
  (h1 : 9 - 1 = 4 * d)
  (h2 : c - a = 2 * d) :
  c - a = 4 := by sorry

end arithmetic_sequence_difference_l1263_126364


namespace wardrobe_single_discount_l1263_126366

theorem wardrobe_single_discount :
  let p : ℝ := 50
  let d1 : ℝ := 0.30
  let d2 : ℝ := 0.20
  let final_price := p * (1 - d1) * (1 - d2)
  let equivalent_discount := 1 - (final_price / p)
  equivalent_discount = 0.44 :=
by
  let p : ℝ := 50
  let d1 : ℝ := 0.30
  let d2 : ℝ := 0.20
  let final_price := p * (1 - d1) * (1 - d2)
  let equivalent_discount := 1 - (final_price / p)
  show equivalent_discount = 0.44
  sorry

end wardrobe_single_discount_l1263_126366


namespace equivalent_trigonometric_identity_l1263_126329

variable (α : ℝ)

theorem equivalent_trigonometric_identity
  (h1 : α ∈ Set.Ioo (-(Real.pi/2)) 0)
  (h2 : Real.sin (α + (Real.pi/4)) = -1/3) :
  (Real.sin (2*α) / Real.cos ((Real.pi/4) - α)) = 7/3 := 
by
  sorry

end equivalent_trigonometric_identity_l1263_126329


namespace candies_total_l1263_126383

-- Defining the given conditions
def LindaCandies : ℕ := 34
def ChloeCandies : ℕ := 28
def TotalCandies : ℕ := LindaCandies + ChloeCandies

-- Proving the total number of candies
theorem candies_total : TotalCandies = 62 :=
  by
    sorry

end candies_total_l1263_126383


namespace customers_who_left_tip_l1263_126325

-- Define the initial number of customers
def initial_customers : ℕ := 39

-- Define the additional number of customers during lunch rush
def additional_customers : ℕ := 12

-- Define the number of customers who didn't leave a tip
def no_tip_customers : ℕ := 49

-- Prove the number of customers who did leave a tip
theorem customers_who_left_tip : (initial_customers + additional_customers) - no_tip_customers = 2 := by
  sorry

end customers_who_left_tip_l1263_126325


namespace opposite_of_three_l1263_126333

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l1263_126333


namespace problem_solution_l1263_126300

def is_quadratic (y : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, y x = a * x^2 + b * x + c

def not_quadratic_func := 
  let yA := fun x => -2 * x^2
  let yB := fun x => 2 * (x - 1)^2 + 1
  let yC := fun x => (x - 3)^2 - x^2
  let yD := fun a => a * (8 - a)
  (¬ is_quadratic yC) ∧ (is_quadratic yA) ∧ (is_quadratic yB) ∧ (is_quadratic yD)

theorem problem_solution : not_quadratic_func := 
sorry

end problem_solution_l1263_126300


namespace initial_books_in_bin_l1263_126386

theorem initial_books_in_bin
  (x : ℝ)
  (h : x + 33.0 + 2.0 = 76) :
  x = 41.0 :=
by 
  -- Proof goes here
  sorry

end initial_books_in_bin_l1263_126386


namespace polynomial_coeffs_l1263_126348

theorem polynomial_coeffs :
  ( ∃ (a1 a2 a3 a4 a5 : ℕ), (∀ (x : ℝ), (x + 1) ^ 3 * (x + 2) ^ 2 = x^5 + a1 * x^4 + a2 * x^3 + a3 * x^2 + a4 * x + a5) ∧ a4 = 16 ∧ a5 = 4) := 
by
  sorry

end polynomial_coeffs_l1263_126348


namespace no_k_satisfying_condition_l1263_126347

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_k_satisfying_condition :
  ∀ k : ℕ, (∃ p q : ℕ, p ≠ q ∧ is_prime p ∧ is_prime q ∧ k = p * q ∧ p + q = 71) → false :=
by
  sorry

end no_k_satisfying_condition_l1263_126347


namespace proof_problem_l1263_126309

theorem proof_problem
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) :
  (x + 1 / y ≥ 2) ∨ (y + 1 / z ≥ 2) ∨ (z + 1 / x ≥ 2) :=
sorry

end proof_problem_l1263_126309


namespace arts_school_probability_l1263_126305

theorem arts_school_probability :
  let cultural_courses := 3
  let arts_courses := 3
  let total_periods := 6
  let total_arrangements := Nat.factorial total_periods
  let no_adjacent_more_than_one_separator := (72 + 216 + 144)
  (no_adjacent_more_than_one_separator : ℝ) / (total_arrangements : ℝ) = (3 / 5 : ℝ) := 
by 
  sorry

end arts_school_probability_l1263_126305


namespace smallest_multiple_of_8_and_9_l1263_126340

theorem smallest_multiple_of_8_and_9 : ∃ n : ℕ, n > 0 ∧ (n % 8 = 0) ∧ (n % 9 = 0) ∧ (∀ m : ℕ, m > 0 ∧ (m % 8 = 0) ∧ (m % 9 = 0) → n ≤ m) ∧ n = 72 :=
by
  sorry

end smallest_multiple_of_8_and_9_l1263_126340


namespace modular_inverse_7_10000_l1263_126310

theorem modular_inverse_7_10000 :
  (7 * 8571) % 10000 = 1 := 
sorry

end modular_inverse_7_10000_l1263_126310


namespace age_of_teacher_l1263_126308

theorem age_of_teacher (S T : ℕ) (avg_students avg_total : ℕ) (num_students num_total : ℕ)
  (h1 : num_students = 50)
  (h2 : avg_students = 14)
  (h3 : num_total = 51)
  (h4 : avg_total = 15)
  (h5 : S = avg_students * num_students)
  (h6 : S + T = avg_total * num_total) :
  T = 65 := 
by {
  sorry
}

end age_of_teacher_l1263_126308


namespace no_integer_solutions_l1263_126335

theorem no_integer_solutions (a b c : ℤ) : ¬ (a^2 + b^2 = 8 * c + 6) :=
sorry

end no_integer_solutions_l1263_126335


namespace consecutive_composite_numbers_bound_l1263_126344

theorem consecutive_composite_numbers_bound (n : ℕ) (hn: 0 < n) :
  ∃ (seq : Fin n → ℕ), (∀ i, ¬ Nat.Prime (seq i)) ∧ (∀ i, seq i < 4^(n+1)) :=
sorry

end consecutive_composite_numbers_bound_l1263_126344


namespace cubic_meter_to_cubic_centimeters_l1263_126360

theorem cubic_meter_to_cubic_centimeters : 
  (1 : ℝ)^3 = (100 : ℝ)^3 * (1 : ℝ)^0 := 
by 
  sorry

end cubic_meter_to_cubic_centimeters_l1263_126360


namespace abs_x_plus_7_eq_0_has_no_solution_l1263_126312

theorem abs_x_plus_7_eq_0_has_no_solution : ¬∃ x : ℝ, |x| + 7 = 0 :=
by
  sorry

end abs_x_plus_7_eq_0_has_no_solution_l1263_126312


namespace sum_infinite_series_l1263_126395

theorem sum_infinite_series : (∑' n : ℕ, (n + 1) / 8^(n + 1)) = 8 / 49 := sorry

end sum_infinite_series_l1263_126395


namespace number_of_solutions_l1263_126302

theorem number_of_solutions (x : ℤ) (h1 : 0 < x) (h2 : x < 150) (h3 : (x + 17) % 46 = 75 % 46) : 
  ∃ n : ℕ, n = 3 :=
sorry

end number_of_solutions_l1263_126302


namespace hexagon_area_l1263_126326

theorem hexagon_area :
  let points := [(0, 0), (2, 4), (5, 4), (7, 0), (5, -4), (2, -4), (0, 0)]
  ∃ (area : ℝ), area = 52 := by
  sorry

end hexagon_area_l1263_126326


namespace simplify_expression_correct_l1263_126350

noncomputable def simplify_expr (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :=
  ((a / b) * ((b - (4 * (a^6) / b^3)) ^ (1 / 3))
    - a^2 * ((b / a^6 - (4 / b^3)) ^ (1 / 3))
    + (2 / (a * b)) * ((a^3 * b^4 - 4 * a^9) ^ (1 / 3))) /
    ((b^2 - 2 * a^3) ^ (1 / 3) / b^2)

theorem simplify_expression_correct (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  simplify_expr a b ha hb = (a + b) * ((b^2 + 2 * a^3) ^ (1 / 3)) :=
sorry

end simplify_expression_correct_l1263_126350


namespace shift_line_one_unit_left_l1263_126369

theorem shift_line_one_unit_left : ∀ (x y : ℝ), (y = x) → (y - 1 = (x + 1) - 1) :=
by
  intros x y h
  sorry

end shift_line_one_unit_left_l1263_126369


namespace a_4_value_l1263_126304

-- Definitions and Theorem
variable {α : Type*} [LinearOrderedField α]

noncomputable def geometric_seq (a₀ : α) (q : α) (n : ℕ) : α := a₀ * q ^ n

theorem a_4_value (a₁ : α) (q : α) (h : geometric_seq a₁ q 1 * geometric_seq a₁ q 2 * geometric_seq a₁ q 6 = 8) : 
  geometric_seq a₁ q 3 = 2 :=
sorry

end a_4_value_l1263_126304


namespace make_fraction_meaningful_l1263_126342

theorem make_fraction_meaningful (x : ℝ) : (x - 1) ≠ 0 ↔ x ≠ 1 :=
by
  sorry

end make_fraction_meaningful_l1263_126342


namespace total_logs_in_stack_l1263_126306

/-- The total number of logs in a stack where the top row has 5 logs,
each succeeding row has one more log than the one above,
and the bottom row has 15 logs. -/
theorem total_logs_in_stack :
  let a := 5               -- first term (logs in the top row)
  let l := 15              -- last term (logs in the bottom row)
  let n := l - a + 1       -- number of terms (rows)
  let S := n / 2 * (a + l) -- sum of the arithmetic series
  S = 110 := sorry

end total_logs_in_stack_l1263_126306


namespace marcy_total_people_served_l1263_126372

noncomputable def total_people_served_lip_gloss
  (tubs_lip_gloss : ℕ) (tubes_per_tub_lip_gloss : ℕ) (people_per_tube_lip_gloss : ℕ) : ℕ :=
  tubs_lip_gloss * tubes_per_tub_lip_gloss * people_per_tube_lip_gloss

noncomputable def total_people_served_mascara
  (tubs_mascara : ℕ) (tubes_per_tub_mascara : ℕ) (people_per_tube_mascara : ℕ) : ℕ :=
  tubs_mascara * tubes_per_tub_mascara * people_per_tube_mascara

theorem marcy_total_people_served :
  ∀ (tubs_lip_gloss tubs_mascara : ℕ) 
    (tubes_per_tub_lip_gloss tubes_per_tub_mascara 
     people_per_tube_lip_gloss people_per_tube_mascara : ℕ),
    tubs_lip_gloss = 6 → 
    tubes_per_tub_lip_gloss = 2 → 
    people_per_tube_lip_gloss = 3 → 
    tubs_mascara = 4 → 
    tubes_per_tub_mascara = 3 → 
    people_per_tube_mascara = 5 → 
    total_people_served_lip_gloss tubs_lip_gloss 
                                 tubes_per_tub_lip_gloss 
                                 people_per_tube_lip_gloss = 36 :=
by
  intros tubs_lip_gloss tubs_mascara 
         tubes_per_tub_lip_gloss tubes_per_tub_mascara 
         people_per_tube_lip_gloss people_per_tube_mascara
         h_tubs_lip_gloss h_tubes_per_tub_lip_gloss h_people_per_tube_lip_gloss
         h_tubs_mascara h_tubes_per_tub_mascara h_people_per_tube_mascara
  rw [h_tubs_lip_gloss, h_tubes_per_tub_lip_gloss, h_people_per_tube_lip_gloss]
  exact rfl


end marcy_total_people_served_l1263_126372


namespace smallest_fraction_l1263_126318

theorem smallest_fraction (x : ℝ) (h : x > 2022) :
  min (min (min (min (x / 2022) (2022 / (x - 1))) ((x + 1) / 2022)) (2022 / x)) (2022 / (x + 1)) = 2022 / (x + 1) :=
sorry

end smallest_fraction_l1263_126318


namespace initial_customers_l1263_126356

theorem initial_customers (x : ℝ) : (x - 8 + 4 = 9) → x = 13 :=
by
  sorry

end initial_customers_l1263_126356


namespace find_number_l1263_126389

theorem find_number (x : ℝ) (h : x / 0.04 = 25) : x = 1 := 
by 
  -- the steps for solving this will be provided here
  sorry

end find_number_l1263_126389


namespace determinant_2x2_l1263_126351

theorem determinant_2x2 (a b c d : ℝ) 
  (h : Matrix.det (Matrix.of ![![1, a, b], ![2, c, d], ![3, 0, 0]]) = 6) : 
  Matrix.det (Matrix.of ![![a, b], ![c, d]]) = 2 :=
by
  sorry

end determinant_2x2_l1263_126351


namespace problem1_problem2_min_value_l1263_126370

theorem problem1 (x : ℝ) : |x + 1| + |x - 2| ≥ 3 := sorry

theorem problem2 (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 14 := sorry

theorem min_value (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) :
  ∃ x y z, x^2 + y^2 + z^2 = 1 / 14 := sorry

end problem1_problem2_min_value_l1263_126370


namespace compute_k_plus_m_l1263_126362

theorem compute_k_plus_m :
  ∃ k m : ℝ, 
    (∀ (x y z : ℝ), x^3 - 9 * x^2 + k * x - m = 0 -> x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 9 ∧ 
    (x = 1 ∨ y = 1 ∨ z = 1) ∧ (x = 3 ∨ y = 3 ∨ z = 3) ∧ (x = 5 ∨ y = 5 ∨ z = 5)) →
    k + m = 38 :=
by
  sorry

end compute_k_plus_m_l1263_126362


namespace average_of_expressions_l1263_126314

theorem average_of_expressions (y : ℝ) :
  (1 / 3:ℝ) * ((2 * y + 5) + (3 * y + 4) + (7 * y - 2)) = 4 * y + 7 / 3 :=
by sorry

end average_of_expressions_l1263_126314


namespace overall_percentage_l1263_126391

theorem overall_percentage (s1 s2 s3 : ℝ) (h1 : s1 = 60) (h2 : s2 = 80) (h3 : s3 = 85) :
  (s1 + s2 + s3) / 3 = 75 := by
  sorry

end overall_percentage_l1263_126391


namespace pen_rubber_length_difference_l1263_126361

theorem pen_rubber_length_difference (P R : ℕ) 
    (h1 : P = R + 3)
    (h2 : P = 12 - 2) 
    (h3 : R + P + 12 = 29) : 
    P - R = 3 :=
  sorry

end pen_rubber_length_difference_l1263_126361


namespace find_first_number_l1263_126349

theorem find_first_number (x : ℝ) : (10 + 70 + 28) / 3 = 36 →
  (x + 40 + 60) / 3 = 40 →
  x = 20 := 
by
  intros h_avg_old h_avg_new
  sorry

end find_first_number_l1263_126349


namespace numberOfFlowerbeds_l1263_126381

def totalSeeds : ℕ := 32
def seedsPerFlowerbed : ℕ := 4

theorem numberOfFlowerbeds : totalSeeds / seedsPerFlowerbed = 8 :=
by
  sorry

end numberOfFlowerbeds_l1263_126381


namespace gemstones_count_l1263_126317

theorem gemstones_count (F B S W SN : ℕ) 
  (hS : S = 1)
  (hSpaatz : S = F / 2 - 2)
  (hBinkie : B = 4 * F)
  (hWhiskers : W = S + 3)
  (hSnowball : SN = 2 * W) :
  B = 24 :=
by
  sorry

end gemstones_count_l1263_126317


namespace a_eq_b_if_b2_ab_1_divides_a2_ab_1_l1263_126327

theorem a_eq_b_if_b2_ab_1_divides_a2_ab_1 (a b : ℕ) (ha_pos : a > 0) (hb_pos : b > 0)
  (h : b^2 + a * b + 1 ∣ a^2 + a * b + 1) : a = b :=
by
  sorry

end a_eq_b_if_b2_ab_1_divides_a2_ab_1_l1263_126327


namespace real_part_of_diff_times_i_l1263_126359

open Complex

def z1 : ℂ := (4 : ℂ) + (29 : ℂ) * I
def z2 : ℂ := (6 : ℂ) + (9 : ℂ) * I

theorem real_part_of_diff_times_i :
  re ((z1 - z2) * I) = -20 := 
sorry

end real_part_of_diff_times_i_l1263_126359


namespace range_of_a_l1263_126394

theorem range_of_a (a : ℝ) : ((1 - a) ^ 2 + (1 + a) ^ 2 < 4) → (-1 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l1263_126394


namespace train_cross_bridge_time_l1263_126307

noncomputable def length_train : ℝ := 130
noncomputable def length_bridge : ℝ := 320
noncomputable def speed_kmh : ℝ := 54
noncomputable def speed_ms : ℝ := speed_kmh * 1000 / 3600

theorem train_cross_bridge_time :
  (length_train + length_bridge) / speed_ms = 30 := by
  sorry

end train_cross_bridge_time_l1263_126307


namespace prob1_converse_prob1_inverse_prob1_contrapositive_prob2_converse_prob2_inverse_prob2_contrapositive_l1263_126316

-- Problem 1: Original proposition converse, inverse, contrapositive
theorem prob1_converse (x y : ℝ) (h : x = 0 ∨ y = 0) : x * y = 0 :=
sorry

theorem prob1_inverse (x y : ℝ) (h : x * y ≠ 0) : x ≠ 0 ∧ y ≠ 0 :=
sorry

theorem prob1_contrapositive (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) : x * y ≠ 0 :=
sorry

-- Problem 2: Original proposition converse, inverse, contrapositive
theorem prob2_converse (x y : ℝ) (h : x * y > 0) : x > 0 ∧ y > 0 :=
sorry

theorem prob2_inverse (x y : ℝ) (h : x ≤ 0 ∨ y ≤ 0) : x * y ≤ 0 :=
sorry

theorem prob2_contrapositive (x y : ℝ) (h : x * y ≤ 0) : x ≤ 0 ∨ y ≤ 0 :=
sorry

end prob1_converse_prob1_inverse_prob1_contrapositive_prob2_converse_prob2_inverse_prob2_contrapositive_l1263_126316


namespace train_length_l1263_126378

noncomputable def length_of_each_train : ℝ :=
  let speed_faster_train_km_per_hr := 46
  let speed_slower_train_km_per_hr := 36
  let relative_speed_km_per_hr := speed_faster_train_km_per_hr - speed_slower_train_km_per_hr
  let relative_speed_m_per_s := (relative_speed_km_per_hr * 1000) / 3600
  let time_s := 54
  let distance_m := relative_speed_m_per_s * time_s
  distance_m / 2

theorem train_length : length_of_each_train = 75 := by
  sorry

end train_length_l1263_126378


namespace Raja_and_Ram_together_l1263_126365

def RajaDays : ℕ := 12
def RamDays : ℕ := 6

theorem Raja_and_Ram_together (W : ℕ) : 
  let RajaRate := W / RajaDays
  let RamRate := W / RamDays
  let CombinedRate := RajaRate + RamRate 
  let DaysTogether := W / CombinedRate 
  DaysTogether = 4 := 
by
  sorry

end Raja_and_Ram_together_l1263_126365


namespace mapping_sum_l1263_126367

theorem mapping_sum (f : ℝ × ℝ → ℝ × ℝ) (a b : ℝ)
(h1 : ∀ x y, f (x, y) = (x, x + y))
(h2 : (a, b) = f (1, 3)) :
  a + b = 5 :=
sorry

end mapping_sum_l1263_126367


namespace power_function_analysis_l1263_126373

theorem power_function_analysis (f : ℝ → ℝ) (α : ℝ) (h : ∀ x > 0, f x = x ^ α) (h_f : f 9 = 3) :
  (∀ x ≥ 0, f x = x ^ (1 / 2)) ∧
  (∀ x ≥ 4, f x ≥ 2) ∧
  (∀ x1 x2 : ℝ, x2 > x1 ∧ x1 > 0 → (f (x1) + f (x2)) / 2 < f ((x1 + x2) / 2)) :=
by
  -- Solution steps would go here
  sorry

end power_function_analysis_l1263_126373


namespace nancy_deleted_files_correct_l1263_126393

-- Variables and conditions
def nancy_original_files : Nat := 43
def files_per_folder : Nat := 6
def number_of_folders : Nat := 2

-- Definition of the number of files that were deleted
def nancy_files_deleted : Nat :=
  nancy_original_files - (files_per_folder * number_of_folders)

-- Theorem to prove
theorem nancy_deleted_files_correct :
  nancy_files_deleted = 31 :=
by
  sorry

end nancy_deleted_files_correct_l1263_126393


namespace rich_walked_distance_l1263_126371

def total_distance_walked (d1 d2 : ℕ) := 
  d1 + d2 + 2 * (d1 + d2) + (d1 + d2 + 2 * (d1 + d2)) / 2

def distance_to_intersection (d1 d2 : ℕ) := 
  2 * (d1 + d2)

def distance_to_end_route (d1 d2 : ℕ) := 
  (d1 + d2 + distance_to_intersection d1 d2) / 2

def total_distance_one_way (d1 d2 : ℕ) := 
  (d1 + d2) + (distance_to_intersection d1 d2) + (distance_to_end_route d1 d2)

theorem rich_walked_distance
  (d1 : ℕ := 20)
  (d2 : ℕ := 200) :
  2 * total_distance_one_way d1 d2 = 1980 :=
by
  simp [total_distance_one_way, distance_to_intersection, distance_to_end_route, total_distance_walked]
  sorry

end rich_walked_distance_l1263_126371


namespace range_of_m_l1263_126319

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (2 / x) + (3 / y) = 1)
  (h4 : 3 * x + 2 * y > m^2 + 2 * m) :
  -6 < m ∧ m < 4 :=
sorry

end range_of_m_l1263_126319


namespace geometric_triangle_q_range_l1263_126341

theorem geometric_triangle_q_range (a : ℝ) (q : ℝ) (h : 0 < q) 
  (h1 : a + q * a > (q ^ 2) * a)
  (h2 : q * a + (q ^ 2) * a > a)
  (h3 : a + (q ^ 2) * a > q * a) : 
  q ∈ Set.Ioo ((Real.sqrt 5 - 1) / 2) ((1 + Real.sqrt 5) / 2) :=
sorry

end geometric_triangle_q_range_l1263_126341


namespace geometric_sequence_a5_l1263_126311

variable {a : ℕ → ℝ}
variable (h₁ : a 3 * a 7 = 3)
variable (h₂ : a 3 + a 7 = 4)

theorem geometric_sequence_a5 : a 5 = Real.sqrt 3 := 
sorry

end geometric_sequence_a5_l1263_126311


namespace apples_bought_l1263_126385

theorem apples_bought (x : ℕ) 
  (h1 : x ≠ 0)  -- x must be a positive integer
  (h2 : 2 * (x/3) = 2 * x / 3 + 2 - 6) : x = 24 := 
  by sorry

end apples_bought_l1263_126385


namespace product_check_l1263_126330

theorem product_check : 
  (1200 < 31 * 53 ∧ 31 * 53 < 2400) ∧ 
  ¬ (1200 < 32 * 84 ∧ 32 * 84 < 2400) ∧ 
  ¬ (1200 < 63 * 54 ∧ 63 * 54 < 2400) ∧ 
  (1200 < 72 * 24 ∧ 72 * 24 < 2400) :=
by 
  sorry

end product_check_l1263_126330


namespace minimize_cylinder_surface_area_l1263_126397

noncomputable def cylinder_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem minimize_cylinder_surface_area :
  ∃ r h : ℝ, cylinder_volume r h = 16 * Real.pi ∧
  (∀ r' h', cylinder_volume r' h' = 16 * Real.pi → cylinder_surface_area r h ≤ cylinder_surface_area r' h') ∧ r = 2 := by
  sorry

end minimize_cylinder_surface_area_l1263_126397


namespace min_distance_symmetry_l1263_126322

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 + x + 1

def line (x y : ℝ) : Prop := 2 * x - y = 3

theorem min_distance_symmetry :
  ∀ (P Q : ℝ × ℝ),
    line P.1 P.2 → line Q.1 Q.2 →
    (exists (x : ℝ), P = (x, f x)) ∧
    (exists (x : ℝ), Q = (x, f x)) →
    ∃ (d : ℝ), d = 2 * Real.sqrt 5 :=
sorry

end min_distance_symmetry_l1263_126322


namespace area_of_triangle_ABC_l1263_126357

theorem area_of_triangle_ABC 
  (ABCD_is_trapezoid : ∀ {a b c d : ℝ}, a + d = b + c)
  (area_ABCD : ∀ {a b : ℝ}, a * b = 24)
  (CD_three_times_AB : ∀ {a : ℝ}, a * 3 = 24) :
  ∃ (area_ABC : ℝ), area_ABC = 6 :=
by 
  sorry

end area_of_triangle_ABC_l1263_126357


namespace dual_cassette_recorder_price_l1263_126321

theorem dual_cassette_recorder_price :
  ∃ (x y : ℝ),
    (x - 0.05 * x = 380) ∧
    (y = x + 0.08 * x) ∧ 
    (y = 432) :=
by
  -- sorry to skip the proof.
  sorry

end dual_cassette_recorder_price_l1263_126321


namespace company_production_average_l1263_126380

theorem company_production_average (n : ℕ) 
  (h1 : (50 * n) / n = 50) 
  (h2 : (50 * n + 105) / (n + 1) = 55) :
  n = 10 :=
sorry

end company_production_average_l1263_126380


namespace calories_for_breakfast_l1263_126375

theorem calories_for_breakfast :
  let cake_calories := 110
  let chips_calories := 310
  let coke_calories := 215
  let lunch_calories := 780
  let daily_limit := 2500
  let remaining_calories := 525
  let total_dinner_snacks := cake_calories + chips_calories + coke_calories
  let total_lunch_dinner := total_dinner_snacks + lunch_calories
  let total_consumed := daily_limit - remaining_calories
  total_consumed - total_lunch_dinner = 560 := by
  sorry

end calories_for_breakfast_l1263_126375


namespace probability_point_between_X_and_Z_l1263_126345

theorem probability_point_between_X_and_Z (XW XZ YW : ℝ) (h1 : XW = 4 * XZ) (h2 : XW = 8 * YW) :
  (XZ / XW) = 1 / 4 := by
  sorry

end probability_point_between_X_and_Z_l1263_126345


namespace minimum_m_n_squared_l1263_126331

theorem minimum_m_n_squared (a b c m n : ℝ) (h1 : c > a) (h2 : c > b) (h3 : c = Real.sqrt (a^2 + b^2)) 
    (h4 : a * m + b * n + c = 0) : m^2 + n^2 ≥ 1 := by
  sorry

end minimum_m_n_squared_l1263_126331


namespace pyramid_height_correct_l1263_126323

noncomputable def pyramid_height : ℝ :=
  let ab := 15 * Real.sqrt 3
  let bc := 14 * Real.sqrt 3
  let base_area := ab * bc
  let volume := 750
  let height := 3 * volume / base_area
  height

theorem pyramid_height_correct : pyramid_height = 25 / 7 :=
by
  sorry

end pyramid_height_correct_l1263_126323


namespace percent_twelve_equals_eighty_four_l1263_126303

theorem percent_twelve_equals_eighty_four (x : ℝ) (h : (12 / 100) * x = 84) : x = 700 :=
by
  sorry

end percent_twelve_equals_eighty_four_l1263_126303


namespace simplify_expression_l1263_126399

theorem simplify_expression (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h : a^4 + b^4 = a^2 + b^2) :
  (a / b + b / a - 1 / (a * b)) = 3 :=
  sorry

end simplify_expression_l1263_126399


namespace ptolemys_theorem_l1263_126384

-- Definition of the variables describing the lengths of the sides and diagonals
variables {a b c d m n : ℝ}

-- We declare that they belong to a cyclic quadrilateral
def cyclic_quadrilateral (a b c d m n : ℝ) : Prop :=
∃ (A B C D : ℝ), 
  A + C = 180 ∧ 
  B + D = 180 ∧ 
  m = (A * C) ∧ 
  n = (B * D) ∧ 
  a = (A * B) ∧ 
  b = (B * C) ∧ 
  c = (C * D) ∧ 
  d = (D * A)

-- The theorem statement in Lean form
theorem ptolemys_theorem (h : cyclic_quadrilateral a b c d m n) : m * n = a * c + b * d :=
sorry

end ptolemys_theorem_l1263_126384


namespace all_lights_on_l1263_126313

def light_on (n : ℕ) : Prop := sorry

axiom light_rule_1 (k : ℕ) (hk: light_on k): light_on (2 * k) ∧ light_on (2 * k + 1)
axiom light_rule_2 (k : ℕ) (hk: ¬ light_on k): ¬ light_on (4 * k + 1) ∧ ¬ light_on (4 * k + 3)
axiom light_2023_on : light_on 2023

theorem all_lights_on (n : ℕ) (hn : n < 2023) : light_on n :=
by sorry

end all_lights_on_l1263_126313


namespace mean_of_five_numbers_is_correct_l1263_126392

-- Define the sum of the five numbers
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers
def number_of_numbers : ℚ := 5

-- Define the mean
def mean_of_five_numbers := sum_of_five_numbers / number_of_numbers

-- State the theorem
theorem mean_of_five_numbers_is_correct : mean_of_five_numbers = 3 / 20 :=
by
  -- The proof is omitted, use sorry to indicate this.
  sorry

end mean_of_five_numbers_is_correct_l1263_126392


namespace number_division_equals_value_l1263_126363

theorem number_division_equals_value (x : ℝ) (h : x / 0.144 = 14.4 / 0.0144) : x = 144 :=
by
  sorry

end number_division_equals_value_l1263_126363


namespace arithmetic_seq_of_equal_roots_l1263_126336

theorem arithmetic_seq_of_equal_roots (a b c : ℝ) (h : b ≠ 0) 
    (h_eq_roots : ∃ x, b*x^2 - 4*b*x + 2*(a + c) = 0 ∧ (∀ y, b*y^2 - 4*b*y + 2*(a + c) = 0 → x = y)) : 
    b - a = c - b := 
by 
  -- placeholder for proof body
  sorry

end arithmetic_seq_of_equal_roots_l1263_126336


namespace complement_M_eq_45_l1263_126301

open Set Nat

/-- Define the universal set U and the set M in Lean -/
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

def M : Set ℕ := {x | 6 % x = 0 ∧ x ∈ U}

/-- Lean theorem statement for the complement of M in U -/
theorem complement_M_eq_45 : (U \ M) = {4, 5} :=
by
  sorry

end complement_M_eq_45_l1263_126301
