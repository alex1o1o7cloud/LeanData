import Mathlib

namespace calc1_calc2_l107_107423

-- Problem 1
theorem calc1 : 2 * Real.sqrt 3 - 3 * Real.sqrt 12 + 5 * Real.sqrt 27 = 11 * Real.sqrt 3 := 
by sorry

-- Problem 2
theorem calc2 : (1 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 6) - (2 * Real.sqrt 3 - 1)^2 
              = -2 * Real.sqrt 2 + 4 * Real.sqrt 3 - 13 := 
by sorry

end calc1_calc2_l107_107423


namespace solve_quadratic_eq_l107_107279

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem solve_quadratic_eq :
  ∀ a b c x1 x2 : ℝ,
  a = 2 →
  b = -2 →
  c = -1 →
  quadratic_eq a b c x1 ∧ quadratic_eq a b c x2 →
  (x1 = (1 + Real.sqrt 3) / 2 ∧ x2 = (1 - Real.sqrt 3) / 2) :=
by
  intros a b c x1 x2 ha hb hc h
  sorry

end solve_quadratic_eq_l107_107279


namespace ratio_problem_l107_107521

theorem ratio_problem (c d : ℚ) (h1 : c / d = 4) (h2 : c = 15 - 3 * d) : d = 15 / 7 := by
  sorry

end ratio_problem_l107_107521


namespace celia_time_correct_lexie_time_correct_nik_time_correct_l107_107468

noncomputable def lexie_time_per_mile : ℝ := 20
noncomputable def celia_time_per_mile : ℝ := lexie_time_per_mile / 2
noncomputable def nik_time_per_mile : ℝ := lexie_time_per_mile / 1.5

noncomputable def total_distance : ℝ := 30

-- Calculate the baseline running time without obstacles
noncomputable def lexie_baseline_time : ℝ := lexie_time_per_mile * total_distance
noncomputable def celia_baseline_time : ℝ := celia_time_per_mile * total_distance
noncomputable def nik_baseline_time : ℝ := nik_time_per_mile * total_distance

-- Additional time due to obstacles
noncomputable def celia_muddy_extra_time : ℝ := 2 * (celia_time_per_mile * 1.25 - celia_time_per_mile)
noncomputable def lexie_bee_extra_time : ℝ := 2 * 10
noncomputable def nik_detour_extra_time : ℝ := 0.5 * nik_time_per_mile

-- Total time taken including obstacles
noncomputable def celia_total_time : ℝ := celia_baseline_time + celia_muddy_extra_time
noncomputable def lexie_total_time : ℝ := lexie_baseline_time + lexie_bee_extra_time
noncomputable def nik_total_time : ℝ := nik_baseline_time + nik_detour_extra_time

theorem celia_time_correct : celia_total_time = 305 := by sorry
theorem lexie_time_correct : lexie_total_time = 620 := by sorry
theorem nik_time_correct : nik_total_time = 406.565 := by sorry

end celia_time_correct_lexie_time_correct_nik_time_correct_l107_107468


namespace placement_proof_l107_107563

def claimed_first_place (p: String) : Prop := 
  p = "Olya" ∨ p = "Oleg" ∨ p = "Pasha"

def odd_places_boys (positions: ℕ → String) : Prop := 
  (positions 1 = "Oleg" ∨ positions 1 = "Pasha") ∧ (positions 3 = "Oleg" ∨ positions 3 = "Pasha")

def olya_wrong (positions : ℕ → String) : Prop := 
  ¬odd_places_boys positions

def always_truthful_or_lying (Olya_st: Prop) (Oleg_st: Prop) (Pasha_st: Prop) : Prop := 
  Olya_st = Oleg_st ∧ Oleg_st = Pasha_st

def competition_placement : Prop :=
  ∃ (positions: ℕ → String),
    claimed_first_place (positions 1) ∧
    claimed_first_place (positions 2) ∧
    claimed_first_place (positions 3) ∧
    (positions 1 = "Oleg") ∧
    (positions 2 = "Pasha") ∧
    (positions 3 = "Olya") ∧
    olya_wrong positions ∧
    always_truthful_or_lying
      ((claimed_first_place "Olya" ∧ odd_places_boys positions))
      ((claimed_first_place "Oleg" ∧ olya_wrong positions))
      (claimed_first_place "Pasha")

theorem placement_proof : competition_placement :=
  sorry

end placement_proof_l107_107563


namespace number_times_frac_eq_cube_l107_107599

theorem number_times_frac_eq_cube (x : ℕ) : x * (1/6)^2 = 6^3 → x = 7776 :=
by
  intro h
  -- skipped proof
  sorry

end number_times_frac_eq_cube_l107_107599


namespace walking_time_estimate_l107_107596

-- Define constants for distance, speed, and time conversion factor
def distance : ℝ := 1000
def speed : ℝ := 4000
def time_conversion : ℝ := 60

-- Define the expected time to walk from home to school in minutes
def expected_time : ℝ := 15

-- Prove the time calculation
theorem walking_time_estimate : (distance / speed) * time_conversion = expected_time :=
by
  sorry

end walking_time_estimate_l107_107596


namespace unique_quantities_not_determinable_l107_107080

noncomputable def impossible_to_determine_unique_quantities 
(x y : ℝ) : Prop :=
  let acid1 := 54 * 0.35
  let acid2 := 48 * 0.25
  ∀ (final_acid : ℝ), ¬(0.35 * x + 0.25 * y = final_acid ∧ final_acid = 0.75 * (x + y))

theorem unique_quantities_not_determinable :
  impossible_to_determine_unique_quantities 54 48 :=
by
  sorry

end unique_quantities_not_determinable_l107_107080


namespace find_a_plus_k_l107_107150

-- Define the conditions.
def foci1 : (ℝ × ℝ) := (2, 0)
def foci2 : (ℝ × ℝ) := (2, 4)
def ellipse_point : (ℝ × ℝ) := (7, 2)

-- Statement of the equivalent proof problem.
theorem find_a_plus_k (a b h k : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (∀ x y, ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ↔ (x, y) = ellipse_point) →
  h = 2 → k = 2 → a = 5 →
  a + k = 7 :=
by
  sorry

end find_a_plus_k_l107_107150


namespace initial_stops_eq_l107_107765

-- Define the total number of stops S
def total_stops : ℕ := 7

-- Define the number of stops made after the initial deliveries
def additional_stops : ℕ := 4

-- Define the number of initial stops as a proof problem
theorem initial_stops_eq : total_stops - additional_stops = 3 :=
by
sorry

end initial_stops_eq_l107_107765


namespace bill_order_combinations_l107_107616

theorem bill_order_combinations :
  let k := 3  -- number of specific kinds with restriction
      m := 8  -- total number of donuts
      n := 6  -- total number of kinds
  in (∀ (purchase : Fin k → Nat), (∀ i, purchase i ≥ 2) ∧ (∑ i, purchase i = m)).count
    = 21 := by
    sorry

end bill_order_combinations_l107_107616


namespace slopes_of_intersecting_line_l107_107975

theorem slopes_of_intersecting_line {m : ℝ} :
  (∃ x y : ℝ, y = m * x + 4 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ Set.Iic (-Real.sqrt 0.48) ∪ Set.Ici (Real.sqrt 0.48) :=
by
  sorry

end slopes_of_intersecting_line_l107_107975


namespace triangle_possible_sides_l107_107222

theorem triangle_possible_sides : 
  let unknown_side_lengths := {x : ℤ | 3 < x ∧ x < 13}
  in set.finite unknown_side_lengths ∧ set.card unknown_side_lengths = 9 :=
by
  sorry

end triangle_possible_sides_l107_107222


namespace solve_for_x_l107_107027

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 1) (h_eq : y = 1 / (3 * x^2 + 2 * x + 1)) : x = 0 ∨ x = -2 / 3 :=
by
  sorry

end solve_for_x_l107_107027


namespace complex_subtraction_l107_107686

theorem complex_subtraction (z1 z2 : ℂ) (h1 : z1 = 2 + 3 * I) (h2 : z2 = 3 + I) :
  z1 - z2 = -1 + 2 * I := 
by
  sorry

end complex_subtraction_l107_107686


namespace find_T_l107_107670

variable {n : ℕ}
variable {a b : ℕ → ℕ}
variable {S T : ℕ → ℕ}

-- Conditions
axiom h1 : ∀ n, b n - a n = 2^n + 1
axiom h2 : ∀ n, S n + T n = 2^(n + 1) + n^2 - 2

-- Goal
theorem find_T (n : ℕ) (a b S T : ℕ → ℕ)
  (h1 : ∀ n, b n - a n = 2^n + 1)
  (h2 : ∀ n, S n + T n = 2^(n + 1) + n^2 - 2) :
  T n = 2^(n + 1) + n * (n + 1) / 2 - 5 := sorry

end find_T_l107_107670


namespace Dave_won_tickets_l107_107156

theorem Dave_won_tickets :
  ∀ (tickets_toys tickets_clothes total_tickets : ℕ),
  (tickets_toys = 8) →
  (tickets_clothes = 18) →
  (tickets_clothes = tickets_toys + 10) →
  (total_tickets = tickets_toys + tickets_clothes) →
  total_tickets = 26 :=
by
  intros tickets_toys tickets_clothes total_tickets h1 h2 h3 h4
  have h5 : tickets_clothes = 8 + 10 := by sorry
  have h6 : tickets_clothes = 18 := by sorry
  have h7 : tickets_clothes = 18 := by sorry
  exact sorry

end Dave_won_tickets_l107_107156


namespace min_training_iterations_l107_107628

/-- The model of exponentially decaying learning rate is given by L = L0 * D^(G / G0)
    where
    L  : the learning rate used in each round of optimization,
    L0 : the initial learning rate,
    D  : the decay coefficient,
    G  : the number of training iterations,
    G0 : the decay rate.

    Given:
    - the initial learning rate L0 = 0.5,
    - the decay rate G0 = 18,
    - when G = 18, L = 0.4,

    Prove: 
    The minimum number of training iterations required for the learning rate to decay to below 0.1 (excluding 0.1) is 130.
-/
theorem min_training_iterations
  (L0 : ℝ) (G0 : ℝ) (D : ℝ) (G : ℝ) (L : ℝ)
  (h1 : L0 = 0.5)
  (h2 : G0 = 18)
  (h3 : L = 0.4)
  (h4 : G = 18)
  (h5 : L0 * D^(G / G0) = 0.4)
  : ∃ G, G ≥ 130 ∧ L0 * D^(G / G0) < 0.1 := sorry

end min_training_iterations_l107_107628


namespace triangle_side_lengths_l107_107203

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end triangle_side_lengths_l107_107203


namespace probability_adjacent_vertices_in_decagon_l107_107091

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l107_107091


namespace similar_triangles_area_ratio_l107_107501

theorem similar_triangles_area_ratio (ratio_angles : ℕ) (area_larger : ℕ) (h_ratio : ratio_angles = 3) (h_area_larger : area_larger = 400) :
  ∃ area_smaller : ℕ, area_smaller = 36 :=
by
  sorry

end similar_triangles_area_ratio_l107_107501


namespace decagon_adjacent_probability_l107_107094

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l107_107094


namespace count_numbers_with_perfect_square_factors_l107_107680

open Finset

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ k * k ∣ n

def count_elements_with_perfect_square_factor_other_than_one (s : Finset ℕ) : ℕ :=
  s.filter has_perfect_square_factor_other_than_one).card

theorem count_numbers_with_perfect_square_factors :
  count_elements_with_perfect_square_factor_other_than_one (range 101) = 39 := 
begin
  -- initialization and calculation steps go here
  sorry
end

end count_numbers_with_perfect_square_factors_l107_107680


namespace triangle_possible_sides_l107_107221

theorem triangle_possible_sides : 
  let unknown_side_lengths := {x : ℤ | 3 < x ∧ x < 13}
  in set.finite unknown_side_lengths ∧ set.card unknown_side_lengths = 9 :=
by
  sorry

end triangle_possible_sides_l107_107221


namespace expand_binomials_l107_107639

theorem expand_binomials : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by 
  sorry

end expand_binomials_l107_107639


namespace parabola_tangents_coprime_l107_107546

theorem parabola_tangents_coprime {d e f : ℤ} (hd : d ≠ 0) (he : e ≠ 0)
  (h_coprime: Int.gcd (Int.gcd d e) f = 1)
  (h_tangent1 : d^2 - 4 * e * (2 * e - f) = 0)
  (h_tangent2 : (e + d)^2 - 4 * d * (8 * d - f) = 0) :
  d + e + f = 8 := by
  sorry

end parabola_tangents_coprime_l107_107546


namespace cannot_determine_x_l107_107771

theorem cannot_determine_x
  (n m : ℝ) (x : ℝ)
  (h1 : n + m = 8) 
  (h2 : n * x + m * (1/5) = 1) : true :=
by {
  sorry
}

end cannot_determine_x_l107_107771


namespace part_a_part_b_l107_107539

def can_cut_into_equal_dominoes (n : ℕ) : Prop :=
  ∃ horiz_vert_dominoes : ℕ × ℕ,
    n % 2 = 1 ∧
    (n * n - 1) / 2 = horiz_vert_dominoes.1 + horiz_vert_dominoes.2 ∧
    horiz_vert_dominoes.1 = horiz_vert_dominoes.2

theorem part_a : can_cut_into_equal_dominoes 101 :=
by {
  sorry
}

theorem part_b : ¬can_cut_into_equal_dominoes 99 :=
by {
  sorry
}

end part_a_part_b_l107_107539


namespace no_intersection_l107_107887

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l107_107887


namespace expand_product_l107_107634

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 :=
by
  sorry

end expand_product_l107_107634


namespace ivy_stripping_days_l107_107989

theorem ivy_stripping_days :
  ∃ (days_needed : ℕ), (days_needed * (6 - 2) = 40) ∧ days_needed = 10 :=
by {
  use 10,
  split,
  { simp,
    norm_num,
  },
  { simp }
}

end ivy_stripping_days_l107_107989


namespace probability_odd_divisor_15_factorial_l107_107074

theorem probability_odd_divisor_15_factorial :
  let number_of_divisors_15_fact : ℕ := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  let number_of_odd_divisors_15_fact : ℕ := (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  (number_of_odd_divisors_15_fact : ℝ) / (number_of_divisors_15_fact : ℝ) = 1 / 12 :=
by
  sorry

end probability_odd_divisor_15_factorial_l107_107074


namespace circle_intersection_area_l107_107287

theorem circle_intersection_area
  (r : ℝ)
  (θ : ℝ)
  (a b c : ℝ)
  (h_r : r = 5)
  (h_θ : θ = π / 2)
  (h_expr : a * Real.sqrt b + c * π = 5 * 5 * π / 2 - 5 * 5 * Real.sqrt 3 / 2 ) :
  a + b + c = -9.5 :=
by
  sorry

end circle_intersection_area_l107_107287


namespace part_a_int_values_part_b_int_values_l107_107182

-- Part (a)
theorem part_a_int_values (n : ℤ) :
  ∃ k : ℤ, (n^4 + 3) = k * (n^2 + n + 1) ↔ n = -3 ∨ n = -1 ∨ n = 0 :=
sorry

-- Part (b)
theorem part_b_int_values (n : ℤ) :
  ∃ m : ℤ, (n^3 + n + 1) = m * (n^2 - n + 1) ↔ n = 0 ∨ n = 1 :=
sorry

end part_a_int_values_part_b_int_values_l107_107182


namespace problem_n6_l107_107763

def is_fibonacci (n : ℕ) : Prop :=
  ∃ k : ℕ, n = fibonacci k

theorem problem_n6 (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ∃ k : ℕ, (a + 1) / b + (b + 1) / a = k) :
  is_fibonacci ((a + b) / gcd a b ^ 2) :=
by
  sorry

end problem_n6_l107_107763


namespace triangle_side_lengths_count_l107_107199

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end triangle_side_lengths_count_l107_107199


namespace obtuse_triangle_condition_l107_107497

theorem obtuse_triangle_condition
  (a b c : ℝ) 
  (h : ∃ A B C : ℝ, A + B + C = 180 ∧ A > 90 ∧ a^2 + b^2 - c^2 < 0)
  : (∃ A B C : ℝ, A + B + C = 180 ∧ A > 90 → a^2 + b^2 - c^2 < 0) := 
sorry

end obtuse_triangle_condition_l107_107497


namespace remainder_500th_in_T_l107_107045

def sequence_T (n : ℕ) : ℕ := sorry -- Assume a definition for the sequence T where n represents the position and the sequence contains numbers having exactly 9 ones in their binary representation.

theorem remainder_500th_in_T :
  (sequence_T 500) % 500 = 191 := 
sorry

end remainder_500th_in_T_l107_107045


namespace shop_weekly_earnings_l107_107155

theorem shop_weekly_earnings
  (price_women: ℕ := 18)
  (price_men: ℕ := 15)
  (time_open_hours: ℕ := 12)
  (minutes_per_hour: ℕ := 60)
  (weekly_days: ℕ := 7)
  (sell_rate_women: ℕ := 30)
  (sell_rate_men: ℕ := 40) :
  (time_open_hours * (minutes_per_hour / sell_rate_women) * price_women +
   time_open_hours * (minutes_per_hour / sell_rate_men) * price_men) * weekly_days = 4914 := 
sorry

end shop_weekly_earnings_l107_107155


namespace boxes_of_bolts_purchased_l107_107181

theorem boxes_of_bolts_purchased 
  (bolts_per_box : ℕ) 
  (nuts_per_box : ℕ) 
  (num_nut_boxes : ℕ) 
  (leftover_bolts : ℕ) 
  (leftover_nuts : ℕ) 
  (total_bolts_nuts_used : ℕ)
  (B : ℕ) :
  bolts_per_box = 11 →
  nuts_per_box = 15 →
  num_nut_boxes = 3 →
  leftover_bolts = 3 →
  leftover_nuts = 6 →
  total_bolts_nuts_used = 113 →
  B = 7 :=
by
  intros
  sorry

end boxes_of_bolts_purchased_l107_107181


namespace prob_divisible_by_5_of_digits_ending_in_7_l107_107746

theorem prob_divisible_by_5_of_digits_ending_in_7 :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000 ∧ N % 10 = 7) → (0 : ℚ) = 0 :=
by
  intro N
  sorry

end prob_divisible_by_5_of_digits_ending_in_7_l107_107746


namespace min_value_of_reciprocal_sum_l107_107830

theorem min_value_of_reciprocal_sum (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + y = 2) : 
  ∃ (z : ℝ), z = (1 / x + 1 / y) ∧ z = (3 / 2 + Real.sqrt 2) :=
sorry

end min_value_of_reciprocal_sum_l107_107830


namespace length_of_platform_is_300_meters_l107_107776

-- Definitions used in the proof
def kmph_to_mps (v: ℕ) : ℕ := (v * 1000) / 3600

def speed := kmph_to_mps 72

def time_cross_man := 15

def length_train := speed * time_cross_man

def time_cross_platform := 30

def total_distance_cross_platform := speed * time_cross_platform

def length_platform := total_distance_cross_platform - length_train

theorem length_of_platform_is_300_meters :
  length_platform = 300 :=
by
  sorry

end length_of_platform_is_300_meters_l107_107776


namespace christopher_strolled_5_miles_l107_107162

theorem christopher_strolled_5_miles (s t : ℝ) (hs : s = 4) (ht : t = 1.25) : s * t = 5 :=
by
  rw [hs, ht]
  norm_num

end christopher_strolled_5_miles_l107_107162


namespace gcd_of_459_and_357_l107_107336

theorem gcd_of_459_and_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_of_459_and_357_l107_107336


namespace lee_charge_per_action_figure_l107_107543

def cost_of_sneakers : ℕ := 90
def amount_saved : ℕ := 15
def action_figures_sold : ℕ := 10
def amount_left_after_purchase : ℕ := 25
def amount_charged_per_action_figure : ℕ := 10

theorem lee_charge_per_action_figure :
  (cost_of_sneakers - amount_saved + amount_left_after_purchase = 
  action_figures_sold * amount_charged_per_action_figure) :=
by
  -- The proof steps will go here, but they are not required in the statement.
  sorry

end lee_charge_per_action_figure_l107_107543


namespace opposite_of_2023_l107_107947

def opposite (n x : ℤ) := n + x = 0 

theorem opposite_of_2023 : ∃ x : ℤ, opposite 2023 x ∧ x = -2023 := 
by
  sorry

end opposite_of_2023_l107_107947


namespace table_can_be_zeroed_out_l107_107243

open Matrix

-- Define the dimensions of the table
def m := 8
def n := 5

-- Define the operation of doubling all elements in a row
def double_row (table : Matrix (Fin m) (Fin n) ℕ) (i : Fin m) : Matrix (Fin m) (Fin n) ℕ :=
  fun i' j => if i' = i then 2 * table i' j else table i' j

-- Define the operation of subtracting one from all elements in a column
def subtract_one_column (table : Matrix (Fin m) (Fin n) ℕ) (j : Fin n) : Matrix (Fin m) (Fin n) ℕ :=
  fun i j' => if j' = j then table i j' - 1 else table i j'

-- The main theorem stating that it is possible to transform any table to a table of all zeros
theorem table_can_be_zeroed_out (table : Matrix (Fin m) (Fin n) ℕ) : 
  ∃ (ops : List (Matrix (Fin m) (Fin n) ℕ → Matrix (Fin m) (Fin n) ℕ)), 
    (ops.foldl (fun t op => op t) table) = fun _ _ => 0 :=
sorry

end table_can_be_zeroed_out_l107_107243


namespace jellybean_count_l107_107804

theorem jellybean_count (x : ℝ) (h : (0.75^3) * x = 27) : x = 64 :=
sorry

end jellybean_count_l107_107804


namespace animal_count_l107_107435

variable (H C D : Nat)

theorem animal_count :
  (H + C + D = 72) → 
  (2 * H + 4 * C + 2 * D = 212) → 
  (C = 34) → 
  (H + D = 38) :=
by
  intros h1 h2 hc
  sorry

end animal_count_l107_107435


namespace expand_polynomial_l107_107643

theorem expand_polynomial (x : ℝ) :
  (x + 4) * (x - 9) = x^2 - 5 * x - 36 := 
sorry

end expand_polynomial_l107_107643


namespace line_circle_no_intersection_l107_107850

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l107_107850


namespace horizontal_shift_equivalence_l107_107575

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (x - Real.pi / 6)
noncomputable def resulting_function (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6)

theorem horizontal_shift_equivalence :
  ∀ x : ℝ, resulting_function x = original_function (x + Real.pi / 3) :=
by sorry

end horizontal_shift_equivalence_l107_107575


namespace unique_fraction_difference_l107_107823

theorem unique_fraction_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) :
  (1 / x) - (1 / y) = (y - x) / (x * y) :=
by sorry

end unique_fraction_difference_l107_107823


namespace num_pieces_l107_107682

theorem num_pieces (total_length : ℝ) (piece_length : ℝ) 
  (h1: total_length = 253.75) (h2: piece_length = 0.425) :
  ⌊total_length / piece_length⌋ = 597 :=
by
  rw [h1, h2]
  sorry

end num_pieces_l107_107682


namespace solution_set_inequality_l107_107289

theorem solution_set_inequality (a b : ℝ)
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → x^2 + a * x + b ≤ 0) :
  a * b = 6 :=
by {
  sorry
}

end solution_set_inequality_l107_107289


namespace triangle_side_lengths_count_l107_107201

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end triangle_side_lengths_count_l107_107201


namespace lab_tech_ratio_l107_107403

theorem lab_tech_ratio (U T C : ℕ) (hU : U = 12) (hC : C = 6 * U) (hT : T = (C + U) / 14) :
  (T : ℚ) / U = 1 / 2 :=
by
  sorry

end lab_tech_ratio_l107_107403


namespace expand_binomials_l107_107640

theorem expand_binomials : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by 
  sorry

end expand_binomials_l107_107640


namespace goblin_treasure_l107_107293

theorem goblin_treasure : 
  (∃ d : ℕ, 8000 + 300 * d = 5000 + 500 * d) ↔ ∃ (d : ℕ), d = 15 :=
by
  sorry

end goblin_treasure_l107_107293


namespace jade_tower_levels_l107_107378

theorem jade_tower_levels (total_pieces : ℕ) (pieces_per_level : ℕ) (pieces_left : ℕ) 
  (hypo1 : total_pieces = 100) (hypo2 : pieces_per_level = 7) (hypo3 : pieces_left = 23) : 
  (total_pieces - pieces_left) / pieces_per_level = 11 :=
by
  have h1 : 100 - 23 = 77, sorry
  have h2 : 77 / 7 = 11, sorry
  exact h2

end jade_tower_levels_l107_107378


namespace determinant_of_matrixA_l107_107632

variable (x : ℝ)

def matrixA : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x + 2, x, x], ![x, x + 2, x], ![x, x, x + 2]]

theorem determinant_of_matrixA : Matrix.det (matrixA x) = 8 * x + 8 := by
  sorry

end determinant_of_matrixA_l107_107632


namespace smallest_solution_of_quadratic_eq_l107_107297

theorem smallest_solution_of_quadratic_eq : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ < x₂) ∧ (x₁^2 + 10 * x₁ - 40 = 0) ∧ (x₂^2 + 10 * x₂ - 40 = 0) ∧ x₁ = -8 :=
by {
  sorry
}

end smallest_solution_of_quadratic_eq_l107_107297


namespace relationship_among_a_ab_ab2_l107_107703

theorem relationship_among_a_ab_ab2 (a b : ℝ) (h_a : a < 0) (h_b1 : -1 < b) (h_b2 : b < 0) :
  a < a * b ∧ a * b < a * b^2 :=
by
  sorry

end relationship_among_a_ab_ab2_l107_107703


namespace sum_y_coords_l107_107623

theorem sum_y_coords (h1 : ∃(y : ℝ), (0 + 3)^2 + (y - 5)^2 = 64) : 
  ∃ y1 y2 : ℝ, y1 + y2 = 10 ∧ (0, y1) ∈ ({ p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 5)^2 = 64 }) ∧ 
                            (0, y2) ∈ ({ p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 5)^2 = 64 }) := 
by
  sorry

end sum_y_coords_l107_107623


namespace probability_both_in_picture_l107_107274

-- Define the conditions
def completes_lap (laps_time: ℕ) (time: ℕ) : ℕ := time / laps_time

def position_into_lap (laps_time: ℕ) (time: ℕ) : ℕ := time % laps_time

-- Define the positions of Rachel and Robert
def rachel_position (time: ℕ) : ℚ :=
  let rachel_lap_time := 100
  let laps_completed := completes_lap rachel_lap_time time
  let time_into_lap := position_into_lap rachel_lap_time time
  (laps_completed : ℚ) + (time_into_lap : ℚ) / rachel_lap_time

def robert_position (time: ℕ) : ℚ :=
  let robert_lap_time := 70
  let laps_completed := completes_lap robert_lap_time time
  let time_into_lap := position_into_lap robert_lap_time time
  (laps_completed : ℚ) + (time_into_lap : ℚ) / robert_lap_time

-- Define the probability that both are in the picture
theorem probability_both_in_picture :
  let rachel_lap_time := 100
  let robert_lap_time := 70
  let start_time := 720
  let end_time := 780
  ∃ (overlap_time: ℚ) (total_time: ℚ),
    overlap_time / total_time = 1 / 16 :=
sorry

end probability_both_in_picture_l107_107274


namespace who_scored_full_marks_l107_107149

-- Define students and their statements
inductive Student
| A | B | C

open Student

def scored_full_marks (s : Student) : Prop :=
  match s with
  | A => true
  | B => true
  | C => true

def statement_A : Prop := scored_full_marks A
def statement_B : Prop := ¬ scored_full_marks C
def statement_C : Prop := statement_B

-- Given conditions
def exactly_one_lied (a b c : Prop) : Prop :=
  (a ∧ ¬ b ∧ ¬ c) ∨ (¬ a ∧ b ∧ ¬ c) ∨ (¬ a ∧ ¬ b ∧ c)

-- Main proof statement: Prove that B scored full marks
theorem who_scored_full_marks (h : exactly_one_lied statement_A statement_B statement_C) : scored_full_marks B :=
sorry

end who_scored_full_marks_l107_107149


namespace simplify_expression_l107_107934

variable (x : ℝ)

theorem simplify_expression :
  2 * x * (4 * x^2 - 3 * x + 1) - 4 * (2 * x^2 - 3 * x + 5) =
  8 * x^3 - 14 * x^2 + 14 * x - 20 := 
  sorry

end simplify_expression_l107_107934


namespace probability_of_three_white_balls_equals_8_over_65_l107_107138

noncomputable def probability_three_white_balls (n_white n_black : ℕ) (draws : ℕ) : ℚ :=
  (Nat.choose n_white draws : ℚ) / Nat.choose (n_white + n_black) draws

theorem probability_of_three_white_balls_equals_8_over_65 :
  probability_three_white_balls 8 7 3 = 8 / 65 :=
by
  sorry

end probability_of_three_white_balls_equals_8_over_65_l107_107138


namespace inverse_value_at_2_l107_107340

noncomputable def f (x : ℝ) : ℝ := x / (2 * x + 1)

noncomputable def f_inv (x : ℝ) : ℝ := x / (1 - 2 * x)

theorem inverse_value_at_2 :
  f_inv 2 = -2/3 := by
  sorry

end inverse_value_at_2_l107_107340


namespace total_children_correct_l107_107332

def blocks : ℕ := 9
def children_per_block : ℕ := 6
def total_children : ℕ := blocks * children_per_block

theorem total_children_correct : total_children = 54 := by
  sorry

end total_children_correct_l107_107332


namespace rectangle_total_area_l107_107242

-- Let s be the side length of the smaller squares
variable (s : ℕ)

-- Define the areas of the squares
def smaller_square_area := s ^ 2
def larger_square_area := (3 * s) ^ 2

-- Define the total_area
def total_area : ℕ := 2 * smaller_square_area s + larger_square_area s

-- Assert the total area of the rectangle ABCD is 11s^2
theorem rectangle_total_area (s : ℕ) : total_area s = 11 * s ^ 2 := 
by 
  -- the proof is skipped
  sorry

end rectangle_total_area_l107_107242


namespace probability_adjacent_vertices_in_decagon_l107_107108

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l107_107108


namespace quadratic_roots_l107_107002

theorem quadratic_roots (p q r : ℝ) (h : p ≠ q) (k : ℝ) :
  (p * (q - r) * (-1)^2 + q * (r - p) * (-1) + r * (p - q) = 0) →
  ((p * (q - r)) * k^2 + (q * (r - p)) * k + r * (p - q) = 0) →
  k = - (r * (p - q)) / (p * (q - r)) :=
by
  sorry

end quadratic_roots_l107_107002


namespace juliet_supporters_in_capulet_rate_l107_107131

theorem juliet_supporters_in_capulet_rate :
  ∀ (P : ℕ),
  ((6 / 9) * P) * 0.8 = (16 / 30) * P ∧
  ((1 / 3) * P) * 0.7 = (7 / 30) * P ∧
  ((6 / 9) * P) * 0.2 = (4 / 30) * P ∧
  ((7 / 30) * P + (4 / 30) * P) = (11 / 30) * P →
  ((7 / 30) * P / (11 / 30) * P) * 100 = 64 :=
by
  sorry

end juliet_supporters_in_capulet_rate_l107_107131


namespace minimal_divisors_at_kth_place_l107_107307

open Nat

theorem minimal_divisors_at_kth_place (n k : ℕ) (hnk : n ≥ k) (S : ℕ) (hS : ∃ d : ℕ, d ≥ n ∧ d = S ∧ ∀ i, i ≤ d → exists m, m = d):
  ∃ (min_div : ℕ), min_div = ⌈ (n : ℝ) / k ⌉ :=
by
  sorry

end minimal_divisors_at_kth_place_l107_107307


namespace swim_distance_l107_107324

theorem swim_distance 
  (v c d : ℝ)
  (h₁ : c = 2)
  (h₂ : (d / (v + c) = 5))
  (h₃ : (25 / (v - c) = 5)) :
  d = 45 :=
by
  sorry

end swim_distance_l107_107324


namespace min_value_of_z_l107_107659

-- Define the conditions and objective function
def constraints (x y : ℝ) : Prop :=
  (y ≥ x + 2) ∧ 
  (x + y ≤ 6) ∧ 
  (x ≥ 1)

def z (x y : ℝ) : ℝ :=
  2 * |x - 2| + |y|

-- The formal theorem stating the minimum value of z under the given constraints
theorem min_value_of_z : ∃ x y : ℝ, constraints x y ∧ z x y = 4 :=
sorry

end min_value_of_z_l107_107659


namespace equivalence_sufficient_necessary_l107_107153

-- Definitions for conditions
variables (A B : Prop)

-- Statement to prove
theorem equivalence_sufficient_necessary :
  (A → B) ↔ (¬B → ¬A) :=
by sorry

end equivalence_sufficient_necessary_l107_107153


namespace prove_m_value_l107_107516

theorem prove_m_value (m : ℕ) : 8^4 = 4^m → m = 6 := by
  sorry

end prove_m_value_l107_107516


namespace speed_with_stream_l107_107769

variable (V_m V_s : ℝ)

def against_speed : Prop := V_m - V_s = 13
def still_water_rate : Prop := V_m = 6

theorem speed_with_stream (h1 : against_speed V_m V_s) (h2 : still_water_rate V_m) : V_m + V_s = 13 := 
sorry

end speed_with_stream_l107_107769


namespace total_revenue_correct_l107_107341

def original_price_sneakers : ℝ := 80
def discount_sneakers : ℝ := 0.25
def pairs_sneakers_sold : ℕ := 2

def original_price_sandals : ℝ := 60
def discount_sandals : ℝ := 0.35
def pairs_sandals_sold : ℕ := 4

def original_price_boots : ℝ := 120
def discount_boots : ℝ := 0.40
def pairs_boots_sold : ℕ := 11

def calculate_total_revenue : ℝ := 
  let revenue_sneakers := pairs_sneakers_sold * (original_price_sneakers * (1 - discount_sneakers))
  let revenue_sandals := pairs_sandals_sold * (original_price_sandals * (1 - discount_sandals))
  let revenue_boots := pairs_boots_sold * (original_price_boots * (1 - discount_boots))
  revenue_sneakers + revenue_sandals + revenue_boots

theorem total_revenue_correct : calculate_total_revenue = 1068 := by
  sorry

end total_revenue_correct_l107_107341


namespace quadratic_coeff_sum_l107_107742

theorem quadratic_coeff_sum {a b c : ℝ} (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 1) * (x - 5))
    (h2 : a * 3^2 + b * 3 + c = 36) : a + b + c = 0 :=
by
  sorry

end quadratic_coeff_sum_l107_107742


namespace mod_equivalence_l107_107593

theorem mod_equivalence (n : ℤ) (hn₁ : 0 ≤ n) (hn₂ : n < 23) (hmod : -250 % 23 = n % 23) : n = 3 := by
  sorry

end mod_equivalence_l107_107593


namespace coords_with_respect_to_origin_l107_107534

/-- Given that the coordinates of point A are (-1, 2), prove that the coordinates of point A
with respect to the origin in the plane rectangular coordinate system xOy are (-1, 2). -/
theorem coords_with_respect_to_origin (A : (ℝ × ℝ)) (h : A = (-1, 2)) : A = (-1, 2) :=
sorry

end coords_with_respect_to_origin_l107_107534


namespace cary_strips_ivy_l107_107990

variable (strip_per_day : ℕ) (grow_per_night : ℕ) (total_ivy : ℕ)

theorem cary_strips_ivy (h1 : strip_per_day = 6) (h2 : grow_per_night = 2) (h3 : total_ivy = 40) :
  (total_ivy / (strip_per_day - grow_per_night)) = 10 := by
  sorry

end cary_strips_ivy_l107_107990


namespace greatest_possible_value_q_minus_r_l107_107394

theorem greatest_possible_value_q_minus_r : ∃ q r : ℕ, 1025 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q - r = 31 :=
by {
  sorry
}

end greatest_possible_value_q_minus_r_l107_107394


namespace marquita_garden_width_l107_107711

theorem marquita_garden_width
  (mancino_gardens : ℕ) (marquita_gardens : ℕ)
  (mancino_length mancnio_width marquita_length total_area : ℕ)
  (h1 : mancino_gardens = 3)
  (h2 : mancino_length = 16)
  (h3 : mancnio_width = 5)
  (h4 : marquita_gardens = 2)
  (h5 : marquita_length = 8)
  (h6 : total_area = 304) :
  ∃ (marquita_width : ℕ), marquita_width = 4 :=
by
  sorry

end marquita_garden_width_l107_107711


namespace tangent_line_at_point_l107_107001

theorem tangent_line_at_point (x y : ℝ) (h : y = x / (x - 2)) (hx : x = 1) (hy : y = -1) : y = -2 * x + 1 :=
sorry

end tangent_line_at_point_l107_107001


namespace decagon_adjacent_probability_l107_107095

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l107_107095


namespace union_of_sets_l107_107917

open Set

variable (a b : ℕ)

noncomputable def M : Set ℕ := {3, 2 * a}
noncomputable def N : Set ℕ := {a, b}

theorem union_of_sets (h : M a ∩ N a b = {2}) : M a ∪ N a b = {1, 2, 3} :=
by
  -- skipped proof
  sorry

end union_of_sets_l107_107917


namespace find_g_six_l107_107941

noncomputable def g : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : g (x + y) = g x + g y
axiom g_five : g 5 = 6

theorem find_g_six : g 6 = 36/5 := 
by 
  -- proof to be filled in
  sorry

end find_g_six_l107_107941


namespace brazil_medal_fraction_closest_l107_107333

theorem brazil_medal_fraction_closest :
  let frac_win : ℚ := 23 / 150
  let frac_1_6 : ℚ := 1 / 6
  let frac_1_7 : ℚ := 1 / 7
  let frac_1_8 : ℚ := 1 / 8
  let frac_1_9 : ℚ := 1 / 9
  let frac_1_10 : ℚ := 1 / 10
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_6) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_8) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_9) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_10) :=
by
  sorry

end brazil_medal_fraction_closest_l107_107333


namespace problem_l107_107499

-- Define the functions f and g with their properties
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Express the given conditions in Lean
axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom g_odd : ∀ x : ℝ, g (-x) = -g x
axiom g_def : ∀ x : ℝ, g x = f (x - 1)
axiom f_at_2 : f 2 = 2

-- What we need to prove
theorem problem : f 2014 = 2 := 
by sorry

end problem_l107_107499


namespace least_five_digit_integer_congruent_3_mod_17_l107_107751

theorem least_five_digit_integer_congruent_3_mod_17 : 
  ∃ n, n ≥ 10000 ∧ n % 17 = 3 ∧ ∀ m, (m ≥ 10000 ∧ m % 17 = 3) → n ≤ m := 
sorry

end least_five_digit_integer_congruent_3_mod_17_l107_107751


namespace math_problem_l107_107362

theorem math_problem (m : ℝ) (h : m^2 - m = 2) : (m - 1)^2 + (m + 2) * (m - 2) = 1 := 
by sorry

end math_problem_l107_107362


namespace smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum_l107_107075

theorem smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum :
  ∃ (a : ℤ), (∃ (l : List ℤ), l.length = 50 ∧ List.prod l = 0 ∧ 0 < List.sum l ∧ List.sum l = 25) :=
by
  sorry

end smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum_l107_107075


namespace geometric_sequence_condition_l107_107192

variable (a_1 : ℝ) (q : ℝ)

noncomputable def geometric_sum (n : ℕ) : ℝ :=
if q = 1 then a_1 * n else a_1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_condition (a_1 : ℝ) (q : ℝ) :
  (a_1 > 0) ↔ (geometric_sum a_1 q 2017 > 0) :=
by sorry

end geometric_sequence_condition_l107_107192


namespace competition_result_l107_107559

theorem competition_result :
  (∀ (Olya Oleg Pasha : Nat), Olya = 1 ∨ Oleg = 1 ∨ Pasha = 1) → 
  (∀ (Olya Oleg Pasha : Nat), (Olya = 1 ∨ Olya = 3) → false) →
  (∀ (Olya Oleg Pasha : Nat), Oleg ≠ 1) →
  (∀ (Olya Oleg Pasha : Nat), (Olya = Oleg ∧ Olya ≠ Pasha)) →
  ∃ (Olya Oleg Pasha : Nat), Oleg = 1 ∧ Pasha = 2 ∧ Olya = 3 :=
begin
  assume each_claims_first,
  assume olya_odd_places_false,
  assume oleg_truthful,
  assume truth_liar_cond,
  sorry
end

end competition_result_l107_107559


namespace min_k_squared_floor_l107_107343

open Nat

theorem min_k_squared_floor (n : ℕ) :
  (∀ k : ℕ, k >= 1 → k^2 + (n / k^2) ≥ 1991) ∧
  (∃ k : ℕ, k >= 1 ∧ k^2 + (n / k^2) < 1992) ↔
  1024 * 967 ≤ n ∧ n ≤ 1024 * 967 + 1023 := 
by
  sorry

end min_k_squared_floor_l107_107343


namespace lending_rate_is_8_percent_l107_107439

-- Define all given conditions.
def principal₁ : ℝ := 5000
def time₁ : ℝ := 2
def rate₁ : ℝ := 4  -- in percentage
def gain_per_year : ℝ := 200

-- Prove that the interest rate for lending is 8%
theorem lending_rate_is_8_percent :
  ∃ (rate₂ : ℝ), rate₂ = 8 :=
by
  let interest₁ := principal₁ * rate₁ * time₁ / 100
  let interest_per_year₁ := interest₁ / time₁
  let total_interest_received_per_year := gain_per_year + interest_per_year₁
  let rate₂ := (total_interest_received_per_year * 100) / principal₁
  use rate₂
  sorry

end lending_rate_is_8_percent_l107_107439


namespace train_crossing_tree_time_l107_107132

noncomputable def time_to_cross_platform (train_length : ℕ) (platform_length : ℕ) (time_to_cross_platform : ℕ) : ℕ :=
  (train_length + platform_length) / time_to_cross_platform

noncomputable def time_to_cross_tree (train_length : ℕ) (speed : ℕ) : ℕ :=
  train_length / speed

theorem train_crossing_tree_time :
  ∀ (train_length platform_length time platform_time speed : ℕ),
  train_length = 1200 →
  platform_length = 900 →
  platform_time = 210 →
  speed = (train_length + platform_length) / platform_time →
  time = train_length / speed →
  time = 120 :=
by
  intros train_length platform_length time platform_time speed h_train_length h_platform_length h_platform_time h_speed h_time
  sorry

end train_crossing_tree_time_l107_107132


namespace range_of_x_l107_107482

theorem range_of_x (x p : ℝ) (hp : 0 ≤ p ∧ p ≤ 4) :
  (x^2 + p*x > 4*x + p - 3) ↔ (x > 3 ∨ x < -1) :=
by {
  sorry
}

end range_of_x_l107_107482


namespace max_digit_sum_watch_l107_107317

def digit_sum (n : Nat) : Nat :=
  (n / 10) + (n % 10)

theorem max_digit_sum_watch :
  ∃ (h m : Nat), (1 <= h ∧ h <= 12) ∧ (0 <= m ∧ m <= 59) 
  ∧ (digit_sum h + digit_sum m = 23) :=
by 
  sorry

end max_digit_sum_watch_l107_107317


namespace min_value_expression_l107_107662

theorem min_value_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : ∀ x y : ℝ, x + y + a = 0 → (x - b)^2 + (y - 1)^2 = 2) : 
    (∃ c : ℝ,  c = 4 ∧ ∀ a b : ℝ, (0 < a → 0 < b → x + y + a = 0 → (x - b)^2 + (y - 1)^2 = 2 →  (3 - 2 * b)^2 / (2 * a) ≥ c)) :=
by
  sorry

end min_value_expression_l107_107662


namespace largest_possible_value_of_c_l107_107261

theorem largest_possible_value_of_c (c : ℚ) : (3 * c + 4) * (c - 2) = 9 * c → c ≤ 4 :=
by
  intro h
  have : (3 * c + 4) * (c - 2) = 3 * c^2 - 6 * c + 4 * c - 8 := 
    calc 
    (3 * c + 4) * (c - 2) = (3 * c) * (c - 2) + 4 * (c - 2) : by ring
                         ... = (3 * c) * c - (3 * c) * 2 + 4 * c - 4 * 2 : by ring
                         ... = 3 * c^2 - 6 * c + 4 * c - 8 : by ring
  rw this at h
  have h2 : 3 * c^2 - 11 * c - 8 = 0 := by nlinarith
  sorry

end largest_possible_value_of_c_l107_107261


namespace line_circle_no_intersection_l107_107848

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l107_107848


namespace distance_between_A_and_B_l107_107724

theorem distance_between_A_and_B (v_A v_B d d' : ℝ)
  (h1 : v_B = 50)
  (h2 : (v_A - v_B) * 30 = d')
  (h3 : (v_A + v_B) * 6 = d) :
  d = 750 :=
sorry

end distance_between_A_and_B_l107_107724


namespace grandma_finishes_at_l107_107514

-- Definitions based on conditions
def fold_time_per_crane : Int := 3
def rest_time_per_crane : Int := 1
def start_time : Time := ⟨14, 30⟩  -- representing 2:30 PM in 24-hour format

-- Define the function to calculate the final time
def calculate_final_time (num_cranes : Nat) : Time :=
start_time + Time.Duration.mk (num_cranes * fold_time_per_crane + (num_cranes - 1) * rest_time_per_crane) 0

-- Statement to prove
theorem grandma_finishes_at : calculate_final_time 5 = ⟨14, 49⟩ := by
  sorry

end grandma_finishes_at_l107_107514


namespace canvas_bag_lower_carbon_solution_l107_107266

def canvas_bag_emission := 600 -- pounds of CO2
def plastic_bag_emission := 4 -- ounces of CO2 per bag
def bags_per_trip := 8 
def ounce_to_pound := 16 -- 16 ounces in a pound
def co2_trip := (plastic_bag_emission * bags_per_trip) / ounce_to_pound -- CO2 emission in pounds per trip

theorem canvas_bag_lower_carbon_solution : 
  co2_trip * 300 >= canvas_bag_emission :=
by
  unfold canvas_bag_emission plastic_bag_emission bags_per_trip ounce_to_pound co2_trip 
  sorry

end canvas_bag_lower_carbon_solution_l107_107266


namespace socks_count_l107_107915

theorem socks_count :
  ∃ (x y z : ℕ), x + y + z = 12 ∧ x + 3 * y + 4 * z = 24 ∧ 1 ≤ x ∧ 1 ≤ y ∧ 1 <= z ∧ x = 7 :=
by
  sorry

end socks_count_l107_107915


namespace parabola_min_value_sum_abc_zero_l107_107735

theorem parabola_min_value_sum_abc_zero
  (a b c : ℝ)
  (h1 : ∀ x y : ℝ, y = ax^2 + bx + c → y = 0 ∧ ((x = 1) ∨ (x = 5)))
  (h2 : ∃ x : ℝ, (∀ t : ℝ, ax^2 + bx + c ≤ t^2) ∧ (ax^2 + bx + c = 36)) :
  a + b + c = 0 :=
sorry

end parabola_min_value_sum_abc_zero_l107_107735


namespace remainder_of_num_five_element_subsets_with_two_consecutive_l107_107918

-- Define the set and the problem
noncomputable def num_five_element_subsets_with_two_consecutive (n : ℕ) : ℕ := 
  Nat.choose 14 5 - Nat.choose 10 5

-- Main Lean statement: prove the final condition
theorem remainder_of_num_five_element_subsets_with_two_consecutive :
  (num_five_element_subsets_with_two_consecutive 14) % 1000 = 750 :=
by
  -- Proof goes here
  sorry

end remainder_of_num_five_element_subsets_with_two_consecutive_l107_107918


namespace smallest_sphere_radius_l107_107484

theorem smallest_sphere_radius :
  ∃ (R : ℝ), (∀ (a b : ℝ), a = 14 → b = 12 → ∃ (h : ℝ), h = Real.sqrt (12^2 - (14 * Real.sqrt 2 / 2)^2) ∧ R = 7 * Real.sqrt 2 ∧ h ≤ R) :=
sorry

end smallest_sphere_radius_l107_107484


namespace Morse_code_distinct_symbols_l107_107241

-- Morse code sequences conditions
def MorseCodeSequence (n : ℕ) := n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4

-- Total number of distinct symbols calculation
def total_distinct_symbols : ℕ :=
  2 + 4 + 8 + 16

-- The theorem to prove
theorem Morse_code_distinct_symbols : total_distinct_symbols = 30 := by
  sorry

end Morse_code_distinct_symbols_l107_107241


namespace quotient_A_div_B_l107_107974

-- Define A according to the given conditions
def A : ℕ := (8 * 10) + (13 * 1)

-- Define B according to the given conditions
def B : ℕ := 30 - 9 - 9 - 9

-- Prove that the quotient of A divided by B is 31
theorem quotient_A_div_B : (A / B) = 31 := by
  sorry

end quotient_A_div_B_l107_107974


namespace solve_equations_l107_107003

theorem solve_equations :
  (∃ x : ℝ, (x + 2) ^ 3 + 1 = 0 ∧ x = -3) ∧
  (∃ x : ℝ, ((3 * x - 2) ^ 2 = 64 ∧ (x = 10/3 ∨ x = -2))) :=
by {
  -- Prove the existence of solutions for both problems
  sorry
}

end solve_equations_l107_107003


namespace percentage_of_left_handed_women_l107_107087

variable (x y : Nat) (h_ratio_rh_lh : 3 * x = 1 * x)
variable (h_ratio_men_women : 3 * y = 2 * y)
variable (h_rh_men_max : True)

theorem percentage_of_left_handed_women :
  (x / (4 * x)) * 100 = 25 :=
by sorry

end percentage_of_left_handed_women_l107_107087


namespace test_point_selection_l107_107245

theorem test_point_selection (x_1 x_2 : ℝ)
    (interval_begin interval_end : ℝ) (h_interval : interval_begin = 2 ∧ interval_end = 4)
    (h_better_result : x_1 < x_2 ∨ x_1 > x_2)
    (h_test_points : (x_1 = interval_begin + 0.618 * (interval_end - interval_begin) ∧ 
                     x_2 = interval_begin + interval_end - x_1) ∨ 
                    (x_1 = interval_begin + interval_end - (interval_begin + 0.618 * (interval_end - interval_begin)) ∧ 
                     x_2 = interval_begin + 0.618 * (interval_end - interval_begin)))
  : ∃ x_3, x_3 = 3.528 ∨ x_3 = 2.472 := by
    sorry

end test_point_selection_l107_107245


namespace canoe_downstream_speed_l107_107968

-- Definitions based on conditions
def upstream_speed : ℝ := 9  -- upspeed
def stream_speed : ℝ := 1.5  -- vspeed

-- Theorem to prove the downstream speed
theorem canoe_downstream_speed (V_c : ℝ) (V_d : ℝ) :
  (V_c - stream_speed = upstream_speed) →
  (V_d = V_c + stream_speed) →
  V_d = 12 := by 
  intro h1 h2
  sorry

end canoe_downstream_speed_l107_107968


namespace count_perfect_square_factors_l107_107679

noncomputable def has_perfect_square_factor (n : ℕ) (squares : Set ℕ) : Prop :=
  ∃ m ∈ squares, m ∣ n

theorem count_perfect_square_factors :
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  count.card = 41 :=
by
  let squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let range := Finset.range 100
  let count := range.filter (λ n, has_perfect_square_factor (n + 1) squares)
  have : count.card = 41 := sorry
  exact this

end count_perfect_square_factors_l107_107679


namespace cindy_correct_result_l107_107992

theorem cindy_correct_result (x : ℝ) (h: (x - 7) / 5 = 27) : (x - 5) / 7 = 20 :=
by
  sorry

end cindy_correct_result_l107_107992


namespace trig_expression_evaluation_l107_107013

theorem trig_expression_evaluation
  (α : ℝ)
  (h_tan_α : Real.tan α = 3) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / 
  (Real.cos (3 * π / 2 - α) + 2 * Real.cos (-π + α)) = -2 / 5 := by
  sorry

end trig_expression_evaluation_l107_107013


namespace rectangle_ratio_l107_107909

theorem rectangle_ratio (a b : ℝ) (side : ℝ) (M N : ℝ → ℝ) (P Q : ℝ → ℝ)
  (h_side : side = 4)
  (h_M : M 0 = 4 / 3 ∧ M 4 = 8 / 3)
  (h_N : N 0 = 4 / 3 ∧ N 4 = 8 / 3)
  (h_perpendicular : P 0 = Q 0 ∧ P 4 = Q 4)
  (h_area : side * side = 16) :
  let UV := 6 / 5
  let VW := 40 / 3
  UV / VW = 9 / 100 :=
sorry

end rectangle_ratio_l107_107909


namespace pow_two_sub_one_not_square_l107_107902

theorem pow_two_sub_one_not_square (n : ℕ) (h : n > 1) : ¬ ∃ k : ℕ, 2^n - 1 = k^2 := by
  sorry

end pow_two_sub_one_not_square_l107_107902


namespace count_perfect_square_factors_except_one_l107_107678

def has_perfect_square_factor_except_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m ≠ 1 ∧ m * m ∣ n

theorem count_perfect_square_factors_except_one :
  ∃ (count : ℕ), count = 40 ∧ 
    (count = (Finset.filter has_perfect_square_factor_except_one (Finset.range 101)).card) :=
by
  sorry

end count_perfect_square_factors_except_one_l107_107678


namespace intersection_A_compB_l107_107671

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 7}

-- Define the complement of B relative to ℝ
def comp_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 7}

-- State the main theorem to prove
theorem intersection_A_compB : A ∩ comp_B = {x | -3 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_A_compB_l107_107671


namespace binomial_expansion_sum_l107_107369

theorem binomial_expansion_sum (n : ℕ) (h : (2:ℕ)^n = 256) : n = 8 :=
sorry

end binomial_expansion_sum_l107_107369


namespace jellybean_count_l107_107806

theorem jellybean_count (x : ℝ) (h : (0.75^3) * x = 27) : x = 64 :=
sorry

end jellybean_count_l107_107806


namespace area_of_lune_l107_107449

/-- A theorem to calculate the area of the lune formed by two semicircles 
    with diameters 3 and 4 -/
theorem area_of_lune (r1 r2 : ℝ) (h1 : r1 = 3/2) (h2 : r2 = 4/2) :
  let area_larger_semicircle := (1 / 2) * Real.pi * r2^2,
      area_smaller_semicircle := (1 / 2) * Real.pi * r1^2,
      area_triangle := (1 / 2) * 4 * (3 / 2)
  in (area_larger_semicircle - (area_smaller_semicircle + area_triangle)) = ((7 / 4) * Real.pi - 3) :=
by
  sorry

end area_of_lune_l107_107449


namespace cookie_recipe_total_cups_l107_107692

theorem cookie_recipe_total_cups (r_butter : ℕ) (r_flour : ℕ) (r_sugar : ℕ) (sugar_cups : ℕ) 
  (h_ratio : r_butter = 1 ∧ r_flour = 2 ∧ r_sugar = 3) (h_sugar : sugar_cups = 9) : 
  r_butter * (sugar_cups / r_sugar) + r_flour * (sugar_cups / r_sugar) + sugar_cups = 18 := 
by 
  sorry

end cookie_recipe_total_cups_l107_107692


namespace tanya_work_time_l107_107276

theorem tanya_work_time (
    sakshi_work_time : ℝ := 20,
    tanya_efficiency : ℝ := 1.25
) : 
    let sakshi_rate : ℝ := 1 / sakshi_work_time in
    let tanya_rate : ℝ := tanya_efficiency * sakshi_rate in
    let tanya_work_time := 1 / tanya_rate in
    tanya_work_time = 16 :=
by
    let sakshi_rate : ℝ := 1 / sakshi_work_time
    let tanya_rate : ℝ := tanya_efficiency * sakshi_rate
    let tanya_time : ℝ := 1 / tanya_rate
    show tanya_time = 16
    sorry

end tanya_work_time_l107_107276


namespace sequence_general_formula_l107_107831

theorem sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (rec : ∀ n : ℕ, n > 0 → a n = n * (a (n + 1) - a n)) : 
  ∀ n, a n = n := 
by 
  sorry

end sequence_general_formula_l107_107831


namespace downstream_speed_is_45_l107_107319

-- Define the conditions
def upstream_speed := 35 -- The man can row upstream at 35 kmph
def still_water_speed := 40 -- The speed of the man in still water is 40 kmph

-- Define the speed of the stream based on the given conditions
def stream_speed := still_water_speed - upstream_speed 

-- Define the speed of the man rowing downstream
def downstream_speed := still_water_speed + stream_speed

-- The assertion to prove
theorem downstream_speed_is_45 : downstream_speed = 45 := by
  sorry

end downstream_speed_is_45_l107_107319


namespace f_zero_eq_one_f_pos_all_f_increasing_l107_107994

noncomputable def f : ℝ → ℝ := sorry

axiom f_nonzero : f 0 ≠ 0
axiom f_pos : ∀ x, 0 < x → 1 < f x
axiom f_mul : ∀ a b : ℝ, f (a + b) = f a * f b

theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem f_pos_all : ∀ x : ℝ, 0 < f x :=
sorry

theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ :=
sorry

end f_zero_eq_one_f_pos_all_f_increasing_l107_107994


namespace households_soap_usage_l107_107436

theorem households_soap_usage
  (total_households : ℕ)
  (neither : ℕ)
  (both : ℕ)
  (only_B_ratio : ℕ)
  (B := only_B_ratio * both) :
  total_households = 200 →
  neither = 80 →
  both = 40 →
  only_B_ratio = 3 →
  (total_households - neither - both - B = 40) :=
by
  intros
  sorry

end households_soap_usage_l107_107436


namespace find_ABC_l107_107935

variables (A B C D : ℕ)

-- Conditions
def non_zero_distinct_digits_less_than_7 : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A < 7 ∧ B < 7 ∧ C < 7 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

def ab_c_seven : Prop := 
  (A * 7 + B) + C = C * 7

def ab_ba_dc_seven : Prop :=
  (A * 7 + B) + (B * 7 + A) = D * 7 + C

-- Theorem to prove
theorem find_ABC 
  (h1 : non_zero_distinct_digits_less_than_7 A B C) 
  (h2 : ab_c_seven A B C) 
  (h3 : ab_ba_dc_seven A B C D) : 
  A * 100 + B * 10 + C = 516 :=
sorry

end find_ABC_l107_107935


namespace volunteer_comprehensive_score_l107_107429

theorem volunteer_comprehensive_score :
  let written_score := 90
  let trial_score := 94
  let interview_score := 92
  let written_weight := 0.30
  let trial_weight := 0.50
  let interview_weight := 0.20
  (written_score * written_weight + trial_score * trial_weight + interview_score * interview_weight = 92.4) := by
  sorry

end volunteer_comprehensive_score_l107_107429


namespace range_a_l107_107004

def f (x a : ℝ) := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

theorem range_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_a_l107_107004


namespace correct_condition_l107_107082

section proof_problem

variable (a : ℝ)

def cond1 : Prop := (a ^ 6 / a ^ 3 = a ^ 2)
def cond2 : Prop := (2 * a ^ 2 + 3 * a ^ 3 = 5 * a ^ 5)
def cond3 : Prop := (a ^ 4 * a ^ 2 = a ^ 8)
def cond4 : Prop := ((-a ^ 3) ^ 2 = a ^ 6)

theorem correct_condition : cond4 a :=
by
  sorry

end proof_problem

end correct_condition_l107_107082


namespace harry_travel_ratio_l107_107673

theorem harry_travel_ratio
  (bus_initial_time : ℕ)
  (bus_rest_time : ℕ)
  (total_travel_time : ℕ)
  (walking_time : ℕ := total_travel_time - (bus_initial_time + bus_rest_time))
  (bus_total_time : ℕ := bus_initial_time + bus_rest_time)
  (ratio : ℚ := walking_time / bus_total_time)
  (h1 : bus_initial_time = 15)
  (h2 : bus_rest_time = 25)
  (h3 : total_travel_time = 60)
  : ratio = (1 / 2) := 
sorry

end harry_travel_ratio_l107_107673


namespace parabola_latus_rectum_l107_107512

theorem parabola_latus_rectum (x p y : ℝ) (hp : p > 0) (h_eq : x^2 = 2 * p * y) (hl : y = -3) :
  p = 6 :=
by
  sorry

end parabola_latus_rectum_l107_107512


namespace expand_polynomial_l107_107641

theorem expand_polynomial (x : ℝ) :
  (x + 4) * (x - 9) = x^2 - 5 * x - 36 := 
sorry

end expand_polynomial_l107_107641


namespace correct_def_E_bar_C_plus_E_bar_Riemann_integral_E_bar_correct_Riesz_Lebesgue_integrability_C_plus_not_nonneg_bounded_l107_107707

-- Define relevant structures and concepts
noncomputable theory

def Omega := Set.Ioo 0 1
def BorelOmega := MeasurableSpace.comap (Subtype.val : Omega → ℝ) volume

structure StepFunction (ω : Omega) :=
(c : ℕ → ℝ)
(ω_incr : ∀ i, ∃ a b, (0:ℝ) ≤ a ∧ b ≤ 1 ∧ (a < b ∧ ω ∈ Ioo a b))

noncomputable def E_bar (ξ : StepFunction) : ℝ :=
sum (λ i, ξ.c i * (ω_incr i).2.1)

def C_plus (ξ : StepFunction) :=
∀ n, ∃ ξ_n : StepFunction, (ξ_n.c = ξ.c ∧ E_bar ξ_n < ∞) ∧ ∀ᵐ ω, ξ_n ω ⟶ ξ ω

theorem correct_def_E_bar_C_plus :
  ∀ ξ : StepFunction, (C_plus ξ) → ∃! (lim_n : ℕ → ℝ), lim (λ n, E_bar (ξ_n n)) = lim_n := sorry

theorem E_bar_Riemann_integral (ξ : StepFunction) :
  (RiemannIntegrable ξ) → (C_plus ξ) ∧ (E_bar ξ = ∫ x in Omega, ξ.val x) := sorry

theorem E_bar_correct (ξ : StepFunction) :
  ∀ ξ_+, ξ_- : StepFunction, (ξ = ξ_+ - ξ_-) → C_plus ξ_+ ∧ C_plus ξ_-  ∧ E_bar ξ = E_bar ξ_+ - E_bar ξ_- := sorry

theorem Riesz_Lebesgue_integrability (ξ : StepFunction) :
  (LebesgueIntegrable ξ) → ξ_val := sorry

theorem C_plus_not_nonneg_bounded (ξ : StepFunction) :
  ((∀ ω, 0 ≤ ξ ω) ∧ (∫ x in Omega, ξ.val x < ∞)) → ¬ C_plus ξ := sorry

end correct_def_E_bar_C_plus_E_bar_Riemann_integral_E_bar_correct_Riesz_Lebesgue_integrability_C_plus_not_nonneg_bounded_l107_107707


namespace greatest_odd_factors_l107_107714

theorem greatest_odd_factors (n : ℕ) : n < 200 ∧ (∃ m : ℕ, m * m = n) → n = 196 := by
  sorry

end greatest_odd_factors_l107_107714


namespace chip_notebook_packs_l107_107160

theorem chip_notebook_packs (pages_per_day : ℕ) (days_per_week : ℕ) (classes : ℕ) (sheets_per_pack : ℕ) (weeks : ℕ) :
  pages_per_day = 2 → days_per_week = 5 → classes = 5 → sheets_per_pack = 100 → weeks = 6 →
  (classes * pages_per_day * days_per_week * weeks) / sheets_per_pack = 3 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  sorry

end chip_notebook_packs_l107_107160


namespace max_fractions_with_integer_values_l107_107592

noncomputable def max_integer_valued_fractions (S : Set ℕ) : ℕ :=
  let fractions := { (a, b) | a ∈ S ∧ b ∈ S ∧ b ≠ 0 ∧ a % b = 0 }
  in fractions.to_finset.card

theorem max_fractions_with_integer_values (S : Set ℕ) (hS : S = {1, 2, ..., 22}) : max_integer_valued_fractions S = 10 :=
  sorry

end max_fractions_with_integer_values_l107_107592


namespace max_value_l107_107661

-- Definitions for conditions
variables {a b : ℝ}
variables (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (1 / b) = 2)

-- Statement of the theorem
theorem max_value : (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (1 / a) + (1 / b) = 2 ∧ ∀ y : ℝ,
  (1 / y) * ((2 / (y * (3 * y - 1)⁻¹)) + 1) ≤ 25 / 8) :=
sorry

end max_value_l107_107661


namespace car_journey_delay_l107_107969

theorem car_journey_delay (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time1 : ℝ) (time2 : ℝ) (delay : ℝ) :
  distance = 225 ∧ speed1 = 60 ∧ speed2 = 50 ∧ time1 = distance / speed1 ∧ time2 = distance / speed2 ∧ 
  delay = (time2 - time1) * 60 → delay = 45 :=
by
  sorry

end car_journey_delay_l107_107969


namespace coordinates_of_point_A_l107_107537

theorem coordinates_of_point_A (x y : ℤ) (h : x = -1 ∧ y = 2) : (x, y) = (-1, 2) :=
by {
  cases h,
  rw [h_left, h_right],
}

end coordinates_of_point_A_l107_107537


namespace percent_of_a_is_b_l107_107421

variable (a b c : ℝ)
variable (h1 : c = 0.20 * a) (h2 : c = 0.10 * b)

theorem percent_of_a_is_b : b = 2 * a :=
by sorry

end percent_of_a_is_b_l107_107421


namespace probability_of_adjacent_vertices_in_decagon_l107_107115

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l107_107115


namespace complex_ab_value_l107_107186

theorem complex_ab_value (a b : ℝ) (i : ℂ) (h_i : i = Complex.I) (h_z : a + b * i = (4 + 3 * i) * i) : a * b = -12 :=
by {
  sorry
}

end complex_ab_value_l107_107186


namespace triangle_area_l107_107950

theorem triangle_area (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39) (h4 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 270 :=
by
  sorry

end triangle_area_l107_107950


namespace line_circle_no_intersection_l107_107894

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l107_107894


namespace tory_earned_more_than_bert_l107_107463

open Real

noncomputable def bert_day1_earnings : ℝ :=
  let initial_sales := 12 * 18
  let discounted_sales := 3 * (18 - 0.15 * 18)
  let total_sales := initial_sales - 3 * 18 + discounted_sales
  total_sales * 0.95

noncomputable def tory_day1_earnings : ℝ :=
  let initial_sales := 15 * 20
  let discounted_sales := 5 * (20 - 0.10 * 20)
  let total_sales := initial_sales - 5 * 20 + discounted_sales
  total_sales * 0.95

noncomputable def bert_day2_earnings : ℝ :=
  let sales := 10 * 15
  (sales * 0.95) * 1.4

noncomputable def tory_day2_earnings : ℝ :=
  let sales := 8 * 18
  (sales * 0.95) * 1.4

noncomputable def bert_total_earnings : ℝ := bert_day1_earnings + bert_day2_earnings

noncomputable def tory_total_earnings : ℝ := tory_day1_earnings + tory_day2_earnings

noncomputable def earnings_difference : ℝ := tory_total_earnings - bert_total_earnings

theorem tory_earned_more_than_bert :
  earnings_difference = 71.82 := by
  sorry

end tory_earned_more_than_bert_l107_107463


namespace coordinates_of_point_A_l107_107536

theorem coordinates_of_point_A (x y : ℤ) (h : x = -1 ∧ y = 2) : (x, y) = (-1, 2) :=
by {
  cases h,
  rw [h_left, h_right],
}

end coordinates_of_point_A_l107_107536


namespace smallest_difference_l107_107058

theorem smallest_difference {a b : ℕ} (h1: a * b = 2010) (h2: a > b) : a - b = 37 :=
sorry

end smallest_difference_l107_107058


namespace probability_Hugo_first_roll_is_six_l107_107526

/-
In a dice game, each of 5 players, including Hugo, rolls a standard 6-sided die. 
The winner is the player who rolls the highest number. 
In the event of a tie for the highest roll, those involved in the tie roll again until a clear winner emerges.
-/
variable (HugoRoll : Nat) (A1 B1 C1 D1 : Nat)
variable (W : Bool)

-- Conditions in the problem
def isWinner (HugoRoll : Nat) (W : Bool) : Prop := (W = true)
def firstRollAtLeastFour (HugoRoll : Nat) : Prop := HugoRoll >= 4
def firstRollIsSix (HugoRoll : Nat) : Prop := HugoRoll = 6

-- Hypotheses: Hugo's event conditions
axiom HugoWonAndRollsAtLeastFour : isWinner HugoRoll W ∧ firstRollAtLeastFour HugoRoll

-- Target probability based on problem statement
noncomputable def probability (p : ℚ) : Prop := p = 625 / 4626

-- Main statement
theorem probability_Hugo_first_roll_is_six (HugoRoll : Nat) (A1 B1 C1 D1 : Nat) (W : Bool) :
  isWinner HugoRoll W ∧ firstRollAtLeastFour HugoRoll → 
  probability (625 / 4626) := by
  sorry


end probability_Hugo_first_roll_is_six_l107_107526


namespace total_children_on_playground_l107_107405

theorem total_children_on_playground (boys girls : ℕ) (hb : boys = 27) (hg : girls = 35) : boys + girls = 62 :=
  by
  -- Proof goes here
  sorry

end total_children_on_playground_l107_107405


namespace line_circle_no_intersection_l107_107849

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l107_107849


namespace largest_possible_c_l107_107259

theorem largest_possible_c (c : ℝ) (hc : (3 * c + 4) * (c - 2) = 9 * c) : c ≤ 4 :=
sorry

end largest_possible_c_l107_107259


namespace average_cost_per_pencil_proof_l107_107430

noncomputable def average_cost_per_pencil (pencils_qty: ℕ) (price: ℝ) (discount_percent: ℝ) (shipping_cost: ℝ) : ℝ :=
  let discounted_price := price * (1 - discount_percent / 100)
  let total_cost := discounted_price + shipping_cost
  let cost_in_cents := total_cost * 100
  cost_in_cents / pencils_qty

theorem average_cost_per_pencil_proof :
  average_cost_per_pencil 300 29.85 10 7.50 = 11 :=
by
  sorry

end average_cost_per_pencil_proof_l107_107430


namespace minimum_value_fraction_l107_107518

theorem minimum_value_fraction (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 1) :
  (1 / a) + (4 / b) ≥ 9 :=
sorry

end minimum_value_fraction_l107_107518


namespace correct_sum_of_satisfying_values_l107_107833

def g (x : Nat) : Nat :=
  match x with
  | 0 => 0
  | 1 => 2
  | 2 => 1
  | _ => 0  -- This handles the out-of-bounds case, though it's not needed here

def f (x : Nat) : Nat :=
  match x with
  | 0 => 2
  | 1 => 1
  | 2 => 0
  | _ => 0  -- This handles the out-of-bounds case, though it's not needed here

def satisfies_condition (x : Nat) : Bool :=
  f (g x) > g (f x)

def sum_of_satisfying_values : Nat :=
  List.sum (List.filter satisfies_condition [0, 1, 2])

theorem correct_sum_of_satisfying_values : sum_of_satisfying_values = 2 :=
  sorry

end correct_sum_of_satisfying_values_l107_107833


namespace find_correct_value_l107_107962

theorem find_correct_value (incorrect_value : ℝ) (subtracted_value : ℝ) (added_value : ℝ) (h_sub : subtracted_value = -added_value)
(h_incorrect : incorrect_value = 8.8) (h_subtracted : subtracted_value = -4.3) (h_added : added_value = 4.3) : incorrect_value + added_value + added_value = 17.4 :=
by
  sorry

end find_correct_value_l107_107962


namespace intersection_range_l107_107837

noncomputable def function1 (x : ℝ) : ℝ := abs (x^2 - 1) / (x - 1)
noncomputable def function2 (k x : ℝ) : ℝ := k * x - 2

theorem intersection_range (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ function1 x₁ = function2 k x₁ ∧ function1 x₂ = function2 k x₂) ↔ 
  (0 < k ∧ k < 1) ∨ (1 < k ∧ k < 4) := 
sorry

end intersection_range_l107_107837


namespace quadratic_inequality_range_of_k_l107_107508

theorem quadratic_inequality (a b x : ℝ) (h1 : a = 1) (h2 : b > 1) :
  (a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
sorry

theorem range_of_k (x y k : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1/x) + (2/y) = 1) (h4 : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 :=
sorry

end quadratic_inequality_range_of_k_l107_107508


namespace geom_seq_a3_a5_product_l107_107689

-- Defining the conditions: a sequence and its sum formula
def geom_seq (a : ℕ → ℕ) := ∃ r : ℕ, ∀ n, a (n+1) = a n * r

def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = 2^(n-1) + a 1

-- The theorem statement
theorem geom_seq_a3_a5_product (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : geom_seq a) (h2 : sum_first_n_terms a S) : a 3 * a 5 = 16 := 
sorry

end geom_seq_a3_a5_product_l107_107689


namespace triangle_properties_l107_107240

theorem triangle_properties
  (a b : ℝ)
  (C : ℝ)
  (ha : a = 2)
  (hb : b = 3)
  (hC : C = Real.pi / 3)
  :
  let c := Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)
  let area := (1 / 2) * a * b * Real.sin C
  let sin2A := 2 * (a * Real.sin C / c) * Real.sqrt (1 - (a * Real.sin C / c)^2)
  c = Real.sqrt 7 
  ∧ area = (3 * Real.sqrt 3) / 2 
  ∧ sin2A = (4 * Real.sqrt 3) / 7 :=
by
  sorry

end triangle_properties_l107_107240


namespace triangle_inequality_l107_107836

variable (R r e f : ℝ)

theorem triangle_inequality (h1 : ∃ (A B C : ℝ × ℝ), true)
                            (h2 : true) :
  R^2 - e^2 ≥ 4 * (r^2 - f^2) :=
by sorry

end triangle_inequality_l107_107836


namespace ticket_cost_l107_107467

theorem ticket_cost (total_amount_collected : ℕ) (average_tickets_per_day : ℕ) (days : ℕ) 
  (h1 : total_amount_collected = 960) 
  (h2 : average_tickets_per_day = 80) 
  (h3 : days = 3) : 
  total_amount_collected / (average_tickets_per_day * days) = 4 :=
  sorry

end ticket_cost_l107_107467


namespace calculate_star_operation_l107_107527

def operation (a b : ℚ) : ℚ := 2 * a - b + 1

theorem calculate_star_operation :
  operation 1 (operation 3 (-2)) = -6 :=
by
  sorry

end calculate_star_operation_l107_107527


namespace difference_apples_peaches_pears_l107_107555

-- Definitions based on the problem conditions
def apples : ℕ := 60
def peaches : ℕ := 3 * apples
def pears : ℕ := apples / 2

-- Statement of the proof problem
theorem difference_apples_peaches_pears : (apples + peaches) - pears = 210 := by
  sorry

end difference_apples_peaches_pears_l107_107555


namespace fractional_equation_solution_l107_107825

theorem fractional_equation_solution (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (3 / (x + 1) = 2 / (x - 1)) → (x = 5) :=
sorry

end fractional_equation_solution_l107_107825


namespace Connie_total_markers_l107_107993

/--
Connie has 41 red markers and 64 blue markers. 
We want to prove that the total number of markers Connie has is 105.
-/
theorem Connie_total_markers : 
  let red_markers := 41
  let blue_markers := 64
  let total_markers := red_markers + blue_markers
  total_markers = 105 :=
by
  sorry

end Connie_total_markers_l107_107993


namespace bread_rise_time_l107_107384

theorem bread_rise_time (x : ℕ) (kneading_time : ℕ) (baking_time : ℕ) (total_time : ℕ) 
  (h1 : kneading_time = 10) 
  (h2 : baking_time = 30) 
  (h3 : total_time = 280) 
  (h4 : kneading_time + baking_time + 2 * x = total_time) : 
  x = 120 :=
sorry

end bread_rise_time_l107_107384


namespace fraction_identity_l107_107190

open Real

theorem fraction_identity
  (p q r : ℝ)
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) = 8) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) = 2.2 :=
  sorry

end fraction_identity_l107_107190


namespace competition_result_l107_107562

-- Define the participants
inductive Person
| Olya | Oleg | Pasha
deriving DecidableEq, Repr

-- Define the placement as an enumeration
inductive Place
| first | second | third
deriving DecidableEq, Repr

-- Define the statements
structure Statements :=
(olyas_claim : Place)
(olyas_statement : Prop)
(olegs_statement : Prop)

-- Define the conditions
def conditions (s : Statements) : Prop :=
  -- All claimed first place
  s.olyas_claim = Place.first ∧ s.olyas_statement ∧ s.olegs_statement

-- Define the final placement
structure Placement :=
(olyas_place : Place)
(olegs_place : Place)
(pashas_place : Place)

-- Define the correct answer
def correct_placement : Placement :=
{ olyas_place := Place.third,
  olegs_place := Place.first,
  pashas_place := Place.second }

-- Lean statement for the problem
theorem competition_result (s : Statements) (h : conditions s) : Placement :=
{ olyas_place := Place.third,
  olegs_place := Place.first,
  pashas_place := Place.second } := sorry

end competition_result_l107_107562


namespace function_identity_l107_107656

theorem function_identity (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end function_identity_l107_107656


namespace a_and_b_together_complete_in_10_days_l107_107762

noncomputable def a_works_twice_as_fast_as_b (a b : ℝ) : Prop :=
  a = 2 * b

noncomputable def b_can_complete_work_in_30_days (b : ℝ) : Prop :=
  b = 1/30

theorem a_and_b_together_complete_in_10_days (a b : ℝ) 
  (h₁ : a_works_twice_as_fast_as_b a b)
  (h₂ : b_can_complete_work_in_30_days b) : 
  (1 / (a + b)) = 10 := 
sorry

end a_and_b_together_complete_in_10_days_l107_107762


namespace Suzanne_runs_5_kilometers_l107_107068

theorem Suzanne_runs_5_kilometers 
  (a : ℕ) 
  (r : ℕ) 
  (total_donation : ℕ) 
  (n : ℕ)
  (h1 : a = 10) 
  (h2 : r = 2) 
  (h3 : total_donation = 310) 
  (h4 : total_donation = a * (1 - r^n) / (1 - r)) 
  : n = 5 :=
by
  sorry

end Suzanne_runs_5_kilometers_l107_107068


namespace mitch_family_milk_l107_107032

variable (total_milk soy_milk regular_milk : ℚ)

-- Conditions
axiom cond1 : total_milk = 0.6
axiom cond2 : soy_milk = 0.1
axiom cond3 : regular_milk + soy_milk = total_milk

-- Theorem statement
theorem mitch_family_milk : regular_milk = 0.5 :=
by
  sorry

end mitch_family_milk_l107_107032


namespace number_of_sides_possibilities_l107_107211

theorem number_of_sides_possibilities (a b : ℕ) (h₁ : a = 8) (h₂ : b = 5) : 
  { x : ℕ // 3 < x ∧ x < 13 }.to_list.length = 9 :=
by
  sorry

end number_of_sides_possibilities_l107_107211


namespace M_empty_iff_k_range_M_interval_iff_k_range_l107_107357

-- Part 1
theorem M_empty_iff_k_range (k : ℝ) :
  (∀ x : ℝ, (k^2 + 2 * k - 3) * x^2 + (k + 3) * x - 1 ≤ 0) ↔ -3 ≤ k ∧ k ≤ 1 / 5 := sorry

-- Part 2
theorem M_interval_iff_k_range (k a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_ab : a < b) :
  (∀ x : ℝ, (k^2 + 2 * k - 3) * x^2 + (k + 3) * x - 1 > 0 ↔ a < x ∧ x < b) ↔ 1 / 5 < k ∧ k < 1 := sorry

end M_empty_iff_k_range_M_interval_iff_k_range_l107_107357


namespace cylinder_surface_area_l107_107062

theorem cylinder_surface_area (a b : ℝ) (h1 : a = 4 * Real.pi) (h2 : b = 8 * Real.pi) :
  (∃ S, S = 32 * Real.pi^2 + 8 * Real.pi ∨ S = 32 * Real.pi^2 + 32 * Real.pi) :=
by
  sorry

end cylinder_surface_area_l107_107062


namespace cos_in_third_quadrant_l107_107904

theorem cos_in_third_quadrant (B : ℝ) (h_sin_B : Real.sin B = -5/13) (h_quadrant : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 :=
by
  sorry

end cos_in_third_quadrant_l107_107904


namespace find_k_from_inequality_l107_107668

variable (k x : ℝ)

theorem find_k_from_inequality (h : ∀ x ∈ Set.Ico (-2 : ℝ) 1, 1 + k / (x - 1) ≤ 0)
  (h₂: 1 + k / (-2 - 1) = 0) :
  k = 3 :=
by
  sorry

end find_k_from_inequality_l107_107668


namespace expand_binomials_l107_107638

theorem expand_binomials : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by 
  sorry

end expand_binomials_l107_107638


namespace hyperbola_vertex_distance_l107_107817

theorem hyperbola_vertex_distance (a b : ℝ) (h_eq : a^2 = 16) (hyperbola_eq : ∀ x y : ℝ, 
  (x^2 / 16) - (y^2 / 9) = 1) : 
  (2 * a) = 8 :=
by
  have h_a : a = 4 := by sorry
  rw [h_a]
  norm_num

end hyperbola_vertex_distance_l107_107817


namespace probability_X_lt_6_l107_107490

noncomputable def X : ℝ → ℝ := sorry -- Define the random variable X properly

theorem probability_X_lt_6 :
  (X ~ N(4, σ^2)) → (P (λ x, X x ≤ 2) = 0.3) → (P (λ x, X x < 6) = 0.7) :=
by
  intro h1 h2
  sorry

end probability_X_lt_6_l107_107490


namespace upgraded_fraction_l107_107320

theorem upgraded_fraction (N U : ℕ) (h1 : ∀ (k : ℕ), k = 24)
  (h2 : ∀ (n : ℕ), N = n) (h3 : ∀ (u : ℕ), U = u)
  (h4 : N = U / 8) : U / (24 * N + U) = 1 / 4 := by
  sorry

end upgraded_fraction_l107_107320


namespace jellybeans_original_count_l107_107810

theorem jellybeans_original_count (x : ℝ) (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_original_count_l107_107810


namespace line_circle_no_intersection_l107_107852

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l107_107852


namespace quadratic_roster_method_l107_107814

theorem quadratic_roster_method :
  {x : ℝ | x^2 - 3 * x + 2 = 0} = {1, 2} :=
by
  sorry

end quadratic_roster_method_l107_107814


namespace algebra_expression_value_l107_107828

theorem algebra_expression_value (a b : ℝ) 
  (h₁ : a - b = 5) 
  (h₂ : a * b = -1) : 
  (2 * a + 3 * b - 2 * a * b) 
  - (a + 4 * b + a * b) 
  - (3 * a * b + 2 * b - 2 * a) = 21 := 
by
  sorry

end algebra_expression_value_l107_107828


namespace polynomial_coefficient_B_l107_107611

theorem polynomial_coefficient_B : 
  ∃ (A C D : ℤ), 
    (∀ z : ℤ, (z > 0) → (z^6 - 15 * z^5 + A * z^4 + B * z^3 + C * z^2 + D * z + 64 = 0)) ∧ 
    (B = -244) := 
by
  sorry

end polynomial_coefficient_B_l107_107611


namespace solve_for_a_l107_107899

theorem solve_for_a (x : ℤ) (a : ℤ) (h : 3 * x + 2 * a + 1 = 2) (hx : x = -1) : a = 2 :=
by
  sorry

end solve_for_a_l107_107899


namespace digit_864_div_5_appending_zero_possibilities_l107_107766

theorem digit_864_div_5_appending_zero_possibilities :
  ∀ X : ℕ, (X * 1000 + 864) % 5 ≠ 0 :=
by sorry

end digit_864_div_5_appending_zero_possibilities_l107_107766


namespace quadratic_sum_is_zero_l107_107739

-- Definition of a quadratic function with given conditions and the final result to prove
theorem quadratic_sum_is_zero {a b c : ℝ} 
  (h₁ : ∀ x : ℝ, x = 3 → a * (x - 1) * (x - 5) = 36) 
  (h₂ : a * 1^2 + b * 1 + c = 0) 
  (h₃ : a * 5^2 + b * 5 + c = 0) : 
  a + b + c = 0 := 
sorry

end quadratic_sum_is_zero_l107_107739


namespace arithmetic_sequence_common_difference_l107_107088

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)

-- Conditions
def condition1 : Prop := ∀ n, S n = (n * (2*a 1 + (n-1) * d)) / 2
def condition2 : Prop := S 3 = 6
def condition3 : Prop := a 3 = 0

-- Question
def question : ℝ := d

-- Correct Answer
def correct_answer : ℝ := -2

-- Proof Problem Statement
theorem arithmetic_sequence_common_difference : 
  condition1 a S d ∧ condition2 S ∧ condition3 a →
  question d = correct_answer :=
sorry

end arithmetic_sequence_common_difference_l107_107088


namespace jellybeans_original_count_l107_107809

theorem jellybeans_original_count (x : ℝ) (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_original_count_l107_107809


namespace bounded_region_area_l107_107625

theorem bounded_region_area : 
  (∀ x y : ℝ, (y^2 + 4*x*y + 50*|x| = 500) → (x ≥ 0 ∧ y = 25 - 4*x) ∨ (x ≤ 0 ∧ y = -12.5 - 4*x)) →
  ∃ (A : ℝ), A = 156.25 :=
by
  sorry

end bounded_region_area_l107_107625


namespace twin_primes_iff_congruence_l107_107573

theorem twin_primes_iff_congruence (p : ℕ) : 
  Prime p ∧ Prime (p + 2) ↔ 4 * ((p - 1)! + 1) + p ≡ 0 [MOD p^2 + 2 * p] :=
by 
  sorry

end twin_primes_iff_congruence_l107_107573


namespace value_of_f_5_l107_107685

-- Define the function f
def f (x y : ℕ) : ℕ := 2 * x ^ 2 + y

-- Given conditions
variable (some_value : ℕ)
axiom h1 : f some_value 52 = 60
axiom h2 : f 5 52 = 102

-- Proof statement
theorem value_of_f_5 : f 5 52 = 102 := by
  sorry

end value_of_f_5_l107_107685


namespace necessary_but_not_sufficient_condition_for_x_gt_2_l107_107012

theorem necessary_but_not_sufficient_condition_for_x_gt_2 :
  ∀ (x : ℝ), (2 / x < 1 → x > 2) ∧ (x > 2 → 2 / x < 1) → (¬ (x > 2 → 2 / x < 1) ∨ ¬ (2 / x < 1 → x > 2)) :=
by
  intro x h
  sorry

end necessary_but_not_sufficient_condition_for_x_gt_2_l107_107012


namespace cary_ivy_removal_days_correct_l107_107987

noncomputable def cary_ivy_removal_days (initial_ivy : ℕ) (ivy_removed_per_day : ℕ) (ivy_growth_per_night : ℕ) : ℕ :=
  initial_ivy / (ivy_removed_per_day - ivy_growth_per_night)

theorem cary_ivy_removal_days_correct :
  cary_ivy_removal_days 40 6 2 = 10 :=
by
  -- The body of the proof is omitted; it will be filled with the actual proof.
  sorry

end cary_ivy_removal_days_correct_l107_107987


namespace quadratic_coeff_sum_l107_107741

theorem quadratic_coeff_sum {a b c : ℝ} (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 1) * (x - 5))
    (h2 : a * 3^2 + b * 3 + c = 36) : a + b + c = 0 :=
by
  sorry

end quadratic_coeff_sum_l107_107741


namespace system1_solution_system2_solution_l107_107391

-- For System (1)
theorem system1_solution (x y : ℝ) (h1 : y = 2 * x) (h2 : 3 * y + 2 * x = 8) : x = 1 ∧ y = 2 :=
by
  sorry

-- For System (2)
theorem system2_solution (s t : ℝ) (h1 : 2 * s - 3 * t = 2) (h2 : (s + 2 * t) / 3 = 3 / 2) : s = 5 / 2 ∧ t = 1 :=
by
  sorry

end system1_solution_system2_solution_l107_107391


namespace ratio_proof_l107_107021

theorem ratio_proof (a b c : ℝ) (h1 : b / a = 4) (h2 : c / b = 5) : (a + 2 * b) / (3 * b + c) = 9 / 32 :=
by
  sorry

end ratio_proof_l107_107021


namespace race_problem_l107_107371

theorem race_problem 
  (A B C : ℝ) 
  (h1 : A = 100) 
  (h2 : B = 100 - x) 
  (h3 : C = 72) 
  (h4 : B = C + 4)
  : x = 24 := 
by 
  sorry

end race_problem_l107_107371


namespace salary_percentage_difference_l107_107232

theorem salary_percentage_difference (A B : ℝ) (h : A = 0.8 * B) :
  (B - A) / A * 100 = 25 :=
sorry

end salary_percentage_difference_l107_107232


namespace value_of_y_at_x8_l107_107903

theorem value_of_y_at_x8 (k : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = k * x^(1 / 3)) (h2 : f 64 = 4) : f 8 = 2 :=
sorry

end value_of_y_at_x8_l107_107903


namespace parabola_vertex_eq_l107_107798

theorem parabola_vertex_eq : 
  ∃ (x y : ℝ), y = -3 * x^2 + 6 * x + 1 ∧ (x = 1) ∧ (y = 4) := 
by
  sorry

end parabola_vertex_eq_l107_107798


namespace max_real_roots_quadratics_l107_107549

theorem max_real_roots_quadratics (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∀ f g h :∃(f = λ x : ℝ, a * x^2 + b * x + c ), ∃(g = λ x : ℝ, b * x^2 + c * x + a), ∃(h = λ x : ℝ, c * x^2 + a * x + b), 
  ∃(f_roots : ∀(x1 x2 : ℝ), (f(x1)=0 -> (f(x2)=0 -> x1=x2) /\ (x1=x2)), (∀(x3 x4 : ℝ), (g(x3)=0 -> (g(x4)=0 -> x3=x4) /\ (x3=x4)), 
  (∀(x5 x6 : ℝ), (h(x5)=0 -> (h(x6)=0 -> x5=x6) /\ (x5=x6)), 
  (4 >= condition : bowers(e_null_roots) /\ all.equal_values (bowers(f_roots) bowers(g_roots) bowers(h_roots)))
 :=
sorry

end max_real_roots_quadratics_l107_107549


namespace tenth_term_arithmetic_sequence_l107_107296

theorem tenth_term_arithmetic_sequence :
  ∀ (a₁ a₃₀ : ℕ) (d : ℕ) (n : ℕ), a₁ = 3 → a₃₀ = 89 → n = 10 → 
  (a₃₀ - a₁) / 29 = d → a₁ + (n - 1) * d = 30 :=
by
  intros a₁ a₃₀ d n h₁ h₃₀ hn hd
  sorry

end tenth_term_arithmetic_sequence_l107_107296


namespace largest_w_l107_107025

variable {x y z w : ℝ}

def x_value (x y z w : ℝ) := 
  x + 3 = y - 1 ∧ x + 3 = z + 5 ∧ x + 3 = w - 4

theorem largest_w (h : x_value x y z w) : 
  max x (max y (max z w)) = w := 
sorry

end largest_w_l107_107025


namespace greatest_odd_factors_l107_107721

theorem greatest_odd_factors (n : ℕ) (h1 : n < 200) (h2 : ∀ k < 200, k ≠ 196 → odd (number_of_factors k) = false) : n = 196 :=
sorry

end greatest_odd_factors_l107_107721


namespace no_integer_solutions_l107_107223

theorem no_integer_solutions (x y z : ℤ) :
  x^2 - 3 * x * y + 2 * y^2 - z^2 = 31 ∧
  -x^2 + 6 * y * z + 2 * z^2 = 44 ∧
  x^2 + x * y + 8 * z^2 = 100 →
  false :=
by
  sorry

end no_integer_solutions_l107_107223


namespace jellybeans_initial_amount_l107_107802

theorem jellybeans_initial_amount (x : ℝ) 
  (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_initial_amount_l107_107802


namespace floor_floor_3x_eq_floor_x_plus_1_l107_107175

theorem floor_floor_3x_eq_floor_x_plus_1 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1⌋ = ⌊x + 1⌋) ↔ (2 / 3 ≤ x ∧ x < 4 / 3) :=
by
  sorry

end floor_floor_3x_eq_floor_x_plus_1_l107_107175


namespace line_circle_no_intersection_l107_107870

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l107_107870


namespace houses_with_pools_l107_107693

theorem houses_with_pools (total G overlap N P : ℕ) 
  (h1 : total = 70) 
  (h2 : G = 50) 
  (h3 : overlap = 35) 
  (h4 : N = 15) 
  (h_eq : total = G + P - overlap + N) : 
  P = 40 := by
  sorry

end houses_with_pools_l107_107693


namespace probability_adjacent_vertices_of_decagon_l107_107100

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l107_107100


namespace multiple_of_9_l107_107422

noncomputable def digit_sum (x : ℕ) : ℕ := sorry  -- Placeholder for the digit sum function

theorem multiple_of_9 (n : ℕ) (h1 : digit_sum n = digit_sum (3 * n))
  (h2 : ∀ x, x % 9 = digit_sum x % 9) :
  n % 9 = 0 :=
by
  sorry

end multiple_of_9_l107_107422


namespace range_of_a_l107_107191

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x ≥ 4 ∧ y ≥ 4 ∧ x ≤ y → (x^2 + 2*(a-1)*x + 2) ≤ (y^2 + 2*(a-1)*y + 2)) ↔ a ∈ Set.Ici (-3) :=
by
  sorry

end range_of_a_l107_107191


namespace prob_diff_sets_is_three_fourths_E_X_is_three_fourths_l107_107408

-- Define the problem context
noncomputable def housing : Type := Fin 4

-- Applicants
inductive Applicant
| A | B | C
open Applicant

-- Event Definitions
def chooses (p : housing → Prop) (applicant : Applicant) : Prop :=
  ∃ (h : housing), p h

-- Problem 1: Probability that A and B do not apply for the same set of housing
def prob_diff_sets : ℝ :=
  let pABsame := (1 : ℝ) / 4
  1 - pABsame

theorem prob_diff_sets_is_three_fourths : prob_diff_sets = 3 / 4 := sorry

-- Problem 2: Defining X and calculating its expectation
def X (h : housing → Prop) : ℝ := 
  (if chooses h A then 1 else 0) +
  (if chooses h B then 1 else 0) +
  (if chooses h C then 1 else 0)

def E_X : ℝ := 
  (3 : ℝ) / 4

theorem E_X_is_three_fourths : E_X = 3 / 4 := sorry

end prob_diff_sets_is_three_fourths_E_X_is_three_fourths_l107_107408


namespace problem_l107_107231

variable (x y z : ℚ)

-- Conditions as definitions
def cond1 : Prop := x / y = 3
def cond2 : Prop := y / z = 5 / 2

-- Theorem statement with the final proof goal
theorem problem (h1 : cond1 x y) (h2 : cond2 y z) : z / x = 2 / 15 := 
by 
  sorry

end problem_l107_107231


namespace probability_of_dice_outcome_l107_107309

theorem probability_of_dice_outcome : 
  let p_one_digit := 3 / 4
  let p_two_digit := 1 / 4
  let comb := Nat.choose 5 3
  (comb * (p_one_digit^3) * (p_two_digit^2)) = 135 / 512 := 
by
  sorry

end probability_of_dice_outcome_l107_107309


namespace arithmetic_mean_of_two_numbers_l107_107271

def is_arithmetic_mean (x y z : ℚ) : Prop :=
  (x + z) / 2 = y

theorem arithmetic_mean_of_two_numbers :
  is_arithmetic_mean (9 / 12) (5 / 6) (7 / 8) :=
by
  sorry

end arithmetic_mean_of_two_numbers_l107_107271


namespace distance_between_vertices_of_hyperbola_l107_107819

theorem distance_between_vertices_of_hyperbola :
  ∀ x y : ℝ, (x^2 / 16 - y^2 / 9 = 1) → 8 :=
by
  sorry

end distance_between_vertices_of_hyperbola_l107_107819


namespace problem_statement_l107_107007

-- Conditions
def circle_M (x y : ℝ) : Prop := (x + real.sqrt 3)^2 + y^2 = 16
def passes_through_F (x : ℝ) (y : ℝ) : Prop := x = real.sqrt 3 ∧ y = 0

-- Target Ellipse: Equation of trajectory E
def trajectory_E (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Symmetric Points and Length Conditions
def is_symmetric_to_origin (A B : ℝ × ℝ) : Prop := A.1 = -B.1 ∧ A.2 = -B.2
def distance_AC_CB (A C B : ℝ × ℝ) : Prop := real.dist A C = real.dist C B

-- Minimum area of triangle ABC
def min_area_triangle (A B C : ℝ × ℝ) : ℝ := 
  1/2 * real.abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Line equations AB
def line_eq (A B : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, A.2 = k * A.1 ∧ B.2 = k * B.1

-- Lean Statement
theorem problem_statement :
  (∀ x y, circle_M x y) →
  (∃ N_center, passes_through_F N_center.1 N_center.2 ∧ trajectory_E N_center.1 N_center.2) ∧
  (
    ∀ A B C,
    is_symmetric_to_origin A B →
    distance_AC_CB A C B →
    A ∈ trajectory_E ∧ B ∈ trajectory_E ∧ C ∈ trajectory_E →
    line_eq A B ↔ min_area_triangle A B C = 8 / 5
  ) :=
by sorry

end problem_statement_l107_107007


namespace subsets_with_sum_2023060_l107_107166

open Finset

def B : Finset ℕ := range 2012

theorem subsets_with_sum_2023060 :
  (B.filter (λ s, s.val.sum = 2023060)).card = 4 := sorry

end subsets_with_sum_2023060_l107_107166


namespace quadratic_inequality_solution_set_l107_107180

theorem quadratic_inequality_solution_set (x : ℝ) : 
  (x^2 - 2 * x < 0) ↔ (0 < x ∧ x < 2) := 
sorry

end quadratic_inequality_solution_set_l107_107180


namespace range_of_a_for_monotonic_increasing_f_l107_107832

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * x - 2 * Real.log x

theorem range_of_a_for_monotonic_increasing_f (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x → (x - a - 2 / x) ≥ 0) : a ≤ -1 :=
by {
  -- Placeholder for the detailed proof steps
  sorry
}

end range_of_a_for_monotonic_increasing_f_l107_107832


namespace sum_of_values_of_n_l107_107997

theorem sum_of_values_of_n (n₁ n₂ : ℚ) (h1 : 3 * n₁ - 8 = 5) (h2 : 3 * n₂ - 8 = -5) : n₁ + n₂ = 16 / 3 := 
by {
  -- Use the provided conditions to solve the problem
  sorry 
}

end sum_of_values_of_n_l107_107997


namespace triangle_side_lengths_l107_107204

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end triangle_side_lengths_l107_107204


namespace total_cans_l107_107272

def bag1 := 5
def bag2 := 7
def bag3 := 12
def bag4 := 4
def bag5 := 8
def bag6 := 10

theorem total_cans : bag1 + bag2 + bag3 + bag4 + bag5 + bag6 = 46 := by
  sorry

end total_cans_l107_107272


namespace sum_max_min_on_interval_l107_107077

-- Defining the function f
def f (x : ℝ) : ℝ := x + 2

-- The proof statement
theorem sum_max_min_on_interval : 
  let M := max (f 0) (f 4)
  let N := min (f 0) (f 4)
  M + N = 8 := by
  -- Placeholder for proof
  sorry

end sum_max_min_on_interval_l107_107077


namespace distance_between_points_l107_107380

open Complex Real

def joe_point : ℂ := 2 + 3 * I
def gracie_point : ℂ := -2 + 2 * I

theorem distance_between_points : abs (joe_point - gracie_point) = sqrt 17 := by
  sorry

end distance_between_points_l107_107380


namespace even_function_maximum_l107_107235

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

noncomputable def has_maximum_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x : ℝ, a ≤ x ∧ x ≤ b ∧ ∀ y : ℝ, a ≤ y ∧ y ≤ b → f y ≤ f x

theorem even_function_maximum 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_max_1_7 : has_maximum_on_interval f 1 7) :
  has_maximum_on_interval f (-7) (-1) :=
sorry

end even_function_maximum_l107_107235


namespace squares_with_center_35_65_l107_107387

theorem squares_with_center_35_65 : 
  (∃ (n : ℕ), n = 1190 ∧ ∀ (x y : ℕ), x ≠ y → (x, y) = (35, 65)) :=
sorry

end squares_with_center_35_65_l107_107387


namespace sufficient_but_not_necessary_condition_l107_107897

theorem sufficient_but_not_necessary_condition (a b : ℝ) (hb : b < -1) : |a| + |b| > 1 := 
by
  sorry

end sufficient_but_not_necessary_condition_l107_107897


namespace batch_preparation_l107_107614

theorem batch_preparation (total_students cupcakes_per_student cupcakes_per_batch percent_not_attending : ℕ)
    (hlt1 : total_students = 150)
    (hlt2 : cupcakes_per_student = 3)
    (hlt3 : cupcakes_per_batch = 20)
    (hlt4 : percent_not_attending = 20)
    : (total_students * (80 / 100) * cupcakes_per_student) / cupcakes_per_batch = 18 := by
  sorry

end batch_preparation_l107_107614


namespace jeffs_mean_l107_107916

-- Define Jeff's scores as a list or array
def jeffsScores : List ℚ := [86, 94, 87, 96, 92, 89]

-- Prove that the arithmetic mean of Jeff's scores is 544 / 6
theorem jeffs_mean : (jeffsScores.sum / jeffsScores.length) = (544 / 6) := by
  sorry

end jeffs_mean_l107_107916


namespace incorrect_arrangements_hello_l107_107366

-- Given conditions: the word "hello" with letters 'h', 'e', 'l', 'l', 'o'
def letters : List Char := ['h', 'e', 'l', 'l', 'o']

-- The number of permutations of the letters in "hello" excluding the correct order
-- We need to prove that the number of incorrect arrangements is 59.
theorem incorrect_arrangements_hello : 
  (List.permutations letters).length - 1 = 59 := 
by sorry

end incorrect_arrangements_hello_l107_107366


namespace valid_votes_l107_107304

theorem valid_votes (V : ℝ) 
  (h1 : 0.70 * V - 0.30 * V = 176): V = 440 :=
  sorry

end valid_votes_l107_107304


namespace fraction_to_decimal_l107_107795

theorem fraction_to_decimal : (22 / 8 : ℝ) = 2.75 := 
sorry

end fraction_to_decimal_l107_107795


namespace original_triangle_area_l107_107072

theorem original_triangle_area (A_orig A_new : ℝ) (h1 : A_new = 256) (h2 : A_new = 16 * A_orig) : A_orig = 16 :=
by
  sorry

end original_triangle_area_l107_107072


namespace area_of_lune_l107_107451

theorem area_of_lune :
  ∃ (A L : ℝ), A = (3/2) ∧ L = 2 ∧
  (Lune_area : ℝ) = (9 * Real.sqrt 3 / 4) - (55 * π / 24) →
  Lune_area = (9 * Real.sqrt 3 / 4) - (55 * π / 24) :=
by
  sorry

end area_of_lune_l107_107451


namespace line_circle_no_intersection_l107_107877

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l107_107877


namespace apples_in_box_l107_107939

-- Define the initial conditions
def oranges : ℕ := 12
def removed_oranges : ℕ := 6
def target_percentage : ℚ := 0.70

-- Define the function that models the problem
def fruit_after_removal (apples : ℕ) : ℕ := apples + (oranges - removed_oranges)
def apples_percentage (apples : ℕ) : ℚ := (apples : ℚ) / (fruit_after_removal apples : ℚ)

-- The theorem states the question and expected answer
theorem apples_in_box : ∃ (apples : ℕ), apples_percentage apples = target_percentage ∧ apples = 14 :=
by
  sorry

end apples_in_box_l107_107939


namespace smallest_sphere_radius_l107_107483

noncomputable def sphere_contains_pyramid (base_edge apothem : ℝ) : Prop :=
  ∃ (R : ℝ), ∀ base_edge = 14, apothem = 12, R = 7 * Real.sqrt 2
  
theorem smallest_sphere_radius: sphere_contains_pyramid 14 12 :=
by 
  sorry

end smallest_sphere_radius_l107_107483


namespace probability_perfect_square_product_l107_107432

open BigOperators

def fair_seven_sided_die : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

noncomputable def rolling_outcome (n: ℕ) : Set (Finset ℕ) :=
  { x | x.card = n ∧ x ⊆ fair_seven_sided_die }

theorem probability_perfect_square_product :
  let total_outcomes := (fair_seven_sided_die.card)^4 in
  let perfect_square_outcomes := 164 in
  gcd perfect_square_outcomes total_outcomes = 1 →
  perfect_square_outcomes + total_outcomes = 2565 :=
by
  intros total_outcomes perfect_square_outcomes gcd_condition
  have total_outcomes_def : total_outcomes = 2401 :=
    by sorry
  have perfect_square_outcomes_def : perfect_square_outcomes = 164 :=
    by sorry
  rw [total_outcomes_def, perfect_square_outcomes_def]
  sorry

end probability_perfect_square_product_l107_107432


namespace problem1_problem2_l107_107747

-- Problem 1
theorem problem1 : -9 + (-4 * 5) = -29 :=
by
  sorry

-- Problem 2
theorem problem2 : (-(6) * -2) / (2 / 3) = -18 :=
by
  sorry

end problem1_problem2_l107_107747


namespace breadth_of_garden_l107_107031

theorem breadth_of_garden (P L B : ℝ) (hP : P = 1800) (hL : L = 500) : B = 400 :=
by
  sorry

end breadth_of_garden_l107_107031


namespace marys_mother_paid_correct_total_l107_107268

def mary_and_friends_payment_per_person : ℕ := 1 -- $1 each
def number_of_people : ℕ := 3 -- Mary and two friends

def total_chicken_cost : ℕ := mary_and_friends_payment_per_person * number_of_people -- Total cost of the chicken

def beef_cost_per_pound : ℕ := 4 -- $4 per pound
def total_beef_pounds : ℕ := 3 -- 3 pounds of beef
def total_beef_cost : ℕ := beef_cost_per_pound * total_beef_pounds -- Total cost of the beef

def oil_cost : ℕ := 1 -- $1 for 1 liter of oil

def total_grocery_cost : ℕ := total_chicken_cost + total_beef_cost + oil_cost -- Total grocery cost

theorem marys_mother_paid_correct_total : total_grocery_cost = 16 := by
  -- Here you would normally provide the proof steps which we're skipping per instructions.
  sorry

end marys_mother_paid_correct_total_l107_107268


namespace pollywogs_disappear_in_44_days_l107_107812

theorem pollywogs_disappear_in_44_days :
  ∀ (initial_count rate_mature rate_caught first_period_days : ℕ),
  initial_count = 2400 →
  rate_mature = 50 →
  rate_caught = 10 →
  first_period_days = 20 →
  (initial_count - first_period_days * (rate_mature + rate_caught)) / rate_mature + first_period_days = 44 := 
by
  intros initial_count rate_mature rate_caught first_period_days h1 h2 h3 h4
  sorry

end pollywogs_disappear_in_44_days_l107_107812


namespace sin_minus_cos_eq_one_l107_107815

theorem sin_minus_cos_eq_one (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = 1) : x = Real.pi / 2 :=
by sorry

end sin_minus_cos_eq_one_l107_107815


namespace scientific_notation_l107_107485

theorem scientific_notation : 350000000 = 3.5 * 10^8 :=
by
  sorry

end scientific_notation_l107_107485


namespace max_value_frac_l107_107653
noncomputable section

open Real

variables (a b x y : ℝ)

theorem max_value_frac :
  a > 1 → b > 1 → 
  a^x = 2 → b^y = 2 →
  a + sqrt b = 4 →
  (2/x + 1/y) ≤ 4 :=
by
  intros ha hb hax hby hab
  sorry

end max_value_frac_l107_107653


namespace pure_alcohol_added_l107_107134

theorem pure_alcohol_added (x : ℝ) (h1 : 6 * 0.40 = 2.4)
    (h2 : (2.4 + x) / (6 + x) = 0.50) : x = 1.2 :=
by
  sorry

end pure_alcohol_added_l107_107134


namespace product_of_A_and_B_l107_107647

theorem product_of_A_and_B (A B : ℕ) (h1 : 3 / 9 = 6 / A) (h2 : B / 63 = 6 / A) : A * B = 378 :=
  sorry

end product_of_A_and_B_l107_107647


namespace balls_into_boxes_l107_107019

def ways_to_distribute_balls_in_boxes (balls boxes : ℕ) : ℕ :=
  ∑ (p : Multiset ℕ) in Multiset.powersetLen boxes (Multiset.replicate balls (1 : ℕ)).eraseDup,
  if p.sum = balls then p.map (fun x => x.card).prod else 0

theorem balls_into_boxes :
  ways_to_distribute_balls_in_boxes 6 4 = 84 :=
  sorry

end balls_into_boxes_l107_107019


namespace solve_inequality_l107_107392

theorem solve_inequality :
  { x : ℝ | (9 * x^2 + 27 * x - 64) / ((3 * x - 4) * (x + 5) * (x - 1)) < 4 } = 
    { x : ℝ | -5 < x ∧ x < -17 / 3 } ∪ { x : ℝ | 1 < x ∧ x < 4 } :=
by
  sorry

end solve_inequality_l107_107392


namespace infinite_solutions_implies_a_eq_2_l107_107958

theorem infinite_solutions_implies_a_eq_2 (a b : ℝ) (h : b = 1) :
  (∀ x : ℝ, a * (3 * x - 2) + b * (2 * x - 3) = 8 * x - 7) → a = 2 :=
by
  intro H
  sorry

end infinite_solutions_implies_a_eq_2_l107_107958


namespace boxes_contain_neither_markers_nor_sharpies_l107_107338

theorem boxes_contain_neither_markers_nor_sharpies :
  (∀ (total_boxes markers_boxes sharpies_boxes both_boxes neither_boxes : ℕ),
    total_boxes = 15 → markers_boxes = 8 → sharpies_boxes = 5 → both_boxes = 4 →
    neither_boxes = total_boxes - (markers_boxes + sharpies_boxes - both_boxes) →
    neither_boxes = 6) :=
by
  intros total_boxes markers_boxes sharpies_boxes both_boxes neither_boxes
  intros htotal hmarkers hsharpies hboth hcalc
  rw [htotal, hmarkers, hsharpies, hboth] at hcalc
  exact hcalc

end boxes_contain_neither_markers_nor_sharpies_l107_107338


namespace triangle_inequality_range_x_l107_107690

theorem triangle_inequality_range_x (x : ℝ) :
  let a := 3;
  let b := 8;
  let c := 1 + 2 * x;
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ↔ (2 < x ∧ x < 5) :=
by
  sorry

end triangle_inequality_range_x_l107_107690


namespace acute_triangle_angles_l107_107152

theorem acute_triangle_angles (x y z : ℕ) (angle1 angle2 angle3 : ℕ) 
  (h1 : angle1 = 7 * x) 
  (h2 : angle2 = 9 * y) 
  (h3 : angle3 = 11 * z) 
  (h4 : angle1 + angle2 + angle3 = 180)
  (hx : 1 ≤ x ∧ x ≤ 12)
  (hy : 1 ≤ y ∧ y ≤ 9)
  (hz : 1 ≤ z ∧ z ≤ 8)
  (ha1 : angle1 < 90)
  (ha2 : angle2 < 90)
  (ha3 : angle3 < 90)
  : angle1 = 42 ∧ angle2 = 72 ∧ angle3 = 66 
  ∨ angle1 = 49 ∧ angle2 = 54 ∧ angle3 = 77 
  ∨ angle1 = 56 ∧ angle2 = 36 ∧ angle3 = 88 
  ∨ angle1 = 84 ∧ angle2 = 63 ∧ angle3 = 33 :=
sorry

end acute_triangle_angles_l107_107152


namespace probability_adjacent_vertices_in_decagon_l107_107090

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l107_107090


namespace line_circle_no_intersection_l107_107874

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l107_107874


namespace crayons_in_judahs_box_l107_107255

theorem crayons_in_judahs_box (karen_crayons beatrice_crayons gilbert_crayons judah_crayons : ℕ)
  (h1 : karen_crayons = 128)
  (h2 : beatrice_crayons = karen_crayons / 2)
  (h3 : gilbert_crayons = beatrice_crayons / 2)
  (h4 : judah_crayons = gilbert_crayons / 4) :
  judah_crayons = 8 :=
by {
  sorry
}

end crayons_in_judahs_box_l107_107255


namespace line_circle_no_intersection_l107_107895

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l107_107895


namespace sum_of_reciprocals_squares_l107_107284

theorem sum_of_reciprocals_squares (a b : ℕ) (h : a * b = 17) :
  (1 : ℚ) / (a * a) + 1 / (b * b) = 290 / 289 :=
sorry

end sum_of_reciprocals_squares_l107_107284


namespace no_three_even_segments_with_odd_intersections_l107_107540

open Set

def is_even_length (s : Set ℝ) : Prop :=
  ∃ a b : ℝ, s = Icc a b ∧ (b - a) % 2 = 0

def is_odd_length (s : Set ℝ) : Prop :=
  ∃ a b : ℝ, s = Icc a b ∧ (b - a) % 2 = 1

theorem no_three_even_segments_with_odd_intersections :
  ¬ ∃ (S1 S2 S3 : Set ℝ),
    (is_even_length S1) ∧
    (is_even_length S2) ∧
    (is_even_length S3) ∧
    (is_odd_length (S1 ∩ S2)) ∧
    (is_odd_length (S1 ∩ S3)) ∧
    (is_odd_length (S2 ∩ S3)) :=
by
  -- Proof here
  sorry

end no_three_even_segments_with_odd_intersections_l107_107540


namespace line_circle_no_intersection_l107_107851

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l107_107851


namespace sqrt_nine_over_four_l107_107586

theorem sqrt_nine_over_four (x : ℝ) : x = 3 / 2 ∨ x = - (3 / 2) ↔ x * x = 9 / 4 :=
by {
  sorry
}

end sqrt_nine_over_four_l107_107586


namespace inequalities_for_m_gt_n_l107_107273

open Real

theorem inequalities_for_m_gt_n (m n : ℕ) (hmn : m > n) : 
  (1 + 1 / (m : ℝ)) ^ m > (1 + 1 / (n : ℝ)) ^ n ∧ 
  (1 + 1 / (m : ℝ)) ^ (m + 1) < (1 + 1 / (n : ℝ)) ^ (n + 1) := 
by
  sorry

end inequalities_for_m_gt_n_l107_107273


namespace lily_pads_half_lake_l107_107597

theorem lily_pads_half_lake (n : ℕ) (h : n = 39) :
  (n - 1) = 38 :=
by
  sorry

end lily_pads_half_lake_l107_107597


namespace max_planes_determined_by_15_points_l107_107445

theorem max_planes_determined_by_15_points : ∃ (n : ℕ), n = 455 ∧ 
  ∀ (P : Finset (Fin 15)), (15 + 1) ∣ (P.card * (P.card - 1) * (P.card - 2) / 6) → n = (15 * 14 * 13) / 6 :=
by
  sorry

end max_planes_determined_by_15_points_l107_107445


namespace maurice_age_l107_107326

theorem maurice_age (M : ℕ) 
  (h₁ : 48 = 4 * (M + 5)) : M = 7 := 
by
  sorry

end maurice_age_l107_107326


namespace lcm_of_three_numbers_l107_107406

theorem lcm_of_three_numbers (x : ℕ) :
  (Nat.gcd (3 * x) (Nat.gcd (4 * x) (5 * x)) = 40) →
  Nat.lcm (3 * x) (Nat.lcm (4 * x) (5 * x)) = 2400 :=
by
  sorry

end lcm_of_three_numbers_l107_107406


namespace angela_insects_l107_107459

theorem angela_insects:
  ∀ (A J D : ℕ), 
    A = J / 2 → 
    J = 5 * D → 
    D = 30 → 
    A = 75 :=
by
  intro A J D
  intro hA hJ hD
  sorry

end angela_insects_l107_107459


namespace decagon_adjacent_probability_l107_107105

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l107_107105


namespace rhombus_diagonal_l107_107282

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d1 = 14) (h2 : area = 126) (h3 : area = (d1 * d2) / 2) : d2 = 18 := 
by
  -- h1, h2, and h3 are the conditions
  sorry

end rhombus_diagonal_l107_107282


namespace miles_walked_on_Tuesday_l107_107041

theorem miles_walked_on_Tuesday (monday_miles total_miles : ℕ) (hmonday : monday_miles = 9) (htotal : total_miles = 18) :
  total_miles - monday_miles = 9 :=
by
  sorry

end miles_walked_on_Tuesday_l107_107041


namespace num_possible_triangle_sides_l107_107205

theorem num_possible_triangle_sides (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (a_eq_8 : a = 8) (b_eq_5 : b = 5) : 
  let x := {n : ℕ | 3 < n ∧ n < 13} in
  set.card x = 9 :=
by
  sorry

end num_possible_triangle_sides_l107_107205


namespace average_salary_all_workers_l107_107938

/-- The total number of workers in the workshop is 15 -/
def total_number_of_workers : ℕ := 15

/-- The number of technicians is 5 -/
def number_of_technicians : ℕ := 5

/-- The number of other workers is given by the total number minus technicians -/
def number_of_other_workers : ℕ := total_number_of_workers - number_of_technicians

/-- The average salary per head of the technicians is Rs. 800 -/
def average_salary_per_technician : ℕ := 800

/-- The average salary per head of the other workers is Rs. 650 -/
def average_salary_per_other_worker : ℕ := 650

/-- The total salary for all the workers -/
def total_salary : ℕ := (number_of_technicians * average_salary_per_technician) + (number_of_other_workers * average_salary_per_other_worker)

/-- The average salary per head of all the workers in the workshop is Rs. 700 -/
theorem average_salary_all_workers :
  total_salary / total_number_of_workers = 700 := by
  sorry

end average_salary_all_workers_l107_107938


namespace problem_II_l107_107038

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 3)^n

noncomputable def S_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 2) * (1 - (1 / 3)^n)

lemma problem_I_1 (n : ℕ) (hn : n > 0) : a_n n = (1 / 3)^n := by
  sorry

lemma problem_I_2 (n : ℕ) (hn : n > 0) : S_n n = (1 / 2) * (1 - (1 / 3)^n) := by
  sorry

theorem problem_II (t : ℝ) : S_n 1 = 1 / 3 ∧ S_n 2 = 4 / 9 ∧ S_n 3 = 13 / 27 ∧
  (S_n 1 + 3 * (S_n 2 + S_n 3) = 2 * (S_n 1 + S_n 2) * t) ↔ t = 2 := by
  sorry

end problem_II_l107_107038


namespace find_m_range_l107_107491

def proposition_p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m < 0) ∧ (1 > 0)

def proposition_q (m : ℝ) : Prop :=
  16 * (m - 2)^2 - 16 < 0

theorem find_m_range : {m : ℝ // proposition_p m ∧ proposition_q m} = {m : ℝ // 2 < m ∧ m < 3} :=
by
  sorry

end find_m_range_l107_107491


namespace largest_angle_of_pentagon_l107_107608

theorem largest_angle_of_pentagon (x : ℝ) : 
  (2*x + 2) + 3*x + 4*x + 5*x + (6*x - 2) = 540 → 
  6*x - 2 = 160 :=
by
  intro h
  sorry

end largest_angle_of_pentagon_l107_107608


namespace compare_abc_l107_107184

noncomputable def a : ℝ := 2^(4/3)
noncomputable def b : ℝ := 4^(2/5)
noncomputable def c : ℝ := 5^(2/3)

theorem compare_abc : c > a ∧ a > b := 
by
  sorry

end compare_abc_l107_107184


namespace probability_sum_not_less_than_14_l107_107310

theorem probability_sum_not_less_than_14 :
  let bag := Finset.range 8    -- This creates the set {0, 1, ..., 7}
  let cards := bag.map (λ x, x + 1)  -- Map to set {1, 2, ..., 8}
  let card_draws := cards.powerset.filter (λ s, s.card = 2)
  let favorable_draws := card_draws.filter (λ s, s.sum ≥ 14)
  (favorable_draws.card : ℚ) / card_draws.card = 1 / 14 := by
sorry

end probability_sum_not_less_than_14_l107_107310


namespace candidate_votes_percentage_l107_107606

-- Conditions
variables {P : ℝ} 
variables (totalVotes : ℝ := 8000)
variables (differenceVotes : ℝ := 2400)

-- Proof Problem
theorem candidate_votes_percentage (h : ((P / 100) * totalVotes + ((P / 100) * totalVotes + differenceVotes) = totalVotes)) : P = 35 :=
by
  sorry

end candidate_votes_percentage_l107_107606


namespace decagon_adjacent_probability_l107_107110

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l107_107110


namespace problem_l107_107229

variable (x y z : ℚ)

-- Conditions as definitions
def cond1 : Prop := x / y = 3
def cond2 : Prop := y / z = 5 / 2

-- Theorem statement with the final proof goal
theorem problem (h1 : cond1 x y) (h2 : cond2 y z) : z / x = 2 / 15 := 
by 
  sorry

end problem_l107_107229


namespace track_length_proof_l107_107789

noncomputable def track_length : ℝ :=
  let x := 541.67
  x

theorem track_length_proof
  (p : ℝ)
  (q : ℝ)
  (h1 : p = 1 / 4)
  (h2 : q = 120)
  (h3 : ¬(p = q))
  (h4 : ∃ r : ℝ, r = 180)
  (speed_constant : ∃ b_speed, ∃ s_speed, b_speed * t = q ∧ s_speed * t = r) :
  track_length = 541.67 :=
sorry

end track_length_proof_l107_107789


namespace sum_of_edge_lengths_of_cube_l107_107745

-- Define the problem conditions
def surface_area (a : ℝ) : ℝ := 6 * a^2

-- The final statement to prove
theorem sum_of_edge_lengths_of_cube (a : ℝ) (ha : surface_area a = 150) : 12 * a = 60 :=
by
  sorry

end sum_of_edge_lengths_of_cube_l107_107745


namespace passing_marks_l107_107303

theorem passing_marks (T P : ℝ) 
  (h1 : 0.30 * T = P - 60) 
  (h2 : 0.45 * T = P + 30) : 
  P = 240 := 
by
  sorry

end passing_marks_l107_107303


namespace triangle_side_count_l107_107216

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end triangle_side_count_l107_107216


namespace line_circle_no_intersection_l107_107863

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l107_107863


namespace clock_angle_at_3_40_l107_107752

noncomputable def hour_hand_angle (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
noncomputable def minute_hand_angle (m : ℕ) : ℝ := m * 6
noncomputable def angle_between_hands (h m : ℕ) : ℝ := 
  let angle := |minute_hand_angle m - hour_hand_angle h m|
  if angle > 180 then 360 - angle else angle

theorem clock_angle_at_3_40 : angle_between_hands 3 40 = 130.0 := 
by
  sorry

end clock_angle_at_3_40_l107_107752


namespace tangent_length_to_circle_l107_107654

-- Definitions capturing the conditions
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y + 1 = 0
def line_l (x y a : ℝ) : Prop := x + a * y - 1 = 0
def point_A (a : ℝ) : ℝ × ℝ := (-4, a)

-- Main theorem statement proving the question against the answer
theorem tangent_length_to_circle (a : ℝ) (x y : ℝ) (hC : circle_C x y) (hl : line_l 2 1 a) :
  (a = -1) -> (point_A a = (-4, -1)) -> ∃ b : ℝ, b = 6 := 
sorry

end tangent_length_to_circle_l107_107654


namespace mary_flour_amount_l107_107269

noncomputable def cups_of_flour_already_put_in
    (total_flour_needed : ℕ)
    (total_sugar_needed : ℕ)
    (extra_flour_needed : ℕ)
    (flour_to_be_added : ℕ) : ℕ :=
total_flour_needed - (total_sugar_needed + extra_flour_needed)

theorem mary_flour_amount
    (total_flour_needed : ℕ := 9)
    (total_sugar_needed : ℕ := 6)
    (extra_flour_needed : ℕ := 1) :
    cups_of_flour_already_put_in total_flour_needed total_sugar_needed extra_flour_needed (total_sugar_needed + extra_flour_needed) = 2 := by
  sorry

end mary_flour_amount_l107_107269


namespace line_circle_no_intersection_l107_107878

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l107_107878


namespace kenneth_past_finish_line_when_biff_finishes_l107_107787

-- Given conditions:
def race_distance : ℕ := 500
def biff_speed : ℕ := 50
def kenneth_speed : ℕ := 51

-- The statement to prove:
theorem kenneth_past_finish_line_when_biff_finishes :
  let time_biff_to_finish := race_distance / biff_speed in
  let distance_kenneth_in_that_time := kenneth_speed * time_biff_to_finish in
  distance_kenneth_in_that_time - race_distance = 10 :=
by
  sorry

end kenneth_past_finish_line_when_biff_finishes_l107_107787


namespace size_ratio_l107_107790

variable {U : ℝ} (h1 : C = 1.5 * U) (h2 : R = 4 / 3 * C)

theorem size_ratio : R = 8 / 3 * U :=
by
  sorry

end size_ratio_l107_107790


namespace relationship_m_n_l107_107489

theorem relationship_m_n (b : ℝ) (m : ℝ) (n : ℝ) (h1 : m = 2 * b + 2022) (h2 : n = b^2 + 2023) : m ≤ n :=
by
  sorry

end relationship_m_n_l107_107489


namespace jelly_bean_matching_probability_l107_107778

theorem jelly_bean_matching_probability :
  let Abe_jelly_beans := [2, 3] : List ℕ, -- 2 green, 3 red
      Bob_jelly_beans := [2, 2, 3] : List ℕ, -- 2 green, 2 yellow, 3 red
      total_beans := List.sum Abe_jelly_beans, -- 5 for Abe
      total_beans_bob := List.sum Bob_jelly_beans, -- 7 for Bob
      p_abe_green := (Abe_jelly_beans.head!) / total_beans,
      p_bob_green := (Bob_jelly_beans.head!) / total_beans_bob,
      p_abe_red := (Abe_jelly_beans.tail!.head!) / total_beans,
      p_bob_red := (Bob_jelly_beans.tail!.tail!.head!) / total_beans_bob
  in (p_abe_green * p_bob_green + p_abe_red * p_bob_red) = (13 / 35) := by
  sorry

end jelly_bean_matching_probability_l107_107778


namespace possible_integer_side_lengths_l107_107208

theorem possible_integer_side_lengths : 
  {x : ℤ | x > 3 ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l107_107208


namespace hexagon_inequality_l107_107244

variable {Point : Type*} [MetricSpace Point]

-- Define points A1, A2, A3, A4, A5, A6 in a Metric Space
variables (A1 A2 A3 A4 A5 A6 O : Point)

-- Conditions
def angle_condition (O A1 A2 A3 A4 A5 A6 : Point) : Prop :=
  -- Points form a hexagon where each side is visible from O at 60 degrees
  -- We assume MetricSpace has a function measuring angles such as angle O x y = 60
  true -- A simplified condition; the actual angle measurement needs more geometry setup

def distance_condition_odd (O A1 A3 A5 : Point) : Prop := dist O A1 > dist O A3 ∧ dist O A3 > dist O A5
def distance_condition_even (O A2 A4 A6 : Point) : Prop := dist O A2 > dist O A4 ∧ dist O A4 > dist O A6

-- Question to prove
theorem hexagon_inequality 
  (hc : angle_condition O A1 A2 A3 A4 A5 A6) 
  (ho : distance_condition_odd O A1 A3 A5)
  (he : distance_condition_even O A2 A4 A6) : 
  dist A1 A2 + dist A3 A4 + dist A5 A6 < dist A2 A3 + dist A4 A5 + dist A6 A1 := 
sorry

end hexagon_inequality_l107_107244


namespace range_of_n_l107_107708

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (n : ℝ) : Set ℝ := {x | n-1 < x ∧ x < n+1}

-- Define the condition A ∩ B ≠ ∅
def A_inter_B_nonempty (n : ℝ) : Prop := ∃ x, x ∈ A ∧ x ∈ B n

-- Prove the range of n for which A ∩ B ≠ ∅ is (-2, 2)
theorem range_of_n : ∀ n, A_inter_B_nonempty n ↔ (-2 < n ∧ n < 2) := by
  sorry

end range_of_n_l107_107708


namespace spherical_to_cartesian_l107_107167

theorem spherical_to_cartesian 
  (ρ θ φ : ℝ)
  (hρ : ρ = 3) 
  (hθ : θ = 7 * Real.pi / 12) 
  (hφ : φ = Real.pi / 4) :
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ) = 
  (3 * Real.sqrt 2 / 2 * Real.cos (7 * Real.pi / 12), 
   3 * Real.sqrt 2 / 2 * Real.sin (7 * Real.pi / 12), 
   3 * Real.sqrt 2 / 2) :=
by
  sorry

end spherical_to_cartesian_l107_107167


namespace number_of_answer_choices_l107_107609

theorem number_of_answer_choices (n : ℕ) (H1 : (n + 1)^4 = 625) : n = 4 :=
sorry

end number_of_answer_choices_l107_107609


namespace line_circle_no_intersection_l107_107867

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l107_107867


namespace line_circle_no_intersection_l107_107890

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l107_107890


namespace max_real_roots_l107_107550

theorem max_real_roots (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (b^2 - 4 * a * c < 0 ∨ c^2 - 4 * b * a < 0 ∨ a^2 - 4 * c * b < 0) ∧ 
    (b^2 - 4 * a * c ≥ 0 ∧ c^2 - 4 * b * a ≥ 0 ∧ a^2 - 4 * c * b < 0 ∨
     b^2 - 4 * a * c ≥ 0 ∧ c^2 - 4 * b * a < 0 ∧ a^2 - 4 * c * b ≥ 0 ∨
     b^2 - 4 * a * c < 0 ∧ c^2 - 4 * b * a ≥ 0 ∧ a^2 - 4 * c * b ≥ 0 ∨
     b^2 - 4 * a * c ≥ 0 ∧ c^2 - 4 * b * a ≥ 0 ∧ a^2 - 4 * c * b ≥ 0) 
    → 4 ≤ ∑ i in [ax^2 + bx + c, bx^2 + cx + a, cx^2 + ax + b], (roots i).length


end max_real_roots_l107_107550


namespace number_of_pencils_l107_107168

theorem number_of_pencils (E P : ℕ) (h1 : E + P = 8) (h2 : 300 * E + 500 * P = 3000) (hE : E ≥ 1) (hP : P ≥ 1) : P = 3 :=
by
  sorry

end number_of_pencils_l107_107168


namespace hyperbola_eccentricity_l107_107943

theorem hyperbola_eccentricity
  (a b m : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (PA_perpendicular_to_l2 : (b/a * m) / (m + a) * (-b/a) = -1)
  (PB_parallel_to_l2 : (b/a * m) / (m - a) = -b/a) :
  (∃ e, e = 2) :=
by sorry

end hyperbola_eccentricity_l107_107943


namespace difference_between_numbers_l107_107952

-- Given definitions based on conditions
def sum_of_two_numbers (x y : ℝ) : Prop := x + y = 15
def difference_of_two_numbers (x y : ℝ) : Prop := x - y = 10
def difference_of_squares (x y : ℝ) : Prop := x^2 - y^2 = 150

theorem difference_between_numbers (x y : ℝ) 
  (h1 : sum_of_two_numbers x y) 
  (h2 : difference_of_two_numbers x y) 
  (h3 : difference_of_squares x y) :
  x - y = 10 :=
by
  sorry

end difference_between_numbers_l107_107952


namespace maximum_profit_l107_107602

def radioactive_marble_problem : ℕ :=
    let total_marbles := 100
    let radioactive_marbles := 1
    let non_radioactive_profit := 1
    let measurement_cost := 1
    let max_profit := 92 
    max_profit

theorem maximum_profit 
    (total_marbles : ℕ := 100) 
    (radioactive_marbles : ℕ := 1) 
    (non_radioactive_profit : ℕ := 1) 
    (measurement_cost : ℕ := 1) :
    radioactive_marble_problem = 92 :=
by sorry

end maximum_profit_l107_107602


namespace parabola_min_value_sum_abc_zero_l107_107736

theorem parabola_min_value_sum_abc_zero
  (a b c : ℝ)
  (h1 : ∀ x y : ℝ, y = ax^2 + bx + c → y = 0 ∧ ((x = 1) ∨ (x = 5)))
  (h2 : ∃ x : ℝ, (∀ t : ℝ, ax^2 + bx + c ≤ t^2) ∧ (ax^2 + bx + c = 36)) :
  a + b + c = 0 :=
sorry

end parabola_min_value_sum_abc_zero_l107_107736


namespace miles_mike_ride_l107_107923

theorem miles_mike_ride
  (cost_per_mile : ℝ) (start_fee : ℝ) (bridge_toll : ℝ)
  (annie_miles : ℝ) (annie_total_cost : ℝ)
  (mike_total_cost : ℝ) (M : ℝ)
  (h1 : cost_per_mile = 0.25)
  (h2 : start_fee = 2.50)
  (h3 : bridge_toll = 5.00)
  (h4 : annie_miles = 26)
  (h5 : annie_total_cost = start_fee + bridge_toll + cost_per_mile * annie_miles)
  (h6 : mike_total_cost = start_fee + cost_per_mile * M)
  (h7 : mike_total_cost = annie_total_cost) :
  M = 36 := 
sorry

end miles_mike_ride_l107_107923


namespace max_cookies_ben_could_have_eaten_l107_107591

theorem max_cookies_ben_could_have_eaten (c : ℕ) (h_total : c = 36)
  (h_beth : ∃ n: ℕ, (n = 2 ∨ n = 3) ∧ c = (n + 1) * ben)
  (h_max : ∀ n, (n = 2 ∨ n = 3) → n * 12 ≤ n * ben)
  : ben = 12 := 
sorry

end max_cookies_ben_could_have_eaten_l107_107591


namespace min_A_div_B_l107_107664

theorem min_A_div_B (x A B : ℝ) (hx_pos : 0 < x) (hA_pos : 0 < A) (hB_pos : 0 < B) 
  (h1 : x^2 + 1 / x^2 = A) (h2 : x - 1 / x = B + 3) : 
  (A / B) = 6 + 2 * Real.sqrt 11 :=
sorry

end min_A_div_B_l107_107664


namespace number_of_solutions_l107_107018

theorem number_of_solutions :
  (∃ (xs : List ℤ), (∀ x ∈ xs, |3 * x + 4| ≤ 10) ∧ xs.length = 7) := sorry

end number_of_solutions_l107_107018


namespace wooden_easel_cost_l107_107148

noncomputable def cost_paintbrush : ℝ := 1.5
noncomputable def cost_set_of_paints : ℝ := 4.35
noncomputable def amount_already_have : ℝ := 6.5
noncomputable def additional_amount_needed : ℝ := 12
noncomputable def total_cost_items : ℝ := cost_paintbrush + cost_set_of_paints
noncomputable def total_amount_needed : ℝ := amount_already_have + additional_amount_needed

theorem wooden_easel_cost :
  total_amount_needed - total_cost_items = 12.65 :=
by
  sorry

end wooden_easel_cost_l107_107148


namespace necessary_and_sufficient_condition_l107_107383

theorem necessary_and_sufficient_condition (x : ℝ) : (|x - 2| < 1) ↔ (1 < x ∧ x < 3) := 
by
  sorry

end necessary_and_sufficient_condition_l107_107383


namespace find_m_b_l107_107143

noncomputable def line_equation (x y : ℝ) :=
  (⟨-1, 4⟩ : ℝ × ℝ) • (⟨x, y⟩ - ⟨3, -5⟩ : ℝ × ℝ) = 0

theorem find_m_b : ∃ m b : ℝ, (∀ (x y : ℝ), line_equation x y → y = m * x + b) ∧ m = 1 / 4 ∧ b = -23 / 4 :=
by
  sorry

end find_m_b_l107_107143


namespace worth_of_each_gift_is_4_l107_107033

noncomputable def worth_of_each_gift
  (workers_per_block : ℕ)
  (total_blocks : ℕ)
  (total_amount : ℝ) : ℝ :=
  total_amount / (workers_per_block * total_blocks)

theorem worth_of_each_gift_is_4 (workers_per_block total_blocks : ℕ) (total_amount : ℝ)
  (h1 : workers_per_block = 100)
  (h2 : total_blocks = 10)
  (h3 : total_amount = 4000) :
  worth_of_each_gift workers_per_block total_blocks total_amount = 4 :=
by
  sorry

end worth_of_each_gift_is_4_l107_107033


namespace Kim_min_score_for_target_l107_107256

noncomputable def Kim_exam_scores : List ℚ := [86, 82, 89]

theorem Kim_min_score_for_target :
  ∃ x : ℚ, ↑((Kim_exam_scores.sum + x) / (Kim_exam_scores.length + 1) ≥ (Kim_exam_scores.sum / Kim_exam_scores.length) + 2)
  ∧ x = 94 := sorry

end Kim_min_score_for_target_l107_107256


namespace probability_adjacent_vertices_of_decagon_l107_107098

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l107_107098


namespace mona_drives_125_miles_l107_107386

/-- Mona can drive 125 miles with $25 worth of gas, given the car mileage
    and the cost per gallon of gas. -/
theorem mona_drives_125_miles (miles_per_gallon : ℕ) (cost_per_gallon : ℕ) (total_money : ℕ)
  (h_miles_per_gallon : miles_per_gallon = 25) (h_cost_per_gallon : cost_per_gallon = 5)
  (h_total_money : total_money = 25) :
  (total_money / cost_per_gallon) * miles_per_gallon = 125 :=
by
  sorry

end mona_drives_125_miles_l107_107386


namespace part_i_part_ii_l107_107777

-- Define the operations for the weird calculator.
def Dsharp (n : ℕ) : ℕ := 2 * n + 1
def Dflat (n : ℕ) : ℕ := 2 * n - 1

-- Define the initial starting point.
def initial_display : ℕ := 1

-- Define a function to execute a sequence of button presses.
def execute_sequence (seq : List (ℕ → ℕ)) (initial : ℕ) : ℕ :=
  seq.foldl (fun x f => f x) initial

-- Problem (i): Prove there is a sequence that results in 313 starting from 1 after eight presses.
theorem part_i : ∃ seq : List (ℕ → ℕ), seq.length = 8 ∧ execute_sequence seq 1 = 313 :=
by sorry

-- Problem (ii): Describe all numbers that can be achieved from exactly eight button presses starting from 1.
theorem part_ii : 
  ∀ n : ℕ, n % 2 = 1 ∧ n < 2^9 →
  ∃ seq : List (ℕ → ℕ), seq.length = 8 ∧ execute_sequence seq 1 = n :=
by sorry

end part_i_part_ii_l107_107777


namespace decagon_adjacent_probability_l107_107127

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l107_107127


namespace octagon_area_is_six_and_m_plus_n_is_seven_l107_107749

noncomputable def area_of_octagon (side_length : ℕ) (segment_length : ℚ) : ℚ :=
  let triangle_area := 1 / 2 * side_length * segment_length
  let octagon_area := 8 * triangle_area
  octagon_area

theorem octagon_area_is_six_and_m_plus_n_is_seven :
  area_of_octagon 2 (3/4) = 6 ∧ (6 + 1 = 7) :=
by
  sorry

end octagon_area_is_six_and_m_plus_n_is_seven_l107_107749


namespace boys_in_class_l107_107598

theorem boys_in_class (total_students : ℕ) (fraction_girls : ℝ) (fraction_girls_eq : fraction_girls = 1 / 4) (total_students_eq : total_students = 160) :
  (total_students - fraction_girls * total_students = 120) :=
by
  rw [fraction_girls_eq, total_students_eq]
  -- Here, additional lines proving the steps would follow, but we use sorry for completeness.
  sorry

end boys_in_class_l107_107598


namespace no_real_intersections_l107_107844

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l107_107844


namespace sum_y_coords_of_circle_y_axis_points_l107_107624

theorem sum_y_coords_of_circle_y_axis_points 
  (h : ∀ x y : ℝ, (x + 3)^2 + (y - 5)^2 = 64) :
  (-3, 5).snd + sqrt 55 + (-3, 5).snd - sqrt 55 = 10 :=
by
  sorry

end sum_y_coords_of_circle_y_axis_points_l107_107624


namespace find_K_l107_107754

theorem find_K : ∃ K : ℕ, (64 ^ (2 / 3) * 16 ^ 2) / 4 = 2 ^ K :=
by
  use 10
  sorry

end find_K_l107_107754


namespace airport_exchange_rate_frac_l107_107301

variable (euros_received : ℕ) (euros : ℕ) (official_exchange_rate : ℕ) (dollars_received : ℕ)

theorem airport_exchange_rate_frac (h1 : euros = 70) (h2 : official_exchange_rate = 5) (h3 : dollars_received = 10) :
  (euros_received * dollars_received) = (euros * official_exchange_rate) →
  euros_received = 5 / 7 :=
  sorry

end airport_exchange_rate_frac_l107_107301


namespace smallest_digit_to_correct_sum_l107_107375

theorem smallest_digit_to_correct_sum :
  ∃ (d : ℕ), d = 3 ∧
  (3 ∈ [3, 5, 7]) ∧
  (371 + 569 + 784 + (d*100) = 1824) := sorry

end smallest_digit_to_correct_sum_l107_107375


namespace solve_equation_l107_107729

theorem solve_equation : ∀ x : ℝ, x * (x + 2) = 3 * x + 6 ↔ (x = -2 ∨ x = 3) := by
  sorry

end solve_equation_l107_107729


namespace min_log_value_l107_107522

theorem min_log_value (x y : ℝ) (h : 2 * x + 3 * y = 3) : ∃ (z : ℝ), z = Real.log (2^(4 * x) + 2^(3 * y)) / Real.log 2 ∧ z = 5 / 2 := 
by
  sorry

end min_log_value_l107_107522


namespace monica_studied_32_67_hours_l107_107712

noncomputable def monica_total_study_time : ℚ :=
  let monday := 1
  let tuesday := 2 * monday
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let total_weekday := monday + tuesday + wednesday + thursday + friday
  let saturday := total_weekday
  let sunday := saturday / 3
  total_weekday + saturday + sunday

theorem monica_studied_32_67_hours :
  monica_total_study_time = 32.67 := by
  sorry

end monica_studied_32_67_hours_l107_107712


namespace expand_polynomial_l107_107644

theorem expand_polynomial (x : ℝ) :
  (x + 4) * (x - 9) = x^2 - 5 * x - 36 := 
sorry

end expand_polynomial_l107_107644


namespace remaining_lawn_area_l107_107035

theorem remaining_lawn_area (lawn_length lawn_width path_width : ℕ) 
  (h_lawn_length : lawn_length = 10) 
  (h_lawn_width : lawn_width = 5) 
  (h_path_width : path_width = 1) : 
  (lawn_length * lawn_width - lawn_length * path_width) = 40 := 
by 
  sorry

end remaining_lawn_area_l107_107035


namespace remaining_sweet_cookies_correct_remaining_salty_cookies_correct_remaining_chocolate_cookies_correct_l107_107929

-- Definition of initial conditions
def initial_sweet_cookies := 34
def initial_salty_cookies := 97
def initial_chocolate_cookies := 45

def sweet_cookies_eaten := 15
def salty_cookies_eaten := 56
def chocolate_cookies_given_away := 22
def chocolate_cookies_given_back := 7

-- Calculate remaining cookies
def remaining_sweet_cookies : Nat := initial_sweet_cookies - sweet_cookies_eaten
def remaining_salty_cookies : Nat := initial_salty_cookies - salty_cookies_eaten
def remaining_chocolate_cookies : Nat := (initial_chocolate_cookies - chocolate_cookies_given_away) + chocolate_cookies_given_back

-- Theorem statements
theorem remaining_sweet_cookies_correct : remaining_sweet_cookies = 19 := 
by sorry

theorem remaining_salty_cookies_correct : remaining_salty_cookies = 41 := 
by sorry

theorem remaining_chocolate_cookies_correct : remaining_chocolate_cookies = 30 := 
by sorry

end remaining_sweet_cookies_correct_remaining_salty_cookies_correct_remaining_chocolate_cookies_correct_l107_107929


namespace circle_line_distance_range_l107_107349

open Real

theorem circle_line_distance_range (a : ℝ) :
  ∃ (x y : ℝ), (x^2 + y^2 = 4) ∧ (|x + y - a| = 1) ↔ -3 * sqrt 2 < a ∧ a < 3 * sqrt 2 :=
by
  sorry

end circle_line_distance_range_l107_107349


namespace missing_coins_l107_107791

-- Definition representing the total number of coins Charlie received
variable (y : ℚ)

-- Conditions
def initial_lost_coins (y : ℚ) := (1 / 3) * y
def recovered_coins (y : ℚ) := (2 / 9) * y

-- Main Theorem
theorem missing_coins (y : ℚ) :
  y - (y * (8 / 9)) = y * (1 / 9) :=
by
  sorry

end missing_coins_l107_107791


namespace problem_area_of_circle_l107_107413

noncomputable def circleAreaPortion : ℝ :=
  let r := Real.sqrt 59
  let theta := 135 * Real.pi / 180
  (theta / (2 * Real.pi)) * (Real.pi * r^2)

theorem problem_area_of_circle :
  circleAreaPortion = (177 / 8) * Real.pi := by
  sorry

end problem_area_of_circle_l107_107413


namespace quadratic_inequality_solution_l107_107797

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, k * x^2 + k * x - (3 / 4) < 0) ↔ -3 < k ∧ k ≤ 0 :=
by
  sorry

end quadratic_inequality_solution_l107_107797


namespace negation_of_exists_l107_107237

theorem negation_of_exists (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l107_107237


namespace greatest_odd_factors_l107_107713

theorem greatest_odd_factors (n : ℕ) : n < 200 ∧ (∃ m : ℕ, m * m = n) → n = 196 := by
  sorry

end greatest_odd_factors_l107_107713


namespace average_speed_of_train_l107_107775

theorem average_speed_of_train (x : ℝ) (h₀ : x > 0) :
  let time_1 := x / 40
  let time_2 := 2 * x / 20
  let total_time := time_1 + time_2
  let total_distance := 6 * x
  let avg_speed := total_distance / total_time
  avg_speed = 48 := by
  let time_1 := x / 40
  let time_2 := 2 * x / 20
  let total_time := time_1 + time_2
  let total_distance := 6 * x
  let avg_speed := total_distance / total_time
  sorry

end average_speed_of_train_l107_107775


namespace line_circle_no_intersection_l107_107860

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l107_107860


namespace jeremy_watermelons_l107_107541

theorem jeremy_watermelons :
  ∀ (total_watermelons : ℕ) (weeks : ℕ) (consumption_per_week : ℕ) (eaten_per_week : ℕ),
  total_watermelons = 30 →
  weeks = 6 →
  eaten_per_week = 3 →
  consumption_per_week = total_watermelons / weeks →
  (consumption_per_week - eaten_per_week) = 2 :=
by
  intros total_watermelons weeks consumption_per_week eaten_per_week h1 h2 h3 h4
  sorry

end jeremy_watermelons_l107_107541


namespace prime_p4_minus_one_sometimes_divisible_by_48_l107_107517

theorem prime_p4_minus_one_sometimes_divisible_by_48 (p : ℕ) (hp : Nat.Prime p) (hge : p ≥ 7) : 
  ∃ k : ℕ, k ≥ 1 ∧ 48 ∣ p^4 - 1 :=
sorry

end prime_p4_minus_one_sometimes_divisible_by_48_l107_107517


namespace no_monotonically_decreasing_l107_107704

variable (f : ℝ → ℝ)

theorem no_monotonically_decreasing (x1 x2 : ℝ) (h1 : ∃ x1 x2, x1 < x2 ∧ f x1 ≤ f x2) : ∀ x1 x2, x1 < x2 → f x1 > f x2 → False :=
by
  intros x1 x2 h2 h3
  obtain ⟨a, b, h4, h5⟩ := h1
  have contra := h5
  sorry

end no_monotonically_decreasing_l107_107704


namespace exists_n_lt_p_minus_1_not_div_p2_l107_107919

theorem exists_n_lt_p_minus_1_not_div_p2 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_3 : p > 3) :
  ∃ (n : ℕ), n < p - 1 ∧ ¬(p^2 ∣ (n^((p - 1)) - 1)) ∧ ¬(p^2 ∣ ((n + 1)^((p - 1)) - 1)) := 
sorry

end exists_n_lt_p_minus_1_not_div_p2_l107_107919


namespace part_a_part_b_l107_107306

-- Definition for the number of triangles when the n-gon is divided using non-intersecting diagonals
theorem part_a (n : ℕ) (h : n ≥ 3) : 
  ∃ k, k = n - 2 := 
sorry

-- Definition for the number of diagonals when the n-gon is divided using non-intersecting diagonals
theorem part_b (n : ℕ) (h : n ≥ 3) : 
  ∃ l, l = n - 3 := 
sorry

end part_a_part_b_l107_107306


namespace Jims_apples_fits_into_average_l107_107900

def Jim_apples : Nat := 20
def Jane_apples : Nat := 60
def Jerry_apples : Nat := 40

def total_apples : Nat := Jim_apples + Jane_apples + Jerry_apples
def number_of_people : Nat := 3
def average_apples_per_person : Nat := total_apples / number_of_people

theorem Jims_apples_fits_into_average :
  average_apples_per_person / Jim_apples = 2 := by
  sorry

end Jims_apples_fits_into_average_l107_107900


namespace daily_expenses_increase_l107_107295

theorem daily_expenses_increase 
  (init_students : ℕ) (new_students : ℕ) (diminish_amount : ℝ) (orig_expenditure : ℝ)
  (orig_expenditure_eq : init_students = 35)
  (new_students_eq : new_students = 42)
  (diminish_amount_eq : diminish_amount = 1)
  (orig_expenditure_val : orig_expenditure = 400)
  (orig_average_expenditure : ℝ) (increase_expenditure : ℝ)
  (orig_avg_calc : orig_average_expenditure = orig_expenditure / init_students)
  (new_total_expenditure : ℝ)
  (new_expenditure_eq : new_total_expenditure = orig_expenditure + increase_expenditure) :
  (42 * (orig_average_expenditure - diminish_amount) = new_total_expenditure) → increase_expenditure = 38 := 
by 
  sorry

end daily_expenses_increase_l107_107295


namespace decagon_adjacent_probability_l107_107126

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l107_107126


namespace square_side_measurement_error_l107_107331

theorem square_side_measurement_error {S S' : ℝ} (h1 : S' = S * Real.sqrt 1.0816) :
  ((S' - S) / S) * 100 = 4 := by
  sorry

end square_side_measurement_error_l107_107331


namespace cost_of_burger_l107_107781

theorem cost_of_burger :
  ∃ (b s f : ℕ), 
    4 * b + 3 * s + f = 540 ∧
    3 * b + 2 * s + 2 * f = 580 ∧
    b = 100 :=
by {
  sorry
}

end cost_of_burger_l107_107781


namespace triangle_side_lengths_l107_107202

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end triangle_side_lengths_l107_107202


namespace integer_solution_pairs_l107_107000

theorem integer_solution_pairs (a b : ℕ) (h_pos : a > 0 ∧ b > 0):
  (∃ k : ℕ, k > 0 ∧ a^2 = k * (2 * a * b^2 - b^3 + 1)) ↔ 
  (∃ l : ℕ, l > 0 ∧ ((a = 2 * l ∧ b = 1) ∨ (a = l ∧ b = 2 * l) ∨ (a = 8 * l^4 - l ∧ b = 2 * l))) :=
sorry

end integer_solution_pairs_l107_107000


namespace days_playing_video_games_l107_107050

-- Define the conditions
def watchesTVDailyHours : ℕ := 4
def videoGameHoursPerPlay : ℕ := 2
def totalWeeklyHours : ℕ := 34
def weeklyTVDailyHours : ℕ := 7 * watchesTVDailyHours

-- Define the number of days playing video games
def playsVideoGamesDays (d : ℕ) : ℕ := d * videoGameHoursPerPlay

-- Define the number of days Mike plays video games
theorem days_playing_video_games (d : ℕ) :
  weeklyTVDailyHours + playsVideoGamesDays d = totalWeeklyHours → d = 3 :=
by
  -- The proof is omitted
  sorry

end days_playing_video_games_l107_107050


namespace remaining_area_correct_l107_107768

-- Define the side lengths of the large rectangle
def large_rectangle_length1 (x : ℝ) := 2 * x + 5
def large_rectangle_length2 (x : ℝ) := x + 8

-- Define the side lengths of the rectangular hole
def hole_length1 (x : ℝ) := 3 * x - 2
def hole_length2 (x : ℝ) := x + 1

-- Define the area of the large rectangle
def large_rectangle_area (x : ℝ) := (large_rectangle_length1 x) * (large_rectangle_length2 x)

-- Define the area of the hole
def hole_area (x : ℝ) := (hole_length1 x) * (hole_length2 x)

-- Prove the remaining area after accounting for the hole
theorem remaining_area_correct (x : ℝ) : 
  large_rectangle_area x - hole_area x = -x^2 + 20 * x + 42 := 
  by 
    sorry

end remaining_area_correct_l107_107768


namespace sum_of_fractions_to_decimal_l107_107622

theorem sum_of_fractions_to_decimal :
  ((2 / 40 : ℚ) + (4 / 80) + (6 / 120) + (9 / 180) : ℚ) = 0.2 :=
by
  sorry

end sum_of_fractions_to_decimal_l107_107622


namespace max_planes_determined_by_15_points_l107_107444

theorem max_planes_determined_by_15_points : ∃ (n : ℕ), n = 455 ∧ 
  ∀ (P : Finset (Fin 15)), (15 + 1) ∣ (P.card * (P.card - 1) * (P.card - 2) / 6) → n = (15 * 14 * 13) / 6 :=
by
  sorry

end max_planes_determined_by_15_points_l107_107444


namespace expression_value_l107_107755

theorem expression_value (x y : ℝ) (h : x - y = 1) : 
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 := 
by
  sorry

end expression_value_l107_107755


namespace count_numbers_with_square_factors_l107_107681

theorem count_numbers_with_square_factors :
  let S := {n | 1 ≤ n ∧ n ≤ 100}
      factors := [4, 9, 16, 25, 36, 49, 64]
      has_square_factor := λ n => ∃ f ∈ factors, f ∣ n
  in (S.filter has_square_factor).card = 40 := 
by 
  sorry

end count_numbers_with_square_factors_l107_107681


namespace fuel_spending_reduction_l107_107813

-- Define the variables and the conditions
variable (x c : ℝ) -- x for efficiency and c for cost
variable (newEfficiency oldEfficiency newCost oldCost : ℝ)

-- Define the conditions
def conditions := (oldEfficiency = x) ∧ (newEfficiency = 1.75 * oldEfficiency)
                 ∧ (oldCost = c) ∧ (newCost = 1.3 * oldCost)

-- Define the expected reduction in cost
def expectedReduction : ℝ := 25.7142857142857 -- approximately 25 5/7 %

-- Define the assertion that Elmer will reduce his fuel spending by the expected reduction percentage
theorem fuel_spending_reduction : conditions x c oldEfficiency newEfficiency oldCost newCost →
  ((oldCost - (newCost / newEfficiency) * oldEfficiency) / oldCost) * 100 = expectedReduction :=
by
  sorry

end fuel_spending_reduction_l107_107813


namespace quadratic_rewrite_l107_107922

theorem quadratic_rewrite :
  ∃ d e f : ℤ, (4 * (x : ℝ)^2 - 24 * x + 35 = (d * x + e)^2 + f) ∧ (d * e = -12) :=
by
  sorry

end quadratic_rewrite_l107_107922


namespace largest_y_coordinate_l107_107503

theorem largest_y_coordinate (x y : ℝ) (h : (x^2 / 49) + ((y - 3)^2 / 25) = 0) : y = 3 :=
sorry

end largest_y_coordinate_l107_107503


namespace Maaza_liters_l107_107972

theorem Maaza_liters 
  (M L : ℕ)
  (Pepsi : ℕ := 144)
  (Sprite : ℕ := 368)
  (total_liters := M + Pepsi + Sprite)
  (cans_required : ℕ := 281)
  (H : total_liters = cans_required * L)
  : M = 50 :=
by
  sorry

end Maaza_liters_l107_107972


namespace sales_neither_notebooks_nor_markers_l107_107731

theorem sales_neither_notebooks_nor_markers (percent_notebooks percent_markers percent_staplers : ℝ) 
  (h1 : percent_notebooks = 25)
  (h2 : percent_markers = 40)
  (h3 : percent_staplers = 15) : 
  percent_staplers + (100 - (percent_notebooks + percent_markers + percent_staplers)) = 35 :=
by
  sorry

end sales_neither_notebooks_nor_markers_l107_107731


namespace probability_even_sum_l107_107481

noncomputable def twelvePrimes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

noncomputable def even_sum_probability :=
  have total_combinations := (Finset.choose 12 5).val
  have valid_combinations := (Finset.choose 11 4).val
  (valid_combinations : ℚ) / total_combinations

theorem probability_even_sum :
  even_sum_probability = 55 / 132 :=
by
  sorry

end probability_even_sum_l107_107481


namespace sin_triple_alpha_minus_beta_l107_107494

open Real 

theorem sin_triple_alpha_minus_beta (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : π / 2 < β ∧ β < π)
  (h1 : cos (α - β) = 1 / 2)
  (h2 : sin (α + β) = 1 / 2) :
  sin (3 * α - β) = 1 / 2 :=
by
  sorry

end sin_triple_alpha_minus_beta_l107_107494


namespace no_hamiltonian_circuit_in_G_l107_107733

-- Suppose we have a specific graph G with vertices and specific edges
def G : SimpleGraph := {
  adj := λ x y,
    (x = 'A' ∧ y = 'B') ∨ (x = 'B' ∧ y = 'A') ∨
    (x = 'B' ∧ y = 'C') ∨ (x = 'C' ∧ y = 'B') ∨
    (x = 'C' ∧ y = 'D') ∨ (x = 'D' ∧ y = 'C') ∨
    (x = 'D' ∧ y = 'A') ∨ (x = 'A' ∧ y = 'D')
}

-- The vertices are {'A', 'B', 'C', 'D'}
def V := ['A', 'B', 'C', 'D']

-- The main theorem stating that no Hamiltonian circuit exists in the graph G
theorem no_hamiltonian_circuit_in_G : ¬∃ c : List V, G.isHamiltonianCircuit c :=
by sorry

end no_hamiltonian_circuit_in_G_l107_107733


namespace area_triangle_parabola_l107_107257

noncomputable def area_of_triangle_ABC (d : ℝ) (x : ℝ) : ℝ :=
  let A := (x, x^2)
  let B := (x + d, (x + d)^2)
  let C := (x + 2 * d, (x + 2 * d)^2)
  1 / 2 * abs (x * ((x + 2 * d)^2 - (x + d)^2) + (x + d) * ((x + 2 * d)^2 - x^2) + (x + 2 * d) * (x^2 - (x + d)^2))

theorem area_triangle_parabola (d : ℝ) (h_d : 0 < d) (x : ℝ) : 
  area_of_triangle_ABC d x = d^2 := sorry

end area_triangle_parabola_l107_107257


namespace sqrt_expression_eval_l107_107465

theorem sqrt_expression_eval :
  (Real.sqrt 48 / Real.sqrt 3) - (Real.sqrt (1 / 6) * Real.sqrt 12) + Real.sqrt 24 = 4 - Real.sqrt 2 + 2 * Real.sqrt 6 :=
by
  sorry

end sqrt_expression_eval_l107_107465


namespace interest_earned_after_4_years_l107_107779

noncomputable def calculate_total_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  let A := P * (1 + r) ^ t
  A - P

theorem interest_earned_after_4_years :
  calculate_total_interest 2000 0.12 4 = 1147.04 :=
by
  sorry

end interest_earned_after_4_years_l107_107779


namespace find_DP_l107_107079

theorem find_DP (AP BP CP DP : ℚ) (h1 : AP = 4) (h2 : BP = 6) (h3 : CP = 9) (h4 : AP * BP = CP * DP) :
  DP = 8 / 3 :=
by
  rw [h1, h2, h3] at h4
  sorry

end find_DP_l107_107079


namespace quadratic_sum_is_zero_l107_107740

-- Definition of a quadratic function with given conditions and the final result to prove
theorem quadratic_sum_is_zero {a b c : ℝ} 
  (h₁ : ∀ x : ℝ, x = 3 → a * (x - 1) * (x - 5) = 36) 
  (h₂ : a * 1^2 + b * 1 + c = 0) 
  (h₃ : a * 5^2 + b * 5 + c = 0) : 
  a + b + c = 0 := 
sorry

end quadratic_sum_is_zero_l107_107740


namespace ratio_a7_b7_l107_107824

variable (a b : ℕ → ℕ) -- Define sequences a and b
variable (S T : ℕ → ℕ) -- Define sums S and T

-- Define conditions: arithmetic sequences and given ratio
variable (h_arith_a : ∀ n, a (n + 1) - a n = a 1)
variable (h_arith_b : ∀ n, b (n + 1) - b n = b 1)
variable (h_sum_a : ∀ n, S n = (n + 1) * a 1 + n * a n)
variable (h_sum_b : ∀ n, T n = (n + 1) * b 1 + n * b n)
variable (h_ratio : ∀ n, (S n) / (T n) = (3 * n + 2) / (2 * n))

-- Define the problem statement using the given conditions
theorem ratio_a7_b7 : (a 7) / (b 7) = 41 / 26 :=
by
  sorry

end ratio_a7_b7_l107_107824


namespace probability_same_color_l107_107476

-- Definitions for the conditions
def blue_balls : Nat := 8
def yellow_balls : Nat := 5
def total_balls : Nat := blue_balls + yellow_balls

def prob_two_balls_same_color : ℚ :=
  (blue_balls/total_balls) * (blue_balls/total_balls) + (yellow_balls/total_balls) * (yellow_balls/total_balls)

-- Lean statement to be proved
theorem probability_same_color : prob_two_balls_same_color = 89 / 169 :=
by
  -- The proof is omitted as per the instruction
  sorry

end probability_same_color_l107_107476


namespace hypothesis_test_l107_107615

def X : List ℕ := [3, 4, 6, 10, 13, 17]
def Y : List ℕ := [1, 2, 5, 7, 16, 20, 22]

def alpha : ℝ := 0.01
def W_lower : ℕ := 24
def W_upper : ℕ := 60
def W1 : ℕ := 41

-- stating the null hypothesis test condition
theorem hypothesis_test : (24 < 41) ∧ (41 < 60) :=
by
  sorry

end hypothesis_test_l107_107615


namespace area_ratio_of_circles_l107_107028

-- Define the circles and lengths of arcs
variables {R_C R_D : ℝ} (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D))

-- Theorem proving the ratio of the areas
theorem area_ratio_of_circles (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 := sorry

end area_ratio_of_circles_l107_107028


namespace solve_equation_l107_107065

theorem solve_equation (x : ℂ) (h : (x^2 + 3*x + 4) / (x + 3) = x + 6) : x = -7 / 3 := sorry

end solve_equation_l107_107065


namespace sides_of_nth_hexagon_l107_107513

-- Definition of the arithmetic sequence condition.
def first_term : ℕ := 6
def common_difference : ℕ := 5

-- The function representing the n-th term of the sequence.
def num_sides (n : ℕ) : ℕ := first_term + (n - 1) * common_difference

-- Now, we state the theorem that the n-th term equals 5n + 1.
theorem sides_of_nth_hexagon (n : ℕ) : num_sides n = 5 * n + 1 := by
  sorry

end sides_of_nth_hexagon_l107_107513


namespace possible_integer_side_lengths_l107_107217

theorem possible_integer_side_lengths (a b : ℕ) (h1 : a = 8) (h2 : b = 5) :
  {x : ℕ | 3 < x ∧ x < 13}.finite ∧ {x : ℕ | 3 < x ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l107_107217


namespace max_value_of_M_l107_107822

def J (k : ℕ) := 10^(k + 3) + 256

def M (k : ℕ) := Nat.factors (J k) |>.count 2

theorem max_value_of_M (k : ℕ) (hk : k > 0) :
  M k = 8 := by
  sorry

end max_value_of_M_l107_107822


namespace no_real_solutions_cubic_eq_l107_107645

theorem no_real_solutions_cubic_eq : ∀ x : ℝ, ¬ (∃ (y : ℝ), y = x^(1/3) ∧ y = 15 / (6 - y)) :=
by
  intro x
  intro hexist
  obtain ⟨y, hy1, hy2⟩ := hexist
  have h_cubic : y * (6 - y) = 15 := by sorry -- from y = 15 / (6 - y)
  have h_quad : y^2 - 6 * y + 15 = 0 := by sorry -- after expanding y(6 - y) = 15
  sorry -- remainder to show no real solution due to negative discriminant

end no_real_solutions_cubic_eq_l107_107645


namespace find_B_l107_107248

noncomputable def triangle_angle_B (a b : ℝ) (A : ℝ) : ℝ :=
  let sin_B := (b * Real.sin (A * Real.pi / 180)) / a
  in if sin_B = Real.sin (60 * Real.pi / 180) then 60 else 120

theorem find_B (a b : ℝ) (A B : ℝ) (h_a : a = 4) (h_b : b = 4 * Real.sqrt 3) (h_A : A = 30) :
  B = 60 ∨ B = 120 := by
  simp [h_a, h_b, h_A]
  have h_sinA : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
    simp [Real.sin_pi_div_two_half]
  have h_sinB := (4 * Real.sqrt 3 * (1 / 2)) / 4
  simp [triangle_angle_B, h_sinA, Real.sin_pi_div_three, Real.sqrt_eq_rpow, Real.rpow_div, Real.rpow_one, Complex.pi_ne_zero, Real.sin_eq, h_sinB ]
  have h₁ : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := Real.sin_pi_div_three
  have h₂ : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 := Real.sin_tpi_div_3_eq_pi_div_3
  simp [h₁, h₂]
  exact Or.inl rfl -- due to the integration range on h₁
  sorry -- skip remaining proof steps

end find_B_l107_107248


namespace normal_distribution_interval_probability_l107_107372

open ProbabilityTheory

theorem normal_distribution_interval_probability 
  (σ : ℝ) (hσ : σ > 0) (h0 : ProbabilityTheory.PDF (Normal 1 σ^2) (set.Ioo 0 1) = 0.4) :
  ProbabilityTheory.PDF (Normal 1 σ^2) (set.Ioo 0 2) = 0.8 :=
sorry

end normal_distribution_interval_probability_l107_107372


namespace competition_result_l107_107560

-- Define the participants
inductive Person
| Olya | Oleg | Pasha
deriving DecidableEq, Repr

-- Define the placement as an enumeration
inductive Place
| first | second | third
deriving DecidableEq, Repr

-- Define the statements
structure Statements :=
(olyas_claim : Place)
(olyas_statement : Prop)
(olegs_statement : Prop)

-- Define the conditions
def conditions (s : Statements) : Prop :=
  -- All claimed first place
  s.olyas_claim = Place.first ∧ s.olyas_statement ∧ s.olegs_statement

-- Define the final placement
structure Placement :=
(olyas_place : Place)
(olegs_place : Place)
(pashas_place : Place)

-- Define the correct answer
def correct_placement : Placement :=
{ olyas_place := Place.third,
  olegs_place := Place.first,
  pashas_place := Place.second }

-- Lean statement for the problem
theorem competition_result (s : Statements) (h : conditions s) : Placement :=
{ olyas_place := Place.third,
  olegs_place := Place.first,
  pashas_place := Place.second } := sorry

end competition_result_l107_107560


namespace scientific_notation_350_million_l107_107488

theorem scientific_notation_350_million : 350000000 = 3.5 * 10^8 := 
  sorry

end scientific_notation_350_million_l107_107488


namespace digit_in_92nd_place_l107_107684

/-- The fraction 5/33 is expressed in decimal form as a repeating decimal 0.151515... -/
def fraction_to_decimal : ℚ := 5 / 33

/-- The repeated pattern in the decimal expansion of 5/33 is 15, which is a cycle of length 2 -/
def repeated_pattern (n : ℕ) : ℕ :=
  if n % 2 = 0 then 5 else 1

/-- The digit at the 92nd place in the decimal expansion of 5/33 is 5 -/
theorem digit_in_92nd_place : repeated_pattern 92 = 5 :=
by sorry

end digit_in_92nd_place_l107_107684


namespace line_circle_no_intersection_l107_107882

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l107_107882


namespace car_distance_and_velocity_l107_107428

def acceleration : ℝ := 12 -- constant acceleration in m/s^2
def time : ℝ := 36 -- time in seconds
def conversion_factor : ℝ := 3.6 -- conversion factor from m/s to km/h

theorem car_distance_and_velocity :
  (1/2 * acceleration * time^2 = 7776) ∧ (acceleration * time * conversion_factor = 1555.2) :=
by
  sorry

end car_distance_and_velocity_l107_107428


namespace tan_eq_tan_of_period_for_405_l107_107478

theorem tan_eq_tan_of_period_for_405 (m : ℤ) (h : -180 < m ∧ m < 180) :
  (Real.tan (m * (Real.pi / 180))) = (Real.tan (405 * (Real.pi / 180))) ↔ m = 45 ∨ m = -135 :=
by sorry

end tan_eq_tan_of_period_for_405_l107_107478


namespace calculate_triple_transform_l107_107629

def transformation (N : ℝ) : ℝ :=
  0.4 * N + 2

theorem calculate_triple_transform :
  transformation (transformation (transformation 20)) = 4.4 :=
by
  sorry

end calculate_triple_transform_l107_107629


namespace count_numbers_with_perfect_square_factors_l107_107676

theorem count_numbers_with_perfect_square_factors:
  let S := {n | n ∈ Finset.range (100 + 1)}
  let perfect_squares := {4, 9, 16, 25, 36, 49, 64, 81}
  let has_perfect_square_factor := λ n, ∃ m ∈ perfect_squares, m ≤ n ∧ n % m = 0
  (Finset.filter has_perfect_square_factor S).card = 40 := by
  sorry

end count_numbers_with_perfect_square_factors_l107_107676


namespace xy_value_l107_107024

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 10) : x * y = 10 :=
by
  sorry

end xy_value_l107_107024


namespace at_least_one_inequality_false_l107_107965

open Classical

theorem at_least_one_inequality_false (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
  ¬ (a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) :=
by
  sorry

end at_least_one_inequality_false_l107_107965


namespace find_numbers_l107_107581

variables {x y : ℤ}

theorem find_numbers (x y : ℤ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 185) (h3 : (x - y)^2 = 121) :
  (x = 13 ∧ y = 2) ∨ (x = -5 ∧ y = -16) :=
sorry

end find_numbers_l107_107581


namespace min_episodes_to_watch_l107_107578

theorem min_episodes_to_watch (T W H F Sa Su M trip_days total_episodes: ℕ)
  (hW: W = 1) (hTh: H = 1) (hF: F = 1) (hSa: Sa = 2) (hSu: Su = 2) (hMo: M = 0)
  (total_episodes_eq: total_episodes = 60)
  (trip_days_eq: trip_days = 17):
  total_episodes - ((4 * W + 2 * Sa + 1 * M) * (trip_days / 7) + (trip_days % 7) * (W + Sa + Su + Mo)) = 39 := 
by
  sorry

end min_episodes_to_watch_l107_107578


namespace longest_playing_time_l107_107528

theorem longest_playing_time (total_playtime : ℕ) (n : ℕ) (k : ℕ) (standard_time : ℚ) (long_time : ℚ) :
  total_playtime = 120 ∧ n = 6 ∧ k = 2 ∧ long_time = k * standard_time →
  5 * standard_time + long_time = 240 →
  long_time = 68 :=
by
  sorry

end longest_playing_time_l107_107528


namespace perfect_square_factors_l107_107677

theorem perfect_square_factors (n : ℕ) (h₁ : n ≤ 100) (h₂ : n > 0) : 
  (∃ k : ℕ, k ∈ {4, 9, 16, 25, 36, 49, 64, 81} ∧ k ∣ n) →
  ∃ count : ℕ, count = 40 :=
sorry

end perfect_square_factors_l107_107677


namespace triangle_side_length_l107_107247

theorem triangle_side_length 
  (X Z : ℝ) (x z y : ℝ)
  (h1 : x = 36)
  (h2 : z = 72)
  (h3 : Z = 4 * X) :
  y = 72 := by
  sorry

end triangle_side_length_l107_107247


namespace largest_value_is_D_l107_107083

theorem largest_value_is_D :
  let A := 15432 + 1/3241
  let B := 15432 - 1/3241
  let C := 15432 * (1/3241)
  let D := 15432 / (1/3241)
  let E := 15432.3241
  max (max (max A B) (max C D)) E = D := by
{
  sorry -- proof not required
}

end largest_value_is_D_l107_107083


namespace six_digit_even_numbers_count_l107_107946

theorem six_digit_even_numbers_count :
  let digits : Finset ℕ := {1, 2, 3, 4, 5, 6},
      even_digits := {2, 4, 6},
      positions := Finset.range 6 in
  ∃ count : ℕ,
  count = 360 ∧
  ∀ (perm : List ℕ), perm ∈ List.permutations digits.to_list →
                       perm.length = 6 →
                       (perm.last ∈ even_digits) →
                       (1 ∈ perm ∧ 3 ∈ perm) →
                       ¬ (List.indexOf 1 perm + 1 = List.indexOf 3 perm ∨
                          List.indexOf 3 perm + 1 = List.indexOf 1 perm) →
  List.countp (λ p : List ℕ, true) [perm] = count := by
  sorry

end six_digit_even_numbers_count_l107_107946


namespace union_set_equiv_l107_107493

namespace ProofProblem

-- Define the sets A and B
def A : Set ℝ := { x | x - 1 > 0 }
def B : Set ℝ := { x | x^2 - x - 2 > 0 }

-- Define the union of A and B
def unionAB : Set ℝ := A ∪ B

-- State the proof problem
theorem union_set_equiv : unionAB = (Set.Iio (-1)) ∪ (Set.Ioi 1) := by
  sorry

end ProofProblem

end union_set_equiv_l107_107493


namespace full_time_worked_year_l107_107964

-- Define the conditions as constants
def total_employees : ℕ := 130
def full_time : ℕ := 80
def worked_year : ℕ := 100
def neither : ℕ := 20

-- Define the question as a theorem stating the correct answer
theorem full_time_worked_year : full_time + worked_year - total_employees + neither = 70 :=
by
  sorry

end full_time_worked_year_l107_107964


namespace myrtle_eggs_count_l107_107051

-- Definition for daily egg production
def daily_eggs : ℕ := 3 * 3

-- Definition for the number of days Myrtle is gone
def days_gone : ℕ := 7

-- Definition for total eggs laid
def total_eggs : ℕ := daily_eggs * days_gone

-- Definition for eggs taken by neighbor
def eggs_taken_by_neighbor : ℕ := 12

-- Definition for eggs remaining after neighbor takes some
def eggs_after_neighbor : ℕ := total_eggs - eggs_taken_by_neighbor

-- Definition for eggs dropped by Myrtle
def eggs_dropped_by_myrtle : ℕ := 5

-- Definition for total remaining eggs Myrtle has
def eggs_remaining : ℕ := eggs_after_neighbor - eggs_dropped_by_myrtle

-- Theorem statement
theorem myrtle_eggs_count : eggs_remaining = 46 := by
  sorry

end myrtle_eggs_count_l107_107051


namespace no_valid_N_for_case1_valid_N_values_for_case2_l107_107373

variable (P R N : ℕ)
variable (N_less_than_40 : N < 40)
variable (avg_all : ℕ)
variable (avg_promoted : ℕ)
variable (avg_repeaters : ℕ)
variable (new_avg_promoted : ℕ)
variable (new_avg_repeaters : ℕ)

variables
  (promoted_condition : (71 * P + 56 * R) / N = 66)
  (increase_condition : (76 * P) / (P + R) = 75 ∧ (61 * R) / (P + R) = 59)
  (equation1 : 71 * P = 2 * R)
  (equation2: P + R = N)

-- Proof for part (a)
theorem no_valid_N_for_case1 
  (new_avg_promoted' : ℕ := 75) 
  (new_avg_repeaters' : ℕ := 59)
  : ∀ N, ¬ N < 40 ∨ ¬ ((76 * P) / (P + R) = new_avg_promoted' ∧ (61 * R) / (P + R) = new_avg_repeaters') := 
  sorry

-- Proof for part (b)
theorem valid_N_values_for_case2
  (possible_N_values : Finset ℕ := {6, 12, 18, 24, 30, 36})
  (new_avg_promoted'' : ℕ := 79)
  (new_avg_repeaters'' : ℕ := 47)
  : ∀ N, N ∈ possible_N_values ↔ (((76 * P) / (P + R) = new_avg_promoted'') ∧ (61 * R) / (P + R) = new_avg_repeaters'') := 
  sorry

end no_valid_N_for_case1_valid_N_values_for_case2_l107_107373


namespace square_area_proof_square_area_square_area_final_square_area_correct_l107_107055

theorem square_area_proof (x : ℝ) (s1 : ℝ) (s2 : ℝ) (A : ℝ)
  (h1 : s1 = 5 * x - 20)
  (h2 : s2 = 25 - 2 * x)
  (h3 : s1 = s2) :
  A = (s1 * s1) := by
  -- We need to prove A = s1 * s1
  sorry

theorem square_area (x : ℝ) (s : ℝ) (h : s = 85 / 7) :
  s ^ 2 = 7225 / 49 := by
  -- We need to prove s^2 = 7225 / 49
  sorry

theorem square_area_final (x : ℝ)
  (h1 : 5 * x - 20 = 25 - 2 * x)
  (A : ℝ) :
  A = (85 / 7) ^ 2 := by
  -- We need to prove A = (85 / 7) ^ 2
  sorry

theorem square_area_correct (x : ℝ)
  (A : ℝ)
  (h1 : 5 * x - 20 = 25 - 2 * x)
  (h2 : A = (85 / 7) ^ 2) :
  A = 7225 / 49 := by
  -- We need to prove A = 7225 / 49
  sorry

end square_area_proof_square_area_square_area_final_square_area_correct_l107_107055


namespace sum_rational_irrational_not_rational_l107_107727

theorem sum_rational_irrational_not_rational (r i : ℚ) (hi : ¬ ∃ q : ℚ, i = q) : ¬ ∃ s : ℚ, r + i = s :=
by
  sorry

end sum_rational_irrational_not_rational_l107_107727


namespace female_democrats_count_l107_107305

theorem female_democrats_count :
  ∃ (F : ℕ) (M : ℕ),
    F + M = 750 ∧
    (F / 2) + (M / 4) = 250 ∧
    1 / 3 * 750 = 250 ∧
    F / 2 = 125 := sorry

end female_democrats_count_l107_107305


namespace digit_7_occurrences_in_range_20_to_199_l107_107538

open Set

noncomputable def countDigitOccurrences (low high : ℕ) (digit : ℕ) : ℕ :=
  sorry

theorem digit_7_occurrences_in_range_20_to_199 : 
  countDigitOccurrences 20 199 7 = 38 := 
by
  sorry

end digit_7_occurrences_in_range_20_to_199_l107_107538


namespace shaded_area_l107_107323

theorem shaded_area (x1 y1 x2 y2 x3 y3 : ℝ) 
  (vA vB vC vD vE vF : ℝ × ℝ)
  (h1 : vA = (0, 0))
  (h2 : vB = (0, 12))
  (h3 : vC = (12, 12))
  (h4 : vD = (12, 0))
  (h5 : vE = (24, 0))
  (h6 : vF = (18, 12))
  (h_base : 32 - 12 = 20)
  (h_height : 12 = 12) :
  (1 / 2 : ℝ) * 20 * 12 = 120 :=
by
  sorry

end shaded_area_l107_107323


namespace total_points_first_half_l107_107529

def geometric_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (1 - r ^ n) / (1 - r)

def arithmetic_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * a + d * (n * (n - 1) / 2)

-- Given conditions:
variables (a r b d : ℕ)
variables (h1 : a = b)
variables (h2 : geometric_sum a r 4 = b + (b + d) + (b + 2 * d) + (b + 3 * d) + 2)
variables (h3 : a * (1 + r + r^2 + r^3) ≤ 120)
variables (h4 : b + (b + d) + (b + 2 * d) + (b + 3 * d) ≤ 120)

theorem total_points_first_half (a r b d : ℕ) (h1 : a = b) (h2 : a * (1 + r + r ^ 2 + r ^ 3) = b + (b + d) + (b + 2 * d) + (b + 3 * d) + 2)
  (h3 : a * (1 + r + r ^ 2 + r ^ 3) ≤ 120) (h4 : b + (b + d) + (b + 2 * d) + (b + 3 * d) ≤ 120) : 
  a + a * r + b + (b + d) = 45 :=
by
  sorry

end total_points_first_half_l107_107529


namespace power_of_a_point_l107_107545

noncomputable def PA : ℝ := 4
noncomputable def PB : ℝ := 14 + 2 * Real.sqrt 13
noncomputable def PT : ℝ := PB - 8
noncomputable def AB : ℝ := PB - PA

theorem power_of_a_point (PA PB PT : ℝ) (h1 : PA = 4) (h2 : PB = 14 + 2 * Real.sqrt 13) (h3 : PT = PB - 8) : 
  PA * PB = PT * PT :=
by
  rw [h1, h2, h3]
  sorry

end power_of_a_point_l107_107545


namespace question_equals_answer_l107_107170

def heartsuit (a b : ℤ) : ℤ := |a + b|

theorem question_equals_answer : heartsuit (-3) (heartsuit 5 (-8)) = 0 := 
by
  sorry

end question_equals_answer_l107_107170


namespace min_S_min_S_values_range_of_c_l107_107652

-- Part 1
theorem min_S (a b c : ℝ) (h : a + b + c = 1) : 
  2 * a^2 + 3 * b^2 + c^2 ≥ (6 / 11) :=
sorry

-- Part 1, finding exact values of a, b, c where minimum is reached
theorem min_S_values (a b c : ℝ) (h : a + b + c = 1) :
  2 * a^2 + 3 * b^2 + c^2 = (6 / 11) ↔ a = (3 / 11) ∧ b = (2 / 11) ∧ c = (6 / 11) :=
sorry
  
-- Part 2
theorem range_of_c (a b c : ℝ) (h1 : 2 * a^2 + 3 * b^2 + c^2 = 1) : 
  (1 / 11) ≤ c ∧ c ≤ 1 :=
sorry

end min_S_min_S_values_range_of_c_l107_107652


namespace jellybeans_initial_amount_l107_107800

theorem jellybeans_initial_amount (x : ℝ) 
  (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_initial_amount_l107_107800


namespace faye_age_l107_107799
open Nat

theorem faye_age :
  ∃ (C D E F : ℕ), 
    (D = E - 3) ∧ 
    (E = C + 4) ∧ 
    (F = C + 3) ∧ 
    (D = 14) ∧ 
    (F = 16) :=
by
  sorry

end faye_age_l107_107799


namespace sum_m_n_l107_107194

open Real

noncomputable def f (x : ℝ) : ℝ := |log x / log 2|

theorem sum_m_n (m n : ℝ) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_mn : m < n) 
  (h_f_eq : f m = f n) (h_max_f : ∀ x : ℝ, m^2 ≤ x ∧ x ≤ n → f x ≤ 2) :
  m + n = 5 / 2 :=
sorry

end sum_m_n_l107_107194


namespace calculate_expression_l107_107620

theorem calculate_expression : (36 / (9 + 2 - 6)) * 4 = 28.8 := 
by
    sorry

end calculate_expression_l107_107620


namespace men_wages_eq_13_5_l107_107426

-- Definitions based on problem conditions
def wages (men women boys : ℕ) : ℝ :=
  if 9 * men + women + 7 * boys = 216 then
    men
  else 
    0

def equivalent_wage (men_wage women_wage boy_wage : ℝ) : Prop :=
  9 * men_wage = women_wage ∧
  women_wage = 7 * boy_wage

def total_earning (men_wage women_wage boy_wage : ℝ) : Prop :=
  9 * men_wage + 7 * boy_wage = 216

-- Theorem statement
theorem men_wages_eq_13_5 (M_wage W_wage B_wage : ℝ) :
  equivalent_wage M_wage W_wage B_wage →
  total_earning M_wage W_wage B_wage →
  M_wage = 13.5 :=
by 
  intros h_equiv h_total
  sorry

end men_wages_eq_13_5_l107_107426


namespace probability_of_adjacent_vertices_in_decagon_l107_107114

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l107_107114


namespace expand_product_l107_107636

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 :=
by
  sorry

end expand_product_l107_107636


namespace mechanical_moles_l107_107748

-- Define the conditions
def condition_one (x y : ℝ) : Prop :=
  x + y = 1 / 5

def condition_two (x y : ℝ) : Prop :=
  (1 / (3 * x)) + (2 / (3 * y)) = 10

-- Define the main theorem using the defined conditions
theorem mechanical_moles (x y : ℝ) (h1 : condition_one x y) (h2 : condition_two x y) :
  x = 1 / 30 ∧ y = 1 / 6 :=
  sorry

end mechanical_moles_l107_107748


namespace total_revenue_from_selling_snakes_l107_107699

-- Definitions based on conditions
def num_snakes := 3
def eggs_per_snake := 2
def standard_price := 250
def rare_multiplier := 4

-- Prove the total revenue Jake gets from selling all baby snakes is $2250
theorem total_revenue_from_selling_snakes : 
  (num_snakes * eggs_per_snake - 1) * standard_price + (standard_price * rare_multiplier) = 2250 := 
by
  sorry

end total_revenue_from_selling_snakes_l107_107699


namespace competition_result_l107_107566

variables (Olya Oleg Pasha : ℕ)

theorem competition_result 
  (h1 : Olya ≠ 1 → Olya ≠ 3 → False)
  (h2 : (Oleg = 1 ∨ Oleg = 3) → Olya = 3)
  (h3 : (Oleg ≠ 1 → (Olya = 2 ∨ Olya = 3)))
  (h4 : Olya ≠ 1 ∧ Oleg ≠ 2 ∧ Pasha ≠ 3) :
  Oleg = 1 ∧ Pasha = 2 ∧ Olya = 3 := 
by {
  sorry
}

end competition_result_l107_107566


namespace quadratic_inequality_range_of_k_l107_107507

theorem quadratic_inequality (a b x : ℝ) (h1 : a = 1) (h2 : b > 1) :
  (a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
sorry

theorem range_of_k (x y k : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1/x) + (2/y) = 1) (h4 : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 :=
sorry

end quadratic_inequality_range_of_k_l107_107507


namespace cat_walking_rate_l107_107251

theorem cat_walking_rate :
  let resisting_time := 20 -- minutes
  let total_distance := 64 -- feet
  let total_time := 28 -- minutes
  let walking_time := total_time - resisting_time
  (total_distance / walking_time = 8) :=
by
  let resisting_time := 20
  let total_distance := 64
  let total_time := 28
  let walking_time := total_time - resisting_time
  have : total_distance / walking_time = 8 :=
    by norm_num [total_distance, walking_time]
  exact this

end cat_walking_rate_l107_107251


namespace inscribed_triangle_area_is_12_l107_107454

noncomputable def area_of_triangle_in_inscribed_circle 
  (a b c : ℝ) 
  (h_ratio : a = 2 * b ∧ c = 2 * a) 
  (h_radius : ∀ R, R = 4) 
  (h_inscribed : ∃ x, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x ∧ c = 2 * 4) : 
  ℝ := 
1 / 2 * (2 * (4 / 2)) * (3 * (4 / 2))

theorem inscribed_triangle_area_is_12 
  (a b c : ℝ) 
  (h_ratio : a = 2 * b ∧ c = 2 * a) 
  (h_radius : ∀ R, R = 4) 
  (h_inscribed : ∃ x, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x ∧ c = 2 * 4) :
  area_of_triangle_in_inscribed_circle a b c h_ratio h_radius h_inscribed = 12 :=
sorry

end inscribed_triangle_area_is_12_l107_107454


namespace tradesman_gain_on_outlay_l107_107085

-- Define the percentage defrauded and the percentage gain in both buying and selling
def defraud_percent := 20
def original_value := 100
def buying_price := original_value * (1 - (defraud_percent / 100))
def selling_price := original_value * (1 + (defraud_percent / 100))
def gain := selling_price - buying_price
def gain_percent := (gain / buying_price) * 100

theorem tradesman_gain_on_outlay :
  gain_percent = 50 := 
sorry

end tradesman_gain_on_outlay_l107_107085


namespace basketball_minutes_played_l107_107960

-- Definitions of the conditions in Lean
def football_minutes : ℕ := 60
def total_hours : ℕ := 2
def total_minutes : ℕ := total_hours * 60

-- The statement we need to prove (that basketball_minutes = 60)
theorem basketball_minutes_played : 
  (120 - football_minutes = 60) := by
  sorry

end basketball_minutes_played_l107_107960


namespace min_value_f_l107_107821

noncomputable def f (x : ℝ) : ℝ :=
  7 * (Real.sin x)^2 + 5 * (Real.cos x)^2 + 2 * Real.sin x

theorem min_value_f : ∃ x : ℝ, f x = 4.5 :=
  sorry

end min_value_f_l107_107821


namespace probability_of_adjacent_vertices_in_decagon_l107_107116

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l107_107116


namespace bus_driver_total_hours_l107_107311

theorem bus_driver_total_hours
  (reg_rate : ℝ := 16)
  (ot_rate : ℝ := 28)
  (total_hours : ℝ)
  (total_compensation : ℝ := 920)
  (h : total_compensation = reg_rate * 40 + ot_rate * (total_hours - 40)) :
  total_hours = 50 := 
by 
  sorry

end bus_driver_total_hours_l107_107311


namespace domain_of_k_l107_107796

noncomputable def k (x : ℝ) := (1 / (x + 9)) + (1 / (x^2 + 9)) + (1 / (x^5 + 9)) + (1 / (x - 9))

theorem domain_of_k :
  ∀ x : ℝ, x ≠ -9 ∧ x ≠ -1.551 ∧ x ≠ 9 → ∃ y, y = k x := 
by
  sorry

end domain_of_k_l107_107796


namespace isosceles_triangles_count_isosceles_triangles_l107_107674

theorem isosceles_triangles (x : ℕ) (b : ℕ) : 
  (2 * x + b = 29 ∧ b % 2 = 1) ∧ (b < 14) → 
  (b = 1 ∧ x = 14 ∨ b = 3 ∧ x = 13 ∨ b = 5 ∧ x = 12 ∨ b = 7 ∧ x = 11 ∨ b = 9 ∧ x = 10) :=
by sorry

theorem count_isosceles_triangles : 
  (∃ x b, (2 * x + b = 29 ∧ b % 2 = 1) ∧ (b < 14)) → 
  (5 = 5) :=
by sorry

end isosceles_triangles_count_isosceles_triangles_l107_107674


namespace decagon_adjacent_probability_l107_107112

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l107_107112


namespace Jims_apples_fits_into_average_l107_107901

def Jim_apples : Nat := 20
def Jane_apples : Nat := 60
def Jerry_apples : Nat := 40

def total_apples : Nat := Jim_apples + Jane_apples + Jerry_apples
def number_of_people : Nat := 3
def average_apples_per_person : Nat := total_apples / number_of_people

theorem Jims_apples_fits_into_average :
  average_apples_per_person / Jim_apples = 2 := by
  sorry

end Jims_apples_fits_into_average_l107_107901


namespace alison_birth_weekday_l107_107042

-- Definitions for the problem conditions
def days_in_week : ℕ := 7

-- John's birth day
def john_birth_weekday : ℕ := 3  -- Assuming Monday=0, Tuesday=1, ..., Wednesday=3, ...

-- Number of days Alison was born later
def days_later : ℕ := 72

-- Proof that the resultant day is Friday
theorem alison_birth_weekday : (john_birth_weekday + days_later) % days_in_week = 5 :=
by
  sorry

end alison_birth_weekday_l107_107042


namespace percentage_of_water_in_nectar_l107_107757

-- Define the necessary conditions and variables
def weight_of_nectar : ℝ := 1.7 -- kg
def weight_of_honey : ℝ := 1 -- kg
def honey_water_percentage : ℝ := 0.15 -- 15%

noncomputable def water_in_honey : ℝ := weight_of_honey * honey_water_percentage -- Water content in 1 kg of honey

noncomputable def total_water_in_nectar : ℝ := water_in_honey + (weight_of_nectar - weight_of_honey) -- Total water content in nectar

-- The theorem to prove
theorem percentage_of_water_in_nectar :
    (total_water_in_nectar / weight_of_nectar) * 100 = 50 := 
by 
    -- Skipping the proof by using sorry as it is not required
    sorry

end percentage_of_water_in_nectar_l107_107757


namespace num_possible_triangle_sides_l107_107206

theorem num_possible_triangle_sides (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (a_eq_8 : a = 8) (b_eq_5 : b = 5) : 
  let x := {n : ℕ | 3 < n ∧ n < 13} in
  set.card x = 9 :=
by
  sorry

end num_possible_triangle_sides_l107_107206


namespace correct_propositions_l107_107060

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3) + Real.cos (2 * x + Real.pi / 6)

theorem correct_propositions :
  (∀ x, f x = Real.sqrt 2 * Real.cos (2 * x - Real.pi / 12)) ∧
  (Real.sqrt 2 = f (Real.pi / 24)) ∧
  (f (-1) ≠ f 1) ∧
  (∀ x, Real.pi / 24 ≤ x ∧ x ≤ 13 * Real.pi / 24 -> (f (x + 1e-6) < f x)) ∧
  (∀ x, (Real.sqrt 2 * Real.cos (2 * (x - Real.pi / 24))) = f x)
  := by
    sorry

end correct_propositions_l107_107060


namespace roots_diff_eq_4_l107_107920

theorem roots_diff_eq_4 {r s : ℝ} (h₁ : r ≠ s) (h₂ : r > s) (h₃ : r^2 - 10 * r + 21 = 0) (h₄ : s^2 - 10 * s + 21 = 0) : r - s = 4 := 
by
  sorry

end roots_diff_eq_4_l107_107920


namespace function_decreasing_in_interval_l107_107458

theorem function_decreasing_in_interval :
  ∀ (x1 x2 : ℝ), (0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2) → 
  (x1 - x2) * ((1 / x1 - x1) - (1 / x2 - x2)) < 0 :=
by
  intros x1 x2 hx
  sorry

end function_decreasing_in_interval_l107_107458


namespace possible_integer_side_lengths_l107_107209

theorem possible_integer_side_lengths : 
  {x : ℤ | x > 3 ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l107_107209


namespace mandy_gets_15_pieces_l107_107605

def initial_pieces : ℕ := 75
def michael_takes (pieces : ℕ) : ℕ := pieces / 3
def paige_takes (pieces : ℕ) : ℕ := (pieces - michael_takes pieces) / 2
def ben_takes (pieces : ℕ) : ℕ := 2 * (pieces - michael_takes pieces - paige_takes pieces) / 5
def mandy_takes (pieces : ℕ) : ℕ := pieces - michael_takes pieces - paige_takes pieces - ben_takes pieces

theorem mandy_gets_15_pieces :
  mandy_takes initial_pieces = 15 :=
by
  sorry

end mandy_gets_15_pieces_l107_107605


namespace evaluate_expression_l107_107967

theorem evaluate_expression : 2 + 0 - 2 * 0 = 2 :=
by
  sorry

end evaluate_expression_l107_107967


namespace second_digging_breadth_l107_107139

theorem second_digging_breadth :
  ∀ (A B depth1 length1 breadth1 depth2 length2 : ℕ),
  (A / B) = 1 → -- Assuming equal number of days and people
  depth1 = 100 → length1 = 25 → breadth1 = 30 → 
  depth2 = 75 → length2 = 20 → 
  (A = depth1 * length1 * breadth1) → 
  (B = depth2 * length2 * x) →
  x = 50 :=
by sorry

end second_digging_breadth_l107_107139


namespace dave_apps_files_difference_l107_107339

theorem dave_apps_files_difference :
  let initial_apps := 15
  let initial_files := 24
  let final_apps := 21
  let final_files := 4
  final_apps - final_files = 17 :=
by
  intros
  sorry

end dave_apps_files_difference_l107_107339


namespace probability_one_even_dice_l107_107998

noncomputable def probability_exactly_one_even (p : ℚ) : Prop :=
  ∃ (n : ℕ), (p = (4 * (1/2)^4 )) ∧ (n = 1) → p = 1/4

theorem probability_one_even_dice : probability_exactly_one_even (1/4) :=
by
  unfold probability_exactly_one_even
  sorry

end probability_one_even_dice_l107_107998


namespace initially_calculated_average_weight_l107_107580

theorem initially_calculated_average_weight (n : ℕ) (misread_diff correct_avg_weight : ℝ)
  (hn : n = 20) (hmisread_diff : misread_diff = 10) (hcorrect_avg_weight : correct_avg_weight = 58.9) :
  ((correct_avg_weight * n - misread_diff) / n) = 58.4 :=
by
  rw [hn, hmisread_diff, hcorrect_avg_weight]
  sorry

end initially_calculated_average_weight_l107_107580


namespace probability_of_adjacent_vertices_in_decagon_l107_107117

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l107_107117


namespace math_problem_l107_107023

theorem math_problem (d r : ℕ) (hd : d > 1)
  (h1 : 1259 % d = r) 
  (h2 : 1567 % d = r) 
  (h3 : 2257 % d = r) : d - r = 1 :=
by
  sorry

end math_problem_l107_107023


namespace no_real_intersections_l107_107846

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l107_107846


namespace simplify_expression_l107_107758

theorem simplify_expression :
  ( (2^2 - 1) * (3^2 - 1) * (4^2 - 1) * (5^2 - 1) ) / ( (2 * 3) * (3 * 4) * (4 * 5) * (5 * 6) ) = 1 / 5 :=
by
  sorry

end simplify_expression_l107_107758


namespace angles_around_point_sum_l107_107911

theorem angles_around_point_sum 
  (x y : ℝ)
  (h1 : 130 + x + y = 360)
  (h2 : y = x + 30) :
  x = 100 ∧ y = 130 :=
by
  sorry

end angles_around_point_sum_l107_107911


namespace ball_hits_ground_time_l107_107604

theorem ball_hits_ground_time :
  ∃ t : ℝ, -16 * t^2 - 30 * t + 180 = 0 ∧ t = 1.25 := by
  sorry

end ball_hits_ground_time_l107_107604


namespace decagon_adjacent_probability_l107_107104

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l107_107104


namespace box_tape_length_l107_107984

variable (L S : ℕ)
variable (tape_total : ℕ)
variable (num_boxes : ℕ)
variable (square_side : ℕ)

theorem box_tape_length (h1 : num_boxes = 5) (h2 : square_side = 40) (h3 : tape_total = 540) :
  tape_total = 5 * (L + 2 * S) + 2 * 3 * square_side → L = 60 - 2 * S := 
by
  sorry

end box_tape_length_l107_107984


namespace min_value_frac_sum_l107_107835

open Real

theorem min_value_frac_sum (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 2 * a + b = 1) :
  1 / a + 2 / b = 8 :=
sorry

end min_value_frac_sum_l107_107835


namespace possible_integer_side_lengths_l107_107218

theorem possible_integer_side_lengths (a b : ℕ) (h1 : a = 8) (h2 : b = 5) :
  {x : ℕ | 3 < x ∧ x < 13}.finite ∧ {x : ℕ | 3 < x ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l107_107218


namespace roger_earned_correct_amount_l107_107061

def small_lawn_rate : ℕ := 9
def medium_lawn_rate : ℕ := 12
def large_lawn_rate : ℕ := 15

def initial_small_lawns : ℕ := 5
def initial_medium_lawns : ℕ := 4
def initial_large_lawns : ℕ := 5

def forgot_small_lawns : ℕ := 2
def forgot_medium_lawns : ℕ := 3
def forgot_large_lawns : ℕ := 3

def actual_small_lawns := initial_small_lawns - forgot_small_lawns
def actual_medium_lawns := initial_medium_lawns - forgot_medium_lawns
def actual_large_lawns := initial_large_lawns - forgot_large_lawns

def money_earned_small := actual_small_lawns * small_lawn_rate
def money_earned_medium := actual_medium_lawns * medium_lawn_rate
def money_earned_large := actual_large_lawns * large_lawn_rate

def total_money_earned := money_earned_small + money_earned_medium + money_earned_large

theorem roger_earned_correct_amount : total_money_earned = 69 := by
  sorry

end roger_earned_correct_amount_l107_107061


namespace number_of_masters_students_l107_107342

theorem number_of_masters_students (total_sample : ℕ) (ratio_assoc : ℕ) (ratio_undergrad : ℕ) (ratio_masters : ℕ) (ratio_doctoral : ℕ) 
(h1 : ratio_assoc = 5) (h2 : ratio_undergrad = 15) (h3 : ratio_masters = 9) (h4 : ratio_doctoral = 1) (h_total_sample : total_sample = 120) :
  (ratio_masters * total_sample) / (ratio_assoc + ratio_undergrad + ratio_masters + ratio_doctoral) = 36 :=
by
  sorry

end number_of_masters_students_l107_107342


namespace congruence_from_overlap_l107_107299

-- Definitions used in the conditions
def figure := Type
def equal_area (f1 f2 : figure) : Prop := sorry
def equal_perimeter (f1 f2 : figure) : Prop := sorry
def equilateral_triangle (f : figure) : Prop := sorry
def can_completely_overlap (f1 f2 : figure) : Prop := sorry

-- Theorem that should be proven
theorem congruence_from_overlap (f1 f2 : figure) (h: can_completely_overlap f1 f2) : f1 = f2 := sorry

end congruence_from_overlap_l107_107299


namespace probability_range_inequality_l107_107691

theorem probability_range_inequality :
  ∀ p : ℝ, 0 ≤ p → p ≤ 1 →
  (4 * p * (1 - p)^3 ≤ 6 * p^2 * (1 - p)^2 → 0.4 ≤ p ∧ p < 1) := sorry

end probability_range_inequality_l107_107691


namespace comprehensive_survey_is_C_l107_107759

def option (label : String) (description : String) := (label, description)

def A := option "A" "Investigating the current mental health status of middle school students nationwide"
def B := option "B" "Investigating the compliance of food in our city"
def C := option "C" "Investigating the physical and mental conditions of classmates in the class"
def D := option "D" "Investigating the viewership ratings of Nanjing TV's 'Today's Life'"

theorem comprehensive_survey_is_C (suitable: (String × String → Prop)) :
  suitable C :=
sorry

end comprehensive_survey_is_C_l107_107759


namespace geometric_quadratic_root_l107_107948

theorem geometric_quadratic_root (a b c : ℝ) (h1 : a > 0) (h2 : b = a * (1 / 4)) (h3 : c = a * (1 / 16)) (h4 : a * a * (1 / 4)^2 = 4 * a * a * (1 / 16)) : 
    -b / (2 * a) = -1 / 8 :=
by 
    sorry

end geometric_quadratic_root_l107_107948


namespace abs_inequality_solution_l107_107996

theorem abs_inequality_solution (x : ℝ) : 
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ (1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5) := 
sorry

end abs_inequality_solution_l107_107996


namespace unique_ordered_triple_l107_107480

theorem unique_ordered_triple : 
  ∃! (x y z : ℝ), x + y = 4 ∧ xy - z^2 = 4 ∧ x = 2 ∧ y = 2 ∧ z = 0 :=
by
  sorry

end unique_ordered_triple_l107_107480


namespace jane_buys_4_bagels_l107_107250

theorem jane_buys_4_bagels (b m : ℕ) (h1 : b + m = 7) (h2 : (80 * b + 60 * m) % 100 = 0) : b = 4 := 
by sorry

end jane_buys_4_bagels_l107_107250


namespace line_circle_no_intersection_l107_107891

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l107_107891


namespace saltwater_concentration_l107_107364

theorem saltwater_concentration (salt_mass water_mass : ℝ) (h₁ : salt_mass = 8) (h₂ : water_mass = 32) : 
  salt_mass / (salt_mass + water_mass) * 100 = 20 := 
by
  sorry

end saltwater_concentration_l107_107364


namespace first_dig_site_date_difference_l107_107612

-- Definitions for the conditions
def F : Int := sorry  -- The age of the first dig site
def S : Int := sorry  -- The age of the second dig site
def T : Int := sorry  -- The age of the third dig site
def Fo : Int := 8400  -- The age of the fourth dig site
def x : Int := (S - F)

-- The conditions
axiom condition1 : F = S + x
axiom condition2 : T = F + 3700
axiom condition3 : Fo = 2 * T
axiom condition4 : S = 852
axiom condition5 : S > F  -- Ensuring S is older than F for meaningfulness

-- The theorem to prove
theorem first_dig_site_date_difference : x = 352 :=
by
  -- Proof goes here
  sorry

end first_dig_site_date_difference_l107_107612


namespace line_circle_no_intersection_l107_107862

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l107_107862


namespace average_weight_of_abc_l107_107732

theorem average_weight_of_abc (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 43) (h3 : B = 40) : 
  (A + B + C) / 3 = 42 := 
sorry

end average_weight_of_abc_l107_107732


namespace find_rosy_age_l107_107169

-- Definitions and conditions
def rosy_current_age (R : ℕ) : Prop :=
  ∃ D : ℕ,
    (D = R + 18) ∧ -- David is 18 years older than Rosy
    (D + 6 = 2 * (R + 6)) -- In 6 years, David will be twice as old as Rosy

-- Proof statement: Rosy's current age is 12
theorem find_rosy_age : rosy_current_age 12 :=
  sorry

end find_rosy_age_l107_107169


namespace other_x_intercept_of_parabola_l107_107348

theorem other_x_intercept_of_parabola (a b c : ℝ) :
  (∃ x : ℝ, y = a * x ^ 2 + b * x + c) ∧ (2, 10) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)} ∧ (1, 0) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)}
  → ∃ x : ℝ, x = 3 ∧ (x, 0) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)} :=
by
  sorry

end other_x_intercept_of_parabola_l107_107348


namespace neil_baked_cookies_l107_107925

theorem neil_baked_cookies (total_cookies : ℕ) (given_to_friend : ℕ) (cookies_left : ℕ)
    (h1 : given_to_friend = (2 / 5) * total_cookies)
    (h2 : cookies_left = (3 / 5) * total_cookies)
    (h3 : cookies_left = 12) : total_cookies = 20 :=
by
  sorry

end neil_baked_cookies_l107_107925


namespace number_of_cow_herds_l107_107157

theorem number_of_cow_herds 
    (total_cows : ℕ) 
    (cows_per_herd : ℕ) 
    (h1 : total_cows = 320)
    (h2 : cows_per_herd = 40) : 
    total_cows / cows_per_herd = 8 :=
by
  sorry

end number_of_cow_herds_l107_107157


namespace max_planes_15_points_l107_107440

theorem max_planes_15_points (P : Finset (Fin 15)) (hP : ∀ (p1 p2 p3 : Fin 15), p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3) :
  P.card = 15 → (∃ planes : Finset (Finset (Fin 15)), planes.card = 455) := by
  sorry

end max_planes_15_points_l107_107440


namespace rectangle_square_overlap_l107_107932

theorem rectangle_square_overlap (ABCD EFGH : Type) (s x y : ℝ)
  (h1 : 0.3 * s^2 = 0.6 * x * y)
  (h2 : AB = 2 * s)
  (h3 : AD = y)
  (h4 : x * y = 0.5 * s^2) :
  x / y = 8 :=
sorry

end rectangle_square_overlap_l107_107932


namespace parametric_curve_to_general_form_l107_107431

theorem parametric_curve_to_general_form :
  ∃ (a b c : ℚ), ∀ (t : ℝ), 
  (a = 8 / 225) ∧ (b = 4 / 75) ∧ (c = 1 / 25) ∧ 
  (a * (3 * Real.sin t)^2 + b * (3 * Real.sin t) * (5 * Real.cos t - 2 * Real.sin t) + c * (5 * Real.cos t - 2 * Real.sin t)^2 = 1) :=
by
  use 8 / 225, 4 / 75, 1 / 25
  sorry

end parametric_curve_to_general_form_l107_107431


namespace proof_problem_l107_107358

-- Defining the statement in Lean 4.

noncomputable def p : Prop :=
  ∀ x : ℝ, x > Real.sin x

noncomputable def neg_p : Prop :=
  ∃ x : ℝ, x ≤ Real.sin x

theorem proof_problem : ¬p ↔ neg_p := 
by sorry

end proof_problem_l107_107358


namespace volume_of_cube_is_correct_surface_area_of_cube_is_correct_l107_107135

-- Define the conditions: total edge length of the cube frame
def total_edge_length : ℕ := 60
def number_of_edges : ℕ := 12

-- Define the edge length of the cube
def edge_length (total_edge_length number_of_edges : ℕ) : ℕ := total_edge_length / number_of_edges

-- Define the volume of the cube
def cube_volume (a : ℕ) : ℕ := a ^ 3

-- Define the surface area of the cube
def cube_surface_area (a : ℕ) : ℕ := 6 * (a ^ 2)

-- Volume Proof Statement
theorem volume_of_cube_is_correct : cube_volume (edge_length total_edge_length number_of_edges) = 125 :=
by
  sorry

-- Surface Area Proof Statement
theorem surface_area_of_cube_is_correct : cube_surface_area (edge_length total_edge_length number_of_edges) = 150 :=
by
  sorry

end volume_of_cube_is_correct_surface_area_of_cube_is_correct_l107_107135


namespace line_circle_no_intersection_l107_107892

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l107_107892


namespace avg_cost_apple_tv_200_l107_107613

noncomputable def average_cost_apple_tv (iphones_sold ipads_sold apple_tvs_sold iphone_cost ipad_cost overall_avg_cost: ℝ) : ℝ :=
  (overall_avg_cost * (iphones_sold + ipads_sold + apple_tvs_sold) - (iphones_sold * iphone_cost + ipads_sold * ipad_cost)) / apple_tvs_sold

theorem avg_cost_apple_tv_200 :
  let iphones_sold := 100
  let ipads_sold := 20
  let apple_tvs_sold := 80
  let iphone_cost := 1000
  let ipad_cost := 900
  let overall_avg_cost := 670
  average_cost_apple_tv iphones_sold ipads_sold apple_tvs_sold iphone_cost ipad_cost overall_avg_cost = 200 :=
by
  sorry

end avg_cost_apple_tv_200_l107_107613


namespace charlie_book_pages_l107_107792

theorem charlie_book_pages :
  (2 * 40) + (4 * 45) + 20 = 280 :=
by 
  sorry

end charlie_book_pages_l107_107792


namespace find_a_l107_107185

theorem find_a (f : ℝ → ℝ) (a x : ℝ) 
  (h1 : ∀ x, f (1/2 * x - 1) = 2 * x - 5) 
  (h2 : f a = 6) : a = 7 / 4 := 
by
  sorry

end find_a_l107_107185


namespace cats_awake_l107_107953

theorem cats_awake (total_cats asleep_cats cats_awake : ℕ) (h1 : total_cats = 98) (h2 : asleep_cats = 92) (h3 : cats_awake = total_cats - asleep_cats) : cats_awake = 6 :=
by
  -- Definitions and conditions
  subst h1
  subst h2
  subst h3
  -- The statement we need to prove
  sorry

end cats_awake_l107_107953


namespace geom_sequence_general_formula_lambda_range_l107_107188

variable {a : ℕ → ℤ}

-- Let 'a' be a sequence satisfying the given initial conditions.
def seq_conditions (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n : ℕ, 0 < n → a (n + 1) - 2 * a n = 2 * n

-- Statement 1: Prove that the given sequence adjusted by 2n + 2 forms a geometric sequence.
theorem geom_sequence {a : ℕ → ℤ} (h : seq_conditions a) :
  ∃ r (c : ℕ → ℤ), (r = 2 ∧ c 1 = 4) ∧ (∀ n : ℕ, 0 < n → a (n + 1) + 2 * (n + 1) + 2 = r * (a n + 2 * n + 2)) :=
sorry

-- Statement 2: Find the general formula for the sequence.
theorem general_formula {a : ℕ → ℤ} (h : seq_conditions a) :
  ∀ n : ℕ, 0 < n → a n = 2 ^ (n + 1) - 2 * n - 2 :=
sorry

-- Statement 3: Given the inequality condition involving lambda, find its range.
theorem lambda_range {a : ℕ → ℤ} (h : seq_conditions a) (λ : ℝ) :
  (∀ n : ℕ, 0 < n → a n > λ * (2 * n + 1) * (-1) ^ (n - 1)) ↔ ( -2 / 5 < λ ∧ λ < 0) :=
sorry

end geom_sequence_general_formula_lambda_range_l107_107188


namespace truck_dirt_road_time_l107_107321

noncomputable def time_on_dirt_road (time_paved : ℝ) (speed_increment : ℝ) (total_distance : ℝ) (dirt_speed : ℝ) : ℝ :=
  let paved_speed := dirt_speed + speed_increment
  let distance_paved := paved_speed * time_paved
  let distance_dirt := total_distance - distance_paved
  distance_dirt / dirt_speed

theorem truck_dirt_road_time :
  time_on_dirt_road 2 20 200 32 = 3 :=
by
  sorry

end truck_dirt_road_time_l107_107321


namespace sequence_geometric_and_lambda_range_l107_107187

theorem sequence_geometric_and_lambda_range:
  (∃ (a : ℕ → ℤ), a 1 = 0 ∧ (∀ n : ℕ, n ≥ 1 → a (n+1) - 2 * a n = 2 * n)) →
  (∃ n : ℕ, n ≥ 1 → a n = 2^(n+1) - 2*n - 2) ∧ 
  (∀ λ : ℚ, (∀ n : ℕ, n ≥ 1 → a n > λ * (2*n + 1) * (-1)^(n-1)) →
    -2/5 < λ ∧ λ < 0) :=
by
  sorry

end sequence_geometric_and_lambda_range_l107_107187


namespace sqrt_four_l107_107290

theorem sqrt_four : {x : ℝ | x ^ 2 = 4} = {-2, 2} := by
  sorry

end sqrt_four_l107_107290


namespace decagon_adjacent_probability_l107_107103

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l107_107103


namespace find_constants_l107_107665

theorem find_constants (a b c : ℝ) (h_neq_0_a : a ≠ 0) (h_neq_0_b : b ≠ 0) 
(h_neq_0_c : c ≠ 0) 
(h_eq1 : a * b = 3 * (a + b)) 
(h_eq2 : b * c = 4 * (b + c)) 
(h_eq3 : a * c = 5 * (a + c)) : 
a = 120 / 17 ∧ b = 120 / 23 ∧ c = 120 / 7 := 
  sorry

end find_constants_l107_107665


namespace probability_two_approvals_of_four_l107_107979

/-- Definition of the problem condition -/
def probability_approval (P_Y : ℚ) (P_N : ℚ) : Prop :=
  P_Y = 0.6 ∧ P_N = 1 - P_Y

/-- The main theorem stating the probability of exactly two approvals out of four trials. -/
theorem probability_two_approvals_of_four (P_Y P_N : ℚ)
  (h : probability_approval P_Y P_N) :
  (4.choose 2) * P_Y^2 * P_N^2 = 0.3456 := by
  sorry

end probability_two_approvals_of_four_l107_107979


namespace sum_of_three_pairwise_relatively_prime_integers_l107_107407

theorem sum_of_three_pairwise_relatively_prime_integers
  (a b c : ℕ)
  (h1 : a > 1)
  (h2 : b > 1)
  (h3 : c > 1)
  (h4 : a * b * c = 13824)
  (h5 : Nat.gcd a b = 1)
  (h6 : Nat.gcd b c = 1)
  (h7 : Nat.gcd a c = 1) :
  a + b + c = 144 :=
by
  sorry

end sum_of_three_pairwise_relatively_prime_integers_l107_107407


namespace weekly_earnings_l107_107154

def shop_opening_hours_per_day := 720 -- in minutes
def women_tshirt_selling_interval := 30 -- in minutes
def price_per_women_tshirt := 18 -- in dollars
def men_tshirt_selling_interval := 40 -- in minutes
def price_per_men_tshirt := 15 -- in dollars
def days_per_week := 7 -- days

theorem weekly_earnings :
  let women_tshirts_sold_per_day := shop_opening_hours_per_day / women_tshirt_selling_interval in
  let daily_earnings_women_tshirts := women_tshirts_sold_per_day * price_per_women_tshirt in
  let men_tshirts_sold_per_day := shop_opening_hours_per_day / men_tshirt_selling_interval in
  let daily_earnings_men_tshirts := men_tshirts_sold_per_day * price_per_men_tshirt in
  let total_daily_earnings := daily_earnings_women_tshirts + daily_earnings_men_tshirts in
  let weekly_earnings := total_daily_earnings * days_per_week in
  weekly_earnings = 4914 := 
by {
  -- Proof steps go here
  sorry
}

end weekly_earnings_l107_107154


namespace fourth_square_area_l107_107955

theorem fourth_square_area (AB BC CD AD AC : ℝ) (h1 : AB^2 = 25) (h2 : BC^2 = 49) (h3 : CD^2 = 64) (h4 : AC^2 = AB^2 + BC^2)
  (h5 : AD^2 = AC^2 + CD^2) : AD^2 = 138 :=
by
  sorry

end fourth_square_area_l107_107955


namespace no_fixed_points_l107_107655

def f (x a : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem no_fixed_points (a : ℝ) :
  (∀ x : ℝ, f x a ≠ x) ↔ (-1/2 < a ∧ a < 3/2) := by
    sorry

end no_fixed_points_l107_107655


namespace transform_binomial_expansion_l107_107696

variable (a b : ℝ)

theorem transform_binomial_expansion (h : (a + b)^4 = a^4 + 4 * a^3 * b + 6 * a^2 * b^2 + 4 * a * b^3 + b^4) :
  (a - b)^4 = a^4 - 4 * a^3 * b + 6 * a^2 * b^2 - 4 * a * b^3 + b^4 :=
by
  sorry

end transform_binomial_expansion_l107_107696


namespace lesser_number_is_21_5_l107_107292

theorem lesser_number_is_21_5
  (x y : ℝ)
  (h1 : x + y = 50)
  (h2 : x - y = 7) :
  y = 21.5 :=
by
  sorry

end lesser_number_is_21_5_l107_107292


namespace ducks_in_the_marsh_l107_107954

-- Define the conditions
def number_of_geese : ℕ := 58
def total_number_of_birds : ℕ := 95
def number_of_ducks : ℕ := total_number_of_birds - number_of_geese

-- Prove the conclusion
theorem ducks_in_the_marsh : number_of_ducks = 37 := by
  -- subtraction to find number_of_ducks
  sorry

end ducks_in_the_marsh_l107_107954


namespace sin_x_eq_x_has_unique_root_in_interval_l107_107945

theorem sin_x_eq_x_has_unique_root_in_interval :
  ∃! x : ℝ, x ∈ Set.Icc (-Real.pi) Real.pi ∧ x = Real.sin x :=
sorry

end sin_x_eq_x_has_unique_root_in_interval_l107_107945


namespace min_value_of_function_l107_107346

open Real

theorem min_value_of_function (x y : ℝ) (h : 2 * x + 8 * y = 3) : ∃ (min_value : ℝ), min_value = -19 / 20 ∧ ∀ (x y : ℝ), 2 * x + 8 * y = 3 → x^2 + 4 * y^2 - 2 * x ≥ -19 / 20 :=
by
  sorry

end min_value_of_function_l107_107346


namespace no_real_intersections_l107_107847

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l107_107847


namespace min_value_1_a_plus_2_b_l107_107500

open Real

theorem min_value_1_a_plus_2_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (∀ a b, 0 < a → 0 < b → a + b = 1 → 3 + 2 * sqrt 2 ≤ 1 / a + 2 / b) := sorry

end min_value_1_a_plus_2_b_l107_107500


namespace line_circle_no_intersect_l107_107857

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l107_107857


namespace union_sets_l107_107196

def A : Set ℝ := { x | (2 / x) > 1 }
def B : Set ℝ := { x | Real.log x < 0 }

theorem union_sets : (A ∪ B) = { x : ℝ | 0 < x ∧ x < 2 } := by
  sorry

end union_sets_l107_107196


namespace count_perfect_square_factors_l107_107675

/-
We want to prove the following statement: 
The number of elements from 1 to 100 that have a perfect square factor other than 1 is 40.
-/

theorem count_perfect_square_factors:
  (∃ (S : Finset ℕ), S = (Finset.range 100).filter (λ n, ∃ m, m * m ∣ n ∧ m ≠ 1) ∧ S.card = 40) :=
sorry

end count_perfect_square_factors_l107_107675


namespace greatest_perfect_square_below_200_l107_107718

theorem greatest_perfect_square_below_200 : 
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (∃ k : ℕ, m = k^2) → m ≤ n) ∧ (∃ k : ℕ, n = k^2) := 
by 
  use 196, 14
  split
  -- 196 is less than 200, and 196 is a perfect square
  {
    exact 196 < 200,
    use 14,
    exact 196 = 14^2,
  },
  -- no perfect square less than 200 is greater than 196
  sorry

end greatest_perfect_square_below_200_l107_107718


namespace sum_of_remainders_is_six_l107_107277

def sum_of_remainders (n : ℕ) : ℕ :=
  n % 4 + (n + 1) % 4 + (n + 2) % 4 + (n + 3) % 4

theorem sum_of_remainders_is_six : ∀ n : ℕ, sum_of_remainders n = 6 :=
by
  intro n
  sorry

end sum_of_remainders_is_six_l107_107277


namespace geometric_progression_common_ratio_l107_107908

theorem geometric_progression_common_ratio (r : ℝ) :
  (r > 0) ∧ (r^3 + r^2 + r - 1 = 0) ↔
  r = ( -1 + ((19 + 3 * Real.sqrt 33)^(1/3)) + ((19 - 3 * Real.sqrt 33)^(1/3)) ) / 3 :=
by
  sorry

end geometric_progression_common_ratio_l107_107908


namespace smallest_even_number_l107_107951

theorem smallest_even_number (n1 n2 n3 n4 n5 n6 n7 : ℤ) 
  (h_sum_seven : n1 + n2 + n3 + n4 + n5 + n6 + n7 = 700)
  (h_sum_first_three : n1 + n2 + n3 > 200)
  (h_consecutive : n2 = n1 + 2 ∧ n3 = n2 + 2 ∧ n4 = n3 + 2 ∧ n5 = n4 + 2 ∧ n6 = n5 + 2 ∧ n7 = n6 + 2) :
  n1 = 94 := 
sorry

end smallest_even_number_l107_107951


namespace mean_of_xyz_l107_107579

theorem mean_of_xyz (x y z : ℝ) (seven_mean : ℝ)
  (h1 : seven_mean = 45)
  (h2 : (7 * seven_mean + x + y + z) / 10 = 58) :
  (x + y + z) / 3 = 265 / 3 :=
by
  sorry

end mean_of_xyz_l107_107579


namespace shopkeeper_packets_l107_107322

noncomputable def milk_packets (oz_to_ml: ℝ) (ml_per_packet: ℝ) (total_milk_oz: ℝ) : ℝ :=
  (total_milk_oz * oz_to_ml) / ml_per_packet

theorem shopkeeper_packets (oz_to_ml: ℝ) (ml_per_packet: ℝ) (total_milk_oz: ℝ) :
  oz_to_ml = 30 → ml_per_packet = 250 → total_milk_oz = 1250 → milk_packets oz_to_ml ml_per_packet total_milk_oz = 150 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end shopkeeper_packets_l107_107322


namespace product_pattern_l107_107270

theorem product_pattern (m n : ℝ) : 
  m * n = ( ( m + n ) / 2 ) ^ 2 - ( ( m - n ) / 2 ) ^ 2 := 
by 
  sorry

end product_pattern_l107_107270


namespace scientific_notation_350_million_l107_107487

theorem scientific_notation_350_million : 350000000 = 3.5 * 10^8 := 
  sorry

end scientific_notation_350_million_l107_107487


namespace perpendicular_lines_a_l107_107672

theorem perpendicular_lines_a {a : ℝ} :
  ((∀ x y : ℝ, (2 * a - 1) * x + a * y + a = 0) → (∀ x y : ℝ, a * x - y + 2 * a = 0) → a = 0 ∨ a = 1) :=
by
  intro h₁ h₂
  sorry

end perpendicular_lines_a_l107_107672


namespace connie_total_markers_l107_107165

def red_markers : ℕ := 5420
def blue_markers : ℕ := 3875
def green_markers : ℕ := 2910
def yellow_markers : ℕ := 6740

def total_markers : ℕ := red_markers + blue_markers + green_markers + yellow_markers

theorem connie_total_markers : total_markers = 18945 := by
  sorry

end connie_total_markers_l107_107165


namespace marbles_solid_color_non_yellow_l107_107966

theorem marbles_solid_color_non_yellow (total_marble solid_colored solid_yellow : ℝ)
    (h1: solid_colored = 0.90 * total_marble)
    (h2: solid_yellow = 0.05 * total_marble) :
    (solid_colored - solid_yellow) / total_marble = 0.85 := by
  -- sorry is used to skip the proof
  sorry

end marbles_solid_color_non_yellow_l107_107966


namespace max_value_of_x_and_y_l107_107683

theorem max_value_of_x_and_y (x y : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : (x - 4) * (x - 10) = 2 ^ y) : x + y ≤ 16 :=
sorry

end max_value_of_x_and_y_l107_107683


namespace units_digit_7_pow_5_pow_3_l107_107158

theorem units_digit_7_pow_5_pow_3 : (7 ^ (5 ^ 3)) % 10 = 7 := by
  sorry

end units_digit_7_pow_5_pow_3_l107_107158


namespace number_of_cars_repaired_l107_107793

theorem number_of_cars_repaired
  (oil_change_cost repair_cost car_wash_cost : ℕ)
  (oil_changes repairs car_washes total_earnings : ℕ)
  (h₁ : oil_change_cost = 20)
  (h₂ : repair_cost = 30)
  (h₃ : car_wash_cost = 5)
  (h₄ : oil_changes = 5)
  (h₅ : car_washes = 15)
  (h₆ : total_earnings = 475)
  (h₇ : 5 * oil_change_cost + 15 * car_wash_cost + repairs * repair_cost = total_earnings) :
  repairs = 10 :=
by sorry

end number_of_cars_repaired_l107_107793


namespace nonoverlapping_unit_squares_in_figure_100_l107_107973

theorem nonoverlapping_unit_squares_in_figure_100 :
  ∃ f : ℕ → ℕ, (f 0 = 3 ∧ f 1 = 7 ∧ f 2 = 15 ∧ f 3 = 27) ∧ f 100 = 20203 :=
by
  sorry

end nonoverlapping_unit_squares_in_figure_100_l107_107973


namespace equilateral_triangle_l107_107238

theorem equilateral_triangle {a b c : ℝ} (h1 : a + b - c = 2) (h2 : 2 * a * b - c^2 = 4) : a = b ∧ b = c :=
by {
  sorry
}

end equilateral_triangle_l107_107238


namespace parabola_min_value_sum_abc_zero_l107_107737

theorem parabola_min_value_sum_abc_zero
  (a b c : ℝ)
  (h1 : ∀ x y : ℝ, y = ax^2 + bx + c → y = 0 ∧ ((x = 1) ∨ (x = 5)))
  (h2 : ∃ x : ℝ, (∀ t : ℝ, ax^2 + bx + c ≤ t^2) ∧ (ax^2 + bx + c = 36)) :
  a + b + c = 0 :=
sorry

end parabola_min_value_sum_abc_zero_l107_107737


namespace alicia_stickers_l107_107414

theorem alicia_stickers :
  ∃ S : ℕ, S > 2 ∧
  (S % 5 = 2) ∧ (S % 11 = 2) ∧ (S % 13 = 2) ∧
  S = 717 :=
sorry

end alicia_stickers_l107_107414


namespace zilla_savings_deposit_l107_107418

-- Definitions based on problem conditions
def monthly_earnings (E : ℝ) : Prop :=
  0.07 * E = 133

def tax_deduction (E : ℝ) : ℝ :=
  E - 0.10 * E

def expenditure (earnings : ℝ) : ℝ :=
  133 +  0.30 * earnings + 0.20 * earnings + 0.12 * earnings

def savings_deposit (remaining_earnings : ℝ) : ℝ :=
  0.15 * remaining_earnings

-- The final proof statement
theorem zilla_savings_deposit (E : ℝ) (total_spent : ℝ) (earnings_after_tax : ℝ) (remaining_earnings : ℝ) : 
  monthly_earnings E →
  tax_deduction E = earnings_after_tax →
  expenditure earnings_after_tax = total_spent →
  remaining_earnings = earnings_after_tax - total_spent →
  savings_deposit remaining_earnings = 77.52 :=
by
  intros
  sorry

end zilla_savings_deposit_l107_107418


namespace consecutive_negative_integers_sum_l107_107397

theorem consecutive_negative_integers_sum (n : ℤ) (hn : n < 0) (hn1 : n + 1 < 0) (hprod : n * (n + 1) = 2550) : n + (n + 1) = -101 :=
by
  sorry

end consecutive_negative_integers_sum_l107_107397


namespace square_area_l107_107452

-- Define the radius of the circles
def circle_radius : ℝ := 3

-- Define the side length of the square based on the arrangement of circles
def square_side_length : ℝ := 2 * (2 * circle_radius)

-- State the theorem to prove the area of the square
theorem square_area : (square_side_length * square_side_length) = 144 :=
by
  sorry

end square_area_l107_107452


namespace line_circle_no_intersection_l107_107868

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l107_107868


namespace price_difference_l107_107470

theorem price_difference (P F : ℝ) (h1 : 0.85 * P = 78.2) (h2 : F = 78.2 * 1.25) : F - P = 5.75 :=
by
  sorry

end price_difference_l107_107470


namespace max_value_y2_minus_x2_plus_x_plus_5_l107_107368

theorem max_value_y2_minus_x2_plus_x_plus_5 (x y : ℝ) (h : y^2 + x - 2 = 0) : 
  ∃ M, M = 7 ∧ ∀ u v, v^2 + u - 2 = 0 → y^2 - x^2 + x + 5 ≤ M :=
by
  sorry

end max_value_y2_minus_x2_plus_x_plus_5_l107_107368


namespace determine_positions_l107_107569

-- Defining the participants
inductive Participant
| Olya
| Oleg
| Pasha

open Participant

-- Defining the possible places
inductive Place
| First
| Second
| Third

open Place

-- Define the conditions
def condition1 (pos : Participant → Place) : Prop := 
  pos Olya = First ∨ pos Oleg = First ∨ pos Pasha = First

def condition2 (pos : Participant → Place) : Prop :=
  (pos Olya = First ∧ pos Olya = Second ∧ pos Olya = Third) ∨
  (pos Oleg = First ∧ pos Oleg = Second ∧ pos Oleg = Third) ∨
  (pos Pasha = First ∧ pos Pasha = Second ∧ pos Pasha = Third)

def condition3 (pos : Participant → Place) : Prop :=
  ∀ p, pos p ≠ First ∧ pos p ≠ Second ∧ pos p ≠ Third

def condition4 (pos : Participant → Place) : Prop :=
  (pos Olya = First → (pos Oleg = First ∨ pos Pasha = First)) ∧
  (pos Oleg = First → pos Olya ≠ First) ∧
  (pos Pasha = First → (pos Oleg = First ∨ pos Olya = First))

def always_true_or_false : Prop :=
  (∀ p, p = Olya ∨ p = Oleg ∨ p = Pasha )

-- Main theorem
theorem determine_positions (pos : Participant → Place) :
  condition1 pos ∧ condition2 pos ∧ condition3 pos ∧ condition4 pos ∧ always_true_or_false →
  pos Oleg = First ∧ pos Pasha = Second ∧ pos Olya = Third := 
by
  sorry

end determine_positions_l107_107569


namespace Berry_read_pages_thursday_l107_107786

theorem Berry_read_pages_thursday :
  ∀ (pages_per_day : ℕ) (pages_sunday : ℕ) (pages_monday : ℕ) (pages_tuesday : ℕ) 
    (pages_wednesday : ℕ) (pages_friday : ℕ) (pages_saturday : ℕ),
    (pages_per_day = 50) →
    (pages_sunday = 43) →
    (pages_monday = 65) →
    (pages_tuesday = 28) →
    (pages_wednesday = 0) →
    (pages_friday = 56) →
    (pages_saturday = 88) →
    pages_sunday + pages_monday + pages_tuesday +
    pages_wednesday + pages_friday + pages_saturday + x = 350 →
    x = 70 := by
  sorry

end Berry_read_pages_thursday_l107_107786


namespace no_real_intersections_l107_107845

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l107_107845


namespace greatest_odd_factors_below_200_l107_107715

theorem greatest_odd_factors_below_200 : ∃ n : ℕ, (n < 200) ∧ (n = 196) ∧ (∃ k : ℕ, n = k^2) ∧ ∀ m : ℕ, (m < 200) ∧ (∃ j : ℕ, m = j^2) → m ≤ n := by
  sorry

end greatest_odd_factors_below_200_l107_107715


namespace probability_adjacent_vertices_decagon_l107_107120

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l107_107120


namespace simplify_expression_of_triangle_side_lengths_l107_107009

theorem simplify_expression_of_triangle_side_lengths
  (a b c : ℝ) 
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  |a - b - c| - |c - a + b| = 0 :=
by
  sorry

end simplify_expression_of_triangle_side_lengths_l107_107009


namespace average_speed_with_stoppages_l107_107302

theorem average_speed_with_stoppages
    (D : ℝ) -- distance the train travels
    (T_no_stop : ℝ := D / 250) -- time taken to cover the distance without stoppages
    (T_with_stop : ℝ := 2 * T_no_stop) -- total time with stoppages
    : (D / T_with_stop) = 125 := 
by sorry

end average_speed_with_stoppages_l107_107302


namespace reduced_price_l107_107772

variable (P : ℝ)  -- the original price per kg
variable (reduction_factor : ℝ := 0.5)  -- 50% reduction
variable (extra_kgs : ℝ := 5)  -- 5 kgs more
variable (total_cost : ℝ := 800)  -- Rs. 800

theorem reduced_price :
  total_cost / (P * (1 - reduction_factor)) = total_cost / P + extra_kgs → 
  P / 2 = 80 :=
by
  sorry

end reduced_price_l107_107772


namespace find_friends_l107_107542

-- Definitions
def shells_Jillian : Nat := 29
def shells_Savannah : Nat := 17
def shells_Clayton : Nat := 8
def shells_per_friend : Nat := 27

-- Main statement
theorem find_friends :
  (shells_Jillian + shells_Savannah + shells_Clayton) / shells_per_friend = 2 :=
by
  sorry

end find_friends_l107_107542


namespace circle_radius_squared_l107_107140

theorem circle_radius_squared (r : ℝ) 
  (AB CD: ℝ) 
  (BP angleAPD : ℝ) 
  (P_outside_circle: True) 
  (AB_eq_12 : AB = 12) 
  (CD_eq_9 : CD = 9) 
  (AngleAPD_eq_45 : angleAPD = 45) 
  (BP_eq_10 : BP = 10) : r^2 = 73 :=
sorry

end circle_radius_squared_l107_107140


namespace distance_between_vertices_hyperbola_l107_107818

-- Definitions as per conditions
def hyperbola_eq (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Statement of the problem in Lean
theorem distance_between_vertices_hyperbola :
  ∀ x y : ℝ, hyperbola_eq x y → ∃ d : ℝ, d = 8 :=
by
  intros x y h
  use 8
  sorry

end distance_between_vertices_hyperbola_l107_107818


namespace decagon_adjacent_probability_l107_107111

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l107_107111


namespace trains_length_difference_eq_zero_l107_107410

theorem trains_length_difference_eq_zero
  (T1_pole_time : ℕ) (T1_platform_time : ℕ) (T2_pole_time : ℕ) (T2_platform_time : ℕ) (platform_length : ℕ)
  (h1 : T1_pole_time = 11)
  (h2 : T1_platform_time = 22)
  (h3 : T2_pole_time = 15)
  (h4 : T2_platform_time = 30)
  (h5 : platform_length = 120) :
  let L1 := T1_pole_time * platform_length / (T1_platform_time - T1_pole_time)
  let L2 := T2_pole_time * platform_length / (T2_platform_time - T2_pole_time)
  L1 = L2 :=
by
  sorry

end trains_length_difference_eq_zero_l107_107410


namespace solve_for_a_l107_107649

noncomputable def a := 3.6

theorem solve_for_a (h : 4 * ((a * 0.48 * 2.5) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) : 
    a = 3.6 :=
by
  sorry

end solve_for_a_l107_107649


namespace canvas_bag_lower_carbon_solution_l107_107267

theorem canvas_bag_lower_carbon_solution :
  ∀ (canvas_bag_CO2_pounds : ℕ) (plastic_bag_CO2_ounces : ℕ) 
    (plastic_bags_per_trip : ℕ) (ounces_per_pound : ℕ),
    canvas_bag_CO2_pounds = 600 →
    plastic_bag_CO2_ounces = 4 →
    plastic_bags_per_trip = 8 →
    ounces_per_pound = 16 →
    let total_CO2_per_trip := plastic_bags_per_trip * plastic_bag_CO2_ounces / ounces_per_pound in
    (canvas_bag_CO2_pounds / total_CO2_per_trip) = 300 :=
by
  -- Assume the given conditions
  assume canvas_bag_CO2_pounds plastic_bag_CO2_ounces plastic_bags_per_trip ounces_per_pound,
  assume canvas_bag_CO2_pounds_eq plastic_bag_CO2_ounces_eq plastic_bags_per_trip_eq ounces_per_pound_eq,
  -- Introduce the total carbon dioxide per trip
  let total_CO2_per_trip := plastic_bags_per_trip * plastic_bag_CO2_ounces / ounces_per_pound,
  -- Verify that the number of trips is 300
  show ((canvas_bag_CO2_pounds / total_CO2_per_trip) = 300),
  sorry

end canvas_bag_lower_carbon_solution_l107_107267


namespace line_circle_no_intersection_l107_107881

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l107_107881


namespace sector_max_area_l107_107036

theorem sector_max_area (P : ℝ) (R l S : ℝ) :
  (P > 0) → (2 * R + l = P) → (S = 1/2 * R * l) →
  (R = P / 4) ∧ (S = P^2 / 16) :=
by
  sorry

end sector_max_area_l107_107036


namespace union_complement_l107_107651

open Set Real

def P : Set ℝ := { x | x^2 - 4 * x + 3 ≤ 0 }
def Q : Set ℝ := { x | x^2 - 4 < 0 }

theorem union_complement :
  P ∪ (compl Q) = (Iic (-2)) ∪ Ici 1 :=
by
  sorry

end union_complement_l107_107651


namespace decagon_adjacent_vertex_probability_l107_107122

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l107_107122


namespace possible_integer_side_lengths_l107_107219

theorem possible_integer_side_lengths (a b : ℕ) (h1 : a = 8) (h2 : b = 5) :
  {x : ℕ | 3 < x ∧ x < 13}.finite ∧ {x : ℕ | 3 < x ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l107_107219


namespace num_ordered_pairs_l107_107347

theorem num_ordered_pairs : ∃ (n : ℕ), n = 24 ∧ ∀ (a b : ℂ), a^4 * b^6 = 1 ∧ a^8 * b^3 = 1 → n = 24 :=
by
  sorry

end num_ordered_pairs_l107_107347


namespace martian_calendar_months_l107_107530

theorem martian_calendar_months (x y : ℕ) 
  (h1 : 100 * x + 77 * y = 5882) : x + y = 74 :=
sorry

end martian_calendar_months_l107_107530


namespace max_yellow_apples_can_take_max_total_apples_can_take_l107_107587

structure Basket :=
  (total_apples : ℕ)
  (green_apples : ℕ)
  (yellow_apples : ℕ)
  (red_apples : ℕ)
  (green_lt_yellow : green_apples < yellow_apples)
  (yellow_lt_red : yellow_apples < red_apples)

def basket_conditions : Basket :=
  { total_apples := 44,
    green_apples := 11,
    yellow_apples := 14,
    red_apples := 19,
    green_lt_yellow := sorry,  -- 11 < 14
    yellow_lt_red := sorry }   -- 14 < 19

theorem max_yellow_apples_can_take : basket_conditions.yellow_apples = 14 := sorry

theorem max_total_apples_can_take : basket_conditions.green_apples 
                                     + basket_conditions.yellow_apples 
                                     + (basket_conditions.red_apples - 2) = 42 := sorry

end max_yellow_apples_can_take_max_total_apples_can_take_l107_107587


namespace total_apple_weight_proof_l107_107385

-- Define the weights of each fruit in terms of ounces
def weight_apple : ℕ := 4
def weight_orange : ℕ := 3
def weight_plum : ℕ := 2

-- Define the bag's capacity and the number of bags
def bag_capacity : ℕ := 49
def number_of_bags : ℕ := 5

-- Define the least common multiple (LCM) of the weights
def lcm_weight : ℕ := Nat.lcm weight_apple (Nat.lcm weight_orange weight_plum)

-- Define the largest multiple of LCM that is less than or equal to the bag's capacity
def max_lcm_multiple : ℕ := (bag_capacity / lcm_weight) * lcm_weight

-- Determine the number of each fruit per bag
def sets_per_bag : ℕ := max_lcm_multiple / lcm_weight
def apples_per_bag : ℕ := sets_per_bag * 1  -- 1 apple per set

-- Calculate the weight of apples per bag and total needed in all bags
def apple_weight_per_bag : ℕ := apples_per_bag * weight_apple
def total_apple_weight : ℕ := apple_weight_per_bag * number_of_bags

-- The statement to be proved in Lean
theorem total_apple_weight_proof : total_apple_weight = 80 := by
  sorry

end total_apple_weight_proof_l107_107385


namespace probability_quarter_circle_is_pi_div_16_l107_107198

open Real

noncomputable def probability_quarter_circle : ℝ :=
  let side_length := 2
  let total_area := side_length * side_length
  let quarter_circle_area := π / 4
  quarter_circle_area / total_area

theorem probability_quarter_circle_is_pi_div_16 :
  probability_quarter_circle = π / 16 :=
by
  sorry

end probability_quarter_circle_is_pi_div_16_l107_107198


namespace maurice_age_l107_107327

theorem maurice_age (M : ℕ) 
  (h₁ : 48 = 4 * (M + 5)) : M = 7 := 
by
  sorry

end maurice_age_l107_107327


namespace lego_tower_levels_l107_107379

theorem lego_tower_levels (initial_pieces : ℕ) (pieces_per_level : ℕ) (pieces_left : ℕ) 
    (h1 : initial_pieces = 100) (h2 : pieces_per_level = 7) (h3 : pieces_left = 23) :
    (initial_pieces - pieces_left) / pieces_per_level = 11 := 
by
  sorry

end lego_tower_levels_l107_107379


namespace problem_1_problem_2_l107_107505

noncomputable def f (x : ℝ) : ℝ := x^3 + 1 / (x + 1)

theorem problem_1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x ≥ 1 - x + x^2 := 
sorry

theorem problem_2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (h1 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≥ 1 - x + x^2) : f x > 3 / 4 := 
sorry

end problem_1_problem_2_l107_107505


namespace range_of_x_for_positive_y_l107_107015

theorem range_of_x_for_positive_y (x : ℝ) : 
  (-1 < x ∧ x < 3) ↔ (-x^2 + 2*x + 3 > 0) :=
sorry

end range_of_x_for_positive_y_l107_107015


namespace solve_system_of_equations_l107_107577

theorem solve_system_of_equations (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x^4 + y^4 - x^2 * y^2 = 13)
  (h2 : x^2 - y^2 + 2 * x * y = 1) :
  x = 1 ∧ y = 2 :=
sorry

end solve_system_of_equations_l107_107577


namespace find_incorrect_option_l107_107069

-- The given conditions from the problem
def incomes : List ℝ := [2, 2.5, 2.5, 2.5, 3, 3, 3, 3, 3, 4, 4, 5, 5, 9, 13]
def mean_incorrect : Prop := (incomes.sum / incomes.length) = 4
def option_incorrect : Prop := ¬ mean_incorrect

-- The goal is to prove that the statement about the mean being 4 is incorrect
theorem find_incorrect_option : option_incorrect := by
  sorry

end find_incorrect_option_l107_107069


namespace prime_solution_l107_107898

theorem prime_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : 5 * p + 3 * q = 91) : p = 17 ∧ q = 2 :=
by
  sorry

end prime_solution_l107_107898


namespace second_player_wins_l107_107294

noncomputable def is_winning_position (n : ℕ) : Prop :=
  n % 4 = 0

theorem second_player_wins (n : ℕ) (h : n = 100) :
  ∃ f : ℕ → ℕ, (∀ k, 0 < k → k ≤ n → (k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 5) → is_winning_position (n - k)) ∧ is_winning_position n := 
sorry

end second_player_wins_l107_107294


namespace bacteria_fill_sixteenth_of_dish_in_26_days_l107_107963

theorem bacteria_fill_sixteenth_of_dish_in_26_days
  (days_to_fill_dish : ℕ)
  (doubling_rate : ℕ → ℕ)
  (H1 : days_to_fill_dish = 30)
  (H2 : ∀ n, doubling_rate (n + 1) = 2 * doubling_rate n) :
  doubling_rate 26 = doubling_rate 30 / 2^4 :=
sorry

end bacteria_fill_sixteenth_of_dish_in_26_days_l107_107963


namespace course_selection_schemes_l107_107773

theorem course_selection_schemes (h1 : 7 > 4) (h2 : 2> 1) (h3 : ∀ (A B : Type) (s : Finset A), card s = 7) :
  (choose 7 4) - ((choose 2 2) * (choose 5 2)) = 25 :=
by sorry

end course_selection_schemes_l107_107773


namespace revenue_increase_l107_107141

theorem revenue_increase
  (P Q : ℝ)
  (h : 0 < P)
  (hQ : 0 < Q)
  (price_decrease : 0.90 = 0.90)
  (unit_increase : 2 = 2) :
  (0.90 * P) * (2 * Q) = 1.80 * (P * Q) :=
by
  sorry

end revenue_increase_l107_107141


namespace find_four_digit_numbers_l107_107709

theorem find_four_digit_numbers (a b c d : ℕ) : 
  (1000 ≤ 1000 * a + 100 * b + 10 * c + d) ∧ 
  (1000 * a + 100 * b + 10 * c + d ≤ 9999) ∧ 
  (1000 ≤ 1000 * d + 100 * c + 10 * b + a) ∧ 
  (1000 * d + 100 * c + 10 * b + a ≤ 9999) ∧
  (a + d = 9) ∧ 
  (b + c = 13) ∧
  (1001 * (a + d) + 110 * (b + c) = 19448) → 
  (1000 * a + 100 * b + 10 * c + d = 9949 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9859 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9769 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9679 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9589 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9499) :=
sorry

end find_four_digit_numbers_l107_107709


namespace kenneth_past_finish_line_l107_107788

theorem kenneth_past_finish_line (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) (time_biff : ℕ) (distance_kenneth : ℕ) :
  race_distance = 500 → biff_speed = 50 → kenneth_speed = 51 → time_biff = race_distance / biff_speed → distance_kenneth = kenneth_speed * time_biff → 
  distance_kenneth - race_distance = 10 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end kenneth_past_finish_line_l107_107788


namespace tan_11pi_over_6_l107_107298

theorem tan_11pi_over_6 : Real.tan (11 * Real.pi / 6) = - (Real.sqrt 3 / 3) :=
by
  sorry

end tan_11pi_over_6_l107_107298


namespace symmetric_points_sum_l107_107666

theorem symmetric_points_sum (a b : ℝ) (P Q : ℝ × ℝ) 
    (hP : P = (3, a)) (hQ : Q = (b, 2))
    (symm : P = (-Q.1, Q.2)) : a + b = -1 := by
  sorry

end symmetric_points_sum_l107_107666


namespace purchase_price_of_jacket_l107_107437

theorem purchase_price_of_jacket (S P : ℝ) (h1 : S = P + 0.30 * S)
                                (SP : ℝ) (h2 : SP = 0.80 * S)
                                (h3 : 8 = SP - P) :
                                P = 56 := by
  sorry

end purchase_price_of_jacket_l107_107437


namespace g_odd_find_a_f_increasing_l107_107195

-- Problem (I): Prove that if g(x) = f(x) - a is an odd function, then a = 1, given f(x) = 1 - 2/x.
theorem g_odd_find_a (f : ℝ → ℝ) (g : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = 1 - (2 / x)) → 
  (∀ x, g x = f x - a) → 
  (∀ x, g (-x) = - g x) → 
  a = 1 := 
  by
  intros h1 h2 h3
  sorry

-- Problem (II): Prove that f(x) is monotonically increasing on (0, +∞),
-- given f(x) = 1 - 2/x.

theorem f_increasing (f : ℝ → ℝ) : 
  (∀ x, f x = 1 - (2 / x)) → 
  ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → f x1 < f x2 := 
  by
  intros h1 x1 x2 hx1 hx12
  sorry

end g_odd_find_a_f_increasing_l107_107195


namespace part1_part2_l107_107510

theorem part1 (a b : ℝ) (h1 : ∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) (hb : b > 1) : a = 1 ∧ b = 2 :=
sorry

theorem part2 (k : ℝ) (x y : ℝ) (hx : x > 0) (hy : y > 0) (a b : ℝ) 
  (ha : a = 1) (hb : b = 2) 
  (h2 : a / x + b / y = 1)
  (h3 : 2 * x + y ≥ k^2 + k + 2) : -3 ≤ k ∧ k ≤ 2 :=
sorry

end part1_part2_l107_107510


namespace contrapositive_necessary_condition_l107_107663

theorem contrapositive_necessary_condition (a b : Prop) (h : a → b) : ¬b → ¬a :=
by
  sorry

end contrapositive_necessary_condition_l107_107663


namespace rectangle_area_l107_107944

variable (w l : ℕ)
variable (A : ℕ)
variable (H1 : l = 5 * w)
variable (H2 : 2 * l + 2 * w = 180)

theorem rectangle_area : A = 1125 :=
by
  sorry

end rectangle_area_l107_107944


namespace houses_with_both_l107_107525

theorem houses_with_both (G P N Total B : ℕ) 
  (hG : G = 50) 
  (hP : P = 40) 
  (hN : N = 10) 
  (hTotal : Total = 65)
  (hEquation : G + P - B = Total - N) 
  : B = 35 := 
by 
  sorry

end houses_with_both_l107_107525


namespace no_intersection_l107_107888

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l107_107888


namespace line_circle_no_intersection_l107_107893

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l107_107893


namespace range_of_a_l107_107839

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem range_of_a :
  (∃ (a : ℝ), (a ≤ -2 ∨ a ≥ 0) ∧ (∃ (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 4), f x ≤ a^2 + 2 * a)) :=
by sorry

end range_of_a_l107_107839


namespace simplify_expression_l107_107933

theorem simplify_expression (x : ℝ) (hx : x^2 - 2*x = 0) (hx_nonzero : x ≠ 0) :
  (1 + 1 / (x - 1)) / (x / (x^2 - 1)) = 3 :=
sorry

end simplify_expression_l107_107933


namespace books_in_shipment_l107_107427

theorem books_in_shipment (B : ℕ) (h : 3 / 4 * B = 180) : B = 240 :=
sorry

end books_in_shipment_l107_107427


namespace min_value_of_expression_l107_107046

theorem min_value_of_expression (a b c : ℝ) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) (h3 : a * b * c = 27) :
  a^2 + 2*a*b + b^2 + 3*c^2 ≥ 324 :=
sorry

end min_value_of_expression_l107_107046


namespace part_1_part_2_1_part_2_2_l107_107006

variable {k x : ℝ}
def y (k : ℝ) (x : ℝ) := k * x^2 - 2 * k * x + 2 * k - 1

theorem part_1 (k : ℝ) : (∀ x, y k x ≥ 4 * k - 2) ↔ (0 ≤ k ∧ k ≤ 1 / 3) := by
  sorry

theorem part_2_1 (k : ℝ) : ¬∃ x1 x2 : ℝ, y k x = 0 ∧ y k x = 0 ∧ x1^2 + x2^2 = 3 * x1 * x2 - 4 := by
  sorry

theorem part_2_2 (k : ℝ) : (∀ x1 x2 : ℝ, y k x = 0 ∧ y k x = 0 ∧ x1 > 0 ∧ x2 > 0) ↔ (1 / 2 < k ∧ k < 1) := by
  sorry

end part_1_part_2_1_part_2_2_l107_107006


namespace geometric_sequence_and_sum_l107_107840

theorem geometric_sequence_and_sum (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ)
  (h_a1 : a 1 = 3/2)
  (h_a_recur : ∀ n : ℕ, a (n + 1) = 3 * a n - 1)
  (h_b_def : ∀ n : ℕ, b n = a n - 1/2) :
  (∀ n : ℕ, b (n + 1) = 3 * b n ∧ b 1 = 1) ∧ 
  (∀ n : ℕ, S n = (3^n + n - 1) / 2) :=
sorry

end geometric_sequence_and_sum_l107_107840


namespace line_circle_no_intersection_l107_107869

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l107_107869


namespace determine_positions_l107_107570

-- Defining the participants
inductive Participant
| Olya
| Oleg
| Pasha

open Participant

-- Defining the possible places
inductive Place
| First
| Second
| Third

open Place

-- Define the conditions
def condition1 (pos : Participant → Place) : Prop := 
  pos Olya = First ∨ pos Oleg = First ∨ pos Pasha = First

def condition2 (pos : Participant → Place) : Prop :=
  (pos Olya = First ∧ pos Olya = Second ∧ pos Olya = Third) ∨
  (pos Oleg = First ∧ pos Oleg = Second ∧ pos Oleg = Third) ∨
  (pos Pasha = First ∧ pos Pasha = Second ∧ pos Pasha = Third)

def condition3 (pos : Participant → Place) : Prop :=
  ∀ p, pos p ≠ First ∧ pos p ≠ Second ∧ pos p ≠ Third

def condition4 (pos : Participant → Place) : Prop :=
  (pos Olya = First → (pos Oleg = First ∨ pos Pasha = First)) ∧
  (pos Oleg = First → pos Olya ≠ First) ∧
  (pos Pasha = First → (pos Oleg = First ∨ pos Olya = First))

def always_true_or_false : Prop :=
  (∀ p, p = Olya ∨ p = Oleg ∨ p = Pasha )

-- Main theorem
theorem determine_positions (pos : Participant → Place) :
  condition1 pos ∧ condition2 pos ∧ condition3 pos ∧ condition4 pos ∧ always_true_or_false →
  pos Oleg = First ∧ pos Pasha = Second ∧ pos Olya = Third := 
by
  sorry

end determine_positions_l107_107570


namespace sin_double_angle_l107_107906

-- Lean code to define the conditions and represent the problem
variable (α : ℝ)
variable (x y : ℝ) 
variable (r : ℝ := Real.sqrt (x^2 + y^2))

-- Given conditions
def point_on_terminal_side (x y : ℝ) (h : x = 1 ∧ y = -2) : Prop :=
  ∃ α, (⟨1, -2⟩ : ℝ × ℝ) = ⟨Real.cos α * (Real.sqrt (1^2 + (-2)^2)), Real.sin α * (Real.sqrt (1^2 + (-2)^2))⟩

-- The theorem to prove
theorem sin_double_angle (h : point_on_terminal_side 1 (-2) ⟨rfl, rfl⟩) : 
  Real.sin (2 * α) = -4 / 5 := 
sorry

end sin_double_angle_l107_107906


namespace line_circle_no_intersection_l107_107866

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l107_107866


namespace abigail_fence_building_time_l107_107325

def abigail_time_per_fence (total_built: ℕ) (additional_hours: ℕ) (total_fences: ℕ): ℕ :=
  (additional_hours * 60) / (total_fences - total_built)

theorem abigail_fence_building_time :
  abigail_time_per_fence 10 8 26 = 30 :=
sorry

end abigail_fence_building_time_l107_107325


namespace men_in_second_group_l107_107603

theorem men_in_second_group (m w : ℝ) (x : ℝ) 
  (h1 : 3 * m + 8 * w = x * m + 2 * w) 
  (h2 : 2 * m + 2 * w = (3 / 7) * (3 * m + 8 * w)) : x = 6 :=
by
  sorry

end men_in_second_group_l107_107603


namespace intersection_is_expected_result_l107_107197

def set_A : Set ℝ := { x | x * (x + 1) > 0 }
def set_B : Set ℝ := { x | ∃ y, y = Real.sqrt (x - 1) }
def expected_result : Set ℝ := { x | x ≥ 1 }

theorem intersection_is_expected_result : set_A ∩ set_B = expected_result := by
  sorry

end intersection_is_expected_result_l107_107197


namespace problem_l107_107230

variable (x y z : ℚ)

-- Conditions as definitions
def cond1 : Prop := x / y = 3
def cond2 : Prop := y / z = 5 / 2

-- Theorem statement with the final proof goal
theorem problem (h1 : cond1 x y) (h2 : cond2 y z) : z / x = 2 / 15 := 
by 
  sorry

end problem_l107_107230


namespace exist_N_for_fn_eq_n_l107_107669

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom f_condition1 (m n : ℕ+) : (f m, f n) ≤ (m, n) ^ 2014
axiom f_condition2 (n : ℕ+) : n ≤ f n ∧ f n ≤ n + 2014

theorem exist_N_for_fn_eq_n :
  ∃ N : ℕ+, ∀ n : ℕ+, n ≥ N → f n = n := sorry

end exist_N_for_fn_eq_n_l107_107669


namespace probability_of_all_female_l107_107927

noncomputable def probability_all_females_final (females males total chosen : ℕ) : ℚ :=
  (females.choose chosen) / (total.choose chosen)

theorem probability_of_all_female:
  probability_all_females_final 5 3 8 3 = 5 / 28 :=
by
  sorry

end probability_of_all_female_l107_107927


namespace scientific_notation_l107_107486

theorem scientific_notation : 350000000 = 3.5 * 10^8 :=
by
  sorry

end scientific_notation_l107_107486


namespace line_circle_no_intersection_l107_107865

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l107_107865


namespace bill_earnings_per_ounce_l107_107617

-- Given conditions
def ounces_sold : Nat := 8
def fine : Nat := 50
def money_left : Nat := 22
def total_money_earned : Nat := money_left + fine -- $72

-- The amount earned for every ounce of fool's gold
def price_per_ounce : Nat := total_money_earned / ounces_sold -- 72 / 8

-- The proof statement
theorem bill_earnings_per_ounce (h: price_per_ounce = 9) : True :=
by
  trivial

end bill_earnings_per_ounce_l107_107617


namespace carla_marble_purchase_l107_107337

variable (started_with : ℕ) (now_has : ℕ) (bought : ℕ)

theorem carla_marble_purchase (h1 : started_with = 53) (h2 : now_has = 187) : bought = 134 := by
  sorry

end carla_marble_purchase_l107_107337


namespace arithmetic_sequence_general_formula_l107_107400

def f (x : ℝ) : ℝ := x^2 - 4 * x + 2

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem arithmetic_sequence_general_formula (a : ℕ → ℝ) (x : ℝ)
  (h_arith : arithmetic_seq a)
  (h1 : a 1 = f (x + 1))
  (h2 : a 2 = 0)
  (h3 : a 3 = f (x - 1)) :
  ∀ n, a n = 2 * n - 4 ∨ a n = 4 - 2 * n :=
by
  sorry

end arithmetic_sequence_general_formula_l107_107400


namespace value_of_expression_l107_107905

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) : 4 / 7 + (2 * q - p) / (2 * q + p) = 1 := by
  sorry

end value_of_expression_l107_107905


namespace probability_adjacent_vertices_in_decagon_l107_107092

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l107_107092


namespace max_planes_l107_107446

theorem max_planes (n : ℕ) (h_pos : n = 15) : 
    ∃ planes : ℕ, planes = Nat.choose 15 3 ∧ planes = 455 :=
by
  use Nat.choose 15 3
  split
  . rfl
  . simp [Nat.choose]
  sorry

end max_planes_l107_107446


namespace light_intensity_after_glass_pieces_minimum_glass_pieces_l107_107710

theorem light_intensity_after_glass_pieces (a : ℝ) (x : ℕ) : 
  (y : ℝ) = a * (0.9 ^ x) :=
sorry

theorem minimum_glass_pieces (a : ℝ) (x : ℕ) : 
  a * (0.9 ^ x) < a / 3 ↔ x ≥ 11 :=
sorry

end light_intensity_after_glass_pieces_minimum_glass_pieces_l107_107710


namespace not_perfect_cube_of_N_l107_107760

-- Define a twelve-digit number
def N : ℕ := 100000000000

-- Define the condition that a number is a perfect cube
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℤ, n = k ^ 3

-- Problem statement: Prove that 100000000000 is not a perfect cube
theorem not_perfect_cube_of_N : ¬ is_perfect_cube N :=
by sorry

end not_perfect_cube_of_N_l107_107760


namespace line_circle_no_intersection_l107_107883

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l107_107883


namespace exists_neg_monomial_l107_107961

theorem exists_neg_monomial (a : ℤ) (x y : ℤ) (m n : ℕ) (hq : a < 0) (hd : m + n = 5) :
  ∃ a m n, a < 0 ∧ m + n = 5 ∧ a * x^m * y^n = -x^2 * y^3 :=
by
  sorry

end exists_neg_monomial_l107_107961


namespace mrs_hilt_water_fountain_trips_l107_107924

theorem mrs_hilt_water_fountain_trips (d : ℕ) (t : ℕ) (n : ℕ) 
  (h1 : d = 30) 
  (h2 : t = 120) 
  (h3 : 2 * d * n = t) : 
  n = 2 :=
by
  -- Proof omitted
  sorry

end mrs_hilt_water_fountain_trips_l107_107924


namespace max_value_expression_l107_107396

theorem max_value_expression (a b c d : ℝ) 
  (h1 : -11.5 ≤ a ∧ a ≤ 11.5)
  (h2 : -11.5 ≤ b ∧ b ≤ 11.5)
  (h3 : -11.5 ≤ c ∧ c ≤ 11.5)
  (h4 : -11.5 ≤ d ∧ d ≤ 11.5):
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a ≤ 552 :=
by
  sorry

end max_value_expression_l107_107396


namespace f_monotonicity_l107_107504

noncomputable def f (a x : ℝ) : ℝ := a^x + x^2 - x * Real.log a

theorem f_monotonicity (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x : ℝ, x > 0 → deriv (f a) x > 0) ∧ (∀ x : ℝ, x < 0 → deriv (f a) x < 0) :=
by
  sorry

end f_monotonicity_l107_107504


namespace sum_of_cubes_minus_tripled_product_l107_107022

theorem sum_of_cubes_minus_tripled_product (a b c d : ℝ) 
  (h1 : a + b + c + d = 15)
  (h2 : ab + ac + ad + bc + bd + cd = 40) :
  a^3 + b^3 + c^3 + d^3 - 3 * a * b * c * d = 1695 :=
by
  sorry

end sum_of_cubes_minus_tripled_product_l107_107022


namespace greatest_odd_factors_below_200_l107_107720

theorem greatest_odd_factors_below_200 :
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (square m → m ≤ n)) ∧ n = 196 :=
by
sorry

end greatest_odd_factors_below_200_l107_107720


namespace vote_ratio_l107_107694

theorem vote_ratio (X Y Z : ℕ) (hZ : Z = 25000) (hX : X = 22500) (hX_Y : X = Y + (1/2 : ℚ) * Y) 
    : Y / (Z - Y) = 2 / 5 := 
by 
  sorry

end vote_ratio_l107_107694


namespace largest_unrepresentable_l107_107551

theorem largest_unrepresentable (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : Nat.gcd b c = 1) (h3 : Nat.gcd c a = 1)
  : ¬ ∃ (x y z : ℕ), x * b * c + y * c * a + z * a * b = 2 * a * b * c - a * b - b * c - c * a :=
by
  -- The proof is omitted
  sorry

end largest_unrepresentable_l107_107551


namespace packs_used_after_6_weeks_l107_107161

-- Define the conditions as constants or definitions.
def pages_per_class_per_day : ℕ := 2
def num_classes : ℕ := 5
def days_per_week : ℕ := 5
def weeks : ℕ := 6
def pages_per_pack : ℕ := 100

-- The total number of packs of notebook paper Chip will use after 6 weeks
theorem packs_used_after_6_weeks : (pages_per_class_per_day * num_classes * days_per_week * weeks) / pages_per_pack = 3 := 
by
  -- skip the proof
  sorry

end packs_used_after_6_weeks_l107_107161


namespace max_planes_l107_107447

theorem max_planes (n : ℕ) (h_pos : n = 15) : 
    ∃ planes : ℕ, planes = Nat.choose 15 3 ∧ planes = 455 :=
by
  use Nat.choose 15 3
  split
  . rfl
  . simp [Nat.choose]
  sorry

end max_planes_l107_107447


namespace find_a_plus_b_plus_c_l107_107697

noncomputable def max_possible_area_triangle_BPE
    (A B C D P E : Point)
    (AB AC BC : ℝ)
    (h_ABC : Triangle A B C)
    (AB_eq : distance A B = AB)
    (BC_eq : distance B C = BC)
    (CA_eq : distance C A = AC)
    (D_on_BC : collinear B D C)
    (E_mid_BC : middle E B C)
    (I_B I_C : Point)
    (I_B_incenter_ABD : incenter I_B (Triangle A B D))
    (I_C_incenter_ACD : incenter I_C (Triangle A C D))
    (P_intersections : on_circumcircle P (Triangle B I_B D) ∧ on_circumcircle P (Triangle C I_C D))
    : ℝ :=
    let maximum_area := 0 - 25 * Real.sqrt 3
    in 28

theorem find_a_plus_b_plus_c :
    ∃ a b c : ℕ, a - b * Real.sqrt c = max_possible_area_triangle_BPE A B C D P E 8 12 10 h_ABC AB_eq BC_eq CA_eq D_on_BC E_mid_BC I_B I_C I_B_incenter_ABD I_C_incenter_ACD P_intersections ∧ c ∉ SquareNumbers ∧ a + b + c = 28 :=
by {
  sorry
}

end find_a_plus_b_plus_c_l107_107697


namespace sequence_m_value_l107_107548

theorem sequence_m_value (m : ℕ) (a : ℕ → ℝ) (h₀ : a 0 = 37) (h₁ : a 1 = 72)
  (hm : a m = 0) (h_rec : ∀ k, 1 ≤ k ∧ k < m → a (k + 1) = a (k - 1) - 3 / a k) : m = 889 :=
sorry

end sequence_m_value_l107_107548


namespace triangle_side_count_l107_107214

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end triangle_side_count_l107_107214


namespace range_exponential_2_intersection_exponential_fn_xx_l107_107355

noncomputable def lg (x : ℝ) : ℝ := Math.log x

noncomputable def A := {x : ℝ | -1 < x ∧ x < 2}

def exponential_fn (x : ℝ) (a : ℝ) := a ^ x

theorem range_exponential_2 (A : Set ℝ) (B : Set ℝ) : 
  A = {x | -1 < x ∧ x <2} → 
  ∃ A B, (exponential_fn x 2) x ∈ ExponentialFn x 2) = (B ∪ A) → 
  B = {y | (1 / 2) < y ∧ y < 4} → 
  (A ∪ B = {x | -1 < x ∧ x < 4})
:= 
begin
    sorry
end

theorem intersection_exponential_fn_xx (A : Set ℝ) (a : ℝ) (B : Set ℝ) :
  a ≠ 1 ∧ 0 < a → 
  A = {x | -1 < x ∧ x < 2} →
  A ∩ {y | exbonential_fn x a |}  = {\frac{1}{2} , 2\} → 
  a = 2 := 
  begin
    sorry
end

end range_exponential_2_intersection_exponential_fn_xx_l107_107355


namespace largest_possible_c_l107_107258

theorem largest_possible_c (c : ℝ) (hc : (3 * c + 4) * (c - 2) = 9 * c) : c ≤ 4 :=
sorry

end largest_possible_c_l107_107258


namespace wall_length_is_800_l107_107764

def brick_volume : ℝ := 50 * 11.25 * 6
def total_brick_volume : ℝ := 3200 * brick_volume
def wall_volume (x : ℝ) : ℝ := x * 600 * 22.5

theorem wall_length_is_800 :
  ∀ (x : ℝ), total_brick_volume = wall_volume x → x = 800 :=
by
  intros x h
  sorry

end wall_length_is_800_l107_107764


namespace cary_strips_ivy_l107_107991

variable (strip_per_day : ℕ) (grow_per_night : ℕ) (total_ivy : ℕ)

theorem cary_strips_ivy (h1 : strip_per_day = 6) (h2 : grow_per_night = 2) (h3 : total_ivy = 40) :
  (total_ivy / (strip_per_day - grow_per_night)) = 10 := by
  sorry

end cary_strips_ivy_l107_107991


namespace dice_probability_l107_107425

theorem dice_probability :
  let num_dice := 6
  let prob_one_digit := 9 / 20
  let prob_two_digit := 11 / 20
  let num_combinations := Nat.choose num_dice (num_dice / 2)
  let prob_each_combination := (prob_one_digit ^ 3) * (prob_two_digit ^ 3)
  let total_probability := num_combinations * prob_each_combination
  total_probability = 4851495 / 16000000 := by
    let num_dice := 6
    let prob_one_digit := 9 / 20
    let prob_two_digit := 11 / 20
    let num_combinations := Nat.choose num_dice (num_dice / 2)
    let prob_each_combination := (prob_one_digit ^ 3) * (prob_two_digit ^ 3)
    let total_probability := num_combinations * prob_each_combination
    sorry

end dice_probability_l107_107425


namespace tree_heights_l107_107589

theorem tree_heights :
  let Tree_A := 150
  let Tree_B := (2/3 : ℝ) * Tree_A
  let Tree_C := (1/2 : ℝ) * Tree_B
  let Tree_D := Tree_C + 25
  let Tree_E := 0.40 * Tree_A
  let Tree_F := (Tree_B + Tree_D) / 2
  let Tree_G := (3/8 : ℝ) * Tree_A
  let Tree_H := 1.25 * Tree_F
  let Tree_I := 0.60 * (Tree_E + Tree_G)
  let total_height := Tree_A + Tree_B + Tree_C + Tree_D + Tree_E + Tree_F + Tree_G + Tree_H + Tree_I
  Tree_A = 150 ∧
  Tree_B = 100 ∧
  Tree_C = 50 ∧
  Tree_D = 75 ∧
  Tree_E = 60 ∧
  Tree_F = 87.5 ∧
  Tree_G = 56.25 ∧
  Tree_H = 109.375 ∧
  Tree_I = 69.75 ∧
  total_height = 758.125 :=
by
  sorry

end tree_heights_l107_107589


namespace max_planes_15_points_l107_107441

theorem max_planes_15_points (P : Finset (Fin 15)) (hP : ∀ (p1 p2 p3 : Fin 15), p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3) :
  P.card = 15 → (∃ planes : Finset (Finset (Fin 15)), planes.card = 455) := by
  sorry

end max_planes_15_points_l107_107441


namespace price_of_each_book_l107_107970

theorem price_of_each_book (B P : ℕ) 
  (h1 : (1 / 3 : ℚ) * B = 36) -- Number of unsold books is 1/3 of the total books and it equals 36
  (h2 : (2 / 3 : ℚ) * B * P = 144) -- Total amount received for the books sold is $144
  : P = 2 := 
by
  sorry

end price_of_each_book_l107_107970


namespace no_intersection_l107_107886

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l107_107886


namespace tan_phi_l107_107005

theorem tan_phi (φ : ℝ) (h1 : Real.cos (π / 2 + φ) = 2 / 3) (h2 : abs φ < π / 2) : 
  Real.tan φ = -2 * Real.sqrt 5 / 5 := 
by 
  sorry

end tan_phi_l107_107005


namespace probability_adjacent_vertices_in_decagon_l107_107109

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l107_107109


namespace incorrect_proposition_b_l107_107838

axiom plane (α β : Type) : Prop
axiom line (m n : Type) : Prop
axiom parallel (a b : Type) : Prop
axiom perpendicular (a b : Type) : Prop
axiom intersection (α β : Type) (n : Type) : Prop
axiom contained (a b : Type) : Prop

theorem incorrect_proposition_b (α β m n : Type)
  (hαβ_plane : plane α β)
  (hmn_line : line m n)
  (h_parallel_m_α : parallel m α)
  (h_intersection : intersection α β n) :
  ¬ parallel m n :=
sorry

end incorrect_proposition_b_l107_107838


namespace placement_proof_l107_107564

def claimed_first_place (p: String) : Prop := 
  p = "Olya" ∨ p = "Oleg" ∨ p = "Pasha"

def odd_places_boys (positions: ℕ → String) : Prop := 
  (positions 1 = "Oleg" ∨ positions 1 = "Pasha") ∧ (positions 3 = "Oleg" ∨ positions 3 = "Pasha")

def olya_wrong (positions : ℕ → String) : Prop := 
  ¬odd_places_boys positions

def always_truthful_or_lying (Olya_st: Prop) (Oleg_st: Prop) (Pasha_st: Prop) : Prop := 
  Olya_st = Oleg_st ∧ Oleg_st = Pasha_st

def competition_placement : Prop :=
  ∃ (positions: ℕ → String),
    claimed_first_place (positions 1) ∧
    claimed_first_place (positions 2) ∧
    claimed_first_place (positions 3) ∧
    (positions 1 = "Oleg") ∧
    (positions 2 = "Pasha") ∧
    (positions 3 = "Olya") ∧
    olya_wrong positions ∧
    always_truthful_or_lying
      ((claimed_first_place "Olya" ∧ odd_places_boys positions))
      ((claimed_first_place "Oleg" ∧ olya_wrong positions))
      (claimed_first_place "Pasha")

theorem placement_proof : competition_placement :=
  sorry

end placement_proof_l107_107564


namespace extreme_values_number_of_zeros_l107_107354

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5
noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem extreme_values :
  (∀ x : ℝ, f x ≤ 12) ∧ (f (-1) = 12) ∧ (∀ x : ℝ, -15 ≤ f x) ∧ (f 2 = -15) := 
sorry

theorem number_of_zeros (m : ℝ) :
  (m > 12 ∨ m < -15 → ∃! x : ℝ, g x m = 0) ∧
  (m = 12 ∨ m = -15 → ∃ x y : ℝ, x ≠ y ∧ g x m = 0 ∧ g y m = 0) ∧
  (-15 < m ∧ m < 12 → ∃ x y z : ℝ, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ g x m = 0 ∧ g y m = 0 ∧ g z m = 0) :=
sorry

end extreme_values_number_of_zeros_l107_107354


namespace cacti_average_height_l107_107980

variables {Cactus1 Cactus2 Cactus3 Cactus4 Cactus5 Cactus6 : ℕ}
variables (condition1 : Cactus1 = 14)
variables (condition3 : Cactus3 = 7)
variables (condition6 : Cactus6 = 28)
variables (condition2 : Cactus2 = 14)
variables (condition4 : Cactus4 = 14)
variables (condition5 : Cactus5 = 14)

theorem cacti_average_height : 
  (Cactus1 + Cactus2 + Cactus3 + Cactus4 + Cactus5 + Cactus6 : ℕ) = 91 → 
  (91 : ℝ) / 6 = (15.2 : ℝ) :=
by
  sorry

end cacti_average_height_l107_107980


namespace problem_1_problem_2_problem_3_l107_107183

variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π)
variable (h2 : Real.tan α = -2)

theorem problem_1 : Real.sin (α + (π / 6)) = (2 * Real.sqrt 15 - Real.sqrt 5) / 10 := by
  sorry

theorem problem_2 : (2 * Real.cos ((π / 2) + α) - Real.cos (π - α)) / (Real.sin ((π / 2) - α) - 3 * Real.sin (π + α)) = 5 / 7 := by
  sorry

theorem problem_3 : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 11 / 5 := by
  sorry

end problem_1_problem_2_problem_3_l107_107183


namespace polyhedron_with_12_edges_l107_107783

def prism_edges (n : Nat) : Nat :=
  3 * n

def pyramid_edges (n : Nat) : Nat :=
  2 * n

def Quadrangular_prism : Nat := prism_edges 4
def Quadrangular_pyramid : Nat := pyramid_edges 4
def Pentagonal_pyramid : Nat := pyramid_edges 5
def Pentagonal_prism : Nat := prism_edges 5

theorem polyhedron_with_12_edges :
  (Quadrangular_prism = 12) ∧
  (Quadrangular_pyramid ≠ 12) ∧
  (Pentagonal_pyramid ≠ 12) ∧
  (Pentagonal_prism ≠ 12) := by
  sorry

end polyhedron_with_12_edges_l107_107783


namespace vector_identity_l107_107382

-- Define unit vectors and vector cross and dot products
variables {V : Type*} [inner_product_space ℝ V]

-- Assumptions
variables (a b c : V)
variable [decidable_eq V]

-- Define the unit vector property
axiom unit_vectors : ∥a∥ = 1 ∧ ∥b∥ = 1

-- Define the given vector relations
axiom c_def : c = a × b + b
axiom cross_relation : c × b = a

-- Main theorem statement
theorem vector_identity : b ⋅ (a × c) = -1 := 
sorry

end vector_identity_l107_107382


namespace maximum_planes_l107_107442

-- Definitions for conditions
def is_non_collinear (points : set (ℝ^3)) : Prop :=
  ∀ (p1 p2 p3 : ℝ^3), {p1, p2, p3} ⊆ points → (∃ plane : set (ℝ^3), ∀ p ∈ {p1, p2, p3}, p ∈ plane) ∧ ¬collinear p1 p2 p3

def is_non_coplanar (points : set (ℝ^3)) : Prop :=
  ∀ (p1 p2 p3 p4 : ℝ^3), {p1, p2, p3, p4} ⊆ points → ¬coplanar p1 p2 p3 p4

noncomputable def combination_3 (n : ℕ) : ℕ :=
  nat.choose n 3

-- Main theorem to be proven
theorem maximum_planes (S : set (ℝ^3)) (h1 : is_non_collinear S) (h2 : is_non_coplanar S) (h3 : finset.card S = 15) :
  (combination_3 15) = 455 :=
by
  sorry -- this skips the actual proof

end maximum_planes_l107_107442


namespace neither_sufficient_nor_necessary_l107_107067

theorem neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a + b > 0 → ab > 0) ∧ (ab > 0 → a + b > 0)) :=
by {
  sorry
}

end neither_sufficient_nor_necessary_l107_107067


namespace second_term_of_geometric_series_l107_107151

noncomputable def geometric_series_second_term (a r : ℝ) (S : ℝ) : ℝ :=
a * r

theorem second_term_of_geometric_series 
  (a r S : ℝ) 
  (h1 : r = 1 / 4) 
  (h2 : S = 10) 
  (h3 : S = a / (1 - r)) 
  : geometric_series_second_term a r S = 1.875 :=
by
  sorry

end second_term_of_geometric_series_l107_107151


namespace length_of_second_platform_l107_107147

/-- 
Let L be the length of the second platform.
A train crosses a platform of 100 m in 15 sec.
The same train crosses another platform in 20 sec.
The length of the train is 350 m.
Prove that the length of the second platform is 250 meters.
-/
theorem length_of_second_platform (L : ℕ) (train_length : ℕ) (platform1_length : ℕ) (time1 : ℕ) (time2 : ℕ):
  train_length = 350 → platform1_length = 100 → time1 = 15 → time2 = 20 → L = 250 :=
by
  sorry

end length_of_second_platform_l107_107147


namespace find_number_l107_107081

theorem find_number (x : ℝ) (h : 7 * x = 50.68) : x = 7.24 :=
sorry

end find_number_l107_107081


namespace largest_possible_value_of_c_l107_107260

theorem largest_possible_value_of_c (c : ℚ) : (3 * c + 4) * (c - 2) = 9 * c → c ≤ 4 :=
by
  intro h
  have : (3 * c + 4) * (c - 2) = 3 * c^2 - 6 * c + 4 * c - 8 := 
    calc 
    (3 * c + 4) * (c - 2) = (3 * c) * (c - 2) + 4 * (c - 2) : by ring
                         ... = (3 * c) * c - (3 * c) * 2 + 4 * c - 4 * 2 : by ring
                         ... = 3 * c^2 - 6 * c + 4 * c - 8 : by ring
  rw this at h
  have h2 : 3 * c^2 - 11 * c - 8 = 0 := by nlinarith
  sorry

end largest_possible_value_of_c_l107_107260


namespace line_circle_no_intersection_l107_107853

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l107_107853


namespace decagon_adjacent_vertex_probability_l107_107123

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l107_107123


namespace minimum_degree_q_l107_107412

variable (p q r : Polynomial ℝ)

theorem minimum_degree_q (h1 : 2 * p + 5 * q = r)
                        (hp : p.degree = 7)
                        (hr : r.degree = 10) :
  q.degree = 10 :=
sorry

end minimum_degree_q_l107_107412


namespace purchase_price_is_600_l107_107700

open Real

def daily_food_cost : ℝ := 20
def num_days : ℝ := 40
def vaccination_cost : ℝ := 500
def selling_price : ℝ := 2500
def profit : ℝ := 600

def total_food_cost : ℝ := daily_food_cost * num_days
def total_expenses : ℝ := total_food_cost + vaccination_cost
def total_cost : ℝ := selling_price - profit
def purchase_price : ℝ := total_cost - total_expenses

theorem purchase_price_is_600 : purchase_price = 600 := by
  sorry

end purchase_price_is_600_l107_107700


namespace vendor_profit_l107_107455

theorem vendor_profit {s₁ s₂ c₁ c₂ : ℝ} (h₁ : s₁ = 80) (h₂ : s₂ = 80) (profit₁ : s₁ = c₁ * 1.60) (loss₂ : s₂ = c₂ * 0.80) 
: (s₁ + s₂) - (c₁ + c₂) = 10 := by 
  sorry

end vendor_profit_l107_107455


namespace cross_section_area_l107_107448

-- Definitions for the conditions stated in the problem
def frustum_height : ℝ := 6
def upper_base_side : ℝ := 4
def lower_base_side : ℝ := 8

-- The main statement to be proved
theorem cross_section_area :
  (exists (cross_section_area : ℝ),
    cross_section_area = 16 * Real.sqrt 6) :=
sorry

end cross_section_area_l107_107448


namespace determine_initial_sum_l107_107982

def initial_sum_of_money (P r : ℝ) : Prop :=
  (600 = P + 2 * P * r) ∧ (700 = P + 2 * P * (r + 0.1))

theorem determine_initial_sum (P r : ℝ) (h : initial_sum_of_money P r) : P = 500 :=
by
  cases h with
  | intro h1 h2 =>
    sorry

end determine_initial_sum_l107_107982


namespace kittens_count_l107_107701

def initial_kittens : ℕ := 8
def additional_kittens : ℕ := 2
def total_kittens : ℕ := 10

theorem kittens_count : initial_kittens + additional_kittens = total_kittens := by
  -- Proof will go here
  sorry

end kittens_count_l107_107701


namespace hyperbola_center_l107_107176

theorem hyperbola_center :
  (∃ h k : ℝ,
    (∀ x y : ℝ, ((4 * x - 8) / 9)^2 - ((5 * y + 5) / 7)^2 = 1 ↔ (x - h)^2 / (81 / 16) - (y - k)^2 / (49 / 25) = 1) ∧
    (h = 2) ∧ (k = -1)) :=
sorry

end hyperbola_center_l107_107176


namespace competition_result_l107_107557

theorem competition_result :
  (∀ (Olya Oleg Pasha : Nat), Olya = 1 ∨ Oleg = 1 ∨ Pasha = 1) → 
  (∀ (Olya Oleg Pasha : Nat), (Olya = 1 ∨ Olya = 3) → false) →
  (∀ (Olya Oleg Pasha : Nat), Oleg ≠ 1) →
  (∀ (Olya Oleg Pasha : Nat), (Olya = Oleg ∧ Olya ≠ Pasha)) →
  ∃ (Olya Oleg Pasha : Nat), Oleg = 1 ∧ Pasha = 2 ∧ Olya = 3 :=
begin
  assume each_claims_first,
  assume olya_odd_places_false,
  assume oleg_truthful,
  assume truth_liar_cond,
  sorry
end

end competition_result_l107_107557


namespace part1_simplification_part2_inequality_l107_107308

-- Part 1: Prove the simplification of the algebraic expression
theorem part1_simplification (x : ℝ) (h₁ : x ≠ 3):
  (2 * x + 4) / (x^2 - 6 * x + 9) / ((2 * x - 1) / (x - 3) - 1) = 2 / (x - 3) :=
sorry

-- Part 2: Prove the solution set for the inequality system
theorem part2_inequality (x : ℝ) :
  (5 * x - 2 > 3 * (x + 1)) → (1/2 * x - 1 ≥ 7 - 3/2 * x) → x ≥ 4 :=
sorry

end part1_simplification_part2_inequality_l107_107308


namespace more_balloons_l107_107761

theorem more_balloons (you_balloons : ℕ) (friend_balloons : ℕ) (h_you : you_balloons = 7) (h_friend : friend_balloons = 5) : 
  you_balloons - friend_balloons = 2 :=
sorry

end more_balloons_l107_107761


namespace possible_integer_side_lengths_l107_107210

theorem possible_integer_side_lengths : 
  {x : ℤ | x > 3 ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l107_107210


namespace unique_solution_exists_l107_107475

theorem unique_solution_exists (k : ℝ) :
  (16 + 12 * k = 0) → ∃! x : ℝ, k * x^2 - 4 * x - 3 = 0 :=
by
  intro hk
  sorry

end unique_solution_exists_l107_107475


namespace cyclist_time_to_climb_and_descend_hill_l107_107971

noncomputable def hill_length : ℝ := 400 -- hill length in meters
noncomputable def ascent_speed_kmh : ℝ := 7.2 -- ascent speed in km/h
noncomputable def ascent_speed_ms : ℝ := ascent_speed_kmh * 1000 / 3600 -- ascent speed converted in m/s
noncomputable def descent_speed_ms : ℝ := 2 * ascent_speed_ms -- descent speed in m/s

noncomputable def time_to_climb : ℝ := hill_length / ascent_speed_ms -- time to climb in seconds
noncomputable def time_to_descend : ℝ := hill_length / descent_speed_ms -- time to descend in seconds
noncomputable def total_time : ℝ := time_to_climb + time_to_descend -- total time in seconds

theorem cyclist_time_to_climb_and_descend_hill : total_time = 300 :=
by
  sorry

end cyclist_time_to_climb_and_descend_hill_l107_107971


namespace binary_arithmetic_correct_l107_107471

def bin_add_sub_addition : Prop :=
  let b1 := 0b1101 in
  let b2 := 0b0111 in
  let b3 := 0b1010 in
  let b4 := 0b1001 in
  b1 + b2 - b3 + b4 = 0b10001

theorem binary_arithmetic_correct : bin_add_sub_addition := by 
  sorry

end binary_arithmetic_correct_l107_107471


namespace sum_roots_of_quadratic_eq_l107_107753

theorem sum_roots_of_quadratic_eq (a b c: ℝ) (x: ℝ) :
    (a = 1) →
    (b = -7) →
    (c = -9) →
    (x ^ 2 - 7 * x + 2 = 11) →
    (∃ r1 r2 : ℝ, x ^ 2 - 7 * x - 9 = 0 ∧ r1 + r2 = 7) :=
by
  sorry

end sum_roots_of_quadratic_eq_l107_107753


namespace proper_subsets_count_l107_107017

open Finset

theorem proper_subsets_count (B : Finset ℕ) (h : B = {2, 3, 4}) : B.card = 3 → (2 ^ B.card - 1) = 7 :=
by
  intro hB
  rw [h] at hB
  simp at hB
  sorry

end proper_subsets_count_l107_107017


namespace crates_of_oranges_l107_107590

theorem crates_of_oranges (C : ℕ) (h1 : ∀ crate, crate = 150) (h2 : ∀ box, box = 30) (num_boxes : ℕ) (total_fruits : ℕ) : 
  num_boxes = 16 → total_fruits = 2280 → 150 * C + 16 * 30 = 2280 → C = 12 :=
by
  intros num_boxes_eq total_fruits_eq fruit_eq
  sorry

end crates_of_oranges_l107_107590


namespace line_circle_no_intersection_l107_107871

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l107_107871


namespace canvas_bag_lower_carbon_solution_l107_107265

theorem canvas_bag_lower_carbon_solution :
  let canvas_release_oz := 9600
  let plastic_per_trip_oz := 32
  canvas_release_oz / plastic_per_trip_oz = 300 :=
by
  sorry

end canvas_bag_lower_carbon_solution_l107_107265


namespace no_intersection_l107_107889

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l107_107889


namespace percentage_of_fruits_in_good_condition_l107_107774

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

end percentage_of_fruits_in_good_condition_l107_107774


namespace cubic_roots_fraction_l107_107164

theorem cubic_roots_fraction 
  (a b c d : ℝ)
  (h_eq : ∀ x: ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ (x = -2 ∨ x = 3 ∨ x = 4)) :
  c / d = -1 / 12 :=
by
  sorry

end cubic_roots_fraction_l107_107164


namespace gcd_polynomial_multiple_l107_107496

theorem gcd_polynomial_multiple (b : ℕ) (hb : 620 ∣ b) : gcd (4 * b^3 + 2 * b^2 + 5 * b + 93) b = 93 := by
  sorry

end gcd_polynomial_multiple_l107_107496


namespace r_sq_plus_s_sq_l107_107705

variable {r s : ℝ}

theorem r_sq_plus_s_sq (h1 : r * s = 16) (h2 : r + s = 10) : r^2 + s^2 = 68 := 
by
  sorry

end r_sq_plus_s_sq_l107_107705


namespace alloy_parts_separation_l107_107977

theorem alloy_parts_separation {p q x : ℝ} (h0 : p ≠ q)
  (h1 : 6 * p ≠ 16 * q)
  (h2 : 6 * x * p + 2 * (8 - 2 * x) * q = 8 * (8 - x) * p + 6 * x * q) :
  x = 2.4 :=
by
  sorry

end alloy_parts_separation_l107_107977


namespace benny_seashells_l107_107462

-- Define the initial number of seashells Benny found
def seashells_found : ℝ := 66.5

-- Define the percentage of seashells Benny gave away
def percentage_given_away : ℝ := 0.75

-- Calculate the number of seashells Benny gave away
def seashells_given_away : ℝ := percentage_given_away * seashells_found

-- Calculate the number of seashells Benny now has
def seashells_left : ℝ := seashells_found - seashells_given_away

-- Prove that Benny now has 16.625 seashells
theorem benny_seashells : seashells_left = 16.625 :=
by
  sorry

end benny_seashells_l107_107462


namespace line_circle_no_intersection_l107_107879

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l107_107879


namespace power_inequality_l107_107063

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a ^ (3 / 4) + b ^ (3 / 4) + c ^ (3 / 4) > (a + b + c) ^ (3 / 4) :=
sorry

end power_inequality_l107_107063


namespace count_valid_four_digit_numbers_l107_107841

def four_digit_numbers_count : Nat :=
  let valid_a := [3, 4]
  let valid_d := [0, 5]
  let valid_pairs_bc := [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
  valid_a.length * valid_d.length * valid_pairs_bc.length

theorem count_valid_four_digit_numbers :
  four_digit_numbers_count = 24 :=
by
  -- To be proved
  sorry

end count_valid_four_digit_numbers_l107_107841


namespace drum_capacity_ratio_l107_107171

variable {C_X C_Y : ℝ}

theorem drum_capacity_ratio (h1 : C_X / 2 + C_Y / 2 = 3 * C_Y / 4) : C_Y / C_X = 2 :=
by
  have h2: C_X / 2 = C_Y / 4 := by
    sorry
  have h3: C_X = C_Y / 2 := by
    sorry
  rw [h3]
  have h4: C_Y / (C_Y / 2) = 2 := by
    sorry
  exact h4

end drum_capacity_ratio_l107_107171


namespace simplify_expression_l107_107389

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x - 2) / (x ^ 2 - 1) / (1 - 1 / (x - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_expression_l107_107389


namespace simplify_expression_l107_107576

theorem simplify_expression : 5 * (14 / 3) * (21 / -70) = - 35 / 2 := by
  sorry

end simplify_expression_l107_107576


namespace correct_proposition_l107_107498

-- Definitions
def p (x : ℝ) : Prop := x > 2 → x > 1 ∧ ¬ (x > 1 → x > 2)

def q (a b : ℝ) : Prop := a > b → 1 / a < 1 / b

-- Propositions
def p_and_q (x a b : ℝ) := p x ∧ q a b
def not_p_or_q (x a b : ℝ) := ¬ (p x) ∨ q a b
def p_and_not_q (x a b : ℝ) := p x ∧ ¬ (q a b)
def not_p_and_not_q (x a b : ℝ) := ¬ (p x) ∧ ¬ (q a b)

-- Main theorem
theorem correct_proposition (x a b : ℝ) (h_p : p x) (h_q : ¬ (q a b)) :
  (p_and_q x a b = false) ∧
  (not_p_or_q x a b = false) ∧
  (p_and_not_q x a b = true) ∧
  (not_p_and_not_q x a b = false) :=
by
  sorry

end correct_proposition_l107_107498


namespace log5_x_equals_neg_two_log5_2_l107_107363

theorem log5_x_equals_neg_two_log5_2 (x : ℝ) (h : x = (Real.log 3 / Real.log 9) ^ (Real.log 9 / Real.log 3)) :
  Real.log x / Real.log 5 = -2 * (Real.log 2 / Real.log 5) :=
by
  sorry

end log5_x_equals_neg_two_log5_2_l107_107363


namespace birds_in_nests_l107_107278

theorem birds_in_nests (birds : Fin 6 → Fin 6) (move : ∀ n : Fin 6, birds n = (birds n + 1) % 6 ∨ birds n = (birds n - 1) % 6) :
  ¬ ∃ n : Fin 6, ∀ m : Fin 6, birds m = n := 
sorry

end birds_in_nests_l107_107278


namespace midpoint_polar_coordinates_l107_107374

noncomputable def polar_midpoint :=
  let A := (10, 7 * Real.pi / 6)
  let B := (10, 11 * Real.pi / 6)
  let A_cartesian := (10 * Real.cos (7 * Real.pi / 6), 10 * Real.sin (7 * Real.pi / 6))
  let B_cartesian := (10 * Real.cos (11 * Real.pi / 6), 10 * Real.sin (11 * Real.pi / 6))
  let midpoint_cartesian := ((A_cartesian.1 + B_cartesian.1) / 2, (A_cartesian.2 + B_cartesian.2) / 2)
  let r := Real.sqrt (midpoint_cartesian.1 ^ 2 + midpoint_cartesian.2 ^ 2)
  let θ := if midpoint_cartesian.1 = 0 then 0 else Real.arctan (midpoint_cartesian.2 / midpoint_cartesian.1)
  (r, θ)

theorem midpoint_polar_coordinates :
  polar_midpoint = (5 * Real.sqrt 3, Real.pi) := by
  sorry

end midpoint_polar_coordinates_l107_107374


namespace greatest_integer_less_than_N_div_100_l107_107660

theorem greatest_integer_less_than_N_div_100 
    (N : ℕ)
    (h : 1/(2!*17!) + 1/(3!*16!) + 1/(4!*15!) + 1/(5!*14!) + 1/(6!*13!) + 1/(7!*12!) + 1/(8!*11!) + 1/(9!*10!) = N / (1!*18!)) :
    (⌊ N / 100 ⌋ : ℕ) = 137 := 
sorry

end greatest_integer_less_than_N_div_100_l107_107660


namespace smallest_three_digit_number_satisfying_conditions_l107_107782

theorem smallest_three_digit_number_satisfying_conditions :
  ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n + 6) % 9 = 0 ∧ (n - 4) % 6 = 0 ∧ n = 112 :=
by
  -- Proof goes here
  sorry

end smallest_three_digit_number_satisfying_conditions_l107_107782


namespace exists_k_l107_107461

-- Definitions of the conditions
def sequence_def (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a (n+1) = Nat.lcm (a n) (a (n-1)) - Nat.lcm (a (n-1)) (a (n-2))

theorem exists_k (a : ℕ → ℕ) (a₁ a₂ a₃ : ℕ) (h₁ : a 1 = a₁) (h₂ : a 2 = a₂) (h₃ : a 3 = a₃)
  (h_seq : sequence_def a) : ∃ k : ℕ, k ≤ a₃ + 4 ∧ a k = 0 := 
sorry

end exists_k_l107_107461


namespace chocolate_bars_in_small_box_l107_107318

-- Given conditions
def num_small_boxes : ℕ := 21
def total_chocolate_bars : ℕ := 525

-- Statement to prove
theorem chocolate_bars_in_small_box : total_chocolate_bars / num_small_boxes = 25 := by
  sorry

end chocolate_bars_in_small_box_l107_107318


namespace gcd_bezout_663_182_l107_107646

theorem gcd_bezout_663_182 :
  let a := 182
  let b := 663
  ∃ d u v : ℤ, d = Int.gcd a b ∧ d = a * u + b * v ∧ d = 13 ∧ u = 11 ∧ v = -3 :=
by 
  let a := 182
  let b := 663
  use 13, 11, -3
  sorry

end gcd_bezout_663_182_l107_107646


namespace probability_adjacent_vertices_of_decagon_l107_107101

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l107_107101


namespace susan_vacation_pay_missed_l107_107730

noncomputable def susan_weekly_pay (hours_worked : ℕ) : ℕ :=
  let regular_hours := min 40 hours_worked
  let overtime_hours := max (hours_worked - 40) 0
  15 * regular_hours + 20 * overtime_hours

noncomputable def susan_sunday_pay (num_sundays : ℕ) (hours_per_sunday : ℕ) : ℕ :=
  25 * num_sundays * hours_per_sunday

noncomputable def pay_without_sundays : ℕ :=
  susan_weekly_pay 48
    
noncomputable def total_three_week_pay : ℕ :=
  let weeks_normal_pay := 3 * pay_without_sundays
  let sunday_hours_1 := 1 * 8
  let sunday_hours_2 := 2 * 8
  let sunday_hours_3 := 0 * 8
  let sundays_total_pay := susan_sunday_pay 1 8 + susan_sunday_pay 2 8 + susan_sunday_pay 0 8
  weeks_normal_pay + sundays_total_pay
  
noncomputable def paid_vacation_pay : ℕ :=
  let paid_days := 6
  let paid_weeks_pay := susan_weekly_pay 40 + susan_weekly_pay (paid_days % 5 * 8)
  paid_weeks_pay

theorem susan_vacation_pay_missed :
  let missed_pay := total_three_week_pay - paid_vacation_pay
  missed_pay = 2160 := sorry

end susan_vacation_pay_missed_l107_107730


namespace line_circle_no_intersection_l107_107861

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l107_107861


namespace necessary_condition_for_x_gt_5_l107_107770

theorem necessary_condition_for_x_gt_5 (x : ℝ) : x > 5 → x > 3 :=
by
  intros h
  exact lt_trans (show 3 < 5 from by linarith) h

end necessary_condition_for_x_gt_5_l107_107770


namespace line_circle_no_intersect_l107_107854

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l107_107854


namespace valentines_given_l107_107556

theorem valentines_given (original current given : ℕ) (h1 : original = 58) (h2 : current = 16) (h3 : given = original - current) : given = 42 := by
  sorry

end valentines_given_l107_107556


namespace grass_knot_segments_butterfly_knot_segments_l107_107409

-- Definitions for the grass knot problem
def outer_loops_cut : Nat := 5
def segments_after_outer_loops_cut : Nat := 6

-- Theorem for the grass knot
theorem grass_knot_segments (n : Nat) (h : n = outer_loops_cut) : (n + 1 = segments_after_outer_loops_cut) :=
sorry

-- Definitions for the butterfly knot problem
def butterfly_wings_loops_per_wing : Nat := 7
def segments_after_butterfly_wings_cut : Nat := 15

-- Theorem for the butterfly knot
theorem butterfly_knot_segments (w : Nat) (h : w = butterfly_wings_loops_per_wing) : ((w * 2 * 2 + 2) / 2 = segments_after_butterfly_wings_cut) :=
sorry

end grass_knot_segments_butterfly_knot_segments_l107_107409


namespace no_100_roads_l107_107466

theorem no_100_roads (k : ℕ) (hk : 3 * k % 2 = 0) : 100 ≠ 3 * k / 2 := 
by
  sorry

end no_100_roads_l107_107466


namespace bike_cost_l107_107726

theorem bike_cost (price_per_apple repairs_share remaining_share apples_sold earnings repairs_cost bike_cost : ℝ) :
  price_per_apple = 1.25 →
  repairs_share = 0.25 →
  remaining_share = 1/5 →
  apples_sold = 20 →
  earnings = apples_sold * price_per_apple →
  repairs_cost = earnings * 4/5 →
  repairs_cost = bike_cost * repairs_share →
  bike_cost = 80 :=
by
  intros;
  sorry

end bike_cost_l107_107726


namespace find_coordinates_l107_107827

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -3⟩

def satisfiesCondition (A B P : Point) : Prop :=
  2 * (P.x - A.x) = (B.x - P.x) ∧ 2 * (P.y - A.y) = (B.y - P.y)

theorem find_coordinates (P : Point) (h : satisfiesCondition A B P) : 
  P = ⟨6, -9⟩ :=
  sorry

end find_coordinates_l107_107827


namespace cuboid_length_l107_107816

theorem cuboid_length (b h : ℝ) (A : ℝ) (l : ℝ) : b = 6 → h = 5 → A = 120 → 2 * (l * b + b * h + h * l) = A → l = 30 / 11 :=
by
  intros hb hh hA hSurfaceArea
  rw [hb, hh] at hSurfaceArea
  sorry

end cuboid_length_l107_107816


namespace leibo_orange_price_l107_107921

variable (x y m : ℝ)

theorem leibo_orange_price :
  (3 * x + 2 * y = 78) ∧ (2 * x + 3 * y = 72) ∧ (18 * m + 12 * (100 - m) ≤ 1440) → (x = 18) ∧ (y = 12) ∧ (m ≤ 40) :=
by
  intros h
  sorry

end leibo_orange_price_l107_107921


namespace Maurice_current_age_l107_107329

variable (Ron_now Maurice_now : ℕ)

theorem Maurice_current_age
  (h1 : Ron_now = 43)
  (h2 : ∀ t Ron_future Maurice_future : ℕ, Ron_future = 4 * Maurice_future → Ron_future = Ron_now + 5 → Maurice_future = Maurice_now + 5) :
  Maurice_now = 7 := 
sorry

end Maurice_current_age_l107_107329


namespace salary_problem_l107_107288

theorem salary_problem
  (A B : ℝ)
  (h1 : A + B = 3000)
  (h2 : 0.05 * A = 0.15 * B) :
  A = 2250 :=
sorry

end salary_problem_l107_107288


namespace units_digit_x_pow_75_plus_6_eq_9_l107_107957

theorem units_digit_x_pow_75_plus_6_eq_9 (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9)
  (h3 : (x ^ 75 + 6) % 10 = 9) : x = 3 :=
sorry

end units_digit_x_pow_75_plus_6_eq_9_l107_107957


namespace determine_positions_l107_107571

-- Defining the participants
inductive Participant
| Olya
| Oleg
| Pasha

open Participant

-- Defining the possible places
inductive Place
| First
| Second
| Third

open Place

-- Define the conditions
def condition1 (pos : Participant → Place) : Prop := 
  pos Olya = First ∨ pos Oleg = First ∨ pos Pasha = First

def condition2 (pos : Participant → Place) : Prop :=
  (pos Olya = First ∧ pos Olya = Second ∧ pos Olya = Third) ∨
  (pos Oleg = First ∧ pos Oleg = Second ∧ pos Oleg = Third) ∨
  (pos Pasha = First ∧ pos Pasha = Second ∧ pos Pasha = Third)

def condition3 (pos : Participant → Place) : Prop :=
  ∀ p, pos p ≠ First ∧ pos p ≠ Second ∧ pos p ≠ Third

def condition4 (pos : Participant → Place) : Prop :=
  (pos Olya = First → (pos Oleg = First ∨ pos Pasha = First)) ∧
  (pos Oleg = First → pos Olya ≠ First) ∧
  (pos Pasha = First → (pos Oleg = First ∨ pos Olya = First))

def always_true_or_false : Prop :=
  (∀ p, p = Olya ∨ p = Oleg ∨ p = Pasha )

-- Main theorem
theorem determine_positions (pos : Participant → Place) :
  condition1 pos ∧ condition2 pos ∧ condition3 pos ∧ condition4 pos ∧ always_true_or_false →
  pos Oleg = First ∧ pos Pasha = Second ∧ pos Olya = Third := 
by
  sorry

end determine_positions_l107_107571


namespace line_circle_no_intersection_l107_107875

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l107_107875


namespace quadratic_coeff_sum_l107_107743

theorem quadratic_coeff_sum {a b c : ℝ} (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 1) * (x - 5))
    (h2 : a * 3^2 + b * 3 + c = 36) : a + b + c = 0 :=
by
  sorry

end quadratic_coeff_sum_l107_107743


namespace no_x_intersect_one_x_intersect_l107_107359

variable (m : ℝ)

-- Define the original quadratic function
def quadratic_function (x : ℝ) := x^2 - 2 * m * x + m^2 + 3

-- 1. Prove the function does not intersect the x-axis
theorem no_x_intersect : ∀ m, ∀ x : ℝ, quadratic_function m x ≠ 0 := by
  intros
  unfold quadratic_function
  sorry

-- 2. Prove that translating down by 3 units intersects the x-axis at one point
def translated_quadratic (x : ℝ) := (x - m)^2

theorem one_x_intersect : ∃ x : ℝ, translated_quadratic m x = 0 := by
  unfold translated_quadratic
  sorry

end no_x_intersect_one_x_intersect_l107_107359


namespace quadratic_complete_square_l107_107399

theorem quadratic_complete_square : 
  ∃ d e : ℝ, ((x^2 - 16*x + 15) = ((x + d)^2 + e)) ∧ (d + e = -57) := by
  sorry

end quadratic_complete_square_l107_107399


namespace rectangle_area_l107_107583

theorem rectangle_area (b : ℕ) (l : ℕ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 48) : l * b = 108 := 
by
  sorry

end rectangle_area_l107_107583


namespace sector_area_eq_three_halves_l107_107365

theorem sector_area_eq_three_halves (θ R S : ℝ) (hθ : θ = 3) (h₁ : 2 * R + θ * R = 5) :
  S = 3 / 2 :=
by
  sorry

end sector_area_eq_three_halves_l107_107365


namespace solve_for_y_l107_107064

theorem solve_for_y (y : ℕ) (h : 2^y + 8 = 4 * 2^y - 40) : y = 4 :=
by
  sorry

end solve_for_y_l107_107064


namespace impossible_to_form_16_unique_remainders_with_3_digits_l107_107377

theorem impossible_to_form_16_unique_remainders_with_3_digits :
  ¬∃ (digits : Finset ℕ) (num_fun : Fin 16 → ℕ), digits.card = 3 ∧ 
  ∀ i j : Fin 16, i ≠ j → num_fun i % 16 ≠ num_fun j % 16 ∧ 
  ∀ n : ℕ, n ∈ (digits : Set ℕ) → 100 ≤ num_fun i ∧ num_fun i < 1000 :=
sorry

end impossible_to_form_16_unique_remainders_with_3_digits_l107_107377


namespace pen_price_l107_107981

theorem pen_price (p : ℝ) (h : 30 = 10 * p + 10 * (p / 2)) : p = 2 :=
sorry

end pen_price_l107_107981


namespace average_temp_addington_l107_107621

def temperatures : List ℚ := [60, 59, 56, 53, 49, 48, 46]

def average_temp (temps : List ℚ) : ℚ := (temps.sum) / temps.length

theorem average_temp_addington :
  average_temp temperatures = 53 := by
  sorry

end average_temp_addington_l107_107621


namespace decagon_adjacent_probability_l107_107129

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l107_107129


namespace hyperbola_vertex_distance_l107_107820

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), (x^2 / 16 - y^2 / 9 = 1) → (vertex_distance : ℝ := 8) := sorry

end hyperbola_vertex_distance_l107_107820


namespace christopher_strolled_distance_l107_107163

variable speed : ℝ := 4
variable time : ℝ := 1.25

theorem christopher_strolled_distance : speed * time = 5 := by
  sorry

end christopher_strolled_distance_l107_107163


namespace corn_purchase_l107_107626

theorem corn_purchase : ∃ c b : ℝ, c + b = 30 ∧ 89 * c + 55 * b = 2170 ∧ c = 15.3 := 
by
  sorry

end corn_purchase_l107_107626


namespace large_cube_surface_area_l107_107086

-- Define given conditions
def small_cube_volume := 512 -- volume in cm^3
def num_small_cubes := 8

-- Define side length of small cube
def small_cube_side_length := (small_cube_volume : ℝ)^(1/3)

-- Define side length of large cube
def large_cube_side_length := 2 * small_cube_side_length

-- Surface area formula for a cube
def surface_area (side_length : ℝ) := 6 * side_length^2

-- Theorem: The surface area of the large cube is 1536 cm^2
theorem large_cube_surface_area :
  surface_area large_cube_side_length = 1536 :=
sorry

end large_cube_surface_area_l107_107086


namespace coast_guard_overtake_smuggler_l107_107610

noncomputable def time_of_overtake (initial_distance : ℝ) (initial_time : ℝ) 
                                   (smuggler_speed1 coast_guard_speed : ℝ) 
                                   (duration1 new_smuggler_speed : ℝ) : ℝ :=
  let distance_after_duration1 := initial_distance + (smuggler_speed1 * duration1) - (coast_guard_speed * duration1)
  let relative_speed_new := coast_guard_speed - new_smuggler_speed
  duration1 + (distance_after_duration1 / relative_speed_new)

theorem coast_guard_overtake_smuggler : 
  time_of_overtake 15 0 18 20 1 16 = 4.25 := by
  sorry

end coast_guard_overtake_smuggler_l107_107610


namespace expand_product_l107_107635

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 :=
by
  sorry

end expand_product_l107_107635


namespace range_of_k_no_third_quadrant_l107_107236

theorem range_of_k_no_third_quadrant (k : ℝ) : ¬(∃ x : ℝ, ∃ y : ℝ, x < 0 ∧ y < 0 ∧ y = k * x + 3) → k ≤ 0 := 
sorry

end range_of_k_no_third_quadrant_l107_107236


namespace number_of_players_l107_107040

/-- Jane bought 600 minnows, each prize has 3 minnows, 15% of the players win a prize, 
and 240 minnows are left over. To find the total number of players -/
theorem number_of_players (total_minnows left_over_minnows minnows_per_prize prizes_win_percent : ℕ) 
(h1 : total_minnows = 600) 
(h2 : minnows_per_prize = 3)
(h3 : prizes_win_percent * 100 = 15)
(h4 : left_over_minnows = 240) : 
total_minnows - left_over_minnows = 360 → 
  360 / minnows_per_prize = 120 → 
  (prizes_win_percent * 100 / 100) * P = 120 → 
  P = 800 := 
by 
  sorry

end number_of_players_l107_107040


namespace probability_adjacent_vertices_of_decagon_l107_107099

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l107_107099


namespace maximize_product_minimize_product_l107_107477

-- Define lists of the digits to be used
def digits : List ℕ := [2, 4, 6, 8]

-- Function to calculate the number from a list of digits
def toNumber (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * 10 + d) 0

-- Function to calculate the product given two numbers represented as lists of digits
def product (digits1 digits2 : List ℕ) : ℕ :=
  toNumber digits1 * toNumber digits2

-- Definitions of specific permutations to be used
def maxDigits1 : List ℕ := [8, 6, 4]
def maxDigit2 : List ℕ := [2]
def minDigits1 : List ℕ := [2, 4, 6]
def minDigit2 : List ℕ := [8]

-- Theorem statements
theorem maximize_product : product maxDigits1 maxDigit2 = 864 * 2 := by
  sorry

theorem minimize_product : product minDigits1 minDigit2 = 246 * 8 := by
  sorry

end maximize_product_minimize_product_l107_107477


namespace rectangle_constant_k_l107_107286

theorem rectangle_constant_k (d : ℝ) (x : ℝ) (h_ratio : 4 * x = (4 / 3) * (3 * x)) (h_diagonal : d^2 = (4 * x)^2 + (3 * x)^2) : 
  ∃ k : ℝ, k = 12 / 25 ∧ (4 * x) * (3 * x) = k * d^2 := 
sorry

end rectangle_constant_k_l107_107286


namespace president_savings_l107_107390

theorem president_savings (total_funds : ℕ) (friends_percentage : ℕ) (family_percentage : ℕ) 
  (friends_contradiction funds_left family_contribution fundraising_amount : ℕ) :
  total_funds = 10000 →
  friends_percentage = 40 →
  family_percentage = 30 →
  friends_contradiction = (total_funds * friends_percentage) / 100 →
  funds_left = total_funds - friends_contradiction →
  family_contribution = (funds_left * family_percentage) / 100 →
  fundraising_amount = funds_left - family_contribution →
  fundraising_amount = 4200 :=
by
  intros
  sorry

end president_savings_l107_107390


namespace solution_of_fractional_inequality_l107_107401

noncomputable def solution_set_of_inequality : Set ℝ :=
  {x : ℝ | -3 < x ∨ x > 1/2 }

theorem solution_of_fractional_inequality :
  {x : ℝ | (2 * x - 1) / (x + 3) > 0} = solution_set_of_inequality :=
by
  sorry

end solution_of_fractional_inequality_l107_107401


namespace expand_product_l107_107633

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 :=
by
  sorry

end expand_product_l107_107633


namespace simplify_expression_l107_107728

theorem simplify_expression (m n : ℝ) (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3^(m * n / (m - n))) : 
  ((
    (x^(2 / m) - 9 * x^(2 / n)) *
    ((x^(1 - m))^(1 / m) - 3 * (x^(1 - n))^(1 / n))
  ) / (
    (x^(1 / m) + 3 * x^(1 / n))^2 - 12 * x^((m + n) / (m * n))
  ) = (x^(1 / m) + 3 * x^(1 / n)) / x) := 
sorry

end simplify_expression_l107_107728


namespace stone_travel_distance_l107_107940

/-- Define the radii --/
def radius_fountain := 15
def radius_stone := 3

/-- Prove the distance the stone needs to travel along the fountain's edge --/
theorem stone_travel_distance :
  let circumference_fountain := 2 * Real.pi * ↑radius_fountain
  let circumference_stone := 2 * Real.pi * ↑radius_stone
  let distance_traveled := circumference_stone
  distance_traveled = 6 * Real.pi := by
  -- Placeholder for proof, based on conditions given
  sorry

end stone_travel_distance_l107_107940


namespace quadratic_transformation_l107_107584

theorem quadratic_transformation (x d e : ℝ) (h : x^2 - 24*x + 45 = (x+d)^2 + e) : d + e = -111 :=
sorry

end quadratic_transformation_l107_107584


namespace pythagorean_numbers_b_l107_107931

-- Define Pythagorean numbers and conditions
variable (a b c m : ℕ)
variable (h1 : a = 1/2 * m^2 - 1/2)
variable (h2 : c = 1/2 * m^2 + 1/2)
variable (h3 : m > 1 ∧ ¬ even m)

theorem pythagorean_numbers_b (h4 : c^2 = a^2 + b^2) : b = m :=
sorry

end pythagorean_numbers_b_l107_107931


namespace value_of_expr_l107_107224

theorem value_of_expr (a b c d : ℝ) (h1 : a = 3 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) : 
  a * c / (b * d) = 15 := 
by
  sorry

end value_of_expr_l107_107224


namespace solve_system_eqs_l107_107280
noncomputable section

theorem solve_system_eqs (x y z : ℝ) :
  (x * y = 5 * (x + y) ∧ x * z = 4 * (x + z) ∧ y * z = 2 * (y + z))
  ↔ (x = 0 ∧ y = 0 ∧ z = 0)
  ∨ (x = -40 ∧ y = 40 / 9 ∧ z = 40 / 11) := sorry

end solve_system_eqs_l107_107280


namespace exists_m_n_for_any_d_l107_107725

theorem exists_m_n_for_any_d (d : ℤ) : ∃ m n : ℤ, d = (n - 2 * m + 1) / (m^2 - n) :=
by
  sorry

end exists_m_n_for_any_d_l107_107725


namespace intersection_trace_l107_107695

-- Define the given conditions
variable (A B C D M : Point)
variable (a c k : ℝ)
variable (AB AC BD CD : Line)

-- Hypotheses
hypothesis h1 : ConvexTrapezoid A B C D
hypothesis h2 : length AB = a
hypothesis h3 : length CD = c
hypothesis h4 : c < a
hypothesis h5 : perimeter A B C D = k

-- Define the point of intersection of the extensions of the non-parallel sides
noncomputable def Intersection_M (A B C D : Point) : Point := sorry -- Definitions for point intersection based on geometry

-- State the problem
theorem intersection_trace (A B C D : Point) (AB AC BD CD : Line)
  (h1 : ConvexTrapezoid A B C D) (h2 : length AB = a)
  (h3 : length CD = c) (h4 : c < a) (h5 : perimeter A B C D = k) :
  ∃ (E : Ellipse), (Intersection_M A B C D) ∈ E := 
sorry

end intersection_trace_l107_107695


namespace decagon_adjacent_probability_l107_107113

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l107_107113


namespace compound_interest_l107_107239

theorem compound_interest (P R T : ℝ) (SI CI : ℝ)
  (hSI : SI = P * R * T / 100)
  (h_given_SI : SI = 50)
  (h_given_R : R = 5)
  (h_given_T : T = 2)
  (h_compound_interest : CI = P * ((1 + R / 100)^T - 1)) :
  CI = 51.25 :=
by
  -- Since we are only required to state the theorem, we add 'sorry' here.
  sorry

end compound_interest_l107_107239


namespace expected_value_of_boy_girl_pairs_l107_107936

noncomputable def expected_value_of_T (boys girls : ℕ) : ℚ :=
  24 * ((boys / 24) * (girls / 23) + (girls / 24) * (boys / 23))

theorem expected_value_of_boy_girl_pairs (boys girls : ℕ) (h_boys : boys = 10) (h_girls : girls = 14) :
  expected_value_of_T boys girls = 12 :=
by
  rw [h_boys, h_girls]
  norm_num
  sorry

end expected_value_of_boy_girl_pairs_l107_107936


namespace sum_of_reciprocals_eq_six_l107_107291

theorem sum_of_reciprocals_eq_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  (1 / x + 1 / y) = 6 :=
by
  sorry

end sum_of_reciprocals_eq_six_l107_107291


namespace trig_problem_part_1_trig_problem_part_2_l107_107353

open Real

-- Definitions from conditions
def equation_has_trig_roots (m : ℝ) (θ : ℝ) : Prop :=
  2 * sin θ ^ 2 - (sqrt 3 + 1) * sin θ + m = 0 ∧
  2 * cos θ ^ 2 - (sqrt 3 + 1) * cos θ + m = 0 ∧
  0 < θ ∧ θ < 2 * π

noncomputable def problem_1 (θ : ℝ) : ℝ :=
  (sin θ ^ 2 / (sin θ - cos θ)) + (cos θ ^ 2 / (cos θ - sin θ))

theorem trig_problem_part_1 (m : ℝ) (θ : ℝ) (h : equation_has_trig_roots m θ) :
  problem_1 θ = (sqrt 3 + 1) / 2 := sorry

theorem trig_problem_part_2 (m : ℝ) (θ : ℝ) (h : equation_has_trig_roots m θ) :
  m = sqrt 3 / 2 := sorry

end trig_problem_part_1_trig_problem_part_2_l107_107353


namespace bouquets_ratio_l107_107434

theorem bouquets_ratio (monday tuesday wednesday : ℕ) 
  (h1 : monday = 12) 
  (h2 : tuesday = 3 * monday) 
  (h3 : monday + tuesday + wednesday = 60) :
  wednesday / tuesday = 1 / 3 :=
by sorry

end bouquets_ratio_l107_107434


namespace part1_part2_l107_107509

theorem part1 (a b : ℝ) (h1 : ∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) (hb : b > 1) : a = 1 ∧ b = 2 :=
sorry

theorem part2 (k : ℝ) (x y : ℝ) (hx : x > 0) (hy : y > 0) (a b : ℝ) 
  (ha : a = 1) (hb : b = 2) 
  (h2 : a / x + b / y = 1)
  (h3 : 2 * x + y ≥ k^2 + k + 2) : -3 ≤ k ∧ k ≤ 2 :=
sorry

end part1_part2_l107_107509


namespace triangle_possible_sides_l107_107220

theorem triangle_possible_sides : 
  let unknown_side_lengths := {x : ℤ | 3 < x ∧ x < 13}
  in set.finite unknown_side_lengths ∧ set.card unknown_side_lengths = 9 :=
by
  sorry

end triangle_possible_sides_l107_107220


namespace dutch_americans_with_window_seats_l107_107053

theorem dutch_americans_with_window_seats :
  let total_people := 90
  let dutch_fraction := 3 / 5
  let dutch_american_fraction := 1 / 2
  let window_seat_fraction := 1 / 3
  let dutch_people := total_people * dutch_fraction
  let dutch_americans := dutch_people * dutch_american_fraction
  let dutch_americans_window_seats := dutch_americans * window_seat_fraction
  dutch_americans_window_seats = 9 := by
sorry

end dutch_americans_with_window_seats_l107_107053


namespace a_value_intersection_l107_107511

open Set

noncomputable def a_intersection_problem (a : ℝ) : Prop :=
  let A := { x : ℝ | x^2 < a^2 }
  let B := { x : ℝ | 1 < x ∧ x < 3 }
  let C := { x : ℝ | 1 < x ∧ x < 2 }
  A ∩ B = C → (a = 2 ∨ a = -2)

-- The theorem statement corresponding to the problem
theorem a_value_intersection (a : ℝ) :
  a_intersection_problem a :=
sorry

end a_value_intersection_l107_107511


namespace geometric_sequence_sum_l107_107376

-- Define the geometric sequence {a_n}
def a (n : ℕ) : ℕ := 2 * (1 ^ (n - 1))

-- Define the sum of the first n terms, s_n
def s (n : ℕ) : ℕ := (Finset.range n).sum (a)

-- The transformed sequence {a_n + 1} assumed also geometric
def b (n : ℕ) : ℕ := a n + 1

-- Lean theorem that s_n = 2n
theorem geometric_sequence_sum (n : ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, (b (n + 1)) * (b (n + 1)) = (b n * b (n + 2))) : 
  s n = 2 * n :=
sorry

end geometric_sequence_sum_l107_107376


namespace factor_by_resultant_l107_107438

theorem factor_by_resultant (x f : ℤ) (h1 : x = 17) (h2 : (2 * x + 5) * f = 117) : f = 3 := 
by
  sorry

end factor_by_resultant_l107_107438


namespace car_B_speed_is_50_l107_107159

def car_speeds (v_A v_B : ℕ) (d_init d_ahead t : ℝ) : Prop :=
  v_A * t = v_B * t + d_init + d_ahead

theorem car_B_speed_is_50 :
  car_speeds 58 50 10 8 2.25 :=
by
  sorry

end car_B_speed_is_50_l107_107159


namespace range_of_a_l107_107687

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ x1^3 - 3*x1 + a = 0 ∧ x2^3 - 3*x2 + a = 0 ∧ x3^3 - 3*x3 + a = 0) 
  ↔ -2 < a ∧ a < 2 :=
sorry

end range_of_a_l107_107687


namespace ellipse_major_axis_length_l107_107978

-- Given conditions
variable (radius : ℝ) (h_radius : radius = 2)
variable (minor_axis : ℝ) (h_minor_axis : minor_axis = 2 * radius)
variable (major_axis : ℝ) (h_major_axis : major_axis = 1.4 * minor_axis)

-- Proof problem statement
theorem ellipse_major_axis_length : major_axis = 5.6 :=
by
  sorry

end ellipse_major_axis_length_l107_107978


namespace number_of_months_in_martian_calendar_l107_107533

theorem number_of_months_in_martian_calendar
  (x y : ℕ) 
  (h1 : 100 * x + 77 * y = 5882) 
  (h2 : x + y = 74) :
  x + y = 74 := 
by
  sorry

end number_of_months_in_martian_calendar_l107_107533


namespace div_z_x_l107_107226

variables (x y z : ℚ)

theorem div_z_x (h1 : x / y = 3) (h2 : y / z = 5 / 2) : z / x = 2 / 15 :=
sorry

end div_z_x_l107_107226


namespace decagon_adjacent_probability_l107_107097

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l107_107097


namespace picked_tomatoes_eq_53_l107_107142

-- Definitions based on the conditions
def initial_tomatoes : ℕ := 177
def initial_potatoes : ℕ := 12
def items_left : ℕ := 136

-- Define what we need to prove
theorem picked_tomatoes_eq_53 : initial_tomatoes + initial_potatoes - items_left = 53 :=
by sorry

end picked_tomatoes_eq_53_l107_107142


namespace find_a_l107_107225

noncomputable def f (x a : ℝ) : ℝ := (2 * x + a) ^ 2

theorem find_a (a : ℝ) (h1 : f 2 a = 20) : a = 1 :=
sorry

end find_a_l107_107225


namespace roses_carnations_price_comparison_l107_107914

variables (x y : ℝ)

theorem roses_carnations_price_comparison
  (h1 : 6 * x + 3 * y > 24)
  (h2 : 4 * x + 5 * y < 22) :
  2 * x > 3 * y :=
sorry

end roses_carnations_price_comparison_l107_107914


namespace find_angle_C_l107_107912

open Real -- Opening Real to directly use real number functions and constants

noncomputable def triangle_angles_condition (A B C: ℝ) : Prop :=
  2 * sin A + 5 * cos B = 5 ∧ 5 * sin B + 2 * cos A = 2

-- Theorem statement
theorem find_angle_C (A B C: ℝ) (h: triangle_angles_condition A B C):
  C = arcsin (1 / 5) ∨ C = 180 - arcsin (1 / 5) :=
sorry

end find_angle_C_l107_107912


namespace parabola_y_intercepts_l107_107472

theorem parabola_y_intercepts : 
  (∃ y1 y2 : ℝ, 3 * y1^2 - 4 * y1 + 1 = 0 ∧ 3 * y2^2 - 4 * y2 + 1 = 0 ∧ y1 ≠ y2) :=
by
  sorry

end parabola_y_intercepts_l107_107472


namespace cupcakes_per_package_calculation_l107_107574

noncomputable def sarah_total_cupcakes := 38
noncomputable def cupcakes_eaten_by_todd := 14
noncomputable def number_of_packages := 3
noncomputable def remaining_cupcakes := sarah_total_cupcakes - cupcakes_eaten_by_todd
noncomputable def cupcakes_per_package := remaining_cupcakes / number_of_packages

theorem cupcakes_per_package_calculation : cupcakes_per_package = 8 := by
  sorry

end cupcakes_per_package_calculation_l107_107574


namespace recruits_count_l107_107949

def x := 50
def y := 100
def z := 170

theorem recruits_count :
  ∃ n : ℕ, n = 211 ∧ (∀ a b c : ℕ, (b = 4 * a ∨ a = 4 * c ∨ c = 4 * b) → (b + 100 = a + 150) ∨ (a + 50 = c + 150) ∨ (c + 170 = b + 100)) :=
sorry

end recruits_count_l107_107949


namespace hypotenuse_length_l107_107688

theorem hypotenuse_length (a b c : ℕ) (h1 : a = 12) (h2 : b = 5) (h3 : c^2 = a^2 + b^2) : c = 13 := by
  sorry

end hypotenuse_length_l107_107688


namespace hexagon_circle_radius_l107_107315

noncomputable def hexagon_radius (sides : List ℝ) (probability : ℝ) : ℝ :=
  let total_angle := 360.0
  let visible_angle := probability * total_angle
  let side_length_average := (sides.sum / sides.length : ℝ)
  let theta := (visible_angle / 6 : ℝ) -- assuming θ approximately splits equally among 6 gaps
  side_length_average / Real.sin (theta / 2 * Real.pi / 180.0)

theorem hexagon_circle_radius :
  hexagon_radius [3, 2, 4, 3, 2, 4] (1 / 3) = 17.28 :=
by
  sorry

end hexagon_circle_radius_l107_107315


namespace num_possible_triangle_sides_l107_107207

theorem num_possible_triangle_sides (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (a_eq_8 : a = 8) (b_eq_5 : b = 5) : 
  let x := {n : ℕ | 3 < n ∧ n < 13} in
  set.card x = 9 :=
by
  sorry

end num_possible_triangle_sides_l107_107207


namespace main_theorem_l107_107314

noncomputable def circle_center : Prop :=
  ∃ x y : ℝ, 2*x - y - 7 = 0 ∧ y = -3 ∧ x = 2

noncomputable def circle_equation : Prop :=
  (∀ (x y : ℝ), (x - 2)^2 + (y + 3)^2 = 5)

noncomputable def tangent_condition (k : ℝ) : Prop :=
  (3 + 3*k)^2 / (1 + k^2) = 5

noncomputable def symmetric_circle_center : Prop :=
  ∃ x y : ℝ, x = -22/5 ∧ y = 1/5

noncomputable def symmetric_circle_equation : Prop :=
  (∀ (x y : ℝ), (x + 22/5)^2 + (y - 1/5)^2 = 5)

theorem main_theorem : circle_center → circle_equation ∧ (∃ k : ℝ, tangent_condition k) ∧ symmetric_circle_center → symmetric_circle_equation :=
  by sorry

end main_theorem_l107_107314


namespace probability_adjacent_vertices_in_decagon_l107_107107

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l107_107107


namespace intersection_A_complement_B_l107_107553

def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B : Set ℝ := { y | 0 ≤ y }

theorem intersection_A_complement_B : A ∩ -B = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end intersection_A_complement_B_l107_107553


namespace days_c_worked_l107_107420

theorem days_c_worked (Da Db Dc : ℕ) (Wa Wb Wc : ℕ)
  (h1 : Da = 6) (h2 : Db = 9) (h3 : Wc = 100) (h4 : 3 * Wc = 5 * Wa)
  (h5 : 4 * Wc = 5 * Wb)
  (h6 : Wa * Da + Wb * Db + Wc * Dc = 1480) : Dc = 4 :=
by
  sorry

end days_c_worked_l107_107420


namespace no_real_intersections_l107_107843

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l107_107843


namespace average_rst_l107_107020

theorem average_rst (r s t : ℝ) (h : (5 / 4) * (r + s + t - 2) = 15) : (r + s + t) / 3 = 14 / 3 :=
by
  sorry

end average_rst_l107_107020


namespace stability_comparison_probability_calculation_l107_107585

-- Define the scores for class A and B
def scoresA : List ℕ := [5, 8, 9, 9, 9]
def scoresB : List ℕ := [6, 7, 8, 9, 10]

-- Define helper functions to calculate mean and variance for a list
def mean (scores : List ℕ) : ℝ :=
  (scores.foldl (λ acc x => acc + x) 0 : ℝ) / scores.length

def variance (scores : List ℕ) : ℝ :=
  let μ := mean scores
  (scores.foldl (λ acc x => acc + (x : ℝ - μ) ^ 2) 0) / scores.length

def populationMeanB := mean scoresB
def populationVarianceA := variance scoresA
def populationVarianceB := variance scoresB

-- Define a function to calculate the absolute difference between the sample mean and population mean
def sampleMeanDifference (sample : List ℕ) : ℝ :=
  abs (mean sample - populationMeanB)

-- Define the possible samples of size 2 from scoresB
def samplesB : List (List ℕ) := (scoresB.combinations 2).map id

-- Calculate the probability that the absolute difference is not less than 1
def probabilityOfDifferenceNotLessThan1 : ℝ :=
  let satisfyingSamples := samplesB.filter (λ sample => sampleMeanDifference sample ≥ 1)
  satisfyingSamples.length / samplesB.length

theorem stability_comparison : populationVarianceB < populationVarianceA := by
  sorry

theorem probability_calculation : probabilityOfDifferenceNotLessThan1 = 2 / 5 := by
  sorry

end stability_comparison_probability_calculation_l107_107585


namespace decagon_adjacent_probability_l107_107096

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l107_107096


namespace competition_result_l107_107558

theorem competition_result :
  (∀ (Olya Oleg Pasha : Nat), Olya = 1 ∨ Oleg = 1 ∨ Pasha = 1) → 
  (∀ (Olya Oleg Pasha : Nat), (Olya = 1 ∨ Olya = 3) → false) →
  (∀ (Olya Oleg Pasha : Nat), Oleg ≠ 1) →
  (∀ (Olya Oleg Pasha : Nat), (Olya = Oleg ∧ Olya ≠ Pasha)) →
  ∃ (Olya Oleg Pasha : Nat), Oleg = 1 ∧ Pasha = 2 ∧ Olya = 3 :=
begin
  assume each_claims_first,
  assume olya_odd_places_false,
  assume oleg_truthful,
  assume truth_liar_cond,
  sorry
end

end competition_result_l107_107558


namespace maximum_planes_l107_107443

-- Definitions for conditions
def is_non_collinear (points : set (ℝ^3)) : Prop :=
  ∀ (p1 p2 p3 : ℝ^3), {p1, p2, p3} ⊆ points → (∃ plane : set (ℝ^3), ∀ p ∈ {p1, p2, p3}, p ∈ plane) ∧ ¬collinear p1 p2 p3

def is_non_coplanar (points : set (ℝ^3)) : Prop :=
  ∀ (p1 p2 p3 p4 : ℝ^3), {p1, p2, p3, p4} ⊆ points → ¬coplanar p1 p2 p3 p4

noncomputable def combination_3 (n : ℕ) : ℕ :=
  nat.choose n 3

-- Main theorem to be proven
theorem maximum_planes (S : set (ℝ^3)) (h1 : is_non_collinear S) (h2 : is_non_coplanar S) (h3 : finset.card S = 15) :
  (combination_3 15) = 455 :=
by
  sorry -- this skips the actual proof

end maximum_planes_l107_107443


namespace cakes_difference_l107_107335

theorem cakes_difference :
  let bought := 154
  let sold := 91
  bought - sold = 63 :=
by
  let bought := 154
  let sold := 91
  show bought - sold = 63
  sorry

end cakes_difference_l107_107335


namespace inequality_ln_x_lt_x_lt_exp_x_l107_107193

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def g (x : ℝ) : ℝ := Real.log x - x

theorem inequality_ln_x_lt_x_lt_exp_x (x : ℝ) (h : x > 0) : Real.log x < x ∧ x < Real.exp x := by
  -- We need to supply the proof here
  sorry

end inequality_ln_x_lt_x_lt_exp_x_l107_107193


namespace smallest_constant_N_l107_107179

theorem smallest_constant_N (a : ℝ) (ha : a > 0) : 
  let b := a
  let c := a
  (a = b ∧ b = c) → (a^2 + b^2 + c^2) / (a + b + c) > (0 : ℝ) := 
by
  -- Assuming the proof steps are written here
  sorry

end smallest_constant_N_l107_107179


namespace line_circle_no_intersection_l107_107880

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l107_107880


namespace no_intersection_l107_107884

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l107_107884


namespace number_of_sides_possibilities_l107_107212

theorem number_of_sides_possibilities (a b : ℕ) (h₁ : a = 8) (h₂ : b = 5) : 
  { x : ℕ // 3 < x ∧ x < 13 }.to_list.length = 9 :=
by
  sorry

end number_of_sides_possibilities_l107_107212


namespace werewolf_is_A_l107_107246

def is_liar (x : ℕ) : Prop := sorry
def is_knight (x : ℕ) : Prop := sorry
def is_werewolf (x : ℕ) : Prop := sorry

axiom A : ℕ
axiom B : ℕ
axiom C : ℕ

-- Conditions from the problem
axiom A_statement : is_liar A ∨ is_liar B
axiom B_statement : is_werewolf C
axiom exactly_one_werewolf : 
  (is_werewolf A ∧ ¬ is_werewolf B ∧ ¬ is_werewolf C) ∨
  (is_werewolf B ∧ ¬ is_werewolf A ∧ ¬ is_werewolf C) ∨
  (is_werewolf C ∧ ¬ is_werewolf A ∧ ¬ is_werewolf B)
axiom werewolf_is_knight : ∀ x : ℕ, is_werewolf x → is_knight x

-- Prove the conclusion
theorem werewolf_is_A : 
  is_werewolf A ∧ is_knight A :=
sorry

end werewolf_is_A_l107_107246


namespace probability_adjacent_vertices_decagon_l107_107118

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l107_107118


namespace smallest_value_floor_l107_107361

theorem smallest_value_floor (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) :
  (⌊(2 * a + b) / c⌋ + ⌊(b + 2 * c) / a⌋ + ⌊(c + 2 * a) / b⌋) = 9 :=
sorry

end smallest_value_floor_l107_107361


namespace hypotenuse_longer_side_difference_l107_107744

theorem hypotenuse_longer_side_difference
  (x : ℝ)
  (h1 : 17^2 = x^2 + (x - 7)^2)
  (h2 : x = 15)
  : 17 - x = 2 := by
  sorry

end hypotenuse_longer_side_difference_l107_107744


namespace blueberries_in_each_blue_box_l107_107411

theorem blueberries_in_each_blue_box (S B : ℕ) (h1 : S - B = 12) (h2 : 2 * S = 76) : B = 26 := by
  sorry

end blueberries_in_each_blue_box_l107_107411


namespace T_n_sum_general_term_b_b_n_comparison_l107_107008

noncomputable def sequence_a (n : ℕ) : ℕ := sorry  -- Placeholder for sequence {a_n}
noncomputable def S (n : ℕ) : ℕ := sorry  -- Placeholder for sum of first n terms S_n
noncomputable def sequence_b (n : ℕ) (q : ℝ) : ℝ := sorry  -- Placeholder for sequence {b_n}

axiom sequence_a_def : ∀ n : ℕ, 2 * sequence_a (n + 1) = sequence_a n + sequence_a (n + 2)
axiom sequence_a_5 : sequence_a 5 = 5
axiom S_7 : S 7 = 28

noncomputable def T (n : ℕ) : ℝ := (2 * n : ℝ) / (n + 1 : ℝ)

theorem T_n_sum : ∀ n : ℕ, T n = 2 * (1 - 1 / (n + 1)) := sorry

axiom b1 : ℝ
axiom b_def : ∀ (n : ℕ) (q : ℝ), q > 0 → sequence_b (n + 1) q = sequence_b n q + q ^ (sequence_a n)

theorem general_term_b (q : ℝ) (n : ℕ) (hq : q > 0) : 
  (if q = 1 then sequence_b n q = n else sequence_b n q = (1 - q ^ n) / (1 - q)) := sorry

theorem b_n_comparison (q : ℝ) (n : ℕ) (hq : q > 0) : 
  sequence_b n q * sequence_b (n + 2) q < (sequence_b (n + 1) q) ^ 2 := sorry

end T_n_sum_general_term_b_b_n_comparison_l107_107008


namespace jellybean_count_l107_107807

theorem jellybean_count (x : ℝ) (h : (0.75^3) * x = 27) : x = 64 :=
sorry

end jellybean_count_l107_107807


namespace placement_proof_l107_107565

def claimed_first_place (p: String) : Prop := 
  p = "Olya" ∨ p = "Oleg" ∨ p = "Pasha"

def odd_places_boys (positions: ℕ → String) : Prop := 
  (positions 1 = "Oleg" ∨ positions 1 = "Pasha") ∧ (positions 3 = "Oleg" ∨ positions 3 = "Pasha")

def olya_wrong (positions : ℕ → String) : Prop := 
  ¬odd_places_boys positions

def always_truthful_or_lying (Olya_st: Prop) (Oleg_st: Prop) (Pasha_st: Prop) : Prop := 
  Olya_st = Oleg_st ∧ Oleg_st = Pasha_st

def competition_placement : Prop :=
  ∃ (positions: ℕ → String),
    claimed_first_place (positions 1) ∧
    claimed_first_place (positions 2) ∧
    claimed_first_place (positions 3) ∧
    (positions 1 = "Oleg") ∧
    (positions 2 = "Pasha") ∧
    (positions 3 = "Olya") ∧
    olya_wrong positions ∧
    always_truthful_or_lying
      ((claimed_first_place "Olya" ∧ odd_places_boys positions))
      ((claimed_first_place "Oleg" ∧ olya_wrong positions))
      (claimed_first_place "Pasha")

theorem placement_proof : competition_placement :=
  sorry

end placement_proof_l107_107565


namespace line_circle_no_intersection_l107_107864

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l107_107864


namespace PolygonNumberSides_l107_107233

theorem PolygonNumberSides (n : ℕ) (h : n - (1 / 2 : ℝ) * (n * (n - 3)) / 2 = 0) : n = 7 :=
by
  sorry

end PolygonNumberSides_l107_107233


namespace expand_expression_l107_107172

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 * y + 6) = 36 * x + 48 * y + 72 :=
  sorry

end expand_expression_l107_107172


namespace no_intersection_l107_107885

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l107_107885


namespace partition_subset_sum_l107_107657

variable {p k : ℕ}

def V_p (p : ℕ) := {k : ℕ | p ∣ (k * (k + 1) / 2) ∧ k ≥ 2 * p - 1}

theorem partition_subset_sum (p : ℕ) (hp : Nat.Prime p) (k : ℕ) : k ∈ V_p p := sorry

end partition_subset_sum_l107_107657


namespace youseff_distance_l107_107084

theorem youseff_distance (x : ℕ) 
  (walk_time_per_block : ℕ := 1)
  (bike_time_per_block_secs : ℕ := 20)
  (time_difference : ℕ := 12) :
  (x : ℕ) = 18 :=
by
  -- walking time
  let walk_time := x * walk_time_per_block
  
  -- convert bike time per block to minutes
  let bike_time_per_block := (bike_time_per_block_secs : ℚ) / 60

  -- biking time
  let bike_time := x * bike_time_per_block

  -- set up the equation for time difference
  have time_eq := walk_time - bike_time = time_difference
  
  -- from here, the actual proof steps would follow, 
  -- but we include "sorry" as a placeholder since the focus is on the statement.
  sorry

end youseff_distance_l107_107084


namespace maximum_fraction_l107_107130

theorem maximum_fraction (A B : ℕ) (h1 : A ≠ B) (h2 : 0 < A ∧ A < 1000) (h3 : 0 < B ∧ B < 1000) :
  ∃ (A B : ℕ), (A = 500) ∧ (B = 499) ∧ (A ≠ B) ∧ (0 < A ∧ A < 1000) ∧ (0 < B ∧ B < 1000) ∧ (A - B = 1) ∧ (A + B = 999) ∧ (499 / 500 = 0.998) := sorry

end maximum_fraction_l107_107130


namespace decagon_adjacent_probability_l107_107128

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l107_107128


namespace smallest_positive_integer_l107_107595

theorem smallest_positive_integer (n : ℕ) :
  (∃ n : ℕ, n > 0 ∧ n % 30 = 0 ∧ n % 40 = 0 ∧ n % 16 ≠ 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 30 = 0 ∧ m % 40 = 0 ∧ m % 16 ≠ 0) → n ≤ m) ↔ n = 120 :=
by
  sorry

end smallest_positive_integer_l107_107595


namespace line_circle_no_intersect_l107_107858

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l107_107858


namespace tetrahedron_edge_length_correct_l107_107650

noncomputable def radius := Real.sqrt 2
noncomputable def center_to_center_distance := 2 * radius
noncomputable def tetrahedron_edge_length := center_to_center_distance

theorem tetrahedron_edge_length_correct :
  tetrahedron_edge_length = 2 * Real.sqrt 2 := by
  sorry

end tetrahedron_edge_length_correct_l107_107650


namespace coords_with_respect_to_origin_l107_107535

/-- Given that the coordinates of point A are (-1, 2), prove that the coordinates of point A
with respect to the origin in the plane rectangular coordinate system xOy are (-1, 2). -/
theorem coords_with_respect_to_origin (A : (ℝ × ℝ)) (h : A = (-1, 2)) : A = (-1, 2) :=
sorry

end coords_with_respect_to_origin_l107_107535


namespace abs_div_inequality_l107_107178

theorem abs_div_inequality (x : ℝ) : 
  (|-((x+1)/x)| > (x+1)/x) ↔ (-1 < x ∧ x < 0) :=
sorry

end abs_div_inequality_l107_107178


namespace planes_perpendicular_l107_107010

variables {m n : Type} -- lines
variables {α β : Type} -- planes

axiom lines_different : m ≠ n
axiom planes_different : α ≠ β
axiom parallel_lines : ∀ (m n : Type), Prop -- m ∥ n
axiom parallel_plane_line : ∀ (m α : Type), Prop -- m ∥ α
axiom perp_plane_line : ∀ (n β : Type), Prop -- n ⊥ β
axiom perp_planes : ∀ (α β : Type), Prop -- α ⊥ β

theorem planes_perpendicular 
  (h1 : parallel_lines m n) 
  (h2 : parallel_plane_line m α) 
  (h3 : perp_plane_line n β) 
: perp_planes α β := 
sorry

end planes_perpendicular_l107_107010


namespace bronson_yellow_leaves_l107_107618

theorem bronson_yellow_leaves :
  let thursday_leaves := 12 in
  let friday_leaves := 13 in
  let total_leaves := thursday_leaves + friday_leaves in
  let brown_leaves := total_leaves * 20 / 100 in
  let green_leaves := total_leaves * 20 / 100 in
  let yellow_leaves := total_leaves - (brown_leaves + green_leaves) in
  yellow_leaves = 15 :=
by
  sorry

end bronson_yellow_leaves_l107_107618


namespace coordinates_after_5_seconds_l107_107930

-- Define the initial coordinates of point P
def initial_coordinates : ℚ × ℚ := (-10, 10)

-- Define the velocity vector of point P
def velocity_vector : ℚ × ℚ := (4, -3)

-- Asserting the coordinates of point P after 5 seconds
theorem coordinates_after_5_seconds : 
   initial_coordinates + 5 • velocity_vector = (10, -5) :=
by 
  sorry

end coordinates_after_5_seconds_l107_107930


namespace probability_adjacent_vertices_in_decagon_l107_107106

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l107_107106


namespace percentage_less_than_l107_107370

theorem percentage_less_than (x y z : Real) (h1 : x = 1.20 * y) (h2 : x = 0.84 * z) : 
  ((z - y) / z) * 100 = 30 := 
sorry

end percentage_less_than_l107_107370


namespace total_unique_working_games_l107_107043

-- Define the given conditions
def initial_games_from_friend := 25
def non_working_games_from_friend := 12

def games_from_garage_sale := 15
def non_working_games_from_garage_sale := 8
def duplicate_games := 3

-- Calculate the number of working games from each source
def working_games_from_friend := initial_games_from_friend - non_working_games_from_friend
def total_garage_sale_games := games_from_garage_sale - non_working_games_from_garage_sale
def unique_working_games_from_garage_sale := total_garage_sale_games - duplicate_games

-- Theorem statement
theorem total_unique_working_games : 
  working_games_from_friend + unique_working_games_from_garage_sale = 17 := by
  sorry

end total_unique_working_games_l107_107043


namespace prove_range_of_p_l107_107048

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x - 1

def A (x : ℝ) : Prop := x > 2
def no_pre_image_in_A (p : ℝ) : Prop := ∀ x, A x → f x ≠ p

theorem prove_range_of_p (p : ℝ) : no_pre_image_in_A p ↔ p > -1 := by
  sorry

end prove_range_of_p_l107_107048


namespace triangle_area_inscribed_circle_l107_107453

noncomputable def area_of_triangle_ratio (r : ℝ) (area : ℝ) : Prop :=
  let scale := r / 4
  let s1 := 2 * scale
  let s2 := 3 * scale
  let s3 := 4 * scale
  let s := (s1 + s2 + s3) / 2
  let heron := sqrt (s * (s - s1) * (s - s2) * (s - s3))
  area = heron

theorem triangle_area_inscribed_circle :
  ∀(r : ℝ), r = 4 → area_of_triangle_ratio r (3 * sqrt 15) :=
by
  intro r hr
  rw [hr]
  sorry

end triangle_area_inscribed_circle_l107_107453


namespace geom_sixth_term_is_31104_l107_107234

theorem geom_sixth_term_is_31104 :
  ∃ (r : ℝ), 4 * r^8 = 39366 ∧ 4 * r^(6-1) = 31104 :=
by
  sorry

end geom_sixth_term_is_31104_l107_107234


namespace condition_sufficient_but_not_necessary_l107_107706

theorem condition_sufficient_but_not_necessary (x : ℝ) :
  (x^3 > 8 → |x| > 2) ∧ (|x| > 2 → ¬ (x^3 ≤ 8 ∨ x^3 ≥ 8)) := by
  sorry

end condition_sufficient_but_not_necessary_l107_107706


namespace tan_neg_240_eq_neg_sqrt_3_l107_107648

theorem tan_neg_240_eq_neg_sqrt_3 : Real.tan (-4 * Real.pi / 3) = -Real.sqrt 3 :=
by
  sorry

end tan_neg_240_eq_neg_sqrt_3_l107_107648


namespace gcd_35_x_eq_7_in_range_80_90_l107_107942

theorem gcd_35_x_eq_7_in_range_80_90 {n : ℕ} (h₁ : Nat.gcd 35 n = 7) (h₂ : 80 < n) (h₃ : n < 90) : n = 84 :=
by
  sorry

end gcd_35_x_eq_7_in_range_80_90_l107_107942


namespace decagon_adjacent_vertex_probability_l107_107125

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l107_107125


namespace probability_of_five_out_of_seven_days_l107_107275

noncomputable def probability_exactly_5_out_of_7 (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * (p^(n-k))

theorem probability_of_five_out_of_seven_days :
  probability_exactly_5_out_of_7 7 5 (1/2) = 21 / 128 := by
  sorry

end probability_of_five_out_of_seven_days_l107_107275


namespace equation_of_parallel_line_passing_through_point_l107_107582

variable (x y : ℝ)

def is_point_on_line (x_val y_val : ℝ) (a b c : ℝ) : Prop := a * x_val + b * y_val + c = 0

def is_parallel (slope1 slope2 : ℝ) : Prop := slope1 = slope2

theorem equation_of_parallel_line_passing_through_point :
  (is_point_on_line (-1) 3 1 (-2) 7) ∧ (is_parallel (1 / 2) (1 / 2)) → (∀ x y, is_point_on_line x y 1 (-2) 7) :=
by
  sorry

end equation_of_parallel_line_passing_through_point_l107_107582


namespace farmer_harvest_correct_l107_107433

-- Define the conditions
def estimated_harvest : ℕ := 48097
def additional_harvest : ℕ := 684
def total_harvest : ℕ := 48781

-- The proof statement
theorem farmer_harvest_correct :
  estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end farmer_harvest_correct_l107_107433


namespace expand_polynomial_l107_107642

theorem expand_polynomial (x : ℝ) :
  (x + 4) * (x - 9) = x^2 - 5 * x - 36 := 
sorry

end expand_polynomial_l107_107642


namespace cary_ivy_removal_days_correct_l107_107986

noncomputable def cary_ivy_removal_days (initial_ivy : ℕ) (ivy_removed_per_day : ℕ) (ivy_growth_per_night : ℕ) : ℕ :=
  initial_ivy / (ivy_removed_per_day - ivy_growth_per_night)

theorem cary_ivy_removal_days_correct :
  cary_ivy_removal_days 40 6 2 = 10 :=
by
  -- The body of the proof is omitted; it will be filled with the actual proof.
  sorry

end cary_ivy_removal_days_correct_l107_107986


namespace lemonade_served_l107_107469

def glasses_per_pitcher : ℕ := 5
def number_of_pitchers : ℕ := 6
def total_glasses_served : ℕ := glasses_per_pitcher * number_of_pitchers

theorem lemonade_served : total_glasses_served = 30 :=
by
  -- proof goes here
  sorry

end lemonade_served_l107_107469


namespace greatest_odd_factors_below_200_l107_107716

theorem greatest_odd_factors_below_200 : ∃ n : ℕ, (n < 200) ∧ (n = 196) ∧ (∃ k : ℕ, n = k^2) ∧ ∀ m : ℕ, (m < 200) ∧ (∃ j : ℕ, m = j^2) → m ≤ n := by
  sorry

end greatest_odd_factors_below_200_l107_107716


namespace tanker_fill_rate_l107_107146

theorem tanker_fill_rate :
  let barrels_per_min := 2
  let liters_per_barrel := 159
  let cubic_meters_per_liter := 0.001
  let minutes_per_hour := 60
  let liters_per_min := barrels_per_min * liters_per_barrel
  let liters_per_hour := liters_per_min * minutes_per_hour
  let cubic_meters_per_hour := liters_per_hour * cubic_meters_per_liter
  cubic_meters_per_hour = 19.08 :=
  by {
    sorry
  }

end tanker_fill_rate_l107_107146


namespace judah_crayons_l107_107252

theorem judah_crayons (karen beatrice gilbert judah : ℕ) 
  (h1 : karen = 128)
  (h2 : karen = 2 * beatrice)
  (h3 : beatrice = 2 * gilbert)
  (h4 : gilbert = 4 * judah) : 
  judah = 8 :=
by
  sorry

end judah_crayons_l107_107252


namespace rectangles_with_one_gray_cell_l107_107515

/- Definitions from conditions -/
def total_gray_cells : ℕ := 40
def blue_cells : ℕ := 36
def red_cells : ℕ := 4

/- The number of rectangles containing exactly one gray cell is the proof goal -/
theorem rectangles_with_one_gray_cell :
  (blue_cells * 4 + red_cells * 8) = 176 :=
sorry

end rectangles_with_one_gray_cell_l107_107515


namespace find_k_l107_107039

theorem find_k
  (angle_C : ℝ)
  (AB : ℝ × ℝ)
  (AC : ℝ × ℝ)
  (h1 : angle_C = 90)
  (h2 : AB = (k, 1))
  (h3 : AC = (2, 3)) :
  k = 5 := by
  sorry

end find_k_l107_107039


namespace log_4_135_eq_half_log_2_45_l107_107011

noncomputable def a : ℝ := Real.log 135 / Real.log 4
noncomputable def b : ℝ := Real.log 45 / Real.log 2

theorem log_4_135_eq_half_log_2_45 : a = b / 2 :=
by
  sorry

end log_4_135_eq_half_log_2_45_l107_107011


namespace math_team_combinations_l107_107073

def numGirls : ℕ := 4
def numBoys : ℕ := 7
def girlsToChoose : ℕ := 2
def boysToChoose : ℕ := 3

def comb (n k : ℕ) : ℕ := n.choose k

theorem math_team_combinations : 
  comb numGirls girlsToChoose * comb numBoys boysToChoose = 210 := 
by
  sorry

end math_team_combinations_l107_107073


namespace function_properties_l107_107506

variable (f : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, 2 * f x * f y = f (x + y) + f (x - y))
variable (h2 : f 1 = -1)

theorem function_properties :
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f x + f (1 - x) = 0) :=
sorry

end function_properties_l107_107506


namespace find_other_number_l107_107070

theorem find_other_number 
  (A B : ℕ) 
  (h1 : A = 385) 
  (h2 : Nat.lcm A B = 2310) 
  (h3 : Nat.gcd A B = 30) : 
  B = 180 := 
by
  sorry

end find_other_number_l107_107070


namespace M_intersection_P_l107_107519

namespace IntersectionProof

-- Defining the sets M and P with given conditions
def M : Set ℝ := {y | ∃ x : ℝ, y = 3 ^ x}
def P : Set ℝ := {y | y ≥ 1}

-- The theorem that corresponds to the problem statement
theorem M_intersection_P : (M ∩ P) = {y | y ≥ 1} :=
sorry

end IntersectionProof

end M_intersection_P_l107_107519


namespace problem_I_problem_II_l107_107263

-- Problem (I)
def A : Set ℝ := { x | x > 2 ∨ x < -1 }
def B : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }
def A_inter_B : Set ℝ := { x | (-3 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 3) }

theorem problem_I : A ∩ B = A_inter_B :=
by
  sorry

-- Problem (II)
def C (m : ℝ) : Set ℝ := { x | 2 * m - 1 < x ∧ x < m + 1 }

theorem problem_II (m : ℝ) : (C m ⊆ B) → m ≥ -1 :=
by
  sorry

end problem_I_problem_II_l107_107263


namespace jellybeans_original_count_l107_107811

theorem jellybeans_original_count (x : ℝ) (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_original_count_l107_107811


namespace octadecagon_diagonals_l107_107177

def num_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octadecagon_diagonals : num_of_diagonals 18 = 135 := by
  sorry

end octadecagon_diagonals_l107_107177


namespace lune_area_l107_107450

-- Definition of a semicircle's area given its diameter
def area_of_semicircle (d : ℝ) : ℝ := (1 / 2) * Real.pi * (d / 2) ^ 2

-- Definition of the lune area
def area_of_lune : ℝ :=
  let smaller_semicircle_area := area_of_semicircle 3
  let overlapping_sector_area := (1 / 3) * Real.pi * (4 / 2) ^ 2
  smaller_semicircle_area - overlapping_sector_area

-- Theorem statement declaring the solution to be proved
theorem lune_area : area_of_lune = (11 / 24) * Real.pi :=
by
  sorry

end lune_area_l107_107450


namespace third_number_is_forty_four_l107_107983

theorem third_number_is_forty_four (a b c d e : ℕ) (h1 : a = e + 1) (h2 : b = e) 
  (h3 : c = e - 1) (h4 : d = e - 2) (h5 : e = e - 3) 
  (h6 : (a + b + c) / 3 = 45) (h7 : (c + d + e) / 3 = 43) : 
  c = 44 := 
sorry

end third_number_is_forty_four_l107_107983


namespace probability_different_colors_l107_107588

theorem probability_different_colors
  (red_chips green_chips : ℕ)
  (total_chips : red_chips + green_chips = 10)
  (prob_red : ℚ := red_chips / 10)
  (prob_green : ℚ := green_chips / 10) :
  ((prob_red * prob_green) + (prob_green * prob_red) = 12 / 25) := by
sorry

end probability_different_colors_l107_107588


namespace height_after_five_years_l107_107145

namespace PapayaTreeGrowth

def growth_first_year := true → ℝ
def growth_second_year (x : ℝ) := 1.5 * x
def growth_third_year (x : ℝ) := 1.5 * growth_second_year x
def growth_fourth_year (x : ℝ) := 2 * growth_third_year x
def growth_fifth_year (x : ℝ) := 0.5 * growth_fourth_year x

def total_growth (x : ℝ) := x + growth_second_year x + growth_third_year x +
                             growth_fourth_year x + growth_fifth_year x

theorem height_after_five_years (x : ℝ) (H : total_growth x = 23) : x = 2 :=
by
  sorry

end PapayaTreeGrowth

end height_after_five_years_l107_107145


namespace probability_of_yellow_marble_l107_107334

noncomputable def marbles_prob : ℚ := 
  let prob_white_A := 4 / 9
  let prob_black_A := 5 / 9
  let prob_yellow_B := 7 / 10
  let prob_blue_B := 3 / 10
  let prob_yellow_C := 3 / 9
  let prob_blue_C := 6 / 9
  let prob_yellow_D := 5 / 9
  let prob_blue_D := 4 / 9
  let prob_white_A_and_yellow_B := prob_white_A * prob_yellow_B
  let prob_black_A_and_blue_C := prob_black_A * prob_blue_C
  let prob_black_and_C_and_yellow_D := prob_black_A_and_blue_C * prob_yellow_D
  (prob_white_A_and_yellow_B + prob_black_and_C_and_yellow_D).reduce

theorem probability_of_yellow_marble :
  marbles_prob = 1884 / 3645 :=
sorry

end probability_of_yellow_marble_l107_107334


namespace innings_count_l107_107137

-- Definitions of the problem conditions
def total_runs (n : ℕ) : ℕ := 63 * n
def highest_score : ℕ := 248
def lowest_score : ℕ := 98

theorem innings_count (n : ℕ) (h : total_runs n - highest_score - lowest_score = 58 * (n - 2)) : n = 46 :=
  sorry

end innings_count_l107_107137


namespace dance_team_recruits_l107_107456

theorem dance_team_recruits :
  ∃ (x : ℕ), x + 2 * x + (2 * x + 10) = 100 ∧ (2 * x + 10) = 46 :=
by
  sorry

end dance_team_recruits_l107_107456


namespace probability_adjacent_vertices_in_decagon_l107_107093

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l107_107093


namespace min_questions_to_determine_number_l107_107959

theorem min_questions_to_determine_number : 
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 50) → 
  ∃ (q : ℕ), q = 15 ∧ 
  ∀ (primes : ℕ → Prop), 
  (∀ p, primes p → Nat.Prime p ∧ p ≤ 50) → 
  (∀ p, primes p → (n % p = 0 ↔ p ∣ n)) → 
  (∃ m, (∀ k, k < m → primes k → k ∣ n)) :=
sorry

end min_questions_to_determine_number_l107_107959


namespace part1_part2_l107_107495

theorem part1 (θ : ℝ) (h : Real.sin (2 * θ) = 2 * Real.cos θ) :
  (6 * Real.sin θ + Real.cos θ) / (3 * Real.sin θ - 2 * Real.cos θ) = 13 / 4 :=
sorry

theorem part2 (θ : ℝ) (h : Real.sin (2 * θ) = 2 * Real.cos θ) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (2 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 8 / 7 :=
sorry

end part1_part2_l107_107495


namespace john_bought_3_tshirts_l107_107381

theorem john_bought_3_tshirts (T : ℕ) (h : 20 * T + 50 = 110) : T = 3 := 
by 
  sorry

end john_bought_3_tshirts_l107_107381


namespace triangle_side_count_l107_107215

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end triangle_side_count_l107_107215


namespace max_composite_rel_prime_set_l107_107601

theorem max_composite_rel_prime_set : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 10 ≤ n ∧ n ≤ 99 ∧ ¬Nat.Prime n) ∧ 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → Nat.gcd a b = 1) ∧ 
  S.card = 4 := by
sorry

end max_composite_rel_prime_set_l107_107601


namespace spider_eyes_solution_l107_107052

def spider_eyes_problem: Prop :=
  ∃ (x : ℕ), (3 * x) + (50 * 2) = 124 ∧ x = 8

theorem spider_eyes_solution : spider_eyes_problem :=
  sorry

end spider_eyes_solution_l107_107052


namespace inverse_value_l107_107014

def f (x : ℤ) : ℤ := 5 * x ^ 3 - 3

theorem inverse_value : ∀ y, (f y) = 4 → y = 317 :=
by
  intros
  sorry

end inverse_value_l107_107014


namespace alia_markers_count_l107_107780

theorem alia_markers_count :
  ∀ (Alia Austin Steve Bella : ℕ),
  (Alia = 2 * Austin) →
  (Austin = (1 / 3) * Steve) →
  (Steve = 60) →
  (Bella = (3 / 2) * Alia) →
  Alia = 40 :=
by
  intros Alia Austin Steve Bella H1 H2 H3 H4
  sorry

end alia_markers_count_l107_107780


namespace passed_boys_count_l107_107393

theorem passed_boys_count (P F : ℕ) 
  (h1 : P + F = 120) 
  (h2 : 37 * 120 = 39 * P + 15 * F) : 
  P = 110 :=
sorry

end passed_boys_count_l107_107393


namespace sum_arithmetic_sequence_has_max_value_l107_107030

noncomputable section
open Classical

-- Defining an arithmetic sequence with first term a1 and common difference d
def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + d * (n - 1)

-- Defining the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

-- The main statement to prove: Sn has a maximum value given conditions a1 > 0 and d < 0
theorem sum_arithmetic_sequence_has_max_value (a1 d : ℝ) (h1 : a1 > 0) (h2 : d < 0) :
  ∃ M, ∀ n, sum_arithmetic_sequence a1 d n ≤ M :=
by
  sorry

end sum_arithmetic_sequence_has_max_value_l107_107030


namespace line_circle_no_intersect_l107_107856

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l107_107856


namespace students_taking_either_but_not_both_l107_107174

-- Definitions to encapsulate the conditions
def students_taking_both : ℕ := 15
def students_taking_mathematics : ℕ := 30
def students_taking_history_only : ℕ := 12

-- The goal is to prove the number of students taking mathematics or history but not both
theorem students_taking_either_but_not_both
  (hb : students_taking_both = 15)
  (hm : students_taking_mathematics = 30)
  (hh : students_taking_history_only = 12) : 
  students_taking_mathematics - students_taking_both + students_taking_history_only = 27 :=
by
  sorry

end students_taking_either_but_not_both_l107_107174


namespace jellybeans_initial_amount_l107_107801

theorem jellybeans_initial_amount (x : ℝ) 
  (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_initial_amount_l107_107801


namespace decagon_adjacent_probability_l107_107102

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l107_107102


namespace line_circle_no_intersect_l107_107859

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l107_107859


namespace impossible_five_correct_l107_107404

open Finset

theorem impossible_five_correct (P L : Finset ℕ) (hP : P.card = 6) (hL : L.card = 6)
  (letters_assigned : P → L) :
  ¬ ∃ H : Π p ∈ P, p = letters_assigned p, card P - card (↑((λ p, p = letters_assigned p) '' P.toFin)) = 5 := 
by 
  sorry

end impossible_five_correct_l107_107404


namespace find_y_l107_107016

open Complex

theorem find_y (y : ℝ) (h₁ : (3 : ℂ) + (↑y : ℂ) * I = z₁) 
  (h₂ : (2 : ℂ) - I = z₂) 
  (h₃ : z₁ / z₂ = 1 + I) 
  (h₄ : z₁ = (3 : ℂ) + (↑y : ℂ) * I) 
  (h₅ : z₂ = (2 : ℂ) - I)
  : y = 1 :=
sorry


end find_y_l107_107016


namespace greatest_root_f_l107_107345

noncomputable def f (x : ℝ) : ℝ := 21 * x ^ 4 - 20 * x ^ 2 + 3

theorem greatest_root_f :
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x :=
sorry

end greatest_root_f_l107_107345


namespace g_1200_value_l107_107047

noncomputable def g : ℝ → ℝ := sorry

-- Assume the given condition as a definition
axiom functional_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y

-- Assume the given value of g(1000)
axiom g_1000_value : g 1000 = 4

-- Prove that g(1200) = 10/3
theorem g_1200_value : g 1200 = 10 / 3 := by
  sorry

end g_1200_value_l107_107047


namespace option_A_option_B_option_C_option_D_l107_107667

-- Define the equation of the curve
def curve (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 1) + y^2 / (5 - k) = 1

-- Prove that when k=2, the curve is a circle
theorem option_A (x y : ℝ) : curve 2 x y ↔ x^2 + y^2 = 3 :=
by
  sorry

-- Prove the necessary and sufficient condition for the curve to be an ellipse
theorem option_B (k : ℝ) : (-1 < k ∧ k < 5) ↔ ∃ x y, curve k x y ∧ (k ≠ 2) :=
by
  sorry

-- Prove the condition for the curve to be a hyperbola with foci on the y-axis
theorem option_C (k : ℝ) : k < -1 ↔ ∃ x y, curve k x y ∧ (k < -1 ∧ k < 5) :=
by
  sorry

-- Prove that there does not exist a real number k such that the curve is a parabola
theorem option_D : ¬ (∃ k x y, curve k x y ∧ ∃ a b, x = a ∧ y = b) :=
by
  sorry

end option_A_option_B_option_C_option_D_l107_107667


namespace new_time_between_maintenance_checks_l107_107313

-- Definitions based on the conditions
def original_time : ℝ := 25
def percentage_increase : ℝ := 0.20

-- Statement to be proved
theorem new_time_between_maintenance_checks : original_time * (1 + percentage_increase) = 30 := by
  sorry

end new_time_between_maintenance_checks_l107_107313


namespace find_fraction_l107_107344

-- Variables and Definitions
variables (x : ℚ)

-- Conditions
def condition1 := (2 / 3) / x = (3 / 5) / (7 / 15)

-- Theorem to prove the certain fraction
theorem find_fraction (h : condition1 x) : x = 14 / 27 :=
by sorry

end find_fraction_l107_107344


namespace probability_at_tree_correct_expected_distance_correct_l107_107926

-- Define the initial conditions
def initial_tree (n : ℕ) : ℕ := n + 1
def total_trees (n : ℕ) : ℕ := 2 * n + 1

-- Define the probability that the drunkard is at each tree T_i (1 <= i <= 2n+1) at the end of the nth minute
def probability_at_tree (n i : ℕ) : ℚ :=
  if 1 ≤ i ∧ i ≤ total_trees n then
    (Nat.choose (2*n) (i-1)) / (2^(2*n))
  else
    0

-- Define the expected distance between the final position and the initial tree T_{n+1}
def expected_distance (n : ℕ) : ℚ :=
  n * (Nat.choose (2*n) n) / (2^(2*n))

-- Statements to prove
theorem probability_at_tree_correct (n i : ℕ) (hi : 1 ≤ i ∧ i ≤ total_trees n)  :
  probability_at_tree n i = (Nat.choose (2*n) (i-1)) / (2^(2*n)) :=
by
  sorry

theorem expected_distance_correct (n : ℕ) :
  expected_distance n = n * (Nat.choose (2*n) n) / (2^(2*n)) :=
by
  sorry

end probability_at_tree_correct_expected_distance_correct_l107_107926


namespace hydrated_aluminum_iodide_props_l107_107473

noncomputable def Al_mass : ℝ := 26.98
noncomputable def I_mass : ℝ := 126.90
noncomputable def H2O_mass : ℝ := 18.015
noncomputable def AlI3_mass (mass_AlI3: ℝ) : ℝ := 26.98 + 3 * 126.90

noncomputable def mass_percentage_iodine (mass_AlI3 mass_sample: ℝ) : ℝ :=
  (mass_AlI3 * (3 * I_mass / (Al_mass + 3 * I_mass)) / mass_sample) * 100

noncomputable def value_x (mass_H2O mass_AlI3: ℝ) : ℝ :=
  (mass_H2O / H2O_mass) / (mass_AlI3 / (Al_mass + 3 * I_mass))

theorem hydrated_aluminum_iodide_props (mass_AlI3 mass_H2O mass_sample: ℝ)
    (h_sample: mass_AlI3 + mass_H2O = mass_sample) :
    ∃ (percentage: ℝ) (x: ℝ), percentage = mass_percentage_iodine mass_AlI3 mass_sample ∧
                                      x = value_x mass_H2O mass_AlI3 :=
by
  sorry

end hydrated_aluminum_iodide_props_l107_107473


namespace greatest_odd_factors_l107_107722

theorem greatest_odd_factors (n : ℕ) (h1 : n < 200) (h2 : ∀ k < 200, k ≠ 196 → odd (number_of_factors k) = false) : n = 196 :=
sorry

end greatest_odd_factors_l107_107722


namespace brick_height_calc_l107_107607

theorem brick_height_calc 
  (length_wall : ℝ) (height_wall : ℝ) (width_wall : ℝ) 
  (num_bricks : ℕ) 
  (length_brick : ℝ) (width_brick : ℝ) 
  (H : ℝ) 
  (volume_wall : ℝ) 
  (volume_brick : ℝ)
  (condition1 : length_wall = 800) 
  (condition2 : height_wall = 600) 
  (condition3 : width_wall = 22.5)
  (condition4 : num_bricks = 3200) 
  (condition5 : length_brick = 50) 
  (condition6 : width_brick = 11.25) 
  (condition7 : volume_wall = length_wall * height_wall * width_wall) 
  (condition8 : volume_brick = length_brick * width_brick * H) 
  (condition9 : num_bricks * volume_brick = volume_wall) 
  : H = 6 := 
by
  sorry

end brick_height_calc_l107_107607


namespace max_sum_of_digits_in_watch_l107_107316

theorem max_sum_of_digits_in_watch : ∃ max_sum : ℕ, max_sum = 23 ∧ 
  ∀ hours minutes : ℕ, 
  (1 ≤ hours ∧ hours ≤ 12) → 
  (0 ≤ minutes ∧ minutes < 60) → 
  let hour_digits_sum := (hours / 10) + (hours % 10) in
  let minute_digits_sum := (minutes / 10) + (minutes % 10) in
  hour_digits_sum + minute_digits_sum ≤ max_sum :=
sorry

end max_sum_of_digits_in_watch_l107_107316


namespace calculate_final_speed_l107_107464

noncomputable def final_speed : ℝ :=
  let v1 : ℝ := (150 * 1.60934 * 1000) / 3600
  let v2 : ℝ := (170 * 1000) / 3600
  let v_decreased : ℝ := v1 - v2
  let a : ℝ := (500000 * 0.01) / 60
  v_decreased + a * (30 * 60)

theorem calculate_final_speed : final_speed = 150013.45 :=
by
  sorry

end calculate_final_speed_l107_107464


namespace log_identity_l107_107416

noncomputable def my_log (base x : ℝ) := Real.log x / Real.log base

theorem log_identity (x : ℝ) (h : x > 0) (h1 : x ≠ 1) : 
  (my_log 4 x) * (my_log x 5) = my_log 4 5 :=
by
  sorry

end log_identity_l107_107416


namespace votes_diff_eq_70_l107_107054

noncomputable def T : ℝ := 350
def votes_against (T : ℝ) : ℝ := 0.40 * T
def votes_favor (T : ℝ) (X : ℝ) : ℝ := votes_against T + X

theorem votes_diff_eq_70 :
  ∃ X : ℝ, 350 = votes_against T + votes_favor T X → X = 70 :=
by
  sorry

end votes_diff_eq_70_l107_107054


namespace equal_roots_for_specific_k_l107_107520

theorem equal_roots_for_specific_k (k : ℝ) :
  ((k - 1) * x^2 + 6 * x + 9 = 0) → (6^2 - 4*(k-1)*9 = 0) → (k = 2) :=
by sorry

end equal_roots_for_specific_k_l107_107520


namespace book_cost_price_l107_107985

theorem book_cost_price (SP : ℝ) (P : ℝ) (C : ℝ) (hSP: SP = 260) (hP: P = 0.20) : C = 216.67 :=
by 
  sorry

end book_cost_price_l107_107985


namespace statement_B_correct_statement_C_correct_l107_107415

theorem statement_B_correct :
  let total_outcomes := Nat.choose 5 2
  let diff_color_outcomes := 6
  (diff_color_outcomes / total_outcomes : Real) = 3 / 5 := by
  sorry

theorem statement_C_correct :
  let p_A := 0.8
  let p_B := 0.9
  1 - (1 - p_A) * (1 - p_B) = 0.98 := by
  sorry

end statement_B_correct_statement_C_correct_l107_107415


namespace badgers_at_least_five_wins_l107_107281

open BigOperators

noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem badgers_at_least_five_wins :
  let n := 9
  let p := 0.5
  (∑ k in (finset.range (n + 1)).filter (λ k, 5 ≤ k), binomial_prob n k p) = 1 / 2 :=
by
  sorry

end badgers_at_least_five_wins_l107_107281


namespace melinda_payment_l107_107572

theorem melinda_payment
  (D C : ℝ)
  (h1 : 3 * D + 4 * C = 4.91)
  (h2 : D = 0.45) :
  5 * D + 6 * C = 7.59 := 
by 
-- proof steps go here
sorry

end melinda_payment_l107_107572


namespace probability_adjacent_vertices_decagon_l107_107119

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l107_107119


namespace second_monkey_took_20_peaches_l107_107524

theorem second_monkey_took_20_peaches (total_peaches : ℕ) 
  (h1 : total_peaches > 0)
  (eldest_share : ℕ)
  (middle_share : ℕ)
  (youngest_share : ℕ)
  (h3 : total_peaches = eldest_share + middle_share + youngest_share)
  (h4 : eldest_share = (total_peaches * 5) / 9)
  (second_total : ℕ := total_peaches - eldest_share)
  (h5 : middle_share = (second_total * 5) / 9)
  (h6 : youngest_share = second_total - middle_share)
  (h7 : eldest_share - youngest_share = 29) :
  middle_share = 20 :=
by
  sorry

end second_monkey_took_20_peaches_l107_107524


namespace Maurice_current_age_l107_107328

variable (Ron_now Maurice_now : ℕ)

theorem Maurice_current_age
  (h1 : Ron_now = 43)
  (h2 : ∀ t Ron_future Maurice_future : ℕ, Ron_future = 4 * Maurice_future → Ron_future = Ron_now + 5 → Maurice_future = Maurice_now + 5) :
  Maurice_now = 7 := 
sorry

end Maurice_current_age_l107_107328


namespace competition_result_l107_107561

-- Define the participants
inductive Person
| Olya | Oleg | Pasha
deriving DecidableEq, Repr

-- Define the placement as an enumeration
inductive Place
| first | second | third
deriving DecidableEq, Repr

-- Define the statements
structure Statements :=
(olyas_claim : Place)
(olyas_statement : Prop)
(olegs_statement : Prop)

-- Define the conditions
def conditions (s : Statements) : Prop :=
  -- All claimed first place
  s.olyas_claim = Place.first ∧ s.olyas_statement ∧ s.olegs_statement

-- Define the final placement
structure Placement :=
(olyas_place : Place)
(olegs_place : Place)
(pashas_place : Place)

-- Define the correct answer
def correct_placement : Placement :=
{ olyas_place := Place.third,
  olegs_place := Place.first,
  pashas_place := Place.second }

-- Lean statement for the problem
theorem competition_result (s : Statements) (h : conditions s) : Placement :=
{ olyas_place := Place.third,
  olegs_place := Place.first,
  pashas_place := Place.second } := sorry

end competition_result_l107_107561


namespace max_pens_l107_107417

theorem max_pens (total_money notebook_cost pen_cost num_notebooks : ℝ) (notebook_qty pen_qty : ℕ):
  total_money = 18 ∧ notebook_cost = 3.6 ∧ pen_cost = 3 ∧ num_notebooks = 2 →
  (pen_qty = 1 ∨ pen_qty = 2 ∨ pen_qty = 3) ↔ (2 * notebook_cost + pen_qty * pen_cost ≤ total_money) :=
by {
  sorry
}

end max_pens_l107_107417


namespace solve_coffee_problem_l107_107767

variables (initial_stock new_purchase : ℕ)
           (initial_decaf_percentage new_decaf_percentage : ℚ)
           (total_stock total_decaf weight_percentage_decaf : ℚ)

def coffee_problem :=
  initial_stock = 400 ∧
  initial_decaf_percentage = 0.20 ∧
  new_purchase = 100 ∧
  new_decaf_percentage = 0.50 ∧
  total_stock = initial_stock + new_purchase ∧
  total_decaf = initial_stock * initial_decaf_percentage + new_purchase * new_decaf_percentage ∧
  weight_percentage_decaf = (total_decaf / total_stock) * 100

theorem solve_coffee_problem : coffee_problem 400 100 0.20 0.50 500 130 26 :=
by {
  sorry
}

end solve_coffee_problem_l107_107767


namespace ivy_stripping_days_l107_107988

theorem ivy_stripping_days :
  ∃ (days_needed : ℕ), (days_needed * (6 - 2) = 40) ∧ days_needed = 10 :=
by {
  use 10,
  split,
  { simp,
    norm_num,
  },
  { simp }
}

end ivy_stripping_days_l107_107988


namespace equal_area_intersection_l107_107552

variable (p q r s : ℚ)
noncomputable def intersection_point (x y : ℚ) : Prop :=
  4 * x + 5 * p / q = 12 * p / q ∧ 8 * y = p 

theorem equal_area_intersection :
  intersection_point p q r s /\
  p + q + r + s = 60 := 
by 
  sorry

end equal_area_intersection_l107_107552


namespace carter_students_received_grades_l107_107523

theorem carter_students_received_grades
  (students_thompson : ℕ)
  (a_thompson : ℕ)
  (remaining_students_thompson : ℕ)
  (b_thompson : ℕ)
  (students_carter : ℕ)
  (ratio_A_thompson : ℚ)
  (ratio_B_thompson : ℚ)
  (A_carter : ℕ)
  (B_carter : ℕ) :
  students_thompson = 20 →
  a_thompson = 12 →
  remaining_students_thompson = 8 →
  b_thompson = 5 →
  students_carter = 30 →
  ratio_A_thompson = (a_thompson : ℚ) / students_thompson →
  ratio_B_thompson = (b_thompson : ℚ) / remaining_students_thompson →
  A_carter = ratio_A_thompson * students_carter →
  B_carter = (b_thompson : ℚ) / remaining_students_thompson * (students_carter - A_carter) →
  A_carter = 18 ∧ B_carter = 8 := 
by 
  intros;
  sorry

end carter_students_received_grades_l107_107523


namespace judson_contribution_l107_107702

theorem judson_contribution (J K C : ℝ) (hK : K = 1.20 * J) (hC : C = K + 200) (h_total : J + K + C = 1900) : J = 500 :=
by
  -- This is where the proof would go, but we are skipping it as per the instructions.
  sorry

end judson_contribution_l107_107702


namespace combined_rainfall_is_23_l107_107249

-- Define the conditions
def monday_hours : ℕ := 7
def monday_rate : ℕ := 1
def tuesday_hours : ℕ := 4
def tuesday_rate : ℕ := 2
def wednesday_hours : ℕ := 2
def wednesday_rate (tuesday_rate : ℕ) : ℕ := 2 * tuesday_rate

-- Calculate the rainfalls
def monday_rainfall : ℕ := monday_hours * monday_rate
def tuesday_rainfall : ℕ := tuesday_hours * tuesday_rate
def wednesday_rainfall (wednesday_rate : ℕ) : ℕ := wednesday_hours * wednesday_rate

-- Define the total rainfall
def total_rainfall : ℕ :=
  monday_rainfall + tuesday_rainfall + wednesday_rainfall (wednesday_rate tuesday_rate)

theorem combined_rainfall_is_23 : total_rainfall = 23 := by
  -- Proof to be filled in
  sorry

end combined_rainfall_is_23_l107_107249


namespace crayons_in_judahs_box_l107_107254

theorem crayons_in_judahs_box (karen_crayons beatrice_crayons gilbert_crayons judah_crayons : ℕ)
  (h1 : karen_crayons = 128)
  (h2 : beatrice_crayons = karen_crayons / 2)
  (h3 : gilbert_crayons = beatrice_crayons / 2)
  (h4 : judah_crayons = gilbert_crayons / 4) :
  judah_crayons = 8 :=
by {
  sorry
}

end crayons_in_judahs_box_l107_107254


namespace product_of_points_l107_107457

def f (n : ℕ) : ℕ :=
  if n % 6 = 0 then 6
  else if n % 3 = 0 then 3
  else if n % 2 = 0 then 2
  else 1

def allie_rolls := [5, 6, 1, 2, 3]
def betty_rolls := [6, 1, 1, 2, 3]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.foldl (fun acc n => acc + f n) 0

theorem product_of_points :
  total_points allie_rolls * total_points betty_rolls = 169 :=
by
  sorry

end product_of_points_l107_107457


namespace powers_of_i_l107_107999

theorem powers_of_i (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) : 
  i^22 + i^222 = -2 :=
by {
  -- Proof will go here
  sorry
}

end powers_of_i_l107_107999


namespace counterexample_to_proposition_l107_107398

theorem counterexample_to_proposition : ∃ (a : ℝ), a^2 > 0 ∧ a ≤ 0 :=
  sorry

end counterexample_to_proposition_l107_107398


namespace line_circle_no_intersect_l107_107855

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l107_107855


namespace decagon_adjacent_vertex_probability_l107_107124

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l107_107124


namespace find_a10_l107_107547

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ := n * a1 + (n * (n - 1) / 2) * d

theorem find_a10 (a1 d : ℝ)
  (h1 : a_n a1 d 2 + a_n a1 d 4 = 2)
  (h2 : S_n a1 d 2 + S_n a1 d 4 = 1) :
  a_n a1 d 10 = 8 :=
sorry

end find_a10_l107_107547


namespace systematic_sampling_interval_l107_107076

-- Definitions for the given conditions
def total_students : ℕ := 1203
def sample_size : ℕ := 40

-- Theorem statement to be proven
theorem systematic_sampling_interval (N n : ℕ) (hN : N = total_students) (hn : n = sample_size) : 
  N % n ≠ 0 → ∃ k : ℕ, k = 30 :=
by
  sorry

end systematic_sampling_interval_l107_107076


namespace domain_and_range_of_g_l107_107262

noncomputable def f : ℝ → ℝ := sorry-- Given: a function f with domain [0,2] and range [0,1]
noncomputable def g (x : ℝ) := 1 - f (x / 2 + 1)

theorem domain_and_range_of_g :
  let dom_g := { x | -2 ≤ x ∧ x ≤ 2 }
  let range_g := { y | 0 ≤ y ∧ y ≤ 1 }
  ∀ (x : ℝ), (x ∈ dom_g → (g x) ∈ range_g) := 
sorry

end domain_and_range_of_g_l107_107262


namespace min_squares_to_cover_staircase_l107_107594

-- Definition of the staircase and the constraints
def is_staircase (n : ℕ) (s : ℕ → ℕ) : Prop :=
  ∀ i, i < n → s i = i + 1

-- The proof problem statement
theorem min_squares_to_cover_staircase : 
  ∀ n : ℕ, n = 15 →
  ∀ s : ℕ → ℕ, is_staircase n s →
  ∃ k : ℕ, k = 15 ∧ (∀ i, i < n → ∃ a b : ℕ, a ≤ i ∧ b ≤ s a ∧ ∃ (l : ℕ), l = 1) :=
by
  sorry

end min_squares_to_cover_staircase_l107_107594


namespace range_of_k_l107_107356

noncomputable def f (k : ℝ) (x : ℝ) := 1 - k * x^2
noncomputable def g (x : ℝ) := Real.cos x

theorem range_of_k (k : ℝ) : (∀ x : ℝ, f k x < g x) ↔ k ≥ (1 / 2) :=
by
  sorry

end range_of_k_l107_107356


namespace coordinates_of_point_l107_107910

theorem coordinates_of_point (x y : ℝ) (h : (x, y) = (-2, 3)) : (x, y) = (-2, 3) :=
by
  exact h

end coordinates_of_point_l107_107910


namespace A_wins_B_no_more_than_two_throws_C_treats_after_two_throws_C_treats_exactly_two_days_l107_107330

def prob_A_wins_B_one_throw : ℚ := 1 / 3
def prob_tie_one_throw : ℚ := 1 / 3
def prob_A_wins_B_no_more_2_throws : ℚ := 4 / 9

def prob_C_treats_two_throws : ℚ := 2 / 9

def prob_C_treats_exactly_2_days_out_of_3 : ℚ := 28 / 243

theorem A_wins_B_no_more_than_two_throws (P1 : ℚ := prob_A_wins_B_one_throw) (P2 : ℚ := prob_tie_one_throw) :
  P1 + P2 * P1 = prob_A_wins_B_no_more_2_throws := 
by
  sorry

theorem C_treats_after_two_throws : prob_tie_one_throw ^ 2 = prob_C_treats_two_throws :=
by
  sorry

theorem C_treats_exactly_two_days (n : ℕ := 3) (k : ℕ := 2) (p_success : ℚ := prob_C_treats_two_throws) :
  (n.choose k) * (p_success ^ k) * ((1 - p_success) ^ (n - k)) = prob_C_treats_exactly_2_days_out_of_3 :=
by
  sorry

end A_wins_B_no_more_than_two_throws_C_treats_after_two_throws_C_treats_exactly_two_days_l107_107330


namespace tn_range_l107_107189

noncomputable def a (n : ℕ) : ℚ :=
  (2 * n - 1) / 10

noncomputable def b (n : ℕ) : ℚ :=
  2^(n - 1)

noncomputable def c (n : ℕ) : ℚ :=
  (1 + a n) / (4 * b n)

noncomputable def T (n : ℕ) : ℚ :=
  (1 / 10) * (2 - (n + 2) / (2^n)) + (9 / 20) * (2 - 1 / (2^(n-1)))

theorem tn_range (n : ℕ) : (101 / 400 : ℚ) ≤ T n ∧ T n < (103 / 200 : ℚ) :=
sorry

end tn_range_l107_107189


namespace ordered_triples_count_eq_4_l107_107995

theorem ordered_triples_count_eq_4 :
  ∃ (S : Finset (ℝ × ℝ × ℝ)), 
    (∀ x y z : ℝ, (x, y, z) ∈ S ↔ (x ≠ 0) ∧ (y ≠ 0) ∧ (z ≠ 0) ∧ (xy + 1 = z) ∧ (yz + 1 = x) ∧ (zx + 1 = y)) ∧
    S.card = 4 :=
sorry

end ordered_triples_count_eq_4_l107_107995


namespace geometric_sequence_sum_l107_107351

theorem geometric_sequence_sum {a : ℕ → ℤ} (r : ℤ) (h1 : a 1 = 1) (h2 : r = -2) 
(h3 : ∀ n, a (n + 1) = a n * r) : 
  a 1 + |a 2| + |a 3| + a 4 = 15 := 
by sorry

end geometric_sequence_sum_l107_107351


namespace exists_naturals_l107_107544

def sum_of_digits (a : ℕ) : ℕ := sorry

theorem exists_naturals (R : ℕ) (hR : R > 0) :
  ∃ n : ℕ, n > 0 ∧ (sum_of_digits (n^2)) / (sum_of_digits n) = R :=
by
  sorry

end exists_naturals_l107_107544


namespace maximum_absolute_sum_l107_107026

theorem maximum_absolute_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : |x| + |y| + |z| ≤ 2 :=
sorry

end maximum_absolute_sum_l107_107026


namespace proof_problem_l107_107057

-- Let P, Q, R be points on a circle of radius s
-- Given: PQ = PR, PQ > s, and minor arc QR is 2s
-- Prove: PQ / QR = sin(1)

noncomputable def point_on_circle (s : ℝ) : ℝ → ℝ × ℝ := sorry
def radius {s : ℝ} (P Q : ℝ × ℝ ) : Prop := dist P Q = s

theorem proof_problem (s : ℝ) (P Q R : ℝ × ℝ)
  (hPQ : dist P Q = dist P R)
  (hPQ_gt_s : dist P Q > s)
  (hQR_arc_len : 1 = s) :
  dist P Q / (2 * s) = Real.sin 1 := 
sorry

end proof_problem_l107_107057


namespace determine_real_numbers_l107_107630

theorem determine_real_numbers (x y : ℝ) (h1 : x + y = 1) (h2 : x^3 + y^3 = 19) :
    (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) :=
sorry

end determine_real_numbers_l107_107630


namespace number_of_months_in_martian_calendar_l107_107532

theorem number_of_months_in_martian_calendar
  (x y : ℕ) 
  (h1 : 100 * x + 77 * y = 5882) 
  (h2 : x + y = 74) :
  x + y = 74 := 
by
  sorry

end number_of_months_in_martian_calendar_l107_107532


namespace sheets_bought_l107_107136

variable (x y : ℕ)

-- Conditions based on the problem statement
def A_condition (x y : ℕ) : Prop := x + 40 = y
def B_condition (x y : ℕ) : Prop := 3 * x + 40 = y

-- Proven that if these conditions are met, then the number of sheets of stationery bought by A and B is 120
theorem sheets_bought (x y : ℕ) (hA : A_condition x y) (hB : B_condition x y) : y = 120 :=
by
  sorry

end sheets_bought_l107_107136


namespace eval_expression_l107_107402

theorem eval_expression : (2^5 - 5^2) = 7 :=
by {
  -- Proof steps will be here
  sorry
}

end eval_expression_l107_107402


namespace tin_silver_ratio_l107_107419

theorem tin_silver_ratio (T S : ℝ) 
  (h1 : T + S = 50) 
  (h2 : 0.1375 * T + 0.075 * S = 5) : 
  T / S = 2 / 3 :=
by
  sorry

end tin_silver_ratio_l107_107419


namespace pauline_total_spent_l107_107388

variable {items_total : ℝ} (discount_rate : ℝ) (discount_limit : ℝ) (sales_tax_rate : ℝ)

def total_spent (items_total discount_rate discount_limit sales_tax_rate : ℝ) : ℝ :=
  let discount_amount := discount_rate * discount_limit
  let discounted_total := discount_limit - discount_amount
  let non_discounted_total := items_total - discount_limit
  let subtotal := discounted_total + non_discounted_total
  let sales_tax := sales_tax_rate * subtotal
  subtotal + sales_tax

theorem pauline_total_spent :
  total_spent 250 0.15 100 0.08 = 253.80 :=
by
  sorry

end pauline_total_spent_l107_107388


namespace fuel_tank_capacity_l107_107784

theorem fuel_tank_capacity (C : ℝ) (h1 : 0.12 * 106 + 0.16 * (C - 106) = 30) : C = 214 :=
by
  sorry

end fuel_tank_capacity_l107_107784


namespace expand_binomials_l107_107637

theorem expand_binomials : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by 
  sorry

end expand_binomials_l107_107637


namespace factor_expression_l107_107173

theorem factor_expression (x : ℝ) : 46 * x^3 - 115 * x^7 = -23 * x^3 * (5 * x^4 - 2) := 
by
  sorry

end factor_expression_l107_107173


namespace line_circle_no_intersection_l107_107872

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l107_107872


namespace remainder_5n_minus_12_l107_107756

theorem remainder_5n_minus_12 (n : ℤ) (hn : n % 9 = 4) : (5 * n - 12) % 9 = 8 := 
by sorry

end remainder_5n_minus_12_l107_107756


namespace greatest_odd_factors_below_200_l107_107719

theorem greatest_odd_factors_below_200 :
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (square m → m ≤ n)) ∧ n = 196 :=
by
sorry

end greatest_odd_factors_below_200_l107_107719


namespace squares_on_grid_l107_107785

-- Defining the problem conditions
def grid_size : ℕ := 5
def total_points : ℕ := grid_size * grid_size
def used_points : ℕ := 20

-- Stating the theorem to prove the total number of squares formed
theorem squares_on_grid : 
  (total_points = 25) ∧ (used_points = 20) →
  (∃ all_squares : ℕ, all_squares = 21) :=
by
  intros
  sorry

end squares_on_grid_l107_107785


namespace compute_expression_l107_107794

theorem compute_expression : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end compute_expression_l107_107794


namespace Dad_steps_l107_107627

variable (d m y : ℕ)

-- Conditions
def condition_1 : Prop := d = 3 → m = 5
def condition_2 : Prop := m = 3 → y = 5
def condition_3 : Prop := m + y = 400

-- Question and Answer
theorem Dad_steps : condition_1 d m → condition_2 m y → condition_3 m y → d = 90 :=
by
  intros
  sorry

end Dad_steps_l107_107627


namespace min_value_of_expression_l107_107479

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 - Real.sqrt 3 * |x| + 1) + Real.sqrt (x^2 + Real.sqrt 3 * |x| + 3)

theorem min_value_of_expression : 
  (∀ x : ℝ, f x ≥ Real.sqrt 7) ∧ (∀ x : ℝ, f x = Real.sqrt 7 → x = Real.sqrt 3 / 4 ∨ x = -Real.sqrt 3 / 4) :=
sorry

end min_value_of_expression_l107_107479


namespace line_circle_no_intersection_l107_107873

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l107_107873


namespace max_f_theta_l107_107352

noncomputable def determinant (a b c d : ℝ) : ℝ := a*d - b*c

noncomputable def f (θ : ℝ) : ℝ :=
  determinant (Real.sin θ) (Real.cos θ) (-1) 1

theorem max_f_theta :
  ∀ θ : ℝ, 0 < θ ∧ θ < (Real.pi / 3) →
  f θ ≤ (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end max_f_theta_l107_107352


namespace triangle_area_290_l107_107554

theorem triangle_area_290 
  (P Q R : ℝ × ℝ)
  (h1 : (R.1 - P.1) * (R.1 - Q.1) + (R.2 - P.2) * (R.2 - Q.2) = 0) -- Right triangle condition
  (h2 : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 50^2) -- Length of hypotenuse PQ
  (h3 : ∀ x: ℝ, (x, x - 2) = P) -- Median through P
  (h4 : ∀ x: ℝ, (x, 3 * x + 3) = Q) -- Median through Q
  :
  ∃ (area : ℝ), area = 290 := 
sorry

end triangle_area_290_l107_107554


namespace judah_crayons_l107_107253

theorem judah_crayons (karen beatrice gilbert judah : ℕ) 
  (h1 : karen = 128)
  (h2 : karen = 2 * beatrice)
  (h3 : beatrice = 2 * gilbert)
  (h4 : gilbert = 4 * judah) : 
  judah = 8 :=
by
  sorry

end judah_crayons_l107_107253


namespace lucas_150_mod_9_l107_107071

-- Define the Lucas sequence recursively
def lucas (n : ℕ) : ℕ :=
  match n with
  | 0 => 1 -- Since L_1 in the sequence provided is actually the first Lucas number (index starts from 1)
  | 1 => 3
  | (n + 2) => lucas n + lucas (n + 1)

-- Define the theorem for the remainder when the 150th term is divided by 9
theorem lucas_150_mod_9 : lucas 149 % 9 = 3 := by
  sorry

end lucas_150_mod_9_l107_107071


namespace sunil_investment_l107_107066

noncomputable def total_amount (P : ℝ) : ℝ :=
  let r1 := 0.025  -- 5% per annum compounded semi-annually
  let r2 := 0.03   -- 6% per annum compounded semi-annually
  let A2 := P * (1 + r1) * (1 + r1)
  let A3 := (A2 + 0.5 * P) * (1 + r2)
  let A4 := A3 * (1 + r2)
  A4

theorem sunil_investment (P : ℝ) : total_amount P = 1.645187625 * P :=
by
  sorry

end sunil_investment_l107_107066


namespace fraction_one_bedroom_apartments_l107_107907

theorem fraction_one_bedroom_apartments :
  ∃ x : ℝ, (x + 0.33 = 0.5) ∧ x = 0.17 :=
by
  sorry

end fraction_one_bedroom_apartments_l107_107907


namespace apples_harvested_l107_107956

theorem apples_harvested (weight_juice weight_restaurant weight_per_bag sales_price total_sales : ℤ) 
  (h1 : weight_juice = 90) 
  (h2 : weight_restaurant = 60) 
  (h3 : weight_per_bag = 5) 
  (h4 : sales_price = 8) 
  (h5 : total_sales = 408) : 
  (weight_juice + weight_restaurant + (total_sales / sales_price) * weight_per_bag = 405) :=
by
  sorry

end apples_harvested_l107_107956


namespace find_k_and_prove_geometric_sequence_l107_107658

/-
Given conditions:
1. Sequence sa : ℕ → ℝ with sum sequence S : ℕ → ℝ satisfying the recurrence relation S (n + 1) = (k + 1) * S n + 2
2. Initial terms a_1 = 2 and a_2 = 1
-/

def sequence_sum_relation (S : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n : ℕ, S (n + 1) = (k + 1) * S n + 2

def init_sequence_terms (a : ℕ → ℝ) : Prop :=
a 1 = 2 ∧ a 2 = 1

/-
Proof goal:
1. Prove k = -1/2 given the conditions.
2. Prove sequence a is a geometric sequence with common ratio 1/2 given the conditions.
-/

theorem find_k_and_prove_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ) :
  sequence_sum_relation S k →
  init_sequence_terms a →
  (k = (-1:ℝ)/2) ∧ (∀ n: ℕ, n ≥ 1 → a (n+1) = (1/2) * a n) :=
by
  sorry

end find_k_and_prove_geometric_sequence_l107_107658


namespace ganesh_average_speed_l107_107600

variable (D : ℝ) (hD : D > 0)

/-- Ganesh's average speed over the entire journey is 45 km/hr.
    Given:
    - Speed from X to Y is 60 km/hr
    - Speed from Y to X is 36 km/hr
--/
theorem ganesh_average_speed :
  let T1 := D / 60
  let T2 := D / 36
  let total_distance := 2 * D
  let total_time := T1 + T2
  (total_distance / total_time) = 45 :=
by
  sorry

end ganesh_average_speed_l107_107600


namespace a_pow_10_add_b_pow_10_eq_123_l107_107723

variable (a b : ℕ) -- better as non-negative integers for sequence progression

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem a_pow_10_add_b_pow_10_eq_123 : a^10 + b^10 = 123 := by
  sorry

end a_pow_10_add_b_pow_10_eq_123_l107_107723


namespace can_form_all_numbers_l107_107460

noncomputable def domino_tiles : List (ℕ × ℕ) := [(1, 3), (6, 6), (6, 2), (3, 2)]

def form_any_number (n : ℕ) : Prop :=
  ∃ (comb : List (ℕ × ℕ)), comb ⊆ domino_tiles ∧ (comb.bind (λ p => [p.1, p.2])).sum = n

theorem can_form_all_numbers : ∀ n, 1 ≤ n → n ≤ 23 → form_any_number n :=
by sorry

end can_form_all_numbers_l107_107460


namespace jellybeans_initial_amount_l107_107803

theorem jellybeans_initial_amount (x : ℝ) 
  (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_initial_amount_l107_107803


namespace Warriors_won_25_games_l107_107395

def CricketResults (Sharks Falcons Warriors Foxes Knights : ℕ) :=
  Sharks > Falcons ∧
  (Warriors > Foxes ∧ Warriors < Knights) ∧
  Foxes > 15 ∧
  (Foxes = 20 ∨ Foxes = 25 ∨ Foxes = 30) ∧
  (Warriors = 20 ∨ Warriors = 25 ∨ Warriors = 30) ∧
  (Knights = 20 ∨ Knights = 25 ∨ Knights = 30)

theorem Warriors_won_25_games (Sharks Falcons Warriors Foxes Knights : ℕ) 
  (h : CricketResults Sharks Falcons Warriors Foxes Knights) :
  Warriors = 25 :=
by
  sorry

end Warriors_won_25_games_l107_107395


namespace variance_of_red_ball_draws_l107_107937

noncomputable def variance_red_ball_draws : ℚ :=
let n := 3
let p := (2 : ℚ) / 3
in n * p * (1 - p)

theorem variance_of_red_ball_draws :
  variance_red_ball_draws = (2 : ℚ) / 3 :=
by
  -- We assume the conditions in the problem
  let n := 3
  let p := (2 : ℚ) / 3
  have h1 : variance_red_ball_draws = n * p * (1 - p) := rfl
  rw [h1]
  sorry

end variance_of_red_ball_draws_l107_107937


namespace parabola_through_origin_l107_107734

theorem parabola_through_origin {a b c : ℝ} :
  (c = 0 ↔ ∀ x, (0, 0) = (x, a * x^2 + b * x + c)) :=
sorry

end parabola_through_origin_l107_107734


namespace paige_mp3_player_songs_l107_107056

/--
Paige had 11 songs on her mp3 player.
She deleted 9 old songs.
She added 8 new songs.

We are to prove:
- The final number of songs on her mp3 player is 10.
-/
theorem paige_mp3_player_songs (initial_songs deleted_songs added_songs final_songs : ℕ)
  (h₁ : initial_songs = 11)
  (h₂ : deleted_songs = 9)
  (h₃ : added_songs = 8) :
  final_songs = initial_songs - deleted_songs + added_songs :=
by
  sorry

end paige_mp3_player_songs_l107_107056


namespace greg_spent_on_shirt_l107_107300

-- Define the conditions in Lean
variables (S H : ℤ)
axiom condition1 : H = 2 * S + 9
axiom condition2 : S + H = 300

-- State the theorem to prove
theorem greg_spent_on_shirt : S = 97 :=
by
  sorry

end greg_spent_on_shirt_l107_107300


namespace third_day_sales_correct_l107_107312

variable (a : ℕ)

def firstDaySales := a
def secondDaySales := a + 4
def thirdDaySales := 2 * (a + 4) - 7
def expectedSales := 2 * a + 1

theorem third_day_sales_correct : thirdDaySales a = expectedSales a :=
by
  -- Main proof goes here
  sorry

end third_day_sales_correct_l107_107312


namespace books_left_correct_l107_107049

variable (initial_books : ℝ) (sold_books : ℝ)

def number_of_books_left (initial_books sold_books : ℝ) : ℝ :=
  initial_books - sold_books

theorem books_left_correct :
  number_of_books_left 51.5 45.75 = 5.75 :=
by
  sorry

end books_left_correct_l107_107049


namespace maximum_n_l107_107474

theorem maximum_n (n : ℕ) (G : SimpleGraph (Fin n)) :
  (∃ (A : Fin n → Set (Fin 2020)),  ∀ i j, (G.Adj i j ↔ (A i ∩ A j ≠ ∅)) →
  n ≤ 89) := sorry

end maximum_n_l107_107474


namespace find_f_of_2011_l107_107829

-- Define the function f
def f (x : ℝ) (a b c : ℝ) := a * x^5 + b * x^3 + c * x + 7

-- The main statement we need to prove
theorem find_f_of_2011 (a b c : ℝ) (h : f (-2011) a b c = -17) : f 2011 a b c = 31 :=
by
  sorry

end find_f_of_2011_l107_107829


namespace log2_a_plus_log2_b_zero_l107_107029

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_a_plus_log2_b_zero 
    (a b : ℝ) 
    (h : (Nat.choose 6 3) * (a^3) * (b^3) = 20) 
    (hc : (a^2 + b / a)^(3) = 20 * x^(3)) :
  log2 a + log2 b = 0 :=
by
  sorry

end log2_a_plus_log2_b_zero_l107_107029


namespace div_relation_l107_107360

theorem div_relation (a b c : ℚ) (h1 : a / b = 3) (h2 : b / c = 2 / 3) : c / a = 1 / 2 := 
by 
  sorry

end div_relation_l107_107360


namespace jake_snake_sales_l107_107698

theorem jake_snake_sales 
  (num_snakes : ℕ)
  (eggs_per_snake : ℕ)
  (regular_price : ℕ)
  (super_rare_multiplier : ℕ)
  (num_snakes = 3)
  (eggs_per_snake = 2)
  (regular_price = 250)
  (super_rare_multiplier = 4) : 
  (num_snakes * eggs_per_snake - 1) * regular_price + regular_price * super_rare_multiplier = 2250 :=
sorry

end jake_snake_sales_l107_107698


namespace number_of_sides_possibilities_l107_107213

theorem number_of_sides_possibilities (a b : ℕ) (h₁ : a = 8) (h₂ : b = 5) : 
  { x : ℕ // 3 < x ∧ x < 13 }.to_list.length = 9 :=
by
  sorry

end number_of_sides_possibilities_l107_107213


namespace jellybean_count_l107_107805

theorem jellybean_count (x : ℝ) (h : (0.75^3) * x = 27) : x = 64 :=
sorry

end jellybean_count_l107_107805


namespace div_z_x_l107_107228

variables (x y z : ℚ)

theorem div_z_x (h1 : x / y = 3) (h2 : y / z = 5 / 2) : z / x = 2 / 15 :=
sorry

end div_z_x_l107_107228


namespace like_terms_value_l107_107502

theorem like_terms_value (a b : ℤ) (h1 : a + b = 2) (h2 : a - 1 = 1) : a - b = 2 :=
sorry

end like_terms_value_l107_107502


namespace smallest_possible_value_of_c_l107_107059

theorem smallest_possible_value_of_c (b c : ℝ) (h1 : 1 < b) (h2 : b < c)
    (h3 : ¬∃ (u v w : ℝ), u = 1 ∧ v = b ∧ w = c ∧ u + v > w ∧ u + w > v ∧ v + w > u)
    (h4 : ¬∃ (x y z : ℝ), x = 1 ∧ y = 1/b ∧ z = 1/c ∧ x + y > z ∧ x + z > y ∧ y + z > x) :
    c = (5 + Real.sqrt 5) / 2 :=
by
  sorry

end smallest_possible_value_of_c_l107_107059


namespace range_of_f_l107_107896

open Real

noncomputable def f (x : ℝ) : ℝ := (sqrt 3) * sin x + cos x

theorem range_of_f :
  ∀ x : ℝ, -π/2 ≤ x ∧ x ≤ π/2 → - (sqrt 3) ≤ f x ∧ f x ≤ 2 := by
  sorry

end range_of_f_l107_107896


namespace train_speed_l107_107133

theorem train_speed (d t : ℝ) (h1 : d = 500) (h2 : t = 3) : d / t = 166.67 := by
  sorry

end train_speed_l107_107133


namespace triangle_side_lengths_count_l107_107200

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end triangle_side_lengths_count_l107_107200


namespace bronson_yellow_leaves_l107_107619

-- Bronson collects 12 leaves on Thursday
def leaves_thursday : ℕ := 12

-- Bronson collects 13 leaves on Friday
def leaves_friday : ℕ := 13

-- 20% of the leaves are Brown (as a fraction)
def percent_brown : ℚ := 0.2

-- 20% of the leaves are Green (as a fraction)
def percent_green : ℚ := 0.2

theorem bronson_yellow_leaves : 
  (leaves_thursday + leaves_friday) * (1 - percent_brown - percent_green) = 15 := by
sorry

end bronson_yellow_leaves_l107_107619


namespace competition_result_l107_107568

variables (Olya Oleg Pasha : ℕ)

theorem competition_result 
  (h1 : Olya ≠ 1 → Olya ≠ 3 → False)
  (h2 : (Oleg = 1 ∨ Oleg = 3) → Olya = 3)
  (h3 : (Oleg ≠ 1 → (Olya = 2 ∨ Olya = 3)))
  (h4 : Olya ≠ 1 ∧ Oleg ≠ 2 ∧ Pasha ≠ 3) :
  Oleg = 1 ∧ Pasha = 2 ∧ Olya = 3 := 
by {
  sorry
}

end competition_result_l107_107568


namespace TruckY_average_speed_is_63_l107_107078

noncomputable def average_speed_TruckY (initial_gap : ℕ) (extra_distance : ℕ) (hours : ℕ) (distance_X_per_hour : ℕ) : ℕ :=
  let distance_X := distance_X_per_hour * hours
  let total_distance_Y := distance_X + initial_gap + extra_distance
  total_distance_Y / hours

theorem TruckY_average_speed_is_63 
  (initial_gap : ℕ := 14) 
  (extra_distance : ℕ := 4) 
  (hours : ℕ := 3)
  (distance_X_per_hour : ℕ := 57) : 
  average_speed_TruckY initial_gap extra_distance hours distance_X_per_hour = 63 :=
by
  -- Proof goes here
  sorry

end TruckY_average_speed_is_63_l107_107078


namespace no_real_intersections_l107_107842

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l107_107842


namespace set_union_inter_proof_l107_107089

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem set_union_inter_proof : A ∪ B = {0, 1, 2, 3} ∧ A ∩ B = {1, 2} := by
  sorry

end set_union_inter_proof_l107_107089


namespace opposite_of_a_is_2_l107_107367

theorem opposite_of_a_is_2 (a : ℤ) (h : -a = 2) : a = -2 := 
by
  -- proof to be provided
  sorry

end opposite_of_a_is_2_l107_107367


namespace paco_ate_more_salty_than_sweet_l107_107928

-- Define the initial conditions
def sweet_start := 8
def salty_start := 6
def sweet_ate := 20
def salty_ate := 34

-- Define the statement to prove
theorem paco_ate_more_salty_than_sweet : (salty_ate - sweet_ate) = 14 := by
    sorry

end paco_ate_more_salty_than_sweet_l107_107928


namespace scale_of_diagram_l107_107144

-- Definitions for the given conditions
def length_miniature_component_mm : ℕ := 4
def length_diagram_cm : ℕ := 8
def length_diagram_mm : ℕ := 80  -- Converted length from cm to mm

-- The problem statement
theorem scale_of_diagram :
  (length_diagram_mm : ℕ) / (length_miniature_component_mm : ℕ) = 20 :=
by
  have conversion : length_diagram_mm = length_diagram_cm * 10 := by sorry
  -- conversion states the formula for converting cm to mm
  have ratio : length_diagram_mm / length_miniature_component_mm = 80 / 4 := by sorry
  -- ratio states the initial computed ratio
  exact sorry

end scale_of_diagram_l107_107144


namespace rectangle_constant_k_l107_107285

theorem rectangle_constant_k (d : ℝ) (x : ℝ) (h_ratio : 4 * x = (4 / 3) * (3 * x)) (h_diagonal : d^2 = (4 * x)^2 + (3 * x)^2) : 
  ∃ k : ℝ, k = 12 / 25 ∧ (4 * x) * (3 * x) = k * d^2 := 
sorry

end rectangle_constant_k_l107_107285


namespace problem1_problem2_l107_107492

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - x - 2 ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - x - m^2 - m ≤ 0

-- Problem 1: If ¬p is true, find the range of values for x
theorem problem1 {x : ℝ} (h : ¬ p x) : x > 2 ∨ x < -1 :=
by
  -- Proof omitted
  sorry

-- Problem 2: If ¬q is a sufficient but not necessary condition for ¬p, find the range of values for m
theorem problem2 {m : ℝ} (h : ∀ x : ℝ, ¬ q x m → ¬ p x) : m > 1 ∨ m < -2 :=
by
  -- Proof omitted
  sorry

end problem1_problem2_l107_107492


namespace competition_result_l107_107567

variables (Olya Oleg Pasha : ℕ)

theorem competition_result 
  (h1 : Olya ≠ 1 → Olya ≠ 3 → False)
  (h2 : (Oleg = 1 ∨ Oleg = 3) → Olya = 3)
  (h3 : (Oleg ≠ 1 → (Olya = 2 ∨ Olya = 3)))
  (h4 : Olya ≠ 1 ∧ Oleg ≠ 2 ∧ Pasha ≠ 3) :
  Oleg = 1 ∧ Pasha = 2 ∧ Olya = 3 := 
by {
  sorry
}

end competition_result_l107_107567


namespace victoria_worked_weeks_l107_107750

-- Definitions for given conditions
def hours_worked_per_day : ℕ := 9
def total_hours_worked : ℕ := 315
def days_in_week : ℕ := 7

-- Main theorem to prove
theorem victoria_worked_weeks : total_hours_worked / hours_worked_per_day / days_in_week = 5 :=
by
  sorry

end victoria_worked_weeks_l107_107750


namespace sin_double_angle_l107_107826

theorem sin_double_angle (x : ℝ) (h : Real.cos (π / 4 - x) = -3 / 5) : Real.sin (2 * x) = -7 / 25 :=
by
  sorry

end sin_double_angle_l107_107826


namespace number_of_primary_schools_l107_107037

theorem number_of_primary_schools (A B total : ℕ) (h1 : A = 2 * 400)
  (h2 : B = 2 * 340) (h3 : total = 1480) (h4 : total = A + B) :
  2 + 2 = 4 :=
by
  sorry

end number_of_primary_schools_l107_107037


namespace gum_lcm_l107_107264

theorem gum_lcm (strawberry blueberry cherry : ℕ) (h₁ : strawberry = 6) (h₂ : blueberry = 5) (h₃ : cherry = 8) :
  Nat.lcm (Nat.lcm strawberry blueberry) cherry = 120 :=
by
  rw [h₁, h₂, h₃]
  -- LCM(6, 5, 8) = LCM(LCM(6, 5), 8)
  sorry

end gum_lcm_l107_107264


namespace area_hexagon_DEFD_EFE_l107_107283

variable (D E F D' E' F' : Type)
variable (perimeter_DEF : ℝ) (radius_circumcircle : ℝ)
variable (area_hexagon : ℝ)

theorem area_hexagon_DEFD_EFE' (h1 : perimeter_DEF = 42)
    (h2 : radius_circumcircle = 7)
    (h_def : area_hexagon = 147) :
  area_hexagon = 147 := 
sorry

end area_hexagon_DEFD_EFE_l107_107283


namespace average_minutes_run_per_day_l107_107034

-- Define the given averages for each grade
def sixth_grade_avg : ℕ := 10
def seventh_grade_avg : ℕ := 18
def eighth_grade_avg : ℕ := 12

-- Define the ratios of the number of students in each grade
def num_sixth_eq_three_times_num_seventh (num_seventh : ℕ) : ℕ := 3 * num_seventh
def num_eighth_eq_half_num_seventh (num_seventh : ℕ) : ℕ := num_seventh / 2

-- Average number of minutes run per day by all students
theorem average_minutes_run_per_day (num_seventh : ℕ) :
  (sixth_grade_avg * num_sixth_eq_three_times_num_seventh num_seventh +
   seventh_grade_avg * num_seventh +
   eighth_grade_avg * num_eighth_eq_half_num_seventh num_seventh) / 
  (num_sixth_eq_three_times_num_seventh num_seventh + 
   num_seventh + 
   num_eighth_eq_half_num_seventh num_seventh) = 12 := 
sorry

end average_minutes_run_per_day_l107_107034


namespace possible_ways_to_choose_gates_l107_107976

theorem possible_ways_to_choose_gates : 
  ∃! (ways : ℕ), ways = 20 := 
by
  sorry

end possible_ways_to_choose_gates_l107_107976


namespace tan_beta_value_l107_107834

theorem tan_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.tan α = 4 / 3)
  (h4 : Real.cos (α + β) = Real.sqrt 5 / 5) :
  Real.tan β = 2 / 11 := 
sorry

end tan_beta_value_l107_107834


namespace quadratic_sum_is_zero_l107_107738

-- Definition of a quadratic function with given conditions and the final result to prove
theorem quadratic_sum_is_zero {a b c : ℝ} 
  (h₁ : ∀ x : ℝ, x = 3 → a * (x - 1) * (x - 5) = 36) 
  (h₂ : a * 1^2 + b * 1 + c = 0) 
  (h₃ : a * 5^2 + b * 5 + c = 0) : 
  a + b + c = 0 := 
sorry

end quadratic_sum_is_zero_l107_107738


namespace consecutive_numbers_N_l107_107424

theorem consecutive_numbers_N (N : ℕ) (h : ∀ k, 0 < k → k < 15 → N + k < 81) : N = 66 :=
sorry

end consecutive_numbers_N_l107_107424


namespace line_circle_no_intersection_l107_107876

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l107_107876


namespace value_of_m_l107_107631

theorem value_of_m : 5^2 + 7 = 4^3 + m → m = -32 :=
by
  intro h
  sorry

end value_of_m_l107_107631


namespace greatest_perfect_square_below_200_l107_107717

theorem greatest_perfect_square_below_200 : 
  ∃ n : ℕ, n < 200 ∧ (∀ m : ℕ, m < 200 → (∃ k : ℕ, m = k^2) → m ≤ n) ∧ (∃ k : ℕ, n = k^2) := 
by 
  use 196, 14
  split
  -- 196 is less than 200, and 196 is a perfect square
  {
    exact 196 < 200,
    use 14,
    exact 196 = 14^2,
  },
  -- no perfect square less than 200 is greater than 196
  sorry

end greatest_perfect_square_below_200_l107_107717


namespace minimum_queries_2022_gon_l107_107044

theorem minimum_queries_2022_gon : ∃ (Q : ℕ), Q = 22 ∧ ∀ (choose_point_color : ℕ → Prop), 
  let A : Fin 2022 → (ℕ × ℕ) := fun i => has_property (colors : Fin 2022 → Bool),
    ∃Q, Q = 22 ∧ ∀(choose_point_color : Fin 2022 → Bool), 
      ∃Q, Q = 22 ∧ ∀(choose_point_color : Fin 2022 → Bool),
        Q = 22 :=

begin
  sorry
end

end minimum_queries_2022_gon_l107_107044


namespace probability_adjacent_vertices_decagon_l107_107121

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l107_107121


namespace jellybeans_original_count_l107_107808

theorem jellybeans_original_count (x : ℝ) (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_original_count_l107_107808


namespace find_a20_l107_107350

variables {a : ℕ → ℤ} {S : ℕ → ℤ}
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem find_a20 (h_arith : is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_a1 : a 1 = -1)
  (h_S10 : S 10 = 35) :
  a 20 = 18 :=
sorry

end find_a20_l107_107350


namespace martian_calendar_months_l107_107531

theorem martian_calendar_months (x y : ℕ) 
  (h1 : 100 * x + 77 * y = 5882) : x + y = 74 :=
sorry

end martian_calendar_months_l107_107531


namespace Ingrid_cookie_percentage_l107_107913

theorem Ingrid_cookie_percentage : 
  let irin_ratio := 9.18
  let ingrid_ratio := 5.17
  let nell_ratio := 2.05
  let kim_ratio := 3.45
  let linda_ratio := 4.56
  let total_cookies := 800
  let total_ratio := irin_ratio + ingrid_ratio + nell_ratio + kim_ratio + linda_ratio
  let ingrid_share := ingrid_ratio / total_ratio
  let ingrid_cookies := ingrid_share * total_cookies
  let ingrid_percentage := (ingrid_cookies / total_cookies) * 100
  ingrid_percentage = 21.25 :=
by
  sorry

end Ingrid_cookie_percentage_l107_107913


namespace div_z_x_l107_107227

variables (x y z : ℚ)

theorem div_z_x (h1 : x / y = 3) (h2 : y / z = 5 / 2) : z / x = 2 / 15 :=
sorry

end div_z_x_l107_107227
