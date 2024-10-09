import Mathlib

namespace max_new_cars_l6_629

theorem max_new_cars (b₁ : ℕ) (r : ℝ) (M : ℕ) (L : ℕ) (x : ℝ) (h₀ : b₁ = 30) (h₁ : r = 0.94) (h₂ : M = 600000) (h₃ : L = 300000) :
  x ≤ (3.6 * 10^4) :=
sorry

end max_new_cars_l6_629


namespace find_side_a_l6_605

theorem find_side_a (a b c : ℝ) (B : ℝ)
  (h1 : c = Real.sqrt 2)
  (h2 : b = Real.sqrt 6)
  (h3 : B = 120) :
  a = Real.sqrt 2 :=
sorry

end find_side_a_l6_605


namespace find_divisor_l6_698

theorem find_divisor 
  (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  (dividend = 172) → (quotient = 10) → (remainder = 2) → (dividend = (divisor * quotient) + remainder) → divisor = 17 :=
by 
  sorry

end find_divisor_l6_698


namespace shirley_cases_l6_670

-- Given conditions
def T : ℕ := 54  -- boxes of Trefoils sold
def S : ℕ := 36  -- boxes of Samoas sold
def M : ℕ := 48  -- boxes of Thin Mints sold
def t_per_case : ℕ := 4  -- boxes of Trefoils per case
def s_per_case : ℕ := 3  -- boxes of Samoas per case
def m_per_case : ℕ := 5  -- boxes of Thin Mints per case

-- Amount of boxes delivered per case should meet the required demand
theorem shirley_cases : ∃ (n_cases : ℕ), 
  n_cases * t_per_case ≥ T ∧ 
  n_cases * s_per_case ≥ S ∧ 
  n_cases * m_per_case ≥ M :=
by
  use 14
  sorry

end shirley_cases_l6_670


namespace find_integer_part_of_m_l6_676

theorem find_integer_part_of_m {m : ℝ} (h_lecture_duration : m > 0) 
    (h_swap_positions : ∃ k : ℤ, 120 + m = 60 + k * 12 * 60 / 13 ∧ (120 + m) % 60 = 60 * (120 + m) / 720) : 
    ⌊m⌋ = 46 :=
by
  sorry

end find_integer_part_of_m_l6_676


namespace max_voters_is_five_l6_630

noncomputable def max_voters_after_T (x : ℕ) : ℕ :=
if h : 0 ≤ (x - 11) then x - 11 else 0

theorem max_voters_is_five (x : ℕ) (hx : 0 ≤ x ∧ x ≤ 10) :
  max_voters_after_T x = 5 :=
by
  sorry

end max_voters_is_five_l6_630


namespace calculate_bmw_sales_and_revenue_l6_622

variable (total_cars : ℕ) (percentage_ford percentage_toyota percentage_nissan percentage_audi : ℕ) (avg_price_bmw : ℕ)
variable (h_total_cars : total_cars = 300) (h_percentage_ford : percentage_ford = 10)
variable (h_percentage_toyota : percentage_toyota = 25) (h_percentage_nissan : percentage_nissan = 20)
variable (h_percentage_audi : percentage_audi = 15) (h_avg_price_bmw : avg_price_bmw = 35000)

theorem calculate_bmw_sales_and_revenue :
  let percentage_non_bmw := percentage_ford + percentage_toyota + percentage_nissan + percentage_audi
  let percentage_bmw := 100 - percentage_non_bmw
  let number_bmw := total_cars * percentage_bmw / 100
  let total_revenue := number_bmw * avg_price_bmw
  (number_bmw = 90) ∧ (total_revenue = 3150000) := by
  -- Definitions are taken from conditions and used directly in the theorem statement
  sorry

end calculate_bmw_sales_and_revenue_l6_622


namespace student_fraction_mistake_l6_683

theorem student_fraction_mistake (n : ℕ) (h_n : n = 576) 
(h_mistake : ∃ r : ℚ, r * n = (5 / 16) * n + 300) : ∃ r : ℚ, r = 5 / 6 :=
by
  sorry

end student_fraction_mistake_l6_683


namespace number_of_red_parrots_l6_641

-- Defining the conditions from a)
def fraction_yellow_parrots : ℚ := 2 / 3
def total_birds : ℕ := 120

-- Stating the theorem we want to prove
theorem number_of_red_parrots (H1 : fraction_yellow_parrots = 2 / 3) (H2 : total_birds = 120) : 
  (1 - fraction_yellow_parrots) * total_birds = 40 := 
by 
  sorry

end number_of_red_parrots_l6_641


namespace calculate_LN_l6_685

theorem calculate_LN (sinN : ℝ) (LM LN : ℝ) (h1 : sinN = 4 / 5) (h2 : LM = 20) : LN = 25 :=
by
  sorry

end calculate_LN_l6_685


namespace initial_apples_value_l6_692

-- Definitions for the conditions
def picked_apples : ℤ := 105
def total_apples : ℤ := 161

-- Statement to prove
theorem initial_apples_value : ∀ (initial_apples : ℤ), 
  initial_apples + picked_apples = total_apples → 
  initial_apples = total_apples - picked_apples := 
by 
  sorry

end initial_apples_value_l6_692


namespace smallest_integer_among_three_l6_600

theorem smallest_integer_among_three 
  (x y z : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (hxy : y - x ≤ 6)
  (hxz : z - x ≤ 6) 
  (hprod : x * y * z = 2808) : 
  x = 12 := 
sorry

end smallest_integer_among_three_l6_600


namespace original_number_of_girls_l6_664

theorem original_number_of_girls (b g : ℕ) (h1 : b = g)
                                (h2 : 3 * (g - 25) = b)
                                (h3 : 6 * (b - 60) = g - 25) :
  g = 67 :=
by sorry

end original_number_of_girls_l6_664


namespace initial_teach_count_l6_672

theorem initial_teach_count :
  ∃ (x y : ℕ), (x + x * y + (x + x * y) * (y + x * y) = 195) ∧
               (y + x * y + (y + x * y) * (x + x * y) = 192) ∧
               x = 5 ∧ y = 2 :=
by {
  sorry
}

end initial_teach_count_l6_672


namespace elena_allowance_fraction_l6_640

variable {A m s : ℝ}

theorem elena_allowance_fraction {A : ℝ} (h1 : m = 0.25 * (A - s)) (h2 : s = 0.10 * (A - m)) : m + s = (4 / 13) * A :=
by
  sorry

end elena_allowance_fraction_l6_640


namespace dimension_sum_l6_614

-- Define the dimensions A, B, C and areas AB, AC, BC
variables (A B C : ℝ) (AB AC BC : ℝ)

-- Conditions
def conditions := AB = 40 ∧ AC = 90 ∧ BC = 100 ∧ A * B = AB ∧ A * C = AC ∧ B * C = BC

-- Theorem statement
theorem dimension_sum : conditions A B C AB AC BC → A + B + C = (83 : ℝ) / 3 :=
by
  intro h
  sorry

end dimension_sum_l6_614


namespace F_transformed_l6_661

-- Define the coordinates of point F
def F : ℝ × ℝ := (1, 0)

-- Reflection over the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Reflection over the y-axis
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

-- Reflection over the line y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

-- Point F after all transformations
def F_final : ℝ × ℝ :=
  reflect_y_eq_x (reflect_y (reflect_x F))

-- Statement to prove
theorem F_transformed : F_final = (0, -1) :=
  sorry

end F_transformed_l6_661


namespace smallest_value_of_n_l6_639

theorem smallest_value_of_n : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n + 6) % 7 = 0 ∧ (n - 9) % 4 = 0 ∧ n = 113 :=
by
  sorry

end smallest_value_of_n_l6_639


namespace part_a_part_b_l6_604

-- Part (a)
theorem part_a (students : Fin 67) (answers : Fin 6 → Bool) :
  ∃ (s1 s2 : Fin 67), s1 ≠ s2 ∧ answers s1 = answers s2 := by
  sorry

-- Part (b)
theorem part_b (students : Fin 67) (points : Fin 6 → ℤ)
  (h_points : ∀ k, points k = k ∨ points k = -k) :
  ∃ (scores : Fin 67 → ℤ), ∃ (s1 s2 s3 s4 : Fin 67),
  s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4 ∧
  scores s1 = scores s2 ∧ scores s1 = scores s3 ∧ scores s1 = scores s4 := by
  sorry

end part_a_part_b_l6_604


namespace g_inv_g_inv_14_l6_651

noncomputable def g (x : ℝ) := 3 * x - 4

noncomputable def g_inv (x : ℝ) := (x + 4) / 3

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by sorry

end g_inv_g_inv_14_l6_651


namespace jenna_hike_duration_l6_642

-- Definitions from conditions
def initial_speed : ℝ := 25
def exhausted_speed : ℝ := 10
def total_distance : ℝ := 140
def total_time : ℝ := 8

-- The statement to prove:
theorem jenna_hike_duration : ∃ x : ℝ, 25 * x + 10 * (8 - x) = 140 ∧ x = 4 := by
  sorry

end jenna_hike_duration_l6_642


namespace sum_of_series_equals_negative_682_l6_625

noncomputable def geometric_sum : ℤ :=
  let a := 2
  let r := -2
  let n := 10
  (a * (r ^ n - 1)) / (r - 1)

theorem sum_of_series_equals_negative_682 : geometric_sum = -682 := 
by sorry

end sum_of_series_equals_negative_682_l6_625


namespace min_a_value_l6_690

theorem min_a_value {a b : ℕ} (h : 1998 * a = b^4) : a = 1215672 :=
sorry

end min_a_value_l6_690


namespace distance_between_points_l6_623

theorem distance_between_points (points : Fin 7 → ℝ × ℝ) (diameter : ℝ)
  (h_diameter : diameter = 1)
  (h_points_in_circle : ∀ i : Fin 7, (points i).fst^2 + (points i).snd^2 ≤ (diameter / 2)^2) :
  ∃ (i j : Fin 7), i ≠ j ∧ (dist (points i) (points j) ≤ 1 / 2) := 
by
  sorry

end distance_between_points_l6_623


namespace richard_cleans_in_45_minutes_l6_655
noncomputable def richard_time (R : ℝ) := 
  let cory_time := R + 3
  let blake_time := (R + 3) - 4
  (R + cory_time + blake_time = 136) -> R = 45

theorem richard_cleans_in_45_minutes : 
  ∃ R : ℝ, richard_time R := 
sorry

end richard_cleans_in_45_minutes_l6_655


namespace micah_has_seven_fish_l6_665

-- Definitions from problem conditions
def micahFish (M : ℕ) : Prop :=
  let kennethFish := 3 * M
  let matthiasFish := kennethFish - 15
  M + kennethFish + matthiasFish = 34

-- Main statement: prove that the number of fish Micah has is 7
theorem micah_has_seven_fish : ∃ M : ℕ, micahFish M ∧ M = 7 :=
by
  sorry

end micah_has_seven_fish_l6_665


namespace similar_triangle_shortest_side_l6_669

theorem similar_triangle_shortest_side (a b c : ℕ) (H1 : a^2 + b^2 = c^2) (H2 : a = 15) (H3 : c = 34) (H4 : b = Int.sqrt 931) : 
  ∃ d : ℝ, d = 3 * Int.sqrt 931 ∧ d = 102  :=
by
  sorry

end similar_triangle_shortest_side_l6_669


namespace extreme_value_result_l6_619

open Real

-- Conditions
def function_has_extreme_value_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop := 
  deriv f x₀ = 0

-- The given function
noncomputable def f (x : ℝ) : ℝ := x * sin x

-- The problem statement (to prove)
theorem extreme_value_result (x₀ : ℝ) 
  (h : function_has_extreme_value_at f x₀) :
  (1 + x₀^2) * (1 + cos (2 * x₀)) = 2 :=
sorry

end extreme_value_result_l6_619


namespace negation_of_prop_l6_608

theorem negation_of_prop :
  (¬ ∀ x : ℝ, x^2 > x - 1) ↔ ∃ x : ℝ, x^2 ≤ x - 1 :=
sorry

end negation_of_prop_l6_608


namespace complement_event_A_l6_689

def is_at_least_two_defective (n : ℕ) : Prop :=
  n ≥ 2

def is_at_most_one_defective (n : ℕ) : Prop :=
  n ≤ 1

theorem complement_event_A (n : ℕ) :
  (¬ is_at_least_two_defective n) ↔ is_at_most_one_defective n :=
by
  sorry

end complement_event_A_l6_689


namespace smallest_X_divisible_15_l6_660

theorem smallest_X_divisible_15 (T X : ℕ) 
  (h1 : T > 0) 
  (h2 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) 
  (h3 : T % 15 = 0) 
  (h4 : X = T / 15) : 
  X = 74 :=
sorry

end smallest_X_divisible_15_l6_660


namespace find_n_mod_10_l6_684

theorem find_n_mod_10 :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n % 10 = (-2023) % 10 ∧ n = 7 :=
sorry

end find_n_mod_10_l6_684


namespace jellybean_mass_l6_652

noncomputable def cost_per_gram : ℚ := 7.50 / 250
noncomputable def mass_for_180_cents : ℚ := 1.80 / cost_per_gram

theorem jellybean_mass :
  mass_for_180_cents = 60 := 
  sorry

end jellybean_mass_l6_652


namespace walkway_area_l6_650

/--
Tara has four rows of three 8-feet by 3-feet flower beds in her garden. The beds are separated
and surrounded by 2-feet-wide walkways. Prove that the total area of the walkways is 416 square feet.
-/
theorem walkway_area :
  let flower_bed_width := 8
  let flower_bed_height := 3
  let num_rows := 4
  let num_columns := 3
  let walkway_width := 2
  let total_width := (num_columns * flower_bed_width) + (num_columns + 1) * walkway_width
  let total_height := (num_rows * flower_bed_height) + (num_rows + 1) * walkway_width
  let total_garden_area := total_width * total_height
  let flower_bed_area := flower_bed_width * flower_bed_height * num_rows * num_columns
  total_garden_area - flower_bed_area = 416 :=
by
  -- Proof omitted
  sorry

end walkway_area_l6_650


namespace single_elimination_matches_l6_627

theorem single_elimination_matches (players byes : ℕ)
  (h1 : players = 100)
  (h2 : byes = 28) :
  (players - 1) = 99 :=
by
  -- The proof would go here if it were needed
  sorry

end single_elimination_matches_l6_627


namespace cannot_be_expressed_as_difference_of_squares_l6_611

theorem cannot_be_expressed_as_difference_of_squares : 
  ¬ ∃ (a b : ℤ), 2006 = a^2 - b^2 :=
sorry

end cannot_be_expressed_as_difference_of_squares_l6_611


namespace sum_of_integers_l6_678

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 168) : x + y = 32 :=
by
  sorry

end sum_of_integers_l6_678


namespace triangle_side_length_l6_644

theorem triangle_side_length (A : ℝ) (b : ℝ) (S : ℝ) (hA : A = 120) (hb : b = 4) (hS: S = 2 * Real.sqrt 3) : 
  ∃ c : ℝ, c = 2 := 
by 
  sorry

end triangle_side_length_l6_644


namespace find_a_for_exactly_two_solutions_l6_647

theorem find_a_for_exactly_two_solutions :
  ∃ a : ℝ, (∀ x : ℝ, (|x + a| = 1/x) ↔ (a = -2) ∧ (x ≠ 0)) ∧ ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (|x1 + a| = 1/x1) ∧ (|x2 + a| = 1/x2) :=
sorry

end find_a_for_exactly_two_solutions_l6_647


namespace ratio_prikya_ladonna_l6_686

def total_cans : Nat := 85
def ladonna_cans : Nat := 25
def yoki_cans : Nat := 10
def prikya_cans : Nat := total_cans - ladonna_cans - yoki_cans

theorem ratio_prikya_ladonna : prikya_cans.toFloat / ladonna_cans.toFloat = 2 / 1 := 
by sorry

end ratio_prikya_ladonna_l6_686


namespace smallest_natural_number_divisible_l6_663

theorem smallest_natural_number_divisible :
  ∃ n : ℕ, (n^2 + 14 * n + 13) % 68 = 0 ∧ 
          ∀ m : ℕ, (m^2 + 14 * m + 13) % 68 = 0 → 21 ≤ m :=
by 
  sorry

end smallest_natural_number_divisible_l6_663


namespace correct_arrangements_l6_620

open Finset Nat

-- Definitions for combinations and powers
def comb (n k : ℕ) : ℕ := choose n k

-- The number of computer rooms
def num_computer_rooms : ℕ := 6

-- The number of arrangements to open at least 2 out of 6 computer rooms
def arrangement_count1 : ℕ := 2^num_computer_rooms - (comb num_computer_rooms 0 + comb num_computer_rooms 1)

-- Another calculation for the number of arrangements to open at least 2 out of 6 computer rooms
def arrangement_count2 : ℕ := comb num_computer_rooms 2 + comb num_computer_rooms 3 + comb num_computer_rooms 4 + 
                               comb num_computer_rooms 5 + comb num_computer_rooms 6

theorem correct_arrangements :
  arrangement_count1 = arrangement_count2 := 
  sorry

end correct_arrangements_l6_620


namespace f_relation_l6_694

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem f_relation :
  f (-Real.pi / 3) > f 1 ∧ f 1 > f (Real.pi / 5) :=
by
  sorry

end f_relation_l6_694


namespace greatest_value_a_maximum_value_a_l6_666

-- Define the quadratic polynomial
def quadratic (a : ℝ) : ℝ := -a^2 + 9 * a - 20

-- The statement to be proven:
theorem greatest_value_a : ∀ a : ℝ, (quadratic a ≥ 0) → a ≤ 5 := 
sorry

theorem maximum_value_a : quadratic 5 = 0 :=
sorry

end greatest_value_a_maximum_value_a_l6_666


namespace cricket_initial_matches_l6_636

theorem cricket_initial_matches (x : ℝ) :
  (0.28 * x + 60 = 0.52 * (x + 60)) → x = 120 :=
by
  sorry

end cricket_initial_matches_l6_636


namespace trigonometric_solution_l6_682

theorem trigonometric_solution (x : Real) :
  (2 * Real.sin x * Real.cos (3 * Real.pi / 2 + x) 
  - 3 * Real.sin (Real.pi - x) * Real.cos x 
  + Real.sin (Real.pi / 2 + x) * Real.cos x = 0) ↔ 
  (∃ k : Int, x = Real.arctan ((3 + Real.sqrt 17) / -4) + k * Real.pi) ∨ 
  (∃ n : Int, x = Real.arctan ((3 - Real.sqrt 17) / -4) + n * Real.pi) :=
sorry

end trigonometric_solution_l6_682


namespace fraction_of_girls_is_half_l6_680

variables (T G B : ℝ)
def fraction_x_of_girls (x : ℝ) : Prop :=
  x * G = (1/5) * T ∧ B / G = 1.5 ∧ T = B + G

theorem fraction_of_girls_is_half (x : ℝ) (h : fraction_x_of_girls T G B x) : x = 0.5 :=
sorry

end fraction_of_girls_is_half_l6_680


namespace huanhuan_initial_coins_l6_696

theorem huanhuan_initial_coins :
  ∃ (H L n : ℕ), H = 7 * L ∧ (H + n = 6 * (L + n)) ∧ (H + 2 * n = 5 * (L + 2 * n)) ∧ H = 70 :=
by
  sorry

end huanhuan_initial_coins_l6_696


namespace solve_for_c_l6_613

theorem solve_for_c (a b c d e : ℝ) 
  (h1 : a + b + c = 48)
  (h2 : c + d + e = 78)
  (h3 : a + b + c + d + e = 100) :
  c = 26 :=
by
sorry

end solve_for_c_l6_613


namespace spacy_subsets_15_l6_637

def spacy (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 2
  | 2     => 3
  | n + 3 => spacy n + spacy (n-2)

theorem spacy_subsets_15 : spacy 15 = 406 := 
  sorry

end spacy_subsets_15_l6_637


namespace eval_arith_expression_l6_606

theorem eval_arith_expression : 2 + 3^2 * 4 - 5 + 6 / 2 = 36 := 
by sorry

end eval_arith_expression_l6_606


namespace train_speed_correct_l6_612

noncomputable def train_speed (length_meters : ℕ) (time_seconds : ℕ) : ℝ :=
  (length_meters : ℝ) / 1000 / (time_seconds / 3600)

theorem train_speed_correct :
  train_speed 2500 50 = 180 := 
by
  -- We leave the proof as sorry, the statement is sufficient
  sorry

end train_speed_correct_l6_612


namespace find_fraction_of_difference_eq_halves_l6_649

theorem find_fraction_of_difference_eq_halves (x : ℚ) (h : 9 - x = 2.25) : x = 27 / 4 :=
by sorry

end find_fraction_of_difference_eq_halves_l6_649


namespace smallest_n_divisible_by_13_l6_673

theorem smallest_n_divisible_by_13 : ∃ (n : ℕ), 5^n + n^5 ≡ 0 [MOD 13] ∧ ∀ (m : ℕ), m < n → ¬(5^m + m^5 ≡ 0 [MOD 13]) :=
sorry

end smallest_n_divisible_by_13_l6_673


namespace probability_of_victory_l6_671

theorem probability_of_victory (p_A p_B : ℝ) (h_A : p_A = 0.3) (h_B : p_B = 0.6) (independent : true) :
  p_A * p_B = 0.18 :=
by
  -- placeholder for proof
  sorry

end probability_of_victory_l6_671


namespace total_hours_worked_l6_695

-- Definition of the given conditions.
def hours_software : ℕ := 24
def hours_help_user : ℕ := 17
def percentage_other_services : ℚ := 0.4

-- Statement to prove.
theorem total_hours_worked : ∃ (T : ℕ), hours_software + hours_help_user + percentage_other_services * T = T ∧ T = 68 :=
by {
  -- The proof will go here.
  sorry
}

end total_hours_worked_l6_695


namespace line_intersection_l6_645

/-- Prove the intersection of the lines given by the equations
    8x - 5y = 10 and 3x + 2y = 1 is (25/31, -22/31) -/
theorem line_intersection :
  ∃ (x y : ℚ), 8 * x - 5 * y = 10 ∧ 3 * x + 2 * y = 1 ∧ x = 25 / 31 ∧ y = -22 / 31 :=
by
  sorry

end line_intersection_l6_645


namespace three_angles_difference_is_2pi_over_3_l6_621

theorem three_angles_difference_is_2pi_over_3 (α β γ : ℝ) 
    (h1 : 0 ≤ α) (h2 : α ≤ β) (h3 : β < γ) (h4 : γ ≤ 2 * Real.pi)
    (h5 : Real.cos α + Real.cos β + Real.cos γ = 0)
    (h6 : Real.sin α + Real.sin β + Real.sin γ = 0) :
    β - α = 2 * Real.pi / 3 :=
sorry

end three_angles_difference_is_2pi_over_3_l6_621


namespace necessary_but_not_sufficient_condition_proof_l6_658

noncomputable def necessary_but_not_sufficient_condition (x : ℝ) : Prop :=
  2 * x ^ 2 - 5 * x - 3 ≥ 0

theorem necessary_but_not_sufficient_condition_proof (x : ℝ) :
  (x < 0 ∨ x > 2) → necessary_but_not_sufficient_condition x :=
  sorry

end necessary_but_not_sufficient_condition_proof_l6_658


namespace discarded_number_l6_654

theorem discarded_number (S S_48 : ℝ) (h1 : S = 1000) (h2 : S_48 = 900) (h3 : ∃ x : ℝ, S - S_48 = 45 + x): 
  ∃ x : ℝ, x = 55 :=
by {
  -- Using the conditions provided to derive the theorem.
  sorry 
}

end discarded_number_l6_654


namespace prove_x3_y3_le_2_l6_677

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

axiom positive_x : 0 < x
axiom positive_y : 0 < y
axiom condition : x^3 + y^4 ≤ x^2 + y^3

theorem prove_x3_y3_le_2 : x^3 + y^3 ≤ 2 := 
by
  sorry

end prove_x3_y3_le_2_l6_677


namespace range_of_a_minus_b_l6_693

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : 0 < b ∧ b < 1) : -1 < a - b ∧ a - b < 2 := 
by
  sorry

end range_of_a_minus_b_l6_693


namespace pencil_sharpening_and_breaking_l6_618

/-- Isha's pencil initially has a length of 31 inches. After sharpening, it has a length of 14 inches.
Prove that:
1. The pencil was shortened by 17 inches.
2. Each half of the pencil, after being broken in half, is 7 inches long. -/
theorem pencil_sharpening_and_breaking 
  (initial_length : ℕ) 
  (length_after_sharpening : ℕ) 
  (sharpened_length : ℕ) 
  (half_length : ℕ) 
  (h1 : initial_length = 31) 
  (h2 : length_after_sharpening = 14) 
  (h3 : sharpened_length = initial_length - length_after_sharpening) 
  (h4 : half_length = length_after_sharpening / 2) : 
  sharpened_length = 17 ∧ half_length = 7 := 
by {
  sorry
}

end pencil_sharpening_and_breaking_l6_618


namespace todd_ingredients_l6_610

variables (B R N : ℕ) (P A : ℝ) (I : ℝ)

def todd_problem (B R N : ℕ) (P A I : ℝ) : Prop := 
  B = 100 ∧ 
  R = 110 ∧ 
  N = 200 ∧ 
  P = 0.75 ∧ 
  A = 65 ∧ 
  I = 25

theorem todd_ingredients :
  todd_problem 100 110 200 0.75 65 25 :=
by sorry

end todd_ingredients_l6_610


namespace ratio_of_games_played_to_losses_l6_657

-- Conditions
def games_played : ℕ := 10
def games_won : ℕ := 5
def games_lost : ℕ := games_played - games_won

-- Prove the ratio of games played to games lost is 2:1
theorem ratio_of_games_played_to_losses
  (h_played : games_played = 10)
  (h_won : games_won = 5) :
  (games_played / Nat.gcd games_played games_lost : ℕ) /
  (games_lost / Nat.gcd games_played games_lost : ℕ) = 2 / 1 :=
by
  sorry

end ratio_of_games_played_to_losses_l6_657


namespace subsequence_sum_q_l6_688

theorem subsequence_sum_q (S : Fin 1995 → ℕ) (m : ℕ) (hS_pos : ∀ i : Fin 1995, 0 < S i)
  (hS_sum : (Finset.univ : Finset (Fin 1995)).sum S = m) (h_m_lt : m < 3990) :
  ∀ q : ℕ, 1 ≤ q → q ≤ m → ∃ (I : Finset (Fin 1995)), I.sum S = q := 
sorry

end subsequence_sum_q_l6_688


namespace average_weight_men_women_l6_634

theorem average_weight_men_women (n_men n_women : ℕ) (avg_weight_men avg_weight_women : ℚ)
  (h_men : n_men = 8) (h_women : n_women = 6) (h_avg_weight_men : avg_weight_men = 190)
  (h_avg_weight_women : avg_weight_women = 120) :
  (n_men * avg_weight_men + n_women * avg_weight_women) / (n_men + n_women) = 160 := 
by
  sorry

end average_weight_men_women_l6_634


namespace dana_hours_sunday_l6_667

-- Define the constants given in the problem
def hourly_rate : ℝ := 13
def hours_worked_friday : ℝ := 9
def hours_worked_saturday : ℝ := 10
def total_earnings : ℝ := 286

-- Define the function to compute total earnings from worked hours and hourly rate
def earnings (hours : ℝ) (rate : ℝ) : ℝ := hours * rate

-- Define the proof problem to show the number of hours worked on Sunday
theorem dana_hours_sunday (hours_sunday : ℝ) :
  earnings hours_worked_friday hourly_rate
  + earnings hours_worked_saturday hourly_rate
  + earnings hours_sunday hourly_rate = total_earnings ->
  hours_sunday = 3 :=
by
  sorry -- proof to be filled in

end dana_hours_sunday_l6_667


namespace sum_of_interior_angles_pentagon_l6_648

theorem sum_of_interior_angles_pentagon : (5 - 2) * 180 = 540 := by
  sorry

end sum_of_interior_angles_pentagon_l6_648


namespace boxes_needed_l6_631

-- Let's define the conditions
def total_paper_clips : ℕ := 81
def paper_clips_per_box : ℕ := 9

-- Define the target of our proof, which is that the number of boxes needed is 9
theorem boxes_needed : total_paper_clips / paper_clips_per_box = 9 := by
  sorry

end boxes_needed_l6_631


namespace find_number_l6_616

theorem find_number (x : ℝ) (h : 120 = 1.5 * x) : x = 80 :=
by
  sorry

end find_number_l6_616


namespace apples_in_box_l6_617

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

end apples_in_box_l6_617


namespace percentage_difference_l6_626

theorem percentage_difference : (0.5 * 56) - (0.3 * 50) = 13 := by
  sorry

end percentage_difference_l6_626


namespace minimum_squares_required_l6_646

theorem minimum_squares_required (length : ℚ) (width : ℚ) (M N : ℕ) :
  (length = 121 / 2) → (width = 143 / 3) → (M / N = 33 / 26) → (M * N = 858) :=
by
  intros hL hW hMN
  -- Proof skipped
  sorry

end minimum_squares_required_l6_646


namespace no_rotation_of_11_gears_l6_601

theorem no_rotation_of_11_gears :
  ∀ (gears : Fin 11 → ℕ → Prop), 
    (∀ i, gears i 0 ∧ gears (i + 1) 1 → gears i 0 = ¬gears (i + 1) 1) →
    gears 10 0 = gears 0 0 →
    False :=
by
  sorry

end no_rotation_of_11_gears_l6_601


namespace find_points_on_number_line_l6_656

noncomputable def numbers_are_opposite (x y : ℝ) : Prop :=
  x = -y

theorem find_points_on_number_line (A B : ℝ) 
  (h1 : numbers_are_opposite A B) 
  (h2 : |A - B| = 8) 
  (h3 : A < B) : 
  (A = -4 ∧ B = 4) :=
by
  sorry

end find_points_on_number_line_l6_656


namespace ryan_more_hours_english_than_chinese_l6_681

-- Definitions for the time Ryan spends on subjects
def weekday_hours_english : ℕ := 6 * 5
def weekend_hours_english : ℕ := 2 * 2
def total_hours_english : ℕ := weekday_hours_english + weekend_hours_english

def weekday_hours_chinese : ℕ := 3 * 5
def weekend_hours_chinese : ℕ := 1 * 2
def total_hours_chinese : ℕ := weekday_hours_chinese + weekend_hours_chinese

-- Theorem stating the difference in hours spent on English vs Chinese
theorem ryan_more_hours_english_than_chinese :
  (total_hours_english - total_hours_chinese) = 17 := by
  sorry

end ryan_more_hours_english_than_chinese_l6_681


namespace parabola_axis_l6_602

theorem parabola_axis (p : ℝ) (h_parabola : ∀ x : ℝ, y = x^2 → x^2 = y) : (y = - p / 2) :=
by
  sorry

end parabola_axis_l6_602


namespace monotonic_f_inequality_f_over_h_l6_687

noncomputable def f (x : ℝ) : ℝ := 1 + (1 / x) + Real.log x + (Real.log x / x)

theorem monotonic_f :
  ∀ x : ℝ, x > 0 → ∃ I : Set ℝ, (I = Set.Ioo 0 x ∨ I = Set.Icc 0 x) ∧ (∀ y ∈ I, y > 0 → f y = f x) :=
by
  sorry

theorem inequality_f_over_h :
  ∀ x : ℝ, x > 1 → (f x) / (Real.exp 1 + 1) > (2 * Real.exp (x - 1)) / (x * Real.exp x + 1) :=
by
  sorry

end monotonic_f_inequality_f_over_h_l6_687


namespace nonagon_diagonals_l6_635

-- Define the number of sides of the polygon (nonagon)
def num_sides : ℕ := 9

-- Define the formula for the number of diagonals in a convex n-sided polygon
def number_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem
theorem nonagon_diagonals : number_diagonals num_sides = 27 := 
by
--placeholder for the proof
sorry

end nonagon_diagonals_l6_635


namespace eval_g_at_neg2_l6_699

def g (x : ℝ) : ℝ := 5 * x + 2

theorem eval_g_at_neg2 : g (-2) = -8 := by
  sorry

end eval_g_at_neg2_l6_699


namespace Robert_photo_count_l6_653

theorem Robert_photo_count (k : ℕ) (hLisa : ∃ n : ℕ, k = 8 * n) : k = 24 - 16 → k = 24 :=
by
  intro h
  sorry

end Robert_photo_count_l6_653


namespace intersection_of_A_and_B_l6_662

def A : Set ℝ := { x | -2 < x ∧ x < 2 }
def B : Set ℝ := { x | x ≤ 1 ∨ x ≥ 3 }

theorem intersection_of_A_and_B : 
  (A ∩ B) = { x : ℝ | -2 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_A_and_B_l6_662


namespace gcd_k_power_eq_k_minus_one_l6_624

noncomputable def gcd_k_power (k : ℤ) : ℤ := 
  Int.gcd (k^1024 - 1) (k^1035 - 1)

theorem gcd_k_power_eq_k_minus_one (k : ℤ) : gcd_k_power k = k - 1 := 
  sorry

end gcd_k_power_eq_k_minus_one_l6_624


namespace librarian_donated_200_books_this_year_l6_615

noncomputable def total_books_five_years_ago : ℕ := 500
noncomputable def books_bought_two_years_ago : ℕ := 300
noncomputable def books_bought_last_year : ℕ := books_bought_two_years_ago + 100
noncomputable def total_books_current : ℕ := 1000

-- The Lean statement to prove the librarian donated 200 old books this year
theorem librarian_donated_200_books_this_year :
  total_books_five_years_ago + books_bought_two_years_ago + books_bought_last_year - total_books_current = 200 :=
by sorry

end librarian_donated_200_books_this_year_l6_615


namespace combined_avg_of_remaining_two_subjects_l6_603

noncomputable def avg (scores : List ℝ) : ℝ :=
  scores.foldl (· + ·) 0 / scores.length

theorem combined_avg_of_remaining_two_subjects 
  (S1_avg S2_part_avg all_avg : ℝ)
  (S1_count S2_part_count S2_total_count : ℕ)
  (h1 : S1_avg = 85) 
  (h2 : S2_part_avg = 78) 
  (h3 : all_avg = 80) 
  (h4 : S1_count = 3)
  (h5 : S2_part_count = 5)
  (h6 : S2_total_count = 7) :
  avg [all_avg * (S1_count + S2_total_count) 
       - S1_count * S1_avg 
       - S2_part_count * S2_part_avg] / (S2_total_count - S2_part_count)
  = 77.5 := by
  sorry

end combined_avg_of_remaining_two_subjects_l6_603


namespace expand_polynomial_l6_609

theorem expand_polynomial :
  (7 * x^2 + 5 * x - 3) * (3 * x^3 + 2 * x^2 - x + 4) = 21 * x^5 + 29 * x^4 - 6 * x^3 + 17 * x^2 + 23 * x - 12 :=
by
  sorry

end expand_polynomial_l6_609


namespace baker_cake_count_l6_679

theorem baker_cake_count :
  let initial_cakes := 62
  let additional_cakes := 149
  let sold_cakes := 144
  initial_cakes + additional_cakes - sold_cakes = 67 :=
by
  sorry

end baker_cake_count_l6_679


namespace rope_cut_ratio_l6_675

theorem rope_cut_ratio (L : ℕ) (a b : ℕ) (hL : L = 40) (ha : a = 2) (hb : b = 3) :
  L / (a + b) * a = 16 :=
by
  sorry

end rope_cut_ratio_l6_675


namespace ratio_of_time_spent_l6_697

theorem ratio_of_time_spent {total_minutes type_a_minutes type_b_minutes : ℝ}
  (h1 : total_minutes = 180)
  (h2 : type_a_minutes = 32.73)
  (h3 : type_b_minutes = total_minutes - type_a_minutes) :
  type_a_minutes / type_a_minutes = 1 ∧ type_b_minutes / type_a_minutes = 4.5 := by
  sorry

end ratio_of_time_spent_l6_697


namespace average_weight_decrease_l6_668

theorem average_weight_decrease 
  (num_persons : ℕ)
  (avg_weight_initial : ℕ)
  (new_person_weight : ℕ)
  (new_avg_weight : ℚ)
  (weight_decrease : ℚ)
  (h1 : num_persons = 20)
  (h2 : avg_weight_initial = 60)
  (h3 : new_person_weight = 45)
  (h4 : new_avg_weight = (1200 + 45) / 21) : 
  weight_decrease = avg_weight_initial - new_avg_weight :=
by
  sorry

end average_weight_decrease_l6_668


namespace no_solution_implies_b_positive_l6_659

theorem no_solution_implies_b_positive (a b : ℝ) :
  (¬ ∃ x y : ℝ, y = x^2 + a * x + b ∧ x = y^2 + a * y + b) → b > 0 :=
by
  sorry

end no_solution_implies_b_positive_l6_659


namespace projection_of_A_onto_Oxz_is_B_l6_607

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def projection_onto_Oxz (A : Point3D) : Point3D :=
  { x := A.x, y := 0, z := A.z }

theorem projection_of_A_onto_Oxz_is_B :
  let A := Point3D.mk 2 3 6
  let B := Point3D.mk 2 0 6
  projection_onto_Oxz A = B :=
by
  let A := Point3D.mk 2 3 6
  let B := Point3D.mk 2 0 6
  have h : projection_onto_Oxz A = B := rfl
  exact h

end projection_of_A_onto_Oxz_is_B_l6_607


namespace tangent_circles_BC_length_l6_643

theorem tangent_circles_BC_length
  (rA rB : ℝ) (A B C : ℝ × ℝ) (distAB distAC : ℝ) 
  (hAB : rA + rB = distAB)
  (hAC : distAB + 2 = distAC) 
  (h_sim : ∀ AD BE BC AC : ℝ, AD / BE = rA / rB → BC / AC = rB / rA) :
  BC = 52 / 7 := sorry

end tangent_circles_BC_length_l6_643


namespace intersection_complement_eq_l6_638

def setA : Set ℝ := { x | (x - 6) * (x + 1) ≤ 0 }
def setB : Set ℝ := { x | x ≥ 2 }

theorem intersection_complement_eq :
  setA ∩ (Set.univ \ setB) = { x | -1 ≤ x ∧ x < 2 } := 
by 
  sorry

end intersection_complement_eq_l6_638


namespace tim_income_percent_less_than_juan_l6_691

theorem tim_income_percent_less_than_juan (T M J : ℝ) (h1 : M = 1.5 * T) (h2 : M = 0.9 * J) :
  (J - T) / J = 0.4 :=
by
  sorry

end tim_income_percent_less_than_juan_l6_691


namespace polynomial_expansion_l6_632

theorem polynomial_expansion :
  ∃ A B C D : ℝ, (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) 
  ∧ (A + B + C + D = 36) :=
by {
  sorry
}

end polynomial_expansion_l6_632


namespace pow_two_ge_square_l6_674

theorem pow_two_ge_square {n : ℕ} (hn : n ≥ 4) : 2^n ≥ n^2 :=
sorry

end pow_two_ge_square_l6_674


namespace trapezoid_equilateral_triangle_ratio_l6_633

theorem trapezoid_equilateral_triangle_ratio (s d : ℝ) (AB CD : ℝ) 
  (h1 : AB = s) 
  (h2 : CD = 2 * d)
  (h3 : d = s) : 
  AB / CD = 1 / 2 := 
by
  sorry

end trapezoid_equilateral_triangle_ratio_l6_633


namespace find_y_l6_628

theorem find_y (x y : ℤ) (h1 : x + y = 250) (h2 : x - y = 200) : y = 25 :=
by
  sorry

end find_y_l6_628
