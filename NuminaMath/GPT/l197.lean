import Mathlib

namespace NUMINAMATH_GPT_member_number_property_l197_19736

theorem member_number_property :
  ∃ (country : Fin 6) (member_number : Fin 1978),
    (∀ (i j : Fin 1978), i ≠ j → member_number ≠ i + j) ∨
    (∀ (k : Fin 1978), member_number ≠ 2 * k) :=
by
  sorry

end NUMINAMATH_GPT_member_number_property_l197_19736


namespace NUMINAMATH_GPT_great_eighteen_hockey_league_games_l197_19746

theorem great_eighteen_hockey_league_games :
  (let teams_per_division := 9
   let games_intra_division_per_team := 8 * 3
   let games_inter_division_per_team := teams_per_division * 2
   let total_games_per_team := games_intra_division_per_team + games_inter_division_per_team
   let total_game_instances := 18 * total_games_per_team
   let unique_games := total_game_instances / 2
   unique_games = 378) :=
by
  sorry

end NUMINAMATH_GPT_great_eighteen_hockey_league_games_l197_19746


namespace NUMINAMATH_GPT_diagonal_length_of_regular_hexagon_l197_19703

theorem diagonal_length_of_regular_hexagon (
  side_length : ℝ
) (h_side_length : side_length = 12) : 
  ∃ DA, DA = 12 * Real.sqrt 3 :=
by 
  sorry

end NUMINAMATH_GPT_diagonal_length_of_regular_hexagon_l197_19703


namespace NUMINAMATH_GPT_externally_tangent_circles_radius_l197_19792

theorem externally_tangent_circles_radius :
  ∃ r : ℝ, r > 0 ∧ (∀ x y, (x^2 + y^2 = 1 ∧ ((x - 3)^2 + y^2 = r^2)) → r = 2) :=
sorry

end NUMINAMATH_GPT_externally_tangent_circles_radius_l197_19792


namespace NUMINAMATH_GPT_initial_bacteria_count_l197_19729

theorem initial_bacteria_count (d: ℕ) (t_final: ℕ) (N_final: ℕ) 
    (h1: t_final = 4 * 60)  -- 4 minutes equals 240 seconds
    (h2: d = 15)            -- Doubling interval is 15 seconds
    (h3: N_final = 2097152) -- Final bacteria count is 2,097,152
    :
    ∃ n: ℕ, N_final = n * 2^((t_final / d)) ∧ n = 32 :=
by
  sorry

end NUMINAMATH_GPT_initial_bacteria_count_l197_19729


namespace NUMINAMATH_GPT_magician_earning_l197_19773

-- Definitions based on conditions
def price_per_deck : ℕ := 2
def initial_decks : ℕ := 5
def remaining_decks : ℕ := 3

-- Theorem statement
theorem magician_earning :
  let sold_decks := initial_decks - remaining_decks
  let earning := sold_decks * price_per_deck
  earning = 4 := by
  sorry

end NUMINAMATH_GPT_magician_earning_l197_19773


namespace NUMINAMATH_GPT_factorization_example_l197_19716

theorem factorization_example (C D : ℤ) (h : 20 * y^2 - 122 * y + 72 = (C * y - 8) * (D * y - 9)) : C * D + C = 25 := by
  sorry

end NUMINAMATH_GPT_factorization_example_l197_19716


namespace NUMINAMATH_GPT_solution_set_l197_19728

-- Define the system of equations
def system_of_equations (x y : ℤ) : Prop :=
  4 * x^2 = y^2 + 2 * y + 4 ∧
  (2 * x)^2 - (y + 1)^2 = 3 ∧
  (2 * x - (y + 1)) * (2 * x + (y + 1)) = 3

-- Prove that the solutions to the system are the set we expect
theorem solution_set : 
  { (x, y) : ℤ × ℤ | system_of_equations x y } = { (1, 0), (1, -2), (-1, 0), (-1, -2) } := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_solution_set_l197_19728


namespace NUMINAMATH_GPT_total_shared_amount_l197_19765

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := sorry

axiom h1 : A = 1 / 3 * (B + C)
axiom h2 : B = 2 / 7 * (A + C)
axiom h3 : A = B + 20

theorem total_shared_amount : A + B + C = 720 := by
  sorry

end NUMINAMATH_GPT_total_shared_amount_l197_19765


namespace NUMINAMATH_GPT_max_volume_of_prism_l197_19747

theorem max_volume_of_prism (a b c s : ℝ) (h : a + b + c = 3 * s) : a * b * c ≤ s^3 :=
by {
    -- placeholder for the proof
    sorry
}

end NUMINAMATH_GPT_max_volume_of_prism_l197_19747


namespace NUMINAMATH_GPT_crayons_difference_l197_19731

def initial_crayons : ℕ := 8597
def crayons_given : ℕ := 7255
def crayons_lost : ℕ := 3689

theorem crayons_difference : crayons_given - crayons_lost = 3566 := by
  sorry

end NUMINAMATH_GPT_crayons_difference_l197_19731


namespace NUMINAMATH_GPT_find_breadth_of_rectangular_plot_l197_19735

-- Define the conditions
def length_is_thrice_breadth (b l : ℕ) : Prop := l = 3 * b
def area_is_363 (b l : ℕ) : Prop := l * b = 363

-- State the theorem
theorem find_breadth_of_rectangular_plot : ∃ b : ℕ, ∀ l : ℕ, length_is_thrice_breadth b l ∧ area_is_363 b l → b = 11 := 
by
  sorry

end NUMINAMATH_GPT_find_breadth_of_rectangular_plot_l197_19735


namespace NUMINAMATH_GPT_four_by_four_increasing_matrices_l197_19779

noncomputable def count_increasing_matrices (n : ℕ) : ℕ := sorry

theorem four_by_four_increasing_matrices :
  count_increasing_matrices 4 = 320 :=
sorry

end NUMINAMATH_GPT_four_by_four_increasing_matrices_l197_19779


namespace NUMINAMATH_GPT_vasya_fraction_l197_19768

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end NUMINAMATH_GPT_vasya_fraction_l197_19768


namespace NUMINAMATH_GPT_present_age_of_father_l197_19782

-- Definitions based on the conditions
variables (F S : ℕ)
axiom cond1 : F = 3 * S + 3
axiom cond2 : F + 3 = 2 * (S + 3) + 8

-- The theorem to prove
theorem present_age_of_father : F = 27 :=
by
  sorry

end NUMINAMATH_GPT_present_age_of_father_l197_19782


namespace NUMINAMATH_GPT_ways_to_insert_plus_l197_19720

-- Definition of the problem conditions
def num_ones : ℕ := 15
def target_sum : ℕ := 0 

-- Binomial coefficient calculation
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to be proven
theorem ways_to_insert_plus :
  binomial 14 9 = 2002 :=
by
  sorry

end NUMINAMATH_GPT_ways_to_insert_plus_l197_19720


namespace NUMINAMATH_GPT_arccos_one_over_sqrt_two_eq_pi_four_l197_19740

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end NUMINAMATH_GPT_arccos_one_over_sqrt_two_eq_pi_four_l197_19740


namespace NUMINAMATH_GPT_determine_compound_impossible_l197_19784

-- Define the conditions
def contains_Cl (compound : Type) : Prop := true -- Placeholder definition
def mass_percentage_Cl (compound : Type) : ℝ := 0 -- Placeholder definition

-- Define the main statement
theorem determine_compound_impossible (compound : Type) 
  (containsCl : contains_Cl compound) 
  (massPercentageCl : mass_percentage_Cl compound = 47.3) : 
  ∃ (distinct_element : Type), compound = distinct_element := 
sorry

end NUMINAMATH_GPT_determine_compound_impossible_l197_19784


namespace NUMINAMATH_GPT_calculate_stripes_l197_19707

theorem calculate_stripes :
  let olga_stripes_per_shoe := 3
  let rick_stripes_per_shoe := olga_stripes_per_shoe - 1
  let hortense_stripes_per_shoe := olga_stripes_per_shoe * 2
  let ethan_stripes_per_shoe := hortense_stripes_per_shoe + 2
  (olga_stripes_per_shoe * 2 + rick_stripes_per_shoe * 2 + hortense_stripes_per_shoe * 2 + ethan_stripes_per_shoe * 2) / 2 = 19 := 
by
  sorry

end NUMINAMATH_GPT_calculate_stripes_l197_19707


namespace NUMINAMATH_GPT_one_minus_repeating_eight_l197_19769

-- Given the condition
def b : ℚ := 8 / 9

-- The proof problem statement
theorem one_minus_repeating_eight : 1 - b = 1 / 9 := 
by
  sorry  -- proof to be provided

end NUMINAMATH_GPT_one_minus_repeating_eight_l197_19769


namespace NUMINAMATH_GPT_range_of_a_l197_19739

noncomputable def p (x : ℝ) : Prop := abs (3 * x - 4) > 2
noncomputable def q (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0
noncomputable def r (x a : ℝ) : Prop := (x - a) * (x - a - 1) < 0

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, ¬ r x a → ¬ p x) → (a ≥ 2 ∨ a ≤ -1/3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l197_19739


namespace NUMINAMATH_GPT_area_ratio_GHI_JKL_l197_19761

-- Given conditions
def side_lengths_GHI : ℕ × ℕ × ℕ := (6, 8, 10)
def side_lengths_JKL : ℕ × ℕ × ℕ := (9, 12, 15)

-- Function to calculate the area of a right triangle given the lengths of the legs
def right_triangle_area (a b : ℕ) : ℕ :=
  (a * b) / 2

-- Function to determine if a triangle is a right triangle given its side lengths
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Define the main theorem
theorem area_ratio_GHI_JKL :
  let (a₁, b₁, c₁) := side_lengths_GHI
  let (a₂, b₂, c₂) := side_lengths_JKL
  is_right_triangle a₁ b₁ c₁ →
  is_right_triangle a₂ b₂ c₂ →
  right_triangle_area a₁ b₁ % right_triangle_area a₂ b₂ = 4 / 9 :=
by sorry

end NUMINAMATH_GPT_area_ratio_GHI_JKL_l197_19761


namespace NUMINAMATH_GPT_value_of_other_number_l197_19776

theorem value_of_other_number (k : ℕ) (other_number : ℕ) (h1 : k = 2) (h2 : (5 + k) * (5 - k) = 5^2 - other_number) : other_number = 21 :=
  sorry

end NUMINAMATH_GPT_value_of_other_number_l197_19776


namespace NUMINAMATH_GPT_min_value_func_y_l197_19721

noncomputable def geometric_sum (t : ℝ) (n : ℕ) : ℝ :=
  t * 3^(n-1) - (1 / 3)

noncomputable def func_y (x t : ℝ) : ℝ :=
  (x + 2) * (x + 10) / (x + t)

theorem min_value_func_y :
  ∀ (t : ℝ), (∀ n : ℕ, geometric_sum t n = (1) → (∀ x > 0, func_y x t ≥ 16)) :=
  sorry

end NUMINAMATH_GPT_min_value_func_y_l197_19721


namespace NUMINAMATH_GPT_f_sum_positive_l197_19764

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem f_sum_positive (x1 x2 : ℝ) (hx : x1 + x2 > 0) : f x1 + f x2 > 0 :=
sorry

end NUMINAMATH_GPT_f_sum_positive_l197_19764


namespace NUMINAMATH_GPT_rem_frac_l197_19710

def rem (x y : ℚ) : ℚ := x - y * (⌊x / y⌋ : ℤ)

theorem rem_frac : rem (7 / 12) (-3 / 4) = -1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_rem_frac_l197_19710


namespace NUMINAMATH_GPT_two_sectors_area_l197_19727

theorem two_sectors_area {r : ℝ} {θ : ℝ} (h_radius : r = 15) (h_angle : θ = 45) : 
  2 * (θ / 360) * (π * r^2) = 56.25 * π := 
by
  rw [h_radius, h_angle]
  norm_num
  sorry

end NUMINAMATH_GPT_two_sectors_area_l197_19727


namespace NUMINAMATH_GPT_gcd_of_sum_and_product_l197_19726

theorem gcd_of_sum_and_product (x y : ℕ) (h1 : x + y = 1130) (h2 : x * y = 100000) : Int.gcd x y = 2 := 
sorry

end NUMINAMATH_GPT_gcd_of_sum_and_product_l197_19726


namespace NUMINAMATH_GPT_circle_eq_l197_19777

theorem circle_eq (A B : ℝ × ℝ) (hA1 : A = (5, 2)) (hA2 : B = (-1, 4)) (hx : ∃ (c : ℝ), (c, 0) = (c, 0)) :
  ∃ (C : ℝ) (D : ℝ) (x y : ℝ), (x + C) ^ 2 + y ^ 2 = D ∧ D = 20 ∧ (x - 1) ^ 2 + y ^ 2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_circle_eq_l197_19777


namespace NUMINAMATH_GPT_total_peaches_is_85_l197_19762

-- Definitions based on conditions
def initial_peaches : ℝ := 61.0
def additional_peaches : ℝ := 24.0

-- Statement to prove
theorem total_peaches_is_85 :
  initial_peaches + additional_peaches = 85.0 := 
by sorry

end NUMINAMATH_GPT_total_peaches_is_85_l197_19762


namespace NUMINAMATH_GPT_sum_of_coordinates_D_l197_19757

theorem sum_of_coordinates_D (x y : Int) :
  let N := (4, 10)
  let C := (14, 6)
  let D := (x, y)
  N = ((x + 14) / 2, (y + 6) / 2) →
  x + y = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_D_l197_19757


namespace NUMINAMATH_GPT_solve_for_x_l197_19737

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l197_19737


namespace NUMINAMATH_GPT_area_of_quadrilateral_ABDE_l197_19742

-- Definitions for the given problem
variable (AB CE AC DE : ℝ)
variable (parABCE parACDE : Prop)
variable (areaCOD : ℝ)

-- Lean 4 statement for the proof problem
theorem area_of_quadrilateral_ABDE
  (h1 : parABCE)
  (h2 : parACDE)
  (h3 : AB = 5)
  (h4 : AC = 5)
  (h5 : CE = 10)
  (h6 : DE = 10)
  (h7 : areaCOD = 10)
  : (AB + AC + CE + DE) / 2 + areaCOD = 52.5 := 
sorry

end NUMINAMATH_GPT_area_of_quadrilateral_ABDE_l197_19742


namespace NUMINAMATH_GPT_canoe_rental_cost_l197_19767

theorem canoe_rental_cost (C : ℕ) (K : ℕ) :
  18 * K + C * (K + 5) = 405 → 
  3 * K = 2 * (K + 5) → 
  C = 15 :=
by
  intros revenue_eq ratio_eq
  sorry

end NUMINAMATH_GPT_canoe_rental_cost_l197_19767


namespace NUMINAMATH_GPT_volume_of_inscribed_tetrahedron_l197_19741

theorem volume_of_inscribed_tetrahedron (r h : ℝ) (V : ℝ) (tetrahedron_inscribed : Prop) 
  (cylinder_condition : π * r^2 * h = 1) 
  (inscribed : tetrahedron_inscribed → True) : 
  V ≤ 2 / (3 * π) :=
sorry

end NUMINAMATH_GPT_volume_of_inscribed_tetrahedron_l197_19741


namespace NUMINAMATH_GPT_circle_center_radius_l197_19704

theorem circle_center_radius (x y : ℝ) :
  x^2 - 6*x + y^2 + 2*y - 9 = 0 ↔ (x-3)^2 + (y+1)^2 = 19 :=
sorry

end NUMINAMATH_GPT_circle_center_radius_l197_19704


namespace NUMINAMATH_GPT_profit_percentage_is_ten_l197_19734

-- Define the cost price (CP) and selling price (SP) as constants
def CP : ℝ := 90.91
def SP : ℝ := 100

-- Define a theorem to prove the profit percentage is 10%
theorem profit_percentage_is_ten : ((SP - CP) / CP) * 100 = 10 := 
by 
  -- Skip the proof.
  sorry

end NUMINAMATH_GPT_profit_percentage_is_ten_l197_19734


namespace NUMINAMATH_GPT_math_problem_l197_19701

theorem math_problem :
  (Int.ceil ((16 / 5 : ℚ) * (-34 / 4 : ℚ)) - Int.floor ((16 / 5 : ℚ) * Int.floor (-34 / 4 : ℚ))) = 2 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l197_19701


namespace NUMINAMATH_GPT_find_a_of_pure_imaginary_l197_19778

noncomputable def isPureImaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = ⟨0, b⟩  -- complex number z is purely imaginary if it can be written as 0 + bi

theorem find_a_of_pure_imaginary (a : ℝ) (i : ℂ) (ha : i*i = -1) :
  isPureImaginary ((1 - i) * (a + i)) → a = -1 := by
  sorry

end NUMINAMATH_GPT_find_a_of_pure_imaginary_l197_19778


namespace NUMINAMATH_GPT_ratio_of_altitude_to_radius_l197_19705

theorem ratio_of_altitude_to_radius (r R h : ℝ)
  (hR : R = 2 * r)
  (hV : (1/3) * π * R^2 * h = (1/3) * (4/3) * π * r^3) :
  h / R = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_ratio_of_altitude_to_radius_l197_19705


namespace NUMINAMATH_GPT_value_of_x_l197_19770

theorem value_of_x (x : ℝ) (h1 : (x^2 - 4) / (x + 2) = 0) : x = 2 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l197_19770


namespace NUMINAMATH_GPT_polynomial_evaluation_l197_19772

-- Define the polynomial p(x) and the condition p(x) - p'(x) = x^2 + 2x + 1
variable (p : ℝ → ℝ)
variable (hp : ∀ x, p x - (deriv p x) = x^2 + 2 * x + 1)

-- Statement to prove p(5) = 50 given the conditions
theorem polynomial_evaluation : p 5 = 50 := 
sorry

end NUMINAMATH_GPT_polynomial_evaluation_l197_19772


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l197_19789

-- Define initial setup and conditions
def average (scores: List ℚ) : ℚ :=
  scores.sum / scores.length

-- Part (a)
theorem part_a (A B : List ℚ) (a b : ℚ) (A' : List ℚ) (B' : List ℚ) :
  average A = a ∧ average B = b ∧ average A' = a ∧ average B' = b ∧
  average A' > a ∧ average B' > b :=
sorry

-- Part (b)
theorem part_b (A B : List ℚ) : 
  ∀ a b : ℚ, (average A = a ∧ average B = b ∧ ∀ A' : List ℚ, average A' > a ∧ ∀ B' : List ℚ, average B' > b) :=
sorry

-- Part (c)
theorem part_c (A B C : List ℚ) (a b c : ℚ) (A' B' C' A'' B'' C'' : List ℚ) :
  average A = a ∧ average B = b ∧ average C = c ∧
  average A' = a ∧ average B' = b ∧ average C' = c ∧
  average A'' = a ∧ average B'' = b ∧ average C'' = c ∧
  average A' > a ∧ average B' > b ∧ average C' > c ∧
  average A'' > average A' ∧ average B'' > average B' ∧ average C'' > average C' :=
sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l197_19789


namespace NUMINAMATH_GPT_max_has_two_nickels_l197_19745

theorem max_has_two_nickels (n : ℕ) (nickels : ℕ) (coins_value_total : ℕ) :
  (coins_value_total = 15 * n) -> (coins_value_total + 10 = 16 * (n + 1)) -> 
  coins_value_total - nickels * 5 + nickels + 25 = 90 -> 
  n = 6 -> 
  2 = nickels := 
by 
  sorry

end NUMINAMATH_GPT_max_has_two_nickels_l197_19745


namespace NUMINAMATH_GPT_hilton_final_marbles_l197_19752

theorem hilton_final_marbles :
  let initial_marbles := 26
  let found_marbles := 6
  let lost_marbles := 10
  let gift_multiplication_factor := 2
  let marbles_after_find_and_lose := initial_marbles + found_marbles - lost_marbles
  let gift_marbles := gift_multiplication_factor * lost_marbles
  let final_marbles := marbles_after_find_and_lose + gift_marbles
  final_marbles = 42 :=
by
  -- Proof to be filled
  sorry

end NUMINAMATH_GPT_hilton_final_marbles_l197_19752


namespace NUMINAMATH_GPT_lemmings_distance_average_l197_19724

noncomputable def diagonal_length (side: ℝ) : ℝ :=
  Real.sqrt (side^2 + side^2)

noncomputable def fraction_traveled (side: ℝ) (distance: ℝ) : ℝ :=
  distance / (Real.sqrt 2 * side)

noncomputable def final_coordinates (side: ℝ) (distance1: ℝ) (angle: ℝ) (distance2: ℝ) : (ℝ × ℝ) :=
  let frac := fraction_traveled side distance1
  let initial_pos := (frac * side, frac * side)
  let move_dist := distance2 * (Real.sqrt 2 / 2)
  (initial_pos.1 + move_dist, initial_pos.2 + move_dist)

noncomputable def average_shortest_distances (side: ℝ) (coords: ℝ × ℝ) : ℝ :=
  let x_dist := min coords.1 (side - coords.1)
  let y_dist := min coords.2 (side - coords.2)
  (x_dist + (side - x_dist) + y_dist + (side - y_dist)) / 4

theorem lemmings_distance_average :
  let side := 15
  let distance1 := 9.3
  let angle := 45 / 180 * Real.pi -- convert to radians
  let distance2 := 3
  let coords := final_coordinates side distance1 angle distance2
  average_shortest_distances side coords = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_lemmings_distance_average_l197_19724


namespace NUMINAMATH_GPT_average_salary_for_company_l197_19751

variable (n_m : ℕ) -- number of managers
variable (n_a : ℕ) -- number of associates
variable (avg_salary_m : ℕ) -- average salary of managers
variable (avg_salary_a : ℕ) -- average salary of associates

theorem average_salary_for_company (h_n_m : n_m = 15) (h_n_a : n_a = 75) 
  (h_avg_salary_m : avg_salary_m = 90000) (h_avg_salary_a : avg_salary_a = 30000) : 
  (n_m * avg_salary_m + n_a * avg_salary_a) / (n_m + n_a) = 40000 := 
by
  sorry

end NUMINAMATH_GPT_average_salary_for_company_l197_19751


namespace NUMINAMATH_GPT_evaluate_expression_l197_19714

theorem evaluate_expression : 
  let e := 3 + 2 * Real.sqrt 3 + 1 / (3 + 2 * Real.sqrt 3) + 1 / (2 * Real.sqrt 3 - 3)
  e = 3 + 10 * Real.sqrt 3 / 3 :=
by
  let e := 3 + 2 * Real.sqrt 3 + 1 / (3 + 2 * Real.sqrt 3) + 1 / (2 * Real.sqrt 3 - 3)
  have h : e = 3 + 10 * Real.sqrt 3 / 3 := sorry
  exact h

end NUMINAMATH_GPT_evaluate_expression_l197_19714


namespace NUMINAMATH_GPT_cost_of_each_muffin_l197_19718

-- Define the cost of juice
def juice_cost : ℝ := 1.45

-- Define the total cost paid by Kevin
def total_cost : ℝ := 3.70

-- Assume the cost of each muffin
def muffin_cost (M : ℝ) : Prop := 
  3 * M + juice_cost = total_cost

-- The theorem we aim to prove
theorem cost_of_each_muffin : muffin_cost 0.75 :=
by
  -- Here the proof would go
  sorry

end NUMINAMATH_GPT_cost_of_each_muffin_l197_19718


namespace NUMINAMATH_GPT_problem_inequality_l197_19730

theorem problem_inequality (a b : ℝ) (hab : 1 / a + 1 / b = 1) : 
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) := 
by
  sorry

end NUMINAMATH_GPT_problem_inequality_l197_19730


namespace NUMINAMATH_GPT_maximize_xyz_l197_19712

theorem maximize_xyz (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 60) :
    (x, y, z) = (20, 40 / 3, 80 / 3) → x^3 * y^2 * z^4 ≤ 20^3 * (40 / 3)^2 * (80 / 3)^4 :=
by
  sorry

end NUMINAMATH_GPT_maximize_xyz_l197_19712


namespace NUMINAMATH_GPT_no_solution_xy_l197_19788

theorem no_solution_xy (x y : ℕ) : ¬ (x * (x + 1) = 4 * y * (y + 1)) :=
sorry

end NUMINAMATH_GPT_no_solution_xy_l197_19788


namespace NUMINAMATH_GPT_correct_mark_l197_19733

theorem correct_mark
  (n : ℕ)
  (initial_avg : ℝ)
  (wrong_mark : ℝ)
  (correct_avg : ℝ)
  (correct_total_marks : ℝ)
  (actual_total_marks : ℝ)
  (final_mark : ℝ) :
  n = 25 →
  initial_avg = 100 →
  wrong_mark = 60 →
  correct_avg = 98 →
  correct_total_marks = (n * correct_avg) →
  actual_total_marks = (n * initial_avg - wrong_mark + final_mark) →
  correct_total_marks = actual_total_marks →
  final_mark = 10 :=
by
  intros h_n h_initial_avg h_wrong_mark h_correct_avg h_correct_total_marks h_actual_total_marks h_eq
  sorry

end NUMINAMATH_GPT_correct_mark_l197_19733


namespace NUMINAMATH_GPT_math_problem_l197_19775

noncomputable def base10_b := 25 + 1  -- 101_5 in base 10
noncomputable def base10_c := 343 + 98 + 21 + 4  -- 1234_7 in base 10
noncomputable def base10_d := 2187 + 324 + 45 + 6  -- 3456_9 in base 10

theorem math_problem (a : ℕ) (b c d : ℕ) (h_a : a = 2468)
  (h_b : b = base10_b) (h_c : c = base10_c) (h_d : d = base10_d) :
  (a / b) * c - d = 41708 :=
  by {
  sorry
}

end NUMINAMATH_GPT_math_problem_l197_19775


namespace NUMINAMATH_GPT_number_of_correct_statements_l197_19750

def statement1_condition : Prop :=
∀ a b : ℝ, (a - b > 0) → (a > 0 ∧ b > 0)

def statement2_condition : Prop :=
∀ a b : ℝ, a - b = a + (-b)

def statement3_condition : Prop :=
∀ a : ℝ, (a - (-a) = 0)

def statement4_condition : Prop :=
∀ a : ℝ, 0 - a = -a

theorem number_of_correct_statements : 
  (¬ statement1_condition ∧ statement2_condition ∧ ¬ statement3_condition ∧ statement4_condition) →
  (2 = 2) :=
by
  intros
  trivial

end NUMINAMATH_GPT_number_of_correct_statements_l197_19750


namespace NUMINAMATH_GPT_single_discount_equivalent_l197_19754

theorem single_discount_equivalent :
  ∀ (original final: ℝ) (d1 d2 d3 total_discount: ℝ),
  original = 800 →
  d1 = 0.15 →
  d2 = 0.10 →
  d3 = 0.05 →
  final = original * (1 - d1) * (1 - d2) * (1 - d3) →
  total_discount = 1 - (final / original) →
  total_discount = 0.27325 :=
by
  intros original final d1 d2 d3 total_discount h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_single_discount_equivalent_l197_19754


namespace NUMINAMATH_GPT_mira_result_l197_19771

def round_to_nearest_hundred (n : ℤ) : ℤ :=
  if n % 100 >= 50 then n / 100 * 100 + 100 else n / 100 * 100

theorem mira_result :
  round_to_nearest_hundred ((63 + 48) - 21) = 100 :=
by
  sorry

end NUMINAMATH_GPT_mira_result_l197_19771


namespace NUMINAMATH_GPT_average_probable_weight_l197_19717

-- Define the conditions
def Arun_opinion (w : ℝ) : Prop := 64 < w ∧ w < 72
def Brother_opinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def Mother_opinion (w : ℝ) : Prop := w ≤ 67

-- The proof problem statement
theorem average_probable_weight :
  ∃ (w : ℝ), Arun_opinion w ∧ Brother_opinion w ∧ Mother_opinion w →
  (64 + 67) / 2 = 65.5 :=
by
  sorry

end NUMINAMATH_GPT_average_probable_weight_l197_19717


namespace NUMINAMATH_GPT_find_length_AB_l197_19797

open Real

noncomputable def AB_length := 
  let r := 4
  let V_total := 320 * π
  ∃ (L : ℝ), 16 * π * L + (256 / 3) * π = V_total ∧ L = 44 / 3

theorem find_length_AB :
  AB_length := by
  sorry

end NUMINAMATH_GPT_find_length_AB_l197_19797


namespace NUMINAMATH_GPT_students_taking_all_three_classes_l197_19756

variable (students : Finset ℕ)
variable (yoga bridge painting : Finset ℕ)

variables (yoga_count bridge_count painting_count at_least_two exactly_two all_three : ℕ)

variable (total_students : students.card = 25)
variable (yoga_students : yoga.card = 12)
variable (bridge_students : bridge.card = 15)
variable (painting_students : painting.card = 11)
variable (at_least_two_classes : at_least_two = 10)
variable (exactly_two_classes : exactly_two = 7)

theorem students_taking_all_three_classes :
  all_three = 3 :=
sorry

end NUMINAMATH_GPT_students_taking_all_three_classes_l197_19756


namespace NUMINAMATH_GPT_total_number_of_cows_l197_19702

theorem total_number_of_cows (n : ℕ) 
  (h1 : n > 0) 
  (h2 : (1/3) * n + (1/6) * n + (1/8) * n + 9 = n) : n = 216 :=
sorry

end NUMINAMATH_GPT_total_number_of_cows_l197_19702


namespace NUMINAMATH_GPT_tangerine_count_l197_19715

-- Definitions based directly on the conditions
def initial_oranges : ℕ := 5
def remaining_oranges : ℕ := initial_oranges - 2
def remaining_tangerines (T : ℕ) : ℕ := T - 10
def condition1 (T : ℕ) : Prop := remaining_tangerines T = remaining_oranges + 4

-- Theorem to prove the number of tangerines in the bag
theorem tangerine_count (T : ℕ) (h : condition1 T) : T = 17 :=
by
  sorry

end NUMINAMATH_GPT_tangerine_count_l197_19715


namespace NUMINAMATH_GPT_frac_m_q_eq_one_l197_19744

theorem frac_m_q_eq_one (m n p q : ℕ) 
  (h1 : m = 40 * n)
  (h2 : p = 5 * n)
  (h3 : p = q / 8) : (m / q = 1) :=
by
  sorry

end NUMINAMATH_GPT_frac_m_q_eq_one_l197_19744


namespace NUMINAMATH_GPT_bugs_meet_at_point_P_l197_19781

theorem bugs_meet_at_point_P (r1 r2 v1 v2 t : ℝ) (h1 : r1 = 7) (h2 : r2 = 3) (h3 : v1 = 4 * Real.pi) (h4 : v2 = 3 * Real.pi) :
  t = 14 :=
by
  repeat { sorry }

end NUMINAMATH_GPT_bugs_meet_at_point_P_l197_19781


namespace NUMINAMATH_GPT_fraction_addition_l197_19723

theorem fraction_addition (a b : ℕ) (hb : b ≠ 0) (h : a / (b : ℚ) = 3 / 5) : (a + b) / (b : ℚ) = 8 / 5 := 
by
sorry

end NUMINAMATH_GPT_fraction_addition_l197_19723


namespace NUMINAMATH_GPT_solution_set_of_abs_2x_minus_1_ge_3_l197_19787

theorem solution_set_of_abs_2x_minus_1_ge_3 :
  { x : ℝ | |2 * x - 1| ≥ 3 } = { x : ℝ | x ≤ -1 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end NUMINAMATH_GPT_solution_set_of_abs_2x_minus_1_ge_3_l197_19787


namespace NUMINAMATH_GPT_temperature_difference_l197_19793

theorem temperature_difference (highest lowest : ℝ) (h_high : highest = 27) (h_low : lowest = 17) :
  highest - lowest = 10 :=
by
  sorry

end NUMINAMATH_GPT_temperature_difference_l197_19793


namespace NUMINAMATH_GPT_car_trip_time_l197_19758

theorem car_trip_time (T A : ℕ) (h1 : 50 * T = 140 + 53 * A) (h2 : T = 4 + A) : T = 24 := by
  sorry

end NUMINAMATH_GPT_car_trip_time_l197_19758


namespace NUMINAMATH_GPT_original_movie_length_l197_19713

theorem original_movie_length (final_length cut_scene original_length : ℕ) 
    (h1 : cut_scene = 3) (h2 : final_length = 57) (h3 : final_length + cut_scene = original_length) : 
  original_length = 60 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_original_movie_length_l197_19713


namespace NUMINAMATH_GPT_find_number_l197_19743

noncomputable def least_common_multiple (a b : ℕ) : ℕ := Nat.lcm a b

theorem find_number (n : ℕ) (h1 : least_common_multiple (least_common_multiple n 16) (least_common_multiple 18 24) = 144) : n = 9 :=
sorry

end NUMINAMATH_GPT_find_number_l197_19743


namespace NUMINAMATH_GPT_value_of_2_pow_5_plus_5_l197_19760

theorem value_of_2_pow_5_plus_5 : 2^5 + 5 = 37 := by
  sorry

end NUMINAMATH_GPT_value_of_2_pow_5_plus_5_l197_19760


namespace NUMINAMATH_GPT_find_term_number_l197_19759

-- Define the arithmetic sequence
def arithmetic_seq (a d : Int) (n : Int) := a + (n - 1) * d

-- Define the condition: first term and common difference
def a1 := 4
def d := 3

-- Prove that the 672nd term is 2017
theorem find_term_number (n : Int) (h : arithmetic_seq a1 d n = 2017) : n = 672 := by
  sorry

end NUMINAMATH_GPT_find_term_number_l197_19759


namespace NUMINAMATH_GPT_sum_of_digits_of_d_l197_19766

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem sum_of_digits_of_d (d : ℕ) 
  (h_exchange : 15 * d = 9 * (d * 5 / 3)) 
  (h_spending : (5 * d / 3) - 120 = d) 
  (h_d_eq : d = 180) : sum_of_digits d = 9 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_d_l197_19766


namespace NUMINAMATH_GPT_johns_age_l197_19774

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end NUMINAMATH_GPT_johns_age_l197_19774


namespace NUMINAMATH_GPT_find_side_length_of_square_l197_19709

theorem find_side_length_of_square (n k : ℕ) (hk : k ≥ 1) (h : (n + k) * (n + k) - n * n = 47) : n = 23 :=
  sorry

end NUMINAMATH_GPT_find_side_length_of_square_l197_19709


namespace NUMINAMATH_GPT_trains_cross_time_l197_19795

noncomputable def time_to_cross (length_train : ℝ) (speed_train_kmph : ℝ) : ℝ :=
  let relative_speed_kmph := speed_train_kmph + speed_train_kmph
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600)
  let total_distance := length_train + length_train
  total_distance / relative_speed_mps

theorem trains_cross_time :
  time_to_cross 180 80 = 8.1 := 
by
  sorry

end NUMINAMATH_GPT_trains_cross_time_l197_19795


namespace NUMINAMATH_GPT_february_max_diff_percentage_l197_19748

noncomputable def max_diff_percentage (D B F : ℕ) : ℚ :=
  let avg_others := (B + F) / 2
  let high_sales := max (max D B) F
  (high_sales - avg_others) / avg_others * 100

theorem february_max_diff_percentage :
  max_diff_percentage 8 5 6 = 45.45 := by
  sorry

end NUMINAMATH_GPT_february_max_diff_percentage_l197_19748


namespace NUMINAMATH_GPT_percentage_difference_l197_19706

theorem percentage_difference :
    let A := (40 / 100) * ((50 / 100) * 60)
    let B := (50 / 100) * ((60 / 100) * 70)
    (B - A) = 9 :=
by
    sorry

end NUMINAMATH_GPT_percentage_difference_l197_19706


namespace NUMINAMATH_GPT_partition_natural_numbers_l197_19719

theorem partition_natural_numbers :
  ∃ (f : ℕ → ℕ), (∀ n, 1 ≤ f n ∧ f n ≤ 100) ∧
  (∀ a b c, a + 99 * b = c → f a = f c ∨ f a = f b ∨ f b = f c) :=
sorry

end NUMINAMATH_GPT_partition_natural_numbers_l197_19719


namespace NUMINAMATH_GPT_smallest_possible_value_of_n_l197_19791

theorem smallest_possible_value_of_n 
  {a b c m n : ℕ} 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) 
  (hc_pos : c > 0) 
  (h_ordering : a ≥ b ∧ b ≥ c) 
  (h_sum : a + b + c = 3010) 
  (h_factorial : a.factorial * b.factorial * c.factorial = m * 10^n) 
  (h_m_not_div_10 : ¬ (10 ∣ m)) 
  : n = 746 := 
sorry

end NUMINAMATH_GPT_smallest_possible_value_of_n_l197_19791


namespace NUMINAMATH_GPT_sum_of_possible_values_l197_19732

variable {S : ℝ} (h : S ≠ 0)

theorem sum_of_possible_values (h : S ≠ 0) : ∃ N : ℝ, N ≠ 0 ∧ 6 * N + 2 / N = S → ∀ N1 N2 : ℝ, (6 * N1 + 2 / N1 = S ∧ 6 * N2 + 2 / N2 = S) → (N1 + N2) = S / 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_l197_19732


namespace NUMINAMATH_GPT_gardening_project_cost_l197_19738

def cost_rose_bushes (number_of_bushes: ℕ) (cost_per_bush: ℕ) : ℕ := number_of_bushes * cost_per_bush
def cost_gardener (hourly_rate: ℕ) (hours_per_day: ℕ) (days: ℕ) : ℕ := hourly_rate * hours_per_day * days
def cost_soil (cubic_feet: ℕ) (cost_per_cubic_foot: ℕ) : ℕ := cubic_feet * cost_per_cubic_foot

theorem gardening_project_cost :
  cost_rose_bushes 20 150 + cost_gardener 30 5 4 + cost_soil 100 5 = 4100 :=
by
  sorry

end NUMINAMATH_GPT_gardening_project_cost_l197_19738


namespace NUMINAMATH_GPT_count_5_numbers_after_996_l197_19780

theorem count_5_numbers_after_996 : 
  ∃ a b c d e, a = 997 ∧ b = 998 ∧ c = 999 ∧ d = 1000 ∧ e = 1001 :=
sorry

end NUMINAMATH_GPT_count_5_numbers_after_996_l197_19780


namespace NUMINAMATH_GPT_product_of_areas_eq_k3_times_square_of_volume_l197_19798

variables (a b c k : ℝ)

-- Defining the areas of bottom, side, and front of the box as provided
def area_bottom := k * a * b
def area_side := k * b * c
def area_front := k * c * a

-- Volume of the box
def volume := a * b * c

-- The lean statement to be proved
theorem product_of_areas_eq_k3_times_square_of_volume :
  (area_bottom a b k) * (area_side b c k) * (area_front c a k) = k^3 * (volume a b c)^2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_areas_eq_k3_times_square_of_volume_l197_19798


namespace NUMINAMATH_GPT_triangle_inequality_l197_19799

open Real

theorem triangle_inequality (A B C : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) (h_sum : A + B + C = π) :
  sin A * cos C + A * cos B > 0 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l197_19799


namespace NUMINAMATH_GPT_tan_15_deg_product_l197_19725

theorem tan_15_deg_product : (1 + Real.tan 15) * (1 + Real.tan 15) = 2.1433 := by
  sorry

end NUMINAMATH_GPT_tan_15_deg_product_l197_19725


namespace NUMINAMATH_GPT_find_side_length_l197_19794

theorem find_side_length
  (n : ℕ) 
  (h : (6 * n^2) / (6 * n^3) = 1 / 3) : 
  n = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_side_length_l197_19794


namespace NUMINAMATH_GPT_difference_between_numbers_l197_19700

theorem difference_between_numbers :
  ∃ S : ℝ, L = 1650 ∧ L = 6 * S + 15 ∧ L - S = 1377.5 :=
sorry

end NUMINAMATH_GPT_difference_between_numbers_l197_19700


namespace NUMINAMATH_GPT_transport_cost_l197_19708

theorem transport_cost (weight_g : ℕ) (cost_per_kg : ℕ) (weight_kg : ℕ) (total_cost : ℕ)
  (h1 : weight_g = 2000)
  (h2 : cost_per_kg = 15000)
  (h3 : weight_kg = weight_g / 1000)
  (h4 : total_cost = weight_kg * cost_per_kg) :
  total_cost = 30000 :=
by
  sorry

end NUMINAMATH_GPT_transport_cost_l197_19708


namespace NUMINAMATH_GPT_range_of_k_has_extreme_values_on_interval_l197_19790

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * Real.log x - x^2 + 3 * x

theorem range_of_k_has_extreme_values_on_interval (k : ℝ) (h : k ≠ 0) :
  -9/8 < k ∧ k < 0 :=
sorry

end NUMINAMATH_GPT_range_of_k_has_extreme_values_on_interval_l197_19790


namespace NUMINAMATH_GPT_divides_power_sum_l197_19749

theorem divides_power_sum (a b c : ℤ) (h : a + b + c ∣ a^2 + b^2 + c^2) : ∀ k : ℕ, a + b + c ∣ a^(2^k) + b^(2^k) + c^(2^k) :=
by
  intro k
  induction k with
  | zero =>
    sorry -- Base case proof
  | succ k ih =>
    sorry -- Inductive step proof

end NUMINAMATH_GPT_divides_power_sum_l197_19749


namespace NUMINAMATH_GPT_complement_of_intersection_l197_19755

theorem complement_of_intersection (U M N : Set ℕ)
  (hU : U = {1, 2, 3, 4})
  (hM : M = {1, 2, 3})
  (hN : N = {2, 3, 4}) :
  (U \ (M ∩ N)) = {1, 4} :=
by
  rw [hU, hM, hN]
  sorry

end NUMINAMATH_GPT_complement_of_intersection_l197_19755


namespace NUMINAMATH_GPT_canonical_equations_of_line_intersection_l197_19785

theorem canonical_equations_of_line_intersection
  (x y z : ℝ)
  (h1 : 2 * x - 3 * y + z + 6 = 0)
  (h2 : x - 3 * y - 2 * z + 3 = 0) :
  (∃ (m n p x0 y0 z0 : ℝ), 
  m * (x + 3) = n * y ∧ n * y = p * z ∧ 
  m = 9 ∧ n = 5 ∧ p = -3 ∧ 
  x0 = -3 ∧ y0 = 0 ∧ z0 = 0) :=
sorry

end NUMINAMATH_GPT_canonical_equations_of_line_intersection_l197_19785


namespace NUMINAMATH_GPT_distinct_pairs_disjoint_subsets_l197_19763

theorem distinct_pairs_disjoint_subsets (n : ℕ) : 
  ∃ k, k = (3^n + 1) / 2 := 
sorry

end NUMINAMATH_GPT_distinct_pairs_disjoint_subsets_l197_19763


namespace NUMINAMATH_GPT_number_of_pencils_l197_19786

-- Define the given conditions
def circle_radius : ℝ := 14 -- 14 feet radius
def pencil_length_inches : ℝ := 6 -- 6-inch pencil

noncomputable def pencil_length_feet : ℝ := pencil_length_inches / 12 -- convert 6 inches to feet

-- Statement of the problem in Lean
theorem number_of_pencils (r : ℝ) (p_len_inch : ℝ) (d : ℝ) (p_len_feet : ℝ) :
  r = circle_radius →
  p_len_inch = pencil_length_inches →
  d = 2 * r →
  p_len_feet = pencil_length_feet →
  d / p_len_feet = 56 :=
by
  intros hr hp hd hpl
  sorry

end NUMINAMATH_GPT_number_of_pencils_l197_19786


namespace NUMINAMATH_GPT_tanA_over_tanB_l197_19722

noncomputable def tan_ratios (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A + 2 * c = 0

theorem tanA_over_tanB {A B C a b c : ℝ} (h : tan_ratios A B C a b c) : 
  Real.tan A / Real.tan B = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tanA_over_tanB_l197_19722


namespace NUMINAMATH_GPT_find_pairs_l197_19753

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ q r : ℕ, a^2 + b^2 = (a + b) * q + r ∧ q^2 + r = 1977) →
  (a, b) = (50, 37) ∨ (a, b) = (37, 50) ∨ (a, b) = (50, 7) ∨ (a, b) = (7, 50) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l197_19753


namespace NUMINAMATH_GPT_find_rate_percent_l197_19711

-- Definitions based on the given conditions
def principal : ℕ := 800
def time : ℕ := 4
def simple_interest : ℕ := 192
def si_formula (P R T : ℕ) : ℕ := P * R * T / 100

-- Statement: prove that the rate percent (R) is 6%
theorem find_rate_percent (R : ℕ) (h : simple_interest = si_formula principal R time) : R = 6 :=
sorry

end NUMINAMATH_GPT_find_rate_percent_l197_19711


namespace NUMINAMATH_GPT_part1_part2_part3_l197_19796

-- Part 1
def harmonic_fraction (num denom : ℚ) : Prop :=
  ∃ a b : ℚ, num = a - 2 * b ∧ denom = a^2 - b^2 ∧ ¬(∃ x : ℚ, a - 2 * b = (a - b) * x)

theorem part1 (a b : ℚ) (h : harmonic_fraction (a - 2 * b) (a^2 - b^2)) : true :=
  by sorry

-- Part 2
theorem part2 (a : ℕ) (h : harmonic_fraction (x - 1) (x^2 + a * x + 4)) : a = 4 ∨ a = 5 :=
  by sorry

-- Part 3
theorem part3 (a b : ℚ) :
  (4 * a^2 / (a * b^2 - b^3) - a / b * 4 / b) = (4 * a / (ab - b^2)) :=
  by sorry

end NUMINAMATH_GPT_part1_part2_part3_l197_19796


namespace NUMINAMATH_GPT_chocolates_150_satisfies_l197_19783

def chocolates_required (chocolates : ℕ) : Prop :=
  chocolates ≥ 150 ∧ chocolates % 19 = 17

theorem chocolates_150_satisfies : chocolates_required 150 :=
by
  -- We need to show that 150 satisfies the conditions:
  -- 1. 150 ≥ 150
  -- 2. 150 % 19 = 17
  unfold chocolates_required
  -- Both conditions hold:
  exact And.intro (by linarith) (by norm_num)

end NUMINAMATH_GPT_chocolates_150_satisfies_l197_19783
