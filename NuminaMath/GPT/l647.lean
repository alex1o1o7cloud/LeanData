import Mathlib

namespace NUMINAMATH_GPT_geometric_progression_common_point_l647_64797

theorem geometric_progression_common_point (a r : ℝ) :
  ∀ x y : ℝ, (a ≠ 0 ∧ x = 1 ∧ y = 0) ↔ (a * x + (a * r) * y = a * r^2) := by
  sorry

end NUMINAMATH_GPT_geometric_progression_common_point_l647_64797


namespace NUMINAMATH_GPT_intersection_at_most_one_l647_64769

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the statement to be proved
theorem intersection_at_most_one (a : ℝ) :
  ∀ x1 x2 : ℝ, f x1 = f x2 → x1 = x2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_at_most_one_l647_64769


namespace NUMINAMATH_GPT_find_pool_depth_l647_64789

noncomputable def pool_depth (rate volume capacity_percent time length width : ℝ) :=
  volume / (length * width * rate * time / capacity_percent)

theorem find_pool_depth :
  pool_depth 60 75000 0.8 1000 150 50 = 10 := by
  simp [pool_depth] -- Simplifying the complex expression should lead to the solution.
  sorry

end NUMINAMATH_GPT_find_pool_depth_l647_64789


namespace NUMINAMATH_GPT_sum_series_l647_64713

theorem sum_series :
  ∑' n:ℕ, (4 * n ^ 2 - 2 * n + 3) / 3 ^ n = 21 / 4 :=
sorry

end NUMINAMATH_GPT_sum_series_l647_64713


namespace NUMINAMATH_GPT_alyssa_limes_correct_l647_64768

-- Definitions representing the conditions
def fred_limes : Nat := 36
def nancy_limes : Nat := 35
def total_limes : Nat := 103

-- Definition of the number of limes Alyssa picked
def alyssa_limes : Nat := total_limes - (fred_limes + nancy_limes)

-- The theorem we need to prove
theorem alyssa_limes_correct : alyssa_limes = 32 := by
  sorry

end NUMINAMATH_GPT_alyssa_limes_correct_l647_64768


namespace NUMINAMATH_GPT_probability_correct_l647_64757

-- Definitions of the problem components
def total_beads : Nat := 7
def red_beads : Nat := 4
def white_beads : Nat := 2
def green_bead : Nat := 1

-- The total number of permutations of the given multiset
def total_permutations : Nat :=
  Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial green_bead)

-- The number of valid permutations where no two neighboring beads are the same color
def valid_permutations : Nat := 14 -- As derived in the solution steps

-- The probability that no two neighboring beads are the same color
def probability_no_adjacent_same_color : Rat :=
  valid_permutations / total_permutations

-- The theorem to be proven
theorem probability_correct :
  probability_no_adjacent_same_color = 2 / 15 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_probability_correct_l647_64757


namespace NUMINAMATH_GPT_find_de_l647_64756

def magic_square (f : ℕ × ℕ → ℕ) : Prop :=
  (f (0, 0) = 30) ∧ (f (0, 1) = 20) ∧ (f (0, 2) = f (0, 2)) ∧
  (f (1, 0) = f (1, 0)) ∧ (f (1, 1) = f (1, 1)) ∧ (f (1, 2) = f (1, 2)) ∧
  (f (2, 0) = 24) ∧ (f (2, 1) = 32) ∧ (f (2, 2) = f (2, 2)) ∧
  (f (0, 0) + f (0, 1) + f (0, 2) = f (1, 0) + f (1, 1) + f (1, 2)) ∧
  (f (0, 0) + f (0, 1) + f (0, 2) = f (2, 0) + f (2, 1) + f (2, 2)) ∧
  (f (0, 0) + f (0, 1) + f (0, 2) = f (0, 0) + f (1, 0) + f (2, 0)) ∧
  (f (0, 0) + f (1, 1) + f (2, 2) = f (0, 2) + f (1, 1) + f (2, 0)) 

theorem find_de (f : ℕ × ℕ → ℕ) (h : magic_square f) : 
  (f (1, 0) + f (1, 1) = 54) :=
sorry

end NUMINAMATH_GPT_find_de_l647_64756


namespace NUMINAMATH_GPT_odd_coefficients_in_binomial_expansion_l647_64700

theorem odd_coefficients_in_binomial_expansion :
  let a : Fin 9 → ℕ := fun k => Nat.choose 8 k
  (Finset.filter (fun k => a k % 2 = 1) (Finset.Icc 0 8)).card = 2 := by
  sorry

end NUMINAMATH_GPT_odd_coefficients_in_binomial_expansion_l647_64700


namespace NUMINAMATH_GPT_compute_expression_l647_64746

theorem compute_expression : 12 * (1 / 15) * 30 = 24 := 
by 
  sorry

end NUMINAMATH_GPT_compute_expression_l647_64746


namespace NUMINAMATH_GPT_math_problem_l647_64773

theorem math_problem :
  (625.3729 * (4500 + 2300 ^ 2) - Real.sqrt 84630) / (1500 ^ 3 * 48 ^ 2) = 0.0004257 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l647_64773


namespace NUMINAMATH_GPT_prime_sum_of_digits_base_31_l647_64736

-- Define the sum of digits function in base k
def sum_of_digits_in_base (k n : ℕ) : ℕ :=
  let digits := (Nat.digits k n)
  digits.foldr (· + ·) 0

theorem prime_sum_of_digits_base_31 (p : ℕ) (hp : Nat.Prime p) (h_bound : p < 20000) : 
  sum_of_digits_in_base 31 p = 49 ∨ sum_of_digits_in_base 31 p = 77 :=
by
  sorry

end NUMINAMATH_GPT_prime_sum_of_digits_base_31_l647_64736


namespace NUMINAMATH_GPT_tan_beta_half_l647_64752

theorem tan_beta_half (α β : ℝ)
    (h1 : Real.tan α = 1 / 3)
    (h2 : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
    Real.tan β = 1 / 2 := 
sorry

end NUMINAMATH_GPT_tan_beta_half_l647_64752


namespace NUMINAMATH_GPT_conic_sections_l647_64745

theorem conic_sections (x y : ℝ) :
  y^4 - 9 * x^4 = 3 * y^2 - 4 →
  (∃ c : ℝ, (c = 5/2 ∨ c = 1) ∧ y^2 - 3 * x^2 = c) :=
by
  sorry

end NUMINAMATH_GPT_conic_sections_l647_64745


namespace NUMINAMATH_GPT_remainder_369963_div_6_is_3_l647_64706

def is_divisible_by (a b : ℕ) : Prop := b ∣ a

def remainder_when_divided (a b : ℕ) (r : ℕ) : Prop := a % b = r

theorem remainder_369963_div_6_is_3 :
  remainder_when_divided 369963 6 3 :=
by
  have h₁ : 369963 % 2 = 1 := by
    sorry -- It is known that 369963 is not divisible by 2.
  have h₂ : 369963 % 3 = 0 := by
    sorry -- It is known that 369963 is divisible by 3.
  have h₃ : 369963 % 6 = 3 := by
    sorry -- From the above properties, derive that the remainder when 369963 is divided by 6 is 3.
  exact h₃

end NUMINAMATH_GPT_remainder_369963_div_6_is_3_l647_64706


namespace NUMINAMATH_GPT_range_of_a_l647_64755

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a) ↔ (a < 1 ∨ a > 3) := 
sorry

end NUMINAMATH_GPT_range_of_a_l647_64755


namespace NUMINAMATH_GPT_total_pages_is_360_l647_64759

-- Definitions from conditions
variable (A B : ℕ) -- Rates of printer A and printer B in pages per minute.
variable (total_pages : ℕ) -- Total number of pages of the task.

-- Given conditions
axiom h1 : 24 * (A + B) = total_pages -- Condition from both printers working together.
axiom h2 : 60 * A = total_pages -- Condition from printer A alone.
axiom h3 : B = A + 3 -- Condition of printer B printing 3 more pages per minute.

-- Goal: Prove the total number of pages is 360
theorem total_pages_is_360 : total_pages = 360 := 
by 
  sorry

end NUMINAMATH_GPT_total_pages_is_360_l647_64759


namespace NUMINAMATH_GPT_parallel_lines_sufficient_necessity_l647_64764

theorem parallel_lines_sufficient_necessity (a : ℝ) :
  ¬ (a = 1 ↔ (∀ x : ℝ, a^2 * x + 1 = x - 1)) := 
sorry

end NUMINAMATH_GPT_parallel_lines_sufficient_necessity_l647_64764


namespace NUMINAMATH_GPT_trapezoid_area_is_8_l647_64702

noncomputable def trapezoid_area
  (AB CD : ℝ)        -- lengths of the bases
  (h : ℝ)            -- height (distance between the bases)
  : ℝ :=
  0.5 * (AB + CD) * h

theorem trapezoid_area_is_8 
  (AB CD : ℝ) 
  (h : ℝ) 
  (K M : ℝ) 
  (height_condition : h = 2)
  (AB_condition : AB = 5)
  (CD_condition : CD = 3)
  (K_midpoint : K = AB / 2) 
  (M_midpoint : M = CD / 2)
  : trapezoid_area AB CD h = 8 :=
by
  rw [trapezoid_area, AB_condition, CD_condition, height_condition]
  norm_num

end NUMINAMATH_GPT_trapezoid_area_is_8_l647_64702


namespace NUMINAMATH_GPT_scientific_notation_example_l647_64741

theorem scientific_notation_example : (5.2 * 10^5) = 520000 := sorry

end NUMINAMATH_GPT_scientific_notation_example_l647_64741


namespace NUMINAMATH_GPT_product_of_three_numbers_l647_64725

theorem product_of_three_numbers (x y z n : ℕ) 
  (h1 : x + y + z = 200)
  (h2 : n = 8 * x)
  (h3 : n = y - 5)
  (h4 : n = z + 5) :
  x * y * z = 372462 :=
by sorry

end NUMINAMATH_GPT_product_of_three_numbers_l647_64725


namespace NUMINAMATH_GPT_checkerboard_red_squares_l647_64718

/-- Define the properties of the checkerboard -/
structure Checkerboard :=
  (size : ℕ)
  (colors : ℕ → ℕ → String)
  (corner_color : String)

/-- Our checkerboard patterning function -/
def checkerboard_colors (i j : ℕ) : String :=
  match (i + j) % 3 with
  | 0 => "blue"
  | 1 => "yellow"
  | _ => "red"

/-- Our checkerboard of size 33x33 -/
def chubby_checkerboard : Checkerboard :=
  { size := 33,
    colors := checkerboard_colors,
    corner_color := "blue" }

/-- Proof that the number of red squares is 363 -/
theorem checkerboard_red_squares (b : Checkerboard) (h1 : b.size = 33) (h2 : b.colors = checkerboard_colors) : ∃ n, n = 363 :=
  by sorry

end NUMINAMATH_GPT_checkerboard_red_squares_l647_64718


namespace NUMINAMATH_GPT_sum_of_youngest_and_oldest_cousins_l647_64778

theorem sum_of_youngest_and_oldest_cousins :
  ∃ (ages : Fin 5 → ℝ), (∃ (a1 a5 : ℝ), ages 0 = a1 ∧ ages 4 = a5 ∧ a1 + a5 = 29) ∧
                        (∃ (median : ℝ), median = ages 2 ∧ median = 7) ∧
                        (∃ (mean : ℝ), mean = (ages 0 + ages 1 + ages 2 + ages 3 + ages 4) / 5 ∧ mean = 10) :=
by sorry

end NUMINAMATH_GPT_sum_of_youngest_and_oldest_cousins_l647_64778


namespace NUMINAMATH_GPT_total_selling_price_l647_64720

theorem total_selling_price (CP : ℕ) (num_toys : ℕ) (gain_toys : ℕ) (TSP : ℕ)
  (h1 : CP = 1300)
  (h2 : num_toys = 18)
  (h3 : gain_toys = 3) :
  TSP = 27300 := by
  sorry

end NUMINAMATH_GPT_total_selling_price_l647_64720


namespace NUMINAMATH_GPT_betty_paid_total_l647_64723

def cost_slippers (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_lipsticks (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_hair_colors (count : ℕ) (price : ℝ) : ℝ := count * price

def total_cost := 
  cost_slippers 6 2.5 +
  cost_lipsticks 4 1.25 +
  cost_hair_colors 8 3

theorem betty_paid_total :
  total_cost = 44 := 
  sorry

end NUMINAMATH_GPT_betty_paid_total_l647_64723


namespace NUMINAMATH_GPT_parabola_directrix_x_eq_neg1_eqn_l647_64737

theorem parabola_directrix_x_eq_neg1_eqn :
  (∀ y : ℝ, ∃ x : ℝ, x = -1 → y^2 = 4 * x) :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_x_eq_neg1_eqn_l647_64737


namespace NUMINAMATH_GPT_zoo_with_hippos_only_l647_64721

variables {Z : Type} -- The type of all zoos
variables (H R G : Set Z) -- Subsets of zoos with hippos, rhinos, and giraffes respectively

-- Conditions
def condition1 : Prop := ∀ (z : Z), z ∈ H ∧ z ∈ R → z ∉ G
def condition2 : Prop := ∀ (z : Z), z ∈ R ∧ z ∉ G → z ∈ H
def condition3 : Prop := ∀ (z : Z), z ∈ H ∧ z ∈ G → z ∈ R

-- Goal
def goal : Prop := ∃ (z : Z), z ∈ H ∧ z ∉ G ∧ z ∉ R

-- Theorem statement
theorem zoo_with_hippos_only (h1 : condition1 H R G) (h2 : condition2 H R G) (h3 : condition3 H R G) : goal H R G :=
sorry

end NUMINAMATH_GPT_zoo_with_hippos_only_l647_64721


namespace NUMINAMATH_GPT_min_surveyed_consumers_l647_64754

theorem min_surveyed_consumers (N : ℕ) 
    (h10 : ∃ k : ℕ, N = 10 * k)
    (h30 : ∃ l : ℕ, N = 10 * l) 
    (h40 : ∃ m : ℕ, N = 5 * m) : 
    N = 10 :=
by
  sorry

end NUMINAMATH_GPT_min_surveyed_consumers_l647_64754


namespace NUMINAMATH_GPT_quadratic_distinct_roots_l647_64748

theorem quadratic_distinct_roots (a : ℝ) : 
  (a > -1 ∧ a ≠ 3) ↔ 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (a - 3) * x₁^2 - 4 * x₁ - 1 = 0 ∧ 
    (a - 3) * x₂^2 - 4 * x₂ - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_roots_l647_64748


namespace NUMINAMATH_GPT_sum_of_digits_of_7_pow_1974_l647_64761

-- Define the number \(7^{1974}\)
def num := 7^1974

-- Function to extract the last two digits
def last_two_digits (n : ℕ) : ℕ := n % 100

-- Function to compute the sum of the tens and units digits
def sum_tens_units (n : ℕ) : ℕ :=
  let last_two := last_two_digits n
  (last_two / 10) + (last_two % 10)

theorem sum_of_digits_of_7_pow_1974 : sum_tens_units num = 9 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_7_pow_1974_l647_64761


namespace NUMINAMATH_GPT_polygon_num_sides_l647_64716

-- Define the given conditions
def perimeter : ℕ := 150
def side_length : ℕ := 15

-- State the theorem to prove the number of sides of the polygon
theorem polygon_num_sides (P : ℕ) (s : ℕ) (hP : P = perimeter) (hs : s = side_length) : P / s = 10 :=
by
  sorry

end NUMINAMATH_GPT_polygon_num_sides_l647_64716


namespace NUMINAMATH_GPT_parallelogram_sides_l647_64731

theorem parallelogram_sides (x y : ℝ) 
    (h1 : 4 * y + 2 = 12) 
    (h2 : 6 * x - 2 = 10)
    (h3 : 10 + 12 + (6 * x - 2) + (4 * y + 2) = 68) :
    x + y = 4.5 := 
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_parallelogram_sides_l647_64731


namespace NUMINAMATH_GPT_number_of_games_in_season_l647_64729

-- Define the number of teams and divisions
def num_teams := 20
def num_divisions := 4
def teams_per_division := 5

-- Define the games played within and between divisions
def intra_division_games_per_team := 12  -- 4 teams * 3 games each
def inter_division_games_per_team := 15  -- (20 - 5) teams * 1 game each

-- Define the total number of games played by each team
def total_games_per_team := intra_division_games_per_team + inter_division_games_per_team

-- Define the total number of games played (double-counting needs to be halved)
def total_games (num_teams : ℕ) (total_games_per_team : ℕ) : ℕ :=
  (num_teams * total_games_per_team) / 2

-- The theorem to be proven
theorem number_of_games_in_season :
  total_games num_teams total_games_per_team = 270 :=
by
  sorry

end NUMINAMATH_GPT_number_of_games_in_season_l647_64729


namespace NUMINAMATH_GPT_rainfall_ratio_l647_64727

theorem rainfall_ratio (R1 R2 : ℕ) (hR2 : R2 = 24) (hTotal : R1 + R2 = 40) : 
  R2 / R1 = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_rainfall_ratio_l647_64727


namespace NUMINAMATH_GPT_jason_spent_at_music_store_l647_64751

theorem jason_spent_at_music_store 
  (cost_flute : ℝ) (cost_music_tool : ℝ) (cost_song_book : ℝ)
  (h1 : cost_flute = 142.46)
  (h2 : cost_music_tool = 8.89)
  (h3 : cost_song_book = 7) :
  cost_flute + cost_music_tool + cost_song_book = 158.35 :=
by
  -- assumption proof
  sorry

end NUMINAMATH_GPT_jason_spent_at_music_store_l647_64751


namespace NUMINAMATH_GPT_externally_tangent_circles_solution_l647_64770

theorem externally_tangent_circles_solution (R1 R2 d : Real)
  (h1 : R1 > 0) (h2 : R2 > 0) (h3 : R1 + R2 > d) :
  (1/R1) + (1/R2) = 2/d :=
sorry

end NUMINAMATH_GPT_externally_tangent_circles_solution_l647_64770


namespace NUMINAMATH_GPT_simplify_fraction_product_l647_64777

theorem simplify_fraction_product :
  8 * (15 / 14) * (-49 / 45) = - (28 / 3) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_product_l647_64777


namespace NUMINAMATH_GPT_convert_degrees_to_radians_l647_64710

theorem convert_degrees_to_radians : 
  (-390) * (Real.pi / 180) = - (13 * Real.pi / 6) := 
by 
  sorry

end NUMINAMATH_GPT_convert_degrees_to_radians_l647_64710


namespace NUMINAMATH_GPT_survival_rate_is_98_l647_64701

def total_flowers := 150
def unsurviving_flowers := 3
def surviving_flowers := total_flowers - unsurviving_flowers

theorem survival_rate_is_98 : (surviving_flowers : ℝ) / total_flowers * 100 = 98 := by
  sorry

end NUMINAMATH_GPT_survival_rate_is_98_l647_64701


namespace NUMINAMATH_GPT_n_is_power_of_three_l647_64787

theorem n_is_power_of_three {n : ℕ} (hn_pos : 0 < n) (p : Nat.Prime (4^n + 2^n + 1)) :
  ∃ (a : ℕ), n = 3^a :=
by
  sorry

end NUMINAMATH_GPT_n_is_power_of_three_l647_64787


namespace NUMINAMATH_GPT_base8_to_base10_correct_l647_64784

def base8_to_base10_conversion : Prop :=
  (2 * 8^2 + 4 * 8^1 + 6 * 8^0 = 166)

theorem base8_to_base10_correct : base8_to_base10_conversion :=
by
  sorry

end NUMINAMATH_GPT_base8_to_base10_correct_l647_64784


namespace NUMINAMATH_GPT_find_m_l647_64743

-- Define the sets A and B and the conditions
def A : Set ℝ := {x | x ≥ 3}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- Define the conditions on these sets
def conditions (m : ℝ) : Prop :=
  (∀ x, x ∈ A ∨ x ∈ B m) ∧ (∀ x, ¬(x ∈ A ∧ x ∈ B m))

-- State the theorem
theorem find_m : ∃ m : ℝ, conditions m ∧ m = 3 :=
  sorry

end NUMINAMATH_GPT_find_m_l647_64743


namespace NUMINAMATH_GPT_class_size_l647_64712

theorem class_size (g : ℕ) (h1 : g + (g + 3) = 44) (h2 : g^2 + (g + 3)^2 = 540) : g + (g + 3) = 44 :=
by
  sorry

end NUMINAMATH_GPT_class_size_l647_64712


namespace NUMINAMATH_GPT_positive_difference_of_perimeters_is_zero_l647_64775

-- Definitions of given conditions
def rect1_length : ℕ := 5
def rect1_width : ℕ := 1
def rect2_first_rect_length : ℕ := 3
def rect2_first_rect_width : ℕ := 2
def rect2_second_rect_length : ℕ := 1
def rect2_second_rect_width : ℕ := 2

-- Perimeter calculation functions
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)
def rect1_perimeter := perimeter rect1_length rect1_width
def rect2_extended_length : ℕ := rect2_first_rect_length + rect2_second_rect_length
def rect2_extended_width : ℕ := rect2_first_rect_width
def rect2_perimeter := perimeter rect2_extended_length rect2_extended_width

-- The positive difference of the perimeters
def positive_difference (a b : ℕ) : ℕ := if a > b then a - b else b - a

-- The Lean 4 statement to be proven
theorem positive_difference_of_perimeters_is_zero :
    positive_difference rect1_perimeter rect2_perimeter = 0 := by
  sorry

end NUMINAMATH_GPT_positive_difference_of_perimeters_is_zero_l647_64775


namespace NUMINAMATH_GPT_sum_fourth_powers_eq_t_l647_64753

theorem sum_fourth_powers_eq_t (a b t : ℝ) (h1 : a + b = t) (h2 : a^2 + b^2 = t) (h3 : a^3 + b^3 = t) : 
  a^4 + b^4 = t := 
by
  sorry

end NUMINAMATH_GPT_sum_fourth_powers_eq_t_l647_64753


namespace NUMINAMATH_GPT_gcd_3060_561_l647_64747

theorem gcd_3060_561 : Nat.gcd 3060 561 = 51 :=
by
  sorry

end NUMINAMATH_GPT_gcd_3060_561_l647_64747


namespace NUMINAMATH_GPT_min_total_bags_l647_64734

theorem min_total_bags (x y : ℕ) (h : 15 * x + 8 * y = 1998) (hy_min : ∀ y', (15 * x + 8 * y' = 1998) → y ≤ y') :
  x + y = 140 :=
by
  sorry

end NUMINAMATH_GPT_min_total_bags_l647_64734


namespace NUMINAMATH_GPT_wire_ratio_is_one_l647_64732

theorem wire_ratio_is_one (a b : ℝ) (h1 : a = b) : a / b = 1 := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_wire_ratio_is_one_l647_64732


namespace NUMINAMATH_GPT_pairings_without_alice_and_bob_l647_64717

theorem pairings_without_alice_and_bob (n : ℕ) (h : n = 12) : 
    ∃ k : ℕ, k = ((n * (n - 1)) / 2) - 1 ∧ k = 65 :=
by
  sorry

end NUMINAMATH_GPT_pairings_without_alice_and_bob_l647_64717


namespace NUMINAMATH_GPT_smallest_multiple_of_45_and_60_not_divisible_by_18_l647_64742

noncomputable def smallest_multiple_not_18 (n : ℕ) : Prop :=
  (n % 45 = 0) ∧
  (n % 60 = 0) ∧
  (n % 18 ≠ 0) ∧
  ∀ m : ℕ, (m % 45 = 0) ∧ (m % 60 = 0) ∧ (m % 18 ≠ 0) → n ≤ m

theorem smallest_multiple_of_45_and_60_not_divisible_by_18 : ∃ n : ℕ, smallest_multiple_not_18 n ∧ n = 810 := 
by
  existsi 810
  sorry

end NUMINAMATH_GPT_smallest_multiple_of_45_and_60_not_divisible_by_18_l647_64742


namespace NUMINAMATH_GPT_sue_final_answer_is_67_l647_64765

-- Declare the initial value Ben thinks of
def ben_initial_number : ℕ := 4

-- Ben's calculation function
def ben_number (b : ℕ) : ℕ := ((b + 2) * 3) + 5

-- Sue's calculation function
def sue_number (x : ℕ) : ℕ := ((x - 3) * 3) + 7

-- Define the final number Sue calculates
def final_sue_number : ℕ := sue_number (ben_number ben_initial_number)

-- Prove that Sue's final number is 67
theorem sue_final_answer_is_67 : final_sue_number = 67 :=
by 
  sorry

end NUMINAMATH_GPT_sue_final_answer_is_67_l647_64765


namespace NUMINAMATH_GPT_bus_stop_time_l647_64795

theorem bus_stop_time
  (speed_without_stoppage : ℝ := 54)
  (speed_with_stoppage : ℝ := 45)
  (distance_diff : ℝ := speed_without_stoppage - speed_with_stoppage)
  (distance : ℝ := distance_diff)
  (speed_km_per_min : ℝ := speed_without_stoppage / 60) :
  distance / speed_km_per_min = 10 :=
by
  -- The proof steps would go here.
  sorry

end NUMINAMATH_GPT_bus_stop_time_l647_64795


namespace NUMINAMATH_GPT_potassium_salt_average_molar_mass_l647_64760

noncomputable def average_molar_mass (total_weight : ℕ) (num_moles : ℕ) : ℕ :=
  total_weight / num_moles

theorem potassium_salt_average_molar_mass :
  let total_weight := 672
  let num_moles := 4
  average_molar_mass total_weight num_moles = 168 := by
    sorry

end NUMINAMATH_GPT_potassium_salt_average_molar_mass_l647_64760


namespace NUMINAMATH_GPT_repeating_decimal_division_l647_64707

def repeating_decimal_081_as_fraction : ℚ := 9 / 11
def repeating_decimal_272_as_fraction : ℚ := 30 / 11

theorem repeating_decimal_division : 
  (repeating_decimal_081_as_fraction / repeating_decimal_272_as_fraction) = (3 / 10) := 
by 
  sorry

end NUMINAMATH_GPT_repeating_decimal_division_l647_64707


namespace NUMINAMATH_GPT_trains_crossing_time_l647_64704

noncomputable def time_to_cross_each_other (L T1 T2 : ℝ) (H1 : L = 120) (H2 : T1 = 10) (H3 : T2 = 16) : ℝ :=
  let S1 := L / T1
  let S2 := L / T2
  let S := S1 + S2
  let D := L + L
  D / S

theorem trains_crossing_time : time_to_cross_each_other 120 10 16 (by rfl) (by rfl) (by rfl) = 240 / (12 + 7.5) :=
  sorry

end NUMINAMATH_GPT_trains_crossing_time_l647_64704


namespace NUMINAMATH_GPT_solve_real_equation_l647_64798

theorem solve_real_equation (x : ℝ) (h : (x + 2)^4 + x^4 = 82) : x = 1 ∨ x = -3 :=
  sorry

end NUMINAMATH_GPT_solve_real_equation_l647_64798


namespace NUMINAMATH_GPT_adjugate_power_null_l647_64708

variable {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ)

def adjugate (A : Matrix (Fin n) (Fin n) ℂ) : Matrix (Fin n) (Fin n) ℂ := sorry

theorem adjugate_power_null (A : Matrix (Fin n) (Fin n) ℂ) (m : ℕ) (hm : 0 < m) (h : (adjugate A) ^ m = 0) : 
  (adjugate A) ^ 2 = 0 := 
sorry

end NUMINAMATH_GPT_adjugate_power_null_l647_64708


namespace NUMINAMATH_GPT_average_is_12_or_15_l647_64766

variable {N : ℝ} (h : 12 < N ∧ N < 22)

theorem average_is_12_or_15 : (∃ x ∈ ({12, 15} : Set ℝ), x = (9 + 15 + N) / 3) :=
by
  have h1 : 12 < (24 + N) / 3 := by sorry
  have h2 : (24 + N) / 3 < 15.3333 := by sorry
  sorry

end NUMINAMATH_GPT_average_is_12_or_15_l647_64766


namespace NUMINAMATH_GPT_money_left_is_40_l647_64749

-- Define the quantities spent by Mildred and Candice, and the total money provided by their mom.
def MildredSpent : ℕ := 25
def CandiceSpent : ℕ := 35
def TotalGiven : ℕ := 100

-- Calculate the total spent and the remaining money.
def TotalSpent := MildredSpent + CandiceSpent
def MoneyLeft := TotalGiven - TotalSpent

-- Prove that the money left is 40.
theorem money_left_is_40 : MoneyLeft = 40 := by
  sorry

end NUMINAMATH_GPT_money_left_is_40_l647_64749


namespace NUMINAMATH_GPT_f_always_positive_l647_64771

def f (x : ℝ) : ℝ := x^2 + 3 * x + 4

theorem f_always_positive : ∀ x : ℝ, f x > 0 := 
by 
  sorry

end NUMINAMATH_GPT_f_always_positive_l647_64771


namespace NUMINAMATH_GPT_list_scores_lowest_highest_l647_64791

variable (M Q S T : ℕ)

axiom Quay_thinks : Q = T
axiom Marty_thinks : M > T
axiom Shana_thinks : S < T
axiom Tana_thinks : T ≠ max M (max Q (max S T)) ∧ T ≠ min M (min Q (min S T))

theorem list_scores_lowest_highest : (S < T) ∧ (T = Q) ∧ (Q < M) ↔ (S < T) ∧ (T < M) :=
by
  sorry

end NUMINAMATH_GPT_list_scores_lowest_highest_l647_64791


namespace NUMINAMATH_GPT_solve_ab_sum_l647_64794

theorem solve_ab_sum (x a b : ℝ) (ha : ℕ) (hb : ℕ)
  (h1 : a = ha)
  (h2 : b = hb)
  (h3 : x = a + Real.sqrt b)
  (h4 : x^2 + 3 * x + 3 / x + 1 / x^2 = 26) :
  (ha + hb = 5) :=
sorry

end NUMINAMATH_GPT_solve_ab_sum_l647_64794


namespace NUMINAMATH_GPT_roots_of_polynomial_l647_64730

def p (x : ℝ) : ℝ := x^3 + x^2 - 4*x - 4

theorem roots_of_polynomial :
  (p (-1) = 0) ∧ (p 2 = 0) ∧ (p (-2) = 0) ∧ 
  ∀ x, p x = 0 → (x = -1 ∨ x = 2 ∨ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l647_64730


namespace NUMINAMATH_GPT_log_x_squared_y_squared_l647_64774

theorem log_x_squared_y_squared (x y : ℝ) (h1 : Real.log (x * y^2) = 2) (h2 : Real.log (x^3 * y) = 2) : 
  Real.log (x^2 * y^2) = 12 / 5 := 
by
  sorry

end NUMINAMATH_GPT_log_x_squared_y_squared_l647_64774


namespace NUMINAMATH_GPT_solve_price_per_litre_second_oil_l647_64726

variable (P : ℝ)

def price_per_litre_second_oil :=
  10 * 55 + 5 * P = 15 * 58.67

theorem solve_price_per_litre_second_oil (h : price_per_litre_second_oil P) : P = 66.01 :=
  by
  sorry

end NUMINAMATH_GPT_solve_price_per_litre_second_oil_l647_64726


namespace NUMINAMATH_GPT_sin_of_3halfpiplus2theta_l647_64776

theorem sin_of_3halfpiplus2theta (θ : ℝ) (h : Real.tan θ = 1 / 3) : Real.sin (3 * π / 2 + 2 * θ) = -4 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_sin_of_3halfpiplus2theta_l647_64776


namespace NUMINAMATH_GPT_find_remainder_l647_64703

theorem find_remainder : ∃ r : ℝ, r = 14 ∧ 13698 = (153.75280898876406 * 89) + r := 
by
  sorry

end NUMINAMATH_GPT_find_remainder_l647_64703


namespace NUMINAMATH_GPT_infinite_series_sum_l647_64709

theorem infinite_series_sum (c d : ℝ) (h1 : 0 < d) (h2 : 0 < c) (h3 : c > d) :
  (∑' n, 1 / (((2 * n + 1) * c - n * d) * ((2 * (n+1) - 1) * c - (n + 1 - 1) * d))) = 1 / ((c - d) * d) :=
sorry

end NUMINAMATH_GPT_infinite_series_sum_l647_64709


namespace NUMINAMATH_GPT_hockey_season_duration_l647_64788

theorem hockey_season_duration 
  (total_games : ℕ)
  (games_per_month : ℕ)
  (h_total : total_games = 182)
  (h_monthly : games_per_month = 13) : 
  total_games / games_per_month = 14 := 
by
  sorry

end NUMINAMATH_GPT_hockey_season_duration_l647_64788


namespace NUMINAMATH_GPT_recruit_people_l647_64782

variable (average_contribution : ℝ) (total_funds_needed : ℝ) (current_funds : ℝ)

theorem recruit_people (h₁ : average_contribution = 10) (h₂ : current_funds = 200) (h₃ : total_funds_needed = 1000) : 
    (total_funds_needed - current_funds) / average_contribution = 80 := by
  sorry

end NUMINAMATH_GPT_recruit_people_l647_64782


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_35_l647_64728

/-- The smallest four-digit number that is divisible by 35 is 1050. -/
theorem smallest_four_digit_divisible_by_35 : ∃ n, (1000 <= n) ∧ (n <= 9999) ∧ (n % 35 = 0) ∧ ∀ m, (1000 <= m) ∧ (m <= 9999) ∧ (m % 35 = 0) → n <= m :=
by
  existsi (1050 : ℕ)
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_35_l647_64728


namespace NUMINAMATH_GPT_factorization_correct_l647_64735

theorem factorization_correct: ∀ (x : ℝ), (x^2 - 9 = (x + 3) * (x - 3)) := 
sorry

end NUMINAMATH_GPT_factorization_correct_l647_64735


namespace NUMINAMATH_GPT_maximize_f_l647_64799

noncomputable def f (x : ℝ) : ℝ := 4 * x - 2 + 1 / (4 * x - 5)

theorem maximize_f (x : ℝ) (h : x < 5 / 4): ∃ M, (∀ y, (y < 5 / 4) → f y ≤ M) ∧ M = 1 := by
  sorry

end NUMINAMATH_GPT_maximize_f_l647_64799


namespace NUMINAMATH_GPT_required_barrels_of_pitch_l647_64785

def total_road_length : ℕ := 16
def bags_of_gravel_per_truckload : ℕ := 2
def barrels_of_pitch_per_truckload (bgt : ℕ) : ℚ := bgt / 5
def truckloads_per_mile : ℕ := 3

def miles_paved_day1 : ℕ := 4
def miles_paved_day2 : ℕ := (miles_paved_day1 * 2) - 1
def total_miles_paved_first_two_days : ℕ := miles_paved_day1 + miles_paved_day2
def remaining_miles_paved_day3 : ℕ := total_road_length - total_miles_paved_first_two_days

def truckloads_needed (miles : ℕ) : ℕ := miles * truckloads_per_mile
def barrels_of_pitch_needed (truckloads : ℕ) (bgt : ℕ) : ℚ := truckloads * barrels_of_pitch_per_truckload bgt

theorem required_barrels_of_pitch : 
  barrels_of_pitch_needed (truckloads_needed remaining_miles_paved_day3) bags_of_gravel_per_truckload = 6 := 
by
  sorry

end NUMINAMATH_GPT_required_barrels_of_pitch_l647_64785


namespace NUMINAMATH_GPT_width_of_paving_stone_l647_64758

-- Given conditions as definitions
def length_of_courtyard : ℝ := 40
def width_of_courtyard : ℝ := 16.5
def number_of_stones : ℕ := 132
def length_of_stone : ℝ := 2.5

-- Define the total area of the courtyard
def area_of_courtyard := length_of_courtyard * width_of_courtyard

-- Define the equation we need to prove
theorem width_of_paving_stone :
  (length_of_stone * W * number_of_stones = area_of_courtyard) → W = 2 :=
by
  sorry

end NUMINAMATH_GPT_width_of_paving_stone_l647_64758


namespace NUMINAMATH_GPT_wall_length_proof_l647_64796

-- Define the initial conditions
def men1 : ℕ := 20
def days1 : ℕ := 8
def men2 : ℕ := 86
def days2 : ℕ := 8
def wall_length2 : ℝ := 283.8

-- Define the expected length of the wall for the first condition
def expected_length : ℝ := 65.7

-- The proof statement.
theorem wall_length_proof : ((men1 * days1) / (men2 * days2)) * wall_length2 = expected_length :=
sorry

end NUMINAMATH_GPT_wall_length_proof_l647_64796


namespace NUMINAMATH_GPT_probability_of_distinct_dice_numbers_l647_64793

/-- Total number of outcomes when rolling five six-sided dice. -/
def total_outcomes : ℕ := 6 ^ 5

/-- Number of favorable outcomes where all five dice show distinct numbers. -/
def favorable_outcomes : ℕ := 6 * 5 * 4 * 3 * 2

/-- Calculating the probability as a fraction. -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_distinct_dice_numbers :
  probability = 5 / 54 :=
by
  -- Proof is required here.
  sorry

end NUMINAMATH_GPT_probability_of_distinct_dice_numbers_l647_64793


namespace NUMINAMATH_GPT_impossible_all_black_l647_64781

def initial_white_chessboard (n : ℕ) : Prop :=
  n = 0

def move_inverts_three (move : ℕ → ℕ) : Prop :=
  ∀ n, move n = n + 3 ∨ move n = n - 3

theorem impossible_all_black (move : ℕ → ℕ) (n : ℕ) (initial : initial_white_chessboard n) (invert : move_inverts_three move) : ¬ ∃ k, move^[k] n = 64 :=
by sorry

end NUMINAMATH_GPT_impossible_all_black_l647_64781


namespace NUMINAMATH_GPT_Raine_total_steps_l647_64739

-- Define the steps taken to and from school each day
def Monday_steps_to_school := 150
def Monday_steps_back := 170
def Tuesday_steps_to_school := 140
def Tuesday_steps_back := 140 + 30
def Wednesday_steps_to_school := 160
def Wednesday_steps_back := 210
def Thursday_steps_to_school := 150
def Thursday_steps_back := 140 + 30
def Friday_steps_to_school := 180
def Friday_steps_back := 200

-- Define total steps for each day
def Monday_total_steps := Monday_steps_to_school + Monday_steps_back
def Tuesday_total_steps := Tuesday_steps_to_school + Tuesday_steps_back
def Wednesday_total_steps := Wednesday_steps_to_school + Wednesday_steps_back
def Thursday_total_steps := Thursday_steps_to_school + Thursday_steps_back
def Friday_total_steps := Friday_steps_to_school + Friday_steps_back

-- Define the total steps for all five days
def total_steps :=
  Monday_total_steps +
  Tuesday_total_steps +
  Wednesday_total_steps +
  Thursday_total_steps +
  Friday_total_steps

-- Prove that the total steps equals 1700
theorem Raine_total_steps : total_steps = 1700 := 
by 
  unfold total_steps
  unfold Monday_total_steps Tuesday_total_steps Wednesday_total_steps Thursday_total_steps Friday_total_steps
  unfold Monday_steps_to_school Monday_steps_back
  unfold Tuesday_steps_to_school Tuesday_steps_back
  unfold Wednesday_steps_to_school Wednesday_steps_back
  unfold Thursday_steps_to_school Thursday_steps_back
  unfold Friday_steps_to_school Friday_steps_back
  sorry

end NUMINAMATH_GPT_Raine_total_steps_l647_64739


namespace NUMINAMATH_GPT_wall_height_l647_64714

theorem wall_height (length width depth total_bricks: ℕ) (h: ℕ) (H_length: length = 20) (H_width: width = 4) (H_depth: depth = 2) (H_total_bricks: total_bricks = 800) :
  80 * depth * h = total_bricks → h = 5 :=
by
  intros H_eq
  sorry

end NUMINAMATH_GPT_wall_height_l647_64714


namespace NUMINAMATH_GPT_arithmetic_common_difference_l647_64783

-- Define the conditions of the arithmetic sequence
def a (n : ℕ) := 0 -- This is a placeholder definition since we only care about a_5 and a_12
def a5 : ℝ := 10
def a12 : ℝ := 31

-- State the proof problem
theorem arithmetic_common_difference :
  ∃ d : ℝ, a5 + 7 * d = a12 :=
by
  use 3
  simp [a5, a12]
  sorry

end NUMINAMATH_GPT_arithmetic_common_difference_l647_64783


namespace NUMINAMATH_GPT_compare_abc_l647_64762

noncomputable def a : ℝ := (0.6)^(2/5)
noncomputable def b : ℝ := (0.4)^(2/5)
noncomputable def c : ℝ := (0.4)^(3/5)

theorem compare_abc : a > b ∧ b > c := 
by
  sorry

end NUMINAMATH_GPT_compare_abc_l647_64762


namespace NUMINAMATH_GPT_Jill_water_volume_l647_64779

theorem Jill_water_volume 
  (n : ℕ) (h₀ : 3 * n = 48) :
  n * (1 / 4) + n * (1 / 2) + n * 1 = 28 := 
by 
  sorry

end NUMINAMATH_GPT_Jill_water_volume_l647_64779


namespace NUMINAMATH_GPT_wrong_observation_value_l647_64767

theorem wrong_observation_value (n : ℕ) (initial_mean corrected_mean correct_value wrong_value : ℚ) 
  (h₁ : n = 50)
  (h₂ : initial_mean = 36)
  (h₃ : corrected_mean = 36.5)
  (h₄ : correct_value = 60)
  (h₅ : n * corrected_mean = n * initial_mean - wrong_value + correct_value) :
  wrong_value = 35 := by
  have htotal₁ : n * initial_mean = 1800 := by sorry
  have htotal₂ : n * corrected_mean = 1825 := by sorry
  linarith

end NUMINAMATH_GPT_wrong_observation_value_l647_64767


namespace NUMINAMATH_GPT_ordered_pair_count_l647_64763

noncomputable def count_pairs (n : ℕ) (c : ℕ) : ℕ := 
  (if n < c then 0 else n - c + 1)

theorem ordered_pair_count :
  (count_pairs 39 5 = 35) :=
sorry

end NUMINAMATH_GPT_ordered_pair_count_l647_64763


namespace NUMINAMATH_GPT_factor_expression_l647_64715

theorem factor_expression (x : ℝ) : 100 * x ^ 23 + 225 * x ^ 46 = 25 * x ^ 23 * (4 + 9 * x ^ 23) :=
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_factor_expression_l647_64715


namespace NUMINAMATH_GPT_area_of_circumscribed_circle_l647_64733

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  ∃ A : ℝ, A = 48 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_area_of_circumscribed_circle_l647_64733


namespace NUMINAMATH_GPT_joaozinho_multiplication_l647_64722

theorem joaozinho_multiplication :
  12345679 * 9 = 111111111 :=
by
  sorry

end NUMINAMATH_GPT_joaozinho_multiplication_l647_64722


namespace NUMINAMATH_GPT_sum_of_factors_of_1000_l647_64750

-- Define what it means for an integer to not contain the digit '0'
def no_zero_digits (n : ℕ) : Prop :=
∀ c ∈ (n.digits 10), c ≠ 0

-- Define the problem statement
theorem sum_of_factors_of_1000 :
  ∃ (a b : ℕ), a * b = 1000 ∧ no_zero_digits a ∧ no_zero_digits b ∧ (a + b = 133) :=
sorry

end NUMINAMATH_GPT_sum_of_factors_of_1000_l647_64750


namespace NUMINAMATH_GPT_domain_of_f_l647_64780

noncomputable def f (x : ℝ) := 2 ^ (Real.sqrt (3 - x)) + 1 / (x - 1)

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, y = f x) ↔ (x ≤ 3 ∧ x ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l647_64780


namespace NUMINAMATH_GPT_sandy_money_left_l647_64740

theorem sandy_money_left (total_money : ℝ) (spent_percentage : ℝ) (money_left : ℝ) : 
  total_money = 320 → spent_percentage = 0.30 → money_left = (total_money * (1 - spent_percentage)) → 
  money_left = 224 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_sandy_money_left_l647_64740


namespace NUMINAMATH_GPT_P_2n_expression_l647_64711

noncomputable def a (n : ℕ) : ℕ :=
  2 * n + 1

noncomputable def S (n : ℕ) : ℕ :=
  n * (n + 2)

noncomputable def b (n : ℕ) : ℕ :=
  2 ^ (n - 1)

noncomputable def T (n : ℕ) : ℕ :=
  2 * b n - 1

noncomputable def c (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 / S n else a n * b n
  
noncomputable def P (n : ℕ) : ℕ :=
  if n % 2 = 0 then (12 * n - 1) * 2^(2 * n + 1) / 9 + 2 / 9 + 2 * n / (2 * n + 1) else 0

theorem P_2n_expression (n : ℕ) : 
  P (2 * n) = (12 * n - 1) * 2^(2 * n + 1) / 9 + 2 / 9 + 2 * n / (2 * n + 1) :=
sorry

end NUMINAMATH_GPT_P_2n_expression_l647_64711


namespace NUMINAMATH_GPT_inequality_preserved_l647_64744

variable {a b c : ℝ}

theorem inequality_preserved (h : abs ((a^2 + b^2 - c^2) / (a * b)) < 2) :
    abs ((b^2 + c^2 - a^2) / (b * c)) < 2 ∧ abs ((c^2 + a^2 - b^2) / (c * a)) < 2 := 
sorry

end NUMINAMATH_GPT_inequality_preserved_l647_64744


namespace NUMINAMATH_GPT_part_a_part_b_l647_64792

-- Let γ and δ represent acute angles, γ < δ implies γ - sin γ < δ - sin δ 
theorem part_a (alpha beta : ℝ) (h_alpha : 0 < alpha) (h_alpha2 : alpha < π/2) 
  (h_beta : 0 < beta) (h_beta2 : beta < π/2) (h : alpha < beta) : 
  alpha - Real.sin alpha < beta - Real.sin beta := sorry

-- Let γ and δ represent acute angles, γ < δ implies tan γ - γ < tan δ - δ 
theorem part_b (alpha beta : ℝ) (h_alpha : 0 < alpha) (h_alpha2 : alpha < π/2) 
  (h_beta : 0 < beta) (h_beta2 : beta < π/2) (h : alpha < beta) : 
  Real.tan alpha - alpha < Real.tan beta - beta := sorry

end NUMINAMATH_GPT_part_a_part_b_l647_64792


namespace NUMINAMATH_GPT_f_odd_f_monotonic_range_of_x_l647_64786

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 ((1 + x) / (1 - x))

theorem f_odd : ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x := by
  sorry

theorem f_monotonic : ∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂ := by
  sorry

theorem range_of_x (x : ℝ) : f (1 / (x - 3)) + f (- 1 / 3) < 0 → x < 2 ∨ x > 6 := by
  sorry

end NUMINAMATH_GPT_f_odd_f_monotonic_range_of_x_l647_64786


namespace NUMINAMATH_GPT_colin_avg_time_l647_64724

def totalTime (a b c d : ℕ) : ℕ := a + b + c + d

def averageTime (total_time miles : ℕ) : ℕ := total_time / miles

theorem colin_avg_time :
  let first_mile := 6
  let second_mile := 5
  let third_mile := 5
  let fourth_mile := 4
  let total_time := totalTime first_mile second_mile third_mile fourth_mile
  4 > 0 -> averageTime total_time 4 = 5 :=
by
  intros
  -- proof goes here
  sorry

end NUMINAMATH_GPT_colin_avg_time_l647_64724


namespace NUMINAMATH_GPT_total_questions_l647_64705

theorem total_questions (f s k : ℕ) (hf : f = 36) (hs : s = 2 * f) (hk : k = (f + s) / 2) :
  2 * (f + s + k) = 324 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_questions_l647_64705


namespace NUMINAMATH_GPT_total_boys_went_down_slide_l647_64738

theorem total_boys_went_down_slide :
  let boys_first_10_minutes := 22
  let boys_next_5_minutes := 13
  let boys_last_20_minutes := 35
  (boys_first_10_minutes + boys_next_5_minutes + boys_last_20_minutes) = 70 :=
by
  sorry

end NUMINAMATH_GPT_total_boys_went_down_slide_l647_64738


namespace NUMINAMATH_GPT_cost_of_each_book_l647_64772

theorem cost_of_each_book 
  (B : ℝ)
  (num_books_plant : ℕ)
  (num_books_fish : ℕ)
  (num_magazines : ℕ)
  (cost_magazine : ℝ)
  (total_spent : ℝ) :
  num_books_plant = 9 →
  num_books_fish = 1 →
  num_magazines = 10 →
  cost_magazine = 2 →
  total_spent = 170 →
  10 * B + 10 * cost_magazine = total_spent →
  B = 15 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_cost_of_each_book_l647_64772


namespace NUMINAMATH_GPT_arithmetic_mean_correct_l647_64719

-- Define the expressions
def expr1 (x : ℤ) := x + 12
def expr2 (y : ℤ) := y
def expr3 (x : ℤ) := 3 * x
def expr4 := 18
def expr5 (x : ℤ) := 3 * x + 6

-- The condition as a hypothesis
def condition (x y : ℤ) : Prop := (expr1 x + expr2 y + expr3 x + expr4 + expr5 x) / 5 = 30

-- The theorem to prove
theorem arithmetic_mean_correct : condition 6 72 :=
sorry

end NUMINAMATH_GPT_arithmetic_mean_correct_l647_64719


namespace NUMINAMATH_GPT_geometric_sequence_b_value_l647_64790

theorem geometric_sequence_b_value
  (b : ℝ)
  (hb_pos : b > 0)
  (hgeom : ∃ r : ℝ, 30 * r = b ∧ b * r = 3 / 8) :
  b = 7.5 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_b_value_l647_64790
