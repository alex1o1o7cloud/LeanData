import Mathlib

namespace unique_multiple_of_6_l358_35840

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def last_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem unique_multiple_of_6 :
  ∀ n : ℕ, 63470 ≤ n ∧ n ≤ 63479 →
    (is_multiple_of_6 n ↔ n = 63474) :=
by sorry

end unique_multiple_of_6_l358_35840


namespace exhibits_permutation_l358_35832

theorem exhibits_permutation : Nat.factorial 5 = 120 := by
  sorry

end exhibits_permutation_l358_35832


namespace m_greater_than_one_l358_35811

theorem m_greater_than_one (m : ℝ) : (∀ x : ℝ, |x| ≤ 1 → x < m) → m > 1 := by
  sorry

end m_greater_than_one_l358_35811


namespace two_solutions_l358_35855

/-- A solution to the system of equations is a triple of positive integers (x, y, z) 
    satisfying the given conditions. -/
def IsSolution (x y z : ℕ+) : Prop :=
  x * y + y * z = 63 ∧ x * z + y * z = 23

/-- The theorem states that there are exactly two solutions to the system of equations. -/
theorem two_solutions : 
  ∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), 
    (∀ (x y z : ℕ+), (x, y, z) ∈ s ↔ IsSolution x y z) ∧ 
    Finset.card s = 2 := by
  sorry

end two_solutions_l358_35855


namespace square_1600_product_l358_35808

theorem square_1600_product (x : ℤ) (h : x^2 = 1600) : (x + 2) * (x - 2) = 1596 := by
  sorry

end square_1600_product_l358_35808


namespace isosceles_triangle_perimeter_l358_35810

-- Define the equation
def equation (m : ℝ) (x : ℝ) : Prop := x^2 - 2*m*x + 3*m = 0

-- Define the isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  base_positive : base > 0
  side_positive : side > 0
  triangle_inequality : 2 * side > base

-- Theorem statement
theorem isosceles_triangle_perimeter : ∃ (m : ℝ) (t : IsoscelesTriangle),
  equation m 2 ∧ 
  (equation m t.base ∨ equation m t.side) ∧
  (t.base = 2 ∨ t.side = 2) ∧
  t.base + 2 * t.side = 14 := by
  sorry

end isosceles_triangle_perimeter_l358_35810


namespace apollonius_circle_tangency_locus_l358_35848

/-- Apollonius circle associated with segment AB -/
structure ApolloniusCircle (A B : ℝ × ℝ) where
  center : ℝ × ℝ
  radius : ℝ
  divides_ratio : ℝ → ℝ → Prop

/-- Point of tangency from A to the Apollonius circle -/
def tangency_point (A B : ℝ × ℝ) (circle : ApolloniusCircle A B) : ℝ × ℝ := sorry

/-- Line perpendicular to AB at point B -/
def perpendicular_line_at_B (A B : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

theorem apollonius_circle_tangency_locus 
  (A B : ℝ × ℝ) 
  (p : ℝ) 
  (h_p : p > 1) 
  (circle : ApolloniusCircle A B) 
  (h_circle : circle.divides_ratio p 1) :
  tangency_point A B circle ∈ perpendicular_line_at_B A B :=
sorry

end apollonius_circle_tangency_locus_l358_35848


namespace season_games_l358_35870

/-- The number of teams in the league -/
def num_teams : ℕ := 20

/-- The number of times each team faces another team -/
def games_per_matchup : ℕ := 10

/-- Calculate the number of unique matchups in the league -/
def unique_matchups (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculate the total number of games in the season -/
def total_games (n : ℕ) (g : ℕ) : ℕ := unique_matchups n * g

theorem season_games :
  total_games num_teams games_per_matchup = 1900 := by sorry

end season_games_l358_35870


namespace product_evaluation_l358_35860

theorem product_evaluation (n : ℕ) (h : n = 4) : n * (n + 1) * (n + 2) = 120 := by
  sorry

end product_evaluation_l358_35860


namespace geometric_sequence_ratio_l358_35887

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  (a 1 + a 1 * q + a 1 * q^2 + a 1 * q^3 = 10 * (a 1 + a 1 * q)) →
  q = 3 := by
sorry

end geometric_sequence_ratio_l358_35887


namespace horizontal_line_slope_line_2023_slope_l358_35842

/-- The slope of a horizontal line y = k is 0 -/
theorem horizontal_line_slope (k : ℝ) : 
  let f : ℝ → ℝ := λ x => k
  (∀ x : ℝ, (f x) = k) → 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) = 0) :=
by
  sorry

/-- The slope of the line y = 2023 is 0 -/
theorem line_2023_slope : 
  let f : ℝ → ℝ := λ x => 2023
  (∀ x : ℝ, (f x) = 2023) → 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) = 0) :=
by
  sorry

end horizontal_line_slope_line_2023_slope_l358_35842


namespace infinitely_many_composites_in_sequence_l358_35857

theorem infinitely_many_composites_in_sequence :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ 
    (∃ m : ℕ, (10^(16*k+8) - 1) / 3 = 17 * m) :=
sorry

end infinitely_many_composites_in_sequence_l358_35857


namespace fractional_equation_solution_l358_35818

theorem fractional_equation_solution :
  ∃ x : ℝ, (1 / x = 2 / (x + 3)) ∧ (x = 3) := by
  sorry

end fractional_equation_solution_l358_35818


namespace doctor_lawyer_engineer_ratio_l358_35865

-- Define the number of doctors, lawyers, and engineers
variable (d l e : ℕ)

-- Define the average ages
def avg_all : ℚ := 45
def avg_doctors : ℕ := 40
def avg_lawyers : ℕ := 55
def avg_engineers : ℕ := 35

-- State the theorem
theorem doctor_lawyer_engineer_ratio :
  (avg_all : ℚ) * (d + l + e : ℚ) = avg_doctors * d + avg_lawyers * l + avg_engineers * e →
  l = d + 2 * e :=
by sorry

end doctor_lawyer_engineer_ratio_l358_35865


namespace binomial_expansion_problem_l358_35809

theorem binomial_expansion_problem (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = 3125 := by
  sorry

end binomial_expansion_problem_l358_35809


namespace measure_8_liters_possible_min_operations_is_30_l358_35883

/-- Represents the state of the two vessels --/
structure VesselState :=
  (vessel15 : ℕ)
  (vessel16 : ℕ)

/-- Represents an operation on the vessels --/
inductive Operation
  | Fill15
  | Fill16
  | Empty15
  | Empty16
  | Pour15To16
  | Pour16To15

/-- Applies an operation to a vessel state --/
def applyOperation (state : VesselState) (op : Operation) : VesselState :=
  match op with
  | Operation.Fill15 => ⟨15, state.vessel16⟩
  | Operation.Fill16 => ⟨state.vessel15, 16⟩
  | Operation.Empty15 => ⟨0, state.vessel16⟩
  | Operation.Empty16 => ⟨state.vessel15, 0⟩
  | Operation.Pour15To16 => 
      let amount := min state.vessel15 (16 - state.vessel16)
      ⟨state.vessel15 - amount, state.vessel16 + amount⟩
  | Operation.Pour16To15 => 
      let amount := min state.vessel16 (15 - state.vessel15)
      ⟨state.vessel15 + amount, state.vessel16 - amount⟩

/-- Checks if a sequence of operations results in 8 liters in either vessel --/
def achieves8Liters (ops : List Operation) : Prop :=
  let finalState := ops.foldl applyOperation ⟨0, 0⟩
  finalState.vessel15 = 8 ∨ finalState.vessel16 = 8

/-- The main theorem stating that it's possible to measure 8 liters --/
theorem measure_8_liters_possible : ∃ (ops : List Operation), achieves8Liters ops :=
  sorry

/-- The theorem stating that the minimum number of operations is 30 --/
theorem min_operations_is_30 : 
  (∃ (ops : List Operation), achieves8Liters ops ∧ ops.length = 30) ∧
  (∀ (ops : List Operation), achieves8Liters ops → ops.length ≥ 30) :=
  sorry

end measure_8_liters_possible_min_operations_is_30_l358_35883


namespace number_division_problem_l358_35892

theorem number_division_problem (x y : ℚ) : 
  (x - 5) / y = 7 → (x - 2) / 13 = 4 → y = 7 := by
  sorry

end number_division_problem_l358_35892


namespace stating_all_magpies_fly_away_l358_35888

/-- Represents the number of magpies remaining on a tree after a hunting incident -/
def magpies_remaining (initial : ℕ) (killed : ℕ) : ℕ :=
  0

/-- 
Theorem stating that regardless of the initial number of magpies and the number killed,
no magpies remain on the tree after the incident.
-/
theorem all_magpies_fly_away (initial : ℕ) (killed : ℕ) :
  magpies_remaining initial killed = 0 := by
  sorry

end stating_all_magpies_fly_away_l358_35888


namespace eccentricity_relation_l358_35862

/-- Given an ellipse with eccentricity e₁ and a hyperbola with eccentricity e₂,
    both sharing common foci F₁ and F₂, and a common point P such that
    the vectors PF₁ and PF₂ are perpendicular, prove that
    (e₁² + e₂²) / (e₁e₂)² = 2 -/
theorem eccentricity_relation (e₁ e₂ : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  e₁ > 0 ∧ e₁ < 1 →  -- Condition for ellipse eccentricity
  e₂ > 1 →  -- Condition for hyperbola eccentricity
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0 →  -- Perpendicularity condition
  (e₁^2 + e₂^2) / (e₁ * e₂)^2 = 2 := by
  sorry

end eccentricity_relation_l358_35862


namespace cubic_roots_sum_l358_35866

theorem cubic_roots_sum (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) →
  (b^3 - 2*b^2 + 3*b - 4 = 0) →
  (c^3 - 2*c^2 + 3*c - 4 = 0) →
  a ≠ b → b ≠ c → a ≠ c →
  1/(a*(b^2 + c^2 - a^2)) + 1/(b*(c^2 + a^2 - b^2)) + 1/(c*(a^2 + b^2 - c^2)) = -1/8 := by
sorry

end cubic_roots_sum_l358_35866


namespace baseball_league_games_l358_35833

/-- The number of teams in the baseball league -/
def num_teams : ℕ := 9

/-- The number of games each team plays with every other team -/
def games_per_pair : ℕ := 4

/-- The total number of games played in the season -/
def total_games : ℕ := (num_teams * (num_teams - 1) / 2) * games_per_pair

theorem baseball_league_games :
  total_games = 144 :=
sorry

end baseball_league_games_l358_35833


namespace min_value_sum_squares_and_reciprocal_cube_min_value_achievable_l358_35837

theorem min_value_sum_squares_and_reciprocal_cube (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^2 + b^2 + c^2 + 1 / (a + b + c)^3 ≥ (1/12)^(1/3) := by
  sorry

theorem min_value_achievable (ε : ℝ) (hε : ε > 0) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a^2 + b^2 + c^2 + 1 / (a + b + c)^3 < (1/12)^(1/3) + ε := by
  sorry

end min_value_sum_squares_and_reciprocal_cube_min_value_achievable_l358_35837


namespace bianca_carrots_l358_35861

def carrot_problem (initial : ℕ) (thrown_out : ℕ) (additional : ℕ) : ℕ :=
  initial - thrown_out + additional

theorem bianca_carrots : carrot_problem 23 10 47 = 60 := by
  sorry

end bianca_carrots_l358_35861


namespace right_triangle_count_l358_35874

/-- Count of right triangles with integer leg lengths a and b, hypotenuse b+2, and b < 50 -/
theorem right_triangle_count : 
  (Finset.filter (fun p : ℕ × ℕ => 
    let a := p.1
    let b := p.2
    a * a + b * b = (b + 2) * (b + 2) ∧ 
    0 < a ∧ 
    0 < b ∧ 
    b < 50
  ) (Finset.product (Finset.range 200) (Finset.range 50))).card = 7 :=
sorry

end right_triangle_count_l358_35874


namespace fuel_consumption_analysis_l358_35829

/-- Represents the fuel consumption data for a sedan --/
structure FuelData where
  initial_fuel : ℝ
  distance : ℝ
  remaining_fuel : ℝ

/-- Theorem about fuel consumption of a sedan --/
theorem fuel_consumption_analysis 
  (data : List FuelData)
  (h1 : data.length ≥ 2)
  (h2 : data[0].distance = 0 ∧ data[0].remaining_fuel = 50)
  (h3 : data[1].distance = 100 ∧ data[1].remaining_fuel = 42)
  (h4 : ∀ d ∈ data, d.initial_fuel = 50)
  (h5 : ∀ d ∈ data, d.remaining_fuel = d.initial_fuel - 0.08 * d.distance) :
  (∀ d ∈ data, d.initial_fuel = 50) ∧ 
  (∀ d ∈ data, d.remaining_fuel = -0.08 * d.distance + 50) := by
  sorry


end fuel_consumption_analysis_l358_35829


namespace complex_magnitude_fourth_power_l358_35831

theorem complex_magnitude_fourth_power : 
  Complex.abs ((4 + 2 * Real.sqrt 2 * Complex.I) ^ 4) = 576 := by sorry

end complex_magnitude_fourth_power_l358_35831


namespace total_donation_is_65_inches_l358_35807

/-- Represents the hair donation of a person -/
structure HairDonation where
  initialLength : ℕ
  keptLength : ℕ
  donatedLength : ℕ
  donation_calculation : donatedLength = initialLength - keptLength

/-- The total hair donation of the five friends -/
def totalDonation (isabella damien ella toby lisa : HairDonation) : ℕ :=
  isabella.donatedLength + damien.donatedLength + ella.donatedLength + toby.donatedLength + lisa.donatedLength

/-- Theorem stating the total hair donation is 65 inches -/
theorem total_donation_is_65_inches : 
  ∃ (isabella damien ella toby lisa : HairDonation),
    isabella.initialLength = 18 ∧ isabella.keptLength = 9 ∧
    damien.initialLength = 24 ∧ damien.keptLength = 12 ∧
    ella.initialLength = 30 ∧ ella.keptLength = 10 ∧
    toby.initialLength = 16 ∧ toby.keptLength = 0 ∧
    lisa.initialLength = 28 ∧ lisa.donatedLength = 8 ∧
    totalDonation isabella damien ella toby lisa = 65 := by
  sorry

end total_donation_is_65_inches_l358_35807


namespace camp_cedar_counselors_l358_35880

def camp_cedar (num_boys : ℕ) (girl_ratio : ℕ) (children_per_counselor : ℕ) : ℕ :=
  let num_girls := num_boys * girl_ratio
  let total_children := num_boys + num_girls
  total_children / children_per_counselor

theorem camp_cedar_counselors :
  camp_cedar 40 3 8 = 20 := by
  sorry

end camp_cedar_counselors_l358_35880


namespace sarah_age_l358_35839

/-- Given the ages of Billy, Joe, and Sarah, prove that Sarah is 10 years old -/
theorem sarah_age (B J S : ℕ) 
  (h1 : B = 2 * J)           -- Billy's age is twice Joe's age
  (h2 : B + J = 60)          -- The sum of Billy's and Joe's ages is 60 years
  (h3 : S = J - 10)          -- Sarah's age is 10 years less than Joe's age
  : S = 10 := by             -- Prove that Sarah is 10 years old
  sorry

end sarah_age_l358_35839


namespace equal_exchange_ways_l358_35824

/-- Represents the number of ways to exchange money -/
def exchange_ways (n a b : ℕ) (use_blue : Bool) : ℕ :=
  sorry

/-- The main theorem stating that the number of ways to exchange is equal for both scenarios -/
theorem equal_exchange_ways (n a b : ℕ) :
  exchange_ways n a b true = exchange_ways n a b false :=
sorry

end equal_exchange_ways_l358_35824


namespace subset_implies_m_equals_three_l358_35806

theorem subset_implies_m_equals_three (A B : Set ℝ) (m : ℝ) :
  A = {1, 3} →
  B = {1, 2, m} →
  A ⊆ B →
  m = 3 := by
sorry

end subset_implies_m_equals_three_l358_35806


namespace fruit_salad_composition_l358_35816

theorem fruit_salad_composition (total : ℕ) (red_grapes : ℕ) (green_grapes : ℕ) (raspberries : ℕ) :
  total = 102 →
  red_grapes = 67 →
  raspberries = green_grapes - 5 →
  red_grapes = 3 * green_grapes + (red_grapes - 3 * green_grapes) →
  red_grapes - 3 * green_grapes = 7 :=
by
  sorry

end fruit_salad_composition_l358_35816


namespace banana_arrangements_count_l358_35802

/-- The number of distinct arrangements of the letters in the word BANANA -/
def banana_arrangements : ℕ := 180

/-- The total number of letters in the word BANANA -/
def total_letters : ℕ := 6

/-- The number of A's in the word BANANA -/
def num_a : ℕ := 3

/-- The number of N's in the word BANANA -/
def num_n : ℕ := 2

/-- The number of B's in the word BANANA -/
def num_b : ℕ := 1

/-- Theorem stating that the number of distinct arrangements of the letters in BANANA is 180 -/
theorem banana_arrangements_count :
  banana_arrangements = Nat.factorial total_letters / (Nat.factorial num_a * Nat.factorial num_n) :=
by sorry

end banana_arrangements_count_l358_35802


namespace car_profit_percentage_l358_35858

theorem car_profit_percentage (original_price : ℝ) (h1 : original_price > 0) : 
  let discount_rate : ℝ := 0.2
  let increase_rate : ℝ := 1
  let buying_price : ℝ := original_price * (1 - discount_rate)
  let selling_price : ℝ := buying_price * (1 + increase_rate)
  let profit : ℝ := selling_price - original_price
  let profit_percentage : ℝ := (profit / original_price) * 100
  profit_percentage = 60 := by sorry

end car_profit_percentage_l358_35858


namespace suitcase_weight_problem_l358_35800

/-- Proves that given the initial ratio of books : clothes : electronics as 5 : 4 : 2, 
    and after removing 9 pounds of clothing, which doubles the ratio of books to clothes, 
    the weight of electronics is 9 pounds. -/
theorem suitcase_weight_problem (B C E : ℝ) : 
  B / C = 5 / 4 →  -- Initial ratio of books to clothes
  B / E = 5 / 2 →  -- Initial ratio of books to electronics
  B / (C - 9) = 10 / 4 →  -- New ratio after removing 9 pounds of clothes
  E = 9 := by
  sorry


end suitcase_weight_problem_l358_35800


namespace fish_per_multicolor_duck_l358_35877

theorem fish_per_multicolor_duck 
  (white_fish_ratio : ℕ) 
  (black_fish_ratio : ℕ) 
  (white_ducks : ℕ) 
  (black_ducks : ℕ) 
  (multicolor_ducks : ℕ) 
  (total_fish : ℕ) 
  (h1 : white_fish_ratio = 5)
  (h2 : black_fish_ratio = 10)
  (h3 : white_ducks = 3)
  (h4 : black_ducks = 7)
  (h5 : multicolor_ducks = 6)
  (h6 : total_fish = 157) :
  (total_fish - (white_fish_ratio * white_ducks + black_fish_ratio * black_ducks)) / multicolor_ducks = 12 := by
sorry

end fish_per_multicolor_duck_l358_35877


namespace problem_1_l358_35850

theorem problem_1 : (1) - 2 + 8 - (-30) = 36 := by
  sorry

end problem_1_l358_35850


namespace no_square_divisible_by_six_in_range_l358_35817

theorem no_square_divisible_by_six_in_range : ¬∃ y : ℕ, 
  (∃ n : ℕ, y = n^2) ∧ 
  (y % 6 = 0) ∧ 
  (50 ≤ y) ∧ 
  (y ≤ 120) := by
  sorry

end no_square_divisible_by_six_in_range_l358_35817


namespace quadratic_equal_roots_l358_35873

theorem quadratic_equal_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 0 → y = x) ↔ m = 4 := by
sorry

end quadratic_equal_roots_l358_35873


namespace max_abs_z_plus_4_l358_35849

theorem max_abs_z_plus_4 (z : ℂ) (h : Complex.abs (z + 3 * Complex.I) = 5) :
  ∃ (max_val : ℝ), max_val = 10 ∧ ∀ (w : ℂ), Complex.abs (w + 3 * Complex.I) = 5 → Complex.abs (w + 4) ≤ max_val :=
sorry

end max_abs_z_plus_4_l358_35849


namespace min_difference_l358_35854

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x - 3)
noncomputable def g (x : ℝ) : ℝ := 1 / 4 + Real.log (x / 2)

theorem min_difference (m n : ℝ) (h : f m = g n) :
  ∃ (d : ℝ), d = 1 / 2 + Real.log 2 ∧ n - m ≥ d ∧ ∃ (m₀ n₀ : ℝ), f m₀ = g n₀ ∧ n₀ - m₀ = d :=
sorry

end min_difference_l358_35854


namespace mean_median_difference_l358_35814

/-- Represents the frequency of students for each number of days missed -/
def frequency : List (ℕ × ℕ) := [(0, 4), (1, 2), (2, 5), (3, 3), (4, 2), (5, 4)]

/-- Total number of students -/
def total_students : ℕ := 20

/-- Calculates the median of the dataset -/
def median (freq : List (ℕ × ℕ)) (total : ℕ) : ℚ := sorry

/-- Calculates the mean of the dataset -/
def mean (freq : List (ℕ × ℕ)) (total : ℕ) : ℚ := sorry

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference :
  mean frequency total_students - median frequency total_students = 9 / 20 := by sorry

end mean_median_difference_l358_35814


namespace absolute_value_equality_l358_35879

theorem absolute_value_equality (x : ℝ) (h : x > 0) :
  |x + Real.sqrt ((x + 1)^2)| = 2*x + 1 := by
  sorry

end absolute_value_equality_l358_35879


namespace negation_equivalence_l358_35801

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5*x₀ + 6 > 0) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) :=
by sorry

end negation_equivalence_l358_35801


namespace function_determination_l358_35872

theorem function_determination (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → f ((x - 2) / (x + 1)) + f ((3 + x) / (1 - x)) = x) →
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → f x = (x^3 + 7*x) / (2 - 2*x^2)) := by
sorry

end function_determination_l358_35872


namespace set_of_possible_a_l358_35875

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N (a : ℝ) : Set ℝ := {x | x^2 - a*x + 3*a - 5 = 0}

theorem set_of_possible_a (a : ℝ) : M ∪ N a = M → 2 ≤ a ∧ a < 10 := by
  sorry

end set_of_possible_a_l358_35875


namespace xy_square_value_l358_35821

theorem xy_square_value (x y : ℝ) 
  (h1 : x * (x + y) = 22)
  (h2 : y * (x + y) = 78 - y) : 
  (x + y)^2 = 100 := by
sorry

end xy_square_value_l358_35821


namespace min_blocks_needed_l358_35841

/-- Represents a three-dimensional structure made of cube blocks -/
structure CubeStructure where
  blocks : ℕ → ℕ → ℕ → Bool

/-- The front view of the structure shows a 2x2 grid -/
def front_view_valid (s : CubeStructure) : Prop :=
  ∃ (i j : Fin 2), s.blocks i.val j.val 0 = true

/-- The left side view of the structure shows a 2x2 grid -/
def left_view_valid (s : CubeStructure) : Prop :=
  ∃ (i k : Fin 2), s.blocks 0 i.val k.val = true

/-- Count the number of blocks in the structure -/
def block_count (s : CubeStructure) : ℕ :=
  (Finset.range 2).sum fun i =>
    (Finset.range 2).sum fun j =>
      (Finset.range 2).sum fun k =>
        if s.blocks i j k then 1 else 0

/-- The main theorem: minimum number of blocks needed is 4 -/
theorem min_blocks_needed (s : CubeStructure) 
  (h_front : front_view_valid s) (h_left : left_view_valid s) :
  block_count s ≥ 4 := by
  sorry


end min_blocks_needed_l358_35841


namespace units_digit_product_minus_power_l358_35853

def units_digit (n : ℤ) : ℕ :=
  (n % 10).toNat

theorem units_digit_product_minus_power : units_digit (8 * 18 * 1988 - 8^4) = 6 := by
  sorry

end units_digit_product_minus_power_l358_35853


namespace intersection_distance_and_difference_l358_35882

def f (x : ℝ) := 5 * x^2 + 3 * x - 2

theorem intersection_distance_and_difference :
  ∃ (C D : ℝ × ℝ),
    (f C.1 = 4 ∧ C.2 = 4) ∧
    (f D.1 = 4 ∧ D.2 = 4) ∧
    C ≠ D ∧
    ∃ (p q : ℕ),
      p = 129 ∧
      q = 5 ∧
      (C.1 - D.1)^2 + (C.2 - D.2)^2 = (Real.sqrt p / q)^2 ∧
      p - q = 124 := by
  sorry

end intersection_distance_and_difference_l358_35882


namespace sum_distinct_prime_factors_156000_l358_35843

def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_distinct_prime_factors_156000 :
  sum_of_distinct_prime_factors 156000 = 23 := by
  sorry

end sum_distinct_prime_factors_156000_l358_35843


namespace quadratic_function_properties_l358_35812

def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by sorry

end quadratic_function_properties_l358_35812


namespace function_value_at_negative_m_l358_35894

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

-- State the theorem
theorem function_value_at_negative_m (a b m : ℝ) :
  f a b m = 6 → f a b (-m) = -4 := by
  sorry

end function_value_at_negative_m_l358_35894


namespace sum_of_squares_squared_l358_35891

theorem sum_of_squares_squared (n a b c : ℕ) (h : n = a^2 + b^2 + c^2) :
  ∃ x y z : ℕ, n^2 = x^2 + y^2 + z^2 := by
sorry

end sum_of_squares_squared_l358_35891


namespace solve_potatoes_problem_l358_35878

def potatoes_problem (total : ℕ) (gina : ℕ) : Prop :=
  let tom := 2 * gina
  let anne := tom / 3
  let remaining := total - (gina + tom + anne)
  remaining = 47

theorem solve_potatoes_problem :
  potatoes_problem 300 69 := by
  sorry

end solve_potatoes_problem_l358_35878


namespace unique_divisor_perfect_square_l358_35895

theorem unique_divisor_perfect_square (p n : ℕ) (hp : Prime p) (hp2 : p > 2) :
  ∃! d : ℕ, d ∣ (p * n^2) ∧ ∃ m : ℕ, n^2 + d = m^2 :=
sorry

end unique_divisor_perfect_square_l358_35895


namespace purple_ring_weight_l358_35896

/-- The weight of the purple ring in an experiment, given the weights of other rings and the total weight -/
theorem purple_ring_weight :
  let orange_weight : ℚ := 0.08333333333333333
  let white_weight : ℚ := 0.4166666666666667
  let total_weight : ℚ := 0.8333333333
  let purple_weight : ℚ := total_weight - orange_weight - white_weight
  purple_weight = 0.3333333333 := by
  sorry

end purple_ring_weight_l358_35896


namespace half_correct_probability_l358_35838

def num_questions : ℕ := 10
def num_correct : ℕ := 5
def probability_correct : ℚ := 1/2

theorem half_correct_probability :
  (Nat.choose num_questions num_correct) * (probability_correct ^ num_correct) * ((1 - probability_correct) ^ (num_questions - num_correct)) = 63/256 := by
  sorry

end half_correct_probability_l358_35838


namespace zoes_overall_accuracy_l358_35822

/-- Represents the problem of calculating Zoe's overall accuracy rate -/
theorem zoes_overall_accuracy 
  (x : ℝ) -- Total number of problems
  (h_positive : x > 0) -- Ensure x is positive
  (h_chloe_indep : ℝ) -- Chloe's independent accuracy rate
  (h_chloe_indep_val : h_chloe_indep = 0.8) -- Chloe's independent accuracy is 80%
  (h_overall : ℝ) -- Overall accuracy rate for all problems
  (h_overall_val : h_overall = 0.88) -- Overall accuracy is 88%
  (h_zoe_indep : ℝ) -- Zoe's independent accuracy rate
  (h_zoe_indep_val : h_zoe_indep = 0.9) -- Zoe's independent accuracy is 90%
  : ∃ (y : ℝ), -- y represents the accuracy rate of problems solved together
    (0.5 * x * h_chloe_indep + 0.5 * x * y) / x = h_overall ∧ 
    (0.5 * x * h_zoe_indep + 0.5 * x * y) / x = 0.93 := by
  sorry

end zoes_overall_accuracy_l358_35822


namespace sum_max_min_cubes_l358_35885

/-- Represents a view of the geometric figure -/
structure View where
  (front : Set (ℕ × ℕ))
  (left : Set (ℕ × ℕ))
  (top : Set (ℕ × ℕ))

/-- Counts the number of cubes in a valid configuration -/
def count_cubes (v : View) : ℕ → Bool := sorry

/-- The maximum number of cubes that can form the figure -/
def max_cubes (v : View) : ℕ := sorry

/-- The minimum number of cubes that can form the figure -/
def min_cubes (v : View) : ℕ := sorry

/-- The theorem stating that the sum of max and min cubes is 20 -/
theorem sum_max_min_cubes (v : View) : max_cubes v + min_cubes v = 20 := by sorry

end sum_max_min_cubes_l358_35885


namespace taxi_fare_for_80_miles_l358_35881

/-- Represents the fare structure of a taxi company -/
structure TaxiFare where
  fixedFare : ℝ
  costPerMile : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.fixedFare + tf.costPerMile * distance

theorem taxi_fare_for_80_miles :
  ∃ (tf : TaxiFare),
    tf.fixedFare = 15 ∧
    totalFare tf 60 = 135 ∧
    totalFare tf 80 = 175 := by
  sorry

end taxi_fare_for_80_miles_l358_35881


namespace max_books_borrowed_l358_35820

theorem max_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) 
  (two_books : Nat) (min_three_books : Nat) (avg_books : Nat) :
  total_students = 20 →
  zero_books = 2 →
  one_book = 10 →
  two_books = 5 →
  min_three_books = total_students - zero_books - one_book - two_books →
  avg_books = 2 →
  ∃ (max_books : Nat), 
    max_books = (total_students * avg_books) - 
      (one_book * 1 + two_books * 2 + (min_three_books - 1) * 3) ∧
    max_books ≤ 14 :=
by sorry

end max_books_borrowed_l358_35820


namespace unit_conversions_l358_35864

-- Define conversion factors
def meters_to_decimeters : ℝ → ℝ := (· * 10)
def minutes_to_seconds : ℝ → ℝ := (· * 60)

-- Theorem to prove the conversions
theorem unit_conversions :
  (meters_to_decimeters 2 = 20) ∧
  (minutes_to_seconds 2 = 120) ∧
  (minutes_to_seconds (600 / 60) = 10) := by
  sorry

end unit_conversions_l358_35864


namespace phase_shift_of_sine_function_l358_35823

/-- The phase shift of the function y = 2 sin(2x + π/3) is -π/6 -/
theorem phase_shift_of_sine_function :
  let f : ℝ → ℝ := λ x => 2 * Real.sin (2 * x + π / 3)
  ∃ (A B C D : ℝ), A ≠ 0 ∧ B ≠ 0 ∧
    (∀ x, f x = A * Real.sin (B * (x - C)) + D) ∧
    C = -π / 6 :=
by sorry

end phase_shift_of_sine_function_l358_35823


namespace additional_as_needed_l358_35827

/-- Given initial grades and A's, and subsequent increases in A proportion,
    calculate additional A's needed for a further increase. -/
theorem additional_as_needed
  (n k : ℕ)  -- Initial number of grades and A's
  (h1 : (k + 1 : ℚ) / (n + 1) - k / n = 15 / 100)  -- First increase
  (h2 : (k + 2 : ℚ) / (n + 2) - (k + 1) / (n + 1) = 1 / 10)  -- Second increase
  (h3 : (k + 2 : ℚ) / (n + 2) = 2 / 3)  -- Current proportion
  : ∃ m : ℕ, (k + 2 + m : ℚ) / (n + 2 + m) = 7 / 10 ∧ m = 4 := by
  sorry


end additional_as_needed_l358_35827


namespace correct_propositions_l358_35834

theorem correct_propositions (a b c d : ℝ) : 
  (∀ (a b c d : ℝ), a > b → c > d → a + c > b + d) ∧ 
  (∃ (a b c d : ℝ), a > b ∧ c > d ∧ ¬(a - c > b - d)) ∧
  (∃ (a b c d : ℝ), a > b ∧ c > d ∧ ¬(a * c > b * d)) ∧
  (∀ (a b c : ℝ), a > b → c > 0 → a * c > b * c) := by
  sorry

end correct_propositions_l358_35834


namespace corrected_mean_calculation_l358_35815

def correct_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n : ℚ) * original_mean - incorrect_value + correct_value

theorem corrected_mean_calculation (n : ℕ) (original_mean incorrect_value correct_value : ℚ) :
  n = 50 →
  original_mean = 36 →
  incorrect_value = 23 →
  correct_value = 45 →
  (correct_mean n original_mean incorrect_value correct_value) / n = 36.44 :=
by
  sorry

#eval (correct_mean 50 36 23 45) / 50

end corrected_mean_calculation_l358_35815


namespace garden_ratio_l358_35884

def garden_problem (table_price bench_price : ℕ) : Prop :=
  table_price + bench_price = 450 ∧
  ∃ k : ℕ, table_price = k * bench_price ∧
  bench_price = 150

theorem garden_ratio :
  ∀ table_price bench_price : ℕ,
  garden_problem table_price bench_price →
  table_price / bench_price = 2 :=
by
  sorry

end garden_ratio_l358_35884


namespace problem_statement_l358_35899

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) : 
  (1 / a^2 + 1 / b^2 ≥ 1 / 2) ∧ 
  (∃ (m : ℝ), m = 2 * Real.sqrt 6 + 3 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = x * y → |2*x - 1| + |3*y - 1| ≥ m) := by
  sorry

end problem_statement_l358_35899


namespace sum_of_bases_equals_1135_l358_35830

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (13 ^ i)) 0

/-- The value of digit C in base 13 -/
def C : Nat := 12

/-- The theorem to prove -/
theorem sum_of_bases_equals_1135 :
  base9ToBase10 [1, 6, 3] + base13ToBase10 [5, C, 4] = 1135 := by
  sorry

end sum_of_bases_equals_1135_l358_35830


namespace cricket_team_average_age_l358_35897

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (A : ℝ),
    team_size = 11 →
    captain_age = 27 →
    wicket_keeper_age_diff = 3 →
    (team_size : ℝ) * A = 
      (captain_age : ℝ) + (captain_age + wicket_keeper_age_diff : ℝ) + 
      ((team_size - 2 : ℝ) * (A - 1)) →
    A = 24 := by
  sorry

end cricket_team_average_age_l358_35897


namespace decagon_diagonal_intersections_l358_35876

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def intersection_points : ℕ := Nat.choose n 4

theorem decagon_diagonal_intersections :
  intersection_points = 210 :=
sorry

end decagon_diagonal_intersections_l358_35876


namespace super_k_conference_l358_35890

theorem super_k_conference (n : ℕ) : 
  (n * (n - 1)) / 2 = 45 → n = 10 := by
  sorry

end super_k_conference_l358_35890


namespace helen_hand_wash_frequency_l358_35856

/-- The frequency of Helen's hand washing her pillowcases in weeks -/
def hand_wash_frequency (time_per_wash : ℕ) (total_time_per_year : ℕ) (weeks_per_year : ℕ) : ℕ :=
  weeks_per_year / (total_time_per_year / time_per_wash)

/-- Theorem stating that Helen hand washes her pillowcases every 4 weeks -/
theorem helen_hand_wash_frequency :
  hand_wash_frequency 30 390 52 = 4 := by
  sorry

end helen_hand_wash_frequency_l358_35856


namespace exists_abs_neq_self_l358_35825

theorem exists_abs_neq_self : ∃ a : ℝ, |a| ≠ a := by
  sorry

end exists_abs_neq_self_l358_35825


namespace average_book_width_l358_35804

def book_widths : List ℝ := [6, 50, 1, 35, 3, 5, 75, 20]

theorem average_book_width :
  let total_width := book_widths.sum
  let num_books := book_widths.length
  total_width / num_books = 24.375 := by
sorry

end average_book_width_l358_35804


namespace domain_log_range_exp_intersection_empty_l358_35835

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x < 0}
def B : Set ℝ := {y : ℝ | y > 0}

-- State the theorem
theorem domain_log_range_exp_intersection_empty : A ∩ B = ∅ := by
  sorry

end domain_log_range_exp_intersection_empty_l358_35835


namespace net_rate_of_pay_l358_35828

/-- Calculates the net rate of pay for a driver given travel conditions and expenses. -/
theorem net_rate_of_pay
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (gasoline_cost : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_rate = 0.60)
  (h5 : gasoline_cost = 2.50) :
  (pay_rate * speed * travel_time - (speed * travel_time / fuel_efficiency) * gasoline_cost) / travel_time = 25 := by
  sorry


end net_rate_of_pay_l358_35828


namespace square_land_side_length_l358_35898

theorem square_land_side_length (area : ℝ) (side : ℝ) : 
  area = 1024 → side * side = area → side = 32 := by
  sorry

end square_land_side_length_l358_35898


namespace solution_to_equation_l358_35803

theorem solution_to_equation : ∃ x : ℝ, 0.2 * x + (0.6 * 0.8) = 0.56 ∧ x = 0.4 := by
  sorry

end solution_to_equation_l358_35803


namespace sum_of_digits_in_19_minutes_l358_35845

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def time_to_minutes (hours minutes : Nat) : Nat :=
  (hours % 12) * 60 + minutes

def minutes_to_time (total_minutes : Nat) : (Nat × Nat) :=
  ((total_minutes / 60) % 12, total_minutes % 60)

theorem sum_of_digits_in_19_minutes 
  (current_hours current_minutes : Nat) 
  (h_valid_time : current_hours < 12 ∧ current_minutes < 60) 
  (h_sum_condition : 
    let (prev_hours, prev_minutes) := minutes_to_time (time_to_minutes current_hours current_minutes - 19)
    sum_of_digits prev_hours + sum_of_digits prev_minutes = 
      sum_of_digits current_hours + sum_of_digits current_minutes - 2) :
  let (future_hours, future_minutes) := minutes_to_time (time_to_minutes current_hours current_minutes + 19)
  sum_of_digits future_hours + sum_of_digits future_minutes = 11 := by
sorry

end sum_of_digits_in_19_minutes_l358_35845


namespace cube_volume_l358_35847

theorem cube_volume (a : ℤ) : 
  (∃ (x y : ℤ), x = a + 2 ∧ y = a - 2 ∧ 
    x * a * y = a^3 - 16 ∧
    2 * (x + a) = 2 * (a + a) + 4) →
  a^3 = 216 := by
sorry

end cube_volume_l358_35847


namespace value_of_T_l358_35805

theorem value_of_T : ∃ T : ℝ, (1/3 : ℝ) * (1/6 : ℝ) * T = (1/4 : ℝ) * (1/8 : ℝ) * 120 ∧ T = 67.5 := by
  sorry

end value_of_T_l358_35805


namespace distribution_ratio_l358_35844

def num_balls : ℕ := 20
def num_bins : ℕ := 5

def distribution_A : List ℕ := [3, 5, 4, 4, 4]
def distribution_B : List ℕ := [4, 4, 4, 4, 4]

def count_distributions (n : ℕ) (k : ℕ) (dist : List ℕ) : ℕ :=
  sorry

theorem distribution_ratio :
  (count_distributions num_balls num_bins distribution_A) /
  (count_distributions num_balls num_bins distribution_B) = 4 :=
by sorry

end distribution_ratio_l358_35844


namespace equation_solution_set_l358_35893

theorem equation_solution_set : ∃ (S : Set ℝ), 
  S = {x : ℝ | (1 / (x^2 + 8*x - 12) + 1 / (x^2 + 5*x - 12) + 1 / (x^2 - 10*x - 12) = 0)} ∧ 
  S = {Real.sqrt 12, -Real.sqrt 12, 4, 3} := by
sorry

end equation_solution_set_l358_35893


namespace systematic_sampling_l358_35868

theorem systematic_sampling (total_students : Nat) (sample_size : Nat) (part_size : Nat) (first_drawn : Nat) :
  total_students = 1000 →
  sample_size = 50 →
  part_size = 20 →
  first_drawn = 15 →
  (third_drawn : Nat) = 55 :=
sorry

end systematic_sampling_l358_35868


namespace vector_problem_l358_35813

theorem vector_problem (α β : ℝ) (a b c : ℝ × ℝ) :
  a = (Real.cos α, Real.sin α) →
  b = (Real.cos β, Real.sin β) →
  c = (1, 2) →
  a.1 * b.1 + a.2 * b.2 = Real.sqrt 2 / 2 →
  ∃ (k : ℝ), a = k • c →
  0 < β →
  β < α →
  α < Real.pi / 2 →
  Real.cos (α - β) = Real.sqrt 2 / 2 ∧ Real.cos β = 3 * Real.sqrt 10 / 10 := by
  sorry

end vector_problem_l358_35813


namespace least_multiple_squared_l358_35846

theorem least_multiple_squared (X Y : ℕ) : 
  (∃ Y, 3456^2 * X = 6789^2 * Y) ∧ 
  (∀ Z, Z < X → ¬∃ W, 3456^2 * Z = 6789^2 * W) →
  X = 290521 := by
sorry

end least_multiple_squared_l358_35846


namespace smallest_x_for_perfect_cube_l358_35869

theorem smallest_x_for_perfect_cube (x : ℕ) : x = 36 ↔ 
  (x > 0 ∧ ∃ y : ℕ, 1152 * x = y^3 ∧ ∀ z < x, z > 0 → ¬∃ w : ℕ, 1152 * z = w^3) :=
sorry

end smallest_x_for_perfect_cube_l358_35869


namespace cos_75_degrees_l358_35863

theorem cos_75_degrees :
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_degrees_l358_35863


namespace rectangle_thirteen_squares_l358_35889

/-- A rectangle can be divided into 13 equal squares if and only if its side length ratio is 13:1 -/
theorem rectangle_thirteen_squares (a b : ℕ) (h : a > 0 ∧ b > 0) :
  (∃ (s : ℕ), s > 0 ∧ a * b = 13 * s * s) ↔ (a = 13 * b ∨ b = 13 * a) :=
sorry

end rectangle_thirteen_squares_l358_35889


namespace correct_calculation_l358_35871

theorem correct_calculation (x : ℤ) (h : x + 392 = 541) : x + 293 = 442 := by
  sorry

end correct_calculation_l358_35871


namespace sheila_saves_for_four_years_l358_35852

/-- Calculates the number of years Sheila plans to save. -/
def sheilas_savings_years (initial_savings : ℕ) (monthly_savings : ℕ) (family_addition : ℕ) (final_amount : ℕ) : ℕ :=
  ((final_amount - family_addition - initial_savings) / monthly_savings) / 12

/-- Theorem stating that Sheila plans to save for 4 years. -/
theorem sheila_saves_for_four_years :
  sheilas_savings_years 3000 276 7000 23248 = 4 := by
  sorry

end sheila_saves_for_four_years_l358_35852


namespace three_unit_fractions_sum_to_one_l358_35851

theorem three_unit_fractions_sum_to_one :
  ∀ a b c : ℕ+,
    a ≠ b → b ≠ c → a ≠ c →
    (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ = 1 →
    ({a, b, c} : Set ℕ+) = {2, 3, 6} := by
  sorry

end three_unit_fractions_sum_to_one_l358_35851


namespace range_of_sum_l358_35867

theorem range_of_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 1) 
  (square_sum_condition : a^2 + b^2 + c^2 = 1) : 
  0 ≤ a + b ∧ a + b ≤ 4/3 := by
sorry

end range_of_sum_l358_35867


namespace average_stickers_per_pack_l358_35836

def sticker_counts : List ℕ := [5, 7, 9, 9, 11, 15, 15, 17, 19, 21]

def total_stickers : ℕ := sticker_counts.sum

def num_packs : ℕ := sticker_counts.length

theorem average_stickers_per_pack :
  (total_stickers : ℚ) / num_packs = 12.8 := by
  sorry

end average_stickers_per_pack_l358_35836


namespace two_valid_colorings_l358_35859

/-- Represents the three possible colors for a hexagon -/
inductive Color
| Red
| Yellow
| Green

/-- Represents a column of hexagons -/
structure Column where
  hexagons : List Color
  size : Nat
  size_eq : hexagons.length = size

/-- Represents the entire figure of hexagons -/
structure HexagonFigure where
  column1 : Column
  column2 : Column
  column3 : Column
  column4 : Column
  col1_size : column1.size = 3
  col2_size : column2.size = 4
  col3_size : column3.size = 4
  col4_size : column4.size = 3
  bottom_red : column1.hexagons.head? = some Color.Red

/-- Predicate to check if two colors are different -/
def differentColors (c1 c2 : Color) : Prop :=
  c1 ≠ c2

/-- Predicate to check if a coloring is valid (no adjacent hexagons have the same color) -/
def validColoring (figure : HexagonFigure) : Prop :=
  -- Add conditions to check adjacent hexagons in each column and between columns
  sorry

/-- The number of valid colorings for the hexagon figure -/
def numValidColorings : Nat :=
  -- Count the number of valid colorings
  sorry

/-- Theorem stating that there are exactly 2 valid colorings -/
theorem two_valid_colorings : numValidColorings = 2 := by
  sorry

end two_valid_colorings_l358_35859


namespace vector_sum_equality_l358_35826

variable (V : Type*) [AddCommGroup V]

theorem vector_sum_equality (a : V) : a + 2 • a = 3 • a := by sorry

end vector_sum_equality_l358_35826


namespace largest_c_for_one_in_range_l358_35886

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

-- State the theorem
theorem largest_c_for_one_in_range : 
  (∃ (c : ℝ), ∀ (d : ℝ), (∃ (x : ℝ), f d x = 1) → d ≤ c) ∧ 
  (∃ (x : ℝ), f 10 x = 1) :=
by sorry

end largest_c_for_one_in_range_l358_35886


namespace alice_purchases_cost_l358_35819

/-- The exchange rate from British Pounds to USD -/
def gbp_to_usd : ℝ := 1.25

/-- The exchange rate from Euros to USD -/
def eur_to_usd : ℝ := 1.10

/-- The cost of the book in British Pounds -/
def book_cost_gbp : ℝ := 15

/-- The cost of the souvenir in Euros -/
def souvenir_cost_eur : ℝ := 20

/-- The total cost of Alice's purchases in USD -/
def total_cost_usd : ℝ := book_cost_gbp * gbp_to_usd + souvenir_cost_eur * eur_to_usd

theorem alice_purchases_cost : total_cost_usd = 40.75 := by
  sorry

end alice_purchases_cost_l358_35819
