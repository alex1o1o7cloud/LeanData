import Mathlib

namespace math_team_combinations_l1398_139897

theorem math_team_combinations (girls : ℕ) (boys : ℕ) : 
  girls = 3 → boys = 5 → (girls.choose 2) * (boys.choose 2) = 30 := by
  sorry

end math_team_combinations_l1398_139897


namespace car_distance_proof_l1398_139817

/-- Proves that the distance covered by a car is 144 km given specific conditions -/
theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) (time_factor : ℝ) :
  initial_time = 6 →
  speed = 16 →
  time_factor = 3/2 →
  initial_time * time_factor * speed = 144 := by
  sorry

end car_distance_proof_l1398_139817


namespace jim_investment_approx_l1398_139849

/-- Represents the investment ratios of John, James, Jim, and Jordan respectively -/
def investment_ratio : Fin 4 → ℕ
  | 0 => 8   -- John
  | 1 => 11  -- James
  | 2 => 15  -- Jim
  | 3 => 19  -- Jordan

/-- The total investment amount in dollars -/
def total_investment : ℚ := 127000

/-- Jim's index in the investment ratio -/
def jim_index : Fin 4 := 2

/-- Calculate Jim's investment amount -/
def jim_investment : ℚ :=
  (total_investment * investment_ratio jim_index) /
  (Finset.sum Finset.univ investment_ratio)

theorem jim_investment_approx :
  ∃ ε > 0, |jim_investment - 35943.40| < ε := by sorry

end jim_investment_approx_l1398_139849


namespace complement_A_union_B_l1398_139814

def U : Set ℕ := {n | n > 0 ∧ n < 9}
def A : Set ℕ := {n ∈ U | n % 2 = 1}
def B : Set ℕ := {n ∈ U | n % 3 = 0}

theorem complement_A_union_B : (U \ (A ∪ B)) = {2, 4, 8} := by sorry

end complement_A_union_B_l1398_139814


namespace certain_fraction_proof_l1398_139832

theorem certain_fraction_proof :
  ∀ (x y : ℚ),
  (3 : ℚ) / 5 / ((6 : ℚ) / 7) = (7 : ℚ) / 15 / (x / y) →
  x / y = (2 : ℚ) / 3 := by
  sorry

end certain_fraction_proof_l1398_139832


namespace rice_weight_calculation_l1398_139802

theorem rice_weight_calculation (total : ℚ) : 
  (total * (1 - 3/10) * (1 - 2/5) = 210) → total = 500 := by
  sorry

end rice_weight_calculation_l1398_139802


namespace prob_same_gender_is_one_third_l1398_139854

/-- Represents the gender of a student -/
inductive Gender
| Male
| Female

/-- Represents a group of students -/
structure StudentGroup where
  males : Finset Gender
  females : Finset Gender
  male_count : males.card = 2
  female_count : females.card = 2

/-- Represents a selection of two students -/
structure Selection where
  first : Gender
  second : Gender

/-- The probability of selecting two students of the same gender -/
def prob_same_gender (group : StudentGroup) : ℚ :=
  (2 : ℚ) / 6

theorem prob_same_gender_is_one_third (group : StudentGroup) :
  prob_same_gender group = (1 : ℚ) / 3 := by
  sorry

end prob_same_gender_is_one_third_l1398_139854


namespace interview_panel_seating_l1398_139839

/-- Represents the number of players in each team --/
structure TeamSizes :=
  (team1 : Nat)
  (team2 : Nat)
  (team3 : Nat)

/-- Calculates the number of seating arrangements for players from different teams
    where teammates must sit together --/
def seatingArrangements (sizes : TeamSizes) : Nat :=
  Nat.factorial 3 * Nat.factorial sizes.team1 * Nat.factorial sizes.team2 * Nat.factorial sizes.team3

/-- Theorem stating that for the given team sizes, there are 1728 seating arrangements --/
theorem interview_panel_seating :
  seatingArrangements ⟨4, 3, 2⟩ = 1728 := by
  sorry

#eval seatingArrangements ⟨4, 3, 2⟩

end interview_panel_seating_l1398_139839


namespace geometric_sequence_sum_l1398_139867

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 20 + a 21 = 10) →
  (a 22 + a 23 = 20) →
  (a 24 + a 25 = 40) :=
by
  sorry

end geometric_sequence_sum_l1398_139867


namespace value_of_x_l1398_139884

theorem value_of_x (x y z : ℚ) : 
  x = (1 / 3) * y → 
  y = (1 / 4) * z → 
  z = 96 → 
  x = 8 := by
sorry

end value_of_x_l1398_139884


namespace negation_equivalence_l1398_139892

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) := by
  sorry

end negation_equivalence_l1398_139892


namespace expression_evaluation_l1398_139870

theorem expression_evaluation :
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 := by
  sorry

end expression_evaluation_l1398_139870


namespace complex_number_in_second_quadrant_l1398_139805

theorem complex_number_in_second_quadrant : 
  let z : ℂ := (Complex.I / (1 + Complex.I)) + (1 + Complex.I * Real.sqrt 3) ^ 2
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end complex_number_in_second_quadrant_l1398_139805


namespace no_right_prism_with_diagonals_4_5_7_l1398_139801

theorem no_right_prism_with_diagonals_4_5_7 :
  ¬∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x^2 + y^2 = 16 ∧ x^2 + z^2 = 25 ∧ y^2 + z^2 = 49 := by
  sorry

end no_right_prism_with_diagonals_4_5_7_l1398_139801


namespace combined_degrees_sum_l1398_139890

/-- The combined number of degrees for Summer and Jolly -/
def combined_degrees (summer_degrees : ℕ) (difference : ℕ) : ℕ :=
  summer_degrees + (summer_degrees - difference)

/-- Theorem stating that given Summer has 150 degrees and 5 more degrees than Jolly,
    the combined number of degrees for Summer and Jolly is 295 -/
theorem combined_degrees_sum (summer_degrees : ℕ) (difference : ℕ)
  (h1 : summer_degrees = 150)
  (h2 : difference = 5) :
  combined_degrees summer_degrees difference = 295 := by
sorry

end combined_degrees_sum_l1398_139890


namespace complex_equation_solution_l1398_139880

theorem complex_equation_solution (a b : ℝ) (z : ℂ) :
  z = a + 4*Complex.I ∧ z / (z + b) = 4*Complex.I → b = 17 := by
  sorry

end complex_equation_solution_l1398_139880


namespace cookie_ratio_l1398_139820

/-- Proves that the ratio of Glenn's cookies to Kenny's cookies is 4:1 given the problem conditions --/
theorem cookie_ratio (kenny : ℕ) (glenn : ℕ) (chris : ℕ) : 
  chris = kenny / 2 → 
  glenn = 24 → 
  chris + kenny + glenn = 33 → 
  glenn / kenny = 4 :=
by
  sorry

end cookie_ratio_l1398_139820


namespace arithmetic_evaluation_l1398_139882

theorem arithmetic_evaluation : 8 * (6 - 4) + 2 = 18 := by
  sorry

end arithmetic_evaluation_l1398_139882


namespace trigonometric_equation_solution_l1398_139812

open Real

theorem trigonometric_equation_solution (x : ℝ) :
  sin x + sin (2*x) + sin (3*x) = 1 + cos x + cos (2*x) ↔
  (∃ k : ℤ, x = π/2 + k * π) ∨
  (∃ k : ℤ, x = 2*π/3 + k * 2*π) ∨
  (∃ k : ℤ, x = 4*π/3 + k * 2*π) ∨
  (∃ k : ℤ, x = π/6 + k * 2*π) ∨
  (∃ k : ℤ, x = 5*π/6 + k * 2*π) :=
by sorry

end trigonometric_equation_solution_l1398_139812


namespace factorization_proof_l1398_139813

/-- Prove the factorization of two polynomial expressions -/
theorem factorization_proof (x y : ℝ) : 
  (2 * x^2 * y - 4 * x * y + 2 * y = 2 * y * (x - 1)^2) ∧ 
  (x^4 - 9 * x^2 = x^2 * (x + 3) * (x - 3)) := by
  sorry

end factorization_proof_l1398_139813


namespace inequality_solution_set_l1398_139831

/-- If the solution set of (1-m^2)x^2-(1+m)x-1<0 with respect to x is ℝ,
    then m satisfies m ≤ -1 or m > 5/3 -/
theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (1 - m^2) * x^2 - (1 + m) * x - 1 < 0) →
  (m ≤ -1 ∨ m > 5/3) :=
by sorry

end inequality_solution_set_l1398_139831


namespace nth_monomial_form_l1398_139804

/-- A sequence of monomials is defined as follows:
    1st term: a
    2nd term: 3a²
    3rd term: 5a³
    4th term: 7a⁴
    5th term: 9a⁵
    ...
    This function represents the coefficient of the nth term in this sequence. -/
def monomial_coefficient (n : ℕ) : ℕ := 2 * n - 1

/-- This function represents the exponent of 'a' in the nth term of the sequence. -/
def monomial_exponent (n : ℕ) : ℕ := n

/-- This theorem states that the nth term of the sequence is (2n - 1)aⁿ -/
theorem nth_monomial_form (n : ℕ) (a : ℝ) :
  monomial_coefficient n * a ^ monomial_exponent n = (2 * n - 1) * a ^ n :=
sorry

end nth_monomial_form_l1398_139804


namespace probability_three_same_color_l1398_139876

/-- Represents a person in the block placement scenario -/
structure Person where
  name : String
  blocks : Fin 5 → Color

/-- Represents the colors of the blocks -/
inductive Color
  | Red
  | Blue
  | Yellow
  | White
  | Green

/-- Represents the result of a single trial -/
structure Trial where
  placements : Person → Fin 6 → Option (Fin 5)

/-- The probability of a specific event occurring in the trial -/
def probability (event : Trial → Prop) : ℚ :=
  sorry

/-- Checks if a trial results in at least one box with 3 blocks of the same color -/
def has_three_same_color (t : Trial) : Prop :=
  sorry

/-- The main theorem stating the probability of the event -/
theorem probability_three_same_color 
  (ang ben jasmin : Person)
  (h1 : ang.name = "Ang" ∧ ben.name = "Ben" ∧ jasmin.name = "Jasmin")
  (h2 : ∀ p : Person, p = ang ∨ p = ben ∨ p = jasmin → 
        ∀ i : Fin 5, ∃! c : Color, p.blocks i = c) :
  probability has_three_same_color = 5 / 216 :=
sorry

end probability_three_same_color_l1398_139876


namespace hares_per_rabbit_l1398_139837

theorem hares_per_rabbit (dog : Nat) (cats : Nat) (rabbits_per_cat : Nat) (total_animals : Nat) :
  dog = 1 →
  cats = 4 →
  rabbits_per_cat = 2 →
  total_animals = 37 →
  ∃ hares_per_rabbit : Nat, 
    total_animals = dog + cats + (cats * rabbits_per_cat) + (cats * rabbits_per_cat * hares_per_rabbit) ∧
    hares_per_rabbit = 3 := by
  sorry

end hares_per_rabbit_l1398_139837


namespace complex_sum_of_powers_l1398_139887

theorem complex_sum_of_powers : 
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2
  z₁^12 + z₂^12 = 2 := by sorry

end complex_sum_of_powers_l1398_139887


namespace imaginary_part_of_complex_division_l1398_139868

theorem imaginary_part_of_complex_division (i : ℂ) : 
  i * i = -1 → Complex.im ((4 - 3 * i) / i) = -4 := by
  sorry

end imaginary_part_of_complex_division_l1398_139868


namespace trig_identity_l1398_139835

theorem trig_identity (x : Real) : 
  (Real.cos x)^4 + (Real.sin x)^4 + 3*(Real.sin x)^2*(Real.cos x)^2 = 
  (Real.cos x)^6 + (Real.sin x)^6 + 4*(Real.sin x)^2*(Real.cos x)^2 := by
  sorry

end trig_identity_l1398_139835


namespace equation_solution_iff_m_equals_p_l1398_139889

theorem equation_solution_iff_m_equals_p (p m : ℕ) (hp : Prime p) (hm : m ≥ 2) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (x, y) ≠ (1, 1) ∧
    (x^p + y^p) / 2 = ((x + y) / 2)^m) ↔ m = p :=
by sorry

end equation_solution_iff_m_equals_p_l1398_139889


namespace cos_negative_1830_degrees_l1398_139842

theorem cos_negative_1830_degrees : Real.cos ((-1830 : ℝ) * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_negative_1830_degrees_l1398_139842


namespace quadratic_point_distance_l1398_139878

/-- Given a quadratic function f(x) = ax² - 2ax + b where a > 0,
    and two points (x₁, y₁) and (x₂, y₂) on its graph where y₁ > y₂,
    prove that |x₁ - 1| > |x₂ - 1| -/
theorem quadratic_point_distance (a b x₁ y₁ x₂ y₂ : ℝ) 
  (ha : a > 0)
  (hf₁ : y₁ = a * x₁^2 - 2 * a * x₁ + b)
  (hf₂ : y₂ = a * x₂^2 - 2 * a * x₂ + b)
  (hy : y₁ > y₂) :
  |x₁ - 1| > |x₂ - 1| := by
  sorry

end quadratic_point_distance_l1398_139878


namespace trig_identity_l1398_139848

theorem trig_identity (α : ℝ) : 
  (Real.sin (α - π/6))^2 + (Real.sin (α + π/6))^2 - (Real.sin α)^2 = 1/2 := by
  sorry

end trig_identity_l1398_139848


namespace perfect_square_in_base_n_l1398_139873

theorem perfect_square_in_base_n (n : ℕ) (hn : n ≥ 2) :
  ∃ m : ℕ, m^2 = n^4 + n^3 + n^2 + n + 1 ↔ n = 3 := by sorry

end perfect_square_in_base_n_l1398_139873


namespace arithmetic_mean_of_fractions_l1398_139864

theorem arithmetic_mean_of_fractions :
  let a := 5 / 8
  let b := 9 / 16
  let c := 11 / 16
  a = (b + c) / 2 := by
  sorry

end arithmetic_mean_of_fractions_l1398_139864


namespace line_parabola_intersection_l1398_139857

-- Define the line l passing through (-2, 1) with slope k
def line (k : ℝ) (x y : ℝ) : Prop :=
  y - 1 = k * (x + 2)

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the condition for the line to intersect the parabola at only one point
def unique_intersection (k : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, line k p.1 p.2 ∧ parabola p.1 p.2

-- Theorem statement
theorem line_parabola_intersection (k : ℝ) :
  unique_intersection k → k = 0 ∨ k = -1 ∨ k = 1/2 := by
  sorry

end line_parabola_intersection_l1398_139857


namespace omitted_angle_measure_l1398_139885

theorem omitted_angle_measure (n : ℕ) (sum_without_one : ℝ) : 
  n ≥ 3 → 
  sum_without_one = 1958 → 
  (n - 2) * 180 - sum_without_one = 22 :=
by sorry

end omitted_angle_measure_l1398_139885


namespace halfway_point_fractions_l1398_139833

theorem halfway_point_fractions (a b : ℚ) (ha : a = 1/8) (hb : b = 1/3) :
  (a + b) / 2 = 11/48 := by
  sorry

end halfway_point_fractions_l1398_139833


namespace cut_cube_edges_l1398_139894

/-- Represents a cube with cut corners -/
structure CutCube where
  originalEdges : Nat
  vertices : Nat
  cutsPerVertex : Nat
  newFacesPerCut : Nat
  newEdgesPerFace : Nat

/-- The number of edges in a cube with cut corners -/
def edgesAfterCut (c : CutCube) : Nat :=
  c.originalEdges + c.vertices * c.cutsPerVertex * c.newEdgesPerFace / 2

/-- Theorem stating that a cube with cut corners has 36 edges -/
theorem cut_cube_edges :
  ∀ c : CutCube,
  c.originalEdges = 12 ∧
  c.vertices = 8 ∧
  c.cutsPerVertex = 1 ∧
  c.newFacesPerCut = 1 ∧
  c.newEdgesPerFace = 4 →
  edgesAfterCut c = 36 := by
  sorry

#check cut_cube_edges

end cut_cube_edges_l1398_139894


namespace binomial_odd_iff_power_of_two_minus_one_l1398_139850

theorem binomial_odd_iff_power_of_two_minus_one (n : ℕ) :
  (∀ k : ℕ, k ≤ n → Odd (Nat.choose n k)) ↔
  ∃ m : ℕ, m ≥ 1 ∧ n = 2^m - 1 := by
  sorry

end binomial_odd_iff_power_of_two_minus_one_l1398_139850


namespace existence_of_n_for_prime_divisibility_l1398_139869

theorem existence_of_n_for_prime_divisibility (p : ℕ) (hp : Nat.Prime p) :
  ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) := by
  sorry

end existence_of_n_for_prime_divisibility_l1398_139869


namespace hyperbola_equation_l1398_139898

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and right focus at (c, 0),
    if a circle with radius 4 centered at the right focus passes through
    the origin and the point (a, b) on the asymptote, then a² = 4 and b² = 12 -/
theorem hyperbola_equation (a b c : ℝ) (h1 : c > 0) (h2 : a > 0) (h3 : b > 0) :
  (c = 4) →
  ((a - c)^2 + b^2 = 16) →
  (a^2 + b^2 = c^2) →
  (a^2 = 4 ∧ b^2 = 12) := by
  sorry

#check hyperbola_equation

end hyperbola_equation_l1398_139898


namespace hyperbola_asymptote_implies_a_value_l1398_139881

-- Define the hyperbola equation
def hyperbola_equation (x y a : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 81 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = 3 * x

-- Theorem statement
theorem hyperbola_asymptote_implies_a_value (a : ℝ) :
  (a > 0) →
  (∃ x y : ℝ, hyperbola_equation x y a ∧ asymptote_equation x y) →
  a = 3 :=
by sorry

end hyperbola_asymptote_implies_a_value_l1398_139881


namespace bicycle_cost_proof_l1398_139819

def bicycle_cost (car_wash_income : ℕ) (lawn_mow_income : ℕ) (additional_needed : ℕ) : ℕ :=
  car_wash_income + lawn_mow_income + additional_needed

theorem bicycle_cost_proof :
  let car_wash_income := 3 * 10
  let lawn_mow_income := 2 * 13
  let additional_needed := 24
  bicycle_cost car_wash_income lawn_mow_income additional_needed = 80 := by
  sorry

end bicycle_cost_proof_l1398_139819


namespace inequality_proof_l1398_139871

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 4) 
  (h2 : c^2 + d^2 = 16) : 
  a*c + b*d ≤ 8 := by
sorry

end inequality_proof_l1398_139871


namespace hyperbola_eccentricity_l1398_139823

/-- The eccentricity of a hyperbola with equation x^2/2 - y^2 = 1 is √6/2 -/
theorem hyperbola_eccentricity :
  let a : ℝ := Real.sqrt 2
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 3
  let e : ℝ := c / a
  e = Real.sqrt 6 / 2 := by sorry

end hyperbola_eccentricity_l1398_139823


namespace cos_arcsin_three_fifths_l1398_139808

theorem cos_arcsin_three_fifths : 
  Real.cos (Real.arcsin (3/5)) = 4/5 := by sorry

end cos_arcsin_three_fifths_l1398_139808


namespace infinite_series_sum_l1398_139811

open Real

noncomputable def series_sum (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n+2) + 3^(2*n+2))

theorem infinite_series_sum :
  (∑' n, series_sum n) = (1 : ℝ) / 4 := by
  sorry

end infinite_series_sum_l1398_139811


namespace range_of_f_l1398_139803

-- Define the function f(x) = |x| - 4
def f (x : ℝ) : ℝ := |x| - 4

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ -4} :=
sorry

end range_of_f_l1398_139803


namespace katie_miles_ran_l1398_139893

theorem katie_miles_ran (katie_miles : ℝ) (adam_miles : ℝ) : 
  adam_miles = 3 * katie_miles →
  katie_miles + adam_miles = 240 →
  katie_miles = 60 := by
sorry

end katie_miles_ran_l1398_139893


namespace coin_and_die_probability_l1398_139891

theorem coin_and_die_probability : 
  let coin_prob := 1 / 2  -- Probability of getting heads on a fair coin
  let die_prob := 1 / 6   -- Probability of rolling a multiple of 5 on a 6-sided die
  coin_prob * die_prob = 1 / 12 :=
by sorry

end coin_and_die_probability_l1398_139891


namespace inequality_proof_l1398_139822

theorem inequality_proof (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3*y + x^2*y^2 - x*y^3 + y^4 > x^2 + y^2 := by
  sorry

end inequality_proof_l1398_139822


namespace divisible_by_42_l1398_139828

theorem divisible_by_42 (a : ℤ) : ∃ k : ℤ, a^7 - a = 42 * k := by sorry

end divisible_by_42_l1398_139828


namespace mixed_fractions_sum_product_l1398_139825

theorem mixed_fractions_sum_product : 
  (9 + 1/2 + 7 + 1/6 + 5 + 1/12 + 3 + 1/20 + 1 + 1/30) * 12 = 310 := by
  sorry

end mixed_fractions_sum_product_l1398_139825


namespace consecutive_non_prime_powers_l1398_139888

theorem consecutive_non_prime_powers (n : ℕ) : 
  ∃ x : ℤ, ∀ k ∈ Finset.range n, ¬ ∃ (p : ℕ) (m : ℕ), Prime p ∧ x + k.succ = p ^ m := by
  sorry

end consecutive_non_prime_powers_l1398_139888


namespace no_real_solutions_to_inequality_l1398_139827

theorem no_real_solutions_to_inequality :
  ¬∃ x : ℝ, x ≠ 5 ∧ (x^3 - 125) / (x - 5) < 0 := by
  sorry

end no_real_solutions_to_inequality_l1398_139827


namespace sum_of_zeros_transformed_parabola_l1398_139852

/-- The sum of zeros of a transformed parabola -/
theorem sum_of_zeros_transformed_parabola : 
  let f (x : ℝ) := (x - 3)^2 + 4
  let g (x : ℝ) := -(x - 7)^2 + 7
  ∃ a b : ℝ, g a = 0 ∧ g b = 0 ∧ a + b = 14 := by
sorry

end sum_of_zeros_transformed_parabola_l1398_139852


namespace jeff_shelter_cats_l1398_139860

def cat_shelter_problem (initial_cats : ℕ) 
  (monday_added : ℕ) (tuesday_added : ℕ) 
  (wednesday_adopted wednesday_added : ℕ) 
  (thursday_adopted thursday_added : ℕ)
  (friday_adopted friday_added : ℕ) : Prop :=
  let after_monday := initial_cats + monday_added
  let after_tuesday := after_monday + tuesday_added
  let after_wednesday := after_tuesday + wednesday_added - wednesday_adopted
  let after_thursday := after_wednesday + thursday_added - thursday_adopted
  let final_count := after_thursday + friday_added - friday_adopted
  final_count = 30

theorem jeff_shelter_cats : 
  cat_shelter_problem 20 9 6 8 2 3 3 2 3 :=
by sorry

end jeff_shelter_cats_l1398_139860


namespace chips_count_proof_l1398_139836

/-- The total number of chips Viviana and Susana have together -/
def total_chips (viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla : ℕ) : ℕ :=
  viviana_chocolate + viviana_vanilla + susana_chocolate + susana_vanilla

theorem chips_count_proof :
  ∀ (viviana_vanilla susana_chocolate : ℕ),
    viviana_vanilla = 20 →
    susana_chocolate = 25 →
    ∃ (viviana_chocolate susana_vanilla : ℕ),
      viviana_chocolate = susana_chocolate + 5 ∧
      susana_vanilla = (3 * viviana_vanilla) / 4 ∧
      total_chips viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla = 90 := by
  sorry

end chips_count_proof_l1398_139836


namespace arithmetic_sequence_geometric_mean_l1398_139863

def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 9 * d + (n - 1 : ℝ) * d

theorem arithmetic_sequence_geometric_mean (d : ℝ) :
  d ≠ 0 →
  ∃ k : ℕ, k > 0 ∧ 
    (arithmetic_sequence d k) ^ 2 = 
    (arithmetic_sequence d 1) * (arithmetic_sequence d (2 * k)) ∧
    k = 4 := by
  sorry

end arithmetic_sequence_geometric_mean_l1398_139863


namespace min_moves_to_equalize_l1398_139821

/-- Represents a casket with coins -/
structure Casket :=
  (coins : ℕ)

/-- Represents the circular arrangement of caskets -/
def CasketCircle := Vector Casket 7

/-- A move transfers one coin between neighboring caskets -/
def Move := Fin 7 → Fin 7

/-- Checks if a move is valid (transfers to a neighboring casket) -/
def isValidMove (m : Move) : Prop :=
  ∀ i, m i = (i + 1) % 7 ∨ m i = (i + 6) % 7 ∨ m i = i

/-- Applies a move to a casket circle -/
def applyMove (circle : CasketCircle) (m : Move) : CasketCircle :=
  sorry

/-- Checks if all caskets have the same number of coins -/
def isEqualized (circle : CasketCircle) : Prop :=
  ∀ i j, (circle.get i).coins = (circle.get j).coins

/-- The initial arrangement of caskets -/
def initialCircle : CasketCircle :=
  Vector.ofFn (λ i => match i with
    | 0 => ⟨9⟩
    | 1 => ⟨17⟩
    | 2 => ⟨12⟩
    | 3 => ⟨5⟩
    | 4 => ⟨18⟩
    | 5 => ⟨10⟩
    | 6 => ⟨20⟩)

/-- The main theorem to be proved -/
theorem min_moves_to_equalize :
  ∃ (moves : List Move),
    moves.length = 22 ∧
    (∀ m ∈ moves, isValidMove m) ∧
    isEqualized (moves.foldl applyMove initialCircle) ∧
    (∀ (otherMoves : List Move),
      otherMoves.length < 22 →
      ¬isEqualized (otherMoves.foldl applyMove initialCircle)) :=
  sorry

end min_moves_to_equalize_l1398_139821


namespace boys_in_class_l1398_139840

theorem boys_in_class (total_students : ℕ) (total_cost : ℕ) (boys_cost : ℕ) (girls_cost : ℕ)
  (h1 : total_students = 43)
  (h2 : total_cost = 1101)
  (h3 : boys_cost = 24)
  (h4 : girls_cost = 27) :
  ∃ (boys girls : ℕ),
    boys + girls = total_students ∧
    boys * boys_cost + girls * girls_cost = total_cost ∧
    boys = 20 := by
  sorry

end boys_in_class_l1398_139840


namespace most_likely_car_count_l1398_139816

/-- Represents the number of cars counted in a given time interval -/
structure CarCount where
  cars : ℕ
  seconds : ℕ

/-- Represents the total time taken by the train to pass -/
structure TotalTime where
  minutes : ℕ
  seconds : ℕ

/-- Calculates the most likely number of cars in the train -/
def calculateTotalCars (initial_count : CarCount) (total_time : TotalTime) : ℕ :=
  let total_seconds := total_time.minutes * 60 + total_time.seconds
  let rate := initial_count.cars / initial_count.seconds
  rate * total_seconds

/-- Theorem stating that given the conditions, the most likely number of cars is 70 -/
theorem most_likely_car_count 
  (initial_count : CarCount)
  (total_time : TotalTime)
  (h1 : initial_count = ⟨5, 15⟩)
  (h2 : total_time = ⟨3, 30⟩) :
  calculateTotalCars initial_count total_time = 70 := by
  sorry

#eval calculateTotalCars ⟨5, 15⟩ ⟨3, 30⟩

end most_likely_car_count_l1398_139816


namespace perfect_square_3_4_4_6_5_6_l1398_139859

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem perfect_square_3_4_4_6_5_6 :
  is_perfect_square (3^4 * 4^6 * 5^6) :=
by
  sorry

end perfect_square_3_4_4_6_5_6_l1398_139859


namespace midpoint_coordinate_sum_l1398_139875

theorem midpoint_coordinate_sum : 
  let p₁ : ℝ × ℝ := (10, -3)
  let p₂ : ℝ × ℝ := (-4, 7)
  let midpoint := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  (midpoint.1 + midpoint.2 : ℝ) = 5 := by
sorry

end midpoint_coordinate_sum_l1398_139875


namespace lcm_18_45_l1398_139807

theorem lcm_18_45 : Nat.lcm 18 45 = 90 := by
  sorry

end lcm_18_45_l1398_139807


namespace cube_paint_equality_l1398_139834

/-- The number of unit cubes with exactly one face painted in a cube of side length n -/
def one_face_painted (n : ℕ) : ℕ := 6 * (n - 2)^2

/-- The number of unit cubes with exactly two faces painted in a cube of side length n -/
def two_faces_painted (n : ℕ) : ℕ := 12 * (n - 2)

theorem cube_paint_equality (n : ℕ) (h : n > 3) :
  one_face_painted n = two_faces_painted n ↔ n = 4 := by
  sorry

end cube_paint_equality_l1398_139834


namespace negative_sqrt_16_l1398_139895

theorem negative_sqrt_16 : -Real.sqrt 16 = -4 := by sorry

end negative_sqrt_16_l1398_139895


namespace parabola_equation_l1398_139809

/-- A parabola in the Cartesian coordinate system -/
structure Parabola where
  -- The equation of the parabola in the form y^2 = ax
  a : ℝ
  -- Condition that the parabola is symmetric with respect to the x-axis
  x_axis_symmetry : True
  -- Condition that the vertex is at the origin
  vertex_at_origin : True
  -- Condition that the parabola passes through the point (2, 4)
  passes_through_point : a * 2 = 4^2

/-- Theorem stating that the parabola y^2 = 8x satisfies the given conditions -/
theorem parabola_equation : ∃ (p : Parabola), p.a = 8 := by
  sorry

end parabola_equation_l1398_139809


namespace right_triangle_leg_sum_equals_circle_diameters_sum_l1398_139838

/-- 
A right triangle with inscribed and circumscribed circles.
-/
structure RightTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Radii of inscribed and circumscribed circles
  r : ℝ
  R : ℝ
  -- Conditions
  right_angle : c^2 = a^2 + b^2
  c_is_diameter : c = 2 * R
  nonneg : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < r ∧ 0 < R

/-- 
In a right triangle, the sum of the legs is equal to the sum of 
the diameters of the inscribed and circumscribed circles.
-/
theorem right_triangle_leg_sum_equals_circle_diameters_sum 
  (t : RightTriangle) : t.a + t.b = 2 * t.R + 2 * t.r := by
  sorry

end right_triangle_leg_sum_equals_circle_diameters_sum_l1398_139838


namespace quadratic_roots_distance_l1398_139886

theorem quadratic_roots_distance (t : ℝ) (x₁ x₂ : ℂ) :
  x₁^2 + t*x₁ + 2 = 0 →
  x₂^2 + t*x₂ + 2 = 0 →
  x₁ ≠ x₂ →
  Complex.abs (x₁ - x₂) = 2 * Real.sqrt 2 →
  t = -4 ∨ t = 0 ∨ t = 4 := by
sorry

end quadratic_roots_distance_l1398_139886


namespace stating_traffic_light_probability_l1398_139874

/-- Represents the duration of a traffic light cycle in seconds. -/
def cycleDuration : ℕ := 80

/-- Represents the duration of time when proceeding is allowed (green + yellow) in seconds. -/
def proceedDuration : ℕ := 50

/-- Represents the duration of time when proceeding is not allowed (red) in seconds. -/
def stopDuration : ℕ := 30

/-- Represents the maximum waiting time in seconds for the probability calculation. -/
def maxWaitTime : ℕ := 10

/-- 
Theorem stating that the probability of waiting no more than 10 seconds to proceed 
in the given traffic light cycle is 3/4.
-/
theorem traffic_light_probability : 
  (proceedDuration + maxWaitTime : ℚ) / cycleDuration = 3/4 := by
  sorry

end stating_traffic_light_probability_l1398_139874


namespace midpoint_locus_is_annulus_l1398_139844

/-- Two non-intersecting circles in a plane --/
structure TwoCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  radius1 : ℝ
  radius2 : ℝ
  h1 : radius1 > 0
  h2 : radius2 > 0
  h3 : radius1 > radius2
  h4 : dist center1 center2 > radius1 + radius2

/-- The locus of midpoints of segments with endpoints on two non-intersecting circles --/
def midpointLocus (c : TwoCircles) : Set (ℝ × ℝ) :=
  {p | ∃ (a b : ℝ × ℝ), 
    dist a c.center1 = c.radius1 ∧ 
    dist b c.center2 = c.radius2 ∧ 
    p = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)}

/-- An annulus (ring) in a plane --/
def annulus (center : ℝ × ℝ) (inner_radius outer_radius : ℝ) : Set (ℝ × ℝ) :=
  {p | inner_radius ≤ dist p center ∧ dist p center ≤ outer_radius}

/-- The main theorem: the locus of midpoints is an annulus --/
theorem midpoint_locus_is_annulus (c : TwoCircles) :
  ∃ (center : ℝ × ℝ),
    midpointLocus c = annulus center ((c.radius1 - c.radius2) / 2) ((c.radius1 + c.radius2) / 2) :=
by sorry

end midpoint_locus_is_annulus_l1398_139844


namespace max_true_statements_l1398_139800

theorem max_true_statements (a b : ℝ) : 
  (∃ (s : Finset (Prop)), s.card = 2 ∧ 
    (∀ (p : Prop), p ∈ s → p) ∧
    s ⊆ {1/a > 1/b, a^3 > b^3, a > b, a < 0, b > 0}) ∧
  (∀ (s : Finset (Prop)), s.card > 2 → 
    s ⊆ {1/a > 1/b, a^3 > b^3, a > b, a < 0, b > 0} → 
    ∃ (p : Prop), p ∈ s ∧ ¬p) :=
sorry

end max_true_statements_l1398_139800


namespace sqrt_expression_equals_two_l1398_139856

theorem sqrt_expression_equals_two :
  Real.sqrt 12 + Real.sqrt 4 * (Real.sqrt 5 - Real.pi) ^ 0 - |(-2 * Real.sqrt 3)| = 2 := by
  sorry

end sqrt_expression_equals_two_l1398_139856


namespace repeating_decimal_sum_diff_l1398_139855

-- Define the repeating decimals
def repeating_234 : ℚ := 234 / 999
def repeating_567 : ℚ := 567 / 999
def repeating_891 : ℚ := 891 / 999

-- State the theorem
theorem repeating_decimal_sum_diff : 
  repeating_234 + repeating_567 - repeating_891 = -10 / 111 := by
sorry

end repeating_decimal_sum_diff_l1398_139855


namespace toms_marble_expense_l1398_139862

/-- Given Tom's expenses, prove the amount spent on marbles --/
theorem toms_marble_expense (skateboard_cost shorts_cost total_toys_cost : ℚ)
  (h1 : skateboard_cost = 9.46)
  (h2 : shorts_cost = 14.50)
  (h3 : total_toys_cost = 19.02) :
  total_toys_cost - skateboard_cost = 9.56 := by
  sorry

#check toms_marble_expense

end toms_marble_expense_l1398_139862


namespace shaded_area_theorem_total_shaded_area_l1398_139829

-- Define the length of the diagonal
def diagonal_length : ℝ := 8

-- Define the number of congruent squares
def num_squares : ℕ := 25

-- Theorem statement
theorem shaded_area_theorem (diagonal : ℝ) (num_squares : ℕ) 
  (h1 : diagonal = diagonal_length) 
  (h2 : num_squares = num_squares) : 
  (diagonal^2 / 2) = 32 := by
  sorry

-- Main theorem connecting the given conditions to the final area
theorem total_shaded_area : 
  (diagonal_length^2 / 2) = 32 := by
  exact shaded_area_theorem diagonal_length num_squares rfl rfl

end shaded_area_theorem_total_shaded_area_l1398_139829


namespace no_real_solutions_for_complex_norm_equation_l1398_139883

theorem no_real_solutions_for_complex_norm_equation :
  ¬∃ c : ℝ, Complex.abs (1 + c - 3*I) = 2 := by
sorry

end no_real_solutions_for_complex_norm_equation_l1398_139883


namespace negation_of_universal_statement_l1398_139847

-- Define the universe of discourse
variable (U : Type)

-- Define the predicate for being a domestic mobile phone
variable (D : U → Prop)

-- Define the predicate for having trap consumption
variable (T : U → Prop)

-- State the theorem
theorem negation_of_universal_statement :
  (¬ ∀ x, D x → T x) ↔ (∃ x, D x ∧ ¬ T x) :=
by sorry

end negation_of_universal_statement_l1398_139847


namespace expression_evaluation_l1398_139865

theorem expression_evaluation (y : ℝ) : 
  (1 : ℝ)^(4*y - 1) / (2 * ((7 : ℝ)⁻¹ + (4 : ℝ)⁻¹)) = 14/11 := by
  sorry

end expression_evaluation_l1398_139865


namespace unique_mod_10_solution_l1398_139841

theorem unique_mod_10_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -4229 [ZMOD 10] := by
  sorry

end unique_mod_10_solution_l1398_139841


namespace ball_drawing_theorem_l1398_139810

def num_red_balls : ℕ := 5
def num_black_balls : ℕ := 7
def red_ball_score : ℕ := 2
def black_ball_score : ℕ := 1
def total_balls_drawn : ℕ := 6
def max_score : ℕ := 8

def ways_to_draw_balls : ℕ :=
  (Nat.choose num_black_balls total_balls_drawn) +
  (Nat.choose num_red_balls 1 * Nat.choose num_black_balls (total_balls_drawn - 1))

theorem ball_drawing_theorem :
  ways_to_draw_balls = 112 :=
sorry

end ball_drawing_theorem_l1398_139810


namespace sum_of_squares_l1398_139853

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 400) : x^2 + y^2 = 60 := by
  sorry

end sum_of_squares_l1398_139853


namespace a_5_equals_17_l1398_139872

theorem a_5_equals_17 (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 3) : a 5 = 17 := by
  sorry

end a_5_equals_17_l1398_139872


namespace episodes_watched_per_day_l1398_139899

theorem episodes_watched_per_day 
  (total_episodes : ℕ) 
  (total_days : ℕ) 
  (h1 : total_episodes = 50) 
  (h2 : total_days = 10) 
  (h3 : total_episodes > 0) 
  (h4 : total_days > 0) : 
  (total_episodes : ℚ) / total_days = 1 / 10 := by
  sorry

#check episodes_watched_per_day

end episodes_watched_per_day_l1398_139899


namespace prime_equation_solutions_l1398_139861

theorem prime_equation_solutions (p : ℕ) (hp : Prime p) :
  ∀ x y : ℤ, p * (x + y) = x * y ↔
    (x = p * (p + 1) ∧ y = p + 1) ∨
    (x = 2 * p ∧ y = 2 * p) ∨
    (x = 0 ∧ y = 0) ∨
    (x = p * (1 - p) ∧ y = p - 1) := by
  sorry

end prime_equation_solutions_l1398_139861


namespace power_seven_mod_eight_l1398_139851

theorem power_seven_mod_eight : 7^202 % 8 = 1 := by sorry

end power_seven_mod_eight_l1398_139851


namespace paul_pencil_production_l1398_139879

/-- Calculates the number of pencils made per day given the initial stock, 
    final stock, number of pencils sold, and number of working days. -/
def pencils_per_day (initial_stock final_stock pencils_sold working_days : ℕ) : ℕ :=
  ((final_stock + pencils_sold) - initial_stock) / working_days

/-- Proves that Paul makes 100 pencils per day given the problem conditions. -/
theorem paul_pencil_production : 
  pencils_per_day 80 230 350 5 = 100 := by
  sorry

end paul_pencil_production_l1398_139879


namespace parallel_lines_a_value_l1398_139846

/-- Two lines are parallel if and only if their slopes are equal. -/
def parallel_lines (m1 a1 b1 m2 a2 b2 : ℝ) : Prop :=
  m1 * a2 = m2 * a1

/-- Given that the line 2x + ay + 1 = 0 is parallel to x - 4y - 1 = 0, prove that a = -8 -/
theorem parallel_lines_a_value (a : ℝ) :
  parallel_lines 2 a 1 1 (-4) (-1) → a = -8 := by
  sorry

end parallel_lines_a_value_l1398_139846


namespace sqrt_of_negative_eight_squared_l1398_139896

theorem sqrt_of_negative_eight_squared : Real.sqrt ((-8)^2) = 8 := by sorry

end sqrt_of_negative_eight_squared_l1398_139896


namespace line_slope_problem_l1398_139806

theorem line_slope_problem (k : ℝ) (h1 : k > 0) 
  (h2 : (k + 1) * (2 - k) = k - 5) : k = (1 + Real.sqrt 29) / 2 := by
  sorry

end line_slope_problem_l1398_139806


namespace cultural_festival_talents_l1398_139845

theorem cultural_festival_talents (total_students : ℕ) 
  (cannot_sing cannot_dance cannot_act no_talents : ℕ) : ℕ :=
by
  -- Define the conditions
  have h1 : total_students = 150 := by sorry
  have h2 : cannot_sing = 75 := by sorry
  have h3 : cannot_dance = 95 := by sorry
  have h4 : cannot_act = 40 := by sorry
  have h5 : no_talents = 20 := by sorry
  
  -- Define the number of students with each talent
  let can_sing := total_students - cannot_sing
  let can_dance := total_students - cannot_dance
  let can_act := total_students - cannot_act
  
  -- Define the sum of students with at least one talent
  let with_talents := total_students - no_talents
  
  -- Define the sum of all talents (ignoring overlaps)
  let sum_talents := can_sing + can_dance + can_act
  
  -- Calculate the number of students with exactly two talents
  let two_talents := sum_talents - with_talents
  
  -- Prove that two_talents equals 90
  have h6 : two_talents = 90 := by sorry
  
  -- Return the result
  exact two_talents

-- The theorem states that given the conditions, 
-- the number of students with exactly two talents is 90

end cultural_festival_talents_l1398_139845


namespace tenth_term_is_negative_eight_l1398_139815

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  first_positive : a 1 > 0
  sum_condition : a 1 + a 7 = 2
  product_condition : a 5 * a 6 = -8
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = a n * q

/-- The 10th term of the geometric sequence is -8 -/
theorem tenth_term_is_negative_eight (seq : GeometricSequence) : seq.a 10 = -8 := by
  sorry

end tenth_term_is_negative_eight_l1398_139815


namespace infinite_solutions_equation_l1398_139830

theorem infinite_solutions_equation :
  ∃ (S : Set (ℕ × ℕ × ℕ)), 
    (∀ (x y z : ℕ), (x, y, z) ∈ S → 
      x > 2008 ∧ y > 2008 ∧ z > 2008 ∧ 
      x^2 + y^2 + z^2 - x*y*z + 10 = 0) ∧
    Set.Infinite S :=
sorry

end infinite_solutions_equation_l1398_139830


namespace sin_180_degrees_l1398_139824

theorem sin_180_degrees : Real.sin (π) = 0 := by sorry

end sin_180_degrees_l1398_139824


namespace cheaper_plan_threshold_min_gigabytes_for_cheaper_plan_y_l1398_139877

/-- Represents the cost of an internet plan in cents -/
def PlanCost (initialFee : ℕ) (costPerGB : ℕ) (gigabytes : ℕ) : ℕ :=
  initialFee * 100 + costPerGB * gigabytes

theorem cheaper_plan_threshold :
  ∀ g : ℕ, PlanCost 0 20 g ≤ PlanCost 30 10 g ↔ g ≤ 300 :=
by sorry

theorem min_gigabytes_for_cheaper_plan_y :
  ∃ g : ℕ, g = 301 ∧
    (∀ h : ℕ, PlanCost 0 20 h > PlanCost 30 10 h → h ≥ g) ∧
    PlanCost 0 20 g > PlanCost 30 10 g :=
by sorry

end cheaper_plan_threshold_min_gigabytes_for_cheaper_plan_y_l1398_139877


namespace weight_of_raisins_l1398_139826

/-- Given the total weight of snacks and the weight of peanuts, 
    prove that the weight of raisins is 0.4 pounds. -/
theorem weight_of_raisins (total_weight peanuts_weight : ℝ) 
  (h1 : total_weight = 0.5)
  (h2 : peanuts_weight = 0.1) : 
  total_weight - peanuts_weight = 0.4 := by
sorry

end weight_of_raisins_l1398_139826


namespace yard_area_l1398_139858

/-- Given a rectangular yard where one side is 40 feet and the sum of the other three sides is 56 feet,
    the area of the yard is 320 square feet. -/
theorem yard_area (length width : ℝ) : 
  length = 40 →
  2 * width + length = 56 →
  length * width = 320 := by
  sorry

end yard_area_l1398_139858


namespace max_value_of_y_l1398_139818

open Complex

theorem max_value_of_y (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  let z : ℂ := 3 * cos θ + 2 * I * sin θ
  let y : Real := θ - arg z
  ∃ (max_y : Real), ∀ (θ' : Real), 0 < θ' ∧ θ' < Real.pi / 2 →
    let z' : ℂ := 3 * cos θ' + 2 * I * sin θ'
    let y' : Real := θ' - arg z'
    y' ≤ max_y ∧ max_y = Real.arctan (Real.sqrt 6 / 12) := by
  sorry

end max_value_of_y_l1398_139818


namespace product_selection_probabilities_l1398_139843

/-- A box containing products -/
structure Box where
  total : ℕ
  good : ℕ
  defective : ℕ
  h_total : total = good + defective

/-- The probability of an event when selecting two products from a box -/
def probability (box : Box) (favorable : ℕ) : ℚ :=
  favorable / (box.total.choose 2)

theorem product_selection_probabilities (box : Box) 
  (h_total : box.total = 6)
  (h_good : box.good = 4)
  (h_defective : box.defective = 2) :
  probability box (box.good * box.defective) = 8 / 15 ∧
  probability box (box.good.choose 2) = 2 / 5 ∧
  1 - probability box (box.good.choose 2) = 3 / 5 := by
  sorry

end product_selection_probabilities_l1398_139843


namespace marbleChoices_eq_56_l1398_139866

/-- A function that returns the number of ways to choose one marble from a set of 15 
    and two ordered marbles from a set of 8 such that the sum of the two chosen marbles 
    equals the number on the single chosen marble -/
def marbleChoices : ℕ :=
  let jessicaMarbles := Finset.range 15
  let myMarbles := Finset.range 8
  Finset.sum jessicaMarbles (λ j => 
    Finset.sum myMarbles (λ m1 => 
      Finset.sum myMarbles (λ m2 => 
        if m1 + m2 + 2 = j + 1 then 1 else 0)))

theorem marbleChoices_eq_56 : marbleChoices = 56 := by
  sorry

end marbleChoices_eq_56_l1398_139866
