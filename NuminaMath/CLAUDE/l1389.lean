import Mathlib

namespace NUMINAMATH_CALUDE_valid_numbers_characterization_l1389_138950

def is_valid (n : ℕ) : Prop :=
  n ≥ 10 ∧ (100 * (n / 10) + n % 10) % n = 0

def S : Set ℕ := {10, 20, 30, 40, 50, 60, 70, 80, 90, 15, 18, 45}

theorem valid_numbers_characterization :
  ∀ n : ℕ, is_valid n ↔ n ∈ S :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_characterization_l1389_138950


namespace NUMINAMATH_CALUDE_prob_eight_rolls_divisible_by_four_l1389_138955

/-- The probability that a single die roll is even -/
def p_even : ℚ := 1/2

/-- The number of dice rolls -/
def n : ℕ := 8

/-- The probability mass function of the binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability that the product of n dice rolls is divisible by 4 -/
def prob_divisible_by_four (n : ℕ) (p : ℚ) : ℚ :=
  1 - (binomial_pmf n p 0 + binomial_pmf n p 1)

theorem prob_eight_rolls_divisible_by_four :
  prob_divisible_by_four n p_even = 247/256 := by
  sorry

#eval prob_divisible_by_four n p_even

end NUMINAMATH_CALUDE_prob_eight_rolls_divisible_by_four_l1389_138955


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1389_138993

/-- Given a sum of money P put at simple interest for 7 years at rate R%,
    if increasing the rate by 2% results in 140 more interest, then P = 1000. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 2) * 7) / 100 = (P * R * 7) / 100 + 140 → P = 1000 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1389_138993


namespace NUMINAMATH_CALUDE_problem_statement_l1389_138912

theorem problem_statement (A B : ℝ) :
  (∀ x : ℝ, x ≠ 5 → A / (x - 5) + B * (x + 1) = (-2 * x^2 + 16 * x + 18) / (x - 5)) →
  A + B = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1389_138912


namespace NUMINAMATH_CALUDE_euler_line_equation_l1389_138953

/-- Triangle ABC with vertices A(3,1), B(4,2), and C(2,3) -/
structure Triangle where
  A : ℝ × ℝ := (3, 1)
  B : ℝ × ℝ := (4, 2)
  C : ℝ × ℝ := (2, 3)

/-- The Euler line of a triangle -/
def EulerLine (t : Triangle) : ℝ → ℝ → Prop :=
  fun x y => x + y - 5 = 0

/-- Theorem: The equation of the Euler line for the given triangle ABC is x + y - 5 = 0 -/
theorem euler_line_equation (t : Triangle) : EulerLine t = fun x y => x + y - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_euler_line_equation_l1389_138953


namespace NUMINAMATH_CALUDE_correct_senior_sample_l1389_138931

/-- Represents the number of students to be selected from each grade level in a stratified sample -/
structure StratifiedSample where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- Calculates the correct stratified sample given the school's demographics -/
def calculateStratifiedSample (totalStudents : ℕ) (freshmen : ℕ) (sophomoreProbability : ℚ) (sampleSize : ℕ) : StratifiedSample :=
  sorry

theorem correct_senior_sample :
  let totalStudents : ℕ := 2000
  let freshmen : ℕ := 760
  let sophomoreProbability : ℚ := 37/100
  let sampleSize : ℕ := 20
  let sample := calculateStratifiedSample totalStudents freshmen sophomoreProbability sampleSize
  sample.seniors = 5 := by sorry

end NUMINAMATH_CALUDE_correct_senior_sample_l1389_138931


namespace NUMINAMATH_CALUDE_carol_extra_invitations_l1389_138902

def invitation_problem (packs_bought : ℕ) (invitations_per_pack : ℕ) (friends_to_invite : ℕ) : ℕ :=
  let total_invitations := packs_bought * invitations_per_pack
  let additional_packs_needed := ((friends_to_invite - total_invitations) + invitations_per_pack - 1) / invitations_per_pack
  let final_invitations := total_invitations + additional_packs_needed * invitations_per_pack
  final_invitations - friends_to_invite

theorem carol_extra_invitations :
  invitation_problem 3 5 23 = 2 :=
sorry

end NUMINAMATH_CALUDE_carol_extra_invitations_l1389_138902


namespace NUMINAMATH_CALUDE_circle_collinearity_l1389_138901

-- Define the circle ω
def circle_ω (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | dist P ((A + B) / 2) = dist A B / 2}

-- Define a point on the circle
def point_on_circle (O : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : Prop :=
  O ∈ ω

-- Define orthogonal projection
def orthogonal_projection (O H : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  (H.1 - A.1) * (B.2 - A.2) = (H.2 - A.2) * (B.1 - A.1) ∧
  (O.1 - H.1) * (B.1 - A.1) + (O.2 - H.2) * (B.2 - A.2) = 0

-- Define the intersection of two circles
def circle_intersection (O H X Y : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : Prop :=
  X ∈ ω ∧ Y ∈ ω ∧
  dist X O = dist O H ∧ dist Y O = dist O H

-- Define collinearity
def collinear (P Q R : ℝ × ℝ) : Prop :=
  (Q.2 - P.2) * (R.1 - P.1) = (R.2 - P.2) * (Q.1 - P.1)

-- The main theorem
theorem circle_collinearity 
  (A B O H X Y : ℝ × ℝ) (ω : Set (ℝ × ℝ)) :
  ω = circle_ω A B →
  point_on_circle O ω →
  orthogonal_projection O H A B →
  circle_intersection O H X Y ω →
  collinear X Y ((O + H) / 2) :=
sorry

end NUMINAMATH_CALUDE_circle_collinearity_l1389_138901


namespace NUMINAMATH_CALUDE_festival_remaining_money_l1389_138945

def festival_spending (total_budget food_cost : ℕ) : ℕ :=
  let ride_cost := 3 * food_cost
  let game_cost := ride_cost / 2
  total_budget - (food_cost + ride_cost + game_cost)

theorem festival_remaining_money :
  festival_spending 100 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_festival_remaining_money_l1389_138945


namespace NUMINAMATH_CALUDE_pairwise_sum_difference_l1389_138900

theorem pairwise_sum_difference (n : ℕ) (x : Fin n → ℝ) 
  (h_n : n ≥ 4) 
  (h_pos : ∀ i, x i > 0) : 
  ∃ (i j k l : Fin n), i ≠ j ∧ k ≠ l ∧ 
    (x i + x j) ≤ (x k + x l) * (2 : ℝ)^(1 / (n - 2 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_pairwise_sum_difference_l1389_138900


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_l1389_138905

theorem smallest_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 3 ∣ m ∧ 4 ∣ m ∧ 5 ∣ m → m ≥ n) ∧
  n = 1020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_l1389_138905


namespace NUMINAMATH_CALUDE_min_integer_value_of_fraction_l1389_138964

theorem min_integer_value_of_fraction (x : ℝ) : 
  ⌊(4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 3)⌋ ≥ -15 ∧ 
  ∃ y : ℝ, ⌊(4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 3)⌋ = -15 := by
  sorry

end NUMINAMATH_CALUDE_min_integer_value_of_fraction_l1389_138964


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1389_138929

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4/3))
  A ≤ 3 ∧ (A = 3 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1389_138929


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1389_138999

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1389_138999


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l1389_138925

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]

theorem matrix_inverse_proof :
  A⁻¹ = !![(-1), (-3); (-2), (-5)] := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l1389_138925


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l1389_138954

theorem sqrt_sum_equals_seven (y : ℝ) 
  (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) : 
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l1389_138954


namespace NUMINAMATH_CALUDE_union_subset_iff_m_in_range_l1389_138920

def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def C (m : ℝ) : Set ℝ := {x : ℝ | m * x + 1 > 0}

theorem union_subset_iff_m_in_range :
  ∀ m : ℝ, (A ∪ B) ⊆ C m ↔ m ∈ Set.Icc (-1/2) 1 := by sorry

end NUMINAMATH_CALUDE_union_subset_iff_m_in_range_l1389_138920


namespace NUMINAMATH_CALUDE_bales_in_barn_l1389_138926

/-- The number of bales in the barn after stacking more bales -/
def total_bales (initial : ℕ) (stacked : ℕ) : ℕ := initial + stacked

/-- Theorem: Given 22 initial bales and 67 newly stacked bales, the total is 89 bales -/
theorem bales_in_barn : total_bales 22 67 = 89 := by
  sorry

end NUMINAMATH_CALUDE_bales_in_barn_l1389_138926


namespace NUMINAMATH_CALUDE_billy_tickets_l1389_138990

/-- The number of times Billy rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride -/
def tickets_per_ride : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := (ferris_rides + bumper_rides) * tickets_per_ride

theorem billy_tickets : total_tickets = 50 := by
  sorry

end NUMINAMATH_CALUDE_billy_tickets_l1389_138990


namespace NUMINAMATH_CALUDE_m_range_l1389_138966

theorem m_range (x m : ℝ) : 
  (∀ x, x^2 + 3*x - 4 < 0 → (x - m)^2 > 3*(x - m)) ∧ 
  (∃ x, (x - m)^2 > 3*(x - m) ∧ x^2 + 3*x - 4 ≥ 0) → 
  m ≥ 1 ∨ m ≤ -7 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l1389_138966


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l1389_138930

theorem scientific_notation_of_small_number : 
  ∃ (a : ℝ) (n : ℤ), 0.0000001 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l1389_138930


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_l1389_138914

/-- The pricing function for the first caterer -/
def first_caterer (x : ℕ) : ℚ := 150 + 18 * x

/-- The pricing function for the second caterer -/
def second_caterer (x : ℕ) : ℚ := 250 + 15 * x

/-- The least number of people for which the second caterer is cheaper -/
def least_people : ℕ := 34

theorem second_caterer_cheaper :
  (∀ n : ℕ, n ≥ least_people → second_caterer n < first_caterer n) ∧
  (∀ n : ℕ, n < least_people → second_caterer n ≥ first_caterer n) := by
  sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_l1389_138914


namespace NUMINAMATH_CALUDE_total_toys_proof_l1389_138957

/-- The number of toys Kamari has -/
def kamari_toys : ℝ := 65

/-- The number of toys Anais has -/
def anais_toys : ℝ := kamari_toys + 30.5

/-- The number of toys Lucien has -/
def lucien_toys : ℝ := 2 * kamari_toys

/-- The total number of toys Anais and Kamari have together -/
def anais_kamari_total : ℝ := 160.5

theorem total_toys_proof :
  kamari_toys + anais_toys + lucien_toys = 290.5 ∧
  anais_toys = kamari_toys + 30.5 ∧
  lucien_toys = 2 * kamari_toys ∧
  anais_toys + kamari_toys = anais_kamari_total :=
by sorry

end NUMINAMATH_CALUDE_total_toys_proof_l1389_138957


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_solution_x_l1389_138909

/-- The percentage of alcohol in a solution that, when mixed with another solution,
    results in a specific alcohol concentration. -/
theorem alcohol_percentage_in_solution_x 
  (volume_x : ℝ) 
  (volume_y : ℝ) 
  (percent_y : ℝ) 
  (percent_final : ℝ) 
  (h1 : volume_x = 300)
  (h2 : volume_y = 900)
  (h3 : percent_y = 0.30)
  (h4 : percent_final = 0.25)
  : ∃ (percent_x : ℝ), 
    percent_x = 0.10 ∧ 
    volume_x * percent_x + volume_y * percent_y = (volume_x + volume_y) * percent_final :=
sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_solution_x_l1389_138909


namespace NUMINAMATH_CALUDE_constant_term_value_l1389_138913

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion of (1+2x)^n
def expansion (n : ℕ) (x : ℝ) : ℝ := sorry

-- Define the coefficient of the r-th term in the expansion
def coefficient (n r : ℕ) : ℝ := sorry

-- Define the condition that only the fourth term has the maximum coefficient
def fourth_term_max (n : ℕ) : Prop :=
  ∀ r, r ≠ 3 → coefficient n r ≤ coefficient n 3

-- Main theorem
theorem constant_term_value (n : ℕ) :
  fourth_term_max n →
  (expansion n 0 + coefficient n 2) = 61 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_value_l1389_138913


namespace NUMINAMATH_CALUDE_quadratic_rational_root_implies_even_coeff_l1389_138969

theorem quadratic_rational_root_implies_even_coeff 
  (a b c : ℤ) (h_a_nonzero : a ≠ 0) 
  (h_rational_root : ∃ (p q : ℤ) (h_q_nonzero : q ≠ 0), a * (p * p) + b * (p * q) + c * (q * q) = 0) :
  Even a ∨ Even b ∨ Even c := by
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_implies_even_coeff_l1389_138969


namespace NUMINAMATH_CALUDE_fraction_of_one_third_is_one_eighth_l1389_138908

theorem fraction_of_one_third_is_one_eighth (a b c d : ℚ) : 
  a = 1/3 → b = 1/8 → (b/a = c/d) → (c = 3 ∧ d = 8) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_one_third_is_one_eighth_l1389_138908


namespace NUMINAMATH_CALUDE_tetrahedron_volume_bound_l1389_138949

/-- A tetrahedron represented by four vertices in 3D space -/
structure Tetrahedron where
  v1 : Fin 3 → ℝ
  v2 : Fin 3 → ℝ
  v3 : Fin 3 → ℝ
  v4 : Fin 3 → ℝ

/-- A cube represented by its lower and upper bounds in 3D space -/
structure Cube where
  lower : Fin 3 → ℝ
  upper : Fin 3 → ℝ

/-- Function to calculate the volume of a tetrahedron -/
def volume_tetrahedron (t : Tetrahedron) : ℝ := sorry

/-- Function to calculate the volume of a cube -/
def volume_cube (c : Cube) : ℝ := sorry

/-- Function to check if a tetrahedron is inside a cube -/
def is_inside (t : Tetrahedron) (c : Cube) : Prop := sorry

/-- The main theorem: volume of tetrahedron inside unit cube is at most 1/3 -/
theorem tetrahedron_volume_bound (t : Tetrahedron) (c : Cube) :
  is_inside t c →
  (∀ i, c.lower i = 0 ∧ c.upper i = 1) →
  volume_tetrahedron t ≤ (1/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_bound_l1389_138949


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l1389_138944

theorem quadratic_roots_nature (k : ℝ) : 
  (∃ x y : ℝ, x * y = 12 ∧ x^2 - 4*k*x + 3*k^2 + 1 = 0 ∧ y^2 - 4*k*y + 3*k^2 + 1 = 0) →
  (∃ x : ℝ, x^2 - 4*k*x + 3*k^2 + 1 = 0 ∧ (∀ y : ℝ, y^2 - 4*k*y + 3*k^2 + 1 = 0 → y = x)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l1389_138944


namespace NUMINAMATH_CALUDE_wall_painting_problem_l1389_138973

theorem wall_painting_problem (heidi_rate peter_rate : ℚ) 
  (heidi_time peter_time painting_time : ℕ) :
  heidi_rate = 1 / 60 →
  peter_rate = 1 / 75 →
  heidi_time = 60 →
  peter_time = 75 →
  painting_time = 15 →
  (heidi_rate + peter_rate) * painting_time = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_wall_painting_problem_l1389_138973


namespace NUMINAMATH_CALUDE_rectangular_solid_length_l1389_138907

/-- Represents the dimensions of a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.depth + solid.width * solid.depth)

theorem rectangular_solid_length 
  (solid : RectangularSolid) 
  (h1 : solid.width = 5)
  (h2 : solid.depth = 2)
  (h3 : surfaceArea solid = 104) : 
  solid.length = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_length_l1389_138907


namespace NUMINAMATH_CALUDE_temperature_frequency_l1389_138924

def temperatures : List ℤ := [-2, 0, 3, -1, 1, 0, 4]

theorem temperature_frequency :
  (temperatures.filter (λ t => t > 0)).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_temperature_frequency_l1389_138924


namespace NUMINAMATH_CALUDE_trig_simplification_l1389_138906

theorem trig_simplification :
  (Real.cos (5 * π / 180))^2 - (Real.sin (5 * π / 180))^2 =
  2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l1389_138906


namespace NUMINAMATH_CALUDE_ratio_equality_l1389_138984

theorem ratio_equality (a b : ℝ) (h1 : 5 * a = 3 * b) (h2 : a ≠ 0) (h3 : b ≠ 0) : b / a = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1389_138984


namespace NUMINAMATH_CALUDE_rectangle_area_l1389_138904

theorem rectangle_area (x y : ℕ) : 
  1 ≤ x ∧ x < 10 ∧ 1 ≤ y ∧ y < 10 →
  ∃ n : ℕ, (1100 * x + 11 * y) = n^2 →
  x * y = 28 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1389_138904


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_60_l1389_138985

theorem smallest_divisible_by_18_and_60 : ∀ n : ℕ, n > 0 ∧ 18 ∣ n ∧ 60 ∣ n → n ≥ 180 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_60_l1389_138985


namespace NUMINAMATH_CALUDE_certain_number_exists_and_unique_l1389_138923

theorem certain_number_exists_and_unique : 
  ∃! x : ℝ, (40 * 30 + (12 + x) * 3) / 5 = 1212 := by sorry

end NUMINAMATH_CALUDE_certain_number_exists_and_unique_l1389_138923


namespace NUMINAMATH_CALUDE_geometric_proof_l1389_138938

/-- The problem setup for the geometric proof -/
structure GeometricSetup where
  -- Line l equation
  l : ℝ → ℝ → Prop
  l_def : ∀ x y, l x y ↔ x + 2 * y - 1 = 0

  -- Circle C equations
  C : ℝ → ℝ → Prop
  C_def : ∀ x y, C x y ↔ ∃ φ, x = 3 + 3 * Real.cos φ ∧ y = 3 * Real.sin φ

  -- Ray OM
  α : ℝ
  α_range : 0 < α ∧ α < Real.pi / 2

  -- Function to convert Cartesian to polar coordinates
  to_polar : ℝ × ℝ → ℝ × ℝ

  -- Function to get the length of OP
  OP_length : ℝ

  -- Function to get the length of OQ
  OQ_length : ℝ

/-- The main theorem to be proved -/
theorem geometric_proof (setup : GeometricSetup) : 
  setup.OP_length * setup.OQ_length = 6 → setup.α = Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_proof_l1389_138938


namespace NUMINAMATH_CALUDE_toothpick_grid_50x40_l1389_138922

/-- Calculates the total number of toothpicks in a rectangular grid. -/
def total_toothpicks (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem stating that a 50x40 toothpick grid uses 4090 toothpicks. -/
theorem toothpick_grid_50x40 :
  total_toothpicks 50 40 = 4090 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_grid_50x40_l1389_138922


namespace NUMINAMATH_CALUDE_cube_edge_color_probability_l1389_138946

theorem cube_edge_color_probability :
  let num_edges : ℕ := 12
  let num_colors : ℕ := 2
  let num_visible_faces : ℕ := 4
  let prob_same_color_face : ℝ := 2 / 2^4

  (1 : ℝ) / 256 = prob_same_color_face^num_visible_faces := by sorry

end NUMINAMATH_CALUDE_cube_edge_color_probability_l1389_138946


namespace NUMINAMATH_CALUDE_share_calculation_l1389_138960

theorem share_calculation (total : ℝ) (a b c : ℝ) : 
  total = 300 →
  a + b + c = total →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a = 120 := by sorry

end NUMINAMATH_CALUDE_share_calculation_l1389_138960


namespace NUMINAMATH_CALUDE_scarf_price_reduction_l1389_138915

/-- Calculates the final price of a scarf after two successive price reductions -/
theorem scarf_price_reduction (original_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ) :
  original_price = 10 ∧ first_reduction = 0.3 ∧ second_reduction = 0.5 →
  original_price * (1 - first_reduction) * (1 - second_reduction) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_scarf_price_reduction_l1389_138915


namespace NUMINAMATH_CALUDE_power_two_vs_square_l1389_138941

theorem power_two_vs_square (n : ℕ+) :
  (n = 2 ∨ n = 4 → 2^(n:ℕ) = n^2) ∧
  (n = 3 → 2^(n:ℕ) < n^2) ∧
  (n = 1 ∨ n > 4 → 2^(n:ℕ) > n^2) := by
  sorry

end NUMINAMATH_CALUDE_power_two_vs_square_l1389_138941


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l1389_138903

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    n ≥ 100 ∧ 
    n < 1000 ∧ 
    75 * n % 345 = 225 ∧ 
    (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 75 * m % 345 = 225 → m ≥ n) ∧
    n = 118 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l1389_138903


namespace NUMINAMATH_CALUDE_exponential_inequality_l1389_138943

theorem exponential_inequality (a b m : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : a ≠ 1) 
  (h4 : b ≠ 1) 
  (h5 : 0 < m) 
  (h6 : m < 1) : 
  m^a < m^b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1389_138943


namespace NUMINAMATH_CALUDE_factorization_condition_l1389_138958

-- Define the polynomial
def polynomial (x y m : ℤ) : ℤ := x^2 + 5*x*y + x + 2*m*y - 10

-- Define what it means for a polynomial to be factorizable into linear factors with integer coefficients
def is_factorizable (m : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), ∀ (x y : ℤ),
    polynomial x y m = (a*x + b*y + c) * (d*x + e*y + f)

-- State the theorem
theorem factorization_condition :
  ∀ m : ℤ, is_factorizable m ↔ m = 5 := by sorry

end NUMINAMATH_CALUDE_factorization_condition_l1389_138958


namespace NUMINAMATH_CALUDE_fraction_simplification_l1389_138935

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) : (x - y) / (y - x) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1389_138935


namespace NUMINAMATH_CALUDE_museum_trip_total_l1389_138971

/-- The number of people on the first bus -/
def first_bus : ℕ := 12

/-- The number of people on the second bus -/
def second_bus : ℕ := 2 * first_bus

/-- The number of people on the third bus -/
def third_bus : ℕ := second_bus - 6

/-- The number of people on the fourth bus -/
def fourth_bus : ℕ := first_bus + 9

/-- The total number of people going to the museum -/
def total_people : ℕ := first_bus + second_bus + third_bus + fourth_bus

theorem museum_trip_total : total_people = 75 := by
  sorry

end NUMINAMATH_CALUDE_museum_trip_total_l1389_138971


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1389_138997

-- Define the points
def A : ℝ × ℝ := (-6, -1)
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (-3, -2)

-- Define the parallelogram property
def is_parallelogram (A B C M : ℝ × ℝ) : Prop :=
  (B.1 - A.1, B.2 - A.2) = (M.1 - C.1, M.2 - C.2)

-- Theorem statement
theorem parallelogram_fourth_vertex :
  ∃ M : ℝ × ℝ, is_parallelogram A B C M ∧ M = (4, 1) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1389_138997


namespace NUMINAMATH_CALUDE_original_number_exists_and_unique_l1389_138910

theorem original_number_exists_and_unique : 
  ∃! x : ℚ, 4 * (3 * x + 29) = 212 := by sorry

end NUMINAMATH_CALUDE_original_number_exists_and_unique_l1389_138910


namespace NUMINAMATH_CALUDE_periodic_sequence_prime_period_l1389_138975

/-- A sequence a is periodic with period m if a(m+n) = a(n) for all n -/
def isPeriodic (a : ℕ → ℂ) (m : ℕ) : Prop :=
  ∀ n, a (m + n) = a n

/-- m is the smallest positive period of sequence a -/
def isSmallestPeriod (a : ℕ → ℂ) (m : ℕ) : Prop :=
  isPeriodic a m ∧ ∀ k, 0 < k → k < m → ¬isPeriodic a k

/-- q is an m-th root of unity -/
def isRootOfUnity (q : ℂ) (m : ℕ) : Prop :=
  q ^ m = 1

theorem periodic_sequence_prime_period
  (q : ℂ) (m : ℕ) 
  (h1 : isSmallestPeriod (fun n => q^n) m)
  (h2 : m ≥ 2)
  (h3 : Nat.Prime m) :
  isRootOfUnity q m ∧ q ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_sequence_prime_period_l1389_138975


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1389_138959

theorem inequality_system_solution (m : ℝ) :
  (∀ x : ℝ, (3 * x - 9 > 0 ∧ x > m) ↔ x > 3) →
  m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1389_138959


namespace NUMINAMATH_CALUDE_george_total_blocks_l1389_138968

/-- The total number of blocks George has when combining large, small, and medium blocks. -/
def total_blocks (large_boxes small_boxes large_per_box small_per_box case_boxes medium_per_box : ℕ) : ℕ :=
  (large_boxes * large_per_box) + (small_boxes * small_per_box) + (case_boxes * medium_per_box)

/-- Theorem stating that George has 86 blocks in total. -/
theorem george_total_blocks :
  total_blocks 2 3 6 8 5 10 = 86 := by
  sorry

end NUMINAMATH_CALUDE_george_total_blocks_l1389_138968


namespace NUMINAMATH_CALUDE_simplify_fraction_l1389_138951

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1389_138951


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1389_138980

theorem cubic_root_sum (a b c : ℝ) : 
  (15 * a^3 - 30 * a^2 + 20 * a - 2 = 0) →
  (15 * b^3 - 30 * b^2 + 20 * b - 2 = 0) →
  (15 * c^3 - 30 * c^2 + 20 * c - 2 = 0) →
  (0 < a ∧ a < 1) →
  (0 < b ∧ b < 1) →
  (0 < c ∧ c < 1) →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1389_138980


namespace NUMINAMATH_CALUDE_triangle_least_perimeter_l1389_138948

theorem triangle_least_perimeter (a b c : ℕ) : 
  a = 24 → b = 37 → c > 0 → a + b > c → a + c > b → b + c > a → 
  (∀ x : ℕ, x > 0 → a + b > x → a + x > b → b + x > a → a + b + c ≤ a + b + x) →
  a + b + c = 75 :=
sorry

end NUMINAMATH_CALUDE_triangle_least_perimeter_l1389_138948


namespace NUMINAMATH_CALUDE_find_divisor_l1389_138919

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 23) (h2 : quotient = 4) (h3 : remainder = 3) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 5 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1389_138919


namespace NUMINAMATH_CALUDE_kathryn_salary_l1389_138916

/-- Calculates Kathryn's monthly salary given her expenses and remaining money --/
def monthly_salary (rent : ℕ) (remaining : ℕ) : ℕ :=
  let food_travel := 2 * rent
  let total_expenses := rent + food_travel
  let shared_rent := rent / 2
  let adjusted_expenses := total_expenses - (rent - shared_rent)
  adjusted_expenses + remaining

/-- Proves that Kathryn's monthly salary is $5000 given the problem conditions --/
theorem kathryn_salary :
  let rent : ℕ := 1200
  let remaining : ℕ := 2000
  monthly_salary rent remaining = 5000 := by
  sorry

#eval monthly_salary 1200 2000

end NUMINAMATH_CALUDE_kathryn_salary_l1389_138916


namespace NUMINAMATH_CALUDE_students_exceed_pets_l1389_138983

/-- Proves that in 6 classrooms, where each classroom has 22 students, 3 pet rabbits, 
    and 1 pet hamster, the number of students exceeds the number of pets by 108. -/
theorem students_exceed_pets : 
  let num_classrooms : ℕ := 6
  let students_per_classroom : ℕ := 22
  let rabbits_per_classroom : ℕ := 3
  let hamsters_per_classroom : ℕ := 1
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_pets : ℕ := num_classrooms * (rabbits_per_classroom + hamsters_per_classroom)
  total_students - total_pets = 108 := by
  sorry

end NUMINAMATH_CALUDE_students_exceed_pets_l1389_138983


namespace NUMINAMATH_CALUDE_reeyas_average_score_l1389_138978

def reeyas_scores : List ℝ := [65, 67, 76, 82, 85]

theorem reeyas_average_score : 
  (List.sum reeyas_scores) / (List.length reeyas_scores) = 75 := by
  sorry

end NUMINAMATH_CALUDE_reeyas_average_score_l1389_138978


namespace NUMINAMATH_CALUDE_van_helsing_werewolf_removal_percentage_l1389_138992

def vampire_price : ℕ := 5
def werewolf_price : ℕ := 10
def total_earnings : ℕ := 105
def werewolves_removed : ℕ := 8
def werewolf_vampire_ratio : ℕ := 4

theorem van_helsing_werewolf_removal_percentage 
  (vampires : ℕ) (werewolves : ℕ) : 
  vampire_price * (vampires / 2) + werewolf_price * werewolves_removed = total_earnings →
  werewolves = werewolf_vampire_ratio * vampires →
  (werewolves_removed : ℚ) / werewolves * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_van_helsing_werewolf_removal_percentage_l1389_138992


namespace NUMINAMATH_CALUDE_sam_yellow_marbles_l1389_138995

theorem sam_yellow_marbles (initial_yellow : ℝ) (received_yellow : ℝ) 
  (h1 : initial_yellow = 86.0) (h2 : received_yellow = 25.0) :
  initial_yellow + received_yellow = 111.0 := by
  sorry

end NUMINAMATH_CALUDE_sam_yellow_marbles_l1389_138995


namespace NUMINAMATH_CALUDE_first_student_guess_l1389_138952

/-- Represents the number of jellybeans guessed by each student -/
structure JellybeanGuesses where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The conditions of the jellybean guessing problem -/
def jellybean_problem (g : JellybeanGuesses) : Prop :=
  g.second = 8 * g.first ∧
  g.third = g.second - 200 ∧
  g.fourth = (g.first + g.second + g.third) / 3 + 25 ∧
  g.fourth = 525

/-- Theorem stating that the first student's guess is 100 jellybeans -/
theorem first_student_guess :
  ∀ g : JellybeanGuesses, jellybean_problem g → g.first = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_first_student_guess_l1389_138952


namespace NUMINAMATH_CALUDE_quadratic_maximum_quadratic_maximum_achieved_l1389_138967

theorem quadratic_maximum (r : ℝ) : -3 * r^2 + 30 * r + 24 ≤ 99 :=
sorry

theorem quadratic_maximum_achieved : ∃ r : ℝ, -3 * r^2 + 30 * r + 24 = 99 :=
sorry

end NUMINAMATH_CALUDE_quadratic_maximum_quadratic_maximum_achieved_l1389_138967


namespace NUMINAMATH_CALUDE_roses_cut_l1389_138940

theorem roses_cut (initial_roses final_roses : ℕ) (h1 : initial_roses = 3) (h2 : final_roses = 14) :
  final_roses - initial_roses = 11 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l1389_138940


namespace NUMINAMATH_CALUDE_exam_class_size_l1389_138996

/-- Represents a class of students with their exam marks. -/
structure ExamClass where
  totalStudents : ℕ
  averageMark : ℚ
  excludedStudents : ℕ
  excludedAverage : ℚ
  remainingAverage : ℚ

/-- Theorem stating the number of students in the class given the conditions. -/
theorem exam_class_size (c : ExamClass)
  (h1 : c.averageMark = 80)
  (h2 : c.excludedStudents = 5)
  (h3 : c.excludedAverage = 50)
  (h4 : c.remainingAverage = 90)
  (h5 : c.totalStudents * c.averageMark = 
        (c.totalStudents - c.excludedStudents) * c.remainingAverage + 
        c.excludedStudents * c.excludedAverage) :
  c.totalStudents = 20 := by
  sorry


end NUMINAMATH_CALUDE_exam_class_size_l1389_138996


namespace NUMINAMATH_CALUDE_same_group_probability_correct_l1389_138970

def card_count : ℕ := 20
def people_count : ℕ := 4
def drawn_card1 : ℕ := 5
def drawn_card2 : ℕ := 14

def same_group_probability : ℚ := 7/51

theorem same_group_probability_correct :
  let remaining_cards := card_count - 2
  let smaller_group_cases := (card_count - drawn_card2) * (card_count - drawn_card2 - 1) / 2
  let larger_group_cases := (drawn_card1 - 1) * (drawn_card1 - 2) / 2
  let favorable_outcomes := smaller_group_cases + larger_group_cases
  let total_outcomes := remaining_cards * (remaining_cards - 1) / 2
  (favorable_outcomes : ℚ) / total_outcomes = same_group_probability := by
  sorry

end NUMINAMATH_CALUDE_same_group_probability_correct_l1389_138970


namespace NUMINAMATH_CALUDE_cone_volume_contradiction_l1389_138947

theorem cone_volume_contradiction (base_area height volume : ℝ) : 
  base_area = 9 → height = 5 → volume = 45 → (1/3) * base_area * height ≠ volume :=
by
  sorry

end NUMINAMATH_CALUDE_cone_volume_contradiction_l1389_138947


namespace NUMINAMATH_CALUDE_true_discount_equals_bankers_gain_l1389_138987

/-- Present worth of a sum due -/
def present_worth : ℝ := 576

/-- Banker's gain -/
def bankers_gain : ℝ := 16

/-- True discount -/
def true_discount : ℝ := bankers_gain

theorem true_discount_equals_bankers_gain :
  true_discount = bankers_gain :=
by sorry

end NUMINAMATH_CALUDE_true_discount_equals_bankers_gain_l1389_138987


namespace NUMINAMATH_CALUDE_sum_first_50_digits_1001_l1389_138942

/-- The decimal expansion of 1/1001 -/
def decimalExpansion1001 : ℕ → ℕ
| n => match n % 6 with
  | 0 => 0
  | 1 => 0
  | 2 => 0
  | 3 => 9
  | 4 => 9
  | 5 => 9
  | _ => 0  -- This case should never occur

/-- The sum of the first n digits in the decimal expansion of 1/1001 -/
def sumFirstNDigits (n : ℕ) : ℕ :=
  (List.range n).map decimalExpansion1001 |> List.sum

/-- Theorem: The sum of the first 50 digits after the decimal point
    in the decimal expansion of 1/1001 is 216 -/
theorem sum_first_50_digits_1001 : sumFirstNDigits 50 = 216 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_50_digits_1001_l1389_138942


namespace NUMINAMATH_CALUDE_investment_growth_l1389_138986

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Theorem: Initial investment of $5000 at 10% p.a. for 2 years yields $6050.000000000001 -/
theorem investment_growth :
  let principal : ℝ := 5000
  let rate : ℝ := 0.1
  let time : ℕ := 2
  compound_interest principal rate time = 6050.000000000001 :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l1389_138986


namespace NUMINAMATH_CALUDE_isabel_bouquets_l1389_138936

def flowers_to_bouquets (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : ℕ :=
  (initial_flowers - wilted_flowers) / flowers_per_bouquet

theorem isabel_bouquets :
  flowers_to_bouquets 66 8 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_isabel_bouquets_l1389_138936


namespace NUMINAMATH_CALUDE_distance_to_origin_l1389_138911

theorem distance_to_origin (x y n : ℝ) : 
  y = 15 → 
  x = 2 + Real.sqrt 105 → 
  x > 2 → 
  n = Real.sqrt (x^2 + y^2) →
  n = Real.sqrt (334 + 4 * Real.sqrt 105) := by
sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1389_138911


namespace NUMINAMATH_CALUDE_convex_quadrilateral_count_lower_bound_l1389_138961

/-- A set of points in a plane -/
structure PointSet where
  n : ℕ
  points : Fin n → ℝ × ℝ

/-- Predicate to check if three points are collinear -/
def collinear (p q r : ℝ × ℝ) : Prop := sorry

/-- Count of convex quadrilaterals in a set of points -/
def convexQuadrilateralCount (s : PointSet) : ℕ := sorry

theorem convex_quadrilateral_count_lower_bound (s : PointSet) 
  (h1 : s.n > 4)
  (h2 : ∀ p q r, p ≠ q ∧ q ≠ r ∧ p ≠ r → ¬collinear (s.points p) (s.points q) (s.points r)) :
  convexQuadrilateralCount s ≥ (s.n - 3) * (s.n - 4) / 2 := by sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_count_lower_bound_l1389_138961


namespace NUMINAMATH_CALUDE_range_of_a_l1389_138998

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → a ∈ Set.Ioo (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1389_138998


namespace NUMINAMATH_CALUDE_strawberry_weight_difference_l1389_138939

theorem strawberry_weight_difference (marco_weight dad_weight total_weight : ℕ) 
  (h1 : marco_weight = 30)
  (h2 : total_weight = 47)
  (h3 : total_weight = marco_weight + dad_weight) :
  marco_weight - dad_weight = 13 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_weight_difference_l1389_138939


namespace NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l1389_138934

/-- 
Given a quadratic equation 5x^2 - 2x + 2 = 0, 
the coefficient of the linear term is -2 
-/
theorem linear_coefficient_of_quadratic (x : ℝ) : 
  (5 * x^2 - 2 * x + 2 = 0) → 
  (∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ b = -2) :=
by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l1389_138934


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l1389_138962

theorem sqrt_difference_equality : 3 * Real.sqrt 5 - Real.sqrt 20 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l1389_138962


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1389_138991

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 1 * a 5 = 16)
  (h_fourth : a 4 = 8) :
  a 5 = 16 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1389_138991


namespace NUMINAMATH_CALUDE_product_of_five_terms_l1389_138927

/-- A line passing through the origin with normal vector (3,1) -/
def line_l (x y : ℝ) : Prop := 3 * x + y = 0

/-- Sequence a_n where (a_{n+1}, a_n) lies on the line for all positive integers n -/
def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, line_l (a (n + 1)) (a n)

theorem product_of_five_terms (a : ℕ → ℝ) :
  sequence_property a → a 2 = 6 → a 1 * a 2 * a 3 * a 4 * a 5 = -32 := by
  sorry

end NUMINAMATH_CALUDE_product_of_five_terms_l1389_138927


namespace NUMINAMATH_CALUDE_ocean_area_scientific_notation_l1389_138976

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem ocean_area_scientific_notation :
  toScientificNotation 361000000 = ScientificNotation.mk 3.61 8 sorry := by
  sorry

end NUMINAMATH_CALUDE_ocean_area_scientific_notation_l1389_138976


namespace NUMINAMATH_CALUDE_pills_per_week_calculation_l1389_138981

/-- Calculates the number of pills taken in a week given the frequency of pill intake -/
def pills_per_week (hours_between_pills : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  (hours_per_day / hours_between_pills) * days_per_week

/-- Theorem stating that taking a pill every 6 hours results in 28 pills per week -/
theorem pills_per_week_calculation :
  pills_per_week 6 24 7 = 28 := by
  sorry

#eval pills_per_week 6 24 7

end NUMINAMATH_CALUDE_pills_per_week_calculation_l1389_138981


namespace NUMINAMATH_CALUDE_cosine_value_on_unit_circle_l1389_138937

theorem cosine_value_on_unit_circle (α : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = 1/2 ∧ y = Real.sqrt 3 / 2) →
  Real.cos α = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_cosine_value_on_unit_circle_l1389_138937


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1389_138994

theorem absolute_value_inequality (x : ℝ) :
  (|x - 1| + |x + 2| ≤ 5) ↔ (x ∈ Set.Icc (-3) 2) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1389_138994


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_is_sqrt3_l1389_138989

noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ := 
  let b : ℝ := Real.sqrt 8
  let c : ℝ := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_eccentricity_is_sqrt3 :
  ∃ (a : ℝ), a > 0 ∧ 
  (∃ (x y : ℝ), x = y ∧ x^2/a^2 - y^2/8 = 1) ∧
  (∃ (x1 x2 : ℝ), x1 * x2 = -8 ∧ 
    x1^2/a^2 - x1^2/8 = 1 ∧ 
    x2^2/a^2 - x2^2/8 = 1) ∧
  hyperbola_eccentricity a = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_is_sqrt3_l1389_138989


namespace NUMINAMATH_CALUDE_birthday_celebration_friends_l1389_138979

/-- The number of friends attending Paolo and Sevilla's birthday celebration -/
def num_friends : ℕ := sorry

/-- The total bill amount -/
def total_bill : ℕ := sorry

theorem birthday_celebration_friends :
  (total_bill = 12 * (num_friends + 2)) ∧
  (total_bill = 16 * num_friends) →
  num_friends = 6 := by sorry

end NUMINAMATH_CALUDE_birthday_celebration_friends_l1389_138979


namespace NUMINAMATH_CALUDE_perpendicular_line_proof_l1389_138977

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (-2, 3),
    prove that the line L2 with equation y = -2x - 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_proof (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y => y = -2 * x - 1
  let P : ℝ × ℝ := (-2, 3)
  (∀ x y, L1 x y ↔ y = (1/2) * x - 3/2) ∧
  (L2 P.1 P.2) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ ∧ L2 x₂ y₂ → (x₂ - x₁) * ((1/2) * (x₂ - x₁) + (y₂ - y₁)) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_proof_l1389_138977


namespace NUMINAMATH_CALUDE_valid_schedules_l1389_138972

/-- Represents the number of periods in a day -/
def total_periods : ℕ := 9

/-- Represents the number of morning periods -/
def morning_periods : ℕ := 5

/-- Represents the number of afternoon periods -/
def afternoon_periods : ℕ := 4

/-- Represents the number of classes to be taught -/
def classes_to_teach : ℕ := 3

/-- Calculates the number of ways to arrange n items taken k at a time -/
def arrangement (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- Calculates the number of prohibited arrangements in the morning -/
def morning_prohibited : ℕ := 3 * Nat.factorial classes_to_teach

/-- Calculates the number of prohibited arrangements in the afternoon -/
def afternoon_prohibited : ℕ := 2 * Nat.factorial classes_to_teach

/-- The main theorem stating the number of valid schedules -/
theorem valid_schedules : 
  arrangement total_periods classes_to_teach - morning_prohibited - afternoon_prohibited = 474 := by
  sorry


end NUMINAMATH_CALUDE_valid_schedules_l1389_138972


namespace NUMINAMATH_CALUDE_exponential_solution_l1389_138956

/-- A function satisfying f(x+1) = 2f(x) for all real x -/
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = 2 * f x

/-- The theorem stating that if f satisfies the functional equation,
    then f(x) = C * 2^x for some constant C -/
theorem exponential_solution (f : ℝ → ℝ) (h : functional_equation f) :
  ∃ C, ∀ x, f x = C * 2^x := by
  sorry

end NUMINAMATH_CALUDE_exponential_solution_l1389_138956


namespace NUMINAMATH_CALUDE_grandma_olga_grandchildren_l1389_138932

/-- Represents the number of grandchildren Grandma Olga has -/
def total_grandchildren (num_daughters num_sons : ℕ) 
                        (sons_per_daughter daughters_per_son : ℕ) : ℕ :=
  num_daughters * sons_per_daughter + num_sons * daughters_per_son

/-- Theorem stating that Grandma Olga has 33 grandchildren -/
theorem grandma_olga_grandchildren : 
  total_grandchildren 3 3 6 5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_grandma_olga_grandchildren_l1389_138932


namespace NUMINAMATH_CALUDE_max_vertices_1000_triangles_l1389_138963

/-- The maximum number of distinct points that can be vertices of 1000 triangles in a quadrilateral -/
def max_distinct_vertices (num_triangles : ℕ) (quadrilateral_angle_sum : ℕ) : ℕ :=
  let triangle_angle_sum := 180
  let total_angle_sum := num_triangles * triangle_angle_sum
  let excess_angle_sum := total_angle_sum - quadrilateral_angle_sum
  let side_vertices := excess_angle_sum / triangle_angle_sum
  let original_vertices := 4
  side_vertices + original_vertices

/-- Theorem stating that the maximum number of distinct vertices is 1002 -/
theorem max_vertices_1000_triangles :
  max_distinct_vertices 1000 360 = 1002 := by
  sorry

end NUMINAMATH_CALUDE_max_vertices_1000_triangles_l1389_138963


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1389_138928

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem: If a - b > 0 and ab < 0, then the point P(a,b) lies in the fourth quadrant -/
theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a - b > 0) (h2 : a * b < 0) :
  fourth_quadrant (Point.mk a b) := by
  sorry


end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1389_138928


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1389_138982

/-- The discriminant of a quadratic equation ax² + bx + c is equal to b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x² + (5 + 1/5)x + 1/5 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/5
def c : ℚ := 1/5

theorem quadratic_discriminant : discriminant a b c = 576/25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1389_138982


namespace NUMINAMATH_CALUDE_summer_camp_duration_l1389_138917

def summer_camp (n : ℕ) (k : ℕ) (d : ℕ) : Prop :=
  -- n is the number of participants
  -- k is the number of participants chosen each day
  -- d is the number of days
  n = 15 ∧ 
  k = 3 ∧
  Nat.choose n 2 = d * Nat.choose k 2

theorem summer_camp_duration : 
  ∃ d : ℕ, summer_camp 15 3 d ∧ d = 35 := by
  sorry

end NUMINAMATH_CALUDE_summer_camp_duration_l1389_138917


namespace NUMINAMATH_CALUDE_lcm_inequality_l1389_138988

theorem lcm_inequality (m n : ℕ) (h1 : 0 < m) (h2 : m < n) :
  Nat.lcm m n + Nat.lcm (m + 1) (n + 1) ≥ 2 * m * Real.sqrt n :=
by sorry

end NUMINAMATH_CALUDE_lcm_inequality_l1389_138988


namespace NUMINAMATH_CALUDE_inner_diagonal_sum_bound_l1389_138918

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  /-- Sum of the lengths of the diagonals -/
  diagonalSum : ℝ
  /-- Convexity condition -/
  convex : diagonalSum > 0

/-- Theorem: For any two convex quadrilaterals where one is inside the other,
    the sum of the diagonals of the inner quadrilateral is less than twice
    the sum of the diagonals of the outer quadrilateral -/
theorem inner_diagonal_sum_bound
  (outer inner : ConvexQuadrilateral)
  (h : inner.diagonalSum < outer.diagonalSum) :
  inner.diagonalSum < 2 * outer.diagonalSum :=
by
  sorry


end NUMINAMATH_CALUDE_inner_diagonal_sum_bound_l1389_138918


namespace NUMINAMATH_CALUDE_cat_direction_at_noon_l1389_138921

/-- Represents the direction the Cat is going -/
inductive Direction
  | Left  -- telling a tale
  | Right -- singing a song

/-- Represents the Cat's activities -/
structure CatActivity where
  tale_time : ℕ -- time to tell a tale in minutes
  song_time : ℕ -- time to sing a song in minutes

/-- Determines the Cat's direction after a given number of minutes -/
def cat_direction (activity : CatActivity) (minutes : ℕ) : Direction :=
  let cycle_time := activity.tale_time + activity.song_time
  let remaining_time := minutes % cycle_time
  if remaining_time < activity.tale_time then Direction.Left else Direction.Right

/-- The main theorem to prove -/
theorem cat_direction_at_noon (activity : CatActivity) 
    (h1 : activity.tale_time = 5)
    (h2 : activity.song_time = 4)
    (h3 : (12 - 10) * 60 = 120) : 
    cat_direction activity 120 = Direction.Left := by
  sorry


end NUMINAMATH_CALUDE_cat_direction_at_noon_l1389_138921


namespace NUMINAMATH_CALUDE_max_value_sum_reciprocals_l1389_138974

theorem max_value_sum_reciprocals (a b : ℝ) (h : a + b = 4) :
  (∃ x y : ℝ, x + y = 4 ∧ (1 / (x^2 + 1) + 1 / (y^2 + 1) ≤ 1 / (a^2 + 1) + 1 / (b^2 + 1))) ∧
  1 / (a^2 + 1) + 1 / (b^2 + 1) ≤ (Real.sqrt 5 + 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_sum_reciprocals_l1389_138974


namespace NUMINAMATH_CALUDE_square_minus_four_equals_negative_three_l1389_138965

theorem square_minus_four_equals_negative_three (a : ℤ) (h : a = -1) : a^2 - 4 = -3 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_four_equals_negative_three_l1389_138965


namespace NUMINAMATH_CALUDE_tangent_line_of_even_cubic_l1389_138933

/-- Given a real number a, if f(x) = x^3 + ax^2 + (a-3)x is an even function,
    then the equation of the tangent line to y = f(x) at (2, f(2)) is 9x - y - 16 = 0 -/
theorem tangent_line_of_even_cubic (a : ℝ) : 
  (∀ x, (x^3 + a*x^2 + (a-3)*x) = ((- x)^3 + a*(- x)^2 + (a-3)*(- x))) →
  ∃ m b, (m * 2 + b = 2^3 + a*2^2 + (a-3)*2) ∧ 
         (∀ x y, y = x^3 + a*x^2 + (a-3)*x → m*x - y - b = 0) ∧
         (m = 9 ∧ b = 16) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_of_even_cubic_l1389_138933
