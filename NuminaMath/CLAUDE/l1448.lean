import Mathlib

namespace NUMINAMATH_CALUDE_max_value_quadratic_l1448_144842

theorem max_value_quadratic (s t : ℝ) (h : t = 4) :
  ∃ (max : ℝ), max = 46 ∧ ∀ s, -2 * s^2 + 24 * s + 3 * t - 38 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1448_144842


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l1448_144880

/-- Rectangle represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The intersection area of two rectangles -/
def intersection_area (r1 r2 : Rectangle) : ℝ := sorry

/-- Checks if two integers are relatively prime -/
def are_relatively_prime (m n : ℕ) : Prop := sorry

theorem intersection_area_theorem (abcd aecf : Rectangle) 
  (h1 : abcd.width = 11 ∧ abcd.height = 3)
  (h2 : aecf.width = 9 ∧ aecf.height = 7) :
  ∃ (m n : ℕ), 
    (intersection_area abcd aecf = m / n) ∧ 
    (are_relatively_prime m n) ∧ 
    (m + n = 109) := by sorry

end NUMINAMATH_CALUDE_intersection_area_theorem_l1448_144880


namespace NUMINAMATH_CALUDE_problem_statement_l1448_144810

theorem problem_statement :
  (∀ x : ℝ, x > 0 → x + 4 / x ≥ 4) ∧
  ¬(∃ x₀ : ℝ, x₀ > 0 ∧ 2 * x₀ = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1448_144810


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1448_144864

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) : n > 0 → (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1448_144864


namespace NUMINAMATH_CALUDE_hiking_rate_ratio_l1448_144887

/-- Proves that the ratio of the hiking rate down to the rate up is 1.5 -/
theorem hiking_rate_ratio : 
  let rate_up : ℝ := 7 -- miles per day
  let days_up : ℝ := 2
  let distance_down : ℝ := 21 -- miles
  let days_down : ℝ := days_up -- same time for both routes
  let rate_down : ℝ := distance_down / days_down
  rate_down / rate_up = 1.5 := by
sorry


end NUMINAMATH_CALUDE_hiking_rate_ratio_l1448_144887


namespace NUMINAMATH_CALUDE_inequality_relation_l1448_144893

theorem inequality_relation (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_relation_l1448_144893


namespace NUMINAMATH_CALUDE_participation_difference_l1448_144817

def participants_2018 : ℕ := 150

def participants_2019 : ℕ := 2 * participants_2018 + 20

def participants_2020 : ℕ := participants_2019 / 2 - 40

def participants_2021 : ℕ := 30 + (participants_2018 - participants_2020)

theorem participation_difference : participants_2019 - participants_2020 = 200 := by
  sorry

end NUMINAMATH_CALUDE_participation_difference_l1448_144817


namespace NUMINAMATH_CALUDE_one_true_proposition_l1448_144823

theorem one_true_proposition :
  (∃! i : Fin 4, 
    (i = 0 ∧ (∀ x y : ℝ, ¬(x = -y) → x + y ≠ 0)) ∨
    (i = 1 ∧ (∀ a b : ℝ, a^2 > b^2 → a > b)) ∨
    (i = 2 ∧ (∃ x : ℝ, x ≤ -3 ∧ x^2 - x - 6 ≤ 0)) ∨
    (i = 3 ∧ (∀ a b : ℝ, Irrational a ∧ Irrational b → Irrational (a^b)))) :=
sorry

end NUMINAMATH_CALUDE_one_true_proposition_l1448_144823


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1448_144853

-- Define the sets p and q
def p : Set ℝ := {x | |2*x - 3| > 1}
def q : Set ℝ := {x | x^2 + x - 6 > 0}

-- Define what it means for one set to be a sufficient condition for another
def is_sufficient_condition (A B : Set ℝ) : Prop := B ⊆ A

-- Define what it means for one set to be a necessary condition for another
def is_necessary_condition (A B : Set ℝ) : Prop := A ⊆ B

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  is_sufficient_condition (Set.univ \ p) (Set.univ \ q) ∧
  ¬ is_necessary_condition (Set.univ \ p) (Set.univ \ q) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1448_144853


namespace NUMINAMATH_CALUDE_cost_of_500_sheets_l1448_144815

/-- The cost in dollars of a given number of sheets of paper. -/
def paper_cost (sheets : ℕ) : ℚ :=
  (sheets * 2 : ℚ) / 100

/-- Theorem stating that 500 sheets of paper cost $10.00. -/
theorem cost_of_500_sheets :
  paper_cost 500 = 10 := by sorry

end NUMINAMATH_CALUDE_cost_of_500_sheets_l1448_144815


namespace NUMINAMATH_CALUDE_inequality_proof_l1448_144888

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1448_144888


namespace NUMINAMATH_CALUDE_min_average_of_four_integers_l1448_144894

theorem min_average_of_four_integers (a b c d : ℕ+) 
  (ha : a = 3 * b)
  (hc : c = b + 2)
  (hd : d ≥ 2) :
  (a + b + c + d : ℚ) / 4 ≥ 9/4 :=
sorry

end NUMINAMATH_CALUDE_min_average_of_four_integers_l1448_144894


namespace NUMINAMATH_CALUDE_minimal_value_of_f_l1448_144877

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem minimal_value_of_f :
  ∃ (x_min : ℝ), f x_min = Real.exp (-1) ∧ ∀ (x : ℝ), f x ≥ Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_minimal_value_of_f_l1448_144877


namespace NUMINAMATH_CALUDE_smallest_N_property_l1448_144873

/-- The smallest natural number N such that N × 999 consists entirely of the digit seven in its decimal representation -/
def smallest_N : ℕ := 778556334111889667445223

/-- Predicate to check if a natural number consists entirely of the digit seven -/
def all_sevens (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 7

theorem smallest_N_property :
  (all_sevens (smallest_N * 999)) ∧
  (∀ m : ℕ, m < smallest_N → ¬(all_sevens (m * 999))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_N_property_l1448_144873


namespace NUMINAMATH_CALUDE_charity_event_equation_l1448_144879

theorem charity_event_equation :
  ∀ x : ℕ,
  (x + (12 - x) = 12) →  -- Total number of banknotes is 12
  (x ≤ 12) →             -- Ensure x doesn't exceed total banknotes
  (x + 5 * (12 - x) = 48) -- The equation correctly represents the problem
  :=
by
  sorry

end NUMINAMATH_CALUDE_charity_event_equation_l1448_144879


namespace NUMINAMATH_CALUDE_investment_problem_l1448_144822

/-- The solution to Susie Q's investment problem -/
theorem investment_problem (total_investment : ℝ) (interest_rate1 : ℝ) (interest_rate2 : ℝ) 
  (years : ℕ) (final_amount : ℝ) (investment1 : ℝ) :
  total_investment = 1500 →
  interest_rate1 = 0.04 →
  interest_rate2 = 0.06 →
  years = 2 →
  final_amount = 1700.02 →
  investment1 * (1 + interest_rate1) ^ years + 
    (total_investment - investment1) * (1 + interest_rate2) ^ years = final_amount →
  investment1 = 348.095 := by
    sorry

end NUMINAMATH_CALUDE_investment_problem_l1448_144822


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l1448_144857

-- Define the quadratic polynomial
def p (x : ℚ) : ℚ := (13/6) * x^2 - (7/6) * x + 2

-- State the theorem
theorem quadratic_polynomial_satisfies_conditions :
  p 1 = 3 ∧ p 0 = 2 ∧ p 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l1448_144857


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_roots_l1448_144805

theorem fourth_degree_polynomial_roots : ∃ (a b c d : ℝ),
  (a = 1 - Real.sqrt 3) ∧
  (b = 1 + Real.sqrt 3) ∧
  (c = (1 - Real.sqrt 13) / 2) ∧
  (d = (1 + Real.sqrt 13) / 2) ∧
  (∀ x : ℝ, x^4 - 3*x^3 + 3*x^2 - x - 6 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_fourth_degree_polynomial_roots_l1448_144805


namespace NUMINAMATH_CALUDE_ratio_fraction_equality_l1448_144891

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_equality_l1448_144891


namespace NUMINAMATH_CALUDE_llama_breeding_problem_llama_breeding_solution_l1448_144868

theorem llama_breeding_problem (pregnant_llamas : ℕ) (twin_pregnancies : ℕ) 
  (traded_calves : ℕ) (new_adults : ℕ) (final_herd : ℕ) : ℕ :=
  let single_pregnancies := pregnant_llamas - twin_pregnancies
  let total_calves := twin_pregnancies * 2 + single_pregnancies * 1
  let remaining_calves := total_calves - traded_calves
  let pre_sale_herd := final_herd / (2/3)
  let pre_sale_adults := pre_sale_herd - remaining_calves - new_adults
  let original_adults := pre_sale_adults - new_adults
  total_calves

theorem llama_breeding_solution : 
  llama_breeding_problem 9 5 8 2 18 = 14 := by sorry

end NUMINAMATH_CALUDE_llama_breeding_problem_llama_breeding_solution_l1448_144868


namespace NUMINAMATH_CALUDE_ratio_w_y_is_15_4_l1448_144881

-- Define the ratios as fractions
def ratio_w_x : ℚ := 5 / 4
def ratio_y_z : ℚ := 5 / 3
def ratio_z_x : ℚ := 1 / 5

-- Theorem statement
theorem ratio_w_y_is_15_4 :
  let ratio_w_y := ratio_w_x / (ratio_y_z * ratio_z_x)
  ratio_w_y = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_w_y_is_15_4_l1448_144881


namespace NUMINAMATH_CALUDE_exists_x_squared_minus_one_nonnegative_l1448_144892

theorem exists_x_squared_minus_one_nonnegative :
  ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ x^2 - 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_squared_minus_one_nonnegative_l1448_144892


namespace NUMINAMATH_CALUDE_A_inter_B_a_upper_bound_a_sufficient_l1448_144820

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 < x ∧ x ≤ 5}
def B : Set ℝ := {x | (2*x - 1)/(x - 3) > 0}
def C (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 4*a - 3}

-- Theorem for A ∩ B
theorem A_inter_B : A ∩ B = {x : ℝ | 3 < x ∧ x ≤ 5} := by sorry

-- Theorem for the upper bound of a
theorem a_upper_bound (a : ℝ) (h : C a ∪ A = A) : a ≤ 2 := by sorry

-- Theorem for the sufficiency of a ≤ 2
theorem a_sufficient (a : ℝ) (h : a ≤ 2) : C a ∪ A = A := by sorry

end NUMINAMATH_CALUDE_A_inter_B_a_upper_bound_a_sufficient_l1448_144820


namespace NUMINAMATH_CALUDE_unique_function_characterization_l1448_144804

theorem unique_function_characterization :
  ∀ f : ℕ+ → ℕ+,
  (∀ x y : ℕ+, x < y → f x < f y) →
  (∀ x y : ℕ+, f (y * f x) = x^2 * f (x * y)) →
  ∀ x : ℕ+, f x = x^2 := by
sorry

end NUMINAMATH_CALUDE_unique_function_characterization_l1448_144804


namespace NUMINAMATH_CALUDE_chalk_problem_l1448_144849

theorem chalk_problem (total_people : ℕ) (added_chalk : ℕ) (lost_chalk : ℕ) (final_per_person : ℚ) :
  total_people = 11 →
  added_chalk = 28 →
  lost_chalk = 4 →
  final_per_person = 5.5 →
  ∃ (original_chalk : ℕ), original_chalk = 37 ∧ 
    (↑original_chalk - ↑lost_chalk + ↑added_chalk : ℚ) = ↑total_people * final_per_person :=
by sorry

end NUMINAMATH_CALUDE_chalk_problem_l1448_144849


namespace NUMINAMATH_CALUDE_school_election_l1448_144869

theorem school_election (total_students : ℕ) : total_students = 2000 :=
  let voter_percentage : ℚ := 25 / 100
  let winner_vote_percentage : ℚ := 55 / 100
  let loser_vote_percentage : ℚ := 1 - winner_vote_percentage
  let vote_difference : ℕ := 50
  have h1 : (winner_vote_percentage * voter_percentage * total_students : ℚ) = 
            (loser_vote_percentage * voter_percentage * total_students + vote_difference : ℚ) := by sorry
  sorry

end NUMINAMATH_CALUDE_school_election_l1448_144869


namespace NUMINAMATH_CALUDE_correct_calculation_l1448_144803

theorem correct_calculation (x : ℤ) (h : x - 48 = 52) : x + 48 = 148 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1448_144803


namespace NUMINAMATH_CALUDE_min_value_expression_l1448_144859

theorem min_value_expression (m n : ℝ) (h : m > 2 * n) :
  m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) ≥ 6 ∧ 
  (m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) = 6 ↔ m - 2 * n = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1448_144859


namespace NUMINAMATH_CALUDE_water_added_to_container_l1448_144899

theorem water_added_to_container (capacity : ℝ) (initial_fullness : ℝ) (final_fullness : ℝ) : 
  capacity = 120 →
  initial_fullness = 0.3 →
  final_fullness = 0.75 →
  (final_fullness - initial_fullness) * capacity = 54 := by
sorry

end NUMINAMATH_CALUDE_water_added_to_container_l1448_144899


namespace NUMINAMATH_CALUDE_pear_sales_l1448_144866

theorem pear_sales (morning_sales afternoon_sales total_sales : ℕ) :
  morning_sales = 120 →
  afternoon_sales = 2 * morning_sales →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 360 →
  afternoon_sales = 240 := by
sorry

end NUMINAMATH_CALUDE_pear_sales_l1448_144866


namespace NUMINAMATH_CALUDE_smallest_result_l1448_144829

def S : Set Int := {-10, -4, 0, 2, 7}

theorem smallest_result (x y : Int) (hx : x ∈ S) (hy : y ∈ S) :
  (x * y ≥ -70 ∧ x + y ≥ -70) ∧ ∃ a b : Int, a ∈ S ∧ b ∈ S ∧ (a * b = -70 ∨ a + b = -70) :=
by sorry

end NUMINAMATH_CALUDE_smallest_result_l1448_144829


namespace NUMINAMATH_CALUDE_cuboid_height_l1448_144897

/-- The surface area of a rectangular cuboid given its length, width, and height. -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: For a rectangular cuboid with surface area 442 cm², width 7 cm, and length 8 cm, the height is 11 cm. -/
theorem cuboid_height (A l w h : ℝ) 
  (h_area : A = 442)
  (h_width : w = 7)
  (h_length : l = 8)
  (h_surface : surface_area l w h = A) : h = 11 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_l1448_144897


namespace NUMINAMATH_CALUDE_square_area_ratio_l1448_144884

theorem square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : a^2 / b^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1448_144884


namespace NUMINAMATH_CALUDE_tomato_plants_problem_l1448_144847

theorem tomato_plants_problem (plant1 plant2 plant3 plant4 : ℕ) : 
  plant1 = 8 →
  plant2 = plant1 + 4 →
  plant3 = 3 * (plant1 + plant2) →
  plant4 = 3 * (plant1 + plant2) →
  plant1 + plant2 + plant3 + plant4 = 140 →
  plant2 - plant1 = 4 := by
sorry

end NUMINAMATH_CALUDE_tomato_plants_problem_l1448_144847


namespace NUMINAMATH_CALUDE_blue_stamp_price_l1448_144867

/-- Given a collection of stamps and their prices, prove the price of blue stamps --/
theorem blue_stamp_price
  (red_count : ℕ)
  (blue_count : ℕ)
  (yellow_count : ℕ)
  (red_price : ℚ)
  (yellow_price : ℚ)
  (total_earnings : ℚ)
  (h1 : red_count = 20)
  (h2 : blue_count = 80)
  (h3 : yellow_count = 7)
  (h4 : red_price = 11/10)
  (h5 : yellow_price = 2)
  (h6 : total_earnings = 100) :
  (total_earnings - red_count * red_price - yellow_count * yellow_price) / blue_count = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_blue_stamp_price_l1448_144867


namespace NUMINAMATH_CALUDE_ladder_problem_l1448_144858

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ base : ℝ, base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l1448_144858


namespace NUMINAMATH_CALUDE_parabola_properties_l1448_144896

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem parabola_properties (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hc : c > 1) 
  (h_point : parabola a b c 2 = 0) 
  (h_symmetry : -b / (2 * a) = 1/2) :
  abc < 0 ∧ 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ parabola a b c x₁ = a ∧ parabola a b c x₂ = a) ∧
  a < -1/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_properties_l1448_144896


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_cubes_l1448_144819

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed (l w h : ℕ) : ℕ :=
  l + w + h
  - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l)
  + Nat.gcd l (Nat.gcd w h)

/-- Theorem stating the number of cubes passed through by the internal diagonal -/
theorem rectangular_solid_diagonal_cubes :
  cubes_passed 150 324 375 = 768 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_cubes_l1448_144819


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1448_144875

theorem fraction_sum_equals_decimal : (1 : ℚ) / 20 + 2 / 10 + 4 / 40 = (35 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1448_144875


namespace NUMINAMATH_CALUDE_anniversary_count_l1448_144839

def founding_year : Nat := 1949
def current_year : Nat := 2015

theorem anniversary_count :
  current_year - founding_year = 66 := by sorry

end NUMINAMATH_CALUDE_anniversary_count_l1448_144839


namespace NUMINAMATH_CALUDE_missing_digits_sum_l1448_144898

/-- Given an addition problem 7□8 + 2182 = 863□91 where □ represents a single digit (0-9),
    the sum of the two missing digits is 7. -/
theorem missing_digits_sum (d1 d2 : Nat) : 
  d1 ≤ 9 → d2 ≤ 9 → 
  708 + d1 * 10 + 2182 = 86300 + d2 * 10 + 91 →
  d1 + d2 = 7 := by
sorry

end NUMINAMATH_CALUDE_missing_digits_sum_l1448_144898


namespace NUMINAMATH_CALUDE_system_equation_solution_l1448_144855

theorem system_equation_solution (x y c d : ℝ) (h1 : 4 * x + 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l1448_144855


namespace NUMINAMATH_CALUDE_max_value_2a_plus_b_l1448_144865

theorem max_value_2a_plus_b (a b : ℝ) (h : 4 * a^2 + b^2 + a * b = 1) :
  2 * a + b ≤ 2 * Real.sqrt 10 / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_2a_plus_b_l1448_144865


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1448_144802

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (x + 2)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a) →
  a₁ + a₂ + a₃ + a₄ = 65 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1448_144802


namespace NUMINAMATH_CALUDE_continuous_of_strictly_increasing_and_continuous_compose_l1448_144841

/-- Given a strictly increasing function f: ℝ → ℝ where f ∘ f is continuous, f is continuous. -/
theorem continuous_of_strictly_increasing_and_continuous_compose (f : ℝ → ℝ)
  (h_increasing : StrictMono f) (h_continuous_compose : Continuous (f ∘ f)) :
  Continuous f := by
  sorry

end NUMINAMATH_CALUDE_continuous_of_strictly_increasing_and_continuous_compose_l1448_144841


namespace NUMINAMATH_CALUDE_equation_solutions_l1448_144885

theorem equation_solutions : ∃! (s : Set ℝ), 
  (∀ x ∈ s, (x - 4)^4 + (x - 6)^4 = 16) ∧
  (s = {4, 6}) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1448_144885


namespace NUMINAMATH_CALUDE_base_dimensions_of_divided_volume_l1448_144890

/-- Given a volume of 120 cubic cubits divided into 10 parts, each with a height of 1 cubit,
    and a rectangular base with sides in the ratio 1:3/4, prove that the dimensions of the base
    are 4 cubits and 3 cubits. -/
theorem base_dimensions_of_divided_volume (total_volume : ℝ) (num_parts : ℕ) 
    (part_height : ℝ) (base_ratio : ℝ) :
  total_volume = 120 →
  num_parts = 10 →
  part_height = 1 →
  base_ratio = 3/4 →
  ∃ (a b : ℝ), a = 4 ∧ b = 3 ∧
    a * b * part_height * num_parts = total_volume ∧
    b / a = base_ratio :=
by sorry

end NUMINAMATH_CALUDE_base_dimensions_of_divided_volume_l1448_144890


namespace NUMINAMATH_CALUDE_equation_solution_l1448_144816

theorem equation_solution : ∃ x : ℝ, 3 * x - 6 = |(-25 * 2 + 50)| ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1448_144816


namespace NUMINAMATH_CALUDE_largest_divisor_of_2n3_minus_2n_l1448_144835

theorem largest_divisor_of_2n3_minus_2n (n : ℤ) : 
  (∃ (k : ℤ), 2 * n^3 - 2 * n = 12 * k) ∧ 
  (∀ (m : ℤ), m > 12 → ∃ (l : ℤ), 2 * l^3 - 2 * l ≠ m * (2 * l^3 - 2 * l) / m) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_2n3_minus_2n_l1448_144835


namespace NUMINAMATH_CALUDE_pet_shop_dogs_count_l1448_144826

/-- Given a pet shop with dogs, cats, and bunnies in stock, this theorem proves
    the number of dogs based on the given ratio and total count of dogs and bunnies. -/
theorem pet_shop_dogs_count
  (ratio_dogs : ℕ)
  (ratio_cats : ℕ)
  (ratio_bunnies : ℕ)
  (total_dogs_and_bunnies : ℕ)
  (h_ratio : ratio_dogs = 3 ∧ ratio_cats = 5 ∧ ratio_bunnies = 9)
  (h_total : total_dogs_and_bunnies = 204) :
  (ratio_dogs * total_dogs_and_bunnies) / (ratio_dogs + ratio_bunnies) = 51 :=
by
  sorry


end NUMINAMATH_CALUDE_pet_shop_dogs_count_l1448_144826


namespace NUMINAMATH_CALUDE_triangle_properties_l1448_144844

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem triangle_properties (t : Triangle) :
  (Real.cos t.C + (Real.cos t.A - Real.sqrt 3 * Real.sin t.A) * Real.cos t.B = 0) →
  (t.a + t.c = 1) →
  (t.B = π / 3 ∧ 1 / 2 ≤ t.b ∧ t.b < 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1448_144844


namespace NUMINAMATH_CALUDE_average_salary_calculation_l1448_144878

/-- Average salary calculation problem -/
theorem average_salary_calculation (n : ℕ) 
  (avg_all : ℕ) 
  (avg_feb_may : ℕ) 
  (salary_may : ℕ) 
  (salary_jan : ℕ) 
  (h1 : avg_all = 8000)
  (h2 : avg_feb_may = 8700)
  (h3 : salary_may = 6500)
  (h4 : salary_jan = 3700) :
  (salary_jan + (4 * avg_feb_may - salary_may)) / 4 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_calculation_l1448_144878


namespace NUMINAMATH_CALUDE_overall_gain_percentage_is_10_51_l1448_144851

/-- Represents a transaction with quantity, buy price, and sell price -/
structure Transaction where
  quantity : ℕ
  buyPrice : ℚ
  sellPrice : ℚ

/-- Calculates the profit or loss for a single transaction -/
def transactionProfit (t : Transaction) : ℚ :=
  t.quantity * (t.sellPrice - t.buyPrice)

/-- Calculates the cost for a single transaction -/
def transactionCost (t : Transaction) : ℚ :=
  t.quantity * t.buyPrice

/-- Calculates the overall gain percentage for a list of transactions -/
def overallGainPercentage (transactions : List Transaction) : ℚ :=
  let totalProfit := (transactions.map transactionProfit).sum
  let totalCost := (transactions.map transactionCost).sum
  totalProfit / totalCost * 100

/-- The main theorem stating that the overall gain percentage for the given transactions is 10.51% -/
theorem overall_gain_percentage_is_10_51 :
  let transactions := [
    ⟨10, 8, 10⟩,
    ⟨7, 15, 18⟩,
    ⟨5, 22, 20⟩
  ]
  overallGainPercentage transactions = 10.51 := by
  sorry

end NUMINAMATH_CALUDE_overall_gain_percentage_is_10_51_l1448_144851


namespace NUMINAMATH_CALUDE_binomial_sum_l1448_144818

theorem binomial_sum : (7 : ℕ).choose 2 + (6 : ℕ).choose 4 = 36 := by sorry

end NUMINAMATH_CALUDE_binomial_sum_l1448_144818


namespace NUMINAMATH_CALUDE_kaydence_age_l1448_144806

/-- The age of Kaydence given the ages of her family members and the total family age -/
theorem kaydence_age (total_age father_age mother_age brother_age sister_age : ℕ)
  (h_total : total_age = 200)
  (h_father : father_age = 60)
  (h_mother : mother_age = father_age - 2)
  (h_brother : brother_age = father_age / 2)
  (h_sister : sister_age = 40) :
  total_age - (father_age + mother_age + brother_age + sister_age) = 12 := by
  sorry

end NUMINAMATH_CALUDE_kaydence_age_l1448_144806


namespace NUMINAMATH_CALUDE_concert_drive_distance_l1448_144889

/-- Calculates the remaining distance to drive given the total distance and the distance already driven. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem stating that for a total distance of 78 miles and a driven distance of 32 miles, 
    the remaining distance is 46 miles. -/
theorem concert_drive_distance : remaining_distance 78 32 = 46 := by
  sorry

end NUMINAMATH_CALUDE_concert_drive_distance_l1448_144889


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1448_144883

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 5) * (x - 6) = -34 + k * x) ↔ 
  (k = -13 + 4 * Real.sqrt 3 ∨ k = -13 - 4 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1448_144883


namespace NUMINAMATH_CALUDE_inequality_implication_l1448_144895

theorem inequality_implication (a b : ℝ) (h : a > b) : -6*a < -6*b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1448_144895


namespace NUMINAMATH_CALUDE_beach_ball_surface_area_l1448_144830

theorem beach_ball_surface_area (d : ℝ) (h : d = 15) :
  4 * Real.pi * (d / 2)^2 = 225 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_beach_ball_surface_area_l1448_144830


namespace NUMINAMATH_CALUDE_klinker_age_problem_l1448_144809

/-- The age difference between Mr. Klinker and his daughter remains constant -/
theorem klinker_age_problem (klinker_age : ℕ) (daughter_age : ℕ) (years : ℕ) :
  klinker_age = 47 →
  daughter_age = 13 →
  klinker_age + years = 3 * (daughter_age + years) →
  years = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_klinker_age_problem_l1448_144809


namespace NUMINAMATH_CALUDE_grid_30_8_uses_598_toothpicks_l1448_144801

/-- Calculates the total number of toothpicks in a reinforced rectangular grid. -/
def toothpicks_in_grid (height : ℕ) (width : ℕ) : ℕ :=
  let internal_horizontal := (height + 1) * width
  let internal_vertical := (width + 1) * height
  let external_horizontal := 2 * width
  let external_vertical := 2 * (height + 2)
  internal_horizontal + internal_vertical + external_horizontal + external_vertical

/-- Theorem stating that a reinforced rectangular grid of 30x8 uses 598 toothpicks. -/
theorem grid_30_8_uses_598_toothpicks :
  toothpicks_in_grid 30 8 = 598 := by
  sorry

#eval toothpicks_in_grid 30 8

end NUMINAMATH_CALUDE_grid_30_8_uses_598_toothpicks_l1448_144801


namespace NUMINAMATH_CALUDE_f_of_x_plus_one_l1448_144821

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_of_x_plus_one (x : ℝ) : f (x + 1) = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_x_plus_one_l1448_144821


namespace NUMINAMATH_CALUDE_figure_to_square_possible_l1448_144827

/-- Represents a point on a grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a triangle on a grid --/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Represents a figure on a grid --/
structure GridFigure where
  points : List GridPoint

/-- Function to calculate the area of a grid figure --/
def calculateArea (figure : GridFigure) : ℕ :=
  sorry

/-- Function to check if a list of triangles forms a square --/
def formsSquare (triangles : List GridTriangle) : Prop :=
  sorry

/-- The main theorem --/
theorem figure_to_square_possible (figure : GridFigure) : 
  ∃ (triangles : List GridTriangle), 
    triangles.length = 5 ∧ 
    (calculateArea figure = calculateArea (GridFigure.mk (triangles.bind (λ t => [t.p1, t.p2, t.p3]))) ∧
    formsSquare triangles) :=
  sorry

end NUMINAMATH_CALUDE_figure_to_square_possible_l1448_144827


namespace NUMINAMATH_CALUDE_min_dot_product_ellipse_l1448_144824

/-- The minimum dot product of OP and FP for an ellipse -/
theorem min_dot_product_ellipse :
  ∀ (x y : ℝ), 
  x^2 / 9 + y^2 / 8 = 1 →
  ∃ (min : ℝ), 
  (∀ (x' y' : ℝ), x'^2 / 9 + y'^2 / 8 = 1 → 
    x'^2 + x' + y'^2 ≥ min) ∧
  min = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_ellipse_l1448_144824


namespace NUMINAMATH_CALUDE_triangle_sine_theorem_l1448_144854

/-- Given a triangle with area 30, a side of length 12, and a median to that side of length 8,
    the sine of the angle between the side and the median is 5/8. -/
theorem triangle_sine_theorem (A : ℝ) (a m θ : ℝ) 
    (h_area : A = 30)
    (h_side : a = 12)
    (h_median : m = 8)
    (h_angle : 0 < θ ∧ θ < π / 2)
    (h_triangle_area : A = 1/2 * a * m * Real.sin θ) : 
  Real.sin θ = 5/8 := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_theorem_l1448_144854


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1448_144811

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℝ) (d q : ℝ) :
  a 1 = 2 →
  d ≠ 0 →
  (∀ n, a (n + 1) = a n + d) →
  a 3 = a 1 * q →
  a 11 = a 1 * q^2 →
  q = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1448_144811


namespace NUMINAMATH_CALUDE_m_equality_l1448_144846

theorem m_equality (M : ℕ) (h : M^2 = 16^81 * 81^16) : M = 6^64 * 2^260 := by
  sorry

end NUMINAMATH_CALUDE_m_equality_l1448_144846


namespace NUMINAMATH_CALUDE_milk_level_lowered_l1448_144836

/-- Proves that removing 5250 gallons of milk from a 56ft by 25ft rectangular box
    lowers the milk level by 6 inches. -/
theorem milk_level_lowered (box_length box_width : ℝ)
                            (milk_volume_gallons : ℝ)
                            (cubic_feet_to_gallons : ℝ)
                            (inches_per_foot : ℝ) :
  box_length = 56 →
  box_width = 25 →
  milk_volume_gallons = 5250 →
  cubic_feet_to_gallons = 7.5 →
  inches_per_foot = 12 →
  (milk_volume_gallons / cubic_feet_to_gallons) /
  (box_length * box_width) * inches_per_foot = 6 :=
by sorry

end NUMINAMATH_CALUDE_milk_level_lowered_l1448_144836


namespace NUMINAMATH_CALUDE_ln_geq_num_prime_factors_ln2_l1448_144852

/-- The number of prime factors of a positive integer -/
def num_prime_factors (n : ℕ+) : ℕ := sorry

/-- For any positive integer n, ln n ≥ p(n) ln 2, where p(n) is the number of prime factors of n -/
theorem ln_geq_num_prime_factors_ln2 (n : ℕ+) : Real.log n ≥ (num_prime_factors n : ℝ) * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ln_geq_num_prime_factors_ln2_l1448_144852


namespace NUMINAMATH_CALUDE_smallest_ending_in_9_divisible_by_13_l1448_144861

theorem smallest_ending_in_9_divisible_by_13 : 
  ∃ (n : ℕ), n > 0 ∧ n % 10 = 9 ∧ n % 13 = 0 ∧ n = 69 ∧ 
  ∀ (m : ℕ), m > 0 → m % 10 = 9 → m % 13 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_ending_in_9_divisible_by_13_l1448_144861


namespace NUMINAMATH_CALUDE_total_spent_pinball_l1448_144845

/-- The amount of money in dollars represented by a half-dollar coin -/
def half_dollar_value : ℚ := 0.5

/-- The number of half-dollars Joan spent on Wednesday -/
def wednesday_spent : ℕ := 4

/-- The number of half-dollars Joan spent on Thursday -/
def thursday_spent : ℕ := 14

/-- The number of half-dollars Joan spent on Friday -/
def friday_spent : ℕ := 8

/-- Theorem: The total amount Joan spent playing pinball over three days is $13.00 -/
theorem total_spent_pinball :
  (wednesday_spent + thursday_spent + friday_spent : ℚ) * half_dollar_value = 13 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_pinball_l1448_144845


namespace NUMINAMATH_CALUDE_units_digit_of_8421_to_1287_l1448_144800

theorem units_digit_of_8421_to_1287 : ∃ n : ℕ, 8421^1287 = 10 * n + 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_8421_to_1287_l1448_144800


namespace NUMINAMATH_CALUDE_parabola_focus_and_directrix_l1448_144876

/-- A parabola is defined by the equation x² = 8y -/
def is_parabola (x y : ℝ) : Prop := x^2 = 8*y

/-- The focus of a parabola is a point on its axis of symmetry -/
def is_focus (f : ℝ × ℝ) (x y : ℝ) : Prop :=
  is_parabola x y → f = (0, 2)

/-- The directrix of a parabola is a line perpendicular to its axis of symmetry -/
def is_directrix (y : ℝ) : Prop :=
  ∀ x, is_parabola x y → y = -2

/-- Theorem: For the parabola x² = 8y, the focus is at (0, 2) and the directrix is y = -2 -/
theorem parabola_focus_and_directrix :
  (∀ x y, is_focus (0, 2) x y) ∧ is_directrix (-2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_and_directrix_l1448_144876


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1448_144828

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (5 + 7 * i) / (3 - 4 * i + 2 * i^2) = -23/17 + (27/17) * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1448_144828


namespace NUMINAMATH_CALUDE_function_equation_solution_l1448_144831

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x * y * f y) : 
  ∀ x : ℝ, f x = 0 ∨ f x = x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l1448_144831


namespace NUMINAMATH_CALUDE_lambda_5_lower_bound_l1448_144886

/-- The ratio of the longest distance to the shortest distance for n points in a plane -/
def lambda (n : ℕ) : ℝ := sorry

/-- Theorem: For 5 points in a plane, the ratio of the longest distance to the shortest distance
    is greater than or equal to 2 sin 54° -/
theorem lambda_5_lower_bound : lambda 5 ≥ 2 * Real.sin (54 * π / 180) := by sorry

end NUMINAMATH_CALUDE_lambda_5_lower_bound_l1448_144886


namespace NUMINAMATH_CALUDE_unique_real_solution_iff_a_in_range_l1448_144863

/-- The equation x^3 - ax^2 - 4ax + 4a^2 - 1 = 0 has exactly one real solution in x if and only if a ∈ (-∞, 3/4). -/
theorem unique_real_solution_iff_a_in_range (a : ℝ) : 
  (∃! x : ℝ, x^3 - a*x^2 - 4*a*x + 4*a^2 - 1 = 0) ↔ a < 3/4 :=
sorry

end NUMINAMATH_CALUDE_unique_real_solution_iff_a_in_range_l1448_144863


namespace NUMINAMATH_CALUDE_shirt_cost_calculation_l1448_144808

/-- The amount Sandy spent on clothes -/
def total_spent : ℚ := 33.56

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℚ := 13.99

/-- The amount Sandy spent on a jacket -/
def jacket_cost : ℚ := 7.43

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℚ := total_spent - shorts_cost - jacket_cost

theorem shirt_cost_calculation :
  shirt_cost = 12.14 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_calculation_l1448_144808


namespace NUMINAMATH_CALUDE_first_character_lines_l1448_144871

/-- Represents the number of lines for each character in Jerry's skit script. -/
structure ScriptLines where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Defines the conditions for Jerry's skit script. -/
def valid_script (s : ScriptLines) : Prop :=
  s.first = s.second + 8 ∧
  s.third = 2 ∧
  s.second = 3 * s.third + 6

/-- Theorem stating that the first character has 20 lines in a valid script. -/
theorem first_character_lines (s : ScriptLines) (h : valid_script s) : s.first = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_character_lines_l1448_144871


namespace NUMINAMATH_CALUDE_repeating_block_11_13_l1448_144840

def decimal_expansion (n d : ℕ) : List ℕ :=
  sorry

def is_repeating_block (l : List ℕ) (block : List ℕ) : Prop :=
  sorry

theorem repeating_block_11_13 :
  ∃ (block : List ℕ),
    block.length = 6 ∧
    is_repeating_block (decimal_expansion 11 13) block ∧
    ∀ (smaller_block : List ℕ),
      smaller_block.length < 6 →
      ¬ is_repeating_block (decimal_expansion 11 13) smaller_block :=
by sorry

end NUMINAMATH_CALUDE_repeating_block_11_13_l1448_144840


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1448_144850

theorem polynomial_remainder (x : ℝ) : (x^15 + 1) % (x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1448_144850


namespace NUMINAMATH_CALUDE_monotonic_range_a_l1448_144825

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

def is_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x)

theorem monotonic_range_a :
  ∀ a : ℝ, is_monotonic (f a) (-2) 3 ↔ a ≤ -27 ∨ 0 ≤ a :=
by sorry

end NUMINAMATH_CALUDE_monotonic_range_a_l1448_144825


namespace NUMINAMATH_CALUDE_all_balls_are_red_l1448_144814

/-- 
Given a bag of 12 balls that are either red or blue, 
prove that if the probability of drawing two red balls simultaneously is 1/10, 
then all 12 balls must be red.
-/
theorem all_balls_are_red (total_balls : ℕ) (red_balls : ℕ) 
  (h1 : total_balls = 12)
  (h2 : red_balls ≤ total_balls)
  (h3 : (red_balls : ℚ) / total_balls * (red_balls - 1) / (total_balls - 1) = 1 / 10) :
  red_balls = total_balls :=
sorry

end NUMINAMATH_CALUDE_all_balls_are_red_l1448_144814


namespace NUMINAMATH_CALUDE_angle_ratio_not_implies_right_triangle_l1448_144874

/-- Triangle ABC with angles A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

/-- The condition that angles are in the ratio 3:4:5 -/
def angle_ratio (t : Triangle) : Prop :=
  ∃ (x : ℝ), t.A = 3*x ∧ t.B = 4*x ∧ t.C = 5*x

/-- A triangle is right if one of its angles is 90 degrees -/
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

/-- The main theorem: a triangle with angles in ratio 3:4:5 is not necessarily right -/
theorem angle_ratio_not_implies_right_triangle :
  ∃ (t : Triangle), angle_ratio t ∧ ¬is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_angle_ratio_not_implies_right_triangle_l1448_144874


namespace NUMINAMATH_CALUDE_largest_and_smallest_results_l1448_144848

/-- The type representing our expression with parentheses -/
inductive Expr
  | num : ℕ → Expr
  | op : Expr → Expr → Expr

/-- Evaluate an expression to a rational number -/
def eval : Expr → ℚ
  | Expr.num n => n
  | Expr.op e₁ e₂ => (eval e₁) / (eval e₂)

/-- Check if a rational number is an integer -/
def isInteger (q : ℚ) : Prop := ∃ n : ℤ, q = n

/-- The set of all possible expressions using numbers 1 to 10 -/
def validExpr : Set Expr := sorry

/-- The theorem stating the largest and smallest possible integer results -/
theorem largest_and_smallest_results :
  (∃ e ∈ validExpr, eval e = 44800 ∧ isInteger (eval e)) ∧
  (∃ e ∈ validExpr, eval e = 7 ∧ isInteger (eval e)) ∧
  (∀ e ∈ validExpr, isInteger (eval e) → 7 ≤ eval e ∧ eval e ≤ 44800) :=
sorry

end NUMINAMATH_CALUDE_largest_and_smallest_results_l1448_144848


namespace NUMINAMATH_CALUDE_two_parts_divisibility_l1448_144832

theorem two_parts_divisibility (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ 13 * x + 17 * y = 283 → 
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a + b = 283 ∧ 13 ∣ a ∧ 17 ∣ b :=
by sorry

end NUMINAMATH_CALUDE_two_parts_divisibility_l1448_144832


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1448_144833

theorem sum_with_radical_conjugate : 
  let x : ℝ := 15 - Real.sqrt 5000
  let y : ℝ := 15 + Real.sqrt 5000
  x + y = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1448_144833


namespace NUMINAMATH_CALUDE_correct_transformation_l1448_144860

theorem correct_transformation (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : 3 * a = 2 * b) :
  a / 2 = b / 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l1448_144860


namespace NUMINAMATH_CALUDE_fraction_problem_l1448_144813

theorem fraction_problem (f : ℚ) : f * 12 + 5 = 11 → f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1448_144813


namespace NUMINAMATH_CALUDE_sum_of_factorials_last_two_digits_l1448_144870

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def isExcluded (n : ℕ) : Bool := n % 3 == 0 && n % 5 == 0

def sumOfFactorials : ℕ := 
  (List.range 100).foldl (λ acc n => 
    if !isExcluded (n + 1) then 
      (acc + lastTwoDigits (factorial (n + 1))) % 100 
    else 
      acc
  ) 0

theorem sum_of_factorials_last_two_digits : 
  sumOfFactorials = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_factorials_last_two_digits_l1448_144870


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1448_144807

theorem quadratic_no_real_roots 
  (p q a b c : ℝ) 
  (hp_pos : p > 0) (hq_pos : q > 0) (ha_pos : a > 0) (hb_pos : b > 0) (hc_pos : c > 0)
  (hp_neq_q : p ≠ q)
  (h_geom : a^2 = p * q)
  (h_arith : b = (2*p + q)/3 ∧ c = (p + 2*q)/3) :
  (2*a)^2 - 4*b*c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1448_144807


namespace NUMINAMATH_CALUDE_touch_football_point_difference_l1448_144837

/-- The point difference between two teams in a touch football game -/
def point_difference (
  touchdown_points : ℕ)
  (extra_point_points : ℕ)
  (field_goal_points : ℕ)
  (team1_touchdowns : ℕ)
  (team1_extra_points : ℕ)
  (team1_field_goals : ℕ)
  (team2_touchdowns : ℕ)
  (team2_extra_points : ℕ)
  (team2_field_goals : ℕ) : ℕ :=
  (team2_touchdowns * touchdown_points +
   team2_extra_points * extra_point_points +
   team2_field_goals * field_goal_points) -
  (team1_touchdowns * touchdown_points +
   team1_extra_points * extra_point_points +
   team1_field_goals * field_goal_points)

theorem touch_football_point_difference :
  point_difference 7 1 3 6 4 2 8 6 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_touch_football_point_difference_l1448_144837


namespace NUMINAMATH_CALUDE_greatest_b_value_l1448_144843

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, x^2 - 12*x + 32 ≤ 0 → x ≤ 8) ∧ 
  (8^2 - 12*8 + 32 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l1448_144843


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1448_144834

/-- Represents a parabola in the form y = ax² + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically --/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_shift_theorem (x : ℝ) :
  let original := Parabola.mk 2 0 0  -- y = 2x²
  let shifted := shift_parabola original 3 1  -- Shift 3 right, 1 down
  shifted.a * x^2 + shifted.b * x + shifted.c = 2 * (x - 3)^2 - 1 := by
  sorry

#check parabola_shift_theorem

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1448_144834


namespace NUMINAMATH_CALUDE_intersection_distance_implies_a_value_l1448_144812

-- Define the curve C
def curve_C (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x ∧ a > 0

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define the intersection points
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ curve_C a x y ∧ line_l x y}

-- Theorem statement
theorem intersection_distance_implies_a_value (a : ℝ) :
  (∃ (A B : ℝ × ℝ), A ∈ intersection_points a ∧ B ∈ intersection_points a ∧ 
    A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 10) →
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_implies_a_value_l1448_144812


namespace NUMINAMATH_CALUDE_room_width_proof_l1448_144862

/-- Given a rectangular room with specified dimensions and veranda, prove its width. -/
theorem room_width_proof (room_length : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) : 
  room_length = 17 →
  veranda_width = 2 →
  veranda_area = 132 →
  ∃ room_width : ℝ,
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - 
    (room_length * room_width) = veranda_area ∧
    room_width = 12 := by
  sorry

end NUMINAMATH_CALUDE_room_width_proof_l1448_144862


namespace NUMINAMATH_CALUDE_calculator_change_l1448_144882

/-- Calculates the change received after buying three types of calculators. -/
theorem calculator_change (total_money : ℕ) (basic_cost : ℕ) : 
  total_money = 100 →
  basic_cost = 8 →
  total_money - (basic_cost + 2 * basic_cost + 3 * (2 * basic_cost)) = 28 := by
  sorry

#check calculator_change

end NUMINAMATH_CALUDE_calculator_change_l1448_144882


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1448_144856

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 7*x < 12 ↔ -4 < x ∧ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1448_144856


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l1448_144838

theorem solve_system_of_equations :
  ∀ (x y m n : ℝ),
  (4 * x + 3 * y = m) →
  (6 * x - y = n) →
  ((m / 3 + n / 8 = 8) ∧ (m / 6 + n / 2 = 11)) →
  (x = 3 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l1448_144838


namespace NUMINAMATH_CALUDE_average_player_minutes_is_two_l1448_144872

/-- Represents the highlight film about Patricia's basketball team. -/
structure HighlightFilm where
  /-- Footage duration for each player in seconds -/
  point_guard : ℕ
  shooting_guard : ℕ
  small_forward : ℕ
  power_forward : ℕ
  center : ℕ
  /-- Additional content durations in seconds -/
  game_footage : ℕ
  interviews : ℕ
  opening_closing : ℕ
  /-- Pause duration between segments in seconds -/
  pause_duration : ℕ

/-- Calculates the average number of minutes attributed to each player's footage -/
def averagePlayerMinutes (film : HighlightFilm) : ℚ :=
  let total_player_footage := film.point_guard + film.shooting_guard + film.small_forward + 
                              film.power_forward + film.center
  let total_additional_content := film.game_footage + film.interviews + film.opening_closing
  let total_pause_time := film.pause_duration * 8
  let total_film_time := total_player_footage + total_additional_content + total_pause_time
  (total_player_footage : ℚ) / (5 * 60)

/-- Theorem stating that the average number of minutes attributed to each player's footage is 2 minutes -/
theorem average_player_minutes_is_two (film : HighlightFilm) 
  (h1 : film.point_guard = 130)
  (h2 : film.shooting_guard = 145)
  (h3 : film.small_forward = 85)
  (h4 : film.power_forward = 60)
  (h5 : film.center = 180)
  (h6 : film.game_footage = 120)
  (h7 : film.interviews = 90)
  (h8 : film.opening_closing = 30)
  (h9 : film.pause_duration = 15) :
  averagePlayerMinutes film = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_player_minutes_is_two_l1448_144872
