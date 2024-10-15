import Mathlib

namespace NUMINAMATH_CALUDE_abc_sum_product_l3489_348948

theorem abc_sum_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 0) (h2 : a^4 + b^4 + c^4 = 128) :
  a*b + b*c + c*a = -8 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_product_l3489_348948


namespace NUMINAMATH_CALUDE_hall_length_l3489_348967

/-- Hall represents a rectangular hall with specific properties -/
structure Hall where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ

/-- Properties of the hall -/
def hall_properties (h : Hall) : Prop :=
  h.width = 15 ∧
  h.volume = 1687.5 ∧
  2 * (h.length * h.width) = 2 * (h.length * h.height) + 2 * (h.width * h.height)

/-- Theorem stating that a hall with the given properties has a length of 15 meters -/
theorem hall_length (h : Hall) (hp : hall_properties h) : h.length = 15 := by
  sorry

end NUMINAMATH_CALUDE_hall_length_l3489_348967


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3489_348939

/-- Given vectors a, b, and c in ℝ², prove that if a is parallel to m*b - c, then m = -3. -/
theorem parallel_vectors_m_value (a b c : ℝ × ℝ) (m : ℝ) 
    (ha : a = (2, -1))
    (hb : b = (1, 0))
    (hc : c = (1, -2))
    (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a = k • (m • b - c)) :
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3489_348939


namespace NUMINAMATH_CALUDE_triangle_properties_l3489_348958

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle ABC -/
theorem triangle_properties (t : Triangle) 
  (h1 : 3 * Real.cos t.A * Real.cos t.C * (Real.tan t.A * Real.tan t.C - 1) = 1)
  (h2 : t.a + t.c = 3 * Real.sqrt 3 / 2)
  (h3 : t.b = Real.sqrt 3) : 
  Real.sin (2 * t.B - 5 * Real.pi / 6) = (7 - 4 * Real.sqrt 6) / 18 ∧ 
  t.a * t.c * Real.sin t.B / 2 = 15 * Real.sqrt 2 / 32 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3489_348958


namespace NUMINAMATH_CALUDE_pages_used_l3489_348926

def cards_per_page : ℕ := 3
def new_cards : ℕ := 3
def old_cards : ℕ := 9

theorem pages_used :
  (new_cards + old_cards) / cards_per_page = 4 :=
by sorry

end NUMINAMATH_CALUDE_pages_used_l3489_348926


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3489_348921

theorem rectangular_to_polar_conversion :
  ∃ (r : ℝ) (θ : ℝ), 
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r * Real.cos θ = 2 ∧
    r * Real.sin θ = -2 ∧
    r = 2 * Real.sqrt 2 ∧
    θ = 7 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3489_348921


namespace NUMINAMATH_CALUDE_complement_of_intersection_in_S_l3489_348954

def S : Set ℝ := {-2, -1, 0, 1, 2}
def T : Set ℝ := {x | x + 1 ≤ 2}

theorem complement_of_intersection_in_S :
  (S \ (S ∩ T)) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_in_S_l3489_348954


namespace NUMINAMATH_CALUDE_problem_solution_l3489_348987

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (a x : ℝ) : ℝ := 1/2 * x^2 - 4*a*x + a * log x + a + 1/2

/-- The function g(x) as defined in the problem -/
noncomputable def g (a x : ℝ) : ℝ := f a x + 2*a

/-- The derivative of g(x) -/
noncomputable def g' (a x : ℝ) : ℝ := x - 4*a + a/x

theorem problem_solution (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    g' a x₁ = 0 ∧ g' a x₂ = 0 ∧
    g a x₁ + g a x₂ ≥ g' a (x₁ * x₂)) →
  1/4 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3489_348987


namespace NUMINAMATH_CALUDE_prob_four_ones_twelve_dice_l3489_348992

def n : ℕ := 12  -- total number of dice
def k : ℕ := 4   -- number of dice showing 1
def s : ℕ := 6   -- number of sides on each die

-- Probability of rolling exactly k ones out of n dice
def prob_exactly_k_ones : ℚ :=
  (Nat.choose n k : ℚ) * (1 / s) ^ k * ((s - 1) / s) ^ (n - k)

theorem prob_four_ones_twelve_dice :
  prob_exactly_k_ones = 495 * 390625 / 2176782336 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_ones_twelve_dice_l3489_348992


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_first_four_primes_l3489_348932

def first_four_primes : List ℕ := [2, 3, 5, 7]

theorem arithmetic_mean_of_reciprocals_first_four_primes :
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length : ℚ) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_first_four_primes_l3489_348932


namespace NUMINAMATH_CALUDE_value_of_d_l3489_348963

theorem value_of_d (r s t u d : ℕ+) 
  (h1 : r^5 = s^4)
  (h2 : t^3 = u^2)
  (h3 : t - r = 19)
  (h4 : d = u - s) :
  d = 757 := by
  sorry

end NUMINAMATH_CALUDE_value_of_d_l3489_348963


namespace NUMINAMATH_CALUDE_positive_c_geq_one_l3489_348991

theorem positive_c_geq_one (a b : ℕ+) (c : ℝ) 
  (h_c_pos : c > 0) 
  (h_eq : (a + 1 : ℝ) / (b + c) = (b : ℝ) / a) : 
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_positive_c_geq_one_l3489_348991


namespace NUMINAMATH_CALUDE_product_remainder_one_mod_three_l3489_348941

theorem product_remainder_one_mod_three (a b : ℕ) :
  a % 3 = 1 → b % 3 = 1 → (a * b) % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_one_mod_three_l3489_348941


namespace NUMINAMATH_CALUDE_no_distinct_cube_sum_equality_l3489_348969

theorem no_distinct_cube_sum_equality (a b c d : ℕ) :
  a^3 + b^3 = c^3 + d^3 → a + b = c + d → ¬(a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_cube_sum_equality_l3489_348969


namespace NUMINAMATH_CALUDE_jason_gave_nine_cards_l3489_348961

/-- The number of Pokemon cards Jason started with -/
def initial_cards : ℕ := 13

/-- The number of Pokemon cards Jason has left -/
def remaining_cards : ℕ := 4

/-- The number of Pokemon cards Jason gave to his friends -/
def cards_given : ℕ := initial_cards - remaining_cards

theorem jason_gave_nine_cards : cards_given = 9 := by sorry

end NUMINAMATH_CALUDE_jason_gave_nine_cards_l3489_348961


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l3489_348964

/-- Given three square regions A, B, and C with perimeters 16, 20, and 40 units respectively,
    the ratio of the area of region B to the area of region C is 1/4 -/
theorem area_ratio_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (pa : 4 * a = 16) (pb : 4 * b = 20) (pc : 4 * c = 40) :
  (b ^ 2) / (c ^ 2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l3489_348964


namespace NUMINAMATH_CALUDE_figurine_cost_is_17_l3489_348938

/-- The cost of each figurine in Annie's purchase --/
def figurine_cost : ℚ :=
  let brand_a_cost : ℚ := 65
  let brand_b_cost : ℚ := 75
  let brand_c_cost : ℚ := 85
  let brand_a_count : ℕ := 3
  let brand_b_count : ℕ := 2
  let brand_c_count : ℕ := 4
  let figurine_count : ℕ := 10
  let figurine_total_cost : ℚ := 2 * brand_c_cost
  figurine_total_cost / figurine_count

theorem figurine_cost_is_17 : figurine_cost = 17 := by
  sorry

end NUMINAMATH_CALUDE_figurine_cost_is_17_l3489_348938


namespace NUMINAMATH_CALUDE_total_cost_of_items_l3489_348980

/-- The total cost of items given their price relationships -/
theorem total_cost_of_items (chair_price : ℝ) : 
  chair_price > 0 →
  let table_price := 3 * chair_price
  let couch_price := 5 * table_price
  couch_price = 300 →
  chair_price + table_price + couch_price = 380 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_items_l3489_348980


namespace NUMINAMATH_CALUDE_problem_solution_l3489_348905

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x/3 = y^2) 
  (h3 : x/5 = 5*y + 2) : 
  x = (685 + 25 * Real.sqrt 745) / 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3489_348905


namespace NUMINAMATH_CALUDE_r_earnings_l3489_348904

/-- Represents the daily earnings of individuals p, q, and r -/
structure Earnings where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The conditions given in the problem -/
def problem_conditions (e : Earnings) : Prop :=
  e.p + e.q + e.r = 1980 / 9 ∧
  e.p + e.r = 600 / 5 ∧
  e.q + e.r = 910 / 7

/-- The theorem stating that under the given conditions, r earns 30 rs per day -/
theorem r_earnings (e : Earnings) : problem_conditions e → e.r = 30 := by
  sorry

end NUMINAMATH_CALUDE_r_earnings_l3489_348904


namespace NUMINAMATH_CALUDE_max_value_theorem_l3489_348962

theorem max_value_theorem (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : 2*x + y + z = 4) : 
  x^2 + x*(y + z) + y*z ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3489_348962


namespace NUMINAMATH_CALUDE_triangle_area_from_square_sides_l3489_348950

theorem triangle_area_from_square_sides (a b c : Real) 
  (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) : 
  (1/2) * a * b = 24 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_from_square_sides_l3489_348950


namespace NUMINAMATH_CALUDE_summer_reading_goal_l3489_348988

/-- The number of books Carlos read in June -/
def june_books : ℕ := 42

/-- The number of books Carlos read in July -/
def july_books : ℕ := 28

/-- The number of books Carlos read in August -/
def august_books : ℕ := 30

/-- Carlos' goal for the number of books to read during summer vacation -/
def summer_goal : ℕ := june_books + july_books + august_books

theorem summer_reading_goal : summer_goal = 100 := by
  sorry

end NUMINAMATH_CALUDE_summer_reading_goal_l3489_348988


namespace NUMINAMATH_CALUDE_x_value_l3489_348922

def x : ℚ := (320 / 2) / 3

theorem x_value : x = 160 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3489_348922


namespace NUMINAMATH_CALUDE_probability_theorem_l3489_348960

def is_valid_pair (b c : Int) : Prop :=
  (b.natAbs ≤ 6) ∧ (c.natAbs ≤ 6)

def has_non_real_or_non_positive_roots (b c : Int) : Prop :=
  (b^2 < 4*c) ∨ (b ≥ 0) ∨ (b^2 ≤ 4*c)

def total_pairs : Nat := 13 * 13

def valid_pairs : Nat := 150

theorem probability_theorem :
  (Nat.cast valid_pairs / Nat.cast total_pairs : ℚ) = 150 / 169 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l3489_348960


namespace NUMINAMATH_CALUDE_division_problem_l3489_348947

theorem division_problem :
  ∃! x : ℕ, x < 50 ∧ ∃ m : ℕ, 100 = m * x + 6 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_division_problem_l3489_348947


namespace NUMINAMATH_CALUDE_rearrangement_divisible_by_seven_l3489_348929

/-- A function that checks if a natural number contains the digits 1, 3, 7, and 9 -/
def containsRequiredDigits (n : ℕ) : Prop := sorry

/-- A function that represents all possible rearrangements of digits in a natural number -/
def rearrangeDigits (n : ℕ) : Set ℕ := sorry

/-- Theorem: For any natural number containing the digits 1, 3, 7, and 9,
    there exists a rearrangement of its digits that is divisible by 7 -/
theorem rearrangement_divisible_by_seven (n : ℕ) :
  containsRequiredDigits n →
  ∃ m ∈ rearrangeDigits n, m % 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_rearrangement_divisible_by_seven_l3489_348929


namespace NUMINAMATH_CALUDE_golden_ratio_greater_than_half_l3489_348977

theorem golden_ratio_greater_than_half : (Real.sqrt 5 - 1) / 2 > 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_greater_than_half_l3489_348977


namespace NUMINAMATH_CALUDE_calculation_proof_l3489_348976

theorem calculation_proof :
  (6.42 - 2.8 + 3.58 = 7.2) ∧ (0.36 / (0.4 * (6.1 - 4.6)) = 0.6) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3489_348976


namespace NUMINAMATH_CALUDE_banana_arrangements_l3489_348911

theorem banana_arrangements : 
  let total_letters : ℕ := 6
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  let b_count : ℕ := 1
  (total_letters.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l3489_348911


namespace NUMINAMATH_CALUDE_correct_proposition_l3489_348993

-- Define proposition P
def P : Prop := ∀ x : ℝ, x^2 ≥ 0

-- Define proposition Q
def Q : Prop := ∃ x : ℚ, x^2 ≠ 3

-- Theorem to prove
theorem correct_proposition : P ∨ (¬Q) := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l3489_348993


namespace NUMINAMATH_CALUDE_x_intercepts_count_l3489_348901

theorem x_intercepts_count : 
  (⌊(100000 : ℝ) / Real.pi⌋ - ⌊(10000 : ℝ) / Real.pi⌋ : ℤ) = 28647 := by
  sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l3489_348901


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3489_348971

theorem simplify_fraction_product : 
  10 * (15 / 8) * (-28 / 45) * (3 / 5) = -7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3489_348971


namespace NUMINAMATH_CALUDE_water_tank_capacity_l3489_348919

/-- Represents a cylindrical water tank with a given capacity and initial water level. -/
structure WaterTank where
  capacity : ℝ
  initialWater : ℝ

/-- Proves that a water tank with the given properties has a capacity of 30 liters. -/
theorem water_tank_capacity (tank : WaterTank)
  (h1 : tank.initialWater / tank.capacity = 1 / 6)
  (h2 : (tank.initialWater + 5) / tank.capacity = 1 / 3) :
  tank.capacity = 30 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l3489_348919


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l3489_348973

theorem right_triangle_hypotenuse_and_perimeter :
  let a : ℝ := 8.5
  let b : ℝ := 15
  let h : ℝ := Real.sqrt (a^2 + b^2)
  let perimeter : ℝ := a + b + h
  h = 17.25 ∧ perimeter = 40.75 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l3489_348973


namespace NUMINAMATH_CALUDE_expected_pollen_allergy_l3489_348903

theorem expected_pollen_allergy (total_sample : ℕ) (allergy_ratio : ℚ) 
  (h1 : total_sample = 400) 
  (h2 : allergy_ratio = 1 / 4) : 
  ↑total_sample * allergy_ratio = 100 := by
  sorry

end NUMINAMATH_CALUDE_expected_pollen_allergy_l3489_348903


namespace NUMINAMATH_CALUDE_condition_relationship_l3489_348974

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a + b = 1 → 4 * a * b ≤ 1) ∧
  (∃ a b, 4 * a * b ≤ 1 ∧ a + b ≠ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l3489_348974


namespace NUMINAMATH_CALUDE_lois_final_book_count_l3489_348951

def calculate_final_books (initial_books : ℕ) : ℕ :=
  let books_after_giving := initial_books - (initial_books / 4)
  let nonfiction_books := (books_after_giving * 60) / 100
  let kept_nonfiction := nonfiction_books / 2
  let fiction_books := books_after_giving - nonfiction_books
  let kept_fiction := fiction_books - (fiction_books / 3)
  let new_books := 12
  kept_nonfiction + kept_fiction + new_books

theorem lois_final_book_count :
  calculate_final_books 150 = 76 := by
  sorry

end NUMINAMATH_CALUDE_lois_final_book_count_l3489_348951


namespace NUMINAMATH_CALUDE_largest_expression_l3489_348998

theorem largest_expression (P Q : ℝ) (h1 : P = 1000) (h2 : Q = 0.01) :
  (P / Q > P + Q) ∧ (P / Q > P * Q) ∧ (P / Q > Q / P) ∧ (P / Q > P - Q) :=
by sorry

end NUMINAMATH_CALUDE_largest_expression_l3489_348998


namespace NUMINAMATH_CALUDE_trivia_team_score_l3489_348902

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_points : ℕ) :
  total_members = 12 →
  absent_members = 4 →
  total_points = 64 →
  (total_points / (total_members - absent_members) = 8) :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_score_l3489_348902


namespace NUMINAMATH_CALUDE_auspicious_count_l3489_348959

/-- Returns true if n is an auspicious number (multiple of 6 with digit sum 6) -/
def isAuspicious (n : Nat) : Bool :=
  n % 6 = 0 && (n / 100 + (n / 10) % 10 + n % 10 = 6)

/-- Count of auspicious numbers between 100 and 999 -/
def countAuspicious : Nat :=
  (List.range 900).map (· + 100)
    |>.filter isAuspicious
    |>.length

theorem auspicious_count : countAuspicious = 12 := by
  sorry

end NUMINAMATH_CALUDE_auspicious_count_l3489_348959


namespace NUMINAMATH_CALUDE_all_stars_arrangement_l3489_348915

/-- The number of ways to arrange All-Stars in a row -/
def arrange_all_stars (total : ℕ) (cubs : ℕ) (red_sox : ℕ) (yankees : ℕ) (dodgers : ℕ) : ℕ :=
  Nat.factorial 4 * Nat.factorial cubs * Nat.factorial red_sox * Nat.factorial yankees * Nat.factorial dodgers

/-- Theorem stating the number of arrangements for the given problem -/
theorem all_stars_arrangement :
  arrange_all_stars 10 4 3 2 1 = 6912 := by
  sorry

end NUMINAMATH_CALUDE_all_stars_arrangement_l3489_348915


namespace NUMINAMATH_CALUDE_min_probability_cards_unique_min_probability_cards_l3489_348975

/-- Represents the probability of a card being red-side up after flips -/
def probability_red_up (k : ℕ) : ℚ :=
  if k ≤ 25 then
    (676 - 52 * k + 2 * k^2) / 676
  else
    (676 - 52 * (51 - k) + 2 * (51 - k)^2) / 676

/-- The statement to prove -/
theorem min_probability_cards :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 50 →
    (probability_red_up 13 ≤ probability_red_up k ∧
     probability_red_up 38 ≤ probability_red_up k) :=
by sorry

/-- Uniqueness of the minimum probability cards -/
theorem unique_min_probability_cards :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 50 →
    (k ≠ 13 ∧ k ≠ 38 →
      probability_red_up 13 < probability_red_up k ∧
      probability_red_up 38 < probability_red_up k) :=
by sorry

end NUMINAMATH_CALUDE_min_probability_cards_unique_min_probability_cards_l3489_348975


namespace NUMINAMATH_CALUDE_remainder_53_pow_10_mod_8_l3489_348908

theorem remainder_53_pow_10_mod_8 : 53^10 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_53_pow_10_mod_8_l3489_348908


namespace NUMINAMATH_CALUDE_actual_tissue_diameter_l3489_348949

/-- Given a circular piece of tissue magnified by an electron microscope, 
    this theorem proves that the actual diameter of the tissue is 0.001 centimeters. -/
theorem actual_tissue_diameter 
  (magnification : ℝ) 
  (magnified_diameter : ℝ) 
  (h1 : magnification = 1000)
  (h2 : magnified_diameter = 1) : 
  magnified_diameter / magnification = 0.001 := by
  sorry

end NUMINAMATH_CALUDE_actual_tissue_diameter_l3489_348949


namespace NUMINAMATH_CALUDE_exists_phi_and_x0_for_sin_product_equals_one_l3489_348930

theorem exists_phi_and_x0_for_sin_product_equals_one : 
  ∃ (φ : ℝ) (x₀ : ℝ), Real.sin x₀ * Real.sin (x₀ + φ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_phi_and_x0_for_sin_product_equals_one_l3489_348930


namespace NUMINAMATH_CALUDE_inscribed_prism_volume_l3489_348906

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- A regular triangular prism inscribed in a regular tetrahedron -/
structure InscribedPrism (a : ℝ) extends RegularTetrahedron a where
  /-- One base of the prism has vertices on the lateral edges of the tetrahedron -/
  base_on_edges : Bool
  /-- The other base of the prism lies in the plane of the tetrahedron's base -/
  base_in_plane : Bool
  /-- All edges of the prism are equal -/
  equal_edges : Bool

/-- The volume of the inscribed prism -/
noncomputable def prism_volume (p : InscribedPrism a) : ℝ :=
  (a^3 * (27 * Real.sqrt 2 - 22 * Real.sqrt 3)) / 2

/-- Theorem: The volume of the inscribed prism is (a³(27√2 - 22√3))/2 -/
theorem inscribed_prism_volume (a : ℝ) (p : InscribedPrism a) :
  prism_volume p = (a^3 * (27 * Real.sqrt 2 - 22 * Real.sqrt 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_prism_volume_l3489_348906


namespace NUMINAMATH_CALUDE_solution_set_l3489_348924

theorem solution_set (x : ℝ) : 
  x > 4 → x^3 - 8*x^2 + 16*x > 64 ∧ x^2 - 4*x + 5 > 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l3489_348924


namespace NUMINAMATH_CALUDE_rectangle_length_from_square_l3489_348982

theorem rectangle_length_from_square (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 20 →
  rect_width = 14 →
  4 * square_side = 2 * (rect_width + rect_length) →
  rect_length = 26 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_from_square_l3489_348982


namespace NUMINAMATH_CALUDE_integer_solution_correct_rational_solution_correct_l3489_348981

-- Define the equation
def equation (x y : ℚ) : Prop := 2 * x^3 + x * y - 7 = 0

-- Define the set of integer solutions
def integer_solutions : Set (ℤ × ℤ) :=
  {(1, 5), (-1, -9), (7, -97), (-7, -99)}

-- Define the rational solution function
def rational_solution (x : ℚ) : ℚ := 7 / x - 2 * x^2

-- Theorem for integer solutions
theorem integer_solution_correct :
  ∀ (x y : ℤ), (x, y) ∈ integer_solutions → equation (x : ℚ) (y : ℚ) :=
sorry

-- Theorem for rational solutions
theorem rational_solution_correct :
  ∀ (x : ℚ), x ≠ 0 → equation x (rational_solution x) :=
sorry

end NUMINAMATH_CALUDE_integer_solution_correct_rational_solution_correct_l3489_348981


namespace NUMINAMATH_CALUDE_no_solution_exists_l3489_348978

theorem no_solution_exists : ¬∃ (a b : ℕ+), 
  a * b + 52 = 20 * Nat.lcm a b + 15 * Nat.gcd a b := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3489_348978


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_l3489_348907

theorem smallest_solution_quartic (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 →
  x ≥ -Real.sqrt 26 ∧
  ∃ y, y^4 - 50*y^2 + 576 = 0 ∧ y = -Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_l3489_348907


namespace NUMINAMATH_CALUDE_wire_length_proof_l3489_348916

/-- The length of wire used to make an equilateral triangle plus the leftover wire -/
def total_wire_length (side_length : ℝ) (leftover : ℝ) : ℝ :=
  3 * side_length + leftover

/-- Theorem: Given an equilateral triangle with side length 19 cm and 15 cm of leftover wire,
    the total length of wire is 72 cm. -/
theorem wire_length_proof :
  total_wire_length 19 15 = 72 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_proof_l3489_348916


namespace NUMINAMATH_CALUDE_problem_statement_l3489_348970

theorem problem_statement (m n : ℤ) : 
  |m - 2023| + (n + 2024)^2 = 0 → (m + n)^2023 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3489_348970


namespace NUMINAMATH_CALUDE_power_four_times_base_equals_power_five_l3489_348953

theorem power_four_times_base_equals_power_five (a : ℝ) : a^4 * a = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_four_times_base_equals_power_five_l3489_348953


namespace NUMINAMATH_CALUDE_right_triangle_area_l3489_348944

/-- The area of a right triangle with given side lengths -/
theorem right_triangle_area 
  (X Y Z : ℝ × ℝ) -- Points in 2D plane
  (h_right : (X.1 - Y.1) * (X.1 - Z.1) + (X.2 - Y.2) * (X.2 - Z.2) = 0) -- Right angle at X
  (h_xy : Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 15) -- XY = 15
  (h_xz : Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) = 10) -- XZ = 10
  (h_median : ∃ M : ℝ × ℝ, M.1 = (Y.1 + Z.1) / 2 ∧ M.2 = (Y.2 + Z.2) / 2 ∧ 
    (X.1 - M.1) * (Y.1 - Z.1) + (X.2 - M.2) * (Y.2 - Z.2) = 0) -- Median bisects angle X
  : (1 / 2 : ℝ) * 15 * 10 = 75 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3489_348944


namespace NUMINAMATH_CALUDE_total_waiting_time_l3489_348989

def days_first_appointment : ℕ := 4
def days_second_appointment : ℕ := 20
def weeks_for_effectiveness : ℕ := 2
def days_per_week : ℕ := 7

theorem total_waiting_time :
  days_first_appointment + days_second_appointment + (weeks_for_effectiveness * days_per_week) = 38 := by
  sorry

end NUMINAMATH_CALUDE_total_waiting_time_l3489_348989


namespace NUMINAMATH_CALUDE_factorization_equality_l3489_348990

theorem factorization_equality (x y : ℝ) : x^3*y - x*y = x*y*(x - 1)*(x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3489_348990


namespace NUMINAMATH_CALUDE_helios_population_2060_l3489_348923

/-- The population growth function for Helios -/
def helios_population (initial_population : ℕ) (years_passed : ℕ) : ℕ :=
  initial_population * (2 ^ (years_passed / 20))

/-- Theorem stating the population of Helios in 2060 -/
theorem helios_population_2060 :
  helios_population 250 60 = 2000 := by
  sorry

#eval helios_population 250 60

end NUMINAMATH_CALUDE_helios_population_2060_l3489_348923


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3489_348966

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (x_ge : x ≥ -2/3)
  (y_ge : y ≥ -1)
  (z_ge : z ≥ -2) :
  ∃ (max : ℝ), max = Real.sqrt 57 ∧ 
    ∀ a b c : ℝ, a + b + c = 2 → a ≥ -2/3 → b ≥ -1 → c ≥ -2 →
      Real.sqrt (3*a + 2) + Real.sqrt (3*b + 4) + Real.sqrt (3*c + 7) ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3489_348966


namespace NUMINAMATH_CALUDE_num_assignment_plans_l3489_348996

/-- The number of male doctors -/
def num_male_doctors : ℕ := 6

/-- The number of female doctors -/
def num_female_doctors : ℕ := 4

/-- The number of male doctors to be selected -/
def selected_male_doctors : ℕ := 3

/-- The number of female doctors to be selected -/
def selected_female_doctors : ℕ := 2

/-- The number of regions -/
def num_regions : ℕ := 5

/-- Function to calculate the number of assignment plans -/
def calculate_assignment_plans : ℕ := sorry

/-- Theorem stating the number of different assignment plans -/
theorem num_assignment_plans : 
  calculate_assignment_plans = 12960 := by sorry

end NUMINAMATH_CALUDE_num_assignment_plans_l3489_348996


namespace NUMINAMATH_CALUDE_percentage_relation_l3489_348968

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.08 * x) (h2 : b = 0.16 * x) :
  a = 0.5 * b := by sorry

end NUMINAMATH_CALUDE_percentage_relation_l3489_348968


namespace NUMINAMATH_CALUDE_shift_by_two_equiv_l3489_348917

/-- A function that represents a vertical shift of another function -/
def verticalShift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := fun x ↦ f x + shift

/-- Theorem stating that f(x) + 2 is equivalent to shifting f(x) upward by 2 units -/
theorem shift_by_two_equiv (f : ℝ → ℝ) (x : ℝ) : 
  f x + 2 = verticalShift f 2 x := by sorry

end NUMINAMATH_CALUDE_shift_by_two_equiv_l3489_348917


namespace NUMINAMATH_CALUDE_right_triangle_side_lengths_l3489_348994

/-- A right-angled triangle with given incircle and circumcircle radii -/
structure RightTriangle where
  -- The radius of the incircle
  inradius : ℝ
  -- The radius of the circumcircle
  circumradius : ℝ
  -- The lengths of the three sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- Conditions
  inradius_positive : 0 < inradius
  circumradius_positive : 0 < circumradius
  right_angle : a ^ 2 + b ^ 2 = c ^ 2
  incircle_condition : a + b - c = 2 * inradius
  circumcircle_condition : c = 2 * circumradius

/-- The main theorem stating the side lengths of the triangle -/
theorem right_triangle_side_lengths (t : RightTriangle) 
    (h1 : t.inradius = 8)
    (h2 : t.circumradius = 41) :
    (t.a = 18 ∧ t.b = 80 ∧ t.c = 82) ∨ (t.a = 80 ∧ t.b = 18 ∧ t.c = 82) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_lengths_l3489_348994


namespace NUMINAMATH_CALUDE_power_sum_equals_123_l3489_348965

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^10 + b^10 = 123 -/
theorem power_sum_equals_123 (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_123_l3489_348965


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l3489_348943

/-- The volume of a cube with edge length a -/
def cube_volume (a : ℝ) : ℝ := a ^ 3

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The volume of the remaining part after cutting small cubes from vertices -/
def remaining_volume (edge_length : ℝ) (small_cube_volume : ℝ) : ℝ :=
  cube_volume edge_length - (cube_vertices : ℝ) * small_cube_volume

/-- Theorem: The volume of a cube with edge length 3 cm, after removing 
    small cubes of volume 1 cm³ from each of its vertices, is 19 cm³ -/
theorem remaining_cube_volume : 
  remaining_volume 3 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cube_volume_l3489_348943


namespace NUMINAMATH_CALUDE_cubic_point_tangent_l3489_348935

theorem cubic_point_tangent (a : ℝ) (h : a^3 = 27) : 
  Real.tan (π / a) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_point_tangent_l3489_348935


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3489_348972

theorem complex_fraction_simplification :
  (3 + Complex.I) / (1 - Complex.I) = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3489_348972


namespace NUMINAMATH_CALUDE_reciprocal_of_two_thirds_l3489_348957

def reciprocal (a b : ℚ) : ℚ := b / a

theorem reciprocal_of_two_thirds :
  reciprocal (2 : ℚ) 3 = (3 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_two_thirds_l3489_348957


namespace NUMINAMATH_CALUDE_chocolate_division_l3489_348940

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_given : ℕ) :
  total_chocolate = 60 / 7 →
  num_piles = 5 →
  piles_given = 2 →
  piles_given * (total_chocolate / num_piles) = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l3489_348940


namespace NUMINAMATH_CALUDE_characterize_satisfying_functions_l3489_348956

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, x * (f (x + 1) - f x) = f x) ∧
  (∀ x y : ℝ, |f x - f y| ≤ |x - y|)

/-- The theorem stating the form of functions satisfying the conditions -/
theorem characterize_satisfying_functions :
  ∀ f : ℝ → ℝ, SatisfyingFunction f →
  ∃ k : ℝ, (∀ x : ℝ, f x = k * x) ∧ |k| ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_characterize_satisfying_functions_l3489_348956


namespace NUMINAMATH_CALUDE_legitimate_paths_count_l3489_348995

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Defines the grid dimensions -/
def gridWidth : Nat := 12
def gridHeight : Nat := 4

/-- Defines the start and end points -/
def pointA : GridPoint := ⟨0, 0⟩
def pointB : GridPoint := ⟨gridWidth - 1, gridHeight - 1⟩

/-- Checks if a path is legitimate based on the column restrictions -/
def isLegitimate (path : List GridPoint) : Bool :=
  path.all fun p =>
    (p.x ≠ 2 || p.y = 0 || p.y = 1 || p.y = gridHeight - 1) &&
    (p.x ≠ 4 || p.y = 0 || p.y = gridHeight - 2 || p.y = gridHeight - 1)

/-- Counts the number of legitimate paths from A to B -/
def countLegitimatePaths : Nat :=
  sorry -- The actual implementation would go here

/-- The main theorem to prove -/
theorem legitimate_paths_count :
  countLegitimatePaths = 1289 := by
  sorry

end NUMINAMATH_CALUDE_legitimate_paths_count_l3489_348995


namespace NUMINAMATH_CALUDE_largest_unreachable_score_l3489_348986

/-- 
Given that:
1. Easy questions earn 3 points.
2. Harder questions earn 7 points.
3. Scores are achieved by combinations of these point values.

Prove that 11 is the largest integer that cannot be expressed as a linear combination of 3 and 7 
with non-negative integer coefficients.
-/
theorem largest_unreachable_score : 
  ∀ n : ℕ, n > 11 → ∃ x y : ℕ, n = 3 * x + 7 * y :=
by sorry

end NUMINAMATH_CALUDE_largest_unreachable_score_l3489_348986


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3489_348983

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x^2 - x - 6 > 0) ↔ (x < -3/2 ∨ x > 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3489_348983


namespace NUMINAMATH_CALUDE_fourth_pill_time_l3489_348955

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := (totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  ⟨newHours, newMinutes, by sorry, by sorry⟩

/-- The time interval between pills in minutes -/
def pillInterval : Nat := 75

/-- The starting time when the first pill is taken -/
def startTime : Time := ⟨11, 5, by sorry, by sorry⟩

/-- The number of pills taken -/
def pillCount : Nat := 4

theorem fourth_pill_time :
  addMinutes startTime ((pillCount - 1) * pillInterval) = ⟨14, 50, by sorry, by sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_fourth_pill_time_l3489_348955


namespace NUMINAMATH_CALUDE_max_value_when_t_2_t_value_when_max_2_l3489_348913

-- Define the function f(x, t)
def f (x t : ℝ) : ℝ := |2 * x - 1| - |t * x + 3|

-- Theorem 1: Maximum value of f(x) when t = 2 is 4
theorem max_value_when_t_2 :
  ∃ M : ℝ, M = 4 ∧ ∀ x : ℝ, f x 2 ≤ M :=
sorry

-- Theorem 2: When maximum value of f(x) is 2, t = 6
theorem t_value_when_max_2 :
  ∃ t : ℝ, t > 0 ∧ (∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, f x t ≤ M) → t = 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_when_t_2_t_value_when_max_2_l3489_348913


namespace NUMINAMATH_CALUDE_redistribution_amount_l3489_348928

def earnings : List ℕ := [18, 22, 26, 32, 47]

theorem redistribution_amount (earnings : List ℕ) (h1 : earnings = [18, 22, 26, 32, 47]) :
  let total := earnings.sum
  let equalShare := total / earnings.length
  let maxEarning := earnings.maximum?
  maxEarning.map (λ max => max - equalShare) = some 18 := by
  sorry

end NUMINAMATH_CALUDE_redistribution_amount_l3489_348928


namespace NUMINAMATH_CALUDE_remainder_3_1000_mod_7_l3489_348936

theorem remainder_3_1000_mod_7 : 3^1000 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_1000_mod_7_l3489_348936


namespace NUMINAMATH_CALUDE_bowling_tournament_sequences_l3489_348999

/-- Represents a tournament with a fixed number of players and rounds. -/
structure Tournament :=
  (num_players : ℕ)
  (num_rounds : ℕ)
  (outcomes_per_match : ℕ)

/-- Calculates the number of possible award distribution sequences for a given tournament. -/
def award_sequences (t : Tournament) : ℕ :=
  t.outcomes_per_match ^ t.num_rounds

/-- Theorem stating that a tournament with 5 players, 4 rounds, and 2 outcomes per match has 16 possible award sequences. -/
theorem bowling_tournament_sequences :
  ∃ t : Tournament, t.num_players = 5 ∧ t.num_rounds = 4 ∧ t.outcomes_per_match = 2 ∧ award_sequences t = 16 :=
sorry

end NUMINAMATH_CALUDE_bowling_tournament_sequences_l3489_348999


namespace NUMINAMATH_CALUDE_max_k_value_l3489_348952

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 4 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ 4 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 = 4^2 * (x^2 / y^2 + y^2 / x^2) + 4 * (x / y + y / x) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3489_348952


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3489_348933

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 2 + a 4024 = 4) :
  a 2013 = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3489_348933


namespace NUMINAMATH_CALUDE_building_height_l3489_348920

/-- Given a flagpole and a building casting shadows under similar conditions,
    prove that the height of the building is 22 meters. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 55)
  : (flagpole_height / flagpole_shadow) * building_shadow = 22 :=
by sorry

end NUMINAMATH_CALUDE_building_height_l3489_348920


namespace NUMINAMATH_CALUDE_g_of_13_l3489_348934

def g (x : ℝ) : ℝ := x^2 + 2*x + 25

theorem g_of_13 : g 13 = 220 := by
  sorry

end NUMINAMATH_CALUDE_g_of_13_l3489_348934


namespace NUMINAMATH_CALUDE_difference_max_min_both_l3489_348997

def total_students : ℕ := 1500

def spanish_min : ℕ := 1050
def spanish_max : ℕ := 1125

def french_min : ℕ := 525
def french_max : ℕ := 675

def min_both : ℕ := spanish_min + french_min - total_students
def max_both : ℕ := spanish_max + french_max - total_students

theorem difference_max_min_both : max_both - min_both = 225 := by
  sorry

end NUMINAMATH_CALUDE_difference_max_min_both_l3489_348997


namespace NUMINAMATH_CALUDE_same_terminal_side_as_405_degrees_l3489_348925

theorem same_terminal_side_as_405_degrees : ∀ (k : ℤ),
  ∃ (n : ℤ), 405 = n * 360 + 45 ∧ (k * 360 + 45) % 360 = 45 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_as_405_degrees_l3489_348925


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3489_348984

theorem sqrt_inequality (x : ℝ) : 
  Real.sqrt (3 - x) - Real.sqrt (x + 1) > (1 : ℝ) / 2 ↔ 
  -1 ≤ x ∧ x < 1 - Real.sqrt 31 / 8 := by
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3489_348984


namespace NUMINAMATH_CALUDE_factorization_of_4x_cubed_minus_x_l3489_348927

theorem factorization_of_4x_cubed_minus_x (x : ℝ) : 4 * x^3 - x = x * (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_cubed_minus_x_l3489_348927


namespace NUMINAMATH_CALUDE_no_real_roots_l3489_348979

theorem no_real_roots : ∀ x : ℝ, x^2 + |x| + 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3489_348979


namespace NUMINAMATH_CALUDE_share_distribution_l3489_348909

theorem share_distribution (total : ℕ) (ratio_a ratio_b ratio_c : ℕ) (share_c : ℕ) :
  total = 945 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 4 →
  share_c = (total * ratio_c) / (ratio_a + ratio_b + ratio_c) →
  share_c = 420 :=
by sorry

end NUMINAMATH_CALUDE_share_distribution_l3489_348909


namespace NUMINAMATH_CALUDE_indeterminate_neutral_eight_year_boys_l3489_348985

structure Classroom where
  total_children : Nat
  happy_children : Nat
  sad_children : Nat
  neutral_children : Nat
  total_boys : Nat
  total_girls : Nat
  happy_boys : Nat
  happy_girls : Nat
  sad_boys : Nat
  sad_girls : Nat
  age_seven_total : Nat
  age_seven_boys : Nat
  age_seven_girls : Nat
  age_eight_total : Nat
  age_eight_boys : Nat
  age_eight_girls : Nat
  age_nine_total : Nat
  age_nine_boys : Nat
  age_nine_girls : Nat

def classroom : Classroom := {
  total_children := 60,
  happy_children := 30,
  sad_children := 10,
  neutral_children := 20,
  total_boys := 16,
  total_girls := 44,
  happy_boys := 6,
  happy_girls := 12,
  sad_boys := 6,
  sad_girls := 4,
  age_seven_total := 20,
  age_seven_boys := 8,
  age_seven_girls := 12,
  age_eight_total := 25,
  age_eight_boys := 5,
  age_eight_girls := 20,
  age_nine_total := 15,
  age_nine_boys := 3,
  age_nine_girls := 12
}

theorem indeterminate_neutral_eight_year_boys (c : Classroom) : 
  c = classroom → 
  ¬∃ (n : Nat), n = c.age_eight_boys - (number_of_happy_eight_year_boys + number_of_sad_eight_year_boys) :=
by sorry

end NUMINAMATH_CALUDE_indeterminate_neutral_eight_year_boys_l3489_348985


namespace NUMINAMATH_CALUDE_eight_team_tournament_l3489_348945

/-- The number of matches in a single-elimination tournament -/
def num_matches (n : ℕ) : ℕ := n - 1

/-- Theorem: A single-elimination tournament with 8 teams requires 7 matches -/
theorem eight_team_tournament : num_matches 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_eight_team_tournament_l3489_348945


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_l3489_348910

theorem zeros_before_first_nonzero (n : ℕ) (m : ℕ) : 
  let fraction := 1 / (2^n * 5^m)
  let zeros := m - n
  zeros > 0 → zeros = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_l3489_348910


namespace NUMINAMATH_CALUDE_fireflies_win_by_five_l3489_348900

/-- Represents a basketball team's score -/
structure TeamScore where
  initial : ℕ
  final_baskets : ℕ
  basket_value : ℕ

/-- Calculates the final score of a team -/
def final_score (team : TeamScore) : ℕ :=
  team.initial + team.final_baskets * team.basket_value

/-- Represents the scores of both teams in the basketball game -/
structure GameScore where
  hornets : TeamScore
  fireflies : TeamScore

/-- The theorem stating the final score difference between Fireflies and Hornets -/
theorem fireflies_win_by_five (game : GameScore)
  (h1 : game.hornets = ⟨86, 2, 2⟩)
  (h2 : game.fireflies = ⟨74, 7, 3⟩) :
  final_score game.fireflies - final_score game.hornets = 5 := by
  sorry

#check fireflies_win_by_five

end NUMINAMATH_CALUDE_fireflies_win_by_five_l3489_348900


namespace NUMINAMATH_CALUDE_count_possible_sums_l3489_348946

def bag_A : Finset ℕ := {0, 1, 3, 5}
def bag_B : Finset ℕ := {0, 2, 4, 6}

def possible_sums : Finset ℕ := (bag_A.product bag_B).image (fun p => p.1 + p.2)

theorem count_possible_sums : possible_sums.card = 10 := by
  sorry

end NUMINAMATH_CALUDE_count_possible_sums_l3489_348946


namespace NUMINAMATH_CALUDE_mrsHiltFramePerimeter_l3489_348914

/-- Represents an irregular pentagonal picture frame with given side lengths -/
structure IrregularPentagon where
  base : ℝ
  leftSide : ℝ
  rightSide : ℝ
  topLeftDiagonal : ℝ
  topRightDiagonal : ℝ

/-- Calculates the perimeter of an irregular pentagonal picture frame -/
def perimeter (p : IrregularPentagon) : ℝ :=
  p.base + p.leftSide + p.rightSide + p.topLeftDiagonal + p.topRightDiagonal

/-- Mrs. Hilt's irregular pentagonal picture frame -/
def mrsHiltFrame : IrregularPentagon :=
  { base := 10
    leftSide := 12
    rightSide := 11
    topLeftDiagonal := 6
    topRightDiagonal := 7 }

/-- Theorem: The perimeter of Mrs. Hilt's irregular pentagonal picture frame is 46 inches -/
theorem mrsHiltFramePerimeter : perimeter mrsHiltFrame = 46 := by
  sorry

end NUMINAMATH_CALUDE_mrsHiltFramePerimeter_l3489_348914


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l3489_348942

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  totalChairs : ℕ
  seatedPeople : ℕ

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone. -/
def validSeating (table : CircularTable) : Prop :=
  table.seatedPeople > 0 ∧ 
  table.totalChairs / table.seatedPeople ≤ 4

/-- The theorem stating the smallest number of people that can be seated while satisfying the condition. -/
theorem smallest_valid_seating (table : CircularTable) : 
  table.totalChairs = 72 → 
  (∀ n : ℕ, n < table.seatedPeople → ¬validSeating ⟨table.totalChairs, n⟩) →
  validSeating table →
  table.seatedPeople = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_l3489_348942


namespace NUMINAMATH_CALUDE_conic_eccentricity_l3489_348918

/-- A conic section with foci F₁ and F₂ -/
structure ConicSection where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of a conic section -/
def eccentricity (c : ConicSection) : ℝ := sorry

/-- A point on a conic section -/
def Point (c : ConicSection) := ℝ × ℝ

theorem conic_eccentricity (c : ConicSection) :
  ∃ (P : Point c), 
    distance P c.F₁ / distance c.F₁ c.F₂ = 4/3 ∧
    distance c.F₁ c.F₂ / distance P c.F₂ = 3/2 →
    eccentricity c = 1/2 ∨ eccentricity c = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_conic_eccentricity_l3489_348918


namespace NUMINAMATH_CALUDE_average_weight_problem_l3489_348912

theorem average_weight_problem (a b c : ℝ) :
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  b = 37 →
  (b + c) / 2 = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l3489_348912


namespace NUMINAMATH_CALUDE_james_new_hourly_wage_l3489_348931

/-- Jame's hourly wage calculation --/
theorem james_new_hourly_wage :
  ∀ (new_hours_per_week old_hours_per_week old_hourly_wage : ℕ)
    (weeks_per_year : ℕ) (yearly_increase : ℕ),
  new_hours_per_week = 40 →
  old_hours_per_week = 25 →
  old_hourly_wage = 16 →
  weeks_per_year = 52 →
  yearly_increase = 20800 →
  ∃ (new_hourly_wage : ℕ),
    new_hourly_wage = 530 ∧
    new_hourly_wage * new_hours_per_week * weeks_per_year =
      old_hourly_wage * old_hours_per_week * weeks_per_year + yearly_increase :=
by
  sorry

end NUMINAMATH_CALUDE_james_new_hourly_wage_l3489_348931


namespace NUMINAMATH_CALUDE_F_neg_one_eq_zero_l3489_348937

noncomputable def F (x : ℝ) : ℝ := Real.sqrt (abs (x + 1)) + (9 / Real.pi) * Real.arctan (Real.sqrt (abs (x + 1)))

theorem F_neg_one_eq_zero : F (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_F_neg_one_eq_zero_l3489_348937
