import Mathlib

namespace NUMINAMATH_GPT_plantable_area_l446_44695

noncomputable def flowerbed_r := 10
noncomputable def path_w := 4
noncomputable def full_area := 100 * Real.pi
noncomputable def segment_area := 20.67 * Real.pi * 2 -- each path affects two segments

theorem plantable_area :
  full_area - segment_area = 58.66 * Real.pi := 
by sorry

end NUMINAMATH_GPT_plantable_area_l446_44695


namespace NUMINAMATH_GPT_quadratic_inverse_sum_roots_l446_44687

theorem quadratic_inverse_sum_roots (x1 x2 : ℝ) (h1 : x1^2 - 2023 * x1 + 1 = 0) (h2 : x2^2 - 2023 * x2 + 1 = 0) : 
  (1/x1 + 1/x2) = 2023 :=
by
  -- We outline the proof steps that should be accomplished.
  -- These will be placeholders and not part of the actual statement.
  -- sorry allows us to skip the proof.
  sorry

end NUMINAMATH_GPT_quadratic_inverse_sum_roots_l446_44687


namespace NUMINAMATH_GPT_repeating_decimal_product_l446_44659

noncomputable def x : ℚ := 1 / 33
noncomputable def y : ℚ := 1 / 3

theorem repeating_decimal_product :
  (x * y) = 1 / 99 :=
by
  -- Definitions of x and y
  sorry

end NUMINAMATH_GPT_repeating_decimal_product_l446_44659


namespace NUMINAMATH_GPT_max_gold_coins_l446_44638

theorem max_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 100) : n = 94 := by
  sorry

end NUMINAMATH_GPT_max_gold_coins_l446_44638


namespace NUMINAMATH_GPT_total_books_l446_44661

def shelves : ℕ := 150
def books_per_shelf : ℕ := 15

theorem total_books (shelves books_per_shelf : ℕ) : shelves * books_per_shelf = 2250 := by
  sorry

end NUMINAMATH_GPT_total_books_l446_44661


namespace NUMINAMATH_GPT_base_16_zeros_in_15_factorial_l446_44699

-- Definition of the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definition of the power function to generalize \( a^b \)
def power (a b : ℕ) : ℕ :=
  if b = 0 then 1 else a * power a (b - 1)

-- The constraints of the problem
def k_zeros_base_16 (n : ℕ) (k : ℕ) : Prop :=
  ∃ p, factorial n = p * power 16 k ∧ ¬ (∃ q, factorial n = q * power 16 (k + 1))

-- The main theorem we want to prove
theorem base_16_zeros_in_15_factorial : ∃ k, k_zeros_base_16 15 k ∧ k = 3 :=
by 
  sorry -- Proof to be found

end NUMINAMATH_GPT_base_16_zeros_in_15_factorial_l446_44699


namespace NUMINAMATH_GPT_equivalent_contrapositive_l446_44617

-- Given definitions
variables {Person : Type} (possess : Person → Prop) (happy : Person → Prop)

-- The original statement: "If someone is happy, then they possess it."
def original_statement : Prop := ∀ p : Person, happy p → possess p

-- The contrapositive: "If someone does not possess it, then they are not happy."
def contrapositive_statement : Prop := ∀ p : Person, ¬ possess p → ¬ happy p

-- The theorem to prove logical equivalence
theorem equivalent_contrapositive : original_statement possess happy ↔ contrapositive_statement possess happy := 
by sorry

end NUMINAMATH_GPT_equivalent_contrapositive_l446_44617


namespace NUMINAMATH_GPT_no_rational_roots_l446_44683

theorem no_rational_roots {p q : ℤ} (hp : p % 2 = 1) (hq : q % 2 = 1) :
  ¬ ∃ x : ℚ, x^2 + (2 * p) * x + (2 * q) = 0 :=
by
  -- proof using contradiction technique
  sorry

end NUMINAMATH_GPT_no_rational_roots_l446_44683


namespace NUMINAMATH_GPT_fraction_computation_l446_44663

theorem fraction_computation :
  ((11^4 + 324) * (23^4 + 324) * (35^4 + 324) * (47^4 + 324) * (59^4 + 324)) / 
  ((5^4 + 324) * (17^4 + 324) * (29^4 + 324) * (41^4 + 324) * (53^4 + 324)) = 295.615 := 
sorry

end NUMINAMATH_GPT_fraction_computation_l446_44663


namespace NUMINAMATH_GPT_arithmetic_seq_a₄_l446_44614

-- Definitions for conditions in the given problem
def S₅ (a₁ a₅ : ℕ) : ℕ := ((a₁ + a₅) * 5) / 2
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Final proof statement to show that a₄ = 9
theorem arithmetic_seq_a₄ (a₁ a₅ : ℕ) (d : ℕ) (h₁ : S₅ a₁ a₅ = 35) (h₂ : a₅ = 11) (h₃ : d = (a₅ - a₁) / 4) :
  arithmetic_sequence a₁ d 4 = 9 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_a₄_l446_44614


namespace NUMINAMATH_GPT_production_volume_bounds_l446_44604

theorem production_volume_bounds:
  ∀ (x : ℕ),
  (10 * x ≤ 800 * 2400) ∧ 
  (10 * x ≤ 4000000 + 16000000) ∧
  (x ≥ 1800000) →
  (1800000 ≤ x ∧ x ≤ 1920000) :=
by
  sorry

end NUMINAMATH_GPT_production_volume_bounds_l446_44604


namespace NUMINAMATH_GPT_tomato_plants_count_l446_44648

theorem tomato_plants_count :
  ∀ (sunflowers corn tomatoes total_rows plants_per_row : ℕ),
  sunflowers = 45 →
  corn = 81 →
  plants_per_row = 9 →
  total_rows = (sunflowers / plants_per_row) + (corn / plants_per_row) →
  tomatoes = total_rows * plants_per_row →
  tomatoes = 126 :=
by
  intros sunflowers corn tomatoes total_rows plants_per_row Hs Hc Hp Ht Hm
  rw [Hs, Hc, Hp] at *
  -- Additional calculation steps could go here to prove the theorem if needed
  sorry

end NUMINAMATH_GPT_tomato_plants_count_l446_44648


namespace NUMINAMATH_GPT_repetend_of_five_over_eleven_l446_44653

noncomputable def repetend_of_decimal_expansion (n d : ℕ) : ℕ := sorry

theorem repetend_of_five_over_eleven : repetend_of_decimal_expansion 5 11 = 45 :=
by sorry

end NUMINAMATH_GPT_repetend_of_five_over_eleven_l446_44653


namespace NUMINAMATH_GPT_hexagon_transformation_l446_44625

-- Define a shape composed of 36 identical small equilateral triangles
def Shape := { s : ℕ // s = 36 }

-- Define the number of triangles needed to form a hexagon
def TrianglesNeededForHexagon : ℕ := 18

-- Proof statement: Given a shape of 36 small triangles, we need 18 more triangles to form a hexagon
theorem hexagon_transformation (shape : Shape) : TrianglesNeededForHexagon = 18 :=
by
  -- This is our formalization of the problem statement which asserts
  -- that the transformation to a hexagon needs exactly 18 additional triangles.
  sorry

end NUMINAMATH_GPT_hexagon_transformation_l446_44625


namespace NUMINAMATH_GPT_sequence_arith_l446_44606

theorem sequence_arith {a : ℕ → ℕ} (h_initial : a 2 = 2) (h_recursive : ∀ n ≥ 2, a (n + 1) = a n + 1) :
  ∀ n ≥ 2, a n = n :=
by
  sorry

end NUMINAMATH_GPT_sequence_arith_l446_44606


namespace NUMINAMATH_GPT_total_animals_l446_44628

theorem total_animals (initial_elephants initial_hippos : ℕ) 
  (ratio_female_hippos : ℚ)
  (births_per_female_hippo : ℕ)
  (newborn_elephants_diff : ℕ)
  (he : initial_elephants = 20)
  (hh : initial_hippos = 35)
  (rfh : ratio_female_hippos = 5 / 7)
  (bpfh : births_per_female_hippo = 5)
  (ned : newborn_elephants_diff = 10) :
  ∃ (total_animals : ℕ), total_animals = 315 :=
by sorry

end NUMINAMATH_GPT_total_animals_l446_44628


namespace NUMINAMATH_GPT_diagonals_in_nonagon_l446_44684

-- Define the properties of the polygon
def convex : Prop := true
def sides (n : ℕ) : Prop := n = 9
def right_angles (count : ℕ) : Prop := count = 2

-- Define the formula for the number of diagonals in a polygon with 'n' sides
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The theorem definition
theorem diagonals_in_nonagon :
  convex →
  (sides 9) →
  (right_angles 2) →
  number_of_diagonals 9 = 27 :=
by
  sorry

end NUMINAMATH_GPT_diagonals_in_nonagon_l446_44684


namespace NUMINAMATH_GPT_sampling_methods_l446_44602
-- Import the necessary library

-- Definitions for the conditions of the problem:
def NumberOfFamilies := 500
def HighIncomeFamilies := 125
def MiddleIncomeFamilies := 280
def LowIncomeFamilies := 95
def SampleSize := 100

def FemaleStudentAthletes := 12
def NumberToChoose := 3

-- Define the appropriate sampling methods
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

-- Stating the proof problem in Lean 4
theorem sampling_methods :
  SamplingMethod.Stratified = SamplingMethod.Stratified ∧
  SamplingMethod.SimpleRandom = SamplingMethod.SimpleRandom :=
by
  -- Proof is omitted in this theorem statement
  sorry

end NUMINAMATH_GPT_sampling_methods_l446_44602


namespace NUMINAMATH_GPT_twentieth_common_number_l446_44693

theorem twentieth_common_number : 
  (∃ (m n : ℤ), (4 * m - 1) = (3 * n + 2) ∧ 20 * 12 - 1 = 239) := 
by
  sorry

end NUMINAMATH_GPT_twentieth_common_number_l446_44693


namespace NUMINAMATH_GPT_ellipse_min_area_contains_circles_l446_44647

-- Define the ellipse and circles
def ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 9) = 1
def circle1 (x y : ℝ) := ((x - 2)^2 + y^2 = 4)
def circle2 (x y : ℝ) := ((x + 2)^2 + y^2 = 4)

-- Proof statement: The smallest possible area of the ellipse containing the circles
theorem ellipse_min_area_contains_circles : 
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), 
    (circle1 x y → ellipse x y) ∧ 
    (circle2 x y → ellipse x y)) ∧
  (k = 12) := 
sorry

end NUMINAMATH_GPT_ellipse_min_area_contains_circles_l446_44647


namespace NUMINAMATH_GPT_fraction_of_termite_ridden_homes_collapsing_l446_44685

variable (T : ℕ) -- T represents the total number of homes
variable (termiteRiddenFraction : ℚ := 1/3) -- Fraction of homes that are termite-ridden
variable (termiteRiddenNotCollapsingFraction : ℚ := 1/7) -- Fraction of homes that are termite-ridden but not collapsing

theorem fraction_of_termite_ridden_homes_collapsing :
  termiteRiddenFraction - termiteRiddenNotCollapsingFraction = 4/21 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fraction_of_termite_ridden_homes_collapsing_l446_44685


namespace NUMINAMATH_GPT_pipe_R_fill_time_l446_44676

theorem pipe_R_fill_time (P_rate Q_rate combined_rate : ℝ) (hP : P_rate = 1 / 2) (hQ : Q_rate = 1 / 4)
  (h_combined : combined_rate = 1 / 1.2) : (∃ R_rate : ℝ, R_rate = 1 / 12) :=
by
  sorry

end NUMINAMATH_GPT_pipe_R_fill_time_l446_44676


namespace NUMINAMATH_GPT_expression_divisible_by_25_l446_44677

theorem expression_divisible_by_25 (n : ℕ) : 
    (2^(n+2) * 3^n + 5 * n - 4) % 25 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_expression_divisible_by_25_l446_44677


namespace NUMINAMATH_GPT_bill_annual_healthcare_cost_l446_44694

def hourly_wage := 25
def weekly_hours := 30
def weeks_per_month := 4
def months_per_year := 12
def normal_monthly_price := 500
def annual_income := hourly_wage * weekly_hours * weeks_per_month * months_per_year
def subsidy (income : ℕ) : ℕ :=
  if income < 10000 then 90
  else if income ≤ 40000 then 50
  else if income > 50000 then 20
  else 0
def monthly_cost_after_subsidy := (normal_monthly_price * (100 - subsidy annual_income)) / 100
def annual_cost := monthly_cost_after_subsidy * months_per_year

theorem bill_annual_healthcare_cost : annual_cost = 3000 := by
  sorry

end NUMINAMATH_GPT_bill_annual_healthcare_cost_l446_44694


namespace NUMINAMATH_GPT_five_g_speeds_l446_44696

theorem five_g_speeds (m : ℝ) :
  (1400 / 50) - (1400 / (50 * m)) = 24 → m = 7 :=
by
  sorry

end NUMINAMATH_GPT_five_g_speeds_l446_44696


namespace NUMINAMATH_GPT_stratified_sampling_third_grade_students_l446_44652

variable (total_students : ℕ) (second_year_female_probability : ℚ) (sample_size : ℕ)

theorem stratified_sampling_third_grade_students
  (h_total : total_students = 2000)
  (h_probability : second_year_female_probability = 0.19)
  (h_sample_size : sample_size = 64) :
  let sampling_fraction := 64 / 2000
  let third_grade_students := 2000 * sampling_fraction
  third_grade_students = 16 :=
by
  -- the proof would go here, but we're skipping it per instructions
  sorry

end NUMINAMATH_GPT_stratified_sampling_third_grade_students_l446_44652


namespace NUMINAMATH_GPT_expand_product_l446_44601

theorem expand_product (x : ℝ) : 
  5 * (x + 6) * (x^2 + 2 * x + 3) = 5 * x^3 + 40 * x^2 + 75 * x + 90 := 
by 
  sorry

end NUMINAMATH_GPT_expand_product_l446_44601


namespace NUMINAMATH_GPT_parabola_vertex_on_x_axis_l446_44664

theorem parabola_vertex_on_x_axis (c : ℝ) : 
    (∃ h k, h = -3 ∧ k = 0 ∧ ∀ x, x^2 + 6 * x + c = x^2 + 6 * x + (c - (h^2)/4)) → c = 9 :=
by
    sorry

end NUMINAMATH_GPT_parabola_vertex_on_x_axis_l446_44664


namespace NUMINAMATH_GPT_number_of_students_selected_from_school2_l446_44633

-- Definitions from conditions
def total_students : ℕ := 360
def students_school1 : ℕ := 123
def students_school2 : ℕ := 123
def students_school3 : ℕ := 114
def selected_students : ℕ := 60
def initial_selected_from_school1 : ℕ := 1 -- Student 002 is already selected

-- Proportion calculation
def remaining_selected_students : ℕ := selected_students - initial_selected_from_school1
def remaining_students : ℕ := total_students - initial_selected_from_school1

-- Placeholder for calculation used in the proof
def students_selected_from_school2 : ℕ := 20

-- The Lean proof statement
theorem number_of_students_selected_from_school2 :
  students_selected_from_school2 =
  Nat.ceil ((students_school2 * remaining_selected_students : ℚ) / remaining_students) :=
sorry

end NUMINAMATH_GPT_number_of_students_selected_from_school2_l446_44633


namespace NUMINAMATH_GPT_max_n_value_l446_44671

theorem max_n_value (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (9 / (a - d)) :=
by
  sorry

end NUMINAMATH_GPT_max_n_value_l446_44671


namespace NUMINAMATH_GPT_rain_probability_tel_aviv_l446_44682

open scoped Classical

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv :
  binomial_probability 6 4 0.5 = 0.234375 :=
by 
  sorry

end NUMINAMATH_GPT_rain_probability_tel_aviv_l446_44682


namespace NUMINAMATH_GPT_dan_money_left_l446_44626

theorem dan_money_left
  (initial_amount : ℝ := 45)
  (cost_per_candy_bar : ℝ := 4)
  (num_candy_bars : ℕ := 4)
  (price_toy_car : ℝ := 15)
  (discount_rate_toy_car : ℝ := 0.10)
  (sales_tax_rate : ℝ := 0.05) :
  initial_amount - ((num_candy_bars * cost_per_candy_bar) + ((price_toy_car - (price_toy_car * discount_rate_toy_car)) * (1 + sales_tax_rate))) = 14.02 :=
by
  sorry

end NUMINAMATH_GPT_dan_money_left_l446_44626


namespace NUMINAMATH_GPT_Bennett_has_6_brothers_l446_44639

theorem Bennett_has_6_brothers (num_aaron_brothers : ℕ) (num_bennett_brothers : ℕ) 
  (h1 : num_aaron_brothers = 4) 
  (h2 : num_bennett_brothers = 2 * num_aaron_brothers - 2) : 
  num_bennett_brothers = 6 := by
  sorry

end NUMINAMATH_GPT_Bennett_has_6_brothers_l446_44639


namespace NUMINAMATH_GPT_ratio_sum_of_squares_l446_44622

theorem ratio_sum_of_squares (a b c : ℕ) (h : a = 6 ∧ b = 1 ∧ c = 7 ∧ 72 / 98 = (a * (b.sqrt^2)).sqrt / c) : a + b + c = 14 := by 
  sorry

end NUMINAMATH_GPT_ratio_sum_of_squares_l446_44622


namespace NUMINAMATH_GPT_single_elimination_games_l446_44610

theorem single_elimination_games (n : ℕ) (h : n = 23) : 
  ∃ g : ℕ, g = n - 1 :=
by
  sorry

end NUMINAMATH_GPT_single_elimination_games_l446_44610


namespace NUMINAMATH_GPT_tuples_satisfy_equation_l446_44673

theorem tuples_satisfy_equation (a b c : ℤ) :
  (a - b)^3 * (a + b)^2 = c^2 + 2 * (a - b) + 1 ↔ (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = 0 ∧ c = 0) :=
sorry

end NUMINAMATH_GPT_tuples_satisfy_equation_l446_44673


namespace NUMINAMATH_GPT_no_integer_n_exists_l446_44616

theorem no_integer_n_exists (n : ℤ) : ¬(∃ n : ℤ, ∃ k : ℤ, ∃ m : ℤ, (n - 6) = 15 * k ∧ (n - 5) = 24 * m) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_n_exists_l446_44616


namespace NUMINAMATH_GPT_integer_pairs_prime_P_l446_44621

theorem integer_pairs_prime_P (P : ℕ) (hP_prime : Prime P) 
  (h_condition : ∃ a b : ℤ, |a + b| + (a - b)^2 = P) : 
  P = 2 ∧ ((∃ a b : ℤ, |a + b| = 2 ∧ a - b = 0) ∨ 
           (∃ a b : ℤ, |a + b| = 1 ∧ (a - b = 1 ∨ a - b = -1))) :=
by
  sorry

end NUMINAMATH_GPT_integer_pairs_prime_P_l446_44621


namespace NUMINAMATH_GPT_opposite_sides_line_l446_44670

theorem opposite_sides_line (a : ℝ) : (0 + 0 - a) * (1 + 1 - a) < 0 → 0 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_GPT_opposite_sides_line_l446_44670


namespace NUMINAMATH_GPT_smallest_solution_is_9_l446_44662

noncomputable def smallest_positive_solution (x : ℝ) : Prop :=
  (3*x / (x - 3) + (3*x^2 - 45) / (x + 3) = 14) ∧ (x > 3) ∧ (∀ y : ℝ, (3*y / (y - 3) + (3*y^2 - 45) / (y + 3) = 14) → (y > 3) → (y ≥ 9))

theorem smallest_solution_is_9 : ∃ x : ℝ, smallest_positive_solution x ∧ x = 9 :=
by
  exists 9
  have : smallest_positive_solution 9 := sorry
  exact ⟨this, rfl⟩

end NUMINAMATH_GPT_smallest_solution_is_9_l446_44662


namespace NUMINAMATH_GPT_carlson_max_jars_l446_44656

theorem carlson_max_jars (n a : ℕ) (h1 : 13 * n - a = 8 * (n + a)) : 
  ∃ (k : ℕ), a = 5 * k ∧ n = 9 * k ∧ 13 * n = 117 * k ∧ 23 ≤ 13 * k := by {
  sorry
}

end NUMINAMATH_GPT_carlson_max_jars_l446_44656


namespace NUMINAMATH_GPT_ninety_eight_squared_l446_44667

theorem ninety_eight_squared : 98^2 = 9604 :=
by 
  -- The proof steps are omitted and replaced with 'sorry'
  sorry

end NUMINAMATH_GPT_ninety_eight_squared_l446_44667


namespace NUMINAMATH_GPT_num_friends_bought_robots_l446_44698

def robot_cost : Real := 8.75
def tax_charged : Real := 7.22
def change_left : Real := 11.53
def initial_amount : Real := 80.0
def friends_bought_robots : Nat := 7

theorem num_friends_bought_robots :
  (initial_amount - (change_left + tax_charged)) / robot_cost = friends_bought_robots := sorry

end NUMINAMATH_GPT_num_friends_bought_robots_l446_44698


namespace NUMINAMATH_GPT_rowing_distance_l446_44668

theorem rowing_distance :
  let row_speed := 4 -- kmph
  let river_speed := 2 -- kmph
  let total_time := 1.5 -- hours
  ∃ d, 
    let downstream_speed := row_speed + river_speed
    let upstream_speed := row_speed - river_speed
    let downstream_time := d / downstream_speed
    let upstream_time := d / upstream_speed
    downstream_time + upstream_time = total_time ∧ d = 2.25 :=
by
  sorry

end NUMINAMATH_GPT_rowing_distance_l446_44668


namespace NUMINAMATH_GPT_polygon_sides_and_diagonals_l446_44613

theorem polygon_sides_and_diagonals (n : ℕ) (h : (n-2) * 180 / 360 = 13 / 2) : 
  n = 15 ∧ (n * (n - 3) / 2 = 90) :=
by {
  sorry
}

end NUMINAMATH_GPT_polygon_sides_and_diagonals_l446_44613


namespace NUMINAMATH_GPT_set_elements_l446_44697

def is_divisor (a b : ℤ) : Prop := ∃ k : ℤ, b = k * a

theorem set_elements:
  {x : ℤ | ∃ d : ℤ, is_divisor d 12 ∧ d = 6 - x ∧ x ≥ 0} = 
  {0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 18} :=
by {
  sorry
}

end NUMINAMATH_GPT_set_elements_l446_44697


namespace NUMINAMATH_GPT_find_k_l446_44690

theorem find_k (k : ℕ) : 5 ^ k = 5 * 25^2 * 125^3 → k = 14 := by
  sorry

end NUMINAMATH_GPT_find_k_l446_44690


namespace NUMINAMATH_GPT_positive_number_decreased_by_4_is_21_times_reciprocal_l446_44688

theorem positive_number_decreased_by_4_is_21_times_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x - 4 = 21 * (1 / x)) : x = 7 := 
sorry

end NUMINAMATH_GPT_positive_number_decreased_by_4_is_21_times_reciprocal_l446_44688


namespace NUMINAMATH_GPT_max_value_of_expression_l446_44643

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x^2 - 2 * x * y + y^2 = 6) :
  ∃ (a b c d : ℕ), (a + b * Real.sqrt c) / d = 9 + 3 * Real.sqrt 3 ∧ a + b + c + d = 16 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l446_44643


namespace NUMINAMATH_GPT_pyramid_height_l446_44691

-- Define the heights of individual blocks and the structure of the pyramid.
def block_height := 10 -- in centimeters
def num_layers := 3

-- Define the total height of the pyramid as the sum of the heights of all blocks.
def total_height (block_height : Nat) (num_layers : Nat) := block_height * num_layers

-- The theorem stating that the total height of the stack is 30 cm given the conditions.
theorem pyramid_height : total_height block_height num_layers = 30 := by
  sorry

end NUMINAMATH_GPT_pyramid_height_l446_44691


namespace NUMINAMATH_GPT_retailer_markup_percentage_l446_44634

-- Definitions of initial conditions
def CP : ℝ := 100
def intended_profit_percentage : ℝ := 0.25
def discount_percentage : ℝ := 0.25
def actual_profit_percentage : ℝ := 0.2375

-- Proving the retailer marked his goods at 65% above the cost price
theorem retailer_markup_percentage : ∃ (MP : ℝ), ((0.75 * MP - CP) / CP) * 100 = actual_profit_percentage * 100 ∧ ((MP - CP) / CP) * 100 = 65 := 
by
  -- The mathematical proof steps mean to be filled here  
  sorry

end NUMINAMATH_GPT_retailer_markup_percentage_l446_44634


namespace NUMINAMATH_GPT_andre_tuesday_ladybugs_l446_44631

theorem andre_tuesday_ladybugs (M T : ℕ) (dots_per_ladybug total_dots monday_dots tuesday_dots : ℕ)
  (h1 : M = 8)
  (h2 : dots_per_ladybug = 6)
  (h3 : total_dots = 78)
  (h4 : monday_dots = M * dots_per_ladybug)
  (h5 : tuesday_dots = total_dots - monday_dots)
  (h6 : tuesday_dots = T * dots_per_ladybug) :
  T = 5 :=
sorry

end NUMINAMATH_GPT_andre_tuesday_ladybugs_l446_44631


namespace NUMINAMATH_GPT_sum_infinite_series_l446_44651

theorem sum_infinite_series : ∑' k : ℕ, (k^2 : ℝ) / (3^k) = 7 / 8 :=
sorry

end NUMINAMATH_GPT_sum_infinite_series_l446_44651


namespace NUMINAMATH_GPT_initial_men_in_camp_l446_44609

theorem initial_men_in_camp (M F : ℕ) 
  (h1 : F = M * 50)
  (h2 : F = (M + 10) * 25) : 
  M = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_men_in_camp_l446_44609


namespace NUMINAMATH_GPT_probability_at_least_one_first_class_part_l446_44692

-- Define the problem constants
def total_parts : ℕ := 6
def first_class_parts : ℕ := 4
def second_class_parts : ℕ := 2
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Define the target probability
def target_probability : ℚ := 14 / 15

-- Statement of the problem as a Lean theorem
theorem probability_at_least_one_first_class_part :
  (1 - (choose second_class_parts 2 : ℚ) / (choose total_parts 2 : ℚ)) = target_probability :=
by
  -- the proof is omitted
  sorry

end NUMINAMATH_GPT_probability_at_least_one_first_class_part_l446_44692


namespace NUMINAMATH_GPT_find_number_l446_44605

theorem find_number (x : ℝ) : (x * 12) / (180 / 3) + 80 = 81 → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l446_44605


namespace NUMINAMATH_GPT_theta_quadrant_l446_44665

theorem theta_quadrant (θ : ℝ) (h : Real.sin (2 * θ) < 0) : 
  (Real.sin θ < 0 ∧ Real.cos θ > 0) ∨ (Real.sin θ > 0 ∧ Real.cos θ < 0) :=
sorry

end NUMINAMATH_GPT_theta_quadrant_l446_44665


namespace NUMINAMATH_GPT_find_sum_x_y_l446_44658

theorem find_sum_x_y (x y : ℝ) 
  (h1 : x^3 - 3 * x^2 + 2026 * x = 2023)
  (h2 : y^3 + 6 * y^2 + 2035 * y = -4053) : 
  x + y = -1 := 
sorry

end NUMINAMATH_GPT_find_sum_x_y_l446_44658


namespace NUMINAMATH_GPT_tangent_line_parabola_l446_44600

theorem tangent_line_parabola (P : ℝ × ℝ) (hP : P = (-1, 0)) :
  ∀ x y : ℝ, (y^2 = 4 * x) ∧ (P = (-1, 0)) → (x + y + 1 = 0) ∨ (x - y + 1 = 0) := by
  sorry

end NUMINAMATH_GPT_tangent_line_parabola_l446_44600


namespace NUMINAMATH_GPT_on_real_axis_in_first_quadrant_on_line_l446_44678

theorem on_real_axis (m : ℝ) : 
  (m = -3 ∨ m = 5) ↔ (m^2 - 2 * m - 15 = 0) := 
sorry

theorem in_first_quadrant (m : ℝ) : 
  (m < -3 ∨ m > 5) ↔ ((m^2 + 5 * m + 6 > 0) ∧ (m^2 - 2 * m - 15 > 0)) := 
sorry

theorem on_line (m : ℝ) : 
  (m = 1 ∨ m = -5 / 2) ↔ ((m^2 + 5 * m + 6) + (m^2 - 2 * m - 15) + 5 = 0) := 
sorry

end NUMINAMATH_GPT_on_real_axis_in_first_quadrant_on_line_l446_44678


namespace NUMINAMATH_GPT_outfit_count_l446_44619

def num_shirts := 8
def num_hats := 8
def num_pants := 4

def shirt_colors := 6
def hat_colors := 6
def pants_colors := 4

def total_possible_outfits := num_shirts * num_hats * num_pants

def same_color_restricted_outfits := 4 * 8 * 7

def num_valid_outfits := total_possible_outfits - same_color_restricted_outfits

theorem outfit_count (h1 : num_shirts = 8) (h2 : num_hats = 8) (h3 : num_pants = 4)
                     (h4 : shirt_colors = 6) (h5 : hat_colors = 6) (h6 : pants_colors = 4)
                     (h7 : total_possible_outfits = 256) (h8 : same_color_restricted_outfits = 224) :
  num_valid_outfits = 32 :=
by
  sorry

end NUMINAMATH_GPT_outfit_count_l446_44619


namespace NUMINAMATH_GPT_compare_neg_rational_numbers_l446_44646

theorem compare_neg_rational_numbers :
  - (3 / 2) > - (5 / 3) := 
sorry

end NUMINAMATH_GPT_compare_neg_rational_numbers_l446_44646


namespace NUMINAMATH_GPT_probability_no_shaded_square_l446_44669

theorem probability_no_shaded_square : 
  let n : ℕ := 502 * 1004
  let m : ℕ := 502^2
  let total_rectangles := 3 * n
  let rectangles_with_shaded := 3 * m
  let probability_includes_shaded := rectangles_with_shaded / total_rectangles
  1 - probability_includes_shaded = (1 : ℚ) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_probability_no_shaded_square_l446_44669


namespace NUMINAMATH_GPT_max_value_of_square_diff_max_value_of_square_diff_achieved_l446_44623

theorem max_value_of_square_diff (a b : ℝ) (h : a^2 + b^2 = 4) : (a - b)^2 ≤ 8 :=
sorry

theorem max_value_of_square_diff_achieved (a b : ℝ) (h : a^2 + b^2 = 4) : ∃ a b : ℝ, (a - b)^2 = 8 :=
sorry

end NUMINAMATH_GPT_max_value_of_square_diff_max_value_of_square_diff_achieved_l446_44623


namespace NUMINAMATH_GPT_factors_of_48_multiples_of_8_l446_44607

theorem factors_of_48_multiples_of_8 : 
  ∃ count : ℕ, count = 4 ∧ (∀ d ∈ {d | d ∣ 48 ∧ (∃ k, d = 8 * k)}, true) :=
by {
  sorry  -- This is a placeholder for the actual proof
}

end NUMINAMATH_GPT_factors_of_48_multiples_of_8_l446_44607


namespace NUMINAMATH_GPT_find_original_number_l446_44666

theorem find_original_number (x : ℝ) (h : 1.125 * x - 0.75 * x = 30) : x = 80 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l446_44666


namespace NUMINAMATH_GPT_find_m_of_equation_has_positive_root_l446_44620

theorem find_m_of_equation_has_positive_root :
  (∃ x : ℝ, 0 < x ∧ (x - 1) / (x - 5) = (m * x) / (10 - 2 * x)) → m = -8 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_m_of_equation_has_positive_root_l446_44620


namespace NUMINAMATH_GPT_missing_digit_in_decimal_representation_of_power_of_two_l446_44672

theorem missing_digit_in_decimal_representation_of_power_of_two :
  (∃ m : ℕ, m < 10 ∧
   ∀ (n : ℕ), (0 ≤ n ∧ n < 10 → n ≠ m) →
     (45 - m) % 9 = (2^29) % 9) :=
sorry

end NUMINAMATH_GPT_missing_digit_in_decimal_representation_of_power_of_two_l446_44672


namespace NUMINAMATH_GPT_fuel_for_empty_plane_per_mile_l446_44657

theorem fuel_for_empty_plane_per_mile :
  let F := 106000 / 400 - (35 * 3 + 70 * 2)
  F = 20 := 
by
  sorry

end NUMINAMATH_GPT_fuel_for_empty_plane_per_mile_l446_44657


namespace NUMINAMATH_GPT_question_proof_l446_44675

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end NUMINAMATH_GPT_question_proof_l446_44675


namespace NUMINAMATH_GPT_fraction_inequality_l446_44644

theorem fraction_inequality (a b m : ℝ) (h1 : b > a) (h2 : m > 0) : 
  (b / a) > ((b + m) / (a + m)) :=
sorry

end NUMINAMATH_GPT_fraction_inequality_l446_44644


namespace NUMINAMATH_GPT_find_n_l446_44615

noncomputable def arithmeticSequenceTerm (a b : ℝ) (n : ℕ) : ℝ :=
  let A := Real.log a
  let B := Real.log b
  6 * B + (n - 1) * 11 * B

theorem find_n 
  (a b : ℝ) 
  (h1 : Real.log (a^2 * b^4) = 2 * Real.log a + 4 * Real.log b)
  (h2 : Real.log (a^6 * b^11) = 6 * Real.log a + 11 * Real.log b)
  (h3 : Real.log (a^12 * b^20) = 12 * Real.log a + 20 * Real.log b) 
  (h_diff : (6 * Real.log a + 11 * Real.log b) - (2 * Real.log a + 4 * Real.log b) = 
            (12 * Real.log a + 20 * Real.log b) - (6 * Real.log a + 11 * Real.log b))
  : ∃ n : ℕ, arithmeticSequenceTerm a b 15 = Real.log (b^n) ∧ n = 160 :=
by
  use 160
  sorry

end NUMINAMATH_GPT_find_n_l446_44615


namespace NUMINAMATH_GPT_correct_answer_l446_44636

def M : Set ℕ := {1}
def N : Set ℕ := {1, 2, 3}

theorem correct_answer : M ⊆ N := by
  sorry

end NUMINAMATH_GPT_correct_answer_l446_44636


namespace NUMINAMATH_GPT_coin_flip_probability_l446_44689

theorem coin_flip_probability :
  let total_flips := 8
  let num_heads := 6
  let total_outcomes := (2: ℝ) ^ total_flips
  let favorable_outcomes := (Nat.choose total_flips num_heads)
  let probability := favorable_outcomes / total_outcomes
  probability = (7 / 64 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_coin_flip_probability_l446_44689


namespace NUMINAMATH_GPT_employees_in_factory_l446_44680

theorem employees_in_factory (initial_total : ℕ) (init_prod : ℕ) (init_admin : ℕ)
  (increase_prod_frac : ℚ) (increase_admin_frac : ℚ) :
  initial_total = 1200 →
  init_prod = 800 →
  init_admin = 400 →
  increase_prod_frac = 0.35 →
  increase_admin_frac = 3 / 5 →
  init_prod + init_prod * increase_prod_frac +
  init_admin + init_admin * increase_admin_frac = 1720 := by
  intros h_total h_prod h_admin h_inc_prod h_inc_admin
  sorry

end NUMINAMATH_GPT_employees_in_factory_l446_44680


namespace NUMINAMATH_GPT_emily_euros_contribution_l446_44649

-- Declare the conditions as a definition
def conditions : Prop :=
  ∃ (cost_of_pie : ℝ) (emily_usd : ℝ) (berengere_euros : ℝ) (exchange_rate : ℝ),
    cost_of_pie = 15 ∧
    emily_usd = 10 ∧
    berengere_euros = 3 ∧
    exchange_rate = 1.1

-- Define the proof problem based on the conditions and required contribution
theorem emily_euros_contribution : conditions → (∃ emily_euros_more : ℝ, emily_euros_more = 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_emily_euros_contribution_l446_44649


namespace NUMINAMATH_GPT_circle_equation_l446_44641

theorem circle_equation
  (a b r : ℝ) 
  (h1 : a^2 + b^2 = r^2) 
  (h2 : (a - 2)^2 + b^2 = r^2) 
  (h3 : b / (a - 2) = 1) : 
  (x - 1)^2 + (y + 1)^2 = 2 := 
by
  sorry

end NUMINAMATH_GPT_circle_equation_l446_44641


namespace NUMINAMATH_GPT_regression_coeff_nonzero_l446_44645

theorem regression_coeff_nonzero (a b r : ℝ) (h : b = 0 → r = 0) : b ≠ 0 :=
sorry

end NUMINAMATH_GPT_regression_coeff_nonzero_l446_44645


namespace NUMINAMATH_GPT_abs_inequality_solution_l446_44630

theorem abs_inequality_solution (x : ℝ) :
  3 ≤ |x + 2| ∧ |x + 2| ≤ 7 ↔ (1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5) :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l446_44630


namespace NUMINAMATH_GPT_find_b_c_l446_44612

theorem find_b_c (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 40) 
  (h2 : a + d = 6)
  (h3 : a * b + b * c + c * d + d * a = 28) : 
  b + c = 17 / 3 := 
by
  sorry

end NUMINAMATH_GPT_find_b_c_l446_44612


namespace NUMINAMATH_GPT_cuboid_edge_sum_l446_44686

-- Define the properties of a cuboid
structure Cuboid (α : Type) [LinearOrderedField α] where
  length : α
  width : α
  height : α

-- Define the volume of a cuboid
def volume {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  c.length * c.width * c.height

-- Define the surface area of a cuboid
def surface_area {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  2 * (c.length * c.width + c.width * c.height + c.height * c.length)

-- Define the sum of all edges of a cuboid
def edge_sum {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  4 * (c.length + c.width + c.height)

-- Given a geometric progression property
def gp_property {α : Type} [LinearOrderedField α] (c : Cuboid α) (q a : α) : Prop :=
  c.length = q * a ∧ c.width = a ∧ c.height = a / q

-- The main problem to be stated in Lean
theorem cuboid_edge_sum (α : Type) [LinearOrderedField α] (c : Cuboid α) (a q : α)
  (h1 : volume c = 8)
  (h2 : surface_area c = 32)
  (h3 : gp_property c q a) :
  edge_sum c = 32 := by
    sorry

end NUMINAMATH_GPT_cuboid_edge_sum_l446_44686


namespace NUMINAMATH_GPT_distinct_real_roots_iff_l446_44627

theorem distinct_real_roots_iff (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (∀ x, x^2 + 3 * x - a = 0 → (x = x₁ ∨ x = x₂))) ↔ a > - (9 : ℝ) / 4 :=
sorry

end NUMINAMATH_GPT_distinct_real_roots_iff_l446_44627


namespace NUMINAMATH_GPT_original_cost_price_l446_44608

-- Define the conditions
def selling_price : ℝ := 24000
def discount_rate : ℝ := 0.15
def tax_rate : ℝ := 0.02
def profit_rate : ℝ := 0.12

-- Define the necessary calculations
def discounted_price (sp : ℝ) (dr : ℝ) : ℝ := sp * (1 - dr)
def total_tax (sp : ℝ) (tr : ℝ) : ℝ := sp * tr
def profit (c : ℝ) (pr : ℝ) : ℝ := c * (1 + pr)

-- The problem is to prove that the original cost price is $17,785.71
theorem original_cost_price : 
  ∃ (C : ℝ), C = 17785.71 ∧ 
  selling_price * (1 - discount_rate - tax_rate) = (1 + profit_rate) * C :=
sorry

end NUMINAMATH_GPT_original_cost_price_l446_44608


namespace NUMINAMATH_GPT_find_k_l446_44654

theorem find_k (k : ℝ) (h : ∀ x: ℝ, (x = -2) → (1 + k / (x - 1) = 0)) : k = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l446_44654


namespace NUMINAMATH_GPT_fraction_condition_l446_44640

theorem fraction_condition (x : ℝ) (h₁ : x > 1) (h₂ : 1 / x < 1) : false :=
sorry

end NUMINAMATH_GPT_fraction_condition_l446_44640


namespace NUMINAMATH_GPT_find_a_b_l446_44679

noncomputable def f (a b x : ℝ) := b * a^x

def passes_through (a b : ℝ) : Prop :=
  f a b 1 = 27 ∧ f a b (-1) = 3

theorem find_a_b (a b : ℝ) (h : passes_through a b) : 
  a = 3 ∧ b = 9 :=
  sorry

end NUMINAMATH_GPT_find_a_b_l446_44679


namespace NUMINAMATH_GPT_additional_male_students_l446_44681

variable (a : ℕ)

theorem additional_male_students (h : a > 20) : a - 20 = (a - 20) := 
by 
  sorry

end NUMINAMATH_GPT_additional_male_students_l446_44681


namespace NUMINAMATH_GPT_transportation_degrees_l446_44655

theorem transportation_degrees
  (salaries : ℕ) (r_and_d : ℕ) (utilities : ℕ) (equipment : ℕ) (supplies : ℕ) (total_degrees : ℕ)
  (h_salaries : salaries = 60)
  (h_r_and_d : r_and_d = 9)
  (h_utilities : utilities = 5)
  (h_equipment : equipment = 4)
  (h_supplies : supplies = 2)
  (h_total_degrees : total_degrees = 360) :
  (total_degrees * (100 - (salaries + r_and_d + utilities + equipment + supplies)) / 100 = 72) :=
by {
  sorry
}

end NUMINAMATH_GPT_transportation_degrees_l446_44655


namespace NUMINAMATH_GPT_find_first_two_solutions_l446_44624

theorem find_first_two_solutions :
  ∃ (n1 n2 : ℕ), 
    (n1 ≡ 3 [MOD 7]) ∧ (n1 ≡ 4 [MOD 9]) ∧ 
    (n2 ≡ 3 [MOD 7]) ∧ (n2 ≡ 4 [MOD 9]) ∧ 
    n1 < n2 ∧ 
    n1 = 31 ∧ n2 = 94 := 
by 
  sorry

end NUMINAMATH_GPT_find_first_two_solutions_l446_44624


namespace NUMINAMATH_GPT_chef_served_173_guests_l446_44660

noncomputable def total_guests_served : ℕ :=
  let adults := 58
  let children := adults - 35
  let seniors := 2 * children
  let teenagers := seniors - 15
  let toddlers := teenagers / 2
  adults + children + seniors + teenagers + toddlers

theorem chef_served_173_guests : total_guests_served = 173 :=
  by
    -- Proof will be provided here.
    sorry

end NUMINAMATH_GPT_chef_served_173_guests_l446_44660


namespace NUMINAMATH_GPT_parabola_behavior_l446_44674

-- Definitions for the conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- The proof statement
theorem parabola_behavior (a : ℝ) (x : ℝ) (ha : 0 < a) : 
  (0 < a ∧ a < 1 → parabola a x < x^2) ∧
  (a > 1 → parabola a x > x^2) ∧
  (∀ ε > 0, ∃ δ > 0, δ ≤ a → |parabola a x - 0| < ε) := 
sorry

end NUMINAMATH_GPT_parabola_behavior_l446_44674


namespace NUMINAMATH_GPT_tallest_building_height_l446_44629

theorem tallest_building_height :
  ∃ H : ℝ, H + (1/2) * H + (1/4) * H + (1/20) * H = 180 ∧ H = 100 := by
  sorry

end NUMINAMATH_GPT_tallest_building_height_l446_44629


namespace NUMINAMATH_GPT_capital_at_end_of_2014_year_capital_exceeds_32dot5_billion_l446_44635

noncomputable def company_capital (n : ℕ) : ℝ :=
  if n = 0 then 1000
  else 2 * company_capital (n - 1) - 500

theorem capital_at_end_of_2014 : company_capital 4 = 8500 :=
by sorry

theorem year_capital_exceeds_32dot5_billion : ∀ n : ℕ, company_capital n > 32500 → n ≥ 7 :=
by sorry

end NUMINAMATH_GPT_capital_at_end_of_2014_year_capital_exceeds_32dot5_billion_l446_44635


namespace NUMINAMATH_GPT_factorize_expression_l446_44650

theorem factorize_expression : (x^2 + 9)^2 - 36*x^2 = (x + 3)^2 * (x - 3)^2 := 
by 
  sorry

end NUMINAMATH_GPT_factorize_expression_l446_44650


namespace NUMINAMATH_GPT_negation_statement_l446_44611

variable {α : Type} (S : Set α)

theorem negation_statement (P : α → Prop) :
  (∀ x ∈ S, ¬ P x) ↔ (∃ x ∈ S, P x) :=
by
  sorry

end NUMINAMATH_GPT_negation_statement_l446_44611


namespace NUMINAMATH_GPT_dessert_probability_l446_44642

noncomputable def P (e : Prop) : ℝ := sorry

variables (D C : Prop)

theorem dessert_probability 
  (P_D : P D = 0.6)
  (P_D_and_not_C : P (D ∧ ¬C) = 0.12) :
  P (¬ D) = 0.4 :=
by
  -- Proof is skipped using sorry, as instructed.
  sorry

end NUMINAMATH_GPT_dessert_probability_l446_44642


namespace NUMINAMATH_GPT_december_25_is_thursday_l446_44632

theorem december_25_is_thursday (thanksgiving : ℕ) (h : thanksgiving = 27) :
  (∀ n, n % 7 = 0 → n + thanksgiving = 25 → n / 7 = 4) :=
by
  sorry

end NUMINAMATH_GPT_december_25_is_thursday_l446_44632


namespace NUMINAMATH_GPT_fenced_yard_area_l446_44618

theorem fenced_yard_area :
  let yard := 20 * 18
  let cutout1 := 3 * 3
  let cutout2 := 4 * 2
  yard - cutout1 - cutout2 = 343 := by
  let yard := 20 * 18
  let cutout1 := 3 * 3
  let cutout2 := 4 * 2
  have h : yard - cutout1 - cutout2 = 343 := sorry
  exact h

end NUMINAMATH_GPT_fenced_yard_area_l446_44618


namespace NUMINAMATH_GPT_square_mirror_side_length_l446_44603

theorem square_mirror_side_length :
  ∃ (side_length : ℝ),
  let wall_width := 42
  let wall_length := 27.428571428571427
  let wall_area := wall_width * wall_length
  let mirror_area := wall_area / 2
  (side_length * side_length = mirror_area) → side_length = 24 :=
by
  use 24
  intro h
  sorry

end NUMINAMATH_GPT_square_mirror_side_length_l446_44603


namespace NUMINAMATH_GPT_cubics_of_sum_and_product_l446_44637

theorem cubics_of_sum_and_product (x y : ℝ) (h₁ : x + y = 10) (h₂ : x * y = 11) : 
  x^3 + y^3 = 670 :=
by
  sorry

end NUMINAMATH_GPT_cubics_of_sum_and_product_l446_44637
