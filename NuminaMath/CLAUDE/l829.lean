import Mathlib

namespace arithmetic_progression_equality_l829_82915

/-- An arithmetic progression with first term and difference as natural numbers -/
structure ArithmeticProgression :=
  (first : ℕ)
  (diff : ℕ)
  (coprime : Nat.Coprime first diff)

/-- The nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℕ :=
  ap.first + (n - 1) * ap.diff

theorem arithmetic_progression_equality (ap1 ap2 : ArithmeticProgression) :
  (∀ n : ℕ, 
    (ArithmeticProgression.nthTerm ap1 n ^ 2 + ArithmeticProgression.nthTerm ap1 (n + 1) ^ 2) *
    (ArithmeticProgression.nthTerm ap2 n ^ 2 + ArithmeticProgression.nthTerm ap2 (n + 1) ^ 2) = m ^ 2 ∨
    (ArithmeticProgression.nthTerm ap1 n ^ 2 + ArithmeticProgression.nthTerm ap2 n ^ 2) *
    (ArithmeticProgression.nthTerm ap1 (n + 1) ^ 2 + ArithmeticProgression.nthTerm ap2 (n + 1) ^ 2) = k ^ 2) →
  ∀ n : ℕ, ArithmeticProgression.nthTerm ap1 n = ArithmeticProgression.nthTerm ap2 n :=
by sorry

end arithmetic_progression_equality_l829_82915


namespace adjacent_integers_product_l829_82969

theorem adjacent_integers_product (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 740 → (x - 1) * x * (x + 1) = 17550 := by
  sorry

end adjacent_integers_product_l829_82969


namespace sqrt_product_l829_82962

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end sqrt_product_l829_82962


namespace non_negative_integer_solutions_of_inequality_l829_82925

theorem non_negative_integer_solutions_of_inequality :
  {x : ℕ | x + 1 < 4} = {0, 1, 2} := by
  sorry

end non_negative_integer_solutions_of_inequality_l829_82925


namespace student_average_score_l829_82947

theorem student_average_score (math physics chem : ℕ) : 
  math + physics = 32 →
  chem = physics + 20 →
  (math + chem) / 2 = 26 := by
sorry

end student_average_score_l829_82947


namespace complex_modulus_problem_l829_82973

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + 2 * Complex.I) = 3 - Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l829_82973


namespace monotonicity_interval_min_value_on_interval_max_value_on_interval_l829_82995

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem for the interval of monotonicity
theorem monotonicity_interval :
  ∃ (a b : ℝ), a < b ∧ (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∨ (∀ x y, a < x ∧ x < y ∧ y < b → f x > f y) :=
sorry

-- Theorem for the minimum value on the interval [-3, 2]
theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-3) 2 ∧ f x = -18 ∧ ∀ y ∈ Set.Icc (-3) 2, f y ≥ f x :=
sorry

-- Theorem for the maximum value on the interval [-3, 2]
theorem max_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-3) 2 ∧ f x = 2 ∧ ∀ y ∈ Set.Icc (-3) 2, f y ≤ f x :=
sorry

end monotonicity_interval_min_value_on_interval_max_value_on_interval_l829_82995


namespace stratified_sampling_size_l829_82964

theorem stratified_sampling_size (undergrads : ℕ) (masters : ℕ) (doctorates : ℕ) 
  (doctoral_sample : ℕ) (n : ℕ) : 
  undergrads = 12000 →
  masters = 1000 →
  doctorates = 200 →
  doctoral_sample = 20 →
  n = (undergrads + masters + doctorates) * doctoral_sample / doctorates →
  n = 1320 := by
sorry

end stratified_sampling_size_l829_82964


namespace sandbox_capacity_doubled_l829_82921

/-- Represents the dimensions and capacity of a sandbox -/
structure Sandbox where
  length : ℝ
  width : ℝ
  height : ℝ
  capacity : ℝ

/-- Theorem: Doubling the dimensions of a sandbox increases its capacity by a factor of 8 -/
theorem sandbox_capacity_doubled (original : Sandbox) 
  (h_original_capacity : original.capacity = 10) :
  let new_sandbox := Sandbox.mk 
    (2 * original.length) 
    (2 * original.width) 
    (2 * original.height) 
    ((2 * original.length) * (2 * original.width) * (2 * original.height))
  new_sandbox.capacity = 80 := by
  sorry


end sandbox_capacity_doubled_l829_82921


namespace quadratic_function_a_range_l829_82917

/-- The function y = (a + 1)x^2 - 2x + 3 is quadratic with respect to x -/
def is_quadratic (a : ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, y = (a + 1) * x^2 - 2 * x + 3

/-- The range of values for a in the quadratic function y = (a + 1)x^2 - 2x + 3 -/
theorem quadratic_function_a_range :
  ∀ a : ℝ, is_quadratic a ↔ a ≠ -1 :=
sorry

end quadratic_function_a_range_l829_82917


namespace cube_root_of_a_minus_m_l829_82936

theorem cube_root_of_a_minus_m (a m : ℝ) (ha : 0 < a) 
  (h1 : (m + 7)^2 = a) (h2 : (2*m - 1)^2 = a) : 
  (a - m)^(1/3 : ℝ) = 3 := by
  sorry

end cube_root_of_a_minus_m_l829_82936


namespace unique_solution_l829_82942

theorem unique_solution : ∃! x : ℝ, 
  (|x - 3| + |x + 4| < 8) ∧ (x^2 - x - 12 = 0) :=
by
  -- The unique solution is x = -3
  use -3
  constructor
  · -- Prove that x = -3 satisfies both conditions
    constructor
    · -- Prove |(-3) - 3| + |(-3) + 4| < 8
      sorry
    · -- Prove (-3)^2 - (-3) - 12 = 0
      sorry
  · -- Prove that no other value satisfies both conditions
    sorry

#check unique_solution

end unique_solution_l829_82942


namespace vector_BC_l829_82902

def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (3, 2)
def AC : ℝ × ℝ := (-4, -3)

theorem vector_BC : 
  let C : ℝ × ℝ := (A.1 + AC.1, A.2 + AC.2)
  (C.1 - B.1, C.2 - B.2) = (-7, -4) := by sorry

end vector_BC_l829_82902


namespace pizza_slices_left_over_is_ten_l829_82929

/-- Calculates the number of pizza slices left over given the conditions of the problem. -/
def pizza_slices_left_over : ℕ :=
  let small_pizza_slices : ℕ := 4
  let large_pizza_slices : ℕ := 8
  let small_pizzas_bought : ℕ := 3
  let large_pizzas_bought : ℕ := 2
  let george_slices : ℕ := 3
  let bob_slices : ℕ := george_slices + 1
  let susie_slices : ℕ := bob_slices / 2
  let bill_fred_mark_slices : ℕ := 3 * 3

  let total_slices : ℕ := small_pizza_slices * small_pizzas_bought + large_pizza_slices * large_pizzas_bought
  let total_eaten : ℕ := george_slices + bob_slices + susie_slices + bill_fred_mark_slices

  total_slices - total_eaten

theorem pizza_slices_left_over_is_ten : pizza_slices_left_over = 10 := by
  sorry

end pizza_slices_left_over_is_ten_l829_82929


namespace inequality_iff_in_solution_set_l829_82910

/-- The solution set for the inequality 1/(x(x+2)) - 1/((x+2)(x+3)) < 1/4 -/
def solution_set : Set ℝ :=
  { x | x < -3 ∨ (-2 < x ∧ x < 0) ∨ 1 < x }

/-- The inequality function -/
def inequality (x : ℝ) : Prop :=
  1 / (x * (x + 2)) - 1 / ((x + 2) * (x + 3)) < 1 / 4

theorem inequality_iff_in_solution_set :
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
sorry

end inequality_iff_in_solution_set_l829_82910


namespace fruit_seller_apples_l829_82961

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℝ) * 0.6 = 420 → initial_apples = 700 :=
by
  sorry

end fruit_seller_apples_l829_82961


namespace correct_number_of_fills_l829_82940

/-- The number of times Alice must fill her measuring cup -/
def number_of_fills : ℕ := 12

/-- The amount of sugar Alice needs in cups -/
def sugar_needed : ℚ := 15/4

/-- The capacity of Alice's measuring cup in cups -/
def cup_capacity : ℚ := 1/3

/-- Theorem stating that the number of fills is correct -/
theorem correct_number_of_fills :
  (↑number_of_fills : ℚ) * cup_capacity ≥ sugar_needed ∧
  ((↑number_of_fills - 1 : ℚ) * cup_capacity < sugar_needed) :=
by sorry

end correct_number_of_fills_l829_82940


namespace m_positive_sufficient_not_necessary_for_hyperbola_l829_82982

-- Define a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  m ≠ 0 ∧ ∃ (x y : ℝ), x^2 / m - y^2 / m = 1

-- State the theorem
theorem m_positive_sufficient_not_necessary_for_hyperbola :
  ∃ (m : ℝ), m ≠ 0 ∧
  (∀ (m : ℝ), m > 0 → is_hyperbola m) ∧
  (∃ (m : ℝ), m < 0 ∧ is_hyperbola m) :=
sorry

end m_positive_sufficient_not_necessary_for_hyperbola_l829_82982


namespace fundraising_problem_l829_82992

/-- The number of workers in a fundraising scenario -/
def number_of_workers : ℕ := sorry

/-- The original contribution amount per worker -/
def original_contribution : ℕ := sorry

theorem fundraising_problem :
  (number_of_workers * original_contribution = 300000) ∧
  (number_of_workers * (original_contribution + 50) = 350000) →
  number_of_workers = 1000 := by
  sorry

end fundraising_problem_l829_82992


namespace additional_wolves_in_pack_l829_82967

/-- The number of additional wolves in a pack, given hunting conditions -/
def additional_wolves (hunting_wolves : ℕ) (meat_per_day : ℕ) (hunting_period : ℕ) 
                      (meat_per_deer : ℕ) (deer_per_hunter : ℕ) : ℕ :=
  let total_meat := hunting_wolves * deer_per_hunter * meat_per_deer
  let wolves_fed := total_meat / (meat_per_day * hunting_period)
  wolves_fed - hunting_wolves

theorem additional_wolves_in_pack : 
  additional_wolves 4 8 5 200 1 = 16 := by
  sorry

end additional_wolves_in_pack_l829_82967


namespace intersection_of_A_and_B_l829_82948

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 3 * x + 1 > 0}
def B : Set ℝ := {x : ℝ | |x - 1| < 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo (-1/3 : ℝ) 3 := by sorry

end intersection_of_A_and_B_l829_82948


namespace locus_of_parabola_vertices_l829_82989

/-- The locus of vertices of parabolas y = x^2 + tx + 1 is y = 1 - x^2 -/
theorem locus_of_parabola_vertices :
  ∀ (t : ℝ), 
  let f (x : ℝ) := x^2 + t*x + 1
  let vertex := (- t/2, f (- t/2))
  (vertex.1)^2 + vertex.2 = 1 := by
sorry

end locus_of_parabola_vertices_l829_82989


namespace factorial_squared_gt_power_l829_82909

theorem factorial_squared_gt_power (n : ℕ) (h : n > 2) : (n.factorial ^ 2 : ℕ) > n ^ n := by
  sorry

end factorial_squared_gt_power_l829_82909


namespace bookshop_unsold_percentage_l829_82927

def initial_stock : ℕ := 1200
def sales : List ℕ := [75, 50, 64, 78, 135]

def books_sold (sales : List ℕ) : ℕ := sales.sum

def books_not_sold (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

def percentage_not_sold (initial : ℕ) (not_sold : ℕ) : ℚ :=
  (not_sold : ℚ) / (initial : ℚ) * 100

theorem bookshop_unsold_percentage :
  let sold := books_sold sales
  let not_sold := books_not_sold initial_stock sold
  percentage_not_sold initial_stock not_sold = 66.5 := by
  sorry

end bookshop_unsold_percentage_l829_82927


namespace triangular_array_coin_sum_l829_82978

theorem triangular_array_coin_sum (N : ℕ) : 
  (N * (N + 1)) / 2 = 2016 → (N / 10 + N % 10 = 9) := by
  sorry

end triangular_array_coin_sum_l829_82978


namespace negation_of_forall_positive_l829_82974

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- State the theorem
theorem negation_of_forall_positive (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) := by
  sorry

end negation_of_forall_positive_l829_82974


namespace smallest_x_absolute_value_equation_l829_82923

theorem smallest_x_absolute_value_equation : 
  (∃ x : ℝ, |4*x - 5| = 29) ∧ 
  (∀ x : ℝ, |4*x - 5| = 29 → x ≥ -6) ∧ 
  |4*(-6) - 5| = 29 := by
  sorry

end smallest_x_absolute_value_equation_l829_82923


namespace max_red_socks_l829_82959

theorem max_red_socks (a b : ℕ) : 
  a + b ≤ 1991 →
  (a.choose 2 + b.choose 2 : ℚ) / ((a + b).choose 2) = 1/2 →
  a ≤ 990 :=
sorry

end max_red_socks_l829_82959


namespace cubic_root_sum_square_l829_82918

theorem cubic_root_sum_square (a b c s : ℝ) : 
  (a^3 - 12*a^2 + 14*a - 1 = 0) →
  (b^3 - 12*b^2 + 14*b - 1 = 0) →
  (c^3 - 12*c^2 + 14*c - 1 = 0) →
  (s = Real.sqrt a + Real.sqrt b + Real.sqrt c) →
  (s^4 - 24*s^2 - 10*s = -144) :=
by sorry

end cubic_root_sum_square_l829_82918


namespace triangle_angle_from_complex_trig_l829_82924

theorem triangle_angle_from_complex_trig (A B C : Real) : 
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  A + B + C = π →
  (Complex.exp (I * A)) * (Complex.exp (I * B)) = Complex.exp (I * C) →
  C = π / 4 := by
  sorry

end triangle_angle_from_complex_trig_l829_82924


namespace triangle_area_l829_82968

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) :
  (1/2) * a * b = 180 := by
  sorry

end triangle_area_l829_82968


namespace probability_black_not_white_is_three_fifths_l829_82970

structure Bag where
  total : ℕ
  white : ℕ
  black : ℕ
  red : ℕ

def probability_black_given_not_white (b : Bag) : ℚ :=
  b.black / (b.total - b.white)

theorem probability_black_not_white_is_three_fifths (b : Bag) 
  (h1 : b.total = 10)
  (h2 : b.white = 5)
  (h3 : b.black = 3)
  (h4 : b.red = 2) :
  probability_black_given_not_white b = 3/5 := by
  sorry

end probability_black_not_white_is_three_fifths_l829_82970


namespace chocolate_price_in_first_store_l829_82950

def chocolates_per_week : ℕ := 2
def weeks : ℕ := 3
def promotion_price : ℚ := 2
def savings : ℚ := 6

theorem chocolate_price_in_first_store :
  let total_chocolates := chocolates_per_week * weeks
  let promotion_total := total_chocolates * promotion_price
  let first_store_total := promotion_total + savings
  first_store_total / total_chocolates = 3 := by
sorry

end chocolate_price_in_first_store_l829_82950


namespace total_annual_earnings_l829_82957

/-- Represents the harvest frequency (in months) and sale price for a fruit type -/
structure FruitInfo where
  harvestFrequency : Nat
  salePrice : Nat

/-- Calculates the annual earnings for a single fruit type -/
def annualEarnings (fruit : FruitInfo) : Nat :=
  (12 / fruit.harvestFrequency) * fruit.salePrice

/-- The farm's fruit information -/
def farmFruits : List FruitInfo := [
  ⟨2, 50⟩,  -- Oranges
  ⟨3, 30⟩,  -- Apples
  ⟨4, 45⟩,  -- Peaches
  ⟨6, 70⟩   -- Blackberries
]

/-- Theorem: The total annual earnings from selling the farm's fruits is $695 -/
theorem total_annual_earnings : 
  (farmFruits.map annualEarnings).sum = 695 := by
  sorry

end total_annual_earnings_l829_82957


namespace hurricane_damage_in_cad_l829_82986

/-- Converts American dollars to Canadian dollars given a conversion rate -/
def convert_usd_to_cad (usd : ℝ) (rate : ℝ) : ℝ := usd * rate

/-- The damage caused by the hurricane in American dollars -/
def damage_usd : ℝ := 60000000

/-- The conversion rate from American dollars to Canadian dollars -/
def usd_to_cad_rate : ℝ := 1.25

/-- Theorem stating the equivalent damage in Canadian dollars -/
theorem hurricane_damage_in_cad :
  convert_usd_to_cad damage_usd usd_to_cad_rate = 75000000 := by
  sorry

end hurricane_damage_in_cad_l829_82986


namespace age_ratio_problem_l829_82914

/-- Proves that given the conditions of the age problem, the ratio of Michael's age to Monica's age is 3:5 -/
theorem age_ratio_problem (patrick_age michael_age monica_age : ℕ) : 
  patrick_age * 5 = michael_age * 3 →  -- Patrick and Michael's ages are in ratio 3:5
  patrick_age + michael_age + monica_age = 196 →  -- Sum of ages is 196
  monica_age - patrick_age = 64 →  -- Difference between Monica's and Patrick's ages is 64
  michael_age * 5 = monica_age * 3  -- Conclusion: Michael and Monica's ages are in ratio 3:5
:= by sorry

end age_ratio_problem_l829_82914


namespace binomial_coefficient_sequence_periodic_l829_82932

/-- 
Given positive integers k and m, the sequence of binomial coefficients (n choose k) mod m,
where n ≥ k, is periodic.
-/
theorem binomial_coefficient_sequence_periodic (k m : ℕ+) :
  ∃ (p : ℕ+), ∀ (n : ℕ), n ≥ k →
    (n.choose k : ZMod m) = ((n + p) : ℕ).choose k := by
  sorry

end binomial_coefficient_sequence_periodic_l829_82932


namespace insurance_claims_percentage_l829_82994

theorem insurance_claims_percentage (jan_claims missy_claims : ℕ) 
  (h1 : jan_claims = 20)
  (h2 : missy_claims = 41)
  (h3 : missy_claims = jan_claims + 15 + (jan_claims * 30 / 100)) :
  ∃ (john_claims : ℕ), 
    missy_claims = john_claims + 15 ∧ 
    john_claims = jan_claims + (jan_claims * 30 / 100) := by
  sorry

end insurance_claims_percentage_l829_82994


namespace fifth_term_of_sequence_l829_82979

/-- Given an arithmetic sequence with first term a₁, common difference d, and n-th term aₙ,
    this function calculates the n-th term. -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 7) (h₃ : a₃ = 11) :
  arithmetic_sequence a₁ (a₂ - a₁) 5 = 19 := by
  sorry

#check fifth_term_of_sequence

end fifth_term_of_sequence_l829_82979


namespace distribution_ways_4_5_l829_82955

/-- The number of ways to distribute men and women into groups -/
def distribution_ways (num_men num_women : ℕ) : ℕ :=
  let scenario1 := num_men.choose 1 * num_women.choose 2 * (3 * 2)
  let scenario2 := num_men.choose 2 * num_women.choose 1 * (num_women.choose 2 * 2)
  scenario1 + scenario2

/-- Theorem stating the number of ways to distribute 4 men and 5 women -/
theorem distribution_ways_4_5 :
  distribution_ways 4 5 = 600 := by
  sorry

#eval distribution_ways 4 5

end distribution_ways_4_5_l829_82955


namespace sum_always_negative_l829_82943

/-- The function f(x) = -x - x^3 -/
def f (x : ℝ) : ℝ := -x - x^3

/-- Theorem stating that f(α) + f(β) + f(γ) is always negative under given conditions -/
theorem sum_always_negative (α β γ : ℝ) 
  (h1 : α + β > 0) (h2 : β + γ > 0) (h3 : γ + α > 0) : 
  f α + f β + f γ < 0 := by
  sorry

end sum_always_negative_l829_82943


namespace gcd_count_for_product_360_l829_82952

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ+), S.card = 11 ∧ (∀ d, d ∈ S ↔ ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360)) :=
by sorry

end gcd_count_for_product_360_l829_82952


namespace complement_A_intersection_A_complement_B_l829_82938

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | x > 1}

-- Theorem for the first part
theorem complement_A : Set.compl A = {x : ℝ | x ≥ 2} := by sorry

-- Theorem for the second part
theorem intersection_A_complement_B : A ∩ Set.compl B = {x : ℝ | x ≤ 1} := by sorry

end complement_A_intersection_A_complement_B_l829_82938


namespace finite_solutions_cube_sum_l829_82999

theorem finite_solutions_cube_sum (n : ℕ) : 
  Finite {p : ℤ × ℤ | (p.1 ^ 3 + p.2 ^ 3 : ℤ) = n} := by
  sorry

end finite_solutions_cube_sum_l829_82999


namespace four_Y_three_equals_twentyfive_l829_82953

-- Define the Y operation
def Y (a b : ℝ) : ℝ := (2 * a^2 - 3 * a * b + b^2)^2

-- Theorem statement
theorem four_Y_three_equals_twentyfive : Y 4 3 = 25 := by
  sorry

end four_Y_three_equals_twentyfive_l829_82953


namespace equal_intercept_line_equation_l829_82954

/-- A line passing through (3, 2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The slope of the line
  m : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- The line passes through (3, 2)
  point_condition : 2 = m * 3 + b
  -- The x and y intercepts are equal and non-zero
  intercept_condition : ∃ (a : ℝ), a ≠ 0 ∧ a = b ∧ a = -b/m

/-- The equation of the line with equal intercepts passing through (3, 2) is x + y = 5 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  l.m = -1 ∧ l.b = 5 :=
by sorry

end equal_intercept_line_equation_l829_82954


namespace unique_consecutive_sum_36_l829_82949

/-- A set of consecutive positive integers -/
def ConsecutiveSet (start : ℕ) (length : ℕ) : Set ℕ :=
  {n : ℕ | start ≤ n ∧ n < start + length}

/-- The sum of a set of consecutive positive integers -/
def ConsecutiveSum (start : ℕ) (length : ℕ) : ℕ :=
  (length * (2 * start + length - 1)) / 2

/-- Theorem: There exists exactly one set of consecutive positive integers,
    containing at least two integers, whose sum is 36 -/
theorem unique_consecutive_sum_36 :
  ∃! (start length : ℕ), 
    length ≥ 2 ∧ 
    ConsecutiveSum start length = 36 :=
sorry

end unique_consecutive_sum_36_l829_82949


namespace man_rowing_speed_l829_82945

/-- The speed of the current downstream in kilometers per hour -/
def current_speed : ℝ := 3

/-- The time taken to cover the distance downstream in seconds -/
def time_downstream : ℝ := 9.390553103577801

/-- The distance covered downstream in meters -/
def distance_downstream : ℝ := 60

/-- The speed at which the man can row in still water in kilometers per hour -/
def rowing_speed : ℝ := 20

theorem man_rowing_speed :
  rowing_speed = 
    (distance_downstream / 1000) / (time_downstream / 3600) - current_speed :=
by sorry

end man_rowing_speed_l829_82945


namespace functional_inequality_domain_l829_82912

-- Define the function f
def f (n : ℕ) (x : ℝ) : ℝ := x^n

-- Define the theorem
theorem functional_inequality_domain (n : ℕ) (h_n : n > 1) :
  ∀ x : ℝ, (f n x + f n (1 - x) > 1) ↔ (x < 0 ∨ x > 1) :=
sorry

end functional_inequality_domain_l829_82912


namespace impossible_card_arrangement_l829_82944

/-- Represents the arrangement of cards --/
def CardArrangement := List ℕ

/-- Calculates the sum of spaces between pairs of identical digits --/
def sumOfSpaces (arr : CardArrangement) : ℕ := sorry

/-- Checks if an arrangement is valid according to the problem's conditions --/
def isValidArrangement (arr : CardArrangement) : Prop := sorry

/-- Theorem stating the impossibility of the desired arrangement --/
theorem impossible_card_arrangement : 
  ¬ ∃ (arr : CardArrangement), 
    (arr.length = 20) ∧ 
    (∀ d, (arr.count d = 2) ∨ (arr.count d = 0)) ∧
    (isValidArrangement arr) := by
  sorry

end impossible_card_arrangement_l829_82944


namespace problem_statement_l829_82934

theorem problem_statement (x : ℝ) (h : x + 1/x = 5) :
  (x - 2)^2 + 25/((x - 2)^2) = 11 := by
  sorry

end problem_statement_l829_82934


namespace cubic_polynomial_property_l829_82996

/-- A cubic polynomial with real coefficients. -/
def CubicPolynomial := ℝ → ℝ

/-- The property that a cubic polynomial satisfies the given conditions. -/
def SatisfiesConditions (g : CubicPolynomial) : Prop :=
  ∃ (a b c d : ℝ), 
    (∀ x, g x = a * x^3 + b * x^2 + c * x + d) ∧
    (|g (-2)| = 6) ∧ (|g 0| = 6) ∧ (|g 1| = 6) ∧ (|g 4| = 6)

/-- The theorem stating that if a cubic polynomial satisfies the conditions, then |g(-1)| = 27/2. -/
theorem cubic_polynomial_property (g : CubicPolynomial) 
  (h : SatisfiesConditions g) : |g (-1)| = 27/2 := by
  sorry

end cubic_polynomial_property_l829_82996


namespace algebraic_expression_value_l829_82928

theorem algebraic_expression_value (a b : ℝ) (h : 2 * a - b = 2) :
  4 * a^2 - b^2 - 4 * b = 4 := by
  sorry

end algebraic_expression_value_l829_82928


namespace sameColorPairWithBlueCount_l829_82998

/-- The number of ways to choose a pair of socks of the same color with at least one blue sock -/
def sameColorPairWithBlue (whiteCount brownCount blueCount greenCount : ℕ) : ℕ :=
  Nat.choose blueCount 2

theorem sameColorPairWithBlueCount :
  sameColorPairWithBlue 5 5 5 5 = 10 := by
  sorry

end sameColorPairWithBlueCount_l829_82998


namespace complement_of_M_l829_82939

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- State the theorem
theorem complement_of_M : 
  Set.compl M = {x : ℝ | x < 0 ∨ x > 2} := by sorry

end complement_of_M_l829_82939


namespace quadratic_equation_roots_l829_82983

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m - 2 = 0 ∧ x = -3) → 
  (m = -1 ∧ ∃ y : ℝ, y^2 + 2*y + m - 2 = 0 ∧ y = 1) :=
by sorry

end quadratic_equation_roots_l829_82983


namespace train_speed_calculation_l829_82913

/-- Proves that a train with given length, crossing a bridge of given length in a given time, has a specific speed in km/hr -/
theorem train_speed_calculation (train_length bridge_length : Real) (crossing_time : Real) :
  train_length = 110 →
  bridge_length = 175 →
  crossing_time = 14.248860091192705 →
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l829_82913


namespace circle_square_area_ratio_l829_82965

theorem circle_square_area_ratio :
  ∀ (r : ℝ) (s : ℝ),
  r > 0 →
  s > 0 →
  r = s * (Real.sqrt 2) / 2 →
  (π * r^2) / (s^2) = π / 2 := by
sorry

end circle_square_area_ratio_l829_82965


namespace area_of_divided_square_l829_82920

/-- A square divided into rectangles of equal area with specific properties -/
structure DividedSquare where
  side : ℝ
  segment_AB : ℝ
  is_divided : Bool
  A_is_midpoint : Bool

/-- The area of a DividedSquare with given properties -/
def square_area (s : DividedSquare) : ℝ := s.side ^ 2

/-- Theorem stating the area of the square under given conditions -/
theorem area_of_divided_square (s : DividedSquare) 
  (h1 : s.is_divided = true)
  (h2 : s.segment_AB = 1)
  (h3 : s.A_is_midpoint = true) :
  square_area s = 4 := by
  sorry

end area_of_divided_square_l829_82920


namespace thickness_after_13_folds_l829_82908

/-- The thickness of a paper after n folds, given an initial thickness of a millimeters -/
def paper_thickness (a : ℝ) (n : ℕ) : ℝ :=
  a * 2^n

/-- Theorem: The thickness of a paper after 13 folds is 2^13 times its initial thickness -/
theorem thickness_after_13_folds (a : ℝ) :
  paper_thickness a 13 = a * 2^13 := by
  sorry

#check thickness_after_13_folds

end thickness_after_13_folds_l829_82908


namespace egg_collection_sum_l829_82972

theorem egg_collection_sum (n : ℕ) (a₁ : ℕ) (d : ℕ) (S : ℕ) : 
  n = 12 → a₁ = 25 → d = 5 → S = n * (2 * a₁ + (n - 1) * d) / 2 → S = 630 := by sorry

end egg_collection_sum_l829_82972


namespace cost_difference_l829_82975

def candy_bar_cost : ℝ := 6
def chocolate_cost : ℝ := 3

theorem cost_difference : candy_bar_cost - chocolate_cost = 3 := by
  sorry

end cost_difference_l829_82975


namespace kid_tickets_sold_l829_82951

theorem kid_tickets_sold (adult_price kid_price total_tickets total_profit : ℕ) 
  (h1 : adult_price = 12)
  (h2 : kid_price = 5)
  (h3 : total_tickets = 275)
  (h4 : total_profit = 2150) :
  ∃ (adult_tickets kid_tickets : ℕ),
    adult_tickets + kid_tickets = total_tickets ∧
    adult_price * adult_tickets + kid_price * kid_tickets = total_profit ∧
    kid_tickets = 164 := by
  sorry

end kid_tickets_sold_l829_82951


namespace max_candies_after_20_hours_l829_82991

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Next state of candies after one hour -/
def nextState (n : ℕ) : ℕ := n + sumOfDigits n

/-- State of candies after t hours, starting from 1 -/
def candyState (t : ℕ) : ℕ :=
  match t with
  | 0 => 1
  | t + 1 => nextState (candyState t)

theorem max_candies_after_20_hours :
  candyState 20 = 148 := by sorry

end max_candies_after_20_hours_l829_82991


namespace original_plan_pages_l829_82935

-- Define the total number of pages in the book
def total_pages : ℕ := 200

-- Define the number of days before changing the plan
def days_before_change : ℕ := 5

-- Define the additional pages read per day after changing the plan
def additional_pages : ℕ := 5

-- Define the number of days earlier the book was finished
def days_earlier : ℕ := 1

-- Define the function to calculate the total pages read
def total_pages_read (x : ℕ) : ℕ :=
  (days_before_change * x) + 
  ((x + additional_pages) * (total_pages / x - days_before_change - days_earlier))

-- Theorem stating that the original plan was to read 20 pages per day
theorem original_plan_pages : 
  ∃ (x : ℕ), x > 0 ∧ total_pages_read x = total_pages ∧ x = 20 := by
  sorry

end original_plan_pages_l829_82935


namespace arithmetic_sequence_inequality_l829_82990

theorem arithmetic_sequence_inequality (a b c : ℝ) (h : ∃ d : ℝ, d ≠ 0 ∧ b - a = d ∧ c - b = d) :
  ¬ (∀ a b c : ℝ, a^3 * b + b^3 * c + c^3 * a ≥ a^4 + b^4 + c^4) := by
  sorry

end arithmetic_sequence_inequality_l829_82990


namespace at_op_difference_l829_82919

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y + x - y

-- State the theorem
theorem at_op_difference : at_op 7 4 - at_op 4 7 = 6 := by sorry

end at_op_difference_l829_82919


namespace sqrt_equation_solution_l829_82933

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (1 + Real.sqrt (2 * y - 5)) = Real.sqrt 7 → y = 20.5 := by
  sorry

end sqrt_equation_solution_l829_82933


namespace lcm_12_18_l829_82985

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l829_82985


namespace equilateral_triangle_existence_l829_82984

/-- Represents a color: Blue, White, or Red -/
inductive Color
| Blue
| White
| Red

/-- Represents a point in the triangle -/
structure Point where
  x : ℤ
  y : ℤ

/-- The set of points S satisfying the given conditions -/
def S (h : ℕ) : Set Point :=
  {p : Point | 0 ≤ p.x ∧ 0 ≤ p.y ∧ p.x + p.y ≤ h}

/-- A coloring function that respects the given constraints -/
def coloringFunction (h : ℕ) (p : Point) : Color :=
  sorry

/-- Checks if three points form an equilateral triangle with side length 1 -/
def isUnitEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  sorry

theorem equilateral_triangle_existence (h : ℕ) (h_pos : h > 0) :
  ∃ (p1 p2 p3 : Point),
    p1 ∈ S h ∧ p2 ∈ S h ∧ p3 ∈ S h ∧
    isUnitEquilateralTriangle p1 p2 p3 ∧
    coloringFunction h p1 ≠ coloringFunction h p2 ∧
    coloringFunction h p2 ≠ coloringFunction h p3 ∧
    coloringFunction h p3 ≠ coloringFunction h p1 :=
sorry

end equilateral_triangle_existence_l829_82984


namespace cream_ratio_proof_l829_82963

-- Define the given constants
def servings : ℕ := 4
def fat_per_cup : ℕ := 88
def fat_per_serving : ℕ := 11

-- Define the ratio we want to prove
def cream_ratio : ℚ := 1 / 2

-- Theorem statement
theorem cream_ratio_proof :
  (servings * fat_per_serving : ℚ) / fat_per_cup = cream_ratio := by
  sorry

end cream_ratio_proof_l829_82963


namespace cos_eight_degrees_l829_82946

theorem cos_eight_degrees (m : ℝ) (h : Real.sin (74 * π / 180) = m) :
  Real.cos (8 * π / 180) = Real.sqrt ((1 + m) / 2) := by
  sorry

end cos_eight_degrees_l829_82946


namespace karen_late_start_l829_82971

/-- Proves that Karen starts the race 4 minutes late given the conditions of the car race. -/
theorem karen_late_start (karen_speed tom_speed : ℝ) (tom_distance : ℝ) (karen_win_margin : ℝ) :
  karen_speed = 60 →
  tom_speed = 45 →
  tom_distance = 24 →
  karen_win_margin = 4 →
  (tom_distance / tom_speed * 60 - (tom_distance + karen_win_margin) / karen_speed * 60 : ℝ) = 4 := by
  sorry

#check karen_late_start

end karen_late_start_l829_82971


namespace min_distance_to_point_l829_82960

theorem min_distance_to_point (x y : ℝ) (h : x^2 + y^2 - 4*x + 2 = 0) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' + 2 = 0 → x'^2 + (y' - 2)^2 ≥ min :=
sorry

end min_distance_to_point_l829_82960


namespace equilateral_triangle_hexagon_area_l829_82966

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  ∃ (s : ℝ), s > 0 ∧ 
  (dist P Q = s) ∧ (dist Q R = s) ∧ (dist R P = s)

-- Define the perimeter of the triangle
def Perimeter (P Q R : ℝ × ℝ) : ℝ :=
  dist P Q + dist Q R + dist R P

-- Define the circumcircle and perpendicular bisectors
def CircumcirclePerpBisectors (P Q R P' Q' R' : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ), 
    dist O P = dist O Q ∧ dist O Q = dist O R ∧
    dist O P' = dist O Q' ∧ dist O Q' = dist O R' ∧
    (P'.1 + Q.1) / 2 = P'.1 ∧ (P'.2 + Q.2) / 2 = P'.2 ∧
    (Q'.1 + R.1) / 2 = Q'.1 ∧ (Q'.2 + R.2) / 2 = Q'.2 ∧
    (R'.1 + P.1) / 2 = R'.1 ∧ (R'.2 + P.2) / 2 = R'.2

-- Define the area of a hexagon
def HexagonArea (P Q' R Q' P R' : ℝ × ℝ) : ℝ :=
  sorry  -- Actual calculation of area would go here

-- The main theorem
theorem equilateral_triangle_hexagon_area 
  (P Q R P' Q' R' : ℝ × ℝ) :
  Triangle P Q R →
  Perimeter P Q R = 42 →
  CircumcirclePerpBisectors P Q R P' Q' R' →
  HexagonArea P Q' R Q' P R' = 49 * Real.sqrt 3 :=
sorry

end equilateral_triangle_hexagon_area_l829_82966


namespace range_of_a_l829_82903

theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x > 0, (1 / a) - (1 / x) ≤ 2 * x) : 
  a ≥ Real.sqrt 2 / 4 := by
sorry

end range_of_a_l829_82903


namespace f_properties_l829_82931

noncomputable def f (x : ℝ) : ℝ := 2 * x / Real.log x

theorem f_properties :
  let e := Real.exp 1
  -- 1. f'(e^2) = 1/2
  (deriv f (e^2) = 1/2) ∧
  -- 2. f is monotonically decreasing on (0, 1) and (1, e)
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f y < f x) ∧
  (∀ x y, 1 < x ∧ x < y ∧ y < e → f y < f x) ∧
  -- 3. For all x > 0, x ≠ 1, f(x) > 2 / ln(x) + 2√x
  (∀ x, x > 0 ∧ x ≠ 1 → f x > 2 / Real.log x + 2 * Real.sqrt x) :=
by sorry


end f_properties_l829_82931


namespace fixed_point_transformation_l829_82922

theorem fixed_point_transformation (f : ℝ → ℝ) (h : f 1 = 1) : f (4 - 3) = 1 := by
  sorry

end fixed_point_transformation_l829_82922


namespace seedling_count_l829_82976

theorem seedling_count (packets : ℕ) (seeds_per_packet : ℕ) 
  (h1 : packets = 60) (h2 : seeds_per_packet = 7) :
  packets * seeds_per_packet = 420 := by
  sorry

end seedling_count_l829_82976


namespace junior_score_l829_82916

theorem junior_score (n : ℝ) (junior_score : ℝ) :
  n > 0 →
  0.15 * n * junior_score + 0.85 * n * 87 = n * 88 →
  junior_score = 94 := by
  sorry

end junior_score_l829_82916


namespace division_theorem_l829_82900

-- Define the dividend polynomial
def dividend (x : ℝ) : ℝ := x^5 + 2*x^3 + x^2 + 3

-- Define the divisor polynomial
def divisor (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Define the quotient polynomial
def quotient (x : ℝ) : ℝ := x^3 + 4*x^2 + 12*x

-- Define the remainder polynomial
def remainder (x : ℝ) : ℝ := 25*x^2 - 72*x + 3

-- Theorem statement
theorem division_theorem :
  ∀ x : ℝ, dividend x = divisor x * quotient x + remainder x :=
by sorry

end division_theorem_l829_82900


namespace sphere_volume_from_circumference_l829_82904

/-- The volume of a sphere with circumference 30 cm is 4500/π² cm³ -/
theorem sphere_volume_from_circumference :
  ∀ (r : ℝ), 
    2 * π * r = 30 → 
    (4 / 3) * π * r ^ 3 = 4500 / π ^ 2 := by
  sorry

end sphere_volume_from_circumference_l829_82904


namespace equation_equivalence_l829_82905

theorem equation_equivalence (a : ℝ) : (a - 1)^2 = a^3 - 2*a + 1 ↔ a = 0 ∨ a = 1 := by
  sorry

end equation_equivalence_l829_82905


namespace undefined_expression_l829_82930

theorem undefined_expression (y : ℝ) : 
  (y^2 - 10*y + 25 = 0) ↔ (y = 5) := by
  sorry

#check undefined_expression

end undefined_expression_l829_82930


namespace power_function_properties_l829_82907

-- Define the power function
def f (m : ℕ) (x : ℝ) : ℝ := x^(3*m - 5)

-- Define the theorem
theorem power_function_properties (m : ℕ) :
  (∀ x y, 0 < x ∧ x < y → f m y < f m x) ∧  -- f is decreasing on (0, +∞)
  (∀ x, f m (-x) = f m x) →                 -- f(-x) = f(x)
  m = 1 := by sorry

end power_function_properties_l829_82907


namespace lunch_total_is_fifteen_l829_82958

/-- The total amount spent on lunch given the conditions -/
def total_lunch_amount (friend_spent : ℕ) (difference : ℕ) : ℕ :=
  friend_spent + (friend_spent - difference)

/-- Theorem: The total amount spent on lunch is $15 -/
theorem lunch_total_is_fifteen :
  total_lunch_amount 10 5 = 15 := by
  sorry

end lunch_total_is_fifteen_l829_82958


namespace marble_box_problem_l829_82987

theorem marble_box_problem :
  ∀ (red blue : ℕ),
  red = blue →
  20 + red + blue - 2 * (20 - blue) = 40 →
  20 + red + blue = 50 :=
by
  sorry

end marble_box_problem_l829_82987


namespace lower_bound_x_l829_82981

theorem lower_bound_x (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8)
  (h_diff : ∃ (n : ℤ), n = ⌊y - x⌋ ∧ n = 4) : 3 < x :=
sorry

end lower_bound_x_l829_82981


namespace difference_of_squares_l829_82997

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end difference_of_squares_l829_82997


namespace sine_cosine_inequality_l829_82993

theorem sine_cosine_inequality : 
  Real.sin (11 * π / 180) < Real.sin (168 * π / 180) ∧ 
  Real.sin (168 * π / 180) < Real.cos (10 * π / 180) := by
  sorry

end sine_cosine_inequality_l829_82993


namespace paper_fold_unfold_holes_l829_82977

/-- Represents a rectangular piece of paper --/
structure Paper where
  height : ℕ
  width : ℕ

/-- Represents the position of a hole on the paper --/
structure HolePosition where
  x : ℕ
  y : ℕ

/-- Represents the state of the paper after folding and hole punching --/
structure FoldedPaper where
  original : Paper
  folds : List (Paper → Paper)
  holePosition : HolePosition

/-- Function to calculate the number and arrangement of holes after unfolding --/
def unfoldAndCount (fp : FoldedPaper) : ℕ × List HolePosition :=
  sorry

/-- Theorem stating the result of folding and unfolding the paper --/
theorem paper_fold_unfold_holes :
  ∀ (fp : FoldedPaper),
    fp.original = Paper.mk 8 12 →
    fp.folds.length = 3 →
    (unfoldAndCount fp).1 = 8 ∧
    (∃ (col1 col2 : ℕ), ∀ (pos : HolePosition),
      pos ∈ (unfoldAndCount fp).2 →
      (pos.x = col1 ∨ pos.x = col2) ∧
      pos.y ≤ 8) :=
by sorry

end paper_fold_unfold_holes_l829_82977


namespace x_squared_eq_one_is_quadratic_l829_82980

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: The equation x^2 = 1 is a quadratic equation -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f := by
  sorry

end x_squared_eq_one_is_quadratic_l829_82980


namespace vector_dot_product_zero_l829_82901

theorem vector_dot_product_zero (a b : ℝ × ℝ) (h1 : a = (2, 0)) (h2 : b = (1/2, Real.sqrt 3 / 2)) :
  b • (a - b) = 0 := by sorry

end vector_dot_product_zero_l829_82901


namespace problem_1_problem_2_l829_82988

-- Problem 1
theorem problem_1 : -105 - (-112) + 20 + 18 = 45 := by
  sorry

-- Problem 2
theorem problem_2 : 13 + (-22) - 25 - (-18) = -16 := by
  sorry

end problem_1_problem_2_l829_82988


namespace arithmetic_sequence_a10_l829_82941

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a10
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a3 : a 3 = 12)
  (h_a6 : a 6 = 27) :
  a 10 = 47 := by
  sorry

end arithmetic_sequence_a10_l829_82941


namespace smallest_n_for_jason_win_l829_82926

/-- Represents the game board -/
structure GameBoard :=
  (width : Nat)
  (length : Nat)

/-- Represents a block that can be placed on the game board -/
structure Block :=
  (width : Nat)
  (length : Nat)

/-- Represents a player in the game -/
inductive Player
  | Jason
  | Jared

/-- Defines the game rules and conditions -/
def GameRules (board : GameBoard) (jasonBlock : Block) (jaredBlock : Block) :=
  board.width = 3 ∧
  board.length = 300 ∧
  jasonBlock.width = 2 ∧
  jasonBlock.length = 100 ∧
  jaredBlock.width = 2 ∧
  jaredBlock.length > 3

/-- Determines if a player can win given the game rules and block sizes -/
def CanWin (player : Player) (board : GameBoard) (jasonBlock : Block) (jaredBlock : Block) : Prop :=
  sorry

/-- The main theorem stating that 51 is the smallest n for Jason to guarantee a win -/
theorem smallest_n_for_jason_win (board : GameBoard) (jasonBlock : Block) (jaredBlock : Block) :
  GameRules board jasonBlock jaredBlock →
  (∀ n : Nat, n > 3 → n < 51 → ¬CanWin Player.Jason board jasonBlock {width := 2, length := n}) ∧
  CanWin Player.Jason board jasonBlock {width := 2, length := 51} :=
sorry

end smallest_n_for_jason_win_l829_82926


namespace purchase_problem_l829_82906

theorem purchase_problem (a b c : ℕ) : 
  a + b + c = 50 →
  60 * a + 500 * b + 400 * c = 10000 →
  a = 30 :=
by sorry

end purchase_problem_l829_82906


namespace college_student_count_l829_82937

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- The ratio of boys to girls is 5:7 -/
def ratio_condition (c : College) : Prop :=
  7 * c.boys = 5 * c.girls

/-- There are 140 girls -/
def girls_count (c : College) : Prop :=
  c.girls = 140

/-- The total number of students -/
def total_students (c : College) : ℕ :=
  c.boys + c.girls

/-- Theorem stating the total number of students in the college -/
theorem college_student_count (c : College) 
  (h1 : ratio_condition c) (h2 : girls_count c) : 
  total_students c = 240 := by
  sorry

end college_student_count_l829_82937


namespace contains_2850_of_0_001_l829_82956

theorem contains_2850_of_0_001 : 2.85 = 2850 * 0.001 := by
  sorry

end contains_2850_of_0_001_l829_82956


namespace pet_ownership_l829_82911

theorem pet_ownership (S : Finset Nat) (D C B : Finset Nat) : 
  S.card = 60 ∧
  (∀ s ∈ S, s ∈ D ∪ C ∪ B) ∧
  D.card = 35 ∧
  C.card = 45 ∧
  B.card = 10 ∧
  (∀ b ∈ B, b ∈ D ∪ C) →
  ((D ∩ C) \ B).card = 10 := by
sorry

end pet_ownership_l829_82911
