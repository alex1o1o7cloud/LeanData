import Mathlib

namespace other_root_of_quadratic_l2713_271384

theorem other_root_of_quadratic (m : ℝ) : 
  (3 : ℝ) ^ 2 + m * 3 - 12 = 0 → 
  (-4 : ℝ) ^ 2 + m * (-4) - 12 = 0 := by
sorry

end other_root_of_quadratic_l2713_271384


namespace q_squared_minus_one_div_fifteen_l2713_271358

/-- The largest prime with 2023 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2023 digits -/
axiom q_digits : 10^2022 ≤ q ∧ q < 10^2023

/-- q is the largest prime with 2023 digits -/
axiom q_largest : ∀ p, Nat.Prime p → 10^2022 ≤ p ∧ p < 10^2023 → p ≤ q

theorem q_squared_minus_one_div_fifteen : 15 ∣ (q^2 - 1) := by sorry

end q_squared_minus_one_div_fifteen_l2713_271358


namespace money_ratio_l2713_271311

theorem money_ratio (rodney ian jessica : ℕ) : 
  rodney = ian + 35 →
  jessica = 100 →
  jessica = rodney + 15 →
  ian * 2 = jessica := by sorry

end money_ratio_l2713_271311


namespace cubic_function_two_zeros_l2713_271350

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^3 - x + a

-- State the theorem
theorem cubic_function_two_zeros (a : ℝ) (h : a > 0) :
  (∃! x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ a = 2/9 := by
  sorry

end cubic_function_two_zeros_l2713_271350


namespace coupon_a_best_at_220_l2713_271366

def coupon_a_discount (price : ℝ) : ℝ := 0.12 * price

def coupon_b_discount (price : ℝ) : ℝ := 25

def coupon_c_discount (price : ℝ) : ℝ := 0.2 * (price - 120)

theorem coupon_a_best_at_220 :
  let price := 220
  coupon_a_discount price > coupon_b_discount price ∧
  coupon_a_discount price > coupon_c_discount price :=
by sorry

end coupon_a_best_at_220_l2713_271366


namespace necklace_bead_count_l2713_271323

/-- Proves that the total number of beads in a necklace is 40 -/
theorem necklace_bead_count :
  let amethyst_count : ℕ := 7
  let amber_count : ℕ := 2 * amethyst_count
  let turquoise_count : ℕ := 19
  let total_count : ℕ := amethyst_count + amber_count + turquoise_count
  total_count = 40 := by sorry

end necklace_bead_count_l2713_271323


namespace student_mistake_fraction_l2713_271353

theorem student_mistake_fraction (original_number : ℕ) 
  (h1 : original_number = 384) 
  (correct_fraction : ℚ) 
  (h2 : correct_fraction = 5 / 16) 
  (mistake_fraction : ℚ) : 
  (mistake_fraction * original_number = correct_fraction * original_number + 200) → 
  mistake_fraction = 5 / 6 := by
sorry

end student_mistake_fraction_l2713_271353


namespace combined_girls_average_is_89_l2713_271341

/-- Represents a high school with average test scores -/
structure School where
  boyAvg : ℝ
  girlAvg : ℝ
  combinedAvg : ℝ

/-- Calculates the combined average score for girls given two schools and the combined boys' average -/
def combinedGirlsAverage (lincoln : School) (monroe : School) (combinedBoysAvg : ℝ) : ℝ :=
  sorry

theorem combined_girls_average_is_89 (lincoln : School) (monroe : School) (combinedBoysAvg : ℝ) :
  lincoln.boyAvg = 75 ∧
  lincoln.girlAvg = 78 ∧
  lincoln.combinedAvg = 76 ∧
  monroe.boyAvg = 85 ∧
  monroe.girlAvg = 92 ∧
  monroe.combinedAvg = 88 ∧
  combinedBoysAvg = 82 →
  combinedGirlsAverage lincoln monroe combinedBoysAvg = 89 := by
  sorry

end combined_girls_average_is_89_l2713_271341


namespace expression_simplification_and_evaluation_l2713_271393

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ 2 ∧ x ≠ -2 →
  (1 - 1 / (x - 1)) / ((x^2 - 4) / (x - 1)) = 1 / (x + 2) ∧
  (1 - 1 / (-1 - 1)) / (((-1)^2 - 4) / (-1 - 1)) = 1 := by
sorry

end expression_simplification_and_evaluation_l2713_271393


namespace complex_sum_cube_ratio_l2713_271313

theorem complex_sum_cube_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 18)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = (x*y*z)/3) :
  (x^3 + y^3 + z^3) / (x*y*z) = 6 := by
  sorry

end complex_sum_cube_ratio_l2713_271313


namespace wax_needed_proof_l2713_271380

/-- Given an amount of wax and a required amount, calculate the additional wax needed -/
def additional_wax_needed (current_amount required_amount : ℕ) : ℕ :=
  required_amount - current_amount

/-- Theorem stating that 17 grams of additional wax are needed -/
theorem wax_needed_proof (current_amount required_amount : ℕ) 
  (h1 : current_amount = 557)
  (h2 : required_amount = 574) :
  additional_wax_needed current_amount required_amount = 17 := by
  sorry

end wax_needed_proof_l2713_271380


namespace positive_difference_l2713_271351

theorem positive_difference (a b c d : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : 0 < c) (h4 : c < d) :
  d - c - b - a > 0 := by
  sorry

end positive_difference_l2713_271351


namespace negation_of_forall_geq_two_squared_geq_four_l2713_271314

theorem negation_of_forall_geq_two_squared_geq_four :
  (¬ (∀ x : ℝ, x ≥ 2 → x^2 ≥ 4)) ↔ (∃ x₀ : ℝ, x₀ ≥ 2 ∧ x₀^2 < 4) :=
by sorry

end negation_of_forall_geq_two_squared_geq_four_l2713_271314


namespace part_one_part_two_l2713_271317

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

-- Part 1
theorem part_one (x : ℝ) (h1 : p 1 x) (h2 : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : ∀ x, q x → p a x) : 1 < a ∧ a ≤ 2 := by
  sorry

end part_one_part_two_l2713_271317


namespace obtuse_triangle_x_range_l2713_271389

/-- Given three line segments with lengths x^2+4, 4x, and x^2+8,
    this theorem states the range of x values that can form an obtuse triangle. -/
theorem obtuse_triangle_x_range (x : ℝ) :
  (∃ (a b c : ℝ), a = x^2 + 4 ∧ b = 4*x ∧ c = x^2 + 8 ∧
   a > 0 ∧ b > 0 ∧ c > 0 ∧
   a + b > c ∧ b + c > a ∧ a + c > b ∧
   c^2 > a^2 + b^2) ↔ 
  (1 < x ∧ x < Real.sqrt 6) :=
by sorry

end obtuse_triangle_x_range_l2713_271389


namespace range_of_a_l2713_271370

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x + 3*a else a^x

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  (∀ x y : ℝ, x < y → f a x > f a y) →
  (1/3 ≤ a ∧ a < 1) :=
by sorry

end range_of_a_l2713_271370


namespace novels_pages_per_book_l2713_271364

theorem novels_pages_per_book (novels_per_month : ℕ) (pages_per_year : ℕ) : 
  novels_per_month = 4 → pages_per_year = 9600 → 
  (pages_per_year / (novels_per_month * 12) : ℚ) = 200 := by
  sorry

end novels_pages_per_book_l2713_271364


namespace cube_gt_of_gt_l2713_271302

theorem cube_gt_of_gt (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_gt_of_gt_l2713_271302


namespace origin_on_circle_circle_through_P_l2713_271355

-- Define the parabola C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 2 * p.1}

-- Define the point (2, 0)
def point_2_0 : ℝ × ℝ := (2, 0)

-- Define the line l passing through (2, 0)
def l (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * (p.1 - 2)}

-- Define the intersection points A and B
def A (k : ℝ) : ℝ × ℝ := sorry
def B (k : ℝ) : ℝ × ℝ := sorry

-- Define the circle M with diameter AB
def M (k : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define point P
def P : ℝ × ℝ := (4, -2)

-- Theorem 1: The origin O is on circle M
theorem origin_on_circle (k : ℝ) : O ∈ M k := sorry

-- Theorem 2: If M passes through P, then l and M have specific equations
theorem circle_through_P (k : ℝ) (h : P ∈ M k) :
  (k = -2 ∧ l k = {p : ℝ × ℝ | p.2 = -2 * p.1 + 4} ∧ 
   M k = {p : ℝ × ℝ | (p.1 - 9/4)^2 + (p.2 + 1/2)^2 = 85/16}) ∨
  (k = 1 ∧ l k = {p : ℝ × ℝ | p.2 = p.1 - 2} ∧ 
   M k = {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 10}) := sorry

end origin_on_circle_circle_through_P_l2713_271355


namespace total_cost_for_index_finger_rings_l2713_271398

def cost_per_ring : ℕ := 24
def index_fingers_per_person : ℕ := 2

theorem total_cost_for_index_finger_rings :
  cost_per_ring * index_fingers_per_person = 48 := by
  sorry

end total_cost_for_index_finger_rings_l2713_271398


namespace stratified_sampling_female_count_l2713_271348

theorem stratified_sampling_female_count 
  (total_employees : ℕ) 
  (female_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 200)
  (h2 : female_employees = 80)
  (h3 : sample_size = 20) :
  (female_employees : ℚ) / total_employees * sample_size = 8 := by
  sorry

end stratified_sampling_female_count_l2713_271348


namespace square_difference_given_sum_and_product_l2713_271340

theorem square_difference_given_sum_and_product (x y : ℝ) 
  (h1 : x^2 + y^2 = 10) (h2 : x * y = 2) : (x - y)^2 = 6 := by
  sorry

end square_difference_given_sum_and_product_l2713_271340


namespace factorization_equality_l2713_271397

theorem factorization_equality (a b : ℝ) : a^2 * b - b = b * (a + 1) * (a - 1) := by
  sorry

end factorization_equality_l2713_271397


namespace range_of_m_l2713_271333

def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≥ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≥ 0

def not_p (x : ℝ) : Prop := -2 < x ∧ x < 10

def not_q (x m : ℝ) : Prop := 1 - m < x ∧ x < 1 + m

theorem range_of_m :
  ∀ m : ℝ, (m > 0 ∧
    (∀ x : ℝ, not_q x m → not_p x) ∧
    (∃ x : ℝ, not_p x ∧ ¬(not_q x m))) ↔
  (0 < m ∧ m ≤ 3) :=
sorry

end range_of_m_l2713_271333


namespace repeating_decimal_multiplication_l2713_271309

/-- Given a real number x where x = 0.000272727... (27 repeats indefinitely),
    prove that (10^5 - 10^3) * x = 27 -/
theorem repeating_decimal_multiplication (x : ℝ) : 
  (∃ (n : ℕ), x * 10^(n+5) - x * 10^5 = 27 * (10^n - 1) / 99) → 
  (10^5 - 10^3) * x = 27 := by
  sorry

end repeating_decimal_multiplication_l2713_271309


namespace pure_imaginary_fraction_l2713_271395

theorem pure_imaginary_fraction (b : ℝ) : 
  (∃ (y : ℝ), (b + Complex.I) / (2 + Complex.I) = Complex.I * y) → b = -1/2 := by
  sorry

end pure_imaginary_fraction_l2713_271395


namespace normal_distribution_symmetry_l2713_271354

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- Probability of a normal random variable being less than or equal to a value -/
noncomputable def probability (X : NormalRV) (x : ℝ) : ℝ := sorry

theorem normal_distribution_symmetry (X : NormalRV) (h : X.μ = 2) (h_prob : probability X 4 = 0.84) :
  probability X 0 = 0.16 := by sorry

end normal_distribution_symmetry_l2713_271354


namespace watermelon_price_l2713_271334

theorem watermelon_price : 
  let base_price : ℕ := 5000
  let additional_cost : ℕ := 200
  let total_price : ℕ := base_price + additional_cost
  let price_in_thousands : ℚ := total_price / 1000
  price_in_thousands = 5.2 := by sorry

end watermelon_price_l2713_271334


namespace det_M_eq_26_l2713_271339

/-- The determinant of a 2x2 matrix -/
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- The specific 2x2 matrix we're interested in -/
def M : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3; -2, 4]

/-- Theorem stating that the determinant of M is 26 -/
theorem det_M_eq_26 : det2x2 (M 0 0) (M 0 1) (M 1 0) (M 1 1) = 26 := by
  sorry

end det_M_eq_26_l2713_271339


namespace garbage_ratio_proof_l2713_271349

def garbage_problem (collection_days_per_week : ℕ) 
                    (avg_collection_per_day : ℕ) 
                    (weeks_without_collection : ℕ) 
                    (total_accumulated : ℕ) : Prop :=
  let weekly_collection := collection_days_per_week * avg_collection_per_day
  let total_normal_collection := weekly_collection * weeks_without_collection
  let first_week_garbage := weekly_collection
  let second_week_garbage := total_accumulated - first_week_garbage
  (2 : ℚ) * second_week_garbage = first_week_garbage

theorem garbage_ratio_proof : 
  garbage_problem 3 200 2 900 := by
  sorry

#check garbage_ratio_proof

end garbage_ratio_proof_l2713_271349


namespace a_3_value_l2713_271359

/-- Given a polynomial expansion of (1+x)(a-x)^6, prove that a₃ = -5 when the sum of all coefficients is zero. -/
theorem a_3_value (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) : 
  (∀ x, (1 + x) * (a - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0) →
  a₃ = -5 := by
sorry

end a_3_value_l2713_271359


namespace polynomial_root_l2713_271360

theorem polynomial_root : ∃ (x : ℝ), 2 * x^5 + x^4 - 20 * x^3 - 10 * x^2 + 2 * x + 1 = 0 ∧ x = Real.sqrt 3 + Real.sqrt 2 := by
  sorry

end polynomial_root_l2713_271360


namespace vector_properties_l2713_271387

/-- Given vectors a and b, prove the sine of their angle and the value of m for perpendicularity. -/
theorem vector_properties (a b : ℝ × ℝ) (h1 : a = (3, -4)) (h2 : b = (1, 2)) :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  ∃ (m : ℝ),
    Real.sin θ = (2 * Real.sqrt 5) / 5 ∧
    (m * a.1 - b.1) * (a.1 + b.1) + (m * a.2 - b.2) * (a.2 + b.2) = 0 ∧
    m = 0 :=
by sorry

end vector_properties_l2713_271387


namespace cos_2alpha_minus_2pi_3_l2713_271367

theorem cos_2alpha_minus_2pi_3 (α : Real) (h : Real.sin (π/6 + α) = 3/5) : 
  Real.cos (2*α - 2*π/3) = -7/25 := by
  sorry

end cos_2alpha_minus_2pi_3_l2713_271367


namespace alcohol_dilution_l2713_271306

theorem alcohol_dilution (original_volume : ℝ) (original_percentage : ℝ) 
  (added_water : ℝ) (new_percentage : ℝ) :
  original_volume = 15 →
  original_percentage = 0.2 →
  added_water = 3 →
  new_percentage = 1/6 →
  (original_volume * original_percentage) / (original_volume + added_water) = new_percentage := by
  sorry

#check alcohol_dilution

end alcohol_dilution_l2713_271306


namespace unique_solution_l2713_271361

theorem unique_solution (a b c : ℝ) : 
  a > 4 ∧ b > 4 ∧ c > 4 ∧
  (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 48 →
  a = 9 ∧ b = 8 ∧ c = 8 :=
by sorry

end unique_solution_l2713_271361


namespace hotdogs_served_today_l2713_271356

/-- The number of hot dogs served during lunch today -/
def lunch_hotdogs : ℕ := 9

/-- The number of hot dogs served during dinner today -/
def dinner_hotdogs : ℕ := 2

/-- The total number of hot dogs served today -/
def total_hotdogs : ℕ := lunch_hotdogs + dinner_hotdogs

theorem hotdogs_served_today : total_hotdogs = 11 := by
  sorry

end hotdogs_served_today_l2713_271356


namespace multiplication_puzzle_l2713_271377

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem multiplication_puzzle : ∃! (a b : ℕ), 
  (1000 ≤ a * b) ∧ (a * b < 10000) ∧  -- 4-digit product
  (digit_sum a = digit_sum b) ∧       -- same digit sum
  (a * b % 10 = a % 10) ∧             -- ones digit condition
  ((a * b / 10) % 10 = 2) ∧           -- tens digit condition
  a = 2231 ∧ b = 26 := by
sorry

end multiplication_puzzle_l2713_271377


namespace meeting_distance_l2713_271352

/-- Proves that given two people 75 miles apart, walking towards each other at constant speeds of 4 mph and 6 mph respectively, the person walking at 6 mph will have walked 45 miles when they meet. -/
theorem meeting_distance (initial_distance : ℝ) (speed_fred : ℝ) (speed_sam : ℝ) 
  (h1 : initial_distance = 75)
  (h2 : speed_fred = 4)
  (h3 : speed_sam = 6) :
  let distance_sam := initial_distance * speed_sam / (speed_fred + speed_sam)
  distance_sam = 45 := by
  sorry

#check meeting_distance

end meeting_distance_l2713_271352


namespace largest_s_value_l2713_271362

/-- The largest possible value of s for regular polygons satisfying given conditions -/
theorem largest_s_value : ∃ (s : ℕ), s = 121 ∧ 
  (∀ (r s' : ℕ), r ≥ s' ∧ s' ≥ 3 →
    (r - 2 : ℚ) / r * 60 = (s' - 2 : ℚ) / s' * 61 →
    s' ≤ s) :=
sorry

end largest_s_value_l2713_271362


namespace x_squared_mod_25_l2713_271319

theorem x_squared_mod_25 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 25]) 
  (h2 : 4 * x ≡ 20 [ZMOD 25]) : 
  x^2 ≡ 4 [ZMOD 25] := by
  sorry

end x_squared_mod_25_l2713_271319


namespace april_rose_price_l2713_271310

/-- Calculates the price per rose given the initial number of roses, remaining roses, and total earnings. -/
def price_per_rose (initial_roses : ℕ) (remaining_roses : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (initial_roses - remaining_roses)

/-- Proves that the price per rose is $4 given the problem conditions. -/
theorem april_rose_price : price_per_rose 13 4 36 = 4 := by
  sorry

end april_rose_price_l2713_271310


namespace complex_fraction_equality_l2713_271386

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) :
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 41 / 20 :=
by sorry

end complex_fraction_equality_l2713_271386


namespace fixed_point_parabola_l2713_271327

theorem fixed_point_parabola (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 + k * x - 2 * k
  f 2 = 12 := by
  sorry

end fixed_point_parabola_l2713_271327


namespace cube_volume_from_surface_area_l2713_271307

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 →
  volume = (surface_area / 6) ^ (3/2) →
  volume = 125 := by
sorry

end cube_volume_from_surface_area_l2713_271307


namespace expected_twos_is_half_l2713_271375

/-- The probability of rolling a 2 on a standard die -/
def prob_two : ℚ := 1/6

/-- The probability of not rolling a 2 on a standard die -/
def prob_not_two : ℚ := 1 - prob_two

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 2's when rolling three standard dice -/
def expected_twos : ℚ :=
  0 * (prob_not_two ^ num_dice) +
  1 * (num_dice * prob_two * prob_not_two ^ 2) +
  2 * (num_dice * prob_two ^ 2 * prob_not_two) +
  3 * (prob_two ^ num_dice)

theorem expected_twos_is_half : expected_twos = 1/2 := by
  sorry

end expected_twos_is_half_l2713_271375


namespace function_inequality_l2713_271332

theorem function_inequality 
  (f : Real → Real) 
  (h_domain : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x) 
  (h_ineq : ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → 
    (f x + f y) / 2 ≤ f ((x + y) / 2) + 1) 
  (u v w : Real) 
  (h_order : 0 ≤ u ∧ u < v ∧ v < w ∧ w ≤ 1) : 
  ((w - v) / (w - u)) * f u + ((v - u) / (w - u)) * f w ≤ f v + 2 := by
sorry

end function_inequality_l2713_271332


namespace repeating_decimal_sum_l2713_271376

theorem repeating_decimal_sum (x : ℚ) : x = 23 / 99 → (x.num + x.den = 122) := by
  sorry

end repeating_decimal_sum_l2713_271376


namespace trailing_zeros_340_factorial_l2713_271330

-- Define a function to count trailing zeros in a factorial
def trailingZerosInFactorial (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

-- Theorem statement
theorem trailing_zeros_340_factorial :
  trailingZerosInFactorial 340 = 83 := by
  sorry

end trailing_zeros_340_factorial_l2713_271330


namespace cross_product_zero_implies_values_l2713_271342

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.2.2 * b.1 - a.1 * b.2.2,
   a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_zero_implies_values (x y : ℝ) :
  cross_product (3, x, -9) (4, 6, y) = (0, 0, 0) →
  x = 9/2 ∧ y = -12 := by
  sorry

end cross_product_zero_implies_values_l2713_271342


namespace fraction_power_product_specific_fraction_product_l2713_271368

theorem fraction_power_product (a b c d : ℚ) (j : ℕ) :
  (a / b) ^ j * (c / d) ^ j = ((a * c) / (b * d)) ^ j :=
sorry

theorem specific_fraction_product :
  (3 / 4 : ℚ) ^ 3 * (2 / 5 : ℚ) ^ 3 = 27 / 1000 :=
sorry

end fraction_power_product_specific_fraction_product_l2713_271368


namespace sams_new_books_l2713_271382

theorem sams_new_books : 
  ∀ (adventure_books mystery_books used_books : ℕ),
    adventure_books = 24 →
    mystery_books = 37 →
    used_books = 18 →
    adventure_books + mystery_books - used_books = 43 := by
  sorry

end sams_new_books_l2713_271382


namespace degree_of_monomial_l2713_271301

/-- The degree of a monomial is the sum of the exponents of its variables -/
def monomialDegree (coefficient : ℤ) (xExponent yExponent : ℕ) : ℕ :=
  xExponent + yExponent

/-- The monomial -3x^5y^2 has degree 7 -/
theorem degree_of_monomial :
  monomialDegree (-3) 5 2 = 7 := by sorry

end degree_of_monomial_l2713_271301


namespace angle_between_diagonals_l2713_271329

/-- 
Given a quadrilateral with area A, and diagonals d₁ and d₂, 
the angle α between the diagonals satisfies the equation:
A = (1/2) * d₁ * d₂ * sin(α)
-/
def quadrilateral_area_diagonals (A d₁ d₂ α : ℝ) : Prop :=
  A = (1/2) * d₁ * d₂ * Real.sin α

theorem angle_between_diagonals (A d₁ d₂ α : ℝ) 
  (h_area : A = 3)
  (h_diag1 : d₁ = 6)
  (h_diag2 : d₂ = 2)
  (h_quad : quadrilateral_area_diagonals A d₁ d₂ α) :
  α = π / 6 := by
  sorry

#check angle_between_diagonals

end angle_between_diagonals_l2713_271329


namespace three_digit_with_repeat_l2713_271365

/-- The number of digits available (0 to 9) -/
def num_digits : ℕ := 10

/-- The total number of three-digit numbers -/
def total_three_digit : ℕ := 900

/-- The number of three-digit numbers without repeated digits -/
def no_repeat_three_digit : ℕ := 9 * 9 * 8

/-- Theorem: The number of three-digit numbers with repeated digits using digits 0 to 9 is 252 -/
theorem three_digit_with_repeat : 
  total_three_digit - no_repeat_three_digit = 252 := by
  sorry

end three_digit_with_repeat_l2713_271365


namespace savings_account_theorem_l2713_271363

def initial_deposit : ℚ := 5 / 100
def daily_multiplier : ℚ := 3
def target_amount : ℚ := 500

def total_amount (n : ℕ) : ℚ :=
  initial_deposit * (1 - daily_multiplier^n) / (1 - daily_multiplier)

def exceeds_target (n : ℕ) : Prop :=
  total_amount n > target_amount

theorem savings_account_theorem :
  ∃ (n : ℕ), exceeds_target n ∧ ∀ (m : ℕ), m < n → ¬(exceeds_target m) :=
by sorry

end savings_account_theorem_l2713_271363


namespace cube_of_complex_root_of_unity_l2713_271374

theorem cube_of_complex_root_of_unity (z : ℂ) : 
  z = Complex.cos (2 * Real.pi / 3) - Complex.I * Complex.sin (Real.pi / 3) → 
  z^3 = 1 := by
  sorry

end cube_of_complex_root_of_unity_l2713_271374


namespace isabel_morning_runs_l2713_271369

/-- Represents the number of times Isabel runs the circuit in the morning -/
def morning_runs : ℕ := 7

/-- Represents the length of the circuit in meters -/
def circuit_length : ℕ := 365

/-- Represents the number of times Isabel runs the circuit in the afternoon -/
def afternoon_runs : ℕ := 3

/-- Represents the total distance Isabel runs in a week in meters -/
def weekly_distance : ℕ := 25550

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

theorem isabel_morning_runs :
  morning_runs * circuit_length * days_in_week +
  afternoon_runs * circuit_length * days_in_week = weekly_distance :=
sorry

end isabel_morning_runs_l2713_271369


namespace integer_solutions_of_equation_l2713_271379

theorem integer_solutions_of_equation :
  ∀ n : ℤ, (1/3 : ℚ) * n^4 - (1/21 : ℚ) * n^3 - n^2 - (11/21 : ℚ) * n + (4/42 : ℚ) = 0 ↔ n = -1 ∨ n = 2 := by
  sorry

end integer_solutions_of_equation_l2713_271379


namespace polygon_sides_when_interior_triple_exterior_polygon_sides_proof_l2713_271335

theorem polygon_sides_when_interior_triple_exterior : ℕ → Prop :=
  fun n =>
    (((n : ℝ) - 2) * 180 = 3 * 360) →
    n = 8

-- Proof
theorem polygon_sides_proof : polygon_sides_when_interior_triple_exterior 8 := by
  sorry

end polygon_sides_when_interior_triple_exterior_polygon_sides_proof_l2713_271335


namespace car_airplane_energy_consumption_ratio_l2713_271388

theorem car_airplane_energy_consumption_ratio :
  ∀ (maglev airplane car : ℝ),
    maglev > 0 → airplane > 0 → car > 0 →
    maglev = (1/3) * airplane →
    maglev = 0.7 * car →
    car = (10/21) * airplane :=
by sorry

end car_airplane_energy_consumption_ratio_l2713_271388


namespace specific_event_handshakes_l2713_271336

/-- Represents a social event with two groups of people -/
structure SocialEvent where
  total_people : ℕ
  group_a_size : ℕ
  group_b_size : ℕ
  h_total : total_people = group_a_size + group_b_size
  h_group_a : group_a_size > 0
  h_group_b : group_b_size > 0

/-- Calculates the number of handshakes in a social event -/
def handshakes (event : SocialEvent) : ℕ :=
  event.group_a_size * event.group_b_size

/-- Theorem stating the number of handshakes in the specific social event -/
theorem specific_event_handshakes :
  ∃ (event : SocialEvent),
    event.total_people = 40 ∧
    event.group_a_size = 25 ∧
    event.group_b_size = 15 ∧
    handshakes event = 375 := by
  sorry

end specific_event_handshakes_l2713_271336


namespace prime_factors_sum_l2713_271390

theorem prime_factors_sum (w x y z t : ℕ) : 
  2^w * 3^x * 5^y * 7^z * 11^t = 2310 → 2*w + 3*x + 5*y + 7*z + 11*t = 28 := by
sorry

end prime_factors_sum_l2713_271390


namespace prism_volume_l2713_271347

/-- The volume of a right rectangular prism given its face areas -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 50) (h2 : a * c = 72) (h3 : b * c = 45) :
  a * b * c = 180 * Real.sqrt 5 := by
  sorry

end prism_volume_l2713_271347


namespace exponential_function_property_l2713_271308

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_function_property (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  ∀ x y : ℝ, f a (x + y) = f a x * f a y :=
by sorry

end exponential_function_property_l2713_271308


namespace fenced_square_cost_l2713_271320

/-- A square with fenced sides -/
structure FencedSquare where
  side_cost : ℕ
  sides : ℕ

/-- The total cost of fencing a square -/
def total_fencing_cost (s : FencedSquare) : ℕ :=
  s.side_cost * s.sides

/-- Theorem: The total cost of fencing a square with 4 sides at $69 per side is $276 -/
theorem fenced_square_cost :
  ∀ (s : FencedSquare), s.side_cost = 69 → s.sides = 4 → total_fencing_cost s = 276 :=
by
  sorry

end fenced_square_cost_l2713_271320


namespace cos_160_eq_neg_cos_20_l2713_271316

/-- Proves that cos 160° equals -cos 20° --/
theorem cos_160_eq_neg_cos_20 : 
  Real.cos (160 * π / 180) = - Real.cos (20 * π / 180) := by
  sorry

end cos_160_eq_neg_cos_20_l2713_271316


namespace wrapping_paper_fraction_l2713_271391

theorem wrapping_paper_fraction (total_fraction : ℚ) (num_presents : ℕ) :
  total_fraction = 5/12 ∧ num_presents = 5 →
  total_fraction / num_presents = 1/12 := by
sorry

end wrapping_paper_fraction_l2713_271391


namespace arctan_sum_special_case_l2713_271381

theorem arctan_sum_special_case (a b : ℝ) : 
  a = 1/3 → (a + 1) * (b + 1) = 3 → Real.arctan a + Real.arctan b = π/3 := by
  sorry

end arctan_sum_special_case_l2713_271381


namespace root_value_l2713_271392

-- Define the polynomials and their roots
def f (x : ℝ) := x^3 + 5*x^2 + 2*x - 8
def g (x p q r : ℝ) := x^3 + p*x^2 + q*x + r

-- Define the roots
variable (a b c : ℝ)

-- State the conditions
axiom root_f : f a = 0 ∧ f b = 0 ∧ f c = 0
axiom root_g : ∃ p q r, g (2*a + b) p q r = 0 ∧ g (2*b + c) p q r = 0 ∧ g (2*c + a) p q r = 0

-- State the theorem to be proved
theorem root_value : ∃ p q, g (2*a + b) p q 18 = 0 ∧ g (2*b + c) p q 18 = 0 ∧ g (2*c + a) p q 18 = 0 :=
sorry

end root_value_l2713_271392


namespace existence_of_unequal_indices_l2713_271322

theorem existence_of_unequal_indices (a b c : ℕ → ℕ) : 
  ∃ m n : ℕ, m ≠ n ∧ a m ≥ a n ∧ b m ≥ b n ∧ c m ≥ c n := by
  sorry

end existence_of_unequal_indices_l2713_271322


namespace regular_ngon_max_area_and_perimeter_l2713_271325

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- An n-gon inscribed in a circle. -/
structure InscribedNGon (n : ℕ) (c : Circle) where
  vertices : Fin n → ℝ × ℝ
  inscribed : ∀ i, ((vertices i).1 - c.center.1)^2 + ((vertices i).2 - c.center.2)^2 = c.radius^2

/-- The area of an n-gon. -/
def area {n : ℕ} {c : Circle} (ngon : InscribedNGon n c) : ℝ :=
  sorry

/-- The perimeter of an n-gon. -/
def perimeter {n : ℕ} {c : Circle} (ngon : InscribedNGon n c) : ℝ :=
  sorry

/-- A regular n-gon inscribed in a circle. -/
def regularNGon (n : ℕ) (c : Circle) : InscribedNGon n c :=
  sorry

/-- Theorem: The regular n-gon has maximum area and perimeter among all inscribed n-gons. -/
theorem regular_ngon_max_area_and_perimeter (n : ℕ) (c : Circle) :
  ∀ (ngon : InscribedNGon n c),
    area ngon ≤ area (regularNGon n c) ∧
    perimeter ngon ≤ perimeter (regularNGon n c) :=
  sorry

end regular_ngon_max_area_and_perimeter_l2713_271325


namespace x_values_when_two_in_M_l2713_271396

def M (x : ℝ) : Set ℝ := {-2, 3*x^2 + 3*x - 4}

theorem x_values_when_two_in_M (x : ℝ) : 2 ∈ M x → x = 1 ∨ x = -2 := by
  sorry

end x_values_when_two_in_M_l2713_271396


namespace gravel_path_cost_l2713_271338

/-- The cost of gravelling a path inside a rectangular plot -/
theorem gravel_path_cost 
  (length width path_width : ℝ) 
  (cost_per_sqm : ℝ) 
  (h_length : length = 110) 
  (h_width : width = 65) 
  (h_path_width : path_width = 2.5) 
  (h_cost_per_sqm : cost_per_sqm = 0.4) : 
  ((length + 2 * path_width) * (width + 2 * path_width) - length * width) * cost_per_sqm = 360 := by
sorry

end gravel_path_cost_l2713_271338


namespace pentagon_perimeter_even_l2713_271383

-- Define a point with integer coordinates
structure IntPoint where
  x : Int
  y : Int

-- Define a pentagon as a list of 5 points
def Pentagon : Type := List IntPoint

-- Function to calculate the distance between two points
def distance (p1 p2 : IntPoint) : Int :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

-- Function to check if a pentagon has integer side lengths
def hasIntegerSideLengths (p : Pentagon) : Prop :=
  match p with
  | [a, b, c, d, e] => 
    ∃ (l1 l2 l3 l4 l5 : Int),
      distance a b = l1 ^ 2 ∧
      distance b c = l2 ^ 2 ∧
      distance c d = l3 ^ 2 ∧
      distance d e = l4 ^ 2 ∧
      distance e a = l5 ^ 2
  | _ => False

-- Function to calculate the perimeter of a pentagon
def perimeter (p : Pentagon) : Int :=
  match p with
  | [a, b, c, d, e] => 
    Int.sqrt (distance a b) +
    Int.sqrt (distance b c) +
    Int.sqrt (distance c d) +
    Int.sqrt (distance d e) +
    Int.sqrt (distance e a)
  | _ => 0

-- Theorem statement
theorem pentagon_perimeter_even (p : Pentagon) 
  (h1 : p.length = 5)
  (h2 : hasIntegerSideLengths p) :
  Even (perimeter p) := by
  sorry


end pentagon_perimeter_even_l2713_271383


namespace y_over_z_equals_negative_five_l2713_271321

theorem y_over_z_equals_negative_five 
  (x y z : ℚ) 
  (eq1 : x + y = 2*x + z) 
  (eq2 : x - 2*y = 4*z) 
  (eq3 : x + y + z = 21) : 
  y / z = -5 := by sorry

end y_over_z_equals_negative_five_l2713_271321


namespace garden_feet_count_l2713_271315

/-- Calculates the total number of feet in a garden with dogs and ducks -/
def total_feet (num_dogs : ℕ) (num_ducks : ℕ) (feet_per_dog : ℕ) (feet_per_duck : ℕ) : ℕ :=
  num_dogs * feet_per_dog + num_ducks * feet_per_duck

/-- Theorem: The total number of feet in a garden with 6 dogs and 2 ducks is 28 -/
theorem garden_feet_count : total_feet 6 2 4 2 = 28 := by
  sorry

end garden_feet_count_l2713_271315


namespace coloring_satisfies_conditions_l2713_271394

-- Define the color type
inductive Color
| White
| Red
| Black

-- Define a lattice point
structure LatticePoint where
  x : Int
  y : Int

-- Define the coloring function
def color (p : LatticePoint) : Color :=
  match p.x, p.y with
  | x, y => if x % 2 = 0 then Color.Red
            else if y % 2 = 0 then Color.Black
            else Color.White

-- Define a line parallel to x-axis
def Line (y : Int) := { p : LatticePoint | p.y = y }

-- Define a parallelogram
def isParallelogram (a b c d : LatticePoint) : Prop :=
  d.x = a.x + c.x - b.x ∧ d.y = a.y + c.y - b.y

-- Main theorem
theorem coloring_satisfies_conditions :
  (∀ c : Color, ∃ (S : Set Int), Infinite S ∧ ∀ y ∈ S, ∃ x : Int, color ⟨x, y⟩ = c) ∧
  (∀ a b c : LatticePoint, 
    color a = Color.White → color b = Color.Red → color c = Color.Black →
    ∃ d : LatticePoint, color d = Color.Red ∧ isParallelogram a b c d) :=
sorry


end coloring_satisfies_conditions_l2713_271394


namespace vector_magnitude_l2713_271312

def m : ℝ × ℝ := (2, 4)

theorem vector_magnitude (m : ℝ × ℝ) (n : ℝ × ℝ) : 
  let angle := π / 3
  norm m = 2 * Real.sqrt 5 →
  norm n = Real.sqrt 5 →
  m.1 * n.1 + m.2 * n.2 = norm m * norm n * Real.cos angle →
  norm (2 • m - 3 • n) = Real.sqrt 65 := by
  sorry

end vector_magnitude_l2713_271312


namespace wax_required_for_feathers_l2713_271337

/-- The amount of wax Icarus has, in grams. -/
def total_wax : ℕ := 557

/-- The amount of wax needed for the feathers, in grams. -/
def wax_needed : ℕ := 17

/-- Theorem stating that the amount of wax required for the feathers is equal to the amount needed, regardless of the total amount available. -/
theorem wax_required_for_feathers : wax_needed = 17 := by
  sorry

end wax_required_for_feathers_l2713_271337


namespace triangle_QCA_area_l2713_271399

/-- The area of triangle QCA given the coordinates of Q, A, and C, and that QA is perpendicular to QC -/
theorem triangle_QCA_area (p : ℝ) : 
  let Q : ℝ × ℝ := (0, 15)
  let A : ℝ × ℝ := (3, 15)
  let C : ℝ × ℝ := (0, p)
  -- QA is perpendicular to QC (implicit in the coordinate system)
  (45 - 3 * p) / 2 = (1 / 2) * 3 * (15 - p) := by
  sorry

end triangle_QCA_area_l2713_271399


namespace extended_parallelepiped_volume_calculation_l2713_271372

/-- The volume of the set of points inside or within two units of a rectangular parallelepiped with dimensions 2 by 3 by 4 units -/
def extended_parallelepiped_volume : ℝ := sorry

/-- The dimensions of the rectangular parallelepiped -/
def parallelepiped_dimensions : Fin 3 → ℝ
| 0 => 2
| 1 => 3
| 2 => 4
| _ => 0

/-- The extension distance around the parallelepiped -/
def extension_distance : ℝ := 2

theorem extended_parallelepiped_volume_calculation :
  extended_parallelepiped_volume = (384 + 140 * Real.pi) / 3 := by sorry

end extended_parallelepiped_volume_calculation_l2713_271372


namespace arithmetic_square_root_of_five_l2713_271303

theorem arithmetic_square_root_of_five :
  ∃ x : ℝ, x > 0 ∧ x^2 = 5 ∧ ∀ y : ℝ, y^2 = 5 → y = x ∨ y = -x :=
by sorry

end arithmetic_square_root_of_five_l2713_271303


namespace smallest_common_factor_l2713_271328

theorem smallest_common_factor (n : ℕ) : n = 85 ↔ 
  (n > 0 ∧ 
   ∃ (k : ℕ), k > 1 ∧ k ∣ (11*n - 4) ∧ k ∣ (8*n + 6) ∧
   ∀ (m : ℕ), m < n → 
     (∀ (j : ℕ), j > 1 → ¬(j ∣ (11*m - 4) ∧ j ∣ (8*m + 6)))) := by
  sorry

end smallest_common_factor_l2713_271328


namespace division_equivalence_l2713_271318

theorem division_equivalence (h : 43 * 47 = 2021) : (-43) / (1 / 47) = -2021 := by
  sorry

end division_equivalence_l2713_271318


namespace cos_315_degrees_l2713_271385

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_315_degrees_l2713_271385


namespace composite_quotient_l2713_271304

theorem composite_quotient (n : ℕ) (h1 : n ≥ 4) (h2 : n ∣ 2^n - 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (2^n - 2) / n = a * b := by
  sorry

end composite_quotient_l2713_271304


namespace fraction_equality_l2713_271326

theorem fraction_equality (x y : ℝ) : 
  (5 + 2*x) / (7 + 3*x + y) = (3 + 4*x) / (4 + 2*x + y) ↔ 
  8*x^2 + 19*x + 5*x*y = -1 - 5*y :=
by sorry

end fraction_equality_l2713_271326


namespace calculation_difference_l2713_271324

def correct_calculation : ℤ := 12 - (3 * 4)
def incorrect_calculation : ℤ := (12 - 3) * 4

theorem calculation_difference :
  correct_calculation - incorrect_calculation = -36 := by
  sorry

end calculation_difference_l2713_271324


namespace probability_even_sum_is_one_third_l2713_271300

def digits : Finset ℕ := {2, 3, 5}

def is_valid_arrangement (a b c d : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  (a = 2 ∧ b = 2) ∨ (a = 2 ∧ c = 2) ∨ (a = 2 ∧ d = 2) ∨
  (b = 2 ∧ c = 2) ∨ (b = 2 ∧ d = 2) ∨ (c = 2 ∧ d = 2)

def sum_first_last_even (a d : ℕ) : Prop :=
  (a + d) % 2 = 0

def count_valid_arrangements : ℕ := 12

def count_even_sum_arrangements : ℕ := 4

theorem probability_even_sum_is_one_third :
  (count_even_sum_arrangements : ℚ) / count_valid_arrangements = 1 / 3 := by
  sorry

end probability_even_sum_is_one_third_l2713_271300


namespace x_value_l2713_271343

theorem x_value : ∃ x : ℝ, (3 * x) / 7 = 15 ∧ x = 35 := by
  sorry

end x_value_l2713_271343


namespace candy_problem_l2713_271344

/-- Given an initial amount of candy and the amounts eaten in two stages,
    calculate the remaining amount of candy. -/
def remaining_candy (initial : ℕ) (first_eaten : ℕ) (second_eaten : ℕ) : ℕ :=
  initial - (first_eaten + second_eaten)

/-- Theorem stating that given 36 initial pieces of candy, 
    after eating 17 and then 15 pieces, 4 pieces remain. -/
theorem candy_problem : remaining_candy 36 17 15 = 4 := by
  sorry

end candy_problem_l2713_271344


namespace race_theorem_l2713_271371

/-- Represents a runner in the race -/
structure Runner :=
  (speed : ℝ)

/-- The length of the race in meters -/
def race_length : ℝ := 1000

/-- The distance runner A finishes ahead of runner C -/
def a_ahead_of_c : ℝ := 200

/-- The distance runner B finishes ahead of runner C -/
def b_ahead_of_c : ℝ := 157.89473684210532

theorem race_theorem (A B C : Runner) :
  A.speed > B.speed ∧ B.speed > C.speed →
  a_ahead_of_c = A.speed * race_length / C.speed - race_length →
  b_ahead_of_c = B.speed * race_length / C.speed - race_length →
  A.speed * race_length / B.speed - race_length = a_ahead_of_c - b_ahead_of_c :=
by sorry

end race_theorem_l2713_271371


namespace min_balls_for_same_color_l2713_271345

def box : Finset (Fin 6) := Finset.univ
def color : Fin 6 → ℕ
  | 0 => 28  -- red
  | 1 => 20  -- green
  | 2 => 19  -- yellow
  | 3 => 13  -- blue
  | 4 => 11  -- white
  | 5 => 9   -- black

theorem min_balls_for_same_color : 
  ∀ n : ℕ, (∀ s : Finset (Fin 6), s.card = n → 
    (∃ c : Fin 6, (s.filter (λ i => color i = color c)).card < 15)) → 
  n < 76 :=
sorry

end min_balls_for_same_color_l2713_271345


namespace scientific_notation_pm25_express_y_in_terms_of_x_power_evaluation_l2713_271378

-- Problem 1
theorem scientific_notation_pm25 : 
  ∃ (a : ℝ) (n : ℤ), 0.0000025 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.5 ∧ n = -6 :=
sorry

-- Problem 2
theorem express_y_in_terms_of_x (x y : ℝ) :
  2 * x - 5 * y = 5 → y = 0.4 * x - 1 :=
sorry

-- Problem 3
theorem power_evaluation (x y : ℝ) :
  x + 2 * y - 4 = 0 → (2 : ℝ) ^ (2 * y) * (2 : ℝ) ^ (x - 2) = 4 :=
sorry

end scientific_notation_pm25_express_y_in_terms_of_x_power_evaluation_l2713_271378


namespace rectangular_prism_volume_l2713_271357

/-- The volume of a rectangular prism with face areas √2, √3, and √6 is √6 -/
theorem rectangular_prism_volume (a b c : ℝ) 
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : a * c = Real.sqrt 6) :
  a * b * c = Real.sqrt 6 := by
  sorry

end rectangular_prism_volume_l2713_271357


namespace prime_triplet_equation_l2713_271305

theorem prime_triplet_equation (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r →
  (p : ℚ) / q - 4 / (r + 1) = 1 →
  ((p = 7 ∧ q = 3 ∧ r = 2) ∨ (p = 5 ∧ q = 3 ∧ r = 5)) := by
  sorry

end prime_triplet_equation_l2713_271305


namespace murtha_pebble_collection_l2713_271373

/-- The sum of an arithmetic sequence with n terms, starting from a, with a common difference of d -/
def arithmetic_sum (n : ℕ) (a : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- The number of days Murtha collects pebbles -/
def days : ℕ := 15

/-- The number of pebbles Murtha collects on the first day -/
def initial_pebbles : ℕ := 2

/-- The daily increase in pebble collection -/
def daily_increase : ℕ := 1

theorem murtha_pebble_collection :
  arithmetic_sum days initial_pebbles daily_increase = 135 := by
  sorry

end murtha_pebble_collection_l2713_271373


namespace line_not_in_second_quadrant_l2713_271331

/-- A line that does not pass through the second quadrant -/
theorem line_not_in_second_quadrant (a : ℝ) (h : a ≠ 0) :
  ∀ x y : ℝ, x - y - a^2 = 0 → ¬(x < 0 ∧ y > 0) :=
by sorry

end line_not_in_second_quadrant_l2713_271331


namespace football_team_progress_l2713_271346

/-- Calculates the total progress of a football team given yards lost and gained -/
def footballProgress (yardsLost : Int) (yardsGained : Int) : Int :=
  yardsGained - yardsLost

/-- Theorem: A football team that lost 5 yards and then gained 11 yards has a total progress of 6 yards -/
theorem football_team_progress :
  footballProgress 5 11 = 6 := by
  sorry

end football_team_progress_l2713_271346
