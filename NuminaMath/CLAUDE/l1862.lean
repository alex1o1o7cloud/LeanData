import Mathlib

namespace NUMINAMATH_CALUDE_pasture_rent_is_870_l1862_186232

/-- Represents the rental information for a person --/
structure RentalInfo where
  horses : ℕ
  months : ℕ

/-- Calculates the total rent for a pasture given rental information and a known payment --/
def calculate_total_rent (a b c : RentalInfo) (b_payment : ℕ) : ℕ :=
  let total_horse_months := a.horses * a.months + b.horses * b.months + c.horses * c.months
  let cost_per_horse_month := b_payment / (b.horses * b.months)
  cost_per_horse_month * total_horse_months

/-- Theorem stating that the total rent for the pasture is 870 --/
theorem pasture_rent_is_870 (a b c : RentalInfo) (h1 : a.horses = 12) (h2 : a.months = 8)
    (h3 : b.horses = 16) (h4 : b.months = 9) (h5 : c.horses = 18) (h6 : c.months = 6)
    (h7 : calculate_total_rent a b c 360 = 870) : 
  calculate_total_rent a b c 360 = 870 := by
  sorry

end NUMINAMATH_CALUDE_pasture_rent_is_870_l1862_186232


namespace NUMINAMATH_CALUDE_sum_of_2_and_odd_prime_last_digit_l1862_186281

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def last_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_2_and_odd_prime_last_digit (p : ℕ) 
  (h_prime : is_prime p) 
  (h_odd : p % 2 = 1) 
  (h_greater_7 : p > 7) 
  (h_sum_not_single_digit : p + 2 ≥ 10) : 
  last_digit (p + 2) = 1 ∨ last_digit (p + 2) = 3 ∨ last_digit (p + 2) = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_2_and_odd_prime_last_digit_l1862_186281


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l1862_186235

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  let angle := Real.pi / 3  -- 60 degrees in radians
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
  angle = Real.arccos (dot_product / (magnitude a * magnitude b)) →
  magnitude a = 2 →
  magnitude b = 5 →
  magnitude (2 • a - b) = Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l1862_186235


namespace NUMINAMATH_CALUDE_function_range_l1862_186265

theorem function_range : 
  ∃ (min max : ℝ), min = -1 ∧ max = 3 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → min ≤ x^2 - 2*x ∧ x^2 - 2*x ≤ max) ∧
  (∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 3 ∧ 0 ≤ x₂ ∧ x₂ ≤ 3 ∧ 
    x₁^2 - 2*x₁ = min ∧ x₂^2 - 2*x₂ = max) := by
  sorry

end NUMINAMATH_CALUDE_function_range_l1862_186265


namespace NUMINAMATH_CALUDE_tobys_friends_l1862_186288

theorem tobys_friends (total_friends : ℕ) (boy_friends : ℕ) (girl_friends : ℕ) : 
  (boy_friends : ℚ) / (total_friends : ℚ) = 55 / 100 →
  boy_friends = 33 →
  girl_friends = 27 :=
by sorry

end NUMINAMATH_CALUDE_tobys_friends_l1862_186288


namespace NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l1862_186222

theorem tan_fifteen_pi_fourths : Real.tan (15 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l1862_186222


namespace NUMINAMATH_CALUDE_orange_bin_problem_l1862_186283

theorem orange_bin_problem (initial : ℕ) (removed : ℕ) (added : ℕ) : 
  initial = 50 → removed = 40 → added = 24 → initial - removed + added = 34 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_problem_l1862_186283


namespace NUMINAMATH_CALUDE_valid_words_length_10_l1862_186295

/-- Represents the number of valid words of length n -/
def validWords : ℕ → ℕ
  | 0 => 1  -- Base case: empty word
  | 1 => 2  -- Base case: "a" and "b"
  | (n+2) => validWords (n+1) + validWords n

/-- The problem statement -/
theorem valid_words_length_10 : validWords 10 = 144 := by
  sorry

end NUMINAMATH_CALUDE_valid_words_length_10_l1862_186295


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l1862_186299

theorem degree_to_radian_conversion (π : ℝ) :
  (1 : ℝ) * π / 180 = π / 180 →
  (-300 : ℝ) * π / 180 = -5 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l1862_186299


namespace NUMINAMATH_CALUDE_age_difference_l1862_186211

/-- Given three people A, B, and C, where C is 12 years younger than A,
    prove that the total age of A and B is 12 years more than the total age of B and C. -/
theorem age_difference (A B C : ℕ) (h : C = A - 12) :
  (A + B) - (B + C) = 12 :=
sorry

end NUMINAMATH_CALUDE_age_difference_l1862_186211


namespace NUMINAMATH_CALUDE_min_value_expression_l1862_186219

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → 42 + b^2 + 1/(a*b) ≤ 42 + y^2 + 1/(x*y) ∧
  ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ 42 + b₀^2 + 1/(a₀*b₀) = 17/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1862_186219


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1862_186290

theorem max_sum_of_factors (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 2277 →
  a + b + c + d ≤ 84 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1862_186290


namespace NUMINAMATH_CALUDE_B_power_200_l1862_186206

def B : Matrix (Fin 4) (Fin 4) ℤ :=
  !![0,0,0,1;
     1,0,0,0;
     0,1,0,0;
     0,0,1,0]

theorem B_power_200 : B ^ 200 = 1 := by sorry

end NUMINAMATH_CALUDE_B_power_200_l1862_186206


namespace NUMINAMATH_CALUDE_line_parallel_from_perpendicular_to_parallel_planes_l1862_186274

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Define the theorem
theorem line_parallel_from_perpendicular_to_parallel_planes
  (m n : Line) (α β : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_perp_α : perpendicular m α)
  (h_n_perp_β : perpendicular n β)
  (h_α_parallel_β : plane_parallel α β) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_line_parallel_from_perpendicular_to_parallel_planes_l1862_186274


namespace NUMINAMATH_CALUDE_trout_weight_l1862_186271

theorem trout_weight (num_trout num_catfish num_bluegill : ℕ) 
                     (weight_catfish weight_bluegill total_weight : ℚ) :
  num_trout = 4 →
  num_catfish = 3 →
  num_bluegill = 5 →
  weight_catfish = 3/2 →
  weight_bluegill = 5/2 →
  total_weight = 25 →
  ∃ weight_trout : ℚ,
    weight_trout * num_trout + weight_catfish * num_catfish + weight_bluegill * num_bluegill = total_weight ∧
    weight_trout = 2 :=
by sorry

end NUMINAMATH_CALUDE_trout_weight_l1862_186271


namespace NUMINAMATH_CALUDE_yuanxiao_sales_problem_l1862_186260

/-- Yuanxiao sales problem -/
theorem yuanxiao_sales_problem 
  (cost : ℝ) 
  (min_price : ℝ) 
  (base_sales : ℝ) 
  (base_price : ℝ) 
  (price_sensitivity : ℝ) 
  (max_price : ℝ) 
  (min_profit : ℝ)
  (h1 : cost = 20)
  (h2 : min_price = 25)
  (h3 : base_sales = 250)
  (h4 : base_price = 25)
  (h5 : price_sensitivity = 10)
  (h6 : max_price = 38)
  (h7 : min_profit = 2000) :
  let sales_volume (x : ℝ) := -price_sensitivity * x + (base_sales + price_sensitivity * base_price)
  let profit (x : ℝ) := (x - cost) * (sales_volume x)
  ∃ (optimal_price : ℝ) (max_profit : ℝ) (min_sales : ℝ),
    (∀ x, sales_volume x = -10 * x + 500) ∧
    (optimal_price = 35 ∧ max_profit = 2250 ∧ 
     ∀ x, x ≥ min_price → profit x ≤ max_profit) ∧
    (min_sales = 120 ∧
     ∀ x, min_price ≤ x ∧ x ≤ max_price → 
     profit x ≥ min_profit → sales_volume x ≥ min_sales) := by
  sorry

end NUMINAMATH_CALUDE_yuanxiao_sales_problem_l1862_186260


namespace NUMINAMATH_CALUDE_runner_daily_distance_l1862_186245

theorem runner_daily_distance (total_distance : ℝ) (total_weeks : ℝ) (daily_distance : ℝ) : 
  total_distance = 42 ∧ total_weeks = 3 ∧ daily_distance * (total_weeks * 7) = total_distance →
  daily_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_runner_daily_distance_l1862_186245


namespace NUMINAMATH_CALUDE_unique_prime_with_14_divisors_l1862_186238

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The theorem stating that there is exactly one prime p such that p^2 + 23 has 14 positive divisors -/
theorem unique_prime_with_14_divisors :
  ∃! p : ℕ, Nat.Prime p ∧ num_divisors (p^2 + 23) = 14 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_with_14_divisors_l1862_186238


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_nonnegative_l1862_186278

theorem negation_of_absolute_value_nonnegative :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x : ℝ, |x| < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_nonnegative_l1862_186278


namespace NUMINAMATH_CALUDE_zero_point_of_f_l1862_186207

-- Define the function
def f (x : ℝ) : ℝ := (x + 1)^2

-- State the theorem
theorem zero_point_of_f : 
  ∃ (x : ℝ), f x = 0 ∧ x = -1 :=
sorry

end NUMINAMATH_CALUDE_zero_point_of_f_l1862_186207


namespace NUMINAMATH_CALUDE_remainder_problem_l1862_186203

theorem remainder_problem (n : ℤ) : (3 * n) % 7 = 3 → n % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1862_186203


namespace NUMINAMATH_CALUDE_market_spending_l1862_186289

theorem market_spending (total_amount mildred_spent candice_spent : ℕ) 
  (h1 : total_amount = 100)
  (h2 : mildred_spent = 25)
  (h3 : candice_spent = 35) :
  total_amount - (mildred_spent + candice_spent) = 40 := by
  sorry

end NUMINAMATH_CALUDE_market_spending_l1862_186289


namespace NUMINAMATH_CALUDE_polynomial_coefficient_problem_l1862_186221

theorem polynomial_coefficient_problem (a b : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + b) * (2*x^2 - 3*x - 1) = 
    2*x^4 + (-5)*x^3 + (-6)*x^2 + ((-3*b - a)*x - b)) → 
  a = -1 ∧ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_problem_l1862_186221


namespace NUMINAMATH_CALUDE_product_equals_square_l1862_186286

theorem product_equals_square : 1000 * 1993 * 0.1993 * 10 = (1993 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l1862_186286


namespace NUMINAMATH_CALUDE_lines_properties_l1862_186287

/-- Two lines in 2D space -/
structure TwoLines where
  m : ℝ
  l1 : ℝ → ℝ → Prop := λ x y ↦ x + m * y - 1 = 0
  l2 : ℝ → ℝ → Prop := λ x y ↦ m * x + y - 1 = 0

/-- The distance between two parallel lines -/
def distance_between_parallel_lines (lines : TwoLines) : ℝ :=
  sorry

/-- Predicate for perpendicular lines -/
def are_perpendicular (lines : TwoLines) : Prop :=
  sorry

theorem lines_properties (lines : TwoLines) :
  (lines.l1 = lines.l2 → distance_between_parallel_lines lines = Real.sqrt 2) ∧
  (are_perpendicular lines → lines.m = 0) ∧
  lines.l2 0 1 := by
  sorry

end NUMINAMATH_CALUDE_lines_properties_l1862_186287


namespace NUMINAMATH_CALUDE_anthony_total_pencils_l1862_186272

/-- The total number of pencils Anthony has after receiving pencils from others -/
def total_pencils (initial : ℕ) (from_kathryn : ℕ) (from_greg : ℕ) (from_maria : ℕ) : ℕ :=
  initial + from_kathryn + from_greg + from_maria

/-- Theorem stating that Anthony's total pencils is 287 -/
theorem anthony_total_pencils :
  total_pencils 9 56 84 138 = 287 := by
  sorry

end NUMINAMATH_CALUDE_anthony_total_pencils_l1862_186272


namespace NUMINAMATH_CALUDE_unique_pie_purchase_l1862_186208

/-- Represents the number of pies bought by each classmate -/
structure PiePurchase where
  kostya : Nat
  volodya : Nat
  tolya : Nat

/-- Checks if a PiePurchase satisfies all the conditions of the problem -/
def isValidPurchase (p : PiePurchase) : Prop :=
  p.kostya + p.volodya + p.tolya = 13 ∧
  p.tolya = 2 * p.kostya ∧
  p.kostya < p.volodya ∧
  p.volodya < p.tolya

/-- The theorem stating that there is only one valid solution to the problem -/
theorem unique_pie_purchase :
  ∃! p : PiePurchase, isValidPurchase p ∧ p = ⟨3, 4, 6⟩ := by
  sorry

end NUMINAMATH_CALUDE_unique_pie_purchase_l1862_186208


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1862_186273

theorem complex_equation_solution :
  ∃ z : ℂ, (4 - 3 * Complex.I * z = 1 + 5 * Complex.I * z) ∧ (z = -3/8 * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1862_186273


namespace NUMINAMATH_CALUDE_ship_cannot_escape_illumination_l1862_186249

/-- Represents a lighthouse with a rotating beam -/
structure Lighthouse where
  beam_length : ℝ
  beam_velocity : ℝ

/-- Represents a ship moving towards the lighthouse -/
structure Ship where
  speed : ℝ
  initial_distance : ℝ

/-- Theorem: A ship cannot reach the lighthouse without being illuminated -/
theorem ship_cannot_escape_illumination (L : Lighthouse) (S : Ship) 
  (h1 : S.speed ≤ L.beam_velocity / 8)
  (h2 : S.initial_distance = L.beam_length) : 
  ∃ (t : ℝ), t > 0 ∧ S.initial_distance - S.speed * t > 0 ∧ 
  2 * π * L.beam_length / L.beam_velocity ≥ t :=
sorry

end NUMINAMATH_CALUDE_ship_cannot_escape_illumination_l1862_186249


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l1862_186269

/-- A parabola with equation y = x^2 + 6x + c has its vertex on the x-axis if and only if c = 9 -/
theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 + 6*x + c = 0 ∧ ∀ y : ℝ, y^2 + 6*y + c ≥ x^2 + 6*x + c) ↔ c = 9 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l1862_186269


namespace NUMINAMATH_CALUDE_larger_number_problem_l1862_186298

theorem larger_number_problem (x y : ℝ) (h_product : x * y = 30) (h_sum : x + y = 13) :
  max x y = 10 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1862_186298


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l1862_186294

theorem no_solutions_for_equation : ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ (2 / x + 3 / y = 1 / (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l1862_186294


namespace NUMINAMATH_CALUDE_photo_selection_choices_l1862_186205

theorem photo_selection_choices : ∀ n : ℕ, n = 5 →
  (Nat.choose n 3 + Nat.choose n 4 = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_photo_selection_choices_l1862_186205


namespace NUMINAMATH_CALUDE_sixth_root_of_unity_product_l1862_186293

theorem sixth_root_of_unity_product (s : ℂ) (h1 : s^6 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) = -6 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_unity_product_l1862_186293


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1862_186266

/-- An arithmetic sequence with common difference d and first term a_1 -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : d ≠ 0)
  (h3 : a 1 ≠ 0)
  (h4 : geometric_sequence (a 2) (a 4) (a 8)) :
  (a 1 + a 5 + a 9) / (a 2 + a 3) = 3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1862_186266


namespace NUMINAMATH_CALUDE_negation_equivalence_l1862_186248

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 5*x + 6 > 0) ↔ (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1862_186248


namespace NUMINAMATH_CALUDE_star_polygon_n_is_24_l1862_186250

/-- Represents an n-pointed star polygon -/
structure StarPolygon where
  n : ℕ
  angle_A : ℝ
  angle_B : ℝ
  angles_congruent : True  -- Represents that A₁, A₂, ..., Aₙ are congruent and B₁, B₂, ..., Bₙ are congruent
  angle_difference : angle_B = angle_A + 15

/-- Theorem stating that in a star polygon with the given properties, n = 24 -/
theorem star_polygon_n_is_24 (star : StarPolygon) : star.n = 24 := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_n_is_24_l1862_186250


namespace NUMINAMATH_CALUDE_garden_remaining_area_l1862_186213

/-- A rectangular garden plot with a shed in one corner -/
structure GardenPlot where
  length : ℝ
  width : ℝ
  shedSide : ℝ

/-- Calculate the remaining area of a garden plot available for planting -/
def remainingArea (garden : GardenPlot) : ℝ :=
  garden.length * garden.width - garden.shedSide * garden.shedSide

/-- Theorem: The remaining area of a 20ft by 18ft garden plot with a 4ft by 4ft shed is 344 sq ft -/
theorem garden_remaining_area :
  let garden : GardenPlot := { length := 20, width := 18, shedSide := 4 }
  remainingArea garden = 344 := by sorry

end NUMINAMATH_CALUDE_garden_remaining_area_l1862_186213


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l1862_186255

theorem weight_loss_challenge (original_weight : ℝ) (h : original_weight > 0) :
  let weight_after_loss := 0.87 * original_weight
  let final_measured_weight := 0.8874 * original_weight
  let clothes_weight := final_measured_weight - weight_after_loss
  clothes_weight / weight_after_loss = 0.02 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l1862_186255


namespace NUMINAMATH_CALUDE_lawn_mowing_solution_l1862_186230

/-- Represents the lawn mowing problem --/
def LawnMowingProblem (lawn_length lawn_width swath_width overlap flowerbed_diameter walking_rate : ℝ) : Prop :=
  let effective_width := (swath_width - overlap) / 12  -- Convert to feet
  let flowerbed_area := Real.pi * (flowerbed_diameter / 2) ^ 2
  let mowing_area := lawn_length * lawn_width - flowerbed_area
  let num_strips := lawn_width / effective_width
  let total_distance := num_strips * lawn_length
  let mowing_time := total_distance / walking_rate
  mowing_time = 2

/-- The main theorem stating the solution to the lawn mowing problem --/
theorem lawn_mowing_solution :
  LawnMowingProblem 100 160 30 6 20 4000 := by
  sorry

#check lawn_mowing_solution

end NUMINAMATH_CALUDE_lawn_mowing_solution_l1862_186230


namespace NUMINAMATH_CALUDE_smallest_integer_cube_root_l1862_186270

theorem smallest_integer_cube_root (m n : ℕ) (r : ℝ) : 
  (∀ k < n, ¬∃ (m' : ℕ) (r' : ℝ), m' < m ∧ 0 < r' ∧ r' < 1/500 ∧ m'^(1/3 : ℝ) = k + r') →
  0 < r →
  r < 1/500 →
  m^(1/3 : ℝ) = n + r →
  n = 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_cube_root_l1862_186270


namespace NUMINAMATH_CALUDE_price_reduction_l1862_186247

theorem price_reduction (x : ℝ) : 
  (1 - x / 100) * (1 - 25 / 100) = 1 - 77.5 / 100 → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_l1862_186247


namespace NUMINAMATH_CALUDE_bread_distribution_l1862_186244

theorem bread_distribution (total_loaves : ℕ) (num_people : ℕ) : 
  total_loaves = 100 →
  num_people = 5 →
  ∃ (a d : ℚ), 
    (∀ i : ℕ, i ≤ 5 → a + (i - 1) * d ≥ 0) ∧
    (a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = total_loaves) ∧
    ((a + 2*d) + (a + 3*d) + (a + 4*d) = 3 * (a + (a + d))) →
  (a + 4*d ≤ 30) :=
by sorry

end NUMINAMATH_CALUDE_bread_distribution_l1862_186244


namespace NUMINAMATH_CALUDE_even_function_decreasing_interval_l1862_186225

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + m * x + 3

-- Define the property of being an even function
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the decreasing interval
def decreasingInterval (f : ℝ → ℝ) : Set ℝ := {x | ∀ y, x ≤ y → f y ≤ f x}

-- State the theorem
theorem even_function_decreasing_interval :
  ∀ m : ℝ, isEven (f m) → decreasingInterval (f m) = Set.Ici 0 :=
sorry

end NUMINAMATH_CALUDE_even_function_decreasing_interval_l1862_186225


namespace NUMINAMATH_CALUDE_fraction_value_l1862_186210

theorem fraction_value (t k : ℚ) (f : ℚ) : 
  t = f * (k - 32) → t = 35 → k = 95 → f = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1862_186210


namespace NUMINAMATH_CALUDE_cubic_quadratic_comparison_quadratic_inequality_l1862_186279

-- Problem 1
theorem cubic_quadratic_comparison (x : ℝ) (h : x ≥ -1) :
  x^3 + 1 ≥ x^2 + x ∧ (x^3 + 1 = x^2 + x ↔ x = 1 ∨ x = -1) := by sorry

-- Problem 2
theorem quadratic_inequality (a x : ℝ) (h : a < 0) :
  x^2 - a*x - 6*a^2 > 0 ↔ x < 3*a ∨ x > -2*a := by sorry

end NUMINAMATH_CALUDE_cubic_quadratic_comparison_quadratic_inequality_l1862_186279


namespace NUMINAMATH_CALUDE_pet_store_puzzle_l1862_186209

theorem pet_store_puzzle (initial_birds initial_puppies initial_cats initial_spiders : ℕ)
  (final_total : ℕ) :
  initial_birds = 12 →
  initial_puppies = 9 →
  initial_cats = 5 →
  initial_spiders = 15 →
  final_total = 25 →
  ∃ (adopted_puppies : ℕ),
    adopted_puppies = initial_puppies - (
      initial_birds + initial_puppies + initial_cats + initial_spiders -
      (initial_birds / 2 + 7 + final_total)
    ) :=
by sorry

end NUMINAMATH_CALUDE_pet_store_puzzle_l1862_186209


namespace NUMINAMATH_CALUDE_rate_increase_factor_l1862_186282

/-- Reaction rate equation -/
def reaction_rate (k : ℝ) (C_CO : ℝ) (C_O2 : ℝ) : ℝ :=
  k * C_CO^2 * C_O2

/-- Theorem: When concentrations triple, rate increases by factor of 27 -/
theorem rate_increase_factor (k : ℝ) (C_CO : ℝ) (C_O2 : ℝ) :
  reaction_rate k (3 * C_CO) (3 * C_O2) = 27 * reaction_rate k C_CO C_O2 := by
  sorry


end NUMINAMATH_CALUDE_rate_increase_factor_l1862_186282


namespace NUMINAMATH_CALUDE_min_value_theorem_l1862_186241

theorem min_value_theorem (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_prod : x * y * z = 1) :
  x^2 + 8*x*y + 9*y^2 + 8*y*z + 2*z^2 ≥ 18 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧ 
    a^2 + 8*a*b + 9*b^2 + 8*b*c + 2*c^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1862_186241


namespace NUMINAMATH_CALUDE_quadratic_roots_conditions_l1862_186215

theorem quadratic_roots_conditions (k : ℝ) :
  let f : ℝ → ℝ := λ x => 2 * x^2 - 4 * k * x + x - (1 - 2 * k^2)
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) ↔ k ≤ 9/8 ∧
  (∀ x : ℝ, f x ≠ 0) ↔ k > 9/8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_conditions_l1862_186215


namespace NUMINAMATH_CALUDE_price_change_difference_l1862_186234

/-- 
Given that a price is increased by x percent and then decreased by y percent, 
resulting in the same price as the initial price, prove that 1/x - 1/y = -1/100.
-/
theorem price_change_difference (x y : ℝ) 
  (h : (1 + x/100) * (1 - y/100) = 1) : 
  1/x - 1/y = -1/100 :=
sorry

end NUMINAMATH_CALUDE_price_change_difference_l1862_186234


namespace NUMINAMATH_CALUDE_complex_multiplication_l1862_186267

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (1 - i)^2 * i = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1862_186267


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1862_186258

theorem z_in_fourth_quadrant (z : ℂ) (h : z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3)) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1862_186258


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1862_186262

/-- A line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 1 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x ∧ 
    ∀ x' y' : ℝ, y' = 3*x' + c → y'^2 = 12*x' → (x', y') = (x, y)) ↔ 
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1862_186262


namespace NUMINAMATH_CALUDE_textbook_profit_example_l1862_186268

/-- The profit of a textbook sale given its cost and selling prices -/
def textbook_profit (cost_price selling_price : ℝ) : ℝ :=
  selling_price - cost_price

/-- Theorem: The profit of a textbook sold by a bookstore is $11,
    given that the cost price is $44 and the selling price is $55. -/
theorem textbook_profit_example : textbook_profit 44 55 = 11 := by
  sorry

end NUMINAMATH_CALUDE_textbook_profit_example_l1862_186268


namespace NUMINAMATH_CALUDE_samantha_birth_year_proof_l1862_186280

/-- The year of the first AMC 8 -/
def first_amc8_year : ℕ := 1983

/-- The number of AMC 8 contests Samantha has taken -/
def samantha_amc8_count : ℕ := 9

/-- Samantha's age when she took her last AMC 8 -/
def samantha_age : ℕ := 13

/-- The year Samantha was born -/
def samantha_birth_year : ℕ := 1978

theorem samantha_birth_year_proof :
  samantha_birth_year = first_amc8_year + samantha_amc8_count - 1 - samantha_age :=
by sorry

end NUMINAMATH_CALUDE_samantha_birth_year_proof_l1862_186280


namespace NUMINAMATH_CALUDE_bus_ticket_solution_l1862_186254

/-- Represents the number and cost of bus tickets -/
structure BusTickets where
  total_tickets : ℕ
  total_cost : ℕ
  one_way_cost : ℕ
  round_trip_cost : ℕ

/-- Theorem stating the correct number of one-way and round-trip tickets -/
theorem bus_ticket_solution (tickets : BusTickets)
  (h1 : tickets.total_tickets = 99)
  (h2 : tickets.total_cost = 280)
  (h3 : tickets.one_way_cost = 2)
  (h4 : tickets.round_trip_cost = 3) :
  ∃ (one_way round_trip : ℕ),
    one_way + round_trip = tickets.total_tickets ∧
    one_way * tickets.one_way_cost + round_trip * tickets.round_trip_cost = tickets.total_cost ∧
    one_way = 17 ∧
    round_trip = 82 := by
  sorry

end NUMINAMATH_CALUDE_bus_ticket_solution_l1862_186254


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l1862_186224

theorem product_of_sum_and_difference (a b : ℝ) 
  (sum_eq : a + b = 7)
  (diff_eq : a - b = 2) :
  a * b = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l1862_186224


namespace NUMINAMATH_CALUDE_tank_dimension_l1862_186200

/-- Given a rectangular tank with dimensions 3 feet, 7 feet, and x feet,
    if the total surface area is 82 square feet, then x = 2 feet. -/
theorem tank_dimension (x : ℝ) : 
  2 * (3 * 7 + 3 * x + 7 * x) = 82 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_tank_dimension_l1862_186200


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l1862_186202

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ),
    (x^2 + a*x + b = 0 ↔ x = -3 - Complex.I * Real.sqrt 7 ∨ x = -3 + Complex.I * Real.sqrt 7) ∧
    (a = 6 ∧ b = 16) := by
  sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l1862_186202


namespace NUMINAMATH_CALUDE_subsidy_and_job_creation_l1862_186231

/-- Data for SZ province's "home appliances to the countryside" program in 2008 -/
structure ProgramData2008 where
  new_shops : ℕ
  jobs_created : ℕ
  units_sold : ℕ
  sales_amount : ℝ
  consumption_increase : ℝ
  subsidy_rate : ℝ

/-- Data for the program from 2008 to 2010 -/
structure ProgramData2008To2010 where
  total_jobs : ℕ
  increase_2010_vs_2009 : ℝ
  jobs_increase_2010_vs_2009 : ℝ

/-- Theorem about the subsidy funds needed in 2008 and job creation rate -/
theorem subsidy_and_job_creation 
  (data_2008 : ProgramData2008)
  (data_2008_to_2010 : ProgramData2008To2010)
  (h1 : data_2008.new_shops = 8000)
  (h2 : data_2008.jobs_created = 75000)
  (h3 : data_2008.units_sold = 1130000)
  (h4 : data_2008.sales_amount = 1.6 * 10^9)
  (h5 : data_2008.consumption_increase = 1.7)
  (h6 : data_2008.subsidy_rate = 0.13)
  (h7 : data_2008_to_2010.total_jobs = 247000)
  (h8 : data_2008_to_2010.increase_2010_vs_2009 = 0.5)
  (h9 : data_2008_to_2010.jobs_increase_2010_vs_2009 = 10/81) :
  ∃ (subsidy_funds : ℝ) (jobs_per_point : ℝ),
    subsidy_funds = 2.08 * 10^9 ∧ 
    jobs_per_point = 20000 := by
  sorry

end NUMINAMATH_CALUDE_subsidy_and_job_creation_l1862_186231


namespace NUMINAMATH_CALUDE_inequality_proof_l1862_186264

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) : 
  b / Real.sqrt (a + 2 * c) + c / Real.sqrt (b + 2 * d) + 
  d / Real.sqrt (c + 2 * a) + a / Real.sqrt (d + 2 * b) ≥ 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1862_186264


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l1862_186233

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 30*x - 33

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 11, f' x < 0 :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l1862_186233


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1862_186217

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 3) :
  (1/a + 1/b) ≥ 1 + 2*Real.sqrt 2/3 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 3 ∧ 1/a₀ + 1/b₀ = 1 + 2*Real.sqrt 2/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1862_186217


namespace NUMINAMATH_CALUDE_bus_capacity_is_193_l1862_186284

/-- Represents the seating capacity of a double-decker bus -/
def double_decker_bus_capacity (lower_left : ℕ) (lower_right : ℕ) (regular_seat_capacity : ℕ)
  (priority_seats : ℕ) (priority_seat_capacity : ℕ) (upper_left : ℕ) (upper_right : ℕ)
  (upper_seat_capacity : ℕ) (upper_back : ℕ) : ℕ :=
  (lower_left + lower_right) * regular_seat_capacity +
  priority_seats * priority_seat_capacity +
  (upper_left + upper_right) * upper_seat_capacity +
  upper_back

/-- Theorem stating the total seating capacity of the given double-decker bus -/
theorem bus_capacity_is_193 :
  double_decker_bus_capacity 15 12 2 4 1 20 20 3 15 = 193 := by
  sorry


end NUMINAMATH_CALUDE_bus_capacity_is_193_l1862_186284


namespace NUMINAMATH_CALUDE_ratio_sum_to_y_l1862_186292

theorem ratio_sum_to_y (w x y : ℚ) 
  (h1 : w / x = 2 / 3) 
  (h2 : w / y = 6 / 15) : 
  (x + y) / y = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_to_y_l1862_186292


namespace NUMINAMATH_CALUDE_vertical_line_intercept_difference_l1862_186263

/-- A vertical line passing through two points -/
structure VerticalLine where
  x : ℝ
  y1 : ℝ
  y2 : ℝ

/-- The x-intercept of a vertical line -/
def x_intercept (l : VerticalLine) : ℝ := l.x

/-- Theorem: For a vertical line passing through points C(7, 5) and D(7, -3),
    the difference between the x-intercept of the line and the y-coordinate of point C is 2 -/
theorem vertical_line_intercept_difference (l : VerticalLine) 
    (h1 : l.x = 7) 
    (h2 : l.y1 = 5) 
    (h3 : l.y2 = -3) : 
  x_intercept l - l.y1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_vertical_line_intercept_difference_l1862_186263


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1862_186242

/-- Given a triangle with inradius 2.5 cm and area 40 cm², its perimeter is 32 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 40 → A = r * (p / 2) → p = 32 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1862_186242


namespace NUMINAMATH_CALUDE_set_operation_equality_l1862_186204

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {1, 2}
def B : Finset Int := {-2, 1, 2}

theorem set_operation_equality : A ∪ (U \ B) = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_operation_equality_l1862_186204


namespace NUMINAMATH_CALUDE_mistaken_divisor_l1862_186228

/-- Given a division with remainder 0, correct divisor 21, correct quotient 24,
    and a mistaken quotient of 42, prove that the mistaken divisor is 12. -/
theorem mistaken_divisor (dividend : ℕ) (mistaken_divisor : ℕ) : 
  dividend % 21 = 0 ∧ 
  dividend / 21 = 24 ∧ 
  dividend / mistaken_divisor = 42 →
  mistaken_divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_mistaken_divisor_l1862_186228


namespace NUMINAMATH_CALUDE_sin_45_75_plus_sin_45_15_l1862_186261

theorem sin_45_75_plus_sin_45_15 :
  Real.sin (45 * π / 180) * Real.sin (75 * π / 180) +
  Real.sin (45 * π / 180) * Real.sin (15 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_75_plus_sin_45_15_l1862_186261


namespace NUMINAMATH_CALUDE_small_mold_radius_l1862_186227

theorem small_mold_radius (R : ℝ) (n : ℕ) (r : ℝ) : 
  R = 2 → n = 64 → (2 / 3 * Real.pi * R^3) = (n * (2 / 3 * Real.pi * r^3)) → r = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_small_mold_radius_l1862_186227


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1862_186201

/-- The imaginary part of 2i / (2 + i^3) is equal to 4/5 -/
theorem imaginary_part_of_complex_fraction :
  Complex.im (2 * Complex.I / (2 + Complex.I ^ 3)) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1862_186201


namespace NUMINAMATH_CALUDE_f_negative_three_l1862_186216

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- f is an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- For any positive number x, f(2+x) = -2f(2-x)
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f (2 + x) = -2 * f (2 - x)

-- Main theorem
theorem f_negative_three (h1 : is_even f) (h2 : satisfies_condition f) (h3 : f (-1) = 4) :
  f (-3) = -8 := by
  sorry


end NUMINAMATH_CALUDE_f_negative_three_l1862_186216


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1862_186212

theorem arithmetic_computation : 2 + 5 * 3 - 4 + 7 * 2 / 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1862_186212


namespace NUMINAMATH_CALUDE_solution_set_f_geq_8_range_of_a_when_solution_exists_l1862_186297

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Theorem for the solution set of f(x) ≥ 8
theorem solution_set_f_geq_8 :
  {x : ℝ | f x ≥ 8} = {x : ℝ | x ≤ -5 ∨ x ≥ 3} := by sorry

-- Theorem for the range of a when the solution set of f(x) < a^2 - 3a is not empty
theorem range_of_a_when_solution_exists (a : ℝ) :
  (∃ x, f x < a^2 - 3*a) → (a < -1 ∨ a > 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_8_range_of_a_when_solution_exists_l1862_186297


namespace NUMINAMATH_CALUDE_thursday_tuesday_difference_l1862_186259

/-- The amount of money Max's mom gave him on Tuesday -/
def tuesday_amount : ℕ := 8

/-- The amount of money Max's mom gave him on Wednesday -/
def wednesday_amount : ℕ := 5 * tuesday_amount

/-- The amount of money Max's mom gave him on Thursday -/
def thursday_amount : ℕ := wednesday_amount + 9

/-- The theorem stating the difference between Thursday's and Tuesday's amounts -/
theorem thursday_tuesday_difference :
  thursday_amount - tuesday_amount = 41 := by sorry

end NUMINAMATH_CALUDE_thursday_tuesday_difference_l1862_186259


namespace NUMINAMATH_CALUDE_angle_of_inclination_30_degrees_l1862_186291

theorem angle_of_inclination_30_degrees (x y : ℝ) :
  2 * x - 2 * Real.sqrt 3 * y + 1 = 0 →
  Real.arctan (Real.sqrt 3 / 3) = 30 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_angle_of_inclination_30_degrees_l1862_186291


namespace NUMINAMATH_CALUDE_two_rats_boring_theorem_l1862_186236

/-- The distance burrowed by the larger rat on day n -/
def larger_rat_distance (n : ℕ) : ℚ := 2^(n-1)

/-- The distance burrowed by the smaller rat on day n -/
def smaller_rat_distance (n : ℕ) : ℚ := (1/2)^(n-1)

/-- The total distance burrowed by both rats after n days -/
def total_distance (n : ℕ) : ℚ := 
  (Finset.range n).sum (λ i => larger_rat_distance (i+1) + smaller_rat_distance (i+1))

/-- The theorem stating that the total distance burrowed after 5 days is 32 15/16 -/
theorem two_rats_boring_theorem : total_distance 5 = 32 + 15/16 := by sorry

end NUMINAMATH_CALUDE_two_rats_boring_theorem_l1862_186236


namespace NUMINAMATH_CALUDE_differential_equation_satisfaction_l1862_186277

open Real

theorem differential_equation_satisfaction (n : ℝ) (x : ℝ) (h : x ≠ -1) :
  let y : ℝ → ℝ := λ x => (x + 1)^n * (exp x - 1)
  deriv y x - (n * y x) / (x + 1) = exp x * (1 + x)^n := by
  sorry

end NUMINAMATH_CALUDE_differential_equation_satisfaction_l1862_186277


namespace NUMINAMATH_CALUDE_prob_one_makes_shot_is_point_seven_l1862_186275

/-- The probability that at least one player makes a shot -/
def prob_at_least_one_makes_shot (prob_a prob_b : ℝ) : ℝ :=
  1 - (1 - prob_a) * (1 - prob_b)

/-- Theorem: Given the shooting success rates of players A and B,
    the probability that at least one of them makes a shot is 0.7 -/
theorem prob_one_makes_shot_is_point_seven :
  prob_at_least_one_makes_shot 0.5 0.4 = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_makes_shot_is_point_seven_l1862_186275


namespace NUMINAMATH_CALUDE_divisor_power_result_l1862_186243

theorem divisor_power_result (k : ℕ) : 
  21^k ∣ 435961 → 7^k - k^7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_power_result_l1862_186243


namespace NUMINAMATH_CALUDE_square_measurement_error_l1862_186214

theorem square_measurement_error (actual_side : ℝ) (measured_side : ℝ) :
  measured_side^2 = actual_side^2 * (1 + 0.050625) →
  (measured_side - actual_side) / actual_side = 0.025 := by
sorry

end NUMINAMATH_CALUDE_square_measurement_error_l1862_186214


namespace NUMINAMATH_CALUDE_equation_solutions_l1862_186276

/-- Given two equations about x and k -/
theorem equation_solutions (x k : ℚ) : 
  (3 * (2 * x - 1) = k + 2 * x) →
  ((x - k) / 2 = x + 2 * k) →
  (
    /- Part 1 -/
    (x = 4 → (x - k) / 2 = x + 2 * k → x = -65) ∧
    /- Part 2 -/
    (∃ x, 3 * (2 * x - 1) = k + 2 * x ∧ (x - k) / 2 = x + 2 * k) → k = -1/7
  ) := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1862_186276


namespace NUMINAMATH_CALUDE_odd_numbers_mean_median_impossibility_l1862_186253

theorem odd_numbers_mean_median_impossibility :
  ∀ (a b c d e f g : ℤ),
    a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g →
    Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f ∧ Odd g →
    (a + b + c + d + e + f + g) / 7 ≠ d + 3 / 7 :=
by sorry

end NUMINAMATH_CALUDE_odd_numbers_mean_median_impossibility_l1862_186253


namespace NUMINAMATH_CALUDE_product_derivative_at_zero_l1862_186226

/-- Given differentiable real functions f, g, h, prove that (fgh)'(0) = 16 -/
theorem product_derivative_at_zero
  (f g h : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (hh : Differentiable ℝ h)
  (hf0 : f 0 = 1)
  (hg0 : g 0 = 2)
  (hh0 : h 0 = 3)
  (hgh : deriv (g * h) 0 = 4)
  (hhf : deriv (h * f) 0 = 5)
  (hfg : deriv (f * g) 0 = 6) :
  deriv (f * g * h) 0 = 16 := by
sorry

end NUMINAMATH_CALUDE_product_derivative_at_zero_l1862_186226


namespace NUMINAMATH_CALUDE_marble_sum_l1862_186251

def marble_problem (fabian kyle miles : ℕ) : Prop :=
  fabian = 15 ∧ fabian = 3 * kyle ∧ fabian = 5 * miles

theorem marble_sum (fabian kyle miles : ℕ) 
  (h : marble_problem fabian kyle miles) : kyle + miles = 8 := by
  sorry

end NUMINAMATH_CALUDE_marble_sum_l1862_186251


namespace NUMINAMATH_CALUDE_sanchez_grade_calculation_l1862_186256

theorem sanchez_grade_calculation (total_students : ℕ) (below_b_percentage : ℚ) 
  (h1 : total_students = 60) 
  (h2 : below_b_percentage = 40 / 100) : 
  ↑total_students * (1 - below_b_percentage) = 36 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_grade_calculation_l1862_186256


namespace NUMINAMATH_CALUDE_alice_probability_after_three_turns_l1862_186285

-- Define the probabilities
def alice_to_bob : ℚ := 2/3
def alice_keeps : ℚ := 1/3
def bob_to_alice : ℚ := 1/3
def bob_keeps : ℚ := 2/3

-- Define the game state after three turns
def alice_has_ball_after_three_turns : ℚ :=
  alice_to_bob * bob_to_alice +
  alice_to_bob * bob_keeps * bob_to_alice +
  alice_keeps * alice_to_bob * bob_to_alice +
  alice_keeps * alice_keeps

-- Theorem statement
theorem alice_probability_after_three_turns :
  alice_has_ball_after_three_turns = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_alice_probability_after_three_turns_l1862_186285


namespace NUMINAMATH_CALUDE_textbook_selection_ways_l1862_186257

/-- The number of ways to select textbooks from two categories -/
def select_textbooks (required : ℕ) (selective : ℕ) (total : ℕ) : ℕ :=
  (required.choose 1 * selective.choose 2) + (required.choose 2 * selective.choose 1)

/-- Theorem stating that selecting 3 textbooks from 2 required and 3 selective, 
    with at least one from each category, can be done in 9 ways -/
theorem textbook_selection_ways :
  select_textbooks 2 3 3 = 9 := by
  sorry

#eval select_textbooks 2 3 3

end NUMINAMATH_CALUDE_textbook_selection_ways_l1862_186257


namespace NUMINAMATH_CALUDE_pizza_area_increase_l1862_186223

/-- The percentage increase in area from a circle with radius 5 to a circle with radius 7 -/
theorem pizza_area_increase : ∀ (π : ℝ), π > 0 →
  (π * 7^2 - π * 5^2) / (π * 5^2) * 100 = 96 := by
  sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l1862_186223


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1862_186252

-- Define the solution set for the inequality
def solution_set (m : ℝ) : Set ℝ :=
  if m < -4 then { x | -1 < x ∧ x < 1 / (m + 3) }
  else if m = -4 then ∅
  else if m > -4 ∧ m < -3 then { x | 1 / (m + 3) < x ∧ x < -1 }
  else if m = -3 then { x | x > -1 }
  else { x | x < -1 ∨ x > 1 / (m + 3) }

-- Theorem statement
theorem inequality_solution_set (m : ℝ) :
  { x : ℝ | (m + 3) * x - 1 > 0 } = solution_set m := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1862_186252


namespace NUMINAMATH_CALUDE_expression_evaluation_l1862_186220

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/3
  ((2*x + 3*y)^2 - (2*x + 3*y)*(2*x - 3*y)) / (3*y) = -6 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1862_186220


namespace NUMINAMATH_CALUDE_total_cookies_l1862_186246

/-- Given that each bag contains 11 cookies and there are 3 bags,
    prove that the total number of cookies is 33. -/
theorem total_cookies (cookies_per_bag : ℕ) (num_bags : ℕ) (h1 : cookies_per_bag = 11) (h2 : num_bags = 3) :
  cookies_per_bag * num_bags = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l1862_186246


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1862_186237

theorem fraction_equals_zero (x : ℝ) (h : x ≠ 0) :
  x = 3 → (2 * x - 6) / (5 * x) = 0 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1862_186237


namespace NUMINAMATH_CALUDE_red_ball_probability_l1862_186229

/-- Given a bag of balls with the following properties:
  * There are n total balls
  * There are m white balls
  * The probability of drawing at least one red ball when two balls are drawn is 3/5
  * The expected number of white balls in 6 draws with replacement is 4
  Prove that the probability of drawing a red ball on the second draw,
  given that the first draw was red, is 1/5. -/
theorem red_ball_probability (n m : ℕ) 
  (h1 : 1 - (m.choose 2 : ℚ) / (n.choose 2 : ℚ) = 3/5)
  (h2 : 6 * (m : ℚ) / (n : ℚ) = 4) :
  (n - m : ℚ) / ((n - 1) : ℚ) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_red_ball_probability_l1862_186229


namespace NUMINAMATH_CALUDE_square_binomial_constant_l1862_186240

theorem square_binomial_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 200*x + c = (x + a)^2) → c = 10000 := by
sorry

end NUMINAMATH_CALUDE_square_binomial_constant_l1862_186240


namespace NUMINAMATH_CALUDE_length_AB_l1862_186239

/-- A line passing through (2,0) with slope 2 -/
def line_l (x y : ℝ) : Prop := y = 2 * x - 4

/-- The curve y^2 - 4x = 0 -/
def curve (x y : ℝ) : Prop := y^2 = 4 * x

/-- Point A is on both the line and the curve -/
def point_A (x y : ℝ) : Prop := line_l x y ∧ curve x y

/-- Point B is on both the line and the curve, and is different from A -/
def point_B (x y : ℝ) : Prop := line_l x y ∧ curve x y ∧ (x, y) ≠ (1, -2)

/-- The main theorem: the length of AB is 3√5 -/
theorem length_AB :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  point_A x₁ y₁ → point_B x₂ y₂ →
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 3 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_length_AB_l1862_186239


namespace NUMINAMATH_CALUDE_composite_n_pow_2016_plus_4_l1862_186296

theorem composite_n_pow_2016_plus_4 (n : ℕ) (h : n > 1) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^2016 + 4 = a * b :=
by
  sorry

end NUMINAMATH_CALUDE_composite_n_pow_2016_plus_4_l1862_186296


namespace NUMINAMATH_CALUDE_bench_press_increase_factor_l1862_186218

theorem bench_press_increase_factor 
  (initial_weight : ℝ) 
  (injury_decrease_percent : ℝ) 
  (final_weight : ℝ) 
  (h1 : initial_weight = 500) 
  (h2 : injury_decrease_percent = 80) 
  (h3 : final_weight = 300) : 
  final_weight / (initial_weight * (1 - injury_decrease_percent / 100)) = 3 := by
sorry

end NUMINAMATH_CALUDE_bench_press_increase_factor_l1862_186218
