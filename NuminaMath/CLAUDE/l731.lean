import Mathlib

namespace NUMINAMATH_CALUDE_roots_sum_relation_l731_73141

theorem roots_sum_relation (a b c d : ℝ) : 
  (∀ x, x^2 + c*x + d = 0 ↔ x = a ∨ x = b) → a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_relation_l731_73141


namespace NUMINAMATH_CALUDE_exists_interior_points_l731_73107

/-- A point in the plane represented by its coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Check if a point is in the interior of a triangle -/
def interior_point (p : Point) (a b c : Point) : Prop :=
  ∃ (u v w : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧ u + v + w = 1 ∧
  p.x = u * a.x + v * b.x + w * c.x ∧
  p.y = u * a.y + v * b.y + w * c.y

/-- The main theorem -/
theorem exists_interior_points (n : ℕ) (S : Finset Point) :
  S.card = n →
  (∀ (a b c : Point), a ∈ S → b ∈ S → c ∈ S → ¬collinear a b c) →
  ∃ (P : Finset Point), P.card = 2 * n - 5 ∧
    ∀ (a b c : Point), a ∈ S → b ∈ S → c ∈ S →
      ∃ (p : Point), p ∈ P ∧ interior_point p a b c :=
sorry

end NUMINAMATH_CALUDE_exists_interior_points_l731_73107


namespace NUMINAMATH_CALUDE_min_m_value_x_range_l731_73150

-- Define the variables and conditions
variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hab : a + b = 1)

-- Theorem 1: Minimum value of m
theorem min_m_value : 
  (∀ m : ℝ, a * b ≤ m → m ≥ 1/4) ∧ 
  (∃ m : ℝ, m = 1/4 ∧ a * b ≤ m) :=
sorry

-- Theorem 2: Range of x
theorem x_range : 
  (∀ x : ℝ, 4/a + 1/b ≥ |2*x - 1| - |x + 2|) ↔ 
  (∀ x : ℝ, x ∈ Set.Icc (-6) 12) :=
sorry

end NUMINAMATH_CALUDE_min_m_value_x_range_l731_73150


namespace NUMINAMATH_CALUDE_sum_first_12_even_numbers_l731_73119

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * (i + 1))

theorem sum_first_12_even_numbers :
  (first_n_even_numbers 12).sum = 156 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_12_even_numbers_l731_73119


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l731_73101

theorem area_ratio_of_squares (side_A side_B : ℝ) (h1 : side_A = 36) (h2 : side_B = 42) :
  (side_A ^ 2) / (side_B ^ 2) = 36 / 49 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l731_73101


namespace NUMINAMATH_CALUDE_farm_animals_l731_73180

theorem farm_animals (pigs : ℕ) (cows : ℕ) (goats : ℕ) : 
  cows = 2 * pigs - 3 →
  goats = cows + 6 →
  pigs + cows + goats = 50 →
  pigs = 10 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l731_73180


namespace NUMINAMATH_CALUDE_paving_cost_proof_l731_73116

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Proof that the cost of paving the given floor is Rs. 28,875 -/
theorem paving_cost_proof :
  paving_cost 5.5 3.75 1400 = 28875 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_proof_l731_73116


namespace NUMINAMATH_CALUDE_phone_number_probability_l731_73112

theorem phone_number_probability (n : ℕ) (h : n = 10) :
  let p : ℚ := 1 / n
  1 - (1 - p) * (1 - p) = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_phone_number_probability_l731_73112


namespace NUMINAMATH_CALUDE_probability_is_two_fifths_l731_73158

/-- A diagram with five triangles, two of which are shaded. -/
structure Diagram where
  triangles : Finset (Fin 5)
  shaded : Finset (Fin 5)
  total_triangles : triangles.card = 5
  shaded_triangles : shaded.card = 2
  shaded_subset : shaded ⊆ triangles

/-- The probability of selecting a shaded triangle from the diagram. -/
def probability_shaded (d : Diagram) : ℚ :=
  d.shaded.card / d.triangles.card

/-- Theorem stating that the probability of selecting a shaded triangle is 2/5. -/
theorem probability_is_two_fifths (d : Diagram) :
  probability_shaded d = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_two_fifths_l731_73158


namespace NUMINAMATH_CALUDE_polynomial_has_solution_mod_prime_l731_73105

/-- The polynomial f(x) = x^6 - 11x^4 + 36x^2 - 36 -/
def f (x : ℤ) : ℤ := x^6 - 11*x^4 + 36*x^2 - 36

/-- For any prime p, there exists an x such that f(x) ≡ 0 (mod p) -/
theorem polynomial_has_solution_mod_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ x : ℤ, f x ≡ 0 [ZMOD p] := by sorry

end NUMINAMATH_CALUDE_polynomial_has_solution_mod_prime_l731_73105


namespace NUMINAMATH_CALUDE_milk_packets_returned_l731_73190

/-- Given information about milk packets and their prices, prove the number of returned packets. -/
theorem milk_packets_returned (total : ℕ) (avg_price all_remaining returned : ℚ) :
  total = 5 ∧ 
  avg_price = 20 ∧ 
  all_remaining = 12 ∧ 
  returned = 32 →
  ∃ (x : ℕ), 
    x ≤ total ∧ 
    (total : ℚ) * avg_price = (total - x : ℚ) * all_remaining + (x : ℚ) * returned ∧
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_milk_packets_returned_l731_73190


namespace NUMINAMATH_CALUDE_factorization_equality_l731_73125

theorem factorization_equality (x y : ℝ) : 3 * x^2 * y - 3 * y^3 = 3 * y * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l731_73125


namespace NUMINAMATH_CALUDE_mary_stickers_remaining_l731_73169

theorem mary_stickers_remaining (initial_stickers : ℕ) 
                                 (front_page_stickers : ℕ) 
                                 (other_pages : ℕ) 
                                 (stickers_per_other_page : ℕ) 
                                 (h1 : initial_stickers = 89)
                                 (h2 : front_page_stickers = 3)
                                 (h3 : other_pages = 6)
                                 (h4 : stickers_per_other_page = 7) : 
  initial_stickers - (front_page_stickers + other_pages * stickers_per_other_page) = 44 := by
  sorry

end NUMINAMATH_CALUDE_mary_stickers_remaining_l731_73169


namespace NUMINAMATH_CALUDE_unique_solution_l731_73155

theorem unique_solution (x y z : ℝ) 
  (hx : x > 3) (hy : y > 3) (hz : z > 3)
  (h : ((x + 2)^2) / (y + z - 2) + ((y + 4)^2) / (z + x - 4) + ((z + 6)^2) / (x + y - 6) = 36) :
  x = 10 ∧ y = 8 ∧ z = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l731_73155


namespace NUMINAMATH_CALUDE_total_tv_time_l731_73185

def missy_reality_shows : List ℕ := [28, 35, 42, 39, 29]
def missy_cartoons : ℕ := 2
def missy_cartoon_duration : ℕ := 10

def john_action_movies : List ℕ := [90, 110, 95]
def john_comedy_duration : ℕ := 25

def lily_documentaries : List ℕ := [45, 55, 60, 52]

def ad_breaks : List ℕ := [8, 6, 12, 9, 7, 11]

def num_viewers : ℕ := 3

theorem total_tv_time :
  (missy_reality_shows.sum + missy_cartoons * missy_cartoon_duration +
   john_action_movies.sum + john_comedy_duration +
   lily_documentaries.sum +
   num_viewers * ad_breaks.sum) = 884 := by
  sorry

end NUMINAMATH_CALUDE_total_tv_time_l731_73185


namespace NUMINAMATH_CALUDE_one_third_squared_times_one_eighth_l731_73166

theorem one_third_squared_times_one_eighth : (1 / 3 : ℚ)^2 * (1 / 8 : ℚ) = 1 / 72 := by
  sorry

end NUMINAMATH_CALUDE_one_third_squared_times_one_eighth_l731_73166


namespace NUMINAMATH_CALUDE_tangent_line_equation_l731_73102

/-- The circle C -/
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

/-- The point P -/
def P : ℝ × ℝ := (2, 4)

/-- The tangent line -/
def tangent_line (x y : ℝ) : Prop := x + 2*y - 10 = 0

/-- Theorem: The tangent line to circle C passing through point P has the equation x + 2y - 10 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, C x y → (x = P.1 ∧ y = P.2) → tangent_line x y :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l731_73102


namespace NUMINAMATH_CALUDE_total_parents_is_fourteen_l731_73161

/-- Represents the field trip to the zoo --/
structure FieldTrip where
  fifth_graders : ℕ
  sixth_graders : ℕ
  seventh_graders : ℕ
  teachers : ℕ
  buses : ℕ
  seats_per_bus : ℕ

/-- Calculates the total number of parents on the field trip --/
def total_parents (trip : FieldTrip) : ℕ :=
  trip.buses * trip.seats_per_bus - (trip.fifth_graders + trip.sixth_graders + trip.seventh_graders + trip.teachers)

/-- Theorem stating that the total number of parents on the trip is 14 --/
theorem total_parents_is_fourteen (trip : FieldTrip) 
  (h1 : trip.fifth_graders = 109)
  (h2 : trip.sixth_graders = 115)
  (h3 : trip.seventh_graders = 118)
  (h4 : trip.teachers = 4)
  (h5 : trip.buses = 5)
  (h6 : trip.seats_per_bus = 72) :
  total_parents trip = 14 := by
  sorry

#eval total_parents { fifth_graders := 109, sixth_graders := 115, seventh_graders := 118, teachers := 4, buses := 5, seats_per_bus := 72 }

end NUMINAMATH_CALUDE_total_parents_is_fourteen_l731_73161


namespace NUMINAMATH_CALUDE_gas_volume_ranking_l731_73143

-- Define the regions
inductive Region
| West
| NonWest
| Russia

-- Define the gas volume per capita for each region
def gas_volume (r : Region) : ℝ :=
  match r with
  | Region.West => 21428
  | Region.NonWest => 26848.55
  | Region.Russia => 302790.13

-- Theorem to prove the ranking
theorem gas_volume_ranking :
  gas_volume Region.Russia > gas_volume Region.NonWest ∧
  gas_volume Region.NonWest > gas_volume Region.West :=
by sorry

end NUMINAMATH_CALUDE_gas_volume_ranking_l731_73143


namespace NUMINAMATH_CALUDE_partnership_profit_b_profit_calculation_l731_73170

/-- Profit calculation in a partnership --/
theorem partnership_profit (a_investment b_investment : ℕ) 
  (a_period b_period : ℕ) (total_profit : ℕ) : ℕ :=
  let a_share := a_investment * a_period
  let b_share := b_investment * b_period
  let total_share := a_share + b_share
  let b_profit := (b_share * total_profit) / total_share
  b_profit

/-- B's profit in the given partnership scenario --/
theorem b_profit_calculation : 
  partnership_profit 3 1 2 1 31500 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_b_profit_calculation_l731_73170


namespace NUMINAMATH_CALUDE_min_value_on_negative_reals_l731_73164

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

-- State the theorem
theorem min_value_on_negative_reals (a b : ℝ) :
  (∀ x > 0, f a b x ≤ 5) ∧ (∃ x > 0, f a b x = 5) →
  (∀ x < 0, f a b x ≥ -1) ∧ (∃ x < 0, f a b x = -1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_negative_reals_l731_73164


namespace NUMINAMATH_CALUDE_triangle_to_square_area_ratio_l731_73194

/-- Represents a square divided into a 5x5 grid -/
structure GridSquare where
  side_length : ℝ
  small_square_count : ℕ
  small_square_count_eq : small_square_count = 5

/-- Represents a triangle within the GridSquare -/
structure Triangle where
  grid : GridSquare
  covered_squares : ℝ
  covered_squares_eq : covered_squares = 3.5

theorem triangle_to_square_area_ratio 
  (grid : GridSquare) 
  (triangle : Triangle) 
  (h_triangle : triangle.grid = grid) :
  (triangle.covered_squares * (grid.side_length / grid.small_square_count)^2) / 
  (grid.side_length^2) = 7 / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_to_square_area_ratio_l731_73194


namespace NUMINAMATH_CALUDE_roots_sum_greater_than_twice_sqrt_a_l731_73195

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x

theorem roots_sum_greater_than_twice_sqrt_a (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > Real.exp 1) 
  (hx₁ : f a x₁ = 0) 
  (hx₂ : f a x₂ = 0) 
  (hx_dist : x₁ ≠ x₂) : 
  x₁ + x₂ > 2 * Real.sqrt a := by
sorry

end NUMINAMATH_CALUDE_roots_sum_greater_than_twice_sqrt_a_l731_73195


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l731_73149

/-- The eccentricity of a hyperbola whose focus coincides with the focus of a specific parabola -/
theorem hyperbola_eccentricity (a : ℝ) : 
  let parabola := {(x, y) : ℝ × ℝ | x^2 = -4 * Real.sqrt 5 * y}
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a + y^2 / 4 = 1}
  let parabola_focus : ℝ × ℝ := (0, -Real.sqrt 5)
  ∃ (c : ℝ), c > 0 ∧ (c, 0) ∈ hyperbola ∧ (-c, 0) ∈ hyperbola ∧ 
    (parabola_focus ∈ hyperbola → (Real.sqrt 5) / 2 = c / 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l731_73149


namespace NUMINAMATH_CALUDE_partnership_profit_l731_73184

/-- Calculates the total profit of a partnership given the investments and one partner's share of the profit. -/
def calculate_total_profit (a_investment b_investment c_investment c_profit : ℕ) : ℕ :=
  let ratio_sum := (a_investment / (a_investment.gcd b_investment).gcd c_investment) +
                   (b_investment / (a_investment.gcd b_investment).gcd c_investment) +
                   (c_investment / (a_investment.gcd b_investment).gcd c_investment)
  let c_ratio := c_investment / (a_investment.gcd b_investment).gcd c_investment
  (ratio_sum * c_profit) / c_ratio

/-- The total profit of the partnership is 80000 given the investments and C's profit share. -/
theorem partnership_profit :
  calculate_total_profit 27000 72000 81000 36000 = 80000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l731_73184


namespace NUMINAMATH_CALUDE_division_problem_l731_73134

theorem division_problem (n : ℕ) : 
  n / 16 = 10 ∧ n % 16 = 1 → n = 161 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l731_73134


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_l731_73174

theorem product_of_four_consecutive_integers (n : ℤ) :
  (n - 1) * n * (n + 1) * (n + 2) = (n^2 + n - 1)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_l731_73174


namespace NUMINAMATH_CALUDE_soap_brand_usage_ratio_l731_73118

/-- Given a survey of households and their soap brand usage, prove the ratio of households
    using only brand B to those using both brands A and B. -/
theorem soap_brand_usage_ratio 
  (total : ℕ) 
  (neither : ℕ) 
  (only_A : ℕ) 
  (both : ℕ) 
  (h1 : total = 200)
  (h2 : neither = 80)
  (h3 : only_A = 60)
  (h4 : both = 5)
  (h5 : neither + only_A + both < total) :
  (total - (neither + only_A + both)) / both = 11 :=
by sorry

end NUMINAMATH_CALUDE_soap_brand_usage_ratio_l731_73118


namespace NUMINAMATH_CALUDE_tutor_schedules_lcm_l731_73186

/-- The work schedules of the tutors -/
def tutor_schedules : List Nat := [5, 6, 8, 9, 10]

/-- The theorem stating that the LCM of the tutor schedules is 360 -/
theorem tutor_schedules_lcm :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9) 10 = 360 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedules_lcm_l731_73186


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l731_73179

theorem expression_simplification_and_evaluation :
  let f (x : ℚ) := (x^2 - 4) / (x^2 - 4*x + 4) + (x / (x^2 - x)) / ((x - 2) / (x - 1))
  f (-1) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l731_73179


namespace NUMINAMATH_CALUDE_linear_equation_condition_l731_73111

theorem linear_equation_condition (a : ℝ) : 
  (∀ x, ∃ k m, (a + 3) * x^(|a| - 2) + 5 = k * x + m) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l731_73111


namespace NUMINAMATH_CALUDE_c_class_size_l731_73113

/-- The number of students in each class -/
structure ClassSize where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The conditions of the problem -/
def problem_conditions (s : ClassSize) : Prop :=
  s.A = 44 ∧ s.B = s.A + 2 ∧ s.C = s.B - 1

/-- The theorem to prove -/
theorem c_class_size (s : ClassSize) (h : problem_conditions s) : s.C = 45 := by
  sorry


end NUMINAMATH_CALUDE_c_class_size_l731_73113


namespace NUMINAMATH_CALUDE_pascals_triangle_divisibility_l731_73117

theorem pascals_triangle_divisibility (p : ℕ) (hp : Prime p) (n : ℕ) :
  (∀ k : ℕ, k ≤ n → ¬(p ∣ Nat.choose n k)) ↔
  ∃ (s q : ℕ), s ≥ 0 ∧ 0 < q ∧ q < p ∧ n = p^s * q - 1 :=
by sorry

end NUMINAMATH_CALUDE_pascals_triangle_divisibility_l731_73117


namespace NUMINAMATH_CALUDE_exists_multiple_factorization_l731_73126

/-- The set Vn for a given n > 2 -/
def Vn (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ k : ℕ, m = 1 + k * n}

/-- A number is indecomposable in Vn if it cannot be expressed as a product of two elements in Vn -/
def Indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ Vn n ∧ ¬∃ p q : ℕ, p ∈ Vn n ∧ q ∈ Vn n ∧ p * q = m

/-- The main theorem statement -/
theorem exists_multiple_factorization (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ Vn n ∧
    ∃ (f₁ f₂ : List ℕ),
      f₁ ≠ f₂ ∧
      (∀ x ∈ f₁, Indecomposable n x) ∧
      (∀ x ∈ f₂, Indecomposable n x) ∧
      r = f₁.prod ∧
      r = f₂.prod :=
sorry

end NUMINAMATH_CALUDE_exists_multiple_factorization_l731_73126


namespace NUMINAMATH_CALUDE_relationship_abc_l731_73163

theorem relationship_abc : 
  let a := (3/4)^(2/3)
  let b := (2/3)^(3/4)
  let c := Real.log (4/3) / Real.log (2/3)
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l731_73163


namespace NUMINAMATH_CALUDE_population_reaches_target_in_2095_l731_73160

/-- The initial population of the island -/
def initial_population : ℕ := 450

/-- The year when the population count starts -/
def initial_year : ℕ := 2020

/-- The number of years it takes for the population to triple -/
def tripling_period : ℕ := 25

/-- The target population we want to reach or exceed -/
def target_population : ℕ := 10800

/-- Function to calculate the population after a given number of periods -/
def population_after_periods (periods : ℕ) : ℕ :=
  initial_population * (3 ^ periods)

/-- Function to calculate the year after a given number of periods -/
def year_after_periods (periods : ℕ) : ℕ :=
  initial_year + (periods * tripling_period)

/-- Theorem stating that 2095 is the closest year to when the population reaches or exceeds the target -/
theorem population_reaches_target_in_2095 :
  ∃ (n : ℕ), 
    (population_after_periods n ≥ target_population) ∧
    (population_after_periods (n - 1) < target_population) ∧
    (year_after_periods n = 2095) :=
  sorry

end NUMINAMATH_CALUDE_population_reaches_target_in_2095_l731_73160


namespace NUMINAMATH_CALUDE_least_four_digit_13_heavy_l731_73115

theorem least_four_digit_13_heavy : ∀ n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧ n % 13 > 8 → n ≥ 1004 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_13_heavy_l731_73115


namespace NUMINAMATH_CALUDE_completing_square_transformation_l731_73145

theorem completing_square_transformation (x : ℝ) :
  x^2 - 2*x - 7 = 0 ↔ (x - 1)^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l731_73145


namespace NUMINAMATH_CALUDE_era_burger_left_l731_73159

/-- Represents the problem of Era's burger distribution --/
def era_burger_problem (total_burgers : ℕ) (num_friends : ℕ) (slices_per_burger : ℕ) 
  (friend1_slices : ℕ) (friend2_slices : ℕ) (friend3_slices : ℕ) (friend4_slices : ℕ) : Prop :=
  total_burgers = 5 ∧
  num_friends = 4 ∧
  slices_per_burger = 2 ∧
  friend1_slices = 1 ∧
  friend2_slices = 2 ∧
  friend3_slices = 3 ∧
  friend4_slices = 3

/-- Theorem stating that Era has 1 slice of burger left --/
theorem era_burger_left (total_burgers num_friends slices_per_burger 
  friend1_slices friend2_slices friend3_slices friend4_slices : ℕ) :
  era_burger_problem total_burgers num_friends slices_per_burger 
    friend1_slices friend2_slices friend3_slices friend4_slices →
  total_burgers * slices_per_burger - (friend1_slices + friend2_slices + friend3_slices + friend4_slices) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_era_burger_left_l731_73159


namespace NUMINAMATH_CALUDE_tan_alpha_value_l731_73177

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 3) : Real.tan α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l731_73177


namespace NUMINAMATH_CALUDE_saree_price_calculation_l731_73197

theorem saree_price_calculation (P : ℝ) : 
  P * (1 - 0.20) * (1 - 0.05) = 152 → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l731_73197


namespace NUMINAMATH_CALUDE_parabola_point_distances_l731_73103

theorem parabola_point_distances (a c : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  a > 0 →
  y₂ = -9 * a + c →
  y₁ = a * x₁^2 - 6 * a * x₁ + c →
  y₂ = a * x₂^2 - 6 * a * x₂ + c →
  y₃ = a * x₃^2 - 6 * a * x₃ + c →
  y₁ > y₃ →
  y₃ ≥ y₂ →
  |x₁ - x₂| > |x₂ - x₃| :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_distances_l731_73103


namespace NUMINAMATH_CALUDE_abs_two_minus_sqrt_five_l731_73109

theorem abs_two_minus_sqrt_five : |2 - Real.sqrt 5| = Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_minus_sqrt_five_l731_73109


namespace NUMINAMATH_CALUDE_max_value_of_f_min_value_of_expression_equality_condition_l731_73129

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x|

-- Theorem for the maximum value of f
theorem max_value_of_f : ∃ (m : ℝ), ∀ (x : ℝ), f x ≤ m ∧ ∃ (y : ℝ), f y = m :=
sorry

-- Theorem for the minimum value of the expression
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a^2 / (b + 1)) + (b^2 / (a + 1)) ≥ 1/3 :=
sorry

-- Theorem for the equality condition
theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a^2 / (b + 1)) + (b^2 / (a + 1)) = 1/3 ↔ a = 1/2 ∧ b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_min_value_of_expression_equality_condition_l731_73129


namespace NUMINAMATH_CALUDE_fox_initial_money_l731_73173

/-- The amount of money the fox has after n bridge crossings -/
def fox_money (a₀ : ℕ) : ℕ → ℕ
  | 0 => a₀
  | n + 1 => 2 * fox_money a₀ n - 2^2019

theorem fox_initial_money :
  ∀ a₀ : ℕ, fox_money a₀ 2019 = 0 → a₀ = 2^2019 - 1 := by
  sorry

#check fox_initial_money

end NUMINAMATH_CALUDE_fox_initial_money_l731_73173


namespace NUMINAMATH_CALUDE_gcd_n4_plus_27_and_n_plus_3_l731_73151

theorem gcd_n4_plus_27_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^4 + 27) (n + 3) = if n % 3 = 0 then 3 else 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n4_plus_27_and_n_plus_3_l731_73151


namespace NUMINAMATH_CALUDE_range_of_a_l731_73137

/-- The set of x satisfying p -/
def set_p (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

/-- The set of x satisfying q -/
def set_q : Set ℝ := {x | x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0}

/-- The theorem stating the range of values for a -/
theorem range_of_a (a : ℝ) (h1 : a < 0) 
  (h2 : set_p a ⊆ set_q)
  (h3 : (Set.univ \ set_p a) ⊂ (Set.univ \ set_q)) :
  -4 ≤ a ∧ a < 0 ∨ a ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l731_73137


namespace NUMINAMATH_CALUDE_sallys_payment_l731_73122

/-- Proves that the amount Sally paid with is $20, given that she bought 3 frames at $3 each and received $11 in change. -/
theorem sallys_payment (num_frames : ℕ) (frame_cost : ℕ) (change : ℕ) : 
  num_frames = 3 → frame_cost = 3 → change = 11 → 
  num_frames * frame_cost + change = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_sallys_payment_l731_73122


namespace NUMINAMATH_CALUDE_cubic_sum_ratio_l731_73199

theorem cubic_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 15)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 18 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_ratio_l731_73199


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l731_73196

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_condition : a 1 * (a 8)^3 * a 15 = 243) :
  (a 9)^3 / a 11 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l731_73196


namespace NUMINAMATH_CALUDE_truck_gas_ratio_l731_73167

/-- Proves the ratio of gas in a truck's tank to its total capacity before filling -/
theorem truck_gas_ratio (truck_capacity car_capacity added_gas : ℚ) 
  (h1 : truck_capacity = 20)
  (h2 : car_capacity = 12)
  (h3 : added_gas = 18)
  (h4 : (1/3) * car_capacity + added_gas = truck_capacity + car_capacity) :
  (truck_capacity - ((1/3) * car_capacity + added_gas - car_capacity)) / truck_capacity = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_truck_gas_ratio_l731_73167


namespace NUMINAMATH_CALUDE_sqrt_86400_simplified_l731_73142

theorem sqrt_86400_simplified : Real.sqrt 86400 = 120 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_86400_simplified_l731_73142


namespace NUMINAMATH_CALUDE_pigs_joined_l731_73182

theorem pigs_joined (initial_pigs final_pigs : ℕ) 
  (h1 : initial_pigs = 64)
  (h2 : final_pigs = 86) :
  final_pigs - initial_pigs = 22 :=
by sorry

end NUMINAMATH_CALUDE_pigs_joined_l731_73182


namespace NUMINAMATH_CALUDE_job_completion_time_l731_73130

/-- The time taken for three workers to complete a job together, given their individual work rates -/
theorem job_completion_time 
  (rate_a rate_b rate_c : ℚ) 
  (h_a : rate_a = 1 / 8) 
  (h_b : rate_b = 1 / 16) 
  (h_c : rate_c = 1 / 16) : 
  1 / (rate_a + rate_b + rate_c) = 4 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l731_73130


namespace NUMINAMATH_CALUDE_triangle_problem_l731_73140

open Real

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isAcute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < π/2 ∧
  0 < t.B ∧ t.B < π/2 ∧
  0 < t.C ∧ t.C < π/2

def satisfiesSineLaw (t : Triangle) : Prop :=
  t.a / sin t.A = t.b / sin t.B ∧
  t.b / sin t.B = t.c / sin t.C

def satisfiesGivenCondition (t : Triangle) : Prop :=
  2 * t.a * sin t.B = t.b

-- The main theorem
theorem triangle_problem (t : Triangle)
  (h_acute : isAcute t)
  (h_sine_law : satisfiesSineLaw t)
  (h_condition : satisfiesGivenCondition t) :
  t.A = π/6 ∧
  (t.a = 6 ∧ t.b + t.c = 8 →
    1/2 * t.b * t.c * sin t.A = 14 - 7 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l731_73140


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l731_73198

def reading_speed : ℝ := 120
def pages_per_book : ℕ := 360
def available_time : ℝ := 10

def books_read : ℕ := 3

theorem robert_reading_capacity : 
  (reading_speed * available_time) / pages_per_book ≥ books_read ∧ 
  (reading_speed * available_time) / pages_per_book < books_read + 1 :=
sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l731_73198


namespace NUMINAMATH_CALUDE_complex_norm_squared_l731_73106

theorem complex_norm_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 3 - 5*I) : 
  Complex.abs z^2 = 17/3 := by
sorry

end NUMINAMATH_CALUDE_complex_norm_squared_l731_73106


namespace NUMINAMATH_CALUDE_dinner_bill_calculation_l731_73192

theorem dinner_bill_calculation 
  (appetizer_cost : ℝ) 
  (entree_cost : ℝ) 
  (dessert_cost : ℝ) 
  (tip_percentage : ℝ) 
  (h1 : appetizer_cost = 9)
  (h2 : entree_cost = 20)
  (h3 : dessert_cost = 11)
  (h4 : tip_percentage = 0.3) :
  appetizer_cost + 2 * entree_cost + dessert_cost + 
  (appetizer_cost + 2 * entree_cost + dessert_cost) * tip_percentage = 78 :=
by sorry

end NUMINAMATH_CALUDE_dinner_bill_calculation_l731_73192


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l731_73154

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a (n + 1) > a n) →
  a 5 ^ 2 = a 10 →
  (∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) →
  ∃ c, ∀ n, a n = c * 2^n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l731_73154


namespace NUMINAMATH_CALUDE_equation_pattern_l731_73104

theorem equation_pattern (n : ℕ) : 2*n * (2*n + 2) + 1 = (2*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_pattern_l731_73104


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_l731_73176

theorem consecutive_squares_sum (x : ℕ) :
  x^2 + (x+1)^2 + (x+2)^2 = 2030 → x + 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_l731_73176


namespace NUMINAMATH_CALUDE_cos_is_even_l731_73123

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- State the theorem
theorem cos_is_even : IsEven Real.cos := by
  sorry

end NUMINAMATH_CALUDE_cos_is_even_l731_73123


namespace NUMINAMATH_CALUDE_tax_savings_proof_l731_73162

def initial_tax_rate : ℝ := 0.46
def new_tax_rate : ℝ := 0.32
def annual_income : ℝ := 36000

def differential_savings : ℝ :=
  annual_income * initial_tax_rate - annual_income * new_tax_rate

theorem tax_savings_proof :
  differential_savings = 5040 := by
  sorry

end NUMINAMATH_CALUDE_tax_savings_proof_l731_73162


namespace NUMINAMATH_CALUDE_x_plus_y_equals_five_l731_73181

theorem x_plus_y_equals_five (x y : ℝ) (h : (x + 1)^2 + |y - 6| = 0) : x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_five_l731_73181


namespace NUMINAMATH_CALUDE_jacket_purchase_price_l731_73120

/-- Proves that the purchase price of a jacket is $56 given the specified conditions --/
theorem jacket_purchase_price :
  ∀ (purchase_price selling_price sale_price : ℝ),
  selling_price = purchase_price + 0.3 * selling_price →
  sale_price = 0.8 * selling_price →
  sale_price - purchase_price = 8 →
  purchase_price = 56 := by
sorry

end NUMINAMATH_CALUDE_jacket_purchase_price_l731_73120


namespace NUMINAMATH_CALUDE_farm_sheep_ratio_l731_73156

/-- Proves that the ratio of sheep sold to total sheep is 2:3 given the farm conditions --/
theorem farm_sheep_ratio :
  ∀ (goats sheep sold_sheep : ℕ) (sale_amount : ℚ),
    goats + sheep = 360 →
    goats * 7 = sheep * 5 →
    sale_amount = 7200 →
    sale_amount = (goats / 2) * 40 + sold_sheep * 30 →
    sold_sheep / sheep = 2 / 3 :=
by sorry


end NUMINAMATH_CALUDE_farm_sheep_ratio_l731_73156


namespace NUMINAMATH_CALUDE_tom_bonus_percentage_l731_73189

/-- Calculates the percentage of bonus points per customer served -/
def bonus_percentage (customers_per_hour : ℕ) (hours_worked : ℕ) (total_bonus_points : ℕ) : ℚ :=
  (total_bonus_points : ℚ) / ((customers_per_hour * hours_worked) : ℚ) * 100

/-- Proves that the bonus percentage for Tom is 20% -/
theorem tom_bonus_percentage :
  bonus_percentage 10 8 16 = 20 := by
  sorry

#eval bonus_percentage 10 8 16

end NUMINAMATH_CALUDE_tom_bonus_percentage_l731_73189


namespace NUMINAMATH_CALUDE_round_trip_speed_l731_73168

/-- Given a round trip where:
    - The outbound journey is at speed p km/h
    - The return journey is at 3 km/h
    - The average speed is (24/q) km/h
    - p = 4
    Then q = 7 -/
theorem round_trip_speed (p q : ℝ) (hp : p = 4) : 
  (2 / ((1/p) + (1/3)) = 24/q) → q = 7 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_speed_l731_73168


namespace NUMINAMATH_CALUDE_divisible_by_six_l731_73108

theorem divisible_by_six (m : ℕ) : ∃ k : ℤ, (m : ℤ)^3 + 11*(m : ℤ) = 6*k := by sorry

end NUMINAMATH_CALUDE_divisible_by_six_l731_73108


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l731_73172

theorem simplify_sqrt_expression :
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 242 / Real.sqrt 121) = (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l731_73172


namespace NUMINAMATH_CALUDE_may_total_scarves_l731_73132

/-- The number of scarves that can be knitted from one yarn -/
def scarves_per_yarn : ℕ := 3

/-- The number of red yarns May bought -/
def red_yarns : ℕ := 2

/-- The number of blue yarns May bought -/
def blue_yarns : ℕ := 6

/-- The number of yellow yarns May bought -/
def yellow_yarns : ℕ := 4

/-- The total number of scarves May can make -/
def total_scarves : ℕ := scarves_per_yarn * (red_yarns + blue_yarns + yellow_yarns)

theorem may_total_scarves : total_scarves = 36 := by
  sorry

end NUMINAMATH_CALUDE_may_total_scarves_l731_73132


namespace NUMINAMATH_CALUDE_solution_sum_l731_73121

/-- Given a system of equations, prove that the sum of its solutions is 2020 -/
theorem solution_sum (x₀ y₀ : ℝ) 
  (eq1 : x₀^3 - 2023*x₀ = 2023*y₀ - y₀^3 - 2020)
  (eq2 : x₀^2 - x₀*y₀ + y₀^2 = 2022) :
  x₀ + y₀ = 2020 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l731_73121


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l731_73136

-- Define the complex number z
def z : ℂ := -1 + 3 * Complex.I

-- Theorem stating that z is in the second quadrant
theorem z_in_second_quadrant : 
  z.re < 0 ∧ z.im > 0 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l731_73136


namespace NUMINAMATH_CALUDE_inequality_solution_l731_73110

theorem inequality_solution (x : ℝ) : (x - 2) / (x + 5) ≤ 1 / 2 ↔ -5 < x ∧ x ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l731_73110


namespace NUMINAMATH_CALUDE_rain_probability_both_days_l731_73191

theorem rain_probability_both_days 
  (prob_saturday : ℝ) 
  (prob_sunday : ℝ) 
  (prob_sunday_given_saturday : ℝ) 
  (h1 : prob_saturday = 0.4)
  (h2 : prob_sunday = 0.3)
  (h3 : prob_sunday_given_saturday = 0.5) :
  prob_saturday * prob_sunday_given_saturday = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_both_days_l731_73191


namespace NUMINAMATH_CALUDE_cost_splitting_difference_l731_73127

def bob_paid : ℚ := 130
def alice_paid : ℚ := 110
def jessica_paid : ℚ := 160

def total_paid : ℚ := bob_paid + alice_paid + jessica_paid
def share_per_person : ℚ := total_paid / 3

def bob_owes : ℚ := share_per_person - bob_paid
def alice_owes : ℚ := share_per_person - alice_paid
def jessica_receives : ℚ := jessica_paid - share_per_person

theorem cost_splitting_difference :
  bob_owes - alice_owes = -20 := by sorry

end NUMINAMATH_CALUDE_cost_splitting_difference_l731_73127


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l731_73131

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l731_73131


namespace NUMINAMATH_CALUDE_probability_theorem_l731_73144

def total_balls : ℕ := 22
def red_balls : ℕ := 5
def blue_balls : ℕ := 6
def green_balls : ℕ := 7
def yellow_balls : ℕ := 4
def balls_picked : ℕ := 3

def probability_at_least_two_red_not_blue : ℚ :=
  (Nat.choose red_balls 2 * (green_balls + yellow_balls) +
   Nat.choose red_balls 3) /
  Nat.choose total_balls balls_picked

theorem probability_theorem :
  probability_at_least_two_red_not_blue = 12 / 154 :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l731_73144


namespace NUMINAMATH_CALUDE_p_recurrence_l731_73178

/-- Probability of having a group of length k or more in n tosses of a symmetric coin -/
def p (n k : ℕ) : ℝ :=
  sorry

/-- The recurrence relation for p(n,k) -/
theorem p_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end NUMINAMATH_CALUDE_p_recurrence_l731_73178


namespace NUMINAMATH_CALUDE_sqrt_32_div_sqrt_2_eq_4_l731_73135

theorem sqrt_32_div_sqrt_2_eq_4 : Real.sqrt 32 / Real.sqrt 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_32_div_sqrt_2_eq_4_l731_73135


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l731_73139

/-- Given a right triangle with sides 5, 12, and 13, where the vertices are centers of
    three mutually externally tangent circles, the sum of the areas of these circles is 113π. -/
theorem sum_of_circle_areas (a b c : ℝ) (r s t : ℝ) : 
  a = 5 → b = 12 → c = 13 →
  a^2 + b^2 = c^2 →
  r + s = b →
  s + t = a →
  r + t = c →
  π * (r^2 + s^2 + t^2) = 113 * π := by
sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l731_73139


namespace NUMINAMATH_CALUDE_vector_parallel_coordinates_l731_73193

/-- Given two vectors a and b in ℝ², prove that if |a| = 2√5, b = (1,2), and a is parallel to b,
    then a = (2,4) or a = (-2,-4) -/
theorem vector_parallel_coordinates (a b : ℝ × ℝ) :
  (norm a = 2 * Real.sqrt 5) →
  (b = (1, 2)) →
  (∃ (k : ℝ), a = k • b) →
  (a = (2, 4) ∨ a = (-2, -4)) :=
sorry

end NUMINAMATH_CALUDE_vector_parallel_coordinates_l731_73193


namespace NUMINAMATH_CALUDE_distance_scientific_notation_equivalence_l731_73114

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The distance between two mountain peaks in meters -/
def distance : ℝ := 14000000

/-- The scientific notation representation of the distance -/
def distanceScientific : ScientificNotation := {
  coefficient := 1.4
  exponent := 7
  h1 := by sorry
}

theorem distance_scientific_notation_equivalence :
  distance = distanceScientific.coefficient * (10 : ℝ) ^ distanceScientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_distance_scientific_notation_equivalence_l731_73114


namespace NUMINAMATH_CALUDE_bulls_win_probability_l731_73138

/-- The probability of the Knicks winning a single game -/
def p_knicks : ℚ := 3/5

/-- The probability of the Bulls winning a single game -/
def p_bulls : ℚ := 1 - p_knicks

/-- The number of ways to choose 3 games out of 6 -/
def ways_to_choose : ℕ := 20

/-- The probability of the Bulls winning the playoff series in exactly 7 games -/
def prob_bulls_win_in_seven : ℚ :=
  ways_to_choose * p_bulls^3 * p_knicks^3 * p_bulls

theorem bulls_win_probability :
  prob_bulls_win_in_seven = 864/15625 := by sorry

end NUMINAMATH_CALUDE_bulls_win_probability_l731_73138


namespace NUMINAMATH_CALUDE_infinitely_many_powers_of_two_in_floor_sqrt_two_n_l731_73153

theorem infinitely_many_powers_of_two_in_floor_sqrt_two_n : 
  ∀ m : ℕ, ∃ n k : ℕ, n > m ∧ k > 0 ∧ ⌊Real.sqrt 2 * n⌋ = 2^k :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_powers_of_two_in_floor_sqrt_two_n_l731_73153


namespace NUMINAMATH_CALUDE_divisors_of_36_count_divisors_of_36_l731_73183

theorem divisors_of_36 : Finset Int → Prop :=
  fun s => ∀ n : Int, n ∈ s ↔ 36 % n = 0

theorem count_divisors_of_36 : 
  ∃ s : Finset Int, divisors_of_36 s ∧ s.card = 18 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_36_count_divisors_of_36_l731_73183


namespace NUMINAMATH_CALUDE_smallest_square_multiplier_l731_73146

def y : ℕ := 2^10 * 3^15 * 4^20 * 5^25 * 6^30 * 7^35 * 8^40 * 9^45

theorem smallest_square_multiplier (n : ℕ) : 
  (∃ m : ℕ, n * y = m^2) ∧ 
  (∀ k : ℕ, k < n → ¬∃ m : ℕ, k * y = m^2) ↔ 
  n = 105 :=
sorry

end NUMINAMATH_CALUDE_smallest_square_multiplier_l731_73146


namespace NUMINAMATH_CALUDE_greendale_points_greendale_points_equals_130_l731_73148

/-- Calculates the total points for Greendale High School in a basketball tournament --/
theorem greendale_points (roosevelt_first_game : ℕ) (bonus : ℕ) (difference : ℕ) : ℕ :=
  let roosevelt_second_game := roosevelt_first_game / 2
  let roosevelt_third_game := roosevelt_second_game * 3
  let roosevelt_total := roosevelt_first_game + roosevelt_second_game + roosevelt_third_game + bonus
  roosevelt_total - difference

/-- Proves that Greendale High School's total points equal 130 --/
theorem greendale_points_equals_130 : greendale_points 30 50 10 = 130 := by
  sorry

end NUMINAMATH_CALUDE_greendale_points_greendale_points_equals_130_l731_73148


namespace NUMINAMATH_CALUDE_train_speed_increase_time_l731_73147

/-- The speed equation for a subway train -/
def speed_equation (s : ℝ) : ℝ := s^2 + 2*s

/-- Theorem: The time when the train's speed increases by 39 km/h from its speed at 4 seconds is 7 seconds -/
theorem train_speed_increase_time : 
  ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 7 ∧ 
  speed_equation s = speed_equation 4 + 39 ∧
  s = 7 := by
  sorry

#check train_speed_increase_time

end NUMINAMATH_CALUDE_train_speed_increase_time_l731_73147


namespace NUMINAMATH_CALUDE_cube_dot_path_length_l731_73124

theorem cube_dot_path_length (cube_edge : ℝ) (dot_path : ℝ) : 
  cube_edge = 2 →
  dot_path = Real.sqrt 5 * Real.pi →
  ∃ (rotation_radius : ℝ),
    rotation_radius = Real.sqrt (1^2 + 2^2) ∧
    dot_path = 4 * (1/4 * 2 * Real.pi * rotation_radius) :=
by sorry

end NUMINAMATH_CALUDE_cube_dot_path_length_l731_73124


namespace NUMINAMATH_CALUDE_largest_non_expressible_l731_73165

/-- A positive integer is composite if it has a proper divisor greater than 1. -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- A function that checks if a number can be expressed as 42k + c, 
    where k is a positive integer and c is a positive composite integer. -/
def CanBeExpressed (n : ℕ) : Prop :=
  ∃ (k c : ℕ), k > 0 ∧ IsComposite c ∧ n = 42 * k + c

/-- The theorem stating that 215 is the largest positive integer that cannot be expressed
    as the sum of a positive integral multiple of 42 and a positive composite integer. -/
theorem largest_non_expressible : 
  (∀ n : ℕ, n > 215 → CanBeExpressed n) ∧ 
  (¬ CanBeExpressed 215) := by
  sorry

#check largest_non_expressible

end NUMINAMATH_CALUDE_largest_non_expressible_l731_73165


namespace NUMINAMATH_CALUDE_point_movement_theorem_l731_73128

/-- The initial position of a point on a number line that ends at the origin after moving right 7 units and then left 4 units -/
def initial_position : ℤ := -3

/-- A point's movement on a number line -/
def point_movement (start : ℤ) : ℤ := start + 7 - 4

theorem point_movement_theorem :
  point_movement initial_position = 0 :=
by sorry

end NUMINAMATH_CALUDE_point_movement_theorem_l731_73128


namespace NUMINAMATH_CALUDE_tangent_circle_rectangle_area_l731_73157

/-- A rectangle with a tangent circle passing through one vertex -/
structure TangentCircleRectangle where
  /-- Length of the rectangle -/
  l : ℝ
  /-- Width of the rectangle -/
  w : ℝ
  /-- Radius of the circle -/
  r : ℝ
  /-- The circle is tangent to two adjacent sides of the rectangle -/
  tangent : l = 2 * r
  /-- The circle passes through the opposite corner -/
  passes_through : w = r

/-- The area of a rectangle with a tangent circle passing through one vertex is 2r² -/
theorem tangent_circle_rectangle_area (rect : TangentCircleRectangle) : 
  rect.l * rect.w = 2 * rect.r^2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_rectangle_area_l731_73157


namespace NUMINAMATH_CALUDE_quadratic_roots_k_less_than_9_l731_73187

theorem quadratic_roots_k_less_than_9 (k : ℝ) (h : k < 9) :
  (∃ x : ℝ, (k - 5) * x^2 - 2 * (k - 3) * x + k = 0) ∧
  ((∃! x : ℝ, (k - 5) * x^2 - 2 * (k - 3) * x + k = 0) ∨
   (∃ x y : ℝ, x ≠ y ∧ (k - 5) * x^2 - 2 * (k - 3) * x + k = 0 ∧
                      (k - 5) * y^2 - 2 * (k - 3) * y + k = 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_k_less_than_9_l731_73187


namespace NUMINAMATH_CALUDE_pencil_black_fraction_l731_73100

theorem pencil_black_fraction :
  ∀ (total_length blue_length white_length black_length : ℝ),
    total_length = 8 →
    blue_length = 3.5 →
    white_length = (total_length - blue_length) / 2 →
    black_length = total_length - blue_length - white_length →
    black_length / total_length = 9 / 32 := by
  sorry

end NUMINAMATH_CALUDE_pencil_black_fraction_l731_73100


namespace NUMINAMATH_CALUDE_prime_quadratic_integer_roots_l731_73175

theorem prime_quadratic_integer_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x ^ 2 - p * x - 156 * p = 0 ∧ y ^ 2 - p * y - 156 * p = 0) → 
  p = 13 := by
sorry

end NUMINAMATH_CALUDE_prime_quadratic_integer_roots_l731_73175


namespace NUMINAMATH_CALUDE_rectangle_unique_symmetric_shape_l731_73188

-- Define the shapes
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | Rectangle
  | RegularPentagon

-- Define axisymmetry and central symmetry
def isAxisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.Parallelogram => false
  | Shape.Rectangle => true
  | Shape.RegularPentagon => true

def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => false
  | Shape.Parallelogram => true
  | Shape.Rectangle => true
  | Shape.RegularPentagon => false

-- Theorem statement
theorem rectangle_unique_symmetric_shape :
  ∀ s : Shape, isAxisymmetric s ∧ isCentrallySymmetric s ↔ s = Shape.Rectangle :=
by sorry

end NUMINAMATH_CALUDE_rectangle_unique_symmetric_shape_l731_73188


namespace NUMINAMATH_CALUDE_sticker_distribution_l731_73171

/-- The number of ways to distribute indistinguishable objects among distinct containers -/
def distribute (objects : ℕ) (containers : ℕ) : ℕ :=
  Nat.choose (objects + containers - 1) (containers - 1)

/-- Theorem: Distributing 10 indistinguishable stickers among 5 distinct sheets of paper -/
theorem sticker_distribution : distribute 10 5 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l731_73171


namespace NUMINAMATH_CALUDE_percentage_of_450_to_325x_l731_73152

theorem percentage_of_450_to_325x (x : ℝ) (h : x ≠ 0) :
  (450 : ℝ) / (325 * x) * 100 = 138.46153846 / x :=
sorry

end NUMINAMATH_CALUDE_percentage_of_450_to_325x_l731_73152


namespace NUMINAMATH_CALUDE_gumball_theorem_l731_73133

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : Nat)
  (white : Nat)
  (blue : Nat)
  (green : Nat)

/-- The least number of gumballs needed to guarantee four of the same color -/
def leastGumballsNeeded (machine : GumballMachine) : Nat :=
  13

/-- Theorem stating that for the given gumball machine, 
    the least number of gumballs needed is 13 -/
theorem gumball_theorem (machine : GumballMachine) 
  (h1 : machine.red = 10)
  (h2 : machine.white = 9)
  (h3 : machine.blue = 8)
  (h4 : machine.green = 7) :
  leastGumballsNeeded machine = 13 := by
  sorry

#check gumball_theorem

end NUMINAMATH_CALUDE_gumball_theorem_l731_73133
