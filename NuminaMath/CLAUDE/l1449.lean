import Mathlib

namespace NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l1449_144957

theorem ceiling_fraction_evaluation :
  (⌈(19 : ℚ) / 11 - ⌈(35 : ℚ) / 22⌉⌉) / (⌈(35 : ℚ) / 11 + ⌈(11 * 22 : ℚ) / 35⌉⌉) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l1449_144957


namespace NUMINAMATH_CALUDE_f_log_one_third_36_l1449_144916

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x % 3 ∧ x % 3 < 1 then 3^(x % 3) - 1
  else if 1 ≤ x % 3 ∧ x % 3 < 2 then -(3^(2 - (x % 3)) - 1)
  else -(3^((x % 3) - 2) - 1)

-- State the theorem
theorem f_log_one_third_36 (h1 : ∀ x, f (-x) = -f x) 
                            (h2 : ∀ x, f (x + 3) = f x) 
                            (h3 : ∀ x, 0 ≤ x → x < 1 → f x = 3^x - 1) :
  f (Real.log 36 / Real.log (1/3)) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_f_log_one_third_36_l1449_144916


namespace NUMINAMATH_CALUDE_sum_distinct_prime_divisors_of_1260_l1449_144922

/-- The sum of the distinct prime integer divisors of 1260 is 17. -/
theorem sum_distinct_prime_divisors_of_1260 : 
  (Finset.sum (Finset.filter Nat.Prime (Nat.divisors 1260)) id) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_distinct_prime_divisors_of_1260_l1449_144922


namespace NUMINAMATH_CALUDE_school_students_count_l1449_144936

theorem school_students_count (football cricket both neither : ℕ) 
  (h1 : football = 325)
  (h2 : cricket = 175)
  (h3 : neither = 50)
  (h4 : both = 90) :
  football + cricket - both + neither = 460 :=
by sorry

end NUMINAMATH_CALUDE_school_students_count_l1449_144936


namespace NUMINAMATH_CALUDE_duck_price_is_correct_l1449_144909

/-- The price of a duck given the conditions of the problem -/
def duck_price : ℝ :=
  let chicken_price : ℝ := 8
  let num_chickens : ℕ := 5
  let num_ducks : ℕ := 2
  let additional_earnings : ℝ := 60
  10

theorem duck_price_is_correct :
  let chicken_price : ℝ := 8
  let num_chickens : ℕ := 5
  let num_ducks : ℕ := 2
  let additional_earnings : ℝ := 60
  let total_earnings := chicken_price * num_chickens + duck_price * num_ducks
  let wheelbarrow_cost := total_earnings / 2
  wheelbarrow_cost * 2 = additional_earnings ∧ duck_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_duck_price_is_correct_l1449_144909


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1449_144993

theorem quadratic_factorization (c d : ℤ) :
  (∀ x : ℝ, (5*x + c) * (5*x + d) = 25*x^2 - 135*x - 150) →
  c + 2*d = -59 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1449_144993


namespace NUMINAMATH_CALUDE_negation_of_existence_l1449_144984

theorem negation_of_existence (n : ℝ) :
  (¬ ∃ a : ℝ, a ≥ -1 ∧ Real.log (Real.exp n + 1) > 1/2) ↔
  (∀ a : ℝ, a ≥ -1 → Real.log (Real.exp n + 1) ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l1449_144984


namespace NUMINAMATH_CALUDE_mary_picked_14_oranges_l1449_144950

/-- The number of oranges picked by Jason -/
def jason_oranges : ℕ := 41

/-- The total number of oranges picked -/
def total_oranges : ℕ := 55

/-- The number of oranges picked by Mary -/
def mary_oranges : ℕ := total_oranges - jason_oranges

theorem mary_picked_14_oranges : mary_oranges = 14 := by
  sorry

end NUMINAMATH_CALUDE_mary_picked_14_oranges_l1449_144950


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l1449_144976

theorem arithmetic_sequence_count (a₁ aₙ d : ℤ) (n : ℕ) : 
  a₁ = 165 ∧ aₙ = 40 ∧ d = -5 ∧ aₙ = a₁ + (n - 1) * d → n = 26 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l1449_144976


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1449_144985

/-- A point in a 2D Cartesian coordinate system. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis. -/
def symmetricXAxis (p q : Point) : Prop :=
  q.x = p.x ∧ q.y = -p.y

/-- The theorem stating that if Q is symmetric to P(-3, 2) with respect to the x-axis,
    then Q has coordinates (-3, -2). -/
theorem symmetric_point_coordinates :
  let p : Point := ⟨-3, 2⟩
  let q : Point := ⟨-3, -2⟩
  symmetricXAxis p q → q = ⟨-3, -2⟩ := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1449_144985


namespace NUMINAMATH_CALUDE_S_remainder_mod_1000_l1449_144933

/-- The sum of all three-digit positive integers from 500 to 999 with all digits distinct -/
def S : ℕ := sorry

/-- Theorem stating that the remainder of S divided by 1000 is 720 -/
theorem S_remainder_mod_1000 : S % 1000 = 720 := by sorry

end NUMINAMATH_CALUDE_S_remainder_mod_1000_l1449_144933


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1449_144919

theorem fraction_multiplication : (1 : ℚ) / 3 * 4 / 7 * 9 / 13 * 2 / 5 = 72 / 1365 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1449_144919


namespace NUMINAMATH_CALUDE_range_of_m_l1449_144975

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - 2*x + 2 ≠ m) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1449_144975


namespace NUMINAMATH_CALUDE_systematic_sampling_5_from_100_correct_sequence_l1449_144900

/-- Systematic sampling function that returns the nth selected individual -/
def systematicSample (totalPopulation : ℕ) (sampleSize : ℕ) (n : ℕ) : ℕ :=
  n * (totalPopulation / sampleSize)

/-- Theorem stating that systematic sampling of 5 from 100 yields the correct sequence -/
theorem systematic_sampling_5_from_100 :
  let totalPopulation : ℕ := 100
  let sampleSize : ℕ := 5
  (systematicSample totalPopulation sampleSize 1 = 20) ∧
  (systematicSample totalPopulation sampleSize 2 = 40) ∧
  (systematicSample totalPopulation sampleSize 3 = 60) ∧
  (systematicSample totalPopulation sampleSize 4 = 80) ∧
  (systematicSample totalPopulation sampleSize 5 = 100) :=
by
  sorry

/-- Theorem stating that the correct sequence is 10, 30, 50, 70, 90 -/
theorem correct_sequence :
  let totalPopulation : ℕ := 100
  let sampleSize : ℕ := 5
  (systematicSample totalPopulation sampleSize 1 - 10 = 10) ∧
  (systematicSample totalPopulation sampleSize 2 - 10 = 30) ∧
  (systematicSample totalPopulation sampleSize 3 - 10 = 50) ∧
  (systematicSample totalPopulation sampleSize 4 - 10 = 70) ∧
  (systematicSample totalPopulation sampleSize 5 - 10 = 90) :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_5_from_100_correct_sequence_l1449_144900


namespace NUMINAMATH_CALUDE_existence_of_m_l1449_144910

def z : ℕ → ℚ
  | 0 => 3
  | n + 1 => (2 * (z n)^2 + 3 * (z n) + 6) / (z n + 8)

theorem existence_of_m :
  ∃ m : ℕ, m ∈ Finset.Icc 27 80 ∧
    z m ≤ 2 + 1 / 2^10 ∧
    ∀ k : ℕ, k > 0 ∧ k < 27 → z k > 2 + 1 / 2^10 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_m_l1449_144910


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1449_144981

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | 0 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1449_144981


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l1449_144999

theorem smallest_number_with_remainders : ∃! x : ℕ,
  (x % 5 = 2) ∧ (x % 6 = 2) ∧ (x % 7 = 3) ∧
  (∀ y : ℕ, (y % 5 = 2) ∧ (y % 6 = 2) ∧ (y % 7 = 3) → x ≤ y) ∧
  x = 122 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l1449_144999


namespace NUMINAMATH_CALUDE_special_sequence_2003_l1449_144920

/-- The sequence formed by removing multiples of 3 and 4 (except multiples of 5) from positive integers -/
def special_sequence : ℕ → ℕ := sorry

/-- The 2003rd term of the special sequence -/
def a_2003 : ℕ := special_sequence 2003

/-- Theorem stating that the 2003rd term of the special sequence is 3338 -/
theorem special_sequence_2003 : a_2003 = 3338 := by sorry

end NUMINAMATH_CALUDE_special_sequence_2003_l1449_144920


namespace NUMINAMATH_CALUDE_games_expenditure_l1449_144941

def allowance : ℚ := 48

def clothes_fraction : ℚ := 1/4
def books_fraction : ℚ := 1/3
def snacks_fraction : ℚ := 1/6

def amount_on_games : ℚ := allowance - (clothes_fraction * allowance + books_fraction * allowance + snacks_fraction * allowance)

theorem games_expenditure : amount_on_games = 12 := by
  sorry

end NUMINAMATH_CALUDE_games_expenditure_l1449_144941


namespace NUMINAMATH_CALUDE_k_increasing_on_neg_reals_l1449_144965

/-- The function k(x) = 3 - x is increasing on the interval (-∞, 0). -/
theorem k_increasing_on_neg_reals :
  StrictMonoOn (fun x : ℝ => 3 - x) (Set.Iio 0) := by
  sorry

end NUMINAMATH_CALUDE_k_increasing_on_neg_reals_l1449_144965


namespace NUMINAMATH_CALUDE_square_side_length_l1449_144998

/-- Given two identical overlapping squares where the upper square is moved 3 cm right and 5 cm down,
    resulting in a shaded area of 57 square centimeters, prove that the side length of each square is 9 cm. -/
theorem square_side_length (a : ℝ) 
  (h1 : 3 * a + 5 * (a - 3) = 57) : 
  a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1449_144998


namespace NUMINAMATH_CALUDE_pattern_3_7_verify_other_pairs_l1449_144959

/-- The pattern function that transforms two numbers according to the given rule -/
def pattern (a b : ℕ) : ℕ := (a + b) * a - a

/-- The theorem stating that the pattern applied to (3, 7) results in 27 -/
theorem pattern_3_7 : pattern 3 7 = 27 := by
  sorry

/-- Verification of other given pairs -/
theorem verify_other_pairs :
  pattern 2 3 = 8 ∧
  pattern 4 5 = 32 ∧
  pattern 5 8 = 60 ∧
  pattern 6 7 = 72 ∧
  pattern 7 8 = 98 := by
  sorry

end NUMINAMATH_CALUDE_pattern_3_7_verify_other_pairs_l1449_144959


namespace NUMINAMATH_CALUDE_area_of_non_intersecting_graphs_l1449_144954

/-- The area of the set A of points (a, b) such that the graphs of 
    f(x) = x^2 - 2ax + 1 and g(x) = 2b(a-x) do not intersect is π. -/
theorem area_of_non_intersecting_graphs (a b x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 2*a*x + 1
  let g : ℝ → ℝ := λ x => 2*b*(a-x)
  let A : Set (ℝ × ℝ) := {(a, b) | ∀ x, f x ≠ g x}
  MeasureTheory.volume A = π := by
sorry

end NUMINAMATH_CALUDE_area_of_non_intersecting_graphs_l1449_144954


namespace NUMINAMATH_CALUDE_rational_sqrt_one_minus_ab_l1449_144945

theorem rational_sqrt_one_minus_ab (a b : ℚ) 
  (h : a^3 * b + a * b^3 + 2 * a^2 * b^2 + 2 * a + 2 * b + 1 = 0) : 
  ∃ q : ℚ, q^2 = 1 - a * b := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt_one_minus_ab_l1449_144945


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l1449_144934

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithmic_equation_solution :
  ∃! x : ℝ, x > 0 ∧ 
    log_base 4 (x - 1) + log_base (Real.sqrt 4) (x^2 - 1) + log_base (1/4) (x - 1) = 2 ∧
    x = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l1449_144934


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1449_144937

/-- The expression (19a + b)^18 + (a + b)^18 + (a + 19b)^18 is a perfect square if and only if a = 0 and b = 0, where a and b are integers. -/
theorem perfect_square_condition (a b : ℤ) : 
  (∃ (k : ℤ), (19*a + b)^18 + (a + b)^18 + (a + 19*b)^18 = k^2) ↔ (a = 0 ∧ b = 0) := by
  sorry

#check perfect_square_condition

end NUMINAMATH_CALUDE_perfect_square_condition_l1449_144937


namespace NUMINAMATH_CALUDE_john_skateboard_distance_l1449_144953

/-- Represents John's journey with skateboarding distances -/
structure JourneyDistances where
  to_park_skateboard : ℕ
  to_park_walk : ℕ
  to_park_bike : ℕ
  park_jog : ℕ
  from_park_bike : ℕ
  from_park_swim : ℕ
  from_park_skateboard : ℕ

/-- Calculates the total skateboarding distance for John's journey -/
def total_skateboard_distance (j : JourneyDistances) : ℕ :=
  j.to_park_skateboard + j.from_park_skateboard

/-- Theorem: John's total skateboarding distance is 25 miles -/
theorem john_skateboard_distance (j : JourneyDistances)
  (h1 : j.to_park_skateboard = 16)
  (h2 : j.to_park_walk = 8)
  (h3 : j.to_park_bike = 6)
  (h4 : j.park_jog = 3)
  (h5 : j.from_park_bike = 5)
  (h6 : j.from_park_swim = 1)
  (h7 : j.from_park_skateboard = 9) :
  total_skateboard_distance j = 25 := by
  sorry

end NUMINAMATH_CALUDE_john_skateboard_distance_l1449_144953


namespace NUMINAMATH_CALUDE_grocery_shop_sales_l1449_144987

theorem grocery_shop_sales (sale1 sale2 sale4 sale5 sale6 average_sale : ℕ) 
  (h1 : sale1 = 6235)
  (h2 : sale2 = 6927)
  (h4 : sale4 = 7230)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 5191)
  (h_avg : average_sale = 6500) :
  ∃ sale3 : ℕ, 
    sale3 = 6855 ∧ 
    (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale :=
sorry

end NUMINAMATH_CALUDE_grocery_shop_sales_l1449_144987


namespace NUMINAMATH_CALUDE_intersection_angle_proof_l1449_144914

/-- Given two curves in polar coordinates and a ray that intersects both curves, 
    prove that the angle of the ray is π/4 when the product of the distances 
    from the origin to the intersection points is 12. -/
theorem intersection_angle_proof (θ₀ : Real) 
  (h1 : 0 < θ₀) (h2 : θ₀ < Real.pi / 2) : 
  let curve_m := fun (θ : Real) => 4 * Real.cos θ
  let curve_n := fun (ρ θ : Real) => ρ^2 * Real.sin (2 * θ) = 18
  let ray := fun (ρ : Real) => (ρ * Real.cos θ₀, ρ * Real.sin θ₀)
  let point_a := (curve_m θ₀ * Real.cos θ₀, curve_m θ₀ * Real.sin θ₀)
  let point_b := 
    (Real.sqrt (18 / Real.sin (2 * θ₀)) * Real.cos θ₀, 
     Real.sqrt (18 / Real.sin (2 * θ₀)) * Real.sin θ₀)
  (curve_m θ₀ * Real.sqrt (18 / Real.sin (2 * θ₀)) = 12) → 
  θ₀ = Real.pi / 4 := by
sorry


end NUMINAMATH_CALUDE_intersection_angle_proof_l1449_144914


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1449_144996

theorem fractional_equation_solution :
  ∃ (x : ℝ), (2 / (x - 2) - (2 * x) / (2 - x) = 1) ∧ (x - 2 ≠ 0) ∧ (2 - x ≠ 0) ∧ (x = -4) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1449_144996


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1449_144980

theorem quadratic_equation_solution :
  let f (x : ℝ) := x^2 - (2/3)*x - 1
  ∃ (x₁ x₂ : ℝ), 
    f x₁ = 0 ∧ 
    f x₂ = 0 ∧ 
    x₁ = (Real.sqrt 10)/3 + 1/3 ∧ 
    x₂ = -(Real.sqrt 10)/3 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1449_144980


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1449_144974

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 4) → x ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1449_144974


namespace NUMINAMATH_CALUDE_remarkable_number_l1449_144961

theorem remarkable_number : ∃ (x : ℝ), 
  x > 0 ∧ 
  (x - ⌊x⌋) * ⌊x⌋ = (x - ⌊x⌋)^2 ∧ 
  x = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_remarkable_number_l1449_144961


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1449_144946

theorem no_solution_for_equation :
  ¬ ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (2 / a + 2 / b = 1 / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1449_144946


namespace NUMINAMATH_CALUDE_regular_polygon_perimeters_l1449_144995

/-- Regular polygon perimeters for a unit circle -/
noncomputable def RegularPolygonPerimeters (n : ℕ) : ℝ × ℝ :=
  sorry

/-- Circumscribed polygon perimeter -/
noncomputable def P (n : ℕ) : ℝ := (RegularPolygonPerimeters n).1

/-- Inscribed polygon perimeter -/
noncomputable def p (n : ℕ) : ℝ := (RegularPolygonPerimeters n).2

theorem regular_polygon_perimeters :
  (P 4 = 8 ∧ p 4 = 4 * Real.sqrt 2 ∧ P 6 = 4 * Real.sqrt 3 ∧ p 6 = 6) ∧
  (∀ n ≥ 3, P (2 * n) = (2 * P n * p n) / (P n + p n) ∧
            p (2 * n) = Real.sqrt (p n * P (2 * n))) ∧
  (3^10 / 71 < Real.pi ∧ Real.pi < 22 / 7) :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeters_l1449_144995


namespace NUMINAMATH_CALUDE_used_car_clients_l1449_144972

theorem used_car_clients (total_cars : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) :
  total_cars = 12 →
  cars_per_client = 4 →
  selections_per_car = 3 →
  (total_cars * selections_per_car) / cars_per_client = 9 :=
by sorry

end NUMINAMATH_CALUDE_used_car_clients_l1449_144972


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l1449_144973

def B : Set ℕ := {x | ∃ n : ℕ, x = 4*n + 6 ∧ n > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 1 ∧ (∀ x ∈ B, d ∣ x) ∧ 
  (∀ k : ℕ, k > 1 → (∀ x ∈ B, k ∣ x) → k ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l1449_144973


namespace NUMINAMATH_CALUDE_total_lives_game_lives_calculation_l1449_144979

theorem total_lives (initial_lives : ℕ) (extra_lives_level1 : ℕ) (extra_lives_level2 : ℕ) :
  initial_lives + extra_lives_level1 + extra_lives_level2 =
  initial_lives + extra_lives_level1 + extra_lives_level2 :=
by sorry

theorem game_lives_calculation :
  let initial_lives : ℕ := 2
  let extra_lives_level1 : ℕ := 6
  let extra_lives_level2 : ℕ := 11
  initial_lives + extra_lives_level1 + extra_lives_level2 = 19 :=
by sorry

end NUMINAMATH_CALUDE_total_lives_game_lives_calculation_l1449_144979


namespace NUMINAMATH_CALUDE_unique_a_value_l1449_144955

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 4

-- Define the condition for the function to be positive outside [2, 8]
def positive_outside (a : ℝ) : Prop :=
  ∀ x, (x < 2 ∨ x > 8) → f a x > 0

-- Theorem statement
theorem unique_a_value : ∃! a : ℝ, positive_outside a :=
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1449_144955


namespace NUMINAMATH_CALUDE_inverse_quadratic_equation_l1449_144968

theorem inverse_quadratic_equation (x : ℝ) :
  (1 : ℝ) = 1 / (3 * x^2 + 2 * x + 1) → x = 0 ∨ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_quadratic_equation_l1449_144968


namespace NUMINAMATH_CALUDE_trigonometric_properties_l1449_144949

theorem trigonometric_properties :
  (¬ ∃ α : ℝ, Real.sin α + Real.cos α = 3/2) ∧
  (∀ x : ℝ, Real.cos (7 * Real.pi / 2 - 3 * x) = -Real.cos (7 * Real.pi / 2 + 3 * x)) ∧
  (∀ x : ℝ, 4 * Real.sin (2 * (-9 * Real.pi / 8 + x) + 5 * Real.pi / 4) = 
            4 * Real.sin (2 * (-9 * Real.pi / 8 - x) + 5 * Real.pi / 4)) ∧
  (∃ x : ℝ, Real.sin (2 * x - Real.pi / 4) ≠ Real.sin (2 * (x - Real.pi / 8))) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_properties_l1449_144949


namespace NUMINAMATH_CALUDE_silver_dollar_difference_l1449_144904

/-- The number of silver dollars owned by Mr. Chiu -/
def chiu_dollars : ℕ := 56

/-- The number of silver dollars owned by Mr. Phung -/
def phung_dollars : ℕ := chiu_dollars + 16

/-- The number of silver dollars owned by Mr. Ha -/
def ha_dollars : ℕ := 205 - phung_dollars - chiu_dollars

theorem silver_dollar_difference : ha_dollars - phung_dollars = 5 := by
  sorry

end NUMINAMATH_CALUDE_silver_dollar_difference_l1449_144904


namespace NUMINAMATH_CALUDE_max_value_sum_l1449_144939

theorem max_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt (3 * b^2) = Real.sqrt ((1 - a) * (1 + a))) : 
  ∃ (x : ℝ), x = a + Real.sqrt (3 * b^2) ∧ x ≤ Real.sqrt 2 ∧ 
  ∀ (y : ℝ), y = a + Real.sqrt (3 * b^2) → y ≤ x := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_l1449_144939


namespace NUMINAMATH_CALUDE_lowest_common_multiple_even_14_to_21_l1449_144947

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem lowest_common_multiple_even_14_to_21 :
  ∀ n : ℕ, n > 0 →
  (∀ k : ℕ, 14 ≤ k → k ≤ 21 → is_even k → divides k n) →
  n ≥ 5040 :=
sorry

end NUMINAMATH_CALUDE_lowest_common_multiple_even_14_to_21_l1449_144947


namespace NUMINAMATH_CALUDE_probability_consecutive_days_l1449_144948

-- Define the number of days
def total_days : ℕ := 10

-- Define the number of days to be selected
def selected_days : ℕ := 3

-- Define the number of ways to select 3 consecutive days
def consecutive_selections : ℕ := total_days - selected_days + 1

-- Define the total number of ways to select 3 days from 10 days
def total_selections : ℕ := Nat.choose total_days selected_days

-- Theorem statement
theorem probability_consecutive_days :
  (consecutive_selections : ℚ) / total_selections = 1 / 15 :=
sorry

end NUMINAMATH_CALUDE_probability_consecutive_days_l1449_144948


namespace NUMINAMATH_CALUDE_balloons_kept_winnie_keeps_balloons_l1449_144908

def total_balloons : ℕ := 22 + 44 + 78 + 90
def num_friends : ℕ := 10

theorem balloons_kept (total : ℕ) (friends : ℕ) (h : friends > 0) :
  total % friends = total - friends * (total / friends) :=
by sorry

theorem winnie_keeps_balloons :
  total_balloons % num_friends = 4 :=
by sorry

end NUMINAMATH_CALUDE_balloons_kept_winnie_keeps_balloons_l1449_144908


namespace NUMINAMATH_CALUDE_divisibility_conditions_l1449_144942

def number (B : Nat) : Nat := 35380840 + B

theorem divisibility_conditions (B : Nat) : 
  B < 10 →
  (number B % 2 = 0 ∧ 
   number B % 4 = 0 ∧ 
   number B % 5 = 0 ∧ 
   number B % 6 = 0 ∧ 
   number B % 8 = 0) →
  B = 0 := by sorry

end NUMINAMATH_CALUDE_divisibility_conditions_l1449_144942


namespace NUMINAMATH_CALUDE_bryans_milk_volume_l1449_144935

/-- The volume of milk in the first bottle, given the conditions of Bryan's milk purchase --/
theorem bryans_milk_volume (total_volume : ℚ) (second_bottle : ℚ) (third_bottle : ℚ) 
  (h1 : total_volume = 3)
  (h2 : second_bottle = 750 / 1000)
  (h3 : third_bottle = 250 / 1000) :
  total_volume - second_bottle - third_bottle = 2 := by
  sorry

end NUMINAMATH_CALUDE_bryans_milk_volume_l1449_144935


namespace NUMINAMATH_CALUDE_complement_of_37_12_l1449_144901

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem complement_of_37_12 :
  let a : Angle := ⟨37, 12⟩
  complement a = ⟨52, 48⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_37_12_l1449_144901


namespace NUMINAMATH_CALUDE_arithmetic_sequence_probability_l1449_144986

-- Define the set of numbers
def S : Set Nat := Finset.range 20

-- Define a function to check if three numbers form an arithmetic sequence
def isArithmeticSequence (a b c : Nat) : Prop := a + c = 2 * b

-- Define the total number of ways to choose 3 numbers from 20
def totalCombinations : Nat := Nat.choose 20 3

-- Define the number of valid arithmetic sequences
def validSequences : Nat := 90

-- State the theorem
theorem arithmetic_sequence_probability :
  (validSequences : ℚ) / totalCombinations = 1 / 38 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_probability_l1449_144986


namespace NUMINAMATH_CALUDE_areas_product_eq_volume_squared_l1449_144915

/-- A rectangular prism with dimensions x, y, and z. -/
structure RectangularPrism where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The volume of a rectangular prism. -/
def volume (p : RectangularPrism) : ℝ :=
  p.x * p.y * p.z

/-- The areas of the top, back, and lateral face of a rectangular prism. -/
def areas (p : RectangularPrism) : ℝ × ℝ × ℝ :=
  (p.x * p.y, p.y * p.z, p.z * p.x)

/-- The theorem stating that the product of the areas equals the square of the volume. -/
theorem areas_product_eq_volume_squared (p : RectangularPrism) :
  let (top, back, lateral) := areas p
  top * back * lateral = (volume p) ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_areas_product_eq_volume_squared_l1449_144915


namespace NUMINAMATH_CALUDE_evaluate_g_l1449_144930

/-- The function g(x) = 3x^2 - 5x + 7 -/
def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

/-- Theorem: 3g(2) + 2g(-4) = 177 -/
theorem evaluate_g : 3 * g 2 + 2 * g (-4) = 177 := by sorry

end NUMINAMATH_CALUDE_evaluate_g_l1449_144930


namespace NUMINAMATH_CALUDE_number_of_bowls_l1449_144921

/-- Given a table with bowls of grapes, prove that there are 16 bowls when:
  - 8 grapes are added to each of 12 bowls
  - The average number of grapes in all bowls increases by 6
-/
theorem number_of_bowls : ℕ → Prop := λ n =>
  -- n is the number of bowls
  -- Define the increase in total grapes
  let total_increase : ℕ := 12 * 8
  -- Define the increase in average
  let avg_increase : ℕ := 6
  -- The theorem: if the total increase divided by the average increase equals n, 
  -- then n is the number of bowls
  total_increase / avg_increase = n

-- The proof (skipped with sorry)
example : number_of_bowls 16 := by sorry

end NUMINAMATH_CALUDE_number_of_bowls_l1449_144921


namespace NUMINAMATH_CALUDE_weight_difference_l1449_144906

theorem weight_difference (steve jim stan : ℕ) : 
  stan = steve + 5 →
  jim = 110 →
  steve + stan + jim = 319 →
  jim - steve = 8 := by
sorry

end NUMINAMATH_CALUDE_weight_difference_l1449_144906


namespace NUMINAMATH_CALUDE_congruence_solution_l1449_144994

theorem congruence_solution (m : ℕ) : m ∈ Finset.range 47 → (13 * m ≡ 9 [ZMOD 47]) ↔ m = 29 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1449_144994


namespace NUMINAMATH_CALUDE_cos_315_degrees_l1449_144964

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l1449_144964


namespace NUMINAMATH_CALUDE_chess_tournament_matches_prime_l1449_144997

/-- Represents a single-elimination chess tournament --/
structure ChessTournament where
  totalPlayers : ℕ
  byePlayers : ℕ
  initialPlayers : ℕ

/-- Calculates the number of matches in a single-elimination tournament --/
def matchesPlayed (t : ChessTournament) : ℕ := t.totalPlayers - 1

/-- Theorem: In the given chess tournament, 119 matches are played and this number is prime --/
theorem chess_tournament_matches_prime (t : ChessTournament) 
  (h1 : t.totalPlayers = 120) 
  (h2 : t.byePlayers = 40) 
  (h3 : t.initialPlayers = 80) : 
  matchesPlayed t = 119 ∧ Nat.Prime 119 := by
  sorry

#eval Nat.Prime 119  -- To verify that 119 is indeed prime

end NUMINAMATH_CALUDE_chess_tournament_matches_prime_l1449_144997


namespace NUMINAMATH_CALUDE_cupric_cyanide_formed_l1449_144918

-- Define the chemical equation
structure ChemicalEquation :=
  (CuSO₄ : ℕ) (HCN : ℕ) (Cu_CN_₂ : ℕ) (H₂SO₄ : ℕ)

-- Define the balanced equation
def balanced_equation : ChemicalEquation :=
  ⟨1, 4, 1, 1⟩

-- Define the actual reactants
def actual_reactants : ChemicalEquation :=
  ⟨1, 2, 0, 1⟩

-- Define the function to calculate the limiting reagent
def limiting_reagent (eq : ChemicalEquation) (reactants : ChemicalEquation) : ℕ :=
  min (reactants.CuSO₄ / eq.CuSO₄) (reactants.HCN / eq.HCN)

-- Theorem to prove
theorem cupric_cyanide_formed (eq : ChemicalEquation) (reactants : ChemicalEquation) :
  eq = balanced_equation ∧ reactants = actual_reactants →
  limiting_reagent eq reactants * eq.Cu_CN_₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_cupric_cyanide_formed_l1449_144918


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1449_144969

/-- Given a rectangle with area 540 square centimeters, if its length is increased by 15%
    and its width is decreased by 20%, then its new area is 496.8 square centimeters. -/
theorem rectangle_area_change (l w : ℝ) (h1 : l * w = 540) : 
  (1.15 * l) * (0.8 * w) = 496.8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1449_144969


namespace NUMINAMATH_CALUDE_ap_has_twelve_terms_l1449_144967

/-- Represents an arithmetic progression with specific properties -/
structure ArithmeticProgression where
  n : ℕ                  -- number of terms
  first_term : ℝ         -- first term
  last_term : ℝ          -- last term
  odd_sum : ℝ            -- sum of odd-numbered terms
  even_sum : ℝ           -- sum of even-numbered terms
  even_terms : Even n    -- n is even
  first_term_eq : first_term = 3
  last_term_diff : last_term = first_term + 22.5
  odd_sum_eq : odd_sum = 42
  even_sum_eq : even_sum = 48

/-- Theorem stating that the arithmetic progression satisfying given conditions has 12 terms -/
theorem ap_has_twelve_terms (ap : ArithmeticProgression) : ap.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ap_has_twelve_terms_l1449_144967


namespace NUMINAMATH_CALUDE_kays_total_exercise_time_l1449_144913

/-- Kay's weekly exercise routine -/
structure ExerciseRoutine where
  aerobics : ℕ
  weightTraining : ℕ

/-- The total exercise time is the sum of aerobics and weight training times -/
def totalExerciseTime (routine : ExerciseRoutine) : ℕ :=
  routine.aerobics + routine.weightTraining

/-- Kay's actual exercise routine -/
def kaysRoutine : ExerciseRoutine :=
  { aerobics := 150, weightTraining := 100 }

/-- Theorem: Kay's total exercise time is 250 minutes per week -/
theorem kays_total_exercise_time :
  totalExerciseTime kaysRoutine = 250 := by
  sorry

end NUMINAMATH_CALUDE_kays_total_exercise_time_l1449_144913


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1449_144951

/-- Simple interest calculation -/
theorem simple_interest_problem (interest_rate : ℚ) (time_period : ℚ) (earned_interest : ℕ) :
  interest_rate = 50 / 3 →
  time_period = 3 / 4 →
  earned_interest = 8625 →
  ∃ (principal : ℕ), 
    principal = 6900000 ∧
    earned_interest = (principal * interest_rate * time_period : ℚ).num / 100 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1449_144951


namespace NUMINAMATH_CALUDE_sqrt_360000_equals_600_l1449_144932

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_equals_600_l1449_144932


namespace NUMINAMATH_CALUDE_initial_ratio_proof_l1449_144931

theorem initial_ratio_proof (initial_boarders : ℕ) (new_boarders : ℕ) :
  initial_boarders = 560 →
  new_boarders = 80 →
  ∃ (initial_day_scholars : ℕ),
    (initial_boarders : ℚ) / initial_day_scholars = 7 / 16 ∧
    (initial_boarders + new_boarders : ℚ) / initial_day_scholars = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_ratio_proof_l1449_144931


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_property_l1449_144966

/-- Definition of the sum of the first n terms of the sequence -/
def S (n : ℕ) (k : ℝ) : ℝ := k + 3^n

/-- Definition of a term in the sequence -/
def a (n : ℕ) (k : ℝ) : ℝ := S n k - S (n-1) k

/-- Theorem stating that k = -1 for the given conditions -/
theorem geometric_sequence_sum_property (k : ℝ) :
  (∀ n : ℕ, n ≥ 1 → a (n+1) k / a n k = a (n+2) k / a (n+1) k) →
  k = -1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_property_l1449_144966


namespace NUMINAMATH_CALUDE_trapezoid_x_squared_l1449_144952

/-- A trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  x : ℝ
  shorter_base_length : shorter_base = 50
  longer_base_length : longer_base = shorter_base + 50
  midpoint_ratio : (shorter_base + (shorter_base + longer_base) / 2) / ((shorter_base + longer_base) / 2 + longer_base) = 1 / 2
  equal_area : x > shorter_base ∧ x < longer_base ∧ 
    (x - shorter_base) / (longer_base - shorter_base) = 
    (x - shorter_base) * (x + shorter_base) / ((longer_base - shorter_base) * (longer_base + shorter_base))

theorem trapezoid_x_squared (t : Trapezoid) : t.x^2 = 6875 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_x_squared_l1449_144952


namespace NUMINAMATH_CALUDE_greg_earnings_l1449_144991

/-- Represents the rates for a dog size --/
structure DogRate where
  baseCharge : ℝ
  perMinuteCharge : ℝ

/-- Represents a group of dogs walked --/
structure DogGroup where
  count : ℕ
  minutes : ℕ

/-- Calculates the earnings for a group of dogs --/
def calculateEarnings (rate : DogRate) (group : DogGroup) : ℝ :=
  rate.baseCharge * group.count + rate.perMinuteCharge * group.count * group.minutes

/-- Theorem: Greg's total earnings for the day --/
theorem greg_earnings : 
  let extraSmallRate : DogRate := ⟨12, 0.80⟩
  let smallRate : DogRate := ⟨15, 1⟩
  let mediumRate : DogRate := ⟨20, 1.25⟩
  let largeRate : DogRate := ⟨25, 1.50⟩
  let extraLargeRate : DogRate := ⟨30, 1.75⟩

  let extraSmallGroup : DogGroup := ⟨2, 10⟩
  let smallGroup : DogGroup := ⟨3, 12⟩
  let mediumGroup : DogGroup := ⟨1, 18⟩
  let largeGroup : DogGroup := ⟨2, 25⟩
  let extraLargeGroup : DogGroup := ⟨1, 30⟩

  let totalEarnings := 
    calculateEarnings extraSmallRate extraSmallGroup +
    calculateEarnings smallRate smallGroup +
    calculateEarnings mediumRate mediumGroup +
    calculateEarnings largeRate largeGroup +
    calculateEarnings extraLargeRate extraLargeGroup

  totalEarnings = 371 := by sorry

end NUMINAMATH_CALUDE_greg_earnings_l1449_144991


namespace NUMINAMATH_CALUDE_bills_theorem_l1449_144960

/-- Represents the water and electricity bills for DingDing's family -/
structure Bills where
  may_water : ℝ
  may_total : ℝ
  june_water_increase : ℝ
  june_electricity_increase : ℝ

/-- Calculates the total bill for June -/
def june_total (b : Bills) : ℝ :=
  b.may_water * (1 + b.june_water_increase) + 
  (b.may_total - b.may_water) * (1 + b.june_electricity_increase)

/-- Calculates the total bill for May and June -/
def may_june_total (b : Bills) : ℝ :=
  b.may_total + june_total b

/-- Theorem stating the properties of the bills -/
theorem bills_theorem (b : Bills) 
  (h1 : b.may_total = 140)
  (h2 : b.june_water_increase = 0.1)
  (h3 : b.june_electricity_increase = 0.2) :
  june_total b = -0.1 * b.may_water + 168 ∧
  may_june_total b = 304 ↔ b.may_water = 40 := by
  sorry

end NUMINAMATH_CALUDE_bills_theorem_l1449_144960


namespace NUMINAMATH_CALUDE_complex_sum_equality_l1449_144963

theorem complex_sum_equality : 
  let B : ℂ := 3 - 2*I
  let Q : ℂ := 1 + 3*I
  let R : ℂ := -2 + 4*I
  let T : ℂ := 5 - 3*I
  B + Q + R + T = 7 + 2*I := by
sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l1449_144963


namespace NUMINAMATH_CALUDE_total_yield_before_change_l1449_144971

theorem total_yield_before_change (x y z : ℝ) 
  (h1 : 0.4 * x + 0.2 * y = 5)
  (h2 : 0.4 * y + 0.2 * z = 10)
  (h3 : 0.4 * z + 0.2 * x = 9) :
  x + y + z = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_yield_before_change_l1449_144971


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l1449_144907

theorem min_value_expression (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (h : x * y * z * w = 16) :
  x + 2 * y + 4 * z + 8 * w ≥ 16 :=
sorry

theorem min_value_achieved (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (h : x * y * z * w = 16) :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * b * c * d = 16 ∧ a + 2 * b + 4 * c + 8 * d = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l1449_144907


namespace NUMINAMATH_CALUDE_y_derivative_l1449_144912

noncomputable def y (x : ℝ) : ℝ := 
  (1/12) * Real.log ((x^4 - x^2 + 1) / (x^2 + 1)^2) - 
  (1 / (2 * Real.sqrt 3)) * Real.arctan (Real.sqrt 3 / (2*x^2 - 1))

theorem y_derivative (x : ℝ) : 
  deriv y x = x^3 / ((x^4 - x^2 + 1) * (x^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l1449_144912


namespace NUMINAMATH_CALUDE_volume_not_occupied_by_cones_l1449_144988

/-- The volume of a cylinder not occupied by two identical cones -/
theorem volume_not_occupied_by_cones (r h_cyl h_cone : ℝ) 
  (hr : r = 10)
  (h_cyl_height : h_cyl = 30)
  (h_cone_height : h_cone = 15) :
  π * r^2 * h_cyl - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π := by
  sorry

end NUMINAMATH_CALUDE_volume_not_occupied_by_cones_l1449_144988


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1449_144911

theorem complex_equation_sum (a b : ℝ) : 
  (a + Complex.I) * Complex.I = b + (5 / (2 - Complex.I)) → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1449_144911


namespace NUMINAMATH_CALUDE_square_of_sum_l1449_144962

theorem square_of_sum (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l1449_144962


namespace NUMINAMATH_CALUDE_red_surface_area_fraction_is_three_fourths_l1449_144970

/-- Represents a cube constructed from smaller cubes -/
structure CompositeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  red_cube_count : ℕ
  blue_cube_count : ℕ

/-- Represents the arrangement of blue cubes on the surface -/
structure BlueCubeArrangement where
  blue_corners_per_face : ℕ

/-- Calculates the fraction of red surface area -/
def red_surface_area_fraction (cube : CompositeCube) (arrangement : BlueCubeArrangement) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem red_surface_area_fraction_is_three_fourths 
  (cube : CompositeCube) 
  (arrangement : BlueCubeArrangement) : 
  cube.edge_length = 4 ∧ 
  cube.small_cube_count = 64 ∧ 
  cube.red_cube_count = 32 ∧ 
  cube.blue_cube_count = 32 ∧
  arrangement.blue_corners_per_face = 4 →
  red_surface_area_fraction cube arrangement = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_red_surface_area_fraction_is_three_fourths_l1449_144970


namespace NUMINAMATH_CALUDE_secant_slope_on_curve_l1449_144958

def f (x : ℝ) : ℝ := x^2 + x

theorem secant_slope_on_curve (Δx : ℝ) (Δy : ℝ) 
  (h1 : f 2 = 6)  -- P(2, 6) is on the curve
  (h2 : f (2 + Δx) = 6 + Δy)  -- Q(2 + Δx, 6 + Δy) is on the curve
  (h3 : Δx ≠ 0)  -- Ensure Δx is not zero for division
  : Δy / Δx = Δx + 5 := by
  sorry

#check secant_slope_on_curve

end NUMINAMATH_CALUDE_secant_slope_on_curve_l1449_144958


namespace NUMINAMATH_CALUDE_max_value_of_sum_l1449_144928

theorem max_value_of_sum (a c d : ℤ) (b : ℕ+) 
  (h1 : a + b = c) 
  (h2 : b + c = d) 
  (h3 : c + d = a) : 
  (a + b + c + d : ℤ) ≤ -5 ∧ ∃ (a₀ c₀ d₀ : ℤ) (b₀ : ℕ+), 
    a₀ + b₀ = c₀ ∧ b₀ + c₀ = d₀ ∧ c₀ + d₀ = a₀ ∧ a₀ + b₀ + c₀ + d₀ = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l1449_144928


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1449_144925

def solution_set (x : ℝ) : Prop := x ≥ 0 ∨ x ≤ -2

theorem inequality_solution_set :
  ∀ x : ℝ, x * (x + 2) ≥ 0 ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1449_144925


namespace NUMINAMATH_CALUDE_inverse_function_point_and_sum_l1449_144983

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverses of each other
axiom inverse_relation : ∀ x, f_inv (f x) = x

-- Given condition: (2,3) is on the graph of y = f(x)/2
axiom point_on_f : f 2 = 6

-- Theorem to prove
theorem inverse_function_point_and_sum :
  (f_inv 6 = 2) ∧ (6, 1) ∈ {p : ℝ × ℝ | p.2 = f_inv p.1 / 2} ∧ (6 + 1 = 7) :=
sorry

end NUMINAMATH_CALUDE_inverse_function_point_and_sum_l1449_144983


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1449_144977

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ x y : ℝ, x > y → x + 1 > y) ∧
  (∃ x y : ℝ, x + 1 > y ∧ ¬(x > y)) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1449_144977


namespace NUMINAMATH_CALUDE_equation_solution_l1449_144917

theorem equation_solution :
  ∃! x : ℝ, x ≠ 0 ∧ (9 * x)^18 = (27 * x)^9 ∧ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1449_144917


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1449_144923

/-- Given two vectors a and b in ℝ², where a is parallel to b, 
    prove that the magnitude of 2a + 3b is 4√5. -/
theorem parallel_vectors_magnitude (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (∃ (k : ℝ), a = k • b) →
  ‖(2 : ℝ) • a + (3 : ℝ) • b‖ = 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1449_144923


namespace NUMINAMATH_CALUDE_area_of_original_figure_l1449_144924

/-- Represents the properties of an oblique diametric view of a figure -/
structure ObliqueView where
  is_isosceles_trapezoid : Bool
  base_angle : ℝ
  leg_length : ℝ
  top_base_length : ℝ

/-- Calculates the area of the original plane figure given its oblique diametric view -/
def original_area (view : ObliqueView) : ℝ :=
  sorry

/-- Theorem stating the area of the original plane figure given specific oblique view properties -/
theorem area_of_original_figure (view : ObliqueView) 
  (h1 : view.is_isosceles_trapezoid = true)
  (h2 : view.base_angle = π / 4)  -- 45° in radians
  (h3 : view.leg_length = 1)
  (h4 : view.top_base_length = 1) :
  original_area view = 2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_area_of_original_figure_l1449_144924


namespace NUMINAMATH_CALUDE_a_squared_ge_three_l1449_144903

theorem a_squared_ge_three (a b c : ℝ) (h1 : a ≠ 0) (h2 : a + b + c = a * b * c) (h3 : a^2 = b * c) : a^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_a_squared_ge_three_l1449_144903


namespace NUMINAMATH_CALUDE_max_balls_theorem_l1449_144927

/-- The maximum number of balls that can be counted while maintaining at least 90% red balls -/
def max_balls : ℕ := 210

/-- The proportion of red balls in the first 50 counted -/
def initial_red_ratio : ℚ := 49 / 50

/-- The proportion of red balls in each subsequent batch of 8 -/
def subsequent_red_ratio : ℚ := 7 / 8

/-- The minimum required proportion of red balls -/
def min_red_ratio : ℚ := 9 / 10

/-- Theorem stating that max_balls is the maximum number of balls that can be counted
    while maintaining at least 90% red balls -/
theorem max_balls_theorem (n : ℕ) :
  n ≤ max_balls ↔
  (∃ x : ℕ, n = 50 + 8 * x ∧
    (initial_red_ratio * 50 + subsequent_red_ratio * 8 * x) / n ≥ min_red_ratio) :=
sorry

end NUMINAMATH_CALUDE_max_balls_theorem_l1449_144927


namespace NUMINAMATH_CALUDE_discount_difference_l1449_144902

theorem discount_difference : 
  let first_discount : ℝ := 0.25
  let second_discount : ℝ := 0.15
  let third_discount : ℝ := 0.10
  let claimed_discount : ℝ := 0.45
  let true_discount : ℝ := 1 - (1 - first_discount) * (1 - second_discount) * (1 - third_discount)
  claimed_discount - true_discount = 0.02375 := by
sorry

end NUMINAMATH_CALUDE_discount_difference_l1449_144902


namespace NUMINAMATH_CALUDE_village_population_theorem_l1449_144992

/-- Given a village with a total population and a subset of that population,
    calculate the percentage that the subset represents. -/
def village_population_percentage (total : ℕ) (subset : ℕ) : ℚ :=
  (subset : ℚ) / (total : ℚ) * 100

/-- Theorem stating that 45,000 is 90% of 50,000 -/
theorem village_population_theorem :
  village_population_percentage 50000 45000 = 90 := by
  sorry

end NUMINAMATH_CALUDE_village_population_theorem_l1449_144992


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l1449_144982

theorem complex_magnitude_proof (i a : ℂ) : 
  i ^ 2 = -1 →
  a.im = 0 →
  (∃ k : ℝ, (2 - i) / (a + i) = k * i) →
  Complex.abs ((2 * a + 1) + Real.sqrt 2 * i) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l1449_144982


namespace NUMINAMATH_CALUDE_base_number_is_two_l1449_144990

theorem base_number_is_two (a : ℕ) (x : ℕ) (h1 : a^x - a^(x-2) = 3 * 2^11) (h2 : x = 13) :
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_number_is_two_l1449_144990


namespace NUMINAMATH_CALUDE_instrument_probability_l1449_144905

theorem instrument_probability (total : ℕ) (at_least_one : ℚ) (two_or_more : ℕ) : 
  total = 800 → 
  at_least_one = 3/5 → 
  two_or_more = 96 → 
  (((at_least_one * total) - two_or_more) : ℚ) / total = 48/100 := by
sorry

end NUMINAMATH_CALUDE_instrument_probability_l1449_144905


namespace NUMINAMATH_CALUDE_correct_employee_count_l1449_144940

/-- The number of employees in John's company --/
def number_of_employees : ℕ := 85

/-- The cost of each turkey in dollars --/
def cost_per_turkey : ℕ := 25

/-- The total amount spent on turkeys in dollars --/
def total_spent : ℕ := 2125

/-- Theorem stating that the number of employees is correct given the conditions --/
theorem correct_employee_count :
  number_of_employees * cost_per_turkey = total_spent :=
by sorry

end NUMINAMATH_CALUDE_correct_employee_count_l1449_144940


namespace NUMINAMATH_CALUDE_jellybean_problem_l1449_144929

theorem jellybean_problem (initial_count : ℕ) : 
  (((initial_count : ℚ) * (3/4)^3).floor = 27) → initial_count = 64 := by
sorry

end NUMINAMATH_CALUDE_jellybean_problem_l1449_144929


namespace NUMINAMATH_CALUDE_student_count_l1449_144943

theorem student_count : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 15 = 0 ∧ 
  n % 6 = 0 ∧ 
  (∀ m : ℕ, m > 0 → m % 15 = 0 → m % 6 = 0 → m ≥ n) ∧
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_student_count_l1449_144943


namespace NUMINAMATH_CALUDE_meaningful_expression_l1449_144989

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 1)) ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l1449_144989


namespace NUMINAMATH_CALUDE_smallest_b_value_l1449_144956

theorem smallest_b_value (a b c : ℕ+) (h : (31 : ℚ) / 72 = a / 8 + b / 9 - c) : 
  ∀ b' : ℕ+, b' < b → ¬∃ (a' c' : ℕ+), (31 : ℚ) / 72 = a' / 8 + b' / 9 - c' :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1449_144956


namespace NUMINAMATH_CALUDE_sqrt_rational_l1449_144926

theorem sqrt_rational (a b c : ℚ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : Real.sqrt a + Real.sqrt b = c) : 
  ∃ (q r : ℚ), Real.sqrt a = q ∧ Real.sqrt b = r := by
sorry

end NUMINAMATH_CALUDE_sqrt_rational_l1449_144926


namespace NUMINAMATH_CALUDE_soda_cost_calculation_l1449_144978

theorem soda_cost_calculation (regular_bottles : ℕ) (regular_price : ℚ) 
  (diet_bottles : ℕ) (diet_price : ℚ) (regular_discount : ℚ) (diet_tax : ℚ) :
  regular_bottles = 49 →
  regular_price = 120/100 →
  diet_bottles = 40 →
  diet_price = 110/100 →
  regular_discount = 10/100 →
  diet_tax = 8/100 →
  (regular_bottles : ℚ) * regular_price * (1 - regular_discount) + 
  (diet_bottles : ℚ) * diet_price * (1 + diet_tax) = 10044/100 := by
sorry

end NUMINAMATH_CALUDE_soda_cost_calculation_l1449_144978


namespace NUMINAMATH_CALUDE_lucky_lacy_correct_percentage_l1449_144938

theorem lucky_lacy_correct_percentage (x : ℕ) : 
  let total_problems : ℕ := 4 * x
  let missed_problems : ℕ := 2 * x
  let correct_problems : ℕ := total_problems - missed_problems
  (correct_problems : ℚ) / (total_problems : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_lucky_lacy_correct_percentage_l1449_144938


namespace NUMINAMATH_CALUDE_simplest_fraction_of_0375_l1449_144944

theorem simplest_fraction_of_0375 (c d : ℕ+) : 
  (c : ℚ) / (d : ℚ) = 0.375 ∧ 
  (∀ (m n : ℕ+), (m : ℚ) / (n : ℚ) = 0.375 → c ≤ m ∧ d ≤ n) →
  c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_simplest_fraction_of_0375_l1449_144944
