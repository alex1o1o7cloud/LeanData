import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_range_l3387_338731

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - 3| < a) → a > 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3387_338731


namespace NUMINAMATH_CALUDE_soup_donation_theorem_l3387_338753

theorem soup_donation_theorem (shelters cans_per_person total_cans : ℕ) 
  (h1 : shelters = 6)
  (h2 : cans_per_person = 10)
  (h3 : total_cans = 1800) :
  total_cans / (shelters * cans_per_person) = 30 := by
  sorry

end NUMINAMATH_CALUDE_soup_donation_theorem_l3387_338753


namespace NUMINAMATH_CALUDE_harmonic_sum_identity_l3387_338780

def h (n : ℕ) : ℚ :=
  (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

theorem harmonic_sum_identity (n : ℕ) (hn : n ≥ 2) :
  n + (Finset.range (n - 1)).sum h = n * h n := by
  sorry

end NUMINAMATH_CALUDE_harmonic_sum_identity_l3387_338780


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3387_338791

/-- Given a quadratic equation mx^2 + x - m^2 + 1 = 0 with -1 as a root, m must equal 1 -/
theorem quadratic_root_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, m*x^2 + x - m^2 + 1 = 0 → x = -1) → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3387_338791


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_attained_l3387_338712

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
    (h_sum : 3 * x₁ + 2 * x₂ + x₃ = 30) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 125 := by
  sorry

theorem min_sum_squares_attained (ε : ℝ) (h_pos : ε > 0) : 
  ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ 
    3 * x₁ + 2 * x₂ + x₃ = 30 ∧ 
    x₁^2 + x₂^2 + x₃^2 < 125 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_attained_l3387_338712


namespace NUMINAMATH_CALUDE_unique_number_with_gcd_l3387_338742

theorem unique_number_with_gcd : ∃! n : ℕ, 70 ≤ n ∧ n < 80 ∧ Nat.gcd 30 n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_gcd_l3387_338742


namespace NUMINAMATH_CALUDE_sum_20_is_850_l3387_338721

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums of the sequence
  sum_5 : S 5 = 10
  sum_10 : S 10 = 50

/-- The sum of the first 20 terms of the geometric sequence is 850 -/
theorem sum_20_is_850 (seq : GeometricSequence) : seq.S 20 = 850 := by
  sorry

end NUMINAMATH_CALUDE_sum_20_is_850_l3387_338721


namespace NUMINAMATH_CALUDE_parallel_lines_k_values_l3387_338716

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line l₁: (k-3)x+(4-k)y+1=0 -/
def l1 (k : ℝ) : Line :=
  { a := k - 3, b := 4 - k, c := 1 }

/-- The second line l₂: 2(k-3)-2y+3=0, rewritten as 2(k-3)x-2y+3=0 -/
def l2 (k : ℝ) : Line :=
  { a := 2 * (k - 3), b := -2, c := 3 }

/-- Theorem stating that if l₁ and l₂ are parallel, then k is either 3 or 5 -/
theorem parallel_lines_k_values :
  ∀ k, parallel (l1 k) (l2 k) → k = 3 ∨ k = 5 := by
  sorry

#check parallel_lines_k_values

end NUMINAMATH_CALUDE_parallel_lines_k_values_l3387_338716


namespace NUMINAMATH_CALUDE_discounted_price_l3387_338725

/-- Given a top with an original price of m yuan and a discount of 20%,
    the actual selling price is 0.8m yuan. -/
theorem discounted_price (m : ℝ) : 
  let original_price := m
  let discount_rate := 0.2
  let selling_price := m * (1 - discount_rate)
  selling_price = 0.8 * m := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_l3387_338725


namespace NUMINAMATH_CALUDE_function_value_2007_l3387_338763

def is_multiplicative (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, f (x + y) = f x * f y

theorem function_value_2007 (f : ℕ+ → ℕ+) 
  (h_mult : is_multiplicative f) (h_base : f 1 = 2) : 
  f 2007 = 2^2007 := by
  sorry

end NUMINAMATH_CALUDE_function_value_2007_l3387_338763


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l3387_338737

theorem triangle_angle_inequality (α β γ s R : Real) : 
  α > 0 → β > 0 → γ > 0 → 
  α + β + γ = π →
  s > 0 → R > 0 →
  (α + β) * (β + γ) * (γ + α) ≤ 4 * (π / Real.sqrt 3)^3 * R / s := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l3387_338737


namespace NUMINAMATH_CALUDE_range_of_k_l3387_338781

theorem range_of_k (k : ℝ) : 
  (k ≠ 0) → 
  (k^2 * 1^2 - 6*k*1 + 8 ≥ 0) → 
  ((k ≥ 4) ∨ (k ≤ 2)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l3387_338781


namespace NUMINAMATH_CALUDE_scientific_notation_conversion_l3387_338797

theorem scientific_notation_conversion :
  (1.8 : ℝ) * (10 ^ 8) = 180000000 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_conversion_l3387_338797


namespace NUMINAMATH_CALUDE_amy_garden_seeds_l3387_338740

theorem amy_garden_seeds (initial_seeds : ℕ) (big_garden_seeds : ℕ) (small_gardens : ℕ) 
  (h1 : initial_seeds = 101)
  (h2 : big_garden_seeds = 47)
  (h3 : small_gardens = 9) :
  (initial_seeds - big_garden_seeds) / small_gardens = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_garden_seeds_l3387_338740


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3387_338744

/-- Proves that given specific conditions on the original price, final price, and second discount,
    the first discount must be 12%. -/
theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 400 →
  final_price = 334.4 →
  second_discount = 5 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    first_discount = 12 := by
  sorry

#check first_discount_percentage

end NUMINAMATH_CALUDE_first_discount_percentage_l3387_338744


namespace NUMINAMATH_CALUDE_train_crossing_time_l3387_338794

/-- The time taken for a train to cross a telegraph post -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 320 ∧ train_speed_kmh = 72 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 16 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3387_338794


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l3387_338720

theorem simultaneous_equations_solution :
  ∀ (a x : ℝ),
    (5 * x^3 + a * x^2 + 8 = 0 ∧ 5 * x^3 + 8 * x^2 + a = 0) ↔
    ((a = -13 ∧ x = 1) ∨ (a = -3 ∧ x = -1) ∨ (a = 8 ∧ x = -2)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l3387_338720


namespace NUMINAMATH_CALUDE_hiker_speed_l3387_338762

/-- Proves that given a cyclist traveling at 10 miles per hour who passes a hiker, 
    stops 5 minutes later, and waits 7.5 minutes for the hiker to catch up, 
    the hiker's constant speed is 50/7.5 miles per hour. -/
theorem hiker_speed (cyclist_speed : ℝ) (cyclist_travel_time : ℝ) (hiker_catch_up_time : ℝ) :
  cyclist_speed = 10 →
  cyclist_travel_time = 5 / 60 →
  hiker_catch_up_time = 7.5 / 60 →
  (cyclist_speed * cyclist_travel_time) / hiker_catch_up_time = 50 / 7.5 := by
  sorry

#eval (50 : ℚ) / 7.5

end NUMINAMATH_CALUDE_hiker_speed_l3387_338762


namespace NUMINAMATH_CALUDE_area_between_curves_l3387_338701

theorem area_between_curves : 
  let f (x : ℝ) := Real.exp x
  let g (x : ℝ) := Real.exp (-x)
  let a := 0
  let b := 1
  ∫ x in a..b, (f x - g x) = Real.exp 1 + Real.exp (-1) - 2 := by
  sorry

end NUMINAMATH_CALUDE_area_between_curves_l3387_338701


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_is_34_min_value_achieved_l3387_338713

theorem min_value_expression (a b c d : ℕ) : 
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∀ x y z w : ℕ, Odd x ∧ Odd y ∧ Odd z ∧ Odd w ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
  2 * a * b * c * d - (a * b * c + a * b * d + a * c * d + b * c * d) ≥
  2 * x * y * z * w - (x * y * z + x * y * w + x * z * w + y * z * w) :=
by sorry

theorem min_value_is_34 (a b c d : ℕ) : 
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  2 * a * b * c * d - (a * b * c + a * b * d + a * c * d + b * c * d) ≥ 34 :=
by sorry

theorem min_value_achieved (a b c d : ℕ) : 
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∃ x y z w : ℕ, Odd x ∧ Odd y ∧ Odd z ∧ Odd w ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
  2 * x * y * z * w - (x * y * z + x * y * w + x * z * w + y * z * w) = 34 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_is_34_min_value_achieved_l3387_338713


namespace NUMINAMATH_CALUDE_dish_price_l3387_338733

/-- The original price of a dish given specific discount and tip conditions --/
def original_price : ℝ → Prop :=
  λ price =>
    let john_payment := price * 0.9 + price * 0.15
    let jane_payment := price * 0.9 + price * 0.9 * 0.15
    john_payment - jane_payment = 0.60

theorem dish_price : ∃ (price : ℝ), original_price price ∧ price = 40 := by
  sorry

end NUMINAMATH_CALUDE_dish_price_l3387_338733


namespace NUMINAMATH_CALUDE_quadratic_point_theorem_l3387_338722

/-- A quadratic function f(x) = ax^2 + bx + c passing through (2, 6) -/
def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 4

/-- The theorem stating that if f(2) = 6, then 2a - 3b + 4c = 29 -/
theorem quadratic_point_theorem : f 2 = 6 → 2 * 2 - 3 * (-3) + 4 * 4 = 29 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_theorem_l3387_338722


namespace NUMINAMATH_CALUDE_sequence_expression_l3387_338782

theorem sequence_expression (a : ℕ → ℕ) :
  a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) - 2 * a n = 2^n) →
  ∀ n : ℕ, n ≥ 1 → a n = n * 2^(n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_expression_l3387_338782


namespace NUMINAMATH_CALUDE_reciprocal_problem_l3387_338790

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 3) : 200 * (1 / x) = 1600 / 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l3387_338790


namespace NUMINAMATH_CALUDE_largest_reciprocal_l3387_338708

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 2/7 → b = 3/8 → c = 1 → d = 4 → e = 2000 → 
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l3387_338708


namespace NUMINAMATH_CALUDE_ratio_problem_l3387_338750

theorem ratio_problem (a b c d : ℝ) 
  (h1 : b / a = 3)
  (h2 : d / b = 4)
  (h3 : c = (a + b) / 2) :
  (a + b + c) / (b + c + d) = 8 / 17 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3387_338750


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3387_338747

theorem gcd_of_three_numbers : Nat.gcd 13926 (Nat.gcd 20031 47058) = 33 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3387_338747


namespace NUMINAMATH_CALUDE_odd_function_property_l3387_338711

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (h_odd : IsOdd f) (h_diff : f 3 - f 2 = 1) :
  f (-2) - f (-3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3387_338711


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3387_338726

/-- A quadratic expression x^2 + bx + c is a perfect square trinomial if there exists a real number k such that x^2 + bx + c = (x + k)^2 for all x. -/
def IsPerfectSquareTrinomial (b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x^2 + b*x + c = (x + k)^2

/-- If x^2 - 8x + a is a perfect square trinomial, then a = 16. -/
theorem perfect_square_trinomial_condition (a : ℝ) :
  IsPerfectSquareTrinomial (-8) a → a = 16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3387_338726


namespace NUMINAMATH_CALUDE_females_band_not_orchestra_l3387_338743

/-- Represents the number of students in different groups -/
structure StudentGroups where
  bandFemales : ℕ
  bandMales : ℕ
  orchestraFemales : ℕ
  orchestraMales : ℕ
  bothFemales : ℕ
  totalStudents : ℕ

/-- Theorem stating the number of females in the band but not in the orchestra -/
theorem females_band_not_orchestra (g : StudentGroups)
  (h1 : g.bandFemales = 120)
  (h2 : g.bandMales = 70)
  (h3 : g.orchestraFemales = 70)
  (h4 : g.orchestraMales = 110)
  (h5 : g.bothFemales = 45)
  (h6 : g.totalStudents = 250) :
  g.bandFemales - g.bothFemales = 75 := by
  sorry

#check females_band_not_orchestra

end NUMINAMATH_CALUDE_females_band_not_orchestra_l3387_338743


namespace NUMINAMATH_CALUDE_terminal_side_quadrant_l3387_338751

-- Define the angle in degrees
def angle : ℤ := -1060

-- Define a function to convert an angle to its equivalent angle between 0° and 360°
def normalizeAngle (θ : ℤ) : ℤ :=
  θ % 360

-- Define a function to determine the quadrant of an angle
def determineQuadrant (θ : ℤ) : ℕ :=
  let normalizedAngle := normalizeAngle θ
  if 0 ≤ normalizedAngle ∧ normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle ∧ normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle ∧ normalizedAngle < 270 then 3
  else 4

-- Theorem statement
theorem terminal_side_quadrant :
  determineQuadrant angle = 1 := by sorry

end NUMINAMATH_CALUDE_terminal_side_quadrant_l3387_338751


namespace NUMINAMATH_CALUDE_snow_probability_in_ten_days_l3387_338727

/-- Probability of snow on a given day -/
def snow_prob (day : ℕ) : ℚ :=
  if day ≤ 5 then 1/5 else 1/3

/-- Probability of temperature dropping below 0°C -/
def cold_prob : ℚ := 1/2

/-- Increase in snow probability when temperature drops below 0°C -/
def snow_prob_increase : ℚ := 1/10

/-- Adjusted probability of no snow on a given day -/
def adj_no_snow_prob (day : ℕ) : ℚ :=
  cold_prob * (1 - snow_prob day) + (1 - cold_prob) * (1 - snow_prob day - snow_prob_increase)

/-- Probability of no snow for the entire period -/
def no_snow_prob : ℚ :=
  (adj_no_snow_prob 1)^5 * (adj_no_snow_prob 6)^5

theorem snow_probability_in_ten_days :
  1 - no_snow_prob = 58806/59049 :=
sorry

end NUMINAMATH_CALUDE_snow_probability_in_ten_days_l3387_338727


namespace NUMINAMATH_CALUDE_representatives_selection_l3387_338761

/-- The number of ways to select representatives from a group of male and female students. -/
def select_representatives (num_male num_female num_total num_min_female : ℕ) : ℕ :=
  (Nat.choose num_female 2 * Nat.choose num_male 2) +
  (Nat.choose num_female 3 * Nat.choose num_male 1) +
  (Nat.choose num_female 4 * Nat.choose num_male 0)

/-- Theorem stating that selecting 4 representatives from 5 male and 4 female students,
    with at least 2 females, can be done in 81 ways. -/
theorem representatives_selection :
  select_representatives 5 4 4 2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_representatives_selection_l3387_338761


namespace NUMINAMATH_CALUDE_sum_of_multiples_l3387_338764

theorem sum_of_multiples (x y : ℤ) (hx : 6 ∣ x) (hy : 9 ∣ y) : 3 ∣ (x + y) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l3387_338764


namespace NUMINAMATH_CALUDE_expression_simplification_l3387_338775

theorem expression_simplification (y : ℝ) : 
  y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3387_338775


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_bound_l3387_338759

/-- Two circles touching the sides of an angle but not each other -/
structure AngleTouchingCircles where
  R : ℝ
  r : ℝ
  h1 : R > r
  h2 : R > 0
  h3 : r > 0
  PQ : ℝ
  h4 : PQ > 0

/-- The length of the common internal tangent segment is greater than twice the geometric mean of the radii -/
theorem common_internal_tangent_length_bound (c : AngleTouchingCircles) : 
  c.PQ > 2 * Real.sqrt (c.R * c.r) := by
  sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_bound_l3387_338759


namespace NUMINAMATH_CALUDE_factorization_implies_m_values_l3387_338724

theorem factorization_implies_m_values (m : ℤ) :
  (∃ (a b : ℤ), ∀ (x : ℤ), x^2 + m*x - 4 = a*x + b) →
  m ∈ ({-3, 0, 3} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_factorization_implies_m_values_l3387_338724


namespace NUMINAMATH_CALUDE_sequence_general_term_l3387_338715

/-- Given a sequence {aₙ} where the sum of its first n terms Sₙ satisfies
    Sₙ = (1/3)aₙ + (2/3), prove that aₙ = (-1/2)^(n-1) for all n ≥ 1. -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = (1/3) * a n + 2/3) :
  ∀ n : ℕ, n ≥ 1 → a n = (-1/2)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3387_338715


namespace NUMINAMATH_CALUDE_circle_radius_in_ellipse_l3387_338739

/-- Two circles of radius r are externally tangent to each other and internally tangent to the ellipse x^2 + 4y^2 = 5. This theorem proves that the radius r is equal to √15/4. -/
theorem circle_radius_in_ellipse (r : ℝ) : 
  (∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x - r)^2 + y^2 = r^2) →
  (∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x + r)^2 + y^2 = r^2) →
  r = Real.sqrt 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_in_ellipse_l3387_338739


namespace NUMINAMATH_CALUDE_next_two_numbers_l3387_338757

def arithmetic_sequence (n : ℕ) : ℕ := n + 1

theorem next_two_numbers (n : ℕ) (h : n ≥ 6) :
  arithmetic_sequence n = n + 1 ∧
  arithmetic_sequence (n + 1) = n + 2 :=
by sorry

end NUMINAMATH_CALUDE_next_two_numbers_l3387_338757


namespace NUMINAMATH_CALUDE_ellipse_dot_product_range_l3387_338749

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of the left focus -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- Definition of the right focus -/
def F₂ : ℝ × ℝ := (1, 0)

/-- Definition of a point being on a line through F₂ -/
def is_on_line_through_F₂ (x y : ℝ) : Prop :=
  ∃ k : ℝ, y = k * (x - F₂.1)

/-- The dot product of vectors F₁M and F₁N -/
def F₁M_dot_F₁N (M N : ℝ × ℝ) : ℝ :=
  (M.1 - F₁.1) * (N.1 - F₁.1) + (M.2 - F₁.2) * (N.2 - F₁.2)

/-- The main theorem -/
theorem ellipse_dot_product_range :
  ∀ M N : ℝ × ℝ,
  is_on_ellipse M.1 M.2 →
  is_on_ellipse N.1 N.2 →
  is_on_line_through_F₂ M.1 M.2 →
  is_on_line_through_F₂ N.1 N.2 →
  M ≠ N →
  -1 ≤ F₁M_dot_F₁N M N ∧ F₁M_dot_F₁N M N ≤ 7/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_range_l3387_338749


namespace NUMINAMATH_CALUDE_albert_additional_laps_l3387_338773

/-- Calculates the number of additional complete laps needed to finish a given distance. -/
def additional_laps (total_distance : ℕ) (track_length : ℕ) (completed_laps : ℕ) : ℕ :=
  ((total_distance - completed_laps * track_length) / track_length : ℕ)

/-- Theorem: Given the specific conditions, the number of additional complete laps is 5. -/
theorem albert_additional_laps :
  additional_laps 99 9 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_albert_additional_laps_l3387_338773


namespace NUMINAMATH_CALUDE_negation_of_forall_leq_zero_l3387_338702

theorem negation_of_forall_leq_zero :
  (¬ ∀ x : ℝ, x^2 - x ≤ 0) ↔ (∃ x₀ : ℝ, x₀^2 - x₀ > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_leq_zero_l3387_338702


namespace NUMINAMATH_CALUDE_choose_three_from_ten_l3387_338723

theorem choose_three_from_ten : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_ten_l3387_338723


namespace NUMINAMATH_CALUDE_prove_age_difference_l3387_338745

-- Define the given information
def wayne_age_2021 : ℕ := 37
def peter_age_diff : ℕ := 3
def julia_birth_year : ℕ := 1979

-- Define the current year
def current_year : ℕ := 2021

-- Define the age difference between Julia and Peter
def julia_peter_age_diff : ℕ := 2

-- Theorem to prove
theorem prove_age_difference :
  (current_year - wayne_age_2021 - peter_age_diff) - julia_birth_year = julia_peter_age_diff :=
by sorry

end NUMINAMATH_CALUDE_prove_age_difference_l3387_338745


namespace NUMINAMATH_CALUDE_circle_center_l3387_338707

/-- The center of the circle defined by x^2 + y^2 - 4x - 2y - 5 = 0 is (2, 1) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 - 4*x - 2*y - 5 = 0) → (∃ r : ℝ, (x - 2)^2 + (y - 1)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l3387_338707


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3387_338756

theorem polynomial_division_theorem (x : ℝ) :
  8 * x^3 - 2 * x^2 + 4 * x - 7 = (x - 1) * (8 * x^2 + 6 * x + 10) + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3387_338756


namespace NUMINAMATH_CALUDE_medical_team_composition_l3387_338729

theorem medical_team_composition (total : ℕ) 
  (female_nurses male_nurses female_doctors male_doctors : ℕ) :
  total = 13 →
  female_nurses + male_nurses + female_doctors + male_doctors = total →
  female_nurses + male_nurses ≥ female_doctors + male_doctors →
  male_doctors > female_nurses →
  female_nurses > male_nurses →
  female_doctors ≥ 1 →
  female_nurses = 4 ∧ male_nurses = 3 ∧ female_doctors = 1 ∧ male_doctors = 5 :=
by sorry

end NUMINAMATH_CALUDE_medical_team_composition_l3387_338729


namespace NUMINAMATH_CALUDE_summer_birth_year_divisibility_l3387_338783

theorem summer_birth_year_divisibility : ∃ (x y : ℕ), 
  x < y ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  (1961 - x) % x = 0 ∧ 
  (1961 - y) % y = 0 := by
sorry

end NUMINAMATH_CALUDE_summer_birth_year_divisibility_l3387_338783


namespace NUMINAMATH_CALUDE_tadd_3000th_number_l3387_338719

/-- Represents the counting game with Tadd, Todd, and Tucker --/
structure CountingGame where
  max_count : Nat
  tadd_start : Nat
  todd_initial_count : Nat
  tucker_initial_count : Nat
  increment : Nat

/-- Calculates Tadd's nth number in the game --/
def tadd_nth_number (game : CountingGame) (n : Nat) : Nat :=
  sorry

/-- The main theorem stating that Tadd's 3000th number is X --/
theorem tadd_3000th_number (game : CountingGame) 
  (h1 : game.max_count = 15000)
  (h2 : game.tadd_start = 1)
  (h3 : game.todd_initial_count = 3)
  (h4 : game.tucker_initial_count = 5)
  (h5 : game.increment = 2) :
  tadd_nth_number game 3000 = X :=
  sorry

end NUMINAMATH_CALUDE_tadd_3000th_number_l3387_338719


namespace NUMINAMATH_CALUDE_imo_1996_p5_l3387_338795

theorem imo_1996_p5 (n p q : ℕ+) (x : ℕ → ℤ)
  (h_npq : n > p + q)
  (h_x0 : x 0 = 0)
  (h_xn : x n = 0)
  (h_diff : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (x i - x (i-1) = p ∨ x i - x (i-1) = -q)) :
  ∃ i j : ℕ, 0 ≤ i ∧ i < j ∧ j ≤ n ∧ (i, j) ≠ (0, n) ∧ x i = x j :=
sorry

end NUMINAMATH_CALUDE_imo_1996_p5_l3387_338795


namespace NUMINAMATH_CALUDE_rice_restocking_solution_l3387_338789

def rice_restocking_problem (initial_stock sold final_stock : ℕ) : ℕ :=
  final_stock - (initial_stock - sold)

theorem rice_restocking_solution :
  rice_restocking_problem 55 23 164 = 132 := by
  sorry

end NUMINAMATH_CALUDE_rice_restocking_solution_l3387_338789


namespace NUMINAMATH_CALUDE_tom_dance_frequency_l3387_338770

/-- Represents the number of times Tom dances per week -/
def dance_frequency (hours_per_session : ℕ) (years : ℕ) (total_hours : ℕ) (weeks_per_year : ℕ) : ℕ :=
  (total_hours / (years * weeks_per_year)) / hours_per_session

/-- Proves that Tom dances 4 times a week given the conditions -/
theorem tom_dance_frequency :
  dance_frequency 2 10 4160 52 = 4 := by
sorry

end NUMINAMATH_CALUDE_tom_dance_frequency_l3387_338770


namespace NUMINAMATH_CALUDE_triangle_shape_l3387_338758

theorem triangle_shape (A B C : ℝ) (hA : 0 < A ∧ A < π/2) (hB : 0 < B ∧ B < π/2) 
  (hcos : Real.cos A > Real.sin B) : 
  A + B + C = π ∧ C > π/2 :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l3387_338758


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l3387_338796

theorem binomial_coefficient_divisibility (p n : ℕ) (hp : p.Prime) (hn : n ≥ p) :
  ∃ k : ℤ, (Nat.choose n p : ℤ) - (n / p : ℤ) = k * p := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l3387_338796


namespace NUMINAMATH_CALUDE_expression_evaluation_l3387_338755

theorem expression_evaluation : 
  let a : ℚ := 2
  let b : ℚ := 1/3
  3*(a^2 - a*b + 7) - 2*(3*a*b - a^2 + 1) + 3 = 36 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3387_338755


namespace NUMINAMATH_CALUDE_line_equation_of_points_on_parabola_l3387_338767

/-- Given a parabola y² = 4x and two points on it with midpoint (2, 2), 
    the line through these points has equation x - y = 0 -/
theorem line_equation_of_points_on_parabola (A B : ℝ × ℝ) : 
  (A.2^2 = 4 * A.1) →  -- A is on the parabola
  (B.2^2 = 4 * B.1) →  -- B is on the parabola
  ((A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 2) →  -- midpoint is (2, 2)
  ∃ (k : ℝ), ∀ (x y : ℝ), (x - A.1) = k * (y - A.2) ∧ x - y = 0 :=
sorry

end NUMINAMATH_CALUDE_line_equation_of_points_on_parabola_l3387_338767


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3387_338786

/-- The ratio of the area to the square of the perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  let perimeter : ℝ := 3 * side_length
  area / perimeter^2 = Real.sqrt 3 / 36 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3387_338786


namespace NUMINAMATH_CALUDE_system_is_linear_l3387_338718

-- Define what a linear equation is
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ (x y : ℝ), f x y = a * x + b * y + c

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  (4 * x - y = 1) ∧ (y = 2 * x + 3)

-- Theorem statement
theorem system_is_linear : 
  (∃ (f g : ℝ → ℝ → ℝ), (∀ x y, system x y ↔ f x y = 0 ∧ g x y = 0) ∧ 
                          is_linear_equation f ∧ is_linear_equation g) :=
sorry

end NUMINAMATH_CALUDE_system_is_linear_l3387_338718


namespace NUMINAMATH_CALUDE_marcos_boat_distance_l3387_338732

/-- The distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proof that given a speed of 30 mph and a time of 10 minutes, the distance traveled is 5 miles -/
theorem marcos_boat_distance :
  let speed : ℝ := 30  -- Speed in miles per hour
  let time : ℝ := 10 / 60  -- Time in hours (10 minutes converted to hours)
  distance speed time = 5 := by
  sorry

end NUMINAMATH_CALUDE_marcos_boat_distance_l3387_338732


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3387_338703

/-- A polynomial with integer coefficients of the form x^3 + b₂x^2 + b₁x + 18 = 0 -/
def IntPolynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ :=
  x^3 + b₂ * x^2 + b₁ * x + 18

/-- The set of all possible integer roots of the polynomial -/
def PossibleRoots : Set ℤ :=
  {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  ∀ x : ℤ, IntPolynomial b₂ b₁ x = 0 → x ∈ PossibleRoots :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3387_338703


namespace NUMINAMATH_CALUDE_book_club_hardcover_cost_l3387_338768

/-- Proves that the cost of each hardcover book is $30 given the book club fee structure --/
theorem book_club_hardcover_cost :
  let members : ℕ := 6
  let snack_fee : ℕ := 150
  let hardcover_count : ℕ := 6
  let paperback_count : ℕ := 6
  let paperback_cost : ℕ := 12
  let total_collected : ℕ := 2412
  ∃ (hardcover_cost : ℕ),
    members * (snack_fee + hardcover_count * hardcover_cost + paperback_count * paperback_cost) = total_collected ∧
    hardcover_cost = 30 :=
by sorry

end NUMINAMATH_CALUDE_book_club_hardcover_cost_l3387_338768


namespace NUMINAMATH_CALUDE_book_cost_price_l3387_338710

/-- Given a book sold at 10% profit, and if sold for $140 more would result in 15% profit,
    prove that the cost price of the book is $2800. -/
theorem book_cost_price (C : ℝ) 
  (h1 : 1.10 * C + 140 = 1.15 * C) : C = 2800 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l3387_338710


namespace NUMINAMATH_CALUDE_f_at_negative_two_l3387_338709

/-- Given a function f(x) = 2x^2 - 3x + 1, prove that f(-2) = 15 -/
theorem f_at_negative_two (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x^2 - 3 * x + 1) : 
  f (-2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_two_l3387_338709


namespace NUMINAMATH_CALUDE_pet_shop_dogs_l3387_338730

/-- Given a pet shop with dogs, bunnies, and birds in the ratio 3:9:11,
    and a total of 816 animals, prove that there are 105 dogs. -/
theorem pet_shop_dogs (total : ℕ) (h_total : total = 816) :
  let ratio_sum := 3 + 9 + 11
  let part_size := total / ratio_sum
  let dogs := 3 * part_size
  dogs = 105 := by
  sorry

#check pet_shop_dogs

end NUMINAMATH_CALUDE_pet_shop_dogs_l3387_338730


namespace NUMINAMATH_CALUDE_math_homework_pages_l3387_338746

theorem math_homework_pages (total_problems : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) 
  (h1 : total_problems = 30)
  (h2 : reading_pages = 4)
  (h3 : problems_per_page = 3) :
  total_problems - reading_pages * problems_per_page = 6 * problems_per_page :=
by sorry

end NUMINAMATH_CALUDE_math_homework_pages_l3387_338746


namespace NUMINAMATH_CALUDE_unique_root_in_interval_l3387_338728

open Complex

theorem unique_root_in_interval : ∃! x : ℝ, 0 ≤ x ∧ x < 2 * π ∧
  2 + exp (I * x) - 2 * exp (2 * I * x) + exp (3 * I * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_in_interval_l3387_338728


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3387_338779

theorem floor_negative_seven_fourths : ⌊(-7 : ℤ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3387_338779


namespace NUMINAMATH_CALUDE_unique_coprime_squares_l3387_338774

theorem unique_coprime_squares (m n : ℕ+) : 
  m.val.Coprime n.val ∧ 
  ∃ x y : ℕ, (m.val^2 - 5*n.val^2 = x^2) ∧ (m.val^2 + 5*n.val^2 = y^2) →
  m.val = 41 ∧ n.val = 12 :=
by sorry

end NUMINAMATH_CALUDE_unique_coprime_squares_l3387_338774


namespace NUMINAMATH_CALUDE_power_two_1000_mod_13_l3387_338793

theorem power_two_1000_mod_13 : 2^1000 % 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_two_1000_mod_13_l3387_338793


namespace NUMINAMATH_CALUDE_omega_range_l3387_338748

theorem omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∃ a b : ℝ, π ≤ a ∧ a < b ∧ b ≤ 2*π ∧ Real.sin (ω * a) + Real.sin (ω * b) = 2) →
  (ω ∈ Set.Icc (9/4) (5/2) ∪ Set.Ici (13/4)) :=
by sorry

end NUMINAMATH_CALUDE_omega_range_l3387_338748


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l3387_338752

theorem circle_area_with_diameter_10 (π : ℝ) :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l3387_338752


namespace NUMINAMATH_CALUDE_lineup_count_is_636_l3387_338785

/-- Represents a basketball team with specified number of players and positions -/
structure BasketballTeam where
  total_players : ℕ
  forwards : ℕ
  guards : ℕ
  versatile_players : ℕ
  lineup_forwards : ℕ
  lineup_guards : ℕ

/-- Calculates the number of different lineups for a given basketball team -/
def count_lineups (team : BasketballTeam) : ℕ :=
  sorry

/-- Theorem stating that the number of different lineups is 636 for the given team configuration -/
theorem lineup_count_is_636 : 
  let team : BasketballTeam := {
    total_players := 12,
    forwards := 6,
    guards := 4,
    versatile_players := 2,
    lineup_forwards := 3,
    lineup_guards := 2
  }
  count_lineups team = 636 := by sorry

end NUMINAMATH_CALUDE_lineup_count_is_636_l3387_338785


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3387_338717

def P : Set ℝ := {x | x^2 + x - 6 = 0}
def S (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (S a ⊆ P) ↔ (a = 0 ∨ a = 1/3 ∨ a = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3387_338717


namespace NUMINAMATH_CALUDE_min_translation_for_symmetry_l3387_338769

/-- The minimum positive translation that makes the graph of a sine function symmetric about the origin -/
theorem min_translation_for_symmetry (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = 2 * Real.sin (x + π / 3 - φ)) →
  φ > 0 →
  (∀ x, f x = -f (-x)) →
  φ ≥ π / 3 ∧ 
  ∃ (φ_min : ℝ), φ_min = π / 3 ∧ 
    ∀ (ψ : ℝ), ψ > 0 → 
      (∀ x, 2 * Real.sin (x + π / 3 - ψ) = -(2 * Real.sin (-x + π / 3 - ψ))) → 
      ψ ≥ φ_min :=
by sorry


end NUMINAMATH_CALUDE_min_translation_for_symmetry_l3387_338769


namespace NUMINAMATH_CALUDE_small_planks_count_l3387_338736

/-- Represents the number of planks used in building a house wall. -/
structure Planks where
  total : ℕ
  large : ℕ
  small : ℕ

/-- Theorem stating that given 29 total planks and 12 large planks, the number of small planks is 17. -/
theorem small_planks_count (p : Planks) (h1 : p.total = 29) (h2 : p.large = 12) : p.small = 17 := by
  sorry

end NUMINAMATH_CALUDE_small_planks_count_l3387_338736


namespace NUMINAMATH_CALUDE_min_value_expression_l3387_338771

theorem min_value_expression (a b : ℝ) (h : a^2 * b^2 + 2*a*b + 2*a + 1 = 0) :
  ∃ (x : ℝ), x = a*b*(a*b+2) + (b+1)^2 + 2*a ∧ 
  (∀ (y : ℝ), y = a*b*(a*b+2) + (b+1)^2 + 2*a → x ≤ y) ∧
  x = -3/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3387_338771


namespace NUMINAMATH_CALUDE_area_covered_by_two_squares_l3387_338735

/-- The area covered by two congruent squares with side length 12, where one vertex of one square coincides with a vertex of the other square -/
theorem area_covered_by_two_squares (side_length : ℝ) (h1 : side_length = 12) :
  let square_area := side_length ^ 2
  let total_area := 2 * square_area - square_area
  total_area = 144 := by sorry

end NUMINAMATH_CALUDE_area_covered_by_two_squares_l3387_338735


namespace NUMINAMATH_CALUDE_james_tv_watching_time_l3387_338799

/-- The duration of a Jeopardy episode in minutes -/
def jeopardy_duration : ℕ := 20

/-- The number of Jeopardy episodes watched -/
def jeopardy_episodes : ℕ := 2

/-- The duration of a Wheel of Fortune episode in minutes -/
def wheel_of_fortune_duration : ℕ := 2 * jeopardy_duration

/-- The number of Wheel of Fortune episodes watched -/
def wheel_of_fortune_episodes : ℕ := 2

/-- The total time spent watching TV in minutes -/
def total_time_minutes : ℕ := 
  jeopardy_duration * jeopardy_episodes + 
  wheel_of_fortune_duration * wheel_of_fortune_episodes

/-- Conversion factor from minutes to hours -/
def minutes_per_hour : ℕ := 60

theorem james_tv_watching_time : 
  total_time_minutes / minutes_per_hour = 2 := by sorry

end NUMINAMATH_CALUDE_james_tv_watching_time_l3387_338799


namespace NUMINAMATH_CALUDE_sugar_difference_l3387_338784

theorem sugar_difference (brown_sugar white_sugar : ℝ) 
  (h1 : brown_sugar = 0.62)
  (h2 : white_sugar = 0.25) :
  brown_sugar - white_sugar = 0.37 := by
  sorry

end NUMINAMATH_CALUDE_sugar_difference_l3387_338784


namespace NUMINAMATH_CALUDE_max_a_condition_1_range_a_condition_2_l3387_338734

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Theorem for the first part of the problem
theorem max_a_condition_1 :
  (∀ a : ℝ, (∀ x : ℝ, g x ≤ 5 → f a x ≤ 6) → a ≤ 1) ∧
  (∃ a : ℝ, a = 1 ∧ ∀ x : ℝ, g x ≤ 5 → f a x ≤ 6) :=
sorry

-- Theorem for the second part of the problem
theorem range_a_condition_2 :
  ∀ a : ℝ, (∀ x : ℝ, f a x + g x ≥ 3) → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_max_a_condition_1_range_a_condition_2_l3387_338734


namespace NUMINAMATH_CALUDE_all_propositions_false_l3387_338706

-- Define a plane α
variable (α : Set (ℝ × ℝ × ℝ))

-- Define lines in 3D space
def Line3D : Type := Set (ℝ × ℝ × ℝ)

-- Define the projection of a line onto a plane
def project (l : Line3D) (p : Set (ℝ × ℝ × ℝ)) : Line3D := sorry

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define parallel lines
def parallel (l1 l2 : Line3D) : Prop := sorry

-- Define intersecting lines
def intersect (l1 l2 : Line3D) : Prop := sorry

-- Define coincident lines
def coincide (l1 l2 : Line3D) : Prop := sorry

-- Define a line not on a plane
def not_on_plane (l : Line3D) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

theorem all_propositions_false (α : Set (ℝ × ℝ × ℝ)) :
  ∀ (m n : Line3D),
    not_on_plane m α → not_on_plane n α →
    (¬ (perpendicular (project m α) (project n α) → perpendicular m n)) ∧
    (¬ (perpendicular m n → perpendicular (project m α) (project n α))) ∧
    (¬ (intersect (project m α) (project n α) → intersect m n ∨ coincide m n)) ∧
    (¬ (parallel (project m α) (project n α) → parallel m n ∨ coincide m n)) :=
by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l3387_338706


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l3387_338765

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 20 / 100 →
  germination_rate2 = 35 / 100 →
  let total_seeds := seeds_plot1 + seeds_plot2
  let germinated_seeds1 := (seeds_plot1 : ℚ) * germination_rate1
  let germinated_seeds2 := (seeds_plot2 : ℚ) * germination_rate2
  let total_germinated := germinated_seeds1 + germinated_seeds2
  total_germinated / total_seeds = 26 / 100 := by
sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l3387_338765


namespace NUMINAMATH_CALUDE_trip_time_difference_l3387_338798

-- Define the given conditions
def speed : ℝ := 60
def distance1 : ℝ := 510
def distance2 : ℝ := 540

-- Define the theorem
theorem trip_time_difference : 
  (distance2 - distance1) / speed * 60 = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_l3387_338798


namespace NUMINAMATH_CALUDE_seed_germination_percentage_experiment_result_l3387_338741

/-- Calculates the percentage of total seeds germinated in an agricultural experiment. -/
theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) : ℚ :=
  let total_seeds := seeds_plot1 + seeds_plot2
  let germinated_seeds1 := (seeds_plot1 : ℚ) * germination_rate1
  let germinated_seeds2 := (seeds_plot2 : ℚ) * germination_rate2
  let total_germinated := germinated_seeds1 + germinated_seeds2
  (total_germinated / total_seeds) * 100

/-- The percentage of total seeds germinated in the given agricultural experiment. -/
theorem experiment_result : 
  seed_germination_percentage 500 200 (30/100) (50/100) = 250/700 * 100 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_experiment_result_l3387_338741


namespace NUMINAMATH_CALUDE_journey_speed_proof_l3387_338778

theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 672 ∧ total_time = 30 ∧ second_half_speed = 24 →
  ∃ first_half_speed : ℝ,
    first_half_speed = 21 ∧
    first_half_speed * (total_time / 2) + second_half_speed * (total_time / 2) = total_distance :=
by
  sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l3387_338778


namespace NUMINAMATH_CALUDE_balloon_count_correct_l3387_338754

/-- The number of red balloons Fred has -/
def fred_balloons : ℕ := 10

/-- The number of red balloons Sam has -/
def sam_balloons : ℕ := 46

/-- The number of red balloons Dan has -/
def dan_balloons : ℕ := 16

/-- The total number of red balloons -/
def total_balloons : ℕ := 72

theorem balloon_count_correct : fred_balloons + sam_balloons + dan_balloons = total_balloons := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_correct_l3387_338754


namespace NUMINAMATH_CALUDE_first_five_average_l3387_338787

theorem first_five_average (total_average : ℝ) (last_seven_average : ℝ) (fifth_result : ℝ) :
  total_average = 42 →
  last_seven_average = 52 →
  fifth_result = 147 →
  (5 * ((11 * total_average - (7 * last_seven_average - fifth_result)) / 5) = 245) ∧
  ((11 * total_average - (7 * last_seven_average - fifth_result)) / 5 = 49) :=
by
  sorry

end NUMINAMATH_CALUDE_first_five_average_l3387_338787


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l3387_338777

theorem number_exceeding_fraction (x : ℝ) : x = (3/7 + 0.8 * (3/7)) * x → x = (35/27) * x := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l3387_338777


namespace NUMINAMATH_CALUDE_simplify_expression_l3387_338760

theorem simplify_expression (a : ℝ) : 2*a*(2*a^2 + a) - a^2 = 4*a^3 + a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3387_338760


namespace NUMINAMATH_CALUDE_fine_arts_packaging_volume_l3387_338792

/-- The volume needed to package a fine arts collection given box dimensions, cost per box, and minimum total cost. -/
theorem fine_arts_packaging_volume 
  (box_length : ℝ) 
  (box_width : ℝ) 
  (box_height : ℝ) 
  (cost_per_box : ℝ) 
  (min_total_cost : ℝ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 12)
  (h4 : cost_per_box = 0.5)
  (h5 : min_total_cost = 225) :
  (min_total_cost / cost_per_box) * (box_length * box_width * box_height) = 2160000 := by
  sorry

#check fine_arts_packaging_volume

end NUMINAMATH_CALUDE_fine_arts_packaging_volume_l3387_338792


namespace NUMINAMATH_CALUDE_f_of_2_equals_4_l3387_338705

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 2

-- Theorem statement
theorem f_of_2_equals_4 : f 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_4_l3387_338705


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3387_338700

theorem quadratic_factorization (c d : ℕ) (hc : c > d) : 
  (∀ x, x^2 - 20*x + 91 = (x - c)*(x - d)) → 2*d - c = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3387_338700


namespace NUMINAMATH_CALUDE_restaurant_bill_rounding_l3387_338738

theorem restaurant_bill_rounding (people : ℕ) (bill : ℚ) : 
  people = 9 → 
  bill = 314.16 → 
  ∃ (rounded_total : ℚ), 
    rounded_total = (people : ℚ) * (⌈(bill / people) * 100⌉ / 100) ∧ 
    rounded_total = 314.19 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_rounding_l3387_338738


namespace NUMINAMATH_CALUDE_triangle_square_diagonal_l3387_338776

/-- Given a triangle with base 6 and height 4, the length of the diagonal of a square 
    with the same area as the triangle is √24. -/
theorem triangle_square_diagonal : 
  ∀ (triangle_base triangle_height : ℝ),
  triangle_base = 6 →
  triangle_height = 4 →
  ∃ (square_diagonal : ℝ),
    (1/2 * triangle_base * triangle_height) = square_diagonal^2 / 2 ∧
    square_diagonal = Real.sqrt 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_square_diagonal_l3387_338776


namespace NUMINAMATH_CALUDE_mammaad_arrangements_l3387_338788

theorem mammaad_arrangements : 
  let total_letters : ℕ := 7
  let m_count : ℕ := 3
  let a_count : ℕ := 3
  let d_count : ℕ := 1
  (total_letters.factorial) / (m_count.factorial * a_count.factorial * d_count.factorial) = 140 := by
  sorry

end NUMINAMATH_CALUDE_mammaad_arrangements_l3387_338788


namespace NUMINAMATH_CALUDE_problem_statement_l3387_338704

theorem problem_statement (x y z : ℝ) 
  (h1 : (1/x) + (2/y) + (3/z) = 0)
  (h2 : (1/x) - (6/y) - (5/z) = 0) :
  (x/y) + (y/z) + (z/x) = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3387_338704


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3387_338714

theorem unique_solution_condition (t : ℝ) : 
  (∃! x y z v : ℝ, x + y + z + v = 0 ∧ (x*y + y*z + z*v) + t*(x*z + x*v + y*v) = 0) ↔ 
  ((3 - Real.sqrt 5) / 2 < t ∧ t < (3 + Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3387_338714


namespace NUMINAMATH_CALUDE_max_runs_in_one_day_match_l3387_338772

/-- Represents the number of overs in a cricket one-day match -/
def overs : ℕ := 50

/-- Represents the number of legal deliveries per over -/
def deliveries_per_over : ℕ := 6

/-- Represents the maximum number of runs that can be scored on a single delivery -/
def max_runs_per_delivery : ℕ := 6

/-- Theorem stating the maximum number of runs a batsman can score in an ideal scenario -/
theorem max_runs_in_one_day_match :
  overs * deliveries_per_over * max_runs_per_delivery = 1800 := by
  sorry

end NUMINAMATH_CALUDE_max_runs_in_one_day_match_l3387_338772


namespace NUMINAMATH_CALUDE_divisor_of_number_minus_one_l3387_338766

theorem divisor_of_number_minus_one (n : ℕ) (h : n = 5026) : 5 ∣ (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_number_minus_one_l3387_338766
