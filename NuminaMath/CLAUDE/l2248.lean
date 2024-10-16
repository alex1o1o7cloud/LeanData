import Mathlib

namespace NUMINAMATH_CALUDE_fastest_growing_function_l2248_224829

/-- Proves that 0.001e^x grows faster than 1000ln(x), x^1000, and 1000⋅2^x as x approaches infinity -/
theorem fastest_growing_function :
  ∀ (ε : ℝ), ε > 0 → ∃ (X : ℝ), ∀ (x : ℝ), x > X →
    (0.001 * Real.exp x > 1000 * Real.log x) ∧
    (0.001 * Real.exp x > x^1000) ∧
    (0.001 * Real.exp x > 1000 * 2^x) :=
sorry

end NUMINAMATH_CALUDE_fastest_growing_function_l2248_224829


namespace NUMINAMATH_CALUDE_negative_number_identification_l2248_224899

theorem negative_number_identification :
  (|-2023| ≥ 0) ∧ 
  (Real.sqrt ((-2)^2) ≥ 0) ∧ 
  (0 ≥ 0) ∧ 
  (-3^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_number_identification_l2248_224899


namespace NUMINAMATH_CALUDE_four_digit_count_l2248_224865

/-- The count of four-digit numbers -/
def count_four_digit_numbers : ℕ := 9999 - 1000 + 1

/-- The first four-digit number -/
def first_four_digit : ℕ := 1000

/-- The last four-digit number -/
def last_four_digit : ℕ := 9999

theorem four_digit_count :
  count_four_digit_numbers = 9000 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_count_l2248_224865


namespace NUMINAMATH_CALUDE_admin_personnel_count_l2248_224875

/-- Represents the total number of employees in the unit -/
def total_employees : ℕ := 280

/-- Represents the sample size -/
def sample_size : ℕ := 56

/-- Represents the number of ordinary staff sampled -/
def ordinary_staff_sampled : ℕ := 49

/-- Calculates the number of administrative personnel -/
def admin_personnel : ℕ := total_employees - (total_employees * ordinary_staff_sampled / sample_size)

/-- Theorem stating that the number of administrative personnel is 35 -/
theorem admin_personnel_count : admin_personnel = 35 := by
  sorry

end NUMINAMATH_CALUDE_admin_personnel_count_l2248_224875


namespace NUMINAMATH_CALUDE_cricket_average_score_l2248_224867

theorem cricket_average_score (score1 score2 : ℝ) (n1 n2 : ℕ) (h1 : score1 = 27) (h2 : score2 = 32) (h3 : n1 = 2) (h4 : n2 = 3) :
  (score1 * n1 + score2 * n2) / (n1 + n2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_score_l2248_224867


namespace NUMINAMATH_CALUDE_dress_price_ratio_l2248_224853

theorem dress_price_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_ratio : ℝ := 2/3
  let cost : ℝ := cost_ratio * selling_price
  cost / marked_price = 1/2 := by
sorry

end NUMINAMATH_CALUDE_dress_price_ratio_l2248_224853


namespace NUMINAMATH_CALUDE_clothing_color_theorem_l2248_224864

-- Define the colors
inductive Color
| Red
| Blue

-- Define a structure for clothing
structure Clothing :=
  (tshirt : Color)
  (shorts : Color)

-- Define a function to check if two colors are different
def different_colors (c1 c2 : Color) : Prop :=
  c1 ≠ c2

-- Define the problem statement
theorem clothing_color_theorem 
  (alyna bohdan vika grysha : Clothing) : 
  (alyna.tshirt = Color.Red) →
  (bohdan.tshirt = Color.Red) →
  (different_colors alyna.shorts bohdan.shorts) →
  (different_colors vika.tshirt grysha.tshirt) →
  (vika.shorts = Color.Blue) →
  (grysha.shorts = Color.Blue) →
  (different_colors alyna.tshirt vika.tshirt) →
  (different_colors alyna.shorts vika.shorts) →
  (alyna = ⟨Color.Red, Color.Red⟩ ∧
   bohdan = ⟨Color.Red, Color.Blue⟩ ∧
   vika = ⟨Color.Blue, Color.Blue⟩ ∧
   grysha = ⟨Color.Red, Color.Blue⟩) :=
by sorry


end NUMINAMATH_CALUDE_clothing_color_theorem_l2248_224864


namespace NUMINAMATH_CALUDE_cube_root_minus_square_root_plus_power_l2248_224834

theorem cube_root_minus_square_root_plus_power : 
  ((-2 : ℝ)^3)^(1/3) - Real.sqrt 4 + (Real.sqrt 3)^0 = -3 := by sorry

end NUMINAMATH_CALUDE_cube_root_minus_square_root_plus_power_l2248_224834


namespace NUMINAMATH_CALUDE_min_distance_to_i_l2248_224806

-- Define the complex number z
variable (z : ℂ)

-- State the theorem
theorem min_distance_to_i (h : Complex.abs (z^2 - 1) = Complex.abs (z * (z - Complex.I))) :
  Complex.abs (z - Complex.I) ≥ (3 * Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_i_l2248_224806


namespace NUMINAMATH_CALUDE_village_population_percentage_l2248_224808

theorem village_population_percentage (total : ℕ) (part : ℕ) (h1 : total = 9000) (h2 : part = 8100) :
  (part : ℚ) / total * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_village_population_percentage_l2248_224808


namespace NUMINAMATH_CALUDE_x_plus_y_values_l2248_224876

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 2) (h3 : x > y) :
  x + y = 5 ∨ x + y = 1 :=
by sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l2248_224876


namespace NUMINAMATH_CALUDE_bakers_earnings_l2248_224862

/-- The baker's earnings problem -/
theorem bakers_earnings (cakes_sold : ℕ) (cake_price : ℕ) (pies_sold : ℕ) (pie_price : ℕ) 
  (h1 : cakes_sold = 453)
  (h2 : cake_price = 12)
  (h3 : pies_sold = 126)
  (h4 : pie_price = 7) :
  cakes_sold * cake_price + pies_sold * pie_price = 6318 := by
  sorry


end NUMINAMATH_CALUDE_bakers_earnings_l2248_224862


namespace NUMINAMATH_CALUDE_aquarium_feeding_ratio_l2248_224840

/-- The ratio of buckets fed to other sea animals to buckets fed to sharks -/
def ratio_other_to_sharks : ℚ := 5

theorem aquarium_feeding_ratio : 
  let sharks_buckets : ℕ := 4
  let dolphins_buckets : ℕ := sharks_buckets / 2
  let total_buckets : ℕ := 546
  let days : ℕ := 21
  
  ∃ (other_buckets : ℚ),
    other_buckets = ratio_other_to_sharks * sharks_buckets ∧
    total_buckets = (sharks_buckets + dolphins_buckets + other_buckets) * days :=
by sorry

end NUMINAMATH_CALUDE_aquarium_feeding_ratio_l2248_224840


namespace NUMINAMATH_CALUDE_forgotten_lawns_l2248_224855

/-- 
Given that:
- Roger earns $9 for each lawn he mows
- He had 14 lawns to mow
- He actually earned $54

Prove that the number of lawns Roger forgot to mow is equal to 14 minus the quotient of 54 and 9.
-/
theorem forgotten_lawns (earnings_per_lawn : ℕ) (total_lawns : ℕ) (actual_earnings : ℕ) :
  earnings_per_lawn = 9 →
  total_lawns = 14 →
  actual_earnings = 54 →
  total_lawns - (actual_earnings / earnings_per_lawn) = 8 :=
by sorry

end NUMINAMATH_CALUDE_forgotten_lawns_l2248_224855


namespace NUMINAMATH_CALUDE_cubic_difference_l2248_224880

theorem cubic_difference (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 135 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l2248_224880


namespace NUMINAMATH_CALUDE_grass_seed_cost_l2248_224890

/-- Represents the cost and weight of a bag of grass seed -/
structure BagInfo where
  weight : ℕ
  cost : ℚ

/-- Calculates the total cost of a given number of bags -/
def totalCost (bag : BagInfo) (count : ℕ) : ℚ :=
  bag.cost * count

/-- Calculates the total weight of a given number of bags -/
def totalWeight (bag : BagInfo) (count : ℕ) : ℕ :=
  bag.weight * count

theorem grass_seed_cost
  (bag5 : BagInfo)
  (bag10 : BagInfo)
  (bag25 : BagInfo)
  (h1 : bag5.weight = 5)
  (h2 : bag10.weight = 10)
  (h3 : bag10.cost = 20.43)
  (h4 : bag25.weight = 25)
  (h5 : bag25.cost = 32.25)
  (h6 : ∃ (c5 c10 c25 : ℕ), 
    65 ≤ totalWeight bag5 c5 + totalWeight bag10 c10 + totalWeight bag25 c25 ∧
    totalWeight bag5 c5 + totalWeight bag10 c10 + totalWeight bag25 c25 ≤ 80 ∧
    totalCost bag5 c5 + totalCost bag10 c10 + totalCost bag25 c25 = 98.75 ∧
    ∀ (d5 d10 d25 : ℕ),
      65 ≤ totalWeight bag5 d5 + totalWeight bag10 d10 + totalWeight bag25 d25 →
      totalWeight bag5 d5 + totalWeight bag10 d10 + totalWeight bag25 d25 ≤ 80 →
      totalCost bag5 d5 + totalCost bag10 d10 + totalCost bag25 d25 ≥ 98.75) :
  bag5.cost = 13.82 := by
sorry

end NUMINAMATH_CALUDE_grass_seed_cost_l2248_224890


namespace NUMINAMATH_CALUDE_sum_and_multiply_base8_l2248_224803

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 8 -/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Sums numbers from 1 to n in base 8 -/
def sumBase8 (n : ℕ) : ℕ := sorry

theorem sum_and_multiply_base8 :
  base10ToBase8 (3 * (sumBase8 (base8ToBase10 30))) = 1604 := by sorry

end NUMINAMATH_CALUDE_sum_and_multiply_base8_l2248_224803


namespace NUMINAMATH_CALUDE_commuter_distance_commuter_distance_is_12_sqrt_2_l2248_224883

/-- The distance from the starting point after a commuter drives 21 miles east, 
    15 miles south, 9 miles west, and 3 miles north. -/
theorem commuter_distance : ℝ :=
  let east : ℝ := 21
  let south : ℝ := 15
  let west : ℝ := 9
  let north : ℝ := 3
  let net_east_west : ℝ := east - west
  let net_south_north : ℝ := south - north
  Real.sqrt (net_east_west ^ 2 + net_south_north ^ 2)

/-- Proof that the commuter's distance from the starting point is 12√2 miles. -/
theorem commuter_distance_is_12_sqrt_2 : commuter_distance = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_commuter_distance_commuter_distance_is_12_sqrt_2_l2248_224883


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l2248_224809

theorem smallest_angle_in_triangle (angle1 angle2 y : ℝ) : 
  angle1 = 60 → 
  angle2 = 65 → 
  angle1 + angle2 + y = 180 → 
  min angle1 (min angle2 y) = 55 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l2248_224809


namespace NUMINAMATH_CALUDE_unique_two_digit_sum_reverse_prime_l2248_224868

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The main theorem -/
theorem unique_two_digit_sum_reverse_prime :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ isPrime (n + reverseDigits n) :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_sum_reverse_prime_l2248_224868


namespace NUMINAMATH_CALUDE_product_of_roots_undefined_expression_l2248_224826

theorem product_of_roots_undefined_expression : ∃ (x y : ℝ),
  x^2 + 4*x - 5 = 0 ∧ 
  y^2 + 4*y - 5 = 0 ∧ 
  x * y = -5 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_undefined_expression_l2248_224826


namespace NUMINAMATH_CALUDE_parallel_resistors_l2248_224838

/-- Given two resistors connected in parallel with resistances x and y,
    where the combined resistance r satisfies 1/r = 1/x + 1/y,
    prove that when x = 4 ohms and r = 2.4 ohms, y = 6 ohms. -/
theorem parallel_resistors (x y r : ℝ) 
  (hx : x = 4)
  (hr : r = 2.4)
  (h_combined : 1 / r = 1 / x + 1 / y) :
  y = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_resistors_l2248_224838


namespace NUMINAMATH_CALUDE_john_january_savings_l2248_224819

theorem john_january_savings :
  let base_income : ℝ := 2000
  let bonus_rate : ℝ := 0.15
  let transport_rate : ℝ := 0.05
  let rent : ℝ := 500
  let utilities : ℝ := 100
  let food : ℝ := 300
  let misc_rate : ℝ := 0.10

  let total_income : ℝ := base_income * (1 + bonus_rate)
  let transport_expense : ℝ := total_income * transport_rate
  let misc_expense : ℝ := total_income * misc_rate
  let total_expenses : ℝ := transport_expense + rent + utilities + food + misc_expense
  let savings : ℝ := total_income - total_expenses

  savings = 1055 := by sorry

end NUMINAMATH_CALUDE_john_january_savings_l2248_224819


namespace NUMINAMATH_CALUDE_remainder_conversion_l2248_224801

theorem remainder_conversion (N : ℕ) : 
  N % 72 = 68 → N % 24 = 20 := by
sorry

end NUMINAMATH_CALUDE_remainder_conversion_l2248_224801


namespace NUMINAMATH_CALUDE_not_divisible_by_67_l2248_224879

theorem not_divisible_by_67 (x y : ℕ) 
  (h1 : ¬(67 ∣ x))
  (h2 : ¬(67 ∣ y))
  (h3 : 67 ∣ (7*x + 32*y)) :
  ¬(67 ∣ (10*x + 17*y + 1)) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_67_l2248_224879


namespace NUMINAMATH_CALUDE_friends_hiking_distance_l2248_224873

-- Define the hiking scenario
structure HikingScenario where
  total_time : Real
  birgit_time_diff : Real
  birgit_time : Real
  birgit_distance : Real

-- Define the theorem
theorem friends_hiking_distance (h : HikingScenario) 
  (h_total_time : h.total_time = 3.5) 
  (h_birgit_time_diff : h.birgit_time_diff = 4) 
  (h_birgit_time : h.birgit_time = 48) 
  (h_birgit_distance : h.birgit_distance = 8) : 
  (h.total_time * 60) / (h.birgit_time / h.birgit_distance + h.birgit_time_diff) = 21 := by
  sorry


end NUMINAMATH_CALUDE_friends_hiking_distance_l2248_224873


namespace NUMINAMATH_CALUDE_spherical_point_equivalence_l2248_224814

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Checks if a SphericalPoint is in standard form -/
def isStandardForm (p : SphericalPoint) : Prop :=
  p.ρ > 0 ∧ 0 ≤ p.θ ∧ p.θ < 2 * Real.pi ∧ 0 ≤ p.φ ∧ p.φ ≤ Real.pi

/-- Converts a SphericalPoint to standard form -/
def toStandardForm (p : SphericalPoint) : SphericalPoint :=
  sorry

theorem spherical_point_equivalence :
  let p : SphericalPoint := ⟨4, Real.pi / 4, 9 * Real.pi / 5⟩
  let p_std : SphericalPoint := toStandardForm p
  p_std = ⟨4, 5 * Real.pi / 4, Real.pi / 5⟩ ∧ isStandardForm p_std :=
by
  sorry

end NUMINAMATH_CALUDE_spherical_point_equivalence_l2248_224814


namespace NUMINAMATH_CALUDE_smallest_debate_club_size_l2248_224805

/-- Represents the number of students in each grade --/
structure GradeCount where
  eighth : ℕ
  sixth : ℕ
  seventh : ℕ
  ninth : ℕ

/-- Checks if the given counts satisfy the ratio conditions --/
def satisfiesRatios (counts : GradeCount) : Prop :=
  7 * counts.sixth = 4 * counts.eighth ∧
  6 * counts.seventh = 5 * counts.eighth ∧
  9 * counts.ninth = 2 * counts.eighth

/-- Calculates the total number of students --/
def totalStudents (counts : GradeCount) : ℕ :=
  counts.eighth + counts.sixth + counts.seventh + counts.ninth

/-- Theorem stating that the smallest number of students satisfying the ratios is 331 --/
theorem smallest_debate_club_size :
  ∀ counts : GradeCount,
    satisfiesRatios counts →
    totalStudents counts ≥ 331 ∧
    ∃ counts' : GradeCount, satisfiesRatios counts' ∧ totalStudents counts' = 331 :=
by sorry

end NUMINAMATH_CALUDE_smallest_debate_club_size_l2248_224805


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l2248_224874

theorem smallest_k_for_inequality : ∃ k : ℕ, k = 8 ∧ 
  (∀ m : ℕ, 64^m > 4^22 → m ≥ k) ∧ 64^k > 4^22 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l2248_224874


namespace NUMINAMATH_CALUDE_equation_solution_l2248_224869

theorem equation_solution :
  ∃ x : ℚ, (3 * x - 17) / 4 = (x + 9) / 6 ∧ x = 69 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2248_224869


namespace NUMINAMATH_CALUDE_bridget_sarah_money_l2248_224860

/-- The amount of money Bridget and Sarah have together in dollars -/
def total_money (sarah_cents bridget_cents : ℕ) : ℚ :=
  (sarah_cents + bridget_cents : ℚ) / 100

theorem bridget_sarah_money :
  ∀ (sarah_cents : ℕ),
    sarah_cents = 125 →
    ∀ (bridget_cents : ℕ),
      bridget_cents = sarah_cents + 50 →
      total_money sarah_cents bridget_cents = 3 := by
sorry

end NUMINAMATH_CALUDE_bridget_sarah_money_l2248_224860


namespace NUMINAMATH_CALUDE_vector_AB_l2248_224810

-- Define the vector type
def Vector2D := ℝ × ℝ

-- Define the vector OA
def OA : Vector2D := (2, 8)

-- Define the vector OB
def OB : Vector2D := (-7, 2)

-- Define vector subtraction
def vectorSub (v1 v2 : Vector2D) : Vector2D :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Theorem statement
theorem vector_AB (OA OB : Vector2D) (h1 : OA = (2, 8)) (h2 : OB = (-7, 2)) :
  vectorSub OB OA = (-9, -6) := by
  sorry

end NUMINAMATH_CALUDE_vector_AB_l2248_224810


namespace NUMINAMATH_CALUDE_integer_decimal_parts_sum_l2248_224845

theorem integer_decimal_parts_sum (m n : ℝ) : 
  (∃ k : ℤ, 7 + Real.sqrt 13 = k + m ∧ k ≤ 7 + Real.sqrt 13 ∧ 7 + Real.sqrt 13 < k + 1) →
  (∃ j : ℤ, Real.sqrt 13 = j + n ∧ j ≤ Real.sqrt 13 ∧ Real.sqrt 13 < j + 1) →
  m + n = 7 + Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_integer_decimal_parts_sum_l2248_224845


namespace NUMINAMATH_CALUDE_negation_of_implication_negation_of_x_squared_positive_l2248_224882

theorem negation_of_implication (P Q : Prop) :
  ¬(P → Q) ↔ (P ∧ ¬Q) :=
by sorry

theorem negation_of_x_squared_positive :
  ¬(∀ x : ℝ, x > 0 → x^2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_negation_of_x_squared_positive_l2248_224882


namespace NUMINAMATH_CALUDE_natural_number_divisibility_l2248_224835

theorem natural_number_divisibility (a b : ℕ) : 
  (∃ k : ℕ, a = k * (b + 1)) → 
  (∃ m : ℕ, 43 = m * (a + b)) → 
  ((a = 22 ∧ b = 21) ∨ 
   (a = 33 ∧ b = 10) ∨ 
   (a = 40 ∧ b = 3) ∨ 
   (a = 42 ∧ b = 1)) := by
sorry

end NUMINAMATH_CALUDE_natural_number_divisibility_l2248_224835


namespace NUMINAMATH_CALUDE_scale_length_90_inches_l2248_224822

/-- Given a scale divided into equal parts, calculates the total length of the scale. -/
def scale_length (num_parts : ℕ) (part_length : ℕ) : ℕ :=
  num_parts * part_length

/-- Theorem stating that a scale with 5 parts of 18 inches each has a total length of 90 inches. -/
theorem scale_length_90_inches :
  scale_length 5 18 = 90 := by
  sorry

end NUMINAMATH_CALUDE_scale_length_90_inches_l2248_224822


namespace NUMINAMATH_CALUDE_lattice_point_bounds_l2248_224844

/-- The minimum number of points in ℤ^d such that any set of these points
    will contain n points whose centroid is a lattice point -/
def f (n d : ℕ) : ℕ :=
  sorry

theorem lattice_point_bounds (n d : ℕ) (hn : n > 0) (hd : d > 0) :
  (n - 1) * 2^d + 1 ≤ f n d ∧ f n d ≤ (n - 1) * n^d + 1 :=
by sorry

end NUMINAMATH_CALUDE_lattice_point_bounds_l2248_224844


namespace NUMINAMATH_CALUDE_dog_walking_distance_l2248_224802

theorem dog_walking_distance (total_weekly_miles : ℝ) (dog2_daily_miles : ℝ) :
  total_weekly_miles = 70 →
  dog2_daily_miles = 8 →
  ∃ dog1_daily_miles : ℝ, 
    dog1_daily_miles * 7 + dog2_daily_miles * 7 = total_weekly_miles ∧
    dog1_daily_miles = 2 := by
  sorry

end NUMINAMATH_CALUDE_dog_walking_distance_l2248_224802


namespace NUMINAMATH_CALUDE_trucks_per_lane_l2248_224851

theorem trucks_per_lane (lanes : ℕ) (total_vehicles : ℕ) : 
  lanes = 4 → 
  total_vehicles = 2160 → 
  (∃ trucks_per_lane : ℕ, 
    total_vehicles = lanes * trucks_per_lane + lanes * (2 * (lanes * trucks_per_lane)) ∧
    trucks_per_lane = 60) :=
by sorry

end NUMINAMATH_CALUDE_trucks_per_lane_l2248_224851


namespace NUMINAMATH_CALUDE_f_at_one_equals_neg_7007_l2248_224811

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 10
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 100*x + c

-- State the theorem
theorem f_at_one_equals_neg_7007 (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0 ∧
    f b c r₁ = 0 ∧ f b c r₂ = 0 ∧ f b c r₃ = 0) →
  f b c 1 = -7007 :=
by sorry


end NUMINAMATH_CALUDE_f_at_one_equals_neg_7007_l2248_224811


namespace NUMINAMATH_CALUDE_unique_integer_solution_l2248_224828

theorem unique_integer_solution (a b c : ℤ) : 
  a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l2248_224828


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2248_224821

theorem solution_set_of_inequality (x : ℝ) : 
  (3 * x + 1) * (1 - 2 * x) > 0 ↔ -1/3 < x ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2248_224821


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l2248_224872

theorem cousins_ages_sum : 
  ∀ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    0 < a ∧ a < 10 →
    0 < b ∧ b < 10 →
    0 < c ∧ c < 10 →
    0 < d ∧ d < 10 →
    (a * b = 40 ∨ a * c = 40 ∨ a * d = 40 ∨ b * c = 40 ∨ b * d = 40 ∨ c * d = 40) →
    (a * b = 36 ∨ a * c = 36 ∨ a * d = 36 ∨ b * c = 36 ∨ b * d = 36 ∨ c * d = 36) →
    a + b + c + d = 26 :=
by sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l2248_224872


namespace NUMINAMATH_CALUDE_salon_non_clients_l2248_224878

theorem salon_non_clients (manicure_cost : ℝ) (total_earnings : ℝ) (total_fingers : ℕ) (fingers_per_person : ℕ) :
  manicure_cost = 20 →
  total_earnings = 200 →
  total_fingers = 210 →
  fingers_per_person = 10 →
  (total_fingers / fingers_per_person : ℝ) - (total_earnings / manicure_cost) = 11 :=
by sorry

end NUMINAMATH_CALUDE_salon_non_clients_l2248_224878


namespace NUMINAMATH_CALUDE_fraction_equality_l2248_224824

theorem fraction_equality : (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2248_224824


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_thirty_one_is_max_max_consecutive_integers_sum_is_31_l2248_224884

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 31 ↔ n * (n + 1) ≤ 1000 := by
  sorry

theorem thirty_one_is_max : ∀ m : ℕ, m > 31 → m * (m + 1) > 1000 := by
  sorry

theorem max_consecutive_integers_sum_is_31 :
  (∃ n : ℕ, n * (n + 1) ≤ 1000 ∧ ∀ m : ℕ, m > n → m * (m + 1) > 1000) ∧
  (∀ n : ℕ, n * (n + 1) ≤ 1000 ∧ (∀ m : ℕ, m > n → m * (m + 1) > 1000) → n = 31) := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_thirty_one_is_max_max_consecutive_integers_sum_is_31_l2248_224884


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_four_l2248_224863

theorem sum_of_solutions_is_four :
  let f : ℝ → ℝ := λ N ↦ N * (N - 4) - 12
  ∃ N₁ N₂ : ℝ, (f N₁ = 0 ∧ f N₂ = 0) ∧ N₁ + N₂ = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_four_l2248_224863


namespace NUMINAMATH_CALUDE_gcd_problems_l2248_224816

theorem gcd_problems :
  (Nat.gcd 840 1764 = 84) ∧ (Nat.gcd 440 556 = 4) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l2248_224816


namespace NUMINAMATH_CALUDE_misread_number_correction_l2248_224841

theorem misread_number_correction (n : ℕ) (incorrect_avg correct_avg misread_value : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 14)
  (h3 : correct_avg = 15)
  (h4 : misread_value = 26) : 
  ∃ (actual_value : ℚ), 
    (n : ℚ) * correct_avg - (n : ℚ) * incorrect_avg = misread_value - actual_value ∧ 
    actual_value = 16 := by
  sorry

end NUMINAMATH_CALUDE_misread_number_correction_l2248_224841


namespace NUMINAMATH_CALUDE_parabola_focus_l2248_224815

/-- A parabola with equation y^2 = 2px (p > 0) and directrix x = -1 has its focus at (1, 0) -/
theorem parabola_focus (p : ℝ) (h : p > 0) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let directrix := {(x, y) : ℝ × ℝ | x = -1}
  let focus := (1, 0)
  (∀ (point : ℝ × ℝ), point ∈ parabola ↔ 
    Real.sqrt ((point.1 - focus.1)^2 + (point.2 - focus.2)^2) = 
    |point.1 - (-1)|) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l2248_224815


namespace NUMINAMATH_CALUDE_mother_bought_pencils_l2248_224820

def dozen : ℕ := 12

def initial_pencils : ℕ := 17

def total_pencils : ℕ := 2 * dozen

theorem mother_bought_pencils : total_pencils - initial_pencils = 7 := by
  sorry

end NUMINAMATH_CALUDE_mother_bought_pencils_l2248_224820


namespace NUMINAMATH_CALUDE_onions_left_on_scale_l2248_224804

/-- Represents the problem of calculating the number of onions left on a scale. -/
def OnionProblem (initial_count : ℕ) (total_weight : ℝ) (removed_count : ℕ) (remaining_avg_weight : ℝ) (removed_avg_weight : ℝ) : Prop :=
  let remaining_count := initial_count - removed_count
  let total_weight_grams := total_weight * 1000
  let remaining_weight := remaining_count * remaining_avg_weight
  let removed_weight := removed_count * removed_avg_weight
  (remaining_weight + removed_weight = total_weight_grams) ∧
  (remaining_count = 35)

/-- Theorem stating that given the problem conditions, 35 onions are left on the scale. -/
theorem onions_left_on_scale :
  OnionProblem 40 7.68 5 190 206 :=
by
  sorry

end NUMINAMATH_CALUDE_onions_left_on_scale_l2248_224804


namespace NUMINAMATH_CALUDE_find_n_l2248_224850

theorem find_n : ∃ n : ℝ, ∀ x : ℝ, (x - 2) * (x + 1) = x^2 + n*x - 2 → n = -1 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l2248_224850


namespace NUMINAMATH_CALUDE_tan_sum_product_identity_l2248_224823

open Real

theorem tan_sum_product_identity (α β γ : ℝ) : 
  0 < α ∧ α < π/2 ∧ 
  0 < β ∧ β < π/2 ∧ 
  0 < γ ∧ γ < π/2 ∧ 
  α + β + γ = π/2 ∧ 
  (∀ k : ℤ, α ≠ k * π + π/2) ∧
  (∀ k : ℤ, β ≠ k * π + π/2) ∧
  (∀ k : ℤ, γ ≠ k * π + π/2) →
  tan α * tan β + tan β * tan γ + tan γ * tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_product_identity_l2248_224823


namespace NUMINAMATH_CALUDE_adjacent_to_five_sum_seven_l2248_224870

/-- Represents the five corners of a pentagon -/
inductive Corner
  | a | b | c | d | e

/-- A configuration of numbers in the pentagon corners -/
def Configuration := Corner → Fin 5

/-- Two corners are adjacent if they share an edge in the pentagon -/
def adjacent (x y : Corner) : Prop :=
  match x, y with
  | Corner.a, Corner.b | Corner.b, Corner.a => true
  | Corner.b, Corner.c | Corner.c, Corner.b => true
  | Corner.c, Corner.d | Corner.d, Corner.c => true
  | Corner.d, Corner.e | Corner.e, Corner.d => true
  | Corner.e, Corner.a | Corner.a, Corner.e => true
  | _, _ => false

/-- A valid configuration satisfies the adjacency condition -/
def valid_configuration (config : Configuration) : Prop :=
  ∀ x y : Corner, adjacent x y → |config x - config y| > 1

/-- The main theorem -/
theorem adjacent_to_five_sum_seven (config : Configuration) 
  (h_valid : valid_configuration config) 
  (h_five : ∃ x : Corner, config x = 5) :
  ∃ y z : Corner, 
    adjacent x y ∧ adjacent x z ∧ y ≠ z ∧ 
    config y + config z = 7 ∧ config x = 5 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_to_five_sum_seven_l2248_224870


namespace NUMINAMATH_CALUDE_mixed_doubles_handshakes_l2248_224847

/-- Represents a mixed doubles tennis tournament -/
structure MixedDoublesTournament where
  teams : Nat
  players_per_team : Nat
  opposite_gender_players : Nat

/-- Calculates the number of handshakes in a mixed doubles tournament -/
def handshakes (tournament : MixedDoublesTournament) : Nat :=
  tournament.teams * (tournament.opposite_gender_players - 1)

/-- Theorem: In a mixed doubles tennis tournament with 4 teams, 
    where each player shakes hands once with every player of the 
    opposite gender except their own partner, the total number 
    of handshakes is 12. -/
theorem mixed_doubles_handshakes :
  let tournament : MixedDoublesTournament := {
    teams := 4,
    players_per_team := 2,
    opposite_gender_players := 4
  }
  handshakes tournament = 12 := by
  sorry

end NUMINAMATH_CALUDE_mixed_doubles_handshakes_l2248_224847


namespace NUMINAMATH_CALUDE_repeating_decimal_28_l2248_224807

/-- The repeating decimal 0.2828... is equal to 28/99 -/
theorem repeating_decimal_28 : ∃ (x : ℚ), x = 28 / 99 ∧ x = 0 + (28 / 100) * (1 / (1 - 1 / 100)) :=
by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_28_l2248_224807


namespace NUMINAMATH_CALUDE_pen_cost_is_six_l2248_224898

/-- The cost of the pen Joshua wants to buy -/
def pen_cost : ℚ := 6

/-- The amount of money Joshua has in his pocket -/
def pocket_money : ℚ := 5

/-- The amount of money Joshua borrowed from his neighbor -/
def borrowed_money : ℚ := 68 / 100

/-- The additional amount Joshua needs to buy the pen -/
def additional_money_needed : ℚ := 32 / 100

/-- Theorem stating that the cost of the pen is $6.00 -/
theorem pen_cost_is_six :
  pen_cost = pocket_money + borrowed_money + additional_money_needed :=
by sorry

end NUMINAMATH_CALUDE_pen_cost_is_six_l2248_224898


namespace NUMINAMATH_CALUDE_min_value_constrained_l2248_224894

theorem min_value_constrained (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (∃ m : ℝ, ∀ x y : ℝ, a * x^2 + b * y^2 = 1 → c * x + d * y^2 ≥ m) ∧
  (∀ ε > 0, ∃ x y : ℝ, a * x^2 + b * y^2 = 1 ∧ c * x + d * y^2 < -c / Real.sqrt a + ε) :=
sorry

end NUMINAMATH_CALUDE_min_value_constrained_l2248_224894


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2248_224858

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α + Real.cos α) = -1) : 
  Real.tan α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2248_224858


namespace NUMINAMATH_CALUDE_inclination_angle_range_l2248_224885

theorem inclination_angle_range (α : ℝ) (θ : ℝ) : 
  (∃ x y : ℝ, x * Real.sin α - y + 1 = 0) →
  0 ≤ θ ∧ θ < π →
  (θ ∈ Set.Icc 0 (π/4) ∪ Set.Ico (3*π/4) π) ↔ 
  (∃ x y : ℝ, x * Real.sin α - y + 1 = 0 ∧ θ = Real.arctan (Real.sin α)) :=
by sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l2248_224885


namespace NUMINAMATH_CALUDE_truck_speed_truck_speed_proof_l2248_224888

/-- Proves that a truck traveling 600 meters in 60 seconds has a speed of 36 kilometers per hour -/
theorem truck_speed : ℝ → Prop :=
  fun (speed : ℝ) =>
    let distance : ℝ := 600  -- meters
    let time : ℝ := 60       -- seconds
    let meters_per_km : ℝ := 1000
    let seconds_per_hour : ℝ := 3600
    speed = (distance / time) * (seconds_per_hour / meters_per_km) → speed = 36

/-- The actual speed of the truck in km/h -/
def actual_speed : ℝ := 36

theorem truck_speed_proof : truck_speed actual_speed :=
  sorry

end NUMINAMATH_CALUDE_truck_speed_truck_speed_proof_l2248_224888


namespace NUMINAMATH_CALUDE_house_size_ratio_l2248_224861

/-- The size of Kennedy's house in square feet -/
def kennedy_house_size : ℝ := 10000

/-- The size of Benedict's house in square feet -/
def benedict_house_size : ℝ := 2350

/-- The additional size in square feet added to the ratio of house sizes -/
def additional_size : ℝ := 600

/-- Theorem stating that the ratio of (Kennedy's house size - additional size) to Benedict's house size is 4 -/
theorem house_size_ratio : 
  (kennedy_house_size - additional_size) / benedict_house_size = 4 := by
  sorry

end NUMINAMATH_CALUDE_house_size_ratio_l2248_224861


namespace NUMINAMATH_CALUDE_value_of_c_l2248_224877

theorem value_of_c (a c : ℕ) (h1 : a = 105) (h2 : a^5 = 3^3 * 5^2 * 7^2 * 11^2 * 13 * c) : c = 385875 := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l2248_224877


namespace NUMINAMATH_CALUDE_largest_divisor_power_l2248_224893

-- Define pow function
def pow (n : ℕ) : ℕ :=
  sorry

-- Define X
def X : ℕ := 2310

-- Define the product of pow(n) from 2 to 5400
def product : ℕ :=
  sorry

-- Theorem statement
theorem largest_divisor_power : 
  (∃ m : ℕ, X^m ∣ product ∧ ∀ k > m, ¬(X^k ∣ product)) → 
  (∃ m : ℕ, X^m ∣ product ∧ ∀ k > m, ¬(X^k ∣ product) ∧ m = 534) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_power_l2248_224893


namespace NUMINAMATH_CALUDE_expression_evaluation_l2248_224848

theorem expression_evaluation :
  let x : ℚ := 1/2
  6 * x^2 - (2*x + 1) * (3*x - 2) + (x + 3) * (x - 3) = -25/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2248_224848


namespace NUMINAMATH_CALUDE_invalid_triangle_1_invalid_triangle_2_l2248_224825

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the property that triangle angles sum to 180 degrees
def valid_triangle (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

-- Theorem: A triangle with angles 90°, 60°, and 60° cannot exist
theorem invalid_triangle_1 : 
  ¬ ∃ (t : Triangle), t.angle1 = 90 ∧ t.angle2 = 60 ∧ t.angle3 = 60 ∧ valid_triangle t :=
sorry

-- Theorem: A triangle with angles 90°, 50°, and 50° cannot exist
theorem invalid_triangle_2 :
  ¬ ∃ (t : Triangle), t.angle1 = 90 ∧ t.angle2 = 50 ∧ t.angle3 = 50 ∧ valid_triangle t :=
sorry

end NUMINAMATH_CALUDE_invalid_triangle_1_invalid_triangle_2_l2248_224825


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l2248_224839

/-- Calculates the weight loss given initial and current weights -/
def weight_loss (initial_weight current_weight : ℕ) : ℕ :=
  initial_weight - current_weight

/-- Proves that Jessie's weight loss is 7 kilograms -/
theorem jessie_weight_loss : weight_loss 74 67 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l2248_224839


namespace NUMINAMATH_CALUDE_a_value_range_l2248_224832

/-- Proposition p: For any x, ax^2 + ax + 1 > 0 always holds true -/
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Proposition q: The equation x^2 - x + a = 0 has real roots -/
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

/-- The range of values for a satisfying the given conditions -/
def a_range (a : ℝ) : Prop := a < 0 ∨ (1/4 < a ∧ a < 4)

theorem a_value_range :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → a_range a :=
by sorry

end NUMINAMATH_CALUDE_a_value_range_l2248_224832


namespace NUMINAMATH_CALUDE_trees_left_unwatered_l2248_224896

theorem trees_left_unwatered :
  let total_trees : ℕ := 29
  let boys_watering : ℕ := 9
  let trees_watered_per_boy : List ℕ := [2, 3, 1, 3, 2, 4, 3, 2, 5]
  let total_watered : ℕ := trees_watered_per_boy.sum
  total_trees - total_watered = 4 := by
sorry

end NUMINAMATH_CALUDE_trees_left_unwatered_l2248_224896


namespace NUMINAMATH_CALUDE_eating_contest_l2248_224842

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight : ℕ)
  (noah_burgers jacob_pies mason_hotdogs : ℕ) :
  hot_dog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  jacob_pies = noah_burgers - 3 →
  mason_hotdogs = 3 * jacob_pies →
  noah_burgers = 8 →
  mason_hotdogs * hot_dog_weight = 30 := by
sorry

end NUMINAMATH_CALUDE_eating_contest_l2248_224842


namespace NUMINAMATH_CALUDE_initial_jasmine_percentage_l2248_224818

/-- Proof of initial jasmine percentage in a solution --/
theorem initial_jasmine_percentage
  (initial_volume : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (final_jasmine_percentage : ℝ)
  (h1 : initial_volume = 100)
  (h2 : added_jasmine = 5)
  (h3 : added_water = 10)
  (h4 : final_jasmine_percentage = 8.695652173913043)
  : (100 * (initial_volume * (final_jasmine_percentage / 100) - added_jasmine) / initial_volume) = 5 := by
  sorry

end NUMINAMATH_CALUDE_initial_jasmine_percentage_l2248_224818


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l2248_224857

/-- Represents the four grade levels --/
inductive Grade
| Freshman
| Sophomore
| Junior
| Senior

/-- Represents a student --/
structure Student where
  grade : Grade
  isTwin : Bool

/-- Represents the arrangement of students in a car --/
def CarArrangement := List Student

/-- Total number of students --/
def totalStudents : Nat := 8

/-- Number of students per grade --/
def studentsPerGrade : Nat := 2

/-- Number of students per car --/
def studentsPerCar : Nat := 4

/-- The twin sisters are freshmen --/
def twinSisters : List Student := [
  { grade := Grade.Freshman, isTwin := true },
  { grade := Grade.Freshman, isTwin := true }
]

/-- Checks if an arrangement has exactly two students from the same grade --/
def hasTwoSameGrade (arrangement : CarArrangement) : Bool :=
  sorry

/-- Counts the number of valid arrangements for Car A --/
def countValidArrangements : Nat :=
  sorry

theorem valid_arrangements_count :
  countValidArrangements = 24 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l2248_224857


namespace NUMINAMATH_CALUDE_circle_line_intersection_l2248_224859

/-- Given a circle and a line, prove the value of m when the chord length is 4 -/
theorem circle_line_intersection (m : ℝ) : 
  (∃ x y : ℝ, (x + 1)^2 + (y - 1)^2 = 2 - m ∧ x + y + 2 = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + 1)^2 + (y₁ - 1)^2 = 2 - m ∧
    x₁ + y₁ + 2 = 0 ∧
    (x₂ + 1)^2 + (y₂ - 1)^2 = 2 - m ∧
    x₂ + y₂ + 2 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) →
  m = -4 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l2248_224859


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l2248_224800

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_decreasing : ∀ x y, 0 ≤ x → x < y → f y < f x)

-- Define the solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f (2*x - 1) > f (1/3)}

-- Theorem statement
theorem solution_set_is_open_interval :
  solution_set f = Set.Ioo (1/3) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l2248_224800


namespace NUMINAMATH_CALUDE_jony_start_time_l2248_224892

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculate the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

/-- Represents Jony's walk -/
structure Walk where
  startBlock : Nat
  turnaroundBlock : Nat
  endBlock : Nat
  blockLength : Nat
  speed : Nat
  endTime : Time

theorem jony_start_time (w : Walk) (h1 : w.startBlock = 10)
    (h2 : w.turnaroundBlock = 90) (h3 : w.endBlock = 70)
    (h4 : w.blockLength = 40) (h5 : w.speed = 100)
    (h6 : w.endTime = ⟨7, 40⟩) :
    timeDifference w.endTime ⟨7, 0⟩ =
      ((w.turnaroundBlock - w.startBlock + w.turnaroundBlock - w.endBlock) * w.blockLength) / w.speed :=
  sorry

end NUMINAMATH_CALUDE_jony_start_time_l2248_224892


namespace NUMINAMATH_CALUDE_percentage_of_games_sold_l2248_224854

theorem percentage_of_games_sold (initial_cost : ℝ) (sold_price : ℝ) : 
  initial_cost = 200 → 
  sold_price = 240 → 
  (sold_price / (initial_cost * 3)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_games_sold_l2248_224854


namespace NUMINAMATH_CALUDE_yuri_puppies_l2248_224813

def puppies_week1 : ℕ := 20

def puppies_week2 : ℕ := (2 * puppies_week1) / 5

def puppies_week3 : ℕ := 2 * puppies_week2

def puppies_week4 : ℕ := puppies_week1 + 10

def total_puppies : ℕ := puppies_week1 + puppies_week2 + puppies_week3 + puppies_week4

theorem yuri_puppies : total_puppies = 74 := by
  sorry

end NUMINAMATH_CALUDE_yuri_puppies_l2248_224813


namespace NUMINAMATH_CALUDE_linda_total_coins_l2248_224891

/-- Represents the number of coins Linda has -/
structure Coins where
  dimes : Nat
  quarters : Nat
  nickels : Nat

/-- Calculates the total number of coins -/
def totalCoins (c : Coins) : Nat :=
  c.dimes + c.quarters + c.nickels

/-- Linda's initial coins -/
def initialCoins : Coins :=
  { dimes := 2, quarters := 6, nickels := 5 }

/-- Coins given by Linda's mother -/
def givenCoins (initial : Coins) : Coins :=
  { dimes := 2, quarters := 10, nickels := 2 * initial.nickels }

/-- Linda's final coins after receiving coins from her mother -/
def finalCoins (initial : Coins) : Coins :=
  { dimes := initial.dimes + (givenCoins initial).dimes,
    quarters := initial.quarters + (givenCoins initial).quarters,
    nickels := initial.nickels + (givenCoins initial).nickels }

theorem linda_total_coins :
  totalCoins (finalCoins initialCoins) = 35 := by
  sorry

end NUMINAMATH_CALUDE_linda_total_coins_l2248_224891


namespace NUMINAMATH_CALUDE_work_completion_time_l2248_224827

/-- 
Given that Paul completes a piece of work in 80 days and Rose completes the same work in 120 days,
prove that they will complete the work together in 48 days.
-/
theorem work_completion_time 
  (paul_time : ℕ) 
  (rose_time : ℕ) 
  (h_paul : paul_time = 80) 
  (h_rose : rose_time = 120) : 
  (paul_time * rose_time) / (paul_time + rose_time) = 48 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2248_224827


namespace NUMINAMATH_CALUDE_circle_center_coordinate_difference_l2248_224887

/-- Given two points as endpoints of a circle's diameter, 
    calculate the absolute difference between the x and y coordinates of the circle's center -/
theorem circle_center_coordinate_difference (x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : x₁ = 8 ∧ y₁ = -7)
  (h2 : x₂ = -4 ∧ y₂ = 5) : 
  |((x₁ + x₂) / 2) - ((y₁ + y₂) / 2)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_difference_l2248_224887


namespace NUMINAMATH_CALUDE_bert_sandwiches_remaining_l2248_224886

def sandwiches_remaining (initial : ℕ) (first_day : ℕ) (second_day : ℕ) : ℕ :=
  initial - (first_day + second_day)

theorem bert_sandwiches_remaining :
  let initial := 12
  let first_day := initial / 2
  let second_day := first_day - 2
  sandwiches_remaining initial first_day second_day = 2 := by
sorry

end NUMINAMATH_CALUDE_bert_sandwiches_remaining_l2248_224886


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2248_224895

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | x < -2 ∨ x > 2}

-- State the theorem
theorem intersection_complement_equality :
  A ∩ (Set.compl B) = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2248_224895


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2248_224849

theorem sum_of_fractions : 
  (1 / 15 : ℚ) + (2 / 15 : ℚ) + (3 / 15 : ℚ) + (4 / 15 : ℚ) + 
  (5 / 15 : ℚ) + (6 / 15 : ℚ) + (7 / 15 : ℚ) + (8 / 15 : ℚ) + 
  (30 / 15 : ℚ) = 4 + (2 / 5 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2248_224849


namespace NUMINAMATH_CALUDE_opposite_absolute_values_sum_l2248_224833

theorem opposite_absolute_values_sum (a b : ℝ) : 
  |a - 2| = -|b + 3| → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_absolute_values_sum_l2248_224833


namespace NUMINAMATH_CALUDE_negation_equivalence_l2248_224871

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 3*x + 3 < 0) ↔ (∀ x : ℝ, x^2 - 3*x + 3 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2248_224871


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2248_224856

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2248_224856


namespace NUMINAMATH_CALUDE_average_weight_decrease_l2248_224897

/-- Given a group of people and a new person joining, calculate the decrease in average weight -/
theorem average_weight_decrease (initial_count : ℕ) (initial_average : ℝ) (new_person_weight : ℝ) : 
  initial_count = 20 →
  initial_average = 55 →
  new_person_weight = 50 →
  let total_weight := initial_count * initial_average
  let new_total_weight := total_weight + new_person_weight
  let new_count := initial_count + 1
  let new_average := new_total_weight / new_count
  abs (initial_average - new_average - 0.24) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l2248_224897


namespace NUMINAMATH_CALUDE_sin_45_degrees_l2248_224846

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l2248_224846


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2248_224866

/-- Expresses 0.3̄56 as a rational number -/
theorem repeating_decimal_to_fraction : 
  ∃ (n d : ℕ), d ≠ 0 ∧ (0.3 + (56 : ℚ) / 99 / 10) = (n : ℚ) / d := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2248_224866


namespace NUMINAMATH_CALUDE_correct_sample_l2248_224812

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Represents a sample of student numbers --/
def Sample := List Nat

/-- Checks if a number is a valid student number (between 1 and 50) --/
def isValidStudentNumber (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 50

/-- Selects a sample of distinct student numbers from the random number table --/
def selectSample (table : RandomNumberTable) (startRow : Nat) (startCol : Nat) (sampleSize : Nat) : Sample :=
  sorry

/-- The specific random number table given in the problem --/
def givenTable : RandomNumberTable :=
  [[03, 47, 43, 73, 86, 36, 96, 47, 36, 61, 46, 98, 63, 71, 62, 33, 26, 16, 80],
   [45, 60, 11, 14, 10, 95, 97, 74, 24, 67, 62, 42, 81, 14, 57, 20, 42, 53],
   [32, 37, 32, 27, 07, 36, 07, 51, 24, 51, 79, 89, 73, 16, 76, 62, 27, 66],
   [56, 50, 26, 71, 07, 32, 90, 79, 78, 53, 13, 55, 38, 58, 59, 88, 97, 54],
   [14, 10, 12, 56, 85, 99, 26, 96, 96, 68, 27, 31, 05, 03, 72, 93, 15, 57],
   [12, 10, 14, 21, 88, 26, 49, 81, 76, 55, 59, 56, 35, 64, 38, 54, 82, 46],
   [22, 31, 62, 43, 09, 90, 06, 18, 44, 32, 53, 23, 83, 01, 30, 30]]

theorem correct_sample :
  selectSample givenTable 3 6 5 = [22, 2, 10, 29, 7] :=
sorry

end NUMINAMATH_CALUDE_correct_sample_l2248_224812


namespace NUMINAMATH_CALUDE_f_max_min_implies_a_range_l2248_224852

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + a + 6

/-- Theorem: If f has both a maximum and a minimum, then a < -3 or a > 6 -/
theorem f_max_min_implies_a_range (a : ℝ) : 
  (∃ (x_max x_min : ℝ), ∀ x, f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_implies_a_range_l2248_224852


namespace NUMINAMATH_CALUDE_total_beanie_babies_l2248_224817

theorem total_beanie_babies (lori_beanie_babies sydney_beanie_babies : ℕ) :
  lori_beanie_babies = 300 →
  lori_beanie_babies = 15 * sydney_beanie_babies →
  lori_beanie_babies + sydney_beanie_babies = 320 := by
  sorry

end NUMINAMATH_CALUDE_total_beanie_babies_l2248_224817


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l2248_224837

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem smallest_number_with_conditions : ℕ := by
  let n : ℕ := 2979942
  have h1 : tens_digit n = 4 := by sorry
  have h2 : units_digit n = 2 := by sorry
  have h3 : sum_of_digits n = 42 := by sorry
  have h4 : n % 42 = 0 := by sorry
  have h5 : ∀ m : ℕ, m < n →
    ¬(tens_digit m = 4 ∧ units_digit m = 2 ∧ sum_of_digits m = 42 ∧ m % 42 = 0) := by sorry
  exact n

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l2248_224837


namespace NUMINAMATH_CALUDE_student_selection_l2248_224889

theorem student_selection (total : ℕ) (singers : ℕ) (dancers : ℕ) (both : ℕ) :
  total = 6 ∧ singers = 3 ∧ dancers = 2 ∧ both = 1 →
  Nat.choose singers 2 * dancers = 6 :=
by sorry

end NUMINAMATH_CALUDE_student_selection_l2248_224889


namespace NUMINAMATH_CALUDE_village_population_percentage_l2248_224843

theorem village_population_percentage : 
  let total_population : ℕ := 25600
  let part_population : ℕ := 23040
  (part_population : ℚ) / total_population * 100 = 90 := by sorry

end NUMINAMATH_CALUDE_village_population_percentage_l2248_224843


namespace NUMINAMATH_CALUDE_existence_of_special_integer_l2248_224836

theorem existence_of_special_integer :
  ∃ (A : ℕ), 
    (∃ (n : ℕ), A = n * (n + 1) * (n + 2)) ∧
    (∃ (k : ℕ), (A / 10^k) % 10^99 = 10^99 - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_integer_l2248_224836


namespace NUMINAMATH_CALUDE_wall_width_calculation_l2248_224830

/-- The width of a wall given a string length and a relation to that length -/
def wall_width (string_length_m : ℕ) (string_length_cm : ℕ) : ℕ :=
  let string_length_total_cm := string_length_m * 100 + string_length_cm
  5 * string_length_total_cm + 80

theorem wall_width_calculation :
  wall_width 1 70 = 930 := by sorry

end NUMINAMATH_CALUDE_wall_width_calculation_l2248_224830


namespace NUMINAMATH_CALUDE_c_monthly_income_l2248_224881

/-- The monthly income ratio between A and B -/
def income_ratio : ℚ := 5 / 2

/-- The percentage increase of B's income over C's income -/
def income_increase_percentage : ℚ := 12 / 100

/-- A's annual income in rupees -/
def a_annual_income : ℕ := 470400

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- Theorem stating C's monthly income -/
theorem c_monthly_income :
  let a_monthly_income : ℚ := a_annual_income / months_per_year
  let b_monthly_income : ℚ := a_monthly_income / income_ratio
  let c_monthly_income : ℚ := b_monthly_income / (1 + income_increase_percentage)
  c_monthly_income = 14000 := by sorry

end NUMINAMATH_CALUDE_c_monthly_income_l2248_224881


namespace NUMINAMATH_CALUDE_inequality_proof_l2248_224831

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2248_224831
