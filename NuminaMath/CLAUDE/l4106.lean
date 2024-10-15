import Mathlib

namespace NUMINAMATH_CALUDE_calculate_overall_profit_l4106_410644

/-- Calculate the overall profit from selling two items with given purchase prices and profit/loss percentages -/
theorem calculate_overall_profit
  (grinder_price mobile_price : ℕ)
  (grinder_loss_percent mobile_profit_percent : ℚ)
  (h1 : grinder_price = 15000)
  (h2 : mobile_price = 10000)
  (h3 : grinder_loss_percent = 4 / 100)
  (h4 : mobile_profit_percent = 10 / 100)
  : ↑grinder_price * (1 - grinder_loss_percent) + 
    ↑mobile_price * (1 + mobile_profit_percent) - 
    ↑(grinder_price + mobile_price) = 400 := by
  sorry


end NUMINAMATH_CALUDE_calculate_overall_profit_l4106_410644


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_specific_hyperbola_focal_length_l4106_410659

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is 2√(a² + b²) -/
theorem hyperbola_focal_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let focal_length := 2 * Real.sqrt (a^2 + b^2)
  focal_length = 2 * Real.sqrt 41 :=
by
  sorry

/-- The focal length of the hyperbola x²/16 - y²/25 = 1 is 2√41 -/
theorem specific_hyperbola_focal_length :
  let focal_length := 2 * Real.sqrt (16 + 25)
  focal_length = 2 * Real.sqrt 41 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_specific_hyperbola_focal_length_l4106_410659


namespace NUMINAMATH_CALUDE_complex_power_500_l4106_410668

theorem complex_power_500 : ((1 + 2 * Complex.I) / (1 - 2 * Complex.I)) ^ 500 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_500_l4106_410668


namespace NUMINAMATH_CALUDE_intersection_when_a_is_half_intersection_empty_iff_l4106_410627

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem 1: When a = 1/2, A ∩ B = {x | 0 < x < 1}
theorem intersection_when_a_is_half : 
  A (1/2) ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: A ∩ B = ∅ if and only if a ≤ -1/2 or a ≥ 2
theorem intersection_empty_iff : 
  ∀ a : ℝ, A a ∩ B = ∅ ↔ a ≤ -1/2 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_half_intersection_empty_iff_l4106_410627


namespace NUMINAMATH_CALUDE_count_even_numbers_between_300_and_600_l4106_410606

theorem count_even_numbers_between_300_and_600 :
  (Finset.filter (fun n => n % 2 = 0 ∧ 300 < n ∧ n < 600) (Finset.range 600)).card = 149 := by
  sorry

end NUMINAMATH_CALUDE_count_even_numbers_between_300_and_600_l4106_410606


namespace NUMINAMATH_CALUDE_maggies_work_hours_l4106_410612

/-- Maggie's work hours problem -/
theorem maggies_work_hours 
  (office_rate : ℝ) 
  (tractor_rate : ℝ) 
  (total_income : ℝ) 
  (h1 : office_rate = 10)
  (h2 : tractor_rate = 12)
  (h3 : total_income = 416) : 
  ∃ (tractor_hours : ℝ),
    tractor_hours = 13 ∧ 
    office_rate * (2 * tractor_hours) + tractor_rate * tractor_hours = total_income :=
by sorry

end NUMINAMATH_CALUDE_maggies_work_hours_l4106_410612


namespace NUMINAMATH_CALUDE_integral_f_minus_x_equals_five_sixths_l4106_410691

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x

-- State the theorem
theorem integral_f_minus_x_equals_five_sixths :
  (∀ x, deriv f x = 2 * x + 1) →
  ∫ x in (1)..(2), f (-x) = 5/6 := by sorry

end NUMINAMATH_CALUDE_integral_f_minus_x_equals_five_sixths_l4106_410691


namespace NUMINAMATH_CALUDE_inequality_proof_l4106_410679

theorem inequality_proof (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 2 * x + y ≤ Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4106_410679


namespace NUMINAMATH_CALUDE_log_equation_implies_ratio_l4106_410680

theorem log_equation_implies_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log (x - y) + Real.log (x + 2*y) = Real.log 2 + Real.log x + Real.log y) : 
  x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_implies_ratio_l4106_410680


namespace NUMINAMATH_CALUDE_length_of_CE_l4106_410610

/-- Given a plot ABCD with specific measurements, prove the length of CE -/
theorem length_of_CE (AF ED AE : ℝ) (area_ABCD : ℝ) :
  AF = 30 ∧ ED = 50 ∧ AE = 120 ∧ area_ABCD = 7200 →
  ∃ CE : ℝ, CE = 138 ∧
    area_ABCD = (1/2 * AE * ED) + (1/2 * (AF + CE) * ED) := by
  sorry

end NUMINAMATH_CALUDE_length_of_CE_l4106_410610


namespace NUMINAMATH_CALUDE_horner_method_example_l4106_410672

def f (x : ℝ) : ℝ := 6*x^5 + 5*x^4 - 4*x^3 + 3*x^2 - 2*x + 1

theorem horner_method_example : f 2 = 249 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_example_l4106_410672


namespace NUMINAMATH_CALUDE_product_equals_three_l4106_410670

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1 / 3

/-- The product of 0.333... and 9 --/
def product : ℚ := repeating_third * 9

theorem product_equals_three : product = 3 := by sorry

end NUMINAMATH_CALUDE_product_equals_three_l4106_410670


namespace NUMINAMATH_CALUDE_increasing_function_parameter_range_l4106_410613

/-- Given that f(x) = x^3 + ax + 1/x is an increasing function on (1/2, +∞),
    prove that a ∈ [13/4, +∞) -/
theorem increasing_function_parameter_range
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : ∀ x > 1/2, f x = x^3 + a*x + 1/x)
  (h2 : StrictMono f) :
  a ∈ Set.Ici (13/4) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_parameter_range_l4106_410613


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l4106_410648

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  q > 0 →
  a 3 + a 4 = a 5 →
  (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l4106_410648


namespace NUMINAMATH_CALUDE_min_disks_needed_l4106_410693

/-- Represents the capacity of a disk in MB -/
def diskCapacity : ℚ := 2.88

/-- Represents the sizes of files in MB -/
def fileSizes : List ℚ := [1.2, 0.9, 0.6, 0.3]

/-- Represents the quantities of files for each size -/
def fileQuantities : List ℕ := [5, 10, 8, 7]

/-- Calculates the total size of all files -/
def totalFileSize : ℚ := (List.zip fileSizes fileQuantities).foldl (λ acc (size, quantity) => acc + size * quantity) 0

/-- Theorem stating the minimum number of disks needed -/
theorem min_disks_needed : 
  ∃ (arrangement : List (List ℚ)), 
    (∀ disk ∈ arrangement, disk.sum ≤ diskCapacity) ∧ 
    (arrangement.map (List.length)).sum = (fileQuantities.sum) ∧
    arrangement.length = 14 :=
sorry

end NUMINAMATH_CALUDE_min_disks_needed_l4106_410693


namespace NUMINAMATH_CALUDE_vector_angle_theorem_l4106_410628

-- Define a type for 3D vectors
def Vector3D := ℝ × ℝ × ℝ

-- Define a function to calculate the angle between two vectors
noncomputable def angle (v1 v2 : Vector3D) : ℝ := sorry

-- Define a predicate for non-zero vectors
def nonzero (v : Vector3D) : Prop := v ≠ (0, 0, 0)

theorem vector_angle_theorem (vectors : Fin 30 → Vector3D) 
  (h : ∀ i, nonzero (vectors i)) : 
  ∃ i j, i ≠ j ∧ angle (vectors i) (vectors j) < Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_vector_angle_theorem_l4106_410628


namespace NUMINAMATH_CALUDE_grazing_area_expansion_l4106_410620

/-- Given a circular grazing area with an initial radius of 9 meters,
    if the area is increased by 1408 square meters,
    the new radius will be 23 meters. -/
theorem grazing_area_expansion (π : ℝ) (h : π > 0) :
  let r₁ : ℝ := 9
  let additional_area : ℝ := 1408
  let r₂ : ℝ := Real.sqrt (r₁^2 + additional_area / π)
  r₂ = 23 := by sorry

end NUMINAMATH_CALUDE_grazing_area_expansion_l4106_410620


namespace NUMINAMATH_CALUDE_luke_fish_fillets_l4106_410682

/-- Calculates the number of fillets per fish given the total number of fish caught and total fillets obtained. -/
def filletsPerFish (fishPerDay : ℕ) (days : ℕ) (totalFillets : ℕ) : ℚ :=
  totalFillets / (fishPerDay * days)

/-- Proves that the number of fillets per fish is 2 given the problem conditions. -/
theorem luke_fish_fillets : filletsPerFish 2 30 120 = 2 := by
  sorry

end NUMINAMATH_CALUDE_luke_fish_fillets_l4106_410682


namespace NUMINAMATH_CALUDE_ad_transmission_cost_l4106_410626

/-- The cost of transmitting advertisements during a race -/
theorem ad_transmission_cost
  (num_ads : ℕ)
  (ad_duration : ℕ)
  (cost_per_minute : ℕ)
  (h1 : num_ads = 5)
  (h2 : ad_duration = 3)
  (h3 : cost_per_minute = 4000) :
  num_ads * ad_duration * cost_per_minute = 60000 :=
by sorry

end NUMINAMATH_CALUDE_ad_transmission_cost_l4106_410626


namespace NUMINAMATH_CALUDE_probability_club_then_heart_l4106_410678

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of clubs in a standard deck
def num_clubs : ℕ := 13

-- Define the number of hearts in a standard deck
def num_hearts : ℕ := 13

-- Theorem statement
theorem probability_club_then_heart :
  (num_clubs : ℚ) / total_cards * num_hearts / (total_cards - 1) = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_probability_club_then_heart_l4106_410678


namespace NUMINAMATH_CALUDE_special_polygon_properties_l4106_410653

/-- A polygon where the sum of interior angles is 1/4 more than the sum of exterior angles -/
structure SpecialPolygon where
  n : ℕ  -- number of sides
  h : (n - 2) * 180 = 360 + (1/4) * 360

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem special_polygon_properties (p : SpecialPolygon) :
  p.n = 12 ∧ num_diagonals p.n = 54 := by
  sorry

#check special_polygon_properties

end NUMINAMATH_CALUDE_special_polygon_properties_l4106_410653


namespace NUMINAMATH_CALUDE_folded_paper_distance_l4106_410642

theorem folded_paper_distance (area : ℝ) (h_area : area = 12) : ℝ :=
  let side_length := Real.sqrt area
  let folded_side_length := Real.sqrt (area / 2)
  let distance := Real.sqrt (2 * folded_side_length ^ 2)
  
  have h_distance : distance = 2 * Real.sqrt 6 := by sorry
  
  distance

end NUMINAMATH_CALUDE_folded_paper_distance_l4106_410642


namespace NUMINAMATH_CALUDE_equation_solution_l4106_410632

theorem equation_solution :
  ∃ (x : ℝ), x > 0 ∧ 4 * Real.sqrt (9 + x) + 4 * Real.sqrt (9 - x) = 10 * Real.sqrt 3 ∧
  x = Real.sqrt 80.859375 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4106_410632


namespace NUMINAMATH_CALUDE_multiple_with_binary_digits_l4106_410600

theorem multiple_with_binary_digits (n : ℤ) : ∃ k : ℤ,
  (∃ m : ℤ, k = n * m) ∧ 
  (∃ d : ℕ, d ≤ n ∧ k < 10^d) ∧
  (∀ i : ℕ, i < n → (k / 10^i) % 10 = 0 ∨ (k / 10^i) % 10 = 1) :=
sorry

end NUMINAMATH_CALUDE_multiple_with_binary_digits_l4106_410600


namespace NUMINAMATH_CALUDE_range_of_c_l4106_410640

theorem range_of_c (a b c : ℝ) 
  (ha : 6 < a ∧ a < 10) 
  (hb : a / 2 ≤ b ∧ b ≤ 2 * a) 
  (hc : c = a + b) : 
  9 < c ∧ c < 30 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l4106_410640


namespace NUMINAMATH_CALUDE_uniform_random_transform_l4106_410645

/-- A uniform random number on an interval -/
structure UniformRandom (a b : ℝ) where
  value : ℝ
  in_range : a ≤ value ∧ value ≤ b

/-- Theorem: If b₁ is a uniform random number on [0,1] and b = (b₁ - 0.5) * 6,
    then b is a uniform random number on [-3,3] -/
theorem uniform_random_transform (b₁ : UniformRandom 0 1) :
  let b := (b₁.value - 0.5) * 6
  ∃ (b' : UniformRandom (-3) 3), b'.value = b := by
  sorry

end NUMINAMATH_CALUDE_uniform_random_transform_l4106_410645


namespace NUMINAMATH_CALUDE_fish_corn_equivalence_l4106_410663

theorem fish_corn_equivalence :
  ∀ (fish honey corn : ℚ),
  (5 * fish = 3 * honey) →
  (honey = 6 * corn) →
  (fish = 3.6 * corn) :=
by sorry

end NUMINAMATH_CALUDE_fish_corn_equivalence_l4106_410663


namespace NUMINAMATH_CALUDE_siblings_weekly_water_consumption_l4106_410676

/-- The amount of water consumed by three siblings in a week -/
def water_consumption (theo_daily : ℕ) (mason_daily : ℕ) (roxy_daily : ℕ) (days_in_week : ℕ) : ℕ :=
  (theo_daily + mason_daily + roxy_daily) * days_in_week

/-- Theorem stating that the siblings drink 168 cups of water in a week -/
theorem siblings_weekly_water_consumption :
  water_consumption 8 7 9 7 = 168 := by
  sorry

end NUMINAMATH_CALUDE_siblings_weekly_water_consumption_l4106_410676


namespace NUMINAMATH_CALUDE_decimal_6_to_binary_l4106_410636

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_6_to_binary :
  decimal_to_binary 6 = [1, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_decimal_6_to_binary_l4106_410636


namespace NUMINAMATH_CALUDE_defeat_points_zero_l4106_410656

/-- Represents the point system and match results for a football competition. -/
structure FootballCompetition where
  victoryPoints : ℕ := 3
  drawPoints : ℕ := 1
  defeatPoints : ℕ
  totalMatches : ℕ := 20
  pointsAfter5Games : ℕ := 14
  minVictoriesRemaining : ℕ := 6
  finalPointTarget : ℕ := 40

/-- Theorem stating that the points for a defeat must be zero under the given conditions. -/
theorem defeat_points_zero (fc : FootballCompetition) : fc.defeatPoints = 0 := by
  sorry

#check defeat_points_zero

end NUMINAMATH_CALUDE_defeat_points_zero_l4106_410656


namespace NUMINAMATH_CALUDE_inequality_relation_l4106_410623

theorem inequality_relation (a b : ℝ) : 
  ¬(∀ a b : ℝ, a > b → 1/a < 1/b) ∧ ¬(∀ a b : ℝ, 1/a < 1/b → a > b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relation_l4106_410623


namespace NUMINAMATH_CALUDE_coin_identification_l4106_410637

/-- Represents the type of a coin -/
inductive CoinType
| Genuine
| Counterfeit

/-- Represents the result of weighing two groups of coins -/
inductive WeighResult
| Even
| Odd

/-- Function to determine the coin type based on the weighing result -/
def determineCoinType (result : WeighResult) : CoinType :=
  match result with
  | WeighResult.Even => CoinType.Genuine
  | WeighResult.Odd => CoinType.Counterfeit

theorem coin_identification
  (total_coins : Nat)
  (counterfeit_coins : Nat)
  (weight_difference : Nat)
  (h1 : total_coins = 101)
  (h2 : counterfeit_coins = 50)
  (h3 : weight_difference = 1)
  : ∀ (specified_coin : CoinType) (weigh_result : WeighResult),
    determineCoinType weigh_result = specified_coin :=
  sorry

end NUMINAMATH_CALUDE_coin_identification_l4106_410637


namespace NUMINAMATH_CALUDE_arithmetic_mean_two_digit_multiples_of_8_l4106_410688

/-- The first two-digit multiple of 8 -/
def first_multiple : Nat := 16

/-- The last two-digit multiple of 8 -/
def last_multiple : Nat := 96

/-- The common difference between consecutive multiples of 8 -/
def common_difference : Nat := 8

/-- The number of two-digit multiples of 8 -/
def num_multiples : Nat := (last_multiple - first_multiple) / common_difference + 1

/-- The arithmetic mean of all positive two-digit multiples of 8 is 56 -/
theorem arithmetic_mean_two_digit_multiples_of_8 :
  (first_multiple + last_multiple) * num_multiples / (2 * num_multiples) = 56 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_mean_two_digit_multiples_of_8_l4106_410688


namespace NUMINAMATH_CALUDE_line_parabola_circle_intersection_l4106_410631

/-- A line intersecting a parabola and a circle with specific conditions -/
theorem line_parabola_circle_intersection
  (k m : ℝ)
  (l : Set (ℝ × ℝ))
  (A B C D : ℝ × ℝ)
  (h_line : l = {(x, y) | y = k * x + m})
  (h_parabola : A ∈ l ∧ B ∈ l ∧ A.1^2 = 2 * A.2 ∧ B.1^2 = 2 * B.2)
  (h_midpoint : (A.1 + B.1) / 2 = 1)
  (h_circle : C ∈ l ∧ D ∈ l ∧ C.1^2 + C.2^2 = 12 ∧ D.1^2 + D.2^2 = 12)
  (h_equal_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2) :
  k = 1 ∧ m = 2 := by sorry

end NUMINAMATH_CALUDE_line_parabola_circle_intersection_l4106_410631


namespace NUMINAMATH_CALUDE_composition_inverse_implies_value_l4106_410660

-- Define the functions
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem composition_inverse_implies_value (a b : ℝ) :
  (∀ x, h a b x = x - 9) →
  2 * a - 3 * b = 22 := by
  sorry

end NUMINAMATH_CALUDE_composition_inverse_implies_value_l4106_410660


namespace NUMINAMATH_CALUDE_log_101600_value_l4106_410624

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_101600_value (h : log 102 = 0.3010) : log 101600 = 2.3010 := by
  sorry

end NUMINAMATH_CALUDE_log_101600_value_l4106_410624


namespace NUMINAMATH_CALUDE_circle_area_above_line_l4106_410616

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 16*y + 48 = 0

-- Define the line equation
def line_equation (y : ℝ) : Prop := y = 4

-- Theorem statement
theorem circle_area_above_line :
  ∃ (A : ℝ), 
    (∀ x y : ℝ, circle_equation x y → 
      (y > 4 → (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ A})) ∧
    A = 24 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circle_area_above_line_l4106_410616


namespace NUMINAMATH_CALUDE_parallel_linear_function_through_point_l4106_410658

-- Define a linear function
def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

-- State the theorem
theorem parallel_linear_function_through_point :
  ∀ (k b : ℝ),
  -- The linear function is parallel to y = 2x + 1
  k = 2 →
  -- The linear function passes through the point (-1, 1)
  linear_function k b (-1) = 1 →
  -- The linear function is equal to y = 2x + 3
  linear_function k b = linear_function 2 3 := by
sorry


end NUMINAMATH_CALUDE_parallel_linear_function_through_point_l4106_410658


namespace NUMINAMATH_CALUDE_f_maximum_l4106_410638

/-- The quadratic function f(x) = -2x^2 + 8x - 6 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

/-- The point where the maximum occurs -/
def x_max : ℝ := 2

theorem f_maximum :
  ∀ x : ℝ, f x ≤ f x_max :=
by sorry

end NUMINAMATH_CALUDE_f_maximum_l4106_410638


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l4106_410652

theorem sin_2alpha_value (α : Real) (h : Real.sin α - Real.cos α = 4/3) : 
  Real.sin (2 * α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l4106_410652


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l4106_410666

theorem smallest_positive_integer_with_remainders : ∃ x : ℕ, 
  (x > 0) ∧ 
  (x % 3 = 2) ∧ 
  (x % 4 = 3) ∧ 
  (x % 5 = 4) ∧ 
  (∀ y : ℕ, y > 0 → (y % 3 = 2) → (y % 4 = 3) → (y % 5 = 4) → y ≥ x) ∧
  x = 59 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l4106_410666


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l4106_410690

/-- The number of vertices in a pentadecagon -/
def n : ℕ := 15

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The number of sides in a pentadecagon, which is equal to the number of triangles 
    that have a side coinciding with a side of the pentadecagon -/
def excluded_triangles : ℕ := n

theorem pentadecagon_triangles : 
  (Nat.choose n k) - excluded_triangles = 440 :=
sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l4106_410690


namespace NUMINAMATH_CALUDE_two_trucks_meeting_problem_l4106_410697

/-- The problem of two trucks meeting under different conditions -/
theorem two_trucks_meeting_problem 
  (t : ℝ) -- Time of meeting in normal conditions
  (s : ℝ) -- Length of the route AB
  (v1 v2 : ℝ) -- Speeds of trucks from A and B respectively
  (h1 : t = 8 + 40/60) -- Meeting time is 8 hours 40 minutes
  (h2 : v1 * t = s - 62/5) -- Distance traveled by first truck in normal conditions
  (h3 : v2 * t = 62/5) -- Distance traveled by second truck in normal conditions
  (h4 : v1 * (t - 1/12) = 62/5) -- Distance traveled by first truck in modified conditions
  (h5 : v2 * (t + 1/8) = s - 62/5) -- Distance traveled by second truck in modified conditions
  : v1 = 38.4 ∧ v2 = 25.6 ∧ s = 16 := by
  sorry


end NUMINAMATH_CALUDE_two_trucks_meeting_problem_l4106_410697


namespace NUMINAMATH_CALUDE_tiles_on_floor_l4106_410641

/-- Represents a rectangular floor covered with square tiles. -/
structure TiledFloor where
  width : ℕ
  length : ℕ
  diagonal_tiles : ℕ

/-- Calculates the total number of tiles on the floor. -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.width * floor.length

/-- Theorem: For a rectangular floor with length twice the width and 25 tiles along the diagonal,
    the total number of tiles is 242. -/
theorem tiles_on_floor (floor : TiledFloor) 
    (h1 : floor.length = 2 * floor.width)
    (h2 : floor.diagonal_tiles = 25) :
    total_tiles floor = 242 := by
  sorry

#eval total_tiles { width := 11, length := 22, diagonal_tiles := 25 }

end NUMINAMATH_CALUDE_tiles_on_floor_l4106_410641


namespace NUMINAMATH_CALUDE_output_for_15_l4106_410657

def function_machine (input : ℤ) : ℤ :=
  let step1 := input * 3
  if step1 > 25 then step1 - 7 else step1 + 10

theorem output_for_15 : function_machine 15 = 38 := by
  sorry

end NUMINAMATH_CALUDE_output_for_15_l4106_410657


namespace NUMINAMATH_CALUDE_equidistant_point_distance_l4106_410604

-- Define the equilateral triangle DEF
def triangle_DEF : Set (ℝ × ℝ × ℝ) := sorry

-- Define the side length of triangle DEF
def side_length : ℝ := 300

-- Define points X and Y
def X : ℝ × ℝ × ℝ := sorry
def Y : ℝ × ℝ × ℝ := sorry

-- Define the property that X and Y are equidistant from vertices of DEF
def equidistant_X (X : ℝ × ℝ × ℝ) : Prop := sorry
def equidistant_Y (Y : ℝ × ℝ × ℝ) : Prop := sorry

-- Define the 90° dihedral angle between planes XDE and YDE
def dihedral_angle_90 (X Y : ℝ × ℝ × ℝ) : Prop := sorry

-- Define point R
def R : ℝ × ℝ × ℝ := sorry

-- Define the distance r
def r : ℝ := sorry

-- Define the property that R is equidistant from D, E, F, X, and Y
def equidistant_R (R : ℝ × ℝ × ℝ) : Prop := sorry

theorem equidistant_point_distance :
  equidistant_X X →
  equidistant_Y Y →
  dihedral_angle_90 X Y →
  equidistant_R R →
  r = 50 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_equidistant_point_distance_l4106_410604


namespace NUMINAMATH_CALUDE_sector_arc_length_l4106_410698

/-- Given a sector with central angle 1 radian and radius 5 cm, the arc length is 5 cm. -/
theorem sector_arc_length (θ : Real) (r : Real) (l : Real) : 
  θ = 1 → r = 5 → l = r * θ → l = 5 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l4106_410698


namespace NUMINAMATH_CALUDE_stamp_collection_value_l4106_410685

/-- Given a collection of stamps with equal individual value, 
    calculate the total value of the collection. -/
theorem stamp_collection_value 
  (total_stamps : ℕ) 
  (sample_stamps : ℕ) 
  (sample_value : ℝ) 
  (h1 : total_stamps = 30)
  (h2 : sample_stamps = 10)
  (h3 : sample_value = 45) :
  (total_stamps : ℝ) * (sample_value / sample_stamps) = 135 :=
by sorry

end NUMINAMATH_CALUDE_stamp_collection_value_l4106_410685


namespace NUMINAMATH_CALUDE_b_inequalities_l4106_410630

theorem b_inequalities (a : ℝ) (h : a ∈ Set.Icc 0 1) :
  let b := a^3 + 1 / (1 + a)
  (b ≥ 1 - a + a^2) ∧ (3/4 < b ∧ b ≤ 3/2) := by
  sorry

end NUMINAMATH_CALUDE_b_inequalities_l4106_410630


namespace NUMINAMATH_CALUDE_units_digit_difference_l4106_410684

def is_positive_even_integer (p : ℕ) : Prop := p > 0 ∧ p % 2 = 0

def has_positive_units_digit (p : ℕ) : Prop := p % 10 > 0

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_difference (p : ℕ) 
  (h1 : is_positive_even_integer p) 
  (h2 : has_positive_units_digit p) 
  (h3 : units_digit (p + 5) = 1) : 
  units_digit (p^3) - units_digit (p^2) = 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_difference_l4106_410684


namespace NUMINAMATH_CALUDE_min_value_expression_l4106_410686

theorem min_value_expression (x y : ℝ) :
  Real.sqrt (4 + y^2) + Real.sqrt (x^2 + y^2 - 4*x - 4*y + 8) + Real.sqrt (x^2 - 8*x + 17) ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l4106_410686


namespace NUMINAMATH_CALUDE_small_apple_cost_is_correct_l4106_410677

/-- The cost of a small apple -/
def small_apple_cost : ℚ := 1.5

/-- The cost of a medium apple -/
def medium_apple_cost : ℚ := 2

/-- The cost of a big apple -/
def big_apple_cost : ℚ := 3

/-- The number of small and medium apples bought -/
def small_medium_apples : ℕ := 6

/-- The number of big apples bought -/
def big_apples : ℕ := 8

/-- The total cost of all apples bought -/
def total_cost : ℚ := 45

/-- Theorem stating that the cost of each small apple is $1.50 -/
theorem small_apple_cost_is_correct : 
  small_apple_cost * small_medium_apples + 
  medium_apple_cost * small_medium_apples + 
  big_apple_cost * big_apples = total_cost := by
sorry

end NUMINAMATH_CALUDE_small_apple_cost_is_correct_l4106_410677


namespace NUMINAMATH_CALUDE_parallel_condition_l4106_410661

/-- Two lines are parallel if and only if their slopes are equal and they have different y-intercepts -/
def are_parallel (m : ℝ) : Prop :=
  ((-1 : ℝ) / (1 + m) = -m / 2) ∧ 
  ((2 - m) / (1 + m) ≠ -4)

/-- Line l₁: x + (1+m)y = 2-m -/
def line_l1 (m : ℝ) (x y : ℝ) : Prop :=
  x + (1 + m) * y = 2 - m

/-- Line l₂: 2mx + 4y = -16 -/
def line_l2 (m : ℝ) (x y : ℝ) : Prop :=
  2 * m * x + 4 * y = -16

theorem parallel_condition :
  ∀ m : ℝ, (m = 1) ↔ are_parallel m :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l4106_410661


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l4106_410667

theorem subtraction_of_decimals : 5.18 - 3.45 = 1.73 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l4106_410667


namespace NUMINAMATH_CALUDE_constant_distance_l4106_410689

/-- Represents an ellipse centered at the origin with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h : 0 < b ∧ b < a
  h_e : e = Real.sqrt 2 / 2
  h_eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_intersect : ∃ (x y : ℝ), x^2 / E.a^2 + y^2 / E.b^2 = 1 ∧ y = k * x + m

/-- The theorem to be proved -/
theorem constant_distance (E : Ellipse) (l : IntersectingLine E) :
  ∃ (P Q : ℝ × ℝ) (d : ℝ),
    P ∈ {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1} ∧
    Q ∈ {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1} ∧
    P.1 * Q.1 + P.2 * Q.2 = 0 ∧
    d = Real.sqrt 6 / 3 ∧
    d = abs m / Real.sqrt (l.k^2 + 1) :=
  sorry

end NUMINAMATH_CALUDE_constant_distance_l4106_410689


namespace NUMINAMATH_CALUDE_insect_count_l4106_410665

/-- Given a number of leaves, ladybugs per leaf, and ants per leaf, 
    calculate the total number of ladybugs and ants combined. -/
def total_insects (leaves : ℕ) (ladybugs_per_leaf : ℕ) (ants_per_leaf : ℕ) : ℕ :=
  leaves * ladybugs_per_leaf + leaves * ants_per_leaf

/-- Theorem stating that given 84 leaves, 139 ladybugs per leaf, and 97 ants per leaf,
    the total number of ladybugs and ants combined is 19,824. -/
theorem insect_count : total_insects 84 139 97 = 19824 := by
  sorry

end NUMINAMATH_CALUDE_insect_count_l4106_410665


namespace NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l4106_410655

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) : 
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l4106_410655


namespace NUMINAMATH_CALUDE_baker_cake_difference_l4106_410699

/-- Given Baker's cake inventory and transactions, prove the difference between sold and bought cakes. -/
theorem baker_cake_difference (initial_cakes bought_cakes sold_cakes : ℚ) 
  (h1 : initial_cakes = 8.5)
  (h2 : bought_cakes = 139.25)
  (h3 : sold_cakes = 145.75) :
  sold_cakes - bought_cakes = 6.5 := by
  sorry

#eval (145.75 : ℚ) - (139.25 : ℚ)

end NUMINAMATH_CALUDE_baker_cake_difference_l4106_410699


namespace NUMINAMATH_CALUDE_determinant_is_zero_l4106_410696

-- Define the polynomial and its roots
variable (p q r : ℝ)
variable (a b c d : ℝ)

-- Define the condition that a, b, c, d are roots of the polynomial
def are_roots (a b c d p q r : ℝ) : Prop :=
  a^4 + 2*a^3 + p*a^2 + q*a + r = 0 ∧
  b^4 + 2*b^3 + p*b^2 + q*b + r = 0 ∧
  c^4 + 2*c^3 + p*c^2 + q*c + r = 0 ∧
  d^4 + 2*d^3 + p*d^2 + q*d + r = 0

-- Define the matrix
def matrix (a b c d : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  !![a, b, c, d;
     b, c, d, a;
     c, d, a, b;
     d, a, b, c]

-- State the theorem
theorem determinant_is_zero (p q r : ℝ) (a b c d : ℝ) 
  (h : are_roots a b c d p q r) : 
  Matrix.det (matrix a b c d) = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_is_zero_l4106_410696


namespace NUMINAMATH_CALUDE_percent_of_whole_six_point_two_percent_of_thousand_l4106_410619

theorem percent_of_whole (part whole : ℝ) (h : whole ≠ 0) :
  (part / whole) * 100 = part * 100 / whole := by sorry

theorem six_point_two_percent_of_thousand :
  (6.2 / 1000) * 100 = 0.62 := by sorry

end NUMINAMATH_CALUDE_percent_of_whole_six_point_two_percent_of_thousand_l4106_410619


namespace NUMINAMATH_CALUDE_ball_returns_after_12_throws_l4106_410681

/-- Represents the number of girls in the circle -/
def n : ℕ := 15

/-- Represents the number of girls skipped in each throw -/
def skip : ℕ := 4

/-- The function that determines the next girl to receive the ball -/
def next (x : ℕ) : ℕ := (x + skip + 1) % n + 1

/-- Represents the sequence of girls receiving the ball -/
def ball_sequence (k : ℕ) : ℕ := 
  Nat.iterate next 1 k

theorem ball_returns_after_12_throws : 
  ball_sequence 12 = 1 := by sorry

end NUMINAMATH_CALUDE_ball_returns_after_12_throws_l4106_410681


namespace NUMINAMATH_CALUDE_condition_relationship_l4106_410622

theorem condition_relationship :
  (∀ x : ℝ, (0 < x ∧ x < 5) → (|x - 2| < 3)) ∧
  (∃ x : ℝ, (|x - 2| < 3) ∧ ¬(0 < x ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l4106_410622


namespace NUMINAMATH_CALUDE_circles_intersection_range_l4106_410675

-- Define the circles
def C₁ (t x y : ℝ) : Prop := x^2 + y^2 - 2*t*x + t^2 - 4 = 0
def C₂ (t x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*t*y + 4*t^2 - 8 = 0

-- Define the intersection condition
def intersect (t : ℝ) : Prop := ∃ x y : ℝ, C₁ t x y ∧ C₂ t x y

-- State the theorem
theorem circles_intersection_range :
  ∀ t : ℝ, intersect t ↔ ((-12/5 < t ∧ t < -2/5) ∨ (0 < t ∧ t < 2)) :=
by sorry

end NUMINAMATH_CALUDE_circles_intersection_range_l4106_410675


namespace NUMINAMATH_CALUDE_airport_exchange_calculation_l4106_410605

/-- Calculates the amount of dollars received when exchanging euros at an airport with a reduced exchange rate. -/
theorem airport_exchange_calculation (euros : ℝ) (normal_rate : ℝ) (airport_rate_fraction : ℝ) : 
  euros / normal_rate * airport_rate_fraction = 10 :=
by
  -- Assuming euros = 70, normal_rate = 5, and airport_rate_fraction = 5/7
  sorry

#check airport_exchange_calculation

end NUMINAMATH_CALUDE_airport_exchange_calculation_l4106_410605


namespace NUMINAMATH_CALUDE_length_of_AD_rhombus_condition_l4106_410615

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℝ := x^2 - (a - 4) * x + a - 1

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (AB AD : ℝ)
  (a : ℝ)
  (eq_AB : quadratic_equation a AB = 0)
  (eq_AD : quadratic_equation a AD = 0)

-- Theorem 1: Length of AD
theorem length_of_AD (ABCD : Quadrilateral) (h : ABCD.AB = 2) : ABCD.AD = 5 := by
  sorry

-- Theorem 2: Condition for rhombus
theorem rhombus_condition (ABCD : Quadrilateral) : ABCD.AB = ABCD.AD ↔ ABCD.a = 10 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AD_rhombus_condition_l4106_410615


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l4106_410618

-- Define the circle equation
def circle_eq (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define perpendicularity of two points from origin
def perp_from_origin (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_intersection_theorem :
  ∀ m : ℝ,
  (∃ x y : ℝ, circle_eq x y m) ↔ m < 5 ∧
  (∃ x1 y1 x2 y2 : ℝ,
    circle_eq x1 y1 m ∧ circle_eq x2 y2 m ∧
    line_eq x1 y1 ∧ line_eq x2 y2 ∧
    perp_from_origin x1 y1 x2 y2 → m = 8/5) ∧
  (∃ x1 y1 x2 y2 : ℝ,
    circle_eq x1 y1 (8/5) ∧ circle_eq x2 y2 (8/5) ∧
    line_eq x1 y1 ∧ line_eq x2 y2 ∧
    perp_from_origin x1 y1 x2 y2 →
    ∀ x y : ℝ, x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔
    ∃ t : ℝ, x = x1 + t*(x2 - x1) ∧ y = y1 + t*(y2 - y1) ∧ 0 ≤ t ∧ t ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l4106_410618


namespace NUMINAMATH_CALUDE_simplify_expressions_l4106_410634

theorem simplify_expressions 
  (x y a b : ℝ) : 
  (-3*x + 2*y - 5*x - 7*y = -8*x - 5*y) ∧ 
  (5*(3*a^2*b - a*b^2) - 4*(-a*b^2 + 3*a^2*b) = 3*a^2*b - a*b^2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l4106_410634


namespace NUMINAMATH_CALUDE_odd_square_plus_even_product_is_odd_l4106_410674

theorem odd_square_plus_even_product_is_odd (k m : ℤ) : 
  let o : ℤ := 2 * k + 3
  let n : ℤ := 2 * m
  Odd (o^2 + n * o) := by
sorry

end NUMINAMATH_CALUDE_odd_square_plus_even_product_is_odd_l4106_410674


namespace NUMINAMATH_CALUDE_sea_glass_collection_l4106_410647

theorem sea_glass_collection (blanche_green : ℕ) (rose_red rose_blue : ℕ) (dorothy_total : ℕ)
  (h1 : blanche_green = 12)
  (h2 : rose_red = 9)
  (h3 : rose_blue = 11)
  (h4 : dorothy_total = 57) :
  ∃ (blanche_red : ℕ),
    dorothy_total = 2 * (blanche_red + rose_red) + 3 * rose_blue ∧
    blanche_red = 3 :=
by sorry

end NUMINAMATH_CALUDE_sea_glass_collection_l4106_410647


namespace NUMINAMATH_CALUDE_expression_equality_l4106_410692

theorem expression_equality (a b c : ℝ) (h : a^2 + b = b^2 + c ∧ b^2 + c = c^2 + a) :
  a*(a^2 - b^2) + b*(b^2 - c^2) + c*(c^2 - a^2) = 0 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l4106_410692


namespace NUMINAMATH_CALUDE_f_value_at_2_l4106_410639

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l4106_410639


namespace NUMINAMATH_CALUDE_min_droppers_required_l4106_410629

theorem min_droppers_required (container_volume : ℕ) (dropper_volume : ℕ) : container_volume = 265 → dropper_volume = 19 → (14 : ℕ) = (container_volume + dropper_volume - 1) / dropper_volume := by
  sorry

end NUMINAMATH_CALUDE_min_droppers_required_l4106_410629


namespace NUMINAMATH_CALUDE_train_circuit_time_l4106_410602

/-- Represents the time in seconds -/
def seconds_per_circuit : ℕ := 71

/-- Represents the number of circuits -/
def num_circuits : ℕ := 6

/-- Converts seconds to minutes and remaining seconds -/
def seconds_to_minutes_and_seconds (total_seconds : ℕ) : ℕ × ℕ :=
  (total_seconds / 60, total_seconds % 60)

theorem train_circuit_time : 
  seconds_to_minutes_and_seconds (num_circuits * seconds_per_circuit) = (7, 6) := by
  sorry

end NUMINAMATH_CALUDE_train_circuit_time_l4106_410602


namespace NUMINAMATH_CALUDE_part_one_part_two_l4106_410649

-- Define the sets A and B
def A : Set ℝ := {x | -2 + 3*x - x^2 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 2}

-- Part 1: Prove that when a = 1, (∁A) ∩ B = (1, 2)
theorem part_one : (Set.compl A) ∩ (B 1) = Set.Ioo 1 2 := by sorry

-- Part 2: Prove that (∁A) ∩ B = ∅ if and only if a ≤ -1 or a ≥ 2
theorem part_two (a : ℝ) : (Set.compl A) ∩ (B a) = ∅ ↔ a ≤ -1 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4106_410649


namespace NUMINAMATH_CALUDE_angle_c_measure_l4106_410671

theorem angle_c_measure (A B C : ℝ) : 
  A = 86 →
  B = 3 * C + 22 →
  A + B + C = 180 →
  C = 18 := by
sorry

end NUMINAMATH_CALUDE_angle_c_measure_l4106_410671


namespace NUMINAMATH_CALUDE_vectors_collinear_l4106_410664

/-- The problem setup -/
structure GeometrySetup where
  -- The coordinate system
  P : ℝ × ℝ
  Q : ℝ × ℝ
  S : ℝ × ℝ
  T : ℝ × ℝ
  N : ℝ × ℝ
  M : ℝ × ℝ
  -- Conditions
  hl : S.1 = -1
  hT : T = (3, 0)
  hPl : S.2 = P.2
  hOP_ST : P.1 * 4 - P.2 * S.2 = 0
  hC : Q.2^2 = 4 * Q.1
  hPQ : ∃ (t : ℝ), (1 - P.1) * t + P.1 = Q.1 ∧ (0 - P.2) * t + P.2 = Q.2
  hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  hN : N = (-1, 0)

/-- The theorem to be proved -/
theorem vectors_collinear (g : GeometrySetup) : 
  ∃ (k : ℝ), (g.M.1 - g.S.1, g.M.2 - g.S.2) = k • (g.Q.1 - g.N.1, g.Q.2 - g.N.2) := by
  sorry

end NUMINAMATH_CALUDE_vectors_collinear_l4106_410664


namespace NUMINAMATH_CALUDE_sin_6theta_l4106_410609

theorem sin_6theta (θ : ℝ) :
  Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 4 →
  Real.sin (6 * θ) = -855 * Real.sqrt 2 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_sin_6theta_l4106_410609


namespace NUMINAMATH_CALUDE_smallest_non_triangle_forming_subtraction_l4106_410633

theorem smallest_non_triangle_forming_subtraction : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (∀ (y : ℕ), y < x → (7 - y) + (24 - y) > (26 - y)) ∧
  ((7 - x) + (24 - x) ≤ (26 - x)) ∧
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_non_triangle_forming_subtraction_l4106_410633


namespace NUMINAMATH_CALUDE_square_greater_than_abs_l4106_410617

theorem square_greater_than_abs (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_abs_l4106_410617


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l4106_410611

/-- 
Given a quadratic equation (m-1)x^2 + 3x - 1 = 0,
prove that for the equation to have real roots,
m must satisfy: m ≥ -5/4 and m ≠ 1
-/
theorem quadratic_real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 3 * x - 1 = 0) ↔ 
  (m ≥ -5/4 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l4106_410611


namespace NUMINAMATH_CALUDE_game_theorem_l4106_410603

/-- Represents the game "What? Where? When?" with given conditions -/
structure Game where
  envelopes : ℕ := 13
  win_points : ℕ := 6
  win_prob : ℚ := 1/2

/-- Expected number of points for a single game -/
def expected_points (g : Game) : ℚ := sorry

/-- Expected number of points over 100 games -/
def expected_points_100 (g : Game) : ℚ := 100 * expected_points g

/-- Probability of an envelope being chosen in a game -/
def envelope_prob (g : Game) : ℚ := sorry

theorem game_theorem (g : Game) :
  expected_points_100 g = 465 ∧ envelope_prob g = 12/13 := by sorry

end NUMINAMATH_CALUDE_game_theorem_l4106_410603


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_9999_l4106_410694

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The sum of factorials from 1 to n -/
def sumFactorials (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => factorial (n + 1) + sumFactorials n

theorem units_digit_sum_factorials_9999 :
  unitsDigit (sumFactorials 9999) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_9999_l4106_410694


namespace NUMINAMATH_CALUDE_min_brilliant_product_l4106_410695

/-- A triple of integers (a, b, c) is brilliant if:
    1. a > b > c are prime numbers
    2. a = b + 2c
    3. a + b + c is a perfect square number -/
def is_brilliant (a b c : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧
  a > b ∧ b > c ∧
  a = b + 2 * c ∧
  ∃ k, a + b + c = k * k

/-- The minimum value of abc for a brilliant triple (a, b, c) is 35651 -/
theorem min_brilliant_product :
  (∀ a b c : ℕ, is_brilliant a b c → a * b * c ≥ 35651) ∧
  ∃ a b c : ℕ, is_brilliant a b c ∧ a * b * c = 35651 :=
sorry

end NUMINAMATH_CALUDE_min_brilliant_product_l4106_410695


namespace NUMINAMATH_CALUDE_right_triangle_integer_sides_l4106_410669

theorem right_triangle_integer_sides (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem (right triangle condition)
  Nat.gcd a (Nat.gcd b c) = 1 →  -- GCD of sides is 1
  ∃ m n : ℕ, 
    (a = 2*m*n ∧ b = m^2 - n^2 ∧ c = m^2 + n^2) ∨ 
    (b = 2*m*n ∧ a = m^2 - n^2 ∧ c = m^2 + n^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_integer_sides_l4106_410669


namespace NUMINAMATH_CALUDE_P_range_l4106_410687

theorem P_range (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let P := (a^2 / (a^2 + b^2 + c^2)) + (b^2 / (b^2 + c^2 + d^2)) +
           (c^2 / (c^2 + d^2 + a^2)) + (d^2 / (d^2 + a^2 + b^2))
  1 < P ∧ P < 2 := by
  sorry

end NUMINAMATH_CALUDE_P_range_l4106_410687


namespace NUMINAMATH_CALUDE_min_value_theorem_l4106_410646

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  x + 3 * y ≥ 18 + 21 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    x₀ + 3 * y₀ = 18 + 21 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4106_410646


namespace NUMINAMATH_CALUDE_coplanar_iff_k_eq_neg_eight_l4106_410643

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points as vectors
variable (O A B C D E : V)

-- Define the scalar k
variable (k : ℝ)

-- Define the condition from the problem
def vector_equation (O A B C D E : V) (k : ℝ) : Prop :=
  4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) + (E - O) = 0

-- Define coplanarity
def coplanar (A B C D E : V) : Prop :=
  ∃ (a b c d : ℝ), a • (B - A) + b • (C - A) + c • (D - A) + d • (E - A) = 0

-- State the theorem
theorem coplanar_iff_k_eq_neg_eight
  (O A B C D E : V) (k : ℝ) :
  vector_equation O A B C D E k →
  (coplanar A B C D E ↔ k = -8) :=
sorry

end NUMINAMATH_CALUDE_coplanar_iff_k_eq_neg_eight_l4106_410643


namespace NUMINAMATH_CALUDE_problem_solution_l4106_410662

theorem problem_solution (d : ℝ) (a b c : ℤ) (h1 : d ≠ 0) 
  (h2 : (18 * d + 19 + 20 * d^2) + (4 * d + 3 - 2 * d^2) = a * d + b + c * d^2) : 
  a + b + c = 62 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4106_410662


namespace NUMINAMATH_CALUDE_candy_bar_cost_l4106_410601

theorem candy_bar_cost (selling_price : ℕ) (bought : ℕ) (sold : ℕ) (profit : ℕ) : 
  selling_price = 100 ∧ bought = 50 ∧ sold = 48 ∧ profit = 800 →
  (selling_price * sold - profit) / bought = 80 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l4106_410601


namespace NUMINAMATH_CALUDE_solve_for_q_l4106_410614

theorem solve_for_q (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 16 / 3) : 
  q = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l4106_410614


namespace NUMINAMATH_CALUDE_other_diagonal_length_l4106_410635

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  diag1 : ℝ
  diag2 : ℝ
  area : ℝ

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.diag1 * r.diag2) / 2

theorem other_diagonal_length (r : Rhombus) 
  (h1 : r.diag1 = 14)
  (h2 : r.area = 140) : 
  r.diag2 = 20 := by
sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l4106_410635


namespace NUMINAMATH_CALUDE_one_true_related_proposition_l4106_410625

theorem one_true_related_proposition :
  let P : ℝ → Prop := λ b => b = 3
  let Q : ℝ → Prop := λ b => b^2 = 9
  let converse := ∀ b, Q b → P b
  let negation := ∀ b, ¬(P b) → ¬(Q b)
  let inverse := ∀ b, ¬(Q b) → ¬(P b)
  (converse ∨ negation ∨ inverse) ∧ ¬(converse ∧ negation) ∧ ¬(converse ∧ inverse) ∧ ¬(negation ∧ inverse) :=
by
  sorry

#check one_true_related_proposition

end NUMINAMATH_CALUDE_one_true_related_proposition_l4106_410625


namespace NUMINAMATH_CALUDE_intersection_point_l4106_410608

/-- The first curve equation -/
def curve1 (x : ℝ) : ℝ := x^3 + 3*x^2 + 4*x - 5

/-- The second curve equation -/
def curve2 (x : ℝ) : ℝ := 2*x^2 + 11

/-- Theorem stating that (2, 19) is the only intersection point of the two curves -/
theorem intersection_point : 
  (∃! p : ℝ × ℝ, curve1 p.1 = curve2 p.1 ∧ p.2 = curve1 p.1) ∧ 
  (∀ p : ℝ × ℝ, curve1 p.1 = curve2 p.1 → p = (2, 19)) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l4106_410608


namespace NUMINAMATH_CALUDE_gear_teeth_problem_l4106_410654

theorem gear_teeth_problem (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 60) (h4 : 4 * x - 20 = 5 * y) (h5 : 5 * y = 10 * z) : x = 30 ∧ y = 20 ∧ z = 10 := by
  sorry

end NUMINAMATH_CALUDE_gear_teeth_problem_l4106_410654


namespace NUMINAMATH_CALUDE_mn_length_is_8_l4106_410607

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points on a line parallel to the x-axis -/
def distanceOnXParallelLine (p1 p2 : Point) : ℝ :=
  |p1.x - p2.x|

theorem mn_length_is_8 (x : ℝ) :
  let m : Point := ⟨x + 5, x - 4⟩
  let n : Point := ⟨-1, -2⟩
  m.y = n.y → distanceOnXParallelLine m n = 8 := by
  sorry

end NUMINAMATH_CALUDE_mn_length_is_8_l4106_410607


namespace NUMINAMATH_CALUDE_soft_drink_storage_l4106_410673

theorem soft_drink_storage (small_initial big_initial : ℕ) 
  (big_sold_percent : ℚ) (total_remaining : ℕ) :
  small_initial = 6000 →
  big_initial = 14000 →
  big_sold_percent = 23 / 100 →
  total_remaining = 15580 →
  ∃ (small_sold_percent : ℚ),
    small_sold_percent = 37 / 100 ∧
    (small_initial : ℚ) * (1 - small_sold_percent) + 
    (big_initial : ℚ) * (1 - big_sold_percent) = total_remaining := by
  sorry

end NUMINAMATH_CALUDE_soft_drink_storage_l4106_410673


namespace NUMINAMATH_CALUDE_last_remaining_200_l4106_410650

/-- The last remaining number after the marking process -/
def lastRemainingNumber (n : ℕ) : ℕ :=
  if n ≤ 1 then n else 2 * lastRemainingNumber ((n + 1) / 2)

/-- The theorem stating that for 200 numbers, the last remaining is 128 -/
theorem last_remaining_200 :
  lastRemainingNumber 200 = 128 := by
  sorry

end NUMINAMATH_CALUDE_last_remaining_200_l4106_410650


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l4106_410621

def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem quadratic_coefficients :
  ∃ (a b c : ℝ), (∀ x, f x = a*x^2 + b*x + c) ∧ a = 1 ∧ b = -4 ∧ c = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l4106_410621


namespace NUMINAMATH_CALUDE_min_radius_circle_line_intersection_l4106_410683

theorem min_radius_circle_line_intersection (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi)) :
  let circle := fun (x y : Real) => (x - Real.cos θ)^2 + (y - Real.sin θ)^2
  let line := fun (x y : Real) => 2 * x - y - 10
  ∃ (r : Real), r > 0 ∧ ∃ (x y : Real), circle x y = r^2 ∧ line x y = 0 →
  ∀ (r' : Real), (∃ (x y : Real), circle x y = r'^2 ∧ line x y = 0) → r' ≥ 2 * Real.sqrt 5 - 1 :=
sorry

end NUMINAMATH_CALUDE_min_radius_circle_line_intersection_l4106_410683


namespace NUMINAMATH_CALUDE_total_egg_weight_in_pounds_l4106_410651

-- Define the weight of a single egg in pounds
def egg_weight : ℚ := 1 / 16

-- Define the number of dozens of eggs needed
def dozens_needed : ℕ := 8

-- Define the number of eggs in a dozen
def eggs_per_dozen : ℕ := 12

-- Theorem to prove
theorem total_egg_weight_in_pounds : 
  (dozens_needed * eggs_per_dozen : ℚ) * egg_weight = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_egg_weight_in_pounds_l4106_410651
