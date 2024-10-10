import Mathlib

namespace lap_time_improvement_is_two_thirds_l2789_278910

/-- Represents running data with number of laps and total time in minutes -/
structure RunningData where
  laps : ℕ
  time : ℚ

/-- Calculates the lap time in minutes for given running data -/
def lapTime (data : RunningData) : ℚ :=
  data.time / data.laps

/-- The initial running data -/
def initialData : RunningData :=
  { laps := 15, time := 45 }

/-- The final running data after training -/
def finalData : RunningData :=
  { laps := 18, time := 42 }

/-- The improvement in lap time -/
def lapTimeImprovement : ℚ :=
  lapTime initialData - lapTime finalData

theorem lap_time_improvement_is_two_thirds :
  lapTimeImprovement = 2 / 3 := by
  sorry

end lap_time_improvement_is_two_thirds_l2789_278910


namespace lid_circumference_l2789_278923

theorem lid_circumference (diameter : ℝ) (h : diameter = 2) :
  Real.pi * diameter = 2 * Real.pi :=
by sorry

end lid_circumference_l2789_278923


namespace smallest_d_for_3150_perfect_square_l2789_278935

theorem smallest_d_for_3150_perfect_square : 
  ∃ (d : ℕ), d > 0 ∧ d = 14 ∧ 
  (∃ (n : ℕ), 3150 * d = n^2) ∧
  (∀ (k : ℕ), k > 0 → k < d → ¬∃ (m : ℕ), 3150 * k = m^2) := by
  sorry

end smallest_d_for_3150_perfect_square_l2789_278935


namespace triangle_angle_problem_l2789_278904

theorem triangle_angle_problem (x z : ℝ) : 
  (2*x + 3*x + x = 180) → 
  (x + z = 180) → 
  z = 150 := by
sorry

end triangle_angle_problem_l2789_278904


namespace calculation_proof_l2789_278901

theorem calculation_proof :
  ((-3)^2 - 60 / 10 * (1 / 10) - |(-2)|) = 32 / 5 ∧
  (-4 / 5 * (9 / 4) + (-1 / 4) * (4 / 5) - (3 / 2) * (-4 / 5)) = -4 / 5 := by
  sorry

end calculation_proof_l2789_278901


namespace geometric_sequence_ratio_l2789_278986

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geometric : geometric_sequence a q)
  (h_product : a 1 * a 2 * a 3 = 27)
  (h_sum : a 2 + a 4 = 30) :
  q = 3 ∨ q = -3 :=
sorry

end geometric_sequence_ratio_l2789_278986


namespace total_gas_usage_l2789_278967

theorem total_gas_usage (adhira_usage : ℕ) (felicity_usage : ℕ) : 
  felicity_usage = 4 * adhira_usage - 5 →
  felicity_usage = 23 →
  felicity_usage + adhira_usage = 30 :=
by
  sorry

end total_gas_usage_l2789_278967


namespace investment_calculation_l2789_278939

/-- Given two investors p and q with an investment ratio of 4:5, 
    where q invests 65000, prove that p's investment is 52000. -/
theorem investment_calculation (p q : ℕ) : 
  (p : ℚ) / q = 4 / 5 → q = 65000 → p = 52000 := by
  sorry

end investment_calculation_l2789_278939


namespace goldbach_negation_equivalence_l2789_278964

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → n % 2 = 0 → ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

theorem goldbach_negation_equivalence :
  (¬ goldbach_conjecture) ↔
  (∃ n : ℕ, n > 2 ∧ n % 2 = 0 ∧ ∀ p q : ℕ, is_prime p → is_prime q → n ≠ p + q) :=
sorry

end goldbach_negation_equivalence_l2789_278964


namespace cube_root_opposite_zero_l2789_278900

theorem cube_root_opposite_zero :
  ∀ x : ℝ, (x^(1/3) = -x) ↔ (x = 0) :=
sorry

end cube_root_opposite_zero_l2789_278900


namespace area_triangle_BXD_l2789_278996

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  base_AB : ℝ
  base_CD : ℝ
  area : ℝ

/-- Theorem about the area of triangle BXD in a trapezoid -/
theorem area_triangle_BXD (ABCD : Trapezoid) (h1 : ABCD.base_AB = 24)
    (h2 : ABCD.base_CD = 36) (h3 : ABCD.area = 360) : ℝ := by
  -- The area of triangle BXD is 57.6 square units
  sorry

#check area_triangle_BXD

end area_triangle_BXD_l2789_278996


namespace no_function_satisfies_conditions_l2789_278931

theorem no_function_satisfies_conditions :
  ¬∃ (f : ℚ → ℝ),
    (f 0 = 0) ∧
    (∀ a : ℚ, a ≠ 0 → f a > 0) ∧
    (∀ x y : ℚ, f (x + y) = f x * f y) ∧
    (∀ x y : ℚ, x ≠ 0 → y ≠ 0 → f (x + y) ≤ max (f x) (f y)) ∧
    (∃ x : ℤ, f x ≠ 1) ∧
    (∀ n : ℕ, n > 0 → ∀ x : ℤ, f (1 + x + x^2 + (x^n - 1) / (x - 1)) = 1) :=
by sorry

end no_function_satisfies_conditions_l2789_278931


namespace remainder_7n_mod_4_l2789_278971

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l2789_278971


namespace fifteen_dogs_like_neither_l2789_278917

/-- Represents the number of dogs in different categories -/
structure DogCounts where
  total : Nat
  likesChicken : Nat
  likesBeef : Nat
  likesBoth : Nat

/-- Calculates the number of dogs that don't like either chicken or beef -/
def dogsLikingNeither (counts : DogCounts) : Nat :=
  counts.total - (counts.likesChicken + counts.likesBeef - counts.likesBoth)

/-- Theorem stating that 15 dogs don't like either chicken or beef -/
theorem fifteen_dogs_like_neither (counts : DogCounts)
  (h1 : counts.total = 75)
  (h2 : counts.likesChicken = 13)
  (h3 : counts.likesBeef = 55)
  (h4 : counts.likesBoth = 8) :
  dogsLikingNeither counts = 15 := by
  sorry

end fifteen_dogs_like_neither_l2789_278917


namespace equidistant_function_b_squared_l2789_278911

/-- A complex function that is equidistant from its input and the origin -/
def equidistant_function (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : ℂ → ℂ := 
  fun z ↦ (a + b * Complex.I) * z

/-- The main theorem -/
theorem equidistant_function_b_squared 
  (a b : ℝ) 
  (h₁ : 0 < a) 
  (h₂ : 0 < b) 
  (h₃ : ∀ z : ℂ, Complex.abs (equidistant_function a b h₁ h₂ z - z) = Complex.abs (equidistant_function a b h₁ h₂ z))
  (h₄ : Complex.abs (a + b * Complex.I) = 10) :
  b^2 = 99.75 := by
sorry

end equidistant_function_b_squared_l2789_278911


namespace power_algorithm_correct_l2789_278918

/-- Algorithm to compute B^N -/
def power_algorithm (B : ℝ) (N : ℕ) : ℝ :=
  if N = 0 then 1
  else
    let rec loop (a b : ℝ) (n : ℕ) : ℝ :=
      if n = 0 then a
      else if n % 2 = 0 then loop a (b * b) (n / 2)
      else loop (a * b) (b * b) (n / 2)
    loop 1 B N

/-- Theorem stating that the algorithm computes B^N -/
theorem power_algorithm_correct (B : ℝ) (N : ℕ) (hB : B > 0) :
  power_algorithm B N = B ^ N := by
  sorry

#check power_algorithm_correct

end power_algorithm_correct_l2789_278918


namespace sum_reciprocals_l2789_278958

theorem sum_reciprocals (x y z : ℝ) (ω : ℂ) 
  (hx : x ≠ -1) (hy : y ≠ -1) (hz : z ≠ -1)
  (hω1 : ω^3 = 1) (hω2 : ω ≠ 1)
  (h : 1/(x + ω) + 1/(y + ω) + 1/(z + ω) = ω) :
  1/(x + 1) + 1/(y + 1) + 1/(z + 1) = -1/3 := by
sorry

end sum_reciprocals_l2789_278958


namespace fraction_decomposition_l2789_278956

theorem fraction_decomposition (n : ℕ) (hn : n > 0) :
  (∃ (a b : ℕ), a ≠ b ∧ 3 / (5 * n) = 1 / a + 1 / b) ∧
  ((∃ (x : ℤ), 3 / (5 * n) = 1 / x + 1 / x) ↔ ∃ (k : ℕ), n = 3 * k) ∧
  (n > 1 → ∃ (c d : ℕ), 3 / (5 * n) = 1 / c - 1 / d) :=
by sorry

end fraction_decomposition_l2789_278956


namespace inserted_eights_composite_l2789_278993

def insert_eights (n : ℕ) : ℕ :=
  2000 * 10^n + 8 * ((10^n - 1) / 9) + 21

theorem inserted_eights_composite (n : ℕ) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ insert_eights n = a * b :=
sorry

end inserted_eights_composite_l2789_278993


namespace function_properties_l2789_278976

/-- Given function f with parameter a > 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1/a) * Real.log x + 1/x - x

theorem function_properties (a : ℝ) (h : a > 1) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/a), (deriv (f a)) x < 0) ∧
  (∀ x ∈ Set.Ioo (1/a) 1, (deriv (f a)) x > 0) ∧
  (a ≥ 3 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (deriv (f a)) x₁ = (deriv (f a)) x₂ ∧ x₁ + x₂ > 6/5) :=
by sorry

end function_properties_l2789_278976


namespace rotation_transformation_l2789_278979

-- Define the triangles
def triangle_DEF : List (ℝ × ℝ) := [(0, 0), (0, 10), (14, 0)]
def triangle_DEF_prime : List (ℝ × ℝ) := [(28, 14), (40, 14), (28, 4)]

-- Define the rotation function
def rotate (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

theorem rotation_transformation (n p q : ℝ) :
  0 < n → n < 180 →
  (∀ (point : ℝ × ℝ), point ∈ triangle_DEF →
    rotate (p, q) n point ∈ triangle_DEF_prime) →
  n + p + q = 104 := by sorry

end rotation_transformation_l2789_278979


namespace not_p_sufficient_not_necessary_for_not_p_and_q_l2789_278953

theorem not_p_sufficient_not_necessary_for_not_p_and_q (p q : Prop) :
  (∀ (h : ¬p), ¬(p ∧ q)) ∧
  ¬(∀ (h : ¬(p ∧ q)), ¬p) :=
sorry

end not_p_sufficient_not_necessary_for_not_p_and_q_l2789_278953


namespace nuts_left_over_project_nuts_left_over_l2789_278987

theorem nuts_left_over (bolt_boxes : ℕ) (bolts_per_box : ℕ) (nut_boxes : ℕ) (nuts_per_box : ℕ) 
  (bolts_left : ℕ) (total_used : ℕ) : ℕ :=
  let total_bolts := bolt_boxes * bolts_per_box
  let total_nuts := nut_boxes * nuts_per_box
  let bolts_used := total_bolts - bolts_left
  let nuts_used := total_used - bolts_used
  let nuts_left := total_nuts - nuts_used
  nuts_left

theorem project_nuts_left_over : 
  nuts_left_over 7 11 3 15 3 113 = 6 := by
  sorry

end nuts_left_over_project_nuts_left_over_l2789_278987


namespace next_simultaneous_ringing_l2789_278954

def town_hall_period : ℕ := 18
def university_tower_period : ℕ := 24
def fire_station_period : ℕ := 30

def minutes_in_hour : ℕ := 60

theorem next_simultaneous_ringing :
  ∃ (n : ℕ), n > 0 ∧ 
    n % town_hall_period = 0 ∧
    n % university_tower_period = 0 ∧
    n % fire_station_period = 0 ∧
    n / minutes_in_hour = 6 :=
sorry

end next_simultaneous_ringing_l2789_278954


namespace shorter_leg_length_l2789_278962

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  /-- The length of the shorter leg -/
  shorter_leg : ℝ
  /-- The length of the longer leg -/
  longer_leg : ℝ
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The length of the median to the hypotenuse -/
  median_to_hypotenuse : ℝ
  /-- Constraint: The median to the hypotenuse is half the hypotenuse -/
  median_hypotenuse_relation : median_to_hypotenuse = hypotenuse / 2
  /-- Constraint: The shorter leg is half the hypotenuse -/
  shorter_leg_relation : shorter_leg = hypotenuse / 2
  /-- Constraint: The longer leg is √3 times the shorter leg -/
  longer_leg_relation : longer_leg = shorter_leg * Real.sqrt 3

/-- 
Theorem: In a 30-60-90 triangle, if the length of the median to the hypotenuse is 15 units, 
then the length of the shorter leg is 15 units.
-/
theorem shorter_leg_length (t : Triangle30_60_90) (h : t.median_to_hypotenuse = 15) : 
  t.shorter_leg = 15 := by
  sorry

end shorter_leg_length_l2789_278962


namespace shop_monthly_rent_l2789_278955

/-- Calculates the monthly rent of a shop given its dimensions and annual rent per square foot. -/
def monthly_rent (length width annual_rent_per_sqft : ℕ) : ℕ :=
  let area := length * width
  let annual_rent := area * annual_rent_per_sqft
  annual_rent / 12

/-- Theorem stating that for a shop with given dimensions and annual rent per square foot,
    the monthly rent is 3600. -/
theorem shop_monthly_rent :
  monthly_rent 20 15 144 = 3600 := by
  sorry

end shop_monthly_rent_l2789_278955


namespace cow_count_l2789_278988

theorem cow_count (D C : ℕ) : 
  2 * D + 4 * C = 2 * (D + C) + 30 → C = 15 := by
  sorry

end cow_count_l2789_278988


namespace rational_equation_solution_l2789_278934

theorem rational_equation_solution (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 → (B * x - 13) / (x^2 - 8*x + 15) = A / (x - 3) + 4 / (x - 5)) →
  A + B = 22 / 5 := by
sorry

end rational_equation_solution_l2789_278934


namespace train_crossing_time_l2789_278952

theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ) 
  (h1 : train_length = 900)
  (h2 : platform_length = 1050)
  (h3 : platform_crossing_time = 39)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 := by
  sorry

end train_crossing_time_l2789_278952


namespace arithmetic_triangle_sum_l2789_278920

-- Define a triangle with angles in arithmetic progression and side lengths 6, 7, and y
structure ArithmeticTriangle where
  y : ℝ
  angle_progression : ℝ → ℝ → ℝ → Prop
  side_lengths : ℝ → ℝ → ℝ → Prop

-- Define the sum of possible y values
def sum_of_y_values (t : ArithmeticTriangle) : ℝ := sorry

-- Define positive integers a, b, and c
def a : ℕ := sorry
def b : ℕ := sorry
def c : ℕ := sorry

-- Theorem statement
theorem arithmetic_triangle_sum :
  ∃ (t : ArithmeticTriangle),
    t.angle_progression 60 60 60 ∧
    t.side_lengths 6 7 t.y ∧
    sum_of_y_values t = a + Real.sqrt b + Real.sqrt c ∧
    a + b + c = 68 := by
  sorry

end arithmetic_triangle_sum_l2789_278920


namespace max_value_of_fraction_l2789_278982

theorem max_value_of_fraction (x y z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) → 
  (10 ≤ y ∧ y ≤ 99) → 
  (10 ≤ z ∧ z ≤ 99) → 
  ((x + y + z) / 3 = 60) → 
  ((x + y) / z ≤ 17) ∧ 
  (∃ (a b c : ℕ), (10 ≤ a ∧ a ≤ 99) ∧ 
                  (10 ≤ b ∧ b ≤ 99) ∧ 
                  (10 ≤ c ∧ c ≤ 99) ∧ 
                  ((a + b + c) / 3 = 60) ∧ 
                  ((a + b) / c = 17)) :=
by sorry

end max_value_of_fraction_l2789_278982


namespace gcd_765432_654321_l2789_278980

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l2789_278980


namespace memory_card_picture_size_l2789_278949

/-- Represents a memory card with a given capacity and picture storage capabilities. -/
structure MemoryCard where
  capacity : ℕ  -- Total capacity in megabytes
  large_pics : ℕ  -- Number of large pictures it can hold
  small_pics : ℕ  -- Number of small pictures it can hold
  small_pic_size : ℕ  -- Size of small pictures in megabytes

/-- Calculates the size of pictures when the card is filled with large pictures. -/
def large_pic_size (card : MemoryCard) : ℕ :=
  card.capacity / card.large_pics

theorem memory_card_picture_size (card : MemoryCard) 
  (h1 : card.small_pics = 3000)
  (h2 : card.large_pics = 4000)
  (h3 : large_pic_size card = 6) :
  card.small_pic_size = 8 := by
  sorry

#check memory_card_picture_size

end memory_card_picture_size_l2789_278949


namespace multiplication_of_powers_l2789_278975

theorem multiplication_of_powers (a : ℝ) : 4 * (a^2) * (a^3) = 4 * (a^5) := by
  sorry

end multiplication_of_powers_l2789_278975


namespace min_value_of_complex_l2789_278957

open Complex

theorem min_value_of_complex (z : ℂ) (h : abs (z + I) + abs (z - I) = 2) :
  (∀ w : ℂ, abs (w + I) + abs (w - I) = 2 → abs (z + I + 1) ≤ abs (w + I + 1)) ∧
  (∃ z₀ : ℂ, abs (z₀ + I) + abs (z₀ - I) = 2 ∧ abs (z₀ + I + 1) = 1) :=
by sorry

end min_value_of_complex_l2789_278957


namespace whole_number_between_values_l2789_278946

theorem whole_number_between_values (N : ℤ) : 
  (6.75 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 7.25) → N = 28 := by
  sorry

end whole_number_between_values_l2789_278946


namespace chris_age_l2789_278974

/-- The ages of four friends satisfying certain conditions -/
def FriendsAges (a b c d : ℝ) : Prop :=
  -- The average age is 12
  (a + b + c + d) / 4 = 12 ∧
  -- Five years ago, Chris was twice as old as Amy
  c - 5 = 2 * (a - 5) ∧
  -- In 2 years, Ben's age will be three-quarters of Amy's age
  b + 2 = 3/4 * (a + 2) ∧
  -- Diana is 15 years old
  d = 15

/-- Chris's age is 16 given the conditions -/
theorem chris_age (a b c d : ℝ) (h : FriendsAges a b c d) : c = 16 := by
  sorry

end chris_age_l2789_278974


namespace train_speed_calculation_l2789_278936

/-- Calculates the speed of a train given its length, time to cross a man, and the man's speed -/
theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) :
  train_length = 120 →
  crossing_time = 6 →
  man_speed_kmh = 5 →
  ∃ (train_speed_kmh : ℝ), 
    (train_speed_kmh ≥ 66.9) ∧ 
    (train_speed_kmh ≤ 67.1) ∧
    (train_speed_kmh * 1000 / 3600 + man_speed_kmh * 1000 / 3600) * crossing_time = train_length :=
by sorry


end train_speed_calculation_l2789_278936


namespace prime_sum_problem_l2789_278927

theorem prime_sum_problem (p q s r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime s → Nat.Prime r →
  p + q + s = r →
  1 < p → p < q → q < s →
  p = 2 := by
sorry

end prime_sum_problem_l2789_278927


namespace pecan_mixture_amount_l2789_278914

/-- Prove that the amount of pecans in a mixture is correct given the specified conditions. -/
theorem pecan_mixture_amount 
  (cashew_amount : ℝ) 
  (cashew_price : ℝ) 
  (mixture_price : ℝ) 
  (pecan_amount : ℝ) :
  cashew_amount = 2 ∧ 
  cashew_price = 3.5 ∧ 
  mixture_price = 4.34 ∧
  pecan_amount = 1.33333333333 →
  pecan_amount = 1.33333333333 :=
by sorry

end pecan_mixture_amount_l2789_278914


namespace negative_three_times_inequality_l2789_278961

theorem negative_three_times_inequality {a b : ℝ} (h : a < b) : -3 * a > -3 * b := by
  sorry

end negative_three_times_inequality_l2789_278961


namespace monthly_expenses_ratio_l2789_278905

theorem monthly_expenses_ratio (E : ℝ) (rent_percentage : ℝ) (rent_amount : ℝ) (savings : ℝ)
  (h1 : rent_percentage = 0.07)
  (h2 : rent_amount = 133)
  (h3 : savings = 817)
  (h4 : rent_amount = E * rent_percentage) :
  (E - rent_amount - savings) / E = 0.5 := by
sorry

end monthly_expenses_ratio_l2789_278905


namespace fraction_equality_l2789_278941

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 21)
  (h2 : p / n = 7)
  (h3 : p / q = 1 / 9) :
  m / q = 1 / 3 := by
sorry

end fraction_equality_l2789_278941


namespace geometric_sequence_property_l2789_278943

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: For a geometric sequence {a_n}, if a_2 * a_4 = 1/2, then a_1 * a_3^2 * a_5 = 1/4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : geometric_sequence a) (h_cond : a 2 * a 4 = 1/2) : 
    a 1 * (a 3)^2 * a 5 = 1/4 := by
  sorry

end geometric_sequence_property_l2789_278943


namespace path_area_and_cost_l2789_278998

/-- Represents the dimensions of a rectangular field with a path around it -/
structure FieldWithPath where
  fieldLength : ℝ
  fieldWidth : ℝ
  pathWidth : ℝ

/-- Calculates the area of the path around a rectangular field -/
def areaOfPath (f : FieldWithPath) : ℝ :=
  (f.fieldLength + 2 * f.pathWidth) * (f.fieldWidth + 2 * f.pathWidth) - f.fieldLength * f.fieldWidth

/-- Calculates the cost of constructing the path given the cost per square meter -/
def costOfPath (f : FieldWithPath) (costPerSqm : ℝ) : ℝ :=
  areaOfPath f * costPerSqm

/-- Theorem stating the area of the path and its construction cost for the given field dimensions -/
theorem path_area_and_cost (f : FieldWithPath) (h1 : f.fieldLength = 65) (h2 : f.fieldWidth = 55) 
    (h3 : f.pathWidth = 2.5) (h4 : costPerSqm = 2) : 
    areaOfPath f = 625 ∧ costOfPath f costPerSqm = 1250 := by
  sorry

end path_area_and_cost_l2789_278998


namespace alien_energy_conversion_l2789_278913

def base5_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem alien_energy_conversion :
  base5_to_base10 [0, 2, 3] = 85 := by
  sorry

end alien_energy_conversion_l2789_278913


namespace bonus_distribution_l2789_278990

theorem bonus_distribution (total_bonus : ℕ) (difference : ℕ) (junior_share : ℕ) : 
  total_bonus = 5000 →
  difference = 1200 →
  junior_share + (junior_share + difference) = total_bonus →
  junior_share = 1900 := by
sorry

end bonus_distribution_l2789_278990


namespace smallest_sum_of_consecutive_multiples_l2789_278959

theorem smallest_sum_of_consecutive_multiples : ∃ (a b c : ℕ),
  (b = a + 1) ∧
  (c = a + 2) ∧
  (a % 9 = 0) ∧
  (b % 8 = 0) ∧
  (c % 7 = 0) ∧
  (a + b + c = 1488) ∧
  (∀ (x y z : ℕ), (y = x + 1) ∧ (z = x + 2) ∧ (x % 9 = 0) ∧ (y % 8 = 0) ∧ (z % 7 = 0) → (x + y + z ≥ 1488)) :=
by sorry

end smallest_sum_of_consecutive_multiples_l2789_278959


namespace perpendicular_lines_n_value_l2789_278968

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem perpendicular_lines_n_value (m n : ℝ) (p : ℝ) :
  let l₁ : Line := ⟨m, 4, -2⟩
  let l₂ : Line := ⟨2, -5, n⟩
  let foot : Point := ⟨1, p⟩
  perpendicular (m / -4) (2 / 5) →
  point_on_line foot l₁ →
  point_on_line foot l₂ →
  n = -12 := by
  sorry

#check perpendicular_lines_n_value

end perpendicular_lines_n_value_l2789_278968


namespace sequence_max_value_l2789_278924

theorem sequence_max_value (n : ℕ+) : 
  let a := λ (k : ℕ+) => (k : ℝ) / ((k : ℝ)^2 + 6)
  (∀ k : ℕ+, a k ≤ 1/5) ∧ (∃ k : ℕ+, a k = 1/5) :=
sorry

end sequence_max_value_l2789_278924


namespace min_total_cost_l2789_278995

/-- Represents the number of book corners of each size -/
structure BookCorners where
  medium : ℕ
  small : ℕ

/-- Calculates the total cost for a given configuration of book corners -/
def total_cost (corners : BookCorners) : ℕ :=
  860 * corners.medium + 570 * corners.small

/-- Checks if a configuration of book corners is valid according to the given constraints -/
def is_valid_configuration (corners : BookCorners) : Prop :=
  corners.medium + corners.small = 30 ∧
  80 * corners.medium + 30 * corners.small ≤ 1900 ∧
  50 * corners.medium + 60 * corners.small ≤ 1620

/-- Theorem stating that the minimum total cost is 22320 yuan -/
theorem min_total_cost :
  ∃ (corners : BookCorners),
    is_valid_configuration corners ∧
    total_cost corners = 22320 ∧
    ∀ (other : BookCorners), is_valid_configuration other → total_cost other ≥ 22320 := by
  sorry

end min_total_cost_l2789_278995


namespace final_racers_count_l2789_278989

def race_elimination (initial_racers : ℕ) : ℕ :=
  let after_first := initial_racers - 10
  let after_second := after_first - (after_first / 3)
  let after_third := after_second - (after_second / 2)
  after_third

theorem final_racers_count :
  race_elimination 100 = 30 := by sorry

end final_racers_count_l2789_278989


namespace cos_double_angle_special_l2789_278928

/-- Given an angle θ formed by the positive x-axis and a line passing through
    the origin and the point (-3, 4), prove that cos(2θ) = -7/25 -/
theorem cos_double_angle_special (θ : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos θ = -3 ∧ r * Real.sin θ = 4) → 
  Real.cos (2 * θ) = -7/25 := by
sorry

end cos_double_angle_special_l2789_278928


namespace final_cell_count_l2789_278965

/-- Calculates the number of cells after a given number of days, 
    given an initial population and a tripling period. -/
def cell_count (initial_cells : ℕ) (tripling_period : ℕ) (total_days : ℕ) : ℕ :=
  initial_cells * (3 ^ (total_days / tripling_period))

/-- Theorem stating that given 5 initial cells, tripling every 3 days for 9 days, 
    the final cell count is 135. -/
theorem final_cell_count : cell_count 5 3 9 = 135 := by
  sorry

#eval cell_count 5 3 9

end final_cell_count_l2789_278965


namespace inverse_proposition_l2789_278915

theorem inverse_proposition (a b : ℝ) :
  (∀ x y : ℝ, (|x| > |y| → x > y)) →
  (a > b → |a| > |b|) :=
sorry

end inverse_proposition_l2789_278915


namespace bus_passengers_l2789_278963

theorem bus_passengers (initial : ℕ) (difference : ℕ) (final : ℕ) : 
  initial = 38 → difference = 9 → final = initial - difference → final = 29 := by sorry

end bus_passengers_l2789_278963


namespace polynomial_expansion_equality_l2789_278922

theorem polynomial_expansion_equality (x y : ℤ) :
  5 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 4 * (x + y)^2 =
  5*x^4 + 35*x^3 + 960*x^2 + 1649*x + 4000 - 8*x*y - 4*y^2 :=
by sorry

end polynomial_expansion_equality_l2789_278922


namespace sin_cos_bound_l2789_278970

theorem sin_cos_bound (t : ℝ) : -5 ≤ 4 * Real.sin t + Real.cos (2 * t) ∧ 4 * Real.sin t + Real.cos (2 * t) ≤ 3 := by
  sorry

end sin_cos_bound_l2789_278970


namespace x_twenty_percent_greater_than_52_l2789_278930

theorem x_twenty_percent_greater_than_52 (x : ℝ) : 
  x = 52 * (1 + 20 / 100) → x = 62.4 := by
sorry

end x_twenty_percent_greater_than_52_l2789_278930


namespace divisible_by_twelve_l2789_278992

/-- The function that constructs the number 534n given n -/
def number (n : ℕ) : ℕ := 5340 + n

/-- Predicate to check if a number is four-digit -/
def is_four_digit (x : ℕ) : Prop := 1000 ≤ x ∧ x < 10000

theorem divisible_by_twelve (n : ℕ) : 
  (is_four_digit (number n)) → 
  (n < 10) → 
  ((number n) % 12 = 0 ↔ n = 0) := by
  sorry

end divisible_by_twelve_l2789_278992


namespace age_ratio_in_two_years_l2789_278994

def maya_age : ℕ := 15
def drew_age : ℕ := maya_age + 5
def peter_age : ℕ := drew_age + 4
def john_age : ℕ := 30
def jacob_age : ℕ := 11

theorem age_ratio_in_two_years :
  (jacob_age + 2) / (peter_age + 2) = 1 / 2 :=
by sorry

end age_ratio_in_two_years_l2789_278994


namespace event_attendees_l2789_278973

theorem event_attendees (num_children : ℕ) (num_adults : ℕ) : 
  num_children = 28 → 
  num_children = 2 * num_adults → 
  num_children + num_adults = 42 := by
  sorry

end event_attendees_l2789_278973


namespace points_per_enemy_is_10_l2789_278985

/-- The number of points for killing one enemy in Tom's game -/
def points_per_enemy : ℕ := sorry

/-- The number of enemies Tom killed -/
def enemies_killed : ℕ := 150

/-- Tom's total score -/
def total_score : ℕ := 2250

/-- The bonus multiplier for killing at least 100 enemies -/
def bonus_multiplier : ℚ := 1.5

theorem points_per_enemy_is_10 :
  points_per_enemy = 10 ∧
  enemies_killed ≥ 100 ∧
  (points_per_enemy * enemies_killed : ℚ) * bonus_multiplier = total_score := by
  sorry

end points_per_enemy_is_10_l2789_278985


namespace daily_savings_amount_l2789_278919

/-- Proves that saving the same amount daily for 20 days totaling 2 dimes equals 1 cent per day -/
theorem daily_savings_amount (savings_period : ℕ) (total_saved : ℕ) (daily_amount : ℚ) : 
  savings_period = 20 →
  total_saved = 20 →  -- 2 dimes = 20 cents
  daily_amount * savings_period = total_saved →
  daily_amount = 1 := by
sorry

end daily_savings_amount_l2789_278919


namespace distance_between_points_l2789_278945

/-- The distance between two points (-3, -4) and (5, 6) is 2√41 -/
theorem distance_between_points : 
  let a : ℝ × ℝ := (-3, -4)
  let b : ℝ × ℝ := (5, 6)
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2) = 2 * Real.sqrt 41 := by
  sorry

end distance_between_points_l2789_278945


namespace jessica_calculation_l2789_278940

theorem jessica_calculation (y : ℝ) : (y - 8) / 4 = 22 → (y - 4) / 8 = 11.5 := by
  sorry

end jessica_calculation_l2789_278940


namespace division_remainder_l2789_278909

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 11 →
  divisor = 4 →
  quotient = 2 →
  dividend = divisor * quotient + remainder →
  remainder = 3 := by
sorry

end division_remainder_l2789_278909


namespace total_marbles_l2789_278906

def marbles_bought : ℝ := 5423.6
def marbles_before : ℝ := 12834.9

theorem total_marbles :
  marbles_bought + marbles_before = 18258.5 := by
  sorry

end total_marbles_l2789_278906


namespace isosceles_triangle_base_angle_l2789_278947

-- Define an isosceles triangle with one angle of 80 degrees
structure IsoscelesTriangle :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)
  (is_isosceles : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3))
  (has_80_degree : angle1 = 80 ∨ angle2 = 80 ∨ angle3 = 80)
  (sum_180 : angle1 + angle2 + angle3 = 180)

-- Theorem statement
theorem isosceles_triangle_base_angle (t : IsoscelesTriangle) :
  (t.angle1 = 50 ∨ t.angle1 = 80) ∨
  (t.angle2 = 50 ∨ t.angle2 = 80) ∨
  (t.angle3 = 50 ∨ t.angle3 = 80) :=
sorry

end isosceles_triangle_base_angle_l2789_278947


namespace jenny_calculation_l2789_278991

theorem jenny_calculation (x : ℚ) : (x - 14) / 5 = 11 → (x - 5) / 7 = 64/7 := by
  sorry

end jenny_calculation_l2789_278991


namespace inverse_function_point_l2789_278969

/-- Given a function f(x) = 2^x + m, prove that if its inverse passes through (3,1), then m = 1 -/
theorem inverse_function_point (m : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = 2^x + m) ∧ (∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ g 3 = 1)) → 
  m = 1 := by
  sorry

end inverse_function_point_l2789_278969


namespace ab_value_l2789_278942

theorem ab_value (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 80) : a * b = 32 := by
  sorry

end ab_value_l2789_278942


namespace harold_marbles_l2789_278960

/-- Given that Harold had 100 marbles, shared them evenly among 5 friends,
    and each friend received 16 marbles, prove that Harold kept 20 marbles. -/
theorem harold_marbles :
  ∀ (total_marbles friends_count marbles_per_friend marbles_kept : ℕ),
    total_marbles = 100 →
    friends_count = 5 →
    marbles_per_friend = 16 →
    marbles_kept + (friends_count * marbles_per_friend) = total_marbles →
    marbles_kept = 20 :=
by sorry

end harold_marbles_l2789_278960


namespace boxes_with_neither_l2789_278938

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (crayons : ℕ) (both : ℕ)
  (h1 : total = 15)
  (h2 : markers = 10)
  (h3 : crayons = 8)
  (h4 : both = 4) :
  total - (markers + crayons - both) = 1 := by
  sorry

end boxes_with_neither_l2789_278938


namespace maxwell_brad_meeting_time_l2789_278997

/-- The time it takes for Maxwell and Brad to meet, given their speeds and the distance between their homes. -/
theorem maxwell_brad_meeting_time 
  (distance : ℝ) 
  (maxwell_speed : ℝ) 
  (brad_speed : ℝ) 
  (head_start : ℝ) 
  (h1 : distance = 54) 
  (h2 : maxwell_speed = 4) 
  (h3 : brad_speed = 6) 
  (h4 : head_start = 1) :
  ∃ (t : ℝ), t + head_start = 6 ∧ 
  maxwell_speed * (t + head_start) + brad_speed * t = distance :=
sorry

end maxwell_brad_meeting_time_l2789_278997


namespace lighter_cost_difference_l2789_278908

/-- Calculates the cost of buying lighters at the gas station with a "buy 4 get 1 free" offer -/
def gas_station_cost (price_per_lighter : ℚ) (num_lighters : ℕ) : ℚ :=
  let sets := (num_lighters + 4) / 5
  let lighters_to_pay := sets * 4
  lighters_to_pay * price_per_lighter

/-- Calculates the cost of buying lighters on Amazon including tax and shipping -/
def amazon_cost (price_per_pack : ℚ) (lighters_per_pack : ℕ) (num_lighters : ℕ) 
                (tax_rate : ℚ) (shipping_cost : ℚ) : ℚ :=
  let packs_needed := (num_lighters + lighters_per_pack - 1) / lighters_per_pack
  let subtotal := packs_needed * price_per_pack
  let tax := subtotal * tax_rate
  subtotal + tax + shipping_cost

theorem lighter_cost_difference : 
  gas_station_cost (175/100) 24 - amazon_cost 5 12 24 (5/100) (7/2) = 1925/100 := by
  sorry

end lighter_cost_difference_l2789_278908


namespace canoe_production_sum_l2789_278948

theorem canoe_production_sum : ∀ (a₁ r n : ℕ), 
  a₁ = 5 → r = 3 → n = 4 → 
  a₁ * (r^n - 1) / (r - 1) = 200 := by sorry

end canoe_production_sum_l2789_278948


namespace tangent_line_at_one_l2789_278932

noncomputable def f (x : ℝ) := x^4 - 2*x^3

theorem tangent_line_at_one (x : ℝ) : 
  let p := (1, f 1)
  let m := deriv f 1
  (fun x => m * (x - p.1) + p.2) = (fun x => -2 * x + 1) :=
by sorry

end tangent_line_at_one_l2789_278932


namespace partial_fraction_sum_zero_l2789_278925

theorem partial_fraction_sum_zero (A B C D E F : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 := by
sorry

end partial_fraction_sum_zero_l2789_278925


namespace factorial_315_trailing_zeros_l2789_278977

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- The factorial of 315 ends with 77 zeros -/
theorem factorial_315_trailing_zeros :
  trailingZeros 315 = 77 := by
  sorry

end factorial_315_trailing_zeros_l2789_278977


namespace symmetric_point_origin_specific_symmetric_point_l2789_278978

def symmetric_point (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem symmetric_point_origin (x y : ℝ) : 
  symmetric_point x y = (-x, -y) := by sorry

theorem specific_symmetric_point : 
  symmetric_point (-2) 5 = (2, -5) := by sorry

end symmetric_point_origin_specific_symmetric_point_l2789_278978


namespace ram_birthday_is_19th_l2789_278912

/-- Represents the number of languages learned per day -/
def languages_per_day : ℕ := sorry

/-- Represents the number of languages known on the first day of the month -/
def languages_first_day : ℕ := 820

/-- Represents the number of languages known on the last day of the month -/
def languages_last_day : ℕ := 1100

/-- Represents the number of languages known on the birthday -/
def languages_birthday : ℕ := 1000

/-- Represents the day of the month on which the birthday falls -/
def birthday : ℕ := sorry

theorem ram_birthday_is_19th : 
  birthday = 19 ∧
  languages_per_day * (birthday - 1) + languages_first_day = languages_birthday ∧
  languages_per_day * (31 - 1) + languages_first_day = languages_last_day :=
sorry

end ram_birthday_is_19th_l2789_278912


namespace sum_of_digits_of_power_l2789_278903

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Define a function to get the ones digit of a natural number
def ones_digit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem sum_of_digits_of_power : 
  tens_digit ((3 + 4)^17) + ones_digit ((3 + 4)^17) = 7 := by
  sorry

end sum_of_digits_of_power_l2789_278903


namespace root_relation_l2789_278907

theorem root_relation (a b c : ℂ) (p q u : ℂ) : 
  (a^3 + 2*a^2 + 5*a - 8 = 0) → 
  (b^3 + 2*b^2 + 5*b - 8 = 0) → 
  (c^3 + 2*c^2 + 5*c - 8 = 0) → 
  ((a+b)^3 + p*(a+b)^2 + q*(a+b) + u = 0) → 
  ((b+c)^3 + p*(b+c)^2 + q*(b+c) + u = 0) → 
  ((c+a)^3 + p*(c+a)^2 + q*(c+a) + u = 0) → 
  u = 18 := by
sorry

end root_relation_l2789_278907


namespace mrs_heine_dogs_l2789_278933

/-- Given that Mrs. Heine buys 3 heart biscuits for each dog and needs to buy 6 biscuits in total,
    prove that she has 2 dogs. -/
theorem mrs_heine_dogs :
  ∀ (total_biscuits biscuits_per_dog : ℕ),
    total_biscuits = 6 →
    biscuits_per_dog = 3 →
    total_biscuits / biscuits_per_dog = 2 :=
by sorry

end mrs_heine_dogs_l2789_278933


namespace book_distribution_l2789_278902

theorem book_distribution (total : ℕ) (books_A books_B : ℕ) : 
  total = 282 → 
  4 * books_A = 3 * total → 
  9 * books_B = 5 * total → 
  books_A + books_B = total → 
  books_A = 120 ∧ books_B = 162 := by
  sorry

end book_distribution_l2789_278902


namespace joan_change_theorem_l2789_278921

def change_received (cat_toy_cost cage_cost amount_paid : ℚ) : ℚ :=
  amount_paid - (cat_toy_cost + cage_cost)

theorem joan_change_theorem (cat_toy_cost cage_cost amount_paid : ℚ) 
  (h1 : cat_toy_cost = 8.77)
  (h2 : cage_cost = 10.97)
  (h3 : amount_paid = 20) :
  change_received cat_toy_cost cage_cost amount_paid = 0.26 := by
  sorry

#eval change_received 8.77 10.97 20

end joan_change_theorem_l2789_278921


namespace distance_to_xy_plane_l2789_278944

/-- The distance from a point (3, 2, -5) to the xy-plane is 5. -/
theorem distance_to_xy_plane : 
  let p : ℝ × ℝ × ℝ := (3, 2, -5)
  abs (p.2) = 5 := by sorry

end distance_to_xy_plane_l2789_278944


namespace remainder_of_2543_base12_div_7_l2789_278983

/-- Converts a base-12 number to decimal --/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ i)) 0

/-- The base-12 representation of 2543₁₂ --/
def number : List Nat := [3, 4, 5, 2]

theorem remainder_of_2543_base12_div_7 :
  (base12ToDecimal number) % 7 = 6 := by
  sorry

end remainder_of_2543_base12_div_7_l2789_278983


namespace certain_number_proof_l2789_278950

theorem certain_number_proof : 
  ∃ x : ℝ, 0.8 * x = (4 / 5) * 25 + 16 ∧ x = 45 := by
sorry

end certain_number_proof_l2789_278950


namespace bird_difference_l2789_278999

/-- Proves the difference between white birds and original grey birds -/
theorem bird_difference (initial_grey : ℕ) (total_remaining : ℕ) 
  (h1 : initial_grey = 40)
  (h2 : total_remaining = 66) :
  total_remaining - initial_grey / 2 - initial_grey = 6 := by
  sorry

end bird_difference_l2789_278999


namespace interest_rate_problem_l2789_278966

/-- Calculates simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_problem (principal : ℝ) (time : ℝ) (rate : ℝ) 
    (h1 : principal = 15000)
    (h2 : time = 2)
    (h3 : simpleInterest principal rate time = simpleInterest principal 0.12 time + 900) :
  rate = 0.15 := by
  sorry

end interest_rate_problem_l2789_278966


namespace min_value_collinear_points_l2789_278929

/-- Given points A(3,-1), B(x,y), and C(0,1) are collinear, and x > 0, y > 0, 
    the minimum value of (3/x + 2/y) is 8 -/
theorem min_value_collinear_points (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : (y + 1) / (x - 3) = 2 / (-3)) : 
  (∀ a b : ℝ, a > 0 → b > 0 → (y + 1) / (x - 3) = 2 / (-3) → 3 / x + 2 / y ≤ 3 / a + 2 / b) → 
  3 / x + 2 / y = 8 := by
sorry

end min_value_collinear_points_l2789_278929


namespace quadratic_inequality_requires_conditional_branch_l2789_278926

-- Define the type for algorithms
inductive Algorithm
  | ProductOfTwoNumbers
  | DistancePointToLine
  | SolveQuadraticInequality
  | AreaOfTrapezoid

-- Define a function to check if an algorithm requires a conditional branch
def requiresConditionalBranch (a : Algorithm) : Prop :=
  match a with
  | Algorithm.SolveQuadraticInequality => True
  | _ => False

-- State the theorem
theorem quadratic_inequality_requires_conditional_branch :
  ∀ (a : Algorithm),
    requiresConditionalBranch a ↔ a = Algorithm.SolveQuadraticInequality :=
by sorry

#check quadratic_inequality_requires_conditional_branch

end quadratic_inequality_requires_conditional_branch_l2789_278926


namespace min_value_theorem_min_value_achievable_l2789_278951

theorem min_value_theorem (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  4 * a^3 + 8 * b^3 + 27 * c^3 + 64 * d^3 + 2 / (a * b * c * d) ≥ 16 * Real.sqrt 3 :=
by
  sorry

theorem min_value_achievable :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    4 * a^3 + 8 * b^3 + 27 * c^3 + 64 * d^3 + 2 / (a * b * c * d) = 16 * Real.sqrt 3 :=
by
  sorry

end min_value_theorem_min_value_achievable_l2789_278951


namespace polynomial_division_remainder_l2789_278981

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^6 + X^5 + 2*X^3 - X^2 + 3 = (X + 2) * (X - 1) * q + (-X + 5) := by
  sorry

end polynomial_division_remainder_l2789_278981


namespace choose_four_from_ten_l2789_278984

theorem choose_four_from_ten (n : ℕ) (k : ℕ) : n = 10 ∧ k = 4 → Nat.choose n k = 210 := by
  sorry

end choose_four_from_ten_l2789_278984


namespace coin_toss_sequences_coin_toss_theorem_l2789_278937

/-- The number of ways to place n indistinguishable balls into k distinguishable urns -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) n

/-- The number of different sequences of 20 coin tosses with specific subsequence counts -/
theorem coin_toss_sequences : ℕ := 
  let hh_placements := stars_and_bars 3 4
  let tt_placements := stars_and_bars 7 5
  hh_placements * tt_placements

/-- The main theorem stating the number of valid sequences -/
theorem coin_toss_theorem : coin_toss_sequences = 6600 := by sorry

end coin_toss_sequences_coin_toss_theorem_l2789_278937


namespace sum_of_even_coefficients_zero_l2789_278972

theorem sum_of_even_coefficients_zero (a : Fin 7 → ℝ) :
  (∀ x : ℝ, (x - 1)^6 = a 0 * x^6 + a 1 * x^5 + a 2 * x^4 + a 3 * x^3 + a 4 * x^2 + a 5 * x + a 6) →
  a 0 + a 2 + a 4 + a 6 = 0 := by
sorry

end sum_of_even_coefficients_zero_l2789_278972


namespace triangle_area_l2789_278916

/-- Prove that the area of the triangle formed by the lines x = -5, y = x, and the x-axis is 12.5 -/
theorem triangle_area : 
  let line1 : ℝ → ℝ := λ x => -5
  let line2 : ℝ → ℝ := λ x => x
  let intersection_x : ℝ := -5
  let intersection_y : ℝ := line2 intersection_x
  let base : ℝ := abs intersection_x
  let height : ℝ := abs intersection_y
  (1/2) * base * height = 12.5 := by sorry

end triangle_area_l2789_278916
