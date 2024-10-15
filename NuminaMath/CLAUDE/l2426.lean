import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_product_l2426_242693

theorem polynomial_product (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a + a₂ + a₄) * (a₁ + a₃ + a₅) = -256 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_l2426_242693


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l2426_242636

/-- The product of two specific repeating decimals -/
theorem product_of_repeating_decimals :
  (63 : ℚ) / 99 * (54 : ℚ) / 99 = (14 : ℚ) / 41 := by
  sorry

#check product_of_repeating_decimals

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l2426_242636


namespace NUMINAMATH_CALUDE_phi_value_l2426_242677

theorem phi_value : ∃ (Φ : ℕ), Φ < 10 ∧ (220 : ℚ) / Φ = 40 + 3 * Φ := by
  sorry

end NUMINAMATH_CALUDE_phi_value_l2426_242677


namespace NUMINAMATH_CALUDE_foci_of_hyperbola_l2426_242678

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := 4 * y^2 - 25 * x^2 = 100

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(0, -Real.sqrt 29), (0, Real.sqrt 29)}

/-- Theorem: The given coordinates are the foci of the hyperbola -/
theorem foci_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ foci_coordinates :=
sorry

end NUMINAMATH_CALUDE_foci_of_hyperbola_l2426_242678


namespace NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l2426_242635

theorem sum_of_roots_squared_equation (x : ℝ) : 
  (∃ a b : ℝ, (a - 7)^2 = 16 ∧ (b - 7)^2 = 16 ∧ a + b = 14) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l2426_242635


namespace NUMINAMATH_CALUDE_pump_rates_determination_l2426_242687

/-- Represents the pumping rates and durations of three water pumps -/
structure PumpSystem where
  rate1 : ℝ  -- Pumping rate of the first pump
  rate2 : ℝ  -- Pumping rate of the second pump
  rate3 : ℝ  -- Pumping rate of the third pump
  time1 : ℝ  -- Working time of the first pump
  time2 : ℝ  -- Working time of the second pump
  time3 : ℝ  -- Working time of the third pump

/-- Checks if the given pump system satisfies all the conditions -/
def satisfiesConditions (p : PumpSystem) : Prop :=
  p.time1 = p.time3 ∧  -- First and third pumps finish simultaneously
  p.time2 = 2 ∧  -- Second pump works for 2 hours
  p.rate1 * p.time1 = 9 ∧  -- First pump pumps 9 m³
  p.rate2 * p.time2 + p.rate3 * p.time3 = 28 ∧  -- Second and third pumps pump 28 m³ together
  p.rate3 = p.rate1 + 3 ∧  -- Third pump pumps 3 m³ more per hour than the first
  p.rate1 + p.rate2 + p.rate3 = 14  -- Three pumps together pump 14 m³ per hour

/-- Theorem stating that the given conditions imply specific pumping rates -/
theorem pump_rates_determination (p : PumpSystem) 
  (h : satisfiesConditions p) : p.rate1 = 3 ∧ p.rate2 = 5 ∧ p.rate3 = 6 := by
  sorry


end NUMINAMATH_CALUDE_pump_rates_determination_l2426_242687


namespace NUMINAMATH_CALUDE_max_distance_complex_circle_l2426_242633

theorem max_distance_complex_circle (z : ℂ) (z₀ : ℂ) :
  z₀ = 1 - 2*I →
  Complex.abs z = 3 →
  ∃ (max_dist : ℝ), max_dist = 3 + Real.sqrt 5 ∧
    ∀ (w : ℂ), Complex.abs w = 3 → Complex.abs (w - z₀) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_circle_l2426_242633


namespace NUMINAMATH_CALUDE_like_terms_exponent_l2426_242672

/-- 
Given two terms -3x^(2m)y^3 and 2x^4y^n are like terms,
prove that m^n = 8
-/
theorem like_terms_exponent (m n : ℕ) : 
  (∃ (x y : ℝ), -3 * x^(2*m) * y^3 = 2 * x^4 * y^n) → m^n = 8 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponent_l2426_242672


namespace NUMINAMATH_CALUDE_pizza_order_l2426_242634

theorem pizza_order (num_people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_people = 10)
  (h2 : slices_per_person = 2)
  (h3 : slices_per_pizza = 4) :
  (num_people * slices_per_person + slices_per_pizza - 1) / slices_per_pizza = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_l2426_242634


namespace NUMINAMATH_CALUDE_existence_of_non_divisible_pair_l2426_242613

theorem existence_of_non_divisible_pair (p : Nat) (h_prime : Prime p) (h_p_gt_3 : p > 3) :
  ∃ n : Nat, n > 0 ∧ n < p - 1 ∧
    ¬(p^2 ∣ n^(p-1) - 1) ∧ ¬(p^2 ∣ (n+1)^(p-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_divisible_pair_l2426_242613


namespace NUMINAMATH_CALUDE_womens_average_age_l2426_242650

/-- Represents the problem of finding the average age of two women -/
theorem womens_average_age 
  (n : ℕ) 
  (initial_total_age : ℝ) 
  (age_increase : ℝ) 
  (man1_age man2_age : ℝ) :
  n = 10 →
  age_increase = 6 →
  man1_age = 18 →
  man2_age = 22 →
  (initial_total_age / n + age_increase) * n = initial_total_age - man1_age - man2_age + 2 * ((initial_total_age / n + age_increase) * n - initial_total_age + man1_age + man2_age) / 2 →
  ((initial_total_age / n + age_increase) * n - initial_total_age + man1_age + man2_age) / 2 = 50 :=
by sorry

end NUMINAMATH_CALUDE_womens_average_age_l2426_242650


namespace NUMINAMATH_CALUDE_point_on_circle_x_value_l2426_242615

-- Define the circle
def circle_center : ℝ × ℝ := (12, 0)
def circle_radius : ℝ := 15

-- Define the point on the circle
def point_on_circle (x : ℝ) : ℝ × ℝ := (x, 12)

-- Theorem statement
theorem point_on_circle_x_value (x : ℝ) :
  (point_on_circle x).1 - circle_center.1 ^ 2 + 
  (point_on_circle x).2 - circle_center.2 ^ 2 = circle_radius ^ 2 →
  x = 3 ∨ x = 21 := by
  sorry

end NUMINAMATH_CALUDE_point_on_circle_x_value_l2426_242615


namespace NUMINAMATH_CALUDE_product_of_integers_with_given_sum_and_difference_l2426_242679

theorem product_of_integers_with_given_sum_and_difference :
  ∀ x y : ℕ+, 
    (x : ℤ) + (y : ℤ) = 72 → 
    (x : ℤ) - (y : ℤ) = 18 → 
    (x : ℤ) * (y : ℤ) = 1215 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_with_given_sum_and_difference_l2426_242679


namespace NUMINAMATH_CALUDE_sales_decrease_equation_l2426_242694

/-- Represents the monthly decrease rate as a real number between 0 and 1 -/
def monthly_decrease_rate : ℝ := sorry

/-- The initial sales amount in August -/
def initial_sales : ℝ := 42

/-- The final sales amount in October -/
def final_sales : ℝ := 27

/-- The number of months between August and October -/
def months_elapsed : ℕ := 2

theorem sales_decrease_equation :
  initial_sales * (1 - monthly_decrease_rate) ^ months_elapsed = final_sales :=
sorry

end NUMINAMATH_CALUDE_sales_decrease_equation_l2426_242694


namespace NUMINAMATH_CALUDE_leftover_value_is_five_fifty_l2426_242688

/-- Represents the number of coins in a roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents the number of coins a person has --/
structure CoinCount where
  quarters : Nat
  dimes : Nat

/-- Calculates the value of leftover coins in dollars --/
def leftoverValue (rollSize : RollSize) (james : CoinCount) (lindsay : CoinCount) : ℚ :=
  let totalQuarters := james.quarters + lindsay.quarters
  let totalDimes := james.dimes + lindsay.dimes
  let leftoverQuarters := totalQuarters % rollSize.quarters
  let leftoverDimes := totalDimes % rollSize.dimes
  (leftoverQuarters : ℚ) * (1 / 4) + (leftoverDimes : ℚ) * (1 / 10)

theorem leftover_value_is_five_fifty :
  let rollSize : RollSize := { quarters := 40, dimes := 50 }
  let james : CoinCount := { quarters := 83, dimes := 159 }
  let lindsay : CoinCount := { quarters := 129, dimes := 266 }
  leftoverValue rollSize james lindsay = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_leftover_value_is_five_fifty_l2426_242688


namespace NUMINAMATH_CALUDE_abs_difference_equals_sum_of_abs_l2426_242664

theorem abs_difference_equals_sum_of_abs (a b c : ℚ) 
  (h1 : a < b) (h2 : b < 0) (h3 : 0 < c) : 
  |a - c| = |a| + c := by sorry

end NUMINAMATH_CALUDE_abs_difference_equals_sum_of_abs_l2426_242664


namespace NUMINAMATH_CALUDE_max_piles_660_max_piles_optimal_l2426_242690

/-- The maximum number of piles that can be created from a given number of stones,
    where any two piles differ by strictly less than 2 times. -/
def maxPiles (totalStones : ℕ) : ℕ :=
  30  -- The actual implementation is not provided, just the result

theorem max_piles_660 :
  maxPiles 660 = 30 := by sorry

/-- A function to check if two pile sizes are similar (differ by strictly less than 2 times) -/
def areSimilarSizes (a b : ℕ) : Prop :=
  a < 2 * b ∧ b < 2 * a

/-- A function to represent a valid distribution of stones into piles -/
def isValidDistribution (piles : List ℕ) (totalStones : ℕ) : Prop :=
  piles.sum = totalStones ∧
  ∀ (a b : ℕ), a ∈ piles → b ∈ piles → areSimilarSizes a b

theorem max_piles_optimal (piles : List ℕ) :
  isValidDistribution piles 660 →
  piles.length ≤ 30 := by sorry

end NUMINAMATH_CALUDE_max_piles_660_max_piles_optimal_l2426_242690


namespace NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_m_l2426_242692

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x - 3|

-- Theorem for the solution set of f(x) ≤ 6
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = Set.Icc (-1/2) (5/2) := by sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, 6*m^2 - 4*m < f x} = Set.Ioo (-1/3) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_m_l2426_242692


namespace NUMINAMATH_CALUDE_code_deciphering_probability_l2426_242602

theorem code_deciphering_probability 
  (p_a p_b : ℝ) 
  (h_a : p_a = 0.3) 
  (h_b : p_b = 0.3) 
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - p_a) * (1 - p_b) = 0.51 :=
sorry

end NUMINAMATH_CALUDE_code_deciphering_probability_l2426_242602


namespace NUMINAMATH_CALUDE_cubic_difference_l2426_242651

theorem cubic_difference (a b : ℝ) 
  (h1 : a - b = 7)
  (h2 : a^2 + b^2 = 65)
  (h3 : a + b = 6) :
  a^3 - b^3 = 432.25 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l2426_242651


namespace NUMINAMATH_CALUDE_new_person_weight_l2426_242621

def initial_persons : ℕ := 6
def average_weight_increase : ℝ := 2
def replaced_person_weight : ℝ := 75

theorem new_person_weight :
  ∃ (new_weight : ℝ),
    new_weight = replaced_person_weight + initial_persons * average_weight_increase :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2426_242621


namespace NUMINAMATH_CALUDE_vectors_collinear_l2426_242663

/-- Given vectors a and b in ℝ³, prove that c₁ and c₂ are collinear -/
theorem vectors_collinear (a b : Fin 3 → ℝ) 
  (ha : a = ![1, -2, 5])
  (hb : b = ![3, -1, 0])
  (c₁ : Fin 3 → ℝ) (hc₁ : c₁ = 4 • a - 2 • b)
  (c₂ : Fin 3 → ℝ) (hc₂ : c₂ = b - 2 • a) :
  ∃ k : ℝ, c₁ = k • c₂ := by
sorry

end NUMINAMATH_CALUDE_vectors_collinear_l2426_242663


namespace NUMINAMATH_CALUDE_fence_perimeter_l2426_242665

/-- The number of posts used to enclose the garden -/
def num_posts : ℕ := 36

/-- The width of each post in inches -/
def post_width_inches : ℕ := 3

/-- The space between adjacent posts in feet -/
def post_spacing_feet : ℕ := 4

/-- Conversion factor from inches to feet -/
def inches_to_feet : ℚ := 1 / 12

/-- The width of each post in feet -/
def post_width_feet : ℚ := post_width_inches * inches_to_feet

/-- The number of posts on each side of the square garden -/
def posts_per_side : ℕ := num_posts / 4 + 1

/-- The number of spaces between posts on each side -/
def spaces_per_side : ℕ := posts_per_side - 1

/-- The length of one side of the square garden in feet -/
def side_length : ℚ := posts_per_side * post_width_feet + spaces_per_side * post_spacing_feet

/-- The outer perimeter of the fence surrounding the square garden -/
def outer_perimeter : ℚ := 4 * side_length

/-- Theorem stating that the outer perimeter of the fence is 137 feet -/
theorem fence_perimeter : outer_perimeter = 137 := by
  sorry

end NUMINAMATH_CALUDE_fence_perimeter_l2426_242665


namespace NUMINAMATH_CALUDE_power_of_ten_problem_l2426_242691

theorem power_of_ten_problem (a b : ℝ) 
  (h1 : (40 : ℝ) ^ a = 5) 
  (h2 : (40 : ℝ) ^ b = 8) : 
  (10 : ℝ) ^ ((1 - a - b) / (2 * (1 - b))) = 1 := by
sorry

end NUMINAMATH_CALUDE_power_of_ten_problem_l2426_242691


namespace NUMINAMATH_CALUDE_infinite_primes_4k_plus_3_l2426_242695

theorem infinite_primes_4k_plus_3 :
  ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4*k + 3) →
  ∃ q, Nat.Prime q ∧ (∃ m, q = 4*m + 3) ∧ q ∉ S :=
by sorry

end NUMINAMATH_CALUDE_infinite_primes_4k_plus_3_l2426_242695


namespace NUMINAMATH_CALUDE_salmon_trip_count_l2426_242675

/-- The number of male salmon that returned to their rivers -/
def male_salmon : ℕ := 712261

/-- The number of female salmon that returned to their rivers -/
def female_salmon : ℕ := 259378

/-- The total number of salmon that made the trip -/
def total_salmon : ℕ := male_salmon + female_salmon

theorem salmon_trip_count : total_salmon = 971639 := by
  sorry

end NUMINAMATH_CALUDE_salmon_trip_count_l2426_242675


namespace NUMINAMATH_CALUDE_savings_multiple_l2426_242629

/-- Represents a worker's monthly finances -/
structure WorkerFinances where
  takehome : ℝ  -- Monthly take-home pay
  savingsRate : ℝ  -- Fraction of take-home pay saved each month
  months : ℕ  -- Number of months

/-- Calculates the total amount saved over a given number of months -/
def totalSaved (w : WorkerFinances) : ℝ :=
  w.takehome * w.savingsRate * w.months

/-- Calculates the amount not saved in one month -/
def monthlyUnsaved (w : WorkerFinances) : ℝ :=
  w.takehome * (1 - w.savingsRate)

/-- Theorem stating that for a worker saving 1/4 of their take-home pay,
    the total saved over 12 months is 4 times the monthly unsaved amount -/
theorem savings_multiple (w : WorkerFinances)
    (h1 : w.savingsRate = 1/4)
    (h2 : w.months = 12) :
    totalSaved w = 4 * monthlyUnsaved w := by
  sorry


end NUMINAMATH_CALUDE_savings_multiple_l2426_242629


namespace NUMINAMATH_CALUDE_f_1384_bounds_l2426_242620

/-- An n-mino is a shape made up of n equal squares connected edge-to-edge. -/
def Mino (n : ℕ) : Type := Unit  -- We don't need to define the full structure for this proof

/-- f(n) is the least number such that there exists an f(n)-mino containing every n-mino -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating the bounds for f(1384) -/
theorem f_1384_bounds : 10000 ≤ f 1384 ∧ f 1384 ≤ 960000 := by sorry

end NUMINAMATH_CALUDE_f_1384_bounds_l2426_242620


namespace NUMINAMATH_CALUDE_max_M_min_N_equals_two_thirds_l2426_242680

theorem max_M_min_N_equals_two_thirds (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let M := x / (2 * x + y) + y / (x + 2 * y)
  let N := x / (x + 2 * y) + y / (2 * x + y)
  (∀ a b : ℝ, a > 0 → b > 0 → M ≤ (a / (2 * a + b) + b / (a + 2 * b))) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → N ≥ (a / (a + 2 * b) + b / (2 * a + b))) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ M = 2/3) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ N = 2/3) :=
by sorry


end NUMINAMATH_CALUDE_max_M_min_N_equals_two_thirds_l2426_242680


namespace NUMINAMATH_CALUDE_slide_total_l2426_242676

theorem slide_total (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 22 → additional = 13 → total = initial + additional → total = 35 := by
  sorry

end NUMINAMATH_CALUDE_slide_total_l2426_242676


namespace NUMINAMATH_CALUDE_nail_fraction_sum_l2426_242671

theorem nail_fraction_sum : 
  let size_2d : ℚ := 1/6
  let size_3d : ℚ := 2/15
  let size_4d : ℚ := 3/20
  let size_5d : ℚ := 1/10
  let size_6d : ℚ := 1/4
  let size_7d : ℚ := 1/12
  let size_8d : ℚ := 1/8
  let size_9d : ℚ := 1/30
  size_2d + size_3d + size_5d + size_8d = 21/40 := by
  sorry

end NUMINAMATH_CALUDE_nail_fraction_sum_l2426_242671


namespace NUMINAMATH_CALUDE_fraction_value_l2426_242616

theorem fraction_value (p q : ℚ) (h : p / q = 7) : (p + q) / (p - q) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2426_242616


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2426_242670

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + 2*b + 3*c = 4) :
  a^2 + b^2 + c^2 ≥ 8/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2426_242670


namespace NUMINAMATH_CALUDE_root_sum_product_theorem_l2426_242642

theorem root_sum_product_theorem (m : ℚ) :
  (∃ x y : ℚ, 
    (2*(x-1)*(x-3*m) = x*(m-4)) ∧ 
    (x + y = x * y) ∧
    (∀ z : ℚ, 2*z^2 + (5*m + 6)*z + 6*m = 0 ↔ (z = x ∨ z = y))) →
  m = -2/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_theorem_l2426_242642


namespace NUMINAMATH_CALUDE_percentage_reading_two_novels_l2426_242600

theorem percentage_reading_two_novels
  (total_students : ℕ)
  (three_or_more : ℚ)
  (one_novel : ℚ)
  (no_novels : ℕ)
  (h1 : total_students = 240)
  (h2 : three_or_more = 1 / 6)
  (h3 : one_novel = 5 / 12)
  (h4 : no_novels = 16) :
  (total_students - (three_or_more * total_students).num - (one_novel * total_students).num - no_novels : ℚ) / total_students * 100 = 35 := by
sorry


end NUMINAMATH_CALUDE_percentage_reading_two_novels_l2426_242600


namespace NUMINAMATH_CALUDE_four_boys_three_girls_144_arrangements_l2426_242658

/-- The number of ways to arrange alternating boys and girls in a row -/
def alternatingArrangements (boys girls : ℕ) : ℕ := boys.factorial * girls.factorial

/-- Theorem stating that if there are 3 girls and 144 alternating arrangements, there must be 4 boys -/
theorem four_boys_three_girls_144_arrangements :
  ∃ (boys : ℕ), boys > 0 ∧ alternatingArrangements boys 3 = 144 → boys = 4 := by
  sorry

#check four_boys_three_girls_144_arrangements

end NUMINAMATH_CALUDE_four_boys_three_girls_144_arrangements_l2426_242658


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l2426_242623

theorem gcd_from_lcm_and_ratio (X Y : ℕ+) :
  Nat.lcm X Y = 180 →
  (X : ℚ) / Y = 2 / 5 →
  Nat.gcd X Y = 18 := by
sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l2426_242623


namespace NUMINAMATH_CALUDE_poojas_speed_l2426_242618

/-- 
Given:
- Roja moves in the opposite direction from Pooja at 5 km/hr
- After 4 hours, the distance between Roja and Pooja is 32 km

Prove that Pooja's speed is 3 km/hr
-/
theorem poojas_speed (roja_speed : ℝ) (time : ℝ) (distance : ℝ) :
  roja_speed = 5 →
  time = 4 →
  distance = 32 →
  ∃ (pooja_speed : ℝ), pooja_speed = 3 ∧ distance = (roja_speed + pooja_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_poojas_speed_l2426_242618


namespace NUMINAMATH_CALUDE_apple_lemon_equivalence_l2426_242666

/-- Represents the value of fruits in terms of a common unit -/
structure FruitValue where
  apple : ℚ
  lemon : ℚ

/-- Given that 3/4 of 14 apples are worth 9 lemons, 
    prove that 5/7 of 7 apples are worth 30/7 lemons -/
theorem apple_lemon_equivalence (v : FruitValue) 
  (h : (3/4 : ℚ) * 14 * v.apple = 9 * v.lemon) :
  (5/7 : ℚ) * 7 * v.apple = (30/7 : ℚ) * v.lemon := by
  sorry

#check apple_lemon_equivalence

end NUMINAMATH_CALUDE_apple_lemon_equivalence_l2426_242666


namespace NUMINAMATH_CALUDE_calculate_expression_l2426_242661

theorem calculate_expression : -1^2 + 8 / (-2)^2 - (-4) * (-3) = -11 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2426_242661


namespace NUMINAMATH_CALUDE_sample_size_is_twenty_l2426_242606

/-- Represents the number of brands for each dairy product type -/
structure DairyBrands where
  pureMilk : ℕ
  yogurt : ℕ
  infantFormula : ℕ
  adultFormula : ℕ

/-- Represents the sample sizes for each dairy product type -/
structure SampleSizes where
  pureMilk : ℕ
  yogurt : ℕ
  infantFormula : ℕ
  adultFormula : ℕ

/-- Calculates the total sample size given the sample sizes for each product type -/
def totalSampleSize (s : SampleSizes) : ℕ :=
  s.pureMilk + s.yogurt + s.infantFormula + s.adultFormula

/-- Theorem stating that the total sample size is 20 given the problem conditions -/
theorem sample_size_is_twenty (brands : DairyBrands)
    (h1 : brands.pureMilk = 30)
    (h2 : brands.yogurt = 10)
    (h3 : brands.infantFormula = 35)
    (h4 : brands.adultFormula = 25)
    (sample : SampleSizes)
    (h5 : sample.infantFormula = 7)
    (h6 : sample.pureMilk * brands.infantFormula = brands.pureMilk * sample.infantFormula)
    (h7 : sample.yogurt * brands.infantFormula = brands.yogurt * sample.infantFormula)
    (h8 : sample.adultFormula * brands.infantFormula = brands.adultFormula * sample.infantFormula) :
  totalSampleSize sample = 20 := by
  sorry


end NUMINAMATH_CALUDE_sample_size_is_twenty_l2426_242606


namespace NUMINAMATH_CALUDE_circle_pi_value_l2426_242648

theorem circle_pi_value (d c : ℝ) (hd : d = 8) (hc : c = 25.12) :
  c / d = 3.14 := by sorry

end NUMINAMATH_CALUDE_circle_pi_value_l2426_242648


namespace NUMINAMATH_CALUDE_angle_conversion_negative_1125_conversion_l2426_242627

theorem angle_conversion (angle : ℝ) : ∃ (k : ℤ) (α : ℝ), 
  angle = k * 360 + α ∧ 0 ≤ α ∧ α < 360 :=
by
  -- Proof goes here
  sorry

theorem negative_1125_conversion : 
  ∃ (k : ℤ) (α : ℝ), -1125 = k * 360 + α ∧ 0 ≤ α ∧ α < 360 ∧ k = -4 ∧ α = 315 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_angle_conversion_negative_1125_conversion_l2426_242627


namespace NUMINAMATH_CALUDE_boy_running_duration_l2426_242640

theorem boy_running_duration (initial_speed initial_time second_distance second_speed : ℝ) 
  (h1 : initial_speed = 15)
  (h2 : initial_time = 3)
  (h3 : second_distance = 190)
  (h4 : second_speed = 19) : 
  initial_time + second_distance / second_speed = 13 := by
  sorry

end NUMINAMATH_CALUDE_boy_running_duration_l2426_242640


namespace NUMINAMATH_CALUDE_constant_term_value_l2426_242660

theorem constant_term_value (x y C : ℝ) 
  (eq1 : 7 * x + y = C)
  (eq2 : x + 3 * y = 1)
  (eq3 : 2 * x + y = 5) : 
  C = 19 := by
sorry

end NUMINAMATH_CALUDE_constant_term_value_l2426_242660


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l2426_242638

/-- Given a line with equation x - 2y + 1 = 0, its symmetric line with respect to the y-axis
    has the equation x + 2y - 1 = 0 -/
theorem symmetric_line_wrt_y_axis :
  ∀ (x y : ℝ), x - 2*y + 1 = 0 → ∃ (x' y' : ℝ), x' + 2*y' - 1 = 0 ∧ x' = -x ∧ y' = y :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l2426_242638


namespace NUMINAMATH_CALUDE_cone_volume_increase_l2426_242647

theorem cone_volume_increase (R H : ℝ) (hR : R = 5) (hH : H = 12) :
  ∃ y : ℝ, y > 0 ∧ (1 / 3) * π * (R + y)^2 * H = (1 / 3) * π * R^2 * (H + y) ∧ y = 31 / 12 :=
sorry

end NUMINAMATH_CALUDE_cone_volume_increase_l2426_242647


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2426_242628

theorem gcd_lcm_product (a b : ℕ) (ha : a = 24) (hb : b = 54) :
  Nat.gcd a b * Nat.lcm a b = a * b := by
sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2426_242628


namespace NUMINAMATH_CALUDE_chocolate_bars_original_count_l2426_242608

/-- The number of chocolate bars remaining after eating a certain percentage each day for a given number of days -/
def remaining_bars (initial : ℕ) (eat_percentage : ℚ) (days : ℕ) : ℚ :=
  initial * (1 - eat_percentage) ^ days

/-- The theorem stating the original number of chocolate bars given the remaining bars after 4 days -/
theorem chocolate_bars_original_count :
  ∃ (initial : ℕ),
    remaining_bars initial (30 / 100) 4 = 16 ∧
    initial = 67 :=
sorry

end NUMINAMATH_CALUDE_chocolate_bars_original_count_l2426_242608


namespace NUMINAMATH_CALUDE_x_value_proof_l2426_242612

theorem x_value_proof : 
  ∀ x : ℝ, x = 143 * (1 + 32.5 / 100) → x = 189.475 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2426_242612


namespace NUMINAMATH_CALUDE_inequality_proof_l2426_242652

theorem inequality_proof (x : ℝ) (h : x ≥ 4) :
  Real.sqrt (x - 3) - Real.sqrt (x - 1) > Real.sqrt (x - 4) - Real.sqrt (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2426_242652


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2426_242696

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  (l.a * x₀ + l.b * y₀ + l.c)^2 = (l.a^2 + l.b^2) * c.radius^2

theorem tangent_line_to_circle (c : Circle) (l : Line) (p : ℝ × ℝ) :
  c.center = (0, 0) ∧ c.radius = 5 ∧
  l.a = 3 ∧ l.b = 4 ∧ l.c = -25 ∧
  p = (3, 4) →
  is_tangent l c ∧ l.contains p :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2426_242696


namespace NUMINAMATH_CALUDE_expression_simplification_l2426_242632

theorem expression_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (x + a)^4 / ((a - b) * (a - c)) + (x + b)^4 / ((b - a) * (b - c)) + (x + c)^4 / ((c - a) * (c - b)) =
  a + b + c + 3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2426_242632


namespace NUMINAMATH_CALUDE_integral_square_le_four_integral_derivative_square_l2426_242699

open MeasureTheory Interval RealInnerProductSpace

theorem integral_square_le_four_integral_derivative_square
  (f : ℝ → ℝ) (hf : ContDiff ℝ 1 f) (h : ∃ x₀ ∈ Set.Icc 0 1, f x₀ = 0) :
  ∫ x in Set.Icc 0 1, (f x)^2 ≤ 4 * ∫ x in Set.Icc 0 1, (deriv f x)^2 :=
sorry

end NUMINAMATH_CALUDE_integral_square_le_four_integral_derivative_square_l2426_242699


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2426_242614

def A : Matrix (Fin 3) (Fin 3) ℤ := !![0, 1, 2; 1, 0, 1; 2, 1, 0]

theorem matrix_equation_solution :
  ∃ (p q r : ℤ), A^3 + p • A^2 + q • A + r • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 ∧ p = 0 ∧ q = -6 ∧ r = -4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2426_242614


namespace NUMINAMATH_CALUDE_range_of_a_l2426_242656

-- Define the conditions α and β
def α (x : ℝ) : Prop := x ≤ -1 ∨ x > 3
def β (a x : ℝ) : Prop := a - 1 ≤ x ∧ x < a + 2

-- State the theorem
theorem range_of_a :
  (∀ x, β a x → α x) ∧ 
  (∃ x, α x ∧ ¬β a x) →
  a ≤ -3 ∨ a > 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2426_242656


namespace NUMINAMATH_CALUDE_inverse_abs_equality_false_l2426_242625

theorem inverse_abs_equality_false : ¬ ∀ a b : ℝ, |a| = |b| → a = b := by
  sorry

end NUMINAMATH_CALUDE_inverse_abs_equality_false_l2426_242625


namespace NUMINAMATH_CALUDE_pizza_dough_production_l2426_242662

-- Define the given conditions
def batches_per_sack : ℕ := 15
def sacks_per_day : ℕ := 5
def days_per_week : ℕ := 7

-- Define the theorem to be proved
theorem pizza_dough_production :
  batches_per_sack * sacks_per_day * days_per_week = 525 := by
  sorry

end NUMINAMATH_CALUDE_pizza_dough_production_l2426_242662


namespace NUMINAMATH_CALUDE_prism_volume_l2426_242657

/-- The volume of a right rectangular prism with face areas 10, 15, and 18 square inches is 30√3 cubic inches. -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 10) 
  (h2 : b * c = 15) 
  (h3 : c * a = 18) : 
  a * b * c = 30 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_l2426_242657


namespace NUMINAMATH_CALUDE_tv_cost_is_1060_l2426_242610

-- Define the given values
def total_initial_purchase : ℝ := 3000
def returned_bike_cost : ℝ := 500
def toaster_cost : ℝ := 100
def total_out_of_pocket : ℝ := 2020

-- Define the TV cost as a variable
def tv_cost : ℝ := sorry

-- Define the sold bike cost
def sold_bike_cost : ℝ := returned_bike_cost * 1.2

-- Define the sale price of the sold bike
def sold_bike_sale_price : ℝ := sold_bike_cost * 0.8

-- Theorem stating that the TV cost is $1060
theorem tv_cost_is_1060 :
  tv_cost = 1060 :=
by
  sorry

#check tv_cost_is_1060

end NUMINAMATH_CALUDE_tv_cost_is_1060_l2426_242610


namespace NUMINAMATH_CALUDE_problem_statement_l2426_242654

theorem problem_statement (t : ℚ) (x y : ℚ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 5 * t + 6) 
  (h3 : x = -2) : 
  y = 37 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2426_242654


namespace NUMINAMATH_CALUDE_barbell_cost_l2426_242644

/-- Given that John buys barbells, gives money, and receives change, 
    prove the cost of each barbell. -/
theorem barbell_cost (num_barbells : ℕ) (money_given : ℕ) (change_received : ℕ) : 
  num_barbells = 3 → money_given = 850 → change_received = 40 → 
  (money_given - change_received) / num_barbells = 270 := by
  sorry

#check barbell_cost

end NUMINAMATH_CALUDE_barbell_cost_l2426_242644


namespace NUMINAMATH_CALUDE_dog_food_weight_l2426_242684

/-- Proves that given the conditions, each sack of dog food weighs 50 kilograms -/
theorem dog_food_weight 
  (num_dogs : ℕ) 
  (meals_per_day : ℕ) 
  (food_per_meal : ℕ) 
  (num_sacks : ℕ) 
  (days_lasting : ℕ) 
  (h1 : num_dogs = 4)
  (h2 : meals_per_day = 2)
  (h3 : food_per_meal = 250)
  (h4 : num_sacks = 2)
  (h5 : days_lasting = 50) :
  (num_dogs * meals_per_day * food_per_meal * days_lasting) / (1000 * num_sacks) = 50 := by
  sorry

#check dog_food_weight

end NUMINAMATH_CALUDE_dog_food_weight_l2426_242684


namespace NUMINAMATH_CALUDE_fraction_subtraction_multiplication_l2426_242668

theorem fraction_subtraction_multiplication :
  (5/6 - 1/3) * 3/4 = 3/8 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_multiplication_l2426_242668


namespace NUMINAMATH_CALUDE_min_n_for_inequality_l2426_242697

-- Define the notation x[n] for repeated exponentiation
def repeated_exp (x : ℕ) : ℕ → ℕ
| 0 => x
| n + 1 => x ^ (repeated_exp x n)

-- Define the specific case for 3[n]
def three_exp (n : ℕ) : ℕ := repeated_exp 3 n

-- Theorem statement
theorem min_n_for_inequality : 
  ∀ n : ℕ, three_exp n > 3^(2^9) ↔ n ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_min_n_for_inequality_l2426_242697


namespace NUMINAMATH_CALUDE_sum_difference_l2426_242611

def mena_sequence : List Nat := List.range 30

def emily_sequence : List Nat :=
  mena_sequence.map (fun n => 
    let tens := n / 10
    let ones := n % 10
    if tens = 2 then 10 + ones
    else if ones = 2 then tens * 10 + 1
    else n)

theorem sum_difference : 
  mena_sequence.sum - emily_sequence.sum = 103 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_l2426_242611


namespace NUMINAMATH_CALUDE_percentage_defective_meters_l2426_242683

theorem percentage_defective_meters 
  (total_meters : ℕ) 
  (rejected_meters : ℕ) 
  (h1 : total_meters = 2500) 
  (h2 : rejected_meters = 2) : 
  (rejected_meters : ℝ) / total_meters * 100 = 0.08 := by
sorry

end NUMINAMATH_CALUDE_percentage_defective_meters_l2426_242683


namespace NUMINAMATH_CALUDE_exists_scientist_with_one_friend_l2426_242604

-- Define the type for scientists
variable (Scientist : Type)

-- Define the friendship relation
variable (is_friend : Scientist → Scientist → Prop)

-- Define the number of friends function
variable (num_friends : Scientist → ℕ)

-- State the theorem
theorem exists_scientist_with_one_friend
  (h1 : ∀ (s1 s2 : Scientist), num_friends s1 = num_friends s2 → ¬∃ (s3 : Scientist), is_friend s1 s3 ∧ is_friend s2 s3)
  (h2 : ∀ (s1 s2 : Scientist), is_friend s1 s2 → is_friend s2 s1)
  (h3 : ∀ (s : Scientist), ¬is_friend s s)
  : ∃ (s : Scientist), num_friends s = 1 :=
by sorry

end NUMINAMATH_CALUDE_exists_scientist_with_one_friend_l2426_242604


namespace NUMINAMATH_CALUDE_square_perimeter_diagonal_ratio_l2426_242645

theorem square_perimeter_diagonal_ratio 
  (s₁ s₂ : ℝ) 
  (h_positive₁ : s₁ > 0) 
  (h_positive₂ : s₂ > 0) 
  (h_perimeter_ratio : 4 * s₂ = 5 * (4 * s₁)) :
  s₂ * Real.sqrt 2 = 5 * (s₁ * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_diagonal_ratio_l2426_242645


namespace NUMINAMATH_CALUDE_statement_c_not_always_true_l2426_242667

theorem statement_c_not_always_true : 
  ¬ ∀ (a b c : ℝ), a > b → a * c^2 > b * c^2 := by
sorry

end NUMINAMATH_CALUDE_statement_c_not_always_true_l2426_242667


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2426_242641

theorem polynomial_factorization (x : ℝ) : 
  x^12 + x^6 + 1 = (x^2 + 1) * (x^4 - x^2 + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2426_242641


namespace NUMINAMATH_CALUDE_carson_gold_stars_l2426_242682

/-- 
Given:
- Carson earned 6 gold stars yesterday
- Carson earned 9 gold stars today

Prove: The total number of gold stars Carson earned is 15
-/
theorem carson_gold_stars (yesterday_stars today_stars : ℕ) 
  (h1 : yesterday_stars = 6) 
  (h2 : today_stars = 9) : 
  yesterday_stars + today_stars = 15 := by
  sorry

end NUMINAMATH_CALUDE_carson_gold_stars_l2426_242682


namespace NUMINAMATH_CALUDE_fish_count_l2426_242630

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 14

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 24 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l2426_242630


namespace NUMINAMATH_CALUDE_stating_solution_count_56_l2426_242669

/-- 
Given a positive integer n, count_solutions n returns the number of solutions 
to the equation xy + z = n where x, y, and z are positive integers.
-/
def count_solutions (n : ℕ+) : ℕ := sorry

/-- 
Theorem stating that if count_solutions n = 56, then n = 34 or n = 35
-/
theorem solution_count_56 (n : ℕ+) : 
  count_solutions n = 56 → n = 34 ∨ n = 35 := by sorry

end NUMINAMATH_CALUDE_stating_solution_count_56_l2426_242669


namespace NUMINAMATH_CALUDE_sine_ratio_zero_l2426_242681

theorem sine_ratio_zero (c : Real) (h : c = π / 12) :
  (Real.sin (4 * c) * Real.sin (8 * c) * Real.sin (12 * c)) /
  (Real.sin (2 * c) * Real.sin (4 * c) * Real.sin (6 * c)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_ratio_zero_l2426_242681


namespace NUMINAMATH_CALUDE_dot_product_of_vectors_l2426_242622

theorem dot_product_of_vectors (a b : ℝ × ℝ) :
  a = (2, 1) →
  ‖b‖ = Real.sqrt 3 →
  ‖a + b‖ = 4 →
  a • b = 4 := by sorry

end NUMINAMATH_CALUDE_dot_product_of_vectors_l2426_242622


namespace NUMINAMATH_CALUDE_cylinder_radius_ratio_l2426_242637

/-- Represents a right circular cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The cost of filling a cylinder with gasoline --/
def fillCost (c : Cylinder) (fullness : ℝ) : ℝ := sorry

/-- The problem statement --/
theorem cylinder_radius_ratio 
  (V B : Cylinder) 
  (h_height : V.height = B.height / 2)
  (h_cost_B : fillCost B 0.5 = 4)
  (h_cost_V : fillCost V 1 = 16) :
  V.radius / B.radius = 2 := by 
  sorry


end NUMINAMATH_CALUDE_cylinder_radius_ratio_l2426_242637


namespace NUMINAMATH_CALUDE_dot_product_AB_normal_is_zero_l2426_242617

def A : ℝ × ℝ := (3, -1)
def B : ℝ × ℝ := (6, 1)
def l (x y : ℝ) : Prop := 2 * x - 3 * y - 9 = 0

def normal_vector (l : (ℝ → ℝ → Prop)) : ℝ × ℝ := (2, -3)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem dot_product_AB_normal_is_zero :
  (vector_AB.1 * (normal_vector l).1 + vector_AB.2 * (normal_vector l).2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_AB_normal_is_zero_l2426_242617


namespace NUMINAMATH_CALUDE_x_equation_implies_zero_l2426_242601

theorem x_equation_implies_zero (x : ℝ) (h : x + 1/x = Real.sqrt 5) :
  x^11 - 7*x^7 + x^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_implies_zero_l2426_242601


namespace NUMINAMATH_CALUDE_marquita_garden_count_l2426_242686

/-- The number of gardens Mancino is tending -/
def mancino_gardens : ℕ := 3

/-- The length of each of Mancino's gardens in feet -/
def mancino_garden_length : ℕ := 16

/-- The width of each of Mancino's gardens in feet -/
def mancino_garden_width : ℕ := 5

/-- The length of each of Marquita's gardens in feet -/
def marquita_garden_length : ℕ := 8

/-- The width of each of Marquita's gardens in feet -/
def marquita_garden_width : ℕ := 4

/-- The total area of all gardens combined in square feet -/
def total_garden_area : ℕ := 304

/-- Theorem stating the number of gardens Marquita is tilling -/
theorem marquita_garden_count : 
  ∃ n : ℕ, n * (marquita_garden_length * marquita_garden_width) = 
    total_garden_area - mancino_gardens * (mancino_garden_length * mancino_garden_width) ∧ 
    n = 2 := by
  sorry

end NUMINAMATH_CALUDE_marquita_garden_count_l2426_242686


namespace NUMINAMATH_CALUDE_greatest_difference_of_units_digit_l2426_242605

theorem greatest_difference_of_units_digit (x : ℕ) : 
  x < 10 →
  (720 + x) % 4 = 0 →
  ∃ y z, y < 10 ∧ z < 10 ∧ 
         (720 + y) % 4 = 0 ∧ 
         (720 + z) % 4 = 0 ∧ 
         y - z ≤ 8 ∧
         ∀ w, w < 10 → (720 + w) % 4 = 0 → y - w ≤ 8 ∧ w - z ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_greatest_difference_of_units_digit_l2426_242605


namespace NUMINAMATH_CALUDE_election_winning_probability_l2426_242649

/-- Represents the number of voters in the election -/
def total_voters : ℕ := 2019

/-- Represents the number of initial votes for the leading candidate -/
def initial_leading_votes : ℕ := 2

/-- Represents the number of initial votes for the trailing candidate -/
def initial_trailing_votes : ℕ := 1

/-- Represents the number of undecided voters -/
def undecided_voters : ℕ := total_voters - initial_leading_votes - initial_trailing_votes

/-- Calculates the probability of a candidate winning given their initial vote advantage -/
def winning_probability (initial_advantage : ℕ) : ℚ :=
  (1513 : ℚ) / 2017

/-- Theorem stating the probability of the leading candidate winning the election -/
theorem election_winning_probability :
  winning_probability (initial_leading_votes - initial_trailing_votes) = 1513 / 2017 := by
  sorry

end NUMINAMATH_CALUDE_election_winning_probability_l2426_242649


namespace NUMINAMATH_CALUDE_days_worked_by_c_l2426_242626

-- Define the problem parameters
def days_a : ℕ := 6
def days_b : ℕ := 9
def wage_ratio_a : ℕ := 3
def wage_ratio_b : ℕ := 4
def wage_ratio_c : ℕ := 5
def daily_wage_c : ℕ := 100
def total_earning : ℕ := 1480

-- Theorem statement
theorem days_worked_by_c :
  ∃ (days_c : ℕ),
    days_c * daily_wage_c +
    days_a * (daily_wage_c * wage_ratio_a / wage_ratio_c) +
    days_b * (daily_wage_c * wage_ratio_b / wage_ratio_c) = total_earning ∧
    days_c = 4 := by
  sorry


end NUMINAMATH_CALUDE_days_worked_by_c_l2426_242626


namespace NUMINAMATH_CALUDE_jessica_exam_time_l2426_242619

/-- Calculates the remaining time for Jessica to finish her exam -/
def remaining_time (total_time minutes_used questions_total questions_answered : ℕ) : ℕ :=
  total_time - minutes_used

/-- Proves that Jessica will have 48 minutes left when she finishes the exam -/
theorem jessica_exam_time : remaining_time 60 12 80 16 = 48 := by
  sorry

end NUMINAMATH_CALUDE_jessica_exam_time_l2426_242619


namespace NUMINAMATH_CALUDE_stating_third_shirt_discount_is_sixty_percent_l2426_242646

/-- Represents the discount on a shirt as a fraction between 0 and 1 -/
def Discount : Type := { d : ℝ // 0 ≤ d ∧ d ≤ 1 }

/-- The regular price of a shirt -/
def regularPrice : ℝ := 10

/-- The discount on the second shirt -/
def secondShirtDiscount : Discount := ⟨0.5, by norm_num⟩

/-- The total savings when buying three shirts -/
def totalSavings : ℝ := 11

/-- The discount on the third shirt -/
def thirdShirtDiscount : Discount := ⟨0.6, by norm_num⟩

/-- 
Theorem stating that given the regular price, second shirt discount, and total savings,
the discount on the third shirt is 60%.
-/
theorem third_shirt_discount_is_sixty_percent :
  (1 - thirdShirtDiscount.val) * regularPrice = 
    3 * regularPrice - totalSavings - regularPrice - (1 - secondShirtDiscount.val) * regularPrice :=
by sorry

end NUMINAMATH_CALUDE_stating_third_shirt_discount_is_sixty_percent_l2426_242646


namespace NUMINAMATH_CALUDE_simplify_product_of_radicals_l2426_242674

theorem simplify_product_of_radicals (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (54 * x) * Real.sqrt (20 * x) * Real.sqrt (14 * x) = 12 * Real.sqrt (105 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_radicals_l2426_242674


namespace NUMINAMATH_CALUDE_ratio_problem_l2426_242653

theorem ratio_problem (x y : ℝ) (h : (8 * x - 5 * y) / (11 * x - 3 * y) = 2 / 3) : 
  x / y = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2426_242653


namespace NUMINAMATH_CALUDE_unique_values_a_k_l2426_242698

-- Define the sets A and B
def A (k : ℕ) : Set ℕ := {1, 2, 3, k}
def B (a : ℕ) : Set ℕ := {4, 7, a^4, a^2 + 3*a}

-- Define the function f
def f (x : ℕ) : ℕ := 3*x + 1

-- Theorem statement
theorem unique_values_a_k :
  ∃! (a k : ℕ), 
    a > 0 ∧ 
    (∀ x ∈ A k, ∃ y ∈ B a, f x = y) ∧ 
    (∀ y ∈ B a, ∃ x ∈ A k, f x = y) ∧
    a = 2 ∧ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_values_a_k_l2426_242698


namespace NUMINAMATH_CALUDE_not_perfect_square_l2426_242607

theorem not_perfect_square (n : ℕ) : ¬ ∃ (a : ℕ), 3 * n + 2 = a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2426_242607


namespace NUMINAMATH_CALUDE_equation_solution_l2426_242639

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 2, 3), (2, 3, 2), (3, 2, 2), (5, 1, 4), (5, 4, 1), (4, 1, 5), (4, 5, 1),
   (1, 4, 5), (1, 5, 4), (8, 1, 3), (8, 3, 1), (3, 1, 8), (3, 8, 1), (1, 3, 8), (1, 8, 3)}

def satisfies_equation (x y z : ℕ) : Prop :=
  (x + 1) * (y + 1) * (z + 1) = 3 * x * y * z

theorem equation_solution :
  ∀ x y z : ℕ, satisfies_equation x y z ↔ (x, y, z) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2426_242639


namespace NUMINAMATH_CALUDE_stream_speed_l2426_242689

/-- The speed of a stream given upstream and downstream canoe speeds -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 4)
  (h2 : downstream_speed = 12) :
  (downstream_speed - upstream_speed) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2426_242689


namespace NUMINAMATH_CALUDE_fifteen_degrees_to_radians_l2426_242673

theorem fifteen_degrees_to_radians :
  ∀ (π : ℝ), 180 * (π / 12) = π → 15 * (π / 180) = π / 12 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_degrees_to_radians_l2426_242673


namespace NUMINAMATH_CALUDE_sqrt_neg_2x_cubed_eq_neg_x_sqrt_neg_2x_l2426_242631

theorem sqrt_neg_2x_cubed_eq_neg_x_sqrt_neg_2x :
  ∀ x : ℝ, x ≤ 0 → Real.sqrt (-2 * x^3) = -x * Real.sqrt (-2 * x) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_neg_2x_cubed_eq_neg_x_sqrt_neg_2x_l2426_242631


namespace NUMINAMATH_CALUDE_function_machine_output_l2426_242685

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 > 30 then step1 - 4 else step1
  if step2 ≤ 20 then step2 + 8 else step2 - 5

theorem function_machine_output : function_machine 10 = 25 := by
  sorry

end NUMINAMATH_CALUDE_function_machine_output_l2426_242685


namespace NUMINAMATH_CALUDE_prob_no_increasing_pie_is_correct_l2426_242609

/-- Represents the number of pies Alice has initially -/
def total_pies : ℕ := 6

/-- Represents the number of pies that increase in size -/
def increasing_pies : ℕ := 2

/-- Represents the number of pies that decrease in size -/
def decreasing_pies : ℕ := 4

/-- Represents the number of pies Alice gives to Mary -/
def pies_given : ℕ := 3

/-- Calculates the probability that one of the girls does not have a single size-increasing pie -/
def prob_no_increasing_pie : ℚ := 7/10

/-- Theorem stating that the probability of one girl having no increasing pie is 0.7 -/
theorem prob_no_increasing_pie_is_correct : 
  prob_no_increasing_pie = 7/10 :=
sorry

end NUMINAMATH_CALUDE_prob_no_increasing_pie_is_correct_l2426_242609


namespace NUMINAMATH_CALUDE_product_of_decimals_l2426_242655

theorem product_of_decimals : 3.6 * 0.25 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l2426_242655


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2426_242603

theorem possible_values_of_a (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^3 + y^3 = a) 
  (h3 : x^5 + y^5 = a) : 
  a = 0 ∨ a = 1 ∨ a = -1 ∨ a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2426_242603


namespace NUMINAMATH_CALUDE_angle_from_coordinates_l2426_242624

theorem angle_from_coordinates (a : Real) (h1 : 0 < a) (h2 : a < π / 2) :
  ∃ (P : ℝ × ℝ), P.1 = 4 * Real.sin 3 ∧ P.2 = -4 * Real.cos 3 →
  a = 3 - π / 2 :=
sorry

end NUMINAMATH_CALUDE_angle_from_coordinates_l2426_242624


namespace NUMINAMATH_CALUDE_hypotenuse_to_brush_ratio_l2426_242643

/-- A right triangle with hypotenuse 2a and a brush of width w painting one-third of its area -/
structure PaintedTriangle (a : ℝ) where
  w : ℝ
  area_painted : (a ^ 2) / 3 = a * w

/-- The ratio of the hypotenuse to the brush width is 6 -/
theorem hypotenuse_to_brush_ratio (a : ℝ) (t : PaintedTriangle a) :
  (2 * a) / t.w = 6 := by sorry

end NUMINAMATH_CALUDE_hypotenuse_to_brush_ratio_l2426_242643


namespace NUMINAMATH_CALUDE_min_side_length_is_correct_l2426_242659

/-- The sequence of side lengths of squares to be packed -/
def a (n : ℕ+) : ℚ := 1 / n

/-- The minimum side length of the square that can contain all smaller squares -/
def min_side_length : ℚ := 3 / 2

/-- Theorem stating that min_side_length is the minimum side length of a square
    that can contain all squares with side lengths a(n) without overlapping -/
theorem min_side_length_is_correct :
  ∀ (s : ℚ), (∀ (arrangement : ℕ+ → ℚ × ℚ),
    (∀ (m n : ℕ+), m ≠ n →
      (abs (arrangement m).1 - (arrangement n).1 ≥ min (a m) (a n) ∨
       abs (arrangement m).2 - (arrangement n).2 ≥ min (a m) (a n))) →
    (∀ (n : ℕ+), (arrangement n).1 + a n ≤ s ∧ (arrangement n).2 + a n ≤ s)) →
  s ≥ min_side_length :=
sorry

end NUMINAMATH_CALUDE_min_side_length_is_correct_l2426_242659
