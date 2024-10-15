import Mathlib

namespace NUMINAMATH_CALUDE_percentage_problem_l3588_358894

theorem percentage_problem (x : ℝ) (h : x / 100 * 60 = 12) : 15 / 100 * x = 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3588_358894


namespace NUMINAMATH_CALUDE_exterior_angle_smaller_implies_obtuse_l3588_358839

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Predicate to check if a triangle is obtuse -/
def is_obtuse_triangle (t : Triangle) : Prop := sorry

/-- Predicate to check if an exterior angle is smaller than its adjacent interior angle -/
def exterior_angle_smaller_than_interior (t : Triangle) : Prop := sorry

/-- Theorem: If an exterior angle of a triangle is smaller than its adjacent interior angle, 
    then the triangle is obtuse -/
theorem exterior_angle_smaller_implies_obtuse (t : Triangle) :
  exterior_angle_smaller_than_interior t → is_obtuse_triangle t := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_smaller_implies_obtuse_l3588_358839


namespace NUMINAMATH_CALUDE_parallel_iff_abs_x_eq_two_l3588_358828

def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (x^2, 4*x)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_iff_abs_x_eq_two (x : ℝ) :
  (vector_a x ≠ (0, 0)) → (vector_b x ≠ (0, 0)) →
  (parallel (vector_a x) (vector_b x) ↔ |x| = 2) :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_abs_x_eq_two_l3588_358828


namespace NUMINAMATH_CALUDE_larger_number_problem_l3588_358881

theorem larger_number_problem (x y : ℕ) 
  (h1 : y - x = 1365)
  (h2 : y = 6 * x + 15) : 
  y = 1635 := by sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3588_358881


namespace NUMINAMATH_CALUDE_initial_amount_proof_l3588_358816

/-- 
Proves that if an amount increases by 1/8th of itself each year for two years 
and becomes 72900, then the initial amount was 57600.
-/
theorem initial_amount_proof (P : ℝ) : 
  (((P + P / 8) + (P + P / 8) / 8) = 72900) → P = 57600 :=
by sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l3588_358816


namespace NUMINAMATH_CALUDE_exercise_minutes_proof_l3588_358887

/-- The number of minutes Javier exercised daily -/
def javier_daily_minutes : ℕ := 50

/-- The number of days Javier exercised in a week -/
def javier_days : ℕ := 7

/-- The number of minutes Sanda exercised on each day she exercised -/
def sanda_daily_minutes : ℕ := 90

/-- The number of days Sanda exercised -/
def sanda_days : ℕ := 3

/-- The total number of minutes Javier and Sanda exercised -/
def total_exercise_minutes : ℕ := javier_daily_minutes * javier_days + sanda_daily_minutes * sanda_days

theorem exercise_minutes_proof : total_exercise_minutes = 620 := by
  sorry

end NUMINAMATH_CALUDE_exercise_minutes_proof_l3588_358887


namespace NUMINAMATH_CALUDE_smallest_k_for_largest_three_digit_prime_l3588_358835

theorem smallest_k_for_largest_three_digit_prime (p k : ℕ) : 
  p = 997 →  -- p is the largest 3-digit prime
  k > 0 →    -- k is positive
  (∀ m : ℕ, m > 0 ∧ m < k → ¬(10 ∣ (p^2 - m))) →  -- k is the smallest such positive integer
  (10 ∣ (p^2 - k)) →  -- p^2 - k is divisible by 10
  k = 9 :=  -- k equals 9
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_largest_three_digit_prime_l3588_358835


namespace NUMINAMATH_CALUDE_solution_satisfies_conditions_l3588_358888

/-- Represents the number of bicycles, cars, and carts passing in front of a house. -/
structure VehicleCount where
  bicycles : ℕ
  cars : ℕ
  carts : ℕ

/-- Checks if the given vehicle count satisfies the problem conditions. -/
def satisfiesConditions (vc : VehicleCount) : Prop :=
  (vc.bicycles + vc.cars = 3 * vc.carts) ∧
  (2 * vc.bicycles + 4 * vc.cars + vc.bicycles + vc.cars = 100)

/-- The solution to the vehicle counting problem. -/
def solution : VehicleCount :=
  { bicycles := 10, cars := 14, carts := 8 }

/-- Theorem stating that the given solution satisfies the problem conditions. -/
theorem solution_satisfies_conditions : satisfiesConditions solution := by
  sorry


end NUMINAMATH_CALUDE_solution_satisfies_conditions_l3588_358888


namespace NUMINAMATH_CALUDE_inequality_implies_m_range_l3588_358820

theorem inequality_implies_m_range (m : ℝ) :
  (∀ x : ℝ, 4^x - m * 2^x + 1 > 0) → -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_m_range_l3588_358820


namespace NUMINAMATH_CALUDE_nine_sequences_exist_l3588_358862

/-- An arithmetic sequence of natural numbers. -/
structure ArithSeq where
  first : ℕ
  diff : ℕ

/-- The nth term of an arithmetic sequence. -/
def ArithSeq.nthTerm (seq : ArithSeq) (n : ℕ) : ℕ :=
  seq.first + (n - 1) * seq.diff

/-- The sum of the first n terms of an arithmetic sequence. -/
def ArithSeq.sumFirstN (seq : ArithSeq) (n : ℕ) : ℕ :=
  n * (2 * seq.first + (n - 1) * seq.diff) / 2

/-- The property that the ratio of sum of first 2n terms to sum of first n terms is constant. -/
def ArithSeq.hasConstantRatio (seq : ArithSeq) : Prop :=
  ∀ n : ℕ, n > 0 → (seq.sumFirstN (2*n)) / (seq.sumFirstN n) = 4

/-- The property that 1971 is a term in the sequence. -/
def ArithSeq.contains1971 (seq : ArithSeq) : Prop :=
  ∃ k : ℕ, seq.nthTerm k = 1971

/-- The main theorem stating that there are exactly 9 sequences satisfying both properties. -/
theorem nine_sequences_exist : 
  ∃! (s : Finset ArithSeq), 
    s.card = 9 ∧ 
    (∀ seq ∈ s, seq.hasConstantRatio ∧ seq.contains1971) ∧
    (∀ seq : ArithSeq, seq.hasConstantRatio ∧ seq.contains1971 → seq ∈ s) :=
sorry

end NUMINAMATH_CALUDE_nine_sequences_exist_l3588_358862


namespace NUMINAMATH_CALUDE_flora_milk_consumption_l3588_358885

/-- Calculates the total amount of milk Flora needs to drink based on the given conditions -/
def total_milk_gallons (weeks : ℕ) (flora_estimate : ℕ) (brother_additional : ℕ) : ℕ :=
  let days := weeks * 7
  let daily_amount := flora_estimate + brother_additional
  days * daily_amount

/-- Theorem stating that the total amount of milk Flora needs to drink is 105 gallons -/
theorem flora_milk_consumption :
  total_milk_gallons 3 3 2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_flora_milk_consumption_l3588_358885


namespace NUMINAMATH_CALUDE_parabolas_intersection_sum_l3588_358845

/-- The parabolas y = x^2 + 15x + 32 and x = y^2 + 49y + 593 meet at one point (x₀, y₀). -/
theorem parabolas_intersection_sum (x₀ y₀ : ℝ) :
  y₀ = x₀^2 + 15*x₀ + 32 ∧ 
  x₀ = y₀^2 + 49*y₀ + 593 →
  x₀ + y₀ = -33 :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_sum_l3588_358845


namespace NUMINAMATH_CALUDE_bus_journey_speed_l3588_358841

/-- Given a bus journey with specific conditions, prove the average speed for the remaining distance -/
theorem bus_journey_speed (total_distance : ℝ) (total_time : ℝ) (partial_distance : ℝ) (partial_speed : ℝ)
  (h1 : total_distance = 250)
  (h2 : total_time = 6)
  (h3 : partial_distance = 220)
  (h4 : partial_speed = 40)
  (h5 : partial_distance / partial_speed + (total_distance - partial_distance) / (total_time - partial_distance / partial_speed) = total_time) :
  (total_distance - partial_distance) / (total_time - partial_distance / partial_speed) = 60 := by
  sorry

#check bus_journey_speed

end NUMINAMATH_CALUDE_bus_journey_speed_l3588_358841


namespace NUMINAMATH_CALUDE_factor_proof_l3588_358830

theorem factor_proof :
  (∃ n : ℕ, 24 = 4 * n) ∧ (∃ m : ℕ, 162 = 9 * m) := by
  sorry

end NUMINAMATH_CALUDE_factor_proof_l3588_358830


namespace NUMINAMATH_CALUDE_equation_solution_range_l3588_358817

theorem equation_solution_range (a : ℝ) : 
  (∃ x : ℝ, (Real.exp (2 * x) + a * Real.exp x + 1 = 0)) ↔ a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3588_358817


namespace NUMINAMATH_CALUDE_approximate_root_of_f_l3588_358853

def f (x : ℝ) := x^3 + x^2 - 2*x - 2

theorem approximate_root_of_f :
  f 1 = -2 →
  f 1.5 = 0.625 →
  f 1.25 = -0.984 →
  f 1.375 = -0.260 →
  f 1.438 = 0.165 →
  f 1.4065 = -0.052 →
  ∃ (root : ℝ), f root = 0 ∧ |root - 1.43| < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_approximate_root_of_f_l3588_358853


namespace NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_16_l3588_358858

theorem three_digit_perfect_cube_divisible_by_16 :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^3 ∧ n % 16 = 0 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_16_l3588_358858


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l3588_358892

theorem factor_implies_c_value (c : ℝ) : 
  (∀ x : ℝ, (x - 5) * ((c/100) * x^2 + (23/100) * x - (c/20) + 11/20) = 
             c * x^3 + 23 * x^2 - 5 * c * x + 55) → 
  c = -6.3 := by sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l3588_358892


namespace NUMINAMATH_CALUDE_star_value_l3588_358813

theorem star_value : ∃ (x : ℚ), 
  45 - ((28 * 3) - (37 - (15 / (x - 2)))) = 57 ∧ x = 103/59 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l3588_358813


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l3588_358878

/-- A quadratic function that takes specific values for consecutive integers -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 13 ∧ f (n + 1) = 13 ∧ f (n + 2) = 35

/-- The theorem stating the minimum value of the quadratic function -/
theorem quadratic_function_minimum (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = 41 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l3588_358878


namespace NUMINAMATH_CALUDE_function_properties_l3588_358843

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1)

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := a^(2*x) - 4*a^x + 8

-- State the theorem
theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 3 = 1/9) :
  a = 1/3 ∧ Set.Icc 4 53 = Set.image (g (1/3)) (Set.Icc (-2) 1) := by sorry

end

end NUMINAMATH_CALUDE_function_properties_l3588_358843


namespace NUMINAMATH_CALUDE_multiple_of_seven_in_range_l3588_358812

theorem multiple_of_seven_in_range (y : ℕ) (h1 : ∃ k : ℕ, y = 7 * k)
    (h2 : y * y > 225) (h3 : y < 30) : y = 21 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_seven_in_range_l3588_358812


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l3588_358822

/-- Represents a parabola with vertex at the origin -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 4 * p * x

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : ℝ → ℝ → Prop := fun x y => x^2 / a^2 - y^2 / b^2 = 1

/-- The problem statement -/
theorem parabola_hyperbola_intersection (p : Parabola) (h : Hyperbola) : 
  (h.a > 0) →
  (h.b > 0) →
  (p.p = 2 * h.a) →  -- directrix passes through one focus of hyperbola
  (p.eq (3/2) (Real.sqrt 6)) →  -- intersection point
  (h.eq (3/2) (Real.sqrt 6)) →  -- intersection point
  (p.p = 1) ∧ (h.a^2 = 1/4) ∧ (h.b^2 = 3/4) := by
  sorry

#check parabola_hyperbola_intersection

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l3588_358822


namespace NUMINAMATH_CALUDE_heartsuit_ratio_l3588_358876

def heartsuit (n m : ℝ) : ℝ := n^2 * m^3

theorem heartsuit_ratio : (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_ratio_l3588_358876


namespace NUMINAMATH_CALUDE_kilmer_park_tree_height_l3588_358811

/-- Calculates the height of a tree in inches after a given number of years -/
def tree_height_in_inches (initial_height : ℕ) (growth_rate : ℕ) (years : ℕ) (inches_per_foot : ℕ) : ℕ :=
  (initial_height + growth_rate * years) * inches_per_foot

/-- Proves that the height of the tree in Kilmer Park after 8 years is 1104 inches -/
theorem kilmer_park_tree_height : tree_height_in_inches 52 5 8 12 = 1104 := by
  sorry

end NUMINAMATH_CALUDE_kilmer_park_tree_height_l3588_358811


namespace NUMINAMATH_CALUDE_jeffreys_steps_calculation_l3588_358880

-- Define the number of steps for Andrew and Jeffrey
def andrews_steps : ℕ := 150
def jeffreys_steps : ℕ := 200

-- Define the ratio of Andrew's steps to Jeffrey's steps
def step_ratio : ℚ := 3 / 4

-- Theorem statement
theorem jeffreys_steps_calculation :
  andrews_steps * 4 = jeffreys_steps * 3 :=
by sorry

end NUMINAMATH_CALUDE_jeffreys_steps_calculation_l3588_358880


namespace NUMINAMATH_CALUDE_norris_remaining_money_l3588_358855

/-- Calculates the remaining money for Norris after savings and spending --/
def remaining_money (september_savings october_savings november_savings spent : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - spent

/-- Theorem stating that Norris has $10 left after his savings and spending --/
theorem norris_remaining_money :
  remaining_money 29 25 31 75 = 10 := by
  sorry

end NUMINAMATH_CALUDE_norris_remaining_money_l3588_358855


namespace NUMINAMATH_CALUDE_binomial_expansion_difference_l3588_358851

theorem binomial_expansion_difference : 
  3^7 + (Nat.choose 7 2) * 3^5 + (Nat.choose 7 4) * 3^3 + (Nat.choose 7 6) * 3 -
  ((Nat.choose 7 1) * 3^6 + (Nat.choose 7 3) * 3^4 + (Nat.choose 7 5) * 3^2 + 1) = 128 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_difference_l3588_358851


namespace NUMINAMATH_CALUDE_yahs_to_bahs_1500_l3588_358829

/-- Conversion rates between bahs, rahs, and yahs -/
structure ConversionRates where
  bahs_to_rahs : ℚ
  rahs_to_yahs : ℚ

/-- Calculate the number of bahs equivalent to a given number of yahs -/
def yahs_to_bahs (rates : ConversionRates) (yahs : ℚ) : ℚ :=
  yahs * rates.rahs_to_yahs * rates.bahs_to_rahs

/-- Theorem stating that 1500 yahs are equivalent to 600 bahs given the conversion rates -/
theorem yahs_to_bahs_1500 (rates : ConversionRates) 
    (h1 : rates.bahs_to_rahs = 30 / 20)
    (h2 : rates.rahs_to_yahs = 12 / 20) : 
  yahs_to_bahs rates 1500 = 600 := by
  sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_1500_l3588_358829


namespace NUMINAMATH_CALUDE_tailor_cut_difference_l3588_358819

theorem tailor_cut_difference (dress_outer dress_middle dress_inner pants_outer pants_inner : ℝ) 
  (h1 : dress_outer = 0.75)
  (h2 : dress_middle = 0.60)
  (h3 : dress_inner = 0.55)
  (h4 : pants_outer = 0.50)
  (h5 : pants_inner = 0.45) :
  (dress_outer + dress_middle + dress_inner) - (pants_outer + pants_inner) = 0.95 := by
  sorry

end NUMINAMATH_CALUDE_tailor_cut_difference_l3588_358819


namespace NUMINAMATH_CALUDE_arrangement_counts_l3588_358865

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- Calculates the number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of arrangements where the three girls must stand together -/
def arrangements_girls_together : ℕ := 
  permutations num_girls num_girls * permutations (num_boys + 1) (num_boys + 1)

/-- The number of arrangements where no two girls are next to each other -/
def arrangements_girls_apart : ℕ := 
  permutations num_boys num_boys * permutations (num_boys + 1) num_girls

/-- The number of arrangements where there are exactly three people between person A and person B -/
def arrangements_three_between : ℕ := 
  permutations 2 2 * permutations (total_people - 2) 3 * permutations 5 5

/-- The number of arrangements where persons A and B are adjacent, but neither is next to person C -/
def arrangements_ab_adjacent_not_c : ℕ := 
  permutations 2 2 * permutations (total_people - 3) (total_people - 3) * permutations 5 2

theorem arrangement_counts :
  arrangements_girls_together = 720 ∧
  arrangements_girls_apart = 1440 ∧
  arrangements_three_between = 720 ∧
  arrangements_ab_adjacent_not_c = 960 := by sorry

end NUMINAMATH_CALUDE_arrangement_counts_l3588_358865


namespace NUMINAMATH_CALUDE_myrtle_final_eggs_l3588_358837

/-- Calculate the number of eggs Myrtle has after her trip --/
def myrtle_eggs (num_hens : ℕ) (eggs_per_hen_per_day : ℕ) (days_away : ℕ) 
                (eggs_taken_by_neighbor : ℕ) (eggs_dropped : ℕ) : ℕ :=
  num_hens * eggs_per_hen_per_day * days_away - eggs_taken_by_neighbor - eggs_dropped

/-- Theorem stating the number of eggs Myrtle has --/
theorem myrtle_final_eggs : 
  myrtle_eggs 3 3 7 12 5 = 46 := by
  sorry

end NUMINAMATH_CALUDE_myrtle_final_eggs_l3588_358837


namespace NUMINAMATH_CALUDE_gmat_scores_l3588_358827

theorem gmat_scores (u v : ℝ) (h1 : u > v) (h2 : u - v = (u + v) / 2) : v / u = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gmat_scores_l3588_358827


namespace NUMINAMATH_CALUDE_min_correct_responses_l3588_358825

def score (correct : ℕ) : ℤ :=
  8 * (correct : ℤ) - 20

theorem min_correct_responses : ∃ n : ℕ, 
  (∀ m : ℕ, m < n → score m + 10 < 120) ∧ 
  (score n + 10 ≥ 120) ∧
  n = 17 :=
sorry

end NUMINAMATH_CALUDE_min_correct_responses_l3588_358825


namespace NUMINAMATH_CALUDE_find_divisor_l3588_358863

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 17698) (h2 : quotient = 89) (h3 : remainder = 14) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 198 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3588_358863


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l3588_358860

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x < 6} = Set.Ioo (-3) 3 := by sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ (m n : ℝ), m > 0 → n > 0 → m + n = 1 → 
    ∃ x₀ : ℝ, 1/m + 1/n ≥ f a x₀} = Set.Icc (-5) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l3588_358860


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3588_358846

def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {0, 2, 3, 5}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3588_358846


namespace NUMINAMATH_CALUDE_number_problem_l3588_358809

theorem number_problem (x : ℝ) : (0.5 * x = (3/5) * x - 10) → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3588_358809


namespace NUMINAMATH_CALUDE_no_hexagon_with_special_point_l3588_358815

/-- A convex hexagon is represented by its vertices -/
def ConvexHexagon := Fin 6 → ℝ × ℝ

/-- Check if a hexagon is convex -/
def is_convex (h : ConvexHexagon) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Check if a point is inside a hexagon -/
def is_inside (p : ℝ × ℝ) (h : ConvexHexagon) : Prop := sorry

/-- The main theorem -/
theorem no_hexagon_with_special_point :
  ¬ ∃ (h : ConvexHexagon) (m : ℝ × ℝ),
    is_convex h ∧
    (∀ i : Fin 5, distance (h i) (h (i.succ)) > 1) ∧
    distance (h 5) (h 0) > 1 ∧
    is_inside m h ∧
    (∀ i : Fin 6, distance m (h i) < 1) :=
sorry

end NUMINAMATH_CALUDE_no_hexagon_with_special_point_l3588_358815


namespace NUMINAMATH_CALUDE_jakes_lawn_mowing_time_l3588_358850

/-- Jake's lawn mowing problem -/
theorem jakes_lawn_mowing_time
  (desired_hourly_rate : ℝ)
  (flower_planting_time : ℝ)
  (flower_planting_charge : ℝ)
  (lawn_mowing_pay : ℝ)
  (h1 : desired_hourly_rate = 20)
  (h2 : flower_planting_time = 2)
  (h3 : flower_planting_charge = 45)
  (h4 : lawn_mowing_pay = 15) :
  (flower_planting_charge + lawn_mowing_pay) / desired_hourly_rate - flower_planting_time = 1 := by
  sorry

#check jakes_lawn_mowing_time

end NUMINAMATH_CALUDE_jakes_lawn_mowing_time_l3588_358850


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cos_sum_l3588_358879

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cos_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 5 + a 9 = 5 * Real.pi →
  Real.cos (a 2 + a 8) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cos_sum_l3588_358879


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3588_358882

-- Define the distance in meters
def distance : ℝ := 200

-- Define the time in seconds (as a variable)
variable (p : ℝ)

-- Define the speed conversion factor from m/s to km/h
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_calculation (p : ℝ) (h : p > 0) :
  (distance / p) * conversion_factor = 720 / p := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l3588_358882


namespace NUMINAMATH_CALUDE_final_chicken_count_l3588_358897

def chicken_count (initial : ℕ) (second_factor : ℕ) (second_subtract : ℕ) (dog_eat : ℕ) (final_factor : ℕ) (final_subtract : ℕ) : ℕ :=
  let after_second := initial + (second_factor * initial - second_subtract)
  let after_dog := after_second - dog_eat
  let final_addition := final_factor * (final_factor * after_dog - final_subtract)
  after_dog + final_addition

theorem final_chicken_count :
  chicken_count 12 3 8 2 2 10 = 246 := by
  sorry

end NUMINAMATH_CALUDE_final_chicken_count_l3588_358897


namespace NUMINAMATH_CALUDE_abs_add_ge_abs_add_x_range_when_eq_possible_values_l3588_358808

-- 1. Triangle inequality for absolute values
theorem abs_add_ge_abs_add (a b : ℚ) : |a| + |b| ≥ |a + b| := by sorry

-- 2. Range of x when |x|+2015=|x-2015|
theorem x_range_when_eq (x : ℝ) (h : |x| + 2015 = |x - 2015|) : x ≤ 0 := by sorry

-- 3. Possible values of a₁+a₂ given conditions
theorem possible_values (a₁ a₂ a₃ a₄ : ℝ) 
  (h1 : |a₁ + a₂| + |a₃ + a₄| = 15) 
  (h2 : |a₁ + a₂ + a₃ + a₄| = 5) : 
  (a₁ + a₂ = 10) ∨ (a₁ + a₂ = -10) ∨ (a₁ + a₂ = 5) ∨ (a₁ + a₂ = -5) := by sorry

end NUMINAMATH_CALUDE_abs_add_ge_abs_add_x_range_when_eq_possible_values_l3588_358808


namespace NUMINAMATH_CALUDE_sticker_packs_total_cost_l3588_358884

/-- Calculates the total cost of sticker packs bought over three days --/
def total_cost (monday_packs : ℕ) (monday_price : ℚ) (monday_discount : ℚ)
                (tuesday_packs : ℕ) (tuesday_price : ℚ) (tuesday_tax : ℚ)
                (wednesday_packs : ℕ) (wednesday_price : ℚ) (wednesday_discount : ℚ) (wednesday_tax : ℚ) : ℚ :=
  let monday_cost := (monday_packs : ℚ) * monday_price * (1 - monday_discount)
  let tuesday_cost := (tuesday_packs : ℚ) * tuesday_price * (1 + tuesday_tax)
  let wednesday_cost := (wednesday_packs : ℚ) * wednesday_price * (1 - wednesday_discount) * (1 + wednesday_tax)
  monday_cost + tuesday_cost + wednesday_cost

/-- Theorem stating the total cost of sticker packs over three days --/
theorem sticker_packs_total_cost :
  total_cost 15 (5/2) (1/10) 25 3 (1/20) 30 (7/2) (3/20) (2/25) = 20889/100 :=
by sorry

end NUMINAMATH_CALUDE_sticker_packs_total_cost_l3588_358884


namespace NUMINAMATH_CALUDE_total_pears_picked_l3588_358875

theorem total_pears_picked (jason_pears keith_pears mike_pears : ℕ)
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_pears = 12) :
  jason_pears + keith_pears + mike_pears = 105 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l3588_358875


namespace NUMINAMATH_CALUDE_parkway_elementary_soccer_l3588_358891

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) (boys_soccer_percent : ℚ) :
  total_students = 470 →
  boys = 300 →
  soccer_players = 250 →
  boys_soccer_percent = 86 / 100 →
  (total_students - boys) - (soccer_players - (boys_soccer_percent * soccer_players).floor) = 135 := by
sorry

end NUMINAMATH_CALUDE_parkway_elementary_soccer_l3588_358891


namespace NUMINAMATH_CALUDE_number_of_divisors_of_360_l3588_358801

theorem number_of_divisors_of_360 : Finset.card (Nat.divisors 360) = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_360_l3588_358801


namespace NUMINAMATH_CALUDE_profit_2004_l3588_358899

/-- Represents the profit of a company over years -/
def CompanyProfit (initialProfit : ℝ) (growthRate : ℝ) (year : ℕ) : ℝ :=
  initialProfit * (1 + growthRate) ^ (year - 2002)

/-- Theorem stating the profit in 2004 given initial conditions -/
theorem profit_2004 (initialProfit growthRate : ℝ) :
  initialProfit = 10 →
  CompanyProfit initialProfit growthRate 2004 = 1000 * (1 + growthRate)^2 := by
  sorry

#check profit_2004

end NUMINAMATH_CALUDE_profit_2004_l3588_358899


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3588_358870

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 1 / b ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3588_358870


namespace NUMINAMATH_CALUDE_valid_stacks_count_l3588_358814

/-- Represents a card with a color and number -/
structure Card where
  color : Nat
  number : Nat

/-- Represents a stack of cards -/
def Stack := List Card

/-- Checks if a stack is valid according to the rules -/
def isValidStack (stack : Stack) : Bool :=
  sorry

/-- Generates all possible stacks -/
def generateStacks : List Stack :=
  sorry

/-- Counts the number of valid stacking sequences -/
def countValidStacks : Nat :=
  (generateStacks.filter isValidStack).length

/-- The main theorem stating that the number of valid stacking sequences is 6 -/
theorem valid_stacks_count :
  let redCards := [1, 2, 3, 4]
  let blueCards := [2, 3, 4]
  let greenCards := [5, 6, 7]
  countValidStacks = 6 := by
  sorry

end NUMINAMATH_CALUDE_valid_stacks_count_l3588_358814


namespace NUMINAMATH_CALUDE_distance_between_points_l3588_358883

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3.5, -2)
  let p2 : ℝ × ℝ := (7.5, 5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 65 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l3588_358883


namespace NUMINAMATH_CALUDE_repeating_decimal_23_value_l3588_358886

/-- The value of the infinite repeating decimal 0.overline{23} -/
def repeating_decimal_23 : ℚ := 23 / 99

/-- Theorem stating that the infinite repeating decimal 0.overline{23} is equal to 23/99 -/
theorem repeating_decimal_23_value : 
  repeating_decimal_23 = 23 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_23_value_l3588_358886


namespace NUMINAMATH_CALUDE_max_value_theorem_l3588_358872

theorem max_value_theorem (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  ∃ (max : ℝ), max = 1 + Real.sqrt 3 ∧ 
  ∀ (z : ℝ), z = (y + x) / x → z ≤ max := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3588_358872


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3588_358898

/-- The coefficient of x^n in the expansion of (x^2 + a/x)^m -/
def coeff (a : ℝ) (m n : ℕ) : ℝ := sorry

theorem binomial_expansion_coefficient (a : ℝ) :
  coeff a 5 7 = -15 → a = -3 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3588_358898


namespace NUMINAMATH_CALUDE_four_numbers_lcm_l3588_358861

theorem four_numbers_lcm (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a + b + c + d = 2020 →
  Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 202 →
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 2424 := by
sorry

end NUMINAMATH_CALUDE_four_numbers_lcm_l3588_358861


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l3588_358889

theorem correct_average_after_error_correction (numbers : List ℝ) 
  (h1 : numbers.length = 15)
  (h2 : numbers.sum / numbers.length = 20)
  (h3 : ∃ a b c, a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ 
               a = 35 ∧ b = 60 ∧ c = 25) :
  let corrected_numbers := numbers.map (fun x => 
    if x = 35 then 45 else if x = 60 then 58 else if x = 25 then 30 else x)
  corrected_numbers.sum / corrected_numbers.length = 20.8666666667 := by
sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l3588_358889


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3588_358852

theorem fraction_equivalence : (8 : ℚ) / (7 * 67) = (0.8 : ℚ) / (0.7 * 67) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3588_358852


namespace NUMINAMATH_CALUDE_select_two_each_select_at_least_one_each_select_with_restriction_student_selection_methods_l3588_358804

/-- The number of female students -/
def num_females : ℕ := 5

/-- The number of male students -/
def num_males : ℕ := 4

/-- The number of students to be selected -/
def num_selected : ℕ := 4

/-- Theorem for the number of ways to select 2 males and 2 females -/
theorem select_two_each : ℕ := by sorry

/-- Theorem for the number of ways to select at least 1 male and 1 female -/
theorem select_at_least_one_each : ℕ := by sorry

/-- Theorem for the number of ways to select at least 1 male and 1 female, 
    but not both male A and female B -/
theorem select_with_restriction : ℕ := by sorry

/-- Main theorem combining all selection methods -/
theorem student_selection_methods :
  select_two_each = 1440 ∧
  select_at_least_one_each = 2880 ∧
  select_with_restriction = 2376 := by sorry

end NUMINAMATH_CALUDE_select_two_each_select_at_least_one_each_select_with_restriction_student_selection_methods_l3588_358804


namespace NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l3588_358821

def number_of_divisors (n : ℕ) : ℕ := sorry

def is_prime_factorization (n : ℕ) (factors : List (ℕ × ℕ)) : Prop := sorry

theorem smallest_number_with_2020_divisors :
  ∃ (n : ℕ) (factors : List (ℕ × ℕ)),
    is_prime_factorization n factors ∧
    number_of_divisors n = 2020 ∧
    (∀ m : ℕ, m < n → number_of_divisors m ≠ 2020) ∧
    factors = [(2, 100), (3, 4), (5, 1), (7, 1)] :=
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l3588_358821


namespace NUMINAMATH_CALUDE_insects_in_laboratory_l3588_358824

/-- The number of insects in a laboratory given the total number of insect legs and legs per insect. -/
def number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Theorem stating that there are 9 insects in the laboratory given the conditions. -/
theorem insects_in_laboratory : number_of_insects 54 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_insects_in_laboratory_l3588_358824


namespace NUMINAMATH_CALUDE_custodian_jugs_theorem_l3588_358877

/-- The number of jugs needed to provide water for students -/
def jugs_needed (jug_capacity : ℕ) (num_students : ℕ) (cups_per_student : ℕ) : ℕ :=
  (num_students * cups_per_student + jug_capacity - 1) / jug_capacity

/-- Theorem: Given the conditions, 50 jugs are needed -/
theorem custodian_jugs_theorem :
  jugs_needed 40 200 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_custodian_jugs_theorem_l3588_358877


namespace NUMINAMATH_CALUDE_rearrangement_theorem_l3588_358893

/-- Represents the number of people in the line -/
def n : ℕ := 8

/-- Represents the number of people moved to the front -/
def k : ℕ := 3

/-- Calculates the number of ways to rearrange people in a line
    under the given conditions -/
def rearrangement_count (n k : ℕ) : ℕ :=
  (n - k - 1) * (n - k) * (n - k + 1)

/-- The theorem stating that the number of rearrangements is 210 -/
theorem rearrangement_theorem :
  rearrangement_count n k = 210 := by sorry

end NUMINAMATH_CALUDE_rearrangement_theorem_l3588_358893


namespace NUMINAMATH_CALUDE_boxes_sold_saturday_l3588_358847

theorem boxes_sold_saturday (saturday_sales : ℕ) (sunday_sales : ℕ) : 
  sunday_sales = saturday_sales + saturday_sales / 2 →
  saturday_sales + sunday_sales = 150 →
  saturday_sales = 60 := by
sorry

end NUMINAMATH_CALUDE_boxes_sold_saturday_l3588_358847


namespace NUMINAMATH_CALUDE_number_of_nieces_l3588_358806

def hand_mitts_price : ℚ := 14
def apron_price : ℚ := 16
def utensils_price : ℚ := 10
def knife_price : ℚ := 2 * utensils_price
def discount_rate : ℚ := 1/4
def total_spending : ℚ := 135

def discounted_price (price : ℚ) : ℚ :=
  price * (1 - discount_rate)

def gift_set_price : ℚ :=
  discounted_price hand_mitts_price +
  discounted_price apron_price +
  discounted_price utensils_price +
  discounted_price knife_price

theorem number_of_nieces :
  total_spending / gift_set_price = 3 := by sorry

end NUMINAMATH_CALUDE_number_of_nieces_l3588_358806


namespace NUMINAMATH_CALUDE_bead_arrangement_probability_l3588_358826

/-- The number of red beads -/
def num_red : ℕ := 4

/-- The number of white beads -/
def num_white : ℕ := 2

/-- The number of green beads -/
def num_green : ℕ := 1

/-- The total number of beads -/
def total_beads : ℕ := num_red + num_white + num_green

/-- A function that calculates the probability of arranging the beads
    such that no two neighboring beads are the same color -/
def prob_no_adjacent_same_color : ℚ :=
  2 / 15

/-- Theorem stating that the probability of arranging the beads
    such that no two neighboring beads are the same color is 2/15 -/
theorem bead_arrangement_probability :
  prob_no_adjacent_same_color = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_bead_arrangement_probability_l3588_358826


namespace NUMINAMATH_CALUDE_expression_evaluation_l3588_358842

theorem expression_evaluation (x y z : ℤ) (hx : x = -2) (hy : y = -4) (hz : z = 3) :
  (5 * (x - y)^2 - x * z^2) / (z - y) = 38 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3588_358842


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_l3588_358857

theorem consecutive_numbers_product (n : ℕ) : 
  (n + (n + 1) = 11) → (n * (n + 1) * (n + 2) = 210) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_l3588_358857


namespace NUMINAMATH_CALUDE_river_width_calculation_l3588_358873

/-- The width of a river given the length of an existing bridge and the additional length needed to cross it. -/
def river_width (existing_bridge_length additional_length : ℕ) : ℕ :=
  existing_bridge_length + additional_length

/-- Theorem: The width of the river is equal to the sum of the existing bridge length and the additional length needed. -/
theorem river_width_calculation (existing_bridge_length additional_length : ℕ) :
  river_width existing_bridge_length additional_length = existing_bridge_length + additional_length :=
by
  sorry

/-- The width of the specific river in the problem. -/
def specific_river_width : ℕ := river_width 295 192

#eval specific_river_width

end NUMINAMATH_CALUDE_river_width_calculation_l3588_358873


namespace NUMINAMATH_CALUDE_negation_of_existence_squared_greater_than_two_l3588_358803

theorem negation_of_existence_squared_greater_than_two :
  (¬ ∃ x : ℝ, x^2 > 2) ↔ (∀ x : ℝ, x^2 ≤ 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_squared_greater_than_two_l3588_358803


namespace NUMINAMATH_CALUDE_proposition_b_l3588_358823

theorem proposition_b (a : ℝ) : 0 < a → a < 1 → a^3 < a := by sorry

end NUMINAMATH_CALUDE_proposition_b_l3588_358823


namespace NUMINAMATH_CALUDE_root_product_l3588_358802

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  (lg x)^2 + (lg 2 + lg 3) * lg x + lg 2 * lg 3 = 0

-- State the theorem
theorem root_product (x₁ x₂ : ℝ) :
  equation x₁ ∧ equation x₂ ∧ x₁ ≠ x₂ → x₁ * x₂ = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_root_product_l3588_358802


namespace NUMINAMATH_CALUDE_inequality_not_always_hold_l3588_358832

theorem inequality_not_always_hold (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ Real.sqrt a + Real.sqrt b > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_hold_l3588_358832


namespace NUMINAMATH_CALUDE_long_distance_bill_calculation_l3588_358849

-- Define the constants
def monthly_fee : ℚ := 2
def per_minute_rate : ℚ := 12 / 100
def minutes_used : ℕ := 178

-- Define the theorem
theorem long_distance_bill_calculation :
  monthly_fee + per_minute_rate * minutes_used = 23.36 := by
  sorry

end NUMINAMATH_CALUDE_long_distance_bill_calculation_l3588_358849


namespace NUMINAMATH_CALUDE_percentage_difference_l3588_358895

theorem percentage_difference : 
  (38 / 100 * 80) - (12 / 100 * 160) = 11.2 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l3588_358895


namespace NUMINAMATH_CALUDE_first_storm_rate_l3588_358874

/-- Represents the rainfall data for a week with two rainstorms -/
structure RainfallData where
  firstStormRate : ℝ
  secondStormRate : ℝ
  totalRainTime : ℝ
  totalRainfall : ℝ
  firstStormDuration : ℝ

/-- Theorem stating that given the rainfall conditions, the first storm's rate was 30 mm/hour -/
theorem first_storm_rate (data : RainfallData)
    (h1 : data.secondStormRate = 15)
    (h2 : data.totalRainTime = 45)
    (h3 : data.totalRainfall = 975)
    (h4 : data.firstStormDuration = 20) :
    data.firstStormRate = 30 := by
  sorry

#check first_storm_rate

end NUMINAMATH_CALUDE_first_storm_rate_l3588_358874


namespace NUMINAMATH_CALUDE_correct_last_digit_prob_l3588_358859

/-- The number of possible digits for each position in the password -/
def num_digits : ℕ := 10

/-- The probability of guessing the correct digit on the first attempt -/
def first_attempt_prob : ℚ := 1 / num_digits

/-- The probability of guessing the correct digit on the second attempt, given the first attempt was incorrect -/
def second_attempt_prob : ℚ := 1 / (num_digits - 1)

/-- The probability of guessing the correct last digit within 2 attempts -/
def two_attempt_prob : ℚ := first_attempt_prob + (1 - first_attempt_prob) * second_attempt_prob

theorem correct_last_digit_prob :
  two_attempt_prob = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_last_digit_prob_l3588_358859


namespace NUMINAMATH_CALUDE_matrix_power_four_l3588_358868

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, 1; 1, 1]

theorem matrix_power_four :
  A ^ 4 = !![34, 21; 21, 13] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l3588_358868


namespace NUMINAMATH_CALUDE_odd_function_proof_l3588_358818

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define h in terms of f
def h (x : ℝ) : ℝ := f x - 9

-- State the theorem
theorem odd_function_proof :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  h 1 = 2 →               -- h(1) = 2
  f (-1) = -11 :=         -- Conclusion: f(-1) = -11
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_odd_function_proof_l3588_358818


namespace NUMINAMATH_CALUDE_more_wins_probability_correct_l3588_358807

/-- The number of matches played by the team -/
def num_matches : ℕ := 5

/-- The probability of winning, losing, or tying a single match -/
def match_probability : ℚ := 1/3

/-- The probability of ending with more wins than losses -/
def more_wins_probability : ℚ := 16/243

theorem more_wins_probability_correct :
  (∀ (outcome : Fin num_matches → Fin 3),
    (∃ (wins losses : ℕ),
      wins > losses ∧
      wins + losses ≤ num_matches ∧
      (∀ i, outcome i = 0 → wins > 0) ∧
      (∀ i, outcome i = 1 → losses > 0))) →
  ∃ (favorable_outcomes : ℕ),
    favorable_outcomes = 16 ∧
    (favorable_outcomes : ℚ) / (3 ^ num_matches) = more_wins_probability :=
sorry

end NUMINAMATH_CALUDE_more_wins_probability_correct_l3588_358807


namespace NUMINAMATH_CALUDE_min_value_x_plus_3y_l3588_358864

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 4 * y = x * y) :
  x + 3 * y ≥ 25 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 3 * x + 4 * y = x * y ∧ x + 3 * y = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_3y_l3588_358864


namespace NUMINAMATH_CALUDE_divisor_problem_l3588_358866

theorem divisor_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 17698 →
  quotient = 89 →
  remainder = 14 →
  ∃ (divisor : ℕ), 
    dividend = divisor * quotient + remainder ∧
    divisor = 198 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3588_358866


namespace NUMINAMATH_CALUDE_book_price_problem_l3588_358867

theorem book_price_problem (cost_price : ℝ) : 
  (110 / 100 * cost_price = 1100) → 
  (80 / 100 * cost_price = 800) := by
  sorry

end NUMINAMATH_CALUDE_book_price_problem_l3588_358867


namespace NUMINAMATH_CALUDE_last_page_cards_l3588_358848

/-- Calculates the number of cards on the last page after reorganization --/
def cards_on_last_page (initial_albums : ℕ) (initial_pages_per_album : ℕ) 
  (initial_cards_per_page : ℕ) (new_cards_per_page : ℕ) (full_albums : ℕ) 
  (extra_full_pages : ℕ) : ℕ :=
  let total_cards := initial_albums * initial_pages_per_album * initial_cards_per_page
  let cards_in_full_albums := full_albums * initial_pages_per_album * new_cards_per_page
  let cards_in_extra_pages := extra_full_pages * new_cards_per_page
  let remaining_cards := total_cards - (cards_in_full_albums + cards_in_extra_pages)
  remaining_cards - (extra_full_pages * new_cards_per_page)

/-- Theorem stating that given the problem conditions, the last page contains 40 cards --/
theorem last_page_cards : 
  cards_on_last_page 10 50 8 12 5 40 = 40 := by
  sorry

end NUMINAMATH_CALUDE_last_page_cards_l3588_358848


namespace NUMINAMATH_CALUDE_exponential_inequality_minimum_value_of_f_l3588_358844

-- Proposition 2
theorem exponential_inequality (x₁ x₂ : ℝ) :
  Real.exp ((x₁ + x₂) / 2) ≤ (Real.exp x₁ + Real.exp x₂) / 2 := by
  sorry

-- Proposition 4
def f (x : ℝ) : ℝ := (x - 2014)^2 - 2

theorem minimum_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = -2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_minimum_value_of_f_l3588_358844


namespace NUMINAMATH_CALUDE_min_beta_delta_sum_l3588_358834

open Complex

/-- A complex function g satisfying certain conditions -/
def g (β δ : ℂ) : ℂ → ℂ := λ z ↦ (3 + 2*I)*z^3 + β*z + δ

/-- The theorem stating the minimum value of |β| + |δ| -/
theorem min_beta_delta_sum (β δ : ℂ) :
  (g β δ 1).im = 0 →
  (g β δ (-I)).im = -Real.pi →
  ∃ (min : ℝ), min = Real.sqrt (Real.pi^2 + 2*Real.pi + 2) + 2 ∧
    ∀ (β' δ' : ℂ), (g β' δ' 1).im = 0 → (g β' δ' (-I)).im = -Real.pi →
      Complex.abs β' + Complex.abs δ' ≥ min :=
sorry


end NUMINAMATH_CALUDE_min_beta_delta_sum_l3588_358834


namespace NUMINAMATH_CALUDE_dollar_composition_30_l3588_358805

/-- The dollar function as defined in the problem -/
noncomputable def dollar (N : ℝ) : ℝ := 0.75 * N + 2

/-- The statement to be proved -/
theorem dollar_composition_30 : dollar (dollar (dollar 30)) = 17.28125 := by
  sorry

end NUMINAMATH_CALUDE_dollar_composition_30_l3588_358805


namespace NUMINAMATH_CALUDE_sum_geq_abs_sum_div_3_l3588_358856

theorem sum_geq_abs_sum_div_3 (a b c : ℝ) 
  (hab : a + b ≥ 0) (hbc : b + c ≥ 0) (hca : c + a ≥ 0) : 
  a + b + c ≥ (|a| + |b| + |c|) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_abs_sum_div_3_l3588_358856


namespace NUMINAMATH_CALUDE_circle_point_x_coordinate_l3588_358896

theorem circle_point_x_coordinate :
  ∀ (x : ℝ),
  let center_x : ℝ := (-3 + 21) / 2
  let center_y : ℝ := 0
  let radius : ℝ := (21 - (-3)) / 2
  (x - center_x)^2 + (12 - center_y)^2 = radius^2 →
  x = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_point_x_coordinate_l3588_358896


namespace NUMINAMATH_CALUDE_survey_result_l3588_358836

theorem survey_result (X : ℝ) (total : ℕ) (h_total : total ≥ 100) : ℝ :=
  let liked_A := X
  let liked_both := 23
  let liked_neither := 23
  let liked_B := 100 - X
  sorry

end NUMINAMATH_CALUDE_survey_result_l3588_358836


namespace NUMINAMATH_CALUDE_seven_people_arrangement_count_l3588_358890

def total_arrangements (n : ℕ) : ℕ := n.factorial

def adjacent_pair_arrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

def front_restricted_arrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

def end_restricted_arrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

def front_and_end_restricted_arrangements (n : ℕ) : ℕ := (n - 2).factorial * 2

theorem seven_people_arrangement_count : 
  total_arrangements 6 - 
  front_restricted_arrangements 6 - 
  end_restricted_arrangements 6 + 
  front_and_end_restricted_arrangements 6 = 1008 :=
by sorry

end NUMINAMATH_CALUDE_seven_people_arrangement_count_l3588_358890


namespace NUMINAMATH_CALUDE_parabola_equation_l3588_358869

/-- A parabola with vertex at the origin and axis of symmetry along a coordinate axis -/
structure Parabola where
  a : ℝ
  axis : Bool -- true for y-axis, false for x-axis

/-- The point (-4, -2) -/
def P : ℝ × ℝ := (-4, -2)

/-- Check if a point satisfies the parabola equation -/
def satisfiesEquation (p : Parabola) (point : ℝ × ℝ) : Prop :=
  if p.axis then
    point.2^2 = p.a * point.1
  else
    point.1^2 = p.a * point.2

theorem parabola_equation :
  ∃ (p1 p2 : Parabola),
    satisfiesEquation p1 P ∧
    satisfiesEquation p2 P ∧
    p1.axis = true ∧
    p2.axis = false ∧
    p1.a = -1 ∧
    p2.a = -8 :=
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3588_358869


namespace NUMINAMATH_CALUDE_range_of_a_l3588_358854

open Real

theorem range_of_a (a : ℝ) : 
  let P := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
  let Q := ∀ x y : ℝ, x < y → -(5-2*a)^x > -(5-2*a)^y
  (P ∧ ¬Q) ∨ (¬P ∧ Q) → a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3588_358854


namespace NUMINAMATH_CALUDE_f_is_periodic_l3588_358840

/-- Given two functions f and g on ℝ satisfying certain conditions, 
    prove that f is periodic -/
theorem f_is_periodic 
  (f g : ℝ → ℝ)
  (h₁ : f 0 = 1)
  (h₂ : ∃ a : ℝ, a ≠ 0 ∧ g a = 1)
  (h₃ : ∀ x, g (-x) = -g x)
  (h₄ : ∀ x y, f (x - y) = f x * f y + g x * g y) :
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x :=
sorry

end NUMINAMATH_CALUDE_f_is_periodic_l3588_358840


namespace NUMINAMATH_CALUDE_two_correct_probability_l3588_358838

/-- The number of houses and packages -/
def n : ℕ := 4

/-- The probability of exactly two packages being delivered to the correct houses -/
def prob_two_correct : ℚ := 1/4

/-- Theorem stating that the probability of exactly two packages being delivered 
    to the correct houses out of four is 1/4 -/
theorem two_correct_probability : 
  (n.choose 2 : ℚ) * (1/n) * (1/(n-1)) * (1/2) = prob_two_correct := by
  sorry

end NUMINAMATH_CALUDE_two_correct_probability_l3588_358838


namespace NUMINAMATH_CALUDE_water_jars_problem_l3588_358800

theorem water_jars_problem (C1 C2 C3 W : ℚ) : 
  W > 0 ∧ C1 > 0 ∧ C2 > 0 ∧ C3 > 0 →
  W = (1/7) * C1 ∧ W = (2/9) * C2 ∧ W = (3/11) * C3 →
  C3 ≥ C1 ∧ C3 ≥ C2 →
  (3 * W) / C3 = 9/11 := by
sorry


end NUMINAMATH_CALUDE_water_jars_problem_l3588_358800


namespace NUMINAMATH_CALUDE_solution_set_equality_l3588_358871

def S : Set ℝ := {x : ℝ | |x - 1| + |x + 2| ≤ 4}

theorem solution_set_equality : S = Set.Icc (-5/2) (3/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3588_358871


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3588_358831

theorem complex_fraction_equality : Complex.I * Complex.I = -1 → (3 : ℂ) / (1 - Complex.I)^2 = (3 / 2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3588_358831


namespace NUMINAMATH_CALUDE_percentage_sum_l3588_358833

theorem percentage_sum : 
  (20 / 100 * 40) + (25 / 100 * 60) = 23 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_l3588_358833


namespace NUMINAMATH_CALUDE_line_PQ_parallel_to_x_axis_l3588_358810

-- Define the points P and Q
def P : ℝ × ℝ := (6, -6)
def Q : ℝ × ℝ := (-6, -6)

-- Define a line as parallel to x-axis if y-coordinates are equal
def parallel_to_x_axis (A B : ℝ × ℝ) : Prop :=
  A.2 = B.2

-- Theorem statement
theorem line_PQ_parallel_to_x_axis :
  parallel_to_x_axis P Q := by sorry

end NUMINAMATH_CALUDE_line_PQ_parallel_to_x_axis_l3588_358810
