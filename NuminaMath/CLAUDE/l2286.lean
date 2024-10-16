import Mathlib

namespace NUMINAMATH_CALUDE_rationalize_denominator_cube_roots_l2286_228689

theorem rationalize_denominator_cube_roots :
  let x := (3 : ℝ)^(1/3)
  let y := (2 : ℝ)^(1/3)
  1 / (x - y) = x^2 + x*y + y^2 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_cube_roots_l2286_228689


namespace NUMINAMATH_CALUDE_jessie_weight_calculation_l2286_228672

/-- Calculates the initial weight given the current weight and weight lost -/
def initial_weight (current_weight weight_lost : ℝ) : ℝ :=
  current_weight + weight_lost

/-- Theorem: If Jessie's current weight is 27 kg and she lost 10 kg, her initial weight was 37 kg -/
theorem jessie_weight_calculation :
  let current_weight : ℝ := 27
  let weight_lost : ℝ := 10
  initial_weight current_weight weight_lost = 37 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_calculation_l2286_228672


namespace NUMINAMATH_CALUDE_age_problem_l2286_228685

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 22 →
  b = 8 := by sorry

end NUMINAMATH_CALUDE_age_problem_l2286_228685


namespace NUMINAMATH_CALUDE_symmetric_graphs_intersection_l2286_228670

noncomputable def f (a b x : ℝ) : ℝ := 2*a + 1/(x-b)

theorem symmetric_graphs_intersection (a b c d : ℝ) :
  (∃! x, f a b x = f c d x) ↔ (a - c) * (b - d) = 2 := by sorry

end NUMINAMATH_CALUDE_symmetric_graphs_intersection_l2286_228670


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l2286_228636

theorem fraction_sum_theorem : 
  (1 : ℚ) / 15 + 2 / 15 + 3 / 15 + 4 / 15 + 5 / 15 + 
  6 / 15 + 7 / 15 + 8 / 15 + 9 / 15 + 46 / 15 = 91 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l2286_228636


namespace NUMINAMATH_CALUDE_mario_flower_count_l2286_228633

/-- The number of flowers on Mario's first hibiscus plant -/
def first_plant_flowers : ℕ := 2

/-- The number of flowers on Mario's second hibiscus plant -/
def second_plant_flowers : ℕ := 2 * first_plant_flowers

/-- The number of flowers on Mario's third hibiscus plant -/
def third_plant_flowers : ℕ := 4 * second_plant_flowers

/-- The total number of flowers on all of Mario's hibiscus plants -/
def total_flowers : ℕ := first_plant_flowers + second_plant_flowers + third_plant_flowers

theorem mario_flower_count : total_flowers = 22 := by
  sorry

end NUMINAMATH_CALUDE_mario_flower_count_l2286_228633


namespace NUMINAMATH_CALUDE_cat_walking_time_l2286_228638

/-- Proves that the total time for Jenny's cat walking process is 28 minutes -/
theorem cat_walking_time (resisting_time : ℝ) (walking_distance : ℝ) (walking_rate : ℝ) : 
  resisting_time = 20 →
  walking_distance = 64 →
  walking_rate = 8 →
  resisting_time + walking_distance / walking_rate = 28 :=
by sorry

end NUMINAMATH_CALUDE_cat_walking_time_l2286_228638


namespace NUMINAMATH_CALUDE_complex_powers_sum_l2286_228614

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_powers_sum : i^245 + i^246 + i^247 + i^248 + i^249 = i := by sorry

end NUMINAMATH_CALUDE_complex_powers_sum_l2286_228614


namespace NUMINAMATH_CALUDE_decimal_41_to_binary_l2286_228644

-- Define a function to convert decimal to binary
def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 2) ((m % 2) :: acc)
    go n []

-- Theorem statement
theorem decimal_41_to_binary :
  decimalToBinary 41 = [1, 0, 1, 0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_decimal_41_to_binary_l2286_228644


namespace NUMINAMATH_CALUDE_ellipse_m_values_l2286_228686

/-- An ellipse with equation x²/5 + y²/m = 1 and eccentricity √10/5 has m equal to 3 or 25/3 -/
theorem ellipse_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2/5 + y^2/m = 1) →  -- Ellipse equation
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (x^2/a^2 + y^2/b^2 = 1 ↔ x^2/5 + y^2/m = 1) ∧  -- Standard form of ellipse
    c^2/a^2 = 10/25) →  -- Eccentricity condition
  m = 3 ∨ m = 25/3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_m_values_l2286_228686


namespace NUMINAMATH_CALUDE_xiao_ming_reading_inequality_l2286_228619

/-- Represents Xiao Ming's reading situation -/
def reading_situation (total_pages : ℕ) (total_days : ℕ) (initial_pages_per_day : ℕ) (initial_days : ℕ) (remaining_pages_per_day : ℝ) : Prop :=
  (initial_pages_per_day * initial_days : ℝ) + (remaining_pages_per_day * (total_days - initial_days)) ≥ total_pages

/-- The inequality correctly represents Xiao Ming's reading situation -/
theorem xiao_ming_reading_inequality :
  reading_situation 72 10 5 2 x ↔ 10 + 8 * x ≥ 72 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_reading_inequality_l2286_228619


namespace NUMINAMATH_CALUDE_equation_solution_l2286_228683

theorem equation_solution : 
  let y : ℚ := 6/7
  ∀ (y : ℚ), y ≠ -2 ∧ y ≠ -1 →
  (7*y) / ((y+2)*(y+1)) - 4 / ((y+2)*(y+1)) = 2 / ((y+2)*(y+1)) →
  y = 6/7 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2286_228683


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l2286_228687

/-- Given a 5-12-13 right triangle with vertices as centers of mutually externally tangent circles,
    where the radius of the circle at the right angle is half that of the circle opposite the shortest side,
    prove that the sum of the areas of these circles is 105π. -/
theorem sum_of_circle_areas (r s t : ℝ) : 
  r > 0 ∧ s > 0 ∧ t > 0 →  -- radii are positive
  r + s = 13 →  -- sum of radii equals hypotenuse
  s + t = 5 →   -- sum of radii equals one side
  r + t = 12 →  -- sum of radii equals other side
  t = r / 2 →   -- radius at right angle is half of radius opposite shortest side
  π * (r^2 + s^2 + t^2) = 105 * π := by
  sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l2286_228687


namespace NUMINAMATH_CALUDE_num_triples_eq_three_l2286_228600

/-- The number of triples (a, b, c) of positive integers satisfying a + ab + abc = 11 -/
def num_triples : Nat :=
  (Finset.filter (fun t : Nat × Nat × Nat =>
    let (a, b, c) := t
    a > 0 ∧ b > 0 ∧ c > 0 ∧ a + a * b + a * b * c = 11)
    (Finset.product (Finset.range 12) (Finset.product (Finset.range 12) (Finset.range 12)))).card

/-- Theorem stating that there are exactly 3 triples (a, b, c) of positive integers
    satisfying a + ab + abc = 11 -/
theorem num_triples_eq_three : num_triples = 3 := by
  sorry

end NUMINAMATH_CALUDE_num_triples_eq_three_l2286_228600


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2286_228661

theorem arithmetic_expression_equality : 8 + 18 / 3 - 4 * 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2286_228661


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2286_228682

theorem smallest_n_congruence (n : ℕ) : n > 0 → (∀ k < n, (7^k : ℤ) % 5 ≠ k^7 % 5) → (7^n : ℤ) % 5 = n^7 % 5 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2286_228682


namespace NUMINAMATH_CALUDE_number_of_children_l2286_228693

/-- Given that each child has 8 crayons and there are 56 crayons in total,
    prove that the number of children is 7. -/
theorem number_of_children (crayons_per_child : ℕ) (total_crayons : ℕ) 
  (h1 : crayons_per_child = 8) (h2 : total_crayons = 56) :
  total_crayons / crayons_per_child = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l2286_228693


namespace NUMINAMATH_CALUDE_alex_age_l2286_228678

theorem alex_age (inez_age : ℕ) (zack_age : ℕ) (jose_age : ℕ) (alex_age : ℕ)
  (h1 : inez_age = 18)
  (h2 : zack_age = inez_age + 5)
  (h3 : jose_age = zack_age - 6)
  (h4 : alex_age = jose_age - 2) :
  alex_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_alex_age_l2286_228678


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2286_228611

def M : Set ℝ := {x | x^2 < x}
def N : Set ℝ := {x | x^2 + 2*x - 3 < 0}

theorem union_of_M_and_N : M ∪ N = Set.Ioo (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2286_228611


namespace NUMINAMATH_CALUDE_orange_count_l2286_228662

theorem orange_count (total : ℕ) (apple_ratio : ℕ) (orange_count : ℕ) : 
  total = 40 →
  apple_ratio = 3 →
  orange_count + apple_ratio * orange_count = total →
  orange_count = 10 := by
sorry

end NUMINAMATH_CALUDE_orange_count_l2286_228662


namespace NUMINAMATH_CALUDE_combination_sum_equals_462_l2286_228697

theorem combination_sum_equals_462 : 
  (Nat.choose 4 4) + (Nat.choose 5 4) + (Nat.choose 6 4) + (Nat.choose 7 4) + 
  (Nat.choose 8 4) + (Nat.choose 9 4) + (Nat.choose 10 4) = 462 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equals_462_l2286_228697


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2286_228691

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_formula :
  ∀ n : ℕ, n ≥ 1 → arithmetic_sequence (-3) 4 n = 4*n - 7 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2286_228691


namespace NUMINAMATH_CALUDE_external_circle_radius_l2286_228675

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right angle at C
  (B.1 - C.1) * (A.1 - C.1) + (B.2 - C.2) * (A.2 - C.2) = 0 ∧
  -- Angle A is 45°
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) / Real.sqrt 2 ∧
  -- AC = 12
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 144

-- Define the external circle
def ExternalCircle (center : ℝ × ℝ) (radius : ℝ) (A B C : ℝ × ℝ) : Prop :=
  -- Circle is tangent to AB
  ((center.1 - A.1) * (B.2 - A.2) - (center.2 - A.2) * (B.1 - A.1))^2 = 
    radius^2 * ((B.1 - A.1)^2 + (B.2 - A.2)^2) ∧
  -- Center lies on line AB
  (center.2 - A.2) * (B.1 - A.1) = (center.1 - A.1) * (B.2 - A.2)

-- Theorem statement
theorem external_circle_radius (A B C : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) :
  Triangle A B C → ExternalCircle center radius A B C → radius = 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_external_circle_radius_l2286_228675


namespace NUMINAMATH_CALUDE_refill_count_is_three_l2286_228696

/-- Calculates the number of daily water bottle refills given the parameters. -/
def daily_refills (bottle_capacity : ℕ) (days : ℕ) (spill1 : ℕ) (spill2 : ℕ) (total_drunk : ℕ) : ℕ :=
  ((total_drunk + spill1 + spill2) / (bottle_capacity * days) : ℕ)

/-- Proves that given the specified parameters, the number of daily refills is 3. -/
theorem refill_count_is_three :
  daily_refills 20 7 5 8 407 = 3 := by
  sorry

end NUMINAMATH_CALUDE_refill_count_is_three_l2286_228696


namespace NUMINAMATH_CALUDE_tv_purchase_hours_l2286_228668

/-- The number of additional hours needed to buy a TV given the TV cost, hourly wage, and weekly work hours. -/
def additional_hours_needed (tv_cost : ℕ) (hourly_wage : ℕ) (weekly_work_hours : ℕ) : ℕ :=
  let monthly_work_hours := weekly_work_hours * 4
  let monthly_earnings := monthly_work_hours * hourly_wage
  let additional_amount_needed := tv_cost - monthly_earnings
  additional_amount_needed / hourly_wage

/-- Theorem stating that given a TV cost of $1700, an hourly wage of $10, and a 30-hour workweek, 
    the additional hours needed to buy the TV is 50. -/
theorem tv_purchase_hours : additional_hours_needed 1700 10 30 = 50 := by
  sorry

end NUMINAMATH_CALUDE_tv_purchase_hours_l2286_228668


namespace NUMINAMATH_CALUDE_iceland_visitors_l2286_228667

theorem iceland_visitors (total : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 90)
  (h2 : norway = 33)
  (h3 : both = 51)
  (h4 : neither = 53) :
  total - neither - norway + both = 55 := by
  sorry

end NUMINAMATH_CALUDE_iceland_visitors_l2286_228667


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_of_primes_mod_240_l2286_228695

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the fourth powers of the first n prime numbers -/
def sumOfFourthPowersOfPrimes (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => (nthPrime (i + 1)) ^ 4)

/-- The main theorem -/
theorem sum_of_fourth_powers_of_primes_mod_240 :
  sumOfFourthPowersOfPrimes 2014 % 240 = 168 := by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_of_primes_mod_240_l2286_228695


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2286_228656

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: If 2S_3 = 3S_2 + 6 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2286_228656


namespace NUMINAMATH_CALUDE_find_a_find_m_range_l2286_228640

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + a

-- Statement 1
theorem find_a :
  (∀ x : ℝ, f 2 x < 5 ↔ -3/2 < x ∧ x < 1) →
  (∃! a : ℝ, ∀ x : ℝ, f a x < 5 ↔ -3/2 < x ∧ x < 1) :=
sorry

-- Statement 2
theorem find_m_range (m : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 5 → 2 * x^2 + x + 2 > m * x) →
  m < 5 :=
sorry

end NUMINAMATH_CALUDE_find_a_find_m_range_l2286_228640


namespace NUMINAMATH_CALUDE_bill_calculation_l2286_228608

theorem bill_calculation (a b c d : ℝ) 
  (h1 : (a - b) + c - d = 19) 
  (h2 : a - b - c - d = 9) : 
  a - b = 14 := by
sorry

end NUMINAMATH_CALUDE_bill_calculation_l2286_228608


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2286_228642

theorem solution_set_inequality (x : ℝ) :
  (x - 4) * (x + 1) > 0 ↔ x > 4 ∨ x < -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2286_228642


namespace NUMINAMATH_CALUDE_placemats_length_l2286_228680

theorem placemats_length (R : ℝ) (n : ℕ) (x : ℝ) : 
  R = 5 ∧ n = 8 → x = 2 * R * Real.sin (π / (2 * n)) := by
  sorry

end NUMINAMATH_CALUDE_placemats_length_l2286_228680


namespace NUMINAMATH_CALUDE_triangle_with_sum_of_two_angles_less_than_third_is_obtuse_l2286_228605

theorem triangle_with_sum_of_two_angles_less_than_third_is_obtuse 
  (α β γ : Real) 
  (triangle_angles : α + β + γ = 180) 
  (angle_sum_condition : α + β < γ) : 
  γ > 90 := by
sorry

end NUMINAMATH_CALUDE_triangle_with_sum_of_two_angles_less_than_third_is_obtuse_l2286_228605


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_l2286_228660

theorem x_eq_one_sufficient_not_necessary :
  (∃ x : ℝ, x ^ 2 = 1 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → x ^ 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_l2286_228660


namespace NUMINAMATH_CALUDE_power_difference_inequality_l2286_228625

theorem power_difference_inequality (n : ℕ) (a b : ℝ) 
  (hn : n > 1) (hab : a > b) (hb : b > 0) :
  (a^n - b^n) * (1/b^(n-1) - 1/a^(n-1)) > 4*n*(n-1)*(Real.sqrt a - Real.sqrt b)^2 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_inequality_l2286_228625


namespace NUMINAMATH_CALUDE_sara_received_six_kittens_l2286_228612

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara (initial : ℕ) (to_jessica : ℕ) (left : ℕ) : ℕ :=
  initial - to_jessica - left

/-- Proof that Tim gave 6 kittens to Sara -/
theorem sara_received_six_kittens :
  kittens_to_sara 18 3 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sara_received_six_kittens_l2286_228612


namespace NUMINAMATH_CALUDE_inverse_of_10_mod_1001_l2286_228674

theorem inverse_of_10_mod_1001 : ∃ x : ℕ, x ∈ Finset.range 1001 ∧ (10 * x) % 1001 = 1 :=
by
  use 901
  sorry

end NUMINAMATH_CALUDE_inverse_of_10_mod_1001_l2286_228674


namespace NUMINAMATH_CALUDE_inverse_proportion_quadrants_l2286_228606

/-- A line passes through the first, second, and fourth quadrants -/
structure LineInQuadrants (k b : ℝ) : Prop :=
  (first_quadrant : ∃ x > 0, k * x + b > 0)
  (second_quadrant : ∃ x < 0, k * x + b > 0)
  (fourth_quadrant : ∃ x > 0, k * x + b < 0)

/-- The graph of an inverse proportion function is in the second and fourth quadrants -/
def InverseProportionInQuadrants (k b : ℝ) : Prop :=
  (∀ x > 0, k * b / x < 0) ∧ (∀ x < 0, k * b / x > 0)

/-- Theorem statement -/
theorem inverse_proportion_quadrants (k b : ℝ) :
  LineInQuadrants k b → InverseProportionInQuadrants k b :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_quadrants_l2286_228606


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2286_228688

theorem arithmetic_sequence_length 
  (a₁ : ℤ) 
  (d : ℤ) 
  (aₙ : ℤ) 
  (h1 : a₁ = 2) 
  (h2 : d = 3) 
  (h3 : aₙ = 110) :
  ∃ n : ℕ, n = 37 ∧ aₙ = a₁ + (n - 1) * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2286_228688


namespace NUMINAMATH_CALUDE_four_digit_divisibility_sum_l2286_228652

/-- The number of four-digit numbers divisible by 3 -/
def C : ℕ := 3000

/-- The number of four-digit multiples of 7 -/
def D : ℕ := 1286

/-- Theorem stating that the sum of four-digit numbers divisible by 3 and multiples of 7 is 4286 -/
theorem four_digit_divisibility_sum : C + D = 4286 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisibility_sum_l2286_228652


namespace NUMINAMATH_CALUDE_chord_length_l2286_228620

/-- The length of the chord formed by the intersection of the line x = 1 and the circle (x-2)² + y² = 4 is 2√3 -/
theorem chord_length : ∃ (A B : ℝ × ℝ), 
  (A.1 = 1 ∧ (A.1 - 2)^2 + A.2^2 = 4) ∧ 
  (B.1 = 1 ∧ (B.1 - 2)^2 + B.2^2 = 4) ∧ 
  A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l2286_228620


namespace NUMINAMATH_CALUDE_equation_solution_l2286_228623

theorem equation_solution (x y z : ℕ) :
  (x : ℚ) + 1 / ((y : ℚ) + 1 / (z : ℚ)) = 10 / 7 →
  x = 1 ∧ y = 2 ∧ z = 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2286_228623


namespace NUMINAMATH_CALUDE_five_digit_sum_contains_zero_l2286_228613

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def digits_differ_by_two (a b : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 e1 e2 e3 e4 e5 : ℕ),
    a = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 ∧
    b = e1 * 10000 + e2 * 1000 + e3 * 100 + e4 * 10 + e5 ∧
    ({d1, d2, d3, d4, d5} : Finset ℕ) = {e1, e2, e3, e4, e5} ∧
    (d1 = e1 ∧ d2 = e2 ∧ d4 = e4 ∧ d5 = e5) ∨
    (d1 = e1 ∧ d2 = e2 ∧ d3 = e3 ∧ d5 = e5) ∨
    (d1 = e1 ∧ d2 = e2 ∧ d3 = e3 ∧ d4 = e4)

def contains_zero (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 : ℕ),
    n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 ∧
    (d1 = 0 ∨ d2 = 0 ∨ d3 = 0 ∨ d4 = 0 ∨ d5 = 0)

theorem five_digit_sum_contains_zero (a b : ℕ) :
  is_five_digit a → is_five_digit b → digits_differ_by_two a b → a + b = 111111 →
  contains_zero a ∨ contains_zero b :=
sorry

end NUMINAMATH_CALUDE_five_digit_sum_contains_zero_l2286_228613


namespace NUMINAMATH_CALUDE_division_result_l2286_228617

theorem division_result : (-1/20) / (-1/4 - 2/5 + 9/10 - 3/2) = 1/25 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l2286_228617


namespace NUMINAMATH_CALUDE_sophie_donuts_result_l2286_228635

def sophie_donuts (budget : ℝ) (box_cost : ℝ) (discount_rate : ℝ) 
                   (boxes_bought : ℕ) (donuts_per_box : ℕ) 
                   (boxes_to_mom : ℕ) (donuts_to_sister : ℕ) : ℝ × ℕ :=
  let total_cost := box_cost * boxes_bought
  let discounted_cost := total_cost * (1 - discount_rate)
  let total_donuts := boxes_bought * donuts_per_box
  let donuts_given := boxes_to_mom * donuts_per_box + donuts_to_sister
  let donuts_left := total_donuts - donuts_given
  (discounted_cost, donuts_left)

theorem sophie_donuts_result : 
  sophie_donuts 50 12 0.1 4 12 1 6 = (43.2, 30) :=
by sorry

end NUMINAMATH_CALUDE_sophie_donuts_result_l2286_228635


namespace NUMINAMATH_CALUDE_arccos_cos_eight_l2286_228601

theorem arccos_cos_eight (h : 0 ≤ 8 - 2 * Real.pi ∧ 8 - 2 * Real.pi < Real.pi) :
  Real.arccos (Real.cos 8) = 8 - 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_eight_l2286_228601


namespace NUMINAMATH_CALUDE_special_triangle_sides_l2286_228676

/-- A triangle with consecutive integer side lengths and a median perpendicular to an angle bisector -/
structure SpecialTriangle where
  a : ℕ
  has_consecutive_sides : a > 0
  median_perpendicular_to_bisector : Bool

/-- The sides of a special triangle are 2, 3, and 4 -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 2 := by
  sorry

#check special_triangle_sides

end NUMINAMATH_CALUDE_special_triangle_sides_l2286_228676


namespace NUMINAMATH_CALUDE_nickels_per_stack_l2286_228610

theorem nickels_per_stack (total_nickels : ℕ) (num_stacks : ℕ) 
  (h1 : total_nickels = 72) 
  (h2 : num_stacks = 9) : 
  total_nickels / num_stacks = 8 := by
  sorry

end NUMINAMATH_CALUDE_nickels_per_stack_l2286_228610


namespace NUMINAMATH_CALUDE_right_triangle_moment_of_inertia_l2286_228666

/-- Moment of inertia of a right triangle relative to its hypotenuse -/
theorem right_triangle_moment_of_inertia (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  let c := Real.sqrt (a^2 + b^2)
  let moment_of_inertia := (a^2 + b^2) / 18
  moment_of_inertia = (a^2 + b^2) / 18 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_moment_of_inertia_l2286_228666


namespace NUMINAMATH_CALUDE_fraction_equality_l2286_228665

theorem fraction_equality (a b : ℝ) : (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2286_228665


namespace NUMINAMATH_CALUDE_stamps_for_heavier_envelopes_l2286_228651

/-- Represents the number of stamps required for each weight category --/
def stamps_required (weight : ℕ) : ℕ :=
  if weight < 5 then 2
  else if weight ≤ 10 then 5
  else 7

/-- The total number of stamps purchased --/
def total_stamps : ℕ := 126

/-- The number of envelopes weighing less than 5 pounds --/
def light_envelopes : ℕ := 6

/-- Theorem stating that the total number of stamps used for envelopes weighing 
    5-10 lbs and >10 lbs is 114 --/
theorem stamps_for_heavier_envelopes :
  ∃ (medium heavy : ℕ),
    total_stamps = 
      light_envelopes * stamps_required 4 + 
      medium * stamps_required 5 + 
      heavy * stamps_required 11 ∧
    medium * stamps_required 5 + heavy * stamps_required 11 = 114 :=
sorry

end NUMINAMATH_CALUDE_stamps_for_heavier_envelopes_l2286_228651


namespace NUMINAMATH_CALUDE_complement_of_at_most_one_hit_l2286_228631

/-- Represents the outcome of a single shot -/
inductive ShotOutcome
| Hit
| Miss

/-- Represents the outcome of two consecutive shots -/
def TwoShotOutcome := ShotOutcome × ShotOutcome

/-- The event "at most one shot hits the target" -/
def atMostOneHit (outcome : TwoShotOutcome) : Prop :=
  match outcome with
  | (ShotOutcome.Hit, ShotOutcome.Miss) => True
  | (ShotOutcome.Miss, ShotOutcome.Hit) => True
  | (ShotOutcome.Miss, ShotOutcome.Miss) => True
  | _ => False

/-- The event "both shots hit the target" -/
def bothHit (outcome : TwoShotOutcome) : Prop :=
  outcome = (ShotOutcome.Hit, ShotOutcome.Hit)

theorem complement_of_at_most_one_hit :
  ∀ (outcome : TwoShotOutcome), ¬(atMostOneHit outcome) ↔ bothHit outcome := by
  sorry

end NUMINAMATH_CALUDE_complement_of_at_most_one_hit_l2286_228631


namespace NUMINAMATH_CALUDE_units_digit_of_product_is_two_l2286_228609

def first_composite : ℕ := 4
def second_composite : ℕ := 6
def third_composite : ℕ := 8

def product_of_first_three_composites : ℕ := first_composite * second_composite * third_composite

theorem units_digit_of_product_is_two :
  product_of_first_three_composites % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_is_two_l2286_228609


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l2286_228698

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 120 → books_sold = 39 → books_per_shelf = 9 →
  (initial_stock - books_sold) / books_per_shelf = 9 := by
sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l2286_228698


namespace NUMINAMATH_CALUDE_specific_polyhedron_volume_l2286_228647

/-- Represents a polyhedron formed by folding a specific figure -/
structure Polyhedron where
  /-- Number of isosceles right triangles in the figure -/
  num_triangles : Nat
  /-- Number of squares in the figure -/
  num_squares : Nat
  /-- Number of regular hexagons in the figure -/
  num_hexagons : Nat
  /-- Side length of the isosceles right triangles -/
  triangle_side : ℝ
  /-- Side length of the squares -/
  square_side : ℝ
  /-- Side length of the regular hexagon -/
  hexagon_side : ℝ

/-- Calculates the volume of the polyhedron -/
def volume (p : Polyhedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific polyhedron -/
theorem specific_polyhedron_volume :
  let p : Polyhedron := {
    num_triangles := 3,
    num_squares := 3,
    num_hexagons := 1,
    triangle_side := 2,
    square_side := 2,
    hexagon_side := Real.sqrt 8
  }
  volume p = 47 / 6 := by
  sorry

end NUMINAMATH_CALUDE_specific_polyhedron_volume_l2286_228647


namespace NUMINAMATH_CALUDE_evaluate_expression_l2286_228654

theorem evaluate_expression (x z : ℤ) (hx : x = 4) (hz : z = -2) :
  z * (z - 4 * x) = 36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2286_228654


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2286_228615

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_ratio : (a 2 + a 3) / (a 1 + a 2) = 2)
  (h_fourth : a 4 = 8) :
  a 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2286_228615


namespace NUMINAMATH_CALUDE_angle_A_measure_l2286_228681

-- Define the angles A and B
def angle_A : ℝ := sorry
def angle_B : ℝ := sorry

-- State the theorem
theorem angle_A_measure :
  (angle_A = 2 * angle_B - 15) →  -- Condition 1
  (angle_A + angle_B = 180) →     -- Condition 2 (supplementary angles)
  angle_A = 115 := by             -- Conclusion
sorry


end NUMINAMATH_CALUDE_angle_A_measure_l2286_228681


namespace NUMINAMATH_CALUDE_speed_conversion_l2286_228649

/-- Conversion factor from kilometers per hour to meters per second -/
def kmph_to_ms : ℝ := 0.277778

/-- Given speed in kilometers per hour -/
def given_speed : ℝ := 252

/-- Equivalent speed in meters per second -/
def equivalent_speed : ℝ := 70

/-- Theorem stating that the given speed in kmph is equal to the equivalent speed in m/s -/
theorem speed_conversion :
  given_speed * kmph_to_ms = equivalent_speed := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l2286_228649


namespace NUMINAMATH_CALUDE_orange_preference_percentage_l2286_228659

def survey_results : List (String × Nat) :=
  [("Red", 70), ("Orange", 50), ("Green", 60), ("Yellow", 80), ("Blue", 40), ("Purple", 50)]

def total_responses : Nat :=
  (survey_results.map (λ (_, count) => count)).sum

def orange_preference : Nat :=
  match survey_results.find? (λ (color, _) => color = "Orange") with
  | some (_, count) => count
  | none => 0

theorem orange_preference_percentage :
  (orange_preference : ℚ) / (total_responses : ℚ) * 100 = 14 := by sorry

end NUMINAMATH_CALUDE_orange_preference_percentage_l2286_228659


namespace NUMINAMATH_CALUDE_max_product_of_sum_77_l2286_228622

theorem max_product_of_sum_77 (a b c d e f g h : ℕ) : 
  a + b + c + d + e + f + g + h = 77 →
  a * e + a * f + a * g + a * h +
  b * e + b * f + b * g + b * h +
  c * e + c * f + c * g + c * h +
  d * e + d * f + d * g + d * h ≤ 1440 := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_sum_77_l2286_228622


namespace NUMINAMATH_CALUDE_remainder_theorem_l2286_228646

theorem remainder_theorem : (439 * 319 * 2012 + 2013) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2286_228646


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_range_a_range_l2286_228653

/-- A function f(x) = x^3 - ax that is monotonically decreasing on (-1/2, 0) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

/-- The property of f being monotonically decreasing on (-1/2, 0) -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, -1/2 < x ∧ x < y ∧ y < 0 → f a x > f a y

/-- The theorem stating that if f is monotonically decreasing on (-1/2, 0), then a ≥ 3/4 -/
theorem monotone_decreasing_implies_a_range (a : ℝ) :
  is_monotone_decreasing a → a ≥ 3/4 := by sorry

/-- The main theorem proving the range of a -/
theorem a_range : 
  {a : ℝ | is_monotone_decreasing a} = {a : ℝ | a ≥ 3/4} := by sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_range_a_range_l2286_228653


namespace NUMINAMATH_CALUDE_exam_score_unique_solution_l2286_228632

theorem exam_score_unique_solution (n : ℕ) : 
  (∃ t : ℚ, t > 0 ∧ 
    15 * t + (1/3 : ℚ) * ((n : ℚ) - 20) * t = (1/2 : ℚ) * (n : ℚ) * t) → 
  n = 50 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_unique_solution_l2286_228632


namespace NUMINAMATH_CALUDE_sector_area_l2286_228655

/-- The area of a circular sector with central angle 120° and radius 3/2 is 3/4 π. -/
theorem sector_area (angle : Real) (radius : Real) : 
  angle = 120 * π / 180 → radius = 3 / 2 → 
  (angle / (2 * π)) * π * radius^2 = 3 / 4 * π := by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l2286_228655


namespace NUMINAMATH_CALUDE_christy_tanya_spending_ratio_l2286_228641

/-- Represents the spending of Christy and Tanya at Target -/
structure TargetShopping where
  christy_spent : ℕ
  tanya_face_moisturizer_price : ℕ
  tanya_face_moisturizer_quantity : ℕ
  tanya_body_lotion_price : ℕ
  tanya_body_lotion_quantity : ℕ
  total_spent : ℕ

/-- Calculates Tanya's total spending -/
def tanya_total_spent (shopping : TargetShopping) : ℕ :=
  shopping.tanya_face_moisturizer_price * shopping.tanya_face_moisturizer_quantity +
  shopping.tanya_body_lotion_price * shopping.tanya_body_lotion_quantity

/-- Theorem stating the ratio of Christy's spending to Tanya's spending -/
theorem christy_tanya_spending_ratio (shopping : TargetShopping)
  (h1 : shopping.tanya_face_moisturizer_price = 50)
  (h2 : shopping.tanya_face_moisturizer_quantity = 2)
  (h3 : shopping.tanya_body_lotion_price = 60)
  (h4 : shopping.tanya_body_lotion_quantity = 4)
  (h5 : shopping.total_spent = 1020)
  (h6 : shopping.christy_spent + tanya_total_spent shopping = shopping.total_spent) :
  2 * tanya_total_spent shopping = shopping.christy_spent := by
  sorry

#check christy_tanya_spending_ratio

end NUMINAMATH_CALUDE_christy_tanya_spending_ratio_l2286_228641


namespace NUMINAMATH_CALUDE_group_size_proof_l2286_228692

def group_size (adult_meal_cost : ℕ) (total_cost : ℕ) (num_kids : ℕ) : ℕ :=
  (total_cost / adult_meal_cost) + num_kids

theorem group_size_proof :
  group_size 2 14 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l2286_228692


namespace NUMINAMATH_CALUDE_smallest_possible_d_l2286_228639

theorem smallest_possible_d : 
  let f : ℝ → ℝ := λ d => Real.sqrt ((2 * Real.sqrt 10)^2 + (d + 5)^2) - 4 * d
  ∃ d : ℝ, f d = 0 ∧ ∀ d' : ℝ, f d' = 0 → d ≤ d' ∧ d = (1 + 2 * Real.sqrt 10) / 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l2286_228639


namespace NUMINAMATH_CALUDE_alfonso_savings_l2286_228648

theorem alfonso_savings (daily_rate : ℕ) (days_per_week : ℕ) (total_weeks : ℕ) (helmet_cost : ℕ) :
  let total_earned := daily_rate * days_per_week * total_weeks
  helmet_cost - total_earned = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_alfonso_savings_l2286_228648


namespace NUMINAMATH_CALUDE_students_playing_sports_l2286_228618

theorem students_playing_sports (basketball cricket both : ℕ) 
  (h1 : basketball = 12) 
  (h2 : cricket = 8) 
  (h3 : both = 3) : 
  basketball + cricket - both = 17 := by
sorry

end NUMINAMATH_CALUDE_students_playing_sports_l2286_228618


namespace NUMINAMATH_CALUDE_dividend_calculation_l2286_228690

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 9) :
  divisor * quotient + remainder = 162 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2286_228690


namespace NUMINAMATH_CALUDE_stream_speed_l2286_228673

/-- Given Julie's rowing distances and times, prove that the speed of the stream is 5 km/h -/
theorem stream_speed (v_j v_s : ℝ) 
  (h1 : 32 / (v_j - v_s) = 4)  -- Upstream equation
  (h2 : 72 / (v_j + v_s) = 4)  -- Downstream equation
  : v_s = 5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2286_228673


namespace NUMINAMATH_CALUDE_set_intersection_equality_l2286_228624

def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem set_intersection_equality : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 16} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l2286_228624


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2286_228694

theorem arithmetic_computation : 12 + 4 * (5 - 2 * 3)^2 = 16 := by sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2286_228694


namespace NUMINAMATH_CALUDE_find_number_l2286_228669

theorem find_number (a b : ℕ+) (hcf : Nat.gcd a b = 12) (lcm : Nat.lcm a b = 396) (hb : b = 198) : a = 24 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2286_228669


namespace NUMINAMATH_CALUDE_adam_bought_more_cat_food_l2286_228650

/-- Represents the number of packages of cat food Adam bought -/
def cat_packages : ℕ := 9

/-- Represents the number of packages of dog food Adam bought -/
def dog_packages : ℕ := 7

/-- Represents the number of cans in each package of cat food -/
def cans_per_cat_package : ℕ := 10

/-- Represents the number of cans in each package of dog food -/
def cans_per_dog_package : ℕ := 5

/-- Calculates the difference between the total number of cans of cat food and dog food -/
def cans_difference : ℕ := 
  cat_packages * cans_per_cat_package - dog_packages * cans_per_dog_package

theorem adam_bought_more_cat_food : cans_difference = 55 := by
  sorry

end NUMINAMATH_CALUDE_adam_bought_more_cat_food_l2286_228650


namespace NUMINAMATH_CALUDE_sandwiches_theorem_l2286_228627

/-- The number of sandwiches Ruth prepared -/
def total_sandwiches : ℕ := 10

/-- The number of sandwiches Ruth ate -/
def ruth_ate : ℕ := 1

/-- The number of sandwiches Ruth's brother ate -/
def brother_ate : ℕ := 2

/-- The number of sandwiches Ruth's first cousin ate -/
def first_cousin_ate : ℕ := 2

/-- The number of sandwiches each of Ruth's other two cousins ate -/
def other_cousins_ate_each : ℕ := 1

/-- The number of sandwiches left -/
def sandwiches_left : ℕ := 3

/-- Theorem stating that the total number of sandwiches Ruth prepared
    is equal to the sum of sandwiches eaten by everyone and those left -/
theorem sandwiches_theorem :
  total_sandwiches = ruth_ate + brother_ate + first_cousin_ate +
    (2 * other_cousins_ate_each) + sandwiches_left :=
by
  sorry

end NUMINAMATH_CALUDE_sandwiches_theorem_l2286_228627


namespace NUMINAMATH_CALUDE_max_distance_line_ellipse_intersection_l2286_228645

/-- The maximum distance between two intersection points of a line and an ellipse -/
theorem max_distance_line_ellipse_intersection :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 + 4*y^2 = 4}
  let line (m : ℝ) := {(x, y) : ℝ × ℝ | y = x + m}
  let intersection (m : ℝ) := {p : ℝ × ℝ | p ∈ ellipse ∧ p ∈ line m}
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (m : ℝ), ∀ (p q : ℝ × ℝ), p ∈ intersection m → q ∈ intersection m → p ≠ q →
    distance p q ≤ (4/5) * Real.sqrt 10 ∧
    ∃ (m' : ℝ) (p' q' : ℝ × ℝ), p' ∈ intersection m' ∧ q' ∈ intersection m' ∧ p' ≠ q' ∧
      distance p' q' = (4/5) * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_max_distance_line_ellipse_intersection_l2286_228645


namespace NUMINAMATH_CALUDE_key_arrangement_theorem_l2286_228629

/-- The number of permutations of n elements -/
def totalPermutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of permutations of n elements with exactly one cycle -/
def onePermutation (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of permutations of 10 elements where at least two cycles are present -/
def atLeastTwoCycles : ℕ := totalPermutations 10 - onePermutation 10

/-- The number of permutations of 10 elements with exactly two cycles -/
def exactlyTwoCycles : ℕ :=
  choose 10 1 * Nat.factorial 8 +
  choose 10 2 * Nat.factorial 7 +
  choose 10 3 * Nat.factorial 2 * Nat.factorial 6 +
  choose 10 4 * Nat.factorial 3 * Nat.factorial 5 +
  (choose 10 5 * Nat.factorial 4 * Nat.factorial 4) / 2

theorem key_arrangement_theorem :
  atLeastTwoCycles = 9 * Nat.factorial 9 ∧ exactlyTwoCycles = 1024576 := by sorry

end NUMINAMATH_CALUDE_key_arrangement_theorem_l2286_228629


namespace NUMINAMATH_CALUDE_third_train_speed_l2286_228657

/-- Calculates the speed of the third train given the conditions of the problem -/
theorem third_train_speed
  (man_train_speed : ℝ)
  (goods_train_length : ℝ)
  (third_train_length : ℝ)
  (goods_train_pass_time : ℝ)
  (third_train_pass_time : ℝ)
  (h_man_train_speed : man_train_speed = 45)
  (h_goods_train_length : goods_train_length = 340)
  (h_third_train_length : third_train_length = 480)
  (h_goods_train_pass_time : goods_train_pass_time = 8)
  (h_third_train_pass_time : third_train_pass_time = 12) :
  ∃ (third_train_speed : ℝ), third_train_speed = 99 := by
  sorry


end NUMINAMATH_CALUDE_third_train_speed_l2286_228657


namespace NUMINAMATH_CALUDE_factorial_ratio_2016_l2286_228630

-- Define factorial
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_ratio_2016 :
  (factorial 2016)^2 / (factorial 2015 * factorial 2017) = 2016 / 2017 :=
by sorry

end NUMINAMATH_CALUDE_factorial_ratio_2016_l2286_228630


namespace NUMINAMATH_CALUDE_opposite_of_neg_three_l2286_228602

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- Theorem stating that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_three_l2286_228602


namespace NUMINAMATH_CALUDE_school_pencils_l2286_228671

theorem school_pencils (num_pens : ℕ) (pencil_cost pen_cost total_cost : ℚ) :
  num_pens = 56 ∧
  pencil_cost = 5/2 ∧
  pen_cost = 7/2 ∧
  total_cost = 291 →
  ∃ num_pencils : ℕ, num_pencils * pencil_cost + num_pens * pen_cost = total_cost ∧ num_pencils = 38 :=
by sorry

end NUMINAMATH_CALUDE_school_pencils_l2286_228671


namespace NUMINAMATH_CALUDE_common_tangent_sum_l2286_228628

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = x^2 + 52/5

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = y^2 + 25/10

/-- Common tangent line L -/
def L (a b c x y : ℝ) : Prop := a*x + b*y = c

/-- Theorem stating the sum of a, b, and c for the common tangent line -/
theorem common_tangent_sum (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), P₁ x₁ y₁ ∧ P₂ x₂ y₂ ∧ L a b c x₁ y₁ ∧ L a b c x₂ y₂) →
  (∃ (k : ℚ), a = k * b) →
  Nat.gcd a (Nat.gcd b c) = 1 →
  a + b + c = 17 := by
  sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l2286_228628


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2286_228607

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  a 0 = 3 →
  a 1 = 8 →
  a 2 = 13 →
  a 5 = 28 →
  (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →
  a 3 + a 4 = 41 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2286_228607


namespace NUMINAMATH_CALUDE_square_equality_solution_l2286_228634

theorem square_equality_solution : ∃ x : ℝ, (9 - x)^2 = x^2 ∧ x = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_solution_l2286_228634


namespace NUMINAMATH_CALUDE_find_x1_l2286_228684

theorem find_x1 (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 3/4) :
  x1 = 3 * Real.sqrt 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_find_x1_l2286_228684


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2286_228637

/-- The equation of an ellipse in its standard form -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 = 1

/-- The length of the major axis of an ellipse -/
def major_axis_length : ℝ := 8

/-- Theorem: The length of the major axis of the ellipse x^2/16 + y^2 = 1 is 8 -/
theorem ellipse_major_axis_length :
  ∀ x y : ℝ, is_ellipse x y → major_axis_length = 8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2286_228637


namespace NUMINAMATH_CALUDE_total_water_consumption_is_417_total_water_consumption_proof_l2286_228626

/-- Represents a washing machine with water consumption rates for different wash types -/
structure WashingMachine where
  heavy_wash : ℕ
  regular_wash : ℕ
  light_wash : ℕ

/-- Calculates the total water consumption for a washing machine -/
def water_consumption (m : WashingMachine) (heavy regular light bleach : ℕ) : ℕ :=
  m.heavy_wash * heavy + m.regular_wash * regular + m.light_wash * (light + bleach)

/-- Theorem: The total water consumption for all machines is 417 gallons -/
theorem total_water_consumption_is_417 : ℕ :=
  let machine_a : WashingMachine := ⟨25, 15, 3⟩
  let machine_b : WashingMachine := ⟨20, 12, 2⟩
  let machine_c : WashingMachine := ⟨30, 18, 4⟩
  
  let total_consumption :=
    water_consumption machine_a 3 4 2 4 +
    water_consumption machine_b 2 3 1 3 +
    water_consumption machine_c 4 2 1 5

  417

theorem total_water_consumption_proof :
  (let machine_a : WashingMachine := ⟨25, 15, 3⟩
   let machine_b : WashingMachine := ⟨20, 12, 2⟩
   let machine_c : WashingMachine := ⟨30, 18, 4⟩
   
   let total_consumption :=
     water_consumption machine_a 3 4 2 4 +
     water_consumption machine_b 2 3 1 3 +
     water_consumption machine_c 4 2 1 5

   total_consumption) = 417 := by
  sorry

end NUMINAMATH_CALUDE_total_water_consumption_is_417_total_water_consumption_proof_l2286_228626


namespace NUMINAMATH_CALUDE_dog_food_calculation_l2286_228643

theorem dog_food_calculation (num_dogs : ℕ) (total_food_kg : ℕ) (num_days : ℕ) 
  (h1 : num_dogs = 4)
  (h2 : total_food_kg = 14)
  (h3 : num_days = 14) :
  (total_food_kg * 1000) / (num_dogs * num_days) = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_dog_food_calculation_l2286_228643


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l2286_228677

theorem cube_surface_area_increase :
  ∀ (s : ℝ), s > 0 →
  let original_surface_area := 6 * s^2
  let new_edge_length := 1.4 * s
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 0.96 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l2286_228677


namespace NUMINAMATH_CALUDE_trajectory_equation_l2286_228664

theorem trajectory_equation (x y : ℝ) (h : x ≠ 0) :
  (y + Real.sqrt 2) / x * (y - Real.sqrt 2) / x = -2 →
  y^2 / 2 + x^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2286_228664


namespace NUMINAMATH_CALUDE_smallest_N_is_110_l2286_228658

/-- Represents a point in the rectangular array -/
structure Point where
  row : Fin 6
  col : ℕ

/-- The x-coordinate of a point after initial numbering -/
def x (p : Point) (N : ℕ) : ℕ := p.row.val * N + p.col

/-- The y-coordinate of a point after renumbering -/
def y (p : Point) : ℕ := (p.col - 1) * 6 + p.row.val + 1

/-- Predicate that checks if the given conditions are satisfied -/
def satisfiesConditions (N : ℕ) (p₁ p₂ p₃ p₄ p₅ p₆ : Point) : Prop :=
  x p₁ N = y p₂ ∧
  x p₂ N = y p₁ ∧
  x p₃ N = y p₄ ∧
  x p₄ N = y p₅ ∧
  x p₅ N = y p₆ ∧
  x p₆ N = y p₃

/-- The main theorem stating that 110 is the smallest N satisfying the conditions -/
theorem smallest_N_is_110 :
  ∃ (p₁ p₂ p₃ p₄ p₅ p₆ : Point),
    satisfiesConditions 110 p₁ p₂ p₃ p₄ p₅ p₆ ∧
    ∀ (N : ℕ), N < 110 → ¬∃ (q₁ q₂ q₃ q₄ q₅ q₆ : Point),
      satisfiesConditions N q₁ q₂ q₃ q₄ q₅ q₆ :=
by sorry

end NUMINAMATH_CALUDE_smallest_N_is_110_l2286_228658


namespace NUMINAMATH_CALUDE_nathan_family_storage_cost_l2286_228663

/-- The cost to store items for a group at the temple shop -/
def storage_cost (num_people : ℕ) (objects_per_person : ℕ) (cost_per_object : ℕ) : ℕ :=
  num_people * objects_per_person * cost_per_object

/-- Proof that the storage cost for Nathan and his parents is 165 dollars -/
theorem nathan_family_storage_cost :
  storage_cost 3 5 11 = 165 := by
  sorry

end NUMINAMATH_CALUDE_nathan_family_storage_cost_l2286_228663


namespace NUMINAMATH_CALUDE_work_completion_time_l2286_228621

/-- 
Given:
- A's work rate is half of B's
- A and B together finish a job in 32 days
Prove that B alone will finish the job in 48 days
-/
theorem work_completion_time (a b : ℝ) (h1 : a = (1/2) * b) (h2 : (a + b) * 32 = 1) :
  (1 / b) = 48 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2286_228621


namespace NUMINAMATH_CALUDE_blackjack_payout_ratio_l2286_228603

/-- Represents the payout ratio for a blackjack in a casino game -/
structure BlackjackPayout where
  original_bet : ℚ
  total_payout : ℚ

/-- Calculates the payout ratio for a blackjack given the original bet and total payout -/
def payout_ratio (bp : BlackjackPayout) : ℚ × ℚ :=
  let winnings := bp.total_payout - bp.original_bet
  (winnings, bp.original_bet)

/-- Theorem stating that for the given conditions, the payout ratio is 1:2 -/
theorem blackjack_payout_ratio :
  let bp := BlackjackPayout.mk 40 60
  payout_ratio bp = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_blackjack_payout_ratio_l2286_228603


namespace NUMINAMATH_CALUDE_max_value_F_l2286_228604

/-- The function f(x) = ax² + bx + c -/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The function g(x) = cx² + bx + a -/
def g (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

/-- The function F(x) = |f(x) · g(x)| -/
def F (a b c x : ℝ) : ℝ := |f a b c x * g a b c x|

theorem max_value_F (a b c : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f a b c x| ≤ 1) →
  ∃ M, M = 2 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, F a b c x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_F_l2286_228604


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2286_228679

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem subset_implies_a_equals_one (a : ℝ) (h1 : B a ⊆ A) (h2 : a > 0) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2286_228679


namespace NUMINAMATH_CALUDE_more_boys_than_girls_boy_girl_difference_l2286_228699

/-- The number of girls in the school -/
def num_girls : ℕ := 635

/-- The number of boys in the school -/
def num_boys : ℕ := 1145

/-- There are more boys than girls -/
theorem more_boys_than_girls : num_boys > num_girls := by sorry

/-- The difference between the number of boys and girls is 510 -/
theorem boy_girl_difference : num_boys - num_girls = 510 := by sorry

end NUMINAMATH_CALUDE_more_boys_than_girls_boy_girl_difference_l2286_228699


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l2286_228616

/-- Two vectors are parallel if and only if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, y)
  parallel a b → y = 6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l2286_228616
