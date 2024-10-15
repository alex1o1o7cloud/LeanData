import Mathlib

namespace NUMINAMATH_CALUDE_starters_count_theorem_l3245_324585

def number_of_players : ℕ := 15
def number_of_starters : ℕ := 5

-- Define a function to calculate the number of ways to choose starters
def choose_starters (total : ℕ) (choose : ℕ) : ℕ := Nat.choose total choose

-- Define a function to calculate the number of ways to choose starters excluding both twins
def choose_starters_excluding_twins (total : ℕ) (choose : ℕ) : ℕ :=
  choose_starters total choose - choose_starters (total - 2) (choose - 2)

theorem starters_count_theorem : 
  choose_starters_excluding_twins number_of_players number_of_starters = 2717 := by
  sorry

end NUMINAMATH_CALUDE_starters_count_theorem_l3245_324585


namespace NUMINAMATH_CALUDE_triangle_max_area_l3245_324532

theorem triangle_max_area (a b c : ℝ) (h : 2 * a^2 + b^2 + c^2 = 4) :
  let S := (1/2) * a * b * Real.sqrt (1 - ((b^2 + c^2 - a^2) / (2*b*c))^2)
  S ≤ Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3245_324532


namespace NUMINAMATH_CALUDE_complex_fraction_real_iff_m_eq_neg_one_l3245_324526

/-- The complex number (m^2 + i) / (1 - mi) is real if and only if m = -1 -/
theorem complex_fraction_real_iff_m_eq_neg_one (m : ℝ) :
  (((m^2 : ℂ) + Complex.I) / (1 - m * Complex.I)).im = 0 ↔ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_iff_m_eq_neg_one_l3245_324526


namespace NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l3245_324530

theorem unique_solution_logarithmic_equation :
  ∃! x : ℝ, x > 0 ∧ x^(Real.log x / Real.log 10) = x^2 / 10 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l3245_324530


namespace NUMINAMATH_CALUDE_tan_sum_problem_l3245_324586

theorem tan_sum_problem (α β : ℝ) 
  (h1 : Real.tan (α + 2 * β) = 2) 
  (h2 : Real.tan β = -3) : 
  Real.tan (α + β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_problem_l3245_324586


namespace NUMINAMATH_CALUDE_sector_area_l3245_324558

/-- The area of a circular sector with given radius and arc length. -/
theorem sector_area (r : ℝ) (arc_length : ℝ) (h : r > 0) :
  let area := (1 / 2) * r * arc_length
  r = 15 ∧ arc_length = π / 3 → area = 5 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3245_324558


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l3245_324520

theorem farmer_tomatoes (initial : ℕ) (picked : ℕ) (difference : ℕ) 
  (h1 : picked = 9)
  (h2 : difference = 8)
  (h3 : initial - picked = difference) :
  initial = 17 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l3245_324520


namespace NUMINAMATH_CALUDE_cube_sum_from_conditions_l3245_324531

theorem cube_sum_from_conditions (x y : ℝ) 
  (sum_condition : x + y = 5)
  (sum_squares_condition : x^2 + y^2 = 20) :
  x^3 + y^3 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_conditions_l3245_324531


namespace NUMINAMATH_CALUDE_lunch_to_novel_ratio_l3245_324503

theorem lunch_to_novel_ratio (initial_amount : ℕ) (novel_cost : ℕ) (remaining_amount : ℕ)
  (h1 : initial_amount = 50)
  (h2 : novel_cost = 7)
  (h3 : remaining_amount = 29) :
  (initial_amount - novel_cost - remaining_amount) / novel_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_lunch_to_novel_ratio_l3245_324503


namespace NUMINAMATH_CALUDE_lemonade_stand_profit_l3245_324554

/-- Calculate the profit from a lemonade stand --/
theorem lemonade_stand_profit
  (lemon_cost sugar_cost cup_cost : ℕ)
  (price_per_cup cups_sold : ℕ)
  (h1 : lemon_cost = 10)
  (h2 : sugar_cost = 5)
  (h3 : cup_cost = 3)
  (h4 : price_per_cup = 4)
  (h5 : cups_sold = 21) :
  (price_per_cup * cups_sold) - (lemon_cost + sugar_cost + cup_cost) = 66 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_stand_profit_l3245_324554


namespace NUMINAMATH_CALUDE_fraction_division_equivalence_l3245_324539

theorem fraction_division_equivalence : 5 / (8 / 13) = 65 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equivalence_l3245_324539


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l3245_324577

theorem degree_to_radian_conversion (π : Real) (h : π * 1 = 180) :
  -885 * (π / 180) = -59 / 12 * π := by
  sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l3245_324577


namespace NUMINAMATH_CALUDE_unbounded_function_identity_l3245_324544

/-- A function f: ℤ → ℤ is unbounded if for any integer N, there exists an x such that |f(x)| > N -/
def Unbounded (f : ℤ → ℤ) : Prop :=
  ∀ N : ℤ, ∃ x : ℤ, |f x| > N

/-- The main theorem: if f is unbounded and satisfies the given condition, then f(x) = x for all x -/
theorem unbounded_function_identity
  (f : ℤ → ℤ)
  (h_unbounded : Unbounded f)
  (h_condition : ∀ x y : ℤ, (f (f x - y)) ∣ (x - f y)) :
  ∀ x : ℤ, f x = x :=
sorry

end NUMINAMATH_CALUDE_unbounded_function_identity_l3245_324544


namespace NUMINAMATH_CALUDE_triangle_angle_C_l3245_324590

theorem triangle_angle_C (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → 
  5 * Real.sin A + 2 * Real.cos B = 3 →
  2 * Real.sin B + 5 * Real.tan A = 7 →
  Real.sin C = Real.sin (A + B) := by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l3245_324590


namespace NUMINAMATH_CALUDE_money_left_over_calculation_l3245_324574

/-- The amount of money left over after purchasing bread and peanut butter -/
def money_left_over (bread_price : ℚ) (bread_quantity : ℕ) (peanut_butter_price : ℚ) (initial_amount : ℚ) : ℚ :=
  initial_amount - (bread_price * bread_quantity + peanut_butter_price)

/-- Theorem stating the amount of money left over in the given scenario -/
theorem money_left_over_calculation :
  let bread_price : ℚ := 9/4  -- $2.25 as a rational number
  let bread_quantity : ℕ := 3
  let peanut_butter_price : ℚ := 2
  let initial_amount : ℚ := 14
  money_left_over bread_price bread_quantity peanut_butter_price initial_amount = 21/4  -- $5.25 as a rational number
  := by sorry

end NUMINAMATH_CALUDE_money_left_over_calculation_l3245_324574


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l3245_324596

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l3245_324596


namespace NUMINAMATH_CALUDE_estimate_fish_population_l3245_324569

/-- Estimates the total number of fish in a pond using the capture-recapture method. -/
theorem estimate_fish_population (initial_catch : ℕ) (second_catch : ℕ) (marked_recaught : ℕ) :
  initial_catch = 60 →
  second_catch = 80 →
  marked_recaught = 5 →
  (initial_catch * second_catch) / marked_recaught = 960 :=
by
  sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l3245_324569


namespace NUMINAMATH_CALUDE_success_probability_given_expectation_l3245_324589

/-- A two-point distribution random variable -/
structure TwoPointDistribution where
  X : ℝ → ℝ  -- The random variable
  p : ℝ      -- Success probability
  h1 : 0 ≤ p ∧ p ≤ 1  -- Probability is between 0 and 1

/-- Expected value of a two-point distribution -/
def expectedValue (T : TwoPointDistribution) : ℝ :=
  T.p * 1 + (1 - T.p) * 0

theorem success_probability_given_expectation 
  (T : TwoPointDistribution) 
  (h : expectedValue T = 0.7) : 
  T.p = 0.7 := by
  sorry


end NUMINAMATH_CALUDE_success_probability_given_expectation_l3245_324589


namespace NUMINAMATH_CALUDE_number_is_perfect_square_l3245_324597

def N : ℕ := (10^1998 * ((10^1997 - 1) / 9)) + 2 * ((10^1998 - 1) / 9)

theorem number_is_perfect_square : 
  N = (10^1998 + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_number_is_perfect_square_l3245_324597


namespace NUMINAMATH_CALUDE_square_greater_than_self_for_x_greater_than_one_l3245_324553

theorem square_greater_than_self_for_x_greater_than_one (x : ℝ) : x > 1 → x^2 > x := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_self_for_x_greater_than_one_l3245_324553


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3245_324550

theorem repeating_decimal_sum (a b : ℕ+) (h1 : (35 : ℚ) / 99 = (a : ℚ) / b) 
  (h2 : Nat.gcd a.val b.val = 1) : a + b = 134 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3245_324550


namespace NUMINAMATH_CALUDE_rectangle_polygon_perimeter_l3245_324529

theorem rectangle_polygon_perimeter : 
  let n : ℕ := 20
  let rectangle_dimensions : ℕ → ℕ × ℕ := λ i => (i, i + 1)
  let perimeter : ℕ := 2 * (List.range (n + 1)).sum
  perimeter = 462 := by sorry

end NUMINAMATH_CALUDE_rectangle_polygon_perimeter_l3245_324529


namespace NUMINAMATH_CALUDE_amoeba_count_after_week_l3245_324556

/-- The number of amoebas in the puddle on a given day -/
def amoeba_count (day : ℕ) : ℕ :=
  if day = 0 then 0
  else if day = 1 then 1
  else 2 * amoeba_count (day - 1)

/-- The theorem stating that after 7 days, there are 64 amoebas in the puddle -/
theorem amoeba_count_after_week : amoeba_count 7 = 64 := by
  sorry

#eval amoeba_count 7  -- This should output 64

end NUMINAMATH_CALUDE_amoeba_count_after_week_l3245_324556


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_and_gcd_l3245_324514

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_divisibility_and_gcd (m n : ℕ) :
  (m ∣ n → fib m ∣ fib n) ∧ (Nat.gcd (fib m) (fib n) = fib (Nat.gcd m n)) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_and_gcd_l3245_324514


namespace NUMINAMATH_CALUDE_subset_intersection_one_element_l3245_324588

/-- Given n+1 distinct subsets of [n], each with exactly 3 elements,
    there must exist a pair of subsets whose intersection has exactly one element. -/
theorem subset_intersection_one_element
  (n : ℕ)
  (A : Fin (n + 1) → Finset (Fin n))
  (h_distinct : ∀ i j, i ≠ j → A i ≠ A j)
  (h_card : ∀ i, (A i).card = 3) :
  ∃ i j, i ≠ j ∧ (A i ∩ A j).card = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_intersection_one_element_l3245_324588


namespace NUMINAMATH_CALUDE_pure_imaginary_quadratic_l3245_324561

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The theorem statement -/
theorem pure_imaginary_quadratic (m : ℝ) :
  IsPureImaginary (Complex.mk (m^2 + m - 2) (m^2 - 1)) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_quadratic_l3245_324561


namespace NUMINAMATH_CALUDE_writer_tea_and_hours_l3245_324521

structure WriterData where
  sunday_hours : ℝ
  sunday_tea : ℝ
  wednesday_hours : ℝ
  thursday_tea : ℝ

def inverse_proportional (x y : ℝ) (k : ℝ) : Prop := x * y = k

theorem writer_tea_and_hours (data : WriterData) :
  inverse_proportional data.sunday_hours data.sunday_tea (data.sunday_hours * data.sunday_tea) →
  inverse_proportional data.wednesday_hours (data.sunday_hours * data.sunday_tea / data.wednesday_hours) (data.sunday_hours * data.sunday_tea) ∧
  inverse_proportional (data.sunday_hours * data.sunday_tea / data.thursday_tea) data.thursday_tea (data.sunday_hours * data.sunday_tea) :=
by
  sorry

#check writer_tea_and_hours

end NUMINAMATH_CALUDE_writer_tea_and_hours_l3245_324521


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3245_324555

theorem quadratic_roots_sum (p q : ℝ) : 
  p^2 - 5*p + 3 = 0 → 
  q^2 - 5*q + 3 = 0 → 
  p^2 + q^2 + p + q = 24 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3245_324555


namespace NUMINAMATH_CALUDE_smallest_number_l3245_324538

theorem smallest_number (S : Set ℤ) (h : S = {-2, 0, -3, 1}) : 
  ∃ m ∈ S, ∀ x ∈ S, m ≤ x ∧ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3245_324538


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_proof_l3245_324563

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def largest_even_digit_multiple_of_9 : ℕ := 8820

theorem largest_even_digit_multiple_of_9_proof :
  (has_only_even_digits largest_even_digit_multiple_of_9) ∧
  (largest_even_digit_multiple_of_9 < 10000) ∧
  (largest_even_digit_multiple_of_9 % 9 = 0) ∧
  (∀ m : ℕ, m > largest_even_digit_multiple_of_9 →
    ¬(has_only_even_digits m ∧ m < 10000 ∧ m % 9 = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_proof_l3245_324563


namespace NUMINAMATH_CALUDE_dealer_truck_sales_l3245_324583

theorem dealer_truck_sales (total : ℕ) (car_truck_diff : ℕ) (trucks : ℕ) : 
  total = 69 → car_truck_diff = 27 → trucks + (trucks + car_truck_diff) = total → trucks = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_dealer_truck_sales_l3245_324583


namespace NUMINAMATH_CALUDE_max_collisions_l3245_324510

/-- Represents an ant walking on a line -/
structure Ant where
  position : ℝ
  speed : ℝ
  direction : Bool -- true for right, false for left

/-- The state of the system at any given time -/
structure AntSystem where
  n : ℕ
  ants : Fin n → Ant

/-- Predicate to check if the total number of collisions is finite -/
def HasFiniteCollisions (system : AntSystem) : Prop := sorry

/-- The number of collisions that have occurred in the system -/
def NumberOfCollisions (system : AntSystem) : ℕ := sorry

/-- Theorem stating the maximum number of collisions possible -/
theorem max_collisions (n : ℕ) (h : n > 0) :
  ∃ (system : AntSystem),
    system.n = n ∧
    HasFiniteCollisions system ∧
    ∀ (other_system : AntSystem),
      other_system.n = n →
      HasFiniteCollisions other_system →
      NumberOfCollisions other_system ≤ NumberOfCollisions system ∧
      NumberOfCollisions system = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_collisions_l3245_324510


namespace NUMINAMATH_CALUDE_star_calculation_l3245_324593

-- Define the * operation
def star (a b : ℤ) : ℤ := a * (a - b)

-- State the theorem
theorem star_calculation : star 2 3 + star (6 - 2) 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l3245_324593


namespace NUMINAMATH_CALUDE_correct_observation_value_l3245_324575

theorem correct_observation_value 
  (n : ℕ) 
  (initial_mean : ℝ) 
  (wrong_value : ℝ) 
  (corrected_mean : ℝ) 
  (h1 : n = 50) 
  (h2 : initial_mean = 36) 
  (h3 : wrong_value = 23) 
  (h4 : corrected_mean = 36.5) : 
  ∃ (correct_value : ℝ), 
    (n : ℝ) * corrected_mean = (n : ℝ) * initial_mean - wrong_value + correct_value ∧ 
    correct_value = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_observation_value_l3245_324575


namespace NUMINAMATH_CALUDE_simplify_expression_l3245_324517

theorem simplify_expression : (5 + 7 + 3) / 3 - 2 / 3 - 1 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3245_324517


namespace NUMINAMATH_CALUDE_square_plus_n_equals_n_times_n_plus_one_l3245_324547

theorem square_plus_n_equals_n_times_n_plus_one (n : ℕ) : n^2 + n = n * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_plus_n_equals_n_times_n_plus_one_l3245_324547


namespace NUMINAMATH_CALUDE_total_oranges_l3245_324567

def orange_groups : ℕ := 16
def oranges_per_group : ℕ := 24

theorem total_oranges : orange_groups * oranges_per_group = 384 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_l3245_324567


namespace NUMINAMATH_CALUDE_bus_travel_time_l3245_324524

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the difference in hours between two times -/
def timeDifference (t1 t2 : TimeOfDay) : Nat :=
  -- Implementation details omitted
  sorry

/-- Theorem: The time difference between 12:30 PM and 9:30 AM is 3 hours -/
theorem bus_travel_time :
  let departure : TimeOfDay := ⟨9, 30, sorry⟩
  let arrival : TimeOfDay := ⟨12, 30, sorry⟩
  timeDifference departure arrival = 3 := by
  sorry

end NUMINAMATH_CALUDE_bus_travel_time_l3245_324524


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3245_324573

theorem cubic_equation_solution (a : ℝ) (h : 2 * a^3 + a^2 - 275 = 0) : a = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3245_324573


namespace NUMINAMATH_CALUDE_f_2008_l3245_324505

-- Define a real-valued function with an inverse
variable (f : ℝ → ℝ)
variable (hf : Function.Bijective f)

-- Define the condition f(9) = 18
axiom f_9 : f 9 = 18

-- Define the inverse relationship for f(x+1)
axiom inverse_shift : Function.LeftInverse (fun x => f⁻¹ (x + 1)) (fun x => f (x + 1))

-- State the theorem
theorem f_2008 : f 2008 = -1981 := by sorry

end NUMINAMATH_CALUDE_f_2008_l3245_324505


namespace NUMINAMATH_CALUDE_care_package_weight_l3245_324560

/-- Represents the weight of the care package contents -/
structure CarePackage where
  jellyBeans : ℝ
  brownies : ℝ
  gummyWorms : ℝ
  chocolateBars : ℝ
  popcorn : ℝ
  cookies : ℝ

/-- Calculates the total weight of the care package -/
def totalWeight (cp : CarePackage) : ℝ :=
  cp.jellyBeans + cp.brownies + cp.gummyWorms + cp.chocolateBars + cp.popcorn + cp.cookies

/-- The final weight of the care package after all modifications -/
def finalWeight (initialWeight : ℝ) : ℝ :=
  let weightAfterChocolate := initialWeight * 1.5
  let weightAfterPopcorn := weightAfterChocolate + 0.5
  let weightAfterCookies := weightAfterPopcorn * 2
  weightAfterCookies - 0.75

theorem care_package_weight :
  let initialPackage : CarePackage := {
    jellyBeans := 1.5,
    brownies := 0.5,
    gummyWorms := 2,
    chocolateBars := 0,
    popcorn := 0,
    cookies := 0
  }
  let initialWeight := totalWeight initialPackage
  finalWeight initialWeight = 12.25 := by
  sorry

end NUMINAMATH_CALUDE_care_package_weight_l3245_324560


namespace NUMINAMATH_CALUDE_zeros_after_decimal_of_fraction_l3245_324598

/-- The number of zeros after the decimal point in the decimal representation of 1/(100^15) -/
def zeros_after_decimal : ℕ := 30

/-- The fraction we're considering -/
def fraction : ℚ := 1 / (100 ^ 15)

theorem zeros_after_decimal_of_fraction :
  (∃ (x : ℚ), x * 10^zeros_after_decimal = fraction ∧ 
   x ≥ 1/10 ∧ x < 1) ∧
  (∀ (n : ℕ), n < zeros_after_decimal → 
   ∃ (y : ℚ), y * 10^n = fraction ∧ y < 1/10) :=
sorry

end NUMINAMATH_CALUDE_zeros_after_decimal_of_fraction_l3245_324598


namespace NUMINAMATH_CALUDE_collinear_points_sum_l3245_324502

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop := sorry

/-- The theorem states that if the given points are collinear, then p + q = 6. -/
theorem collinear_points_sum (p q : ℝ) : 
  collinear (2, p, q) (p, 3, q) (p, q, 4) → p + q = 6 := by sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l3245_324502


namespace NUMINAMATH_CALUDE_raft_drift_theorem_l3245_324546

/-- The time for a raft to drift between two villages -/
def raft_drift_time (distance : ℝ) (steamboat_time : ℝ) (motorboat_time : ℝ) : ℝ :=
  90

/-- Theorem: The raft drift time is 90 minutes given the conditions -/
theorem raft_drift_theorem (distance : ℝ) (steamboat_time : ℝ) (motorboat_time : ℝ) 
  (h1 : distance = 1)
  (h2 : steamboat_time = 1)
  (h3 : motorboat_time = 45 / 60)
  (h4 : ∃ (steamboat_speed : ℝ), 
    motorboat_time = distance / (2 * steamboat_speed + (distance / steamboat_time - steamboat_speed))) :
  raft_drift_time distance steamboat_time motorboat_time = 90 := by
  sorry

#check raft_drift_theorem

end NUMINAMATH_CALUDE_raft_drift_theorem_l3245_324546


namespace NUMINAMATH_CALUDE_mangoes_purchased_correct_mango_kg_l3245_324576

theorem mangoes_purchased (grape_kg : ℕ) (grape_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : ℕ :=
  let mango_kg := (total_paid - grape_kg * grape_rate) / mango_rate
  mango_kg

theorem correct_mango_kg : mangoes_purchased 14 54 62 1376 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_purchased_correct_mango_kg_l3245_324576


namespace NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_x_squared_l3245_324523

theorem max_value_x_sqrt_1_minus_x_squared :
  (∀ x : ℝ, x * Real.sqrt (1 - x^2) ≤ 1/2) ∧
  (∃ x : ℝ, x * Real.sqrt (1 - x^2) = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_x_squared_l3245_324523


namespace NUMINAMATH_CALUDE_mbmt_equation_solution_l3245_324542

theorem mbmt_equation_solution :
  ∃ (T H E M B : ℕ),
    T ≠ H ∧ T ≠ E ∧ T ≠ M ∧ T ≠ B ∧
    H ≠ E ∧ H ≠ M ∧ H ≠ B ∧
    E ≠ M ∧ E ≠ B ∧
    M ≠ B ∧
    T < 10 ∧ H < 10 ∧ E < 10 ∧ M < 10 ∧ B < 10 ∧
    B = 4 ∧ E = 2 ∧ T = 6 ∧
    (100 * T + 10 * H + E) + (1000 * M + 100 * B + 10 * M + T) = 2018 := by
  sorry

end NUMINAMATH_CALUDE_mbmt_equation_solution_l3245_324542


namespace NUMINAMATH_CALUDE_cannot_empty_both_piles_l3245_324515

/-- Represents the state of the two piles of coins -/
structure CoinPiles :=
  (pile1 : ℕ)
  (pile2 : ℕ)

/-- Represents the allowed operations on the piles -/
inductive Operation
  | transferAndAdd : Operation
  | removeFour : Operation

/-- Applies an operation to the current state of the piles -/
def applyOperation (state : CoinPiles) (op : Operation) : CoinPiles :=
  match op with
  | Operation.transferAndAdd => 
      if state.pile1 > 0 then 
        CoinPiles.mk (state.pile1 - 1) (state.pile2 + 3)
      else 
        CoinPiles.mk (state.pile1 + 3) (state.pile2 - 1)
  | Operation.removeFour => 
      if state.pile1 ≥ 4 then 
        CoinPiles.mk (state.pile1 - 4) state.pile2
      else 
        CoinPiles.mk state.pile1 (state.pile2 - 4)

/-- The initial state of the piles -/
def initialState : CoinPiles := CoinPiles.mk 1 0

/-- Theorem stating that it's impossible to empty both piles -/
theorem cannot_empty_both_piles :
  ¬∃ (ops : List Operation), 
    let finalState := ops.foldl applyOperation initialState
    finalState.pile1 = 0 ∧ finalState.pile2 = 0 :=
sorry

end NUMINAMATH_CALUDE_cannot_empty_both_piles_l3245_324515


namespace NUMINAMATH_CALUDE_correct_multiplication_l3245_324591

theorem correct_multiplication (x : ℝ) : x * 51 = 244.8 → x * 15 = 72 := by
  sorry

end NUMINAMATH_CALUDE_correct_multiplication_l3245_324591


namespace NUMINAMATH_CALUDE_min_triangles_to_cover_l3245_324519

theorem min_triangles_to_cover (large_side : ℝ) (small_side : ℝ) : 
  large_side = 8 → small_side = 2 → 
  (large_side^2 / small_side^2 : ℝ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_triangles_to_cover_l3245_324519


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3245_324509

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) (n : ℕ) :
  (∀ k, a (k + 1) = a k * q) →  -- geometric sequence condition
  q > 0 →  -- positive common ratio
  a 1 * a 2 * a 3 = 4 →  -- first condition
  a 4 * a 5 * a 6 = 8 →  -- second condition
  a n * a (n + 1) * a (n + 2) = 128 →  -- third condition
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3245_324509


namespace NUMINAMATH_CALUDE_find_A_l3245_324528

theorem find_A (A B : ℕ) : 
  A ≤ 9 →
  B ≤ 9 →
  100 ≤ A * 100 + 78 →
  A * 100 + 78 < 1000 →
  100 ≤ 200 + B →
  200 + B < 1000 →
  A * 100 + 78 - (200 + B) = 364 →
  A = 5 := by
sorry

end NUMINAMATH_CALUDE_find_A_l3245_324528


namespace NUMINAMATH_CALUDE_distinct_angles_in_twelve_sided_polygon_l3245_324507

/-- A circle with an inscribed regular pentagon and heptagon -/
structure InscribedPolygons where
  circle : Set ℝ × ℝ  -- Representing a circle in 2D plane
  pentagon : Set (ℝ × ℝ)  -- Vertices of the pentagon
  heptagon : Set (ℝ × ℝ)  -- Vertices of the heptagon

/-- The resulting 12-sided polygon -/
def twelveSidedPolygon (ip : InscribedPolygons) : Set (ℝ × ℝ) :=
  ip.pentagon ∪ ip.heptagon

/-- Predicate to check if two polygons have no common vertices -/
def noCommonVertices (p1 p2 : Set (ℝ × ℝ)) : Prop :=
  p1 ∩ p2 = ∅

/-- Predicate to check if two polygons have no common axes of symmetry -/
def noCommonAxesOfSymmetry (p1 p2 : Set (ℝ × ℝ)) : Prop :=
  sorry  -- Definition omitted for brevity

/-- Function to count distinct angle values in a polygon -/
def countDistinctAngles (p : Set (ℝ × ℝ)) : ℕ :=
  sorry  -- Definition omitted for brevity

/-- The main theorem -/
theorem distinct_angles_in_twelve_sided_polygon
  (ip : InscribedPolygons)
  (h1 : noCommonVertices ip.pentagon ip.heptagon)
  (h2 : noCommonAxesOfSymmetry ip.pentagon ip.heptagon)
  : countDistinctAngles (twelveSidedPolygon ip) = 6 :=
sorry

end NUMINAMATH_CALUDE_distinct_angles_in_twelve_sided_polygon_l3245_324507


namespace NUMINAMATH_CALUDE_five_T_three_equals_38_l3245_324534

-- Define the new operation ⊤
def T (a b : ℤ) : ℤ := 4*a + 6*b

-- Theorem statement
theorem five_T_three_equals_38 : T 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_five_T_three_equals_38_l3245_324534


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3245_324506

theorem quadratic_equation_roots (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 3*x - m*x + m - 1
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 3*x₁ - x₁*x₂ + 3*x₂ = 12 →
  x₁ = 0 ∧ x₂ = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3245_324506


namespace NUMINAMATH_CALUDE_sum_first_seven_primes_l3245_324501

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

theorem sum_first_seven_primes : 
  (first_seven_primes.sum = 58) ∧ (∀ p ∈ first_seven_primes, Nat.Prime p) ∧ (first_seven_primes.length = 7) := by
  sorry

end NUMINAMATH_CALUDE_sum_first_seven_primes_l3245_324501


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3245_324541

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 5 + a 10 = 12 → 3 * a 7 + a 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3245_324541


namespace NUMINAMATH_CALUDE_root_sum_equals_square_sum_l3245_324562

theorem root_sum_equals_square_sum (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*a*(x₁-1) - 1 = 0 ∧ 
                x₂^2 - 2*a*(x₂-1) - 1 = 0 ∧ 
                x₁ + x₂ = x₁^2 + x₂^2) ↔ 
  (a = 1 ∨ a = 1/2) := by
sorry

end NUMINAMATH_CALUDE_root_sum_equals_square_sum_l3245_324562


namespace NUMINAMATH_CALUDE_probability_second_class_first_given_first_class_second_is_two_fifths_l3245_324545

/-- Represents the class of an item -/
inductive ItemClass
| FirstClass
| SecondClass

/-- Represents the box with items -/
structure Box where
  firstClassCount : ℕ
  secondClassCount : ℕ

/-- Represents the outcome of drawing two items -/
structure DrawOutcome where
  first : ItemClass
  second : ItemClass

def Box.totalCount (b : Box) : ℕ := b.firstClassCount + b.secondClassCount

/-- The probability of drawing a second-class item first, given that the second item is first-class -/
def probabilitySecondClassFirstGivenFirstClassSecond (b : Box) : ℚ :=
  let totalOutcomes := b.firstClassCount * (b.firstClassCount - 1) + b.secondClassCount * b.firstClassCount
  let favorableOutcomes := b.secondClassCount * b.firstClassCount
  favorableOutcomes / totalOutcomes

theorem probability_second_class_first_given_first_class_second_is_two_fifths :
  let b : Box := { firstClassCount := 4, secondClassCount := 2 }
  probabilitySecondClassFirstGivenFirstClassSecond b = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_class_first_given_first_class_second_is_two_fifths_l3245_324545


namespace NUMINAMATH_CALUDE_subsets_exist_l3245_324522

/-- A type representing a set of subsets of positive integers -/
def SubsetCollection := Finset (Set ℕ+)

/-- A function that constructs the required subsets -/
def constructSubsets (n : ℕ) : SubsetCollection :=
  sorry

/-- Predicate to check if subsets are pairwise nonintersecting -/
def pairwiseNonintersecting (s : SubsetCollection) : Prop :=
  sorry

/-- Predicate to check if all subsets are nonempty -/
def allNonempty (s : SubsetCollection) : Prop :=
  sorry

/-- Predicate to check if each positive integer can be uniquely expressed
    as a sum of at most n integers from different subsets -/
def uniqueRepresentation (s : SubsetCollection) (n : ℕ) : Prop :=
  sorry

/-- The main theorem stating the existence of the required subsets -/
theorem subsets_exist (n : ℕ) (h : n ≥ 2) :
  ∃ s : SubsetCollection,
    s.card = n ∧
    pairwiseNonintersecting s ∧
    allNonempty s ∧
    uniqueRepresentation s n :=
  sorry

end NUMINAMATH_CALUDE_subsets_exist_l3245_324522


namespace NUMINAMATH_CALUDE_light_source_height_l3245_324559

/-- The length of the cube's edge in centimeters -/
def cube_edge : ℝ := 2

/-- The area of the shadow cast by the cube, excluding the area beneath the cube, in square centimeters -/
def shadow_area : ℝ := 98

/-- The height of the light source above a top vertex of the cube in centimeters -/
def y : ℝ := sorry

/-- The theorem stating that the greatest integer not exceeding 1000y is 500 -/
theorem light_source_height : ⌊1000 * y⌋ = 500 := by sorry

end NUMINAMATH_CALUDE_light_source_height_l3245_324559


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3245_324594

theorem trigonometric_identity (α : Real) : 
  4.10 * (Real.cos (π/4 - α))^2 - (Real.cos (π/3 + α))^2 - 
  Real.cos (5*π/12) * Real.sin (5*π/12 - 2*α) = Real.sin (2*α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3245_324594


namespace NUMINAMATH_CALUDE_function_equality_implies_a_equals_two_l3245_324587

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then a^x else 1 - x

theorem function_equality_implies_a_equals_two (a : ℝ) :
  f a 1 = f a (-1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_a_equals_two_l3245_324587


namespace NUMINAMATH_CALUDE_unique_solution_implies_m_equals_3_l3245_324572

/-- For a quadratic equation ax^2 + bx + c = 0 to have exactly one solution,
    its discriminant (b^2 - 4ac) must be zero. -/
def has_exactly_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- The quadratic equation 3x^2 - 6x + m = 0 has exactly one solution
    if and only if m = 3. -/
theorem unique_solution_implies_m_equals_3 :
  ∀ m : ℝ, has_exactly_one_solution 3 (-6) m ↔ m = 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_m_equals_3_l3245_324572


namespace NUMINAMATH_CALUDE_solution_set_x_squared_minus_one_l3245_324512

theorem solution_set_x_squared_minus_one (x : ℝ) : x^2 - 1 ≥ 0 ↔ x ≥ 1 ∨ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_minus_one_l3245_324512


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l3245_324540

theorem smallest_square_containing_circle (r : ℝ) (h : r = 6) : 
  (2 * r) ^ 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l3245_324540


namespace NUMINAMATH_CALUDE_sum_of_recorded_products_25_coins_l3245_324557

/-- Represents the process of dividing coins into groups and recording products. -/
def divide_coins (n : ℕ) : ℕ := 
  (n * (n - 1)) / 2

/-- The theorem stating that the sum of recorded products when dividing 25 coins is 300. -/
theorem sum_of_recorded_products_25_coins : 
  divide_coins 25 = 300 := by sorry

end NUMINAMATH_CALUDE_sum_of_recorded_products_25_coins_l3245_324557


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l3245_324581

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (5678 * 10 + N) % 6 = 0 → N ≤ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l3245_324581


namespace NUMINAMATH_CALUDE_quadrilateral_reconstruction_l3245_324536

/-- A quadrilateral in 2D space -/
structure Quadrilateral (V : Type*) [AddCommGroup V] :=
  (P Q R S : V)

/-- Extended points of a quadrilateral -/
structure ExtendedQuadrilateral (V : Type*) [AddCommGroup V] extends Quadrilateral V :=
  (P' Q' R' S' : V)

/-- Condition that P, Q, R, S are midpoints of PP', QQ', RR', SS' respectively -/
def is_midpoint_quadrilateral {V : Type*} [AddCommGroup V] [Module ℚ V] 
  (quad : Quadrilateral V) (ext_quad : ExtendedQuadrilateral V) : Prop :=
  quad.P = (1/2 : ℚ) • (quad.P + ext_quad.P') ∧
  quad.Q = (1/2 : ℚ) • (quad.Q + ext_quad.Q') ∧
  quad.R = (1/2 : ℚ) • (quad.R + ext_quad.R') ∧
  quad.S = (1/2 : ℚ) • (quad.S + ext_quad.S')

/-- Main theorem -/
theorem quadrilateral_reconstruction {V : Type*} [AddCommGroup V] [Module ℚ V] 
  (quad : Quadrilateral V) (ext_quad : ExtendedQuadrilateral V) 
  (h : is_midpoint_quadrilateral quad ext_quad) :
  quad.P = (1/15 : ℚ) • ext_quad.P' + (2/15 : ℚ) • ext_quad.Q' + 
           (4/15 : ℚ) • ext_quad.R' + (8/15 : ℚ) • ext_quad.S' := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_reconstruction_l3245_324536


namespace NUMINAMATH_CALUDE_definite_integral_x_plus_one_squared_ln_squared_l3245_324570

theorem definite_integral_x_plus_one_squared_ln_squared :
  ∫ x in (0:ℝ)..2, (x + 1)^2 * (Real.log (x + 1))^2 = 9 * (Real.log 3)^2 - 6 * Real.log 3 + 79 / 27 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_x_plus_one_squared_ln_squared_l3245_324570


namespace NUMINAMATH_CALUDE_maple_trees_after_planting_l3245_324516

theorem maple_trees_after_planting (initial_trees : ℕ) (new_trees : ℕ) : 
  initial_trees = 2 → new_trees = 9 → initial_trees + new_trees = 11 := by
  sorry

end NUMINAMATH_CALUDE_maple_trees_after_planting_l3245_324516


namespace NUMINAMATH_CALUDE_ram_independent_time_l3245_324592

/-- The number of days Gohul takes to complete the job independently -/
def gohul_days : ℝ := 15

/-- The number of days Ram and Gohul take to complete the job together -/
def combined_days : ℝ := 6

/-- The number of days Ram takes to complete the job independently -/
def ram_days : ℝ := 10

/-- Theorem stating that given Gohul's time and the combined time, Ram's independent time is 10 days -/
theorem ram_independent_time : 
  (1 / ram_days + 1 / gohul_days = 1 / combined_days) → ram_days = 10 := by
  sorry

end NUMINAMATH_CALUDE_ram_independent_time_l3245_324592


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3245_324582

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 5 + a 7 + a 9 + a 11 = 100) :
  3 * a 9 - a 13 = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3245_324582


namespace NUMINAMATH_CALUDE_toy_cost_proof_l3245_324518

-- Define the number of toys
def num_toys : ℕ := 5

-- Define the discount rate (80% of original price)
def discount_rate : ℚ := 4/5

-- Define the total paid after discount
def total_paid : ℚ := 12

-- Define the cost per toy before discount
def cost_per_toy : ℚ := 3

-- Theorem statement
theorem toy_cost_proof :
  discount_rate * (num_toys : ℚ) * cost_per_toy = total_paid :=
sorry

end NUMINAMATH_CALUDE_toy_cost_proof_l3245_324518


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3245_324513

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : is_geometric_sequence a)
  (h_mean : Real.sqrt (a 4 * a 14) = 2 * Real.sqrt 2) :
  (2 * a 7 + a 11) ≥ 8 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 8 ∧ x * y = 8 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3245_324513


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3245_324527

theorem simplify_fraction_product : 8 * (18 / 5) * (-40 / 27) = -128 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3245_324527


namespace NUMINAMATH_CALUDE_expression_evaluation_l3245_324578

theorem expression_evaluation (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3245_324578


namespace NUMINAMATH_CALUDE_basketball_scores_l3245_324551

/-- Represents the scores of a team over four quarters -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℚ)

/-- Checks if a sequence of four numbers is geometric -/
def isGeometric (s : TeamScores) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if a sequence of four numbers is arithmetic -/
def isArithmetic (s : TeamScores) : Prop :=
  ∃ d : ℚ, s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- Calculates the total score for a team -/
def totalScore (s : TeamScores) : ℚ :=
  s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the first half score for a team -/
def firstHalfScore (s : TeamScores) : ℚ :=
  s.q1 + s.q2

theorem basketball_scores (teamA teamB : TeamScores) :
  teamA.q1 = teamB.q1 →  -- Tied at the end of first quarter
  isGeometric teamA →    -- Team A's scores form a geometric sequence
  isArithmetic teamB →   -- Team B's scores form an arithmetic sequence
  totalScore teamA = totalScore teamB + 2 →  -- Team A won by two points
  totalScore teamA ≤ 80 →  -- Team A's total score is not more than 80
  totalScore teamB ≤ 80 →  -- Team B's total score is not more than 80
  firstHalfScore teamA + firstHalfScore teamB = 41 :=
by sorry

end NUMINAMATH_CALUDE_basketball_scores_l3245_324551


namespace NUMINAMATH_CALUDE_ninety_degrees_to_radians_l3245_324535

theorem ninety_degrees_to_radians :
  (90 : ℝ) * π / 180 = π / 2 := by sorry

end NUMINAMATH_CALUDE_ninety_degrees_to_radians_l3245_324535


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l3245_324580

/-- In a right-angled triangle XYZ, given the following conditions:
  - ∠X = 90°
  - YZ = 20
  - tan Z = 3 sin Y
  Prove that XY = (40√2) / 3 -/
theorem right_triangle_side_length (X Y Z : ℝ) (h1 : X^2 + Y^2 = Z^2) 
  (h2 : Z = 20) (h3 : Real.tan X = 3 * Real.sin Y) : 
  Y = (40 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l3245_324580


namespace NUMINAMATH_CALUDE_no_real_j_for_single_solution_l3245_324525

theorem no_real_j_for_single_solution :
  ¬ ∃ j : ℝ, ∃! x : ℝ, (2 * x + 7) * (x - 5) + 3 * x^2 = -20 + (j + 3) * x + 3 * x^2 :=
by sorry

end NUMINAMATH_CALUDE_no_real_j_for_single_solution_l3245_324525


namespace NUMINAMATH_CALUDE_alicia_tax_payment_l3245_324566

/-- Calculates the total tax paid in cents per hour given an hourly wage and tax rates -/
def total_tax_cents (hourly_wage : ℝ) (local_tax_rate : ℝ) (state_tax_rate : ℝ) : ℝ :=
  hourly_wage * 100 * (local_tax_rate + state_tax_rate)

/-- Proves that Alicia's total tax paid is 62.5 cents per hour -/
theorem alicia_tax_payment :
  total_tax_cents 25 0.02 0.005 = 62.5 := by
  sorry

#eval total_tax_cents 25 0.02 0.005

end NUMINAMATH_CALUDE_alicia_tax_payment_l3245_324566


namespace NUMINAMATH_CALUDE_bargain_bin_books_l3245_324599

/-- The number of books initially in the bargain bin -/
def initial_books : ℝ := 41.0

/-- The number of books added in the first addition -/
def first_addition : ℝ := 33.0

/-- The number of books added in the second addition -/
def second_addition : ℝ := 2.0

/-- The total number of books after both additions -/
def total_books : ℝ := 76.0

/-- Theorem stating that the initial number of books plus the two additions equals the total -/
theorem bargain_bin_books : 
  initial_books + first_addition + second_addition = total_books := by
  sorry

end NUMINAMATH_CALUDE_bargain_bin_books_l3245_324599


namespace NUMINAMATH_CALUDE_complex_point_on_line_l3245_324504

theorem complex_point_on_line (a : ℝ) : 
  (∃ (z : ℂ), z = (a - 1 : ℝ) + 3*I ∧ z.im = z.re + 2) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_point_on_line_l3245_324504


namespace NUMINAMATH_CALUDE_expression_equals_one_l3245_324548

theorem expression_equals_one : (-1)^2 - |(-3)| + (-5) / (-5/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3245_324548


namespace NUMINAMATH_CALUDE_fifteen_initial_points_theorem_l3245_324500

/-- Represents the number of points after k densifications -/
def points_after_densification (initial_points : ℕ) (densifications : ℕ) : ℕ :=
  initial_points * 2^densifications - (2^densifications - 1)

/-- Theorem stating that 15 initial points results in 113 points after 3 densifications -/
theorem fifteen_initial_points_theorem :
  ∃ (n : ℕ), n > 0 ∧ points_after_densification n 3 = 113 → n = 15 :=
sorry

end NUMINAMATH_CALUDE_fifteen_initial_points_theorem_l3245_324500


namespace NUMINAMATH_CALUDE_equation_solution_l3245_324571

theorem equation_solution : ∃ x : ℝ, 24 * 2 - 6 = 3 * x + 6 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3245_324571


namespace NUMINAMATH_CALUDE_sqrt_37_range_l3245_324533

theorem sqrt_37_range : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_37_range_l3245_324533


namespace NUMINAMATH_CALUDE_factorial_ratio_l3245_324508

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_ratio : 
  factorial 10 / (factorial 5 * factorial 2) = 15120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3245_324508


namespace NUMINAMATH_CALUDE_sample_capacity_l3245_324595

theorem sample_capacity (frequency : ℕ) (frequency_rate : ℚ) (n : ℕ) 
  (h1 : frequency = 30)
  (h2 : frequency_rate = 1/4)
  (h3 : n = frequency / frequency_rate) :
  n = 120 := by
sorry

end NUMINAMATH_CALUDE_sample_capacity_l3245_324595


namespace NUMINAMATH_CALUDE_exactly_one_line_with_two_rational_points_l3245_324549

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the Cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A rational point is a point with rational coordinates -/
def RationalPoint (p : Point) : Prop :=
  ∃ (qx qy : ℚ), p.x = qx ∧ p.y = qy

/-- A line passes through a point if the point satisfies the line equation -/
def LinePassesThrough (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- A line contains at least two rational points -/
def LineContainsTwoRationalPoints (l : Line) : Prop :=
  ∃ (p1 p2 : Point), p1 ≠ p2 ∧ RationalPoint p1 ∧ RationalPoint p2 ∧
    LinePassesThrough l p1 ∧ LinePassesThrough l p2

/-- The main theorem -/
theorem exactly_one_line_with_two_rational_points
  (a : ℝ) (h_irrational : ¬ ∃ (q : ℚ), a = q) :
  ∃! (l : Line), LinePassesThrough l (Point.mk a 0) ∧ LineContainsTwoRationalPoints l :=
sorry

end NUMINAMATH_CALUDE_exactly_one_line_with_two_rational_points_l3245_324549


namespace NUMINAMATH_CALUDE_number_of_students_l3245_324579

theorem number_of_students (group_size : ℕ) (num_groups : ℕ) (h1 : group_size = 5) (h2 : num_groups = 6) :
  group_size * num_groups = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l3245_324579


namespace NUMINAMATH_CALUDE_identity_unique_solution_l3245_324511

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- The identity function is the unique solution to the functional equation -/
theorem identity_unique_solution :
  ∃! f : ℝ → ℝ, SatisfiesEquation f ∧ ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_CALUDE_identity_unique_solution_l3245_324511


namespace NUMINAMATH_CALUDE_card_selection_ways_l3245_324552

theorem card_selection_ways (left_cards right_cards : ℕ) 
  (h1 : left_cards = 15) 
  (h2 : right_cards = 20) : 
  left_cards + right_cards = 35 := by
  sorry

end NUMINAMATH_CALUDE_card_selection_ways_l3245_324552


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l3245_324584

theorem smallest_non_factor_product (m n : ℕ) : 
  m ≠ n → 
  m > 0 → 
  n > 0 → 
  m ∣ 48 → 
  n ∣ 48 → 
  ¬(m * n ∣ 48) → 
  (∀ a b : ℕ, a ≠ b → a > 0 → b > 0 → a ∣ 48 → b ∣ 48 → ¬(a * b ∣ 48) → m * n ≤ a * b) →
  m * n = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l3245_324584


namespace NUMINAMATH_CALUDE_min_value_and_valid_a4_l3245_324537

def is_valid_sequence (a : Fin 10 → ℕ) : Prop :=
  ∀ i j : Fin 10, i < j → a i < a j

def lcm_of_sequence (a : Fin 10 → ℕ) : ℕ :=
  Finset.lcm (Finset.range 10) (fun i => a i)

theorem min_value_and_valid_a4 (a : Fin 10 → ℕ) (h : is_valid_sequence a) :
  (∀ b : Fin 10 → ℕ, is_valid_sequence b → lcm_of_sequence a / a 3 ≤ lcm_of_sequence b / b 3) ∧
  (lcm_of_sequence a / a 0 = lcm_of_sequence a / a 3) →
  (lcm_of_sequence a / a 3 = 630) ∧
  (a 3 = 360 ∨ a 3 = 720 ∨ a 3 = 1080) ∧
  (1 ≤ a 3) ∧ (a 3 ≤ 1300) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_valid_a4_l3245_324537


namespace NUMINAMATH_CALUDE_apple_profit_calculation_l3245_324568

/-- Profit percentage for the first half of apples -/
def P : ℝ := sorry

/-- Cost price of 1 kg of apples -/
def C : ℝ := sorry

theorem apple_profit_calculation :
  (50 * C + 50 * C * (P / 100) + 50 * C + 50 * C * (30 / 100) = 100 * C + 100 * C * (27.5 / 100)) →
  P = 25 := by
  sorry

end NUMINAMATH_CALUDE_apple_profit_calculation_l3245_324568


namespace NUMINAMATH_CALUDE_problem_statement_l3245_324565

theorem problem_statement (x y : ℝ) 
  (h1 : x * y + x + y = 17)
  (h2 : x^2 * y + x * y^2 = 66) :
  x^4 + x^3 * y + x^2 * y^2 + x * y^3 + y^4 = 12499 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3245_324565


namespace NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3245_324543

theorem square_ratio_side_length_sum (s1 s2 : ℝ) (h : s1^2 / s2^2 = 32 / 63) :
  ∃ (a b c : ℕ), (s1 / s2 = a * Real.sqrt b / c) ∧ (a + b + c = 39) := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3245_324543


namespace NUMINAMATH_CALUDE_seven_balance_removal_l3245_324564

/-- A function that counts the number of sevens in even positions of a natural number -/
def countSevenEven (n : ℕ) : ℕ := sorry

/-- A function that counts the number of sevens in odd positions of a natural number -/
def countSevenOdd (n : ℕ) : ℕ := sorry

/-- A function that removes the i-th digit from a natural number -/
def removeDigit (n : ℕ) (i : ℕ) : ℕ := sorry

/-- A function that returns the number of digits in a natural number -/
def digitCount (n : ℕ) : ℕ := sorry

theorem seven_balance_removal (n : ℕ) (h : Odd (digitCount n)) :
  ∃ i : ℕ, i < digitCount n ∧ 
    countSevenEven (removeDigit n i) = countSevenOdd (removeDigit n i) := by
  sorry

end NUMINAMATH_CALUDE_seven_balance_removal_l3245_324564
