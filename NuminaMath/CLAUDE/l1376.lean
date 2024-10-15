import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1376_137632

theorem sufficient_not_necessary (x y : ℝ) : 
  (∀ x y : ℝ, x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1376_137632


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l1376_137606

open Real

/-- The least positive angle θ (in degrees) satisfying cos 10° = sin 20° + sin θ is 40°. -/
theorem least_positive_angle_theorem : 
  (∀ θ : ℝ, 0 < θ ∧ θ < 40 → cos (10 * π / 180) ≠ sin (20 * π / 180) + sin (θ * π / 180)) ∧
  cos (10 * π / 180) = sin (20 * π / 180) + sin (40 * π / 180) :=
sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l1376_137606


namespace NUMINAMATH_CALUDE_oreos_and_cookies_problem_l1376_137633

theorem oreos_and_cookies_problem :
  ∀ (oreos cookies : ℕ) (oreo_price cookie_price : ℚ),
    oreos * 9 = cookies * 4 →
    oreo_price = 2 →
    cookie_price = 3 →
    cookies * cookie_price - oreos * oreo_price = 95 →
    oreos + cookies = 65 := by
  sorry

end NUMINAMATH_CALUDE_oreos_and_cookies_problem_l1376_137633


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1376_137638

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 2 then 0 else 2 / (2 - x)

theorem unique_function_satisfying_conditions :
  (∀ x : ℝ, x ≥ 0 → f x ≥ 0) ∧
  (f 2 = 0) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x ≠ 0) ∧
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → f (x * f y) * f y = f (x + y)) ∧
  (∀ g : ℝ → ℝ, 
    ((∀ x : ℝ, x ≥ 0 → g x ≥ 0) ∧
     (g 2 = 0) ∧
     (∀ x : ℝ, 0 ≤ x ∧ x < 2 → g x ≠ 0) ∧
     (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → g (x * g y) * g y = g (x + y))) →
    (∀ x : ℝ, x ≥ 0 → g x = f x)) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1376_137638


namespace NUMINAMATH_CALUDE_jennas_stickers_l1376_137621

/-- Given that the ratio of Kate's stickers to Jenna's stickers is 7:4 and Kate has 21 stickers,
    prove that Jenna has 12 stickers. -/
theorem jennas_stickers (kate_stickers : ℕ) (jenna_stickers : ℕ) : 
  (kate_stickers : ℚ) / jenna_stickers = 7 / 4 → kate_stickers = 21 → jenna_stickers = 12 := by
  sorry

end NUMINAMATH_CALUDE_jennas_stickers_l1376_137621


namespace NUMINAMATH_CALUDE_exam_girls_count_l1376_137634

/-- Proves that the number of girls is 1800 given the exam conditions -/
theorem exam_girls_count :
  ∀ (boys girls : ℕ),
  boys + girls = 2000 →
  (34 * boys + 32 * girls : ℚ) = 331 * 20 →
  girls = 1800 := by
sorry

end NUMINAMATH_CALUDE_exam_girls_count_l1376_137634


namespace NUMINAMATH_CALUDE_divisible_by_five_unit_digits_l1376_137677

theorem divisible_by_five_unit_digits :
  ∃ (S : Finset Nat), (∀ n : Nat, n % 5 = 0 ↔ n % 10 ∈ S) ∧ Finset.card S = 2 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_five_unit_digits_l1376_137677


namespace NUMINAMATH_CALUDE_remaining_nails_l1376_137656

def initial_nails : ℕ := 400

def kitchen_repair (n : ℕ) : ℕ := n - (n * 30 / 100)

def fence_repair (n : ℕ) : ℕ := n - (n * 70 / 100)

def table_repair (n : ℕ) : ℕ := n - (n * 50 / 100)

def floorboard_repair (n : ℕ) : ℕ := n - (n * 25 / 100)

theorem remaining_nails :
  floorboard_repair (table_repair (fence_repair (kitchen_repair initial_nails))) = 32 := by
  sorry

end NUMINAMATH_CALUDE_remaining_nails_l1376_137656


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_related_quadratic_inequality_solution_l1376_137670

/-- Solution set type -/
inductive SolutionSet
  | Empty
  | Interval (lower upper : ℝ)
  | Union (s1 s2 : SolutionSet)

/-- Solve quadratic inequality -/
noncomputable def solveQuadraticInequality (a : ℝ) : SolutionSet :=
  if a = 0 then
    SolutionSet.Empty
  else if a > 0 then
    SolutionSet.Interval (-a) (2 * a)
  else
    SolutionSet.Interval (2 * a) (-a)

/-- Theorem for part 1 -/
theorem quadratic_inequality_solution (a : ℝ) :
  solveQuadraticInequality a =
    if a = 0 then
      SolutionSet.Empty
    else if a > 0 then
      SolutionSet.Interval (-a) (2 * a)
    else
      SolutionSet.Interval (2 * a) (-a) :=
by sorry

/-- Theorem for part 2 -/
theorem related_quadratic_inequality_solution :
  ∃ (a b : ℝ), 
    (∀ x, x^2 - a*x - b < 0 ↔ -1 < x ∧ x < 2) →
    (∀ x, a*x^2 + x - b > 0 ↔ x < -2 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_related_quadratic_inequality_solution_l1376_137670


namespace NUMINAMATH_CALUDE_linear_system_solution_l1376_137658

theorem linear_system_solution (x y a : ℝ) : 
  x + 2*y = 2 → 
  2*x + y = a → 
  x + y = 5 → 
  a = 13 := by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1376_137658


namespace NUMINAMATH_CALUDE_distance_after_walk_on_hexagon_l1376_137688

/-- The distance from the starting point after walking along a regular hexagon's perimeter -/
theorem distance_after_walk_on_hexagon (side_length : ℝ) (walk_distance : ℝ) 
  (h1 : side_length = 3)
  (h2 : walk_distance = 10) :
  ∃ (end_point : ℝ × ℝ),
    (end_point.1^2 + end_point.2^2) = (3 * Real.sqrt 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_walk_on_hexagon_l1376_137688


namespace NUMINAMATH_CALUDE_nasadkas_in_barrel_l1376_137667

/-- The volume of a barrel -/
def barrel : ℝ := sorry

/-- The volume of a nasadka -/
def nasadka : ℝ := sorry

/-- The volume of a bucket -/
def bucket : ℝ := sorry

/-- The first condition: 1 barrel + 20 buckets = 3 barrels -/
axiom condition1 : barrel + 20 * bucket = 3 * barrel

/-- The second condition: 19 barrels + 1 nasadka + 15.5 buckets = 20 barrels + 8 buckets -/
axiom condition2 : 19 * barrel + nasadka + 15.5 * bucket = 20 * barrel + 8 * bucket

/-- The theorem stating that there are 4 nasadkas in a barrel -/
theorem nasadkas_in_barrel : barrel / nasadka = 4 := by sorry

end NUMINAMATH_CALUDE_nasadkas_in_barrel_l1376_137667


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l1376_137600

def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem geometric_sequence_properties
  (a b : ℕ → ℝ)
  (ha : is_geometric_sequence a)
  (hb : is_geometric_sequence b) :
  (¬ ∀ s : ℕ → ℝ, (∀ n, s n = a n + b n) → is_geometric_sequence s) ∧
  (∃ s : ℕ → ℝ, (∀ n, s n = a n * b n) ∧ is_geometric_sequence s) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l1376_137600


namespace NUMINAMATH_CALUDE_abs_sum_zero_implies_sum_l1376_137629

theorem abs_sum_zero_implies_sum (a b : ℝ) : 
  |a - 2| + |b + 3| = 0 → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_zero_implies_sum_l1376_137629


namespace NUMINAMATH_CALUDE_complex_power_215_36_l1376_137660

theorem complex_power_215_36 :
  (Complex.exp (215 * π / 180 * Complex.I)) ^ 36 = 1/2 - Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_215_36_l1376_137660


namespace NUMINAMATH_CALUDE_mary_sheep_problem_l1376_137659

theorem mary_sheep_problem (initial_sheep : ℕ) : 
  (initial_sheep : ℚ) * (3/4) * (1/2) = 150 → initial_sheep = 400 := by
  sorry

end NUMINAMATH_CALUDE_mary_sheep_problem_l1376_137659


namespace NUMINAMATH_CALUDE_incenter_coeff_sum_specific_triangle_incenter_l1376_137627

/-- Given a triangle XYZ with sides x, y, z, the position vector of its incenter J
    can be expressed as J⃗ = p X⃗ + q Y⃗ + r Z⃗, where p, q, r are constants. -/
def incenter_position_vector (x y z : ℝ) (p q r : ℝ) : Prop :=
  p = x / (x + y + z) ∧ q = y / (x + y + z) ∧ r = z / (x + y + z)

/-- The sum of coefficients p, q, r in the incenter position vector equation is 1. -/
theorem incenter_coeff_sum (x y z : ℝ) (p q r : ℝ) 
  (h : incenter_position_vector x y z p q r) : p + q + r = 1 := by sorry

/-- For a triangle with sides 8, 11, and 5, the position vector of its incenter
    is given by (1/3, 11/24, 5/24). -/
theorem specific_triangle_incenter : 
  incenter_position_vector 8 11 5 (1/3) (11/24) (5/24) := by sorry

end NUMINAMATH_CALUDE_incenter_coeff_sum_specific_triangle_incenter_l1376_137627


namespace NUMINAMATH_CALUDE_sin_product_equals_cos_over_eight_l1376_137692

theorem sin_product_equals_cos_over_eight :
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) * Real.sin (80 * π / 180) = 
  Real.cos (10 * π / 180) / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_cos_over_eight_l1376_137692


namespace NUMINAMATH_CALUDE_points_lost_l1376_137628

theorem points_lost (first_round : ℕ) (second_round : ℕ) (final_score : ℕ) :
  first_round = 40 →
  second_round = 50 →
  final_score = 86 →
  (first_round + second_round) - final_score = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_points_lost_l1376_137628


namespace NUMINAMATH_CALUDE_all_naturals_reachable_l1376_137675

def triple_plus_one (x : ℕ) : ℕ := 3 * x + 1

def floor_half (x : ℕ) : ℕ := x / 2

def reachable (n : ℕ) : Prop :=
  ∃ (seq : List (ℕ → ℕ)), seq.foldl (λ acc f => f acc) 1 = n ∧
    ∀ f ∈ seq, f = triple_plus_one ∨ f = floor_half

theorem all_naturals_reachable : ∀ n : ℕ, reachable n := by
  sorry

end NUMINAMATH_CALUDE_all_naturals_reachable_l1376_137675


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1376_137655

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1376_137655


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_eq_four_l1376_137676

/-- The function f(x) = ax³ - 3x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3*x + 1

/-- The set of values for 'a' that satisfy the condition -/
def A : Set ℝ := {a : ℝ | ∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ 0}

theorem f_nonnegative_iff_a_eq_four : A = {4} := by sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_eq_four_l1376_137676


namespace NUMINAMATH_CALUDE_average_temperature_twthf_l1376_137693

/-- The average temperature for Tuesday, Wednesday, Thursday, and Friday is 46 degrees -/
theorem average_temperature_twthf (temp_mon : ℝ) (temp_fri : ℝ) (avg_mtwth : ℝ) :
  temp_mon = 43 →
  temp_fri = 35 →
  avg_mtwth = 48 →
  let temp_twth : ℝ := (4 * avg_mtwth - temp_mon) / 3
  let avg_twthf : ℝ := (3 * temp_twth + temp_fri) / 4
  ∀ ε > 0, |avg_twthf - 46| < ε :=
by sorry

end NUMINAMATH_CALUDE_average_temperature_twthf_l1376_137693


namespace NUMINAMATH_CALUDE_max_green_beads_l1376_137669

/-- A necklace with red, blue, and green beads. -/
structure Necklace :=
  (total : ℕ)
  (red : Finset ℕ)
  (blue : Finset ℕ)
  (green : Finset ℕ)

/-- The necklace satisfies the problem conditions. -/
def ValidNecklace (n : Necklace) : Prop :=
  n.total = 100 ∧
  n.red ∪ n.blue ∪ n.green = Finset.range n.total ∧
  (∀ i : ℕ, ∃ j ∈ n.blue, j % n.total ∈ Finset.range 5 ∪ {n.total - 1, n.total - 2, n.total - 3, n.total - 4}) ∧
  (∀ i : ℕ, ∃ j ∈ n.red, j % n.total ∈ Finset.range 7 ∪ {n.total - 1, n.total - 2, n.total - 3, n.total - 4, n.total - 5, n.total - 6})

/-- The maximum number of green beads in a valid necklace. -/
theorem max_green_beads (n : Necklace) (h : ValidNecklace n) :
  n.green.card ≤ 65 :=
sorry

end NUMINAMATH_CALUDE_max_green_beads_l1376_137669


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1376_137671

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 3 - 5*x^2 - x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l1376_137671


namespace NUMINAMATH_CALUDE_workers_wage_increase_l1376_137672

/-- Proves that if a worker's daily wage is increased by 50% to $42, then the original daily wage was $28. -/
theorem workers_wage_increase (original_wage : ℝ) (increased_wage : ℝ) : 
  increased_wage = 42 ∧ increased_wage = original_wage * 1.5 → original_wage = 28 := by
  sorry

end NUMINAMATH_CALUDE_workers_wage_increase_l1376_137672


namespace NUMINAMATH_CALUDE_work_completion_l1376_137626

/-- The number of days A takes to complete the work alone -/
def days_A : ℝ := 4

/-- The number of days B takes to complete the work alone -/
def days_B : ℝ := 12

/-- The number of days B takes to finish the remaining work after A leaves -/
def days_B_remaining : ℝ := 4.000000000000001

/-- The number of days A and B work together -/
def days_together : ℝ := 2

theorem work_completion :
  let rate_A := 1 / days_A
  let rate_B := 1 / days_B
  let rate_together := rate_A + rate_B
  rate_together * days_together + rate_B * days_B_remaining = 1 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_l1376_137626


namespace NUMINAMATH_CALUDE_seunghye_number_l1376_137650

theorem seunghye_number (x : ℝ) : 10 * x - x = 37.35 → x = 4.15 := by
  sorry

end NUMINAMATH_CALUDE_seunghye_number_l1376_137650


namespace NUMINAMATH_CALUDE_a_months_is_32_l1376_137614

/-- Represents the pasture rental problem -/
structure PastureRental where
  total_cost : ℕ
  a_horses : ℕ
  b_horses : ℕ
  c_horses : ℕ
  b_months : ℕ
  c_months : ℕ
  b_payment : ℕ

/-- Calculates the number of months a put in the horses -/
def calculate_a_months (p : PastureRental) : ℕ :=
  ((p.total_cost - p.b_payment - p.c_horses * p.c_months) / p.a_horses)

/-- Theorem stating that a put in the horses for 32 months -/
theorem a_months_is_32 (p : PastureRental) 
  (h1 : p.total_cost = 841)
  (h2 : p.a_horses = 12)
  (h3 : p.b_horses = 16)
  (h4 : p.c_horses = 18)
  (h5 : p.b_months = 9)
  (h6 : p.c_months = 6)
  (h7 : p.b_payment = 348) :
  calculate_a_months p = 32 := by
  sorry

#eval calculate_a_months { 
  total_cost := 841, 
  a_horses := 12, 
  b_horses := 16, 
  c_horses := 18, 
  b_months := 9, 
  c_months := 6, 
  b_payment := 348 
}

end NUMINAMATH_CALUDE_a_months_is_32_l1376_137614


namespace NUMINAMATH_CALUDE_farm_cows_count_l1376_137649

/-- Represents the number of bags of husk eaten by a group of cows in 30 days -/
def total_bags : ℕ := 30

/-- Represents the number of bags of husk eaten by one cow in 30 days -/
def bags_per_cow : ℕ := 1

/-- Calculates the number of cows on the farm -/
def num_cows : ℕ := total_bags / bags_per_cow

/-- Proves that the number of cows on the farm is 30 -/
theorem farm_cows_count : num_cows = 30 := by
  sorry

end NUMINAMATH_CALUDE_farm_cows_count_l1376_137649


namespace NUMINAMATH_CALUDE_jackson_earnings_l1376_137685

def hourly_rate : ℝ := 5
def vacuuming_time : ℝ := 2
def vacuuming_repetitions : ℕ := 2
def dish_washing_time : ℝ := 0.5
def bathroom_cleaning_multiplier : ℕ := 3

def total_earnings : ℝ :=
  hourly_rate * (vacuuming_time * vacuuming_repetitions +
                 dish_washing_time +
                 bathroom_cleaning_multiplier * dish_washing_time)

theorem jackson_earnings :
  total_earnings = 30 := by
  sorry

end NUMINAMATH_CALUDE_jackson_earnings_l1376_137685


namespace NUMINAMATH_CALUDE_number_problem_l1376_137646

theorem number_problem (x : ℝ) : 5 * (x - 12) = 40 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1376_137646


namespace NUMINAMATH_CALUDE_paco_cookies_l1376_137636

theorem paco_cookies (cookies_eaten : ℕ) (cookies_given : ℕ) : 
  cookies_eaten = 14 →
  cookies_given = 13 →
  cookies_eaten = cookies_given + 1 →
  cookies_eaten + cookies_given = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l1376_137636


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1376_137647

theorem least_subtraction_for_divisibility : ∃ (n : ℕ), n = 8 ∧ 
  (∀ (m : ℕ), m < n → ¬(20 ∣ (50248 - m))) ∧ (20 ∣ (50248 - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1376_137647


namespace NUMINAMATH_CALUDE_first_three_terms_b_is_geometric_T_sum_l1376_137661

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom S_def (n : ℕ) : S n = 2 * sequence_a n - 2 * n

theorem first_three_terms :
  sequence_a 1 = 2 ∧ sequence_a 2 = 6 ∧ sequence_a 3 = 14 := by sorry

def sequence_b (n : ℕ) : ℝ := sequence_a n + 2

theorem b_is_geometric :
  ∃ (r : ℝ), ∀ (n : ℕ), n ≥ 2 → sequence_b n = r * sequence_b (n-1) := by sorry

def T (n : ℕ) : ℝ := sorry

theorem T_sum (n : ℕ) :
  T n = (n + 1) * 2^(n + 2) + 4 - n * (n + 1) := by sorry

end NUMINAMATH_CALUDE_first_three_terms_b_is_geometric_T_sum_l1376_137661


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1376_137691

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_condition (a : ℝ) :
  let z : ℂ := Complex.mk (a - 1) 1
  is_pure_imaginary z → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1376_137691


namespace NUMINAMATH_CALUDE_claire_pets_l1376_137690

theorem claire_pets (total_pets : ℕ) (male_pets : ℕ) 
  (h_total : total_pets = 90)
  (h_male : male_pets = 25) :
  ∃ (gerbils hamsters : ℕ),
    gerbils + hamsters = total_pets ∧
    (1 : ℚ) / 4 * gerbils + (1 : ℚ) / 3 * hamsters = male_pets ∧
    gerbils = 60 := by
  sorry

end NUMINAMATH_CALUDE_claire_pets_l1376_137690


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1376_137651

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (hyp : a^2 + b^2 = c^2) -- Pythagorean theorem
  (hypotenuse : c = 13) 
  (side : a = 12) : 
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1376_137651


namespace NUMINAMATH_CALUDE_equal_distance_travel_l1376_137642

theorem equal_distance_travel (v1 v2 v3 : ℝ) (t : ℝ) (h1 : v1 = 3) (h2 : v2 = 6) (h3 : v3 = 9) (ht : t = 11/60) :
  let d := t / (1/v1 + 1/v2 + 1/v3)
  3 * d = 0.9 := by sorry

end NUMINAMATH_CALUDE_equal_distance_travel_l1376_137642


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1376_137684

/-- 
Given an arithmetic sequence with:
- First term a = -36
- Common difference d = 6
- Last term l = 66

Prove that the number of terms in the sequence is 18.
-/
theorem arithmetic_sequence_length :
  ∀ (a d l : ℤ) (n : ℕ),
    a = -36 →
    d = 6 →
    l = 66 →
    l = a + (n - 1) * d →
    n = 18 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1376_137684


namespace NUMINAMATH_CALUDE_total_flowers_l1376_137610

theorem total_flowers (total_vases : Nat) (vases_with_five : Nat) (flowers_in_four : Nat) (flowers_in_one : Nat) : 
  total_vases = 5 → vases_with_five = 4 → flowers_in_four = 5 → flowers_in_one = 6 → 
  vases_with_five * flowers_in_four + (total_vases - vases_with_five) * flowers_in_one = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l1376_137610


namespace NUMINAMATH_CALUDE_subset_iff_m_le_three_l1376_137630

-- Define the sets A and B
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2*m - 1 }

-- State the theorem
theorem subset_iff_m_le_three (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_iff_m_le_three_l1376_137630


namespace NUMINAMATH_CALUDE_smallest_n_cookies_l1376_137635

theorem smallest_n_cookies : ∃ (n : ℕ), n > 0 ∧ 16 ∣ (25 * n - 3) ∧ ∀ (m : ℕ), m > 0 ∧ 16 ∣ (25 * m - 3) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_cookies_l1376_137635


namespace NUMINAMATH_CALUDE_tan_value_proof_l1376_137631

theorem tan_value_proof (α : Real) 
  (h1 : Real.sin α - Real.cos α = 1/5)
  (h2 : 0 < α ∧ α < Real.pi) : 
  Real.tan α = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_proof_l1376_137631


namespace NUMINAMATH_CALUDE_unique_solution_l1376_137657

/-- The equation has two solutions and their sum is 12 -/
def has_two_solutions_sum_12 (a b : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x ≠ 1 ∧ x ≠ -1 ∧ y ≠ 1 ∧ y ≠ -1 ∧
    (a * x^2 - 24 * x + b) / (x^2 - 1) = x ∧
    (a * y^2 - 24 * y + b) / (y^2 - 1) = y ∧
    x + y = 12

theorem unique_solution :
  ∀ a b : ℝ, has_two_solutions_sum_12 a b ↔ a = 35 ∧ b = -5819 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1376_137657


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l1376_137699

def f (x : ℤ) : ℤ := x^3 - 4*x^2 - 7*x + 10

theorem integer_roots_of_cubic :
  {x : ℤ | f x = 0} = {1, -2, 5} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l1376_137699


namespace NUMINAMATH_CALUDE_sixty_first_sample_number_l1376_137663

/-- Systematic sampling function -/
def systematicSample (totalItems : ℕ) (numGroups : ℕ) (firstItem : ℕ) (groupIndex : ℕ) : ℕ :=
  firstItem + (groupIndex - 1) * (totalItems / numGroups)

/-- Theorem stating the result of the 61st sample in the given conditions -/
theorem sixty_first_sample_number
  (totalItems : ℕ) (numGroups : ℕ) (firstItem : ℕ) (groupIndex : ℕ)
  (h1 : totalItems = 3000)
  (h2 : numGroups = 150)
  (h3 : firstItem = 11)
  (h4 : groupIndex = 61) :
  systematicSample totalItems numGroups firstItem groupIndex = 1211 := by
  sorry

#eval systematicSample 3000 150 11 61

end NUMINAMATH_CALUDE_sixty_first_sample_number_l1376_137663


namespace NUMINAMATH_CALUDE_prime_square_mod_twelve_l1376_137644

theorem prime_square_mod_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt_three : p > 3) :
  p^2 % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_twelve_l1376_137644


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1376_137665

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : ℝ
  hb : ℝ
  hc : ℝ
  area : ℝ
  inradius : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.ha > 0 ∧ t.hb > 0 ∧ t.hc > 0 ∧
  t.area > 0 ∧ t.inradius > 0

def integerAltitudes (t : Triangle) : Prop :=
  ∃ (n₁ n₂ n₃ : ℕ), t.ha = n₁ ∧ t.hb = n₂ ∧ t.hc = n₃

def altitudesSumLessThan20 (t : Triangle) : Prop :=
  t.ha + t.hb + t.hc < 20

def integerInradius (t : Triangle) : Prop :=
  ∃ (n : ℕ), t.inradius = n

-- State the theorem
theorem triangle_area_theorem (t : Triangle) 
  (h1 : validTriangle t)
  (h2 : integerAltitudes t)
  (h3 : altitudesSumLessThan20 t)
  (h4 : integerInradius t) :
  t.area = 6 ∨ t.area = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1376_137665


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1376_137617

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / a + 1 / b ≥ 2 ∧ (1 / a + 1 / b = 2 ↔ a = 1 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1376_137617


namespace NUMINAMATH_CALUDE_tutorial_time_multiplier_l1376_137622

/-- Represents the time spent on various activities before playing a game --/
structure GamePreparationTime where
  download : ℝ
  install : ℝ
  tutorial : ℝ
  total : ℝ

/-- Theorem: Given the conditions, the tutorial time multiplier is 3 --/
theorem tutorial_time_multiplier (t : GamePreparationTime) : 
  t.download = 10 ∧ 
  t.install = t.download / 2 ∧ 
  t.total = 60 ∧ 
  t.total = t.download + t.install + t.tutorial → 
  t.tutorial = 3 * (t.download + t.install) :=
by sorry

end NUMINAMATH_CALUDE_tutorial_time_multiplier_l1376_137622


namespace NUMINAMATH_CALUDE_print_shop_cost_difference_l1376_137607

def print_shop_x_price : ℝ := 1.25
def print_shop_y_price : ℝ := 2.75
def print_shop_x_discount : ℝ := 0.10
def print_shop_y_discount : ℝ := 0.05
def print_shop_x_tax : ℝ := 0.07
def print_shop_y_tax : ℝ := 0.09
def num_copies : ℕ := 40

def calculate_total_cost (base_price discount tax : ℝ) (copies : ℕ) : ℝ :=
  let pre_discount := base_price * copies
  let discounted := pre_discount * (1 - discount)
  discounted * (1 + tax)

theorem print_shop_cost_difference :
  calculate_total_cost print_shop_y_price print_shop_y_discount print_shop_y_tax num_copies -
  calculate_total_cost print_shop_x_price print_shop_x_discount print_shop_x_tax num_copies =
  65.755 := by sorry

end NUMINAMATH_CALUDE_print_shop_cost_difference_l1376_137607


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1376_137696

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1376_137696


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l1376_137698

-- Define the triangle and its properties
structure Triangle where
  A : Real
  B : Real
  C_1 : Real
  C_2 : Real
  B_ext : Real
  h_B_gt_A : B > A
  h_angle_sum : A + B + C_1 + C_2 = 180
  h_ext_angle : B_ext = 180 - B

-- Theorem statement
theorem triangle_angle_relation (t : Triangle) :
  t.C_1 - t.C_2 = t.A + t.B_ext - 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l1376_137698


namespace NUMINAMATH_CALUDE_smallest_linear_combination_3003_55555_l1376_137683

theorem smallest_linear_combination_3003_55555 :
  ∃ (k : ℕ), k > 0 ∧ (∃ (m n : ℤ), k = 3003 * m + 55555 * n) ∧
  ∀ (j : ℕ), j > 0 → (∃ (a b : ℤ), j = 3003 * a + 55555 * b) → j ≥ k :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_linear_combination_3003_55555_l1376_137683


namespace NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l1376_137618

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_six_terms :
  let a : ℚ := 1/3
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 4/3 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l1376_137618


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l1376_137637

theorem solution_set_reciprocal_inequality (x : ℝ) :
  (∃ y ∈ Set.Ioo (0 : ℝ) (1/3 : ℝ), x = y) ↔ 1/x > 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l1376_137637


namespace NUMINAMATH_CALUDE_planes_parallel_or_intersect_l1376_137604

-- Define a type for 3D space
structure Space3D where
  -- Add necessary fields here
  
-- Define a type for planes in 3D space
structure Plane where
  -- Add necessary fields here

-- Define a type for lines in 3D space
structure Line where
  -- Add necessary fields here

-- Define what it means for a line to be parallel to a plane
def Line.parallelTo (l : Line) (p : Plane) : Prop :=
  sorry

-- Define what it means for a plane to contain a line
def Plane.contains (p : Plane) (l : Line) : Prop :=
  sorry

-- Define what it means for two planes to be parallel
def Plane.parallel (p1 p2 : Plane) : Prop :=
  sorry

-- Define what it means for two planes to intersect
def Plane.intersect (p1 p2 : Plane) : Prop :=
  sorry

-- The main theorem
theorem planes_parallel_or_intersect (p1 p2 : Plane) :
  (∃ (S : Set Line), Set.Infinite S ∧ (∀ l ∈ S, p1.contains l ∧ l.parallelTo p2)) →
  (p1.parallel p2 ∨ p1.intersect p2) :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_or_intersect_l1376_137604


namespace NUMINAMATH_CALUDE_quadratic_translation_l1376_137612

/-- Given a quadratic function f and its translated version g, 
    prove that f has the form -2x^2+1 -/
theorem quadratic_translation (f g : ℝ → ℝ) :
  (∀ x, g x = -2*x^2 + 4*x + 1) →
  (∀ x, g x = f (x - 1) + 2) →
  (∀ x, f x = -2*x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_translation_l1376_137612


namespace NUMINAMATH_CALUDE_log_inequality_l1376_137648

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_increasing : ∀ x y, 0 ≤ x → x < y → f x < f y

-- State the theorem
theorem log_inequality (x : ℝ) (h : f (Real.log x) < 0) : 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1376_137648


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l1376_137641

theorem pure_imaginary_solutions (x : ℂ) : 
  (x^5 - 4*x^4 + 6*x^3 - 50*x^2 - 100*x - 120 = 0 ∧ ∃ k : ℝ, x = k*I) ↔ 
  (x = I*Real.sqrt 14 ∨ x = -I*Real.sqrt 14) := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l1376_137641


namespace NUMINAMATH_CALUDE_bird_count_l1376_137603

/-- Represents the count of animals in a nature reserve --/
structure AnimalCount where
  birds : ℕ
  mythical : ℕ
  mammals : ℕ

/-- Theorem stating the number of two-legged birds in the nature reserve --/
theorem bird_count (ac : AnimalCount) : 
  ac.birds + ac.mythical + ac.mammals = 300 →
  2 * ac.birds + 3 * ac.mythical + 4 * ac.mammals = 708 →
  ac.birds = 192 := by
  sorry


end NUMINAMATH_CALUDE_bird_count_l1376_137603


namespace NUMINAMATH_CALUDE_simplify_expression_l1376_137620

theorem simplify_expression (x : ℝ) : 3*x + 4*x^3 + 2 - (7 - 3*x - 4*x^3) = 8*x^3 + 6*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1376_137620


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1376_137679

/-- Given a geometric sequence with sum of first n terms Sn = 24 and sum of first 3n terms S3n = 42,
    prove that the sum of first 2n terms S2n = 36 -/
theorem geometric_sequence_sum (n : ℕ) (Sn S2n S3n : ℝ) : 
  Sn = 24 → S3n = 42 → (S2n - Sn)^2 = Sn * (S3n - S2n) → S2n = 36 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1376_137679


namespace NUMINAMATH_CALUDE_chess_game_probability_l1376_137619

/-- The probability of a chess game resulting in a draw -/
def prob_draw : ℚ := 1/2

/-- The probability of player A winning the chess game -/
def prob_a_win : ℚ := 1/3

/-- The probability of player A not losing the chess game -/
def prob_a_not_lose : ℚ := prob_draw + prob_a_win

theorem chess_game_probability : prob_a_not_lose = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l1376_137619


namespace NUMINAMATH_CALUDE_blue_balls_count_l1376_137695

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := sorry

/-- The total number of balls in the bag -/
def total_balls : ℕ := 5 + blue_balls + 4

/-- The probability of picking two red balls -/
def prob_two_red : ℚ := 5 / total_balls * 4 / (total_balls - 1)

theorem blue_balls_count : 
  (5 : ℕ) > 0 ∧ 
  (4 : ℕ) > 0 ∧ 
  prob_two_red = 0.09523809523809523 →
  blue_balls = 6 := by sorry

end NUMINAMATH_CALUDE_blue_balls_count_l1376_137695


namespace NUMINAMATH_CALUDE_inequality_cube_l1376_137673

theorem inequality_cube (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (a - c)^3 > (b - c)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_cube_l1376_137673


namespace NUMINAMATH_CALUDE_log_identity_l1376_137686

theorem log_identity (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 1) :
  2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l1376_137686


namespace NUMINAMATH_CALUDE_johns_allowance_problem_l1376_137694

/-- The problem of calculating the fraction of John's remaining allowance spent at the toy store -/
theorem johns_allowance_problem (allowance : ℚ) :
  allowance = 345/100 →
  let arcade_spent := (3/5) * allowance
  let remaining_after_arcade := allowance - arcade_spent
  let candy_spent := 92/100
  let toy_spent := remaining_after_arcade - candy_spent
  (toy_spent / remaining_after_arcade) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_problem_l1376_137694


namespace NUMINAMATH_CALUDE_fraction_relation_l1376_137689

theorem fraction_relation (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c = d + 4) :
  d / a = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relation_l1376_137689


namespace NUMINAMATH_CALUDE_cupboard_cost_price_l1376_137625

theorem cupboard_cost_price (selling_price selling_price_with_profit : ℝ) 
  (h1 : selling_price = 0.88 * 6250)
  (h2 : selling_price_with_profit = 1.12 * 6250)
  (h3 : selling_price_with_profit = selling_price + 1500) : 
  6250 = 6250 := by
sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_l1376_137625


namespace NUMINAMATH_CALUDE_rhombus_area_l1376_137643

/-- Rhombus in a plane rectangular coordinate system -/
structure Rhombus where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The area of a rhombus given its vertices -/
def area (r : Rhombus) : ℝ := sorry

/-- Theorem: Area of rhombus ABCD with given conditions -/
theorem rhombus_area : 
  ∀ (r : Rhombus), 
    r.A = (-4, 0) →
    r.B = (0, -3) →
    (∃ (x y : ℝ), r.C = (x, 0) ∧ r.D = (0, y)) →  -- vertices on axes
    area r = 24 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1376_137643


namespace NUMINAMATH_CALUDE_square_fraction_count_l1376_137681

theorem square_fraction_count : 
  ∃! (S : Finset ℤ), 
    S.card = 2 ∧ 
    (∀ n ∈ S, 0 ≤ n ∧ n < 25 ∧ ∃ k : ℤ, (n : ℚ) / (25 - n) = k^2) ∧
    (∀ n : ℤ, 0 ≤ n → n < 25 → (∃ k : ℤ, (n : ℚ) / (25 - n) = k^2) → n ∈ S) :=
sorry

end NUMINAMATH_CALUDE_square_fraction_count_l1376_137681


namespace NUMINAMATH_CALUDE_f_properties_l1376_137687

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (f (π / 12) = 1 / 2) ∧
  (Set.Icc 0 (3 / 2) = Set.image f (Set.Icc 0 (π / 2))) ∧
  (∃ (zeros : Finset ℝ), zeros.card = 5 ∧
    (∀ x ∈ zeros, x ∈ Set.Icc 0 (2 * π) ∧ f x = 0) ∧
    (∀ x ∈ Set.Icc 0 (2 * π), f x = 0 → x ∈ zeros)) ∧
  (∃ (zeros : Finset ℝ), zeros.card = 5 ∧
    (∀ x ∈ zeros, x ∈ Set.Icc 0 (2 * π) ∧ f x = 0) ∧
    (∀ x ∈ Set.Icc 0 (2 * π), f x = 0 → x ∈ zeros) ∧
    (zeros.sum id = 16 * π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1376_137687


namespace NUMINAMATH_CALUDE_i_to_2016_equals_1_l1376_137623

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem i_to_2016_equals_1 : i ^ 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_i_to_2016_equals_1_l1376_137623


namespace NUMINAMATH_CALUDE_no_linear_factor_l1376_137609

theorem no_linear_factor (x y z : ℤ) : 
  ¬ ∃ (a b c d : ℤ), (a*x + b*y + c*z + d) ∣ (x^2 - y^2 - z^2 + 2*x*y + x + y - z) :=
sorry

end NUMINAMATH_CALUDE_no_linear_factor_l1376_137609


namespace NUMINAMATH_CALUDE_system_solution_l1376_137697

theorem system_solution (m n : ℝ) : 
  (m * 2 + n * 4 = 8 ∧ 2 * m * 2 - 3 * n * 4 = -4) → m = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1376_137697


namespace NUMINAMATH_CALUDE_swimmer_distance_l1376_137682

/-- Proves that a swimmer covers 8 km when swimming against a current for 5 hours -/
theorem swimmer_distance (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ) :
  swimmer_speed = 3 →
  current_speed = 1.4 →
  time = 5 →
  (swimmer_speed - current_speed) * time = 8 := by
sorry

end NUMINAMATH_CALUDE_swimmer_distance_l1376_137682


namespace NUMINAMATH_CALUDE_min_operations_needed_l1376_137668

-- Define the type for letters
inductive Letter | A | B | C | D | E | F | G

-- Define the type for positions in the circle
inductive Position | Center | Top | TopRight | BottomRight | Bottom | BottomLeft | TopLeft

-- Define the configuration as a function from Position to Letter
def Configuration := Position → Letter

-- Define the initial configuration
def initial_config : Configuration := sorry

-- Define the final configuration
def final_config : Configuration := sorry

-- Define a valid operation
def valid_operation (c : Configuration) : Configuration := sorry

-- Define the number of operations needed to transform one configuration to another
def operations_needed (start finish : Configuration) : ℕ := sorry

-- The main theorem
theorem min_operations_needed :
  operations_needed initial_config final_config = 3 := by sorry

end NUMINAMATH_CALUDE_min_operations_needed_l1376_137668


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l1376_137662

theorem sqrt_sum_problem (x : ℝ) : 
  Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4 → 
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l1376_137662


namespace NUMINAMATH_CALUDE_hundredth_training_day_l1376_137608

def training_program (start_day : Nat) (n : Nat) : Nat :=
  (start_day + (n - 1) * 8 + (n - 1) % 6) % 7

theorem hundredth_training_day :
  training_program 1 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_training_day_l1376_137608


namespace NUMINAMATH_CALUDE_grape_rate_proof_l1376_137680

/-- The rate per kg for grapes -/
def grape_rate : ℝ := 70

/-- The rate per kg for mangoes -/
def mango_rate : ℝ := 55

/-- The quantity of grapes and mangoes in kg -/
def quantity : ℝ := 8

/-- The total cost paid to the shopkeeper -/
def total_cost : ℝ := 1000

theorem grape_rate_proof :
  grape_rate * quantity + mango_rate * quantity = total_cost :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_proof_l1376_137680


namespace NUMINAMATH_CALUDE_notebook_cost_l1376_137605

/-- The total cost of notebooks with given prices and quantities -/
def total_cost (green_price : ℕ) (green_quantity : ℕ) (black_price : ℕ) (pink_price : ℕ) : ℕ :=
  green_price * green_quantity + black_price + pink_price

/-- Theorem: The total cost of 4 notebooks (2 green at $10 each, 1 black at $15, and 1 pink at $10) is $45 -/
theorem notebook_cost : total_cost 10 2 15 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l1376_137605


namespace NUMINAMATH_CALUDE_teal_survey_result_l1376_137601

theorem teal_survey_result (total : ℕ) (more_green : ℕ) (both : ℕ) (neither : ℕ) (undecided : ℕ) :
  total = 150 →
  more_green = 90 →
  both = 40 →
  neither = 20 →
  undecided = 10 →
  ∃ (more_blue : ℕ), more_blue = 70 ∧ 
    total = more_green + more_blue - both + neither + undecided :=
by sorry

end NUMINAMATH_CALUDE_teal_survey_result_l1376_137601


namespace NUMINAMATH_CALUDE_combined_price_increase_percentage_l1376_137653

def skateboard_initial_price : ℝ := 120
def knee_pads_initial_price : ℝ := 30
def skateboard_increase_percent : ℝ := 8
def knee_pads_increase_percent : ℝ := 15

theorem combined_price_increase_percentage :
  let skateboard_new_price := skateboard_initial_price * (1 + skateboard_increase_percent / 100)
  let knee_pads_new_price := knee_pads_initial_price * (1 + knee_pads_increase_percent / 100)
  let initial_total := skateboard_initial_price + knee_pads_initial_price
  let new_total := skateboard_new_price + knee_pads_new_price
  (new_total - initial_total) / initial_total * 100 = 9.4 := by sorry

end NUMINAMATH_CALUDE_combined_price_increase_percentage_l1376_137653


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1376_137611

/-- The trajectory of the center of a moving circle that is externally tangent to two fixed circles -/
theorem moving_circle_trajectory (x y : ℝ) : 
  (∃ (r : ℝ), 
    -- First fixed circle
    (∃ (x₁ y₁ : ℝ), x₁^2 + y₁^2 + 4*x₁ + 3 = 0 ∧ 
      -- Moving circle is externally tangent to the first fixed circle
      (x - x₁)^2 + (y - y₁)^2 = (r + 1)^2) ∧ 
    -- Second fixed circle
    (∃ (x₂ y₂ : ℝ), x₂^2 + y₂^2 - 4*x₂ - 5 = 0 ∧ 
      -- Moving circle is externally tangent to the second fixed circle
      (x - x₂)^2 + (y - y₂)^2 = (r + 3)^2)) →
  -- The trajectory of the center of the moving circle
  x^2 - 3*y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1376_137611


namespace NUMINAMATH_CALUDE_annas_cupcake_earnings_l1376_137654

/-- Calculates Anna's earnings from selling cupcakes -/
def annas_earnings (num_trays : ℕ) (cupcakes_per_tray : ℕ) (price_per_cupcake : ℚ) (sold_fraction : ℚ) : ℚ :=
  (num_trays * cupcakes_per_tray : ℚ) * sold_fraction * price_per_cupcake

theorem annas_cupcake_earnings :
  annas_earnings 10 30 (5/2) (7/10) = 525 := by
  sorry

end NUMINAMATH_CALUDE_annas_cupcake_earnings_l1376_137654


namespace NUMINAMATH_CALUDE_equation_solution_l1376_137602

theorem equation_solution (z : ℚ) : 
  Real.sqrt (5 - 4 * z) = 10 → z = -95/4 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1376_137602


namespace NUMINAMATH_CALUDE_cody_final_amount_l1376_137678

/-- Given an initial amount, a gift amount, and an expense amount, 
    calculate the final amount of money. -/
def finalAmount (initial gift expense : ℕ) : ℕ :=
  initial + gift - expense

/-- Theorem stating that given the specific values from the problem,
    the final amount is 35 dollars. -/
theorem cody_final_amount : 
  finalAmount 45 9 19 = 35 := by sorry

end NUMINAMATH_CALUDE_cody_final_amount_l1376_137678


namespace NUMINAMATH_CALUDE_gas_bill_payment_l1376_137613

def electricity_bill : ℚ := 60
def gas_bill : ℚ := 40
def water_bill : ℚ := 40
def internet_bill : ℚ := 25

def gas_bill_paid_initially : ℚ := (3 / 4) * gas_bill
def water_bill_paid : ℚ := (1 / 2) * water_bill
def internet_bill_paid : ℚ := 4 * 5

def remaining_to_pay : ℚ := 30

theorem gas_bill_payment (payment : ℚ) : 
  gas_bill + water_bill + internet_bill - 
  (gas_bill_paid_initially + water_bill_paid + internet_bill_paid + payment) = 
  remaining_to_pay → 
  payment = 5 := by sorry

end NUMINAMATH_CALUDE_gas_bill_payment_l1376_137613


namespace NUMINAMATH_CALUDE_kenny_trumpet_practice_time_l1376_137616

/-- Given Kenny's activities last week, prove the time he spent practicing trumpet. -/
theorem kenny_trumpet_practice_time :
  ∀ (basketball_time running_time trumpet_time : ℕ),
  basketball_time = 10 →
  running_time = 2 * basketball_time →
  trumpet_time = 2 * running_time →
  trumpet_time = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_kenny_trumpet_practice_time_l1376_137616


namespace NUMINAMATH_CALUDE_pyramid_cross_section_theorem_l1376_137645

/-- Represents a regular pyramid -/
structure RegularPyramid where
  lateralEdgeLength : ℝ

/-- Represents a cross-section of a pyramid -/
structure CrossSection where
  areaRatio : ℝ  -- ratio of cross-section area to base area

/-- 
Given a regular pyramid with lateral edge length 3 cm, if a plane parallel to the base
creates a cross-section with an area 1/9 of the base area, then the lateral edge length
of the smaller pyramid removed is 1 cm.
-/
theorem pyramid_cross_section_theorem (p : RegularPyramid) (cs : CrossSection) :
  p.lateralEdgeLength = 3 → cs.areaRatio = 1/9 → 
  ∃ (smallerPyramid : RegularPyramid), smallerPyramid.lateralEdgeLength = 1 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_cross_section_theorem_l1376_137645


namespace NUMINAMATH_CALUDE_sqrt_inequality_range_l1376_137664

theorem sqrt_inequality_range (m : ℝ) : 
  Real.sqrt (2 * m + 1) > Real.sqrt (m^2 + m - 1) ↔ 
  m ∈ Set.Ici ((Real.sqrt 5 - 1) / 2) ∩ Set.Iio 2 :=
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_range_l1376_137664


namespace NUMINAMATH_CALUDE_prime_9k_plus_1_divides_cubic_l1376_137666

theorem prime_9k_plus_1_divides_cubic (p : Nat) (k : Nat) (h_prime : Nat.Prime p) (h_form : p = 9*k + 1) :
  ∃ n : ℤ, (n^3 - 3*n + 1) % p = 0 := by sorry

end NUMINAMATH_CALUDE_prime_9k_plus_1_divides_cubic_l1376_137666


namespace NUMINAMATH_CALUDE_range_of_m_l1376_137674

theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc 0 1, x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1376_137674


namespace NUMINAMATH_CALUDE_shirt_tie_outfits_l1376_137639

theorem shirt_tie_outfits (shirts : ℕ) (ties : ℕ) :
  shirts = 7 → ties = 4 → shirts * ties = 28 := by
sorry

end NUMINAMATH_CALUDE_shirt_tie_outfits_l1376_137639


namespace NUMINAMATH_CALUDE_window_width_is_20_inches_l1376_137640

/-- Represents the dimensions of a glass pane -/
structure PaneDimensions where
  width : ℝ
  height : ℝ

/-- Represents the configuration of a window -/
structure WindowConfig where
  pane : PaneDimensions
  columns : ℕ
  rows : ℕ
  borderWidth : ℝ

/-- Calculates the total width of a window given its configuration -/
def totalWidth (config : WindowConfig) : ℝ :=
  config.columns * config.pane.width + (config.columns + 1) * config.borderWidth

/-- Theorem stating the total width of the window is 20 inches -/
theorem window_width_is_20_inches (config : WindowConfig) 
  (h1 : config.columns = 3)
  (h2 : config.rows = 2)
  (h3 : config.pane.height = 3 * config.pane.width)
  (h4 : config.borderWidth = 2) :
  totalWidth config = 20 := by
  sorry

#check window_width_is_20_inches

end NUMINAMATH_CALUDE_window_width_is_20_inches_l1376_137640


namespace NUMINAMATH_CALUDE_probability_of_pair_after_removal_l1376_137624

/-- Represents a deck of cards -/
structure Deck :=
  (cards : Finset (Fin 10 × Fin 4))
  (card_count : cards.card = 40)

/-- Represents the deck after removing a matching pair -/
def RemainingDeck (d : Deck) : Finset (Fin 10 × Fin 4) :=
  d.cards.filter (λ x ↦ x.2 ≠ 3)

/-- The probability of selecting a matching pair from the remaining deck -/
def ProbabilityOfPair (d : Deck) : ℚ :=
  55 / 703

theorem probability_of_pair_after_removal (d : Deck) :
  ProbabilityOfPair d = 55 / 703 :=
sorry

end NUMINAMATH_CALUDE_probability_of_pair_after_removal_l1376_137624


namespace NUMINAMATH_CALUDE_total_panels_eq_600_l1376_137652

/-- The number of houses in the neighborhood -/
def num_houses : ℕ := 10

/-- The number of double windows downstairs in each house -/
def num_double_windows : ℕ := 6

/-- The number of glass panels in each double window -/
def panels_per_double_window : ℕ := 4

/-- The number of single windows upstairs in each house -/
def num_single_windows : ℕ := 8

/-- The number of glass panels in each single window -/
def panels_per_single_window : ℕ := 3

/-- The number of bay windows in each house -/
def num_bay_windows : ℕ := 2

/-- The number of glass panels in each bay window -/
def panels_per_bay_window : ℕ := 6

/-- The total number of glass panels in the neighborhood -/
def total_panels : ℕ := num_houses * (
  num_double_windows * panels_per_double_window +
  num_single_windows * panels_per_single_window +
  num_bay_windows * panels_per_bay_window
)

theorem total_panels_eq_600 : total_panels = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_panels_eq_600_l1376_137652


namespace NUMINAMATH_CALUDE_equilateral_triangle_condition_l1376_137615

theorem equilateral_triangle_condition (a b c : ℂ) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  Complex.abs a = 1 →
  Complex.abs b = 1 →
  Complex.abs c = 1 →
  Complex.abs (a + b - c) ^ 2 + Complex.abs (b + c - a) ^ 2 + Complex.abs (c + a - b) ^ 2 = 12 →
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_condition_l1376_137615
