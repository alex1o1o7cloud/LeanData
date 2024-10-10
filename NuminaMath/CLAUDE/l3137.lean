import Mathlib

namespace sequence_properties_l3137_313735

def sequence_a (n : ℕ) : ℝ := (n + 1 : ℝ) * 2^(n - 1)

def partial_sum (n : ℕ) : ℝ := n * 2^n - 2^n

theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, n > 0 → S n = 2 * a n - 2^n) :
  (∀ n : ℕ, n > 0 → a n / 2^n - a (n-1) / 2^(n-1) = 1/2) ∧
  (∀ n : ℕ, n > 0 → a n = sequence_a n) ∧
  (∀ n : ℕ, n > 0 → S n = partial_sum n) := by
  sorry

end sequence_properties_l3137_313735


namespace wind_power_scientific_notation_l3137_313752

/-- Proves that 56 million kilowatts is equal to 5.6 × 10^7 kilowatts in scientific notation -/
theorem wind_power_scientific_notation : 
  (56000000 : ℝ) = 5.6 * (10 ^ 7) := by
  sorry

end wind_power_scientific_notation_l3137_313752


namespace min_xy_value_l3137_313783

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 9) : 
  (x * y : ℕ) ≥ 108 := by
  sorry

end min_xy_value_l3137_313783


namespace complex_fraction_sum_l3137_313711

theorem complex_fraction_sum (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^3 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω)) = 3 / ω) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1)) = 3 := by
sorry

end complex_fraction_sum_l3137_313711


namespace max_k_for_quadratic_root_difference_l3137_313765

theorem max_k_for_quadratic_root_difference (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   x^2 + k*x + 10 = 0 ∧ 
   y^2 + k*y + 10 = 0 ∧ 
   |x - y| = Real.sqrt 81) →
  k ≤ 11 :=
by sorry

end max_k_for_quadratic_root_difference_l3137_313765


namespace inequality_proof_l3137_313745

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ 1) (h2 : b ≥ 1) (h3 : c ≥ 1) (h4 : a + b + c = 9) :
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end inequality_proof_l3137_313745


namespace square_difference_divided_problem_solution_l3137_313797

theorem square_difference_divided (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution : (305^2 - 275^2) / 30 = 580 :=
by sorry

end square_difference_divided_problem_solution_l3137_313797


namespace existence_of_equal_function_values_l3137_313790

theorem existence_of_equal_function_values (n : ℕ) (h_n : n ≤ 44) 
  (f : ℕ+ × ℕ+ → Fin n) : 
  ∃ (i j l k m p : ℕ+), 
    f (i, j) = f (i, k) ∧ f (i, j) = f (l, j) ∧ f (i, j) = f (l, k) ∧
    1989 * m ≤ i ∧ i < l ∧ l < 1989 + 1989 * m ∧
    1989 * p ≤ j ∧ j < k ∧ k < 1989 + 1989 * p :=
by sorry

end existence_of_equal_function_values_l3137_313790


namespace range_of_a_l3137_313716

-- Define the propositions p, q, and r
def p (x : ℝ) : Prop := (x - 3) * (x + 1) < 0
def q (x : ℝ) : Prop := (x - 2) / (x - 4) < 0
def r (x a : ℝ) : Prop := a < x ∧ x < 2 * a

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, (p x ∧ q x) → r x a) →
  (3 / 2 ≤ a ∧ a ≤ 2) :=
by sorry

end range_of_a_l3137_313716


namespace sqrt_identity_l3137_313723

theorem sqrt_identity (a b : ℝ) (h : a > Real.sqrt b) :
  Real.sqrt ((a + Real.sqrt (a^2 - b)) / 2) + Real.sqrt ((a - Real.sqrt (a^2 - b)) / 2) =
  Real.sqrt (a + Real.sqrt b) ∧
  Real.sqrt ((a + Real.sqrt (a^2 - b)) / 2) - Real.sqrt ((a - Real.sqrt (a^2 - b)) / 2) =
  Real.sqrt (a - Real.sqrt b) := by
sorry

end sqrt_identity_l3137_313723


namespace mother_extra_rides_l3137_313782

/-- The number of times Billy rode his bike -/
def billy_rides : ℕ := 17

/-- The number of times John rode his bike -/
def john_rides : ℕ := 2 * billy_rides

/-- The number of times the mother rode her bike -/
def mother_rides (x : ℕ) : ℕ := john_rides + x

/-- The total number of times they all rode their bikes -/
def total_rides : ℕ := 95

/-- Theorem stating that the mother rode her bike 10 times more than John -/
theorem mother_extra_rides : 
  ∃ x : ℕ, x = 10 ∧ mother_rides x = john_rides + x ∧ 
  billy_rides + john_rides + mother_rides x = total_rides :=
sorry

end mother_extra_rides_l3137_313782


namespace opposite_of_three_l3137_313734

theorem opposite_of_three : -(3 : ℤ) = -3 := by sorry

end opposite_of_three_l3137_313734


namespace egg_collection_total_l3137_313725

/-- The number of dozen eggs Benjamin collects -/
def benjamin_eggs : ℕ := 6

/-- The number of dozen eggs Carla collects -/
def carla_eggs : ℕ := 3 * benjamin_eggs

/-- The number of dozen eggs Trisha collects -/
def trisha_eggs : ℕ := benjamin_eggs - 4

/-- The total number of dozen eggs collected by Benjamin, Carla, and Trisha -/
def total_eggs : ℕ := benjamin_eggs + carla_eggs + trisha_eggs

theorem egg_collection_total : total_eggs = 26 := by
  sorry

end egg_collection_total_l3137_313725


namespace absent_student_percentage_l3137_313775

theorem absent_student_percentage
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (boys_absent_fraction : ℚ)
  (girls_absent_fraction : ℚ)
  (h1 : total_students = 150)
  (h2 : boys = 90)
  (h3 : girls = 60)
  (h4 : boys_absent_fraction = 1 / 6)
  (h5 : girls_absent_fraction = 1 / 4)
  (h6 : total_students = boys + girls) :
  (↑boys * boys_absent_fraction + ↑girls * girls_absent_fraction) / ↑total_students = 1 / 5 :=
by sorry

end absent_student_percentage_l3137_313775


namespace sequence_sum_equals_exp_l3137_313766

/-- Given a positive integer m, y_k is a sequence defined by:
    y_0 = 1
    y_1 = m
    y_{k+2} = ((m+1)y_{k+1} - (m-k)y_k) / (k+1) for k ≥ 0
    This theorem states that the sum of all terms in the sequence equals e^(m+1) -/
theorem sequence_sum_equals_exp (m : ℕ+) : ∃ (y : ℕ → ℝ), 
  y 0 = 1 ∧ 
  y 1 = m ∧ 
  (∀ k : ℕ, y (k + 2) = ((m + 1 : ℝ) * y (k + 1) - (m - k) * y k) / (k + 1)) ∧
  (∑' k, y k) = Real.exp (m + 1) := by
  sorry

end sequence_sum_equals_exp_l3137_313766


namespace complement_intersection_problem_l3137_313769

def U : Set ℕ := {x | x < 6}
def P : Set ℕ := {2, 4}
def Q : Set ℕ := {1, 3, 4, 6}

theorem complement_intersection_problem :
  (U \ P) ∩ Q = {1, 3} := by sorry

end complement_intersection_problem_l3137_313769


namespace cookies_problem_l3137_313780

theorem cookies_problem (glenn_cookies : ℕ) (h1 : glenn_cookies = 24) 
  (h2 : ∃ kenny_cookies : ℕ, glenn_cookies = 4 * kenny_cookies) 
  (h3 : ∃ chris_cookies : ℕ, chris_cookies * 2 = kenny_cookies) : 
  glenn_cookies + kenny_cookies + chris_cookies = 33 :=
by
  sorry

end cookies_problem_l3137_313780


namespace quadratic_function_values_l3137_313707

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: If f(1) = 3 and f(2) = 5, then f(3) = 7 -/
theorem quadratic_function_values (a b c : ℝ) :
  f a b c 1 = 3 → f a b c 2 = 5 → f a b c 3 = 7 := by
  sorry

end quadratic_function_values_l3137_313707


namespace opponent_total_runs_is_67_l3137_313758

/-- Represents the scores of a baseball team in a series of games. -/
structure BaseballScores :=
  (scores : List Nat)
  (lostByTwoGames : Nat)
  (wonByTripleGames : Nat)

/-- Calculates the total runs scored by the opponents. -/
def opponentTotalRuns (bs : BaseballScores) : Nat :=
  sorry

/-- The theorem states that given the specific conditions of the baseball team's games,
    the total runs scored by their opponents is 67. -/
theorem opponent_total_runs_is_67 :
  let bs : BaseballScores := {
    scores := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    lostByTwoGames := 5,
    wonByTripleGames := 5
  }
  opponentTotalRuns bs = 67 := by sorry

end opponent_total_runs_is_67_l3137_313758


namespace not_greater_than_three_equiv_l3137_313748

theorem not_greater_than_three_equiv (a : ℝ) : (¬(a > 3)) ↔ (a ≤ 3) := by
  sorry

end not_greater_than_three_equiv_l3137_313748


namespace expression_independent_of_alpha_l3137_313715

theorem expression_independent_of_alpha :
  ∀ α : ℝ, 
    Real.sin (250 * π / 180 + α) * Real.cos (200 * π / 180 - α) - 
    Real.cos (240 * π / 180) * Real.cos (220 * π / 180 - 2 * α) = 
    1 / 2 := by
  sorry

end expression_independent_of_alpha_l3137_313715


namespace evaluate_expression_l3137_313788

theorem evaluate_expression : (5^5 * 5^3) / 3^6 * 2^5 = 12480000 / 729 := by
  sorry

end evaluate_expression_l3137_313788


namespace rice_cooking_is_algorithm_l3137_313713

/-- Characteristics of an algorithm -/
structure AlgorithmCharacteristics where
  finite : Bool
  definite : Bool
  sequential : Bool
  correct : Bool
  nonUnique : Bool
  universal : Bool

/-- Representation of an algorithm -/
inductive AlgorithmRepresentation
  | NaturalLanguage
  | GraphicalLanguage
  | ProgrammingLanguage

/-- Steps for cooking rice -/
inductive RiceCookingStep
  | WashPot
  | RinseRice
  | AddWater
  | Heat

/-- Definition of an algorithm -/
def isAlgorithm (steps : List RiceCookingStep) (representation : AlgorithmRepresentation) 
  (characteristics : AlgorithmCharacteristics) : Prop :=
  characteristics.finite ∧
  characteristics.definite ∧
  characteristics.sequential ∧
  characteristics.correct ∧
  characteristics.nonUnique ∧
  characteristics.universal

/-- Theorem: The steps for cooking rice form an algorithm -/
theorem rice_cooking_is_algorithm : 
  ∃ (representation : AlgorithmRepresentation) (characteristics : AlgorithmCharacteristics),
    isAlgorithm [RiceCookingStep.WashPot, RiceCookingStep.RinseRice, 
                 RiceCookingStep.AddWater, RiceCookingStep.Heat] 
                representation characteristics :=
  sorry

end rice_cooking_is_algorithm_l3137_313713


namespace mary_jamison_weight_difference_l3137_313798

/-- Proves that Mary weighs 20 lbs less than Jamison given the conditions in the problem -/
theorem mary_jamison_weight_difference :
  ∀ (john mary jamison : ℝ),
    mary = 160 →
    john = mary + (1/4) * mary →
    john + mary + jamison = 540 →
    jamison - mary = 20 := by
  sorry

end mary_jamison_weight_difference_l3137_313798


namespace completing_square_quadratic_l3137_313796

/-- Given a quadratic equation x² - 4x - 2 = 0, prove that the correct completion of the square is (x-2)² = 6 -/
theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 4*x - 2 = 0 → (x - 2)^2 = 6 := by
  sorry

end completing_square_quadratic_l3137_313796


namespace drama_club_two_skills_l3137_313701

/-- Represents the number of students with a particular combination of skills -/
structure SkillCount where
  write : Nat
  direct : Nat
  produce : Nat
  write_direct : Nat
  write_produce : Nat
  direct_produce : Nat

/-- Represents the constraints of the drama club problem -/
def drama_club_constraints (sc : SkillCount) : Prop :=
  sc.write + sc.direct + sc.produce + sc.write_direct + sc.write_produce + sc.direct_produce = 150 ∧
  sc.write + sc.write_direct + sc.write_produce = 90 ∧
  sc.direct + sc.write_direct + sc.direct_produce = 60 ∧
  sc.produce + sc.write_produce + sc.direct_produce = 110

/-- The main theorem stating that under the given constraints, 
    the number of students with exactly two skills is 110 -/
theorem drama_club_two_skills (sc : SkillCount) 
  (h : drama_club_constraints sc) : 
  sc.write_direct + sc.write_produce + sc.direct_produce = 110 := by
  sorry

end drama_club_two_skills_l3137_313701


namespace hypotenuse_length_l3137_313794

theorem hypotenuse_length (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a^2 + b^2) * (a^2 + b^2 + 1) = 12 →
  a^2 + b^2 = c^2 →
  c = Real.sqrt 3 := by
sorry

end hypotenuse_length_l3137_313794


namespace bella_stamps_count_l3137_313719

/-- The number of snowflake stamps Bella bought -/
def snowflake_stamps : ℕ := 11

/-- The number of truck stamps Bella bought -/
def truck_stamps : ℕ := snowflake_stamps + 9

/-- The number of rose stamps Bella bought -/
def rose_stamps : ℕ := truck_stamps - 13

/-- The total number of stamps Bella bought -/
def total_stamps : ℕ := snowflake_stamps + truck_stamps + rose_stamps

theorem bella_stamps_count : total_stamps = 38 := by
  sorry

end bella_stamps_count_l3137_313719


namespace ratio_of_two_numbers_l3137_313742

theorem ratio_of_two_numbers (x y : ℝ) (h1 : x + y = 33) (h2 : x = 22) :
  y / x = 1 / 2 := by sorry

end ratio_of_two_numbers_l3137_313742


namespace complex_fraction_simplification_l3137_313767

theorem complex_fraction_simplification :
  let z : ℂ := (3 - 2*I) / (5 - 2*I)
  z = 19/29 - (4/29)*I :=
by sorry

end complex_fraction_simplification_l3137_313767


namespace triangle_inequality_fraction_l3137_313792

theorem triangle_inequality_fraction (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b) : 
  (a + b) / (1 + a + b) > c / (1 + c) := by
sorry

end triangle_inequality_fraction_l3137_313792


namespace sufficient_not_necessary_l3137_313793

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 2 ∧ y ≥ 2 → x + y ≥ 4) ∧
  (∃ x y : ℝ, x + y ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) :=
by sorry

end sufficient_not_necessary_l3137_313793


namespace total_seeds_calculation_l3137_313773

/-- The number of seeds planted in each flower bed -/
def seeds_per_bed : ℕ := 6

/-- The number of flower beds -/
def num_beds : ℕ := 9

/-- The total number of seeds planted -/
def total_seeds : ℕ := seeds_per_bed * num_beds

theorem total_seeds_calculation : total_seeds = 54 := by
  sorry

end total_seeds_calculation_l3137_313773


namespace circle_ray_no_intersection_l3137_313779

/-- Given a circle (x-a)^2 + y^2 = 4 and a ray y = √3x (x ≥ 0) with no common points,
    the range of values for the real number a is {a | a < -2 or a > (4/3)√3}. -/
theorem circle_ray_no_intersection (a : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + y^2 = 4 → y ≠ Real.sqrt 3 * x ∨ x < 0) ↔ 
  (a < -2 ∨ a > (4/3) * Real.sqrt 3) :=
sorry

end circle_ray_no_intersection_l3137_313779


namespace x_range_for_quadratic_inequality_l3137_313776

theorem x_range_for_quadratic_inequality (x : ℝ) :
  (∀ a : ℝ, -1 ≤ a ∧ a ≤ 1 → x^2 + (a-4)*x + 4-2*a > 0) ↔
  (x < -3 ∨ x > -2) :=
by sorry

end x_range_for_quadratic_inequality_l3137_313776


namespace town_population_problem_l3137_313706

theorem town_population_problem (original_population : ℕ) : 
  (((original_population + 1200) : ℚ) * (1 - 11/100) : ℚ).floor = original_population - 32 → 
  original_population = 10000 := by
  sorry

end town_population_problem_l3137_313706


namespace circle_diameter_from_area_l3137_313709

theorem circle_diameter_from_area (A : Real) (d : Real) :
  A = 225 * Real.pi → d = 30 :=
by
  sorry

end circle_diameter_from_area_l3137_313709


namespace average_speed_calculation_l3137_313730

def distance : ℝ := 360
def time : ℝ := 4.5

theorem average_speed_calculation : distance / time = 80 := by
  sorry

end average_speed_calculation_l3137_313730


namespace school_population_l3137_313762

/-- Given a school with boys, girls, and teachers, prove that the total number of people is 61t, where t is the number of teachers. -/
theorem school_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 12 * t) : 
  b + g + t = 61 * t := by
  sorry

end school_population_l3137_313762


namespace inequality_proof_l3137_313756

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a / (2 * b + 2 * c)) + Real.sqrt (b / (2 * a + 2 * c)) + Real.sqrt (c / (2 * a + 2 * b)) > 2 / 3 := by
  sorry

end inequality_proof_l3137_313756


namespace seojun_pizza_problem_l3137_313738

/-- Seojun's pizza problem -/
theorem seojun_pizza_problem (initial_pizza : ℚ) : 
  initial_pizza - 7/3 = 3/2 →
  initial_pizza + 7/3 = 37/6 := by
  sorry

end seojun_pizza_problem_l3137_313738


namespace largest_n_divisibility_l3137_313727

theorem largest_n_divisibility : ∃ (n : ℕ), n = 302 ∧ 
  (∀ m : ℕ, m > 302 → ¬(m + 11 ∣ m^3 + 101)) ∧
  (302 + 11 ∣ 302^3 + 101) := by
  sorry

end largest_n_divisibility_l3137_313727


namespace insulation_cost_for_given_tank_l3137_313718

/-- Calculates the surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

/-- Calculates the cost of insulating a rectangular tank -/
def insulation_cost (l w h cost_per_sqft : ℝ) : ℝ :=
  surface_area l w h * cost_per_sqft

/-- Theorem: The cost of insulating a rectangular tank with given dimensions -/
theorem insulation_cost_for_given_tank :
  insulation_cost 7 3 2 20 = 1640 := by
  sorry

end insulation_cost_for_given_tank_l3137_313718


namespace divisibility_by_four_l3137_313739

theorem divisibility_by_four (n : ℕ+) :
  4 ∣ (n * Nat.choose (2 * n) n) ↔ ¬∃ k : ℕ, n = 2^k := by
sorry

end divisibility_by_four_l3137_313739


namespace subtraction_problem_l3137_313726

theorem subtraction_problem : 444 - 44 - 4 = 396 := by
  sorry

end subtraction_problem_l3137_313726


namespace arithmetic_sequence_sum_l3137_313702

theorem arithmetic_sequence_sum : ∀ (a₁ a_last d n : ℕ),
  a₁ = 1 →
  a_last = 23 →
  d = 2 →
  n = (a_last - a₁) / d + 1 →
  (n : ℝ) * (a₁ + a_last) / 2 = 144 :=
by
  sorry

end arithmetic_sequence_sum_l3137_313702


namespace johns_total_payment_l3137_313774

/-- Calculates the total amount John paid for his dog's vet appointments and insurance -/
def total_payment (num_appointments : ℕ) (appointment_cost : ℚ) (insurance_cost : ℚ) (insurance_coverage : ℚ) : ℚ :=
  let first_appointment_cost := appointment_cost
  let insurance_payment := insurance_cost
  let subsequent_appointments_cost := appointment_cost * (num_appointments - 1 : ℚ)
  let covered_amount := subsequent_appointments_cost * insurance_coverage
  let out_of_pocket := subsequent_appointments_cost - covered_amount
  first_appointment_cost + insurance_payment + out_of_pocket

/-- Theorem stating that John's total payment for his dog's vet appointments and insurance is $660 -/
theorem johns_total_payment :
  total_payment 3 400 100 0.8 = 660 := by
  sorry

end johns_total_payment_l3137_313774


namespace sum_of_coefficients_l3137_313705

-- Define the polynomial representation
def polynomial (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) (x : ℝ) : ℝ :=
  a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10

-- Define the given equation
def equation (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) (x : ℝ) : Prop :=
  (1 - 2*x)^10 = polynomial a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ x

-- Theorem to prove
theorem sum_of_coefficients 
  (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x, equation a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ x) →
  10*a₁ + 9*a₂ + 8*a₃ + 7*a₄ + 6*a₅ + 5*a₆ + 4*a₇ + 3*a₈ + 2*a₉ + a₁₀ = -20 :=
by
  sorry

end sum_of_coefficients_l3137_313705


namespace solution_set_g_range_of_a_l3137_313712

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + |2*x + 5|
def g (x : ℝ) : ℝ := |x - 1| - |2*x|

-- Theorem for part I
theorem solution_set_g (x : ℝ) : g x > -4 ↔ -5 < x ∧ x < -3 := by sorry

-- Theorem for part II
theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, f a x₁ = g x₂) → -6 ≤ a ∧ a ≤ -4 := by sorry

end solution_set_g_range_of_a_l3137_313712


namespace subtract_negative_two_l3137_313751

theorem subtract_negative_two : 0 - (-2) = 2 := by sorry

end subtract_negative_two_l3137_313751


namespace unique_solution_star_l3137_313777

/-- The star operation defined on real numbers -/
def star (x y : ℝ) : ℝ := 3 * x - 2 * y + x^2 * y

/-- Theorem stating that there's exactly one solution to 2 ⋆ y = 9 -/
theorem unique_solution_star :
  ∃! y : ℝ, star 2 y = 9 := by sorry

end unique_solution_star_l3137_313777


namespace patanjali_walk_l3137_313717

/-- Represents the walking scenario of Patanjali over three days --/
structure WalkingScenario where
  hours_day1 : ℕ
  speed_day1 : ℕ
  total_distance : ℕ

/-- Calculates the distance walked on the first day given a WalkingScenario --/
def distance_day1 (scenario : WalkingScenario) : ℕ :=
  scenario.hours_day1 * scenario.speed_day1

/-- Calculates the total distance walked over three days given a WalkingScenario --/
def total_distance (scenario : WalkingScenario) : ℕ :=
  (distance_day1 scenario) + 
  (scenario.hours_day1 - 1) * (scenario.speed_day1 + 1) + 
  scenario.hours_day1 * (scenario.speed_day1 + 1)

/-- Theorem stating that given the conditions, the distance walked on the first day is 18 miles --/
theorem patanjali_walk (scenario : WalkingScenario) 
  (h1 : scenario.speed_day1 = 3) 
  (h2 : total_distance scenario = 62) : 
  distance_day1 scenario = 18 := by
  sorry

#eval distance_day1 { hours_day1 := 6, speed_day1 := 3, total_distance := 62 }

end patanjali_walk_l3137_313717


namespace point_satisfies_inequality_l3137_313708

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The inequality function -/
def inequality (p : Point) : ℝ :=
  (p.x + 2*p.y - 1) * (p.x - p.y + 3)

/-- Theorem stating that the point (0,2) satisfies the inequality -/
theorem point_satisfies_inequality : 
  let p : Point := ⟨0, 2⟩
  inequality p > 0 := by
  sorry


end point_satisfies_inequality_l3137_313708


namespace quiz_percentage_correct_l3137_313770

theorem quiz_percentage_correct (x : ℕ) : 
  let total_questions : ℕ := 7 * x
  let missed_questions : ℕ := 2 * x
  let correct_questions : ℕ := total_questions - missed_questions
  let percentage_correct : ℚ := (correct_questions : ℚ) / (total_questions : ℚ) * 100
  percentage_correct = 500 / 7 :=
by sorry

end quiz_percentage_correct_l3137_313770


namespace bees_flew_in_l3137_313760

theorem bees_flew_in (initial_bees final_bees : ℕ) (h1 : initial_bees = 16) (h2 : final_bees = 23) :
  final_bees - initial_bees = 7 := by
sorry

end bees_flew_in_l3137_313760


namespace simplify_expression_l3137_313771

theorem simplify_expression : 
  ∃ x : ℚ, (3/4 * 60) - (8/5 * 60) + x = 12 ∧ x = 63 := by sorry

end simplify_expression_l3137_313771


namespace largest_prime_divisor_to_test_l3137_313741

theorem largest_prime_divisor_to_test (n : ℕ) : 
  1000 ≤ n ∧ n ≤ 1100 → 
  (∀ p : ℕ, p.Prime → p ≤ 31 → n % p ≠ 0) → 
  n.Prime ∨ n = 1 :=
sorry

end largest_prime_divisor_to_test_l3137_313741


namespace handshake_count_l3137_313722

theorem handshake_count (n : ℕ) (h : n = 8) : n * (n - 1) / 2 = 28 := by
  sorry

end handshake_count_l3137_313722


namespace yuko_wins_l3137_313787

/-- The minimum value of Yuko's last die to be ahead of Yuri -/
def min_value_to_win (yuri_dice : Fin 3 → Nat) (yuko_dice : Fin 2 → Nat) : Nat :=
  (yuri_dice 0 + yuri_dice 1 + yuri_dice 2) - (yuko_dice 0 + yuko_dice 1) + 1

theorem yuko_wins (yuri_dice : Fin 3 → Nat) (yuko_dice : Fin 2 → Nat) :
  yuri_dice 0 = 2 → yuri_dice 1 = 4 → yuri_dice 2 = 5 →
  yuko_dice 0 = 1 → yuko_dice 1 = 5 →
  min_value_to_win yuri_dice yuko_dice = 6 := by
  sorry

#eval min_value_to_win (![2, 4, 5]) (![1, 5])

end yuko_wins_l3137_313787


namespace simultaneous_strike_l3137_313785

def cymbal_interval : ℕ := 7
def triangle_interval : ℕ := 2

theorem simultaneous_strike :
  ∃ (n : ℕ), n > 0 ∧ n % cymbal_interval = 0 ∧ n % triangle_interval = 0 ∧
  ∀ (m : ℕ), 0 < m ∧ m < n → (m % cymbal_interval ≠ 0 ∨ m % triangle_interval ≠ 0) :=
by sorry

end simultaneous_strike_l3137_313785


namespace range_of_m_l3137_313786

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 1/x + 4/y = 1) :
  (∀ x y, x > 0 → y > 0 → 1/x + 4/y = 1 → x + y > m^2 + 8*m) ↔ -9 < m ∧ m < 1 := by
  sorry

end range_of_m_l3137_313786


namespace sum_after_transformation_l3137_313759

theorem sum_after_transformation (a b S : ℝ) (h : a + b = S) :
  2 * ((a + 3) + (b + 3)) = 2 * S + 12 := by
  sorry

end sum_after_transformation_l3137_313759


namespace number_divided_by_004_equals_25_l3137_313736

theorem number_divided_by_004_equals_25 : ∃ x : ℝ, x / 0.04 = 25 ∧ x = 1 := by
  sorry

end number_divided_by_004_equals_25_l3137_313736


namespace min_sum_quadratic_coeff_l3137_313737

theorem min_sum_quadratic_coeff (a b c : ℕ+) 
  (root_condition : ∃ x₁ x₂ : ℝ, (a:ℝ) * x₁^2 + (b:ℝ) * x₁ + (c:ℝ) = 0 ∧ 
                                (a:ℝ) * x₂^2 + (b:ℝ) * x₂ + (c:ℝ) = 0 ∧
                                x₁ ≠ x₂ ∧ 
                                abs x₁ < (1:ℝ)/3 ∧ 
                                abs x₂ < (1:ℝ)/3) : 
  (a:ℕ) + b + c ≥ 25 := by
  sorry

end min_sum_quadratic_coeff_l3137_313737


namespace average_weight_increase_l3137_313761

theorem average_weight_increase (W : ℝ) : 
  let original_average := (W + 45) / 10
  let new_average := (W + 75) / 10
  new_average - original_average = 3 :=
by
  sorry

end average_weight_increase_l3137_313761


namespace states_joined_fraction_l3137_313754

theorem states_joined_fraction :
  let total_states : ℕ := 30
  let states_1780_to_1789 : ℕ := 12
  let states_1790_to_1799 : ℕ := 5
  let states_1780_to_1799 : ℕ := states_1780_to_1789 + states_1790_to_1799
  (states_1780_to_1799 : ℚ) / total_states = 17 / 30 := by
  sorry

end states_joined_fraction_l3137_313754


namespace smallest_linear_combination_l3137_313703

theorem smallest_linear_combination : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∃ (m n p : ℤ), k = 2010 * m + 44550 * n + 100 * p) ∧
  (∀ (j : ℕ), j > 0 → (∃ (x y z : ℤ), j = 2010 * x + 44550 * y + 100 * z) → j ≥ k) ∧
  k = 10 := by
  sorry

end smallest_linear_combination_l3137_313703


namespace f_properties_l3137_313728

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log (1/2)
  else if x = 0 then 0
  else Real.log (-x) / Real.log (1/2)

-- State the theorem
theorem f_properties :
  (∀ x, f x = f (-x)) ∧  -- f is even
  f 0 = 0 ∧             -- f(0) = 0
  (∀ x > 0, f x = Real.log x / Real.log (1/2)) →  -- f(x) = log₍₁/₂₎(x) for x > 0
  f (-4) = -2 ∧         -- Part 1: f(-4) = -2
  (∀ x, f x = if x > 0 then Real.log x / Real.log (1/2)
              else if x = 0 then 0
              else Real.log (-x) / Real.log (1/2))  -- Part 2: Analytic expression of f
  := by sorry

end f_properties_l3137_313728


namespace sqrt_square_eq_identity_power_zero_eq_one_l3137_313744

-- Option C
theorem sqrt_square_eq_identity (x : ℝ) (h : x ≥ -2) :
  (Real.sqrt (x + 2))^2 = x + 2 := by sorry

-- Option D
theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) :
  x^0 = 1 := by sorry

end sqrt_square_eq_identity_power_zero_eq_one_l3137_313744


namespace char_coeff_pair_example_char_poly_sum_example_char_poly_diff_example_l3137_313763

-- Define the characteristic coefficient pair
def char_coeff_pair (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

-- Define the characteristic polynomial
def char_poly (p : ℝ × ℝ × ℝ) (x : ℝ) : ℝ :=
  let (a, b, c) := p
  a * x^2 + b * x + c

theorem char_coeff_pair_example : char_coeff_pair 3 4 1 = (3, 4, 1) := by sorry

theorem char_poly_sum_example : 
  char_poly (2, 1, 2) x + char_poly (2, -1, 2) x = 4 * x^2 + 4 := by sorry

theorem char_poly_diff_example (m n : ℝ) : 
  (char_poly (1, 2, m) x - char_poly (2, n, 3) x = -x^2 + x - 1) → m * n = 2 := by sorry

end char_coeff_pair_example_char_poly_sum_example_char_poly_diff_example_l3137_313763


namespace multiplication_of_decimals_l3137_313772

theorem multiplication_of_decimals : 3.6 * 0.3 = 1.08 := by
  sorry

end multiplication_of_decimals_l3137_313772


namespace fraction_sum_equals_decimal_l3137_313743

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 20 - 5 / 200 + 7 / 2000 = 0.1285 := by sorry

end fraction_sum_equals_decimal_l3137_313743


namespace library_books_count_l3137_313720

/-- Given the conditions of the library bookshelves, calculate the total number of books -/
theorem library_books_count (num_shelves : ℕ) (floors_per_shelf : ℕ) (books_after_removal : ℕ) : 
  num_shelves = 28 → 
  floors_per_shelf = 6 → 
  books_after_removal = 20 → 
  (num_shelves * floors_per_shelf * (books_after_removal + 2) = 3696) :=
by
  sorry

#check library_books_count

end library_books_count_l3137_313720


namespace mike_picked_52_peaches_l3137_313799

/-- The number of peaches Mike picked at the orchard -/
def peaches_picked (initial : ℕ) (total : ℕ) : ℕ :=
  total - initial

/-- Theorem stating that Mike picked 52 peaches at the orchard -/
theorem mike_picked_52_peaches (initial : ℕ) (total : ℕ) 
  (h1 : initial = 34) 
  (h2 : total = 86) : 
  peaches_picked initial total = 52 := by
  sorry

end mike_picked_52_peaches_l3137_313799


namespace equation_represents_two_intersecting_lines_l3137_313795

-- Define the equation
def equation (x y : ℝ) : Prop :=
  x^3 * (x + y - 2) = y^3 * (x + y - 2)

-- Define what it means for two lines to intersect
def intersecting_lines (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = g x

-- Theorem statement
theorem equation_represents_two_intersecting_lines :
  ∃ (f g : ℝ → ℝ), 
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) ∧
    intersecting_lines f g :=
sorry

end equation_represents_two_intersecting_lines_l3137_313795


namespace unique_solution_exists_l3137_313757

theorem unique_solution_exists : ∃! (x : ℝ), x > 0 ∧ (Int.floor x) * x + x^2 = 93 ∧ ∀ (ε : ℝ), ε > 0 → |x - 7.10| < ε := by
  sorry

end unique_solution_exists_l3137_313757


namespace apollonius_circle_minimum_l3137_313750

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (2, 1)

-- Define the distance ratio condition
def distance_ratio (P : ℝ × ℝ) : Prop :=
  (P.1 + 1)^2 + P.2^2 = 2 * ((P.1 - 2)^2 + (P.2 - 1)^2)

-- Define the symmetry line
def symmetry_line (m n : ℝ) (P : ℝ × ℝ) : Prop :=
  m * P.1 + n * P.2 = 2

-- Main theorem
theorem apollonius_circle_minimum (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ P : ℝ × ℝ, distance_ratio P → ∃ P', distance_ratio P' ∧ symmetry_line m n ((P.1 + P'.1)/2, (P.2 + P'.2)/2)) →
  (2/m + 5/n) ≥ 20 :=
sorry

end apollonius_circle_minimum_l3137_313750


namespace river_joe_collection_l3137_313733

/-- Represents the total money collected by River Joe's Seafood Diner --/
def total_money_collected (catfish_price popcorn_shrimp_price : ℚ) 
  (total_orders popcorn_shrimp_orders : ℕ) : ℚ :=
  let catfish_orders := total_orders - popcorn_shrimp_orders
  catfish_price * catfish_orders + popcorn_shrimp_price * popcorn_shrimp_orders

/-- Proves that River Joe collected $133.50 given the specified conditions --/
theorem river_joe_collection : 
  total_money_collected 6 3.5 26 9 = 133.5 := by
  sorry

#eval total_money_collected 6 3.5 26 9

end river_joe_collection_l3137_313733


namespace line_through_point_unique_l3137_313789

/-- A line passing through a point -/
def line_passes_through_point (k : ℝ) : Prop :=
  2 * k * (-1/2) - 3 = -7 * 3

/-- The value of k that satisfies the line equation -/
def k_value : ℝ := 18

/-- Theorem: k_value is the unique real number that satisfies the line equation -/
theorem line_through_point_unique : 
  line_passes_through_point k_value ∧ 
  ∀ k : ℝ, line_passes_through_point k → k = k_value :=
sorry

end line_through_point_unique_l3137_313789


namespace special_remainder_property_l3137_313732

theorem special_remainder_property (n : ℕ) : 
  (∀ q : ℕ, q > 0 → n % q^2 < q^(q^2) / 2) ↔ (n = 1 ∨ n = 4) :=
by sorry

end special_remainder_property_l3137_313732


namespace coin_bag_total_amount_l3137_313731

theorem coin_bag_total_amount :
  ∃ (x : ℕ),
    let one_cent := x
    let ten_cent := 2 * x
    let twenty_five_cent := 3 * (2 * x)
    let total := one_cent * 1 + ten_cent * 10 + twenty_five_cent * 25
    total = 342 := by
  sorry

end coin_bag_total_amount_l3137_313731


namespace log_3_base_5_l3137_313778

theorem log_3_base_5 (a : ℝ) (h : Real.log 45 / Real.log 5 = a) :
  Real.log 3 / Real.log 5 = (a - 1) / 2 := by
  sorry

end log_3_base_5_l3137_313778


namespace cone_base_circumference_l3137_313764

/-- The circumference of the base of a right circular cone with volume 18π cubic centimeters and height 6 cm is 6π cm. -/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 18 * Real.pi ∧ h = 6 ∧ V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 6 * Real.pi :=
by sorry

end cone_base_circumference_l3137_313764


namespace arcsin_one_equals_pi_half_l3137_313746

theorem arcsin_one_equals_pi_half : Real.arcsin 1 = π / 2 := by
  sorry

end arcsin_one_equals_pi_half_l3137_313746


namespace watch_cost_l3137_313710

theorem watch_cost (watch_cost strap_cost : ℝ) 
  (total_cost : watch_cost + strap_cost = 120)
  (cost_difference : watch_cost = strap_cost + 100) :
  watch_cost = 110 := by
sorry

end watch_cost_l3137_313710


namespace jellybean_probability_l3137_313749

def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 2
def yellow_jellybeans : ℕ := 5
def picked_jellybeans : ℕ := 4

theorem jellybean_probability : 
  (Nat.choose red_jellybeans 3 * Nat.choose (total_jellybeans - red_jellybeans) 1) / 
  Nat.choose total_jellybeans picked_jellybeans = 14 / 99 := by
  sorry

end jellybean_probability_l3137_313749


namespace h_negative_two_equals_eleven_l3137_313753

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 + 1
def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_negative_two_equals_eleven : h (-2) = 11 := by
  sorry

end h_negative_two_equals_eleven_l3137_313753


namespace power_equality_implies_exponent_l3137_313700

theorem power_equality_implies_exponent (q : ℕ) : 27^8 = 9^q → q = 12 := by
  sorry

end power_equality_implies_exponent_l3137_313700


namespace geometric_sequence_common_ratio_sum_l3137_313714

theorem geometric_sequence_common_ratio_sum (k a₂ a₃ b₂ b₃ : ℝ) (p r : ℝ) 
  (hk : k ≠ 0)
  (hp : p ≠ 1)
  (hr : r ≠ 1)
  (hp_neq_r : p ≠ r)
  (ha₂ : a₂ = k * p)
  (ha₃ : a₃ = k * p^2)
  (hb₂ : b₂ = k * r)
  (hb₃ : b₃ = k * r^2)
  (h_eq : a₃^2 - b₃^2 = 3 * (a₂^2 - b₂^2)) :
  p^2 + r^2 = 3 := by
sorry

end geometric_sequence_common_ratio_sum_l3137_313714


namespace bathroom_visits_time_calculation_l3137_313724

/-- Given that it takes 20 minutes for 8 bathroom visits, 
    prove that 6 visits will take 15 minutes. -/
theorem bathroom_visits_time_calculation 
  (total_time : ℝ) 
  (total_visits : ℕ) 
  (target_visits : ℕ) 
  (h1 : total_time = 20) 
  (h2 : total_visits = 8) 
  (h3 : target_visits = 6) : 
  (total_time / total_visits) * target_visits = 15 := by
  sorry

end bathroom_visits_time_calculation_l3137_313724


namespace mean_equality_implies_z_value_l3137_313729

theorem mean_equality_implies_z_value :
  let mean1 := (8 + 12 + 24) / 3
  let mean2 := (16 + z) / 2
  mean1 = mean2 → z = 40 / 3 := by
  sorry

end mean_equality_implies_z_value_l3137_313729


namespace smallest_three_digit_square_append_l3137_313740

theorem smallest_three_digit_square_append : ∃ (n : ℕ), 
  (n = 183) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < n → ¬(∃ k : ℕ, 1000 * m + (m + 1) = k * k)) ∧
  (∃ k : ℕ, 1000 * n + (n + 1) = k * k) :=
by sorry

end smallest_three_digit_square_append_l3137_313740


namespace minimal_solution_l3137_313704

def is_solution (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (1 : ℚ) / A + (1 : ℚ) / B + (1 : ℚ) / C = (1 : ℚ) / 6 ∧
  6 ∣ A ∧ 6 ∣ B ∧ 6 ∣ C

theorem minimal_solution :
  ∀ A B C : ℕ, is_solution A B C →
  A + B + C ≥ 12 + 18 + 36 ∧
  is_solution 12 18 36 :=
sorry

end minimal_solution_l3137_313704


namespace crow_speed_l3137_313755

/-- Crow's flight speed calculation -/
theorem crow_speed (distance_to_ditch : ℝ) (num_trips : ℕ) (time_hours : ℝ) :
  distance_to_ditch = 400 →
  num_trips = 15 →
  time_hours = 1.5 →
  (2 * distance_to_ditch * num_trips) / (1000 * time_hours) = 8 := by
  sorry

end crow_speed_l3137_313755


namespace profit_starts_third_year_option1_more_cost_effective_l3137_313791

/-- Represents the financial state of a fishing company -/
structure FishingCompany where
  initialCost : ℕ
  firstYearExpenses : ℕ
  annualExpenseIncrease : ℕ
  annualIncome : ℕ

/-- Calculates the year when the company starts to make a profit -/
def yearOfFirstProfit (company : FishingCompany) : ℕ :=
  sorry

/-- Calculates the more cost-effective option between two selling strategies -/
def moreCostEffectiveOption (company : FishingCompany) (option1Value : ℕ) (option2Value : ℕ) : Bool :=
  sorry

/-- Theorem stating that the company starts to make a profit in the third year -/
theorem profit_starts_third_year (company : FishingCompany) 
  (h1 : company.initialCost = 980000)
  (h2 : company.firstYearExpenses = 120000)
  (h3 : company.annualExpenseIncrease = 40000)
  (h4 : company.annualIncome = 500000) :
  yearOfFirstProfit company = 3 :=
sorry

/-- Theorem stating that the first option (selling for 260,000) is more cost-effective -/
theorem option1_more_cost_effective (company : FishingCompany)
  (h1 : company.initialCost = 980000)
  (h2 : company.firstYearExpenses = 120000)
  (h3 : company.annualExpenseIncrease = 40000)
  (h4 : company.annualIncome = 500000) :
  moreCostEffectiveOption company 260000 80000 = true :=
sorry

end profit_starts_third_year_option1_more_cost_effective_l3137_313791


namespace bc_fraction_of_ad_l3137_313781

-- Define the points
variable (A B C D : ℝ)

-- Define the conditions
axiom on_line_segment : B ≤ A ∧ B ≤ D ∧ C ≤ A ∧ C ≤ D

-- Define the length relationships
axiom length_AB : A - B = 3 * (D - B)
axiom length_AC : A - C = 7 * (D - C)

-- Theorem to prove
theorem bc_fraction_of_ad : (C - B) = (1/8) * (A - D) := by sorry

end bc_fraction_of_ad_l3137_313781


namespace consecutive_integers_sum_l3137_313721

theorem consecutive_integers_sum (x : ℤ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l3137_313721


namespace triangle_ABC_properties_l3137_313768

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  (2 * c - b) / Real.cos B = a / Real.cos A ∧
  a = Real.sqrt 7 ∧
  2 * b = 3 * c

theorem triangle_ABC_properties {a b c A B C : ℝ} 
  (h : triangle_ABC a b c A B C) :
  A = π / 3 ∧ 
  (1/2 : ℝ) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by sorry

end triangle_ABC_properties_l3137_313768


namespace common_number_in_overlapping_groups_l3137_313747

theorem common_number_in_overlapping_groups (list : List ℝ) : 
  list.length = 9 →
  (list.take 5).sum / 5 = 7 →
  (list.drop 4).sum / 5 = 10 →
  list.sum / 9 = 74 / 9 →
  ∃ x ∈ list, x ∈ list.take 5 ∧ x ∈ list.drop 4 ∧ x = 11 :=
by sorry

end common_number_in_overlapping_groups_l3137_313747


namespace only_set_D_forms_triangle_l3137_313784

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- The given sets of line segments -/
def set_A : Vector ℝ 3 := ⟨[5, 11, 6], by simp⟩
def set_B : Vector ℝ 3 := ⟨[8, 8, 16], by simp⟩
def set_C : Vector ℝ 3 := ⟨[10, 5, 4], by simp⟩
def set_D : Vector ℝ 3 := ⟨[6, 9, 14], by simp⟩

/-- Theorem: Among the given sets, only set D can form a triangle -/
theorem only_set_D_forms_triangle :
  ¬(can_form_triangle set_A[0] set_A[1] set_A[2]) ∧
  ¬(can_form_triangle set_B[0] set_B[1] set_B[2]) ∧
  ¬(can_form_triangle set_C[0] set_C[1] set_C[2]) ∧
  can_form_triangle set_D[0] set_D[1] set_D[2] :=
by sorry

end only_set_D_forms_triangle_l3137_313784
