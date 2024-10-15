import Mathlib

namespace NUMINAMATH_CALUDE_power_seven_mod_eight_l1268_126827

theorem power_seven_mod_eight : 7^51 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_eight_l1268_126827


namespace NUMINAMATH_CALUDE_intersection_symmetric_implies_p_range_l1268_126809

/-- The line equation: x = ky - 1 -/
def line_equation (k : ℝ) (x y : ℝ) : Prop := x = k * y - 1

/-- The circle equation: x² + y² + kx + my + 2p = 0 -/
def circle_equation (k m p : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + k*x + m*y + 2*p = 0

/-- Two points (x₁, y₁) and (x₂, y₂) are symmetric about y = x -/
def symmetric_about_y_eq_x (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = y₂ ∧ y₁ = x₂

theorem intersection_symmetric_implies_p_range
  (k m p : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    line_equation k x₁ y₁ ∧
    line_equation k x₂ y₂ ∧
    circle_equation k m p x₁ y₁ ∧
    circle_equation k m p x₂ y₂ ∧
    symmetric_about_y_eq_x x₁ y₁ x₂ y₂) →
  p < -3/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_symmetric_implies_p_range_l1268_126809


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1268_126823

def M : Set ℤ := {0, 1, 2}

def A : Set ℤ := {y | ∃ x ∈ M, y = 2 * x}

def B : Set ℤ := {y | ∃ x ∈ M, y = 2 * x - 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1268_126823


namespace NUMINAMATH_CALUDE_telescope_purchase_problem_l1268_126891

theorem telescope_purchase_problem (joan_price karl_price : ℝ) 
  (h1 : joan_price + karl_price = 400)
  (h2 : 2 * joan_price = karl_price + 74) :
  joan_price = 158 := by
  sorry

end NUMINAMATH_CALUDE_telescope_purchase_problem_l1268_126891


namespace NUMINAMATH_CALUDE_min_sum_a_b_is_six_l1268_126821

/-- Given that the roots of x^2 + ax + 2b = 0 and x^2 + 2bx + a = 0 are both real,
    and a, b > 0, the minimum value of a + b is 6. -/
theorem min_sum_a_b_is_six (a b : ℝ) 
    (h1 : a > 0)
    (h2 : b > 0)
    (h3 : a^2 - 8*b ≥ 0)  -- Condition for real roots of x^2 + ax + 2b = 0
    (h4 : 4*b^2 - 4*a ≥ 0)  -- Condition for real roots of x^2 + 2bx + a = 0
    : ∀ a' b' : ℝ, a' > 0 → b' > 0 → a'^2 - 8*b' ≥ 0 → 4*b'^2 - 4*a' ≥ 0 → a + b ≤ a' + b' :=
by sorry

end NUMINAMATH_CALUDE_min_sum_a_b_is_six_l1268_126821


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_l1268_126814

theorem square_area_from_rectangle (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ) (rectangle_area : ℝ) :
  rectangle_length = (2 / 5) * circle_radius →
  rectangle_breadth = 10 →
  rectangle_area = 160 →
  rectangle_area = rectangle_length * rectangle_breadth →
  (circle_radius ^ 2 : ℝ) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_l1268_126814


namespace NUMINAMATH_CALUDE_max_value_condition_l1268_126873

/-- The expression 2005 - (x + y)^2 takes its maximum value when x = -y -/
theorem max_value_condition (x y : ℝ) : 
  (∀ a b : ℝ, 2005 - (x + y)^2 ≥ 2005 - (a + b)^2) → x = -y := by
sorry

end NUMINAMATH_CALUDE_max_value_condition_l1268_126873


namespace NUMINAMATH_CALUDE_movie_theater_seating_l1268_126804

/-- The number of ways to seat people in a row of seats with constraints -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  -- Define the function to calculate the number of seating arrangements
  sorry

/-- Theorem stating the number of seating arrangements for the specific problem -/
theorem movie_theater_seating : seating_arrangements 9 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_seating_l1268_126804


namespace NUMINAMATH_CALUDE_cone_height_from_sector_l1268_126816

/-- Given a sector with radius 7 cm and area 21π cm², when used to form the lateral surface of a cone, 
    the height of the cone is 2√10 cm. -/
theorem cone_height_from_sector (r : ℝ) (area : ℝ) (h : ℝ) : 
  r = 7 → 
  area = 21 * Real.pi → 
  area = (1/2) * (2 * Real.pi) * 3 * r → 
  h = Real.sqrt (r^2 - 3^2) → 
  h = 2 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_cone_height_from_sector_l1268_126816


namespace NUMINAMATH_CALUDE_derivative_ln_plus_x_l1268_126893

open Real

theorem derivative_ln_plus_x (x : ℝ) (h : x > 0) : 
  deriv (fun x => log x + x) x = (x + 1) / x := by
sorry

end NUMINAMATH_CALUDE_derivative_ln_plus_x_l1268_126893


namespace NUMINAMATH_CALUDE_circle_plus_four_two_l1268_126836

/-- Definition of the ⊕ operation for real numbers -/
def circle_plus (a b : ℝ) : ℝ := 2 * a + 5 * b

/-- Theorem stating that 4 ⊕ 2 = 18 -/
theorem circle_plus_four_two : circle_plus 4 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_four_two_l1268_126836


namespace NUMINAMATH_CALUDE_computer_software_price_sum_l1268_126802

theorem computer_software_price_sum : 
  ∀ (b a : ℝ),
  (b + 0.3 * b = 351) →
  (a + 0.05 * a = 420) →
  2 * b + 2 * a = 1340 :=
by
  sorry

end NUMINAMATH_CALUDE_computer_software_price_sum_l1268_126802


namespace NUMINAMATH_CALUDE_brown_family_seating_l1268_126881

/-- The number of ways to seat n children in a circle. -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to seat b boys and g girls in a circle
    such that at least two boys are next to each other. -/
def boysNextToEachOther (b g : ℕ) : ℕ :=
  if b > g + 1 then circularArrangements (b + g) else 0

theorem brown_family_seating :
  boysNextToEachOther 5 3 = 5040 := by sorry

end NUMINAMATH_CALUDE_brown_family_seating_l1268_126881


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1268_126817

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + x * y = 3) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + b + a * b = 3 → x + y ≤ a + b :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1268_126817


namespace NUMINAMATH_CALUDE_trivia_team_score_l1268_126852

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_score : ℕ) :
  total_members = 7 →
  absent_members = 2 →
  total_score = 20 →
  (total_score / (total_members - absent_members) : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_score_l1268_126852


namespace NUMINAMATH_CALUDE_cost_price_is_65_l1268_126819

/-- Given a cloth sale scenario, calculate the cost price per metre. -/
def cost_price_per_metre (total_metres : ℕ) (total_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  total_price / total_metres + loss_per_metre

/-- Theorem stating that the cost price per metre is 65 given the problem conditions. -/
theorem cost_price_is_65 :
  cost_price_per_metre 300 18000 5 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_65_l1268_126819


namespace NUMINAMATH_CALUDE_roots_properties_l1268_126847

theorem roots_properties (z₁ z₂ : ℂ) (h : x^2 + x + 1 = 0 ↔ x = z₁ ∨ x = z₂) :
  z₁ * z₂ = 1 ∧ z₁^3 = 1 ∧ z₂^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_properties_l1268_126847


namespace NUMINAMATH_CALUDE_compute_expression_l1268_126870

theorem compute_expression : 7^2 - 2*(5) + 2^3 = 47 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1268_126870


namespace NUMINAMATH_CALUDE_geometric_sequence_linear_system_l1268_126828

theorem geometric_sequence_linear_system (a : ℕ → ℝ) (q : ℝ) (h : q ≠ 0) :
  (∀ n : ℕ, a (n + 1) = q * a n) →
  (∃ x y : ℝ, a 1 * x + a 3 * y = 2 ∧ a 2 * x + a 4 * y = 1) ↔ q = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_linear_system_l1268_126828


namespace NUMINAMATH_CALUDE_triangle_properties_l1268_126896

theorem triangle_properties (a b c : ℝ) (h_ratio : (a, b, c) = (5 * 2, 12 * 2, 13 * 2)) 
  (h_perimeter : a + b + c = 60) : 
  (a^2 + b^2 = c^2) ∧ (a * b / 2 > 100) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1268_126896


namespace NUMINAMATH_CALUDE_negation_equivalence_l1268_126875

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀ ≤ 0 ∧ x₀^2 ≥ 0) ↔ (∀ x : ℝ, x ≤ 0 → x^2 < 0) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1268_126875


namespace NUMINAMATH_CALUDE_average_of_combined_sets_l1268_126851

theorem average_of_combined_sets :
  ∀ (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ),
    n₁ = 30 →
    n₂ = 20 →
    avg₁ = 20 →
    avg₂ = 30 →
    (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = 24 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_sets_l1268_126851


namespace NUMINAMATH_CALUDE_price_difference_is_1090_l1268_126825

/-- The difference in cents between the TV advertiser price and the in-store price for a microwave --/
def price_difference : ℚ :=
  let in_store_price : ℚ := 149.95
  let tv_payment : ℚ := 27.99
  let shipping_fee : ℚ := 14.95
  let warranty_fee : ℚ := 5.95
  let tv_price : ℚ := 5 * tv_payment + shipping_fee + warranty_fee
  (tv_price - in_store_price) * 100

/-- The price difference is 1090 cents --/
theorem price_difference_is_1090 : 
  price_difference = 1090 := by sorry

end NUMINAMATH_CALUDE_price_difference_is_1090_l1268_126825


namespace NUMINAMATH_CALUDE_division_problem_l1268_126833

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 15 ∧ quotient = 4 ∧ remainder = 3 ∧ 
  dividend = divisor * quotient + remainder → 
  divisor = 3 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1268_126833


namespace NUMINAMATH_CALUDE_maintenance_building_length_l1268_126872

/-- The length of the maintenance building on a square playground -/
theorem maintenance_building_length : 
  ∀ (playground_side : ℝ) (building_width : ℝ) (uncovered_area : ℝ),
  playground_side = 12 →
  building_width = 5 →
  uncovered_area = 104 →
  ∃ (building_length : ℝ),
    building_length = 8 ∧
    building_length * building_width = playground_side^2 - uncovered_area :=
by sorry

end NUMINAMATH_CALUDE_maintenance_building_length_l1268_126872


namespace NUMINAMATH_CALUDE_strawberry_count_l1268_126812

/-- Calculates the total number of strawberries after picking more -/
def total_strawberries (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: The total number of strawberries is the sum of initial and additional strawberries -/
theorem strawberry_count (initial additional : ℕ) :
  total_strawberries initial additional = initial + additional := by
  sorry

end NUMINAMATH_CALUDE_strawberry_count_l1268_126812


namespace NUMINAMATH_CALUDE_fraction_to_sofia_is_one_twelfth_l1268_126885

/-- Represents the initial egg distribution and sharing problem --/
structure EggDistribution where
  mia_eggs : ℕ
  sofia_eggs : ℕ
  pablo_eggs : ℕ
  lucas_eggs : ℕ
  (sofia_eggs_def : sofia_eggs = 3 * mia_eggs)
  (pablo_eggs_def : pablo_eggs = 4 * sofia_eggs)
  (lucas_eggs_def : lucas_eggs = 0)

/-- Calculates the fraction of Pablo's eggs given to Sofia --/
def fraction_to_sofia (d : EggDistribution) : ℚ :=
  let total_eggs := d.mia_eggs + d.sofia_eggs + d.pablo_eggs + d.lucas_eggs
  let equal_share := total_eggs / 4
  let sofia_needs := equal_share - d.sofia_eggs
  sofia_needs / d.pablo_eggs

/-- Theorem stating that the fraction of Pablo's eggs given to Sofia is 1/12 --/
theorem fraction_to_sofia_is_one_twelfth (d : EggDistribution) :
  fraction_to_sofia d = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_sofia_is_one_twelfth_l1268_126885


namespace NUMINAMATH_CALUDE_prob_one_male_correct_prob_at_least_one_female_correct_l1268_126846

/-- Represents the number of female students in the group -/
def num_females : ℕ := 2

/-- Represents the number of male students in the group -/
def num_males : ℕ := 3

/-- Represents the total number of students in the group -/
def total_students : ℕ := num_females + num_males

/-- Represents the number of students to be selected -/
def num_selected : ℕ := 2

/-- Calculates the probability of selecting exactly one male student -/
def prob_one_male : ℚ := 3 / 5

/-- Calculates the probability of selecting at least one female student -/
def prob_at_least_one_female : ℚ := 7 / 10

/-- Proves that the probability of selecting exactly one male student is 3/5 -/
theorem prob_one_male_correct : 
  prob_one_male = (num_females * num_males : ℚ) / (total_students.choose num_selected : ℚ) := by
  sorry

/-- Proves that the probability of selecting at least one female student is 7/10 -/
theorem prob_at_least_one_female_correct :
  prob_at_least_one_female = 1 - ((num_males.choose num_selected : ℚ) / (total_students.choose num_selected : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_prob_one_male_correct_prob_at_least_one_female_correct_l1268_126846


namespace NUMINAMATH_CALUDE_even_increasing_function_properties_l1268_126871

/-- A function f: ℝ → ℝ that is even and increasing on (-∞, 0) -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, x < y ∧ y ≤ 0 → f x < f y)

/-- Theorem stating properties of an even increasing function -/
theorem even_increasing_function_properties (f : ℝ → ℝ) 
  (hf : EvenIncreasingFunction f) : 
  (∀ x, f (-x) - f x = 0) ∧ 
  (∀ x y, 0 < x ∧ x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_even_increasing_function_properties_l1268_126871


namespace NUMINAMATH_CALUDE_solutions_count_l1268_126840

/-- The number of solutions to the equation √(x+3) = ax + 2 depends on the value of a -/
theorem solutions_count (a : ℝ) :
  (∃! x, Real.sqrt (x + 3) = a * x + 2) ∨
  (¬ ∃ x, Real.sqrt (x + 3) = a * x + 2) ∨
  (∃ x y, x ≠ y ∧ Real.sqrt (x + 3) = a * x + 2 ∧ Real.sqrt (y + 3) = a * y + 2) :=
  by sorry

end NUMINAMATH_CALUDE_solutions_count_l1268_126840


namespace NUMINAMATH_CALUDE_only_negative_number_l1268_126818

theorem only_negative_number (a b c d : ℝ) : 
  a = 2023 ∧ b = -2023 ∧ c = 1/2023 ∧ d = 0 →
  (b < 0 ∧ a ≥ 0 ∧ c > 0 ∧ d = 0) := by
  sorry

end NUMINAMATH_CALUDE_only_negative_number_l1268_126818


namespace NUMINAMATH_CALUDE_circle_radius_proof_l1268_126880

theorem circle_radius_proof (r : ℝ) : 
  r > 0 → 
  (π * r^2 = 3 * (2 * π * r)) → 
  (π * r^2 + 2 * π * r = 100 * π) → 
  r = 12.5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l1268_126880


namespace NUMINAMATH_CALUDE_average_distance_scientific_notation_l1268_126801

-- Define the average distance between the Earth and the Sun
def average_distance : ℝ := 149600000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.496 * (10 ^ 8)

-- Theorem to prove the equivalence
theorem average_distance_scientific_notation : average_distance = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_average_distance_scientific_notation_l1268_126801


namespace NUMINAMATH_CALUDE_food_production_growth_rate_l1268_126866

theorem food_production_growth_rate 
  (initial_production : ℝ) 
  (a b x : ℝ) 
  (h1 : initial_production = 5000)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : x > 0)
  (h5 : initial_production * (1 + a) * (1 + b) = initial_production * (1 + x)^2) :
  x ≤ (a + b) / 2 := by
sorry

end NUMINAMATH_CALUDE_food_production_growth_rate_l1268_126866


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l1268_126838

-- Define the points
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (7, 0)
def D : ℝ × ℝ := (0, 4)

-- Define the length of AC
def AC : ℝ := 15

-- Theorem statement
theorem area_of_triangle_ABC :
  let A : ℝ × ℝ := (0, 4) -- We know A is on the y-axis at (0,4) because D is at (0,4)
  (1/2 : ℝ) * ‖(A.1 - B.1, A.2 - B.2)‖ * ‖(C.1 - B.1, C.2 - B.2)‖ = 2 * Real.sqrt 209 := by
sorry


end NUMINAMATH_CALUDE_area_of_triangle_ABC_l1268_126838


namespace NUMINAMATH_CALUDE_lcm_12_18_l1268_126855

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l1268_126855


namespace NUMINAMATH_CALUDE_simplify_expression_l1268_126863

theorem simplify_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x) * ((y^3 + 2) / y) - ((x^3 - 2) / y) * ((y^3 - 2) / x) = 4 * (x^2 / y + y^2 / x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1268_126863


namespace NUMINAMATH_CALUDE_sebastian_ticket_cost_l1268_126897

/-- The total cost of tickets for Sebastian and his parents -/
def total_cost (num_people : ℕ) (ticket_price : ℕ) (service_fee : ℕ) : ℕ :=
  num_people * ticket_price + service_fee

/-- Theorem stating that the total cost for Sebastian's tickets is $150 -/
theorem sebastian_ticket_cost :
  total_cost 3 44 18 = 150 := by
  sorry

end NUMINAMATH_CALUDE_sebastian_ticket_cost_l1268_126897


namespace NUMINAMATH_CALUDE_fish_estimation_l1268_126850

/-- The number of fish caught and marked on the first day -/
def marked_fish : ℕ := 30

/-- The number of fish caught on the second day -/
def second_catch : ℕ := 40

/-- The number of marked fish caught on the second day -/
def marked_recaught : ℕ := 2

/-- The estimated number of fish in the pond -/
def estimated_fish : ℕ := marked_fish * second_catch / marked_recaught

theorem fish_estimation :
  estimated_fish = 600 :=
sorry

end NUMINAMATH_CALUDE_fish_estimation_l1268_126850


namespace NUMINAMATH_CALUDE_sqrt_sum_equation_l1268_126878

theorem sqrt_sum_equation (x : ℝ) (h : x ≥ 1/2) :
  (∃ y ∈ Set.Icc (1/2 : ℝ) 1, x = y ↔ Real.sqrt (x + Real.sqrt (2*x - 1)) + Real.sqrt (x - Real.sqrt (2*x - 1)) = Real.sqrt 2) ∧
  (¬ ∃ y ≥ 1/2, Real.sqrt (y + Real.sqrt (2*y - 1)) + Real.sqrt (y - Real.sqrt (2*y - 1)) = 1) ∧
  (x = 3/2 ↔ Real.sqrt (x + Real.sqrt (2*x - 1)) + Real.sqrt (x - Real.sqrt (2*x - 1)) = 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equation_l1268_126878


namespace NUMINAMATH_CALUDE_solution_set_implies_a_range_l1268_126830

theorem solution_set_implies_a_range (a : ℝ) :
  (∀ x, (a - 3) * x > 1 ↔ x < 1 / (a - 3)) →
  a < 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_range_l1268_126830


namespace NUMINAMATH_CALUDE_odd_prime_equation_l1268_126857

theorem odd_prime_equation (p a b : ℕ) : 
  Prime p → 
  Odd p → 
  a > 0 → 
  b > 0 → 
  (p + 1)^a - p^b = 1 → 
  a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_equation_l1268_126857


namespace NUMINAMATH_CALUDE_count_divisible_by_11_is_18_l1268_126868

/-- The number obtained by writing the integers 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- The count of b_k divisible by 11 for 1 ≤ k ≤ 100 -/
def count_divisible_by_11 : ℕ := sorry

theorem count_divisible_by_11_is_18 : count_divisible_by_11 = 18 := by sorry

end NUMINAMATH_CALUDE_count_divisible_by_11_is_18_l1268_126868


namespace NUMINAMATH_CALUDE_kylie_daisies_left_l1268_126845

def daisies_problem (initial_daisies : ℕ) (received_daisies : ℕ) : ℕ :=
  let total_daisies := initial_daisies + received_daisies
  let given_to_mother := total_daisies / 2
  total_daisies - given_to_mother

theorem kylie_daisies_left : daisies_problem 5 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_kylie_daisies_left_l1268_126845


namespace NUMINAMATH_CALUDE_children_age_sum_l1268_126843

theorem children_age_sum :
  let num_children : ℕ := 5
  let age_interval : ℕ := 3
  let youngest_age : ℕ := 6
  let ages : List ℕ := List.range num_children |>.map (fun i => youngest_age + i * age_interval)
  ages.sum = 60 := by
  sorry

end NUMINAMATH_CALUDE_children_age_sum_l1268_126843


namespace NUMINAMATH_CALUDE_third_angle_is_75_l1268_126841

/-- A triangle formed by folding a square piece of paper -/
structure FoldedTriangle where
  /-- Angle formed by splitting a right angle in half -/
  angle_mna : ℝ
  /-- Angle formed by three equal angles adding up to 180° -/
  angle_amn : ℝ
  /-- The third angle of the triangle -/
  angle_anm : ℝ
  /-- Proof that angle_mna is 45° -/
  h_mna : angle_mna = 45
  /-- Proof that angle_amn is 60° -/
  h_amn : angle_amn = 60
  /-- Proof that the sum of all angles is 180° -/
  h_sum : angle_mna + angle_amn + angle_anm = 180

/-- Theorem stating that the third angle is 75° -/
theorem third_angle_is_75 (t : FoldedTriangle) : t.angle_anm = 75 := by
  sorry

end NUMINAMATH_CALUDE_third_angle_is_75_l1268_126841


namespace NUMINAMATH_CALUDE_marble_probability_l1268_126829

/-- Represents a box of marbles -/
structure MarbleBox where
  total : ℕ
  black : ℕ
  white : ℕ
  sum_check : total = black + white

/-- The problem setup -/
def marble_problem (box1 box2 : MarbleBox) : Prop :=
  box1.total + box2.total = 30 ∧
  box1.black = 3 * box2.black ∧
  (box1.black : ℚ) / box1.total * (box2.black : ℚ) / box2.total = 1/2

theorem marble_probability (box1 box2 : MarbleBox) 
  (h : marble_problem box1 box2) : 
  (box1.white : ℚ) / box1.total * (box2.white : ℚ) / box2.total = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l1268_126829


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_value_l1268_126813

theorem set_inclusion_implies_a_value (a : ℝ) :
  let A : Set ℝ := {x | |x| = 1}
  let B : Set ℝ := {x | a * x = 1}
  A ⊇ B →
  a = -1 ∨ a = 0 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_value_l1268_126813


namespace NUMINAMATH_CALUDE_math_test_problem_count_l1268_126877

theorem math_test_problem_count :
  ∀ (total_points three_point_count four_point_count : ℕ),
    total_points = 100 →
    four_point_count = 10 →
    total_points = 3 * three_point_count + 4 * four_point_count →
    three_point_count + four_point_count = 30 :=
by sorry

end NUMINAMATH_CALUDE_math_test_problem_count_l1268_126877


namespace NUMINAMATH_CALUDE_soda_preference_result_l1268_126887

/-- The number of people who chose "Soda" in a survey about carbonated beverages -/
def soda_preference (total_surveyed : ℕ) (soda_angle : ℕ) : ℕ :=
  (total_surveyed * soda_angle) / 360

/-- Theorem stating that 243 people chose "Soda" in the survey -/
theorem soda_preference_result : soda_preference 540 162 = 243 := by
  sorry

end NUMINAMATH_CALUDE_soda_preference_result_l1268_126887


namespace NUMINAMATH_CALUDE_min_green_fraction_of_4x4x4_cube_l1268_126883

/-- Represents a cube with colored unit cubes -/
structure ColoredCube where
  edge_length : ℕ
  total_cubes : ℕ
  blue_cubes : ℕ
  green_cubes : ℕ

/-- Calculates the surface area of a cube -/
def surface_area (c : ColoredCube) : ℕ := 6 * c.edge_length^2

/-- Calculates the minimum visible green surface area -/
def min_green_surface_area (c : ColoredCube) : ℕ := c.green_cubes - 4

theorem min_green_fraction_of_4x4x4_cube (c : ColoredCube) 
  (h1 : c.edge_length = 4)
  (h2 : c.total_cubes = 64)
  (h3 : c.blue_cubes = 56)
  (h4 : c.green_cubes = 8) :
  (min_green_surface_area c : ℚ) / (surface_area c : ℚ) = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_min_green_fraction_of_4x4x4_cube_l1268_126883


namespace NUMINAMATH_CALUDE_monotonicity_condition_l1268_126884

/-- Represents a voting system with n voters and m candidates. -/
structure VotingSystem where
  n : ℕ  -- number of voters
  m : ℕ  -- number of candidates
  k : ℕ  -- number of top choices each voter selects

/-- Represents a poll profile (arrangement of candidate rankings by voters). -/
def PollProfile (vs : VotingSystem) := Fin vs.n → (Fin vs.m → Fin vs.m)

/-- Determines if a candidate is a winner in a given poll profile. -/
def isWinner (vs : VotingSystem) (profile : PollProfile vs) (candidate : Fin vs.m) : Prop :=
  sorry

/-- Determines if one profile is a-good compared to another. -/
def isAGood (vs : VotingSystem) (a : Fin vs.m) (R R' : PollProfile vs) : Prop :=
  ∀ (voter : Fin vs.n) (candidate : Fin vs.m),
    (R voter candidate > R voter a) → (R' voter candidate > R' voter a)

/-- Defines the monotonicity property for a voting system. -/
def isMonotone (vs : VotingSystem) : Prop :=
  ∀ (R R' : PollProfile vs) (a : Fin vs.m),
    isWinner vs R a → isAGood vs a R R' → isWinner vs R' a

/-- The main theorem stating the condition for monotonicity. -/
theorem monotonicity_condition (vs : VotingSystem) :
  isMonotone vs ↔ vs.k > (vs.m * (vs.n - 1)) / vs.n :=
  sorry

end NUMINAMATH_CALUDE_monotonicity_condition_l1268_126884


namespace NUMINAMATH_CALUDE_quadratic_triangle_area_l1268_126899

/-- Given a quadratic function y = ax^2 + bx + c where b^2 - 4ac > 0,
    the area of the triangle formed by its intersections with the x-axis and y-axis is |c|/(2|a|) -/
theorem quadratic_triangle_area (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let f : ℝ → ℝ := λ x => a*x^2 + b*x + c
  let triangle_area := (abs c) / (2 * abs a)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (1/2 * abs (x₁ - x₂) * abs c = triangle_area) :=
sorry

end NUMINAMATH_CALUDE_quadratic_triangle_area_l1268_126899


namespace NUMINAMATH_CALUDE_M_greater_than_N_l1268_126894

theorem M_greater_than_N : ∀ x : ℝ, x^2 + 4*x - 2 > 6*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_M_greater_than_N_l1268_126894


namespace NUMINAMATH_CALUDE_investment_profit_distribution_l1268_126865

/-- Represents the investment and profit distribution problem -/
theorem investment_profit_distribution 
  (total_capital : ℕ) 
  (total_profit : ℕ) 
  (a_invest_diff : ℕ) 
  (b_invest_diff : ℕ) 
  (d_invest_diff : ℕ) 
  (a_duration b_duration c_duration d_duration : ℕ) 
  (h1 : total_capital = 100000)
  (h2 : total_profit = 50000)
  (h3 : a_invest_diff = 10000)
  (h4 : b_invest_diff = 5000)
  (h5 : d_invest_diff = 8000)
  (h6 : a_duration = 12)
  (h7 : b_duration = 10)
  (h8 : c_duration = 8)
  (h9 : d_duration = 6) :
  ∃ (c_invest : ℕ),
    let b_invest := c_invest + b_invest_diff
    let a_invest := b_invest + a_invest_diff
    let d_invest := a_invest + d_invest_diff
    c_invest + b_invest + a_invest + d_invest = total_capital ∧
    (b_invest * b_duration : ℚ) / ((c_invest * c_duration + b_invest * b_duration + a_invest * a_duration + d_invest * d_duration) : ℚ) * total_profit = 10925 := by
  sorry

end NUMINAMATH_CALUDE_investment_profit_distribution_l1268_126865


namespace NUMINAMATH_CALUDE_hall_tiling_proof_l1268_126842

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℚ
  width : ℚ

/-- Converts inches to feet -/
def inchesToFeet (inches : ℚ) : ℚ := inches / 12

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℚ := d.length * d.width

/-- Calculates the number of smaller rectangles needed to cover a larger rectangle -/
def tilesRequired (hall : Dimensions) (tile : Dimensions) : ℕ :=
  (area hall / area tile).ceil.toNat

theorem hall_tiling_proof :
  let hall : Dimensions := { length := 15, width := 18 }
  let tile : Dimensions := { length := inchesToFeet 3, width := inchesToFeet 9 }
  tilesRequired hall tile = 1440 := by
  sorry

end NUMINAMATH_CALUDE_hall_tiling_proof_l1268_126842


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1268_126869

theorem cube_root_equation_solution :
  ∃ x : ℝ, (x - 5)^3 = (1/27)⁻¹ ∧ x = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1268_126869


namespace NUMINAMATH_CALUDE_investment_problem_l1268_126853

/-- Calculates an investor's share of the profit based on their investment, duration, and the total profit --/
def calculate_share (investment : ℕ) (duration : ℕ) (total_investment_time : ℕ) (total_profit : ℕ) : ℚ :=
  (investment * duration : ℚ) / total_investment_time * total_profit

/-- Represents the investment problem with four investors --/
theorem investment_problem (tom_investment : ℕ) (jose_investment : ℕ) (anil_investment : ℕ) (maya_investment : ℕ)
  (tom_duration : ℕ) (jose_duration : ℕ) (anil_duration : ℕ) (maya_duration : ℕ) (total_profit : ℕ) :
  tom_investment = 30000 →
  jose_investment = 45000 →
  anil_investment = 50000 →
  maya_investment = 70000 →
  tom_duration = 12 →
  jose_duration = 10 →
  anil_duration = 7 →
  maya_duration = 1 →
  total_profit = 108000 →
  let total_investment_time := tom_investment * tom_duration + jose_investment * jose_duration +
                               anil_investment * anil_duration + maya_investment * maya_duration
  abs (calculate_share jose_investment jose_duration total_investment_time total_profit - 39512.20) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l1268_126853


namespace NUMINAMATH_CALUDE_fraction_value_l1268_126867

theorem fraction_value (x y : ℝ) (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 3) (h3 : ∃ (n : ℤ), x / y = ↑n) : x / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1268_126867


namespace NUMINAMATH_CALUDE_article_large_font_pages_l1268_126856

/-- Represents the number of pages in large font for an article -/
def large_font_pages (total_words : ℕ) (words_per_large_page : ℕ) (words_per_small_page : ℕ) (total_pages : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of large font pages in the article -/
theorem article_large_font_pages :
  large_font_pages 48000 1800 2400 21 = 4 := by
  sorry

end NUMINAMATH_CALUDE_article_large_font_pages_l1268_126856


namespace NUMINAMATH_CALUDE_probability_is_half_l1268_126854

/-- An isosceles triangle with 45-degree base angles -/
structure IsoscelesTriangle45 where
  -- We don't need to define the specific geometry, just that it exists
  exists_triangle : True

/-- The triangle is divided into six equal areas -/
def divided_into_six_areas (t : IsoscelesTriangle45) : Prop :=
  ∃ (areas : Finset ℝ), areas.card = 6 ∧ ∀ a ∈ areas, a > 0 ∧ (∀ b ∈ areas, a = b)

/-- Three areas are selected -/
def three_areas_selected (t : IsoscelesTriangle45) (areas : Finset ℝ) : Prop :=
  ∃ (selected : Finset ℝ), selected ⊆ areas ∧ selected.card = 3

/-- The probability of a point falling in the selected areas -/
def probability_in_selected (t : IsoscelesTriangle45) (areas selected : Finset ℝ) : ℚ :=
  (selected.card : ℚ) / (areas.card : ℚ)

/-- The main theorem -/
theorem probability_is_half (t : IsoscelesTriangle45) 
  (h1 : divided_into_six_areas t)
  (h2 : ∃ areas selected, three_areas_selected t areas ∧ selected ⊆ areas) :
  ∃ areas selected, three_areas_selected t areas ∧ 
    probability_in_selected t areas selected = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_half_l1268_126854


namespace NUMINAMATH_CALUDE_x_greater_than_sin_x_negation_of_implication_and_sufficient_not_necessary_for_or_negation_of_forall_x_minus_ln_x_positive_l1268_126886

-- Statement 1
theorem x_greater_than_sin_x (x : ℝ) (h : x > 0) : x > Real.sin x := by sorry

-- Statement 2
theorem negation_of_implication :
  (¬ (∀ x : ℝ, x - Real.sin x = 0 → x = 0)) ↔
  (∃ x : ℝ, x - Real.sin x ≠ 0 ∧ x ≠ 0) := by sorry

-- Statement 3
theorem and_sufficient_not_necessary_for_or (p q : Prop) :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := by sorry

-- Statement 4
theorem negation_of_forall_x_minus_ln_x_positive :
  (¬ (∀ x : ℝ, x - Real.log x > 0)) ↔
  (∃ x : ℝ, x - Real.log x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_x_greater_than_sin_x_negation_of_implication_and_sufficient_not_necessary_for_or_negation_of_forall_x_minus_ln_x_positive_l1268_126886


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1268_126889

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set B
def B : Set ℕ := {1, 2}

-- Theorem statement
theorem intersection_A_complement_B (A : Set ℕ) 
  (h1 : A ⊆ U) 
  (h2 : B ⊆ U) 
  (h3 : (U \ (A ∪ B)) = {4}) : 
  A ∩ (U \ B) = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1268_126889


namespace NUMINAMATH_CALUDE_triangle_min_angle_le_60_l1268_126839

theorem triangle_min_angle_le_60 (A B C : ℝ) : 
  A + B + C = 180 → A > 0 → B > 0 → C > 0 → min A (min B C) ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_min_angle_le_60_l1268_126839


namespace NUMINAMATH_CALUDE_linear_function_is_shifted_odd_exponential_function_is_not_shifted_odd_sine_shifted_odd_condition_cubic_function_not_shifted_odd_condition_l1268_126898

/-- A function is a shifted odd function if there exists a real number m such that
    f(x+m) - f(m) is an odd function over ℝ. -/
def is_shifted_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, f (x + m) - f m = -(f (-x + m) - f m)

/-- The function f(x) = 2x + 1 is a shifted odd function. -/
theorem linear_function_is_shifted_odd :
  is_shifted_odd_function (fun x => 2 * x + 1) :=
sorry

/-- The function g(x) = 2^x is not a shifted odd function. -/
theorem exponential_function_is_not_shifted_odd :
  ¬ is_shifted_odd_function (fun x => 2^x) :=
sorry

/-- For f(x) = sin(x + φ) to be a shifted odd function with shift difference π/4,
    φ must equal kπ - π/4 for some integer k. -/
theorem sine_shifted_odd_condition (φ : ℝ) :
  is_shifted_odd_function (fun x => Real.sin (x + φ)) ∧ 
  (∃ m : ℝ, m = π/4 ∧ ∀ x : ℝ, Real.sin (x + m + φ) - Real.sin (m + φ) = -(Real.sin (-x + m + φ) - Real.sin (m + φ))) ↔
  ∃ k : ℤ, φ = k * π - π/4 :=
sorry

/-- For f(x) = x^3 + bx^2 + cx to not be a shifted odd function for any m in [-1/2, +∞),
    b must be greater than 3/2, and c can be any real number. -/
theorem cubic_function_not_shifted_odd_condition (b c : ℝ) :
  (∀ m : ℝ, m ≥ -1/2 → ¬ is_shifted_odd_function (fun x => x^3 + b*x^2 + c*x)) ↔
  b > 3/2 :=
sorry

end NUMINAMATH_CALUDE_linear_function_is_shifted_odd_exponential_function_is_not_shifted_odd_sine_shifted_odd_condition_cubic_function_not_shifted_odd_condition_l1268_126898


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_squared_l1268_126810

theorem x_squared_plus_reciprocal_squared (x : ℝ) (h : x + 1/x = 3.5) : 
  x^2 + 1/x^2 = 10.25 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_squared_l1268_126810


namespace NUMINAMATH_CALUDE_child_tickets_sold_l1268_126807

/-- Proves the number of child's tickets sold in a movie theater -/
theorem child_tickets_sold (adult_price child_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : total_revenue = 5100) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 400 := by
  sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l1268_126807


namespace NUMINAMATH_CALUDE_g_function_equality_l1268_126844

theorem g_function_equality (x : ℝ) :
  let g : ℝ → ℝ := λ x => -4*x^5 + 4*x^3 - 4*x + 6
  4*x^5 + 3*x^3 + x - 2 + g x = 7*x^3 - 5*x + 4 :=
by
  sorry

end NUMINAMATH_CALUDE_g_function_equality_l1268_126844


namespace NUMINAMATH_CALUDE_total_hotdogs_by_wednesday_l1268_126826

def hotdog_sequence (n : ℕ) : ℕ := 10 + 2 * (n - 1)

theorem total_hotdogs_by_wednesday :
  (hotdog_sequence 1) + (hotdog_sequence 2) + (hotdog_sequence 3) = 36 :=
by sorry

end NUMINAMATH_CALUDE_total_hotdogs_by_wednesday_l1268_126826


namespace NUMINAMATH_CALUDE_cosine_sine_ratio_equals_sqrt_three_l1268_126824

theorem cosine_sine_ratio_equals_sqrt_three : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_ratio_equals_sqrt_three_l1268_126824


namespace NUMINAMATH_CALUDE_count_polynomials_l1268_126849

-- Define a function to check if an expression is a polynomial
def is_polynomial (expr : String) : Bool :=
  match expr with
  | "3/4x^2" => true
  | "3ab" => true
  | "x+5" => true
  | "y/(5x)" => false
  | "-1" => true
  | "y/3" => true
  | "a^2-b^2" => true
  | "a" => true
  | _ => false

-- Define the list of expressions
def expressions : List String :=
  ["3/4x^2", "3ab", "x+5", "y/(5x)", "-1", "y/3", "a^2-b^2", "a"]

-- Theorem: There are exactly 7 polynomials in the list of expressions
theorem count_polynomials :
  (expressions.filter is_polynomial).length = 7 :=
by sorry

end NUMINAMATH_CALUDE_count_polynomials_l1268_126849


namespace NUMINAMATH_CALUDE_average_after_removal_l1268_126832

theorem average_after_removal (numbers : Finset ℝ) (sum : ℝ) : 
  Finset.card numbers = 12 →
  sum = Finset.sum numbers id →
  sum / 12 = 90 →
  65 ∈ numbers →
  75 ∈ numbers →
  85 ∈ numbers →
  (sum - 65 - 75 - 85) / 9 = 95 :=
by sorry

end NUMINAMATH_CALUDE_average_after_removal_l1268_126832


namespace NUMINAMATH_CALUDE_division_problem_l1268_126803

theorem division_problem (L S Q : ℕ) : 
  L - S = 1355 → 
  L = 1608 → 
  L = S * Q + 15 → 
  Q = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1268_126803


namespace NUMINAMATH_CALUDE_equation_solutions_l1268_126879

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 5 ∧ x2 = 2 - Real.sqrt 5 ∧
    x1^2 - 4*x1 - 1 = 0 ∧ x2^2 - 4*x2 - 1 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = -3 ∧ x2 = -2 ∧
    (x1 + 3)^2 = x1 + 3 ∧ (x2 + 3)^2 = x2 + 3) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1268_126879


namespace NUMINAMATH_CALUDE_K_characterization_l1268_126882

/-- Function that reverses the digits of a positive integer in decimal notation -/
def f (n : ℕ+) : ℕ+ := sorry

/-- The set of all positive integers k such that, for any multiple n of k, k also divides f(n) -/
def K : Set ℕ+ :=
  {k : ℕ+ | ∀ n : ℕ+, k ∣ n → k ∣ f n}

/-- Theorem stating that K is equal to the set {1, 3, 9, 11, 33, 99} -/
theorem K_characterization : K = {1, 3, 9, 11, 33, 99} := by sorry

end NUMINAMATH_CALUDE_K_characterization_l1268_126882


namespace NUMINAMATH_CALUDE_intersection_A_B_l1268_126890

-- Define set A
def A : Set ℝ := {x | ∃ y : ℝ, y^2 = x}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1268_126890


namespace NUMINAMATH_CALUDE_max_volume_angle_l1268_126861

/-- A square ABCD folded along diagonal AC to form a regular pyramid -/
structure FoldedSquare where
  side : ℝ
  fold_angle : ℝ

/-- The angle between line BD and plane ABC in the folded square -/
def angle_bd_abc (s : FoldedSquare) : ℝ := sorry

/-- The volume of the pyramid formed by the folded square -/
def pyramid_volume (s : FoldedSquare) : ℝ := sorry

theorem max_volume_angle (s : FoldedSquare) :
  (∀ t : FoldedSquare, pyramid_volume t ≤ pyramid_volume s) →
  angle_bd_abc s = 45 := by sorry

end NUMINAMATH_CALUDE_max_volume_angle_l1268_126861


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1268_126860

/-- Proves that a rectangle with width w and length 3w, whose perimeter is twice its area, has width 4/3 and length 4 -/
theorem rectangle_dimensions (w : ℝ) (h1 : w > 0) : 
  (2 * (w + 3*w) = 2 * (w * 3*w)) → w = 4/3 ∧ 3*w = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1268_126860


namespace NUMINAMATH_CALUDE_binomial_150_150_equals_1_l1268_126888

theorem binomial_150_150_equals_1 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_equals_1_l1268_126888


namespace NUMINAMATH_CALUDE_nell_cards_to_john_l1268_126862

def cards_problem (initial_cards : ℕ) (cards_to_jeff : ℕ) (cards_left : ℕ) : Prop :=
  let total_given_away := initial_cards - cards_left
  let cards_to_john := total_given_away - cards_to_jeff
  cards_to_john = 195

theorem nell_cards_to_john :
  cards_problem 573 168 210 :=
by
  sorry

end NUMINAMATH_CALUDE_nell_cards_to_john_l1268_126862


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l1268_126858

-- Define the complex number z
def z : ℂ := (1 + Complex.I) * (2 * Complex.I)

-- Theorem statement
theorem z_in_second_quadrant : Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l1268_126858


namespace NUMINAMATH_CALUDE_binomial_10_choose_2_l1268_126800

theorem binomial_10_choose_2 : Nat.choose 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_2_l1268_126800


namespace NUMINAMATH_CALUDE_mean_problem_l1268_126859

theorem mean_problem (x : ℝ) : 
  (28 + x + 42 + 78 + 104) / 5 = 90 →
  (128 + 255 + 511 + 1023 + x) / 5 = 423 := by
sorry

end NUMINAMATH_CALUDE_mean_problem_l1268_126859


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l1268_126835

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  edge_length : ℝ
  vertex : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the area of the polygon formed by the intersection of a plane and a cube -/
def intersectionArea (c : Cube) (p : Plane) : ℝ := sorry

/-- Theorem stating the area of the intersection polygon -/
theorem intersection_area_theorem (c : Cube) (p q r : Point3D) : 
  c.edge_length = 30 →
  p.x = 10 ∧ p.y = 0 ∧ p.z = 0 →
  q.x = 30 ∧ q.y = 0 ∧ q.z = 10 →
  r.x = 30 ∧ r.y = 20 ∧ r.z = 30 →
  ∃ (plane : Plane), intersectionArea c plane = 450 := by
  sorry

#check intersection_area_theorem

end NUMINAMATH_CALUDE_intersection_area_theorem_l1268_126835


namespace NUMINAMATH_CALUDE_greatest_plants_per_row_l1268_126864

theorem greatest_plants_per_row (sunflowers corn tomatoes : ℕ) 
  (h_sunflowers : sunflowers = 45)
  (h_corn : corn = 81)
  (h_tomatoes : tomatoes = 63) :
  Nat.gcd sunflowers (Nat.gcd corn tomatoes) = 9 := by
  sorry

end NUMINAMATH_CALUDE_greatest_plants_per_row_l1268_126864


namespace NUMINAMATH_CALUDE_transylvanian_truth_telling_l1268_126820

-- Define the types
inductive Being
| Human
| Vampire

-- Define the properties
def declares (b : Being) (x : Prop) : Prop :=
  match b with
  | Being.Human => x
  | Being.Vampire => ¬x

theorem transylvanian_truth_telling (b : Being) (x : Prop) :
  (b = Being.Human → (declares b x → x)) ∧
  (b = Being.Vampire → (declares b x → ¬x)) :=
by sorry

end NUMINAMATH_CALUDE_transylvanian_truth_telling_l1268_126820


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l1268_126876

theorem fixed_point_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 3
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l1268_126876


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l1268_126822

theorem weight_of_replaced_person (n : ℕ) (avg_increase : ℝ) (new_weight : ℝ) :
  n = 8 →
  avg_increase = 2.5 →
  new_weight = 75 →
  ∃ (old_weight : ℝ), old_weight = 55 ∧ n * avg_increase = new_weight - old_weight :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l1268_126822


namespace NUMINAMATH_CALUDE_lanas_winter_clothing_l1268_126808

/-- The number of boxes Lana found -/
def num_boxes : ℕ := 5

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 7

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 8

/-- The total number of pieces of winter clothing Lana had -/
def total_clothing : ℕ := num_boxes * scarves_per_box + num_boxes * mittens_per_box

theorem lanas_winter_clothing : total_clothing = 75 := by
  sorry

end NUMINAMATH_CALUDE_lanas_winter_clothing_l1268_126808


namespace NUMINAMATH_CALUDE_battle_station_staffing_l1268_126806

/-- Represents the number of ways to staff Captain Zarnin's battle station -/
def staff_battle_station (total_applicants : ℕ) (suitable_resumes : ℕ) 
  (assistant_engineer : ℕ) (weapons_maintenance1 : ℕ) (weapons_maintenance2 : ℕ)
  (field_technician : ℕ) (radio_specialist : ℕ) : ℕ :=
  assistant_engineer * weapons_maintenance1 * weapons_maintenance2 * field_technician * radio_specialist

/-- Theorem stating the number of ways to staff the battle station -/
theorem battle_station_staffing :
  staff_battle_station 30 15 3 4 4 5 5 = 960 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l1268_126806


namespace NUMINAMATH_CALUDE_auto_finance_credit_l1268_126895

/-- Proves that the credit extended by automobile finance companies is $40 billion given the specified conditions -/
theorem auto_finance_credit (total_credit : ℝ) (auto_credit_percentage : ℝ) (finance_companies_fraction : ℝ)
  (h1 : total_credit = 342.857)
  (h2 : auto_credit_percentage = 0.35)
  (h3 : finance_companies_fraction = 1/3) :
  finance_companies_fraction * (auto_credit_percentage * total_credit) = 40 := by
  sorry

end NUMINAMATH_CALUDE_auto_finance_credit_l1268_126895


namespace NUMINAMATH_CALUDE_parity_of_cube_plus_multiple_l1268_126834

theorem parity_of_cube_plus_multiple (o n : ℤ) (h_odd : Odd o) :
  Odd (o^3 + n*o) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_parity_of_cube_plus_multiple_l1268_126834


namespace NUMINAMATH_CALUDE_puppies_left_l1268_126811

/-- The number of puppies Alyssa had initially -/
def initial_puppies : ℕ := 7

/-- The number of puppies Alyssa gave to her friends -/
def given_puppies : ℕ := 5

/-- Theorem: Alyssa is left with 2 puppies -/
theorem puppies_left : initial_puppies - given_puppies = 2 := by
  sorry

end NUMINAMATH_CALUDE_puppies_left_l1268_126811


namespace NUMINAMATH_CALUDE_nested_squares_segment_length_l1268_126892

/-- Given four nested squares with known segment lengths, prove that the length of GH
    is the sum of lengths AB, CD, and FE. -/
theorem nested_squares_segment_length 
  (AB CD FE : ℝ) 
  (h1 : AB = 11) 
  (h2 : CD = 5) 
  (h3 : FE = 13) : 
  ∃ GH : ℝ, GH = AB + CD + FE :=
by sorry

end NUMINAMATH_CALUDE_nested_squares_segment_length_l1268_126892


namespace NUMINAMATH_CALUDE_bubble_sort_probability_l1268_126831

theorem bubble_sort_probability (n : ℕ) (h : n = 36) :
  let arrangements := n.factorial
  let favorable_outcomes := (n - 2).factorial
  (favorable_outcomes : ℚ) / arrangements = 1 / 1260 := by
  sorry

end NUMINAMATH_CALUDE_bubble_sort_probability_l1268_126831


namespace NUMINAMATH_CALUDE_mandy_reading_progression_l1268_126815

/-- Calculates the present book length given Mandy's reading progression --/
def present_book_length (starting_age : ℕ) (starting_length : ℕ) 
  (double_age_multiplier : ℕ) (eight_years_later_multiplier : ℕ) 
  (present_multiplier : ℕ) : ℕ :=
  let double_age_length := starting_length * double_age_multiplier
  let eight_years_later_length := double_age_length * eight_years_later_multiplier
  eight_years_later_length * present_multiplier

/-- Theorem stating that the present book length is 480 pages --/
theorem mandy_reading_progression : 
  present_book_length 6 8 5 3 4 = 480 := by
  sorry

#eval present_book_length 6 8 5 3 4

end NUMINAMATH_CALUDE_mandy_reading_progression_l1268_126815


namespace NUMINAMATH_CALUDE_sum_of_digits_of_multiple_of_five_l1268_126805

/-- Given two natural numbers, returns true if they have the same digits in any order -/
def sameDigits (a b : ℕ) : Prop := sorry

/-- Returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_multiple_of_five (a b : ℕ) :
  sameDigits a b → sumOfDigits (5 * a) = sumOfDigits (5 * b) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_multiple_of_five_l1268_126805


namespace NUMINAMATH_CALUDE_sum_of_two_squares_l1268_126874

theorem sum_of_two_squares (u : ℕ) (h : Odd u) :
  ∃ (a b : ℕ), (3^(3*u) - 1) / (3^u - 1) = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_l1268_126874


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1268_126837

theorem inequality_system_solution_set :
  let S := {x : ℝ | (1 + x > -1) ∧ (4 - 2*x ≥ 0)}
  S = {x : ℝ | -2 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1268_126837


namespace NUMINAMATH_CALUDE_x_range_l1268_126848

theorem x_range (m : ℝ) (h1 : 0 < m) (h2 : m ≤ 5) :
  (∀ x : ℝ, x^2 + (2*m - 1)*x > 4*x + 2*m - 4) →
  (∀ x : ℝ, x < -6 ∨ x > 4) :=
by sorry

end NUMINAMATH_CALUDE_x_range_l1268_126848
