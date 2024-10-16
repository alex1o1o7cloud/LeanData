import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l1268_126854

theorem equation_solution :
  ∀ x : ℚ, x ≠ 3 → ((x + 5) / (x - 3) = 4 ↔ x = 17 / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1268_126854


namespace NUMINAMATH_CALUDE_min_value_of_squared_differences_l1268_126842

theorem min_value_of_squared_differences (a b c : ℝ) :
  ∃ (min : ℝ), min = ((a - b)^2 + (b - c)^2 + (a - c)^2) / 3 ∧
  ∀ (x : ℝ), (x - a)^2 + (x - b)^2 + (x - c)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_squared_differences_l1268_126842


namespace NUMINAMATH_CALUDE_min_value_problem_l1268_126876

theorem min_value_problem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 1) : 
  (a * f)^2 + (b * e)^2 + (c * h)^2 + (d * g)^2 ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l1268_126876


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1268_126819

theorem polynomial_division_theorem (x : ℝ) :
  (x - 1) * (8 * x^3 + 15 * x^2 + 18 * x + 13) + 5 = 8 * x^4 + 7 * x^3 + 3 * x^2 - 5 * x - 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1268_126819


namespace NUMINAMATH_CALUDE_tiffany_fastest_l1268_126802

structure Runner where
  name : String
  uphill_blocks : ℕ
  uphill_time : ℕ
  downhill_blocks : ℕ
  downhill_time : ℕ
  flat_blocks : ℕ
  flat_time : ℕ

def total_distance (r : Runner) : ℕ :=
  r.uphill_blocks + r.downhill_blocks + r.flat_blocks

def total_time (r : Runner) : ℕ :=
  r.uphill_time + r.downhill_time + r.flat_time

def average_speed (r : Runner) : ℚ :=
  (total_distance r : ℚ) / (total_time r : ℚ)

def tiffany : Runner :=
  { name := "Tiffany"
    uphill_blocks := 6
    uphill_time := 3
    downhill_blocks := 8
    downhill_time := 5
    flat_blocks := 6
    flat_time := 3 }

def moses : Runner :=
  { name := "Moses"
    uphill_blocks := 5
    uphill_time := 5
    downhill_blocks := 10
    downhill_time := 10
    flat_blocks := 5
    flat_time := 4 }

def morgan : Runner :=
  { name := "Morgan"
    uphill_blocks := 7
    uphill_time := 4
    downhill_blocks := 9
    downhill_time := 6
    flat_blocks := 4
    flat_time := 2 }

theorem tiffany_fastest : 
  average_speed tiffany > average_speed moses ∧ 
  average_speed tiffany > average_speed morgan ∧
  total_distance tiffany = 20 ∧
  total_distance moses = 20 ∧
  total_distance morgan = 20 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_fastest_l1268_126802


namespace NUMINAMATH_CALUDE_expected_bullets_is_1_89_l1268_126805

/-- The expected number of remaining bullets in a shooting scenario -/
def expected_remaining_bullets (total_bullets : ℕ) (hit_probability : ℝ) : ℝ :=
  let miss_probability := 1 - hit_probability
  let p_zero := miss_probability * miss_probability
  let p_one := miss_probability * hit_probability
  let p_two := hit_probability
  1 * p_one + 2 * p_two

/-- The theorem stating that the expected number of remaining bullets is 1.89 -/
theorem expected_bullets_is_1_89 :
  expected_remaining_bullets 3 0.9 = 1.89 := by sorry

end NUMINAMATH_CALUDE_expected_bullets_is_1_89_l1268_126805


namespace NUMINAMATH_CALUDE_tan_ratio_given_sin_condition_l1268_126885

theorem tan_ratio_given_sin_condition (α : Real) 
  (h : 5 * Real.sin (2 * α) = Real.sin (2 * (π / 180))) : 
  Real.tan (α + π / 180) / Real.tan (α - π / 180) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_given_sin_condition_l1268_126885


namespace NUMINAMATH_CALUDE_unique_triple_l1268_126834

theorem unique_triple : ∃! (x y z : ℕ), 
  x > 1 ∧ y > 1 ∧ z > 1 ∧
  (yz - 1) % x = 0 ∧ 
  (zx - 1) % y = 0 ∧ 
  (xy - 1) % z = 0 ∧
  x = 5 ∧ y = 3 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l1268_126834


namespace NUMINAMATH_CALUDE_object_length_increase_l1268_126825

/-- The number of days required for an object to reach 50 times its original length -/
def n : ℕ := 147

/-- The factor by which the object's length increases on day k -/
def increase_factor (k : ℕ) : ℚ := (k + 3 : ℚ) / (k + 2 : ℚ)

/-- The total increase factor after n days -/
def total_increase_factor (n : ℕ) : ℚ := (n + 3 : ℚ) / 3

theorem object_length_increase :
  total_increase_factor n = 50 := by sorry

end NUMINAMATH_CALUDE_object_length_increase_l1268_126825


namespace NUMINAMATH_CALUDE_seventh_group_sample_l1268_126827

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (m : ℕ) (k : ℕ) : ℕ :=
  10 * (k - 1) + (m + k) % 10

/-- The problem statement translated to a theorem -/
theorem seventh_group_sample :
  ∀ m : ℕ,
  m = 6 →
  systematicSample m 7 = 73 :=
by
  sorry

end NUMINAMATH_CALUDE_seventh_group_sample_l1268_126827


namespace NUMINAMATH_CALUDE_sum_x_y_z_l1268_126818

/-- Given that:
    - 0.5% of x equals 0.65 rupees
    - 1.25% of y equals 1.04 rupees
    - 2.5% of z equals 75% of x
    Prove that the sum of x, y, and z is 4113.2 rupees -/
theorem sum_x_y_z (x y z : ℝ) 
  (hx : 0.005 * x = 0.65)
  (hy : 0.0125 * y = 1.04)
  (hz : 0.025 * z = 0.75 * x) :
  x + y + z = 4113.2 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_z_l1268_126818


namespace NUMINAMATH_CALUDE_youngsville_population_change_l1268_126874

def initial_population : ℕ := 684
def growth_rate : ℚ := 25 / 100
def decline_rate : ℚ := 40 / 100

theorem youngsville_population_change :
  let increased_population := initial_population + (initial_population * growth_rate).floor
  let final_population := increased_population - (increased_population * decline_rate).floor
  final_population = 513 := by sorry

end NUMINAMATH_CALUDE_youngsville_population_change_l1268_126874


namespace NUMINAMATH_CALUDE_mariels_dogs_count_l1268_126814

/-- The number of dogs Mariel is walking -/
def mariels_dogs : ℕ := 5

/-- The number of dogs the other walker has -/
def other_walkers_dogs : ℕ := 3

/-- The number of legs each dog has -/
def dog_legs : ℕ := 4

/-- The number of legs each human has -/
def human_legs : ℕ := 2

/-- The total number of legs tangled in leashes -/
def total_legs : ℕ := 36

/-- The number of dog walkers -/
def num_walkers : ℕ := 2

theorem mariels_dogs_count :
  mariels_dogs * dog_legs + 
  other_walkers_dogs * dog_legs + 
  num_walkers * human_legs = total_legs := by sorry

end NUMINAMATH_CALUDE_mariels_dogs_count_l1268_126814


namespace NUMINAMATH_CALUDE_group_size_calculation_l1268_126841

theorem group_size_calculation (n : ℕ) : 
  (n : ℝ) * 14 + 34 = ((n : ℝ) + 1) * 16 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l1268_126841


namespace NUMINAMATH_CALUDE_smallest_period_is_40_l1268_126881

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The smallest positive period of functions satisfying the condition -/
theorem smallest_period_is_40 :
  ∀ f : ℝ → ℝ, SatisfiesCondition f →
    (∃ p : ℝ, p > 0 ∧ IsPeriod f p ∧
      ∀ q : ℝ, q > 0 → IsPeriod f q → p ≤ q) →
    (∃ p : ℝ, p > 0 ∧ IsPeriod f p ∧
      ∀ q : ℝ, q > 0 → IsPeriod f q → p ≤ q) ∧ p = 40 :=
by sorry

end NUMINAMATH_CALUDE_smallest_period_is_40_l1268_126881


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l1268_126864

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem: The volume of the specific tetrahedron PQRS is 1715/(144√2) -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 3,
    PS := 5,
    QR := 5,
    QS := 4,
    RS := 15 / 4 * Real.sqrt 2
  }
  tetrahedronVolume t = 1715 / (144 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l1268_126864


namespace NUMINAMATH_CALUDE_katy_brownies_l1268_126817

/-- The number of brownies Katy made and ate over three days. -/
def brownies_problem (monday : ℕ) : Prop :=
  ∃ (tuesday wednesday : ℕ),
    tuesday = 2 * monday ∧
    wednesday = 3 * tuesday ∧
    monday + tuesday + wednesday = 45

/-- Theorem stating that Katy made 45 brownies in total. -/
theorem katy_brownies : brownies_problem 5 := by
  sorry

end NUMINAMATH_CALUDE_katy_brownies_l1268_126817


namespace NUMINAMATH_CALUDE_rent_utilities_percentage_after_raise_l1268_126846

theorem rent_utilities_percentage_after_raise (initial_income : ℝ) 
  (initial_percentage : ℝ) (salary_increase : ℝ) : 
  initial_income = 1000 →
  initial_percentage = 40 →
  salary_increase = 600 →
  let initial_amount := initial_income * (initial_percentage / 100)
  let new_income := initial_income + salary_increase
  (initial_amount / new_income) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_rent_utilities_percentage_after_raise_l1268_126846


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1268_126837

theorem inequality_solution_set (x : ℝ) : 
  (((x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4) ↔ 
  (x > -1/4 ∧ x < 0) ∨ (x ≥ 3/2 ∧ x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1268_126837


namespace NUMINAMATH_CALUDE_cube_plus_self_equality_l1268_126832

theorem cube_plus_self_equality (m n : ℤ) : m^3 = n^3 + n → m = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_self_equality_l1268_126832


namespace NUMINAMATH_CALUDE_galaxy_composition_l1268_126880

/-- Represents the counts of celestial bodies in a galaxy -/
structure GalaxyComposition where
  planets : ℕ
  solarSystems : ℕ
  stars : ℕ
  moonSystems : ℕ

/-- Calculates the composition of a galaxy based on given ratios and planet count -/
def calculateGalaxyComposition (planetCount : ℕ) : GalaxyComposition :=
  let solarSystems := planetCount * 8
  let stars := solarSystems * 4
  let moonSystems := planetCount * 3 / 5
  { planets := planetCount
  , solarSystems := solarSystems
  , stars := stars
  , moonSystems := moonSystems }

/-- Theorem stating the composition of the galaxy given the conditions -/
theorem galaxy_composition :
  let composition := calculateGalaxyComposition 20
  composition.planets = 20 ∧
  composition.solarSystems = 160 ∧
  composition.stars = 640 ∧
  composition.moonSystems = 12 :=
by sorry

end NUMINAMATH_CALUDE_galaxy_composition_l1268_126880


namespace NUMINAMATH_CALUDE_one_fifth_of_ten_x_plus_five_l1268_126809

theorem one_fifth_of_ten_x_plus_five (x : ℝ) : (1 / 5) * (10 * x + 5) = 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_ten_x_plus_five_l1268_126809


namespace NUMINAMATH_CALUDE_child_tickets_sold_l1268_126871

/-- Proves the number of child tickets sold given the ticket prices and total sales information -/
theorem child_tickets_sold 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (total_sales : ℕ) 
  (total_tickets : ℕ) 
  (h1 : adult_price = 5)
  (h2 : child_price = 3)
  (h3 : total_sales = 178)
  (h4 : total_tickets = 42) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_sales ∧
    child_tickets = 16 := by
  sorry


end NUMINAMATH_CALUDE_child_tickets_sold_l1268_126871


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1268_126889

theorem unique_triple_solution : 
  ∃! (p x y : ℕ), 
    Prime p ∧ 
    p ^ x = y ^ 4 + 4 ∧ 
    p = 5 ∧ x = 1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1268_126889


namespace NUMINAMATH_CALUDE_frank_breakfast_shopping_cost_l1268_126869

/-- Calculates the total cost of Frank's breakfast shopping --/
def breakfast_shopping_cost (bun_price : ℚ) (bun_quantity : ℕ) (milk_price : ℚ) (milk_quantity : ℕ) (egg_price_multiplier : ℕ) : ℚ :=
  let bun_cost := bun_price * bun_quantity
  let milk_cost := milk_price * milk_quantity
  let egg_cost := milk_price * egg_price_multiplier
  bun_cost + milk_cost + egg_cost

/-- Theorem: The total cost of Frank's breakfast shopping is $11.00 --/
theorem frank_breakfast_shopping_cost :
  breakfast_shopping_cost 0.1 10 2 2 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_frank_breakfast_shopping_cost_l1268_126869


namespace NUMINAMATH_CALUDE_discounted_cost_l1268_126824

/-- The cost of a pencil without discount -/
def pencil_cost : ℚ := sorry

/-- The cost of a notebook -/
def notebook_cost : ℚ := sorry

/-- The discount per pencil when buying more than 10 pencils -/
def discount : ℚ := 0.05

/-- Condition: Cost of 8 pencils and 10 notebooks without discount -/
axiom condition1 : 8 * pencil_cost + 10 * notebook_cost = 5.36

/-- Condition: Cost of 12 pencils and 5 notebooks with discount -/
axiom condition2 : 12 * (pencil_cost - discount) + 5 * notebook_cost = 4.05

/-- The cost of 15 pencils and 12 notebooks with discount -/
def total_cost : ℚ := 15 * (pencil_cost - discount) + 12 * notebook_cost

theorem discounted_cost : total_cost = 7.01 := by sorry

end NUMINAMATH_CALUDE_discounted_cost_l1268_126824


namespace NUMINAMATH_CALUDE_function_range_l1268_126815

-- Define the function
def f (x : ℝ) := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- State the theorem
theorem function_range : 
  { y | ∃ x ∈ domain, f x = y } = { y | -1 ≤ y ∧ y ≤ 3 } := by sorry

end NUMINAMATH_CALUDE_function_range_l1268_126815


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1268_126888

theorem sum_of_coefficients : 
  let p (x : ℝ) := 3*(x^8 - x^5 + 2*x^3 - 6) - 5*(x^4 + 3*x^2) + 2*(x^6 - 5)
  (p 1) = -40 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1268_126888


namespace NUMINAMATH_CALUDE_three_elements_satisfy_l1268_126872

/-- The set M containing elements A, A₁, A₂, A₃, A₄, A₅ -/
inductive M
  | A
  | A1
  | A2
  | A3
  | A4
  | A5

/-- The operation ⊗ defined on M -/
def otimes : M → M → M
  | M.A, M.A => M.A
  | M.A, M.A1 => M.A1
  | M.A, M.A2 => M.A2
  | M.A, M.A3 => M.A3
  | M.A, M.A4 => M.A4
  | M.A, M.A5 => M.A1
  | M.A1, M.A => M.A1
  | M.A1, M.A1 => M.A2
  | M.A1, M.A2 => M.A3
  | M.A1, M.A3 => M.A4
  | M.A1, M.A4 => M.A1
  | M.A1, M.A5 => M.A2
  | M.A2, M.A => M.A2
  | M.A2, M.A1 => M.A3
  | M.A2, M.A2 => M.A4
  | M.A2, M.A3 => M.A1
  | M.A2, M.A4 => M.A2
  | M.A2, M.A5 => M.A3
  | M.A3, M.A => M.A3
  | M.A3, M.A1 => M.A4
  | M.A3, M.A2 => M.A1
  | M.A3, M.A3 => M.A2
  | M.A3, M.A4 => M.A3
  | M.A3, M.A5 => M.A4
  | M.A4, M.A => M.A4
  | M.A4, M.A1 => M.A1
  | M.A4, M.A2 => M.A2
  | M.A4, M.A3 => M.A3
  | M.A4, M.A4 => M.A4
  | M.A4, M.A5 => M.A1
  | M.A5, M.A => M.A1
  | M.A5, M.A1 => M.A2
  | M.A5, M.A2 => M.A3
  | M.A5, M.A3 => M.A4
  | M.A5, M.A4 => M.A1
  | M.A5, M.A5 => M.A2

/-- The theorem stating that exactly 3 elements in M satisfy (a ⊗ a) ⊗ A₂ = A -/
theorem three_elements_satisfy :
  (∃! (s : Finset M), s.card = 3 ∧ ∀ a ∈ s, otimes (otimes a a) M.A2 = M.A) :=
sorry

end NUMINAMATH_CALUDE_three_elements_satisfy_l1268_126872


namespace NUMINAMATH_CALUDE_head_start_calculation_l1268_126801

/-- Prove that given A runs 1 ¾ times as fast as B, and A and B reach a winning post 196 m away at the same time, the head start A gives B is 84 meters. -/
theorem head_start_calculation (speed_a speed_b head_start : ℝ) 
  (h1 : speed_a = (7/4) * speed_b)
  (h2 : (196 - head_start) / speed_b = 196 / speed_a) :
  head_start = 84 := by
  sorry

end NUMINAMATH_CALUDE_head_start_calculation_l1268_126801


namespace NUMINAMATH_CALUDE_ticket_distribution_ways_l1268_126828

/-- The number of ways to distribute tickets among programs -/
def distribute_tickets (total_tickets : ℕ) (num_programs : ℕ) (min_tickets_a : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute 6 tickets among 4 programs with program A receiving at least 3 and the most -/
theorem ticket_distribution_ways : distribute_tickets 6 4 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ticket_distribution_ways_l1268_126828


namespace NUMINAMATH_CALUDE_units_digit_of_6541_pow_826_l1268_126823

theorem units_digit_of_6541_pow_826 : (6541^826) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_6541_pow_826_l1268_126823


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l1268_126831

theorem walnut_trees_planted (trees_before planting : ℕ) (trees_after : ℕ) : 
  trees_before = 22 → trees_after = 55 → planting = trees_after - trees_before :=
by
  sorry

#check walnut_trees_planted 22 33 55

end NUMINAMATH_CALUDE_walnut_trees_planted_l1268_126831


namespace NUMINAMATH_CALUDE_tapanga_corey_candy_difference_l1268_126810

theorem tapanga_corey_candy_difference (total : ℕ) (corey : ℕ) (h1 : total = 66) (h2 : corey = 29) (h3 : corey < total - corey) :
  total - corey - corey = 8 := by
  sorry

end NUMINAMATH_CALUDE_tapanga_corey_candy_difference_l1268_126810


namespace NUMINAMATH_CALUDE_single_digit_between_zero_and_two_l1268_126861

theorem single_digit_between_zero_and_two : 
  ∃! n : ℕ, n < 10 ∧ 0 < n ∧ n < 2 :=
by sorry

end NUMINAMATH_CALUDE_single_digit_between_zero_and_two_l1268_126861


namespace NUMINAMATH_CALUDE_total_episodes_watched_l1268_126830

def episode_length : ℕ := 44
def monday_minutes : ℕ := 138
def thursday_minutes : ℕ := 21
def friday_episodes : ℕ := 2
def weekend_minutes : ℕ := 105

theorem total_episodes_watched :
  (monday_minutes + thursday_minutes + friday_episodes * episode_length + weekend_minutes) / episode_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_episodes_watched_l1268_126830


namespace NUMINAMATH_CALUDE_problem_statement_l1268_126813

theorem problem_statement (x : ℝ) (h : 1 - 5/x + 6/x^3 = 0) : 3/x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1268_126813


namespace NUMINAMATH_CALUDE_interval_intersection_l1268_126857

theorem interval_intersection : ∀ x : ℝ, 
  (2 < 4*x ∧ 4*x < 3 ∧ 2 < 5*x ∧ 5*x < 3) ↔ (1/2 < x ∧ x < 3/5) := by sorry

end NUMINAMATH_CALUDE_interval_intersection_l1268_126857


namespace NUMINAMATH_CALUDE_sin_cos_shift_l1268_126836

theorem sin_cos_shift (x : ℝ) : 
  Real.sin (2 * x + π / 6) = Real.cos (2 * x - π / 6 + π / 2 - π / 12) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l1268_126836


namespace NUMINAMATH_CALUDE_complex_square_simplification_l1268_126879

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 7 - 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l1268_126879


namespace NUMINAMATH_CALUDE_max_value_of_sum_l1268_126804

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f x ^ 2 + f (x + 1) ^ 2 = f x + f (x + 1) + 4

-- State the theorem
theorem max_value_of_sum (h : satisfies_condition f) :
  ∃ M, M = 4 ∧ ∀ x y, f x + f y ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l1268_126804


namespace NUMINAMATH_CALUDE_units_digit_of_n_l1268_126816

theorem units_digit_of_n (m n : ℕ) : 
  m * n = 31^6 ∧ m % 10 = 3 → n % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l1268_126816


namespace NUMINAMATH_CALUDE_oliver_old_cards_l1268_126866

/-- Calculates the number of old baseball cards Oliver had. -/
def old_cards (cards_per_page : ℕ) (new_cards : ℕ) (pages_used : ℕ) : ℕ :=
  cards_per_page * pages_used - new_cards

/-- Theorem stating that Oliver had 10 old cards. -/
theorem oliver_old_cards : old_cards 3 2 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_oliver_old_cards_l1268_126866


namespace NUMINAMATH_CALUDE_connecting_line_is_correct_l1268_126878

/-- The equation of a circle in the form (x-h)^2 + (y-k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given two circles, returns the line connecting their centers -/
def line_connecting_centers (c1 c2 : Circle) : Line :=
  sorry

/-- The first circle: x^2+y^2-4x+6y=0 -/
def circle1 : Circle :=
  { h := 2, k := -3, r := 5 }

/-- The second circle: x^2+y^2-6x=0 -/
def circle2 : Circle :=
  { h := 3, k := 0, r := 3 }

/-- The expected line: 3x-y-9=0 -/
def expected_line : Line :=
  { a := 3, b := -1, c := -9 }

theorem connecting_line_is_correct :
  line_connecting_centers circle1 circle2 = expected_line :=
sorry

end NUMINAMATH_CALUDE_connecting_line_is_correct_l1268_126878


namespace NUMINAMATH_CALUDE_afternoon_letters_indeterminate_l1268_126826

/-- Represents the number of items Jack received at different times of the day -/
structure JacksItems where
  morning_emails : ℕ
  morning_letters : ℕ
  afternoon_emails : ℕ
  afternoon_letters : ℕ

/-- The given conditions about Jack's received items -/
def jack_conditions (items : JacksItems) : Prop :=
  items.morning_emails = 10 ∧
  items.morning_letters = 12 ∧
  items.afternoon_emails = 3 ∧
  items.morning_emails = items.afternoon_emails + 7

/-- Theorem stating that the number of afternoon letters cannot be determined -/
theorem afternoon_letters_indeterminate (items : JacksItems) 
  (h : jack_conditions items) : 
  ¬∃ (n : ℕ), ∀ (items' : JacksItems), 
    jack_conditions items' → items'.afternoon_letters = n :=
sorry

end NUMINAMATH_CALUDE_afternoon_letters_indeterminate_l1268_126826


namespace NUMINAMATH_CALUDE_connie_tickets_l1268_126877

/-- The number of tickets Connie redeemed -/
def total_tickets : ℕ := 50

/-- The number of tickets spent on earbuds -/
def earbuds_tickets : ℕ := 10

/-- The number of tickets spent on glow bracelets -/
def glow_bracelets_tickets : ℕ := 15

/-- Theorem stating that Connie redeemed 50 tickets -/
theorem connie_tickets : 
  (total_tickets / 2 : ℚ) + earbuds_tickets + glow_bracelets_tickets = total_tickets := by
  sorry

#check connie_tickets

end NUMINAMATH_CALUDE_connie_tickets_l1268_126877


namespace NUMINAMATH_CALUDE_second_square_area_is_676_l1268_126899

/-- An isosceles right triangle with inscribed squares -/
structure TriangleWithSquares where
  /-- Side length of the first inscribed square -/
  a : ℝ
  /-- Area of the first inscribed square is 169 -/
  h_area : a^2 = 169

/-- The area of the second inscribed square -/
def second_square_area (t : TriangleWithSquares) : ℝ :=
  (2 * t.a)^2

theorem second_square_area_is_676 (t : TriangleWithSquares) :
  second_square_area t = 676 := by
  sorry

end NUMINAMATH_CALUDE_second_square_area_is_676_l1268_126899


namespace NUMINAMATH_CALUDE_camping_trip_purchases_l1268_126884

/-- Given Rebecca's camping trip purchases, prove the difference between water bottles and tent stakes --/
theorem camping_trip_purchases (total_items tent_stakes drink_mix water_bottles : ℕ) : 
  total_items = 22 →
  tent_stakes = 4 →
  drink_mix = 3 * tent_stakes →
  total_items = tent_stakes + drink_mix + water_bottles →
  water_bottles - tent_stakes = 2 := by
  sorry

end NUMINAMATH_CALUDE_camping_trip_purchases_l1268_126884


namespace NUMINAMATH_CALUDE_journey_distance_l1268_126882

theorem journey_distance (total_distance : ℝ) (bike_speed walking_speed : ℝ) 
  (h1 : bike_speed = 12)
  (h2 : walking_speed = 4)
  (h3 : (3/4 * total_distance) / bike_speed + (1/4 * total_distance) / walking_speed = 1) :
  1/4 * total_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l1268_126882


namespace NUMINAMATH_CALUDE_smallest_number_l1268_126849

theorem smallest_number (S : Set ℤ) (h : S = {-4, -2, 0, 1}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = -4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1268_126849


namespace NUMINAMATH_CALUDE_point_outside_circle_l1268_126800

/-- Given a line ax + by = 1 and a circle x^2 + y^2 = 1 that intersect at two distinct points,
    prove that the point (a, b) lies outside the circle. -/
theorem point_outside_circle (a b : ℝ) :
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) →
  (∃ x' y' : ℝ, x' ≠ y' ∧ a * x' + b * y' = 1 ∧ x'^2 + y'^2 = 1) →
  a^2 + b^2 > 1 :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1268_126800


namespace NUMINAMATH_CALUDE_equation_solution_l1268_126870

theorem equation_solution (a : ℝ) : (3 * 5 + 2 * a = 3) → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1268_126870


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l1268_126848

theorem inverse_proportion_k_value (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, x ≠ 0 → y x = k / x) →  -- Inverse proportion function
  y 3 = 2 →                     -- Passes through (3, 2)
  k = 6 :=                      -- Prove k = 6
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l1268_126848


namespace NUMINAMATH_CALUDE_trig_identity_l1268_126851

theorem trig_identity : Real.sin (63 * π / 180) * Real.cos (18 * π / 180) + 
  Real.cos (63 * π / 180) * Real.cos (108 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1268_126851


namespace NUMINAMATH_CALUDE_parallel_line_through_point_main_theorem_l1268_126853

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (given_line : Line) 
  (point : Point) 
  (result_line : Line) : Prop :=
  given_line.a = 2 ∧ 
  given_line.b = -1 ∧ 
  given_line.c = 1 ∧ 
  point.x = -1 ∧ 
  point.y = 0 ∧ 
  result_line.a = 2 ∧ 
  result_line.b = -1 ∧ 
  result_line.c = 2 ∧ 
  point.liesOn result_line ∧ 
  result_line.isParallel given_line

/-- The main theorem stating that the resulting line equation is correct -/
theorem main_theorem : ∃ (given_line result_line : Line) (point : Point), 
  parallel_line_through_point given_line point result_line := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_main_theorem_l1268_126853


namespace NUMINAMATH_CALUDE_weakly_decreasing_exp_weakly_decreasing_ln_condition_weakly_decreasing_cos_condition_l1268_126859

-- Definition of weakly decreasing function
def WeaklyDecreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  (∀ x ∈ I, ∀ y ∈ I, x < y → f x ≥ f y) ∧
  (∀ x ∈ I, ∀ y ∈ I, x < y → x * f x ≤ y * f y)

-- Theorem 1
theorem weakly_decreasing_exp (x : ℝ) :
  WeaklyDecreasing (fun x => x / Real.exp x) (Set.Ioo 1 2) :=
sorry

-- Theorem 2
theorem weakly_decreasing_ln_condition (m : ℝ) :
  WeaklyDecreasing (fun x => Real.log x / x) (Set.Ioi m) → m ≥ Real.exp 1 :=
sorry

-- Theorem 3
theorem weakly_decreasing_cos_condition (k : ℝ) :
  WeaklyDecreasing (fun x => Real.cos x + k * x^2) (Set.Ioo 0 (Real.pi / 2)) →
  2 / (3 * Real.pi) ≤ k ∧ k ≤ 1 / Real.pi :=
sorry

end NUMINAMATH_CALUDE_weakly_decreasing_exp_weakly_decreasing_ln_condition_weakly_decreasing_cos_condition_l1268_126859


namespace NUMINAMATH_CALUDE_solution_existence_condition_l1268_126839

theorem solution_existence_condition (m : ℝ) : 
  (∃ x ∈ Set.Icc 0 2, x^3 - 3*x + m = 0) → m ≤ 2 ∧ 
  ¬(∀ m ≤ 2, ∃ x ∈ Set.Icc 0 2, x^3 - 3*x + m = 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_existence_condition_l1268_126839


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1268_126821

theorem complex_equation_sum (x y : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (x - 2 : ℂ) + y * i = -1 + i) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1268_126821


namespace NUMINAMATH_CALUDE_mistaken_division_l1268_126843

theorem mistaken_division (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 21 = 36) :
  D / 12 = 63 := by
sorry

end NUMINAMATH_CALUDE_mistaken_division_l1268_126843


namespace NUMINAMATH_CALUDE_john_marathon_remainder_l1268_126806

/-- The length of a marathon in miles -/
def marathon_miles : ℕ := 26

/-- The additional length of a marathon in yards -/
def marathon_extra_yards : ℕ := 385

/-- The number of yards in a mile -/
def yards_per_mile : ℕ := 1760

/-- The number of marathons John has run -/
def john_marathons : ℕ := 15

/-- Theorem stating that the remainder of yards after converting the total distance of John's marathons to miles is 495 -/
theorem john_marathon_remainder :
  (john_marathons * (marathon_miles * yards_per_mile + marathon_extra_yards)) % yards_per_mile = 495 := by
  sorry

end NUMINAMATH_CALUDE_john_marathon_remainder_l1268_126806


namespace NUMINAMATH_CALUDE_area_of_graph_region_l1268_126887

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop :=
  |x - 80| + |y| = |x / 5|

/-- The region enclosed by the graph -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | graph_equation p.1 p.2}

/-- The area of the enclosed region -/
noncomputable def area_of_region : ℝ := sorry

theorem area_of_graph_region :
  area_of_region = 800 :=
sorry

end NUMINAMATH_CALUDE_area_of_graph_region_l1268_126887


namespace NUMINAMATH_CALUDE_number_division_problem_l1268_126855

theorem number_division_problem : ∃ N : ℕ, 
  (N / (555 + 445) = 2 * (555 - 445)) ∧ 
  (N % (555 + 445) = 30) ∧ 
  (N = 220030) := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l1268_126855


namespace NUMINAMATH_CALUDE_sin_120_degrees_l1268_126891

theorem sin_120_degrees : Real.sin (120 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l1268_126891


namespace NUMINAMATH_CALUDE_library_visitors_average_l1268_126894

/-- Calculates the average number of visitors per day in a 30-day month starting with a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays : ℕ := 5
  let totalOtherDays : ℕ := 25
  let totalDays : ℕ := 30
  let totalVisitors : ℕ := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  (totalVisitors : ℚ) / totalDays

theorem library_visitors_average (sundayVisitors otherDayVisitors : ℕ) 
  (h1 : sundayVisitors = 540) (h2 : otherDayVisitors = 240) : 
  averageVisitorsPerDay sundayVisitors otherDayVisitors = 290 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l1268_126894


namespace NUMINAMATH_CALUDE_total_cost_proof_l1268_126886

def price_AVN : ℝ := 12
def price_TheDark : ℝ := 2 * price_AVN
def num_TheDark : ℕ := 2
def num_AVN : ℕ := 1
def ratio_90s : ℝ := 0.4
def num_90s : ℕ := 5

theorem total_cost_proof :
  let cost_main := price_TheDark * num_TheDark + price_AVN * num_AVN
  let cost_90s := ratio_90s * cost_main * num_90s
  cost_main + cost_90s = 180 := by sorry

end NUMINAMATH_CALUDE_total_cost_proof_l1268_126886


namespace NUMINAMATH_CALUDE_dividend_calculation_l1268_126820

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 14)
  (h2 : quotient = 12)
  (h3 : remainder = 8) :
  divisor * quotient + remainder = 176 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1268_126820


namespace NUMINAMATH_CALUDE_problem_solving_probability_l1268_126808

theorem problem_solving_probability 
  (xavier_prob : ℚ) 
  (yvonne_prob : ℚ) 
  (zelda_prob : ℚ) 
  (hx : xavier_prob = 1/6)
  (hy : yvonne_prob = 1/2)
  (hz : zelda_prob = 5/8) :
  xavier_prob * yvonne_prob * (1 - zelda_prob) = 1/32 := by
sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l1268_126808


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l1268_126812

/-- Given two positive integers a and b in ratio 4:5 with LCM 180, prove that a = 36 -/
theorem smaller_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / b = 4 / 5 → 
  Nat.lcm a b = 180 → 
  a = 36 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l1268_126812


namespace NUMINAMATH_CALUDE_star_arrangement_count_l1268_126850

/-- The number of distinct arrangements of 12 objects on a regular six-pointed star -/
def star_arrangements : ℕ := 479001600

/-- The number of rotational and reflectional symmetries of a regular six-pointed star -/
def star_symmetries : ℕ := 12

/-- The total number of ways to arrange 12 objects in 12 positions -/
def total_arrangements : ℕ := Nat.factorial 12

theorem star_arrangement_count : 
  star_arrangements = total_arrangements / star_symmetries := by
  sorry

end NUMINAMATH_CALUDE_star_arrangement_count_l1268_126850


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1268_126811

/-- In a class of students, where half of the number of girls equals one-third of the total number of students, 
    the ratio of boys to girls is 1:2. -/
theorem boys_to_girls_ratio (S : ℕ) (B G : ℕ) : 
  S > 0 → 
  S = B + G → 
  (G : ℚ) / 2 = (S : ℚ) / 3 → 
  (B : ℚ) / G = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1268_126811


namespace NUMINAMATH_CALUDE_problem_solution_l1268_126868

-- Define the set M
def M : Set ℝ := {x | x^2 - x - 2 < 0}

-- Define the set N
def N (a b : ℝ) : Set ℝ := {x | a < x ∧ x < b}

theorem problem_solution :
  -- Part 1: M = (-1, 2)
  M = Set.Ioo (-1) 2 ∧
  -- Part 2: If M ⊇ N, then the minimum value of a is -1
  (∀ a b : ℝ, M ⊇ N a b → a ≥ -1) ∧
  (∃ a₀ : ℝ, a₀ = -1 ∧ ∃ b : ℝ, M ⊇ N a₀ b) ∧
  -- Part 3: If M ∩ N = M, then b ∈ [2, +∞)
  (∀ a b : ℝ, M ∩ N a b = M → b ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1268_126868


namespace NUMINAMATH_CALUDE_vectors_are_coplanar_l1268_126833

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
  [Fact (finrank ℝ V = 3)]

def are_coplanar (v₁ v₂ v₃ : V) : Prop :=
  ∃ (a b c : ℝ), a • v₁ + b • v₂ + c • v₃ = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

theorem vectors_are_coplanar (a b : V) : are_coplanar a b (2 • a + 4 • b) := by
  sorry

end NUMINAMATH_CALUDE_vectors_are_coplanar_l1268_126833


namespace NUMINAMATH_CALUDE_congruence_solution_l1268_126856

theorem congruence_solution : 
  {x : ℤ | 20 ≤ x ∧ x ≤ 50 ∧ (6 * x + 5) % 10 = (-19) % 10} = 
  {21, 26, 31, 36, 41, 46} := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1268_126856


namespace NUMINAMATH_CALUDE_a_eq_b_sufficient_not_necessary_l1268_126803

/-- Set A defined by the quadratic inequality with parameter a -/
def set_A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - x + a ≤ 0}

/-- Set B defined by the quadratic inequality with parameter b -/
def set_B (b : ℝ) : Set ℝ := {x : ℝ | x^2 - x + b ≤ 0}

/-- Theorem stating that a = b is sufficient but not necessary for A = B -/
theorem a_eq_b_sufficient_not_necessary :
  (∀ a b : ℝ, a = b → set_A a = set_B b) ∧
  (∃ a b : ℝ, a ≠ b ∧ set_A a = set_B b) := by
  sorry

end NUMINAMATH_CALUDE_a_eq_b_sufficient_not_necessary_l1268_126803


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l1268_126845

theorem absolute_value_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 10*a*b) :
  |((a + b) / (a - b))| = Real.sqrt (3/2) := by
sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l1268_126845


namespace NUMINAMATH_CALUDE_monday_production_tuesday_production_wednesday_production_thursday_pricing_l1268_126844

/-- Represents the recipe and conditions for making Zippies -/
structure ZippieRecipe where
  gluze_per_batch : ℚ
  blurpos_per_batch : ℚ
  zippies_per_batch : ℕ
  gluze_price : ℚ
  zippie_price : ℚ
  zippie_profit : ℚ

/-- The standard Zippie recipe -/
def standard_recipe : ZippieRecipe :=
  { gluze_per_batch := 4
  , blurpos_per_batch := 3
  , zippies_per_batch := 60
  , gluze_price := 1.8
  , zippie_price := 0.5
  , zippie_profit := 0.3 }

/-- Theorem for Monday's production -/
theorem monday_production (recipe : ZippieRecipe) (gluze_used : ℚ) :
  recipe = standard_recipe →
  gluze_used = 28 →
  gluze_used / recipe.gluze_per_batch * recipe.blurpos_per_batch = 21 :=
sorry

/-- Theorem for Tuesday's production -/
theorem tuesday_production (recipe : ZippieRecipe) (ingredient_used : ℚ) :
  recipe = standard_recipe →
  ingredient_used = 48 →
  (ingredient_used / recipe.gluze_per_batch * recipe.blurpos_per_batch = 36 ∨
   ingredient_used / recipe.blurpos_per_batch * recipe.gluze_per_batch = 64) :=
sorry

/-- Theorem for Wednesday's production -/
theorem wednesday_production (recipe : ZippieRecipe) (gluze_available blurpos_available : ℚ) :
  recipe = standard_recipe →
  gluze_available = 64 →
  blurpos_available = 42 →
  min (gluze_available / recipe.gluze_per_batch) (blurpos_available / recipe.blurpos_per_batch) * recipe.zippies_per_batch = 840 :=
sorry

/-- Theorem for Thursday's pricing -/
theorem thursday_pricing (recipe : ZippieRecipe) :
  recipe = standard_recipe →
  (recipe.zippie_price - recipe.zippie_profit) * recipe.zippies_per_batch - recipe.gluze_price * recipe.gluze_per_batch = 1.6 * recipe.blurpos_per_batch :=
sorry

end NUMINAMATH_CALUDE_monday_production_tuesday_production_wednesday_production_thursday_pricing_l1268_126844


namespace NUMINAMATH_CALUDE_fruit_arrangement_l1268_126893

inductive Fruit
| Apple
| Pear
| Orange
| Banana

inductive Box
| One
| Two
| Three
| Four

def Arrangement := Box → Fruit

def label1 (a : Arrangement) : Prop := a Box.One = Fruit.Orange
def label2 (a : Arrangement) : Prop := a Box.Two = Fruit.Pear
def label3 (a : Arrangement) : Prop := a Box.One = Fruit.Banana → (a Box.Three = Fruit.Apple ∨ a Box.Three = Fruit.Pear)
def label4 (a : Arrangement) : Prop := a Box.Four = Fruit.Apple

def all_labels_false (a : Arrangement) : Prop :=
  ¬label1 a ∧ ¬label2 a ∧ ¬label3 a ∧ ¬label4 a

def correct_arrangement (a : Arrangement) : Prop :=
  a Box.One = Fruit.Banana ∧
  a Box.Two = Fruit.Apple ∧
  a Box.Three = Fruit.Orange ∧
  a Box.Four = Fruit.Pear

theorem fruit_arrangement :
  ∀ a : Arrangement,
    (∀ f : Fruit, ∃! b : Box, a b = f) →
    all_labels_false a →
    correct_arrangement a :=
by sorry

end NUMINAMATH_CALUDE_fruit_arrangement_l1268_126893


namespace NUMINAMATH_CALUDE_pension_calculation_l1268_126867

-- Define the pension function
noncomputable def pension (k : ℝ) (x : ℝ) : ℝ := k * Real.sqrt x

-- Define the problem parameters
variable (c d r s : ℝ)

-- State the theorem
theorem pension_calculation (h1 : d ≠ c) 
                            (h2 : pension k x - pension k (x - c) = r) 
                            (h3 : pension k x - pension k (x - d) = s) : 
  pension k x = (r^2 - s^2) / (2 * (r - s)) := by
  sorry

end NUMINAMATH_CALUDE_pension_calculation_l1268_126867


namespace NUMINAMATH_CALUDE_tangent_circles_a_value_l1268_126847

/-- Circle C₁ with equation x² + y² = 16 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 16}

/-- Circle C₂ with equation (x - a)² + y² = 1, parameterized by a -/
def C₂ (a : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + p.2^2 = 1}

/-- Two circles are tangent if they intersect at exactly one point -/
def are_tangent (S T : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ S ∧ p ∈ T

/-- The main theorem: if C₁ and C₂(a) are tangent, then a = ±5 or a = ±3 -/
theorem tangent_circles_a_value :
  ∀ a : ℝ, are_tangent C₁ (C₂ a) → a = 5 ∨ a = -5 ∨ a = 3 ∨ a = -3 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_a_value_l1268_126847


namespace NUMINAMATH_CALUDE_x_value_possibilities_l1268_126896

theorem x_value_possibilities (x y p q : ℝ) (h1 : y ≠ 0) (h2 : q ≠ 0) 
  (h3 : |x / y| < |p| / q^2) :
  ∃ (x_neg x_zero x_pos : ℝ), 
    (x_neg < 0 ∧ |x_neg / y| < |p| / q^2) ∧
    (x_zero = 0 ∧ |x_zero / y| < |p| / q^2) ∧
    (x_pos > 0 ∧ |x_pos / y| < |p| / q^2) :=
by sorry

end NUMINAMATH_CALUDE_x_value_possibilities_l1268_126896


namespace NUMINAMATH_CALUDE_prob_at_least_one_karnataka_is_five_sixths_l1268_126898

/-- The probability of selecting at least one student from Karnataka -/
def prob_at_least_one_karnataka : ℚ :=
  let total_students : ℕ := 10
  let maharashtra_students : ℕ := 4
  let goa_students : ℕ := 3
  let karnataka_students : ℕ := 3
  let students_to_select : ℕ := 4
  1 - (Nat.choose (total_students - karnataka_students) students_to_select : ℚ) / 
      (Nat.choose total_students students_to_select : ℚ)

/-- Theorem stating that the probability of selecting at least one student from Karnataka is 5/6 -/
theorem prob_at_least_one_karnataka_is_five_sixths :
  prob_at_least_one_karnataka = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_karnataka_is_five_sixths_l1268_126898


namespace NUMINAMATH_CALUDE_absolute_value_v_l1268_126863

theorem absolute_value_v (u v : ℂ) : 
  u * v = 20 - 15 * I → Complex.abs u = 5 → Complex.abs v = 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_v_l1268_126863


namespace NUMINAMATH_CALUDE_scientific_notation_of_56_99_million_l1268_126895

theorem scientific_notation_of_56_99_million :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    56990000 = a * (10 : ℝ) ^ n ∧
    a = 5.699 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_56_99_million_l1268_126895


namespace NUMINAMATH_CALUDE_exponent_division_l1268_126858

theorem exponent_division (a : ℝ) : a^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1268_126858


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1268_126875

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 8*x^6 + 24*x^4 - 32*x^2 + 16 = (x - Real.sqrt 2)^4 * (x + Real.sqrt 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1268_126875


namespace NUMINAMATH_CALUDE_solution_set_part_i_solution_set_part_ii_l1268_126829

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 3| - 2 * |x + a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f x 3 > 2} = {x : ℝ | -7 < x ∧ x < -5/3} := by sorry

-- Part II
theorem solution_set_part_ii (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) (-1), f x a + x + 1 ≤ 0) →
  a ≥ 4 ∨ a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_solution_set_part_ii_l1268_126829


namespace NUMINAMATH_CALUDE_tangent_slope_sin_pi_sixth_l1268_126890

theorem tangent_slope_sin_pi_sixth :
  let f : ℝ → ℝ := λ x ↦ Real.sin x
  (deriv f) (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_sin_pi_sixth_l1268_126890


namespace NUMINAMATH_CALUDE_exist_three_integers_sum_zero_thirteenth_powers_square_l1268_126883

theorem exist_three_integers_sum_zero_thirteenth_powers_square :
  ∃ (a b c : ℤ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧  -- nonzero
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧  -- pairwise distinct
    a + b + c = 0 ∧          -- sum is zero
    ∃ (n : ℕ), a^13 + b^13 + c^13 = n^2  -- sum of 13th powers is a perfect square
  := by sorry

end NUMINAMATH_CALUDE_exist_three_integers_sum_zero_thirteenth_powers_square_l1268_126883


namespace NUMINAMATH_CALUDE_equation_solution_l1268_126892

theorem equation_solution : ∃! x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4) ∧ x = -14 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1268_126892


namespace NUMINAMATH_CALUDE_linear_function_properties_l1268_126852

-- Define the linear function
def f (x : ℝ) : ℝ := -2 * x + 3

-- Theorem statement
theorem linear_function_properties :
  (f 1 = 1) ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ f x = y) ∧
  (f (3/2) = 0) ∧
  (∀ (x₁ x₂ : ℝ), x₁ < x₂ → f x₁ > f x₂) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1268_126852


namespace NUMINAMATH_CALUDE_books_sold_l1268_126840

theorem books_sold (total_books : ℕ) (fraction_left : ℚ) (books_sold : ℕ) : 
  total_books = 15750 →
  fraction_left = 7 / 23 →
  books_sold = total_books - (total_books * fraction_left).floor →
  books_sold = 10957 := by
sorry

end NUMINAMATH_CALUDE_books_sold_l1268_126840


namespace NUMINAMATH_CALUDE_john_initial_diamonds_l1268_126873

/-- Represents the number of diamonds each pirate has -/
structure DiamondCount where
  bill : ℕ
  sam : ℕ
  john : ℕ

/-- Represents the average mass of diamonds for each pirate -/
structure AverageMass where
  bill : ℝ
  sam : ℝ
  john : ℝ

/-- The initial distribution of diamonds -/
def initial_distribution : DiamondCount :=
  { bill := 12, sam := 12, john := 9 }

/-- The distribution after the theft events -/
def final_distribution : DiamondCount :=
  { bill := initial_distribution.bill,
    sam := initial_distribution.sam,
    john := initial_distribution.john }

/-- The change in average mass for each pirate -/
def mass_change : AverageMass :=
  { bill := -1, sam := -2, john := 4 }

theorem john_initial_diamonds :
  initial_distribution.john = 9 →
  (initial_distribution.bill * mass_change.bill +
   initial_distribution.sam * mass_change.sam +
   initial_distribution.john * mass_change.john = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_john_initial_diamonds_l1268_126873


namespace NUMINAMATH_CALUDE_small_circle_radius_l1268_126838

theorem small_circle_radius (R : ℝ) (r : ℝ) : 
  R = 10 →  -- radius of the large circle is 10 meters
  4 * (2 * r) = 2 * R →  -- four diameters of small circles equal the diameter of the large circle
  r = 2.5 :=  -- radius of each small circle is 2.5 meters
by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l1268_126838


namespace NUMINAMATH_CALUDE_integral_rational_function_l1268_126862

open Real

theorem integral_rational_function (x : ℝ) :
  deriv (fun x => (1/2) * log (x^2 + 2*x + 5) + (1/2) * arctan ((x + 1)/2)) x
  = (x + 2) / (x^2 + 2*x + 5) := by sorry

end NUMINAMATH_CALUDE_integral_rational_function_l1268_126862


namespace NUMINAMATH_CALUDE_angle_terminal_side_l1268_126822

theorem angle_terminal_side (x : ℝ) (α : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (x, 4) ∧ P.1 = x ∧ P.2 = 4) → 
  Real.sin α = 4/5 → 
  x = 3 ∨ x = -3 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l1268_126822


namespace NUMINAMATH_CALUDE_right_triangle_area_l1268_126860

theorem right_triangle_area (base height hypotenuse : ℝ) :
  base = 12 →
  hypotenuse = 13 →
  base^2 + height^2 = hypotenuse^2 →
  (1/2) * base * height = 30 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1268_126860


namespace NUMINAMATH_CALUDE_zeros_of_f_l1268_126835

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 2*x - 3 else -2 + Real.log x

-- State the theorem about the zeros of f
theorem zeros_of_f :
  ∃ (x₁ x₂ : ℝ), x₁ = -1 ∧ x₂ = Real.exp 2 ∧
  (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_l1268_126835


namespace NUMINAMATH_CALUDE_chess_games_ratio_l1268_126897

/-- Given a chess player who played 44 games and won 16 of them, 
    prove that the ratio of games lost to games won is 7:4 -/
theorem chess_games_ratio (total_games : ℕ) (games_won : ℕ) 
  (h1 : total_games = 44) (h2 : games_won = 16) :
  (total_games - games_won) / games_won = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_chess_games_ratio_l1268_126897


namespace NUMINAMATH_CALUDE_time_to_walk_five_miles_l1268_126807

/-- Given that Tom walks 2 miles in 6 minutes, prove that it takes 15 minutes to walk 5 miles at the same rate. -/
theorem time_to_walk_five_miles (distance_to_jerry : ℝ) (time_to_jerry : ℝ) (distance_to_sam : ℝ) :
  distance_to_jerry = 2 →
  time_to_jerry = 6 →
  distance_to_sam = 5 →
  (distance_to_sam / (distance_to_jerry / time_to_jerry)) = 15 := by
sorry

end NUMINAMATH_CALUDE_time_to_walk_five_miles_l1268_126807


namespace NUMINAMATH_CALUDE_distinct_numbers_squared_differences_l1268_126865

theorem distinct_numbers_squared_differences (n : ℕ) (a : Fin n → ℝ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) (h_n : n = 10) : 
  {x | ∃ i j, i < j ∧ x = (a j - a i)^2} ≠ 
  {y | ∃ i j, i < j ∧ y = |a j^2 - a i^2|} :=
sorry

end NUMINAMATH_CALUDE_distinct_numbers_squared_differences_l1268_126865
