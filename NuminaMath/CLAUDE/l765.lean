import Mathlib

namespace NUMINAMATH_CALUDE_least_five_digit_palindrome_div_25_l765_76531

/-- A function that checks if a natural number is a five-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The theorem stating that 10201 is the least five-digit palindrome divisible by 25 -/
theorem least_five_digit_palindrome_div_25 :
  ∀ n : ℕ, is_five_digit_palindrome n ∧ n % 25 = 0 → n ≥ 10201 :=
sorry

end NUMINAMATH_CALUDE_least_five_digit_palindrome_div_25_l765_76531


namespace NUMINAMATH_CALUDE_line_point_ratio_l765_76515

/-- Given four points A, B, C, D on a directed line such that AC/CB + AD/DB = 0,
    prove that 1/AC + 1/AD = 2/AB -/
theorem line_point_ratio (A B C D : ℝ) (h : (C - A) / (B - C) + (D - A) / (B - D) = 0) :
  1 / (C - A) + 1 / (D - A) = 2 / (B - A) := by
  sorry

end NUMINAMATH_CALUDE_line_point_ratio_l765_76515


namespace NUMINAMATH_CALUDE_sqrt_and_pi_comparisons_l765_76570

theorem sqrt_and_pi_comparisons : 
  (Real.sqrt 2 < Real.sqrt 3) ∧ (3.14 < Real.pi) := by sorry

end NUMINAMATH_CALUDE_sqrt_and_pi_comparisons_l765_76570


namespace NUMINAMATH_CALUDE_stratified_sample_school_b_l765_76502

/-- Represents the number of students in each school -/
structure SchoolPopulation where
  a : ℕ  -- Number of students in school A
  b : ℕ  -- Number of students in school B
  c : ℕ  -- Number of students in school C

/-- The total number of students across all three schools -/
def total_students : ℕ := 1500

/-- The size of the sample to be drawn -/
def sample_size : ℕ := 120

/-- Checks if the given school population forms an arithmetic sequence -/
def is_arithmetic_sequence (pop : SchoolPopulation) : Prop :=
  pop.b - pop.a = pop.c - pop.b

/-- Checks if the given school population sums to the total number of students -/
def is_valid_population (pop : SchoolPopulation) : Prop :=
  pop.a + pop.b + pop.c = total_students

/-- Calculates the number of students to be sampled from a given school -/
def stratified_sample_size (school_size : ℕ) : ℕ :=
  school_size * sample_size / total_students

/-- The main theorem: proves that the number of students to be sampled from school B is 40 -/
theorem stratified_sample_school_b :
  ∀ pop : SchoolPopulation,
  is_arithmetic_sequence pop →
  is_valid_population pop →
  stratified_sample_size pop.b = 40 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_school_b_l765_76502


namespace NUMINAMATH_CALUDE_hyperbola_equation_l765_76510

/-- Given a hyperbola with eccentricity e = √6/2 and the area of rectangle OMPN equal to √2,
    which is also equal to (1/2)ab, prove that the equation of the hyperbola is x^2/4 - y^2/2 = 1. -/
theorem hyperbola_equation (e a b : ℝ) (h1 : e = Real.sqrt 6 / 2) 
    (h2 : (1/2) * a * b = Real.sqrt 2) : 
    ∀ (x y : ℝ), x^2/4 - y^2/2 = 1 ↔ x^2/a^2 - y^2/b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l765_76510


namespace NUMINAMATH_CALUDE_andrew_work_days_l765_76584

/-- Given that Andrew worked 2.5 hours each day and 7.5 hours in total on his Science report,
    prove that he spent 3 days working on it. -/
theorem andrew_work_days (hours_per_day : ℝ) (total_hours : ℝ) (h1 : hours_per_day = 2.5) (h2 : total_hours = 7.5) :
  total_hours / hours_per_day = 3 := by
  sorry

end NUMINAMATH_CALUDE_andrew_work_days_l765_76584


namespace NUMINAMATH_CALUDE_solution_set_inequality_l765_76552

/-- Given that the solution set of x^2 - ax + b < 0 is (1,2), 
    prove that the solution set of 1/x < b/a is (-∞, 0) ∪ (3/2, +∞) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, x^2 - a*x + b < 0 ↔ 1 < x ∧ x < 2) →
  (∀ x, 1/x < b/a ↔ x < 0 ∨ 3/2 < x) := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l765_76552


namespace NUMINAMATH_CALUDE_decimal_to_binary_nineteen_l765_76528

theorem decimal_to_binary_nineteen : 
  (1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 19 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_nineteen_l765_76528


namespace NUMINAMATH_CALUDE_sum_of_areas_l765_76541

/-- A sequence of circles tangent to two half-lines -/
structure TangentCircles where
  d₁ : ℝ
  r₁ : ℝ
  d : ℕ → ℝ
  r : ℕ → ℝ
  h₁ : d₁ > 0
  h₂ : r₁ > 0
  h₃ : ∀ n : ℕ, d n > 0
  h₄ : ∀ n : ℕ, r n > 0
  h₅ : d 1 = d₁
  h₆ : r 1 = r₁
  h₇ : ∀ n : ℕ, n > 1 → d n < d (n-1)
  h₈ : ∀ n : ℕ, r n / d n = r₁ / d₁

theorem sum_of_areas (tc : TangentCircles) :
  (∑' n, π * (tc.r n)^2) = (π/4) * (tc.r₁ * (tc.d₁ + tc.r₁)^2 / tc.d₁) :=
sorry

end NUMINAMATH_CALUDE_sum_of_areas_l765_76541


namespace NUMINAMATH_CALUDE_same_name_existence_l765_76589

/-- Represents a child in the class -/
structure Child where
  forename : Nat
  surname : Nat

/-- The problem statement -/
theorem same_name_existence 
  (children : Finset Child) 
  (h_count : children.card = 33) 
  (h_range : ∀ c ∈ children, c.forename ≤ 10 ∧ c.surname ≤ 10) 
  (h_appear : ∀ n : Nat, n ≤ 10 → 
    (∃ c ∈ children, c.forename = n) ∧ 
    (∃ c ∈ children, c.surname = n)) :
  ∃ c1 c2 : Child, c1 ∈ children ∧ c2 ∈ children ∧ c1 ≠ c2 ∧ 
    c1.forename = c2.forename ∧ c1.surname = c2.surname :=
sorry

end NUMINAMATH_CALUDE_same_name_existence_l765_76589


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l765_76575

/-- The ratio of area to perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let s : ℝ := 10
  let perimeter : ℝ := 3 * s
  let area : ℝ := (Real.sqrt 3 / 4) * s^2
  area / perimeter = 5 * Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l765_76575


namespace NUMINAMATH_CALUDE_some_number_equation_l765_76585

theorem some_number_equation (n : ℤ) (y : ℤ) : 
  (n * (1 + y) + 17 = n * (-1 + y) - 21) → n = -19 := by
  sorry

end NUMINAMATH_CALUDE_some_number_equation_l765_76585


namespace NUMINAMATH_CALUDE_smallest_candy_count_l765_76555

theorem smallest_candy_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  (n + 6) % 9 = 0 ∧ 
  (n - 9) % 6 = 0 ∧ 
  (∀ m : ℕ, m ≥ 100 ∧ m ≤ 999 ∧ (m + 6) % 9 = 0 ∧ (m - 9) % 6 = 0 → m ≥ n) ∧
  n = 111 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l765_76555


namespace NUMINAMATH_CALUDE_todd_gum_problem_l765_76501

theorem todd_gum_problem (initial_gum : ℕ) (steve_gum : ℕ) (total_gum : ℕ) : 
  steve_gum = 16 → total_gum = 54 → total_gum = initial_gum + steve_gum → initial_gum = 38 := by
  sorry

end NUMINAMATH_CALUDE_todd_gum_problem_l765_76501


namespace NUMINAMATH_CALUDE_walking_problem_l765_76587

/-- Proves that the given conditions lead to the correct system of equations --/
theorem walking_problem (x y : ℝ) : 
  (∀ t : ℝ, t * x < t * y) → -- Xiao Wang walks faster than Xiao Zhang
  (3 * x + 210 = 5 * y) →    -- Distance condition after 3 and 5 minutes
  (10 * y - 10 * x = 100) →  -- Time and initial distance condition
  (∃ d : ℝ, d > 0 ∧ 10 * x = d ∧ 10 * y = d + 100) → -- Both reach the museum in 10 minutes
  (3 * x + 210 = 5 * y ∧ 10 * y - 10 * x = 100) :=
by sorry

end NUMINAMATH_CALUDE_walking_problem_l765_76587


namespace NUMINAMATH_CALUDE_mashed_potatoes_count_mashed_potatoes_proof_l765_76525

theorem mashed_potatoes_count : ℕ → ℕ → ℕ → Prop :=
  fun bacon_count difference mashed_count =>
    (bacon_count = 269) →
    (difference = 61) →
    (mashed_count = bacon_count + difference) →
    (mashed_count = 330)

-- The proof is omitted
theorem mashed_potatoes_proof : mashed_potatoes_count 269 61 330 := by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_count_mashed_potatoes_proof_l765_76525


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l765_76549

/-- Represents a truncated cone with horizontal bases -/
structure TruncatedCone where
  bottom_radius : ℝ
  top_radius : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Checks if a sphere is tangent to the truncated cone -/
def is_tangent (cone : TruncatedCone) (sphere : Sphere) : Prop :=
  -- This is a placeholder for the tangency condition
  True

theorem sphere_radius_in_truncated_cone (cone : TruncatedCone) (sphere : Sphere) :
  cone.bottom_radius = 10 ∧ 
  cone.top_radius = 3 ∧ 
  is_tangent cone sphere → 
  sphere.radius = Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l765_76549


namespace NUMINAMATH_CALUDE_line_circle_intersection_l765_76517

/-- Given a point (a, b) outside a circle and a line ax + by = r^2, 
    prove that the line intersects the circle but doesn't pass through the center. -/
theorem line_circle_intersection (a b r : ℝ) (h : a^2 + b^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ a*x + b*y = r^2 ∧ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l765_76517


namespace NUMINAMATH_CALUDE_problem_solving_questions_count_l765_76579

-- Define the total number of multiple-choice questions
def total_mc : ℕ := 35

-- Define the fraction of multiple-choice questions already written
def mc_written_fraction : ℚ := 2/5

-- Define the fraction of problem-solving questions already written
def ps_written_fraction : ℚ := 1/3

-- Define the total number of remaining questions to write
def remaining_questions : ℕ := 31

-- Theorem to prove
theorem problem_solving_questions_count :
  ∃ (total_ps : ℕ),
    (total_ps : ℚ) * (1 - ps_written_fraction) + 
    (total_mc : ℚ) * (1 - mc_written_fraction) = remaining_questions ∧
    total_ps = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_solving_questions_count_l765_76579


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l765_76594

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20250 → n + (n + 1) = 285 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l765_76594


namespace NUMINAMATH_CALUDE_connect_four_ratio_l765_76583

theorem connect_four_ratio (total_games won_games : ℕ) 
  (h1 : total_games = 30) 
  (h2 : won_games = 18) : 
  (won_games : ℚ) / (total_games - won_games) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_connect_four_ratio_l765_76583


namespace NUMINAMATH_CALUDE_M_mod_1500_l765_76521

/-- A sequence of positive integers whose binary representation has exactly 9 ones -/
def T : Nat → Nat := sorry

/-- The 1500th number in the sequence T -/
def M : Nat := T 1500

/-- The remainder when M is divided by 1500 -/
theorem M_mod_1500 : M % 1500 = 500 := by sorry

end NUMINAMATH_CALUDE_M_mod_1500_l765_76521


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_parallel_l765_76527

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpLine : Line → Line → Prop)
variable (perpLinePlane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the "contained in" relation
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_parallel
  (m n : Line) (α : Plane)
  (h1 : perpLinePlane m α)
  (h2 : perpLine n m)
  (h3 : ¬ containedIn n α) :
  parallel n α :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_parallel_l765_76527


namespace NUMINAMATH_CALUDE_correct_ages_unique_solution_l765_76562

/-- Represents the ages of family members -/
structure FamilyAges where
  father : ℕ
  son : ℕ
  brother : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  (ages.father + 15 = 3 * (ages.father - 25)) ∧
  (ages.father + 15 = 2 * (ages.son + 15)) ∧
  (ages.brother = (ages.father + 15) / 2 + 7)

/-- Theorem stating that the ages 45, 15, and 37 satisfy the problem conditions -/
theorem correct_ages : satisfiesConditions { father := 45, son := 15, brother := 37 } := by
  sorry

/-- Theorem stating the uniqueness of the solution -/
theorem unique_solution (ages : FamilyAges) :
  satisfiesConditions ages → ages = { father := 45, son := 15, brother := 37 } := by
  sorry

end NUMINAMATH_CALUDE_correct_ages_unique_solution_l765_76562


namespace NUMINAMATH_CALUDE_balanced_leaving_probability_formula_l765_76597

/-- The probability that 3n students leaving from 3 rows of n students, one at a time
    with all leaving orders equally likely, such that there are never two rows where
    the number of students remaining differs by 2 or more. -/
def balanced_leaving_probability (n : ℕ) : ℚ :=
  (6 * n * (n.factorial ^ 3 : ℚ)) / ((3 * n).factorial : ℚ)

/-- Theorem stating that the probability of balanced leaving for 3n students
    in 3 rows of n is equal to (6n * (n!)^3) / (3n)! -/
theorem balanced_leaving_probability_formula (n : ℕ) (h : n ≥ 1) :
  balanced_leaving_probability n = (6 * n * (n.factorial ^ 3 : ℚ)) / ((3 * n).factorial : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_balanced_leaving_probability_formula_l765_76597


namespace NUMINAMATH_CALUDE_trig_expression_equals_four_l765_76591

theorem trig_expression_equals_four : 
  (Real.sqrt 3 / Real.sin (20 * π / 180)) - (1 / Real.cos (20 * π / 180)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_four_l765_76591


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l765_76569

theorem quadratic_roots_product (a b : ℝ) : 
  (3 * a ^ 2 + 9 * a - 21 = 0) → 
  (3 * b ^ 2 + 9 * b - 21 = 0) → 
  (3 * a - 4) * (6 * b - 8) = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l765_76569


namespace NUMINAMATH_CALUDE_square_minus_product_equals_one_l765_76529

theorem square_minus_product_equals_one : 1999^2 - 2000 * 1998 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_equals_one_l765_76529


namespace NUMINAMATH_CALUDE_jerome_contacts_l765_76557

/-- The number of people on Jerome's contact list -/
def total_contacts (classmates out_of_school_friends parents sisters : ℕ) : ℕ :=
  classmates + out_of_school_friends + parents + sisters

/-- Theorem stating the total number of contacts on Jerome's list -/
theorem jerome_contacts : ∃ (classmates out_of_school_friends parents sisters : ℕ),
  classmates = 20 ∧
  out_of_school_friends = classmates / 2 ∧
  parents = 2 ∧
  sisters = 1 ∧
  total_contacts classmates out_of_school_friends parents sisters = 33 := by
  sorry

end NUMINAMATH_CALUDE_jerome_contacts_l765_76557


namespace NUMINAMATH_CALUDE_floor_sqrt_18_squared_l765_76554

theorem floor_sqrt_18_squared : ⌊Real.sqrt 18⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_18_squared_l765_76554


namespace NUMINAMATH_CALUDE_power_product_equality_l765_76503

theorem power_product_equality : 2^3 * 3 * 5^3 * 7 = 21000 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l765_76503


namespace NUMINAMATH_CALUDE_final_sequence_values_l765_76513

/-- The number of elements in the initial sequence -/
def n : ℕ := 2022

/-- Function to calculate the new value for a given position after one iteration -/
def newValue (i : ℕ) : ℕ := i^2 + 1

/-- The number of iterations required to reduce the sequence to two numbers -/
def iterations : ℕ := (n - 2) / 2

/-- The final two numbers in the sequence after all iterations -/
def finalPair : (ℕ × ℕ) := (newValue (n/2) + iterations, newValue (n/2 + 1) + iterations)

/-- Theorem stating the final two numbers in the sequence -/
theorem final_sequence_values :
  finalPair = (1023131, 1025154) := by sorry

end NUMINAMATH_CALUDE_final_sequence_values_l765_76513


namespace NUMINAMATH_CALUDE_dhoni_rent_percentage_dhoni_rent_percentage_proof_l765_76548

theorem dhoni_rent_percentage : ℝ → Prop :=
  fun rent_percentage =>
    let dishwasher_percentage := rent_percentage - 5
    let leftover_percentage := 61
    rent_percentage + dishwasher_percentage + leftover_percentage = 100 →
    rent_percentage = 22

-- The proof is omitted
theorem dhoni_rent_percentage_proof : dhoni_rent_percentage 22 := by sorry

end NUMINAMATH_CALUDE_dhoni_rent_percentage_dhoni_rent_percentage_proof_l765_76548


namespace NUMINAMATH_CALUDE_gratuity_percentage_l765_76526

theorem gratuity_percentage
  (num_people : ℕ)
  (total_bill : ℝ)
  (avg_cost_before_gratuity : ℝ)
  (h1 : num_people = 7)
  (h2 : total_bill = 840)
  (h3 : avg_cost_before_gratuity = 100) :
  (total_bill - num_people * avg_cost_before_gratuity) / (num_people * avg_cost_before_gratuity) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_gratuity_percentage_l765_76526


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l765_76577

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 - 9 = 0} = {-3, 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l765_76577


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l765_76568

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 2*a^2 - a + 3 = 0) → 
  (b^3 - 2*b^2 - b + 3 = 0) → 
  (c^3 - 2*c^2 - c + 3 = 0) → 
  a^3 + b^3 + c^3 = 5 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l765_76568


namespace NUMINAMATH_CALUDE_defeat_monster_time_l765_76539

/-- The time required to defeat a monster given the attack rates of two Ultramen and the monster's durability. -/
theorem defeat_monster_time 
  (monster_durability : ℕ) 
  (ultraman1_rate : ℕ) 
  (ultraman2_rate : ℕ) 
  (h1 : monster_durability = 100)
  (h2 : ultraman1_rate = 12)
  (h3 : ultraman2_rate = 8) : 
  (monster_durability : ℚ) / (ultraman1_rate + ultraman2_rate : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_defeat_monster_time_l765_76539


namespace NUMINAMATH_CALUDE_triangle_ap_tangent_relation_l765_76504

/-- A triangle with sides in arithmetic progression satisfies 3 * tan(α/2) * tan(γ/2) = 1, 
    where α is the smallest angle and γ is the largest angle. -/
theorem triangle_ap_tangent_relation (a b c : ℝ) (α β γ : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  α > 0 → β > 0 → γ > 0 →
  α + β + γ = π →
  a + c = 2 * b →  -- Arithmetic progression condition
  α ≤ β → β ≤ γ →  -- α is smallest, γ is largest
  3 * Real.tan (α / 2) * Real.tan (γ / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ap_tangent_relation_l765_76504


namespace NUMINAMATH_CALUDE_bumper_car_line_problem_l765_76519

theorem bumper_car_line_problem (initial_people : ℕ) : 
  (initial_people - 10 + 5 = 25) → initial_people = 30 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_problem_l765_76519


namespace NUMINAMATH_CALUDE_tina_crayon_selection_ways_l765_76598

/-- The number of different-colored crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of crayons Tina must select -/
def selected_crayons : ℕ := 6

/-- The number of mandatory crayons (red and blue) -/
def mandatory_crayons : ℕ := 2

/-- Computes the number of ways to select k items from n items -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The main theorem stating the number of ways Tina can select the crayons -/
theorem tina_crayon_selection_ways :
  combinations (total_crayons - mandatory_crayons) (selected_crayons - mandatory_crayons) = 715 := by
  sorry

end NUMINAMATH_CALUDE_tina_crayon_selection_ways_l765_76598


namespace NUMINAMATH_CALUDE_hours_worked_per_day_l765_76571

theorem hours_worked_per_day 
  (total_hours : ℕ) 
  (weeks_worked : ℕ) 
  (h1 : total_hours = 140) 
  (h2 : weeks_worked = 4) :
  (total_hours : ℚ) / (weeks_worked * 7) = 5 := by
  sorry

end NUMINAMATH_CALUDE_hours_worked_per_day_l765_76571


namespace NUMINAMATH_CALUDE_arctan_sum_l765_76500

theorem arctan_sum : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_l765_76500


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l765_76578

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 6 →
  b = 8 →
  c^2 = a^2 + b^2 →
  c = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l765_76578


namespace NUMINAMATH_CALUDE_prob_unit_apart_value_l765_76507

/-- A rectangle with 10 points spaced at unit intervals on corners and edge midpoints -/
structure UnitRectangle :=
  (width : ℕ)
  (height : ℕ)
  (total_points : ℕ)
  (h_width : width = 5)
  (h_height : height = 2)
  (h_total_points : total_points = 10)

/-- The number of pairs of points that are exactly one unit apart -/
def unit_apart_pairs (r : UnitRectangle) : ℕ := 13

/-- The total number of ways to choose two points from the rectangle -/
def total_pairs (r : UnitRectangle) : ℕ := r.total_points.choose 2

/-- The probability of selecting two points that are exactly one unit apart -/
def prob_unit_apart (r : UnitRectangle) : ℚ :=
  (unit_apart_pairs r : ℚ) / (total_pairs r : ℚ)

theorem prob_unit_apart_value (r : UnitRectangle) :
  prob_unit_apart r = 13 / 45 := by sorry

end NUMINAMATH_CALUDE_prob_unit_apart_value_l765_76507


namespace NUMINAMATH_CALUDE_sales_price_ratio_l765_76563

/-- Proves the ratio of percent increase in units sold to combined percent decrease in price -/
theorem sales_price_ratio (P : ℝ) (U : ℝ) (h_positive : P > 0 ∧ U > 0) :
  let price_decrease := 0.20
  let additional_discount := 0.10
  let new_price := P * (1 - price_decrease)
  let new_units := U / (1 - price_decrease)
  let final_price := new_price * (1 - additional_discount)
  let percent_increase_units := (new_units - U) / U
  let percent_decrease_price := (P - final_price) / P
  (percent_increase_units / percent_decrease_price) = 1 / 1.12 :=
by
  sorry

end NUMINAMATH_CALUDE_sales_price_ratio_l765_76563


namespace NUMINAMATH_CALUDE_ellipse_smallest_area_l765_76537

/-- Given an ellipse that contains two specific circles, prove its smallest possible area -/
theorem ellipse_smallest_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
  ((x - 2)^2 + y^2 ≥ 4 ∧ (x + 2)^2 + y^2 ≥ 4)) :
  ∃ k : ℝ, k = Real.sqrt 3 / 3 ∧ 
    ∀ a' b' : ℝ, (∀ x y : ℝ, x^2/a'^2 + y^2/b'^2 = 1 → 
      ((x - 2)^2 + y^2 ≥ 4 ∧ (x + 2)^2 + y^2 ≥ 4)) →
    π * a' * b' ≥ k * π :=
by sorry

end NUMINAMATH_CALUDE_ellipse_smallest_area_l765_76537


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l765_76564

/-- Calculates the cost of tax-free items given total spend, sales tax, and tax rate -/
theorem tax_free_items_cost 
  (total_spend : ℝ) 
  (sales_tax : ℝ) 
  (tax_rate : ℝ) 
  (h1 : total_spend = 25)
  (h2 : sales_tax = 0.30)
  (h3 : tax_rate = 0.05) :
  total_spend - sales_tax / tax_rate = 19 := by
  sorry

#check tax_free_items_cost

end NUMINAMATH_CALUDE_tax_free_items_cost_l765_76564


namespace NUMINAMATH_CALUDE_shooting_scenarios_l765_76586

theorem shooting_scenarios (n : ℕ) (n₁ n₂ n₃ n₄ : ℕ) 
  (h_total : n = n₁ + n₂ + n₃ + n₄)
  (h_n : n = 10)
  (h_n₁ : n₁ = 2)
  (h_n₂ : n₂ = 4)
  (h_n₃ : n₃ = 3)
  (h_n₄ : n₄ = 1) :
  (Nat.factorial n) / (Nat.factorial n₁ * Nat.factorial n₂ * Nat.factorial n₃ * Nat.factorial n₄) = 12600 :=
by sorry

end NUMINAMATH_CALUDE_shooting_scenarios_l765_76586


namespace NUMINAMATH_CALUDE_nikki_movie_length_l765_76573

/-- The lengths of favorite movies for Joyce, Michael, Nikki, and Ryn satisfy certain conditions -/
structure MovieLengths where
  michael : ℝ
  joyce : ℝ
  nikki : ℝ
  ryn : ℝ
  joyce_longer : joyce = michael + 2
  nikki_triple : nikki = 3 * michael
  ryn_proportion : ryn = (4/5) * nikki
  total_length : michael + joyce + nikki + ryn = 76

/-- Given the conditions, Nikki's favorite movie is 30 hours long -/
theorem nikki_movie_length (m : MovieLengths) : m.nikki = 30 := by
  sorry

end NUMINAMATH_CALUDE_nikki_movie_length_l765_76573


namespace NUMINAMATH_CALUDE_intersection_implies_a_geq_two_l765_76518

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x < a}
def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 < 0}

-- State the theorem
theorem intersection_implies_a_geq_two (a : ℝ) (h : A a ∩ B = B) : a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_geq_two_l765_76518


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l765_76574

theorem rectangular_field_dimensions :
  ∀ (width length : ℝ),
  width > 0 →
  length = 2 * width →
  width * length = 800 →
  width = 20 ∧ length = 40 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l765_76574


namespace NUMINAMATH_CALUDE_sum_of_square_and_pentagon_angles_l765_76542

theorem sum_of_square_and_pentagon_angles : 
  let square_angle := 180 * (4 - 2) / 4
  let pentagon_angle := 180 * (5 - 2) / 5
  square_angle + pentagon_angle = 198 := by
sorry

end NUMINAMATH_CALUDE_sum_of_square_and_pentagon_angles_l765_76542


namespace NUMINAMATH_CALUDE_tile_relationship_l765_76520

theorem tile_relationship (r : ℕ) (w : ℕ) : 
  (3 ≤ r ∧ r ≤ 7) → 
  (
    (r = 3 ∧ w = 6) ∨
    (r = 4 ∧ w = 8) ∨
    (r = 5 ∧ w = 10) ∨
    (r = 6 ∧ w = 12) ∨
    (r = 7 ∧ w = 14)
  ) →
  w = 2 * r :=
by sorry

end NUMINAMATH_CALUDE_tile_relationship_l765_76520


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l765_76593

/-- The surface area of a cuboid created by three cubes of side length 8 cm -/
theorem cuboid_surface_area : 
  let cube_side : ℝ := 8
  let cuboid_length : ℝ := 3 * cube_side
  let cuboid_width : ℝ := cube_side
  let cuboid_height : ℝ := cube_side
  let surface_area : ℝ := 2 * (cuboid_length * cuboid_width + 
                               cuboid_length * cuboid_height + 
                               cuboid_width * cuboid_height)
  surface_area = 896 := by
sorry


end NUMINAMATH_CALUDE_cuboid_surface_area_l765_76593


namespace NUMINAMATH_CALUDE_container_volume_comparison_l765_76516

theorem container_volume_comparison (a r : ℝ) (ha : a > 0) (hr : r > 0) 
  (h_eq : (2 * a)^3 = (4/3) * Real.pi * r^3) : 
  (2*a + 2)^3 > (4/3) * Real.pi * (r + 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_comparison_l765_76516


namespace NUMINAMATH_CALUDE_classroom_lights_theorem_l765_76506

/-- The number of lamps in the classroom -/
def num_lamps : ℕ := 4

/-- The total number of possible states for the lights -/
def total_states : ℕ := 2^num_lamps

/-- The number of ways to turn on the lights, excluding the all-off state -/
def ways_to_turn_on : ℕ := total_states - 1

theorem classroom_lights_theorem : ways_to_turn_on = 15 := by
  sorry

end NUMINAMATH_CALUDE_classroom_lights_theorem_l765_76506


namespace NUMINAMATH_CALUDE_train_speed_l765_76535

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 240)
  (h2 : bridge_length = 150)
  (h3 : crossing_time = 20) :
  (train_length + bridge_length) / crossing_time = 19.5 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l765_76535


namespace NUMINAMATH_CALUDE_certain_number_is_sixteen_l765_76532

theorem certain_number_is_sixteen :
  ∃ x : ℝ, (213 * x = 3408) ∧ (21.3 * x = 340.8) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_sixteen_l765_76532


namespace NUMINAMATH_CALUDE_train_passes_jogger_l765_76540

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed train_speed : ℝ) (train_length initial_distance : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 210 →
  initial_distance = 200 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 41 :=
by sorry

end NUMINAMATH_CALUDE_train_passes_jogger_l765_76540


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l765_76534

theorem isosceles_triangle_vertex_angle (a b h : ℝ) : 
  a > 0 → b > 0 → h > 0 →
  a^2 = 3 * b * h →
  b = 2 * a * Real.cos (π / 4) →
  h = a * Real.sin (π / 4) →
  let vertex_angle := π - 2 * (π / 4)
  vertex_angle = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l765_76534


namespace NUMINAMATH_CALUDE_n_over_8_equals_2_pow_3997_l765_76595

theorem n_over_8_equals_2_pow_3997 (n : ℕ) : n = 16^1000 → n/8 = 2^3997 := by
  sorry

end NUMINAMATH_CALUDE_n_over_8_equals_2_pow_3997_l765_76595


namespace NUMINAMATH_CALUDE_not_all_greater_than_quarter_l765_76588

theorem not_all_greater_than_quarter (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬ ((1 - a) * b > 1/4 ∧ (1 - b) * c > 1/4 ∧ (1 - c) * a > 1/4) := by
  sorry

end NUMINAMATH_CALUDE_not_all_greater_than_quarter_l765_76588


namespace NUMINAMATH_CALUDE_three_n_equals_twenty_seven_l765_76572

theorem three_n_equals_twenty_seven (n : ℤ) : 3 * n = 9 + 9 + 9 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_three_n_equals_twenty_seven_l765_76572


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l765_76580

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![5, -2; 4, 3]
  Matrix.det A = 23 := by sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l765_76580


namespace NUMINAMATH_CALUDE_harriets_age_l765_76599

theorem harriets_age (peter_age harriet_age : ℕ) : 
  (peter_age + 4 = 2 * (harriet_age + 4)) →  -- Condition 1
  (peter_age = 60 / 2) →                     -- Conditions 2 and 3 combined
  harriet_age = 13 := by
sorry

end NUMINAMATH_CALUDE_harriets_age_l765_76599


namespace NUMINAMATH_CALUDE_boat_distance_l765_76547

/-- The distance covered by a boat given its speed in still water and the time taken to cover the same distance downstream and upstream. -/
theorem boat_distance (v : ℝ) (t_down t_up : ℝ) (h1 : v = 7) (h2 : t_down = 2) (h3 : t_up = 5) :
  ∃ (d : ℝ), d = 20 ∧ d = (v + (v * t_down - d) / t_down) * t_down ∧ d = (v - (v * t_up - d) / t_up) * t_up :=
sorry

end NUMINAMATH_CALUDE_boat_distance_l765_76547


namespace NUMINAMATH_CALUDE_antihomologous_properties_l765_76505

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Homothety center -/
def S : Point := sorry

/-- Given two circles satisfying the problem conditions -/
def circle1 : Circle := sorry
def circle2 : Circle := sorry

/-- Antihomologous points -/
def isAntihomologous (p q : Point) : Prop := sorry

/-- A point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop := sorry

/-- A circle is tangent to another circle -/
def isTangent (c1 c2 : Circle) : Prop := sorry

/-- Main theorem -/
theorem antihomologous_properties 
  (h1 : circle1.radius > circle2.radius)
  (h2 : isTangent circle1 circle2 ∨ 
        (circle1.center.1 - circle2.center.1)^2 + (circle1.center.2 - circle2.center.2)^2 
        > (circle1.radius + circle2.radius)^2) :
  (∀ (c : Circle) (p1 p2 p3 p4 : Point),
    isAntihomologous p1 p2 →
    onCircle p1 c ∧ onCircle p2 c →
    onCircle p3 circle1 ∧ onCircle p4 circle2 ∧ onCircle p3 c ∧ onCircle p4 c →
    isAntihomologous p3 p4) ∧
  (∀ (c : Circle),
    isTangent c circle1 ∧ isTangent c circle2 →
    ∃ (p1 p2 : Point),
      onCircle p1 circle1 ∧ onCircle p2 circle2 ∧
      onCircle p1 c ∧ onCircle p2 c ∧
      isAntihomologous p1 p2) :=
by sorry

end NUMINAMATH_CALUDE_antihomologous_properties_l765_76505


namespace NUMINAMATH_CALUDE_original_group_size_is_correct_l765_76509

/-- Represents the number of men in the original group -/
def original_group_size : ℕ := 22

/-- Represents the number of days the original group planned to work -/
def original_days : ℕ := 20

/-- Represents the number of men who became absent -/
def absent_men : ℕ := 2

/-- Represents the number of days the remaining group worked -/
def actual_days : ℕ := 22

/-- Theorem stating that the original group size is correct given the conditions -/
theorem original_group_size_is_correct :
  (original_group_size : ℚ) * (actual_days : ℚ) * ((original_group_size - absent_men) : ℚ) = 
  (original_group_size : ℚ) * (original_group_size : ℚ) * (original_days : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_is_correct_l765_76509


namespace NUMINAMATH_CALUDE_max_value_sqrt_expression_l765_76566

theorem max_value_sqrt_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 16) :
  Real.sqrt (x + 64) + Real.sqrt (16 - x) + 2 * Real.sqrt x ≤ 4 * Real.sqrt 5 + 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_expression_l765_76566


namespace NUMINAMATH_CALUDE_travel_expense_fraction_l765_76546

theorem travel_expense_fraction (initial_amount : ℝ) 
  (clothes_fraction : ℝ) (food_fraction : ℝ) (final_amount : ℝ) :
  initial_amount = 1499.9999999999998 →
  clothes_fraction = 1/3 →
  food_fraction = 1/5 →
  final_amount = 600 →
  let remaining_after_clothes := initial_amount * (1 - clothes_fraction)
  let remaining_after_food := remaining_after_clothes * (1 - food_fraction)
  (remaining_after_food - final_amount) / remaining_after_food = 1/4 := by
sorry

end NUMINAMATH_CALUDE_travel_expense_fraction_l765_76546


namespace NUMINAMATH_CALUDE_inclination_angle_PQ_l765_76561

/-- The inclination angle of a line PQ given two points P and Q -/
def inclination_angle (P Q : ℝ × ℝ) : ℝ := sorry

theorem inclination_angle_PQ :
  let P : ℝ × ℝ := (2 * Real.cos (10 * π / 180), 2 * Real.sin (10 * π / 180))
  let Q : ℝ × ℝ := (2 * Real.cos (50 * π / 180), -2 * Real.sin (50 * π / 180))
  inclination_angle P Q = 70 * π / 180 := by sorry

end NUMINAMATH_CALUDE_inclination_angle_PQ_l765_76561


namespace NUMINAMATH_CALUDE_least_k_cube_divisible_by_168_l765_76522

theorem least_k_cube_divisible_by_168 :
  ∀ k : ℕ, k > 0 → k^3 % 168 = 0 → k ≥ 42 :=
by sorry

end NUMINAMATH_CALUDE_least_k_cube_divisible_by_168_l765_76522


namespace NUMINAMATH_CALUDE_beta_value_l765_76560

theorem beta_value (α β : Real) 
  (eq : Real.sin α + Real.sin (α + β) + Real.cos (α + β) = Real.sqrt 3)
  (range : β ∈ Set.Icc (π / 4) π) : 
  β = π / 4 := by
sorry

end NUMINAMATH_CALUDE_beta_value_l765_76560


namespace NUMINAMATH_CALUDE_three_numbers_ratio_l765_76523

theorem three_numbers_ratio (F S T : ℚ) : 
  F + S + T = 550 → 
  S = 150 → 
  T = F / 3 → 
  F / S = 2 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_ratio_l765_76523


namespace NUMINAMATH_CALUDE_decreasing_function_a_range_l765_76530

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (5 * a - 1) * x + 4 * a else Real.log x / Real.log a

-- Theorem statement
theorem decreasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1 / 9 : ℝ) (1 / 5 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_decreasing_function_a_range_l765_76530


namespace NUMINAMATH_CALUDE_sum_equality_in_subset_l765_76581

theorem sum_equality_in_subset (S : Finset ℕ) :
  S ⊆ Finset.range 38 →
  S.card = 10 →
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b = c + d := by
  sorry

end NUMINAMATH_CALUDE_sum_equality_in_subset_l765_76581


namespace NUMINAMATH_CALUDE_all_star_seating_l765_76543

/-- Represents the number of ways to seat 9 baseball All-Stars from 3 teams -/
def seating_arrangements : ℕ :=
  let num_teams : ℕ := 3
  let players_per_team : ℕ := 3
  let team_arrangements : ℕ := Nat.factorial num_teams
  let within_team_arrangements : ℕ := Nat.factorial players_per_team
  team_arrangements * (within_team_arrangements ^ num_teams)

/-- Theorem stating the number of seating arrangements for 9 baseball All-Stars -/
theorem all_star_seating :
  seating_arrangements = 1296 := by
  sorry

end NUMINAMATH_CALUDE_all_star_seating_l765_76543


namespace NUMINAMATH_CALUDE_unique_four_digit_int_l765_76596

/-- Represents a four-digit positive integer --/
structure FourDigitInt where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  positive : 1000 ≤ a * 1000 + b * 100 + c * 10 + d

/-- The conditions given in the problem --/
def satisfiesConditions (n : FourDigitInt) : Prop :=
  n.a + n.b + n.c + n.d = 17 ∧
  n.b + n.c = 9 ∧
  n.a - n.d = 2 ∧
  (n.a * 1000 + n.b * 100 + n.c * 10 + n.d) % 9 = 0

/-- The theorem to be proved --/
theorem unique_four_digit_int :
  ∃! (n : FourDigitInt), satisfiesConditions n ∧ n.a = 5 ∧ n.b = 4 ∧ n.c = 5 ∧ n.d = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_four_digit_int_l765_76596


namespace NUMINAMATH_CALUDE_circle_radius_is_two_l765_76511

theorem circle_radius_is_two (r : ℝ) : r > 0 →
  3 * (2 * Real.pi * r) = 3 * (Real.pi * r^2) → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_two_l765_76511


namespace NUMINAMATH_CALUDE_children_ages_sum_l765_76544

theorem children_ages_sum (a b c d : ℕ) : 
  a < 18 ∧ b < 18 ∧ c < 18 ∧ d < 18 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 882 →
  a + b + c + d = 31 := by
  sorry

end NUMINAMATH_CALUDE_children_ages_sum_l765_76544


namespace NUMINAMATH_CALUDE_existence_of_sum_n_l765_76567

theorem existence_of_sum_n (n : ℕ) (seq1 seq2 : List ℕ) : 
  (∀ x ∈ seq1, x < n) →
  (∀ y ∈ seq2, y < n) →
  seq1.Nodup →
  seq2.Nodup →
  seq1.length + seq2.length ≥ n →
  ∃ x ∈ seq1, ∃ y ∈ seq2, x + y = n :=
by sorry

end NUMINAMATH_CALUDE_existence_of_sum_n_l765_76567


namespace NUMINAMATH_CALUDE_negation_equivalence_l765_76592

theorem negation_equivalence :
  (¬ ∀ a b : ℝ, a = b → a^2 = a*b) ↔ (∀ a b : ℝ, a ≠ b → a^2 ≠ a*b) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l765_76592


namespace NUMINAMATH_CALUDE_no_odd_multiples_of_6_or_8_up_to_60_l765_76538

theorem no_odd_multiples_of_6_or_8_up_to_60 : 
  ¬∃ n : ℕ, n ≤ 60 ∧ n % 2 = 1 ∧ (n % 6 = 0 ∨ n % 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_odd_multiples_of_6_or_8_up_to_60_l765_76538


namespace NUMINAMATH_CALUDE_ratio_equality_l765_76536

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 49)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l765_76536


namespace NUMINAMATH_CALUDE_josh_money_left_l765_76524

/-- The amount of money Josh has left after selling bracelets and buying cookies -/
def money_left (cost_per_bracelet : ℚ) (sell_price : ℚ) (num_bracelets : ℕ) (cookie_cost : ℚ) : ℚ :=
  (sell_price - cost_per_bracelet) * num_bracelets - cookie_cost

/-- Theorem stating that Josh has $3 left after selling bracelets and buying cookies -/
theorem josh_money_left :
  money_left 1 1.5 12 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_josh_money_left_l765_76524


namespace NUMINAMATH_CALUDE_solution_exists_in_interval_l765_76551

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- State the theorem
theorem solution_exists_in_interval :
  ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_in_interval_l765_76551


namespace NUMINAMATH_CALUDE_remainder_theorem_l765_76582

theorem remainder_theorem (w : ℤ) (h : (w + 3) % 11 = 0) : w % 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l765_76582


namespace NUMINAMATH_CALUDE_pencil_price_l765_76559

theorem pencil_price (total_cost : ℝ) (num_pens num_pencils : ℕ) (avg_pen_price : ℝ) :
  total_cost = 690 →
  num_pens = 30 →
  num_pencils = 75 →
  avg_pen_price = 18 →
  (total_cost - num_pens * avg_pen_price) / num_pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_l765_76559


namespace NUMINAMATH_CALUDE_unique_birth_date_l765_76514

/-- Represents a date in the 20th century -/
structure Date where
  day : Nat
  month : Nat
  year : Nat
  h1 : 1 ≤ day ∧ day ≤ 31
  h2 : 1 ≤ month ∧ month ≤ 12
  h3 : 1900 ≤ year ∧ year ≤ 1999

def date_to_number (d : Date) : Nat :=
  d.day * 10000 + d.month * 100 + (d.year - 1900)

/-- The birth dates of two friends satisfy the given conditions -/
def valid_birth_dates (d1 d2 : Date) : Prop :=
  d1.month = d2.month ∧
  d1.year = d2.year ∧
  d2.day = d1.day + 7 ∧
  date_to_number d2 = 6 * date_to_number d1

theorem unique_birth_date :
  ∃! d : Date, ∃ d2 : Date, valid_birth_dates d d2 ∧ d.day = 1 ∧ d.month = 4 ∧ d.year = 1900 :=
by sorry

end NUMINAMATH_CALUDE_unique_birth_date_l765_76514


namespace NUMINAMATH_CALUDE_max_S_n_is_three_halves_l765_76533

/-- Given a geometric sequence {a_n} with first term 3/2 and common ratio -1/2,
    S_n is the sum of the first n terms. -/
def S_n (n : ℕ) : ℚ :=
  (3/2) * (1 - (-1/2)^n) / (1 - (-1/2))

/-- The maximum value of S_n is 3/2. -/
theorem max_S_n_is_three_halves :
  ∃ (M : ℚ), M = 3/2 ∧ ∀ (n : ℕ), S_n n ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_S_n_is_three_halves_l765_76533


namespace NUMINAMATH_CALUDE_rectangular_field_area_l765_76553

theorem rectangular_field_area (length breadth : ℝ) : 
  breadth = 0.6 * length →
  2 * (length + breadth) = 800 →
  length * breadth = 37500 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l765_76553


namespace NUMINAMATH_CALUDE_cricket_team_size_l765_76590

theorem cricket_team_size :
  ∀ (n : ℕ) (captain_age wicket_keeper_age team_avg_age remaining_avg_age : ℝ),
    n > 0 →
    captain_age = 26 →
    wicket_keeper_age = captain_age + 3 →
    team_avg_age = 23 →
    remaining_avg_age = team_avg_age - 1 →
    team_avg_age * n = remaining_avg_age * (n - 2) + captain_age + wicket_keeper_age →
    n = 11 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l765_76590


namespace NUMINAMATH_CALUDE_gcd_lcm_336_1260_l765_76556

theorem gcd_lcm_336_1260 : 
  (Nat.gcd 336 1260 = 84) ∧ (Nat.lcm 336 1260 = 5040) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_336_1260_l765_76556


namespace NUMINAMATH_CALUDE_equilateral_triangle_theorem_l765_76576

open Real

/-- Triangle with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := sorry

/-- The equation given in the problem -/
def equation_holds (t : Triangle) : Prop :=
  (t.a * cos t.A + t.b * cos t.B + t.c * cos t.C) / 
  (t.a * sin t.A + t.b * sin t.B + t.c * sin t.C) = 
  (t.a + t.b + t.c) / (9 * circumradius t)

/-- The main theorem to prove -/
theorem equilateral_triangle_theorem (t : Triangle) :
  equation_holds t → t.a = t.b ∧ t.b = t.c := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_theorem_l765_76576


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l765_76565

theorem inequality_system_solution_set (x : ℝ) :
  (x + 1 > 0 ∧ x - 3 > 0) ↔ x > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l765_76565


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l765_76550

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l765_76550


namespace NUMINAMATH_CALUDE_probability_theorem_l765_76512

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def valid_assignment (al bill cal : ℕ) : Prop :=
  1 ≤ al ∧ al ≤ 12 ∧
  1 ≤ bill ∧ bill ≤ 12 ∧
  1 ≤ cal ∧ cal ≤ 12 ∧
  al ≠ bill ∧ bill ≠ cal ∧ al ≠ cal

def satisfies_conditions (al bill cal : ℕ) : Prop :=
  is_multiple al bill ∧ is_multiple bill cal ∧ is_even (al + bill + cal)

def total_assignments : ℕ := 12 * 11 * 10

theorem probability_theorem :
  (∃ valid_count : ℕ,
    (∀ al bill cal : ℕ, valid_assignment al bill cal → satisfies_conditions al bill cal →
      valid_count > 0) ∧
    (valid_count : ℚ) / total_assignments = 2 / 110) :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l765_76512


namespace NUMINAMATH_CALUDE_number_of_rooms_l765_76508

/-- Calculates the number of equal-sized rooms given the original dimensions,
    increase in dimensions, and total area. --/
def calculate_rooms (original_length original_width increase_dim total_area : ℕ) : ℕ :=
  let new_length := original_length + increase_dim
  let new_width := original_width + increase_dim
  let room_area := new_length * new_width
  let double_room_area := 2 * room_area
  let equal_rooms_area := total_area - double_room_area
  equal_rooms_area / room_area

/-- Theorem stating that the number of equal-sized rooms is 4 --/
theorem number_of_rooms : calculate_rooms 13 18 2 1800 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_rooms_l765_76508


namespace NUMINAMATH_CALUDE_total_cards_l765_76545

/-- The number of cards each person has -/
structure CardCounts where
  heike : ℕ
  anton : ℕ
  ann : ℕ
  bertrand : ℕ

/-- The conditions of the card counting problem -/
def card_problem (c : CardCounts) : Prop :=
  c.anton = 3 * c.heike ∧
  c.ann = 6 * c.heike ∧
  c.bertrand = 2 * c.heike ∧
  c.ann = 60

/-- The theorem stating that under the given conditions, 
    the total number of cards is 120 -/
theorem total_cards (c : CardCounts) : 
  card_problem c → c.heike + c.anton + c.ann + c.bertrand = 120 := by
  sorry


end NUMINAMATH_CALUDE_total_cards_l765_76545


namespace NUMINAMATH_CALUDE_expected_black_balls_l765_76558

/-- The expected number of black balls drawn when drawing 3 balls without replacement from a bag containing 5 red balls and 2 black balls. -/
theorem expected_black_balls (total : Nat) (red : Nat) (black : Nat) (drawn : Nat) 
  (h_total : total = 7)
  (h_red : red = 5)
  (h_black : black = 2)
  (h_drawn : drawn = 3)
  (h_sum : red + black = total) :
  (0 : ℚ) * (Nat.choose red drawn : ℚ) / (Nat.choose total drawn : ℚ) +
  (1 : ℚ) * (Nat.choose red (drawn - 1) * Nat.choose black 1 : ℚ) / (Nat.choose total drawn : ℚ) +
  (2 : ℚ) * (Nat.choose red (drawn - 2) * Nat.choose black 2 : ℚ) / (Nat.choose total drawn : ℚ) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expected_black_balls_l765_76558
