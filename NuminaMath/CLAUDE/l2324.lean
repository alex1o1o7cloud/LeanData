import Mathlib

namespace NUMINAMATH_CALUDE_new_average_after_grace_marks_l2324_232492

theorem new_average_after_grace_marks 
  (num_students : ℕ) 
  (original_average : ℚ) 
  (grace_marks : ℚ) :
  num_students = 35 →
  original_average = 37 →
  grace_marks = 3 →
  (num_students : ℚ) * original_average + num_students * grace_marks = num_students * 40 :=
by sorry

end NUMINAMATH_CALUDE_new_average_after_grace_marks_l2324_232492


namespace NUMINAMATH_CALUDE_hyperbola_standard_form_l2324_232421

-- Define the asymptotes of the hyperbola
def asymptote_slope : ℝ := 2

-- Define the ellipse that shares foci with the hyperbola
def ellipse_equation (x y : ℝ) : Prop := x^2 / 49 + y^2 / 24 = 1

-- Define the standard form of a hyperbola
def hyperbola_equation (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Theorem statement
theorem hyperbola_standard_form :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), y = asymptote_slope * x ∨ y = -asymptote_slope * x) →
  (∀ (x y : ℝ), ellipse_equation x y ↔ 
    ∃ (x' y' : ℝ), hyperbola_equation a b x' y' ∧ 
    (x - x')^2 + (y - y')^2 = 0) →
  a^2 = 5 ∧ b^2 = 20 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_form_l2324_232421


namespace NUMINAMATH_CALUDE_coin_problem_l2324_232483

theorem coin_problem (x : ℕ) :
  (x : ℚ) + x / 2 + x / 4 = 105 →
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_coin_problem_l2324_232483


namespace NUMINAMATH_CALUDE_student_weight_average_l2324_232452

theorem student_weight_average (girls_avg : ℝ) (boys_avg : ℝ) 
  (h1 : girls_avg = 45) 
  (h2 : boys_avg = 55) : 
  (5 * girls_avg + 5 * boys_avg) / 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_student_weight_average_l2324_232452


namespace NUMINAMATH_CALUDE_only_dog_owners_l2324_232451

/-- The number of people who own only dogs -/
def D : ℕ := sorry

/-- The number of people who own only cats -/
def C : ℕ := 10

/-- The number of people who own only snakes -/
def S : ℕ := sorry

/-- The number of people who own only cats and dogs -/
def CD : ℕ := 5

/-- The number of people who own only cats and snakes -/
def CS : ℕ := sorry

/-- The number of people who own only dogs and snakes -/
def DS : ℕ := sorry

/-- The number of people who own cats, dogs, and snakes -/
def CDS : ℕ := 3

/-- The total number of pet owners -/
def total_pet_owners : ℕ := 59

/-- The total number of snake owners -/
def total_snake_owners : ℕ := 29

theorem only_dog_owners : D = 15 := by
  have h1 : D + C + S + CD + CS + DS + CDS = total_pet_owners := by sorry
  have h2 : S + CS + DS + CDS = total_snake_owners := by sorry
  sorry

end NUMINAMATH_CALUDE_only_dog_owners_l2324_232451


namespace NUMINAMATH_CALUDE_arc_length_sector_l2324_232413

/-- The arc length of a sector with central angle 36° and radius 15 is 3π. -/
theorem arc_length_sector (angle : ℝ) (radius : ℝ) : 
  angle = 36 → radius = 15 → (angle * π * radius) / 180 = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_arc_length_sector_l2324_232413


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l2324_232438

theorem cloth_sale_calculation (selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  selling_price = 8925 ∧ 
  profit_per_meter = 10 ∧ 
  cost_price_per_meter = 95 →
  (selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 85 := by
sorry

end NUMINAMATH_CALUDE_cloth_sale_calculation_l2324_232438


namespace NUMINAMATH_CALUDE_triangle_theorem_l2324_232471

theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.cos B = 3 →
  b * Real.cos A = 1 →
  A - B = π / 6 →
  c = 4 ∧ B = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2324_232471


namespace NUMINAMATH_CALUDE_cube_decomposition_smallest_term_l2324_232453

theorem cube_decomposition_smallest_term (m : ℕ) (h1 : m ≥ 2) : 
  (m^2 - m + 1 = 73) → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_decomposition_smallest_term_l2324_232453


namespace NUMINAMATH_CALUDE_total_books_theorem_melanie_books_l2324_232465

/-- Calculates the total number of books after a purchase -/
def total_books_after_purchase (initial_books : ℕ) (books_bought : ℕ) : ℕ :=
  initial_books + books_bought

/-- Theorem: The total number of books after a purchase is the sum of initial books and books bought -/
theorem total_books_theorem (initial_books books_bought : ℕ) :
  total_books_after_purchase initial_books books_bought = initial_books + books_bought :=
by
  sorry

/-- Melanie's book collection problem -/
theorem melanie_books :
  total_books_after_purchase 41 46 = 87 :=
by
  sorry

end NUMINAMATH_CALUDE_total_books_theorem_melanie_books_l2324_232465


namespace NUMINAMATH_CALUDE_certain_value_problem_l2324_232431

theorem certain_value_problem (x y : ℝ) : x = 69 ∧ x - 18 = 3 * (y - x) → y = 86 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_problem_l2324_232431


namespace NUMINAMATH_CALUDE_max_value_theorem_l2324_232447

theorem max_value_theorem (x y z : ℝ) 
  (hx : 0 < x ∧ x < Real.sqrt 5) 
  (hy : 0 < y ∧ y < Real.sqrt 5) 
  (hz : 0 < z ∧ z < Real.sqrt 5) 
  (h_sum : x^4 + y^4 + z^4 ≥ 27) :
  (x / (x^2 - 5)) + (y / (y^2 - 5)) + (z / (z^2 - 5)) ≤ -3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2324_232447


namespace NUMINAMATH_CALUDE_celia_receives_171_spiders_l2324_232450

/-- Represents the number of stickers of each type Célia has -/
structure StickerCount where
  butterfly : ℕ
  shark : ℕ
  snake : ℕ
  parakeet : ℕ
  monkey : ℕ

/-- Represents the conversion rates between different types of stickers -/
structure ConversionRates where
  butterfly_to_shark : ℕ
  snake_to_parakeet : ℕ
  monkey_to_spider : ℕ
  parakeet_to_spider : ℕ
  shark_to_parakeet : ℕ

/-- Calculates the total number of spider stickers Célia can receive -/
def total_spider_stickers (count : StickerCount) (rates : ConversionRates) : ℕ :=
  sorry

/-- Theorem stating that Célia can receive 171 spider stickers -/
theorem celia_receives_171_spiders (count : StickerCount) (rates : ConversionRates) 
    (h1 : count.butterfly = 4)
    (h2 : count.shark = 5)
    (h3 : count.snake = 3)
    (h4 : count.parakeet = 6)
    (h5 : count.monkey = 6)
    (h6 : rates.butterfly_to_shark = 3)
    (h7 : rates.snake_to_parakeet = 3)
    (h8 : rates.monkey_to_spider = 4)
    (h9 : rates.parakeet_to_spider = 3)
    (h10 : rates.shark_to_parakeet = 2) :
    total_spider_stickers count rates = 171 :=
  sorry

end NUMINAMATH_CALUDE_celia_receives_171_spiders_l2324_232450


namespace NUMINAMATH_CALUDE_function_inequality_l2324_232455

theorem function_inequality (f : ℝ → ℝ) (h₁ : Differentiable ℝ f) 
  (h₂ : ∀ x, deriv f x < f x) : 
  f 1 < Real.exp 1 * f 0 ∧ f 2017 < Real.exp 2017 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2324_232455


namespace NUMINAMATH_CALUDE_unique_solution_l2324_232493

theorem unique_solution : ∃! x : ℝ, 70 + 5 * 12 / (180 / x) = 71 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2324_232493


namespace NUMINAMATH_CALUDE_expression_evaluation_l2324_232418

theorem expression_evaluation : 6^3 - 4 * 6^2 + 4 * 6 + 2 = 98 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2324_232418


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2324_232439

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2324_232439


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l2324_232429

theorem least_positive_angle_theorem (θ : Real) : 
  (θ > 0 ∧ Real.cos (5 * π / 180) = Real.sin (25 * π / 180) + Real.sin θ) →
  θ = 35 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l2324_232429


namespace NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l2324_232405

theorem sqrt_x_plus_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l2324_232405


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l2324_232477

/-- Given a triangle ABC with heights ha, hb, hc corresponding to sides a, b, c respectively,
    prove that if ha = 6, hb = 4, and hc = 3, then a : b : c = 2 : 3 : 4 -/
theorem triangle_side_ratio (a b c ha hb hc : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_heights : ha = 6 ∧ hb = 4 ∧ hc = 3) 
  (h_area : a * ha = b * hb ∧ b * hb = c * hc) : 
  ∃ (k : ℝ), k > 0 ∧ a = 2 * k ∧ b = 3 * k ∧ c = 4 * k := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l2324_232477


namespace NUMINAMATH_CALUDE_third_altitude_values_l2324_232472

/-- Triangle with two known altitudes and an integer third altitude -/
structure TriangleWithAltitudes where
  /-- First known altitude -/
  h₁ : ℝ
  /-- Second known altitude -/
  h₂ : ℝ
  /-- Third altitude (integer) -/
  h₃ : ℤ
  /-- Condition that first altitude is 4 -/
  h₁_eq : h₁ = 4
  /-- Condition that second altitude is 12 -/
  h₂_eq : h₂ = 12

/-- Theorem stating the possible values of the third altitude -/
theorem third_altitude_values (t : TriangleWithAltitudes) :
  t.h₃ = 4 ∨ t.h₃ = 5 :=
sorry

end NUMINAMATH_CALUDE_third_altitude_values_l2324_232472


namespace NUMINAMATH_CALUDE_existence_of_lower_bound_upper_bound_l2324_232414

/-- The number of coefficients in (x+1)^a(x+2)^(n-a) divisible by 3 -/
def f (n a : ℕ) : ℕ :=
  sorry

/-- The minimum of f(n,a) for all valid a -/
def F (n : ℕ) : ℕ :=
  sorry

/-- There exist infinitely many positive integers n such that F(n) ≥ (n-1)/3 -/
theorem existence_of_lower_bound : ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, F n ≥ (n - 1) / 3 :=
  sorry

/-- For any positive integer n, F(n) ≤ (n-1)/3 -/
theorem upper_bound (n : ℕ) (hn : n > 0) : F n ≤ (n - 1) / 3 :=
  sorry

end NUMINAMATH_CALUDE_existence_of_lower_bound_upper_bound_l2324_232414


namespace NUMINAMATH_CALUDE_valid_assignments_count_l2324_232466

/-- Represents the set of mascots -/
inductive Mascot
| AXiang
| AHe
| ARu
| AYi
| LeYangyang

/-- Represents the set of volunteers -/
inductive Volunteer
| A
| B
| C
| D
| E

/-- A function that assigns mascots to volunteers -/
def Assignment := Volunteer → Mascot

/-- Predicate that checks if an assignment satisfies the given conditions -/
def ValidAssignment (f : Assignment) : Prop :=
  (f Volunteer.A = Mascot.AXiang ∨ f Volunteer.B = Mascot.AXiang) ∧
  f Volunteer.C ≠ Mascot.LeYangyang

/-- The number of valid assignments -/
def NumValidAssignments : ℕ := sorry

/-- Theorem stating that the number of valid assignments is 36 -/
theorem valid_assignments_count : NumValidAssignments = 36 := by sorry

end NUMINAMATH_CALUDE_valid_assignments_count_l2324_232466


namespace NUMINAMATH_CALUDE_divisors_of_40_and_72_l2324_232400

theorem divisors_of_40_and_72 : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ 40 % n = 0 ∧ 72 % n = 0) ∧ 
  (∀ n : ℕ, n > 0 ∧ 40 % n = 0 ∧ 72 % n = 0 → n ∈ S) ∧
  Finset.card S = 4 := by
sorry

end NUMINAMATH_CALUDE_divisors_of_40_and_72_l2324_232400


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l2324_232496

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 64 * Real.pi) :
  A = Real.pi * r^2 → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l2324_232496


namespace NUMINAMATH_CALUDE_min_games_for_condition_l2324_232474

/-- Represents a football championship. -/
structure Championship where
  teams : Nat
  games_played : Nat

/-- Calculates the total number of possible games in a championship. -/
def total_possible_games (c : Championship) : Nat :=
  c.teams * (c.teams - 1) / 2

/-- Defines the property that among any three teams, at least two have played against each other. -/
def satisfies_condition (c : Championship) : Prop :=
  ∀ (a b d : Fin c.teams), a ≠ b ∧ b ≠ d ∧ a ≠ d →
    ∃ (x y : Fin c.teams), (x = a ∧ y = b) ∨ (x = b ∧ y = d) ∨ (x = a ∧ y = d)

/-- The main theorem to be proved. -/
theorem min_games_for_condition (c : Championship) 
  (h1 : c.teams = 20)
  (h2 : c.games_played ≥ 90)
  (h3 : ∀ (c' : Championship), c'.teams = 20 ∧ c'.games_played < 90 → ¬satisfies_condition c') :
  satisfies_condition c :=
sorry

end NUMINAMATH_CALUDE_min_games_for_condition_l2324_232474


namespace NUMINAMATH_CALUDE_plates_arrangement_theorem_l2324_232459

-- Define the number of plates of each color
def yellow_plates : ℕ := 4
def blue_plates : ℕ := 3
def red_plates : ℕ := 2
def purple_plates : ℕ := 1

-- Define the total number of plates
def total_plates : ℕ := yellow_plates + blue_plates + red_plates + purple_plates

-- Function to calculate circular arrangements
def circular_arrangements (n : ℕ) : ℕ :=
  (Nat.factorial n) / n

-- Function to calculate arrangements with restrictions
def arrangements_with_restrictions (total : ℕ) (y : ℕ) (b : ℕ) (r : ℕ) (p : ℕ) : ℕ :=
  circular_arrangements total - circular_arrangements (total - 1)

-- Theorem statement
theorem plates_arrangement_theorem :
  arrangements_with_restrictions total_plates yellow_plates blue_plates red_plates purple_plates = 980 := by
  sorry

end NUMINAMATH_CALUDE_plates_arrangement_theorem_l2324_232459


namespace NUMINAMATH_CALUDE_opposites_power_2004_l2324_232462

theorem opposites_power_2004 (x y : ℝ) 
  (h : |x + 1| + |y + 2*x| = 0) : 
  (x + y)^2004 = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposites_power_2004_l2324_232462


namespace NUMINAMATH_CALUDE_immersed_cone_specific_gravity_l2324_232482

/-- Represents an equilateral cone immersed in water -/
structure ImmersedCone where
  -- Radius of the base of the cone
  baseRadius : ℝ
  -- Height of the cone
  height : ℝ
  -- Height of the cone above water
  heightAboveWater : ℝ
  -- Specific gravity of the cone material
  specificGravity : ℝ
  -- The cone is equilateral
  equilateral : height = baseRadius * Real.sqrt 3
  -- The area of the water surface circle is one-third of the base area
  waterSurfaceArea : π * (heightAboveWater / 3)^2 = π * baseRadius^2 / 3
  -- The angle between water surface and cone side is 120°
  waterSurfaceAngle : Real.cos (2 * π / 3) = heightAboveWater / (2 * baseRadius)

/-- Theorem stating the specific gravity of the cone -/
theorem immersed_cone_specific_gravity (c : ImmersedCone) :
  c.specificGravity = 1 - Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_immersed_cone_specific_gravity_l2324_232482


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l2324_232478

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N (m : ℝ) : Set ℝ := {x | x^2 - m*x < 0}

-- State the theorem
theorem intersection_implies_m_equals_one :
  ∀ m : ℝ, (M ∩ N m) = {x | 0 < x ∧ x < 1} → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l2324_232478


namespace NUMINAMATH_CALUDE_rotten_bananas_percentage_l2324_232479

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percent : ℚ)
  (good_fruits_percent : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percent = 15 / 100)
  (h4 : good_fruits_percent = 89 / 100)
  : (1 : ℚ) - (good_fruits_percent * (total_oranges + total_bananas : ℚ) - (1 - rotten_oranges_percent) * total_oranges) / total_bananas = 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_rotten_bananas_percentage_l2324_232479


namespace NUMINAMATH_CALUDE_angle_q_measure_l2324_232440

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  -- Angle measures in degrees
  angle_p : ℝ
  angle_q : ℝ
  angle_r : ℝ
  -- Triangle conditions
  sum_of_angles : angle_p + angle_q + angle_r = 180
  isosceles : angle_q = angle_r
  angle_r_five_times_p : angle_r = 5 * angle_p

/-- The measure of angle Q in the specified isosceles triangle is 900/11 degrees -/
theorem angle_q_measure (t : IsoscelesTriangle) : t.angle_q = 900 / 11 := by
  sorry

#check angle_q_measure

end NUMINAMATH_CALUDE_angle_q_measure_l2324_232440


namespace NUMINAMATH_CALUDE_smallest_divisible_number_l2324_232499

theorem smallest_divisible_number : ∃ N : ℕ,
  (∀ k : ℕ, 2 ≤ k → k ≤ 10 → (N + k) % k = 0) ∧
  (∀ M : ℕ, M < N → ∃ j : ℕ, 2 ≤ j ∧ j ≤ 10 ∧ (M + j) % j ≠ 0) ∧
  N = 2520 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_number_l2324_232499


namespace NUMINAMATH_CALUDE_part_one_part_two_l2324_232491

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Part I
theorem part_one : 
  let a : ℝ := -4
  {x : ℝ | f a x ≥ 6} = {x : ℝ | x ≤ 0 ∨ x ≥ 6} := by sorry

-- Part II
theorem part_two :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a x ≤ |x - 3|) → -1 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2324_232491


namespace NUMINAMATH_CALUDE_square_sum_equality_l2324_232403

theorem square_sum_equality (n : ℤ) : n + n + n + n = 4 * n := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l2324_232403


namespace NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l2324_232458

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- The factorial of n -/
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem thirty_factorial_trailing_zeros :
  trailingZeros 30 = 7 :=
by sorry

end NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l2324_232458


namespace NUMINAMATH_CALUDE_problem_statement_l2324_232417

theorem problem_statement (x y a b c : ℝ) : 
  (x = -y) → 
  (a * b = 1) → 
  (|c| = 2) → 
  ((((x + y) / 2)^2023) - ((-a * b)^2023) + c^3 = 9 ∨ 
   (((x + y) / 2)^2023) - ((-a * b)^2023) + c^3 = -7) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2324_232417


namespace NUMINAMATH_CALUDE_max_expression_value_l2324_232407

-- Define a digit as a natural number between 0 and 9
def Digit := {n : ℕ // n ≤ 9}

-- Define the expression as a function of three digits
def Expression (x y z : Digit) : ℕ := 
  100 * x.val + 10 * y.val + z.val + 
  10 * x.val + z.val + 
  x.val

-- Theorem statement
theorem max_expression_value :
  ∃ (x y z : Digit), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    Expression x y z = 992 ∧
    ∀ (a b c : Digit), a ≠ b ∧ b ≠ c ∧ a ≠ c →
      Expression a b c ≤ 992 :=
sorry

end NUMINAMATH_CALUDE_max_expression_value_l2324_232407


namespace NUMINAMATH_CALUDE_os_overhead_cost_value_l2324_232486

/-- The cost per millisecond of computer time -/
def cost_per_ms : ℚ := 23 / 1000

/-- The cost for mounting a data tape -/
def tape_cost : ℚ := 5.35

/-- The total cost for 1 run with 1.5 seconds of computer time -/
def total_cost : ℚ := 40.92

/-- The duration of the computer run in milliseconds -/
def run_duration_ms : ℕ := 1500

/-- The operating-system overhead cost -/
def os_overhead_cost : ℚ := total_cost - (cost_per_ms * run_duration_ms) - tape_cost

theorem os_overhead_cost_value : os_overhead_cost = 1.07 := by
  sorry

end NUMINAMATH_CALUDE_os_overhead_cost_value_l2324_232486


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2324_232419

theorem polynomial_evaluation (y : ℝ) (h : y = 2) : y^4 + y^3 + y^2 + y + 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2324_232419


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l2324_232430

theorem cryptarithmetic_puzzle (F I V E G H T : ℕ) : 
  (F = 8) →
  (V % 2 = 1) →
  (100 * F + 10 * I + V + 100 * F + 10 * I + V = 10000 * E + 1000 * I + 100 * G + 10 * H + T) →
  (F ≠ I ∧ F ≠ V ∧ F ≠ E ∧ F ≠ G ∧ F ≠ H ∧ F ≠ T ∧
   I ≠ V ∧ I ≠ E ∧ I ≠ G ∧ I ≠ H ∧ I ≠ T ∧
   V ≠ E ∧ V ≠ G ∧ V ≠ H ∧ V ≠ T ∧
   E ≠ G ∧ E ≠ H ∧ E ≠ T ∧
   G ≠ H ∧ G ≠ T ∧
   H ≠ T) →
  (F < 10 ∧ I < 10 ∧ V < 10 ∧ E < 10 ∧ G < 10 ∧ H < 10 ∧ T < 10) →
  I = 2 := by
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l2324_232430


namespace NUMINAMATH_CALUDE_tom_annual_cost_l2324_232408

/-- Calculates the total annual cost for Tom's sleep medication and doctor visits -/
def annual_cost (daily_pills : ℕ) (pill_cost : ℚ) (insurance_coverage : ℚ) 
                (yearly_doctor_visits : ℕ) (doctor_visit_cost : ℚ) : ℚ :=
  let daily_medication_cost := daily_pills * pill_cost
  let daily_out_of_pocket := daily_medication_cost * (1 - insurance_coverage)
  let annual_medication_cost := daily_out_of_pocket * 365
  let annual_doctor_cost := yearly_doctor_visits * doctor_visit_cost
  annual_medication_cost + annual_doctor_cost

/-- Theorem stating that Tom's annual cost for sleep medication and doctor visits is $1530 -/
theorem tom_annual_cost : 
  annual_cost 2 5 (4/5) 2 400 = 1530 := by
  sorry

end NUMINAMATH_CALUDE_tom_annual_cost_l2324_232408


namespace NUMINAMATH_CALUDE_curve_E_perpendicular_points_sum_inverse_squares_l2324_232448

-- Define the curve E
def curve_E (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the property of perpendicular vectors
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem curve_E_perpendicular_points_sum_inverse_squares (x₁ y₁ x₂ y₂ : ℝ) :
  curve_E x₁ y₁ → curve_E x₂ y₂ → perpendicular x₁ y₁ x₂ y₂ →
  1 / (x₁^2 + y₁^2) + 1 / (x₂^2 + y₂^2) = 7 / 12 :=
by sorry

end NUMINAMATH_CALUDE_curve_E_perpendicular_points_sum_inverse_squares_l2324_232448


namespace NUMINAMATH_CALUDE_quadratic_intersection_theorem_l2324_232435

/-- Represents the "graph number" of a quadratic function y = ax^2 + bx + c -/
structure GraphNumber where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- A quadratic function intersects the x-axis at only one point if and only if its discriminant is zero -/
def intersects_x_axis_once (g : GraphNumber) : Prop :=
  (g.b ^ 2) - (4 * g.a * g.c) = 0

theorem quadratic_intersection_theorem (m : ℝ) (hm : m ≠ 0) :
  let g := GraphNumber.mk m (2 * m + 4) (2 * m + 4) hm
  intersects_x_axis_once g → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_theorem_l2324_232435


namespace NUMINAMATH_CALUDE_power_product_equals_four_l2324_232456

theorem power_product_equals_four (a b : ℕ+) (h : (3 ^ a.val) ^ b.val = 3 ^ 3) :
  3 ^ a.val * 3 ^ b.val = 3 ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_four_l2324_232456


namespace NUMINAMATH_CALUDE_arrangements_count_is_24_l2324_232432

/-- The number of ways to arrange 5 people in a row with specific adjacency constraints -/
def arrangements_count : ℕ :=
  let total_people : ℕ := 5
  let adjacent_pair : ℕ := 2  -- A and B
  let non_adjacent : ℕ := 1   -- C
  (adjacent_pair.choose 1) * (adjacent_pair.factorial) * ((total_people - adjacent_pair).factorial)

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_count_is_24 : arrangements_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_is_24_l2324_232432


namespace NUMINAMATH_CALUDE_work_completion_time_l2324_232406

/-- 
Given a group of people who can complete a task in 12 days, 
prove that twice that number of people can complete half the task in 3 days.
-/
theorem work_completion_time 
  (people : ℕ) 
  (work : ℝ) 
  (h : people > 0) 
  (complete_time : ℝ → ℝ → ℝ → ℝ) 
  (h_complete : complete_time people work 12 = 1) :
  complete_time (2 * people) (work / 2) 3 = 1 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l2324_232406


namespace NUMINAMATH_CALUDE_no_integer_solution_l2324_232412

theorem no_integer_solution :
  ∀ (x y z : ℤ), x ≠ 0 → 2 * x^4 + 2 * x^2 * y^2 + y^4 ≠ z^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2324_232412


namespace NUMINAMATH_CALUDE_sum_three_numbers_l2324_232445

theorem sum_three_numbers (a b c N : ℝ) : 
  a + b + c = 72 →
  a - 7 = N →
  b + 7 = N →
  2 * c = N →
  N = 28.8 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l2324_232445


namespace NUMINAMATH_CALUDE_distance_from_origin_to_12_5_l2324_232436

/-- The distance from the origin to the point (12, 5) in a rectangular coordinate system is 13 units. -/
theorem distance_from_origin_to_12_5 : 
  Real.sqrt (12^2 + 5^2) = 13 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_12_5_l2324_232436


namespace NUMINAMATH_CALUDE_range_of_a_l2324_232485

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0

theorem range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  (a ∈ Set.Icc (-1) 1) ∨ (a > 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2324_232485


namespace NUMINAMATH_CALUDE_one_match_among_withdrawn_l2324_232480

/-- Represents a table tennis tournament with special conditions -/
structure TableTennisTournament where
  n : ℕ  -- Total number of players
  total_matches : ℕ  -- Total number of matches played
  withdrawn_players : ℕ  -- Number of players who withdrew
  matches_per_withdrawn : ℕ  -- Number of matches each withdrawn player played
  hwithdrawncond : withdrawn_players = 3
  hmatchescond : matches_per_withdrawn = 2
  htotalcond : total_matches = 50

/-- The number of matches played among the withdrawn players -/
def matches_among_withdrawn (t : TableTennisTournament) : ℕ := 
  (t.withdrawn_players * t.matches_per_withdrawn - 
   t.total_matches + (t.n - t.withdrawn_players).choose 2) / 2

/-- Theorem stating that exactly one match was played among the withdrawn players -/
theorem one_match_among_withdrawn (t : TableTennisTournament) : 
  matches_among_withdrawn t = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_match_among_withdrawn_l2324_232480


namespace NUMINAMATH_CALUDE_santa_claus_candy_distribution_l2324_232484

theorem santa_claus_candy_distribution :
  ∃ (n b g c m : ℕ),
    n = b + g ∧
    n > 0 ∧
    b * c + g * (c + 1) = 47 ∧
    b * (m + 1) + g * m = 74 ∧
    n = 11 :=
by sorry

end NUMINAMATH_CALUDE_santa_claus_candy_distribution_l2324_232484


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l2324_232470

/-- A rhombus with an inscribed square -/
structure RhombusWithSquare where
  /-- Length of the first diagonal of the rhombus -/
  d1 : ℝ
  /-- Length of the second diagonal of the rhombus -/
  d2 : ℝ
  /-- The first diagonal is positive -/
  d1_pos : 0 < d1
  /-- The second diagonal is positive -/
  d2_pos : 0 < d2
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- The square is inscribed with sides parallel to rhombus diagonals -/
  inscribed : square_side > 0

/-- Theorem stating the side length of the inscribed square in a rhombus with given diagonals -/
theorem inscribed_square_side_length (r : RhombusWithSquare) (h1 : r.d1 = 8) (h2 : r.d2 = 12) : 
  r.square_side = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l2324_232470


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2324_232424

/-- If 100x^2 - kxy + 49y^2 is a perfect square, then k = ±140 -/
theorem perfect_square_condition (x y k : ℝ) :
  (∃ (z : ℝ), 100 * x^2 - k * x * y + 49 * y^2 = z^2) →
  (k = 140 ∨ k = -140) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2324_232424


namespace NUMINAMATH_CALUDE_population_difference_specific_population_difference_l2324_232410

/-- The population difference between two cities with different volumes, given a constant population density. -/
theorem population_difference (density : ℕ) (volume1 volume2 : ℕ) :
  density * (volume1 - volume2) = density * volume1 - density * volume2 := by
  sorry

/-- The population difference between two specific cities. -/
theorem specific_population_difference :
  let density : ℕ := 80
  let volume1 : ℕ := 9000
  let volume2 : ℕ := 6400
  density * (volume1 - volume2) = 208000 := by
  sorry

end NUMINAMATH_CALUDE_population_difference_specific_population_difference_l2324_232410


namespace NUMINAMATH_CALUDE_annie_walk_distance_l2324_232437

/-- The number of blocks Annie walked from her house to the bus stop -/
def annie_walk : ℕ := sorry

/-- The number of blocks Annie rode on the bus each way -/
def bus_ride : ℕ := 7

/-- The total number of blocks Annie traveled -/
def total_distance : ℕ := 24

theorem annie_walk_distance : annie_walk = 5 := by
  have h1 : 2 * annie_walk + 2 * bus_ride = total_distance := sorry
  sorry

end NUMINAMATH_CALUDE_annie_walk_distance_l2324_232437


namespace NUMINAMATH_CALUDE_colored_area_half_l2324_232443

/-- Triangle ABC with side AB divided into n parts and AC into n+1 parts -/
structure DividedTriangle where
  ABC : Triangle
  n : ℕ

/-- The ratio of the sum of areas of colored triangles to the area of ABC -/
def coloredAreaRatio (dt : DividedTriangle) : ℚ :=
  sorry

/-- Theorem: The colored area ratio is always 1/2 -/
theorem colored_area_half (dt : DividedTriangle) : coloredAreaRatio dt = 1/2 :=
  sorry

end NUMINAMATH_CALUDE_colored_area_half_l2324_232443


namespace NUMINAMATH_CALUDE_projectile_trajectory_l2324_232401

/-- Represents the trajectory of a projectile --/
def trajectory (c g : ℝ) (x y : ℝ) : Prop :=
  x^2 = (2 * c^2 / g) * y

/-- Theorem stating that a projectile follows a parabolic trajectory --/
theorem projectile_trajectory (c g : ℝ) (hc : c > 0) (hg : g > 0) :
  ∀ x y : ℝ, trajectory c g x y ↔ x^2 = (2 * c^2 / g) * y :=
sorry

end NUMINAMATH_CALUDE_projectile_trajectory_l2324_232401


namespace NUMINAMATH_CALUDE_max_gcd_of_sum_1155_l2324_232498

theorem max_gcd_of_sum_1155 :
  ∃ (a b : ℕ+), a + b = 1155 ∧
  ∀ (c d : ℕ+), c + d = 1155 → Nat.gcd c d ≤ Nat.gcd a b ∧
  Nat.gcd a b = 105 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_sum_1155_l2324_232498


namespace NUMINAMATH_CALUDE_sin_cos_shift_l2324_232444

theorem sin_cos_shift (x : ℝ) : 
  Real.sin (2 * x) = Real.cos (2 * (x - π / 3) + π / 6) := by sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l2324_232444


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l2324_232476

theorem sine_cosine_inequality (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l2324_232476


namespace NUMINAMATH_CALUDE_fencing_length_l2324_232463

/-- Given a rectangular field with area 400 sq. ft and one side of 20 feet,
    prove that the fencing required for three sides is 60 feet. -/
theorem fencing_length (area : ℝ) (side : ℝ) (h1 : area = 400) (h2 : side = 20) :
  2 * (area / side) + side = 60 := by
  sorry

end NUMINAMATH_CALUDE_fencing_length_l2324_232463


namespace NUMINAMATH_CALUDE_exists_geometric_progression_shift_l2324_232427

/-- Given a sequence {a_n} defined by a_n = q * a_{n-1} + d where q ≠ 1,
    there exists a constant c such that b_n = a_n + c forms a geometric progression. -/
theorem exists_geometric_progression_shift 
  (q d : ℝ) (hq : q ≠ 1) (a : ℕ → ℝ) 
  (ha : ∀ n : ℕ, a (n + 1) = q * a n + d) :
  ∃ c : ℝ, ∃ r : ℝ, ∀ n : ℕ, a n + c = r^n * (a 0 + c) :=
sorry

end NUMINAMATH_CALUDE_exists_geometric_progression_shift_l2324_232427


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2324_232411

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) : Set ℝ := {x | x ≤ -4 ∨ x ≥ 3}

-- State the theorem
theorem quadratic_inequality_theorem (a b c : ℝ) 
  (h : ∀ x, f a b c x ≤ 0 ↔ x ∈ solution_set a b c) : 
  (a + b + c > 0) ∧ (∀ x, b * x + c > 0 ↔ x < 12) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2324_232411


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2324_232425

theorem complex_equation_solution (z : ℂ) : z * (1 + 2*I) = 3 + I → z = 1 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2324_232425


namespace NUMINAMATH_CALUDE_geometric_roots_poly_n_value_l2324_232434

/-- A polynomial of degree 4 with four distinct real roots in geometric progression -/
structure GeometricRootsPoly where
  m : ℝ
  n : ℝ
  p : ℝ
  roots : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → roots i ≠ roots j
  geometric : ∃ (a r : ℝ), ∀ i, roots i = a * r ^ i.val
  is_root : ∀ i, roots i ^ 4 + m * roots i ^ 3 + n * roots i ^ 2 + p * roots i + 256 = 0

/-- The theorem stating that n = -32 for such polynomials -/
theorem geometric_roots_poly_n_value (poly : GeometricRootsPoly) : poly.n = -32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_roots_poly_n_value_l2324_232434


namespace NUMINAMATH_CALUDE_decimal_digits_of_fraction_l2324_232422

/-- The number of digits to the right of the decimal point when 5^7 / (10^5 * 125) is expressed as a decimal is 5. -/
theorem decimal_digits_of_fraction : ∃ (n : ℕ) (d : ℕ+) (k : ℕ),
  5^7 / (10^5 * 125) = n / d ∧
  10^k ≤ d ∧ d < 10^(k+1) ∧
  k = 5 := by sorry

end NUMINAMATH_CALUDE_decimal_digits_of_fraction_l2324_232422


namespace NUMINAMATH_CALUDE_factor_tree_X_value_l2324_232423

/-- Represents a node in the factor tree -/
structure TreeNode where
  value : ℕ

/-- Represents the factor tree structure -/
structure FactorTree where
  X : TreeNode
  F : TreeNode
  G : TreeNode
  H : TreeNode

/-- The main theorem to prove -/
theorem factor_tree_X_value (tree : FactorTree) : tree.X.value = 6776 :=
  sorry

/-- Axioms representing the given conditions -/
axiom F_value (tree : FactorTree) : tree.F.value = 7 * 4

axiom G_value (tree : FactorTree) : tree.G.value = 11 * tree.H.value

axiom H_value (tree : FactorTree) : tree.H.value = 11 * 2

axiom X_value (tree : FactorTree) : tree.X.value = tree.F.value * tree.G.value

end NUMINAMATH_CALUDE_factor_tree_X_value_l2324_232423


namespace NUMINAMATH_CALUDE_hotel_price_per_night_l2324_232428

def car_value : ℕ := 30000
def house_value : ℕ := 4 * car_value
def total_value : ℕ := 158000

theorem hotel_price_per_night :
  ∃ (price_per_night : ℕ), 
    car_value + house_value + 2 * price_per_night = total_value ∧
    price_per_night = 4000 :=
by sorry

end NUMINAMATH_CALUDE_hotel_price_per_night_l2324_232428


namespace NUMINAMATH_CALUDE_rational_sums_and_products_l2324_232442

-- Define the property of being rational
def IsRational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Main theorem
theorem rational_sums_and_products (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : IsRational (x * y))
  (hyz : IsRational (y * z))
  (hzx : IsRational (z * x)) :
  (IsRational (x^2 + y^2 + z^2)) ∧
  (IsRational (x^3 + y^3 + z^3) → IsRational x ∧ IsRational y ∧ IsRational z) := by
  sorry


end NUMINAMATH_CALUDE_rational_sums_and_products_l2324_232442


namespace NUMINAMATH_CALUDE_safari_lions_count_l2324_232487

theorem safari_lions_count (L S G : ℕ) : 
  S = L / 2 →
  G = S - 10 →
  2 * L + 3 * S + (G + 20) = 410 →
  L = 72 := by
sorry

end NUMINAMATH_CALUDE_safari_lions_count_l2324_232487


namespace NUMINAMATH_CALUDE_sum_divides_10n_count_l2324_232446

theorem sum_divides_10n_count : 
  (∃ (S : Finset ℕ), S.card = 5 ∧ 
    (∀ n : ℕ, n > 0 → (n ∈ S ↔ (10 * n) % ((n * (n + 1)) / 2) = 0))) :=
sorry

end NUMINAMATH_CALUDE_sum_divides_10n_count_l2324_232446


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_81_l2324_232464

theorem factor_x_squared_minus_81 (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_81_l2324_232464


namespace NUMINAMATH_CALUDE_worker_B_completion_time_l2324_232494

-- Define the time it takes for Worker A to complete the job
def worker_A_time : ℝ := 5

-- Define the time it takes for both workers together to complete the job
def combined_time : ℝ := 3.333333333333333

-- Define the time it takes for Worker B to complete the job
def worker_B_time : ℝ := 10

-- Theorem statement
theorem worker_B_completion_time :
  (1 / worker_A_time + 1 / worker_B_time = 1 / combined_time) →
  worker_B_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_worker_B_completion_time_l2324_232494


namespace NUMINAMATH_CALUDE_sara_letters_count_l2324_232426

/-- The number of letters Sara sent in January. -/
def january_letters : ℕ := 6

/-- The number of letters Sara sent in February. -/
def february_letters : ℕ := 9

/-- The number of letters Sara sent in March. -/
def march_letters : ℕ := 3 * january_letters

/-- The total number of letters Sara sent over three months. -/
def total_letters : ℕ := january_letters + february_letters + march_letters

theorem sara_letters_count : total_letters = 33 := by
  sorry

end NUMINAMATH_CALUDE_sara_letters_count_l2324_232426


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2324_232409

theorem product_of_three_numbers (x y z : ℝ) : 
  x + y + z = 30 →
  x = 3 * ((y + z) - 2) →
  y = 4 * z - 1 →
  x * y * z = 294 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2324_232409


namespace NUMINAMATH_CALUDE_parallel_segment_sum_l2324_232497

/-- Given two points A(a,-2) and B(1,b) in a plane rectangular coordinate system,
    if AB is parallel to the x-axis and AB = 3, then a + b = 2 or a + b = -4 -/
theorem parallel_segment_sum (a b : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (a, -2) ∧ B = (1, b) ∧ 
   (A.2 = B.2) ∧  -- AB is parallel to x-axis
   ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 3^2))  -- AB = 3
  → (a + b = 2 ∨ a + b = -4) := by
sorry

end NUMINAMATH_CALUDE_parallel_segment_sum_l2324_232497


namespace NUMINAMATH_CALUDE_remainder_theorem_l2324_232449

theorem remainder_theorem (x y u v : ℕ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x = u * y + v) (h4 : v < y) :
  (x + 4 * u * y) % y = v := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2324_232449


namespace NUMINAMATH_CALUDE_A_intersect_B_l2324_232481

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}

def B : Set ℝ := {x | ∃ k : ℤ, x = 2*k}

theorem A_intersect_B : A ∩ B = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2324_232481


namespace NUMINAMATH_CALUDE_number_division_multiplication_l2324_232473

theorem number_division_multiplication (x : ℚ) : x = 5.5 → (x / 6) * 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_division_multiplication_l2324_232473


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l2324_232402

-- Define the concept of a quadratic radical
def QuadraticRadical (x : ℝ) : Prop := x ≥ 0

-- Define the concept of simplest quadratic radical
def SimplestQuadraticRadical (x : ℝ) : Prop :=
  QuadraticRadical x ∧ 
  ∀ y : ℝ, y > 1 → ¬(∃ z : ℝ, x = y * z^2)

-- Define the set of given expressions
def GivenExpressions : Set ℝ := {8, 1/3, 6, 0.1}

-- Theorem statement
theorem simplest_quadratic_radical :
  ∀ x ∈ GivenExpressions, 
    SimplestQuadraticRadical (Real.sqrt x) → x = 6 :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l2324_232402


namespace NUMINAMATH_CALUDE_count_non_divisible_eq_31_l2324_232415

/-- The product of proper positive integer divisors of n -/
def g_hat (n : ℕ) : ℕ := sorry

/-- Counts the number of integers n between 2 and 100 (inclusive) for which n does not divide g_hat(n) -/
def count_non_divisible : ℕ := sorry

/-- Theorem stating that the count of non-divisible numbers is 31 -/
theorem count_non_divisible_eq_31 : count_non_divisible = 31 := by sorry

end NUMINAMATH_CALUDE_count_non_divisible_eq_31_l2324_232415


namespace NUMINAMATH_CALUDE_cube_root_of_unity_in_finite_field_l2324_232457

theorem cube_root_of_unity_in_finite_field (p : ℕ) (hp : p.Prime) (hp3 : p > 3) :
  let F := ZMod p
  (∃ x : F, x^2 = -3) →
    (∃! (a b c : F), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a^3 = 1 ∧ b^3 = 1 ∧ c^3 = 1) ∧
  (¬∃ x : F, x^2 = -3) →
    (∃! a : F, a^3 = 1) :=
sorry

end NUMINAMATH_CALUDE_cube_root_of_unity_in_finite_field_l2324_232457


namespace NUMINAMATH_CALUDE_crosswalk_distance_l2324_232460

/-- Given a parallelogram with the following properties:
  * One side has length 22 feet
  * An adjacent side has length 65 feet
  * The altitude perpendicular to the 22-foot side is 60 feet
  Then the altitude perpendicular to the 65-foot side is 264/13 feet. -/
theorem crosswalk_distance (a b h₁ h₂ : ℝ) 
  (ha : a = 22) 
  (hb : b = 65) 
  (hh₁ : h₁ = 60) : 
  a * h₁ = b * h₂ → h₂ = 264 / 13 := by
  sorry

#check crosswalk_distance

end NUMINAMATH_CALUDE_crosswalk_distance_l2324_232460


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l2324_232469

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if c² = (a-b)² + 6 and the area of the triangle is 3√3/2,
    then the measure of angle C is π/3 -/
theorem angle_measure_in_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  c^2 = (a - b)^2 + 6 →
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →
  C = π / 3 := by sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l2324_232469


namespace NUMINAMATH_CALUDE_jasons_points_theorem_l2324_232454

/-- Calculates the total points Jason has from seashells and starfish -/
def jasons_total_points (initial_seashells : ℕ) (initial_starfish : ℕ) 
  (seashell_points : ℕ) (starfish_points : ℕ)
  (seashells_given_tim : ℕ) (seashells_given_lily : ℕ)
  (seashells_found : ℕ) (seashells_lost : ℕ) : ℕ :=
  let initial_points := initial_seashells * seashell_points + initial_starfish * starfish_points
  let points_given_away := (seashells_given_tim + seashells_given_lily) * seashell_points
  let net_points_found_lost := (seashells_found - seashells_lost) * seashell_points
  initial_points - points_given_away + net_points_found_lost

theorem jasons_points_theorem :
  jasons_total_points 49 48 2 3 13 7 15 5 = 222 := by
  sorry

end NUMINAMATH_CALUDE_jasons_points_theorem_l2324_232454


namespace NUMINAMATH_CALUDE_function_composition_l2324_232489

/-- Given a function f such that f(3x) = 5 / (3 + x) for all x > 0,
    prove that 3f(x) = 45 / (9 + x) --/
theorem function_composition (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 5 / (3 + x)) :
  ∀ x > 0, 3 * f x = 45 / (9 + x) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l2324_232489


namespace NUMINAMATH_CALUDE_definite_integral_3x_squared_l2324_232416

theorem definite_integral_3x_squared : ∫ x in (1:ℝ)..2, 3 * x^2 = 7 := by sorry

end NUMINAMATH_CALUDE_definite_integral_3x_squared_l2324_232416


namespace NUMINAMATH_CALUDE_expression_not_prime_l2324_232495

theorem expression_not_prime :
  ∀ x : ℕ, 0 < x → x < 100 →
  ∃ k : ℕ, 3^x + 5^x + 7^x + 11^x + 13^x + 17^x + 19^x = 3 * k :=
by sorry

end NUMINAMATH_CALUDE_expression_not_prime_l2324_232495


namespace NUMINAMATH_CALUDE_expression_evaluation_l2324_232433

theorem expression_evaluation :
  (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2324_232433


namespace NUMINAMATH_CALUDE_tshirt_packages_l2324_232420

theorem tshirt_packages (total_tshirts : ℕ) (tshirts_per_package : ℕ) (h1 : total_tshirts = 426) (h2 : tshirts_per_package = 6) :
  total_tshirts / tshirts_per_package = 71 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_packages_l2324_232420


namespace NUMINAMATH_CALUDE_fish_problem_l2324_232461

theorem fish_problem (trout_weight salmon_weight campers fish_per_camper bass_weight : ℕ) 
  (h1 : trout_weight = 8)
  (h2 : salmon_weight = 24)
  (h3 : campers = 22)
  (h4 : fish_per_camper = 2)
  (h5 : bass_weight = 2) :
  (campers * fish_per_camper - (trout_weight + salmon_weight)) / bass_weight = 6 := by
  sorry

end NUMINAMATH_CALUDE_fish_problem_l2324_232461


namespace NUMINAMATH_CALUDE_min_pie_pieces_correct_l2324_232468

/-- The minimum number of pieces a pie can be cut into to be equally divided among either 10 or 11 guests -/
def min_pie_pieces : ℕ := 20

/-- The number of expected guests -/
def possible_guests : Set ℕ := {10, 11}

/-- A function that checks if a given number of pieces can be equally divided among a given number of guests -/
def is_divisible (pieces : ℕ) (guests : ℕ) : Prop :=
  ∃ (k : ℕ), pieces = k * guests

theorem min_pie_pieces_correct :
  (∀ g ∈ possible_guests, is_divisible min_pie_pieces g) ∧
  (∀ p < min_pie_pieces, ∃ g ∈ possible_guests, ¬is_divisible p g) :=
sorry

end NUMINAMATH_CALUDE_min_pie_pieces_correct_l2324_232468


namespace NUMINAMATH_CALUDE_roots_of_equation_l2324_232404

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => 3*x*(x-1) - 2*(x-1)
  (f 1 = 0 ∧ f (2/3) = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 2/3) := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2324_232404


namespace NUMINAMATH_CALUDE_tournament_games_played_23_teams_l2324_232441

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  no_ties : Bool

/-- Calculates the number of games played in a single-elimination tournament. -/
def games_played (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams and no ties,
    22 games must be played before a winner can be declared. -/
theorem tournament_games_played_23_teams :
  ∀ (t : Tournament), t.num_teams = 23 → t.no_ties = true →
  games_played t = 22 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_played_23_teams_l2324_232441


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l2324_232467

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.log x + 1)

def interval : Set ℝ := {x | 1 / Real.exp 2 ≤ x ∧ x ≤ 1}

theorem minimum_value_theorem (m : ℝ) (hm : ∀ x ∈ interval, f x ≥ m) :
  Real.log (abs m) = 1 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l2324_232467


namespace NUMINAMATH_CALUDE_trajectory_and_tangent_line_l2324_232490

-- Define points A, B, and P
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)
def P : ℝ → ℝ → ℝ × ℝ := λ x y => (x, y)

-- Define the condition |PA| = 2|PB|
def condition (x y : ℝ) : Prop :=
  (x + 3)^2 + y^2 = 4 * ((x - 3)^2 + y^2)

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = 16

-- Define the line l
def line_l (c : ℝ) (x y : ℝ) : Prop :=
  x + y + c = 0

-- Define the tangent condition
def is_tangent (c : ℝ) : Prop :=
  ∃ x y : ℝ, trajectory_C x y ∧ line_l c x y ∧
  ∀ x' y' : ℝ, trajectory_C x' y' → line_l c x' y' → (x', y') = (x, y)

-- State the theorem
theorem trajectory_and_tangent_line :
  (∀ x y : ℝ, condition x y ↔ trajectory_C x y) ∧
  (∃ c : ℝ, is_tangent c ∧ c = -5 + 4 * Real.sqrt 2 ∨ c = -5 - 4 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_tangent_line_l2324_232490


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l2324_232475

theorem greatest_three_digit_number : ∃ n : ℕ, 
  (n ≤ 999) ∧ 
  (n ≥ 100) ∧ 
  (∃ k : ℕ, n = 8 * k - 1) ∧ 
  (∃ m : ℕ, n = 7 * m + 4) ∧ 
  (∀ x : ℕ, x ≤ 999 ∧ x ≥ 100 ∧ (∃ a : ℕ, x = 8 * a - 1) ∧ (∃ b : ℕ, x = 7 * b + 4) → x ≤ n) ∧
  n = 967 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l2324_232475


namespace NUMINAMATH_CALUDE_correct_rounding_l2324_232488

def round_to_nearest_hundred (x : ℤ) : ℤ :=
  (x + 50) / 100 * 100

theorem correct_rounding : round_to_nearest_hundred ((58 + 44) * 3) = 300 := by
  sorry

end NUMINAMATH_CALUDE_correct_rounding_l2324_232488
