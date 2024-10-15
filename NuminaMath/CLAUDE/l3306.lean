import Mathlib

namespace NUMINAMATH_CALUDE_least_comic_books_l3306_330627

theorem least_comic_books (n : ℕ) : n > 0 ∧ n % 7 = 3 ∧ n % 4 = 1 → n ≥ 17 :=
by sorry

end NUMINAMATH_CALUDE_least_comic_books_l3306_330627


namespace NUMINAMATH_CALUDE_modified_array_sum_for_five_l3306_330615

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The modified 1/p-array sum -/
def modifiedArraySum (p : ℕ) : ℚ :=
  (3 * p^2) / ((9 * p^2 - 12 * p + 4) * (p - 1))

theorem modified_array_sum_for_five :
  modifiedArraySum 5 = 75 / 676 := by sorry

end NUMINAMATH_CALUDE_modified_array_sum_for_five_l3306_330615


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3306_330672

theorem closest_integer_to_cube_root : ∃ n : ℤ, 
  n = 8 ∧ ∀ m : ℤ, |n - (5^3 + 7^3)^(1/3)| ≤ |m - (5^3 + 7^3)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3306_330672


namespace NUMINAMATH_CALUDE_exactly_one_and_two_black_mutually_exclusive_not_opposite_l3306_330657

-- Define the bag of balls
def bag : Finset (Fin 4) := Finset.univ

-- Define the color of each ball (1 and 2 are red, 3 and 4 are black)
def color : Fin 4 → Bool
  | 1 => false
  | 2 => false
  | 3 => true
  | 4 => true

-- Define a draw as a pair of distinct balls
def Draw := {pair : Fin 4 × Fin 4 // pair.1 ≠ pair.2}

-- Event: Exactly one black ball is drawn
def exactly_one_black (draw : Draw) : Prop :=
  (color draw.val.1 ∧ ¬color draw.val.2) ∨ (¬color draw.val.1 ∧ color draw.val.2)

-- Event: Exactly two black balls are drawn
def exactly_two_black (draw : Draw) : Prop :=
  color draw.val.1 ∧ color draw.val.2

-- Theorem: The events are mutually exclusive but not opposite
theorem exactly_one_and_two_black_mutually_exclusive_not_opposite :
  (∀ draw : Draw, ¬(exactly_one_black draw ∧ exactly_two_black draw)) ∧
  (∃ draw : Draw, ¬exactly_one_black draw ∧ ¬exactly_two_black draw) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_and_two_black_mutually_exclusive_not_opposite_l3306_330657


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_l3306_330603

theorem largest_multiple_of_seven (n : ℤ) : n = 77 ↔ 
  (∃ k : ℤ, n = 7 * k) ∧ 
  (-n > -80) ∧
  (∀ m : ℤ, (∃ j : ℤ, m = 7 * j) → (-m > -80) → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_l3306_330603


namespace NUMINAMATH_CALUDE_quadratic_equation_theorem_l3306_330695

/-- The quadratic equation x^2 - 2(m-1)x + m^2 = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁^2 - 2*(m-1)*x₁ + m^2 = 0 ∧ x₂^2 - 2*(m-1)*x₂ + m^2 = 0

/-- The roots of the quadratic equation satisfy x₁^2 + x₂^2 = 8 - 3*x₁*x₂ -/
def roots_satisfy_condition (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁^2 - 2*(m-1)*x₁ + m^2 = 0 ∧ x₂^2 - 2*(m-1)*x₂ + m^2 = 0 ∧ x₁^2 + x₂^2 = 8 - 3*x₁*x₂

theorem quadratic_equation_theorem :
  (∀ m : ℝ, has_real_roots m → m ≤ 1/2) ∧
  (∀ m : ℝ, roots_satisfy_condition m → m = -2/5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_theorem_l3306_330695


namespace NUMINAMATH_CALUDE_fraction_simplification_l3306_330601

theorem fraction_simplification (m : ℝ) (h : m ≠ 3 ∧ m ≠ -3) : 
  12 / (m^2 - 9) + 2 / (3 - m) = -2 / (m + 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3306_330601


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3306_330644

theorem cubic_equation_solution : 27^3 + 27^3 + 27^3 = 3^10 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3306_330644


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3306_330612

/-- Given that y^3 varies inversely with z^2 and y = 3 when z = 2, 
    prove that z = √2/2 when y = 6 -/
theorem inverse_variation_problem (y z : ℝ) (k : ℝ) :
  (∀ y z, y^3 * z^2 = k) →  -- y^3 varies inversely with z^2
  (3^3 * 2^2 = k) →         -- y = 3 when z = 2
  (6^3 * z^2 = k) →         -- condition for y = 6
  z = Real.sqrt 2 / 2 :=    -- z = √2/2 when y = 6
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3306_330612


namespace NUMINAMATH_CALUDE_number_problem_l3306_330605

theorem number_problem : 
  ∃ x : ℚ, x = (3/7)*x + 200 ∧ x = 350 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3306_330605


namespace NUMINAMATH_CALUDE_period_of_inverse_a_l3306_330670

/-- Represents a 100-digit number with 1 at the start, 6 at the end, and 98 sevens in between -/
def a : ℕ := 1777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777776

/-- The period of the decimal representation of 1/n -/
def decimal_period (n : ℕ) : ℕ := sorry

theorem period_of_inverse_a : decimal_period a = 99 := by sorry

end NUMINAMATH_CALUDE_period_of_inverse_a_l3306_330670


namespace NUMINAMATH_CALUDE_expression_mod_18_l3306_330692

theorem expression_mod_18 : (234 * 18 - 23 * 9 + 5) % 18 = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_mod_18_l3306_330692


namespace NUMINAMATH_CALUDE_gasoline_price_increase_percentage_l3306_330656

def lowest_price : ℝ := 15
def highest_price : ℝ := 24

theorem gasoline_price_increase_percentage :
  (highest_price - lowest_price) / lowest_price * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_percentage_l3306_330656


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3306_330667

-- Problem 1
theorem problem_1 : (-2)^2 + (Real.sqrt 2 - 1)^0 - 1 = 4 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (A B : ℝ) (h1 : A = a - 1) (h2 : B = -a + 3) (h3 : A > B) :
  a > 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3306_330667


namespace NUMINAMATH_CALUDE_particle_movement_l3306_330630

def num_ways_to_point (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem particle_movement :
  (num_ways_to_point 5 4 = 5) ∧ (num_ways_to_point 20 18 = 190) := by
  sorry

end NUMINAMATH_CALUDE_particle_movement_l3306_330630


namespace NUMINAMATH_CALUDE_intersection_when_m_zero_range_of_m_l3306_330616

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (x m : ℝ) : Prop := x ∈ B m

-- Theorem 1: Intersection of A and B when m = 0
theorem intersection_when_m_zero : 
  A ∩ B 0 = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem 2: Range of m when q is necessary but not sufficient for p
theorem range_of_m (h : ∀ x, p x → q x 0 ∧ ¬(∀ x, q x 0 → p x)) : 
  {m : ℝ | m ≤ -2 ∨ m ≥ 4} = Set.univ := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_zero_range_of_m_l3306_330616


namespace NUMINAMATH_CALUDE_exists_multicolor_triangle_l3306_330685

/-- Represents the three possible colors for vertices -/
inductive Color
| Red
| Blue
| Yellow

/-- Represents a vertex in the triangle -/
structure Vertex where
  x : ℝ
  y : ℝ
  color : Color

/-- Represents a small equilateral triangle -/
structure SmallTriangle where
  v1 : Vertex
  v2 : Vertex
  v3 : Vertex

/-- Represents the large equilateral triangle ABC -/
structure LargeTriangle where
  n : ℕ
  smallTriangles : Array SmallTriangle

/-- Predicate to check if a vertex is on side BC -/
def onSideBC (v : Vertex) : Prop := sorry

/-- Predicate to check if a vertex is on side CA -/
def onSideCA (v : Vertex) : Prop := sorry

/-- Predicate to check if a vertex is on side AB -/
def onSideAB (v : Vertex) : Prop := sorry

/-- The main theorem to be proved -/
theorem exists_multicolor_triangle (ABC : LargeTriangle) : 
  (∀ v : Vertex, onSideBC v → v.color ≠ Color.Red) →
  (∀ v : Vertex, onSideCA v → v.color ≠ Color.Blue) →
  (∀ v : Vertex, onSideAB v → v.color ≠ Color.Yellow) →
  ∃ t : SmallTriangle, t ∈ ABC.smallTriangles ∧ 
    t.v1.color ≠ t.v2.color ∧ t.v2.color ≠ t.v3.color ∧ t.v1.color ≠ t.v3.color :=
sorry

end NUMINAMATH_CALUDE_exists_multicolor_triangle_l3306_330685


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3306_330658

theorem other_root_of_quadratic (m : ℝ) : 
  (1^2 + m*1 + 3 = 0) → 
  ∃ (α : ℝ), α ≠ 1 ∧ α^2 + m*α + 3 = 0 ∧ α = 3 := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3306_330658


namespace NUMINAMATH_CALUDE_sum_x_y_equals_twenty_l3306_330642

theorem sum_x_y_equals_twenty (x y : ℝ) 
  (h1 : |x| - x + y = 13) 
  (h2 : x - |y| + y = 7) : 
  x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_twenty_l3306_330642


namespace NUMINAMATH_CALUDE_shopping_solution_l3306_330633

/-- The cost of Liz's shopping trip -/
def shopping_problem (recipe_book_cost : ℝ) : Prop :=
  let baking_dish_cost := 2 * recipe_book_cost
  let ingredients_cost := 5 * 3
  let apron_cost := recipe_book_cost + 1
  recipe_book_cost + baking_dish_cost + ingredients_cost + apron_cost = 40

/-- The solution to the shopping problem -/
theorem shopping_solution : ∃ (recipe_book_cost : ℝ), 
  shopping_problem recipe_book_cost ∧ recipe_book_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_shopping_solution_l3306_330633


namespace NUMINAMATH_CALUDE_money_distribution_l3306_330640

theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 500)
  (h2 : A + C = 200)
  (h3 : B + C = 350) : 
  C = 50 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l3306_330640


namespace NUMINAMATH_CALUDE_factorial_15_not_divisible_by_17_l3306_330634

theorem factorial_15_not_divisible_by_17 : ¬(17 ∣ Nat.factorial 15) := by
  sorry

end NUMINAMATH_CALUDE_factorial_15_not_divisible_by_17_l3306_330634


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l3306_330660

theorem cosine_sine_identity (α : Real) :
  (∃ (x y : Real), x = 2 ∧ y = 1 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos α ^ 2 + Real.sin (2 * α) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l3306_330660


namespace NUMINAMATH_CALUDE_line_through_point_l3306_330613

theorem line_through_point (k : ℚ) : 
  (2 * k * 3 - 5 = 4 * (-4)) → k = -11/6 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l3306_330613


namespace NUMINAMATH_CALUDE_second_question_percentage_l3306_330684

theorem second_question_percentage 
  (first_correct : ℝ) 
  (neither_correct : ℝ) 
  (both_correct : ℝ) 
  (h1 : first_correct = 63) 
  (h2 : neither_correct = 20) 
  (h3 : both_correct = 33) : 
  ∃ second_correct : ℝ, 
    second_correct = 50 ∧ 
    first_correct + second_correct - both_correct = 100 - neither_correct :=
by sorry

end NUMINAMATH_CALUDE_second_question_percentage_l3306_330684


namespace NUMINAMATH_CALUDE_smallest_n_for_monochromatic_isosceles_trapezoid_l3306_330677

/-- A coloring of vertices with three colors -/
def Coloring (n : ℕ) := Fin n → Fin 3

/-- Check if four vertices form an isosceles trapezoid in an n-gon -/
def IsIsoscelesTrapezoid (n : ℕ) (v1 v2 v3 v4 : Fin n) : Prop := sorry

/-- Check if a coloring contains four vertices of the same color forming an isosceles trapezoid -/
def HasMonochromaticIsoscelesTrapezoid (n : ℕ) (c : Coloring n) : Prop :=
  ∃ (v1 v2 v3 v4 : Fin n), 
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4 ∧
    c v1 = c v2 ∧ c v1 = c v3 ∧ c v1 = c v4 ∧
    IsIsoscelesTrapezoid n v1 v2 v3 v4

/-- The main theorem -/
theorem smallest_n_for_monochromatic_isosceles_trapezoid :
  (∀ (c : Coloring 17), HasMonochromaticIsoscelesTrapezoid 17 c) ∧
  (∀ (n : ℕ), n < 17 → ∃ (c : Coloring n), ¬HasMonochromaticIsoscelesTrapezoid n c) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_monochromatic_isosceles_trapezoid_l3306_330677


namespace NUMINAMATH_CALUDE_prob_same_student_given_same_look_l3306_330662

/-- Represents a group of identical students -/
structure IdenticalGroup where
  size : Nat
  count : Nat

/-- Represents the Multiples Obfuscation Program -/
def MultiplesObfuscationProgram : List IdenticalGroup :=
  [⟨1, 1⟩, ⟨2, 1⟩, ⟨3, 1⟩, ⟨4, 1⟩, ⟨5, 1⟩, ⟨6, 1⟩, ⟨7, 1⟩, ⟨8, 1⟩]

/-- Total number of students in the program -/
def totalStudents : Nat :=
  MultiplesObfuscationProgram.foldr (fun g acc => g.size * g.count + acc) 0

/-- Number of pairs where students look the same -/
def sameLookPairs : Nat :=
  MultiplesObfuscationProgram.foldr (fun g acc => g.size * g.size * g.count + acc) 0

/-- Probability of encountering the same student twice -/
def probSameStudent : Rat :=
  totalStudents / (totalStudents * totalStudents)

/-- Probability of encountering students that look the same -/
def probSameLook : Rat :=
  sameLookPairs / (totalStudents * totalStudents)

theorem prob_same_student_given_same_look :
  probSameStudent / probSameLook = 3 / 17 := by sorry

end NUMINAMATH_CALUDE_prob_same_student_given_same_look_l3306_330662


namespace NUMINAMATH_CALUDE_min_staircase_steps_l3306_330621

theorem min_staircase_steps (a b : ℕ+) :
  ∃ (n : ℕ), n = a + b - Nat.gcd a b ∧
  (∀ (m : ℕ), m < n → ¬∃ (k : ℕ), k * a = m ∨ k * a = m + b) ∧
  (∃ (k l : ℕ), k * a = n ∧ l * a = n + b) :=
sorry

end NUMINAMATH_CALUDE_min_staircase_steps_l3306_330621


namespace NUMINAMATH_CALUDE_fraction_sum_equals_point_two_l3306_330632

theorem fraction_sum_equals_point_two :
  2 / 40 + 4 / 80 + 6 / 120 + 9 / 180 = (0.2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_point_two_l3306_330632


namespace NUMINAMATH_CALUDE_sum_lent_is_1000_l3306_330645

/-- Proves that the sum lent is $1000 given the specified conditions --/
theorem sum_lent_is_1000 (annual_rate : ℝ) (duration : ℝ) (interest_difference : ℝ) :
  annual_rate = 0.06 →
  duration = 8 →
  interest_difference = 520 →
  ∃ (P : ℝ), P * annual_rate * duration = P - interest_difference ∧ P = 1000 := by
  sorry

#check sum_lent_is_1000

end NUMINAMATH_CALUDE_sum_lent_is_1000_l3306_330645


namespace NUMINAMATH_CALUDE_carol_peanuts_l3306_330623

/-- Given that Carol initially collects 2 peanuts and receives 5 more from her father,
    prove that Carol has a total of 7 peanuts. -/
theorem carol_peanuts (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 2 → received = 5 → total = initial + received → total = 7 := by
sorry

end NUMINAMATH_CALUDE_carol_peanuts_l3306_330623


namespace NUMINAMATH_CALUDE_max_prob_second_game_l3306_330636

variable (p₁ p₂ p₃ : ℝ)

def P_A := 2 * (p₁ * (p₂ + p₃) - 2 * p₁ * p₂ * p₃)
def P_B := 2 * (p₂ * (p₁ + p₃) - 2 * p₁ * p₂ * p₃)
def P_C := 2 * (p₁ * p₃ + p₂ * p₃ - 2 * p₁ * p₂ * p₃)

theorem max_prob_second_game (h1 : 0 < p₁) (h2 : p₁ < p₂) (h3 : p₂ < p₃) :
  P_C p₁ p₂ p₃ > P_A p₁ p₂ p₃ ∧ P_C p₁ p₂ p₃ > P_B p₁ p₂ p₃ :=
by sorry

end NUMINAMATH_CALUDE_max_prob_second_game_l3306_330636


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_6_mod_19_l3306_330600

theorem least_five_digit_congruent_to_6_mod_19 :
  ∃ n : ℕ, 
    n ≥ 10000 ∧ 
    n < 100000 ∧ 
    n % 19 = 6 ∧ 
    ∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 19 = 6 → m ≥ n :=
by
  use 10011
  sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_6_mod_19_l3306_330600


namespace NUMINAMATH_CALUDE_train_length_l3306_330617

/-- The length of a train given crossing times and platform length -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : platform_length = 350)
  (h2 : platform_time = 39)
  (h3 : pole_time = 18) :
  (platform_length * pole_time) / (platform_time - pole_time) = 300 := by
sorry

end NUMINAMATH_CALUDE_train_length_l3306_330617


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l3306_330671

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ - 4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l3306_330671


namespace NUMINAMATH_CALUDE_max_m_for_right_angle_l3306_330620

-- Define the circle C in polar coordinates
def circle_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

-- Define the line l
def line_l (x y m : ℝ) : Prop := y = 2 * x + 2 * m

-- Define the rectangular coordinates of circle C
def circle_C_rect (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2^2

-- Theorem statement
theorem max_m_for_right_angle (m : ℝ) :
  (∃ x y : ℝ, circle_C_rect x y ∧ line_l x y m) →
  m ≤ Real.sqrt 5 - 2 :=
sorry

end NUMINAMATH_CALUDE_max_m_for_right_angle_l3306_330620


namespace NUMINAMATH_CALUDE_sales_theorem_l3306_330653

def sales_problem (last_four_months : List ℕ) (sixth_month : ℕ) (average : ℕ) : Prop :=
  let total_six_months := average * 6
  let sum_last_four := last_four_months.sum
  let first_month := total_six_months - (sum_last_four + sixth_month)
  first_month = 5420

theorem sales_theorem :
  sales_problem [5660, 6200, 6350, 6500] 7070 6200 := by
  sorry

end NUMINAMATH_CALUDE_sales_theorem_l3306_330653


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l3306_330694

theorem theater_ticket_sales (adult_price kid_price profit kid_tickets : ℕ) 
  (h1 : adult_price = 6)
  (h2 : kid_price = 2)
  (h3 : profit = 750)
  (h4 : kid_tickets = 75) :
  ∃ (adult_tickets : ℕ), adult_tickets * adult_price + kid_tickets * kid_price = profit ∧
                          adult_tickets + kid_tickets = 175 :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l3306_330694


namespace NUMINAMATH_CALUDE_joan_sandwiches_l3306_330647

/-- Represents the number of sandwiches of each type -/
structure Sandwiches where
  ham : ℕ
  grilledCheese : ℕ

/-- Represents the amount of cheese slices used -/
structure CheeseUsed where
  cheddar : ℕ
  swiss : ℕ
  gouda : ℕ

/-- Calculates the total cheese used for a given number of sandwiches -/
def totalCheeseUsed (s : Sandwiches) : CheeseUsed :=
  { cheddar := s.ham + 2 * s.grilledCheese,
    swiss := s.ham,
    gouda := s.grilledCheese }

/-- The main theorem to prove -/
theorem joan_sandwiches :
  ∃ (s : Sandwiches),
    s.ham = 8 ∧
    totalCheeseUsed s = { cheddar := 40, swiss := 20, gouda := 30 } ∧
    s.grilledCheese = 16 := by
  sorry


end NUMINAMATH_CALUDE_joan_sandwiches_l3306_330647


namespace NUMINAMATH_CALUDE_janet_stickers_l3306_330641

theorem janet_stickers (x : ℕ) : 
  x + 53 = 56 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_janet_stickers_l3306_330641


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3306_330610

theorem exponent_multiplication (a : ℝ) : a * a^2 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3306_330610


namespace NUMINAMATH_CALUDE_sequence_appearance_l3306_330654

def sequence_digit (a b c : ℕ) : ℕ :=
  (a + b + c) % 10

def appears_in_sequence (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≥ 3 ∧
  (n / 1000 = sequence_digit 2 1 9) ∧
  (n / 100 % 10 = sequence_digit 1 9 (sequence_digit 2 1 9)) ∧
  (n / 10 % 10 = sequence_digit 9 (sequence_digit 2 1 9) (sequence_digit 1 9 (sequence_digit 2 1 9))) ∧
  (n % 10 = sequence_digit (sequence_digit 2 1 9) (sequence_digit 1 9 (sequence_digit 2 1 9)) (sequence_digit 9 (sequence_digit 2 1 9) (sequence_digit 1 9 (sequence_digit 2 1 9))))

theorem sequence_appearance :
  (¬ appears_in_sequence 1113 ∧ appears_in_sequence 2226 ∧ appears_in_sequence 2125 ∧ appears_in_sequence 2215) ∨
  (appears_in_sequence 1113 ∧ ¬ appears_in_sequence 2226 ∧ appears_in_sequence 2125 ∧ appears_in_sequence 2215) ∨
  (appears_in_sequence 1113 ∧ appears_in_sequence 2226 ∧ ¬ appears_in_sequence 2125 ∧ appears_in_sequence 2215) ∨
  (appears_in_sequence 1113 ∧ appears_in_sequence 2226 ∧ appears_in_sequence 2125 ∧ ¬ appears_in_sequence 2215) :=
by sorry

end NUMINAMATH_CALUDE_sequence_appearance_l3306_330654


namespace NUMINAMATH_CALUDE_total_wood_planks_l3306_330635

def initial_planks : ℕ := 15
def charlie_planks : ℕ := 10
def father_planks : ℕ := 10

theorem total_wood_planks : 
  initial_planks + charlie_planks + father_planks = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_wood_planks_l3306_330635


namespace NUMINAMATH_CALUDE_quadratic_properties_quadratic_max_conditions_l3306_330683

-- Define the quadratic function
def quadratic_function (b c x : ℝ) : ℝ := -x^2 + b*x + c

-- Theorem for part 1
theorem quadratic_properties :
  let f := quadratic_function 4 3
  ∃ (vertex_x vertex_y : ℝ),
    (∀ x, f x ≤ f vertex_x) ∧
    vertex_x = 2 ∧
    vertex_y = 7 ∧
    (∀ x, -1 ≤ x ∧ x ≤ 3 → -2 ≤ f x ∧ f x ≤ 7) :=
sorry

-- Theorem for part 2
theorem quadratic_max_conditions :
  ∃ (b c : ℝ),
    (∀ x ≤ 0, quadratic_function b c x ≤ 2) ∧
    (∀ x > 0, quadratic_function b c x ≤ 3) ∧
    (∃ x ≤ 0, quadratic_function b c x = 2) ∧
    (∃ x > 0, quadratic_function b c x = 3) ∧
    b = 2 ∧
    c = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_quadratic_max_conditions_l3306_330683


namespace NUMINAMATH_CALUDE_candy_calculation_l3306_330663

/-- 
Given the initial amount of candy, the amount eaten, and the amount received,
prove that the final amount of candy is equal to the initial amount minus
the eaten amount plus the received amount.
-/
theorem candy_calculation (initial eaten received : ℕ) :
  initial - eaten + received = (initial - eaten) + received := by
  sorry

end NUMINAMATH_CALUDE_candy_calculation_l3306_330663


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l3306_330606

/-- The atomic weight of Hydrogen in atomic mass units (amu) -/
def atomic_weight_H : ℝ := 1.008

/-- The atomic weight of Chromium in atomic mass units (amu) -/
def atomic_weight_Cr : ℝ := 51.996

/-- The atomic weight of Oxygen in atomic mass units (amu) -/
def atomic_weight_O : ℝ := 15.999

/-- The number of Hydrogen atoms in the compound -/
def num_H : ℕ := 2

/-- The number of Chromium atoms in the compound -/
def num_Cr : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 4

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight : ℝ := 
  (num_H : ℝ) * atomic_weight_H + 
  (num_Cr : ℝ) * atomic_weight_Cr + 
  (num_O : ℝ) * atomic_weight_O

theorem molecular_weight_calculation : 
  molecular_weight = 118.008 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l3306_330606


namespace NUMINAMATH_CALUDE_three_fourths_of_four_fifths_of_two_thirds_l3306_330688

theorem three_fourths_of_four_fifths_of_two_thirds : (3 : ℚ) / 4 * (4 : ℚ) / 5 * (2 : ℚ) / 3 = (2 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_of_four_fifths_of_two_thirds_l3306_330688


namespace NUMINAMATH_CALUDE_april_earnings_l3306_330638

/-- The price of a rose in dollars -/
def rose_price : ℕ := 7

/-- The price of a lily in dollars -/
def lily_price : ℕ := 5

/-- The initial number of roses -/
def initial_roses : ℕ := 9

/-- The initial number of lilies -/
def initial_lilies : ℕ := 6

/-- The remaining number of roses -/
def remaining_roses : ℕ := 4

/-- The remaining number of lilies -/
def remaining_lilies : ℕ := 2

/-- The total earnings from the sale -/
def total_earnings : ℕ := 55

theorem april_earnings : 
  (initial_roses - remaining_roses) * rose_price + 
  (initial_lilies - remaining_lilies) * lily_price = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_april_earnings_l3306_330638


namespace NUMINAMATH_CALUDE_g_limit_pos_infinity_g_limit_neg_infinity_l3306_330626

/-- The function g(x) = -3x^4 + 5x^3 - 6 -/
def g (x : ℝ) : ℝ := -3 * x^4 + 5 * x^3 - 6

/-- The limit of g(x) approaches negative infinity as x approaches positive infinity -/
theorem g_limit_pos_infinity :
  ∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M :=
sorry

/-- The limit of g(x) approaches negative infinity as x approaches negative infinity -/
theorem g_limit_neg_infinity :
  ∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x < M :=
sorry

end NUMINAMATH_CALUDE_g_limit_pos_infinity_g_limit_neg_infinity_l3306_330626


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3306_330614

theorem negation_of_proposition (p : Prop) : 
  (¬(∀ a : ℝ, a > 0 ∧ a ≠ 1 → ∃ x : ℝ, a * x - x - a = 0)) ↔ 
  (∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, a * x - x - a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3306_330614


namespace NUMINAMATH_CALUDE_mean_home_runs_l3306_330650

theorem mean_home_runs : 
  let players_with_5 := 3
  let players_with_6 := 4
  let players_with_8 := 2
  let players_with_9 := 1
  let players_with_11 := 1
  let total_players := players_with_5 + players_with_6 + players_with_8 + players_with_9 + players_with_11
  let total_home_runs := 5 * players_with_5 + 6 * players_with_6 + 8 * players_with_8 + 9 * players_with_9 + 11 * players_with_11
  (total_home_runs : ℚ) / total_players = 75 / 11 := by
sorry

end NUMINAMATH_CALUDE_mean_home_runs_l3306_330650


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3306_330697

/-- Given that (1+i)z = |-4i|, prove that z = 2 - 2i --/
theorem complex_equation_solution :
  ∀ z : ℂ, (Complex.I + 1) * z = Complex.abs (-4 * Complex.I) → z = 2 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3306_330697


namespace NUMINAMATH_CALUDE_final_piggy_bank_amount_l3306_330681

def piggy_bank_savings (initial_amount : ℝ) (weekly_allowance : ℝ) (savings_fraction : ℝ) (weeks : ℕ) : ℝ :=
  initial_amount + (weekly_allowance * savings_fraction * weeks)

theorem final_piggy_bank_amount :
  piggy_bank_savings 43 10 0.5 8 = 83 := by
  sorry

end NUMINAMATH_CALUDE_final_piggy_bank_amount_l3306_330681


namespace NUMINAMATH_CALUDE_rectangular_box_dimensions_l3306_330679

theorem rectangular_box_dimensions (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A * B = 50 →
  A * C = 90 →
  B * C = 100 →
  A + B + C = 24 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_dimensions_l3306_330679


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3306_330675

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 - x + 2

-- Define the solution set
def solution_set : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x ≥ 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3306_330675


namespace NUMINAMATH_CALUDE_perfume_fundraising_l3306_330665

/-- The amount of additional money needed to buy a perfume --/
def additional_money_needed (perfume_cost initial_christian initial_sue yards_mowed yard_price dogs_walked dog_price : ℚ) : ℚ :=
  perfume_cost - (initial_christian + initial_sue + yards_mowed * yard_price + dogs_walked * dog_price)

/-- Theorem stating the additional money needed is $6.00 --/
theorem perfume_fundraising :
  additional_money_needed 50 5 7 4 5 6 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_perfume_fundraising_l3306_330665


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3306_330651

theorem line_passes_through_point :
  ∀ (m : ℝ), m * (-1) - 2 + m + 2 = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3306_330651


namespace NUMINAMATH_CALUDE_root_sum_ratio_l3306_330637

theorem root_sum_ratio (k₁ k₂ : ℝ) : 
  (∃ p q : ℝ, (k₁ * (p^2 - 2*p) + 3*p + 7 = 0 ∧ 
               k₂ * (q^2 - 2*q) + 3*q + 7 = 0) ∧
              (p / q + q / p = 6 / 7)) →
  k₁ / k₂ + k₂ / k₁ = 14 := by
sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l3306_330637


namespace NUMINAMATH_CALUDE_peanut_butter_servings_l3306_330643

/-- The amount of peanut butter in the jar, in tablespoons -/
def jar_amount : ℚ := 37 + 4/5

/-- The amount of peanut butter in one serving, in tablespoons -/
def serving_size : ℚ := 1 + 1/2

/-- The number of servings in the jar -/
def number_of_servings : ℚ := jar_amount / serving_size

theorem peanut_butter_servings : number_of_servings = 25 + 1/5 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_servings_l3306_330643


namespace NUMINAMATH_CALUDE_delta_negative_two_three_l3306_330648

-- Define the Delta operation
def Delta (a b : ℝ) : ℝ := a * b^2 + b + 1

-- Theorem statement
theorem delta_negative_two_three : Delta (-2) 3 = -14 := by
  sorry

end NUMINAMATH_CALUDE_delta_negative_two_three_l3306_330648


namespace NUMINAMATH_CALUDE_students_just_passed_l3306_330625

theorem students_just_passed (total_students : ℕ) 
  (first_division_percent : ℚ) (second_division_percent : ℚ) :
  total_students = 300 →
  first_division_percent = 26 / 100 →
  second_division_percent = 54 / 100 →
  (total_students : ℚ) * (1 - first_division_percent - second_division_percent) = 60 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l3306_330625


namespace NUMINAMATH_CALUDE_total_milks_taken_l3306_330622

/-- The total number of milks taken is the sum of all individual milk selections. -/
theorem total_milks_taken (chocolate : ℕ) (strawberry : ℕ) (regular : ℕ) (almond : ℕ) (soy : ℕ)
  (h1 : chocolate = 120)
  (h2 : strawberry = 315)
  (h3 : regular = 230)
  (h4 : almond = 145)
  (h5 : soy = 97) :
  chocolate + strawberry + regular + almond + soy = 907 := by
  sorry

end NUMINAMATH_CALUDE_total_milks_taken_l3306_330622


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l3306_330669

theorem like_terms_exponent_sum (m n : ℤ) : 
  (∃ (k : ℚ), k * x * y^2 = x^(m-2) * y^(n+3)) → m + n = 2 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l3306_330669


namespace NUMINAMATH_CALUDE_impossible_equal_sum_configuration_l3306_330602

theorem impossible_equal_sum_configuration : ¬ ∃ (a b c d e f : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
   d ≠ e ∧ d ≠ f ∧
   e ≠ f) ∧
  (a + d + e = b + d + f) ∧
  (a + d + e = c + e + f) ∧
  (b + d + f = c + e + f) :=
by
  sorry

end NUMINAMATH_CALUDE_impossible_equal_sum_configuration_l3306_330602


namespace NUMINAMATH_CALUDE_painted_cubes_count_l3306_330673

/-- Represents a 3D shape composed of unit cubes -/
structure CubeShape where
  top_layer : Nat
  middle_layer : Nat
  bottom_layer : Nat
  unpainted_cubes : Nat

/-- Calculates the total number of cubes in the shape -/
def total_cubes (shape : CubeShape) : Nat :=
  shape.top_layer + shape.middle_layer + shape.bottom_layer

/-- Calculates the number of cubes with at least one face painted -/
def painted_cubes (shape : CubeShape) : Nat :=
  total_cubes shape - shape.unpainted_cubes

/-- Theorem stating the number of cubes with at least one face painted -/
theorem painted_cubes_count (shape : CubeShape) 
  (h1 : shape.top_layer = 9)
  (h2 : shape.middle_layer = 16)
  (h3 : shape.bottom_layer = 9)
  (h4 : shape.unpainted_cubes = 26) :
  painted_cubes shape = 8 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l3306_330673


namespace NUMINAMATH_CALUDE_fractional_to_linear_equation_l3306_330646

/-- Given the fractional equation 2/x = 1/(x-1), prove that multiplying both sides
    by x(x-1) results in a linear equation. -/
theorem fractional_to_linear_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  ∃ (a b : ℝ), x * (x - 1) * (2 / x) = a * x + b :=
sorry

end NUMINAMATH_CALUDE_fractional_to_linear_equation_l3306_330646


namespace NUMINAMATH_CALUDE_rhombus_side_length_l3306_330690

/-- A rhombus with a perimeter of 60 centimeters has a side length of 15 centimeters. -/
theorem rhombus_side_length (perimeter : ℝ) (h1 : perimeter = 60) : 
  perimeter / 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l3306_330690


namespace NUMINAMATH_CALUDE_distinct_permutations_count_l3306_330659

def sequence_length : ℕ := 6
def count_of_twos : ℕ := 3
def count_of_sqrt_threes : ℕ := 2
def count_of_fives : ℕ := 1

theorem distinct_permutations_count :
  (sequence_length.factorial) / (count_of_twos.factorial * count_of_sqrt_threes.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_distinct_permutations_count_l3306_330659


namespace NUMINAMATH_CALUDE_nephews_ages_sum_l3306_330674

theorem nephews_ages_sum :
  ∀ (a b c d : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →  -- single-digit
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- distinct
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- positive
    ((a * b = 36 ∧ c * d = 40) ∨ (a * c = 36 ∧ b * d = 40) ∨ 
     (a * d = 36 ∧ b * c = 40) ∨ (b * c = 36 ∧ a * d = 40) ∨ 
     (b * d = 36 ∧ a * c = 40) ∨ (c * d = 36 ∧ a * b = 40)) →
    a + b + c + d = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_nephews_ages_sum_l3306_330674


namespace NUMINAMATH_CALUDE_solution_implies_sum_l3306_330664

/-- The function f(x) = |x+1| + |x-3| -/
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

/-- The function g(x) = a - |x-2| -/
def g (a x : ℝ) : ℝ := a - |x - 2|

/-- The theorem stating that if the solution set of f(x) < g(x) is (b, 7/2), then a + b = 6 -/
theorem solution_implies_sum (a b : ℝ) :
  (∀ x, f x < g a x ↔ b < x ∧ x < 7/2) →
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_sum_l3306_330664


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l3306_330686

/-- Calculates the total number of wheels in a parking lot --/
def total_wheels (num_cars num_motorcycles num_trucks num_vans : ℕ) : ℕ :=
  let car_wheels := 4
  let motorcycle_wheels := 2
  let truck_wheels := 6
  let van_wheels := 4
  num_cars * car_wheels + 
  num_motorcycles * motorcycle_wheels + 
  num_trucks * truck_wheels + 
  num_vans * van_wheels

/-- The number of wheels in Dylan's parents' vehicles --/
def parents_wheels : ℕ := 8

theorem parking_lot_wheels : 
  total_wheels 7 4 3 2 + parents_wheels = 62 := by sorry

end NUMINAMATH_CALUDE_parking_lot_wheels_l3306_330686


namespace NUMINAMATH_CALUDE_dusty_change_l3306_330609

/-- Represents the price of a single layer cake slice in dollars -/
def single_layer_price : ℕ := 4

/-- Represents the price of a double layer cake slice in dollars -/
def double_layer_price : ℕ := 7

/-- Represents the number of single layer cake slices Dusty buys -/
def single_layer_quantity : ℕ := 7

/-- Represents the number of double layer cake slices Dusty buys -/
def double_layer_quantity : ℕ := 5

/-- Represents the amount Dusty pays with in dollars -/
def payment : ℕ := 100

/-- Theorem stating that Dusty's change is $37 -/
theorem dusty_change : 
  payment - (single_layer_price * single_layer_quantity + double_layer_price * double_layer_quantity) = 37 := by
  sorry

end NUMINAMATH_CALUDE_dusty_change_l3306_330609


namespace NUMINAMATH_CALUDE_oak_grove_library_books_l3306_330628

theorem oak_grove_library_books (total_books : ℕ) (public_library_books : ℕ) 
  (h1 : total_books = 7092) (h2 : public_library_books = 1986) :
  total_books - public_library_books = 5106 := by
  sorry

end NUMINAMATH_CALUDE_oak_grove_library_books_l3306_330628


namespace NUMINAMATH_CALUDE_expansion_gameplay_hours_l3306_330691

/-- Calculates the hours of gameplay added by an expansion given the total gameplay hours,
    percentage of boring gameplay, and total enjoyable gameplay hours. -/
theorem expansion_gameplay_hours
  (total_hours : ℝ)
  (boring_percentage : ℝ)
  (total_enjoyable_hours : ℝ)
  (h1 : total_hours = 100)
  (h2 : boring_percentage = 0.8)
  (h3 : total_enjoyable_hours = 50) :
  total_enjoyable_hours - (1 - boring_percentage) * total_hours = 30 :=
by sorry

end NUMINAMATH_CALUDE_expansion_gameplay_hours_l3306_330691


namespace NUMINAMATH_CALUDE_zoo_animal_difference_l3306_330619

theorem zoo_animal_difference : ∀ (parrots snakes monkeys elephants zebras : ℕ),
  parrots = 8 →
  snakes = 3 * parrots →
  monkeys = 2 * snakes →
  elephants = (parrots + snakes) / 2 →
  zebras + 3 = elephants →
  monkeys - zebras = 35 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_difference_l3306_330619


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3306_330608

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and eccentricity √3,
    where the line x = -a²/c (c is the semi-latus rectum) coincides with the latus rectum
    of the parabola y² = 4x, prove that the equation of this hyperbola is x²/3 - y²/6 = 1. -/
theorem hyperbola_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (c / a = Real.sqrt 3) → (a^2 / c = 1) → (x^2 / a^2 - y^2 / b^2 = 1) →
  (x^2 / 3 - y^2 / 6 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3306_330608


namespace NUMINAMATH_CALUDE_zoe_earnings_per_candy_bar_l3306_330661

def trip_cost : ℚ := 485
def grandma_contribution : ℚ := 250
def candy_bars_to_sell : ℕ := 188

theorem zoe_earnings_per_candy_bar :
  (trip_cost - grandma_contribution) / candy_bars_to_sell = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_zoe_earnings_per_candy_bar_l3306_330661


namespace NUMINAMATH_CALUDE_exists_n_with_1000_steps_l3306_330631

def largest_prime_le (n : ℕ) : ℕ :=
  (Finset.filter Nat.Prime (Finset.range (n + 1))).max' sorry

def reduction_process (n : ℕ) : ℕ → ℕ
| 0 => 0
| (k + 1) => 
  let n' := n - largest_prime_le n
  if n' ≤ 1 then n' else reduction_process n' k

theorem exists_n_with_1000_steps : 
  ∃ N : ℕ, reduction_process N 1000 = 0 ∧ ∀ k < 1000, reduction_process N k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_exists_n_with_1000_steps_l3306_330631


namespace NUMINAMATH_CALUDE_b_age_is_eighteen_l3306_330699

/-- Given three people a, b, and c, where:
    - a is two years older than b
    - b is twice as old as c
    - The total of their ages is 47
    Prove that b is 18 years old. -/
theorem b_age_is_eighteen (a b c : ℕ) 
    (h1 : a = b + 2) 
    (h2 : b = 2 * c) 
    (h3 : a + b + c = 47) : 
  b = 18 := by
  sorry

end NUMINAMATH_CALUDE_b_age_is_eighteen_l3306_330699


namespace NUMINAMATH_CALUDE_learning_machine_price_reduction_l3306_330639

/-- Represents the price reduction scenario of a learning machine -/
def price_reduction_equation (initial_price final_price : ℝ) (num_reductions : ℕ) (x : ℝ) : Prop :=
  initial_price * (1 - x)^num_reductions = final_price

/-- The equation 2000(1-x)^2 = 1280 correctly represents the given price reduction scenario -/
theorem learning_machine_price_reduction :
  price_reduction_equation 2000 1280 2 x ↔ 2000 * (1 - x)^2 = 1280 :=
sorry

end NUMINAMATH_CALUDE_learning_machine_price_reduction_l3306_330639


namespace NUMINAMATH_CALUDE_divisible_by_2_3_5_less_than_300_l3306_330680

theorem divisible_by_2_3_5_less_than_300 : 
  (Finset.filter (fun n : ℕ => n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) (Finset.range 300)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_2_3_5_less_than_300_l3306_330680


namespace NUMINAMATH_CALUDE_external_tangent_distance_l3306_330618

/-- Given two externally touching circles with radii R and r, 
    the distance AB between the points where their common external tangent 
    touches the circles is equal to 2√(Rr) -/
theorem external_tangent_distance (R r : ℝ) (hR : R > 0) (hr : r > 0) :
  ∃ (AB : ℝ), AB = 2 * Real.sqrt (R * r) := by
  sorry

end NUMINAMATH_CALUDE_external_tangent_distance_l3306_330618


namespace NUMINAMATH_CALUDE_special_function_ratio_bounds_l3306_330678

open Real

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := Set.Ioi 0
  pos : ∀ x ∈ domain, f x > 0
  deriv_bound : ∀ x ∈ domain, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x

/-- The main theorem -/
theorem special_function_ratio_bounds (sf : SpecialFunction) :
    1/8 < sf.f 1 / sf.f 2 ∧ sf.f 1 / sf.f 2 < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_special_function_ratio_bounds_l3306_330678


namespace NUMINAMATH_CALUDE_sin_period_from_symmetric_center_l3306_330696

/-- Given a function f(x) = sin(ωx), if the minimum distance from a symmetric center
    to the axis of symmetry is π/4, then the minimum positive period of f(x) is π. -/
theorem sin_period_from_symmetric_center (ω : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (ω * x)
  let min_distance_to_axis : ℝ := π / 4
  let period : ℝ := 2 * π / ω
  min_distance_to_axis = period / 2 → period = π :=
by
  sorry


end NUMINAMATH_CALUDE_sin_period_from_symmetric_center_l3306_330696


namespace NUMINAMATH_CALUDE_sum_of_eight_numbers_l3306_330607

/-- Given a list of 8 real numbers with an average of 5.7, prove that their sum is 45.6 -/
theorem sum_of_eight_numbers (numbers : List ℝ) 
  (h1 : numbers.length = 8)
  (h2 : numbers.sum / numbers.length = 5.7) : 
  numbers.sum = 45.6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_eight_numbers_l3306_330607


namespace NUMINAMATH_CALUDE_bouncy_ball_difference_l3306_330666

/-- Proves that the difference between red and yellow bouncy balls is 18 -/
theorem bouncy_ball_difference :
  ∀ (red_packs yellow_packs balls_per_pack : ℕ),
  red_packs = 5 →
  yellow_packs = 4 →
  balls_per_pack = 18 →
  red_packs * balls_per_pack - yellow_packs * balls_per_pack = 18 :=
by
  sorry

#check bouncy_ball_difference

end NUMINAMATH_CALUDE_bouncy_ball_difference_l3306_330666


namespace NUMINAMATH_CALUDE_brick_width_calculation_l3306_330624

/-- Proves that given a courtyard of 25 meters by 16 meters, to be paved with 20,000 bricks of length 20 cm, the width of each brick must be 10 cm. -/
theorem brick_width_calculation (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_length : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 16 →
  brick_length = 0.2 →
  total_bricks = 20000 →
  ∃ (brick_width : ℝ), 
    brick_width = 0.1 ∧ 
    (courtyard_length * courtyard_width * 10000) = (brick_length * brick_width * total_bricks) :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l3306_330624


namespace NUMINAMATH_CALUDE_uba_capital_suvs_l3306_330668

/-- Represents the number of SUVs purchased by UBA Capital --/
def num_suvs (total_vehicles : ℕ) : ℕ :=
  let toyota_count := (9 * total_vehicles) / 10
  let honda_count := total_vehicles - toyota_count
  let toyota_suvs := (90 * toyota_count) / 100
  let honda_suvs := (10 * honda_count) / 100
  toyota_suvs + honda_suvs

/-- Theorem stating that the number of SUVs purchased is 8 --/
theorem uba_capital_suvs :
  ∃ (total_vehicles : ℕ), num_suvs total_vehicles = 8 :=
sorry

end NUMINAMATH_CALUDE_uba_capital_suvs_l3306_330668


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l3306_330655

/-- The number of walnut trees in the park before planting -/
def trees_before : ℕ := 22

/-- The number of walnut trees in the park after planting -/
def trees_after : ℕ := 55

/-- The number of walnut trees planted -/
def trees_planted : ℕ := trees_after - trees_before

theorem walnut_trees_planted : trees_planted = 33 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l3306_330655


namespace NUMINAMATH_CALUDE_max_minute_hands_l3306_330611

/-- Represents the number of coincidences per hour for a pair of hands moving in opposite directions -/
def coincidences_per_pair : ℕ := 120

/-- Represents the total number of coincidences observed in one hour -/
def total_coincidences : ℕ := 54

/-- Proves that the maximum number of minute hands is 28 given the conditions -/
theorem max_minute_hands : 
  ∃ (m n : ℕ), 
    m * n = total_coincidences / 2 ∧ 
    m + n ≤ 28 ∧ 
    ∀ (k l : ℕ), k * l = total_coincidences / 2 → k + l ≤ m + n :=
by sorry

end NUMINAMATH_CALUDE_max_minute_hands_l3306_330611


namespace NUMINAMATH_CALUDE_M_equals_N_l3306_330693

/-- Definition of set M -/
def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}

/-- Definition of set N -/
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

/-- Theorem stating that M equals N -/
theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l3306_330693


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_correct_l3306_330629

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to x-axis -/
def symmetricPointXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem symmetric_point_x_axis_correct :
  let P : Point2D := { x := 5, y := -3 }
  let symmetricP : Point2D := { x := 5, y := 3 }
  symmetricPointXAxis P = symmetricP := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_correct_l3306_330629


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inverse_l3306_330676

theorem sqrt_x_plus_inverse (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 49) : 
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 51 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inverse_l3306_330676


namespace NUMINAMATH_CALUDE_product_one_sum_square_and_products_geq_ten_l3306_330698

theorem product_one_sum_square_and_products_geq_ten 
  (a b c d : ℝ) (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_product_one_sum_square_and_products_geq_ten_l3306_330698


namespace NUMINAMATH_CALUDE_sequence_is_cubic_polynomial_l3306_330682

def fourth_difference (u : ℕ → ℝ) : ℕ → ℝ :=
  λ n => u (n + 4) - 4 * u (n + 3) + 6 * u (n + 2) - 4 * u (n + 1) + u n

theorem sequence_is_cubic_polynomial 
  (u : ℕ → ℝ) 
  (h : ∀ n, fourth_difference u n = 0) : 
  ∃ a b c d : ℝ, ∀ n, u n = a * n^3 + b * n^2 + c * n + d :=
sorry

end NUMINAMATH_CALUDE_sequence_is_cubic_polynomial_l3306_330682


namespace NUMINAMATH_CALUDE_smallest_n_divisible_l3306_330689

theorem smallest_n_divisible : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m^2 % 24 = 0 ∧ m^3 % 450 = 0 → n ≤ m) ∧
  n^2 % 24 = 0 ∧ n^3 % 450 = 0 := by
  use 60
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_l3306_330689


namespace NUMINAMATH_CALUDE_two_leq_three_l3306_330649

theorem two_leq_three : 2 ≤ 3 := by sorry

end NUMINAMATH_CALUDE_two_leq_three_l3306_330649


namespace NUMINAMATH_CALUDE_trig_simplification_l3306_330604

theorem trig_simplification :
  (Real.cos (40 * π / 180)) / (Real.cos (25 * π / 180) * Real.sqrt (1 - Real.sin (40 * π / 180))) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3306_330604


namespace NUMINAMATH_CALUDE_total_legs_on_farm_l3306_330652

/-- The number of legs for each animal type -/
def legs_per_animal (animal : String) : ℕ :=
  match animal with
  | "chicken" => 2
  | "sheep" => 4
  | _ => 0

/-- The total number of animals on the farm -/
def total_animals : ℕ := 12

/-- The number of chickens on the farm -/
def num_chickens : ℕ := 5

/-- Theorem stating the total number of animal legs on the farm -/
theorem total_legs_on_farm : 
  (num_chickens * legs_per_animal "chicken") + 
  ((total_animals - num_chickens) * legs_per_animal "sheep") = 38 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_on_farm_l3306_330652


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3306_330687

theorem chess_tournament_games (n : ℕ) (h : n = 18) : 
  (n * (n - 1)) / 2 = 153 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3306_330687
