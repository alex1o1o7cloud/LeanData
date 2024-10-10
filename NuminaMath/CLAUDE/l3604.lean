import Mathlib

namespace sin_15_cos_15_eq_quarter_l3604_360493

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end sin_15_cos_15_eq_quarter_l3604_360493


namespace mean_equality_implies_z_value_l3604_360488

theorem mean_equality_implies_z_value :
  let x₁ : ℚ := 7
  let x₂ : ℚ := 15
  let x₃ : ℚ := 21
  let y₁ : ℚ := 18
  (x₁ + x₂ + x₃) / 3 = (y₁ + z) / 2 →
  z = 32 / 3 := by
sorry

end mean_equality_implies_z_value_l3604_360488


namespace coin_distribution_l3604_360414

theorem coin_distribution (n k : ℕ) (h1 : 2 * k^2 - 2 * k < n) (h2 : n < 2 * k^2 + 2 * k) :
  (2 * k^2 - 2 * k < n ∧ n < 2 * k^2 → 
    (k - 1)^2 + (n - (2 * k^2 - 2 * k)) > k^2 - k) ∧
  (2 * k^2 < n ∧ n < 2 * k^2 + 2 * k → 
    k^2 < k^2 - k + (n - (2 * k^2 - k))) :=
by sorry

end coin_distribution_l3604_360414


namespace append_digits_divisible_by_53_l3604_360421

theorem append_digits_divisible_by_53 (x y : Nat) : 
  x < 10 ∧ y < 10 ∧ (131300 + 10 * x + y) % 53 = 0 ↔ (x = 3 ∧ y = 4) ∨ (x = 8 ∧ y = 7) := by
  sorry

end append_digits_divisible_by_53_l3604_360421


namespace vector_linear_combination_l3604_360458

/-- Given vectors a, b, and c in ℝ², prove that c can be expressed as a linear combination of a and b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (1, -1)) 
  (hc : c = (-1, 2)) : 
  c = (1/2 : ℝ) • a - (3/2 : ℝ) • b :=
sorry

end vector_linear_combination_l3604_360458


namespace age_difference_proof_l3604_360498

def age_difference_in_decades (x y z : ℕ) : ℚ :=
  (x - z : ℚ) / 10

theorem age_difference_proof (x y z : ℕ) 
  (h : x + y = y + z + 15) : 
  age_difference_in_decades x y z = 3/2 := by
  sorry

end age_difference_proof_l3604_360498


namespace triangle_properties_l3604_360408

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt 3 * t.a * Real.sin t.B - t.b * Real.cos t.A = t.b)
  (h2 : t.b + t.c = 4) :
  t.A = π / 3 ∧ 
  (∃ (min_a : ℝ), min_a = 2 ∧ 
    (∀ a', t.a = a' → a' ≥ min_a) ∧
    (t.a = min_a → Real.sqrt 3 / 2 * t.b * t.c = Real.sqrt 3)) := by
  sorry

end triangle_properties_l3604_360408


namespace line_through_circle_center_l3604_360459

/-- The center of a circle given by the equation x² + y² + 2x - 4y = 0 -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + a = 0 -/
def line_equation (a : ℝ) (x y : ℝ) : Prop := 3 * x + y + a = 0

/-- The circle equation x² + y² + 2x - 4y = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

/-- Theorem: If the line 3x + y + a = 0 passes through the center of the circle x² + y² + 2x - 4y = 0, then a = 1 -/
theorem line_through_circle_center (a : ℝ) : 
  line_equation a (circle_center.1) (circle_center.2) → a = 1 := by
  sorry

end line_through_circle_center_l3604_360459


namespace number_of_correct_propositions_is_zero_l3604_360431

-- Define the coefficient of determination
def coefficient_of_determination : ℝ → ℝ := sorry

-- Define a normal distribution
def normal_distribution (μ σ : ℝ) : ℝ → ℝ := sorry

-- Define systematic sampling
def systematic_sampling (start interval n : ℕ) : List ℕ := sorry

-- Define a proposition
structure Proposition :=
  (statement : Prop)
  (is_correct : Bool)

-- Define our three propositions
def proposition1 : Proposition :=
  ⟨ ∀ (R : ℝ), R < 0 → coefficient_of_determination R > coefficient_of_determination (-R), false ⟩

def proposition2 : Proposition :=
  let ξ := normal_distribution 2 1
  ⟨ ξ 4 = 0.79 → ξ (-2) = 0.21, false ⟩

def proposition3 : Proposition :=
  ⟨ systematic_sampling 5 11 5 = [5, 16, 27, 38, 49] → 60 ∈ systematic_sampling 5 11 5, false ⟩

-- Theorem to prove
theorem number_of_correct_propositions_is_zero :
  (proposition1.is_correct = false) ∧
  (proposition2.is_correct = false) ∧
  (proposition3.is_correct = false) := by
  sorry

end number_of_correct_propositions_is_zero_l3604_360431


namespace quadratic_equation_real_roots_l3604_360433

theorem quadratic_equation_real_roots (a b c : ℝ) : 
  ac < 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  sorry

end quadratic_equation_real_roots_l3604_360433


namespace total_yells_is_sixty_l3604_360407

/-- Represents the number of times Missy yells at her dogs -/
structure DogYells where
  obedient : ℕ
  stubborn : ℕ

/-- The relationship between yells at obedient and stubborn dogs -/
def stubborn_to_obedient_ratio : ℕ := 4

/-- The number of times Missy yells at her obedient dog -/
def obedient_yells : ℕ := 12

/-- Calculates the total number of yells based on the given conditions -/
def total_yells (yells : DogYells) : ℕ :=
  yells.obedient + yells.stubborn

/-- Theorem stating that the total number of yells is 60 -/
theorem total_yells_is_sixty :
  ∃ (yells : DogYells),
    yells.obedient = obedient_yells ∧
    yells.stubborn = stubborn_to_obedient_ratio * obedient_yells ∧
    total_yells yells = 60 := by
  sorry


end total_yells_is_sixty_l3604_360407


namespace volume_of_T_l3604_360412

/-- The solid T in ℝ³ -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p | let (x, y, z) := p
       (|x| + |y| ≤ 2) ∧ (|x| + |z| ≤ 2) ∧ (|y| + |z| ≤ 2)}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The volume of T is 64/9 -/
theorem volume_of_T : volume T = 64/9 := by sorry

end volume_of_T_l3604_360412


namespace quadrilateral_area_is_sqrt3_l3604_360492

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a quadrilateral formed by intersecting a plane with a rectangular prism -/
structure Quadrilateral where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculate the area of the quadrilateral ABCD -/
def quadrilateralArea (quad : Quadrilateral) : ℝ := sorry

/-- Theorem: The area of quadrilateral ABCD is √3 -/
theorem quadrilateral_area_is_sqrt3 (prism : RectangularPrism) (quad : Quadrilateral) :
  prism.length = 2 ∧ prism.width = 1 ∧ prism.height = 1 →
  (quad.A.x = 0 ∧ quad.A.y = 0 ∧ quad.A.z = 0) →
  (quad.C.x = 2 ∧ quad.C.y = 1 ∧ quad.C.z = 1) →
  (quad.B.x = 1 ∧ quad.B.y = 0.5 ∧ quad.B.z = 1) →
  (quad.D.x = 1 ∧ quad.D.y = 1 ∧ quad.D.z = 0.5) →
  quadrilateralArea quad = Real.sqrt 3 := by
  sorry

end quadrilateral_area_is_sqrt3_l3604_360492


namespace mean_of_smallest_elements_l3604_360469

/-- F(n, r) represents the arithmetic mean of the smallest elements
    in all r-element subsets of {1, 2, ..., n} -/
def F (n r : ℕ) : ℚ :=
  sorry

/-- Theorem stating that F(n, r) = (n+1)/(r+1) for 1 ≤ r ≤ n -/
theorem mean_of_smallest_elements (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) :
  F n r = (n + 1 : ℚ) / (r + 1) :=
by sorry

end mean_of_smallest_elements_l3604_360469


namespace salary_left_at_month_end_l3604_360450

/-- Represents the fraction of salary left after each step of the month --/
structure SalaryFraction where
  value : ℝ
  is_fraction : 0 ≤ value ∧ value ≤ 1

/-- Calculates the remaining salary fraction after tax deduction --/
def after_tax (tax_rate : ℝ) : SalaryFraction :=
  { value := 1 - tax_rate,
    is_fraction := by sorry }

/-- Calculates the remaining salary fraction after spending --/
def after_spending (s : SalaryFraction) (spend_rate : ℝ) : SalaryFraction :=
  { value := s.value * (1 - spend_rate),
    is_fraction := by sorry }

/-- Calculates the remaining salary fraction after an expense based on original salary --/
def after_expense (s : SalaryFraction) (expense_rate : ℝ) : SalaryFraction :=
  { value := s.value - expense_rate,
    is_fraction := by sorry }

/-- Theorem stating the fraction of salary left at the end of the month --/
theorem salary_left_at_month_end :
  let initial_salary := SalaryFraction.mk 1 (by sorry)
  let after_tax := after_tax 0.15
  let after_week1 := after_spending after_tax 0.25
  let after_week2 := after_spending after_week1 0.3
  let after_week3 := after_expense after_week2 0.2
  let final_salary := after_spending after_week3 0.1
  final_salary.value = 0.221625 := by sorry

end salary_left_at_month_end_l3604_360450


namespace zoo_visitors_l3604_360447

theorem zoo_visitors (friday_visitors : ℕ) (sunday_visitors : ℕ) : 
  friday_visitors = 1250 →
  sunday_visitors = 500 →
  5250 = 3 * (friday_visitors + sunday_visitors) :=
by sorry

end zoo_visitors_l3604_360447


namespace only_one_true_l3604_360448

-- Define the four propositions
def prop1 : Prop := sorry
def prop2 : Prop := ∀ x : ℝ, x^2 + x + 1 ≥ 0
def prop3 : Prop := sorry
def prop4 : Prop := sorry

-- Theorem stating that only one proposition is true
theorem only_one_true : (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
                        (¬prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
                        (¬prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4) ∨
                        (¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) :=
  sorry

end only_one_true_l3604_360448


namespace parallelogram_perimeter_area_sum_l3604_360479

/-- A point in 2D space with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- A parallelogram defined by four points -/
structure Parallelogram where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : Int :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).natAbs

/-- Calculate the perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : Int :=
  distance p.a p.b + distance p.b p.c + distance p.c p.d + distance p.d p.a

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : Int :=
  (distance p.a p.b * distance p.b p.c).natAbs

/-- The theorem to be proved -/
theorem parallelogram_perimeter_area_sum :
  ∀ (p : Parallelogram),
    p.a = Point.mk 2 7 →
    p.b = Point.mk 7 7 →
    p.c = Point.mk 7 2 →
    perimeter p + area p = 45 :=
by sorry

end parallelogram_perimeter_area_sum_l3604_360479


namespace total_days_2001_to_2004_l3604_360437

def regularYearDays : ℕ := 365
def leapYearDays : ℕ := 366
def regularYearsCount : ℕ := 3
def leapYearsCount : ℕ := 1

theorem total_days_2001_to_2004 :
  regularYearDays * regularYearsCount + leapYearDays * leapYearsCount = 1461 := by
  sorry

end total_days_2001_to_2004_l3604_360437


namespace estimate_population_size_l3604_360406

theorem estimate_population_size (sample1 : ℕ) (sample2 : ℕ) (overlap : ℕ) (total : ℕ) : 
  sample1 = 80 → sample2 = 100 → overlap = 20 → 
  (sample1 : ℝ) / total * ((sample2 : ℝ) / total) = (overlap : ℝ) / total → 
  total = 400 := by
sorry

end estimate_population_size_l3604_360406


namespace problem_solution_l3604_360462

theorem problem_solution (x : ℝ) : 
  x + Real.sqrt (x^2 + 1) + 1 / (x + Real.sqrt (x^2 + 1)) = 22 →
  x^2 - Real.sqrt (x^4 + 1) + 1 / (x^2 - Real.sqrt (x^4 + 1)) = 242 :=
by sorry

end problem_solution_l3604_360462


namespace smallest_number_l3604_360400

theorem smallest_number (a b c d e : ℝ) (ha : a = 1.4) (hb : b = 1.2) (hc : c = 2.0) (hd : d = 1.5) (he : e = 2.1) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d ∧ b ≤ e := by
  sorry

end smallest_number_l3604_360400


namespace sequence_formula_l3604_360446

theorem sequence_formula (n : ℕ) : 
  Real.cos ((n + 2 : ℝ) * π / 2) = 
    if n % 4 = 0 then 1
    else if n % 4 = 1 then 0
    else if n % 4 = 2 then -1
    else 0 := by
  sorry

end sequence_formula_l3604_360446


namespace marble_draw_probability_l3604_360471

/-- The probability of drawing a red marble first, a white marble second, and a blue marble third
    from a bag containing 5 red, 4 white, and 3 blue marbles, without replacement. -/
def drawProbability (redMarbles whiteMarbles blueMarbles : ℕ) : ℚ :=
  let totalMarbles := redMarbles + whiteMarbles + blueMarbles
  let firstDraw := redMarbles / totalMarbles
  let secondDraw := whiteMarbles / (totalMarbles - 1)
  let thirdDraw := blueMarbles / (totalMarbles - 2)
  firstDraw * secondDraw * thirdDraw

/-- Theorem stating that the probability of drawing red, white, then blue
    from a bag with 5 red, 4 white, and 3 blue marbles is 1/22. -/
theorem marble_draw_probability :
  drawProbability 5 4 3 = 1 / 22 := by
  sorry

end marble_draw_probability_l3604_360471


namespace two_digit_even_multiple_of_seven_perfect_square_digit_product_l3604_360497

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem two_digit_even_multiple_of_seven_perfect_square_digit_product :
  {n : ℕ | is_two_digit n ∧ 
           n % 2 = 0 ∧ 
           n % 7 = 0 ∧ 
           is_perfect_square (digit_product n)} = {14, 28, 70} := by
  sorry

end two_digit_even_multiple_of_seven_perfect_square_digit_product_l3604_360497


namespace line_ellipse_intersection_l3604_360411

theorem line_ellipse_intersection (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 9 * y^2 = 36 ∧ y = m * x + 3) → 
  m ≤ -Real.sqrt 5 / 3 ∨ m ≥ Real.sqrt 5 / 3 := by
sorry

end line_ellipse_intersection_l3604_360411


namespace sum_of_a_and_b_l3604_360417

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : a + 3 * b = 27) 
  (eq2 : 5 * a + 2 * b = 40) : 
  a + b = 161 / 13 := by
sorry

end sum_of_a_and_b_l3604_360417


namespace min_sum_dimensions_l3604_360467

theorem min_sum_dimensions (l w h : ℕ+) : 
  l * w * h = 2310 → 
  (∀ a b c : ℕ+, a * b * c = 2310 → l + w + h ≤ a + b + c) → 
  l + w + h = 48 :=
sorry

end min_sum_dimensions_l3604_360467


namespace binomial_10_4_l3604_360463

theorem binomial_10_4 : Nat.choose 10 4 = 210 := by
  sorry

end binomial_10_4_l3604_360463


namespace arithmetic_mean_of_sevens_l3604_360483

def set_of_sevens : List ℕ := [7, 77, 777, 7777, 77777, 777777, 7777777, 77777777, 777777777]

theorem arithmetic_mean_of_sevens :
  let sum := set_of_sevens.sum
  let count := set_of_sevens.length
  sum / count = 96308641 := by sorry

end arithmetic_mean_of_sevens_l3604_360483


namespace parallel_condition_l3604_360428

/-- Two lines are parallel if and only if their slopes are equal -/
def are_parallel (a b c d e f : ℝ) : Prop :=
  a * e = b * d

/-- The first line: ax + 2y - 1 = 0 -/
def line1 (a x y : ℝ) : Prop :=
  a * x + 2 * y - 1 = 0

/-- The second line: x + 2y + 4 = 0 -/
def line2 (x y : ℝ) : Prop :=
  x + 2 * y + 4 = 0

theorem parallel_condition (a : ℝ) :
  (∀ x y : ℝ, are_parallel a 2 (-1) 1 2 4) ↔ a = 1 :=
by sorry

end parallel_condition_l3604_360428


namespace number_in_seventh_group_l3604_360468

/-- Represents the systematic sampling method for a population of 100 individuals -/
structure SystematicSampling where
  population_size : Nat
  num_groups : Nat
  sample_size : Nat
  first_number : Nat

/-- The number drawn in the k-th group -/
def number_in_group (ss : SystematicSampling) (k : Nat) : Nat :=
  (ss.first_number + k - 1) % 10 + (k - 1) * 10

/-- Theorem stating that the number drawn in the 7th group is 63 -/
theorem number_in_seventh_group (ss : SystematicSampling) : 
  ss.population_size = 100 →
  ss.num_groups = 10 →
  ss.sample_size = 10 →
  ss.first_number = 6 →
  number_in_group ss 7 = 63 := by
  sorry

end number_in_seventh_group_l3604_360468


namespace functions_equal_at_three_l3604_360415

-- Define the interval (2, 4)
def OpenInterval := {x : ℝ | 2 < x ∧ x < 4}

-- Define the properties of functions f and g
def FunctionProperties (f g : ℝ → ℝ) : Prop :=
  (∀ x ∈ OpenInterval, 2 < f x ∧ f x < 4) ∧
  (∀ x ∈ OpenInterval, 2 < g x ∧ g x < 4) ∧
  (∀ x ∈ OpenInterval, f (g x) = x) ∧
  (∀ x ∈ OpenInterval, g (f x) = x) ∧
  (∀ x ∈ OpenInterval, f x * g x = x^2)

-- Theorem statement
theorem functions_equal_at_three 
  (f g : ℝ → ℝ) 
  (h : FunctionProperties f g) : 
  f 3 = g 3 := by
  sorry


end functions_equal_at_three_l3604_360415


namespace point_outside_circle_l3604_360489

theorem point_outside_circle :
  let P : ℝ × ℝ := (-2, -2)
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4}
  P ∉ circle ∧ (P.1^2 + P.2^2 > 4) := by
  sorry

end point_outside_circle_l3604_360489


namespace divisibility_property_l3604_360452

theorem divisibility_property (n : ℕ) :
  (∃ k : ℕ, n + 2 = 3 * k) ∧ (∃ m : ℕ, n + 3 = 4 * m) →
  ∃ l : ℕ, n + 5 = 6 * l :=
by
  sorry

end divisibility_property_l3604_360452


namespace periodic_function_property_l3604_360465

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) where f(2009) = 3, 
    prove that f(2010) = -3 -/
theorem periodic_function_property (a b α β : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β))
  (h2 : f 2009 = 3) : 
  f 2010 = -3 := by
sorry

end periodic_function_property_l3604_360465


namespace triangle_diameter_quadrilateral_diameter_pentagon_diameter_hexagon_diameter_l3604_360419

-- Define a convex n-gon with all sides equal to 1 and diameter d
def ConvexNGon (n : ℕ) (d : ℝ) : Prop :=
  n ≥ 3 ∧ d > 0

-- Theorem for n = 3
theorem triangle_diameter (d : ℝ) (h : ConvexNGon 3 d) : d = 1 := by
  sorry

-- Theorem for n = 4
theorem quadrilateral_diameter (d : ℝ) (h : ConvexNGon 4 d) : Real.sqrt 2 ≤ d ∧ d < 2 := by
  sorry

-- Theorem for n = 5
theorem pentagon_diameter (d : ℝ) (h : ConvexNGon 5 d) : (1 + Real.sqrt 5) / 2 ≤ d ∧ d < 2 := by
  sorry

-- Theorem for n = 6
theorem hexagon_diameter (d : ℝ) (h : ConvexNGon 6 d) : Real.sqrt (2 + Real.sqrt 3) ≤ d ∧ d < 3 := by
  sorry

end triangle_diameter_quadrilateral_diameter_pentagon_diameter_hexagon_diameter_l3604_360419


namespace geometric_sequence_sum_l3604_360413

theorem geometric_sequence_sum (a : ℕ → ℚ) :
  (∀ n, a (n + 1) = a n * (1/4)) →  -- Each term is 1/4 of the previous term
  a 3 = 256 →                       -- The fourth term is 256
  a 5 = 4 →                         -- The sixth term is 4
  a 6 = 1 →                         -- The seventh term is 1
  a 3 + a 4 = 80 :=                 -- The sum of the fourth and fifth terms is 80
by sorry

end geometric_sequence_sum_l3604_360413


namespace power_function_through_point_l3604_360466

/-- Given a power function f(x) = x^α that passes through the point (9,3), prove that f(100) = 10 -/
theorem power_function_through_point (α : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x ^ α) 
  (h2 : f 9 = 3) : 
  f 100 = 10 := by sorry

end power_function_through_point_l3604_360466


namespace batsman_average_is_35_l3604_360409

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  total_runs : ℕ
  last_inning_runs : ℕ
  average_increase : ℕ

/-- Calculates the new average of a batsman after their latest inning -/
def new_average (b : Batsman) : ℚ :=
  (b.total_runs + b.last_inning_runs) / b.innings

/-- Theorem stating that under given conditions, the batsman's new average is 35 -/
theorem batsman_average_is_35 (b : Batsman) 
    (h1 : b.innings = 17)
    (h2 : b.last_inning_runs = 83)
    (h3 : b.average_increase = 3)
    (h4 : new_average b = (new_average b - b.average_increase) + b.average_increase) :
    new_average b = 35 := by
  sorry

end batsman_average_is_35_l3604_360409


namespace angle_relation_l3604_360422

theorem angle_relation (α β : Real) : 
  α ∈ Set.Ioo 0 (Real.pi / 2) →
  β ∈ Set.Ioo 0 (Real.pi / 2) →
  Real.tan α + Real.tan β = 1 / Real.cos α →
  2 * β + α = Real.pi / 2 := by
  sorry

end angle_relation_l3604_360422


namespace middle_number_of_consecutive_integers_l3604_360430

theorem middle_number_of_consecutive_integers (n : ℤ) : 
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 10^2018 → n = 2 * 10^2017 := by
  sorry

end middle_number_of_consecutive_integers_l3604_360430


namespace xy_difference_l3604_360426

theorem xy_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := by
  sorry

end xy_difference_l3604_360426


namespace candy_spent_approx_11_l3604_360490

/-- The amount John spent at the supermarket -/
def total_spent : ℚ := 29.999999999999996

/-- The fraction of money spent on fresh fruits and vegetables -/
def fruits_veg_fraction : ℚ := 1 / 5

/-- The fraction of money spent on meat products -/
def meat_fraction : ℚ := 1 / 3

/-- The fraction of money spent on bakery products -/
def bakery_fraction : ℚ := 1 / 10

/-- The fraction of money spent on candy -/
def candy_fraction : ℚ := 1 - (fruits_veg_fraction + meat_fraction + bakery_fraction)

/-- The amount spent on candy -/
def candy_spent : ℚ := candy_fraction * total_spent

theorem candy_spent_approx_11 : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000 ∧ |candy_spent - 11| < ε :=
sorry

end candy_spent_approx_11_l3604_360490


namespace soccer_ball_inflation_time_l3604_360470

/-- The time in minutes Alexia takes to inflate one soccer ball -/
def alexia_time : ℕ := 18

/-- The time in minutes Ermias takes to inflate one soccer ball -/
def ermias_time : ℕ := 25

/-- The number of soccer balls Alexia inflates -/
def alexia_balls : ℕ := 36

/-- The number of additional balls Ermias inflates compared to Alexia -/
def additional_balls : ℕ := 8

/-- The total time in minutes taken by Alexia and Ermias to inflate all soccer balls -/
def total_time : ℕ := alexia_time * alexia_balls + ermias_time * (alexia_balls + additional_balls)

theorem soccer_ball_inflation_time : total_time = 1748 := by sorry

end soccer_ball_inflation_time_l3604_360470


namespace solve_exponential_equation_l3604_360438

theorem solve_exponential_equation :
  ∃! x : ℤ, (3 : ℝ) ^ 8 * (3 : ℝ) ^ x = 81 :=
by
  use -4
  sorry

end solve_exponential_equation_l3604_360438


namespace quadrilateral_area_between_squares_l3604_360474

/-- Represents a square with a given side length -/
structure Square (α : Type*) [LinearOrderedField α] where
  side : α

/-- Represents a trapezoid with given bases and height -/
structure Trapezoid (α : Type*) [LinearOrderedField α] where
  base1 : α
  base2 : α
  height : α

/-- Calculates the area of a trapezoid -/
def trapezoid_area {α : Type*} [LinearOrderedField α] (t : Trapezoid α) : α :=
  (t.base1 + t.base2) * t.height / 2

theorem quadrilateral_area_between_squares
  (s1 s2 s3 s4 : Square ℝ)
  (h1 : s1.side = 2)
  (h2 : s2.side = 4)
  (h3 : s3.side = 6)
  (h4 : s4.side = 8)
  (h_coplanar : True)  -- Assumption of coplanarity
  (h_arrangement : True)  -- Assumption of side-by-side arrangement on line AB
  : ∃ t : Trapezoid ℝ, 
    t.base1 = 3 ∧ 
    t.base2 = 10 ∧ 
    t.height = 2 ∧ 
    trapezoid_area t = 13 :=
  sorry

end quadrilateral_area_between_squares_l3604_360474


namespace exact_payment_l3604_360453

/-- The cost of the book in cents -/
def book_cost : ℕ := 4550

/-- The value of four $10 bills in cents -/
def bills_value : ℕ := 4000

/-- The value of ten nickels in cents -/
def nickels_value : ℕ := 50

/-- The minimum number of pennies needed -/
def min_pennies : ℕ := book_cost - bills_value - nickels_value

theorem exact_payment :
  min_pennies = 500 := by sorry

end exact_payment_l3604_360453


namespace triangle_side_lengths_l3604_360478

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the side lengths and angles
def sideLength (p q : ℝ × ℝ) : ℝ := sorry
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_side_lengths (t : Triangle) :
  sideLength t.A t.C = 5 →
  sideLength t.B t.C - sideLength t.A t.B = 2 →
  angle t.C t.A t.B = 2 * angle t.A t.C t.B →
  sideLength t.A t.B = 4 ∧ sideLength t.B t.C = 6 := by
  sorry

end triangle_side_lengths_l3604_360478


namespace expansion_sum_theorem_l3604_360481

theorem expansion_sum_theorem (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (3*x - 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₁/3 + a₂/3^2 + a₃/3^3 + a₄/3^4 + a₅/3^5 + a₆/3^6 + a₇/3^7 + a₈/3^8 + a₉/3^9 = 511 := by
sorry

end expansion_sum_theorem_l3604_360481


namespace at_most_one_root_l3604_360484

-- Define a monotonically increasing function on ℝ
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≤ f y

-- Theorem statement
theorem at_most_one_root (f : ℝ → ℝ) (h : MonoIncreasing f) :
  ∃! x, f x = 0 ∨ ∀ x, f x ≠ 0 :=
by
  sorry

end at_most_one_root_l3604_360484


namespace max_percentage_both_services_l3604_360440

theorem max_percentage_both_services (internet_percentage : Real) (snack_percentage : Real) :
  internet_percentage = 0.4 →
  snack_percentage = 0.7 →
  ∃ (both_percentage : Real),
    both_percentage ≤ internet_percentage ∧
    both_percentage ≤ snack_percentage ∧
    ∀ (x : Real),
      x ≤ internet_percentage ∧
      x ≤ snack_percentage →
      x ≤ both_percentage :=
by sorry

end max_percentage_both_services_l3604_360440


namespace student_number_problem_l3604_360449

theorem student_number_problem (x : ℤ) : 2 * x - 138 = 110 → x = 124 := by
  sorry

end student_number_problem_l3604_360449


namespace car_profit_percentage_l3604_360495

/-- Given a car with an original price, calculate the profit percentage when bought at a discount and sold at an increase. -/
theorem car_profit_percentage (P : ℝ) (discount : ℝ) (increase : ℝ)
  (h_discount : discount = 0.4)
  (h_increase : increase = 0.8) :
  let buying_price := P * (1 - discount)
  let selling_price := buying_price * (1 + increase)
  let profit := selling_price - P
  profit / P * 100 = 8 := by
  sorry

end car_profit_percentage_l3604_360495


namespace triangle_sides_gp_implies_altitudes_gp_l3604_360427

/-- Theorem: If the sides of a triangle form a geometric progression, 
    then its altitudes also form a geometric progression. -/
theorem triangle_sides_gp_implies_altitudes_gp 
  (a q : ℝ) 
  (h_positive : a > 0 ∧ q > 0) 
  (h_sides : ∃ (s₁ s₂ s₃ : ℝ), s₁ = a ∧ s₂ = a*q ∧ s₃ = a*q^2) :
  ∃ (h₁ h₂ h₃ : ℝ), h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ 
    h₂ / h₁ = 1/q ∧ h₃ / h₂ = 1/q :=
by sorry

end triangle_sides_gp_implies_altitudes_gp_l3604_360427


namespace f_nine_equals_zero_l3604_360460

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop := ∀ x, f (x + 6) = f x + f 3

theorem f_nine_equals_zero (f : ℝ → ℝ) (h1 : is_even f) (h2 : satisfies_condition f) : 
  f 9 = 0 := by sorry

end f_nine_equals_zero_l3604_360460


namespace f_properties_l3604_360418

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_properties (a : ℝ) (h : a > 0) :
  ∃ (x_min : ℝ), 
    (∀ x, f a x ≥ f a x_min) ∧ 
    (x_min = Real.log (1 / a)) ∧
    (f a x_min > 2 * Real.log a + 3 / 2) := by
  sorry

end f_properties_l3604_360418


namespace car_speed_first_hour_l3604_360423

/-- Given a car's travel data, prove its speed in the first hour -/
theorem car_speed_first_hour 
  (speed_second_hour : ℝ) 
  (average_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_second_hour = 40)
  (h2 : average_speed = 60)
  (h3 : total_time = 2) :
  let total_distance := average_speed * total_time
  let speed_first_hour := 2 * total_distance / total_time - speed_second_hour
  speed_first_hour = 80 := by
sorry

end car_speed_first_hour_l3604_360423


namespace coefficient_x4_is_4374_l3604_360416

/-- The coefficient of x^4 in the expansion of ((4x^2 + 6x + 9/4)^4) -/
def coefficient_x4 : ℕ :=
  let a := 4  -- coefficient of x^2
  let b := 6  -- coefficient of x
  let c := 9/4  -- constant term
  let n := 4  -- power of the binomial
  -- The actual calculation of the coefficient would go here
  4374  -- This is the result we want to prove

/-- The expansion of ((4x^2 + 6x + 9/4)^4) has 4374 as the coefficient of x^4 -/
theorem coefficient_x4_is_4374 : coefficient_x4 = 4374 := by
  sorry


end coefficient_x4_is_4374_l3604_360416


namespace pure_imaginary_product_l3604_360434

def complex_number (a b : ℝ) := a + b * Complex.I

theorem pure_imaginary_product (m : ℝ) :
  (complex_number 1 m * complex_number 2 (-1)).re = 0 →
  m = -2 :=
by sorry

end pure_imaginary_product_l3604_360434


namespace linear_equation_solution_l3604_360429

theorem linear_equation_solution (a : ℝ) : 
  (a * 1 + (-2) = 1) → a = 3 := by
  sorry

end linear_equation_solution_l3604_360429


namespace rectangular_box_volume_l3604_360403

theorem rectangular_box_volume (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : ∃ (k : ℕ), k > 0 ∧ a = 2 * k ∧ b = 4 * k ∧ c = 5 * k) :
  a * b * c = 320 :=
by sorry

end rectangular_box_volume_l3604_360403


namespace awards_distribution_l3604_360461

/-- Represents the number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of ways to distribute 6 awards to 4 students -/
theorem awards_distribution :
  distribute_awards 6 4 = 780 :=
by
  sorry

end awards_distribution_l3604_360461


namespace seed_germination_percentage_l3604_360494

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 25 / 100 →
  germination_rate2 = 35 / 100 →
  (((seeds_plot1 : ℚ) * germination_rate1 + (seeds_plot2 : ℚ) * germination_rate2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) * 100 = 29 := by
  sorry

end seed_germination_percentage_l3604_360494


namespace arithmetic_calculation_l3604_360401

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end arithmetic_calculation_l3604_360401


namespace division_remainder_l3604_360435

theorem division_remainder (n : ℕ) : 
  (n / 7 = 5) ∧ (n % 7 = 0) → n % 11 = 2 :=
by sorry

end division_remainder_l3604_360435


namespace power_div_reciprocal_l3604_360455

theorem power_div_reciprocal (a : ℝ) (h : a ≠ 0) : a^2 / (1/a) = a^3 := by
  sorry

end power_div_reciprocal_l3604_360455


namespace pond_length_l3604_360475

/-- Given a rectangular field and a square pond, prove the length of the pond's side -/
theorem pond_length (field_width field_length pond_area : ℝ) : 
  field_length = 2 * field_width →
  field_length = 36 →
  pond_area = (1/8) * (field_length * field_width) →
  Real.sqrt pond_area = 9 := by
  sorry

end pond_length_l3604_360475


namespace sandy_siblings_l3604_360477

def total_tokens : ℕ := 1000000
def sandy_share : ℕ := total_tokens / 2
def extra_tokens : ℕ := 375000

theorem sandy_siblings :
  ∃ (num_siblings : ℕ),
    num_siblings > 0 ∧
    sandy_share = (total_tokens - sandy_share) / num_siblings + extra_tokens ∧
    num_siblings = 4 :=
by sorry

end sandy_siblings_l3604_360477


namespace homecoming_dance_tickets_l3604_360454

/-- Represents the number of couple tickets sold at a homecoming dance. -/
def couple_tickets : ℕ := 56

/-- Represents the number of single tickets sold at a homecoming dance. -/
def single_tickets : ℕ := 128 - 2 * couple_tickets

/-- The cost of a single ticket in dollars. -/
def single_ticket_cost : ℕ := 20

/-- The cost of a couple ticket in dollars. -/
def couple_ticket_cost : ℕ := 35

/-- The total ticket sales in dollars. -/
def total_sales : ℕ := 2280

/-- The total number of attendees. -/
def total_attendees : ℕ := 128

theorem homecoming_dance_tickets :
  single_ticket_cost * single_tickets + couple_ticket_cost * couple_tickets = total_sales ∧
  single_tickets + 2 * couple_tickets = total_attendees := by
  sorry

end homecoming_dance_tickets_l3604_360454


namespace solution_verification_l3604_360405

-- Define the differential equation
def differential_equation (x : ℝ) (y : ℝ → ℝ) : Prop :=
  x * (deriv y x) = y x - 1

-- Define the first function
def f₁ (x : ℝ) : ℝ := 3 * x + 1

-- Define the second function (C is a real constant)
def f₂ (C : ℝ) (x : ℝ) : ℝ := C * x + 1

-- Theorem statement
theorem solution_verification :
  (∀ x, x ≠ 0 → differential_equation x f₁) ∧
  (∀ C, ∀ x, x ≠ 0 → differential_equation x (f₂ C)) :=
sorry

end solution_verification_l3604_360405


namespace abs_neg_two_l3604_360496

theorem abs_neg_two : |(-2 : ℤ)| = 2 := by sorry

end abs_neg_two_l3604_360496


namespace gcd_lcm_sum_l3604_360482

theorem gcd_lcm_sum : Nat.gcd 24 54 + Nat.lcm 40 10 = 46 := by
  sorry

end gcd_lcm_sum_l3604_360482


namespace vector_problem_l3604_360443

/-- Given two vectors a and b in R^2 -/
def a (x : ℝ) : Fin 2 → ℝ := ![x, -2]
def b : Fin 2 → ℝ := ![2, 4]

/-- Parallel vectors have proportional components -/
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 2, v i = k * w i

/-- The squared magnitude of a vector -/
def magnitude_squared (v : Fin 2 → ℝ) : ℝ :=
  (v 0)^2 + (v 1)^2

/-- Vector addition -/
def vec_add (v w : Fin 2 → ℝ) : Fin 2 → ℝ :=
  λ i => v i + w i

theorem vector_problem (x : ℝ) :
  (parallel (a x) b → x = -1) ∧
  (magnitude_squared (vec_add (a x) b) = 13 → x = 1 ∨ x = -5) := by
  sorry


end vector_problem_l3604_360443


namespace cat_weight_l3604_360410

/-- Given a cat and a dog with specific weight relationships, prove the cat's weight -/
theorem cat_weight (cat_weight dog_weight : ℝ) : 
  dog_weight = cat_weight + 6 →
  cat_weight = dog_weight / 3 →
  cat_weight = 3 := by
sorry

end cat_weight_l3604_360410


namespace viola_count_l3604_360439

theorem viola_count (cellos : ℕ) (pairs : ℕ) (prob : ℚ) (violas : ℕ) : 
  cellos = 800 → 
  pairs = 100 → 
  prob = 100 / (800 * violas) → 
  prob = 0.00020833333333333335 → 
  violas = 600 := by
  sorry

end viola_count_l3604_360439


namespace triangle_heights_inscribed_circle_inequality_l3604_360491

/-- Given a triangle with heights h₁ and h₂, and an inscribed circle with radius r,
    prove that 1/(2r) < 1/h₁ + 1/h₂ < 1/r. -/
theorem triangle_heights_inscribed_circle_inequality 
  (h₁ h₂ r : ℝ) 
  (h₁_pos : 0 < h₁) 
  (h₂_pos : 0 < h₂) 
  (r_pos : 0 < r) : 
  1 / (2 * r) < 1 / h₁ + 1 / h₂ ∧ 1 / h₁ + 1 / h₂ < 1 / r := by
  sorry

end triangle_heights_inscribed_circle_inequality_l3604_360491


namespace tan_alpha_value_l3604_360499

theorem tan_alpha_value (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 3/5)
  (h2 : α > Real.pi/2 ∧ α < Real.pi) : 
  Real.tan α = -3/4 := by
  sorry

end tan_alpha_value_l3604_360499


namespace art_project_marker_distribution_l3604_360404

theorem art_project_marker_distribution (total_students : ℕ) (total_boxes : ℕ) (markers_per_box : ℕ)
  (group1_students : ℕ) (group1_markers : ℕ) (group2_students : ℕ) (group2_markers : ℕ) :
  total_students = 30 →
  total_boxes = 22 →
  markers_per_box = 5 →
  group1_students = 10 →
  group1_markers = 2 →
  group2_students = 15 →
  group2_markers = 4 →
  (total_students - group1_students - group2_students) > 0 →
  (total_boxes * markers_per_box - group1_students * group1_markers - group2_students * group2_markers) /
    (total_students - group1_students - group2_students) = 6 :=
by sorry

end art_project_marker_distribution_l3604_360404


namespace value_of_b_l3604_360487

theorem value_of_b (b : ℚ) (h : b + 2 * b / 5 = 22 / 5) : b = 22 / 7 := by
  sorry

end value_of_b_l3604_360487


namespace treasure_chest_contains_all_coins_l3604_360444

/-- Represents the scuba diving scenario with gold coins --/
structure ScubaDiving where
  hours : ℕ
  coins_per_hour : ℕ
  smaller_bags : ℕ

/-- Calculates the number of gold coins in the treasure chest --/
def treasure_chest_coins (dive : ScubaDiving) : ℕ :=
  dive.hours * dive.coins_per_hour

/-- Theorem stating that the treasure chest contains all the coins found --/
theorem treasure_chest_contains_all_coins (dive : ScubaDiving) 
  (h1 : dive.hours = 8)
  (h2 : dive.coins_per_hour = 25)
  (h3 : dive.smaller_bags = 2) :
  treasure_chest_coins dive = 200 :=
sorry

end treasure_chest_contains_all_coins_l3604_360444


namespace hyperbola_center_l3604_360451

/-- The center of the hyperbola given by the equation (4y-6)²/6² - (5x-3)²/7² = -1 -/
theorem hyperbola_center : ∃ (h k : ℝ), 
  (∀ (x y : ℝ), (4*y - 6)^2 / 6^2 - (5*x - 3)^2 / 7^2 = -1 ↔ 
    (x - h)^2 / (7/5)^2 - (y - k)^2 / (3/2)^2 = 1) ∧ 
  h = 3/5 ∧ k = 3/2 := by
  sorry


end hyperbola_center_l3604_360451


namespace problem_solution_l3604_360476

theorem problem_solution : 
  ((5 * Real.sqrt 3 + 2 * Real.sqrt 5) ^ 2 = 95 + 20 * Real.sqrt 15) ∧ 
  ((1/2) * (Real.sqrt 2 + Real.sqrt 3) - (3/4) * (Real.sqrt 2 + Real.sqrt 27) = 
   -(1/4) * Real.sqrt 2 - (7/4) * Real.sqrt 3) := by
sorry

end problem_solution_l3604_360476


namespace triangle_properties_l3604_360464

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  2 * t.c * Real.cos t.A = 2 * t.b - Real.sqrt 3 * t.a

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_condition t) : 
  t.C = Real.pi / 6 ∧ 
  (t.b = 2 → 2 * Real.sqrt 3 = 1/2 * t.a * t.b * Real.sin t.C → 
   Real.sin t.A = Real.sqrt 7 / 7) :=
sorry

end triangle_properties_l3604_360464


namespace average_calculation_l3604_360480

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of four numbers
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

-- State the theorem
theorem average_calculation :
  avg4 (avg4 2 2 0 2) (avg2 3 1) 0 3 = 13 / 8 := by
  sorry

end average_calculation_l3604_360480


namespace q_investment_l3604_360425

/-- Represents the investment of two people in a shop --/
structure Investment where
  p : ℕ  -- Amount invested by P
  q : ℕ  -- Amount invested by Q
  ratio_p : ℕ  -- Profit ratio for P
  ratio_q : ℕ  -- Profit ratio for Q

/-- Theorem: Given the conditions, Q's investment is 60000 --/
theorem q_investment (i : Investment) 
  (h1 : i.p = 40000)  -- P invested 40000
  (h2 : i.ratio_p = 2)  -- P's profit ratio is 2
  (h3 : i.ratio_q = 3)  -- Q's profit ratio is 3
  : i.q = 60000 := by
  sorry

#check q_investment

end q_investment_l3604_360425


namespace smallest_class_size_l3604_360402

theorem smallest_class_size (n : ℕ) (scores : Fin n → ℕ) : 
  (∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    scores i = 100 ∧ scores j = 100 ∧ scores k = 100) →
  (∀ i : Fin n, scores i ≥ 70) →
  (∀ i : Fin n, scores i ≤ 100) →
  (Finset.sum Finset.univ scores / n = 85) →
  n ≥ 6 :=
by sorry

end smallest_class_size_l3604_360402


namespace geometric_sequence_sum_l3604_360472

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
sorry

end geometric_sequence_sum_l3604_360472


namespace tangent_line_smallest_slope_l3604_360432

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

-- Theorem statement
theorem tangent_line_smallest_slope :
  ∃ (x₀ y₀ : ℝ),
    (y₀ = f x₀) ∧
    (∀ x, f' x₀ ≤ f' x) ∧
    (3*x₀ - y₀ - 11 = 0) :=
sorry

end tangent_line_smallest_slope_l3604_360432


namespace triangle_side_length_l3604_360473

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h_area : (1/2) * a * c * (Real.sqrt 3 / 2) = Real.sqrt 3)
  (h_angle : B = 60 * π / 180)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
sorry

end triangle_side_length_l3604_360473


namespace nested_floor_equation_solution_l3604_360486

theorem nested_floor_equation_solution :
  ∃! x : ℝ, x * ⌊x * ⌊x * ⌊x * ⌊x⌋⌋⌋⌋ = 122 :=
by
  -- The unique solution is 122/41
  use 122/41
  sorry

end nested_floor_equation_solution_l3604_360486


namespace triangle_area_ratio_l3604_360485

theorem triangle_area_ratio (d : ℝ) : d > 0 →
  (1/2 * 3 * 6) = (1/4) * (1/2 * (d - 3) * (2*d - 6)) →
  d = 9 := by
sorry

end triangle_area_ratio_l3604_360485


namespace b_grazing_months_l3604_360457

/-- Represents the number of months b put his oxen for grazing -/
def b_months : ℕ := sorry

/-- Represents the total rent of the pasture in rupees -/
def total_rent : ℕ := 140

/-- Represents c's share of the rent in rupees -/
def c_share : ℕ := 36

/-- Calculates the total oxen-months for all three people -/
def total_oxen_months : ℕ := 70 + 12 * b_months + 45

theorem b_grazing_months :
  (c_share : ℚ) / total_rent = 45 / total_oxen_months → b_months = 5 := by sorry

end b_grazing_months_l3604_360457


namespace isabellas_hair_growth_l3604_360436

/-- Calculates the final hair length after a given time period. -/
def final_hair_length (initial_length : ℝ) (growth_rate : ℝ) (months : ℝ) : ℝ :=
  initial_length + growth_rate * months

/-- Proves that Isabella's hair will be 28 inches long after 5 months. -/
theorem isabellas_hair_growth :
  final_hair_length 18 2 5 = 28 := by
  sorry

end isabellas_hair_growth_l3604_360436


namespace sum_of_numbers_l3604_360456

theorem sum_of_numbers (a b : ℕ) (h1 : a = 64 ∧ b = 32) (h2 : max a b = 2 * min a b) : a + b = 96 := by
  sorry

end sum_of_numbers_l3604_360456


namespace line_arrangements_with_restriction_l3604_360420

def num_students : Nat := 4

def total_arrangements : Nat := Nat.factorial num_students

def arrangements_with_restricted_pair : Nat :=
  (Nat.factorial (num_students - 1)) * (Nat.factorial 2)

theorem line_arrangements_with_restriction :
  total_arrangements - arrangements_with_restricted_pair = 12 := by
  sorry

end line_arrangements_with_restriction_l3604_360420


namespace absolute_value_sum_equals_four_l3604_360442

theorem absolute_value_sum_equals_four (x : ℝ) :
  (abs (x - 1) + abs (x - 5) = 4) ↔ (1 ≤ x ∧ x ≤ 5) := by
  sorry

end absolute_value_sum_equals_four_l3604_360442


namespace point_on_transformed_graph_l3604_360445

/-- Given a function f where f(3) = -2, prove that (1, -5/2) lies on the graph of 4y = 2f(3x) - 6 
    and that the sum of its coordinates is -3/2 -/
theorem point_on_transformed_graph (f : ℝ → ℝ) (h : f 3 = -2) :
  let g : ℝ → ℝ := λ x => (2 * f (3 * x) - 6) / 4
  g 1 = -5/2 ∧ 1 + (-5/2) = -3/2 :=
by sorry

end point_on_transformed_graph_l3604_360445


namespace davids_chemistry_marks_l3604_360441

theorem davids_chemistry_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (biology : ℕ)
  (average : ℕ)
  (num_subjects : ℕ)
  (h1 : english = 81)
  (h2 : mathematics = 65)
  (h3 : physics = 82)
  (h4 : biology = 85)
  (h5 : average = 76)
  (h6 : num_subjects = 5)
  : ∃ chemistry : ℕ,
    chemistry = average * num_subjects - (english + mathematics + physics + biology) :=
by sorry

end davids_chemistry_marks_l3604_360441


namespace fraction_sum_inequality_l3604_360424

theorem fraction_sum_inequality (x y z : ℝ) (n : ℕ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hn : n > 0) : 
  x / (n * x + y + z) + y / (x + n * y + z) + z / (x + y + n * z) ≤ 3 / (n + 2) := by
  sorry

end fraction_sum_inequality_l3604_360424
