import Mathlib

namespace new_average_age_with_teacher_l1846_184643

theorem new_average_age_with_teacher 
  (num_students : ℕ) 
  (student_avg_age : ℝ) 
  (teacher_age : ℝ) : 
  num_students = 10 → 
  student_avg_age = 15 → 
  teacher_age = 26 → 
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = 16 :=
by
  sorry

end new_average_age_with_teacher_l1846_184643


namespace simplify_sqrt_sum_l1846_184649

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end simplify_sqrt_sum_l1846_184649


namespace class_test_probabilities_l1846_184676

theorem class_test_probabilities (p_first p_second p_both : ℝ) 
  (h1 : p_first = 0.63)
  (h2 : p_second = 0.49)
  (h3 : p_both = 0.32) :
  1 - (p_first + p_second - p_both) = 0.20 := by
  sorry

end class_test_probabilities_l1846_184676


namespace x_cubed_plus_two_x_squared_plus_2007_l1846_184624

theorem x_cubed_plus_two_x_squared_plus_2007 (x : ℝ) (h : x^2 + x - 1 = 0) :
  x^3 + 2*x^2 + 2007 = 2008 := by
  sorry

end x_cubed_plus_two_x_squared_plus_2007_l1846_184624


namespace trinomial_perfect_fourth_power_l1846_184618

/-- A trinomial is a perfect fourth power for all integers if and only if its quadratic and linear coefficients are zero. -/
theorem trinomial_perfect_fourth_power (a b c : ℤ) :
  (∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^4) → a = 0 ∧ b = 0 := by
  sorry

end trinomial_perfect_fourth_power_l1846_184618


namespace max_distance_and_squared_distance_coincide_l1846_184640

-- Define a triangle ABC in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the sum of distances from a point to the vertices
def sumDistances (t : Triangle) (p : ℝ × ℝ) : ℝ :=
  distance p t.A + distance p t.B + distance p t.C

-- Define a function to calculate the sum of squared distances from a point to the vertices
def sumSquaredDistances (t : Triangle) (p : ℝ × ℝ) : ℝ :=
  (distance p t.A)^2 + (distance p t.B)^2 + (distance p t.C)^2

-- Define a function to find the shortest side of a triangle
def shortestSide (t : Triangle) : ℝ := sorry

-- Define a function to find the vertex opposite the shortest side
def vertexOppositeShortestSide (t : Triangle) : ℝ × ℝ := sorry

theorem max_distance_and_squared_distance_coincide (t : Triangle) :
  ∃ p : ℝ × ℝ,
    (∀ q : ℝ × ℝ, sumDistances t q ≤ sumDistances t p) ∧
    (∀ q : ℝ × ℝ, sumSquaredDistances t q ≤ sumSquaredDistances t p) ∧
    p = vertexOppositeShortestSide t :=
  sorry


end max_distance_and_squared_distance_coincide_l1846_184640


namespace system_solution_l1846_184626

theorem system_solution (x y : ℤ) (h1 : 7 - x = 15) (h2 : y - 3 = 4 + x) :
  x = -8 ∧ y = -1 := by
  sorry

end system_solution_l1846_184626


namespace triangle_angle_C_l1846_184689

theorem triangle_angle_C (A B C : ℝ) (h_triangle : A + B + C = PI) 
  (h_eq1 : 5 * Real.sin A + 3 * Real.cos B = 8)
  (h_eq2 : 3 * Real.sin B + 5 * Real.cos A = 0) :
  C = PI / 2 := by
sorry

end triangle_angle_C_l1846_184689


namespace union_complement_and_quadratic_set_l1846_184637

/-- Given sets S and T, prove that their union is equal to the set of all real numbers less than or equal to 1 -/
theorem union_complement_and_quadratic_set :
  let S : Set ℝ := {x | x > -2}
  let T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}
  (Sᶜ ∪ T) = {x : ℝ | x ≤ 1} := by sorry

end union_complement_and_quadratic_set_l1846_184637


namespace alexandrov_theorem_l1846_184673

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a face of a polyhedron -/
structure Face where
  -- Add necessary fields here
  mk ::

/-- Represents a planar angle in a polyhedron -/
def PlanarAngle : Type := ℝ

/-- Represents a dihedral angle in a polyhedron -/
def DihedralAngle : Type := ℝ

/-- Check if two polyhedra have correspondingly equal faces -/
def has_equal_faces (P Q : ConvexPolyhedron) : Prop :=
  sorry

/-- Check if two polyhedra have equal corresponding planar angles -/
def has_equal_planar_angles (P Q : ConvexPolyhedron) : Prop :=
  sorry

/-- Check if two polyhedra have equal corresponding dihedral angles -/
def has_equal_dihedral_angles (P Q : ConvexPolyhedron) : Prop :=
  sorry

/-- Alexandrov's Theorem -/
theorem alexandrov_theorem (P Q : ConvexPolyhedron) :
  has_equal_faces P Q → has_equal_planar_angles P Q → has_equal_dihedral_angles P Q :=
by
  sorry

end alexandrov_theorem_l1846_184673


namespace inequality_solution_set_l1846_184612

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8*x + 20) / (m*x^2 + 2*(m+1)*x + 9*m + 4) < 0) ↔ m < -1/2 :=
by sorry

end inequality_solution_set_l1846_184612


namespace intersection_points_quadratic_linear_l1846_184682

theorem intersection_points_quadratic_linear 
  (x y : ℝ) : 
  (y = 3 * x^2 - 6 * x + 5 ∧ y = 2 * x + 1) ↔ 
  ((x = 2 ∧ y = 5) ∨ (x = 2/3 ∧ y = 7/3)) :=
by sorry

end intersection_points_quadratic_linear_l1846_184682


namespace gcd_preservation_l1846_184633

theorem gcd_preservation (a b c d x y z G : ℤ) 
  (h : G = Int.gcd a (Int.gcd b (Int.gcd c d))) : 
  G = Int.gcd a (Int.gcd b (Int.gcd c (Int.gcd d (Int.gcd (G*x) (Int.gcd (G*y) (G*z)))))) :=
by sorry

end gcd_preservation_l1846_184633


namespace remainder_5_pow_2021_mod_17_l1846_184699

theorem remainder_5_pow_2021_mod_17 : 5^2021 % 17 = 14 := by
  sorry

end remainder_5_pow_2021_mod_17_l1846_184699


namespace factor_sum_l1846_184655

theorem factor_sum (R S : ℝ) : 
  (∃ d e : ℝ, (X ^ 2 + 3 * X + 7) * (X ^ 2 + d * X + e) = X ^ 4 + R * X ^ 2 + S) →
  R + S = 54 := by
sorry

end factor_sum_l1846_184655


namespace margarita_run_distance_l1846_184604

/-- Proves that Margarita ran 18 feet given the conditions of the long jump event -/
theorem margarita_run_distance (ricciana_run : ℝ) (ricciana_jump : ℝ) (margarita_total : ℝ) :
  ricciana_run = 20 →
  ricciana_jump = 4 →
  margarita_total = ricciana_run + ricciana_jump + 1 →
  ∃ (margarita_run : ℝ) (margarita_jump : ℝ),
    margarita_jump = 2 * ricciana_jump - 1 ∧
    margarita_total = margarita_run + margarita_jump ∧
    margarita_run = 18 :=
by
  sorry

end margarita_run_distance_l1846_184604


namespace polynomial_value_at_zero_l1846_184650

def is_valid_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ), 
    ∀ x, p x = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6

theorem polynomial_value_at_zero 
  (p : ℝ → ℝ) 
  (h_valid : is_valid_polynomial p) 
  (h_values : ∀ n : ℕ, n ≤ 6 → p (3^n) = (1 : ℝ) / (3^n)) :
  p 0 = 29523 / 2187 := by
  sorry

end polynomial_value_at_zero_l1846_184650


namespace inequalities_proof_l1846_184672

theorem inequalities_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (a^3 * b > a * b^3) ∧ (a - b/a > b - a/b) := by
  sorry

end inequalities_proof_l1846_184672


namespace circle_distance_relation_l1846_184683

/-- Given a circle with center O and radius 2a, prove the relationship between x and y -/
theorem circle_distance_relation (a : ℝ) (x y : ℝ) : 
  x > 0 → a > 0 → y^2 = x^3 / (2*a + x) := by
  sorry


end circle_distance_relation_l1846_184683


namespace P_smallest_l1846_184661

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3
def is_perfect_fifth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m^5

def H : ℕ := sorry

axiom H_def : H > 0 ∧ 
  is_perfect_cube (H / 2) ∧ 
  is_perfect_fifth_power (H / 3) ∧ 
  is_perfect_square (H / 5)

axiom H_minimal : ∀ n : ℕ, n > 0 → 
  is_perfect_cube (n / 2) → 
  is_perfect_fifth_power (n / 3) → 
  is_perfect_square (n / 5) → 
  H ≤ n

def P : ℕ := sorry

axiom P_def : P > 0 ∧ 
  is_perfect_square (P / 2) ∧ 
  is_perfect_cube (P / 3) ∧ 
  is_perfect_fifth_power (P / 5)

axiom P_minimal : ∀ n : ℕ, n > 0 → 
  is_perfect_square (n / 2) → 
  is_perfect_cube (n / 3) → 
  is_perfect_fifth_power (n / 5) → 
  P ≤ n

def S : ℕ := sorry

axiom S_def : S > 0 ∧ 
  is_perfect_fifth_power (S / 2) ∧ 
  is_perfect_square (S / 3) ∧ 
  is_perfect_cube (S / 5)

axiom S_minimal : ∀ n : ℕ, n > 0 → 
  is_perfect_fifth_power (n / 2) → 
  is_perfect_square (n / 3) → 
  is_perfect_cube (n / 5) → 
  S ≤ n

theorem P_smallest : P < S ∧ P < H := by sorry

end P_smallest_l1846_184661


namespace triangle_problem_l1846_184638

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the sine law
axiom sine_law (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

-- Define the cosine law
axiom cosine_law (t : Triangle) : Real.cos t.C = (t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)

theorem triangle_problem (t : Triangle) 
  (ha : t.a = 4)
  (hc : t.c = Real.sqrt 13)
  (hsin : Real.sin t.A = 4 * Real.sin t.B) :
  t.b = 1 ∧ Real.cos t.C = 1/2 := by
  sorry


end triangle_problem_l1846_184638


namespace davids_chemistry_marks_l1846_184600

theorem davids_chemistry_marks
  (english_marks : ℕ)
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (h1 : english_marks = 72)
  (h2 : math_marks = 60)
  (h3 : physics_marks = 35)
  (h4 : biology_marks = 84)
  (h5 : average_marks = 62.6)
  (h6 : (english_marks + math_marks + physics_marks + biology_marks + chemistry_marks : ℚ) / 5 = average_marks) :
  chemistry_marks = 62 :=
by sorry

end davids_chemistry_marks_l1846_184600


namespace fruits_per_person_correct_l1846_184679

/-- The number of fruits each person gets when evenly distributed -/
def fruits_per_person (
  kim_strawberry_multiplier : ℕ)
  (strawberries_per_basket : ℕ)
  (kim_blueberry_baskets : ℕ)
  (blueberries_per_kim_basket : ℕ)
  (brother_strawberry_baskets : ℕ)
  (brother_blackberry_baskets : ℕ)
  (blackberries_per_basket : ℕ)
  (parents_blackberry_difference : ℕ)
  (parents_extra_blueberry_baskets : ℕ)
  (parents_extra_blueberries_per_basket : ℕ)
  (family_size : ℕ) : ℕ :=
  let total_strawberries := 
    (kim_strawberry_multiplier * brother_strawberry_baskets + brother_strawberry_baskets) * strawberries_per_basket
  let total_blueberries := 
    kim_blueberry_baskets * blueberries_per_kim_basket + 
    (kim_blueberry_baskets + parents_extra_blueberry_baskets) * (blueberries_per_kim_basket + parents_extra_blueberries_per_basket)
  let total_blackberries := 
    brother_blackberry_baskets * blackberries_per_basket + 
    (brother_blackberry_baskets * blackberries_per_basket - parents_blackberry_difference)
  let total_fruits := total_strawberries + total_blueberries + total_blackberries
  total_fruits / family_size

theorem fruits_per_person_correct : 
  fruits_per_person 8 15 5 40 3 4 30 75 4 15 4 = 316 := by sorry

end fruits_per_person_correct_l1846_184679


namespace shelter_animals_count_l1846_184641

/-- Calculates the total number of animals in the shelter after adoption and new arrivals --/
def total_animals_after_events (initial_cats initial_dogs initial_rabbits : ℕ) : ℕ :=
  let adopted_cats := initial_cats / 4
  let adopted_dogs := initial_dogs / 3
  let new_cats := 3 * adopted_cats
  let new_dogs := 2 * adopted_dogs
  let final_cats := initial_cats - adopted_cats + new_cats
  let final_dogs := initial_dogs - adopted_dogs + new_dogs
  let final_rabbits := 2 * initial_rabbits
  final_cats + final_dogs + final_rabbits

/-- Theorem stating that given the initial conditions, the total number of animals after events is 210 --/
theorem shelter_animals_count : total_animals_after_events 60 45 30 = 210 := by
  sorry

end shelter_animals_count_l1846_184641


namespace intersection_A_B_l1846_184619

def A : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def B : Set ℝ := {-1, 0, 1}

theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end intersection_A_B_l1846_184619


namespace car_production_total_l1846_184603

/-- The number of cars produced in North America -/
def north_america_cars : ℕ := 3884

/-- The number of cars produced in Europe -/
def europe_cars : ℕ := 2871

/-- The total number of cars produced -/
def total_cars : ℕ := north_america_cars + europe_cars

theorem car_production_total : total_cars = 6755 := by
  sorry

end car_production_total_l1846_184603


namespace circle_center_and_radius_l1846_184601

/-- Given a circle with equation (x-2)^2 + y^2 = 4, prove its center and radius -/
theorem circle_center_and_radius :
  let equation := (fun (x y : ℝ) => (x - 2)^2 + y^2 = 4)
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (2, 0) ∧ radius = 2 ∧
    ∀ (x y : ℝ), equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end circle_center_and_radius_l1846_184601


namespace percent_gain_is_588_l1846_184628

-- Define the number of sheep bought and sold
def total_sheep : ℕ := 900
def sold_first : ℕ := 850
def sold_second : ℕ := 50

-- Define the cost and revenue functions
def cost (price_per_sheep : ℚ) : ℚ := price_per_sheep * total_sheep
def revenue_first (price_per_sheep : ℚ) : ℚ := cost price_per_sheep
def revenue_second (price_per_sheep : ℚ) : ℚ := 
  (revenue_first price_per_sheep / sold_first) * sold_second

-- Define the total revenue and profit
def total_revenue (price_per_sheep : ℚ) : ℚ := 
  revenue_first price_per_sheep + revenue_second price_per_sheep
def profit (price_per_sheep : ℚ) : ℚ := 
  total_revenue price_per_sheep - cost price_per_sheep

-- Define the percent gain
def percent_gain (price_per_sheep : ℚ) : ℚ := 
  (profit price_per_sheep / cost price_per_sheep) * 100

-- Theorem statement
theorem percent_gain_is_588 (price_per_sheep : ℚ) :
  percent_gain price_per_sheep = 52.94 / 9 :=
by sorry

end percent_gain_is_588_l1846_184628


namespace like_terms_exponent_sum_l1846_184617

-- Define variables
variable (a b : ℝ)
variable (m n : ℕ)

-- Define the condition that the terms are like terms
def are_like_terms : Prop :=
  ∃ (k₁ k₂ : ℝ), k₁ * a^(2*m) * b = k₂ * a^4 * b^n

-- Theorem statement
theorem like_terms_exponent_sum :
  are_like_terms a b m n → m + n = 3 :=
by sorry

end like_terms_exponent_sum_l1846_184617


namespace parabola_point_order_l1846_184630

/-- A parabola with equation y = -x^2 + 2x + c -/
structure Parabola where
  c : ℝ

/-- A point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the parabola -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = -p.x^2 + 2*p.x + para.c

theorem parabola_point_order (para : Parabola) (p1 p2 p3 : Point)
  (h1 : p1.x = 0) (h2 : p2.x = 1) (h3 : p3.x = 3)
  (h4 : lies_on p1 para) (h5 : lies_on p2 para) (h6 : lies_on p3 para) :
  p2.y > p1.y ∧ p1.y > p3.y := by
  sorry

end parabola_point_order_l1846_184630


namespace team_selection_ways_l1846_184611

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the required team size
def team_size : ℕ := 8

-- Define the number of quadruplets that must be in the team
def required_quadruplets : ℕ := 2

-- Define the number of non-quadruplet players
def non_quadruplet_players : ℕ := total_players - num_quadruplets

-- Define the number of additional players needed after selecting quadruplets
def additional_players : ℕ := team_size - required_quadruplets

-- Theorem statement
theorem team_selection_ways : 
  (Nat.choose num_quadruplets required_quadruplets) * 
  (Nat.choose non_quadruplet_players additional_players) = 18018 := by
  sorry

end team_selection_ways_l1846_184611


namespace largest_undefined_x_l1846_184610

theorem largest_undefined_x (f : ℝ → ℝ) :
  (∀ x, f x = (x + 2) / (10 * x^2 - 85 * x + 10)) →
  (∃ x, 10 * x^2 - 85 * x + 10 = 0) →
  (∀ x, 10 * x^2 - 85 * x + 10 = 0 → x ≤ 10) :=
by
  sorry

end largest_undefined_x_l1846_184610


namespace simplify_expression_l1846_184651

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) :
  (a - 2) * (b - 2) = -2 * m := by sorry

end simplify_expression_l1846_184651


namespace normal_pdf_max_normal_pdf_max_decreases_normal_spread_increases_l1846_184681

/-- The probability density function of a normal distribution -/
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

/-- The maximum height of the normal distribution occurs at x = μ -/
theorem normal_pdf_max (μ σ : ℝ) (h : σ > 0) :
  ∀ x, normal_pdf μ σ x ≤ normal_pdf μ σ μ :=
sorry

/-- As σ increases, the maximum height of the normal distribution decreases -/
theorem normal_pdf_max_decreases (μ : ℝ) (σ₁ σ₂ : ℝ) (h₁ : σ₁ > 0) (h₂ : σ₂ > 0) (h₃ : σ₁ < σ₂) :
  normal_pdf μ σ₂ μ < normal_pdf μ σ₁ μ :=
sorry

/-- The spread of the normal distribution increases as σ increases -/
theorem normal_spread_increases (μ : ℝ) (σ₁ σ₂ : ℝ) (h₁ : σ₁ > 0) (h₂ : σ₂ > 0) (h₃ : σ₁ < σ₂) (ε : ℝ) (hε : ε > 0) :
  ∃ x, |x - μ| > ε ∧ normal_pdf μ σ₂ x > normal_pdf μ σ₁ x :=
sorry

end normal_pdf_max_normal_pdf_max_decreases_normal_spread_increases_l1846_184681


namespace remainder_problem_l1846_184622

theorem remainder_problem (x y : ℤ) 
  (hx : x % 60 = 53)
  (hy : y % 45 = 28) :
  (3 * x - 2 * y) % 30 = 13 := by
sorry

end remainder_problem_l1846_184622


namespace f_properties_l1846_184644

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sin x * Real.cos x - 1/2

theorem f_properties :
  ∃ (k : ℤ), 
    (∀ x : ℝ, f x = f (π/4 - x)) ∧ 
    (∀ x : ℝ, (∃ k : ℤ, x ∈ Set.Icc (k * π - 3*π/8) (k * π + π/8)) → (deriv f) x > 0) ∧
    (f (π/2) = -1/2 ∧ ∀ x ∈ Set.Icc 0 (π/2), f x ≥ -1/2) :=
by sorry

end f_properties_l1846_184644


namespace binary_111111_equals_63_l1846_184653

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_111111_equals_63 :
  binary_to_decimal [true, true, true, true, true, true] = 63 := by
  sorry

end binary_111111_equals_63_l1846_184653


namespace trigonometric_identity_l1846_184667

theorem trigonometric_identity (x : ℝ) :
  Real.sin (x + 2 * Real.pi) * Real.cos (2 * x - 7 * Real.pi / 2) +
  Real.sin (3 * Real.pi / 2 - x) * Real.sin (2 * x - 5 * Real.pi / 2) =
  Real.cos (3 * x) := by sorry

end trigonometric_identity_l1846_184667


namespace bee_count_l1846_184647

/-- The total number of bees in a hive after additional bees fly in -/
def total_bees (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given 16 initial bees and 9 additional bees, the total is 25 -/
theorem bee_count : total_bees 16 9 = 25 := by
  sorry

end bee_count_l1846_184647


namespace min_value_of_exponential_sum_l1846_184648

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 3 * y - 2 = 0) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b : ℝ), a + 3 * b - 2 = 0 → 2^a + 8^b ≥ m :=
sorry

end min_value_of_exponential_sum_l1846_184648


namespace polynomial_expansion_l1846_184697

theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 4 * z^2 - 5) * (4 * z^4 - 3 * z^2 + 2) =
  12 * z^7 + 16 * z^6 - 9 * z^5 - 32 * z^4 + 6 * z^3 + 23 * z^2 - 10 := by
  sorry

end polynomial_expansion_l1846_184697


namespace fraction_difference_numerator_l1846_184690

theorem fraction_difference_numerator : ∃ (p q : ℕ+), 
  (2024 : ℚ) / 2023 - (2023 : ℚ) / 2024 = (p : ℚ) / q ∧ 
  Nat.gcd p q = 1 ∧ 
  p = 4047 := by
sorry

end fraction_difference_numerator_l1846_184690


namespace repeating_decimal_35_sum_l1846_184609

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (n : ℕ) : ℚ :=
  n / 99

theorem repeating_decimal_35_sum : 
  ∀ a b : ℕ, 
  a > 0 → b > 0 →
  RepeatingDecimal 35 = a / b →
  Nat.gcd a b = 1 →
  a + b = 134 := by
  sorry

end repeating_decimal_35_sum_l1846_184609


namespace rickey_race_time_l1846_184646

/-- 
Given:
- Prejean's speed is three-quarters that of Rickey's
- The total time for both to run the race is 70 minutes
Prove: Rickey took 30 minutes to finish the race
-/
theorem rickey_race_time :
  ∀ (rickey_speed prejean_speed rickey_time prejean_time : ℚ),
  prejean_speed = (3 / 4) * rickey_speed →
  rickey_time + prejean_time = 70 →
  prejean_time = (4 / 3) * rickey_time →
  rickey_time = 30 :=
by sorry

end rickey_race_time_l1846_184646


namespace unique_covering_100x100_l1846_184635

/-- A frame is the border of a square in a grid. -/
structure Frame where
  side_length : ℕ

/-- A covering is a list of non-overlapping frames that completely cover a square grid. -/
structure Covering where
  frames : List Frame
  non_overlapping : ∀ (f1 f2 : Frame), f1 ∈ frames → f2 ∈ frames → f1 ≠ f2 → 
    f1.side_length ≠ f2.side_length
  complete : ∀ (n : ℕ), n ∈ List.range 50 → 
    ∃ (f : Frame), f ∈ frames ∧ f.side_length = 100 - 2 * n

/-- The theorem states that there is a unique covering of a 100×100 grid with 50 frames. -/
theorem unique_covering_100x100 : 
  ∃! (c : Covering), c.frames.length = 50 ∧ 
    (∀ (f : Frame), f ∈ c.frames → f.side_length ≤ 100 ∧ f.side_length % 2 = 0) :=
sorry

end unique_covering_100x100_l1846_184635


namespace phoenix_hike_l1846_184639

/-- Phoenix's hiking problem -/
theorem phoenix_hike (a b c d e : ℝ) 
  (h1 : a + b + c = 36)
  (h2 : (b + c + d) / 3 = 16)
  (h3 : c + d + e = 45)
  (h4 : a + d = 31) :
  a + b + c + d + e = 81 := by
  sorry

end phoenix_hike_l1846_184639


namespace test_subjects_count_l1846_184696

def number_of_colors : ℕ := 8
def colors_per_code : ℕ := 4
def unidentified_subjects : ℕ := 19

theorem test_subjects_count :
  (number_of_colors.choose colors_per_code) + unidentified_subjects = 299 :=
by sorry

end test_subjects_count_l1846_184696


namespace circle_theorem_l1846_184605

-- Define the circle and points
def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

def O : ℝ × ℝ := (0, 0)
def r : ℝ := 52

-- Define the points based on the given conditions
theorem circle_theorem (A B : ℝ × ℝ) (P Q : ℝ × ℝ) :
  A ∈ Circle O r →
  B ∈ Circle O r →
  P.1 = 28 ∧ P.2 = 0 →
  (Q.1 - A.1)^2 + (Q.2 - A.2)^2 = 15^2 →
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 15^2 →
  ∃ t : ℝ, Q = (t * B.1, t * B.2) →
  (B.1 - Q.1)^2 + (B.2 - Q.2)^2 = 11^2 :=
by sorry


end circle_theorem_l1846_184605


namespace bowling_ball_weight_l1846_184654

/-- Given that nine identical bowling balls weigh the same as four identical canoes,
    and one canoe weighs 36 pounds, prove that one bowling ball weighs 16 pounds. -/
theorem bowling_ball_weight (canoe_weight : ℕ) (ball_weight : ℚ)
  (h1 : canoe_weight = 36)
  (h2 : 9 * ball_weight = 4 * canoe_weight) :
  ball_weight = 16 := by
  sorry

end bowling_ball_weight_l1846_184654


namespace remaining_bonus_l1846_184657

def bonus : ℚ := 1496
def kitchen_fraction : ℚ := 1 / 22
def holiday_fraction : ℚ := 1 / 4
def christmas_fraction : ℚ := 1 / 8

theorem remaining_bonus : 
  bonus - (bonus * kitchen_fraction + bonus * holiday_fraction + bonus * christmas_fraction) = 867 := by
  sorry

end remaining_bonus_l1846_184657


namespace anika_maddie_age_ratio_l1846_184692

def anika_age : ℕ := 30
def future_years : ℕ := 15
def future_average_age : ℕ := 50

theorem anika_maddie_age_ratio :
  ∃ (maddie_age : ℕ),
    (anika_age + future_years + maddie_age + future_years) / 2 = future_average_age ∧
    anika_age * 4 = maddie_age * 3 := by
  sorry

end anika_maddie_age_ratio_l1846_184692


namespace vacation_savings_l1846_184680

def total_income : ℝ := 72800
def total_expenses : ℝ := 54200
def deposit_rate : ℝ := 0.1

theorem vacation_savings : 
  (total_income - total_expenses) * (1 - deposit_rate) = 16740 := by
  sorry

end vacation_savings_l1846_184680


namespace symmetric_points_sum_l1846_184614

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

/-- Given two points M(a,3) and N(-4,b) symmetric with respect to the origin, prove that a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin a 3 (-4) b) : a + b = 1 := by
  sorry

end symmetric_points_sum_l1846_184614


namespace socks_cost_theorem_l1846_184675

def flat_rate : ℝ := 5
def shipping_rate : ℝ := 0.2
def shirt_price : ℝ := 12
def shirt_quantity : ℕ := 3
def shorts_price : ℝ := 15
def shorts_quantity : ℕ := 2
def swim_trunks_price : ℝ := 14
def swim_trunks_quantity : ℕ := 1
def total_bill : ℝ := 102

def known_items_cost : ℝ := 
  shirt_price * shirt_quantity + 
  shorts_price * shorts_quantity + 
  swim_trunks_price * swim_trunks_quantity

theorem socks_cost_theorem (socks_price : ℝ) : 
  (known_items_cost + socks_price > 50 → 
    known_items_cost + socks_price + shipping_rate * (known_items_cost + socks_price) = total_bill) →
  (known_items_cost + socks_price ≤ 50 → 
    known_items_cost + socks_price + flat_rate = total_bill) →
  socks_price = 5 := by
sorry

end socks_cost_theorem_l1846_184675


namespace scientific_notation_of_190_million_l1846_184659

theorem scientific_notation_of_190_million :
  (190000000 : ℝ) = 1.9 * (10 : ℝ)^8 := by
  sorry

end scientific_notation_of_190_million_l1846_184659


namespace tan_half_sum_of_angles_l1846_184616

theorem tan_half_sum_of_angles (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 3/5)
  (h2 : Real.sin a + Real.sin b = 5/13) : 
  Real.tan ((a + b)/2) = 25/39 := by
  sorry

end tan_half_sum_of_angles_l1846_184616


namespace completing_square_quadratic_l1846_184684

theorem completing_square_quadratic (x : ℝ) :
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) := by
  sorry

end completing_square_quadratic_l1846_184684


namespace complex_equation_solution_l1846_184668

theorem complex_equation_solution :
  ∃ (z : ℂ), (1 - Complex.I) * z = 2 * Complex.I ∧ z = -1 + Complex.I := by
  sorry

end complex_equation_solution_l1846_184668


namespace neighbor_dog_ate_five_chickens_l1846_184620

/-- The number of chickens eaten by the neighbor's dog -/
def chickens_eaten (initial : ℕ) (final : ℕ) : ℕ :=
  2 * initial + 6 - final

theorem neighbor_dog_ate_five_chickens : chickens_eaten 4 13 = 5 := by
  sorry

end neighbor_dog_ate_five_chickens_l1846_184620


namespace number_of_boys_in_class_l1846_184666

/-- The number of boys in a class with given height information -/
theorem number_of_boys_in_class :
  ∀ (n : ℕ) (initial_avg real_avg wrong_height actual_height : ℝ),
  initial_avg = 180 →
  wrong_height = 166 →
  actual_height = 106 →
  real_avg = 178 →
  initial_avg * n - (wrong_height - actual_height) = real_avg * n →
  n = 30 := by
sorry

end number_of_boys_in_class_l1846_184666


namespace simple_interest_rate_l1846_184615

/-- Given a principal amount that grows to 7/6 of itself in 7 years under simple interest, 
    the annual interest rate is 1/42. -/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) : 
  ∃ R : ℝ, R > 0 ∧ P * (1 + 7 * R) = 7/6 * P ∧ R = 1/42 := by
sorry

end simple_interest_rate_l1846_184615


namespace tangent_range_l1846_184625

/-- The circle C in the Cartesian coordinate system -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The line equation -/
def Line (k x y : ℝ) : Prop := y = k*(x + 1)

/-- Point P is on the line -/
def PointOnLine (P : ℝ × ℝ) (k : ℝ) : Prop :=
  Line k P.1 P.2

/-- Two tangents from P to the circle are perpendicular -/
def PerpendicularTangents (P : ℝ × ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, Circle A.1 A.2 ∧ Circle B.1 B.2 ∧
    (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0

/-- The main theorem -/
theorem tangent_range (k : ℝ) :
  (∃ P : ℝ × ℝ, PointOnLine P k ∧ PerpendicularTangents P) →
  k ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
sorry

end tangent_range_l1846_184625


namespace charlies_lollipops_l1846_184698

/-- Given the number of lollipops of each flavor and the number of friends,
    prove that the number of lollipops Charlie keeps is the remainder when
    the total number of lollipops is divided by the number of friends. -/
theorem charlies_lollipops
  (cherry wintergreen grape shrimp_cocktail raspberry : ℕ)
  (friends : ℕ) (friends_pos : friends > 0) :
  let total := cherry + wintergreen + grape + shrimp_cocktail + raspberry
  (total % friends) = total - friends * (total / friends) :=
by sorry

end charlies_lollipops_l1846_184698


namespace sufficient_condition_B_proper_subset_A_l1846_184694

/-- Set A is defined as {x | x^2 + x - 6 = 0} -/
def A : Set ℝ := {x | x^2 + x - 6 = 0}

/-- Set B is defined as {x | x * m + 1 = 0} -/
def B (m : ℝ) : Set ℝ := {x | x * m + 1 = 0}

/-- Theorem stating a sufficient condition for B to be a proper subset of A -/
theorem sufficient_condition_B_proper_subset_A :
  ∀ m : ℝ, m ∈ ({0, 1/3} : Set ℝ) → B m ⊂ A ∧ B m ≠ A :=
sorry

end sufficient_condition_B_proper_subset_A_l1846_184694


namespace sqrt_seven_to_sixth_l1846_184671

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by sorry

end sqrt_seven_to_sixth_l1846_184671


namespace cosine_identity_73_47_l1846_184602

theorem cosine_identity_73_47 :
  let α : Real := 73 * π / 180
  let β : Real := 47 * π / 180
  (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos α) * (Real.cos β) = 3/4 := by
  sorry

end cosine_identity_73_47_l1846_184602


namespace nitin_ranks_from_last_l1846_184685

/-- Calculates the rank from last given the total number of students and rank from first -/
def rankFromLast (totalStudents : ℕ) (rankFromFirst : ℕ) : ℕ :=
  totalStudents - rankFromFirst + 1

theorem nitin_ranks_from_last 
  (totalStudents : ℕ) 
  (mathRank : ℕ) 
  (englishRank : ℕ) 
  (h1 : totalStudents = 75) 
  (h2 : mathRank = 24) 
  (h3 : englishRank = 18) : 
  (rankFromLast totalStudents mathRank = 52) ∧ 
  (rankFromLast totalStudents englishRank = 58) :=
by
  sorry

end nitin_ranks_from_last_l1846_184685


namespace smallest_prime_factor_in_C_l1846_184688

def C : Set Nat := {70, 72, 75, 76, 78}

theorem smallest_prime_factor_in_C :
  ∃ n ∈ C, (∃ p : Nat, Nat.Prime p ∧ p ∣ n ∧ p = 2) ∧
  ∀ m ∈ C, ∀ q : Nat, Nat.Prime q → q ∣ m → q ≥ 2 :=
sorry

end smallest_prime_factor_in_C_l1846_184688


namespace solve_plug_problem_l1846_184670

def plug_problem (mittens_pairs : ℕ) (original_plug_pairs_diff : ℕ) (final_plugs : ℕ) : Prop :=
  let original_plug_pairs : ℕ := mittens_pairs + original_plug_pairs_diff
  let original_plugs : ℕ := original_plug_pairs * 2
  let added_plugs : ℕ := final_plugs - original_plugs
  let added_pairs : ℕ := added_plugs / 2
  added_pairs = 70

theorem solve_plug_problem :
  plug_problem 150 20 400 :=
by sorry

end solve_plug_problem_l1846_184670


namespace cyclic_quadrilateral_circumcentre_l1846_184693

/-- A point in the Euclidean plane -/
structure Point : Type :=
  (x : ℝ) (y : ℝ)

/-- A line in the Euclidean plane -/
structure Line : Type :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Definition of a cyclic quadrilateral -/
def is_cyclic_quadrilateral (A B X C O : Point) : Prop := sorry

/-- Definition of a point lying on a line -/
def point_on_line (P : Point) (L : Line) : Prop := sorry

/-- Definition of equality of distances -/
def distance_eq (A B C D : Point) : Prop := sorry

/-- Definition of a circumcentre of a triangle -/
def is_circumcentre (O : Point) (A B C : Point) : Prop := sorry

/-- Definition of a perpendicular bisector of a line segment -/
def perpendicular_bisector (L : Line) (A B : Point) : Prop := sorry

/-- Definition of a point lying on a perpendicular bisector -/
def point_on_perp_bisector (P : Point) (L : Line) (A B : Point) : Prop := sorry

theorem cyclic_quadrilateral_circumcentre 
  (A B X C O D E : Point) (BX CX : Line) :
  is_cyclic_quadrilateral A B X C O →
  point_on_line D BX →
  point_on_line E CX →
  distance_eq A D B D →
  distance_eq A E C E →
  ∃ (O₁ : Point), is_circumcentre O₁ D E X ∧
    ∃ (L : Line), perpendicular_bisector L O A ∧ point_on_perp_bisector O₁ L O A :=
sorry

end cyclic_quadrilateral_circumcentre_l1846_184693


namespace min_tangent_length_l1846_184629

/-- The minimum length of a tangent from a point on the line y = x + 2 to the circle (x - 4)² + (y + 2)² = 1 is √31. -/
theorem min_tangent_length (x y : ℝ) : 
  let line := {(x, y) | y = x + 2}
  let circle := {(x, y) | (x - 4)^2 + (y + 2)^2 = 1}
  let center := (4, -2)
  let dist_to_line (p : ℝ × ℝ) := |p.1 + p.2 + 2| / Real.sqrt 2
  let min_dist := dist_to_line center
  let tangent_length := Real.sqrt (min_dist^2 - 1)
  tangent_length = Real.sqrt 31 := by sorry

end min_tangent_length_l1846_184629


namespace expression_value_l1846_184652

theorem expression_value (x y : ℝ) 
  (hx : (x - 15)^2 = 169) 
  (hy : (y - 1)^3 = -0.125) : 
  Real.sqrt x - Real.sqrt (2 * x * y) - (2 * y - x)^(1/3) = 3 ∨
  Real.sqrt x - Real.sqrt (2 * x * y) - (2 * y - x)^(1/3) = 1 := by
  sorry

end expression_value_l1846_184652


namespace circle_bisection_theorem_l1846_184642

/-- A circle in the 2D plane. -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the 2D plane. -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Predicate to check if a line bisects a circle. -/
def bisects (l : Line) (c : Circle) : Prop :=
  ∃ (x y : ℝ), c.equation x y ∧ l.equation x y

/-- The main theorem stating that if a specific circle is bisected by a specific line, then a = 1. -/
theorem circle_bisection_theorem (a : ℝ) : 
  let c : Circle := { equation := fun x y => x^2 + y^2 + 2*x - 4*y = 0 }
  let l : Line := { equation := fun x y => 3*x + y + a = 0 }
  bisects l c → a = 1 := by
  sorry

end circle_bisection_theorem_l1846_184642


namespace segment_mapping_l1846_184678

theorem segment_mapping (a : ℝ) : ∃ (x y : ℝ), 
  (∃ (AB A'B' : ℝ), AB = 3 ∧ A'B' = 6 ∧
  (∀ (P D P' D' : ℝ), 
    (P - D = x ∧ P' - D' = 2*x) →
    (x = a → x + y = 3*a))) :=
by sorry

end segment_mapping_l1846_184678


namespace point_division_vector_representation_l1846_184669

/-- Given a line segment CD and a point Q on CD such that CQ:QD = 3:5,
    prove that Q⃗ = (5/8)C⃗ + (3/8)D⃗ -/
theorem point_division_vector_representation 
  (C D Q : EuclideanSpace ℝ (Fin n)) 
  (h_on_segment : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D) 
  (h_ratio : dist C Q / dist Q D = 3 / 5) :
  Q = (5/8) • C + (3/8) • D :=
by sorry

end point_division_vector_representation_l1846_184669


namespace similar_triangles_corresponding_side_length_l1846_184634

/-- Given two similar right triangles, where the first triangle has a leg of 15 inches and a hypotenuse
    of 17 inches, and the second triangle has a hypotenuse of 34 inches, the length of the side in the
    second triangle corresponding to the 15-inch leg is 30 inches. -/
theorem similar_triangles_corresponding_side_length (a b c d : ℝ) : 
  a = 15 →  -- First leg of the first triangle
  c = 17 →  -- Hypotenuse of the first triangle
  d = 34 →  -- Hypotenuse of the second triangle
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for the first triangle
  ∃ (k : ℝ), k > 0 ∧ d = k * c ∧ k * a = 30  -- The corresponding side in the second triangle is 30 inches
  := by sorry

end similar_triangles_corresponding_side_length_l1846_184634


namespace group_size_proof_l1846_184664

theorem group_size_proof (W : ℝ) (n : ℕ) 
  (h1 : (W + 20) / n = W / n + 4) 
  (h2 : W > 0) 
  (h3 : n > 0) : n = 5 := by
  sorry

end group_size_proof_l1846_184664


namespace attend_both_reunions_l1846_184632

/-- The number of people attending both reunions at the Taj Hotel -/
def both_reunions (total guests : ℕ) (oates hall : ℕ) : ℕ :=
  oates + hall - total

/-- Theorem stating the number of people attending both reunions -/
theorem attend_both_reunions : 
  both_reunions 150 70 52 = 28 := by sorry

end attend_both_reunions_l1846_184632


namespace exam_pass_percentage_l1846_184656

/-- The percentage of students who passed an examination, given the total number of students and the number of students who failed. -/
def percentage_passed (total : ℕ) (failed : ℕ) : ℚ :=
  (total - failed : ℚ) / total * 100

/-- Theorem stating that the percentage of students who passed is 35% -/
theorem exam_pass_percentage :
  percentage_passed 540 351 = 35 := by
  sorry

end exam_pass_percentage_l1846_184656


namespace P_equals_set_l1846_184674

def P : Set ℝ := {x | x^2 = 1}

theorem P_equals_set : P = {-1, 1} := by
  sorry

end P_equals_set_l1846_184674


namespace fraction_product_square_l1846_184627

theorem fraction_product_square : (8 / 9 : ℚ)^2 * (1 / 3 : ℚ)^2 = 64 / 729 := by
  sorry

end fraction_product_square_l1846_184627


namespace prime_sum_91_l1846_184631

theorem prime_sum_91 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h_sum : 5 * p + 3 * q = 91) : p = 17 ∧ q = 2 := by
  sorry

end prime_sum_91_l1846_184631


namespace zoo_visitors_saturday_l1846_184691

/-- The number of people who visited the zoo on Friday -/
def friday_visitors : ℕ := 1250

/-- The ratio of Saturday visitors to Friday visitors -/
def saturday_ratio : ℕ := 3

/-- The number of people who visited the zoo on Saturday -/
def saturday_visitors : ℕ := friday_visitors * saturday_ratio

theorem zoo_visitors_saturday : saturday_visitors = 3750 := by
  sorry

end zoo_visitors_saturday_l1846_184691


namespace hyperbola_tangent_dot_product_l1846_184677

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

/-- The asymptotes of the hyperbola -/
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

/-- A point is on the line l -/
def on_line_l (x y : ℝ) : Prop := sorry

/-- The line l is tangent to the hyperbola at point P -/
def is_tangent (P : ℝ × ℝ) : Prop := 
  hyperbola P.1 P.2 ∧ on_line_l P.1 P.2 ∧ 
  ∀ Q : ℝ × ℝ, Q ≠ P → on_line_l Q.1 Q.2 → ¬hyperbola Q.1 Q.2

theorem hyperbola_tangent_dot_product 
  (P M N : ℝ × ℝ) 
  (h_tangent : is_tangent P) 
  (h_M : on_line_l M.1 M.2 ∧ asymptote M.1 M.2) 
  (h_N : on_line_l N.1 N.2 ∧ asymptote N.1 N.2) :
  M.1 * N.1 + M.2 * N.2 = 3 := 
sorry

end hyperbola_tangent_dot_product_l1846_184677


namespace range_of_f_l1846_184663

/-- The quadratic function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The range of f is [2, +∞) -/
theorem range_of_f : Set.range f = { y | 2 ≤ y } := by sorry

end range_of_f_l1846_184663


namespace problem_solution_l1846_184608

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + 2*x - 3

-- Define the set M
def M : Set ℝ := {x | f x ≤ -1}

-- Theorem statement
theorem problem_solution :
  (M = {x : ℝ | x ≤ 0}) ∧
  (∀ x ∈ M, x * (f x)^2 - x^2 * (f x) ≤ 0) :=
by sorry

end problem_solution_l1846_184608


namespace other_number_is_twenty_l1846_184607

theorem other_number_is_twenty (x y : ℤ) 
  (sum_eq : 3 * x + 2 * y = 145) 
  (one_is_35 : x = 35 ∨ y = 35) : 
  (x ≠ 35 → x = 20) ∧ (y ≠ 35 → y = 20) :=
sorry

end other_number_is_twenty_l1846_184607


namespace cubic_expression_equality_l1846_184613

theorem cubic_expression_equality : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by sorry

end cubic_expression_equality_l1846_184613


namespace triangle_count_bound_l1846_184660

/-- A structure representing a configuration of points and equilateral triangles on a plane. -/
structure PointTriangleConfig where
  n : ℕ  -- number of points
  k : ℕ  -- number of equilateral triangles
  n_gt_3 : n > 3
  convex_n_gon : Bool  -- represents that the n points form a convex n-gon
  unit_triangles : Bool  -- represents that the k triangles are equilateral with side length 1

/-- Theorem stating that the number of equilateral triangles is less than 2/3 of the number of points. -/
theorem triangle_count_bound (config : PointTriangleConfig) : config.k < 2 * config.n / 3 := by
  sorry

end triangle_count_bound_l1846_184660


namespace concat_reverse_divisible_by_99_l1846_184623

def is_valid_permutation (p : List Nat) : Prop :=
  p.length = 10 ∧ 
  p.head? ≠ some 0 ∧ 
  (∀ i, i ∈ p → i < 10) ∧
  (∀ i, i < 10 → i ∈ p)

def concat_with_reverse (p : List Nat) : List Nat :=
  p ++ p.reverse

def to_number (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem concat_reverse_divisible_by_99 (p : List Nat) 
  (h : is_valid_permutation p) : 
  99 ∣ to_number (concat_with_reverse p) := by
  sorry

end concat_reverse_divisible_by_99_l1846_184623


namespace additional_lawn_needed_l1846_184662

/-- The amount LaKeisha charges per square foot of lawn mowed -/
def lawn_rate : ℚ := 1 / 10

/-- The amount LaKeisha charges per linear foot of hedge trimmed -/
def hedge_rate : ℚ := 1 / 20

/-- The amount LaKeisha charges per square foot of leaves raked -/
def rake_rate : ℚ := 1 / 50

/-- The cost of the book set -/
def book_cost : ℚ := 375

/-- The number of lawns LaKeisha has mowed -/
def lawns_mowed : ℕ := 5

/-- The length of each lawn -/
def lawn_length : ℕ := 30

/-- The width of each lawn -/
def lawn_width : ℕ := 20

/-- The number of linear feet of hedges LaKeisha has trimmed -/
def hedges_trimmed : ℕ := 100

/-- The number of square feet of leaves LaKeisha has raked -/
def leaves_raked : ℕ := 500

/-- The additional square feet of lawn LaKeisha needs to mow -/
def additional_lawn : ℕ := 600

theorem additional_lawn_needed :
  (book_cost - (lawn_rate * (lawns_mowed * lawn_length * lawn_width : ℚ) +
    hedge_rate * hedges_trimmed +
    rake_rate * leaves_raked)) / lawn_rate = additional_lawn := by sorry

end additional_lawn_needed_l1846_184662


namespace carolines_socks_l1846_184665

/-- Given Caroline's sock inventory changes, calculate how many pairs she received as a gift. -/
theorem carolines_socks (initial : ℕ) (lost : ℕ) (donation_fraction : ℚ) (purchased : ℕ) (final : ℕ) : 
  initial = 40 →
  lost = 4 →
  donation_fraction = 2/3 →
  purchased = 10 →
  final = 25 →
  final = initial - lost - (initial - lost) * donation_fraction + purchased + (final - (initial - lost - (initial - lost) * donation_fraction + purchased)) :=
by sorry

end carolines_socks_l1846_184665


namespace hyperbola_eccentricity_l1846_184658

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.sqrt 6 / 6) :
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 42 / 6 := by sorry

end hyperbola_eccentricity_l1846_184658


namespace domain_of_f_l1846_184636

def f (x : ℝ) : ℝ := (2 * x - 3) ^ (1/3) + (9 - x) ^ (1/2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iic 9 := by sorry

end domain_of_f_l1846_184636


namespace right_triangle_identification_l1846_184606

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_identification :
  (is_right_triangle 5 12 13) ∧
  ¬(is_right_triangle 6 7 8) ∧
  ¬(is_right_triangle 3 4 6) ∧
  ¬(is_right_triangle 7 12 15) :=
by sorry

end right_triangle_identification_l1846_184606


namespace exists_zero_sum_choice_l1846_184645

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set of 8 points in the plane -/
def Points : Finset Point := sorry

/-- The area of a triangle formed by three points -/
def triangleArea (p₁ p₂ p₃ : Point) : ℝ := sorry

/-- The list of areas of all possible triangles formed by the points -/
def triangleAreas : List ℝ := sorry

/-- A choice of signs for the areas -/
def SignChoice := List Bool

/-- Apply a sign choice to a list of areas -/
def applySignChoice (areas : List ℝ) (choice : SignChoice) : List ℝ := sorry

theorem exists_zero_sum_choice :
  ∃ (choice : SignChoice), List.sum (applySignChoice triangleAreas choice) = 0 := by
  sorry

end exists_zero_sum_choice_l1846_184645


namespace relationship_abcd_l1846_184695

theorem relationship_abcd (a b c d : ℝ) 
  (hab : a < b) 
  (hdc : d < c) 
  (hcab : (c - a) * (c - b) < 0) 
  (hdab : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b := by sorry

end relationship_abcd_l1846_184695


namespace twins_age_problem_l1846_184687

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 15 → age = 7 := by
  sorry

end twins_age_problem_l1846_184687


namespace magnificent_monday_l1846_184686

-- Define a structure for a month
structure Month where
  days : Nat
  firstMonday : Nat

-- Define a function to calculate the date of the nth Monday
def nthMonday (m : Month) (n : Nat) : Nat :=
  m.firstMonday + 7 * (n - 1)

-- Theorem statement
theorem magnificent_monday (m : Month) 
  (h1 : m.days = 31)
  (h2 : m.firstMonday = 2) :
  nthMonday m 5 = 30 := by
  sorry

end magnificent_monday_l1846_184686


namespace special_function_at_two_l1846_184621

/-- A function satisfying the given property for all real x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f x - f (f y + f (-x)) + x

/-- Theorem stating that for any function satisfying the special property, f(2) = -2 -/
theorem special_function_at_two (f : ℝ → ℝ) (h : special_function f) : f 2 = -2 := by
  sorry

end special_function_at_two_l1846_184621
