import Mathlib

namespace inequality_solution_set_l507_50792

theorem inequality_solution_set (a : ℝ) (h : a < 2) :
  {x : ℝ | a * x > 2 * x + a - 2} = {x : ℝ | x < 1} := by
sorry

end inequality_solution_set_l507_50792


namespace max_value_of_function_l507_50745

theorem max_value_of_function (a : ℕ+) :
  (∃ (y : ℕ+), ∀ (x : ℝ), x + Real.sqrt (13 - 2 * (a : ℝ) * x) ≤ (y : ℝ)) →
  (∃ (y_max : ℕ+), ∀ (x : ℝ), x + Real.sqrt (13 - 2 * (a : ℝ) * x) ≤ (y_max : ℝ) ∧ y_max = 7) :=
by sorry

end max_value_of_function_l507_50745


namespace perfect_cubes_difference_l507_50778

theorem perfect_cubes_difference (n : ℕ) : 
  (∃ x y : ℕ, (n + 195 = x^3) ∧ (n - 274 = y^3)) ↔ n = 2002 := by
  sorry

end perfect_cubes_difference_l507_50778


namespace infinitely_many_solutions_implies_a_value_l507_50775

theorem infinitely_many_solutions_implies_a_value 
  (a b : ℝ) 
  (h : ∀ x : ℝ, 2*a*(x-1) = (5-a)*x + 3*b) :
  a = 5/3 := by
sorry

end infinitely_many_solutions_implies_a_value_l507_50775


namespace quadratic_equation_solution_l507_50789

theorem quadratic_equation_solution :
  let a : ℝ := 5
  let b : ℝ := -2 * Real.sqrt 15
  let c : ℝ := -2
  let x₁ : ℝ := -1 + Real.sqrt 15 / 5
  let x₂ : ℝ := 1 + Real.sqrt 15 / 5
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) ∧
  (a * x₁^2 + b * x₁ + c = 0) ∧
  (a * x₂^2 + b * x₂ + c = 0) :=
by sorry

end quadratic_equation_solution_l507_50789


namespace perpendicular_vectors_l507_50751

/-- Given vectors a and b in ℝ², prove that they are perpendicular if and only if x = -3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) : 
  a = (-1, 3) → b = (-3, x) → (a.1 * b.1 + a.2 * b.2 = 0 ↔ x = -3) :=
by sorry

end perpendicular_vectors_l507_50751


namespace geometric_progression_first_term_l507_50704

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 6) 
  (h2 : sum_first_two = 8/3) :
  ∃ (a : ℝ), (a = 6 + 2 * Real.sqrt 5 ∨ a = 6 - 2 * Real.sqrt 5) ∧ 
  (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a * (1 + r)) := by
  sorry

end geometric_progression_first_term_l507_50704


namespace smallest_power_complex_equality_l507_50712

theorem smallest_power_complex_equality (n : ℕ) (c d : ℝ) :
  (n > 0) →
  (c > 0) →
  (d > 0) →
  (∀ k < n, ∃ a b : ℝ, (a > 0 ∧ b > 0 ∧ (a + b * I) ^ (2 * k) ≠ (a - b * I) ^ (2 * k))) →
  ((c + d * I) ^ (2 * n) = (c - d * I) ^ (2 * n)) →
  (d / c = 1) := by
sorry

end smallest_power_complex_equality_l507_50712


namespace third_car_manufacture_year_l507_50766

def year_first_car : ℕ := 1970
def years_between_first_and_second : ℕ := 10
def years_between_second_and_third : ℕ := 20

theorem third_car_manufacture_year :
  year_first_car + years_between_first_and_second + years_between_second_and_third = 2000 := by
  sorry

end third_car_manufacture_year_l507_50766


namespace quadratic_roots_sum_product_l507_50753

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∀ x : ℝ, 3 * x^2 - p * x + q = 0 → 
    (∃ r₁ r₂ : ℝ, r₁ + r₂ = 9 ∧ r₁ * r₂ = 15 ∧ 
      (3 * r₁^2 - p * r₁ + q = 0) ∧ (3 * r₂^2 - p * r₂ + q = 0))) →
  p + q = 72 := by
sorry

end quadratic_roots_sum_product_l507_50753


namespace local_minimum_implies_c_equals_2_l507_50701

/-- The function f(x) = x(x-c)^2 has a local minimum at x=2 -/
def has_local_minimum_at_2 (c : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 2| < δ → x * (x - c)^2 ≥ 2 * (2 - c)^2

theorem local_minimum_implies_c_equals_2 :
  ∀ c : ℝ, has_local_minimum_at_2 c → c = 2 :=
sorry

end local_minimum_implies_c_equals_2_l507_50701


namespace trapezoid_diagonal_length_l507_50783

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of base AD
  ad : ℝ
  -- Length of base BC
  bc : ℝ
  -- Length of diagonal AC
  ac : ℝ
  -- Circles on AB, BC, CD as diameters intersect at a single point
  circles_intersect : Prop

/-- The theorem stating that under given conditions, diagonal BD has length 24 -/
theorem trapezoid_diagonal_length (t : Trapezoid) 
  (h1 : t.ad = 16)
  (h2 : t.bc = 10)
  (h3 : t.ac = 10)
  (h4 : t.circles_intersect) :
  ∃ (bd : ℝ), bd = 24 :=
sorry

end trapezoid_diagonal_length_l507_50783


namespace area_of_parallelogram_EFGH_l507_50710

/-- The area of parallelogram EFGH --/
def area_EFGH : ℝ := 15

/-- The base of parallelogram EFGH --/
def base_FG : ℝ := 3

/-- The height from point E to line FG --/
def height_E_to_FG : ℝ := 5

/-- The theorem stating that the area of parallelogram EFGH is 15 square units --/
theorem area_of_parallelogram_EFGH :
  area_EFGH = base_FG * height_E_to_FG :=
by sorry

end area_of_parallelogram_EFGH_l507_50710


namespace fraction_value_l507_50700

theorem fraction_value : (900 ^ 2 : ℚ) / (153 ^ 2 - 147 ^ 2) = 450 := by
  sorry

end fraction_value_l507_50700


namespace f_is_odd_l507_50760

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom add_property : ∀ x y : ℝ, f (x + y) = f x + f y
axiom not_identically_zero : ∃ x : ℝ, f x ≠ 0

-- Define what it means for f to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem f_is_odd : is_odd f := by sorry

end f_is_odd_l507_50760


namespace smallest_sector_angle_l507_50759

theorem smallest_sector_angle (n : ℕ) (a d : ℤ) : 
  n = 8 ∧ 
  (∀ i : ℕ, i < n → (a + i * d : ℤ) > 0) ∧
  (∀ i : ℕ, i < n → (a + i * d : ℤ).natAbs = a + i * d) ∧
  (n : ℤ) * (2 * a + (n - 1) * d) = 360 * 2 →
  a ≥ 38 :=
by sorry

end smallest_sector_angle_l507_50759


namespace two_intersection_points_l507_50735

-- Define the three lines
def line1 (x y : ℝ) : Prop := 2*y - 3*x = 3
def line2 (x y : ℝ) : Prop := x + 3*y = 3
def line3 (x y : ℝ) : Prop := 5*x - 3*y = 6

-- Define a function to check if a point lies on at least two lines
def onAtLeastTwoLines (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

-- Theorem statement
theorem two_intersection_points :
  ∃ (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧ 
    onAtLeastTwoLines p1.1 p1.2 ∧ 
    onAtLeastTwoLines p2.1 p2.2 ∧
    ∀ (p : ℝ × ℝ), onAtLeastTwoLines p.1 p.2 → p = p1 ∨ p = p2 :=
sorry

end two_intersection_points_l507_50735


namespace cos_2alpha_minus_3pi_over_7_l507_50716

theorem cos_2alpha_minus_3pi_over_7 (α : ℝ) 
  (h : Real.sin (α + 2 * Real.pi / 7) = Real.sqrt 6 / 3) : 
  Real.cos (2 * α - 3 * Real.pi / 7) = 1 / 3 := by
sorry

end cos_2alpha_minus_3pi_over_7_l507_50716


namespace parallel_vectors_x_value_l507_50781

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 * k = b.1 ∧ a.2 * k = b.2

/-- Given parallel vectors (2,3) and (x,-6), x equals -4 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (2, 3) (x, -6) → x = -4 :=
by
  sorry

#check parallel_vectors_x_value

end parallel_vectors_x_value_l507_50781


namespace supremum_of_function_l507_50702

theorem supremum_of_function (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ M : ℝ, (∀ x : ℝ, -1/(2*x) - 2/((1-x)) ≤ M) ∧ 
  (∀ ε > 0, ∃ y : ℝ, -1/(2*y) - 2/((1-y)) > M - ε) ∧
  M = -9/2 := by
  sorry

end supremum_of_function_l507_50702


namespace grid_sum_example_unique_transformed_grid_sum_constant_grid_sum_difference_count_grids_with_sum_104_l507_50737

/-- Definition of a 2x2 grid of positive digits -/
structure Grid :=
  (a b c d : ℕ)
  (ha : 0 < a ∧ a < 10)
  (hb : 0 < b ∧ b < 10)
  (hc : 0 < c ∧ c < 10)
  (hd : 0 < d ∧ d < 10)

/-- The grid sum operation -/
def gridSum (g : Grid) : ℕ := 10*g.a + g.b + 10*g.c + g.d + 10*g.a + g.c + 10*g.b + g.d

/-- Theorem for part (a) -/
theorem grid_sum_example : 
  ∃ g : Grid, g.a = 7 ∧ g.b = 3 ∧ g.c = 2 ∧ g.d = 7 ∧ gridSum g = 209 := sorry

/-- Theorem for part (b) -/
theorem unique_transformed_grid_sum :
  ∃! x y : ℕ, ∀ b c : ℕ, 0 < b ∧ b < 9 ∧ 0 < c ∧ c < 10 →
    ∃ g1 g2 : Grid,
      g1.a = 5 ∧ g1.b = b ∧ g1.c = c ∧ g1.d = 7 ∧
      g2.a = x ∧ g2.b = b+1 ∧ g2.c = c-3 ∧ g2.d = y ∧
      gridSum g1 = gridSum g2 := sorry

/-- Theorem for part (c) -/
theorem constant_grid_sum_difference :
  ∃ k : ℤ, ∀ g : Grid,
    gridSum g - gridSum ⟨g.a+1, g.b-2, g.c-1, g.d+1, sorry, sorry, sorry, sorry⟩ = k := sorry

/-- Theorem for part (d) -/
theorem count_grids_with_sum_104 :
  ∃! ls : List Grid, (∀ g ∈ ls, gridSum g = 104) ∧ ls.length = 5 := sorry

end grid_sum_example_unique_transformed_grid_sum_constant_grid_sum_difference_count_grids_with_sum_104_l507_50737


namespace root_of_unity_sum_iff_cube_root_l507_50715

theorem root_of_unity_sum_iff_cube_root (x y : ℂ) : 
  (Complex.abs x = 1 ∧ Complex.abs y = 1 ∧ x ≠ y) → 
  (Complex.abs (x + y) = 1 ↔ (y / x) ^ 3 = 1) := by
  sorry

end root_of_unity_sum_iff_cube_root_l507_50715


namespace cubic_function_monotonicity_l507_50791

-- Define the function f(x) = ax^3
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3

-- State the theorem
theorem cubic_function_monotonicity (a : ℝ) (h : a ≠ 0) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (a > 0 → f a x₁ < f a x₂) ∧ (a < 0 → f a x₁ > f a x₂)) :=
by sorry

end cubic_function_monotonicity_l507_50791


namespace equilateral_triangle_perimeter_l507_50750

theorem equilateral_triangle_perimeter (area : ℝ) (p : ℝ) : 
  area = 50 * Real.sqrt 12 → p = 60 := by sorry

end equilateral_triangle_perimeter_l507_50750


namespace hyperbola_equation_l507_50782

/-- Given a hyperbola with equation x^2/a^2 - y^2/4 = 1 and an asymptote y = (1/2)x,
    prove that the equation of the hyperbola is x^2/16 - y^2/4 = 1 -/
theorem hyperbola_equation (a : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/4 = 1 → ∃ t : ℝ, y = (1/2) * x * t) →
  (∀ x y : ℝ, x^2/16 - y^2/4 = 1 ↔ x^2/a^2 - y^2/4 = 1) :=
by sorry

end hyperbola_equation_l507_50782


namespace tamika_driving_time_l507_50730

-- Define the variables
def tamika_speed : ℝ := 45
def logan_speed : ℝ := 55
def logan_time : ℝ := 5
def distance_difference : ℝ := 85

-- Theorem statement
theorem tamika_driving_time :
  ∃ (h : ℝ), h * tamika_speed = logan_speed * logan_time + distance_difference ∧ h = 8 := by
  sorry

end tamika_driving_time_l507_50730


namespace log_inequality_l507_50762

theorem log_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  let f : ℝ → ℝ := fun x ↦ |Real.log x / Real.log a|
  f (1/4) > f (1/3) ∧ f (1/3) > f 2 := by
  sorry

end log_inequality_l507_50762


namespace complex_number_in_third_quadrant_l507_50771

/-- The complex number (1-i)^2 / (1+i) lies in the third quadrant of the complex plane -/
theorem complex_number_in_third_quadrant :
  let z : ℂ := (1 - Complex.I)^2 / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end complex_number_in_third_quadrant_l507_50771


namespace no_solution_quadratic_l507_50728

theorem no_solution_quadratic (p q r s : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + p*x + q ≠ 0)
  (h2 : ∀ x : ℝ, x^2 + r*x + s ≠ 0) :
  ∀ x : ℝ, 2017*x^2 + (1009*p + 1008*r)*x + 1009*q + 1008*s ≠ 0 := by
sorry

end no_solution_quadratic_l507_50728


namespace arctan_sum_equation_l507_50747

theorem arctan_sum_equation (y : ℝ) : 
  2 * Real.arctan (1/3) + 2 * Real.arctan (1/15) + Real.arctan (1/y) = π/2 → y = 261/242 := by
  sorry

end arctan_sum_equation_l507_50747


namespace factors_of_36_l507_50714

/-- The number of distinct positive factors of 36 is 9. -/
theorem factors_of_36 : Nat.card {d : ℕ | d > 0 ∧ 36 % d = 0} = 9 := by
  sorry

end factors_of_36_l507_50714


namespace part_one_part_two_l507_50705

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0

-- Part 1
theorem part_one (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : ∀ x, ¬(p x a) → ¬(q x)) 
  (h_not_necessary : ∃ x, q x ∧ p x a) : 1 < a ∧ a ≤ 2 := by sorry

end part_one_part_two_l507_50705


namespace function_range_is_all_reals_l507_50721

theorem function_range_is_all_reals :
  ∀ y : ℝ, ∃ x : ℝ, y = (x^2 + 3*x + 2) / (x^2 + x + 1) := by
  sorry

end function_range_is_all_reals_l507_50721


namespace correct_propositions_l507_50752

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
| SpecificToGeneral
| GeneralToSpecific
| SpecificToSpecific
| GeneralToGeneral

-- Define a function to check if a statement about reasoning is correct
def isCorrectStatement (rt : ReasoningType) (rd : ReasoningDirection) : Prop :=
  match rt, rd with
  | ReasoningType.Inductive, ReasoningDirection.SpecificToGeneral => True
  | ReasoningType.Deductive, ReasoningDirection.GeneralToSpecific => True
  | ReasoningType.Analogical, ReasoningDirection.SpecificToSpecific => True
  | _, _ => False

-- Define the five propositions
def proposition1 := isCorrectStatement ReasoningType.Inductive ReasoningDirection.SpecificToGeneral
def proposition2 := isCorrectStatement ReasoningType.Inductive ReasoningDirection.GeneralToGeneral
def proposition3 := isCorrectStatement ReasoningType.Deductive ReasoningDirection.GeneralToSpecific
def proposition4 := isCorrectStatement ReasoningType.Analogical ReasoningDirection.SpecificToGeneral
def proposition5 := isCorrectStatement ReasoningType.Analogical ReasoningDirection.SpecificToSpecific

-- Theorem to prove
theorem correct_propositions :
  {n : Nat | n ∈ [1, 3, 5]} = {n : Nat | n ∈ [1, 2, 3, 4, 5] ∧ 
    match n with
    | 1 => proposition1
    | 2 => proposition2
    | 3 => proposition3
    | 4 => proposition4
    | 5 => proposition5
    | _ => False} :=
by sorry

end correct_propositions_l507_50752


namespace t_shirts_per_package_l507_50799

theorem t_shirts_per_package (total_t_shirts : ℕ) (num_packages : ℕ) 
  (h1 : total_t_shirts = 39)
  (h2 : num_packages = 3)
  (h3 : total_t_shirts % num_packages = 0) :
  total_t_shirts / num_packages = 13 := by
  sorry

end t_shirts_per_package_l507_50799


namespace equidistant_function_property_l507_50744

/-- A function that scales a complex number by a complex factor -/
def g (c d : ℝ) (z : ℂ) : ℂ := (c + d * Complex.I) * z

/-- The theorem stating the properties and result of the problem -/
theorem equidistant_function_property (c d : ℝ) :
  (c > 0) →
  (d > 0) →
  (∀ z : ℂ, Complex.abs (g c d z - z) = Complex.abs (g c d z)) →
  Complex.abs (c + d * Complex.I) = 10 →
  d^2 = 99.75 := by sorry

end equidistant_function_property_l507_50744


namespace fractional_equation_solution_l507_50703

theorem fractional_equation_solution (k : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0)) 
  ↔ k ≠ -3 ∧ k ≠ 5 := by
  sorry

end fractional_equation_solution_l507_50703


namespace frank_candy_count_l507_50772

/-- Given a number of bags and pieces per bag, calculates the total number of pieces -/
def totalPieces (n m : ℕ) : ℕ := n * m

/-- Theorem: For 2 bags with 21 pieces each, the total number of pieces is 42 -/
theorem frank_candy_count : totalPieces 2 21 = 42 := by
  sorry

end frank_candy_count_l507_50772


namespace isosceles_triangles_not_necessarily_congruent_l507_50740

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  /-- Length of the equal sides -/
  side_length : ℝ
  /-- Length of the base -/
  base_length : ℝ
  /-- Radius of the inscribed circle -/
  incircle_radius : ℝ
  /-- side_length is positive -/
  side_length_pos : 0 < side_length
  /-- base_length is positive -/
  base_length_pos : 0 < base_length
  /-- incircle_radius is positive -/
  incircle_radius_pos : 0 < incircle_radius
  /-- The base cannot be longer than twice the side length -/
  base_bound : base_length ≤ 2 * side_length
  /-- Relation between side length, base length, and incircle radius -/
  geometry_constraint : incircle_radius = (base_length * Real.sqrt (side_length^2 - (base_length/2)^2)) / (2 * side_length + base_length)

/-- Two isosceles triangles with the same side length and incircle radius are not necessarily congruent -/
theorem isosceles_triangles_not_necessarily_congruent :
  ∃ (t1 t2 : IsoscelesTriangle), 
    t1.side_length = t2.side_length ∧ 
    t1.incircle_radius = t2.incircle_radius ∧ 
    t1.base_length ≠ t2.base_length :=
by sorry

end isosceles_triangles_not_necessarily_congruent_l507_50740


namespace area_of_five_presentable_set_l507_50765

/-- A complex number is five-presentable if it can be represented as w - 1/w for some complex number w with |w| = 5 -/
def FivePresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 5 ∧ z = w - 1 / w

/-- The set of all five-presentable complex numbers -/
def S : Set ℂ :=
  {z : ℂ | FivePresentable z}

/-- The area of a set in the complex plane -/
noncomputable def area (A : Set ℂ) : ℝ := sorry

theorem area_of_five_presentable_set :
  area S = 624 * Real.pi / 25 := by sorry

end area_of_five_presentable_set_l507_50765


namespace total_snacks_weight_l507_50785

theorem total_snacks_weight (peanuts_weight raisins_weight : ℝ) 
  (h1 : peanuts_weight = 0.1)
  (h2 : raisins_weight = 0.4) :
  peanuts_weight + raisins_weight = 0.5 := by
  sorry

end total_snacks_weight_l507_50785


namespace right_triangle_hypotenuse_l507_50795

/-- Given a right triangle with one leg of 15 inches and the angle opposite to that leg being 30°,
    the length of the hypotenuse is 30 inches. -/
theorem right_triangle_hypotenuse (a b c : ℝ) (θ : ℝ) : 
  a = 15 →  -- One leg is 15 inches
  θ = 30 * π / 180 →  -- Angle opposite to that leg is 30° (converted to radians)
  θ = Real.arcsin (a / c) →  -- Sine of the angle is opposite over hypotenuse
  a ^ 2 + b ^ 2 = c ^ 2 →  -- Pythagorean theorem
  c = 30 :=  -- Hypotenuse is 30 inches
by sorry

end right_triangle_hypotenuse_l507_50795


namespace original_group_size_l507_50719

/-- Represents the work capacity of a group of men --/
def work_capacity (men : ℕ) (days : ℕ) : ℕ := men * days

theorem original_group_size
  (initial_days : ℕ)
  (absent_men : ℕ)
  (final_days : ℕ)
  (h1 : initial_days = 20)
  (h2 : absent_men = 10)
  (h3 : final_days = 40)
  : ∃ (original_size : ℕ),
    work_capacity original_size initial_days =
    work_capacity (original_size - absent_men) final_days ∧
    original_size = 20 :=
by sorry

end original_group_size_l507_50719


namespace combined_solution_x_percentage_l507_50777

/-- Represents a solution composed of liquid X and water -/
structure Solution where
  total_mass : ℝ
  x_percentage : ℝ

/-- The initial solution Y1 -/
def Y1 : Solution :=
  { total_mass := 12
  , x_percentage := 0.3 }

/-- The mass of water that evaporates -/
def evaporated_water : ℝ := 3

/-- The solution Y2 after evaporation -/
def Y2 : Solution :=
  { total_mass := Y1.total_mass - evaporated_water
  , x_percentage := 0.4 }

/-- The mass of Y2 added to the remaining solution -/
def added_Y2_mass : ℝ := 4

/-- Calculates the mass of liquid X in a given solution -/
def liquid_x_mass (s : Solution) : ℝ :=
  s.total_mass * s.x_percentage

/-- Calculates the mass of water in a given solution -/
def water_mass (s : Solution) : ℝ :=
  s.total_mass * (1 - s.x_percentage)

/-- The combined solution after adding Y2 -/
def combined_solution : Solution :=
  { total_mass := Y2.total_mass + added_Y2_mass
  , x_percentage := 0 }  -- Placeholder value, to be proved

theorem combined_solution_x_percentage :
  combined_solution.x_percentage = 0.4 := by
  sorry

end combined_solution_x_percentage_l507_50777


namespace percentage_relation_l507_50796

theorem percentage_relation (x y z : ℝ) : 
  x = 1.2 * y ∧ y = 0.5 * z → x = 0.6 * z := by
  sorry

end percentage_relation_l507_50796


namespace range_of_a_l507_50743

theorem range_of_a (p q : Prop) (h_p : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x)
  (h_q : ∃ x : ℝ, x^2 - 4*x + a ≤ 0) (h_pq : p ∧ q) :
  a ∈ Set.Icc (Real.exp 1) 4 :=
sorry

end range_of_a_l507_50743


namespace power_sum_div_diff_equals_17_15_l507_50774

theorem power_sum_div_diff_equals_17_15 :
  (2^2020 + 2^2016) / (2^2020 - 2^2016) = 17/15 := by
  sorry

end power_sum_div_diff_equals_17_15_l507_50774


namespace solution_set_abs_inequality_l507_50724

theorem solution_set_abs_inequality :
  {x : ℝ | |2*x - 1| < 1} = Set.Ioo 0 1 := by sorry

end solution_set_abs_inequality_l507_50724


namespace hat_number_sum_l507_50767

/-- Represents a four-digit perfect square number with tens digit 0 and non-zero units digit -/
structure FourDigitPerfectSquare where
  value : Nat
  is_four_digit : value ≥ 1000 ∧ value < 10000
  is_perfect_square : ∃ n, value = n * n
  tens_digit_zero : (value / 10) % 10 = 0
  units_digit_nonzero : value % 10 ≠ 0

/-- The set of all valid FourDigitPerfectSquare numbers -/
def ValidNumbers : Finset FourDigitPerfectSquare := sorry

/-- Predicate to check if two FourDigitPerfectSquare numbers have the same units digit -/
def SameUnitsDigit (a b : FourDigitPerfectSquare) : Prop :=
  a.value % 10 = b.value % 10

/-- Predicate to check if a FourDigitPerfectSquare number has an even units digit -/
def EvenUnitsDigit (a : FourDigitPerfectSquare) : Prop :=
  a.value % 2 = 0

theorem hat_number_sum :
  ∃ (a b c : FourDigitPerfectSquare),
    a ∈ ValidNumbers ∧
    b ∈ ValidNumbers ∧
    c ∈ ValidNumbers ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    SameUnitsDigit b c ∧
    EvenUnitsDigit a ∧
    (∀ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z → SameUnitsDigit y z → x = a ∧ y = b ∧ z = c) ∧
    a.value + b.value + c.value = 14612 :=
  sorry

end hat_number_sum_l507_50767


namespace quadratic_inequality_implies_a_bound_l507_50754

theorem quadratic_inequality_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 := by
  sorry

end quadratic_inequality_implies_a_bound_l507_50754


namespace f_min_at_one_plus_inv_sqrt_three_l507_50780

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 6 * x + 1

-- State the theorem
theorem f_min_at_one_plus_inv_sqrt_three :
  ∃ (x_min : ℝ), x_min = 1 + 1 / Real.sqrt 3 ∧
  ∀ (x : ℝ), f x ≥ f x_min :=
by sorry

end f_min_at_one_plus_inv_sqrt_three_l507_50780


namespace fixed_point_parabola_l507_50720

theorem fixed_point_parabola (k : ℝ) : 9 = 9 * (-1)^2 + k * (-1) - 5 * k := by
  sorry

end fixed_point_parabola_l507_50720


namespace vector_sum_scalar_multiple_l507_50713

/-- Given vectors a and b in ℝ², prove that a + 2b equals the expected result. -/
theorem vector_sum_scalar_multiple (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (-2, 1)) :
  a + 2 • b = (-3, 4) := by
  sorry

end vector_sum_scalar_multiple_l507_50713


namespace trigonometric_identity_l507_50764

theorem trigonometric_identity (x : ℝ) : 
  (4 * Real.sin x ^ 3 * Real.cos (3 * x) + 4 * Real.cos x ^ 3 * Real.sin (3 * x) = 3 * Real.sin (2 * x)) ↔ 
  (∃ n : ℤ, x = π / 6 * (2 * ↑n + 1)) ∨ (∃ k : ℤ, x = π * ↑k) := by
sorry

end trigonometric_identity_l507_50764


namespace dessert_probability_l507_50725

theorem dessert_probability (p_dessert_and_coffee : ℝ) (p_no_coffee_given_dessert : ℝ) :
  p_dessert_and_coffee = 0.6 →
  p_no_coffee_given_dessert = 0.2 →
  1 - (p_dessert_and_coffee / (1 - p_no_coffee_given_dessert)) = 0.4 :=
by sorry

end dessert_probability_l507_50725


namespace inequality_proof_l507_50732

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_eq : a + b + c = a * b + b * c + c * a) : 
  3 + (((a^3 + 1) / 2)^(1/3) + ((b^3 + 1) / 2)^(1/3) + ((c^3 + 1) / 2)^(1/3)) ≤ 2 * (a + b + c) := by
  sorry

end inequality_proof_l507_50732


namespace karlson_candy_theorem_l507_50797

/-- The maximum number of candies Karlson can eat given n initial units -/
def max_candies (n : ℕ) : ℕ := Nat.choose n 2

/-- The theorem stating that for 31 initial units, the maximum number of candies is 465 -/
theorem karlson_candy_theorem :
  max_candies 31 = 465 := by
  sorry

end karlson_candy_theorem_l507_50797


namespace remainder_of_3_power_500_mod_17_l507_50729

theorem remainder_of_3_power_500_mod_17 : 3^500 % 17 = 13 := by
  sorry

end remainder_of_3_power_500_mod_17_l507_50729


namespace min_value_complex_l507_50709

theorem min_value_complex (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (min_val : ℝ), min_val = Real.sqrt (13 + 6 * Real.sqrt 7) ∧
    ∀ w : ℂ, Complex.abs w = 2 → Complex.abs (w + 3 - 4 * Complex.I) ≥ min_val :=
by sorry

end min_value_complex_l507_50709


namespace johns_playing_days_l507_50756

def beats_per_minute : ℕ := 200
def hours_per_day : ℕ := 2
def total_beats : ℕ := 72000

def minutes_per_hour : ℕ := 60
def minutes_per_day : ℕ := hours_per_day * minutes_per_hour
def beats_per_day : ℕ := beats_per_minute * minutes_per_day

theorem johns_playing_days :
  total_beats / beats_per_day = 3 :=
by sorry

end johns_playing_days_l507_50756


namespace minimum_newspapers_to_recover_cost_l507_50770

/-- The cost of Mary's scooter in dollars -/
def scooter_cost : ℕ := 3000

/-- The amount Mary earns per newspaper delivered in dollars -/
def earning_per_newspaper : ℕ := 8

/-- The transportation cost per newspaper delivery in dollars -/
def transport_cost_per_newspaper : ℕ := 4

/-- The net earning per newspaper in dollars -/
def net_earning_per_newspaper : ℕ := earning_per_newspaper - transport_cost_per_newspaper

theorem minimum_newspapers_to_recover_cost :
  ∃ n : ℕ, n * net_earning_per_newspaper ≥ scooter_cost ∧
  ∀ m : ℕ, m * net_earning_per_newspaper ≥ scooter_cost → m ≥ n :=
by sorry

end minimum_newspapers_to_recover_cost_l507_50770


namespace complex_equation_solution_l507_50739

theorem complex_equation_solution :
  let z : ℂ := ((1 + Complex.I)^2 + 3*(1 - Complex.I)) / (2 + Complex.I)
  ∀ a b : ℝ,
  z^2 + a*z + b = 1 + Complex.I →
  a = -3 ∧ b = 4 := by
sorry

end complex_equation_solution_l507_50739


namespace q_zero_at_sqrt2_l507_50784

-- Define the polynomial q
def q (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ : ℝ) (x y : ℝ) : ℝ :=
  b₀ + b₁*x + b₂*y + b₃*x^2 + b₄*x*y + b₅*y^2 + b₆*x^3 + b₈*x*y^2 + b₉*y^3

-- State the theorem
theorem q_zero_at_sqrt2 (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ : ℝ) :
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 0 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 1 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ (-1) 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 0 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 0 (-1) = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 1 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ (-2) 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 3 (-1) = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ (Real.sqrt 2) (Real.sqrt 2) = 0 :=
by
  sorry


end q_zero_at_sqrt2_l507_50784


namespace sachin_lending_rate_l507_50788

/-- Calculates simple interest --/
def simpleInterest (principal time rate : ℚ) : ℚ :=
  principal * rate * time / 100

theorem sachin_lending_rate :
  let borrowed_amount : ℚ := 5000
  let borrowed_time : ℚ := 2
  let borrowed_rate : ℚ := 4
  let sachin_gain_per_year : ℚ := 112.5
  let borrowed_interest := simpleInterest borrowed_amount borrowed_time borrowed_rate
  let total_gain := sachin_gain_per_year * borrowed_time
  let total_interest_from_rahul := borrowed_interest + total_gain
  let rahul_rate := (total_interest_from_rahul * 100) / (borrowed_amount * borrowed_time)
  rahul_rate = 6.25 := by sorry

end sachin_lending_rate_l507_50788


namespace trigonometric_identity_l507_50748

theorem trigonometric_identity : 
  2 * (Real.sin (35 * π / 180) * Real.cos (25 * π / 180) + 
       Real.cos (35 * π / 180) * Real.cos (65 * π / 180)) = Real.sqrt 3 := by
  sorry

end trigonometric_identity_l507_50748


namespace inequality_solution_implies_m_value_l507_50773

theorem inequality_solution_implies_m_value : 
  ∀ m : ℝ, 
  (∀ x : ℝ, 0 < x ∧ x < 2 ↔ -1/2 * x^2 + 2*x > m*x) → 
  m = 1 :=
sorry

end inequality_solution_implies_m_value_l507_50773


namespace quadrilateral_property_l507_50749

theorem quadrilateral_property (α β γ δ : Real) 
  (convex : α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0)
  (sum_angles : α + β + γ + δ = 2 * π)
  (sum_cosines : Real.cos α + Real.cos β + Real.cos γ + Real.cos δ = 0) :
  (α + β = π ∨ γ + δ = π) ∨ (α + γ = β + δ) :=
by sorry

end quadrilateral_property_l507_50749


namespace two_std_dev_below_value_l507_50769

/-- Represents a normal distribution --/
structure NormalDistribution where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation

/-- The value that is exactly 2 standard deviations less than the mean --/
def twoStdDevBelow (nd : NormalDistribution) : ℝ :=
  nd.μ - 2 * nd.σ

theorem two_std_dev_below_value :
  let nd : NormalDistribution := { μ := 14.0, σ := 1.5 }
  twoStdDevBelow nd = 11.0 := by
  sorry

end two_std_dev_below_value_l507_50769


namespace student_average_age_l507_50761

theorem student_average_age 
  (num_students : ℕ) 
  (teacher_age : ℕ) 
  (new_average : ℝ) 
  (h1 : num_students = 10)
  (h2 : teacher_age = 26)
  (h3 : new_average = 16)
  (h4 : (num_students : ℝ) * new_average = (num_students + 1 : ℝ) * new_average - teacher_age) :
  (num_students : ℝ) * new_average - teacher_age = num_students * 15 := by
sorry

end student_average_age_l507_50761


namespace triangle_property_l507_50723

open Real

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def satisfiesCondition (t : Triangle) : Prop :=
  2 * t.a * sin t.A = (2 * t.b + t.c) * sin t.B + (2 * t.c + t.b) * sin t.C

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem triangle_property (t : Triangle) 
  (h : satisfiesCondition t) : 
  t.A = 2 * π / 3 ∧ 
  (t.a = 2 → 4 < perimeter t ∧ perimeter t ≤ 2 + 4 * Real.sqrt 3 / 3) :=
sorry

end triangle_property_l507_50723


namespace base8_175_equals_base10_125_l507_50741

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ :=
  let d₂ := n / 100
  let d₁ := (n / 10) % 10
  let d₀ := n % 10
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

theorem base8_175_equals_base10_125 : base8ToBase10 175 = 125 := by
  sorry

end base8_175_equals_base10_125_l507_50741


namespace scientific_notation_of_1300000_l507_50734

/-- Express 1,300,000 in scientific notation -/
theorem scientific_notation_of_1300000 :
  (1300000 : ℝ) = 1.3 * (10 : ℝ) ^ 6 := by sorry

end scientific_notation_of_1300000_l507_50734


namespace inequality_proof_l507_50746

theorem inequality_proof (a b c : ℝ) 
  (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_c : c ≥ 0)
  (h_ineq_a : a^2 ≤ b^2 + c^2)
  (h_ineq_b : b^2 ≤ c^2 + a^2)
  (h_ineq_c : c^2 ≤ a^2 + b^2) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) ∧
  ((a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) = 4 * (a^6 + b^6 + c^6) ↔ a = b ∧ b = c) :=
by sorry


end inequality_proof_l507_50746


namespace expression_equals_negative_one_l507_50768

theorem expression_equals_negative_one :
  |(-Real.sqrt 2)| + (2016 + Real.pi)^(0 : ℝ) + (-1/2)⁻¹ - 2 * Real.sin (45 * π / 180) = -1 := by
  sorry

end expression_equals_negative_one_l507_50768


namespace four_thirds_of_product_l507_50718

theorem four_thirds_of_product (a b : ℚ) (ha : a = 15/4) (hb : b = 5/2) : 
  (4/3 : ℚ) * (a * b) = 25/2 := by
  sorry

end four_thirds_of_product_l507_50718


namespace ellipse_incircle_area_l507_50779

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)

-- Define collinearity
def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

-- Define the theorem
theorem ellipse_incircle_area (x1 y1 x2 y2 : ℝ) :
  is_on_ellipse x1 y1 →
  is_on_ellipse x2 y2 →
  collinear (x1, y1) (x2, y2) F1 →
  (area_incircle_ABF2 : ℝ) →
  area_incircle_ABF2 = 4 * Real.pi →
  |y1 - y2| = 5 :=
by
  sorry


end ellipse_incircle_area_l507_50779


namespace residue_negative_1237_mod_37_l507_50790

theorem residue_negative_1237_mod_37 : ∃ k : ℤ, -1237 = 37 * k + 21 ∧ (0 ≤ 21 ∧ 21 < 37) := by sorry

end residue_negative_1237_mod_37_l507_50790


namespace exists_quadrilateral_no_triangle_l507_50726

/-- A convex quadrilateral with angles α, β, γ, and δ (in degrees) -/
structure ConvexQuadrilateral where
  α : ℝ
  β : ℝ
  γ : ℝ
  δ : ℝ
  sum_360 : α + β + γ + δ = 360
  all_positive : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < δ
  all_less_180 : α < 180 ∧ β < 180 ∧ γ < 180 ∧ δ < 180

/-- Check if three real numbers can form the sides of a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that there exists a convex quadrilateral where no three angles can form a triangle -/
theorem exists_quadrilateral_no_triangle : ∃ q : ConvexQuadrilateral, 
  ¬(canFormTriangle q.α q.β q.γ ∨ 
    canFormTriangle q.α q.β q.δ ∨ 
    canFormTriangle q.α q.γ q.δ ∨ 
    canFormTriangle q.β q.γ q.δ) := by
  sorry

end exists_quadrilateral_no_triangle_l507_50726


namespace min_max_abs_quadratic_l507_50711

-- Define the function f(x) = x^2 + ax + b
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem min_max_abs_quadratic (a b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f a b x| ≤ 1/2) ↔ a = 0 ∧ b = -1/2 :=
sorry

end min_max_abs_quadratic_l507_50711


namespace number_division_and_addition_l507_50755

theorem number_division_and_addition (x : ℝ) : x / 9 = 8 → x + 11 = 83 := by
  sorry

end number_division_and_addition_l507_50755


namespace friendship_theorem_l507_50757

/-- A graph representing friendships in a city --/
structure FriendshipGraph where
  vertices : Finset ℕ
  edges : Finset (Finset ℕ)
  edge_size : ∀ e ∈ edges, Finset.card e = 2
  vertex_bound : Finset.card vertices = 2000000

/-- Property that every subgraph of 2000 vertices contains a triangle --/
def has_triangle_in_subgraphs (G : FriendshipGraph) : Prop :=
  ∀ S : Finset ℕ, S ⊆ G.vertices → Finset.card S = 2000 →
    ∃ T : Finset ℕ, T ⊆ S ∧ Finset.card T = 3 ∧ T ∈ G.edges

/-- Theorem stating the existence of K₄ in the graph --/
theorem friendship_theorem (G : FriendshipGraph) 
  (h : has_triangle_in_subgraphs G) : 
  ∃ K : Finset ℕ, K ⊆ G.vertices ∧ Finset.card K = 4 ∧ 
    ∀ e : Finset ℕ, e ⊆ K → Finset.card e = 2 → e ∈ G.edges :=
sorry

end friendship_theorem_l507_50757


namespace number_problem_l507_50708

theorem number_problem (x : ℝ) : (x - 14) / 10 = 4 → (x - 5) / 7 = 7 := by
  sorry

end number_problem_l507_50708


namespace sarah_wide_reflections_correct_l507_50742

/-- The number of times Sarah sees her reflection in the room with tall mirrors -/
def sarah_tall_reflections : ℕ := 10

/-- The number of times Ellie sees her reflection in the room with tall mirrors -/
def ellie_tall_reflections : ℕ := 6

/-- The number of times Ellie sees her reflection in the room with wide mirrors -/
def ellie_wide_reflections : ℕ := 3

/-- The number of times they both passed through the room with tall mirrors -/
def tall_mirror_passes : ℕ := 3

/-- The number of times they both passed through the room with wide mirrors -/
def wide_mirror_passes : ℕ := 5

/-- The total number of reflections seen by both Sarah and Ellie -/
def total_reflections : ℕ := 88

/-- The number of times Sarah sees her reflection in the room with wide mirrors -/
def sarah_wide_reflections : ℕ := 5

theorem sarah_wide_reflections_correct :
  sarah_tall_reflections * tall_mirror_passes +
  sarah_wide_reflections * wide_mirror_passes +
  ellie_tall_reflections * tall_mirror_passes +
  ellie_wide_reflections * wide_mirror_passes = total_reflections :=
by sorry

end sarah_wide_reflections_correct_l507_50742


namespace probability_one_triple_one_pair_l507_50731

def num_dice : ℕ := 5
def faces_per_die : ℕ := 6

def favorable_outcomes : ℕ := faces_per_die * (num_dice.choose 3) * (faces_per_die - 1) * 1

def total_outcomes : ℕ := faces_per_die ^ num_dice

theorem probability_one_triple_one_pair :
  (favorable_outcomes : ℚ) / total_outcomes = 25 / 648 := by
  sorry

end probability_one_triple_one_pair_l507_50731


namespace chicken_salad_cost_l507_50706

/-- Given the following conditions:
  - Lee and his friend had a total of $18
  - Chicken wings cost $6
  - Two sodas cost $2 in total
  - The tax was $3
  - They received $3 in change
  Prove that the chicken salad cost $4 -/
theorem chicken_salad_cost (total_money : ℕ) (wings_cost : ℕ) (sodas_cost : ℕ) (tax : ℕ) (change : ℕ) :
  total_money = 18 →
  wings_cost = 6 →
  sodas_cost = 2 →
  tax = 3 →
  change = 3 →
  total_money - change - (wings_cost + sodas_cost + tax) = 4 :=
by sorry

end chicken_salad_cost_l507_50706


namespace arithmetic_sequence_sum_l507_50794

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 5th through 7th terms of an arithmetic sequence. -/
def SumFifthToSeventh (a : ℕ → ℤ) : ℤ := a 5 + a 6 + a 7

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  ArithmeticSequence a →
  a 8 = 16 →
  a 9 = 22 →
  a 10 = 28 →
  SumFifthToSeventh a = 12 := by
  sorry

end arithmetic_sequence_sum_l507_50794


namespace number_division_addition_l507_50786

theorem number_division_addition : ∃ x : ℝ, 7500 + x / 50 = 7525 := by
  sorry

end number_division_addition_l507_50786


namespace sqrt_product_l507_50727

theorem sqrt_product (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b := by
  sorry

end sqrt_product_l507_50727


namespace percentage_calculation_l507_50776

theorem percentage_calculation (A B : ℝ) (x : ℝ) 
  (h1 : A - B = 1670)
  (h2 : A = 2505)
  (h3 : (7.5 / 100) * A = (x / 100) * B) :
  x = 22.5 := by
  sorry

end percentage_calculation_l507_50776


namespace total_cost_pants_and_belt_l507_50707

theorem total_cost_pants_and_belt (pants_price belt_price total_cost : ℝ) :
  pants_price = 34 →
  pants_price = belt_price - 2.93 →
  total_cost = pants_price + belt_price →
  total_cost = 70.93 := by
sorry

end total_cost_pants_and_belt_l507_50707


namespace convex_quadrilateral_from_circles_in_square_l507_50717

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a square -/
structure Square where
  sideLength : ℝ

/-- Predicate to check if a point is inside a circle -/
def isInsideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

/-- Theorem statement -/
theorem convex_quadrilateral_from_circles_in_square 
  (s : Square) 
  (c1 c2 c3 c4 : Circle) 
  (p1 p2 p3 p4 : Point) : 
  -- The circles are centered at the vertices of the square
  (c1.center.x = 0 ∧ c1.center.y = 0) →
  (c2.center.x = s.sideLength ∧ c2.center.y = 0) →
  (c3.center.x = s.sideLength ∧ c3.center.y = s.sideLength) →
  (c4.center.x = 0 ∧ c4.center.y = s.sideLength) →
  -- The sum of the areas of the circles equals the area of the square
  (π * (c1.radius^2 + c2.radius^2 + c3.radius^2 + c4.radius^2) = s.sideLength^2) →
  -- The points are inside their respective circles
  isInsideCircle p1 c1 →
  isInsideCircle p2 c2 →
  isInsideCircle p3 c3 →
  isInsideCircle p4 c4 →
  -- The four points form a convex quadrilateral
  ∃ (a b c : ℝ), a * p1.x + b * p1.y + c < 0 ∧
                 a * p2.x + b * p2.y + c < 0 ∧
                 a * p3.x + b * p3.y + c > 0 ∧
                 a * p4.x + b * p4.y + c > 0 :=
by sorry


end convex_quadrilateral_from_circles_in_square_l507_50717


namespace rainfall_ratio_l507_50758

theorem rainfall_ratio (total_rainfall : ℝ) (second_week_rainfall : ℝ) 
  (h1 : total_rainfall = 40)
  (h2 : second_week_rainfall = 24) :
  (second_week_rainfall) / (total_rainfall - second_week_rainfall) = 3 / 2 := by
  sorry

end rainfall_ratio_l507_50758


namespace bake_sale_group_composition_l507_50736

theorem bake_sale_group_composition (p : ℕ) : 
  (p : ℚ) > 0 →
  (p / 2 : ℚ) / p = 1 / 2 →
  ((p / 2 - 3) : ℚ) / p = 2 / 5 →
  p / 2 = 15 := by
sorry

end bake_sale_group_composition_l507_50736


namespace cryptarithmetic_puzzle_l507_50793

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

theorem cryptarithmetic_puzzle :
  ∀ (I X E L V : ℕ),
    (I < 10) →
    (X < 10) →
    (E < 10) →
    (L < 10) →
    (V < 10) →
    (is_odd X) →
    (I ≠ 7) →
    (X ≠ 7) →
    (E ≠ 7) →
    (L ≠ 7) →
    (V ≠ 7) →
    (I ≠ X) →
    (I ≠ E) →
    (I ≠ L) →
    (I ≠ V) →
    (X ≠ E) →
    (X ≠ L) →
    (X ≠ V) →
    (E ≠ L) →
    (E ≠ V) →
    (L ≠ V) →
    (700 + 10*I + X + 700 + 10*I + X = 1000*E + 100*L + 10*E + V) →
    (I = 2) :=
by sorry

end cryptarithmetic_puzzle_l507_50793


namespace cloth_gain_proof_l507_50733

/-- 
Given:
- A shop owner sells 40 meters of cloth
- The gain percentage is 33.33333333333333%

Prove that the gain is equivalent to the selling price of 10 meters of cloth
-/
theorem cloth_gain_proof (total_meters : ℝ) (gain_percentage : ℝ) 
  (h1 : total_meters = 40)
  (h2 : gain_percentage = 33.33333333333333) :
  (gain_percentage / 100 * total_meters) / (1 + gain_percentage / 100) = 10 := by
  sorry

end cloth_gain_proof_l507_50733


namespace invalid_votes_percentage_l507_50763

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (winner_percentage : ℚ)
  (loser_votes : ℕ)
  (h1 : total_votes = 7500)
  (h2 : winner_percentage = 55 / 100)
  (h3 : loser_votes = 2700) :
  (total_votes - (loser_votes / (1 - winner_percentage))) / total_votes = 1 / 5 :=
sorry

end invalid_votes_percentage_l507_50763


namespace exists_cut_sequence_for_1003_l507_50787

/-- Represents the number of pieces selected for cutting at each step -/
def CutSequence := List Nat

/-- Calculates the number of pieces after a sequence of cuts -/
def numPieces (cuts : CutSequence) : Nat :=
  3 * (cuts.sum + 1) + 1

/-- Theorem: It's possible to obtain 1003 pieces through the cutting process -/
theorem exists_cut_sequence_for_1003 : ∃ (cuts : CutSequence), numPieces cuts = 1003 := by
  sorry

end exists_cut_sequence_for_1003_l507_50787


namespace contrapositive_equivalence_l507_50798

theorem contrapositive_equivalence (p q : Prop) :
  (p → ¬q) ↔ (q → ¬p) := by sorry

end contrapositive_equivalence_l507_50798


namespace second_project_grade_l507_50738

/-- Represents the grading system for a computer programming course project. -/
structure ProjectGrade where
  /-- The proportion of influence from time spent on the project. -/
  timeProportion : ℝ
  /-- The proportion of influence from effort spent on the project. -/
  effortProportion : ℝ
  /-- Calculates the influence score based on time and effort. -/
  influenceScore : ℝ → ℝ → ℝ
  /-- The proportionality constant between influence score and grade. -/
  gradeProportionality : ℝ

/-- Theorem stating the grade for the second project given the conditions. -/
theorem second_project_grade (pg : ProjectGrade)
  (h1 : pg.timeProportion = 0.70)
  (h2 : pg.effortProportion = 0.30)
  (h3 : pg.influenceScore t e = pg.timeProportion * t + pg.effortProportion * e)
  (h4 : pg.gradeProportionality = 84 / (pg.influenceScore 5 70))
  : pg.gradeProportionality * (pg.influenceScore 6 80) = 96.49 := by
  sorry

#check second_project_grade

end second_project_grade_l507_50738


namespace jump_rope_time_difference_l507_50722

/-- Given jump rope times for Cindy, Betsy, and Tina, prove that Tina can jump 6 minutes longer than Cindy -/
theorem jump_rope_time_difference (cindy betsy tina : ℕ) 
  (h1 : cindy = 12)
  (h2 : betsy = cindy / 2)
  (h3 : tina = 3 * betsy) :
  tina - cindy = 6 := by
  sorry

end jump_rope_time_difference_l507_50722
