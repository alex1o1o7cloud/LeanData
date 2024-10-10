import Mathlib

namespace at_least_two_primes_in_base_n_1002_l278_27834

def base_n_1002 (n : ℕ) : ℕ := n^3 + 2

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

theorem at_least_two_primes_in_base_n_1002 : 
  ∃ n1 n2 : ℕ, n1 ≥ 2 ∧ n2 ≥ 2 ∧ n1 ≠ n2 ∧ 
  is_prime (base_n_1002 n1) ∧ is_prime (base_n_1002 n2) := by
  sorry

end at_least_two_primes_in_base_n_1002_l278_27834


namespace root_sum_reciprocals_l278_27852

-- Define the polynomial
def f (x : ℂ) : ℂ := x^4 + 10*x^3 + 20*x^2 + 15*x + 6

-- Define the roots
def p : ℂ := sorry
def q : ℂ := sorry
def r : ℂ := sorry
def s : ℂ := sorry

-- State the theorem
theorem root_sum_reciprocals :
  f p = 0 ∧ f q = 0 ∧ f r = 0 ∧ f s = 0 →
  1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s) = 10 / 3 :=
by sorry

end root_sum_reciprocals_l278_27852


namespace weight_of_oranges_l278_27811

/-- Proves that the weight of oranges is 1 kilogram, given the total weight of fruits
    and the weights of apples, grapes, and strawberries. -/
theorem weight_of_oranges (total_weight apple_weight grape_weight strawberry_weight : ℕ)
  (h_total : total_weight = 10)
  (h_apple : apple_weight = 3)
  (h_grape : grape_weight = 3)
  (h_strawberry : strawberry_weight = 3) :
  total_weight - (apple_weight + grape_weight + strawberry_weight) = 1 := by
  sorry

#check weight_of_oranges

end weight_of_oranges_l278_27811


namespace article_count_l278_27881

theorem article_count (x : ℕ) (cost_price selling_price : ℝ) : 
  (cost_price * x = selling_price * 16) →
  (selling_price = 1.5 * cost_price) →
  x = 24 := by
sorry

end article_count_l278_27881


namespace absolute_value_equation_solution_l278_27840

theorem absolute_value_equation_solution (x : ℝ) : 
  |2*x - 1| + |x - 2| = |x + 1| ↔ 1/2 ≤ x ∧ x ≤ 2 := by sorry

end absolute_value_equation_solution_l278_27840


namespace average_marks_l278_27871

structure Marks where
  physics : ℕ
  chemistry : ℕ
  mathematics : ℕ
  biology : ℕ
  english : ℕ
  history : ℕ
  geography : ℕ

def valid_marks (m : Marks) : Prop :=
  m.chemistry = m.physics + 75 ∧
  m.mathematics = m.chemistry + 30 ∧
  m.biology = m.physics - 15 ∧
  m.english = m.biology - 10 ∧
  m.history = m.biology - 10 ∧
  m.geography = m.biology - 10 ∧
  m.physics + m.chemistry + m.mathematics + m.biology + m.english + m.history + m.geography = m.physics + 520 ∧
  m.physics ≥ 40 ∧ m.chemistry ≥ 40 ∧ m.mathematics ≥ 40 ∧ m.biology ≥ 40 ∧
  m.english ≥ 40 ∧ m.history ≥ 40 ∧ m.geography ≥ 40

theorem average_marks (m : Marks) (h : valid_marks m) :
  (m.mathematics + m.biology + m.history + m.geography) / 4 = 82 := by
  sorry

end average_marks_l278_27871


namespace octahedron_volume_with_unit_inscribed_sphere_l278_27856

/-- An octahedron is a polyhedron with 8 equilateral triangular faces. -/
structure Octahedron where
  -- We don't need to define the full structure, just what we need for this problem
  volume : ℝ

/-- A sphere is a three-dimensional geometric object. -/
structure Sphere where
  radius : ℝ

/-- An octahedron with an inscribed sphere. -/
structure OctahedronWithInscribedSphere where
  octahedron : Octahedron
  sphere : Sphere
  inscribed : sphere.radius = 1  -- The sphere is inscribed and has radius 1

/-- The volume of an octahedron with an inscribed sphere of radius 1 is √6. -/
theorem octahedron_volume_with_unit_inscribed_sphere
  (o : OctahedronWithInscribedSphere) :
  o.octahedron.volume = Real.sqrt 6 := by
  sorry

end octahedron_volume_with_unit_inscribed_sphere_l278_27856


namespace special_triangle_line_BC_l278_27884

/-- A triangle ABC with vertex A at (-4, 2) and two medians on specific lines -/
structure SpecialTriangle where
  /-- Vertex A of the triangle -/
  A : ℝ × ℝ
  /-- The line containing one median -/
  median1 : ℝ → ℝ → ℝ
  /-- The line containing another median -/
  median2 : ℝ → ℝ → ℝ
  /-- Condition: A is at (-4, 2) -/
  h_A : A = (-4, 2)
  /-- Condition: One median lies on 3x - 2y + 2 = 0 -/
  h_median1 : median1 x y = 3*x - 2*y + 2
  /-- Condition: Another median lies on 3x + 5y - 12 = 0 -/
  h_median2 : median2 x y = 3*x + 5*y - 12

/-- The equation of line BC in the special triangle -/
def lineBCEq (t : SpecialTriangle) (x y : ℝ) : ℝ := 2*x + y - 8

/-- Theorem: The equation of line BC in the special triangle is 2x + y - 8 = 0 -/
theorem special_triangle_line_BC (t : SpecialTriangle) :
  ∀ x y, lineBCEq t x y = 0 ↔ y = -2*x + 8 :=
sorry

end special_triangle_line_BC_l278_27884


namespace green_home_construction_l278_27870

theorem green_home_construction (x : ℝ) (h : x > 50) : (300 : ℝ) / (x - 50) = 400 / x := by
  sorry

end green_home_construction_l278_27870


namespace union_A_complement_B_equals_interval_l278_27833

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- State the theorem
theorem union_A_complement_B_equals_interval :
  A ∪ (U \ B) = Set.Iio 2 := by sorry

end union_A_complement_B_equals_interval_l278_27833


namespace floor_add_two_floor_sum_inequality_floor_square_inequality_exists_l278_27880

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem 1
theorem floor_add_two (x : ℝ) : floor (x + 2) = floor x + 2 := by sorry

-- Theorem 2
theorem floor_sum_inequality (x y : ℝ) : floor (x + y) ≤ floor x + floor y := by sorry

-- Theorem 3
theorem floor_square_inequality_exists :
  ∃ x : ℝ, floor (x^2) ≠ (floor x)^2 := by sorry

end floor_add_two_floor_sum_inequality_floor_square_inequality_exists_l278_27880


namespace sin_90_degrees_l278_27848

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by sorry

end sin_90_degrees_l278_27848


namespace scooter_final_price_l278_27851

/-- The final sale price of a scooter after two consecutive discounts -/
theorem scooter_final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  initial_price = 150 ∧ discount1 = 0.4 ∧ discount2 = 0.35 →
  initial_price * (1 - discount1) * (1 - discount2) = 58.50 := by
sorry

end scooter_final_price_l278_27851


namespace ant_path_distance_l278_27873

/-- Represents the rectangle in which the ant walks --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the ant's path --/
structure AntPath where
  start : ℝ  -- Distance from the nearest corner to the starting point X
  angle : ℝ  -- Angle of the path with respect to the sides of the rectangle

/-- Theorem stating the conditions and the result to be proved --/
theorem ant_path_distance (rect : Rectangle) (path : AntPath) :
  rect.width = 18 ∧ 
  rect.height = 150 ∧ 
  path.angle = 45 ∧ 
  path.start ≥ 0 ∧ 
  path.start ≤ rect.width ∧
  (∃ n : ℕ, n * rect.width = rect.height / 2) →
  path.start = 3 := by
  sorry

end ant_path_distance_l278_27873


namespace total_meows_eq_286_l278_27837

/-- The number of meows for eight cats over 12 minutes -/
def total_meows : ℕ :=
  let cat1_meows := 3 * 12
  let cat2_meows := (3 * 2) * 12
  let cat3_meows := ((3 * 2) / 3) * 12
  let cat4_meows := 4 * 12
  let cat5_meows := (60 / 45) * 12
  let cat6_meows := (5 / 2) * 12
  let cat7_meows := ((3 * 2) / 2) * 12
  let cat8_meows := (6 / 3) * 12
  cat1_meows + cat2_meows + cat3_meows + cat4_meows + 
  cat5_meows + cat6_meows + cat7_meows + cat8_meows

theorem total_meows_eq_286 : total_meows = 286 := by
  sorry

#eval total_meows

end total_meows_eq_286_l278_27837


namespace log_equation_solution_l278_27804

-- Define the equation
def log_equation (x : ℝ) : Prop :=
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 4 = 15

-- State the theorem
theorem log_equation_solution :
  ∀ x : ℝ, x > 0 → log_equation x → x = 2^(90/11) :=
by sorry

end log_equation_solution_l278_27804


namespace birth_probability_l278_27853

-- Define the number of children
def n : ℕ := 5

-- Define the probability of a child being a boy or a girl
def p : ℚ := 1/2

-- Define the probability of all children being the same gender
def prob_all_same : ℚ := p^n

-- Define the probability of having 3 of one gender and 2 of the other
def prob_three_two : ℚ := Nat.choose n 3 * p^n

-- Define the probability of having 4 of one gender and 1 of the other
def prob_four_one : ℚ := 2 * Nat.choose n 1 * p^n

theorem birth_probability :
  prob_three_two > prob_all_same ∧
  prob_four_one > prob_all_same ∧
  prob_three_two = prob_four_one :=
by sorry

end birth_probability_l278_27853


namespace folded_rectangle_area_l278_27825

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Represents the folding scenario -/
structure FoldedRectangle where
  original : Rectangle
  t : Point
  u : Point
  qPrime : Point
  pPrime : Point

theorem folded_rectangle_area (fold : FoldedRectangle) 
  (h1 : fold.t.x - fold.original.topLeft.x < fold.original.bottomRight.x - fold.u.x)
  (h2 : (fold.pPrime.x - fold.qPrime.x)^2 + (fold.pPrime.y - fold.qPrime.y)^2 = 
        (fold.original.bottomRight.y - fold.original.topLeft.y)^2)
  (h3 : fold.qPrime.x - fold.original.topLeft.x = 8)
  (h4 : fold.t.x - fold.original.topLeft.x = 36) :
  (fold.original.bottomRight.x - fold.original.topLeft.x) * 
  (fold.original.bottomRight.y - fold.original.topLeft.y) = 288 * Real.sqrt 2 := by
  sorry

end folded_rectangle_area_l278_27825


namespace cost_of_45_daffodils_l278_27865

/-- The cost of a bouquet is directly proportional to the number of daffodils it contains. -/
structure DaffodilBouquet where
  daffodils : ℕ
  cost : ℝ

/-- A bouquet of 15 daffodils costs $25. -/
def standard_bouquet : DaffodilBouquet := ⟨15, 25⟩

/-- The proposition that the cost of a 45-daffodil bouquet is $75. -/
theorem cost_of_45_daffodils : 
  ∀ (b : DaffodilBouquet), b.daffodils = 45 → 
  (b.cost / b.daffodils : ℝ) = (standard_bouquet.cost / standard_bouquet.daffodils) → 
  b.cost = 75 := by
sorry

end cost_of_45_daffodils_l278_27865


namespace petes_total_miles_l278_27843

/-- Represents a pedometer with a maximum step count before resetting --/
structure Pedometer where
  max_steps : ℕ
  resets : ℕ
  final_reading : ℕ

/-- Calculates the total miles walked given a pedometer and steps per mile --/
def total_miles_walked (p : Pedometer) (steps_per_mile : ℕ) : ℚ :=
  ((p.resets * (p.max_steps + 1) + p.final_reading) : ℚ) / steps_per_mile

/-- Theorem stating that Pete walked 2512.5 miles given the problem conditions --/
theorem petes_total_miles :
  let p : Pedometer := ⟨99999, 50, 25000⟩
  let steps_per_mile : ℕ := 2000
  total_miles_walked p steps_per_mile = 2512.5 := by
  sorry


end petes_total_miles_l278_27843


namespace range_of_a_l278_27854

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, Monotone (fun x => (3 - 2*a)^x)

-- Define the theorem
theorem range_of_a (a : ℝ) : p a ∧ ¬(q a) → a ∈ Set.Icc 1 2 := by
  sorry

end range_of_a_l278_27854


namespace product_not_always_minimized_when_closest_l278_27807

theorem product_not_always_minimized_when_closest (d : ℝ) (h : d > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x - y = d ∧
  ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ x' - y' = d ∧
  abs (x' - y') < abs (x - y) ∧ x' * y' < x * y :=
by sorry

-- Other statements (A, B, D, E) are correct, but we don't need to prove them for this task

end product_not_always_minimized_when_closest_l278_27807


namespace trapezoid_existence_l278_27893

/-- Represents a trapezoid ABCD with AB parallel to CD -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Checks if the given points form a valid trapezoid -/
def is_valid_trapezoid (t : Trapezoid) : Prop := sorry

/-- Calculates the perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := sorry

/-- Calculates the length of diagonal AC -/
def diagonal_ac (t : Trapezoid) : ℝ := sorry

/-- Calculates the angle DAB -/
def angle_dab (t : Trapezoid) : ℝ := sorry

/-- Calculates the angle ABC -/
def angle_abc (t : Trapezoid) : ℝ := sorry

/-- Theorem: Given a perimeter k, diagonal e, and angles α and β,
    there exists 0, 1, or 2 trapezoids satisfying these conditions -/
theorem trapezoid_existence (k e α β : ℝ) :
  ∃ n : Fin 3, ∃ ts : Finset Trapezoid,
    ts.card = n ∧
    ∀ t ∈ ts, is_valid_trapezoid t ∧
               perimeter t = k ∧
               diagonal_ac t = e ∧
               angle_dab t = α ∧
               angle_abc t = β := by
  sorry


end trapezoid_existence_l278_27893


namespace sin_2023_closest_to_neg_sqrt2_over_2_l278_27892

-- Define the set of options
def options : Set ℝ := {1/2, Real.sqrt 2 / 2, -1/2, -Real.sqrt 2 / 2}

-- Define the sine function with period 360°
noncomputable def periodic_sin (x : ℝ) : ℝ := Real.sin (2 * Real.pi * (x / 360))

-- State the theorem
theorem sin_2023_closest_to_neg_sqrt2_over_2 :
  ∃ (y : ℝ), y ∈ options ∧ 
  ∀ (z : ℝ), z ∈ options → |periodic_sin 2023 - y| ≤ |periodic_sin 2023 - z| :=
sorry

end sin_2023_closest_to_neg_sqrt2_over_2_l278_27892


namespace walking_problem_l278_27836

/-- The walking problem theorem -/
theorem walking_problem (total_distance : ℝ) (yolanda_rate : ℝ) (bob_distance : ℝ) : 
  total_distance = 24 →
  yolanda_rate = 3 →
  bob_distance = 12 →
  ∃ (bob_rate : ℝ), bob_rate = 12 ∧ bob_distance = bob_rate * 1 := by
  sorry

end walking_problem_l278_27836


namespace conic_is_ellipse_l278_27890

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 + 4*y^2 - 6*x + 8*y + 9 = 0

/-- Definition of an ellipse in standard form -/
def is_ellipse (h k a b : ℝ) (x y : ℝ) : Prop :=
  ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1

/-- Theorem stating that the given equation represents an ellipse -/
theorem conic_is_ellipse :
  ∃ h k a b : ℝ, ∀ x y : ℝ, conic_equation x y ↔ is_ellipse h k a b x y :=
sorry

end conic_is_ellipse_l278_27890


namespace area_of_triangle_with_given_conditions_l278_27882

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C : ℝ × ℝ)

structure TriangleWithPoint extends Triangle :=
  (P : ℝ × ℝ)

-- Define the conditions
def isScaleneRightTriangle (t : Triangle) : Prop := sorry

def isPointOnHypotenuse (t : TriangleWithPoint) : Prop := sorry

def angleABP30 (t : TriangleWithPoint) : Prop := sorry

def lengthAP3 (t : TriangleWithPoint) : Prop := sorry

def lengthCP1 (t : TriangleWithPoint) : Prop := sorry

-- Define the area function
def triangleArea (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_with_given_conditions (t : TriangleWithPoint) 
  (h1 : isScaleneRightTriangle t.toTriangle)
  (h2 : isPointOnHypotenuse t)
  (h3 : angleABP30 t)
  (h4 : lengthAP3 t)
  (h5 : lengthCP1 t) :
  triangleArea t.toTriangle = 12/5 := by sorry

end area_of_triangle_with_given_conditions_l278_27882


namespace min_value_on_negative_interval_l278_27819

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function F defined in terms of f and g -/
def F (f g : ℝ → ℝ) (a b : ℝ) (x : ℝ) : ℝ := a * f x + b * g x + 3

theorem min_value_on_negative_interval
  (f g : ℝ → ℝ) (a b : ℝ)
  (hf : IsOdd f) (hg : IsOdd g)
  (hmax : ∀ x > 0, F f g a b x ≤ 10) :
  ∀ x < 0, F f g a b x ≥ -4 :=
sorry

end min_value_on_negative_interval_l278_27819


namespace intersection_point_l278_27863

/-- The line equation -/
def line_equation (x y z : ℝ) : Prop :=
  (x - 1) / 2 = (y - 1) / (-1) ∧ (y - 1) / (-1) = (z + 2) / 3

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  4 * x + 2 * y - z - 11 = 0

/-- The theorem stating that (3, 0, 1) is the unique intersection point -/
theorem intersection_point :
  ∃! (x y z : ℝ), line_equation x y z ∧ plane_equation x y z ∧ x = 3 ∧ y = 0 ∧ z = 1 := by
  sorry

end intersection_point_l278_27863


namespace quadrilateral_area_l278_27810

/-- Represents a partitioned triangle with four regions -/
structure PartitionedTriangle where
  /-- Area of the first triangle -/
  area1 : ℝ
  /-- Area of the second triangle -/
  area2 : ℝ
  /-- Area of the third triangle -/
  area3 : ℝ
  /-- Area of the fourth triangle -/
  area4 : ℝ
  /-- Area of the quadrilateral -/
  areaQuad : ℝ

/-- Theorem stating that given the areas of the four triangles, 
    the area of the quadrilateral is 18 -/
theorem quadrilateral_area 
  (t : PartitionedTriangle) 
  (h1 : t.area1 = 5) 
  (h2 : t.area2 = 9) 
  (h3 : t.area3 = 24/5) 
  (h4 : t.area4 = 9) : 
  t.areaQuad = 18 := by
  sorry


end quadrilateral_area_l278_27810


namespace geometric_sequence_sum_l278_27897

/-- Given a geometric sequence {aₙ} with positive terms where a₁a₅ + 2a₃a₅ + a₃a₇ = 25, prove that a₃ + a₅ = 5. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) 
    (h2 : ∃ r : ℝ, ∀ n, a (n + 1) = r * a n) 
    (h3 : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25) : 
  a 3 + a 5 = 5 := by
sorry

end geometric_sequence_sum_l278_27897


namespace derivative_log_base_3_l278_27860

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem derivative_log_base_3 (x : ℝ) (h : x > 0) :
  deriv f x = 1 / (x * Real.log 3) :=
by sorry

end derivative_log_base_3_l278_27860


namespace hidden_dots_count_l278_27801

/-- Represents a standard six-sided die -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The sum of all numbers on a standard die -/
def SumOfDie : ℕ := Finset.sum StandardDie id

/-- The number of dice in the stack -/
def NumberOfDice : ℕ := 4

/-- The visible numbers on the dice -/
def VisibleNumbers : Finset ℕ := {1, 2, 2, 3, 4, 4, 5, 6, 6}

/-- The theorem stating the number of hidden dots -/
theorem hidden_dots_count : 
  NumberOfDice * SumOfDie - Finset.sum VisibleNumbers id = 51 := by sorry

end hidden_dots_count_l278_27801


namespace geese_left_is_10_l278_27841

/-- The number of geese that left the duck park -/
def geese_left : ℕ := by sorry

theorem geese_left_is_10 :
  let initial_ducks : ℕ := 25
  let initial_geese : ℕ := 2 * initial_ducks - 10
  let final_ducks : ℕ := initial_ducks + 4
  let final_geese : ℕ := initial_geese - geese_left
  (final_geese = final_ducks + 1) →
  geese_left = 10 := by sorry

end geese_left_is_10_l278_27841


namespace intersection_A_complement_B_l278_27872

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 3}
def B : Set ℝ := {x | x < -2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | -2 ≤ x ∧ x < 3} := by
  sorry

end intersection_A_complement_B_l278_27872


namespace inverse_variation_problem_l278_27831

theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_inverse : ∃ k : ℝ, ∀ x y, x^2 * y = k) 
  (h_initial : 3^2 * 8 = 9 * 8) 
  (h_final : y = 648) : x = 1/3 := by
sorry

end inverse_variation_problem_l278_27831


namespace parallel_lines_condition_l278_27885

/-- Given two lines l₁ and l₂ defined by linear equations with parameter m,
    prove that m = -2 is a sufficient but not necessary condition for l₁ // l₂ -/
theorem parallel_lines_condition (m : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | (m - 4) * x - (2 * m + 4) * y + 2 * m - 4 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | (m - 1) * x + (m + 2) * y + 1 = 0}
  (m = -2 → l₁ = l₂) ∧ ¬(l₁ = l₂ → m = -2) :=
by sorry

end parallel_lines_condition_l278_27885


namespace lcm_of_20_45_60_l278_27849

theorem lcm_of_20_45_60 : Nat.lcm (Nat.lcm 20 45) 60 = 180 := by
  sorry

end lcm_of_20_45_60_l278_27849


namespace square_area_ratio_l278_27850

theorem square_area_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := (s₂ * Real.sqrt 2) / 2
  (s₁^2) / (s₂^2) = 1/2 := by
sorry

end square_area_ratio_l278_27850


namespace min_values_theorem_l278_27820

theorem min_values_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2 * b = 1) :
  (∀ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 → 1 / x + 2 / y ≥ 9) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 ∧ 1 / x + 2 / y = 9) ∧
  (∀ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 → 2^x + 4^y ≥ 2 * Real.sqrt 2) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 ∧ 2^x + 4^y = 2 * Real.sqrt 2) :=
by sorry

end min_values_theorem_l278_27820


namespace train_speed_problem_l278_27815

/-- Proves that the initial speed of a train is 110 km/h given specific journey conditions -/
theorem train_speed_problem (T : ℝ) : ∃ v : ℝ,
  v > 0 ∧
  v - 50 > 0 ∧
  T > 0 ∧
  T + 2/3 = 212/v + 88/(v - 50) ∧
  v = 110 := by
  sorry

end train_speed_problem_l278_27815


namespace notebook_cost_l278_27896

theorem notebook_cost (total_students : Nat) (buyers : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat) :
  total_students = 42 →
  buyers > total_students / 2 →
  notebooks_per_student.Prime →
  cost_per_notebook > notebooks_per_student →
  buyers * notebooks_per_student * cost_per_notebook = 2310 →
  cost_per_notebook = 22 := by
  sorry

end notebook_cost_l278_27896


namespace contrapositive_proof_l278_27864

theorem contrapositive_proof (a : ℝ) : 
  a < 1 → ∀ x : ℝ, x^2 + (2*a+1)*x + a^2 + 2 > 0 := by
  sorry

end contrapositive_proof_l278_27864


namespace percent_of_y_l278_27802

theorem percent_of_y (y : ℝ) (h : y > 0) : ((9 * y) / 20 + (3 * y) / 10) / y = 3 / 4 := by
  sorry

end percent_of_y_l278_27802


namespace product_of_solutions_l278_27859

theorem product_of_solutions (x₁ x₂ : ℚ) : 
  (|6 * x₁ + 2| + 5 = 47) → 
  (|6 * x₂ + 2| + 5 = 47) → 
  x₁ ≠ x₂ → 
  x₁ * x₂ = -440 / 9 := by
sorry

end product_of_solutions_l278_27859


namespace inequality_solution_l278_27861

theorem inequality_solution (x : ℝ) :
  (3*x + 4 ≠ 0) →
  (3 - 2 / (3*x + 4) < 5 ↔ x ∈ Set.Ioo (-5/3) (-4/3) ∪ Set.Ioi (-4/3)) :=
by sorry

end inequality_solution_l278_27861


namespace tan_276_equals_96_l278_27839

theorem tan_276_equals_96 : 
  ∃! (n : ℤ), -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (276 * π / 180) :=
by
  sorry

end tan_276_equals_96_l278_27839


namespace convex_polyhedron_properties_l278_27898

/-- A convex polyhedron with congruent isosceles triangular faces -/
structure ConvexPolyhedron where
  vertices : ℕ
  faces : ℕ
  edges : ℕ
  isConvex : Bool
  hasCongruentIsoscelesFaces : Bool
  formGeometricSequence : Bool

/-- Euler's formula for polyhedra -/
axiom euler_formula {p : ConvexPolyhedron} : p.vertices + p.faces = p.edges + 2

/-- Relation between faces and edges in a polyhedron with triangular faces -/
axiom triangular_faces_relation {p : ConvexPolyhedron} : 2 * p.edges = 3 * p.faces

/-- Geometric sequence property -/
axiom geometric_sequence {p : ConvexPolyhedron} (h : p.formGeometricSequence) :
  p.faces / p.vertices = p.edges / p.faces

/-- Main theorem: A convex polyhedron with the given properties has 8 vertices, 12 faces, and 18 edges -/
theorem convex_polyhedron_properties (p : ConvexPolyhedron)
  (h1 : p.isConvex)
  (h2 : p.hasCongruentIsoscelesFaces)
  (h3 : p.formGeometricSequence) :
  p.vertices = 8 ∧ p.faces = 12 ∧ p.edges = 18 := by
  sorry

end convex_polyhedron_properties_l278_27898


namespace quadratic_square_of_binomial_l278_27844

theorem quadratic_square_of_binomial (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 - 24*x + c = (a*x + b)^2) → c = 16 := by
sorry

end quadratic_square_of_binomial_l278_27844


namespace sara_movie_expenses_l278_27875

/-- The total amount Sara spent on movies -/
def total_spent (ticket_price : ℚ) (num_tickets : ℕ) (rental_cost : ℚ) (purchase_cost : ℚ) : ℚ :=
  ticket_price * num_tickets + rental_cost + purchase_cost

/-- Proof that Sara spent $36.78 on movies -/
theorem sara_movie_expenses : 
  total_spent 10.62 2 1.59 13.95 = 36.78 := by
  sorry

end sara_movie_expenses_l278_27875


namespace three_cubes_volume_l278_27877

-- Define the volume of a cube
def cubeVolume (edge : ℝ) : ℝ := edge ^ 3

-- Define the total volume of three cubes
def totalVolume (edge1 edge2 edge3 : ℝ) : ℝ :=
  cubeVolume edge1 + cubeVolume edge2 + cubeVolume edge3

-- Theorem statement
theorem three_cubes_volume :
  totalVolume 3 5 6 = 368 := by
  sorry

end three_cubes_volume_l278_27877


namespace train_journey_time_l278_27868

/-- Proves that the current time taken to cover a distance is 50 minutes 
    given the conditions from the train problem. -/
theorem train_journey_time : 
  ∀ (distance : ℝ) (current_time : ℝ),
    distance > 0 →
    distance = 48 * (current_time / 60) →
    distance = 60 * (40 / 60) →
    current_time = 50 := by
  sorry

end train_journey_time_l278_27868


namespace area_probability_l278_27883

/-- A square in a 2D plane -/
structure Square :=
  (A B C D : ℝ × ℝ)

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Predicate to check if a point is inside a square -/
def isInside (s : Square) (p : Point) : Prop := sorry

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- The probability of an event occurring when a point is chosen randomly inside a square -/
def probability (s : Square) (event : Point → Prop) : ℝ := sorry

/-- The main theorem -/
theorem area_probability (s : Square) :
  probability s (fun p => 
    isInside s p ∧ 
    triangleArea s.A s.B p > triangleArea s.B s.C p ∧
    triangleArea s.A s.B p > triangleArea s.C s.D p ∧
    triangleArea s.A s.B p > triangleArea s.D s.A p) = 1/4 := by
  sorry

end area_probability_l278_27883


namespace absolute_value_inequality_solution_set_l278_27824

theorem absolute_value_inequality_solution_set (x : ℝ) :
  (|x - 1| < 1) ↔ (x ∈ Set.Ioo 0 2) :=
sorry

end absolute_value_inequality_solution_set_l278_27824


namespace child_admission_price_l278_27869

theorem child_admission_price
  (total_people : ℕ)
  (adult_price : ℚ)
  (total_receipts : ℚ)
  (num_children : ℕ)
  (h1 : total_people = 610)
  (h2 : adult_price = 2)
  (h3 : total_receipts = 960)
  (h4 : num_children = 260) :
  (total_receipts - (adult_price * (total_people - num_children))) / num_children = 1 :=
by sorry

end child_admission_price_l278_27869


namespace unique_solution_exists_l278_27800

theorem unique_solution_exists : ∃! x : ℝ, 0.6667 * x - 10 = 0.25 * x := by sorry

end unique_solution_exists_l278_27800


namespace isosceles_triangle_perimeter_l278_27835

/-- An isosceles triangle with sides a, b, and c, where two sides are equal. -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : (a = b) ∨ (b = c) ∨ (a = c)
  positive : a > 0 ∧ b > 0 ∧ c > 0

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem stating that an isosceles triangle with sides 3 and 4 has perimeter 10 or 11 -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, 
  ((t.a = 3 ∧ t.b = 4) ∨ (t.a = 4 ∧ t.b = 3) ∨ 
   (t.b = 3 ∧ t.c = 4) ∨ (t.b = 4 ∧ t.c = 3) ∨ 
   (t.a = 3 ∧ t.c = 4) ∨ (t.a = 4 ∧ t.c = 3)) →
  (perimeter t = 10 ∨ perimeter t = 11) :=
by sorry


end isosceles_triangle_perimeter_l278_27835


namespace g_5_equals_104_l278_27806

def g (x : ℝ) : ℝ := 3 * x^4 - 22 * x^3 + 47 * x^2 - 44 * x + 24

theorem g_5_equals_104 : g 5 = 104 := by sorry

end g_5_equals_104_l278_27806


namespace lcm_factor_problem_l278_27876

theorem lcm_factor_problem (A B : ℕ+) (X : ℕ+) : 
  Nat.gcd A B = 23 →
  A = 391 →
  Nat.lcm A B = 23 * 16 * X →
  X = 17 := by
sorry

end lcm_factor_problem_l278_27876


namespace blue_whale_tongue_weight_l278_27829

/-- The weight of an adult blue whale's tongue in pounds -/
def tongue_weight : ℕ := 6000

/-- The number of pounds in one ton -/
def pounds_per_ton : ℕ := 2000

/-- The weight of an adult blue whale's tongue in tons -/
def tongue_weight_in_tons : ℚ := tongue_weight / pounds_per_ton

theorem blue_whale_tongue_weight : tongue_weight_in_tons = 3 := by
  sorry

end blue_whale_tongue_weight_l278_27829


namespace orange_juice_serving_size_l278_27822

/-- Given the conditions for preparing orange juice, prove that each serving is 6 ounces. -/
theorem orange_juice_serving_size :
  -- Conditions
  (concentrate_to_water_ratio : ℚ) →
  (concentrate_cans : ℕ) →
  (concentrate_size : ℚ) →
  (total_servings : ℕ) →
  -- Assumptions
  concentrate_to_water_ratio = 1 / 4 →
  concentrate_cans = 45 →
  concentrate_size = 12 →
  total_servings = 360 →
  -- Conclusion
  (total_volume : ℚ) →
  total_volume = concentrate_cans * concentrate_size * (1 + 1 / concentrate_to_water_ratio) →
  total_volume / total_servings = 6 :=
by sorry

end orange_juice_serving_size_l278_27822


namespace least_positive_integer_congruence_l278_27855

theorem least_positive_integer_congruence :
  ∃! x : ℕ+, x.val + 3567 ≡ 1543 [ZMOD 14] ∧
  ∀ y : ℕ+, y.val + 3567 ≡ 1543 [ZMOD 14] → x ≤ y :=
by sorry

end least_positive_integer_congruence_l278_27855


namespace number_of_teams_in_league_l278_27858

theorem number_of_teams_in_league : ∃ n : ℕ, n > 0 ∧ n * (n - 1) / 2 = 36 := by
  sorry

end number_of_teams_in_league_l278_27858


namespace task_completion_probability_l278_27889

theorem task_completion_probability (p1 p2 : ℚ) (h1 : p1 = 2/3) (h2 : p2 = 3/5) :
  p1 * (1 - p2) = 4/15 := by
  sorry

end task_completion_probability_l278_27889


namespace range_of_a_l278_27842

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs (4 * x - 3) ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
  (∀ x : ℝ, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x : ℝ, ¬(p x) ∧ q x a) → 
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end range_of_a_l278_27842


namespace zhang_bin_is_journalist_l278_27813

structure Person where
  name : String
  isJournalist : Bool
  statement : Bool

def liZhiming : Person := { name := "Li Zhiming", isJournalist := false, statement := true }
def zhangBin : Person := { name := "Zhang Bin", isJournalist := false, statement := true }
def wangDawei : Person := { name := "Wang Dawei", isJournalist := false, statement := true }

theorem zhang_bin_is_journalist :
  ∀ (li : Person) (zhang : Person) (wang : Person),
    li.name = "Li Zhiming" →
    zhang.name = "Zhang Bin" →
    wang.name = "Wang Dawei" →
    (li.isJournalist ∨ zhang.isJournalist ∨ wang.isJournalist) →
    (li.isJournalist → ¬zhang.isJournalist ∧ ¬wang.isJournalist) →
    (zhang.isJournalist → ¬li.isJournalist ∧ ¬wang.isJournalist) →
    (wang.isJournalist → ¬li.isJournalist ∧ ¬zhang.isJournalist) →
    li.statement = li.isJournalist →
    zhang.statement = ¬zhang.isJournalist →
    wang.statement = ¬li.statement →
    (li.statement ∨ zhang.statement ∨ wang.statement) →
    (li.statement → ¬zhang.statement ∧ ¬wang.statement) →
    (zhang.statement → ¬li.statement ∧ ¬wang.statement) →
    (wang.statement → ¬li.statement ∧ ¬zhang.statement) →
    zhang.isJournalist := by
  sorry

#check zhang_bin_is_journalist

end zhang_bin_is_journalist_l278_27813


namespace sqrt_sum_equals_two_sqrt_ten_l278_27812

theorem sqrt_sum_equals_two_sqrt_ten : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 10 := by
  sorry

end sqrt_sum_equals_two_sqrt_ten_l278_27812


namespace min_power_of_two_greater_than_factorial_10_l278_27866

-- Define the given logarithm values
def log10_2 : ℝ := 0.301
def log10_3 : ℝ := 0.477
def log10_7 : ℝ := 0.845

-- Define 10!
def factorial_10 : ℕ := 3628800

-- Theorem statement
theorem min_power_of_two_greater_than_factorial_10 :
  ∃ n : ℕ, factorial_10 < 2^n ∧ ∀ m : ℕ, m < n → factorial_10 ≥ 2^m :=
sorry

end min_power_of_two_greater_than_factorial_10_l278_27866


namespace three_dice_not_one_or_six_l278_27827

/-- The probability of a single die not showing 1 or 6 -/
def single_die_prob : ℚ := 4 / 6

/-- The number of dice tossed -/
def num_dice : ℕ := 3

/-- The probability that none of the three dice show 1 or 6 -/
def three_dice_prob : ℚ := single_die_prob ^ num_dice

theorem three_dice_not_one_or_six :
  three_dice_prob = 8 / 27 := by
  sorry

end three_dice_not_one_or_six_l278_27827


namespace addition_subtraction_proof_l278_27847

theorem addition_subtraction_proof : 987 + 113 - 1000 = 100 := by
  sorry

end addition_subtraction_proof_l278_27847


namespace expected_messages_is_27_l278_27899

/-- Calculates the expected number of greeting messages --/
def expected_messages (total_colleagues : ℕ) 
  (probabilities : List ℝ) (people_counts : List ℕ) : ℝ :=
  List.sum (List.zipWith (· * ·) probabilities people_counts)

/-- Theorem: The expected number of greeting messages is 27 --/
theorem expected_messages_is_27 : 
  let total_colleagues : ℕ := 40
  let probabilities : List ℝ := [1, 0.8, 0.5, 0]
  let people_counts : List ℕ := [8, 15, 14, 3]
  expected_messages total_colleagues probabilities people_counts = 27 := by
  sorry

end expected_messages_is_27_l278_27899


namespace mary_score_unique_l278_27818

/-- Represents the scoring system for the AHSME -/
structure AHSMEScore where
  correct : ℕ
  wrong : ℕ
  score : ℕ
  total_problems : ℕ := 30
  score_formula : score = 35 + 5 * correct - wrong
  valid_answers : correct + wrong ≤ total_problems

/-- Represents the condition for John to uniquely determine Mary's score -/
def uniquely_determinable (s : AHSMEScore) : Prop :=
  ∀ s' : AHSMEScore, s'.score > 90 → s'.score ≤ s.score → s' = s

/-- Mary's AHSME score satisfies all conditions and is uniquely determinable -/
theorem mary_score_unique : 
  ∃! s : AHSMEScore, s.score > 90 ∧ uniquely_determinable s ∧ 
  s.correct = 12 ∧ s.wrong = 0 ∧ s.score = 95 := by
  sorry


end mary_score_unique_l278_27818


namespace min_xy_value_l278_27867

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  x * y ≥ 64 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 8*y₀ - x₀*y₀ = 0 ∧ x₀ * y₀ = 64 :=
by sorry

end min_xy_value_l278_27867


namespace dogsled_race_speed_l278_27809

/-- Proves that given a 300-mile course, if one team (A) finishes 3 hours faster than another team (T)
    and has an average speed 5 mph greater, then the slower team's (T) average speed is 20 mph. -/
theorem dogsled_race_speed (course_length : ℝ) (time_difference : ℝ) (speed_difference : ℝ) :
  course_length = 300 →
  time_difference = 3 →
  speed_difference = 5 →
  ∃ (speed_T : ℝ) (time_T : ℝ) (time_A : ℝ),
    course_length = speed_T * time_T ∧
    course_length = (speed_T + speed_difference) * (time_T - time_difference) ∧
    speed_T = 20 := by
  sorry

end dogsled_race_speed_l278_27809


namespace beads_per_necklace_is_eight_l278_27838

/-- The number of beads Emily has -/
def total_beads : ℕ := 16

/-- The number of necklaces Emily can make -/
def num_necklaces : ℕ := 2

/-- The number of beads per necklace -/
def beads_per_necklace : ℕ := total_beads / num_necklaces

theorem beads_per_necklace_is_eight : beads_per_necklace = 8 := by
  sorry

end beads_per_necklace_is_eight_l278_27838


namespace parallel_line_through_point_l278_27887

-- Define the given line
def given_line (x y : ℝ) : Prop := 3 * x - 4 * y + 6 = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, -1)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 3 * x - 4 * y - 16 = 0

-- Theorem statement
theorem parallel_line_through_point :
  (parallel_line point_P.1 point_P.2) ∧ 
  (∀ (x y : ℝ), parallel_line x y ↔ ∃ (k : ℝ), given_line (x + k) (y + (3/4) * k)) :=
sorry

end parallel_line_through_point_l278_27887


namespace greatest_sum_of_valid_pair_l278_27888

/-- Two integers that differ by 2 and have a product less than 500 -/
def ValidPair (n m : ℤ) : Prop :=
  m = n + 2 ∧ n * m < 500

/-- The sum of a valid pair of integers -/
def PairSum (n m : ℤ) : ℤ := n + m

/-- Theorem: The greatest possible sum of two integers that differ by 2 
    and whose product is less than 500 is 44 -/
theorem greatest_sum_of_valid_pair : 
  (∃ (n m : ℤ), ValidPair n m ∧ 
    ∀ (k l : ℤ), ValidPair k l → PairSum k l ≤ PairSum n m) ∧
  (∀ (n m : ℤ), ValidPair n m → PairSum n m ≤ 44) := by
  sorry

end greatest_sum_of_valid_pair_l278_27888


namespace least_possible_lcm_a_c_l278_27845

theorem least_possible_lcm_a_c (a b c : ℕ) 
  (h1 : Nat.lcm a b = 18) 
  (h2 : Nat.lcm b c = 20) : 
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 90 ∧ 
    (∀ (x y : ℕ), Nat.lcm x b = 18 → Nat.lcm b y = 20 → Nat.lcm a' c' ≤ Nat.lcm x y) := by
  sorry

end least_possible_lcm_a_c_l278_27845


namespace student_count_equality_l278_27803

/-- Proves that the number of students in class A equals the number in class C,
    given the average ages of each class and the overall average age. -/
theorem student_count_equality (a b c : ℕ) : 
  (14 * a + 13 * b + 12 * c : ℝ) / (a + b + c : ℝ) = 13 → a = c := by
  sorry

end student_count_equality_l278_27803


namespace company_blocks_l278_27879

/-- Represents the number of workers in each block -/
def workers_per_block : ℕ := 200

/-- Represents the total budget for gifts in dollars -/
def total_budget : ℕ := 6000

/-- Represents the cost of each gift in dollars -/
def gift_cost : ℕ := 2

/-- Calculates the number of blocks in the company -/
def number_of_blocks : ℕ := total_budget / (workers_per_block * gift_cost)

/-- Theorem stating that the number of blocks in the company is 15 -/
theorem company_blocks : number_of_blocks = 15 := by
  sorry

end company_blocks_l278_27879


namespace not_always_true_a_gt_b_implies_ac_sq_gt_bc_sq_l278_27821

theorem not_always_true_a_gt_b_implies_ac_sq_gt_bc_sq :
  ¬ ∀ (a b c : ℝ), a > b → a * c^2 > b * c^2 :=
sorry

end not_always_true_a_gt_b_implies_ac_sq_gt_bc_sq_l278_27821


namespace cistern_filling_time_l278_27886

/-- The time it takes to fill a cistern when two taps (one filling, one emptying) are opened simultaneously -/
theorem cistern_filling_time 
  (fill_time : ℝ) 
  (empty_time : ℝ) 
  (fill_time_pos : 0 < fill_time)
  (empty_time_pos : 0 < empty_time) : 
  (fill_time * empty_time) / (empty_time - fill_time) = 
    1 / (1 / fill_time - 1 / empty_time) :=
by sorry

end cistern_filling_time_l278_27886


namespace average_visitors_is_750_l278_27823

/-- Calculates the average number of visitors per day in a 30-day month starting on Sunday -/
def average_visitors (sunday_visitors : ℕ) (other_day_visitors : ℕ) : ℚ :=
  let total_sundays : ℕ := 5
  let total_other_days : ℕ := 30 - total_sundays
  let total_visitors : ℕ := sunday_visitors * total_sundays + other_day_visitors * total_other_days
  (total_visitors : ℚ) / 30

/-- Theorem stating that the average number of visitors per day is 750 -/
theorem average_visitors_is_750 :
  average_visitors 1000 700 = 750 := by sorry

end average_visitors_is_750_l278_27823


namespace sum_of_integers_with_product_5_4_l278_27828

theorem sum_of_integers_with_product_5_4 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 625 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 131 := by
sorry

end sum_of_integers_with_product_5_4_l278_27828


namespace circle_equation_k_value_l278_27878

theorem circle_equation_k_value (x y k : ℝ) : 
  (∃ h c : ℝ, ∀ x y : ℝ, x^2 + 12*x + y^2 + 14*y - k = 0 ↔ (x - h)^2 + (y - c)^2 = 8^2) ↔ 
  k = 85 := by
sorry

end circle_equation_k_value_l278_27878


namespace sine_cosine_equality_l278_27832

theorem sine_cosine_equality (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) :
  Real.sin (n * Real.pi / 180) = Real.cos (510 * Real.pi / 180) → n = -60 := by
  sorry

end sine_cosine_equality_l278_27832


namespace cos_range_theorem_l278_27826

theorem cos_range_theorem (ω : ℝ) (h_ω : ω > 0) :
  (∀ x ∈ Set.Icc 0 (π / 3), 3 * Real.sin (ω * x) + 4 * Real.cos (ω * x) ∈ Set.Icc 4 5) →
  (∃ y ∈ Set.Icc (7 / 25) (4 / 5), y = Real.cos (π * ω / 3)) ∧
  (∀ y, y = Real.cos (π * ω / 3) → y ∈ Set.Icc (7 / 25) (4 / 5)) :=
by sorry

end cos_range_theorem_l278_27826


namespace amy_chore_money_l278_27805

/-- Calculates the money earned from chores given initial amount, birthday money, and final amount --/
def money_from_chores (initial_amount birthday_money final_amount : ℕ) : ℕ :=
  final_amount - initial_amount - birthday_money

/-- Theorem stating that Amy's money from chores is 13 dollars --/
theorem amy_chore_money :
  money_from_chores 2 3 18 = 13 := by
  sorry

end amy_chore_money_l278_27805


namespace line_tangent_to_parabola_l278_27894

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero -/
axiom tangent_condition (a b c : ℝ) : 
  b^2 - 4*a*c = 0 ↔ ∃ k, (∀ x y, a*y^2 + b*y + c = 0 ∧ y^2 = 16*x ∧ 6*x - 4*y + k = 0)

/-- The value of k for which the line 6x - 4y + k = 0 is tangent to the parabola y^2 = 16x -/
theorem line_tangent_to_parabola : 
  ∃! k, ∀ x y, (y^2 = 16*x ∧ 6*x - 4*y + k = 0) → k = 32/3 := by
  sorry

end line_tangent_to_parabola_l278_27894


namespace sine_cosine_equation_solution_l278_27814

theorem sine_cosine_equation_solution (x : ℝ) :
  12 * Real.sin x - 5 * Real.cos x = 13 →
  ∃ k : ℤ, x = π / 2 + Real.arctan (5 / 12) + 2 * π * ↑k :=
by sorry

end sine_cosine_equation_solution_l278_27814


namespace adam_jackie_apple_difference_l278_27874

theorem adam_jackie_apple_difference :
  ∀ (adam_apples jackie_apples : ℕ),
    adam_apples = 10 →
    jackie_apples = 2 →
    adam_apples - jackie_apples = 8 :=
by
  sorry

end adam_jackie_apple_difference_l278_27874


namespace sqrt_65_bound_l278_27808

theorem sqrt_65_bound (n : ℕ) (h1 : 0 < n) (h2 : n < Real.sqrt 65) (h3 : Real.sqrt 65 < n + 1) : n = 8 := by
  sorry

end sqrt_65_bound_l278_27808


namespace max_regions_three_triangles_is_20_l278_27816

/-- The maximum number of regions formed by three triangles on a plane -/
def max_regions_three_triangles : ℕ := 20

/-- The number of triangles drawn on the plane -/
def num_triangles : ℕ := 3

/-- Theorem stating that the maximum number of regions formed by three triangles is 20 -/
theorem max_regions_three_triangles_is_20 :
  max_regions_three_triangles = 20 ∧ num_triangles = 3 := by sorry

end max_regions_three_triangles_is_20_l278_27816


namespace min_value_of_f_l278_27862

-- Define the function f(x)
def f (x : ℝ) : ℝ := 12 * x - x^3

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-3) 3 ∧ f x = -16 ∧ ∀ y ∈ Set.Icc (-3) 3, f y ≥ f x :=
sorry

end min_value_of_f_l278_27862


namespace square_area_theorem_l278_27817

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (topLeft : Point)
  (bottomRight : Point)

/-- Represents a square -/
structure Square :=
  (topLeft : Point)
  (sideLength : ℝ)

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  (r.bottomRight.x - r.topLeft.x) * (r.topLeft.y - r.bottomRight.y)

/-- Theorem: If a square is divided into four rectangles of equal area and MN = 3,
    then the area of the square is 64 -/
theorem square_area_theorem (s : Square) 
  (r1 r2 r3 r4 : Rectangle) 
  (h1 : rectangleArea r1 = rectangleArea r2)
  (h2 : rectangleArea r2 = rectangleArea r3)
  (h3 : rectangleArea r3 = rectangleArea r4)
  (h4 : r1.topLeft = s.topLeft)
  (h5 : r4.bottomRight.x = s.topLeft.x + s.sideLength)
  (h6 : r4.bottomRight.y = s.topLeft.y - s.sideLength)
  (h7 : r1.bottomRight.x - r1.topLeft.x = 3) : 
  s.sideLength * s.sideLength = 64 :=
sorry

end square_area_theorem_l278_27817


namespace initial_juice_percentage_l278_27830

/-- Proves that the initial percentage of pure fruit juice in a 2-liter mixture is 10% -/
theorem initial_juice_percentage :
  let initial_volume : ℝ := 2
  let added_juice : ℝ := 0.4
  let final_percentage : ℝ := 25
  let final_volume : ℝ := initial_volume + added_juice
  ∀ initial_percentage : ℝ,
    (initial_percentage / 100 * initial_volume + added_juice) / final_volume * 100 = final_percentage →
    initial_percentage = 10 := by
  sorry

end initial_juice_percentage_l278_27830


namespace daily_reading_goal_l278_27895

def september_days : ℕ := 30
def total_pages : ℕ := 600
def unavailable_days : ℕ := 10
def flight_day_pages : ℕ := 100

def available_days : ℕ := september_days - unavailable_days - 1
def remaining_pages : ℕ := total_pages - flight_day_pages

theorem daily_reading_goal :
  ∃ (pages_per_day : ℕ),
    pages_per_day * available_days ≥ remaining_pages ∧
    pages_per_day = 27 := by
  sorry

end daily_reading_goal_l278_27895


namespace apples_left_over_l278_27846

theorem apples_left_over (liam mia noah : ℕ) (h1 : liam = 53) (h2 : mia = 68) (h3 : noah = 22) : 
  (liam + mia + noah) % 10 = 3 := by
  sorry

end apples_left_over_l278_27846


namespace expected_full_circles_l278_27857

/-- Represents the tiling of an equilateral triangle -/
structure TriangleTiling where
  n : ℕ
  sideLength : n > 2

/-- Expected number of full circles in a triangle tiling -/
def expectedFullCircles (t : TriangleTiling) : ℚ :=
  (t.n - 2) * (t.n - 1) / 1458

/-- Theorem stating the expected number of full circles in a triangle tiling -/
theorem expected_full_circles (t : TriangleTiling) :
  expectedFullCircles t = (t.n - 2) * (t.n - 1) / 1458 :=
by sorry

end expected_full_circles_l278_27857


namespace max_value_of_quadratic_function_l278_27891

theorem max_value_of_quadratic_function :
  ∃ (M : ℝ), M = 1/4 ∧ ∀ x : ℝ, x - x^2 ≤ M :=
sorry

end max_value_of_quadratic_function_l278_27891
