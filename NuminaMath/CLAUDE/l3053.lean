import Mathlib

namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3053_305337

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) : 
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution : 
  ∃ (k : ℕ), k = 3 ∧ (427398 - k) % 15 = 0 ∧ ∀ (m : ℕ), m < k → (427398 - m) % 15 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3053_305337


namespace NUMINAMATH_CALUDE_overtime_pay_ratio_l3053_305379

/-- Given Bill's pay structure, prove the ratio of overtime to regular pay rate --/
theorem overtime_pay_ratio (initial_rate : ℝ) (total_pay : ℝ) (total_hours : ℕ) (regular_hours : ℕ) :
  initial_rate = 20 →
  total_pay = 1200 →
  total_hours = 50 →
  regular_hours = 40 →
  (total_pay - initial_rate * regular_hours) / (total_hours - regular_hours) / initial_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_overtime_pay_ratio_l3053_305379


namespace NUMINAMATH_CALUDE_area_equality_l3053_305372

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if a quadrilateral is a parallelogram -/
def isParallelogram (q : Quadrilateral) : Prop := sorry

/-- Calculate the area of a quadrilateral -/
def areaQuadrilateral (q : Quadrilateral) : ℝ := sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (M : Point) (A : Point) (B : Point) : Prop := sorry

/-- Calculate the area of a triangle -/
def areaTriangle (A : Point) (B : Point) (C : Point) : ℝ := sorry

/-- Main theorem -/
theorem area_equality 
  (C D E F G H J : Point)
  (CDEF : Quadrilateral)
  (h1 : isParallelogram CDEF)
  (h2 : areaQuadrilateral CDEF = 36)
  (h3 : isMidpoint G C D)
  (h4 : isMidpoint H E F) :
  areaTriangle C D J = areaQuadrilateral CDEF :=
sorry

end NUMINAMATH_CALUDE_area_equality_l3053_305372


namespace NUMINAMATH_CALUDE_smallest_n_with_triple_sum_l3053_305345

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Proposition: The smallest positive integer N whose sum of digits is three times 
    the sum of digits of N+1 has a sum of digits equal to 12 -/
theorem smallest_n_with_triple_sum : 
  ∃ (N : ℕ), N > 0 ∧ 
  sum_of_digits N = 3 * sum_of_digits (N + 1) ∧
  sum_of_digits N = 12 ∧
  ∀ (M : ℕ), M > 0 → sum_of_digits M = 3 * sum_of_digits (M + 1) → 
    sum_of_digits M ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_triple_sum_l3053_305345


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_common_foci_l3053_305370

/-- The value of m for which the given ellipse and hyperbola share common foci -/
theorem ellipse_hyperbola_common_foci : ∃ m : ℝ,
  (∀ x y : ℝ, x^2 / 25 + y^2 / 16 = 1 → x^2 / m - y^2 / 5 = 1) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 →
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 →
      ∃ c : ℝ, c^2 = a^2 - b^2 ∧ (x = c ∨ x = -c) ∧ y = 0)) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 →
    (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 →
      ∃ c : ℝ, c^2 = a^2 + b^2 ∧ (x = c ∨ x = -c) ∧ y = 0)) →
  m = 4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_common_foci_l3053_305370


namespace NUMINAMATH_CALUDE_system_solution_l3053_305303

theorem system_solution (x₁ x₂ x₃ x₄ : ℝ) : 
  x₁^2 + x₂^2 + x₃^2 + x₄^2 = 4 ∧
  x₁*x₃ + x₂*x₄ + x₃*x₂ + x₄*x₁ = 0 ∧
  x₁*x₂*x₃ + x₁*x₂*x₄ + x₁*x₃*x₄ + x₂*x₃*x₄ = -2 ∧
  x₁*x₂*x₃*x₄ = -1 →
  (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = -1) ∨
  (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = -1 ∧ x₄ = 1) ∨
  (x₁ = 1 ∧ x₂ = -1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
  (x₁ = -1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l3053_305303


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3053_305373

theorem sin_2alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.sqrt 2 * Real.cos (2 * α) = Real.sin (α + π/4)) : 
  Real.sin (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3053_305373


namespace NUMINAMATH_CALUDE_picture_fit_count_l3053_305363

-- Define the number of pictures for each category for Ralph and Derrick
def ralph_wild_animals : ℕ := 75
def ralph_landscapes : ℕ := 36
def ralph_family_events : ℕ := 45
def ralph_cars : ℕ := 20

def derrick_wild_animals : ℕ := 95
def derrick_landscapes : ℕ := 42
def derrick_family_events : ℕ := 55
def derrick_cars : ℕ := 25
def derrick_airplanes : ℕ := 10

-- Calculate total pictures for Ralph and Derrick
def ralph_total : ℕ := ralph_wild_animals + ralph_landscapes + ralph_family_events + ralph_cars
def derrick_total : ℕ := derrick_wild_animals + derrick_landscapes + derrick_family_events + derrick_cars + derrick_airplanes

-- Calculate the combined total of pictures
def combined_total : ℕ := ralph_total + derrick_total

-- Calculate the difference in wild animal pictures
def wild_animals_difference : ℕ := derrick_wild_animals - ralph_wild_animals

-- Theorem to prove
theorem picture_fit_count : (combined_total / wild_animals_difference : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_picture_fit_count_l3053_305363


namespace NUMINAMATH_CALUDE_solution_approximation_l3053_305339

-- Define the equation
def equation (x y : ℝ) : Prop :=
  ((3 * x^2 - 7)^2 / 9) + 5 * y = x^3 - 2 * x

-- State the theorem
theorem solution_approximation :
  ∃ y : ℝ, equation 4 y ∧ abs (y + 26.155) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_solution_approximation_l3053_305339


namespace NUMINAMATH_CALUDE_sum_a_b_c_value_l3053_305355

theorem sum_a_b_c_value :
  ∀ (a b c : ℤ),
  (∀ x : ℤ, x < 0 → x ≤ a) →  -- a is the largest negative integer
  (abs b = 6) →               -- |b| = 6
  (c = -c) →                  -- c is equal to its opposite
  (a + b + c = -7 ∨ a + b + c = 5) := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_c_value_l3053_305355


namespace NUMINAMATH_CALUDE_fraction_simplification_l3053_305311

theorem fraction_simplification : 
  (3+6-12+24+48-96+192) / (6+12-24+48+96-192+384) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3053_305311


namespace NUMINAMATH_CALUDE_glove_selection_theorem_l3053_305327

theorem glove_selection_theorem :
  let total_gloves : ℕ := 8
  let gloves_to_select : ℕ := 4
  let num_pairs : ℕ := 4
  let total_selections : ℕ := Nat.choose total_gloves gloves_to_select
  let no_pair_selections : ℕ := 2^num_pairs
  total_selections - no_pair_selections = 54 :=
by sorry

end NUMINAMATH_CALUDE_glove_selection_theorem_l3053_305327


namespace NUMINAMATH_CALUDE_complex_z_value_l3053_305324

theorem complex_z_value : ∃ z : ℂ, z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3) ∧ 
  z = Complex.mk (1/2) (-Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_z_value_l3053_305324


namespace NUMINAMATH_CALUDE_book_pages_ratio_l3053_305301

theorem book_pages_ratio : 
  let selena_pages : ℕ := 400
  let harry_pages : ℕ := 180
  ∃ (a b : ℕ), (a = 9 ∧ b = 20) ∧ 
    (harry_pages : ℚ) / selena_pages = a / b :=
by sorry

end NUMINAMATH_CALUDE_book_pages_ratio_l3053_305301


namespace NUMINAMATH_CALUDE_square_area_increase_l3053_305349

theorem square_area_increase (a : ℝ) (ha : a > 0) : 
  let side_b := 2 * a
  let side_c := side_b * 1.8
  let area_a := a ^ 2
  let area_b := side_b ^ 2
  let area_c := side_c ^ 2
  (area_c - (area_a + area_b)) / (area_a + area_b) = 1.592 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l3053_305349


namespace NUMINAMATH_CALUDE_product_remainder_theorem_l3053_305333

theorem product_remainder_theorem (x : ℤ) : 
  (37 * x) % 31 = 15 ↔ ∃ k : ℤ, x = 18 + 31 * k := by sorry

end NUMINAMATH_CALUDE_product_remainder_theorem_l3053_305333


namespace NUMINAMATH_CALUDE_volume_of_specific_pyramid_l3053_305369

/-- A right pyramid with a square base and an equilateral triangular face --/
structure RightPyramid where
  -- The side length of the equilateral triangular face
  side_length : ℝ
  -- Assumption that the side length is positive
  side_length_pos : side_length > 0

/-- The volume of the right pyramid --/
noncomputable def volume (p : RightPyramid) : ℝ :=
  (500 * Real.sqrt 3) / 3

/-- Theorem stating the volume of the specific pyramid --/
theorem volume_of_specific_pyramid :
  ∃ (p : RightPyramid), p.side_length = 10 ∧ volume p = (500 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_pyramid_l3053_305369


namespace NUMINAMATH_CALUDE_lcm_hcf_relation_l3053_305367

theorem lcm_hcf_relation (a b : ℕ) (h : a = 24 ∧ b = 198) :
  Nat.lcm a b = 792 :=
by
  sorry

#check lcm_hcf_relation

end NUMINAMATH_CALUDE_lcm_hcf_relation_l3053_305367


namespace NUMINAMATH_CALUDE_martha_blocks_l3053_305380

/-- Given Martha's initial and found blocks, prove the total number of blocks she ends with. -/
theorem martha_blocks (initial_blocks found_blocks : ℕ) 
  (h1 : initial_blocks = 4)
  (h2 : found_blocks = 80) :
  initial_blocks + found_blocks = 84 := by
  sorry

#check martha_blocks

end NUMINAMATH_CALUDE_martha_blocks_l3053_305380


namespace NUMINAMATH_CALUDE_largest_x_absolute_value_equation_l3053_305343

theorem largest_x_absolute_value_equation :
  ∀ x : ℝ, |5 - x| = 15 + x → x ≤ -5 ∧ |-5 - 5| = 15 + (-5) := by
  sorry

end NUMINAMATH_CALUDE_largest_x_absolute_value_equation_l3053_305343


namespace NUMINAMATH_CALUDE_harriets_age_l3053_305368

theorem harriets_age (mother_age : ℕ) (peter_age : ℕ) (harriet_age : ℕ) : 
  mother_age = 60 →
  peter_age = mother_age / 2 →
  peter_age + 4 = 2 * (harriet_age + 4) →
  harriet_age = 13 := by
sorry

end NUMINAMATH_CALUDE_harriets_age_l3053_305368


namespace NUMINAMATH_CALUDE_common_point_on_intersection_circle_l3053_305389

/-- A parabola that intersects the coordinate axes at three distinct points -/
structure TripleIntersectingParabola where
  p : ℝ
  q : ℝ
  x₁ : ℝ
  x₂ : ℝ
  h₁ : x₁ ≠ 0
  h₂ : x₂ ≠ 0
  h₃ : q ≠ 0
  h₄ : x₁ ≠ x₂
  h₅ : x₁^2 + p*x₁ + q = 0  -- x₁ is a root
  h₆ : x₂^2 + p*x₂ + q = 0  -- x₂ is a root

/-- The circle passing through the three intersection points of the parabola with the coordinate axes -/
def intersectionCircle (para : TripleIntersectingParabola) : Set (ℝ × ℝ) :=
  {pt : ℝ × ℝ | ∃ (r : ℝ), (pt.1 - 0)^2 + (pt.2 - 0)^2 = r^2 ∧
                           (pt.1 - para.x₁)^2 + pt.2^2 = r^2 ∧
                           (pt.1 - para.x₂)^2 + pt.2^2 = r^2 ∧
                           (pt.1 - 0)^2 + (pt.2 - para.q)^2 = r^2}

/-- The theorem stating that R(0, 1) lies on the intersection circle for all valid parabolas -/
theorem common_point_on_intersection_circle (para : TripleIntersectingParabola) :
  (0, 1) ∈ intersectionCircle para := by
  sorry

end NUMINAMATH_CALUDE_common_point_on_intersection_circle_l3053_305389


namespace NUMINAMATH_CALUDE_group_bill_proof_l3053_305309

def restaurant_bill (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

theorem group_bill_proof (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ)
  (h1 : total_people = 13)
  (h2 : num_kids = 9)
  (h3 : adult_meal_cost = 7) :
  restaurant_bill total_people num_kids adult_meal_cost = 28 := by
  sorry

end NUMINAMATH_CALUDE_group_bill_proof_l3053_305309


namespace NUMINAMATH_CALUDE_two_valid_solutions_exist_l3053_305360

def is_valid_solution (a b c d e f g h i : ℕ) : Prop :=
  a ∈ Finset.range 10 ∧ b ∈ Finset.range 10 ∧ c ∈ Finset.range 10 ∧
  d ∈ Finset.range 10 ∧ e ∈ Finset.range 10 ∧ f ∈ Finset.range 10 ∧
  g ∈ Finset.range 10 ∧ h ∈ Finset.range 10 ∧ i ∈ Finset.range 10 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i ∧
  (100 * a + 10 * b + c) + d = 10 * e + f ∧
  g * h = 10 * i + f

theorem two_valid_solutions_exist : ∃ (a b c d e f g h i : ℕ),
  is_valid_solution a b c d e f g h i ∧
  ∃ (j k l m n o p q r : ℕ),
  is_valid_solution j k l m n o p q r ∧
  (a ≠ j ∨ b ≠ k ∨ c ≠ l ∨ d ≠ m ∨ e ≠ n ∨ f ≠ o ∨ g ≠ p ∨ h ≠ q ∨ i ≠ r) :=
sorry

end NUMINAMATH_CALUDE_two_valid_solutions_exist_l3053_305360


namespace NUMINAMATH_CALUDE_voting_ratio_l3053_305315

/-- Given a voting scenario where:
    - 2/9 of the votes have been counted
    - 3/4 of the counted votes are in favor
    - 0.7857142857142856 of the remaining votes are against
    Prove that the ratio of total votes against to total votes in favor is 4:1 -/
theorem voting_ratio (V : ℝ) (hV : V > 0) : 
  let counted := (2/9) * V
  let in_favor := (3/4) * counted
  let remaining := V - counted
  let against_remaining := 0.7857142857142856 * remaining
  let total_against := ((1/4) * counted) + against_remaining
  let total_in_favor := in_favor
  (total_against / total_in_favor) = 4 := by
  sorry


end NUMINAMATH_CALUDE_voting_ratio_l3053_305315


namespace NUMINAMATH_CALUDE_average_weight_increase_l3053_305392

theorem average_weight_increase (initial_count : ℕ) (replaced_weight new_weight : ℝ) : 
  initial_count = 8 →
  replaced_weight = 65 →
  new_weight = 93 →
  (new_weight - replaced_weight) / initial_count = 3.5 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3053_305392


namespace NUMINAMATH_CALUDE_optimal_price_for_equipment_l3053_305384

/-- Represents the selling price and annual sales volume relationship for a high-tech equipment -/
structure EquipmentSales where
  cost_price : ℝ
  price_volume_1 : ℝ × ℝ
  price_volume_2 : ℝ × ℝ
  max_price : ℝ
  target_profit : ℝ

/-- Calculates the optimal selling price for the equipment -/
def optimal_selling_price (sales : EquipmentSales) : ℝ :=
  sorry

/-- Theorem stating the optimal selling price for the given conditions -/
theorem optimal_price_for_equipment (sales : EquipmentSales)
  (h1 : sales.cost_price = 300000)
  (h2 : sales.price_volume_1 = (350000, 550))
  (h3 : sales.price_volume_2 = (400000, 500))
  (h4 : sales.max_price = 600000)
  (h5 : sales.target_profit = 80000000) :
  optimal_selling_price sales = 500000 :=
sorry

end NUMINAMATH_CALUDE_optimal_price_for_equipment_l3053_305384


namespace NUMINAMATH_CALUDE_function_property_l3053_305347

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a
  else Real.log x / Real.log a

-- State the theorem
theorem function_property (a : ℝ) (h : a ≠ 0) (h1 : a ≠ 1) :
  f a (f a 1) = 2 → a = -2 := by sorry

end

end NUMINAMATH_CALUDE_function_property_l3053_305347


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3053_305399

/-- Proves that in a right triangle with non-hypotenuse side lengths of 5 and 12, the hypotenuse length is 13 -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ), 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → c = 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3053_305399


namespace NUMINAMATH_CALUDE_complex_number_with_given_real_part_and_magnitude_l3053_305390

theorem complex_number_with_given_real_part_and_magnitude (z : ℂ) : 
  (z.re = 5) → (Complex.abs z = Complex.abs (4 - 3*I)) → (z.im = 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_with_given_real_part_and_magnitude_l3053_305390


namespace NUMINAMATH_CALUDE_add_9999_seconds_to_5_45_00_l3053_305340

/-- Represents a time of day in hours, minutes, and seconds -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time and returns the new time -/
def addSeconds (time : TimeOfDay) (seconds : Nat) : TimeOfDay :=
  sorry

/-- Converts a TimeOfDay to a string in the format "HH:MM:SS" -/
def TimeOfDay.toString (time : TimeOfDay) : String :=
  sorry

theorem add_9999_seconds_to_5_45_00 :
  let initialTime : TimeOfDay := ⟨17, 45, 0⟩
  let secondsToAdd : Nat := 9999
  let finalTime := addSeconds initialTime secondsToAdd
  finalTime.toString = "20:31:39" :=
sorry

end NUMINAMATH_CALUDE_add_9999_seconds_to_5_45_00_l3053_305340


namespace NUMINAMATH_CALUDE_problem_solution_l3053_305314

theorem problem_solution (x y : ℚ) : 
  x = 103 → x^3 * y - 2 * x^2 * y + x * y = 1060900 → y = 100 / 101 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3053_305314


namespace NUMINAMATH_CALUDE_solve_system_l3053_305377

theorem solve_system (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 36) : x = 28 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3053_305377


namespace NUMINAMATH_CALUDE_symmetric_complex_number_l3053_305351

theorem symmetric_complex_number : ∀ z : ℂ, 
  (z.re = (-1 : ℝ) ∧ z.im = (1 : ℝ)) ↔ 
  (z.re = (2 / (Complex.I - 1)).re ∧ z.im = -(2 / (Complex.I - 1)).im) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_complex_number_l3053_305351


namespace NUMINAMATH_CALUDE_unique_intersection_line_l3053_305305

theorem unique_intersection_line (m b : ℝ) : 
  (∃! k : ℝ, ∃ y₁ y₂ : ℝ, 
    y₁ = k^2 + 4*k + 4 ∧ 
    y₂ = m*k + b ∧ 
    |y₁ - y₂| = 6) →
  (7 = 2*m + b) →
  (m = 8 ∧ b = -9) := by
sorry

end NUMINAMATH_CALUDE_unique_intersection_line_l3053_305305


namespace NUMINAMATH_CALUDE_correlatedRelationships_l3053_305319

-- Define the type for relationships
inductive Relationship
  | GreatTeachersAndStudents
  | SphereVolumeAndRadius
  | AppleYieldAndClimate
  | TreeDiameterAndHeight
  | StudentAndID
  | CrowCawAndOmen

-- Define a function to check if a relationship has correlation
def hasCorrelation (r : Relationship) : Prop :=
  match r with
  | Relationship.GreatTeachersAndStudents => True
  | Relationship.SphereVolumeAndRadius => False
  | Relationship.AppleYieldAndClimate => True
  | Relationship.TreeDiameterAndHeight => True
  | Relationship.StudentAndID => False
  | Relationship.CrowCawAndOmen => False

-- Theorem stating which relationships have correlation
theorem correlatedRelationships :
  (hasCorrelation Relationship.GreatTeachersAndStudents) ∧
  (hasCorrelation Relationship.AppleYieldAndClimate) ∧
  (hasCorrelation Relationship.TreeDiameterAndHeight) ∧
  (¬hasCorrelation Relationship.SphereVolumeAndRadius) ∧
  (¬hasCorrelation Relationship.StudentAndID) ∧
  (¬hasCorrelation Relationship.CrowCawAndOmen) :=
by sorry


end NUMINAMATH_CALUDE_correlatedRelationships_l3053_305319


namespace NUMINAMATH_CALUDE_digits_zeros_equality_l3053_305352

/-- 
Given a positive integer n, count_digits n returns the sum of all digits in n.
-/
def count_digits (n : ℕ) : ℕ := sorry

/-- 
Given a positive integer n, count_zeros n returns the number of zeros in n.
-/
def count_zeros (n : ℕ) : ℕ := sorry

/-- 
sum_digits_to_n n returns the sum of digits of all numbers from 1 to n.
-/
def sum_digits_to_n (n : ℕ) : ℕ := sorry

/-- 
sum_zeros_to_n n returns the count of zeros in all numbers from 1 to n.
-/
def sum_zeros_to_n (n : ℕ) : ℕ := sorry

/-- 
For any positive integer k, the sum of digits in all numbers from 1 to 10^k
is equal to the count of zeros in all numbers from 1 to 10^(k+1).
-/
theorem digits_zeros_equality (k : ℕ) (h : k > 0) : 
  sum_digits_to_n (10^k) = sum_zeros_to_n (10^(k+1)) := by sorry

end NUMINAMATH_CALUDE_digits_zeros_equality_l3053_305352


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_l3053_305371

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (notParallel : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicular (l m : Line) (α : Plane) :
  perpendicular l α → notParallel l m → perpendicular m α := by
  sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_l3053_305371


namespace NUMINAMATH_CALUDE_liam_chocolate_consumption_l3053_305331

/-- Given that Liam ate a total of 150 chocolates in five days, and each day after
    the first day he ate 8 more chocolates than the previous day, prove that
    he ate 38 chocolates on the fourth day. -/
theorem liam_chocolate_consumption :
  ∀ (x : ℕ),
  (x + (x + 8) + (x + 16) + (x + 24) + (x + 32) = 150) →
  (x + 24 = 38) :=
by sorry

end NUMINAMATH_CALUDE_liam_chocolate_consumption_l3053_305331


namespace NUMINAMATH_CALUDE_clay_transformation_in_two_operations_l3053_305394

/-- Represents a collection of clay pieces -/
structure ClayCollection where
  pieces : List Nat
  deriving Repr

/-- Represents an operation on clay pieces -/
def combine_operation (c : ClayCollection) (group_size : Nat) : ClayCollection :=
  sorry

/-- The initial state of clay pieces -/
def initial_state : ClayCollection :=
  { pieces := List.replicate 111 1 }

/-- The desired final state of clay pieces -/
def final_state : ClayCollection :=
  { pieces := [1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16] }

/-- Theorem stating that the transformation is possible in 2 operations -/
theorem clay_transformation_in_two_operations :
  ∃ (op1 op2 : Nat),
    (combine_operation (combine_operation initial_state op1) op2) = final_state :=
  sorry

end NUMINAMATH_CALUDE_clay_transformation_in_two_operations_l3053_305394


namespace NUMINAMATH_CALUDE_tangent_parallel_to_line_l3053_305388

theorem tangent_parallel_to_line (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.sin x + Real.cos x
  let tangent_point : ℝ × ℝ := (π / 2, 1)
  let tangent_slope : ℝ := (deriv f) (π / 2)
  let line_slope : ℝ := 1 / a
  (tangent_slope = line_slope) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_line_l3053_305388


namespace NUMINAMATH_CALUDE_eunji_reading_pages_l3053_305300

theorem eunji_reading_pages (pages_tuesday pages_thursday total_pages : ℕ) 
  (h1 : pages_tuesday = 18)
  (h2 : pages_thursday = 23)
  (h3 : total_pages = 60)
  (h4 : pages_tuesday + pages_thursday + (total_pages - pages_tuesday - pages_thursday) = total_pages) :
  total_pages - pages_tuesday - pages_thursday = 19 := by
  sorry

end NUMINAMATH_CALUDE_eunji_reading_pages_l3053_305300


namespace NUMINAMATH_CALUDE_kasun_family_children_l3053_305383

/-- Represents the Kasun family structure and ages -/
structure KasunFamily where
  num_children : ℕ
  father_age : ℕ
  mother_age : ℕ
  dog_age : ℕ
  children_total_age : ℕ

/-- The average age of the entire family is 22 years -/
def family_average (f : KasunFamily) : Prop :=
  (f.father_age + f.mother_age + f.children_total_age + f.dog_age) / (2 + f.num_children + 1) = 22

/-- The average age of the mother, children, and the pet dog is 18 years -/
def partial_average (f : KasunFamily) : Prop :=
  (f.mother_age + f.children_total_age + f.dog_age) / (1 + f.num_children + 1) = 18

/-- The theorem stating that the number of children in the Kasun family is 5 -/
theorem kasun_family_children (f : KasunFamily) 
  (h1 : family_average f)
  (h2 : partial_average f)
  (h3 : f.father_age = 50)
  (h4 : f.dog_age = 10) : 
  f.num_children = 5 := by
  sorry

end NUMINAMATH_CALUDE_kasun_family_children_l3053_305383


namespace NUMINAMATH_CALUDE_apple_basket_problem_l3053_305353

theorem apple_basket_problem :
  ∃ (a b : ℕ), 4 * a + 3 * a + 3 * b + 2 * b = 31 ∧ 3 * a + 2 * b = 13 :=
by sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l3053_305353


namespace NUMINAMATH_CALUDE_triangle_side_length_triangle_angle_measure_l3053_305321

-- Part 1
theorem triangle_side_length (a b : ℝ) (A B C : ℝ) :
  b = 2 →
  B = π / 6 →
  C = 3 * π / 4 →
  a = Real.sqrt 6 - Real.sqrt 2 :=
sorry

-- Part 2
theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  S = (1 / 4) * (a^2 + b^2 - c^2) →
  C = π / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_triangle_angle_measure_l3053_305321


namespace NUMINAMATH_CALUDE_certain_number_proof_l3053_305316

theorem certain_number_proof (x : ℚ) : 
  x^22 * (1/81)^11 = 1/18^22 → x = 1/36 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3053_305316


namespace NUMINAMATH_CALUDE_maddy_chocolate_eggs_l3053_305341

/-- The number of chocolate eggs Maddy eats per day -/
def eggs_per_day : ℕ := 2

/-- The number of weeks the chocolate eggs last -/
def weeks_lasting : ℕ := 4

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Maddy was given 56 chocolate eggs -/
theorem maddy_chocolate_eggs :
  eggs_per_day * weeks_lasting * days_in_week = 56 := by
  sorry

end NUMINAMATH_CALUDE_maddy_chocolate_eggs_l3053_305341


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3053_305378

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- State the theorem
theorem quadratic_function_range :
  ∃ (a b : ℝ), a = -4 ∧ b = 5 ∧
  (∀ x, x ∈ Set.Icc 0 5 → f x ∈ Set.Icc a b) ∧
  (∀ y, y ∈ Set.Icc a b → ∃ x, x ∈ Set.Icc 0 5 ∧ f x = y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3053_305378


namespace NUMINAMATH_CALUDE_chloes_candies_l3053_305395

/-- Given that Linda has 34 candies and the total number of candies is 62,
    prove that Chloe has 28 candies. -/
theorem chloes_candies (linda_candies : ℕ) (total_candies : ℕ) (chloe_candies : ℕ) : 
  linda_candies = 34 → total_candies = 62 → chloe_candies = total_candies - linda_candies →
  chloe_candies = 28 := by
  sorry

end NUMINAMATH_CALUDE_chloes_candies_l3053_305395


namespace NUMINAMATH_CALUDE_problem_statement_l3053_305329

def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {x | x < a}
def M (m : ℝ) : Set ℝ := {x | x^2 - (1+m)*x + m = 0}

theorem problem_statement (a m : ℝ) (h : m > 1) :
  (A ∩ B a = A → a > 2) ∧
  (m ≠ 2 → A ∪ M m = {1, 2, m}) ∧
  (m = 2 → A ∪ M m = {1, 2}) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3053_305329


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l3053_305346

theorem purely_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ).im ≠ 0 → 
  (Complex.re ((1 - a * Complex.I) * (3 + 2 * Complex.I)) = 0 ∧
   Complex.im ((1 - a * Complex.I) * (3 + 2 * Complex.I)) ≠ 0) → 
  a = -3/2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l3053_305346


namespace NUMINAMATH_CALUDE_f_theorem_l3053_305364

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂) ∧
  (∀ x₁ x₂, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₁ < f x₂) ∧
  (f 1 = 2)

theorem f_theorem (f : ℝ → ℝ) (h : f_properties f) :
  (∀ x₁ x₂, 0 ≤ x₁ → x₁ < x₂ → -(f x₁)^2 > -(f x₂)^2) ∧
  (∀ x₁ x₂, x₁ < x₂ → x₂ ≤ 0 → -(f x₁)^2 < -(f x₂)^2) ∧
  (∀ a, f (2 * a^2 - 1) + 2 * f a - 6 < 0 ↔ -2 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_f_theorem_l3053_305364


namespace NUMINAMATH_CALUDE_multiply_by_point_nine_l3053_305307

theorem multiply_by_point_nine (x : ℝ) : 0.9 * x = 0.0063 → x = 0.007 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_point_nine_l3053_305307


namespace NUMINAMATH_CALUDE_janet_investment_interest_l3053_305393

/-- Calculates the total interest earned from an investment --/
def total_interest (total_investment : ℝ) (high_rate_investment : ℝ) (high_rate : ℝ) (low_rate : ℝ) : ℝ :=
  let low_rate_investment := total_investment - high_rate_investment
  let high_rate_interest := high_rate_investment * high_rate
  let low_rate_interest := low_rate_investment * low_rate
  high_rate_interest + low_rate_interest

/-- Proves that Janet's investment yields $1,390 in interest --/
theorem janet_investment_interest :
  total_interest 31000 12000 0.10 0.01 = 1390 := by
  sorry

end NUMINAMATH_CALUDE_janet_investment_interest_l3053_305393


namespace NUMINAMATH_CALUDE_workshop_prize_difference_l3053_305344

theorem workshop_prize_difference (total : ℕ) (wolf : ℕ) (both : ℕ) (nobel : ℕ) 
  (h_total : total = 50)
  (h_wolf : wolf = 31)
  (h_both : both = 14)
  (h_nobel : nobel = 25)
  (h_wolf_less : wolf ≤ total)
  (h_both_less : both ≤ wolf)
  (h_both_less_nobel : both ≤ nobel)
  (h_nobel_less : nobel ≤ total) :
  let non_wolf := total - wolf
  let nobel_non_wolf := nobel - both
  let non_nobel_non_wolf := non_wolf - nobel_non_wolf
  nobel_non_wolf - non_nobel_non_wolf = 3 := by
  sorry

end NUMINAMATH_CALUDE_workshop_prize_difference_l3053_305344


namespace NUMINAMATH_CALUDE_joe_pays_four_more_than_jenny_l3053_305348

/-- Represents the pizza sharing scenario between Jenny and Joe -/
structure PizzaScenario where
  totalSlices : ℕ
  plainPizzaCost : ℚ
  mushroomExtraCost : ℚ
  mushroomSlices : ℕ
  joeMushroomSlices : ℕ
  joePlainSlices : ℕ

/-- Calculates the cost difference between Joe's and Jenny's payments -/
def paymentDifference (scenario : PizzaScenario) : ℚ :=
  let plainSliceCost := scenario.plainPizzaCost / scenario.totalSlices
  let mushroomSliceCost := plainSliceCost + scenario.mushroomExtraCost
  let jennysSlices := scenario.totalSlices - scenario.joeMushroomSlices - scenario.joePlainSlices
  let joePayment := scenario.joeMushroomSlices * mushroomSliceCost + scenario.joePlainSlices * plainSliceCost
  let jennyPayment := jennysSlices * plainSliceCost
  joePayment - jennyPayment

/-- Theorem stating that in the given scenario, Joe pays $4 more than Jenny -/
theorem joe_pays_four_more_than_jenny : 
  let scenario := PizzaScenario.mk 12 12 (1/2) 4 4 3
  paymentDifference scenario = 4 := by sorry

end NUMINAMATH_CALUDE_joe_pays_four_more_than_jenny_l3053_305348


namespace NUMINAMATH_CALUDE_cube_set_closed_under_multiplication_cube_set_not_closed_under_addition_cube_set_not_closed_under_subtraction_cube_set_not_closed_under_division_l3053_305381

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def cube_set : Set ℕ := {n : ℕ | is_cube n ∧ n > 0}

theorem cube_set_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ cube_set → b ∈ cube_set → (a * b) ∈ cube_set :=
sorry

theorem cube_set_not_closed_under_addition :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ (a + b) ∉ cube_set :=
sorry

theorem cube_set_not_closed_under_subtraction :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ a > b ∧ (a - b) ∉ cube_set :=
sorry

theorem cube_set_not_closed_under_division :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ b > 0 ∧ (a / b) ∉ cube_set :=
sorry

end NUMINAMATH_CALUDE_cube_set_closed_under_multiplication_cube_set_not_closed_under_addition_cube_set_not_closed_under_subtraction_cube_set_not_closed_under_division_l3053_305381


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3053_305335

theorem triangle_angle_problem (A B C : ℝ) : 
  A - B = 10 → B = A / 2 → A + B + C = 180 → C = 150 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3053_305335


namespace NUMINAMATH_CALUDE_bird_watching_problem_l3053_305359

/-- Given 3 bird watchers with an average of 9 birds seen per person,
    where one sees 7 birds and another sees 9 birds,
    prove that the third person must see 11 birds. -/
theorem bird_watching_problem (total_watchers : Nat) (average_birds : Nat) 
  (first_watcher_birds : Nat) (second_watcher_birds : Nat) :
  total_watchers = 3 →
  average_birds = 9 →
  first_watcher_birds = 7 →
  second_watcher_birds = 9 →
  (total_watchers * average_birds - first_watcher_birds - second_watcher_birds) = 11 := by
  sorry

#check bird_watching_problem

end NUMINAMATH_CALUDE_bird_watching_problem_l3053_305359


namespace NUMINAMATH_CALUDE_shipment_total_correct_l3053_305332

/-- The total number of novels in the shipment -/
def total_novels : ℕ := 300

/-- The percentage of novels displayed in the window -/
def display_percentage : ℚ := 30 / 100

/-- The number of novels left in the stockroom -/
def stockroom_novels : ℕ := 210

/-- Theorem stating that the total number of novels is correct given the conditions -/
theorem shipment_total_correct :
  (1 - display_percentage) * total_novels = stockroom_novels := by
  sorry

end NUMINAMATH_CALUDE_shipment_total_correct_l3053_305332


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3053_305387

theorem quadratic_equation_roots (c : ℝ) : 
  c = 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3053_305387


namespace NUMINAMATH_CALUDE_cubic_extrema_difference_l3053_305317

open Real

/-- The cubic function f(x) with parameters a and b. -/
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x

/-- The derivative of f(x) with respect to x. -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_extrema_difference (a b : ℝ) 
  (h1 : f' a b 2 = 0)
  (h2 : f' a b 1 = -3) :
  ∃ (x_max x_min : ℝ), 
    (∀ x, f a b x ≤ f a b x_max) ∧ 
    (∀ x, f a b x_min ≤ f a b x) ∧
    (f a b x_max - f a b x_min = 4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_extrema_difference_l3053_305317


namespace NUMINAMATH_CALUDE_inequality_holds_l3053_305362

theorem inequality_holds (x : ℝ) (h : 0 < x ∧ x < 1) : 0 < 1 - x^2 ∧ 1 - x^2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3053_305362


namespace NUMINAMATH_CALUDE_equation_solution_l3053_305312

theorem equation_solution :
  let f (y : ℝ) := (8 * y^2 + 40 * y - 48) / (3 * y + 9) - (4 * y - 8)
  ∀ y : ℝ, f y = 0 ↔ y = (7 + Real.sqrt 73) / 2 ∨ y = (7 - Real.sqrt 73) / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3053_305312


namespace NUMINAMATH_CALUDE_complex_division_l3053_305350

theorem complex_division (z₁ z₂ : ℂ) : z₁ = 1 + I → z₂ = 1 - I → z₁ / z₂ = I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l3053_305350


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l3053_305313

theorem discounted_price_calculation (original_price discount_percentage : ℝ) :
  original_price = 600 ∧ discount_percentage = 20 →
  original_price * (1 - discount_percentage / 100) = 480 := by
sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l3053_305313


namespace NUMINAMATH_CALUDE_diamonds_G15_l3053_305342

/-- The number of diamonds in the nth figure of the sequence -/
def diamonds (n : ℕ) : ℕ :=
  3 * n^2 - 3 * n + 1

/-- The sequence G is constructed such that for n ≥ 3, 
    Gₙ is surrounded by a hexagon with n-1 diamonds on each of its 6 sides -/
axiom sequence_construction (n : ℕ) (h : n ≥ 3) :
  diamonds n = diamonds (n-1) + 6 * (n-1)

/-- G₁ has 1 diamond -/
axiom G1_diamonds : diamonds 1 = 1

/-- The number of diamonds in G₁₅ is 631 -/
theorem diamonds_G15 : diamonds 15 = 631 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_G15_l3053_305342


namespace NUMINAMATH_CALUDE_train_journey_time_l3053_305302

/-- Proves that given a train moving at 6/7 of its usual speed and arriving 30 minutes late, 
    the usual time for the train to complete the journey is 3 hours. -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 → usual_time > 0 →
  (6 / 7 * usual_speed) * (usual_time + 1 / 2) = usual_speed * usual_time →
  usual_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l3053_305302


namespace NUMINAMATH_CALUDE_train_passenger_count_l3053_305330

theorem train_passenger_count (round_trips : ℕ) (return_passengers : ℕ) (total_passengers : ℕ) :
  round_trips = 4 →
  return_passengers = 60 →
  total_passengers = 640 →
  ∃ (one_way_passengers : ℕ),
    one_way_passengers = 100 ∧
    total_passengers = round_trips * (one_way_passengers + return_passengers) :=
by sorry

end NUMINAMATH_CALUDE_train_passenger_count_l3053_305330


namespace NUMINAMATH_CALUDE_line_circle_intersection_and_dot_product_range_l3053_305374

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  m * x - y + 1 - m = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + (y-1)^2 = 5

-- Define points D and E
def point_D : ℝ × ℝ := (-2, 0)
def point_E : ℝ × ℝ := (2, 0)

-- Define the condition for P being inside C
def inside_circle (x y : ℝ) : Prop :=
  x^2 + (y-1)^2 < 5

-- Define the geometric sequence condition
def geometric_sequence (x y : ℝ) : Prop :=
  ((x+2)^2 + y^2) * ((x-2)^2 + y^2) = (x^2 + y^2)^2

-- Theorem statement
theorem line_circle_intersection_and_dot_product_range :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), line_l m x y ∧ circle_C x y) ∧
  (∀ (x y : ℝ), 
    inside_circle x y → 
    geometric_sequence x y → 
    -2 ≤ ((x+2)*(-x+2) + y*(-y)) ∧ 
    ((x+2)*(-x+2) + y*(-y)) < 1 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_and_dot_product_range_l3053_305374


namespace NUMINAMATH_CALUDE_figures_per_shelf_l3053_305391

theorem figures_per_shelf (total_figures : ℕ) (num_shelves : ℕ) 
  (h1 : total_figures = 64) (h2 : num_shelves = 8) :
  total_figures / num_shelves = 8 := by
  sorry

end NUMINAMATH_CALUDE_figures_per_shelf_l3053_305391


namespace NUMINAMATH_CALUDE_thief_speed_l3053_305365

/-- The speed of a thief given chase conditions -/
theorem thief_speed (initial_distance : ℝ) (policeman_speed : ℝ) (thief_distance : ℝ) : 
  initial_distance = 0.2 →
  policeman_speed = 10 →
  thief_distance = 0.8 →
  ∃ (thief_speed : ℝ), 
    thief_speed = 8 ∧ 
    (initial_distance + thief_distance) / policeman_speed = thief_distance / thief_speed :=
by sorry

end NUMINAMATH_CALUDE_thief_speed_l3053_305365


namespace NUMINAMATH_CALUDE_bank_balance_after_two_years_l3053_305310

/-- The amount of money in a bank account after a given number of years,
    given an initial amount and an annual interest rate. -/
def bank_balance (initial_amount : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + interest_rate) ^ years

/-- Theorem stating that $100 invested for 2 years at 10% annual interest results in $121 -/
theorem bank_balance_after_two_years :
  bank_balance 100 0.1 2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_bank_balance_after_two_years_l3053_305310


namespace NUMINAMATH_CALUDE_inequality_preservation_l3053_305358

theorem inequality_preservation (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3053_305358


namespace NUMINAMATH_CALUDE_range_of_cosine_composition_l3053_305398

theorem range_of_cosine_composition (x : ℝ) :
  0.5 ≤ Real.cos ((π / 9) * (Real.cos (2 * x) - 2 * Real.sin x)) ∧
  Real.cos ((π / 9) * (Real.cos (2 * x) - 2 * Real.sin x)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_cosine_composition_l3053_305398


namespace NUMINAMATH_CALUDE_electric_distance_average_costs_annual_mileage_threshold_l3053_305325

-- Define variables and constants
variable (x : ℝ) -- Average charging cost per km for electric vehicle
def fuel_cost_diff : ℝ := 0.6 -- Difference in cost per km between fuel and electric
def charging_cost : ℝ := 300 -- Charging cost for electric vehicle
def refueling_cost : ℝ := 300 -- Refueling cost for fuel vehicle
def distance_ratio : ℝ := 4 -- Ratio of electric vehicle distance to fuel vehicle distance
def other_cost_fuel : ℝ := 4800 -- Other annual costs for fuel vehicle
def other_cost_electric : ℝ := 7800 -- Other annual costs for electric vehicle

-- Theorem statements
theorem electric_distance (hx : x > 0) : 
  (charging_cost : ℝ) / x = 300 / x :=
sorry

theorem average_costs (hx : x > 0) : 
  x = 0.2 ∧ x + fuel_cost_diff = 0.8 :=
sorry

theorem annual_mileage_threshold (y : ℝ) :
  0.2 * y + other_cost_electric < 0.8 * y + other_cost_fuel ↔ y > 5000 :=
sorry

end NUMINAMATH_CALUDE_electric_distance_average_costs_annual_mileage_threshold_l3053_305325


namespace NUMINAMATH_CALUDE_sequence_minimum_l3053_305320

/-- Given a sequence {a_n} satisfying the conditions:
    a_1 = p, a_2 = p + 1, and a_{n+2} - 2a_{n+1} + a_n = n - 20,
    where p is a real number and n is a positive integer,
    prove that a_n is minimized when n = 40. -/
theorem sequence_minimum (p : ℝ) : 
  ∃ (a : ℕ → ℝ), 
    (a 1 = p) ∧ 
    (a 2 = p + 1) ∧ 
    (∀ n : ℕ, n ≥ 1 → a (n + 2) - 2 * a (n + 1) + a n = n - 20) ∧
    (∀ n : ℕ, n ≥ 1 → a 40 ≤ a n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_minimum_l3053_305320


namespace NUMINAMATH_CALUDE_union_S_T_l3053_305338

def U : Finset Nat := {1,2,3,4,5,6}
def S : Finset Nat := {1,3,5}
def T : Finset Nat := {2,3,4,5}

theorem union_S_T : S ∪ T = {1,2,3,4,5} := by sorry

end NUMINAMATH_CALUDE_union_S_T_l3053_305338


namespace NUMINAMATH_CALUDE_smallest_term_of_sequence_l3053_305306

def a (n : ℕ+) : ℤ := n^2 - 9*n - 100

theorem smallest_term_of_sequence (n : ℕ+) :
  ∃ m : ℕ+, (m = 4 ∨ m = 5) ∧ ∀ k : ℕ+, a m ≤ a k :=
sorry

end NUMINAMATH_CALUDE_smallest_term_of_sequence_l3053_305306


namespace NUMINAMATH_CALUDE_finite_perfect_squares_l3053_305375

/-- For positive integers a and b, the set of integers n for which both an^2 + b and a(n+1)^2 + b are perfect squares is finite -/
theorem finite_perfect_squares (a b : ℕ+) :
  {n : ℤ | ∃ x y : ℤ, (a : ℤ) * n^2 + (b : ℤ) = x^2 ∧ (a : ℤ) * (n + 1)^2 + (b : ℤ) = y^2}.Finite :=
by sorry

end NUMINAMATH_CALUDE_finite_perfect_squares_l3053_305375


namespace NUMINAMATH_CALUDE_third_roll_six_prob_l3053_305318

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1/6
def biased_die_six_prob : ℚ := 2/3
def biased_die_other_prob : ℚ := 1/15

-- Define the probability of choosing each die
def die_choice_prob : ℚ := 1/2

-- Define the event of rolling two sixes
def two_sixes_prob (die_prob : ℚ) : ℚ := die_prob * die_prob

-- Define the total probability of rolling two sixes
def total_two_sixes_prob : ℚ := 
  die_choice_prob * two_sixes_prob fair_die_prob + 
  die_choice_prob * two_sixes_prob biased_die_six_prob

-- Define the conditional probability of choosing each die given two sixes
def fair_die_given_two_sixes : ℚ := 
  (two_sixes_prob fair_die_prob * die_choice_prob) / total_two_sixes_prob

def biased_die_given_two_sixes : ℚ := 
  (two_sixes_prob biased_die_six_prob * die_choice_prob) / total_two_sixes_prob

-- Theorem statement
theorem third_roll_six_prob : 
  fair_die_prob * fair_die_given_two_sixes + 
  biased_die_six_prob * biased_die_given_two_sixes = 65/102 := by
  sorry

end NUMINAMATH_CALUDE_third_roll_six_prob_l3053_305318


namespace NUMINAMATH_CALUDE_sin_330_degrees_l3053_305382

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l3053_305382


namespace NUMINAMATH_CALUDE_expression_value_l3053_305326

theorem expression_value (a b c : ℤ) 
  (eq1 : (25 : ℝ) ^ a * 5 ^ (2 * b) = 5 ^ 6)
  (eq2 : (4 : ℝ) ^ b / 4 ^ c = 4) : 
  a ^ 2 + a * b + 3 * c = 6 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3053_305326


namespace NUMINAMATH_CALUDE_circle_diameter_l3053_305385

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 9 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l3053_305385


namespace NUMINAMATH_CALUDE_nested_function_evaluation_l3053_305328

noncomputable def N (x : ℝ) : ℝ := 2 * Real.sqrt x

def O (x : ℝ) : ℝ := x^3

theorem nested_function_evaluation :
  N (O (N (O (N (O 2))))) = 724 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_nested_function_evaluation_l3053_305328


namespace NUMINAMATH_CALUDE_social_dance_attendance_l3053_305396

theorem social_dance_attendance (men : ℕ) (women : ℕ) 
  (men_partners : ℕ) (women_partners : ℕ) :
  men = 15 →
  men_partners = 4 →
  women_partners = 3 →
  men * men_partners = women * women_partners →
  women = 20 := by
sorry

end NUMINAMATH_CALUDE_social_dance_attendance_l3053_305396


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l3053_305357

theorem rectangle_length_proof (area_single : ℝ) (area_overlap : ℝ) (diagonal : ℝ) :
  area_single = 48 →
  area_overlap = 72 →
  diagonal = 6 →
  ∃ (length width : ℝ),
    length * width = area_single ∧
    length = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l3053_305357


namespace NUMINAMATH_CALUDE_mike_picked_seven_apples_l3053_305304

/-- The number of apples picked by Mike, given the total number of apples and the number picked by Nancy and Keith. -/
def mike_apples (total : ℕ) (nancy : ℕ) (keith : ℕ) : ℕ :=
  total - (nancy + keith)

/-- Theorem stating that Mike picked 7 apples given the problem conditions. -/
theorem mike_picked_seven_apples :
  mike_apples 16 3 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_seven_apples_l3053_305304


namespace NUMINAMATH_CALUDE_max_candy_leftover_l3053_305366

theorem max_candy_leftover (x : ℕ) (h : x > 0) : 
  ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l3053_305366


namespace NUMINAMATH_CALUDE_jack_house_height_correct_l3053_305397

/-- The height of Jack's house -/
def jackHouseHeight : ℝ := 49

/-- The length of the shadow cast by Jack's house -/
def jackHouseShadow : ℝ := 56

/-- The height of the tree -/
def treeHeight : ℝ := 21

/-- The length of the shadow cast by the tree -/
def treeShadow : ℝ := 24

/-- Theorem stating that the calculated height of Jack's house is correct -/
theorem jack_house_height_correct :
  jackHouseHeight = (jackHouseShadow * treeHeight) / treeShadow :=
by sorry

end NUMINAMATH_CALUDE_jack_house_height_correct_l3053_305397


namespace NUMINAMATH_CALUDE_inequality_solution_l3053_305323

theorem inequality_solution : 
  {x : ℕ | 3 * x - 2 < 7} = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3053_305323


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3053_305308

theorem min_distance_to_line : 
  ∀ m n : ℝ, 
  (4 * m - 3 * n - 5 * Real.sqrt 2 = 0) → 
  (∀ x y : ℝ, 4 * x - 3 * y - 5 * Real.sqrt 2 = 0 → m^2 + n^2 ≤ x^2 + y^2) → 
  m^2 + n^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3053_305308


namespace NUMINAMATH_CALUDE_quadratic_function_bound_l3053_305336

theorem quadratic_function_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, a * x - b * x^2 ≤ 1) → a ≤ 2 * Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bound_l3053_305336


namespace NUMINAMATH_CALUDE_arctan_arcsin_sum_equals_pi_l3053_305356

theorem arctan_arcsin_sum_equals_pi (x : ℝ) (h : x > 1) :
  2 * Real.arctan x + Real.arcsin (2 * x / (1 + x^2)) = π := by
sorry

end NUMINAMATH_CALUDE_arctan_arcsin_sum_equals_pi_l3053_305356


namespace NUMINAMATH_CALUDE_paperclip_production_l3053_305386

theorem paperclip_production (machines_base : ℕ) (paperclips_per_minute : ℕ) (machines : ℕ) (minutes : ℕ) :
  machines_base = 8 →
  paperclips_per_minute = 560 →
  machines = 18 →
  minutes = 6 →
  (machines * paperclips_per_minute * minutes) / machines_base = 7560 := by
  sorry

end NUMINAMATH_CALUDE_paperclip_production_l3053_305386


namespace NUMINAMATH_CALUDE_homework_challenge_l3053_305361

/-- Calculates the number of assignments needed for a given number of points -/
def assignments_for_points (points : ℕ) : ℕ :=
  if points ≤ 10 then points
  else if points ≤ 20 then 10 + 2 * (points - 10)
  else 30 + 3 * (points - 20)

/-- The homework challenge theorem -/
theorem homework_challenge :
  assignments_for_points 30 = 60 := by
sorry

#eval assignments_for_points 30

end NUMINAMATH_CALUDE_homework_challenge_l3053_305361


namespace NUMINAMATH_CALUDE_paths_4x3_grid_l3053_305376

/-- The number of unique paths in a grid -/
def grid_paths (m n : ℕ) : ℕ := (m + n).choose m

/-- Theorem: The number of unique paths in a 4x3 grid is 35 -/
theorem paths_4x3_grid : grid_paths 4 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_paths_4x3_grid_l3053_305376


namespace NUMINAMATH_CALUDE_square_construction_possible_l3053_305354

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represent a compass operation -/
inductive CompassOp
  | drawCircle (center : Point) (radius : ℝ)
  | findIntersection (c1 : Circle) (c2 : Circle)

/-- Represent a sequence of compass operations -/
def CompassConstruction := List CompassOp

/-- The center of the square -/
def O : Point := sorry

/-- One vertex of the square -/
def A : Point := sorry

/-- The radius of the circumcircle -/
def r : ℝ := sorry

/-- Check if a point is a vertex of the square -/
def isSquareVertex (p : Point) : Prop := sorry

/-- Check if a construction is valid (uses only compass operations) -/
def isValidConstruction (c : CompassConstruction) : Prop := sorry

/-- The main theorem: it's possible to construct the other vertices using only a compass -/
theorem square_construction_possible :
  ∃ (B C D : Point) (construction : CompassConstruction),
    isValidConstruction construction ∧
    isSquareVertex B ∧
    isSquareVertex C ∧
    isSquareVertex D :=
  sorry

end NUMINAMATH_CALUDE_square_construction_possible_l3053_305354


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l3053_305322

def M : ℕ := 33 * 38 * 58 * 462

/-- The sum of odd divisors of a natural number n -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- The sum of even divisors of a natural number n -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors_M :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 14 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l3053_305322


namespace NUMINAMATH_CALUDE_ratio_of_trigonometric_equation_l3053_305334

theorem ratio_of_trigonometric_equation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (a * Real.sin (π / 5) + b * Real.cos (π / 5)) / (a * Real.cos (π / 5) - b * Real.sin (π / 5)) = Real.tan (8 * π / 15)) : 
  b / a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_trigonometric_equation_l3053_305334
