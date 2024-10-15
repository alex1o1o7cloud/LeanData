import Mathlib

namespace NUMINAMATH_CALUDE_three_sequences_comparison_l667_66777

theorem three_sequences_comparison 
  (a b c : ℕ → ℕ) : 
  ∃ m n : ℕ, m ≠ n ∧ 
    a m ≥ a n ∧ 
    b m ≥ b n ∧ 
    c m ≥ c n :=
by sorry

end NUMINAMATH_CALUDE_three_sequences_comparison_l667_66777


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_special_properties_l667_66750

/-- A perfect power is a number of the form n^k where n and k are both natural numbers ≥ 2 -/
def is_perfect_power (x : ℕ) : Prop :=
  ∃ (n k : ℕ), n ≥ 2 ∧ k ≥ 2 ∧ x = n^k

/-- An arithmetic progression is a sequence where the difference between successive terms is constant -/
def is_arithmetic_progression (s : ℕ → ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ i, s i = a + i * d

theorem arithmetic_progression_with_special_properties :
  ∃ (s : ℕ → ℕ),
    is_arithmetic_progression s ∧
    (∀ i ∈ Finset.range 2016, ¬is_perfect_power (s i)) ∧
    is_perfect_power (Finset.prod (Finset.range 2016) s) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_special_properties_l667_66750


namespace NUMINAMATH_CALUDE_square_of_cube_zero_matrix_l667_66789

theorem square_of_cube_zero_matrix (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 3 = 0) : A ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_of_cube_zero_matrix_l667_66789


namespace NUMINAMATH_CALUDE_school_population_l667_66746

theorem school_population (total boys girls : ℕ) : 
  (total = boys + girls) →
  (boys = 50 → girls = total / 2) →
  total = 100 := by
sorry

end NUMINAMATH_CALUDE_school_population_l667_66746


namespace NUMINAMATH_CALUDE_sum_32_45_base5_l667_66723

/-- Converts a base 10 number to base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number --/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_32_45_base5 :
  toBase5 (32 + 45) = [3, 0, 2] :=
sorry

end NUMINAMATH_CALUDE_sum_32_45_base5_l667_66723


namespace NUMINAMATH_CALUDE_days_at_sisters_house_proof_l667_66791

def vacation_duration : ℕ := 3 * 7

def known_days : ℕ := 1 + 5 + 1 + 5 + 1 + 1 + 1 + 1

def days_at_sisters_house : ℕ := vacation_duration - known_days

theorem days_at_sisters_house_proof :
  days_at_sisters_house = 5 := by
  sorry

end NUMINAMATH_CALUDE_days_at_sisters_house_proof_l667_66791


namespace NUMINAMATH_CALUDE_unique_base_representation_l667_66716

theorem unique_base_representation : 
  ∃! (x y z b : ℕ), 
    x * b^2 + y * b + z = 1987 ∧
    x + y + z = 25 ∧
    x ≥ 1 ∧
    y < b ∧
    z < b ∧
    b > 1 ∧
    x = 5 ∧
    y = 9 ∧
    z = 11 ∧
    b = 19 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_representation_l667_66716


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l667_66711

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 4*x - 1 = 0) :
  2*x^4 + 8*x^3 - 4*x^2 - 8*x + 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l667_66711


namespace NUMINAMATH_CALUDE_remove_six_maximizes_probability_l667_66779

def original_list : List ℤ := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def is_valid_pair (x y : ℤ) : Prop :=
  x ∈ original_list ∧ y ∈ original_list ∧ x ≠ y ∧ x + y = 12

def count_valid_pairs (removed : ℤ) : ℕ :=
  (original_list.filter (λ x => x ≠ removed)).length.choose 2

theorem remove_six_maximizes_probability :
  ∀ n ∈ original_list, count_valid_pairs 6 ≥ count_valid_pairs n :=
by sorry

end NUMINAMATH_CALUDE_remove_six_maximizes_probability_l667_66779


namespace NUMINAMATH_CALUDE_even_function_condition_l667_66728

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x - a)

-- State the theorem
theorem even_function_condition (a : ℝ) : 
  (∀ x, f a x = f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_condition_l667_66728


namespace NUMINAMATH_CALUDE_ratio_a_over_4_to_b_over_3_l667_66727

theorem ratio_a_over_4_to_b_over_3 (a b c : ℝ) 
  (h1 : 3 * a^2 = 4 * b^2)
  (h2 : a * b * (c^2 + 2*c + 1) ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a = 2*c^2 + 3*c + c^(1/2))
  (h5 : b = c^2 + 5*c - c^(3/2)) :
  (a / 4) / (b / 3) = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_over_4_to_b_over_3_l667_66727


namespace NUMINAMATH_CALUDE_no_quadratic_factor_l667_66778

def p (x : ℝ) : ℝ := x^4 - 6*x^2 + 25

def q₁ (x : ℝ) : ℝ := x^2 - 3*x + 4
def q₂ (x : ℝ) : ℝ := x^2 - 4
def q₃ (x : ℝ) : ℝ := x^2 + 3
def q₄ (x : ℝ) : ℝ := x^2 + 3*x - 4

theorem no_quadratic_factor :
  (∀ x, p x ≠ 0 → q₁ x ≠ 0) ∧
  (∀ x, p x ≠ 0 → q₂ x ≠ 0) ∧
  (∀ x, p x ≠ 0 → q₃ x ≠ 0) ∧
  (∀ x, p x ≠ 0 → q₄ x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_factor_l667_66778


namespace NUMINAMATH_CALUDE_square_area_ratio_l667_66734

theorem square_area_ratio (y : ℝ) (h : y > 0) : 
  (3 * y)^2 / (9 * y)^2 = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l667_66734


namespace NUMINAMATH_CALUDE_total_amount_is_15_l667_66737

-- Define the shares of w, x, and y
def w_share : ℝ := 10
def x_share : ℝ := w_share * 0.3
def y_share : ℝ := w_share * 0.2

-- Define the total amount
def total_amount : ℝ := w_share + x_share + y_share

-- Theorem statement
theorem total_amount_is_15 : total_amount = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_15_l667_66737


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l667_66738

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  altitude : ℝ
  perimeter : ℝ

/-- The area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area :
  ∀ t : IsoscelesTriangle, t.altitude = 10 ∧ t.perimeter = 40 → area t = 75 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l667_66738


namespace NUMINAMATH_CALUDE_triangle_4_6_9_l667_66758

/-- Defines whether three given lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that lengths 4, 6, and 9 can form a triangle -/
theorem triangle_4_6_9 :
  can_form_triangle 4 6 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_4_6_9_l667_66758


namespace NUMINAMATH_CALUDE_total_money_calculation_l667_66717

theorem total_money_calculation (p q r total : ℝ) 
  (h1 : r = (2/3) * total) 
  (h2 : r = 3600) : 
  total = 5400 := by
sorry

end NUMINAMATH_CALUDE_total_money_calculation_l667_66717


namespace NUMINAMATH_CALUDE_symmetry_implies_constant_l667_66726

/-- A bivariate real-coefficient polynomial -/
structure BivariatePolynomial where
  (p : ℝ → ℝ → ℝ)

/-- The property that P(X, Y) = P(X+Y, X-Y) for all real X and Y -/
def has_symmetry (P : BivariatePolynomial) : Prop :=
  ∀ (X Y : ℝ), P.p X Y = P.p (X + Y) (X - Y)

/-- Main theorem: If P has the symmetry property, then it is constant -/
theorem symmetry_implies_constant (P : BivariatePolynomial) 
  (h : has_symmetry P) : 
  ∃ (c : ℝ), ∀ (X Y : ℝ), P.p X Y = c := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_constant_l667_66726


namespace NUMINAMATH_CALUDE_sin_210_degrees_l667_66776

theorem sin_210_degrees : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l667_66776


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l667_66774

theorem stratified_sampling_sample_size 
  (ratio_old middle_aged young : ℕ) 
  (selected_middle_aged : ℕ) 
  (h1 : ratio_old = 4 ∧ middle_aged = 1 ∧ young = 5)
  (h2 : selected_middle_aged = 10) : 
  (selected_middle_aged : ℚ) / middle_aged * (ratio_old + middle_aged + young) = 100 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l667_66774


namespace NUMINAMATH_CALUDE_flour_amount_proof_l667_66748

/-- The amount of flour in the first combination -/
def flour_amount : ℝ := 17.78

/-- The cost per pound of sugar and flour -/
def cost_per_pound : ℝ := 0.45

/-- The total cost of both combinations -/
def total_cost : ℝ := 26

theorem flour_amount_proof :
  (40 * cost_per_pound + flour_amount * cost_per_pound = total_cost) ∧
  (30 * cost_per_pound + 25 * cost_per_pound = total_cost) →
  flour_amount = 17.78 := by sorry

end NUMINAMATH_CALUDE_flour_amount_proof_l667_66748


namespace NUMINAMATH_CALUDE_problem_statement_l667_66719

theorem problem_statement (a b : ℝ) (h : a - 2*b - 3 = 0) : 9 - 2*a + 4*b = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l667_66719


namespace NUMINAMATH_CALUDE_circle_tangent_radius_l667_66788

/-- Given a system of equations describing the geometry of two circles with a common tangent,
    prove that the radius r of one circle is equal to 2. -/
theorem circle_tangent_radius (a r : ℝ) : 
  ((4 - r)^2 + a^2 = (4 + r)^2) ∧ 
  (r^2 + a^2 = (8 - r)^2) → 
  r = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_radius_l667_66788


namespace NUMINAMATH_CALUDE_trig_identity_l667_66722

theorem trig_identity : 
  Real.sin (63 * π / 180) * Real.cos (18 * π / 180) + 
  Real.cos (63 * π / 180) * Real.cos (108 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l667_66722


namespace NUMINAMATH_CALUDE_det_A_formula_l667_66736

theorem det_A_formula (n : ℕ) (h : n > 2) :
  let φ : ℝ := 2 * Real.pi / n
  let A : Matrix (Fin n) (Fin n) ℝ := λ i j =>
    if i = j then 1 + Real.cos (2 * φ * j) else Real.cos (φ * (i + j))
  Matrix.det A = -n^2 / 4 + 1 := by
  sorry

end NUMINAMATH_CALUDE_det_A_formula_l667_66736


namespace NUMINAMATH_CALUDE_hyperbola_and_related_ellipse_l667_66743

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, prove its asymptotes and related ellipse equations -/
theorem hyperbola_and_related_ellipse 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_imag_axis : b = 1) 
  (h_focal_dist : 2 * Real.sqrt 3 = 2 * Real.sqrt (a^2 + b^2)) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = Real.sqrt 2 / 2 * x) ∧ 
                   (∀ x, f (-x) = -f x) ∧ 
                   (∀ x y, y = f x ∨ y = -f x → x^2/a^2 - y^2/b^2 = 1)) ∧
  (∀ x y, x^2/3 + y^2 = 1 → 
    ∃ (t : ℝ), x = Real.sqrt 3 * Real.cos t ∧ y = Real.sin t) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_and_related_ellipse_l667_66743


namespace NUMINAMATH_CALUDE_garage_bikes_l667_66787

/-- Given a number of wheels and the number of wheels required per bike, 
    calculate the number of bikes that can be assembled -/
def bikes_assembled (total_wheels : ℕ) (wheels_per_bike : ℕ) : ℕ :=
  total_wheels / wheels_per_bike

theorem garage_bikes : bikes_assembled 14 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_garage_bikes_l667_66787


namespace NUMINAMATH_CALUDE_set_union_problem_l667_66705

theorem set_union_problem (A B : Set ℝ) (m : ℝ) :
  A = {0, m} →
  B = {0, 2} →
  A ∪ B = {0, 1, 2} →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l667_66705


namespace NUMINAMATH_CALUDE_quadratic_function_specific_points_l667_66708

/-- A quadratic function passing through three specific points has a specific value for 3a - 2b + c -/
theorem quadratic_function_specific_points (a b c : ℤ) : 
  (1^2 * a + 1 * b + c = 6) → 
  ((-1)^2 * a + (-1) * b + c = 4) → 
  (0^2 * a + 0 * b + c = 3) → 
  3*a - 2*b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_specific_points_l667_66708


namespace NUMINAMATH_CALUDE_congruence_solution_and_sum_l667_66710

theorem congruence_solution_and_sum (x : ℤ) : 
  (15 * x + 3) % 21 = 9 → 
  ∃ (a m : ℤ), x % m = a ∧ 
                a < m ∧ 
                m = 7 ∧ 
                a = 6 ∧ 
                a + m = 13 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_and_sum_l667_66710


namespace NUMINAMATH_CALUDE_intersection_equals_universal_set_l667_66771

theorem intersection_equals_universal_set {α : Type*} (S A B : Set α) 
  (h_universal : ∀ x, x ∈ S) 
  (h_intersection : A ∩ B = S) : 
  A = S ∧ B = S := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_universal_set_l667_66771


namespace NUMINAMATH_CALUDE_max_sum_of_roots_l667_66745

theorem max_sum_of_roots (c b : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - c*x + b = 0 ∧ y^2 - c*y + b = 0 ∧ x - y = 1) →
  c ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_roots_l667_66745


namespace NUMINAMATH_CALUDE_lawn_care_supplies_cost_l667_66785

/-- The total cost of supplies for a lawn care company -/
theorem lawn_care_supplies_cost 
  (num_blades : ℕ) 
  (blade_cost : ℕ) 
  (string_cost : ℕ) : 
  num_blades = 4 → 
  blade_cost = 8 → 
  string_cost = 7 → 
  num_blades * blade_cost + string_cost = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_lawn_care_supplies_cost_l667_66785


namespace NUMINAMATH_CALUDE_book_thickness_theorem_l667_66765

/-- Calculates the number of pages per inch of thickness for a stack of books --/
def pages_per_inch (num_books : ℕ) (avg_pages : ℕ) (total_thickness : ℕ) : ℕ :=
  (num_books * avg_pages) / total_thickness

/-- Theorem: Given a stack of 6 books with an average of 160 pages each and a total thickness of 12 inches,
    the number of pages per inch of thickness is 80. --/
theorem book_thickness_theorem :
  pages_per_inch 6 160 12 = 80 := by
  sorry

end NUMINAMATH_CALUDE_book_thickness_theorem_l667_66765


namespace NUMINAMATH_CALUDE_eggs_needed_is_84_l667_66740

/-- Represents the number of eggs in an omelette type -/
inductive OmeletteType
| ThreeEgg
| FourEgg

/-- Represents an hour's worth of orders -/
structure HourlyOrder where
  customerCount : Nat
  omeletteType : OmeletteType

/-- Calculates the total number of eggs needed for all omelettes -/
def totalEggsNeeded (orders : List HourlyOrder) : Nat :=
  orders.foldl (fun acc order =>
    acc + order.customerCount * match order.omeletteType with
      | OmeletteType.ThreeEgg => 3
      | OmeletteType.FourEgg => 4
  ) 0

theorem eggs_needed_is_84 (orders : List HourlyOrder) 
  (h1 : orders = [
    ⟨5, OmeletteType.ThreeEgg⟩, 
    ⟨7, OmeletteType.FourEgg⟩,
    ⟨3, OmeletteType.ThreeEgg⟩,
    ⟨8, OmeletteType.FourEgg⟩
  ]) : 
  totalEggsNeeded orders = 84 := by
  sorry

#eval totalEggsNeeded [
  ⟨5, OmeletteType.ThreeEgg⟩, 
  ⟨7, OmeletteType.FourEgg⟩,
  ⟨3, OmeletteType.ThreeEgg⟩,
  ⟨8, OmeletteType.FourEgg⟩
]

end NUMINAMATH_CALUDE_eggs_needed_is_84_l667_66740


namespace NUMINAMATH_CALUDE_area_cubic_line_theorem_l667_66704

noncomputable def area_cubic_line (a b c d p q α β : ℝ) : ℝ :=
  |a| / 12 * (β - α)^4

theorem area_cubic_line_theorem (a b c d p q α β : ℝ) 
  (ha : a ≠ 0) 
  (hαβ : α ≠ β) 
  (htouch : ∀ x, a * x^3 + b * x^2 + c * x + d = p * x + q → x = α → 
    (3 * a * x^2 + 2 * b * x + c = p))
  (hintersect : a * β^3 + b * β^2 + c * β + d = p * β + q) :
  area_cubic_line a b c d p q α β = 
    ∫ x in α..β, |a * x^3 + b * x^2 + c * x + d - (p * x + q)| :=
by sorry

end NUMINAMATH_CALUDE_area_cubic_line_theorem_l667_66704


namespace NUMINAMATH_CALUDE_metal_weight_in_compound_l667_66781

/-- The molecular weight of the metal element in a compound with formula (OH)2 -/
def metal_weight (total_weight : ℝ) : ℝ :=
  total_weight - 2 * (16 + 1)

/-- Theorem: The molecular weight of the metal element in a compound with formula (OH)2
    and total molecular weight of 171 g/mol is 171 - 2 * (16 + 1) g/mol -/
theorem metal_weight_in_compound : metal_weight 171 = 137 := by
  sorry

end NUMINAMATH_CALUDE_metal_weight_in_compound_l667_66781


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l667_66798

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l667_66798


namespace NUMINAMATH_CALUDE_no_overlap_for_y_l667_66768

theorem no_overlap_for_y (y : ℝ) : 
  200 ≤ y ∧ y ≤ 300 → 
  ⌊Real.sqrt y⌋ = 16 → 
  ⌊Real.sqrt (50 * y)⌋ ≠ 226 := by
sorry

end NUMINAMATH_CALUDE_no_overlap_for_y_l667_66768


namespace NUMINAMATH_CALUDE_square_to_three_squares_l667_66718

/-- A partition of a square is a list of polygons that cover the square without overlap -/
def Partition (a : ℝ) := List (List (ℝ × ℝ))

/-- A square is a list of four points representing its vertices -/
def Square := List (ℝ × ℝ)

/-- Predicate to check if a partition is valid (covers the whole square without overlap) -/
def is_valid_partition (a : ℝ) (p : Partition a) : Prop := sorry

/-- Predicate to check if a list of points forms a square -/
def is_square (s : Square) : Prop := sorry

/-- Predicate to check if a partition can be rearranged to form given squares -/
def can_form_squares (a : ℝ) (p : Partition a) (squares : List Square) : Prop := sorry

/-- Theorem stating that a square can be cut into 4 parts to form 3 squares -/
theorem square_to_three_squares (a : ℝ) : 
  ∃ (p : Partition a) (s₁ s₂ s₃ : Square), 
    is_valid_partition a p ∧ 
    p.length = 4 ∧
    is_square s₁ ∧ is_square s₂ ∧ is_square s₃ ∧
    can_form_squares a p [s₁, s₂, s₃] := by
  sorry

end NUMINAMATH_CALUDE_square_to_three_squares_l667_66718


namespace NUMINAMATH_CALUDE_intersection_and_complement_l667_66799

def A : Set ℝ := {x : ℝ | -4 < x^2 - 5*x + 2 ∧ x^2 - 5*x + 2 < 26}
def B : Set ℝ := {x : ℝ | -x^2 + 4*x - 3 < 0}

theorem intersection_and_complement :
  (A ∩ B = {x : ℝ | (-3 < x ∧ x < 1) ∨ (3 < x ∧ x < 8)}) ∧
  (Set.compl (A ∩ B) = {x : ℝ | x ≤ -3 ∨ (1 ≤ x ∧ x ≤ 3) ∨ x ≥ 8}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_complement_l667_66799


namespace NUMINAMATH_CALUDE_water_tank_capacity_l667_66763

theorem water_tank_capacity (initial_fraction : ℚ) (added_gallons : ℕ) (total_capacity : ℕ) : 
  initial_fraction = 1/3 →
  added_gallons = 16 →
  initial_fraction * total_capacity + added_gallons = total_capacity →
  total_capacity = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l667_66763


namespace NUMINAMATH_CALUDE_rectangle_area_l667_66707

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 49 → 
  rectangle_width ^ 2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 147 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l667_66707


namespace NUMINAMATH_CALUDE_hyperbola_standard_form_l667_66783

/-- The standard form of a hyperbola with foci on the x-axis -/
def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The relationship between a, b, and c in a hyperbola -/
def hyperbola_relation (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem hyperbola_standard_form :
  ∀ (a b : ℝ),
    a > 0 →
    b > 0 →
    hyperbola_relation a b (Real.sqrt 6) →
    hyperbola_equation a b (-5) 2 →
    hyperbola_equation (Real.sqrt 5) 1 = hyperbola_equation a b :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_form_l667_66783


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l667_66759

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![2, 3; 7, -1]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![1, -5; 0, 4]
  A * B = !![2, 2; 7, -39] := by
sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l667_66759


namespace NUMINAMATH_CALUDE_ordering_abc_l667_66714

theorem ordering_abc (a b c : ℝ) (ha : a = Real.log (11/10)) (hb : b = 1/10) (hc : c = 2/21) : b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ordering_abc_l667_66714


namespace NUMINAMATH_CALUDE_smallest_valid_club_size_l667_66767

def is_valid_club_size (N : ℕ) : Prop :=
  N < 50 ∧
  ((N - 5) % 6 = 0 ∨ (N - 5) % 7 = 0) ∧
  N % 8 = 7

theorem smallest_valid_club_size :
  ∀ n : ℕ, is_valid_club_size n → n ≥ 47 :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_club_size_l667_66767


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l667_66769

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℝ := 5 * n + 4 * n^3

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℝ := S r - S (r - 1)

theorem arithmetic_progression_rth_term (r : ℕ) :
  a r = 12 * r^2 - 12 * r + 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l667_66769


namespace NUMINAMATH_CALUDE_inequality_proof_l667_66773

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l667_66773


namespace NUMINAMATH_CALUDE_four_digit_integer_problem_l667_66757

theorem four_digit_integer_problem (a b c d : ℕ) : 
  a ≠ 0 ∧ 
  a + b + c + d = 16 ∧ 
  b + c = 11 ∧ 
  a - d = 3 ∧ 
  (1000 * a + 100 * b + 10 * c + d) % 11 = 0 →
  1000 * a + 100 * b + 10 * c + d = 4714 := by
sorry

end NUMINAMATH_CALUDE_four_digit_integer_problem_l667_66757


namespace NUMINAMATH_CALUDE_divided_triangle_perimeter_l667_66712

/-- Represents a triangle divided into smaller triangles -/
structure DividedTriangle where
  large_perimeter : ℝ
  num_small_triangles : ℕ
  small_perimeter : ℝ

/-- Theorem stating the relationship between the perimeters of the large and small triangles -/
theorem divided_triangle_perimeter
  (t : DividedTriangle)
  (h1 : t.large_perimeter = 120)
  (h2 : t.num_small_triangles = 9)
  (h3 : t.small_perimeter * 3 = t.large_perimeter) :
  t.small_perimeter = 40 :=
sorry

end NUMINAMATH_CALUDE_divided_triangle_perimeter_l667_66712


namespace NUMINAMATH_CALUDE_existence_of_bounded_difference_l667_66797

theorem existence_of_bounded_difference (n : ℕ) (x : Fin n → ℝ) 
  (h_n : n ≥ 3) 
  (h_pos : ∀ i, x i > 0) 
  (h_distinct : ∀ i j, i ≠ j → x i ≠ x j) : 
  ∃ i j, i ≠ j ∧ 
    0 < (x i - x j) / (1 + x i * x j) ∧ 
    (x i - x j) / (1 + x i * x j) < Real.tan (π / (2 * (n - 1))) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_bounded_difference_l667_66797


namespace NUMINAMATH_CALUDE_infinite_decimal_is_rational_l667_66775

/-- Given an infinite decimal T = 0.a₁a₂a₃..., where aₙ is the remainder when n² is divided by 10,
    prove that T is equal to 166285490 / 1111111111. -/
theorem infinite_decimal_is_rational :
  let a : ℕ → ℕ := λ n => n^2 % 10
  let T : ℝ := ∑' n, (a n : ℝ) / 10^(n + 1)
  T = 166285490 / 1111111111 :=
sorry

end NUMINAMATH_CALUDE_infinite_decimal_is_rational_l667_66775


namespace NUMINAMATH_CALUDE_two_people_two_rooms_probability_two_people_two_rooms_probability_proof_l667_66784

/-- The probability that two people randomly checking into two rooms will each occupy one room -/
theorem two_people_two_rooms_probability : ℝ :=
  1/2

/-- Proof that the probability of two people randomly checking into two rooms 
    and each occupying one room is 1/2 -/
theorem two_people_two_rooms_probability_proof : 
  two_people_two_rooms_probability = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_two_people_two_rooms_probability_two_people_two_rooms_probability_proof_l667_66784


namespace NUMINAMATH_CALUDE_student_arrangement_count_l667_66713

/-- The number of ways to arrange students into communities --/
def arrange_students (total_students : ℕ) (selected_students : ℕ) (communities : ℕ) (min_per_community : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements for the given problem --/
theorem student_arrangement_count :
  arrange_students 7 6 2 2 = 350 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l667_66713


namespace NUMINAMATH_CALUDE_highest_consecutive_number_l667_66792

theorem highest_consecutive_number (n : ℤ) (h1 : n - 3 + (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) = 33 * 7) :
  n + 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_highest_consecutive_number_l667_66792


namespace NUMINAMATH_CALUDE_log_difference_inequality_l667_66709

theorem log_difference_inequality (a b : ℝ) : 
  Real.log a - Real.log b = 3 * b - a → a > b ∧ b > 0 := by sorry

end NUMINAMATH_CALUDE_log_difference_inequality_l667_66709


namespace NUMINAMATH_CALUDE_harrys_morning_routine_time_l667_66730

def morning_routine (coffee_bagel_time : ℕ) (reading_eating_factor : ℕ) : ℕ :=
  coffee_bagel_time + reading_eating_factor * coffee_bagel_time

theorem harrys_morning_routine_time :
  morning_routine 15 2 = 45 :=
by sorry

end NUMINAMATH_CALUDE_harrys_morning_routine_time_l667_66730


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l667_66742

-- Define the function
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- State the theorem
theorem f_strictly_increasing :
  (∀ x y, x < y ∧ ((x < -1/3 ∧ y ≤ -1/3) ∨ (x ≥ 1 ∧ y > 1)) → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l667_66742


namespace NUMINAMATH_CALUDE_expression_evaluation_l667_66700

theorem expression_evaluation (a b c d : ℤ) 
  (ha : a = 10) (hb : b = 15) (hc : c = 3) (hd : d = 2) : 
  (a - (b - c + d)) - ((a - b + d) - c) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l667_66700


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_l667_66755

theorem no_solution_to_inequality : 
  ¬ ∃ x : ℝ, -2 < (x^2 - 10*x + 9) / (x^2 - 4*x + 8) ∧ (x^2 - 10*x + 9) / (x^2 - 4*x + 8) < 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_l667_66755


namespace NUMINAMATH_CALUDE_right_triangle_area_l667_66754

theorem right_triangle_area (a b : ℝ) (h_a : a = 25) (h_b : b = 20) :
  (1 / 2) * a * b = 250 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l667_66754


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_numbers_l667_66766

theorem smallest_distance_between_complex_numbers (z w : ℂ) 
  (hz : Complex.abs (z + 2 + 2*Complex.I) = 2)
  (hw : Complex.abs (w - 5 - 6*Complex.I) = 2) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 113 - 4 ∧ 
    ∀ (z' w' : ℂ), Complex.abs (z' + 2 + 2*Complex.I) = 2 →
      Complex.abs (w' - 5 - 6*Complex.I) = 2 →
      Complex.abs (z' - w') ≥ min_dist := by
  sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_numbers_l667_66766


namespace NUMINAMATH_CALUDE_monday_sales_calculation_l667_66796

def total_stock : ℕ := 1200
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135
def unsold_percentage : ℚ := 665/1000

theorem monday_sales_calculation :
  ∃ (monday_sales : ℕ),
    monday_sales = total_stock - 
      (tuesday_sales + wednesday_sales + thursday_sales + friday_sales) - 
      (unsold_percentage * total_stock).num :=
by sorry

end NUMINAMATH_CALUDE_monday_sales_calculation_l667_66796


namespace NUMINAMATH_CALUDE_typist_salary_problem_l667_66782

/-- Proves that if a salary is first increased by 10% and then decreased by 5%, 
    resulting in Rs. 4180, then the original salary must be Rs. 4000. -/
theorem typist_salary_problem (x : ℝ) : 
  (x * 1.1 * 0.95 = 4180) → x = 4000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l667_66782


namespace NUMINAMATH_CALUDE_silver_status_families_l667_66731

def fundraiser (bronze silver gold : ℕ) : ℕ := 
  25 * bronze + 50 * silver + 100 * gold

theorem silver_status_families : 
  ∃ (silver : ℕ), 
    fundraiser 10 silver 1 = 700 ∧ 
    silver = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_silver_status_families_l667_66731


namespace NUMINAMATH_CALUDE_apple_pie_cost_per_serving_l667_66721

/-- Calculates the cost per serving of an apple pie given the ingredients and their costs. -/
def cost_per_serving (granny_smith_weight : Float) (granny_smith_price : Float)
                     (gala_weight : Float) (gala_price : Float)
                     (honeycrisp_weight : Float) (honeycrisp_price : Float)
                     (pie_crust_price : Float) (lemon_price : Float) (butter_price : Float)
                     (servings : Nat) : Float :=
  let total_cost := granny_smith_weight * granny_smith_price +
                    gala_weight * gala_price +
                    honeycrisp_weight * honeycrisp_price +
                    pie_crust_price + lemon_price + butter_price
  total_cost / servings.toFloat

/-- The cost per serving of the apple pie is $1.16375. -/
theorem apple_pie_cost_per_serving :
  cost_per_serving 0.5 1.80 0.8 2.20 0.7 2.50 2.50 0.60 1.80 8 = 1.16375 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_cost_per_serving_l667_66721


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l667_66741

theorem mango_rate_calculation (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (total_paid : ℕ) :
  apple_quantity = 8 →
  apple_rate = 70 →
  mango_quantity = 9 →
  total_paid = 1055 →
  (total_paid - apple_quantity * apple_rate) / mango_quantity = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l667_66741


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l667_66772

theorem other_root_of_quadratic (x : ℚ) :
  (7 * x^2 - 3 * x = 10) ∧ (7 * (-2)^2 - 3 * (-2) = 10) →
  (7 * (5/7)^2 - 3 * (5/7) = 10) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l667_66772


namespace NUMINAMATH_CALUDE_triangle_side_length_l667_66762

theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  a = 2 →
  B = π / 3 →
  c = 3 →
  b = Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l667_66762


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l667_66724

theorem salary_reduction_percentage (S : ℝ) (P : ℝ) (h : S > 0) :
  2 * (S - (P / 100 * S)) = S → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l667_66724


namespace NUMINAMATH_CALUDE_specific_tank_insulation_cost_l667_66744

/-- The cost to insulate a rectangular tank -/
def insulation_cost (length width height cost_per_sqft : ℝ) : ℝ :=
  (2 * (length * width + length * height + width * height)) * cost_per_sqft

/-- Theorem: The cost to insulate a specific rectangular tank -/
theorem specific_tank_insulation_cost :
  insulation_cost 4 5 3 20 = 1880 := by
  sorry

end NUMINAMATH_CALUDE_specific_tank_insulation_cost_l667_66744


namespace NUMINAMATH_CALUDE_ursula_change_l667_66751

/-- Calculates the change Ursula received after buying hot dogs and salads -/
theorem ursula_change : 
  let hot_dog_price : ℚ := 3/2  -- $1.50 as a rational number
  let salad_price : ℚ := 5/2    -- $2.50 as a rational number
  let hot_dog_count : ℕ := 5
  let salad_count : ℕ := 3
  let bill_value : ℕ := 10
  let bill_count : ℕ := 2
  
  let total_cost : ℚ := hot_dog_price * hot_dog_count + salad_price * salad_count
  let total_paid : ℕ := bill_value * bill_count
  
  (total_paid : ℚ) - total_cost = 5
  := by sorry

end NUMINAMATH_CALUDE_ursula_change_l667_66751


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l667_66725

theorem contrapositive_equivalence (a b : ℝ) :
  (¬((a - b) * (a + b) = 0) → ¬(a - b = 0)) ↔
  ((a - b = 0) → ((a - b) * (a + b) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l667_66725


namespace NUMINAMATH_CALUDE_add_2687_minutes_to_7am_l667_66752

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem add_2687_minutes_to_7am (start : Time) (h : start.hours = 7 ∧ start.minutes = 0) :
  addMinutes start 2687 = { hours := 3, minutes := 47, h_valid := sorry, m_valid := sorry } :=
sorry

end NUMINAMATH_CALUDE_add_2687_minutes_to_7am_l667_66752


namespace NUMINAMATH_CALUDE_total_pepper_weight_l667_66760

-- Define the weights of green and red peppers
def green_peppers : ℝ := 0.33
def red_peppers : ℝ := 0.33

-- Theorem stating the total weight of peppers
theorem total_pepper_weight : green_peppers + red_peppers = 0.66 := by
  sorry

end NUMINAMATH_CALUDE_total_pepper_weight_l667_66760


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l667_66739

theorem sum_of_squares_of_roots (r₁ r₂ : ℝ) : 
  r₁^2 - 10*r₁ + 9 = 0 →
  r₂^2 - 10*r₂ + 9 = 0 →
  (r₁ > 5 ∨ r₂ > 5) →
  r₁^2 + r₂^2 = 82 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l667_66739


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l667_66715

/-- Given a tetrahedron with inradius R and face areas S₁, S₂, S₃, and S₄,
    its volume V is equal to (1/3)R(S₁ + S₂ + S₃ + S₄) -/
theorem tetrahedron_volume (R S₁ S₂ S₃ S₄ : ℝ) (hR : R > 0) (hS : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0) :
  ∃ V : ℝ, V = (1/3) * R * (S₁ + S₂ + S₃ + S₄) ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l667_66715


namespace NUMINAMATH_CALUDE_problem_solution_l667_66706

/-- Proposition p: x² - 4ax + 3a² < 0 -/
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

/-- Proposition q: |x - 3| < 1 -/
def q (x : ℝ) : Prop := |x - 3| < 1

theorem problem_solution :
  (∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3)) ∧
  (∀ a : ℝ, (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) ↔ (4/3 ≤ a ∧ a ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l667_66706


namespace NUMINAMATH_CALUDE_quadrilateral_vector_proof_l667_66701

-- Define the space
variable (V : Type*) [AddCommGroup V] [Module ℝ V]

-- Define the points and vectors
variable (O A B C D M N : V)
variable (a b c : V)

-- State the theorem
theorem quadrilateral_vector_proof 
  (h1 : O + a = A) 
  (h2 : O + b = B) 
  (h3 : O + c = C) 
  (h4 : ∃ t : ℝ, M = O + t • a) 
  (h5 : M - O = 2 • (A - M)) 
  (h6 : N = (1/2) • B + (1/2) • C) :
  M - N = -(2/3) • a + (1/2) • b + (1/2) • c := by sorry

end NUMINAMATH_CALUDE_quadrilateral_vector_proof_l667_66701


namespace NUMINAMATH_CALUDE_number_satisfying_equation_l667_66749

theorem number_satisfying_equation : ∃! x : ℝ, 3 * (x + 2) = 24 + x := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_equation_l667_66749


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l667_66786

theorem same_remainder_divisor : ∃ (d : ℕ), d > 0 ∧ 
  ∀ (k : ℕ), k > d → 
  (∃ (r₁ r₂ r₃ : ℕ), 
    480608 = k * r₁ + d ∧
    508811 = k * r₂ + d ∧
    723217 = k * r₃ + d) → False :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l667_66786


namespace NUMINAMATH_CALUDE_number_equation_solution_l667_66732

theorem number_equation_solution : ∃ x : ℝ, (4 * x - 7 = 13) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l667_66732


namespace NUMINAMATH_CALUDE_angle_inequality_l667_66770

theorem angle_inequality (θ : Real) (h1 : 0 ≤ θ) (h2 : θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 →
    x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ ≥ 0) →
  Real.pi / 12 ≤ θ ∧ θ ≤ 5 * Real.pi / 12 :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_l667_66770


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l667_66780

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9/17) 
  (h2 : x - y = 1/51) : 
  x^2 - y^2 = 9/867 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l667_66780


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l667_66794

theorem quadratic_inequality_solution (m n : ℝ) :
  (∀ x : ℝ, x^2 + m*x + n < 0 ↔ -1 < x ∧ x < 3) →
  m + n = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l667_66794


namespace NUMINAMATH_CALUDE_max_daily_sales_l667_66761

def P (t : ℕ) : ℝ :=
  if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else if 1 ≤ t ∧ t ≤ 24 then t + 20
  else 0

def Q (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 30 then -t + 40
  else 0

def y (t : ℕ) : ℝ := P t * Q t

theorem max_daily_sales :
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ y t = 1125) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 → y t ≤ 1125) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ y t = 1125 → t = 25) :=
by sorry

end NUMINAMATH_CALUDE_max_daily_sales_l667_66761


namespace NUMINAMATH_CALUDE_inverse_composition_problem_l667_66735

def f : Fin 6 → Fin 6
| 1 => 4
| 2 => 5
| 3 => 3
| 4 => 2
| 5 => 1
| 6 => 6

theorem inverse_composition_problem (h : Function.Bijective f) :
  (Function.invFun f) ((Function.invFun f) ((Function.invFun f) 2)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_problem_l667_66735


namespace NUMINAMATH_CALUDE_min_distance_to_line_l667_66747

/-- The minimum distance from the origin to the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧
  ∀ (x y : ℝ), x + y - 4 = 0 → Real.sqrt (x^2 + y^2) ≥ d :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l667_66747


namespace NUMINAMATH_CALUDE_cuboid_inequality_l667_66790

theorem cuboid_inequality (x y z : ℝ) (hxy : x < y) (hyz : y < z)
  (p : ℝ) (hp : p = 4 * (x + y + z))
  (s : ℝ) (hs : s = 2 * (x*y + y*z + z*x))
  (d : ℝ) (hd : d = Real.sqrt (x^2 + y^2 + z^2)) :
  x < (1/3) * ((1/4) * p - Real.sqrt (d^2 - (1/2) * s)) ∧
  z > (1/3) * ((1/4) * p + Real.sqrt (d^2 - (1/2) * s)) := by
sorry

end NUMINAMATH_CALUDE_cuboid_inequality_l667_66790


namespace NUMINAMATH_CALUDE_product_remainder_l667_66703

theorem product_remainder (a b : ℕ) :
  a % 3 = 1 → b % 3 = 2 → (a * b) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l667_66703


namespace NUMINAMATH_CALUDE_reappearance_line_l667_66793

def letter_cycle : List Char := ['B', 'K', 'I', 'G', 'N', 'O']
def digit_cycle : List Nat := [3, 0, 7, 2, 0]

theorem reappearance_line : 
  Nat.lcm (List.length letter_cycle) (List.length digit_cycle) = 30 := by
  sorry

end NUMINAMATH_CALUDE_reappearance_line_l667_66793


namespace NUMINAMATH_CALUDE_max_expression_value_l667_66702

def max_expression (a b c d : ℕ) : ℕ := c * a^(b + d)

theorem max_expression_value :
  ∃ (a b c d : ℕ),
    a ∈ ({0, 1, 2, 3, 4} : Set ℕ) ∧
    b ∈ ({0, 1, 2, 3, 4} : Set ℕ) ∧
    c ∈ ({0, 1, 2, 3, 4} : Set ℕ) ∧
    d ∈ ({0, 1, 2, 3, 4} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    max_expression a b c d = 1024 ∧
    ∀ (w x y z : ℕ),
      w ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
      x ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
      y ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
      z ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
      w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
      max_expression w x y z ≤ 1024 :=
by
  sorry

end NUMINAMATH_CALUDE_max_expression_value_l667_66702


namespace NUMINAMATH_CALUDE_trapezoid_segment_equality_l667_66720

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid -/
structure Trapezoid where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Checks if two line segments are parallel -/
def areParallel (p1 p2 p3 p4 : Point2D) : Prop :=
  (p2.y - p1.y) * (p4.x - p3.x) = (p2.x - p1.x) * (p4.y - p3.y)

/-- Checks if a point is on a line segment -/
def isOnSegment (p q r : Point2D) : Prop :=
  q.x <= max p.x r.x ∧ q.x >= min p.x r.x ∧
  q.y <= max p.y r.y ∧ q.y >= min p.y r.y

/-- Represents the intersection of two line segments -/
def intersect (p1 p2 p3 p4 : Point2D) : Option Point2D :=
  sorry -- Implementation omitted for brevity

theorem trapezoid_segment_equality (ABCD : Trapezoid) (M N K L : Point2D) :
  areParallel ABCD.B ABCD.C M N →
  isOnSegment ABCD.A ABCD.B M →
  isOnSegment ABCD.C ABCD.D N →
  intersect M N ABCD.A ABCD.C = some K →
  intersect M N ABCD.B ABCD.D = some L →
  (K.x - M.x)^2 + (K.y - M.y)^2 = (L.x - N.x)^2 + (L.y - N.y)^2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_equality_l667_66720


namespace NUMINAMATH_CALUDE_polynomial_factorization_l667_66733

theorem polynomial_factorization (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x - 35 = (x - 7)*(x + 5)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l667_66733


namespace NUMINAMATH_CALUDE_original_house_price_l667_66729

/-- Given a house that increases in value by 25% and is then sold to cover 25% of a $500,000 new house,
    prove that the original purchase price of the first house was $100,000. -/
theorem original_house_price (original_price : ℝ) : 
  (original_price * 1.25 = 500000 * 0.25) → original_price = 100000 := by
  sorry

end NUMINAMATH_CALUDE_original_house_price_l667_66729


namespace NUMINAMATH_CALUDE_rons_height_l667_66753

theorem rons_height (dean_height ron_height water_depth : ℝ) : 
  water_depth = 2 * dean_height →
  dean_height = ron_height - 8 →
  water_depth = 12 →
  ron_height = 14 := by
  sorry

end NUMINAMATH_CALUDE_rons_height_l667_66753


namespace NUMINAMATH_CALUDE_quadratic_integer_values_iff_coefficients_integer_l667_66795

theorem quadratic_integer_values_iff_coefficients_integer (a b c : ℚ) :
  (∀ x : ℤ, ∃ n : ℤ, a * x^2 + b * x + c = n) ↔
  (∃ k : ℤ, 2 * a = k) ∧ (∃ m : ℤ, a + b = m) ∧ (∃ p : ℤ, c = p) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_values_iff_coefficients_integer_l667_66795


namespace NUMINAMATH_CALUDE_means_inequality_l667_66756

theorem means_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (a + b + c) / 3 > (a * b * c) ^ (1/3) ∧ (a * b * c) ^ (1/3) > 3 * a * b * c / (a * b + b * c + c * a) :=
sorry

end NUMINAMATH_CALUDE_means_inequality_l667_66756


namespace NUMINAMATH_CALUDE_lunch_scores_pigeonhole_l667_66764

theorem lunch_scores_pigeonhole (n : ℕ) (scores : Fin n → ℕ) 
  (h1 : ∀ i : Fin n, scores i < n) : 
  ∃ i j : Fin n, i ≠ j ∧ scores i = scores j :=
by
  sorry

end NUMINAMATH_CALUDE_lunch_scores_pigeonhole_l667_66764
