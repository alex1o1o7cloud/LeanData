import Mathlib

namespace NUMINAMATH_CALUDE_scientific_notation_correct_l2884_288412

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 2000000

/-- The scientific notation representation of the original number -/
def scientific_repr : ScientificNotation := {
  coefficient := 2
  exponent := 6
  is_valid := by sorry
}

/-- Theorem stating that the scientific notation representation is correct -/
theorem scientific_notation_correct : 
  (scientific_repr.coefficient * (10 ^ scientific_repr.exponent : ℝ)) = original_number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l2884_288412


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2884_288424

/-- An arithmetic sequence with first term 5 and the sum of the 6th and 8th terms equal to 58 has a common difference of 4. -/
theorem arithmetic_sequence_common_difference : ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 5 →
  a 6 + a 8 = 58 →
  a 2 - a 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2884_288424


namespace NUMINAMATH_CALUDE_fifth_term_is_five_l2884_288468

def fibonacci_like_sequence : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci_like_sequence (n + 1) + fibonacci_like_sequence n

theorem fifth_term_is_five : fibonacci_like_sequence 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_five_l2884_288468


namespace NUMINAMATH_CALUDE_exist_six_points_similar_triangles_l2884_288413

/-- A point in a plane represented by its coordinates -/
structure Point (α : Type*) where
  x : α
  y : α

/-- A triangle represented by its three vertices -/
structure Triangle (α : Type*) where
  A : Point α
  B : Point α
  C : Point α

/-- Predicate to check if two triangles are similar -/
def similar {α : Type*} (t1 t2 : Triangle α) : Prop :=
  sorry

/-- Theorem stating the existence of six points forming similar triangles -/
theorem exist_six_points_similar_triangles :
  ∃ (X₁ X₂ Y₁ Y₂ Z₁ Z₂ : Point ℝ),
    ∀ (i j k : Fin 2),
      similar
        (Triangle.mk (if i = 0 then X₁ else X₂) (if j = 0 then Y₁ else Y₂) (if k = 0 then Z₁ else Z₂))
        (Triangle.mk X₁ Y₁ Z₁) :=
  sorry

end NUMINAMATH_CALUDE_exist_six_points_similar_triangles_l2884_288413


namespace NUMINAMATH_CALUDE_set_A_elements_l2884_288418

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | x^2 + 2*x + a = 0}

-- State the theorem
theorem set_A_elements (a : ℝ) (h : 1 ∈ A a) : A a = {-3, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_A_elements_l2884_288418


namespace NUMINAMATH_CALUDE_base12_addition_correct_l2884_288426

/-- Converts a base 12 number represented as a list of digits to its decimal (base 10) equivalent -/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 12 + d) 0

/-- Converts a decimal (base 10) number to its base 12 representation as a list of digits -/
def decimalToBase12 (n : Nat) : List Nat :=
  if n < 12 then [n]
  else (n % 12) :: decimalToBase12 (n / 12)

/-- Represents a number in base 12 -/
structure Base12 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 12

/-- Addition of two Base12 numbers -/
def add (a b : Base12) : Base12 :=
  let sum := base12ToDecimal a.digits + base12ToDecimal b.digits
  ⟨decimalToBase12 sum, sorry⟩

theorem base12_addition_correct :
  let a : Base12 := ⟨[3, 12, 5, 10], sorry⟩  -- 3C5A₁₂
  let b : Base12 := ⟨[4, 10, 3, 11], sorry⟩  -- 4A3B₁₂
  let result : Base12 := ⟨[8, 10, 9, 8], sorry⟩  -- 8A98₁₂
  add a b = result :=
sorry

end NUMINAMATH_CALUDE_base12_addition_correct_l2884_288426


namespace NUMINAMATH_CALUDE_ellipse_equation_equiv_standard_form_l2884_288492

/-- The equation of an ellipse given the sum of distances from any point to two fixed points -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10

/-- The standard form of an ellipse equation -/
def ellipse_standard_form (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 21 = 1

/-- Theorem stating that the ellipse equation is equivalent to its standard form -/
theorem ellipse_equation_equiv_standard_form :
  ∀ x y : ℝ, ellipse_equation x y ↔ ellipse_standard_form x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_equiv_standard_form_l2884_288492


namespace NUMINAMATH_CALUDE_f_properties_l2884_288495

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (2 * x - Real.pi) + 2 * Real.sin (x - Real.pi / 2) * Real.sin (x + Real.pi / 2)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-1 : ℝ) 1 ↔ ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi) Real.pi ∧ f x = y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2884_288495


namespace NUMINAMATH_CALUDE_jenn_savings_problem_l2884_288406

/-- Given information about Jenn's savings for a bike purchase --/
theorem jenn_savings_problem (num_jars : ℕ) (bike_cost : ℕ) (leftover : ℕ) :
  num_jars = 5 →
  bike_cost = 180 →
  leftover = 20 →
  (∃ (quarters_per_jar : ℕ),
    quarters_per_jar * num_jars = (bike_cost + leftover) * 4 ∧
    quarters_per_jar = 160) :=
by sorry

end NUMINAMATH_CALUDE_jenn_savings_problem_l2884_288406


namespace NUMINAMATH_CALUDE_simplify_expression_l2884_288490

theorem simplify_expression (a b c : ℝ) : a - (a - b + c) = b - c := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2884_288490


namespace NUMINAMATH_CALUDE_zlatoust_miass_distance_l2884_288427

theorem zlatoust_miass_distance :
  ∀ (g m k : ℝ), g > 0 → m > 0 → k > 0 →
  ∃ (x : ℝ), x > 0 ∧
  (x + 18) / k = (x - 18) / m ∧
  (x + 25) / k = (x - 25) / g ∧
  (x + 8) / m = (x - 8) / g ∧
  x = 60 :=
by sorry

end NUMINAMATH_CALUDE_zlatoust_miass_distance_l2884_288427


namespace NUMINAMATH_CALUDE_solid_identification_l2884_288421

-- Define the structure of a solid
structure Solid :=
  (faces : Nat)
  (hasParallelCongruentHexagons : Bool)
  (hasRectangularFaces : Bool)
  (hasSquareFace : Bool)
  (hasCongruentTriangles : Bool)
  (hasCommonVertex : Bool)

-- Define the types of solids
inductive SolidType
  | RegularHexagonalPrism
  | RegularSquarePyramid
  | Other

-- Function to determine the type of solid based on its structure
def identifySolid (s : Solid) : SolidType :=
  if s.faces == 8 && s.hasParallelCongruentHexagons && s.hasRectangularFaces then
    SolidType.RegularHexagonalPrism
  else if s.faces == 5 && s.hasSquareFace && s.hasCongruentTriangles && s.hasCommonVertex then
    SolidType.RegularSquarePyramid
  else
    SolidType.Other

-- Theorem stating that the given descriptions correspond to the correct solid types
theorem solid_identification :
  (∀ s : Solid, s.faces = 8 ∧ s.hasParallelCongruentHexagons ∧ s.hasRectangularFaces →
    identifySolid s = SolidType.RegularHexagonalPrism) ∧
  (∀ s : Solid, s.faces = 5 ∧ s.hasSquareFace ∧ s.hasCongruentTriangles ∧ s.hasCommonVertex →
    identifySolid s = SolidType.RegularSquarePyramid) :=
by sorry


end NUMINAMATH_CALUDE_solid_identification_l2884_288421


namespace NUMINAMATH_CALUDE_first_prime_of_nine_sum_100_l2884_288420

theorem first_prime_of_nine_sum_100 (primes : List Nat) : 
  primes.length = 9 ∧ 
  (∀ p ∈ primes, Nat.Prime p) ∧ 
  primes.sum = 100 →
  primes.head? = some 2 := by
sorry

end NUMINAMATH_CALUDE_first_prime_of_nine_sum_100_l2884_288420


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2884_288462

-- Define set A
def A : Set ℝ := {x | |x - 1| < 1}

-- Define set B
def B : Set ℝ := {x | x ≤ 2}

-- Theorem statement
theorem A_intersect_B_eq_open_interval : A ∩ B = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2884_288462


namespace NUMINAMATH_CALUDE_product_of_squares_and_prime_l2884_288422

theorem product_of_squares_and_prime : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squares_and_prime_l2884_288422


namespace NUMINAMATH_CALUDE_inequality_proof_l2884_288454

theorem inequality_proof (a b : Real) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  a^5 + b^3 + (a - b)^2 ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2884_288454


namespace NUMINAMATH_CALUDE_age_difference_l2884_288419

theorem age_difference (ana_age bonita_age : ℕ) : 
  (ana_age - 1 = 3 * (bonita_age - 1)) →  -- Last year's condition
  (ana_age = 2 * bonita_age + 3) →        -- This year's condition
  (ana_age - bonita_age = 8) :=           -- Age difference is 8
by
  sorry


end NUMINAMATH_CALUDE_age_difference_l2884_288419


namespace NUMINAMATH_CALUDE_banana_cost_l2884_288416

-- Define the rate of bananas
def banana_rate : ℚ := 3 / 4

-- Define the amount of bananas to buy
def banana_amount : ℚ := 20

-- Theorem to prove
theorem banana_cost : banana_amount * banana_rate = 15 := by
  sorry

end NUMINAMATH_CALUDE_banana_cost_l2884_288416


namespace NUMINAMATH_CALUDE_symmetric_line_l2884_288473

/-- Given a line L1 with equation x - 2y + 1 = 0 and a line of symmetry x = 1,
    the symmetric line L2 has the equation x + 2y - 3 = 0 -/
theorem symmetric_line (x y : ℝ) :
  (x - 2*y + 1 = 0) →  -- equation of L1
  (x = 1) →            -- line of symmetry
  (x + 2*y - 3 = 0)    -- equation of L2
:= by sorry

end NUMINAMATH_CALUDE_symmetric_line_l2884_288473


namespace NUMINAMATH_CALUDE_rectangle_width_equality_l2884_288440

/-- Given two rectangles of equal area, where one rectangle measures 5 inches by 24 inches
    and the other rectangle is 4 inches long, prove that the width of the second rectangle
    is 30 inches. -/
theorem rectangle_width_equality (area carol_length carol_width jordan_length : ℝ)
    (h1 : area = carol_length * carol_width)
    (h2 : carol_length = 5)
    (h3 : carol_width = 24)
    (h4 : jordan_length = 4)
    (h5 : area = jordan_length * (area / jordan_length)) :
    area / jordan_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_equality_l2884_288440


namespace NUMINAMATH_CALUDE_octagon_dual_reflection_area_l2884_288483

/-- The area of the region bounded by 8 arcs created by dual reflection over consecutive sides of a regular octagon inscribed in a circle -/
theorem octagon_dual_reflection_area (s : ℝ) (h : s = 2) :
  let r := 1 / Real.sin (22.5 * π / 180)
  let sector_area := π * r^2 / 8
  let dual_reflected_sector_area := 2 * sector_area
  8 * dual_reflected_sector_area = 2 * (1 / Real.sin (22.5 * π / 180))^2 * π :=
by sorry

end NUMINAMATH_CALUDE_octagon_dual_reflection_area_l2884_288483


namespace NUMINAMATH_CALUDE_star_expression_l2884_288484

/-- The star operation on real numbers -/
def star (a b : ℝ) : ℝ := a^2 + b^2 - a*b

/-- Theorem stating the result of (x+2y) ⋆ (y+3x) -/
theorem star_expression (x y : ℝ) : star (x + 2*y) (y + 3*x) = 7*x^2 + 3*y^2 + 3*x*y := by
  sorry

end NUMINAMATH_CALUDE_star_expression_l2884_288484


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2884_288442

theorem arithmetic_computation : -9 * 5 - (-7 * -4) + (-12 * -6) = -1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2884_288442


namespace NUMINAMATH_CALUDE_angle_triple_supplement_l2884_288486

theorem angle_triple_supplement (x : ℝ) : 
  (x = 3 * (180 - x)) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_l2884_288486


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2884_288425

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2884_288425


namespace NUMINAMATH_CALUDE_white_squares_37th_row_l2884_288459

/-- Represents the number of squares in a row of the stair-step figure -/
def num_squares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of white squares in a row of the stair-step figure -/
def num_white_squares (n : ℕ) : ℕ := (num_squares n + 1) / 2

theorem white_squares_37th_row :
  num_white_squares 37 = 37 := by sorry

end NUMINAMATH_CALUDE_white_squares_37th_row_l2884_288459


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l2884_288411

theorem tangent_point_coordinates (x y : ℝ) : 
  y = x^2 → -- Point (x, y) is on the curve y = x^2
  (2*x = 1) → -- Tangent line has slope 1 (tan(π/4) = 1)
  (x = 1/2 ∧ y = 1/4) := by -- The coordinates are (1/2, 1/4)
sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l2884_288411


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2884_288477

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3 * x + y = 18) : 
  x^2 - y^2 = -72 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2884_288477


namespace NUMINAMATH_CALUDE_vector_b_exists_l2884_288439

def a : ℝ × ℝ := (1, -2)

theorem vector_b_exists : ∃ (b : ℝ × ℝ), 
  (∃ (k : ℝ), b = k • a) ∧ 
  (‖a + b‖ < ‖a‖) ∧
  b = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_b_exists_l2884_288439


namespace NUMINAMATH_CALUDE_range_of_g_l2884_288414

def f (x : ℝ) : ℝ := 4 * x - 3

def g (x : ℝ) : ℝ := f (f (f (f (f x))))

def domain_g : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }

theorem range_of_g :
  ∀ x ∈ domain_g, 1 ≤ g x ∧ g x ≤ 2049 ∧
  ∃ y ∈ domain_g, g y = 1 ∧
  ∃ z ∈ domain_g, g z = 2049 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l2884_288414


namespace NUMINAMATH_CALUDE_factorization_equality_l2884_288447

theorem factorization_equality (a : ℝ) : -3*a + 12*a^2 - 12*a^3 = -3*a*(1-2*a)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2884_288447


namespace NUMINAMATH_CALUDE_cube_sum_equals_linear_sum_l2884_288480

theorem cube_sum_equals_linear_sum (a b : ℝ) 
  (h : (a / (1 + b)) + (b / (1 + a)) = 1) : 
  a^3 + b^3 = a + b := by sorry

end NUMINAMATH_CALUDE_cube_sum_equals_linear_sum_l2884_288480


namespace NUMINAMATH_CALUDE_unique_prime_triplet_l2884_288423

theorem unique_prime_triplet :
  ∀ a b c : ℕ+,
    (Nat.Prime (a + b * c) ∧ 
     Nat.Prime (b + a * c) ∧ 
     Nat.Prime (c + a * b)) ∧
    ((a + b * c) ∣ ((a ^ 2 + 1) * (b ^ 2 + 1) * (c ^ 2 + 1))) ∧
    ((b + a * c) ∣ ((a ^ 2 + 1) * (b ^ 2 + 1) * (c ^ 2 + 1))) ∧
    ((c + a * b) ∣ ((a ^ 2 + 1) * (b ^ 2 + 1) * (c ^ 2 + 1))) →
    a = 1 ∧ b = 1 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_triplet_l2884_288423


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l2884_288452

/-- The total cost of typing a manuscript with given revision rates and page counts. -/
theorem manuscript_typing_cost
  (initial_rate : ℕ)
  (revision_rate : ℕ)
  (total_pages : ℕ)
  (once_revised_pages : ℕ)
  (twice_revised_pages : ℕ)
  (h1 : initial_rate = 5)
  (h2 : revision_rate = 3)
  (h3 : total_pages = 100)
  (h4 : once_revised_pages = 30)
  (h5 : twice_revised_pages = 20) :
  initial_rate * total_pages +
  revision_rate * once_revised_pages +
  2 * revision_rate * twice_revised_pages = 710 := by
sorry


end NUMINAMATH_CALUDE_manuscript_typing_cost_l2884_288452


namespace NUMINAMATH_CALUDE_different_color_probability_is_two_thirds_l2884_288429

/-- The number of possible colors for the shorts -/
def shorts_colors : ℕ := 2

/-- The number of possible colors for the jersey -/
def jersey_colors : ℕ := 3

/-- The total number of possible color combinations -/
def total_combinations : ℕ := shorts_colors * jersey_colors

/-- The number of combinations where the shorts and jersey colors are different -/
def different_color_combinations : ℕ := shorts_colors * (jersey_colors - 1)

/-- The probability that the shorts will be a different color than the jersey -/
def different_color_probability : ℚ := different_color_combinations / total_combinations

theorem different_color_probability_is_two_thirds :
  different_color_probability = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_different_color_probability_is_two_thirds_l2884_288429


namespace NUMINAMATH_CALUDE_weekday_classes_count_l2884_288400

/-- Represents the Diving Club's class schedule --/
structure DivingClub where
  weekdayClasses : ℕ
  weekendClassesPerDay : ℕ
  peoplePerClass : ℕ
  totalWeeks : ℕ
  totalPeople : ℕ

/-- Calculates the total number of people who can take classes --/
def totalCapacity (club : DivingClub) : ℕ :=
  (club.weekdayClasses * club.totalWeeks + 
   club.weekendClassesPerDay * 2 * club.totalWeeks) * club.peoplePerClass

/-- Theorem stating the number of weekday classes --/
theorem weekday_classes_count (club : DivingClub) 
  (h1 : club.weekendClassesPerDay = 4)
  (h2 : club.peoplePerClass = 5)
  (h3 : club.totalWeeks = 3)
  (h4 : club.totalPeople = 270)
  (h5 : totalCapacity club = club.totalPeople) :
  club.weekdayClasses = 10 := by
  sorry

#check weekday_classes_count

end NUMINAMATH_CALUDE_weekday_classes_count_l2884_288400


namespace NUMINAMATH_CALUDE_blair_received_15_bars_l2884_288461

/-- Represents the distribution of gold bars among three people -/
structure GoldDistribution where
  total_bars : ℕ
  total_weight : ℝ
  brennan_bars : ℕ
  maya_bars : ℕ
  blair_bars : ℕ
  brennan_weight_percent : ℝ
  maya_weight_percent : ℝ

/-- The conditions of the gold bar distribution problem -/
def gold_distribution_conditions (d : GoldDistribution) : Prop :=
  d.total_bars = d.brennan_bars + d.maya_bars + d.blair_bars ∧
  d.brennan_bars = 24 ∧
  d.maya_bars = 13 ∧
  d.brennan_weight_percent = 45 ∧
  d.maya_weight_percent = 26 ∧
  d.brennan_weight_percent + d.maya_weight_percent < 100

/-- Theorem stating that Blair received 15 gold bars -/
theorem blair_received_15_bars (d : GoldDistribution) 
  (h : gold_distribution_conditions d) : d.blair_bars = 15 := by
  sorry

end NUMINAMATH_CALUDE_blair_received_15_bars_l2884_288461


namespace NUMINAMATH_CALUDE_swan_population_after_ten_years_l2884_288415

/-- The number of swans after a given number of years, given an initial population and a doubling period -/
def swan_population (initial_population : ℕ) (doubling_period : ℕ) (years : ℕ) : ℕ :=
  initial_population * 2 ^ (years / doubling_period)

/-- Theorem stating that the swan population after 10 years will be 480, given the initial conditions -/
theorem swan_population_after_ten_years :
  swan_population 15 2 10 = 480 := by
sorry

end NUMINAMATH_CALUDE_swan_population_after_ten_years_l2884_288415


namespace NUMINAMATH_CALUDE_ratio_problem_l2884_288457

theorem ratio_problem (y : ℚ) : (1 : ℚ) / 3 = y / 5 → y = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2884_288457


namespace NUMINAMATH_CALUDE_trig_expression_equality_l2884_288470

theorem trig_expression_equality : 
  (Real.sin (24 * π / 180) * Real.cos (16 * π / 180) + Real.cos (156 * π / 180) * Real.sin (66 * π / 180)) / 
  (Real.sin (28 * π / 180) * Real.cos (12 * π / 180) + Real.cos (152 * π / 180) * Real.sin (72 * π / 180)) = 
  1 / Real.sin (80 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l2884_288470


namespace NUMINAMATH_CALUDE_log_division_simplification_l2884_288403

theorem log_division_simplification :
  (Real.log 16) / (Real.log (1/16)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l2884_288403


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l2884_288401

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → x + 2*y ≥ 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l2884_288401


namespace NUMINAMATH_CALUDE_vikki_hourly_rate_l2884_288443

/-- Vikki's weekly work hours -/
def work_hours : ℝ := 42

/-- Tax deduction rate -/
def tax_rate : ℝ := 0.20

/-- Insurance deduction rate -/
def insurance_rate : ℝ := 0.05

/-- Union dues deduction -/
def union_dues : ℝ := 5

/-- Vikki's take-home pay after deductions -/
def take_home_pay : ℝ := 310

/-- Vikki's hourly pay rate -/
def hourly_rate : ℝ := 10

theorem vikki_hourly_rate :
  work_hours * hourly_rate * (1 - tax_rate - insurance_rate) - union_dues = take_home_pay :=
sorry

end NUMINAMATH_CALUDE_vikki_hourly_rate_l2884_288443


namespace NUMINAMATH_CALUDE_composite_sequence_existence_l2884_288410

theorem composite_sequence_existence (m : ℕ) (hm : m > 0) :
  ∃ n : ℕ, ∀ k : ℤ, |k| ≤ m → 
    (2^n : ℤ) + k > 0 ∧ ¬(Nat.Prime ((2^n : ℤ) + k).natAbs) :=
by sorry

end NUMINAMATH_CALUDE_composite_sequence_existence_l2884_288410


namespace NUMINAMATH_CALUDE_complement_of_angle_alpha_l2884_288472

/-- Represents an angle in degrees, minutes, and seconds -/
structure AngleDMS where
  degrees : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Calculates the complement of an angle in DMS format -/
def angleComplement (α : AngleDMS) : AngleDMS :=
  sorry

/-- The given angle α -/
def α : AngleDMS := ⟨36, 14, 25⟩

/-- Theorem: The complement of angle α is 53°45'35" -/
theorem complement_of_angle_alpha :
  angleComplement α = ⟨53, 45, 35⟩ := by sorry

end NUMINAMATH_CALUDE_complement_of_angle_alpha_l2884_288472


namespace NUMINAMATH_CALUDE_sin_2theta_value_l2884_288441

theorem sin_2theta_value (θ : Real) 
  (h1 : Real.cos (π/4 - θ) * Real.cos (π/4 + θ) = Real.sqrt 2 / 6)
  (h2 : 0 < θ) (h3 : θ < π/2) : 
  Real.sin (2*θ) = Real.sqrt 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l2884_288441


namespace NUMINAMATH_CALUDE_max_absolute_value_z_l2884_288478

theorem max_absolute_value_z (z : ℂ) (h : Complex.abs (z + 3 + 4 * I) ≤ 2) :
  ∃ (M : ℝ), M = 7 ∧ Complex.abs z ≤ M ∧ ∀ (N : ℝ), Complex.abs z ≤ N → M ≤ N :=
sorry

end NUMINAMATH_CALUDE_max_absolute_value_z_l2884_288478


namespace NUMINAMATH_CALUDE_ratio_of_sums_l2884_288405

theorem ratio_of_sums (p q r u v w : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sums_l2884_288405


namespace NUMINAMATH_CALUDE_tan_eight_pi_thirds_l2884_288474

theorem tan_eight_pi_thirds : Real.tan (8 * π / 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_eight_pi_thirds_l2884_288474


namespace NUMINAMATH_CALUDE_asterisk_equation_solution_l2884_288499

theorem asterisk_equation_solution :
  ∃! x : ℝ, x > 0 ∧ (x / 20) * (x / 180) = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_asterisk_equation_solution_l2884_288499


namespace NUMINAMATH_CALUDE_recipe_total_cups_l2884_288493

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients used in a recipe -/
def total_cups (ratio : RecipeRatio) (sugar_cups : ℕ) : ℕ :=
  let part_value := sugar_cups / ratio.sugar
  (ratio.butter + ratio.flour + ratio.sugar) * part_value

/-- Theorem: Given a recipe with ratio 2:7:5 and 10 cups of sugar, the total is 28 cups -/
theorem recipe_total_cups :
  let ratio := RecipeRatio.mk 2 7 5
  total_cups ratio 10 = 28 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l2884_288493


namespace NUMINAMATH_CALUDE_boys_without_calculators_l2884_288460

theorem boys_without_calculators (total_boys : ℕ) (total_with_calculators : ℕ) (girls_with_calculators : ℕ) : 
  total_boys = 20 →
  total_with_calculators = 30 →
  girls_with_calculators = 18 →
  total_boys - (total_with_calculators - girls_with_calculators) = 8 :=
by sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l2884_288460


namespace NUMINAMATH_CALUDE_largest_package_size_l2884_288449

theorem largest_package_size (alex_folders jamie_folders : ℕ) 
  (h1 : alex_folders = 60) (h2 : jamie_folders = 90) : 
  Nat.gcd alex_folders jamie_folders = 30 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l2884_288449


namespace NUMINAMATH_CALUDE_condition_relation_l2884_288448

theorem condition_relation (a : ℝ) :
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧
  (∃ a, a^2 + a ≥ 0 ∧ ¬(a > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relation_l2884_288448


namespace NUMINAMATH_CALUDE_remainder_theorem_l2884_288489

theorem remainder_theorem : (7 * 9^20 - 2^20) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2884_288489


namespace NUMINAMATH_CALUDE_angle_supplement_l2884_288485

theorem angle_supplement (angle : ℝ) : 
  (90 - angle = angle - 18) → (180 - angle = 126) := by
  sorry

end NUMINAMATH_CALUDE_angle_supplement_l2884_288485


namespace NUMINAMATH_CALUDE_marys_current_age_l2884_288435

theorem marys_current_age :
  ∀ (mary_age jay_age : ℕ),
    (jay_age - 5 = (mary_age - 5) + 7) →
    (jay_age + 5 = 2 * (mary_age + 5)) →
    mary_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_marys_current_age_l2884_288435


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l2884_288497

theorem sum_of_x_and_y_is_two (x y : ℝ) 
  (hx : (x - 1)^3 + 1997 * (x - 1) = -1)
  (hy : (y - 1)^3 + 1997 * (y - 1) = 1) : 
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l2884_288497


namespace NUMINAMATH_CALUDE_permutation_sum_theorem_combination_sum_theorem_l2884_288481

-- Define the permutation function
def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Define the combination function
def C (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem permutation_sum_theorem :
  A 5 1 + A 5 2 + A 5 3 + A 5 4 + A 5 5 = 325 := by sorry

theorem combination_sum_theorem (m : ℕ) (h1 : m > 1) (h2 : C 5 m = C 5 (2*m - 1)) :
  C 6 m + C 6 (m+1) + C 7 (m+2) + C 8 (m+3) = 126 := by sorry

end NUMINAMATH_CALUDE_permutation_sum_theorem_combination_sum_theorem_l2884_288481


namespace NUMINAMATH_CALUDE_Q_equals_N_l2884_288455

-- Define the sets Q and N
def Q : Set ℝ := {y | ∃ x, y = x^2 + 1}
def N : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem Q_equals_N : Q = N := by sorry

end NUMINAMATH_CALUDE_Q_equals_N_l2884_288455


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l2884_288496

theorem solution_of_linear_equation (x y m : ℝ) 
  (h1 : x = -1)
  (h2 : y = 2)
  (h3 : m * x + 2 * y = 1) :
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l2884_288496


namespace NUMINAMATH_CALUDE_average_weight_b_c_l2884_288407

/-- Given the weights of three people a, b, and c, prove that the average weight of b and c is 43 kg -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 42 →  -- The average weight of a, b, and c is 42 kg
  (a + b) / 2 = 40 →      -- The average weight of a and b is 40 kg
  b = 40 →                -- The weight of b is 40 kg
  (b + c) / 2 = 43 :=     -- The average weight of b and c is 43 kg
by sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l2884_288407


namespace NUMINAMATH_CALUDE_acid_dilution_l2884_288408

/-- Given an initial acid solution and water added to dilute it, 
    calculate the amount of water needed to reach a specific concentration. -/
theorem acid_dilution (m : ℝ) (hm : m > 50) : 
  ∃ x : ℝ, 
    (m * (m / 100) = (m + x) * ((m - 20) / 100)) → 
    x = (20 * m) / (m + 20) := by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l2884_288408


namespace NUMINAMATH_CALUDE_five_year_compound_interest_l2884_288446

/-- Calculates the final amount after compound interest --/
def compound_interest (m : ℝ) (a : ℝ) (n : ℕ) : ℝ :=
  m * (1 + a) ^ n

/-- Theorem: After 5 years of compound interest, the final amount is m(1+a)^5 --/
theorem five_year_compound_interest (m : ℝ) (a : ℝ) :
  compound_interest m a 5 = m * (1 + a) ^ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_five_year_compound_interest_l2884_288446


namespace NUMINAMATH_CALUDE_trajectory_equation_l2884_288431

/-- The ellipse on which points M and N lie -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The condition for the product of slopes of OM and ON -/
def slope_product (a b : ℝ) (m_slope n_slope : ℝ) : Prop :=
  m_slope * n_slope = b^2 / a^2

/-- The trajectory equation for point P -/
def trajectory (a b m n : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = m^2 + n^2

/-- The main theorem -/
theorem trajectory_equation (a b : ℝ) (m n : ℕ+) (x y : ℝ) :
  a > b ∧ b > 0 →
  ∃ (mx my nx ny : ℝ),
    ellipse a b mx my ∧
    ellipse a b nx ny ∧
    ∃ (m_slope n_slope : ℝ),
      slope_product a b m_slope n_slope →
      x = m * mx + n * nx ∧
      y = m * my + n * ny →
      trajectory a b m n x y :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2884_288431


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2884_288463

-- Define the hyperbola
def hyperbola (x y : ℝ) := y^2 - x^2 = 2

-- Define the foci
def foci : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

-- Define the asymptotes
def asymptotes (x y : ℝ) := x^2/3 - y^2/3 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∀ (x y : ℝ),
  (∃ (f : ℝ × ℝ), f ∈ foci) →
  (∀ (x' y' : ℝ), asymptotes x' y' ↔ asymptotes x y) →
  hyperbola x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2884_288463


namespace NUMINAMATH_CALUDE_equation_roots_imply_a_range_l2884_288450

open Real

theorem equation_roots_imply_a_range (m : ℝ) (a : ℝ) (e : ℝ) :
  m > 0 →
  e > 0 →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁ + a * (2 * x₁ + 2 * m - 4 * e * x₁) * (log (x₁ + m) - log x₁) = 0 ∧
    x₂ + a * (2 * x₂ + 2 * m - 4 * e * x₂) * (log (x₂ + m) - log x₂) = 0) →
  a > 1 / (2 * e) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_imply_a_range_l2884_288450


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l2884_288494

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  -- The length of the two equal sides
  a : ℝ
  -- The length of the base
  b : ℝ
  -- The height of the triangle
  h : ℝ
  -- The radius of the inscribed circle
  r : ℝ
  -- The perimeter is 56
  perimeter_eq : a + a + b = 56
  -- The radius is 2/7 of the height
  radius_height_ratio : r = 2/7 * h
  -- The height relates to sides by Pythagorean theorem
  height_pythagorean : h^2 + (b/2)^2 = a^2
  -- The area can be calculated using sides and radius
  area_eq : (a + a + b) * r / 2 = b * h / 2

/-- The sides of the isosceles triangle with the given properties are 16, 20, and 20 -/
theorem isosceles_triangle_sides (t : IsoscelesTriangle) : t.a = 20 ∧ t.b = 16 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_l2884_288494


namespace NUMINAMATH_CALUDE_distilled_water_remaining_l2884_288479

/-- Represents a mixed number as a pair of integers (whole, fraction) -/
structure MixedNumber where
  whole : Int
  numerator : Int
  denominator : Int
  denom_pos : denominator > 0

/-- Converts a MixedNumber to a rational number -/
def mixedToRational (m : MixedNumber) : Rat :=
  m.whole + (m.numerator : Rat) / (m.denominator : Rat)

theorem distilled_water_remaining
  (initial : MixedNumber)
  (used : MixedNumber)
  (h_initial : initial = ⟨3, 1, 2, by norm_num⟩)
  (h_used : used = ⟨1, 3, 4, by norm_num⟩) :
  mixedToRational initial - mixedToRational used = 7/4 := by
  sorry

#check distilled_water_remaining

end NUMINAMATH_CALUDE_distilled_water_remaining_l2884_288479


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l2884_288464

theorem adult_tickets_sold (adult_price student_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 6)
  (h2 : student_price = 3)
  (h3 : total_tickets = 846)
  (h4 : total_revenue = 3846) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * student_price = total_revenue ∧
    adult_tickets = 436 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l2884_288464


namespace NUMINAMATH_CALUDE_min_sum_squares_l2884_288433

theorem min_sum_squares (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2884_288433


namespace NUMINAMATH_CALUDE_f_5_equals_neg_2_l2884_288451

-- Define the inverse function f⁻¹
def f_inv (x : ℝ) : ℝ := 1 + x^2

-- State the theorem
theorem f_5_equals_neg_2 (f : ℝ → ℝ) (h1 : ∀ x < 0, f_inv (f x) = x) (h2 : ∀ y, y < 0 → f (f_inv y) = y) : 
  f 5 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_neg_2_l2884_288451


namespace NUMINAMATH_CALUDE_hou_debang_developed_alkali_process_l2884_288432

-- Define a type for scientists
inductive Scientist
| HouDebang
| HouGuangtian
| HouXianglin
| HouXueyu

-- Define a type for chemical processes
structure ChemicalProcess where
  name : String
  developer : Scientist
  developmentDate : String

-- Define the Hou's Alkali Process
def housAlkaliProcess : ChemicalProcess := {
  name := "Hou's Alkali Process",
  developer := Scientist.HouDebang,
  developmentDate := "March 1941"
}

-- Theorem statement
theorem hou_debang_developed_alkali_process :
  housAlkaliProcess.developer = Scientist.HouDebang ∧
  housAlkaliProcess.name = "Hou's Alkali Process" ∧
  housAlkaliProcess.developmentDate = "March 1941" :=
by sorry

end NUMINAMATH_CALUDE_hou_debang_developed_alkali_process_l2884_288432


namespace NUMINAMATH_CALUDE_dolphin_training_hours_l2884_288444

theorem dolphin_training_hours (num_dolphins : ℕ) (training_hours_per_dolphin : ℕ) (num_trainers : ℕ) 
  (h1 : num_dolphins = 12)
  (h2 : training_hours_per_dolphin = 5)
  (h3 : num_trainers = 4)
  (h4 : num_trainers > 0) :
  (num_dolphins * training_hours_per_dolphin) / num_trainers = 15 := by
sorry

end NUMINAMATH_CALUDE_dolphin_training_hours_l2884_288444


namespace NUMINAMATH_CALUDE_correlation_strength_linear_correlation_strength_l2884_288475

-- Define the correlation coefficient r
variable (r : ℝ) 

-- Define the absolute value of r
def abs_r := |r|

-- Define the property that r is a valid correlation coefficient
def is_valid_corr_coeff (r : ℝ) : Prop := -1 ≤ r ∧ r ≤ 1

-- Define the degree of correlation as a function of |r|
def degree_of_correlation (abs_r : ℝ) : ℝ := abs_r

-- Define the degree of linear correlation as a function of |r|
def degree_of_linear_correlation (abs_r : ℝ) : ℝ := abs_r

-- Theorem 1: As |r| increases, the degree of correlation increases
theorem correlation_strength (r1 r2 : ℝ) 
  (h1 : is_valid_corr_coeff r1) (h2 : is_valid_corr_coeff r2) :
  abs_r r1 < abs_r r2 → degree_of_correlation (abs_r r1) < degree_of_correlation (abs_r r2) :=
sorry

-- Theorem 2: As |r| approaches 1, the degree of linear correlation strengthens
theorem linear_correlation_strength (r : ℝ) (h : is_valid_corr_coeff r) :
  ∀ ε > 0, ∃ δ > 0, ∀ r', is_valid_corr_coeff r' →
    abs_r r' > 1 - δ → degree_of_linear_correlation (abs_r r') > 1 - ε :=
sorry

end NUMINAMATH_CALUDE_correlation_strength_linear_correlation_strength_l2884_288475


namespace NUMINAMATH_CALUDE_min_value_of_f_l2884_288482

def f (x a b : ℝ) : ℝ := (x + a + b) * (x + a - b) * (x - a + b) * (x - a - b)

theorem min_value_of_f (a b : ℝ) : 
  ∃ (m : ℝ), ∀ (x : ℝ), f x a b ≥ m ∧ ∃ (x₀ : ℝ), f x₀ a b = m ∧ m = -4 * a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2884_288482


namespace NUMINAMATH_CALUDE_correct_number_value_l2884_288436

theorem correct_number_value (n : ℕ) (initial_avg correct_avg wrong_value : ℚ) :
  n = 10 →
  initial_avg = 5 →
  wrong_value = 26 →
  correct_avg = 6 →
  ∃ (correct_value : ℚ),
    correct_value = wrong_value + n * (correct_avg - initial_avg) ∧
    correct_value = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_value_l2884_288436


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_2013_l2884_288453

theorem tens_digit_of_8_pow_2013 : ∃ n : ℕ, 8^2013 ≡ 88 + 100*n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_2013_l2884_288453


namespace NUMINAMATH_CALUDE_unique_divisible_by_33_l2884_288467

/-- Represents a five-digit number in the form 7n742 where n is a single digit -/
def number (n : ℕ) : ℕ := 70000 + n * 1000 + 742

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem unique_divisible_by_33 :
  (isDivisibleBy (number 1) 33) ∧
  (∀ n : ℕ, n ≤ 9 → n ≠ 1 → ¬(isDivisibleBy (number n) 33)) :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_by_33_l2884_288467


namespace NUMINAMATH_CALUDE_card_game_total_l2884_288498

theorem card_game_total (total : ℕ) (ellis orion : ℕ) : 
  ellis = (11 : ℕ) * total / 20 →
  orion = (9 : ℕ) * total / 20 →
  ellis = orion + 50 →
  total = 500 := by
sorry

end NUMINAMATH_CALUDE_card_game_total_l2884_288498


namespace NUMINAMATH_CALUDE_product_division_equality_l2884_288469

theorem product_division_equality : ∃ x : ℝ, (400 * 7000 : ℝ) = 28000 * x^1 ∧ x = 100 := by
  sorry

end NUMINAMATH_CALUDE_product_division_equality_l2884_288469


namespace NUMINAMATH_CALUDE_max_value_of_function_l2884_288476

theorem max_value_of_function (x : ℝ) (h : x > 0) : 2 - 9*x - 4/x ≤ -10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2884_288476


namespace NUMINAMATH_CALUDE_renata_final_amount_l2884_288409

/-- Calculate Renata's final amount after various transactions --/
theorem renata_final_amount 
  (initial_amount : ℕ) 
  (donation : ℕ) 
  (charity_prize : ℕ) 
  (slot_loss1 : ℕ) 
  (slot_loss2 : ℕ) 
  (slot_loss3 : ℕ) 
  (water_cost : ℕ) 
  (lottery_ticket_cost : ℕ) 
  (lottery_prize : ℕ) 
  (h1 : initial_amount = 10) 
  (h2 : donation = 4) 
  (h3 : charity_prize = 90) 
  (h4 : slot_loss1 = 50) 
  (h5 : slot_loss2 = 10) 
  (h6 : slot_loss3 = 5) 
  (h7 : water_cost = 1) 
  (h8 : lottery_ticket_cost = 1) 
  (h9 : lottery_prize = 65) : 
  initial_amount - donation + charity_prize - slot_loss1 - slot_loss2 - slot_loss3 - water_cost - lottery_ticket_cost + lottery_prize = 94 := by
  sorry

end NUMINAMATH_CALUDE_renata_final_amount_l2884_288409


namespace NUMINAMATH_CALUDE_set_forms_triangle_l2884_288428

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set of line segments (1, 2, 2) can form a triangle -/
theorem set_forms_triangle : can_form_triangle 1 2 2 := by
  sorry

end NUMINAMATH_CALUDE_set_forms_triangle_l2884_288428


namespace NUMINAMATH_CALUDE_proposition_evaluation_l2884_288466

-- Define propositions p and q
def p : Prop := ∀ x y : ℝ, x > y → -x < -y
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- State the theorem
theorem proposition_evaluation :
  (p ∧ q = False) ∧
  (p ∨ q = True) ∧
  (p ∧ (¬q) = True) ∧
  ((¬p) ∨ q = False) :=
by sorry

end NUMINAMATH_CALUDE_proposition_evaluation_l2884_288466


namespace NUMINAMATH_CALUDE_unique_two_digit_number_mod_13_l2884_288445

theorem unique_two_digit_number_mod_13 :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ (13 * n) % 100 = 42 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_mod_13_l2884_288445


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l2884_288471

theorem gcd_lcm_problem (a b : ℕ+) 
  (h1 : Nat.gcd a b = 24)
  (h2 : Nat.lcm a b = 432)
  (h3 : a = 144) :
  b = 72 := by
sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l2884_288471


namespace NUMINAMATH_CALUDE_polynomial_range_l2884_288404

noncomputable def P (p q x : ℝ) : ℝ := x^2 + p*x + q

theorem polynomial_range (p q : ℝ) :
  let rangeP := {y | ∃ x ∈ Set.Icc (-1 : ℝ) 1, P p q x = y}
  (p < -2 → rangeP = Set.Icc (1 + p + q) (1 - p + q)) ∧
  (-2 ≤ p ∧ p ≤ 0 → rangeP = Set.Icc (q - p^2/4) (1 - p + q)) ∧
  (0 ≤ p ∧ p ≤ 2 → rangeP = Set.Icc (q - p^2/4) (1 + p + q)) ∧
  (p > 2 → rangeP = Set.Icc (1 - p + q) (1 + p + q)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_range_l2884_288404


namespace NUMINAMATH_CALUDE_race_time_calculation_l2884_288465

theorem race_time_calculation (prejean_speed rickey_speed rickey_time prejean_time : ℝ) : 
  prejean_speed = (3 / 4) * rickey_speed →
  rickey_time + prejean_time = 70 →
  prejean_time = (4 / 3) * rickey_time →
  rickey_time = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_race_time_calculation_l2884_288465


namespace NUMINAMATH_CALUDE_ken_to_don_ratio_l2884_288430

-- Define the painting rates
def don_rate : ℕ := 3
def ken_rate : ℕ := don_rate + 2
def laura_rate : ℕ := 2 * ken_rate
def kim_rate : ℕ := laura_rate - 3

-- Define the total tiles painted in 15 minutes
def total_tiles : ℕ := 375

-- Theorem statement
theorem ken_to_don_ratio : 
  15 * (don_rate + ken_rate + laura_rate + kim_rate) = total_tiles →
  ken_rate / don_rate = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_ken_to_don_ratio_l2884_288430


namespace NUMINAMATH_CALUDE_sum_reciprocals_equals_two_l2884_288487

theorem sum_reciprocals_equals_two
  (a b c d : ℝ)
  (ω : ℂ)
  (ha : a ≠ -1)
  (hb : b ≠ -1)
  (hc : c ≠ -1)
  (hd : d ≠ -1)
  (hω1 : ω^4 = 1)
  (hω2 : ω ≠ 1)
  (h : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 2 / ω^2) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equals_two_l2884_288487


namespace NUMINAMATH_CALUDE_min_value_not_five_max_value_half_x_gt_y_iff_x_over_c_gt_y_over_c_min_value_eight_l2884_288402

-- Statement 1
theorem min_value_not_five : 
  ¬ (∀ x : ℝ, x + 4 / (x - 1) ≥ 5) :=
sorry

-- Statement 2
theorem max_value_half : 
  (∀ x : ℝ, x * Real.sqrt (1 - x^2) ≤ 1/2) ∧ 
  (∃ x : ℝ, x * Real.sqrt (1 - x^2) = 1/2) :=
sorry

-- Statement 3
theorem x_gt_y_iff_x_over_c_gt_y_over_c :
  ∀ x y c : ℝ, c ≠ 0 → (x > y ↔ x / c^2 > y / c^2) :=
sorry

-- Statement 4
theorem min_value_eight :
  ∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 1 →
  (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b = 1 → 2/a + 1/b ≥ 2/x + 1/y) →
  2/x + 1/y = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_not_five_max_value_half_x_gt_y_iff_x_over_c_gt_y_over_c_min_value_eight_l2884_288402


namespace NUMINAMATH_CALUDE_catch_second_messenger_first_is_optimal_l2884_288491

/-- Represents a person's position and movement --/
structure Person where
  position : ℝ
  speed : ℝ
  startTime : ℝ

/-- Represents the problem setup --/
structure ProblemSetup where
  messenger1 : Person
  messenger2 : Person
  cyclist : Person
  startPoint : ℝ

/-- Calculates the time needed for the cyclist to complete the task --/
def timeToComplete (setup : ProblemSetup) (catchSecondFirst : Bool) : ℝ :=
  sorry

/-- Theorem stating that catching the second messenger first is optimal --/
theorem catch_second_messenger_first_is_optimal (setup : ProblemSetup) :
  setup.messenger1.startTime + 0.25 = setup.messenger2.startTime →
  setup.messenger1.speed = setup.messenger2.speed →
  setup.cyclist.speed > setup.messenger1.speed →
  setup.messenger1.position < setup.startPoint →
  setup.messenger2.position > setup.startPoint →
  timeToComplete setup true ≤ timeToComplete setup false :=
sorry

end NUMINAMATH_CALUDE_catch_second_messenger_first_is_optimal_l2884_288491


namespace NUMINAMATH_CALUDE_equation_solution_l2884_288456

theorem equation_solution : 
  {x : ℝ | 3 * (x + 2) = x * (x + 2)} = {-2, 3} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2884_288456


namespace NUMINAMATH_CALUDE_function_F_solution_l2884_288488

theorem function_F_solution (F : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → F x + F ((x - 1) / x) = 1 + x) →
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → F x = (1 + x^2 - x^3) / (2 * x * (1 - x)) :=
by sorry

end NUMINAMATH_CALUDE_function_F_solution_l2884_288488


namespace NUMINAMATH_CALUDE_picnic_cost_is_60_l2884_288417

/-- Calculates the total cost of a picnic basket given the number of people and item costs. -/
def picnic_cost (num_people : ℕ) (sandwich_cost fruit_salad_cost soda_cost snack_cost : ℕ) 
  (num_sodas_per_person num_snack_bags : ℕ) : ℕ :=
  num_people * (sandwich_cost + fruit_salad_cost + num_sodas_per_person * soda_cost) + 
  num_snack_bags * snack_cost

/-- Theorem stating that the total cost of the picnic basket is $60. -/
theorem picnic_cost_is_60 : 
  picnic_cost 4 5 3 2 4 2 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_picnic_cost_is_60_l2884_288417


namespace NUMINAMATH_CALUDE_sqrt_490000_equals_700_l2884_288434

theorem sqrt_490000_equals_700 : Real.sqrt 490000 = 700 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_490000_equals_700_l2884_288434


namespace NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_mean_sum_squares_l2884_288437

theorem arithmetic_geometric_harmonic_mean_sum_squares 
  (x y z : ℝ) 
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 7)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 385.5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_mean_sum_squares_l2884_288437


namespace NUMINAMATH_CALUDE_five_students_three_teams_l2884_288438

/-- The number of ways to assign students to sports teams. -/
def assignStudentsToTeams (numStudents : ℕ) (numTeams : ℕ) : ℕ :=
  numTeams ^ numStudents

/-- Theorem stating that assigning 5 students to 3 teams results in 3^5 possibilities. -/
theorem five_students_three_teams :
  assignStudentsToTeams 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_five_students_three_teams_l2884_288438


namespace NUMINAMATH_CALUDE_duck_average_l2884_288458

theorem duck_average (adelaide ephraim kolton : ℕ) : 
  adelaide = 30 →
  adelaide = 2 * ephraim →
  kolton = ephraim + 45 →
  (adelaide + ephraim + kolton) / 3 = 35 := by
sorry

end NUMINAMATH_CALUDE_duck_average_l2884_288458
