import Mathlib

namespace NUMINAMATH_CALUDE_legs_minus_twice_heads_l744_74422

/-- Represents the number of legs for each animal type -/
def legs_per_animal : Fin 3 → ℕ
  | 0 => 2  -- Ducks
  | 1 => 4  -- Cows
  | 2 => 4  -- Buffaloes

/-- The group of animals -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ
  buffaloes : ℕ
  buffalo_count_eq : buffaloes = 24

/-- Total number of heads in the group -/
def total_heads (g : AnimalGroup) : ℕ :=
  g.ducks + g.cows + g.buffaloes

/-- Total number of legs in the group -/
def total_legs (g : AnimalGroup) : ℕ :=
  g.ducks * legs_per_animal 0 + g.cows * legs_per_animal 1 + g.buffaloes * legs_per_animal 2

/-- The statement to be proven -/
theorem legs_minus_twice_heads (g : AnimalGroup) :
  total_legs g > 2 * total_heads g →
  total_legs g - 2 * total_heads g = 2 * g.cows + 48 :=
sorry

end NUMINAMATH_CALUDE_legs_minus_twice_heads_l744_74422


namespace NUMINAMATH_CALUDE_prove_c_value_l744_74426

theorem prove_c_value (c : ℕ) : 
  (5 ^ 5) * (9 ^ 3) = c * (15 ^ 5) → c = 3 → c = 3 := by
  sorry

end NUMINAMATH_CALUDE_prove_c_value_l744_74426


namespace NUMINAMATH_CALUDE_wage_percentage_proof_l744_74404

def company_finances (revenue : ℝ) (num_employees : ℕ) (tax_rate : ℝ) 
  (marketing_rate : ℝ) (operational_rate : ℝ) (employee_wage : ℝ) : Prop :=
  let after_tax := revenue * (1 - tax_rate)
  let after_marketing := after_tax * (1 - marketing_rate)
  let after_operational := after_marketing * (1 - operational_rate)
  let total_wages := num_employees * employee_wage
  total_wages / after_operational = 0.15

theorem wage_percentage_proof :
  company_finances 400000 10 0.10 0.05 0.20 4104 := by
  sorry

end NUMINAMATH_CALUDE_wage_percentage_proof_l744_74404


namespace NUMINAMATH_CALUDE_triangle_square_perimeter_ratio_l744_74469

theorem triangle_square_perimeter_ratio : 
  let square_side : ℝ := 4
  let square_perimeter : ℝ := 4 * square_side
  let triangle_leg : ℝ := square_side
  let triangle_hypotenuse : ℝ := square_side * Real.sqrt 2
  let triangle_perimeter : ℝ := 2 * triangle_leg + triangle_hypotenuse
  triangle_perimeter / square_perimeter = 1/2 + Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_square_perimeter_ratio_l744_74469


namespace NUMINAMATH_CALUDE_min_swaps_for_initial_number_l744_74492

def initial_number : ℕ := 9072543681

def is_divisible_by_99 (n : ℕ) : Prop :=
  n % 99 = 0

def adjacent_swap (n : ℕ) (i : ℕ) : ℕ :=
  sorry

def min_swaps_to_divisible_by_99 (n : ℕ) : ℕ :=
  sorry

theorem min_swaps_for_initial_number :
  min_swaps_to_divisible_by_99 initial_number = 2 :=
sorry

end NUMINAMATH_CALUDE_min_swaps_for_initial_number_l744_74492


namespace NUMINAMATH_CALUDE_trigonometric_fraction_equals_one_l744_74401

theorem trigonometric_fraction_equals_one : 
  (Real.sin (22 * π / 180) * Real.cos (8 * π / 180) + 
   Real.cos (158 * π / 180) * Real.cos (98 * π / 180)) / 
  (Real.sin (23 * π / 180) * Real.cos (7 * π / 180) + 
   Real.cos (157 * π / 180) * Real.cos (97 * π / 180)) = 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_fraction_equals_one_l744_74401


namespace NUMINAMATH_CALUDE_thirty_in_base_6_l744_74451

/-- Converts a decimal number to its base 6 representation -/
def to_base_6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Converts a list of digits in base 6 to a natural number -/
def from_base_6 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 6 + d) 0

theorem thirty_in_base_6 :
  to_base_6 30 = [5, 0] ∧ from_base_6 [5, 0] = 30 :=
sorry

end NUMINAMATH_CALUDE_thirty_in_base_6_l744_74451


namespace NUMINAMATH_CALUDE_geometric_means_equality_l744_74499

/-- Represents a quadrilateral with sides a, b, c, and d -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents the geometric means of sides in the quadrilateral -/
structure GeometricMeans (q : Quadrilateral) where
  k : ℝ
  l : ℝ
  m : ℝ
  n : ℝ
  hk : k^2 = q.a * q.d
  hl : l^2 = q.a * q.d
  hm : m^2 = q.b * q.c
  hn : n^2 = q.b * q.c

/-- The main theorem stating the condition for KL = MN -/
theorem geometric_means_equality (q : Quadrilateral) (g : GeometricMeans q) :
  (g.k - g.l)^2 = (g.m - g.n)^2 ↔ (q.a + q.b = q.c + q.d ∨ q.a + q.c = q.b + q.d) :=
by sorry


end NUMINAMATH_CALUDE_geometric_means_equality_l744_74499


namespace NUMINAMATH_CALUDE_function_inequality_implies_squares_inequality_l744_74415

theorem function_inequality_implies_squares_inequality 
  (a b A B : ℝ) 
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) : 
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_squares_inequality_l744_74415


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l744_74475

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 2 / 1)
  (hdb : d / b = 2 / 5) :
  a / c = 25 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l744_74475


namespace NUMINAMATH_CALUDE_equation_solution_l744_74417

theorem equation_solution (x : ℝ) : (x - 2) * (x - 3) = 0 ↔ x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l744_74417


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l744_74411

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l744_74411


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l744_74429

/-- Reflects a point (x, y) about the line y = -x --/
def reflect_about_negative_diagonal (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), -(p.1))

/-- The original center of the circle --/
def original_center : ℝ × ℝ := (4, -3)

/-- The expected reflected center of the circle --/
def expected_reflected_center : ℝ × ℝ := (3, -4)

theorem reflection_of_circle_center :
  reflect_about_negative_diagonal original_center = expected_reflected_center := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l744_74429


namespace NUMINAMATH_CALUDE_inequality_proof_l744_74489

theorem inequality_proof (a b c : ℝ) 
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  2 * (a + b + c) * (a^2 + b^2 + c^2) / 3 > a^3 + b^3 + c^3 + a*b*c :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l744_74489


namespace NUMINAMATH_CALUDE_cos_A_eq_one_l744_74432

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_A_eq_C (q : Quadrilateral) : Prop :=
  sorry -- ∠A = ∠C

def side_AB_eq_240 (q : Quadrilateral) : ℝ :=
  sorry -- Distance between A and B

def side_CD_eq_240 (q : Quadrilateral) : ℝ :=
  sorry -- Distance between C and D

def side_AD_ne_BC (q : Quadrilateral) : Prop :=
  sorry -- AD ≠ BC

def perimeter_eq_960 (q : Quadrilateral) : ℝ :=
  sorry -- Perimeter of ABCD

-- Theorem statement
theorem cos_A_eq_one (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_angle : angle_A_eq_C q)
  (h_AB : side_AB_eq_240 q = 240)
  (h_CD : side_CD_eq_240 q = 240)
  (h_AD_ne_BC : side_AD_ne_BC q)
  (h_perimeter : perimeter_eq_960 q = 960) :
  let cos_A := sorry -- Definition of cos A for the quadrilateral
  cos_A = 1 := by sorry

end NUMINAMATH_CALUDE_cos_A_eq_one_l744_74432


namespace NUMINAMATH_CALUDE_rectangular_field_area_l744_74438

/-- Represents a rectangular field with a given length, breadth, and perimeter. -/
structure RectangularField where
  length : ℝ
  breadth : ℝ
  perimeter : ℝ

/-- The area of a rectangular field. -/
def area (field : RectangularField) : ℝ :=
  field.length * field.breadth

/-- Theorem: For a rectangular field where the breadth is 60% of the length
    and the perimeter is 800 m, the area of the field is 37,500 square meters. -/
theorem rectangular_field_area :
  ∀ (field : RectangularField),
    field.breadth = 0.6 * field.length →
    field.perimeter = 800 →
    area field = 37500 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l744_74438


namespace NUMINAMATH_CALUDE_zeros_of_derivative_form_arithmetic_progression_l744_74474

/-- A fourth-degree polynomial whose zeros form an arithmetic progression -/
def ArithmeticZerosPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ (a r : ℝ) (α : ℝ), α ≠ 0 ∧ r > 0 ∧
    ∀ x, f x = α * (x - a) * (x - (a + r)) * (x - (a + 2*r)) * (x - (a + 3*r))

/-- The zeros of a polynomial form an arithmetic progression -/
def ZerosFormArithmeticProgression (f : ℝ → ℝ) : Prop :=
  ∃ (a d : ℝ), ∀ x, f x = 0 → ∃ n : ℕ, x = a + n * d

/-- The main theorem -/
theorem zeros_of_derivative_form_arithmetic_progression
  (f : ℝ → ℝ) (hf : ArithmeticZerosPolynomial f) :
  ZerosFormArithmeticProgression f' :=
sorry

end NUMINAMATH_CALUDE_zeros_of_derivative_form_arithmetic_progression_l744_74474


namespace NUMINAMATH_CALUDE_zero_of_f_l744_74484

-- Define the function f
def f (x : ℝ) : ℝ := x + 2

-- State the theorem
theorem zero_of_f : ∃ x : ℝ, f x = 0 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_zero_of_f_l744_74484


namespace NUMINAMATH_CALUDE_workshop_participation_l744_74479

theorem workshop_participation (total : ℕ) (A B C : ℕ) (at_least_two : ℕ) 
  (h_total : total = 25)
  (h_A : A = 15)
  (h_B : B = 14)
  (h_C : C = 11)
  (h_at_least_two : at_least_two = 12)
  (h_sum : A + B + C ≥ total + at_least_two) :
  ∃ (x y z a b c : ℕ), 
    x + y + z + a + b + c = total ∧
    a + b + c = at_least_two ∧
    x + a + c = A ∧
    y + a + b = B ∧
    z + b + c = C ∧
    0 = total - (x + y + z + a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_workshop_participation_l744_74479


namespace NUMINAMATH_CALUDE_enemy_plane_hit_probability_l744_74472

-- Define the probabilities of hitting for Person A and Person B
def prob_A_hits : ℝ := 0.6
def prob_B_hits : ℝ := 0.5

-- Define the event of the plane being hit
def plane_hit (prob_A prob_B : ℝ) : Prop :=
  1 - (1 - prob_A) * (1 - prob_B) = 0.8

-- State the theorem
theorem enemy_plane_hit_probability :
  plane_hit prob_A_hits prob_B_hits :=
by sorry

end NUMINAMATH_CALUDE_enemy_plane_hit_probability_l744_74472


namespace NUMINAMATH_CALUDE_lecture_hall_rows_l744_74439

/-- Represents the number of seats in a row of the lecture hall. -/
def seatsInRow (n : ℕ) : ℕ := 12 + 2 * (n - 1)

/-- Represents the total number of seats in the first n rows of the lecture hall. -/
def totalSeats (n : ℕ) : ℕ := n * (seatsInRow 1 + seatsInRow n) / 2

/-- States that the number of rows in the lecture hall is 16, given the conditions. -/
theorem lecture_hall_rows :
  ∃ (n : ℕ),
    n > 0 ∧
    totalSeats n > 400 ∧
    totalSeats n ≤ 440 ∧
    seatsInRow 1 = 12 ∧
    ∀ (i : ℕ), i > 0 → seatsInRow (i + 1) = seatsInRow i + 2 ∧
    n = 16 :=
  sorry

end NUMINAMATH_CALUDE_lecture_hall_rows_l744_74439


namespace NUMINAMATH_CALUDE_outfit_choices_l744_74461

/-- The number of color options for each item type -/
def num_colors : ℕ := 6

/-- The number of item types in an outfit -/
def num_items : ℕ := 4

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_colors ^ num_items

/-- The number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- The number of valid outfits (excluding those with all items of the same color) -/
def valid_outfits : ℕ := total_combinations - same_color_outfits

theorem outfit_choices :
  valid_outfits = 1290 :=
sorry

end NUMINAMATH_CALUDE_outfit_choices_l744_74461


namespace NUMINAMATH_CALUDE_fraction_equality_l744_74403

theorem fraction_equality (p q r s : ℚ) 
  (h1 : p / q = 2)
  (h2 : q / r = 4 / 5)
  (h3 : r / s = 3) :
  s / p = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l744_74403


namespace NUMINAMATH_CALUDE_expression_evaluation_l744_74487

theorem expression_evaluation : -24 + 12 * (10 / 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l744_74487


namespace NUMINAMATH_CALUDE_andy_stencils_l744_74463

/-- Calculates the number of stencils painted given the following conditions:
  * Hourly wage
  * Pay per racquet strung
  * Pay per grommet change
  * Pay per stencil painted
  * Hours worked
  * Number of racquets strung
  * Number of grommet sets changed
  * Total earnings -/
def stencils_painted (hourly_wage : ℚ) (pay_per_racquet : ℚ) (pay_per_grommet : ℚ) 
  (pay_per_stencil : ℚ) (hours_worked : ℚ) (racquets_strung : ℕ) (grommets_changed : ℕ) 
  (total_earnings : ℚ) : ℕ :=
  sorry

theorem andy_stencils : 
  stencils_painted 9 15 10 1 8 7 2 202 = 5 :=
sorry

end NUMINAMATH_CALUDE_andy_stencils_l744_74463


namespace NUMINAMATH_CALUDE_janets_crayons_l744_74464

theorem janets_crayons (michelle_initial : ℕ) (michelle_final : ℕ) (janet_initial : ℕ) : 
  michelle_initial = 2 → 
  michelle_final = 4 → 
  michelle_final = michelle_initial + janet_initial → 
  janet_initial = 2 := by
sorry

end NUMINAMATH_CALUDE_janets_crayons_l744_74464


namespace NUMINAMATH_CALUDE_f_derivative_l744_74483

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

theorem f_derivative :
  deriv f = fun x => x * Real.cos x := by sorry

end NUMINAMATH_CALUDE_f_derivative_l744_74483


namespace NUMINAMATH_CALUDE_no_integer_solution_l744_74431

theorem no_integer_solution : ∀ m n : ℤ, m^2 ≠ n^5 - 4 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l744_74431


namespace NUMINAMATH_CALUDE_marks_total_spent_l744_74418

/-- Represents the purchase of a fruit with its quantity and price per pound -/
structure FruitPurchase where
  quantity : ℝ
  price_per_pound : ℝ

/-- Calculates the total cost of a fruit purchase -/
def total_cost (purchase : FruitPurchase) : ℝ :=
  purchase.quantity * purchase.price_per_pound

/-- Represents Mark's shopping list -/
structure ShoppingList where
  tomatoes : FruitPurchase
  apples : FruitPurchase
  oranges : FruitPurchase

/-- Calculates the total cost of all items in the shopping list -/
def total_spent (list : ShoppingList) : ℝ :=
  total_cost list.tomatoes + total_cost list.apples + total_cost list.oranges

/-- Mark's actual shopping list -/
def marks_shopping : ShoppingList :=
  { tomatoes := { quantity := 3, price_per_pound := 4.5 }
  , apples := { quantity := 7, price_per_pound := 3.25 }
  , oranges := { quantity := 4, price_per_pound := 2.75 }
  }

/-- Theorem: The total amount Mark spent is $47.25 -/
theorem marks_total_spent :
  total_spent marks_shopping = 47.25 := by
  sorry


end NUMINAMATH_CALUDE_marks_total_spent_l744_74418


namespace NUMINAMATH_CALUDE_min_value_implies_a_l744_74409

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/x + a * Real.log x

theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, f a x ≥ 0) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 1, f a x = 0) →
  a = 2 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l744_74409


namespace NUMINAMATH_CALUDE_not_monotone_decreasing_f_maps_A_to_B_l744_74440

-- Define the sets A and B
def A : Set ℝ := {1, 4}
def B : Set ℝ := {1, -1, 2, -2}

-- Define the function f
def f (x : ℝ) : ℝ := (x^2)^(1/7)

-- Theorem 1
theorem not_monotone_decreasing (f : ℝ → ℝ) (h : f 2 < f 3) :
  ¬(∀ x y : ℝ, x ≤ y → f x ≥ f y) :=
sorry

-- Theorem 2
theorem f_maps_A_to_B :
  ∀ x ∈ A, f x ∈ B :=
sorry

end NUMINAMATH_CALUDE_not_monotone_decreasing_f_maps_A_to_B_l744_74440


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_range_l744_74442

theorem sine_cosine_inequality_range (θ : Real) :
  θ ∈ Set.Ioo 0 (2 * Real.pi) →
  Real.sin θ ^ 3 - Real.cos θ ^ 3 > (Real.cos θ ^ 5 - Real.sin θ ^ 5) / 7 →
  θ ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_range_l744_74442


namespace NUMINAMATH_CALUDE_evaluate_expression_l744_74452

theorem evaluate_expression (b : ℕ) (h : b = 2) : (b^3 * b^4) - 10 = 118 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l744_74452


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l744_74471

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The property that the range of a function is [0, +∞) -/
def HasNonnegativeRange (f : ℝ → ℝ) : Prop :=
  ∀ y, (∃ x, f x = y) → y ≥ 0

/-- The solution set of f(x) < c is an open interval of length 8 -/
def HasSolutionSetOfLength8 (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∃ m, ∀ x, f x < c ↔ m < x ∧ x < m + 8

theorem quadratic_function_theorem (a b : ℝ) :
  HasNonnegativeRange (QuadraticFunction a b) →
  HasSolutionSetOfLength8 (QuadraticFunction a b) 16 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l744_74471


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l744_74497

theorem fraction_sum_simplification : (3 : ℚ) / 462 + 28 / 42 = 311 / 462 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l744_74497


namespace NUMINAMATH_CALUDE_suzanna_textbooks_pages_l744_74491

/-- Calculates the total number of pages in Suzanna's textbooks -/
def total_pages (history : ℕ) : ℕ :=
  let geography := history + 70
  let math := (history + geography) / 2
  let science := 2 * history
  let literature := history + geography - 30
  let economics := math + literature + 25
  history + geography + math + science + literature + economics

/-- Theorem stating that the total number of pages in Suzanna's textbooks is 1845 -/
theorem suzanna_textbooks_pages : total_pages 160 = 1845 := by
  sorry

end NUMINAMATH_CALUDE_suzanna_textbooks_pages_l744_74491


namespace NUMINAMATH_CALUDE_greatest_rational_root_of_quadratic_l744_74410

theorem greatest_rational_root_of_quadratic (a b c : ℕ) 
  (ha : a ≤ 100) (hb : b ≤ 100) (hc : c ≤ 100) (ha_pos : a > 0) :
  ∃ (x : ℚ), x = -1/99 ∧ 
    (∀ (y : ℚ), y ≠ x → a * y^2 + b * y + c = 0 → y < x) ∧
    a * x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_CALUDE_greatest_rational_root_of_quadratic_l744_74410


namespace NUMINAMATH_CALUDE_ben_catch_count_l744_74423

/-- The number of fish caught by each family member (except Ben) --/
def family_catch : Fin 4 → ℕ
| 0 => 1  -- Judy
| 1 => 3  -- Billy
| 2 => 2  -- Jim
| 3 => 5  -- Susie

/-- The total number of filets they will have --/
def total_filets : ℕ := 24

/-- The number of fish thrown back --/
def thrown_back : ℕ := 3

/-- The number of filets per fish --/
def filets_per_fish : ℕ := 2

theorem ben_catch_count :
  ∃ (ben_catch : ℕ),
    ben_catch = total_filets / filets_per_fish + thrown_back - (family_catch 0 + family_catch 1 + family_catch 2 + family_catch 3) ∧
    ben_catch = 4 := by
  sorry

end NUMINAMATH_CALUDE_ben_catch_count_l744_74423


namespace NUMINAMATH_CALUDE_dropped_student_score_l744_74470

theorem dropped_student_score (initial_students : ℕ) (remaining_students : ℕ) 
  (initial_average : ℚ) (new_average : ℚ) : 
  initial_students = 16 →
  remaining_students = 15 →
  initial_average = 60.5 →
  new_average = 64 →
  (initial_students : ℚ) * initial_average - (remaining_students : ℚ) * new_average = 8 := by
  sorry

end NUMINAMATH_CALUDE_dropped_student_score_l744_74470


namespace NUMINAMATH_CALUDE_sum_of_squares_l744_74466

theorem sum_of_squares (a b c : ℝ) : 
  a * b + b * c + a * c = 131 → a + b + c = 19 → a^2 + b^2 + c^2 = 99 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l744_74466


namespace NUMINAMATH_CALUDE_min_value_parabola_l744_74445

theorem min_value_parabola (x y : ℝ) (h : x^2 = 4*y) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x' y' : ℝ), x'^2 = 4*y' →
    Real.sqrt ((x' - 3)^2 + (y' - 1)^2) + y' ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_parabola_l744_74445


namespace NUMINAMATH_CALUDE_inequality_proof_l744_74449

theorem inequality_proof (x : ℝ) : x > 4 → 3 * x + 5 < 5 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l744_74449


namespace NUMINAMATH_CALUDE_miss_molly_class_size_l744_74495

/-- The number of students in Miss Molly's class -/
def total_students : ℕ := 30

/-- The number of girls in the class -/
def num_girls : ℕ := 18

/-- The number of students who like yellow -/
def yellow_fans : ℕ := 9

/-- Theorem: The total number of students in Miss Molly's class is 30 -/
theorem miss_molly_class_size :
  (total_students / 2 = total_students - (num_girls / 3 + yellow_fans)) ∧
  (num_girls = 18) ∧
  (yellow_fans = 9) →
  total_students = 30 := by
sorry

end NUMINAMATH_CALUDE_miss_molly_class_size_l744_74495


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l744_74458

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  d > 0 →  -- positive common difference
  a 1 + a 2 + a 3 = 15 →  -- first condition
  a 1 * a 2 * a 3 = 80 →  -- second condition
  a 11 + a 12 + a 13 = 105 :=  -- conclusion
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l744_74458


namespace NUMINAMATH_CALUDE_smallest_average_is_16_5_l744_74428

/-- A function that generates all valid combinations of three single-digit
    and three double-digit numbers from the digits 1 to 9 without repetition -/
def generateValidCombinations : List (List ℕ) := sorry

/-- Calculates the average of a list of numbers -/
def average (numbers : List ℕ) : ℚ :=
  (numbers.sum : ℚ) / numbers.length

/-- Theorem stating that the smallest possible average is 16.5 -/
theorem smallest_average_is_16_5 :
  let allCombinations := generateValidCombinations
  let averages := allCombinations.map average
  averages.minimum? = some (33/2) := by sorry

end NUMINAMATH_CALUDE_smallest_average_is_16_5_l744_74428


namespace NUMINAMATH_CALUDE_unique_integer_solution_l744_74457

theorem unique_integer_solution : ∃! x : ℕ+, (4 * x)^2 - x = 2100 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l744_74457


namespace NUMINAMATH_CALUDE_stretch_circle_to_ellipse_l744_74419

/-- Given a circle A and a stretch transformation, prove the equation of the resulting curve C -/
theorem stretch_circle_to_ellipse (x y x' y' : ℝ) :
  (x^2 + y^2 = 1) →  -- Circle A equation
  (x' = 2*x) →       -- Stretch transformation for x
  (y' = 3*y) →       -- Stretch transformation for y
  (x'^2 / 4 + y'^2 / 9 = 1) -- Resulting curve C equation
:= by sorry

end NUMINAMATH_CALUDE_stretch_circle_to_ellipse_l744_74419


namespace NUMINAMATH_CALUDE_abs_a_minus_three_l744_74437

theorem abs_a_minus_three (a : ℝ) (h : a < 3) : |a - 3| = 3 - a := by
  sorry

end NUMINAMATH_CALUDE_abs_a_minus_three_l744_74437


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l744_74448

theorem existence_of_special_integers : ∃ (m n : ℤ), 
  (∃ (k₁ : ℤ), n^2 = k₁ * m) ∧
  (∃ (k₂ : ℤ), m^3 = k₂ * n^2) ∧
  (∃ (k₃ : ℤ), n^4 = k₃ * m^3) ∧
  (∃ (k₄ : ℤ), m^5 = k₄ * n^4) ∧
  (∀ (k₅ : ℤ), n^6 ≠ k₅ * m^5) ∧
  m = 32 ∧ n = 16 := by
sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l744_74448


namespace NUMINAMATH_CALUDE_range_of_m_l744_74498

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}

def B (m : ℝ) : Set ℝ := {x | x^2 - (m+3)*x + m^2 = 0}

theorem range_of_m : 
  ∀ m : ℝ, (A ∪ (Set.univ \ B m) = Set.univ) ↔ (m < -1 ∨ m ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l744_74498


namespace NUMINAMATH_CALUDE_equivalence_of_statements_l744_74488

theorem equivalence_of_statements (p q : Prop) :
  (¬p ∧ ¬q → p ∨ q) ↔ (p ∧ ¬q ∨ ¬p ∧ q) := by sorry

end NUMINAMATH_CALUDE_equivalence_of_statements_l744_74488


namespace NUMINAMATH_CALUDE_cylinder_volume_tripled_radius_cylinder_volume_increase_l744_74465

/-- Proves that tripling the radius of a cylinder while keeping the height constant
    results in a volume that is 9 times the original volume. -/
theorem cylinder_volume_tripled_radius 
  (r h : ℝ) 
  (original_volume : ℝ) 
  (h_original_volume : original_volume = π * r^2 * h) 
  (h_positive : r > 0 ∧ h > 0) :
  let new_volume := π * (3*r)^2 * h
  new_volume = 9 * original_volume :=
by sorry

/-- Proves that if a cylinder with volume 10 cubic feet has its radius tripled
    while its height remains constant, its new volume is 90 cubic feet. -/
theorem cylinder_volume_increase
  (r h : ℝ)
  (h_original_volume : π * r^2 * h = 10)
  (h_positive : r > 0 ∧ h > 0) :
  let new_volume := π * (3*r)^2 * h
  new_volume = 90 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_tripled_radius_cylinder_volume_increase_l744_74465


namespace NUMINAMATH_CALUDE_symmetry_about_y_axis_l744_74478

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x
def g (x : ℝ) : ℝ := x^2 + 2*x

-- State the theorem
theorem symmetry_about_y_axis (x : ℝ) : 
  (∀ (y : ℝ), f x = y ↔ g (-x) = y) → g x = x^2 + 2*x :=
by sorry

end NUMINAMATH_CALUDE_symmetry_about_y_axis_l744_74478


namespace NUMINAMATH_CALUDE_coefficient_sum_of_squares_l744_74427

theorem coefficient_sum_of_squares (p q r s t u : ℤ) :
  (∀ x, 343 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 3506 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_of_squares_l744_74427


namespace NUMINAMATH_CALUDE_intersection_empty_implies_m_leq_neg_one_l744_74446

def M (m : ℝ) : Set ℝ := {x | x - m < 0}

def N : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2*x}

theorem intersection_empty_implies_m_leq_neg_one (m : ℝ) :
  M m ∩ N = ∅ → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_m_leq_neg_one_l744_74446


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_cubed_l744_74430

theorem imaginary_part_of_one_plus_i_cubed (i : ℂ) : Complex.im ((1 + i)^3) = 2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_cubed_l744_74430


namespace NUMINAMATH_CALUDE_chocolate_chip_cookies_baked_l744_74459

/-- The number of dozens of cookies Ann baked for each type -/
structure CookieBatch where
  oatmeal_raisin : ℚ
  sugar : ℚ
  chocolate_chip : ℚ

/-- The number of dozens of cookies Ann gave away for each type -/
structure CookiesGivenAway where
  oatmeal_raisin : ℚ
  sugar : ℚ
  chocolate_chip : ℚ

def cookies_kept (baked : CookieBatch) (given_away : CookiesGivenAway) : ℚ :=
  (baked.oatmeal_raisin - given_away.oatmeal_raisin +
   baked.sugar - given_away.sugar +
   baked.chocolate_chip - given_away.chocolate_chip) * 12

theorem chocolate_chip_cookies_baked 
  (baked : CookieBatch)
  (given_away : CookiesGivenAway)
  (h1 : baked.oatmeal_raisin = 3)
  (h2 : baked.sugar = 2)
  (h3 : given_away.oatmeal_raisin = 2)
  (h4 : given_away.sugar = 3/2)
  (h5 : given_away.chocolate_chip = 5/2)
  (h6 : cookies_kept baked given_away = 36) :
  baked.chocolate_chip = 4 := by
sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookies_baked_l744_74459


namespace NUMINAMATH_CALUDE_median_length_l744_74443

/-- A tetrahedron with vertex D at the origin and right angles at D -/
structure RightTetrahedron where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  right_angles : sorry
  DA_length : ‖A‖ = 1
  DB_length : ‖B‖ = 2
  DC_length : ‖C‖ = 3

/-- The median of a tetrahedron from vertex D -/
def tetrahedron_median (t : RightTetrahedron) : ℝ := sorry

/-- Theorem: The length of the median from D in the specified tetrahedron is √6/3 -/
theorem median_length (t : RightTetrahedron) : 
  tetrahedron_median t = Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_median_length_l744_74443


namespace NUMINAMATH_CALUDE_elderly_sample_count_l744_74447

/-- Represents the composition of employees in a unit -/
structure EmployeeComposition where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Represents a stratified sample of employees -/
structure StratifiedSample where
  youngSampled : ℕ
  elderlySampled : ℕ

/-- Calculates the number of elderly employees in a stratified sample -/
def calculateElderlySampled (comp : EmployeeComposition) (sample : StratifiedSample) : ℚ :=
  (comp.elderly : ℚ) / comp.total * sample.youngSampled

theorem elderly_sample_count (comp : EmployeeComposition) (sample : StratifiedSample) :
  comp.total = 430 →
  comp.young = 160 →
  comp.middleAged = 2 * comp.elderly →
  sample.youngSampled = 32 →
  calculateElderlySampled comp sample = 18 := by
  sorry

end NUMINAMATH_CALUDE_elderly_sample_count_l744_74447


namespace NUMINAMATH_CALUDE_carnival_walk_distance_l744_74486

def total_distance : Real := 0.75
def car_to_entrance : Real := 0.33
def entrance_to_rides : Real := 0.33

theorem carnival_walk_distance : 
  total_distance - (car_to_entrance + entrance_to_rides) = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_carnival_walk_distance_l744_74486


namespace NUMINAMATH_CALUDE_polygon_area_l744_74462

-- Define the polygon as a list of points
def polygon : List (ℕ × ℕ) := [(0, 0), (5, 0), (5, 2), (3, 2), (3, 3), (2, 3), (2, 2), (0, 2), (0, 0)]

-- Define a function to calculate the area of a polygon given its vertices
def calculatePolygonArea (vertices : List (ℕ × ℕ)) : ℕ := sorry

-- Theorem statement
theorem polygon_area : calculatePolygonArea polygon = 11 := by sorry

end NUMINAMATH_CALUDE_polygon_area_l744_74462


namespace NUMINAMATH_CALUDE_power_sum_unique_solution_l744_74473

theorem power_sum_unique_solution (k : ℕ+) :
  (∃ (n : ℕ) (m : ℕ), m > 1 ∧ 3^k.val + 5^k.val = n^m) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_unique_solution_l744_74473


namespace NUMINAMATH_CALUDE_election_vote_count_l744_74433

/-- Represents an election with two candidates -/
structure TwoCandidateElection where
  totalVotes : ℕ
  loserPercentage : ℚ
  voteDifference : ℕ

/-- 
Theorem: In a two-candidate election where the losing candidate received 40% of the votes
and lost by 5000 votes, the total number of votes cast was 25000.
-/
theorem election_vote_count (e : TwoCandidateElection) 
  (h1 : e.loserPercentage = 40 / 100)
  (h2 : e.voteDifference = 5000) : 
  e.totalVotes = 25000 := by
  sorry

#eval (40 : ℚ) / 100  -- To verify the rational number representation

end NUMINAMATH_CALUDE_election_vote_count_l744_74433


namespace NUMINAMATH_CALUDE_min_expression_leq_one_l744_74408

theorem min_expression_leq_one (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_three : x + y + z = 3) : 
  min (x * (x + y - z)) (min (y * (y + z - x)) (z * (z + x - y))) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_expression_leq_one_l744_74408


namespace NUMINAMATH_CALUDE_complex_sum_parts_zero_l744_74490

theorem complex_sum_parts_zero (b : ℝ) : 
  let z : ℂ := 2 - b * I
  (z.re + z.im = 0) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_parts_zero_l744_74490


namespace NUMINAMATH_CALUDE_three_similar_points_l744_74480

-- Define the trapezoid ABCD
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define a point P on AB
def PointOnAB (t : Trapezoid) (x : ℝ) : ℝ × ℝ :=
  (x * t.B.1 + (1 - x) * t.A.1, x * t.B.2 + (1 - x) * t.A.2)

-- Define the similarity condition
def IsSimilar (t : Trapezoid) (x : ℝ) : Prop :=
  let P := PointOnAB t x
  ∃ k : ℝ, k > 0 ∧
    (P.1 - t.A.1)^2 + (P.2 - t.A.2)^2 = k * ((t.C.1 - P.1)^2 + (t.C.2 - P.2)^2) ∧
    (t.D.1 - t.A.1)^2 + (t.D.2 - t.A.2)^2 = k * ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)

-- Theorem statement
theorem three_similar_points (t : Trapezoid) 
  (h1 : t.B.1 - t.A.1 = 7) 
  (h2 : t.D.2 - t.A.2 = 2) 
  (h3 : t.C.1 - t.B.1 = 3) 
  (h4 : t.A.2 = t.B.2) 
  (h5 : t.C.2 = t.D.2) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, 0 ≤ x ∧ x ≤ 1 ∧ IsSimilar t x :=
sorry

end NUMINAMATH_CALUDE_three_similar_points_l744_74480


namespace NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l744_74467

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l744_74467


namespace NUMINAMATH_CALUDE_number_problem_l744_74496

theorem number_problem (x : ℝ) : 3 * (2 * x + 9) = 75 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l744_74496


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l744_74413

theorem complex_magnitude_problem (z : ℂ) : z = Complex.I * (Complex.I - 1) → Complex.abs (z - 1) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l744_74413


namespace NUMINAMATH_CALUDE_SetA_eq_SetB_l744_74468

/-- Set A: integers representable as x^2 + 2y^2 where x and y are integers -/
def SetA : Set Int :=
  {n : Int | ∃ x y : Int, n = x^2 + 2*y^2}

/-- Set B: integers representable as x^2 + 6xy + 11y^2 where x and y are integers -/
def SetB : Set Int :=
  {n : Int | ∃ x y : Int, n = x^2 + 6*x*y + 11*y^2}

/-- Theorem stating that Set A and Set B are equal -/
theorem SetA_eq_SetB : SetA = SetB := by sorry

end NUMINAMATH_CALUDE_SetA_eq_SetB_l744_74468


namespace NUMINAMATH_CALUDE_evaluate_expression_l744_74493

theorem evaluate_expression : 
  (30 ^ 20 : ℝ) / (90 ^ 10) = 10 ^ 10 := by
  sorry

#check evaluate_expression

end NUMINAMATH_CALUDE_evaluate_expression_l744_74493


namespace NUMINAMATH_CALUDE_monday_bonnets_count_l744_74400

/-- Represents the number of bonnets made on each day of the week --/
structure BonnetProduction where
  monday : ℕ
  tuesday_wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total number of bonnets produced --/
def total_bonnets (bp : BonnetProduction) : ℕ :=
  bp.monday + bp.tuesday_wednesday + bp.thursday + bp.friday

theorem monday_bonnets_count :
  ∃ (bp : BonnetProduction),
    bp.tuesday_wednesday = 2 * bp.monday ∧
    bp.thursday = bp.monday + 5 ∧
    bp.friday = bp.thursday - 5 ∧
    total_bonnets bp = 11 * 5 ∧
    bp.monday = 10 := by
  sorry

end NUMINAMATH_CALUDE_monday_bonnets_count_l744_74400


namespace NUMINAMATH_CALUDE_greatest_n_for_2008_l744_74434

-- Define the sum of digits function
def sum_of_digits (a : ℕ) : ℕ := sorry

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => sorry  -- Initial value, not specified in the problem
  | n + 1 => a n + sum_of_digits (a n)

-- Theorem statement
theorem greatest_n_for_2008 : (∃ n : ℕ, a n = 2008) ∧ (∀ m : ℕ, m > 6 → a m ≠ 2008) := by sorry

end NUMINAMATH_CALUDE_greatest_n_for_2008_l744_74434


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l744_74453

/-- The axis of symmetry of a parabola y = ax^2 + bx + c is the vertical line x = -b/(2a) -/
theorem parabola_axis_of_symmetry (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  ∀ x, f (-b / (2 * a) + x) = f (-b / (2 * a) - x) :=
by sorry

/-- The axis of symmetry of the parabola y = x^2 - 2x - 3 is the line x = 1 -/
theorem specific_parabola_axis_of_symmetry :
  let f : ℝ → ℝ := λ x => x^2 - 2 * x - 3
  ∀ x, f (1 + x) = f (1 - x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l744_74453


namespace NUMINAMATH_CALUDE_horseshoe_profit_calculation_l744_74424

/-- Calculates the profit for a horseshoe manufacturing company --/
theorem horseshoe_profit_calculation 
  (initial_outlay : ℕ) 
  (manufacturing_cost_per_set : ℕ) 
  (selling_price_per_set : ℕ) 
  (sets_produced_and_sold : ℕ) :
  initial_outlay = 10000 →
  manufacturing_cost_per_set = 20 →
  selling_price_per_set = 50 →
  sets_produced_and_sold = 500 →
  (selling_price_per_set * sets_produced_and_sold) - 
  (initial_outlay + manufacturing_cost_per_set * sets_produced_and_sold) = 5000 :=
by sorry

end NUMINAMATH_CALUDE_horseshoe_profit_calculation_l744_74424


namespace NUMINAMATH_CALUDE_principal_amount_calculation_l744_74455

/-- Proves that given a principal amount P put at simple interest for 2 years,
    if an increase of 4% in the interest rate results in Rs. 60 more interest,
    then P = 750. -/
theorem principal_amount_calculation (P R : ℝ) (h1 : P > 0) (h2 : R > 0) :
  (P * (R + 4) * 2) / 100 = (P * R * 2) / 100 + 60 → P = 750 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_calculation_l744_74455


namespace NUMINAMATH_CALUDE_f_has_unique_zero_l744_74421

noncomputable def f (x : ℝ) := Real.log x + 2 * x - 6

theorem f_has_unique_zero :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_unique_zero_l744_74421


namespace NUMINAMATH_CALUDE_tetrahedra_triangle_inequality_l744_74407

/-- A finite graph -/
structure FiniteGraph where
  -- We don't need to specify the internal structure of the graph
  -- as it's not directly used in the theorem statement

/-- The number of triangles in a finite graph -/
def numTriangles (G : FiniteGraph) : ℕ := sorry

/-- The number of tetrahedra in a finite graph -/
def numTetrahedra (G : FiniteGraph) : ℕ := sorry

/-- The main theorem stating the inequality between tetrahedra and triangles -/
theorem tetrahedra_triangle_inequality (G : FiniteGraph) :
  (numTetrahedra G)^3 ≤ (3/32) * (numTriangles G)^4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedra_triangle_inequality_l744_74407


namespace NUMINAMATH_CALUDE_cubic_equation_natural_roots_l744_74450

theorem cubic_equation_natural_roots (p : ℝ) :
  (∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    5 * (x : ℝ)^3 - 5*(p+1)*(x : ℝ)^2 + (71*p-1)*(x : ℝ) + 1 = 66*p ∧
    5 * (y : ℝ)^3 - 5*(p+1)*(y : ℝ)^2 + (71*p-1)*(y : ℝ) + 1 = 66*p ∧
    5 * (z : ℝ)^3 - 5*(p+1)*(z : ℝ)^2 + (71*p-1)*(z : ℝ) + 1 = 66*p) ↔
  p = 76 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_natural_roots_l744_74450


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l744_74476

/-- Given a geometric sequence with positive terms and a specific arithmetic sequence condition, prove the common ratio. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_positive : ∀ n, a n > 0)
  (h_arith : a 1 + 2 * a 2 = a 3) : 
  q = 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l744_74476


namespace NUMINAMATH_CALUDE_henry_book_count_l744_74416

/-- Calculates the number of books Henry has after donating and picking up new books -/
def final_book_count (initial_books : ℕ) (box_count : ℕ) (books_per_box : ℕ) 
  (room_books : ℕ) (coffee_table_books : ℕ) (kitchen_books : ℕ) (new_books : ℕ) : ℕ :=
  initial_books - (box_count * books_per_box + room_books + coffee_table_books + kitchen_books) + new_books

/-- Theorem stating that Henry ends up with 23 books -/
theorem henry_book_count : 
  final_book_count 99 3 15 21 4 18 12 = 23 := by
  sorry

end NUMINAMATH_CALUDE_henry_book_count_l744_74416


namespace NUMINAMATH_CALUDE_expressions_equality_l744_74481

theorem expressions_equality (a b c : ℝ) :
  a + 2 * b * c = (a + b) * (a + 2 * c) ↔ a + b + 2 * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_expressions_equality_l744_74481


namespace NUMINAMATH_CALUDE_lena_always_greater_probability_lena_greater_l744_74435

def lena_set : Finset ℕ := {7, 8, 9}
def jonah_set : Finset ℕ := {2, 4, 6}

def lena_result (a b : ℕ) : ℕ := a * b

def jonah_result (a b c : ℕ) : ℕ := (a + b) * c

theorem lena_always_greater :
  ∀ (a b : ℕ) (c d e : ℕ),
    a ∈ lena_set → b ∈ lena_set → a ≠ b →
    c ∈ jonah_set → d ∈ jonah_set → e ∈ jonah_set →
    c ≠ d → c ≠ e → d ≠ e →
    lena_result a b > jonah_result c d e :=
by
  sorry

theorem probability_lena_greater : ℚ :=
  1

#check lena_always_greater
#check probability_lena_greater

end NUMINAMATH_CALUDE_lena_always_greater_probability_lena_greater_l744_74435


namespace NUMINAMATH_CALUDE_sequence_second_term_l744_74406

/-- Given a sequence {a_n} with sum of first n terms S_n, prove a_2 = 4 -/
theorem sequence_second_term (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = 2 * (a n - 1)) → a 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_second_term_l744_74406


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l744_74402

theorem quadratic_equation_roots (a b : ℝ) (h1 : a ≠ 0) :
  (∃ x : ℝ, a * x^2 = b ∧ x = 2) → (∃ y : ℝ, a * y^2 = b ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l744_74402


namespace NUMINAMATH_CALUDE_least_whole_number_ratio_l744_74405

theorem least_whole_number_ratio (x : ℕ) : 
  (x > 0 ∧ (6 - x : ℚ) / (7 - x) < 16 / 21) ↔ x ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_least_whole_number_ratio_l744_74405


namespace NUMINAMATH_CALUDE_cylinder_plane_intersection_l744_74477

/-- The equation of the curve formed by intersecting a cylinder with a plane -/
theorem cylinder_plane_intersection
  (r h : ℝ) -- radius and height of the cylinder
  (α : ℝ) -- angle between cutting plane and base plane
  (hr : r > 0)
  (hh : h > 0)
  (hα : 0 < α ∧ α < π/2) :
  ∃ f : ℝ → ℝ,
    (∀ x, 0 < x → x < 2*π*r →
      f x = r * Real.tan α * Real.sin (x/r - π/2)) ∧
    (∀ x, f x = 0 → (x = 0 ∨ x = 2*π*r)) :=
sorry

end NUMINAMATH_CALUDE_cylinder_plane_intersection_l744_74477


namespace NUMINAMATH_CALUDE_circle_divides_sides_l744_74485

/-- An isosceles trapezoid with bases in ratio 3:2 and a circle on the larger base -/
structure IsoscelesTrapezoidWithCircle where
  /-- Length of the smaller base -/
  b : ℝ
  /-- Length of the larger base -/
  a : ℝ
  /-- The bases are in ratio 3:2 -/
  base_ratio : a = (3/2) * b
  /-- The trapezoid is isosceles -/
  isosceles : True
  /-- Radius of the circle (half of the larger base) -/
  r : ℝ
  circle_diameter : r = a / 2
  /-- Length of the segment cut off on the smaller base by the circle -/
  m : ℝ
  segment_half_base : m = b / 2

/-- The circle divides the non-parallel sides of the trapezoid in the ratio 1:2 -/
theorem circle_divides_sides (t : IsoscelesTrapezoidWithCircle) :
  ∃ (x y : ℝ), x + y = t.a - t.b ∧ x / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_divides_sides_l744_74485


namespace NUMINAMATH_CALUDE_one_distinct_real_root_l744_74412

theorem one_distinct_real_root :
  ∃! x : ℝ, x ≠ 0 ∧ (abs x - 4 / x = 3 * abs x / x) :=
by sorry

end NUMINAMATH_CALUDE_one_distinct_real_root_l744_74412


namespace NUMINAMATH_CALUDE_rectangular_lot_area_l744_74460

/-- Represents a rectangular lot with given properties -/
structure RectangularLot where
  width : ℝ
  length : ℝ
  length_constraint : length = 2 * width + 35
  perimeter_constraint : 2 * (width + length) = 850

/-- The area of a rectangular lot -/
def area (lot : RectangularLot) : ℝ := lot.width * lot.length

/-- Theorem stating that a rectangular lot with the given properties has an area of 38350 square feet -/
theorem rectangular_lot_area : 
  ∀ (lot : RectangularLot), area lot = 38350 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_lot_area_l744_74460


namespace NUMINAMATH_CALUDE_fortune_telling_app_probability_l744_74482

theorem fortune_telling_app_probability :
  let n : ℕ := 7  -- Total number of trials
  let k : ℕ := 3  -- Number of successful trials
  let p : ℚ := 1/3  -- Probability of success in each trial
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 := by
  sorry

end NUMINAMATH_CALUDE_fortune_telling_app_probability_l744_74482


namespace NUMINAMATH_CALUDE_goldfish_count_l744_74414

theorem goldfish_count (daily_food_per_fish : ℝ) (special_food_percentage : ℝ) 
  (special_food_cost_per_ounce : ℝ) (total_special_food_cost : ℝ) 
  (h1 : daily_food_per_fish = 1.5)
  (h2 : special_food_percentage = 0.2)
  (h3 : special_food_cost_per_ounce = 3)
  (h4 : total_special_food_cost = 45) : 
  ∃ (total_fish : ℕ), total_fish = 50 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_count_l744_74414


namespace NUMINAMATH_CALUDE_x_plus_2y_squared_value_l744_74494

theorem x_plus_2y_squared_value (x y : ℝ) :
  8 * y^4 + 4 * x^2 * y^2 + 4 * x * y^2 + 2 * x^3 + 2 * y^2 + 2 * x = x^2 + 1 →
  x + 2 * y^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_squared_value_l744_74494


namespace NUMINAMATH_CALUDE_negative_nine_plus_sixteen_y_squared_equals_seven_y_squared_l744_74444

theorem negative_nine_plus_sixteen_y_squared_equals_seven_y_squared (y : ℝ) : 
  -9 * y^2 + 16 * y^2 = 7 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_nine_plus_sixteen_y_squared_equals_seven_y_squared_l744_74444


namespace NUMINAMATH_CALUDE_problem_1_l744_74436

theorem problem_1 : (-13/2 : ℚ) * (4/13 : ℚ) - 8 / |(-4 + 2)| = -6 := by sorry

end NUMINAMATH_CALUDE_problem_1_l744_74436


namespace NUMINAMATH_CALUDE_birds_in_tree_l744_74425

theorem birds_in_tree (initial_birds : ℝ) (birds_flew_away : ℝ) (birds_remaining : ℝ) : 
  birds_flew_away = 14.0 → birds_remaining = 7 → initial_birds = birds_flew_away + birds_remaining :=
by sorry

end NUMINAMATH_CALUDE_birds_in_tree_l744_74425


namespace NUMINAMATH_CALUDE_connie_marbles_l744_74454

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 183

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 593

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem connie_marbles : initial_marbles = 776 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l744_74454


namespace NUMINAMATH_CALUDE_prism_volume_l744_74420

/-- The volume of a right rectangular prism given its face areas -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 36)
  (h2 : a * c = 72)
  (h3 : b * c = 48) :
  a * b * c = 352.8 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_l744_74420


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l744_74441

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l744_74441


namespace NUMINAMATH_CALUDE_first_wave_infections_count_l744_74456

/-- The number of infections per day during the first wave of coronavirus -/
def first_wave_infections : ℕ := 375

/-- The number of infections per day during the second wave of coronavirus -/
def second_wave_infections : ℕ := 4 * first_wave_infections

/-- The duration of the second wave in days -/
def second_wave_duration : ℕ := 14

/-- The total number of infections during the second wave -/
def total_second_wave_infections : ℕ := 21000

/-- Theorem stating that the number of infections per day during the first wave was 375 -/
theorem first_wave_infections_count : 
  first_wave_infections = 375 ∧ 
  second_wave_infections = 4 * first_wave_infections ∧
  total_second_wave_infections = second_wave_infections * second_wave_duration :=
sorry

end NUMINAMATH_CALUDE_first_wave_infections_count_l744_74456
