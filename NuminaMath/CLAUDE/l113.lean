import Mathlib

namespace NUMINAMATH_CALUDE_fraction_evaluation_l113_11306

theorem fraction_evaluation : (10^7 : ℝ) / (5 * 10^4) = 200 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l113_11306


namespace NUMINAMATH_CALUDE_smaller_circle_area_l113_11372

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  r : ℝ  -- radius of smaller circle
  center_small : ℝ × ℝ  -- center of smaller circle
  center_large : ℝ × ℝ  -- center of larger circle
  P : ℝ × ℝ  -- point P
  A : ℝ × ℝ  -- point A on smaller circle
  B : ℝ × ℝ  -- point B on larger circle
  externally_tangent : (center_small.1 - center_large.1)^2 + (center_small.2 - center_large.2)^2 = (r + 3*r)^2
  on_smaller_circle : (A.1 - center_small.1)^2 + (A.2 - center_small.2)^2 = r^2
  on_larger_circle : (B.1 - center_large.1)^2 + (B.2 - center_large.2)^2 = (3*r)^2
  PA_tangent : ((P.1 - A.1)*(A.1 - center_small.1) + (P.2 - A.2)*(A.2 - center_small.2))^2 = 
               ((P.1 - A.1)^2 + (P.2 - A.2)^2)*r^2
  AB_tangent : ((A.1 - B.1)*(B.1 - center_large.1) + (A.2 - B.2)*(B.2 - center_large.2))^2 = 
               ((A.1 - B.1)^2 + (A.2 - B.2)^2)*(3*r)^2
  PA_length : (P.1 - A.1)^2 + (P.2 - A.2)^2 = 36
  AB_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36

theorem smaller_circle_area (tc : TangentCircles) : Real.pi * tc.r^2 = 36 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_smaller_circle_area_l113_11372


namespace NUMINAMATH_CALUDE_lucille_remaining_cents_l113_11381

-- Define the problem parameters
def cents_per_weed : ℕ := 6
def weeds_flower_bed : ℕ := 11
def weeds_vegetable_patch : ℕ := 14
def weeds_grass : ℕ := 32
def soda_cost : ℕ := 99

-- Calculate the total weeds pulled
def total_weeds_pulled : ℕ := weeds_flower_bed + weeds_vegetable_patch + weeds_grass / 2

-- Calculate the earnings
def earnings : ℕ := total_weeds_pulled * cents_per_weed

-- Calculate the remaining cents
def remaining_cents : ℕ := earnings - soda_cost

-- Theorem to prove
theorem lucille_remaining_cents : remaining_cents = 147 := by
  sorry

end NUMINAMATH_CALUDE_lucille_remaining_cents_l113_11381


namespace NUMINAMATH_CALUDE_function_range_l113_11316

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 4

-- Define the domain
def domain : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem function_range :
  Set.range (fun x => f x) = Set.Icc (-5) 4 := by sorry

end NUMINAMATH_CALUDE_function_range_l113_11316


namespace NUMINAMATH_CALUDE_specific_triangle_BD_length_l113_11301

/-- A right triangle with a perpendicular from the right angle to the hypotenuse -/
structure RightTriangleWithAltitude where
  -- The lengths of the sides
  AB : ℝ
  AC : ℝ
  BC : ℝ
  -- The length of the altitude
  AD : ℝ
  -- The length of the segment from B to D
  BD : ℝ
  -- Conditions
  right_angle : AB^2 + AC^2 = BC^2
  altitude_perpendicular : AD * BC = AB * AC
  pythagoras_BD : BD^2 + AD^2 = BC^2

/-- The main theorem about the specific triangle in the problem -/
theorem specific_triangle_BD_length 
  (triangle : RightTriangleWithAltitude)
  (h_AB : triangle.AB = 45)
  (h_AC : triangle.AC = 60) :
  triangle.BD = 63 := by
  sorry

#check specific_triangle_BD_length

end NUMINAMATH_CALUDE_specific_triangle_BD_length_l113_11301


namespace NUMINAMATH_CALUDE_ellipse_isosceles_triangle_existence_l113_11366

/-- Ellipse C with equation x²/9 + y²/8 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/9 + y^2/8 = 1

/-- Line l passing through point P(0, 2) with slope k -/
def line_l (k x y : ℝ) : Prop := y = k*x + 2

/-- Point D on the x-axis -/
def point_D (m : ℝ) : Prop := ∃ y, y = 0

/-- Isosceles triangle condition -/
def isosceles_triangle (xA yA xB yB xD : ℝ) : Prop :=
  (xA - xD)^2 + yA^2 = (xB - xD)^2 + yB^2

theorem ellipse_isosceles_triangle_existence :
  ∀ k > 0,
  ∃ xA yA xB yB m,
    ellipse_C xA yA ∧
    ellipse_C xB yB ∧
    line_l k xA yA ∧
    line_l k xB yB ∧
    point_D m ∧
    isosceles_triangle xA yA xB yB m ∧
    -Real.sqrt 2 / 12 ≤ m ∧
    m < 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_isosceles_triangle_existence_l113_11366


namespace NUMINAMATH_CALUDE_ball_game_proof_l113_11363

theorem ball_game_proof (total_balls : ℕ) (red_prob_1 : ℚ) (black_prob_2 red_prob_2 : ℚ) 
  (green_balls : ℕ) (red_prob_3 : ℚ) :
  total_balls = 10 →
  red_prob_1 = 1 →
  black_prob_2 = 1/2 →
  red_prob_2 = 1/2 →
  green_balls = 2 →
  red_prob_3 = 7/10 →
  ∃ (black_balls : ℕ), black_balls = 1 := by
  sorry

#check ball_game_proof

end NUMINAMATH_CALUDE_ball_game_proof_l113_11363


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l113_11392

/-- An arithmetic sequence with given terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 1 - seq.a 0

theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h2 : seq.a 2 = 12)
  (h6 : seq.a 6 = 4) :
  common_difference seq = -2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l113_11392


namespace NUMINAMATH_CALUDE_cupcake_packages_l113_11315

/-- Given the initial number of cupcakes, the number of cupcakes eaten, and the number of cupcakes per package,
    calculate the number of complete packages that can be made. -/
def packages_made (initial : ℕ) (eaten : ℕ) (per_package : ℕ) : ℕ :=
  (initial - eaten) / per_package

/-- Theorem stating that with 18 initial cupcakes, 8 eaten, and 2 cupcakes per package,
    the number of packages that can be made is 5. -/
theorem cupcake_packages : packages_made 18 8 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_packages_l113_11315


namespace NUMINAMATH_CALUDE_ball_drawing_probability_l113_11350

-- Define the sample space
def Ω : Type := Fin 4 × Fin 3

-- Define the events
def A : Set Ω := {ω | (ω.1 < 2 ∧ ω.2 < 1) ∨ (ω.1 ≥ 2 ∧ ω.2 ≥ 1)}
def B : Set Ω := {ω | ω.1 < 2}
def C : Set Ω := {ω | (ω.1 < 2 ∧ ω.2 < 1) ∨ (ω.1 ≥ 2 ∧ ω.2 = 1)}
def D : Set Ω := {ω | (ω.1 < 2 ∧ ω.2 ≥ 1) ∨ (ω.1 ≥ 2 ∧ ω.2 < 1)}

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- State the theorem
theorem ball_drawing_probability :
  (P A + P D = 1) ∧
  (P (A ∩ B) = P A * P B) ∧
  (P (C ∩ D) = P C * P D) := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_probability_l113_11350


namespace NUMINAMATH_CALUDE_stickers_for_square_window_l113_11391

/-- Calculates the number of stickers needed to decorate a square window -/
theorem stickers_for_square_window (side_length interval : ℕ) : 
  side_length = 90 → interval = 3 → (4 * side_length) / interval = 120 := by
  sorry

end NUMINAMATH_CALUDE_stickers_for_square_window_l113_11391


namespace NUMINAMATH_CALUDE_base_10_to_12_256_l113_11347

/-- Converts a base-10 number to its base-12 representation -/
def toBase12 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base-12 to a natural number -/
def fromBase12 (digits : List ℕ) : ℕ :=
  sorry

theorem base_10_to_12_256 :
  toBase12 256 = [1, 9, 4] ∧ fromBase12 [1, 9, 4] = 256 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_12_256_l113_11347


namespace NUMINAMATH_CALUDE_optimal_import_quantity_l113_11344

/-- Represents the annual import volume in units -/
def annual_volume : ℕ := 10000

/-- Represents the shipping cost per import in yuan -/
def shipping_cost : ℕ := 100

/-- Represents the rent cost per unit in yuan -/
def rent_cost_per_unit : ℕ := 2

/-- Calculates the number of imports per year given the quantity per import -/
def imports_per_year (quantity_per_import : ℕ) : ℕ :=
  annual_volume / quantity_per_import

/-- Calculates the total annual shipping cost -/
def annual_shipping_cost (quantity_per_import : ℕ) : ℕ :=
  shipping_cost * imports_per_year quantity_per_import

/-- Calculates the total annual rent cost -/
def annual_rent_cost (quantity_per_import : ℕ) : ℕ :=
  rent_cost_per_unit * (quantity_per_import / 2)

/-- Calculates the total annual cost (shipping + rent) -/
def total_annual_cost (quantity_per_import : ℕ) : ℕ :=
  annual_shipping_cost quantity_per_import + annual_rent_cost quantity_per_import

/-- Theorem stating that 1000 units per import minimizes the total annual cost -/
theorem optimal_import_quantity :
  ∀ q : ℕ, q > 0 → q ≤ annual_volume → total_annual_cost 1000 ≤ total_annual_cost q :=
sorry

end NUMINAMATH_CALUDE_optimal_import_quantity_l113_11344


namespace NUMINAMATH_CALUDE_pentagon_quadrilateral_angle_sum_l113_11380

theorem pentagon_quadrilateral_angle_sum :
  ∀ (pentagon_interior_angle quadrilateral_reflex_angle : ℝ),
  pentagon_interior_angle = 108 →
  quadrilateral_reflex_angle = 360 - pentagon_interior_angle →
  360 - quadrilateral_reflex_angle = 108 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_quadrilateral_angle_sum_l113_11380


namespace NUMINAMATH_CALUDE_x_equals_six_l113_11314

theorem x_equals_six (a b x : ℝ) 
  (h1 : 2^a = x) 
  (h2 : 3^b = x) 
  (h3 : 1/a + 1/b = 1) : 
  x = 6 := by sorry

end NUMINAMATH_CALUDE_x_equals_six_l113_11314


namespace NUMINAMATH_CALUDE_cistern_fill_time_l113_11337

/-- Represents the time (in hours) it takes to fill a cistern when three pipes are opened simultaneously. -/
def fill_time (rate_A rate_B rate_C : ℚ) : ℚ :=
  1 / (rate_A + rate_B + rate_C)

/-- Theorem stating that given the specific fill/empty rates of pipes A, B, and C,
    the cistern will be filled in 12 hours when all pipes are opened simultaneously. -/
theorem cistern_fill_time :
  fill_time (1/10) (1/15) (-1/12) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l113_11337


namespace NUMINAMATH_CALUDE_marble_bag_count_l113_11370

theorem marble_bag_count :
  ∀ (total white : ℕ),
  (6 : ℝ) + 9 + white = total →
  (9 + white : ℝ) / total = 0.7 →
  total = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_bag_count_l113_11370


namespace NUMINAMATH_CALUDE_sugar_remaining_l113_11328

/-- Given 24 kilos of sugar divided into 4 bags with specified losses, 
    prove that 19.8 kilos of sugar remain. -/
theorem sugar_remaining (total_sugar : ℝ) (num_bags : ℕ) 
  (loss1 loss2 loss3 loss4 : ℝ) :
  total_sugar = 24 ∧ 
  num_bags = 4 ∧ 
  loss1 = 0.1 ∧ 
  loss2 = 0.15 ∧ 
  loss3 = 0.2 ∧ 
  loss4 = 0.25 → 
  (total_sugar / num_bags) * 
    ((1 - loss1) + (1 - loss2) + (1 - loss3) + (1 - loss4)) = 19.8 :=
by sorry

end NUMINAMATH_CALUDE_sugar_remaining_l113_11328


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l113_11388

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.000000007 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = -9 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l113_11388


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l113_11334

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 6*x + 4 = 0 ↔ (x - 3)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l113_11334


namespace NUMINAMATH_CALUDE_min_value_of_expression_l113_11326

theorem min_value_of_expression (x : ℝ) : 4^x - 2^x + 2 ≥ (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l113_11326


namespace NUMINAMATH_CALUDE_domain_of_f_l113_11302

theorem domain_of_f (m : ℝ) : 
  (∀ x : ℝ, (m^2 - 3*m + 2)*x^2 + (m - 1)*x + 1 > 0) ↔ (m > 7/3 ∨ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_l113_11302


namespace NUMINAMATH_CALUDE_complex_modulus_reciprocal_l113_11364

theorem complex_modulus_reciprocal (z : ℂ) (h : (1 + z) / (1 + Complex.I) = 2 - Complex.I) :
  Complex.abs (1 / z) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_reciprocal_l113_11364


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l113_11374

theorem smallest_positive_integer_with_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
  x % 5 = 4 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧ 
  ∀ (y : ℕ), y > 0 → 
    y % 5 = 4 → 
    y % 7 = 6 → 
    y % 8 = 7 → 
    x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l113_11374


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l113_11351

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 37 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 37) = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l113_11351


namespace NUMINAMATH_CALUDE_inequality_solution_set_l113_11385

theorem inequality_solution_set (x : ℝ) : 
  (2 * x / 5 ≤ 3 + x ∧ 3 + x < 4 - x / 3) ↔ -5 ≤ x ∧ x < 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l113_11385


namespace NUMINAMATH_CALUDE_stratified_sampling_girls_l113_11338

theorem stratified_sampling_girls (total_students : ℕ) (sample_size : ℕ) (girls_in_sample : ℕ) 
  (h1 : total_students = 2000)
  (h2 : sample_size = 200)
  (h3 : girls_in_sample = 103) :
  (girls_in_sample : ℚ) / sample_size * total_students = 970 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_girls_l113_11338


namespace NUMINAMATH_CALUDE_largest_minus_smallest_difference_l113_11398

def digits : List Nat := [3, 9, 6, 0, 5, 1, 7]

def largest_number (ds : List Nat) : Nat :=
  sorry

def smallest_number (ds : List Nat) : Nat :=
  sorry

theorem largest_minus_smallest_difference :
  largest_number digits - smallest_number digits = 8729631 := by
  sorry

end NUMINAMATH_CALUDE_largest_minus_smallest_difference_l113_11398


namespace NUMINAMATH_CALUDE_digit_equation_sum_l113_11373

theorem digit_equation_sum : 
  ∃ (Y M E T : ℕ), 
    Y < 10 ∧ M < 10 ∧ E < 10 ∧ T < 10 ∧  -- digits are less than 10
    Y ≠ M ∧ Y ≠ E ∧ Y ≠ T ∧ M ≠ E ∧ M ≠ T ∧ E ≠ T ∧  -- digits are unique
    (10 * Y + E) * (10 * M + E) = T * T * T ∧  -- (YE) * (ME) = T * T * T
    T % 2 = 0 ∧  -- T is even
    E + M + T + Y = 10 :=  -- sum equals 10
by sorry

end NUMINAMATH_CALUDE_digit_equation_sum_l113_11373


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l113_11386

theorem tan_alpha_plus_pi_fourth (α : ℝ) (M : ℝ × ℝ) :
  M.1 = 1 ∧ M.2 = Real.sqrt 3 →
  (∃ t : ℝ, t > 0 ∧ t * M.1 = 1 ∧ t * M.2 = Real.tan α) →
  Real.tan (α + π / 4) = -2 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l113_11386


namespace NUMINAMATH_CALUDE_mango_purchase_quantity_l113_11343

/-- Calculates the quantity of mangoes purchased given the total payment, apple quantity, apple price, and mango price -/
def mango_quantity (total_payment : ℕ) (apple_quantity : ℕ) (apple_price : ℕ) (mango_price : ℕ) : ℕ :=
  ((total_payment - apple_quantity * apple_price) / mango_price)

/-- Theorem stating that the quantity of mangoes purchased is 9 kg -/
theorem mango_purchase_quantity :
  mango_quantity 1055 8 70 55 = 9 := by
  sorry

end NUMINAMATH_CALUDE_mango_purchase_quantity_l113_11343


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l113_11303

def M : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def N : Set ℝ := {x : ℝ | x ≤ -3}

theorem union_of_M_and_N : M ∪ N = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l113_11303


namespace NUMINAMATH_CALUDE_triangle_inequality_l113_11312

theorem triangle_inequality (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  Real.sqrt x + Real.sqrt y > Real.sqrt z →
  Real.sqrt y + Real.sqrt z > Real.sqrt x →
  Real.sqrt z + Real.sqrt x > Real.sqrt y →
  x / y + y / z + z / x = 5 →
  x * (y^2 - 2*z^2) / z + y * (z^2 - 2*x^2) / x + z * (x^2 - 2*y^2) / y ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l113_11312


namespace NUMINAMATH_CALUDE_playground_fence_length_l113_11352

/-- The side length of the square fence around the playground -/
def playground_side_length : ℝ := 27

/-- The length of the garden -/
def garden_length : ℝ := 12

/-- The width of the garden -/
def garden_width : ℝ := 9

/-- The total fencing for both the playground and the garden -/
def total_fencing : ℝ := 150

/-- Theorem stating that the side length of the square fence around the playground is 27 yards -/
theorem playground_fence_length :
  4 * playground_side_length + 2 * (garden_length + garden_width) = total_fencing :=
sorry

end NUMINAMATH_CALUDE_playground_fence_length_l113_11352


namespace NUMINAMATH_CALUDE_new_person_weight_l113_11397

theorem new_person_weight (n : ℕ) (initial_weight replaced_weight avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : replaced_weight = 70)
  (h3 : avg_increase = 2.5) :
  let new_weight := replaced_weight + n * avg_increase
  new_weight = 90 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l113_11397


namespace NUMINAMATH_CALUDE_sugar_price_increase_l113_11396

theorem sugar_price_increase (initial_price : ℝ) (consumption_reduction : ℝ) (new_price : ℝ) :
  initial_price = 2 →
  consumption_reduction = 0.6 →
  (1 - consumption_reduction) * new_price = initial_price →
  new_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_sugar_price_increase_l113_11396


namespace NUMINAMATH_CALUDE_cost_of_pens_l113_11341

/-- Given a pack of 150 pens costs $45, prove that the cost of 3600 pens is $1080 -/
theorem cost_of_pens (pack_size : ℕ) (pack_cost : ℝ) (total_pens : ℕ) :
  pack_size = 150 →
  pack_cost = 45 →
  total_pens = 3600 →
  (total_pens : ℝ) * (pack_cost / pack_size) = 1080 := by
sorry

end NUMINAMATH_CALUDE_cost_of_pens_l113_11341


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l113_11377

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 2) ↔ (∃ x : ℝ, x^2 < 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l113_11377


namespace NUMINAMATH_CALUDE_triangle_formation_range_l113_11371

theorem triangle_formation_range (x : ℝ) : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let D : ℝ × ℝ := (x, 0)
  let C : ℝ × ℝ := (x, Real.sqrt (1 - x^2))
  let AD := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let BD := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  (AD + BD > CD ∧ AD + CD > BD ∧ BD + CD > AD) ↔ 
  (x > 2 - Real.sqrt 5 ∧ x < Real.sqrt 5 - 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_range_l113_11371


namespace NUMINAMATH_CALUDE_honey_harvest_increase_l113_11340

theorem honey_harvest_increase (last_year harvest_this_year increase : ℕ) : 
  last_year = 2479 → 
  harvest_this_year = 8564 → 
  increase = harvest_this_year - last_year → 
  increase = 6085 := by
  sorry

end NUMINAMATH_CALUDE_honey_harvest_increase_l113_11340


namespace NUMINAMATH_CALUDE_find_other_number_l113_11383

theorem find_other_number (a b : ℤ) (h1 : 3 * a + 4 * b = 161) (h2 : a = 17 ∨ b = 17) : 
  (a = 31 ∨ b = 31) :=
sorry

end NUMINAMATH_CALUDE_find_other_number_l113_11383


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l113_11342

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : 
  (∀ x : ℝ, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l113_11342


namespace NUMINAMATH_CALUDE_distance_calculation_l113_11322

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 74

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- Time difference between Maxwell's start and Brad's start in hours -/
def time_difference : ℝ := 1

/-- Total time until Maxwell and Brad meet in hours -/
def total_time : ℝ := 8

theorem distance_calculation :
  distance_between_homes = 
    maxwell_speed * total_time + 
    brad_speed * (total_time - time_difference) :=
by sorry

end NUMINAMATH_CALUDE_distance_calculation_l113_11322


namespace NUMINAMATH_CALUDE_roses_sold_l113_11325

theorem roses_sold (initial : ℕ) (picked : ℕ) (final : ℕ) (sold : ℕ) : 
  initial = 37 → picked = 19 → final = 40 → 
  initial - sold + picked = final → 
  sold = 16 := by
sorry

end NUMINAMATH_CALUDE_roses_sold_l113_11325


namespace NUMINAMATH_CALUDE_goods_train_speed_l113_11387

/-- Proves that the speed of a goods train is 100 km/h given specific conditions --/
theorem goods_train_speed (man_train_speed : ℝ) (passing_time : ℝ) (goods_train_length : ℝ) :
  man_train_speed = 80 →
  passing_time = 8 →
  goods_train_length = 400 →
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 100 ∧
    (goods_train_speed + man_train_speed) * (5 / 18) * passing_time = goods_train_length :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l113_11387


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_sixth_l113_11327

theorem reciprocal_of_negative_one_sixth :
  ∃ x : ℚ, x * (-1/6 : ℚ) = 1 ∧ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_sixth_l113_11327


namespace NUMINAMATH_CALUDE_additional_savings_when_combined_l113_11395

/-- The regular price of a window -/
def window_price : ℕ := 120

/-- The number of windows that need to be bought to get one free -/
def windows_for_free : ℕ := 6

/-- The number of windows Dave needs -/
def dave_windows : ℕ := 9

/-- The number of windows Doug needs -/
def doug_windows : ℕ := 10

/-- Calculate the cost of windows with the offer -/
def cost_with_offer (n : ℕ) : ℕ :=
  ((n + windows_for_free - 1) / windows_for_free * windows_for_free) * window_price

/-- Calculate the savings for a given number of windows -/
def savings (n : ℕ) : ℕ :=
  n * window_price - cost_with_offer n

/-- The theorem to be proved -/
theorem additional_savings_when_combined :
  savings (dave_windows + doug_windows) - (savings dave_windows + savings doug_windows) = 240 := by
  sorry

end NUMINAMATH_CALUDE_additional_savings_when_combined_l113_11395


namespace NUMINAMATH_CALUDE_sqrt_81_equals_9_l113_11345

theorem sqrt_81_equals_9 : Real.sqrt 81 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_equals_9_l113_11345


namespace NUMINAMATH_CALUDE_securities_stamp_duty_difference_l113_11359

/-- The securities transaction stamp duty problem -/
theorem securities_stamp_duty_difference :
  let old_rate : ℚ := 3 / 1000
  let new_rate : ℚ := 1 / 1000
  let purchase_value : ℚ := 100000
  (purchase_value * old_rate - purchase_value * new_rate) = 200 := by
  sorry

end NUMINAMATH_CALUDE_securities_stamp_duty_difference_l113_11359


namespace NUMINAMATH_CALUDE_car_speed_first_hour_l113_11330

/-- Proves that given a car's speed in the second hour and its average speed over two hours, we can determine its speed in the first hour. -/
theorem car_speed_first_hour 
  (speed_second_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_second_hour = 80) 
  (h2 : average_speed = 90) : 
  ∃ (speed_first_hour : ℝ), 
    speed_first_hour = 100 ∧ 
    average_speed = (speed_first_hour + speed_second_hour) / 2 := by
  sorry

#check car_speed_first_hour

end NUMINAMATH_CALUDE_car_speed_first_hour_l113_11330


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l113_11390

theorem fifteenth_student_age
  (total_students : Nat)
  (avg_age_all : ℝ)
  (num_group1 : Nat)
  (avg_age_group1 : ℝ)
  (num_group2 : Nat)
  (avg_age_group2 : ℝ)
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 8)
  (h4 : avg_age_group1 = 14)
  (h5 : num_group2 = 6)
  (h6 : avg_age_group2 = 16)
  : ℝ :=
by
  sorry

#check fifteenth_student_age

end NUMINAMATH_CALUDE_fifteenth_student_age_l113_11390


namespace NUMINAMATH_CALUDE_intersection_polar_coords_l113_11353

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 2
def C₂ (t x y : ℝ) : Prop := x = 2 - t ∧ y = t

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop :=
  C₁ x y ∧ ∃ t, C₂ t x y

-- Define polar coordinates
def polar_coords (x y ρ θ : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem statement
theorem intersection_polar_coords :
  ∃ x y : ℝ, intersection_point x y ∧ 
  polar_coords x y (Real.sqrt 2) (Real.pi / 4) :=
sorry

end NUMINAMATH_CALUDE_intersection_polar_coords_l113_11353


namespace NUMINAMATH_CALUDE_sum_of_ages_l113_11399

/-- Given the ages of siblings and cousins, calculate the sum of their ages. -/
theorem sum_of_ages (juliet ralph maggie nicky lucy lily alex : ℕ) : 
  juliet = 10 ∧ 
  juliet = maggie + 3 ∧ 
  ralph = juliet + 2 ∧ 
  nicky * 2 = ralph ∧ 
  lucy = ralph + 1 ∧ 
  lily = ralph + 1 ∧ 
  alex + 5 = lucy → 
  maggie + ralph + nicky + lucy + lily + alex = 59 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l113_11399


namespace NUMINAMATH_CALUDE_second_table_trays_count_l113_11376

/-- Represents the number of trays Jerry picked up -/
structure TrayPickup where
  capacity : Nat
  firstTable : Nat
  trips : Nat
  total : Nat

/-- Calculates the number of trays picked up from the second table -/
def secondTableTrays (pickup : TrayPickup) : Nat :=
  pickup.total - pickup.firstTable

/-- Theorem stating the number of trays picked up from the second table -/
theorem second_table_trays_count (pickup : TrayPickup) 
  (h1 : pickup.capacity = 8)
  (h2 : pickup.firstTable = 9)
  (h3 : pickup.trips = 2)
  (h4 : pickup.total = pickup.capacity * pickup.trips) :
  secondTableTrays pickup = 7 := by
  sorry

#check second_table_trays_count

end NUMINAMATH_CALUDE_second_table_trays_count_l113_11376


namespace NUMINAMATH_CALUDE_coffee_needed_l113_11348

/-- The amount of coffee needed for Taylor's house guests -/
theorem coffee_needed (cups_weak cups_strong : ℕ) 
  (h1 : cups_weak = cups_strong)
  (h2 : cups_weak + cups_strong = 24) : ℕ :=
by
  sorry

#check coffee_needed

end NUMINAMATH_CALUDE_coffee_needed_l113_11348


namespace NUMINAMATH_CALUDE_coin_problem_l113_11355

/-- Represents the value of a coin in paise -/
inductive CoinValue
  | paise20 : CoinValue
  | paise25 : CoinValue

/-- Calculates the total value in rupees given the number of coins of each type -/
def totalValueInRupees (coins20 : ℕ) (coins25 : ℕ) : ℚ :=
  (coins20 * 20 + coins25 * 25) / 100

theorem coin_problem :
  let totalCoins : ℕ := 344
  let coins20 : ℕ := 300
  let coins25 : ℕ := totalCoins - coins20
  totalValueInRupees coins20 coins25 = 71 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l113_11355


namespace NUMINAMATH_CALUDE_marble_difference_l113_11389

/-- The number of marbles Amon and Rhonda have combined -/
def total_marbles : ℕ := 215

/-- The number of marbles Rhonda has -/
def rhonda_marbles : ℕ := 80

/-- Amon has more marbles than Rhonda -/
axiom amon_has_more : ∃ (amon_marbles : ℕ), amon_marbles > rhonda_marbles ∧ amon_marbles + rhonda_marbles = total_marbles

/-- The difference between Amon's and Rhonda's marbles is 55 -/
theorem marble_difference : ∃ (amon_marbles : ℕ), amon_marbles - rhonda_marbles = 55 := by sorry

end NUMINAMATH_CALUDE_marble_difference_l113_11389


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l113_11321

theorem complex_fraction_simplification :
  (1 - 2*Complex.I) / (2 + Complex.I) = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l113_11321


namespace NUMINAMATH_CALUDE_det_trig_matrix_zero_l113_11310

/-- The determinant of a specific 3x3 matrix involving trigonometric functions is zero. -/
theorem det_trig_matrix_zero (θ φ : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![0, 2 * Real.sin θ, - Real.cos θ;
                                       -2 * Real.sin θ, 0, Real.sin φ;
                                       Real.cos θ, - Real.sin φ, 0]
  Matrix.det M = 0 := by sorry

end NUMINAMATH_CALUDE_det_trig_matrix_zero_l113_11310


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l113_11304

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l113_11304


namespace NUMINAMATH_CALUDE_john_work_hours_l113_11362

def hours_per_day : ℕ := 8
def start_day : ℕ := 3
def end_day : ℕ := 8

def total_days : ℕ := end_day - start_day

theorem john_work_hours : hours_per_day * total_days = 40 := by
  sorry

end NUMINAMATH_CALUDE_john_work_hours_l113_11362


namespace NUMINAMATH_CALUDE_function_decomposition_l113_11335

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ (f_even f_odd : ℝ → ℝ),
    (∀ x, f x = f_even x + f_odd x) ∧
    (∀ x, f_even (-x) = f_even x) ∧
    (∀ x, f_odd (-x) = -f_odd x) :=
by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l113_11335


namespace NUMINAMATH_CALUDE_sum_of_cubes_l113_11360

theorem sum_of_cubes (x y : ℝ) (h1 : x * y = 15) (h2 : x + y = 11) :
  x^3 + y^3 = 836 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l113_11360


namespace NUMINAMATH_CALUDE_ellipse_triangle_problem_l113_11384

-- Define the ellipse
def ellipse (x y : ℝ) (b : ℝ) : Prop := x^2/4 + y^2/b^2 = 1

-- Define the line L
def line_L (x y : ℝ) : Prop := y = x + 2

-- Define parallel lines
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the problem statement
theorem ellipse_triangle_problem 
  (b : ℝ) 
  (ABC : Triangle) 
  (h_ellipse : ellipse ABC.A.1 ABC.A.2 b ∧ ellipse ABC.B.1 ABC.B.2 b)
  (h_C_on_L : line_L ABC.C.1 ABC.C.2)
  (h_AB_parallel_L : parallel ((ABC.B.2 - ABC.A.2) / (ABC.B.1 - ABC.A.1)) 1)
  (h_eccentricity : b^2 = 4/3) :
  (∀ (O : ℝ × ℝ), O = (0, 0) → (ABC.A.1 - O.1) * (ABC.B.2 - O.2) = (ABC.A.2 - O.2) * (ABC.B.1 - O.1) →
    (ABC.B.1 - ABC.A.1)^2 + (ABC.B.2 - ABC.A.2)^2 = 8 ∧ 
    (ABC.B.1 - ABC.A.1) * (ABC.C.2 - ABC.A.2) - (ABC.B.2 - ABC.A.2) * (ABC.C.1 - ABC.A.1) = 4) ∧
  (∀ (m : ℝ), (ABC.B.1 - ABC.A.1)^2 + (ABC.B.2 - ABC.A.2)^2 = (ABC.C.1 - ABC.A.1)^2 + (ABC.C.2 - ABC.A.2)^2 →
    (ABC.C.1 - ABC.A.1)^2 + (ABC.C.2 - ABC.A.2)^2 ≥ (ABC.C.1 - ABC.B.1)^2 + (ABC.C.2 - ABC.B.2)^2 →
    ABC.B.2 - ABC.A.2 = ABC.B.1 - ABC.A.1 - (ABC.B.1 - ABC.A.1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_problem_l113_11384


namespace NUMINAMATH_CALUDE_initial_doctors_count_l113_11346

theorem initial_doctors_count (initial_nurses : ℕ) (remaining_staff : ℕ) : initial_nurses = 18 → remaining_staff = 22 → ∃ initial_doctors : ℕ, initial_doctors = 11 ∧ initial_doctors + initial_nurses - 5 - 2 = remaining_staff :=
by
  sorry

end NUMINAMATH_CALUDE_initial_doctors_count_l113_11346


namespace NUMINAMATH_CALUDE_min_value_expression_l113_11357

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + 3 ≥ 9 ∧
  ((x + y) / z + (x + z) / y + (y + z) / x + 3 = 9 ↔ x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l113_11357


namespace NUMINAMATH_CALUDE_expression_value_at_three_l113_11308

theorem expression_value_at_three :
  let x : ℝ := 3
  let expr := (Real.sqrt (x - 2 * Real.sqrt 2) / Real.sqrt (x^2 - 4 * Real.sqrt 2 * x + 8)) -
              (Real.sqrt (x + 2 * Real.sqrt 2) / Real.sqrt (x^2 + 4 * Real.sqrt 2 * x + 8))
  expr = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l113_11308


namespace NUMINAMATH_CALUDE_triangle_properties_l113_11339

/-- Represents a triangle with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem stating properties of triangles -/
theorem triangle_properties (t : Triangle) :
  (t.A > t.B → Real.sin t.A > Real.sin t.B) ∧
  (t.A = π / 6 ∧ t.b = 4 ∧ t.a = 3 → ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ 
    t1.A = t.A ∧ t1.b = t.b ∧ t1.a = t.a ∧
    t2.A = t.A ∧ t2.b = t.b ∧ t2.a = t.a) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l113_11339


namespace NUMINAMATH_CALUDE_percentage_of_120_to_50_l113_11365

theorem percentage_of_120_to_50 : 
  (120 : ℝ) / 50 * 100 = 240 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_120_to_50_l113_11365


namespace NUMINAMATH_CALUDE_largest_five_digit_negative_congruent_to_one_mod_23_l113_11378

theorem largest_five_digit_negative_congruent_to_one_mod_23 :
  ∀ n : ℤ, -99999 ≤ n ∧ n < -9999 ∧ n ≡ 1 [ZMOD 23] → n ≤ -9993 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_negative_congruent_to_one_mod_23_l113_11378


namespace NUMINAMATH_CALUDE_allowance_increase_l113_11309

theorem allowance_increase (base_amount : ℝ) (middle_school_extra : ℝ) (percentage_increase : ℝ) : 
  let middle_school_allowance := base_amount + middle_school_extra
  let senior_year_allowance := middle_school_allowance * (1 + percentage_increase / 100)
  base_amount = 8 ∧ middle_school_extra = 2 ∧ percentage_increase = 150 →
  senior_year_allowance - 2 * middle_school_allowance = 5 := by sorry

end NUMINAMATH_CALUDE_allowance_increase_l113_11309


namespace NUMINAMATH_CALUDE_social_media_time_theorem_l113_11336

/-- Calculates the weekly time spent on social media given daily phone usage and social media ratio -/
def weekly_social_media_time (daily_phone_time : ℝ) (social_media_ratio : ℝ) : ℝ :=
  daily_phone_time * social_media_ratio * 7

/-- Theorem: Given 6 hours of daily phone usage and half spent on social media, 
    the weekly social media time is 21 hours -/
theorem social_media_time_theorem :
  weekly_social_media_time 6 0.5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_social_media_time_theorem_l113_11336


namespace NUMINAMATH_CALUDE_vector_sum_l113_11324

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (x, 1)
def b (y : ℝ) : ℝ × ℝ := (1, y)
def c : ℝ × ℝ := (2, -4)

-- Define perpendicularity for 2D vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Define parallelism for 2D vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

-- Theorem statement
theorem vector_sum (x y : ℝ) 
  (h1 : perpendicular (a x) c) 
  (h2 : parallel (b y) c) : 
  a x + b y = (3, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_l113_11324


namespace NUMINAMATH_CALUDE_angle_A_range_l113_11320

theorem angle_A_range (a b c : ℝ) (A : ℝ) :
  a = 2 →
  b = 2 * Real.sqrt 2 →
  c^2 = a^2 + b^2 - 2*a*b * Real.cos A →
  0 < A ∧ A ≤ π/4 :=
by sorry

end NUMINAMATH_CALUDE_angle_A_range_l113_11320


namespace NUMINAMATH_CALUDE_train_length_l113_11332

/-- The length of a train given its speed and time to pass a stationary object -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 36 → speed * time * (1000 / 3600) = 630 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l113_11332


namespace NUMINAMATH_CALUDE_probability_problem_l113_11311

theorem probability_problem (P : Set α → ℝ) (A B : Set α)
  (h1 : P (A ∩ B) = 1/6)
  (h2 : P (Aᶜ) = 2/3)
  (h3 : P B = 1/2) :
  (P (A ∩ B) ≠ 0 ∧ P A * P B = P (A ∩ B)) :=
by sorry

end NUMINAMATH_CALUDE_probability_problem_l113_11311


namespace NUMINAMATH_CALUDE_percentage_less_than_50k_l113_11394

/-- Represents the percentage of counties in each population category -/
structure PopulationDistribution :=
  (less_than_50k : ℝ)
  (between_50k_and_150k : ℝ)
  (more_than_150k : ℝ)

/-- The given population distribution from the pie chart -/
def given_distribution : PopulationDistribution :=
  { less_than_50k := 35,
    between_50k_and_150k := 40,
    more_than_150k := 25 }

/-- Theorem stating that the percentage of counties with fewer than 50,000 residents is 35% -/
theorem percentage_less_than_50k (dist : PopulationDistribution) 
  (h1 : dist = given_distribution) : 
  dist.less_than_50k = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_less_than_50k_l113_11394


namespace NUMINAMATH_CALUDE_min_value_3a_plus_2_l113_11375

theorem min_value_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) :
  ∃ (m : ℝ), (3 * a + 2 ≥ m) ∧ (∀ (x : ℝ), 8 * x^2 + 10 * x + 6 = 2 → 3 * x + 2 ≥ m) ∧ m = -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_3a_plus_2_l113_11375


namespace NUMINAMATH_CALUDE_unique_y_value_l113_11358

theorem unique_y_value : ∃! y : ℝ, y > 0 ∧ (y / 100) * y = 9 := by sorry

end NUMINAMATH_CALUDE_unique_y_value_l113_11358


namespace NUMINAMATH_CALUDE_max_xy_over_x2_plus_y2_l113_11393

theorem max_xy_over_x2_plus_y2 (x y : ℝ) 
  (hx : 1/3 ≤ x ∧ x ≤ 3/5) (hy : 1/4 ≤ y ∧ y ≤ 1/2) :
  (x * y) / (x^2 + y^2) ≤ 6/13 :=
sorry

end NUMINAMATH_CALUDE_max_xy_over_x2_plus_y2_l113_11393


namespace NUMINAMATH_CALUDE_fraction_addition_l113_11368

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l113_11368


namespace NUMINAMATH_CALUDE_sum_of_two_valid_numbers_l113_11379

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ),
    n = d1 * 100000000 + d2 * 10000000 + d3 * 1000000 + d4 * 100000 + 
        d5 * 10000 + d6 * 1000 + d7 * 100 + d8 * 10 + d9 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧ d1 ≠ d9 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧ d2 ≠ d9 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧ d3 ≠ d9 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧ d4 ≠ d9 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧ d5 ≠ d9 ∧
    d6 ≠ d7 ∧ d6 ≠ d8 ∧ d6 ≠ d9 ∧
    d7 ≠ d8 ∧ d7 ≠ d9 ∧
    d8 ≠ d9 ∧
    0 < d1 ∧ d1 ≤ 9 ∧ 0 < d2 ∧ d2 ≤ 9 ∧ 0 < d3 ∧ d3 ≤ 9 ∧ 0 < d4 ∧ d4 ≤ 9 ∧
    0 < d5 ∧ d5 ≤ 9 ∧ 0 < d6 ∧ d6 ≤ 9 ∧ 0 < d7 ∧ d7 ≤ 9 ∧ 0 < d8 ∧ d8 ≤ 9 ∧
    0 < d9 ∧ d9 ≤ 9

theorem sum_of_two_valid_numbers :
  ∃ (a b : ℕ), is_valid_number a ∧ is_valid_number b ∧ a + b = 987654321 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_two_valid_numbers_l113_11379


namespace NUMINAMATH_CALUDE_unique_intersection_l113_11354

/-- The value of b for which the graphs of y = bx^2 + 5x + 3 and y = -2x - 3 intersect at exactly one point -/
def b : ℚ := 49 / 24

/-- The first function: f(x) = bx^2 + 5x + 3 -/
def f (x : ℝ) : ℝ := b * x^2 + 5 * x + 3

/-- The second function: g(x) = -2x - 3 -/
def g (x : ℝ) : ℝ := -2 * x - 3

/-- Theorem stating that the graphs of f and g intersect at exactly one point -/
theorem unique_intersection : ∃! x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_unique_intersection_l113_11354


namespace NUMINAMATH_CALUDE_red_minus_white_equals_three_l113_11333

-- Define the flower counts for each category
def total_flowers : ℕ := 100
def yellow_white : ℕ := 13
def red_yellow : ℕ := 17
def red_white : ℕ := 14
def blue_yellow : ℕ := 16
def blue_white : ℕ := 9
def red_blue_yellow : ℕ := 8
def red_white_blue : ℕ := 6

-- Define the number of flowers containing red
def red_flowers : ℕ := red_yellow + red_white + red_blue_yellow + red_white_blue

-- Define the number of flowers containing white
def white_flowers : ℕ := yellow_white + red_white + blue_white + red_white_blue

-- Theorem statement
theorem red_minus_white_equals_three :
  red_flowers - white_flowers = 3 :=
by sorry

end NUMINAMATH_CALUDE_red_minus_white_equals_three_l113_11333


namespace NUMINAMATH_CALUDE_total_presents_l113_11319

theorem total_presents (christmas : ℕ) (easter : ℕ) (birthday : ℕ) (halloween : ℕ) : 
  christmas = 60 →
  birthday = 3 * easter →
  easter = christmas / 2 - 10 →
  halloween = birthday - easter →
  christmas + easter + birthday + halloween = 180 := by
sorry

end NUMINAMATH_CALUDE_total_presents_l113_11319


namespace NUMINAMATH_CALUDE_odd_square_minus_one_divisible_by_24_l113_11331

theorem odd_square_minus_one_divisible_by_24 (n : ℤ) : 
  Odd (n^2) → (n^2 % 9 ≠ 0) → (n^2 - 1) % 24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_divisible_by_24_l113_11331


namespace NUMINAMATH_CALUDE_angle_sum_quarter_range_l113_11317

-- Define acute and obtuse angles
def acute_angle (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2
def obtuse_angle (β : Real) : Prop := Real.pi / 2 < β ∧ β < Real.pi

-- Theorem statement
theorem angle_sum_quarter_range (α β : Real) 
  (h_acute : acute_angle α) (h_obtuse : obtuse_angle β) :
  Real.pi / 8 < (α + β) / 4 ∧ (α + β) / 4 < 3 * Real.pi / 8 := by
  sorry

#check angle_sum_quarter_range

end NUMINAMATH_CALUDE_angle_sum_quarter_range_l113_11317


namespace NUMINAMATH_CALUDE_ice_cream_volume_l113_11356

/-- The volume of ice cream in a cone and sphere -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1 / 3) * π * r^2 * h
  let sphere_volume := (4 / 3) * π * r^3
  h = 12 ∧ r = 3 → cone_volume + sphere_volume = 72 * π := by sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l113_11356


namespace NUMINAMATH_CALUDE_prime_cube_plus_one_l113_11307

theorem prime_cube_plus_one (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ p^x = y^3 + 1) ↔ p = 2 ∨ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_plus_one_l113_11307


namespace NUMINAMATH_CALUDE_lauren_reaches_andrea_in_30_minutes_l113_11361

/-- Represents the scenario of Andrea and Lauren biking towards each other --/
structure BikingScenario where
  initial_distance : ℝ
  andrea_speed : ℝ
  lauren_speed : ℝ
  decrease_rate : ℝ
  flat_tire_time : ℝ
  lauren_delay : ℝ

/-- Calculates the total time for Lauren to reach Andrea --/
def totalTime (scenario : BikingScenario) : ℝ :=
  sorry

/-- The theorem stating that Lauren reaches Andrea after 30 minutes --/
theorem lauren_reaches_andrea_in_30_minutes (scenario : BikingScenario)
  (h1 : scenario.initial_distance = 30)
  (h2 : scenario.andrea_speed = 2 * scenario.lauren_speed)
  (h3 : scenario.decrease_rate = 2)
  (h4 : scenario.flat_tire_time = 10)
  (h5 : scenario.lauren_delay = 5) :
  totalTime scenario = 30 :=
sorry

end NUMINAMATH_CALUDE_lauren_reaches_andrea_in_30_minutes_l113_11361


namespace NUMINAMATH_CALUDE_square_area_from_rectangles_l113_11369

/-- The area of a square formed by three identical rectangles -/
theorem square_area_from_rectangles (width : ℝ) (h1 : width = 4) : 
  let length := 3 * width
  let square_side := length + width
  square_side ^ 2 = 256 := by sorry

end NUMINAMATH_CALUDE_square_area_from_rectangles_l113_11369


namespace NUMINAMATH_CALUDE_second_divisor_problem_l113_11323

theorem second_divisor_problem (x : ℚ) : 
  (((377 / 13) / x) * (1 / 4)) / 2 = 0.125 → x = 29 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l113_11323


namespace NUMINAMATH_CALUDE_distribute_7_4_l113_11318

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 indistinguishable balls into 4 distinguishable boxes is 128 -/
theorem distribute_7_4 : distribute 7 4 = 128 := by sorry

end NUMINAMATH_CALUDE_distribute_7_4_l113_11318


namespace NUMINAMATH_CALUDE_third_root_of_cubic_l113_11305

theorem third_root_of_cubic (a b : ℚ) :
  (∀ x : ℚ, a * x^3 + (a + 2*b) * x^2 + (b - 3*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = -3/17) :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_cubic_l113_11305


namespace NUMINAMATH_CALUDE_range_of_a_given_quadratic_inequality_l113_11300

theorem range_of_a_given_quadratic_inequality (a : ℝ) : 
  (∀ x : ℝ, 4 * x^2 + (a - 2) * x + (1/4 : ℝ) > 0) → 
  0 < a ∧ a < 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_given_quadratic_inequality_l113_11300


namespace NUMINAMATH_CALUDE_xin_xin_family_stay_l113_11329

/-- Represents a date with a month and day -/
structure Date where
  month : Nat
  day : Nat

/-- Calculates the number of nights between two dates -/
def nights_between (arrival : Date) (departure : Date) : Nat :=
  sorry

theorem xin_xin_family_stay :
  let arrival : Date := ⟨5, 30⟩  -- May 30
  let departure : Date := ⟨6, 4⟩  -- June 4
  nights_between arrival departure = 5 := by
  sorry

end NUMINAMATH_CALUDE_xin_xin_family_stay_l113_11329


namespace NUMINAMATH_CALUDE_not_expressible_as_difference_of_squares_l113_11367

theorem not_expressible_as_difference_of_squares (k x y : ℤ) : 
  ¬ (∃ n : ℤ, (n = 8*k + 3 ∨ n = 8*k + 5) ∧ n = x^2 - 2*y^2) :=
sorry

end NUMINAMATH_CALUDE_not_expressible_as_difference_of_squares_l113_11367


namespace NUMINAMATH_CALUDE_switch_connections_l113_11382

theorem switch_connections (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_switch_connections_l113_11382


namespace NUMINAMATH_CALUDE_arithmetic_polynomial_root_count_l113_11313

/-- Represents a polynomial of degree 5 with integer coefficients forming an arithmetic sequence. -/
structure ArithmeticPolynomial where
  a : ℤ
  d : ℤ  -- Common difference of the arithmetic sequence

/-- The number of integer roots (counting multiplicity) of an ArithmeticPolynomial. -/
def integerRootCount (p : ArithmeticPolynomial) : ℕ :=
  sorry

/-- Theorem stating the possible values for the number of integer roots. -/
theorem arithmetic_polynomial_root_count (p : ArithmeticPolynomial) :
  integerRootCount p ∈ ({0, 1, 2, 3, 5} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_polynomial_root_count_l113_11313


namespace NUMINAMATH_CALUDE_cost_to_replace_movies_l113_11349

/-- The cost to replace VHS movies with DVDs -/
theorem cost_to_replace_movies 
  (num_movies : ℕ) 
  (vhs_trade_in : ℚ) 
  (dvd_cost : ℚ) : 
  (num_movies : ℚ) * (dvd_cost - vhs_trade_in) = 800 :=
by
  sorry

#check cost_to_replace_movies 100 2 10

end NUMINAMATH_CALUDE_cost_to_replace_movies_l113_11349
