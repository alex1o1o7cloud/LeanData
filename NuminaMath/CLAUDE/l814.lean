import Mathlib

namespace NUMINAMATH_CALUDE_parabola_shift_l814_81490

/-- A parabola shifted 1 unit to the left -/
def shifted_parabola (x : ℝ) : ℝ := (x + 1)^2

/-- The original parabola -/
def original_parabola (x : ℝ) : ℝ := x^2

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l814_81490


namespace NUMINAMATH_CALUDE_mono_properties_l814_81496

/-- Represents a monomial with coefficient and variables --/
structure Monomial where
  coeff : ℤ
  vars : List (Char × ℕ)

/-- Calculate the degree of a monomial --/
def degree (m : Monomial) : ℕ :=
  m.vars.foldl (fun acc (_, exp) => acc + exp) 0

/-- The monomial -4mn^5 --/
def mono : Monomial :=
  { coeff := -4
    vars := [('m', 1), ('n', 5)] }

theorem mono_properties : (mono.coeff = -4) ∧ (degree mono = 6) := by
  sorry

end NUMINAMATH_CALUDE_mono_properties_l814_81496


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l814_81418

theorem imaginary_part_of_z (z : ℂ) : z = (1 : ℂ) / (1 + Complex.I) → z.im = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l814_81418


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l814_81461

/-- A quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  /-- Side length EF -/
  ef : ℝ
  /-- Side length FG -/
  fg : ℝ
  /-- Side length GH -/
  gh : ℝ
  /-- Side length HE -/
  he : ℝ
  /-- Area of EFGH -/
  area : ℝ
  /-- Extension ratio for EF -/
  ef_ratio : ℝ
  /-- Extension ratio for FG -/
  fg_ratio : ℝ
  /-- Extension ratio for GH -/
  gh_ratio : ℝ
  /-- Extension ratio for HE -/
  he_ratio : ℝ

/-- The area of the extended quadrilateral E'F'G'H' -/
def extended_area (q : ExtendedQuadrilateral) : ℝ := sorry

/-- Theorem stating the area of E'F'G'H' given specific conditions -/
theorem extended_quadrilateral_area 
  (q : ExtendedQuadrilateral)
  (h1 : q.ef = 5)
  (h2 : q.fg = 6)
  (h3 : q.gh = 7)
  (h4 : q.he = 8)
  (h5 : q.area = 12)
  (h6 : q.ef_ratio = 2)
  (h7 : q.fg_ratio = 3/2)
  (h8 : q.gh_ratio = 4/3)
  (h9 : q.he_ratio = 5/4) :
  extended_area q = 84 := by sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l814_81461


namespace NUMINAMATH_CALUDE_min_students_with_both_traits_l814_81469

theorem min_students_with_both_traits (total : ℕ) (blue_eyes : ℕ) (lunch_box : ℕ)
  (h1 : total = 35)
  (h2 : blue_eyes = 15)
  (h3 : lunch_box = 23)
  (h4 : blue_eyes ≤ total)
  (h5 : lunch_box ≤ total) :
  total - (total - blue_eyes) - (total - lunch_box) ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_students_with_both_traits_l814_81469


namespace NUMINAMATH_CALUDE_relationship_abc_l814_81483

theorem relationship_abc (a b c : ℚ) : 
  (2 * a + a = 1) → (2 * b + b = 2) → (3 * c + c = 2) → a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l814_81483


namespace NUMINAMATH_CALUDE_ben_peas_count_l814_81486

/-- The number of sugar snap peas Ben wants to pick initially -/
def total_peas : ℕ := 56

/-- The time it takes Ben to pick all the peas (in minutes) -/
def total_time : ℕ := 7

/-- The number of peas Ben can pick in 9 minutes -/
def peas_in_9_min : ℕ := 72

/-- The time it takes Ben to pick 72 peas (in minutes) -/
def time_for_72_peas : ℕ := 9

/-- Theorem stating that the number of sugar snap peas Ben wants to pick initially is 56 -/
theorem ben_peas_count : 
  (total_peas : ℚ) / total_time = (peas_in_9_min : ℚ) / time_for_72_peas ∧
  total_peas = 56 := by
  sorry


end NUMINAMATH_CALUDE_ben_peas_count_l814_81486


namespace NUMINAMATH_CALUDE_f_min_at_one_plus_inv_sqrt_three_l814_81420

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 6 * x + 1

-- State the theorem
theorem f_min_at_one_plus_inv_sqrt_three :
  ∃ (x_min : ℝ), x_min = 1 + 1 / Real.sqrt 3 ∧
  ∀ (x : ℝ), f x ≥ f x_min :=
by sorry

end NUMINAMATH_CALUDE_f_min_at_one_plus_inv_sqrt_three_l814_81420


namespace NUMINAMATH_CALUDE_ellipse_incircle_area_l814_81419

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


end NUMINAMATH_CALUDE_ellipse_incircle_area_l814_81419


namespace NUMINAMATH_CALUDE_ball_count_proof_l814_81451

theorem ball_count_proof (total : ℕ) (p_yellow p_blue p_red : ℚ) 
  (h_total : total = 80)
  (h_yellow : p_yellow = 1/4)
  (h_blue : p_blue = 7/20)
  (h_red : p_red = 2/5)
  (h_sum : p_yellow + p_blue + p_red = 1) :
  ∃ (yellow blue red : ℕ),
    yellow = 20 ∧ 
    blue = 28 ∧ 
    red = 32 ∧
    yellow + blue + red = total ∧
    (yellow : ℚ) / total = p_yellow ∧
    (blue : ℚ) / total = p_blue ∧
    (red : ℚ) / total = p_red :=
by
  sorry

end NUMINAMATH_CALUDE_ball_count_proof_l814_81451


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l814_81499

theorem greatest_divisor_with_remainders : Nat.gcd (3589 - 23) (5273 - 41) = 2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l814_81499


namespace NUMINAMATH_CALUDE_base8_175_equals_base10_125_l814_81473

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ :=
  let d₂ := n / 100
  let d₁ := (n / 10) % 10
  let d₀ := n % 10
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

theorem base8_175_equals_base10_125 : base8ToBase10 175 = 125 := by
  sorry

end NUMINAMATH_CALUDE_base8_175_equals_base10_125_l814_81473


namespace NUMINAMATH_CALUDE_cosine_inequality_l814_81449

theorem cosine_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : 0 < x^2 + y^2 ∧ x^2 + y^2 ≤ π) :
  1 + Real.cos (x * y) ≥ Real.cos x + Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_l814_81449


namespace NUMINAMATH_CALUDE_distinct_triangles_in_2x4_grid_l814_81405

/-- Represents a 2x4 grid of points -/
def Grid := Fin 2 × Fin 4

/-- Checks if three points in the grid are collinear -/
def collinear (p q r : Grid) : Prop := sorry

/-- The number of ways to choose 3 points from 8 points -/
def total_combinations : ℕ := Nat.choose 8 3

/-- The number of collinear triples in a 2x4 grid -/
def collinear_triples : ℕ := sorry

/-- The number of distinct triangles in a 2x4 grid -/
def distinct_triangles : ℕ := total_combinations - collinear_triples

theorem distinct_triangles_in_2x4_grid :
  distinct_triangles = 44 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_2x4_grid_l814_81405


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_possibilities_l814_81415

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
def similar_triangles (t1 t2 : Triangle) : Prop := sorry

/-- A triangle is defined by its three side lengths. -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- The perimeter of a triangle is the sum of its side lengths. -/
def perimeter (t : Triangle) : ℝ := t.side1 + t.side2 + t.side3

theorem similar_triangles_perimeter_possibilities :
  ∀ (t1 t2 : Triangle),
    similar_triangles t1 t2 →
    t1.side1 = 4 ∧ t1.side2 = 6 ∧ t1.side3 = 8 →
    (t2.side1 = 2 ∨ t2.side2 = 2 ∨ t2.side3 = 2) →
    (perimeter t2 = 4.5 ∨ perimeter t2 = 6 ∨ perimeter t2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_possibilities_l814_81415


namespace NUMINAMATH_CALUDE_unique_solution_from_distinct_points_l814_81439

/-- Given two distinct points on a line, the system of equations formed by these points always has a unique solution -/
theorem unique_solution_from_distinct_points 
  (k : ℝ) (a b₁ b₂ a₁ a₂ : ℝ) :
  (a ≠ 2) →  -- P₁ and P₂ are distinct
  (b₁ = k * a + 1) →  -- P₁ is on the line
  (b₂ = k * 2 + 1) →  -- P₂ is on the line
  ∃! (x y : ℝ), a₁ * x + b₁ * y = 1 ∧ a₂ * x + b₂ * y = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_from_distinct_points_l814_81439


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l814_81467

theorem unequal_gender_probability : 
  let n : ℕ := 12  -- Total number of grandchildren
  let p : ℚ := 1/2 -- Probability of each child being male (or female)
  -- Probability of unequal number of grandsons and granddaughters
  (1 : ℚ) - (n.choose (n/2) : ℚ) / 2^n = 793/1024 :=
by sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l814_81467


namespace NUMINAMATH_CALUDE_special_calculator_problem_l814_81462

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Applies the calculator's operation to a two-digit number -/
def calculator_operation (x : ℕ) : ℕ :=
  reverse_digits (2 * x) + 2

theorem special_calculator_problem (x : ℕ) :
  x ≥ 10 ∧ x < 100 → calculator_operation x = 27 → x = 26 := by
sorry

end NUMINAMATH_CALUDE_special_calculator_problem_l814_81462


namespace NUMINAMATH_CALUDE_car_A_time_l814_81440

/-- Proves that Car A takes 8 hours to reach its destination given the specified conditions -/
theorem car_A_time (speed_A speed_B time_B : ℝ) (ratio : ℝ) : 
  speed_A = 50 →
  speed_B = 25 →
  time_B = 4 →
  ratio = 4 →
  speed_A * (ratio * speed_B * time_B) / speed_A = 8 :=
by
  sorry

#check car_A_time

end NUMINAMATH_CALUDE_car_A_time_l814_81440


namespace NUMINAMATH_CALUDE_sin_sum_of_roots_l814_81441

theorem sin_sum_of_roots (a b c : ℝ) (α β : ℝ) :
  (0 < α) → (α < π) →
  (0 < β) → (β < π) →
  (α ≠ β) →
  (a * Real.cos α + b * Real.sin α + c = 0) →
  (a * Real.cos β + b * Real.sin β + c = 0) →
  Real.sin (α + β) = (2 * a * b) / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_roots_l814_81441


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l814_81452

theorem solve_exponential_equation :
  ∃ x : ℝ, 4^x = Real.sqrt 64 ∧ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l814_81452


namespace NUMINAMATH_CALUDE_complex_sum_power_l814_81497

theorem complex_sum_power (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^96 + z^97 + z^98 + z^99 + z^100 + z^101 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_power_l814_81497


namespace NUMINAMATH_CALUDE_circle_diameter_l814_81427

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 9 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l814_81427


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_l814_81437

theorem imaginary_part_of_complex (z : ℂ) :
  z = 2 + (1 / (3 * I)) → z.im = -1/3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_l814_81437


namespace NUMINAMATH_CALUDE_mary_baking_cake_l814_81463

theorem mary_baking_cake (total_flour total_sugar remaining_flour_diff : ℕ) 
  (h1 : total_flour = 9)
  (h2 : total_sugar = 6)
  (h3 : remaining_flour_diff = 7) :
  total_sugar - (total_flour - remaining_flour_diff) = 4 := by
  sorry

end NUMINAMATH_CALUDE_mary_baking_cake_l814_81463


namespace NUMINAMATH_CALUDE_dessert_division_l814_81485

/-- Represents the number of dessert items -/
structure DessertItems where
  cinnamon_swirls : ℕ
  brownie_bites : ℕ
  fruit_tartlets : ℕ

/-- Represents the number of people sharing the desserts -/
def num_people : ℕ := 8

/-- The actual dessert items from the problem -/
def desserts : DessertItems := {
  cinnamon_swirls := 15,
  brownie_bites := 24,
  fruit_tartlets := 18
}

/-- Theorem stating that brownie bites can be equally divided, while others cannot -/
theorem dessert_division (d : DessertItems) (p : ℕ) (h_p : p = num_people) :
  d.brownie_bites / p = 3 ∧
  ¬(∃ (n : ℕ), n * p = d.cinnamon_swirls) ∧
  ¬(∃ (m : ℕ), m * p = d.fruit_tartlets) :=
sorry

end NUMINAMATH_CALUDE_dessert_division_l814_81485


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l814_81424

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  8 * a^3 + 27 * b^3 + 125 * c^3 + 1 / (a * b * c) ≥ 10 * Real.sqrt 6 :=
by sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  8 * a^3 + 27 * b^3 + 125 * c^3 + 1 / (a * b * c) = 10 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l814_81424


namespace NUMINAMATH_CALUDE_cubic_root_sum_l814_81412

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 9*p - 3 = 0 →
  q^3 - 8*q^2 + 9*q - 3 = 0 →
  r^3 - 8*r^2 + 9*r - 3 = 0 →
  p/(q*r + 1) + q/(p*r + 1) + r/(p*q + 1) = 83/43 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l814_81412


namespace NUMINAMATH_CALUDE_total_snacks_weight_l814_81444

theorem total_snacks_weight (peanuts_weight raisins_weight : ℝ) 
  (h1 : peanuts_weight = 0.1)
  (h2 : raisins_weight = 0.4) :
  peanuts_weight + raisins_weight = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_total_snacks_weight_l814_81444


namespace NUMINAMATH_CALUDE_sum_of_decimals_l814_81466

theorem sum_of_decimals : (5.47 + 4.96 : ℝ) = 10.43 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l814_81466


namespace NUMINAMATH_CALUDE_natasha_maria_earnings_l814_81487

theorem natasha_maria_earnings (t : ℚ) : 
  (t - 4) * (3 * t - 4) = (3 * t - 12) * (t + 2) → t = 20 / 11 := by
  sorry

end NUMINAMATH_CALUDE_natasha_maria_earnings_l814_81487


namespace NUMINAMATH_CALUDE_online_store_commission_l814_81416

/-- Calculates the commission percentage of an online store given the cost price,
    desired profit percentage, and final observed price. -/
theorem online_store_commission
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (observed_price : ℝ)
  (h1 : cost_price = 15)
  (h2 : profit_percentage = 0.1)
  (h3 : observed_price = 19.8) :
  let distributor_price := cost_price * (1 + profit_percentage)
  let commission_percentage := (observed_price / distributor_price - 1) * 100
  commission_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_online_store_commission_l814_81416


namespace NUMINAMATH_CALUDE_william_land_percentage_l814_81435

-- Define the total tax collected from the village
def total_tax : ℝ := 3840

-- Define Mr. William's tax payment
def william_tax : ℝ := 480

-- Define the percentage of cultivated land that is taxed
def tax_percentage : ℝ := 0.9

-- Theorem statement
theorem william_land_percentage :
  (william_tax / total_tax) * 100 = 12.5 := by
sorry

end NUMINAMATH_CALUDE_william_land_percentage_l814_81435


namespace NUMINAMATH_CALUDE_chemistry_mixture_volume_l814_81446

theorem chemistry_mixture_volume (V : ℝ) :
  (0.6 * V + 100) / (V + 100) = 0.7 →
  V = 300 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_mixture_volume_l814_81446


namespace NUMINAMATH_CALUDE_train_length_l814_81411

/-- Given a train that crosses a tree in 120 seconds and takes 200 seconds to pass
    a platform 800 m long, moving at a constant speed, prove that the length of the train
    is 1200 meters. -/
theorem train_length (tree_time platform_time platform_length : ℝ)
    (h1 : tree_time = 120)
    (h2 : platform_time = 200)
    (h3 : platform_length = 800) :
  let train_length := (platform_time * platform_length) / (platform_time - tree_time)
  train_length = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l814_81411


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l814_81476

/-- Given a mixture of pure water and salt solution, prove the amount of salt solution needed. -/
theorem salt_solution_mixture (x : ℝ) : 
  (0.30 * x = 0.20 * (x + 1)) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l814_81476


namespace NUMINAMATH_CALUDE_hyperbola_equation_l814_81430

/-- A hyperbola is defined by its equation and properties -/
structure Hyperbola where
  -- The equation of the hyperbola in the form (y²/a² - x²/b² = 1)
  a : ℝ
  b : ℝ
  -- The hyperbola passes through the point (2, -2)
  passes_through : a^2 * 4 - b^2 * 4 = a^2 * b^2
  -- The hyperbola has asymptotes y = ± (√2/2)x
  asymptotes : a / b = Real.sqrt 2 / 2

/-- The equation of the hyperbola is y²/2 - x²/4 = 1 -/
theorem hyperbola_equation (h : Hyperbola) : h.a^2 = 2 ∧ h.b^2 = 4 :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l814_81430


namespace NUMINAMATH_CALUDE_power_of_power_three_l814_81443

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l814_81443


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l814_81434

theorem consecutive_integers_sum (x : ℤ) : 
  x * (x + 1) * (x + 2) = 336 → x + (x + 1) + (x + 2) = 21 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l814_81434


namespace NUMINAMATH_CALUDE_inequalities_hold_l814_81494

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ((a + b) * (1/a + 1/b) ≥ 4) ∧ 
  (a^2 + b^2 + 2 ≥ 2*a + 2*b) ∧ 
  (Real.sqrt (abs (a - b)) ≥ Real.sqrt a - Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_hold_l814_81494


namespace NUMINAMATH_CALUDE_square_area_6cm_l814_81404

theorem square_area_6cm (side_length : ℝ) (h : side_length = 6) :
  side_length * side_length = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_6cm_l814_81404


namespace NUMINAMATH_CALUDE_abs_5x_minus_3_not_positive_l814_81475

theorem abs_5x_minus_3_not_positive (x : ℚ) : 
  ¬(|5*x - 3| > 0) ↔ x = 3/5 := by sorry

end NUMINAMATH_CALUDE_abs_5x_minus_3_not_positive_l814_81475


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l814_81478

theorem polynomial_divisibility (r : ℝ) :
  (∃ s : ℝ, 10 * X^3 - 5 * X^2 - 52 * X + 60 = 10 * (X - r)^2 * (X - s)) →
  r = -3/2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l814_81478


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_l814_81409

theorem consecutive_page_numbers (n : ℕ) : 
  n * (n + 1) * (n + 2) = 35280 → n + (n + 1) + (n + 2) = 96 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_l814_81409


namespace NUMINAMATH_CALUDE_given_scenario_is_combination_l814_81471

/-- Represents the type of course selection problem -/
inductive SelectionType
  | Permutation
  | Combination

/-- Represents a course selection scenario -/
structure CourseSelection where
  typeA : ℕ  -- Number of type A courses
  typeB : ℕ  -- Number of type B courses
  total : ℕ  -- Total number of courses to be selected
  atLeastOneEach : Bool  -- Whether at least one of each type is required

/-- Determines the type of selection problem based on the given scenario -/
def selectionProblemType (scenario : CourseSelection) : SelectionType :=
  sorry

/-- The specific scenario from the problem -/
def givenScenario : CourseSelection := {
  typeA := 3
  typeB := 4
  total := 3
  atLeastOneEach := true
}

/-- Theorem stating that the given scenario is a combination problem -/
theorem given_scenario_is_combination :
  selectionProblemType givenScenario = SelectionType.Combination := by
  sorry

end NUMINAMATH_CALUDE_given_scenario_is_combination_l814_81471


namespace NUMINAMATH_CALUDE_point_coordinates_l814_81493

theorem point_coordinates (x y : ℝ) : 
  (x < 0 ∧ y > 0) →  -- Point P is in the second quadrant
  (|x| = 2) →        -- |x| = 2
  (y^2 = 1) →        -- y is the square root of 1
  (x = -2 ∧ y = 1)   -- Coordinates of P are (-2, 1)
  := by sorry

end NUMINAMATH_CALUDE_point_coordinates_l814_81493


namespace NUMINAMATH_CALUDE_leahs_calculation_l814_81438

theorem leahs_calculation (y : ℕ) (h : (y + 4) * 5 = 140) : y * 5 + 4 = 124 := by
  sorry

end NUMINAMATH_CALUDE_leahs_calculation_l814_81438


namespace NUMINAMATH_CALUDE_total_spending_equals_49_l814_81450

/-- Represents the total amount spent by Paula and Olive at the kiddy gift shop -/
def total_spent (bracelet_price keychain_price coloring_book_price sticker_price toy_car_price : ℕ)
  (paula_bracelets paula_keychains paula_coloring_books paula_stickers : ℕ)
  (olive_coloring_books olive_bracelets olive_toy_cars olive_stickers : ℕ) : ℕ :=
  (bracelet_price * (paula_bracelets + olive_bracelets)) +
  (keychain_price * paula_keychains) +
  (coloring_book_price * (paula_coloring_books + olive_coloring_books)) +
  (sticker_price * (paula_stickers + olive_stickers)) +
  (toy_car_price * olive_toy_cars)

/-- Theorem stating that Paula and Olive's total spending equals $49 -/
theorem total_spending_equals_49 :
  total_spent 4 5 3 1 6 3 2 1 4 1 2 1 3 = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_spending_equals_49_l814_81450


namespace NUMINAMATH_CALUDE_solve_christmas_decorations_problem_l814_81447

def christmas_decorations_problem (decorations_per_box : ℕ) (decorations_used : ℕ) (decorations_given_away : ℕ) : Prop :=
  decorations_per_box = 15 ∧ 
  decorations_used = 35 ∧ 
  decorations_given_away = 25 →
  (decorations_used + decorations_given_away) / decorations_per_box = 4

theorem solve_christmas_decorations_problem :
  ∀ (decorations_per_box : ℕ) (decorations_used : ℕ) (decorations_given_away : ℕ),
  christmas_decorations_problem decorations_per_box decorations_used decorations_given_away :=
by
  sorry

end NUMINAMATH_CALUDE_solve_christmas_decorations_problem_l814_81447


namespace NUMINAMATH_CALUDE_gcd_bound_l814_81431

theorem gcd_bound (a b : ℕ) (h : ℕ) (h_int : (a + 1) / b + (b + 1) / a = h) :
  Nat.gcd a b ≤ Nat.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_gcd_bound_l814_81431


namespace NUMINAMATH_CALUDE_scientific_notation_87000000_l814_81484

theorem scientific_notation_87000000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 87000000 = a * (10 : ℝ) ^ n ∧ a = 8.7 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_87000000_l814_81484


namespace NUMINAMATH_CALUDE_not_all_axially_symmetric_figures_have_one_axis_l814_81400

/-- A type representing geometric figures -/
structure Figure where
  -- Add necessary fields here
  
/-- Predicate to check if a figure is axially symmetric -/
def is_axially_symmetric (f : Figure) : Prop :=
  sorry

/-- Function to count the number of axes of symmetry for a figure -/
def count_axes_of_symmetry (f : Figure) : ℕ :=
  sorry

/-- Theorem stating that not all axially symmetric figures have only one axis of symmetry -/
theorem not_all_axially_symmetric_figures_have_one_axis :
  ¬ (∀ f : Figure, is_axially_symmetric f → count_axes_of_symmetry f = 1) :=
sorry

end NUMINAMATH_CALUDE_not_all_axially_symmetric_figures_have_one_axis_l814_81400


namespace NUMINAMATH_CALUDE_abs_lt_sufficient_not_necessary_l814_81417

theorem abs_lt_sufficient_not_necessary (a b : ℝ) (ha : a > 0) :
  (∀ a b, (abs a < b) → (-a < b)) ∧
  (∃ a b, a > 0 ∧ (-a < b) ∧ (abs a ≥ b)) :=
sorry

end NUMINAMATH_CALUDE_abs_lt_sufficient_not_necessary_l814_81417


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l814_81482

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 5)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l814_81482


namespace NUMINAMATH_CALUDE_simplify_expression_l814_81426

theorem simplify_expression :
  4 * (12 / 9) * (36 / -45) = -12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l814_81426


namespace NUMINAMATH_CALUDE_expression_never_equals_negative_one_l814_81454

theorem expression_never_equals_negative_one (a y : ℝ) (ha : a ≠ 0) (hy1 : y ≠ -a) (hy2 : y ≠ 2*a) :
  (2*a^2 + y^2) / (a*y - y^2 - a^2) ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_never_equals_negative_one_l814_81454


namespace NUMINAMATH_CALUDE_deepak_age_l814_81498

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 2 / 5 →
  arun_age + 10 = 30 →
  deepak_age = 50 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l814_81498


namespace NUMINAMATH_CALUDE_max_correct_answers_l814_81492

theorem max_correct_answers (total_questions : ℕ) (correct_score : ℤ) (incorrect_score : ℤ) (total_score : ℤ) :
  total_questions = 25 →
  correct_score = 5 →
  incorrect_score = -2 →
  total_score = 60 →
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct_score * correct + incorrect_score * incorrect = total_score ∧
    correct ≤ 14 ∧
    ∀ c, c > 14 →
      ¬∃ (i u : ℕ), c + i + u = total_questions ∧
                    correct_score * c + incorrect_score * i = total_score :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l814_81492


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l814_81406

/-- Given a cube with surface area 54 square centimeters, its volume is 27 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ (side_length : ℝ),
  (6 * side_length^2 = 54) →
  side_length^3 = 27 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l814_81406


namespace NUMINAMATH_CALUDE_problem_statement_l814_81453

theorem problem_statement (a b : ℝ) (h : a + b - 3 = 0) :
  2*a^2 + 4*a*b + 2*b^2 - 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l814_81453


namespace NUMINAMATH_CALUDE_angle_triple_complement_l814_81408

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l814_81408


namespace NUMINAMATH_CALUDE_hillarys_descending_rate_l814_81472

/-- Proves that Hillary's descending rate is 1000 ft/hr given the conditions of the climbing problem -/
theorem hillarys_descending_rate 
  (base_camp_distance : ℝ) 
  (hillary_climbing_rate : ℝ) 
  (eddy_climbing_rate : ℝ) 
  (hillary_stop_distance : ℝ) 
  (start_time : ℝ) 
  (passing_time : ℝ) :
  base_camp_distance = 5000 →
  hillary_climbing_rate = 800 →
  eddy_climbing_rate = 500 →
  hillary_stop_distance = 1000 →
  start_time = 6 →
  passing_time = 12 →
  ∃ (hillary_descending_rate : ℝ), hillary_descending_rate = 1000 := by
  sorry

#check hillarys_descending_rate

end NUMINAMATH_CALUDE_hillarys_descending_rate_l814_81472


namespace NUMINAMATH_CALUDE_cone_height_equals_cube_volume_l814_81414

/-- The height of a circular cone with base radius 5 units and volume equal to that of a cube with edge length 5 units is 15/π units. -/
theorem cone_height_equals_cube_volume (h : ℝ) : h = 15 / π := by
  -- Define the edge length of the cube
  let cube_edge : ℝ := 5

  -- Define the base radius of the cone
  let cone_radius : ℝ := 5

  -- Define the volume of the cube
  let cube_volume : ℝ := cube_edge ^ 3

  -- Define the volume of the cone
  let cone_volume : ℝ := (1 / 3) * π * cone_radius ^ 2 * h

  -- Assume the volumes are equal
  have volumes_equal : cube_volume = cone_volume := by sorry

  sorry

end NUMINAMATH_CALUDE_cone_height_equals_cube_volume_l814_81414


namespace NUMINAMATH_CALUDE_exists_non_isosceles_with_isosceles_bisector_base_l814_81442

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define an angle bisector
def AngleBisector (T : Triangle) (vertex : ℕ) : ℝ × ℝ → ℝ × ℝ := sorry

-- Define the base of an angle bisector
def BaseBisector (T : Triangle) (vertex : ℕ) : ℝ × ℝ := sorry

-- Define isosceles property for a triangle
def IsIsosceles (T : Triangle) : Prop := sorry

-- Define the triangle formed by the bases of angle bisectors
def BisectorBaseTriangle (T : Triangle) : Triangle :=
  { A := BaseBisector T 0,
    B := BaseBisector T 1,
    C := BaseBisector T 2 }

theorem exists_non_isosceles_with_isosceles_bisector_base :
  ∃ T : Triangle,
    IsIsosceles (BisectorBaseTriangle T) ∧
    ¬IsIsosceles T :=
  sorry

end NUMINAMATH_CALUDE_exists_non_isosceles_with_isosceles_bisector_base_l814_81442


namespace NUMINAMATH_CALUDE_no_nines_in_product_l814_81407

def first_number : Nat := 123456789
def second_number : Nat := 999999999

theorem no_nines_in_product : 
  ∀ d : Nat, d ∈ (first_number * second_number).digits 10 → d ≠ 9 := by
  sorry

end NUMINAMATH_CALUDE_no_nines_in_product_l814_81407


namespace NUMINAMATH_CALUDE_starting_number_proof_l814_81445

theorem starting_number_proof : ∃ x : ℝ, ((x - 2 + 4) / 1) / 2 * 8 = 77 ∧ x = 17.25 := by
  sorry

end NUMINAMATH_CALUDE_starting_number_proof_l814_81445


namespace NUMINAMATH_CALUDE_triangle_property_l814_81474

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

end NUMINAMATH_CALUDE_triangle_property_l814_81474


namespace NUMINAMATH_CALUDE_green_balls_count_l814_81470

/-- Represents the contents and properties of a bag of colored balls. -/
structure BagOfBalls where
  total : Nat
  white : Nat
  yellow : Nat
  red : Nat
  purple : Nat
  prob_not_red_purple : Rat

/-- Calculates the number of green balls in the bag. -/
def green_balls (bag : BagOfBalls) : Nat :=
  bag.total - bag.white - bag.yellow - bag.red - bag.purple

/-- Theorem stating the number of green balls in the specific bag described in the problem. -/
theorem green_balls_count (bag : BagOfBalls) 
  (h1 : bag.total = 60)
  (h2 : bag.white = 22)
  (h3 : bag.yellow = 5)
  (h4 : bag.red = 6)
  (h5 : bag.purple = 9)
  (h6 : bag.prob_not_red_purple = 3/4) :
  green_balls bag = 18 := by
  sorry

#eval green_balls { 
  total := 60, 
  white := 22, 
  yellow := 5, 
  red := 6, 
  purple := 9, 
  prob_not_red_purple := 3/4 
}

end NUMINAMATH_CALUDE_green_balls_count_l814_81470


namespace NUMINAMATH_CALUDE_mode_is_180_l814_81421

/-- Represents the electricity consumption data for households -/
structure ElectricityData where
  consumption : List Nat
  frequency : List Nat
  total_households : Nat

/-- Calculates the mode of a list of numbers -/
def mode (data : ElectricityData) : Nat :=
  let paired_data := data.consumption.zip data.frequency
  let max_frequency := paired_data.map Prod.snd |>.maximum?
  match max_frequency with
  | none => 0  -- Default value if the list is empty
  | some max => 
      (paired_data.filter (fun p => p.2 = max)).map Prod.fst |>.head!

/-- The electricity consumption survey data -/
def survey_data : ElectricityData := {
  consumption := [120, 140, 160, 180, 200],
  frequency := [5, 5, 3, 6, 1],
  total_households := 20
}

theorem mode_is_180 : mode survey_data = 180 := by
  sorry

end NUMINAMATH_CALUDE_mode_is_180_l814_81421


namespace NUMINAMATH_CALUDE_cube_root_x_plus_3y_equals_3_l814_81433

theorem cube_root_x_plus_3y_equals_3 (x y : ℝ) 
  (h : y = Real.sqrt (3 - x) + Real.sqrt (x - 3) + 8) :
  (x + 3 * y) ^ (1/3 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_x_plus_3y_equals_3_l814_81433


namespace NUMINAMATH_CALUDE_certain_number_sum_l814_81422

theorem certain_number_sum (n : ℕ) : 
  (n % 423 = 0) → 
  (n / 423 = 423 - 421) → 
  (n + 421 = 1267) := by
sorry

end NUMINAMATH_CALUDE_certain_number_sum_l814_81422


namespace NUMINAMATH_CALUDE_weight_of_new_person_l814_81477

/-- Given a group of 8 people, if replacing a 65 kg person with a new person
    increases the average weight by 1.5 kg, then the weight of the new person is 77 kg. -/
theorem weight_of_new_person
  (initial_count : ℕ)
  (weight_replaced : ℝ)
  (avg_increase : ℝ)
  (h_count : initial_count = 8)
  (h_replaced : weight_replaced = 65)
  (h_increase : avg_increase = 1.5) :
  weight_replaced + initial_count * avg_increase = 77 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l814_81477


namespace NUMINAMATH_CALUDE_power_of_power_of_five_l814_81480

theorem power_of_power_of_five : (5^4)^2 = 390625 := by sorry

end NUMINAMATH_CALUDE_power_of_power_of_five_l814_81480


namespace NUMINAMATH_CALUDE_point_not_in_region_l814_81458

def plane_region (x y : ℝ) : Prop := 3 * x + 2 * y < 6

theorem point_not_in_region : ¬ plane_region 2 0 := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_region_l814_81458


namespace NUMINAMATH_CALUDE_divisibility_pairs_l814_81464

theorem divisibility_pairs : 
  ∀ m n : ℕ+, 
  (∃ k : ℤ, (2 * m.val : ℤ) * k = 3 * n.val - 2) ∧ 
  (∃ l : ℤ, (2 * n.val : ℤ) * l = 3 * m.val - 2) →
  ((m.val = 2 ∧ n.val = 2) ∨ (m.val = 10 ∧ n.val = 14) ∨ (m.val = 14 ∧ n.val = 10)) :=
by sorry

#check divisibility_pairs

end NUMINAMATH_CALUDE_divisibility_pairs_l814_81464


namespace NUMINAMATH_CALUDE_combination_problem_l814_81488

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem combination_problem (n : ℕ) : 
  factorial 98 / (factorial n * factorial (98 - n)) = 4753 ∧
  factorial (98 - n) = factorial 2 →
  n = 96 := by sorry

end NUMINAMATH_CALUDE_combination_problem_l814_81488


namespace NUMINAMATH_CALUDE_cos_2theta_minus_15_deg_l814_81459

theorem cos_2theta_minus_15_deg (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sin (θ + π / 12) = 4 / 5) : 
  Real.cos (2 * θ - π / 12) = 17 * Real.sqrt 2 / 50 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_minus_15_deg_l814_81459


namespace NUMINAMATH_CALUDE_evans_books_in_eight_years_l814_81479

def books_six_years_ago : ℕ := 500
def books_reduced : ℕ := 100
def books_given_away_fraction : ℚ := 1/2
def books_replaced_fraction : ℚ := 1/4
def books_increase_fraction : ℚ := 3/2
def books_gifted : ℕ := 30

theorem evans_books_in_eight_years :
  let current_books := books_six_years_ago - books_reduced
  let books_after_giving_away := (current_books : ℚ) * (1 - books_given_away_fraction)
  let books_after_replacing := books_after_giving_away + books_after_giving_away * books_replaced_fraction
  let final_books := books_after_replacing + books_after_replacing * books_increase_fraction + books_gifted
  final_books = 655 := by sorry

end NUMINAMATH_CALUDE_evans_books_in_eight_years_l814_81479


namespace NUMINAMATH_CALUDE_min_value_expression_l814_81403

theorem min_value_expression (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  b^2 / (a - 1) + a^2 / (b - 1) ≥ 8 ∧
  (b^2 / (a - 1) + a^2 / (b - 1) = 8 ↔ a = 2 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l814_81403


namespace NUMINAMATH_CALUDE_jump_rope_time_difference_l814_81423

/-- Given jump rope times for Cindy, Betsy, and Tina, prove that Tina can jump 6 minutes longer than Cindy -/
theorem jump_rope_time_difference (cindy betsy tina : ℕ) 
  (h1 : cindy = 12)
  (h2 : betsy = cindy / 2)
  (h3 : tina = 3 * betsy) :
  tina - cindy = 6 := by
  sorry

end NUMINAMATH_CALUDE_jump_rope_time_difference_l814_81423


namespace NUMINAMATH_CALUDE_min_t_value_l814_81495

/-- Ellipse C with eccentricity sqrt(2)/2 passing through (1, sqrt(2)/2) -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Point M -/
def M : ℝ × ℝ := (2, 0)

/-- Line containing point P -/
def line_P (x y : ℝ) : Prop := x + y = 1

/-- Vector relation between OA, OB, and OP -/
def vector_relation (A B P : ℝ × ℝ) (t : ℝ) : Prop :=
  A.1 + B.1 = t * P.1 ∧ A.2 + B.2 = t * P.2

/-- Main theorem: Minimum value of t -/
theorem min_t_value :
  ∃ (t_min : ℝ), 
    (∀ (A B P : ℝ × ℝ) (t : ℝ),
      ellipse_C A.1 A.2 → 
      ellipse_C B.1 B.2 → 
      line_P P.1 P.2 →
      vector_relation A B P t →
      t ≥ t_min) ∧
    t_min = 2 - Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_min_t_value_l814_81495


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l814_81410

theorem multiplication_addition_equality : 85 * 1500 + (1 / 2) * 1500 = 128250 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l814_81410


namespace NUMINAMATH_CALUDE_smallest_positive_solution_congruence_l814_81468

theorem smallest_positive_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (4 * x) % 27 = 13 % 27 ∧
  ∀ (y : ℕ), y > 0 ∧ (4 * y) % 27 = 13 % 27 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_congruence_l814_81468


namespace NUMINAMATH_CALUDE_triangle_area_l814_81413

/-- Given a triangle ABC where c² = (a-b)² + 6 and angle C = π/3, 
    prove that its area is 3√3/2 -/
theorem triangle_area (a b c : ℝ) (h1 : c^2 = (a-b)^2 + 6) (h2 : Real.pi / 3 = Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) : 
  (1/2) * a * b * Real.sin (Real.pi / 3) = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l814_81413


namespace NUMINAMATH_CALUDE_correct_average_l814_81432

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_readings : List (ℚ × ℚ)) : 
  n = 20 ∧ 
  initial_avg = 15 ∧ 
  incorrect_readings = [(42, 52), (68, 78), (85, 95)] →
  (n : ℚ) * initial_avg + (incorrect_readings.map (λ p => p.2 - p.1)).sum = n * (16.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l814_81432


namespace NUMINAMATH_CALUDE_supplement_quadruple_complement_30_l814_81429

/-- The degree measure of the supplement of the quadruple of the complement of a 30-degree angle is 120 degrees. -/
theorem supplement_quadruple_complement_30 : 
  let initial_angle : ℝ := 30
  let complement := 90 - initial_angle
  let quadruple := 4 * complement
  let supplement := if quadruple ≤ 180 then 180 - quadruple else 360 - quadruple
  supplement = 120 := by sorry

end NUMINAMATH_CALUDE_supplement_quadruple_complement_30_l814_81429


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l814_81456

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_planes 
  (α β : Plane) (l : Line) 
  (h1 : perpendicular l α) 
  (h2 : parallel α β) : 
  perpendicular l β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l814_81456


namespace NUMINAMATH_CALUDE_geometric_progression_iff_equal_first_two_l814_81457

/-- A sequence of positive real numbers -/
def Sequence := ℕ → ℝ

/-- Predicate to check if a sequence is positive -/
def IsPositive (a : Sequence) : Prop :=
  ∀ n, a n > 0

/-- Predicate to check if a sequence satisfies the given recurrence relation -/
def SatisfiesRecurrence (a : Sequence) (b : ℝ) : Prop :=
  ∀ n, a (n + 2) = (b + 1) * a n * a (n + 1)

/-- Predicate to check if a sequence is a geometric progression -/
def IsGeometricProgression (a : Sequence) : Prop :=
  ∃ r, ∀ n, a (n + 1) = r * a n

/-- Main theorem -/
theorem geometric_progression_iff_equal_first_two (a : Sequence) (b : ℝ) :
  b > 0 ∧ IsPositive a ∧ SatisfiesRecurrence a b →
  IsGeometricProgression a ↔ a 1 = a 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_iff_equal_first_two_l814_81457


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l814_81448

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- Define the derivative of f
def f_deriv (x : ℝ) : ℝ := 3*x^2 - 30*x - 33

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 11, f_deriv x < 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l814_81448


namespace NUMINAMATH_CALUDE_min_hindi_speakers_l814_81489

theorem min_hindi_speakers (total : ℕ) (english : ℕ) (both : ℕ) (hindi : ℕ) : 
  total = 40 → 
  english = 20 → 
  both ≥ 10 → 
  hindi = total + both - english →
  hindi ≥ 30 :=
by sorry

end NUMINAMATH_CALUDE_min_hindi_speakers_l814_81489


namespace NUMINAMATH_CALUDE_max_value_2sin_l814_81402

theorem max_value_2sin (f : ℝ → ℝ) (h : f = λ x => 2 * Real.sin x) :
  ∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, f x ≤ M := by
sorry

end NUMINAMATH_CALUDE_max_value_2sin_l814_81402


namespace NUMINAMATH_CALUDE_square_diagonal_triangle_l814_81481

theorem square_diagonal_triangle (s : ℝ) (h : s = 12) :
  let square_side := s
  let triangle_leg := s
  let triangle_hypotenuse := s * Real.sqrt 2
  let triangle_area := (s^2) / 2
  (triangle_leg = 12 ∧ 
   triangle_hypotenuse = 12 * Real.sqrt 2 ∧ 
   triangle_area = 72) :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_triangle_l814_81481


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l814_81491

theorem no_positive_integer_solution :
  ¬∃ (x y : ℕ+), x^2017 - 1 = (x - 1) * (y^2015 - 1) := by
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l814_81491


namespace NUMINAMATH_CALUDE_tangent_line_at_negative_one_l814_81460

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2

-- Define the point of tangency
def x₀ : ℝ := -1

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 3*x - y + 3 = 0

-- Theorem statement
theorem tangent_line_at_negative_one :
  tangent_line x₀ (f x₀) ∧
  ∀ x : ℝ, tangent_line x (f x₀ + f' x₀ * (x - x₀)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_negative_one_l814_81460


namespace NUMINAMATH_CALUDE_initial_display_cheesecakes_initial_display_cheesecakes_proof_l814_81465

/-- Proves that the initial number of cheesecakes on display is 10 -/
theorem initial_display_cheesecakes : ℕ → ℕ → ℕ → ℕ :=
  fun (fridge_cakes : ℕ) (sold_cakes : ℕ) (total_left : ℕ) =>
    if fridge_cakes = 15 ∧ sold_cakes = 7 ∧ total_left = 18 then
      10
    else
      0

#check initial_display_cheesecakes
-- The proof is omitted
theorem initial_display_cheesecakes_proof :
  initial_display_cheesecakes 15 7 18 = 10 := by sorry

end NUMINAMATH_CALUDE_initial_display_cheesecakes_initial_display_cheesecakes_proof_l814_81465


namespace NUMINAMATH_CALUDE_calculation_proof_l814_81425

theorem calculation_proof : ((-2)^2 : ℝ) + Real.sqrt 16 - 2 * Real.sin (π / 6) + (2023 - Real.pi)^0 = 8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l814_81425


namespace NUMINAMATH_CALUDE_constant_term_expansion_l814_81428

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function for the expansion
def expansion_term (x : ℝ) (r : ℕ) : ℝ :=
  (-1)^r * binomial 16 r * x^(16 - 4*r/3)

-- State the theorem
theorem constant_term_expansion :
  expansion_term 1 12 = 1820 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l814_81428


namespace NUMINAMATH_CALUDE_inequality_proof_l814_81436

theorem inequality_proof (a b c : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l814_81436


namespace NUMINAMATH_CALUDE_fresh_grape_weight_l814_81401

/-- Theorem: Weight of fresh grapes given dried grape weight and water content -/
theorem fresh_grape_weight
  (fresh_water_content : ℝ)
  (dried_water_content : ℝ)
  (dried_grape_weight : ℝ)
  (h1 : fresh_water_content = 0.7)
  (h2 : dried_water_content = 0.1)
  (h3 : dried_grape_weight = 33.33333333333333)
  : ∃ (fresh_grape_weight : ℝ),
    fresh_grape_weight * (1 - fresh_water_content) =
    dried_grape_weight * (1 - dried_water_content) ∧
    fresh_grape_weight = 100 := by
  sorry

end NUMINAMATH_CALUDE_fresh_grape_weight_l814_81401


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l814_81455

theorem quadratic_completing_square : ∀ x : ℝ, x^2 - 4*x + 5 = (x - 2)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l814_81455
