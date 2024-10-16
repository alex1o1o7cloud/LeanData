import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1634_163408

/-- Represents the number of cities in a group -/
def num_cities : ℕ := 8

/-- Represents the probability of a city being selected -/
def selection_probability : ℚ := 1/4

/-- Represents the number of cities drawn from the group -/
def cities_drawn : ℚ := num_cities * selection_probability

theorem stratified_sampling_theorem :
  cities_drawn = 2 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1634_163408


namespace NUMINAMATH_CALUDE_baba_yaga_powder_division_l1634_163403

/-- Represents the weight measurement system with a possible consistent error --/
structure ScaleSystem where
  total_shown : ℤ
  part1_shown : ℤ
  part2_shown : ℤ
  error : ℤ

/-- The actual weights of the two parts of the powder --/
def actual_weights (s : ScaleSystem) : ℤ × ℤ :=
  (s.part1_shown - s.error, s.part2_shown - s.error)

/-- Theorem stating the correct weights given the scale measurements --/
theorem baba_yaga_powder_division (s : ScaleSystem) 
  (h1 : s.total_shown = 6)
  (h2 : s.part1_shown = 3)
  (h3 : s.part2_shown = 2)
  (h4 : s.total_shown = s.part1_shown + s.part2_shown - s.error) :
  actual_weights s = (4, 3) := by
  sorry


end NUMINAMATH_CALUDE_baba_yaga_powder_division_l1634_163403


namespace NUMINAMATH_CALUDE_allison_extra_glue_sticks_l1634_163428

/-- Represents the number of items bought by a person -/
structure Items where
  glue_sticks : ℕ
  construction_paper : ℕ

/-- The problem setup -/
def craft_store_problem (allison marie : Items) : Prop :=
  allison.glue_sticks > marie.glue_sticks ∧
  marie.construction_paper = 6 * allison.construction_paper ∧
  marie.glue_sticks = 15 ∧
  marie.construction_paper = 30 ∧
  allison.glue_sticks + allison.construction_paper = 28

/-- The theorem to prove -/
theorem allison_extra_glue_sticks (allison marie : Items) 
  (h : craft_store_problem allison marie) : 
  allison.glue_sticks - marie.glue_sticks = 8 := by
  sorry


end NUMINAMATH_CALUDE_allison_extra_glue_sticks_l1634_163428


namespace NUMINAMATH_CALUDE_alices_number_l1634_163409

theorem alices_number (n : ℕ) : 
  180 ∣ n → 75 ∣ n → 900 ≤ n → n < 3000 → n = 900 ∨ n = 1800 ∨ n = 2700 := by
  sorry

end NUMINAMATH_CALUDE_alices_number_l1634_163409


namespace NUMINAMATH_CALUDE_lacson_unsold_sweet_potatoes_l1634_163438

/-- The number of sweet potatoes Mrs. Lacson has not yet sold -/
def sweet_potatoes_not_sold (total : ℕ) (sold_to_adams : ℕ) (sold_to_lenon : ℕ) : ℕ :=
  total - (sold_to_adams + sold_to_lenon)

/-- Theorem stating that Mrs. Lacson has 45 sweet potatoes not yet sold -/
theorem lacson_unsold_sweet_potatoes : 
  sweet_potatoes_not_sold 80 20 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_lacson_unsold_sweet_potatoes_l1634_163438


namespace NUMINAMATH_CALUDE_side_c_values_simplify_expression_l1634_163414

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the specific triangle with given conditions
def SpecificTriangle : Triangle → Prop
  | t => t.a = 4 ∧ t.b = 6 ∧ t.a + t.b + t.c < 18 ∧ Even (t.a + t.b + t.c)

-- Theorem 1: If the perimeter is less than 18 and even, then c = 4 or c = 6
theorem side_c_values (t : Triangle) (h : SpecificTriangle t) :
  t.c = 4 ∨ t.c = 6 := by
  sorry

-- Theorem 2: Simplification of |a+b-c|+|c-a-b|
theorem simplify_expression (t : Triangle) :
  |t.a + t.b - t.c| + |t.c - t.a - t.b| = 2*t.a + 2*t.b - 2*t.c := by
  sorry

end NUMINAMATH_CALUDE_side_c_values_simplify_expression_l1634_163414


namespace NUMINAMATH_CALUDE_min_value_theorem_l1634_163489

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y + 2 * x + 3 * y = 42) :
  x * y + 5 * x + 4 * y ≥ 55 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ + 2 * x₀ + 3 * y₀ = 42 ∧ x₀ * y₀ + 5 * x₀ + 4 * y₀ = 55 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1634_163489


namespace NUMINAMATH_CALUDE_mary_shirts_left_l1634_163452

def blue_shirts : ℕ := 30
def brown_shirts : ℕ := 40
def red_shirts : ℕ := 20
def yellow_shirts : ℕ := 25

def blue_fraction : ℚ := 3/5
def brown_fraction : ℚ := 1/4
def red_fraction : ℚ := 2/3
def yellow_fraction : ℚ := 1/5

def shirts_left : ℕ := 69

theorem mary_shirts_left : 
  blue_shirts - Int.floor (blue_fraction * blue_shirts) +
  brown_shirts - Int.floor (brown_fraction * brown_shirts) +
  red_shirts - Int.floor (red_fraction * red_shirts) +
  yellow_shirts - Int.floor (yellow_fraction * yellow_shirts) = shirts_left := by
  sorry

end NUMINAMATH_CALUDE_mary_shirts_left_l1634_163452


namespace NUMINAMATH_CALUDE_leading_coefficient_is_negative_seven_l1634_163432

def polynomial (x : ℝ) : ℝ := -3 * (x^4 - 2*x^3 + 3*x) + 8 * (x^4 + 5) - 4 * (3*x^4 + x^3 + 1)

theorem leading_coefficient_is_negative_seven :
  ∃ (f : ℝ → ℝ) (a : ℝ), a ≠ 0 ∧ (∀ x, polynomial x = a * x^4 + f x) ∧ a = -7 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_is_negative_seven_l1634_163432


namespace NUMINAMATH_CALUDE_triangle_problem_l1634_163458

theorem triangle_problem (A B C : Real) (a b c : Real) :
  B = 2 * C →
  c = 2 →
  a = 1 →
  b = Real.sqrt 6 ∧
  Real.sin (2 * B - π / 3) = (7 * Real.sqrt 3 - Real.sqrt 15) / 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1634_163458


namespace NUMINAMATH_CALUDE_min_trucks_for_10_tons_l1634_163463

/-- Represents the minimum number of trucks needed to transport a given weight of boxes -/
def min_trucks (total_weight : ℝ) (box_max_weight : ℝ) (truck_capacity : ℝ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of 3-ton trucks needed to transport 10 tons of boxes -/
theorem min_trucks_for_10_tons :
  min_trucks 10 1 3 = 5 := by sorry

end NUMINAMATH_CALUDE_min_trucks_for_10_tons_l1634_163463


namespace NUMINAMATH_CALUDE_special_sequence_property_l1634_163483

/-- A sequence of natural numbers with specific properties -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ k, a (k + 1) - a k ∈ ({0, 1} : Set ℕ)

theorem special_sequence_property (a : ℕ → ℕ) (m : ℕ) :
  SpecialSequence a →
  (∃ m, a m = m / 1000) →
  ∃ n, a n = n / 500 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_property_l1634_163483


namespace NUMINAMATH_CALUDE_max_triangle_area_l1634_163496

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | (p.1^2 + (p.2 - 2)^2 = 4)}

-- Define the line l
def l : Set (ℝ × ℝ) := {p | p.1 - Real.sqrt 3 * p.2 + Real.sqrt 3 = 0}

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define a function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_triangle_area :
  ∀ P ∈ C, P ≠ A → P ≠ B →
  triangleArea P A B ≤ (4 * Real.sqrt 13 + Real.sqrt 39) / 4 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l1634_163496


namespace NUMINAMATH_CALUDE_tom_bought_three_decks_l1634_163431

/-- The number of decks Tom bought -/
def tom_decks : ℕ := 3

/-- The cost of each deck in dollars -/
def deck_cost : ℕ := 8

/-- The number of decks Tom's friend bought -/
def friend_decks : ℕ := 5

/-- The total amount spent in dollars -/
def total_spent : ℕ := 64

/-- Theorem stating that Tom bought 3 decks given the conditions -/
theorem tom_bought_three_decks : 
  deck_cost * (tom_decks + friend_decks) = total_spent := by
  sorry

end NUMINAMATH_CALUDE_tom_bought_three_decks_l1634_163431


namespace NUMINAMATH_CALUDE_circle_condition_l1634_163480

/-- The equation x^2 + y^2 + 4x - 2y + 5m = 0 represents a circle if and only if m < 1 -/
theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4*x - 2*y + 5*m = 0 ∧ 
   ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 + 4*x - 2*y + 5*m = 0) 
  ↔ m < 1 := by
sorry


end NUMINAMATH_CALUDE_circle_condition_l1634_163480


namespace NUMINAMATH_CALUDE_sets_intersection_empty_l1634_163469

-- Define the sets A, B, and C
def A : Set (ℝ × ℝ) := {p | p.2^2 - p.1 - 1 = 0}
def B : Set (ℝ × ℝ) := {p | 4*p.1^2 + 2*p.1 - 2*p.2 + 5 = 0}
def C (k b : ℕ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 + b}

-- State the theorem
theorem sets_intersection_empty :
  ∃! k b : ℕ, (A ∪ B) ∩ C k b = ∅ ∧ k = 1 ∧ b = 2 := by sorry

end NUMINAMATH_CALUDE_sets_intersection_empty_l1634_163469


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_range_l1634_163449

theorem right_triangle_leg_sum_range (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  x^2 + y^2 = 5 → Real.sqrt 5 < x + y ∧ x + y ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_range_l1634_163449


namespace NUMINAMATH_CALUDE_rational_equation_solution_l1634_163441

theorem rational_equation_solution (C D : ℝ) :
  (∀ x : ℝ, x ≠ 4 → C / (x - 4) + D * (x + 2) = (-2 * x^3 + 8 * x^2 + 35 * x + 48) / (x - 4)) →
  C + D = 174 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l1634_163441


namespace NUMINAMATH_CALUDE_washer_dryer_price_ratio_l1634_163407

theorem washer_dryer_price_ratio :
  ∀ (washer_price dryer_price : ℕ),
    washer_price + dryer_price = 600 →
    ∃ k : ℕ, washer_price = k * dryer_price →
    dryer_price = 150 →
    washer_price / dryer_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_washer_dryer_price_ratio_l1634_163407


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_9_and_22_l1634_163424

theorem six_digit_divisible_by_9_and_22 : ∃! n : ℕ, 
  220140 ≤ n ∧ n < 220150 ∧ 
  n % 9 = 0 ∧ 
  n % 22 = 0 ∧
  n = 520146 := by
sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_9_and_22_l1634_163424


namespace NUMINAMATH_CALUDE_function_has_max_and_min_l1634_163484

/-- The function f(x) = x^3 - ax^2 + ax has both a maximum and a minimum value 
    if and only if a is in the range (-∞, 0) ∪ (3, +∞) -/
theorem function_has_max_and_min (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^3 - a*x^2 + a*x ≤ x₁^3 - a*x₁^2 + a*x₁) ∧
    (∀ x : ℝ, x^3 - a*x^2 + a*x ≥ x₂^3 - a*x₂^2 + a*x₂)) ↔ 
  (a < 0 ∨ a > 3) := by
  sorry

#check function_has_max_and_min

end NUMINAMATH_CALUDE_function_has_max_and_min_l1634_163484


namespace NUMINAMATH_CALUDE_two_common_tangents_l1634_163419

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 9 = 0

-- Define the number of common tangent lines
def num_common_tangents : ℕ := 2

-- Theorem statement
theorem two_common_tangents :
  num_common_tangents = 2 :=
sorry

end NUMINAMATH_CALUDE_two_common_tangents_l1634_163419


namespace NUMINAMATH_CALUDE_book_distribution_l1634_163401

theorem book_distribution (n : Nat) (k : Nat) : 
  n = 5 → k = 4 → (k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n) = 292 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_l1634_163401


namespace NUMINAMATH_CALUDE_female_employees_at_least_60_l1634_163486

/-- Represents the number of employees in different categories -/
structure EmployeeCount where
  total : Nat
  advancedDegree : Nat
  collegeDegreeOnly : Nat
  maleCollegeDegreeOnly : Nat

/-- Theorem stating that the number of female employees is at least 60 -/
theorem female_employees_at_least_60 (e : EmployeeCount)
  (h1 : e.total = 200)
  (h2 : e.advancedDegree = 100)
  (h3 : e.collegeDegreeOnly = 100)
  (h4 : e.maleCollegeDegreeOnly = 40) :
  ∃ (femaleCount : Nat), femaleCount ≥ 60 ∧ femaleCount ≤ e.total :=
by sorry

end NUMINAMATH_CALUDE_female_employees_at_least_60_l1634_163486


namespace NUMINAMATH_CALUDE_max_trig_ratio_max_trig_ratio_equals_one_l1634_163422

theorem max_trig_ratio (x : ℝ) : 
  (Real.sin x)^4 + (Real.cos x)^4 + 2 ≤ (Real.sin x)^2 + (Real.cos x)^2 + 2 := by
  sorry

theorem max_trig_ratio_equals_one :
  ∃ x : ℝ, (Real.sin x)^4 + (Real.cos x)^4 + 2 = (Real.sin x)^2 + (Real.cos x)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_max_trig_ratio_max_trig_ratio_equals_one_l1634_163422


namespace NUMINAMATH_CALUDE_range_of_a_l1634_163427

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + 2*x + a ≥ 0) ↔ a ∈ Set.Ici (-8) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1634_163427


namespace NUMINAMATH_CALUDE_delta_problem_l1634_163474

-- Define the Δ operation
def delta (a b : ℕ) : ℕ := a^2 + b

-- State the theorem
theorem delta_problem : delta (3^(delta 2 6)) (4^(delta 4 2)) = 72201960037 := by
  sorry

end NUMINAMATH_CALUDE_delta_problem_l1634_163474


namespace NUMINAMATH_CALUDE_range_of_a_l1634_163443

/-- The function f(x) = x³ + x + 1 -/
def f (x : ℝ) : ℝ := x^3 + x + 1

/-- Theorem stating that if f(x² + a) + f(ax) > 2 for all x, then 0 < a < 4 -/
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f (x^2 + a) + f (a*x) > 2) → 0 < a ∧ a < 4 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l1634_163443


namespace NUMINAMATH_CALUDE_cube_skeleton_theorem_l1634_163413

/-- The number of small cubes forming the skeleton of an n x n x n cube -/
def skeleton_cubes (n : ℕ) : ℕ := 12 * n - 16

/-- The number of small cubes to be removed to obtain the skeleton of an n x n x n cube -/
def removed_cubes (n : ℕ) : ℕ := n^3 - skeleton_cubes n

theorem cube_skeleton_theorem (n : ℕ) (h : n > 2) :
  skeleton_cubes n = 12 * n - 16 ∧
  removed_cubes n = n^3 - (12 * n - 16) := by
  sorry

#eval skeleton_cubes 6  -- Expected: 56
#eval removed_cubes 7   -- Expected: 275

end NUMINAMATH_CALUDE_cube_skeleton_theorem_l1634_163413


namespace NUMINAMATH_CALUDE_total_hockey_games_l1634_163475

theorem total_hockey_games (attended : ℕ) (missed : ℕ) 
  (h1 : attended = 13) (h2 : missed = 18) : 
  attended + missed = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_hockey_games_l1634_163475


namespace NUMINAMATH_CALUDE_parabola_equation_l1634_163494

/-- A parabola with focus on the x-axis, vertex at the origin, and opening to the right -/
structure RightParabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h1 : ∀ x y, eq x y ↔ y^2 = 2*p*x
  h2 : p > 0

/-- The point (1, 2) lies on the parabola -/
def PassesThroughPoint (par : RightParabola) : Prop :=
  par.eq 1 2

theorem parabola_equation (par : RightParabola) (h : PassesThroughPoint par) :
  par.p = 2 ∧ ∀ x y, par.eq x y ↔ y^2 = 4*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1634_163494


namespace NUMINAMATH_CALUDE_pr_length_l1634_163459

-- Define the triangles and their side lengths
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

def PQR : Triangle := { side1 := 30, side2 := 18, side3 := 22.5 }
def STU : Triangle := { side1 := 24, side2 := 18, side3 := 18 }

-- Define similarity of triangles
def similar (t1 t2 : Triangle) : Prop :=
  t1.side1 / t2.side1 = t1.side2 / t2.side2 ∧
  t1.side1 / t2.side1 = t1.side3 / t2.side3

-- Theorem statement
theorem pr_length :
  similar PQR STU → PQR.side3 = 22.5 :=
by
  sorry

end NUMINAMATH_CALUDE_pr_length_l1634_163459


namespace NUMINAMATH_CALUDE_fruit_basket_count_l1634_163465

/-- The number of ways to choose items from a set of n identical items -/
def chooseOptions (n : ℕ) : ℕ := n + 1

/-- The number of different fruit baskets that can be created -/
def fruitBaskets (apples oranges : ℕ) : ℕ :=
  chooseOptions apples * chooseOptions oranges - 1

theorem fruit_basket_count :
  fruitBaskets 6 8 = 62 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l1634_163465


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1634_163430

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : a > 0) (k : b > 0) :
  (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b) →
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = 1 / (a + b)^5 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1634_163430


namespace NUMINAMATH_CALUDE_halloween_candy_l1634_163411

theorem halloween_candy (debby_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : 
  debby_candy = 32 → sister_candy = 42 → eaten_candy = 35 →
  debby_candy + sister_candy - eaten_candy = 39 := by
sorry

end NUMINAMATH_CALUDE_halloween_candy_l1634_163411


namespace NUMINAMATH_CALUDE_evaluate_expression_l1634_163473

theorem evaluate_expression : 3000 * (3000^1500) = 3000^1501 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1634_163473


namespace NUMINAMATH_CALUDE_hyperbola_asymptotic_lines_l1634_163448

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := 3 * x^2 - y^2 = 3

/-- The asymptotic lines equation -/
def asymptotic_lines_eq (x y : ℝ) : Prop := y^2 = 3 * x^2

/-- Theorem: The asymptotic lines of the hyperbola 3x^2 - y^2 = 3 are y = ± √3x -/
theorem hyperbola_asymptotic_lines :
  ∀ x y : ℝ, hyperbola_eq x y → asymptotic_lines_eq x y :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotic_lines_l1634_163448


namespace NUMINAMATH_CALUDE_actual_average_height_after_correction_actual_average_height_is_184_cm_l1634_163405

/-- The actual average height of boys in a class after correcting measurement errors -/
theorem actual_average_height_after_correction (num_boys : ℕ) 
  (initial_avg : ℝ) (wrong_heights : Fin 4 → ℝ) (correct_heights : Fin 4 → ℝ) : ℝ :=
  let inch_to_cm : ℝ := 2.54
  let total_initial_height : ℝ := num_boys * initial_avg
  let height_difference : ℝ := (wrong_heights 0 - correct_heights 0) + 
                                (wrong_heights 1 - correct_heights 1) + 
                                (wrong_heights 2 - correct_heights 2) + 
                                (wrong_heights 3 * inch_to_cm - correct_heights 3 * inch_to_cm)
  let corrected_total_height : ℝ := total_initial_height - height_difference
  let actual_avg : ℝ := corrected_total_height / num_boys
  actual_avg

/-- The actual average height of boys in the class is 184.00 cm (rounded to two decimal places) -/
theorem actual_average_height_is_184_cm : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
  |actual_average_height_after_correction 75 185 
    (λ i => [170, 195, 160, 70][i]) 
    (λ i => [140, 165, 190, 64][i]) - 184| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_actual_average_height_after_correction_actual_average_height_is_184_cm_l1634_163405


namespace NUMINAMATH_CALUDE_almas_test_score_l1634_163481

/-- Proves that Alma's test score is 45 given the specified conditions. -/
theorem almas_test_score (alma_age melina_age carlos_age alma_score carlos_score : ℕ) : 
  alma_age + melina_age + carlos_age = 3 * alma_score →
  melina_age = 3 * alma_age →
  carlos_age = 4 * alma_age →
  melina_age = 60 →
  carlos_score = 2 * alma_score + 15 →
  carlos_score - alma_score = melina_age →
  alma_score = 45 := by
  sorry

#check almas_test_score

end NUMINAMATH_CALUDE_almas_test_score_l1634_163481


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_length_l1634_163499

/-- A circle passes through points A(1, 3), B(4, 2), and C(1, -7). 
    The segment MN is formed by the intersection of this circle with the y-axis. -/
theorem circle_y_axis_intersection_length :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    let circle := {(x, y) : ℝ × ℝ | (x - center.1)^2 + (y - center.2)^2 = radius^2}
    (1, 3) ∈ circle ∧ (4, 2) ∈ circle ∧ (1, -7) ∈ circle →
    let y_intersections := {y : ℝ | (0, y) ∈ circle}
    ∃ (m n : ℝ), m ∈ y_intersections ∧ n ∈ y_intersections ∧ m ≠ n ∧ 
    |m - n| = 4 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_length_l1634_163499


namespace NUMINAMATH_CALUDE_abs_two_implies_plus_minus_two_l1634_163426

theorem abs_two_implies_plus_minus_two (a : ℝ) : |a| = 2 → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_implies_plus_minus_two_l1634_163426


namespace NUMINAMATH_CALUDE_triangle_vector_equality_l1634_163490

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)

-- Define vectors m and n
def m (t : Triangle) : ℝ × ℝ := t.B - t.C
def n (t : Triangle) : ℝ × ℝ := t.D - t.C

-- State the theorem
theorem triangle_vector_equality (t : Triangle) 
  (h1 : t.D.1 = t.A.1 + (2/3) * (t.B.1 - t.A.1) ∧ t.D.2 = t.A.2 + (2/3) * (t.B.2 - t.A.2)) :
  t.A - t.C = -1/2 * (m t) + 3/2 * (n t) := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_equality_l1634_163490


namespace NUMINAMATH_CALUDE_online_store_prices_l1634_163468

/-- Represents the pricing structure for an online store --/
structure StorePricing where
  flatFee : ℝ
  commissionRate : ℝ

/-- Calculates the final price for a given store --/
def calculateFinalPrice (costPrice profit : ℝ) (store : StorePricing) : ℝ :=
  let sellingPrice := costPrice + profit
  sellingPrice + store.flatFee + store.commissionRate * sellingPrice

theorem online_store_prices (costPrice : ℝ) (profitRate : ℝ) 
    (storeA storeB storeC : StorePricing) : 
    costPrice = 18 ∧ 
    profitRate = 0.2 ∧
    storeA = { flatFee := 0, commissionRate := 0.2 } ∧
    storeB = { flatFee := 5, commissionRate := 0.1 } ∧
    storeC = { flatFee := 0, commissionRate := 0.15 } →
    let profit := profitRate * costPrice
    calculateFinalPrice costPrice profit storeA = 25.92 ∧
    calculateFinalPrice costPrice profit storeB = 28.76 ∧
    calculateFinalPrice costPrice profit storeC = 24.84 := by
  sorry

end NUMINAMATH_CALUDE_online_store_prices_l1634_163468


namespace NUMINAMATH_CALUDE_total_marbles_l1634_163412

/-- Given a collection of red, blue, and green marbles, where:
  1. There are 25% more red marbles than blue marbles
  2. There are 60% more green marbles than red marbles
  3. The number of red marbles is r
Prove that the total number of marbles in the collection is 3.4r -/
theorem total_marbles (r : ℝ) (b : ℝ) (g : ℝ) 
  (h1 : r = 1.25 * b) 
  (h2 : g = 1.6 * r) : 
  r + b + g = 3.4 * r := by
  sorry


end NUMINAMATH_CALUDE_total_marbles_l1634_163412


namespace NUMINAMATH_CALUDE_two_layer_triangle_structure_l1634_163446

/-- Calculates the number of small triangles in a layer given the number of triangles in the base row -/
def trianglesInLayer (baseTriangles : ℕ) : ℕ :=
  (baseTriangles * (baseTriangles + 1)) / 2

/-- Calculates the total number of toothpicks required for the two-layer structure -/
def totalToothpicks (lowerBaseTriangles upperBaseTriangles : ℕ) : ℕ :=
  let lowerTriangles := trianglesInLayer lowerBaseTriangles
  let upperTriangles := trianglesInLayer upperBaseTriangles
  let totalTriangles := lowerTriangles + upperTriangles
  let totalEdges := 3 * totalTriangles
  let boundaryEdges := 3 * lowerBaseTriangles + 3 * upperBaseTriangles - 3
  (totalEdges - boundaryEdges) / 2 + boundaryEdges

/-- The main theorem stating that the structure with 100 triangles in the lower base
    and 99 in the upper base requires 15596 toothpicks -/
theorem two_layer_triangle_structure :
  totalToothpicks 100 99 = 15596 := by
  sorry


end NUMINAMATH_CALUDE_two_layer_triangle_structure_l1634_163446


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l1634_163492

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = x^2 + (m-1)x - 3 -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + (m-1)*x - 3

theorem even_function_implies_m_equals_one :
  ∀ m : ℝ, IsEven (f m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l1634_163492


namespace NUMINAMATH_CALUDE_price_reduction_achieves_profit_l1634_163491

/-- Represents the store's sales and pricing data -/
structure StoreSales where
  initial_cost : ℝ
  initial_price : ℝ
  january_sales : ℝ
  march_sales : ℝ
  sales_increase_per_yuan : ℝ
  desired_profit : ℝ

/-- Calculates the required price reduction to achieve the desired profit -/
def calculate_price_reduction (s : StoreSales) : ℝ :=
  sorry

/-- Theorem stating that the calculated price reduction achieves the desired profit -/
theorem price_reduction_achieves_profit (s : StoreSales) 
  (h1 : s.initial_cost = 25)
  (h2 : s.initial_price = 40)
  (h3 : s.january_sales = 256)
  (h4 : s.march_sales = 400)
  (h5 : s.sales_increase_per_yuan = 5)
  (h6 : s.desired_profit = 4250) :
  let y := calculate_price_reduction s
  (s.initial_price - y - s.initial_cost) * (s.march_sales + s.sales_increase_per_yuan * y) = s.desired_profit :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_achieves_profit_l1634_163491


namespace NUMINAMATH_CALUDE_bob_dogs_count_l1634_163495

/-- Represents the number of cats Bob has -/
def num_cats : ℕ := 4

/-- Represents the portion of the food bag a single cat receives -/
def cat_portion : ℚ := 125 / 4000

/-- Represents the portion of the food bag a single dog receives -/
def dog_portion : ℚ := num_cats * cat_portion

/-- The number of dogs Bob has -/
def num_dogs : ℕ := 7

theorem bob_dogs_count :
  (num_dogs : ℚ) * dog_portion + (num_cats : ℚ) * cat_portion = 1 :=
sorry

#check bob_dogs_count

end NUMINAMATH_CALUDE_bob_dogs_count_l1634_163495


namespace NUMINAMATH_CALUDE_stamps_from_other_countries_l1634_163450

def total_stamps : ℕ := 500
def chinese_percent : ℚ := 40 / 100
def us_percent : ℚ := 25 / 100
def japanese_percent : ℚ := 15 / 100
def british_percent : ℚ := 10 / 100

theorem stamps_from_other_countries :
  total_stamps * (1 - (chinese_percent + us_percent + japanese_percent + british_percent)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_stamps_from_other_countries_l1634_163450


namespace NUMINAMATH_CALUDE_inequality_proof_l1634_163461

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1/a + 1/b + 1/c = a + b + c) :
  1/(2*a + b + c)^2 + 1/(2*b + c + a)^2 + 1/(2*c + a + b)^2 ≤ 3/16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1634_163461


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l1634_163400

/-- Proves that adding 3.6 liters of pure alcohol to a 6-liter solution
    that is 20% alcohol results in a solution that is 50% alcohol. -/
theorem alcohol_mixture_proof
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_alcohol : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.20)
  (h3 : added_alcohol = 3.6)
  (h4 : final_concentration = 0.50) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l1634_163400


namespace NUMINAMATH_CALUDE_floor_of_e_l1634_163404

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_e_l1634_163404


namespace NUMINAMATH_CALUDE_segment_ratio_l1634_163470

/-- Given points E, F, G, and H on a line in that order, with specified distances between them,
    prove that the ratio of EG to FH is 9:17. -/
theorem segment_ratio (E F G H : ℝ) : 
  F - E = 3 →
  G - F = 6 →
  H - G = 4 →
  H - E = 20 →
  (G - E) / (H - F) = 9 / 17 := by
  sorry

end NUMINAMATH_CALUDE_segment_ratio_l1634_163470


namespace NUMINAMATH_CALUDE_kitchen_renovation_time_percentage_l1634_163498

theorem kitchen_renovation_time_percentage (
  bedroom_count : Nat) 
  (bedroom_time : Nat) 
  (total_time : Nat) 
  (kitchen_time : Nat) : 
  bedroom_count = 3 → 
  bedroom_time = 4 → 
  total_time = 54 → 
  kitchen_time = 6 → 
  (kitchen_time - bedroom_time) / bedroom_time * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_renovation_time_percentage_l1634_163498


namespace NUMINAMATH_CALUDE_positive_real_inequality_l1634_163439

theorem positive_real_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 1) : 
  (x^2011 + y^2011) / (x^2009 + y^2009) + 
  (y^2011 + z^2011) / (y^2009 + z^2009) + 
  (z^2011 + x^2011) / (z^2009 + x^2009) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l1634_163439


namespace NUMINAMATH_CALUDE_three_chords_for_sixty_degrees_l1634_163425

/-- Represents a pair of concentric circles with chords drawn on the larger circle -/
structure ConcentricCirclesWithChords where
  /-- The measure of the angle formed by two adjacent chords at their intersection point -/
  chord_angle : ℝ
  /-- The number of chords needed to complete a full revolution -/
  num_chords : ℕ

/-- Theorem stating that for a 60° chord angle, 3 chords are needed to complete a revolution -/
theorem three_chords_for_sixty_degrees (circles : ConcentricCirclesWithChords) 
  (h : circles.chord_angle = 60) : circles.num_chords = 3 := by
  sorry

#check three_chords_for_sixty_degrees

end NUMINAMATH_CALUDE_three_chords_for_sixty_degrees_l1634_163425


namespace NUMINAMATH_CALUDE_chameleon_color_change_l1634_163456

theorem chameleon_color_change (total : ℕ) (blue_initial red_initial : ℕ) 
  (blue_final red_final : ℕ) (changed : ℕ) : 
  total = 140 →
  total = blue_initial + red_initial →
  total = blue_final + red_final →
  blue_initial = 5 * blue_final →
  red_final = 3 * red_initial →
  changed = blue_initial - blue_final →
  changed = 80 := by
sorry

end NUMINAMATH_CALUDE_chameleon_color_change_l1634_163456


namespace NUMINAMATH_CALUDE_marcus_pebbles_l1634_163457

theorem marcus_pebbles (P : ℕ) : 
  P / 2 + 30 = 39 → P = 18 := by
  sorry

end NUMINAMATH_CALUDE_marcus_pebbles_l1634_163457


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l1634_163429

theorem imaginary_part_of_complex_number :
  let z : ℂ := -1/2 + (1/2) * Complex.I
  Complex.im z = 1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l1634_163429


namespace NUMINAMATH_CALUDE_A_obtuse_sufficient_not_necessary_l1634_163485

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define an obtuse angle
def is_obtuse (angle : Real) : Prop := angle > 90

-- Define an obtuse triangle
def is_obtuse_triangle (t : Triangle) : Prop :=
  is_obtuse t.A ∨ is_obtuse t.B ∨ is_obtuse t.C

-- Theorem statement
theorem A_obtuse_sufficient_not_necessary (t : Triangle) :
  (is_obtuse t.A → is_obtuse_triangle t) ∧
  ∃ (t' : Triangle), is_obtuse_triangle t' ∧ ¬is_obtuse t'.A :=
sorry

end NUMINAMATH_CALUDE_A_obtuse_sufficient_not_necessary_l1634_163485


namespace NUMINAMATH_CALUDE_solution_in_third_quadrant_implies_k_bound_l1634_163472

theorem solution_in_third_quadrant_implies_k_bound 
  (k : ℝ) 
  (h : ∃ x : ℝ, 
    π < x ∧ x < 3*π/2 ∧ 
    k * Real.cos x + Real.arccos (π/4) = 0) : 
  k > Real.arccos (π/4) := by
sorry

end NUMINAMATH_CALUDE_solution_in_third_quadrant_implies_k_bound_l1634_163472


namespace NUMINAMATH_CALUDE_floor_of_7_8_l1634_163466

theorem floor_of_7_8 : ⌊(7.8 : ℝ)⌋ = 7 := by sorry

end NUMINAMATH_CALUDE_floor_of_7_8_l1634_163466


namespace NUMINAMATH_CALUDE_thomas_money_left_l1634_163487

/-- Calculates the money left over after selling books and buying records. -/
def money_left_over (num_books : ℕ) (book_price : ℚ) (num_records : ℕ) (record_price : ℚ) : ℚ :=
  num_books * book_price - num_records * record_price

/-- Proves that Thomas has $75 left over after selling his books and buying records. -/
theorem thomas_money_left : money_left_over 200 (3/2) 75 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_thomas_money_left_l1634_163487


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_is_7_l1634_163454

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The property that the sum of digits of a number is congruent to the number itself modulo 9 -/
axiom sum_of_digits_mod_9 (n : ℕ) : sumOfDigits n ≡ n [ZMOD 9]

/-- A is the sum of digits of 4444^444 -/
def A : ℕ := sumOfDigits (4444^444)

/-- B is the sum of digits of A -/
def B : ℕ := sumOfDigits A

/-- The main theorem: the sum of digits of B is 7 -/
theorem sum_of_digits_of_B_is_7 : sumOfDigits B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_is_7_l1634_163454


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1634_163460

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 4, -2; 0, 3, 1; 5, -1, 3]
  Matrix.det A = 70 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1634_163460


namespace NUMINAMATH_CALUDE_app_cost_proof_l1634_163440

theorem app_cost_proof (monthly_fee : ℕ) (months_played : ℕ) (total_spent : ℕ) (initial_cost : ℕ) :
  monthly_fee = 8 →
  months_played = 2 →
  total_spent = 21 →
  initial_cost + monthly_fee * months_played = total_spent →
  initial_cost = 5 := by
sorry

end NUMINAMATH_CALUDE_app_cost_proof_l1634_163440


namespace NUMINAMATH_CALUDE_min_value_theorem_l1634_163478

theorem min_value_theorem (x : ℝ) (h : x > 2) :
  (x + 4) / Real.sqrt (x - 2) ≥ 2 * Real.sqrt 6 ∧
  ∃ y : ℝ, y > 2 ∧ (y + 4) / Real.sqrt (y - 2) = 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1634_163478


namespace NUMINAMATH_CALUDE_area_to_paint_dining_room_l1634_163423

/-- The area to be painted on a wall with a painting hanging on it -/
def area_to_paint (wall_height wall_length painting_height painting_length : ℝ) : ℝ :=
  wall_height * wall_length - painting_height * painting_length

/-- Theorem: The area to be painted is 135 square feet -/
theorem area_to_paint_dining_room : 
  area_to_paint 10 15 3 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_area_to_paint_dining_room_l1634_163423


namespace NUMINAMATH_CALUDE_gold_coin_percentage_is_49_percent_l1634_163402

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  bead_percentage : ℝ
  silver_coin_percentage : ℝ

/-- Calculates the percentage of gold coins in the urn -/
def gold_coin_percentage (urn : UrnComposition) : ℝ :=
  (1 - urn.bead_percentage) * (1 - urn.silver_coin_percentage)

/-- Theorem stating that for the given urn composition, 
    the percentage of gold coins is 49% -/
theorem gold_coin_percentage_is_49_percent 
  (urn : UrnComposition) 
  (h1 : urn.bead_percentage = 0.3) 
  (h2 : urn.silver_coin_percentage = 0.3) : 
  gold_coin_percentage urn = 0.49 := by
  sorry

#eval gold_coin_percentage ⟨0.3, 0.3⟩

end NUMINAMATH_CALUDE_gold_coin_percentage_is_49_percent_l1634_163402


namespace NUMINAMATH_CALUDE_intersection_M_N_l1634_163410

def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1634_163410


namespace NUMINAMATH_CALUDE_remainder_of_1234567_divided_by_257_l1634_163420

theorem remainder_of_1234567_divided_by_257 : 
  1234567 % 257 = 774 := by sorry

end NUMINAMATH_CALUDE_remainder_of_1234567_divided_by_257_l1634_163420


namespace NUMINAMATH_CALUDE_chord_length_perpendicular_bisector_l1634_163437

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 10) : 
  let chord_length := 2 * (r^2 - (r/2)^2).sqrt
  chord_length = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_perpendicular_bisector_l1634_163437


namespace NUMINAMATH_CALUDE_base_10_to_9_conversion_l1634_163451

-- Define a custom type for base-9 digits
inductive Base9Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | H

def base9ToNat : List Base9Digit → Nat
  | [] => 0
  | d::ds => 
    match d with
    | Base9Digit.D0 => 0 + 9 * base9ToNat ds
    | Base9Digit.D1 => 1 + 9 * base9ToNat ds
    | Base9Digit.D2 => 2 + 9 * base9ToNat ds
    | Base9Digit.D3 => 3 + 9 * base9ToNat ds
    | Base9Digit.D4 => 4 + 9 * base9ToNat ds
    | Base9Digit.D5 => 5 + 9 * base9ToNat ds
    | Base9Digit.D6 => 6 + 9 * base9ToNat ds
    | Base9Digit.D7 => 7 + 9 * base9ToNat ds
    | Base9Digit.D8 => 8 + 9 * base9ToNat ds
    | Base9Digit.H => 8 + 9 * base9ToNat ds

theorem base_10_to_9_conversion :
  base9ToNat [Base9Digit.D3, Base9Digit.D1, Base9Digit.D4] = 256 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_9_conversion_l1634_163451


namespace NUMINAMATH_CALUDE_susannah_swims_24_times_l1634_163442

/-- The number of times Camden went swimming in March -/
def camden_swims : ℕ := 16

/-- The number of weeks in March -/
def weeks_in_march : ℕ := 4

/-- The number of times Camden swam per week -/
def camden_swims_per_week : ℕ := camden_swims / weeks_in_march

/-- The number of additional times Susannah swam per week compared to Camden -/
def susannah_additional_swims : ℕ := 2

/-- The number of times Susannah swam per week -/
def susannah_swims_per_week : ℕ := camden_swims_per_week + susannah_additional_swims

/-- The total number of times Susannah went swimming in March -/
def susannah_total_swims : ℕ := susannah_swims_per_week * weeks_in_march

theorem susannah_swims_24_times : susannah_total_swims = 24 := by
  sorry

end NUMINAMATH_CALUDE_susannah_swims_24_times_l1634_163442


namespace NUMINAMATH_CALUDE_ball_selection_probabilities_l1634_163416

/-- Represents a bag of balls -/
structure Bag where
  white : ℕ
  red : ℕ

/-- The probability of drawing a white ball from a bag -/
def probWhite (bag : Bag) : ℚ :=
  bag.white / (bag.white + bag.red)

/-- The probability of drawing a red ball from a bag -/
def probRed (bag : Bag) : ℚ :=
  bag.red / (bag.white + bag.red)

/-- The bags used in the problem -/
def bagA : Bag := ⟨8, 4⟩
def bagB : Bag := ⟨6, 6⟩

/-- The theorem to be proved -/
theorem ball_selection_probabilities :
  /- The probability of selecting two balls of the same color is 1/2 -/
  (probWhite bagA * probWhite bagB + probRed bagA * probRed bagB = 1/2) ∧
  /- The probability of selecting at least one red ball is 2/3 -/
  (1 - probWhite bagA * probWhite bagB = 2/3) := by
  sorry


end NUMINAMATH_CALUDE_ball_selection_probabilities_l1634_163416


namespace NUMINAMATH_CALUDE_find_number_from_announcements_l1634_163455

def circle_number_game (numbers : Fin 15 → ℝ) (announcements : Fin 15 → ℝ) : Prop :=
  ∀ i : Fin 15, announcements i = (numbers (i - 1) + numbers (i + 1)) / 2

theorem find_number_from_announcements 
  (numbers : Fin 15 → ℝ) (announcements : Fin 15 → ℝ)
  (h_circle : circle_number_game numbers announcements)
  (h_8th : announcements 7 = 10)
  (h_exists_5 : ∃ j : Fin 15, announcements j = 5) :
  ∃ k : Fin 15, announcements k = 5 ∧ numbers k = 0 := by
sorry

end NUMINAMATH_CALUDE_find_number_from_announcements_l1634_163455


namespace NUMINAMATH_CALUDE_factorization_left_to_right_l1634_163435

theorem factorization_left_to_right (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_left_to_right_l1634_163435


namespace NUMINAMATH_CALUDE_total_pets_is_54_l1634_163493

/-- The number of pets owned by Teddy, Ben, and Dave -/
def total_pets : ℕ :=
  let teddy_dogs : ℕ := 7
  let teddy_cats : ℕ := 8
  let ben_extra_dogs : ℕ := 9
  let dave_extra_cats : ℕ := 13
  let dave_fewer_dogs : ℕ := 5

  let teddy_total : ℕ := teddy_dogs + teddy_cats
  let ben_total : ℕ := (teddy_dogs + ben_extra_dogs)
  let dave_total : ℕ := (teddy_cats + dave_extra_cats) + (teddy_dogs - dave_fewer_dogs)

  teddy_total + ben_total + dave_total

/-- Theorem stating that the total number of pets is 54 -/
theorem total_pets_is_54 : total_pets = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_is_54_l1634_163493


namespace NUMINAMATH_CALUDE_gcd_204_85_l1634_163464

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l1634_163464


namespace NUMINAMATH_CALUDE_expenditure_representation_l1634_163444

def represent_income (amount : ℝ) : ℝ := amount

theorem expenditure_representation (amount : ℝ) :
  (represent_income amount = amount) →
  (∃ (f : ℝ → ℝ), f amount = -amount) :=
by sorry

end NUMINAMATH_CALUDE_expenditure_representation_l1634_163444


namespace NUMINAMATH_CALUDE_socks_selection_with_red_l1634_163415

def total_socks : ℕ := 10
def red_socks : ℕ := 1
def socks_to_choose : ℕ := 4

theorem socks_selection_with_red :
  (Nat.choose total_socks socks_to_choose) - 
  (Nat.choose (total_socks - red_socks) socks_to_choose) = 84 := by
  sorry

end NUMINAMATH_CALUDE_socks_selection_with_red_l1634_163415


namespace NUMINAMATH_CALUDE_mod_fifteen_equivalence_l1634_163479

theorem mod_fifteen_equivalence (n : ℤ) : 
  0 ≤ n ∧ n ≤ 14 ∧ n ≡ 15827 [ZMOD 15] → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_fifteen_equivalence_l1634_163479


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_games_l1634_163418

theorem tic_tac_toe_tie_games 
  (amy_wins : ℚ) 
  (lily_wins : ℚ) 
  (h1 : amy_wins = 5 / 12) 
  (h2 : lily_wins = 1 / 4) : 
  1 - (amy_wins + lily_wins) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_games_l1634_163418


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l1634_163406

/-- A right parallelepiped with a parallelogram base -/
structure RightParallelepiped where
  base_angle : ℝ
  base_area : ℝ
  lateral_face_area1 : ℝ
  lateral_face_area2 : ℝ

/-- The volume of the right parallelepiped -/
def volume (p : RightParallelepiped) : ℝ := sorry

theorem parallelepiped_volume (p : RightParallelepiped) 
  (h1 : p.base_angle = π / 6)
  (h2 : p.base_area = 4)
  (h3 : p.lateral_face_area1 = 6)
  (h4 : p.lateral_face_area2 = 12) :
  volume p = 12 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l1634_163406


namespace NUMINAMATH_CALUDE_convenience_store_distance_l1634_163471

def work_distance : ℝ := 6
def dog_walk_distance : ℝ := 2
def friend_house_distance : ℝ := 1
def total_weekly_distance : ℝ := 95
def work_days_per_week : ℕ := 5
def dog_walks_per_day : ℕ := 2
def days_per_week : ℕ := 7
def friend_visits_per_week : ℕ := 1
def convenience_store_visits_per_week : ℕ := 2

theorem convenience_store_distance :
  let work_total := work_distance * 2 * work_days_per_week
  let dog_walk_total := dog_walk_distance * dog_walks_per_day * days_per_week
  let friend_visit_total := friend_house_distance * 2 * friend_visits_per_week
  let other_activities_total := work_total + dog_walk_total + friend_visit_total
  let convenience_store_total := total_weekly_distance - other_activities_total
  convenience_store_total / convenience_store_visits_per_week = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_convenience_store_distance_l1634_163471


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1634_163467

theorem algebraic_expression_value (a b : ℝ) (h : a - 2*b = 2) : 2*a - 4*b + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1634_163467


namespace NUMINAMATH_CALUDE_x_squared_plus_nine_x_over_x_minus_three_squared_equals_90_l1634_163434

theorem x_squared_plus_nine_x_over_x_minus_three_squared_equals_90 (x : ℝ) :
  x^2 + 9 * (x / (x - 3))^2 = 90 →
  ((x - 3)^2 * (x + 4)) / (3 * x - 4) = 36 / 11 ∨
  ((x - 3)^2 * (x + 4)) / (3 * x - 4) = 468 / 23 :=
by sorry

end NUMINAMATH_CALUDE_x_squared_plus_nine_x_over_x_minus_three_squared_equals_90_l1634_163434


namespace NUMINAMATH_CALUDE_day_of_week_n_minus_one_l1634_163488

-- Define a type for days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to add days to a given day
def addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (addDays d n)

-- Define the theorem
theorem day_of_week_n_minus_one (n : Nat) :
  -- Given conditions
  (addDays DayOfWeek.Friday (150 % 7) = DayOfWeek.Friday) →
  (addDays DayOfWeek.Wednesday (210 % 7) = DayOfWeek.Wednesday) →
  -- Conclusion
  (addDays DayOfWeek.Monday 50 = DayOfWeek.Tuesday) :=
by
  sorry


end NUMINAMATH_CALUDE_day_of_week_n_minus_one_l1634_163488


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1634_163433

theorem inverse_variation_problem (y x : ℝ) (k : ℝ) (h1 : y * x^2 = k) 
  (h2 : 6 * 3^2 = k) (h3 : 2 * x^2 = k) : x = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1634_163433


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_5_8_2_l1634_163436

theorem largest_three_digit_divisible_by_5_8_2 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 5 = 0 ∧ n % 8 = 0 → n ≤ 960 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_5_8_2_l1634_163436


namespace NUMINAMATH_CALUDE_square_one_fifth_equals_point_zero_four_l1634_163421

theorem square_one_fifth_equals_point_zero_four (ε : ℝ) :
  ∃ ε > 0, (1 / 5 : ℝ)^2 = 0.04 + ε ∧ ε < 0.00000000000000001 := by
  sorry

end NUMINAMATH_CALUDE_square_one_fifth_equals_point_zero_four_l1634_163421


namespace NUMINAMATH_CALUDE_canoe_rental_cost_l1634_163462

/-- Represents the daily rental cost and quantities for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℝ
  kayak_cost : ℝ
  canoe_count : ℕ
  kayak_count : ℕ

/-- Calculates the total revenue from canoe and kayak rentals --/
def total_revenue (r : RentalInfo) : ℝ :=
  r.canoe_cost * r.canoe_count + r.kayak_cost * r.kayak_count

/-- Theorem stating the canoe rental cost given the problem conditions --/
theorem canoe_rental_cost :
  ∀ (r : RentalInfo),
    r.kayak_cost = 15 →
    r.canoe_count = (3 * r.kayak_count) / 2 →
    total_revenue r = 288 →
    r.canoe_count = r.kayak_count + 4 →
    r.canoe_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_canoe_rental_cost_l1634_163462


namespace NUMINAMATH_CALUDE_sophia_reading_progress_l1634_163445

theorem sophia_reading_progress (total_pages : ℕ) (pages_finished : ℚ) : 
  total_pages = 270 → pages_finished = 2/3 → 
  (pages_finished * total_pages : ℚ) - ((1 - pages_finished) * total_pages : ℚ) = 90 := by
  sorry


end NUMINAMATH_CALUDE_sophia_reading_progress_l1634_163445


namespace NUMINAMATH_CALUDE_sector_area_l1634_163497

/-- The area of a sector with perimeter 1 and central angle 1 radian is 1/18 -/
theorem sector_area (r : ℝ) (l : ℝ) (h1 : l + 2*r = 1) (h2 : l = r) : r^2/2 = 1/18 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1634_163497


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l1634_163477

/-- A rectangular prism with its properties -/
structure RectangularPrism where
  vertices : Nat
  edges : Nat
  dimensions : Nat
  has_face_diagonals : Bool
  has_space_diagonals : Bool

/-- The total number of diagonals in a rectangular prism -/
def total_diagonals (prism : RectangularPrism) : Nat :=
  sorry

/-- Theorem stating that the total number of diagonals in a rectangular prism is 16 -/
theorem rectangular_prism_diagonals :
  ∀ (prism : RectangularPrism),
    prism.vertices = 8 ∧
    prism.edges = 12 ∧
    prism.dimensions = 3 ∧
    prism.has_face_diagonals = true ∧
    prism.has_space_diagonals = true →
    total_diagonals prism = 16 :=
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l1634_163477


namespace NUMINAMATH_CALUDE_divisors_of_10n_l1634_163447

/-- Given a natural number n where 100n^2 has exactly 55 different natural divisors,
    prove that 10n has exactly 18 natural divisors. -/
theorem divisors_of_10n (n : ℕ) (h : (Nat.divisors (100 * n^2)).card = 55) :
  (Nat.divisors (10 * n)).card = 18 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_10n_l1634_163447


namespace NUMINAMATH_CALUDE_dice_prime_sum_probability_l1634_163482

/-- The number of dice being rolled -/
def num_dice : ℕ := 7

/-- The number of sides on each die -/
def die_sides : ℕ := 6

/-- The minimum possible sum when rolling the dice -/
def min_sum : ℕ := num_dice

/-- The maximum possible sum when rolling the dice -/
def max_sum : ℕ := num_dice * die_sides

/-- The set of prime numbers between the minimum and maximum possible sums -/
def relevant_primes : List ℕ := [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

/-- The total number of possible outcomes when rolling the dice -/
def total_outcomes : ℕ := die_sides ^ num_dice

/-- The probability of rolling a prime sum with the given dice -/
def prime_sum_probability : ℚ := 26875 / 93312

theorem dice_prime_sum_probability :
  prime_sum_probability = 26875 / 93312 := by sorry

end NUMINAMATH_CALUDE_dice_prime_sum_probability_l1634_163482


namespace NUMINAMATH_CALUDE_cadence_worked_five_months_longer_l1634_163476

/-- Calculates the number of months longer Cadence worked at her new company --/
def months_longer_at_new_company (
  old_salary : ℕ)
  (salary_increase_percent : ℕ)
  (old_company_months : ℕ)
  (total_earnings : ℕ) : ℕ :=
  let new_salary := old_salary + (old_salary * salary_increase_percent) / 100
  let x := (total_earnings - old_salary * old_company_months) / new_salary - old_company_months
  x

/-- Proves that Cadence worked 5 months longer at her new company --/
theorem cadence_worked_five_months_longer :
  months_longer_at_new_company 5000 20 36 426000 = 5 := by
  sorry

#eval months_longer_at_new_company 5000 20 36 426000

end NUMINAMATH_CALUDE_cadence_worked_five_months_longer_l1634_163476


namespace NUMINAMATH_CALUDE_waiter_tables_l1634_163453

theorem waiter_tables (people_per_table : ℕ) (total_customers : ℕ) (h1 : people_per_table = 9) (h2 : total_customers = 63) :
  total_customers / people_per_table = 7 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_l1634_163453


namespace NUMINAMATH_CALUDE_no_equal_partition_for_2002_l1634_163417

theorem no_equal_partition_for_2002 :
  ¬ ∃ (S : Finset ℕ),
    S ⊆ Finset.range 2003 ∧
    S.sum id = ((Finset.range 2003).sum id) / 2 :=
by sorry

end NUMINAMATH_CALUDE_no_equal_partition_for_2002_l1634_163417
