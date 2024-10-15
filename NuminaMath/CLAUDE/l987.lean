import Mathlib

namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l987_98719

/-- The number of ways to place n distinguishable objects into k distinguishable containers -/
def placement_count (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to place 5 distinguishable balls into 4 distinguishable boxes is 1024 -/
theorem five_balls_four_boxes : placement_count 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l987_98719


namespace NUMINAMATH_CALUDE_smallest_value_problem_l987_98778

theorem smallest_value_problem (m n x : ℕ) : 
  m > 0 → n > 0 → x > 0 → m = 77 →
  Nat.gcd m n = x + 7 →
  Nat.lcm m n = x * (x + 7) →
  ∃ (n_min : ℕ), n_min > 0 ∧ n_min ≤ n ∧ n_min = 22 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_problem_l987_98778


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l987_98708

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a 2D translation -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The original parabola y = x² -/
def original_parabola : Parabola :=
  { a := 1, b := 0, c := 0 }

/-- The translation of 1 unit left and 2 units down -/
def translation : Translation :=
  { dx := -1, dy := -2 }

/-- The resulting parabola after translation -/
def translated_parabola (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a
    b := -2 * p.a * t.dx
    c := p.a * t.dx^2 + p.b * t.dx + p.c + t.dy }

theorem parabola_translation_theorem :
  let p := original_parabola
  let t := translation
  let result := translated_parabola p t
  result.a = 1 ∧ result.b = 2 ∧ result.c = -2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l987_98708


namespace NUMINAMATH_CALUDE_e_squared_f_2_gt_e_cubed_f_3_l987_98711

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the condition that f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Define the given condition
axiom condition : ∀ x, f x + f' x < 0

-- State the theorem to be proved
theorem e_squared_f_2_gt_e_cubed_f_3 : e^2 * f 2 > e^3 * f 3 := by sorry

end NUMINAMATH_CALUDE_e_squared_f_2_gt_e_cubed_f_3_l987_98711


namespace NUMINAMATH_CALUDE_mary_fruits_left_l987_98786

/-- The number of fruits Mary has left after buying and eating some -/
def fruits_left (apples oranges blueberries : ℕ) : ℕ :=
  (apples + oranges + blueberries) - 3

theorem mary_fruits_left : fruits_left 14 9 6 = 26 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruits_left_l987_98786


namespace NUMINAMATH_CALUDE_college_students_count_l987_98764

theorem college_students_count :
  ∀ (students professors : ℕ),
  students = 15 * professors →
  students + professors = 40000 →
  students = 37500 :=
by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l987_98764


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l987_98782

/-- For a regular polygon with n sides, if each exterior angle measures 45°, then n = 8. -/
theorem regular_polygon_exterior_angle (n : ℕ) : n > 2 → (360 : ℝ) / n = 45 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l987_98782


namespace NUMINAMATH_CALUDE_fraction_evaluation_l987_98789

theorem fraction_evaluation : (3106 - 2935 + 17)^2 / 121 = 292 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l987_98789


namespace NUMINAMATH_CALUDE_fejes_toth_inequality_l987_98706

/-- A convex function on [-1, 1] with absolute value at most 1 -/
structure ConvexBoundedFunction :=
  (f : ℝ → ℝ)
  (convex : ConvexOn ℝ (Set.Icc (-1) 1) f)
  (bounded : ∀ x ∈ Set.Icc (-1) 1, |f x| ≤ 1)

/-- The theorem statement -/
theorem fejes_toth_inequality (F : ConvexBoundedFunction) :
  ∃ (a b : ℝ), ∫ x in Set.Icc (-1) 1, |F.f x - (a * x + b)| ≤ 4 - Real.sqrt 8 := by
  sorry

end NUMINAMATH_CALUDE_fejes_toth_inequality_l987_98706


namespace NUMINAMATH_CALUDE_xy_yz_zx_geq_3_l987_98739

theorem xy_yz_zx_geq_3 (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (heq : x + y + z = 1/x + 1/y + 1/z) : x*y + y*z + z*x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_yz_zx_geq_3_l987_98739


namespace NUMINAMATH_CALUDE_only_expr3_not_factorable_l987_98741

-- Define the expressions
def expr1 (a b : ℝ) := a^2 - b^2
def expr2 (x y z : ℝ) := 49*x^2 - y^2*z^2
def expr3 (x y : ℝ) := -x^2 - y^2
def expr4 (m n p : ℝ) := 16*m^2*n^2 - 25*p^2

-- Define the difference of squares formula
def diff_of_squares (a b : ℝ) := (a + b) * (a - b)

-- Theorem statement
theorem only_expr3_not_factorable :
  (∃ (a b : ℝ), expr1 a b = diff_of_squares a b) ∧
  (∃ (x y z : ℝ), expr2 x y z = diff_of_squares (7*x) (y*z)) ∧
  (∀ (x y : ℝ), ¬∃ (a b : ℝ), expr3 x y = diff_of_squares a b) ∧
  (∃ (m n p : ℝ), expr4 m n p = diff_of_squares (4*m*n) (5*p)) :=
sorry

end NUMINAMATH_CALUDE_only_expr3_not_factorable_l987_98741


namespace NUMINAMATH_CALUDE_f_composition_value_l987_98774

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem f_composition_value :
  f (f (π / 12)) = (1 / 2) * Real.sin (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l987_98774


namespace NUMINAMATH_CALUDE_arcade_tickets_l987_98730

/-- The number of tickets Dave and Alex had combined at the start -/
def total_tickets (dave_spent dave_left alex_spent alex_left : ℕ) : ℕ :=
  (dave_spent + dave_left) + (alex_spent + alex_left)

/-- Theorem stating the total number of tickets Dave and Alex had at the start -/
theorem arcade_tickets : total_tickets 43 55 65 42 = 205 := by
  sorry

end NUMINAMATH_CALUDE_arcade_tickets_l987_98730


namespace NUMINAMATH_CALUDE_roots_sum_powers_l987_98736

theorem roots_sum_powers (c d : ℝ) : 
  c^2 - 6*c + 10 = 0 → 
  d^2 - 6*d + 10 = 0 → 
  c^3 + c^5 * d^3 + c^3 * d^5 + d^3 = 16036 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l987_98736


namespace NUMINAMATH_CALUDE_f_is_quadratic_l987_98772

/-- Definition of a quadratic equation in one variable x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l987_98772


namespace NUMINAMATH_CALUDE_no_real_solutions_greater_than_one_l987_98776

theorem no_real_solutions_greater_than_one :
  ∀ x : ℝ, x > 1 → (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 22 * x^9 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_greater_than_one_l987_98776


namespace NUMINAMATH_CALUDE_cos_minus_sin_special_angle_l987_98704

/-- An angle whose initial side coincides with the non-negative x-axis
    and whose terminal side lies on the ray 4x - 3y = 0 (x ≤ 0) -/
def special_angle (α : Real) : Prop :=
  ∃ (x y : Real), x ≤ 0 ∧ 4 * x - 3 * y = 0 ∧
  Real.cos α = x / Real.sqrt (x^2 + y^2) ∧
  Real.sin α = y / Real.sqrt (x^2 + y^2)

/-- Theorem: For a special angle α, cos α - sin α = 1/5 -/
theorem cos_minus_sin_special_angle (α : Real) (h : special_angle α) :
  Real.cos α - Real.sin α = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_minus_sin_special_angle_l987_98704


namespace NUMINAMATH_CALUDE_negative_of_negative_equals_absolute_value_l987_98795

theorem negative_of_negative_equals_absolute_value : -(-5) = |(-5)| := by
  sorry

end NUMINAMATH_CALUDE_negative_of_negative_equals_absolute_value_l987_98795


namespace NUMINAMATH_CALUDE_soup_tasting_equivalent_to_sample_estimation_l987_98744

/-- Represents the entire soup -/
def Soup : Type := Unit

/-- Represents a small portion of the soup -/
def SoupSample : Type := Unit

/-- Represents the action of tasting a small portion of soup -/
def TasteSoup : SoupSample → Bool := fun _ => true

/-- Represents a population in a statistical survey -/
def Population : Type := Unit

/-- Represents a sample from a population -/
def PopulationSample : Type := Unit

/-- Represents the process of sample estimation in statistics -/
def SampleEstimation : PopulationSample → Population → Prop := fun _ _ => true

/-- Theorem stating that tasting a small portion of soup is mathematically equivalent
    to using sample estimation in statistical surveys -/
theorem soup_tasting_equivalent_to_sample_estimation :
  ∀ (soup : Soup) (sample : SoupSample) (pop : Population) (pop_sample : PopulationSample),
  TasteSoup sample ↔ SampleEstimation pop_sample pop :=
sorry

end NUMINAMATH_CALUDE_soup_tasting_equivalent_to_sample_estimation_l987_98744


namespace NUMINAMATH_CALUDE_committee_probability_l987_98717

def total_members : ℕ := 30
def boys : ℕ := 12
def girls : ℕ := 18
def committee_size : ℕ := 5

theorem committee_probability :
  let total_combinations := Nat.choose total_members committee_size
  let all_boys_combinations := Nat.choose boys committee_size
  let all_girls_combinations := Nat.choose girls committee_size
  let favorable_combinations := total_combinations - (all_boys_combinations + all_girls_combinations)
  (favorable_combinations : ℚ) / total_combinations = 59 / 63 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l987_98717


namespace NUMINAMATH_CALUDE_obtuse_triangle_side_length_range_l987_98753

theorem obtuse_triangle_side_length_range (a : ℝ) :
  (∃ (x y z : ℝ), x = a ∧ y = a + 3 ∧ z = a + 6 ∧
   x + y > z ∧ y + z > x ∧ z + x > y ∧  -- triangle inequality
   z^2 > x^2 + y^2)  -- obtuse triangle condition
  ↔ 3 < a ∧ a < 9 := by
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_length_range_l987_98753


namespace NUMINAMATH_CALUDE_sin_translation_l987_98751

theorem sin_translation (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 3)
  let translation : ℝ := π / 6
  let result : ℝ → ℝ := λ x => Real.sin (2 * x + 2 * π / 3)
  (λ x => f (x + translation)) = result := by
sorry

end NUMINAMATH_CALUDE_sin_translation_l987_98751


namespace NUMINAMATH_CALUDE_smallest_m_for_exact_tax_l987_98797

theorem smallest_m_for_exact_tax : ∃ (x : ℕ+), 
  (106 * x : ℕ) % 100 = 0 ∧ 
  (106 * x : ℕ) / 100 = 53 ∧ 
  ∀ (m : ℕ+), m < 53 → ¬∃ (y : ℕ+), (106 * y : ℕ) % 100 = 0 ∧ (106 * y : ℕ) / 100 = m := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_for_exact_tax_l987_98797


namespace NUMINAMATH_CALUDE_intersection_correct_l987_98758

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℚ
  y : ℚ

/-- Represents a parametric line in 2D -/
structure ParametricLine where
  origin : Vector2D
  direction : Vector2D

/-- The first line -/
def line1 : ParametricLine :=
  { origin := { x := 2, y := 2 },
    direction := { x := 3, y := -4 } }

/-- The second line -/
def line2 : ParametricLine :=
  { origin := { x := 4, y := -6 },
    direction := { x := 5, y := 3 } }

/-- Calculates the point on a parametric line given a parameter value -/
def pointOnLine (line : ParametricLine) (t : ℚ) : Vector2D :=
  { x := line.origin.x + t * line.direction.x,
    y := line.origin.y + t * line.direction.y }

/-- The intersection point of the two lines -/
def intersectionPoint : Vector2D :=
  { x := 160 / 29, y := -160 / 29 }

/-- Theorem stating that the calculated intersection point is correct -/
theorem intersection_correct :
  ∃ (t u : ℚ), pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint :=
by
  sorry


end NUMINAMATH_CALUDE_intersection_correct_l987_98758


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l987_98703

/-- 
Given an arithmetic sequence {a_n} with common difference 3,
where a_1, a_3, and a_4 form a geometric sequence,
prove that a_2 = -9
-/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = 3) →  -- arithmetic sequence with common difference 3
  (a 3)^2 = a 1 * a 4 →         -- a_1, a_3, a_4 form a geometric sequence
  a 2 = -9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l987_98703


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l987_98714

theorem min_value_squared_sum (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) 
  (h2 : t * u * v * w = 25) : 
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 400 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l987_98714


namespace NUMINAMATH_CALUDE_largest_k_for_right_triangle_inequality_l987_98726

theorem largest_k_for_right_triangle_inequality :
  ∃ (k : ℝ), k = (3 * Real.sqrt 2 - 4) / 2 ∧
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 → a^2 + b^2 = c^2 →
    a^3 + b^3 + c^3 ≥ k * (a + b + c)^3) ∧
  (∀ (k' : ℝ), k' > k →
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧
      a^3 + b^3 + c^3 < k' * (a + b + c)^3) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_right_triangle_inequality_l987_98726


namespace NUMINAMATH_CALUDE_ice_cream_theorem_l987_98731

/-- The number of permutations of n distinct elements -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- There are 5 distinct flavors of ice cream -/
def num_flavors : ℕ := 5

/-- The number of ways to arrange ice cream scoops -/
def ice_cream_arrangements : ℕ := permutations num_flavors

theorem ice_cream_theorem : ice_cream_arrangements = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_theorem_l987_98731


namespace NUMINAMATH_CALUDE_opposite_of_negative_nine_l987_98792

theorem opposite_of_negative_nine :
  ∃ (x : ℤ), (x + (-9) = 0) ∧ (x = 9) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_nine_l987_98792


namespace NUMINAMATH_CALUDE_school_classes_count_l987_98701

theorem school_classes_count (sheets_per_class_per_day : ℕ) 
                              (total_sheets_per_week : ℕ) 
                              (school_days_per_week : ℕ) :
  sheets_per_class_per_day = 200 →
  total_sheets_per_week = 9000 →
  school_days_per_week = 5 →
  (total_sheets_per_week / (sheets_per_class_per_day * school_days_per_week) : ℕ) = 9 := by
sorry

end NUMINAMATH_CALUDE_school_classes_count_l987_98701


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l987_98724

/-- Theorem: For a parabola y = ax^2 where a > 0, if the distance from the focus to the directrix is 1, then a = 1/2 -/
theorem parabola_focus_directrix_distance (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, y = a * x^2) → -- Parabola equation
  (∃ p : ℝ, p = 1 ∧ p = 1 / (2 * a)) → -- Distance from focus to directrix is 1
  a = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l987_98724


namespace NUMINAMATH_CALUDE_distance_between_points_l987_98728

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (6, 5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l987_98728


namespace NUMINAMATH_CALUDE_three_even_out_of_five_probability_l987_98712

/-- A fair 10-sided die -/
def TenSidedDie : Type := Fin 10

/-- The probability of rolling an even number on a 10-sided die -/
def probEven : ℚ := 1/2

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The number of dice that should show an even number -/
def numEven : ℕ := 3

/-- The probability of exactly three out of five 10-sided dice showing an even number -/
def probThreeEvenOutOfFive : ℚ := 5/16

theorem three_even_out_of_five_probability :
  (Nat.choose numDice numEven : ℚ) * probEven^numEven * (1 - probEven)^(numDice - numEven) = probThreeEvenOutOfFive :=
sorry

end NUMINAMATH_CALUDE_three_even_out_of_five_probability_l987_98712


namespace NUMINAMATH_CALUDE_square_of_linear_expression_l987_98775

/-- F is a quadratic function of x with parameter m -/
def F (x m : ℚ) : ℚ := (6 * x^2 + 16 * x + 3 * m) / 6

/-- A linear function of x -/
def linear (a b x : ℚ) : ℚ := a * x + b

theorem square_of_linear_expression (m : ℚ) :
  (∃ a b : ℚ, ∀ x : ℚ, F x m = (linear a b x)^2) → m = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_linear_expression_l987_98775


namespace NUMINAMATH_CALUDE_diophantine_equation_solvable_l987_98750

theorem diophantine_equation_solvable (n : ℤ) :
  ∃ (x y z : ℤ), 10 * x * y + 17 * y * z + 27 * z * x = n :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solvable_l987_98750


namespace NUMINAMATH_CALUDE_max_product_on_circle_l987_98754

/-- The maximum product of xy for integer points on x^2 + y^2 = 100 is 48 -/
theorem max_product_on_circle : 
  (∃ (a b : ℤ), a^2 + b^2 = 100 ∧ a * b = 48) ∧ 
  (∀ (x y : ℤ), x^2 + y^2 = 100 → x * y ≤ 48) := by
  sorry

#check max_product_on_circle

end NUMINAMATH_CALUDE_max_product_on_circle_l987_98754


namespace NUMINAMATH_CALUDE_greg_adam_marble_difference_l987_98771

/-- Given that Adam has 29 marbles, Greg has 43 marbles, and Greg has more marbles than Adam,
    prove that Greg has 14 more marbles than Adam. -/
theorem greg_adam_marble_difference :
  ∀ (adam_marbles greg_marbles : ℕ),
    adam_marbles = 29 →
    greg_marbles = 43 →
    greg_marbles > adam_marbles →
    greg_marbles - adam_marbles = 14 := by
  sorry

end NUMINAMATH_CALUDE_greg_adam_marble_difference_l987_98771


namespace NUMINAMATH_CALUDE_extra_oil_amount_l987_98715

-- Define the given conditions
def price_reduction : ℚ := 25 / 100
def reduced_price : ℚ := 40
def total_money : ℚ := 800

-- Define the function to calculate the original price
def original_price : ℚ := reduced_price / (1 - price_reduction)

-- Define the function to calculate the amount of oil that can be bought
def oil_amount (price : ℚ) : ℚ := total_money / price

-- State the theorem
theorem extra_oil_amount : 
  oil_amount reduced_price - oil_amount original_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_extra_oil_amount_l987_98715


namespace NUMINAMATH_CALUDE_children_savings_l987_98799

def josiah_daily_savings : ℚ := 0.25
def josiah_days : ℕ := 24

def leah_daily_savings : ℚ := 0.50
def leah_days : ℕ := 20

def megan_days : ℕ := 12

def total_savings (j_daily : ℚ) (j_days : ℕ) (l_daily : ℚ) (l_days : ℕ) (m_days : ℕ) : ℚ :=
  j_daily * j_days + l_daily * l_days + 2 * l_daily * m_days

theorem children_savings : 
  total_savings josiah_daily_savings josiah_days leah_daily_savings leah_days megan_days = 28 := by
  sorry

end NUMINAMATH_CALUDE_children_savings_l987_98799


namespace NUMINAMATH_CALUDE_cans_collected_l987_98735

theorem cans_collected (total_cans : ℕ) (ladonna_cans : ℕ) (prikya_cans : ℕ) (yoki_cans : ℕ) :
  total_cans = 85 →
  ladonna_cans = 25 →
  prikya_cans = 2 * ladonna_cans →
  yoki_cans = total_cans - (ladonna_cans + prikya_cans) →
  yoki_cans = 10 := by
  sorry

end NUMINAMATH_CALUDE_cans_collected_l987_98735


namespace NUMINAMATH_CALUDE_discounted_price_l987_98702

theorem discounted_price (original_price : ℝ) : 
  original_price * (1 - 0.20) * (1 - 0.10) * (1 - 0.05) = 6840 → 
  original_price = 10000 := by
sorry

end NUMINAMATH_CALUDE_discounted_price_l987_98702


namespace NUMINAMATH_CALUDE_simon_practice_requirement_l987_98761

def week1_hours : ℝ := 12
def week2_hours : ℝ := 16
def week3_hours : ℝ := 14
def total_weeks : ℝ := 4
def required_average : ℝ := 15

def fourth_week_hours : ℝ := 18

theorem simon_practice_requirement :
  (week1_hours + week2_hours + week3_hours + fourth_week_hours) / total_weeks = required_average :=
by sorry

end NUMINAMATH_CALUDE_simon_practice_requirement_l987_98761


namespace NUMINAMATH_CALUDE_vasya_clock_problem_l987_98709

theorem vasya_clock_problem :
  ¬ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 12 ∧ b ≤ 59 ∧ (100 * a + b) % (60 * a + b) = 0 :=
by sorry

end NUMINAMATH_CALUDE_vasya_clock_problem_l987_98709


namespace NUMINAMATH_CALUDE_vector_sum_proof_l987_98725

/-- Given two vectors a and b in ℝ², prove that their sum is (2, 4) -/
theorem vector_sum_proof :
  let a : ℝ × ℝ := (-1, 6)
  let b : ℝ × ℝ := (3, -2)
  a + b = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l987_98725


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l987_98763

theorem fraction_subtraction_simplification :
  (9 : ℚ) / 19 - 3 / 57 - 1 / 3 = 5 / 57 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l987_98763


namespace NUMINAMATH_CALUDE_gary_remaining_money_l987_98769

/-- The amount of money Gary has left after buying a pet snake -/
def money_left (initial_amount spent_amount : ℕ) : ℕ :=
  initial_amount - spent_amount

/-- Theorem stating that Gary has 18 dollars left after buying a pet snake -/
theorem gary_remaining_money :
  money_left 73 55 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gary_remaining_money_l987_98769


namespace NUMINAMATH_CALUDE_exponent_problem_l987_98746

theorem exponent_problem (x y : ℝ) (m n : ℕ) (h : x ≠ 0) (h' : y ≠ 0) :
  x^m * y^n / ((1/4) * x^3 * y) = 4 * x^2 → m = 5 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l987_98746


namespace NUMINAMATH_CALUDE_order_of_powers_l987_98707

theorem order_of_powers : (3/5)^(1/5 : ℝ) > (1/5 : ℝ)^(1/5 : ℝ) ∧ (1/5 : ℝ)^(1/5 : ℝ) > (1/5 : ℝ)^(3/5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_order_of_powers_l987_98707


namespace NUMINAMATH_CALUDE_unique_solution_system_l987_98796

theorem unique_solution_system (x y z : ℝ) : 
  (x + y = 2 ∧ x * y - z^2 = 1) → (x = 1 ∧ y = 1 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l987_98796


namespace NUMINAMATH_CALUDE_range_of_y_over_x_l987_98738

theorem range_of_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 3) :
  ∃ (k : ℝ), y / x = k ∧ -Real.sqrt 3 ≤ k ∧ k ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_y_over_x_l987_98738


namespace NUMINAMATH_CALUDE_units_digit_of_power_l987_98700

theorem units_digit_of_power (n : ℕ) : n > 0 → (7^(7 * (13^13))) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_l987_98700


namespace NUMINAMATH_CALUDE_madeline_money_l987_98760

theorem madeline_money (madeline_money : ℝ) (brother_money : ℝ) : 
  brother_money = madeline_money / 2 →
  madeline_money + brother_money = 72 →
  madeline_money = 48 := by
sorry

end NUMINAMATH_CALUDE_madeline_money_l987_98760


namespace NUMINAMATH_CALUDE_circle_equations_l987_98723

/-- A circle in the Cartesian coordinate system -/
structure Circle where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ α, x α = 2 + 2 * Real.cos α
  h_y : ∀ α, y α = 2 * Real.sin α

/-- The Cartesian equation of the circle -/
def cartesian_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

/-- The polar equation of the circle -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.cos θ

theorem circle_equations (c : Circle) :
  (∀ x y, cartesian_equation c x y ↔ ∃ α, c.x α = x ∧ c.y α = y) ∧
  (∀ ρ θ, polar_equation ρ θ ↔ cartesian_equation c (ρ * Real.cos θ) (ρ * Real.sin θ)) := by
  sorry

end NUMINAMATH_CALUDE_circle_equations_l987_98723


namespace NUMINAMATH_CALUDE_smallest_N_for_P_less_than_four_fifths_l987_98755

/-- The probability function P(N) as described in the problem -/
def P (N : ℕ) : ℚ :=
  (2 * N * N) / (9 * (N + 2) * (N + 3))

/-- Predicate to check if a number is a multiple of 6 -/
def isMultipleOf6 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 6 * k

theorem smallest_N_for_P_less_than_four_fifths :
  ∀ N : ℕ, isMultipleOf6 N → N < 600 → P N ≥ 4/5 ∧
  isMultipleOf6 600 ∧ P 600 < 4/5 := by
  sorry

#eval P 600 -- To verify that P(600) is indeed less than 4/5

end NUMINAMATH_CALUDE_smallest_N_for_P_less_than_four_fifths_l987_98755


namespace NUMINAMATH_CALUDE_sum_seven_to_ten_l987_98740

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = a n * q
  sum_first_two : a 1 + a 2 = 2
  sum_third_fourth : a 3 + a 4 = 4

/-- The sum of the 7th to 10th terms of the geometric sequence is 48 -/
theorem sum_seven_to_ten (seq : GeometricSequence) :
  seq.a 7 + seq.a 8 + seq.a 9 + seq.a 10 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_to_ten_l987_98740


namespace NUMINAMATH_CALUDE_expression_value_l987_98737

theorem expression_value : 
  1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l987_98737


namespace NUMINAMATH_CALUDE_factorization_equalities_l987_98727

theorem factorization_equalities (x y a : ℝ) : 
  (x^4 - 9*x^2 = x^2*(x+3)*(x-3)) ∧ 
  (25*x^2*y + 20*x*y^2 + 4*y^3 = y*(5*x+2*y)^2) ∧ 
  (x^2*(a-1) + y^2*(1-a) = (a-1)*(x+y)*(x-y)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equalities_l987_98727


namespace NUMINAMATH_CALUDE_college_student_count_l987_98748

theorem college_student_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 200) :
  boys + girls = 520 := by
  sorry

end NUMINAMATH_CALUDE_college_student_count_l987_98748


namespace NUMINAMATH_CALUDE_cubic_difference_division_l987_98791

theorem cubic_difference_division (a b : ℝ) (ha : a = 6) (hb : b = 3) :
  (a^3 - b^3) / (a^2 + a*b + b^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_division_l987_98791


namespace NUMINAMATH_CALUDE_problem_solution_l987_98779

def A : Set ℝ := {x | x^2 + 5*x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + 2*(m+1)*x + m^2 - 3 = 0}

theorem problem_solution :
  (A ∪ B 0 = {-6, 1, -3}) ∧
  (∀ m : ℝ, B m ⊆ A ↔ m ≤ -2) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l987_98779


namespace NUMINAMATH_CALUDE_find_n_l987_98768

/-- Definition of S_n -/
def S (n : ℕ) : ℚ := n / (n + 1)

/-- Theorem stating that n = 6 satisfies the given conditions -/
theorem find_n : ∃ (n : ℕ), S n * S (n + 1) = 3/4 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l987_98768


namespace NUMINAMATH_CALUDE_speech_arrangement_count_l987_98722

/-- The number of ways to arrange speeches for 3 boys and 2 girls chosen from a group of 4 boys and 3 girls, where the girls do not give consecutive speeches. -/
def speech_arrangements (total_boys : ℕ) (total_girls : ℕ) (chosen_boys : ℕ) (chosen_girls : ℕ) : ℕ :=
  (Nat.choose total_boys chosen_boys) * 
  (Nat.choose total_girls chosen_girls) * 
  (Nat.factorial chosen_boys) * 
  (Nat.factorial (chosen_boys + 1))

theorem speech_arrangement_count :
  speech_arrangements 4 3 3 2 = 864 := by
  sorry

end NUMINAMATH_CALUDE_speech_arrangement_count_l987_98722


namespace NUMINAMATH_CALUDE_no_solution_exists_l987_98767

theorem no_solution_exists : ¬ ∃ (x : ℕ), 
  (18 + x = 2 * (5 + x)) ∧ 
  (18 + x = 3 * (2 + x)) ∧ 
  ((18 + x) + (5 + x) + (2 + x) = 50) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l987_98767


namespace NUMINAMATH_CALUDE_rohans_age_puzzle_l987_98721

/-- 
Given that Rohan is currently 25 years old, this theorem proves that the number of years 
into the future when Rohan will be 4 times as old as he was the same number of years ago is 15.
-/
theorem rohans_age_puzzle : 
  ∃ x : ℕ, (25 + x = 4 * (25 - x)) ∧ x = 15 :=
by sorry

end NUMINAMATH_CALUDE_rohans_age_puzzle_l987_98721


namespace NUMINAMATH_CALUDE_probability_vowel_in_mathematics_l987_98756

def english_alphabet : Finset Char := sorry

def vowels : Finset Char := {'a', 'e', 'i', 'o', 'u'}

def mathematics : List Char := ['m', 'a', 't', 'h', 'e', 'm', 'a', 't', 'i', 'c', 's']

theorem probability_vowel_in_mathematics :
  (Finset.filter (fun c => c ∈ vowels) mathematics.toFinset).card / mathematics.length = 4 / 11 :=
by sorry

end NUMINAMATH_CALUDE_probability_vowel_in_mathematics_l987_98756


namespace NUMINAMATH_CALUDE_circle_area_outside_triangle_l987_98718

-- Define the right triangle ABC
structure RightTriangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  right_angle : AC^2 = AB^2 + BC^2

-- Define the circle
structure TangentCircle (t : RightTriangle) where
  radius : ℝ
  tangent_AB : radius = t.AB / 2
  diametric_point_on_BC : radius * 2 ≤ t.BC

-- Main theorem
theorem circle_area_outside_triangle (t : RightTriangle) (c : TangentCircle t)
  (h1 : t.AB = 8)
  (h2 : t.BC = 10) :
  (π * c.radius^2 / 4) - (c.radius^2 / 2) = 4*π - 8 := by
  sorry


end NUMINAMATH_CALUDE_circle_area_outside_triangle_l987_98718


namespace NUMINAMATH_CALUDE_classroom_desks_l987_98798

theorem classroom_desks :
  ∀ N y : ℕ,
  (3 * N = 4 * y) →  -- After 1/4 of students leave, 3/4N = 4/7y simplifies to 3N = 4y
  y ≤ 30 →
  y = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_desks_l987_98798


namespace NUMINAMATH_CALUDE_power_property_iff_square_property_l987_98729

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, a * b ≠ 0 → f (a * b) ≥ f a + f b

/-- The property that f(a^n) = n * f(a) for all non-zero a and natural n -/
def SatisfiesPowerProperty (f : ℤ → ℤ) : Prop :=
  ∀ a : ℤ, a ≠ 0 → ∀ n : ℕ, f (a ^ n) = n * f a

/-- The property that f(a^2) = 2 * f(a) for all non-zero a -/
def SatisfiesSquareProperty (f : ℤ → ℤ) : Prop :=
  ∀ a : ℤ, a ≠ 0 → f (a ^ 2) = 2 * f a

theorem power_property_iff_square_property (f : ℤ → ℤ) (h : SatisfiesInequality f) :
  SatisfiesPowerProperty f ↔ SatisfiesSquareProperty f :=
sorry

end NUMINAMATH_CALUDE_power_property_iff_square_property_l987_98729


namespace NUMINAMATH_CALUDE_h_of_two_equals_fifteen_l987_98762

theorem h_of_two_equals_fifteen (h : ℝ → ℝ) 
  (h_def : ∀ x : ℝ, h (3 * x - 4) = 4 * x + 7) : 
  h 2 = 15 := by
sorry

end NUMINAMATH_CALUDE_h_of_two_equals_fifteen_l987_98762


namespace NUMINAMATH_CALUDE_boat_distance_theorem_l987_98752

/-- The distance traveled by a boat against a water flow -/
def distance_traveled (a : ℝ) : ℝ :=
  3 * (a - 3)

/-- Theorem: The distance traveled by a boat against a water flow in 3 hours
    is 3(a-3) km, given that the boat's speed in still water is a km/h
    and the water flow speed is 3 km/h. -/
theorem boat_distance_theorem (a : ℝ) :
  let boat_speed := a
  let water_flow_speed := (3 : ℝ)
  let travel_time := (3 : ℝ)
  distance_traveled a = travel_time * (boat_speed - water_flow_speed) :=
by
  sorry


end NUMINAMATH_CALUDE_boat_distance_theorem_l987_98752


namespace NUMINAMATH_CALUDE_square_perimeters_l987_98790

theorem square_perimeters (a b : ℝ) (h1 : a = 3 * b) 
  (h2 : a ^ 2 + b ^ 2 = 130) (h3 : a ^ 2 - b ^ 2 = 108) : 
  4 * a + 4 * b = 16 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_square_perimeters_l987_98790


namespace NUMINAMATH_CALUDE_number_fraction_problem_l987_98773

theorem number_fraction_problem (n : ℝ) : 
  (1/3) * (1/4) * (1/5) * n = 15 → (3/10) * n = 270 := by
sorry

end NUMINAMATH_CALUDE_number_fraction_problem_l987_98773


namespace NUMINAMATH_CALUDE_employee_selection_distribution_l987_98781

theorem employee_selection_distribution 
  (total_employees : ℕ) 
  (under_35 : ℕ) 
  (between_35_49 : ℕ) 
  (over_50 : ℕ) 
  (selected : ℕ) 
  (h1 : total_employees = 500) 
  (h2 : under_35 = 125) 
  (h3 : between_35_49 = 280) 
  (h4 : over_50 = 95) 
  (h5 : selected = 100) 
  (h6 : total_employees = under_35 + between_35_49 + over_50) :
  let select_under_35 := (under_35 * selected) / total_employees
  let select_between_35_49 := (between_35_49 * selected) / total_employees
  let select_over_50 := (over_50 * selected) / total_employees
  select_under_35 = 25 ∧ select_between_35_49 = 56 ∧ select_over_50 = 19 := by
  sorry

end NUMINAMATH_CALUDE_employee_selection_distribution_l987_98781


namespace NUMINAMATH_CALUDE_other_diagonal_length_l987_98733

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ
  d2 : ℝ
  area : ℝ

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.d1 * r.d2) / 2

/-- Given a rhombus with one diagonal of 17 cm and area of 170 cm², 
    the other diagonal is 20 cm -/
theorem other_diagonal_length :
  ∀ (r : Rhombus), r.d1 = 17 ∧ r.area = 170 → r.d2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l987_98733


namespace NUMINAMATH_CALUDE_polar_equation_is_circle_l987_98749

/-- The curve represented by the polar equation ρ = sin θ + cos θ is a circle. -/
theorem polar_equation_is_circle :
  ∀ (ρ θ : ℝ), ρ = Real.sin θ + Real.cos θ →
  ∃ (x₀ y₀ r : ℝ), ∀ (x y : ℝ),
    (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
    (x - x₀)^2 + (y - y₀)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_polar_equation_is_circle_l987_98749


namespace NUMINAMATH_CALUDE_problem_statement_l987_98742

theorem problem_statement (x : ℝ) (h : Real.exp (x * Real.log 9) + Real.exp (x * Real.log 3) = 6) :
  Real.exp ((1 / x) * Real.log 16) + Real.exp ((1 / x) * Real.log 4) = 90 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l987_98742


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l987_98785

theorem partial_fraction_decomposition :
  ∃! (A B C D : ℚ),
    ∀ (x : ℝ), x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ -1 →
      (x^2 - 9) / ((x - 2) * (x - 3) * (x - 5) * (x + 1)) =
      A / (x - 2) + B / (x - 3) + C / (x - 5) + D / (x + 1) ∧
      A = -5/9 ∧ B = 0 ∧ C = 4/9 ∧ D = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l987_98785


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l987_98720

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - 2*k + 3 ≠ 0) → k < 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l987_98720


namespace NUMINAMATH_CALUDE_sum_of_specific_primes_l987_98745

def smallest_odd_prime : ℕ := 3

def largest_prime_less_than_50 : ℕ := 47

def smallest_prime_greater_than_60 : ℕ := 61

theorem sum_of_specific_primes :
  smallest_odd_prime + largest_prime_less_than_50 + smallest_prime_greater_than_60 = 111 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_specific_primes_l987_98745


namespace NUMINAMATH_CALUDE_square_area_decrease_l987_98757

theorem square_area_decrease (s : ℝ) (h : s = 12) :
  let new_s := s * (1 - 0.125)
  (s^2 - new_s^2) / s^2 = 0.25 := by sorry

end NUMINAMATH_CALUDE_square_area_decrease_l987_98757


namespace NUMINAMATH_CALUDE_problem_solution_l987_98793

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the conditions of the problem
def is_purely_imaginary (z : ℂ) : Prop := ∃ (a : ℝ), z = a * i

-- State the theorem
theorem problem_solution (x y : ℂ) 
  (h1 : is_purely_imaginary x) 
  (h2 : y.im = 0) 
  (h3 : 2 * x - 1 + i = y - (3 - y) * i) : 
  x + y = -1 - (5/2) * i := by sorry

end NUMINAMATH_CALUDE_problem_solution_l987_98793


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l987_98759

theorem largest_integer_in_interval (x : ℤ) : 
  (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 3/5 → x ≤ 4 ∧ 
  ∃ y : ℤ, (1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 3/5 ∧ y = 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l987_98759


namespace NUMINAMATH_CALUDE_remaining_money_after_shopping_l987_98788

/-- The amount of money remaining after spending 30% of $500 is $350. -/
theorem remaining_money_after_shopping (initial_amount : ℝ) (spent_percentage : ℝ) 
  (h1 : initial_amount = 500)
  (h2 : spent_percentage = 0.30) :
  initial_amount - (spent_percentage * initial_amount) = 350 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_after_shopping_l987_98788


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l987_98734

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- Calculates the number of games needed to declare a winner in a single-elimination tournament -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams and no ties, 
    the number of games required to declare a winner is 22 -/
theorem single_elimination_tournament_games :
  ∀ (t : Tournament), t.num_teams = 23 → t.no_ties = true → 
  games_to_winner t = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l987_98734


namespace NUMINAMATH_CALUDE_unique_function_f_l987_98716

/-- A function from [1,+∞) to [1,+∞) satisfying given conditions -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, x ≥ 1 → f x ≥ 1) ∧ 
  (∀ x : ℝ, x ≥ 1 → f x ≤ 2 * (x + 1)) ∧
  (∀ x : ℝ, x ≥ 1 → f (x + 1) = (1 / x) * ((f x)^2 - 1))

/-- The unique function satisfying the conditions is f(x) = x + 1 -/
theorem unique_function_f :
  ∃! f : ℝ → ℝ, FunctionF f ∧ ∀ x : ℝ, x ≥ 1 → f x = x + 1 :=
sorry

end NUMINAMATH_CALUDE_unique_function_f_l987_98716


namespace NUMINAMATH_CALUDE_tim_manicure_payment_l987_98747

/-- The total amount paid for a manicure with tip, given the base cost and tip percentage. -/
def total_paid (base_cost : ℝ) (tip_percentage : ℝ) : ℝ :=
  base_cost * (1 + tip_percentage)

/-- Theorem stating that the total amount Tim paid for the manicure is $39. -/
theorem tim_manicure_payment : total_paid 30 0.3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_tim_manicure_payment_l987_98747


namespace NUMINAMATH_CALUDE_isabel_spending_ratio_l987_98713

/-- Given Isabel's initial amount, toy purchase, and final remaining amount,
    prove that the ratio of book cost to money after toy purchase is 1:2 -/
theorem isabel_spending_ratio (initial_amount : ℕ) (remaining_amount : ℕ)
    (h1 : initial_amount = 204)
    (h2 : remaining_amount = 51) :
  let toy_cost : ℕ := initial_amount / 2
  let after_toy : ℕ := initial_amount - toy_cost
  let book_cost : ℕ := after_toy - remaining_amount
  (book_cost : ℚ) / after_toy = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_isabel_spending_ratio_l987_98713


namespace NUMINAMATH_CALUDE_income_savings_theorem_l987_98743

def income_savings_problem (income : ℝ) (savings : ℝ) : Prop :=
  let income_year2 : ℝ := income * 1.25
  let savings_year2 : ℝ := savings * 2
  let expenditure_year1 : ℝ := income - savings
  let expenditure_year2 : ℝ := income_year2 - savings_year2
  (expenditure_year1 + expenditure_year2 = 2 * expenditure_year1) ∧
  (savings / income = 0.25)

theorem income_savings_theorem (income : ℝ) (savings : ℝ) 
  (h : income > 0) : income_savings_problem income savings :=
by
  sorry

#check income_savings_theorem

end NUMINAMATH_CALUDE_income_savings_theorem_l987_98743


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l987_98783

/-- 
Given a quadratic equation x^2 - 4x - m = 0 with two equal real roots,
prove that m = -4.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x - m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y - m = 0 → y = x) → 
  m = -4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l987_98783


namespace NUMINAMATH_CALUDE_no_solution_exists_l987_98732

theorem no_solution_exists : 
  ¬ ∃ (a b : ℕ+), a * b + 75 = 15 * Nat.lcm a b + 10 * Nat.gcd a b :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l987_98732


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_fourth_power_l987_98784

def x : ℕ := 5 * 27 * 64

theorem smallest_y_for_perfect_fourth_power (y : ℕ) : 
  y > 0 ∧ 
  (∀ z : ℕ, z > 0 ∧ z < y → ¬ ∃ n : ℕ, x * z = n^4) ∧
  (∃ n : ℕ, x * y = n^4) → 
  y = 1500 := by sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_fourth_power_l987_98784


namespace NUMINAMATH_CALUDE_third_card_different_suit_probability_l987_98780

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards per suit in a standard deck -/
def CardsPerSuit : ℕ := StandardDeck / NumberOfSuits

/-- The probability of picking a third card of a different suit than the first two,
    given that the first two cards are of different suits -/
def thirdCardDifferentSuitProbability : ℚ :=
  (StandardDeck - 2 - 2 * CardsPerSuit) / (StandardDeck - 2)

/-- Theorem stating that the probability of the third card being of a different suit
    than the first two is 12/25, given the conditions of the problem -/
theorem third_card_different_suit_probability :
  thirdCardDifferentSuitProbability = 12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_third_card_different_suit_probability_l987_98780


namespace NUMINAMATH_CALUDE_perpendicular_planes_l987_98705

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (non_coincident : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (l : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : parallel l β)
  (h3 : contained_in l α)
  (h4 : non_coincident α β) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l987_98705


namespace NUMINAMATH_CALUDE_original_selling_price_l987_98777

theorem original_selling_price (CP : ℝ) : 
  CP * 0.85 = 544 → CP * 1.25 = 800 := by
  sorry

end NUMINAMATH_CALUDE_original_selling_price_l987_98777


namespace NUMINAMATH_CALUDE_number_wall_solution_l987_98766

/-- Represents a number wall with four levels -/
structure NumberWall :=
  (bottom_left : ℕ)
  (bottom_middle_left : ℕ)
  (bottom_middle_right : ℕ)
  (bottom_right : ℕ)

/-- Calculates the value of the top block in the number wall -/
def top_block (wall : NumberWall) : ℕ :=
  wall.bottom_left + wall.bottom_middle_left + wall.bottom_middle_right + wall.bottom_right + 30

/-- Theorem: In a number wall where the top block is 42, and the bottom row contains m, 5, 3, and 6 from left to right, the value of m is 12 -/
theorem number_wall_solution (wall : NumberWall) 
  (h1 : wall.bottom_middle_left = 5)
  (h2 : wall.bottom_middle_right = 3)
  (h3 : wall.bottom_right = 6)
  (h4 : top_block wall = 42) : 
  wall.bottom_left = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_solution_l987_98766


namespace NUMINAMATH_CALUDE_train_crossing_time_l987_98794

/-- Calculates the time for a train to cross a signal pole given its length, platform length, and time to cross the platform. -/
theorem train_crossing_time (train_length platform_length time_cross_platform : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 600.0000000000001)
  (h3 : time_cross_platform = 54) : 
  ∃ (time_cross_pole : ℝ), 
    (time_cross_pole ≥ 17.99 ∧ time_cross_pole ≤ 18.01) ∧
    time_cross_pole = train_length / ((train_length + platform_length) / time_cross_platform) :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l987_98794


namespace NUMINAMATH_CALUDE_power_sum_equality_l987_98770

theorem power_sum_equality : (-2)^23 + 5^(2^4 + 3^3 - 4^2) = -8388608 + 5^27 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l987_98770


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l987_98710

def scores : List ℕ := [87, 90, 85, 93, 89, 92]

theorem arithmetic_mean_of_scores :
  (scores.sum : ℚ) / scores.length = 268 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l987_98710


namespace NUMINAMATH_CALUDE_square_diagonal_l987_98787

/-- The diagonal length of a square with area 338 square meters is 26 meters. -/
theorem square_diagonal (area : ℝ) (diagonal : ℝ) : 
  area = 338 → diagonal = 26 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_l987_98787


namespace NUMINAMATH_CALUDE_pascal_row_12_left_half_sum_l987_98765

/-- The sum of the left half of a row in Pascal's Triangle -/
def pascal_left_half_sum (n : ℕ) : ℕ :=
  2^n

/-- Row 12 of Pascal's Triangle -/
def pascal_row_12 : ℕ := 12

theorem pascal_row_12_left_half_sum :
  pascal_left_half_sum pascal_row_12 = 2048 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row_12_left_half_sum_l987_98765
