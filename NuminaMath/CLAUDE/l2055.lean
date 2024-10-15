import Mathlib

namespace NUMINAMATH_CALUDE_triangle_circumradius_l2055_205558

theorem triangle_circumradius (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) :
  let s := (a + b + c) / 2
  (a * b * c) / (4 * Real.sqrt (s * (s - a) * (s - b) * (s - c))) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l2055_205558


namespace NUMINAMATH_CALUDE_sum_divisors_450_prime_factors_l2055_205515

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the positive divisors of 450 has exactly 3 distinct prime factors -/
theorem sum_divisors_450_prime_factors :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by sorry

end NUMINAMATH_CALUDE_sum_divisors_450_prime_factors_l2055_205515


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2055_205573

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l2055_205573


namespace NUMINAMATH_CALUDE_aarti_work_theorem_l2055_205527

/-- Given that Aarti can complete a piece of work in a certain number of days,
    this function calculates how many days she needs to complete a multiple of that work. -/
def days_for_multiple_work (base_days : ℕ) (multiple : ℕ) : ℕ :=
  base_days * multiple

/-- Theorem stating that if Aarti can complete a piece of work in 5 days,
    then she will need 15 days to complete three times the work of the same type. -/
theorem aarti_work_theorem :
  days_for_multiple_work 5 3 = 15 := by sorry

end NUMINAMATH_CALUDE_aarti_work_theorem_l2055_205527


namespace NUMINAMATH_CALUDE_inventory_net_change_l2055_205516

/-- Represents the quantity of an ingredient on a given day -/
structure IngredientQuantity where
  day1 : Float
  day7 : Float

/-- Calculates the change in quantity for an ingredient -/
def calculateChange (q : IngredientQuantity) : Float :=
  q.day1 - q.day7

/-- Represents the inventory of all ingredients -/
structure Inventory where
  bakingPowder : IngredientQuantity
  flour : IngredientQuantity
  sugar : IngredientQuantity
  chocolateChips : IngredientQuantity

/-- Calculates the net change for all ingredients -/
def calculateNetChange (inv : Inventory) : Float :=
  calculateChange inv.bakingPowder +
  calculateChange inv.flour +
  calculateChange inv.sugar +
  calculateChange inv.chocolateChips

theorem inventory_net_change (inv : Inventory) 
  (h1 : inv.bakingPowder = { day1 := 4, day7 := 2.5 })
  (h2 : inv.flour = { day1 := 12, day7 := 7 })
  (h3 : inv.sugar = { day1 := 10, day7 := 6.5 })
  (h4 : inv.chocolateChips = { day1 := 6, day7 := 3.7 }) :
  calculateNetChange inv = 12.3 := by
  sorry

end NUMINAMATH_CALUDE_inventory_net_change_l2055_205516


namespace NUMINAMATH_CALUDE_negation_equivalence_l2055_205598

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Child : U → Prop)
variable (CarefulInvestor : U → Prop)
variable (RecklessInvestor : U → Prop)

-- Define the statements
def AllChildrenAreCareful : Prop := ∀ x, Child x → CarefulInvestor x
def AtLeastOneChildIsReckless : Prop := ∃ x, Child x ∧ RecklessInvestor x

-- The theorem to prove
theorem negation_equivalence : 
  AtLeastOneChildIsReckless U Child RecklessInvestor ↔ 
  ¬(AllChildrenAreCareful U Child CarefulInvestor) :=
sorry

-- Additional assumption: being reckless is the opposite of being careful
axiom reckless_careful_opposite : 
  ∀ x, RecklessInvestor x ↔ ¬(CarefulInvestor x)

end NUMINAMATH_CALUDE_negation_equivalence_l2055_205598


namespace NUMINAMATH_CALUDE_total_cupcakes_calculation_l2055_205545

/-- The number of cupcakes ordered for each event -/
def cupcakes_per_event : ℝ := 96.0

/-- The number of different children's events -/
def number_of_events : ℝ := 8.0

/-- The total number of cupcakes needed -/
def total_cupcakes : ℝ := cupcakes_per_event * number_of_events

theorem total_cupcakes_calculation : total_cupcakes = 768.0 := by
  sorry

end NUMINAMATH_CALUDE_total_cupcakes_calculation_l2055_205545


namespace NUMINAMATH_CALUDE_hexagon_angle_sum_l2055_205511

/-- A hexagon is a polygon with six vertices and six edges. -/
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- The sum of interior angles of a hexagon in degrees. -/
def sum_of_angles (h : Hexagon) : ℝ := 
  sorry

/-- Theorem: In a hexagon where the sum of all interior angles is 90n degrees, n must equal 4. -/
theorem hexagon_angle_sum (h : Hexagon) (n : ℝ) 
  (h_sum : sum_of_angles h = 90 * n) : n = 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_sum_l2055_205511


namespace NUMINAMATH_CALUDE_units_digit_of_power_difference_l2055_205517

theorem units_digit_of_power_difference : (5^35 - 6^21) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_difference_l2055_205517


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2055_205533

/-- A line tangent to a circle and passing through a point -/
theorem tangent_line_to_circle (x y : ℝ) : 
  -- The line equation
  (y - 4 = 3/4 * (x + 3)) →
  -- The line passes through (-3, 4)
  ((-3 : ℝ), (4 : ℝ)) ∈ {(x, y) | y - 4 = 3/4 * (x + 3)} →
  -- The line is tangent to the circle
  (∃! (p : ℝ × ℝ), p ∈ {(x, y) | x^2 + y^2 = 25} ∩ {(x, y) | y - 4 = 3/4 * (x + 3)}) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2055_205533


namespace NUMINAMATH_CALUDE_segment_length_is_70_l2055_205554

/-- Represents a point on a line segment -/
structure PointOnSegment (A B : ℝ) where
  position : ℝ
  h1 : A ≤ position
  h2 : position ≤ B

/-- The line segment AB -/
def lineSegment (A B : ℝ) := {x : ℝ | A ≤ x ∧ x ≤ B}

theorem segment_length_is_70 
  (A B : ℝ) 
  (P Q : PointOnSegment A B) 
  (h_order : P.position < Q.position) 
  (h_same_side : (P.position - A) / (B - A) < 1/2 ∧ (Q.position - A) / (B - A) < 1/2) 
  (h_P_ratio : (P.position - A) / (B - P.position) = 2/3) 
  (h_Q_ratio : (Q.position - A) / (B - Q.position) = 3/4) 
  (h_PQ_length : Q.position - P.position = 2) :
  B - A = 70 := by
  sorry

#check segment_length_is_70

end NUMINAMATH_CALUDE_segment_length_is_70_l2055_205554


namespace NUMINAMATH_CALUDE_dot_product_of_complex_vectors_l2055_205518

def complex_to_vector (z : ℂ) : ℝ × ℝ := (z.re, z.im)

theorem dot_product_of_complex_vectors :
  let Z₁ : ℂ := (1 - 2*I)*I
  let Z₂ : ℂ := (1 - 3*I) / (1 - I)
  let a : ℝ × ℝ := complex_to_vector Z₁
  let b : ℝ × ℝ := complex_to_vector Z₂
  (a.1 * b.1 + a.2 * b.2) = 3 := by sorry

end NUMINAMATH_CALUDE_dot_product_of_complex_vectors_l2055_205518


namespace NUMINAMATH_CALUDE_fixed_fee_december_l2055_205596

/-- Represents the billing information for an online service provider --/
structure BillingInfo where
  dec_fixed_fee : ℝ
  hourly_charge : ℝ
  dec_connect_time : ℝ
  jan_connect_time : ℝ
  dec_bill : ℝ
  jan_bill : ℝ
  jan_fee_increase : ℝ

/-- The fixed monthly fee in December is $10.80 --/
theorem fixed_fee_december (info : BillingInfo) : info.dec_fixed_fee = 10.80 :=
  by
  have h1 : info.dec_bill = 15.00 := by sorry
  have h2 : info.jan_bill = 25.40 := by sorry
  have h3 : info.jan_connect_time = 3 * info.dec_connect_time := by sorry
  have h4 : info.jan_fee_increase = 2 := by sorry
  have h5 : info.dec_fixed_fee + info.hourly_charge * info.dec_connect_time = info.dec_bill := by sorry
  have h6 : (info.dec_fixed_fee + info.jan_fee_increase) + info.hourly_charge * info.jan_connect_time = info.jan_bill := by sorry
  sorry

#check fixed_fee_december

end NUMINAMATH_CALUDE_fixed_fee_december_l2055_205596


namespace NUMINAMATH_CALUDE_matrix_equation_result_l2055_205542

/-- Given two 2x2 matrices A and B, where A is fixed and B has variable entries,
    if AB = BA and 4y ≠ z, then (x - w) / (z - 4y) = 3/8 -/
theorem matrix_equation_result (x y z w : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  4 * y ≠ z →
  A * B = B * A →
  (x - w) / (z - 4 * y) = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_result_l2055_205542


namespace NUMINAMATH_CALUDE_tan_theta_value_l2055_205562

theorem tan_theta_value (θ : Real) (h1 : 0 < θ) (h2 : θ < π/4) 
  (h3 : Real.tan θ + Real.tan (4*θ) = 0) : 
  Real.tan θ = Real.sqrt (5 - 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l2055_205562


namespace NUMINAMATH_CALUDE_total_pokemon_cards_l2055_205553

/-- The number of people with Pokemon cards -/
def num_people : ℕ := 6

/-- The number of Pokemon cards each person has -/
def cards_per_person : ℕ := 100

/-- Theorem: The total number of Pokemon cards for 6 people, each having 100 cards, is equal to 600 -/
theorem total_pokemon_cards : num_people * cards_per_person = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_pokemon_cards_l2055_205553


namespace NUMINAMATH_CALUDE_circle_passes_through_intersections_l2055_205532

/-- Line l₁ -/
def l₁ (x y : ℝ) : Prop := x - 2*y = 0

/-- Line l₂ -/
def l₂ (x y : ℝ) : Prop := y + 1 = 0

/-- Line l₃ -/
def l₃ (x y : ℝ) : Prop := 2*x + y - 1 = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + x + 2*y - 1 = 0

/-- Theorem stating that the circle passes through the intersection points of the lines -/
theorem circle_passes_through_intersections :
  ∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
  (l₁ x₁ y₁ ∧ l₂ x₁ y₁) →
  (l₁ x₂ y₂ ∧ l₃ x₂ y₂) →
  (l₂ x₃ y₃ ∧ l₃ x₃ y₃) →
  circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧ circle_equation x₃ y₃ :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_intersections_l2055_205532


namespace NUMINAMATH_CALUDE_every_positive_integer_appears_l2055_205544

/-- The smallest prime that doesn't divide k -/
def p (k : ℕ+) : ℕ := sorry

/-- The sequence a_n -/
def a : ℕ → ℕ+ → ℕ+
  | 0, a₀ => a₀
  | n + 1, a₀ => sorry

/-- Main theorem: every positive integer appears in the sequence -/
theorem every_positive_integer_appears (a₀ : ℕ+) :
  ∀ m : ℕ+, ∃ n : ℕ, a n a₀ = m := by sorry

end NUMINAMATH_CALUDE_every_positive_integer_appears_l2055_205544


namespace NUMINAMATH_CALUDE_divisibility_by_112_l2055_205535

theorem divisibility_by_112 (m : ℕ) (h1 : m > 0) (h2 : m % 2 = 1) (h3 : m % 3 ≠ 0) :
  112 ∣ ⌊4^m - (2 + Real.sqrt 2)^m⌋ := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_112_l2055_205535


namespace NUMINAMATH_CALUDE_expression_equals_one_l2055_205501

theorem expression_equals_one (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h_sum : p + q + r = 0) :
  (p^2 * q^2 / ((p^2 - q*r) * (q^2 - p*r))) +
  (p^2 * r^2 / ((p^2 - q*r) * (r^2 - p*q))) +
  (q^2 * r^2 / ((q^2 - p*r) * (r^2 - p*q))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2055_205501


namespace NUMINAMATH_CALUDE_flour_already_added_is_three_l2055_205557

/-- The number of cups of flour required by the recipe -/
def total_flour : ℕ := 9

/-- The number of cups of flour Mary still needs to add -/
def flour_to_add : ℕ := 6

/-- The number of cups of flour Mary has already put in -/
def flour_already_added : ℕ := total_flour - flour_to_add

theorem flour_already_added_is_three : flour_already_added = 3 := by
  sorry

end NUMINAMATH_CALUDE_flour_already_added_is_three_l2055_205557


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l2055_205597

theorem rectangle_circle_area_ratio 
  (l w r : ℝ) 
  (h1 : 2 * l + 2 * w = 2 * Real.pi * r) 
  (h2 : l = 3 * w) : 
  (l * w) / (Real.pi * r^2) = 3 * Real.pi / 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l2055_205597


namespace NUMINAMATH_CALUDE_square_field_area_l2055_205570

theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  wire_cost_per_meter = 3.5 →
  total_cost = 2331 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    (4 * side_length - num_gates * gate_width) * wire_cost_per_meter = total_cost ∧
    side_length^2 = 27889 :=
by sorry

end NUMINAMATH_CALUDE_square_field_area_l2055_205570


namespace NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_l2055_205595

theorem polygon_sides_from_interior_angle (n : ℕ) (angle : ℝ) : 
  (n ≥ 3) → (angle = 140) → (n * angle = (n - 2) * 180) → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_l2055_205595


namespace NUMINAMATH_CALUDE_ratio_xyz_l2055_205548

theorem ratio_xyz (x y z : ℝ) (h1 : 0.1 * x = 0.2 * y) (h2 : 0.3 * y = 0.4 * z) :
  ∃ (k : ℝ), k > 0 ∧ x = 8 * k ∧ y = 4 * k ∧ z = 3 * k :=
sorry

end NUMINAMATH_CALUDE_ratio_xyz_l2055_205548


namespace NUMINAMATH_CALUDE_banana_group_size_l2055_205514

def total_bananas : ℕ := 290
def banana_groups : ℕ := 2
def total_oranges : ℕ := 87
def orange_groups : ℕ := 93

theorem banana_group_size : total_bananas / banana_groups = 145 := by
  sorry

end NUMINAMATH_CALUDE_banana_group_size_l2055_205514


namespace NUMINAMATH_CALUDE_problem_solution_l2055_205508

-- Define the solution set
def SolutionSet (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

-- Define the inequality
def Inequality (x m n : ℝ) : Prop := |x - m| ≤ n

theorem problem_solution :
  -- Conditions
  (∀ x, Inequality x m n ↔ SolutionSet x) →
  -- Part 1: Prove m = 2 and n = 2
  (m = 2 ∧ n = 2) ∧
  -- Part 2: Prove minimum value of a + b
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = m/a + n/b → a + b ≥ 2 * Real.sqrt 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = m/a + n/b ∧ a + b = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2055_205508


namespace NUMINAMATH_CALUDE_simplest_fraction_sum_l2055_205538

theorem simplest_fraction_sum (p q : ℕ+) : 
  (p : ℚ) / q = 83125 / 100000 ∧ 
  ∀ (a b : ℕ+), (a : ℚ) / b = p / q → a ≤ p ∧ b ≤ q →
  p + q = 293 := by
sorry

end NUMINAMATH_CALUDE_simplest_fraction_sum_l2055_205538


namespace NUMINAMATH_CALUDE_score_difference_is_1_25_l2055_205512

def score_distribution : List (Float × Float) :=
  [(0.20, 70), (0.20, 80), (0.25, 85), (0.25, 90), (0.10, 100)]

def median_score : Float := 85

def mean_score : Float :=
  score_distribution.foldl (λ acc (percent, score) => acc + percent * score) 0

theorem score_difference_is_1_25 :
  median_score - mean_score = 1.25 := by sorry

end NUMINAMATH_CALUDE_score_difference_is_1_25_l2055_205512


namespace NUMINAMATH_CALUDE_correct_factorization_l2055_205536

theorem correct_factorization (x y : ℝ) : x * (x - y) + y * (y - x) = (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l2055_205536


namespace NUMINAMATH_CALUDE_length_of_BD_l2055_205510

-- Define the triangles and their properties
def right_triangle_ABC (b c : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ), 
    (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) ∧ 
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = b^2 ∧
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = c^2

def right_triangle_ABD (b c : ℝ) : Prop :=
  ∃ (A B D : ℝ × ℝ), 
    (B.1 - A.1) * (D.2 - A.2) = (D.1 - A.1) * (B.2 - A.2) ∧
    (A.1 - D.1)^2 + (A.2 - D.2)^2 = 9 ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = b^2 + c^2

-- The theorem to prove
theorem length_of_BD (b c : ℝ) (h1 : b > 0) (h2 : c > 0) 
  (h3 : right_triangle_ABC b c) (h4 : right_triangle_ABD b c) :
  ∃ (B D : ℝ × ℝ), (B.1 - D.1)^2 + (B.2 - D.2)^2 = b^2 + c^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_length_of_BD_l2055_205510


namespace NUMINAMATH_CALUDE_minuend_value_l2055_205506

theorem minuend_value (M S D : ℤ) : 
  M + S + D = 2016 → M - S = D → M = 1008 := by
  sorry

end NUMINAMATH_CALUDE_minuend_value_l2055_205506


namespace NUMINAMATH_CALUDE_sqrt_32_div_sqrt_8_eq_2_l2055_205500

theorem sqrt_32_div_sqrt_8_eq_2 : Real.sqrt 32 / Real.sqrt 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_32_div_sqrt_8_eq_2_l2055_205500


namespace NUMINAMATH_CALUDE_unique_positive_number_l2055_205559

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 * (1/x) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l2055_205559


namespace NUMINAMATH_CALUDE_remainder_sum_l2055_205572

theorem remainder_sum (n : ℤ) (h : n % 18 = 11) : (n % 2 + n % 9 = 3) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2055_205572


namespace NUMINAMATH_CALUDE_solve_equation_l2055_205552

theorem solve_equation (x : ℝ) : 9 / (5 + 3 / x) = 1 → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2055_205552


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2055_205504

theorem triangle_angle_measure (A B C : Real) : 
  A + B + C = 180 →
  B = A + 20 →
  C = 50 →
  B = 75 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2055_205504


namespace NUMINAMATH_CALUDE_muffin_count_l2055_205567

/-- Given a number of doughnuts and a ratio of doughnuts to muffins, 
    calculate the number of muffins -/
def calculate_muffins (num_doughnuts : ℕ) (doughnut_ratio : ℕ) (muffin_ratio : ℕ) : ℕ :=
  (num_doughnuts / doughnut_ratio) * muffin_ratio

/-- Theorem: Given 50 doughnuts and a ratio of 5 doughnuts to 1 muffin, 
    the number of muffins is 10 -/
theorem muffin_count : calculate_muffins 50 5 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_muffin_count_l2055_205567


namespace NUMINAMATH_CALUDE_range_of_f_set_where_g_less_than_f_l2055_205574

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| + x
def g (x : ℝ) : ℝ := |x + 1|

-- Statement for the range of f
theorem range_of_f : Set.range f = Set.Ici 2 := by sorry

-- Statement for the set where g(x) < f(x)
theorem set_where_g_less_than_f : 
  {x : ℝ | g x < f x} = Set.union (Set.Ioo (-3) 1) (Set.Ioi 3) := by sorry

end NUMINAMATH_CALUDE_range_of_f_set_where_g_less_than_f_l2055_205574


namespace NUMINAMATH_CALUDE_largest_x_value_l2055_205590

theorem largest_x_value : 
  let f (x : ℝ) := (17 * x^2 - 46 * x + 21) / (5 * x - 3) + 7 * x
  ∀ x : ℝ, f x = 8 * x - 2 → x ≤ 5/3 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l2055_205590


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2055_205551

theorem sum_of_fractions : 
  (2 / 10 : ℚ) + (4 / 10 : ℚ) + (6 / 10 : ℚ) + (8 / 10 : ℚ) + (10 / 10 : ℚ) + 
  (12 / 10 : ℚ) + (14 / 10 : ℚ) + (16 / 10 : ℚ) + (18 / 10 : ℚ) + (32 / 10 : ℚ) = 
  (122 / 10 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2055_205551


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1365_l2055_205513

theorem sum_of_largest_and_smallest_prime_factors_of_1365 :
  ∃ (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p ∣ 1365 ∧ 
    q ∣ 1365 ∧ 
    (∀ r : ℕ, Nat.Prime r → r ∣ 1365 → p ≤ r ∧ r ≤ q) ∧ 
    p + q = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1365_l2055_205513


namespace NUMINAMATH_CALUDE_factor_expression_l2055_205565

theorem factor_expression (x y z : ℝ) :
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) = (x + y) * (y + z) * (z + x) :=
by sorry

end NUMINAMATH_CALUDE_factor_expression_l2055_205565


namespace NUMINAMATH_CALUDE_unique_g_function_l2055_205539

-- Define the properties of function g
def is_valid_g (g : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, g (x₁ + x₂) = g x₁ * g x₂) ∧
  (g 1 = 3) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂)

-- Theorem statement
theorem unique_g_function :
  ∃! g : ℝ → ℝ, is_valid_g g ∧ (∀ x : ℝ, g x = 3^x) :=
by sorry

end NUMINAMATH_CALUDE_unique_g_function_l2055_205539


namespace NUMINAMATH_CALUDE_cube_expansion_sum_l2055_205550

/-- Given that for any real number x, x^3 = a₀ + a₁(x-2) + a₂(x-2)² + a₃(x-2)³, 
    prove that a₁ + a₂ + a₃ = 19 -/
theorem cube_expansion_sum (a₀ a₁ a₂ a₃ : ℝ) 
    (h : ∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) : 
  a₁ + a₂ + a₃ = 19 := by
  sorry

end NUMINAMATH_CALUDE_cube_expansion_sum_l2055_205550


namespace NUMINAMATH_CALUDE_total_people_in_program_l2055_205594

theorem total_people_in_program (parents pupils teachers : ℕ) 
  (h1 : parents = 73)
  (h2 : pupils = 724)
  (h3 : teachers = 744) :
  parents + pupils + teachers = 1541 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l2055_205594


namespace NUMINAMATH_CALUDE_upper_line_formula_l2055_205589

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- The sequence of numbers in the lower line -/
def x : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n + 2) => x (n + 1) + x n + 1

/-- The sequence of numbers in the upper line -/
def a (n : ℕ) : ℕ := x (n + 1) - 1

theorem upper_line_formula (n : ℕ) : a n = fib (n + 3) - 2 := by
  sorry

#check upper_line_formula

end NUMINAMATH_CALUDE_upper_line_formula_l2055_205589


namespace NUMINAMATH_CALUDE_angle_c_value_l2055_205502

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem angle_c_value (t : Triangle) 
  (h : t.a^2 + t.b^2 = t.c^2 + Real.sqrt 3 * t.a * t.b) : 
  t.C = π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_value_l2055_205502


namespace NUMINAMATH_CALUDE_irrational_number_existence_l2055_205588

theorem irrational_number_existence : ∃ α : ℝ, (α > 1) ∧ (Irrational α) ∧
  (∀ n : ℕ, n ≥ 1 → (⌊α^n⌋ : ℤ) % 2017 = 0) := by
  sorry

end NUMINAMATH_CALUDE_irrational_number_existence_l2055_205588


namespace NUMINAMATH_CALUDE_tulip_probability_l2055_205509

structure FlowerSet where
  roses : ℕ
  tulips : ℕ
  daisies : ℕ
  lilies : ℕ

def total_flowers (fs : FlowerSet) : ℕ :=
  fs.roses + fs.tulips + fs.daisies + fs.lilies

def probability_of_tulip (fs : FlowerSet) : ℚ :=
  fs.tulips / (total_flowers fs)

theorem tulip_probability (fs : FlowerSet) (h : fs = ⟨3, 2, 4, 6⟩) :
  probability_of_tulip fs = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tulip_probability_l2055_205509


namespace NUMINAMATH_CALUDE_partition_seven_students_l2055_205547

/-- The number of ways to partition 7 students into groups of 2 or 3 -/
def partition_ways : ℕ := 105

/-- The number of students -/
def num_students : ℕ := 7

/-- The possible group sizes -/
def group_sizes : List ℕ := [2, 3]

/-- Theorem stating that the number of ways to partition 7 students into groups of 2 or 3 is 105 -/
theorem partition_seven_students :
  (∀ g ∈ group_sizes, g ≤ num_students) →
  (∃ f : List ℕ, (∀ x ∈ f, x ∈ group_sizes) ∧ f.sum = num_students) →
  partition_ways = 105 := by
  sorry

end NUMINAMATH_CALUDE_partition_seven_students_l2055_205547


namespace NUMINAMATH_CALUDE_expression_simplification_l2055_205537

theorem expression_simplification (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2055_205537


namespace NUMINAMATH_CALUDE_sunset_time_correct_l2055_205586

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

/-- Adds a duration to a time -/
def addDuration (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + d.hours * 60 + d.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem sunset_time_correct 
  (sunrise : Time)
  (daylight : Duration)
  (sunset : Time)
  (h1 : sunrise = { hours := 6, minutes := 45 })
  (h2 : daylight = { hours := 11, minutes := 12 })
  (h3 : sunset = { hours := 17, minutes := 57 }) :
  addDuration sunrise daylight = sunset :=
sorry

end NUMINAMATH_CALUDE_sunset_time_correct_l2055_205586


namespace NUMINAMATH_CALUDE_marbles_remaining_l2055_205543

theorem marbles_remaining (total : ℕ) (white : ℕ) (removed : ℕ) : 
  total = 50 → 
  white = 20 → 
  removed = 2 * (white - (total - white) / 2) → 
  total - removed = 40 := by
sorry

end NUMINAMATH_CALUDE_marbles_remaining_l2055_205543


namespace NUMINAMATH_CALUDE_pi_irrational_among_given_numbers_l2055_205507

theorem pi_irrational_among_given_numbers :
  (∃ (a b : ℤ), (1 : ℝ) / 3 = a / b) ∧
  (∃ (c d : ℤ), (0.201 : ℝ) = c / d) ∧
  (∃ (e f : ℤ), Real.sqrt 9 = e / f) →
  ¬∃ (m n : ℤ), Real.pi = m / n :=
by sorry

end NUMINAMATH_CALUDE_pi_irrational_among_given_numbers_l2055_205507


namespace NUMINAMATH_CALUDE_entire_square_shaded_l2055_205531

/-- The fraction of area shaded in the first step -/
def initial_shaded : ℚ := 5 / 9

/-- The fraction of area remaining unshaded after each step -/
def unshaded_fraction : ℚ := 4 / 9

/-- The sum of the infinite geometric series representing the total shaded area -/
def total_shaded_area : ℚ := initial_shaded / (1 - unshaded_fraction)

/-- Theorem stating that the entire square is shaded in the limit -/
theorem entire_square_shaded : total_shaded_area = 1 := by sorry

end NUMINAMATH_CALUDE_entire_square_shaded_l2055_205531


namespace NUMINAMATH_CALUDE_jacque_suitcase_weight_l2055_205566

/-- The weight of Jacque's suitcase when he arrived in France -/
def initial_weight : ℝ := 5

/-- The weight of one bottle of perfume in ounces -/
def perfume_weight : ℝ := 1.2

/-- The number of bottles of perfume Jacque bought -/
def perfume_count : ℕ := 5

/-- The weight of chocolate in pounds -/
def chocolate_weight : ℝ := 4

/-- The weight of one bar of soap in ounces -/
def soap_weight : ℝ := 5

/-- The number of bars of soap Jacque bought -/
def soap_count : ℕ := 2

/-- The weight of one jar of jam in ounces -/
def jam_weight : ℝ := 8

/-- The number of jars of jam Jacque bought -/
def jam_count : ℕ := 2

/-- The number of ounces in a pound -/
def ounces_per_pound : ℝ := 16

/-- The total weight of Jacque's suitcase on the return flight in pounds -/
def return_weight : ℝ := 11

theorem jacque_suitcase_weight :
  initial_weight + 
  (perfume_weight * perfume_count + soap_weight * soap_count + jam_weight * jam_count) / ounces_per_pound + 
  chocolate_weight = return_weight := by
  sorry

end NUMINAMATH_CALUDE_jacque_suitcase_weight_l2055_205566


namespace NUMINAMATH_CALUDE_ratio_is_sixteen_thirteenths_l2055_205561

/-- An arithmetic sequence with a non-zero common difference where a₉, a₃, and a₁ form a geometric sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  is_geometric : a 3 ^ 2 = a 1 * a 9

/-- The ratio of (a₂ + a₄ + a₁₀) to (a₁ + a₃ + a₉) is 16/13 -/
theorem ratio_is_sixteen_thirteenths (seq : ArithmeticSequence) :
  (seq.a 2 + seq.a 4 + seq.a 10) / (seq.a 1 + seq.a 3 + seq.a 9) = 16 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ratio_is_sixteen_thirteenths_l2055_205561


namespace NUMINAMATH_CALUDE_min_container_cost_l2055_205549

/-- Represents the dimensions and cost of a rectangular container -/
structure Container where
  length : ℝ
  width : ℝ
  height : ℝ
  baseUnitCost : ℝ
  sideUnitCost : ℝ

/-- Calculates the total cost of the container -/
def totalCost (c : Container) : ℝ :=
  c.baseUnitCost * c.length * c.width + 
  c.sideUnitCost * 2 * (c.length + c.width) * c.height

/-- Theorem stating the minimum cost of the container -/
theorem min_container_cost :
  ∃ (c : Container),
    c.height = 1 ∧
    c.length * c.width * c.height = 4 ∧
    c.baseUnitCost = 20 ∧
    c.sideUnitCost = 10 ∧
    (∀ (d : Container),
      d.height = 1 →
      d.length * d.width * d.height = 4 →
      d.baseUnitCost = 20 →
      d.sideUnitCost = 10 →
      totalCost c ≤ totalCost d) ∧
    totalCost c = 160 := by
  sorry

end NUMINAMATH_CALUDE_min_container_cost_l2055_205549


namespace NUMINAMATH_CALUDE_units_digit_sum_in_base_7_l2055_205568

/-- The base of the number system we're working in -/
def base : ℕ := 7

/-- Function to get the units digit of a number in the given base -/
def unitsDigit (n : ℕ) : ℕ := n % base

/-- First number in the sum -/
def num1 : ℕ := 52

/-- Second number in the sum -/
def num2 : ℕ := 62

/-- Theorem stating that the units digit of the sum of num1 and num2 in base 7 is 4 -/
theorem units_digit_sum_in_base_7 : 
  unitsDigit (num1 + num2) = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_in_base_7_l2055_205568


namespace NUMINAMATH_CALUDE_symmetric_line_l2055_205580

/-- Given a line L1 with equation 2x-y+3=0 and a point M(-1,2),
    prove that the line L2 symmetric to L1 with respect to M
    has the equation 2x-y+5=0 -/
theorem symmetric_line (x y : ℝ) :
  (2 * x - y + 3 = 0) →
  (2 * (-2 - x) - (4 - y) + 3 = 0) →
  (2 * x - y + 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_l2055_205580


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l2055_205584

theorem pet_store_siamese_cats 
  (total_cats : ℕ) 
  (siamese_cats : ℕ) 
  (house_cats : ℕ) 
  (sold_cats : ℕ) 
  (remaining_cats : ℕ) :
  house_cats = 49 →
  sold_cats = 19 →
  remaining_cats = 45 →
  total_cats = siamese_cats + house_cats →
  total_cats = remaining_cats + sold_cats →
  siamese_cats = 15 := by
sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l2055_205584


namespace NUMINAMATH_CALUDE_compound_interest_years_l2055_205541

/-- Compound interest calculation --/
theorem compound_interest_years (P : ℝ) (r : ℝ) (CI : ℝ) (n : ℕ) : 
  P > 0 → r > 0 → CI > 0 → n > 0 →
  let A := P + CI
  let t := Real.log (A / P) / Real.log (1 + r / n)
  P = 1200 → r = 0.20 → n = 1 → CI = 873.60 →
  ⌈t⌉ = 3 := by sorry

#check compound_interest_years

end NUMINAMATH_CALUDE_compound_interest_years_l2055_205541


namespace NUMINAMATH_CALUDE_divisors_of_pq_divisors_of_p2q_divisors_of_p2q2_divisors_of_pmqn_l2055_205523

-- Define p and q as distinct prime numbers
variable (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (h : p ≠ q)

-- Define m and n as natural numbers
variable (m n : ℕ)

-- Function to count divisors
noncomputable def countDivisors (n : ℕ) : ℕ := (Nat.divisors n).card

-- Theorems to prove
theorem divisors_of_pq : countDivisors (p * q) = 4 := by sorry

theorem divisors_of_p2q : countDivisors (p^2 * q) = 6 := by sorry

theorem divisors_of_p2q2 : countDivisors (p^2 * q^2) = 9 := by sorry

theorem divisors_of_pmqn : countDivisors (p^m * q^n) = (m + 1) * (n + 1) := by sorry

end NUMINAMATH_CALUDE_divisors_of_pq_divisors_of_p2q_divisors_of_p2q2_divisors_of_pmqn_l2055_205523


namespace NUMINAMATH_CALUDE_expression_value_l2055_205524

theorem expression_value (x : ℝ) (h : x = 5) : 3 * x + 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2055_205524


namespace NUMINAMATH_CALUDE_storage_house_blocks_l2055_205575

/-- Represents the dimensions of a rectangular prism -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular prism -/
def volume (d : Dimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the specifications of the storage house -/
structure StorageHouse where
  outerDimensions : Dimensions
  wallThickness : ℕ

/-- Calculates the inner dimensions of the storage house -/
def innerDimensions (s : StorageHouse) : Dimensions :=
  { length := s.outerDimensions.length - 2 * s.wallThickness,
    width := s.outerDimensions.width - 2 * s.wallThickness,
    height := s.outerDimensions.height - s.wallThickness }

/-- Calculates the number of blocks needed for the storage house -/
def blocksNeeded (s : StorageHouse) : ℕ :=
  volume s.outerDimensions - volume (innerDimensions s)

theorem storage_house_blocks :
  let s : StorageHouse :=
    { outerDimensions := { length := 15, width := 12, height := 8 },
      wallThickness := 2 }
  blocksNeeded s = 912 := by sorry

end NUMINAMATH_CALUDE_storage_house_blocks_l2055_205575


namespace NUMINAMATH_CALUDE_shopping_cost_l2055_205582

/-- The cost of items in a shopping mall with discount --/
theorem shopping_cost (tshirt_cost pants_cost shoe_cost : ℝ) 
  (h1 : tshirt_cost = 20)
  (h2 : pants_cost = 80)
  (h3 : (4 * tshirt_cost + 3 * pants_cost + 2 * shoe_cost) * 0.9 = 558) :
  shoe_cost = 150 := by
sorry

end NUMINAMATH_CALUDE_shopping_cost_l2055_205582


namespace NUMINAMATH_CALUDE_return_time_is_11pm_l2055_205526

structure Journey where
  startTime : Nat
  totalDistance : Nat
  speedLevel : Nat
  speedUphill : Nat
  speedDownhill : Nat
  terrainDistribution : Nat

def calculateReturnTime (j : Journey) : Nat :=
  let oneWayTime := j.terrainDistribution / j.speedLevel +
                    j.terrainDistribution / j.speedUphill +
                    j.terrainDistribution / j.speedDownhill +
                    j.terrainDistribution / j.speedLevel
  let totalTime := 2 * oneWayTime
  j.startTime + totalTime

theorem return_time_is_11pm (j : Journey) 
  (h1 : j.startTime = 15) -- 3 pm in 24-hour format
  (h2 : j.totalDistance = 12)
  (h3 : j.speedLevel = 4)
  (h4 : j.speedUphill = 3)
  (h5 : j.speedDownhill = 6)
  (h6 : j.terrainDistribution = 4) -- Assumption of equal distribution
  : calculateReturnTime j = 23 := by -- 11 pm in 24-hour format
  sorry


end NUMINAMATH_CALUDE_return_time_is_11pm_l2055_205526


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_210_l2055_205528

theorem greatest_prime_factor_of_210 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 210 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 210 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_210_l2055_205528


namespace NUMINAMATH_CALUDE_equation_solution_l2055_205525

theorem equation_solution (x y z : ℝ) :
  (x - y - 3)^2 + (y - z)^2 + (x - z)^2 = 3 →
  x = z + 1 ∧ y = z - 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2055_205525


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2055_205579

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (sum_eq_one : a + b + c + d = 1) :
  (b * c * d / (1 - a)^2) + (a * c * d / (1 - b)^2) + 
  (a * b * d / (1 - c)^2) + (a * b * c / (1 - d)^2) ≤ 1/9 ∧
  ((b * c * d / (1 - a)^2) + (a * c * d / (1 - b)^2) + 
   (a * b * d / (1 - c)^2) + (a * b * c / (1 - d)^2) = 1/9 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2055_205579


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2055_205560

theorem fraction_multiplication : (2 : ℚ) / 5 * (5 : ℚ) / 9 * (1 : ℚ) / 2 = (1 : ℚ) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2055_205560


namespace NUMINAMATH_CALUDE_decreasing_function_k_bound_l2055_205540

/-- The function f(x) = kx³ + 3(k-1)x² - k² + 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^3 + 3 * (k - 1) * x^2 - k^2 + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (k : ℝ) (x : ℝ) : ℝ := 3 * k * x^2 + 6 * (k - 1) * x

theorem decreasing_function_k_bound :
  ∀ k : ℝ, (∀ x ∈ Set.Ioo 0 4, f_deriv k x ≤ 0) → k ≤ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_k_bound_l2055_205540


namespace NUMINAMATH_CALUDE_marius_darius_score_difference_l2055_205519

/-- The difference in scores between Marius and Darius in a table football game -/
theorem marius_darius_score_difference :
  ∀ (marius_score darius_score matt_score : ℕ),
    darius_score = 10 →
    matt_score = darius_score + 5 →
    marius_score + darius_score + matt_score = 38 →
    marius_score - darius_score = 3 := by
  sorry

end NUMINAMATH_CALUDE_marius_darius_score_difference_l2055_205519


namespace NUMINAMATH_CALUDE_max_displayed_games_l2055_205578

/-- Represents the number of games that can be displayed for each genre -/
structure DisplayedGames where
  action : ℕ
  adventure : ℕ
  simulation : ℕ

/-- Represents the shelf capacity for each genre -/
structure ShelfCapacity where
  action : ℕ
  adventure : ℕ
  simulation : ℕ

/-- Represents the total number of games for each genre -/
structure TotalGames where
  action : ℕ
  adventure : ℕ
  simulation : ℕ

def store_promotion : ℕ := 10

def total_games : TotalGames :=
  { action := 73, adventure := 51, simulation := 39 }

def shelf_capacity : ShelfCapacity :=
  { action := 60, adventure := 45, simulation := 35 }

def displayed_games (t : TotalGames) (s : ShelfCapacity) : DisplayedGames :=
  { action := min (t.action - store_promotion) s.action + store_promotion,
    adventure := min (t.adventure - store_promotion) s.adventure + store_promotion,
    simulation := min (t.simulation - store_promotion) s.simulation + store_promotion }

def total_displayed (d : DisplayedGames) : ℕ :=
  d.action + d.adventure + d.simulation

theorem max_displayed_games :
  total_displayed (displayed_games total_games shelf_capacity) = 160 :=
by sorry

end NUMINAMATH_CALUDE_max_displayed_games_l2055_205578


namespace NUMINAMATH_CALUDE_lcm_and_prime_factorization_l2055_205530

theorem lcm_and_prime_factorization :
  let a := 48
  let b := 180
  let c := 250
  let lcm_result := Nat.lcm (Nat.lcm a b) c
  lcm_result = 18000 ∧ 
  18000 = 2^4 * 3^2 * 5^3 := by
sorry

end NUMINAMATH_CALUDE_lcm_and_prime_factorization_l2055_205530


namespace NUMINAMATH_CALUDE_sum_of_abs_coeffs_of_2x_minus_1_to_6th_l2055_205564

theorem sum_of_abs_coeffs_of_2x_minus_1_to_6th (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) :
  (∀ x, (2*x - 1)^6 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 729) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_abs_coeffs_of_2x_minus_1_to_6th_l2055_205564


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2055_205521

/-- The radius of a cylinder inscribed in a cone with specific dimensions -/
theorem inscribed_cylinder_radius (cylinder_height : ℝ) (cylinder_radius : ℝ) 
  (cone_diameter : ℝ) (cone_altitude : ℝ) : 
  cylinder_height = 2 * cylinder_radius →
  cone_diameter = 8 →
  cone_altitude = 10 →
  cylinder_radius = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2055_205521


namespace NUMINAMATH_CALUDE_rational_equation_implication_l2055_205577

theorem rational_equation_implication (a b : ℚ) 
  (h : Real.sqrt (a + 4) + (b - 2)^2 = 0) : a - b = -6 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_implication_l2055_205577


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2055_205503

theorem polynomial_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x*y + x*z + y*z) := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2055_205503


namespace NUMINAMATH_CALUDE_projection_implies_coplanar_and_parallel_l2055_205599

-- Define a type for 3D points
def Point3D := ℝ × ℝ × ℝ

-- Define a type for 2D points (projections)
def Point2D := ℝ × ℝ

-- Define a plane in 3D space
structure Plane where
  normal : ℝ × ℝ × ℝ
  d : ℝ

-- Define a projection function
def project (p : Point3D) (plane : Plane) : Point2D :=
  sorry

-- Define a predicate for points being on a line
def onLine (points : List Point2D) : Prop :=
  sorry

-- Define a predicate for points being coplanar
def coplanar (points : List Point3D) : Prop :=
  sorry

-- Define a predicate for points being parallel
def parallel (points : List Point3D) : Prop :=
  sorry

-- The main theorem
theorem projection_implies_coplanar_and_parallel 
  (points : List Point3D) (plane : Plane) :
  onLine (points.map (λ p => project p plane)) →
  coplanar points ∧ parallel points :=
sorry

end NUMINAMATH_CALUDE_projection_implies_coplanar_and_parallel_l2055_205599


namespace NUMINAMATH_CALUDE_nathaniel_best_friends_l2055_205585

def initial_tickets : ℕ := 11
def remaining_tickets : ℕ := 3
def tickets_per_friend : ℕ := 2

def number_of_friends : ℕ := (initial_tickets - remaining_tickets) / tickets_per_friend

theorem nathaniel_best_friends : number_of_friends = 4 := by
  sorry

end NUMINAMATH_CALUDE_nathaniel_best_friends_l2055_205585


namespace NUMINAMATH_CALUDE_room_width_calculation_l2055_205583

/-- Given a room with known length, paving cost per square meter, and total paving cost,
    calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 5.5 →
  cost_per_sqm = 800 →
  total_cost = 16500 →
  (total_cost / cost_per_sqm) / length = 3.75 := by
  sorry

#check room_width_calculation

end NUMINAMATH_CALUDE_room_width_calculation_l2055_205583


namespace NUMINAMATH_CALUDE_remainder_problem_l2055_205591

theorem remainder_problem (n : ℕ) : 
  n % 6 = 4 ∧ n / 6 = 124 → (n + 24) % 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2055_205591


namespace NUMINAMATH_CALUDE_line_l_general_form_l2055_205593

/-- A line passing through point A(-2, 2) with the same y-intercept as y = x + 6 -/
def line_l (x y : ℝ) : Prop :=
  ∃ (m b : ℝ), y = m * x + b ∧ 2 = m * (-2) + b ∧ b = 6

/-- The general form equation of line l -/
def general_form (x y : ℝ) : Prop := 2 * x - y + 6 = 0

/-- Theorem stating that the general form equation of line l is 2x - y + 6 = 0 -/
theorem line_l_general_form : 
  ∀ x y : ℝ, line_l x y ↔ general_form x y :=
sorry

end NUMINAMATH_CALUDE_line_l_general_form_l2055_205593


namespace NUMINAMATH_CALUDE_boy_scouts_permission_slips_l2055_205563

theorem boy_scouts_permission_slips 
  (total_scouts : ℕ) 
  (total_with_slips : ℝ) 
  (total_boys : ℝ) 
  (girl_scouts_with_slips : ℝ) 
  (h1 : total_with_slips = 0.8 * total_scouts)
  (h2 : total_boys = 0.4 * total_scouts)
  (h3 : girl_scouts_with_slips = 0.8333 * (total_scouts - total_boys)) :
  (total_with_slips - girl_scouts_with_slips) / total_boys = 0.75 := by
sorry

end NUMINAMATH_CALUDE_boy_scouts_permission_slips_l2055_205563


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l2055_205587

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, x^2 + x = y^4 + y^3 + y^2 + y ↔
    (x = 0 ∧ y = -1) ∨
    (x = -1 ∧ y = -1) ∨
    (x = 0 ∧ y = 0) ∨
    (x = -1 ∧ y = 0) ∨
    (x = 5 ∧ y = 2) ∨
    (x = -6 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l2055_205587


namespace NUMINAMATH_CALUDE_min_honey_purchase_l2055_205520

def is_valid_purchase (o h : ℕ) : Prop :=
  o ≥ 7 + h / 2 ∧ 
  o ≤ 3 * h ∧ 
  2 * o + 3 * h ≤ 36

theorem min_honey_purchase : 
  (∃ (o h : ℕ), is_valid_purchase o h) ∧ 
  (∀ (o h : ℕ), is_valid_purchase o h → h ≥ 4) ∧
  (∃ (o : ℕ), is_valid_purchase o 4) :=
sorry

end NUMINAMATH_CALUDE_min_honey_purchase_l2055_205520


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l2055_205546

theorem largest_multiple_of_8_under_100 : 
  ∃ n : ℕ, n * 8 = 96 ∧ 
  96 < 100 ∧ 
  ∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l2055_205546


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2055_205556

/-- Given a circle with equation x² + y² - 4x + 2y + 2 = 0, 
    its center is at (2, -1) and its radius is √3. -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (2, -1) ∧ 
    radius = Real.sqrt 3 ∧
    ∀ (x y : ℝ), x^2 + y^2 - 4*x + 2*y + 2 = 0 ↔ 
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l2055_205556


namespace NUMINAMATH_CALUDE_triangle_side_values_l2055_205529

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_values :
  ∀ y : ℕ+,
  (triangle_exists 8 11 (y.val ^ 2)) ↔ (y = 2 ∨ y = 3 ∨ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l2055_205529


namespace NUMINAMATH_CALUDE_multiply_specific_numbers_l2055_205522

theorem multiply_specific_numbers : 469160 * 9999 = 4691183840 := by
  sorry

end NUMINAMATH_CALUDE_multiply_specific_numbers_l2055_205522


namespace NUMINAMATH_CALUDE_largest_exponent_inequality_l2055_205569

theorem largest_exponent_inequality (n : ℕ) : 64^8 > 4^n ↔ n ≤ 23 := by
  sorry

end NUMINAMATH_CALUDE_largest_exponent_inequality_l2055_205569


namespace NUMINAMATH_CALUDE_choose_two_from_four_l2055_205581

theorem choose_two_from_four : Nat.choose 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_four_l2055_205581


namespace NUMINAMATH_CALUDE_binomial_product_factorial_l2055_205505

theorem binomial_product_factorial (n : ℕ) : 
  (Nat.choose (n + 2) n) * n.factorial = ((n + 2) * (n + 1) * n.factorial) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_factorial_l2055_205505


namespace NUMINAMATH_CALUDE_distance_eq_speed_times_time_l2055_205555

/-- The distance between Martin's house and Lawrence's house -/
def distance : ℝ := 12

/-- The time Martin spent walking -/
def time : ℝ := 6

/-- Martin's walking speed -/
def speed : ℝ := 2

/-- Theorem stating that the distance is equal to speed multiplied by time -/
theorem distance_eq_speed_times_time : distance = speed * time := by
  sorry

end NUMINAMATH_CALUDE_distance_eq_speed_times_time_l2055_205555


namespace NUMINAMATH_CALUDE_choir_composition_l2055_205592

theorem choir_composition (initial_total : ℕ) (initial_blonde : ℕ) (added_blonde : ℕ) :
  initial_total = 80 →
  initial_blonde = 30 →
  added_blonde = 10 →
  initial_total - initial_blonde + added_blonde = 50 :=
by sorry

end NUMINAMATH_CALUDE_choir_composition_l2055_205592


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2055_205576

theorem complex_fraction_simplification :
  (Complex.I + 3) / (Complex.I + 1) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2055_205576


namespace NUMINAMATH_CALUDE_storeroom_contains_912_blocks_l2055_205571

/-- Calculates the number of blocks in a rectangular storeroom with given dimensions and wall thickness -/
def storeroom_blocks (length width height wall_thickness : ℕ) : ℕ :=
  let total_volume := length * width * height
  let internal_length := length - 2 * wall_thickness
  let internal_width := width - 2 * wall_thickness
  let internal_height := height - wall_thickness
  let internal_volume := internal_length * internal_width * internal_height
  total_volume - internal_volume

/-- Theorem stating that a storeroom with given dimensions contains 912 blocks -/
theorem storeroom_contains_912_blocks :
  storeroom_blocks 15 12 8 2 = 912 := by
  sorry

#eval storeroom_blocks 15 12 8 2

end NUMINAMATH_CALUDE_storeroom_contains_912_blocks_l2055_205571


namespace NUMINAMATH_CALUDE_inequality_solution_l2055_205534

/-- A quadratic function f(x) = ax^2 - bx + c where f(x) > 0 for x in (1, 3) -/
def f (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 - b * x + c

/-- The solution set of f(x) > 0 is (1, 3) -/
def f_positive_interval (a b c : ℝ) : Prop :=
  ∀ x, f a b c x > 0 ↔ 1 < x ∧ x < 3

theorem inequality_solution (a b c : ℝ) (h : f_positive_interval a b c) :
  ∀ t : ℝ, f a b c (|t| + 8) < f a b c (2 + t^2) ↔ -3 < t ∧ t < 3 ∧ t ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2055_205534
