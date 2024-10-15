import Mathlib

namespace NUMINAMATH_CALUDE_inequality_equivalence_l1035_103562

theorem inequality_equivalence (x : ℝ) : 
  (2 * x + 3) / (x^2 - 2 * x + 4) > (4 * x + 5) / (2 * x^2 + 5 * x + 7) ↔ 
  x > (-23 - Real.sqrt 453) / 38 ∧ x < (-23 + Real.sqrt 453) / 38 :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1035_103562


namespace NUMINAMATH_CALUDE_dividend_percentage_l1035_103588

theorem dividend_percentage 
  (face_value : ℝ) 
  (purchase_price : ℝ) 
  (return_on_investment : ℝ) 
  (h1 : face_value = 50) 
  (h2 : purchase_price = 31) 
  (h3 : return_on_investment = 0.25) : 
  (return_on_investment * purchase_price) / face_value * 100 = 15.5 := by
sorry

end NUMINAMATH_CALUDE_dividend_percentage_l1035_103588


namespace NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l1035_103527

/-- Given an integer N represented as 777 in base b, 
    18 is the smallest b for which N is a fourth power -/
theorem smallest_base_for_fourth_power (N : ℤ) (b : ℤ) : 
  N = 7 * b^2 + 7 * b + 7 → -- N's representation in base b is 777
  (∃ (a : ℤ), N = a^4) →    -- N is a fourth power
  b ≥ 18 :=                 -- 18 is the smallest such b
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l1035_103527


namespace NUMINAMATH_CALUDE_total_flower_cost_l1035_103500

-- Define the promenade perimeter in meters
def promenade_perimeter : ℕ := 1500

-- Define the planting interval in meters
def planting_interval : ℕ := 30

-- Define the cost per flower in won
def cost_per_flower : ℕ := 5000

-- Theorem to prove
theorem total_flower_cost : 
  (promenade_perimeter / planting_interval) * cost_per_flower = 250000 := by
sorry

end NUMINAMATH_CALUDE_total_flower_cost_l1035_103500


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l1035_103525

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 36) : x + y ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l1035_103525


namespace NUMINAMATH_CALUDE_imon_disentanglement_l1035_103584

-- Define a graph structure to represent imons and their entanglements
structure ImonGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)

-- Define the two operations
def destroyOddDegreeImon (g : ImonGraph) (v : Nat) : ImonGraph :=
  sorry

def doubleImonFamily (g : ImonGraph) : ImonGraph :=
  sorry

-- Define a predicate to check if a graph has no entanglements
def noEntanglements (g : ImonGraph) : Prop :=
  ∀ v ∈ g.vertices, ∀ w ∈ g.vertices, v ≠ w → (v, w) ∉ g.edges

-- Main theorem
theorem imon_disentanglement (g : ImonGraph) :
  ∃ (ops : List (ImonGraph → ImonGraph)), noEntanglements ((ops.foldl (· ∘ ·) id) g) :=
  sorry

end NUMINAMATH_CALUDE_imon_disentanglement_l1035_103584


namespace NUMINAMATH_CALUDE_fraction_equals_sixtythree_l1035_103539

theorem fraction_equals_sixtythree : (2200 - 2089)^2 / 196 = 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_sixtythree_l1035_103539


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l1035_103561

theorem complex_fraction_equals_i : (Complex.I + 1) / (1 - Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l1035_103561


namespace NUMINAMATH_CALUDE_simplify_expression_l1035_103589

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2) :
  (x^2 - x) / (x^2 - 2*x + 1) = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1035_103589


namespace NUMINAMATH_CALUDE_function_equality_implies_m_range_l1035_103595

theorem function_equality_implies_m_range :
  ∀ (m : ℝ), m > 0 →
  (∃ (x₁ : ℝ), x₁ ∈ Set.Icc 0 3 ∧
    (∀ (x₂ : ℝ), x₂ ∈ Set.Icc 0 3 →
      x₁^2 - 4*x₁ + 3 = m*(x₂ - 1) + 2)) →
  m ∈ Set.Ioo 0 (1/2) := by
sorry

end NUMINAMATH_CALUDE_function_equality_implies_m_range_l1035_103595


namespace NUMINAMATH_CALUDE_intersection_and_length_l1035_103597

-- Define the coordinate system
variable (O : ℝ × ℝ)
variable (A : ℝ × ℝ)
variable (B : ℝ × ℝ)

-- Define lines l₁ and l₂
def l₁ (p : ℝ × ℝ) : Prop := p.1 + p.2 = 4
def l₂ (p : ℝ × ℝ) : Prop := p.2 = 2 * p.1

-- Define the conditions
axiom O_origin : O = (0, 0)
axiom A_on_l₁ : l₁ A
axiom A_on_l₂ : l₂ A
axiom B_on_l₁ : l₁ B
axiom OA_perp_OB : (A.1 * B.1 + A.2 * B.2) = 0

-- State the theorem
theorem intersection_and_length :
  A = (4/3, 8/3) ∧
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 20 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_and_length_l1035_103597


namespace NUMINAMATH_CALUDE_harpers_daughter_has_four_teachers_l1035_103538

/-- Represents the problem of finding Harper's daughter's teachers -/
def harpers_daughter_teachers (total_spent : ℕ) (gift_cost : ℕ) (sons_teachers : ℕ) : ℕ :=
  total_spent / gift_cost - sons_teachers

/-- Theorem stating that Harper's daughter has 4 teachers -/
theorem harpers_daughter_has_four_teachers :
  harpers_daughter_teachers 70 10 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_harpers_daughter_has_four_teachers_l1035_103538


namespace NUMINAMATH_CALUDE_kingfisher_to_warbler_ratio_l1035_103504

/-- Represents the composition of bird species in the Goshawk-Eurasian Nature Reserve -/
structure BirdPopulation where
  hawks : ℝ
  paddyfieldWarblers : ℝ
  kingfishers : ℝ
  others : ℝ

/-- The conditions of the bird population in the nature reserve -/
def validBirdPopulation (bp : BirdPopulation) : Prop :=
  bp.hawks = 0.3 ∧
  bp.paddyfieldWarblers = 0.4 * (1 - bp.hawks) ∧
  bp.others = 0.35 ∧
  bp.hawks + bp.paddyfieldWarblers + bp.kingfishers + bp.others = 1

/-- The theorem stating the relationship between kingfishers and paddyfield-warblers -/
theorem kingfisher_to_warbler_ratio (bp : BirdPopulation) 
  (h : validBirdPopulation bp) : 
  bp.kingfishers / bp.paddyfieldWarblers = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_kingfisher_to_warbler_ratio_l1035_103504


namespace NUMINAMATH_CALUDE_pat_kate_ratio_l1035_103582

/-- Represents the hours charged by each person -/
structure ProjectHours where
  pat : ℝ
  kate : ℝ
  mark : ℝ

/-- Defines the conditions of the problem -/
def satisfiesConditions (h : ProjectHours) : Prop :=
  h.pat + h.kate + h.mark = 135 ∧
  ∃ r : ℝ, h.pat = r * h.kate ∧
  h.pat = (1/3) * h.mark ∧
  h.mark = h.kate + 75

/-- The main theorem to prove -/
theorem pat_kate_ratio (h : ProjectHours) 
  (hcond : satisfiesConditions h) : h.pat / h.kate = 2 := by
  sorry

end NUMINAMATH_CALUDE_pat_kate_ratio_l1035_103582


namespace NUMINAMATH_CALUDE_periodic_double_period_l1035_103548

open Real

/-- A function f is a-periodic if it satisfies the given functional equation. -/
def IsPeriodic (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a > 0 ∧ ∀ x, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)

/-- If a function is a-periodic, then it is also 2a-periodic. -/
theorem periodic_double_period (f : ℝ → ℝ) (a : ℝ) (h : IsPeriodic f a) :
  ∀ x, f (x + 2*a) = f x := by
  sorry

end NUMINAMATH_CALUDE_periodic_double_period_l1035_103548


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1035_103587

/-- The lateral surface area of a cone with base radius 1 and height √3 is 2π -/
theorem cone_lateral_surface_area : 
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (h^2 + r^2)
  let lateral_area : ℝ := π * r * l
  lateral_area = 2 * π := by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1035_103587


namespace NUMINAMATH_CALUDE_restaurant_hiring_l1035_103545

/-- Given a restaurant with cooks and waiters, prove the number of newly hired waiters. -/
theorem restaurant_hiring (initial_cooks initial_waiters new_waiters : ℕ) : 
  initial_cooks * 11 = initial_waiters * 3 →  -- Initial ratio of cooks to waiters is 3:11
  initial_cooks * 5 = (initial_waiters + new_waiters) * 1 →  -- New ratio is 1:5
  initial_cooks = 9 →  -- There are 9 cooks
  new_waiters = 12 :=  -- Prove that 12 waiters were hired
by sorry

end NUMINAMATH_CALUDE_restaurant_hiring_l1035_103545


namespace NUMINAMATH_CALUDE_complex_number_problem_l1035_103598

theorem complex_number_problem (a : ℝ) : 
  let z : ℂ := (a / Complex.I) + ((1 - Complex.I) / 2) * Complex.I
  (z.re = 0 ∨ z.im = 0) ∧ (z.re + z.im = 0) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1035_103598


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_third_l1035_103590

theorem opposite_of_negative_one_third :
  -(-(1/3 : ℚ)) = 1/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_third_l1035_103590


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1035_103536

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  x^2 + 2*x*y + 3*y^2 ≤ 20 + 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1035_103536


namespace NUMINAMATH_CALUDE_expected_value_of_sum_l1035_103581

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def combinations : Finset (Finset ℕ) :=
  marbles.powerset.filter (λ s => s.card = 3)

def sum_of_combination (c : Finset ℕ) : ℕ := c.sum id

def total_sum : ℕ := combinations.sum sum_of_combination

def num_combinations : ℕ := combinations.card

theorem expected_value_of_sum :
  (total_sum : ℚ) / num_combinations = 21/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_sum_l1035_103581


namespace NUMINAMATH_CALUDE_fraction_simplification_l1035_103518

theorem fraction_simplification :
  (1/2 + 1/5) / (3/7 - 1/14) = 49/25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1035_103518


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1035_103593

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = (11 + Real.sqrt 13) / 6 ∧
  x₂ = (11 - Real.sqrt 13) / 6 ∧
  (x₁ - 2) * (3 * x₁ - 5) = 1 ∧
  (x₂ - 2) * (3 * x₂ - 5) = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1035_103593


namespace NUMINAMATH_CALUDE_dinner_lunch_cake_difference_l1035_103571

theorem dinner_lunch_cake_difference : 
  let lunch_cakes : ℕ := 6
  let dinner_cakes : ℕ := 9
  dinner_cakes - lunch_cakes = 3 := by sorry

end NUMINAMATH_CALUDE_dinner_lunch_cake_difference_l1035_103571


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l1035_103592

/-- Represents a point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a point lies on a line segment between two other points -/
def lies_on (P Q R : Point) : Prop := sorry

/-- Checks if two line segments intersect -/
def intersect (P Q R S : Point) : Prop := sorry

/-- Represents the ratio of distances between points -/
def distance_ratio (P Q R S : Point) : ℚ := sorry

theorem triangle_ratio_theorem (ABC : Triangle) (D E T : Point) :
  lies_on D ABC.B ABC.C →
  lies_on E ABC.A ABC.B →
  intersect ABC.A D ABC.B E →
  (∃ (t : Point), intersect ABC.A D ABC.B E ∧ t = T) →
  distance_ratio ABC.A T T D = 2 →
  distance_ratio ABC.B T T E = 3 →
  distance_ratio ABC.C D D ABC.B = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l1035_103592


namespace NUMINAMATH_CALUDE_nina_jerome_age_ratio_l1035_103580

-- Define the ages as natural numbers
def leonard : ℕ := 6
def nina : ℕ := leonard + 4
def jerome : ℕ := 36 - nina - leonard

-- Theorem statement
theorem nina_jerome_age_ratio :
  nina * 2 = jerome :=
sorry

end NUMINAMATH_CALUDE_nina_jerome_age_ratio_l1035_103580


namespace NUMINAMATH_CALUDE_solution_set_equality_l1035_103533

theorem solution_set_equality : 
  {x : ℝ | (x + 3) * (1 - x) ≥ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l1035_103533


namespace NUMINAMATH_CALUDE_meet_once_l1035_103522

/-- Represents the movement of Michael and the garbage truck --/
structure Movement where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the garbage truck --/
def number_of_meetings (m : Movement) : ℕ :=
  sorry

/-- The theorem stating that Michael and the garbage truck meet exactly once --/
theorem meet_once (m : Movement) 
  (h1 : m.michael_speed = 3)
  (h2 : m.truck_speed = 6)
  (h3 : m.pail_distance = 100)
  (h4 : m.truck_stop_time = 20)
  (h5 : m.initial_distance = 100) : 
  number_of_meetings m = 1 :=
sorry

end NUMINAMATH_CALUDE_meet_once_l1035_103522


namespace NUMINAMATH_CALUDE_application_outcomes_count_l1035_103596

/-- The number of colleges available for applications -/
def num_colleges : ℕ := 3

/-- The number of students applying to colleges -/
def num_students : ℕ := 3

/-- The total number of possible application outcomes when three students apply to three colleges,
    with the condition that the first two students must apply to different colleges -/
def total_outcomes : ℕ := num_colleges * (num_colleges - 1) * num_colleges

theorem application_outcomes_count : total_outcomes = 18 := by
  sorry

end NUMINAMATH_CALUDE_application_outcomes_count_l1035_103596


namespace NUMINAMATH_CALUDE_part_one_part_two_l1035_103523

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| ≤ 3
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Part I
theorem part_one :
  let S := {x : ℝ | (p x ∨ q x 2) ∧ ¬(p x ∧ q x 2)}
  S = {x : ℝ | -4 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3} :=
sorry

-- Part II
theorem part_two :
  let T := {m : ℝ | m > 0 ∧ {x : ℝ | p x} ⊃ {x : ℝ | q x m} ∧ {x : ℝ | p x} ≠ {x : ℝ | q x m}}
  T = {m : ℝ | 0 < m ∧ m ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1035_103523


namespace NUMINAMATH_CALUDE_perfect_square_swap_l1035_103532

theorem perfect_square_swap (a b : ℕ) (ha : a > b) (hb : b > 0) 
  (hA : ∃ k : ℕ, a^2 + 4*b + 1 = k^2) 
  (hB : ∃ m : ℕ, b^2 + 4*a + 1 = m^2) : 
  a = 8 ∧ b = 4 := by sorry

end NUMINAMATH_CALUDE_perfect_square_swap_l1035_103532


namespace NUMINAMATH_CALUDE_max_police_officers_l1035_103540

theorem max_police_officers (n : ℕ) (h : n = 8) : 
  (n * (n - 1)) / 2 = 28 :=
sorry

end NUMINAMATH_CALUDE_max_police_officers_l1035_103540


namespace NUMINAMATH_CALUDE_robin_cupcakes_l1035_103509

/-- The number of cupcakes with chocolate sauce Robin ate -/
def chocolate_cupcakes : ℕ := sorry

/-- The number of cupcakes with buttercream frosting Robin ate -/
def buttercream_cupcakes : ℕ := sorry

/-- The total number of cupcakes Robin ate -/
def total_cupcakes : ℕ := 12

theorem robin_cupcakes :
  chocolate_cupcakes + buttercream_cupcakes = total_cupcakes ∧
  buttercream_cupcakes = 2 * chocolate_cupcakes →
  chocolate_cupcakes = 4 :=
by sorry

end NUMINAMATH_CALUDE_robin_cupcakes_l1035_103509


namespace NUMINAMATH_CALUDE_expression_evaluation_l1035_103549

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := 5
  let z : ℚ := 3
  (3 * x^5 + 4 * y^3 + z^2) / 7 = 605 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1035_103549


namespace NUMINAMATH_CALUDE_min_value_expression_l1035_103517

theorem min_value_expression (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  m + (n^2 - m*n + 4) / (m - n) ≥ 4 ∧
  (m + (n^2 - m*n + 4) / (m - n) = 4 ↔ m - n = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1035_103517


namespace NUMINAMATH_CALUDE_fruit_ratio_l1035_103560

def total_fruit : ℕ := 13
def remaining_fruit : ℕ := 9

def fruit_fell_out : ℕ := total_fruit - remaining_fruit

theorem fruit_ratio : 
  (fruit_fell_out : ℚ) / total_fruit = 4 / 13 := by sorry

end NUMINAMATH_CALUDE_fruit_ratio_l1035_103560


namespace NUMINAMATH_CALUDE_pages_difference_l1035_103519

/-- The number of pages Juwella read over four nights -/
def total_pages : ℕ := 100

/-- The number of pages Juwella will read tonight -/
def pages_tonight : ℕ := 20

/-- The number of pages Juwella read three nights ago -/
def pages_three_nights_ago : ℕ := 15

/-- The number of pages Juwella read two nights ago -/
def pages_two_nights_ago : ℕ := 2 * pages_three_nights_ago

/-- The number of pages Juwella read last night -/
def pages_last_night : ℕ := total_pages - pages_tonight - pages_three_nights_ago - pages_two_nights_ago

theorem pages_difference : pages_last_night - pages_two_nights_ago = 5 := by
  sorry

end NUMINAMATH_CALUDE_pages_difference_l1035_103519


namespace NUMINAMATH_CALUDE_triangle_inequality_condition_l1035_103529

theorem triangle_inequality_condition (k l : ℝ) :
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b →
    k * a^2 + l * b^2 > c^2) ↔
  k * l ≥ k + l ∧ k > 1 ∧ l > 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_condition_l1035_103529


namespace NUMINAMATH_CALUDE_factor_implies_sum_l1035_103572

theorem factor_implies_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 - 2*X + 5) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) →
  P + Q = 31 :=
by sorry

end NUMINAMATH_CALUDE_factor_implies_sum_l1035_103572


namespace NUMINAMATH_CALUDE_unique_sums_count_l1035_103514

def bag_C : Finset ℕ := {1, 2, 3, 4}
def bag_D : Finset ℕ := {3, 5, 7}

theorem unique_sums_count : 
  Finset.card ((bag_C.product bag_D).image (fun (p : ℕ × ℕ) => p.1 + p.2)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_sums_count_l1035_103514


namespace NUMINAMATH_CALUDE_max_value_condition_l1035_103599

/-- The function f(x) = -x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x

/-- The maximum value of f(x) on the interval [0, 1] is 2 iff a = -2√2 or a = 3 -/
theorem max_value_condition (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧ (∃ x ∈ Set.Icc 0 1, f a x = 2) ↔ 
  (a = -2 * Real.sqrt 2 ∨ a = 3) := by
sorry

end NUMINAMATH_CALUDE_max_value_condition_l1035_103599


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_exists_185_solution_is_185_l1035_103516

theorem greatest_integer_with_gcf_five (n : ℕ) : n < 200 ∧ Nat.gcd n 30 = 5 → n ≤ 185 :=
by
  sorry

theorem exists_185 : 185 < 200 ∧ Nat.gcd 185 30 = 5 :=
by
  sorry

theorem solution_is_185 : ∃ (n : ℕ), n = 185 ∧ n < 200 ∧ Nat.gcd n 30 = 5 ∧
  ∀ (m : ℕ), m < 200 ∧ Nat.gcd m 30 = 5 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_exists_185_solution_is_185_l1035_103516


namespace NUMINAMATH_CALUDE_reflection_matrix_iff_l1035_103578

def reflection_matrix (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, b],
    ![-3/4, 1/4]]

theorem reflection_matrix_iff (a b : ℚ) :
  (reflection_matrix a b)^2 = 1 ↔ a = -1/4 ∧ b = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_reflection_matrix_iff_l1035_103578


namespace NUMINAMATH_CALUDE_trajectory_equation_l1035_103563

/-- Given two points A and B symmetric about the origin, and a moving point P such that 
    the product of the slopes of AP and BP is -1/3, prove that the trajectory of P 
    is described by the equation x^2 + 3y^2 = 4, where x ≠ ±1 -/
theorem trajectory_equation (A B P : ℝ × ℝ) : 
  A = (-1, 1) →
  B = (1, -1) →
  (∀ x y, P = (x, y) → x ≠ 1 ∧ x ≠ -1 →
    ((y - 1) / (x + 1)) * ((y + 1) / (x - 1)) = -1/3) →
  ∃ x y, P = (x, y) ∧ x^2 + 3*y^2 = 4 ∧ x ≠ 1 ∧ x ≠ -1 :=
by sorry


end NUMINAMATH_CALUDE_trajectory_equation_l1035_103563


namespace NUMINAMATH_CALUDE_digit_81_of_325_over_999_l1035_103553

theorem digit_81_of_325_over_999 (n : ℕ) (h : n = 81) :
  (325 : ℚ) / 999 * 10^n % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_81_of_325_over_999_l1035_103553


namespace NUMINAMATH_CALUDE_min_value_product_sum_l1035_103535

theorem min_value_product_sum (p q r s t u v w : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) 
  (ht : 0 < t) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p * q * r * s = 16) (h2 : t * u * v * w = 25) :
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 40 ∧
  ∃ (p' q' r' s' t' u' v' w' : ℝ),
    0 < p' ∧ 0 < q' ∧ 0 < r' ∧ 0 < s' ∧
    0 < t' ∧ 0 < u' ∧ 0 < v' ∧ 0 < w' ∧
    p' * q' * r' * s' = 16 ∧
    t' * u' * v' * w' = 25 ∧
    (p' * t')^2 + (q' * u')^2 + (r' * v')^2 + (s' * w')^2 = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_product_sum_l1035_103535


namespace NUMINAMATH_CALUDE_select_three_from_four_l1035_103566

theorem select_three_from_four : Nat.choose 4 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_select_three_from_four_l1035_103566


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1035_103530

def S : Set Int := {-1, -2, -3, -4, -5}

theorem max_value_of_expression (a b c d : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  (∀ (w x y z : Int), w ∈ S → x ∈ S → y ∈ S → z ∈ S →
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    (w^x + y^z : Rat) ≤ 10/9) ∧
  (∃ (w x y z : Int), w ∈ S ∧ x ∈ S ∧ y ∈ S ∧ z ∈ S ∧
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    (w^x + y^z : Rat) = 10/9) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1035_103530


namespace NUMINAMATH_CALUDE_at_least_one_negative_l1035_103569

theorem at_least_one_negative (a b c d : ℝ) 
  (sum1 : a + b = 1) 
  (sum2 : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  (a < 0) ∨ (b < 0) ∨ (c < 0) ∨ (d < 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l1035_103569


namespace NUMINAMATH_CALUDE_apple_basket_problem_l1035_103551

theorem apple_basket_problem (x : ℕ) : 
  (x * 22 = (x + 45) * (22 - 9)) → x * 22 = 1430 := by
sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l1035_103551


namespace NUMINAMATH_CALUDE_special_bin_op_property_l1035_103503

/-- A binary operation on a set S satisfying (a * b) * a = b for all a, b ∈ S -/
class SpecialBinOp (S : Type) where
  op : S → S → S
  identity : ∀ a b : S, op (op a b) a = b

/-- 
If S has a binary operation satisfying (a * b) * a = b for all a, b ∈ S,
then a * (b * a) = b for all a, b ∈ S
-/
theorem special_bin_op_property {S : Type} [SpecialBinOp S] :
  ∀ a b : S, SpecialBinOp.op a (SpecialBinOp.op b a) = b :=
by sorry

end NUMINAMATH_CALUDE_special_bin_op_property_l1035_103503


namespace NUMINAMATH_CALUDE_morning_orange_sales_l1035_103510

/-- Proves the number of oranges sold in the morning given fruit prices and sales data --/
theorem morning_orange_sales
  (apple_price : ℚ)
  (orange_price : ℚ)
  (morning_apples : ℕ)
  (afternoon_apples : ℕ)
  (afternoon_oranges : ℕ)
  (total_sales : ℚ)
  (h1 : apple_price = 3/2)
  (h2 : orange_price = 1)
  (h3 : morning_apples = 40)
  (h4 : afternoon_apples = 50)
  (h5 : afternoon_oranges = 40)
  (h6 : total_sales = 205) :
  ∃ morning_oranges : ℕ,
    morning_oranges = 30 ∧
    total_sales = apple_price * (morning_apples + afternoon_apples : ℚ) +
                  orange_price * (morning_oranges + afternoon_oranges : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_morning_orange_sales_l1035_103510


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1035_103521

/-- Given a triangle with vertices at (0, 0), (x, 3x), and (x, 0), where x > 0,
    if the area of the triangle is 81 square units, then x = 3√6. -/
theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 81 → x = 3 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1035_103521


namespace NUMINAMATH_CALUDE_lights_remaining_on_l1035_103565

def total_lights : ℕ := 2007

def lights_on_after_toggle (n : ℕ) : Prop :=
  let multiples_of_2 := (total_lights - 1) / 2
  let multiples_of_3 := total_lights / 3
  let multiples_of_5 := (total_lights - 2) / 5
  let multiples_of_6 := (total_lights - 3) / 6
  let multiples_of_10 := (total_lights - 7) / 10
  let multiples_of_15 := (total_lights - 12) / 15
  let multiples_of_30 := (total_lights - 27) / 30
  let toggled := multiples_of_2 + multiples_of_3 + multiples_of_5 - 
                 multiples_of_6 - multiples_of_10 - multiples_of_15 + 
                 multiples_of_30
  n = total_lights - toggled

theorem lights_remaining_on : lights_on_after_toggle 1004 := by sorry

end NUMINAMATH_CALUDE_lights_remaining_on_l1035_103565


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l1035_103577

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem: In a triangle ABC, if 2b * sin(2A) = a * sin(B) and c = 2b, then a/b = 2 -/
theorem triangle_ratio_theorem (t : Triangle) 
  (h1 : 2 * t.b * Real.sin (2 * t.A) = t.a * Real.sin t.B)
  (h2 : t.c = 2 * t.b) :
  t.a / t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l1035_103577


namespace NUMINAMATH_CALUDE_negative_abs_negative_three_l1035_103524

theorem negative_abs_negative_three : -|-3| = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_negative_three_l1035_103524


namespace NUMINAMATH_CALUDE_boys_equation_holds_l1035_103591

structure School where
  name : String
  total_students : ℕ

def calculate_boys (s : School) : ℚ :=
  s.total_students / (1 + s.total_students / 100)

theorem boys_equation_holds (s : School) :
  let x := calculate_boys s
  x + (x/100) * s.total_students = s.total_students :=
by sorry

def school_A : School := ⟨"A", 900⟩
def school_B : School := ⟨"B", 1200⟩
def school_C : School := ⟨"C", 1500⟩

#eval calculate_boys school_A
#eval calculate_boys school_B
#eval calculate_boys school_C

end NUMINAMATH_CALUDE_boys_equation_holds_l1035_103591


namespace NUMINAMATH_CALUDE_line_passes_through_point_l1035_103515

/-- Given a line with equation 2 + 3kx = -7y that passes through the point (-1/3, 4),
    prove that k = 30. -/
theorem line_passes_through_point (k : ℝ) : 
  (2 + 3 * k * (-1/3) = -7 * 4) → k = 30 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l1035_103515


namespace NUMINAMATH_CALUDE_largest_number_l1035_103567

theorem largest_number (a b c d e : ℝ) : 
  a = 24680 + 1 / 1357 →
  b = 24680 - 1 / 1357 →
  c = 24680 * (1 / 1357) →
  d = 24680 / (1 / 1357) →
  e = 24680.1357 →
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1035_103567


namespace NUMINAMATH_CALUDE_notebook_spending_l1035_103573

/-- Calculates the total amount spent on notebooks --/
def total_spent (total_notebooks : ℕ) (red_notebooks : ℕ) (green_notebooks : ℕ) 
  (red_price : ℕ) (green_price : ℕ) (blue_price : ℕ) : ℕ :=
  let blue_notebooks := total_notebooks - red_notebooks - green_notebooks
  red_notebooks * red_price + green_notebooks * green_price + blue_notebooks * blue_price

/-- Proves that the total amount spent on notebooks is $37 --/
theorem notebook_spending : 
  total_spent 12 3 2 4 2 3 = 37 := by
  sorry

end NUMINAMATH_CALUDE_notebook_spending_l1035_103573


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l1035_103531

theorem unique_two_digit_integer (s : ℕ) : 
  (10 ≤ s ∧ s < 100) ∧ (13 * s) % 100 = 52 ↔ s = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l1035_103531


namespace NUMINAMATH_CALUDE_m_range_l1035_103554

def f (x : ℝ) : ℝ := x^3 + x

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → f (m * Real.sin θ) + f (1 - m) > 0) →
  m < 1 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l1035_103554


namespace NUMINAMATH_CALUDE_adoption_fee_calculation_l1035_103513

theorem adoption_fee_calculation (james_payment : ℝ) (friend_percentage : ℝ) : 
  james_payment = 150 → friend_percentage = 0.25 → 
  ∃ (total_fee : ℝ), total_fee = 200 ∧ james_payment = (1 - friend_percentage) * total_fee :=
sorry

end NUMINAMATH_CALUDE_adoption_fee_calculation_l1035_103513


namespace NUMINAMATH_CALUDE_factorial_divisibility_l1035_103556

theorem factorial_divisibility (m n : ℕ) : 
  ∃ k : ℕ, (Nat.factorial (2*m) * Nat.factorial (2*n)) = 
    k * (Nat.factorial m * Nat.factorial n * Nat.factorial (m+n)) :=
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l1035_103556


namespace NUMINAMATH_CALUDE_prob_same_color_is_69_200_l1035_103564

def total_balls : ℕ := 8 + 5 + 7

def prob_blue : ℚ := 8 / total_balls
def prob_green : ℚ := 5 / total_balls
def prob_red : ℚ := 7 / total_balls

def prob_same_color : ℚ := prob_blue^2 + prob_green^2 + prob_red^2

theorem prob_same_color_is_69_200 : prob_same_color = 69 / 200 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_69_200_l1035_103564


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1035_103557

theorem chess_tournament_games (n : ℕ) 
  (total_players : ℕ) (total_games : ℕ) :
  total_players = 6 →
  total_games = 30 →
  total_games = n * (total_players.choose 2) →
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1035_103557


namespace NUMINAMATH_CALUDE_tan_negative_two_fraction_l1035_103574

theorem tan_negative_two_fraction (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_two_fraction_l1035_103574


namespace NUMINAMATH_CALUDE_wednesday_occurs_five_times_in_august_l1035_103505

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- July of year N -/
def july : Month := { days := 31, firstDay := DayOfWeek.Tuesday }

/-- August of year N -/
def august : Month := { days := 31, firstDay := sorry }

/-- Counts the occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat := sorry

/-- The main theorem -/
theorem wednesday_occurs_five_times_in_august :
  (countDayOccurrences july DayOfWeek.Tuesday = 5) →
  (countDayOccurrences august DayOfWeek.Wednesday = 5) := by
  sorry

end NUMINAMATH_CALUDE_wednesday_occurs_five_times_in_august_l1035_103505


namespace NUMINAMATH_CALUDE_sofa_loveseat_ratio_l1035_103507

theorem sofa_loveseat_ratio (total_cost love_seat_cost sofa_cost : ℚ) : 
  total_cost = 444 →
  love_seat_cost = 148 →
  total_cost = sofa_cost + love_seat_cost →
  sofa_cost / love_seat_cost = 2 := by
sorry

end NUMINAMATH_CALUDE_sofa_loveseat_ratio_l1035_103507


namespace NUMINAMATH_CALUDE_log_sum_equality_l1035_103594

theorem log_sum_equality (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq1 : q ≠ 1) :
  Real.log p + Real.log q = Real.log (p + q) ↔ p = q / (q - 1) :=
sorry

end NUMINAMATH_CALUDE_log_sum_equality_l1035_103594


namespace NUMINAMATH_CALUDE_excircle_geometric_mean_implies_side_relation_l1035_103511

/-- 
Given a triangle with sides a, b, and c, and excircle radii ra, rb, and rc opposite to sides a, b, and c respectively,
if rc is the geometric mean of ra and rb, then c = (a^2 + b^2) / (a + b).
-/
theorem excircle_geometric_mean_implies_side_relation 
  {a b c ra rb rc : ℝ} 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ ra > 0 ∧ rb > 0 ∧ rc > 0)
  (h_triangle : c < a + b)
  (h_geometric_mean : rc^2 = ra * rb) :
  c = (a^2 + b^2) / (a + b) := by
sorry

end NUMINAMATH_CALUDE_excircle_geometric_mean_implies_side_relation_l1035_103511


namespace NUMINAMATH_CALUDE_fertility_rate_not_valid_indicator_other_indicators_are_valid_l1035_103520

-- Define the type for population growth indicators
inductive PopulationGrowthIndicator
  | BirthRate
  | MortalityRate
  | NaturalIncreaseRate
  | FertilityRate

-- Define the set of valid indicators
def validIndicators : Set PopulationGrowthIndicator :=
  {PopulationGrowthIndicator.BirthRate,
   PopulationGrowthIndicator.MortalityRate,
   PopulationGrowthIndicator.NaturalIncreaseRate}

-- Theorem: Fertility rate is not a valid indicator
theorem fertility_rate_not_valid_indicator :
  PopulationGrowthIndicator.FertilityRate ∉ validIndicators :=
by
  sorry

-- Theorem: All other indicators are valid
theorem other_indicators_are_valid :
  PopulationGrowthIndicator.BirthRate ∈ validIndicators ∧
  PopulationGrowthIndicator.MortalityRate ∈ validIndicators ∧
  PopulationGrowthIndicator.NaturalIncreaseRate ∈ validIndicators :=
by
  sorry

end NUMINAMATH_CALUDE_fertility_rate_not_valid_indicator_other_indicators_are_valid_l1035_103520


namespace NUMINAMATH_CALUDE_max_value_fraction_l1035_103585

theorem max_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * y / (4 * x + 9 * y) ≤ a * b / (4 * a + 9 * b)) → 
  a * b / (4 * a + 9 * b) = 1 / 25 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1035_103585


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l1035_103537

/-- Systematic sampling function that returns the next sample number -/
def nextSample (total : ℕ) (sampleSize : ℕ) (current : ℕ) : ℕ :=
  (current + total / sampleSize) % total

/-- Proposition: In a systematic sampling of 4 items from 56 items, 
    if items 7 and 35 are selected, then the other two selected items 
    are numbered 21 and 49 -/
theorem systematic_sampling_proof :
  let total := 56
  let sampleSize := 4
  let first := 7
  let second := 35
  nextSample total sampleSize first = 21 ∧
  nextSample total sampleSize second = 49 := by
  sorry

#eval nextSample 56 4 7  -- Should output 21
#eval nextSample 56 4 35 -- Should output 49

end NUMINAMATH_CALUDE_systematic_sampling_proof_l1035_103537


namespace NUMINAMATH_CALUDE_final_jellybeans_count_l1035_103546

-- Define the initial number of jellybeans
def initial_jellybeans : ℕ := 90

-- Define the number of jellybeans Samantha took
def samantha_took : ℕ := 24

-- Define the number of jellybeans Shelby ate
def shelby_ate : ℕ := 12

-- Define the function to calculate the final number of jellybeans
def final_jellybeans : ℕ :=
  initial_jellybeans - (samantha_took + shelby_ate) + (samantha_took + shelby_ate) / 2

-- Theorem statement
theorem final_jellybeans_count : final_jellybeans = 72 := by
  sorry

end NUMINAMATH_CALUDE_final_jellybeans_count_l1035_103546


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l1035_103586

theorem halfway_between_fractions :
  let a : ℚ := 1/8
  let b : ℚ := 1/3
  (a + b) / 2 = 11/48 := by
sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l1035_103586


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1035_103547

/-- Represents a company with employees -/
structure Company where
  total_employees : ℕ
  male_employees : ℕ
  female_employees : ℕ

/-- Represents a sample drawn from the company -/
structure Sample where
  female_count : ℕ
  male_count : ℕ

/-- Calculates the sample size given a company and a sample -/
def sample_size (c : Company) (s : Sample) : ℕ :=
  s.female_count + s.male_count

/-- Theorem stating that for a company with 120 employees, of which 90 are male,
    if a stratified sample by gender contains 3 female employees,
    then the total sample size is 12 -/
theorem stratified_sample_size 
  (c : Company) 
  (s : Sample) 
  (h1 : c.total_employees = 120) 
  (h2 : c.male_employees = 90) 
  (h3 : c.female_employees = c.total_employees - c.male_employees) 
  (h4 : s.female_count = 3) :
  sample_size c s = 12 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l1035_103547


namespace NUMINAMATH_CALUDE_fifth_day_sale_l1035_103559

def average_sale : ℕ := 625
def num_days : ℕ := 5
def day1_sale : ℕ := 435
def day2_sale : ℕ := 927
def day3_sale : ℕ := 855
def day4_sale : ℕ := 230

theorem fifth_day_sale :
  ∃ (day5_sale : ℕ),
    day5_sale = average_sale * num_days - (day1_sale + day2_sale + day3_sale + day4_sale) ∧
    day5_sale = 678 := by
  sorry

end NUMINAMATH_CALUDE_fifth_day_sale_l1035_103559


namespace NUMINAMATH_CALUDE_square_difference_formula_l1035_103501

theorem square_difference_formula (x y : ℚ) 
  (sum_eq : x + y = 8/15)
  (diff_eq : x - y = 2/15) :
  x^2 - y^2 = 16/225 := by
sorry

end NUMINAMATH_CALUDE_square_difference_formula_l1035_103501


namespace NUMINAMATH_CALUDE_unique_a_value_l1035_103575

def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

theorem unique_a_value (a : ℝ) (h : 1 ∈ A a) : a = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1035_103575


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l1035_103568

theorem simplify_complex_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (((a^4 * b^2 * c)^(5/3) * (a^3 * b^2 * c)^2)^3)^(1/11) / a^(5/11) = a^3 * b^2 * c :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l1035_103568


namespace NUMINAMATH_CALUDE_not_sum_of_three_cubes_l1035_103583

theorem not_sum_of_three_cubes : ¬ ∃ (x y z : ℤ), x^3 + y^3 + z^3 = 20042005 := by
  sorry

end NUMINAMATH_CALUDE_not_sum_of_three_cubes_l1035_103583


namespace NUMINAMATH_CALUDE_xyz_value_l1035_103570

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 21) :
  x * y * z = 28 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1035_103570


namespace NUMINAMATH_CALUDE_open_box_volume_l1035_103552

/-- The volume of an open box formed by cutting squares from a rectangular sheet -/
theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_length : ℝ)
  (h_length : sheet_length = 100)
  (h_width : sheet_width = 50)
  (h_cut : cut_length = 10) :
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 24000 := by
sorry


end NUMINAMATH_CALUDE_open_box_volume_l1035_103552


namespace NUMINAMATH_CALUDE_arithmetic_seq_2015th_term_l1035_103508

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a_1_eq_1 : a 1 = 1
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  d_neq_0 : a 2 - a 1 ≠ 0
  geometric_subseq : (a 2)^2 = a 1 * a 5

/-- The 2015th term of the arithmetic sequence is 4029 -/
theorem arithmetic_seq_2015th_term (seq : ArithmeticSequence) : seq.a 2015 = 4029 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_2015th_term_l1035_103508


namespace NUMINAMATH_CALUDE_function_properties_l1035_103541

noncomputable def f (x : ℝ) (φ : ℝ) := Real.cos (2 * x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi / 2)
  (h3 : ∀ x, f x φ = f (-Real.pi/3 - x) φ) :
  (f (Real.pi/6) φ = -1/2) ∧ 
  (∃! x, x ∈ Set.Ioo (-Real.pi/2) (Real.pi/2) ∧ 
    ∀ y ∈ Set.Ioo (-Real.pi/2) (Real.pi/2), f y φ ≤ f x φ) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1035_103541


namespace NUMINAMATH_CALUDE_triangle_sine_identity_l1035_103526

theorem triangle_sine_identity (A B C : ℝ) (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
  (h₄ : A + B + C = π) (h₅ : 4 * Real.sin A * Real.sin B * Real.cos C = Real.sin A ^ 2 + Real.sin B ^ 2) : 
  Real.sin A ^ 2 + Real.sin B ^ 2 = 2 * Real.sin C ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_identity_l1035_103526


namespace NUMINAMATH_CALUDE_min_value_sum_product_l1035_103534

theorem min_value_sum_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = x * y) :
  x + y ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l1035_103534


namespace NUMINAMATH_CALUDE_inequality_solution_l1035_103555

theorem inequality_solution (x : ℝ) : 
  (2*x - 1) / (x + 3) > (3*x - 2) / (x - 4) ↔ x ∈ Set.Ioo (-9 : ℝ) (-3) ∪ Set.Ioo 2 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1035_103555


namespace NUMINAMATH_CALUDE_binary_product_theorem_l1035_103502

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.reverse.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its binary representation. -/
def natToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

theorem binary_product_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, true, true, true, false, false, true]  -- 1001111₂
  binaryToNat a * binaryToNat b = binaryToNat c := by
  sorry

end NUMINAMATH_CALUDE_binary_product_theorem_l1035_103502


namespace NUMINAMATH_CALUDE_round_robin_tournament_l1035_103544

theorem round_robin_tournament (n : ℕ) (h : n * (n - 1) / 2 = 15) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_l1035_103544


namespace NUMINAMATH_CALUDE_smallest_odd_number_l1035_103512

theorem smallest_odd_number (x : ℤ) : 
  (x % 2 = 1) →  -- x is odd
  ((x + 2) % 2 = 1) →  -- x + 2 is odd
  ((x + 4) % 2 = 1) →  -- x + 4 is odd
  (x + (x + 2) + (x + 4) = (x + 4) + 28) →  -- sum condition
  (x = 13) :=  -- conclusion
by sorry

end NUMINAMATH_CALUDE_smallest_odd_number_l1035_103512


namespace NUMINAMATH_CALUDE_ellipse_intersecting_line_fixed_point_l1035_103542

/-- An ellipse with center at origin and axes along coordinate axes -/
structure Ellipse where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0
  hne : a ≠ b

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in slope-intercept form -/
structure Line where
  k : ℝ
  t : ℝ

def Ellipse.standardEq (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def Line.eq (l : Line) (p : Point) : Prop :=
  p.y = l.k * p.x + l.t

def tangentAt (e : Ellipse) (l : Line) (p : Point) : Prop :=
  Ellipse.standardEq e p ∧ Line.eq l p

def intersects (e : Ellipse) (l : Line) (a b : Point) : Prop :=
  Ellipse.standardEq e a ∧ Ellipse.standardEq e b ∧ Line.eq l a ∧ Line.eq l b

def circleDiameterPassesThrough (a b c : Point) : Prop :=
  (a.x - b.x) * (c.x - b.x) + (a.y - b.y) * (c.y - b.y) = 0

theorem ellipse_intersecting_line_fixed_point 
  (e : Ellipse) (l : Line) (p a b : Point) :
  e.a^2 = 3 →
  e.b^2 = 4 →
  p.x = 3/2 →
  p.y = 1 →
  tangentAt e { k := 2, t := 4 } p →
  (∃ a b, intersects e l a b ∧ 
    a ≠ b ∧ 
    a.x ≠ e.a ∧ a.x ≠ -e.a ∧ 
    b.x ≠ e.a ∧ b.x ≠ -e.a ∧
    circleDiameterPassesThrough a b { x := 0, y := 2 }) →
  l.eq { x := 0, y := 2/7 } :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersecting_line_fixed_point_l1035_103542


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l1035_103558

theorem increasing_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_sum_positive : a + b > 0) : 
  f a + f b > f (-a) + f (-b) := by
sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l1035_103558


namespace NUMINAMATH_CALUDE_wages_problem_l1035_103543

/-- Represents the wages of a group of people -/
structure Wages where
  men : ℕ
  women : ℕ
  boys : ℕ
  menWage : ℚ
  womenWage : ℚ
  boysWage : ℚ

/-- The problem statement -/
theorem wages_problem (w : Wages) (h1 : w.men = 5) (h2 : w.boys = 8) 
    (h3 : w.men * w.menWage = w.women * w.womenWage) 
    (h4 : w.women * w.womenWage = w.boys * w.boysWage)
    (h5 : w.men * w.menWage + w.women * w.womenWage + w.boys * w.boysWage = 60) :
  w.men * w.menWage = 30 := by
  sorry

end NUMINAMATH_CALUDE_wages_problem_l1035_103543


namespace NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l1035_103506

/-- Given an ellipse with specified center, focus, and endpoint of semi-major axis,
    prove that its semi-minor axis has length √7 -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (semi_major_endpoint : ℝ × ℝ)
  (h1 : center = (-2, 1))
  (h2 : focus = (-3, 0))
  (h3 : semi_major_endpoint = (-2, 4)) :
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2)
  let a := Real.sqrt ((center.1 - semi_major_endpoint.1)^2 + (center.2 - semi_major_endpoint.2)^2)
  let b := Real.sqrt (a^2 - c^2)
  b = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l1035_103506


namespace NUMINAMATH_CALUDE_carl_index_cards_cost_l1035_103576

/-- Calculates the total cost of index cards for Carl's students --/
def total_cost_index_cards (
  cards_6th : ℕ)  -- Number of cards for each 6th grader
  (cards_7th : ℕ) -- Number of cards for each 7th grader
  (cards_8th : ℕ) -- Number of cards for each 8th grader
  (students_6th : ℕ) -- Number of 6th grade students per period
  (students_7th : ℕ) -- Number of 7th grade students per period
  (students_8th : ℕ) -- Number of 8th grade students per period
  (periods : ℕ) -- Number of periods per day
  (pack_size : ℕ) -- Number of cards per pack
  (cost_3x5 : ℕ) -- Cost of a pack of 3x5 cards in dollars
  (cost_4x6 : ℕ) -- Cost of a pack of 4x6 cards in dollars
  : ℕ :=
  let total_cards_6th := cards_6th * students_6th * periods
  let total_cards_7th := cards_7th * students_7th * periods
  let total_cards_8th := cards_8th * students_8th * periods
  let packs_6th := (total_cards_6th + pack_size - 1) / pack_size
  let packs_7th := (total_cards_7th + pack_size - 1) / pack_size
  let packs_8th := (total_cards_8th + pack_size - 1) / pack_size
  packs_6th * cost_3x5 + packs_7th * cost_3x5 + packs_8th * cost_4x6

theorem carl_index_cards_cost : 
  total_cost_index_cards 8 10 12 20 25 30 6 50 3 4 = 326 := by
  sorry

end NUMINAMATH_CALUDE_carl_index_cards_cost_l1035_103576


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1035_103528

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_rhombus_perimeter_l1035_103528


namespace NUMINAMATH_CALUDE_tangent_function_property_l1035_103579

theorem tangent_function_property (ω : ℝ) (h1 : ω > 0) : 
  (∀ x : ℝ, ∃ y : ℝ, y > x ∧ Real.tan (ω * y) = Real.tan (ω * x) ∧ y - x = π / 4) → 
  Real.tan (ω * (π / 4)) = 0 := by
sorry

end NUMINAMATH_CALUDE_tangent_function_property_l1035_103579


namespace NUMINAMATH_CALUDE_johnson_family_seating_l1035_103550

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def boys : ℕ := 5
def girls : ℕ := 4
def total_children : ℕ := boys + girls

def block_arrangements : ℕ := 7 * (factorial boys) * (factorial (total_children - 3))

theorem johnson_family_seating :
  factorial total_children - block_arrangements = 60480 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l1035_103550
