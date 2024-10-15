import Mathlib

namespace NUMINAMATH_CALUDE_first_day_rainfall_is_26_l567_56708

/-- Rainfall data for May -/
structure RainfallData where
  day2 : ℝ
  day3_diff : ℝ
  normal_average : ℝ
  less_than_average : ℝ

/-- Calculate the rainfall on the first day -/
def calculate_first_day_rainfall (data : RainfallData) : ℝ :=
  3 * data.normal_average - data.less_than_average - data.day2 - (data.day2 - data.day3_diff)

/-- Theorem stating that the rainfall on the first day is 26 cm -/
theorem first_day_rainfall_is_26 (data : RainfallData)
  (h1 : data.day2 = 34)
  (h2 : data.day3_diff = 12)
  (h3 : data.normal_average = 140)
  (h4 : data.less_than_average = 58) :
  calculate_first_day_rainfall data = 26 := by
  sorry

#eval calculate_first_day_rainfall ⟨34, 12, 140, 58⟩

end NUMINAMATH_CALUDE_first_day_rainfall_is_26_l567_56708


namespace NUMINAMATH_CALUDE_func_f_properties_l567_56743

/-- A function satisfying the given functional equation -/
noncomputable def FuncF (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a * f b) ∧ 
  (f 0 ≠ 0) ∧
  (∃ c : ℝ, c > 0 ∧ f (c / 2) = 0)

theorem func_f_properties (f : ℝ → ℝ) (h : FuncF f) :
  (f 0 = 1) ∧ 
  (∀ x : ℝ, f (-x) = f x) ∧
  (∃ c : ℝ, c > 0 ∧ ∀ x : ℝ, f (x + 2 * c) = f x) :=
by sorry

end NUMINAMATH_CALUDE_func_f_properties_l567_56743


namespace NUMINAMATH_CALUDE_parabola_r_value_l567_56759

/-- A parabola in the xy-plane defined by x = py^2 + qy + r -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (para : Parabola) (y : ℝ) : ℝ :=
  para.p * y^2 + para.q * y + para.r

theorem parabola_r_value (para : Parabola) :
  para.x_coord 4 = 5 →
  para.x_coord 6 = 3 →
  para.x_coord 0 = 3 →
  para.r = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_r_value_l567_56759


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l567_56790

theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : current_speed = 5) 
  (h2 : downstream_distance = 6.25) 
  (h3 : downstream_time = 0.25) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 20 ∧ 
    downstream_distance = (still_water_speed + current_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l567_56790


namespace NUMINAMATH_CALUDE_triangle_ratio_equals_two_l567_56770

/-- In triangle ABC, if angle A is 60 degrees and side a is √3, 
    then (a + b) / (sin A + sin B) = 2 -/
theorem triangle_ratio_equals_two (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧ 
  A = π / 3 ∧ 
  a = Real.sqrt 3 ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C →
  (a + b) / (Real.sin A + Real.sin B) = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_equals_two_l567_56770


namespace NUMINAMATH_CALUDE_modulus_z₂_l567_56707

-- Define the complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- State the conditions
axiom z₁_condition : (z₁ - 2) * Complex.I = 1 + Complex.I
axiom z₂_imag_part : z₂.im = 2
axiom product_real : (z₁ * z₂).im = 0

-- State the theorem
theorem modulus_z₂ : Complex.abs z₂ = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_z₂_l567_56707


namespace NUMINAMATH_CALUDE_book_arrangement_count_l567_56728

def num_math_books : ℕ := 4
def num_history_books : ℕ := 6
def total_books : ℕ := num_math_books + num_history_books

def arrange_books : ℕ := num_math_books * (num_math_books - 1) * Nat.factorial (total_books - 2)

theorem book_arrangement_count :
  arrange_books = 145152 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l567_56728


namespace NUMINAMATH_CALUDE_saltwater_solution_volume_l567_56724

theorem saltwater_solution_volume :
  -- Initial conditions
  ∀ x : ℝ,
  let initial_salt_volume := 0.20 * x
  let evaporated_volume := 0.25 * x
  let remaining_volume := x - evaporated_volume
  let added_water := 6
  let added_salt := 12
  let final_volume := remaining_volume + added_water + added_salt
  let final_salt_volume := initial_salt_volume + added_salt
  -- Final salt concentration condition
  final_salt_volume / final_volume = 1/3 →
  -- Conclusion
  x = 120 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_solution_volume_l567_56724


namespace NUMINAMATH_CALUDE_two_possible_products_l567_56748

theorem two_possible_products (a b : ℝ) (ha : |a| = 5) (hb : |b| = 3) :
  ∃ (x y : ℝ), (∀ z, a * b = z → z = x ∨ z = y) ∧ x ≠ y :=
sorry

end NUMINAMATH_CALUDE_two_possible_products_l567_56748


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l567_56797

theorem quadratic_equation_roots (a b c : ℝ) (h1 : a = 1) (h2 : b = 2023) (h3 : c = 2035) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l567_56797


namespace NUMINAMATH_CALUDE_function_identity_l567_56750

theorem function_identity (f : ℕ → ℕ) 
  (h1 : ∀ (m n : ℕ), f (m^2 + n^2) = (f m)^2 + (f n)^2) 
  (h2 : f 1 > 0) : 
  ∀ (n : ℕ), f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l567_56750


namespace NUMINAMATH_CALUDE_complement_of_intersection_l567_56772

open Set

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

theorem complement_of_intersection (U A B : Finset ℕ) 
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 2, 3})
  (hB : B = {2, 3, 4}) :
  (U \ (A ∩ B)) = {1, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l567_56772


namespace NUMINAMATH_CALUDE_clown_balloons_l567_56788

/-- The number of balloons a clown has after blowing up two sets of balloons -/
def total_balloons (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that the clown has 60 balloons after blowing up 47 and then 13 more -/
theorem clown_balloons :
  total_balloons 47 13 = 60 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l567_56788


namespace NUMINAMATH_CALUDE_train_length_train_length_is_240_l567_56739

/-- The length of a train crossing a bridge -/
theorem train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - bridge_length

/-- Proof that the train length is 240 meters -/
theorem train_length_is_240 :
  train_length 150 20 70.2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_is_240_l567_56739


namespace NUMINAMATH_CALUDE_binomial_square_constant_l567_56785

theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 150*x + c = (x + a)^2) → c = 5625 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l567_56785


namespace NUMINAMATH_CALUDE_harrys_creation_weight_is_25_l567_56791

/-- The weight of Harry's custom creation at the gym -/
def harrys_creation_weight (blue_weight green_weight : ℕ) (blue_count green_count bar_weight : ℕ) : ℕ :=
  blue_weight * blue_count + green_weight * green_count + bar_weight

/-- Theorem stating that Harry's creation weighs 25 pounds -/
theorem harrys_creation_weight_is_25 :
  harrys_creation_weight 2 3 4 5 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_harrys_creation_weight_is_25_l567_56791


namespace NUMINAMATH_CALUDE_two_mice_boring_l567_56737

/-- The sum of distances bored by two mice in n days -/
def S (n : ℕ) : ℚ :=
  let big_mouse := 2^n - 1  -- Sum of geometric sequence with a₁ = 1, r = 2
  let small_mouse := 2 - 1 / 2^(n-1)  -- Sum of geometric sequence with a₁ = 1, r = 1/2
  big_mouse + small_mouse

theorem two_mice_boring (n : ℕ) : S n = 2^n - 1/2^(n-1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_two_mice_boring_l567_56737


namespace NUMINAMATH_CALUDE_inequality_solution_set_l567_56704

theorem inequality_solution_set (x : ℝ) : 2 ≤ x / (2 * x - 1) ∧ x / (2 * x - 1) < 5 ↔ x ∈ Set.Ioo (5/9 : ℝ) (2/3 : ℝ) ∪ {2/3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l567_56704


namespace NUMINAMATH_CALUDE_equation_solution_l567_56764

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = -4/3 ∧ x₂ = 3 ∧
  ∀ (x : ℝ), x ≠ 2/3 → x ≠ -4/3 →
  ((6*x + 4) / (3*x^2 + 6*x - 8) = (3*x) / (3*x - 2) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l567_56764


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l567_56710

/-- Given that f(x) = e^(2x) - ae^x + 2x is an increasing function on ℝ, 
    prove that the range of a is (-∞, 4]. -/
theorem range_of_a_for_increasing_f (a : ℝ) : 
  (∀ x : ℝ, Monotone (fun x => Real.exp (2 * x) - a * Real.exp x + 2 * x)) →
  a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l567_56710


namespace NUMINAMATH_CALUDE_smallest_multiple_thirty_two_satisfies_smallest_multiple_is_32_l567_56775

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 900 * x % 1152 = 0 → x ≥ 32 := by
  sorry

theorem thirty_two_satisfies : 900 * 32 % 1152 = 0 := by
  sorry

theorem smallest_multiple_is_32 : 
  ∃ (x : ℕ), x > 0 ∧ 900 * x % 1152 = 0 ∧ ∀ (y : ℕ), y > 0 ∧ 900 * y % 1152 = 0 → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_thirty_two_satisfies_smallest_multiple_is_32_l567_56775


namespace NUMINAMATH_CALUDE_three_propositions_l567_56752

theorem three_propositions :
  (∀ a b : ℝ, |a - b| < 1 → |a| < |b| + 1) ∧
  (∀ a b : ℝ, |a + b| - 2*|a| ≤ |a - b|) ∧
  (∀ x y : ℝ, |x| < 2 ∧ |y| > 3 → |x / y| < 2/3) := by
  sorry

end NUMINAMATH_CALUDE_three_propositions_l567_56752


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l567_56763

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x < 2) ↔ (∀ x : ℝ, x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l567_56763


namespace NUMINAMATH_CALUDE_intersection_equality_subset_condition_l567_56757

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 < 2*x + 1 ∧ 2*x + 1 < 7}
def B : Set ℝ := {x | x < -4 ∨ x > 2}
def C (a : ℝ) : Set ℝ := {x | 3*a - 2 < x ∧ x < a + 1}

-- Statement 1: A ∩ (C_R B) = {x | -2 < x ≤ 2}
theorem intersection_equality : A ∩ (Set.Icc (-4) 2) = {x : ℝ | -2 < x ∧ x ≤ 2} := by sorry

-- Statement 2: C_R (A∪B) ⊆ C if and only if -3 < a < -2/3
theorem subset_condition (a : ℝ) : Set.Icc (-4) 2 ⊆ C a ↔ -3 < a ∧ a < -2/3 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_subset_condition_l567_56757


namespace NUMINAMATH_CALUDE_x_minus_y_equals_106_over_21_l567_56735

theorem x_minus_y_equals_106_over_21 (x y : ℚ) : 
  x + 2*y = 16/3 → 5*x + 3*y = 26 → x - y = 106/21 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_106_over_21_l567_56735


namespace NUMINAMATH_CALUDE_odd_products_fraction_l567_56731

/-- The number of integers from 0 to 15 inclusive -/
def table_size : ℕ := 16

/-- The count of odd numbers from 0 to 15 inclusive -/
def odd_count : ℕ := 8

/-- The total number of entries in the multiplication table -/
def total_entries : ℕ := table_size * table_size

/-- The number of odd products in the multiplication table -/
def odd_products : ℕ := odd_count * odd_count

/-- The fraction of odd products in the multiplication table -/
def odd_fraction : ℚ := odd_products / total_entries

theorem odd_products_fraction :
  odd_fraction = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_odd_products_fraction_l567_56731


namespace NUMINAMATH_CALUDE_currency_comparisons_l567_56753

-- Define the conversion rate from jiao to yuan
def jiao_to_yuan (jiao : ℚ) : ℚ := jiao / 10

-- Define the theorem
theorem currency_comparisons :
  (2.3 < 3.2) ∧
  (10 > 9.9) ∧
  (1 + jiao_to_yuan 6 = 1.6) ∧
  (15 * 4 < 14 * 5) :=
by sorry

end NUMINAMATH_CALUDE_currency_comparisons_l567_56753


namespace NUMINAMATH_CALUDE_probability_consecutive_cards_l567_56722

/-- A type representing the cards labeled A, B, C, D, E -/
inductive Card : Type
  | A | B | C | D | E

/-- A function to check if two cards are consecutive -/
def consecutive (c1 c2 : Card) : Bool :=
  match c1, c2 with
  | Card.A, Card.B | Card.B, Card.A => true
  | Card.B, Card.C | Card.C, Card.B => true
  | Card.C, Card.D | Card.D, Card.C => true
  | Card.D, Card.E | Card.E, Card.D => true
  | _, _ => false

/-- The total number of ways to choose 2 cards from 5 -/
def totalChoices : Nat := 10

/-- The number of ways to choose 2 consecutive cards -/
def consecutiveChoices : Nat := 4

/-- Theorem stating the probability of drawing two consecutive cards -/
theorem probability_consecutive_cards :
  (consecutiveChoices : ℚ) / totalChoices = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_probability_consecutive_cards_l567_56722


namespace NUMINAMATH_CALUDE_exists_coverable_prism_l567_56776

/-- A regular triangular prism with side edge length √3 times the base edge length -/
structure RegularTriangularPrism where
  base_edge : ℝ
  side_edge : ℝ
  side_edge_eq : side_edge = base_edge * Real.sqrt 3

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ

/-- A covering of a prism by equilateral triangles -/
structure PrismCovering where
  prism : RegularTriangularPrism
  triangles : Set EquilateralTriangle
  covers_prism : Bool
  no_overlaps : Bool

/-- Theorem stating the existence of a regular triangular prism that can be covered by equilateral triangles -/
theorem exists_coverable_prism : ∃ (p : RegularTriangularPrism) (c : PrismCovering), 
  c.prism = p ∧ c.covers_prism ∧ c.no_overlaps := by
  sorry

end NUMINAMATH_CALUDE_exists_coverable_prism_l567_56776


namespace NUMINAMATH_CALUDE_scientific_notation_505000_l567_56784

theorem scientific_notation_505000 :
  505000 = 5.05 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_505000_l567_56784


namespace NUMINAMATH_CALUDE_volume_union_tetrahedrons_is_half_l567_56768

/-- A regular tetrahedron formed from vertices of a unit cube -/
structure CubeTetrahedron where
  vertices : Finset (Fin 8)
  is_regular : Bool
  from_cube : Bool

/-- The volume of the union of two regular tetrahedrons formed from the vertices of a unit cube -/
def volume_union_tetrahedrons (t1 t2 : CubeTetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the union of two regular tetrahedrons
    formed from the vertices of a unit cube is 1/2 -/
theorem volume_union_tetrahedrons_is_half
  (t1 t2 : CubeTetrahedron)
  (h1 : t1.is_regular)
  (h2 : t2.is_regular)
  (h3 : t1.from_cube)
  (h4 : t2.from_cube)
  (h5 : t1.vertices ≠ t2.vertices)
  : volume_union_tetrahedrons t1 t2 = 1/2 :=
sorry

end NUMINAMATH_CALUDE_volume_union_tetrahedrons_is_half_l567_56768


namespace NUMINAMATH_CALUDE_root_product_equation_l567_56723

theorem root_product_equation (m p q : ℝ) (a b : ℝ) : 
  (a^2 - m*a + 6 = 0) → 
  (b^2 - m*b + 6 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 49/6 := by
sorry

end NUMINAMATH_CALUDE_root_product_equation_l567_56723


namespace NUMINAMATH_CALUDE_circle_area_in_square_l567_56754

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + y^2 - 16*y + 65 = 0

-- Define the square
def square : Set (ℝ × ℝ) :=
  {p | 3 ≤ p.1 ∧ p.1 ≤ 8 ∧ 8 ≤ p.2 ∧ p.2 ≤ 13}

-- Theorem statement
theorem circle_area_in_square :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    (∀ x y, circle_equation x y → (x, y) ∈ square) ∧
    (π * radius^2 = 24 * π) :=
sorry

end NUMINAMATH_CALUDE_circle_area_in_square_l567_56754


namespace NUMINAMATH_CALUDE_abs_sum_eq_sum_abs_iff_product_nonneg_l567_56703

theorem abs_sum_eq_sum_abs_iff_product_nonneg (x y : ℝ) :
  abs (x + y) = abs x + abs y ↔ x * y ≥ 0 := by sorry

end NUMINAMATH_CALUDE_abs_sum_eq_sum_abs_iff_product_nonneg_l567_56703


namespace NUMINAMATH_CALUDE_gym_equipment_cost_l567_56786

/-- The cost of replacing all cardio machines in a global gym chain --/
theorem gym_equipment_cost (num_gyms : ℕ) (num_bikes : ℕ) (num_treadmills : ℕ) (num_ellipticals : ℕ)
  (treadmill_cost_factor : ℚ) (elliptical_cost_factor : ℚ) (total_cost : ℚ) :
  num_gyms = 20 →
  num_bikes = 10 →
  num_treadmills = 5 →
  num_ellipticals = 5 →
  treadmill_cost_factor = 3/2 →
  elliptical_cost_factor = 2 →
  total_cost = 455000 →
  ∃ (bike_cost : ℚ),
    bike_cost = 700 ∧
    total_cost = num_gyms * (num_bikes * bike_cost +
                             num_treadmills * treadmill_cost_factor * bike_cost +
                             num_ellipticals * elliptical_cost_factor * treadmill_cost_factor * bike_cost) :=
by sorry

end NUMINAMATH_CALUDE_gym_equipment_cost_l567_56786


namespace NUMINAMATH_CALUDE_unripe_apples_correct_l567_56725

/-- Calculates the number of unripe apples given the total number of apples picked,
    the number of pies that can be made, and the number of apples needed per pie. -/
def unripe_apples (total_apples : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  total_apples - (num_pies * apples_per_pie)

/-- Proves that the number of unripe apples is correct for the given scenario. -/
theorem unripe_apples_correct : unripe_apples 34 7 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_unripe_apples_correct_l567_56725


namespace NUMINAMATH_CALUDE_length_of_PQ_l567_56781

-- Define the points and lines
def R : ℝ × ℝ := (10, 8)
def line1 (x y : ℝ) : Prop := 7 * y = 9 * x
def line2 (x y : ℝ) : Prop := 12 * y = 5 * x

-- Define the theorem
theorem length_of_PQ : 
  ∀ (P Q : ℝ × ℝ),
  -- R is the midpoint of PQ
  (P.1 + Q.1) / 2 = R.1 ∧ (P.2 + Q.2) / 2 = R.2 ∧
  -- P is on line1
  line1 P.1 P.2 ∧
  -- Q is on line2
  line2 Q.1 Q.2 →
  -- The length of PQ is 4√134481/73
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 4 * Real.sqrt 134481 / 73 := by
  sorry

end NUMINAMATH_CALUDE_length_of_PQ_l567_56781


namespace NUMINAMATH_CALUDE_cookies_sum_l567_56749

/-- The number of cookies Mona brought -/
def mona_cookies : ℕ := 20

/-- The number of cookies Jasmine brought -/
def jasmine_cookies : ℕ := mona_cookies - 5

/-- The number of cookies Rachel brought -/
def rachel_cookies : ℕ := jasmine_cookies + 10

/-- The total number of cookies brought by Mona, Jasmine, and Rachel -/
def total_cookies : ℕ := mona_cookies + jasmine_cookies + rachel_cookies

theorem cookies_sum : total_cookies = 60 := by
  sorry

end NUMINAMATH_CALUDE_cookies_sum_l567_56749


namespace NUMINAMATH_CALUDE_liters_to_gallons_conversion_l567_56730

/-- Conversion factor from liters to gallons -/
def liters_to_gallons : ℝ := 0.26

/-- The volume in liters -/
def volume_in_liters : ℝ := 2.5

/-- Theorem stating that 2.5 liters is equal to 0.65 gallons -/
theorem liters_to_gallons_conversion :
  volume_in_liters * liters_to_gallons = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_liters_to_gallons_conversion_l567_56730


namespace NUMINAMATH_CALUDE_sugar_amount_is_correct_l567_56799

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- Calculates the amount of sugar needed in the new recipe -/
def sugar_needed (original : RecipeRatio) (water_new : ℚ) : ℚ :=
  let new_flour_water_ratio := (original.flour / original.water) * 3
  let new_flour_sugar_ratio := (original.flour / original.sugar) / 3
  let flour_new := (new_flour_water_ratio * water_new)
  flour_new / new_flour_sugar_ratio

/-- Theorem: Given the conditions, the amount of sugar needed is 0.75 cups -/
theorem sugar_amount_is_correct (original : RecipeRatio) 
  (h1 : original.flour = 11)
  (h2 : original.water = 8)
  (h3 : original.sugar = 1)
  (h4 : sugar_needed original 6 = 3/4) : 
  sugar_needed original 6 = 0.75 := by
  sorry

#eval sugar_needed ⟨11, 8, 1⟩ 6

end NUMINAMATH_CALUDE_sugar_amount_is_correct_l567_56799


namespace NUMINAMATH_CALUDE_min_box_value_l567_56702

theorem min_box_value (a b Box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 15 * x^2 + Box * x + 15) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  (∃ min_Box : ℤ, 
    (∀ a' b' Box' : ℤ, 
      (∀ x, (a' * x + b') * (b' * x + a') = 15 * x^2 + Box' * x + 15) →
      a' ≠ b' ∧ b' ≠ Box' ∧ a' ≠ Box' →
      Box' ≥ min_Box) ∧
    min_Box = 34 ∧
    ((a = 3 ∧ b = 5) ∨ (a = -3 ∧ b = -5) ∨ (a = 5 ∧ b = 3) ∨ (a = -5 ∧ b = -3))) :=
by sorry

end NUMINAMATH_CALUDE_min_box_value_l567_56702


namespace NUMINAMATH_CALUDE_continued_fraction_value_l567_56798

theorem continued_fraction_value : ∃ x : ℝ, x = 3 + 5 / (2 + 5 / x) ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l567_56798


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l567_56751

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) : 
  (∀ angle : ℝ, angle = 150 ∧ (n * angle : ℝ) = 180 * (n - 2 : ℝ)) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l567_56751


namespace NUMINAMATH_CALUDE_jellybean_problem_l567_56716

theorem jellybean_problem : ∃ (n : ℕ), n ≥ 150 ∧ n % 19 = 17 ∧ ∀ (m : ℕ), m ≥ 150 ∧ m % 19 = 17 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_jellybean_problem_l567_56716


namespace NUMINAMATH_CALUDE_sum_of_squares_geq_sum_of_products_sqrt_inequality_l567_56796

-- Statement 1
theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) :
  a^2 + b^2 + c^2 ≥ a*b + a*c + b*c :=
sorry

-- Statement 2
theorem sqrt_inequality :
  Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_geq_sum_of_products_sqrt_inequality_l567_56796


namespace NUMINAMATH_CALUDE_complement_of_A_l567_56705

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem complement_of_A (A B : Set ℕ) 
  (h1 : A ∪ B = {1, 2, 3, 4, 5})
  (h2 : A ∩ B = {3, 4, 5}) :
  (U \ A) = {6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l567_56705


namespace NUMINAMATH_CALUDE_a_share_is_288_l567_56718

/-- Calculates the share of profit for investor A given the initial investments,
    changes after 8 months, and total profit over a year. -/
def calculate_share_a (a_initial : ℕ) (b_initial : ℕ) (a_change : ℕ) (b_change : ℕ) (total_profit : ℕ) : ℕ :=
  let a_investment_months := a_initial * 8 + (a_initial - a_change) * 4
  let b_investment_months := b_initial * 8 + (b_initial + b_change) * 4
  let total_investment_months := a_investment_months + b_investment_months
  let a_ratio := a_investment_months * total_profit / total_investment_months
  a_ratio

/-- Theorem stating that A's share of the profit is 288 Rs given the problem conditions. -/
theorem a_share_is_288 :
  calculate_share_a 3000 4000 1000 1000 756 = 288 := by
  sorry

end NUMINAMATH_CALUDE_a_share_is_288_l567_56718


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l567_56701

/-- The set of real numbers x such that x^2 - 9 > 0 -/
def A : Set ℝ := {x | x^2 - 9 > 0}

/-- The set of real numbers x such that x^2 - 5/6*x + 1/6 > 0 -/
def B : Set ℝ := {x | x^2 - 5/6*x + 1/6 > 0}

/-- Theorem stating that A is a subset of B and there exists an element in B that is not in A -/
theorem sufficient_not_necessary : A ⊆ B ∧ ∃ x, x ∈ B ∧ x ∉ A :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l567_56701


namespace NUMINAMATH_CALUDE_modulus_of_z_l567_56744

theorem modulus_of_z (z : ℂ) (h : (3 + 4 * Complex.I) * z = 1) : Complex.abs z = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l567_56744


namespace NUMINAMATH_CALUDE_largest_roots_ratio_l567_56762

/-- The polynomial f(x) = 1 - x - 4x² + x⁴ -/
def f (x : ℝ) : ℝ := 1 - x - 4*x^2 + x^4

/-- The polynomial g(x) = 16 - 8x - 16x² + x⁴ -/
def g (x : ℝ) : ℝ := 16 - 8*x - 16*x^2 + x^4

/-- x₁ is the largest root of f -/
def x₁ : ℝ := sorry

/-- x₂ is the largest root of g -/
def x₂ : ℝ := sorry

theorem largest_roots_ratio :
  x₁ / x₂ = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_largest_roots_ratio_l567_56762


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l567_56782

theorem purely_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := m + 2 + (m - 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l567_56782


namespace NUMINAMATH_CALUDE_female_officers_count_l567_56740

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 500 →
  female_on_duty_ratio = 1/4 →
  female_ratio = 1/2 →
  (female_on_duty_ratio * (total_on_duty * female_ratio)) / female_on_duty_ratio = 1000 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l567_56740


namespace NUMINAMATH_CALUDE_nested_radical_solution_l567_56714

theorem nested_radical_solution :
  ∃ x : ℝ, x = Real.sqrt (3 - x) → x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_solution_l567_56714


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l567_56794

-- Define the repeating decimal
def repeating_decimal : ℚ := 37 / 100 + 264 / 99900

-- Define the fraction
def fraction : ℚ := 37189162 / 99900

-- Theorem statement
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l567_56794


namespace NUMINAMATH_CALUDE_inequality_proof_l567_56738

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l567_56738


namespace NUMINAMATH_CALUDE_domain_and_even_function_implies_a_eq_neg_one_l567_56789

/-- A function is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem domain_and_even_function_implies_a_eq_neg_one
  (a : ℝ)
  (f : ℝ → ℝ)
  (h_domain : Set.Ioo (4*a - 3) (3 - 2*a^2) = Set.range f)
  (h_even : IsEven (fun x ↦ f (2*x - 3))) :
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_domain_and_even_function_implies_a_eq_neg_one_l567_56789


namespace NUMINAMATH_CALUDE_tank_width_proof_l567_56715

/-- Proves that a tank with given dimensions and plastering cost has a width of 12 meters -/
theorem tank_width_proof (length depth : ℝ) (cost_per_sqm total_cost : ℝ) :
  length = 25 →
  depth = 6 →
  cost_per_sqm = 0.30 →
  total_cost = 223.2 →
  ∃ width : ℝ,
    width = 12 ∧
    total_cost = cost_per_sqm * (length * width + 2 * (length * depth + width * depth)) :=
by sorry

end NUMINAMATH_CALUDE_tank_width_proof_l567_56715


namespace NUMINAMATH_CALUDE_point_moved_upwards_l567_56777

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point upwards by a given distance -/
def moveUpwards (p : Point) (distance : ℝ) : Point :=
  { x := p.x, y := p.y + distance }

theorem point_moved_upwards (P : Point) (Q : Point) :
  P.x = -3 ∧ P.y = 1 ∧ Q = moveUpwards P 2 → Q.x = -3 ∧ Q.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_moved_upwards_l567_56777


namespace NUMINAMATH_CALUDE_circle_line_intersection_l567_56783

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 8*x + 15 = 0

-- Define the line
def line (x y k : ℝ) : Prop := y = k*x - 2

-- Define the condition for common points
def has_common_points (k : ℝ) : Prop :=
  ∃ x y : ℝ, line x y k ∧ 
    ∃ x' y' : ℝ, circle_C x' y' ∧ 
      (x - x')^2 + (y - y')^2 ≤ 4

-- The main theorem
theorem circle_line_intersection (k : ℝ) :
  has_common_points k → -4/3 ≤ k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l567_56783


namespace NUMINAMATH_CALUDE_largest_c_for_range_containing_negative_five_l567_56756

theorem largest_c_for_range_containing_negative_five :
  let f (x c : ℝ) := x^2 + 5*x + c
  ∃ (c_max : ℝ), c_max = 5/4 ∧
    (∀ c : ℝ, (∃ x : ℝ, f x c = -5) → c ≤ c_max) ∧
    (∃ x : ℝ, f x c_max = -5) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_range_containing_negative_five_l567_56756


namespace NUMINAMATH_CALUDE_last_two_digits_product_l567_56717

/-- Given an integer n, returns its last two digits as a pair of natural numbers -/
def lastTwoDigits (n : ℤ) : ℕ × ℕ :=
  let tens := (n % 100 / 10).toNat
  let units := (n % 10).toNat
  (tens, units)

/-- Theorem: For any integer divisible by 4 with the sum of its last two digits equal to 17,
    the product of its last two digits is 72 -/
theorem last_two_digits_product (n : ℤ) 
  (div_by_4 : 4 ∣ n) 
  (sum_17 : (lastTwoDigits n).1 + (lastTwoDigits n).2 = 17) : 
  (lastTwoDigits n).1 * (lastTwoDigits n).2 = 72 :=
by
  sorry

#check last_two_digits_product

end NUMINAMATH_CALUDE_last_two_digits_product_l567_56717


namespace NUMINAMATH_CALUDE_disprove_combined_average_formula_l567_56792

theorem disprove_combined_average_formula :
  ∃ (a b : ℕ+), a ≠ b ∧
    ∀ (m n : ℕ+), m ≠ n →
      (m.val * a.val + n.val * b.val) / (m.val + n.val) ≠ (a.val + b.val) / 2 := by
  sorry

end NUMINAMATH_CALUDE_disprove_combined_average_formula_l567_56792


namespace NUMINAMATH_CALUDE_remainder_of_difference_l567_56706

theorem remainder_of_difference (s t : ℕ) (hs : s > 0) (ht : t > 0) 
  (h_s_mod : s % 6 = 2) (h_t_mod : t % 6 = 3) (h_s_gt_t : s > t) : 
  (s - t) % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_difference_l567_56706


namespace NUMINAMATH_CALUDE_part1_part2_l567_56713

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem part1 (t : Triangle) 
  (h1 : t.b = Real.sqrt 3) 
  (h2 : t.C = 5 * Real.pi / 6) 
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2) : 
  t.c = Real.sqrt 13 := by
sorry

-- Part 2
theorem part2 (t : Triangle) 
  (h1 : t.b = Real.sqrt 3) 
  (h2 : t.B = Real.pi / 3) : 
  ∃ (x y : ℝ), x = -Real.sqrt 3 ∧ y = 2 * Real.sqrt 3 ∧ 
  ∀ z, (2 * t.c - t.a = z) → (x < z ∧ z < y) := by
sorry

end NUMINAMATH_CALUDE_part1_part2_l567_56713


namespace NUMINAMATH_CALUDE_difference_of_squares_l567_56767

theorem difference_of_squares (t : ℝ) : t^2 - 121 = (t - 11) * (t + 11) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l567_56767


namespace NUMINAMATH_CALUDE_complex_number_location_l567_56760

theorem complex_number_location :
  let z : ℂ := 1 / (2 + Complex.I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l567_56760


namespace NUMINAMATH_CALUDE_probability_five_blue_marbles_in_eight_draws_l567_56729

/-- The probability of drawing exactly k blue marbles in n draws with replacement -/
def probability_k_blue_marbles (total_marbles blue_marbles k n : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (blue_marbles / total_marbles : ℚ) ^ k * 
  ((total_marbles - blue_marbles) / total_marbles : ℚ) ^ (n - k)

/-- The probability of drawing exactly 5 blue marbles in 8 draws with replacement
    from a bag containing 9 blue marbles and 6 red marbles -/
theorem probability_five_blue_marbles_in_eight_draws : 
  probability_k_blue_marbles 15 9 5 8 = 108864 / 390625 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_blue_marbles_in_eight_draws_l567_56729


namespace NUMINAMATH_CALUDE_factor_theorem_p_factorization_p_factorization_q_l567_56733

-- Define the polynomials
def p (x : ℝ) := 6 * x^2 - x - 5
def q (x : ℝ) := x^3 - 7 * x + 6

-- State the theorems
theorem factor_theorem_p : ∃ (r : ℝ → ℝ), ∀ x, p x = (x - 1) * r x := by sorry

theorem factorization_p : ∀ x, p x = (x - 1) * (6 * x + 5) := by sorry

theorem factorization_q : ∀ x, q x = (x - 1) * (x + 3) * (x - 2) := by sorry

-- Given condition
axiom p_root : p 1 = 0

end NUMINAMATH_CALUDE_factor_theorem_p_factorization_p_factorization_q_l567_56733


namespace NUMINAMATH_CALUDE_ratio_sum_difference_l567_56755

theorem ratio_sum_difference (a b c : ℝ) : 
  (a : ℝ) / 1 = (b : ℝ) / 3 ∧ (b : ℝ) / 3 = (c : ℝ) / 6 →
  a + b + c = 30 →
  c - b - a = 6 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_difference_l567_56755


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l567_56787

/-- Represents an ellipse with semi-major axis 2 and semi-minor axis b -/
structure Ellipse (b : ℝ) :=
  (equation : ℝ → ℝ → Prop)
  (b_pos : b > 0)

/-- Represents a point on the ellipse -/
structure EllipsePoint (E : Ellipse b) :=
  (x y : ℝ)
  (on_ellipse : E.equation x y)

/-- The left focus of the ellipse -/
def left_focus (E : Ellipse b) : ℝ × ℝ := sorry

/-- The right focus of the ellipse -/
def right_focus (E : Ellipse b) : ℝ × ℝ := sorry

/-- A line passing through the left focus -/
structure FocalLine (E : Ellipse b) :=
  (passes_through_left_focus : Prop)

/-- Intersection points of a focal line with the ellipse -/
def intersection_points (E : Ellipse b) (l : FocalLine E) : EllipsePoint E × EllipsePoint E := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Maximum sum of distances from intersection points to the right focus -/
def max_sum_distances (E : Ellipse b) : ℝ := sorry

/-- Eccentricity of the ellipse -/
def eccentricity (E : Ellipse b) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_eccentricity (b : ℝ) (E : Ellipse b) :
  max_sum_distances E = 5 → eccentricity E = 1/2 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l567_56787


namespace NUMINAMATH_CALUDE_unique_k_solution_l567_56711

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem unique_k_solution (k : ℤ) : 
  k % 2 = 1 ∧ f (f (f k)) = 35 → k = 55 :=
by sorry

end NUMINAMATH_CALUDE_unique_k_solution_l567_56711


namespace NUMINAMATH_CALUDE_f_sum_lower_bound_f_squared_sum_lower_bound_l567_56766

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 1)

-- Theorem 1
theorem f_sum_lower_bound : ∀ x : ℝ, f x + f (1 - x) ≥ 1 := by sorry

-- Theorem 2
theorem f_squared_sum_lower_bound (a b : ℝ) (h : a + 2 * b = 8) : f a ^ 2 + f b ^ 2 ≥ 5 := by sorry

end NUMINAMATH_CALUDE_f_sum_lower_bound_f_squared_sum_lower_bound_l567_56766


namespace NUMINAMATH_CALUDE_collinearity_iff_harmonic_l567_56758

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the incidence relation
variable (incident : Point → Line → Prop)

-- Define the collinearity relation
variable (collinear : Point → Point → Point → Prop)

-- Define the harmonic relation
variable (harmonic : Point → Point → Point → Point → Prop)

-- Define the points and lines
variable (A B C D E F H P X Y : Point)
variable (hA hB gA gB : Line)

-- Define the geometric conditions
variable (h1 : incident A hA)
variable (h2 : incident A gA)
variable (h3 : incident B hB)
variable (h4 : incident B gB)
variable (h5 : incident C hA ∧ incident C gB)
variable (h6 : incident D hB ∧ incident D gA)
variable (h7 : incident E gA ∧ incident E gB)
variable (h8 : incident F hA ∧ incident F hB)
variable (h9 : incident P hB)
variable (h10 : incident H gA)
variable (h11 : ∃ CP EF, incident X CP ∧ incident X EF ∧ incident C CP ∧ incident P CP ∧ incident E EF ∧ incident F EF)
variable (h12 : ∃ EP HF, incident Y EP ∧ incident Y HF ∧ incident E EP ∧ incident P EP ∧ incident H HF ∧ incident F HF)

-- State the theorem
theorem collinearity_iff_harmonic :
  collinear X Y B ↔ harmonic A H E D :=
sorry

end NUMINAMATH_CALUDE_collinearity_iff_harmonic_l567_56758


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l567_56746

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^3 - 15*x^2 + 50*x - 60

-- Define the theorem
theorem root_sum_reciprocal (p q r A B C : ℝ) :
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →  -- p, q, r are distinct
  (poly p = 0 ∧ poly q = 0 ∧ poly r = 0) →  -- p, q, r are roots of poly
  (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    1 / (s^3 - 15*s^2 + 50*s - 60) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 135 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l567_56746


namespace NUMINAMATH_CALUDE_fraction_of_ripe_oranges_eaten_l567_56741

def total_oranges : ℕ := 96
def ripe_oranges : ℕ := total_oranges / 2
def unripe_oranges : ℕ := total_oranges - ripe_oranges
def eaten_unripe : ℕ := unripe_oranges / 8
def uneaten_oranges : ℕ := 78

theorem fraction_of_ripe_oranges_eaten :
  (total_oranges - uneaten_oranges - eaten_unripe) / ripe_oranges = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_ripe_oranges_eaten_l567_56741


namespace NUMINAMATH_CALUDE_process_termination_and_difference_l567_56747

-- Define the lists and their properties
def List1 : Type := {l : List ℕ // ∀ x ∈ l, x % 5 = 1}
def List2 : Type := {l : List ℕ // ∀ x ∈ l, x % 5 = 4}

-- Define the operation
def operation (l1 : List1) (l2 : List2) : List1 × List2 :=
  sorry

-- Define the termination condition
def is_terminated (l1 : List1) (l2 : List2) : Prop :=
  l1.val.length = 1 ∧ l2.val.length = 1

-- Theorem statement
theorem process_termination_and_difference 
  (l1_init : List1) (l2_init : List2) : 
  ∃ (l1_final : List1) (l2_final : List2),
    (is_terminated l1_final l2_final) ∧ 
    (l1_final.val.head? ≠ l2_final.val.head?) :=
  sorry

end NUMINAMATH_CALUDE_process_termination_and_difference_l567_56747


namespace NUMINAMATH_CALUDE_circle_center_l567_56700

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in the form (x - h)² + (y - k)² = r² -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The given circle equation -/
def given_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem circle_center :
  ∃ (c : Circle), (∀ x y : ℝ, given_equation x y ↔ c.equation x y) ∧ c.center = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l567_56700


namespace NUMINAMATH_CALUDE_unique_solution_for_circ_equation_l567_56778

-- Define the operation ∘
def circ (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y

-- Theorem statement
theorem unique_solution_for_circ_equation :
  ∃! y : ℝ, circ 2 y = 10 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_circ_equation_l567_56778


namespace NUMINAMATH_CALUDE_sequence_bounds_l567_56727

variable (n : ℕ)

def a : ℕ → ℚ
  | 0 => 1/2
  | k + 1 => a k + (1/n : ℚ) * (a k)^2

theorem sequence_bounds (hn : n > 0) : 1 - 1/n < a n n ∧ a n n < 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_bounds_l567_56727


namespace NUMINAMATH_CALUDE_smallest_sum_ABAb_l567_56709

/-- Represents a digit in base 4 -/
def Base4Digit := Fin 4

theorem smallest_sum_ABAb (A B : Base4Digit) (b : ℕ) : 
  A ≠ B →
  b > 5 →
  16 * A.val + 4 * B.val + A.val = 3 * b + 3 →
  ∀ (A' B' : Base4Digit) (b' : ℕ),
    A' ≠ B' →
    b' > 5 →
    16 * A'.val + 4 * B'.val + A'.val = 3 * b' + 3 →
    A.val + B.val + b ≤ A'.val + B'.val + b' →
  A.val + B.val + b = 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_ABAb_l567_56709


namespace NUMINAMATH_CALUDE_min_fence_length_is_650_l567_56769

/-- Represents a triangular grid with side length 50 meters -/
structure TriangularGrid where
  side_length : ℝ
  side_length_eq : side_length = 50

/-- Represents the number of paths between cabbage and goat areas -/
def num_paths : ℕ := 13

/-- The minimum total length of fences required to separate cabbage from goats -/
def min_fence_length (grid : TriangularGrid) : ℝ :=
  (num_paths : ℝ) * grid.side_length

/-- Theorem stating the minimum fence length required -/
theorem min_fence_length_is_650 (grid : TriangularGrid) :
  min_fence_length grid = 650 := by
  sorry

#check min_fence_length_is_650

end NUMINAMATH_CALUDE_min_fence_length_is_650_l567_56769


namespace NUMINAMATH_CALUDE_airport_exchange_rate_fraction_l567_56745

def official_rate : ℚ := 5 / 1
def willie_euros : ℚ := 70
def airport_dollars : ℚ := 10

theorem airport_exchange_rate_fraction : 
  (airport_dollars / (willie_euros / official_rate)) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_airport_exchange_rate_fraction_l567_56745


namespace NUMINAMATH_CALUDE_difference_implies_70_l567_56734

/-- Represents a two-digit numeral -/
structure TwoDigitNumeral where
  tens : Nat
  ones : Nat
  tens_lt_10 : tens < 10
  ones_lt_10 : ones < 10

/-- The place value of a digit in a two-digit numeral -/
def placeValue (n : TwoDigitNumeral) (d : Nat) : Nat :=
  if d = n.tens then 10 * n.tens else n.ones

/-- The face value of a digit -/
def faceValue (d : Nat) : Nat := d

/-- The theorem stating that if the difference between the place value and face value
    of 7 in a two-digit numeral is 63, then the numeral is 70 -/
theorem difference_implies_70 (n : TwoDigitNumeral) :
  placeValue n 7 - faceValue 7 = 63 → n.tens = 7 ∧ n.ones = 0 := by
  sorry

#check difference_implies_70

end NUMINAMATH_CALUDE_difference_implies_70_l567_56734


namespace NUMINAMATH_CALUDE_simplify_expression_solve_inequality_system_l567_56726

-- Part 1: Simplification
theorem simplify_expression (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (2 - (x - 1) / (x + 2)) / ((x^2 + 10*x + 25) / (x^2 - 4)) = (x - 2) / (x + 5) := by
  sorry

-- Part 2: Inequality System
theorem solve_inequality_system (x : ℝ) :
  (2*x + 7 > 3 ∧ (x + 1) / 3 > (x - 1) / 2) ↔ -2 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_solve_inequality_system_l567_56726


namespace NUMINAMATH_CALUDE_ratio_problem_l567_56732

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 5) :
  d / a = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l567_56732


namespace NUMINAMATH_CALUDE_factorial_ratio_l567_56771

theorem factorial_ratio (n : ℕ) (h : n > 0) : (Nat.factorial n) / (Nat.factorial (n - 1)) = n := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l567_56771


namespace NUMINAMATH_CALUDE_triangle_k_values_l567_56780

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the vectors
def vectorAB (t : Triangle) : ℝ × ℝ := (t.B.1 - t.A.1, t.B.2 - t.A.2)
def vectorAC (t : Triangle) (k : ℝ) : ℝ × ℝ := (t.C.1 - t.A.1, t.C.2 - t.A.2)

-- Define the dot product
def dotProduct (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition for a right angle
def hasRightAngle (t : Triangle) (k : ℝ) : Prop :=
  dotProduct (vectorAB t) (vectorAC t k) = 0 ∨
  dotProduct (vectorAB t) (1, k - 3) = 0 ∨
  dotProduct (vectorAC t k) ((-1 : ℝ), k - 3) = 0

-- The main theorem
theorem triangle_k_values (t : Triangle) (k : ℝ) 
  (h1 : vectorAB t = (2, 3))
  (h2 : vectorAC t k = (1, k))
  (h3 : hasRightAngle t k) :
  k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 13)/2 ∨ k = (3 - Real.sqrt 13)/2 :=
sorry

end NUMINAMATH_CALUDE_triangle_k_values_l567_56780


namespace NUMINAMATH_CALUDE_break_even_point_l567_56721

/-- The break-even point for a plastic handle molding company -/
theorem break_even_point
  (cost_per_handle : ℝ)
  (fixed_cost : ℝ)
  (selling_price : ℝ)
  (h1 : cost_per_handle = 0.60)
  (h2 : fixed_cost = 7640)
  (h3 : selling_price = 4.60) :
  ∃ x : ℕ, x = 1910 ∧ selling_price * x = fixed_cost + cost_per_handle * x :=
by sorry

end NUMINAMATH_CALUDE_break_even_point_l567_56721


namespace NUMINAMATH_CALUDE_function_solution_l567_56719

open Real

-- Define the function property
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x ≠ 0, f (1 / x) + (5 / x) * f x = 3 / x^3

-- State the theorem
theorem function_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
    ∀ x ≠ 0, f x = 5 / (8 * x^2) - x^3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_function_solution_l567_56719


namespace NUMINAMATH_CALUDE_adults_trekking_l567_56779

/-- The number of adults who went for trekking -/
def numAdults : ℕ := 56

/-- The number of children who went for trekking -/
def numChildren : ℕ := 70

/-- The number of adults the meal can feed -/
def mealAdults : ℕ := 70

/-- The number of children the meal can feed -/
def mealChildren : ℕ := 90

/-- The number of adults who have already eaten -/
def adultsEaten : ℕ := 14

/-- The number of children that can be fed with remaining food after some adults eat -/
def remainingChildren : ℕ := 72

theorem adults_trekking :
  numAdults = mealAdults - adultsEaten ∧
  numChildren = 70 ∧
  mealAdults = 70 ∧
  mealChildren = 90 ∧
  adultsEaten = 14 ∧
  remainingChildren = 72 ∧
  mealChildren = remainingChildren + adultsEaten * mealChildren / mealAdults :=
by sorry

end NUMINAMATH_CALUDE_adults_trekking_l567_56779


namespace NUMINAMATH_CALUDE_mike_seashell_count_l567_56720

/-- The number of seashells Mike found initially -/
def initial_seashells : ℝ := 6.0

/-- The number of seashells Mike found later -/
def later_seashells : ℝ := 4.0

/-- The total number of seashells Mike found -/
def total_seashells : ℝ := initial_seashells + later_seashells

theorem mike_seashell_count : total_seashells = 10.0 := by
  sorry

end NUMINAMATH_CALUDE_mike_seashell_count_l567_56720


namespace NUMINAMATH_CALUDE_smallest_reducible_even_l567_56774

def is_reducible (n : ℕ) : Prop :=
  ∃ (k : ℕ), k > 1 ∧ (15 * n - 7) % k = 0 ∧ (22 * n - 5) % k = 0

theorem smallest_reducible_even : 
  (∀ n : ℕ, n > 2013 → n % 2 = 0 → is_reducible n → n ≥ 2144) ∧ 
  (2144 > 2013 ∧ 2144 % 2 = 0 ∧ is_reducible 2144) :=
sorry

end NUMINAMATH_CALUDE_smallest_reducible_even_l567_56774


namespace NUMINAMATH_CALUDE_simplified_expression_equals_22_5_l567_56712

theorem simplified_expression_equals_22_5 : 
  1.5 * (Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + Real.sqrt (1+3+5+7+9)) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_22_5_l567_56712


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l567_56793

/-- The eccentricity of a hyperbola with special properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    (x = c ∨ x = -c) → 
    (y = b^2 / a ∨ y = -b^2 / a)) →
  2 * c = 2 * b^2 / a →
  c^2 = a^2 * (e^2 - 1) →
  e = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l567_56793


namespace NUMINAMATH_CALUDE_domain_of_sqrt_sin_minus_cos_l567_56765

open Real

theorem domain_of_sqrt_sin_minus_cos (x : ℝ) :
  (∃ y : ℝ, y = Real.sqrt (sin x - cos x)) ↔
  (∃ k : ℤ, 2 * k * π + π / 4 ≤ x ∧ x ≤ 2 * k * π + 5 * π / 4) :=
by sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_sin_minus_cos_l567_56765


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l567_56761

/-- Given points A, B, C, and D where D is the midpoint of AB, 
    prove that the sum of the slope and y-intercept of line CD is 27/10 -/
theorem line_slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 8) → 
  B = (0, -2) → 
  C = (10, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let slope := (C.2 - D.2) / (C.1 - D.1)
  let y_intercept := D.2
  slope + y_intercept = 27 / 10 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l567_56761


namespace NUMINAMATH_CALUDE_cos_36_minus_cos_72_eq_half_l567_56742

theorem cos_36_minus_cos_72_eq_half :
  Real.cos (36 * π / 180) - Real.cos (72 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_minus_cos_72_eq_half_l567_56742


namespace NUMINAMATH_CALUDE_worker_a_payment_share_l567_56736

/-- Calculates the share of payment for worker A given the total days needed for each worker and the total payment -/
def worker_a_share (days_a days_b : ℕ) (total_payment : ℚ) : ℚ :=
  let work_rate_a := 1 / days_a
  let work_rate_b := 1 / days_b
  let combined_rate := work_rate_a + work_rate_b
  let a_share_ratio := work_rate_a / combined_rate
  a_share_ratio * total_payment

/-- Theorem stating that worker A's share is 89.55 given the problem conditions -/
theorem worker_a_payment_share :
  worker_a_share 12 18 (149.25 : ℚ) = (8955 : ℚ) / 100 := by
  sorry

#eval worker_a_share 12 18 (149.25 : ℚ)

end NUMINAMATH_CALUDE_worker_a_payment_share_l567_56736


namespace NUMINAMATH_CALUDE_number_difference_problem_l567_56773

theorem number_difference_problem : ∃ x : ℚ, x - (3/5) * x = 60 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_problem_l567_56773


namespace NUMINAMATH_CALUDE_rental_problem_l567_56795

/-- Calculates the number of days a house can be rented given the daily rate, 14-day rate, and total cost. -/
def daysRented (dailyRate : ℚ) (fourteenDayRate : ℚ) (totalCost : ℚ) : ℕ :=
  sorry

/-- Theorem stating that given the specific rates and total cost, the number of days rented is 20. -/
theorem rental_problem :
  daysRented 50 500 800 = 20 :=
sorry

end NUMINAMATH_CALUDE_rental_problem_l567_56795
