import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_base_length_l3944_394483

/-- The length of the base of an isosceles triangle, given specific conditions -/
theorem isosceles_base_length 
  (equilateral_perimeter : ℝ) 
  (isosceles_perimeter : ℝ) 
  (h1 : equilateral_perimeter = 45) 
  (h2 : isosceles_perimeter = 40) : ℝ :=
by
  -- The length of the base of the isosceles triangle is 10
  sorry

#check isosceles_base_length

end NUMINAMATH_CALUDE_isosceles_base_length_l3944_394483


namespace NUMINAMATH_CALUDE_distilled_water_amount_l3944_394459

/-- Given the initial mixture ratios and the required amount of final solution,
    prove that the amount of distilled water needed is 0.2 liters. -/
theorem distilled_water_amount
  (nutrient_concentrate : ℝ)
  (initial_distilled_water : ℝ)
  (initial_total_solution : ℝ)
  (required_solution : ℝ)
  (h1 : nutrient_concentrate = 0.05)
  (h2 : initial_distilled_water = 0.025)
  (h3 : initial_total_solution = 0.075)
  (h4 : required_solution = 0.6)
  (h5 : initial_total_solution = nutrient_concentrate + initial_distilled_water) :
  (required_solution * (initial_distilled_water / initial_total_solution)) = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_distilled_water_amount_l3944_394459


namespace NUMINAMATH_CALUDE_marble_distribution_l3944_394493

theorem marble_distribution (total_marbles : ℕ) (num_children : ℕ) 
  (h1 : total_marbles = 60) 
  (h2 : num_children = 7) : 
  (num_children - (total_marbles % num_children)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l3944_394493


namespace NUMINAMATH_CALUDE_billy_soda_theorem_l3944_394410

def billy_soda_distribution (num_sisters : ℕ) (soda_pack : ℕ) : Prop :=
  let num_brothers := 2 * num_sisters
  let total_siblings := num_brothers + num_sisters
  let sodas_per_sibling := soda_pack / total_siblings
  (num_sisters = 2) ∧ (soda_pack = 12) → (sodas_per_sibling = 2)

theorem billy_soda_theorem : billy_soda_distribution 2 12 := by
  sorry

end NUMINAMATH_CALUDE_billy_soda_theorem_l3944_394410


namespace NUMINAMATH_CALUDE_event_probability_l3944_394424

theorem event_probability (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →
  (1 - p)^3 = 1 - 63/64 →
  3 * p * (1 - p)^2 = 9/64 :=
by sorry

end NUMINAMATH_CALUDE_event_probability_l3944_394424


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3944_394418

theorem sum_of_fractions : 
  let a := 1 + 3 + 5
  let b := 2 + 4 + 6
  (a / b) + (b / a) = 25 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3944_394418


namespace NUMINAMATH_CALUDE_length_AE_l3944_394438

-- Define the circle
def Circle := {c : ℝ × ℝ | c.1^2 + c.2^2 = 4}

-- Define points A, B, C, D, E
variable (A B C D E : ℝ × ℝ)

-- AB is a diameter of the circle
axiom diam : A ∈ Circle ∧ B ∈ Circle ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16

-- ABC is an equilateral triangle
axiom equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

-- D is the intersection of the circle and AC
axiom D_on_circle : D ∈ Circle
axiom D_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)

-- E is the intersection of the circle and BC
axiom E_on_circle : E ∈ Circle
axiom E_on_BC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (s * B.1 + (1 - s) * C.1, s * B.2 + (1 - s) * C.2)

-- Theorem: The length of AE is 2√3
theorem length_AE : (A.1 - E.1)^2 + (A.2 - E.2)^2 = 12 := by sorry

end NUMINAMATH_CALUDE_length_AE_l3944_394438


namespace NUMINAMATH_CALUDE_circle_central_symmetry_l3944_394499

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Central symmetry for a figure in 2D plane --/
def CentralSymmetry (F : Set (ℝ × ℝ)) :=
  ∃ c : ℝ × ℝ, ∀ p : ℝ × ℝ, p ∈ F → (2 * c.1 - p.1, 2 * c.2 - p.2) ∈ F

/-- The set of points in a circle --/
def CirclePoints (c : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2}

/-- Theorem: A circle has central symmetry --/
theorem circle_central_symmetry (c : Circle) : CentralSymmetry (CirclePoints c) := by
  sorry


end NUMINAMATH_CALUDE_circle_central_symmetry_l3944_394499


namespace NUMINAMATH_CALUDE_triangle_properties_l3944_394481

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  c * (Real.cos A) + Real.sqrt 3 * c * (Real.sin A) - b - a = 0 →
  (C = Real.pi / 3 ∧
   (c = 1 → ∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' > c →
     1 / 2 * a' * b' * Real.sin C ≤ Real.sqrt 3 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3944_394481


namespace NUMINAMATH_CALUDE_train_length_l3944_394498

/-- The length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man completely. -/
theorem train_length (train_speed man_speed : ℝ) (time_to_cross : ℝ) :
  train_speed = 54.99520038396929 →
  man_speed = 5 →
  time_to_cross = 6 →
  let relative_speed := (train_speed + man_speed) * (1000 / 3600)
  let train_length := relative_speed * time_to_cross
  train_length = 99.99180063994882 := by
sorry

end NUMINAMATH_CALUDE_train_length_l3944_394498


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3944_394440

theorem polynomial_remainder : ∀ x : ℝ, 
  (4 * x^8 - 3 * x^6 - 6 * x^4 + x^3 + 5 * x^2 - 9) = 
  (x - 1) * (4 * x^7 + 4 * x^6 + x^5 - 2 * x^4 - 2 * x^3 + 4 * x^2 + 4 * x + 4) + (-9) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3944_394440


namespace NUMINAMATH_CALUDE_five_workers_completion_time_l3944_394420

/-- The productivity rates of five workers -/
structure WorkerRates where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ
  x₅ : ℝ

/-- The total amount of work to be done -/
def total_work : ℝ → ℝ := id

theorem five_workers_completion_time 
  (rates : WorkerRates) 
  (y : ℝ) 
  (h₁ : rates.x₁ + rates.x₂ + rates.x₃ = y / 327.5)
  (h₂ : rates.x₁ + rates.x₃ + rates.x₅ = y / 5)
  (h₃ : rates.x₁ + rates.x₃ + rates.x₄ = y / 6)
  (h₄ : rates.x₂ + rates.x₄ + rates.x₅ = y / 4) :
  y / (rates.x₁ + rates.x₂ + rates.x₃ + rates.x₄ + rates.x₅) = 3 := by
  sorry

end NUMINAMATH_CALUDE_five_workers_completion_time_l3944_394420


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3944_394422

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset Int), (∀ n : Int, (n - 3) * (n + 5) * (n - 1) < 0 ↔ n ∈ S) ∧ Finset.card S = 6 :=
by sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3944_394422


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l3944_394432

theorem quadratic_inequality_always_positive (r : ℝ) :
  (∀ x : ℝ, (r^2 - 1) * x^2 + 2 * (r - 1) * x + 1 > 0) ↔ r > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l3944_394432


namespace NUMINAMATH_CALUDE_nail_polish_drying_time_l3944_394488

theorem nail_polish_drying_time (total_time color_coat_time top_coat_time : ℕ) 
  (h1 : total_time = 13)
  (h2 : color_coat_time = 3)
  (h3 : top_coat_time = 5) :
  total_time - (2 * color_coat_time + top_coat_time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nail_polish_drying_time_l3944_394488


namespace NUMINAMATH_CALUDE_ninth_power_negative_fourth_l3944_394469

theorem ninth_power_negative_fourth : (1 / 9)^(-1/4 : ℝ) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ninth_power_negative_fourth_l3944_394469


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3944_394411

theorem inequality_equivalence (x : ℝ) : (x - 2)^2 < 9 ↔ -1 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3944_394411


namespace NUMINAMATH_CALUDE_infinite_non_sum_of_three_cubes_l3944_394463

theorem infinite_non_sum_of_three_cubes :
  ∀ k : ℤ, ¬∃ a b c : ℤ, (9*k + 4 = a^3 + b^3 + c^3) ∧ ¬∃ a b c : ℤ, (9*k - 4 = a^3 + b^3 + c^3) :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_non_sum_of_three_cubes_l3944_394463


namespace NUMINAMATH_CALUDE_worker_y_fraction_l3944_394454

theorem worker_y_fraction (P : ℝ) (Px Py : ℝ) (h1 : P > 0) (h2 : Px ≥ 0) (h3 : Py ≥ 0) :
  Px + Py = P →
  0.005 * Px + 0.008 * Py = 0.007 * P →
  Py / P = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_worker_y_fraction_l3944_394454


namespace NUMINAMATH_CALUDE_positive_sum_and_product_imply_positive_quadratic_root_conditions_quadratic_root_conditions_not_sufficient_l3944_394408

-- Statement 3
theorem positive_sum_and_product_imply_positive (a b : ℝ) :
  a + b > 0 → a * b > 0 → a > 0 ∧ b > 0 := by sorry

-- Statement 4
def has_two_distinct_positive_roots (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0

theorem quadratic_root_conditions (a b c : ℝ) (h : a ≠ 0) :
  has_two_distinct_positive_roots a b c →
  b / a < 0 ∧ c / a > 0 := by sorry

theorem quadratic_root_conditions_not_sufficient (a b c : ℝ) (h : a ≠ 0) :
  b / a < 0 ∧ c / a > 0 →
  ¬(has_two_distinct_positive_roots a b c ↔ True) := by sorry

end NUMINAMATH_CALUDE_positive_sum_and_product_imply_positive_quadratic_root_conditions_quadratic_root_conditions_not_sufficient_l3944_394408


namespace NUMINAMATH_CALUDE_jia_zi_second_occurrence_l3944_394443

/-- The number of Heavenly Stems -/
def heavenly_stems : ℕ := 10

/-- The number of Earthly Branches -/
def earthly_branches : ℕ := 12

/-- The column number when Jia and Zi are in the same column for the second time -/
def second_occurrence : ℕ := 61

/-- Proves that the column number when Jia and Zi are in the same column for the second time is 61 -/
theorem jia_zi_second_occurrence :
  second_occurrence = Nat.lcm heavenly_stems earthly_branches + 1 := by
  sorry

end NUMINAMATH_CALUDE_jia_zi_second_occurrence_l3944_394443


namespace NUMINAMATH_CALUDE_function_range_and_inequality_l3944_394437

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp x * Real.sin x

theorem function_range_and_inequality (e : ℝ) (π : ℝ) :
  (∀ x ∈ Set.Icc 0 (π / 2), f x ∈ Set.Icc 0 1) ∧
  (∃ k : ℝ, k = Real.exp (π / 2) / (π / 2 - 1) ∧
    ∀ x ∈ Set.Icc 0 (π / 2), f x ≥ k * (x - 1) * (1 - Real.sin x) ∧
    ∀ k' > k, ∃ x ∈ Set.Icc 0 (π / 2), f x < k' * (x - 1) * (1 - Real.sin x)) :=
by sorry

end NUMINAMATH_CALUDE_function_range_and_inequality_l3944_394437


namespace NUMINAMATH_CALUDE_b_paisa_per_a_rupee_l3944_394435

-- Define the total sum of money in rupees
def total_sum : ℚ := 164

-- Define C's share in rupees
def c_share : ℚ := 32

-- Define the ratio of C's paisa to A's rupees
def c_to_a_ratio : ℚ := 40 / 100

-- Define A's share in rupees
def a_share : ℚ := c_share / c_to_a_ratio

-- Define B's share in paisa
def b_share : ℚ := (total_sum - a_share - c_share) * 100

-- Theorem to prove
theorem b_paisa_per_a_rupee : b_share / a_share = 65 := by
  sorry

end NUMINAMATH_CALUDE_b_paisa_per_a_rupee_l3944_394435


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l3944_394405

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l3944_394405


namespace NUMINAMATH_CALUDE_not_good_pair_3_3_l3944_394450

/-- A pair of natural numbers is good if there exists a polynomial with integer coefficients and distinct integers satisfying certain conditions. -/
def is_good_pair (r s : ℕ) : Prop :=
  ∃ (P : ℤ → ℤ) (a b : Fin r → ℤ) (c d : Fin s → ℤ),
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    (∀ i j, i ≠ j → c i ≠ c j) ∧
    (∀ i j, a i ≠ c j) ∧
    (∀ i, P (a i) = 2) ∧
    (∀ i, P (c i) = 5) ∧
    (∀ x y : ℤ, (x - y) ∣ (P x - P y))

/-- Theorem stating that (3, 3) is not a good pair. -/
theorem not_good_pair_3_3 : ¬ is_good_pair 3 3 := by
  sorry

end NUMINAMATH_CALUDE_not_good_pair_3_3_l3944_394450


namespace NUMINAMATH_CALUDE_elf_goblin_theorem_l3944_394431

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of valid arrangements of elves and goblins -/
def elf_goblin_arrangements (n : ℕ) : ℕ := fib (n + 1)

/-- Theorem: The number of valid arrangements of n elves and n goblins,
    where no two goblins can be adjacent, is equal to the (n+2)th Fibonacci number -/
theorem elf_goblin_theorem (n : ℕ) :
  elf_goblin_arrangements n = fib (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_elf_goblin_theorem_l3944_394431


namespace NUMINAMATH_CALUDE_john_initial_money_l3944_394414

/-- Represents John's financial transactions and final balance --/
def john_money (initial spent allowance final : ℕ) : Prop :=
  initial - spent + allowance = final

/-- Proves that John's initial money was $5 --/
theorem john_initial_money : 
  ∃ (initial : ℕ), john_money initial 2 26 29 ∧ initial = 5 := by
  sorry

end NUMINAMATH_CALUDE_john_initial_money_l3944_394414


namespace NUMINAMATH_CALUDE_iron_wire_remainder_l3944_394406

theorem iron_wire_remainder (total_length : ℚ) : 
  total_length > 0 → 
  total_length - (2/9 * total_length) - (3/9 * total_length) = 4/9 * total_length := by
sorry

end NUMINAMATH_CALUDE_iron_wire_remainder_l3944_394406


namespace NUMINAMATH_CALUDE_odd_sum_prob_is_five_thirteenths_l3944_394473

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of even sides on each die -/
def num_even_sides : ℕ := 3

/-- The number of odd sides on each die -/
def num_odd_sides : ℕ := 3

/-- The total number of possible outcomes when rolling the dice -/
def total_outcomes : ℕ := num_sides ^ num_dice

/-- The number of outcomes where all dice show odd numbers -/
def all_odd_outcomes : ℕ := num_odd_sides ^ num_dice

/-- The number of outcomes where the product of dice values is even -/
def even_product_outcomes : ℕ := total_outcomes - all_odd_outcomes

/-- The probability of rolling an odd sum given an even product -/
def prob_odd_sum_given_even_product : ℚ := 5 / 13

theorem odd_sum_prob_is_five_thirteenths :
  prob_odd_sum_given_even_product = 5 / 13 := by sorry

end NUMINAMATH_CALUDE_odd_sum_prob_is_five_thirteenths_l3944_394473


namespace NUMINAMATH_CALUDE_u_2023_equals_3_l3944_394466

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 1
| 4 => 2
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence u
def u : ℕ → ℕ
| 0 => 5
| (n + 1) => g (u n)

-- Theorem statement
theorem u_2023_equals_3 : u 2023 = 3 := by
  sorry

end NUMINAMATH_CALUDE_u_2023_equals_3_l3944_394466


namespace NUMINAMATH_CALUDE_hot_dog_stand_mayo_bottles_l3944_394467

/-- Given a ratio of ketchup : mustard : mayo bottles and the number of ketchup bottles,
    calculate the number of mayo bottles -/
def mayo_bottles (ketchup_ratio mustard_ratio mayo_ratio ketchup_bottles : ℕ) : ℕ :=
  (mayo_ratio * ketchup_bottles) / ketchup_ratio

/-- Theorem: Given the ratio 3:3:2 for ketchup:mustard:mayo and 6 ketchup bottles,
    there are 4 mayo bottles -/
theorem hot_dog_stand_mayo_bottles :
  mayo_bottles 3 3 2 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_stand_mayo_bottles_l3944_394467


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l3944_394484

/-- A hexagon is a polygon with 6 sides. -/
def Hexagon : Type := Unit

/-- The number of sides in a hexagon. -/
def num_sides (h : Hexagon) : ℕ := 6

/-- The number of diagonals in a polygon. -/
def num_diagonals (h : Hexagon) : ℕ := sorry

/-- Theorem: The number of diagonals in a hexagon is 9. -/
theorem hexagon_diagonals (h : Hexagon) : num_diagonals h = 9 := by sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l3944_394484


namespace NUMINAMATH_CALUDE_hexagon_walk_l3944_394480

/-- A regular hexagon with side length 3 km -/
structure RegularHexagon where
  sideLength : ℝ
  is_regular : sideLength = 3

/-- A point on the perimeter of the hexagon, represented by the distance traveled from a corner -/
def PerimeterPoint (h : RegularHexagon) (distance : ℝ) : ℝ × ℝ :=
  sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

theorem hexagon_walk (h : RegularHexagon) :
  let start := (0, 0)
  let end_point := PerimeterPoint h 8
  distance start end_point = 1 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_walk_l3944_394480


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3944_394433

theorem arithmetic_expression_equality : 6 + 18 / 3 - 4 * 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3944_394433


namespace NUMINAMATH_CALUDE_work_hours_per_day_l3944_394446

theorem work_hours_per_day (days : ℕ) (total_hours : ℕ) (h1 : days = 5) (h2 : total_hours = 40) :
  total_hours / days = 8 :=
by sorry

end NUMINAMATH_CALUDE_work_hours_per_day_l3944_394446


namespace NUMINAMATH_CALUDE_rachel_class_selection_l3944_394417

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem rachel_class_selection :
  let total_classes : ℕ := 10
  let mandatory_classes : ℕ := 2
  let classes_to_choose : ℕ := 5
  let remaining_classes := total_classes - mandatory_classes
  let additional_classes := classes_to_choose - mandatory_classes
  choose remaining_classes additional_classes = 56 := by sorry

end NUMINAMATH_CALUDE_rachel_class_selection_l3944_394417


namespace NUMINAMATH_CALUDE_parallel_line_slope_l3944_394457

theorem parallel_line_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let given_line := {(x, y) : ℝ × ℝ | a * x - b * y = c}
  let slope := a / b
  ∀ m : ℝ, (∃ k : ℝ, ∀ x y : ℝ, y = m * x + k ↔ (x, y) ∈ given_line) → m = slope :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l3944_394457


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l3944_394407

theorem square_perimeter_from_area (s : Real) (area : Real) (perimeter : Real) :
  (s ^ 2 = area) → (area = 36) → (perimeter = 4 * s) → (perimeter = 24) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l3944_394407


namespace NUMINAMATH_CALUDE_tangent_circle_intersection_theorem_l3944_394497

/-- A circle with center on y = 4x, tangent to x + y - 2 = 0 at (1,1) --/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : center.2 = 4 * center.1
  tangent_at_point : (1 : ℝ) + 1 - 2 = 0
  tangent_condition : (center.1 - 1)^2 + (center.2 - 1)^2 = radius^2

/-- The equation of the circle --/
def circle_equation (c : TangentCircle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The intersecting line --/
def intersecting_line (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y + 3 = 0

/-- Points A and B are on both the circle and the line --/
def intersection_points (c : TangentCircle) (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  circle_equation c A.1 A.2 ∧ circle_equation c B.1 B.2 ∧
  intersecting_line k A.1 A.2 ∧ intersecting_line k B.1 B.2

/-- Point M on the circle with OM = OA + OB --/
def point_M (c : TangentCircle) (A B M : ℝ × ℝ) : Prop :=
  circle_equation c M.1 M.2 ∧ M.1 = A.1 + B.1 ∧ M.2 = A.2 + B.2

/-- The main theorem --/
theorem tangent_circle_intersection_theorem (c : TangentCircle) 
  (k : ℝ) (A B M : ℝ × ℝ) :
  intersection_points c k A B → point_M c A B M → k^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_intersection_theorem_l3944_394497


namespace NUMINAMATH_CALUDE_min_value_reciprocal_product_l3944_394475

theorem min_value_reciprocal_product (a b : ℝ) 
  (h1 : a + a * b + 2 * b = 30) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  ∀ x y : ℝ, x > 0 → y > 0 → x + x * y + 2 * y = 30 → 1 / (a * b) ≤ 1 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_product_l3944_394475


namespace NUMINAMATH_CALUDE_polynomial_equality_l3944_394403

/-- Given that 2x^5 - 4x^3 + 3x^2 + g(x) = 7x^4 - 5x^3 + x^2 - 9x + 2,
    prove that g(x) = -2x^5 + 7x^4 - x^3 - 2x^2 - 9x + 2 -/
theorem polynomial_equality (x : ℝ) (g : ℝ → ℝ) : 
  2*x^5 - 4*x^3 + 3*x^2 + g x = 7*x^4 - 5*x^3 + x^2 - 9*x + 2 → 
  g x = -2*x^5 + 7*x^4 - x^3 - 2*x^2 - 9*x + 2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3944_394403


namespace NUMINAMATH_CALUDE_no_right_triangle_sine_roots_l3944_394402

theorem no_right_triangle_sine_roots (k : ℝ) : 
  ¬ (∃ x₁ x₂ : ℝ, 
    (8 * x₁^2 + 6 * k * x₁ + 2 * k + 1 = 0) ∧ 
    (8 * x₂^2 + 6 * k * x₂ + 2 * k + 1 = 0) ∧
    (0 < x₁ ∧ x₁ < 1) ∧ 
    (0 < x₂ ∧ x₂ < 1) ∧ 
    (x₁ + x₂ ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_right_triangle_sine_roots_l3944_394402


namespace NUMINAMATH_CALUDE_negation_equivalence_l3944_394472

theorem negation_equivalence :
  (¬ ∃ x₀ > 0, x₀^2 - 5*x₀ + 6 > 0) ↔ (∀ x > 0, x^2 - 5*x + 6 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3944_394472


namespace NUMINAMATH_CALUDE_value_of_y_l3944_394479

theorem value_of_y (y : ℝ) (h : 2/3 - 1/4 = 4/y) : y = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l3944_394479


namespace NUMINAMATH_CALUDE_athlete_A_second_day_prob_l3944_394453

-- Define the probabilities
def prob_A_first_day : ℝ := 0.5
def prob_B_first_day : ℝ := 0.5
def prob_A_second_day_given_A_first : ℝ := 0.6
def prob_A_second_day_given_B_first : ℝ := 0.5

-- State the theorem
theorem athlete_A_second_day_prob :
  prob_A_first_day * prob_A_second_day_given_A_first +
  prob_B_first_day * prob_A_second_day_given_B_first = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_athlete_A_second_day_prob_l3944_394453


namespace NUMINAMATH_CALUDE_marias_car_trip_l3944_394452

theorem marias_car_trip (D : ℝ) : 
  (D / 2 / 4 / 3 + D / 2 / 4 * 2 / 3 + D / 2 * 3 / 4) = 630 → D = 840 := by
  sorry

end NUMINAMATH_CALUDE_marias_car_trip_l3944_394452


namespace NUMINAMATH_CALUDE_min_pairs_for_flashlight_l3944_394477

/-- Represents the minimum number of pairs to test to guarantee finding a working pair of batteries -/
def min_pairs_to_test (total_batteries : ℕ) (working_batteries : ℕ) : ℕ :=
  total_batteries / 2 - working_batteries / 2 + 1

/-- Theorem stating the minimum number of pairs to test for the given problem -/
theorem min_pairs_for_flashlight :
  min_pairs_to_test 8 4 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_pairs_for_flashlight_l3944_394477


namespace NUMINAMATH_CALUDE_num_grade_assignments_l3944_394460

/-- The number of students in the class -/
def num_students : ℕ := 10

/-- The number of possible grades (A, B, C) -/
def num_grades : ℕ := 3

/-- Theorem: The number of ways to assign grades to all students -/
theorem num_grade_assignments : (num_grades ^ num_students : ℕ) = 59049 := by
  sorry

end NUMINAMATH_CALUDE_num_grade_assignments_l3944_394460


namespace NUMINAMATH_CALUDE_point_equal_distance_to_axes_l3944_394491

/-- A point P with coordinates (m-4, 2m+7) has equal distance from both coordinate axes if and only if m = -11 or m = -1 -/
theorem point_equal_distance_to_axes (m : ℝ) : 
  |m - 4| = |2*m + 7| ↔ m = -11 ∨ m = -1 := by
sorry

end NUMINAMATH_CALUDE_point_equal_distance_to_axes_l3944_394491


namespace NUMINAMATH_CALUDE_right_pentagonal_prism_characterization_cone_characterization_l3944_394421

-- Define geometric shapes
def RightPentagonalPrism : Type := sorry
def Cone : Type := sorry

-- Define properties of shapes
def has_seven_faces (shape : Type) : Prop := sorry
def has_two_parallel_congruent_pentagons (shape : Type) : Prop := sorry
def has_congruent_rectangle_faces (shape : Type) : Prop := sorry
def formed_by_rotating_isosceles_triangle (shape : Type) : Prop := sorry
def rotated_180_degrees (shape : Type) : Prop := sorry
def rotated_around_height_line (shape : Type) : Prop := sorry

-- Theorem 1
theorem right_pentagonal_prism_characterization (shape : Type) :
  has_seven_faces shape ∧
  has_two_parallel_congruent_pentagons shape ∧
  has_congruent_rectangle_faces shape →
  shape = RightPentagonalPrism :=
sorry

-- Theorem 2
theorem cone_characterization (shape : Type) :
  formed_by_rotating_isosceles_triangle shape ∧
  rotated_180_degrees shape ∧
  rotated_around_height_line shape →
  shape = Cone :=
sorry

end NUMINAMATH_CALUDE_right_pentagonal_prism_characterization_cone_characterization_l3944_394421


namespace NUMINAMATH_CALUDE_positive_expression_l3944_394495

theorem positive_expression (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -2 < y ∧ y < 0) 
  (hz : 2 < z ∧ z < 3) : 
  y + 2*z > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l3944_394495


namespace NUMINAMATH_CALUDE_constant_term_of_given_equation_l3944_394447

/-- The quadratic equation 2x^2 - 3x - 1 = 0 -/
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 3 * x - 1 = 0

/-- The constant term of a quadratic equation ax^2 + bx + c = 0 is c -/
def constant_term (a b c : ℝ) : ℝ := c

theorem constant_term_of_given_equation :
  constant_term 2 (-3) (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_constant_term_of_given_equation_l3944_394447


namespace NUMINAMATH_CALUDE_polygon_with_16_diagonals_has_7_sides_l3944_394442

/-- The number of sides in a regular polygon with 16 diagonals -/
def num_sides_of_polygon_with_16_diagonals : ℕ := 7

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem polygon_with_16_diagonals_has_7_sides :
  num_diagonals num_sides_of_polygon_with_16_diagonals = 16 :=
by sorry

end NUMINAMATH_CALUDE_polygon_with_16_diagonals_has_7_sides_l3944_394442


namespace NUMINAMATH_CALUDE_boris_candy_problem_l3944_394461

theorem boris_candy_problem (initial_candy : ℕ) : 
  let daughter_eats : ℕ := 8
  let num_bowls : ℕ := 4
  let boris_takes_per_bowl : ℕ := 3
  let candy_left_in_one_bowl : ℕ := 20
  (initial_candy - daughter_eats) / num_bowls - boris_takes_per_bowl = candy_left_in_one_bowl →
  initial_candy = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_boris_candy_problem_l3944_394461


namespace NUMINAMATH_CALUDE_stickers_in_red_folder_l3944_394474

/-- The number of stickers on each sheet in the red folder -/
def red_stickers : ℕ := 3

/-- The number of sheets in each folder -/
def sheets_per_folder : ℕ := 10

/-- The number of stickers on each sheet in the green folder -/
def green_stickers : ℕ := 2

/-- The number of stickers on each sheet in the blue folder -/
def blue_stickers : ℕ := 1

/-- The total number of stickers used -/
def total_stickers : ℕ := 60

theorem stickers_in_red_folder :
  red_stickers * sheets_per_folder +
  green_stickers * sheets_per_folder +
  blue_stickers * sheets_per_folder = total_stickers :=
by sorry

end NUMINAMATH_CALUDE_stickers_in_red_folder_l3944_394474


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l3944_394470

theorem fraction_zero_implies_x_one (x : ℝ) (h : (x - 1) / x = 0) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l3944_394470


namespace NUMINAMATH_CALUDE_infinitely_many_N_with_same_digit_sum_l3944_394401

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a natural number is composed of digits 1 to 9 only -/
def isComposedOf1to9 (n : ℕ) : Prop := sorry

/-- The main theorem -/
theorem infinitely_many_N_with_same_digit_sum (A : ℕ) :
  ∃ f : ℕ → ℕ, Monotone f ∧ (∀ m : ℕ, 
    isComposedOf1to9 (f m) ∧ 
    sumOfDigits (f m) = sumOfDigits (A * f m)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_N_with_same_digit_sum_l3944_394401


namespace NUMINAMATH_CALUDE_solve_equation_l3944_394416

theorem solve_equation (x : ℚ) : (2 * x + 7) / 5 = 22 → x = 103 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3944_394416


namespace NUMINAMATH_CALUDE_contrapositive_theorem_l3944_394409

theorem contrapositive_theorem (x : ℝ) :
  (x = 1 ∨ x = 2 → x^2 - 3*x + 2 ≤ 0) ↔ (x^2 - 3*x + 2 > 0 → x ≠ 1 ∧ x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_theorem_l3944_394409


namespace NUMINAMATH_CALUDE_complex_arithmetic_proof_l3944_394487

theorem complex_arithmetic_proof :
  let A : ℂ := 3 + 2*Complex.I
  let B : ℂ := -5
  let C : ℂ := 2*Complex.I
  let D : ℂ := 1 + 3*Complex.I
  A - B + C - D = 7 + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_proof_l3944_394487


namespace NUMINAMATH_CALUDE_check_amount_proof_l3944_394436

theorem check_amount_proof :
  ∃! (x y : ℕ), 
    y ≤ 99 ∧
    (y : ℚ) + (x : ℚ) / 100 - 5 / 100 = 2 * ((x : ℚ) + (y : ℚ) / 100) ∧
    x = 31 ∧ y = 63 := by
  sorry

end NUMINAMATH_CALUDE_check_amount_proof_l3944_394436


namespace NUMINAMATH_CALUDE_honor_roll_fraction_l3944_394428

theorem honor_roll_fraction (female_honor : Rat) (male_honor : Rat) (female_ratio : Rat) : 
  female_honor = 7/12 →
  male_honor = 11/15 →
  female_ratio = 13/27 →
  (female_ratio * female_honor) + ((1 - female_ratio) * male_honor) = 1071/1620 := by
sorry

end NUMINAMATH_CALUDE_honor_roll_fraction_l3944_394428


namespace NUMINAMATH_CALUDE_negation_of_forall_square_geq_one_l3944_394486

theorem negation_of_forall_square_geq_one :
  ¬(∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) ↔ ∃ x : ℝ, x ≥ 1 ∧ x^2 < 1 := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_square_geq_one_l3944_394486


namespace NUMINAMATH_CALUDE_collinear_points_xy_value_l3944_394413

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point3D) : Prop :=
  ∃ t : ℝ, (r.x - p.x) = t * (q.x - p.x) ∧
            (r.y - p.y) = t * (q.y - p.y) ∧
            (r.z - p.z) = t * (q.z - p.z)

/-- The main theorem -/
theorem collinear_points_xy_value :
  ∀ (x y : ℝ),
  let A : Point3D := ⟨1, -2, 11⟩
  let B : Point3D := ⟨4, 2, 3⟩
  let C : Point3D := ⟨x, y, 15⟩
  collinear A B C → x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_xy_value_l3944_394413


namespace NUMINAMATH_CALUDE_sqrt_one_third_equals_sqrt_three_over_three_l3944_394427

theorem sqrt_one_third_equals_sqrt_three_over_three :
  Real.sqrt (1 / 3) = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_one_third_equals_sqrt_three_over_three_l3944_394427


namespace NUMINAMATH_CALUDE_circus_tent_capacity_l3944_394426

/-- The number of sections in the circus tent -/
def num_sections : ℕ := 4

/-- The capacity of each section in the circus tent -/
def section_capacity : ℕ := 246

/-- The total capacity of the circus tent -/
def total_capacity : ℕ := num_sections * section_capacity

theorem circus_tent_capacity : total_capacity = 984 := by
  sorry

end NUMINAMATH_CALUDE_circus_tent_capacity_l3944_394426


namespace NUMINAMATH_CALUDE_train_length_l3944_394490

/-- Calculates the length of a train given its speed and time to cross an electric pole. -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) : 
  speed_kmh = 50.4 → time_sec = 20 → speed_kmh * (1000 / 3600) * time_sec = 280 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3944_394490


namespace NUMINAMATH_CALUDE_abc_product_l3944_394449

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 24) (hac : a * c = 40) (hbc : b * c = 60) :
  a * b * c = 240 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l3944_394449


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3944_394458

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) : 
  (∀ x : ℝ, (1 - x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 2^7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3944_394458


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3944_394434

theorem sphere_surface_area (diameter : ℝ) (h : diameter = 10) :
  4 * Real.pi * (diameter / 2)^2 = 100 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3944_394434


namespace NUMINAMATH_CALUDE_exponent_and_polynomial_identities_l3944_394430

variable (a b : ℝ)

theorem exponent_and_polynomial_identities : 
  ((a^2)^3 / (-a)^2 = a^4) ∧ 
  ((a+2*b)*(a+b)-3*a*(a+b) = -2*a^2 + 2*b^2) := by sorry

end NUMINAMATH_CALUDE_exponent_and_polynomial_identities_l3944_394430


namespace NUMINAMATH_CALUDE_ball_hits_ground_l3944_394465

def ball_height (t : ℝ) : ℝ := -18 * t^2 + 30 * t + 60

theorem ball_hits_ground :
  ∃ t : ℝ, t > 0 ∧ ball_height t = 0 ∧ t = (5 + Real.sqrt 145) / 6 :=
sorry

end NUMINAMATH_CALUDE_ball_hits_ground_l3944_394465


namespace NUMINAMATH_CALUDE_frank_reading_days_l3944_394404

def pages_per_weekday : ℝ := 5.7
def pages_per_weekend_day : ℝ := 9.5
def total_pages : ℕ := 576
def start_day : String := "Monday"

theorem frank_reading_days : ℕ := by
  -- Prove that Frank takes 85 days to finish the book
  sorry

end NUMINAMATH_CALUDE_frank_reading_days_l3944_394404


namespace NUMINAMATH_CALUDE_green_pill_cost_calculation_l3944_394441

def green_pill_cost (total_cost : ℚ) (days : ℕ) (green_daily : ℕ) (pink_daily : ℕ) : ℚ :=
  (total_cost / days + 2 * pink_daily) / (green_daily + pink_daily)

theorem green_pill_cost_calculation :
  let total_cost : ℚ := 600
  let days : ℕ := 10
  let green_daily : ℕ := 2
  let pink_daily : ℕ := 1
  green_pill_cost total_cost days green_daily pink_daily = 62/3 := by
sorry

end NUMINAMATH_CALUDE_green_pill_cost_calculation_l3944_394441


namespace NUMINAMATH_CALUDE_car_journey_cost_l3944_394468

/-- Calculates the total cost of a car journey given various expenses -/
theorem car_journey_cost
  (rental_cost : ℝ)
  (rental_discount_percent : ℝ)
  (gas_cost_per_gallon : ℝ)
  (gas_gallons : ℝ)
  (driving_cost_per_mile : ℝ)
  (miles_driven : ℝ)
  (toll_fees : ℝ)
  (parking_cost_per_day : ℝ)
  (parking_days : ℝ)
  (h1 : rental_cost = 150)
  (h2 : rental_discount_percent = 15)
  (h3 : gas_cost_per_gallon = 3.5)
  (h4 : gas_gallons = 8)
  (h5 : driving_cost_per_mile = 0.5)
  (h6 : miles_driven = 320)
  (h7 : toll_fees = 15)
  (h8 : parking_cost_per_day = 20)
  (h9 : parking_days = 3) :
  rental_cost * (1 - rental_discount_percent / 100) +
  gas_cost_per_gallon * gas_gallons +
  driving_cost_per_mile * miles_driven +
  toll_fees +
  parking_cost_per_day * parking_days = 390.5 := by
  sorry


end NUMINAMATH_CALUDE_car_journey_cost_l3944_394468


namespace NUMINAMATH_CALUDE_min_value_expression_l3944_394464

theorem min_value_expression (x : ℝ) (h : x > 1) :
  x + 9 / x - 2 ≥ 4 ∧ ∃ y > 1, y + 9 / y - 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3944_394464


namespace NUMINAMATH_CALUDE_trailing_zeros_factorial_product_mod_100_l3944_394485

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- The product of factorials from 1 to n -/
def factorialProduct (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * Nat.factorial (i + 1)) 1

theorem trailing_zeros_factorial_product_mod_100 :
  trailingZeros (factorialProduct 50) % 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_factorial_product_mod_100_l3944_394485


namespace NUMINAMATH_CALUDE_contest_questions_l3944_394492

theorem contest_questions (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ (a b c : ℕ), 10 < a ∧ a ≤ b ∧ b ≤ c ∧ c < 13) 
  (h3 : 4 * n = 10 + 13 + a + b + c) : n = 14 := by
  sorry

end NUMINAMATH_CALUDE_contest_questions_l3944_394492


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_inverse_trig_functions_l3944_394451

theorem min_value_of_sum_of_inverse_trig_functions
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (m : ℝ), ∀ (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2),
    m ≤ a / (Real.sin θ)^3 + b / (Real.cos θ)^3 ∧
    m = (a^(2/5) + b^(2/5))^(5/2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_inverse_trig_functions_l3944_394451


namespace NUMINAMATH_CALUDE_second_day_study_hours_l3944_394482

/-- Represents the relationship between study hours and performance score for a given day -/
structure StudyDay where
  hours : ℝ
  score : ℝ

/-- The constant product of hours and score, representing the inverse relationship -/
def inverse_constant (day : StudyDay) : ℝ := day.hours * day.score

theorem second_day_study_hours 
  (day1 : StudyDay)
  (avg_score : ℝ)
  (h1 : day1.hours = 5)
  (h2 : day1.score = 80)
  (h3 : avg_score = 85) :
  ∃ (day2 : StudyDay), 
    inverse_constant day1 = inverse_constant day2 ∧
    (day1.score + day2.score) / 2 = avg_score ∧
    day2.hours = 40 / 9 := by
  sorry

end NUMINAMATH_CALUDE_second_day_study_hours_l3944_394482


namespace NUMINAMATH_CALUDE_square_sum_lower_bound_l3944_394412

theorem square_sum_lower_bound (x y θ : ℝ) 
  (h : (x * Real.cos θ + y * Real.sin θ)^2 + x * Real.sin θ - y * Real.cos θ = 1) : 
  x^2 + y^2 ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_lower_bound_l3944_394412


namespace NUMINAMATH_CALUDE_max_expr_value_l3944_394496

def S : Finset ℕ := {1, 2, 3, 4}

def expr (e f g h : ℕ) : ℕ := e * f^g - h

theorem max_expr_value :
  ∃ (e f g h : ℕ), e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ f ≠ g ∧ f ≠ h ∧ g ≠ h ∧
  expr e f g h = 161 ∧
  ∀ (a b c d : ℕ), a ∈ S → b ∈ S → c ∈ S → d ∈ S →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  expr a b c d ≤ 161 :=
by sorry

end NUMINAMATH_CALUDE_max_expr_value_l3944_394496


namespace NUMINAMATH_CALUDE_ed_doug_marble_difference_l3944_394445

theorem ed_doug_marble_difference (ed_initial : ℕ) (doug_initial : ℕ) (ed_lost : ℕ) (ed_final : ℕ) :
  ed_initial = doug_initial + 30 →
  ed_initial = ed_final + ed_lost →
  ed_lost = 21 →
  ed_final = 91 →
  ed_final - doug_initial = 9 :=
by sorry

end NUMINAMATH_CALUDE_ed_doug_marble_difference_l3944_394445


namespace NUMINAMATH_CALUDE_kitten_growth_theorem_l3944_394462

/-- Represents the length of a kitten at different stages of growth -/
structure KittenGrowth where
  initial_length : ℝ
  first_double : ℝ
  second_double : ℝ

/-- Theorem stating that if a kitten's length doubles twice and ends at 16 inches, its initial length was 4 inches -/
theorem kitten_growth_theorem (k : KittenGrowth) :
  k.second_double = 16 ∧ 
  k.first_double = 2 * k.initial_length ∧ 
  k.second_double = 2 * k.first_double →
  k.initial_length = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_kitten_growth_theorem_l3944_394462


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_4x_l3944_394423

theorem factorization_xy_squared_minus_4x (x y : ℝ) : 
  x * y^2 - 4 * x = x * (y + 2) * (y - 2) := by sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_4x_l3944_394423


namespace NUMINAMATH_CALUDE_student_activity_arrangements_l3944_394415

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of arrangements for distributing students between two activities -/
def total_arrangements (n : ℕ) : ℕ :=
  choose n 4 + choose n 3 + choose n 2

theorem student_activity_arrangements :
  total_arrangements 6 = 50 := by sorry

end NUMINAMATH_CALUDE_student_activity_arrangements_l3944_394415


namespace NUMINAMATH_CALUDE_binomial_probability_three_out_of_six_l3944_394489

/-- The probability mass function for a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The problem statement -/
theorem binomial_probability_three_out_of_six :
  binomial_pmf 6 (1/2) 3 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_three_out_of_six_l3944_394489


namespace NUMINAMATH_CALUDE_specific_marathon_distance_l3944_394455

/-- A circular marathon with four checkpoints -/
structure CircularMarathon where
  /-- Number of checkpoints -/
  num_checkpoints : Nat
  /-- Distance from start to first checkpoint -/
  start_to_first : ℝ
  /-- Distance from last checkpoint to finish -/
  last_to_finish : ℝ
  /-- Distance between consecutive checkpoints -/
  checkpoint_distance : ℝ

/-- The total distance of the marathon -/
def marathon_distance (m : CircularMarathon) : ℝ :=
  m.start_to_first + 
  m.last_to_finish + 
  (m.num_checkpoints - 1 : ℝ) * m.checkpoint_distance

/-- Theorem stating the total distance of the specific marathon -/
theorem specific_marathon_distance : 
  ∀ (m : CircularMarathon), 
    m.num_checkpoints = 4 ∧ 
    m.start_to_first = 1 ∧ 
    m.last_to_finish = 1 ∧ 
    m.checkpoint_distance = 6 → 
    marathon_distance m = 20 := by
  sorry

end NUMINAMATH_CALUDE_specific_marathon_distance_l3944_394455


namespace NUMINAMATH_CALUDE_point_outside_circle_l3944_394456

/-- A line intersects a circle at two distinct points if and only if 
    the distance from the circle's center to the line is less than the radius -/
axiom line_intersects_circle_iff_distance_lt_radius 
  (a b : ℝ) : (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    a * x₁ + b * y₁ = 4 ∧ x₁^2 + y₁^2 = 4 ∧
    a * x₂ + b * y₂ = 4 ∧ x₂^2 + y₂^2 = 4) ↔
  (4 / Real.sqrt (a^2 + b^2) < 2)

/-- The distance from a point to the origin is greater than 2 
    if and only if the point is outside the circle with radius 2 centered at the origin -/
axiom outside_circle_iff_distance_gt_radius 
  (a b : ℝ) : Real.sqrt (a^2 + b^2) > 2 ↔ (a, b) ∉ {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4}

theorem point_outside_circle (a b : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    a * x₁ + b * y₁ = 4 ∧ x₁^2 + y₁^2 = 4 ∧
    a * x₂ + b * y₂ = 4 ∧ x₂^2 + y₂^2 = 4) →
  (a, b) ∉ {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l3944_394456


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l3944_394478

/-- The sum of an arithmetic sequence with first term a, last term l, and common difference d -/
def arithmetic_sum (a l d : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

/-- The theorem stating that the remainder of the sum of the given arithmetic sequence when divided by 8 is 2 -/
theorem arithmetic_sequence_sum_remainder :
  (arithmetic_sum 3 299 8) % 8 = 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l3944_394478


namespace NUMINAMATH_CALUDE_root_range_implies_m_range_l3944_394444

theorem root_range_implies_m_range :
  ∀ m : ℝ,
  (∀ x : ℝ, x^2 - 2*m*x + m^2 - 1 = 0 → x > -2) →
  m > -1 :=
by sorry

end NUMINAMATH_CALUDE_root_range_implies_m_range_l3944_394444


namespace NUMINAMATH_CALUDE_mrs_hilt_pizzas_l3944_394476

/-- The number of slices in each pizza -/
def slices_per_pizza : ℕ := 8

/-- The total number of slices Mrs. Hilt had -/
def total_slices : ℕ := 16

/-- The number of pizzas Mrs. Hilt bought -/
def pizzas_bought : ℕ := total_slices / slices_per_pizza

theorem mrs_hilt_pizzas : pizzas_bought = 2 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pizzas_l3944_394476


namespace NUMINAMATH_CALUDE_candy_distribution_l3944_394429

theorem candy_distribution (initial_candies : ℕ) (friends : ℕ) (additional_candies : ℕ) :
  initial_candies = 20 →
  friends = 6 →
  additional_candies = 4 →
  (initial_candies + additional_candies) / friends = 4 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3944_394429


namespace NUMINAMATH_CALUDE_inequality_proof_l3944_394419

theorem inequality_proof (a b c d e f : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h5 : 0 ≤ e) (h6 : 0 ≤ f)
  (h7 : a + b ≤ e) (h8 : c + d ≤ f) : 
  Real.sqrt (a * c) + Real.sqrt (b * d) ≤ Real.sqrt (e * f) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3944_394419


namespace NUMINAMATH_CALUDE_wood_per_sack_l3944_394425

/-- Given that 4 sacks were filled with a total of 80 pieces of wood,
    prove that each sack contains 20 pieces of wood. -/
theorem wood_per_sack (total_wood : ℕ) (num_sacks : ℕ) 
  (h1 : total_wood = 80) (h2 : num_sacks = 4) :
  total_wood / num_sacks = 20 := by
  sorry

end NUMINAMATH_CALUDE_wood_per_sack_l3944_394425


namespace NUMINAMATH_CALUDE_ale_age_l3944_394448

/-- Represents a year-month combination -/
structure YearMonth where
  year : ℕ
  month : ℕ
  h_month_valid : month ≥ 1 ∧ month ≤ 12

/-- Calculates the age in years between two YearMonth dates -/
def ageInYears (birth death : YearMonth) : ℕ :=
  death.year - birth.year

theorem ale_age :
  let birth := YearMonth.mk 1859 1 (by simp)
  let death := YearMonth.mk 2014 8 (by simp)
  ageInYears birth death = 155 := by
  sorry

#check ale_age

end NUMINAMATH_CALUDE_ale_age_l3944_394448


namespace NUMINAMATH_CALUDE_instant_noodle_change_l3944_394494

theorem instant_noodle_change (total_change : ℕ) (total_notes : ℕ) (x : ℕ) (y : ℕ) : 
  total_change = 95 →
  total_notes = 16 →
  x + y = total_notes →
  10 * x + 5 * y = total_change →
  x = 3 := by
  sorry

end NUMINAMATH_CALUDE_instant_noodle_change_l3944_394494


namespace NUMINAMATH_CALUDE_bottle_cap_groups_l3944_394400

theorem bottle_cap_groups (total_caps : ℕ) (caps_per_group : ℕ) (num_groups : ℕ) : 
  total_caps = 35 → caps_per_group = 5 → num_groups = total_caps / caps_per_group → num_groups = 7 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_groups_l3944_394400


namespace NUMINAMATH_CALUDE_gary_stickers_left_l3944_394439

/-- The number of stickers Gary had initially -/
def initial_stickers : ℕ := 99

/-- The number of stickers Gary gave to Lucy -/
def stickers_to_lucy : ℕ := 42

/-- The number of stickers Gary gave to Alex -/
def stickers_to_alex : ℕ := 26

/-- The number of stickers Gary had left after giving stickers to Lucy and Alex -/
def stickers_left : ℕ := initial_stickers - (stickers_to_lucy + stickers_to_alex)

theorem gary_stickers_left : stickers_left = 31 := by
  sorry

end NUMINAMATH_CALUDE_gary_stickers_left_l3944_394439


namespace NUMINAMATH_CALUDE_smallest_possible_b_l3944_394471

theorem smallest_possible_b (a b : ℝ) : 
  2 < a → a < b → 
  (2 + a ≤ b) →
  (1 / b + 1 / a ≤ 2) →
  b ≥ (5 + Real.sqrt 17) / 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l3944_394471
