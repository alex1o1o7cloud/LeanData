import Mathlib

namespace NUMINAMATH_CALUDE_pyramid_levels_theorem_l774_77458

/-- Represents a pyramid of blocks -/
structure BlockPyramid where
  firstRowBlocks : ℕ
  decreaseRate : ℕ
  totalBlocks : ℕ

/-- Calculate the number of levels in a BlockPyramid -/
def pyramidLevels (p : BlockPyramid) : ℕ :=
  sorry

/-- Theorem: A pyramid with 25 total blocks, 9 blocks in the first row,
    and decreasing by 2 blocks in each row has 5 levels -/
theorem pyramid_levels_theorem (p : BlockPyramid) 
  (h1 : p.firstRowBlocks = 9)
  (h2 : p.decreaseRate = 2)
  (h3 : p.totalBlocks = 25) :
  pyramidLevels p = 5 :=
  sorry

end NUMINAMATH_CALUDE_pyramid_levels_theorem_l774_77458


namespace NUMINAMATH_CALUDE_hyperbola_C_tangent_intersection_product_l774_77416

/-- Hyperbola C -/
def hyperbola_C (x y : ℝ) : Prop := x^2 / 6 - y^2 / 3 = 1

/-- Point P on the line x = 2 -/
def point_P (t : ℝ) : ℝ × ℝ := (2, t)

/-- Function to calculate mn given t -/
noncomputable def mn (t : ℝ) : ℝ := 6 * Real.sqrt 6 - 15

theorem hyperbola_C_tangent_intersection_product :
  hyperbola_C (-3) (Real.sqrt 6 / 2) →
  ∀ t : ℝ, ∃ m n : ℝ,
    (∃ A B : ℝ × ℝ, 
      hyperbola_C A.1 A.2 ∧ 
      hyperbola_C B.1 B.2 ∧ 
      -- PA and PB are tangent to C
      -- M and N are defined as in the problem
      mn t = m * n) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_C_tangent_intersection_product_l774_77416


namespace NUMINAMATH_CALUDE_decimal_point_removal_l774_77473

theorem decimal_point_removal (x y z : ℝ) (hx : x = 1.6) (hy : y = 16) (hz : z = 14.4) :
  y - x = z := by sorry

end NUMINAMATH_CALUDE_decimal_point_removal_l774_77473


namespace NUMINAMATH_CALUDE_area_of_four_presentable_set_l774_77413

/-- A complex number is four-presentable if there exists a complex number w 
    with absolute value 4 such that z = w - 1/w -/
def FourPresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 4 ∧ z = w - 1 / w

/-- The set of all four-presentable complex numbers -/
def U : Set ℂ :=
  {z : ℂ | FourPresentable z}

/-- The area of a set in the complex plane -/
noncomputable def Area (S : Set ℂ) : ℝ := sorry

theorem area_of_four_presentable_set :
  Area U = 255 / 16 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_of_four_presentable_set_l774_77413


namespace NUMINAMATH_CALUDE_subcommittee_count_l774_77418

def planning_committee_size : ℕ := 12
def teachers_in_committee : ℕ := 5
def subcommittee_size : ℕ := 5
def min_teachers_in_subcommittee : ℕ := 2

theorem subcommittee_count : 
  (Finset.sum (Finset.range (teachers_in_committee - min_teachers_in_subcommittee + 1))
    (fun k => Nat.choose teachers_in_committee (k + min_teachers_in_subcommittee) * 
              Nat.choose (planning_committee_size - teachers_in_committee) (subcommittee_size - (k + min_teachers_in_subcommittee)))) = 596 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_count_l774_77418


namespace NUMINAMATH_CALUDE_number_calculation_l774_77435

theorem number_calculation (x : ℝ) : 0.2 * x = 0.4 * 140 + 80 → x = 680 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l774_77435


namespace NUMINAMATH_CALUDE_square_difference_identity_nine_point_five_squared_l774_77427

theorem square_difference_identity (x : ℝ) : (10 - x)^2 = 10^2 - 2 * 10 * x + x^2 := by sorry

theorem nine_point_five_squared :
  (9.5 : ℝ)^2 = 10^2 - 2 * 10 * 0.5 + 0.5^2 := by sorry

end NUMINAMATH_CALUDE_square_difference_identity_nine_point_five_squared_l774_77427


namespace NUMINAMATH_CALUDE_expected_digits_is_31_20_l774_77431

/-- A fair 20-sided die numbered from 1 to 20 -/
def icosahedralDie : Finset ℕ := Finset.range 20

/-- The number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

/-- The expected number of digits when rolling the die -/
def expectedDigits : ℚ :=
  (icosahedralDie.sum fun i => numDigits (i + 1)) / icosahedralDie.card

/-- Theorem stating the expected number of digits -/
theorem expected_digits_is_31_20 : expectedDigits = 31 / 20 := by
  sorry

end NUMINAMATH_CALUDE_expected_digits_is_31_20_l774_77431


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l774_77429

theorem cubic_sum_theorem (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a)
  (h : (a^3 + 9)/a = (b^3 + 9)/b ∧ (b^3 + 9)/b = (c^3 + 9)/c) :
  a^3 + b^3 + c^3 = -27 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l774_77429


namespace NUMINAMATH_CALUDE_car_sale_profit_l774_77410

theorem car_sale_profit (P : ℝ) (h : P > 0) :
  let buying_price := 0.80 * P
  let selling_price := 1.3600000000000001 * P
  let percentage_increase := (selling_price / buying_price - 1) * 100
  percentage_increase = 70.00000000000002 := by
sorry

end NUMINAMATH_CALUDE_car_sale_profit_l774_77410


namespace NUMINAMATH_CALUDE_final_sign_is_minus_l774_77493

/-- Represents the two types of signs on the board -/
inductive Sign
| Plus
| Minus

/-- Represents the state of the board -/
structure Board :=
  (plus_count : ℕ)
  (minus_count : ℕ)

/-- Performs one operation on the board -/
def perform_operation (b : Board) : Board :=
  sorry

/-- Performs n operations on the board -/
def perform_n_operations (b : Board) (n : ℕ) : Board :=
  sorry

/-- The main theorem to prove -/
theorem final_sign_is_minus :
  let initial_board : Board := ⟨10, 15⟩
  let final_board := perform_n_operations initial_board 24
  final_board.plus_count = 0 ∧ final_board.minus_count = 1 :=
sorry

end NUMINAMATH_CALUDE_final_sign_is_minus_l774_77493


namespace NUMINAMATH_CALUDE_subset_implies_m_leq_5_l774_77400

/-- Given sets A and B, prove that if B is a subset of A, then m ≤ 5 -/
theorem subset_implies_m_leq_5 (m : ℝ) : 
  let A : Set ℝ := {x | 4 ≤ x ∧ x ≤ 8}
  let B : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 2}
  B ⊆ A → m ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_leq_5_l774_77400


namespace NUMINAMATH_CALUDE_katie_candy_problem_l774_77412

theorem katie_candy_problem (x : ℕ) : 
  x + 6 - 9 = 7 → x = 10 := by sorry

end NUMINAMATH_CALUDE_katie_candy_problem_l774_77412


namespace NUMINAMATH_CALUDE_problem_solution_l774_77496

theorem problem_solution (a b c : ℤ) 
  (h1 : a + b = c) 
  (h2 : b + 2*c = 10) 
  (h3 : c = 4) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l774_77496


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l774_77421

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, -1; 2, -4]

theorem inverse_as_linear_combination :
  ∃ (c d : ℚ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧ c = 1/10 ∧ d = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l774_77421


namespace NUMINAMATH_CALUDE_example_is_quadratic_l774_77440

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² = 2 + 3x is a quadratic equation -/
theorem example_is_quadratic : is_quadratic_equation (λ x => x^2 - 3*x - 2) := by
  sorry

end NUMINAMATH_CALUDE_example_is_quadratic_l774_77440


namespace NUMINAMATH_CALUDE_red_marbles_in_bag_l774_77461

theorem red_marbles_in_bag (total_marbles : ℕ) (red_marbles : ℕ) 
  (h1 : total_marbles = red_marbles + 3)
  (h2 : (red_marbles : ℝ) / total_marbles * ((red_marbles - 1) : ℝ) / (total_marbles - 1) = 0.1) :
  red_marbles = 2 := by
sorry

end NUMINAMATH_CALUDE_red_marbles_in_bag_l774_77461


namespace NUMINAMATH_CALUDE_always_real_roots_discriminant_one_implies_m_two_l774_77433

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ :=
  2 * m * x^2 - (5 * m - 1) * x + 3 * m - 1

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ :=
  (5 * m - 1)^2 - 4 * 2 * m * (3 * m - 1)

-- Theorem stating that the equation always has real roots
theorem always_real_roots (m : ℝ) :
  ∃ x : ℝ, quadratic_equation m x = 0 :=
sorry

-- Theorem stating that when the discriminant is 1, m = 2
theorem discriminant_one_implies_m_two :
  ∀ m : ℝ, discriminant m = 1 → m = 2 :=
sorry

end NUMINAMATH_CALUDE_always_real_roots_discriminant_one_implies_m_two_l774_77433


namespace NUMINAMATH_CALUDE_unique_square_double_reverse_l774_77486

theorem unique_square_double_reverse : ∃! x : ℕ,
  (10 ≤ x^2 ∧ x^2 < 100) ∧
  (10 ≤ 2*x ∧ 2*x < 100) ∧
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ x^2 = 10*a + b ∧ 2*x = 10*b + a) ∧
  x^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_double_reverse_l774_77486


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l774_77451

theorem pirate_treasure_probability :
  let n : ℕ := 5  -- number of islands
  let p_treasure : ℚ := 1/3  -- probability of treasure on an island
  let p_trap : ℚ := 1/6  -- probability of trap on an island
  let p_neither : ℚ := 1/2  -- probability of neither treasure nor trap on an island
  
  -- Probability of exactly 4 islands with treasure and 1 with neither
  (Nat.choose n 4 : ℚ) * p_treasure^4 * p_neither = 5/162 :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l774_77451


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l774_77454

theorem simplify_and_evaluate (x y : ℚ) (hx : x = 1/2) (hy : y = 2023) :
  (x + y)^2 + (x + y)*(x - y) - 2*x^2 = 2*x*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l774_77454


namespace NUMINAMATH_CALUDE_p_geq_q_l774_77425

theorem p_geq_q (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : a^a * b^b ≥ a^b * b^a := by
  sorry

end NUMINAMATH_CALUDE_p_geq_q_l774_77425


namespace NUMINAMATH_CALUDE_roots_of_unity_count_l774_77497

theorem roots_of_unity_count (a b c : ℤ) : 
  ∃ (roots : Finset ℂ), 
    (∀ z ∈ roots, z^3 = 1 ∧ z^3 + a*z^2 + b*z + c = 0) ∧ 
    Finset.card roots = 3 :=
sorry

end NUMINAMATH_CALUDE_roots_of_unity_count_l774_77497


namespace NUMINAMATH_CALUDE_central_angles_sum_l774_77468

theorem central_angles_sum (y : ℝ) : 
  (6 * y + 7 * y + 3 * y + y) * (π / 180) = 2 * π → y = 360 / 17 := by
sorry

end NUMINAMATH_CALUDE_central_angles_sum_l774_77468


namespace NUMINAMATH_CALUDE_abc_maximum_l774_77472

theorem abc_maximum (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a * b + c = (a + c) * (b + c)) (h_sum : a + b + c = 2) :
  a * b * c ≤ 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_abc_maximum_l774_77472


namespace NUMINAMATH_CALUDE_west_7m_is_negative_7m_l774_77424

/-- Represents the direction of movement on an east-west road -/
inductive Direction
  | East
  | West

/-- Represents a movement on the road with a direction and distance -/
structure Movement where
  direction : Direction
  distance : ℝ

/-- Converts a movement to its signed representation -/
def Movement.toSigned (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.distance
  | Direction.West => -m.distance

/-- The theorem stating that moving west by 7m should be denoted as -7m -/
theorem west_7m_is_negative_7m
  (h : Movement.toSigned { direction := Direction.East, distance := 3 } = 3) :
  Movement.toSigned { direction := Direction.West, distance := 7 } = -7 := by
  sorry

end NUMINAMATH_CALUDE_west_7m_is_negative_7m_l774_77424


namespace NUMINAMATH_CALUDE_modular_congruence_l774_77417

theorem modular_congruence (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ 99 * n ≡ 65 [ZMOD 103] → n ≡ 68 [ZMOD 103] := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_l774_77417


namespace NUMINAMATH_CALUDE_volvox_face_difference_l774_77406

/-- A spherical polyhedron where each face has 5, 6, or 7 sides, and exactly three faces meet at each vertex. -/
structure VolvoxPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  f₅ : ℕ  -- number of pentagonal faces
  f₆ : ℕ  -- number of hexagonal faces
  f₇ : ℕ  -- number of heptagonal faces
  euler : V - E + F = 2
  face_sum : F = f₅ + f₆ + f₇
  edge_sum : 2 * E = 5 * f₅ + 6 * f₆ + 7 * f₇
  vertex_sum : 3 * V = 5 * f₅ + 6 * f₆ + 7 * f₇

/-- The number of pentagonal faces is always 12 more than the number of heptagonal faces. -/
theorem volvox_face_difference (p : VolvoxPolyhedron) : p.f₅ = p.f₇ + 12 := by
  sorry

end NUMINAMATH_CALUDE_volvox_face_difference_l774_77406


namespace NUMINAMATH_CALUDE_pizza_combinations_l774_77447

theorem pizza_combinations (n k : ℕ) (h1 : n = 8) (h2 : k = 5) : 
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l774_77447


namespace NUMINAMATH_CALUDE_ice_skating_falls_l774_77448

theorem ice_skating_falls (steven_falls sonya_falls : ℕ) 
  (h1 : steven_falls = 3)
  (h2 : sonya_falls = 6) : 
  (steven_falls + 13) / 2 - sonya_falls = 2 := by
  sorry

end NUMINAMATH_CALUDE_ice_skating_falls_l774_77448


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l774_77430

theorem consecutive_odd_integers_sum (n : ℤ) : 
  (n % 2 = 1) → -- n is odd
  (n + (n + 4) = 150) → -- sum of first and third is 150
  (n + (n + 2) + (n + 4) = 225) -- sum of all three is 225
:= by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l774_77430


namespace NUMINAMATH_CALUDE_reflect_P_across_x_axis_l774_77466

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The coordinates of P(-3,2) reflected across the x-axis -/
theorem reflect_P_across_x_axis : 
  reflect_x (-3, 2) = (-3, -2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_P_across_x_axis_l774_77466


namespace NUMINAMATH_CALUDE_rationalize_denominator_l774_77438

theorem rationalize_denominator : 
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l774_77438


namespace NUMINAMATH_CALUDE_no_solution_exists_l774_77405

theorem no_solution_exists (k m : ℕ) : k.factorial + 48 ≠ 48 * (k + 1) ^ m := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l774_77405


namespace NUMINAMATH_CALUDE_inverse_square_relation_l774_77481

/-- A function that varies inversely as the square of its input -/
noncomputable def f (y : ℝ) : ℝ := 4 / y^2

theorem inverse_square_relation (y₀ : ℝ) :
  f 6 = 0.1111111111111111 →
  (∃ y, f y = 1) →
  f y₀ = 1 →
  y₀ = 2 := by
sorry

end NUMINAMATH_CALUDE_inverse_square_relation_l774_77481


namespace NUMINAMATH_CALUDE_angelina_walking_speed_l774_77475

/-- Angelina's walking problem -/
theorem angelina_walking_speed
  (distance_home_grocery : ℝ)
  (distance_grocery_gym : ℝ)
  (time_difference : ℝ)
  (h1 : distance_home_grocery = 960)
  (h2 : distance_grocery_gym = 480)
  (h3 : time_difference = 40)
  (h4 : distance_grocery_gym / (distance_home_grocery / speed_home_grocery) 
      = distance_grocery_gym / ((distance_home_grocery / speed_home_grocery) - time_difference))
  (h5 : speed_grocery_gym = 2 * speed_home_grocery) :
  speed_grocery_gym = 36 :=
by sorry

#check angelina_walking_speed

end NUMINAMATH_CALUDE_angelina_walking_speed_l774_77475


namespace NUMINAMATH_CALUDE_haley_candy_eaten_l774_77459

/-- Given Haley's initial candy count, the amount her sister gave her, and her final candy count,
    calculate how many pieces of candy Haley ate on the first night. -/
theorem haley_candy_eaten (initial : ℕ) (sister_gave : ℕ) (final : ℕ) : 
  initial = 33 → sister_gave = 19 → final = 35 → initial - (final - sister_gave) = 17 := by
  sorry

end NUMINAMATH_CALUDE_haley_candy_eaten_l774_77459


namespace NUMINAMATH_CALUDE_find_p_l774_77487

theorem find_p (m : ℕ) (p : ℕ) :
  m = 34 →
  ((1 ^ (m + 1)) / (5 ^ (m + 1))) * ((1 ^ 18) / (4 ^ 18)) = 1 / (2 * (10 ^ p)) →
  p = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_p_l774_77487


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l774_77442

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x^2 + 5*x + 6 < 0 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x^2 + 5*x + 6 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l774_77442


namespace NUMINAMATH_CALUDE_trillion_equals_ten_to_sixteen_l774_77414

-- Define the relationships between numbers
def ten_thousand : ℕ := 10^4
def million : ℕ := 10^6
def billion : ℕ := 10^8
def trillion : ℕ := ten_thousand * million * billion

-- Theorem statement
theorem trillion_equals_ten_to_sixteen : trillion = 10^16 := by
  sorry

end NUMINAMATH_CALUDE_trillion_equals_ten_to_sixteen_l774_77414


namespace NUMINAMATH_CALUDE_product_multiple_of_60_probability_l774_77415

def is_multiple_of_60 (n : ℕ) : Prop := ∃ k : ℕ, n = 60 * k

def count_favorable_pairs : ℕ := 732

def total_pairs : ℕ := 60 * 60

theorem product_multiple_of_60_probability :
  (count_favorable_pairs : ℚ) / (total_pairs : ℚ) = 61 / 300 := by sorry

end NUMINAMATH_CALUDE_product_multiple_of_60_probability_l774_77415


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_one_l774_77463

theorem negation_of_universal_positive_square_plus_one :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_one_l774_77463


namespace NUMINAMATH_CALUDE_max_catered_children_correct_l774_77484

structure MealData where
  total_adults : ℕ
  total_children : ℕ
  prepared_veg_adults : ℕ
  prepared_nonveg_adults : ℕ
  prepared_vegan_adults : ℕ
  prepared_veg_children : ℕ
  prepared_nonveg_children : ℕ
  prepared_vegan_children : ℕ
  pref_veg_adults : ℕ
  pref_nonveg_adults : ℕ
  pref_vegan_adults : ℕ
  pref_veg_children : ℕ
  pref_nonveg_children : ℕ
  pref_vegan_children : ℕ
  eaten_veg_adults : ℕ
  eaten_nonveg_adults : ℕ
  eaten_vegan_adults : ℕ

def max_catered_children (data : MealData) : ℕ × ℕ × ℕ :=
  let remaining_veg := data.prepared_veg_adults + data.prepared_veg_children - data.eaten_veg_adults
  let remaining_nonveg := data.prepared_nonveg_adults + data.prepared_nonveg_children - data.eaten_nonveg_adults
  let remaining_vegan := data.prepared_vegan_adults + data.prepared_vegan_children - data.eaten_vegan_adults
  (min remaining_veg data.pref_veg_children,
   min remaining_nonveg data.pref_nonveg_children,
   min remaining_vegan data.pref_vegan_children)

theorem max_catered_children_correct (data : MealData) : 
  data.total_adults = 80 ∧
  data.total_children = 120 ∧
  data.prepared_veg_adults = 70 ∧
  data.prepared_nonveg_adults = 75 ∧
  data.prepared_vegan_adults = 5 ∧
  data.prepared_veg_children = 90 ∧
  data.prepared_nonveg_children = 25 ∧
  data.prepared_vegan_children = 5 ∧
  data.pref_veg_adults = 45 ∧
  data.pref_nonveg_adults = 30 ∧
  data.pref_vegan_adults = 5 ∧
  data.pref_veg_children = 100 ∧
  data.pref_nonveg_children = 15 ∧
  data.pref_vegan_children = 5 ∧
  data.eaten_veg_adults = 42 ∧
  data.eaten_nonveg_adults = 25 ∧
  data.eaten_vegan_adults = 5
  →
  max_catered_children data = (100, 15, 5) := by
sorry

end NUMINAMATH_CALUDE_max_catered_children_correct_l774_77484


namespace NUMINAMATH_CALUDE_curve_symmetry_l774_77460

theorem curve_symmetry (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + a^2*x + (1-a^2)*y - 4 = 0 ↔ 
              y^2 + x^2 + a^2*y + (1-a^2)*x - 4 = 0) →
  a = Real.sqrt 2 / 2 ∨ a = -Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_curve_symmetry_l774_77460


namespace NUMINAMATH_CALUDE_inverse_sum_l774_77491

def f (a b x : ℝ) : ℝ := a * x^2 + b * x

def f_inv (a b x : ℝ) : ℝ := b * x^2 + a * x

theorem inverse_sum (a b : ℝ) :
  (∀ x, f a b (f_inv a b x) = x) → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_l774_77491


namespace NUMINAMATH_CALUDE_quadratic_inequality_region_l774_77483

theorem quadratic_inequality_region (x y : ℝ) :
  (∀ t : ℝ, t^2 ≤ 1 → t^2 + y*t + x ≥ 0) →
  (y ≤ x + 1 ∧ y ≥ -x - 1 ∧ x ≥ y^2/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_region_l774_77483


namespace NUMINAMATH_CALUDE_age_relation_proof_l774_77450

/-- Represents the current ages and future time when Alex's age is thrice Ben's --/
structure AgeRelation where
  ben_age : ℕ
  alex_age : ℕ
  michael_age : ℕ
  future_years : ℕ

/-- The conditions of the problem --/
def age_conditions (ar : AgeRelation) : Prop :=
  ar.ben_age = 4 ∧
  ar.alex_age = ar.ben_age + 30 ∧
  ar.michael_age = ar.alex_age + 4 ∧
  ar.alex_age + ar.future_years = 3 * (ar.ben_age + ar.future_years)

/-- The theorem to prove --/
theorem age_relation_proof :
  ∃ (ar : AgeRelation), age_conditions ar ∧ ar.future_years = 11 :=
sorry

end NUMINAMATH_CALUDE_age_relation_proof_l774_77450


namespace NUMINAMATH_CALUDE_socks_expense_is_eleven_l774_77403

/-- The amount spent on socks given a budget and other expenses --/
def socks_expense (budget : ℕ) (shirt_cost pants_cost coat_cost belt_cost shoes_cost amount_left : ℕ) : ℕ :=
  budget - (shirt_cost + pants_cost + coat_cost + belt_cost + shoes_cost + amount_left)

/-- Theorem: Given the specific budget and expenses, the amount spent on socks is $11 --/
theorem socks_expense_is_eleven :
  socks_expense 200 30 46 38 18 41 16 = 11 := by
  sorry

end NUMINAMATH_CALUDE_socks_expense_is_eleven_l774_77403


namespace NUMINAMATH_CALUDE_solve_equation_l774_77411

theorem solve_equation (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (9 * x) * Real.sqrt (12 * x) * Real.sqrt (4 * x) * Real.sqrt (18 * x) = 36) :
  x = Real.sqrt (9 / 22) :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l774_77411


namespace NUMINAMATH_CALUDE_ratio_of_y_coordinates_l774_77485

-- Define the ellipse
def Γ (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the point P
def P : ℝ × ℝ := (1, 0)

-- Define the lines l₁ and l₂
def l₁ (x : ℝ) : Prop := x = -2
def l₂ (x : ℝ) : Prop := x = 2

-- Define the line l_CD
def l_CD (x : ℝ) : Prop := x = 1

-- Define the chords AB and CD (implicitly by their properties)
def chord_passes_through_P (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, A = (1 - t) • P + t • B

-- Define points E and F
def E : ℝ × ℝ := (2, sorry)  -- y-coordinate to be determined
def F : ℝ × ℝ := (-2, sorry) -- y-coordinate to be determined

theorem ratio_of_y_coordinates :
  ∃ (A B C D : ℝ × ℝ),
    Γ A.1 A.2 ∧ Γ B.1 B.2 ∧ Γ C.1 C.2 ∧ Γ D.1 D.2 ∧
    chord_passes_through_P A B ∧ chord_passes_through_P C D ∧
    l_CD C.1 ∧ l_CD D.1 ∧
    (E.2 : ℝ) / (F.2 : ℝ) = -1/3 :=
sorry

end NUMINAMATH_CALUDE_ratio_of_y_coordinates_l774_77485


namespace NUMINAMATH_CALUDE_sum_of_digits_l774_77479

/-- Given two single-digit numbers x and y, prove that x + y = 6 under certain conditions. -/
theorem sum_of_digits (x y : ℕ) : 
  (0 ≤ x ∧ x ≤ 9) →
  (0 ≤ y ∧ y ≤ 9) →
  (200 + 10 * x + 3) + 326 = (500 + 10 * y + 9) →
  (500 + 10 * y + 9) % 9 = 0 →
  x + y = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l774_77479


namespace NUMINAMATH_CALUDE_sine_sum_acute_triangle_l774_77402

theorem sine_sum_acute_triangle (α β γ : Real) 
  (acute_triangle : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π)
  (acute_angles : α < π/2 ∧ β < π/2 ∧ γ < π/2) : 
  Real.sin α + Real.sin β + Real.sin γ > 2 := by
sorry

end NUMINAMATH_CALUDE_sine_sum_acute_triangle_l774_77402


namespace NUMINAMATH_CALUDE_complex_equation_solution_l774_77499

theorem complex_equation_solution (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 1))
  (eq2 : b = (a + c) / (y - 1))
  (eq3 : c = (a + b) / (z - 1))
  (eq4 : x * y + x * z + y * z = 7)
  (eq5 : x + y + z = 3) :
  x * y * z = 9 := by
  sorry


end NUMINAMATH_CALUDE_complex_equation_solution_l774_77499


namespace NUMINAMATH_CALUDE_scholarship_difference_l774_77453

theorem scholarship_difference (nina kelly wendy : ℕ) : 
  nina < kelly →
  kelly = 2 * wendy →
  wendy = 20000 →
  nina + kelly + wendy = 92000 →
  kelly - nina = 8000 := by
sorry

end NUMINAMATH_CALUDE_scholarship_difference_l774_77453


namespace NUMINAMATH_CALUDE_triangle_side_length_l774_77420

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 120 * π / 180 →  -- Convert 120° to radians
  a = 2 * Real.sqrt 3 → 
  b = 2 → 
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) → 
  c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l774_77420


namespace NUMINAMATH_CALUDE_xy_value_l774_77419

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x / 2 + 2 * y - 2 = Real.log x + Real.log y) : 
  x ^ y = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l774_77419


namespace NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l774_77436

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_prime_divisor_factorial_sum :
  ∃ (p : ℕ), isPrime p ∧ p ∣ (factorial 13 + factorial 14) ∧
  ∀ (q : ℕ), isPrime q → q ∣ (factorial 13 + factorial 14) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l774_77436


namespace NUMINAMATH_CALUDE_gathering_handshakes_l774_77443

/-- Calculates the number of handshakes in a gathering with specific rules -/
def handshakes (n : ℕ) : ℕ :=
  let couples := n
  let men := couples
  let women := couples
  let guest := 1
  let total_people := men + women + guest
  let handshakes_among_men := men * (men - 1) / 2
  let handshakes_men_women := men * (women - 1)
  let handshakes_with_guest := total_people - 1
  handshakes_among_men + handshakes_men_women + handshakes_with_guest

/-- Theorem stating that in a gathering of 15 married couples and 1 special guest,
    with specific handshake rules, the total number of handshakes is 345 -/
theorem gathering_handshakes : handshakes 15 = 345 := by
  sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l774_77443


namespace NUMINAMATH_CALUDE_geometric_sequence_a10_l774_77480

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a10 (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_a6 : a 6 = 162) : 
  a 10 = 13122 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a10_l774_77480


namespace NUMINAMATH_CALUDE_lollipop_ratio_l774_77467

theorem lollipop_ratio : 
  ∀ (alison henry diane : ℕ),
    alison = 60 →
    henry = alison + 30 →
    alison + henry + diane = 45 * 6 →
    (alison : ℚ) / diane = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_ratio_l774_77467


namespace NUMINAMATH_CALUDE_problem_solution_l774_77470

-- Define the function f(x) = x^3 + ax^2 - x
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 1

theorem problem_solution (a : ℝ) (h : f' a 1 = 4) :
  a = 1 ∧
  ∃ (m b : ℝ), m = 4 ∧ b = -3 ∧ ∀ x y, y = f a x → (y - f a 1 = m * (x - 1) ↔ m*x - y - b = 0) ∧
  ∃ (lower upper : ℝ), lower = -5/27 ∧ upper = 10 ∧
    (∀ x, x ∈ Set.Icc 0 2 → f a x ∈ Set.Icc lower upper) ∧
    (∃ x₁ x₂, x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ f a x₁ = lower ∧ f a x₂ = upper) :=
by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l774_77470


namespace NUMINAMATH_CALUDE_burning_time_3x5_rectangle_l774_77495

/-- Represents a rectangle of toothpicks -/
structure ToothpickRectangle where
  rows : Nat
  cols : Nat
  burnTime : Nat  -- Time to burn one toothpick

/-- Calculates the burning time for a ToothpickRectangle -/
def burningTime (rect : ToothpickRectangle) : Nat :=
  let maxDim := max rect.rows rect.cols
  (maxDim - 1) * rect.burnTime + 5

theorem burning_time_3x5_rectangle :
  let rect : ToothpickRectangle := {
    rows := 3,
    cols := 5,
    burnTime := 10
  }
  burningTime rect = 65 := by sorry

end NUMINAMATH_CALUDE_burning_time_3x5_rectangle_l774_77495


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l774_77498

-- Define the sets A and B
def A : Set ℝ := {x | 2^x ≤ 2 * Real.sqrt 2}
def B : Set ℝ := {x | Real.log (2 - x) < 0}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = Set.Ioo (3/2) 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l774_77498


namespace NUMINAMATH_CALUDE_sequence_property_l774_77428

def sequence_sum (a : ℕ+ → ℚ) (n : ℕ+) : ℚ :=
  (Finset.range n.val).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_property (a : ℕ+ → ℚ) :
  (∀ n : ℕ+, sequence_sum a n + a n = 4 - 1 / (2 ^ (n.val - 2))) →
  (∀ n : ℕ+, a n = n.val / (2 ^ (n.val - 1))) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l774_77428


namespace NUMINAMATH_CALUDE_fraction_simplification_l774_77408

theorem fraction_simplification : (150 : ℚ) / 4500 = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l774_77408


namespace NUMINAMATH_CALUDE_max_boxes_in_wooden_box_l774_77404

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Converts meters to centimeters -/
def metersToCentimeters (m : ℕ) : ℕ :=
  m * 100

theorem max_boxes_in_wooden_box :
  let largeBox : BoxDimensions := {
    length := metersToCentimeters 8,
    width := metersToCentimeters 10,
    height := metersToCentimeters 6
  }
  let smallBox : BoxDimensions := {
    length := 4,
    width := 5,
    height := 6
  }
  (boxVolume largeBox) / (boxVolume smallBox) = 4000000 := by
  sorry

end NUMINAMATH_CALUDE_max_boxes_in_wooden_box_l774_77404


namespace NUMINAMATH_CALUDE_percentage_fraction_proof_l774_77492

theorem percentage_fraction_proof (P : ℚ) : 
  P < 35 → (P / 100) * 180 = 42 → P / 100 = 7 / 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_fraction_proof_l774_77492


namespace NUMINAMATH_CALUDE_product_difference_squares_divisible_by_three_l774_77457

theorem product_difference_squares_divisible_by_three (m n : ℤ) :
  ∃ k : ℤ, m * n * (m^2 - n^2) = 3 * k := by
sorry

end NUMINAMATH_CALUDE_product_difference_squares_divisible_by_three_l774_77457


namespace NUMINAMATH_CALUDE_second_number_calculation_l774_77469

theorem second_number_calculation (a b : ℝ) (h1 : a = 1600) (h2 : 0.20 * a = 0.20 * b + 190) : b = 650 := by
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_l774_77469


namespace NUMINAMATH_CALUDE_forty_percent_value_l774_77449

theorem forty_percent_value (x : ℝ) (h : 0.1 * x = 40) : 0.4 * x = 160 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_value_l774_77449


namespace NUMINAMATH_CALUDE_taylor_books_l774_77452

theorem taylor_books (candice amanda kara patricia taylor : ℕ) : 
  candice = 3 * amanda →
  kara = amanda / 2 →
  patricia = 7 * kara →
  taylor = (candice + amanda + kara + patricia) / 4 →
  candice = 18 →
  taylor = 12 := by
sorry

end NUMINAMATH_CALUDE_taylor_books_l774_77452


namespace NUMINAMATH_CALUDE_eventB_mutually_exclusive_not_complementary_to_eventA_l774_77439

/-- Represents the possible outcomes when drawing balls from a bag -/
inductive BallDraw
  | TwoBlack
  | ThreeBlack
  | OneBlack
  | NoBlack

/-- The total number of balls in the bag -/
def totalBalls : ℕ := 6

/-- The number of black balls in the bag -/
def blackBalls : ℕ := 3

/-- The number of red balls in the bag -/
def redBalls : ℕ := 3

/-- The number of balls drawn -/
def ballsDrawn : ℕ := 3

/-- Event A: At least 2 black balls are drawn -/
def eventA : Set BallDraw := {BallDraw.TwoBlack, BallDraw.ThreeBlack}

/-- Event B: Exactly 1 black ball is drawn -/
def eventB : Set BallDraw := {BallDraw.OneBlack}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (S T : Set BallDraw) : Prop := S ∩ T = ∅

/-- Two events are complementary if their union is the entire sample space -/
def complementary (S T : Set BallDraw) : Prop := S ∪ T = Set.univ

theorem eventB_mutually_exclusive_not_complementary_to_eventA :
  mutuallyExclusive eventA eventB ∧ ¬complementary eventA eventB := by sorry

end NUMINAMATH_CALUDE_eventB_mutually_exclusive_not_complementary_to_eventA_l774_77439


namespace NUMINAMATH_CALUDE_class_b_more_uniform_l774_77478

/-- Represents a class of students participating in a gymnastics competition -/
structure GymClass where
  name : String
  num_students : Nat
  avg_height : Float
  height_variance : Float

/-- Determines which of two classes has more uniform heights based on their variances -/
def more_uniform_heights (class_a class_b : GymClass) : Prop :=
  class_a.height_variance < class_b.height_variance

/-- Theorem: Given the variances of Class A and Class B, Class B has more uniform heights -/
theorem class_b_more_uniform (class_a class_b : GymClass) 
  (h1 : class_a.name = "A" ∧ class_b.name = "B")
  (h2 : class_a.num_students = 18 ∧ class_b.num_students = 18)
  (h3 : class_a.avg_height = 1.72 ∧ class_b.avg_height = 1.72)
  (h4 : class_a.height_variance = 3.24)
  (h5 : class_b.height_variance = 1.63) :
  more_uniform_heights class_b class_a :=
by sorry

end NUMINAMATH_CALUDE_class_b_more_uniform_l774_77478


namespace NUMINAMATH_CALUDE_divisor_inequality_l774_77494

theorem divisor_inequality (d d' n : ℕ) (h1 : d' > d) (h2 : d ∣ n) (h3 : d' ∣ n) :
  d' > d + d^2 / n :=
by sorry

end NUMINAMATH_CALUDE_divisor_inequality_l774_77494


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l774_77482

def dress_shirt_price : ℝ := 25
def pants_price : ℝ := 35
def socks_price : ℝ := 10
def dress_shirt_quantity : ℕ := 4
def pants_quantity : ℕ := 2
def socks_quantity : ℕ := 3
def dress_shirt_discount : ℝ := 0.15
def pants_discount : ℝ := 0.20
def socks_discount : ℝ := 0.10
def tax_rate : ℝ := 0.10
def shipping_fee : ℝ := 12.50

def total_cost : ℝ :=
  let dress_shirts_total := dress_shirt_price * dress_shirt_quantity * (1 - dress_shirt_discount)
  let pants_total := pants_price * pants_quantity * (1 - pants_discount)
  let socks_total := socks_price * socks_quantity * (1 - socks_discount)
  let subtotal := dress_shirts_total + pants_total + socks_total
  let tax := subtotal * tax_rate
  subtotal + tax + shipping_fee

theorem total_cost_is_correct : total_cost = 197.30 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l774_77482


namespace NUMINAMATH_CALUDE_original_count_pingpong_shuttlecock_l774_77474

theorem original_count_pingpong_shuttlecock : ∀ (n : ℕ),
  (∃ (x : ℕ), n = 5 * x ∧ n = 3 * x + 16) →
  n = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_count_pingpong_shuttlecock_l774_77474


namespace NUMINAMATH_CALUDE_min_value_of_function_l774_77455

theorem min_value_of_function (θ a b : ℝ) (n : ℕ) 
  (h1 : 0 < θ) (h2 : θ < π/2) (h3 : a > 0) (h4 : b > 0) (h5 : n > 0) :
  let f := fun θ => a / (Real.sin θ)^n + b / (Real.cos θ)^n
  ∃ (θ_min : ℝ), ∀ θ', 0 < θ' ∧ θ' < π/2 → 
    f θ' ≥ f θ_min ∧ f θ_min = (a^(2/(n+2:ℝ)) + b^(2/(n+2:ℝ)))^((n+2)/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l774_77455


namespace NUMINAMATH_CALUDE_min_value_x_l774_77476

theorem min_value_x (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x - 2*y = (x + 16*y) / (2*x*y)) : 
  x ≥ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ = 4 ∧ y₀ > 0 ∧ x₀ - 2*y₀ = (x₀ + 16*y₀) / (2*x₀*y₀) := by
  sorry

#check min_value_x

end NUMINAMATH_CALUDE_min_value_x_l774_77476


namespace NUMINAMATH_CALUDE_commission_calculation_l774_77445

/-- The original commission held by the company for John --/
def original_commission : ℕ := sorry

/-- The advance agency fees taken by John --/
def advance_fees : ℕ := 8280

/-- The amount given to John by the accountant after one month --/
def amount_given : ℕ := 18500

/-- The incentive amount given to John --/
def incentive_amount : ℕ := 1780

/-- Theorem stating the relationship between the original commission and other amounts --/
theorem commission_calculation : 
  original_commission = amount_given + advance_fees - incentive_amount :=
by sorry

end NUMINAMATH_CALUDE_commission_calculation_l774_77445


namespace NUMINAMATH_CALUDE_suitcase_theorem_l774_77489

/-- Represents the suitcase scenario at the airport -/
structure SuitcaseScenario where
  total_suitcases : ℕ
  business_suitcases : ℕ
  placement_interval : ℕ

/-- The probability of businesspeople waiting exactly 2 minutes for their last suitcase -/
def exact_wait_probability (s : SuitcaseScenario) : ℚ :=
  (Nat.choose 59 9 : ℚ) / (Nat.choose s.total_suitcases s.business_suitcases)

/-- The expected waiting time for businesspeople's last suitcase in seconds -/
def expected_wait_time (s : SuitcaseScenario) : ℚ :=
  4020 / 11

/-- Theorem stating the probability and expected waiting time for the suitcase scenario -/
theorem suitcase_theorem (s : SuitcaseScenario) 
  (h1 : s.total_suitcases = 200)
  (h2 : s.business_suitcases = 10)
  (h3 : s.placement_interval = 2) :
  exact_wait_probability s = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10) ∧
  expected_wait_time s = 4020 / 11 := by
  sorry

#eval exact_wait_probability ⟨200, 10, 2⟩
#eval expected_wait_time ⟨200, 10, 2⟩

end NUMINAMATH_CALUDE_suitcase_theorem_l774_77489


namespace NUMINAMATH_CALUDE_infinitely_many_primes_3_mod_4_infinitely_many_primes_5_mod_6_l774_77446

-- Part 1: Infinitely many primes congruent to 3 modulo 4
theorem infinitely_many_primes_3_mod_4 : 
  ∀ S : Finset Nat, ∃ p : Nat, p ∉ S ∧ Prime p ∧ p % 4 = 3 := by sorry

-- Part 2: Infinitely many primes congruent to 5 modulo 6
theorem infinitely_many_primes_5_mod_6 : 
  ∀ S : Finset Nat, ∃ p : Nat, p ∉ S ∧ Prime p ∧ p % 6 = 5 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_3_mod_4_infinitely_many_primes_5_mod_6_l774_77446


namespace NUMINAMATH_CALUDE_cost_of_pencils_l774_77434

/-- Given that 100 pencils cost $30, prove that 1500 pencils cost $450. -/
theorem cost_of_pencils :
  (∃ (cost_per_100 : ℝ), cost_per_100 = 30 ∧ 
   (1500 / 100) * cost_per_100 = 450) :=
by sorry

end NUMINAMATH_CALUDE_cost_of_pencils_l774_77434


namespace NUMINAMATH_CALUDE_repeated_root_implies_m_equals_two_l774_77456

/-- Given that the equation (m-1)/(x-1) - x/(x-1) = 0 has a repeated root, prove that m = 2 -/
theorem repeated_root_implies_m_equals_two (m : ℝ) :
  (∃ x : ℝ, x ≠ 1 ∧ (m - 1) / (x - 1) - x / (x - 1) = 0 ∧
   ∀ y : ℝ, y ≠ 1 → ((m - 1) / (y - 1) - y / (y - 1) = 0 → y = x)) →
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_repeated_root_implies_m_equals_two_l774_77456


namespace NUMINAMATH_CALUDE_range_of_m_l774_77407

def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 11

def q (x m : ℝ) : Prop := 1 - 3*m ≤ x ∧ x ≤ 3 + m

theorem range_of_m (h : ∀ x m : ℝ, m > 0 → (¬(p x) → ¬(q x m)) ∧ ∃ x', ¬(q x' m) ∧ p x') :
  ∀ m : ℝ, m ∈ Set.Ici 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l774_77407


namespace NUMINAMATH_CALUDE_distributive_property_division_l774_77464

theorem distributive_property_division (a b c : ℝ) (hc : c ≠ 0) :
  (a + b) * c = a * c + b * c → (a + b) / c = a / c + b / c := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_division_l774_77464


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l774_77444

def inequality_system (x : ℝ) : Prop :=
  (3*x - 2)/3 ≥ 1 ∧ 3*x + 5 > 4*x - 2

def integer_solutions : Set ℤ := {2, 3, 4, 5, 6}

theorem inequality_system_integer_solutions :
  ∀ (n : ℤ), n ∈ integer_solutions ↔ inequality_system (n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l774_77444


namespace NUMINAMATH_CALUDE_monotonic_sequence_divisor_property_l774_77471

def divisor_count (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def is_monotonic_increasing (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i < j → a i < a j

theorem monotonic_sequence_divisor_property (a : ℕ → ℕ) :
  is_monotonic_increasing a →
  (∀ i j : ℕ, divisor_count (i + j) = divisor_count (a i + a j)) →
  ∀ n : ℕ, a n = n :=
sorry

end NUMINAMATH_CALUDE_monotonic_sequence_divisor_property_l774_77471


namespace NUMINAMATH_CALUDE_books_read_l774_77465

theorem books_read (total_books : ℕ) (total_movies : ℕ) (movies_watched : ℕ) (books_read : ℕ) :
  total_books = 10 →
  total_movies = 11 →
  movies_watched = 12 →
  books_read = (min movies_watched total_movies) + 1 →
  books_read = 12 := by
sorry

end NUMINAMATH_CALUDE_books_read_l774_77465


namespace NUMINAMATH_CALUDE_imaginary_part_of_1_plus_2i_l774_77477

theorem imaginary_part_of_1_plus_2i : Complex.im (1 + 2*Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_1_plus_2i_l774_77477


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l774_77441

theorem binomial_expansion_problem (n : ℕ) :
  (∀ k, 0 ≤ k ∧ k ≤ n → Nat.choose n 6 ≥ Nat.choose n k) ∧
  (∀ k, k ≠ 6 → Nat.choose n 6 > Nat.choose n k) →
  n = 12 ∧ 2^(n+4) % 7 = 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l774_77441


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l774_77488

/-- The perimeter of a rectangular field with length 7/5 of its width and width of 80 meters is 384 meters. -/
theorem rectangular_field_perimeter : 
  ∀ (length width : ℝ),
  length = (7/5) * width →
  width = 80 →
  2 * (length + width) = 384 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l774_77488


namespace NUMINAMATH_CALUDE_cherries_used_for_pie_l774_77437

theorem cherries_used_for_pie (initial_cherries remaining_cherries : ℕ) 
  (h1 : initial_cherries = 77)
  (h2 : remaining_cherries = 17) :
  initial_cherries - remaining_cherries = 60 := by
sorry

end NUMINAMATH_CALUDE_cherries_used_for_pie_l774_77437


namespace NUMINAMATH_CALUDE_smallest_square_enclosing_circle_area_l774_77426

-- Define the radius of the circle
def radius : ℝ := 5

-- Define the area of the smallest enclosing square
def smallest_enclosing_square_area (r : ℝ) : ℝ := (2 * r) ^ 2

-- Theorem statement
theorem smallest_square_enclosing_circle_area :
  smallest_enclosing_square_area radius = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_enclosing_circle_area_l774_77426


namespace NUMINAMATH_CALUDE_tree_height_difference_l774_77432

theorem tree_height_difference : 
  let pine_height : ℚ := 49/4
  let maple_height : ℚ := 75/4
  maple_height - pine_height = 13/2 := by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l774_77432


namespace NUMINAMATH_CALUDE_larry_jogging_time_l774_77409

/-- Calculates the total jogging time in hours for two weeks given daily jogging time and days jogged each week -/
def total_jogging_time (daily_time : ℕ) (days_week1 : ℕ) (days_week2 : ℕ) : ℚ :=
  ((daily_time * days_week1 + daily_time * days_week2) : ℚ) / 60

/-- Theorem stating that Larry's total jogging time for two weeks is 4 hours -/
theorem larry_jogging_time :
  total_jogging_time 30 3 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_larry_jogging_time_l774_77409


namespace NUMINAMATH_CALUDE_x_intercepts_count_l774_77490

-- Define the polynomial
def f (x : ℝ) : ℝ := (x - 5) * (x^2 + 8*x + 12)

-- State the theorem
theorem x_intercepts_count : 
  ∃ (a b c : ℝ), (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) := by
  sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l774_77490


namespace NUMINAMATH_CALUDE_burgers_spent_l774_77422

def total_allowance : ℚ := 50

def movies_fraction : ℚ := 1/4
def music_fraction : ℚ := 3/10
def ice_cream_fraction : ℚ := 2/5

def burgers_amount : ℚ := total_allowance - (movies_fraction * total_allowance + music_fraction * total_allowance + ice_cream_fraction * total_allowance)

theorem burgers_spent :
  burgers_amount = 5/2 := by sorry

end NUMINAMATH_CALUDE_burgers_spent_l774_77422


namespace NUMINAMATH_CALUDE_quadratic_roots_not_uniformly_increased_l774_77423

theorem quadratic_roots_not_uniformly_increased (b c : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + b*x1 + c = 0 ∧ x2^2 + b*x2 + c = 0) :
  ¬∃ y1 y2 : ℝ, y1 ≠ y2 ∧ 
    y1^2 + (b+1)*y1 + (c+1) = 0 ∧ 
    y2^2 + (b+1)*y2 + (c+1) = 0 ∧
    ∃ x1 x2 : ℝ, x1^2 + b*x1 + c = 0 ∧ x2^2 + b*x2 + c = 0 ∧ y1 = x1 + 1 ∧ y2 = x2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_not_uniformly_increased_l774_77423


namespace NUMINAMATH_CALUDE_triangle_proof_l774_77462

/-- Given an acute triangle ABC with sides a and b, prove angle A and area. -/
theorem triangle_proof 
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Sides of the triangle
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) -- Acute triangle condition
  (h_a : a = Real.sqrt 7) -- Side a
  (h_b : b = 3) -- Side b
  (h_sin_sum : Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3) -- Given equation
  : A = π / 3 ∧ (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_proof_l774_77462


namespace NUMINAMATH_CALUDE_max_value_of_expression_l774_77401

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  (∃ m : ℝ, ∀ a b : ℝ, a + b = 5 → 
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 ≤ m ∧
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 = m) ∧
  (∀ m : ℝ, (∀ a b : ℝ, a + b = 5 → 
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 ≤ m ∧
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 = m) → 
  m = 625/4) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l774_77401
