import Mathlib

namespace NUMINAMATH_GPT_perpendicular_chords_square_sum_l2225_222512

theorem perpendicular_chords_square_sum (d : ℝ) (r : ℝ) (x y : ℝ) 
  (h1 : r = d / 2)
  (h2 : x = r)
  (h3 : y = r) 
  : (x^2 + y^2) + (x^2 + y^2) = d^2 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_chords_square_sum_l2225_222512


namespace NUMINAMATH_GPT_no_solutions_exists_unique_l2225_222537

def is_solution (a b c x y z : ℤ) : Prop :=
  2 * x - b * y + z = 2 * b ∧
  a * x + 5 * y - c * z = a

def no_solutions_for (a b c : ℤ) : Prop :=
  ∀ x y z : ℤ, ¬ is_solution a b c x y z

theorem no_solutions_exists_unique (a b c : ℤ) :
  (a = -2 ∧ b = 5 ∧ c = 1) ∨
  (a = 2 ∧ b = -5 ∧ c = -1) ∨
  (a = 10 ∧ b = -1 ∧ c = -5) ↔
  no_solutions_for a b c := 
sorry

end NUMINAMATH_GPT_no_solutions_exists_unique_l2225_222537


namespace NUMINAMATH_GPT_find_x_l2225_222583

theorem find_x {x y : ℝ} (h1 : 3 * x - 2 * y = 7) (h2 : x^2 + 3 * y = 17) : x = 3.5 :=
sorry

end NUMINAMATH_GPT_find_x_l2225_222583


namespace NUMINAMATH_GPT_abc_equal_l2225_222542

theorem abc_equal (a b c : ℝ) (h : a^2 + b^2 + c^2 - ab - bc - ac = 0) : a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_abc_equal_l2225_222542


namespace NUMINAMATH_GPT_product_of_intersection_coordinates_l2225_222595

theorem product_of_intersection_coordinates :
  let circle1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 4)^2 = 4}
  let circle2 := {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 - 4)^2 = 9}
  ∃ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 ∧ p.1 * p.2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_product_of_intersection_coordinates_l2225_222595


namespace NUMINAMATH_GPT_find_m_value_l2225_222551

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)
def vector_sum (m : ℝ) : ℝ × ℝ := (1 + 3, m - 2)

-- Define the condition that vector_sum is parallel to vector_b
def vectors_parallel (m : ℝ) : Prop :=
  let (x1, y1) := vector_sum m
  let (x2, y2) := vector_b
  x1 * y2 - x2 * y1 = 0

-- The statement to prove
theorem find_m_value : ∃ m : ℝ, vectors_parallel m ∧ m = -2 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_value_l2225_222551


namespace NUMINAMATH_GPT_gazprom_rnd_costs_calc_l2225_222554

theorem gazprom_rnd_costs_calc (R_D_t ΔAPL_t1 : ℝ) (h1 : R_D_t = 3157.61) (h2 : ΔAPL_t1 = 0.69) :
  R_D_t / ΔAPL_t1 = 4576 :=
by
  sorry

end NUMINAMATH_GPT_gazprom_rnd_costs_calc_l2225_222554


namespace NUMINAMATH_GPT_sum_of_coordinates_l2225_222564

theorem sum_of_coordinates (C D : ℝ × ℝ) (hC : C = (0, 0)) (hD : D.snd = 6) (h_slope : (D.snd - C.snd) / (D.fst - C.fst) = 3/4) : D.fst + D.snd = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_l2225_222564


namespace NUMINAMATH_GPT_remainder_when_divided_by_296_l2225_222549

theorem remainder_when_divided_by_296 (N : ℤ) (Q : ℤ) (R : ℤ)
  (h1 : N % 37 = 1)
  (h2 : N = 296 * Q + R)
  (h3 : 0 ≤ R) 
  (h4 : R < 296) :
  R = 260 := 
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_296_l2225_222549


namespace NUMINAMATH_GPT_find_a10_l2225_222504

def seq (a : ℕ → ℝ) : Prop :=
∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p + a q

theorem find_a10 (a : ℕ → ℝ) (h_seq : seq a) (h_a2 : a 2 = -6) : a 10 = -30 :=
by
  sorry

end NUMINAMATH_GPT_find_a10_l2225_222504


namespace NUMINAMATH_GPT_equation_of_line_through_point_with_equal_intercepts_l2225_222515

open LinearAlgebra

theorem equation_of_line_through_point_with_equal_intercepts :
  ∃ (a b c : ℝ), (a * 1 + b * 2 + c = 0) ∧ (a * b < 0) ∧ ∀ x y : ℝ, 
  (a * x + b * y + c = 0 ↔ (2 * x - y = 0 ∨ x + y - 3 = 0)) :=
sorry

end NUMINAMATH_GPT_equation_of_line_through_point_with_equal_intercepts_l2225_222515


namespace NUMINAMATH_GPT_intersection_complement_A_U_B_l2225_222505

def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def set_A : Set ℕ := {2, 4, 6}
def set_B : Set ℕ := {1, 3, 5, 7}

theorem intersection_complement_A_U_B :
  set_A ∩ (universal_set \ set_B) = {2, 4, 6} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_complement_A_U_B_l2225_222505


namespace NUMINAMATH_GPT_alyssa_spent_on_grapes_l2225_222500

theorem alyssa_spent_on_grapes (t c g : ℝ) (h1 : t = 21.93) (h2 : c = 9.85) (h3 : t = g + c) : g = 12.08 :=
by
  sorry

end NUMINAMATH_GPT_alyssa_spent_on_grapes_l2225_222500


namespace NUMINAMATH_GPT_depreciation_rate_l2225_222511

theorem depreciation_rate (initial_value final_value : ℝ) (years : ℕ) (r : ℝ)
  (h_initial : initial_value = 128000)
  (h_final : final_value = 54000)
  (h_years : years = 3)
  (h_equation : final_value = initial_value * (1 - r) ^ years) :
  r = 0.247 :=
sorry

end NUMINAMATH_GPT_depreciation_rate_l2225_222511


namespace NUMINAMATH_GPT_find_extrema_l2225_222584

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

theorem find_extrema :
  (∀ x, f x ≤ 17) ∧ (∃ x, f x = 17) ∧ (∀ x, f x ≥ -47) ∧ (∃ x, f x = -47) :=
by
  sorry

end NUMINAMATH_GPT_find_extrema_l2225_222584


namespace NUMINAMATH_GPT_absolute_difference_AB_l2225_222569

noncomputable def A : Real := 12 / 7
noncomputable def B : Real := 20 / 7

theorem absolute_difference_AB : |A - B| = 8 / 7 := by
  sorry

end NUMINAMATH_GPT_absolute_difference_AB_l2225_222569


namespace NUMINAMATH_GPT_cute_2020_all_integers_cute_l2225_222579

-- Definition of "cute" integer
def is_cute (n : ℤ) : Prop :=
  ∃ (a b c d : ℤ), n = a^2 + b^3 + c^3 + d^5

-- Proof problem 1: Assert that 2020 is cute
theorem cute_2020 : is_cute 2020 :=
sorry

-- Proof problem 2: Assert that every integer is cute
theorem all_integers_cute (n : ℤ) : is_cute n :=
sorry

end NUMINAMATH_GPT_cute_2020_all_integers_cute_l2225_222579


namespace NUMINAMATH_GPT_solve_inequality_l2225_222535

def p (x : ℝ) : ℝ := x^2 - 5*x + 3

theorem solve_inequality (x : ℝ) : 
  abs (p x) < 9 ↔ (-1 < x ∧ x < 3) ∨ (4 < x ∧ x < 6) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l2225_222535


namespace NUMINAMATH_GPT_algebraic_expression_positive_l2225_222557

theorem algebraic_expression_positive (a b : ℝ) : 
  a^2 + b^2 + 4*b - 2*a + 6 > 0 :=
by sorry

end NUMINAMATH_GPT_algebraic_expression_positive_l2225_222557


namespace NUMINAMATH_GPT_smallest_positive_integer_l2225_222575

theorem smallest_positive_integer (a : ℤ)
  (h1 : a % 2 = 1)
  (h2 : a % 3 = 2)
  (h3 : a % 4 = 3)
  (h4 : a % 5 = 4) :
  a = 59 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_l2225_222575


namespace NUMINAMATH_GPT_metallic_sheet_dimension_l2225_222589

theorem metallic_sheet_dimension (x : ℝ) (h₁ : ∀ (l w h : ℝ), l = x - 8 → w = 28 → h = 4 → l * w * h = 4480) : x = 48 :=
sorry

end NUMINAMATH_GPT_metallic_sheet_dimension_l2225_222589


namespace NUMINAMATH_GPT_tiles_needed_l2225_222539

def ft_to_inch (x : ℕ) : ℕ := x * 12

def height_ft : ℕ := 10
def length_ft : ℕ := 15
def tile_size_sq_inch : ℕ := 1

def height_inch : ℕ := ft_to_inch height_ft
def length_inch : ℕ := ft_to_inch length_ft
def area_sq_inch : ℕ := height_inch * length_inch

theorem tiles_needed : 
  height_ft = 10 ∧ length_ft = 15 ∧ tile_size_sq_inch = 1 →
  area_sq_inch = 21600 :=
by
  intro h
  exact sorry

end NUMINAMATH_GPT_tiles_needed_l2225_222539


namespace NUMINAMATH_GPT_find_m_value_l2225_222524

theorem find_m_value
    (x y m : ℝ)
    (hx : x = -1)
    (hy : y = 2)
    (hxy : m * x + 2 * y = 1) :
    m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l2225_222524


namespace NUMINAMATH_GPT_part1_part2_l2225_222547

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x - (a / 2) * x^2

-- Define the line l
noncomputable def l (k : ℤ) (x : ℝ) : ℝ := (k - 2) * x - k + 1

-- Theorem for part (1)
theorem part1 (x : ℝ) (a : ℝ) (h₁ : e ≤ x) (h₂ : x ≤ e^2) (h₃ : f a x > 0) : a < 2 / e :=
sorry

-- Theorem for part (2)
theorem part2 (k : ℤ) (h₁ : a = 0) (h₂ : ∀ (x : ℝ), 1 < x → f 0 x > l k x) : k ≤ 4 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2225_222547


namespace NUMINAMATH_GPT_fraction_of_girls_is_one_third_l2225_222556

-- Define the number of children and number of boys
def total_children : Nat := 45
def boys : Nat := 30

-- Calculate the number of girls
def girls : Nat := total_children - boys

-- Calculate the fraction of girls
def fraction_of_girls : Rat := (girls : Rat) / (total_children : Rat)

theorem fraction_of_girls_is_one_third : fraction_of_girls = 1 / 3 :=
by
  sorry -- Proof is not required

end NUMINAMATH_GPT_fraction_of_girls_is_one_third_l2225_222556


namespace NUMINAMATH_GPT_square_pyramid_sum_l2225_222570

-- Define the number of faces, edges, and vertices of a square pyramid.
def faces_square_base : Nat := 1
def faces_lateral : Nat := 4
def edges_base : Nat := 4
def edges_lateral : Nat := 4
def vertices_base : Nat := 4
def vertices_apex : Nat := 1

-- Summing the faces, edges, and vertices
def total_faces : Nat := faces_square_base + faces_lateral
def total_edges : Nat := edges_base + edges_lateral
def total_vertices : Nat := vertices_base + vertices_apex

theorem square_pyramid_sum : (total_faces + total_edges + total_vertices = 18) :=
by
  sorry

end NUMINAMATH_GPT_square_pyramid_sum_l2225_222570


namespace NUMINAMATH_GPT_num_squares_less_than_1000_with_ones_digit_2_3_or_4_l2225_222523

-- Define a function that checks if the one's digit of a number is one of 2, 3, or 4.
def ends_in (n : ℕ) (d : ℕ) : Prop := n % 10 = d

-- Define the main theorem to prove
theorem num_squares_less_than_1000_with_ones_digit_2_3_or_4 : 
  ∃ n, n = 6 ∧ ∀ m < 1000, ∃ k, m = k^2 → ends_in m 2 ∨ ends_in m 3 ∨ ends_in m 4 :=
sorry

end NUMINAMATH_GPT_num_squares_less_than_1000_with_ones_digit_2_3_or_4_l2225_222523


namespace NUMINAMATH_GPT_joy_reading_rate_l2225_222506

theorem joy_reading_rate
  (h1 : ∀ t: ℕ, t = 20 → ∀ p: ℕ, p = 8 → ∀ t': ℕ, t' = 60 → ∃ p': ℕ, p' = (p * t') / t)
  (h2 : ∀ t: ℕ, t = 5 * 60 → ∀ p: ℕ, p = 120):
  ∃ r: ℕ, r = 24 :=
by
  sorry

end NUMINAMATH_GPT_joy_reading_rate_l2225_222506


namespace NUMINAMATH_GPT_solve_for_x_l2225_222562

theorem solve_for_x (x : ℝ) (h : 0.60 * 500 = 0.50 * x) : x = 600 :=
  sorry

end NUMINAMATH_GPT_solve_for_x_l2225_222562


namespace NUMINAMATH_GPT_orchard_produce_l2225_222521

theorem orchard_produce (num_apple_trees num_orange_trees apple_baskets_per_tree apples_per_basket orange_baskets_per_tree oranges_per_basket : ℕ) 
  (h1 : num_apple_trees = 50) 
  (h2 : num_orange_trees = 30) 
  (h3 : apple_baskets_per_tree = 25) 
  (h4 : apples_per_basket = 18)
  (h5 : orange_baskets_per_tree = 15) 
  (h6 : oranges_per_basket = 12) 
: (num_apple_trees * (apple_baskets_per_tree * apples_per_basket) = 22500) ∧ 
  (num_orange_trees * (orange_baskets_per_tree * oranges_per_basket) = 5400) :=
  by 
  sorry

end NUMINAMATH_GPT_orchard_produce_l2225_222521


namespace NUMINAMATH_GPT_infinite_solutions_congruence_l2225_222585

theorem infinite_solutions_congruence (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ᶠ x in at_top, a ^ x + x ≡ b [MOD c] :=
sorry

end NUMINAMATH_GPT_infinite_solutions_congruence_l2225_222585


namespace NUMINAMATH_GPT_eval_special_op_l2225_222596

variable {α : Type*} [LinearOrderedField α]

def op (a b : α) : α := (a - b) ^ 2

theorem eval_special_op (x y z : α) : op ((x - y + z)^2) ((y - x - z)^2) = 0 := by
  sorry

end NUMINAMATH_GPT_eval_special_op_l2225_222596


namespace NUMINAMATH_GPT_first_term_of_arithmetic_sequence_l2225_222591

theorem first_term_of_arithmetic_sequence (a : ℕ) (median last_term : ℕ) 
  (h_arithmetic_progression : true) (h_median : median = 1010) (h_last_term : last_term = 2015) :
  a = 5 :=
by
  have h1 : 2 * median = 2020 := by sorry
  have h2 : last_term + a = 2020 := by sorry
  have h3 : 2015 + a = 2020 := by sorry
  have h4 : a = 2020 - 2015 := by sorry
  have h5 : a = 5 := by sorry
  exact h5

end NUMINAMATH_GPT_first_term_of_arithmetic_sequence_l2225_222591


namespace NUMINAMATH_GPT_train_distance_difference_l2225_222594

theorem train_distance_difference:
  ∀ (D1 D2 : ℕ) (t : ℕ), 
    (D1 = 20 * t) →            -- Slower train's distance
    (D2 = 25 * t) →           -- Faster train's distance
    (D1 + D2 = 450) →         -- Total distance between stations
    (D2 - D1 = 50) := 
by
  intros D1 D2 t h1 h2 h3
  sorry

end NUMINAMATH_GPT_train_distance_difference_l2225_222594


namespace NUMINAMATH_GPT_intersection_eq_M_l2225_222587

-- Define the sets M and N according to the given conditions
def M : Set ℝ := {x : ℝ | x^2 - x < 0}
def N : Set ℝ := {x : ℝ | |x| < 2}

-- The 'theorem' statement to prove M ∩ N = M
theorem intersection_eq_M : M ∩ N = M :=
  sorry

end NUMINAMATH_GPT_intersection_eq_M_l2225_222587


namespace NUMINAMATH_GPT_beads_probability_l2225_222592

/-
  Four red beads, three white beads, and two blue beads are placed in a line in random order.
  Prove that the probability that no two neighboring beads are the same color is 1/70.
-/
theorem beads_probability :
  let total_permutations := Nat.factorial 9 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 2)
  let valid_permutations := 18 -- conservative estimate from the solution
  (valid_permutations : ℚ) / total_permutations = 1 / 70 :=
by
  let total_permutations := Nat.factorial 9 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 2)
  let valid_permutations := 18
  show (valid_permutations : ℚ) / total_permutations = 1 / 70
  -- skipping proof details
  sorry

end NUMINAMATH_GPT_beads_probability_l2225_222592


namespace NUMINAMATH_GPT_M_is_real_l2225_222555

open Complex

-- Define the condition that characterizes the set M
def M (Z : ℂ) : Prop := (Z - 1)^2 = abs (Z - 1)^2

-- Prove that M is exactly the set of real numbers
theorem M_is_real : ∀ (Z : ℂ), M Z ↔ Z.im = 0 :=
by
  sorry

end NUMINAMATH_GPT_M_is_real_l2225_222555


namespace NUMINAMATH_GPT_probability_floor_sqrt_even_l2225_222516

/-- Suppose x and y are chosen randomly and uniformly from (0,1). The probability that
    ⌊√(x/y)⌋ is even is 1 - π²/24. -/
theorem probability_floor_sqrt_even (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
  (1 - Real.pi ^ 2 / 24) = sorry :=
sorry

end NUMINAMATH_GPT_probability_floor_sqrt_even_l2225_222516


namespace NUMINAMATH_GPT_earrings_ratio_l2225_222540

theorem earrings_ratio :
  ∀ (total_pairs : ℕ) (given_pairs : ℕ) (total_earrings : ℕ) (given_earrings : ℕ),
    total_pairs = 12 →
    given_pairs = total_pairs / 2 →
    total_earrings = total_pairs * 2 →
    given_earrings = total_earrings / 2 →
    total_earrings = 36 →
    given_earrings = 12 →
    (total_earrings / given_earrings = 3) :=
by
  sorry

end NUMINAMATH_GPT_earrings_ratio_l2225_222540


namespace NUMINAMATH_GPT_dishonest_shopkeeper_gain_l2225_222526

-- Conditions: false weight used by shopkeeper
def false_weight : ℚ := 930
def true_weight : ℚ := 1000

-- Correct answer: gain percentage
def gain_percentage (false_weight true_weight : ℚ) : ℚ :=
  ((true_weight - false_weight) / false_weight) * 100

theorem dishonest_shopkeeper_gain :
  gain_percentage false_weight true_weight = 7.53 := by
  sorry

end NUMINAMATH_GPT_dishonest_shopkeeper_gain_l2225_222526


namespace NUMINAMATH_GPT_betty_needs_more_flies_l2225_222581

def betty_frog_food (daily_flies: ℕ) (days_per_week: ℕ) (morning_catch: ℕ) 
  (afternoon_catch: ℕ) (flies_escaped: ℕ) : ℕ :=
  days_per_week * daily_flies - (morning_catch + afternoon_catch - flies_escaped)

theorem betty_needs_more_flies :
  betty_frog_food 2 7 5 6 1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_betty_needs_more_flies_l2225_222581


namespace NUMINAMATH_GPT_largest_six_consecutive_nonprime_under_50_l2225_222577

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → (m = 1 ∨ m = n)

def consecutiveNonPrimes (m : ℕ) : Prop :=
  ∀ i : ℕ, i < 6 → ¬ isPrime (m + i)

theorem largest_six_consecutive_nonprime_under_50 (n : ℕ) :
  (n < 50 ∧ consecutiveNonPrimes n) →
  n + 5 = 35 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_largest_six_consecutive_nonprime_under_50_l2225_222577


namespace NUMINAMATH_GPT_sum_of_cubes_consecutive_integers_l2225_222550

theorem sum_of_cubes_consecutive_integers (x : ℕ) (h1 : 0 < x) (h2 : x * (x + 1) * (x + 2) = 12 * (3 * x + 3)) :
  x^3 + (x + 1)^3 + (x + 2)^3 = 216 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_sum_of_cubes_consecutive_integers_l2225_222550


namespace NUMINAMATH_GPT_remaining_structure_volume_and_surface_area_l2225_222546

-- Define the dimensions of the large cube and the small cubes
def large_cube_volume := 12 * 12 * 12
def small_cube_volume := 2 * 2 * 2

-- Define the number of smaller cubes in the large cube
def num_small_cubes := (12 / 2) * (12 / 2) * (12 / 2)

-- Define the number of smaller cubes removed (central on each face and very center)
def removed_cubes := 7

-- The volume of a small cube after removing its center unit
def single_small_cube_remaining_volume := small_cube_volume - 1

-- Calculate the remaining volume after all removals
def remaining_volume := (num_small_cubes - removed_cubes) * single_small_cube_remaining_volume

-- Initial surface area of a small cube and increase per removal of central unit
def single_small_cube_initial_surface_area := 6 * 4 -- 6 faces of 2*2*2 cube, each face has 4 units
def single_small_cube_surface_increase := 6

-- Calculate the adjusted surface area considering internal faces' reduction
def single_cube_adjusted_surface_area := single_small_cube_initial_surface_area + single_small_cube_surface_increase
def total_initial_surface_area := single_cube_adjusted_surface_area * (num_small_cubes - removed_cubes)
def total_internal_faces_area := (num_small_cubes - removed_cubes) * 2 * 4
def final_surface_area := total_initial_surface_area - total_internal_faces_area

theorem remaining_structure_volume_and_surface_area :
  remaining_volume = 1463 ∧ final_surface_area = 4598 :=
by
  -- Proof logic goes here
  sorry

end NUMINAMATH_GPT_remaining_structure_volume_and_surface_area_l2225_222546


namespace NUMINAMATH_GPT_part1_inequality_part2_range_of_a_l2225_222525

-- Definitions and conditions
def f (x a : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- First proof problem for a = 1
theorem part1_inequality (x : ℝ) : f x 1 > 1 ↔ x > 1/2 :=
by sorry

-- Second proof problem for range of a when f(x) > x in (0, 1)
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → f x a > x) → 0 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_part1_inequality_part2_range_of_a_l2225_222525


namespace NUMINAMATH_GPT_doubled_container_volume_l2225_222573

theorem doubled_container_volume (v : ℝ) (h₁ : v = 4) (h₂ : ∀ l w h : ℝ, v = l * w * h) : 8 * v = 32 := 
by
  -- The proof will go here, this is just the statement
  sorry

end NUMINAMATH_GPT_doubled_container_volume_l2225_222573


namespace NUMINAMATH_GPT_probability_at_least_one_multiple_of_4_is_correct_l2225_222563

noncomputable def probability_at_least_one_multiple_of_4 : ℚ :=
  let total_numbers := 100
  let multiples_of_4 := 25
  let non_multiples_of_4 := total_numbers - multiples_of_4
  let p_non_multiple := (non_multiples_of_4 : ℚ) / total_numbers
  let p_both_non_multiples := p_non_multiple^2
  let p_at_least_one_multiple := 1 - p_both_non_multiples
  p_at_least_one_multiple

theorem probability_at_least_one_multiple_of_4_is_correct :
  probability_at_least_one_multiple_of_4 = 7 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_multiple_of_4_is_correct_l2225_222563


namespace NUMINAMATH_GPT_power_calculation_l2225_222597

theorem power_calculation : (3^4)^2 = 6561 := by 
  sorry

end NUMINAMATH_GPT_power_calculation_l2225_222597


namespace NUMINAMATH_GPT_red_balls_in_bag_l2225_222530

theorem red_balls_in_bag : 
  ∃ (r : ℕ), (r * (r - 1) = 22) ∧ (r ≤ 12) :=
by { sorry }

end NUMINAMATH_GPT_red_balls_in_bag_l2225_222530


namespace NUMINAMATH_GPT_function_range_l2225_222590

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 2 * Real.sin x

theorem function_range : 
  ∀ x : ℝ, (0 < x ∧ x < Real.pi) → 1 ≤ f x ∧ f x ≤ 3 / 2 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_function_range_l2225_222590


namespace NUMINAMATH_GPT_mudit_age_l2225_222538

theorem mudit_age :
    ∃ x : ℤ, x + 16 = 3 * (x - 4) ∧ x = 14 :=
by
  use 14
  sorry -- Proof goes here

end NUMINAMATH_GPT_mudit_age_l2225_222538


namespace NUMINAMATH_GPT_find_a_l2225_222536

def A (x : ℝ) : Prop := x^2 + 6 * x < 0
def B (a x : ℝ) : Prop := x^2 - (a - 2) * x - 2 * a < 0
def U (x : ℝ) : Prop := -6 < x ∧ x < 5

theorem find_a : (∀ x, A x ∨ ∃ a, B a x) = U x -> a = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2225_222536


namespace NUMINAMATH_GPT_probability_wife_selection_l2225_222548

theorem probability_wife_selection (P_H P_only_one P_W : ℝ)
  (h1 : P_H = 1 / 7)
  (h2 : P_only_one = 0.28571428571428575)
  (h3 : P_only_one = (P_H * (1 - P_W)) + (P_W * (1 - P_H))) :
  P_W = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_wife_selection_l2225_222548


namespace NUMINAMATH_GPT_find_extra_digit_l2225_222571

theorem find_extra_digit (x y a : ℕ) (hx : x + y = 23456) (h10x : 10 * x + a + y = 55555) (ha : 0 ≤ a ∧ a ≤ 9) : a = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_extra_digit_l2225_222571


namespace NUMINAMATH_GPT_cakes_given_away_l2225_222510

theorem cakes_given_away 
  (cakes_baked : ℕ) 
  (candles_per_cake : ℕ) 
  (total_candles : ℕ) 
  (cakes_given : ℕ) 
  (cakes_left : ℕ) 
  (h1 : cakes_baked = 8) 
  (h2 : candles_per_cake = 6) 
  (h3 : total_candles = 36) 
  (h4 : total_candles = candles_per_cake * cakes_left) 
  (h5 : cakes_given = cakes_baked - cakes_left) 
  : cakes_given = 2 :=
sorry

end NUMINAMATH_GPT_cakes_given_away_l2225_222510


namespace NUMINAMATH_GPT_map_representation_l2225_222586

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end NUMINAMATH_GPT_map_representation_l2225_222586


namespace NUMINAMATH_GPT_relationship_between_number_and_value_l2225_222544

theorem relationship_between_number_and_value (n v : ℝ) (h1 : n = 7) (h2 : n - 4 = 21 * v) : v = 1 / 7 :=
  sorry

end NUMINAMATH_GPT_relationship_between_number_and_value_l2225_222544


namespace NUMINAMATH_GPT_conditional_probability_P_B_given_A_l2225_222565

-- Let E be an enumeration type with exactly five values, each representing one attraction.
inductive Attraction : Type
| dayu_yashan : Attraction
| qiyunshan : Attraction
| tianlongshan : Attraction
| jiulianshan : Attraction
| sanbaishan : Attraction

open Attraction

-- Define A and B's choices as random variables.
axiom A_choice : Attraction
axiom B_choice : Attraction

-- Event A is that A and B choose different attractions.
def event_A : Prop := A_choice ≠ B_choice

-- Event B is that A and B each choose Chongyi Qiyunshan.
def event_B : Prop := A_choice = qiyunshan ∧ B_choice = qiyunshan

-- Calculate the conditional probability P(B|A)
theorem conditional_probability_P_B_given_A : 
  (1 - (1 / 5)) * (1 - (1 / 5)) = 2 / 5 :=
sorry

end NUMINAMATH_GPT_conditional_probability_P_B_given_A_l2225_222565


namespace NUMINAMATH_GPT_part_a_l2225_222578

theorem part_a (a b c : ℝ) (m : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hm : 0 < m) :
  (a + b)^m + (b + c)^m + (c + a)^m ≤ 2^m * (a^m + b^m + c^m) :=
by
  sorry

end NUMINAMATH_GPT_part_a_l2225_222578


namespace NUMINAMATH_GPT_value_of_f_3_div_2_l2225_222552

noncomputable def f : ℝ → ℝ := sorry

axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x
axiom even_f : ∀ x : ℝ, f (x) = f (-x)
axiom f_in_0_1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f (x) = x + 1

theorem value_of_f_3_div_2 : f (3 / 2) = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_value_of_f_3_div_2_l2225_222552


namespace NUMINAMATH_GPT_correct_option_c_l2225_222507

theorem correct_option_c (x : ℝ) : -2 * (x + 1) = -2 * x - 2 :=
  by
  -- Proof can be omitted
  sorry

end NUMINAMATH_GPT_correct_option_c_l2225_222507


namespace NUMINAMATH_GPT_files_deleted_is_3_l2225_222508

-- Define the initial number of files
def initial_files : Nat := 24

-- Define the remaining number of files
def remaining_files : Nat := 21

-- Define the number of files deleted
def files_deleted : Nat := initial_files - remaining_files

-- Prove that the number of files deleted is 3
theorem files_deleted_is_3 : files_deleted = 3 :=
by
  sorry

end NUMINAMATH_GPT_files_deleted_is_3_l2225_222508


namespace NUMINAMATH_GPT_inverse_of_h_l2225_222559

def h (x : ℝ) : ℝ := 3 + 6 * x

noncomputable def k (x : ℝ) : ℝ := (x - 3) / 6

theorem inverse_of_h : ∀ x, h (k x) = x :=
by
  intro x
  unfold h k
  sorry

end NUMINAMATH_GPT_inverse_of_h_l2225_222559


namespace NUMINAMATH_GPT_average_rainfall_virginia_l2225_222514

noncomputable def average_rainfall : ℝ :=
  (3.79 + 4.5 + 3.95 + 3.09 + 4.67) / 5

theorem average_rainfall_virginia : average_rainfall = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_rainfall_virginia_l2225_222514


namespace NUMINAMATH_GPT_angle_sum_unique_l2225_222533

theorem angle_sum_unique (α β : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : β ∈ Set.Ioo (π / 2) π) 
  (h3 : Real.tan α + Real.tan β - Real.tan α * Real.tan β + 1 = 0) : 
  α + β = 7 * π / 4 :=
sorry

end NUMINAMATH_GPT_angle_sum_unique_l2225_222533


namespace NUMINAMATH_GPT_inverse_proportion_l2225_222560

theorem inverse_proportion (x : ℝ) (y : ℝ) (f₁ f₂ f₃ f₄ : ℝ → ℝ) (h₁ : f₁ x = 2 * x) (h₂ : f₂ x = x / 2) (h₃ : f₃ x = 2 / x) (h₄ : f₄ x = 2 / (x - 1)) :
  f₃ x * x = 2 := sorry

end NUMINAMATH_GPT_inverse_proportion_l2225_222560


namespace NUMINAMATH_GPT_highest_throw_is_37_feet_l2225_222520

theorem highest_throw_is_37_feet :
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  max (max C1 (max C2 C3)) (max J1 (max J2 J3)) = 37 := by
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  sorry

end NUMINAMATH_GPT_highest_throw_is_37_feet_l2225_222520


namespace NUMINAMATH_GPT_smallest_multiple_of_18_all_digits_9_or_0_l2225_222568

theorem smallest_multiple_of_18_all_digits_9_or_0 :
  ∃ (m : ℕ), (m > 0) ∧ (m % 18 = 0) ∧ (∀ d ∈ (m.digits 10), d = 9 ∨ d = 0) ∧ (m / 18 = 5) :=
sorry

end NUMINAMATH_GPT_smallest_multiple_of_18_all_digits_9_or_0_l2225_222568


namespace NUMINAMATH_GPT_compute_x_squared_y_plus_x_y_squared_l2225_222528

open Real

theorem compute_x_squared_y_plus_x_y_squared (x y : ℝ) 
  (h1 : (1/x) + (1/y) = 5) 
  (h2 : x * y + 2 * x + 2 * y = 7) : 
  x^2 * y + x * y^2 = 245 / 121 := 
by 
  sorry

end NUMINAMATH_GPT_compute_x_squared_y_plus_x_y_squared_l2225_222528


namespace NUMINAMATH_GPT_absolute_value_inequality_l2225_222518

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 4) ↔ (-1 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 7) := 
by sorry

end NUMINAMATH_GPT_absolute_value_inequality_l2225_222518


namespace NUMINAMATH_GPT_part1_part2_l2225_222574

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 / (3 ^ x + 1) + a

theorem part1 (h : ∀ x : ℝ, f (-x) a = -f x a) : a = -1 :=
by sorry

noncomputable def f' (x : ℝ) : ℝ := 2 / (3 ^ x + 1) - 1

theorem part2 : ∀ t : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ f' x + 1 = t ↔ 1 / 2 ≤ t ∧ t ≤ 1 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l2225_222574


namespace NUMINAMATH_GPT_data_transmission_time_l2225_222517

def chunks_per_block : ℕ := 1024
def blocks : ℕ := 30
def transmission_rate : ℕ := 256
def seconds_in_minute : ℕ := 60

theorem data_transmission_time :
  (blocks * chunks_per_block) / transmission_rate / seconds_in_minute = 2 :=
by
  sorry

end NUMINAMATH_GPT_data_transmission_time_l2225_222517


namespace NUMINAMATH_GPT_initial_principal_amount_l2225_222566

theorem initial_principal_amount
  (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) (P : ℝ)
  (hA : A = 8400) 
  (hr : r = 0.05)
  (hn : n = 1) 
  (ht : t = 1) 
  (hformula : A = P * (1 + r / n) ^ (n * t)) : 
  P = 8000 :=
by
  rw [hA, hr, hn, ht] at hformula
  sorry

end NUMINAMATH_GPT_initial_principal_amount_l2225_222566


namespace NUMINAMATH_GPT_cube_side_length_l2225_222534

theorem cube_side_length (s : ℝ) (h : 6 * s^2 = 864) : s = 12 := by
  sorry

end NUMINAMATH_GPT_cube_side_length_l2225_222534


namespace NUMINAMATH_GPT_solved_distance_l2225_222598

variable (D : ℝ) 

-- Time for A to cover the distance
variable (tA : ℝ) (tB : ℝ)
variable (dA : ℝ) (dB : ℝ := D - 26)

-- A covers the distance in 36 seconds
axiom hA : tA = 36

-- B covers the distance in 45 seconds
axiom hB : tB = 45

-- A beats B by 26 meters implies B covers (D - 26) in the time A covers D
axiom h_diff : dB = dA - 26

theorem solved_distance :
  D = 130 := 
by 
  sorry

end NUMINAMATH_GPT_solved_distance_l2225_222598


namespace NUMINAMATH_GPT_day_after_75_days_l2225_222509

theorem day_after_75_days (day_of_week : ℕ → String) (h : day_of_week 0 = "Tuesday") :
  day_of_week 75 = "Sunday" :=
sorry

end NUMINAMATH_GPT_day_after_75_days_l2225_222509


namespace NUMINAMATH_GPT_delores_initial_money_l2225_222567

theorem delores_initial_money (cost_computer : ℕ) (cost_printer : ℕ) (money_left : ℕ) (initial_money : ℕ) :
  cost_computer = 400 → cost_printer = 40 → money_left = 10 → initial_money = cost_computer + cost_printer + money_left → initial_money = 450 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_delores_initial_money_l2225_222567


namespace NUMINAMATH_GPT_david_biology_marks_l2225_222541

theorem david_biology_marks
  (english math physics chemistry avg_marks num_subjects : ℕ)
  (h_english : english = 86)
  (h_math : math = 85)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 87)
  (h_avg_marks : avg_marks = 85)
  (h_num_subjects : num_subjects = 5) :
  ∃ (biology : ℕ), biology = 85 :=
by
  -- Total marks for all subjects
  let total_marks_for_all_subjects := avg_marks * num_subjects
  -- Total marks in English, Mathematics, Physics, and Chemistry
  let total_marks_in_other_subjects := english + math + physics + chemistry
  -- Marks in Biology
  let biology := total_marks_for_all_subjects - total_marks_in_other_subjects
  existsi biology
  sorry

end NUMINAMATH_GPT_david_biology_marks_l2225_222541


namespace NUMINAMATH_GPT_commuting_days_l2225_222545

theorem commuting_days 
  (a b c d x : ℕ)
  (cond1 : b + c = 12)
  (cond2 : a + c = 20)
  (cond3 : a + b + 2 * d = 14)
  (cond4 : d = 2) :
  a + b + c + d = 23 := sorry

end NUMINAMATH_GPT_commuting_days_l2225_222545


namespace NUMINAMATH_GPT_andrea_needs_1500_sod_squares_l2225_222593

-- Define the measurements of the yard sections
def section1_length : ℕ := 30
def section1_width : ℕ := 40
def section2_length : ℕ := 60
def section2_width : ℕ := 80

-- Define the measurements of the sod square
def sod_length : ℕ := 2
def sod_width : ℕ := 2

-- Compute the areas
def area_section1 : ℕ := section1_length * section1_width
def area_section2 : ℕ := section2_length * section2_width
def total_area : ℕ := area_section1 + area_section2

-- Compute the area of one sod square
def area_sod : ℕ := sod_length * sod_width

-- Compute the number of sod squares needed
def num_sod_squares : ℕ := total_area / area_sod

-- Theorem and proof placeholder
theorem andrea_needs_1500_sod_squares : num_sod_squares = 1500 :=
by {
  -- Place proof here
  sorry
}

end NUMINAMATH_GPT_andrea_needs_1500_sod_squares_l2225_222593


namespace NUMINAMATH_GPT_total_volume_structure_l2225_222527

theorem total_volume_structure (d : ℝ) (h_cone : ℝ) (h_cylinder : ℝ) 
  (r := d / 2) 
  (V_cone := (1 / 3) * π * r^2 * h_cone) 
  (V_cylinder := π * r^2 * h_cylinder) 
  (V_total := V_cone + V_cylinder) :
  d = 8 → h_cone = 9 → h_cylinder = 4 → V_total = 112 * π :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_volume_structure_l2225_222527


namespace NUMINAMATH_GPT_stratified_sample_size_is_correct_l2225_222553

def workshop_A_produces : ℕ := 120
def workshop_B_produces : ℕ := 90
def workshop_C_produces : ℕ := 60
def sample_from_C : ℕ := 4

def total_products : ℕ := workshop_A_produces + workshop_B_produces + workshop_C_produces

noncomputable def sampling_ratio : ℚ := (sample_from_C:ℚ) / (workshop_C_produces:ℚ)

noncomputable def sample_size : ℚ := total_products * sampling_ratio

theorem stratified_sample_size_is_correct :
  sample_size = 18 := by
  sorry

end NUMINAMATH_GPT_stratified_sample_size_is_correct_l2225_222553


namespace NUMINAMATH_GPT_bc_money_l2225_222532

variables (A B C : ℕ)

theorem bc_money (h1 : A + B + C = 400) (h2 : A + C = 300) (h3 : C = 50) : B + C = 150 :=
sorry

end NUMINAMATH_GPT_bc_money_l2225_222532


namespace NUMINAMATH_GPT_option_d_is_correct_l2225_222543

theorem option_d_is_correct : (-2 : ℤ) ^ 3 = -8 := by
  sorry

end NUMINAMATH_GPT_option_d_is_correct_l2225_222543


namespace NUMINAMATH_GPT_find_range_of_x_l2225_222519

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 2 ^ x else 2 ^ (-x)

theorem find_range_of_x (x : ℝ) : 
  f (1 - 2 * x) < f 3 ↔ (-1 < x ∧ x < 2) := 
sorry

end NUMINAMATH_GPT_find_range_of_x_l2225_222519


namespace NUMINAMATH_GPT_work_done_by_force_l2225_222513

def F (x : ℝ) := 4 * x - 1

theorem work_done_by_force :
  let a := 1
  let b := 3
  (∫ x in a..b, F x) = 14 := by
  sorry

end NUMINAMATH_GPT_work_done_by_force_l2225_222513


namespace NUMINAMATH_GPT_tan_diff_identity_l2225_222572

theorem tan_diff_identity 
  (α : ℝ)
  (h : Real.tan α = -4/3) : Real.tan (α - Real.pi / 4) = 7 := 
sorry

end NUMINAMATH_GPT_tan_diff_identity_l2225_222572


namespace NUMINAMATH_GPT_wendy_time_correct_l2225_222561

variable (bonnie_time wendy_difference : ℝ)

theorem wendy_time_correct (h1 : bonnie_time = 7.80) (h2 : wendy_difference = 0.25) : 
  (bonnie_time - wendy_difference = 7.55) :=
by
  sorry

end NUMINAMATH_GPT_wendy_time_correct_l2225_222561


namespace NUMINAMATH_GPT_degree_difference_l2225_222582

variable (S J : ℕ)

theorem degree_difference :
  S = 150 → S + J = 295 → S - J = 5 :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_degree_difference_l2225_222582


namespace NUMINAMATH_GPT_arc_length_of_polar_curve_l2225_222529

noncomputable def arc_length (f : ℝ → ℝ) (df : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt ((f x)^2 + (df x)^2)

theorem arc_length_of_polar_curve :
  arc_length (λ φ => 3 * (1 + Real.sin φ)) (λ φ => 3 * Real.cos φ) (-Real.pi / 6) 0 = 
  6 * (Real.sqrt 3 - Real.sqrt 2) :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_arc_length_of_polar_curve_l2225_222529


namespace NUMINAMATH_GPT_pedal_triangle_angle_pedal_triangle_angle_equality_l2225_222580

variables {A B C T_A T_B T_C: Type*}
variables {α β γ : Real}
variables {triangle : ∀ (A B C : Type*) (α β γ : Real), α ≤ β ∧ β ≤ γ ∧ γ < 90}

theorem pedal_triangle_angle
  (h : α ≤ β ∧ β ≤ γ ∧ γ < 90)
  (angles : 180 - 2 * α ≥ γ) :
  true :=
sorry

theorem pedal_triangle_angle_equality
  (h : α = β)
  (angles : (45 < α ∧ α = β ∧ α ≤ 60) ∧ (60 ≤ γ ∧ γ < 90)) :
  true :=
sorry

end NUMINAMATH_GPT_pedal_triangle_angle_pedal_triangle_angle_equality_l2225_222580


namespace NUMINAMATH_GPT_ratio_of_length_to_width_l2225_222576

variable (L W : ℕ)
variable (H1 : W = 50)
variable (H2 : 2 * L + 2 * W = 240)

theorem ratio_of_length_to_width : L / W = 7 / 5 := 
by sorry

end NUMINAMATH_GPT_ratio_of_length_to_width_l2225_222576


namespace NUMINAMATH_GPT_calculation_not_minus_one_l2225_222558

theorem calculation_not_minus_one :
  (-1 : ℤ) * 1 ≠ 1 ∧
  (-1 : ℤ) / (-1) = 1 ∧
  (-2015 : ℤ) / 2015 ≠ 1 ∧
  (-1 : ℤ)^9 * (-1 : ℤ)^2 ≠ 1 := by 
  sorry

end NUMINAMATH_GPT_calculation_not_minus_one_l2225_222558


namespace NUMINAMATH_GPT_sector_area_l2225_222531

theorem sector_area (s θ r : ℝ) (hs : s = 4) (hθ : θ = 2) (hr : r = s / θ) : (1/2) * r^2 * θ = 4 := by
  sorry

end NUMINAMATH_GPT_sector_area_l2225_222531


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l2225_222588

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.1 * Real.log 2)
noncomputable def c : ℝ := Real.exp (1.3 * Real.log 0.2)

theorem relationship_among_a_b_c : a < c ∧ c < b :=
by
  have a_neg : a < 0 :=
    by sorry
  have b_pos : b > 1 :=
    by sorry
  have c_pos : c < 1 :=
    by sorry
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l2225_222588


namespace NUMINAMATH_GPT_canFormTriangle_cannotFormIsoscelesTriangle_l2225_222599

section TriangleSticks

noncomputable def stickLengths : List ℝ := 
  List.range 10 |>.map (λ n => 1.9 ^ n)

def satisfiesTriangleInequality (a b c : ℝ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem canFormTriangle : ∃ (a b c : ℝ), a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧ satisfiesTriangleInequality a b c :=
sorry

theorem cannotFormIsoscelesTriangle : ¬∃ (a b c : ℝ), a = b ∧ a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧ satisfiesTriangleInequality a b c :=
sorry

end TriangleSticks

end NUMINAMATH_GPT_canFormTriangle_cannotFormIsoscelesTriangle_l2225_222599


namespace NUMINAMATH_GPT_find_x_l2225_222503

variable (N x : ℕ)
variable (h1 : N = 500 * x + 20)
variable (h2 : 4 * 500 + 20 = 2020)

theorem find_x : x = 4 := by
  -- The proof code will go here
  sorry

end NUMINAMATH_GPT_find_x_l2225_222503


namespace NUMINAMATH_GPT_smallest_k_l2225_222501

theorem smallest_k (M : Finset ℕ) (H : ∀ (a b c d : ℕ), a ∈ M → b ∈ M → c ∈ M → d ∈ M → a ≠ b → b ≠ c → c ≠ d → d ≠ a → 20 ∣ (a - b + c - d)) :
  ∃ k, k = 7 ∧ ∀ (M' : Finset ℕ), M'.card = k → ∀ (a b c d : ℕ), a ∈ M' → b ∈ M' → c ∈ M' → d ∈ M' → a ≠ b → b ≠ c → c ≠ d → d ≠ a → 20 ∣ (a - b + c - d) :=
sorry

end NUMINAMATH_GPT_smallest_k_l2225_222501


namespace NUMINAMATH_GPT_parabola_focus_distance_l2225_222502

theorem parabola_focus_distance (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1) (h_dist_y_axis : |P.1| = 4) : 
  dist P (4, 0) = 5 :=
sorry

end NUMINAMATH_GPT_parabola_focus_distance_l2225_222502


namespace NUMINAMATH_GPT_range_of_k_l2225_222522

theorem range_of_k (k : Real) : 
  (∀ (x y : Real), x^2 + y^2 - 12 * x - 4 * y + 37 = 0)
  → ((k < -Real.sqrt 2) ∨ (k > Real.sqrt 2)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l2225_222522
