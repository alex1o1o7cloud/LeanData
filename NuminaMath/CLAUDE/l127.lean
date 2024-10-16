import Mathlib

namespace NUMINAMATH_CALUDE_cone_height_equal_cylinder_l127_12793

/-- Given a cylinder M with base radius 2 and height 6, and a cone N with base diameter
    equal to its slant height, if their volumes are equal, then the height of cone N is 6. -/
theorem cone_height_equal_cylinder (r : ℝ) :
  let cylinder_volume := π * 2^2 * 6
  let cone_base_radius := r
  let cone_height := Real.sqrt 3 * r
  let cone_volume := (1/3) * π * cone_base_radius^2 * cone_height
  cylinder_volume = cone_volume →
  cone_height = 6 := by
sorry

end NUMINAMATH_CALUDE_cone_height_equal_cylinder_l127_12793


namespace NUMINAMATH_CALUDE_system_solution_l127_12779

theorem system_solution (x y u v : ℝ) : 
  (x = -2 ∧ y = 2 ∧ u = 2 ∧ v = -2) →
  (x + 7*y + 3*v + 5*u = 16) ∧
  (8*x + 4*y + 6*v + 2*u = -16) ∧
  (2*x + 6*y + 4*v + 8*u = 16) ∧
  (5*x + 3*y + 7*v + u = -16) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l127_12779


namespace NUMINAMATH_CALUDE_arctan_equation_solutions_l127_12712

theorem arctan_equation_solutions (x : ℝ) : 
  (Real.arctan (2 / x) + Real.arctan (1 / x^2) = π / 4) ↔ 
  (x = 3 ∨ x = (-3 + Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_arctan_equation_solutions_l127_12712


namespace NUMINAMATH_CALUDE_store_discount_percentage_l127_12700

/-- Proves that the discount percentage is 9% given the specified markups and profit -/
theorem store_discount_percentage (C : ℝ) (h : C > 0) : 
  let initial_price := 1.20 * C
  let marked_up_price := 1.25 * initial_price
  let final_profit := 0.365 * C
  ∃ (D : ℝ), 
    marked_up_price * (1 - D) - C = final_profit ∧ 
    D = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_store_discount_percentage_l127_12700


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l127_12768

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 - 3 * Complex.I) : 
  z.im = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l127_12768


namespace NUMINAMATH_CALUDE_odd_function_with_period_two_negation_at_six_l127_12702

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_two_negation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

theorem odd_function_with_period_two_negation_at_six
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : has_period_two_negation f) :
  f 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_odd_function_with_period_two_negation_at_six_l127_12702


namespace NUMINAMATH_CALUDE_sequence_sum_property_l127_12733

/-- Given a positive sequence {a_n}, prove that a_n = 2n - 1 for all positive integers n,
    where S_n = (a_n + 1)^2 / 4 is the sum of the first n terms. -/
theorem sequence_sum_property (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, S n = (a n + 1)^2 / 4) →
  ∀ n, a n = 2 * n - 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l127_12733


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_under_cube_root_2000_l127_12789

theorem greatest_multiple_of_four_under_cube_root_2000 :
  ∀ x : ℕ, 
    x > 0 → 
    x % 4 = 0 → 
    x^3 < 2000 → 
    x ≤ 12 ∧ 
    ∃ y : ℕ, y > 0 ∧ y % 4 = 0 ∧ y^3 < 2000 ∧ y = 12 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_under_cube_root_2000_l127_12789


namespace NUMINAMATH_CALUDE_larger_integer_value_l127_12753

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) : 
  max a b = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l127_12753


namespace NUMINAMATH_CALUDE_only_negative_three_less_than_negative_two_l127_12761

theorem only_negative_three_less_than_negative_two :
  ((-3 : ℝ) < -2) ∧
  ((-1 : ℝ) > -2) ∧
  ((-Real.sqrt 2 : ℝ) > -2) ∧
  ((-Real.pi / 2 : ℝ) > -2) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_three_less_than_negative_two_l127_12761


namespace NUMINAMATH_CALUDE_clock_rings_seven_times_l127_12798

/-- Calculates the number of rings for a clock with given interval and day length -/
def number_of_rings (interval : ℕ) (day_length : ℕ) : ℕ :=
  (day_length / interval) + 1

/-- Theorem: A clock ringing every 4 hours in a 24-hour day rings 7 times -/
theorem clock_rings_seven_times : number_of_rings 4 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_clock_rings_seven_times_l127_12798


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l127_12782

/-- Given an arithmetic sequence {a_n}, if a_3 + a_4 + a_5 = 12, 
    then a_1 + a_2 + ... + a_7 = 28 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) →
  (a 3 + a 4 + a 5 = 12) →
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l127_12782


namespace NUMINAMATH_CALUDE_box_maker_is_bellini_l127_12719

-- Define the possible makers of the box
inductive Maker
  | Bellini
  | Cellini
  | BelliniSon
  | CelliniSon

-- Define the inscription on the box
def inscription (maker : Maker) : Prop :=
  maker ≠ Maker.BelliniSon

-- Define the condition that the box was made by Bellini, Cellini, or one of their sons
def possibleMakers (maker : Maker) : Prop :=
  maker = Maker.Bellini ∨ maker = Maker.Cellini ∨ maker = Maker.BelliniSon ∨ maker = Maker.CelliniSon

-- Theorem: The maker of the box is Bellini
theorem box_maker_is_bellini :
  ∃ (maker : Maker), possibleMakers maker ∧ inscription maker → maker = Maker.Bellini :=
sorry

end NUMINAMATH_CALUDE_box_maker_is_bellini_l127_12719


namespace NUMINAMATH_CALUDE_max_distance_to_origin_l127_12759

open Complex

theorem max_distance_to_origin (z : ℂ) (h_norm : abs z = 1) : 
  let w := 2*z - Complex.I*z
  ∀ ε > 0, abs w ≤ 3 + ε :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_origin_l127_12759


namespace NUMINAMATH_CALUDE_roots_expression_l127_12715

theorem roots_expression (a b : ℝ) (α β γ δ : ℝ) 
  (hα : α^2 - a*α - 1 = 0)
  (hβ : β^2 - a*β - 1 = 0)
  (hγ : γ^2 - b*γ - 1 = 0)
  (hδ : δ^2 - b*δ - 1 = 0) :
  (α - γ)^2 * (β - γ)^2 * (α + δ)^2 * (β + δ)^2 = (b^2 - a^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_expression_l127_12715


namespace NUMINAMATH_CALUDE_complement_union_theorem_l127_12711

/-- The universal set I -/
def I : Set ℕ := {0, 1, 2, 3, 4}

/-- Set A -/
def A : Set ℕ := {0, 1, 2, 3}

/-- Set B -/
def B : Set ℕ := {2, 3, 4}

/-- The main theorem -/
theorem complement_union_theorem :
  (I \ A) ∪ (I \ B) = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l127_12711


namespace NUMINAMATH_CALUDE_sum_first_four_terms_l127_12743

def a (n : ℕ) : ℤ := (-1)^n * (3*n - 2)

theorem sum_first_four_terms : 
  (a 1) + (a 2) + (a 3) + (a 4) = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_first_four_terms_l127_12743


namespace NUMINAMATH_CALUDE_isabellas_original_hair_length_l127_12718

/-- The length of Isabella's hair before the haircut -/
def original_length : ℝ := sorry

/-- The length of Isabella's hair after the haircut -/
def after_haircut_length : ℝ := 9

/-- The length of hair that was cut off -/
def cut_length : ℝ := 9

/-- Theorem stating that Isabella's original hair length was 18 inches -/
theorem isabellas_original_hair_length :
  original_length = after_haircut_length + cut_length ∧ original_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_original_hair_length_l127_12718


namespace NUMINAMATH_CALUDE_binomial_9_8_l127_12788

theorem binomial_9_8 : Nat.choose 9 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_8_l127_12788


namespace NUMINAMATH_CALUDE_sqrt_expressions_l127_12754

-- Define the theorem
theorem sqrt_expressions :
  -- Part 1
  (∀ (a b m n : ℤ), a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 → 
    a = m^2 + 3*n^2 ∧ b = 2*m*n) ∧
  -- Part 2
  (∀ (a m n : ℕ+), a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 → 
    a = 13 ∨ a = 7) ∧
  -- Part 3
  Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 := by
sorry


end NUMINAMATH_CALUDE_sqrt_expressions_l127_12754


namespace NUMINAMATH_CALUDE_calculate_expression_l127_12722

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 3 * x

-- Define the # operation
def hash_op (x y : ℤ) : ℤ := x * y + y

-- Theorem statement
theorem calculate_expression : (at_op 8 5) - (at_op 5 8) + (hash_op 8 5) = 36 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l127_12722


namespace NUMINAMATH_CALUDE_alice_minimum_speed_l127_12760

-- Define the problem parameters
def distance : ℝ := 180
def bob_speed : ℝ := 40
def alice_delay : ℝ := 0.5

-- Define the theorem
theorem alice_minimum_speed :
  ∀ (alice_speed : ℝ),
  alice_speed > distance / (distance / bob_speed - alice_delay) →
  alice_speed * (distance / bob_speed - alice_delay) > distance :=
by sorry

end NUMINAMATH_CALUDE_alice_minimum_speed_l127_12760


namespace NUMINAMATH_CALUDE_spring_sports_event_probabilities_l127_12797

def male_volunteers : ℕ := 4
def female_volunteers : ℕ := 3
def team_size : ℕ := 3

def total_volunteers : ℕ := male_volunteers + female_volunteers

theorem spring_sports_event_probabilities :
  let p_at_least_one_female := 1 - (Nat.choose male_volunteers team_size : ℚ) / (Nat.choose total_volunteers team_size : ℚ)
  let p_all_male_given_at_least_one_male := 
    (Nat.choose male_volunteers team_size : ℚ) / 
    ((Nat.choose total_volunteers team_size : ℚ) - (Nat.choose female_volunteers team_size : ℚ))
  p_at_least_one_female = 31 / 35 ∧ 
  p_all_male_given_at_least_one_male = 2 / 17 := by
  sorry

end NUMINAMATH_CALUDE_spring_sports_event_probabilities_l127_12797


namespace NUMINAMATH_CALUDE_deck_width_l127_12756

/-- Given a rectangular deck with the following properties:
  * length is 30 feet
  * total cost per square foot (including construction and sealant) is $4
  * total payment is $4800
  prove that the width of the deck is 40 feet -/
theorem deck_width (length : ℝ) (cost_per_sqft : ℝ) (total_cost : ℝ) :
  length = 30 →
  cost_per_sqft = 4 →
  total_cost = 4800 →
  (length * (total_cost / cost_per_sqft)) / length = 40 := by
  sorry

end NUMINAMATH_CALUDE_deck_width_l127_12756


namespace NUMINAMATH_CALUDE_complex_square_equality_l127_12735

theorem complex_square_equality : (((3 : ℂ) - I) / ((1 : ℂ) + I))^2 = -3 - 4*I := by sorry

end NUMINAMATH_CALUDE_complex_square_equality_l127_12735


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l127_12709

/-- A shape formed by unit cubes in a straight line -/
structure LineShape where
  num_cubes : ℕ

/-- Volume of a LineShape -/
def volume (shape : LineShape) : ℕ :=
  shape.num_cubes

/-- Surface area of a LineShape -/
def surface_area (shape : LineShape) : ℕ :=
  2 * 5 + (shape.num_cubes - 2) * 4

/-- Theorem stating the ratio of volume to surface area for a LineShape with 8 cubes -/
theorem volume_to_surface_area_ratio (shape : LineShape) (h : shape.num_cubes = 8) :
  (volume shape : ℚ) / (surface_area shape : ℚ) = 4 / 17 := by
  sorry


end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l127_12709


namespace NUMINAMATH_CALUDE_domain_of_composition_l127_12749

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := { x | x ≥ 4 }

-- State the theorem
theorem domain_of_composition :
  { x : ℝ | ∃ y ∈ dom_f, x = y^2 } = { x : ℝ | x ≥ 16 } := by sorry

end NUMINAMATH_CALUDE_domain_of_composition_l127_12749


namespace NUMINAMATH_CALUDE_fraction_inequality_l127_12740

theorem fraction_inequality (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) 
  (h5 : e < 0) : 
  e / (a - c) > e / (b - d) := by
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l127_12740


namespace NUMINAMATH_CALUDE_binomial_9_choose_3_l127_12771

theorem binomial_9_choose_3 : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_choose_3_l127_12771


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l127_12766

/-- Given a quadratic expression x^2 - 24x + 50 that can be rewritten as (x+d)^2 + e,
    this theorem states that d + e = -106. -/
theorem quadratic_rewrite_sum (d e : ℝ) : 
  (∀ x, x^2 - 24*x + 50 = (x + d)^2 + e) → d + e = -106 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l127_12766


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_6_l127_12790

/-- The area of a circle with diameter 6 meters is 9π square meters. -/
theorem circle_area_with_diameter_6 :
  let diameter : ℝ := 6
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius^2
  area = 9 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_6_l127_12790


namespace NUMINAMATH_CALUDE_ace_king_queen_probability_l127_12727

def standard_deck : ℕ := 52
def num_aces : ℕ := 4
def num_kings : ℕ := 4
def num_queens : ℕ := 4

theorem ace_king_queen_probability :
  (num_aces / standard_deck) * (num_kings / (standard_deck - 1)) * (num_queens / (standard_deck - 2)) = 16 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_ace_king_queen_probability_l127_12727


namespace NUMINAMATH_CALUDE_f_properties_l127_12701

def f (x m : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + m

theorem f_properties (m : ℝ) :
  (∀ x y, x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 3 ∧ y > 3)) → f x m > f y m) ∧
  (∃ x₀ ∈ Set.Icc (-2) 2, ∀ x ∈ Set.Icc (-2) 2, f x m ≤ f x₀ m) ∧
  f x₀ m = 20 →
  ∃ x₁ ∈ Set.Icc (-2) 2, ∀ x ∈ Set.Icc (-2) 2, f x m ≥ f x₁ m ∧ f x₁ m = -7 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l127_12701


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l127_12786

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

-- State the theorem
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l127_12786


namespace NUMINAMATH_CALUDE_inequality_range_l127_12739

theorem inequality_range (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, x^2 + a*x > 4*x + a - 3 ↔ x < -1 ∨ x > 3 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l127_12739


namespace NUMINAMATH_CALUDE_min_xy_and_x_plus_y_l127_12724

/-- Given positive real numbers x and y satisfying x + 8y - xy = 0,
    proves that the minimum value of xy is 32 and
    the minimum value of x + y is 9 + 4√2 -/
theorem min_xy_and_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8*y - x*y = 0) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 8*y' - x'*y' = 0 → x*y ≤ x'*y') ∧
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 8*y' - x'*y' = 0 → x + y ≤ x' + y') ∧
  x*y = 32 ∧ x + y = 9 + 4*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_and_x_plus_y_l127_12724


namespace NUMINAMATH_CALUDE_square_and_add_l127_12707

theorem square_and_add (x : ℝ) (h : x = 5) : 2 * x^2 + 5 = 55 := by
  sorry

end NUMINAMATH_CALUDE_square_and_add_l127_12707


namespace NUMINAMATH_CALUDE_average_boxes_theorem_l127_12742

def boxes_day1 : ℕ := 318
def boxes_day2 : ℕ := 312
def boxes_day3_part1 : ℕ := 180
def boxes_day3_part2 : ℕ := 162
def total_days : ℕ := 3

def average_boxes_per_day : ℚ :=
  (boxes_day1 + boxes_day2 + boxes_day3_part1 + boxes_day3_part2) / total_days

theorem average_boxes_theorem : average_boxes_per_day = 324 := by
  sorry

end NUMINAMATH_CALUDE_average_boxes_theorem_l127_12742


namespace NUMINAMATH_CALUDE_certain_event_good_product_l127_12781

theorem certain_event_good_product (total : Nat) (good : Nat) (defective : Nat) (draw : Nat) :
  total = good + defective →
  good = 10 →
  defective = 2 →
  draw = 3 →
  Fintype.card {s : Finset (Fin total) // s.card = draw ∧ (∃ i ∈ s, i.val < good)} / Fintype.card {s : Finset (Fin total) // s.card = draw} = 1 :=
sorry

end NUMINAMATH_CALUDE_certain_event_good_product_l127_12781


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l127_12794

theorem sqrt_equation_solution :
  ∀ x : ℚ, (x > 2) → (Real.sqrt (7 * x) / Real.sqrt (2 * (x - 2)) = 3) → x = 36 / 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l127_12794


namespace NUMINAMATH_CALUDE_water_tank_capacity_l127_12783

theorem water_tank_capacity (initial_fraction : ℚ) (added_gallons : ℕ) (total_capacity : ℕ) : 
  initial_fraction = 1/3 →
  added_gallons = 16 →
  initial_fraction * total_capacity + added_gallons = total_capacity →
  total_capacity = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l127_12783


namespace NUMINAMATH_CALUDE_sum_of_digits_of_n_is_nine_l127_12744

/-- Two distinct digits -/
def distinct_digits (d e : Nat) : Prop :=
  d ≠ e ∧ d < 10 ∧ e < 10

/-- Sum of digits is prime -/
def sum_is_prime (d e : Nat) : Prop :=
  Nat.Prime (d + e)

/-- k is prime and greater than both d and e -/
def k_is_valid_prime (d e k : Nat) : Prop :=
  Nat.Prime k ∧ k > d ∧ k > e

/-- n is the product of d, e, and k -/
def n_is_product (n d e k : Nat) : Prop :=
  n = d * e * k

/-- k is related to d and e -/
def k_relation (d e k : Nat) : Prop :=
  k = 10 * d + e

/-- n is the largest such product -/
def n_is_largest (n : Nat) : Prop :=
  ∀ m d e k, distinct_digits d e → sum_is_prime d e → k_is_valid_prime d e k →
    k_relation d e k → n_is_product m d e k → m ≤ n

/-- n is the smallest multiple of k -/
def n_is_smallest_multiple (n k : Nat) : Prop :=
  k ∣ n ∧ ∀ m, m < n → ¬(k ∣ m)

/-- Sum of digits of a number -/
def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_n_is_nine :
  ∃ n d e k, distinct_digits d e ∧ sum_is_prime d e ∧ k_is_valid_prime d e k ∧
    k_relation d e k ∧ n_is_product n d e k ∧ n_is_largest n ∧
    n_is_smallest_multiple n k ∧ sum_of_digits n = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_n_is_nine_l127_12744


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l127_12784

theorem solution_to_linear_equation (x y m : ℝ) : 
  x = 1 → y = 3 → x - 2 * y = m → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l127_12784


namespace NUMINAMATH_CALUDE_not_subset_iff_exists_not_mem_l127_12708

theorem not_subset_iff_exists_not_mem {M P : Set α} (hM : M.Nonempty) :
  ¬(M ⊆ P) ↔ ∃ x ∈ M, x ∉ P := by
  sorry

end NUMINAMATH_CALUDE_not_subset_iff_exists_not_mem_l127_12708


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l127_12772

theorem line_intercepts_sum (c : ℝ) : 
  (∃ x y : ℝ, 3 * x + 5 * y + c = 0 ∧ x + y = 30) → c = -56.25 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l127_12772


namespace NUMINAMATH_CALUDE_grade_ratio_is_two_to_one_l127_12710

/-- The ratio of students in the third grade to the second grade -/
def grade_ratio (boys_2nd : ℕ) (girls_2nd : ℕ) (total_students : ℕ) : ℚ :=
  let students_2nd := boys_2nd + girls_2nd
  let students_3rd := total_students - students_2nd
  students_3rd / students_2nd

/-- Theorem stating the ratio of students in the third grade to the second grade -/
theorem grade_ratio_is_two_to_one :
  grade_ratio 20 11 93 = 2 := by
  sorry

#eval grade_ratio 20 11 93

end NUMINAMATH_CALUDE_grade_ratio_is_two_to_one_l127_12710


namespace NUMINAMATH_CALUDE_average_movie_length_l127_12729

def miles_run : ℕ := 15
def minutes_per_mile : ℕ := 12
def number_of_movies : ℕ := 2

theorem average_movie_length :
  (miles_run * minutes_per_mile) / number_of_movies = 90 :=
by sorry

end NUMINAMATH_CALUDE_average_movie_length_l127_12729


namespace NUMINAMATH_CALUDE_degree_not_determined_by_A_P_l127_12736

/-- A characteristic associated with a polynomial -/
def A_P (P : Polynomial ℝ) : Set ℝ := sorry

/-- Theorem stating that the degree of a polynomial cannot be uniquely determined from A_P -/
theorem degree_not_determined_by_A_P :
  ∃ (P1 P2 : Polynomial ℝ), A_P P1 = A_P P2 ∧ P1.degree ≠ P2.degree := by
  sorry

end NUMINAMATH_CALUDE_degree_not_determined_by_A_P_l127_12736


namespace NUMINAMATH_CALUDE_sues_mother_cookies_l127_12769

/-- The number of cookies Sue's mother made -/
def total_cookies (bags : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  bags * cookies_per_bag

/-- Proof that Sue's mother made 75 cookies -/
theorem sues_mother_cookies : total_cookies 25 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_sues_mother_cookies_l127_12769


namespace NUMINAMATH_CALUDE_stock_profit_is_447_5_l127_12730

/-- Calculate the profit from a stock transaction with given parameters -/
def calculate_profit (num_shares : ℕ) (buy_price sell_price : ℚ) 
  (stamp_duty_rate transfer_fee_rate commission_rate : ℚ) 
  (min_commission : ℚ) : ℚ :=
  let total_cost := num_shares * buy_price
  let total_income := num_shares * sell_price
  let total_transaction := total_cost + total_income
  let stamp_duty := total_transaction * stamp_duty_rate
  let transfer_fee := total_transaction * transfer_fee_rate
  let commission := max (total_transaction * commission_rate) min_commission
  total_income - total_cost - stamp_duty - transfer_fee - commission

/-- The profit from the given stock transaction is 447.5 yuan -/
theorem stock_profit_is_447_5 : 
  calculate_profit 1000 5 (11/2) (1/1000) (1/1000) (3/1000) 5 = 447.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_profit_is_447_5_l127_12730


namespace NUMINAMATH_CALUDE_bacon_to_eggs_ratio_l127_12750

/-- Represents a breakfast plate with eggs and bacon strips -/
structure BreakfastPlate where
  eggs : ℕ
  bacon : ℕ

/-- Represents the cafe's breakfast order -/
structure CafeOrder where
  plates : ℕ
  totalBacon : ℕ

theorem bacon_to_eggs_ratio (order : CafeOrder) (plate : BreakfastPlate) :
  order.plates = 14 →
  order.totalBacon = 56 →
  plate.eggs = 2 →
  (order.totalBacon / order.plates : ℚ) / plate.eggs = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_bacon_to_eggs_ratio_l127_12750


namespace NUMINAMATH_CALUDE_strawberry_cost_l127_12778

/-- The price of one basket of strawberries in dollars -/
def price_per_basket : ℚ := 16.5

/-- The number of baskets to be purchased -/
def number_of_baskets : ℕ := 4

/-- The total cost of purchasing the strawberries -/
def total_cost : ℚ := price_per_basket * number_of_baskets

/-- Theorem stating that the total cost of 4 baskets of strawberries at $16.50 each is $66.00 -/
theorem strawberry_cost : total_cost = 66 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_cost_l127_12778


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l127_12746

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 24 →
  a^2 + b^2 + c^2 = 2500 →
  a^2 + b^2 = c^2 →
  c = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l127_12746


namespace NUMINAMATH_CALUDE_modified_ohara_triple_solution_l127_12765

/-- Definition of a Modified O'Hara Triple -/
def is_modified_ohara_triple (a b x k : ℕ+) : Prop :=
  k * (a : ℝ).sqrt + (b : ℝ).sqrt = x

/-- Theorem: If (49, 16, x, 2) is a Modified O'Hara Triple, then x = 18 -/
theorem modified_ohara_triple_solution :
  ∀ x : ℕ+, is_modified_ohara_triple 49 16 x 2 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_modified_ohara_triple_solution_l127_12765


namespace NUMINAMATH_CALUDE_equation_solutions_l127_12731

theorem equation_solutions :
  (∃ x : ℚ, 27 * (x + 1)^3 = -64 ∧ x = -7/3) ∧
  (∃ x : ℤ, (x + 1)^2 = 25 ∧ (x = 4 ∨ x = -6)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l127_12731


namespace NUMINAMATH_CALUDE_b_55_mod_56_l127_12770

/-- b_n is the integer obtained by writing all integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- The theorem states that b_55 mod 56 = 0 -/
theorem b_55_mod_56 : b 55 % 56 = 0 := by sorry

end NUMINAMATH_CALUDE_b_55_mod_56_l127_12770


namespace NUMINAMATH_CALUDE_balls_total_weight_l127_12776

/-- Represents the total weight of five colored metal balls -/
def total_weight (blue brown green red yellow : ℝ) : ℝ :=
  blue + brown + green + red + yellow

/-- Theorem stating the total weight of the balls -/
theorem balls_total_weight :
  ∃ (blue brown green red yellow : ℝ),
    blue = 6 ∧
    brown = 3.12 ∧
    green = 4.25 ∧
    red = 2 * green ∧
    yellow = red - 1.5 ∧
    total_weight blue brown green red yellow = 28.87 := by
  sorry

end NUMINAMATH_CALUDE_balls_total_weight_l127_12776


namespace NUMINAMATH_CALUDE_emily_max_servings_l127_12737

/-- Represents the recipe and available ingredients for a fruit smoothie --/
structure SmoothieIngredients where
  recipe_bananas : ℕ
  recipe_strawberries : ℕ
  recipe_yogurt : ℕ
  available_bananas : ℕ
  available_strawberries : ℕ
  available_yogurt : ℕ

/-- Calculates the maximum number of servings that can be made --/
def max_servings (ingredients : SmoothieIngredients) : ℕ :=
  min
    (ingredients.available_bananas * 3 / ingredients.recipe_bananas)
    (min
      (ingredients.available_strawberries * 3 / ingredients.recipe_strawberries)
      (ingredients.available_yogurt * 3 / ingredients.recipe_yogurt))

/-- Theorem stating that Emily can make at most 6 servings --/
theorem emily_max_servings :
  let emily_ingredients : SmoothieIngredients := {
    recipe_bananas := 2,
    recipe_strawberries := 1,
    recipe_yogurt := 2,
    available_bananas := 4,
    available_strawberries := 3,
    available_yogurt := 6
  }
  max_servings emily_ingredients = 6 := by
  sorry

end NUMINAMATH_CALUDE_emily_max_servings_l127_12737


namespace NUMINAMATH_CALUDE_integer_division_implication_l127_12764

theorem integer_division_implication (n : ℕ) (m : ℤ) :
  2^n - 2 = m * n →
  ∃ k : ℤ, (2^(2^n - 1) - 2) / (2^n - 1) = 2 * k :=
sorry

end NUMINAMATH_CALUDE_integer_division_implication_l127_12764


namespace NUMINAMATH_CALUDE_hyperbola_condition_l127_12752

-- Define the equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 3) - y^2 / (k + 3) = 1

-- State the theorem
theorem hyperbola_condition (k : ℝ) :
  (k > 3 → is_hyperbola k) ∧ (∃ k₀ ≤ 3, is_hyperbola k₀) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l127_12752


namespace NUMINAMATH_CALUDE_toaster_cost_is_72_l127_12706

/-- Calculates the total cost of a toaster including insurance, fees, and taxes. -/
def toaster_total_cost (msrp : ℝ) (insurance_rate : ℝ) (premium_upgrade : ℝ) 
  (recycling_fee : ℝ) (tax_rate : ℝ) : ℝ :=
  let insurance_cost := msrp * insurance_rate
  let total_insurance := insurance_cost + premium_upgrade
  let cost_before_tax := msrp + total_insurance + recycling_fee
  let tax := cost_before_tax * tax_rate
  cost_before_tax + tax

/-- Theorem stating that the total cost of the toaster is $72 given the specified conditions. -/
theorem toaster_cost_is_72 : 
  toaster_total_cost 30 0.2 7 5 0.5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_toaster_cost_is_72_l127_12706


namespace NUMINAMATH_CALUDE_walker_cyclist_speed_ratio_l127_12796

/-- Given two people, a walker and a cyclist, prove that the walker is twice as slow as the cyclist
    when the cyclist's speed is three times the walker's speed. -/
theorem walker_cyclist_speed_ratio
  (S : ℝ) -- distance between home and lake
  (x : ℝ) -- walking speed
  (h1 : 0 < x) -- walking speed is positive
  (h2 : 0 < S) -- distance is positive
  (v : ℝ) -- cycling speed
  (h3 : v = 3 * x) -- cyclist speed is 3 times walker speed
  : (S / x) / (S / v) = 2 := by
  sorry

end NUMINAMATH_CALUDE_walker_cyclist_speed_ratio_l127_12796


namespace NUMINAMATH_CALUDE_inequality_holds_l127_12757

/-- The inequality holds for the given pairs of non-negative integers -/
theorem inequality_holds (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∀ k n : ℕ, 
    (1 + y^n / x^k ≥ (1 + y)^n / (1 + x)^k) ↔ 
    ((k = 0 ∧ n ≥ 0) ∨ 
     (k = 1 ∧ n = 0) ∨ 
     (k = 0 ∧ n = 0) ∨ 
     (k ≥ n - 1 ∧ n ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_l127_12757


namespace NUMINAMATH_CALUDE_martin_fruit_ratio_l127_12785

/-- Given that Martin has twice as many oranges as limes now, 50 oranges, and initially had 150 fruits,
    prove that the ratio of fruits eaten to initial fruits is 1/2 -/
theorem martin_fruit_ratio :
  ∀ (oranges_now limes_now fruits_initial : ℕ),
    oranges_now = 50 →
    fruits_initial = 150 →
    oranges_now = 2 * limes_now →
    (fruits_initial - (oranges_now + limes_now)) / fruits_initial = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_martin_fruit_ratio_l127_12785


namespace NUMINAMATH_CALUDE_right_triangle_angle_calculation_l127_12725

theorem right_triangle_angle_calculation (A B C : Real) 
  (h1 : A + B + C = 180) -- Sum of angles in a triangle is 180°
  (h2 : C = 90) -- Angle C is 90°
  (h3 : A = 35.5) -- Angle A is 35.5°
  : B = 54.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_calculation_l127_12725


namespace NUMINAMATH_CALUDE_mean_squared_sum_l127_12704

theorem mean_squared_sum (x y z : ℝ) 
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_mean_squared_sum_l127_12704


namespace NUMINAMATH_CALUDE_determine_back_iff_conditions_met_l127_12717

/-- Represents a card with two sides -/
structure Card where
  side1 : Nat
  side2 : Nat

/-- Checks if a number appears on a card -/
def numberOnCard (c : Card) (n : Nat) : Prop :=
  c.side1 = n ∨ c.side2 = n

/-- Represents the deck of n cards -/
def deck (n : Nat) : List Card :=
  List.range n |>.map (λ i => ⟨i, i + 1⟩)

/-- Represents the cards seen so far -/
def SeenCards := List Nat

/-- Determines if the back of the last card can be identified -/
def canDetermineBack (n : Nat) (k : Nat) (seen : SeenCards) : Prop :=
  (k = 0 ∨ k = n) ∨
  (0 < k ∧ k < n ∧
    (seen.count (k + 1) = 2 ∨
     (∃ j, 1 ≤ j ∧ j ≤ n - k - 1 ∧
       (∀ i, k + 1 ≤ i ∧ i ≤ k + j → seen.count i ≥ 1) ∧
       (if k + j + 1 = n then seen.count n ≥ 1 else seen.count (k + j + 1) = 2)) ∨
     seen.count (k - 1) = 2 ∨
     (∃ j, 1 ≤ j ∧ j ≤ k - 1 ∧
       (∀ i, k - j ≤ i ∧ i ≤ k - 1 → seen.count i ≥ 1) ∧
       (if k - j - 1 = 0 then seen.count 0 ≥ 1 else seen.count (k - j - 1) = 2))))

/-- The main theorem to be proved -/
theorem determine_back_iff_conditions_met (n : Nat) (k : Nat) (seen : SeenCards) :
  canDetermineBack n k seen ↔
  (∀ (lastCard : Card),
    numberOnCard lastCard k →
    lastCard ∈ deck n →
    ∃! backNumber, numberOnCard lastCard backNumber ∧ backNumber ≠ k) := by
  sorry

end NUMINAMATH_CALUDE_determine_back_iff_conditions_met_l127_12717


namespace NUMINAMATH_CALUDE_lg_equation_l127_12721

-- Define the logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem lg_equation : lg 2 * lg 50 + lg 25 - lg 5 * lg 20 = 1 := by sorry

end NUMINAMATH_CALUDE_lg_equation_l127_12721


namespace NUMINAMATH_CALUDE_orchard_sections_count_l127_12716

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := 360

/-- The number of sections in the orchard -/
def num_sections : ℕ := total_sacks / sacks_per_section

theorem orchard_sections_count :
  num_sections = 8 :=
sorry

end NUMINAMATH_CALUDE_orchard_sections_count_l127_12716


namespace NUMINAMATH_CALUDE_magnitude_a_minus_2b_equals_sqrt_17_l127_12726

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

theorem magnitude_a_minus_2b_equals_sqrt_17 :
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_minus_2b_equals_sqrt_17_l127_12726


namespace NUMINAMATH_CALUDE_company_fund_problem_l127_12762

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  initial_fund = 60 * n - 10 →
  initial_fund = 50 * n + 120 →
  initial_fund = 770 :=
by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l127_12762


namespace NUMINAMATH_CALUDE_computer_cost_is_400_l127_12787

/-- The cost of the computer Delores bought -/
def computer_cost : ℕ := sorry

/-- The initial amount of money Delores had -/
def initial_money : ℕ := 450

/-- The combined cost of the computer and printer -/
def total_purchase : ℕ := 40

/-- The amount of money Delores had left after the purchase -/
def money_left : ℕ := 10

/-- Theorem stating that the computer cost $400 -/
theorem computer_cost_is_400 :
  computer_cost = 400 :=
by sorry

end NUMINAMATH_CALUDE_computer_cost_is_400_l127_12787


namespace NUMINAMATH_CALUDE_odd_function_and_inequality_l127_12795

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - 4 / (2 * a^x + a)

theorem odd_function_and_inequality 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x, f a x = -f a (-x)) -- odd function condition
  (h4 : ∀ x, f a x ∈ Set.univ) -- defined on (-∞, +∞)
  : 
  (a = 2) ∧ 
  (∀ t : ℝ, (∀ x ∈ Set.Ioc 0 1, t * f a x ≥ 2^x - 2) ↔ t ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_odd_function_and_inequality_l127_12795


namespace NUMINAMATH_CALUDE_water_molecule_radius_scientific_notation_l127_12713

theorem water_molecule_radius_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.00000000192 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -9 :=
by sorry

end NUMINAMATH_CALUDE_water_molecule_radius_scientific_notation_l127_12713


namespace NUMINAMATH_CALUDE_ones_digit_of_large_power_l127_12734

theorem ones_digit_of_large_power : ∃ n : ℕ, 34^(11^34) ≡ 4 [ZMOD 10] :=
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_large_power_l127_12734


namespace NUMINAMATH_CALUDE_solution_set_equality_l127_12777

theorem solution_set_equality (x : ℝ) : 
  (1 / ((x + 1) * (x - 1)) ≤ 0) ↔ (-1 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l127_12777


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l127_12723

theorem root_sum_reciprocal (p q r : ℂ) : 
  (p^3 - 2*p^2 + p - 1 = 0) → 
  (q^3 - 2*q^2 + q - 1 = 0) → 
  (r^3 - 2*r^2 + r - 1 = 0) → 
  (1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) = 20 / 19) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l127_12723


namespace NUMINAMATH_CALUDE_scissors_count_l127_12705

/-- The total number of scissors after addition -/
def total_scissors (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that the total number of scissors is 52 -/
theorem scissors_count : total_scissors 39 13 = 52 := by
  sorry

end NUMINAMATH_CALUDE_scissors_count_l127_12705


namespace NUMINAMATH_CALUDE_smallest_share_for_200_people_l127_12703

/-- Represents a family with land inheritance rules -/
structure Family :=
  (size : ℕ)
  (has_founder : size > 0)

/-- The smallest possible share of the original plot for any family member -/
def smallest_share (f : Family) : ℚ :=
  1 / (4 * 3^65)

/-- Theorem stating the smallest possible share for a family of 200 people -/
theorem smallest_share_for_200_people (f : Family) (h : f.size = 200) :
  smallest_share f = 1 / (4 * 3^65) := by
  sorry

end NUMINAMATH_CALUDE_smallest_share_for_200_people_l127_12703


namespace NUMINAMATH_CALUDE_complex_root_implies_positive_triangle_l127_12728

theorem complex_root_implies_positive_triangle (a b c α β : ℝ) :
  α > 0 →
  β ≠ 0 →
  Complex.I ^ 2 = -1 →
  (α + β * Complex.I) ^ 2 - (a + b + c) * (α + β * Complex.I) + (a * b + b * c + c * a) = 0 →
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  Real.sqrt a + Real.sqrt b > Real.sqrt c ∧
  Real.sqrt b + Real.sqrt c > Real.sqrt a ∧
  Real.sqrt c + Real.sqrt a > Real.sqrt b :=
by sorry

end NUMINAMATH_CALUDE_complex_root_implies_positive_triangle_l127_12728


namespace NUMINAMATH_CALUDE_exists_perpendicular_line_l127_12763

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define a relation for a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define a relation for perpendicularity between lines
variable (perpendicular : Line → Line → Prop)

-- Theorem statement
theorem exists_perpendicular_line (a : Line) (α : Plane) :
  ∃ l : Line, in_plane l α ∧ perpendicular l a :=
sorry

end NUMINAMATH_CALUDE_exists_perpendicular_line_l127_12763


namespace NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l127_12774

theorem min_values_ab_and_a_plus_2b (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 2) : 
  (∀ a b, a > 0 → b > 0 → 1/a + 2/b = 2 → a * b ≥ 2) ∧ 
  (∀ a b, a > 0 → b > 0 → 1/a + 2/b = 2 → a + 2*b ≥ 9/2) ∧
  (a = 3/2 ∧ b = 3/2 → a * b = 2 ∧ a + 2*b = 9/2) :=
sorry

end NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l127_12774


namespace NUMINAMATH_CALUDE_max_cone_bound_for_f_l127_12714

/-- A function f: ℝ → ℝ is cone-bottomed if there exists a constant M > 0
    such that |f(x)| ≥ M|x| for all x ∈ ℝ -/
def ConeBounded (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, |f x| ≥ M * |x|

/-- The function f(x) = x^2 + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

theorem max_cone_bound_for_f :
  (ConeBounded f) ∧ (∀ M : ℝ, (∀ x : ℝ, |f x| ≥ M * |x|) → M ≤ 2) ∧
  (∃ x : ℝ, |f x| = 2 * |x|) := by
  sorry


end NUMINAMATH_CALUDE_max_cone_bound_for_f_l127_12714


namespace NUMINAMATH_CALUDE_system_solution_l127_12791

theorem system_solution (x y k : ℝ) : 
  x - y = k + 2 →
  x + 3*y = k →
  x + y = 2 →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l127_12791


namespace NUMINAMATH_CALUDE_turner_ticket_count_l127_12741

/-- The number of times Turner wants to ride the rollercoaster -/
def rollercoaster_rides : ℕ := 3

/-- The number of times Turner wants to ride the Catapult -/
def catapult_rides : ℕ := 2

/-- The number of times Turner wants to ride the Ferris wheel -/
def ferris_wheel_rides : ℕ := 1

/-- The number of tickets required for one rollercoaster ride -/
def rollercoaster_cost : ℕ := 4

/-- The number of tickets required for one Catapult ride -/
def catapult_cost : ℕ := 4

/-- The number of tickets required for one Ferris wheel ride -/
def ferris_wheel_cost : ℕ := 1

/-- The total number of tickets Turner needs -/
def total_tickets : ℕ := 
  rollercoaster_rides * rollercoaster_cost + 
  catapult_rides * catapult_cost + 
  ferris_wheel_rides * ferris_wheel_cost

theorem turner_ticket_count : total_tickets = 21 := by
  sorry

end NUMINAMATH_CALUDE_turner_ticket_count_l127_12741


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l127_12748

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def B : Set ℝ := {y | y^2 - 2*y - 3 ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l127_12748


namespace NUMINAMATH_CALUDE_petya_running_time_l127_12738

theorem petya_running_time (V D : ℝ) (h1 : V > 0) (h2 : D > 0) : 
  (D / (1.25 * V) / 2) + (D / (0.8 * V) / 2) > D / V := by
  sorry

end NUMINAMATH_CALUDE_petya_running_time_l127_12738


namespace NUMINAMATH_CALUDE_alien_arms_count_l127_12773

/-- The number of arms an alien has -/
def alien_arms : ℕ := sorry

/-- The number of legs an alien has -/
def alien_legs : ℕ := 8

/-- The number of arms a Martian has -/
def martian_arms : ℕ := 2 * alien_arms

/-- The number of legs a Martian has -/
def martian_legs : ℕ := alien_legs / 2

theorem alien_arms_count : alien_arms = 3 :=
  by
    have h1 : 5 * (alien_arms + alien_legs) = 5 * (martian_arms + martian_legs) + 5 := by sorry
    sorry

end NUMINAMATH_CALUDE_alien_arms_count_l127_12773


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l127_12745

theorem simplify_sqrt_difference : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l127_12745


namespace NUMINAMATH_CALUDE_cube_sum_is_42_l127_12720

/-- Represents a cube with numbers on its faces -/
structure NumberedCube where
  /-- The smallest number on the cube's faces -/
  smallest : ℕ
  /-- Proof that the smallest number is even -/
  smallest_even : Even smallest

/-- The sum of numbers on opposite faces of the cube -/
def opposite_face_sum (cube : NumberedCube) : ℕ :=
  2 * cube.smallest + 10

/-- The sum of all numbers on the cube's faces -/
def total_sum (cube : NumberedCube) : ℕ :=
  6 * cube.smallest + 30

/-- Theorem stating that the sum of numbers on a cube with the given properties is 42 -/
theorem cube_sum_is_42 (cube : NumberedCube) 
  (h : ∀ (i : Fin 3), opposite_face_sum cube = 2 * cube.smallest + 2 * i + 10) :
  total_sum cube = 42 := by
  sorry


end NUMINAMATH_CALUDE_cube_sum_is_42_l127_12720


namespace NUMINAMATH_CALUDE_triangle_properties_l127_12758

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Theorem statement
theorem triangle_properties (t : Triangle) : 
  -- 1. If A > B, then sin A > sin B
  (t.A > t.B → Real.sin t.A > Real.sin t.B) ∧ 
  -- 2. sin 2A = sin 2B does not necessarily imply isosceles
  ¬(Real.sin (2 * t.A) = Real.sin (2 * t.B) → t.a = t.b) ∧ 
  -- 3. a² + b² = c² does not necessarily imply isosceles
  ¬(t.a^2 + t.b^2 = t.c^2 → t.a = t.b) ∧ 
  -- 4. a² + b² > c² does not necessarily imply largest angle is obtuse
  ¬(t.a^2 + t.b^2 > t.c^2 → t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2) := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l127_12758


namespace NUMINAMATH_CALUDE_product_of_two_numbers_l127_12780

theorem product_of_two_numbers (x y : ℝ) 
  (sum_condition : x + y = 24) 
  (sum_squares_condition : x^2 + y^2 = 400) : 
  x * y = 88 := by
  sorry

end NUMINAMATH_CALUDE_product_of_two_numbers_l127_12780


namespace NUMINAMATH_CALUDE_painted_cubes_multiple_of_unpainted_l127_12792

theorem painted_cubes_multiple_of_unpainted (n : ℕ) : ∃ n, n > 0 ∧ (n + 2)^3 > 10 ∧ n^3 ∣ ((n + 2)^3 - n^3) := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_multiple_of_unpainted_l127_12792


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_inequality_l127_12751

theorem unique_integer_satisfying_inequality :
  ∃! x : ℤ, 3 * x^2 + 14 * x + 24 ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_inequality_l127_12751


namespace NUMINAMATH_CALUDE_circle_distance_relation_l127_12775

/-- Given a circle with center O and radius 2a, prove the relationship between x and y -/
theorem circle_distance_relation (a : ℝ) (x y : ℝ) : 
  x > 0 → a > 0 → y^2 = x^3 / (2*a + x) := by
  sorry


end NUMINAMATH_CALUDE_circle_distance_relation_l127_12775


namespace NUMINAMATH_CALUDE_number_pairing_l127_12767

theorem number_pairing (numbers : List ℕ) (h1 : numbers = [41, 35, 19, 9, 26, 45, 13, 28]) :
  let total_sum := numbers.sum
  let pair_sum := total_sum / 4
  ∃ (pairs : List (ℕ × ℕ)), 
    (∀ p ∈ pairs, p.1 + p.2 = pair_sum) ∧ 
    (∀ n ∈ numbers, ∃ p ∈ pairs, n = p.1 ∨ n = p.2) ∧
    (∃ p ∈ pairs, p = (13, 41) ∨ p = (41, 13)) :=
by sorry

end NUMINAMATH_CALUDE_number_pairing_l127_12767


namespace NUMINAMATH_CALUDE_camel_traveler_water_ratio_l127_12732

/-- Proves that the ratio of water drunk by a camel to that drunk by a traveler is 7:1 under given conditions. -/
theorem camel_traveler_water_ratio : 
  let traveler_water : ℕ := 32
  let ounces_per_gallon : ℕ := 128
  let total_gallons : ℕ := 2
  let total_water : ℕ := total_gallons * ounces_per_gallon
  let camel_water : ℕ := total_water - traveler_water
  (camel_water : ℚ) / traveler_water = 7 := by sorry

end NUMINAMATH_CALUDE_camel_traveler_water_ratio_l127_12732


namespace NUMINAMATH_CALUDE_calculation_proof_l127_12747

theorem calculation_proof :
  (125 * 76 * 4 * 8 * 25 = 7600000) ∧
  ((6742 + 6743 + 6738 + 6739 + 6741 + 6743) / 6 = 6741) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l127_12747


namespace NUMINAMATH_CALUDE_tangent_line_minimum_b_l127_12799

/-- Given a > 0 and y = 2x + b is tangent to y = 2a ln x, the minimum value of b is -2 -/
theorem tangent_line_minimum_b (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, (2 * x + b = 2 * a * Real.log x) ∧ 
             (∀ y : ℝ, y ≠ x → 2 * y + b > 2 * a * Real.log y)) → 
  (∀ c : ℝ, (∃ x : ℝ, (2 * x + c = 2 * a * Real.log x) ∧ 
                       (∀ y : ℝ, y ≠ x → 2 * y + c > 2 * a * Real.log y)) → 
            c ≥ -2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_b_l127_12799


namespace NUMINAMATH_CALUDE_equation_solution_l127_12755

theorem equation_solution (y : ℝ) : 
  (|y - 4|^2 + 3*y = 14) ↔ (y = (5 + Real.sqrt 17)/2 ∨ y = (5 - Real.sqrt 17)/2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l127_12755
