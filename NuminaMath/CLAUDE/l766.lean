import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l766_76610

theorem problem_solution (x y : ℤ) (hx : x = 3) (hy : y = 4) : 3 * x - 5 * y = -11 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l766_76610


namespace NUMINAMATH_CALUDE_age_difference_is_24_l766_76699

/-- Proves that the age difference between Ana and Claudia is 24 years --/
theorem age_difference_is_24 (A C : ℕ) (n : ℕ) : 
  A = C + n →                 -- Ana is n years older than Claudia
  A - 3 = 6 * (C - 3) →       -- Three years ago, Ana was 6 times as old as Claudia
  A = C^3 →                   -- This year Ana's age is the cube of Claudia's age
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_age_difference_is_24_l766_76699


namespace NUMINAMATH_CALUDE_equation_solution_l766_76605

theorem equation_solution :
  ∃ x : ℝ, (7 - 2*x = -3) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l766_76605


namespace NUMINAMATH_CALUDE_train_speed_l766_76637

/-- A train passes a pole in 5 seconds and crosses a 360-meter long stationary train in 25 seconds. -/
theorem train_speed (pole_passing_time : ℝ) (stationary_train_length : ℝ) (crossing_time : ℝ)
  (h1 : pole_passing_time = 5)
  (h2 : stationary_train_length = 360)
  (h3 : crossing_time = 25) :
  ∃ (speed : ℝ), speed = 18 ∧ 
    speed * pole_passing_time = speed * crossing_time - stationary_train_length :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l766_76637


namespace NUMINAMATH_CALUDE_triangle_area_l766_76678

/-- Given a triangle PQR with inradius r, circumradius R, and angles P, Q, R satisfying certain conditions,
    prove that its area is (7√3201)/3 -/
theorem triangle_area (P Q R : ℝ) (r R : ℝ) (h1 : r = 7) (h2 : R = 25) 
    (h3 : 2 * Real.cos Q = Real.cos P + Real.cos R) : 
    ∃ (area : ℝ), area = (7 * Real.sqrt 3201) / 3 ∧ 
    area = r * (P + Q + R) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l766_76678


namespace NUMINAMATH_CALUDE_max_k_value_l766_76642

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 4 = k^2 * (x^2 / y^2 + y^2 / x^2 + 2) + 2 * k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_k_value_l766_76642


namespace NUMINAMATH_CALUDE_circle_from_equation_l766_76674

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in general form -/
def CircleEquation (x y : ℝ) (A B C D E : ℝ) : Prop :=
  A * x^2 + B * x + C * y^2 + D * y + E = 0

theorem circle_from_equation :
  ∃ (c : Circle), 
    (∀ (x y : ℝ), CircleEquation x y 1 (-6) 1 2 (-12) ↔ 
      (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
    c.center = (3, -1) ∧
    c.radius = Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_circle_from_equation_l766_76674


namespace NUMINAMATH_CALUDE_irwin_score_product_l766_76626

/-- Represents the types of baskets in Jamshid and Irwin's basketball game -/
inductive BasketType
  | Two
  | Five
  | Eleven
  | Thirteen

/-- Returns the point value of a given basket type -/
def basketValue (b : BasketType) : ℕ :=
  match b with
  | BasketType.Two => 2
  | BasketType.Five => 5
  | BasketType.Eleven => 11
  | BasketType.Thirteen => 13

/-- Irwin's score at halftime -/
def irwinScore : ℕ := 2 * basketValue BasketType.Eleven

theorem irwin_score_product : irwinScore = 22 := by
  sorry

end NUMINAMATH_CALUDE_irwin_score_product_l766_76626


namespace NUMINAMATH_CALUDE_expression_evaluation_l766_76698

theorem expression_evaluation : (50 - (2210 - 251)) + (2210 - (251 - 50)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l766_76698


namespace NUMINAMATH_CALUDE_line_equal_intercepts_l766_76622

/-- A line with equation ax + y - 2 - a = 0 has equal intercepts on the x-axis and y-axis if and only if a = -2 or a = 1 -/
theorem line_equal_intercepts (a : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ 
    (a * k = k - 2 + a) ∧ 
    (k = k - 2 + a)) ↔ 
  (a = -2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_line_equal_intercepts_l766_76622


namespace NUMINAMATH_CALUDE_complex_equation_magnitude_l766_76688

theorem complex_equation_magnitude (z : ℂ) (a b : ℝ) (n : ℕ) 
  (h : a * z^n + b * Complex.I * z^(n-1) + b * Complex.I * z - a = 0) : 
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_magnitude_l766_76688


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_f_one_over_e_gt_f_one_half_l766_76648

noncomputable def f (x : ℝ) : ℝ := -x / Real.exp x + Real.log 2

theorem f_decreasing_on_interval :
  ∀ x : ℝ, x < 1 → ∀ y : ℝ, x < y → f y < f x :=
sorry

theorem f_one_over_e_gt_f_one_half : f (1 / Real.exp 1) > f (1 / 2) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_f_one_over_e_gt_f_one_half_l766_76648


namespace NUMINAMATH_CALUDE_ellipse_slope_at_pi_third_l766_76683

/-- The slope of the line connecting the origin to a point on an ellipse --/
theorem ellipse_slope_at_pi_third :
  let x (t : Real) := 2 * Real.cos t
  let y (t : Real) := 4 * Real.sin t
  let t₀ : Real := Real.pi / 3
  let x₀ : Real := x t₀
  let y₀ : Real := y t₀
  (y₀ - 0) / (x₀ - 0) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_slope_at_pi_third_l766_76683


namespace NUMINAMATH_CALUDE_circle_equation_and_intersection_points_l766_76614

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 - 1)^2 = 2}

-- Define the line y = x
def line_tangent : Set (ℝ × ℝ) :=
  {p | p.1 = p.2}

-- Define the line l: x - y + a = 0
def line_l (a : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1 - p.2 + a = 0}

theorem circle_equation_and_intersection_points (a : ℝ) 
  (h1 : a ≠ 0)
  (h2 : ∃ p, p ∈ circle_C ∩ line_tangent)
  (h3 : ∃ A B, A ∈ circle_C ∩ line_l a ∧ B ∈ circle_C ∩ line_l a ∧ A ≠ B)
  (h4 : ∀ A B, A ∈ circle_C ∩ line_l a → B ∈ circle_C ∩ line_l a → A ≠ B → 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2) :
  (∀ p, p ∈ circle_C ↔ (p.1 - 3)^2 + (p.2 - 1)^2 = 2) ∧
  (a = Real.sqrt 2 - 2 ∨ a = -Real.sqrt 2 - 2) := by sorry

end NUMINAMATH_CALUDE_circle_equation_and_intersection_points_l766_76614


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l766_76664

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 11*x + 15

-- Define the partial fraction decomposition
def pfd (x A B C : ℝ) : Prop :=
  1 / p x = A / (x - 5) + B / (x + 3) + C / ((x + 3)^2)

-- State the theorem
theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x, pfd x A B C) → (∀ x, p x = (x - 5) * (x + 3)^2) → A = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l766_76664


namespace NUMINAMATH_CALUDE_iced_cube_theorem_l766_76632

/-- Represents a cube with icing on some faces -/
structure IcedCube :=
  (size : ℕ)
  (has_top_icing : Bool)
  (has_lateral_icing : Bool)
  (has_bottom_icing : Bool)

/-- Counts the number of subcubes with icing on exactly two sides -/
def count_two_sided_iced_subcubes (cube : IcedCube) : ℕ :=
  sorry

/-- The main theorem about the 5x5x5 iced cube -/
theorem iced_cube_theorem :
  let cake : IcedCube := {
    size := 5,
    has_top_icing := true,
    has_lateral_icing := true,
    has_bottom_icing := false
  }
  count_two_sided_iced_subcubes cake = 32 :=
sorry

end NUMINAMATH_CALUDE_iced_cube_theorem_l766_76632


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l766_76627

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 - 2 * (a - 1) * x - 2 < 0) ↔ -1 < a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l766_76627


namespace NUMINAMATH_CALUDE_sqrt_sum_division_l766_76601

theorem sqrt_sum_division (x y z : ℝ) : (2 * Real.sqrt 24 + 3 * Real.sqrt 6) / Real.sqrt 3 = 7 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_division_l766_76601


namespace NUMINAMATH_CALUDE_wood_square_weight_relation_l766_76690

/-- Represents the properties of a square piece of wood -/
structure WoodSquare where
  side_length : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two square pieces of wood with uniform density and thickness -/
theorem wood_square_weight_relation 
  (w1 w2 : WoodSquare)
  (uniform_density : True)  -- Represents the assumption of uniform density and thickness
  (h1 : w1.side_length = 4)
  (h2 : w1.weight = 16)
  (h3 : w2.side_length = 6) :
  w2.weight = 36 := by
  sorry

#check wood_square_weight_relation

end NUMINAMATH_CALUDE_wood_square_weight_relation_l766_76690


namespace NUMINAMATH_CALUDE_distance_to_point_one_zero_l766_76673

theorem distance_to_point_one_zero (z : ℂ) (h : z * (1 + Complex.I) = 4) :
  Complex.abs (z - 1) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_one_zero_l766_76673


namespace NUMINAMATH_CALUDE_product_of_ratios_l766_76656

theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 4*x₁*y₁^2 = 3003)
  (h₂ : y₁^3 - 4*x₁^2*y₁ = 3002)
  (h₃ : x₂^3 - 4*x₂*y₂^2 = 3003)
  (h₄ : y₂^3 - 4*x₂^2*y₂ = 3002)
  (h₅ : x₃^3 - 4*x₃*y₃^2 = 3003)
  (h₆ : y₃^3 - 4*x₃^2*y₃ = 3002) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 3/3002 := by
sorry

end NUMINAMATH_CALUDE_product_of_ratios_l766_76656


namespace NUMINAMATH_CALUDE_proportional_function_decreasing_l766_76659

theorem proportional_function_decreasing (k : ℝ) (h1 : k ≠ 0) :
  (∀ x₁ x₂ y₁ y₂ : ℝ, x₁ < x₂ → k * x₁ = y₁ → k * x₂ = y₂ → y₁ > y₂) → k < 0 :=
by sorry

end NUMINAMATH_CALUDE_proportional_function_decreasing_l766_76659


namespace NUMINAMATH_CALUDE_age_difference_l766_76628

/-- Given that Frank and Gabriel's ages sum to 17, Frank is 10 years old,
    and Gabriel is younger than Frank, prove that Gabriel is 3 years younger than Frank. -/
theorem age_difference (frank_age gabriel_age : ℕ) : 
  frank_age + gabriel_age = 17 →
  frank_age = 10 →
  gabriel_age < frank_age →
  frank_age - gabriel_age = 3 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l766_76628


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l766_76639

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of enchanted stones available. -/
def num_stones : ℕ := 6

/-- The number of stones incompatible with a specific herb. -/
def incompatible_stones : ℕ := 3

/-- The number of herbs that have incompatible stones. -/
def herbs_with_incompatibility : ℕ := 1

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_stones - incompatible_stones * herbs_with_incompatibility

theorem wizard_elixir_combinations :
  valid_combinations = 21 :=
sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l766_76639


namespace NUMINAMATH_CALUDE_floor_sum_opposite_l766_76611

theorem floor_sum_opposite (x : ℝ) (h : x = 15.8) : 
  ⌊x⌋ + ⌊-x⌋ = -1 := by sorry

end NUMINAMATH_CALUDE_floor_sum_opposite_l766_76611


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l766_76681

/-- An increasing arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b → b 4 * b 5 = 21 → b 3 * b 6 = -779 ∨ b 3 * b 6 = -11 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l766_76681


namespace NUMINAMATH_CALUDE_equation_solutions_l766_76644

theorem equation_solutions :
  (∀ x : ℝ, 9 * (x - 1)^2 = 25 ↔ x = 8/3 ∨ x = -2/3) ∧
  (∀ x : ℝ, (1/3) * (x + 2)^3 - 9 = 0 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l766_76644


namespace NUMINAMATH_CALUDE_smallest_with_twelve_factors_l766_76630

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The set of positive factors of a positive integer -/
def factors (n : ℕ+) : Set ℕ+ := sorry

theorem smallest_with_twelve_factors :
  ∃ (n : ℕ+), (num_factors n = 12) ∧
    (∀ m : ℕ+, m < n → num_factors m ≠ 12) ∧
    (n = 60) := by sorry

end NUMINAMATH_CALUDE_smallest_with_twelve_factors_l766_76630


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l766_76697

theorem sin_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (γ * Complex.I) = 4/5 + 3/5 * Complex.I →
  Complex.exp (δ * Complex.I) = -5/13 + 12/13 * Complex.I →
  Real.sin (γ + δ) = 33/65 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l766_76697


namespace NUMINAMATH_CALUDE_vector_relationships_l766_76657

/-- Given vector a and unit vector b, prove their parallel and perpendicular relationships -/
theorem vector_relationships (a b : ℝ × ℝ) : 
  a = (3, 4) → 
  (b.1^2 + b.2^2 = 1) → 
  ((∃ k : ℝ, b = k • a) → (b = (3/5, 4/5) ∨ b = (-3/5, -4/5))) ∧ 
  ((a.1 * b.1 + a.2 * b.2 = 0) → (b = (-4/5, 3/5) ∨ b = (4/5, -3/5))) := by
  sorry

end NUMINAMATH_CALUDE_vector_relationships_l766_76657


namespace NUMINAMATH_CALUDE_compound_carbon_count_l766_76670

/-- Represents the number of atoms of a given element in a compound -/
structure AtomCount where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : ℝ
  hydrogen : ℝ
  oxygen : ℝ

/-- Calculates the molecular weight of a compound given its atom count and atomic weights -/
def molecularWeight (count : AtomCount) (weights : AtomicWeights) : ℝ :=
  count.carbon * weights.carbon +
  count.hydrogen * weights.hydrogen +
  count.oxygen * weights.oxygen

/-- The theorem to be proved -/
theorem compound_carbon_count (weights : AtomicWeights)
    (h_carbon : weights.carbon = 12.01)
    (h_hydrogen : weights.hydrogen = 1.01)
    (h_oxygen : weights.oxygen = 16.00) :
    ∃ (count : AtomCount),
      count.hydrogen = 8 ∧
      count.oxygen = 2 ∧
      molecularWeight count weights = 88 ∧
      count.carbon = 4 := by
  sorry

end NUMINAMATH_CALUDE_compound_carbon_count_l766_76670


namespace NUMINAMATH_CALUDE_sum_of_parts_l766_76695

theorem sum_of_parts (x y : ℝ) : x + y = 24 → y = 13 → y > x → 7 * x + 5 * y = 142 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_l766_76695


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l766_76602

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def digit_sum (n : ℕ) : ℕ := (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digits_sum (n : ℕ) : ℕ := ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℕ := (n / 1000) - (n % 10)

theorem unique_four_digit_number : 
  ∃! n : ℕ, 
    is_four_digit n ∧ 
    digit_sum n = 16 ∧ 
    middle_digits_sum n = 10 ∧ 
    thousands_minus_units n = 2 ∧ 
    n % 11 = 0 ∧
    n = 4642 := by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l766_76602


namespace NUMINAMATH_CALUDE_rope_length_l766_76652

/-- Given a rope cut into two parts with a ratio of 2:3, where the shorter part is 16 meters long,
    the total length of the rope is 40 meters. -/
theorem rope_length (shorter_part : ℝ) (ratio_short : ℝ) (ratio_long : ℝ) :
  shorter_part = 16 →
  ratio_short = 2 →
  ratio_long = 3 →
  (shorter_part / ratio_short) * (ratio_short + ratio_long) = 40 :=
by sorry

end NUMINAMATH_CALUDE_rope_length_l766_76652


namespace NUMINAMATH_CALUDE_alcohol_percentage_solution_y_l766_76679

theorem alcohol_percentage_solution_y :
  let alcohol_x : ℝ := 0.1  -- 10% alcohol in solution x
  let volume_x : ℝ := 300   -- 300 mL of solution x
  let volume_y : ℝ := 900   -- 900 mL of solution y
  let total_volume : ℝ := volume_x + volume_y
  let final_alcohol_percentage : ℝ := 0.25  -- 25% alcohol in final solution
  let alcohol_y : ℝ := (final_alcohol_percentage * total_volume - alcohol_x * volume_x) / volume_y
  alcohol_y = 0.3  -- 30% alcohol in solution y
  := by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_solution_y_l766_76679


namespace NUMINAMATH_CALUDE_vampire_population_after_two_nights_l766_76671

def vampire_growth (initial_vampires : ℕ) (new_vampires_per_night : ℕ) (nights : ℕ) : ℕ :=
  initial_vampires * (new_vampires_per_night + 1)^nights

theorem vampire_population_after_two_nights :
  vampire_growth 3 7 2 = 192 :=
by sorry

end NUMINAMATH_CALUDE_vampire_population_after_two_nights_l766_76671


namespace NUMINAMATH_CALUDE_particle_speed_l766_76696

/-- A particle moves so that its position at time t is (3t + 5, 6t - 11).
    This function represents the particle's position vector at time t. -/
def particle_position (t : ℝ) : ℝ × ℝ := (3 * t + 5, 6 * t - 11)

/-- The speed of the particle is the magnitude of the change in position vector
    per unit time interval. -/
theorem particle_speed : 
  let v := particle_position 1 - particle_position 0
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_particle_speed_l766_76696


namespace NUMINAMATH_CALUDE_proposition_b_l766_76617

theorem proposition_b (a b c : ℝ) : a < b → a * c^2 ≤ b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_proposition_b_l766_76617


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_l766_76613

theorem rectangular_prism_surface_area
  (r : ℝ) (l w h : ℝ) 
  (h_r : r = 3 * (36 / Real.pi))
  (h_l : l = 6)
  (h_w : w = 4)
  (h_vol_eq : (4 / 3) * Real.pi * r^3 = l * w * h) :
  2 * (l * w + l * h + w * h) = 88 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_l766_76613


namespace NUMINAMATH_CALUDE_school_population_theorem_l766_76667

theorem school_population_theorem :
  ∀ (boys girls : ℕ),
  boys + girls = 150 →
  girls = (boys * 100) / 150 →
  boys = 90 := by
sorry

end NUMINAMATH_CALUDE_school_population_theorem_l766_76667


namespace NUMINAMATH_CALUDE_equivalent_division_l766_76693

theorem equivalent_division (x : ℝ) :
  x / (4^3 / 8) * Real.sqrt (7 / 5) = x / ((8 * Real.sqrt 35) / 5) := by sorry

end NUMINAMATH_CALUDE_equivalent_division_l766_76693


namespace NUMINAMATH_CALUDE_infinitely_many_triangular_squares_l766_76665

/-- Definition of triangular numbers -/
def T (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate for a number being square -/
def is_square (k : ℕ) : Prop := ∃ m : ℕ, k = m * m

/-- The recurrence relation for generating triangular square numbers -/
axiom recurrence_relation (n : ℕ) : T (4 * n * (n + 1)) = 4 * T n * (2 * n + 1)^2

/-- Theorem: There are infinitely many numbers that are both triangular and square -/
theorem infinitely_many_triangular_squares :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ is_square (T n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_triangular_squares_l766_76665


namespace NUMINAMATH_CALUDE_correct_total_distance_l766_76621

/-- The total distance to fly from Germany to Russia and then return to Spain -/
def totalDistance (spainRussia : ℕ) (spainGermany : ℕ) : ℕ :=
  (spainRussia - spainGermany) + spainRussia

theorem correct_total_distance :
  totalDistance 7019 1615 = 12423 := by
  sorry

#eval totalDistance 7019 1615

end NUMINAMATH_CALUDE_correct_total_distance_l766_76621


namespace NUMINAMATH_CALUDE_instantaneous_rate_of_change_l766_76606

/-- Given a curve y = x^2 + 2x, prove that if the instantaneous rate of change
    at a point M is 6, then the coordinates of point M are (2, 8). -/
theorem instantaneous_rate_of_change (x y : ℝ) : 
  y = x^2 + 2*x →                             -- Curve equation
  (2*x + 2 : ℝ) = 6 →                         -- Instantaneous rate of change is 6
  (x, y) = (2, 8) :=                          -- Coordinates of point M
by sorry

end NUMINAMATH_CALUDE_instantaneous_rate_of_change_l766_76606


namespace NUMINAMATH_CALUDE_unique_n_mod_59_l766_76607

theorem unique_n_mod_59 : ∃! n : ℤ, 0 ≤ n ∧ n < 59 ∧ 58 * n % 59 = 20 % 59 ∧ n = 39 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_mod_59_l766_76607


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l766_76666

/-- The focus of a parabola y² = 12x -/
def parabola_focus : ℝ × ℝ := (3, 0)

/-- The equation of a hyperbola -/
def is_hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 = 1

/-- The equation of asymptotes of a hyperbola -/
def is_asymptote (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x ∨ y = -k * x

/-- Main theorem -/
theorem hyperbola_asymptotes :
  ∃ (a : ℝ), (is_hyperbola a (parabola_focus.1) (parabola_focus.2)) →
  (∀ (x y : ℝ), is_asymptote (1/3) x y ↔ is_hyperbola a x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l766_76666


namespace NUMINAMATH_CALUDE_count_two_digit_multiples_of_eight_l766_76668

theorem count_two_digit_multiples_of_eight : 
  (Finset.filter (fun n => n % 8 = 0) (Finset.range 100)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_two_digit_multiples_of_eight_l766_76668


namespace NUMINAMATH_CALUDE_drawings_on_last_page_l766_76643

theorem drawings_on_last_page 
  (initial_notebooks : Nat) 
  (pages_per_notebook : Nat) 
  (initial_drawings_per_page : Nat) 
  (reorganized_drawings_per_page : Nat)
  (filled_notebooks : Nat)
  (filled_pages_last_notebook : Nat) :
  initial_notebooks = 12 →
  pages_per_notebook = 35 →
  initial_drawings_per_page = 4 →
  reorganized_drawings_per_page = 7 →
  filled_notebooks = 6 →
  filled_pages_last_notebook = 25 →
  (initial_notebooks * pages_per_notebook * initial_drawings_per_page) -
  (filled_notebooks * pages_per_notebook * reorganized_drawings_per_page) -
  (filled_pages_last_notebook * reorganized_drawings_per_page) = 5 := by
  sorry

end NUMINAMATH_CALUDE_drawings_on_last_page_l766_76643


namespace NUMINAMATH_CALUDE_toy_store_spending_l766_76612

/-- Proof of student's spending at toy store -/
theorem toy_store_spending (total_allowance : ℚ) 
  (arcade_fraction : ℚ) (candy_spending : ℚ) :
  total_allowance = 4.5 →
  arcade_fraction = 3/5 →
  candy_spending = 1.2 →
  let remaining_after_arcade := total_allowance - (arcade_fraction * total_allowance)
  let toy_store_spending := remaining_after_arcade - candy_spending
  toy_store_spending / remaining_after_arcade = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_spending_l766_76612


namespace NUMINAMATH_CALUDE_smallest_n_value_l766_76647

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3000 ∧ Even a

def factorial_product_divisibility (a b c n : ℕ) (m : ℤ) : Prop :=
  ∃ (k : ℤ), (a.factorial * b.factorial * c.factorial : ℤ) = m * 10^n ∧ ¬(10 ∣ m)

theorem smallest_n_value (a b c : ℕ) (m : ℤ) :
  is_valid_triple a b c →
  (∃ n : ℕ, factorial_product_divisibility a b c n m) →
  ∃ n : ℕ, factorial_product_divisibility a b c n m ∧
    ∀ k : ℕ, factorial_product_divisibility a b c k m → n ≤ k ∧ n = 496 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l766_76647


namespace NUMINAMATH_CALUDE_max_pieces_on_chessboard_l766_76619

/-- Represents a chessboard configuration -/
def ChessboardConfiguration := Fin 8 → Fin 8 → Bool

/-- Checks if a given position is on the board -/
def isOnBoard (row col : ℕ) : Prop := row < 8 ∧ col < 8

/-- Checks if a piece is placed at a given position -/
def hasPiece (config : ChessboardConfiguration) (row col : Fin 8) : Prop :=
  config row col = true

/-- Counts the number of pieces on a given diagonal -/
def piecesOnDiagonal (config : ChessboardConfiguration) (startRow startCol : Fin 8) (rowStep colStep : Int) : ℕ :=
  sorry

/-- Checks if the configuration is valid (no more than 3 pieces on any diagonal) -/
def isValidConfiguration (config : ChessboardConfiguration) : Prop :=
  ∀ (startRow startCol : Fin 8) (rowStep colStep : Int),
    piecesOnDiagonal config startRow startCol rowStep colStep ≤ 3

/-- Counts the total number of pieces on the board -/
def totalPieces (config : ChessboardConfiguration) : ℕ :=
  sorry

/-- The main theorem -/
theorem max_pieces_on_chessboard :
  ∃ (config : ChessboardConfiguration),
    isValidConfiguration config ∧
    totalPieces config = 38 ∧
    ∀ (otherConfig : ChessboardConfiguration),
      isValidConfiguration otherConfig →
      totalPieces otherConfig ≤ 38 :=
  sorry

end NUMINAMATH_CALUDE_max_pieces_on_chessboard_l766_76619


namespace NUMINAMATH_CALUDE_triangle_properties_l766_76638

/-- Represents a triangle with sides x, y, and z -/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if the triangle satisfies the given conditions -/
def satisfiesConditions (t : Triangle) (a : ℝ) : Prop :=
  t.x + t.y = 3 * t.z ∧
  t.z + t.y = t.x + a ∧
  t.x + t.z = 60

/-- Theorem stating the properties of the triangle -/
theorem triangle_properties :
  ∀ (t : Triangle) (a : ℝ),
    satisfiesConditions t a →
    (0 < a ∧ a < 60) ∧
    (a = 30 → t.x = 35 ∧ t.y = 40 ∧ t.z = 25) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l766_76638


namespace NUMINAMATH_CALUDE_last_locker_theorem_l766_76620

/-- The number of lockers in the hall -/
def num_lockers : ℕ := 2048

/-- The pattern of opening lockers -/
def open_pattern (n : ℕ) : Bool :=
  if n % 3 = 1 then true  -- opened in first pass
  else if n % 3 = 2 then true  -- opened in second pass
  else false  -- opened in third pass

/-- The last locker opened is the largest multiple of 3 not exceeding the number of lockers -/
def last_locker_opened (total : ℕ) : ℕ :=
  total - (total % 3)

theorem last_locker_theorem :
  last_locker_opened num_lockers = 2046 ∧
  ∀ n, n > last_locker_opened num_lockers → n ≤ num_lockers → open_pattern n = false :=
by sorry

end NUMINAMATH_CALUDE_last_locker_theorem_l766_76620


namespace NUMINAMATH_CALUDE_delta_cheaper_from_min_posters_gamma_cheaper_or_equal_before_min_posters_l766_76608

/-- Delta Printing Company's pricing function -/
def delta_price (n : ℕ) : ℝ := 40 + 7 * n

/-- Gamma Printing Company's pricing function -/
def gamma_price (n : ℕ) : ℝ := 11 * n

/-- The minimum number of posters for which Delta is cheaper than Gamma -/
def min_posters_for_delta : ℕ := 11

theorem delta_cheaper_from_min_posters :
  ∀ n : ℕ, n ≥ min_posters_for_delta → delta_price n < gamma_price n :=
sorry

theorem gamma_cheaper_or_equal_before_min_posters :
  ∀ n : ℕ, n < min_posters_for_delta → delta_price n ≥ gamma_price n :=
sorry

end NUMINAMATH_CALUDE_delta_cheaper_from_min_posters_gamma_cheaper_or_equal_before_min_posters_l766_76608


namespace NUMINAMATH_CALUDE_marcus_walking_speed_l766_76629

/-- Calculates Marcus's walking speed given the conditions of his dog care routine -/
theorem marcus_walking_speed (bath_time : ℝ) (total_time : ℝ) (walk_distance : ℝ) : 
  bath_time = 20 →
  total_time = 60 →
  walk_distance = 3 →
  (walk_distance / (total_time - bath_time - bath_time / 2)) * 60 = 6 := by
sorry

end NUMINAMATH_CALUDE_marcus_walking_speed_l766_76629


namespace NUMINAMATH_CALUDE_car_hire_problem_l766_76663

theorem car_hire_problem (total_cost : ℝ) (a_hours c_hours : ℝ) (b_cost : ℝ) :
  total_cost = 520 →
  a_hours = 7 →
  c_hours = 11 →
  b_cost = 160 →
  ∃ b_hours : ℝ,
    b_cost = (total_cost / (a_hours + b_hours + c_hours)) * b_hours ∧
    b_hours = 8 := by
  sorry

end NUMINAMATH_CALUDE_car_hire_problem_l766_76663


namespace NUMINAMATH_CALUDE_choose_four_from_seven_l766_76616

theorem choose_four_from_seven : Nat.choose 7 4 = 35 := by sorry

end NUMINAMATH_CALUDE_choose_four_from_seven_l766_76616


namespace NUMINAMATH_CALUDE_profit_percentage_l766_76691

theorem profit_percentage (selling_price cost_price : ℝ) :
  cost_price = 0.96 * selling_price →
  (selling_price - cost_price) / cost_price * 100 = (1 / 24) * 100 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_l766_76691


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l766_76692

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x > 1}

-- State the theorem
theorem domain_of_composite_function :
  {x : ℝ | ∃ y ∈ domain_f, y = 2*x + 1} = {x : ℝ | x > 0} := by sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l766_76692


namespace NUMINAMATH_CALUDE_like_terms_difference_l766_76651

def are_like_terms (a b : ℕ → ℕ → ℚ) : Prop :=
  ∃ (c d : ℚ) (m n : ℕ), ∀ (x y : ℕ), a x y = c * x^m * y^3 ∧ b x y = d * x^4 * y^n

theorem like_terms_difference (m n : ℕ) :
  are_like_terms (λ x y => -3 * x^m * y^3) (λ x y => 2 * x^4 * y^n) → m - n = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_difference_l766_76651


namespace NUMINAMATH_CALUDE_solution_difference_l766_76687

theorem solution_difference (r s : ℝ) : 
  (∀ x : ℝ, (5 * x - 15) / (x^2 + 3 * x - 18) = x + 3 → x = r ∨ x = s) →
  r ≠ s →
  r > s →
  r - s = Real.sqrt 29 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l766_76687


namespace NUMINAMATH_CALUDE_square_difference_area_l766_76625

theorem square_difference_area (a b : ℝ) : 
  (a + b)^2 - a^2 = 2*a*b + b^2 := by sorry

end NUMINAMATH_CALUDE_square_difference_area_l766_76625


namespace NUMINAMATH_CALUDE_arithmetic_square_root_property_l766_76661

theorem arithmetic_square_root_property (π : ℝ) : 
  Real.sqrt ((π - 4)^2) = 4 - π := by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_property_l766_76661


namespace NUMINAMATH_CALUDE_star_interior_angle_sum_l766_76618

/-- An n-pointed star constructed from an n-sided convex polygon -/
structure StarPolygon where
  n : ℕ
  n_ge_6 : n ≥ 6

/-- The sum of interior angles at the vertices of the star -/
def interior_angle_sum (star : StarPolygon) : ℝ :=
  180 * (star.n - 2)

/-- Theorem: The sum of interior angles at the vertices of an n-pointed star
    constructed by extending every third side of an n-sided convex polygon (n ≥ 6)
    is equal to 180° * (n - 2) -/
theorem star_interior_angle_sum (star : StarPolygon) :
  interior_angle_sum star = 180 * (star.n - 2) := by
  sorry

end NUMINAMATH_CALUDE_star_interior_angle_sum_l766_76618


namespace NUMINAMATH_CALUDE_polynomial_division_l766_76672

theorem polynomial_division (x : ℝ) (h : x ≠ 0) : 2 * x^3 / x^2 = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l766_76672


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l766_76646

def income : ℕ := 21000
def savings : ℕ := 7000
def expenditure : ℕ := income - savings

theorem income_expenditure_ratio : 
  (income : ℚ) / (expenditure : ℚ) = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l766_76646


namespace NUMINAMATH_CALUDE_x_value_when_one_in_set_l766_76675

theorem x_value_when_one_in_set (x : ℝ) : 1 ∈ ({x, x^2} : Set ℝ) → x ≠ x^2 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_one_in_set_l766_76675


namespace NUMINAMATH_CALUDE_square_plus_one_nonzero_l766_76633

theorem square_plus_one_nonzero : ∀ x : ℝ, x^2 + 1 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_square_plus_one_nonzero_l766_76633


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l766_76604

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 4*x - 7 = 0 ↔ (x - 2)^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l766_76604


namespace NUMINAMATH_CALUDE_find_x_l766_76685

theorem find_x : ∃ x : ℤ, (9873 + x = 13800) ∧ (x = 3927) := by
  sorry

end NUMINAMATH_CALUDE_find_x_l766_76685


namespace NUMINAMATH_CALUDE_largest_five_digit_multiple_largest_five_digit_multiple_exists_l766_76603

theorem largest_five_digit_multiple (n : Nat) : n ≤ 99999 ∧ n ≥ 10000 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n → n ≤ 99936 :=
by
  sorry

theorem largest_five_digit_multiple_exists : ∃ n : Nat, n = 99936 ∧ n ≤ 99999 ∧ n ≥ 10000 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_multiple_largest_five_digit_multiple_exists_l766_76603


namespace NUMINAMATH_CALUDE_volume_for_56_ounces_l766_76634

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℚ
  /-- Assumption that k is positive -/
  k_pos : k > 0

/-- The volume of the substance given its weight -/
def volume (s : Substance) (weight : ℚ) : ℚ :=
  s.k * weight

theorem volume_for_56_ounces (s : Substance) 
  (h : volume s 112 = 48) : volume s 56 = 24 := by
  sorry

#check volume_for_56_ounces

end NUMINAMATH_CALUDE_volume_for_56_ounces_l766_76634


namespace NUMINAMATH_CALUDE_min_value_of_polynomial_l766_76624

theorem min_value_of_polynomial (x : ℝ) : 
  x * (x + 4) * (x + 8) * (x + 12) ≥ -256 ∧ 
  ∃ y : ℝ, y * (y + 4) * (y + 8) * (y + 12) = -256 := by sorry

end NUMINAMATH_CALUDE_min_value_of_polynomial_l766_76624


namespace NUMINAMATH_CALUDE_area_smaller_circle_l766_76600

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  r : ℝ  -- radius of the smaller circle
  R : ℝ  -- radius of the larger circle
  h_positive : 0 < r
  h_tangent : R = 2 * r
  h_common_tangent : ∃ (P A B : ℝ × ℝ), 
    let d := Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)
    d = 5 ∧ d = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

/-- The area of the smaller circle in a TangentCircles configuration is 25π/8 -/
theorem area_smaller_circle (tc : TangentCircles) : 
  π * tc.r^2 = 25 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_area_smaller_circle_l766_76600


namespace NUMINAMATH_CALUDE_wage_cut_and_raise_l766_76682

theorem wage_cut_and_raise (original_wage : ℝ) (h : original_wage > 0) :
  let reduced_wage := 0.8 * original_wage
  let raised_wage := reduced_wage * 1.25
  raised_wage = original_wage :=
by sorry

end NUMINAMATH_CALUDE_wage_cut_and_raise_l766_76682


namespace NUMINAMATH_CALUDE_candy_store_sales_l766_76680

-- Define the quantities and prices
def fudge_pounds : ℕ := 20
def fudge_price : ℚ := 2.5
def truffle_dozens : ℕ := 5
def truffle_price : ℚ := 1.5
def pretzel_dozens : ℕ := 3
def pretzel_price : ℚ := 2

-- Define the calculation for total sales
def total_sales : ℚ :=
  fudge_pounds * fudge_price +
  truffle_dozens * 12 * truffle_price +
  pretzel_dozens * 12 * pretzel_price

-- Theorem statement
theorem candy_store_sales : total_sales = 212 := by
  sorry

end NUMINAMATH_CALUDE_candy_store_sales_l766_76680


namespace NUMINAMATH_CALUDE_quadratic_inequality_l766_76689

theorem quadratic_inequality (a x : ℝ) : 
  a * x^2 - (a + 1) * x + 1 < 0 ↔ 
    (a = 0 ∧ x > 1) ∨
    (a < 0 ∧ (x < 1/a ∨ x > 1)) ∨
    (0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1/a) ∨
    (a > 1 ∧ 1/a < x ∧ x < 1) ∨
    (a ≠ 1) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l766_76689


namespace NUMINAMATH_CALUDE_triangle_side_angle_ratio_l766_76645

theorem triangle_side_angle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  b^2 = a * c →
  a^2 - c^2 = a * c - b * c →
  c / (b * Real.sin B) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_angle_ratio_l766_76645


namespace NUMINAMATH_CALUDE_calculator_time_saved_l766_76649

/-- Proves that using a calculator saves 150 minutes for Matt's math homework -/
theorem calculator_time_saved 
  (time_with_calc : ℕ) 
  (time_without_calc : ℕ) 
  (num_problems : ℕ) 
  (h1 : time_with_calc = 3)
  (h2 : time_without_calc = 8)
  (h3 : num_problems = 30) :
  time_without_calc * num_problems - time_with_calc * num_problems = 150 :=
by sorry

end NUMINAMATH_CALUDE_calculator_time_saved_l766_76649


namespace NUMINAMATH_CALUDE_complex_on_ray_unit_circle_l766_76655

theorem complex_on_ray_unit_circle (z : ℂ) (a b : ℝ) :
  z = a + b * I →
  a = b →
  a ≥ 0 →
  Complex.abs z = 1 →
  z = Complex.mk (Real.sqrt 2 / 2) (Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_on_ray_unit_circle_l766_76655


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l766_76641

-- Define sets A, B, and C
def A : Set ℝ := {x | (x + 4) * (x - 2) > 0}
def B : Set ℝ := {y | ∃ x, y = x^2 - 2*x + 2}
def C (a : ℝ) : Set ℝ := {x | -4 ≤ x ∧ x ≤ a}

-- Define the complement of A relative to C
def C_R_A (a : ℝ) : Set ℝ := {x | x ∈ C a ∧ x ∉ A}

-- Theorem for part (I)
theorem intersection_A_B : A ∩ B = {x | x > 2} := by sorry

-- Theorem for part (II)
theorem range_of_a (a : ℝ) : C_R_A a ⊆ C a → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l766_76641


namespace NUMINAMATH_CALUDE_percentage_difference_l766_76636

theorem percentage_difference (x y : ℝ) (h : y = x + 0.6667 * x) :
  x = y * (1 - 0.6667) := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l766_76636


namespace NUMINAMATH_CALUDE_number_equation_solution_l766_76658

theorem number_equation_solution : 
  ∀ x : ℝ, (2/5 : ℝ) * x - 3 * ((1/4 : ℝ) * x) + 7 = 14 → x = -20 := by
sorry

end NUMINAMATH_CALUDE_number_equation_solution_l766_76658


namespace NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l766_76654

/-- The coefficient of the linear term in the quadratic equation x^2 - x = 0 is -1 -/
theorem linear_coefficient_of_quadratic (x : ℝ) : 
  (fun x => x^2 - x) = (fun x => x^2 - 1*x) :=
by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l766_76654


namespace NUMINAMATH_CALUDE_plane_stops_at_20_seconds_stop_time_unique_l766_76650

/-- The distance function representing the plane's movement after landing -/
def s (t : ℝ) : ℝ := -1.5 * t^2 + 60 * t

/-- The time at which the plane stops -/
def stop_time : ℝ := 20

/-- Theorem stating that the plane stops at 20 seconds -/
theorem plane_stops_at_20_seconds :
  (∀ t : ℝ, t ≥ 0 → s t ≤ s stop_time) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ t, |t - stop_time| < δ → s t < s stop_time) := by
  sorry

/-- Corollary: The stop time is unique -/
theorem stop_time_unique (t : ℝ) :
  (∀ τ : ℝ, τ ≥ 0 → s τ ≤ s t) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ τ, |τ - t| < δ → s τ < s t) →
  t = stop_time := by
  sorry

end NUMINAMATH_CALUDE_plane_stops_at_20_seconds_stop_time_unique_l766_76650


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l766_76677

-- Problem 1
theorem problem_1 : 4.7 + (-2.5) - (-5.3) - 7.5 = 0 := by sorry

-- Problem 2
theorem problem_2 : 18 + 48 / (-2)^2 - (-4)^2 * 5 = -50 := by sorry

-- Problem 3
theorem problem_3 : -1^4 + (-2)^2 / 4 * (5 - (-3)^2) = -5 := by sorry

-- Problem 4
theorem problem_4 : (-19 - 15/16) * 8 = -159 - 1/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l766_76677


namespace NUMINAMATH_CALUDE_square_perimeter_l766_76676

theorem square_perimeter (s : ℝ) (h : s = 13) : 4 * s = 52 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l766_76676


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_7_l766_76609

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isFirstYearAfter2010WithDigitSum7 (year : ℕ) : Prop :=
  year > 2010 ∧ 
  sumOfDigits year = 7 ∧ 
  ∀ y, 2010 < y ∧ y < year → sumOfDigits y ≠ 7

theorem first_year_after_2010_with_digit_sum_7 : 
  isFirstYearAfter2010WithDigitSum7 2014 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_7_l766_76609


namespace NUMINAMATH_CALUDE_polynomial_factorization_l766_76640

theorem polynomial_factorization (x y z : ℝ) :
  x * (y - z)^4 + y * (z - x)^4 + z * (x - y)^4 =
  (x - y) * (y - z) * (z - x) * (-(x - y)^2 - (y - z)^2 - (z - x)^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l766_76640


namespace NUMINAMATH_CALUDE_deductive_reasoning_correctness_l766_76669

-- Define the components of deductive reasoning
structure DeductiveReasoning where
  majorPremise : Prop
  minorPremise : Prop
  formOfReasoning : Prop
  conclusion : Prop

-- Define the correctness of each component
def isCorrect (p : Prop) : Prop := p

-- Theorem statement
theorem deductive_reasoning_correctness 
  (dr : DeductiveReasoning) 
  (h1 : isCorrect dr.majorPremise) 
  (h2 : isCorrect dr.minorPremise) 
  (h3 : isCorrect dr.formOfReasoning) : 
  isCorrect dr.conclusion :=
sorry

end NUMINAMATH_CALUDE_deductive_reasoning_correctness_l766_76669


namespace NUMINAMATH_CALUDE_lottery_prizes_approx_10_l766_76662

-- Define the number of blanks
def num_blanks : ℕ := 25

-- Define the probability of drawing a blank
def blank_probability : ℚ := 5000000000000000/7000000000000000

-- Define the function to calculate the number of prizes
def calculate_prizes (blanks : ℕ) (prob : ℚ) : ℚ :=
  (blanks : ℚ) / prob - blanks

-- Theorem statement
theorem lottery_prizes_approx_10 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |calculate_prizes num_blanks blank_probability - 10| < ε :=
sorry

end NUMINAMATH_CALUDE_lottery_prizes_approx_10_l766_76662


namespace NUMINAMATH_CALUDE_clock_setback_radians_l766_76694

theorem clock_setback_radians (minutes_per_revolution : ℝ) (radians_per_revolution : ℝ) 
  (setback_minutes : ℝ) : 
  minutes_per_revolution = 60 → 
  radians_per_revolution = 2 * Real.pi → 
  setback_minutes = 10 → 
  (setback_minutes / minutes_per_revolution) * radians_per_revolution = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_clock_setback_radians_l766_76694


namespace NUMINAMATH_CALUDE_seven_valid_triples_l766_76635

/-- The number of valid triples (a, b, c) for the prism cutting problem -/
def count_valid_triples : ℕ :=
  let b := 2023
  (Finset.filter (fun p : ℕ × ℕ =>
    let a := p.1
    let c := p.2
    a ≤ b ∧ b ≤ c ∧ a * c = b * b
  ) (Finset.product (Finset.range (b + 1)) (Finset.range (b * b + 1)))).card

/-- The main theorem stating there are exactly 7 valid triples -/
theorem seven_valid_triples : count_valid_triples = 7 := by
  sorry


end NUMINAMATH_CALUDE_seven_valid_triples_l766_76635


namespace NUMINAMATH_CALUDE_cosine_equation_solution_l766_76631

theorem cosine_equation_solution (x : ℝ) : 
  (1 + Real.cos (3 * x) = 2 * Real.cos (2 * x)) ↔ 
  (∃ k : ℤ, x = 2 * k * Real.pi ∨ x = Real.pi / 6 + k * Real.pi ∨ x = 5 * Real.pi / 6 + k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_cosine_equation_solution_l766_76631


namespace NUMINAMATH_CALUDE_die_roll_frequency_l766_76615

def is_die_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6

def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let μ := average l
  (l.map (λ x => ((x : ℚ) - μ) ^ 2)).sum / l.length

theorem die_roll_frequency (l : List ℕ) :
  l.length = 5 ∧
  (∀ n ∈ l, is_die_roll n) ∧
  average l = 3 ∧
  variance l = 0.4 →
  l.count 2 = 1 := by sorry

end NUMINAMATH_CALUDE_die_roll_frequency_l766_76615


namespace NUMINAMATH_CALUDE_overlap_time_theorem_l766_76623

structure MovingSegment where
  length : ℝ
  initialPosition : ℝ
  speed : ℝ

def positionAt (s : MovingSegment) (t : ℝ) : ℝ :=
  s.initialPosition + s.speed * t

theorem overlap_time_theorem (ab mn : MovingSegment)
  (hab : ab.length = 100)
  (hmn : mn.length = 40)
  (hab_init : ab.initialPosition = 120)
  (hab_speed : ab.speed = -50)
  (hmn_init : mn.initialPosition = -30)
  (hmn_speed : mn.speed = 30)
  (overlap : ℝ) (hoverlap : overlap = 32) :
  ∃ t : ℝ, (t = 71/40 ∨ t = 109/40) ∧
    (positionAt ab t + ab.length - positionAt mn t = overlap ∨
     positionAt mn t + mn.length - positionAt ab t = overlap) :=
sorry

end NUMINAMATH_CALUDE_overlap_time_theorem_l766_76623


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l766_76660

theorem consecutive_integers_problem (x y z : ℤ) :
  (x = y + 1) →
  (y = z + 1) →
  (x > y) →
  (y > z) →
  (2 * x + 3 * y + 3 * z = 5 * y + 8) →
  z = 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l766_76660


namespace NUMINAMATH_CALUDE_whatsapp_messages_l766_76653

theorem whatsapp_messages (monday tuesday wednesday thursday : ℕ) :
  monday = 300 →
  tuesday = 200 →
  thursday = 2 * wednesday →
  monday + tuesday + wednesday + thursday = 2000 →
  wednesday - tuesday = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_whatsapp_messages_l766_76653


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l766_76684

/-- An arithmetic sequence where each term is not 0 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≠ 0 ∧ ∃ d, ∀ k, a (k + 1) = a k + d

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r ≠ 0, ∀ n, b (n + 1) = r * b n

theorem arithmetic_geometric_sequence_product (a b : ℕ → ℝ) :
  ArithmeticSequence a →
  GeometricSequence b →
  a 3 - (a 7)^2 / 2 + a 11 = 0 →
  b 7 = a 7 →
  b 1 * b 13 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l766_76684


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l766_76686

/-- Given a line l₁: 3x - 6y = 9 and a point P(-2, 4), 
    prove that the line l₂: y = -2x is perpendicular to l₁ and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 6 * y = 9
  let l₂ : ℝ → ℝ := λ x ↦ -2 * x
  let P : ℝ × ℝ := (-2, 4)
  (∀ x y, l₁ x y ↔ y = 1/2 * x - 3/2) ∧  -- l₁ in slope-intercept form
  (l₂ P.1 = P.2) ∧                      -- l₂ passes through P
  ((-2) * (1/2) = -1)                   -- l₁ and l₂ are perpendicular
  :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l766_76686
