import Mathlib

namespace NUMINAMATH_CALUDE_bethany_current_age_l956_95635

def bethany_age_problem (bethany_age : ℕ) (sister_age : ℕ) : Prop :=
  (bethany_age - 3 = 2 * (sister_age - 3)) ∧ (sister_age + 5 = 16)

theorem bethany_current_age :
  ∃ (bethany_age : ℕ) (sister_age : ℕ), bethany_age_problem bethany_age sister_age ∧ bethany_age = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_bethany_current_age_l956_95635


namespace NUMINAMATH_CALUDE_unique_root_of_sqrt_equation_l956_95666

theorem unique_root_of_sqrt_equation :
  ∃! x : ℝ, x + 9 ≥ 0 ∧ x - 2 ≥ 0 ∧ Real.sqrt (x + 9) - Real.sqrt (x - 2) = 3 :=
by
  -- The unique solution is x = 19/9
  use 19/9
  sorry

end NUMINAMATH_CALUDE_unique_root_of_sqrt_equation_l956_95666


namespace NUMINAMATH_CALUDE_vanilla_cookie_price_l956_95623

theorem vanilla_cookie_price 
  (chocolate_count : ℕ) 
  (vanilla_count : ℕ) 
  (chocolate_price : ℚ) 
  (total_revenue : ℚ) 
  (h1 : chocolate_count = 220)
  (h2 : vanilla_count = 70)
  (h3 : chocolate_price = 1)
  (h4 : total_revenue = 360) :
  ∃ (vanilla_price : ℚ), 
    vanilla_price = 2 ∧ 
    chocolate_count * chocolate_price + vanilla_count * vanilla_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_vanilla_cookie_price_l956_95623


namespace NUMINAMATH_CALUDE_travel_time_proof_l956_95650

def speed1 : ℝ := 6
def speed2 : ℝ := 12
def speed3 : ℝ := 18
def total_distance : ℝ := 1.8 -- 1800 meters converted to kilometers

theorem travel_time_proof :
  let d := total_distance / 3
  let time1 := d / speed1
  let time2 := d / speed2
  let time3 := d / speed3
  let total_time := (time1 + time2 + time3) * 60
  total_time = 11 := by
sorry

end NUMINAMATH_CALUDE_travel_time_proof_l956_95650


namespace NUMINAMATH_CALUDE_bridge_length_l956_95630

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 265 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l956_95630


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l956_95663

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℕ+, x ≤ 3 ↔ (x : ℝ)^4 / (x : ℝ)^2 < 15 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l956_95663


namespace NUMINAMATH_CALUDE_blueberry_muffin_percentage_is_fifty_percent_l956_95654

/-- Calculates the percentage of blueberry muffins given the number of blueberry cartons,
    blueberries per carton, blueberries per muffin, and number of cinnamon muffins. -/
def blueberry_muffin_percentage (
  cartons : ℕ
  ) (blueberries_per_carton : ℕ
  ) (blueberries_per_muffin : ℕ
  ) (cinnamon_muffins : ℕ
  ) : ℚ :=
  let total_blueberries := cartons * blueberries_per_carton
  let blueberry_muffins := total_blueberries / blueberries_per_muffin
  let total_muffins := blueberry_muffins + cinnamon_muffins
  (blueberry_muffins : ℚ) / (total_muffins : ℚ) * 100

/-- Proves that given 3 cartons of 200 blueberries, making muffins with 10 blueberries each,
    and 60 additional cinnamon muffins, the percentage of blueberry muffins is 50% of the total muffins. -/
theorem blueberry_muffin_percentage_is_fifty_percent :
  blueberry_muffin_percentage 3 200 10 60 = 50 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_muffin_percentage_is_fifty_percent_l956_95654


namespace NUMINAMATH_CALUDE_division_and_multiplication_l956_95637

theorem division_and_multiplication (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (result : ℕ) : 
  dividend = 24 → 
  divisor = 3 → 
  dividend = divisor * quotient → 
  result = quotient * 5 → 
  quotient = 8 ∧ result = 40 := by
sorry

end NUMINAMATH_CALUDE_division_and_multiplication_l956_95637


namespace NUMINAMATH_CALUDE_finite_game_has_winning_strategy_l956_95662

/-- Represents a two-player game with finite choices and finite length -/
structure FiniteGame where
  /-- The maximum number of moves before the game ends -/
  max_moves : ℕ
  /-- The number of possible choices for each move -/
  num_choices : ℕ
  /-- Predicate to check if the game has ended -/
  is_game_over : (List ℕ) → Bool
  /-- Predicate to determine the winner (true for player A, false for player B) -/
  winner : (List ℕ) → Bool

/-- Definition of a winning strategy for a player -/
def has_winning_strategy (game : FiniteGame) (player : Bool) : Prop :=
  ∃ (strategy : List ℕ → ℕ),
    ∀ (game_state : List ℕ),
      (game_state.length < game.max_moves) →
      (game.is_game_over game_state = false) →
      (game_state.length % 2 = if player then 0 else 1) →
      (strategy game_state ≤ game.num_choices) ∧
      (∃ (final_state : List ℕ),
        final_state.length ≤ game.max_moves ∧
        game.is_game_over final_state = true ∧
        game.winner final_state = player)

/-- Theorem: In a finite two-player game, one player must have a winning strategy -/
theorem finite_game_has_winning_strategy (game : FiniteGame) :
  has_winning_strategy game true ∨ has_winning_strategy game false :=
sorry

end NUMINAMATH_CALUDE_finite_game_has_winning_strategy_l956_95662


namespace NUMINAMATH_CALUDE_special_triangle_sides_l956_95658

/-- Represents a triangle with known height, base, and sum of two sides --/
structure SpecialTriangle where
  height : ℝ
  base : ℝ
  sum_of_sides : ℝ

/-- The two unknown sides of the triangle --/
structure TriangleSides where
  side1 : ℝ
  side2 : ℝ

/-- Theorem stating that for a triangle with height 24, base 28, and sum of two sides 56,
    the lengths of these two sides are 26 and 30 --/
theorem special_triangle_sides (t : SpecialTriangle) 
    (h1 : t.height = 24)
    (h2 : t.base = 28)
    (h3 : t.sum_of_sides = 56) :
  ∃ (s : TriangleSides), s.side1 = 26 ∧ s.side2 = 30 ∧ s.side1 + s.side2 = t.sum_of_sides :=
by
  sorry


end NUMINAMATH_CALUDE_special_triangle_sides_l956_95658


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_expression_l956_95682

theorem smallest_prime_dividing_expression : 
  ∃ (a : ℕ), a > 1 ∧ 179 ∣ (a^89 - 1) / (a - 1) ∧
  ∀ (p : ℕ), p > 100 → p < 179 → Prime p → 
    ¬(∃ (b : ℕ), b > 1 ∧ p ∣ (b^89 - 1) / (b - 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_expression_l956_95682


namespace NUMINAMATH_CALUDE_area_ratio_when_diameter_tripled_l956_95669

/-- The ratio of areas when a circle's diameter is tripled -/
theorem area_ratio_when_diameter_tripled (d : ℝ) (h : d > 0) :
  let r := d / 2
  let new_r := 3 * r
  (π * new_r ^ 2) / (π * r ^ 2) = 9 := by
sorry


end NUMINAMATH_CALUDE_area_ratio_when_diameter_tripled_l956_95669


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l956_95671

theorem hyperbola_foci_distance (x y : ℝ) :
  x^2 / 25 - y^2 / 9 = 1 →
  ∃ (f₁ f₂ : ℝ × ℝ), (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (2 * Real.sqrt 34)^2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l956_95671


namespace NUMINAMATH_CALUDE_joseph_running_distance_l956_95675

/-- Calculates the daily running distance given the total distance and number of days -/
def dailyDistance (totalDistance : ℕ) (days : ℕ) : ℕ :=
  totalDistance / days

theorem joseph_running_distance :
  let totalDistance : ℕ := 2700
  let days : ℕ := 3
  dailyDistance totalDistance days = 900 := by
  sorry

end NUMINAMATH_CALUDE_joseph_running_distance_l956_95675


namespace NUMINAMATH_CALUDE_twelve_factorial_base_nine_zeroes_l956_95621

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ := sorry

/-- 12! ends with 2 zeroes when written in base 9 -/
theorem twelve_factorial_base_nine_zeroes : trailingZeroes 12 9 = 2 := by sorry

end NUMINAMATH_CALUDE_twelve_factorial_base_nine_zeroes_l956_95621


namespace NUMINAMATH_CALUDE_arccos_one_half_l956_95629

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_l956_95629


namespace NUMINAMATH_CALUDE_number_calculation_l956_95647

theorem number_calculation (x : ℚ) : (30 / 100 * x = 25 / 100 * 50) → x = 125 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l956_95647


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l956_95698

/-- Represents the price of a muffin -/
def muffin_price : ℝ := sorry

/-- Represents the price of a banana -/
def banana_price : ℝ := sorry

/-- Jenny's total spending -/
def jenny_spending : ℝ := 5 * muffin_price + 4 * banana_price

/-- Michael's total spending -/
def michael_spending : ℝ := 3 * muffin_price + 20 * banana_price

/-- Theorem stating the price ratio of muffin to banana -/
theorem muffin_banana_price_ratio :
  michael_spending = 3 * jenny_spending →
  muffin_price / banana_price = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l956_95698


namespace NUMINAMATH_CALUDE_min_triangular_faces_l956_95659

/-- Represents a convex polyhedron --/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  non_triangular_faces : ℕ
  euler : faces + vertices = edges + 2
  more_faces : faces > vertices
  face_sum : faces = triangular_faces + non_triangular_faces
  edge_inequality : edges ≥ (3 * triangular_faces + 4 * non_triangular_faces) / 2

/-- The minimum number of triangular faces in a convex polyhedron with more faces than vertices is 6 --/
theorem min_triangular_faces (p : ConvexPolyhedron) : p.triangular_faces ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_triangular_faces_l956_95659


namespace NUMINAMATH_CALUDE_right_triangle_max_sin_product_l956_95617

theorem right_triangle_max_sin_product (A B C : Real) : 
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ -- Angles are non-negative
  A + B + C = π ∧ -- Sum of angles in a triangle
  C = π / 2 → -- Right angle condition
  ∀ (x y : Real), 0 ≤ x ∧ 0 ≤ y ∧ x + y = π / 2 → 
    Real.sin x * Real.sin y ≤ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_max_sin_product_l956_95617


namespace NUMINAMATH_CALUDE_wednesday_sales_l956_95641

/-- Represents the number of crates of eggs sold on each day of the week -/
structure EggSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Calculates the total number of crates sold over 4 days -/
def total_sales (sales : EggSales) : ℕ :=
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday

/-- Theorem stating the number of crates sold on Wednesday -/
theorem wednesday_sales (sales : EggSales) 
  (h1 : sales.monday = 5)
  (h2 : sales.tuesday = 2 * sales.monday)
  (h3 : sales.thursday = sales.tuesday / 2)
  (h4 : total_sales sales = 28) :
  sales.wednesday = 8 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_sales_l956_95641


namespace NUMINAMATH_CALUDE_pyramid_solution_l956_95686

/-- Represents a pyramid of numbers --/
structure Pyramid :=
  (bottom_row : List ℝ)
  (is_valid : bottom_row.length = 4)

/-- Checks if a pyramid satisfies the given conditions --/
def satisfies_conditions (p : Pyramid) : Prop :=
  ∃ x : ℝ,
    p.bottom_row = [13, x, 11, 2*x] ∧
    (13 + x) + (11 + 2*x) = 42

/-- The main theorem to prove --/
theorem pyramid_solution {p : Pyramid} (h : satisfies_conditions p) :
  ∃ x : ℝ, x = 6 ∧ p.bottom_row = [13, x, 11, 2*x] := by
  sorry

end NUMINAMATH_CALUDE_pyramid_solution_l956_95686


namespace NUMINAMATH_CALUDE_no_real_solutions_l956_95668

theorem no_real_solutions :
  ¬∃ x : ℝ, (3 * x^2) / (x - 2) - (5 * x + 4) / 4 + (10 - 9 * x) / (x - 2) + 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l956_95668


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l956_95684

theorem triangle_angle_calculation (D E F : ℝ) : 
  D = 90 →
  E = 2 * F + 15 →
  D + E + F = 180 →
  F = 25 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l956_95684


namespace NUMINAMATH_CALUDE_beads_per_necklace_l956_95678

theorem beads_per_necklace (total_beads : ℕ) (num_necklaces : ℕ) 
  (h1 : total_beads = 18) (h2 : num_necklaces = 6) :
  total_beads / num_necklaces = 3 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l956_95678


namespace NUMINAMATH_CALUDE_cookie_count_l956_95694

theorem cookie_count (bags : ℕ) (cookies_per_bag : ℕ) (total_cookies : ℕ) : 
  bags = 37 → cookies_per_bag = 19 → total_cookies = bags * cookies_per_bag → total_cookies = 703 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l956_95694


namespace NUMINAMATH_CALUDE_sqrt_real_implies_x_geq_8_l956_95622

theorem sqrt_real_implies_x_geq_8 (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 8) → x ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_real_implies_x_geq_8_l956_95622


namespace NUMINAMATH_CALUDE_square_equal_implies_abs_equal_l956_95667

theorem square_equal_implies_abs_equal (a b : ℝ) : a^2 = b^2 → |a| = |b| := by
  sorry

end NUMINAMATH_CALUDE_square_equal_implies_abs_equal_l956_95667


namespace NUMINAMATH_CALUDE_cos_alpha_value_l956_95606

theorem cos_alpha_value (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (-(2 / tanα) = 8/3) →  -- slope of the line 2x + (tanα)y + 1 = 0 is 8/3
  cosα = -4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l956_95606


namespace NUMINAMATH_CALUDE_complement_A_union_B_l956_95652

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_A_union_B :
  (𝒰 \ A) ∪ B = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l956_95652


namespace NUMINAMATH_CALUDE_gcd_consecutive_fib_46368_75025_l956_95653

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Theorem: The GCD of two consecutive Fibonacci numbers 46368 and 75025 is 1 -/
theorem gcd_consecutive_fib_46368_75025 :
  ∃ n : ℕ, fib n = 46368 ∧ fib (n + 1) = 75025 ∧ Nat.gcd 46368 75025 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_consecutive_fib_46368_75025_l956_95653


namespace NUMINAMATH_CALUDE_no_real_roots_of_composition_l956_95627

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
def f (a b c : ℝ) (ha : a ≠ 0) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- Theorem: If f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_real_roots_of_composition
  (a b c : ℝ) (ha : a ≠ 0)
  (h_no_roots : ∀ x : ℝ, f a b c ha x ≠ x) :
  ∀ x : ℝ, f a b c ha (f a b c ha x) ≠ x :=
sorry

end NUMINAMATH_CALUDE_no_real_roots_of_composition_l956_95627


namespace NUMINAMATH_CALUDE_probability_A_not_lose_l956_95607

-- Define the probabilities
def prob_A_win : ℝ := 0.3
def prob_draw : ℝ := 0.5

-- Define the probability of A not losing
def prob_A_not_lose : ℝ := prob_A_win + prob_draw

-- Theorem statement
theorem probability_A_not_lose : prob_A_not_lose = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_probability_A_not_lose_l956_95607


namespace NUMINAMATH_CALUDE_circle_properties_l956_95610

def is_tangent_to_x_axis (a b r : ℝ) : Prop :=
  r^2 = b^2

def center_on_line (a b : ℝ) : Prop :=
  3 * a - b = 0

def intersects_line_with_chord (a b r : ℝ) : Prop :=
  2 * r^2 = (a - b)^2 + 14

def circle_equation (a b r x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

theorem circle_properties (a b r : ℝ) :
  is_tangent_to_x_axis a b r →
  center_on_line a b →
  intersects_line_with_chord a b r →
  (circle_equation 1 3 3 a b ∨ circle_equation (-1) (-3) 3 a b) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l956_95610


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l956_95640

theorem smallest_constant_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hx₃ : x₃ > 0) (hx₄ : x₄ > 0) (hx₅ : x₅ > 0) :
  ∃ C : ℝ, C = 5^15 ∧ 
  (∀ D : ℝ, D < C → ∃ y₁ y₂ y₃ y₄ y₅ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ y₄ > 0 ∧ y₅ > 0 ∧
    D * (y₁^2005 + y₂^2005 + y₃^2005 + y₄^2005 + y₅^2005) < 
    y₁*y₂*y₃*y₄*y₅ * (y₁^125 + y₂^125 + y₃^125 + y₄^125 + y₅^125)^16) ∧
  C * (x₁^2005 + x₂^2005 + x₃^2005 + x₄^2005 + x₅^2005) ≥ 
  x₁*x₂*x₃*x₄*x₅ * (x₁^125 + x₂^125 + x₃^125 + x₄^125 + x₅^125)^16 := by
sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l956_95640


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l956_95624

/-- The quadratic equation (k-1)x^2 + 4x + 1 = 0 has two distinct real roots
    if and only if k < 5 and k ≠ 1 -/
theorem quadratic_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧
    (k - 1) * x^2 + 4 * x + 1 = 0 ∧
    (k - 1) * y^2 + 4 * y + 1 = 0) ↔
  (k < 5 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l956_95624


namespace NUMINAMATH_CALUDE_marble_problem_l956_95674

theorem marble_problem (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ) 
  (h1 : angela = a) 
  (h2 : brian = 3 * a) 
  (h3 : caden = 4 * brian) 
  (h4 : daryl = 6 * caden) 
  (h5 : angela + brian + caden + daryl = 186) : 
  a = 93 / 44 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l956_95674


namespace NUMINAMATH_CALUDE_right_triangle_and_perimeter_range_l956_95656

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the right triangle and its perimeter range -/
theorem right_triangle_and_perimeter_range (t : Triangle) (h : t.a * (Real.cos t.B + Real.cos t.C) = t.b + t.c) :
  t.A = π / 2 ∧ 
  (∀ (r : ℝ), r = 1 → 4 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 2 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_and_perimeter_range_l956_95656


namespace NUMINAMATH_CALUDE_smaller_integer_problem_l956_95692

theorem smaller_integer_problem (x y : ℤ) : 
  y = 2 * x → x + y = 96 → x = 32 := by
  sorry

end NUMINAMATH_CALUDE_smaller_integer_problem_l956_95692


namespace NUMINAMATH_CALUDE_first_player_can_ensure_nonzero_solution_l956_95665

/-- Represents a linear equation in three variables -/
structure LinearEquation where
  coeff_x : ℝ
  coeff_y : ℝ
  coeff_z : ℝ
  constant : ℝ

/-- Represents a system of three linear equations -/
structure LinearSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation
  eq3 : LinearEquation

/-- Represents a player's strategy for choosing coefficients -/
def Strategy := LinearSystem → LinearEquation → ℝ

/-- Checks if a given solution satisfies the linear system -/
def is_solution (system : LinearSystem) (x y z : ℝ) : Prop :=
  system.eq1.coeff_x * x + system.eq1.coeff_y * y + system.eq1.coeff_z * z = system.eq1.constant ∧
  system.eq2.coeff_x * x + system.eq2.coeff_y * y + system.eq2.coeff_z * z = system.eq2.constant ∧
  system.eq3.coeff_x * x + system.eq3.coeff_y * y + system.eq3.coeff_z * z = system.eq3.constant

/-- Represents the game where players choose coefficients -/
def play_game (player1_strategy : Strategy) (player2_strategy : Strategy) : LinearSystem :=
  sorry  -- Implementation of the game play

/-- The main theorem stating that the first player can ensure a nonzero solution -/
theorem first_player_can_ensure_nonzero_solution :
  ∃ (player1_strategy : Strategy),
    ∀ (player2_strategy : Strategy),
      ∃ (x y z : ℝ),
        (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
        is_solution (play_game player1_strategy player2_strategy) x y z :=
by sorry

end NUMINAMATH_CALUDE_first_player_can_ensure_nonzero_solution_l956_95665


namespace NUMINAMATH_CALUDE_max_consecutive_positive_terms_l956_95644

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = a (n - 1) + a (n + 2)

/-- The property that a sequence has k consecutive positive terms starting from index n -/
def HasConsecutivePositiveTerms (a : ℕ → ℝ) (n k : ℕ) : Prop :=
  ∀ i : ℕ, i ∈ Finset.range k → a (n + i) > 0

/-- The main theorem stating that the maximum number of consecutive positive terms is 5 -/
theorem max_consecutive_positive_terms
  (a : ℕ → ℝ) (h : RecurrenceSequence a) :
  (∃ n k : ℕ, k > 5 ∧ HasConsecutivePositiveTerms a n k) → False :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_positive_terms_l956_95644


namespace NUMINAMATH_CALUDE_zeros_in_quotient_l956_95634

/-- S_k represents the k-length sequence of twos in its decimal presentation -/
def S (k : ℕ) : ℕ := (2 * (10^k - 1)) / 9

/-- The quotient of S_30 divided by S_5 -/
def Q : ℕ := S 30 / S 5

/-- The number of zeros in the decimal representation of Q -/
def num_zeros (n : ℕ) : ℕ := sorry

theorem zeros_in_quotient : num_zeros Q = 20 := by sorry

end NUMINAMATH_CALUDE_zeros_in_quotient_l956_95634


namespace NUMINAMATH_CALUDE_rectangle_cannot_fit_in_square_l956_95608

/-- Proves that a rectangle with area 90 cm² and length-to-width ratio 5:3 cannot fit in a 100 cm² square -/
theorem rectangle_cannot_fit_in_square : ¬ ∃ (length width : ℝ),
  (length * width = 90) ∧ 
  (length / width = 5 / 3) ∧
  (length ≤ 10) ∧
  (width ≤ 10) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_cannot_fit_in_square_l956_95608


namespace NUMINAMATH_CALUDE_digit_placement_theorem_l956_95618

def number_of_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  Nat.choose n k * Nat.factorial m

theorem digit_placement_theorem :
  number_of_arrangements 6 2 4 = 360 :=
sorry

end NUMINAMATH_CALUDE_digit_placement_theorem_l956_95618


namespace NUMINAMATH_CALUDE_photo_selection_choices_l956_95646

-- Define the number of items to choose from
def n : ℕ := 10

-- Define the possible numbers of items to be chosen
def k₁ : ℕ := 5
def k₂ : ℕ := 6

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem photo_selection_choices : 
  combination n k₁ + combination n k₂ = 462 := by sorry

end NUMINAMATH_CALUDE_photo_selection_choices_l956_95646


namespace NUMINAMATH_CALUDE_max_plane_division_prove_max_plane_division_l956_95664

/-- The maximum number of parts the plane can be divided into by two specific parabolas and a line -/
theorem max_plane_division (b k : ℝ) : ℕ :=
  let parabola1 := fun x : ℝ => x^2 - b*x
  let parabola2 := fun x : ℝ => -x^2 + b*x
  let line := fun x : ℝ => k*x
  9

/-- Proof of the maximum number of plane divisions -/
theorem prove_max_plane_division (b k : ℝ) : 
  max_plane_division b k = 9 := by sorry

end NUMINAMATH_CALUDE_max_plane_division_prove_max_plane_division_l956_95664


namespace NUMINAMATH_CALUDE_f_properties_l956_95604

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x) / Real.log 2

theorem f_properties :
  (∀ x ≠ 0, f x = f (-x)) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l956_95604


namespace NUMINAMATH_CALUDE_right_triangle_area_l956_95616

/-- A right triangle with vertices A(0, 0), B(0, 5), and C(3, 0) has an area of 7.5 square units. -/
theorem right_triangle_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 5)
  let C : ℝ × ℝ := (3, 0)
  let triangle_area := (1/2) * 3 * 5
  triangle_area = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l956_95616


namespace NUMINAMATH_CALUDE_lines_parallel_l956_95628

/-- Two lines in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The property of two lines being parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept ≠ l2.intercept

theorem lines_parallel : 
  let l1 : Line := { slope := 2, intercept := 1 }
  let l2 : Line := { slope := 2, intercept := 5 }
  parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_l956_95628


namespace NUMINAMATH_CALUDE_will_baseball_card_pages_l956_95676

/-- Calculates the number of pages needed to arrange baseball cards. -/
def pages_needed (cards_per_page : ℕ) (cards_2020 : ℕ) (cards_2015_2019 : ℕ) (duplicates : ℕ) : ℕ :=
  let unique_2020 := cards_2020
  let unique_2015_2019 := cards_2015_2019 - duplicates
  let pages_2020 := (unique_2020 + cards_per_page - 1) / cards_per_page
  let pages_2015_2019 := (unique_2015_2019 + cards_per_page - 1) / cards_per_page
  pages_2020 + pages_2015_2019

/-- Theorem stating the number of pages needed for Will's baseball card arrangement. -/
theorem will_baseball_card_pages :
  pages_needed 3 8 10 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_will_baseball_card_pages_l956_95676


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l956_95690

theorem ratio_x_to_y (x y : ℝ) (h : (15 * x - 4 * y) / (18 * x - 3 * y) = 4 / 7) :
  x / y = 16 / 33 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l956_95690


namespace NUMINAMATH_CALUDE_no_solution_exists_l956_95642

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_solution_exists : ∀ n : ℕ, n * sum_of_digits n ≠ 100200300 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l956_95642


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l956_95681

theorem smallest_multiple_of_6_and_15 : 
  ∃ (a : ℕ), a > 0 ∧ 6 ∣ a ∧ 15 ∣ a ∧ ∀ (b : ℕ), b > 0 ∧ 6 ∣ b ∧ 15 ∣ b → a ≤ b :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l956_95681


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l956_95673

/-- Number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 60 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l956_95673


namespace NUMINAMATH_CALUDE_reciprocal_expression_l956_95696

theorem reciprocal_expression (a b c : ℝ) (h : a * b = 1) : a * b * c - (c - 2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_expression_l956_95696


namespace NUMINAMATH_CALUDE_power_of_three_mod_eight_l956_95638

theorem power_of_three_mod_eight : 3^1988 ≡ 1 [MOD 8] := by sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eight_l956_95638


namespace NUMINAMATH_CALUDE_a_squared_b_irrational_l956_95632

def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem a_squared_b_irrational 
  (a b : ℝ) 
  (h_a_rational : is_rational a) 
  (h_b_irrational : ¬ is_rational b) 
  (h_ab_rational : is_rational (a * b)) : 
  ¬ is_rational (a^2 * b) :=
sorry

end NUMINAMATH_CALUDE_a_squared_b_irrational_l956_95632


namespace NUMINAMATH_CALUDE_water_depth_is_60_l956_95679

/-- The depth of water given Ron's height -/
def water_depth (ron_height : ℝ) : ℝ := 5 * ron_height

/-- Ron's height in feet -/
def ron_height : ℝ := 12

/-- Theorem: The water depth is 60 feet -/
theorem water_depth_is_60 : water_depth ron_height = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_is_60_l956_95679


namespace NUMINAMATH_CALUDE_max_water_depth_l956_95612

/-- The maximum water depth during a swim, given the swimmer's height,
    the ratio of water depth to height, and the wave increase percentage. -/
theorem max_water_depth
  (height : ℝ)
  (depth_ratio : ℝ)
  (wave_increase : ℝ)
  (h1 : height = 6)
  (h2 : depth_ratio = 10)
  (h3 : wave_increase = 0.25)
  : height * depth_ratio * (1 + wave_increase) = 75 := by
  sorry

#check max_water_depth

end NUMINAMATH_CALUDE_max_water_depth_l956_95612


namespace NUMINAMATH_CALUDE_parabola_vertex_l956_95699

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 4*y + 3*x + 8 = 0

-- Define the vertex of a parabola
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x' y', eq x' y' → x' ≥ x

-- Theorem stating that (-4/3, -2) is the vertex of the parabola
theorem parabola_vertex :
  is_vertex (-4/3) (-2) parabola_equation :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l956_95699


namespace NUMINAMATH_CALUDE_charlie_apple_picking_l956_95680

theorem charlie_apple_picking (total bags_golden bags_cortland bags_macintosh : ℚ) : 
  total = 0.67 ∧ 
  bags_golden = 0.17 ∧ 
  bags_cortland = 0.33 ∧ 
  total = bags_golden + bags_macintosh + bags_cortland → 
  bags_macintosh = 0.17 := by
sorry

end NUMINAMATH_CALUDE_charlie_apple_picking_l956_95680


namespace NUMINAMATH_CALUDE_error_percentage_division_vs_multiplication_error_percentage_proof_l956_95625

theorem error_percentage_division_vs_multiplication : ℝ → Prop :=
  fun x => x ≠ 0 →
    let correct_result := 5 * x
    let incorrect_result := x / 10
    let error := correct_result - incorrect_result
    let percentage_error := (error / correct_result) * 100
    percentage_error = 98

-- The proof is omitted
theorem error_percentage_proof : ∀ x : ℝ, error_percentage_division_vs_multiplication x :=
sorry

end NUMINAMATH_CALUDE_error_percentage_division_vs_multiplication_error_percentage_proof_l956_95625


namespace NUMINAMATH_CALUDE_only_D_is_comprehensive_l956_95605

/-- Represents the type of survey --/
inductive SurveyType
  | Comprehensive
  | Sampling

/-- Represents the different survey options --/
inductive SurveyOption
  | A  -- Understanding the service life of a certain light bulb
  | B  -- Understanding whether a batch of cold drinks meets quality standards
  | C  -- Understanding the vision status of eighth-grade students nationwide
  | D  -- Understanding which month has the most births in a certain class

/-- Determines the appropriate survey type for a given option --/
def determineSurveyType (option : SurveyOption) : SurveyType :=
  match option with
  | SurveyOption.A => SurveyType.Sampling
  | SurveyOption.B => SurveyType.Sampling
  | SurveyOption.C => SurveyType.Sampling
  | SurveyOption.D => SurveyType.Comprehensive

/-- Theorem stating that only Option D is suitable for a comprehensive survey --/
theorem only_D_is_comprehensive :
  ∀ (option : SurveyOption),
    determineSurveyType option = SurveyType.Comprehensive ↔ option = SurveyOption.D :=
by sorry

#check only_D_is_comprehensive

end NUMINAMATH_CALUDE_only_D_is_comprehensive_l956_95605


namespace NUMINAMATH_CALUDE_or_false_sufficient_not_necessary_for_and_false_l956_95689

theorem or_false_sufficient_not_necessary_for_and_false (p q : Prop) :
  (¬(p ∨ q) → ¬p ∧ ¬q) ∧ ∃ (p q : Prop), (¬p ∧ ¬q) ∧ ¬¬(p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_or_false_sufficient_not_necessary_for_and_false_l956_95689


namespace NUMINAMATH_CALUDE_vector_addition_proof_l956_95677

theorem vector_addition_proof (a b : ℝ × ℝ) (h1 : a = (1, -2)) (h2 : b = (-2, 2)) :
  a + 2 • b = (-3, 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_proof_l956_95677


namespace NUMINAMATH_CALUDE_complex_real_condition_l956_95691

theorem complex_real_condition (a : ℝ) : 
  let z : ℂ := (a - 3 : ℂ) + (a^2 - 2*a - 3 : ℂ) * Complex.I
  (z.im = 0) → (a = 3 ∨ a = -1) := by
sorry

end NUMINAMATH_CALUDE_complex_real_condition_l956_95691


namespace NUMINAMATH_CALUDE_vector_magnitude_l956_95695

/-- Given vectors a and b in ℝ², if a is collinear with a + b, 
    then the magnitude of a - b is 2√5 -/
theorem vector_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![3, x]
  (∃ (k : ℝ), a = k • (a + b)) → ‖a - b‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l956_95695


namespace NUMINAMATH_CALUDE_park_available_spaces_l956_95611

/-- Calculates the number of available spaces in a park given the number of benches, 
    capacity per bench, and number of people currently sitting. -/
def available_spaces (num_benches : ℕ) (capacity_per_bench : ℕ) (people_sitting : ℕ) : ℕ :=
  num_benches * capacity_per_bench - people_sitting

/-- Theorem stating that in a park with 50 benches, each with a capacity of 4 people, 
    and 80 people currently sitting, there are 120 available spaces. -/
theorem park_available_spaces : 
  available_spaces 50 4 80 = 120 := by
  sorry

end NUMINAMATH_CALUDE_park_available_spaces_l956_95611


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l956_95648

theorem smallest_number_satisfying_conditions : 
  ∃ N : ℕ, 
    N > 0 ∧ 
    N % 4 = 0 ∧ 
    (N + 9) % 2 = 1 ∧ 
    (∀ M : ℕ, M > 0 → M % 4 = 0 → (M + 9) % 2 = 1 → M ≥ N) ∧
    N = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l956_95648


namespace NUMINAMATH_CALUDE_greene_family_spending_l956_95688

theorem greene_family_spending (admission_cost food_cost total_cost : ℕ) : 
  admission_cost = 45 →
  food_cost = admission_cost - 13 →
  total_cost = admission_cost + food_cost →
  total_cost = 77 := by
sorry

end NUMINAMATH_CALUDE_greene_family_spending_l956_95688


namespace NUMINAMATH_CALUDE_consecutive_numbers_proof_l956_95619

theorem consecutive_numbers_proof (x y z : ℤ) : 
  (x = y + 1) →  -- x, y are consecutive
  (y = z + 1) →  -- y, z are consecutive
  (x > y) →      -- x > y
  (y > z) →      -- y > z
  (2*x + 3*y + 3*z = 5*y + 11) →  -- given equation
  (z = 3) →      -- given value of z
  (3*y = 12) :=  -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_proof_l956_95619


namespace NUMINAMATH_CALUDE_height_side_relation_l956_95683

/-- Triangle with sides and heights -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  H_A : ℝ
  H_B : ℝ
  H_C : ℝ

/-- Theorem: In a triangle, if one height is greater than another, the side opposite to the greater height is shorter than the side opposite to the smaller height -/
theorem height_side_relation (t : Triangle) :
  t.H_A > t.H_B → t.B < t.A :=
by sorry

end NUMINAMATH_CALUDE_height_side_relation_l956_95683


namespace NUMINAMATH_CALUDE_find_x_value_l956_95603

def A (x : ℕ) : Set ℕ := {1, 4, x}
def B (x : ℕ) : Set ℕ := {1, x^2}

theorem find_x_value (x : ℕ) (h : A x ∪ B x = A x) : x = 0 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l956_95603


namespace NUMINAMATH_CALUDE_symmetry_probability_l956_95649

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- The size of the square grid -/
def gridSize : Nat := 11

/-- The center point of the grid -/
def centerPoint : GridPoint := ⟨gridSize / 2 + 1, gridSize / 2 + 1⟩

/-- The total number of points in the grid -/
def totalPoints : Nat := gridSize * gridSize

/-- The number of points excluding the center point -/
def remainingPoints : Nat := totalPoints - 1

/-- Checks if a point forms a line of symmetry with the center point -/
def isSymmetryPoint (p : GridPoint) : Bool :=
  p.x = centerPoint.x ∨ 
  p.y = centerPoint.y ∨ 
  p.x - centerPoint.x = p.y - centerPoint.y ∨
  p.x - centerPoint.x = centerPoint.y - p.y

/-- The number of points that form lines of symmetry -/
def symmetryPoints : Nat := 4 * (gridSize - 1)

/-- The probability theorem -/
theorem symmetry_probability : 
  (symmetryPoints : ℚ) / remainingPoints = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_symmetry_probability_l956_95649


namespace NUMINAMATH_CALUDE_ratio_equivalence_l956_95631

theorem ratio_equivalence (x y m n : ℚ) 
  (h : (5 * x + 7 * y) / (3 * x + 2 * y) = m / n) :
  (13 * x + 16 * y) / (2 * x + 5 * y) = (2 * m + n) / (m - n) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalence_l956_95631


namespace NUMINAMATH_CALUDE_square_sum_geq_product_l956_95636

theorem square_sum_geq_product {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : 2 * (a + b + c + d) ≥ a * b * c * d) :
  a^2 + b^2 + c^2 + d^2 ≥ a * b * c * d := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_l956_95636


namespace NUMINAMATH_CALUDE_max_product_difference_l956_95600

theorem max_product_difference (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : 0 ≤ a₁ ∧ a₁ ≤ 1) (h₂ : 0 ≤ a₂ ∧ a₂ ≤ 1) (h₃ : 0 ≤ a₃ ∧ a₃ ≤ 1) 
  (h₄ : 0 ≤ a₄ ∧ a₄ ≤ 1) (h₅ : 0 ≤ a₅ ∧ a₅ ≤ 1) : 
  |a₁ - a₂| * |a₁ - a₃| * |a₁ - a₄| * |a₁ - a₅| * 
  |a₂ - a₃| * |a₂ - a₄| * |a₂ - a₅| * 
  |a₃ - a₄| * |a₃ - a₅| * 
  |a₄ - a₅| ≤ 3 * Real.sqrt 21 / 38416 := by
  sorry

end NUMINAMATH_CALUDE_max_product_difference_l956_95600


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l956_95614

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 6 * x + c = 0) →  -- exactly one solution
  (a + c = 7) →                      -- sum condition
  (a < c) →                          -- order condition
  (a = (7 - Real.sqrt 13) / 2 ∧ c = (7 + Real.sqrt 13) / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l956_95614


namespace NUMINAMATH_CALUDE_bucket_capacity_l956_95657

theorem bucket_capacity (x : ℝ) 
  (h1 : 12 * x = 84 * 7) : x = 49 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_l956_95657


namespace NUMINAMATH_CALUDE_complex_addition_result_l956_95645

theorem complex_addition_result : ∃ z : ℂ, (5 - 3*I + z = -4 + 9*I) ∧ (z = -9 + 12*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_addition_result_l956_95645


namespace NUMINAMATH_CALUDE_problem_solution_l956_95661

theorem problem_solution (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x*y - x - y = -7) : 
  x^2*y + x*y^2 = 196/9 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l956_95661


namespace NUMINAMATH_CALUDE_meat_for_35_tacos_l956_95602

/-- The amount of meat (in pounds) needed to make a given number of tacos, 
    given that 4 pounds of meat make 10 tacos -/
def meat_needed (tacos : ℕ) : ℚ :=
  (4 : ℚ) * tacos / 10

theorem meat_for_35_tacos : meat_needed 35 = 14 := by
  sorry

end NUMINAMATH_CALUDE_meat_for_35_tacos_l956_95602


namespace NUMINAMATH_CALUDE_penalty_kicks_count_l956_95633

theorem penalty_kicks_count (total_players : ℕ) (goalies : ℕ) 
  (h1 : total_players = 25) 
  (h2 : goalies = 4) 
  (h3 : goalies ≤ total_players) : 
  goalies * (total_players - 1) = 96 := by
  sorry

end NUMINAMATH_CALUDE_penalty_kicks_count_l956_95633


namespace NUMINAMATH_CALUDE_root_implies_a_value_l956_95615

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x - 48

-- State the theorem
theorem root_implies_a_value (a b : ℚ) :
  f a b (-2 - 3 * Real.sqrt 3) = 0 → a = 44 / 23 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_value_l956_95615


namespace NUMINAMATH_CALUDE_autumn_pencils_l956_95620

def pencil_count (initial misplaced broken found bought : ℕ) : ℕ :=
  initial - misplaced - broken + found + bought

theorem autumn_pencils : pencil_count 20 7 3 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_autumn_pencils_l956_95620


namespace NUMINAMATH_CALUDE_base_eight_to_ten_l956_95687

theorem base_eight_to_ten : 
  (1 * 8^3 + 7 * 8^2 + 2 * 8^1 + 4 * 8^0 : ℕ) = 980 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_to_ten_l956_95687


namespace NUMINAMATH_CALUDE_parallel_lines_same_black_cells_l956_95672

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a line in the grid (horizontal, vertical, or diagonal) -/
inductive Line
  | Horizontal : Nat → Line
  | Vertical : Nat → Line
  | LeftDiagonal : Nat → Line
  | RightDiagonal : Nat → Line

/-- The grid configuration -/
structure GridConfig where
  n : Nat
  blackCells : Set Cell

/-- Two lines are parallel -/
def areLinesParallel (l1 l2 : Line) : Prop :=
  match l1, l2 with
  | Line.Horizontal _, Line.Horizontal _ => true
  | Line.Vertical _, Line.Vertical _ => true
  | Line.LeftDiagonal _, Line.LeftDiagonal _ => true
  | Line.RightDiagonal _, Line.RightDiagonal _ => true
  | _, _ => false

/-- Count black cells in a line -/
def countBlackCells (g : GridConfig) (l : Line) : Nat :=
  sorry

/-- Main theorem -/
theorem parallel_lines_same_black_cells 
  (g : GridConfig) 
  (h : g.n ≥ 3) :
  ∃ l1 l2 : Line, 
    areLinesParallel l1 l2 ∧ 
    l1 ≠ l2 ∧ 
    countBlackCells g l1 = countBlackCells g l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_same_black_cells_l956_95672


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l956_95685

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a7 : a 7 = Real.sqrt 2 / 2) :
  (1 / a 3 + 2 / a 11) ≥ 4 ∧
  ∃ x : ℝ, (1 / a 3 + 2 / a 11) = x → x ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l956_95685


namespace NUMINAMATH_CALUDE_toothpicks_43_10_l956_95693

/-- The number of toothpicks used in a 1 × 10 grid -/
def toothpicks_1_10 : ℕ := 31

/-- The number of toothpicks used in an n × 10 grid -/
def toothpicks_n_10 (n : ℕ) : ℕ := 21 * n + 10

/-- Theorem: The number of toothpicks in a 43 × 10 grid is 913 -/
theorem toothpicks_43_10 :
  toothpicks_n_10 43 = 913 :=
sorry

end NUMINAMATH_CALUDE_toothpicks_43_10_l956_95693


namespace NUMINAMATH_CALUDE_chocolate_sales_l956_95643

theorem chocolate_sales (C S : ℝ) (n : ℕ) 
  (h1 : 81 * C = n * S)  -- Cost price of 81 chocolates equals selling price of n chocolates
  (h2 : S = 1.8 * C)     -- Selling price is 1.8 times the cost price (derived from 80% gain)
  : n = 45 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_sales_l956_95643


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l956_95670

-- Problem 1
theorem problem_1 : -20 - (-8) + (-4) = -16 := by sorry

-- Problem 2
theorem problem_2 : -1^3 * (-2)^2 / (4/3) + |5-8| = 0 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l956_95670


namespace NUMINAMATH_CALUDE_percent_of_x_is_z_l956_95613

theorem percent_of_x_is_z (x y z w : ℝ) 
  (h1 : 0.45 * z = 0.75 * y)
  (h2 : y = 0.8 * w)
  (h3 : w = 0.9 * x) :
  z = 0.54 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_is_z_l956_95613


namespace NUMINAMATH_CALUDE_polynomial_real_root_l956_95609

theorem polynomial_real_root (a : ℝ) : ∃ x : ℝ, x^5 + a*x^4 - x^3 + a*x^2 - x + a = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l956_95609


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l956_95651

theorem chess_tournament_participants :
  ∃ n : ℕ,
    n > 0 ∧
    n * (n - 1) / 2 = 136 ∧
    (∀ m : ℕ, m > 0 ∧ m * (m - 1) / 2 = 136 → m = n) ∧
    n = 17 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l956_95651


namespace NUMINAMATH_CALUDE_fgh_supermarkets_count_l956_95697

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 47

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 10

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 84

theorem fgh_supermarkets_count :
  us_supermarkets = 47 ∧
  us_supermarkets + canada_supermarkets = total_supermarkets ∧
  us_supermarkets = canada_supermarkets + 10 :=
by sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_count_l956_95697


namespace NUMINAMATH_CALUDE_emily_lost_lives_l956_95626

theorem emily_lost_lives (initial_lives : ℕ) (lives_gained : ℕ) (final_lives : ℕ) : 
  initial_lives = 42 → lives_gained = 24 → final_lives = 41 → 
  ∃ (lives_lost : ℕ), initial_lives - lives_lost + lives_gained = final_lives ∧ lives_lost = 25 := by
sorry

end NUMINAMATH_CALUDE_emily_lost_lives_l956_95626


namespace NUMINAMATH_CALUDE_farm_animals_l956_95655

theorem farm_animals (total_legs : ℕ) (total_animals : ℕ) (duck_legs : ℕ) (cow_legs : ℕ) :
  total_legs = 42 →
  total_animals = 15 →
  duck_legs = 2 →
  cow_legs = 4 →
  ∃ (num_ducks : ℕ) (num_cows : ℕ),
    num_ducks + num_cows = total_animals ∧
    num_ducks * duck_legs + num_cows * cow_legs = total_legs ∧
    num_cows = 6 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_l956_95655


namespace NUMINAMATH_CALUDE_otimes_example_otimes_sum_property_l956_95639

-- Define the custom operation
def otimes (a b : ℝ) : ℝ := a * (1 - b)

-- Theorem 1
theorem otimes_example : otimes 2 (-2) = 6 := by sorry

-- Theorem 2
theorem otimes_sum_property (a b : ℝ) (h : a + b = 0) : 
  (otimes a a) + (otimes b b) = 2 * a * b := by sorry

end NUMINAMATH_CALUDE_otimes_example_otimes_sum_property_l956_95639


namespace NUMINAMATH_CALUDE_parabola_theorem_l956_95660

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  eq : (a * x^2 : ℝ) + (b * x * y : ℝ) + (c * y^2 : ℝ) + (d * x : ℝ) + (e * y : ℝ) + (f : ℝ) = 0

/-- The parabola passes through the point (2,6) -/
def passes_through (p : Parabola) : Prop :=
  (p.a * 2^2 : ℝ) + (p.b * 2 * 6 : ℝ) + (p.c * 6^2 : ℝ) + (p.d * 2 : ℝ) + (p.e * 6 : ℝ) + (p.f : ℝ) = 0

/-- The y-coordinate of the focus is 4 -/
def focus_y_coord (p : Parabola) : Prop :=
  ∃ x : ℝ, (p.a * x^2 : ℝ) + (p.b * x * 4 : ℝ) + (p.c * 4^2 : ℝ) + (p.d * x : ℝ) + (p.e * 4 : ℝ) + (p.f : ℝ) = 0

/-- The axis of symmetry is parallel to the x-axis -/
def axis_parallel_x (p : Parabola) : Prop :=
  p.b = 0 ∧ p.c ≠ 0

/-- The vertex lies on the y-axis -/
def vertex_on_y_axis (p : Parabola) : Prop :=
  ∃ y : ℝ, (p.c * y^2 : ℝ) + (p.e * y : ℝ) + (p.f : ℝ) = 0

/-- The coefficients satisfy the required conditions -/
def coeff_conditions (p : Parabola) : Prop :=
  p.c > 0 ∧ Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs p.a) (Int.natAbs p.b)) (Int.natAbs p.c)) (Int.natAbs p.d)) (Int.natAbs p.e)) (Int.natAbs p.f) = 1

/-- The main theorem stating that the given equation represents a parabola satisfying all conditions -/
theorem parabola_theorem : ∃ p : Parabola, 
  p.a = 0 ∧ p.b = 0 ∧ p.c = 1 ∧ p.d = -2 ∧ p.e = -8 ∧ p.f = 16 ∧
  passes_through p ∧
  focus_y_coord p ∧
  axis_parallel_x p ∧
  vertex_on_y_axis p ∧
  coeff_conditions p :=
sorry

end NUMINAMATH_CALUDE_parabola_theorem_l956_95660


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l956_95601

theorem greatest_integer_radius (A : ℝ) (h : A < 200 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi ≤ A ∧ ∀ (s : ℕ), s * s * Real.pi ≤ A → s ≤ r :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l956_95601
