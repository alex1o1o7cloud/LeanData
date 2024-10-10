import Mathlib

namespace quadrilateral_area_is_22_5_l2805_280555

/-- The area of a quadrilateral with vertices at (1, 2), (1, -1), (4, -1), and (7, 8) -/
def quadrilateral_area : ℝ :=
  let A := (1, 2)
  let B := (1, -1)
  let C := (4, -1)
  let D := (7, 8)
  -- Area calculation goes here
  0 -- Placeholder

theorem quadrilateral_area_is_22_5 : quadrilateral_area = 22.5 := by
  sorry

end quadrilateral_area_is_22_5_l2805_280555


namespace complex_equation_solution_l2805_280540

theorem complex_equation_solution (z : ℂ) : 
  (1 : ℂ) + Complex.I * Real.sqrt 3 = z * ((1 : ℂ) - Complex.I * Real.sqrt 3) → 
  z = -1/2 + Complex.I * (Real.sqrt 3 / 2) := by
sorry

end complex_equation_solution_l2805_280540


namespace perfect_square_binomial_l2805_280500

theorem perfect_square_binomial : ∃ a b : ℝ, ∀ x : ℝ, x^2 - 20*x + 100 = (a*x + b)^2 := by
  sorry

end perfect_square_binomial_l2805_280500


namespace pencil_sharpening_l2805_280538

/-- Given a pencil that is shortened from 22 inches to 18 inches over two days
    with equal amounts sharpened each day, the amount sharpened per day is 2 inches. -/
theorem pencil_sharpening (initial_length : ℝ) (final_length : ℝ) (days : ℕ)
  (h1 : initial_length = 22)
  (h2 : final_length = 18)
  (h3 : days = 2) :
  (initial_length - final_length) / days = 2 := by
  sorry

end pencil_sharpening_l2805_280538


namespace first_day_exceeding_500_l2805_280564

def algae_population (n : ℕ) : ℕ := 5 * 3^n

theorem first_day_exceeding_500 :
  (∀ k : ℕ, k < 5 → algae_population k ≤ 500) ∧
  algae_population 5 > 500 :=
sorry

end first_day_exceeding_500_l2805_280564


namespace expression_equality_l2805_280559

theorem expression_equality (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 - 1) / x) * ((y^2 - 1) / y) - ((x^2 - 1) / y) * ((y^2 - 1) / x) = 0 := by
  sorry

end expression_equality_l2805_280559


namespace profit_percent_calculation_l2805_280528

theorem profit_percent_calculation (selling_price cost_price profit : ℝ) :
  cost_price = 0.75 * selling_price →
  profit = selling_price - cost_price →
  (profit / cost_price) * 100 = 33.33333333333333 := by
  sorry

end profit_percent_calculation_l2805_280528


namespace sin_sum_of_complex_exponentials_l2805_280567

theorem sin_sum_of_complex_exponentials
  (γ δ : ℝ)
  (h1 : Complex.exp (γ * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I)
  (h2 : Complex.exp (δ * Complex.I) = -(5 / 13 : ℂ) - (12 / 13 : ℂ) * Complex.I) :
  Real.sin (γ + δ) = -(63 / 65) :=
by sorry

end sin_sum_of_complex_exponentials_l2805_280567


namespace cylinder_lateral_surface_area_l2805_280523

/-- The lateral surface area of a cylinder with base radius 1 and slant height 2 is 4π. -/
theorem cylinder_lateral_surface_area : 
  let r : ℝ := 1  -- radius of the base
  let s : ℝ := 2  -- slant height
  2 * π * r * s = 4 * π := by sorry

end cylinder_lateral_surface_area_l2805_280523


namespace fraction_value_l2805_280583

theorem fraction_value (x y : ℝ) (h1 : 4 < (2*x - 3*y) / (2*x + 3*y)) 
  (h2 : (2*x - 3*y) / (2*x + 3*y) < 8) (h3 : ∃ (n : ℤ), x/y = n) : 
  x/y = -2 := by
sorry

end fraction_value_l2805_280583


namespace tetrahedron_surface_area_bound_l2805_280543

/-- Tetrahedron with given edge lengths and surface area -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  S : ℝ
  edge_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f
  surface_positive : 0 < S

/-- The surface area of a tetrahedron is bounded by a function of its edge lengths -/
theorem tetrahedron_surface_area_bound (t : Tetrahedron) :
    t.S ≤ (Real.sqrt 3 / 6) * (t.a^2 + t.b^2 + t.c^2 + t.d^2 + t.e^2 + t.f^2) := by
  sorry

end tetrahedron_surface_area_bound_l2805_280543


namespace hash_difference_l2805_280591

def hash (x y : ℝ) : ℝ := x * y - 3 * x

theorem hash_difference : (hash 6 4) - (hash 4 6) = -6 := by
  sorry

end hash_difference_l2805_280591


namespace unique_integer_divisible_by_18_with_sqrt_between_26_and_26_2_l2805_280542

theorem unique_integer_divisible_by_18_with_sqrt_between_26_and_26_2 :
  ∃! (N : ℕ), 
    N > 0 ∧ 
    N % 18 = 0 ∧ 
    (26 : ℝ) < Real.sqrt N ∧ 
    Real.sqrt N < 26.2 ∧ 
    N = 684 := by
  sorry

end unique_integer_divisible_by_18_with_sqrt_between_26_and_26_2_l2805_280542


namespace bug_crawl_theorem_l2805_280547

def bug_movements : List Int := [5, -3, 10, -8, -6, 12, -10]

theorem bug_crawl_theorem :
  (List.sum bug_movements = 0) ∧
  (List.sum (List.map Int.natAbs bug_movements) = 54) := by
  sorry

end bug_crawl_theorem_l2805_280547


namespace sum_product_theorem_l2805_280563

theorem sum_product_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 213) 
  (h2 : a + b + c = 15) : 
  a*b + b*c + c*a = 6 := by
sorry

end sum_product_theorem_l2805_280563


namespace lcm_gcd_product_12_75_l2805_280539

theorem lcm_gcd_product_12_75 : Nat.lcm 12 75 * Nat.gcd 12 75 = 900 := by sorry

end lcm_gcd_product_12_75_l2805_280539


namespace shortest_tangent_length_l2805_280596

/-- Circle C3 centered at (8, 0) with radius 5 -/
def C3 (x y : ℝ) : Prop := (x - 8)^2 + y^2 = 25

/-- Circle C4 centered at (-10, 0) with radius 7 -/
def C4 (x y : ℝ) : Prop := (x + 10)^2 + y^2 = 49

/-- Point R on circle C3 -/
def R : ℝ × ℝ := sorry

/-- Point S on circle C4 -/
def S : ℝ × ℝ := sorry

/-- The shortest line segment RS is tangent to C3 at R and C4 at S -/
theorem shortest_tangent_length : 
  C3 R.1 R.2 ∧ C4 S.1 S.2 → 
  ∃ (R S : ℝ × ℝ), C3 R.1 R.2 ∧ C4 S.1 S.2 ∧ 
    Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 15 := by
  sorry

end shortest_tangent_length_l2805_280596


namespace collinear_points_sum_l2805_280582

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (a b c : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), b - a = t • (c - a) ∨ c - a = t • (b - a)

/-- The theorem states that if the given points are collinear, then p + q = 6. -/
theorem collinear_points_sum (p q : ℝ) :
  collinear (2, p, q) (p, 3, q) (p, q, 4) → p + q = 6 := by
  sorry

end collinear_points_sum_l2805_280582


namespace triangle_area_and_angle_l2805_280537

-- Define the triangle ABC
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Add any necessary conditions for a valid triangle
  true

-- Define the dot product of two 2D vectors
def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  x₁ * x₂ + y₁ * y₂

-- Define parallel vectors
def parallel (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * y₂ = x₂ * y₁

theorem triangle_area_and_angle (A B C : ℝ) (a b c : ℝ) :
  triangle A B C a b c →
  Real.cos C = 3/10 →
  dot_product c 0 (-a) 0 = 9/2 →
  parallel (2 * Real.sin B) (-Real.sqrt 3) (Real.cos (2 * B)) (1 - 2 * (Real.sin (B/2))^2) →
  (1/2 * a * b * Real.sin C = (3 * Real.sqrt 91)/4) ∧ B = 5*π/6 :=
by sorry

end triangle_area_and_angle_l2805_280537


namespace smallest_divisible_by_12_l2805_280575

def initial_sequence : List ℕ := [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]

def append_number (seq : List ℕ) (n : ℕ) : List ℕ :=
  seq ++ [n]

def to_single_number (seq : List ℕ) : ℕ :=
  seq.foldl (λ acc x => acc * 10^(Nat.digits 10 x).length + x) 0

def is_divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

theorem smallest_divisible_by_12 :
  ∃ N : ℕ, N ≥ 82 ∧
    is_divisible_by_12 (to_single_number (append_number initial_sequence N)) ∧
    ∀ k : ℕ, 82 ≤ k ∧ k < N →
      ¬is_divisible_by_12 (to_single_number (append_number initial_sequence k)) ∧
    N = 84 := by
  sorry

end smallest_divisible_by_12_l2805_280575


namespace black_population_west_percentage_l2805_280529

def black_population_ne : ℕ := 6
def black_population_mw : ℕ := 7
def black_population_south : ℕ := 18
def black_population_west : ℕ := 4

def total_black_population : ℕ := black_population_ne + black_population_mw + black_population_south + black_population_west

def percentage_in_west : ℚ := black_population_west / total_black_population

theorem black_population_west_percentage :
  ∃ (p : ℚ), abs (percentage_in_west - p) < 1/100 ∧ p = 11/100 := by
  sorry

end black_population_west_percentage_l2805_280529


namespace nested_sqrt_simplification_l2805_280520

theorem nested_sqrt_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x ^ 9) ^ (1/4) := by
  sorry

end nested_sqrt_simplification_l2805_280520


namespace dogwood_trees_tomorrow_l2805_280524

/-- The number of dogwood trees to be planted tomorrow in the park --/
def trees_planted_tomorrow (initial_trees : ℕ) (planted_today : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial_trees + planted_today)

/-- Theorem: Given the initial number of trees, the number planted today, and the final total,
    prove that 20 trees will be planted tomorrow --/
theorem dogwood_trees_tomorrow :
  trees_planted_tomorrow 39 41 100 = 20 := by
  sorry

end dogwood_trees_tomorrow_l2805_280524


namespace circle_area_difference_l2805_280514

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let d2 : ℝ := 30
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 675 * π := by sorry

end circle_area_difference_l2805_280514


namespace total_reading_materials_l2805_280595

def magazines : ℕ := 425
def newspapers : ℕ := 275

theorem total_reading_materials : magazines + newspapers = 700 := by
  sorry

end total_reading_materials_l2805_280595


namespace rachel_essay_pages_l2805_280509

/-- Rachel's essay writing problem -/
theorem rachel_essay_pages :
  let pages_per_30_min : ℕ := 1
  let research_time : ℕ := 45
  let editing_time : ℕ := 75
  let total_time : ℕ := 300
  let writing_time : ℕ := total_time - (research_time + editing_time)
  let pages_written : ℕ := writing_time / 30
  pages_written = 6 := by sorry

end rachel_essay_pages_l2805_280509


namespace complex_fraction_difference_l2805_280522

theorem complex_fraction_difference : 
  (Complex.mk 3 2) / (Complex.mk 2 (-3)) - (Complex.mk 3 (-2)) / (Complex.mk 2 3) = Complex.I * 2 := by
  sorry

end complex_fraction_difference_l2805_280522


namespace limes_remaining_l2805_280527

/-- The number of limes Mike picked -/
def mike_limes : ℝ := 32.0

/-- The number of limes Alyssa ate -/
def alyssa_limes : ℝ := 25.0

/-- The number of limes left -/
def limes_left : ℝ := mike_limes - alyssa_limes

theorem limes_remaining : limes_left = 7.0 := by sorry

end limes_remaining_l2805_280527


namespace pushup_problem_l2805_280561

/-- Given that David did 30 more push-ups than Zachary, Sarah completed twice as many push-ups as Zachary,
    and David did 37 push-ups, prove that Zachary and Sarah did 21 push-ups combined. -/
theorem pushup_problem (david zachary sarah : ℕ) : 
  david = zachary + 30 →
  sarah = 2 * zachary →
  david = 37 →
  zachary + sarah = 21 := by
  sorry

end pushup_problem_l2805_280561


namespace sum_product_over_sum_squares_l2805_280554

theorem sum_product_over_sum_squares (a b c : ℝ) (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h2 : a + b + c = 1) :
  (a * b + b * c + c * a) / (a^2 + b^2 + c^2) = 0 := by
  sorry

end sum_product_over_sum_squares_l2805_280554


namespace system_solution_l2805_280579

theorem system_solution (x y a : ℝ) : 
  3 * x + y = a → 
  2 * x + 5 * y = 2 * a → 
  x = 3 → 
  a = 13 := by sorry

end system_solution_l2805_280579


namespace rectangle_max_area_l2805_280504

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) → 
  l * w = 100 := by
sorry

end rectangle_max_area_l2805_280504


namespace clothing_store_gross_profit_l2805_280597

-- Define the purchase price
def purchase_price : ℚ := 81

-- Define the initial markup percentage
def markup_percentage : ℚ := 1/4

-- Define the price decrease percentage
def price_decrease_percentage : ℚ := 1/5

-- Define the function to calculate the initial selling price
def initial_selling_price (purchase_price : ℚ) (markup_percentage : ℚ) : ℚ :=
  purchase_price / (1 - markup_percentage)

-- Define the function to calculate the new selling price after discount
def new_selling_price (initial_price : ℚ) (decrease_percentage : ℚ) : ℚ :=
  initial_price * (1 - decrease_percentage)

-- Define the function to calculate the gross profit
def gross_profit (new_price : ℚ) (purchase_price : ℚ) : ℚ :=
  new_price - purchase_price

-- Theorem statement
theorem clothing_store_gross_profit :
  let initial_price := initial_selling_price purchase_price markup_percentage
  let new_price := new_selling_price initial_price price_decrease_percentage
  gross_profit new_price purchase_price = 27/5 := by sorry

end clothing_store_gross_profit_l2805_280597


namespace factorization_equality_l2805_280533

theorem factorization_equality (a x y : ℝ) :
  5 * a * x^2 - 5 * a * y^2 = 5 * a * (x + y) * (x - y) := by
  sorry

end factorization_equality_l2805_280533


namespace regular_polygon_sides_l2805_280501

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end regular_polygon_sides_l2805_280501


namespace compare_fractions_compare_specific_fractions_l2805_280503

theorem compare_fractions (a b : ℝ) (h1 : 3 * a > b) (h2 : b > 0) : a / b > (a + 1) / (b + 3) := by
  sorry

theorem compare_specific_fractions : (23 : ℝ) / 68 < 22 / 65 := by
  sorry

end compare_fractions_compare_specific_fractions_l2805_280503


namespace unique_natural_number_solution_l2805_280598

theorem unique_natural_number_solution (n p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → (1 : ℚ) / n = 1 / p + 1 / q + 1 / (p * q) → n = 1 := by
  sorry

end unique_natural_number_solution_l2805_280598


namespace exponential_function_passes_through_point_l2805_280568

theorem exponential_function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end exponential_function_passes_through_point_l2805_280568


namespace balcony_orchestra_difference_is_40_l2805_280511

/-- Represents the ticket sales for a theater performance --/
structure TheaterSales where
  orchestra_price : ℕ
  balcony_price : ℕ
  total_tickets : ℕ
  total_revenue : ℕ

/-- Calculates the difference between balcony and orchestra ticket sales --/
def balcony_orchestra_difference (sales : TheaterSales) : ℕ :=
  sales.total_tickets - 2 * (sales.total_revenue - sales.balcony_price * sales.total_tickets) / (sales.orchestra_price - sales.balcony_price)

/-- Theorem stating the difference between balcony and orchestra ticket sales --/
theorem balcony_orchestra_difference_is_40 (sales : TheaterSales) 
  (h1 : sales.orchestra_price = 12)
  (h2 : sales.balcony_price = 8)
  (h3 : sales.total_tickets = 340)
  (h4 : sales.total_revenue = 3320) :
  balcony_orchestra_difference sales = 40 := by
  sorry

#eval balcony_orchestra_difference ⟨12, 8, 340, 3320⟩

end balcony_orchestra_difference_is_40_l2805_280511


namespace surface_area_ratio_of_cubes_l2805_280584

theorem surface_area_ratio_of_cubes (a b : ℝ) (h : a / b = 5) :
  (6 * a^2) / (6 * b^2) = 25 := by
  sorry

end surface_area_ratio_of_cubes_l2805_280584


namespace batsman_innings_count_l2805_280572

theorem batsman_innings_count
  (avg : ℝ)
  (score_diff : ℕ)
  (avg_excluding : ℝ)
  (highest_score : ℕ)
  (h_avg : avg = 60)
  (h_score_diff : score_diff = 150)
  (h_avg_excluding : avg_excluding = 58)
  (h_highest_score : highest_score = 179)
  : ∃ n : ℕ, n = 46 ∧ 
    avg * n = avg_excluding * (n - 2) + highest_score + (highest_score - score_diff) :=
by sorry

end batsman_innings_count_l2805_280572


namespace arithmetic_sequence_property_l2805_280593

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 6 + a 8 = 10) 
  (h_a3 : a 3 = 1) : 
  a 11 = 9 := by
sorry

end arithmetic_sequence_property_l2805_280593


namespace infinitely_many_solutions_l2805_280576

/-- The sequence defined by a₀ = a₁ = a₂ = 1 and a_{n+2} = (a_n * a_{n+1} + 1) / a_{n-1} for n ≥ 1 -/
def a : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 1
| (n + 3) => (a n * a (n + 1) + 1) / a (n - 1)

/-- The property that needs to be satisfied by the triples (a, b, c) -/
def satisfies_equation (a b c : ℕ) : Prop :=
  1 / a + 1 / b + 1 / c + 1 / (a * b * c) = 12 / (a + b + c)

theorem infinitely_many_solutions :
  ∀ N : ℕ, ∃ a b c : ℕ, a > N ∧ b > N ∧ c > N ∧ satisfies_equation a b c :=
sorry

end infinitely_many_solutions_l2805_280576


namespace complex_cube_l2805_280508

theorem complex_cube (i : ℂ) : i^2 = -1 → (2 - 3*i)^3 = -46 - 9*i := by
  sorry

end complex_cube_l2805_280508


namespace cos_A_minus_B_l2805_280586

theorem cos_A_minus_B (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
  sorry

end cos_A_minus_B_l2805_280586


namespace book_pages_calculation_l2805_280534

theorem book_pages_calculation (first_day_percent : ℝ) (second_day_percent : ℝ) 
  (third_day_pages : ℕ) :
  first_day_percent = 0.1 →
  second_day_percent = 0.25 →
  (first_day_percent + second_day_percent + (third_day_pages : ℝ) / (240 : ℝ)) = 0.5 →
  third_day_pages = 30 →
  (240 : ℕ) = 240 :=
by sorry

end book_pages_calculation_l2805_280534


namespace cubic_equation_root_l2805_280551

theorem cubic_equation_root (c d : ℚ) : 
  (∃ x : ℝ, x^3 + c*x^2 + d*x + 44 = 0 ∧ x = 1 - 3*Real.sqrt 5) → c = -3 := by
  sorry

end cubic_equation_root_l2805_280551


namespace factor_expression_l2805_280573

theorem factor_expression (x : ℝ) : 4 * x * (x + 1) + 9 * (x + 1) = (x + 1) * (4 * x + 9) := by
  sorry

end factor_expression_l2805_280573


namespace product_125_sum_31_l2805_280535

theorem product_125_sum_31 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → a * b * c = 125 → a + b + c = 31 := by
  sorry

end product_125_sum_31_l2805_280535


namespace plane_curve_mass_approx_l2805_280590

noncomputable def curve_mass (a b : Real) : Real :=
  ∫ x in a..b, (1 + x^2) * Real.sqrt (1 + (3 * x^2)^2)

theorem plane_curve_mass_approx : 
  ∃ ε > 0, abs (curve_mass 0 0.1 - 0.099985655) < ε :=
sorry

end plane_curve_mass_approx_l2805_280590


namespace johns_money_ratio_l2805_280585

/-- The ratio of money John got from his grandma to his grandpa -/
theorem johns_money_ratio :
  ∀ (x : ℚ), 
  (30 : ℚ) + 30 * x = 120 →
  (30 * x) / 30 = 3 / 1 :=
by sorry

end johns_money_ratio_l2805_280585


namespace four_customers_no_change_l2805_280546

/-- Represents the auto shop scenario -/
structure AutoShop where
  initial_cars : ℕ
  new_customers : ℕ
  tires_per_car : ℕ
  half_change_customers : ℕ
  tires_left : ℕ

/-- Calculates the number of customers who didn't want their tires changed -/
def customers_no_change (shop : AutoShop) : ℕ :=
  let total_cars := shop.initial_cars + shop.new_customers
  let total_tires_bought := total_cars * shop.tires_per_car
  let half_change_tires := shop.half_change_customers * (shop.tires_per_car / 2)
  let unused_tires := shop.tires_left - half_change_tires
  unused_tires / shop.tires_per_car

/-- Theorem stating that given the conditions, 4 customers decided not to change their tires -/
theorem four_customers_no_change (shop : AutoShop) 
  (h1 : shop.initial_cars = 4)
  (h2 : shop.new_customers = 6)
  (h3 : shop.tires_per_car = 4)
  (h4 : shop.half_change_customers = 2)
  (h5 : shop.tires_left = 20) :
  customers_no_change shop = 4 := by
  sorry

#eval customers_no_change { initial_cars := 4, new_customers := 6, tires_per_car := 4, half_change_customers := 2, tires_left := 20 }

end four_customers_no_change_l2805_280546


namespace proportion_fourth_term_l2805_280557

theorem proportion_fourth_term (x y : ℝ) : 
  (0.25 / x = 2 / y) → x = 0.75 → y = 6 := by sorry

end proportion_fourth_term_l2805_280557


namespace sphere_to_wire_length_l2805_280517

-- Define constants
def sphere_radius : ℝ := 12
def wire_radius : ℝ := 0.8

-- Define the theorem
theorem sphere_to_wire_length :
  let sphere_volume := (4/3) * Real.pi * (sphere_radius ^ 3)
  let wire_volume := Real.pi * (wire_radius ^ 2) * wire_length
  let wire_length := sphere_volume / (Real.pi * (wire_radius ^ 2))
  wire_length = 3600 := by sorry

end sphere_to_wire_length_l2805_280517


namespace f_properties_imply_b_range_l2805_280506

-- Define the function f
noncomputable def f (b : ℝ) : ℝ → ℝ := fun x =>
  if 0 < x ∧ x < 2 then Real.log (x^2 - x + b) else 0  -- placeholder for other x values

-- State the theorem
theorem f_properties_imply_b_range :
  ∀ b : ℝ,
  (∀ x : ℝ, f b (-x) = -(f b x)) →  -- f is odd
  (∀ x : ℝ, f b (x + 4) = f b x) →  -- f has period 4
  (∀ x : ℝ, 0 < x → x < 2 → f b x = Real.log (x^2 - x + b)) →  -- f definition for x ∈ (0, 2)
  (∃ x₁ x₂ x₃ x₄ x₅ : ℝ, -2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧ x₅ ≤ 2 ∧
    f b x₁ = 0 ∧ f b x₂ = 0 ∧ f b x₃ = 0 ∧ f b x₄ = 0 ∧ f b x₅ = 0) →  -- 5 zero points in [-2, 2]
  ((1/4 < b ∧ b ≤ 1) ∨ b = 5/4) :=
by sorry

end f_properties_imply_b_range_l2805_280506


namespace negation_of_proposition_l2805_280519

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, (x ≠ 3 ∧ x ≠ 2) → x^2 - 5*x + 6 ≠ 0)) ↔
  (∀ x : ℝ, (x = 3 ∨ x = 2) → x^2 - 5*x + 6 = 0) :=
by sorry

end negation_of_proposition_l2805_280519


namespace fraction_decomposition_l2805_280577

theorem fraction_decomposition (x A B : ℚ) : 
  (7 * x - 15) / (3 * x^2 + 2 * x - 8) = A / (x + 2) + B / (3 * x - 4) → 
  A = 29 / 10 ∧ B = -17 / 10 := by
sorry

end fraction_decomposition_l2805_280577


namespace sum_of_products_l2805_280525

theorem sum_of_products (x : Fin 150 → ℝ) : 
  (∀ i, x i = Real.sqrt 2 + 1 ∨ x i = Real.sqrt 2 - 1) →
  (∃ x : Fin 150 → ℝ, (∀ i, x i = Real.sqrt 2 + 1 ∨ x i = Real.sqrt 2 - 1) ∧ 
    (Finset.sum (Finset.range 75) (λ i => x (2*i) * x (2*i+1)) = 111)) ∧
  (¬ ∃ x : Fin 150 → ℝ, (∀ i, x i = Real.sqrt 2 + 1 ∨ x i = Real.sqrt 2 - 1) ∧ 
    (Finset.sum (Finset.range 75) (λ i => x (2*i) * x (2*i+1)) = 121)) :=
by sorry

end sum_of_products_l2805_280525


namespace f_extrema_on_interval_l2805_280510

def f (x : ℝ) : ℝ := x^3 + 2*x

theorem f_extrema_on_interval :
  let a := -1
  let b := 1
  ∃ (x_min x_max : ℝ),
    x_min ∈ [a, b] ∧
    x_max ∈ [a, b] ∧
    (∀ x ∈ [a, b], f x ≥ f x_min) ∧
    (∀ x ∈ [a, b], f x ≤ f x_max) ∧
    f x_min = -3 ∧
    f x_max = 3 :=
sorry

end f_extrema_on_interval_l2805_280510


namespace simplify_expression_l2805_280548

theorem simplify_expression (x : ℝ) : 120*x - 72*x + 15*x - 9*x = 54*x := by
  sorry

end simplify_expression_l2805_280548


namespace complex_magnitude_power_l2805_280550

theorem complex_magnitude_power : 
  Complex.abs ((2 : ℂ) + (2 * Complex.I * Real.sqrt 3)) ^ 6 = 4096 := by
  sorry

end complex_magnitude_power_l2805_280550


namespace three_digit_divisible_by_three_l2805_280562

theorem three_digit_divisible_by_three :
  ∀ n : ℕ,
  (n ≥ 100 ∧ n < 1000) →  -- Three-digit number
  (n % 10 = 4) →  -- Units digit is 4
  (n / 100 = 4) →  -- Hundreds digit is 4
  (n % 3 = 0) →  -- Divisible by 3
  (n = 414 ∨ n = 444 ∨ n = 474) :=
by sorry

end three_digit_divisible_by_three_l2805_280562


namespace complement_of_M_l2805_280549

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 < 2*x}

-- State the theorem
theorem complement_of_M :
  (Set.univ : Set ℝ) \ M = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 2} := by sorry

end complement_of_M_l2805_280549


namespace gecko_eating_pattern_l2805_280512

/-- Represents the gecko's eating pattern over three days -/
structure GeckoEating where
  total_crickets : ℕ
  third_day_crickets : ℕ
  second_day_difference : ℕ

/-- Calculates the percentage of crickets eaten on the first day -/
def first_day_percentage (g : GeckoEating) : ℚ :=
  let first_two_days := g.total_crickets - g.third_day_crickets
  let x := (2 * first_two_days + g.second_day_difference) / (2 * g.total_crickets)
  x * 100

/-- Theorem stating that under the given conditions, the gecko eats 30% of crickets on the first day -/
theorem gecko_eating_pattern :
  let g : GeckoEating := {
    total_crickets := 70,
    third_day_crickets := 34,
    second_day_difference := 6
  }
  first_day_percentage g = 30 := by sorry

end gecko_eating_pattern_l2805_280512


namespace intersection_implies_m_eq_neg_two_l2805_280592

-- Define the sets M and N
def M (m : ℝ) : Set ℂ := {1, 2, (m^2 - 2*m - 5 : ℂ) + (m^2 + 5*m + 6 : ℂ)*Complex.I}
def N : Set ℂ := {3}

-- State the theorem
theorem intersection_implies_m_eq_neg_two (m : ℝ) : 
  (M m ∩ N).Nonempty → m = -2 :=
by
  sorry

end intersection_implies_m_eq_neg_two_l2805_280592


namespace ruler_cost_l2805_280531

theorem ruler_cost (total_students : ℕ) (total_expense : ℕ) :
  total_students = 42 →
  total_expense = 2310 →
  ∃ (num_buyers : ℕ) (rulers_per_student : ℕ) (cost_per_ruler : ℕ),
    num_buyers > total_students / 2 ∧
    cost_per_ruler > rulers_per_student ∧
    num_buyers * rulers_per_student * cost_per_ruler = total_expense ∧
    cost_per_ruler = 11 :=
by sorry


end ruler_cost_l2805_280531


namespace triangle_problem_l2805_280552

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a * Real.cos B * Real.cos C + b * Real.cos A * Real.cos C = c / 2 →
  c = Real.sqrt 7 →
  a + b = 5 →
  C = π / 3 ∧ (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
by sorry

end triangle_problem_l2805_280552


namespace f_sum_l2805_280502

/-- A function satisfying the given properties -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- f(t) = f(1-t) for all t ∈ ℝ -/
axiom f_symmetry (t : ℝ) : f t = f (1 - t)

/-- f(x) = -x² for x ∈ [0, 1/2] -/
axiom f_def (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1/2) : f x = -x^2

/-- The main theorem to prove -/
theorem f_sum : f 3 + f (-3/2) = -1/4 := by sorry

end f_sum_l2805_280502


namespace f_property_l2805_280569

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.sqrt x
  else if x ≥ 1 then 2 * (x - 1)
  else 0  -- This case should never occur in our problem

-- State the theorem
theorem f_property (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : f a = f (a + 1)) :
  f (1 / a) = 6 := by
  sorry

end f_property_l2805_280569


namespace left_square_side_length_l2805_280589

/-- Proves that given three squares with specific side length relationships, 
    the left square has a side length of 8 cm. -/
theorem left_square_side_length : 
  ∀ (left middle right : ℝ),
  left + middle + right = 52 →
  middle = left + 17 →
  right = middle - 6 →
  left = 8 :=
by
  sorry

end left_square_side_length_l2805_280589


namespace jasmine_remaining_money_l2805_280505

/-- Calculates the remaining amount after spending on fruits --/
def remaining_amount (initial : ℝ) (spent : ℝ) : ℝ :=
  initial - spent

/-- Theorem: The remaining amount after spending $15.00 from an initial $100.00 is $85.00 --/
theorem jasmine_remaining_money :
  remaining_amount 100 15 = 85 := by
  sorry

end jasmine_remaining_money_l2805_280505


namespace number_difference_l2805_280521

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 23405)
  (b_div_5 : ∃ k : ℕ, b = 5 * k)
  (b_div_10_eq_5a : b / 10 = 5 * a) :
  b - a = 21600 :=
by sorry

end number_difference_l2805_280521


namespace gcd_210_294_l2805_280565

theorem gcd_210_294 : Nat.gcd 210 294 = 42 := by
  sorry

end gcd_210_294_l2805_280565


namespace simplified_A_value_l2805_280558

theorem simplified_A_value (a : ℝ) : 
  let A := (a - 1) / (a + 2) * ((a^2 - 4) / (a^2 - 2*a + 1)) / (1 / (a - 1))
  (a^2 - a = 0) → A = -2 := by
  sorry

end simplified_A_value_l2805_280558


namespace dress_design_combinations_l2805_280518

theorem dress_design_combinations (num_colors num_patterns : ℕ) : 
  num_colors = 5 → num_patterns = 6 → num_colors * num_patterns = 30 := by
  sorry

end dress_design_combinations_l2805_280518


namespace disjoint_quadratic_sets_l2805_280560

theorem disjoint_quadratic_sets (A B : ℤ) : ∃ C : ℤ,
  ∀ x y : ℤ, x^2 + A*x + B ≠ 2*y^2 + 2*y + C :=
by sorry

end disjoint_quadratic_sets_l2805_280560


namespace max_a_is_pi_over_four_l2805_280516

/-- If f(x) = cos x - sin x is a decreasing function on the interval [-a, a], 
    then the maximum value of a is π/4 -/
theorem max_a_is_pi_over_four (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = Real.cos x - Real.sin x) →
  (∀ x y, -a ≤ x ∧ x < y ∧ y ≤ a → f y < f x) →
  a ≤ π / 4 ∧ ∀ b, (∀ x y, -b ≤ x ∧ x < y ∧ y ≤ b → f y < f x) → b ≤ a :=
by sorry

end max_a_is_pi_over_four_l2805_280516


namespace valid_solutions_characterization_l2805_280545

/-- A number is a valid solution if it's a four-digit number,
    divisible by 28, and can be expressed as the sum of squares
    of three consecutive even numbers. -/
def is_valid_solution (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  n % 28 = 0 ∧
  ∃ k : ℕ, n = 12 * k^2 + 8

/-- The set of all valid solutions -/
def solution_set : Set ℕ := {1736, 3080, 4340, 6356, 8120}

/-- Theorem stating that the solution_set contains exactly
    the numbers satisfying is_valid_solution -/
theorem valid_solutions_characterization :
  ∀ n : ℕ, is_valid_solution n ↔ n ∈ solution_set :=
by sorry

#check valid_solutions_characterization

end valid_solutions_characterization_l2805_280545


namespace car_engine_part_cost_l2805_280553

/-- Calculates the cost of a car engine part given labor and total cost information --/
theorem car_engine_part_cost
  (labor_rate : ℕ)
  (labor_hours : ℕ)
  (total_cost : ℕ)
  (h1 : labor_rate = 75)
  (h2 : labor_hours = 16)
  (h3 : total_cost = 2400) :
  total_cost - (labor_rate * labor_hours) = 1200 := by
  sorry

#check car_engine_part_cost

end car_engine_part_cost_l2805_280553


namespace magical_red_knights_fraction_l2805_280530

theorem magical_red_knights_fraction (total : ℕ) (total_pos : 0 < total) :
  let red := (2 : ℚ) / 7 * total
  let blue := total - red
  let magical := (1 : ℚ) / 6 * total
  let red_magical_fraction := magical / red
  let blue_magical_fraction := magical / blue
  red_magical_fraction = 2 * blue_magical_fraction →
  red_magical_fraction = 7 / 27 := by
sorry

end magical_red_knights_fraction_l2805_280530


namespace at_least_one_passes_probability_l2805_280581

/-- Probability of A answering a single question correctly -/
def prob_A : ℚ := 2/3

/-- Probability of B answering a single question correctly -/
def prob_B : ℚ := 1/2

/-- Number of questions in the test -/
def num_questions : ℕ := 3

/-- Number of correct answers required to pass -/
def pass_threshold : ℕ := 2

/-- Probability of at least one of A and B passing the test -/
def prob_at_least_one_passes : ℚ := 47/54

theorem at_least_one_passes_probability :
  prob_at_least_one_passes = 1 - (1 - (Nat.choose num_questions pass_threshold * prob_A^pass_threshold * (1-prob_A)^(num_questions-pass_threshold) + prob_A^num_questions)) *
                                 (1 - (Nat.choose num_questions pass_threshold * prob_B^pass_threshold * (1-prob_B)^(num_questions-pass_threshold) + prob_B^num_questions)) :=
sorry

end at_least_one_passes_probability_l2805_280581


namespace max_sum_arithmetic_sequence_l2805_280532

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

def sum_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem max_sum_arithmetic_sequence 
  (a d : ℤ) 
  (h1 : a + 16 * d = 52) 
  (h2 : a + 29 * d = 13) :
  ∃ n : ℕ, 
    (arithmetic_sequence a d n > 0) ∧ 
    (arithmetic_sequence a d (n + 1) ≤ 0) ∧
    (∀ m : ℕ, m > n → arithmetic_sequence a d m ≤ 0) ∧
    (sum_arithmetic_sequence a d n = 1717) := by
  sorry

end max_sum_arithmetic_sequence_l2805_280532


namespace sum_p_q_r_l2805_280578

/-- The largest real solution to the given equation -/
noncomputable def n : ℝ := 
  Real.sqrt (53 + Real.sqrt 249) + 13

/-- The equation that n satisfies -/
axiom n_eq : (4 / (n - 4)) + (6 / (n - 6)) + (18 / (n - 18)) + (20 / (n - 20)) = n^2 - 13*n - 6

/-- The existence of positive integers p, q, and r -/
axiom exists_p_q_r : ∃ (p q r : ℕ+), n = p + Real.sqrt (q + Real.sqrt r)

/-- The theorem to be proved -/
theorem sum_p_q_r : ∃ (p q r : ℕ+), n = p + Real.sqrt (q + Real.sqrt r) ∧ p + q + r = 315 := by
  sorry

end sum_p_q_r_l2805_280578


namespace probability_diamond_spade_heart_value_l2805_280574

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of cards of each suit in a standard deck -/
def CardsPerSuit : ℕ := 13

/-- Probability of drawing a diamond, then a spade, then a heart from a standard deck -/
def probability_diamond_spade_heart : ℚ :=
  (CardsPerSuit / StandardDeck) *
  (CardsPerSuit / (StandardDeck - 1)) *
  (CardsPerSuit / (StandardDeck - 2))

/-- Theorem stating the probability of drawing a diamond, then a spade, then a heart -/
theorem probability_diamond_spade_heart_value :
  probability_diamond_spade_heart = 169 / 10200 := by
  sorry

end probability_diamond_spade_heart_value_l2805_280574


namespace polynomial_evaluation_l2805_280541

theorem polynomial_evaluation :
  let f : ℝ → ℝ := λ x => 2*x^4 + 3*x^3 + x^2 + 2*x + 3
  f 2 = 67 := by
sorry

end polynomial_evaluation_l2805_280541


namespace inequality_proof_l2805_280566

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end inequality_proof_l2805_280566


namespace remaining_expenses_l2805_280515

def base_8_to_10 (n : ℕ) : ℕ := 
  5 * 8^3 + 4 * 8^2 + 3 * 8^1 + 2 * 8^0

def savings : ℕ := base_8_to_10 5432
def ticket_cost : ℕ := 1200

theorem remaining_expenses : savings - ticket_cost = 1642 := by
  sorry

end remaining_expenses_l2805_280515


namespace negation_equivalence_l2805_280571

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x + |x| < 0) ↔ (∀ x : ℝ, x + |x| ≥ 0) := by
  sorry

end negation_equivalence_l2805_280571


namespace x_value_l2805_280570

def M (x : ℝ) : Set ℝ := {2, 0, x}
def N : Set ℝ := {0, 1}

theorem x_value (h : N ⊆ M x) : x = 1 := by
  sorry

end x_value_l2805_280570


namespace seven_power_plus_one_prime_divisors_l2805_280526

theorem seven_power_plus_one_prime_divisors (n : ℕ) :
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ 
    (∀ p ∈ S, p ∣ (7^(7^n) + 1)) ∧ 
    (Finset.card S ≥ 2*n + 3) :=
by sorry

end seven_power_plus_one_prime_divisors_l2805_280526


namespace sum_of_squares_l2805_280507

theorem sum_of_squares (a b c : ℕ+) (h1 : a < b) (h2 : b < c)
  (h3 : (b.val * c.val - 1) % a.val = 0)
  (h4 : (a.val * c.val - 1) % b.val = 0)
  (h5 : (a.val * b.val - 1) % c.val = 0) :
  a^2 + b^2 + c^2 = 38 := by
sorry

end sum_of_squares_l2805_280507


namespace custom_op_seven_three_l2805_280556

-- Define the custom operation
def custom_op (a b : ℤ) : ℤ := 4*a + 5*b - a*b

-- Theorem statement
theorem custom_op_seven_three :
  custom_op 7 3 = 22 := by
  sorry

end custom_op_seven_three_l2805_280556


namespace geometric_arithmetic_mean_sum_l2805_280588

theorem geometric_arithmetic_mean_sum (a b c x y : ℝ) 
  (h1 : b ^ 2 = a * c)  -- geometric sequence condition
  (h2 : x ≠ 0)
  (h3 : y ≠ 0)
  (h4 : 2 * x = a + b)  -- arithmetic mean condition
  (h5 : 2 * y = b + c)  -- arithmetic mean condition
  : a / x + c / y = 2 := by
  sorry

end geometric_arithmetic_mean_sum_l2805_280588


namespace plane_points_theorem_l2805_280587

def connecting_lines (n : ℕ) : ℕ := n * (n - 1) / 2

theorem plane_points_theorem (n₁ n₂ : ℕ) : 
  (connecting_lines n₁ = connecting_lines n₂ + 27) →
  (connecting_lines n₁ + connecting_lines n₂ = 171) →
  (n₁ = 11 ∧ n₂ = 8) :=
by sorry

end plane_points_theorem_l2805_280587


namespace expression_evaluation_l2805_280513

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = b - 12)
  (h2 : b = a + 4)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  ((a + 3) / (a + 2)) * ((b + 1) / (b - 3)) * ((c + 10) / (c + 7)) = 10 / 3 := by
  sorry

end expression_evaluation_l2805_280513


namespace meal_price_calculation_l2805_280536

/-- Calculate the entire price of a meal given the costs and tip percentage --/
theorem meal_price_calculation 
  (appetizer_cost : ℚ)
  (entree_cost : ℚ)
  (num_entrees : ℕ)
  (dessert_cost : ℚ)
  (tip_percentage : ℚ)
  (h1 : appetizer_cost = 9)
  (h2 : entree_cost = 20)
  (h3 : num_entrees = 2)
  (h4 : dessert_cost = 11)
  (h5 : tip_percentage = 30 / 100) :
  appetizer_cost + num_entrees * entree_cost + dessert_cost + 
  (appetizer_cost + num_entrees * entree_cost + dessert_cost) * tip_percentage = 78 := by
  sorry

end meal_price_calculation_l2805_280536


namespace total_practice_time_is_135_l2805_280580

/-- The number of minutes Daniel practices basketball each day during the school week -/
def school_day_practice : ℕ := 15

/-- The number of days in a school week -/
def school_week_days : ℕ := 5

/-- The number of days in a weekend -/
def weekend_days : ℕ := 2

/-- The total number of minutes Daniel practices during a whole week -/
def total_practice_time : ℕ :=
  (school_day_practice * school_week_days) +
  (2 * school_day_practice * weekend_days)

theorem total_practice_time_is_135 :
  total_practice_time = 135 := by
  sorry

end total_practice_time_is_135_l2805_280580


namespace tangent_perpendicular_to_y_axis_l2805_280599

/-- The curve function -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + 1

/-- The derivative of the curve function -/
def f_prime (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_perpendicular_to_y_axis (a : ℝ) :
  (f a (-1) = a + 2) →
  (f_prime a (-1) = 0) →
  a = -2 := by
  sorry

end tangent_perpendicular_to_y_axis_l2805_280599


namespace arithmetic_sequence_sum_l2805_280594

/-- Given an arithmetic sequence 3, 7, 11, ..., x, y, 35, prove that x + y = 58 -/
theorem arithmetic_sequence_sum (x y : ℝ) : 
  (∃ (n : ℕ), n ≥ 5 ∧ 
    (∀ k : ℕ, k ≤ n → 
      (if k = 1 then 3
       else if k = 2 then 7
       else if k = 3 then 11
       else if k = n - 1 then x
       else if k = n then y
       else if k = n + 1 then 35
       else 0) = 3 + (k - 1) * 4)) →
  x + y = 58 := by
  sorry

end arithmetic_sequence_sum_l2805_280594


namespace lemon_juice_fraction_l2805_280544

theorem lemon_juice_fraction (total_members : ℕ) (orange_juice_orders : ℕ) : 
  total_members = 30 →
  orange_juice_orders = 6 →
  ∃ (lemon_fraction : ℚ),
    lemon_fraction = 7 / 10 ∧
    lemon_fraction * total_members +
    (1 / 3) * (total_members - lemon_fraction * total_members) +
    orange_juice_orders = total_members :=
by sorry

end lemon_juice_fraction_l2805_280544
