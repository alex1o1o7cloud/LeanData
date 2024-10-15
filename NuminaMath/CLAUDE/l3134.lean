import Mathlib

namespace NUMINAMATH_CALUDE_lateral_edge_length_for_specific_pyramid_l3134_313497

/-- A regular quadrilateral pyramid with given base side length and volume -/
structure RegularQuadPyramid where
  base_side : ℝ
  volume : ℝ

/-- The length of the lateral edge of a regular quadrilateral pyramid -/
def lateral_edge_length (p : RegularQuadPyramid) : ℝ :=
  sorry

theorem lateral_edge_length_for_specific_pyramid :
  let p : RegularQuadPyramid := { base_side := 2, volume := 4 * Real.sqrt 3 / 3 }
  lateral_edge_length p = Real.sqrt 5 := by
    sorry

end NUMINAMATH_CALUDE_lateral_edge_length_for_specific_pyramid_l3134_313497


namespace NUMINAMATH_CALUDE_simplify_power_expression_l3134_313416

theorem simplify_power_expression (x y : ℝ) : (3 * x^2 * y)^4 = 81 * x^8 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l3134_313416


namespace NUMINAMATH_CALUDE_tan_function_property_l3134_313405

theorem tan_function_property (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + π / 2))) → 
  a * Real.tan (b * π / 8) = 4 → 
  a * b = 8 := by sorry

end NUMINAMATH_CALUDE_tan_function_property_l3134_313405


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3134_313429

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

theorem geometric_sequence_ratio 
  (a₁ : ℝ) (q : ℝ) (h1 : q > 0) 
  (h2 : (geometric_sequence a₁ q 3) * (geometric_sequence a₁ q 9) = 
        2 * (geometric_sequence a₁ q 5)^2) : 
  q = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3134_313429


namespace NUMINAMATH_CALUDE_negation_of_implication_l3134_313494

theorem negation_of_implication (a b : ℝ) :
  ¬(a^2 > b^2 → a > b) ↔ (a^2 ≤ b^2 → a ≤ b) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3134_313494


namespace NUMINAMATH_CALUDE_complement_of_union_is_two_l3134_313481

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {1, 4}

-- Define set B
def B : Set ℕ := {3, 4}

-- Theorem statement
theorem complement_of_union_is_two :
  (U \ (A ∪ B)) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_two_l3134_313481


namespace NUMINAMATH_CALUDE_tan_theta_plus_pi_sixth_l3134_313454

theorem tan_theta_plus_pi_sixth (θ : Real) 
  (h1 : Real.sqrt 2 * Real.sin (θ - Real.pi/4) * Real.cos (Real.pi + θ) = Real.cos (2*θ))
  (h2 : Real.sin θ ≠ 0) : 
  Real.tan (θ + Real.pi/6) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_plus_pi_sixth_l3134_313454


namespace NUMINAMATH_CALUDE_hexagon_area_in_triangle_l3134_313479

/-- The area of a regular hexagon inscribed in a square, which is inscribed in a circle, 
    which is in turn inscribed in a triangle with side length 6 cm, is 27√3 cm². -/
theorem hexagon_area_in_triangle (s : ℝ) (h : s = 6) : 
  let r := s / 2 * Real.sqrt 3 / 3
  let square_side := 2 * r
  let hexagon_side := r
  let hexagon_area := 3 * Real.sqrt 3 / 2 * hexagon_side ^ 2
  hexagon_area = 27 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_in_triangle_l3134_313479


namespace NUMINAMATH_CALUDE_aquarium_visitors_l3134_313420

-- Define the constants
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group_size : ℕ := 10
def total_earnings : ℕ := 240

-- Define the function to calculate the number of people who only went to the aquarium
def people_only_aquarium : ℕ :=
  (total_earnings - group_size * (admission_fee + tour_fee)) / admission_fee

-- Theorem to prove
theorem aquarium_visitors :
  people_only_aquarium = 5 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_visitors_l3134_313420


namespace NUMINAMATH_CALUDE_coefficient_relation_l3134_313482

/-- A polynomial function with specific properties -/
def g (a b c d e : ℝ) (x : ℝ) : ℝ := a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- Theorem stating the relationship between coefficients a and b -/
theorem coefficient_relation (a b c d e : ℝ) :
  (g a b c d e (-1) = 0) →
  (g a b c d e 0 = 0) →
  (g a b c d e 1 = 0) →
  (g a b c d e 2 = 0) →
  (g a b c d e 0 = 3) →
  b = -2*a := by sorry

end NUMINAMATH_CALUDE_coefficient_relation_l3134_313482


namespace NUMINAMATH_CALUDE_expression_bounds_l3134_313421

theorem expression_bounds (x y z w : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
                    Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2) ∧
  Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
  Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2) ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l3134_313421


namespace NUMINAMATH_CALUDE_shortest_distance_ln_to_line_l3134_313417

open Real

theorem shortest_distance_ln_to_line (x : ℝ) : 
  let g (x : ℝ) := log x
  let P : ℝ × ℝ := (x, g x)
  let d (p : ℝ × ℝ) := |p.1 - p.2| / sqrt 2
  ∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), x > 0 → d P ≥ d (x₀, g x₀) ∧ d (x₀, g x₀) = 1 / sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_ln_to_line_l3134_313417


namespace NUMINAMATH_CALUDE_mayors_cocoa_powder_l3134_313409

theorem mayors_cocoa_powder (total_needed : ℕ) (still_needed : ℕ) (h1 : total_needed = 306) (h2 : still_needed = 47) :
  total_needed - still_needed = 259 := by
  sorry

end NUMINAMATH_CALUDE_mayors_cocoa_powder_l3134_313409


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3134_313446

theorem polynomial_factorization (a b : ℝ) : 
  (∀ x, x^2 - 3*x + a = (x - 5) * (x - b)) → (a = -10 ∧ b = -2) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3134_313446


namespace NUMINAMATH_CALUDE_robin_female_fraction_l3134_313477

theorem robin_female_fraction (total_birds : ℝ) (h1 : total_birds > 0) : 
  let robins : ℝ := (2/5) * total_birds
  let bluejays : ℝ := (3/5) * total_birds
  let female_bluejays : ℝ := (2/3) * bluejays
  let male_birds : ℝ := (7/15) * total_birds
  let female_robins : ℝ := (1/3) * robins
  female_robins + female_bluejays = total_birds - male_birds :=
by
  sorry

#check robin_female_fraction

end NUMINAMATH_CALUDE_robin_female_fraction_l3134_313477


namespace NUMINAMATH_CALUDE_max_k_value_l3134_313483

theorem max_k_value (k : ℤ) : 
  (∀ x : ℝ, x > 1 → x * Real.log x - k * x > 3) → k ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3134_313483


namespace NUMINAMATH_CALUDE_soda_cost_l3134_313451

theorem soda_cost (burger_cost soda_cost : ℚ) : 
  (3 * burger_cost + 2 * soda_cost = 360) →
  (4 * burger_cost + 3 * soda_cost = 490) →
  soda_cost = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l3134_313451


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l3134_313455

theorem rectangular_field_perimeter : 
  ∀ (width length perimeter : ℝ),
  width = 60 →
  length = (7 / 5) * width →
  perimeter = 2 * (length + width) →
  perimeter = 288 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l3134_313455


namespace NUMINAMATH_CALUDE_existence_of_even_odd_composition_l3134_313495

theorem existence_of_even_odd_composition :
  ∃ (p q : ℝ → ℝ),
    (∀ x, p x = p (-x)) ∧
    (∀ x, p (q x) = -(p (q (-x)))) ∧
    (∃ x, p (q x) ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_even_odd_composition_l3134_313495


namespace NUMINAMATH_CALUDE_power_of_power_l3134_313493

theorem power_of_power (a : ℝ) : (a^4)^3 = a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3134_313493


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3134_313406

theorem arithmetic_calculation : 15 * (1/3) + 45 * (2/3) = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3134_313406


namespace NUMINAMATH_CALUDE_smallest_with_16_divisors_l3134_313413

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a number has exactly 16 positive divisors -/
def has_16_divisors (n : ℕ+) : Prop := num_divisors n = 16

theorem smallest_with_16_divisors :
  ∀ n : ℕ+, has_16_divisors n → n ≥ 120 ∧ has_16_divisors 120 := by sorry

end NUMINAMATH_CALUDE_smallest_with_16_divisors_l3134_313413


namespace NUMINAMATH_CALUDE_colored_points_segment_existence_l3134_313496

/-- Represents a color --/
inductive Color
  | Red
  | Blue
  | Green
  | Yellow

/-- Represents a colored point on a line --/
structure ColoredPoint where
  position : ℝ
  color : Color

/-- The main theorem --/
theorem colored_points_segment_existence
  (n : ℕ)
  (h_n : n ≥ 4)
  (points : Fin n → ColoredPoint)
  (h_distinct : ∀ i j, i ≠ j → (points i).position ≠ (points j).position)
  (h_all_colors : ∀ c : Color, ∃ i, (points i).color = c) :
  ∃ (a b : ℝ), a < b ∧
    (∃ (c₁ c₂ : Color), c₁ ≠ c₂ ∧
      (∃! i, a ≤ (points i).position ∧ (points i).position ≤ b ∧ (points i).color = c₁) ∧
      (∃! j, a ≤ (points j).position ∧ (points j).position ≤ b ∧ (points j).color = c₂)) ∧
    (∃ (c₃ c₄ : Color), c₃ ≠ c₄ ∧ c₃ ≠ c₁ ∧ c₃ ≠ c₂ ∧ c₄ ≠ c₁ ∧ c₄ ≠ c₂ ∧
      (∃ i, a ≤ (points i).position ∧ (points i).position ≤ b ∧ (points i).color = c₃) ∧
      (∃ j, a ≤ (points j).position ∧ (points j).position ≤ b ∧ (points j).color = c₄)) :=
by
  sorry


end NUMINAMATH_CALUDE_colored_points_segment_existence_l3134_313496


namespace NUMINAMATH_CALUDE_hijk_is_square_l3134_313475

-- Define the points
variable (A B C D E F G H I J K : EuclideanSpace ℝ (Fin 2))

-- Define the squares
def is_square (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the midpoint
def is_midpoint (M P Q : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- State the theorem
theorem hijk_is_square 
  (h1 : is_square A B C D)
  (h2 : is_square D E F G)
  (h3 : A ≠ D ∧ B ≠ D ∧ C ≠ D ∧ E ≠ D ∧ F ≠ D ∧ G ≠ D)
  (h4 : is_midpoint H A G)
  (h5 : is_midpoint I G E)
  (h6 : is_midpoint J E C)
  (h7 : is_midpoint K C A) :
  is_square H I J K := by sorry

end NUMINAMATH_CALUDE_hijk_is_square_l3134_313475


namespace NUMINAMATH_CALUDE_min_value_and_reciprocal_sum_l3134_313402

-- Define the function f
def f (a b c x : ℝ) : ℝ := |x + a| + |x - b| + c

-- State the theorem
theorem min_value_and_reciprocal_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hmin : ∀ x, f a b c x ≥ 5) 
  (hf_attains_min : ∃ x, f a b c x = 5) : 
  a + b + c = 5 ∧ (1/a + 1/b + 1/c ≥ 9/5 ∧ ∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 1/a' + 1/b' + 1/c' = 9/5) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_reciprocal_sum_l3134_313402


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l3134_313462

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the cost of insulating a rectangular tank -/
def insulationCost (l w h costPerSqFt : ℝ) : ℝ :=
  surfaceArea l w h * costPerSqFt

/-- Proves that the cost to insulate a rectangular tank with given dimensions is $1240 -/
theorem tank_insulation_cost :
  insulationCost 5 3 2 20 = 1240 := by
  sorry

end NUMINAMATH_CALUDE_tank_insulation_cost_l3134_313462


namespace NUMINAMATH_CALUDE_exponential_function_properties_l3134_313485

/-- A function f(x) = b * a^x with specific properties -/
structure ExponentialFunction where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  a_neq_one : a ≠ 1
  f_def : ℝ → ℝ
  f_eq : ∀ x, f_def x = b * a^x
  f_at_one : f_def 1 = 27
  f_at_neg_one : f_def (-1) = 3

/-- The main theorem capturing the properties of the exponential function -/
theorem exponential_function_properties (f : ExponentialFunction) :
  (f.a = 3 ∧ f.b = 9) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ≥ 1 → f.a^x + f.b^x ≥ m) ↔ m ≤ 12) := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_properties_l3134_313485


namespace NUMINAMATH_CALUDE_pen_price_calculation_l3134_313491

theorem pen_price_calculation (num_pens num_pencils total_cost pencil_avg_price : ℝ) 
  (h1 : num_pens = 30)
  (h2 : num_pencils = 75)
  (h3 : total_cost = 750)
  (h4 : pencil_avg_price = 2) : 
  (total_cost - num_pencils * pencil_avg_price) / num_pens = 20 := by
  sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l3134_313491


namespace NUMINAMATH_CALUDE_circle_equation_l3134_313487

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (center : ℝ × ℝ), (center.1 - p.1)^2 + (center.2 - p.2)^2 = 4 ∧ 3 * center.1 - center.2 - 3 = 0}

-- Define points A and B
def point_A : ℝ × ℝ := (2, 5)
def point_B : ℝ × ℝ := (4, 3)

-- Theorem statement
theorem circle_equation : 
  (∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 - 2)^2 + (p.2 - 3)^2 = 4) ∧
  point_A ∈ circle_C ∧
  point_B ∈ circle_C :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3134_313487


namespace NUMINAMATH_CALUDE_complex_product_polar_form_l3134_313407

-- Define the cis function
noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

-- Define the problem
theorem complex_product_polar_form :
  ∃ (r θ : ℝ), 
    r > 0 ∧ 
    0 ≤ θ ∧ 
    θ < 2 * Real.pi ∧
    (4 * cis (30 * Real.pi / 180)) * (-3 * cis (45 * Real.pi / 180)) = r * cis θ ∧
    r = 12 ∧
    θ = 255 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_polar_form_l3134_313407


namespace NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l3134_313426

/-- An ellipse centered at the origin with axes aligned with the coordinate axes -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation_from_conditions 
  (e : Ellipse)
  (h_major_twice_minor : e.a = 2 * e.b)
  (h_point_on_ellipse : ellipse_equation e 4 1) :
  ∀ x y, ellipse_equation e x y ↔ x^2 / 20 + y^2 / 5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l3134_313426


namespace NUMINAMATH_CALUDE_horner_v1_value_l3134_313403

/-- Horner's method for polynomial evaluation -/
def horner_step (a : ℝ) (x : ℝ) (v : ℝ) : ℝ := v * x + a

/-- The polynomial f(x) = 0.5x^5 + 4x^4 - 3x^2 + x - 1 -/
def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

theorem horner_v1_value :
  let x : ℝ := 3
  let v0 : ℝ := 0.5
  let v1 : ℝ := horner_step v0 x 4
  v1 = 5.5 := by sorry

end NUMINAMATH_CALUDE_horner_v1_value_l3134_313403


namespace NUMINAMATH_CALUDE_ice_cube_volume_l3134_313435

theorem ice_cube_volume (V : ℝ) : 
  V > 0 → -- Assume the original volume is positive
  (1/4 * (1/4 * V)) = 0.2 → -- After two hours, the volume is 0.2 cubic inches
  V = 3.2 := by
sorry

end NUMINAMATH_CALUDE_ice_cube_volume_l3134_313435


namespace NUMINAMATH_CALUDE_largest_prime_check_l3134_313423

theorem largest_prime_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  (∀ p : ℕ, p ≤ 31 → Nat.Prime p → ¬(p ∣ n)) → Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_largest_prime_check_l3134_313423


namespace NUMINAMATH_CALUDE_pie_chart_shows_percentage_relation_l3134_313498

/-- Represents different types of statistical graphs -/
inductive StatGraph
  | PieChart
  | BarGraph
  | LineGraph
  | Histogram

/-- Defines the property of showing percentage of a part in relation to the whole -/
def shows_percentage_relation (g : StatGraph) : Prop :=
  match g with
  | StatGraph.PieChart => true
  | _ => false

/-- Theorem stating that the Pie chart is the graph that shows percentage relation -/
theorem pie_chart_shows_percentage_relation :
  ∀ (g : StatGraph), shows_percentage_relation g ↔ g = StatGraph.PieChart :=
by
  sorry

end NUMINAMATH_CALUDE_pie_chart_shows_percentage_relation_l3134_313498


namespace NUMINAMATH_CALUDE_exact_exponent_equality_l3134_313414

theorem exact_exponent_equality (n k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ m : ℕ, (2^(2^n) + 1 = p^k * m) ∧ ¬(∃ l : ℕ, 2^(2^n) + 1 = p^(k+1) * l)) →
  (∃ m : ℕ, (2^(p-1) - 1 = p^k * m) ∧ ¬(∃ l : ℕ, 2^(p-1) - 1 = p^(k+1) * l)) :=
by sorry

end NUMINAMATH_CALUDE_exact_exponent_equality_l3134_313414


namespace NUMINAMATH_CALUDE_ratio_problem_l3134_313468

theorem ratio_problem (a b : ℝ) (h1 : a / b = 150 / 1) (h2 : a = 300) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3134_313468


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3134_313484

/-- Given a hyperbola with asymptotes y = ± 2(x-1) and one focus at (1+2√5, 0),
    prove that its equation is (x - 1)²/5 - y²/20 = 1 -/
theorem hyperbola_equation 
  (asymptotes : ℝ → ℝ → Prop)
  (focus : ℝ × ℝ)
  (h_asymptotes : ∀ x y, asymptotes x y ↔ y = 2*(x-1) ∨ y = -2*(x-1))
  (h_focus : focus = (1 + 2*Real.sqrt 5, 0)) :
  ∀ x y, ((x - 1)^2 / 5 - y^2 / 20 = 1) ↔ 
    (∃ a b c : ℝ, a*(x-1)^2 + b*y^2 + c = 0 ∧ 
    (∀ x' y', asymptotes x' y' → a*(x'-1)^2 + b*y'^2 + c = 0) ∧
    a*(focus.1-1)^2 + b*focus.2^2 + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3134_313484


namespace NUMINAMATH_CALUDE_annie_hamburgers_l3134_313400

/-- Proves that Annie bought 8 hamburgers given the problem conditions -/
theorem annie_hamburgers :
  ∀ (initial_money : ℕ) (hamburger_cost : ℕ) (milkshake_cost : ℕ) 
    (milkshakes_bought : ℕ) (money_left : ℕ),
  initial_money = 132 →
  hamburger_cost = 4 →
  milkshake_cost = 5 →
  milkshakes_bought = 6 →
  money_left = 70 →
  ∃ (hamburgers_bought : ℕ),
    hamburgers_bought * hamburger_cost + milkshakes_bought * milkshake_cost = initial_money - money_left ∧
    hamburgers_bought = 8 :=
by sorry

end NUMINAMATH_CALUDE_annie_hamburgers_l3134_313400


namespace NUMINAMATH_CALUDE_min_value_n_over_2_plus_50_over_n_l3134_313441

theorem min_value_n_over_2_plus_50_over_n (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 2 + 50 / n ≥ 10 ∧
  ((n : ℝ) / 2 + 50 / n = 10 ↔ n = 10) := by
  sorry

end NUMINAMATH_CALUDE_min_value_n_over_2_plus_50_over_n_l3134_313441


namespace NUMINAMATH_CALUDE_functional_equation_implies_g_five_l3134_313422

/-- A function g: ℝ → ℝ satisfying g(xy) = g(x)g(y) for all real x and y, and g(1) = 2 -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x * y) = g x * g y) ∧ (g 1 = 2)

/-- If g satisfies the functional equation, then g(5) = 32 -/
theorem functional_equation_implies_g_five (g : ℝ → ℝ) :
  FunctionalEquation g → g 5 = 32 := by
  sorry


end NUMINAMATH_CALUDE_functional_equation_implies_g_five_l3134_313422


namespace NUMINAMATH_CALUDE_triangle_properties_l3134_313465

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the properties and maximum area of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : 4 * Real.cos t.C + Real.cos (2 * t.C) = 4 * Real.cos t.C * (Real.cos (t.C / 2))^2)
  (h2 : |t.b * Real.cos t.A - (1/2) * t.a * Real.cos t.B| = 2) : 
  t.C = π/3 ∧ 
  (∃ (S : ℝ), S ≤ 2 * Real.sqrt 3 ∧ 
    ∀ (S' : ℝ), S' = 1/2 * t.a * t.b * Real.sin t.C → S' ≤ S) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3134_313465


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3134_313449

theorem quadratic_factorization (p q : ℤ) :
  (∀ x, 20 * x^2 - 110 * x - 120 = (5 * x + p) * (4 * x + q)) →
  p + 2 * q = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3134_313449


namespace NUMINAMATH_CALUDE_square_root_2023_plus_2_squared_minus_4_times_plus_5_l3134_313442

theorem square_root_2023_plus_2_squared_minus_4_times_plus_5 :
  let m : ℝ := Real.sqrt 2023 + 2
  m^2 - 4*m + 5 = 2024 := by
sorry

end NUMINAMATH_CALUDE_square_root_2023_plus_2_squared_minus_4_times_plus_5_l3134_313442


namespace NUMINAMATH_CALUDE_wedge_product_formula_l3134_313415

/-- The wedge product of two 2D vectors -/
def wedge_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

/-- Theorem: The wedge product of two 2D vectors (a₁, a₂) and (b₁, b₂) is equal to a₁b₂ - a₂b₁ -/
theorem wedge_product_formula (a b : ℝ × ℝ) :
  wedge_product a b = a.1 * b.2 - a.2 * b.1 := by
  sorry

end NUMINAMATH_CALUDE_wedge_product_formula_l3134_313415


namespace NUMINAMATH_CALUDE_tangent_line_circle_l3134_313466

theorem tangent_line_circle (r : ℝ) (hr : r > 0) :
  (∀ x y : ℝ, x + y = r → x^2 + y^2 = r → (∀ x' y' : ℝ, x' + y' = r → x'^2 + y'^2 ≤ r)) →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_circle_l3134_313466


namespace NUMINAMATH_CALUDE_relative_error_comparison_l3134_313467

/-- Given two measurements and their respective errors, this theorem states that
    the relative error of the second measurement is less than that of the first. -/
theorem relative_error_comparison
  (measurement1 : ℝ) (error1 : ℝ) (measurement2 : ℝ) (error2 : ℝ)
  (h1 : measurement1 = 0.15)
  (h2 : error1 = 0.03)
  (h3 : measurement2 = 125)
  (h4 : error2 = 0.25)
  : error2 / measurement2 < error1 / measurement1 := by
  sorry

end NUMINAMATH_CALUDE_relative_error_comparison_l3134_313467


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3134_313472

/-- Given that (a - 3)x^(|a| - 2) + 6 = 0 is a linear equation in terms of x,
    prove that the solution is x = 1 -/
theorem linear_equation_solution (a : ℝ) :
  (∀ x, ∃ k m, (a - 3) * x^(|a| - 2) + 6 = k * x + m) →
  ∃! x, (a - 3) * x^(|a| - 2) + 6 = 0 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3134_313472


namespace NUMINAMATH_CALUDE_opposite_of_neg_2023_l3134_313410

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℤ) : ℤ := -x

/-- Theorem stating that the opposite of -2023 is 2023. -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_2023_l3134_313410


namespace NUMINAMATH_CALUDE_evelyn_winning_strategy_l3134_313432

/-- Represents a player in the game -/
inductive Player
| Odin
| Evelyn

/-- Represents the state of a box in the game -/
structure Box where
  value : ℕ
  isEmpty : Bool

/-- Represents the game state -/
structure GameState where
  boxes : List Box
  currentPlayer : Player

/-- Defines a valid move in the game -/
def isValidMove (player : Player) (oldValue newValue : ℕ) : Prop :=
  match player with
  | Player.Odin => newValue < oldValue ∧ Odd newValue
  | Player.Evelyn => newValue < oldValue ∧ Even newValue

/-- Defines the winning condition for Evelyn -/
def isEvelynWin (state : GameState) : Prop :=
  let k := state.boxes.length / 3
  (state.boxes.filter (fun b => b.value = 0)).length = k ∧
  (state.boxes.filter (fun b => b.value ≠ 0)).all (fun b => b.value = 1)

/-- Defines the winning condition for Odin -/
def isOdinWin (state : GameState) : Prop :=
  let k := state.boxes.length / 3
  (state.boxes.filter (fun b => b.value = 0)).length = k ∧
  ¬(state.boxes.filter (fun b => b.value ≠ 0)).all (fun b => b.value = 1)

/-- Theorem stating that Evelyn has a winning strategy for all k -/
theorem evelyn_winning_strategy (k : ℕ) (h : k > 0) :
  ∃ (strategy : GameState → ℕ → ℕ),
    ∀ (initialState : GameState),
      initialState.boxes.length = 3 * k →
      initialState.currentPlayer = Player.Odin →
      (initialState.boxes.all (fun b => b.isEmpty)) →
      (∃ (finalState : GameState),
        (finalState.boxes.all (fun b => ¬b.isEmpty)) ∧
        (isEvelynWin finalState ∨
         (¬∃ (move : ℕ → ℕ), isValidMove Player.Odin (move 0) (move 1)))) :=
sorry

end NUMINAMATH_CALUDE_evelyn_winning_strategy_l3134_313432


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3134_313471

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 2*a - 2 = 0) → 
  (b^3 - 2*b - 2 = 0) → 
  (c^3 - 2*c - 2 = 0) → 
  a*(b - c)^2 + b*(c - a)^2 + c*(a - b)^2 = -18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3134_313471


namespace NUMINAMATH_CALUDE_car_insurance_cost_l3134_313456

theorem car_insurance_cost (nancy_percentage : ℝ) (nancy_annual_payment : ℝ) :
  nancy_percentage = 0.40 →
  nancy_annual_payment = 384 →
  (nancy_annual_payment / nancy_percentage) / 12 = 80 := by
sorry

end NUMINAMATH_CALUDE_car_insurance_cost_l3134_313456


namespace NUMINAMATH_CALUDE_one_totally_damaged_carton_l3134_313404

/-- Represents the milk delivery problem --/
structure MilkDelivery where
  normal_cartons : ℕ
  jars_per_carton : ℕ
  cartons_shortage : ℕ
  damaged_cartons : ℕ
  damaged_jars_per_carton : ℕ
  good_jars : ℕ

/-- Calculates the number of totally damaged cartons --/
def totally_damaged_cartons (md : MilkDelivery) : ℕ :=
  let total_cartons := md.normal_cartons - md.cartons_shortage
  let total_jars := total_cartons * md.jars_per_carton
  let partially_damaged_jars := md.damaged_cartons * md.damaged_jars_per_carton
  let undamaged_jars := total_jars - partially_damaged_jars
  let additional_damaged_jars := undamaged_jars - md.good_jars
  additional_damaged_jars / md.jars_per_carton

/-- Theorem stating that the number of totally damaged cartons is 1 --/
theorem one_totally_damaged_carton (md : MilkDelivery) 
    (h1 : md.normal_cartons = 50)
    (h2 : md.jars_per_carton = 20)
    (h3 : md.cartons_shortage = 20)
    (h4 : md.damaged_cartons = 5)
    (h5 : md.damaged_jars_per_carton = 3)
    (h6 : md.good_jars = 565) :
    totally_damaged_cartons md = 1 := by
  sorry

#eval totally_damaged_cartons ⟨50, 20, 20, 5, 3, 565⟩

end NUMINAMATH_CALUDE_one_totally_damaged_carton_l3134_313404


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3134_313457

theorem polynomial_coefficient_sum (a b c d : ℤ) :
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 - 2*x^3 + 3*x^2 + 4*x - 10) →
  a + b + c + d = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3134_313457


namespace NUMINAMATH_CALUDE_digit_equality_l3134_313464

theorem digit_equality (a b c d e f : ℕ) 
  (h_a : a < 10) (h_b : b < 10) (h_c : c < 10) 
  (h_d : d < 10) (h_e : e < 10) (h_f : f < 10) :
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) -
  (100000 * f + 10000 * d + 1000 * e + 100 * b + 10 * c + a) ∣ 271 →
  b = d ∧ c = e := by
sorry

end NUMINAMATH_CALUDE_digit_equality_l3134_313464


namespace NUMINAMATH_CALUDE_ant_spider_minimum_distance_l3134_313401

/-- The minimum distance between an ant and a spider under specific conditions -/
theorem ant_spider_minimum_distance :
  let ant_position (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let spider_position (x : ℝ) : ℝ × ℝ := (2 * x - 1, 0)
  let distance (θ x : ℝ) : ℝ := Real.sqrt ((ant_position θ).1 - (spider_position x).1)^2 + ((ant_position θ).2 - (spider_position x).2)^2
  ∃ (θ : ℝ), ∀ (φ : ℝ), distance θ θ ≤ distance φ φ ∧ distance θ θ = Real.sqrt 14 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ant_spider_minimum_distance_l3134_313401


namespace NUMINAMATH_CALUDE_joan_video_game_spending_l3134_313492

/-- The cost of the basketball game Joan purchased -/
def basketball_cost : ℚ := 5.2

/-- The cost of the racing game Joan purchased -/
def racing_cost : ℚ := 4.23

/-- The total amount Joan spent on video games -/
def total_spent : ℚ := basketball_cost + racing_cost

/-- Theorem stating that the total amount Joan spent on video games is $9.43 -/
theorem joan_video_game_spending :
  total_spent = 9.43 := by sorry

end NUMINAMATH_CALUDE_joan_video_game_spending_l3134_313492


namespace NUMINAMATH_CALUDE_yellow_balls_count_l3134_313488

/-- Proves the number of yellow balls in a box given specific conditions -/
theorem yellow_balls_count (red yellow green : ℕ) : 
  red + yellow + green = 68 →
  yellow = 2 * red →
  3 * green = 4 * yellow →
  yellow = 24 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l3134_313488


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l3134_313478

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l3134_313478


namespace NUMINAMATH_CALUDE_two_lines_condition_l3134_313489

theorem two_lines_condition (m : ℝ) : 
  (∃ (a b c d : ℝ), ∀ (x y : ℝ), 
    (x^2 - m*y^2 + 2*x + 2*y = 0) ↔ ((a*x + b*y + c = 0) ∧ (a*x + b*y + d = 0))) 
  → m = 1 := by
sorry

end NUMINAMATH_CALUDE_two_lines_condition_l3134_313489


namespace NUMINAMATH_CALUDE_speed_conversion_correct_l3134_313418

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in meters per second -/
def speed_mps : ℝ := 19.445999999999998

/-- Converts speed from m/s to km/h -/
def convert_speed (s : ℝ) : ℝ := s * mps_to_kmph

theorem speed_conversion_correct : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0005 ∧ |convert_speed speed_mps - 70.006| < ε :=
sorry

end NUMINAMATH_CALUDE_speed_conversion_correct_l3134_313418


namespace NUMINAMATH_CALUDE_total_apple_and_cherry_pies_l3134_313448

def apple_pies : ℕ := 6
def pecan_pies : ℕ := 9
def pumpkin_pies : ℕ := 8
def cherry_pies : ℕ := 5
def blueberry_pies : ℕ := 3

theorem total_apple_and_cherry_pies : apple_pies + cherry_pies = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_apple_and_cherry_pies_l3134_313448


namespace NUMINAMATH_CALUDE_solution_range_l3134_313436

theorem solution_range (x : ℝ) :
  x ≥ 2 →
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 2)) = 2) →
  x ∈ Set.Icc 11 18 :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l3134_313436


namespace NUMINAMATH_CALUDE_original_fraction_l3134_313480

theorem original_fraction (x y : ℚ) : 
  (1.2 * x) / (0.9 * y) = 20 / 21 → x / y = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_l3134_313480


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3134_313474

/-- The scale factor of the map, representing km per cm -/
def scale : ℝ := 10

/-- The distance between Stockholm and Uppsala on the map in cm -/
def map_distance : ℝ := 35

/-- The actual distance between Stockholm and Uppsala in km -/
def actual_distance : ℝ := map_distance * scale

theorem stockholm_uppsala_distance : actual_distance = 350 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3134_313474


namespace NUMINAMATH_CALUDE_two_discount_equation_l3134_313499

/-- Proves the equation for a product's price after two consecutive discounts -/
theorem two_discount_equation (original_price final_price x : ℝ) :
  original_price = 400 →
  final_price = 225 →
  0 < x →
  x < 1 →
  original_price * (1 - x)^2 = final_price :=
by sorry

end NUMINAMATH_CALUDE_two_discount_equation_l3134_313499


namespace NUMINAMATH_CALUDE_justine_paper_usage_l3134_313447

theorem justine_paper_usage 
  (total_sheets : ℕ) 
  (num_binders : ℕ) 
  (sheets_per_binder : ℕ) 
  (justine_binder : ℕ) 
  (h1 : total_sheets = 2450)
  (h2 : num_binders = 5)
  (h3 : sheets_per_binder = total_sheets / num_binders)
  (h4 : justine_binder = sheets_per_binder / 2) :
  justine_binder = 245 := by
  sorry

end NUMINAMATH_CALUDE_justine_paper_usage_l3134_313447


namespace NUMINAMATH_CALUDE_power_negative_two_of_five_l3134_313438

theorem power_negative_two_of_five : 5^(-2 : ℤ) = (1 : ℚ) / 25 := by sorry

end NUMINAMATH_CALUDE_power_negative_two_of_five_l3134_313438


namespace NUMINAMATH_CALUDE_abfcde_perimeter_l3134_313461

/-- Represents a square with side length and perimeter -/
structure Square where
  side_length : ℝ
  perimeter : ℝ
  perimeter_eq : perimeter = 4 * side_length

/-- Represents the figure ABFCDE -/
structure ABFCDE where
  square : Square
  perimeter : ℝ

/-- The perimeter of ABFCDE is 80 inches, given a square with perimeter 64 inches -/
theorem abfcde_perimeter (s : Square) (fig : ABFCDE) 
  (h1 : s.perimeter = 64) 
  (h2 : fig.square = s) : 
  fig.perimeter = 80 :=
sorry

end NUMINAMATH_CALUDE_abfcde_perimeter_l3134_313461


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l3134_313439

theorem complex_arithmetic_equality : 
  908 * 501 - (731 * 1389 - (547 * 236 + 842 * 731 - 495 * 361)) = 5448 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l3134_313439


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_array_l3134_313425

/-- The number of rectangles in a square array of dots -/
def rectangles_in_array (n : ℕ) : ℕ := (n.choose 2) * (n.choose 2)

/-- Theorem: In a 5x5 square array of dots, there are 100 different rectangles 
    with sides parallel to the grid that can be formed by connecting four dots -/
theorem rectangles_in_5x5_array : rectangles_in_array 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_array_l3134_313425


namespace NUMINAMATH_CALUDE_intersection_E_F_l3134_313458

open Set Real

def E : Set ℝ := {θ | cos θ < sin θ ∧ 0 ≤ θ ∧ θ ≤ 2 * π}
def F : Set ℝ := {θ | tan θ < sin θ}

theorem intersection_E_F : E ∩ F = Ioo (π / 2) π := by
  sorry

end NUMINAMATH_CALUDE_intersection_E_F_l3134_313458


namespace NUMINAMATH_CALUDE_pump_rate_calculation_l3134_313460

/-- Given two pumps operating for a total of 6 hours, with one pump rated at 250 gallons per hour
    and used for 3.5 hours, and a total volume pumped of 1325 gallons, the rate of the other pump
    is 180 gallons per hour. -/
theorem pump_rate_calculation (total_time : ℝ) (total_volume : ℝ) (pump2_rate : ℝ) (pump2_time : ℝ)
    (h1 : total_time = 6)
    (h2 : total_volume = 1325)
    (h3 : pump2_rate = 250)
    (h4 : pump2_time = 3.5) :
    (total_volume - pump2_rate * pump2_time) / (total_time - pump2_time) = 180 :=
by sorry

end NUMINAMATH_CALUDE_pump_rate_calculation_l3134_313460


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3134_313486

/-- An arithmetic sequence with 5 terms -/
structure ArithmeticSequence5 where
  a : ℝ  -- first term
  b : ℝ  -- second term
  c : ℝ  -- third term (middle term)
  d : ℝ  -- fourth term
  e : ℝ  -- fifth term
  is_arithmetic : ∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r ∧ e = d + r

/-- The theorem stating that in an arithmetic sequence with first term 20, 
    last term 50, and middle term y, the value of y is 35 -/
theorem arithmetic_sequence_middle_term 
  (seq : ArithmeticSequence5) 
  (h1 : seq.a = 20) 
  (h2 : seq.e = 50) : 
  seq.c = 35 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3134_313486


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3134_313473

/-- The polynomial z^5 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^5 - z^3 + 1

/-- n-th roots of unity -/
def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

/-- All roots of f are n-th roots of unity -/
def all_roots_are_nth_roots_of_unity (n : ℕ) : Prop :=
  ∀ z : ℂ, f z = 0 → is_nth_root_of_unity z n

theorem smallest_n_for_roots_of_unity :
  (∃ n : ℕ, n > 0 ∧ all_roots_are_nth_roots_of_unity n) ∧
  (∀ m : ℕ, m > 0 ∧ all_roots_are_nth_roots_of_unity m → m ≥ 30) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3134_313473


namespace NUMINAMATH_CALUDE_function_composition_problem_l3134_313445

theorem function_composition_problem (a : ℝ) : 
  let f (x : ℝ) := x / 4 + 2
  let g (x : ℝ) := 5 - x
  f (g a) = 4 → a = -3 := by
sorry

end NUMINAMATH_CALUDE_function_composition_problem_l3134_313445


namespace NUMINAMATH_CALUDE_linear_regression_at_6_l3134_313419

/-- Linear regression equation -/
def linear_regression (b a x : ℝ) : ℝ := b * x + a

theorem linear_regression_at_6 (b a : ℝ) (h1 : linear_regression b a 4 = 50) (h2 : b = -2) :
  linear_regression b a 6 = 46 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_at_6_l3134_313419


namespace NUMINAMATH_CALUDE_third_root_of_cubic_l3134_313459

theorem third_root_of_cubic (a b : ℚ) (h : a ≠ 0) :
  (∃ x : ℚ, a * x^3 - (3*a - b) * x^2 + 2*(a + b) * x - (6 - 2*a) = 0) ∧
  (a * 1^3 - (3*a - b) * 1^2 + 2*(a + b) * 1 - (6 - 2*a) = 0) ∧
  (a * (-3)^3 - (3*a - b) * (-3)^2 + 2*(a + b) * (-3) - (6 - 2*a) = 0) →
  ∃ x : ℚ, x ≠ 1 ∧ x ≠ -3 ∧ a * x^3 - (3*a - b) * x^2 + 2*(a + b) * x - (6 - 2*a) = 0 ∧ x = 322/21 :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_cubic_l3134_313459


namespace NUMINAMATH_CALUDE_polynomial_equality_l3134_313470

theorem polynomial_equality (m : ℝ) : (2 * m^2 + 3 * m - 4) + (m^2 - 2 * m + 3) = 3 * m^2 + m - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3134_313470


namespace NUMINAMATH_CALUDE_train_length_l3134_313434

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 → time = 9 → speed * time * (1000 / 3600) = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l3134_313434


namespace NUMINAMATH_CALUDE_david_zachary_pushup_difference_l3134_313469

/-- Given that David did 62 push-ups and Zachary did 47 push-ups,
    prove that David did 15 more push-ups than Zachary. -/
theorem david_zachary_pushup_difference :
  let david_pushups : ℕ := 62
  let zachary_pushups : ℕ := 47
  david_pushups - zachary_pushups = 15 := by
  sorry

end NUMINAMATH_CALUDE_david_zachary_pushup_difference_l3134_313469


namespace NUMINAMATH_CALUDE_derivative_sin_pi_sixth_l3134_313476

theorem derivative_sin_pi_sixth (h : Real.sin (π / 6) = (1 : ℝ) / 2) : 
  deriv (λ _ : ℝ => Real.sin (π / 6)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_pi_sixth_l3134_313476


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_ferris_wheel_capacity_proof_l3134_313433

theorem ferris_wheel_capacity (small_seats large_seats : ℕ) 
  (large_seat_capacity : ℕ) (total_large_capacity : ℕ) : Prop :=
  small_seats = 3 ∧ 
  large_seats = 7 ∧ 
  large_seat_capacity = 12 ∧
  total_large_capacity = 84 →
  ¬∃ (small_seat_capacity : ℕ), 
    ∀ (total_capacity : ℕ), 
      total_capacity = small_seats * small_seat_capacity + total_large_capacity

theorem ferris_wheel_capacity_proof : 
  ferris_wheel_capacity 3 7 12 84 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_ferris_wheel_capacity_proof_l3134_313433


namespace NUMINAMATH_CALUDE_sine_value_given_tangent_and_point_l3134_313430

theorem sine_value_given_tangent_and_point (α : Real) (m : Real) :
  (∃ (x y : Real), x = m ∧ y = 9 ∧ x^2 + y^2 ≠ 0 ∧ Real.tan α = y / x) →
  Real.tan α = 3 / 4 →
  Real.sin α = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sine_value_given_tangent_and_point_l3134_313430


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l3134_313412

theorem mark_and_carolyn_money_sum :
  let mark_money : ℚ := 7/8
  let carolyn_money : ℚ := 2/5
  (mark_money + carolyn_money : ℚ) = 1.275 := by sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l3134_313412


namespace NUMINAMATH_CALUDE_remainder_problem_l3134_313463

theorem remainder_problem (y : ℤ) (h : y % 276 = 42) : y % 23 = 19 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3134_313463


namespace NUMINAMATH_CALUDE_larger_number_proof_l3134_313452

theorem larger_number_proof (x y : ℤ) : 
  x + y = 84 → y = x + 12 → y = 48 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3134_313452


namespace NUMINAMATH_CALUDE_discount_per_shirt_calculation_l3134_313440

theorem discount_per_shirt_calculation (num_shirts : ℕ) (total_cost discount_percentage : ℚ) 
  (h1 : num_shirts = 3)
  (h2 : total_cost = 60)
  (h3 : discount_percentage = 40/100) :
  let discount_amount := total_cost * discount_percentage
  let discounted_total := total_cost - discount_amount
  let price_per_shirt := discounted_total / num_shirts
  price_per_shirt = 12 := by
sorry

end NUMINAMATH_CALUDE_discount_per_shirt_calculation_l3134_313440


namespace NUMINAMATH_CALUDE_function_properties_l3134_313453

noncomputable def f (m n x : ℝ) : ℝ := m * x + n / x

theorem function_properties (m n : ℝ) :
  (∃ a, f m n 1 = a ∧ 3 + a - 8 = 0) →
  (m = 1 ∧ n = 4) ∧
  (∀ x, x < -2 → (deriv (f m n)) x > 0) ∧
  (∀ x, -2 < x → x < 0 → (deriv (f m n)) x < 0) ∧
  (∀ x, 0 < x → x < 2 → (deriv (f m n)) x < 0) ∧
  (∀ x, 2 < x → (deriv (f m n)) x > 0) ∧
  (∀ x, x ≠ 0 → (deriv (f m n)) x < 1) ∧
  (∀ α, (0 ≤ α ∧ α < π/4) ∨ (π/2 < α ∧ α < π) ↔ 
    ∃ x, x ≠ 0 ∧ Real.tan α = (deriv (f m n)) x) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3134_313453


namespace NUMINAMATH_CALUDE_special_numbers_are_correct_l3134_313431

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_special (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 9999 ∧
  let ab := n / 100
  let cd := n % 100
  is_perfect_square (ab - cd) ∧
  is_perfect_square (ab + cd) ∧
  (ab - cd) ∣ (ab + cd) ∧
  (ab + cd) ∣ n

def special_numbers : Finset ℕ :=
  {0100, 0400, 0900, 1600, 2500, 3600, 4900, 6400, 8100, 0504, 2016, 4536, 8064}

theorem special_numbers_are_correct :
  ∀ n : ℕ, is_special n ↔ n ∈ special_numbers := by sorry

end NUMINAMATH_CALUDE_special_numbers_are_correct_l3134_313431


namespace NUMINAMATH_CALUDE_max_m_value_l3134_313428

/-- Given m > 0 and the inequality holds for all x > 0, the maximum value of m is e^2 -/
theorem max_m_value (m : ℝ) (hm : m > 0) 
  (h : ∀ x > 0, m * x * Real.log x - (x + m) * Real.exp ((x - m) / m) ≤ 0) : 
  m ≤ Real.exp 2 ∧ ∃ m₀ > 0, ∀ ε > 0, ∃ x > 0, 
    (Real.exp 2 - ε) * x * Real.log x - (x + (Real.exp 2 - ε)) * Real.exp ((x - (Real.exp 2 - ε)) / (Real.exp 2 - ε)) > 0 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l3134_313428


namespace NUMINAMATH_CALUDE_two_digit_number_property_l3134_313424

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  units : ℕ
  tens : ℕ
  unit_constraint : units < 10
  ten_constraint : tens < 10

/-- The property that adding 18 to a number results in its reverse -/
def ReversesWhenAdd18 (n : TwoDigitNumber) : Prop :=
  n.tens * 10 + n.units + 18 = n.units * 10 + n.tens

/-- The main theorem -/
theorem two_digit_number_property (n : TwoDigitNumber) 
  (h1 : n.units + n.tens = 8) 
  (h2 : ReversesWhenAdd18 n) : 
  n.units + n.tens = 8 ∧ n.units + 10 * n.tens + 18 = 10 * n.units + n.tens := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l3134_313424


namespace NUMINAMATH_CALUDE_B_fair_share_l3134_313411

-- Define the total rent
def total_rent : ℕ := 841

-- Define the number of horses and months for each person
def horses_A : ℕ := 12
def months_A : ℕ := 8
def horses_B : ℕ := 16
def months_B : ℕ := 9
def horses_C : ℕ := 18
def months_C : ℕ := 6

-- Calculate the total horse-months
def total_horse_months : ℕ := horses_A * months_A + horses_B * months_B + horses_C * months_C

-- Calculate B's horse-months
def B_horse_months : ℕ := horses_B * months_B

-- Theorem: B's fair share of the rent is 348
theorem B_fair_share : 
  (total_rent : ℚ) * B_horse_months / total_horse_months = 348 := by
  sorry

end NUMINAMATH_CALUDE_B_fair_share_l3134_313411


namespace NUMINAMATH_CALUDE_inequality_proof_l3134_313490

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3134_313490


namespace NUMINAMATH_CALUDE_quadratic_sum_l3134_313427

/-- Given a quadratic function f(x) = 12x^2 + 144x + 1728, 
    prove that when written in the form a(x+b)^2+c, 
    the sum a+b+c equals 18. -/
theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (12 * x^2 + 144 * x + 1728 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 18) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3134_313427


namespace NUMINAMATH_CALUDE_power_function_through_point_power_function_at_4_l3134_313450

/-- A power function that passes through the point (3, √3) -/
def f (x : ℝ) : ℝ := x^(1/2)

theorem power_function_through_point : f 3 = Real.sqrt 3 := by sorry

theorem power_function_at_4 : f 4 = 2 := by sorry

end NUMINAMATH_CALUDE_power_function_through_point_power_function_at_4_l3134_313450


namespace NUMINAMATH_CALUDE_f_simplification_f_value_in_second_quadrant_l3134_313444

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (7 * Real.pi - α) * Real.cos (α + 3 * Real.pi / 2) * Real.cos (3 * Real.pi + α)) /
  (Real.sin (α - 3 * Real.pi / 2) * Real.cos (α + 5 * Real.pi / 2) * Real.tan (α - 5 * Real.pi))

theorem f_simplification (α : ℝ) : f α = Real.cos α := by sorry

theorem f_value_in_second_quadrant (α : ℝ) 
  (h1 : π < α ∧ α < 3 * π / 2) 
  (h2 : Real.cos (3 * Real.pi / 2 + α) = 1 / 7) : 
  f α = -4 * Real.sqrt 3 / 7 := by sorry

end NUMINAMATH_CALUDE_f_simplification_f_value_in_second_quadrant_l3134_313444


namespace NUMINAMATH_CALUDE_hexagon_area_sum_l3134_313437

theorem hexagon_area_sum (u v : ℤ) (hu : 0 < u) (hv : 0 < v) (huv : v < u) :
  let A : ℤ × ℤ := (u, v)
  let B : ℤ × ℤ := (v, u)
  let C : ℤ × ℤ := (-v, u)
  let D : ℤ × ℤ := (-v, -u)
  let E : ℤ × ℤ := (v, -u)
  let F : ℤ × ℤ := (-u, -v)
  let hexagon_area := 8 * u * v + |u^2 - u*v - v^2|
  hexagon_area = 802 → u + v = 27 := by
sorry

end NUMINAMATH_CALUDE_hexagon_area_sum_l3134_313437


namespace NUMINAMATH_CALUDE_power_relation_l3134_313408

theorem power_relation (x : ℝ) (n : ℕ) (h : x^(2*n) = 3) : x^(4*n) = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l3134_313408


namespace NUMINAMATH_CALUDE_sign_sum_theorem_l3134_313443

theorem sign_sum_theorem (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  let sign (x : ℝ) := x / |x|
  let expr := sign p + sign q + sign r + sign (p * q * r) + sign (p * q)
  expr = 5 ∨ expr = 1 ∨ expr = -1 :=
by sorry

end NUMINAMATH_CALUDE_sign_sum_theorem_l3134_313443
