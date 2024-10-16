import Mathlib

namespace NUMINAMATH_CALUDE_leahs_coins_value_l2179_217921

theorem leahs_coins_value :
  ∀ (p d : ℕ),
  p + d = 15 →
  p = d + 1 →
  p * 1 + d * 10 = 87 :=
by sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l2179_217921


namespace NUMINAMATH_CALUDE_discovery_uses_visualization_vr_l2179_217966

/-- Represents different digital Earth technologies -/
inductive DigitalEarthTechnology
  | InformationSuperhighway
  | HighResolutionSatellite
  | SpatialInformation
  | VisualizationAndVirtualReality

/-- Represents a TV program -/
structure TVProgram where
  name : String
  episode : String
  content : String

/-- Determines the digital Earth technology used in a TV program -/
def technology_used (program : TVProgram) : DigitalEarthTechnology :=
  if program.content = "vividly recreated various dinosaurs and their living environments"
  then DigitalEarthTechnology.VisualizationAndVirtualReality
  else DigitalEarthTechnology.InformationSuperhighway

/-- The CCTV Discovery program -/
def discovery_program : TVProgram :=
  { name := "Discovery"
  , episode := "Back to the Dinosaur Era"
  , content := "vividly recreated various dinosaurs and their living environments" }

theorem discovery_uses_visualization_vr :
  technology_used discovery_program = DigitalEarthTechnology.VisualizationAndVirtualReality := by
  sorry


end NUMINAMATH_CALUDE_discovery_uses_visualization_vr_l2179_217966


namespace NUMINAMATH_CALUDE_no_valid_n_l2179_217915

theorem no_valid_n : ¬ ∃ (n : ℕ), 
  n > 0 ∧ 
  (1000 ≤ n / 5) ∧ (n / 5 ≤ 9999) ∧ 
  (1000 ≤ 5 * n) ∧ (5 * n ≤ 9999) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_l2179_217915


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2179_217920

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 2)*x - k + 4 > 0) ↔ 
  k > -2 * Real.sqrt 3 ∧ k < 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2179_217920


namespace NUMINAMATH_CALUDE_kolya_can_prevent_divisibility_by_nine_l2179_217974

def digits : Set Nat := {1, 2, 3, 4, 5}

def alternating_sum (n : Nat) (f : Nat → Nat) : Nat :=
  List.sum (List.range n |>.map f)

theorem kolya_can_prevent_divisibility_by_nine :
  ∃ (kolya : Nat → Nat), ∀ (vasya : Nat → Nat),
    (∀ i, kolya i ∈ digits ∧ vasya i ∈ digits) →
    ¬(alternating_sum 10 kolya + alternating_sum 10 vasya) % 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_kolya_can_prevent_divisibility_by_nine_l2179_217974


namespace NUMINAMATH_CALUDE_value_of_y_l2179_217917

theorem value_of_y : ∃ y : ℝ, (3 * y) / 4 = 15 ∧ y = 20 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2179_217917


namespace NUMINAMATH_CALUDE_at_least_three_prime_factors_l2179_217961

theorem at_least_three_prime_factors (n : ℕ) 
  (h1 : n > 0) 
  (h2 : n < 200) 
  (h3 : ∃ k : ℤ, (14 * n) / 60 = k) : 
  ∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ∣ n ∧ q ∣ n ∧ r ∣ n :=
sorry

end NUMINAMATH_CALUDE_at_least_three_prime_factors_l2179_217961


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_lines_l2179_217916

-- Define the lines l1 and l2
def l1 (a x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l2 (a x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop := a * 1 + 2 * (a - 1) = 0

-- Define parallelism of lines
def parallel (a : ℝ) : Prop := a / 1 = 2 / (a - 1) ∧ a / 1 ≠ 6 / (a^2 - 1)

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) : perpendicular a → a = 2/3 :=
sorry

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : parallel a → a = -1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_lines_l2179_217916


namespace NUMINAMATH_CALUDE_cloth_coloring_problem_l2179_217931

/-- Calculates the length of cloth colored by a group of women in a given number of days -/
def clothLength (women : ℕ) (days : ℕ) (rate : ℝ) : ℝ :=
  women * days * rate

theorem cloth_coloring_problem (rate : ℝ) (h1 : rate > 0) :
  clothLength 5 1 rate = 100 →
  clothLength 6 3 rate = 360 := by
  sorry

end NUMINAMATH_CALUDE_cloth_coloring_problem_l2179_217931


namespace NUMINAMATH_CALUDE_difference_of_squares_l2179_217980

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2179_217980


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_squares_l2179_217904

theorem product_of_difference_and_sum_squares (a b : ℝ) 
  (h1 : a - b = 6) 
  (h2 : a^2 + b^2 = 48) : 
  a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_squares_l2179_217904


namespace NUMINAMATH_CALUDE_china_gdp_growth_l2179_217960

/-- China's GDP growth model from 2011 to 2016 -/
theorem china_gdp_growth (a r : ℝ) (h : a > 0) (h2 : r > 0) :
  let initial_gdp := a
  let growth_rate := r / 100
  let years := 5
  let final_gdp := initial_gdp * (1 + growth_rate) ^ years
  final_gdp = a * (1 + r / 100) ^ 5 := by sorry

end NUMINAMATH_CALUDE_china_gdp_growth_l2179_217960


namespace NUMINAMATH_CALUDE_vector_dot_product_zero_l2179_217963

theorem vector_dot_product_zero (a b : ℝ × ℝ) (h1 : a = (2, 0)) (h2 : b = (1/2, Real.sqrt 3 / 2)) :
  b • (a - b) = 0 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_zero_l2179_217963


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2179_217946

theorem pure_imaginary_fraction (m : ℝ) : 
  (∃ k : ℝ, (Complex.I : ℂ) * k = (m + Complex.I) / (1 - Complex.I)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2179_217946


namespace NUMINAMATH_CALUDE_bridget_apples_l2179_217929

theorem bridget_apples (x : ℕ) : 
  (2 * x) / 3 - 11 = 10 → x = 32 := by sorry

end NUMINAMATH_CALUDE_bridget_apples_l2179_217929


namespace NUMINAMATH_CALUDE_marble_problem_l2179_217972

theorem marble_problem (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ)
  (h1 : angela = a)
  (h2 : brian = 1.5 * a)
  (h3 : caden = 2.5 * brian)
  (h4 : daryl = 4 * caden)
  (h5 : angela + brian + caden + daryl = 90) :
  a = 72 / 17 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l2179_217972


namespace NUMINAMATH_CALUDE_sphere_volume_from_circumference_l2179_217940

/-- The volume of a sphere with circumference 30 cm is 4500/π² cm³ -/
theorem sphere_volume_from_circumference :
  ∀ (r : ℝ), 
    2 * π * r = 30 → 
    (4 / 3) * π * r ^ 3 = 4500 / π ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_circumference_l2179_217940


namespace NUMINAMATH_CALUDE_parabola_parameter_values_l2179_217936

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- Define point M
def M : ℝ × ℝ := (1, 1)

-- Define the distance from M to the directrix
def distance_to_directrix (a : ℝ) : ℝ := 2

theorem parabola_parameter_values :
  ∃ (a : ℝ), (parabola a (M.1) = M.2) ∧ 
             (distance_to_directrix a = 2) ∧ 
             (a = 1/4 ∨ a = -1/12) :=
by sorry

end NUMINAMATH_CALUDE_parabola_parameter_values_l2179_217936


namespace NUMINAMATH_CALUDE_range_of_s_is_composite_positive_integers_l2179_217906

-- Define the set of composite positive integers
def CompositePositiveIntegers : Set ℕ := {n : ℕ | n > 1 ∧ ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b}

-- Define the function s
def s (n : ℕ) : ℕ := n

-- State the theorem
theorem range_of_s_is_composite_positive_integers :
  {s n | n ∈ CompositePositiveIntegers} = CompositePositiveIntegers := by
  sorry


end NUMINAMATH_CALUDE_range_of_s_is_composite_positive_integers_l2179_217906


namespace NUMINAMATH_CALUDE_chocolate_price_in_first_store_l2179_217953

def chocolates_per_week : ℕ := 2
def weeks : ℕ := 3
def promotion_price : ℚ := 2
def savings : ℚ := 6

theorem chocolate_price_in_first_store :
  let total_chocolates := chocolates_per_week * weeks
  let promotion_total := total_chocolates * promotion_price
  let first_store_total := promotion_total + savings
  first_store_total / total_chocolates = 3 := by
sorry

end NUMINAMATH_CALUDE_chocolate_price_in_first_store_l2179_217953


namespace NUMINAMATH_CALUDE_complex_power_result_l2179_217993

theorem complex_power_result : 
  (3 * (Complex.cos (Real.pi / 6) + Complex.I * Complex.sin (Real.pi / 6)))^8 = 
  Complex.mk (-3280.5) (-3280.5 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_complex_power_result_l2179_217993


namespace NUMINAMATH_CALUDE_f1_extrema_f2_extrema_l2179_217976

-- Function 1
def f1 (x : ℝ) : ℝ := x^3 + 2*x

theorem f1_extrema :
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f1 x ≥ f1 x₁) ∧
  (∃ x₂ ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f1 x ≤ f1 x₂) ∧
  (f1 x₁ = -3) ∧ (f1 x₂ = 3) :=
sorry

-- Function 2
def f2 (x : ℝ) : ℝ := (x - 1) * (x - 2)^2

theorem f2_extrema :
  (∃ x₁ ∈ Set.Icc 0 3, ∀ x ∈ Set.Icc 0 3, f2 x ≥ f2 x₁) ∧
  (∃ x₂ ∈ Set.Icc 0 3, ∀ x ∈ Set.Icc 0 3, f2 x ≤ f2 x₂) ∧
  (f2 x₁ = -4) ∧ (f2 x₂ = 2) :=
sorry

end NUMINAMATH_CALUDE_f1_extrema_f2_extrema_l2179_217976


namespace NUMINAMATH_CALUDE_multiplication_subtraction_difference_l2179_217938

theorem multiplication_subtraction_difference : ∀ x : ℤ, 
  x = 11 → (3 * x) - (26 - x) = 18 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_difference_l2179_217938


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2179_217992

theorem quadratic_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2179_217992


namespace NUMINAMATH_CALUDE_sharpen_nine_knives_cost_l2179_217909

/-- Calculates the cost of sharpening knives based on a specific pricing structure -/
def sharpeningCost (n : ℕ) : ℚ :=
  let firstKnifeCost : ℚ := 5
  let nextThreeKnivesCost : ℚ := 4
  let remainingKnivesCost : ℚ := 3
  
  let firstKnife := min n 1
  let nextThreeKnives := min (n - 1) 3
  let remainingKnives := max (n - 4) 0

  firstKnife * firstKnifeCost +
  nextThreeKnives * nextThreeKnivesCost +
  remainingKnives * remainingKnivesCost

/-- Theorem stating that the cost of sharpening 9 knives is $32.00 -/
theorem sharpen_nine_knives_cost :
  sharpeningCost 9 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sharpen_nine_knives_cost_l2179_217909


namespace NUMINAMATH_CALUDE_exists_natural_fifth_fourth_power_l2179_217954

theorem exists_natural_fifth_fourth_power : ∃ n : ℕ, n > 1 ∧ ∃ m : ℕ, n^(5/4) = m := by
  sorry

end NUMINAMATH_CALUDE_exists_natural_fifth_fourth_power_l2179_217954


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2179_217927

theorem complex_arithmetic_equality : 
  -1^10 - (13/14 - 11/12) * (4 - (-2)^2) + 1/2 / 3 = -5/6 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2179_217927


namespace NUMINAMATH_CALUDE_prob_white_ball_l2179_217973

/-- Probability of drawing a white ball from a box with inaccessible balls -/
theorem prob_white_ball (total : ℕ) (white : ℕ) (locked : ℕ) : 
  total = 17 → white = 7 → locked = 3 → 
  (white : ℚ) / (total - locked : ℚ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_prob_white_ball_l2179_217973


namespace NUMINAMATH_CALUDE_division_theorem_l2179_217962

-- Define the dividend polynomial
def dividend (x : ℝ) : ℝ := x^5 + 2*x^3 + x^2 + 3

-- Define the divisor polynomial
def divisor (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Define the quotient polynomial
def quotient (x : ℝ) : ℝ := x^3 + 4*x^2 + 12*x

-- Define the remainder polynomial
def remainder (x : ℝ) : ℝ := 25*x^2 - 72*x + 3

-- Theorem statement
theorem division_theorem :
  ∀ x : ℝ, dividend x = divisor x * quotient x + remainder x :=
by sorry

end NUMINAMATH_CALUDE_division_theorem_l2179_217962


namespace NUMINAMATH_CALUDE_min_omega_value_l2179_217971

theorem min_omega_value (ω : Real) (x₁ x₂ : Real) :
  ω > 0 →
  (fun x ↦ Real.sin (ω * x + π / 3) + Real.sin (ω * x)) x₁ = 0 →
  (fun x ↦ Real.sin (ω * x + π / 3) + Real.sin (ω * x)) x₂ = Real.sqrt 3 →
  |x₁ - x₂| = π →
  ∃ (ω_min : Real), ω_min = 1/2 ∧ ∀ (ω' : Real), ω' > 0 ∧
    (∃ (y₁ y₂ : Real), 
      (fun x ↦ Real.sin (ω' * x + π / 3) + Real.sin (ω' * x)) y₁ = 0 ∧
      (fun x ↦ Real.sin (ω' * x + π / 3) + Real.sin (ω' * x)) y₂ = Real.sqrt 3 ∧
      |y₁ - y₂| = π) →
    ω' ≥ ω_min :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l2179_217971


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2179_217999

theorem trigonometric_simplification (α : ℝ) : 
  Real.sin (π / 2 + α) * Real.cos (α - π / 3) + Real.sin (π - α) * Real.sin (α - π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2179_217999


namespace NUMINAMATH_CALUDE_subset_equivalence_l2179_217937

theorem subset_equivalence (φ A : Set α) (p q : Prop) :
  (φ ⊆ A ↔ (φ = A ∨ φ ⊂ A)) →
  (φ ⊆ A ↔ p ∨ q) :=
by sorry

end NUMINAMATH_CALUDE_subset_equivalence_l2179_217937


namespace NUMINAMATH_CALUDE_math_city_intersections_l2179_217908

/-- Represents a city with streets and tunnels -/
structure City where
  num_streets : ℕ
  num_tunnels : ℕ

/-- Calculates the maximum number of intersections in a city -/
def max_intersections (c : City) : ℕ :=
  (c.num_streets.choose 2) - c.num_tunnels

/-- Theorem stating the maximum number of intersections in Math City -/
theorem math_city_intersections :
  let math_city : City := { num_streets := 10, num_tunnels := 2 }
  max_intersections math_city = 43 := by
  sorry

end NUMINAMATH_CALUDE_math_city_intersections_l2179_217908


namespace NUMINAMATH_CALUDE_line_equation_from_parabola_intersections_l2179_217970

/-- Given a parabola y^2 = 2x and a point G, prove that the line AB formed by
    the intersection of two lines from G to the parabola has a specific equation. -/
theorem line_equation_from_parabola_intersections
  (G : ℝ × ℝ)
  (k₁ k₂ : ℝ)
  (h_G : G = (2, 2))
  (h_parabola : ∀ x y, y^2 = 2*x → (∃ A B : ℝ × ℝ, 
    (A.1 = x ∧ A.2 = y) ∨ (B.1 = x ∧ B.2 = y)))
  (h_slopes : ∀ A B : ℝ × ℝ, 
    (A.2^2 = 2*A.1 ∧ B.2^2 = 2*B.1) → 
    k₁ = (A.2 - G.2) / (A.1 - G.1) ∧
    k₂ = (B.2 - G.2) / (B.1 - G.1))
  (h_sum : k₁ + k₂ = 5)
  (h_product : k₁ * k₂ = -2) :
  ∃ A B : ℝ × ℝ, 2 * A.1 + 9 * A.2 + 12 = 0 ∧
                 2 * B.1 + 9 * B.2 + 12 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_parabola_intersections_l2179_217970


namespace NUMINAMATH_CALUDE_equation_solutions_l2179_217944

theorem equation_solutions : 
  let solutions := {x : ℝ | (x - 1)^2 = 4}
  solutions = {3, -1} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2179_217944


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_problem_l2179_217910

def f (s : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c
  where
    a := -(2*s + 10)
    b := (s + 2)*(s + 8) + (s + 2)*a + (s + 8)*a
    c := -((s + 2)*(s + 8)*a + (s + 2)*(s + 8))

def g (s : ℝ) (x : ℝ) : ℝ := x^3 + d*x^2 + e*x + h
  where
    d := -(2*s + 16)
    e := (s + 5)*(s + 11) + (s + 5)*d + (s + 11)*d
    h := -((s + 5)*(s + 11)*d + (s + 5)*(s + 11))

theorem cubic_polynomial_roots_problem (s : ℝ) :
  (∀ x, f s x - g s x = 2*s) →
  s = 81/4 := by
  sorry


end NUMINAMATH_CALUDE_cubic_polynomial_roots_problem_l2179_217910


namespace NUMINAMATH_CALUDE_kid_tickets_sold_l2179_217990

theorem kid_tickets_sold (adult_price kid_price total_tickets total_profit : ℕ) 
  (h1 : adult_price = 12)
  (h2 : kid_price = 5)
  (h3 : total_tickets = 275)
  (h4 : total_profit = 2150) :
  ∃ (adult_tickets kid_tickets : ℕ),
    adult_tickets + kid_tickets = total_tickets ∧
    adult_price * adult_tickets + kid_price * kid_tickets = total_profit ∧
    kid_tickets = 164 := by
  sorry

end NUMINAMATH_CALUDE_kid_tickets_sold_l2179_217990


namespace NUMINAMATH_CALUDE_reflection_of_point_across_x_axis_l2179_217969

/-- Represents a point in the 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Theorem: The reflection of the point P(-3,1) across the x-axis is (-3,-1) -/
theorem reflection_of_point_across_x_axis :
  let P : Point2D := { x := -3, y := 1 }
  reflectAcrossXAxis P = { x := -3, y := -1 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_point_across_x_axis_l2179_217969


namespace NUMINAMATH_CALUDE_remainder_problem_l2179_217955

theorem remainder_problem (n : ℤ) (h : n % 7 = 2) : (5 * n + 3) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2179_217955


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2179_217968

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x ≥ 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (Set.compl B) = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2179_217968


namespace NUMINAMATH_CALUDE_max_income_at_11_l2179_217975

/-- Represents the number of bicycles available for rent -/
def total_bicycles : ℕ := 50

/-- Represents the daily management cost in yuan -/
def management_cost : ℕ := 115

/-- Calculates the number of bicycles rented based on the price -/
def bicycles_rented (price : ℕ) : ℕ :=
  if price ≤ 6 then total_bicycles
  else max (total_bicycles - 3 * (price - 6)) 0

/-- Calculates the net income based on the rental price -/
def net_income (price : ℕ) : ℤ :=
  (price * bicycles_rented price : ℤ) - management_cost

/-- The domain of valid rental prices -/
def valid_price (price : ℕ) : Prop :=
  3 ≤ price ∧ price ≤ 20 ∧ net_income price > 0

theorem max_income_at_11 :
  ∀ price, valid_price price →
    net_income price ≤ net_income 11 :=
  sorry

end NUMINAMATH_CALUDE_max_income_at_11_l2179_217975


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_solve_linear_equation_l2179_217934

-- Equation 1
theorem solve_quadratic_equation (x : ℝ) :
  2 * x^2 - 5 * x + 1 = 0 ↔ x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4 :=
sorry

-- Equation 2
theorem solve_linear_equation (x : ℝ) :
  3 * x * (x - 2) = 2 * (2 - x) ↔ x = 2 ∨ x = -2/3 :=
sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_solve_linear_equation_l2179_217934


namespace NUMINAMATH_CALUDE_total_frisbees_sold_l2179_217995

/-- Represents the number of frisbees sold at $3 -/
def x : ℕ := sorry

/-- Represents the number of frisbees sold at $4 -/
def y : ℕ := sorry

/-- The total receipts from frisbee sales is $200 -/
axiom total_sales : 3 * x + 4 * y = 200

/-- The fewest number of $4 frisbees sold is 8 -/
axiom min_four_dollar_frisbees : y ≥ 8

/-- The total number of frisbees sold -/
def total_frisbees : ℕ := x + y

theorem total_frisbees_sold : total_frisbees = 64 := by sorry

end NUMINAMATH_CALUDE_total_frisbees_sold_l2179_217995


namespace NUMINAMATH_CALUDE_initial_birds_count_l2179_217907

theorem initial_birds_count (initial_birds additional_birds total_birds : ℕ) 
  (h1 : additional_birds = 13)
  (h2 : total_birds = 42)
  (h3 : initial_birds + additional_birds = total_birds) : 
  initial_birds = 29 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_count_l2179_217907


namespace NUMINAMATH_CALUDE_at_op_difference_l2179_217997

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y + x - y

-- State the theorem
theorem at_op_difference : at_op 7 4 - at_op 4 7 = 6 := by sorry

end NUMINAMATH_CALUDE_at_op_difference_l2179_217997


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_union_equals_A_iff_l2179_217928

-- Define sets A and B
def A : Set ℝ := {x | -3 ≤ x - 2 ∧ x - 2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

-- Theorem for part I
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x : ℝ | 0 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for part II
theorem union_equals_A_iff (a : ℝ) :
  A ∪ B a = A ↔ 0 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_union_equals_A_iff_l2179_217928


namespace NUMINAMATH_CALUDE_college_student_count_l2179_217957

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- The ratio of boys to girls is 5:7 -/
def ratio_condition (c : College) : Prop :=
  7 * c.boys = 5 * c.girls

/-- There are 140 girls -/
def girls_count (c : College) : Prop :=
  c.girls = 140

/-- The total number of students -/
def total_students (c : College) : ℕ :=
  c.boys + c.girls

/-- Theorem stating the total number of students in the college -/
theorem college_student_count (c : College) 
  (h1 : ratio_condition c) (h2 : girls_count c) : 
  total_students c = 240 := by
  sorry

end NUMINAMATH_CALUDE_college_student_count_l2179_217957


namespace NUMINAMATH_CALUDE_speed_of_current_l2179_217923

/-- 
Given a man's speed with and against a current, this theorem proves 
the speed of the current.
-/
theorem speed_of_current 
  (speed_with_current : ℝ) 
  (speed_against_current : ℝ) 
  (h1 : speed_with_current = 20) 
  (h2 : speed_against_current = 18) : 
  ∃ (current_speed : ℝ), current_speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_speed_of_current_l2179_217923


namespace NUMINAMATH_CALUDE_equal_discriminants_l2179_217926

/-- A monic quadratic polynomial with distinct roots -/
structure MonicQuadratic where
  a : ℝ
  b : ℝ
  distinct_roots : a ≠ b

/-- The value of a monic quadratic polynomial at a given point -/
def evaluate (p : MonicQuadratic) (x : ℝ) : ℝ :=
  (x - p.a) * (x - p.b)

/-- The discriminant of a monic quadratic polynomial -/
def discriminant (p : MonicQuadratic) : ℝ :=
  (p.a - p.b)^2

theorem equal_discriminants (P Q : MonicQuadratic)
  (h : evaluate Q P.a + evaluate Q P.b = evaluate P Q.a + evaluate P Q.b) :
  discriminant P = discriminant Q := by
  sorry

end NUMINAMATH_CALUDE_equal_discriminants_l2179_217926


namespace NUMINAMATH_CALUDE_target_hit_probability_l2179_217977

theorem target_hit_probability (p1 p2 : ℝ) (h1 : p1 = 1/2) (h2 : p2 = 1/3) :
  1 - (1 - p1) * (1 - p2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l2179_217977


namespace NUMINAMATH_CALUDE_july_birth_percentage_l2179_217942

theorem july_birth_percentage (total : ℕ) (july_births : ℕ) 
  (h1 : total = 120) (h2 : july_births = 16) : 
  (july_births : ℝ) / total * 100 = 13.33 := by
  sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l2179_217942


namespace NUMINAMATH_CALUDE_units_digit_of_2_power_10_l2179_217947

theorem units_digit_of_2_power_10 : (2^10 : ℕ) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_2_power_10_l2179_217947


namespace NUMINAMATH_CALUDE_hyunseung_outfit_combinations_l2179_217903

/-- The number of types of tops in Hyunseung's closet -/
def num_tops : ℕ := 3

/-- The number of types of bottoms in Hyunseung's closet -/
def num_bottoms : ℕ := 2

/-- The number of types of shoes in Hyunseung's closet -/
def num_shoes : ℕ := 5

/-- The total number of combinations of tops, bottoms, and shoes Hyunseung can wear -/
def total_combinations : ℕ := num_tops * num_bottoms * num_shoes

theorem hyunseung_outfit_combinations : total_combinations = 30 := by
  sorry

end NUMINAMATH_CALUDE_hyunseung_outfit_combinations_l2179_217903


namespace NUMINAMATH_CALUDE_double_reflection_F_l2179_217982

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point F -/
def F : ℝ × ℝ := (-2, -3)

theorem double_reflection_F :
  (reflect_x (reflect_y F)) = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_F_l2179_217982


namespace NUMINAMATH_CALUDE_lizzy_initial_money_l2179_217964

def loan_amount : ℝ := 15
def interest_rate : ℝ := 0.20
def final_amount : ℝ := 33

theorem lizzy_initial_money :
  ∃ (initial_money : ℝ),
    initial_money = loan_amount ∧
    final_amount = initial_money + loan_amount + (interest_rate * loan_amount) :=
by sorry

end NUMINAMATH_CALUDE_lizzy_initial_money_l2179_217964


namespace NUMINAMATH_CALUDE_cube_root_of_a_minus_m_l2179_217956

theorem cube_root_of_a_minus_m (a m : ℝ) (ha : 0 < a) 
  (h1 : (m + 7)^2 = a) (h2 : (2*m - 1)^2 = a) : 
  (a - m)^(1/3 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_a_minus_m_l2179_217956


namespace NUMINAMATH_CALUDE_roller_coaster_cost_proof_l2179_217922

/-- The cost of the Ferris wheel in tickets -/
def ferris_wheel_cost : ℕ := 6

/-- The cost of the log ride in tickets -/
def log_ride_cost : ℕ := 7

/-- The number of tickets Antonieta initially has -/
def initial_tickets : ℕ := 2

/-- The number of additional tickets Antonieta needs to buy -/
def additional_tickets : ℕ := 16

/-- The cost of the roller coaster in tickets -/
def roller_coaster_cost : ℕ := 5

theorem roller_coaster_cost_proof :
  roller_coaster_cost = 
    (initial_tickets + additional_tickets) - (ferris_wheel_cost + log_ride_cost) :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_cost_proof_l2179_217922


namespace NUMINAMATH_CALUDE_symmetric_points_product_l2179_217924

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The problem statement -/
theorem symmetric_points_product (a b : ℝ) 
    (h : symmetric_wrt_origin (a + 2) 2 4 (-b)) : a * b = -12 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_product_l2179_217924


namespace NUMINAMATH_CALUDE_binomial_equation_solutions_l2179_217991

theorem binomial_equation_solutions :
  ∀ m r : ℕ, 2014 ≥ m → m ≥ r → r ≥ 1 →
  (Nat.choose 2014 m + Nat.choose m r = Nat.choose 2014 r + Nat.choose (2014 - r) (m - r)) ↔
  ((m = r ∧ m ≤ 2014) ∨
   (m = 2014 - r ∧ r ≤ 1006) ∨
   (m = 2014 ∧ r ≤ 2013)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_equation_solutions_l2179_217991


namespace NUMINAMATH_CALUDE_units_digit_of_7_451_l2179_217932

theorem units_digit_of_7_451 : (7^451) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_451_l2179_217932


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2179_217994

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^5 - x^4 + 2 * x^3 + 5 * x^2 - 3 * x + 7) + 
  (-x^5 + 4 * x^4 + x^3 - 6 * x^2 + 5 * x - 4) - 
  (2 * x^5 - 3 * x^4 + 4 * x^3 - x^2 - x + 2) = 
  6 * x^4 - x^3 + 3 * x + 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2179_217994


namespace NUMINAMATH_CALUDE_system_solution_l2179_217958

theorem system_solution :
  ∀ x y : ℝ, x^2 + y^2 = 13 ∧ x * y = 6 → x = 3 ∧ y = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2179_217958


namespace NUMINAMATH_CALUDE_fraction_simplification_l2179_217911

theorem fraction_simplification (x : ℝ) (h : x ≠ 0) :
  ((x + 3)^2 + (x + 3)*(x - 3)) / (2*x) = x + 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2179_217911


namespace NUMINAMATH_CALUDE_first_term_value_l2179_217914

/-- A geometric sequence with five terms -/
def GeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ 36 = b * r ∧ c = 36 * r ∧ 144 = c * r

/-- The first term of the geometric sequence is 9/4 -/
theorem first_term_value (a b c : ℝ) (h : GeometricSequence a b c) : a = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_first_term_value_l2179_217914


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l2179_217905

theorem rectangle_area_proof : ∃ (x y : ℚ), 
  (x - (7/2)) * (y + (3/2)) = x * y ∧ 
  (x + (7/2)) * (y - (5/2)) = x * y ∧ 
  x * y = 20/7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l2179_217905


namespace NUMINAMATH_CALUDE_john_needs_72_strings_l2179_217996

/-- The number of strings John needs to restring all instruments -/
def total_strings (num_basses : ℕ) (strings_per_bass : ℕ) (strings_per_guitar : ℕ) (strings_per_8string_guitar : ℕ) : ℕ :=
  let num_guitars := 2 * num_basses
  let num_8string_guitars := num_guitars - 3
  num_basses * strings_per_bass + num_guitars * strings_per_guitar + num_8string_guitars * strings_per_8string_guitar

/-- Theorem stating the total number of strings John needs -/
theorem john_needs_72_strings :
  total_strings 3 4 6 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_72_strings_l2179_217996


namespace NUMINAMATH_CALUDE_parabola_unique_coefficients_l2179_217918

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the parabola at a given x -/
def Parabola.eval (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Calculates the derivative of the parabola at a given x -/
def Parabola.derivative (p : Parabola) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

theorem parabola_unique_coefficients :
  ∀ p : Parabola,
    p.eval 1 = 1 →                        -- Parabola passes through (1, 1)
    p.eval 2 = -1 →                       -- Parabola passes through (2, -1)
    p.derivative 2 = 1 →                  -- Tangent line at (2, -1) has slope 1
    p.a = 3 ∧ p.b = -11 ∧ p.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_unique_coefficients_l2179_217918


namespace NUMINAMATH_CALUDE_marathon_remainder_yards_l2179_217965

/-- The length of a marathon in miles -/
def marathon_miles : ℕ := 28

/-- The additional yards in a marathon beyond the whole miles -/
def marathon_extra_yards : ℕ := 1500

/-- The number of yards in a mile -/
def yards_per_mile : ℕ := 1760

/-- The number of marathons run -/
def marathons_run : ℕ := 15

/-- The total number of yards run in all marathons -/
def total_yards : ℕ := marathons_run * (marathon_miles * yards_per_mile + marathon_extra_yards)

/-- The remainder of yards after converting total yards to miles -/
def remainder_yards : ℕ := total_yards % yards_per_mile

theorem marathon_remainder_yards : remainder_yards = 1200 := by
  sorry

end NUMINAMATH_CALUDE_marathon_remainder_yards_l2179_217965


namespace NUMINAMATH_CALUDE_no_even_integers_satisfying_conditions_l2179_217983

theorem no_even_integers_satisfying_conditions : 
  ¬ ∃ (n : ℤ), 
    (n % 2 = 0) ∧ 
    (100 ≤ n) ∧ (n ≤ 1000) ∧ 
    (∃ (k : ℕ), n = 3 * k + 4) ∧ 
    (∃ (m : ℕ), n = 5 * m + 2) := by
  sorry

end NUMINAMATH_CALUDE_no_even_integers_satisfying_conditions_l2179_217983


namespace NUMINAMATH_CALUDE_max_value_interval_l2179_217939

open Real

-- Define the function f(x) = 4x³ - 3x
def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x

-- State the theorem
theorem max_value_interval (a : ℝ) :
  (∃ c ∈ Set.Ioo a (a + 2), ∀ x ∈ Set.Ioo a (a + 2), f x ≤ f c) →
  a ∈ Set.Ioo (-5/2) (-1/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_interval_l2179_217939


namespace NUMINAMATH_CALUDE_plants_eaten_third_day_l2179_217951

theorem plants_eaten_third_day 
  (initial_plants : ℕ)
  (eaten_first_day : ℕ)
  (fraction_eaten_second_day : ℚ)
  (final_plants : ℕ)
  (h1 : initial_plants = 30)
  (h2 : eaten_first_day = 20)
  (h3 : fraction_eaten_second_day = 1/2)
  (h4 : final_plants = 4)
  : initial_plants - eaten_first_day - 
    (initial_plants - eaten_first_day) * fraction_eaten_second_day - 
    final_plants = 1 := by
  sorry

end NUMINAMATH_CALUDE_plants_eaten_third_day_l2179_217951


namespace NUMINAMATH_CALUDE_avery_donation_l2179_217941

theorem avery_donation (shirts : ℕ) 
  (h1 : shirts + 2 * shirts + shirts = 16) : shirts = 4 := by
  sorry

end NUMINAMATH_CALUDE_avery_donation_l2179_217941


namespace NUMINAMATH_CALUDE_equal_roots_implies_value_l2179_217901

/-- If x^2 + 2kx + k^2 + k + 3 = 0 has two equal real roots with respect to x,
    then k^2 + k + 3 = 9 -/
theorem equal_roots_implies_value (k : ℝ) :
  (∃ x : ℝ, x^2 + 2*k*x + k^2 + k + 3 = 0 ∧
   ∀ y : ℝ, y^2 + 2*k*y + k^2 + k + 3 = 0 → y = x) →
  k^2 + k + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_implies_value_l2179_217901


namespace NUMINAMATH_CALUDE_sandbox_capacity_doubled_l2179_217949

/-- Represents the dimensions and capacity of a sandbox -/
structure Sandbox where
  length : ℝ
  width : ℝ
  height : ℝ
  capacity : ℝ

/-- Theorem: Doubling the dimensions of a sandbox increases its capacity by a factor of 8 -/
theorem sandbox_capacity_doubled (original : Sandbox) 
  (h_original_capacity : original.capacity = 10) :
  let new_sandbox := Sandbox.mk 
    (2 * original.length) 
    (2 * original.width) 
    (2 * original.height) 
    ((2 * original.length) * (2 * original.width) * (2 * original.height))
  new_sandbox.capacity = 80 := by
  sorry


end NUMINAMATH_CALUDE_sandbox_capacity_doubled_l2179_217949


namespace NUMINAMATH_CALUDE_somu_father_age_ratio_l2179_217900

/-- Represents the ages of Somu and his father -/
structure Ages where
  somu : ℕ
  father : ℕ

/-- The condition that Somu's age 10 years ago was one-fifth of his father's age 10 years ago -/
def age_condition (ages : Ages) : Prop :=
  ages.somu - 10 = (ages.father - 10) / 5

/-- The theorem stating that given Somu's present age is 20 and the age condition,
    the ratio of Somu's present age to his father's present age is 1:3 -/
theorem somu_father_age_ratio (ages : Ages) 
    (h1 : ages.somu = 20) 
    (h2 : age_condition ages) : 
    (ages.somu : ℚ) / ages.father = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_somu_father_age_ratio_l2179_217900


namespace NUMINAMATH_CALUDE_fixed_point_transformation_l2179_217950

theorem fixed_point_transformation (f : ℝ → ℝ) (h : f 1 = 1) : f (4 - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_transformation_l2179_217950


namespace NUMINAMATH_CALUDE_orange_tree_problem_l2179_217902

theorem orange_tree_problem (total_trees : ℕ) (tree_a_percent : ℚ) (tree_b_percent : ℚ)
  (tree_b_oranges : ℕ) (tree_b_good_ratio : ℚ) (tree_a_good_ratio : ℚ) (total_good_oranges : ℕ) :
  tree_a_percent = 1/2 →
  tree_b_percent = 1/2 →
  tree_b_oranges = 15 →
  tree_b_good_ratio = 1/3 →
  tree_a_good_ratio = 3/5 →
  total_trees = 10 →
  total_good_oranges = 55 →
  ∃ (tree_a_oranges : ℕ), 
    (tree_a_percent * total_trees : ℚ) * (tree_a_oranges : ℚ) * tree_a_good_ratio +
    (tree_b_percent * total_trees : ℚ) * (tree_b_oranges : ℚ) * tree_b_good_ratio =
    total_good_oranges ∧
    tree_a_oranges = 10 :=
by sorry

end NUMINAMATH_CALUDE_orange_tree_problem_l2179_217902


namespace NUMINAMATH_CALUDE_previous_year_300th_day_is_monday_l2179_217988

/-- Represents days of the week -/
inductive DayOfWeek
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Represents a year -/
structure Year where
  isLeapYear : Bool

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.monday => DayOfWeek.tuesday
  | DayOfWeek.tuesday => DayOfWeek.wednesday
  | DayOfWeek.wednesday => DayOfWeek.thursday
  | DayOfWeek.thursday => DayOfWeek.friday
  | DayOfWeek.friday => DayOfWeek.saturday
  | DayOfWeek.saturday => DayOfWeek.sunday
  | DayOfWeek.sunday => DayOfWeek.monday

/-- Calculates the day of the week after a given number of days -/
def advanceDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => advanceDays (nextDay start) n

/-- Main theorem -/
theorem previous_year_300th_day_is_monday 
  (currentYear : Year)
  (nextYear : Year)
  (h1 : advanceDays DayOfWeek.sunday 200 = DayOfWeek.sunday)
  (h2 : advanceDays DayOfWeek.sunday 100 = DayOfWeek.sunday) :
  advanceDays DayOfWeek.monday 300 = DayOfWeek.sunday :=
sorry

end NUMINAMATH_CALUDE_previous_year_300th_day_is_monday_l2179_217988


namespace NUMINAMATH_CALUDE_tonya_stamps_after_trade_l2179_217981

/-- Represents the trade of matchbooks for stamps between Jimmy and Tonya --/
def matchbook_stamp_trade (stamp_match_ratio : ℕ) (matches_per_book : ℕ) (tonya_initial_stamps : ℕ) (jimmy_matchbooks : ℕ) : ℕ :=
  let jimmy_total_matches := jimmy_matchbooks * matches_per_book
  let jimmy_stamps_worth := jimmy_total_matches / stamp_match_ratio
  tonya_initial_stamps - jimmy_stamps_worth

/-- Theorem stating that Tonya will have 3 stamps left after the trade --/
theorem tonya_stamps_after_trade :
  matchbook_stamp_trade 12 24 13 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tonya_stamps_after_trade_l2179_217981


namespace NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l2179_217984

-- Define the equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + m = 0

-- Define what it means for the equation to represent a circle
def is_circle (m : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y m ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem stating that m = 0 is sufficient but not necessary
theorem m_zero_sufficient_not_necessary :
  (is_circle 0) ∧ (∃ m : ℝ, m ≠ 0 ∧ is_circle m) :=
sorry

end NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l2179_217984


namespace NUMINAMATH_CALUDE_no_solution_for_qt_plus_q_plus_t_eq_6_l2179_217930

theorem no_solution_for_qt_plus_q_plus_t_eq_6 :
  ∀ (q t : ℕ), q > 0 ∧ t > 0 → q * t + q + t ≠ 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_qt_plus_q_plus_t_eq_6_l2179_217930


namespace NUMINAMATH_CALUDE_roof_shingle_width_l2179_217979

/-- The width of a rectangular roof shingle with length 10 inches and area 70 square inches is 7 inches. -/
theorem roof_shingle_width :
  ∀ (width : ℝ), 
    (10 : ℝ) * width = 70 → width = 7 := by
  sorry

end NUMINAMATH_CALUDE_roof_shingle_width_l2179_217979


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2179_217933

theorem fraction_evaluation (a b c : ℚ) (ha : a = 7) (hb : b = 11) (hc : c = 19) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2179_217933


namespace NUMINAMATH_CALUDE_onion_transport_trips_l2179_217945

theorem onion_transport_trips (bags_per_trip : ℕ) (weight_per_bag : ℕ) (total_weight : ℕ) : 
  bags_per_trip = 10 → weight_per_bag = 50 → total_weight = 10000 →
  (total_weight / (bags_per_trip * weight_per_bag) : ℕ) = 20 := by
sorry

end NUMINAMATH_CALUDE_onion_transport_trips_l2179_217945


namespace NUMINAMATH_CALUDE_sum_product_difference_l2179_217913

theorem sum_product_difference (x y : ℝ) : 
  x + y = 500 → x * y = 22000 → y - x = -402.5 := by
sorry

end NUMINAMATH_CALUDE_sum_product_difference_l2179_217913


namespace NUMINAMATH_CALUDE_yi_number_is_seven_eighths_l2179_217919

def card_numbers : Finset ℚ := {1/2, 3/4, 7/8, 15/16, 31/32}

def jia_statement (x : ℚ) : Prop :=
  x ∈ card_numbers ∧ x ≠ 1/2 ∧ x ≠ 31/32

def yi_statement (y : ℚ) : Prop :=
  y ∈ card_numbers ∧ y ≠ 3/4 ∧ y ≠ 15/16

theorem yi_number_is_seven_eighths :
  ∀ (x y : ℚ), x ∈ card_numbers → y ∈ card_numbers → x ≠ y →
  jia_statement x → yi_statement y → y = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_yi_number_is_seven_eighths_l2179_217919


namespace NUMINAMATH_CALUDE_contractor_problem_l2179_217912

/-- A contractor problem -/
theorem contractor_problem (daily_wage : ℚ) (daily_fine : ℚ) (total_earnings : ℚ) (absent_days : ℕ) :
  daily_wage = 25 →
  daily_fine = (15/2) →
  total_earnings = 555 →
  absent_days = 6 →
  ∃ (total_days : ℕ), total_days = 24 ∧ 
    daily_wage * (total_days - absent_days : ℚ) - daily_fine * absent_days = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_contractor_problem_l2179_217912


namespace NUMINAMATH_CALUDE_range_of_a_for_two_distinct_roots_l2179_217987

theorem range_of_a_for_two_distinct_roots : 
  ∀ a : ℝ, (∃! x y : ℝ, x ≠ y ∧ |x^2 - 5*x| = a) → (a = 0 ∨ a > 25/4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_distinct_roots_l2179_217987


namespace NUMINAMATH_CALUDE_paul_needs_21_cans_l2179_217986

/-- Represents the amount of frosting needed for different baked goods -/
structure FrostingNeeds where
  layerCake : ℕ  -- number of layer cakes
  cupcakesDozens : ℕ  -- number of dozens of cupcakes
  singleCakes : ℕ  -- number of single cakes
  browniePans : ℕ  -- number of brownie pans

/-- Calculates the total number of cans of frosting needed -/
def totalFrostingCans (needs : FrostingNeeds) : ℕ :=
  needs.layerCake + (needs.cupcakesDozens + needs.singleCakes + needs.browniePans) / 2

/-- Paul's specific frosting needs for Saturday -/
def paulsFrostingNeeds : FrostingNeeds :=
  { layerCake := 3
  , cupcakesDozens := 6
  , singleCakes := 12
  , browniePans := 18 }

/-- Theorem stating that Paul needs 21 cans of frosting -/
theorem paul_needs_21_cans : totalFrostingCans paulsFrostingNeeds = 21 := by
  sorry

end NUMINAMATH_CALUDE_paul_needs_21_cans_l2179_217986


namespace NUMINAMATH_CALUDE_a4_to_a5_booklet_l2179_217985

theorem a4_to_a5_booklet (n : ℕ) (h : 2 * n + 2 = 74) : n / 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_a4_to_a5_booklet_l2179_217985


namespace NUMINAMATH_CALUDE_four_digit_number_fraction_l2179_217948

theorem four_digit_number_fraction (a b c d : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) →  -- Ensuring each digit is less than 10
  (∃ k : ℚ, a = k * b) →  -- First digit is a fraction of the second
  (c = a + b) →  -- Third digit is the sum of first and second
  (d = 3 * b) →  -- Last digit is 3 times the second
  (1000 * a + 100 * b + 10 * c + d = 1349) →  -- The number is 1349
  (a : ℚ) / b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_four_digit_number_fraction_l2179_217948


namespace NUMINAMATH_CALUDE_max_correct_answers_is_30_l2179_217989

/-- Represents the scoring system and results of a math contest. -/
structure ContestResult where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correct answers possible given a contest result. -/
def max_correct_answers (result : ContestResult) : ℕ :=
  sorry

/-- Theorem stating that for the given contest parameters, the maximum number of correct answers is 30. -/
theorem max_correct_answers_is_30 :
  let result : ContestResult := {
    total_questions := 50,
    correct_points := 5,
    incorrect_points := -2,
    total_score := 115
  }
  max_correct_answers result = 30 := by sorry

end NUMINAMATH_CALUDE_max_correct_answers_is_30_l2179_217989


namespace NUMINAMATH_CALUDE_tv_purchase_price_l2179_217978

/-- The purchase price of a TV -/
def purchase_price : ℝ := 2250

/-- The profit made on each TV -/
def profit : ℝ := 270

/-- The price increase percentage -/
def price_increase : ℝ := 0.4

/-- The discount percentage -/
def discount : ℝ := 0.2

theorem tv_purchase_price :
  (purchase_price + purchase_price * price_increase) * (1 - discount) - purchase_price = profit :=
by sorry

end NUMINAMATH_CALUDE_tv_purchase_price_l2179_217978


namespace NUMINAMATH_CALUDE_money_difference_proof_l2179_217935

/-- The number of nickels in a quarter -/
def nickels_per_quarter : ℕ := 5

/-- Charles' quarters -/
def charles_quarters (q : ℕ) : ℕ := 7 * q + 3

/-- Richard's quarters -/
def richard_quarters (q : ℕ) : ℕ := 3 * q + 7

/-- The difference in money between Charles and Richard, expressed in nickels -/
def money_difference_in_nickels (q : ℕ) : ℕ := 
  nickels_per_quarter * (charles_quarters q - richard_quarters q)

theorem money_difference_proof (q : ℕ) : 
  money_difference_in_nickels q = 20 * (q - 1) := by
  sorry

end NUMINAMATH_CALUDE_money_difference_proof_l2179_217935


namespace NUMINAMATH_CALUDE_factors_of_48_l2179_217943

/-- The number of distinct positive factors of 48 -/
def num_factors_48 : ℕ := (Nat.factors 48).card

/-- Theorem stating that the number of distinct positive factors of 48 is 10 -/
theorem factors_of_48 : num_factors_48 = 10 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_48_l2179_217943


namespace NUMINAMATH_CALUDE_jessica_seashells_l2179_217925

/-- Given that Joan found 6 seashells and the total number of seashells found by Joan and Jessica is 14, prove that Jessica found 8 seashells. -/
theorem jessica_seashells (joan_seashells : ℕ) (total_seashells : ℕ) (h1 : joan_seashells = 6) (h2 : total_seashells = 14) :
  total_seashells - joan_seashells = 8 := by
  sorry

end NUMINAMATH_CALUDE_jessica_seashells_l2179_217925


namespace NUMINAMATH_CALUDE_area_of_divided_square_l2179_217998

/-- A square divided into rectangles of equal area with specific properties -/
structure DividedSquare where
  side : ℝ
  segment_AB : ℝ
  is_divided : Bool
  A_is_midpoint : Bool

/-- The area of a DividedSquare with given properties -/
def square_area (s : DividedSquare) : ℝ := s.side ^ 2

/-- Theorem stating the area of the square under given conditions -/
theorem area_of_divided_square (s : DividedSquare) 
  (h1 : s.is_divided = true)
  (h2 : s.segment_AB = 1)
  (h3 : s.A_is_midpoint = true) :
  square_area s = 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_divided_square_l2179_217998


namespace NUMINAMATH_CALUDE_count_blocks_with_three_differences_l2179_217959

-- Define the properties of a block
structure BlockProperty where
  material : Fin 2
  size : Fin 2
  color : Fin 4
  shape : Fin 4
  pattern : Fin 2

-- Define the set of all possible blocks
def AllBlocks : Finset BlockProperty := sorry

-- Define a function to count the differences between two blocks
def countDifferences (b1 b2 : BlockProperty) : Nat := sorry

-- Define the reference block (plastic large red circle striped)
def referenceBlock : BlockProperty := sorry

-- Theorem statement
theorem count_blocks_with_three_differences :
  (AllBlocks.filter (fun b => countDifferences b referenceBlock = 3)).card = 21 := by sorry

end NUMINAMATH_CALUDE_count_blocks_with_three_differences_l2179_217959


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_36_l2179_217952

/-- A set of consecutive positive integers -/
def ConsecutiveSet (start : ℕ) (length : ℕ) : Set ℕ :=
  {n : ℕ | start ≤ n ∧ n < start + length}

/-- The sum of a set of consecutive positive integers -/
def ConsecutiveSum (start : ℕ) (length : ℕ) : ℕ :=
  (length * (2 * start + length - 1)) / 2

/-- Theorem: There exists exactly one set of consecutive positive integers,
    containing at least two integers, whose sum is 36 -/
theorem unique_consecutive_sum_36 :
  ∃! (start length : ℕ), 
    length ≥ 2 ∧ 
    ConsecutiveSum start length = 36 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_36_l2179_217952


namespace NUMINAMATH_CALUDE_lcm_problem_l2179_217967

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 30 m = 90) 
  (h2 : Nat.lcm m 50 = 200) : 
  m = 10 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2179_217967
