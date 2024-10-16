import Mathlib

namespace NUMINAMATH_CALUDE_parabola_right_angle_l3344_334487

theorem parabola_right_angle (a : ℝ) : 
  let f (x : ℝ) := -(x + 3) * (2 * x + a)
  let x₁ := -3
  let x₂ := -a / 2
  let y_c := f 0
  let A := (x₁, 0)
  let B := (x₂, 0)
  let C := (0, y_c)
  f x₁ = 0 ∧ f x₂ = 0 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 + (B.1 - C.1)^2 + (B.2 - C.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 →
  a = -1/6 := by
sorry

end NUMINAMATH_CALUDE_parabola_right_angle_l3344_334487


namespace NUMINAMATH_CALUDE_intersection_points_count_l3344_334447

/-- Definition of the three lines -/
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := 5 * x + y = 1
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 2

/-- A point lies on at least two of the three lines -/
def point_on_two_lines (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

/-- The main theorem to prove -/
theorem intersection_points_count :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
    point_on_two_lines p1.1 p1.2 ∧
    point_on_two_lines p2.1 p2.2 ∧
    ∀ (p : ℝ × ℝ), point_on_two_lines p.1 p.2 → p = p1 ∨ p = p2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_points_count_l3344_334447


namespace NUMINAMATH_CALUDE_parabola_equation_l3344_334469

/-- A parabola with the origin as vertex, coordinate axes as axes of symmetry, 
    and passing through the point (6, 4) has the equation y² = 8/3 * x or x² = 9 * y -/
theorem parabola_equation : ∃ (f : ℝ → ℝ),
  (∀ x y : ℝ, f x = y ↔ (y^2 = 8/3 * x ∨ x^2 = 9 * y)) ∧
  f 0 = 0 ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  f 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3344_334469


namespace NUMINAMATH_CALUDE_veronica_cherry_pitting_time_l3344_334486

/-- Represents the time needed to pit cherries for a cherry pie --/
def cherry_pitting_time (pounds_needed : ℕ) 
                        (cherries_per_pound : ℕ) 
                        (first_pound_rate : ℚ) 
                        (second_pound_rate : ℚ) 
                        (third_pound_rate : ℚ) 
                        (interruptions : ℕ) 
                        (interruption_duration : ℚ) : ℚ :=
  sorry

theorem veronica_cherry_pitting_time :
  cherry_pitting_time 3 80 (10/20) (8/20) (12/20) 2 15 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_veronica_cherry_pitting_time_l3344_334486


namespace NUMINAMATH_CALUDE_hedge_cost_proof_l3344_334407

/-- Calculates the total cost of concrete blocks for a hedge --/
def total_cost (sections : ℕ) (blocks_per_section : ℕ) (cost_per_block : ℕ) : ℕ :=
  sections * blocks_per_section * cost_per_block

/-- Proves that the total cost of concrete blocks for the hedge is $480 --/
theorem hedge_cost_proof :
  total_cost 8 30 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_hedge_cost_proof_l3344_334407


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l3344_334461

/-- The solution to the equation x = 3 + 9 / (2 + 9 / x) -/
theorem continued_fraction_solution :
  ∃ x : ℝ, x = 3 + 9 / (2 + 9 / x) ∧ x = (3 + 3 * Real.sqrt 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l3344_334461


namespace NUMINAMATH_CALUDE_cycle_selling_price_l3344_334489

/-- Calculates the selling price of a cycle given its original price and loss percentage. -/
theorem cycle_selling_price (original_price loss_percentage : ℝ) :
  original_price = 2300 →
  loss_percentage = 30 →
  original_price * (1 - loss_percentage / 100) = 1610 := by
sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l3344_334489


namespace NUMINAMATH_CALUDE_trapezoid_median_length_l3344_334402

/-- Given a trapezoid and an equilateral triangle with specific properties,
    prove that the median of the trapezoid has length 24. -/
theorem trapezoid_median_length
  (trapezoid_area : ℝ) 
  (triangle_area : ℝ) 
  (trapezoid_height : ℝ) 
  (triangle_height : ℝ) 
  (h1 : trapezoid_area = 3 * triangle_area)
  (h2 : trapezoid_height = 8 * Real.sqrt 3)
  (h3 : triangle_height = 8 * Real.sqrt 3) :
  trapezoid_area / trapezoid_height = 24 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_median_length_l3344_334402


namespace NUMINAMATH_CALUDE_common_remainder_l3344_334471

theorem common_remainder (n : ℕ) (d₁ d₂ d₃ : ℕ) (h₁ : d₁ = 5) (h₂ : d₂ = 11) (h₃ : d₃ = 13) (hn : n = 1433) :
  n % d₁ = n % d₂ ∧ n % d₂ = n % d₃ ∧ n % d₁ = 3 := by
  sorry

end NUMINAMATH_CALUDE_common_remainder_l3344_334471


namespace NUMINAMATH_CALUDE_last_three_average_l3344_334437

theorem last_three_average (list : List ℝ) : 
  list.length = 6 →
  list.sum / 6 = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 3 = 65 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l3344_334437


namespace NUMINAMATH_CALUDE_extremum_implies_f_two_l3344_334483

/-- A cubic function with integer coefficients -/
def f (a b : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- Theorem stating that if f has an extremum of 10 at x = 1, then f(2) = 2 -/
theorem extremum_implies_f_two (a b : ℤ) :
  (f a b 1 = 10) →  -- f(1) = 10
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≤ f a b 1) →  -- local maximum at x = 1
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≥ f a b 1) →  -- local minimum at x = 1
  f a b 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_extremum_implies_f_two_l3344_334483


namespace NUMINAMATH_CALUDE_joey_age_l3344_334403

def brothers_ages : List ℕ := [4, 6, 8, 10, 12]

def movies_condition (a b : ℕ) : Prop := a + b = 18

def park_condition (a b : ℕ) : Prop := a < 9 ∧ b < 9

theorem joey_age : 
  ∃ (a b c d : ℕ),
    a ∈ brothers_ages ∧
    b ∈ brothers_ages ∧
    c ∈ brothers_ages ∧
    d ∈ brothers_ages ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    movies_condition a b ∧
    park_condition c d ∧
    6 ∉ [a, b, c, d] →
    10 ∈ brothers_ages \ [a, b, c, d, 6] :=
by
  sorry

end NUMINAMATH_CALUDE_joey_age_l3344_334403


namespace NUMINAMATH_CALUDE_melody_reading_pages_l3344_334424

theorem melody_reading_pages (english : ℕ) (civics : ℕ) (chinese : ℕ) (science : ℕ) :
  english = 20 →
  civics = 8 →
  chinese = 12 →
  (english / 4 + civics / 4 + chinese / 4 + science / 4 : ℚ) = 14 →
  science = 16 := by
sorry

end NUMINAMATH_CALUDE_melody_reading_pages_l3344_334424


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3344_334468

theorem adult_ticket_cost (child_cost : ℝ) : 
  (child_cost + 6 = 19) ∧ 
  (2 * (child_cost + 6) + 3 * child_cost = 77) := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3344_334468


namespace NUMINAMATH_CALUDE_total_revenue_proof_l3344_334417

def planned_daily_sales : ℕ := 100

def sales_data : List ℤ := [7, -5, -3, 13, -6, 12, 5]

def selling_price : ℚ := 5.5

def shipping_cost : ℚ := 2

def net_income_per_kg : ℚ := selling_price - shipping_cost

def total_planned_sales : ℕ := planned_daily_sales * 7

def actual_sales : ℤ := total_planned_sales + (sales_data.sum)

theorem total_revenue_proof :
  (actual_sales : ℚ) * net_income_per_kg = 2530.5 := by sorry

end NUMINAMATH_CALUDE_total_revenue_proof_l3344_334417


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3344_334408

theorem ratio_x_to_y (x y : ℝ) (h : (12 * x - 5 * y) / (16 * x - 3 * y) = 5 / 7) : 
  x / y = 5 / 1 := by sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3344_334408


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3344_334444

theorem polynomial_multiplication (x : ℝ) :
  (x^4 + 20*x^2 + 400) * (x^2 - 20) = x^6 - 8000 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3344_334444


namespace NUMINAMATH_CALUDE_sixth_score_achieves_target_mean_l3344_334440

def test_scores : List ℝ := [78, 84, 76, 82, 88]
def sixth_score : ℝ := 102
def target_mean : ℝ := 85

theorem sixth_score_achieves_target_mean :
  (List.sum test_scores + sixth_score) / (test_scores.length + 1) = target_mean := by
  sorry

end NUMINAMATH_CALUDE_sixth_score_achieves_target_mean_l3344_334440


namespace NUMINAMATH_CALUDE_max_snacks_with_15_dollars_l3344_334490

/-- Represents the number of snacks that can be bought with a given amount of money -/
def maxSnacks (money : ℕ) : ℕ :=
  let singlePrice := 2  -- Price of a single snack
  let packOf4Price := 5  -- Price of a pack of 4 snacks
  let packOf7Price := 8  -- Price of a pack of 7 snacks
  -- Function to calculate the maximum number of snacks
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the maximum number of snacks that can be bought with $15 is 12 -/
theorem max_snacks_with_15_dollars :
  maxSnacks 15 = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_snacks_with_15_dollars_l3344_334490


namespace NUMINAMATH_CALUDE_julio_fishing_l3344_334494

theorem julio_fishing (fish_per_hour : ℕ) (hours : ℕ) (lost_fish : ℕ) (total_fish : ℕ) : 
  hours = 9 → lost_fish = 15 → total_fish = 48 → fish_per_hour * hours - lost_fish = total_fish → fish_per_hour = 7 := by
sorry

end NUMINAMATH_CALUDE_julio_fishing_l3344_334494


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3344_334441

/-- A geometric sequence with specific conditions -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) ∧
  a 1 + a 3 = 5 ∧
  a 2 + a 4 = 10

/-- The sum of the 6th and 8th terms equals 160 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 6 + a 8 = 160 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3344_334441


namespace NUMINAMATH_CALUDE_function_and_range_theorem_l3344_334492

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Define the function g
def g (m : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - m * x

-- State the theorem
theorem function_and_range_theorem (a b m : ℝ) :
  a ≠ 0 ∧
  (∀ x, f a b (x + 1) - f a b x = 2 * x - 1) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → 
    |g m (f a b) x₁ - g m (f a b) x₂| ≤ 2) →
  (∀ x, f a b x = x^2 - 2*x + 3) ∧
  m ∈ Set.Icc (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_function_and_range_theorem_l3344_334492


namespace NUMINAMATH_CALUDE_second_pipe_fill_time_l3344_334477

theorem second_pipe_fill_time (t1 t2 t3 t_all : ℝ) (h1 : t1 = 10) (h2 : t3 = 40) (h3 : t_all = 6.31578947368421) 
  (h4 : 1 / t1 + 1 / t2 - 1 / t3 = 1 / t_all) : t2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_pipe_fill_time_l3344_334477


namespace NUMINAMATH_CALUDE_equation_graph_l3344_334497

/-- The set of points (x, y) satisfying (x+y)³ = x³ + y³ is equivalent to the union of three lines -/
theorem equation_graph (x y : ℝ) :
  (x + y)^3 = x^3 + y^3 ↔ (x = 0 ∨ y = 0 ∨ x + y = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_graph_l3344_334497


namespace NUMINAMATH_CALUDE_complex_number_solution_binomial_expansion_coefficient_l3344_334410

-- Part 1
def complex_number (b : ℝ) : ℂ := 3 + b * Complex.I

theorem complex_number_solution (b : ℝ) (h1 : b > 0) (h2 : ∃ (k : ℝ), (complex_number b - 2)^2 = k * Complex.I) :
  complex_number b = 3 + Complex.I := by sorry

-- Part 2
def binomial_sum (n : ℕ) : ℕ := 2^n

def expansion_term (n r : ℕ) (x : ℝ) : ℝ :=
  Nat.choose n r * 3^(n - r) * x^(n - 3/2 * r)

theorem binomial_expansion_coefficient (n : ℕ) (h : binomial_sum n = 16) :
  expansion_term n 2 1 = 54 := by sorry

end NUMINAMATH_CALUDE_complex_number_solution_binomial_expansion_coefficient_l3344_334410


namespace NUMINAMATH_CALUDE_complex_to_exponential_form_l3344_334442

theorem complex_to_exponential_form (z : ℂ) : z = 2 - 2 * Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ θ = 5 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_to_exponential_form_l3344_334442


namespace NUMINAMATH_CALUDE_circumscribed_quadrilateral_altitudes_collinear_l3344_334499

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Circle : Type :=
  (center : Point)
  (radius : ℝ)

structure Quadrilateral : Type :=
  (A B C D : Point)

-- Define the properties
def is_circumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

def is_altitude (P Q R S : Point) : Prop := sorry

-- Define the theorem
theorem circumscribed_quadrilateral_altitudes_collinear 
  (ABCD : Quadrilateral) (O : Point) (c : Circle) 
  (A₁ B₁ C₁ D₁ : Point) : 
  is_circumscribed ABCD c →
  c.center = O →
  is_altitude A O B A₁ →
  is_altitude B O A B₁ →
  is_altitude C O D C₁ →
  is_altitude D O C D₁ →
  ∃ (l : Set Point), A₁ ∈ l ∧ B₁ ∈ l ∧ C₁ ∈ l ∧ D₁ ∈ l ∧ 
    ∀ (P Q : Point), P ∈ l → Q ∈ l → ∃ (t : ℝ), Q.x = P.x + t * (Q.y - P.y) :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_quadrilateral_altitudes_collinear_l3344_334499


namespace NUMINAMATH_CALUDE_farmer_randy_cotton_acres_l3344_334439

/-- The number of acres a single tractor can plant in one day -/
def acres_per_day : ℕ := 68

/-- The number of tractors working for the first two days -/
def tractors_first_two_days : ℕ := 2

/-- The number of tractors working for the last three days -/
def tractors_last_three_days : ℕ := 7

/-- The number of days in the first period -/
def first_period_days : ℕ := 2

/-- The number of days in the second period -/
def second_period_days : ℕ := 3

/-- The total number of acres Farmer Randy needs to have planted -/
def total_acres : ℕ := 1700

theorem farmer_randy_cotton_acres :
  total_acres = 
    acres_per_day * first_period_days * tractors_first_two_days + 
    acres_per_day * second_period_days * tractors_last_three_days :=
by sorry

end NUMINAMATH_CALUDE_farmer_randy_cotton_acres_l3344_334439


namespace NUMINAMATH_CALUDE_radical_sum_product_l3344_334456

theorem radical_sum_product (x y : ℝ) : 
  (x + Real.sqrt y) + (x - Real.sqrt y) = 6 →
  (x + Real.sqrt y) * (x - Real.sqrt y) = 4 →
  x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_radical_sum_product_l3344_334456


namespace NUMINAMATH_CALUDE_kerrys_age_l3344_334401

/-- Proves Kerry's age given the conditions of the birthday candle problem -/
theorem kerrys_age (num_cakes : ℕ) (candles_per_box : ℕ) (cost_per_box : ℚ) (total_cost : ℚ) :
  num_cakes = 3 →
  candles_per_box = 12 →
  cost_per_box = 5/2 →
  total_cost = 5 →
  (total_cost / cost_per_box * candles_per_box) / num_cakes = 8 := by
sorry

end NUMINAMATH_CALUDE_kerrys_age_l3344_334401


namespace NUMINAMATH_CALUDE_salt_concentration_after_dilution_l3344_334451

/-- Calculates the new salt concentration after adding water to a salt solution -/
theorem salt_concentration_after_dilution
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 80)
  (h2 : initial_concentration = 0.1)
  (h3 : added_water = 20) :
  let salt_amount := initial_volume * initial_concentration
  let new_volume := initial_volume + added_water
  let new_concentration := salt_amount / new_volume
  new_concentration = 0.08 :=
by sorry

end NUMINAMATH_CALUDE_salt_concentration_after_dilution_l3344_334451


namespace NUMINAMATH_CALUDE_cylinder_cut_surface_increase_l3344_334480

/-- Represents the possible shapes of the increased surface area when cutting a cylinder --/
inductive IncreasedSurfaceShape
  | Circle
  | Rectangle

/-- Represents a cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents a way to cut a cylinder into two equal parts --/
structure CutMethod where
  (cylinder : Cylinder)
  (increasedShape : IncreasedSurfaceShape)

/-- States that there exist at least two different ways to cut a cylinder 
    resulting in different increased surface area shapes --/
theorem cylinder_cut_surface_increase 
  (c : Cylinder) : 
  ∃ (cut1 cut2 : CutMethod), 
    cut1.cylinder = c ∧ 
    cut2.cylinder = c ∧ 
    cut1.increasedShape ≠ cut2.increasedShape :=
sorry

end NUMINAMATH_CALUDE_cylinder_cut_surface_increase_l3344_334480


namespace NUMINAMATH_CALUDE_inequality_proof_l3344_334484

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1) :
  (a + b + c) / 4 ≥ (Real.sqrt (a * b - 1)) / (b + c) + 
                    (Real.sqrt (b * c - 1)) / (c + a) + 
                    (Real.sqrt (c * a - 1)) / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3344_334484


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3344_334428

theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3344_334428


namespace NUMINAMATH_CALUDE_nested_bracket_equals_two_l3344_334479

-- Define the operation [a,b,c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- Theorem statement
theorem nested_bracket_equals_two :
  bracket (bracket 60 30 90) (bracket 2 1 3) (bracket 10 5 15) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_bracket_equals_two_l3344_334479


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3344_334445

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (carbon_count : ℕ) (hydrogen_count : ℕ) (oxygen_count : ℕ) (nitrogen_count : ℕ) (sulfur_count : ℕ) : ℝ :=
  let carbon_weight : ℝ := 12.01
  let hydrogen_weight : ℝ := 1.008
  let oxygen_weight : ℝ := 16.00
  let nitrogen_weight : ℝ := 14.01
  let sulfur_weight : ℝ := 32.07
  carbon_count * carbon_weight + hydrogen_count * hydrogen_weight + oxygen_count * oxygen_weight + 
  nitrogen_count * nitrogen_weight + sulfur_count * sulfur_weight

/-- Theorem stating that the molecular weight of the given compound is approximately 323.46 g/mol -/
theorem compound_molecular_weight : 
  ∃ ε > 0, |molecular_weight 10 15 4 2 3 - 323.46| < ε :=
sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3344_334445


namespace NUMINAMATH_CALUDE_tan_addition_result_l3344_334427

theorem tan_addition_result (x : Real) (h : Real.tan x = 3) :
  Real.tan (x + π / 3) = -(6 + 5 * Real.sqrt 3) / 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_addition_result_l3344_334427


namespace NUMINAMATH_CALUDE_dodecahedron_faces_l3344_334426

/-- A regular dodecahedron is a Platonic solid with 12 faces. -/
def RegularDodecahedron : Type := Unit

/-- The number of faces in a regular dodecahedron. -/
def num_faces (d : RegularDodecahedron) : ℕ := 12

/-- Theorem: A regular dodecahedron has 12 faces. -/
theorem dodecahedron_faces :
  ∀ (d : RegularDodecahedron), num_faces d = 12 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_faces_l3344_334426


namespace NUMINAMATH_CALUDE_cube_edge_length_l3344_334463

/-- Given a cube with surface area 216 cm², prove that the length of its edge is 6 cm. -/
theorem cube_edge_length (surface_area : ℝ) (edge_length : ℝ) 
  (h1 : surface_area = 216)
  (h2 : surface_area = 6 * edge_length^2) : 
  edge_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l3344_334463


namespace NUMINAMATH_CALUDE_rectangle_triangle_length_l3344_334405

/-- Given a rectangle ABCD with side lengths and a triangle DEF inside it, 
    proves that EF has a specific length when certain conditions are met. -/
theorem rectangle_triangle_length (AB BC DE DF EF : ℝ) : 
  AB = 8 → 
  BC = 10 → 
  DE = DF → 
  (1/2 * DE * DF) = (1/3 * AB * BC) → 
  EF = (16 * Real.sqrt 15) / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_length_l3344_334405


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3344_334435

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I)^2 = 3 - 4 * Complex.I) :
  Complex.abs z = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3344_334435


namespace NUMINAMATH_CALUDE_cards_distribution_l3344_334473

/-- Given 60 cards dealt to 7 people as evenly as possible, 
    exactly 3 people will have fewer than 9 cards. -/
theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) (h2 : num_people = 7) :
  (num_people - (total_cards % num_people)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l3344_334473


namespace NUMINAMATH_CALUDE_fraction_meaningful_condition_l3344_334414

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x + 1)) ↔ x ≠ -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_condition_l3344_334414


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l3344_334453

def P (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

theorem quadratic_polynomial_property (a b : ℝ) :
  P a b 10 + P a b 30 = 40 → P a b 20 = -80 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l3344_334453


namespace NUMINAMATH_CALUDE_bus_left_seats_l3344_334433

/-- Represents the seating configuration of a bus -/
structure BusSeating where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeatCapacity : ℕ
  seatCapacity : ℕ
  totalCapacity : ℕ

/-- The bus seating configuration satisfies the given conditions -/
def validBusSeating (bus : BusSeating) : Prop :=
  bus.rightSeats = bus.leftSeats - 3 ∧
  bus.backSeatCapacity = 12 ∧
  bus.seatCapacity = 3 ∧
  bus.totalCapacity = 93 ∧
  bus.totalCapacity = bus.seatCapacity * (bus.leftSeats + bus.rightSeats) + bus.backSeatCapacity

theorem bus_left_seats (bus : BusSeating) (h : validBusSeating bus) : bus.leftSeats = 15 := by
  sorry

end NUMINAMATH_CALUDE_bus_left_seats_l3344_334433


namespace NUMINAMATH_CALUDE_quilt_shaded_area_is_40_percent_l3344_334404

/-- Represents a square quilt with shaded areas -/
structure Quilt where
  total_squares : ℕ
  fully_shaded : ℕ
  half_shaded : ℕ
  quarter_shaded : ℕ

/-- Calculates the percentage of shaded area in the quilt -/
def shaded_percentage (q : Quilt) : ℚ :=
  let shaded_area := q.fully_shaded + q.half_shaded / 2 + q.quarter_shaded / 2
  (shaded_area / q.total_squares) * 100

/-- Theorem stating that the given quilt has 40% shaded area -/
theorem quilt_shaded_area_is_40_percent :
  let q := Quilt.mk 25 4 8 4
  shaded_percentage q = 40 := by sorry

end NUMINAMATH_CALUDE_quilt_shaded_area_is_40_percent_l3344_334404


namespace NUMINAMATH_CALUDE_cube_labeling_impossibility_cube_labeling_with_13_l3344_334458

/-- The number of edges in a cube -/
def num_edges : ℕ := 12

/-- The number of vertices in a cube -/
def num_vertices : ℕ := 8

/-- The number of edges connected to each vertex in a cube -/
def edges_per_vertex : ℕ := 3

/-- A labeling of a cube's edges -/
def Labeling := Fin num_edges → ℕ

/-- The sum of labels at a vertex for a given labeling -/
def vertex_sum (l : Labeling) : ℕ := sorry

/-- Predicate for a valid labeling with values 1 to 12 -/
def valid_labeling (l : Labeling) : Prop :=
  ∀ e : Fin num_edges, l e ∈ Finset.range num_edges

/-- Predicate for a constant sum labeling -/
def constant_sum_labeling (l : Labeling) : Prop :=
  ∃ s : ℕ, ∀ v : Fin num_vertices, vertex_sum l = s

/-- Predicate for a valid labeling with one value replaced by 13 -/
def valid_labeling_with_13 (l : Labeling) : Prop :=
  ∃ e : Fin num_edges, l e = 13 ∧
    ∀ e' : Fin num_edges, e' ≠ e → l e' ∈ Finset.range num_edges

theorem cube_labeling_impossibility :
  ¬∃ l : Labeling, valid_labeling l ∧ constant_sum_labeling l :=
sorry

theorem cube_labeling_with_13 :
  ∃ l : Labeling, valid_labeling_with_13 l ∧ constant_sum_labeling l ↔
    ∃ i ∈ ({3, 7, 11} : Finset ℕ), ∃ l : Labeling,
      valid_labeling_with_13 l ∧ constant_sum_labeling l ∧
      ∃ e : Fin num_edges, l e = 13 ∧ (∀ e' : Fin num_edges, e' ≠ e → l e' ≠ i) :=
sorry

end NUMINAMATH_CALUDE_cube_labeling_impossibility_cube_labeling_with_13_l3344_334458


namespace NUMINAMATH_CALUDE_spinner_direction_l3344_334470

-- Define the four cardinal directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define a function to rotate a direction
def rotate (d : Direction) (revolutions : ℚ) : Direction :=
  match (revolutions % 1).num.mod 4 with
  | 0 => d
  | 1 => match d with
    | Direction.North => Direction.East
    | Direction.East => Direction.South
    | Direction.South => Direction.West
    | Direction.West => Direction.North
  | 2 => match d with
    | Direction.North => Direction.South
    | Direction.East => Direction.West
    | Direction.South => Direction.North
    | Direction.West => Direction.East
  | 3 => match d with
    | Direction.North => Direction.West
    | Direction.East => Direction.North
    | Direction.South => Direction.East
    | Direction.West => Direction.South
  | _ => d  -- This case should never occur due to mod 4

theorem spinner_direction :
  let initial_direction := Direction.North
  let clockwise_rotation := 7/2  -- 3.5 revolutions
  let counterclockwise_rotation := 7/4  -- 1.75 revolutions
  let final_direction := rotate (rotate initial_direction clockwise_rotation) (-counterclockwise_rotation)
  final_direction = Direction.West := by
  sorry

end NUMINAMATH_CALUDE_spinner_direction_l3344_334470


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l3344_334413

def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ+, a (n + 1) = a n * q

theorem geometric_sequence_minimum (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, a n > 0) →
  is_geometric_sequence a →
  a 7 = a 6 + 2 * a 5 →
  (∃ m n : ℕ+, Real.sqrt (a m * a n) = 4 * a 1) →
  (∃ min : ℝ, min = 3/2 ∧ ∀ m n : ℕ+, 1/m + 4/n ≥ min) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l3344_334413


namespace NUMINAMATH_CALUDE_taeyeon_height_l3344_334400

theorem taeyeon_height (seonghee_height : ℝ) (taeyeon_ratio : ℝ) :
  seonghee_height = 134.5 →
  taeyeon_ratio = 1.06 →
  taeyeon_ratio * seonghee_height = 142.57 := by
  sorry

end NUMINAMATH_CALUDE_taeyeon_height_l3344_334400


namespace NUMINAMATH_CALUDE_intersection_M_N_l3344_334420

/-- Set M is defined as the set of all real numbers x where 0 < x < 4 -/
def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}

/-- Set N is defined as the set of all real numbers x where 1/3 ≤ x ≤ 5 -/
def N : Set ℝ := {x : ℝ | 1/3 ≤ x ∧ x ≤ 5}

/-- The intersection of sets M and N -/
theorem intersection_M_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3344_334420


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l3344_334488

theorem probability_nine_heads_in_twelve_flips : 
  (Nat.choose 12 9 : ℚ) / (2^12 : ℚ) = 220 / 4096 :=
by sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l3344_334488


namespace NUMINAMATH_CALUDE_mango_per_tree_l3344_334443

theorem mango_per_tree (papaya_trees : ℕ) (mango_trees : ℕ) (papaya_per_tree : ℕ) (total_fruits : ℕ)
  (h1 : papaya_trees = 2)
  (h2 : mango_trees = 3)
  (h3 : papaya_per_tree = 10)
  (h4 : total_fruits = 80) :
  (total_fruits - papaya_trees * papaya_per_tree) / mango_trees = 20 := by
  sorry

end NUMINAMATH_CALUDE_mango_per_tree_l3344_334443


namespace NUMINAMATH_CALUDE_car_trip_speed_l3344_334472

/-- Given a car trip with specific conditions, prove the average speed for the additional hours -/
theorem car_trip_speed (total_time hours_at_70 : ℝ) (speed_70 speed_total : ℝ) 
  (h_total_time : total_time = 8)
  (h_hours_at_70 : hours_at_70 = 4)
  (h_speed_70 : speed_70 = 70)
  (h_speed_total : speed_total = 65) :
  let remaining_time := total_time - hours_at_70
  let distance_70 := speed_70 * hours_at_70
  let total_distance := speed_total * total_time
  let remaining_distance := total_distance - distance_70
  remaining_distance / remaining_time = 60 := by sorry

end NUMINAMATH_CALUDE_car_trip_speed_l3344_334472


namespace NUMINAMATH_CALUDE_equation_roots_relation_l3344_334419

theorem equation_roots_relation (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, 3 * x₁ - 4 = a ∧ (x₂ + a) / 3 = 1 ∧ x₁ = 2 * x₂) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_relation_l3344_334419


namespace NUMINAMATH_CALUDE_remainder_theorem_l3344_334485

theorem remainder_theorem : 4 * 6^24 + 3^48 ≡ 5 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3344_334485


namespace NUMINAMATH_CALUDE_tim_weekly_earnings_l3344_334482

/-- Tim's daily tasks -/
def daily_tasks : ℕ := 100

/-- Pay per task in dollars -/
def pay_per_task : ℚ := 6/5

/-- Number of working days per week -/
def working_days_per_week : ℕ := 6

/-- Tim's weekly earnings in dollars -/
def weekly_earnings : ℚ := daily_tasks * pay_per_task * working_days_per_week

theorem tim_weekly_earnings : weekly_earnings = 720 := by sorry

end NUMINAMATH_CALUDE_tim_weekly_earnings_l3344_334482


namespace NUMINAMATH_CALUDE_orthocenter_diameter_bisection_l3344_334425

/-- A point in a 2D plane. -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A triangle defined by three points. -/
structure Triangle :=
  (A B C : Point)

/-- The orthocenter of a triangle. -/
def orthocenter (t : Triangle) : Point := sorry

/-- The circumcircle of a triangle. -/
def circumcircle (t : Triangle) : Set Point := sorry

/-- A diameter of a circle. -/
def is_diameter (A A' : Point) (circle : Set Point) : Prop := sorry

/-- A segment bisects another segment. -/
def bisects (P Q : Point) (R S : Point) : Prop := sorry

/-- Main theorem: If H is the orthocenter of triangle ABC and AA' is a diameter
    of its circumcircle, then A'H bisects the side BC. -/
theorem orthocenter_diameter_bisection
  (t : Triangle) (A' : Point) (H : Point) :
  H = orthocenter t →
  is_diameter t.A A' (circumcircle t) →
  bisects A' H t.B t.C :=
sorry

end NUMINAMATH_CALUDE_orthocenter_diameter_bisection_l3344_334425


namespace NUMINAMATH_CALUDE_cousin_distribution_l3344_334450

/-- The number of ways to distribute n indistinguishable objects into k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of cousins -/
def num_cousins : ℕ := 5

/-- The number of rooms -/
def num_rooms : ℕ := 4

theorem cousin_distribution :
  distribute num_cousins num_rooms = 52 := by sorry

end NUMINAMATH_CALUDE_cousin_distribution_l3344_334450


namespace NUMINAMATH_CALUDE_exists_rank_with_profit_2016_l3344_334418

/-- The profit of a firm given its rank -/
def profit : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 1) => profit n + (n + 1)

/-- The theorem stating that there exists a rank with profit 2016 -/
theorem exists_rank_with_profit_2016 : ∃ n : ℕ, profit n = 2016 := by
  sorry

end NUMINAMATH_CALUDE_exists_rank_with_profit_2016_l3344_334418


namespace NUMINAMATH_CALUDE_mass_o2_for_combustion_l3344_334498

/-- The mass of O2 gas required for complete combustion of C8H18 -/
theorem mass_o2_for_combustion (moles_c8h18 : ℝ) (molar_mass_o2 : ℝ) : 
  moles_c8h18 = 7 → molar_mass_o2 = 32 → 
  (25 / 2 * moles_c8h18 * molar_mass_o2 : ℝ) = 2800 := by
  sorry

#check mass_o2_for_combustion

end NUMINAMATH_CALUDE_mass_o2_for_combustion_l3344_334498


namespace NUMINAMATH_CALUDE_waiter_tips_fraction_l3344_334478

theorem waiter_tips_fraction (salary : ℚ) (h : salary > 0) :
  let tips := (5 / 2) * salary
  let total_income := salary + tips
  tips / total_income = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_fraction_l3344_334478


namespace NUMINAMATH_CALUDE_goods_train_speed_l3344_334464

/-- The speed of a goods train passing a man in an opposite moving train -/
theorem goods_train_speed
  (man_train_speed : ℝ)
  (goods_train_length : ℝ)
  (passing_time : ℝ)
  (h1 : man_train_speed = 50)
  (h2 : goods_train_length = 280 / 1000)  -- Convert to km
  (h3 : passing_time = 9 / 3600)  -- Convert to hours
  : ∃ (goods_train_speed : ℝ),
    goods_train_speed = 62 ∧
    (man_train_speed + goods_train_speed) * passing_time = goods_train_length :=
by sorry


end NUMINAMATH_CALUDE_goods_train_speed_l3344_334464


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l3344_334446

/-- Given a person's monthly income and their spending on transport, 
    prove that their income is $2000 if they have $1900 left after transport expenses. -/
theorem monthly_income_calculation (I : ℝ) : 
  I - 0.05 * I = 1900 → I = 2000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l3344_334446


namespace NUMINAMATH_CALUDE_smallest_constant_two_l3344_334457

/-- A function satisfying the given conditions on the interval [0,1] -/
structure SpecialFunction where
  f : Real → Real
  domain : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x
  f_one : f 1 = 1
  subadditive : ∀ x y, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 0 ≤ x + y ∧ x + y ≤ 1 → 
    f x + f y ≤ f (x + y)

/-- The theorem stating that 2 is the smallest constant c such that f(x) ≤ cx for all x ∈ [0,1] -/
theorem smallest_constant_two (sf : SpecialFunction) : 
  (∀ x, 0 ≤ x ∧ x ≤ 1 → sf.f x ≤ 2 * x) ∧ 
  (∀ c, (∀ x, 0 ≤ x ∧ x ≤ 1 → sf.f x ≤ c * x) → 2 ≤ c) :=
sorry

end NUMINAMATH_CALUDE_smallest_constant_two_l3344_334457


namespace NUMINAMATH_CALUDE_orange_juice_mixture_l3344_334421

theorem orange_juice_mixture (pitcher_capacity : ℚ) 
  (first_pitcher_fraction : ℚ) (second_pitcher_fraction : ℚ) : 
  pitcher_capacity > 0 →
  first_pitcher_fraction = 1/4 →
  second_pitcher_fraction = 3/7 →
  (first_pitcher_fraction * pitcher_capacity + 
   second_pitcher_fraction * pitcher_capacity) / 
  (2 * pitcher_capacity) = 95/280 := by
sorry

end NUMINAMATH_CALUDE_orange_juice_mixture_l3344_334421


namespace NUMINAMATH_CALUDE_linear_function_iteration_l3344_334412

/-- Given a linear function f and its iterations, prove that ab = 6 -/
theorem linear_function_iteration (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x + b
  let f₁ : ℝ → ℝ := f
  let f₂ : ℝ → ℝ := λ x ↦ f (f₁ x)
  let f₃ : ℝ → ℝ := λ x ↦ f (f₂ x)
  let f₄ : ℝ → ℝ := λ x ↦ f (f₃ x)
  let f₅ : ℝ → ℝ := λ x ↦ f (f₄ x)
  (∀ x, f₅ x = 32 * x + 93) → a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_linear_function_iteration_l3344_334412


namespace NUMINAMATH_CALUDE_min_value_X_l3344_334432

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- Converts a four-digit number to its integer representation -/
def fourDigitToInt (a b c d : Digit) : ℕ :=
  1000 * (a.val + 1) + 100 * (b.val + 1) + 10 * (c.val + 1) + (d.val + 1)

/-- Converts a two-digit number to its integer representation -/
def twoDigitToInt (e f : Digit) : ℕ :=
  10 * (e.val + 1) + (f.val + 1)

theorem min_value_X (a b c d e f g h i : Digit) 
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i)
  (h2 : b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i)
  (h3 : c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i)
  (h4 : d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i)
  (h5 : e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i)
  (h6 : f ≠ g ∧ f ≠ h ∧ f ≠ i)
  (h7 : g ≠ h ∧ g ≠ i)
  (h8 : h ≠ i) :
  ∃ (x : ℕ), x = fourDigitToInt a b c d + twoDigitToInt e f * twoDigitToInt g h - (i.val + 1) ∧
    x ≥ 2369 ∧
    (∀ (a' b' c' d' e' f' g' h' i' : Digit),
      (a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ a' ≠ e' ∧ a' ≠ f' ∧ a' ≠ g' ∧ a' ≠ h' ∧ a' ≠ i') →
      (b' ≠ c' ∧ b' ≠ d' ∧ b' ≠ e' ∧ b' ≠ f' ∧ b' ≠ g' ∧ b' ≠ h' ∧ b' ≠ i') →
      (c' ≠ d' ∧ c' ≠ e' ∧ c' ≠ f' ∧ c' ≠ g' ∧ c' ≠ h' ∧ c' ≠ i') →
      (d' ≠ e' ∧ d' ≠ f' ∧ d' ≠ g' ∧ d' ≠ h' ∧ d' ≠ i') →
      (e' ≠ f' ∧ e' ≠ g' ∧ e' ≠ h' ∧ e' ≠ i') →
      (f' ≠ g' ∧ f' ≠ h' ∧ f' ≠ i') →
      (g' ≠ h' ∧ g' ≠ i') →
      (h' ≠ i') →
      x ≤ fourDigitToInt a' b' c' d' + twoDigitToInt e' f' * twoDigitToInt g' h' - (i'.val + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_X_l3344_334432


namespace NUMINAMATH_CALUDE_no_roots_of_composite_l3344_334459

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem no_roots_of_composite (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b c x ≠ 2 * x) →
  (∀ x, f a b c (f a b c x) ≠ 4 * x) :=
sorry

end NUMINAMATH_CALUDE_no_roots_of_composite_l3344_334459


namespace NUMINAMATH_CALUDE_horner_rule_f_at_2_f_2_equals_62_l3344_334409

/-- Horner's Rule evaluation of a polynomial -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 + 3x^3 + 5x - 4 -/
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_rule_f_at_2 :
  horner_eval [2, 3, 0, 5, -4] 2 = f 2 := by sorry

theorem f_2_equals_62 : f 2 = 62 := by sorry

end NUMINAMATH_CALUDE_horner_rule_f_at_2_f_2_equals_62_l3344_334409


namespace NUMINAMATH_CALUDE_fraction_inequality_l3344_334460

theorem fraction_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  1 / a + 4 / (1 - a) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3344_334460


namespace NUMINAMATH_CALUDE_simplified_expression_equals_result_l3344_334467

theorem simplified_expression_equals_result (a b : ℝ) 
  (ha : a = 4) (hb : b = 3) : 
  (a * Real.sqrt (1 / a) + Real.sqrt (4 * b)) - (Real.sqrt a / 2 - b * Real.sqrt (1 / b)) = 1 + 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_result_l3344_334467


namespace NUMINAMATH_CALUDE_sum_of_constants_l3344_334474

theorem sum_of_constants (a b : ℝ) : 
  (∀ x : ℝ, (x - a) / (x + b) = (x^2 - 50*x + 621) / (x^2 + 75*x - 3400)) → 
  a + b = 112 := by
sorry

end NUMINAMATH_CALUDE_sum_of_constants_l3344_334474


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l3344_334465

theorem factorial_difference_quotient : (Nat.factorial 13 - Nat.factorial 12) / Nat.factorial 10 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l3344_334465


namespace NUMINAMATH_CALUDE_ede_viv_properties_l3344_334438

theorem ede_viv_properties :
  let ede : ℕ := 242
  let viv : ℕ := 303
  (100 ≤ ede ∧ ede < 1000) ∧  -- EDE is a three-digit number
  (100 ≤ viv ∧ viv < 1000) ∧  -- VIV is a three-digit number
  (ede ≠ viv) ∧               -- EDE and VIV are distinct
  (Nat.gcd ede viv = 1) ∧     -- EDE and VIV are relatively prime
  (ede / viv = 242 / 303) ∧   -- The fraction is correct
  (∃ n : ℕ, (1000 * ede) / viv = 798 + n * 999) -- The decimal repeats as 0.798679867...
  := by sorry

end NUMINAMATH_CALUDE_ede_viv_properties_l3344_334438


namespace NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l3344_334452

theorem sqrt_inequality_solution_set (x : ℝ) :
  Real.sqrt (2 * x + 2) > x - 1 ↔ -1 ≤ x ∧ x ≤ 2 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l3344_334452


namespace NUMINAMATH_CALUDE_room_pave_cost_l3344_334493

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the cost to pave a rectangle given the cost per square meter -/
def pave_cost (r : Rectangle) (cost_per_sqm : ℝ) : ℝ := area r * cost_per_sqm

/-- The total cost to pave two rectangles -/
def total_pave_cost (r1 r2 : Rectangle) (cost1 cost2 : ℝ) : ℝ :=
  pave_cost r1 cost1 + pave_cost r2 cost2

theorem room_pave_cost :
  let rect1 : Rectangle := { length := 6, width := 4.75 }
  let rect2 : Rectangle := { length := 3, width := 2 }
  let cost1 : ℝ := 900
  let cost2 : ℝ := 750
  total_pave_cost rect1 rect2 cost1 cost2 = 30150 := by
  sorry

end NUMINAMATH_CALUDE_room_pave_cost_l3344_334493


namespace NUMINAMATH_CALUDE_distinct_quotients_exist_l3344_334429

/-- A function that checks if a number is composed of five twos and three ones -/
def is_valid_number (n : ℕ) : Prop :=
  (n.digits 10).count 2 = 5 ∧ (n.digits 10).count 1 = 3 ∧ (n.digits 10).length = 8

/-- The theorem statement -/
theorem distinct_quotients_exist : ∃ (a b c d e : ℕ),
  is_valid_number a ∧
  is_valid_number b ∧
  is_valid_number c ∧
  is_valid_number d ∧
  is_valid_number e ∧
  a % 7 = 0 ∧
  b % 7 = 0 ∧
  c % 7 = 0 ∧
  d % 7 = 0 ∧
  e % 7 = 0 ∧
  a / 7 ≠ b / 7 ∧
  a / 7 ≠ c / 7 ∧
  a / 7 ≠ d / 7 ∧
  a / 7 ≠ e / 7 ∧
  b / 7 ≠ c / 7 ∧
  b / 7 ≠ d / 7 ∧
  b / 7 ≠ e / 7 ∧
  c / 7 ≠ d / 7 ∧
  c / 7 ≠ e / 7 ∧
  d / 7 ≠ e / 7 :=
sorry

end NUMINAMATH_CALUDE_distinct_quotients_exist_l3344_334429


namespace NUMINAMATH_CALUDE_additional_yellow_peaches_eq_15_l3344_334496

def red_peaches : ℕ := 7
def yellow_peaches : ℕ := 15
def green_peaches : ℕ := 8

def additional_yellow_peaches : ℕ :=
  2 * (red_peaches + green_peaches) - yellow_peaches

theorem additional_yellow_peaches_eq_15 :
  additional_yellow_peaches = 15 := by
  sorry

end NUMINAMATH_CALUDE_additional_yellow_peaches_eq_15_l3344_334496


namespace NUMINAMATH_CALUDE_remainder_theorem_l3344_334434

/-- Given a polynomial f(x) with the following properties:
    1) When divided by (x-1), the remainder is 8
    2) When divided by (x+1), the remainder is 1
    This theorem states that the remainder when f(x) is divided by (x^2-1) is -7x-9 -/
theorem remainder_theorem (f : ℝ → ℝ) 
  (h1 : ∃ g : ℝ → ℝ, ∀ x, f x = g x * (x - 1) + 8)
  (h2 : ∃ h : ℝ → ℝ, ∀ x, f x = h x * (x + 1) + 1) :
  ∃ q : ℝ → ℝ, ∀ x, f x = q x * (x^2 - 1) + (-7*x - 9) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3344_334434


namespace NUMINAMATH_CALUDE_no_divisible_by_six_l3344_334491

theorem no_divisible_by_six : ∀ y : ℕ, y < 10 → ¬(36000 + 100 * y + 25) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_by_six_l3344_334491


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3344_334422

theorem negation_of_proposition (p : ℝ → Prop) :
  (∀ x : ℝ, x ≥ 2 → p x) ↔ ¬(∃ x : ℝ, x < 2 ∧ ¬(p x)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3344_334422


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_range_l3344_334462

/-- Given three numbers forming a geometric sequence with sum m, prove the range of the middle term -/
theorem geometric_sequence_middle_term_range (a b c m : ℝ) : 
  (∃ r : ℝ, a * r = b ∧ b * r = c) →  -- a, b, c form a geometric sequence
  (a + b + c = m) →                   -- sum of terms is m
  (m > 0) →                           -- m is positive
  (b ∈ Set.Icc (-m) 0 ∪ Set.Ioc 0 (m/3)) :=  -- range of b
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_range_l3344_334462


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3344_334423

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 4 -/
theorem equilateral_triangle_area_perimeter_ratio : 
  let side_length : ℝ := 4
  let area : ℝ := side_length^2 * Real.sqrt 3 / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3344_334423


namespace NUMINAMATH_CALUDE_cubeTowerSurfaceAreaIs1221_l3344_334436

/-- Calculates the surface area of a cube tower given a list of cube side lengths -/
def cubeTowerSurfaceArea (sideLengths : List ℕ) : ℕ :=
  match sideLengths with
  | [] => 0
  | [x] => 6 * x^2
  | x :: xs => 4 * x^2 + cubeTowerSurfaceArea xs

/-- The list of cube side lengths in the tower -/
def towerSideLengths : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- Theorem stating that the surface area of the cube tower is 1221 square units -/
theorem cubeTowerSurfaceAreaIs1221 :
  cubeTowerSurfaceArea towerSideLengths = 1221 := by
  sorry


end NUMINAMATH_CALUDE_cubeTowerSurfaceAreaIs1221_l3344_334436


namespace NUMINAMATH_CALUDE_ice_cream_theorem_l3344_334449

def ice_cream_sales (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    a + b + c + d = n ∧
    b = (n - a + 1) / 2 ∧
    c = ((n - a - b + 1) / 2 : ℕ) ∧
    d = ((n - a - b - c + 1) / 2 : ℕ) ∧
    d = 1

theorem ice_cream_theorem :
  ∀ n : ℕ, ice_cream_sales n → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_theorem_l3344_334449


namespace NUMINAMATH_CALUDE_fish_population_after_bobbit_worm_l3344_334411

/-- Calculates the number of fish remaining in James' aquarium when he discovers the Bobbit worm -/
theorem fish_population_after_bobbit_worm 
  (initial_fish : ℕ) 
  (daily_eaten : ℕ) 
  (days_before_adding : ℕ) 
  (fish_added : ℕ) 
  (total_days : ℕ) 
  (h1 : initial_fish = 60)
  (h2 : daily_eaten = 2)
  (h3 : days_before_adding = 14)
  (h4 : fish_added = 8)
  (h5 : total_days = 21) :
  initial_fish - (daily_eaten * total_days) + fish_added = 26 :=
sorry

end NUMINAMATH_CALUDE_fish_population_after_bobbit_worm_l3344_334411


namespace NUMINAMATH_CALUDE_uncoolParentsOnlyChildCount_l3344_334466

/-- Represents a class of students -/
structure PhysicsClass where
  total : ℕ
  coolDads : ℕ
  coolMoms : ℕ
  coolBothAndSiblings : ℕ

/-- Calculates the number of students with uncool parents and no siblings -/
def uncoolParentsOnlyChild (c : PhysicsClass) : ℕ :=
  c.total - (c.coolDads + c.coolMoms - c.coolBothAndSiblings)

/-- The theorem to be proved -/
theorem uncoolParentsOnlyChildCount (c : PhysicsClass) 
  (h1 : c.total = 40)
  (h2 : c.coolDads = 20)
  (h3 : c.coolMoms = 22)
  (h4 : c.coolBothAndSiblings = 10) :
  uncoolParentsOnlyChild c = 8 := by
  sorry

#eval uncoolParentsOnlyChild { total := 40, coolDads := 20, coolMoms := 22, coolBothAndSiblings := 10 }

end NUMINAMATH_CALUDE_uncoolParentsOnlyChildCount_l3344_334466


namespace NUMINAMATH_CALUDE_root_relation_l3344_334495

theorem root_relation (k : ℤ) : 
  (∃ x₁ x₂ : ℝ, x₁ = x₂ / 3 ∧ 
   4 * x₁^2 - (3*k + 2) * x₁ + (k^2 - 1) = 0 ∧
   4 * x₂^2 - (3*k + 2) * x₂ + (k^2 - 1) = 0) ↔ 
  k = 2 :=
sorry

end NUMINAMATH_CALUDE_root_relation_l3344_334495


namespace NUMINAMATH_CALUDE_circle_tangency_l3344_334481

/-- Definition of circle O₁ -/
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Definition of circle O₂ -/
def circle_O₂ (x y a : ℝ) : Prop := (x + 4)^2 + (y - a)^2 = 25

/-- The distance between the centers of two internally tangent circles
    is equal to the difference of their radii -/
def internally_tangent (a : ℝ) : Prop := 
  (4^2 + a^2).sqrt = 5 - 1

theorem circle_tangency (a : ℝ) 
  (h : internally_tangent a) : a = 0 := by sorry

end NUMINAMATH_CALUDE_circle_tangency_l3344_334481


namespace NUMINAMATH_CALUDE_max_divisor_with_equal_remainders_l3344_334416

theorem max_divisor_with_equal_remainders : 
  ∃ (k : ℕ), 
    (81849 % 243 = k) ∧ 
    (106392 % 243 = k) ∧ 
    (124374 % 243 = k) ∧ 
    (∀ m : ℕ, m > 243 → 
      ¬(∃ r : ℕ, (81849 % m = r) ∧ (106392 % m = r) ∧ (124374 % m = r))) := by
  sorry

end NUMINAMATH_CALUDE_max_divisor_with_equal_remainders_l3344_334416


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l3344_334406

theorem reciprocal_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  1 / (b - c) > 1 / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l3344_334406


namespace NUMINAMATH_CALUDE_sector_arc_length_l3344_334448

/-- Given a sector with a central angle of 60° and a radius of 3,
    the length of the arc is equal to π. -/
theorem sector_arc_length (θ : Real) (r : Real) : 
  θ = 60 * π / 180 → r = 3 → θ * r = π := by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3344_334448


namespace NUMINAMATH_CALUDE_min_value_product_l3344_334430

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 64 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 8 ∧
    (a₀ + 3 * b₀) * (b₀ + 3 * c₀) * (a₀ * c₀ + 2) = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l3344_334430


namespace NUMINAMATH_CALUDE_sams_friend_points_l3344_334475

theorem sams_friend_points (sam_points friend_points total_points : ℕ) :
  sam_points = 75 →
  total_points = 87 →
  total_points = sam_points + friend_points →
  friend_points = 12 := by
sorry

end NUMINAMATH_CALUDE_sams_friend_points_l3344_334475


namespace NUMINAMATH_CALUDE_initial_oak_trees_l3344_334476

theorem initial_oak_trees (initial : ℕ) (planted : ℕ) (total : ℕ) : 
  planted = 2 → total = 11 → initial + planted = total → initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_oak_trees_l3344_334476


namespace NUMINAMATH_CALUDE_smallestDualPalindromeCorrect_l3344_334454

def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = digits.reverse

def smallestDualPalindrome : ℕ := 15

theorem smallestDualPalindromeCorrect :
  (smallestDualPalindrome > 10) ∧
  (isPalindrome smallestDualPalindrome 2) ∧
  (isPalindrome smallestDualPalindrome 4) ∧
  (∀ n : ℕ, n > 10 ∧ n < smallestDualPalindrome →
    ¬(isPalindrome n 2 ∧ isPalindrome n 4)) :=
by sorry

end NUMINAMATH_CALUDE_smallestDualPalindromeCorrect_l3344_334454


namespace NUMINAMATH_CALUDE_shaded_area_of_square_l3344_334455

theorem shaded_area_of_square (r : ℝ) (h1 : r = 1/4) :
  (∑' n, r^n) * r = 1/3 := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_l3344_334455


namespace NUMINAMATH_CALUDE_complex_subtraction_l3344_334415

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + 3*I) :
  a - 3*b = -1 - 12*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3344_334415


namespace NUMINAMATH_CALUDE_binary_1111111111_equals_1023_l3344_334431

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1111111111 -/
def binary_1111111111 : List Bool :=
  [true, true, true, true, true, true, true, true, true, true]

theorem binary_1111111111_equals_1023 :
  binary_to_decimal binary_1111111111 = 1023 := by
  sorry

end NUMINAMATH_CALUDE_binary_1111111111_equals_1023_l3344_334431
