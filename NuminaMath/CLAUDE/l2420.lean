import Mathlib

namespace NUMINAMATH_CALUDE_area_of_ω_l2420_242092

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (4, 15)
def B : ℝ × ℝ := (14, 9)

-- Assume A and B lie on ω
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω

-- Define the tangent lines at A and B
def tangent_A : Set (ℝ × ℝ) := sorry
def tangent_B : Set (ℝ × ℝ) := sorry

-- Assume the intersection point of tangents is on the x-axis
axiom tangents_intersect_x_axis : ∃ x : ℝ, (x, 0) ∈ tangent_A ∩ tangent_B

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_ω : 
  |circle_area ω - 154.73 * Real.pi| < 0.01 := sorry

end NUMINAMATH_CALUDE_area_of_ω_l2420_242092


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l2420_242060

/-- Given positive numbers a, b, c, d with b < d, 
    the maximum value of y = a√(x - b) + c√(d - x) is √((d-b)(a²+c²)) -/
theorem max_value_of_sum_of_roots (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hbd : b < d) :
  (∀ x, b ≤ x ∧ x ≤ d → a * Real.sqrt (x - b) + c * Real.sqrt (d - x) ≤ Real.sqrt ((d - b) * (a^2 + c^2))) ∧
  (∃ x, b < x ∧ x < d ∧ a * Real.sqrt (x - b) + c * Real.sqrt (d - x) = Real.sqrt ((d - b) * (a^2 + c^2))) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l2420_242060


namespace NUMINAMATH_CALUDE_lines_are_parallel_l2420_242032

/-- Two lines in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem lines_are_parallel (l1 l2 : Line) 
  (h1 : l1 = { slope := 2, intercept := 1 })
  (h2 : l2 = { slope := 2, intercept := 5 }) : 
  parallel l1 l2 := by sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l2420_242032


namespace NUMINAMATH_CALUDE_product_in_fourth_quadrant_l2420_242012

/-- Given two complex numbers Z₁ and Z₂, prove that their product Z is in the fourth quadrant -/
theorem product_in_fourth_quadrant (Z₁ Z₂ : ℂ) (h₁ : Z₁ = 3 + I) (h₂ : Z₂ = 1 - I) :
  let Z := Z₁ * Z₂
  (Z.re > 0 ∧ Z.im < 0) := by sorry

end NUMINAMATH_CALUDE_product_in_fourth_quadrant_l2420_242012


namespace NUMINAMATH_CALUDE_mangoes_per_kilogram_l2420_242057

theorem mangoes_per_kilogram (total_harvest : ℕ) (sold_to_market : ℕ) (remaining_mangoes : ℕ) :
  total_harvest = 60 →
  sold_to_market = 20 →
  remaining_mangoes = 160 →
  ∃ (sold_to_community : ℕ),
    sold_to_community = (total_harvest - sold_to_market) / 2 ∧
    remaining_mangoes = (total_harvest - sold_to_market - sold_to_community) * 8 :=
by
  sorry

end NUMINAMATH_CALUDE_mangoes_per_kilogram_l2420_242057


namespace NUMINAMATH_CALUDE_weekly_surplus_and_monthly_income_estimate_l2420_242042

def weekly_income : List ℤ := [65, 68, 50, 66, 50, 75, 74]
def weekly_expenditure : List ℤ := [-60, -64, -63, -58, -60, -64, -65]

def calculate_surplus (income : List ℤ) (expenditure : List ℤ) : ℤ :=
  (income.sum + expenditure.sum)

def estimate_monthly_income (expenditure : List ℤ) : ℤ :=
  (expenditure.map (Int.natAbs)).sum * 30 / 7

theorem weekly_surplus_and_monthly_income_estimate :
  (calculate_surplus weekly_income weekly_expenditure = 14) ∧
  (estimate_monthly_income weekly_expenditure = 1860) := by
  sorry

#eval calculate_surplus weekly_income weekly_expenditure
#eval estimate_monthly_income weekly_expenditure

end NUMINAMATH_CALUDE_weekly_surplus_and_monthly_income_estimate_l2420_242042


namespace NUMINAMATH_CALUDE_class1_participants_l2420_242077

/-- The number of students in Class 1 -/
def class1_students : ℕ := 40

/-- The number of students in Class 2 -/
def class2_students : ℕ := 36

/-- The number of students in Class 3 -/
def class3_students : ℕ := 44

/-- The total number of students who did not participate in the competition -/
def non_participants : ℕ := 30

/-- The proportion of students participating in the competition -/
def participation_rate : ℚ := 3/4

theorem class1_participants :
  (class1_students : ℚ) * participation_rate = 30 :=
sorry

end NUMINAMATH_CALUDE_class1_participants_l2420_242077


namespace NUMINAMATH_CALUDE_problem_statement_l2420_242059

theorem problem_statement (a b : ℝ) (h : |a + 2| + (b - 1)^2 = 0) : (a + b)^2012 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2420_242059


namespace NUMINAMATH_CALUDE_inequality_solution_l2420_242088

theorem inequality_solution (x : ℝ) : (x + 10) / (x^2 + 2*x + 5) ≥ 0 ↔ x ≥ -10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2420_242088


namespace NUMINAMATH_CALUDE_largest_non_sum_of_5_and_6_l2420_242022

def is_sum_of_5_and_6 (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

theorem largest_non_sum_of_5_and_6 :
  (∀ n : ℕ, n > 19 → n ≤ 50 → is_sum_of_5_and_6 n) ∧
  ¬is_sum_of_5_and_6 19 := by
  sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_5_and_6_l2420_242022


namespace NUMINAMATH_CALUDE_school_population_l2420_242093

theorem school_population (num_boys : ℕ) (difference : ℕ) (num_girls : ℕ) : 
  num_boys = 1145 → 
  num_boys = num_girls + difference → 
  difference = 510 → 
  num_girls = 635 := by
sorry

end NUMINAMATH_CALUDE_school_population_l2420_242093


namespace NUMINAMATH_CALUDE_janous_inequality_l2420_242026

theorem janous_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ ∃ (k : ℝ), k > 0 ∧ x = 2*k ∧ y = k ∧ z = k) :=
by sorry

end NUMINAMATH_CALUDE_janous_inequality_l2420_242026


namespace NUMINAMATH_CALUDE_chromosome_size_homology_l2420_242062

/-- Represents a chromosome -/
structure Chromosome where
  size : ℕ
  is_homologous : Bool
  has_centromere : Bool
  gene_order : List ℕ

/-- Represents a pair of chromosomes -/
structure ChromosomePair where
  chromosome1 : Chromosome
  chromosome2 : Chromosome

/-- Defines what it means for chromosomes to be homologous -/
def are_homologous (c1 c2 : Chromosome) : Prop :=
  c1.is_homologous = true ∧ c2.is_homologous = true

/-- Defines what it means for chromosomes to be sister chromatids -/
def are_sister_chromatids (c1 c2 : Chromosome) : Prop :=
  c1.size = c2.size ∧ c1.gene_order = c2.gene_order

/-- Defines a tetrad -/
def is_tetrad (cp : ChromosomePair) : Prop :=
  are_homologous cp.chromosome1 cp.chromosome2

theorem chromosome_size_homology (c1 c2 : Chromosome) :
  c1.size = c2.size → are_homologous c1 c2 → False :=
sorry

#check chromosome_size_homology

end NUMINAMATH_CALUDE_chromosome_size_homology_l2420_242062


namespace NUMINAMATH_CALUDE_harold_finances_theorem_l2420_242075

/-- Harold's monthly finances --/
def harold_finances (income rent car_payment groceries : ℚ) : Prop :=
  let utilities := car_payment / 2
  let total_expenses := rent + car_payment + utilities + groceries
  let remaining := income - total_expenses
  let retirement_savings := remaining / 2
  let final_remaining := remaining - retirement_savings
  income = 2500 ∧ 
  rent = 700 ∧ 
  car_payment = 300 ∧ 
  groceries = 50 ∧ 
  final_remaining = 650

theorem harold_finances_theorem :
  ∀ income rent car_payment groceries : ℚ,
  harold_finances income rent car_payment groceries :=
by
  sorry

end NUMINAMATH_CALUDE_harold_finances_theorem_l2420_242075


namespace NUMINAMATH_CALUDE_race_time_differences_l2420_242064

def runner_A : ℕ := 60
def runner_B : ℕ := 100
def runner_C : ℕ := 80
def runner_D : ℕ := 120

def time_difference (t1 t2 : ℕ) : ℕ := 
  if t1 > t2 then t1 - t2 else t2 - t1

theorem race_time_differences : 
  (time_difference runner_A runner_B = 40) ∧
  (time_difference runner_A runner_C = 20) ∧
  (time_difference runner_A runner_D = 60) ∧
  (time_difference runner_B runner_C = 20) ∧
  (time_difference runner_B runner_D = 20) ∧
  (time_difference runner_C runner_D = 40) :=
by sorry

end NUMINAMATH_CALUDE_race_time_differences_l2420_242064


namespace NUMINAMATH_CALUDE_gcf_154_252_l2420_242073

theorem gcf_154_252 : Nat.gcd 154 252 = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcf_154_252_l2420_242073


namespace NUMINAMATH_CALUDE_tan_two_theta_value_l2420_242021

theorem tan_two_theta_value (θ : Real) 
  (h : 2 * Real.sin (π / 2 + θ) + Real.sin (π + θ) = 0) : 
  Real.tan (2 * θ) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_theta_value_l2420_242021


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l2420_242089

/-- A regular polygon with side length 7 units and exterior angle 45 degrees has a perimeter of 56 units. -/
theorem regular_polygon_perimeter (s : ℝ) (θ : ℝ) (h1 : s = 7) (h2 : θ = 45) :
  let n : ℝ := 360 / θ
  let perimeter : ℝ := n * s
  perimeter = 56 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l2420_242089


namespace NUMINAMATH_CALUDE_propositions_true_l2420_242043

theorem propositions_true :
  (∀ a b c : ℝ, c ≠ 0 → a * c^2 > b * c^2 → a > b) ∧
  (∀ a : ℝ, 1 / a > 1 → 0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_propositions_true_l2420_242043


namespace NUMINAMATH_CALUDE_unique_positive_root_in_interval_l2420_242017

-- Define the function f(x) = x^2 - x - 1
def f (x : ℝ) : ℝ := x^2 - x - 1

-- State the theorem
theorem unique_positive_root_in_interval :
  (∃! r : ℝ, r > 0 ∧ f r = 0) →  -- There exists a unique positive root
  ∃ r : ℝ, r ∈ Set.Ioo 1 2 ∧ f r = 0 :=  -- The root is in the open interval (1, 2)
by
  sorry

end NUMINAMATH_CALUDE_unique_positive_root_in_interval_l2420_242017


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l2420_242063

/-- Proves that given a selling price of Rs. 12,000 for 200 meters of cloth
    and a loss of Rs. 6 per meter, the cost price for one meter of cloth is Rs. 66. -/
theorem cost_price_per_meter
  (total_meters : ℕ)
  (selling_price : ℕ)
  (loss_per_meter : ℕ)
  (h1 : total_meters = 200)
  (h2 : selling_price = 12000)
  (h3 : loss_per_meter = 6) :
  (selling_price + total_meters * loss_per_meter) / total_meters = 66 := by
  sorry

#check cost_price_per_meter

end NUMINAMATH_CALUDE_cost_price_per_meter_l2420_242063


namespace NUMINAMATH_CALUDE_product_remainder_ten_l2420_242052

theorem product_remainder_ten (a b c d : ℕ) (ha : a % 10 = 3) (hb : b % 10 = 7) (hc : c % 10 = 5) (hd : d % 10 = 3) :
  (a * b * c * d) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_ten_l2420_242052


namespace NUMINAMATH_CALUDE_no_valid_assignment_for_45gon_l2420_242097

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  is_regular : sides = n

/-- Represents an assignment of digits to vertices of a polygon -/
def DigitAssignment (n : ℕ) := Fin n → Fin 10

/-- Checks if an assignment satisfies the pairwise condition -/
def SatisfiesPairwiseCondition (n : ℕ) (assignment : DigitAssignment n) : Prop :=
  ∀ (i j : Fin 10), i ≠ j →
    ∃ (v w : Fin n), v ≠ w ∧ 
      assignment v = i ∧ 
      assignment w = j ∧ 
      (v.val + 1) % n = w.val ∨ (w.val + 1) % n = v.val

/-- The main theorem stating that no valid assignment exists for a 45-gon -/
theorem no_valid_assignment_for_45gon :
  ¬∃ (assignment : DigitAssignment 45), 
    SatisfiesPairwiseCondition 45 assignment :=
sorry

end NUMINAMATH_CALUDE_no_valid_assignment_for_45gon_l2420_242097


namespace NUMINAMATH_CALUDE_sixth_card_is_twelve_l2420_242018

/-- A function that checks if a list of 6 integers can be divided into 3 pairs with equal sums -/
def can_be_paired (numbers : List ℕ) : Prop :=
  numbers.length = 6 ∧
  ∃ (a b c d e f : ℕ),
    numbers = [a, b, c, d, e, f] ∧
    a + b = c + d ∧ c + d = e + f

theorem sixth_card_is_twelve :
  ∀ (x : ℕ),
    x ≥ 1 ∧ x ≤ 20 →
    can_be_paired [2, 4, 9, 17, 19, x] →
    x = 12 := by
  sorry

end NUMINAMATH_CALUDE_sixth_card_is_twelve_l2420_242018


namespace NUMINAMATH_CALUDE_number_composition_l2420_242096

/-- The number of hundreds in the given number -/
def hundreds : ℕ := 11

/-- The number of tens in the given number -/
def tens : ℕ := 11

/-- The number of units in the given number -/
def units : ℕ := 11

/-- The theorem stating that the number consisting of 11 hundreds, 11 tens, and 11 units is 1221 -/
theorem number_composition : 
  hundreds * 100 + tens * 10 + units = 1221 := by sorry

end NUMINAMATH_CALUDE_number_composition_l2420_242096


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l2420_242058

theorem geometric_sequence_constant (a : ℕ → ℝ) (c : ℝ) :
  a 1 = 2 →
  (∀ n : ℕ, a (n + 1) = a n + c * n) →
  (∃ r : ℝ, r ≠ 1 ∧ a 2 = r * a 1 ∧ a 3 = r * a 2) →
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_constant_l2420_242058


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2420_242027

open Complex

theorem complex_modulus_problem (z : ℂ) (h : (1 - I) * z = 2 * I) : abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2420_242027


namespace NUMINAMATH_CALUDE_fraction_equality_l2420_242037

theorem fraction_equality : (3+9-27+81-243+729)/(9+27-81+243-729+2187) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2420_242037


namespace NUMINAMATH_CALUDE_simplify_expression_l2420_242020

theorem simplify_expression (x : ℝ) : (x + 15) + (100 * x + 15) = 101 * x + 30 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2420_242020


namespace NUMINAMATH_CALUDE_incorrect_propositions_l2420_242016

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

theorem incorrect_propositions :
  ∃ (l m : Line) (α β : Plane),
    -- Proposition A
    ¬(parallel_line l m ∧ contained_in m α → parallel_plane l α) ∧
    -- Proposition B
    ¬(parallel_plane l α ∧ parallel_plane m α → parallel_line l m) ∧
    -- Proposition C
    ¬(parallel_line l m ∧ parallel_plane m α → parallel_plane l α) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_propositions_l2420_242016


namespace NUMINAMATH_CALUDE_tan_sum_over_cos_simplification_l2420_242046

theorem tan_sum_over_cos_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (40 * π / 180) + Real.tan (50 * π / 180)) / 
  Real.cos (10 * π / 180) = 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_over_cos_simplification_l2420_242046


namespace NUMINAMATH_CALUDE_vector_scalar_add_l2420_242009

theorem vector_scalar_add : 
  3 • !![5, -3] + !![(-4), 9] = !![11, 0] := by sorry

end NUMINAMATH_CALUDE_vector_scalar_add_l2420_242009


namespace NUMINAMATH_CALUDE_population_ratio_l2420_242003

-- Define the populations of cities X, Y, and Z
variable (X Y Z : ℝ)

-- Condition 1: City X has a population 3 times as great as the population of City Y
def condition1 : Prop := X = 3 * Y

-- Condition 2: The ratio of the population of City X to the population of City Z is 6
def condition2 : Prop := X / Z = 6

-- Theorem: The ratio of the population of City Y to the population of City Z is 2
theorem population_ratio (h1 : condition1 X Y) (h2 : condition2 X Z) : Y / Z = 2 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_l2420_242003


namespace NUMINAMATH_CALUDE_trapezoid_height_l2420_242035

/-- A trapezoid with given side lengths has a height of 12 cm -/
theorem trapezoid_height (a b c d : ℝ) (ha : a = 25) (hb : b = 4) (hc : c = 20) (hd : d = 13) :
  ∃ h : ℝ, h = 12 ∧ h^2 = c^2 - ((a - b) / 2)^2 ∧ h^2 = d^2 - ((a - b) / 2)^2 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_height_l2420_242035


namespace NUMINAMATH_CALUDE_roses_sold_l2420_242065

theorem roses_sold (initial : ℕ) (picked : ℕ) (final : ℕ) (sold : ℕ) : 
  initial = 37 → picked = 19 → final = 40 → 
  initial - sold + picked = final → 
  sold = 16 := by
sorry

end NUMINAMATH_CALUDE_roses_sold_l2420_242065


namespace NUMINAMATH_CALUDE_graph_shift_down_2_l2420_242000

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define a vertical shift transformation
def vertical_shift (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := λ x => f x - c

-- Theorem stating that y = f(x) - 2 is equivalent to shifting y = f(x) down by 2 units
theorem graph_shift_down_2 :
  ∀ x : ℝ, vertical_shift f 2 x = f x - 2 :=
by
  sorry

#check graph_shift_down_2

end NUMINAMATH_CALUDE_graph_shift_down_2_l2420_242000


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2420_242002

/-- Given a hyperbola with asymptotes y = ± 1/3 x and one focus at (0, 2√5),
    prove that its standard equation is y²/2 - x²/18 = 1 -/
theorem hyperbola_standard_equation 
  (asymptote : ℝ → ℝ)
  (focus : ℝ × ℝ)
  (h1 : ∀ x, asymptote x = 1/3 * x ∨ asymptote x = -1/3 * x)
  (h2 : focus = (0, 2 * Real.sqrt 5)) :
  ∃ f : ℝ × ℝ → ℝ, ∀ x y, f (x, y) = 0 ↔ y^2/2 - x^2/18 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2420_242002


namespace NUMINAMATH_CALUDE_teresa_total_score_l2420_242083

def teresa_scores (science music social_studies : ℕ) : Prop :=
  ∃ (physics total : ℕ),
    physics = music / 2 ∧
    total = science + music + social_studies + physics

theorem teresa_total_score :
  teresa_scores 70 80 85 → ∃ total : ℕ, total = 275 :=
by sorry

end NUMINAMATH_CALUDE_teresa_total_score_l2420_242083


namespace NUMINAMATH_CALUDE_second_table_trays_count_l2420_242067

/-- Represents the number of trays Jerry picked up -/
structure TrayPickup where
  capacity : Nat
  firstTable : Nat
  trips : Nat
  total : Nat

/-- Calculates the number of trays picked up from the second table -/
def secondTableTrays (pickup : TrayPickup) : Nat :=
  pickup.total - pickup.firstTable

/-- Theorem stating the number of trays picked up from the second table -/
theorem second_table_trays_count (pickup : TrayPickup) 
  (h1 : pickup.capacity = 8)
  (h2 : pickup.firstTable = 9)
  (h3 : pickup.trips = 2)
  (h4 : pickup.total = pickup.capacity * pickup.trips) :
  secondTableTrays pickup = 7 := by
  sorry

#check second_table_trays_count

end NUMINAMATH_CALUDE_second_table_trays_count_l2420_242067


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2420_242030

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 8 = 0) → (x₂^2 - 2*x₂ - 8 = 0) → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2420_242030


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l2420_242082

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- Definition of the line l -/
def line_l (x y m : ℝ) : Prop :=
  y = x + m

/-- Theorem stating the properties of the ellipse and the intersecting line -/
theorem ellipse_and_line_properties :
  ∃ (a b : ℝ),
    -- Foci conditions
    ((-1 : ℝ)^2 + 0^2 = a^2 - b^2) ∧
    ((1 : ℝ)^2 + 0^2 = a^2 - b^2) ∧
    -- Point P on the ellipse
    ellipse_C 1 (Real.sqrt 2 / 2) ∧
    -- Standard equation of the ellipse
    (∀ x y, ellipse_C x y ↔ x^2 / 2 + y^2 = 1) ∧
    -- Maximum intersection distance occurs when m = 0
    (∀ m, ∃ x₁ y₁ x₂ y₂,
      line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧
      ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 ≤ (2 : ℝ)^2 + (2 : ℝ)^2) ∧
    -- The line y = x achieves this maximum
    (∃ x₁ y₁ x₂ y₂,
      line_l x₁ y₁ 0 ∧ line_l x₂ y₂ 0 ∧
      ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = (2 : ℝ)^2 + (2 : ℝ)^2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l2420_242082


namespace NUMINAMATH_CALUDE_integer_solutions_of_system_l2420_242070

theorem integer_solutions_of_system : 
  ∀ x y z : ℤ, 
  x + y + z = 2 ∧ 
  x^3 + y^3 + z^3 = -10 → 
  ((x = 3 ∧ y = 3 ∧ z = -4) ∨ 
   (x = 3 ∧ y = -4 ∧ z = 3) ∨ 
   (x = -4 ∧ y = 3 ∧ z = 3)) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_system_l2420_242070


namespace NUMINAMATH_CALUDE_ab_value_l2420_242051

theorem ab_value (a b : ℝ) (h : (a - 2)^2 + Real.sqrt (b + 3) = 0) : a * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2420_242051


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l2420_242071

theorem solution_set_of_equation : 
  ∃ (S : Set ℂ), S = {6, 2, 4 + 2*I, 4 - 2*I} ∧ 
  ∀ x : ℂ, (x - 2)^4 + (x - 6)^4 = 272 ↔ x ∈ S :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l2420_242071


namespace NUMINAMATH_CALUDE_school_population_problem_l2420_242001

theorem school_population_problem :
  ∀ (initial_girls initial_boys : ℕ),
    initial_boys = initial_girls + 51 →
    (100 * initial_girls) / (initial_girls + initial_boys) = 
      (100 * (initial_girls - 41)) / ((initial_girls - 41) + (initial_boys - 19)) + 4 →
    initial_girls = 187 ∧ initial_boys = 238 :=
by
  sorry

#check school_population_problem

end NUMINAMATH_CALUDE_school_population_problem_l2420_242001


namespace NUMINAMATH_CALUDE_gcd_problem_l2420_242091

theorem gcd_problem : Nat.gcd (122^2 + 234^2 + 345^2 + 10) (123^2 + 233^2 + 347^2 + 10) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2420_242091


namespace NUMINAMATH_CALUDE_product_of_exponents_l2420_242048

theorem product_of_exponents (p r s : ℕ) : 
  4^p + 4^3 = 272 → 
  3^r + 54 = 135 → 
  7^2 + 6^s = 527 → 
  p * r * s = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_exponents_l2420_242048


namespace NUMINAMATH_CALUDE_eccentricity_classification_l2420_242014

theorem eccentricity_classification (x₁ x₂ : ℝ) : 
  2 * x₁^2 - 5 * x₁ + 2 = 0 →
  2 * x₂^2 - 5 * x₂ + 2 = 0 →
  x₁ ≠ x₂ →
  ((0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂) ∨ (0 < x₂ ∧ x₂ < 1 ∧ 1 < x₁)) :=
by sorry

end NUMINAMATH_CALUDE_eccentricity_classification_l2420_242014


namespace NUMINAMATH_CALUDE_tank_full_time_l2420_242061

/-- Represents the state of a water tank system -/
structure TankSystem where
  capacity : ℕ
  fill_rate_a : ℕ
  fill_rate_b : ℕ
  drain_rate : ℕ

/-- Calculates the time required to fill the tank -/
def time_to_fill (system : TankSystem) : ℕ :=
  let net_fill_per_cycle := system.fill_rate_a + system.fill_rate_b - system.drain_rate
  let cycles := system.capacity / net_fill_per_cycle
  cycles * 3 - 1

/-- Theorem stating that the tank will be full in 56 minutes -/
theorem tank_full_time (system : TankSystem) 
    (h1 : system.capacity = 950)
    (h2 : system.fill_rate_a = 40)
    (h3 : system.fill_rate_b = 30)
    (h4 : system.drain_rate = 20) :
  time_to_fill system = 56 := by
  sorry

#eval time_to_fill { capacity := 950, fill_rate_a := 40, fill_rate_b := 30, drain_rate := 20 }

end NUMINAMATH_CALUDE_tank_full_time_l2420_242061


namespace NUMINAMATH_CALUDE_amount_distributed_l2420_242007

/-- Proves that the amount distributed is 12000 given the conditions of the problem -/
theorem amount_distributed (A : ℕ) : 
  (A / 20 = A / 25 + 120) → A = 12000 := by
  sorry

end NUMINAMATH_CALUDE_amount_distributed_l2420_242007


namespace NUMINAMATH_CALUDE_seventh_term_of_specific_geometric_sequence_l2420_242072

/-- A geometric sequence is defined by its first term and common ratio -/
def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r ^ (n - 1)

/-- The seventh term of a geometric sequence with first term 3 and second term -3/2 is 3/64 -/
theorem seventh_term_of_specific_geometric_sequence :
  let a₁ : ℚ := 3
  let a₂ : ℚ := -3/2
  let r : ℚ := a₂ / a₁
  geometric_sequence a₁ r 7 = 3/64 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_specific_geometric_sequence_l2420_242072


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l2420_242081

theorem arithmetic_sequence_proof (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, 2 * S n = a n * (a n + 1)) :
  (∀ n, a n = n) ∧ (∀ n, a (n + 1) - a n = 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l2420_242081


namespace NUMINAMATH_CALUDE_quadratic_always_greater_than_ten_l2420_242019

theorem quadratic_always_greater_than_ten (k : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + k > 10) ↔ k > 11 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_greater_than_ten_l2420_242019


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2420_242055

theorem complex_equation_sum (x y : ℝ) : 
  (x + 2 * Complex.I) * Complex.I = y - Complex.I⁻¹ → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2420_242055


namespace NUMINAMATH_CALUDE_intersection_condition_l2420_242049

-- Define the line and parabola
def line (k x : ℝ) : ℝ := k * x - 2 * k + 2
def parabola (a x : ℝ) : ℝ := a * x^2 - 2 * a * x - 3 * a

-- Define the condition for intersection
def hasCommonPoint (a : ℝ) : Prop :=
  ∀ k, ∃ x, line k x = parabola a x

-- State the theorem
theorem intersection_condition :
  ∀ a : ℝ, hasCommonPoint a ↔ (a ≤ -2/3 ∨ a > 0) :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l2420_242049


namespace NUMINAMATH_CALUDE_remainders_of_p_squared_mod_120_l2420_242015

theorem remainders_of_p_squared_mod_120 (p : Nat) (h_prime : Nat.Prime p) (h_greater_than_5 : p > 5) :
  ∃ (r₁ r₂ : Nat), r₁ ≠ r₂ ∧ 
  (∀ (r : Nat), r < 120 → (p^2 % 120 = r ↔ r = r₁ ∨ r = r₂)) := by
  sorry

end NUMINAMATH_CALUDE_remainders_of_p_squared_mod_120_l2420_242015


namespace NUMINAMATH_CALUDE_data_fraction_less_than_value_l2420_242038

theorem data_fraction_less_than_value (data : List ℝ) (fraction : ℝ) (value : ℝ) : 
  data = [1, 2, 3, 4, 5, 5, 5, 5, 7, 11, 21] →
  fraction = 0.36363636363636365 →
  (data.filter (· < value)).length / data.length = fraction →
  value = 4 := by
  sorry

end NUMINAMATH_CALUDE_data_fraction_less_than_value_l2420_242038


namespace NUMINAMATH_CALUDE_min_time_for_all_flashes_l2420_242095

/-- Represents the three possible colors of the lights -/
inductive Color
  | Red
  | Yellow
  | Green

/-- A flash is a sequence of three different colors -/
def Flash := { seq : Fin 3 → Color // ∀ i j, i ≠ j → seq i ≠ seq j }

/-- The number of different possible flashes -/
def numFlashes : Nat := 6

/-- Duration of one flash in seconds -/
def flashDuration : Nat := 3

/-- Interval between consecutive flashes in seconds -/
def intervalDuration : Nat := 3

/-- Theorem: The minimum time required to achieve all different flashes is 33 seconds -/
theorem min_time_for_all_flashes : 
  numFlashes * flashDuration + (numFlashes - 1) * intervalDuration = 33 := by
  sorry

end NUMINAMATH_CALUDE_min_time_for_all_flashes_l2420_242095


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2420_242006

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = q * a n

theorem geometric_sequence_formula
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_positive : ∀ n : ℕ, n ≥ 1 → a n > 0)
  (h_first : a 1 = 1)
  (h_sum : a 1 + a 2 + a 3 = 7) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n - 1) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2420_242006


namespace NUMINAMATH_CALUDE_trader_gain_percentage_specific_trader_gain_percentage_l2420_242087

/-- 
Given a trader who sells items and gains an amount equal to the cost of some of those items,
this theorem proves that the gain percentage is equal to the ratio of the gained items to the sold items.
-/
theorem trader_gain_percentage 
  (items_sold : ℕ) 
  (items_gained : ℕ) 
  (items_sold_positive : items_sold > 0) 
  (items_gained_positive : items_gained > 0) :
  let gain_percentage := (items_gained : ℚ) / (items_sold : ℚ) * 100
  gain_percentage = (items_gained : ℚ) / (items_sold : ℚ) * 100 := by
  sorry

/-- 
This theorem applies the general trader_gain_percentage theorem to the specific case
where 100 items are sold and the gain is equal to the cost of 30 items.
-/
theorem specific_trader_gain_percentage :
  let items_sold := 100
  let items_gained := 30
  let gain_percentage := (items_gained : ℚ) / (items_sold : ℚ) * 100
  gain_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_specific_trader_gain_percentage_l2420_242087


namespace NUMINAMATH_CALUDE_probability_two_white_balls_l2420_242031

/-- The probability of drawing two white balls from a box containing white and black balls. -/
theorem probability_two_white_balls (total_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = 11) (h2 : white_balls = 5) :
  (white_balls.choose 2 : ℚ) / (total_balls.choose 2) = 2 / 11 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_white_balls_l2420_242031


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2420_242036

theorem quadratic_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ m < -6 ∨ m > 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2420_242036


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2420_242011

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem parallel_vectors_x_value :
  parallel a (b x) → x = 6 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2420_242011


namespace NUMINAMATH_CALUDE_employee_payments_correct_l2420_242008

def video_recorder_price (wholesale : ℝ) (markup : ℝ) : ℝ :=
  wholesale * (1 + markup)

def employee_payment (retail : ℝ) (discount : ℝ) : ℝ :=
  retail * (1 - discount)

theorem employee_payments_correct :
  let wholesale_A := 200
  let wholesale_B := 250
  let wholesale_C := 300
  let markup_A := 0.20
  let markup_B := 0.25
  let markup_C := 0.30
  let discount_X := 0.15
  let discount_Y := 0.18
  let discount_Z := 0.20
  
  let retail_A := video_recorder_price wholesale_A markup_A
  let retail_B := video_recorder_price wholesale_B markup_B
  let retail_C := video_recorder_price wholesale_C markup_C
  
  let payment_X := employee_payment retail_A discount_X
  let payment_Y := employee_payment retail_B discount_Y
  let payment_Z := employee_payment retail_C discount_Z
  
  payment_X = 204 ∧ payment_Y = 256.25 ∧ payment_Z = 312 :=
by sorry

end NUMINAMATH_CALUDE_employee_payments_correct_l2420_242008


namespace NUMINAMATH_CALUDE_matrix_equation_l2420_242041

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -7; 11, 4]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![(44/7), -(57/7); -(49/14), (63/14)]

theorem matrix_equation : N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_l2420_242041


namespace NUMINAMATH_CALUDE_bridge_toll_fee_calculation_l2420_242005

/-- Represents the taxi fare structure -/
structure TaxiFare where
  start_fee : ℝ
  per_mile_rate : ℝ

/-- Calculates the total fare for a given distance -/
def calculate_fare (fare : TaxiFare) (distance : ℝ) : ℝ :=
  fare.start_fee + fare.per_mile_rate * distance

theorem bridge_toll_fee_calculation :
  let mike_fare : TaxiFare := { start_fee := 2.50, per_mile_rate := 0.25 }
  let annie_fare : TaxiFare := { start_fee := 2.50, per_mile_rate := 0.25 }
  let mike_distance : ℝ := 36
  let annie_distance : ℝ := 16
  let mike_total : ℝ := calculate_fare mike_fare mike_distance
  let annie_base : ℝ := calculate_fare annie_fare annie_distance
  let bridge_toll : ℝ := mike_total - annie_base
  bridge_toll = 5 := by sorry

end NUMINAMATH_CALUDE_bridge_toll_fee_calculation_l2420_242005


namespace NUMINAMATH_CALUDE_lcm_24_36_45_l2420_242044

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_36_45_l2420_242044


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l2420_242098

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (h_x : x > 1)
  (h_sin : Real.sin (θ / 2) = Real.sqrt ((x + 1) / (2 * x))) :
  Real.tan θ = -Real.sqrt (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l2420_242098


namespace NUMINAMATH_CALUDE_largest_five_digit_negative_congruent_to_one_mod_23_l2420_242069

theorem largest_five_digit_negative_congruent_to_one_mod_23 :
  ∀ n : ℤ, -99999 ≤ n ∧ n < -9999 ∧ n ≡ 1 [ZMOD 23] → n ≤ -9993 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_negative_congruent_to_one_mod_23_l2420_242069


namespace NUMINAMATH_CALUDE_m_divided_by_8_l2420_242053

theorem m_divided_by_8 (m : ℕ) (h : m = 16^1011) : m / 8 = 2^4041 := by
  sorry

end NUMINAMATH_CALUDE_m_divided_by_8_l2420_242053


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2420_242068

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 2) ↔ (∃ x : ℝ, x^2 < 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2420_242068


namespace NUMINAMATH_CALUDE_base_4_9_digit_difference_l2420_242040

def num_digits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log base n).succ

theorem base_4_9_digit_difference :
  num_digits 1234 4 - num_digits 1234 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_4_9_digit_difference_l2420_242040


namespace NUMINAMATH_CALUDE_point_on_line_l2420_242013

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- A point lies on a line if and only if it can be expressed as a linear combination of two points on that line. -/
theorem point_on_line (O A B X : V) :
  (∃ t : ℝ, X - O = t • (A - O) + (1 - t) • (B - O)) ↔
  ∃ s : ℝ, X - A = s • (B - A) :=
sorry

end NUMINAMATH_CALUDE_point_on_line_l2420_242013


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_l2420_242010

-- Define a quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Theorem statement
theorem quadratic_two_zeros (a b c : ℝ) (h : a * c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  quadratic a b c x₁ = 0 ∧ 
  quadratic a b c x₂ = 0 ∧
  ∀ x : ℝ, quadratic a b c x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_l2420_242010


namespace NUMINAMATH_CALUDE_kelly_initial_games_l2420_242029

/-- The number of games Kelly gave away -/
def games_given_away : ℕ := 91

/-- The number of games Kelly has left -/
def games_left : ℕ := 92

/-- The initial number of games Kelly had -/
def initial_games : ℕ := games_given_away + games_left

theorem kelly_initial_games : initial_games = 183 := by
  sorry

end NUMINAMATH_CALUDE_kelly_initial_games_l2420_242029


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2420_242066

theorem min_value_of_expression (x : ℝ) : 4^x - 2^x + 2 ≥ (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2420_242066


namespace NUMINAMATH_CALUDE_system_solution_l2420_242039

-- Define the system of linear equations
def system (k : ℝ) (x y : ℝ) : Prop :=
  x - y = 9 * k ∧ x + y = 5 * k

-- Define the additional equation
def additional_eq (x y : ℝ) : Prop :=
  2 * x + 3 * y = 8

-- Theorem statement
theorem system_solution :
  ∀ k x y, system k x y → additional_eq x y → k = 1 ∧ x = 7 ∧ y = -2 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2420_242039


namespace NUMINAMATH_CALUDE_initial_speed_is_80_l2420_242045

/-- Represents the speed and duration of a segment of the trip -/
structure TripSegment where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (segment : TripSegment) : ℝ :=
  segment.speed * segment.duration

/-- Represents Jeff's road trip -/
def JeffsTrip (initial_speed : ℝ) : List TripSegment :=
  [{ speed := initial_speed, duration := 6 },
   { speed := 60, duration := 4 },
   { speed := 40, duration := 2 }]

/-- Calculates the total distance of the trip -/
def totalDistance (trip : List TripSegment) : ℝ :=
  trip.map distance |>.sum

theorem initial_speed_is_80 :
  ∃ (v : ℝ), totalDistance (JeffsTrip v) = 800 ∧ v = 80 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_is_80_l2420_242045


namespace NUMINAMATH_CALUDE_percentage_problem_l2420_242023

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : x * (x / 100) = 9) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2420_242023


namespace NUMINAMATH_CALUDE_smallest_k_for_no_real_roots_l2420_242078

theorem smallest_k_for_no_real_roots : ∃ k : ℤ, k = 3 ∧ 
  (∀ x : ℝ, 3 * x * (k * x - 5) - 2 * x^2 + 8 ≠ 0) ∧
  (∀ m : ℤ, m < k → ∃ x : ℝ, 3 * x * (m * x - 5) - 2 * x^2 + 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_no_real_roots_l2420_242078


namespace NUMINAMATH_CALUDE_housing_price_growth_equation_l2420_242047

/-- Represents the annual growth rate of housing prices -/
def average_annual_growth_rate : ℝ := sorry

/-- The initial housing price in 2018 (yuan per square meter) -/
def initial_price : ℝ := 5000

/-- The final housing price in 2020 (yuan per square meter) -/
def final_price : ℝ := 6500

/-- The number of years of growth -/
def years_of_growth : ℕ := 2

/-- Theorem stating that the given equation correctly represents the housing price growth -/
theorem housing_price_growth_equation :
  initial_price * (1 + average_annual_growth_rate) ^ years_of_growth = final_price :=
sorry

end NUMINAMATH_CALUDE_housing_price_growth_equation_l2420_242047


namespace NUMINAMATH_CALUDE_intersection_of_inequalities_l2420_242024

theorem intersection_of_inequalities (m n : ℝ) (h : -1 < m ∧ m < 0 ∧ 0 < n) :
  {x : ℝ | m < x ∧ x < n} ∩ {x : ℝ | -1 < x ∧ x < 0} = {x : ℝ | -1 < x ∧ x < 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_inequalities_l2420_242024


namespace NUMINAMATH_CALUDE_tom_payment_l2420_242099

/-- The total amount Tom paid to the shopkeeper for apples and mangoes -/
def total_amount (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Proof that Tom paid 1190 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 70 = 1190 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_l2420_242099


namespace NUMINAMATH_CALUDE_average_difference_l2420_242028

theorem average_difference : 
  let set1 := [20, 40, 60]
  let set2 := [10, 70, 16]
  let avg1 := (set1.sum) / (set1.length : ℝ)
  let avg2 := (set2.sum) / (set2.length : ℝ)
  avg1 - avg2 = 8 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l2420_242028


namespace NUMINAMATH_CALUDE_solution_set_f_geq_4_min_value_f_l2420_242025

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 3| + |x - 5|

-- Theorem for the solution set of f(x) ≥ 4
theorem solution_set_f_geq_4 :
  {x : ℝ | f x ≥ 4} = {x : ℝ | x ≥ 2 ∨ x ≤ 4/3} :=
by sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (x : ℝ), f x = 7/2 ∧ ∀ (y : ℝ), f y ≥ 7/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_4_min_value_f_l2420_242025


namespace NUMINAMATH_CALUDE_encyclopedia_pages_l2420_242054

/-- The number of chapters in the encyclopedia -/
def num_chapters : ℕ := 7

/-- The number of pages in each chapter of the encyclopedia -/
def pages_per_chapter : ℕ := 566

/-- The total number of pages in the encyclopedia -/
def total_pages : ℕ := num_chapters * pages_per_chapter

/-- Theorem stating that the total number of pages in the encyclopedia is 3962 -/
theorem encyclopedia_pages : total_pages = 3962 := by
  sorry

end NUMINAMATH_CALUDE_encyclopedia_pages_l2420_242054


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l2420_242033

theorem geometric_arithmetic_sequence_problem :
  ∃ (a b c : ℝ) (d : ℝ),
    a + b + c = 114 ∧
    b^2 = a * c ∧
    b ≠ a ∧
    b = a + 3 * d ∧
    c = a + 24 * d ∧
    a = 2 ∧
    b = 14 ∧
    c = 98 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l2420_242033


namespace NUMINAMATH_CALUDE_factorial_simplification_l2420_242084

theorem factorial_simplification : (12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / ((10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) - 2 * (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) = 165 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l2420_242084


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l2420_242076

theorem square_plus_inverse_square (x : ℝ) (h : x - 3/x = 2) : x^2 + 9/x^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l2420_242076


namespace NUMINAMATH_CALUDE_star_five_three_l2420_242090

def star (a b : ℝ) : ℝ := 4 * a - 2 * b

theorem star_five_three : star 5 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_star_five_three_l2420_242090


namespace NUMINAMATH_CALUDE_convex_ngon_regions_l2420_242004

/-- The number of regions in a convex n-gon divided by its diagonals -/
def num_regions (n : ℕ) : ℚ :=
  (n^4 - 6*n^3 + 23*n^2 - 36*n + 24) / 24

/-- Theorem: For a convex n-gon (n ≥ 4) with all its diagonals drawn and 
    no three diagonals intersecting at the same point, the number of regions 
    into which the n-gon is divided is (n^4 - 6n^3 + 23n^2 - 36n + 24) / 24 -/
theorem convex_ngon_regions (n : ℕ) (h : n ≥ 4) :
  num_regions n = (n^4 - 6*n^3 + 23*n^2 - 36*n + 24) / 24 :=
by sorry

end NUMINAMATH_CALUDE_convex_ngon_regions_l2420_242004


namespace NUMINAMATH_CALUDE_investment_sum_l2420_242086

/-- 
Given a sum P invested at 18% p.a. for two years yields Rs. 504 more interest 
than the same sum invested at 12% p.a. for two years, prove that P = 4200.
-/
theorem investment_sum (P : ℝ) : 
  P * (18 / 100) * 2 - P * (12 / 100) * 2 = 504 → P = 4200 := by
sorry

end NUMINAMATH_CALUDE_investment_sum_l2420_242086


namespace NUMINAMATH_CALUDE_cos_inequality_l2420_242050

theorem cos_inequality (ε x y : Real) : 
  ε > 0 → 
  x ∈ Set.Ioo (-π/4) (π/4) → 
  y ∈ Set.Ioo (-π/4) (π/4) → 
  Real.exp (x + ε) * Real.sin y = Real.exp y * Real.sin x → 
  Real.cos x ≤ Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_cos_inequality_l2420_242050


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l2420_242079

/-- The probability of having a boy or a girl -/
def gender_prob : ℚ := 1 / 2

/-- The number of children in the family -/
def num_children : ℕ := 4

/-- The probability of having at least one boy and one girl in a family with four children,
    given that the probability of having a boy or a girl is equally likely -/
theorem prob_at_least_one_boy_one_girl (h : gender_prob = 1 / 2) :
  1 - (gender_prob ^ num_children + (1 - gender_prob) ^ num_children) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l2420_242079


namespace NUMINAMATH_CALUDE_union_triangles_max_vertices_l2420_242085

/-- The maximum number of vertices in a polygon formed by the union of two triangles -/
def max_vertices_union_triangles : ℕ := 12

/-- Each triangle has 3 vertices -/
def vertices_per_triangle : ℕ := 3

/-- Each side of a triangle can intersect with at most 2 sides of the other triangle -/
def max_intersections_per_side : ℕ := 2

/-- Number of sides in a triangle -/
def sides_per_triangle : ℕ := 3

theorem union_triangles_max_vertices :
  max_vertices_union_triangles =
    2 * vertices_per_triangle +
    sides_per_triangle * max_intersections_per_side :=
by sorry

end NUMINAMATH_CALUDE_union_triangles_max_vertices_l2420_242085


namespace NUMINAMATH_CALUDE_butterfly_cocoon_time_l2420_242094

theorem butterfly_cocoon_time :
  ∀ (L C : ℕ),
    L + C = 120 →
    L = 3 * C →
    C = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_butterfly_cocoon_time_l2420_242094


namespace NUMINAMATH_CALUDE_same_color_probability_is_seven_ninths_l2420_242080

/-- Represents a die with a specific number of sides and color distribution -/
structure Die where
  sides : ℕ
  red : ℕ
  blue : ℕ
  green : ℕ
  valid : red + blue + green = sides

/-- Calculate the probability of two dice showing the same color -/
def same_color_probability (d1 d2 : Die) : ℚ :=
  let p_red := (d1.red : ℚ) / d1.sides * (d2.red : ℚ) / d2.sides
  let p_blue := (d1.blue : ℚ) / d1.sides * (d2.blue : ℚ) / d2.sides
  let p_green := (d1.green : ℚ) / d1.sides * (d2.green : ℚ) / d2.sides
  p_red + p_blue + p_green

/-- The first die with 12 sides: 3 red, 4 blue, 5 green -/
def die1 : Die := {
  sides := 12,
  red := 3,
  blue := 4,
  green := 5,
  valid := by simp
}

/-- The second die with 15 sides: 5 red, 3 blue, 7 green -/
def die2 : Die := {
  sides := 15,
  red := 5,
  blue := 3,
  green := 7,
  valid := by simp
}

/-- Theorem stating that the probability of both dice showing the same color is 7/9 -/
theorem same_color_probability_is_seven_ninths :
  same_color_probability die1 die2 = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_is_seven_ninths_l2420_242080


namespace NUMINAMATH_CALUDE_total_books_is_54_l2420_242034

/-- The total number of books Darla, Katie, and Gary have is 54 -/
theorem total_books_is_54 (darla_books : ℕ) (katie_books : ℕ) (gary_books : ℕ)
  (h1 : darla_books = 6)
  (h2 : katie_books = darla_books / 2)
  (h3 : gary_books = 5 * (darla_books + katie_books)) :
  darla_books + katie_books + gary_books = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_54_l2420_242034


namespace NUMINAMATH_CALUDE_shooting_game_propositions_l2420_242074

variable (p₁ p₂ : Prop)

theorem shooting_game_propositions :
  -- Both shots hit the airplane
  (p₁ ∧ p₂) = (p₁ ∧ p₂) ∧
  -- Both shots missed the airplane
  (¬p₁ ∧ ¬p₂) = (¬p₁ ∧ ¬p₂) ∧
  -- Exactly one shot hit the airplane
  ((p₁ ∧ ¬p₂) ∨ (p₂ ∧ ¬p₁)) = ((p₁ ∧ ¬p₂) ∨ (p₂ ∧ ¬p₁)) ∧
  -- At least one shot hit the airplane
  (p₁ ∨ p₂) = (p₁ ∨ p₂) := by sorry

end NUMINAMATH_CALUDE_shooting_game_propositions_l2420_242074


namespace NUMINAMATH_CALUDE_line_inclination_angle_l2420_242056

theorem line_inclination_angle (x y : ℝ) :
  x + y - Real.sqrt 3 = 0 → ∃ θ : ℝ, θ = 135 * π / 180 ∧ Real.tan θ = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l2420_242056
