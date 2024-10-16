import Mathlib

namespace NUMINAMATH_CALUDE_b_over_c_equals_27_l68_6886

-- Define the coefficients of the quadratic equations
variable (a b c : ℝ)

-- Define the roots of the second equation
variable (s₁ s₂ : ℝ)

-- Assumptions
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0
axiom c_nonzero : c ≠ 0

-- Define the relationships between roots and coefficients
axiom vieta_sum : c = -(s₁ + s₂)
axiom vieta_product : a = s₁ * s₂

-- Define the relationship between the roots of the two equations
axiom root_relationship : a = -(3*s₁ + 3*s₂) ∧ b = 9*s₁*s₂

-- Theorem to prove
theorem b_over_c_equals_27 : b / c = 27 := by
  sorry

end NUMINAMATH_CALUDE_b_over_c_equals_27_l68_6886


namespace NUMINAMATH_CALUDE_equation_equivalence_l68_6806

theorem equation_equivalence : ∀ x : ℝ, (x = 3) ↔ (x - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l68_6806


namespace NUMINAMATH_CALUDE_sum_modulo_thirteen_l68_6862

theorem sum_modulo_thirteen : (9375 + 9376 + 9377 + 9378 + 9379) % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_modulo_thirteen_l68_6862


namespace NUMINAMATH_CALUDE_line_equation_transformation_l68_6835

/-- Given a line l: Ax + By + C = 0 and a point (x₀, y₀) on the line,
    prove that the line equation can be transformed to A(x - x₀) + B(y - y₀) = 0 -/
theorem line_equation_transformation 
  (A B C x₀ y₀ : ℝ) 
  (h1 : A ≠ 0 ∨ B ≠ 0) 
  (h2 : A * x₀ + B * y₀ + C = 0) :
  ∀ x y, A * x + B * y + C = 0 ↔ A * (x - x₀) + B * (y - y₀) = 0 :=
sorry

end NUMINAMATH_CALUDE_line_equation_transformation_l68_6835


namespace NUMINAMATH_CALUDE_food_waste_scientific_notation_l68_6875

theorem food_waste_scientific_notation :
  (500 : ℝ) * 1000000000 = 5 * (10 : ℝ)^10 := by sorry

end NUMINAMATH_CALUDE_food_waste_scientific_notation_l68_6875


namespace NUMINAMATH_CALUDE_inverse_variation_cube_l68_6859

/-- Given two positive real numbers x and y that vary inversely with respect to x^3,
    if y = 8 when x = 2, then x = 1 when y = 64. -/
theorem inverse_variation_cube (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_inverse : ∃ (k : ℝ), ∀ x y, x^3 * y = k)
  (h_initial : 2^3 * 8 = (x^3 * y)) :
  y = 64 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_l68_6859


namespace NUMINAMATH_CALUDE_dorothy_doughnut_profit_l68_6853

/-- Dorothy's doughnut business problem -/
theorem dorothy_doughnut_profit :
  let ingredient_cost : ℤ := 53
  let rent_utilities : ℤ := 27
  let num_doughnuts : ℕ := 25
  let price_per_doughnut : ℤ := 3
  let total_expenses : ℤ := ingredient_cost + rent_utilities
  let revenue : ℤ := num_doughnuts * price_per_doughnut
  let profit : ℤ := revenue - total_expenses
  profit = -5 := by
sorry


end NUMINAMATH_CALUDE_dorothy_doughnut_profit_l68_6853


namespace NUMINAMATH_CALUDE_triangle_inequality_l68_6860

theorem triangle_inequality (a b c : ℝ) (α : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hα : 0 < α ∧ α < π) (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) 
  (h_cosine : a^2 = b^2 + c^2 - 2*b*c*(Real.cos α)) :
  (2*b*c*(Real.cos α))/(b + c) < b + c - a ∧ b + c - a < (2*b*c)/a := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l68_6860


namespace NUMINAMATH_CALUDE_factor_polynomial_l68_6825

theorem factor_polynomial (x : ℝ) : 75 * x^3 - 300 * x^7 = 75 * x^3 * (1 - 4 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l68_6825


namespace NUMINAMATH_CALUDE_A_power_101_l68_6829

def A : Matrix (Fin 3) (Fin 3) ℤ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]

theorem A_power_101 : A ^ 101 = !![0, 1, 0; 0, 0, 1; 1, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_A_power_101_l68_6829


namespace NUMINAMATH_CALUDE_division_remainder_problem_l68_6869

theorem division_remainder_problem (L S R : ℝ) : 
  L - S = 1356 →
  S = 268.2 →
  L = 6 * S + R →
  R = 15 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l68_6869


namespace NUMINAMATH_CALUDE_cos_120_degrees_l68_6882

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l68_6882


namespace NUMINAMATH_CALUDE_total_games_calculation_l68_6889

/-- The number of football games in one month -/
def games_per_month : ℝ := 323.0

/-- The number of months in a season -/
def season_duration : ℝ := 17.0

/-- The total number of football games in a season -/
def total_games : ℝ := games_per_month * season_duration

theorem total_games_calculation :
  total_games = 5491.0 := by sorry

end NUMINAMATH_CALUDE_total_games_calculation_l68_6889


namespace NUMINAMATH_CALUDE_no_double_apply_function_exists_l68_6898

theorem no_double_apply_function_exists : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2015 := by
  sorry

end NUMINAMATH_CALUDE_no_double_apply_function_exists_l68_6898


namespace NUMINAMATH_CALUDE_wire_forms_perpendicular_segments_l68_6823

/-- Represents a wire configuration -/
structure WireConfiguration where
  semicircles : ℕ
  straight_segments : ℕ
  segment_length : ℝ

/-- Represents a figure formed by the wire -/
inductive Figure
  | TwoPerpendicularSegments
  | Other

/-- Checks if a wire configuration can form two perpendicular segments -/
def can_form_perpendicular_segments (w : WireConfiguration) : Prop :=
  w.semicircles = 3 ∧ w.straight_segments = 4

/-- Theorem stating that a specific wire configuration can form two perpendicular segments -/
theorem wire_forms_perpendicular_segments (w : WireConfiguration) 
  (h : can_form_perpendicular_segments w) : 
  ∃ (f : Figure), f = Figure.TwoPerpendicularSegments :=
sorry

end NUMINAMATH_CALUDE_wire_forms_perpendicular_segments_l68_6823


namespace NUMINAMATH_CALUDE_expression_evaluation_l68_6890

theorem expression_evaluation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  ((((x - 2)^2 * (x^2 + x + 1)^2) / (x^3 - 1)^2)^2 * 
   (((x + 2)^2 * (x^2 - x + 1)^2) / (x^3 + 1)^2)^2) = (x^2 - 4)^4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l68_6890


namespace NUMINAMATH_CALUDE_max_regions_50_lines_20_parallel_l68_6805

/-- The maximum number of regions created by n lines in a plane -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The number of additional regions created by m parallel lines intersecting n non-parallel lines -/
def parallel_regions (m n : ℕ) : ℕ := m * (n + 1)

/-- The maximum number of regions created by n lines in a plane, where m of them are parallel -/
def max_regions_with_parallel (n m : ℕ) : ℕ :=
  max_regions (n - m) + parallel_regions m (n - m)

theorem max_regions_50_lines_20_parallel :
  max_regions_with_parallel 50 20 = 1086 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_50_lines_20_parallel_l68_6805


namespace NUMINAMATH_CALUDE_balloon_permutations_l68_6816

def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

theorem balloon_permutations :
  balloon_arrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_permutations_l68_6816


namespace NUMINAMATH_CALUDE_tangent_line_circle_min_value_l68_6841

theorem tangent_line_circle_min_value (a b : ℝ) :
  a > 0 →
  b > 0 →
  a^2 + 4*b^2 = 2 →
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a*x + 2*b*y + 2 = 0 ∧ x^2 + y^2 = 2) →
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a'^2 + 4*b'^2 = 2 →
    (∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ a'*x' + 2*b'*y' + 2 = 0 ∧ x'^2 + y'^2 = 2) →
    1/a^2 + 1/b^2 ≤ 1/a'^2 + 1/b'^2) →
  1/a^2 + 1/b^2 = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_min_value_l68_6841


namespace NUMINAMATH_CALUDE_bea_lemonade_sales_l68_6843

theorem bea_lemonade_sales (bea_price dawn_price : ℚ) (dawn_sales : ℕ) (extra_earnings : ℚ) :
  bea_price = 25/100 →
  dawn_price = 28/100 →
  dawn_sales = 8 →
  extra_earnings = 26/100 →
  ∃ bea_sales : ℕ, bea_sales * bea_price = dawn_sales * dawn_price + extra_earnings ∧ bea_sales = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_bea_lemonade_sales_l68_6843


namespace NUMINAMATH_CALUDE_factorization_equality_l68_6810

theorem factorization_equality (a x : ℝ) : -a*x^2 + 2*a*x - a = -a*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l68_6810


namespace NUMINAMATH_CALUDE_negative_300_equals_60_l68_6885

/-- Two angles have the same terminal side if they differ by a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + 360 * k

/-- Prove that -300° and 60° have the same terminal side -/
theorem negative_300_equals_60 : same_terminal_side (-300) 60 := by
  sorry

end NUMINAMATH_CALUDE_negative_300_equals_60_l68_6885


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_min_value_achievable_l68_6828

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3*b = 1) :
  1/a + 3/b ≥ 16 :=
sorry

theorem min_value_achievable (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3*b = 1 ∧ 1/a + 3/b < 16 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_min_value_achievable_l68_6828


namespace NUMINAMATH_CALUDE_marathon_remainder_yards_l68_6873

/-- Represents the distance of a marathon in miles and yards -/
structure Marathon :=
  (miles : ℕ)
  (yards : ℕ)

/-- Represents a total distance in miles and yards -/
structure TotalDistance :=
  (miles : ℕ)
  (yards : ℕ)

def marathon_distance : Marathon :=
  { miles := 26, yards := 395 }

def yards_per_mile : ℕ := 1760

def number_of_marathons : ℕ := 15

theorem marathon_remainder_yards :
  ∃ (m : ℕ) (y : ℕ), 
    y < yards_per_mile ∧
    TotalDistance.yards (
      { miles := m
      , yards := y } : TotalDistance
    ) = 645 ∧
    m * yards_per_mile + y = 
      number_of_marathons * (marathon_distance.miles * yards_per_mile + marathon_distance.yards) :=
by sorry

end NUMINAMATH_CALUDE_marathon_remainder_yards_l68_6873


namespace NUMINAMATH_CALUDE_exists_cheaper_a_l68_6837

/-- Represents the charge for printing x copies from Company A -/
def company_a_charge (x : ℝ) : ℝ := 0.2 * x + 200

/-- Represents the charge for printing x copies from Company B -/
def company_b_charge (x : ℝ) : ℝ := 0.4 * x

/-- Theorem stating that there exists a number of copies where Company A is cheaper than Company B -/
theorem exists_cheaper_a : ∃ x : ℝ, company_a_charge x < company_b_charge x :=
sorry

end NUMINAMATH_CALUDE_exists_cheaper_a_l68_6837


namespace NUMINAMATH_CALUDE_sum_of_roots_l68_6851

theorem sum_of_roots (k m : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ ≠ x₂) 
  (h2 : 4 * x₁^2 - k * x₁ = m) 
  (h3 : 4 * x₂^2 - k * x₂ = m) : 
  x₁ + x₂ = k / 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l68_6851


namespace NUMINAMATH_CALUDE_tenth_number_in_sixteenth_group_l68_6824

/-- The sequence a_n defined by a_n = 2n - 3 -/
def a (n : ℕ) : ℤ := 2 * n - 3

/-- The first number in the kth group -/
def first_in_group (k : ℕ) : ℤ := k^2 - k - 1

/-- The mth number in the kth group -/
def number_in_group (k m : ℕ) : ℤ := first_in_group k + 2 * (m - 1)

theorem tenth_number_in_sixteenth_group :
  number_in_group 16 10 = 257 := by sorry

end NUMINAMATH_CALUDE_tenth_number_in_sixteenth_group_l68_6824


namespace NUMINAMATH_CALUDE_rectangular_plot_perimeter_l68_6848

/-- A rectangular plot with given conditions --/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencing_rate : ℝ
  total_fencing_cost : ℝ
  length_width_relation : length = width + 10
  fencing_cost_relation : fencing_rate * (2 * (length + width)) = total_fencing_cost

/-- The perimeter of the rectangular plot is 180 meters --/
theorem rectangular_plot_perimeter (plot : RectangularPlot) 
  (h_rate : plot.fencing_rate = 6.5)
  (h_cost : plot.total_fencing_cost = 1170) : 
  2 * (plot.length + plot.width) = 180 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_plot_perimeter_l68_6848


namespace NUMINAMATH_CALUDE_diagonal_length_isosceles_trapezoid_l68_6863

-- Define the isosceles trapezoid
structure IsoscelesTrapezoid :=
  (AB : ℝ) -- longer base
  (CD : ℝ) -- shorter base
  (AD : ℝ) -- leg
  (BC : ℝ) -- leg
  (isIsosceles : AD = BC)
  (isPositive : AB > 0 ∧ CD > 0 ∧ AD > 0)
  (baseOrder : AB > CD)

-- Theorem statement
theorem diagonal_length_isosceles_trapezoid (T : IsoscelesTrapezoid) 
  (h1 : T.AB = 25) 
  (h2 : T.CD = 13) 
  (h3 : T.AD = 12) :
  Real.sqrt ((25 - 13) ^ 2 / 4 + (Real.sqrt (12 ^ 2 - ((25 - 13) / 2) ^ 2)) ^ 2) = 12 :=
by sorry

end NUMINAMATH_CALUDE_diagonal_length_isosceles_trapezoid_l68_6863


namespace NUMINAMATH_CALUDE_anyas_initial_seat_l68_6888

/-- Represents the seat numbers in the theater --/
inductive Seat
| one
| two
| three
| four
| five

/-- Represents the friends --/
inductive Friend
| Anya
| Varya
| Galya
| Diana
| Ella

/-- Represents the seating arrangement before and after Anya left --/
structure SeatingArrangement where
  seats : Friend → Seat

/-- Moves a seat to the right --/
def moveRight (s : Seat) : Seat :=
  match s with
  | Seat.one => Seat.two
  | Seat.two => Seat.three
  | Seat.three => Seat.four
  | Seat.four => Seat.five
  | Seat.five => Seat.five

/-- Moves a seat to the left --/
def moveLeft (s : Seat) : Seat :=
  match s with
  | Seat.one => Seat.one
  | Seat.two => Seat.one
  | Seat.three => Seat.two
  | Seat.four => Seat.three
  | Seat.five => Seat.four

/-- Theorem stating Anya's initial seat was four --/
theorem anyas_initial_seat (initial final : SeatingArrangement) :
  (final.seats Friend.Varya = moveRight (initial.seats Friend.Varya)) →
  (final.seats Friend.Galya = moveLeft (moveLeft (initial.seats Friend.Galya))) →
  (final.seats Friend.Diana = initial.seats Friend.Ella) →
  (final.seats Friend.Ella = initial.seats Friend.Diana) →
  (final.seats Friend.Anya = Seat.five) →
  (initial.seats Friend.Anya = Seat.four) :=
by
  sorry


end NUMINAMATH_CALUDE_anyas_initial_seat_l68_6888


namespace NUMINAMATH_CALUDE_divisor_of_infinite_set_l68_6852

theorem divisor_of_infinite_set (A : Set ℕ+) 
  (h_infinite : Set.Infinite A)
  (h_finite_subset : ∀ B : Set ℕ+, B ⊆ A → Set.Finite B → 
    ∃ b : ℕ+, b > 1 ∧ ∀ x ∈ B, b ∣ x) :
  ∃ d : ℕ+, d > 1 ∧ ∀ x ∈ A, d ∣ x :=
sorry

end NUMINAMATH_CALUDE_divisor_of_infinite_set_l68_6852


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l68_6800

-- Define the types for plane and line
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (α : Plane) (a b : Line) :
  perpendicular a α → parallel a b → perpendicular b α := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l68_6800


namespace NUMINAMATH_CALUDE_special_function_value_l68_6858

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ + 2

/-- The main theorem -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) (h₀ : f 1 = 0) : 
  f 2010 = 4018 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l68_6858


namespace NUMINAMATH_CALUDE_triangle_segment_length_l68_6834

/-- Given a triangle ABC with point D on AC and point E on AD, prove that FC = 10.125 -/
theorem triangle_segment_length 
  (DC CB : ℝ) 
  (h_DC : DC = 9)
  (h_CB : CB = 6)
  (AD AB ED : ℝ)
  (h_AB : AB = (1/3) * AD)
  (h_ED : ED = (3/4) * AD)
  : ∃ (FC : ℝ), FC = 10.125 := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_length_l68_6834


namespace NUMINAMATH_CALUDE_sweetsies_leftover_l68_6895

theorem sweetsies_leftover (n : ℕ) (h : n % 8 = 5) :
  (3 * n) % 8 = 7 :=
sorry

end NUMINAMATH_CALUDE_sweetsies_leftover_l68_6895


namespace NUMINAMATH_CALUDE_miles_per_day_l68_6857

theorem miles_per_day (weekly_goal : ℕ) (days_run : ℕ) (miles_left : ℕ) : 
  weekly_goal = 24 → 
  days_run = 6 → 
  miles_left = 6 → 
  (weekly_goal - miles_left) / days_run = 3 := by
sorry

end NUMINAMATH_CALUDE_miles_per_day_l68_6857


namespace NUMINAMATH_CALUDE_x_y_negative_l68_6870

theorem x_y_negative (x y : ℝ) (h1 : x - y > x) (h2 : 3 * x + 2 * y < 2 * y) : x < 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_y_negative_l68_6870


namespace NUMINAMATH_CALUDE_distance_between_points_is_2_5_km_l68_6876

/-- Represents the running scenario with given parameters -/
structure RunningScenario where
  initialStandingTime : Real
  constantRunningRate : Real
  averageRate1 : Real
  averageRate2 : Real

/-- Calculates the distance run between two average rate points -/
def distanceBetweenPoints (scenario : RunningScenario) : Real :=
  sorry

/-- Theorem stating the distance run between the two average rate points -/
theorem distance_between_points_is_2_5_km (scenario : RunningScenario) 
  (h1 : scenario.initialStandingTime = 15 / 60) -- 15 seconds in minutes
  (h2 : scenario.constantRunningRate = 7)
  (h3 : scenario.averageRate1 = 7.5)
  (h4 : scenario.averageRate2 = 85 / 12) : -- 7 minutes 5 seconds in minutes
  distanceBetweenPoints scenario = 2.5 :=
  sorry

#check distance_between_points_is_2_5_km

end NUMINAMATH_CALUDE_distance_between_points_is_2_5_km_l68_6876


namespace NUMINAMATH_CALUDE_park_fencing_cost_l68_6892

/-- The cost of fencing one side of the square park -/
def cost_per_side : ℕ := 56

/-- The number of sides in a square -/
def num_sides : ℕ := 4

/-- The total cost of fencing the square park -/
def total_cost : ℕ := cost_per_side * num_sides

theorem park_fencing_cost : total_cost = 224 := by
  sorry

end NUMINAMATH_CALUDE_park_fencing_cost_l68_6892


namespace NUMINAMATH_CALUDE_function_is_identity_l68_6836

/-- A function satisfying specific functional equations -/
def FunctionWithProperties (f : ℝ → ℝ) (c : ℝ) : Prop :=
  c ≠ 0 ∧ 
  (∀ x : ℝ, f (x + 1) = f x + c) ∧
  (∀ x : ℝ, f (x^2) = (f x)^2)

/-- Theorem stating that a function with the given properties is the identity function with c = 1 -/
theorem function_is_identity 
  {f : ℝ → ℝ} {c : ℝ} 
  (h : FunctionWithProperties f c) : 
  c = 1 ∧ ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_CALUDE_function_is_identity_l68_6836


namespace NUMINAMATH_CALUDE_pancakes_needed_l68_6830

/-- Given a family of 8 people and 12 pancakes already made, prove that 4 more pancakes are needed for everyone to have a second pancake. -/
theorem pancakes_needed (family_size : ℕ) (pancakes_made : ℕ) : 
  family_size = 8 → pancakes_made = 12 → 
  (family_size * 2 - pancakes_made : ℕ) = 4 := by sorry

end NUMINAMATH_CALUDE_pancakes_needed_l68_6830


namespace NUMINAMATH_CALUDE_matrix_power_zero_l68_6866

theorem matrix_power_zero (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : A ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_zero_l68_6866


namespace NUMINAMATH_CALUDE_sequence_term_formula_l68_6845

/-- Given a sequence a_n with sum S_n, prove the general term formula -/
theorem sequence_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n^2 + 1 → a n = 2*n - 1) ∧
  (∀ n, S n = 2*n^2 → a n = 4*n - 2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_formula_l68_6845


namespace NUMINAMATH_CALUDE_initial_speed_is_five_l68_6847

/-- Proves that the initial speed is 5 km/hr given the conditions of the journey --/
theorem initial_speed_is_five (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) 
  (h1 : total_distance = 26.67)
  (h2 : total_time = 6)
  (h3 : second_half_speed = 4)
  (h4 : (total_distance / 2) / v + (total_distance / 2) / second_half_speed = total_time)
  : v = 5 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_is_five_l68_6847


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l68_6883

def small_radius : ℝ := 3
def large_radius : ℝ := 5

def left_rectangle_width : ℝ := small_radius
def left_rectangle_height : ℝ := 2 * small_radius
def right_rectangle_width : ℝ := large_radius
def right_rectangle_height : ℝ := 2 * large_radius

def isosceles_triangle_leg : ℝ := small_radius

theorem shaded_area_calculation :
  let left_rectangle_area := left_rectangle_width * left_rectangle_height
  let right_rectangle_area := right_rectangle_width * right_rectangle_height
  let left_semicircle_area := (1/2) * Real.pi * small_radius^2
  let right_semicircle_area := (1/2) * Real.pi * large_radius^2
  let triangle_area := (1/2) * isosceles_triangle_leg^2
  let total_shaded_area := (left_rectangle_area - left_semicircle_area - triangle_area) + 
                           (right_rectangle_area - right_semicircle_area)
  total_shaded_area = 63.5 - 17 * Real.pi := by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l68_6883


namespace NUMINAMATH_CALUDE_inscribed_rhombus_rectangle_perimeter_l68_6821

/-- A rhombus inscribed in a rectangle -/
structure InscribedRhombus where
  /-- The length of PB -/
  pb : ℝ
  /-- The length of BQ -/
  bq : ℝ
  /-- The length of PR (diagonal) -/
  pr : ℝ
  /-- The length of QS (diagonal) -/
  qs : ℝ
  /-- PB is positive -/
  pb_pos : pb > 0
  /-- BQ is positive -/
  bq_pos : bq > 0
  /-- PR is positive -/
  pr_pos : pr > 0
  /-- QS is positive -/
  qs_pos : qs > 0
  /-- PR ≠ QS (to ensure the rhombus is not a square) -/
  diag_neq : pr ≠ qs

/-- The perimeter of the rectangle containing the inscribed rhombus -/
def rectanglePerimeter (r : InscribedRhombus) : ℝ := sorry

/-- Theorem stating the perimeter of the rectangle for the given measurements -/
theorem inscribed_rhombus_rectangle_perimeter :
  let r : InscribedRhombus := {
    pb := 15,
    bq := 20,
    pr := 30,
    qs := 40,
    pb_pos := by norm_num,
    bq_pos := by norm_num,
    pr_pos := by norm_num,
    qs_pos := by norm_num,
    diag_neq := by norm_num
  }
  rectanglePerimeter r = 672 / 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_rhombus_rectangle_perimeter_l68_6821


namespace NUMINAMATH_CALUDE_yonder_license_plates_l68_6893

/-- The number of possible letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible symbols in a license plate. -/
def num_symbols : ℕ := 5

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of letter positions in a license plate. -/
def letter_positions : ℕ := 2

/-- The number of symbol positions in a license plate. -/
def symbol_positions : ℕ := 1

/-- The number of digit positions in a license plate. -/
def digit_positions : ℕ := 4

/-- The total number of valid license plates in Yonder. -/
def total_license_plates : ℕ := 33800000

/-- Theorem stating the total number of valid license plates in Yonder. -/
theorem yonder_license_plates :
  (num_letters ^ letter_positions) * (num_symbols ^ symbol_positions) * (num_digits ^ digit_positions) = total_license_plates :=
by sorry

end NUMINAMATH_CALUDE_yonder_license_plates_l68_6893


namespace NUMINAMATH_CALUDE_inequality_solution_set_l68_6864

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x / (x - 1) < 1 - a ∧ x ≠ 1}
  (a > 0 → S = Set.Ioo ((a - 1) / a) 1) ∧
  (a = 0 → S = Set.Iio 1) ∧
  (a < 0 → S = Set.Iio 1 ∪ Set.Ioi ((a - 1) / a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l68_6864


namespace NUMINAMATH_CALUDE_simplify_expression_l68_6856

theorem simplify_expression : 3 * Real.sqrt 48 - 6 * Real.sqrt (1/3) + (Real.sqrt 3 - 1)^2 = 8 * Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l68_6856


namespace NUMINAMATH_CALUDE_intersection_volume_is_constant_l68_6817

def cube_side_length : ℝ := 6

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def is_inside_cube (p : Point3D) : Prop :=
  0 < p.x ∧ p.x < cube_side_length ∧
  0 < p.y ∧ p.y < cube_side_length ∧
  0 < p.z ∧ p.z < cube_side_length

def intersection_volume (p : Point3D) : ℝ :=
  cube_side_length ^ 3 - cube_side_length ^ 2

theorem intersection_volume_is_constant (p : Point3D) (h : is_inside_cube p) :
  intersection_volume p = 180 := by sorry

end NUMINAMATH_CALUDE_intersection_volume_is_constant_l68_6817


namespace NUMINAMATH_CALUDE_remainder_71_73_div_9_l68_6872

theorem remainder_71_73_div_9 : (71 * 73) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_71_73_div_9_l68_6872


namespace NUMINAMATH_CALUDE_two_lines_exist_l68_6820

-- Define the lines given in the problem
def line_l1 (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0
def line_l2 (x y : ℝ) : Prop := x + y + 2 = 0
def line_perp (x y : ℝ) : Prop := 2 * x - y + 7 = 0

-- Define the intersection point of l1 and l2
def intersection_point : ℝ × ℝ := (-1, -1)

-- Define the given point
def given_point : ℝ × ℝ := (-3, 1)

-- Define the equations of the lines we need to prove
def line_L1 (x y : ℝ) : Prop := x + 2 * y + 3 = 0
def line_L2 (x y : ℝ) : Prop := x - 3 * y + 6 = 0

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Define the theorem
theorem two_lines_exist :
  ∃ (L1 L2 : ℝ → ℝ → Prop),
    (∀ x y, line_l1 x y ∧ line_l2 x y → L1 x y) ∧
    (∃ m1 m2, perpendicular m1 m2 ∧
      (∀ x y, line_perp x y ↔ y = m1 * x + 7/2) ∧
      (∀ x y, L1 x y ↔ y = m2 * x + (intersection_point.2 - m2 * intersection_point.1))) ∧
    L1 = line_L1 ∧
    L2 given_point.1 given_point.2 ∧
    (∃ a b, a + b = -4 ∧ ∀ x y, L2 x y ↔ x / a + y / b = 1) ∧
    L2 = line_L2 :=
  sorry

end NUMINAMATH_CALUDE_two_lines_exist_l68_6820


namespace NUMINAMATH_CALUDE_triple_composition_even_l68_6819

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

-- State the theorem
theorem triple_composition_even
  (g : ℝ → ℝ)
  (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
sorry

end NUMINAMATH_CALUDE_triple_composition_even_l68_6819


namespace NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l68_6833

/-- A line passing through (1, 2) with equal X and Y intercepts has equation 2x - y = 0 or x + y - 3 = 0 -/
theorem line_through_point_equal_intercepts :
  ∀ (a b c : ℝ),
    (a ≠ 0 ∧ b ≠ 0) →
    (a * 1 + b * 2 + c = 0) →  -- Line passes through (1, 2)
    ((-c/a) = (-c/b)) →        -- Equal X and Y intercepts
    ((a = 2 ∧ b = -1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -3)) := by
  sorry


end NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l68_6833


namespace NUMINAMATH_CALUDE_bake_sale_group_composition_l68_6867

theorem bake_sale_group_composition (p : ℕ) : 
  (p : ℚ) / 2 = (((p : ℚ) / 2 - 5) / p) * 100 → p / 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_group_composition_l68_6867


namespace NUMINAMATH_CALUDE_hiding_ways_correct_l68_6899

/-- The number of ways to hide 3 people in 6 cabinets with at most 2 people per cabinet -/
def hidingWays : ℕ := 210

/-- The number of people to be hidden -/
def numPeople : ℕ := 3

/-- The number of available cabinets -/
def numCabinets : ℕ := 6

/-- The maximum number of people that can be hidden in a single cabinet -/
def maxPerCabinet : ℕ := 2

theorem hiding_ways_correct :
  hidingWays = 
    (numCabinets * (numCabinets - 1) * (numCabinets - 2)) + 
    (Nat.choose numPeople 2 * Nat.choose numCabinets 1 * Nat.choose (numCabinets - 1) 1) := by
  sorry

#check hiding_ways_correct

end NUMINAMATH_CALUDE_hiding_ways_correct_l68_6899


namespace NUMINAMATH_CALUDE_trapezoid_theorem_l68_6884

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ
  longer_base_condition : longer_base = shorter_base + 150
  midline_ratio_condition : midline_ratio = 3 / 4
  equal_area_condition : equal_area_segment > shorter_base ∧ equal_area_segment < longer_base

/-- The main theorem about the trapezoid -/
theorem trapezoid_theorem (t : Trapezoid) : 
  ⌊(t.equal_area_segment^2) / 150⌋ = 416 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_theorem_l68_6884


namespace NUMINAMATH_CALUDE_sodas_per_pack_james_sodas_problem_l68_6832

theorem sodas_per_pack (packs : ℕ) (initial_sodas : ℕ) (days_in_week : ℕ) (sodas_per_day : ℕ) : ℕ :=
  let total_sodas := sodas_per_day * days_in_week
  let new_sodas := total_sodas - initial_sodas
  new_sodas / packs

theorem james_sodas_problem : sodas_per_pack 5 10 7 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sodas_per_pack_james_sodas_problem_l68_6832


namespace NUMINAMATH_CALUDE_correct_factorization_l68_6894

theorem correct_factorization (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l68_6894


namespace NUMINAMATH_CALUDE_factorial_sum_equals_4926_l68_6850

theorem factorial_sum_equals_4926 : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 3 = 4926 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_4926_l68_6850


namespace NUMINAMATH_CALUDE_final_state_l68_6877

/-- Represents the state of variables a, b, and c --/
structure State where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Executes the program statements and returns the final state --/
def execute : State := 
  let s1 : State := ⟨1, 2, 3⟩  -- Initial assignment: a=1, b=2, c=3
  let s2 : State := ⟨s1.a, s1.b, s1.b⟩  -- c = b
  let s3 : State := ⟨s2.a, s2.a, s2.c⟩  -- b = a
  ⟨s3.c, s3.b, s3.c⟩  -- a = c

/-- The theorem stating the final values of a, b, and c --/
theorem final_state : execute = ⟨2, 1, 2⟩ := by
  sorry


end NUMINAMATH_CALUDE_final_state_l68_6877


namespace NUMINAMATH_CALUDE_binomial_probability_theorem_l68_6807

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- The probability of a binomial random variable being equal to k -/
def probability (X : BinomialRV) (k : ℕ) : ℝ :=
  (X.n.choose k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

theorem binomial_probability_theorem (X : BinomialRV) 
  (h2 : expectation X = 2)
  (h3 : variance X = 4/3) :
  probability X 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_theorem_l68_6807


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l68_6887

/-- An isosceles triangle with given altitude and perimeter has area 75 -/
theorem isosceles_triangle_area (b s h : ℝ) : 
  h = 10 →                -- altitude is 10
  2 * s + 2 * b = 40 →    -- perimeter is 40
  s^2 = b^2 + h^2 →       -- Pythagorean theorem
  (1/2) * (2*b) * h = 75  -- area is 75
  := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l68_6887


namespace NUMINAMATH_CALUDE_sector_area_l68_6822

/-- The area of a circular sector with central angle 240° and radius 6 is 24π -/
theorem sector_area (θ : Real) (r : Real) : 
  θ = 240 * π / 180 → r = 6 → (1/2) * r^2 * θ = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l68_6822


namespace NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l68_6874

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem third_term_of_arithmetic_sequence 
  (a : ℤ) (d : ℤ) 
  (h1 : arithmetic_sequence a d 20 = 17) 
  (h2 : arithmetic_sequence a d 21 = 20) : 
  arithmetic_sequence a d 3 = -34 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l68_6874


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l68_6804

theorem cubic_roots_sum (a b c : ℝ) : 
  (3 * a^3 - 5 * a^2 + 90 * a - 2 = 0) →
  (3 * b^3 - 5 * b^2 + 90 * b - 2 = 0) →
  (3 * c^3 - 5 * c^2 + 90 * c - 2 = 0) →
  (a + b + 1)^3 + (b + c + 1)^3 + (c + a + 1)^3 = 259 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l68_6804


namespace NUMINAMATH_CALUDE_triangle_properties_l68_6813

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  (sin A) / a = (sin B) / b ∧ (sin B) / b = (sin C) / c ∧
  cos B * sin (B + π/6) = 1/2 ∧
  c / a + a / c = 4 →
  B = π/3 ∧ 1 / tan A + 1 / tan C = 2 * sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l68_6813


namespace NUMINAMATH_CALUDE_t_value_l68_6827

/-- Linear regression equation for the given data points -/
def linear_regression (x : ℝ) : ℝ := 1.04 * x + 1.9

/-- The value of t in the data set (4, t) -/
def t : ℝ := linear_regression 4

theorem t_value : t = 6.06 := by
  sorry

end NUMINAMATH_CALUDE_t_value_l68_6827


namespace NUMINAMATH_CALUDE_total_coins_l68_6855

def coin_distribution (x : ℕ) : Prop :=
  let paul_coins := x
  let pete_coins := x * (x + 1) / 2
  pete_coins = 5 * paul_coins

theorem total_coins : ∃ x : ℕ, 
  coin_distribution x ∧ 
  x + 5 * x = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_l68_6855


namespace NUMINAMATH_CALUDE_milburg_children_count_l68_6891

/-- The number of children in Milburg -/
def children_count (total_population grown_ups : ℕ) : ℕ :=
  total_population - grown_ups

/-- Theorem stating the number of children in Milburg -/
theorem milburg_children_count :
  children_count 8243 5256 = 2987 := by
  sorry

end NUMINAMATH_CALUDE_milburg_children_count_l68_6891


namespace NUMINAMATH_CALUDE_jills_earnings_l68_6846

/-- Calculates the total earnings of a waitress given her work conditions --/
def waitress_earnings (hourly_wage : ℝ) (tip_rate : ℝ) (shifts : ℕ) (hours_per_shift : ℕ) (average_orders_per_hour : ℝ) : ℝ :=
  let total_hours : ℝ := shifts * hours_per_shift
  let wage_earnings : ℝ := total_hours * hourly_wage
  let total_orders : ℝ := total_hours * average_orders_per_hour
  let tip_earnings : ℝ := total_orders * tip_rate
  wage_earnings + tip_earnings

/-- Proves that Jill's earnings for the week are $240.00 --/
theorem jills_earnings : 
  waitress_earnings 4 0.15 3 8 40 = 240 := by
  sorry

end NUMINAMATH_CALUDE_jills_earnings_l68_6846


namespace NUMINAMATH_CALUDE_sum_of_fractions_theorem_l68_6842

variable (a b c P Q : ℝ)

theorem sum_of_fractions_theorem (h1 : a + b + c = 0) 
  (h2 : a^2 / (2*a^2 + b*c) + b^2 / (2*b^2 + a*c) + c^2 / (2*c^2 + a*b) = P - 3*Q) : 
  Q = 8 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_theorem_l68_6842


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l68_6849

theorem geometric_sequence_sum (a₁ : ℝ) (r : ℝ) :
  a₁ = 3125 →
  r = 1/5 →
  (a₁ * r^5 = 1) →
  (a₁ * r^3 + a₁ * r^4 = 30) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l68_6849


namespace NUMINAMATH_CALUDE_miles_difference_l68_6808

/-- The number of miles Gervais drove per day -/
def gervais_daily_miles : ℕ := 315

/-- The number of days Gervais drove -/
def gervais_days : ℕ := 3

/-- The total number of miles Henri drove -/
def henri_total_miles : ℕ := 1250

/-- Theorem stating the difference in miles driven between Henri and Gervais -/
theorem miles_difference : henri_total_miles - (gervais_daily_miles * gervais_days) = 305 := by
  sorry

end NUMINAMATH_CALUDE_miles_difference_l68_6808


namespace NUMINAMATH_CALUDE_fourth_power_product_l68_6880

theorem fourth_power_product : 
  (((2^4 - 1) / (2^4 + 1)) * ((3^4 - 1) / (3^4 + 1)) * 
   ((4^4 - 1) / (4^4 + 1)) * ((5^4 - 1) / (5^4 + 1))) = 432 / 1105 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_product_l68_6880


namespace NUMINAMATH_CALUDE_methane_required_moles_l68_6831

/-- Represents a chemical species in a reaction -/
structure ChemicalSpecies where
  formula : String
  moles : ℚ

/-- Represents a chemical reaction -/
structure ChemicalReaction where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies

def methane_chlorine_reaction : ChemicalReaction :=
  { reactants := [
      { formula := "CH4", moles := 1 },
      { formula := "Cl2", moles := 1 }
    ],
    products := [
      { formula := "CH3Cl", moles := 1 },
      { formula := "HCl", moles := 1 }
    ]
  }

/-- Theorem stating that 2 moles of CH4 are required to react with 2 moles of Cl2 -/
theorem methane_required_moles 
  (reaction : ChemicalReaction)
  (h_reaction : reaction = methane_chlorine_reaction)
  (h_cl2_moles : ∃ cl2 ∈ reaction.reactants, cl2.formula = "Cl2" ∧ cl2.moles = 2)
  (h_hcl_moles : ∃ hcl ∈ reaction.products, hcl.formula = "HCl" ∧ hcl.moles = 2) :
  ∃ ch4 ∈ reaction.reactants, ch4.formula = "CH4" ∧ ch4.moles = 2 :=
sorry

end NUMINAMATH_CALUDE_methane_required_moles_l68_6831


namespace NUMINAMATH_CALUDE_books_per_box_l68_6814

theorem books_per_box (total_books : ℕ) (num_boxes : ℕ) (h1 : total_books = 24) (h2 : num_boxes = 8) :
  total_books / num_boxes = 3 := by
  sorry

end NUMINAMATH_CALUDE_books_per_box_l68_6814


namespace NUMINAMATH_CALUDE_greatest_valid_partition_l68_6871

/-- A partition of positive integers into k subsets satisfying the sum property -/
def ValidPartition (k : ℕ) : Prop :=
  ∃ (A : Fin k → Set ℕ), 
    (∀ i j, i ≠ j → A i ∩ A j = ∅) ∧ 
    (⋃ i, A i) = {n : ℕ | n > 0} ∧
    ∀ (n : ℕ) (i : Fin k), n ≥ 15 → 
      ∃ (x y : ℕ), x ∈ A i ∧ y ∈ A i ∧ x ≠ y ∧ x + y = n

/-- The main theorem: 3 is the greatest positive integer satisfying the property -/
theorem greatest_valid_partition : 
  ValidPartition 3 ∧ ∀ k > 3, ¬ValidPartition k :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_partition_l68_6871


namespace NUMINAMATH_CALUDE_computer_price_increase_l68_6897

theorem computer_price_increase (original_price : ℝ) : 
  original_price * 1.3 = 351 → 2 * original_price = 540 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l68_6897


namespace NUMINAMATH_CALUDE_smallest_among_given_rationals_l68_6881

theorem smallest_among_given_rationals :
  let S : Set ℚ := {5, -7, 0, -5/3}
  ∀ x ∈ S, -7 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_among_given_rationals_l68_6881


namespace NUMINAMATH_CALUDE_perimeter_difference_inscribed_quadrilateral_l68_6844

/-- A quadrilateral with an inscribed circle and two tangents -/
structure InscribedQuadrilateral where
  -- Sides of the quadrilateral
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  -- Ensure sides are positive
  side1_pos : side1 > 0
  side2_pos : side2 > 0
  side3_pos : side3 > 0
  side4_pos : side4 > 0
  -- Tangent points on each side
  tangent1 : ℝ
  tangent2 : ℝ
  tangent3 : ℝ
  tangent4 : ℝ
  -- Ensure tangent points are within side lengths
  tangent1_valid : 0 < tangent1 ∧ tangent1 < side1
  tangent2_valid : 0 < tangent2 ∧ tangent2 < side2
  tangent3_valid : 0 < tangent3 ∧ tangent3 < side3
  tangent4_valid : 0 < tangent4 ∧ tangent4 < side4

/-- Theorem about the difference in perimeters of cut-off triangles -/
theorem perimeter_difference_inscribed_quadrilateral 
  (q : InscribedQuadrilateral) 
  (h1 : q.side1 = 3) 
  (h2 : q.side2 = 5) 
  (h3 : q.side3 = 9) 
  (h4 : q.side4 = 7) :
  (2 * (q.tangent3 - q.tangent1) = 4 ∨ 2 * (q.tangent3 - q.tangent1) = 8) ∧
  (2 * (q.tangent4 - q.tangent2) = 4 ∨ 2 * (q.tangent4 - q.tangent2) = 8) :=
sorry

end NUMINAMATH_CALUDE_perimeter_difference_inscribed_quadrilateral_l68_6844


namespace NUMINAMATH_CALUDE_g_of_3_l68_6840

def g (x : ℝ) : ℝ := 5 * x^4 + 4 * x^3 - 7 * x^2 + 3 * x - 2

theorem g_of_3 : g 3 = 401 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l68_6840


namespace NUMINAMATH_CALUDE_tan_alpha_problem_l68_6811

theorem tan_alpha_problem (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  ((Real.sin (2*α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_problem_l68_6811


namespace NUMINAMATH_CALUDE_parabola_properties_l68_6803

/-- Parabola properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_c_neg : c < 0
  h_n_ge_3 : ∃ (n : ℝ), n ≥ 3 ∧ a * n^2 + b * n + c = 0
  h_passes_1_1 : a + b + c = 1
  h_passes_m_0 : ∃ (m : ℝ), a * m^2 + b * m + c = 0

/-- Main theorem -/
theorem parabola_properties (p : Parabola) :
  (p.b > 0) ∧
  (4 * p.a * p.c - p.b^2 < 4 * p.a) ∧
  (∀ (t : ℝ), p.a * 2^2 + p.b * 2 + p.c = t → t > 1) ∧
  (∃ (x : ℝ), p.a * x^2 + p.b * x + p.c = x ∧
    (∃ (m : ℝ), p.a * m^2 + p.b * m + p.c = 0 ∧ 0 < m ∧ m ≤ 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l68_6803


namespace NUMINAMATH_CALUDE_equation_transformations_correct_l68_6879

theorem equation_transformations_correct 
  (a b c x y : ℝ) : 
  (a = b → a * c = b * c) ∧ 
  (a * (x^2 + 1) = b * (x^2 + 1) → a = b) ∧ 
  (a = b → a / c^2 = b / c^2) ∧ 
  (x = y → x - 3 = y - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_transformations_correct_l68_6879


namespace NUMINAMATH_CALUDE_linear_arrangement_paths_count_l68_6861

/-- Represents a linear arrangement of nodes -/
structure LinearArrangement (n : ℕ) where
  nodes : Fin n → ℕ

/-- Counts the number of paths of a given length in a linear arrangement -/
def countPaths (arr : LinearArrangement 10) (length : ℕ) : ℕ :=
  sorry

/-- Theorem: The number of paths of length 4 in a linear arrangement of 10 nodes is 2304 -/
theorem linear_arrangement_paths_count :
  ∀ (arr : LinearArrangement 10), countPaths arr 4 = 2304 := by
  sorry

end NUMINAMATH_CALUDE_linear_arrangement_paths_count_l68_6861


namespace NUMINAMATH_CALUDE_prob_at_least_one_target_l68_6809

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards that are either hearts or kings -/
def target_cards : ℕ := 16

/-- The probability of drawing a card that is not a heart or king -/
def prob_not_target : ℚ := (deck_size - target_cards) / deck_size

/-- The number of draws -/
def num_draws : ℕ := 3

/-- The probability of drawing at least one heart or king in three draws with replacement -/
theorem prob_at_least_one_target :
  1 - prob_not_target ^ num_draws = 1468 / 2197 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_target_l68_6809


namespace NUMINAMATH_CALUDE_algebraic_identities_l68_6839

theorem algebraic_identities (x : ℝ) (h : x + 1/x = 8) :
  (x^2 + 1/x^2 = 62) ∧ (x^3 + 1/x^3 = 488) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l68_6839


namespace NUMINAMATH_CALUDE_cost_of_beads_per_bracelet_l68_6896

/-- Proves the cost of beads per bracelet given the selling price, string cost, number of bracelets sold, and total profit -/
theorem cost_of_beads_per_bracelet 
  (selling_price : ℝ)
  (string_cost : ℝ)
  (bracelets_sold : ℕ)
  (total_profit : ℝ)
  (h1 : selling_price = 6)
  (h2 : string_cost = 1)
  (h3 : bracelets_sold = 25)
  (h4 : total_profit = 50) :
  let bead_cost := (bracelets_sold : ℝ) * selling_price - total_profit - bracelets_sold * string_cost
  bead_cost / (bracelets_sold : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_cost_of_beads_per_bracelet_l68_6896


namespace NUMINAMATH_CALUDE_first_player_win_probability_l68_6868

-- Define the probability of winning in one roll
def prob_win_one_roll : ℚ := 21 / 36

-- Define the probability of not winning in one roll
def prob_not_win_one_roll : ℚ := 1 - prob_win_one_roll

-- Define the game
def dice_game_probability : ℚ :=
  prob_win_one_roll / (1 - prob_not_win_one_roll ^ 2)

-- Theorem statement
theorem first_player_win_probability :
  dice_game_probability = 12 / 17 := by sorry

end NUMINAMATH_CALUDE_first_player_win_probability_l68_6868


namespace NUMINAMATH_CALUDE_shots_per_puppy_l68_6812

/-- Calculates the number of shots each puppy needs given the specified conditions -/
theorem shots_per_puppy
  (num_dogs : ℕ)
  (puppies_per_dog : ℕ)
  (cost_per_shot : ℕ)
  (total_cost : ℕ)
  (h1 : num_dogs = 3)
  (h2 : puppies_per_dog = 4)
  (h3 : cost_per_shot = 5)
  (h4 : total_cost = 120) :
  (total_cost / cost_per_shot) / (num_dogs * puppies_per_dog) = 2 := by
  sorry

#check shots_per_puppy

end NUMINAMATH_CALUDE_shots_per_puppy_l68_6812


namespace NUMINAMATH_CALUDE_y_sum_theorem_l68_6854

theorem y_sum_theorem (y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (eq1 : y₁ + 3*y₂ + 6*y₃ + 10*y₄ + 15*y₅ = 3)
  (eq2 : 3*y₁ + 6*y₂ + 10*y₃ + 15*y₄ + 21*y₅ = 20)
  (eq3 : 6*y₁ + 10*y₂ + 15*y₃ + 21*y₄ + 28*y₅ = 86)
  (eq4 : 10*y₁ + 15*y₂ + 21*y₃ + 28*y₄ + 36*y₅ = 225) :
  15*y₁ + 21*y₂ + 28*y₃ + 36*y₄ + 45*y₅ = 395 := by
  sorry

end NUMINAMATH_CALUDE_y_sum_theorem_l68_6854


namespace NUMINAMATH_CALUDE_vincent_book_cost_l68_6838

theorem vincent_book_cost (animal_books : ℕ) (space_books : ℕ) (train_books : ℕ) (cost_per_book : ℕ) :
  animal_books = 15 →
  space_books = 4 →
  train_books = 6 →
  cost_per_book = 26 →
  (animal_books + space_books + train_books) * cost_per_book = 650 :=
by
  sorry

end NUMINAMATH_CALUDE_vincent_book_cost_l68_6838


namespace NUMINAMATH_CALUDE_min_value_expression_l68_6826

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((3 * a * b - 6 * b + a * (1 - a))^2 + (9 * b^2 + 2 * a + 3 * b * (1 - a))^2) / (a^2 + 9 * b^2) ≥ 4 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    ((3 * a₀ * b₀ - 6 * b₀ + a₀ * (1 - a₀))^2 + (9 * b₀^2 + 2 * a₀ + 3 * b₀ * (1 - a₀))^2) / (a₀^2 + 9 * b₀^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l68_6826


namespace NUMINAMATH_CALUDE_gcd_97_power_plus_one_l68_6801

theorem gcd_97_power_plus_one (p : Nat) (h_prime : Nat.Prime p) (h_p : p = 97) :
  Nat.gcd (p^7 + 1) (p^7 + p^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_97_power_plus_one_l68_6801


namespace NUMINAMATH_CALUDE_quadratic_two_roots_condition_l68_6815

/-- 
Given a quadratic equation kx^2 - 1 = 2x, this theorem states that
for the equation to have two distinct real roots, k must satisfy k > -1 and k ≠ 0.
-/
theorem quadratic_two_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 1 = 2*x ∧ k * y^2 - 1 = 2*y) ↔ (k > -1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_condition_l68_6815


namespace NUMINAMATH_CALUDE_removed_triangles_area_l68_6802

/-- The combined area of four isosceles right triangles removed from the corners of a square
    with side length 20 units to form a regular octagon is 512 square units. -/
theorem removed_triangles_area (square_side : ℝ) (triangle_leg : ℝ) : 
  square_side = 20 →
  (square_side - 2 * triangle_leg)^2 + (triangle_leg - (square_side - 2 * triangle_leg))^2 = square_side^2 →
  4 * (1/2 * triangle_leg^2) = 512 :=
by sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l68_6802


namespace NUMINAMATH_CALUDE_expression_evaluation_l68_6818

theorem expression_evaluation :
  60 + 120 / 15 + 25 * 16 - 220 - 420 / 7 + 3^2 = 197 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l68_6818


namespace NUMINAMATH_CALUDE_min_value_of_f_l68_6878

/-- The function f(x) = x^2 + 8x + 25 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 25

/-- The minimum value of f(x) is 9 -/
theorem min_value_of_f : ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l68_6878


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l68_6865

theorem partial_fraction_decomposition (A B : ℝ) :
  (∀ x : ℝ, (2 * x + 1) / ((x + 1) * (x + 2)) = A / (x + 1) + B / (x + 2)) →
  A = -1 ∧ B = 3 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l68_6865
