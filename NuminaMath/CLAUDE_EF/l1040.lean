import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_high_quality_result_l1040_104039

/-- The probability of a randomly selected part being of high quality -/
noncomputable def prob_high_quality (prob_high_quality_machine1 prob_high_quality_machine2 : ℝ) : ℝ :=
  (2/3) * prob_high_quality_machine1 + (1/3) * prob_high_quality_machine2

/-- Theorem stating the probability of a randomly selected part being of high quality -/
theorem prob_high_quality_result :
  prob_high_quality 0.9 0.81 = 0.87 := by
  -- Unfold the definition of prob_high_quality
  unfold prob_high_quality
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_high_quality_result_l1040_104039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1040_104060

theorem trigonometric_identities (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.cos α = 3 / 5) : 
  Real.sin (Real.pi / 6 + α) = (3 + 4 * Real.sqrt 3) / 10 ∧ 
  Real.cos (Real.pi / 3 + 2 * α) = -(7 + 24 * Real.sqrt 3) / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1040_104060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clara_sells_80_boxes_of_type_2_l1040_104094

/-- Represents the number of cookies in each box type -/
def cookies_per_box : Fin 3 → ℕ
  | 0 => 12  -- Type 1
  | 1 => 20  -- Type 2
  | 2 => 16  -- Type 3

/-- Represents the number of boxes sold for each type -/
def boxes_sold (x : ℕ) : Fin 3 → ℕ
  | 0 => 50  -- Type 1
  | 1 => x   -- Type 2 (unknown)
  | 2 => 70  -- Type 3

/-- The total number of cookies sold -/
def total_cookies : ℕ := 3320

/-- Theorem stating that Clara sells 80 boxes of type 2 -/
theorem clara_sells_80_boxes_of_type_2 :
  ∃ x : ℕ, 
    x = 80 ∧ 
    (boxes_sold x 0) * (cookies_per_box 0) + 
    x * (cookies_per_box 1) + 
    (boxes_sold x 2) * (cookies_per_box 2) = total_cookies :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clara_sells_80_boxes_of_type_2_l1040_104094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AC₁_l1040_104077

-- Define the parallelepiped structure
structure Parallelepiped :=
  (AB AD AA₁ : ℝ)
  (angle_BAD angle_BAA₁ angle_DAA₁ : ℝ)

-- Define the theorem
theorem length_AC₁ (p : Parallelepiped) 
  (h1 : p.AB = 4)
  (h2 : p.AD = 3)
  (h3 : p.AA₁ = 3)
  (h4 : p.angle_BAD = Real.pi / 2)
  (h5 : p.angle_BAA₁ = Real.pi / 3)
  (h6 : p.angle_DAA₁ = Real.pi / 3) :
  Real.sqrt (p.AB ^ 2 + p.AD ^ 2 + p.AA₁ ^ 2 + 
    2 * p.AB * p.AA₁ * Real.cos p.angle_BAA₁ + 
    2 * p.AD * p.AA₁ * Real.cos p.angle_DAA₁) = Real.sqrt 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AC₁_l1040_104077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1040_104083

-- Define the function f
noncomputable def f (a m n : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 + a*x else m*x^2 + n*x

-- State the theorem
theorem f_properties (a m n : ℝ) :
  (∀ x, f a m n x = -f a m n (-x)) →  -- f is odd
  (a = 4 → m = 1 ∧ n = 4) ∧  -- Part 1
  (∀ x ≥ 0, ∀ y ≥ x, f a m n y ≤ f a m n x → a ≤ 0) ∧  -- Part 2
  (∀ t > (5/4 : ℝ), ∀ m : ℝ, f a m n (m-1) + f a m n (m^2+t) < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1040_104083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_theorem_l1040_104032

/-- Represents a bicycle with swappable tires -/
structure Bicycle where
  front_tire_life : ℝ
  rear_tire_life : ℝ

/-- Calculates the maximum distance a bicycle can travel with one tire swap -/
noncomputable def max_distance (b : Bicycle) : ℝ :=
  2 / (1 / b.front_tire_life + 1 / b.rear_tire_life)

/-- Calculates the optimal distance to swap tires -/
noncomputable def swap_distance (b : Bicycle) : ℝ :=
  (b.front_tire_life * b.rear_tire_life) / (b.front_tire_life + b.rear_tire_life)

/-- Theorem stating the maximum distance and optimal swap distance for a specific bicycle -/
theorem bicycle_theorem (b : Bicycle) 
  (h1 : b.front_tire_life = 11000)
  (h2 : b.rear_tire_life = 9000) :
  max_distance b = 9900 ∧ swap_distance b = 4950 := by
  sorry

/-- Example calculations (using noncomputable) -/
noncomputable example : ℝ := max_distance { front_tire_life := 11000, rear_tire_life := 9000 }
noncomputable example : ℝ := swap_distance { front_tire_life := 11000, rear_tire_life := 9000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_theorem_l1040_104032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dm_eq_ec_l1040_104021

open EuclideanGeometry

-- Define the points in the Euclidean plane
variable (A B C O M D E : EuclideanSpace ℝ (Fin 2))

-- Define predicates
variable (IsAcute : EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → Prop)
variable (IsCircumcenter : EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → Prop)
variable (OnCircle : EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → Prop)
variable (Collinear : EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → Prop)
variable (OnSegment : EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → Prop)

-- Hypotheses
variable (h_acute : IsAcute A B C)
variable (h_AB_lt_AC : dist A B < dist A C)
variable (h_O_circumcenter : IsCircumcenter O A B C)
variable (h_M_midpoint : M = (B + C) / 2)
variable (h_circle : OnCircle A O M ∧ OnCircle D O M ∧ OnCircle E O M)
variable (h_D_on_AB : Collinear A B D)
variable (h_E_on_AC : OnSegment A C E)

-- Theorem statement
theorem dm_eq_ec : dist D M = dist E C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dm_eq_ec_l1040_104021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_decreasing_function_inequality_l1040_104016

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f y < f x

theorem even_decreasing_function_inequality 
  (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_decreasing : monotone_decreasing_on f (Set.Ioi 0)) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_decreasing_function_inequality_l1040_104016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1040_104090

/-- Theorem about triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Condition 1: sin(2B) = √3 * sin(B)
  Real.sin (2 * B) = Real.sqrt 3 * Real.sin B →
  -- Condition 2: a = 2b * cos(C)
  a = 2 * b * Real.cos C →
  -- Conclusion 1: B = π/6
  B = Real.pi / 6 ∧
  -- Conclusion 2: Triangle ABC is isosceles (b = c)
  b = c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1040_104090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_power_seven_l1040_104013

theorem multiple_of_power_seven (k : ℕ) (a : ℤ) 
  (h1 : ∃ m : ℤ, a - 2 = 7 * m) 
  (h2 : ∃ n : ℤ, a^6 - 1 = (7:ℤ)^k * n) :
  ∃ p : ℤ, (a + 1)^6 - 1 = (7:ℤ)^k * p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_power_seven_l1040_104013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_term_formula_l1040_104065

def sequence_a (b c : ℕ) : ℕ → ℕ
| 0 => b * (b + 1) / 2 + c
| (n + 1) => let an := sequence_a b c n
             if an ≤ n + 1 then an + (n + 1) else an - (n + 1)

def fixed_term (b c : ℕ) (m : ℕ) : ℕ :=
  ((2 * b + 4 * c - 1) * 3^(m - 1) + 1) / 2

theorem fixed_term_formula (b c : ℕ) (hb : b > 0) (hc : c ≤ b - 1) :
  fixed_term b c 1 = b + 2 * c ∧
  ∀ m : ℕ, m > 0 → fixed_term b c m = ((2 * b + 4 * c - 1) * 3^(m - 1) + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_term_formula_l1040_104065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l1040_104028

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 4*x + 3 else -x^2 - 2*x + 3

-- State the theorem
theorem min_a_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f (x + a) ≥ f (2*a - x)) →
  a ≥ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l1040_104028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nikolai_wins_l1040_104050

/-- Represents the investment details of each friend -/
structure Investment where
  initial_amount : ℝ
  final_amount : ℝ

/-- Calculates the yield of an investment -/
noncomputable def yield (inv : Investment) : ℝ :=
  (inv.final_amount - inv.initial_amount) / inv.initial_amount * 100

/-- Nikolai's investment -/
noncomputable def nikolai : Investment :=
  { initial_amount := 150000,
    final_amount := 150000 * (1 + 0.071 / 12) ^ 12 }

/-- Maxim's investment -/
noncomputable def maxim : Investment :=
  { initial_amount := 80000,
    final_amount := (80000 / 58.42) * (1 + 0.036) * 58.61 }

/-- Oksana's investment -/
noncomputable def oksana : Investment :=
  { initial_amount := 197040,
    final_amount := (197040 / 19704) * 19801 }

/-- Olga's investment -/
noncomputable def olga : Investment :=
  { initial_amount := 250000,
    final_amount := (250000 / 5809) * 8074 * 0.96 }

/-- Theorem stating that Nikolai's yield is the highest -/
theorem nikolai_wins :
  yield nikolai > yield maxim ∧
  yield nikolai > yield oksana ∧
  yield nikolai > yield olga := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nikolai_wins_l1040_104050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_of_list_l1040_104015

def median (l : List ℕ) : ℕ := sorry

theorem max_element_of_list (l : List ℕ) : 
  l.length = 5 ∧ 
  l.all (· > 0) ∧
  median l = 3 ∧
  l.sum / l.length = 12 →
  l.maximum ≤ 52 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_of_list_l1040_104015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_rational_root_odd_coefficients_l1040_104078

theorem quadratic_rational_root_odd_coefficients 
  (a b c : ℕ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_quad : ∃ (x : ℚ), a * x^2 + b * x + c = 0) 
  (h_a_nonzero : a ≠ 0) :
  ¬(a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_rational_root_odd_coefficients_l1040_104078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_triangle_points_l1040_104042

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Calculates the area of a triangle given three grid points -/
def triangleArea (a b c : GridPoint) : Int :=
  ((a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)).natAbs) / 2

/-- Theorem: There are exactly 8 grid points A such that the area of triangle ABC is 3 square units -/
theorem eight_triangle_points (b c : GridPoint) : 
  (∃ (points : List GridPoint), 
    points.length = 8 ∧ 
    (∀ a ∈ points, 0 ≤ a.x ∧ a.x ≤ 8 ∧ 0 ≤ a.y ∧ a.y ≤ 8) ∧
    (∀ a ∈ points, triangleArea a b c = 3) ∧
    (∀ a : GridPoint, 0 ≤ a.x ∧ a.x ≤ 8 ∧ 0 ≤ a.y ∧ a.y ≤ 8 → 
      triangleArea a b c = 3 → a ∈ points)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_triangle_points_l1040_104042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1040_104059

noncomputable def f (x : ℝ) : ℝ := (2*x - 5)*(x - 4) / (x + 2)

theorem inequality_solution (x : ℝ) :
  x ≠ -2 →
  (f x ≥ 0 ↔ x ∈ Set.Ici 4 ∪ Set.Iio (-2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1040_104059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_hare_race_head_start_l1040_104047

/-- The head start time needed for a turtle to tie with a hare in a race -/
noncomputable def headStartTime (raceDistance : ℝ) (hareSpeed turtleSpeed : ℝ) : ℝ :=
  raceDistance / turtleSpeed - raceDistance / hareSpeed

/-- Theorem stating that the head start time for the given race conditions is 18 seconds -/
theorem turtle_hare_race_head_start :
  headStartTime 20 10 1 = 18 := by
  -- Unfold the definition of headStartTime
  unfold headStartTime
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_hare_race_head_start_l1040_104047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l1040_104071

/-- The ellipse defined by the equation x²/9 + y²/4 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The line defined by the equation x + 2y - 10 = 0 -/
def line (x y : ℝ) : Prop := x + 2*y - 10 = 0

/-- The distance from a point (x, y) to the line x + 2y - 10 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x + 2*y - 10| / Real.sqrt 5

/-- The minimum distance from any point on the ellipse to the line is √5 -/
theorem min_distance_ellipse_to_line :
  ∃ (d : ℝ), d = Real.sqrt 5 ∧
  ∀ (x y : ℝ), ellipse x y →
    distance_to_line x y ≥ d ∧
    ∃ (x₀ y₀ : ℝ), ellipse x₀ y₀ ∧ distance_to_line x₀ y₀ = d := by
  sorry

#check min_distance_ellipse_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l1040_104071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_factory_theorem_l1040_104020

/-- Furniture factory pricing and discount options -/
structure FurnitureFactory where
  desk_price : ℕ
  chair_price : ℕ
  option1 : ℕ → ℕ  -- Cost function for Option 1
  option2 : ℕ → ℕ  -- Cost function for Option 2
  combined_option : ℕ → ℕ  -- Cost function for combined option

/-- The specific furniture factory in the problem -/
def problem_factory : FurnitureFactory := {
  desk_price := 200,
  chair_price := 80,
  option1 := λ x ↦ 100 * 200 + 80 * (x - 100),
  option2 := λ x ↦ ((100 * 200 + 80 * x) * 80) / 100,
  combined_option := λ x ↦ 100 * 200 + ((x - 100) * 80 * 80) / 100
}

theorem furniture_factory_theorem (f : FurnitureFactory) :
  (f.option1 100 < f.option2 100) ∧
  (∀ x > 100, f.option1 x = 80 * x + 12000 ∧ f.option2 x = 64 * x + 16000) ∧
  (min (f.option1 300) (min (f.option2 300) (f.combined_option 300)) = f.combined_option 300) := by
  sorry

#eval problem_factory.option1 100
#eval problem_factory.option2 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_factory_theorem_l1040_104020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_difference_is_six_l1040_104056

/-- Represents the fuel efficiency of a car in different driving conditions -/
structure CarFuelEfficiency where
  highway_miles_per_tank : ℚ
  city_miles_per_tank : ℚ
  city_miles_per_gallon : ℚ

/-- Calculates the difference in miles per gallon between highway and city driving -/
noncomputable def mpg_difference (car : CarFuelEfficiency) : ℚ :=
  let tank_capacity := car.city_miles_per_tank / car.city_miles_per_gallon
  let highway_mpg := car.highway_miles_per_tank / tank_capacity
  highway_mpg - car.city_miles_per_gallon

/-- Theorem: The difference in miles per gallon between highway and city driving is 6 -/
theorem mpg_difference_is_six (car : CarFuelEfficiency)
    (h1 : car.highway_miles_per_tank = 560)
    (h2 : car.city_miles_per_tank = 336)
    (h3 : car.city_miles_per_gallon = 9) :
    mpg_difference car = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_difference_is_six_l1040_104056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_range_l1040_104067

/-- The range of a for the given quadratic function and intersection condition -/
theorem quadratic_function_range (a : ℝ) : 
  let f := λ x : ℝ => a * x^2 - 2 * x - 2 * a
  let A := {x : ℝ | f x > 0}
  let B := {x : ℝ | 1 < x ∧ x < 3}
  (∃ x, x ∈ A ∩ B) → a ∈ Set.Ioi (-2) ∪ Set.Ioi (6/7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_range_l1040_104067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_cube_root_l1040_104058

/-- Represents the shape of an inverted right circular cone water tank -/
structure WaterTank where
  baseRadius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (r : ℝ) (h : ℝ) : ℝ :=
  (1/3) * Real.pi * r^2 * h

/-- Represents the water level in the tank -/
noncomputable def waterHeight (tank : WaterTank) (fillPercentage : ℝ) : ℝ :=
  tank.height * (fillPercentage^(1/3))

theorem water_height_cube_root (tank : WaterTank) :
  tank.baseRadius = 20 →
  tank.height = 120 →
  waterHeight tank 0.3 = 60 * (0.6 : ℝ)^(1/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_cube_root_l1040_104058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_rental_cost_l1040_104041

/-- Represents the rental business scenario --/
structure RentalBusiness where
  canoe_cost : ℚ
  kayak_cost : ℚ
  canoe_count : ℕ
  kayak_count : ℕ
  total_revenue : ℚ

/-- The rental business satisfies the given conditions --/
def valid_rental_business (rb : RentalBusiness) : Prop :=
  rb.kayak_cost = 18 ∧
  rb.canoe_count = (3 * rb.kayak_count) / 2 ∧
  rb.canoe_count = rb.kayak_count + 7 ∧
  rb.total_revenue = rb.canoe_cost * rb.canoe_count + rb.kayak_cost * rb.kayak_count ∧
  rb.total_revenue = 504

/-- The theorem stating that a canoe rental costs $12 per day --/
theorem canoe_rental_cost (rb : RentalBusiness) 
  (h : valid_rental_business rb) : rb.canoe_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_rental_cost_l1040_104041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_congruence_l1040_104091

/-- S(n) is the sum of the digits of a positive integer n -/
def S (n : ℕ+) : ℕ := sorry

/-- Theorem: For any positive integer n where S(n) = 1465, 
    S(n+1) is not congruent to 2, 5, 24, 1460, or 1464 modulo 9 -/
theorem sum_of_digits_congruence (n : ℕ+) (h : S n = 1465) :
  ¬(S (n + 1) % 9 = 2 % 9 ∨ 
    S (n + 1) % 9 = 5 % 9 ∨ 
    S (n + 1) % 9 = 24 % 9 ∨ 
    S (n + 1) % 9 = 1460 % 9 ∨ 
    S (n + 1) % 9 = 1464 % 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_congruence_l1040_104091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_box_same_color_m_plus_n_equals_471_l1040_104008

-- Define the number of people, boxes, and colors
def num_people : ℕ := 3
def num_boxes : ℕ := 5
def num_colors : ℕ := 5

-- Define the probability of at least one box containing three blocks of the same color
def prob_same_color : ℚ := 71 / 400

-- State the theorem
theorem at_least_one_box_same_color :
  (num_boxes * num_colors * (num_boxes - 1).factorial ^ num_people -
   (num_boxes.choose 2) * num_colors * (num_colors - 1) * (num_boxes - 2).factorial ^ num_people +
   (num_boxes.choose 3) * num_colors * (num_colors - 1) * (num_colors - 2) * (num_boxes - 3).factorial ^ num_people -
   (num_boxes.choose 4) * num_colors * (num_colors - 1) * (num_colors - 2) * (num_colors - 3) * (num_boxes - 4).factorial ^ num_people +
   num_colors.factorial : ℚ) / (num_boxes.factorial ^ num_people : ℚ) = prob_same_color := by
  sorry

-- Define m and n
def m : ℕ := 71
def n : ℕ := 400

-- State the theorem for m + n
theorem m_plus_n_equals_471 : m + n = 471 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_box_same_color_m_plus_n_equals_471_l1040_104008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_significant_digits_eq_area_root_significant_digits_side_length_has_five_significant_digits_l1040_104019

/-- The area of the square in square inches -/
def square_area : ℝ := 1.1025

/-- The precision of the area measurement in square inches -/
def area_precision : ℝ := 0.0001

/-- The number of significant digits in a real number -/
def significant_digits (x : ℝ) : ℕ := sorry

/-- The side length of the square -/
noncomputable def side_length : ℝ := Real.sqrt square_area

theorem side_significant_digits_eq_area_root_significant_digits :
  significant_digits side_length = significant_digits (Real.sqrt square_area) :=
by sorry

/-- The main theorem stating that the number of significant digits in the side length is 5 -/
theorem side_length_has_five_significant_digits :
  significant_digits side_length = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_significant_digits_eq_area_root_significant_digits_side_length_has_five_significant_digits_l1040_104019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1040_104026

-- Define the vector type
def Vector2D := ℝ × ℝ

-- Define vector addition
def add_vectors (v w : Vector2D) : Vector2D :=
  (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def scalar_mult (k : ℝ) (v : Vector2D) : Vector2D :=
  (k * v.1, k * v.2)

-- Define parallel vectors
def parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v = scalar_mult k w

theorem vector_problem (a b c : Vector2D) (m : ℝ) :
  a = (1, -2) →
  b = (2, m) →
  parallel a b →
  add_vectors (add_vectors (scalar_mult 3 a) (scalar_mult 2 b)) c = (0, 0) →
  c = (-7, 14) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1040_104026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l1040_104034

-- Define the ellipse C₁
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the parabola C₂
def parabola (x y : ℝ) : Prop :=
  x^2 = -2*y

-- Define the line l
def line (x y k : ℝ) : Prop :=
  y = k*x + 2

-- Define the set of valid slopes
def valid_slopes : Set ℝ :=
  {k : ℝ | (-2 < k ∧ k < -Real.sqrt 3 / 2) ∨ (Real.sqrt 3 / 2 < k ∧ k < 2)}

-- Main theorem
theorem slope_range :
  ∀ k : ℝ, ∀ P Q : ℝ × ℝ,
  (∃ x y, ellipse x y ∧ line x y k) →
  (∃ x y, ellipse x y ∧ line x y k ∧ (x, y) ≠ P) →
  (P.1^2 + P.2^2) * (Q.1^2 + Q.2^2) = ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) * (P.1*Q.1 + P.2*Q.2) →
  k ∈ valid_slopes :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l1040_104034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1040_104086

noncomputable def f (x : ℝ) := 1 - 2 / (Real.exp (x * Real.log 5) + 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, x < y → f x < f y) ∧
  (Set.Icc (-2/3 : ℝ) (12/13) = {y | ∃ x ∈ Set.Ico (-1 : ℝ) 2, f x = y}) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1040_104086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lewis_weekly_earnings_l1040_104023

/-- Calculates the weekly earnings of a worker given their total earnings, 
    weeks per season, and number of seasons worked. -/
def weekly_earnings (total_earnings : ℚ) (weeks_per_season : ℕ) (num_seasons : ℕ) : ℚ :=
  total_earnings / (weeks_per_season * num_seasons)

/-- Theorem stating that a worker earning $22090603 over 73 seasons of 223 weeks each
    earns $1357.14 per week (rounded to two decimal places). -/
theorem lewis_weekly_earnings :
  let total_earnings : ℚ := 22090603
  let weeks_per_season : ℕ := 223
  let num_seasons : ℕ := 73
  let calculated_earnings := weekly_earnings total_earnings weeks_per_season num_seasons
  (⌊calculated_earnings * 100⌋ : ℚ) / 100 = 1357.14 := by
  sorry

#eval weekly_earnings 22090603 223 73

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lewis_weekly_earnings_l1040_104023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1040_104063

/-- The system of equations from part (a) -/
def system (x y z : ℝ) : Prop :=
  x + 3*y = 4*y^3 ∧ y + 3*z = 4*z^3 ∧ z + 3*x = 4*x^3

/-- The set of solutions for the system -/
noncomputable def solutions : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (1, 1, 1), (-1, -1, -1),
   (Real.cos (Real.pi/14), -Real.cos (5*Real.pi/14), Real.cos (3*Real.pi/14)),
   (-Real.cos (Real.pi/14), Real.cos (5*Real.pi/14), -Real.cos (3*Real.pi/14)),
   (Real.cos (Real.pi/7), -Real.cos (2*Real.pi/7), Real.cos (3*Real.pi/7)),
   (-Real.cos (Real.pi/7), Real.cos (2*Real.pi/7), -Real.cos (3*Real.pi/7)),
   (Real.cos (Real.pi/13), -Real.cos (4*Real.pi/13), Real.cos (3*Real.pi/13)),
   (-Real.cos (Real.pi/13), Real.cos (4*Real.pi/13), -Real.cos (3*Real.pi/13)),
   (Real.cos (2*Real.pi/13), -Real.cos (5*Real.pi/13), Real.cos (6*Real.pi/13)),
   (-Real.cos (2*Real.pi/13), Real.cos (5*Real.pi/13), -Real.cos (6*Real.pi/13))}

/-- The main theorem stating that the solutions set contains exactly all solutions to the system -/
theorem system_solutions :
  ∀ x y z, system x y z ↔ (x, y, z) ∈ solutions := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1040_104063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_scalar_multiple_l1040_104057

/-- Given a 2x2 matrix B with B^(-1) = m * B, prove that e = -3 and m = 1/11 -/
theorem inverse_scalar_multiple (e m : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, e]
  (B⁻¹ = m • B) → e = -3 ∧ m = 1/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_scalar_multiple_l1040_104057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_zero_l1040_104079

-- Define the distribution function F(x)
noncomputable def F (c : ℝ) (x : ℝ) : ℝ :=
  if x ≤ -c then 0
  else if -c < x ∧ x ≤ c then 1/2 + (1/Real.pi) * Real.arcsin (x/c)
  else 1

-- Define the random variable X
def X (c : ℝ) : Type := ℝ

-- Define the expected value function
noncomputable def expectedValue (c : ℝ) (X : Type) (F : ℝ → ℝ) : ℝ :=
  ∫ x in Set.Icc (-c) c, x * (1 / (Real.pi * Real.sqrt (c^2 - x^2)))

-- Theorem statement
theorem expected_value_zero (c : ℝ) (hc : c > 0) :
  expectedValue c (X c) (F c) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_zero_l1040_104079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l1040_104022

/-- The probability of drawing two balls of different colors from a bag containing 5 balls, 
    of which 3 are white and 2 are yellow, when randomly selecting two balls at once. -/
theorem different_color_probability (total : ℕ) (white : ℕ) (yellow : ℕ) : 
  total = 5 → white = 3 → yellow = 2 → 
  (Nat.choose total 2 : ℚ) ≠ 0 → 
  (Nat.choose white 1 * Nat.choose yellow 1 : ℚ) / (Nat.choose total 2) = 6/10 := by
  sorry

#check different_color_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l1040_104022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l1040_104098

/-- The curve function --/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

/-- The center of symmetry --/
noncomputable def center : ℝ × ℝ := (3 * Real.pi / 8, 0)

/-- Theorem stating that the given point is a center of symmetry for the curve --/
theorem center_of_symmetry :
  ∀ (t : ℝ), f (center.1 + t) = f (center.1 - t) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l1040_104098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1040_104045

noncomputable section

variable (f : ℝ → ℝ)

axiom f_nonzero : ∀ x, f x ≠ 0
axiom f_product : ∀ x y, f x * f y = f (x + y)
axiom f_negative : ∀ x, x < 0 → f x > 1

theorem f_properties :
  (f 0 = 1 ∧ ∀ x, f x > 0) ∧
  (∀ x y, x < y → f x > f y) ∧
  (f 4 = 1/16 → ∀ a ∈ Set.Icc (-1) 1, 
    ∀ x, f (x^2 - 2*a*x + 2) ≤ 1/4 → x ≤ -2 ∨ x = 0 ∨ x ≥ 2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1040_104045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_norm_scalar_multiple_l1040_104029

variable {V : Type*} [NormedAddCommGroup V] [Module ℝ V]

theorem norm_scalar_multiple (v : V) (h : ‖v‖ = 6) : ‖(-5 : ℝ) • v‖ = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_norm_scalar_multiple_l1040_104029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1040_104025

noncomputable def f (x : ℝ) := Real.sqrt (-15 * x^2 + 13 * x + 8)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = 
  {x : ℝ | (13 - Real.sqrt 649) / 30 ≤ x ∧ x ≤ (13 + Real.sqrt 649) / 30} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1040_104025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1040_104096

theorem calculate_expression : (-3 : ℤ)^2 * (3 : ℚ)⁻¹ + (-5 + 2) + |(-2 : ℤ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1040_104096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_subsets_of_intersection_l1040_104001

def A : Finset Char := {'a', 'b', 'c', 'd', 'e'}
def B : Finset Char := {'b', 'e', 'f'}

theorem num_subsets_of_intersection : Finset.card (Finset.powerset (A ∩ B)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_subsets_of_intersection_l1040_104001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_pairs_theorem_l1040_104031

noncomputable def tg (x : ℝ) : ℝ := Real.tan x

theorem tangent_pairs_theorem (a b : ℝ) :
  (∃ m n : ℕ, tg a = m ∧ tg b = n) →
  (∃ k : ℤ, tg (a + b) = k) →
  (tg a = 1 ∧ tg b = 2) ∨
  (tg a = 1 ∧ tg b = 3) ∨
  (tg a = 2 ∧ tg b = 1) ∨
  (tg a = 3 ∧ tg b = 1) ∨
  (tg a = 2 ∧ tg b = 3) ∨
  (tg a = 3 ∧ tg b = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_pairs_theorem_l1040_104031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l1040_104064

/-- Given two acute angles α and β in the Cartesian plane, with their terminal sides 
    intersecting the unit circle at points A and B, prove the following properties. -/
theorem angle_properties (α β : ℝ) (A B : ℝ × ℝ) :
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  A.1 = Real.sqrt 5 / 5 →  -- x-coordinate of A
  B.2 = Real.sqrt 2 / 10 →  -- y-coordinate of B
  A.1^2 + A.2^2 = 1 →  -- A is on the unit circle
  B.1^2 + B.2^2 = 1 →  -- B is on the unit circle
  (∃ t : ℝ, t > 0 ∧ A = (t * Real.cos α, t * Real.sin α)) →  -- A is on the terminal side of α
  (∃ t : ℝ, t > 0 ∧ B = (t * Real.cos β, t * Real.sin β)) →  -- B is on the terminal side of β
  Real.tan (α + β) = 3 ∧ 2*α + β = 3*π/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l1040_104064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_points_l1040_104088

/-- An equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 1

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A set of 5 points -/
def FivePoints := Fin 5 → Point

/-- Main theorem -/
theorem exists_close_points (t : EquilateralTriangle) (points : FivePoints) : 
  ∃ (i j : Fin 5), i ≠ j ∧ distance (points i) (points j) < 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_points_l1040_104088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_properties_l1040_104099

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.cos (Real.pi / 2 - 2 * x) - 2 * (Real.cos x) ^ 2 + 1

noncomputable def g (x m : ℝ) := f (x + m)

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) := ∀ x, f (a - x) = f (a + x)

theorem f_period_and_g_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  (∃ m > 0, is_symmetric_about (g · m) (Real.pi / 8)) ∧
  (∀ m > 0, is_symmetric_about (g · m) (Real.pi / 8) → m ≥ 5 * Real.pi / 24) ∧
  (let m := 5 * Real.pi / 24; ∀ x ∈ Set.Icc 0 (Real.pi / 4), Real.sqrt 2 ≤ g x m ∧ g x m ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_properties_l1040_104099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_two_digit_permutation_l1040_104093

/-- Represents the decimal digits of a natural number as a list. -/
def digits (n : ℕ) : List ℕ :=
  sorry

/-- Given two positive integers r and s, if 2^r is a permutation of the digits of 2^s
    in decimal expansion, then r = s. -/
theorem power_two_digit_permutation (r s : ℕ+) 
  (h : ∃ (perm : List ℕ → List ℕ), 
       perm (digits (2^(r:ℕ))) = digits (2^(s:ℕ)) ∧ 
       Function.Bijective perm) : 
  r = s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_two_digit_permutation_l1040_104093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_distance_l1040_104003

-- Define the circle and point P
def Circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def O : ℝ × ℝ := (0, 0)
variable (P : ℝ × ℝ)

-- Define the tangent lines and their angle
variable (tangent1 : ℝ × ℝ → ℝ)
variable (tangent2 : ℝ × ℝ → ℝ)
variable (angle_between_tangents : ℝ)

-- Theorem statement
theorem tangent_circle_distance 
  (h1 : P ∉ Circle)
  (h2 : angle_between_tangents = π/3)
  : Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_distance_l1040_104003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_8_equals_13_div_3_l1040_104055

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (3 * x + 2) / (x - 2)

-- Theorem statement
theorem g_of_8_equals_13_div_3 : g 8 = 13 / 3 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expression
  simp [add_div, mul_div_assoc]
  -- Perform arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_8_equals_13_div_3_l1040_104055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_l1040_104014

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 5

-- Define the line ax + y - 1 = 0
def my_line (a x y : ℝ) : Prop := a * x + y - 1 = 0

-- Define the point M
def point_M : ℝ × ℝ := (1, 1)

-- Define the tangent line l
def line_l (x y : ℝ) : Prop := ∃ (m b : ℝ), y = m * x + b ∧ (1 : ℝ) = m * 1 + b

-- State the theorem
theorem tangent_line_perpendicular (a : ℝ) : 
  (∃ (x y : ℝ), line_l x y ∧ my_circle x y) → -- l is tangent to the circle
  (∀ (x y : ℝ), line_l x y → (x, y) = point_M ∨ (x, y) ≠ point_M) → -- l passes through M(1,1)
  (∀ (x₁ y₁ x₂ y₂ : ℝ), line_l x₁ y₁ → line_l x₂ y₂ → x₁ ≠ x₂ → 
    (y₂ - y₁) / (x₂ - x₁) * (1 / a) = -1) → -- l is perpendicular to ax + y - 1 = 0
  a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_l1040_104014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_ratio_l1040_104006

-- Define the side length of the equilateral triangle
variable (s : ℝ)

-- Define the radius of the circle
variable (r : ℝ)

-- Define the area of the hexagon inscribed in the triangle
noncomputable def area_hexagon_triangle (s : ℝ) : ℝ := (27 * Real.sqrt 3 * s^2) / 32

-- Define the area of the hexagon inscribed in the circle
noncomputable def area_hexagon_circle (r : ℝ) : ℝ := (3 * Real.sqrt 3 * r^2) / 2

-- State the theorem
theorem hexagon_area_ratio (h : s = r * Real.sqrt 3) :
  area_hexagon_triangle s / area_hexagon_circle r = 9 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_ratio_l1040_104006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_30_consecutive_even_integers_sum_12000_l1040_104035

/-- A sequence of consecutive even integers -/
def ConsecutiveEvenIntegers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => start + 2 * i)

theorem largest_of_30_consecutive_even_integers_sum_12000 :
  ∃ start : ℤ,
    let seq := ConsecutiveEvenIntegers start 30
    (seq.sum = 12000) ∧ 
    (seq.getLast? = some 429) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_30_consecutive_even_integers_sum_12000_l1040_104035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l1040_104052

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6)

theorem g_monotone_increasing :
  ∀ k : ℤ, StrictMonoOn g (Set.Icc (- 5 * Real.pi / 12 + k * Real.pi) (Real.pi / 12 + k * Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l1040_104052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sinx_over_2_minus_cosx_l1040_104007

theorem max_value_sinx_over_2_minus_cosx :
  ∀ x : ℝ, Real.sin x / (2 - Real.cos x) ≤ Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sinx_over_2_minus_cosx_l1040_104007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_position_number_and_distance_bounds_l1040_104061

/-- The position number of a grid point (k,k) in the described traversal pattern -/
def position_number (k : ℕ) : ℕ :=
  if k % 2 = 1 then 4 * k^2 - 4 * k + 2
  else 4 * k^2 - 2 * k + 2

/-- The distance from the origin to a point with position number N -/
noncomputable def distance (N : ℕ) : ℝ := Real.sqrt (2 * (N / 4 : ℝ))

theorem position_number_and_distance_bounds :
  ∀ (k : ℕ) (N : ℕ),
    (∃ (x y : ℕ), x^2 + y^2 = k^2 ∧ position_number k = N) →
    (Real.sqrt 2 * ⌊Real.sqrt (N / 4 : ℝ)⌋ ≤ distance N) ∧
    (distance N ≤ Real.sqrt 2 * ⌈Real.sqrt (N / 4 : ℝ)⌉) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_position_number_and_distance_bounds_l1040_104061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_T_is_three_l1040_104097

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def T : ℕ := (List.range 60).map factorial |>.sum

theorem units_digit_of_T_is_three : T % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_T_is_three_l1040_104097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_swims_18km_upstream_l1040_104049

/-- Represents the distance swam upstream by a man given his speed in still water,
    downstream distance, and time spent swimming both downstream and upstream. -/
noncomputable def distance_upstream (speed_still_water : ℝ) (downstream_distance : ℝ) (time : ℝ) : ℝ :=
  let stream_speed := downstream_distance / time - speed_still_water
  (speed_still_water - stream_speed) * time

/-- Theorem stating that under the given conditions, the man swims 18 km upstream. -/
theorem man_swims_18km_upstream (speed_still_water : ℝ) (downstream_distance : ℝ) (time : ℝ)
    (h1 : speed_still_water = 9)
    (h2 : downstream_distance = 36)
    (h3 : time = 3) :
    distance_upstream speed_still_water downstream_distance time = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_swims_18km_upstream_l1040_104049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_four_l1040_104074

theorem reciprocal_of_negative_four :
  (fun x : ℚ => 1 / x) (-4) = -1 / 4 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_four_l1040_104074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_cosine_l1040_104040

theorem arithmetic_progression_cosine (x y z : ℝ) : 
  let α := Real.arccos (-3/7)
  (∃ d, y = x + d ∧ z = y + d) →  -- x, y, z form an arithmetic progression
  (∃ k, 7/Real.cos y = 1/Real.cos x + k ∧ 1/Real.cos z = 7/Real.cos y + k) →  -- 1/cos(x), 7/cos(y), 1/cos(z) form an arithmetic progression
  Real.cos y^2 = 10/13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_cosine_l1040_104040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l1040_104005

-- Define the points
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (16, 0)

-- Define D and E based on the given conditions
def D : ℝ × ℝ := (0, 6)
def E : ℝ × ℝ := (12, 0)

-- Function to calculate distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem area_of_triangle_DBC : 
  distance D B = (3/4) * distance A B ∧ 
  distance B E = (3/4) * distance B C ∧
  (1/2) * distance B C * D.2 = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l1040_104005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_product_is_3_32768_l1040_104033

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of sides on each die -/
def sides_per_die : ℕ := 8

/-- The set of prime numbers achievable as a product when rolling the dice -/
def achievable_primes : Finset ℕ := {2, 3, 5, 7}

/-- The total number of possible outcomes when rolling the dice -/
def total_outcomes : ℕ := sides_per_die ^ num_dice

/-- The number of favorable outcomes (ways to get a prime product) -/
def favorable_outcomes : ℕ := num_dice * achievable_primes.card

/-- The probability of getting a prime product when rolling the dice -/
noncomputable def probability_prime_product : ℚ := 
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_prime_product_is_3_32768 : 
  probability_prime_product = 3 / 32768 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_product_is_3_32768_l1040_104033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin_plus_cos_l1040_104038

theorem min_sin_plus_cos :
  ∃ (min_value : ℝ) (min_angle : ℝ),
    (∀ A : ℝ, Real.sin (A / 2) + Real.cos (A / 2) ≥ min_value) ∧
    (Real.sin (min_angle / 2) + Real.cos (min_angle / 2) = min_value) ∧
    min_value = -Real.sqrt 2 ∧
    min_angle = 450 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin_plus_cos_l1040_104038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_has_ten_cards_l1040_104084

/-- Represents the number of cards each person has -/
structure CardCounts where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ

/-- Checks if the given card counts satisfy all the true statements -/
def satisfiesTrueStatements (counts : CardCounts) : Prop :=
  (counts.A = counts.C + 16 ∨ counts.D = counts.C + 6 ∨ counts.A = counts.D + 9 ∨ counts.D + 2 = 3 * counts.C) ∧
  (counts.A ≠ counts.B) ∧ (counts.A ≠ counts.C) ∧ (counts.A ≠ counts.D) ∧
  (counts.B ≠ counts.C) ∧ (counts.B ≠ counts.D) ∧ (counts.C ≠ counts.D)

/-- Checks if the given card counts have exactly one false statement -/
def hasExactlyOneFalseStatement (counts : CardCounts) : Prop :=
  ((counts.A ≠ counts.C + 16) ↔ (counts.D = counts.C + 6 ∧ counts.A = counts.D + 9 ∧ counts.D + 2 = 3 * counts.C)) ∧
  ((counts.D ≠ counts.C + 6) ↔ (counts.A = counts.C + 16 ∧ counts.A = counts.D + 9 ∧ counts.D + 2 = 3 * counts.C)) ∧
  ((counts.A ≠ counts.D + 9) ↔ (counts.A = counts.C + 16 ∧ counts.D = counts.C + 6 ∧ counts.D + 2 = 3 * counts.C)) ∧
  ((counts.D + 2 ≠ 3 * counts.C) ↔ (counts.A = counts.C + 16 ∧ counts.D = counts.C + 6 ∧ counts.A = counts.D + 9))

/-- The main theorem stating that D has 10 cards -/
theorem d_has_ten_cards : ∃ (counts : CardCounts), 
  satisfiesTrueStatements counts ∧ 
  hasExactlyOneFalseStatement counts ∧ 
  counts.D = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_has_ten_cards_l1040_104084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_formula_l1040_104081

-- Define a point in 2D plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance function between two points
noncomputable def distance (A B : Point2D) : ℝ :=
  Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

-- Theorem: The distance between two points is equal to the given formula
theorem distance_formula (A B : Point2D) :
  distance A B = Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2) := by
  -- Unfold the definition of distance
  unfold distance
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_formula_l1040_104081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_in_equilateral_triangle_l1040_104004

/-- An equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : ∀ X Y, (X = A ∧ Y = B) ∨ (X = B ∧ Y = C) ∨ (X = C ∧ Y = A) → 
    Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 1

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Vector from point X to point Y -/
def vector (X Y : ℝ × ℝ) : ℝ × ℝ := (Y.1 - X.1, Y.2 - X.2)

theorem dot_product_in_equilateral_triangle (t : EquilateralTriangle) :
  dot_product (vector t.A t.B) (vector t.A t.C) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_in_equilateral_triangle_l1040_104004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equidistant_point_l1040_104062

/-- A line passing through (A, A) with slope 0.5 and equidistant from (0, 2) and (12, 8) has A = 4 -/
theorem line_equidistant_point (A : ℝ) : 
  let line := λ x : ℝ ↦ 0.5 * x + 0.5 * A
  let P : ℝ × ℝ := (0, 2)
  let Q : ℝ × ℝ := (12, 8)
  (∀ x : ℝ, line x = A ↔ x = A) ∧ 
  (line ((P.1 + Q.1) / 2) = (P.2 + Q.2) / 2) →
  A = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equidistant_point_l1040_104062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_plus_intercept_eq_three_l1040_104046

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The slope of a line -/
noncomputable def Line.slope (L : Line) : ℝ := (L.y₂ - L.y₁) / (L.x₂ - L.x₁)

/-- The y-intercept of a line -/
noncomputable def Line.yIntercept (L : Line) : ℝ := L.y₁ - L.slope * L.x₁

/-- Theorem: For a line passing through (1, 3) and (3, 7), 
    the sum of its slope and y-intercept is 3 -/
theorem slope_plus_intercept_eq_three :
  let L : Line := { x₁ := 1, y₁ := 3, x₂ := 3, y₂ := 7 }
  L.slope + L.yIntercept = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_plus_intercept_eq_three_l1040_104046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_f_g_increasing_not_symmetry_axis_h_l1040_104076

open Real

-- Define the functions
noncomputable def f (x : ℝ) := |sin (2 * x + π / 3)|
noncomputable def g (x : ℝ) := sin (x - 3 * π / 2)
noncomputable def h (x : ℝ) := sin (2 * x + 5 * π / 6)

-- Statement 1
theorem min_period_f : ∃ (T : ℝ), T > 0 ∧ T = π / 2 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
  ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T' := by
  sorry

-- Statement 2
theorem g_increasing : ∀ (x y : ℝ), x ∈ Set.Icc π (3 * π / 2) → y ∈ Set.Icc π (3 * π / 2) →
  x < y → g x < g y := by
  sorry

-- Statement 3
theorem not_symmetry_axis_h : ¬(∀ (x : ℝ), h (5 * π / 4 + x) = h (5 * π / 4 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_f_g_increasing_not_symmetry_axis_h_l1040_104076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_distance_relation_l1040_104048

/-- The line equation: 4x - 3y + m = 0 where m < 0 -/
def line_equation (x y m : ℝ) : Prop := 4 * x - 3 * y + m = 0 ∧ m < 0

/-- The circle equation: x^2 + y^2 + 2x - 2y - 6 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 6 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 1)

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := 2 * Real.sqrt 2

/-- The length of the chord cut by the circle on the line -/
def chord_length : ℝ := 4

/-- The distance from a point (x, y) to the line 4x - 3y + m = 0 -/
noncomputable def distance_to_line (x y m : ℝ) : ℝ := 
  |4 * x - 3 * y + m| / Real.sqrt (4^2 + 3^2)

/-- The main theorem -/
theorem chord_distance_relation (m : ℝ) : 
  (∀ x y, line_equation x y m → circle_equation x y) →
  chord_length = 2 * distance_to_line (circle_center.1) (circle_center.2) m →
  m = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_distance_relation_l1040_104048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_1_series_sum_2_series_sum_3_l1040_104070

-- Define the series as a function of the starting index
noncomputable def series (start : ℕ) : ℝ := ∑' n, 1 / (n * (n + 1))

-- Theorem for part (a)
theorem series_sum_1 : series 1 = 1 := by sorry

-- Theorem for part (b)
theorem series_sum_2 : series 2 = 1/2 := by sorry

-- Theorem for part (c)
theorem series_sum_3 : series 3 = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_1_series_sum_2_series_sum_3_l1040_104070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_larger_angle_in_special_isosceles_triangle_l1040_104085

/-- An isosceles triangle with an inscribed circle. -/
structure IsoscelesTriangleWithInscribedCircle where
  /-- The base of the triangle -/
  base : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The base is twice the diameter of the inscribed circle -/
  base_eq_2diameter : base = 4 * radius
  /-- The triangle is isosceles -/
  isosceles : True

/-- The sine of the larger angle in an isosceles triangle with an inscribed circle
    where the base is twice the diameter of the inscribed circle is 24/25. -/
theorem sine_of_larger_angle_in_special_isosceles_triangle
  (t : IsoscelesTriangleWithInscribedCircle) :
  ∃ θ : ℝ, θ = Real.arcsin (24/25) ∧ 
    θ = max (Real.arcsin (4/5)) (π - 2 * Real.arcsin (4/5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_larger_angle_in_special_isosceles_triangle_l1040_104085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1040_104036

/-- The problem of finding the closest point on a line to a given point in 3D space. -/
theorem closest_point_on_line (t : ℝ) : 
  let v : Fin 3 → ℝ := ![1 + 8*t, -2 + 4*t, -4 - 2*t]
  let a : Fin 3 → ℝ := ![3, 3, 2]
  let dir : Fin 3 → ℝ := ![8, 4, -2]
  (v 0 - a 0) * dir 0 + (v 1 - a 1) * dir 1 + (v 2 - a 2) * dir 2 = 0 ↔ t = 2/7 := by
  sorry

#check closest_point_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1040_104036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_closure_property_l1040_104002

theorem subset_with_closure_property (p : ℕ) (hp : Nat.Prime p) :
  ∃! (A : Finset ℕ), A ⊂ Finset.range p ∧
    (∀ a b, a ∈ A → b ∈ A → ((a * b + 1) % p) ∈ A) ∧
    ((p > 2 → A.card = 1) ∧ (p = 2 → A.card = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_closure_property_l1040_104002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_even_periodic_function_l1040_104068

noncomputable def f₁ (x : ℝ) := Real.sin (4 * x)
noncomputable def f₂ (x : ℝ) := Real.cos (2 * x)
noncomputable def f₃ (x : ℝ) := Real.tan (2 * x)
noncomputable def f₄ (x : ℝ) := Real.sin (Real.pi / 2 - 4 * x)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem unique_even_periodic_function :
  (is_even f₄ ∧ has_period f₄ (Real.pi / 2)) ∧
  ¬(is_even f₁ ∧ has_period f₁ (Real.pi / 2)) ∧
  ¬(is_even f₂ ∧ has_period f₂ (Real.pi / 2)) ∧
  ¬(is_even f₃ ∧ has_period f₃ (Real.pi / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_even_periodic_function_l1040_104068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_fill_time_l1040_104010

/-- Represents the time it takes to fill a pool with the faster hose -/
def fast_hose_time : ℝ → ℝ := λ t => t

/-- Represents the time it takes to fill a pool with the slower hose -/
def slow_hose_time : ℝ → ℝ := λ t => 5 * t

/-- The number of pools each person fills -/
def pools_per_person : ℕ := 5

/-- The time difference between when Petya and Vasya finish -/
def time_difference : ℝ := 1

theorem vasya_fill_time (t : ℝ) :
  (pools_per_person * slow_hose_time t - pools_per_person * fast_hose_time t = time_difference) →
  pools_per_person * slow_hose_time t = 75 / 60 :=
by
  intro h
  sorry

#check vasya_fill_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_fill_time_l1040_104010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_Y_greater_X_l1040_104054

/-- A random variable uniformly distributed on an interval -/
structure UniformRV (a b : ℝ) where
  value : ℝ
  in_range : a ≤ value ∧ value ≤ b

/-- The probability that one uniform random variable is greater than another -/
noncomputable def prob_greater (X : UniformRV 0 3000) (Y : UniformRV 0 6000) : ℝ :=
  3 / 4

/-- Theorem stating that the probability of Y > X is 3/4 -/
theorem prob_Y_greater_X (X : UniformRV 0 3000) (Y : UniformRV 0 6000) :
  prob_greater X Y = 3 / 4 := by
  sorry

#check prob_Y_greater_X

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_Y_greater_X_l1040_104054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1040_104017

/-- The time taken for two workers to complete a job together, given their individual completion times -/
noncomputable def time_taken_together (time_A : ℝ) (time_B : ℝ) : ℝ :=
  1 / (1 / time_A + 1 / time_B)

/-- Theorem stating that if A can complete a work in 12 days and B in 18 days, 
    then together they will complete the work in 7.2 days -/
theorem work_completion_time :
  time_taken_together 12 18 = 7.2 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_taken_together 12 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1040_104017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₂_div_z₁_in_fourth_quadrant_l1040_104011

-- Define the complex numbers
def z₁ : ℂ := Complex.mk 1 1
def z₂ : ℂ := Complex.mk 3 (-2)

-- Define the fourth quadrant
def fourth_quadrant (z : ℂ) : Prop := 0 < z.re ∧ z.im < 0

-- Theorem statement
theorem z₂_div_z₁_in_fourth_quadrant : fourth_quadrant (z₂ / z₁) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₂_div_z₁_in_fourth_quadrant_l1040_104011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_f_l1040_104072

noncomputable def f (x : ℝ) : ℝ := (2*x + 1) / (3*x - 4)

theorem domain_and_range_of_f :
  (∀ x : ℝ, x ≠ 4/3 → ∃ y : ℝ, f x = y) ∧
  (∀ y : ℝ, y ≠ 2/3 → ∃ x : ℝ, x ≠ 4/3 ∧ f x = y) :=
by
  sorry

#check domain_and_range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_f_l1040_104072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1040_104030

noncomputable def f (x : ℝ) := 2 * Real.sin x + Real.sin (2 * x)

theorem f_minimum_value :
  ∃ (x : ℝ), f x = -3 * Real.sqrt 3 / 2 ∧
  ∀ (y : ℝ), f y ≥ -3 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1040_104030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_lifting_capacity_theorem_l1040_104043

/-- Calculates Tom's total lifting capacity for farmer handles and deadlift --/
noncomputable def total_lifting_capacity (initial_farmer_handles : ℝ) 
                           (training_increase : ℝ) 
                           (specialization_increase : ℝ) 
                           (grip_technique_increase : ℝ)
                           (initial_deadlift_pounds : ℝ)
                           (sumo_style_increase : ℝ)
                           (kg_to_pounds_ratio : ℝ) : ℝ :=
  let farmer_handles_capacity := 
    initial_farmer_handles * (1 + training_increase) * (1 + specialization_increase) * (1 + grip_technique_increase) * 2
  let deadlift_capacity := 
    initial_deadlift_pounds * (1 + sumo_style_increase) / kg_to_pounds_ratio
  farmer_handles_capacity + deadlift_capacity

/-- Tom's total lifting capacity is approximately 905.22 kg --/
theorem total_lifting_capacity_theorem : 
  ∃ ε > 0, |total_lifting_capacity 100 1.5 0.25 0.1 400 0.2 2.20462 - 905.22| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_lifting_capacity_theorem_l1040_104043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solution_l1040_104069

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C (in radians),
    prove that if a = 2, b = 2√3, and B = 2π/3, then A = π/6, C = π/6, and c = 2. -/
theorem triangle_solution (a b c A B C : ℝ) : 
  a = 2 → 
  b = 2 * Real.sqrt 3 → 
  B = 2 * π / 3 → 
  (A + B + C = π) →
  (Real.sin A / a = Real.sin B / b) →
  (Real.sin B / b = Real.sin C / c) →
  (A = π / 6 ∧ C = π / 6 ∧ c = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solution_l1040_104069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_pc_l1040_104095

/-- Triangle ABC with point P inside --/
structure TriangleABCP where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ

/-- Definition of the specific triangle in the problem --/
def SpecialTriangle (t : TriangleABCP) : Prop :=
  -- Triangle ABC has a right angle at B
  (t.A.1 - t.B.1) * (t.C.1 - t.B.1) + (t.A.2 - t.B.2) * (t.C.2 - t.B.2) = 0 ∧
  -- PA = 14
  Real.sqrt ((t.P.1 - t.A.1)^2 + (t.P.2 - t.A.2)^2) = 14 ∧
  -- PB = 8
  Real.sqrt ((t.P.1 - t.B.1)^2 + (t.P.2 - t.B.2)^2) = 8 ∧
  -- ∠APB = 90°
  (t.A.1 - t.P.1) * (t.B.1 - t.P.1) + (t.A.2 - t.P.2) * (t.B.2 - t.P.2) = 0 ∧
  -- ∠BPC = ∠CPA = 135°
  ((t.B.1 - t.P.1) * (t.C.1 - t.P.1) + (t.B.2 - t.P.2) * (t.C.2 - t.P.2)) /
    (Real.sqrt ((t.B.1 - t.P.1)^2 + (t.B.2 - t.P.2)^2) *
     Real.sqrt ((t.C.1 - t.P.1)^2 + (t.C.2 - t.P.2)^2)) = -Real.sqrt 2 / 2 ∧
  ((t.C.1 - t.P.1) * (t.A.1 - t.P.1) + (t.C.2 - t.P.2) * (t.A.2 - t.P.2)) /
    (Real.sqrt ((t.C.1 - t.P.1)^2 + (t.C.2 - t.P.2)^2) *
     Real.sqrt ((t.A.1 - t.P.1)^2 + (t.A.2 - t.P.2)^2)) = -Real.sqrt 2 / 2

/-- The main theorem to prove --/
theorem special_triangle_pc (t : TriangleABCP) (h : SpecialTriangle t) :
  Real.sqrt ((t.P.1 - t.C.1)^2 + (t.P.2 - t.C.2)^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_pc_l1040_104095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_l1040_104037

noncomputable def f (x : ℝ) := 2 * Real.sin (4 * x - Real.pi / 3) + Real.sqrt 3

theorem zeros_of_f :
  ∃ (S : Finset ℝ), S.card = 5 ∧
  (∀ x ∈ S, -Real.pi/2 ≤ x ∧ x ≤ Real.pi/2 ∧ f x = 0) ∧
  (∀ x, -Real.pi/2 ≤ x ∧ x ≤ Real.pi/2 ∧ f x = 0 → x ∈ S) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_l1040_104037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_line_l1040_104051

open Real

-- Define the function f
noncomputable def f (x b a : ℝ) : ℝ := Real.log x + x^2 - b*x + a

-- State the theorem
theorem min_slope_tangent_line (b a : ℝ) (hb : b > 0) :
  ∃ (m : ℝ), m = 2 ∧ ∀ (x : ℝ), x > 0 → ((Real.log x + 2*x - b) : ℝ) ≥ m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_line_l1040_104051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_amount_proof_l1040_104073

/-- Represents a loan with simple interest -/
structure Loan where
  principal : ℚ
  rate : ℚ
  time : ℚ
  interest : ℚ

/-- Calculates the simple interest for a loan -/
def simpleInterest (l : Loan) : ℚ := (l.principal * l.rate * l.time) / 100

theorem loan_amount_proof (l : Loan) 
  (h1 : l.time = l.rate)
  (h2 : l.interest = 729)
  (h3 : l.rate = 9)
  : l.principal = 900 := by
  sorry

#check loan_amount_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_amount_proof_l1040_104073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_lines_l1040_104092

/-- Predicate to check if a real number is the area of a triangle formed by three lines -/
def IsTriangleArea (A : ℝ) (l₁ l₂ l₃ : ℝ → ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (l₁ x₁ = y₁ ∧ l₂ x₁ = y₁) ∧
    (l₂ x₂ = y₂ ∧ l₃ x₂ = y₂) ∧
    (l₃ x₃ = y₃ ∧ l₁ x₃ = y₃) ∧
    A = 1/2 * abs ((x₁ - x₃) * (y₂ - y₃) - (x₂ - x₃) * (y₁ - y₃))

/-- The area of the triangle formed by the intersection of three lines -/
theorem triangle_area_from_lines (l₁ l₂ l₃ : ℝ → ℝ) : 
  (∀ x, l₁ x = 3/4 * x + 1) → 
  (∀ x, l₂ x = -1/2 * x + 5) → 
  (∀ x, l₃ x = 2) → 
  ∃ A : ℝ, A = 49/15 ∧ IsTriangleArea A l₁ l₂ l₃ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_lines_l1040_104092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1040_104018

open Real

theorem f_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x ∈ Set.Ioo 0 π, deriv f x * sin x > f x * cos x) : 
  f (π/4) > Real.sqrt 2 * f (π/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1040_104018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_theorem_l1040_104024

/-- A cube with side length 10 -/
structure Cube :=
  (side_length : ℝ)
  (side_length_eq : side_length = 10)

/-- A point on the face of the cube -/
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

/-- The length of the light path -/
noncomputable def light_path_length (c : Cube) (p : Point) : ℝ :=
  2 * Real.sqrt ((c.side_length - 0)^2 + p.y^2 + p.z^2)

/-- The theorem to be proved -/
theorem light_path_theorem (c : Cube) (p : Point) :
  c.side_length = 10 ∧ p.x = 10 ∧ p.y = 3 ∧ p.z = 4 →
  light_path_length c p = 20 * Real.sqrt 125 := by
  sorry

#check light_path_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_theorem_l1040_104024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_approx_l1040_104009

theorem power_product_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |((5 : ℝ)^(5/4) * (12 : ℝ)^(1/4) * (60 : ℝ)^(3/4)) - 476.736| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_approx_l1040_104009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monge_point_implies_altitude_foot_on_circumcircle_l1040_104080

-- Define the tetrahedron and its points
variable (A B C D T O G H : EuclideanSpace ℝ (Fin 3))

-- Define the Monge point, circumcenter, and centroid
def is_monge_point (T : EuclideanSpace ℝ (Fin 3)) (ABCD : Set (EuclideanSpace ℝ (Fin 3))) : Prop := sorry
def is_circumcenter (O : EuclideanSpace ℝ (Fin 3)) (ABCD : Set (EuclideanSpace ℝ (Fin 3))) : Prop := sorry
def is_centroid (G : EuclideanSpace ℝ (Fin 3)) (ABCD : Set (EuclideanSpace ℝ (Fin 3))) : Prop := sorry

-- Define the symmetry condition
def symmetric_wrt (P Q C : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Define the foot of the altitude
def is_altitude_foot (H : EuclideanSpace ℝ (Fin 3)) (D : EuclideanSpace ℝ (Fin 3)) (ABC : Set (EuclideanSpace ℝ (Fin 3))) : Prop := sorry

-- Define lying on a plane
def lies_on_plane (P : EuclideanSpace ℝ (Fin 3)) (ABC : Set (EuclideanSpace ℝ (Fin 3))) : Prop := sorry

-- Define lying on a circle
def lies_on_circumcircle (P : EuclideanSpace ℝ (Fin 3)) (ABC : Set (EuclideanSpace ℝ (Fin 3))) : Prop := sorry

-- State the theorem
theorem monge_point_implies_altitude_foot_on_circumcircle 
  (ABCD : Set (EuclideanSpace ℝ (Fin 3))) 
  (hT : is_monge_point T ABCD) 
  (hO : is_circumcenter O ABCD) 
  (hG : is_centroid G ABCD) 
  (hSym : symmetric_wrt T O G) 
  (hTPlane : lies_on_plane T {A, B, C}) 
  (hH : is_altitude_foot H D {A, B, C}) : 
  lies_on_circumcircle H {A, B, C} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monge_point_implies_altitude_foot_on_circumcircle_l1040_104080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_rise_ratio_l1040_104053

theorem water_level_rise_ratio 
  (r₁ r₂ : ℝ) 
  (h₁ h₂ : ℝ) 
  (cube_side : ℝ) 
  (h : r₁ = 4 ∧ r₂ = 9 ∧ cube_side = 2)
  (volume_eq : (1/3) * Real.pi * r₁^2 * h₁ = (1/3) * Real.pi * r₂^2 * h₂) :
  (cube_side^3 / (Real.pi * r₁^2)) / (cube_side^3 / (Real.pi * r₂^2)) = 81 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_rise_ratio_l1040_104053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1040_104000

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ m + 1}

-- Part 1
theorem part_one : A ∩ (Set.univ \ B 3) = {x | 0 ≤ x ∧ x < 2} := by sorry

-- Part 2
theorem part_two : 
  (∀ m : ℝ, (∀ x : ℝ, x ∈ B m → x ∈ A) ∧ (∃ x : ℝ, x ∈ A ∧ x ∉ B m)) ↔ 
  (1 ≤ m ∧ m ≤ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1040_104000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_rectangle_area_l1040_104082

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 16 * x - 8 * y + 40

-- Define the rectangle properties
structure Rectangle where
  width : ℝ
  length : ℝ
  parallel_to_x_axis : Prop
  length_twice_width : length = 2 * width

-- Define the inscribed property
def inscribed (c : (ℝ → ℝ → Prop)) (r : Rectangle) : Prop :=
  ∃ (x y : ℝ), c x y ∧ 
    x ≥ 0 ∧ x ≤ r.length ∧ 
    y ≥ 0 ∧ y ≤ r.width

-- Theorem statement
theorem inscribed_circle_rectangle_area :
  ∀ (r : Rectangle),
    inscribed circle_equation r →
    r.parallel_to_x_axis →
    r.width * r.length = 96 := by
  intro r inscribed_prop _
  have h1 : r.length = 2 * r.width := r.length_twice_width
  sorry -- Proof details would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_rectangle_area_l1040_104082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_bathtub_time_l1040_104044

/-- Proves that given the conditions, Jack left the water running for approximately 9 hours -/
theorem jack_bathtub_time (drip_rate : ℝ) (evap_rate : ℝ) (dumped : ℝ) (left : ℝ) : 
  drip_rate = 40 →
  evap_rate = 200 / 60 →
  dumped = 12000 →
  left = 7800 →
  ∃ (time : ℝ), abs (time - 9) < 0.01 ∧ time * 60 * (drip_rate - evap_rate) = dumped + left :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_bathtub_time_l1040_104044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_hatching_experiment_l1040_104012

/-- Hatching experiment data and calculations -/
theorem fish_hatching_experiment 
  (total_eggs : ℕ) 
  (hatched_fry : ℕ) 
  (new_total_eggs : ℕ) 
  (h1 : total_eggs = 10000) 
  (h2 : hatched_fry = 8513) 
  (h3 : new_total_eggs = 30000) : 
  (hatched_fry / total_eggs : ℚ) = 8513 / 10000 ∧ 
  (hatched_fry * new_total_eggs) / total_eggs = 25539 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_hatching_experiment_l1040_104012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_is_correct_l1040_104027

/-- The distance from a point to a plane defined by three points -/
noncomputable def distance_point_to_plane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₀, y₀, z₀) := M₀
  let (x₁, y₁, z₁) := M₁
  let (x₂, y₂, z₂) := M₂
  let (x₃, y₃, z₃) := M₃
  let A := (y₂ - y₁) * (z₃ - z₁) - (z₂ - z₁) * (y₃ - y₁)
  let B := (z₂ - z₁) * (x₃ - x₁) - (x₂ - x₁) * (z₃ - z₁)
  let C := (x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁)
  let D := -(A * x₁ + B * y₁ + C * z₁)
  abs (A * x₀ + B * y₀ + C * z₀ + D) / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_to_plane_is_correct :
  let M₀ : ℝ × ℝ × ℝ := (2, 3, 8)
  let M₁ : ℝ × ℝ × ℝ := (1, 1, 2)
  let M₂ : ℝ × ℝ × ℝ := (-1, 1, 3)
  let M₃ : ℝ × ℝ × ℝ := (2, -2, 4)
  distance_point_to_plane M₀ M₁ M₂ M₃ = 7 * Real.sqrt (7/10) := by
  sorry

#check distance_to_plane_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_is_correct_l1040_104027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_sum_l1040_104089

/-- Triangle with given properties --/
structure SpecialTriangle where
  P : ℝ
  Q : ℝ
  R : ℝ
  inradius : ℝ
  circumradius : ℝ
  cosine_relation : 2 * Real.cos Q = Real.cos P + Real.cos R
  inradius_value : inradius = 6
  circumradius_value : circumradius = 15

/-- The theorem statement --/
theorem special_triangle_sum (t : SpecialTriangle) 
  (p q r : ℕ) 
  (area_form : t.inradius * (t.P + t.Q + t.R) / 2 = p * Real.sqrt q / r) 
  (p_r_coprime : Nat.Coprime p r) 
  (q_square_free : ∀ (prime : ℕ), Nat.Prime prime → ¬(prime^2 ∣ q)) : 
  p + q + r = 260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_sum_l1040_104089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_count_l1040_104075

noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x^2 + 8*x + 15)

theorem vertical_asymptotes_count :
  ∃ (a b : ℝ), a ≠ b ∧
  (∀ (x : ℝ), x ≠ a ∧ x ≠ b → f x ≠ 0) ∧
  (∀ (M : ℝ), ∃ (δ : ℝ), δ > 0 ∧
    (∀ (x : ℝ), 0 < |x - a| ∧ |x - a| < δ → |f x| > M) ∧
    (∀ (x : ℝ), 0 < |x - b| ∧ |x - b| < δ → |f x| > M)) ∧
  (∀ (c : ℝ), c ≠ a ∧ c ≠ b →
    ∃ (L : ℝ), ∀ (ε : ℝ), ε > 0 →
      ∃ (δ : ℝ), δ > 0 ∧
        ∀ (x : ℝ), 0 < |x - c| ∧ |x - c| < δ → |f x - L| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_count_l1040_104075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_pow_1440_is_identity_l1040_104066

open Real

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![cos (π/6), 0, -sin (π/6)],
    ![0, 1, 0],
    ![sin (π/6), 0, cos (π/6)]]

theorem B_pow_1440_is_identity : 
  B^1440 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_pow_1440_is_identity_l1040_104066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flowchart_structure_l1040_104087

/-- Represents a flowchart --/
structure Flowchart where
  nodes : Set Nat
  start : Nat
  ends : Set Nat

/-- A flowchart is valid if it satisfies the standard structure --/
def is_valid_flowchart (f : Flowchart) : Prop :=
  f.start ∈ f.nodes ∧ f.ends ⊆ f.nodes ∧ f.ends.Nonempty

/-- Theorem stating the standard structure of a flowchart --/
theorem flowchart_structure (f : Flowchart) (h : is_valid_flowchart f) :
  (∃! start, start = f.start) ∧ (∃ end_point, end_point ∈ f.ends) := by
  sorry

#check flowchart_structure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flowchart_structure_l1040_104087
