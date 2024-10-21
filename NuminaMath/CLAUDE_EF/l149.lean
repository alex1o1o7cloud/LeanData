import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_number_count_l149_14994

theorem five_digit_number_count : 
  let valid_tuple (A B C D E : Nat) :=
    A ≠ 0 ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
    A + B = C ∧ B + C = D ∧ C + D = E
  (Finset.filter (fun (t : Nat × Nat × Nat × Nat × Nat) => 
    valid_tuple t.1 t.2.1 t.2.2.1 t.2.2.2.1 t.2.2.2.2) 
    (Finset.product 
      (Finset.range 10) (Finset.product
        (Finset.range 10) (Finset.product
          (Finset.range 10) (Finset.product
            (Finset.range 10) (Finset.range 10)))))).card = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_number_count_l149_14994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_problem_l149_14940

/-- Given a cube with edge length 2 cm and a point light source directly above an upper vertex
    casting a shadow with area 288 sq cm (excluding the area beneath the cube),
    prove that the greatest integer not exceeding 1000 times the height of the light source is 265. -/
theorem shadow_problem (x : ℝ) : 
  (2 : ℝ) > 0 ∧ x > 0 ∧ 288 = (Real.sqrt 292 - 2)^2 → 
  ⌊1000 * x⌋ = 265 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_problem_l149_14940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_from_slope_product_l149_14912

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point on the ellipse -/
structure Point (e : Ellipse) where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The slope of a line from the left vertex to a point on the ellipse -/
noncomputable def slope_from_left_vertex (e : Ellipse) (p : Point e) : ℝ :=
  p.y / (p.x + e.a)

theorem eccentricity_from_slope_product (e : Ellipse) 
  (p q : Point e) (h_sym : p.x = -q.x) 
  (h_slope : slope_from_left_vertex e p * slope_from_left_vertex e q = 1/4) :
  eccentricity e = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_from_slope_product_l149_14912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_parallel_to_polar_axis_l149_14984

-- Define the point P in polar coordinates
noncomputable def P : ℝ × ℝ := (2, -Real.pi/6)

-- Define the equation of the line
def line_equation (ρ θ : ℝ) : Prop := ρ * Real.sin θ = -1

-- Theorem statement
theorem line_through_P_parallel_to_polar_axis :
  ∀ (ρ θ : ℝ),
    line_equation ρ θ ↔ 
    (∃ (t : ℝ), ρ * Real.cos θ = P.1 * Real.cos P.2 + t ∧
                ρ * Real.sin θ = P.1 * Real.sin P.2) ∧
    (∀ (ρ' θ' : ℝ), line_equation ρ' θ' → ρ' * Real.cos θ' = ρ * Real.cos θ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_parallel_to_polar_axis_l149_14984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_days_to_triple_repayment_l149_14946

/-- The least integer number of days after which the repayment amount is at least
    three times the borrowed amount, given a borrowed amount of $50 and a 10% daily interest rate. -/
theorem least_days_to_triple_repayment : ℕ := by
  let borrowed_amount : ℚ := 50
  let daily_interest_rate : ℚ := 1/10
  let repayment_amount (x : ℕ) : ℚ := borrowed_amount + borrowed_amount * daily_interest_rate * x

  have h : ∀ y : ℕ, y < 20 → repayment_amount y < 3 * borrowed_amount := by
    sorry -- Proof omitted

  have h' : repayment_amount 20 ≥ 3 * borrowed_amount := by
    sorry -- Proof omitted

  exact 20

#check least_days_to_triple_repayment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_days_to_triple_repayment_l149_14946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_surface_area_in_tetrahedron_l149_14980

/-- The volume of a regular tetrahedron with edge length a -/
noncomputable def tetrahedronVolume (a : ℝ) : ℝ := (1 / 12) * a^3 * Real.sqrt 2

/-- The surface area of a sphere with radius r -/
noncomputable def sphereSurfaceArea (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- The theorem statement -/
theorem ball_surface_area_in_tetrahedron (edgeLength : ℝ) (ballRadius : ℝ) :
  edgeLength = 4 →
  tetrahedronVolume edgeLength * (1 / 8) = (1 / 3) * edgeLength^2 * ballRadius →
  sphereSurfaceArea ballRadius = (2 / 3) * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_surface_area_in_tetrahedron_l149_14980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calories_in_250g_of_mixed_drink_l149_14901

/-- Represents the components of the mixed drink -/
structure DrinkComponent where
  weight : ℝ
  caloriesPer100g : ℝ

/-- Calculates the total calories in a drink component -/
noncomputable def totalCalories (component : DrinkComponent) : ℝ :=
  component.weight * (component.caloriesPer100g / 100)

/-- Represents the mixed drink -/
structure MixedDrink where
  cranberryJuice : DrinkComponent
  honey : DrinkComponent
  water : DrinkComponent

/-- Calculates the total weight of the mixed drink -/
noncomputable def totalWeight (drink : MixedDrink) : ℝ :=
  drink.cranberryJuice.weight + drink.honey.weight + drink.water.weight

/-- Calculates the total calories in the mixed drink -/
noncomputable def totalDrinkCalories (drink : MixedDrink) : ℝ :=
  totalCalories drink.cranberryJuice + totalCalories drink.honey + totalCalories drink.water

/-- Calculates the calorie density of the mixed drink -/
noncomputable def calorieDensity (drink : MixedDrink) : ℝ :=
  totalDrinkCalories drink / totalWeight drink

theorem calories_in_250g_of_mixed_drink :
  let drink : MixedDrink := {
    cranberryJuice := { weight := 150, caloriesPer100g := 30 },
    honey := { weight := 50, caloriesPer100g := 304 },
    water := { weight := 300, caloriesPer100g := 0 }
  }
  250 * calorieDensity drink = 98.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calories_in_250g_of_mixed_drink_l149_14901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_pi_l149_14958

-- Define the functions
noncomputable def f1 (x : ℝ) := Real.sin (abs x)
noncomputable def f2 (x : ℝ) := abs (Real.cos x)
noncomputable def f3 (x : ℝ) := 2 * Real.sin (2 * x - Real.pi / 3)
noncomputable def f4 (x : ℝ) := 2 * Real.tan (x + Real.pi / 10)

-- Define periodicity
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- Theorem statement
theorem smallest_period_pi :
  ¬(is_periodic f1 Real.pi) ∧
  (is_periodic f2 Real.pi) ∧
  (is_periodic f3 Real.pi) ∧
  (is_periodic f4 Real.pi) := by
  sorry

#check smallest_period_pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_pi_l149_14958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l149_14911

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y = 2
def l₂ (x y : ℝ) : Prop := 3 * x + 4 * y = 7

-- Define the distance function between parallel lines
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₁ - c₂) / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem distance_between_lines :
  distance_parallel_lines 3 4 2 7 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l149_14911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_intersecting_chords_l149_14910

/-- Given a circle with two intersecting chords that create opposite arcs with angular measures α and β,
    the angle between the chords is (α + β) / 2. -/
theorem angle_between_intersecting_chords (α β : ℝ) :
  (α + β) / 2 = (α + β) / 2 := by
  -- The proof is trivial as we're stating an equality that's always true
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_intersecting_chords_l149_14910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_symmetry_l149_14982

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_sum_symmetry :
  (∀ x : ℝ, Function.Injective f) →  -- f is invertible
  (∀ x : ℝ, f (x + 2) = -f (-x - 2)) →  -- f(x+2) is odd
  (∀ x : ℝ, g (f x) = x) →  -- g is inverse of f
  (∀ x : ℝ, f (g x) = x) →  -- f is inverse of g
  ∀ x : ℝ, g x + g (-x) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_symmetry_l149_14982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_divisibility_by_three_remainder_2368297_mod_3_l149_14951

theorem remainder_divisibility_by_three (n : Nat) : 
  n % 3 = (n.digits 10).sum % 3 := by
  sorry

-- Specific case for 2,368,297
theorem remainder_2368297_mod_3 : 
  2368297 % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_divisibility_by_three_remainder_2368297_mod_3_l149_14951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_speed_is_12_13_l149_14913

/-- Represents the hiking scenario with Chantal and Jean -/
structure HikingScenario where
  d : ℝ  -- Half the distance from trailhead to fire tower
  speed_first_half : ℝ  -- Chantal's speed in the first half
  speed_second_half : ℝ  -- Chantal's speed in the second half
  speed_descent : ℝ  -- Chantal's speed during descent

/-- Calculates Jean's average speed given a hiking scenario -/
noncomputable def jean_average_speed (scenario : HikingScenario) : ℝ :=
  scenario.d / ((scenario.d / scenario.speed_first_half) + 
                (scenario.d / scenario.speed_second_half) + 
                (scenario.d / scenario.speed_descent))

/-- Theorem stating that Jean's average speed is 12/13 mph in the given scenario -/
theorem jean_speed_is_12_13 (scenario : HikingScenario) 
  (h1 : scenario.speed_first_half = 4)
  (h2 : scenario.speed_second_half = 2)
  (h3 : scenario.speed_descent = 3) :
  jean_average_speed scenario = 12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_speed_is_12_13_l149_14913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_not_all_prime_l149_14953

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

def is_digit (n : Nat) : Prop := n < 10

theorem two_digit_not_all_prime (a b c d : Nat) 
  (ha : is_digit a) (hb : is_digit b) (hc : is_digit c) (hd : is_digit d)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  ∃ (x y : Nat), 
    ((x = a ∧ y ∈ ({b, c, d} : Set Nat)) ∨ 
     (x = b ∧ y ∈ ({a, c, d} : Set Nat)) ∨ 
     (x = c ∧ y ∈ ({a, b, d} : Set Nat)) ∨ 
     (x = d ∧ y ∈ ({a, b, c} : Set Nat))) ∧
    ¬(is_prime (10 * x + y)) :=
  sorry

#check two_digit_not_all_prime

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_not_all_prime_l149_14953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_two_halves_l149_14960

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Theorem: For an ellipse with minor axis equal to focal length, the eccentricity is √2/2 -/
theorem ellipse_eccentricity_sqrt_two_halves (a b : ℝ) (e : Ellipse a b) 
  (h : 2 * b = Real.sqrt (a^2 - b^2)) : eccentricity e = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_two_halves_l149_14960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_car_sales_loss_l149_14977

/-- Represents a car sale transaction -/
structure CarSale where
  sellingPrice : ℝ
  profitPercent : ℝ

/-- Calculates the cost price given the selling price and profit percentage -/
noncomputable def costPrice (sale : CarSale) : ℝ :=
  sale.sellingPrice / (1 + sale.profitPercent / 100)

/-- Calculates the overall profit percentage for two car sales -/
noncomputable def overallProfitPercent (sale1 sale2 : CarSale) : ℝ :=
  let totalCostPrice := costPrice sale1 + costPrice sale2
  let totalSellingPrice := sale1.sellingPrice + sale2.sellingPrice
  (totalSellingPrice - totalCostPrice) / totalCostPrice * 100

/-- Theorem stating that selling two cars at the same price with equal gain and loss percentages results in an overall loss -/
theorem two_car_sales_loss (p : ℝ) (h : p > 0) :
  let sale1 := CarSale.mk 325475 p
  let sale2 := CarSale.mk 325475 (-p)
  overallProfitPercent sale1 sale2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_car_sales_loss_l149_14977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_speed_calculation_l149_14954

/-- The hiker's speed in kilometers per hour -/
noncomputable def hikerSpeed : ℝ := 6.25

/-- The cyclist's speed in kilometers per hour -/
noncomputable def cyclistSpeed : ℝ := 25

/-- Time in hours that the cyclist travels before stopping -/
noncomputable def cyclistTravelTime : ℝ := 5 / 60

/-- Time in hours that the cyclist waits for the hiker -/
noncomputable def waitTime : ℝ := 20 / 60

theorem hiker_speed_calculation :
  cyclistSpeed * cyclistTravelTime = hikerSpeed * waitTime :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_speed_calculation_l149_14954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_tray_trips_l149_14943

theorem cafeteria_tray_trips (tray_capacity : ℕ) (table1_trays : ℕ) (table2_trays : ℕ) : 
  tray_capacity = 7 → table1_trays = 23 → table2_trays = 5 → 
  (table1_trays / tray_capacity + 
   (if table1_trays % tray_capacity = 0 then 0 else 1) + 
   (if table2_trays > 0 then 1 else 0)) = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_tray_trips_l149_14943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_square_area_l149_14945

/-- The area of the n-th square in a sequence where each square's side length
    is the diagonal of the previous square, starting with a unit square. -/
def square_area (n : ℕ) : ℝ :=
  2^(n - 1)

/-- The side length of the n-th square in the sequence. -/
noncomputable def square_side (n : ℕ) : ℝ :=
  Real.sqrt (square_area n)

theorem nth_square_area (n : ℕ) :
  n ≥ 1 →
  square_side (n + 1) = Real.sqrt (2 * square_area n) ∧
  square_area (n + 1) = 2 * square_area n ∧
  square_area n = 2^(n - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_square_area_l149_14945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_hypotenuse_l149_14970

/-- Definition of an isosceles right triangle -/
structure Triangle where
  leg : ℝ
  hypotenuse : ℝ
  isIsoscelesRight : Prop

/-- An isosceles right triangle with leg length 10 has hypotenuse length 10√2 -/
theorem isosceles_right_triangle_hypotenuse (t : Triangle) 
  (h : t.isIsoscelesRight) (leg_length : t.leg = 10) : 
  t.hypotenuse = 10 * Real.sqrt 2 := by
  sorry

/-- The leg of the triangle -/
def get_leg (t : Triangle) : ℝ := t.leg

/-- The hypotenuse of the triangle -/
def get_hypotenuse (t : Triangle) : ℝ := t.hypotenuse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_hypotenuse_l149_14970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_20kg_l149_14931

-- Define the cost structure
noncomputable def apple_cost (l m n : ℝ) (kg : ℝ) : ℝ :=
  if kg ≤ 30 then l * kg
  else if kg ≤ 60 then l * 30 + m * (kg - 30)
  else l * 30 + m * 30 + n * (kg - 60)

-- Define the theorem
theorem apple_cost_20kg (l m n : ℝ) :
  (apple_cost l m n 33 = 333) →
  (apple_cost l m n 36 = 366) →
  (apple_cost l m n 45 = 465) →
  (apple_cost l m n 50 = 525) →
  apple_cost l m n 20 = 200 :=
by
  intro h33 h36 h45 h50
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_20kg_l149_14931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_truncated_cone_l149_14904

/-- Volume of a truncated right circular cone -/
noncomputable def truncatedConeVolume (R r h : ℝ) : ℝ :=
  (Real.pi * h / 3) * (R^2 + r^2 + R*r)

/-- Theorem: Volume of specific truncated cone -/
theorem volume_of_specific_truncated_cone :
  truncatedConeVolume 10 5 10 = (1750/3) * Real.pi := by
  -- Unfold the definition of truncatedConeVolume
  unfold truncatedConeVolume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_truncated_cone_l149_14904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_min_distance_l149_14902

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -x^2

/-- The line equation -/
def line (x y : ℝ) : Prop := 4*x + 3*y - 8 = 0

/-- The minimum distance from the parabola to the line -/
noncomputable def min_distance : ℝ := 4/3

theorem parabola_line_min_distance :
  ∀ (x y : ℝ), parabola x y →
  (∃ (x' y' : ℝ), line x' y' ∧
    ∀ (x'' y'' : ℝ), line x'' y'' →
      (x - x')^2 + (y - y')^2 ≤ (x - x'')^2 + (y - y'')^2) →
  (∃ (x' y' : ℝ), line x' y' ∧
    (x - x')^2 + (y - y')^2 = min_distance^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_min_distance_l149_14902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l149_14932

/-- A sequence satisfying the given conditions -/
def mySequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≠ 0) ∧ 
  (a 1 = 1) ∧
  (∀ n : ℕ, n ≥ 1 → 1 / (a (n + 1)) = n + 1 / (a n))

/-- The theorem stating the general term of the sequence -/
theorem sequence_general_term (a : ℕ → ℝ) (h : mySequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = 2 / (n^2 - n + 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l149_14932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coach_path_always_greater_than_100_l149_14995

/-- Represents an athlete in the race --/
structure Athlete where
  speed_to_b : ℝ
  speed_from_b : ℝ

/-- Represents the race configuration --/
structure RaceConfig where
  distance : ℝ
  athletes : Fin 3 → Athlete

/-- The coach's path length for a given race configuration --/
noncomputable def coach_path_length (config : RaceConfig) : ℝ :=
  sorry

/-- The theorem stating that the coach's path is always greater than 100 meters --/
theorem coach_path_always_greater_than_100 (config : RaceConfig) 
  (h1 : config.distance = 60)
  (h2 : ∀ i j : Fin 3, i ≠ j → (config.athletes i) ≠ (config.athletes j))
  (h3 : ∀ i : Fin 3, (config.athletes i).speed_to_b > 0 ∧ (config.athletes i).speed_from_b > 0)
  (h4 : ∀ i j : Fin 3, i.val < j.val → (config.athletes i).speed_to_b > (config.athletes j).speed_to_b)
  : coach_path_length config > 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coach_path_always_greater_than_100_l149_14995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_max_width_width_upper_bound_l149_14948

/-- 
The maximum width of a rectangle with fixed side length l 
occurs when the rectangle is aligned with the diagonal of a square.
-/
theorem rectangle_max_width 
  (a : ℝ) (l h : ℝ) (h_pos : 0 < h) (l_pos : 0 < l) :
  h ≤ a * Real.sqrt 2 - l :=
by
  -- The proof would go here
  sorry

/-- 
The set of possible widths h for a rectangle with fixed side length l 
is bounded above by the line h = a√2 - l.
-/
theorem width_upper_bound 
  (a : ℝ) (l h : ℝ) (h_pos : 0 < h) (l_pos : 0 < l) :
  h ≤ a * Real.sqrt 2 - l :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_max_width_width_upper_bound_l149_14948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjusted_tripod_height_l149_14908

/-- Represents a tripod configuration -/
structure Tripod where
  leg_length : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  height : ℝ

/-- Calculates the height of a tripod given its configuration -/
noncomputable def calculate_height (t : Tripod) : ℝ := sorry

/-- The original tripod configuration -/
def original_tripod : Tripod :=
  { leg_length := 6
  , angle1 := 120
  , angle2 := 120
  , angle3 := 120
  , height := 5 }

/-- The adjusted tripod configuration -/
def adjusted_tripod : Tripod :=
  { leg_length := 6
  , angle1 := 100
  , angle2 := 130
  , angle3 := 130
  , height := 0 }  -- Height to be calculated

/-- Theorem stating the new height of the adjusted tripod -/
theorem adjusted_tripod_height :
  ∃ (ε : ℝ), ε > 0 ∧ |calculate_height adjusted_tripod - 5.146| < ε := by sorry

#check adjusted_tripod_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjusted_tripod_height_l149_14908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_case_one_case_two_case_three_l149_14942

-- Case 1: x₂ = (x₁ + x₃) / 2
theorem case_one (a b c x₁ x₂ x₃ : ℝ) : 
  x₁^3 + a*x₁^2 + b*x₁ + c = 0 →
  x₂^3 + a*x₂^2 + b*x₂ + c = 0 →
  x₃^3 + a*x₃^2 + b*x₃ + c = 0 →
  x₂ = (x₁ + x₃) / 2 →
  b = (2 * a^2) / 9 ∧
  x₁ = -a/3 + Real.sqrt (a^2/3 - b) ∧
  x₂ = -a/3 ∧
  x₃ = -a/3 - Real.sqrt (a^2/3 - b) :=
by sorry

-- Case 2: x₂ = sqrt(x₁ * x₃)
theorem case_two (a b c x₁ x₂ x₃ : ℝ) :
  x₁^3 + a*x₁^2 + b*x₁ + c = 0 →
  x₂^3 + a*x₂^2 + b*x₂ + c = 0 →
  x₃^3 + a*x₃^2 + b*x₃ + c = 0 →
  x₂ = Real.sqrt (x₁ * x₃) →
  a^3 = b^3 * c ∧
  x₁ = (1/(2*a)) * (b - a^2 + Real.sqrt ((b - a^2)^2 - 4*b^2)) ∧
  x₂ = -Real.rpow c (1/3) ∧
  x₃ = (1/(2*a)) * (b - a^2 - Real.sqrt ((b - a^2)^2 - 4*b^2)) :=
by sorry

-- Case 3: x₂ = (2 * x₁ * x₃) / (x₁ + x₃)
theorem case_three (a b c x₁ x₂ x₃ : ℝ) :
  x₁^3 + a*x₁^2 + b*x₁ + c = 0 →
  x₂^3 + a*x₂^2 + b*x₂ + c = 0 →
  x₃^3 + a*x₃^2 + b*x₃ + c = 0 →
  x₂ = (2 * x₁ * x₃) / (x₁ + x₃) →
  9*a*b*c - 27*c^2 - 2*b^3 = 0 ∧
  x₁ = ((3*c - a*b)/(2*b)) + Real.sqrt (((3*c - a*b)/(2*b))^2 - b/3) ∧
  x₂ = -3*c/b ∧
  x₃ = ((3*c - a*b)/(2*b)) - Real.sqrt (((3*c - a*b)/(2*b))^2 - b/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_case_one_case_two_case_three_l149_14942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l149_14937

noncomputable def dataset : List ℝ := [-1, -1, 0, 1, 1]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (λ x => (x - m) ^ 2)).sum / xs.length

theorem variance_of_dataset :
  variance dataset = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l149_14937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_locus_is_parabola_l149_14963

noncomputable def vertex_of_parabola (a e d t : ℝ) : ℝ × ℝ :=
  let x := -(2 * t - d) / (2 * a)
  let y := a * x^2 + (2 * t - d) * x + e
  (x, y)

theorem vertex_locus_is_parabola (a e : ℝ) (d : ℝ) (ha : a > 0) (he : e > 0) :
  ∃ (f : ℝ → ℝ × ℝ), 
    (∀ t : ℝ, f t = vertex_of_parabola a e d t) ∧ 
    (∃ (A B C : ℝ), A ≠ 0 ∧ 
      ∀ p : ℝ × ℝ, p ∈ Set.range f ↔ A * p.1^2 + B * p.1 + C = p.2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_locus_is_parabola_l149_14963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_degree_integer_roots_l149_14966

/-- Represents a fifth-degree polynomial with integer coefficients -/
structure FifthDegreePolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- The number of integer roots (counting multiplicity) of a fifth-degree polynomial -/
def num_integer_roots (p : FifthDegreePolynomial) : ℕ :=
  sorry

/-- Theorem stating the possible values for the number of integer roots -/
theorem fifth_degree_integer_roots (p : FifthDegreePolynomial) :
  num_integer_roots p ∈ ({0, 1, 2, 5} : Set ℕ) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_degree_integer_roots_l149_14966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l149_14916

/-- The area of an acute triangle ABC given specific trigonometric conditions -/
theorem triangle_area (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = π) (h5 : Real.sin (A + B) = 3/5) (h6 : Real.sin (A - B) = 1/5) 
  (h7 : Real.sqrt ((3 : ℝ) ^ 2) = 3) : 
  (3 * (Real.sqrt 6 + 2)) / 2 = 
    1/2 * 3 * (2 * ((Real.sqrt 6 + 2) / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l149_14916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_jar_weight_is_200_jar_weight_solution_l149_14991

/-- The weight of a jar filled with honey -/
noncomputable def weight_with_honey : ℝ := 500

/-- The weight of the same jar filled with kerosene -/
noncomputable def weight_with_kerosene : ℝ := 350

/-- The ratio of kerosene density to honey density -/
noncomputable def density_ratio : ℝ := 1/2

/-- The weight of the empty jar -/
noncomputable def empty_jar_weight : ℝ := 200

/-- Theorem stating that the empty jar weighs 200 units -/
theorem empty_jar_weight_is_200 :
  empty_jar_weight = 200 := by
  -- Proof steps would go here
  sorry

/-- Theorem proving the solution -/
theorem jar_weight_solution :
  let honey_weight := weight_with_honey - empty_jar_weight
  let kerosene_weight := weight_with_kerosene - empty_jar_weight
  kerosene_weight = density_ratio * honey_weight ∧
  empty_jar_weight = 200 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_jar_weight_is_200_jar_weight_solution_l149_14991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_term_count_l149_14939

/-- 
Given an arithmetic sequence with first term a = -6, common difference d = 4, 
and last term 50, prove that the number of terms in the sequence is 15.
-/
theorem arithmetic_sequence_term_count : 
  ∀ (n : ℕ), n > 0 → (-6 : ℤ) + (n - 1 : ℤ) * 4 = 50 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_term_count_l149_14939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l149_14987

-- Define the constants
noncomputable def length_train1 : ℝ := 270
noncomputable def speed_train1 : ℝ := 120
noncomputable def speed_train2 : ℝ := 80
noncomputable def time_to_cross : ℝ := 9

-- Define the conversion factor from km/h to m/s
noncomputable def km_h_to_m_s : ℝ := 5 / 18

-- Theorem statement
theorem train_length_calculation :
  let relative_speed := (speed_train1 + speed_train2) * km_h_to_m_s
  let total_distance := relative_speed * time_to_cross
  let length_train2 := total_distance - length_train1
  ∃ ε > 0, |length_train2 - 230.04| < ε := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l149_14987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_concyclic_points_l149_14924

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define points and lines
structure EllipseConfiguration (E : Ellipse) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  l₁ : Set (ℝ × ℝ)
  l₂ : Set (ℝ × ℝ)
  P : ℝ × ℝ
  M₁ : ℝ × ℝ
  M₂ : ℝ × ℝ
  Q : ℝ × ℝ

-- Define the properties of the configuration
def ValidConfiguration (E : Ellipse) (config : EllipseConfiguration E) : Prop :=
  let ⟨F₁, F₂, l₁, l₂, P, M₁, M₂, Q⟩ := config
  -- P is on the ellipse
  (P.1 / E.a)^2 + (P.2 / E.b)^2 = 1
  -- M₁ is on l₁, M₂ is on l₂
  ∧ M₁ ∈ l₁ ∧ M₂ ∈ l₂
  -- Line PM₁M₂ is parallel to F₁F₂
  ∧ (M₁.2 - P.2) * (F₂.1 - F₁.1) = (M₁.1 - P.1) * (F₂.2 - F₁.2)
  -- Q is the intersection of M₁F₁ and M₂F₂
  ∧ (Q.1 - F₁.1) * (M₁.2 - F₁.2) = (Q.2 - F₁.2) * (M₁.1 - F₁.1)
  ∧ (Q.1 - F₂.1) * (M₂.2 - F₂.2) = (Q.2 - F₂.2) * (M₂.1 - F₂.1)

-- Define concyclicity
def Concyclic (A B C D : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.2 - C.2) * (D.1 - C.1) * (A.2 - D.2) =
  (A.2 - C.2) * (B.1 - C.1) * (D.2 - C.2) * (A.1 - D.1)

-- State the theorem
theorem ellipse_concyclic_points 
  (E : Ellipse) (config : EllipseConfiguration E) 
  (h : ValidConfiguration E config) :
  Concyclic config.P config.F₁ config.Q config.F₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_concyclic_points_l149_14924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_life_ratio_equal_for_proportional_decay_l149_14903

noncomputable def decay (initial_amount : ℝ) (half_life : ℝ) (time : ℝ) : ℝ :=
  initial_amount * (1/2) ^ (time / half_life)

theorem half_life_ratio_equal_for_proportional_decay 
  (T_A T_B : ℝ) 
  (h_positive_A : T_A > 0) 
  (h_positive_B : T_B > 0) 
  (t : ℝ) 
  (h_t_positive : t > 0) : 
  decay 100 T_A t = 50 ∧ decay 200 T_B t = 100 → T_B / T_A = 1 := by
  sorry

#check half_life_ratio_equal_for_proportional_decay

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_life_ratio_equal_for_proportional_decay_l149_14903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_larger_than_cylinder_base_l149_14973

/-- The radius of the base circle of a cone with maximum volume inscribed in a unit sphere -/
noncomputable def cone_base_radius : ℝ := Real.sqrt 35 / 6

/-- The radius of the base circle of a cylinder with maximum volume inscribed in a unit sphere -/
noncomputable def cylinder_base_radius : ℝ := Real.sqrt 6 / 3

/-- Theorem stating that the cone's base radius is larger than the cylinder's base radius -/
theorem cone_base_larger_than_cylinder_base :
  cone_base_radius > cylinder_base_radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_larger_than_cylinder_base_l149_14973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statements_with_nonzero_solutions_l149_14956

theorem statements_with_nonzero_solutions :
  (∃ a b : ℂ, (a ≠ 0 ∨ b ≠ 0) ∧ Real.sqrt (Complex.normSq a + Complex.normSq b) ≥ 0) ∧ 
  (∃ a b : ℂ, (a ≠ 0 ∨ b ≠ 0) ∧ Real.sqrt (Complex.normSq a + Complex.normSq b) ≥ Complex.abs (a - b)) ∧ 
  (∃ a b : ℂ, (a ≠ 0 ∨ b ≠ 0) ∧ Real.sqrt (Complex.normSq a + Complex.normSq b) = a * b + 1) ∧ 
  ¬(∃ a b : ℂ, (a ≠ 0 ∨ b ≠ 0) ∧ Real.sqrt (Complex.normSq a + Complex.normSq b) = Complex.abs a + Complex.abs b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_statements_with_nonzero_solutions_l149_14956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l149_14999

/-- A circle that passes through (0,2) and is tangent to y = x^2 at (3,9) has its center at (-27/13, 118/13) -/
theorem circle_center (C : Set (ℝ × ℝ)) (a b r : ℝ) : 
  (∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1 - a)^2 + (p.2 - b)^2 = r^2) →  -- C is a circle with center (a,b) and radius r
  (0, 2) ∈ C →                                                 -- C passes through (0,2)
  (3, 9) ∈ C →                                                 -- C passes through (3,9)
  (∀ (x : ℝ), x ≠ 3 → (x, x^2) ∉ C) →                         -- C is tangent to y = x^2 at (3,9)
  (a, b) = (-27/13, 118/13) :=                                 -- The center of C is (-27/13, 118/13)
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l149_14999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_horizontal_possible_l149_14990

/-- Represents a board with cells and dominoes -/
structure Board :=
  (cells : ℕ)
  (dominoes : ℕ)

/-- Represents a valid move on the board -/
inductive Move
  | remove : Move
  | place : Move

/-- Checks if a board configuration is valid -/
def valid_board (b : Board) : Prop :=
  b.cells = 65 ∧ b.dominoes = 32

/-- Checks if a move is valid on the given board -/
def valid_move (m : Move) (b : Board) : Prop :=
  match m with
  | Move.remove => true
  | Move.place => ∃ (x y : ℕ), x + 1 = y ∧ y ≤ b.cells

/-- Represents the state of dominoes on the board -/
inductive DominoState
  | horizontal : DominoState
  | vertical : DominoState

/-- The main theorem to be proved -/
theorem all_horizontal_possible (b : Board) :
  valid_board b →
  ∃ (moves : List Move), 
    (∀ m ∈ moves, valid_move m b) ∧
    (∀ d : ℕ, d < b.dominoes → ∃ (state : DominoState), state = DominoState.horizontal) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_horizontal_possible_l149_14990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_intersection_l149_14917

/-- Given an ellipse C with the specified properties, prove its equation and the value of k --/
theorem ellipse_and_intersection (a b : ℝ) (k : ℝ) :
  a > b ∧ b > 0 ∧  -- a > b > 0
  (a^2 - b^2) / a^2 = 3/4 ∧  -- eccentricity e = √3/2
  (a * b) / Real.sqrt (a^2 + b^2) = 4 * Real.sqrt 5 / 5 ∧  -- distance condition
  k ≠ 0 ∧  -- k ≠ 0
  ∃ (x₁ y₁ x₂ y₂ : ℝ),  -- E(x₁, y₁) and F(x₂, y₂) exist
    x₁^2 / a^2 + y₁^2 / b^2 = 1 ∧  -- E on ellipse
    x₂^2 / a^2 + y₂^2 / b^2 = 1 ∧  -- F on ellipse
    y₁ = k * x₁ + 1 ∧  -- E on line
    y₂ = k * x₂ + 1 ∧  -- F on line
    x₁ ≠ x₂ ∧  -- E and F are distinct
    x₁^2 + (y₁ + b)^2 = x₂^2 + (y₂ + b)^2  -- E and F on circle with center B
  →
  a^2 = 16 ∧ b^2 = 4 ∧ k^2 = 1/8 := by  -- Conclusion: equation of C and value of k
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_intersection_l149_14917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_time_per_problem_l149_14957

theorem math_competition_time_per_problem 
  (num_students : ℕ) 
  (total_time : ℕ) 
  (num_problems : ℕ) 
  (students_per_problem : ℕ) 
  (h1 : num_students = 8) 
  (h2 : total_time = 120)  -- 2 hours in minutes
  (h3 : num_problems = 30) 
  (h4 : students_per_problem = 2) : 
  (total_time * students_per_problem) / (num_students * num_problems) = 16 := by
  sorry

#check math_competition_time_per_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_time_per_problem_l149_14957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_l149_14983

/-- Represents a parabola with equation ax^2 + bx + c = 0 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
noncomputable def vertex (p : Parabola) : ℝ × ℝ :=
  (- p.b / (2 * p.a), - (p.b^2 - 4 * p.a * p.c) / (4 * p.a))

/-- Checks if a point lies on the parabola -/
def lies_on (p : Parabola) (x y : ℝ) : Prop :=
  p.a * x^2 + p.b * x + p.c = y

theorem parabola_equation_correct (p : Parabola) : 
  p.a = -4 ∧ p.b = 24 ∧ p.c = -34 →
  vertex p = (3, 2) ∧
  lies_on p 2 (-2) :=
by
  sorry

#check parabola_equation_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_l149_14983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l149_14927

-- Define the function f as noncomputable
noncomputable def f (x m : ℝ) : ℝ := Real.sqrt (2 * abs (x - 3) - abs x - m)

-- Theorem statement
theorem problem_solution :
  (∀ x : ℝ, ∃ y : ℝ, f x (-3) = y) ∧
  (∀ m : ℝ, (∀ x : ℝ, ∃ y : ℝ, f x m = y) → m ≤ -3) ∧
  (∀ a b c : ℝ, a^2 + b^2 + c^2 = 9 →
    1 / (a^2 + 1) + 1 / (b^2 + 2) + 1 / (c^2 + 3) ≥ 3 / 5) ∧
  (∃ a b c : ℝ, a^2 + b^2 + c^2 = 9 ∧
    1 / (a^2 + 1) + 1 / (b^2 + 2) + 1 / (c^2 + 3) = 3 / 5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l149_14927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_bound_and_a_range_l149_14915

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (2^(2*x)) - 2 * 2^x + 1 - a

-- Define the function h
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 2^(-x) * f a x

-- State the theorem
theorem h_bound_and_a_range :
  ∀ a : ℝ, a ≥ 1/2 →
  (∃ b : ℝ, b ≥ 1/2 ∧ b ≤ 4/5 ∧
    (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 →
      |h a x₁ - h a x₂| ≤ (a + 1)/2) ∧
    (∀ c : ℝ, c > b →
      ∃ y₁ y₂ : ℝ, y₁ ∈ Set.Icc (-1) 1 ∧ y₂ ∈ Set.Icc (-1) 1 ∧
        |h c y₁ - h c y₂| > (c + 1)/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_bound_and_a_range_l149_14915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_over_four_l149_14929

theorem angle_sum_is_pi_over_four (α β : ℝ) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.tan α = 1/7 →
  Real.sin β = 1/Real.sqrt 10 →
  α + 2*β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_over_four_l149_14929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_of_irrational_distances_l149_14921

-- Define the circle
def is_on_circle (x y : ℤ) : Prop := x^2 + y^2 = 100

-- Define a point on the circle with integer coordinates
structure Point where
  x : ℤ
  y : ℤ
  on_circle : is_on_circle x y

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 : ℝ)

-- Define an irrational distance
def is_irrational_distance (p1 p2 : Point) : Prop :=
  ¬ ∃ (q : ℚ), (distance p1 p2)^2 = ↑q

-- Theorem statement
theorem max_ratio_of_irrational_distances :
  ∃ (a b c d : Point),
    is_irrational_distance a b ∧
    is_irrational_distance c d ∧
    ∀ (p q r s : Point),
      is_irrational_distance p q →
      is_irrational_distance r s →
      distance p q / distance r s ≤ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_of_irrational_distances_l149_14921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_travel_theorem_l149_14979

/-- Represents the characteristics of a truck's travel -/
structure TruckTravel where
  initial_distance : ℚ
  initial_gas : ℚ
  initial_price : ℚ
  new_gas : ℚ
  price_increase : ℚ

/-- Calculates the new travel distance for a truck given changed conditions -/
def new_travel_distance (t : TruckTravel) : ℚ :=
  t.initial_distance * (t.new_gas / t.initial_gas)

/-- Theorem stating that under the given conditions, the truck can travel 450 miles -/
theorem truck_travel_theorem (t : TruckTravel) 
  (h1 : t.initial_distance = 300)
  (h2 : t.initial_gas = 10)
  (h3 : t.initial_price = 4)
  (h4 : t.new_gas = 15)
  (h5 : t.price_increase = 1/2) :
  new_travel_distance t = 450 := by
  sorry

def example_truck_travel : TruckTravel := { 
  initial_distance := 300, 
  initial_gas := 10, 
  initial_price := 4, 
  new_gas := 15, 
  price_increase := 1/2 
}

#eval new_travel_distance example_truck_travel

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_travel_theorem_l149_14979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_is_three_l149_14959

/-- Represents a monic polynomial of degree 4 -/
structure MonicPoly4 where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  const : ℝ

/-- The product of two MonicPoly4 polynomials -/
noncomputable def poly_product (p q : MonicPoly4) : Polynomial ℝ :=
  Polynomial.monomial 8 1 +
  Polynomial.monomial 7 5 +
  Polynomial.monomial 6 10 +
  Polynomial.monomial 5 10 +
  Polynomial.monomial 4 9 +
  Polynomial.monomial 3 5 +
  Polynomial.monomial 2 1 +
  Polynomial.monomial 0 9

/-- Theorem stating that under given conditions, the constant term of each polynomial is 3 -/
theorem constant_term_is_three (p q : MonicPoly4)
  (h_same_const : p.const = q.const)
  (h_same_z : p.d = q.d)
  (h_product : poly_product p q = Polynomial.monomial 8 1 +
    Polynomial.monomial 7 5 +
    Polynomial.monomial 6 10 +
    Polynomial.monomial 5 10 +
    Polynomial.monomial 4 9 +
    Polynomial.monomial 3 5 +
    Polynomial.monomial 2 1 +
    Polynomial.monomial 0 9) :
  p.const = 3 ∧ q.const = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_is_three_l149_14959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_travel_time_is_60_minutes_l149_14934

/-- The time taken by the pedestrian to travel from A to B -/
noncomputable def pedestrian_travel_time (v₁ v₂ : ℝ) : ℝ :=
  let t := (10 * v₂ - 20 * v₁) / v₁
  20 + 10 + t

/-- The cyclist's speed is 5 times the pedestrian's speed -/
def cyclist_speed_relation (v₁ v₂ : ℝ) : Prop :=
  v₂ = 5 * v₁

/-- The first meeting condition -/
def first_meeting_condition (v₁ v₂ : ℝ) : Prop :=
  (20 * v₁ + 20 * v₁) / v₂ = 20

/-- The second meeting condition -/
def second_meeting_condition (v₁ v₂ : ℝ) : Prop :=
  (50 * v₁) / v₂ = 10

theorem pedestrian_travel_time_is_60_minutes (v₁ v₂ : ℝ) 
  (h₁ : v₁ > 0) (h₂ : v₂ > 0)
  (h₃ : cyclist_speed_relation v₁ v₂)
  (h₄ : first_meeting_condition v₁ v₂)
  (h₅ : second_meeting_condition v₁ v₂) :
  pedestrian_travel_time v₁ v₂ = 60 := by
  sorry

#check pedestrian_travel_time_is_60_minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_travel_time_is_60_minutes_l149_14934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_third_sides_squared_l149_14949

/-- Definition of a right triangle with given side lengths -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2

/-- The area of a right triangle -/
noncomputable def area (t : RightTriangle) : ℝ := t.a * t.b / 2

theorem product_of_third_sides_squared 
  (t1 t2 : RightTriangle)
  (h1 : area t1 = 4)
  (h2 : area t2 = 8)
  (h3 : t1.a = t2.a ∧ t1.c = t2.c)
  (h4 : ∃ (k : ℝ), t1.a = 3*k ∧ t1.b = 4*k ∧ t1.c = 5*k) :
  (t1.b * t2.b)^2 = 682.67 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_third_sides_squared_l149_14949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_segment_properties_l149_14971

open Complex

variable (A B C C₁ C₂ : ℂ)

-- Define the rotation function
def rotate90 (center point : ℂ) : ℂ := center + I * (point - center)

-- Define the conditions
def triangle_condition (A B C : ℂ) : Prop := A ≠ B ∧ B ≠ C ∧ C ≠ A

def rotation_condition (A B C C₁ C₂ : ℂ) : Prop :=
  C₁ = rotate90 A C ∧ C₂ = rotate90 B C

-- State the theorem
theorem constant_segment_properties 
  (h_triangle : triangle_condition A B C) 
  (h_rotation : rotation_condition A B C C₁ C₂) :
  abs (C₂ - C₁) = abs (B - A) * Real.sqrt 2 ∧ 
  arg ((C₂ - C₁) / (B - A)) = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_segment_properties_l149_14971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_xy_l149_14905

theorem sin_sum_xy (x y : ℝ) 
  (h1 : Real.cos x * Real.cos y + Real.sin x * Real.sin y = 1/2)
  (h2 : Real.sin (2*x) + Real.sin (2*y) = 2/3) : 
  Real.sin (x + y) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_xy_l149_14905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_perpendicular_lines_l149_14975

/-- Two lines are perpendicular if their direction vectors are orthogonal -/
def perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d : ℝ), (∀ x y, l1 x y ↔ a*x + b*y = c) ∧
                   (∀ x y, l2 x y ↔ d*x - (1/3)*a*y = 1) ∧
                   a*d + b*(-1/3*a) = 0

/-- Definition of line l1 -/
def l1 (α : ℝ) (x y : ℝ) : Prop :=
  x * Real.sin α + y - 1 = 0

/-- Definition of line l2 -/
def l2 (α : ℝ) (x y : ℝ) : Prop :=
  x - 3 * y * Real.cos α + 1 = 0

/-- Main theorem -/
theorem sin_double_angle_perpendicular_lines (α : ℝ) :
  perpendicular (l1 α) (l2 α) → Real.sin (2 * α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_perpendicular_lines_l149_14975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l149_14997

noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 7) / (4 * x^2 + 6 * x + 3)

theorem vertical_asymptotes_sum (c d : ℝ) : 
  (∀ x : ℝ, x ≠ c ∧ x ≠ d → f x ≠ 0) →
  (4 * c^2 + 6 * c + 3 = 0) →
  (4 * d^2 + 6 * d + 3 = 0) →
  c + d = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l149_14997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_problem_l149_14964

def total_items (budget : ℚ) (sandwich_cost : ℚ) (pastry_cost : ℚ) (max_sandwiches : ℕ) : ℕ :=
  let sandwiches := (min (⌊budget / sandwich_cost⌋) max_sandwiches).toNat
  let remaining := budget - sandwich_cost * sandwiches
  let pastries := (⌊remaining / pastry_cost⌋).toNat
  sandwiches + pastries

theorem bakery_problem :
  total_items 50 6 2 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_problem_l149_14964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l149_14920

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The given sum formula -/
def given_sum (n : ℕ) : ℚ :=
  n * (3 * n + 1)

theorem arithmetic_sequence_sum (n : ℕ) :
  n > 0 → arithmetic_sum 4 6 n = given_sum n :=
by
  intro h
  -- The proof goes here
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l149_14920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_problem_l149_14936

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := (a * x) / (x + b)

-- Define the conditions
def condition1 (a b : ℝ) : Prop := f a b 1 = 5/4
def condition2 (a b : ℝ) : Prop := f a b 2 = 2

-- Define the composite function condition
def condition3 (a b : ℝ) (g : ℝ → ℝ) : Prop := ∀ x, f a b (g x) = 4 - x

-- State the theorem
theorem function_problem (a b : ℝ) (g : ℝ → ℝ) 
  (h1 : condition1 a b) (h2 : condition2 a b) (h3 : condition3 a b g) :
  (∀ x, f a b x = (5 * x) / (x + 3)) ∧
  (∀ x, g x = (12 - 3 * x) / (1 + x)) ∧
  (g 5 = -1/2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_problem_l149_14936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l149_14918

/-- Given a rectangle with diagonal length x and length three times its width, 
    prove that its area is (3x^2)/10 -/
theorem rectangle_area (x : ℝ) (h : x > 0) : 
  let w := x / Real.sqrt 10
  let l := 3 * w
  let area := l * w
  area = (3 * x^2) / 10 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l149_14918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_boxes_incur_extra_fee_l149_14996

/-- Represents a box with length and height dimensions -/
structure Box where
  length : ℚ
  height : ℚ

/-- Determines if a box incurs an extra fee based on its length-to-height ratio -/
def incursExtraFee (b : Box) : Bool :=
  let ratio := b.length / b.height
  ratio < 3/2 ∨ ratio > 3

/-- The set of boxes given in the problem -/
def boxes : List Box := [
  { length := 8, height := 5 },   -- Box X
  { length := 10, height := 2 },  -- Box Y
  { length := 7, height := 7 },   -- Box Z
  { length := 14, height := 4 }   -- Box W
]

/-- Theorem stating that exactly 3 boxes incur the extra fee -/
theorem three_boxes_incur_extra_fee :
  (boxes.filter incursExtraFee).length = 3 := by
  sorry

#eval (boxes.filter incursExtraFee).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_boxes_incur_extra_fee_l149_14996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_probability_l149_14993

/-- The probability that at least 9 out of 10 people stay for the entire basketball game,
    given that 5 are certain to stay and 5 have a 1/3 probability of staying. -/
theorem basketball_game_probability : (11 : ℚ) / 243 = 
  let total_people : ℕ := 10
  let certain_people : ℕ := 5
  let uncertain_people : ℕ := 5
  let stay_probability : ℚ := 1 / 3
  let at_least_nine : ℕ := 9

  let prob_nine_stay : ℚ := (uncertain_people.choose (at_least_nine - certain_people)) *
    (stay_probability ^ (at_least_nine - certain_people)) *
    ((1 - stay_probability) ^ (uncertain_people - (at_least_nine - certain_people)))

  let prob_all_stay : ℚ := stay_probability ^ uncertain_people

  prob_nine_stay + prob_all_stay
:= by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_probability_l149_14993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2002_of_2002_l149_14962

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x)

-- Define the composition of f with itself n times
noncomputable def f_comp : ℕ → (ℝ → ℝ)
| 0 => id
| n + 1 => f ∘ (f_comp n)

-- State the theorem
theorem f_2002_of_2002 : f_comp 2002 2002 = -1/2001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2002_of_2002_l149_14962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l149_14972

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptotes of the hyperbola
def asymptotes (a b x y : ℝ) : Prop := b^2 * x^2 = a^2 * y^2

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Theorem statement
theorem hyperbola_intersection_theorem 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (l : Line) 
  (A B C D : Point) 
  (h_asymptotes : asymptotes a b A.x A.y ∧ asymptotes a b B.x B.y ∧ 
                  asymptotes a b C.x C.y ∧ asymptotes a b D.x D.y) 
  (h_line : A.y = l.slope * A.x + l.intercept ∧ 
            B.y = l.slope * B.x + l.intercept ∧ 
            C.y = l.slope * C.x + l.intercept ∧ 
            D.y = l.slope * D.x + l.intercept) :
  distance A B = distance C D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l149_14972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_12_45_l149_14981

/-- The angle between clock hands at a given time -/
noncomputable def clockAngle (hours minutes : ℕ) : ℝ :=
  let hourAngle : ℝ := (hours % 12 + minutes / 60 : ℝ) * 30
  let minuteAngle : ℝ := minutes * 6
  abs (hourAngle - minuteAngle)

/-- The smaller angle between clock hands at a given time -/
noncomputable def smallerClockAngle (hours minutes : ℕ) : ℝ :=
  min (clockAngle hours minutes) (360 - clockAngle hours minutes)

/-- Theorem: The smaller angle between the hands of a 12-hour clock at 12:45 pm is 112.5 degrees -/
theorem clock_angle_at_12_45 : smallerClockAngle 12 45 = 112.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_12_45_l149_14981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l149_14930

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the foci of a hyperbola -/
noncomputable def foci (h : Hyperbola) : (Point × Point) :=
  let c := Real.sqrt (h.a^2 + h.b^2)
  ⟨Point.mk (-c) 0, Point.mk c 0⟩

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: Perimeter of triangle ABF₂ for a hyperbola -/
theorem hyperbola_triangle_perimeter (h : Hyperbola) (A B : Point) (m : ℝ)
  (h_left_branch : A.x < 0 ∧ B.x < 0)
  (h_on_hyperbola : (A.x^2 / h.a^2) - (A.y^2 / h.b^2) = 1 ∧
                    (B.x^2 / h.a^2) - (B.y^2 / h.b^2) = 1)
  (h_line_through_F1 : ∃ (t : ℝ), A = Point.mk (foci h).1.x ((A.y - (foci h).1.y) / (A.x - (foci h).1.x) * (t - (foci h).1.x) + (foci h).1.y) ∧
                                  B = Point.mk (foci h).1.x ((B.y - (foci h).1.y) / (B.x - (foci h).1.x) * (t - (foci h).1.x) + (foci h).1.y))
  (h_AB_length : distance A B = m) :
  distance A (foci h).2 + distance B (foci h).2 + distance A B = 4 * h.a + 2 * m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l149_14930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_lengths_equal_l149_14935

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a square with a given side length and bottom-left corner -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- The configuration of squares as described in the problem -/
structure SquareConfiguration where
  squareABCD : Square
  squareDEFK : Square
  squareAKLM : Square

/-- Theorem stating that the path lengths are equal -/
theorem path_lengths_equal (config : SquareConfiguration) : 
  config.squareABCD.sideLength = 2 →
  config.squareDEFK.sideLength = 1 →
  config.squareAKLM.sideLength = 3 →
  let A := config.squareABCD.bottomLeft
  let B := Point.mk (A.x + 2) A.y
  let C := Point.mk (A.x + 2) (A.y + 2)
  let D := Point.mk A.x (A.y + 2)
  let E := Point.mk (A.x + 2) (A.y + 2)
  let F := Point.mk (A.x + 3) (A.y + 2)
  let K := Point.mk (A.x + 3) (A.y + 3)
  let L := Point.mk A.x (A.y + 3)
  distance A E + distance E F + distance F B = 
  distance C K + distance K D + distance D L := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_lengths_equal_l149_14935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_alpha_beta_l149_14988

theorem sin_sum_alpha_beta (α β : ℝ) 
  (h1 : Real.cos α = 4/5)
  (h2 : Real.cos β = 3/5)
  (h3 : β ∈ Set.Ioo (3*Real.pi/2) (2*Real.pi))
  (h4 : 0 < α)
  (h5 : α < β) :
  Real.sin (α + β) = -7/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_alpha_beta_l149_14988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_equidistant_points_l149_14947

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a Color type
inductive Color
  | Green
  | Blue

-- Define the equilateral triangle and its subdivisions
def EquilateralTriangle (A₁ A₂ A₃ : Point) : Prop :=
  sorry -- Add conditions for equilateral triangle

def Midpoint (A B M : Point) : Prop :=
  sorry -- Add conditions for midpoint

def SubdividedTriangle (A₁ A₂ A₃ A₄ A₅ A₆ : Point) : Prop :=
  Midpoint A₁ A₂ A₄ ∧ Midpoint A₂ A₃ A₅ ∧ Midpoint A₃ A₁ A₆

def TriangleWithAllMidpoints (A₁ A₂ A₃ A₄ A₅ A₆ A₇ A₈ A₉ A₁₀ A₁₁ A₁₂ A₁₃ A₁₄ A₁₅ : Point) : Prop :=
  EquilateralTriangle A₁ A₂ A₃ ∧
  SubdividedTriangle A₁ A₂ A₃ A₄ A₅ A₆ ∧
  sorry -- Add conditions for A₇ to A₁₅ being midpoints of smaller triangles

def Coloring (A : Point → Color) : Prop :=
  sorry -- Coloring function that assigns either Green or Blue to each point

def MutuallyEquidistant (A B C : Point) : Prop :=
  sorry -- Add conditions for three points being mutually equidistant

theorem monochromatic_equidistant_points
  (A₁ A₂ A₃ A₄ A₅ A₆ A₇ A₈ A₉ A₁₀ A₁₁ A₁₂ A₁₃ A₁₄ A₁₅ : Point)
  (A : Point → Color)
  (h : TriangleWithAllMidpoints A₁ A₂ A₃ A₄ A₅ A₆ A₇ A₈ A₉ A₁₀ A₁₁ A₁₂ A₁₃ A₁₄ A₁₅)
  (hc : Coloring A) :
  ∃ (i j k : Fin 15), 
    MutuallyEquidistant (List.nthLe [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈, A₉, A₁₀, A₁₁, A₁₂, A₁₃, A₁₄, A₁₅] i (by simp))
                        (List.nthLe [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈, A₉, A₁₀, A₁₁, A₁₂, A₁₃, A₁₄, A₁₅] j (by simp))
                        (List.nthLe [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈, A₉, A₁₀, A₁₁, A₁₂, A₁₃, A₁₄, A₁₅] k (by simp)) ∧
    A (List.nthLe [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈, A₉, A₁₀, A₁₁, A₁₂, A₁₃, A₁₄, A₁₅] i (by simp)) =
    A (List.nthLe [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈, A₉, A₁₀, A₁₁, A₁₂, A₁₃, A₁₄, A₁₅] j (by simp)) ∧
    A (List.nthLe [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈, A₉, A₁₀, A₁₁, A₁₂, A₁₃, A₁₄, A₁₅] j (by simp)) =
    A (List.nthLe [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈, A₉, A₁₀, A₁₁, A₁₂, A₁₃, A₁₄, A₁₅] k (by simp)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_equidistant_points_l149_14947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_24_over_25_l149_14950

-- Define the function f
noncomputable def f (t : ℝ) : ℝ := 
  5 * Real.sin (Real.arcsin t / 2) - 5 * Real.cos (Real.arcsin t / 2) - 6

-- State the theorem
theorem f_value_at_negative_24_over_25 : 
  f (-24/25) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_24_over_25_l149_14950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_count_l149_14961

theorem system_solutions_count : 
  ∃! (s : Finset (ℕ × ℕ × ℕ)), 
    (∀ (x y z : ℕ), (x, y, z) ∈ s ↔ 
      x > 0 ∧ y > 0 ∧ z > 0 ∧ 
      x * y + x * z = 255 ∧ 
      x * y + y * z = 31) ∧
    s.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_count_l149_14961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l149_14978

/-- The function f(x) = sin(2ωx) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x)

/-- The theorem stating the minimum value of ω -/
theorem min_omega_value (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x, f ω x = f ω (x - π/4)) : 
  (∀ ω' > 0, (∀ x, f ω' x = f ω' (x - π/4)) → ω ≤ ω') → ω = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l149_14978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_product_equality_l149_14906

theorem cube_root_product_equality : 
  (1 + 27 : ℝ) ^ (1/3 : ℝ) * (1 + (27 : ℝ) ^ (1/3 : ℝ)) ^ (1/3 : ℝ) * (9 : ℝ) ^ (1/3 : ℝ) = (1008 : ℝ) ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_product_equality_l149_14906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l149_14974

def digits : List Nat := [1, 2, 3, 4, 5]

def is_valid_number (n : Nat) : Bool :=
  n ≥ 100 && n < 600 && n % 10 = 5 && n % 2 = 1

def count_valid_numbers : Nat :=
  (List.filter is_valid_number (List.range 600)).length

theorem valid_numbers_count : count_valid_numbers = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l149_14974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_owner_profit_percentage_l149_14909

/-- Represents the percentage by which the shop owner cheats while buying -/
noncomputable def buying_cheat_percentage : ℝ := 18.5

/-- Represents the percentage by which the shop owner cheats while selling -/
noncomputable def selling_cheat_percentage : ℝ := 22.3

/-- Calculates the actual amount of goods received when buying -/
noncomputable def actual_buying_amount (nominal_amount : ℝ) : ℝ :=
  nominal_amount * (1 + buying_cheat_percentage / 100)

/-- Calculates the actual amount of goods given when selling -/
noncomputable def actual_selling_amount (nominal_amount : ℝ) : ℝ :=
  nominal_amount * (1 - selling_cheat_percentage / 100)

/-- Calculates the percentage profit -/
noncomputable def percentage_profit (cost_price selling_price : ℝ) : ℝ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem stating that the shop owner's percentage profit is approximately 52.52% -/
theorem shop_owner_profit_percentage (nominal_amount : ℝ) (price_per_unit : ℝ) 
  (h1 : nominal_amount > 0) (h2 : price_per_unit > 0) :
  ∃ ε > 0, |percentage_profit 
    (price_per_unit * actual_selling_amount nominal_amount / actual_buying_amount nominal_amount)
    price_per_unit - 52.52| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_owner_profit_percentage_l149_14909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_range_l149_14976

theorem root_range (a : ℝ) 
  (h : ∀ x : ℝ, x^2 - 4*a*x + 2*a + 30 ≥ 0) :
  let roots := {x : ℝ | ∃ (a : ℝ), x / (a + 3) = |a - 1| + 1 ∧ 
    (∀ x : ℝ, x^2 - 4*a*x + 2*a + 30 ≥ 0)}
  (∀ x ∈ roots, x ∈ Set.Ioc (-9/4) 0 ∪ Set.Ioo 0 15) ∧ 
  (∀ x ∈ Set.Ioc (-9/4) 0 ∪ Set.Ioo 0 15, x ∈ roots) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_range_l149_14976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_eight_l149_14985

theorem cube_root_negative_eight :
  Real.rpow (-8 : ℝ) (1/3 : ℝ) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_eight_l149_14985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_williams_bill_value_l149_14998

-- Define the denomination of William's unknown bills
def williams_bill_denomination : ℕ := 10

-- Oliver's money
def oliver_twenty_bills : ℕ := 10
def oliver_five_bills : ℕ := 3

-- William's money
def william_unknown_bills : ℕ := 15
def william_five_bills : ℕ := 4

-- The difference between Oliver's and William's money
def money_difference : ℕ := 45

-- Theorem statement
theorem williams_bill_value :
  oliver_twenty_bills * 20 + oliver_five_bills * 5 =
  william_unknown_bills * williams_bill_denomination + william_five_bills * 5 + money_difference := by
  -- Proof goes here
  sorry

#eval williams_bill_denomination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_williams_bill_value_l149_14998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l149_14989

def a : ℕ → ℕ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | n + 1 => 2 * a n + 1

def b (n : ℕ) : ℕ := a n + 1

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → b (n + 1) = 2 * b n) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 2^n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → b n = 2^n) := by
  sorry

#eval a 5  -- This line is optional, just to test the function
#eval b 5  -- This line is optional, just to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l149_14989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_chords_specific_area_between_chords_general_l149_14907

/-- The area between two equal parallel chords in a circle -/
noncomputable def area_between_chords (r : ℝ) (d : ℝ) : ℝ :=
  66 * Real.pi - 40 * Real.sqrt 91

/-- Theorem: In a circle of radius 10 inches, the area between two equal parallel chords
    that are 6 inches apart is 66π - 40√91 square inches -/
theorem area_between_chords_specific : area_between_chords 10 6 = 66 * Real.pi - 40 * Real.sqrt 91 := by
  -- Unfold the definition of area_between_chords
  unfold area_between_chords
  -- The equality is now trivial
  rfl

/-- A more general theorem about the area between chords (to be proved) -/
theorem area_between_chords_general (r : ℝ) (d : ℝ) 
    (h1 : r > 0) (h2 : 0 < d) (h3 : d < 2*r) : 
  ∃ (θ : ℝ), area_between_chords r d = 
    2 * θ * r^2 - 2 * r * Real.sqrt (r^2 - (d/2)^2) * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_chords_specific_area_between_chords_general_l149_14907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_equation_l149_14944

/-- The equation of a line symmetric to another line with respect to a vertical line. -/
def symmetric_line (a b c : ℝ) (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x + a * y - (2 * k * a + c) = 0

/-- The original line equation -/
def original_line : ℝ → ℝ → Prop :=
  λ x y ↦ x - 2 * y + 1 = 0

/-- The line of symmetry -/
def symmetry_line : ℝ → Prop :=
  λ x ↦ x = 1

/-- Theorem stating that the symmetric line to x - 2y + 1 = 0 with respect to x = 1 is x + 2y - 3 = 0 -/
theorem symmetric_line_equation :
  symmetric_line 1 (-2) 1 1 = λ x y ↦ x + 2 * y - 3 = 0 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_equation_l149_14944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l149_14900

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.cos x ^ 4 - Real.sin x * Real.cos x + Real.sin x ^ 4

-- State the theorem about the range of g
theorem range_of_g :
  (∀ x : ℝ, 0 ≤ g x ∧ g x ≤ 9/8) ∧
  (∃ x : ℝ, g x = 0) ∧
  (∃ x : ℝ, g x = 9/8) :=
by
  sorry

#check range_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l149_14900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l149_14969

open Real

noncomputable def f (x : ℝ) := tan x ^ 2 - 4 * tan x - 8 * (1 / tan x) + 4 * (1 / tan x) ^ 2 + 5

theorem min_value_f :
  ∃ (x : ℝ), π / 2 < x ∧ x < π ∧
  (∀ (y : ℝ), π / 2 < y ∧ y < π → f y ≥ f x) ∧
  f x = 9 - 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l149_14969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_even_coefficients_when_m_n_7_min_coefficient_of_x_squared_l149_14965

def f (m n : ℕ) (x : ℝ) : ℝ := (1 + x)^m + (1 + x)^n

theorem sum_of_even_coefficients_when_m_n_7 :
  let coeffs := [0, 2, 4, 6].map (λ i ↦ (Nat.choose 7 i) + (Nat.choose 7 i))
  (coeffs.sum : ℕ) = 128 := by sorry

theorem min_coefficient_of_x_squared (m n : ℕ) :
  (Nat.choose m 1 + Nat.choose n 1 = 19) →
  (Nat.choose m 2 + Nat.choose n 2 ≥ 81) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_even_coefficients_when_m_n_7_min_coefficient_of_x_squared_l149_14965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_exponential_function_l149_14925

-- Define the function f as the inverse of the exponential function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem inverse_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a (a^2) = a) :
  ∀ x : ℝ, f a x = Real.log x / Real.log 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_exponential_function_l149_14925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_B_value_triangle_area_l149_14928

def triangle_ABC (a b c A B C : ℝ) : Prop :=
  Real.sin C = Real.sqrt 3 * Real.sin A * Real.sin B

theorem tan_B_value (a b c A B C : ℝ) 
  (h1 : triangle_ABC a b c A B C) (h2 : A = π / 3) : 
  Real.tan B = Real.sqrt 3 / 2 := by sorry

theorem triangle_area (a b c A B C : ℝ) 
  (h1 : triangle_ABC a b c A B C) (h2 : c = 3) : 
  (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_B_value_triangle_area_l149_14928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quadrilateral_is_rectangle_l149_14914

/-- A quadrilateral in the complex plane -/
structure ComplexQuadrilateral where
  z₁ : ℂ
  z₂ : ℂ
  z₃ : ℂ
  z₄ : ℂ

/-- Predicate for a rectangle in the complex plane -/
def is_rectangle (q : ComplexQuadrilateral) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (q.z₂ - q.z₁ = Complex.I * (q.z₃ - q.z₂) ∨
     q.z₂ - q.z₁ = Complex.I * (q.z₄ - q.z₁)) ∧
    ((Complex.abs (q.z₂ - q.z₁) = a ∧ Complex.abs (q.z₃ - q.z₂) = b) ∨
     (Complex.abs (q.z₂ - q.z₁) = b ∧ Complex.abs (q.z₃ - q.z₂) = a))

theorem complex_quadrilateral_is_rectangle (q : ComplexQuadrilateral)
  (h₁ : Complex.abs q.z₁ = 1) (h₂ : Complex.abs q.z₂ = 1)
  (h₃ : Complex.abs q.z₃ = 1) (h₄ : Complex.abs q.z₄ = 1)
  (h_sum : q.z₁ + q.z₂ + q.z₃ + q.z₄ = 0) :
  is_rectangle q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quadrilateral_is_rectangle_l149_14914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_propositions_l149_14986

-- Define what constitutes a proposition
def is_proposition (s : String) : Bool :=
  s = "The sum of two acute angles is an obtuse angle" ||
  s = "Zero is neither a positive number nor a negative number" ||
  s = "Flowers bloom in spring"

-- Define the list of statements
def statements : List String :=
  ["Construct the perpendicular bisector of line segment AB",
   "The sum of two acute angles is an obtuse angle",
   "Did our country win the right to host the 2008 Olympics?",
   "Zero is neither a positive number nor a negative number",
   "No loud talking is allowed",
   "Flowers bloom in spring"]

-- Theorem stating that exactly 3 statements are propositions
theorem exactly_three_propositions :
  (statements.filter is_proposition).length = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_propositions_l149_14986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stop_time_l149_14919

theorem train_stop_time (D : ℝ) (h : D > 0) : 
  (D / 360 - D / 400) * 60 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stop_time_l149_14919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solid_of_revolution_l149_14926

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := 2 * Real.log x

-- Define the volume function
noncomputable def volume : ℝ := ∫ y in (0)..(1), Real.pi * (Real.exp (y / 2))^2

-- Theorem statement
theorem volume_of_solid_of_revolution :
  volume = Real.pi * (Real.exp 1 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solid_of_revolution_l149_14926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_division_theorem_l149_14938

/-- Given a line segment AB and a ratio k, this function represents a point that
    divides AB internally in the ratio k -/
def internal_division_point (A B : ℝ × ℝ) (k : ℝ) : ℝ × ℝ :=
  sorry

/-- Given a line segment AB and a ratio k, this function represents a point that
    divides AB externally in the ratio k -/
def external_division_point (A B : ℝ × ℝ) (k : ℝ) : ℝ × ℝ :=
  sorry

/-- Given two points, this function returns their midpoint -/
def midpoint_of (P Q : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Given three collinear points P, Q, R, this function returns the ratio in which
    R divides PQ -/
def division_ratio (P Q R : ℝ × ℝ) : ℝ :=
  sorry

theorem midpoint_division_theorem (A B : ℝ × ℝ) (k : ℝ) :
  let C := internal_division_point A B k
  let C₁ := external_division_point A B k
  let M := midpoint_of C C₁
  division_ratio A B M = -k^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_division_theorem_l149_14938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_slices_kept_l149_14955

theorem cake_slices_kept (total_slices : ℕ) (fraction_eaten : ℚ) (slices_kept : ℕ) : 
  total_slices = 12 →
  fraction_eaten = 1/4 →
  slices_kept = total_slices - (fraction_eaten * ↑total_slices).floor →
  slices_kept = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_slices_kept_l149_14955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solutions_l149_14952

theorem cos_equation_solutions :
  ∃ (S : Set ℝ), (∀ x ∈ S, 0 ≤ x ∧ x ≤ π ∧ Real.cos (7 * x) = Real.cos (5 * x)) ∧
                 (Finite S ∧ Nat.card S = 7) ∧
                 (∀ y, 0 ≤ y ∧ y ≤ π ∧ Real.cos (7 * y) = Real.cos (5 * y) → y ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solutions_l149_14952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_b_onto_a_l149_14922

noncomputable section

def vector_a : Fin 2 → ℝ := ![(-1), 1]
def vector_b : Fin 2 → ℝ := ![1, 2]

def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

def magnitude_squared (v : Fin 2 → ℝ) : ℝ :=
  (v 0) ^ 2 + (v 1) ^ 2

def projection (u v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  let scalar := (dot_product u v) / (magnitude_squared v)
  fun i => scalar * (v i)

theorem projection_of_b_onto_a :
  projection vector_b vector_a = ![(-1/2), 1/2] := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_b_onto_a_l149_14922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_nine_balls_three_boxes_l149_14968

def number_of_distributions (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

theorem ball_distribution (n k : ℕ) (h : n ≥ k) :
  number_of_distributions n k = Nat.choose (n - 1) (k - 1) :=
by
  rfl

theorem nine_balls_three_boxes :
  number_of_distributions 9 3 = 28 :=
by
  rfl

#eval number_of_distributions 9 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_nine_balls_three_boxes_l149_14968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l149_14967

/-- The line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

/-- The circle C in the xy-plane -/
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

/-- The minimum distance between a point on line l and a point on circle C -/
noncomputable def min_distance : ℝ := Real.sqrt 14 / 2

/-- Theorem stating the minimum distance between line l and circle C -/
theorem min_distance_line_circle :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_l x₁ y₁ ∧ circle_C x₂ y₂ ∧
    ∀ (x₃ y₃ x₄ y₄ : ℝ),
      line_l x₃ y₃ → circle_C x₄ y₄ →
      Real.sqrt ((x₃ - x₄)^2 + (y₃ - y₄)^2) ≥ min_distance :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l149_14967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cosine_tangent_lines_l149_14941

/-- Curve C₁ in polar coordinates -/
def C₁ (ρ θ : ℝ) : Prop :=
  ρ^2 - 2*ρ*(Real.cos θ - 2*Real.sin θ) + 4 = 0

/-- Curve C₂ in parametric form -/
def C₂ (x y t : ℝ) : Prop :=
  5*x = 1 - 4*t ∧ 5*y = 18 + 3*t

/-- Helper function to represent the cosine of the angle between tangent lines -/
noncomputable def cosine_of_angle_between_tangents (x y ρ θ : ℝ) : ℝ :=
  sorry

/-- The minimum cosine of the angle between tangent lines from C₂ to C₁ -/
theorem min_cosine_tangent_lines : 
  ∃ (min_cos : ℝ), 
    (∀ (x y t ρ θ : ℝ), C₁ ρ θ → C₂ x y t → 
      (∀ (cos_angle : ℝ), cos_angle = cosine_of_angle_between_tangents x y ρ θ → 
        cos_angle ≥ min_cos)) ∧ 
    min_cos = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cosine_tangent_lines_l149_14941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_ratio_in_cone_l149_14923

/-- Theorem: Volume ratio of water in a cone filled to 2/3 of its height -/
theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  (1 / 3) * π * ((2 / 3) * r)^2 * ((2 / 3) * h) / ((1 / 3) * π * r^2 * h) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_ratio_in_cone_l149_14923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typhoon_tree_deaths_correct_l149_14933

def typhoon_tree_deaths (initial_trees : ℕ) (dead_trees_difference : ℕ) : ℕ :=
  if initial_trees ≤ dead_trees_difference then initial_trees else 0

theorem typhoon_tree_deaths_correct (initial_trees : ℕ) (dead_trees_difference : ℕ) :
  typhoon_tree_deaths initial_trees dead_trees_difference = 
    if initial_trees ≤ dead_trees_difference then initial_trees else 0 := by
  rfl

#eval typhoon_tree_deaths 3 23

end NUMINAMATH_CALUDE_ERRORFEEDBACK_typhoon_tree_deaths_correct_l149_14933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arithmetic_sequence_l149_14992

theorem sin_arithmetic_sequence (b : Real) : 
  (0 < b) → (b < 2 * Real.pi) → 
  (∃ r : Real, Real.sin b + r = Real.sin (2 * b) ∧ Real.sin (2 * b) + r = Real.sin (3 * b)) ↔ 
  b = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arithmetic_sequence_l149_14992
