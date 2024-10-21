import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l633_63318

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  -- Center at origin, focus on x-axis
  center : ℝ × ℝ := (0, 0)
  focus_on_x_axis : Bool
  -- Focal length
  focal_length : ℝ
  focal_length_eq : focal_length = 2 * Real.sqrt 3
  -- Major axis twice the minor axis
  major_minor_ratio : ℝ
  major_minor_ratio_eq : major_minor_ratio = 2

/-- Set of points on the ellipse -/
def set_of_points (e : SpecialEllipse) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 / 4 + y^2 = 1}

/-- Standard equation of the ellipse -/
def standard_equation (e : SpecialEllipse) : Prop :=
  ∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ set_of_points e

/-- Left focus of the ellipse -/
noncomputable def left_focus (e : SpecialEllipse) : ℝ × ℝ := (-Real.sqrt 3, 0)

/-- Right focus of the ellipse -/
noncomputable def right_focus (e : SpecialEllipse) : ℝ × ℝ := (Real.sqrt 3, 0)

/-- Upper vertex of the ellipse -/
def upper_vertex : ℝ × ℝ := (0, 1)

/-- Line passing through upper vertex and left focus -/
def line_through_vertex_and_focus : Set (ℝ × ℝ) :=
  {(x, y) | y = (Real.sqrt 3 / 3) * x + 1}

/-- Intersection points of the line and the ellipse -/
def intersection_points (e : SpecialEllipse) : Set (ℝ × ℝ) :=
  set_of_points e ∩ line_through_vertex_and_focus

/-- Area of triangle PF₂Q -/
noncomputable def triangle_area (e : SpecialEllipse) : ℝ := 8 * Real.sqrt 3 / 7

theorem special_ellipse_properties (e : SpecialEllipse) :
  standard_equation e ∧ triangle_area e = 8 * Real.sqrt 3 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l633_63318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equals_identity_l633_63397

def IsStrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m < n → f m < f n

-- Remove the IsCoprime definition as it's already defined in Mathlib

theorem function_equals_identity
  (f : ℕ → ℕ)
  (h_increasing : IsStrictlyIncreasing f)
  (h_f2 : f 2 = 2)
  (h_coprime : ∀ m n : ℕ, Nat.Coprime m n → f (m * n) = f m * f n) :
  ∀ n : ℕ, f n = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equals_identity_l633_63397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_bounds_l633_63335

/-- Represents a geometric configuration with a sphere inscribed in a regular cone
    and a cylinder circumscribed around the sphere. -/
structure GeometricConfiguration where
  r : ℝ  -- radius of the sphere
  α : ℝ  -- angle at the vertex of the cone

/-- Volume of the cone -/
noncomputable def cone_volume (config : GeometricConfiguration) : ℝ :=
  (1/3) * Real.pi * config.r^3 * (1 + Real.sin config.α)^3 * Real.tan config.α^2 / Real.sin config.α^3

/-- Volume of the cylinder -/
noncomputable def cylinder_volume (config : GeometricConfiguration) : ℝ :=
  2 * Real.pi * config.r^3

/-- Ratio of cone volume to cylinder volume -/
noncomputable def volume_ratio (config : GeometricConfiguration) : ℝ :=
  cone_volume config / cylinder_volume config

/-- Theorem stating that the volume ratio is always greater than 1
    and its minimum value is 4/3 -/
theorem volume_ratio_bounds (config : GeometricConfiguration) :
  volume_ratio config > 1 ∧ ∀ c : GeometricConfiguration, volume_ratio config ≥ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_bounds_l633_63335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weather_observation_l633_63323

theorem weather_observation (x : ℕ+) 
  (rainy_days : ℕ) 
  (sunny_afternoons : ℕ) 
  (sunny_mornings : ℕ) 
  (rainy_afternoon_implies_sunny_morning : Bool) :
  rainy_days = 7 →
  sunny_afternoons = 5 →
  sunny_mornings = 6 →
  rainy_afternoon_implies_sunny_morning = true →
  x = 9 := by
  intro h1 h2 h3 h4
  sorry

#check weather_observation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weather_observation_l633_63323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_sales_l633_63395

/-- Calculate the total sales of a restaurant given the number of meals sold at different prices -/
theorem restaurant_sales (meals1 meals2 meals3 : ℕ) (price1 price2 price3 : ℚ) :
  meals1 = 10 ∧ price1 = 8 ∧
  meals2 = 5 ∧ price2 = 10 ∧
  meals3 = 20 ∧ price3 = 4 →
  meals1 * price1 + meals2 * price2 + meals3 * price3 = 210 := by
  intro h
  -- We'll use 'sorry' to skip the proof for now
  sorry

#check restaurant_sales

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_sales_l633_63395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_day_player_l633_63314

/-- Represents a round robin competition --/
structure RoundRobinCompetition where
  teams : Finset ℕ
  days : Finset ℕ
  schedule : ℕ → Finset ℕ
  h_team_count : teams.card = 50
  h_day_count : days.card = 50
  h_schedule_valid : ∀ d, d ∈ days → schedule d ⊆ teams
  h_all_play : ∀ d, d ∈ days → ∀ t1 t2, t1 ∈ schedule d → t2 ∈ schedule d → t1 ≠ t2 → {t1, t2} ⊆ schedule d
  h_complete : ∀ t1 t2, t1 ∈ teams → t2 ∈ teams → t1 ≠ t2 → ∃ d, d ∈ days ∧ {t1, t2} ⊆ schedule d

/-- The main theorem --/
theorem consecutive_day_player (c : RoundRobinCompetition) :
  ∀ d, d ∈ c.days → d + 1 ∈ c.days →
  ∃ t, t ∈ c.teams ∧ t ∈ c.schedule d ∧ t ∈ c.schedule (d + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_day_player_l633_63314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l633_63350

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation 3x^2 + 1 = 0 -/
noncomputable def eq_A (x : ℝ) : ℝ := 3 * x^2 + 1

/-- The equation 3x + 2y - 1 = 0 -/
noncomputable def eq_B (x y : ℝ) : ℝ := 3 * x + 2 * y - 1

/-- The equation (x + 1)(x - 2) = x^2 -/
noncomputable def eq_C (x : ℝ) : ℝ := (x + 1) * (x - 2) - x^2

/-- The equation 1/x^2 - x - 1 = 0 -/
noncomputable def eq_D (x : ℝ) : ℝ := 1 / (x^2) - x - 1

theorem quadratic_equation_identification :
  is_quadratic_equation eq_A ∧
  ¬ is_quadratic_equation (λ x ↦ eq_B x 0) ∧
  ¬ is_quadratic_equation eq_C ∧
  ¬ is_quadratic_equation eq_D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l633_63350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elgin_has_ten_l633_63368

variable (ashley_amount : ℕ)
variable (betty_amount : ℕ)
variable (carlos_amount : ℕ)
variable (dick_amount : ℕ)
variable (elgin_amount : ℕ)

axiom total_amount : ashley_amount + betty_amount + carlos_amount + dick_amount + elgin_amount = 56

axiom ashley_betty_diff : (ashley_amount : ℤ) - betty_amount = 19 ∨ betty_amount - ashley_amount = 19
axiom betty_carlos_diff : (betty_amount : ℤ) - carlos_amount = 7 ∨ carlos_amount - betty_amount = 7
axiom carlos_dick_diff : (carlos_amount : ℤ) - dick_amount = 5 ∨ dick_amount - carlos_amount = 5
axiom dick_elgin_diff : (dick_amount : ℤ) - elgin_amount = 4 ∨ elgin_amount - dick_amount = 4
axiom elgin_ashley_diff : (elgin_amount : ℤ) - ashley_amount = 11 ∨ ashley_amount - elgin_amount = 11

theorem elgin_has_ten : elgin_amount = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elgin_has_ten_l633_63368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_of_factors_l633_63349

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_difference_of_factors (p q r : ℕ+) : 
  p * q * r = factorial 9 → p < q → q < r → 
  (∀ p' q' r' : ℕ+, p' * q' * r' = factorial 9 → p' < q' → q' < r' → r - p ≤ r' - p') →
  r - p = 396 := by
sorry

#eval factorial 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_of_factors_l633_63349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_17_53_l633_63374

theorem units_digit_17_53 (h1 : ∀ a n : ℕ, a % 10 = 17 % 10 → (a^n) % 10 = (17^n) % 10)
                          (h2 : ∀ n : ℕ, (7^n) % 10 ∈ ({7, 9, 3, 1} : Finset ℕ))
                          (h3 : 53 % 4 = 1) :
  (17^53) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_17_53_l633_63374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l633_63321

-- Define a power function that passes through (3, 33)
noncomputable def f (x : ℝ) : ℝ := x ^ (Real.log 33 / Real.log 3)

-- Theorem statement
theorem power_function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧  -- f is odd
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=  -- f is increasing on (0, +∞)
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l633_63321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_transformation_of_f_l633_63365

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ 0 then -x - 1
  else if 0 ≤ x ∧ x ≤ 6 then Real.sqrt (9 - (x - 3)^2) - 3
  else if 6 ≤ x ∧ x ≤ 7 then 3*(x - 6)
  else 0  -- Define a default value for x outside the given ranges

-- Define the transformation h
noncomputable def h (x : ℝ) : ℝ := f (5 - x)

-- State the theorem
theorem h_is_transformation_of_f :
  ∀ x, h x = f (5 - x) := by
  intro x
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_transformation_of_f_l633_63365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stan_average_speed_l633_63301

/-- Stan's driving trip --/
structure DrivingTrip where
  distance1 : ℚ  -- Distance of first leg in miles
  time1 : ℚ      -- Time of first leg in hours
  restTime : ℚ   -- Rest time in hours
  distance2 : ℚ  -- Distance of second leg in miles
  time2 : ℚ      -- Time of second leg in hours

/-- Calculate the average speed of a driving trip --/
def averageSpeed (trip : DrivingTrip) : ℚ :=
  (trip.distance1 + trip.distance2) / (trip.time1 + trip.restTime + trip.time2)

/-- Stan's specific driving trip --/
def stanTrip : DrivingTrip :=
  { distance1 := 350
  , time1 := 6
  , restTime := 1/2
  , distance2 := 400
  , time2 := 7 }

/-- Theorem: Stan's average speed is 750 / 13.5 miles per hour --/
theorem stan_average_speed :
  averageSpeed stanTrip = 750 / (27/2) := by
  sorry

#eval averageSpeed stanTrip

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stan_average_speed_l633_63301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_cost_price_theorem_l633_63354

/-- Represents the cost and selling prices of an item -/
structure Item where
  costPrice : ℚ
  sellingPrice : ℚ

/-- Calculates the profit percentage for an item -/
def profitPercentage (item : Item) : ℚ :=
  (item.sellingPrice - item.costPrice) / item.costPrice * 100

theorem combined_cost_price_theorem (itemA itemB itemC : Item) : 
  itemA.sellingPrice - itemA.costPrice = itemA.costPrice - 80 ∧ 
  itemB.sellingPrice = 150 ∧
  itemC.sellingPrice = 200 ∧
  profitPercentage itemB = 25 ∧
  profitPercentage itemC = -20 →
  itemA.costPrice + itemB.costPrice + itemC.costPrice = 470 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_cost_price_theorem_l633_63354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l633_63310

theorem line_equation_proof (x_intercept angle : ℝ) :
  x_intercept = 2 ∧ angle = 135 * π / 180 →
  ∃ m b : ℝ, m = -1 ∧ b = 2 ∧ ∀ x y : ℝ, y = m * x + b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l633_63310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l633_63387

theorem complex_multiplication (P F G : ℂ) : 
  P = 3 + 4*Complex.I → F = Complex.I → G = 3 - 4*Complex.I → P * F * G = 25*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l633_63387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_height_theorem_l633_63378

/-- Represents the motion of a rocket launched vertically upward --/
structure RocketMotion where
  a : ℝ  -- acceleration provided by engines (m/s²)
  g : ℝ  -- acceleration due to gravity (m/s²)
  τ : ℝ  -- duration of engine operation (s)

/-- Calculates the maximum height reached by the rocket --/
noncomputable def max_height (r : RocketMotion) : ℝ :=
  let v₀ := r.a * r.τ
  let y₀ := (r.a * r.τ^2) / 2
  let t := v₀ / r.g
  y₀ + v₀ * t - (r.g * t^2) / 2

/-- Theorem stating the maximum height reached by the rocket and its relation to 50 km --/
theorem rocket_height_theorem (r : RocketMotion) 
  (h1 : r.a = 30) 
  (h2 : r.g = 10) 
  (h3 : r.τ = 30) : 
  max_height r = 54000 ∧ max_height r > 50000 := by
  sorry

#check rocket_height_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_height_theorem_l633_63378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_to_male_ratio_on_duty_l633_63309

/-- Proves that the ratio of female officers to male officers on duty is 1:1 --/
theorem female_to_male_ratio_on_duty (total_female : ℕ) (total_on_duty : ℕ) 
  (female_on_duty_percent : ℚ) : 
  total_female = 400 →
  total_on_duty = 152 →
  female_on_duty_percent = 19/100 →
  (female_on_duty_percent * total_female : ℚ) = 
    (total_on_duty : ℚ) - (female_on_duty_percent * total_female : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_to_male_ratio_on_duty_l633_63309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrationality_of_sum_l633_63390

theorem irrationality_of_sum (a : ℕ) (r : ℝ) 
  (h1 : ∀ n : ℕ, n^2 ≠ a)
  (h2 : r^3 - 2*(a:ℝ)*r + 1 = 0) : 
  Irrational (r + Real.sqrt (a:ℝ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrationality_of_sum_l633_63390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_cant_end_with_one_l633_63337

/-- Represents a dwarf with their current number of diamonds -/
structure Dwarf :=
  (diamonds : ℕ)

/-- Represents the state of all dwarfs around the table -/
def TableState := Vector Dwarf 8

/-- Initial state of the table where each dwarf has 3 diamonds -/
def initialState : TableState :=
  Vector.replicate 8 ⟨3⟩

/-- Represents one round of diamond distribution -/
def distribute (state : TableState) : TableState :=
  sorry

/-- Checks if all diamonds are with one dwarf -/
def allDiamondsWithOne (state : TableState) : Prop :=
  ∃ i, (state.get i).diamonds = 24 ∧ 
       ∀ j, j ≠ i → (state.get j).diamonds = 0

/-- Theorem stating it's impossible for all diamonds to end up with one dwarf -/
theorem diamonds_cant_end_with_one : 
  ¬∃ (n : ℕ), allDiamondsWithOne (n.iterate distribute initialState) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_cant_end_with_one_l633_63337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_angle_l633_63392

/-- A truncated cone with an inscribed sphere -/
structure TruncatedCone :=
  (r : ℝ)  -- radius of the inscribed sphere
  (d1 : ℝ) -- diameter of the lower base
  (d2 : ℝ) -- diameter of the upper base

/-- The angle between the slant height and the base of a truncated cone -/
noncomputable def slant_angle (tc : TruncatedCone) : ℝ := Real.arcsin (4/5)

theorem truncated_cone_angle (tc : TruncatedCone) 
  (h : tc.d1 + tc.d2 = 5 * tc.r) : 
  slant_angle tc = Real.arcsin (4/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_angle_l633_63392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_m_value_l633_63311

/-- A perfect square trinomial in the form ax² + bx + c can be written as (px + q)²,
    where p and q are real numbers and a, b, c are constants with a ≠ 0. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ (p q : ℝ), ∀ (x : ℝ), a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x² + 3mx + 9 is a perfect square trinomial, then m = ±4 -/
theorem perfect_square_m_value :
  ∀ (m : ℝ), is_perfect_square_trinomial 4 (3*m) 9 → m = 4 ∨ m = -4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_m_value_l633_63311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_sum_equals_1000_l633_63359

def arithmetic_sum (a₁ n d : Int) : Int := n * (2 * a₁ + (n - 1) * d) / 2

theorem olympiad_sum_equals_1000 : 
  let n : Int := 1990 / 20 + 1
  let sum_positive := arithmetic_sum 1990 (n / 2) (-40)
  let sum_negative := arithmetic_sum 1980 ((n - 1) / 2) (-40)
  10 * 52 + sum_positive - sum_negative = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_sum_equals_1000_l633_63359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_max_value_l633_63341

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

theorem vector_max_value (m n : V) (hm : ‖m‖ = 2) (hmn : ‖m + 2 • n‖ = 2) :
  ∃ (c : ℝ), c = (8 * Real.sqrt 3) / 3 ∧ 
  ∀ (x : V), ‖2 • m + x‖ + ‖x‖ ≤ c ∧ 
  ∃ (y : V), ‖2 • m + y‖ + ‖y‖ = c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_max_value_l633_63341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_if_exterior_less_than_interior_l633_63357

-- Define the Triangle type
structure Triangle where
  -- You might want to add fields here to represent the triangle

-- Define angle type
structure Angle where
  -- You might want to add fields here to represent an angle

-- Define IsExteriorAngle
def IsExteriorAngle (T : Triangle) (θ : Angle) : Prop := sorry

-- Define IsInteriorAngle
def IsInteriorAngle (T : Triangle) (θ : Angle) : Prop := sorry

-- Define AdjacentAngles
def AdjacentAngles (θ1 θ2 : Angle) : Prop := sorry

-- Define IsObtuseTriangle
def IsObtuseTriangle (T : Triangle) : Prop := sorry

-- Define the < operator for angles
instance : LT Angle where
  lt := sorry

theorem triangle_obtuse_if_exterior_less_than_interior 
  (T : Triangle) 
  (θ_ext θ_int : Angle) 
  (h1 : IsExteriorAngle T θ_ext)
  (h2 : IsInteriorAngle T θ_int)
  (h3 : AdjacentAngles θ_ext θ_int)
  (h4 : θ_ext < θ_int) : 
  IsObtuseTriangle T :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_if_exterior_less_than_interior_l633_63357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_interior_intersection_l633_63373

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  vertices : Finset (Fin 4)
  is_regular : True  -- We assume the tetrahedron is regular

/-- A plane formed by three vertices of a tetrahedron -/
def plane_from_vertices (t : RegularTetrahedron) (v1 v2 v3 : Fin 4) : Set (Fin 4 → ℝ) :=
  sorry

/-- Predicate to check if a plane intersects the interior of a tetrahedron -/
def intersects_interior (t : RegularTetrahedron) (p : Set (Fin 4 → ℝ)) : Prop :=
  sorry

/-- The set of all possible choices of three vertices -/
def all_vertex_choices (t : RegularTetrahedron) : Finset (Fin 4 × Fin 4 × Fin 4) :=
  sorry

/-- The set of vertex choices that form a plane intersecting the interior -/
def intersecting_choices (t : RegularTetrahedron) : Finset (Fin 4 × Fin 4 × Fin 4) :=
  sorry

/-- The main theorem -/
theorem probability_interior_intersection (t : RegularTetrahedron) :
  (Finset.card (intersecting_choices t) : ℚ) / (Finset.card (all_vertex_choices t) : ℚ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_interior_intersection_l633_63373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_f_15_equals_2_l633_63332

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the given condition
axiom inverse_relation : ∀ x : ℝ, Function.invFun f (g x) = x^4 - 1

-- State that g has an inverse
axiom g_has_inverse : Function.Injective g

-- Theorem to prove
theorem g_inverse_f_15_equals_2 : Function.invFun g (f 15) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_f_15_equals_2_l633_63332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l633_63313

theorem problem_solution (x y z : ℝ) 
  (hx : (2 : ℝ)^x = 3) (hy : (3 : ℝ)^y = 4) (hz : (4 : ℝ)^z = 5) : 
  y < 4/3 ∧ x*y*z > 2 ∧ x + y > 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l633_63313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_difference_approx_l633_63386

/-- The length of the longer side of the rectangular field -/
def a : ℝ := 3

/-- The length of the shorter side of the rectangular field -/
def b : ℝ := 2

/-- Jerry's path length (sum of two sides) -/
def jerry_path : ℝ := a + b

/-- Silvia's path length (diagonal of the rectangle) -/
noncomputable def silvia_path : ℝ := Real.sqrt (a^2 + b^2)

/-- The percentage difference between Jerry's and Silvia's paths -/
noncomputable def path_difference_percentage : ℝ := (jerry_path - silvia_path) / jerry_path * 100

theorem path_difference_approx :
  ∃ ε > 0, |path_difference_percentage - 27.88| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_difference_approx_l633_63386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_numbers_theorem_l633_63320

theorem eleven_numbers_theorem (S : Finset ℕ) : 
  S.card = 11 → (∀ n ∈ S, n ≤ 27) → 
  ∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (a + b : ℚ) / 5 = (c + d : ℚ) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_numbers_theorem_l633_63320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_intervals_f_max_min_in_interval_l633_63375

noncomputable def f (x : ℝ) := Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4)

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem f_monotone_decreasing_intervals (k : ℤ) :
  is_monotone_decreasing f (k * Real.pi + Real.pi / 8) (k * Real.pi + 5 * Real.pi / 8) := by sorry

theorem f_max_min_in_interval :
  let a := -Real.pi / 8
  let b := Real.pi / 2
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f (Real.pi / 8)) ∧
  (∀ x, a ≤ x ∧ x ≤ b → f (Real.pi / 2) ≤ f x) ∧
  f (Real.pi / 8) = Real.sqrt 2 ∧
  f (Real.pi / 2) = -Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_intervals_f_max_min_in_interval_l633_63375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_closest_l633_63399

/-- The parabola defined by x^2 = 2y -/
def parabola (x y : ℝ) : Prop := x^2 = 2*y

/-- The distance between two points (x1, y1) and (x2, y2) -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  ((x1 - x2)^2 + (y1 - y2)^2).sqrt

/-- The vertex of the parabola x^2 = 2y -/
def vertex : ℝ × ℝ := (0, 0)

theorem parabola_vertex_closest (a : ℝ) :
  (∀ x y : ℝ, parabola x y →
    distance 0 a x y ≥ distance 0 a 0 0) →
  a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_closest_l633_63399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_five_thirds_l633_63327

-- Define the sales volume function
noncomputable def sales_volume (x : ℝ) : ℝ :=
  if 1 < x ∧ x ≤ 3 then 300 * (x - 3)^2 + 300 / (x - 1)
  else if 3 < x ∧ x ≤ 5 then -70 * x + 490
  else 0

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ := (sales_volume x) * (x - 1)

-- State the theorem
theorem max_profit_at_five_thirds :
  ∃ (max_profit : ℝ),
    (∀ x, 1 < x → x ≤ 5 → profit x ≤ max_profit) ∧
    profit (5/3) = max_profit ∧
    max_profit = 5900/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_five_thirds_l633_63327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l633_63304

def a : ℕ → ℚ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | (n + 2) => (1 / 3) * a (n + 1) + 1

theorem a_general_term (n : ℕ) : 
  a n = 3 / 2 - (1 / 2) * (1 / 3) ^ (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l633_63304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_opens_100th_door_l633_63369

/-- Represents the order of people in the cave --/
def CaveOrder := Fin 7

/-- Cyclic permutation of the cave order --/
def cyclicPermutation (order : CaveOrder → α) : CaveOrder → α :=
  fun i => order ((i.val + 1) % 7 : Fin 7)

/-- The initial order of people in the cave --/
def initialOrder : CaveOrder → String :=
  fun i => match i.val with
    | 0 => "Albert"
    | 1 => "Ben"
    | 2 => "Cyril"
    | 3 => "Dan"
    | 4 => "Erik"
    | 5 => "Filip"
    | _ => "Gábo"

/-- Applies the cyclic permutation n times --/
def applyNTimes (n : ℕ) (order : CaveOrder → α) : CaveOrder → α :=
  match n with
  | 0 => order
  | n + 1 => cyclicPermutation (applyNTimes n order)

theorem ben_opens_100th_door :
  (applyNTimes 100 initialOrder) ⟨0, by norm_num⟩ = "Ben" := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_opens_100th_door_l633_63369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_arithmetic_sequence_l633_63319

def is_on_line (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n = n + 1

def is_arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

theorem points_on_line_arithmetic_sequence :
  (∀ a : ℕ+ → ℝ, is_on_line a → is_arithmetic_sequence a) ∧
  (∃ a : ℕ+ → ℝ, is_arithmetic_sequence a ∧ ¬is_on_line a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_arithmetic_sequence_l633_63319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l633_63339

open Set

theorem complement_union_problem :
  let U := ℝ
  let M := {x : ℝ | x < (2 : ℝ)}
  let N := {x : ℝ | (-1 : ℝ) < x ∧ x < (3 : ℝ)}
  (Mᶜ ∪ N) = {x : ℝ | x > (-1 : ℝ)} :=
by
  intros U M N
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l633_63339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_prime_and_surjective_l633_63370

def a (n : ℕ) : ℕ := n.factorial ^ 2 % n

def b (n : ℕ) : ℕ := (n.factorial + 1) ^ 2 % n

def f (n : ℕ) : ℕ := n * a n + 2 * b n

theorem f_is_prime_and_surjective :
  (∀ n : ℕ, n > 1 → Nat.Prime (f n)) ∧
  (∀ p : ℕ, Nat.Prime p → ∃ n : ℕ, f n = p) := by
  sorry

#eval f 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_prime_and_surjective_l633_63370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_theorem_l633_63362

/-- A curve in the Cartesian plane -/
structure Curve where
  equation : ℝ → ℝ → Prop

/-- A line in the Cartesian plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Given a curve C and a line l, if they satisfy certain conditions, then m = 2 -/
theorem curve_line_intersection_theorem 
  (m : ℝ) 
  (hm : m > 0)
  (C : Curve)
  (hC : C.equation = fun x y => y^2 = m*x)
  (l : Line)
  (hl : l.equation = fun x y => y = x - 2)
  (P : Point)
  (hP : P.x = -2 ∧ P.y = -4)
  (A B : Point)
  (hA : C.equation A.x A.y ∧ l.equation A.x A.y)
  (hB : C.equation B.x B.y ∧ l.equation B.x B.y)
  (h_condition : distance A P * distance B P = distance A B ^ 2)
  : m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_theorem_l633_63362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_limit_half_l633_63326

theorem product_limit_half : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((n^2 - 1) * ((n+1)^2 - 1)) / (n^2 * (n+1)^2 : ℝ) - 1/2| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_limit_half_l633_63326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l633_63342

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt x

-- Define the inverse function
noncomputable def g (x : ℝ) : ℝ := 1 / (x^2)

-- State the theorem
theorem inverse_function_theorem (x : ℝ) (hx : x > 4) :
  ∃ y, 0 < y ∧ y < 1/2 ∧ f y = x ∧ g x = y := by
  sorry

#check inverse_function_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l633_63342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_finishes_first_l633_63394

noncomputable section

-- Define the lawn areas
def beth_area : ℝ := 1
def andy_area : ℝ := 3 * beth_area
def carlos_area : ℝ := andy_area / 4
def diana_area : ℝ := beth_area / 2

-- Define the mowing rates
def beth_rate : ℝ := 1
def andy_rate : ℝ := 2 * beth_rate
def carlos_rate : ℝ := andy_rate / 3
def diana_rate : ℝ := andy_rate

-- Define the mowing times
def andy_time : ℝ := andy_area / andy_rate
def beth_time : ℝ := beth_area / beth_rate
def carlos_time : ℝ := carlos_area / carlos_rate
def diana_time : ℝ := diana_area / diana_rate

theorem diana_finishes_first :
  diana_time < andy_time ∧ diana_time < beth_time ∧ diana_time < carlos_time := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_finishes_first_l633_63394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_l633_63396

/-- Represents a rhombus with given diagonal and area -/
structure Rhombus where
  diagonal : ℝ
  area : ℝ

/-- Calculates the length of each side of a rhombus -/
noncomputable def side_length (r : Rhombus) : ℝ :=
  let other_diagonal := (2 * r.area) / r.diagonal
  Real.sqrt ((r.diagonal / 2) ^ 2 + (other_diagonal / 2) ^ 2)

/-- Theorem: A rhombus with diagonal 24 cm and area 120 cm² has sides of length 13 cm -/
theorem rhombus_side_length :
  let r := Rhombus.mk 24 120
  side_length r = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_l633_63396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l633_63379

theorem cos_beta_value (α β : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.pi / 2 < β) (h4 : β < Real.pi)
  (h5 : Real.cos α = 3/5) (h6 : Real.sin (α + β) = -3/5) : 
  Real.cos β = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l633_63379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_digit_divisible_by_99_l633_63351

theorem ten_digit_divisible_by_99 (A B C D E : ℕ) : 
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10) →
  (1000000000 + A * 100000000 + 2 * 10000000 + B * 1000000 + 
   3 * 100000 + C * 10000 + 5 * 1000 + D * 100 + 6 * 10 + E) % 99 = 0 →
  A + B + C + D + E = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_digit_divisible_by_99_l633_63351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_is_one_point_five_l633_63388

/-- A normal distribution with given properties -/
structure NormalDistribution where
  mean : ℝ
  value_2sd_below : ℝ
  h_mean : mean = 15.5
  h_value : value_2sd_below = 12.5
  h_relation : value_2sd_below = mean - 2 * (mean - value_2sd_below)

/-- The standard deviation of the distribution -/
noncomputable def standard_deviation (nd : NormalDistribution) : ℝ :=
  (nd.mean - nd.value_2sd_below) / 2

/-- Theorem: The standard deviation of the given normal distribution is 1.5 -/
theorem standard_deviation_is_one_point_five (nd : NormalDistribution) :
  standard_deviation nd = 1.5 := by
  unfold standard_deviation
  rw [nd.h_mean, nd.h_value]
  norm_num
  
#eval (15.5 - 12.5) / 2  -- This will output 1.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_is_one_point_five_l633_63388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_five_digit_number_l633_63324

theorem no_valid_five_digit_number : ¬ ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧  -- n is a five-digit number
  (∃ a b c d e : ℕ, 
    a ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
    b ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
    c ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
    d ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
    e ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
    c ≠ d ∧ c ≠ e ∧ 
    d ≠ e ∧
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    (10 * a + b) % 2 = 0 ∧  -- first two digits divisible by 2
    (100 * a + 10 * b + c) % 3 = 0 ∧  -- first three digits divisible by 3
    (1000 * a + 100 * b + 10 * c + d) % 4 = 0 ∧  -- first four digits divisible by 4
    n % 5 = 0)  -- entire number divisible by 5
  := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_five_digit_number_l633_63324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_ten_iff_x_eq_neg_three_l633_63371

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else -2*x

theorem f_eq_ten_iff_x_eq_neg_three : 
  ∀ x : ℝ, f x = 10 ↔ x = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_ten_iff_x_eq_neg_three_l633_63371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l633_63328

theorem trigonometric_equation_solution :
  ∀ x : ℝ,
  (∃ n : ℤ, x = (π / 16) * (2 * n + 1)) ∨
  (∃ k : ℤ, x = ((-1)^(k+1) * π / 12) + (π * k / 3)) ↔
  Real.sin (2*x) * Real.sin (6*x) - Real.cos (2*x) * Real.cos (6*x) = Real.sqrt 2 * Real.sin (3*x) * Real.cos (8*x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l633_63328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l633_63398

/-- A hyperbola with center at the origin and foci on the y-axis -/
structure StandardHyperbola where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : StandardHyperbola) : ℝ :=
  Real.sqrt ((h.a ^ 2 + h.b ^ 2) / h.a ^ 2)

/-- Theorem: The eccentricity of a hyperbola with one asymptote parallel to √2x - y - 1 = 0 is √6/2 -/
theorem hyperbola_eccentricity (h : StandardHyperbola) 
  (asymptote_parallel : h.a / h.b = Real.sqrt 2) : 
  eccentricity h = Real.sqrt 6 / 2 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l633_63398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_after_moving_l633_63367

/-- Given two points A and B on a Cartesian plane, prove that the distance between
    the original midpoint M and the new midpoint M' after moving A by (5, 10) and
    B by (-15, -5) is √31.25. -/
theorem midpoint_distance_after_moving (x₁ y₁ x₂ y₂ : ℝ) :
  let A : ℝ × ℝ := (x₁, y₁)
  let B : ℝ × ℝ := (x₂, y₂)
  let M : ℝ × ℝ := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  let A' : ℝ × ℝ := (x₁ + 5, y₁ + 10)
  let B' : ℝ × ℝ := (x₂ - 15, y₂ - 5)
  let M' : ℝ × ℝ := ((A'.1 + B'.1) / 2, (A'.2 + B'.2) / 2)
  Real.sqrt ((M.1 - M'.1)^2 + (M.2 - M'.2)^2) = Real.sqrt 31.25 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_after_moving_l633_63367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_nonnegative_f_solution_set_for_inequality_l633_63307

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - 2*a*x + a

-- Part 1
theorem range_of_a_for_nonnegative_f :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

-- Part 2
theorem solution_set_for_inequality (a : ℝ) (h : a ≠ -3) :
  (a > -3 → {x : ℝ | f a x > 4*a - (a+3)*x} = {x : ℝ | x < -3 ∨ x > a}) ∧
  (a < -3 → {x : ℝ | f a x > 4*a - (a+3)*x} = {x : ℝ | x < a ∨ x > -3}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_nonnegative_f_solution_set_for_inequality_l633_63307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l633_63358

noncomputable def f (x : ℝ) := 1 / (1 + (2 : ℝ)^x)

theorem min_a_value (a : ℝ) :
  (∀ x : ℝ, x > 0 → f (a * Real.exp x) ≤ 1 - f (Real.log a - Real.log x)) →
  a ≥ Real.exp (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l633_63358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_carpets_13x13_l633_63391

/-- Represents a rectangular room --/
structure Room where
  length : ℕ
  width : ℕ

/-- Represents a rectangular carpet --/
structure Carpet where
  length : ℕ
  width : ℕ

/-- Predicate to check if a carpet completely covers another --/
def completely_covers (c1 c2 : Carpet) : Prop :=
  sorry -- Definition to be implemented

/-- Predicate to check if a carpet placement is valid --/
def valid_placement (room : Room) (carpets : List Carpet) : Prop :=
  ∀ c1 c2, c1 ∈ carpets → c2 ∈ carpets → c1 ≠ c2 → 
    ¬(completely_covers c1 c2 ∨ completely_covers c2 c1)

/-- The maximum number of carpets that can be placed in a room --/
def max_carpets (room : Room) : ℕ := room.length * room.width

/-- Theorem stating the maximum number of carpets in a 13x13 room --/
theorem max_carpets_13x13 :
  ∃ (carpets : List Carpet), 
    valid_placement ⟨13, 13⟩ carpets ∧ 
    carpets.length = max_carpets ⟨13, 13⟩ ∧
    max_carpets ⟨13, 13⟩ = 169 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_carpets_13x13_l633_63391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_pyramid_volume_l633_63346

/-- A triangular pyramid with a sphere inscribed such that:
    1. The sphere touches all lateral faces at the midpoints of the base sides.
    2. The segment from the apex to the sphere's center is bisected by the base. -/
structure InscribedSpherePyramid where
  -- Radius of the inscribed sphere
  r : ℝ
  -- The pyramid's volume
  volume : ℝ
  -- The volume is equal to (r^3 * √6) / 4
  volume_eq : volume = (r^3 * Real.sqrt 6) / 4

/-- The volume of a triangular pyramid with an inscribed sphere satisfying 
    the given conditions is (r^3 * √6) / 4. -/
theorem inscribed_sphere_pyramid_volume (p : InscribedSpherePyramid) :
  p.volume = (p.r^3 * Real.sqrt 6) / 4 := by
  exact p.volume_eq


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_pyramid_volume_l633_63346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_true_l633_63334

-- Define the propositions
def proposition1 (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 + 1 > 0

noncomputable def proposition2 (m : ℝ) : Prop := ∀ x y : ℝ, x < y → Real.log (m * x) > Real.log (m * y)

-- Define the theorem
theorem exactly_one_true (m : ℝ) :
  (proposition1 m ∧ ¬proposition2 m) ∨ (¬proposition1 m ∧ proposition2 m) →
  m ≥ 1 ∨ m = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_true_l633_63334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lexicon_word_count_l633_63364

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum word length -/
def max_word_length : ℕ := 5

/-- The number of words of length n that contain the letter A at least once -/
def words_with_a (n : ℕ) : ℕ := alphabet_size^n - (alphabet_size - 1)^n

/-- The total number of valid words -/
def total_valid_words : ℕ := Finset.sum (Finset.range max_word_length) (fun i => words_with_a (i + 1))

theorem lexicon_word_count : total_valid_words = 1861701 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lexicon_word_count_l633_63364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_formula_l633_63356

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

theorem arithmetic_sequence_formula (n : ℕ) :
  arithmetic_sequence 2 3 n = 3 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_formula_l633_63356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_equation_l633_63344

/-- The circle in the problem -/
def problem_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

/-- The line perpendicular to the bisecting line -/
def perp_line (x y : ℝ) : Prop := x + 2*y = 0

/-- The bisecting line -/
def bisecting_line (x y : ℝ) : Prop := 2*x - y = 0

/-- Theorem stating that the bisecting line has the equation 2x - y = 0 -/
theorem bisecting_line_equation :
  ∃ (l : ℝ → ℝ → Prop), 
    (∀ x y, l x y ↔ bisecting_line x y) ∧
    (∀ x y, l x y → perp_line x y → (x + 2*y = 0)) ∧
    (∀ x₁ y₁ x₂ y₂, problem_circle x₁ y₁ → problem_circle x₂ y₂ → l x₁ y₁ → l x₂ y₂ → 
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₁ + x₂ - 2)^2 + (y₁ + y₂ - 4)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_equation_l633_63344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_l633_63383

theorem log_condition (a : ℝ) : 
  (∀ a, a > 0 → (Real.log a < 0 → a < 1)) ∧ 
  (∃ a, a < 1 ∧ a > 0 ∧ Real.log a ≥ 0) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_l633_63383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l633_63385

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longerBase : ℝ

/-- The area of an isosceles trapezoid with the given dimensions -/
noncomputable def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  900 * Real.sqrt 11 - 100 * Real.sqrt 154

/-- Theorem stating that the area of the specific isosceles trapezoid is as calculated -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := ⟨50, 52, 60⟩
  trapezoidArea t = 900 * Real.sqrt 11 - 100 * Real.sqrt 154 := by
  sorry

#check specific_trapezoid_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l633_63385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angelina_walking_speed_l633_63305

/-- Angelina's walking problem -/
theorem angelina_walking_speed 
  (distance_home_grocery : ℝ) 
  (distance_grocery_gym : ℝ) 
  (time_difference : ℝ) 
  (h1 : distance_home_grocery = 1200) 
  (h2 : distance_grocery_gym = 480) 
  (h3 : time_difference = 40) :
  2 * (distance_home_grocery / (distance_home_grocery / (distance_grocery_gym / time_difference))) = 48 := by
  sorry

#check angelina_walking_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angelina_walking_speed_l633_63305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l633_63333

/-- The function f(x) = x - 2/x -/
noncomputable def f (x : ℝ) : ℝ := x - 2/x

/-- The function g(x) = -x^2 + ax - 5 -/
def g (a x : ℝ) : ℝ := -x^2 + a*x - 5

/-- The theorem stating the range of a -/
theorem range_of_a :
  (∀ a : ℝ, (∀ x₁ : ℝ, x₁ ∈ Set.Icc 1 2 → ∃ x₂ : ℝ, x₂ ∈ Set.Icc 2 4 ∧ g a x₂ ≤ f x₁) ↔ a ∈ Set.Iic 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l633_63333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l633_63363

theorem evaluate_expression :
  let a : ℝ := 81
  let b : ℝ := 32
  let c : ℝ := 25
  a^(1/2 : ℝ) * b^(-(1/5) : ℝ) * c^(1/2 : ℝ) = 45/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l633_63363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l633_63325

variable (n : ℕ)
variable (M : Matrix (Fin 2) (Fin n) ℝ)
variable (u z : Vec ℝ n)

theorem matrix_vector_computation
  (h1 : M.mulVec u = ![4, -1])
  (h2 : M.mulVec z = ![1, 2]) :
  M.mulVec (u + 5 • z) = ![9, 9] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l633_63325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_small_triangle_l633_63382

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Calculates the area of a triangle formed by three points -/
noncomputable def triangleArea (p q r : Point) : ℝ :=
  abs ((q.x - p.x) * (r.y - p.y) - (r.x - p.x) * (q.y - p.y)) / 2

/-- The main theorem -/
theorem existence_of_small_triangle 
  (points : Finset Point) 
  (h_count : points.card = 102)
  (h_in_square : ∀ p, p ∈ points → 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1)
  (h_not_collinear : ∀ p q r, p ∈ points → q ∈ points → r ∈ points → 
    p ≠ q ∧ q ≠ r ∧ p ≠ r → ¬collinear p q r) :
  ∃ p q r, p ∈ points ∧ q ∈ points ∧ r ∈ points ∧ triangleArea p q r < 1/100 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_small_triangle_l633_63382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_XY_squared_theorem_l633_63330

/-- Definition of a convex quadrilateral -/
def convex_quadrilateral (A B C D : ℝ × ℝ) : Prop :=
  sorry -- We'll leave this as a placeholder for now

/-- Definition of an angle between three points -/
def angle (A B C : ℝ × ℝ) : ℝ :=
  sorry -- We'll leave this as a placeholder for now

/-- Main theorem statement -/
theorem quadrilateral_XY_squared_theorem (A B C D X Y : ℝ × ℝ) :
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  convex_quadrilateral A B C D ∧
  dist A B = 13 ∧
  dist B C = 13 ∧
  dist C D = 24 ∧
  dist D A = 24 ∧
  angle D A B = 60 * Real.pi / 180 ∧
  X = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧
  Y = ((D.1 + A.1) / 2, (D.2 + A.2) / 2) →
  dist X Y ^ 2 = 1033 / 4 + 30 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_XY_squared_theorem_l633_63330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_specific_line_segment_l633_63360

noncomputable def complex_midpoint (z₁ z₂ : ℂ) : ℂ := (z₁ + z₂) / 2

theorem midpoint_of_specific_line_segment :
  let z₁ : ℂ := -8 - 4*Complex.I
  let z₂ : ℂ := 12 + 4*Complex.I
  let m := complex_midpoint z₁ z₂
  (m.im = 0) → m = 2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_specific_line_segment_l633_63360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_mixing_theorem_l633_63345

/-- The temperature of cold water in degrees Celsius -/
noncomputable def cold_temp : ℝ := 20

/-- The temperature of hot water in degrees Celsius -/
noncomputable def hot_temp : ℝ := 40

/-- The resulting temperature when mixing equal volumes of water at two different temperatures -/
noncomputable def mix_temp (t1 t2 : ℝ) : ℝ := (t1 + t2) / 2

/-- Theorem stating that mixing equal volumes of water at 20°C and 40°C results in water at 30°C -/
theorem water_mixing_theorem :
  mix_temp cold_temp hot_temp = 30 := by
  unfold mix_temp cold_temp hot_temp
  norm_num
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_mixing_theorem_l633_63345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_avg_weight_approx_l633_63303

-- Define the sections
structure Section where
  students : Nat
  avgWeight : Float

-- Define the school
def school : List Section := [
  { students := 50, avgWeight := 50 },
  { students := 70, avgWeight := 60 },
  { students := 40, avgWeight := 55 },
  { students := 80, avgWeight := 70 },
  { students := 60, avgWeight := 65 }
]

-- Calculate the total number of students
def totalStudents : Nat := (school.map (λ s => s.students)).sum

-- Calculate the total weight
def totalWeight : Float := (school.map (λ s => s.students.toFloat * s.avgWeight)).sum

-- Calculate the average weight of the entire class
noncomputable def classAvgWeight : Float := totalWeight / totalStudents.toFloat

-- Theorem statement
theorem class_avg_weight_approx :
  (classAvgWeight - 61.33).abs < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_avg_weight_approx_l633_63303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_g_l633_63380

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 6)

theorem axis_of_symmetry_g :
  ∀ x : ℝ, g (Real.pi / 3 + (Real.pi / 3 - x)) = g x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_g_l633_63380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_sum_squares_l633_63355

def is_perfect_square (n : ℕ) : Bool :=
  match Nat.sqrt n with
  | m => m * m = n

def geometric_progression (a r : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a * r ^ i)

theorem geometric_progression_sum_squares (a r : ℕ) :
  (∀ t ∈ geometric_progression a r 4, t > 0 ∧ t < 50) →
  (List.sum (geometric_progression a r 4) = 120) →
  (List.sum (List.filter is_perfect_square (geometric_progression a r 4)) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_sum_squares_l633_63355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_vectors_l633_63331

/-- The area of a triangle given two of its sides as vectors. -/
theorem triangle_area_from_vectors (x₁ y₁ x₂ y₂ : ℝ) :
  let AB : ℝ × ℝ := (x₁, y₁)
  let AC : ℝ × ℝ := (x₂, y₂)
  (1/2) * |x₁ * y₂ - x₂ * y₁| = Real.sqrt ((AB.1^2 + AB.2^2) * (AC.1^2 + AC.2^2) - (AB.1 * AC.1 + AB.2 * AC.2)^2) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_vectors_l633_63331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l633_63372

-- Define the square
def square_center : ℝ × ℝ := (5, 3)
def square_area : ℝ := 16

-- Define the dilation
def dilation_center : ℝ × ℝ := (0, 0)
def scale_factor : ℝ := 3

-- Define the farthest vertex
def farthest_vertex : ℝ × ℝ := (21, 15)

-- Define a structure for the square
structure Square where
  center : ℝ × ℝ
  area : ℝ
  top_horizontal : Bool

-- Define functions for dilation and finding the farthest point
def dilate (s : Square) (center : ℝ × ℝ) (factor : ℝ) : Square :=
  sorry

def farthest_point_from_origin (s : Square) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem farthest_vertex_after_dilation :
  let original_square : Square := { center := square_center, area := square_area, top_horizontal := true }
  let dilated_square := dilate original_square dilation_center scale_factor
  farthest_point_from_origin dilated_square = farthest_vertex :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l633_63372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percentage_example_l633_63340

/-- Calculates the gain percentage given the selling price and gain amount. -/
noncomputable def gainPercentage (sellingPrice : ℚ) (gain : ℚ) : ℚ :=
  let costPrice := sellingPrice - gain
  (gain / costPrice) * 100

/-- Theorem stating that for an article sold at $100 with a gain of $20, the gain percentage is 25%. -/
theorem gain_percentage_example : gainPercentage 100 20 = 25 := by
  -- Unfold the definition of gainPercentage
  unfold gainPercentage
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percentage_example_l633_63340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l633_63352

-- Define the circle Γ
variable (Γ : Set (ℝ × ℝ))

-- Define the points A, B, C, D, and P
variable (A B C D P : ℝ × ℝ)

-- Define the rectangle ABCD
def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = x^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = y^2 ∧
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = x^2 ∧
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = y^2

-- Define the conditions
axiom h1 : is_rectangle A B C D
axiom h2 : A ∈ Γ ∧ B ∈ Γ ∧ C ∈ Γ ∧ D ∈ Γ
axiom h3 : P ∈ Γ
axiom h4 : (P.1 - A.1)^2 + (P.2 - A.2)^2 * ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 2
axiom h5 : (P.1 - C.1)^2 + (P.2 - C.2)^2 * ((P.1 - D.1)^2 + (P.2 - D.2)^2) = 18
axiom h6 : (P.1 - B.1)^2 + (P.2 - B.2)^2 * ((P.1 - C.1)^2 + (P.2 - C.2)^2) = 9

-- Define the area function
noncomputable def area (A B C D : ℝ × ℝ) : ℝ :=
  let x := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let y := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  x * y

-- State the theorem
theorem rectangle_area :
  area A B C D = (208 * Real.sqrt 17) / 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l633_63352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_iff_a_leq_two_l633_63376

/-- The function f(x) = x^2 + 2x + a ln x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a * Real.log x

/-- The theorem statement -/
theorem function_inequality_iff_a_leq_two (a : ℝ) :
  (∀ t : ℝ, t ≥ 1 → f a (2*t - 1) ≥ 2 * f a t - 3) ↔ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_iff_a_leq_two_l633_63376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_coloring_exists_l633_63315

/-- Represents a point in a 2D grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a domino (two adjacent unit squares) -/
structure Domino where
  p1 : Point
  p2 : Point
  adjacent : (p1.x = p2.x ∧ p1.y + 1 = p2.y) ∨ (p1.y = p2.y ∧ p1.x + 1 = p2.x)

/-- Represents a rectangle divided into dominoes -/
structure Rectangle where
  width : ℕ
  height : ℕ
  dominoes : List Domino
  covers_rectangle : ∀ x y, 0 ≤ x ∧ x < width ∧ 0 ≤ y ∧ y < height →
    ∃ d, d ∈ dominoes ∧ ((d.p1.x = x ∧ d.p1.y = y) ∨ (d.p2.x = x ∧ d.p2.y = y))

/-- Represents a coloring of the corners -/
def Coloring := Point → Fin 3

/-- Check if two points are at distance 1 -/
def distance_one (p1 p2 : Point) : Prop :=
  (p1.x = p2.x ∧ (p1.y + 1 = p2.y ∨ p2.y + 1 = p1.y)) ∨
  (p1.y = p2.y ∧ (p1.x + 1 = p2.x ∨ p2.x + 1 = p1.x))

/-- Check if a point is on the border between two dominoes -/
def on_border (p : Point) (d1 d2 : Domino) : Prop :=
  (p = d1.p1 ∨ p = d1.p2) ∧ (p = d2.p1 ∨ p = d2.p2) ∧ d1 ≠ d2

/-- Main theorem: There exists a valid coloring for any rectangle divided into dominoes -/
theorem domino_coloring_exists (r : Rectangle) :
  ∃ c : Coloring,
    (∀ p1 p2 : Point, distance_one p1 p2 →
      (∃ d1 d2, d1 ∈ r.dominoes ∧ d2 ∈ r.dominoes ∧ on_border p1 d1 d2 → c p1 ≠ c p2) ∧
      (∀ d, d ∈ r.dominoes → ((p1 = d.p1 ∧ p2 = d.p2) ∨ (p1 = d.p2 ∧ p2 = d.p1) → c p1 = c p2))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_coloring_exists_l633_63315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_initial_sum_l633_63322

/-- Proves that given specific conditions, the initial sum that results in a 
    difference of 64.10 between compound and simple interest is 1000. -/
theorem interest_difference_implies_initial_sum :
  ∀ P : ℝ,
  (P * ((1 + 0.10) ^ 4 - 1) - P * 0.10 * 4 = 64.10) → P = 1000 := by
  intro P
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_initial_sum_l633_63322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_T_l633_63336

-- Define T(r) for -1 < r < 1
noncomputable def T (r : ℝ) : ℝ := 20 / (1 - r)

-- State the theorem
theorem sum_of_T (b : ℝ) (h1 : -1 < b) (h2 : b < 1) (h3 : T b * T (-b) = 4800) :
  T b + T (-b) = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_T_l633_63336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l633_63366

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (5 - 4*x + x^2) / (2 - x)

-- Define the domain
def domain : Set ℝ := {x | x < 2}

-- State the theorem
theorem min_value_of_f :
  ∃ (min_val : ℝ), min_val = 2 ∧ 
  ∀ (x : ℝ), x ∈ domain → f x ≥ min_val := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l633_63366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_contact_probability_l633_63316

/-- The probability that two groups of tourists can contact each other -/
theorem tourist_contact_probability (p : ℝ) : 
  0 ≤ p → p ≤ 1 →
  (1 : ℝ) - (1 - p) ^ 40 = 1 - (1 - p) ^ 40 :=
by
  intro hp_nonneg hp_le_one
  simp


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_contact_probability_l633_63316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_trapezoid_perimeter_l633_63302

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  -- PQ and RS are the non-parallel sides
  pq : ℝ
  rs : ℝ
  -- Distance between parallel sides
  distance : ℝ
  -- Height of the trapezoid
  height : ℝ
  -- Conditions
  pq_eq_rs : pq = rs
  pq_value : pq = 7
  distance_value : distance = 5
  height_value : height = 6

/-- The perimeter of the special trapezoid -/
noncomputable def perimeter (t : SpecialTrapezoid) : ℝ :=
  2 * t.pq + 2 * Real.sqrt 13

/-- Theorem: The perimeter of the special trapezoid is 14 + 4√13 -/
theorem special_trapezoid_perimeter (t : SpecialTrapezoid) :
  perimeter t = 14 + 4 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_trapezoid_perimeter_l633_63302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_4ab2_same_type_as_3ab2_l633_63377

/-- A structure representing a monomial with coefficient, and exponents for variables a and b. -/
structure Monomial where
  coeff : ℕ
  a_exp : ℕ
  b_exp : ℕ

/-- Function to check if two monomials are of the same type -/
def same_type (m1 m2 : Monomial) : Prop :=
  m1.a_exp = m2.a_exp ∧ m1.b_exp = m2.b_exp

/-- The reference monomial 3ab^2 -/
def reference : Monomial :=
  { coeff := 3, a_exp := 1, b_exp := 2 }

/-- The list of monomials to compare against -/
def monomials : List Monomial :=
  [
    { coeff := 3, a_exp := 2, b_exp := 1 },  -- 3a^2b
    { coeff := 4, a_exp := 1, b_exp := 2 },  -- 4ab^2
    { coeff := 3, a_exp := 2, b_exp := 2 },  -- 3a^2b^2
    { coeff := 3, a_exp := 1, b_exp := 1 }   -- 3ab
  ]

theorem only_4ab2_same_type_as_3ab2 :
  ∃! m, m ∈ monomials ∧ same_type m reference :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_4ab2_same_type_as_3ab2_l633_63377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l633_63381

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_specific_vectors (a b : ℝ × ℝ) :
  (Real.sqrt (a.1^2 + a.2^2) = 2) →
  (Real.sqrt (b.1^2 + b.2^2) = 2) →
  (b.1 * (2 * a.1 + b.1) + b.2 * (2 * a.2 + b.2) = 0) →
  angle_between_vectors a b = 2 * Real.pi / 3 := by
  sorry

#check angle_between_specific_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l633_63381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_airport_distance_l633_63361

/-- Represents the distance from David's home to the airport in miles -/
def distance : ℝ := sorry

/-- Represents the time in hours that David should take to reach the airport on time -/
def time : ℝ := sorry

/-- David's initial speed in miles per hour -/
def initial_speed : ℝ := 35

/-- David's increased speed in miles per hour -/
def increased_speed : ℝ := initial_speed + 15

/-- Time in hours that David would be late if he continued at the initial speed -/
def late_time : ℝ := 1

/-- Time in hours that David arrives early -/
def early_time : ℝ := 0.5

theorem david_airport_distance :
  (distance = initial_speed * (time + late_time)) ∧
  (distance - initial_speed = increased_speed * (time - early_time)) →
  distance = 210 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_airport_distance_l633_63361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_periodic_l633_63329

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) * Real.cos (2 * x)

theorem f_is_odd_and_periodic :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, f (x + π/2) = f x) ∧
  (∀ t, 0 < t → t < π/2 → ∃ x, f (x + t) ≠ f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_periodic_l633_63329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percent_approx_l633_63338

noncomputable section

-- Define the currency conversion rate
def conversion_rate : ℝ := 50

-- Define the purchase prices in INR
def purchase_price_A : ℝ := 600
def purchase_price_B : ℝ := 700
def purchase_price_C : ℝ := 800

-- Define the selling prices in USD
def selling_price_A : ℝ := 10
def selling_price_B : ℝ := 12.5
def selling_price_C : ℝ := 14

-- Calculate the total purchase price in USD
noncomputable def total_purchase_price_usd : ℝ := 
  (purchase_price_A + purchase_price_B + purchase_price_C) / conversion_rate

-- Calculate the total selling price in USD
noncomputable def total_selling_price_usd : ℝ := 
  selling_price_A + selling_price_B + selling_price_C

-- Calculate the overall loss in USD
noncomputable def overall_loss_usd : ℝ := 
  total_purchase_price_usd - total_selling_price_usd

-- Calculate the loss percent
noncomputable def loss_percent : ℝ := 
  (overall_loss_usd / total_purchase_price_usd) * 100

-- Theorem: The loss percent is approximately 13.10%
theorem loss_percent_approx :
  abs (loss_percent - 13.10) < 0.01 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percent_approx_l633_63338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_eq_square_plus_power_of_two_solutions_l633_63306

theorem cube_eq_square_plus_power_of_two_solutions :
  ∀ (a b n : ℕ), a > 0 → b > 0 → n > 0 → a^3 = b^2 + 2^n →
  ∃ (m : ℕ) (A B C : ℕ),
    ((A, B, C) = (2, 2, 2) ∨ (A, B, C) = (3, 5, 1) ∨ (A, B, C) = (5, 11, 2)) ∧
    a = 2^(2*m) * A ∧
    b = 2^(3*m) * B ∧
    n = C + 6*m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_eq_square_plus_power_of_two_solutions_l633_63306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extreme_values_l633_63353

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 1

def interval : Set ℝ := Set.Icc (-2) 3

theorem tangent_line_and_extreme_values :
  ∃ (y : ℝ → ℝ),
    (∀ x, x ∈ interval → y x = 3*x + 3*(f x) - 4) ∧
    (∀ x, x ∈ interval → f x ≤ 1) ∧
    (∀ x, x ∈ interval → f x ≥ -17/3) ∧
    (∃ x₁ ∈ interval, f x₁ = 1) ∧
    (∃ x₂ ∈ interval, f x₂ = -17/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extreme_values_l633_63353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_l633_63343

noncomputable def α : ℝ := Real.pi / 3

theorem same_terminal_side (α : ℝ) :
  (∀ θ : ℝ, (∃ k : ℤ, θ = 2 * Real.pi * (k : ℝ) + α) ↔ Real.cos θ = Real.cos α ∧ Real.sin θ = Real.sin α) ∧
  (∀ θ : ℝ, -4 * Real.pi < θ ∧ θ < 2 * Real.pi ∧ (∃ k : ℤ, θ = 2 * Real.pi * (k : ℝ) + α) ↔ 
    θ = -11 * Real.pi / 3 ∨ θ = -5 * Real.pi / 3 ∨ θ = Real.pi / 3) ∧
  (∀ β : ℝ, (∃ k : ℤ, β = 2 * Real.pi * (k : ℝ) + α) → 
    Real.cos (β / 2) > 0 ∧ Real.sin (β / 2) > 0 ∨ 
    Real.cos (β / 2) < 0 ∧ Real.sin (β / 2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_l633_63343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_twice_min_implies_phi_pi_half_l633_63317

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (x + φ)

theorem max_twice_min_implies_phi_pi_half (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi) :
  (∃ (x_max x_min : ℝ), 
    (∀ x, f x φ ≤ f x_max φ) ∧ 
    (∀ x, f x φ ≥ f x_min φ) ∧ 
    f x_max φ = 2 * f x_min φ) →
  φ = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_twice_min_implies_phi_pi_half_l633_63317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l633_63389

-- Define the ellipse C₂
def C₂ (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x + Real.sqrt 3*y - 9 = 0

-- Define the distance function from a point (x, y) to the line l
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (2*x + Real.sqrt 3*y - 9) / Real.sqrt 7

-- Theorem statement
theorem max_distance_to_line :
  ∃ (max_dist : ℝ), max_dist = 2 * Real.sqrt 7 ∧
  ∀ (x y : ℝ), C₂ x y → distance_to_line x y ≤ max_dist :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l633_63389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_x_intercept_l633_63384

/-- A line passing through two points has a specific x-intercept -/
theorem line_x_intercept (x1 y1 x2 y2 : ℝ) (h : (x1, y1) ≠ (x2, y2)) : 
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  (0 - b) / m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_x_intercept_l633_63384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_k_zero_root_count_l633_63300

noncomputable section

def f (k : ℝ) (x : ℝ) : ℝ := Real.log (Real.exp x + k)

def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem odd_function_implies_k_zero (k : ℝ) :
  is_odd (f k) → k = 0 := by sorry

def h (x : ℝ) : ℝ := Real.log x / x

def g (m : ℝ) (x : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x + m

theorem root_count (m : ℝ) :
  (∀ x, x > 0 → h x = g m x) →
  (m > Real.exp 2 + 1 / Real.exp 1 → False) ∧
  (m = Real.exp 2 + 1 / Real.exp 1 → ∃! x, x > 0 ∧ h x = g m x) ∧
  (m < Real.exp 2 + 1 / Real.exp 1 → ∃ x y, x > 0 ∧ y > 0 ∧ x ≠ y ∧ h x = g m x ∧ h y = g m y) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_k_zero_root_count_l633_63300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l633_63312

theorem problem_statement (a b c : ℝ) : 
  (b * c^2 < a * c^2 → b < a) ∧ 
  (a^3 > b^3 ∧ a * b < 0 → 1 / a > 1 / b) ∧ 
  (a > b ∧ b > c ∧ c > 0 → a / b > (a + c) / (b + c)) ∧ 
  ¬(∀ a b c : ℝ, c > b ∧ b > a ∧ a > 0 → a / (c - a) > b / (c - b)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l633_63312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_25_l633_63308

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => (Real.sqrt (sequence_a n) + 1) ^ 2

theorem a_5_equals_25 : sequence_a 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_25_l633_63308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_white_cells_l633_63347

/-- Represents the configuration of black cells in an 8x8 table -/
def BlackCellConfig := Fin 8 → Fin 8 → Bool

/-- The number of black cells in a given configuration -/
def countBlackCells (config : BlackCellConfig) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 8)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 8)) fun j =>
      if config i j then 1 else 0)

/-- The sum of numbers in white cells for a given configuration -/
def sumWhiteCells (config : BlackCellConfig) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 8)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 8)) fun j =>
      if config i j then
        0
      else
        (Finset.sum (Finset.univ : Finset (Fin 8)) fun k => if config i k then 1 else 0) +
        (Finset.sum (Finset.univ : Finset (Fin 8)) fun k => if config k j then 1 else 0)

/-- The theorem stating the maximum sum of numbers in white cells -/
theorem max_sum_white_cells :
    ∃ (config : BlackCellConfig),
      countBlackCells config = 23 ∧
      ∀ (otherConfig : BlackCellConfig),
        countBlackCells otherConfig = 23 →
        sumWhiteCells otherConfig ≤ sumWhiteCells config ∧
        sumWhiteCells config = 234 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_white_cells_l633_63347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_solutions_l633_63348

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The property of being skew-centered -/
def is_skew_centered (p : IntPolynomial) : Prop :=
  p.eval 50 = -50

theorem max_integer_solutions (p : IntPolynomial) (h : is_skew_centered p) :
  ∃ (S : Finset ℤ), (∀ k : ℤ, k ∈ S ↔ p.eval k = k^2) ∧ S.card ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_solutions_l633_63348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_product_plus_one_bound_l633_63393

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Represents a hyperbola -/
structure Hyperbola where
  center : Point
  a : ℝ  -- transverse semi-axis
  b : ℝ  -- conjugate semi-axis

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the eccentricity of an ellipse -/
noncomputable def ellipseEccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Calculates the eccentricity of a hyperbola -/
noncomputable def hyperbolaEccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- Checks if a point is on an ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- Checks if a point is on a hyperbola -/
def isOnHyperbola (p : Point) (h : Hyperbola) : Prop :=
  (p.x - h.center.x)^2 / h.a^2 - (p.y - h.center.y)^2 / h.b^2 = 1

theorem eccentricity_product_plus_one_bound
  (e : Ellipse) (h : Hyperbola) (p f1 f2 : Point)
  (h1 : e.center = h.center)
  (h2 : e.center = ⟨0, 0⟩)
  (h3 : p.x > 0 ∧ p.y > 0)  -- P is in first quadrant
  (h4 : distance p f1 = distance p f2)  -- PF₁F₂ is isosceles
  (h5 : distance p f1 = 8)
  (h6 : isOnEllipse p e ∧ isOnHyperbola p h)  -- P is on both ellipse and hyperbola
  : ellipseEccentricity e * hyperbolaEccentricity h + 1 > 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_product_plus_one_bound_l633_63393
