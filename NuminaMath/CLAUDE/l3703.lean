import Mathlib

namespace NUMINAMATH_CALUDE_fraction_multiple_l3703_370387

theorem fraction_multiple (numerator denominator : ℕ) : 
  denominator = 5 →
  numerator = denominator + 4 →
  (numerator + 6) / denominator = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_multiple_l3703_370387


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3703_370366

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 4 + 3 * Real.sqrt 2 ∧ 
  x₂ = 4 - 3 * Real.sqrt 2 ∧ 
  x₁^2 - 8*x₁ - 2 = 0 ∧ 
  x₂^2 - 8*x₂ - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3703_370366


namespace NUMINAMATH_CALUDE_walnut_problem_l3703_370346

theorem walnut_problem (a b c : ℕ) : 
  28 * a + 30 * b + 31 * c = 365 → a + b + c = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_walnut_problem_l3703_370346


namespace NUMINAMATH_CALUDE_probability_edge_endpoints_is_correct_l3703_370301

structure RegularIcosahedron where
  vertices : Finset (Fin 12)
  edges : Finset (Fin 12 × Fin 12)
  vertex_degree : ∀ v : Fin 12, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 5

def probability_edge_endpoints (I : RegularIcosahedron) : ℚ :=
  5 / 11

theorem probability_edge_endpoints_is_correct (I : RegularIcosahedron) :
  probability_edge_endpoints I = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_edge_endpoints_is_correct_l3703_370301


namespace NUMINAMATH_CALUDE_f_range_on_interval_l3703_370384

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.exp x * (Real.sin x + Real.cos x)

theorem f_range_on_interval :
  let a := 0
  let b := Real.pi / 2
  ∃ (min max : ℝ), 
    (∀ x ∈ Set.Icc a b, f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc a b, f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc a b, f x₂ = max) ∧
    min = 1/2 ∧
    max = (1/2) * Real.exp (Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_f_range_on_interval_l3703_370384


namespace NUMINAMATH_CALUDE_integral_odd_function_is_zero_l3703_370326

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The definite integral of an odd function from -1 to 1 is zero -/
theorem integral_odd_function_is_zero (f : ℝ → ℝ) (hf : IsOdd f) :
    ∫ x in (-1)..1, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_integral_odd_function_is_zero_l3703_370326


namespace NUMINAMATH_CALUDE_f_derivative_at_1_l3703_370382

-- Define the function f
def f (x : ℝ) : ℝ := (2023 - 2022 * x) ^ 3

-- State the theorem
theorem f_derivative_at_1 : 
  (deriv f) 1 = -6066 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_1_l3703_370382


namespace NUMINAMATH_CALUDE_grid_sum_theorem_l3703_370383

/-- A 3x3 grid represented as a function from (Fin 3 × Fin 3) to ℕ -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- The sum of numbers on the main diagonal of the grid -/
def mainDiagonalSum (g : Grid) : ℕ :=
  g 0 0 + g 1 1 + g 2 2

/-- The sum of numbers on the other diagonal of the grid -/
def otherDiagonalSum (g : Grid) : ℕ :=
  g 0 2 + g 1 1 + g 2 0

/-- The sum of numbers not on either diagonal -/
def nonDiagonalSum (g : Grid) : ℕ :=
  g 0 1 + g 1 0 + g 1 2 + g 2 1 + g 1 1

/-- The theorem statement -/
theorem grid_sum_theorem (g : Grid) :
  (∀ i j, g i j ∈ Finset.range 10) →
  (mainDiagonalSum g = 7) →
  (otherDiagonalSum g = 21) →
  (nonDiagonalSum g = 25) := by
  sorry

end NUMINAMATH_CALUDE_grid_sum_theorem_l3703_370383


namespace NUMINAMATH_CALUDE_triangle_shape_l3703_370353

theorem triangle_shape (A : Real) (h1 : 0 < A ∧ A < π) 
  (h2 : Real.sin A + Real.cos A = 7/12) : 
  ∃ (B C : Real), 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π ∧ π/2 < A := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l3703_370353


namespace NUMINAMATH_CALUDE_function_is_even_l3703_370345

/-- A function satisfying the given functional equation -/
class FunctionalEquation (f : ℝ → ℝ) : Prop where
  eq : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b
  not_zero : ∃ x : ℝ, f x ≠ 0

/-- The main theorem: if f satisfies the functional equation, then it is even -/
theorem function_is_even (f : ℝ → ℝ) [FunctionalEquation f] : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_is_even_l3703_370345


namespace NUMINAMATH_CALUDE_cone_vertex_angle_l3703_370373

theorem cone_vertex_angle (α β αf : Real) : 
  β = 2 * Real.arcsin (1/4) →
  2 * α = αf →
  2 * α = Real.pi/6 + Real.arcsin (1/4) :=
by sorry

end NUMINAMATH_CALUDE_cone_vertex_angle_l3703_370373


namespace NUMINAMATH_CALUDE_remainder_sum_l3703_370397

theorem remainder_sum (n : ℤ) : n % 20 = 14 → (n % 4 + n % 5 = 6) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3703_370397


namespace NUMINAMATH_CALUDE_inequality_proof_l3703_370374

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c ≤ 3) :
  a / (1 + a^2) + b / (1 + b^2) + c / (1 + c^2) ≤ 3/2 ∧ 3/2 ≤ 1/(1+a) + 1/(1+b) + 1/(1+c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3703_370374


namespace NUMINAMATH_CALUDE_geometric_series_product_l3703_370315

theorem geometric_series_product (y : ℝ) : 
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/y)^n → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_product_l3703_370315


namespace NUMINAMATH_CALUDE_box_neg_two_two_neg_one_l3703_370351

def box (a b c : ℤ) : ℚ := (a ^ b : ℚ) - (b ^ c : ℚ) + (c ^ a : ℚ)

theorem box_neg_two_two_neg_one : box (-2) 2 (-1) = (7 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_box_neg_two_two_neg_one_l3703_370351


namespace NUMINAMATH_CALUDE_reciprocal_of_point_B_is_one_l3703_370388

-- Define the position of point A on the number line
def point_A : ℝ := -3

-- Define the distance between point A and point B
def distance_AB : ℝ := 4

-- Define the position of point B on the number line
def point_B : ℝ := point_A + distance_AB

-- Theorem to prove
theorem reciprocal_of_point_B_is_one : 
  (1 : ℝ) / point_B = 1 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_point_B_is_one_l3703_370388


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l3703_370376

theorem unique_square_divisible_by_three_in_range : ∃! y : ℕ, 
  (∃ x : ℕ, y = x^2) ∧ 
  (∃ k : ℕ, y = 3 * k) ∧ 
  50 < y ∧ y < 120 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l3703_370376


namespace NUMINAMATH_CALUDE_cos_product_equals_one_l3703_370398

theorem cos_product_equals_one : 8 * Real.cos (4 * Real.pi / 9) * Real.cos (2 * Real.pi / 9) * Real.cos (Real.pi / 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_product_equals_one_l3703_370398


namespace NUMINAMATH_CALUDE_solve_for_m_l3703_370361

theorem solve_for_m (x m : ℝ) (h1 : 3 * x - 2 * m = 4) (h2 : x = 6) : m = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3703_370361


namespace NUMINAMATH_CALUDE_eight_power_ten_sum_equals_two_power_y_l3703_370377

theorem eight_power_ten_sum_equals_two_power_y (y : ℕ) :
  8^10 + 8^10 + 8^10 + 8^10 + 8^10 + 8^10 + 8^10 + 8^10 = 2^y → y = 33 := by
  sorry

end NUMINAMATH_CALUDE_eight_power_ten_sum_equals_two_power_y_l3703_370377


namespace NUMINAMATH_CALUDE_negative_root_range_l3703_370381

theorem negative_root_range (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (5 - a)) →
  -3 < a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_negative_root_range_l3703_370381


namespace NUMINAMATH_CALUDE_ln_inequality_condition_l3703_370308

theorem ln_inequality_condition (x : ℝ) :
  (∀ x, (Real.log x < 0 → x < 1)) ∧
  (∃ x, x < 1 ∧ Real.log x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ln_inequality_condition_l3703_370308


namespace NUMINAMATH_CALUDE_total_hotdogs_sold_l3703_370325

theorem total_hotdogs_sold (small_hotdogs large_hotdogs : ℕ) 
  (h1 : small_hotdogs = 58) 
  (h2 : large_hotdogs = 21) : 
  small_hotdogs + large_hotdogs = 79 := by
  sorry

end NUMINAMATH_CALUDE_total_hotdogs_sold_l3703_370325


namespace NUMINAMATH_CALUDE_field_trip_vans_l3703_370389

theorem field_trip_vans (buses : ℝ) (people_per_van : ℝ) (people_per_bus : ℝ) 
  (extra_people_in_buses : ℝ) :
  buses = 8 →
  people_per_van = 6 →
  people_per_bus = 18 →
  extra_people_in_buses = 108 →
  ∃ (vans : ℝ), vans = 6 ∧ people_per_bus * buses - people_per_van * vans = extra_people_in_buses :=
by
  sorry

end NUMINAMATH_CALUDE_field_trip_vans_l3703_370389


namespace NUMINAMATH_CALUDE_walter_bus_time_l3703_370391

def minutes_in_hour : ℕ := 60

def walter_schedule : Prop :=
  let wake_up_time : ℕ := 6 * 60 + 15
  let bus_departure_time : ℕ := 7 * 60
  let class_duration : ℕ := 45
  let num_classes : ℕ := 8
  let lunch_duration : ℕ := 30
  let break_duration : ℕ := 15
  let additional_time : ℕ := 2 * 60
  let return_home_time : ℕ := 16 * 60 + 30
  let total_away_time : ℕ := return_home_time - bus_departure_time
  let school_activities_time : ℕ := num_classes * class_duration + lunch_duration + break_duration + additional_time
  let bus_time : ℕ := total_away_time - school_activities_time
  bus_time = 45

theorem walter_bus_time : walter_schedule := by
  sorry

end NUMINAMATH_CALUDE_walter_bus_time_l3703_370391


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_specific_hyperbola_focal_length_l3703_370303

/-- The focal length of a hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1 is 2√(a^2 + b^2) -/
theorem hyperbola_focal_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let equation := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let focal_length := 2 * Real.sqrt (a^2 + b^2)
  equation 2 3 → focal_length = 2 * Real.sqrt 13 := by
  sorry

/-- The focal length of the hyperbola x^2/4 - y^2/9 = 1 is 2√13 -/
theorem specific_hyperbola_focal_length :
  let equation := fun (x y : ℝ) => x^2 / 4 - y^2 / 9 = 1
  let focal_length := 2 * Real.sqrt (4 + 9)
  equation 2 3 → focal_length = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_specific_hyperbola_focal_length_l3703_370303


namespace NUMINAMATH_CALUDE_titius_bode_ninth_planet_l3703_370380

/-- The Titius-Bode law formula -/
def titius_bode (a b : ℝ) (n : ℕ) : ℝ := a + b * 2^(n - 1)

/-- Theorem stating the Titius-Bode law for the 9th planet -/
theorem titius_bode_ninth_planet :
  ∃ (a b : ℝ),
    (titius_bode a b 1 = 0.7) ∧
    (titius_bode a b 2 = 1) ∧
    (titius_bode a b 9 = 77.2) := by
  sorry

end NUMINAMATH_CALUDE_titius_bode_ninth_planet_l3703_370380


namespace NUMINAMATH_CALUDE_fifteen_tomorrow_fishers_l3703_370370

/-- Represents the fishing pattern in the coastal village --/
structure FishingPattern where
  daily : Nat
  everyOtherDay : Nat
  everyThreeDay : Nat
  yesterday : Nat
  today : Nat

/-- Calculates the number of people fishing tomorrow given the fishing pattern --/
def tomorrowFishers (pattern : FishingPattern) : Nat :=
  pattern.daily + 
  (pattern.everyOtherDay - (pattern.yesterday - pattern.daily)) +
  pattern.everyThreeDay

/-- Theorem stating that given the specific fishing pattern, 15 people will fish tomorrow --/
theorem fifteen_tomorrow_fishers (pattern : FishingPattern) 
  (h1 : pattern.daily = 7)
  (h2 : pattern.everyOtherDay = 8)
  (h3 : pattern.everyThreeDay = 3)
  (h4 : pattern.yesterday = 12)
  (h5 : pattern.today = 10) :
  tomorrowFishers pattern = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_tomorrow_fishers_l3703_370370


namespace NUMINAMATH_CALUDE_possible_value_less_than_five_l3703_370302

theorem possible_value_less_than_five : ∃ x : ℝ, x < 5 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_possible_value_less_than_five_l3703_370302


namespace NUMINAMATH_CALUDE_existence_of_prime_arithmetic_progressions_l3703_370399

/-- An arithmetic progression of n terms starting with a and with common difference d -/
def arithmeticProgression (a : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a + i * d)

/-- Check if a list of natural numbers consists only of prime numbers -/
def allPrime (l : List ℕ) : Prop :=
  l.all Nat.Prime

theorem existence_of_prime_arithmetic_progressions :
  (∃ a d : ℕ, allPrime (arithmeticProgression a d 5)) ∧
  (∃ a d : ℕ, allPrime (arithmeticProgression a d 6)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_prime_arithmetic_progressions_l3703_370399


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3703_370359

/-- Given a sphere with an inscribed triangle on its section, prove its surface area --/
theorem sphere_surface_area (a b c : ℝ) (r R : ℝ) : 
  a = 6 → b = 8 → c = 10 →  -- Triangle side lengths
  r = 5 →  -- Radius of section's circle
  R^2 - (R/2)^2 = r^2 →  -- Relation between sphere radius and section
  4 * π * R^2 = 400 * π / 3 := by
  sorry

#check sphere_surface_area

end NUMINAMATH_CALUDE_sphere_surface_area_l3703_370359


namespace NUMINAMATH_CALUDE_triangle_equality_l3703_370312

theorem triangle_equality (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_equation : (c - b) / a + (a - c) / b + (b - a) / c = 0) :
  a = b ∨ b = c ∨ c = a :=
sorry

end NUMINAMATH_CALUDE_triangle_equality_l3703_370312


namespace NUMINAMATH_CALUDE_lucy_sales_l3703_370344

/-- Given the total number of packs sold and Robyn's sales, calculate Lucy's sales. -/
theorem lucy_sales (total : ℕ) (robyn : ℕ) (h1 : total = 98) (h2 : robyn = 55) :
  total - robyn = 43 := by
  sorry

end NUMINAMATH_CALUDE_lucy_sales_l3703_370344


namespace NUMINAMATH_CALUDE_program_output_l3703_370368

theorem program_output : ∃ i : ℕ, (∀ j < i, 2^j ≤ 2000) ∧ (2^i > 2000) ∧ (i - 1 = 10) := by
  sorry

end NUMINAMATH_CALUDE_program_output_l3703_370368


namespace NUMINAMATH_CALUDE_friends_at_reception_l3703_370375

/-- Calculates the number of friends attending a wedding reception --/
theorem friends_at_reception (total_guests : ℕ) (bride_couples : ℕ) (groom_couples : ℕ) : 
  total_guests - 2 * (bride_couples + groom_couples) = 100 :=
by
  sorry

#check friends_at_reception 180 20 20

end NUMINAMATH_CALUDE_friends_at_reception_l3703_370375


namespace NUMINAMATH_CALUDE_certain_number_problem_l3703_370365

theorem certain_number_problem : ∃ x : ℚ, (x * 30 + (12 + 8) * 3) / 5 = 1212 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3703_370365


namespace NUMINAMATH_CALUDE_cats_remaining_after_missions_l3703_370337

/-- The number of cats remaining on Tatoosh Island after two relocation missions -/
def cats_remaining (initial : ℕ) (first_relocation : ℕ) : ℕ :=
  let after_first := initial - first_relocation
  let second_relocation := after_first / 2
  after_first - second_relocation

/-- Theorem stating that 600 cats remain on the island after the relocation missions -/
theorem cats_remaining_after_missions :
  cats_remaining 1800 600 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_after_missions_l3703_370337


namespace NUMINAMATH_CALUDE_circle_circumference_irrational_l3703_370396

/-- The circumference of a circle with rational radius is irrational -/
theorem circle_circumference_irrational (r : ℚ) : 
  Irrational (2 * Real.pi * (r : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_circle_circumference_irrational_l3703_370396


namespace NUMINAMATH_CALUDE_parabola_c_range_l3703_370306

/-- Given a parabola y = x^2 + bx + c with axis of symmetry at x = 2,
    and the quadratic equation -x^2 - bx - c = 0 having two equal real roots
    within -1 < x < 3, prove that the range of c is -5 < c ≤ 3 or c = 4 -/
theorem parabola_c_range (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = (x - 2)^2 + k) →  -- axis of symmetry at x = 2
  (∃ x, -1 < x ∧ x < 3 ∧ -x^2 - b*x - c = 0) →  -- roots within -1 < x < 3
  (-5 < c ∧ c ≤ 3) ∨ c = 4 := by
  sorry


end NUMINAMATH_CALUDE_parabola_c_range_l3703_370306


namespace NUMINAMATH_CALUDE_pen_purchase_problem_l3703_370307

/-- The problem of calculating the total number of pens purchased --/
theorem pen_purchase_problem (price_x price_y total_spent : ℚ) (num_x : ℕ) : 
  price_x = 4 → 
  price_y = (14/5 : ℚ) → 
  total_spent = 40 → 
  num_x = 8 → 
  ∃ (num_y : ℕ), num_x * price_x + num_y * price_y = total_spent ∧ num_x + num_y = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_pen_purchase_problem_l3703_370307


namespace NUMINAMATH_CALUDE_earth_habitable_fraction_l3703_370327

theorem earth_habitable_fraction :
  (earth_land_fraction : ℚ) →
  (land_habitable_fraction : ℚ) →
  earth_land_fraction = 1/3 →
  land_habitable_fraction = 1/4 →
  earth_land_fraction * land_habitable_fraction = 1/12 :=
by sorry

end NUMINAMATH_CALUDE_earth_habitable_fraction_l3703_370327


namespace NUMINAMATH_CALUDE_classroom_puzzle_l3703_370367

theorem classroom_puzzle (initial_boys initial_girls : ℕ) : 
  initial_boys = initial_girls →
  initial_boys = 2 * (initial_girls - 8) →
  initial_boys + initial_girls = 32 := by
sorry

end NUMINAMATH_CALUDE_classroom_puzzle_l3703_370367


namespace NUMINAMATH_CALUDE_fraction_of_time_at_4kmh_l3703_370390

/-- Represents the walking scenario described in the problem -/
structure WalkScenario where
  totalTime : ℝ
  timeAt2kmh : ℝ
  timeAt3kmh : ℝ
  timeAt4kmh : ℝ
  distanceAt2kmh : ℝ
  distanceAt3kmh : ℝ
  distanceAt4kmh : ℝ

/-- Theorem stating the fraction of time walked at 4 km/h -/
theorem fraction_of_time_at_4kmh (w : WalkScenario) : 
  w.timeAt2kmh = w.totalTime / 2 →
  w.distanceAt3kmh = (w.distanceAt2kmh + w.distanceAt3kmh + w.distanceAt4kmh) / 2 →
  w.distanceAt2kmh = 2 * w.timeAt2kmh →
  w.distanceAt3kmh = 3 * w.timeAt3kmh →
  w.distanceAt4kmh = 4 * w.timeAt4kmh →
  w.totalTime = w.timeAt2kmh + w.timeAt3kmh + w.timeAt4kmh →
  w.timeAt4kmh / w.totalTime = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_time_at_4kmh_l3703_370390


namespace NUMINAMATH_CALUDE_single_burger_cost_l3703_370316

theorem single_burger_cost 
  (total_spent : ℚ)
  (total_burgers : ℕ)
  (double_burger_cost : ℚ)
  (double_burgers : ℕ)
  (h1 : total_spent = 66.5)
  (h2 : total_burgers = 50)
  (h3 : double_burger_cost = 1.5)
  (h4 : double_burgers = 33) :
  (total_spent - double_burger_cost * double_burgers) / (total_burgers - double_burgers) = 1 := by
sorry

end NUMINAMATH_CALUDE_single_burger_cost_l3703_370316


namespace NUMINAMATH_CALUDE_max_sum_expression_l3703_370322

def is_valid_set (s : Finset ℕ) : Prop :=
  s.card = 4 ∧ s ⊆ {1, 3, 5, 7}

def sum_expression (a b c d : ℕ) : ℕ :=
  a * b + b * c + c * d + d * a + a^2 + b^2 + c^2 + d^2

theorem max_sum_expression (s : Finset ℕ) (hs : is_valid_set s) :
  ∃ (a b c d : ℕ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∀ (w x y z : ℕ), w ∈ s → x ∈ s → y ∈ s → z ∈ s →
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
  sum_expression a b c d ≥ sum_expression w x y z ∧
  sum_expression a b c d = 201 :=
sorry

end NUMINAMATH_CALUDE_max_sum_expression_l3703_370322


namespace NUMINAMATH_CALUDE_election_votes_l3703_370331

theorem election_votes (votes_A : ℕ) (ratio_A ratio_B : ℕ) : 
  votes_A = 14 → ratio_A = 2 → ratio_B = 1 → 
  votes_A + (votes_A * ratio_B / ratio_A) = 21 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l3703_370331


namespace NUMINAMATH_CALUDE_reflections_count_l3703_370364

/-- Number of reflections Sarah sees in tall mirrors -/
def sarah_tall : ℕ := 10

/-- Number of reflections Sarah sees in wide mirrors -/
def sarah_wide : ℕ := 5

/-- Number of reflections Sarah sees in narrow mirrors -/
def sarah_narrow : ℕ := 8

/-- Number of reflections Ellie sees in tall mirrors -/
def ellie_tall : ℕ := 6

/-- Number of reflections Ellie sees in wide mirrors -/
def ellie_wide : ℕ := 3

/-- Number of reflections Ellie sees in narrow mirrors -/
def ellie_narrow : ℕ := 4

/-- Number of times they pass through tall mirrors -/
def times_tall : ℕ := 3

/-- Number of times they pass through wide mirrors -/
def times_wide : ℕ := 5

/-- Number of times they pass through narrow mirrors -/
def times_narrow : ℕ := 4

/-- The total number of reflections seen by Sarah and Ellie -/
def total_reflections : ℕ :=
  (sarah_tall * times_tall + sarah_wide * times_wide + sarah_narrow * times_narrow) +
  (ellie_tall * times_tall + ellie_wide * times_wide + ellie_narrow * times_narrow)

theorem reflections_count : total_reflections = 136 := by
  sorry

end NUMINAMATH_CALUDE_reflections_count_l3703_370364


namespace NUMINAMATH_CALUDE_double_age_in_two_years_l3703_370305

/-- The number of years it takes for a man's age to be twice his son's age -/
def yearsUntilDoubleAge (sonAge : ℕ) (ageDifference : ℕ) : ℕ :=
  sorry

/-- Theorem stating that it takes 2 years for the man's age to be twice his son's age -/
theorem double_age_in_two_years (sonAge : ℕ) (ageDifference : ℕ) 
  (h1 : sonAge = 24) 
  (h2 : ageDifference = 26) : 
  yearsUntilDoubleAge sonAge ageDifference = 2 :=
by sorry

end NUMINAMATH_CALUDE_double_age_in_two_years_l3703_370305


namespace NUMINAMATH_CALUDE_max_type_c_tubes_exists_solution_with_73_type_c_l3703_370314

/-- Represents the types of test tubes -/
inductive TubeType
  | A
  | B
  | C

/-- Represents a solution of test tubes -/
structure Solution where
  a : ℕ  -- number of type A tubes
  b : ℕ  -- number of type B tubes
  c : ℕ  -- number of type C tubes

/-- The concentration of the solution in each type of tube -/
def concentration : TubeType → ℚ
  | TubeType.A => 1/10
  | TubeType.B => 1/5
  | TubeType.C => 9/10

/-- The total number of tubes used -/
def Solution.total (s : Solution) : ℕ := s.a + s.b + s.c

/-- The average concentration of the final solution -/
def Solution.averageConcentration (s : Solution) : ℚ :=
  (s.a * concentration TubeType.A + s.b * concentration TubeType.B + s.c * concentration TubeType.C) / s.total

/-- Predicate to check if the solution satisfies the conditions -/
def Solution.isValid (s : Solution) : Prop :=
  s.averageConcentration = 20.17/100 ∧
  s.total ≥ 3 ∧
  s.a > 0 ∧ s.b > 0 ∧ s.c > 0

theorem max_type_c_tubes (s : Solution) (h : s.isValid) :
  s.c ≤ 73 :=
sorry

theorem exists_solution_with_73_type_c :
  ∃ s : Solution, s.isValid ∧ s.c = 73 :=
sorry

end NUMINAMATH_CALUDE_max_type_c_tubes_exists_solution_with_73_type_c_l3703_370314


namespace NUMINAMATH_CALUDE_factorial_loop_condition_l3703_370371

/-- A function that calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

/-- The theorem stating that if a factorial program outputs 720,
    then the loop condition must be i <= 6 -/
theorem factorial_loop_condition (output : ℕ) (loop_condition : ℕ → Bool) :
  output = 720 →
  (∀ n : ℕ, factorial n = output → loop_condition = fun i => i ≤ n) →
  loop_condition = fun i => i ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_factorial_loop_condition_l3703_370371


namespace NUMINAMATH_CALUDE_stationery_cost_is_52_66_l3703_370320

/-- Represents the cost calculation for a set of stationery items -/
def stationery_cost (usd_to_cad_rate : ℝ) : ℝ := by
  -- Define the base costs
  let pencil_cost : ℝ := 2
  let pen_cost : ℝ := pencil_cost + 9
  let notebook_cost : ℝ := 2 * pen_cost

  -- Apply discounts
  let discounted_notebook_cost : ℝ := notebook_cost * 0.85
  let discounted_pen_cost : ℝ := pen_cost * 0.8

  -- Calculate total cost in USD before tax
  let total_usd_before_tax : ℝ := pencil_cost + 2 * discounted_pen_cost + discounted_notebook_cost

  -- Apply tax
  let total_usd_with_tax : ℝ := total_usd_before_tax * 1.1

  -- Convert to CAD
  exact total_usd_with_tax * usd_to_cad_rate

/-- Theorem stating that the total cost of the stationery items is $52.66 CAD -/
theorem stationery_cost_is_52_66 :
  stationery_cost 1.25 = 52.66 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_is_52_66_l3703_370320


namespace NUMINAMATH_CALUDE_equation_solution_l3703_370341

/-- Floor function: greatest integer less than or equal to x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Ceiling function: smallest integer greater than or equal to x -/
noncomputable def ceil (x : ℝ) : ℤ :=
  -Int.floor (-x)

/-- Nearest integer function: integer closest to x (x ≠ n + 0.5 for any integer n) -/
noncomputable def nearest (x : ℝ) : ℤ :=
  if x - Int.floor x < 0.5 then Int.floor x else Int.floor x + 1

/-- Theorem: The equation 3⌊x⌋ + 2⌈x⌉ + ⟨x⟩ = 8 is satisfied if and only if 1 < x < 1.5 -/
theorem equation_solution (x : ℝ) :
  3 * (floor x) + 2 * (ceil x) + (nearest x) = 8 ↔ 1 < x ∧ x < 1.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3703_370341


namespace NUMINAMATH_CALUDE_fourth_person_height_l3703_370369

/-- Represents the heights of four people standing in order of increasing height. -/
structure Heights :=
  (first : ℝ)
  (second : ℝ)
  (third : ℝ)
  (fourth : ℝ)

/-- The conditions of the problem. -/
def HeightConditions (h : Heights) : Prop :=
  h.second = h.first + 2 ∧
  h.third = h.second + 2 ∧
  h.fourth = h.third + 6 ∧
  (h.first + h.second + h.third + h.fourth) / 4 = 78

/-- The theorem stating that under the given conditions, the fourth person's height is 84 inches. -/
theorem fourth_person_height (h : Heights) (hc : HeightConditions h) : h.fourth = 84 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l3703_370369


namespace NUMINAMATH_CALUDE_remainder_after_adding_2025_l3703_370313

theorem remainder_after_adding_2025 (m : ℤ) : 
  m % 9 = 4 → (m + 2025) % 9 = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2025_l3703_370313


namespace NUMINAMATH_CALUDE_grey_purchase_theorem_l3703_370358

/-- Represents the number of chickens and ducks bought by a person -/
structure Purchase where
  chickens : Nat
  ducks : Nat

/-- The problem setup -/
def grey_purchase_problem : Prop :=
  ∃ (mary_purchase : Purchase),
    let ray_purchase : Purchase := ⟨10, 3⟩
    let john_purchase : Purchase := ⟨mary_purchase.chickens + 5, mary_purchase.ducks + 2⟩
    ray_purchase.chickens = mary_purchase.chickens - 6 ∧
    ray_purchase.ducks = mary_purchase.ducks - 1 ∧
    (john_purchase.chickens + john_purchase.ducks) - (ray_purchase.chickens + ray_purchase.ducks) = 14

theorem grey_purchase_theorem : grey_purchase_problem := by
  sorry

end NUMINAMATH_CALUDE_grey_purchase_theorem_l3703_370358


namespace NUMINAMATH_CALUDE_floor_plus_twice_eq_33_l3703_370385

theorem floor_plus_twice_eq_33 :
  ∃! x : ℝ, (⌊x⌋ : ℝ) + 2 * x = 33 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_twice_eq_33_l3703_370385


namespace NUMINAMATH_CALUDE_find_m_l3703_370360

theorem find_m (U A : Set ℤ) (m : ℤ) : 
  U = {2, 3, m^2 + m - 4} →
  A = {m, 2} →
  U \ A = {3} →
  m = -2 := by sorry

end NUMINAMATH_CALUDE_find_m_l3703_370360


namespace NUMINAMATH_CALUDE_solve_equation_solve_inequalities_l3703_370379

-- Part 1: Equation
theorem solve_equation (x : ℝ) : 
  x + 3 ≠ 0 → ((2 * x + 1) / (x + 3) = 1 / (x + 3) + 1 ↔ x = 3) :=
sorry

-- Part 2: System of Inequalities
theorem solve_inequalities (x : ℝ) :
  (2 * x - 2 < x ∧ 3 * (x + 1) ≥ 6) ↔ (1 ≤ x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_solve_equation_solve_inequalities_l3703_370379


namespace NUMINAMATH_CALUDE_point_2_3_in_first_quadrant_l3703_370335

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Theorem: The point (2, 3) is in the first quadrant -/
theorem point_2_3_in_first_quadrant :
  let p : Point := ⟨2, 3⟩
  isInFirstQuadrant p := by
  sorry

end NUMINAMATH_CALUDE_point_2_3_in_first_quadrant_l3703_370335


namespace NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3703_370356

/-- Represents the price reduction in yuan -/
def price_reduction : ℝ := 20

/-- Initial average daily sale in pieces -/
def initial_sale : ℝ := 20

/-- Initial profit per piece in yuan -/
def initial_profit : ℝ := 40

/-- Additional pieces sold per yuan of price reduction -/
def sales_increase_rate : ℝ := 2

/-- Target average daily profit in yuan -/
def target_profit : ℝ := 1200

/-- Theorem stating that the given price reduction achieves the target profit -/
theorem price_reduction_achieves_target_profit :
  (initial_profit - price_reduction) * (initial_sale + sales_increase_rate * price_reduction) = target_profit :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3703_370356


namespace NUMINAMATH_CALUDE_car_speed_problem_l3703_370318

/-- A car travels uphill and downhill. This theorem proves the downhill speed given certain conditions. -/
theorem car_speed_problem (uphill_speed : ℝ) (total_distance : ℝ) (total_time uphill_time downhill_time : ℝ) 
  (h1 : uphill_speed = 30)
  (h2 : total_distance = 650)
  (h3 : total_time = 15)
  (h4 : uphill_time = 5)
  (h5 : downhill_time = 5) :
  ∃ downhill_speed : ℝ, 
    downhill_speed * downhill_time + uphill_speed * uphill_time = total_distance ∧ 
    downhill_speed = 100 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3703_370318


namespace NUMINAMATH_CALUDE_common_roots_imply_c_d_l3703_370342

/-- Two cubic polynomials with two distinct common roots -/
def has_two_common_roots (c d : ℝ) : Prop :=
  ∃ r s : ℝ, r ≠ s ∧
    (r^3 + c*r^2 + 12*r + 7 = 0) ∧ 
    (r^3 + d*r^2 + 15*r + 9 = 0) ∧
    (s^3 + c*s^2 + 12*s + 7 = 0) ∧ 
    (s^3 + d*s^2 + 15*s + 9 = 0)

/-- The theorem stating that if the polynomials have two distinct common roots, then c = 5 and d = 4 -/
theorem common_roots_imply_c_d (c d : ℝ) :
  has_two_common_roots c d → c = 5 ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_common_roots_imply_c_d_l3703_370342


namespace NUMINAMATH_CALUDE_quadratic_equation_with_ratio_l3703_370304

/-- Given a quadratic equation x^2 + 6x + k = 0 where the nonzero roots are in the ratio 2:1, 
    the value of k is 8. -/
theorem quadratic_equation_with_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x = 2*y ∧ 
   x^2 + 6*x + k = 0 ∧ y^2 + 6*y + k = 0) → k = 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_ratio_l3703_370304


namespace NUMINAMATH_CALUDE_domino_probability_and_attempts_l3703_370349

/-- The total number of domino tiles -/
def total_tiles : ℕ := 45

/-- The number of tiles drawn -/
def drawn_tiles : ℕ := 3

/-- The probability of the event occurring in a single attempt -/
def event_probability : ℚ := 54 / 473

/-- The minimum probability we want to achieve -/
def target_probability : ℝ := 0.9

/-- The minimum number of attempts needed -/
def min_attempts : ℕ := 19

/-- Theorem stating the probability of the event and the minimum number of attempts needed -/
theorem domino_probability_and_attempts :
  (event_probability : ℝ) = 54 / 473 ∧
  (1 - (1 - event_probability) ^ min_attempts : ℝ) ≥ target_probability ∧
  ∀ n : ℕ, n < min_attempts → (1 - (1 - event_probability) ^ n : ℝ) < target_probability :=
sorry


end NUMINAMATH_CALUDE_domino_probability_and_attempts_l3703_370349


namespace NUMINAMATH_CALUDE_inequality_proof_l3703_370311

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2014 + b^2014 + c^2014 + a*b*c = 4) :
  (a^2013 + b^2013 - c)/c^2013 + (b^2013 + c^2013 - a)/a^2013 + (c^2013 + a^2013 - b)/b^2013 
  ≥ a^2012 + b^2012 + c^2012 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3703_370311


namespace NUMINAMATH_CALUDE_parabola_focus_l3703_370310

/-- The parabola is defined by the equation x^2 = 20y -/
def parabola (x y : ℝ) : Prop := x^2 = 20 * y

/-- The focus of a parabola with equation x^2 = 4py has coordinates (0, p) -/
def is_focus (x y p : ℝ) : Prop := x = 0 ∧ y = p

/-- Theorem: The focus of the parabola x^2 = 20y has coordinates (0, 5) -/
theorem parabola_focus :
  ∃ (x y : ℝ), parabola x y ∧ is_focus x y 5 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l3703_370310


namespace NUMINAMATH_CALUDE_blood_expiration_theorem_l3703_370334

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Calculates the expiration date and time for a blood donation -/
def calculateExpirationDateTime (donationDateTime : DateTime) : DateTime :=
  sorry

/-- The number of seconds in a day -/
def secondsPerDay : ℕ := 86400

/-- The expiration time in seconds for a unit of blood -/
def bloodExpirationSeconds : ℕ := Nat.factorial 9

/-- Theorem stating that a blood donation made at 8 AM on January 15th 
    will expire on January 19th at 4:48 AM -/
theorem blood_expiration_theorem 
  (donationDateTime : DateTime)
  (h1 : donationDateTime.year = 2023)
  (h2 : donationDateTime.month = 1)
  (h3 : donationDateTime.day = 15)
  (h4 : donationDateTime.hour = 8)
  (h5 : donationDateTime.minute = 0) :
  let expirationDateTime := calculateExpirationDateTime donationDateTime
  expirationDateTime.year = 2023 ∧
  expirationDateTime.month = 1 ∧
  expirationDateTime.day = 19 ∧
  expirationDateTime.hour = 4 ∧
  expirationDateTime.minute = 48 :=
sorry

end NUMINAMATH_CALUDE_blood_expiration_theorem_l3703_370334


namespace NUMINAMATH_CALUDE_quadratic_function_determination_l3703_370328

open Real

/-- Given real numbers a, b, c, and functions f and g,
    if the maximum value of g(x) is 2 when -1 ≤ x ≤ 1,
    then f(x) = 2x^2 - 1 -/
theorem quadratic_function_determination
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_max : ∀ x ∈ Set.Icc (-1) 1, g x ≤ 2)
  (h_reaches_max : ∃ x ∈ Set.Icc (-1) 1, g x = 2) :
  ∀ x, f x = 2 * x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_determination_l3703_370328


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l3703_370372

def f (c x : ℝ) : ℝ := c*x^4 + 15*x^3 - 5*c*x^2 - 45*x + 55

theorem factor_implies_c_value (c : ℝ) :
  (∀ x : ℝ, (x + 5) ∣ f c x) → c = 319 / 100 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l3703_370372


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3703_370386

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3703_370386


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l3703_370309

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l3703_370309


namespace NUMINAMATH_CALUDE_divisor_power_difference_l3703_370324

theorem divisor_power_difference (k : ℕ) : 
  (15 ^ k ∣ 823435) → 5 ^ k - k ^ 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisor_power_difference_l3703_370324


namespace NUMINAMATH_CALUDE_average_of_two_numbers_l3703_370323

theorem average_of_two_numbers (a b c : ℝ) : 
  (a + b + c) / 3 = 48 → c = 32 → (a + b) / 2 = 56 := by
sorry

end NUMINAMATH_CALUDE_average_of_two_numbers_l3703_370323


namespace NUMINAMATH_CALUDE_deck_size_proof_l3703_370395

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b) = 2 / 5 →
  (r : ℚ) / (r + b + 6) = 1 / 3 →
  r + b = 30 := by
  sorry

end NUMINAMATH_CALUDE_deck_size_proof_l3703_370395


namespace NUMINAMATH_CALUDE_parallelogram_area_l3703_370339

/-- The area of a parallelogram with vertices at (0, 0), (3, 0), (5, 12), and (8, 12) is 36 square units. -/
theorem parallelogram_area : 
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (3, 0)
  let v3 : ℝ × ℝ := (5, 12)
  let v4 : ℝ × ℝ := (8, 12)
  let base : ℝ := v2.1 - v1.1
  let height : ℝ := v3.2 - v1.2
  base * height = 36 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3703_370339


namespace NUMINAMATH_CALUDE_average_speed_comparison_l3703_370336

theorem average_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) :
  3 / (1/u + 1/v + 1/w) ≤ (u + v + w) / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_comparison_l3703_370336


namespace NUMINAMATH_CALUDE_less_than_implies_difference_negative_l3703_370343

theorem less_than_implies_difference_negative (a b : ℝ) : a < b → a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_less_than_implies_difference_negative_l3703_370343


namespace NUMINAMATH_CALUDE_law_school_students_l3703_370347

/-- The number of students in the business school -/
def business_students : ℕ := 500

/-- The number of sibling pairs -/
def sibling_pairs : ℕ := 30

/-- The probability of selecting a sibling pair -/
def sibling_pair_probability : ℚ := 7500000000000001 / 100000000000000000

/-- Theorem stating the number of law students -/
theorem law_school_students (L : ℕ) : 
  (sibling_pairs : ℚ) / (business_students * L) = sibling_pair_probability → 
  L = 8000 := by
  sorry

end NUMINAMATH_CALUDE_law_school_students_l3703_370347


namespace NUMINAMATH_CALUDE_average_of_numbers_l3703_370340

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1530, 1200]

theorem average_of_numbers : (numbers.sum / numbers.length : ℚ) = 1380 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l3703_370340


namespace NUMINAMATH_CALUDE_bucket_capacities_solution_l3703_370355

/-- Represents the capacities of three buckets A, B, and C. -/
structure BucketCapacities where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given capacities satisfy the problem conditions. -/
def satisfiesConditions (caps : BucketCapacities) : Prop :=
  caps.a + caps.b + caps.c = 1440 ∧
  caps.a + (1/5) * caps.b = caps.c ∧
  caps.b + (1/3) * caps.a = caps.c

/-- Theorem stating that the unique solution satisfying the conditions is (480, 400, 560). -/
theorem bucket_capacities_solution :
  ∃! (caps : BucketCapacities), satisfiesConditions caps ∧ 
    caps.a = 480 ∧ caps.b = 400 ∧ caps.c = 560 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacities_solution_l3703_370355


namespace NUMINAMATH_CALUDE_ellipse_focus_k_l3703_370392

-- Define the ellipse
def ellipse (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 2) + y^2 / 9 = 1

-- Define the focus
def focus : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem ellipse_focus_k (k : ℝ) :
  (∀ x y, ellipse k x y) → (focus.1 = 0 ∧ focus.2 = 2) → k = 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_k_l3703_370392


namespace NUMINAMATH_CALUDE_removed_number_is_34_l3703_370329

/-- Given n consecutive natural numbers starting from 1, if one number x is removed
    and the average of the remaining numbers is 152/7, then x = 34. -/
theorem removed_number_is_34 (n : ℕ) (x : ℕ) :
  (x ≥ 1 ∧ x ≤ n) →
  (n * (n + 1) / 2 - x) / (n - 1) = 152 / 7 →
  x = 34 := by
  sorry

end NUMINAMATH_CALUDE_removed_number_is_34_l3703_370329


namespace NUMINAMATH_CALUDE_probability_between_R_and_S_l3703_370352

/-- Given points P, Q, R, and S on a line segment PQ where PQ = 4PS and PQ = 8QR,
    the probability that a randomly selected point on PQ is between R and S is 5/8. -/
theorem probability_between_R_and_S (P Q R S : ℝ) : 
  P < R ∧ R < S ∧ S < Q →  -- Points are in order on the line
  Q - P = 4 * (S - P) →    -- PQ = 4PS
  Q - P = 8 * (Q - R) →    -- PQ = 8QR
  (S - R) / (Q - P) = 5/8  -- Probability is length of RS divided by length of PQ
  := by sorry

end NUMINAMATH_CALUDE_probability_between_R_and_S_l3703_370352


namespace NUMINAMATH_CALUDE_five_solutions_l3703_370393

/-- The number of integer solution pairs (x, y) to the equation √x + √y = √336 -/
def num_solutions : ℕ := 5

/-- A predicate that checks if a pair of natural numbers satisfies the equation -/
def is_solution (x y : ℕ) : Prop :=
  Real.sqrt (x : ℝ) + Real.sqrt (y : ℝ) = Real.sqrt 336

/-- The theorem stating that there are exactly 5 solution pairs -/
theorem five_solutions :
  ∃! (s : Finset (ℕ × ℕ)), s.card = num_solutions ∧ ∀ (x y : ℕ), (x, y) ∈ s ↔ is_solution x y :=
sorry

end NUMINAMATH_CALUDE_five_solutions_l3703_370393


namespace NUMINAMATH_CALUDE_H_range_l3703_370338

def H (x : ℝ) : ℝ := |x + 2| - |x - 3| + 3 * x

theorem H_range : Set.range H = Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_H_range_l3703_370338


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2011_l3703_370362

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

/-- The problem statement -/
theorem arithmetic_sequence_2011 :
  arithmeticSequenceTerm 1 3 671 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2011_l3703_370362


namespace NUMINAMATH_CALUDE_candy_shop_ratio_l3703_370378

/-- Proves that the ratio of cherry sours to lemon sours is 4:5 given the conditions of the candy shop problem -/
theorem candy_shop_ratio :
  ∀ (total cherry orange lemon : ℕ),
  total = 96 →
  cherry = 32 →
  orange = total / 4 →
  total = cherry + orange + lemon →
  (cherry : ℚ) / lemon = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_candy_shop_ratio_l3703_370378


namespace NUMINAMATH_CALUDE_no_positive_integers_divisible_by_three_l3703_370317

theorem no_positive_integers_divisible_by_three (n : ℕ) : 
  n > 0 ∧ 3 ∣ n → ¬(28 - 6 * n > 14) :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integers_divisible_by_three_l3703_370317


namespace NUMINAMATH_CALUDE_equation_proof_l3703_370348

theorem equation_proof : 529 - 2 * 23 * 8 + 64 = 225 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3703_370348


namespace NUMINAMATH_CALUDE_percentage_difference_l3703_370363

theorem percentage_difference (total : ℝ) (z_share : ℝ) (x_premium : ℝ) : 
  total = 555 → z_share = 150 → x_premium = 0.25 →
  ∃ y_share : ℝ, 
    y_share = (total - z_share) / (2 + x_premium) ∧
    (y_share - z_share) / z_share = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3703_370363


namespace NUMINAMATH_CALUDE_unique_divisor_square_sum_l3703_370319

theorem unique_divisor_square_sum (p n : ℕ) (hp : Prime p) (hn : n > 0) (hodd : Odd p) :
  ∃! d : ℕ, d > 0 ∧ d ∣ (p * n^2) ∧ ∃ m : ℕ, d + n^2 = m^2 ↔ ∃ k : ℕ, n = k * ((p - 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_unique_divisor_square_sum_l3703_370319


namespace NUMINAMATH_CALUDE_least_number_of_cans_l3703_370330

def maaza_volume : ℕ := 10
def pepsi_volume : ℕ := 144
def sprite_volume : ℕ := 368

theorem least_number_of_cans (can_volume : ℕ) 
  (h1 : can_volume > 0)
  (h2 : can_volume ∣ maaza_volume)
  (h3 : can_volume ∣ pepsi_volume)
  (h4 : can_volume ∣ sprite_volume)
  (h5 : ∀ x : ℕ, x > can_volume → ¬(x ∣ maaza_volume ∧ x ∣ pepsi_volume ∧ x ∣ sprite_volume)) :
  maaza_volume / can_volume + pepsi_volume / can_volume + sprite_volume / can_volume = 261 :=
sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l3703_370330


namespace NUMINAMATH_CALUDE_evaluate_expression_l3703_370321

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 2 * y^x = 533 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3703_370321


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3703_370332

theorem quadratic_equation_solution : ∃ (a b : ℝ), 
  (a^2 - 4*a + 7 = 19) ∧ 
  (b^2 - 4*b + 7 = 19) ∧ 
  (a ≥ b) ∧ 
  (2*a + b = 10) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3703_370332


namespace NUMINAMATH_CALUDE_no_fixed_points_implies_a_range_l3703_370394

/-- A quadratic function f(x) = x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The property of having no fixed points -/
def has_no_fixed_points (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x ≠ x

/-- The main theorem -/
theorem no_fixed_points_implies_a_range (a : ℝ) :
  has_no_fixed_points a → -1 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_no_fixed_points_implies_a_range_l3703_370394


namespace NUMINAMATH_CALUDE_distance_between_3rd_and_21st_red_lights_l3703_370350

/-- Represents the pattern of lights on the string -/
inductive LightColor
| Red
| Green

/-- Defines the repeating pattern of lights -/
def lightPattern : List LightColor :=
  [LightColor.Red, LightColor.Red, LightColor.Green, LightColor.Green, LightColor.Green]

/-- The spacing between lights in inches -/
def lightSpacing : ℕ := 6

/-- The number of inches in a foot -/
def inchesPerFoot : ℕ := 12

/-- Function to get the position of the nth red light -/
def nthRedLightPosition (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the distance between the 3rd and 21st red lights -/
theorem distance_between_3rd_and_21st_red_lights :
  (nthRedLightPosition 21 - nthRedLightPosition 3) * lightSpacing / inchesPerFoot = 22 :=
sorry

end NUMINAMATH_CALUDE_distance_between_3rd_and_21st_red_lights_l3703_370350


namespace NUMINAMATH_CALUDE_newspaper_selling_price_l3703_370333

theorem newspaper_selling_price 
  (total_newspapers : ℕ) 
  (sold_percentage : ℚ)
  (buying_discount : ℚ)
  (total_profit : ℚ) :
  total_newspapers = 500 →
  sold_percentage = 80 / 100 →
  buying_discount = 75 / 100 →
  total_profit = 550 →
  ∃ (selling_price : ℚ),
    selling_price = 2 ∧
    (sold_percentage * total_newspapers : ℚ) * selling_price -
    (1 - buying_discount) * (total_newspapers : ℚ) * selling_price = total_profit :=
by sorry

end NUMINAMATH_CALUDE_newspaper_selling_price_l3703_370333


namespace NUMINAMATH_CALUDE_smallest_number_with_same_prime_factors_l3703_370354

def is_prime_factor (p n : ℕ) : Prop :=
  Nat.Prime p ∧ n % p = 0

def has_all_prime_factors (m n : ℕ) : Prop :=
  ∀ p, is_prime_factor p n → is_prime_factor p m

theorem smallest_number_with_same_prime_factors (n : ℕ) (hn : n = 36) :
  ∃ m : ℕ, m = 6 ∧
    has_all_prime_factors m n ∧
    ∀ k : ℕ, k < m → ¬(has_all_prime_factors k n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_same_prime_factors_l3703_370354


namespace NUMINAMATH_CALUDE_xy_bounds_l3703_370357

theorem xy_bounds (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  -1 ≤ x * y ∧ x * y ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_xy_bounds_l3703_370357


namespace NUMINAMATH_CALUDE_weight_10_moles_CaH2_l3703_370300

/-- The molecular weight of CaH2 in g/mol -/
def molecular_weight_CaH2 : ℝ := 40.08 + 2 * 1.008

/-- The total weight of a given number of moles of CaH2 in grams -/
def total_weight_CaH2 (moles : ℝ) : ℝ := moles * molecular_weight_CaH2

/-- Theorem stating that 10 moles of CaH2 weigh 420.96 grams -/
theorem weight_10_moles_CaH2 : total_weight_CaH2 10 = 420.96 := by sorry

end NUMINAMATH_CALUDE_weight_10_moles_CaH2_l3703_370300
