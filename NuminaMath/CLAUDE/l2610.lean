import Mathlib

namespace NUMINAMATH_CALUDE_smallest_quotient_l2610_261013

def digit_sum_of_squares (n : ℕ) : ℕ :=
  if n < 10 then n * n else (n % 10) * (n % 10) + digit_sum_of_squares (n / 10)

theorem smallest_quotient (n : ℕ) (h : n > 0) :
  (n : ℚ) / (digit_sum_of_squares n) ≥ 1 / 9 ∧ ∃ m : ℕ, m > 0 ∧ (m : ℚ) / (digit_sum_of_squares m) = 1 / 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_quotient_l2610_261013


namespace NUMINAMATH_CALUDE_height_on_longest_side_of_6_8_10_triangle_l2610_261047

theorem height_on_longest_side_of_6_8_10_triangle :
  ∃ (a b c h : ℝ),
    a = 6 ∧ b = 8 ∧ c = 10 ∧
    a^2 + b^2 = c^2 ∧
    c > a ∧ c > b ∧
    h = 4.8 ∧
    (1/2) * c * h = (1/2) * a * b :=
sorry

end NUMINAMATH_CALUDE_height_on_longest_side_of_6_8_10_triangle_l2610_261047


namespace NUMINAMATH_CALUDE_car_price_proof_l2610_261063

/-- Calculates the price of a car given loan terms and payments -/
def carPrice (loanYears : ℕ) (downPayment : ℕ) (monthlyPayment : ℕ) : ℕ :=
  downPayment + loanYears * 12 * monthlyPayment

/-- Proves that the price of the car is $20,000 given the specified conditions -/
theorem car_price_proof :
  carPrice 5 5000 250 = 20000 := by
  sorry

#eval carPrice 5 5000 250

end NUMINAMATH_CALUDE_car_price_proof_l2610_261063


namespace NUMINAMATH_CALUDE_simplify_expression_l2610_261081

theorem simplify_expression (x y : ℝ) : 7*x + 9*y + 3 - x + 12*y + 15 = 6*x + 21*y + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2610_261081


namespace NUMINAMATH_CALUDE_water_amount_in_sport_formulation_l2610_261024

/-- Represents the ratio of ingredients in a flavored drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

/-- The sport formulation of the drink -/
def sport_ratio (r : DrinkRatio) : DrinkRatio :=
  { flavoring := r.flavoring,
    corn_syrup := r.corn_syrup / 3,
    water := r.water * 2 }

theorem water_amount_in_sport_formulation :
  let sport := sport_ratio standard_ratio
  ∀ corn_syrup_oz : ℚ,
    corn_syrup_oz = 1 →
    (sport.water / sport.corn_syrup) * corn_syrup_oz = 15 := by
  sorry

end NUMINAMATH_CALUDE_water_amount_in_sport_formulation_l2610_261024


namespace NUMINAMATH_CALUDE_distance_to_rock_mist_mountains_value_l2610_261023

/-- The distance from the city to Rock Mist Mountains, including detours -/
def distance_to_rock_mist_mountains : ℝ :=
  let sky_falls_distance : ℝ := 8
  let rock_mist_multiplier : ℝ := 50
  let break_point_percentage : ℝ := 0.3
  let cloudy_heights_percentage : ℝ := 0.6
  let thunder_pass_detour : ℝ := 25
  
  sky_falls_distance * rock_mist_multiplier + thunder_pass_detour

/-- Theorem stating the distance to Rock Mist Mountains -/
theorem distance_to_rock_mist_mountains_value :
  distance_to_rock_mist_mountains = 425 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_rock_mist_mountains_value_l2610_261023


namespace NUMINAMATH_CALUDE_correct_division_l2610_261052

theorem correct_division (dividend : ℕ) (wrong_divisor correct_divisor wrong_quotient : ℕ) 
  (h1 : wrong_divisor = 87)
  (h2 : correct_divisor = 36)
  (h3 : wrong_quotient = 24)
  (h4 : dividend = wrong_divisor * wrong_quotient) :
  dividend / correct_divisor = 58 := by
sorry

end NUMINAMATH_CALUDE_correct_division_l2610_261052


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2610_261064

/-- Determinant of a 2x2 matrix -/
def det (a b c d : ℝ) : ℝ := a * d - b * c

/-- Geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_third : a 3 = 1)
  (h_det : det (a 6) 8 8 (a 8) = 0) :
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2610_261064


namespace NUMINAMATH_CALUDE_students_in_circle_l2610_261054

/-- 
Given a circle of students where the 6th and 16th students are opposite each other,
prove that the total number of students is 18.
-/
theorem students_in_circle (n : ℕ) 
  (h1 : n > 0) -- Ensure there are students in the circle
  (h2 : ∃ (a b : ℕ), a = 6 ∧ b = 16 ∧ a ≤ n ∧ b ≤ n) -- 6th and 16th students exist
  (h3 : (16 - 6) * 2 + 2 = n) -- Condition for 6th and 16th being opposite
  : n = 18 := by
  sorry

end NUMINAMATH_CALUDE_students_in_circle_l2610_261054


namespace NUMINAMATH_CALUDE_problem_solution_l2610_261004

theorem problem_solution (a b : ℝ) (h : |a - 1| + (2 + b)^2 = 0) : 
  (a + b)^2009 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2610_261004


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l2610_261036

/-- The number of diagonals in a polygon with N sides -/
def num_diagonals (N : ℕ) : ℕ := N * (N - 3) / 2

/-- A regular hexagon has 9 diagonals -/
theorem hexagon_diagonals :
  num_diagonals 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l2610_261036


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_l2610_261018

def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {3, 4}

theorem intersection_implies_m_value (m : ℝ) :
  A m ∩ B = {3} → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_l2610_261018


namespace NUMINAMATH_CALUDE_blue_pill_cost_l2610_261095

def days_in_three_weeks : ℕ := 21

def blue_pills_per_day : ℕ := 2
def orange_pills_per_day : ℕ := 1

def total_cost : ℚ := 756

def blue_pill_cost_difference : ℚ := 2

theorem blue_pill_cost (x : ℚ) : 
  (blue_pills_per_day * x + orange_pills_per_day * (x - blue_pill_cost_difference)) * days_in_three_weeks = total_cost →
  x = 38 / 3 := by
sorry

end NUMINAMATH_CALUDE_blue_pill_cost_l2610_261095


namespace NUMINAMATH_CALUDE_sheridan_cats_l2610_261020

/-- The number of cats Mrs. Sheridan gave away -/
def cats_given_away : ℝ := 14.0

/-- The number of cats Mrs. Sheridan has left -/
def cats_left : ℕ := 3

/-- The initial number of cats Mrs. Sheridan had -/
def initial_cats : ℕ := 17

theorem sheridan_cats : ↑initial_cats = cats_given_away + cats_left := by sorry

end NUMINAMATH_CALUDE_sheridan_cats_l2610_261020


namespace NUMINAMATH_CALUDE_complex_equation_result_l2610_261080

theorem complex_equation_result (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : a + 2 * i = i * (b - i)) : 
  a - b = -3 := by sorry

end NUMINAMATH_CALUDE_complex_equation_result_l2610_261080


namespace NUMINAMATH_CALUDE_rhombus_perimeter_from_diagonals_l2610_261060

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter_from_diagonals (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 16 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_from_diagonals_l2610_261060


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_for_inequality_l2610_261093

-- Define the function f
def f (x a : ℝ) := |3 * x + 3| + |x - a|

-- Theorem 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 > 4} = {x : ℝ | x > -1/2 ∨ x < -5/4} := by sorry

-- Theorem 2
theorem range_of_a_for_inequality :
  ∀ a : ℝ, (∀ x : ℝ, x > -1 → f x a > 3*x + 4) ↔ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_for_inequality_l2610_261093


namespace NUMINAMATH_CALUDE_judy_caught_one_fish_l2610_261096

/-- Represents the number of fish caught by each family member and other fishing details -/
structure FishingTrip where
  ben_fish : ℕ
  billy_fish : ℕ
  jim_fish : ℕ
  susie_fish : ℕ
  thrown_back : ℕ
  total_filets : ℕ
  filets_per_fish : ℕ

/-- Calculates the number of fish Judy caught based on the fishing trip details -/
def judy_fish (trip : FishingTrip) : ℕ :=
  (trip.total_filets / trip.filets_per_fish) -
  (trip.ben_fish + trip.billy_fish + trip.jim_fish + trip.susie_fish - trip.thrown_back)

/-- Theorem stating that Judy caught 1 fish given the specific conditions of the fishing trip -/
theorem judy_caught_one_fish :
  let trip : FishingTrip := {
    ben_fish := 4,
    billy_fish := 3,
    jim_fish := 2,
    susie_fish := 5,
    thrown_back := 3,
    total_filets := 24,
    filets_per_fish := 2
  }
  judy_fish trip = 1 := by sorry

end NUMINAMATH_CALUDE_judy_caught_one_fish_l2610_261096


namespace NUMINAMATH_CALUDE_parabola_segment_length_l2610_261051

/-- The length of a segment AB on a parabola y = 4x² -/
theorem parabola_segment_length :
  ∀ (x₁ x₂ y₁ y₂ : ℝ),
  y₁ = 4 * x₁^2 →
  y₂ = 4 * x₂^2 →
  ∃ (k : ℝ),
  y₁ = k * x₁ + 1/16 →
  y₂ = k * x₂ + 1/16 →
  y₁ + y₂ = 2 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (17/8)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_segment_length_l2610_261051


namespace NUMINAMATH_CALUDE_max_value_xy_x_minus_y_l2610_261079

theorem max_value_xy_x_minus_y (x y : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) :
  x * y * (x - y) ≤ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xy_x_minus_y_l2610_261079


namespace NUMINAMATH_CALUDE_percentage_commutation_l2610_261040

theorem percentage_commutation (x : ℝ) (h : 0.30 * 0.15 * x = 18) :
  0.15 * 0.30 * x = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutation_l2610_261040


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2610_261045

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p q : Point2D) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

theorem symmetric_point_coordinates :
  let M : Point2D := ⟨1, -2⟩
  let N : Point2D := ⟨-1, 2⟩
  symmetricToOrigin M N → N = ⟨-1, 2⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2610_261045


namespace NUMINAMATH_CALUDE_modular_inverse_97_mod_101_l2610_261068

theorem modular_inverse_97_mod_101 :
  ∃! x : ℕ, x ∈ Finset.range 101 ∧ (97 * x) % 101 = 1 :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_modular_inverse_97_mod_101_l2610_261068


namespace NUMINAMATH_CALUDE_intersection_is_sinusoid_l2610_261099

/-- Represents a cylinder with radius R and height H -/
structure Cylinder where
  R : ℝ
  H : ℝ

/-- Represents the inclined plane intersecting the cylinder -/
structure InclinedPlane where
  α : ℝ  -- Angle of inclination

/-- Represents a point on the unfolded lateral surface of the cylinder -/
structure UnfoldedPoint where
  x : ℝ  -- Horizontal distance along unwrapped cylinder
  z : ℝ  -- Vertical distance

/-- The equation of the intersection line on the unfolded surface -/
def intersectionLine (c : Cylinder) (p : InclinedPlane) (point : UnfoldedPoint) : Prop :=
  ∃ (A z₀ : ℝ), point.z = A * Real.sin (point.x / c.R) + z₀

/-- Theorem stating that the intersection line is sinusoidal -/
theorem intersection_is_sinusoid (c : Cylinder) (p : InclinedPlane) :
  ∀ point : UnfoldedPoint, intersectionLine c p point := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_sinusoid_l2610_261099


namespace NUMINAMATH_CALUDE_test_average_l2610_261011

theorem test_average (male_count : ℕ) (female_count : ℕ) (male_avg : ℝ) (female_avg : ℝ)
  (h1 : male_count = 8)
  (h2 : female_count = 24)
  (h3 : male_avg = 84)
  (h4 : female_avg = 92) :
  (male_count * male_avg + female_count * female_avg) / (male_count + female_count) = 90 := by
  sorry

end NUMINAMATH_CALUDE_test_average_l2610_261011


namespace NUMINAMATH_CALUDE_sara_initial_savings_l2610_261037

/-- Sara's initial savings -/
def S : ℕ := sorry

/-- Number of weeks -/
def weeks : ℕ := 820

/-- Sara's weekly savings -/
def sara_weekly : ℕ := 10

/-- Jim's weekly savings -/
def jim_weekly : ℕ := 15

/-- Theorem stating that Sara's initial savings is $4100 -/
theorem sara_initial_savings :
  S = 4100 ∧
  S + sara_weekly * weeks = jim_weekly * weeks :=
sorry

end NUMINAMATH_CALUDE_sara_initial_savings_l2610_261037


namespace NUMINAMATH_CALUDE_divisibility_of_947B_l2610_261002

-- Define a function to check if a number is divisible by 3
def divisible_by_three (n : ℕ) : Prop := n % 3 = 0

-- Define a function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Theorem statement
theorem divisibility_of_947B (B : ℕ) : 
  B < 10 →  -- B is a single digit
  (∀ (n : ℕ), divisible_by_three n ↔ divisible_by_three (sum_of_digits n)) →  -- Divisibility rule
  (divisible_by_three (9000 + 400 + 70 + B) ↔ (B = 1 ∨ B = 4 ∨ B = 7)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_947B_l2610_261002


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2610_261088

theorem floor_equation_solution (x : ℝ) : 
  (Int.floor (2 * x) + Int.floor (3 * x) = 8 * x - 7 / 2) ↔ (x = 13 / 16 ∨ x = 17 / 16) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2610_261088


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2610_261069

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/3) = 2 * Real.sqrt 3) :
  Real.tan α = Real.sqrt 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2610_261069


namespace NUMINAMATH_CALUDE_tiling_iff_div_four_l2610_261021

/-- A T-tetromino is a shape that covers exactly 4 squares. -/
def TTetromino : Type := Unit

/-- A tiling of an n×n board with T-tetrominos. -/
def Tiling (n : ℕ) : Type := 
  {arrangement : Fin n → Fin n → Option TTetromino // 
    ∀ (i j : Fin n), ∃ (t : TTetromino), arrangement i j = some t}

/-- The main theorem: An n×n board can be tiled with T-tetrominos iff n is divisible by 4. -/
theorem tiling_iff_div_four (n : ℕ) : 
  (∃ (t : Tiling n), True) ↔ 4 ∣ n := by sorry

end NUMINAMATH_CALUDE_tiling_iff_div_four_l2610_261021


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l2610_261027

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 16 →
  set2_count = 9 →
  set2_mean = 20 →
  let total_count := set1_count + set2_count
  let combined_sum := set1_count * set1_mean + set2_count * set2_mean
  combined_sum / total_count = 18.25 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l2610_261027


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l2610_261017

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x+1)(x+a) -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ (x + 1) * (x + a)

/-- If f(x) = (x+1)(x+a) is an even function, then a = -1 -/
theorem even_function_implies_a_eq_neg_one :
  ∃ a : ℝ, IsEven (f a) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l2610_261017


namespace NUMINAMATH_CALUDE_nina_total_spent_l2610_261067

/-- The total amount Nina spends on her children's presents -/
def total_spent (toy_price toy_quantity card_price card_quantity shirt_price shirt_quantity : ℕ) : ℕ :=
  toy_price * toy_quantity + card_price * card_quantity + shirt_price * shirt_quantity

/-- Theorem stating that Nina spends $70 in total -/
theorem nina_total_spent :
  total_spent 10 3 5 2 6 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_nina_total_spent_l2610_261067


namespace NUMINAMATH_CALUDE_compound_interest_rate_proof_l2610_261084

/-- Proves that the given conditions result in the specified annual interest rate -/
theorem compound_interest_rate_proof 
  (principal : ℝ) 
  (time : ℝ) 
  (compounding_frequency : ℝ) 
  (compound_interest : ℝ) 
  (h1 : principal = 50000)
  (h2 : time = 2)
  (h3 : compounding_frequency = 2)
  (h4 : compound_interest = 4121.608)
  : ∃ (rate : ℝ), 
    (abs (rate - 0.0398) < 0.0001) ∧ 
    (principal * (1 + rate / compounding_frequency) ^ (compounding_frequency * time) = 
     principal + compound_interest) :=
by sorry


end NUMINAMATH_CALUDE_compound_interest_rate_proof_l2610_261084


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2610_261076

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_formula 
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_q : ∃ q : ℝ, q ∈ Set.Ioo 0 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n)
  (h_sum : a 1 * a 5 + 2 * a 3 * a 5 + a 2 * a 8 = 25)
  (h_mean : Real.sqrt (a 3 * a 5) = 2) :
  ∀ n : ℕ, a n = 2^(5 - n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2610_261076


namespace NUMINAMATH_CALUDE_x_value_proof_l2610_261032

theorem x_value_proof (x : ℚ) 
  (h1 : 6 * x^2 + 5 * x - 1 = 0) 
  (h2 : 18 * x^2 + 17 * x - 1 = 0) : 
  x = 1/3 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l2610_261032


namespace NUMINAMATH_CALUDE_min_natural_numbers_for_prime_products_l2610_261014

theorem min_natural_numbers_for_prime_products (p : Fin 100 → ℕ) (a : ℕ → ℕ) :
  (∀ i j, i ≠ j → p i ≠ p j) →  -- p₁, ..., p₁₀₀ are distinct
  (∀ i, Prime (p i)) →  -- p₁, ..., p₁₀₀ are prime
  (∀ i, a i > 1) →  -- Each a_i is greater than 1
  (∀ i : Fin 100, ∃ j k, a j * a k = p i * p ((i + 1) % 100)^3) →  -- Each p_i * p_{i+1}³ is a product of two a_i's
  (∃ k, ∀ i, a i ≠ 0 → i < k) →  -- There are finitely many non-zero a_i's
  (∃ k, k ≥ 150 ∧ ∀ i, a i ≠ 0 → i < k) :=  -- There are at least 150 non-zero a_i's
by sorry

end NUMINAMATH_CALUDE_min_natural_numbers_for_prime_products_l2610_261014


namespace NUMINAMATH_CALUDE_dog_reachable_area_l2610_261061

/-- The area a dog can reach when tethered to a vertex of a regular octagonal doghouse -/
theorem dog_reachable_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 1 →
  rope_length = 3 →
  ∃ (area : ℝ), area = 6.5 * Real.pi ∧ 
  area = (rope_length^2 * Real.pi * (240 / 360)) + (2 * (side_length^2 * Real.pi * (45 / 360))) :=
sorry

end NUMINAMATH_CALUDE_dog_reachable_area_l2610_261061


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2610_261053

-- Define the parabola equation
def parabola_eq (x y : ℝ) : Prop :=
  x^2 + 2*x*y + y^2 + 3*x + y = 0

-- Define the axis of symmetry
def axis_of_symmetry (x y : ℝ) : Prop :=
  x + y + 1 = 0

-- Theorem statement
theorem parabola_axis_of_symmetry :
  ∀ (x y : ℝ), parabola_eq x y → axis_of_symmetry x y :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2610_261053


namespace NUMINAMATH_CALUDE_percentage_problem_l2610_261006

theorem percentage_problem (p : ℝ) : 
  (0.2 * 1050 = p / 100 * 1500 - 15) → p = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2610_261006


namespace NUMINAMATH_CALUDE_unknown_van_capacity_l2610_261077

/-- Represents the fleet of vans with their capacities -/
structure Fleet :=
  (total_vans : Nat)
  (standard_capacity : Nat)
  (large_vans : Nat)
  (small_vans : Nat)
  (unknown_van : Nat)
  (total_capacity : Nat)

/-- Theorem stating the capacity of the unknown van -/
theorem unknown_van_capacity (f : Fleet)
  (h1 : f.total_vans = 6)
  (h2 : f.standard_capacity = 8000)
  (h3 : f.large_vans = 3)
  (h4 : f.small_vans = 2)
  (h5 : f.unknown_van = 1)
  (h6 : f.total_capacity = 57600)
  (h7 : f.large_vans * (f.standard_capacity + f.standard_capacity / 2) +
        f.small_vans * f.standard_capacity +
        (f.total_capacity - (f.large_vans * (f.standard_capacity + f.standard_capacity / 2) +
                             f.small_vans * f.standard_capacity)) =
        f.total_capacity) :
  (f.total_capacity - (f.large_vans * (f.standard_capacity + f.standard_capacity / 2) +
                       f.small_vans * f.standard_capacity)) =
  (f.standard_capacity * 7 / 10) :=
by sorry

end NUMINAMATH_CALUDE_unknown_van_capacity_l2610_261077


namespace NUMINAMATH_CALUDE_vertical_asymptote_at_four_sevenths_l2610_261001

/-- The function f(x) = (2x+3)/(7x-4) has a vertical asymptote at x = 4/7 -/
theorem vertical_asymptote_at_four_sevenths :
  ∀ f : ℝ → ℝ, (∀ x : ℝ, f x = (2*x + 3) / (7*x - 4)) →
  ∃! a : ℝ, a = 4/7 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - a| ∧ |x - a| < δ → |f x| > 1/ε := by
  sorry

end NUMINAMATH_CALUDE_vertical_asymptote_at_four_sevenths_l2610_261001


namespace NUMINAMATH_CALUDE_john_bought_360_packs_l2610_261094

def number_of_classes : ℕ := 6
def students_per_class : ℕ := 30
def packs_per_student : ℕ := 2

def total_packs : ℕ := number_of_classes * students_per_class * packs_per_student

theorem john_bought_360_packs : total_packs = 360 := by
  sorry

end NUMINAMATH_CALUDE_john_bought_360_packs_l2610_261094


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2610_261073

-- Define a geometric sequence with common ratio 2
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

-- Theorem statement
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l2610_261073


namespace NUMINAMATH_CALUDE_prob_each_student_gets_book_l2610_261015

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of books --/
def num_books : ℕ := 5

/-- The total number of possible distributions --/
def total_distributions : ℕ := num_students ^ num_books

/-- The number of valid distributions where each student gets at least one book --/
def valid_distributions : ℕ := 
  num_students ^ num_books - 
  num_students * (num_students - 1) ^ num_books + 
  (num_students.choose 2) * (num_students - 2) ^ num_books - 
  num_students

/-- The probability that each student receives at least one book --/
theorem prob_each_student_gets_book : 
  (valid_distributions : ℚ) / total_distributions = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_each_student_gets_book_l2610_261015


namespace NUMINAMATH_CALUDE_coefficient_x5_proof_l2610_261090

/-- The coefficient of x^5 in the expansion of (1+x^3)(1-2x)^6 -/
def coefficient_x5 : ℤ := -132

/-- The expansion of (1+x^3)(1-2x)^6 -/
def expansion (x : ℝ) : ℝ := (1 + x^3) * (1 - 2*x)^6

theorem coefficient_x5_proof : 
  (deriv^[5] expansion 0) / 120 = coefficient_x5 := by sorry

end NUMINAMATH_CALUDE_coefficient_x5_proof_l2610_261090


namespace NUMINAMATH_CALUDE_cube_root_product_equals_48_l2610_261098

theorem cube_root_product_equals_48 : 
  (64 : ℝ) ^ (1/3) * (27 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 48 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_product_equals_48_l2610_261098


namespace NUMINAMATH_CALUDE_logarithm_properties_l2610_261031

variable (a x y : ℝ)

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem logarithm_properties (h : a > 1) :
  (log a 1 = 0) ∧
  (log a a = 1) ∧
  (∀ x > 0, x < 1 → log a x < 0) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < x ∧ x < δ → log a x < -1/ε) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_properties_l2610_261031


namespace NUMINAMATH_CALUDE_election_result_l2610_261026

/-- Represents the total number of votes in the election --/
def total_votes : ℕ := sorry

/-- Represents the initial percentage of votes for Candidate A --/
def initial_votes_A : ℚ := 65 / 100

/-- Represents the initial percentage of votes for Candidate B --/
def initial_votes_B : ℚ := 50 / 100

/-- Represents the initial percentage of votes for Candidate C --/
def initial_votes_C : ℚ := 45 / 100

/-- Represents the number of votes that change from A to B --/
def votes_A_to_B : ℕ := 1000

/-- Represents the number of votes that change from C to B --/
def votes_C_to_B : ℕ := 500

/-- Represents the final percentage of votes for Candidate B --/
def final_votes_B : ℚ := 70 / 100

theorem election_result : 
  initial_votes_B * total_votes + votes_A_to_B + votes_C_to_B = final_votes_B * total_votes ∧
  total_votes = 7500 := by sorry

end NUMINAMATH_CALUDE_election_result_l2610_261026


namespace NUMINAMATH_CALUDE_height_difference_l2610_261075

theorem height_difference (height_a height_b : ℝ) :
  height_b = height_a * (1 + 66.67 / 100) →
  (height_b - height_a) / height_b * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_height_difference_l2610_261075


namespace NUMINAMATH_CALUDE_yuri_roll_less_than_yuko_l2610_261097

/-- Represents a player's dice roll in the board game -/
structure DiceRoll :=
  (d1 d2 d3 : Nat)

/-- The game state after both players have rolled -/
structure GameState :=
  (yuri_roll : DiceRoll)
  (yuko_roll : DiceRoll)
  (yuko_ahead : Bool)

/-- Calculate the sum of a dice roll -/
def roll_sum (roll : DiceRoll) : Nat :=
  roll.d1 + roll.d2 + roll.d3

/-- Theorem stating that if Yuko is ahead, Yuri's roll sum must be less than Yuko's -/
theorem yuri_roll_less_than_yuko (state : GameState) 
  (h1 : state.yuko_roll = DiceRoll.mk 1 5 6)
  (h2 : state.yuko_ahead = true) : 
  roll_sum state.yuri_roll < roll_sum state.yuko_roll :=
by
  sorry

end NUMINAMATH_CALUDE_yuri_roll_less_than_yuko_l2610_261097


namespace NUMINAMATH_CALUDE_negation_statement_1_negation_statement_2_negation_statement_3_l2610_261083

-- Define the set of prime numbers
def isPrime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)
def P : Set ℕ := {p : ℕ | isPrime p}

-- Statement 1
theorem negation_statement_1 :
  (∀ n : ℕ, ∃ p ∈ P, n ≤ p) ↔ (∃ n : ℕ, ∀ p ∈ P, p ≤ n) :=
sorry

-- Statement 2
theorem negation_statement_2 :
  (∀ n : ℤ, ∃! p : ℤ, n + p = 0) ↔ (∃ n : ℤ, ∀ p : ℤ, n + p ≠ 0) :=
sorry

-- Statement 3
theorem negation_statement_3 :
  (∃ y : ℝ, ∀ x : ℝ, ∃ c : ℝ, x * y = c) ↔
  (∀ y : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * y ≠ x₂ * y) :=
sorry

end NUMINAMATH_CALUDE_negation_statement_1_negation_statement_2_negation_statement_3_l2610_261083


namespace NUMINAMATH_CALUDE_stuffed_animals_difference_l2610_261089

theorem stuffed_animals_difference (mckenna kenley tenly : ℕ) : 
  mckenna = 34 →
  kenley = 2 * mckenna →
  mckenna + kenley + tenly = 175 →
  tenly - kenley = 5 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_difference_l2610_261089


namespace NUMINAMATH_CALUDE_restoration_time_is_minimum_l2610_261003

/-- Represents the time required for a process on a handicraft -/
structure ProcessTime :=
  (shaping : ℕ)
  (painting : ℕ)

/-- The set of handicrafts -/
inductive Handicraft
  | A
  | B
  | C

/-- The time required for each handicraft -/
def handicraftTime : Handicraft → ProcessTime
  | Handicraft.A => ⟨9, 15⟩
  | Handicraft.B => ⟨16, 8⟩
  | Handicraft.C => ⟨10, 14⟩

/-- The minimum time required to complete the restoration work -/
def minRestorationTime : ℕ := 46

theorem restoration_time_is_minimum :
  minRestorationTime = 46 ∧
  ∀ (order : List Handicraft), order.length = 3 →
    (order.foldl (λ acc h => acc + (handicraftTime h).shaping) 0) +
    (List.maximum (order.map (λ h => (handicraftTime h).painting)) ).getD 0 ≥ minRestorationTime :=
  sorry

#check restoration_time_is_minimum

end NUMINAMATH_CALUDE_restoration_time_is_minimum_l2610_261003


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l2610_261070

/-- A function f(x) = kx - ln x is monotonically increasing on (1, +∞) if and only if k ≥ 1 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > 1, Monotone (fun x => k * x - Real.log x)) ↔ k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l2610_261070


namespace NUMINAMATH_CALUDE_sector_central_angle_l2610_261044

/-- The central angle of a circular sector, given its radius and area -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = 4) :
  (2 * area) / (r ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2610_261044


namespace NUMINAMATH_CALUDE_number_division_problem_l2610_261025

theorem number_division_problem (n : ℕ) : 
  n % 8 = 2 ∧ n / 8 = 156 → n / 5 - 3 = 247 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2610_261025


namespace NUMINAMATH_CALUDE_sine_C_value_sine_law_extension_l2610_261008

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  area : ℝ

/-- The area of the triangle satisfies the given condition -/
def areaCondition (t : Triangle) : Prop :=
  t.area = (t.a + t.b)^2 - t.c^2

/-- The sum of two sides equals 4 -/
def sideSum (t : Triangle) : Prop :=
  t.a + t.b = 4

/-- First theorem: If the area condition and side sum condition are satisfied, 
    then the sine of angle C equals 8/17 -/
theorem sine_C_value (t : Triangle) 
    (h1 : areaCondition t) (h2 : sideSum t) : 
    Real.sin t.C = 8/17 := by
  sorry

/-- Second theorem: The ratio of difference of squares of sides to 
    the square of the third side equals the ratio of sine of difference 
    of angles to the sine of the third angle -/
theorem sine_law_extension (t : Triangle) :
    (t.a^2 - t.b^2) / t.c^2 = Real.sin (t.A - t.B) / Real.sin t.C := by
  sorry

end NUMINAMATH_CALUDE_sine_C_value_sine_law_extension_l2610_261008


namespace NUMINAMATH_CALUDE_angle_relations_l2610_261050

def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

theorem angle_relations (α β : ℝ) 
  (h_acute_α : acute_angle α) 
  (h_acute_β : acute_angle β) 
  (h_sin_α : Real.sin α = 3/5) 
  (h_tan_diff : Real.tan (α - β) = 1/3) : 
  Real.tan β = 1/3 ∧ 
  Real.sin (2*α - β) = (13 * Real.sqrt 10) / 50 := by
sorry

end NUMINAMATH_CALUDE_angle_relations_l2610_261050


namespace NUMINAMATH_CALUDE_donation_distribution_l2610_261055

theorem donation_distribution (giselle_amount sam_amount isabella_amount : ℕ) :
  giselle_amount = 120 →
  isabella_amount = giselle_amount + 15 →
  isabella_amount = sam_amount + 45 →
  (isabella_amount + giselle_amount + sam_amount) / 3 = 115 :=
by sorry

end NUMINAMATH_CALUDE_donation_distribution_l2610_261055


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l2610_261039

def is_prime (n : ℕ) : Prop := sorry

def is_cube (n : ℕ) : Prop := sorry

def ends_with (a b : ℕ) : Prop := sorry

def digit_sum (n : ℕ) : ℕ := sorry

theorem smallest_number_with_conditions (p : ℕ) (hp_prime : is_prime p) (hp_cube : is_cube p) :
  ∃ (A : ℕ), 
    A = 11713 ∧ 
    p = 13 ∧
    p ∣ A ∧ 
    ends_with A p ∧ 
    digit_sum A = p ∧ 
    ∀ (B : ℕ), (p ∣ B ∧ ends_with B p ∧ digit_sum B = p) → A ≤ B :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l2610_261039


namespace NUMINAMATH_CALUDE_base_irrelevant_l2610_261035

theorem base_irrelevant (b : ℝ) : 
  ∃ (x y : ℝ), 3^x * b^y = 19683 ∧ x - y = 9 ∧ x = 9 → 3^9 * b^0 = 19683 := by
  sorry

end NUMINAMATH_CALUDE_base_irrelevant_l2610_261035


namespace NUMINAMATH_CALUDE_determine_c_l2610_261029

/-- A function f(x) = x^2 + ax + b with domain [0, +∞) -/
def f (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The solution set of f(x) < c is (m, m+6) -/
def solution_set (a b c m : ℝ) : Prop :=
  ∀ x, x ∈ Set.Ioo m (m+6) ↔ f a b x < c

theorem determine_c (a b c m : ℝ) :
  (∀ x, x ≥ 0 → f a b x = x^2 + a*x + b) →
  solution_set a b c m →
  c = 9 := by sorry

end NUMINAMATH_CALUDE_determine_c_l2610_261029


namespace NUMINAMATH_CALUDE_repeating_decimal_proof_l2610_261042

def repeating_decimal : ℚ := 78 / 99

theorem repeating_decimal_proof :
  repeating_decimal = 26 / 33 ∧
  26 + 33 = 59 := by
  sorry

#eval (Nat.gcd 78 99)  -- Expected output: 3
#eval (78 / 3)         -- Expected output: 26
#eval (99 / 3)         -- Expected output: 33

end NUMINAMATH_CALUDE_repeating_decimal_proof_l2610_261042


namespace NUMINAMATH_CALUDE_system_solution_l2610_261086

theorem system_solution : ∃ (x y : ℝ), x = 4 ∧ y = 1 ∧ x - y = 3 ∧ 2*(x - y) = 6*y := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2610_261086


namespace NUMINAMATH_CALUDE_gcd_360_504_l2610_261057

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_504_l2610_261057


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l2610_261087

/-- Proves that in a rhombus with an area of 88 cm² and one diagonal of 11 cm, the length of the other diagonal is 16 cm. -/
theorem rhombus_diagonal_length (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_area : area = 88) 
  (h_d1 : d1 = 11) 
  (h_rhombus_area : area = (d1 * d2) / 2) : d2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l2610_261087


namespace NUMINAMATH_CALUDE_mitch_weekend_hours_l2610_261043

/-- Represents Mitch's work schedule and earnings --/
structure MitchWork where
  weekdayHours : ℕ  -- Hours worked per weekday
  weekdayRate : ℕ   -- Hourly rate for weekdays in dollars
  totalWeeklyEarnings : ℕ  -- Total weekly earnings in dollars
  weekendRate : ℕ   -- Hourly rate for weekends in dollars

/-- Calculates the number of weekend hours Mitch works --/
def weekendHours (m : MitchWork) : ℕ :=
  let weekdayEarnings := m.weekdayHours * 5 * m.weekdayRate
  let weekendEarnings := m.totalWeeklyEarnings - weekdayEarnings
  weekendEarnings / m.weekendRate

/-- Theorem stating that Mitch works 6 hours on weekends --/
theorem mitch_weekend_hours :
  ∀ (m : MitchWork),
  m.weekdayHours = 5 ∧
  m.weekdayRate = 3 ∧
  m.totalWeeklyEarnings = 111 ∧
  m.weekendRate = 6 →
  weekendHours m = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_mitch_weekend_hours_l2610_261043


namespace NUMINAMATH_CALUDE_amount_in_paise_l2610_261085

theorem amount_in_paise : 
  let a : ℝ := 190
  let percentage : ℝ := 0.5
  let amount_in_rupees : ℝ := percentage / 100 * a
  let paise_per_rupee : ℕ := 100
  ⌊amount_in_rupees * paise_per_rupee⌋ = 95 := by sorry

end NUMINAMATH_CALUDE_amount_in_paise_l2610_261085


namespace NUMINAMATH_CALUDE_negative_two_squared_l2610_261007

theorem negative_two_squared : -2^2 = -4 := by sorry

end NUMINAMATH_CALUDE_negative_two_squared_l2610_261007


namespace NUMINAMATH_CALUDE_power_function_with_specific_point_is_odd_l2610_261033

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem power_function_with_specific_point_is_odd
  (f : ℝ → ℝ)
  (h_power : isPowerFunction f)
  (h_point : f (Real.sqrt 3 / 3) = Real.sqrt 3) :
  isOddFunction f :=
sorry

end NUMINAMATH_CALUDE_power_function_with_specific_point_is_odd_l2610_261033


namespace NUMINAMATH_CALUDE_marie_total_sales_l2610_261059

/-- The total number of items Marie sold on Saturday -/
def total_sold (newspapers : ℝ) (magazines : ℕ) : ℝ :=
  newspapers + magazines

/-- Theorem: Marie sold 425 items in total -/
theorem marie_total_sales : total_sold 275.0 150 = 425 := by
  sorry

end NUMINAMATH_CALUDE_marie_total_sales_l2610_261059


namespace NUMINAMATH_CALUDE_range_of_m_l2610_261022

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -Real.sqrt (4 - p.2^2)}

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 6}

-- Define the point A
def A (m : ℝ) : ℝ × ℝ := (m, 0)

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (∃ P ∈ C, ∃ Q ∈ l, A m + P - (A m + Q) = (0, 0)) →
  m ∈ Set.Icc 2 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2610_261022


namespace NUMINAMATH_CALUDE_monotonic_increasing_range_a_l2610_261091

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 12*x - 1

-- State the theorem
theorem monotonic_increasing_range_a :
  (∀ x y : ℝ, x < y → f a x < f a y) → a ∈ Set.Icc (-6) 6 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_range_a_l2610_261091


namespace NUMINAMATH_CALUDE_original_sales_tax_percentage_l2610_261012

/-- Proves that the original sales tax percentage was 0.5%, given the conditions of the problem -/
theorem original_sales_tax_percentage
  (new_tax_rate : ℚ)
  (market_price : ℚ)
  (tax_difference : ℚ)
  (h1 : new_tax_rate = 10 / 3)
  (h2 : market_price = 9000)
  (h3 : tax_difference = 15) :
  ∃ (original_tax_rate : ℚ),
    original_tax_rate = 1 / 2 ∧
    original_tax_rate * market_price - new_tax_rate * market_price = tax_difference :=
by
  sorry

end NUMINAMATH_CALUDE_original_sales_tax_percentage_l2610_261012


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l2610_261038

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l2610_261038


namespace NUMINAMATH_CALUDE_student_count_bound_l2610_261005

theorem student_count_bound (N M k ℓ : ℕ) (h1 : M = k * N / 100) 
  (h2 : 100 * (M + 1) = ℓ * (N + 3)) (h3 : ℓ < 100) (h4 : 3 * ℓ ≠ 100) : N ≤ 197 := by
  sorry

end NUMINAMATH_CALUDE_student_count_bound_l2610_261005


namespace NUMINAMATH_CALUDE_exists_same_color_distance_one_l2610_261062

/-- A coloring of the plane using three colors -/
def Coloring := ℝ × ℝ → Fin 3

/-- Two points in the plane -/
def TwoPoints := (ℝ × ℝ) × (ℝ × ℝ)

/-- The distance between two points is 1 -/
def DistanceOne (p : TwoPoints) : Prop :=
  let (p1, p2) := p
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = 1

/-- Two points have the same color -/
def SameColor (c : Coloring) (p : TwoPoints) : Prop :=
  let (p1, p2) := p
  c p1 = c p2

/-- Main theorem: In any three-coloring of the plane, there exist two points
    of the same color with distance 1 between them -/
theorem exists_same_color_distance_one :
  ∀ c : Coloring, ∃ p : TwoPoints, DistanceOne p ∧ SameColor c p := by
  sorry

end NUMINAMATH_CALUDE_exists_same_color_distance_one_l2610_261062


namespace NUMINAMATH_CALUDE_common_chord_length_l2610_261082

/-- The length of the common chord of two intersecting circles -/
theorem common_chord_length (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 10*y - 24 = 0) →
  (x^2 + y^2 + 2*x + 2*y - 8 = 0) →
  ∃ (l : ℝ), l = 2 * Real.sqrt 5 ∧ 
    (∃ (x1 y1 x2 y2 : ℝ), 
      (x1^2 + y1^2 - 2*x1 + 10*y1 - 24 = 0) ∧
      (x1^2 + y1^2 + 2*x1 + 2*y1 - 8 = 0) ∧
      (x2^2 + y2^2 - 2*x2 + 10*y2 - 24 = 0) ∧
      (x2^2 + y2^2 + 2*x2 + 2*y2 - 8 = 0) ∧
      l^2 = (x2 - x1)^2 + (y2 - y1)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_l2610_261082


namespace NUMINAMATH_CALUDE_otimes_inequality_solution_set_l2610_261049

-- Define the custom operation ⊗
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem otimes_inequality_solution_set :
  ∀ x : ℝ, (otimes (x - 2) (x + 2) < 2) ↔ (x < 0 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_otimes_inequality_solution_set_l2610_261049


namespace NUMINAMATH_CALUDE_circle_range_m_value_l2610_261016

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the condition for a point (x, y) to be on the circle
def on_circle (x y m : ℝ) : Prop :=
  circle_equation x y m

-- Define the condition for a point (x, y) to be on the line
def on_line (x y : ℝ) : Prop :=
  line_equation x y

-- Define the condition for the origin to be on the circle with diameter MN
def origin_on_diameter (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Theorem 1: Range of m for which the equation represents a circle
theorem circle_range (m : ℝ) :
  (∃ x y, circle_equation x y m) → m < 5 :=
sorry

-- Theorem 2: Value of m when the circle intersects the line and origin is on the diameter
theorem m_value (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂, 
    on_circle x₁ y₁ m ∧ 
    on_circle x₂ y₂ m ∧ 
    on_line x₁ y₁ ∧ 
    on_line x₂ y₂ ∧ 
    origin_on_diameter x₁ y₁ x₂ y₂) 
  → m = 8/5 :=
sorry

end NUMINAMATH_CALUDE_circle_range_m_value_l2610_261016


namespace NUMINAMATH_CALUDE_reciprocal_equation_l2610_261072

theorem reciprocal_equation (x : ℚ) : 
  2 - 1 / (1 - x) = 2 * (1 / (1 - x)) → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_l2610_261072


namespace NUMINAMATH_CALUDE_sqrt_6_simplest_l2610_261065

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y z : ℝ, y * y = x → z * z = x → y = z ∨ y = -z

theorem sqrt_6_simplest :
  is_simplest_sqrt 6 ∧
  ¬ is_simplest_sqrt 8 ∧
  ¬ is_simplest_sqrt 12 ∧
  ¬ is_simplest_sqrt 0.3 :=
sorry

end NUMINAMATH_CALUDE_sqrt_6_simplest_l2610_261065


namespace NUMINAMATH_CALUDE_inequality_proof_l2610_261048

theorem inequality_proof (x₁ x₂ x₃ y₁ y₂ y₃ z₁ z₂ z₃ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hx₃ : x₃ > 0)
  (hy₁ : y₁ > 0) (hy₂ : y₂ > 0) (hy₃ : y₃ > 0)
  (hz₁ : z₁ > 0) (hz₂ : z₂ > 0) (hz₃ : z₃ > 0) :
  (x₁^3 + x₂^3 + x₃^3 + 1) * (y₁^3 + y₂^3 + y₃^3 + 1) * (z₁^3 + z₂^3 + z₃^3 + 1) ≥ 
  (9/2) * (x₁ + y₁ + z₁) * (x₂ + y₂ + z₂) * (x₃ + y₃ + z₃) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2610_261048


namespace NUMINAMATH_CALUDE_sheila_attends_probability_l2610_261092

-- Define the probabilities
def prob_rain : ℝ := 0.5
def prob_sunny : ℝ := 1 - prob_rain
def prob_sheila_goes_rain : ℝ := 0.3
def prob_sheila_goes_sunny : ℝ := 0.7
def prob_friend_drives : ℝ := 0.5

-- Define the probability of Sheila attending the picnic
def prob_sheila_attends : ℝ :=
  (prob_rain * prob_sheila_goes_rain + prob_sunny * prob_sheila_goes_sunny) * prob_friend_drives

-- Theorem statement
theorem sheila_attends_probability :
  prob_sheila_attends = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_sheila_attends_probability_l2610_261092


namespace NUMINAMATH_CALUDE_circular_window_panes_l2610_261028

theorem circular_window_panes (r : ℝ) (x : ℝ) : 
  r = 20 → 
  (9 : ℝ) * (π * r^2) = π * (r + x)^2 → 
  x = 40 :=
by sorry

end NUMINAMATH_CALUDE_circular_window_panes_l2610_261028


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l2610_261066

theorem quadrilateral_diagonal_length 
  (A B C D O : ℝ × ℝ) 
  (h1 : dist O A = 5)
  (h2 : dist O C = 12)
  (h3 : dist O D = 5)
  (h4 : dist O B = 7)
  (h5 : dist B D = 9) :
  dist A C = 13 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l2610_261066


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l2610_261010

theorem sphere_radius_ratio (v_large v_small : ℝ) (h1 : v_large = 432 * Real.pi) (h2 : v_small = 0.25 * v_large) :
  (∃ r_small r_large : ℝ, 
    v_small = (4/3) * Real.pi * r_small^3 ∧ 
    v_large = (4/3) * Real.pi * r_large^3 ∧ 
    r_small / r_large = 1 / (2^(2/3))) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l2610_261010


namespace NUMINAMATH_CALUDE_friends_pen_cost_l2610_261071

def robertPens : ℕ := 4
def juliaPens : ℕ := 3 * robertPens
def dorothyPens : ℕ := juliaPens / 2
def penCost : ℚ := 3/2

def totalPens : ℕ := robertPens + juliaPens + dorothyPens
def totalCost : ℚ := (totalPens : ℚ) * penCost

theorem friends_pen_cost : totalCost = 33 := by sorry

end NUMINAMATH_CALUDE_friends_pen_cost_l2610_261071


namespace NUMINAMATH_CALUDE_f_comp_three_roots_l2610_261000

/-- A quadratic function f(x) = x^2 + 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Counts the number of distinct real roots of a function -/
noncomputable def count_distinct_roots (g : ℝ → ℝ) : ℕ := sorry

theorem f_comp_three_roots :
  ∃! c : ℝ, count_distinct_roots (f_comp c) = 3 ∧ c = (11 - Real.sqrt 13) / 2 := by sorry

end NUMINAMATH_CALUDE_f_comp_three_roots_l2610_261000


namespace NUMINAMATH_CALUDE_flagstaff_shadow_length_l2610_261056

/-- Given a flagstaff and a building casting shadows under similar conditions,
    prove that the length of the shadow cast by the flagstaff is 40.1 m. -/
theorem flagstaff_shadow_length 
  (flagstaff_height : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (h1 : flagstaff_height = 17.5)
  (h2 : building_height = 12.5)
  (h3 : building_shadow = 28.75) :
  flagstaff_height / (flagstaff_height * building_shadow / building_height) = 17.5 / 40.1 :=
by sorry

end NUMINAMATH_CALUDE_flagstaff_shadow_length_l2610_261056


namespace NUMINAMATH_CALUDE_board_length_l2610_261030

/-- Given a board cut into two pieces, prove that its total length is 20.0 feet. -/
theorem board_length : 
  ∀ (shorter longer : ℝ),
  shorter = 8.0 →
  2 * shorter = longer + 4 →
  shorter + longer = 20.0 := by
sorry

end NUMINAMATH_CALUDE_board_length_l2610_261030


namespace NUMINAMATH_CALUDE_pie_division_l2610_261019

theorem pie_division (total_pie : ℚ) (num_people : ℕ) : 
  total_pie = 5/6 ∧ num_people = 4 → (total_pie / num_people : ℚ) = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_pie_division_l2610_261019


namespace NUMINAMATH_CALUDE_total_donation_is_1684_l2610_261058

/-- Represents the donations to four forest reserves --/
structure ForestDonations where
  treetown : ℝ
  forest_reserve : ℝ
  animal_preservation : ℝ
  birds_sanctuary : ℝ

/-- Theorem stating the total donation given the conditions --/
theorem total_donation_is_1684 (d : ForestDonations) : 
  d.treetown = 570 ∧ 
  d.forest_reserve = d.animal_preservation + 140 ∧
  5 * d.treetown = 4 * d.forest_reserve ∧
  5 * d.treetown = 2 * d.animal_preservation ∧
  5 * d.treetown = 3 * d.birds_sanctuary →
  d.treetown + d.forest_reserve + d.animal_preservation + d.birds_sanctuary = 1684 :=
by sorry

end NUMINAMATH_CALUDE_total_donation_is_1684_l2610_261058


namespace NUMINAMATH_CALUDE_pyramid_volume_scaling_l2610_261034

/-- Given a pyramid with a rectangular base and initial volume,
    calculate the new volume after scaling its dimensions. -/
theorem pyramid_volume_scaling (l w h : ℝ) (V : ℝ) :
  V = (1 / 3) * l * w * h →
  V = 60 →
  (1 / 3) * (3 * l) * (2 * w) * (2 * h) = 720 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_scaling_l2610_261034


namespace NUMINAMATH_CALUDE_lending_scenario_l2610_261078

/-- Proves that given the conditions of the lending scenario, the principal amount is 3500 Rs. -/
theorem lending_scenario (P : ℝ) 
  (h1 : P + (P * 0.1 * 3) = 1.3 * P)  -- B owes A after 3 years
  (h2 : P + (P * 0.12 * 3) = 1.36 * P)  -- C owes B after 3 years
  (h3 : 1.36 * P - 1.3 * P = 210)  -- B's gain over 3 years
  : P = 3500 := by
  sorry

#check lending_scenario

end NUMINAMATH_CALUDE_lending_scenario_l2610_261078


namespace NUMINAMATH_CALUDE_solutions_when_a_is_one_two_distinct_solutions_inequality_holds_for_all_x_l2610_261046

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 1
def g (a : ℝ) (x : ℝ) : ℝ := a * |x - 1|

-- Theorem 1
theorem solutions_when_a_is_one :
  {x : ℝ | |f x| = g 1 x} = {-2, 0, 1} := by sorry

-- Theorem 2
theorem two_distinct_solutions (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ |f x| = g a x ∧ |f y| = g a y) ↔ (a = 0 ∨ a = 2) := by sorry

-- Theorem 3
theorem inequality_holds_for_all_x (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_solutions_when_a_is_one_two_distinct_solutions_inequality_holds_for_all_x_l2610_261046


namespace NUMINAMATH_CALUDE_total_age_is_23_l2610_261074

/-- Proves that the total combined age of Ryanne, Hezekiah, and Jamison is 23 years -/
theorem total_age_is_23 (hezekiah_age : ℕ) 
  (ryanne_older : hezekiah_age + 7 = ryanne_age)
  (sum_ryanne_hezekiah : ryanne_age + hezekiah_age = 15)
  (jamison_twice : jamison_age = 2 * hezekiah_age) : 
  ryanne_age + hezekiah_age + jamison_age = 23 :=
by
  sorry

#check total_age_is_23

end NUMINAMATH_CALUDE_total_age_is_23_l2610_261074


namespace NUMINAMATH_CALUDE_beautiful_equations_proof_l2610_261041

/-- Two linear equations are "beautiful equations" if the sum of their solutions is 1 -/
def beautiful_equations (eq1 eq2 : ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), eq1 x ∧ eq2 y ∧ x + y = 1

/-- The first pair of equations -/
def eq1 (x : ℝ) : Prop := 4 * x - (x + 5) = 1

/-- The second pair of equations -/
def eq2 (y : ℝ) : Prop := -2 * y - y = 3

/-- The third pair of equations -/
def eq3 (m : ℝ) (x : ℝ) : Prop := x / 2 + m = 0

/-- The fourth pair of equations -/
def eq4 (x : ℝ) : Prop := 3 * x = x + 4

theorem beautiful_equations_proof :
  (beautiful_equations eq1 eq2) ∧
  (∀ m : ℝ, beautiful_equations (eq3 m) eq4 → m = 1/2) := by sorry

end NUMINAMATH_CALUDE_beautiful_equations_proof_l2610_261041


namespace NUMINAMATH_CALUDE_inequality_implications_l2610_261009

theorem inequality_implications (a b : ℝ) (h : a + 1 > b + 1) :
  (a > b) ∧ (a + 2 > b + 2) ∧ (-a < -b) ∧ ¬(∀ a b : ℝ, a + 1 > b + 1 → 2*a > 3*b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_implications_l2610_261009
