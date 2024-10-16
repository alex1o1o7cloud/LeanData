import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3883_388323

def A : Set Int := {-2, -1, 0, 1, 2}

def B : Set Int := {x | ∃ k ∈ A, x = 2 * k}

theorem intersection_of_A_and_B : A ∩ B = {-2, 0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3883_388323


namespace NUMINAMATH_CALUDE_segmented_part_surface_area_l3883_388374

/-- Right prism with isosceles triangle base -/
structure Prism where
  height : ℝ
  baseLength : ℝ
  baseSide : ℝ

/-- Point on an edge of the prism -/
structure EdgePoint where
  edge : Fin 3
  position : ℝ

/-- Segmented part of the prism -/
structure SegmentedPart where
  prism : Prism
  pointX : EdgePoint
  pointY : EdgePoint
  pointZ : EdgePoint

/-- Surface area of the segmented part -/
def surfaceArea (part : SegmentedPart) : ℝ := sorry

/-- Main theorem -/
theorem segmented_part_surface_area 
  (p : Prism) 
  (x y z : EdgePoint) 
  (h1 : p.height = 20)
  (h2 : p.baseLength = 18)
  (h3 : p.baseSide = 15)
  (h4 : x.edge = 0 ∧ x.position = 1/2)
  (h5 : y.edge = 1 ∧ y.position = 1/2)
  (h6 : z.edge = 2 ∧ z.position = 1/2) :
  surfaceArea { prism := p, pointX := x, pointY := y, pointZ := z } = 108 := by sorry

end NUMINAMATH_CALUDE_segmented_part_surface_area_l3883_388374


namespace NUMINAMATH_CALUDE_minimum_votes_to_win_l3883_388361

theorem minimum_votes_to_win (total_votes remaining_votes : ℕ)
  (a_votes b_votes c_votes : ℕ) (h1 : total_votes = 1500)
  (h2 : remaining_votes = 500) (h3 : a_votes + b_votes + c_votes = 1000)
  (h4 : a_votes = 350) (h5 : b_votes = 370) (h6 : c_votes = 280) :
  (∀ x : ℕ, x < 261 → 
    ∃ y : ℕ, y ≤ remaining_votes - x ∧ 
      a_votes + x ≤ b_votes + y) ∧
  (∃ z : ℕ, z = 261 ∧ 
    ∀ y : ℕ, y ≤ remaining_votes - z → 
      a_votes + z > b_votes + y) :=
by sorry

end NUMINAMATH_CALUDE_minimum_votes_to_win_l3883_388361


namespace NUMINAMATH_CALUDE_egyptian_fraction_sum_exists_l3883_388365

theorem egyptian_fraction_sum_exists : ∃ (b₂ b₃ b₄ b₅ b₆ b₇ : ℕ), 
  (b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₂ ≠ b₅ ∧ b₂ ≠ b₆ ∧ b₂ ≠ b₇ ∧
   b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₃ ≠ b₆ ∧ b₃ ≠ b₇ ∧
   b₄ ≠ b₅ ∧ b₄ ≠ b₆ ∧ b₄ ≠ b₇ ∧
   b₅ ≠ b₆ ∧ b₅ ≠ b₇ ∧
   b₆ ≠ b₇) ∧
  (11 : ℚ) / 13 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧
  (b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 7 ∨
   b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 8 ∨
   b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 9 ∨
   b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 10 ∨
   b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 11) :=
by sorry

end NUMINAMATH_CALUDE_egyptian_fraction_sum_exists_l3883_388365


namespace NUMINAMATH_CALUDE_decagon_triangles_l3883_388350

/-- A regular decagon is a polygon with 10 vertices -/
def RegularDecagon : ℕ := 10

/-- No three vertices of a regular decagon are collinear -/
axiom decagon_vertices_not_collinear : True

/-- The number of triangles formed from vertices of a regular decagon -/
def num_triangles_from_decagon : ℕ := Nat.choose RegularDecagon 3

theorem decagon_triangles :
  num_triangles_from_decagon = 120 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangles_l3883_388350


namespace NUMINAMATH_CALUDE_time_after_2500_minutes_l3883_388322

-- Define a custom datetime type
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

def startDateTime : DateTime :=
  { year := 2011, month := 1, day := 1, hour := 0, minute := 0 }

def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry  -- Implementation details omitted

theorem time_after_2500_minutes :
  addMinutes startDateTime 2500 =
    { year := 2011, month := 1, day := 2, hour := 17, minute := 40 } :=
by sorry

end NUMINAMATH_CALUDE_time_after_2500_minutes_l3883_388322


namespace NUMINAMATH_CALUDE_polynomial_equality_l3883_388398

-- Define the polynomial (x+y)^8
def polynomial (x y : ℝ) : ℝ := (x + y)^8

-- Define the third term of the expansion
def third_term (x y : ℝ) : ℝ := 28 * x^6 * y^2

-- Define the fourth term of the expansion
def fourth_term (x y : ℝ) : ℝ := 56 * x^5 * y^3

theorem polynomial_equality (p q : ℝ) :
  p > 0 ∧ q > 0 ∧ p + q = 1 ∧ third_term p q = fourth_term p q → p = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_equality_l3883_388398


namespace NUMINAMATH_CALUDE_melted_spheres_radius_l3883_388332

theorem melted_spheres_radius (r : ℝ) : r > 0 → (4 / 3 * Real.pi * r^3 = 8 / 3 * Real.pi) → r = 2^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_melted_spheres_radius_l3883_388332


namespace NUMINAMATH_CALUDE_coeff_x_squared_expansion_l3883_388383

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the coefficient of x^2 in (1+x)^10
def coeff_1_plus_x : ℕ := binomial 10 2

-- Define the coefficient of x^2 in (1-x)^9
def coeff_1_minus_x : ℕ := binomial 9 2

-- State the theorem
theorem coeff_x_squared_expansion :
  coeff_1_plus_x - coeff_1_minus_x = 9 := by sorry

end NUMINAMATH_CALUDE_coeff_x_squared_expansion_l3883_388383


namespace NUMINAMATH_CALUDE_min_value_expression_l3883_388370

theorem min_value_expression (x y z t : ℝ) 
  (h1 : x + 4*y = 4) 
  (h2 : y > 0) 
  (h3 : 0 < t) 
  (h4 : t < z) : 
  (4*z^2 / abs x) + (abs (x*z^2) / y) + (12 / (t*(z-t))) ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3883_388370


namespace NUMINAMATH_CALUDE_ball_max_height_l3883_388309

/-- The height function of a ball thrown upwards -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 50

/-- The maximum height reached by the ball -/
theorem ball_max_height : ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 130 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l3883_388309


namespace NUMINAMATH_CALUDE_hcf_problem_l3883_388395

theorem hcf_problem (a b : ℕ) (h1 : a = 588) (h2 : a ≥ b) 
  (h3 : ∃ (hcf : ℕ), Nat.lcm a b = hcf * 12 * 14) : 
  Nat.gcd a b = 7 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l3883_388395


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3883_388303

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 * x - 3| < 5} = Set.Ioo (-1 : ℝ) 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3883_388303


namespace NUMINAMATH_CALUDE_ladder_tournament_rankings_ten_player_tournament_rankings_l3883_388326

/-- The number of possible rankings in a ladder-style tournament with n players. -/
def num_rankings (n : ℕ) : ℕ :=
  if n < 2 then 0 else 2^(n-1)

/-- Theorem: The number of possible rankings in a ladder-style tournament with n players (n ≥ 2) is 2^(n-1). -/
theorem ladder_tournament_rankings (n : ℕ) (h : n ≥ 2) :
  num_rankings n = 2^(n-1) := by
  sorry

/-- Corollary: For a tournament with 10 players, there are 512 possible rankings. -/
theorem ten_player_tournament_rankings :
  num_rankings 10 = 512 := by
  sorry

end NUMINAMATH_CALUDE_ladder_tournament_rankings_ten_player_tournament_rankings_l3883_388326


namespace NUMINAMATH_CALUDE_shepherd_puzzle_l3883_388333

def process_gate (n : ℕ) : ℕ :=
  n / 2 + 1

def process_gates (n : ℕ) (gates : ℕ) : ℕ :=
  match gates with
  | 0 => n
  | m + 1 => process_gates (process_gate n) m

theorem shepherd_puzzle :
  ∃ (initial : ℕ), initial > 0 ∧ process_gates initial 6 = 2 :=
sorry

end NUMINAMATH_CALUDE_shepherd_puzzle_l3883_388333


namespace NUMINAMATH_CALUDE_square_root_three_expansion_l3883_388393

theorem square_root_three_expansion 
  (a b c d : ℕ+) 
  (h : (a : ℝ) + (b : ℝ) * Real.sqrt 3 = ((c : ℝ) + (d : ℝ) * Real.sqrt 3) ^ 2) : 
  (a : ℝ) = (c : ℝ) ^ 2 + 3 * (d : ℝ) ^ 2 ∧ (b : ℝ) = 2 * (c : ℝ) * (d : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_square_root_three_expansion_l3883_388393


namespace NUMINAMATH_CALUDE_systematic_sampling_distribution_l3883_388320

/-- Represents a building in the school -/
inductive Building
| A
| B
| C

/-- Represents the systematic sampling method -/
def systematicSampling (total : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  let interval := total / sampleSize
  List.range (total - start + 1)
    |> List.filter (fun i => (i + start - 1) % interval == 0)
    |> List.map (fun i => i + start - 1)

/-- Assigns a student number to a building -/
def assignBuilding (studentNumber : ℕ) : Building :=
  if studentNumber ≤ 200 then Building.A
  else if studentNumber ≤ 295 then Building.B
  else Building.C

/-- Counts the number of students selected for each building -/
def countSelectedStudents (selectedStudents : List ℕ) : ℕ × ℕ × ℕ :=
  selectedStudents.foldl
    (fun (a, b, c) student =>
      match assignBuilding student with
      | Building.A => (a + 1, b, c)
      | Building.B => (a, b + 1, c)
      | Building.C => (a, b, c + 1))
    (0, 0, 0)

theorem systematic_sampling_distribution :
  let totalStudents := 400
  let sampleSize := 50
  let firstNumber := 3
  let selectedStudents := systematicSampling totalStudents sampleSize firstNumber
  let (buildingA, buildingB, buildingC) := countSelectedStudents selectedStudents
  buildingA = 25 ∧ buildingB = 12 ∧ buildingC = 13 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_distribution_l3883_388320


namespace NUMINAMATH_CALUDE_f_properties_l3883_388362

noncomputable def f (x : ℝ) := Real.sin x - Real.cos x

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), (f x)^2 = (f (x + p))^2 ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), (f x)^2 = (f (x + q))^2) → p ≤ q) ∧
  (∀ (x : ℝ), f (2*x - Real.pi/2) = Real.sqrt 2 * Real.sin (x/2)) ∧
  (∃ (M : ℝ), M = 1 + Real.sqrt 3 / 2 ∧
    ∀ (x : ℝ), (f x + Real.cos x) * (Real.sqrt 3 * Real.sin x + Real.cos x) ≤ M ∧
    ∃ (x₀ : ℝ), (f x₀ + Real.cos x₀) * (Real.sqrt 3 * Real.sin x₀ + Real.cos x₀) = M) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3883_388362


namespace NUMINAMATH_CALUDE_max_remainder_eleven_l3883_388375

theorem max_remainder_eleven (A B C : ℕ) (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h2 : A = 11 * B + C) : C ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_remainder_eleven_l3883_388375


namespace NUMINAMATH_CALUDE_systematic_sampling_questionnaire_B_l3883_388388

/-- Systematic sampling problem -/
theorem systematic_sampling_questionnaire_B 
  (total_pool : ℕ) 
  (sample_size : ℕ) 
  (first_selected : ℕ) 
  (questionnaire_B_start : ℕ) 
  (questionnaire_B_end : ℕ) : 
  total_pool = 960 → 
  sample_size = 32 → 
  first_selected = 9 → 
  questionnaire_B_start = 461 → 
  questionnaire_B_end = 761 → 
  (Finset.filter (fun n => 
    questionnaire_B_start ≤ (first_selected + (n - 1) * (total_pool / sample_size)) ∧ 
    (first_selected + (n - 1) * (total_pool / sample_size)) ≤ questionnaire_B_end
  ) (Finset.range (sample_size + 1))).card = 10 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_questionnaire_B_l3883_388388


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3883_388315

-- Problem 1
theorem problem_1 (x : ℝ) : (-2*x)^2 + 3*x*x = 7*x^2 := by sorry

-- Problem 2
theorem problem_2 (m a b : ℝ) : m*a^2 - m*b^2 = m*(a - b)*(a + b) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3883_388315


namespace NUMINAMATH_CALUDE_factory_earnings_l3883_388324

-- Define the parameters
def hours_machines_123 : ℕ := 23
def hours_machine_4 : ℕ := 12
def production_rate_12 : ℕ := 2
def production_rate_34 : ℕ := 3
def price_13 : ℕ := 50
def price_24 : ℕ := 60

-- Define the earnings calculation function
def calculate_earnings (hours : ℕ) (rate : ℕ) (price : ℕ) : ℕ :=
  hours * rate * price

-- Theorem statement
theorem factory_earnings :
  calculate_earnings hours_machines_123 production_rate_12 price_13 +
  calculate_earnings hours_machines_123 production_rate_12 price_24 +
  calculate_earnings hours_machines_123 production_rate_34 price_13 +
  calculate_earnings hours_machine_4 production_rate_34 price_24 = 10670 := by
  sorry


end NUMINAMATH_CALUDE_factory_earnings_l3883_388324


namespace NUMINAMATH_CALUDE_power_five_2023_mod_11_l3883_388381

theorem power_five_2023_mod_11 : 5^2023 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_five_2023_mod_11_l3883_388381


namespace NUMINAMATH_CALUDE_age_difference_32_12_l3883_388308

/-- The difference in ages between two people given their present ages -/
def age_difference (elder_age younger_age : ℕ) : ℕ :=
  elder_age - younger_age

/-- Theorem stating the age difference between two people with given ages -/
theorem age_difference_32_12 :
  age_difference 32 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_32_12_l3883_388308


namespace NUMINAMATH_CALUDE_car_speed_equality_l3883_388307

/-- Proves that given the conditions of the car problem, the average speed of Car Y is equal to the average speed of Car X -/
theorem car_speed_equality (speed_x : ℝ) (start_delay : ℝ) (distance_after_y_starts : ℝ) : 
  speed_x = 35 →
  start_delay = 72 / 60 →
  distance_after_y_starts = 98 →
  ∃ (speed_y : ℝ), speed_y = speed_x :=
by
  sorry

#check car_speed_equality

end NUMINAMATH_CALUDE_car_speed_equality_l3883_388307


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l3883_388334

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_expression : 2 * (5 + i) + i * (-2 - i) = 11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l3883_388334


namespace NUMINAMATH_CALUDE_total_red_balloons_l3883_388338

/-- The number of red balloons Sara has -/
def sara_red : ℕ := 31

/-- The number of green balloons Sara has -/
def sara_green : ℕ := 15

/-- The number of red balloons Sandy has -/
def sandy_red : ℕ := 24

/-- Theorem stating the total number of red balloons Sara and Sandy have -/
theorem total_red_balloons : sara_red + sandy_red = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_red_balloons_l3883_388338


namespace NUMINAMATH_CALUDE_linda_savings_proof_l3883_388360

def linda_savings (total : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : Prop :=
  furniture_fraction = 3/4 ∧ 
  (1 - furniture_fraction) * total = tv_cost ∧
  tv_cost = 450

theorem linda_savings_proof (total : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) :
  linda_savings total furniture_fraction tv_cost → total = 1800 := by
  sorry

end NUMINAMATH_CALUDE_linda_savings_proof_l3883_388360


namespace NUMINAMATH_CALUDE_digit_sequence_equality_l3883_388369

def A (n : ℕ) : ℕ := (10^n - 1) / 9

theorem digit_sequence_equality (n : ℕ) (hn : n > 0) :
  Real.sqrt ((10^n + 1) * A n - 2 * A n) = 3 * A n :=
sorry

end NUMINAMATH_CALUDE_digit_sequence_equality_l3883_388369


namespace NUMINAMATH_CALUDE_eight_b_equals_sixteen_l3883_388318

theorem eight_b_equals_sixteen (a b : ℝ) 
  (h1 : 6 * a + 3 * b = 0) 
  (h2 : a = b - 3) : 
  8 * b = 16 := by
sorry

end NUMINAMATH_CALUDE_eight_b_equals_sixteen_l3883_388318


namespace NUMINAMATH_CALUDE_size_relationship_l3883_388347

theorem size_relationship : ∀ a b c : ℝ,
  a = 2^(1/2) → b = 3^(1/3) → c = 5^(1/5) →
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_size_relationship_l3883_388347


namespace NUMINAMATH_CALUDE_sarah_reading_time_l3883_388343

/-- Calculates the reading time in hours for a given number of books -/
def reading_time (words_per_minute : ℕ) (words_per_page : ℕ) (pages_per_book : ℕ) (num_books : ℕ) : ℕ :=
  let total_words := words_per_page * pages_per_book * num_books
  let total_minutes := total_words / words_per_minute
  total_minutes / 60

/-- Theorem stating that Sarah's reading time for 6 books is 20 hours -/
theorem sarah_reading_time :
  reading_time 40 100 80 6 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sarah_reading_time_l3883_388343


namespace NUMINAMATH_CALUDE_result_has_five_digits_l3883_388305

-- Define a nonzero digit type
def NonzeroDigit := { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

-- Define the operation
def operation (A B C : NonzeroDigit) : ℕ :=
  9876 + A.val * 100 + 54 + B.val * 10 + 2 - C.val

-- Theorem statement
theorem result_has_five_digits (A B C : NonzeroDigit) :
  10000 ≤ operation A B C ∧ operation A B C < 100000 :=
sorry

end NUMINAMATH_CALUDE_result_has_five_digits_l3883_388305


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l3883_388371

/-- A quadratic function with the given properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  f a b c (-2) = -13/2 ∧
  f a b c (-1) = -4 ∧
  f a b c 0 = -5/2 ∧
  f a b c 1 = -2 ∧
  f a b c 2 = -5/2 →
  f a b c 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l3883_388371


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l3883_388352

/-- Proves that two successive discounts are equivalent to a single discount --/
theorem successive_discounts_equivalence (original_price : ℝ) 
  (first_discount second_discount : ℝ) :
  original_price = 800 ∧ 
  first_discount = 0.15 ∧ 
  second_discount = 0.10 →
  let price_after_first := original_price * (1 - first_discount)
  let final_price := price_after_first * (1 - second_discount)
  let equivalent_discount := (original_price - final_price) / original_price
  equivalent_discount = 0.235 := by
  sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l3883_388352


namespace NUMINAMATH_CALUDE_x_age_is_63_l3883_388312

/-- Given the ages of three people X, Y, and Z, prove that X's current age is 63 years. -/
theorem x_age_is_63 (x y z : ℕ) : 
  (x - 3 = 2 * (y - 3)) →  -- Three years ago, X's age was twice that of Y's age
  (y - 3 = 3 * (z - 3)) →  -- Three years ago, Y's age was three times that of Z's age
  ((x + 7) + (y + 7) + (z + 7) = 130) →  -- Seven years from now, the sum of their ages will be 130 years
  x = 63 := by
sorry

end NUMINAMATH_CALUDE_x_age_is_63_l3883_388312


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_4_and_16_l3883_388358

theorem arithmetic_mean_of_4_and_16 (m : ℝ) : 
  m = (4 + 16) / 2 → m = 10 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_4_and_16_l3883_388358


namespace NUMINAMATH_CALUDE_positive_even_product_sum_zero_l3883_388353

theorem positive_even_product_sum_zero (n : ℕ) (h_pos : n > 0) (h_even : Even n) :
  ∃ (a b : ℤ), (n : ℤ) = a * b ∧ a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_even_product_sum_zero_l3883_388353


namespace NUMINAMATH_CALUDE_m_greater_than_n_l3883_388376

theorem m_greater_than_n (a b m n : ℝ) 
  (ha : 0 < a) (hb : 0 < b)
  (h : m^2 * n^2 > a^2 * m^2 + b^2 * n^2) : 
  Real.sqrt (m^2 + n^2) > a + b := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l3883_388376


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3883_388325

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 85 →
  a * d + b * c = 187 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 1486 := by
  sorry


end NUMINAMATH_CALUDE_max_sum_of_squares_l3883_388325


namespace NUMINAMATH_CALUDE_fraction_equality_implies_values_l3883_388396

theorem fraction_equality_implies_values (A B : ℚ) :
  (∀ x : ℚ, x ≠ 2 ∧ x ≠ 5 ∧ x^2 - 7*x + 10 ≠ 0 →
    (B*x - 7) / (x^2 - 7*x + 10) = A / (x - 2) + 5 / (x - 5)) →
  A = -3/5 ∧ B = 22/5 ∧ A + B = 19/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_values_l3883_388396


namespace NUMINAMATH_CALUDE_cookies_baked_l3883_388372

theorem cookies_baked (pans : ℕ) (cookies_per_pan : ℕ) (h1 : pans = 12) (h2 : cookies_per_pan = 15) :
  pans * cookies_per_pan = 180 := by
  sorry

end NUMINAMATH_CALUDE_cookies_baked_l3883_388372


namespace NUMINAMATH_CALUDE_traffic_class_drunk_drivers_l3883_388344

theorem traffic_class_drunk_drivers :
  ∀ (drunk_drivers speeders : ℕ),
    drunk_drivers + speeders = 45 →
    speeders = 7 * drunk_drivers - 3 →
    drunk_drivers = 6 := by
sorry

end NUMINAMATH_CALUDE_traffic_class_drunk_drivers_l3883_388344


namespace NUMINAMATH_CALUDE_max_visible_sum_l3883_388391

/-- Represents a cube with six faces --/
structure Cube :=
  (faces : Finset Nat)
  (face_count : faces.card = 6)
  (valid_faces : faces = {1, 2, 4, 8, 16, 32})

/-- Represents a stack of four cubes --/
def CubeStack := Fin 4 → Cube

/-- The sum of visible numbers in a cube stack --/
def visible_sum (stack : CubeStack) : Nat :=
  sorry

/-- Theorem stating the maximum sum of visible numbers --/
theorem max_visible_sum :
  ∀ stack : CubeStack, visible_sum stack ≤ 244 :=
sorry

end NUMINAMATH_CALUDE_max_visible_sum_l3883_388391


namespace NUMINAMATH_CALUDE_parallel_lines_angle_l3883_388342

/-- Two lines are parallel -/
def parallel (l m : Set (ℝ × ℝ)) : Prop := sorry

/-- A point lies on a line -/
def on_line (P : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop := sorry

/-- Angle measure in degrees -/
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

theorem parallel_lines_angle (l m t : Set (ℝ × ℝ)) (P Q C : ℝ × ℝ) :
  parallel l m →
  on_line P l →
  on_line Q m →
  on_line P t →
  on_line Q t →
  on_line C m →
  angle_measure P Q C = 50 →
  angle_measure A P Q = 130 →
  true := by sorry

end NUMINAMATH_CALUDE_parallel_lines_angle_l3883_388342


namespace NUMINAMATH_CALUDE_expansion_simplification_l3883_388319

theorem expansion_simplification (x y : ℝ) :
  (2*x - y) * (2*x + 3*y) - (x + y) * (x - y) = 3*x^2 + 4*x*y - 2*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_simplification_l3883_388319


namespace NUMINAMATH_CALUDE_sqrt_difference_between_l3883_388355

theorem sqrt_difference_between (a b : ℝ) (h : a < b) : 
  ∃ (n k : ℕ), a < Real.sqrt n - Real.sqrt k ∧ Real.sqrt n - Real.sqrt k < b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_between_l3883_388355


namespace NUMINAMATH_CALUDE_product_mod_450_l3883_388359

theorem product_mod_450 : (2011 * 1537) % 450 = 307 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_450_l3883_388359


namespace NUMINAMATH_CALUDE_triangle_median_inequality_l3883_388321

/-- Given a triangle with sides a, b, and c, and s_c as the length of the median to side c,
    this theorem proves the inequality relating these measurements. -/
theorem triangle_median_inequality (a b c s_c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hs_c : 0 < s_c)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
    (h_median : 2 * s_c^2 = (2 * a^2 + 2 * b^2 - c^2) / 4) :
    (c^2 - (a - b)^2) / (2 * (a + b)) ≤ a + b - 2 * s_c ∧ 
    a + b - 2 * s_c < (c^2 + (a - b)^2) / (4 * s_c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_median_inequality_l3883_388321


namespace NUMINAMATH_CALUDE_remainder_seven_eight_mod_hundred_l3883_388345

theorem remainder_seven_eight_mod_hundred : 7^8 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_eight_mod_hundred_l3883_388345


namespace NUMINAMATH_CALUDE_product_squared_l3883_388302

theorem product_squared (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by sorry

end NUMINAMATH_CALUDE_product_squared_l3883_388302


namespace NUMINAMATH_CALUDE_ten_person_meeting_exchanges_l3883_388311

/-- The number of business card exchanges in a group meeting -/
def business_card_exchanges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 10 people, where each person exchanges business cards
    with every other person exactly once, the total number of exchanges is 45. -/
theorem ten_person_meeting_exchanges :
  business_card_exchanges 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_person_meeting_exchanges_l3883_388311


namespace NUMINAMATH_CALUDE_cylinder_sphere_equality_l3883_388301

/-- The diameter and height of a cylinder that, when melted, forms 12 identical spheres -/
def cylinder_dimension : ℝ :=
  16

theorem cylinder_sphere_equality (d : ℝ) :
  d = cylinder_dimension →
  (d > 0) →
  (π * (d / 2)^2 * d = 12 * (4 / 3) * π * (4)^3) →
  d = 16 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_sphere_equality_l3883_388301


namespace NUMINAMATH_CALUDE_ideal_function_iff_l3883_388379

def IdealFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧
  (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂)

theorem ideal_function_iff (f : ℝ → ℝ) :
  IdealFunction f ↔
  ((∀ x, f x + f (-x) = 0) ∧
   (∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_ideal_function_iff_l3883_388379


namespace NUMINAMATH_CALUDE_current_speed_l3883_388306

/-- Proves that the speed of the current is 8.5 kmph given the specified conditions -/
theorem current_speed (rowing_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  rowing_speed = 9.5 →
  distance = 45.5 →
  time = 9.099272058235341 →
  let downstream_speed := distance / 1000 / (time / 3600)
  downstream_speed = rowing_speed + 8.5 := by
sorry

end NUMINAMATH_CALUDE_current_speed_l3883_388306


namespace NUMINAMATH_CALUDE_plane_equation_l3883_388351

/-- Given a parametric equation of a plane, prove its Cartesian equation. -/
theorem plane_equation (s t : ℝ) :
  let u : ℝ × ℝ × ℝ := (2 + 2*s - 3*t, 1 + s, 4 - s + t)
  ∃ (A B C D : ℤ), A > 0 ∧ 
    (∀ (x y z : ℝ), (x, y, z) ∈ {p : ℝ × ℝ × ℝ | ∃ s t : ℝ, p = u} ↔ A*x + B*y + C*z + D = 0) ∧
    Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1 ∧
    A = 1 ∧ B = 4 ∧ C = 3 ∧ D = -18 := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_l3883_388351


namespace NUMINAMATH_CALUDE_parallelogram_base_l3883_388380

theorem parallelogram_base (area height : ℝ) (h1 : area = 375) (h2 : height = 15) :
  area / height = 25 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l3883_388380


namespace NUMINAMATH_CALUDE_power_zero_equations_l3883_388330

theorem power_zero_equations (a : ℝ) (h : a ≠ 0) :
  (∃ x, (x + 2)^0 ≠ 1) ∧
  ((a^2 + 1)^0 = 1) ∧
  ((-6*a)^0 = 1) ∧
  ((1/a)^0 = 1) :=
sorry

end NUMINAMATH_CALUDE_power_zero_equations_l3883_388330


namespace NUMINAMATH_CALUDE_square_difference_given_sum_and_weighted_sum_l3883_388348

theorem square_difference_given_sum_and_weighted_sum (x y : ℝ) 
  (h1 : x + y = 15) 
  (h2 : 3 * x + y = 20) : 
  x^2 - y^2 = -150 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_sum_and_weighted_sum_l3883_388348


namespace NUMINAMATH_CALUDE_evaluate_expression_l3883_388337

theorem evaluate_expression : -(16 / 4 * 8 - 70 + 4^2 * 7) = -74 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3883_388337


namespace NUMINAMATH_CALUDE_max_sides_is_12_l3883_388377

/-- A convex polygon that can be divided into right triangles with acute angles of 30 and 60 degrees -/
structure ConvexPolygon where
  sides : ℕ
  is_convex : Bool
  divisible_into_right_triangles : Bool
  acute_angles : Set ℝ
  acute_angles_eq : acute_angles = {30, 60}

/-- The maximum number of sides for the described convex polygon -/
def max_sides : ℕ := 12

/-- Theorem stating that the maximum number of sides for the described convex polygon is 12 -/
theorem max_sides_is_12 (p : ConvexPolygon) : p.sides ≤ max_sides := by
  sorry

end NUMINAMATH_CALUDE_max_sides_is_12_l3883_388377


namespace NUMINAMATH_CALUDE_pictures_hanging_l3883_388364

theorem pictures_hanging (total : ℕ) (vertical : ℕ) (horizontal : ℕ) (haphazard : ℕ) : 
  total = 30 →
  vertical = 10 →
  horizontal = total / 2 →
  haphazard = total - vertical - horizontal →
  haphazard = 5 := by
sorry

end NUMINAMATH_CALUDE_pictures_hanging_l3883_388364


namespace NUMINAMATH_CALUDE_cricket_runs_total_l3883_388335

theorem cricket_runs_total (a b c : ℕ) : 
  (a : ℚ) / b = 1 / 3 →
  (b : ℚ) / c = 1 / 5 →
  c = 75 →
  a + b + c = 95 := by
  sorry

end NUMINAMATH_CALUDE_cricket_runs_total_l3883_388335


namespace NUMINAMATH_CALUDE_wifes_ring_to_first_ring_ratio_l3883_388363

/-- The cost of Jim's first ring in dollars -/
def first_ring_cost : ℝ := 10000

/-- The cost of Jim's wife's ring in dollars -/
def wifes_ring_cost : ℝ := 20000

/-- The amount Jim is out of pocket in dollars -/
def out_of_pocket : ℝ := 25000

/-- Theorem stating the ratio of the cost of Jim's wife's ring to the cost of the first ring -/
theorem wifes_ring_to_first_ring_ratio :
  wifes_ring_cost / first_ring_cost = 2 :=
by
  have h1 : first_ring_cost + wifes_ring_cost - first_ring_cost / 2 = out_of_pocket := by sorry
  sorry

#check wifes_ring_to_first_ring_ratio

end NUMINAMATH_CALUDE_wifes_ring_to_first_ring_ratio_l3883_388363


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3883_388336

theorem polynomial_simplification (x : ℝ) :
  (5 * x^10 + 8 * x^8 + 3 * x^6) + (2 * x^12 + 3 * x^10 + x^8 + 4 * x^6 + 2 * x^2 + 7) =
  2 * x^12 + 8 * x^10 + 9 * x^8 + 7 * x^6 + 2 * x^2 + 7 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3883_388336


namespace NUMINAMATH_CALUDE_decimal_72_to_octal_l3883_388314

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem decimal_72_to_octal :
  decimal_to_octal 72 = [1, 1, 0] := by sorry

end NUMINAMATH_CALUDE_decimal_72_to_octal_l3883_388314


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3883_388382

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - x - 6 ≤ 0}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3883_388382


namespace NUMINAMATH_CALUDE_smallest_a_l3883_388340

/-- The polynomial with four positive integer roots -/
def p (a b c : ℤ) (x : ℤ) : ℤ := x^4 - a*x^3 + b*x^2 - c*x + 2520

/-- The proposition that the polynomial has four positive integer roots -/
def has_four_positive_integer_roots (a b c : ℤ) : Prop :=
  ∃ w x y z : ℤ, w > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
    ∀ t : ℤ, p a b c t = 0 ↔ t = w ∨ t = x ∨ t = y ∨ t = z

/-- The theorem stating that 29 is the smallest possible value of a -/
theorem smallest_a :
  ∀ a b c : ℤ, has_four_positive_integer_roots a b c →
  (∀ a' : ℤ, has_four_positive_integer_roots a' b c → a ≤ a') →
  a = 29 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_l3883_388340


namespace NUMINAMATH_CALUDE_propositions_truth_l3883_388329

theorem propositions_truth (a b c : ℝ) (k : ℕ+) :
  (a > b → a^(k : ℝ) > b^(k : ℝ)) ∧
  (c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) :=
sorry

end NUMINAMATH_CALUDE_propositions_truth_l3883_388329


namespace NUMINAMATH_CALUDE_sin_cos_equality_solution_l3883_388386

theorem sin_cos_equality_solution (x : Real) : 
  (Real.sin (3 * x) * Real.sin (5 * x) = Real.cos (3 * x) * Real.cos (5 * x)) → 
  (x = π / 16) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equality_solution_l3883_388386


namespace NUMINAMATH_CALUDE_problem_solution_l3883_388349

noncomputable section

def e : ℝ := Real.exp 1

def f (x : ℝ) : ℝ := x * Real.log x

def g (x : ℝ) : ℝ := f x + x^2 - 2*(e+1)*x + 6

theorem problem_solution :
  (∃ x₀ ∈ Set.Icc 1 e, ∀ m : ℝ, m * (f x₀ - 1) > x₀^2 + 1 → m < -2 ∨ m > (e^2 + 1) / (e - 1)) ∧
  (∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ g x = a ∧ g y = a) → a ∈ Set.Ioo (6 - e^2 - e) 6) ∧
  (HasDerivAt g 0 e) := by sorry

end

end NUMINAMATH_CALUDE_problem_solution_l3883_388349


namespace NUMINAMATH_CALUDE_sum_squares_divisors_1800_l3883_388366

/-- Sum of squares of divisors function -/
def σ (n : ℕ) : ℕ := (Nat.divisors n).sum (λ d => d ^ 2)

/-- Property: σ(a × b) = σ(a) × σ(b) for coprime a and b -/
axiom σ_multiplicative {a b : ℕ} (h : Nat.Coprime a b) : σ (a * b) = σ a * σ b

theorem sum_squares_divisors_1800 : σ 1800 = 5035485 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_divisors_1800_l3883_388366


namespace NUMINAMATH_CALUDE_lyndees_friends_l3883_388384

theorem lyndees_friends (total_chicken : ℕ) (lyndee_ate : ℕ) (friend_ate : ℕ) 
  (h1 : total_chicken = 11)
  (h2 : lyndee_ate = 1)
  (h3 : friend_ate = 2)
  (h4 : total_chicken = lyndee_ate + friend_ate * (total_chicken - lyndee_ate) / friend_ate) :
  (total_chicken - lyndee_ate) / friend_ate = 5 := by
  sorry

end NUMINAMATH_CALUDE_lyndees_friends_l3883_388384


namespace NUMINAMATH_CALUDE_equation_solution_l3883_388354

theorem equation_solution : ∃ x : ℝ, (10 - 2 * x = 16) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3883_388354


namespace NUMINAMATH_CALUDE_product_of_roots_equation_l3883_388341

theorem product_of_roots_equation (y : ℝ) (h1 : y > 0) 
  (h2 : Real.sqrt (5 * y) * Real.sqrt (15 * y) * Real.sqrt (2 * y) * Real.sqrt (6 * y) = 6) : 
  y = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_equation_l3883_388341


namespace NUMINAMATH_CALUDE_guayaquilean_sum_of_digits_l3883_388357

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A natural number is guayaquilean if the sum of its digits equals the sum of digits of its square -/
def is_guayaquilean (n : ℕ) : Prop :=
  sum_of_digits n = sum_of_digits (n^2)

/-- The sum of digits of a guayaquilean number is either 9k or 9k + 1 for some k -/
theorem guayaquilean_sum_of_digits (n : ℕ) (h : is_guayaquilean n) :
  ∃ k : ℕ, sum_of_digits n = 9 * k ∨ sum_of_digits n = 9 * k + 1 :=
sorry

end NUMINAMATH_CALUDE_guayaquilean_sum_of_digits_l3883_388357


namespace NUMINAMATH_CALUDE_star_value_l3883_388389

-- Define the * operation
def star (a b : ℤ) : ℚ :=
  1 / a + 1 / b

-- State the theorem
theorem star_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 7) (h4 : a * b = 12) :
  star a b = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l3883_388389


namespace NUMINAMATH_CALUDE_tire_rotation_mileage_l3883_388316

theorem tire_rotation_mileage (total_tires : ℕ) (simultaneous_tires : ℕ) (total_miles : ℕ) :
  total_tires = 7 →
  simultaneous_tires = 6 →
  total_miles = 42000 →
  (total_miles * simultaneous_tires) / total_tires = 36000 :=
by sorry

end NUMINAMATH_CALUDE_tire_rotation_mileage_l3883_388316


namespace NUMINAMATH_CALUDE_toothpick_arrangement_l3883_388300

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem toothpick_arrangement :
  arithmetic_sequence 5 4 100 = 401 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_arrangement_l3883_388300


namespace NUMINAMATH_CALUDE_square_of_1008_l3883_388390

theorem square_of_1008 : (1008 : ℕ)^2 = 1016064 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1008_l3883_388390


namespace NUMINAMATH_CALUDE_pages_to_read_tonight_l3883_388368

/-- The number of pages in Juwella's book -/
def total_pages : ℕ := 500

/-- The number of pages Juwella read three nights ago -/
def pages_three_nights_ago : ℕ := 20

/-- The number of pages Juwella read two nights ago -/
def pages_two_nights_ago : ℕ := pages_three_nights_ago^2 + 5

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The number of pages Juwella read last night -/
def pages_last_night : ℕ := 3 * sum_of_digits pages_two_nights_ago

/-- The total number of pages Juwella has read so far -/
def total_pages_read : ℕ := pages_three_nights_ago + pages_two_nights_ago + pages_last_night

/-- Theorem stating the number of pages Juwella will read tonight -/
theorem pages_to_read_tonight : total_pages - total_pages_read = 48 := by
  sorry

end NUMINAMATH_CALUDE_pages_to_read_tonight_l3883_388368


namespace NUMINAMATH_CALUDE_stephens_ant_farm_l3883_388317

/-- The number of ants in Stephen's ant farm satisfies the given conditions -/
theorem stephens_ant_farm (total_ants : ℕ) : 
  (total_ants / 2 : ℚ) * (80 / 100 : ℚ) = 44 → total_ants = 110 :=
by
  sorry

#check stephens_ant_farm

end NUMINAMATH_CALUDE_stephens_ant_farm_l3883_388317


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_17_l3883_388328

theorem consecutive_integers_sqrt_17 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 17) → (Real.sqrt 17 < b) → (a + b = 9) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_17_l3883_388328


namespace NUMINAMATH_CALUDE_johnnys_hourly_rate_l3883_388394

def hours_worked : ℝ := 8
def total_earnings : ℝ := 26

theorem johnnys_hourly_rate : total_earnings / hours_worked = 3.25 := by
  sorry

end NUMINAMATH_CALUDE_johnnys_hourly_rate_l3883_388394


namespace NUMINAMATH_CALUDE_divisibility_of_S_l3883_388356

-- Define the conditions
def is_valid_prime_pair (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p > 3 ∧ q > 3 ∧ ∃ n : ℕ, q - p = 2^n ∨ p - q = 2^n

-- Define the function S
def S (p q m : ℕ) : ℕ := p^(2*m+1) + q^(2*m+1)

-- State the theorem
theorem divisibility_of_S (p q : ℕ) (h : is_valid_prime_pair p q) :
  ∀ m : ℕ, (3 : ℕ) ∣ S p q m :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_S_l3883_388356


namespace NUMINAMATH_CALUDE_coin_problem_l3883_388387

theorem coin_problem (initial_coins : ℚ) : 
  initial_coins > 0 →
  let lost_coins := (1 / 3 : ℚ) * initial_coins
  let found_coins := (3 / 4 : ℚ) * lost_coins
  let remaining_coins := initial_coins - lost_coins + found_coins
  (initial_coins - remaining_coins) / initial_coins = 5 / 12 := by
sorry

end NUMINAMATH_CALUDE_coin_problem_l3883_388387


namespace NUMINAMATH_CALUDE_greatest_k_for_inequality_l3883_388339

theorem greatest_k_for_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ k : ℝ, k > 0 ∧ 
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
    a/b + b/c + c/a - 3 ≥ k * (a/(b+c) + b/(c+a) + c/(a+b) - 3/2)) ∧
  (∀ k' : ℝ, k' > k → 
    ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a/b + b/c + c/a - 3 < k' * (a/(b+c) + b/(c+a) + c/(a+b) - 3/2)) ∧
  k = 1 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_for_inequality_l3883_388339


namespace NUMINAMATH_CALUDE_solution_set_for_a_2_find_a_value_l3883_388378

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_for_a_2 :
  {x : ℝ | |x - 2| ≥ 4 - |x - 4|} = {x : ℝ | x ≥ 5 ∨ x ≤ 1} :=
sorry

-- Part 2
theorem find_a_value (a : ℝ) (h : a > 1) :
  ({x : ℝ | |f a (2*x + a) - 2*(f a x)| ≤ 2} = {x : ℝ | 1 ≤ x ∧ x ≤ 2}) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_2_find_a_value_l3883_388378


namespace NUMINAMATH_CALUDE_perpendicular_lines_condition_l3883_388346

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y, m * x + 2 * y - 1 = 0 → 3 * x + (m + 1) * y + 1 = 0 → 
    (m * 3 + 2 * (m + 1) = 0)) ↔ m = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_condition_l3883_388346


namespace NUMINAMATH_CALUDE_circle_area_diameter_4_l3883_388313

/-- The area of a circle with diameter 4 meters is 4π square meters. -/
theorem circle_area_diameter_4 :
  let diameter : ℝ := 4
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_diameter_4_l3883_388313


namespace NUMINAMATH_CALUDE_system_solution_l3883_388304

theorem system_solution (x y : ℚ) : 
  (10 / (2 * x + 3 * y - 29) + 9 / (7 * x - 8 * y + 24) = 8) ∧ 
  ((2 * x + 3 * y - 29) / 2 = (7 * x - 8 * y) / 3 + 8) → 
  x = 5 ∧ y = 7 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3883_388304


namespace NUMINAMATH_CALUDE_inequalities_hold_l3883_388392

theorem inequalities_hold (a b c x y z : ℝ) 
  (h1 : x^2 < a) (h2 : y^2 < b) (h3 : z^2 < c) : 
  (x*y + y*z + z*x < a + b + c) ∧ 
  (x^4 + y^4 + z^4 < a^2 + b^2 + c^2) ∧ 
  (x^3*y^3*z^3 < a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l3883_388392


namespace NUMINAMATH_CALUDE_system_solutions_l3883_388385

-- Define the system of equations
def equation1 (x y : ℝ) : Prop :=
  (-x^7 / y)^(Real.log (-y)) = x^(2 * Real.log (x * y^2))

def equation2 (x y : ℝ) : Prop :=
  y^2 + 2*x*y - 3*x^2 + 12*x + 4*y = 0

-- Define the solution set
def solutions : Set (ℝ × ℝ) :=
  {(2, -2), (3, -9), ((Real.sqrt 17 - 1) / 2, (Real.sqrt 17 - 9) / 2)}

-- Theorem statement
theorem system_solutions :
  ∀ x y : ℝ, x ≠ 0 ∧ y < 0 →
    (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3883_388385


namespace NUMINAMATH_CALUDE_sum_to_target_l3883_388399

theorem sum_to_target : ∃ x : ℝ, 0.003 + 0.158 + x = 2.911 ∧ x = 2.750 := by
  sorry

end NUMINAMATH_CALUDE_sum_to_target_l3883_388399


namespace NUMINAMATH_CALUDE_xy_value_l3883_388327

theorem xy_value (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) : x * y = 32 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3883_388327


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l3883_388310

theorem sum_of_reciprocal_equations (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 3)
  (h2 : x⁻¹ - y⁻¹ = -7) : 
  x + y = -3/10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l3883_388310


namespace NUMINAMATH_CALUDE_rhombus_side_length_l3883_388367

/-- A rhombus with area K and one diagonal three times the length of the other has side length √(5K/3) -/
theorem rhombus_side_length (K : ℝ) (d₁ d₂ : ℝ) (h_area : K = (1/2) * d₁ * d₂) (h_diag : d₂ = 3 * d₁) :
  let s := Real.sqrt ((5 * K) / 3)
  s^2 = ((d₁/2)^2 + (d₂/2)^2) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l3883_388367


namespace NUMINAMATH_CALUDE_square_sum_lower_bound_l3883_388373

theorem square_sum_lower_bound (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 + 2*x*y*z = 1) : 
  x^2 + y^2 + z^2 ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_lower_bound_l3883_388373


namespace NUMINAMATH_CALUDE_factorization_result_quadratic_factorization_l3883_388331

-- Part 1
theorem factorization_result (a b : ℤ) :
  (∀ x, (2*x - 21) * (3*x - 7) - (3*x - 7) * (x - 13) = (3*x + a) * (x + b)) →
  a + 3*b = -31 := by sorry

-- Part 2
theorem quadratic_factorization :
  ∀ x, x^2 - 3*x + 2 = (x - 1) * (x - 2) := by sorry

end NUMINAMATH_CALUDE_factorization_result_quadratic_factorization_l3883_388331


namespace NUMINAMATH_CALUDE_largest_package_size_l3883_388397

theorem largest_package_size (ming_markers : ℕ) (catherine_markers : ℕ)
  (h1 : ming_markers = 72)
  (h2 : catherine_markers = 48) :
  ∃ (package_size : ℕ),
    package_size ∣ ming_markers ∧
    package_size ∣ catherine_markers ∧
    ∀ (n : ℕ), n ∣ ming_markers → n ∣ catherine_markers → n ≤ package_size :=
by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l3883_388397
