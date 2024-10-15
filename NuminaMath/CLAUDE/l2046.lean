import Mathlib

namespace NUMINAMATH_CALUDE_cubic_transformation_l2046_204604

theorem cubic_transformation (x z : ℝ) (hz : z = x + 1/x) :
  x^3 - 3*x^2 + x + 2 = 0 ↔ x^2*(z^2 - z - 1) + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_transformation_l2046_204604


namespace NUMINAMATH_CALUDE_inequalities_properties_l2046_204697

theorem inequalities_properties (a b : ℝ) (h : a < b ∧ b < 0) : 
  abs a > abs b ∧ 
  1 / a > 1 / b ∧ 
  a / b + b / a > 2 ∧ 
  a ^ 2 > b ^ 2 := by
sorry

end NUMINAMATH_CALUDE_inequalities_properties_l2046_204697


namespace NUMINAMATH_CALUDE_expression_equality_l2046_204663

theorem expression_equality : 4⁻¹ - Real.sqrt (1/16) + (3 - Real.sqrt 2)^0 = 1 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l2046_204663


namespace NUMINAMATH_CALUDE_triangular_numbers_and_squares_l2046_204648

theorem triangular_numbers_and_squares (n a b : ℤ) :
  (n = (a^2 + a)/2 + (b^2 + b)/2) →
  (∃ x y : ℤ, 4*n + 1 = x^2 + y^2 ∧ x = a + b + 1 ∧ y = a - b) ∧
  (∀ x y : ℤ, 4*n + 1 = x^2 + y^2 →
    ∃ a' b' : ℤ, n = (a'^2 + a')/2 + (b'^2 + b')/2) :=
by sorry

end NUMINAMATH_CALUDE_triangular_numbers_and_squares_l2046_204648


namespace NUMINAMATH_CALUDE_pages_left_to_read_l2046_204617

def total_pages : ℕ := 563
def pages_read : ℕ := 147

theorem pages_left_to_read : total_pages - pages_read = 416 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l2046_204617


namespace NUMINAMATH_CALUDE_regular_polygon_with_144_degree_angles_has_10_sides_l2046_204691

/-- The number of sides of a regular polygon with interior angles measuring 144 degrees -/
def regular_polygon_sides : ℕ :=
  let interior_angle : ℚ := 144
  let n : ℕ := 10
  n

/-- Theorem stating that a regular polygon with interior angles of 144 degrees has 10 sides -/
theorem regular_polygon_with_144_degree_angles_has_10_sides :
  let interior_angle : ℚ := 144
  (interior_angle = (180 * (regular_polygon_sides - 2) : ℚ) / regular_polygon_sides) ∧
  (regular_polygon_sides > 2) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_with_144_degree_angles_has_10_sides_l2046_204691


namespace NUMINAMATH_CALUDE_touching_balls_in_cylinder_l2046_204618

theorem touching_balls_in_cylinder (a b d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hd : d > 0)
  (h_touch : a + b = d)
  (h_larger_bottom : a ≥ b) : 
  Real.sqrt d = Real.sqrt a + Real.sqrt b := by
sorry

end NUMINAMATH_CALUDE_touching_balls_in_cylinder_l2046_204618


namespace NUMINAMATH_CALUDE_sum_congruence_l2046_204660

theorem sum_congruence : ∃ k : ℤ, (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) = 16 * k + 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l2046_204660


namespace NUMINAMATH_CALUDE_right_pyramid_base_side_l2046_204669

/-- Represents a right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ
  slant_height : ℝ
  lateral_face_area : ℝ

/-- Theorem: If the area of one lateral face of a right pyramid with a square base is 200 square meters
    and the slant height is 40 meters, then the length of the side of its base is 10 meters. -/
theorem right_pyramid_base_side (p : RightPyramid) 
  (h1 : p.lateral_face_area = 200)
  (h2 : p.slant_height = 40) : 
  p.base_side = 10 := by
  sorry

#check right_pyramid_base_side

end NUMINAMATH_CALUDE_right_pyramid_base_side_l2046_204669


namespace NUMINAMATH_CALUDE_expression_equals_zero_l2046_204643

theorem expression_equals_zero (a : ℚ) (h : a = 4/3) : 
  (6*a^2 - 15*a + 5) * (3*a - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l2046_204643


namespace NUMINAMATH_CALUDE_complex_number_problem_l2046_204636

theorem complex_number_problem (a b c : ℂ) (h_a_real : a.im = 0) 
  (h_sum : a + b + c = 5)
  (h_prod_sum : a * b + b * c + c * a = 5)
  (h_prod : a * b * c = 4) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2046_204636


namespace NUMINAMATH_CALUDE_chinese_dream_speech_competition_l2046_204657

theorem chinese_dream_speech_competition :
  let num_contestants : ℕ := 4
  let num_topics : ℕ := 4
  let num_topics_used : ℕ := 3
  
  (num_topics.choose 1) * (num_topics_used ^ num_contestants) = 324 :=
by sorry

end NUMINAMATH_CALUDE_chinese_dream_speech_competition_l2046_204657


namespace NUMINAMATH_CALUDE_canal_construction_l2046_204693

/-- Canal construction problem -/
theorem canal_construction 
  (total_length : ℝ) 
  (team_b_extra : ℝ) 
  (time_ratio : ℝ) 
  (cost_a : ℝ) 
  (cost_b : ℝ) 
  (total_days : ℕ) 
  (h1 : total_length = 1650)
  (h2 : team_b_extra = 30)
  (h3 : time_ratio = 3/2)
  (h4 : cost_a = 90000)
  (h5 : cost_b = 120000)
  (h6 : total_days = 14) :
  ∃ (rate_a rate_b total_cost : ℝ),
    rate_a = 60 ∧ 
    rate_b = 90 ∧ 
    total_cost = 2340000 ∧
    rate_b = rate_a + team_b_extra ∧
    total_length / rate_a = time_ratio * (total_length / rate_b) ∧
    ∃ (days_a_alone : ℝ),
      0 ≤ days_a_alone ∧ 
      days_a_alone ≤ total_days ∧
      rate_a * days_a_alone + (rate_a + rate_b) * (total_days - days_a_alone) = total_length ∧
      total_cost = cost_a * days_a_alone + (cost_a + cost_b) * (total_days - days_a_alone) :=
by sorry

end NUMINAMATH_CALUDE_canal_construction_l2046_204693


namespace NUMINAMATH_CALUDE_number_division_problem_l2046_204659

theorem number_division_problem (x : ℚ) : x / 2 = 100 + x / 5 → x = 1000 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2046_204659


namespace NUMINAMATH_CALUDE_complex_number_location_l2046_204608

/-- The complex number z = 3 / (1 + 2i) is located in the fourth quadrant of the complex plane. -/
theorem complex_number_location (z : ℂ) (h : z = 3 / (1 + 2*I)) : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2046_204608


namespace NUMINAMATH_CALUDE_negation_of_every_scientist_is_curious_l2046_204678

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a scientist and being curious
variable (scientist : U → Prop)
variable (curious : U → Prop)

-- State the theorem
theorem negation_of_every_scientist_is_curious :
  (¬ ∀ x, scientist x → curious x) ↔ (∃ x, scientist x ∧ ¬ curious x) :=
sorry

end NUMINAMATH_CALUDE_negation_of_every_scientist_is_curious_l2046_204678


namespace NUMINAMATH_CALUDE_village_population_l2046_204647

/-- If 80% of a village's population is 64,000, then the total population is 80,000. -/
theorem village_population (population : ℕ) (h : (80 : ℕ) * population = 100 * 64000) :
  population = 80000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l2046_204647


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l2046_204661

theorem floor_plus_self_unique_solution :
  ∃! s : ℝ, ⌊s⌋ + s = 20.75 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l2046_204661


namespace NUMINAMATH_CALUDE_circle_radius_from_circumference_and_area_l2046_204651

/-- Given a circle with specified circumference and area, prove its radius is approximately 4 cm. -/
theorem circle_radius_from_circumference_and_area 
  (circumference : ℝ) 
  (area : ℝ) 
  (h_circumference : circumference = 25.132741228718345)
  (h_area : area = 50.26548245743669) :
  ∃ (radius : ℝ), abs (radius - 4) < 0.0001 ∧ 
    circumference = 2 * Real.pi * radius ∧ 
    area = Real.pi * radius ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_circumference_and_area_l2046_204651


namespace NUMINAMATH_CALUDE_sin_2theta_value_l2046_204616

/-- Given vectors a, b, c, and the condition that (2a - b) is parallel to c, 
    prove that sin(2θ) = -12/13 --/
theorem sin_2theta_value (θ : ℝ) 
  (a b c : ℝ × ℝ)
  (ha : a = (Real.sin θ, 1))
  (hb : b = (-Real.sin θ, 0))
  (hc : c = (Real.cos θ, -1))
  (h_parallel : ∃ (k : ℝ), (2 • a - b) = k • c) :
  Real.sin (2 * θ) = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l2046_204616


namespace NUMINAMATH_CALUDE_power_equation_solution_l2046_204696

theorem power_equation_solution : 2^90 * 8^90 = 64^(90 - 30) := by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2046_204696


namespace NUMINAMATH_CALUDE_lens_focal_length_theorem_l2046_204600

/-- Represents a thin lens with a parallel beam of light falling normally on it. -/
structure ThinLens where
  focal_length : ℝ

/-- Represents a screen that can be placed at different distances from the lens. -/
structure Screen where
  distance : ℝ
  spot_diameter : ℝ

/-- Checks if the spot diameter remains constant when the screen is moved. -/
def constant_spot_diameter (lens : ThinLens) (screen1 screen2 : Screen) : Prop :=
  screen1.spot_diameter = screen2.spot_diameter

/-- Theorem stating the possible focal lengths of the lens given the problem conditions. -/
theorem lens_focal_length_theorem (lens : ThinLens) (screen1 screen2 : Screen) :
  screen1.distance = 80 →
  screen2.distance = 40 →
  constant_spot_diameter lens screen1 screen2 →
  lens.focal_length = 100 ∨ lens.focal_length = 60 :=
sorry

end NUMINAMATH_CALUDE_lens_focal_length_theorem_l2046_204600


namespace NUMINAMATH_CALUDE_minimum_horses_and_ponies_l2046_204611

theorem minimum_horses_and_ponies (ponies horses : ℕ) : 
  (3 * ponies % 10 = 0) →  -- 3/10 of ponies have horseshoes
  (5 * (3 * ponies / 10) % 8 = 0) →  -- 5/8 of ponies with horseshoes are from Iceland
  (horses = ponies + 3) →  -- 3 more horses than ponies
  (∀ p h, p < ponies ∨ h < horses → 
    3 * p % 10 ≠ 0 ∨ 
    5 * (3 * p / 10) % 8 ≠ 0 ∨ 
    h ≠ p + 3) →  -- minimality condition
  ponies + horses = 163 := by
sorry

end NUMINAMATH_CALUDE_minimum_horses_and_ponies_l2046_204611


namespace NUMINAMATH_CALUDE_coconut_grove_yield_l2046_204634

/-- Calculates the yield of the third group of trees in a coconut grove --/
theorem coconut_grove_yield (x : ℕ) (Y : ℕ) : x = 6 →
  ((x + 3) * 60 + x * 120 + (x - 3) * Y) / (3 * x) = 100 →
  Y = 180 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_yield_l2046_204634


namespace NUMINAMATH_CALUDE_smallest_overlap_percentage_l2046_204679

theorem smallest_overlap_percentage (smartphone_users laptop_users : ℝ) 
  (h1 : smartphone_users = 90) 
  (h2 : laptop_users = 80) : 
  ∃ (overlap : ℝ), overlap ≥ 70 ∧ 
    ∀ (x : ℝ), x < 70 → smartphone_users + laptop_users - x > 100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_overlap_percentage_l2046_204679


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2046_204639

theorem cube_root_equation_solution (y : ℝ) : 
  (15 * y + (15 * y + 15) ^ (1/3 : ℝ)) ^ (1/3 : ℝ) = 15 → y = 224 := by
sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2046_204639


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2046_204658

-- Define the quadratic function f(x) = ax^2 + bx + c
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties 
  (a b c : ℝ) 
  (h1 : f a b c 1 = -a/2) 
  (h2 : 3*a > 2*c) 
  (h3 : 2*c > 2*b) :
  (a > 0 ∧ -3 < b/a ∧ b/a < -3/4) ∧ 
  (∃ x, 0 < x ∧ x < 2 ∧ f a b c x = 0) ∧
  (∀ x₁ x₂, f a b c x₁ = 0 → f a b c x₂ = 0 → 
    Real.sqrt 2 ≤ |x₁ - x₂| ∧ |x₁ - x₂| < Real.sqrt 57 / 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2046_204658


namespace NUMINAMATH_CALUDE_triple_sharp_72_l2046_204689

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.4 * N + 3

-- Theorem statement
theorem triple_sharp_72 : sharp (sharp (sharp 72)) = 9.288 := by
  sorry

end NUMINAMATH_CALUDE_triple_sharp_72_l2046_204689


namespace NUMINAMATH_CALUDE_multiplication_value_problem_l2046_204614

theorem multiplication_value_problem : 
  ∃ x : ℝ, (4.5 / 6) * x = 9 ∧ x = 12 := by
sorry

end NUMINAMATH_CALUDE_multiplication_value_problem_l2046_204614


namespace NUMINAMATH_CALUDE_calculation_proof_l2046_204668

theorem calculation_proof : -1^4 - (1 - 0.5) * (2 - (-3)^2) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2046_204668


namespace NUMINAMATH_CALUDE_parabola_rotation_180_l2046_204690

/-- Represents a parabola in the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Rotates a parabola 180° around its vertex -/
def rotate_180 (p : Parabola) : Parabola :=
  { a := -p.a, b := p.b }

theorem parabola_rotation_180 (p : Parabola) (h : p = { a := 1/2, b := 1 }) :
  rotate_180 p = { a := -1/2, b := 1 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_rotation_180_l2046_204690


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2046_204601

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 3rd, 5th, and 7th terms of an arithmetic sequence
    where the sum of the 2nd and 8th terms is 10. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 2 + a 8 = 10) : 
  a 3 + a 5 + a 7 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2046_204601


namespace NUMINAMATH_CALUDE_total_eggs_per_week_l2046_204641

/-- Represents the three chicken breeds -/
inductive Breed
  | BCM  -- Black Copper Marans
  | RIR  -- Rhode Island Reds
  | LH   -- Leghorns

/-- Calculates the number of chickens for a given breed -/
def chickenCount (b : Breed) : Nat :=
  match b with
  | Breed.BCM => 125
  | Breed.RIR => 200
  | Breed.LH  => 175

/-- Calculates the number of hens for a given breed -/
def henCount (b : Breed) : Nat :=
  match b with
  | Breed.BCM => 81
  | Breed.RIR => 110
  | Breed.LH  => 105

/-- Represents the egg-laying rates for each breed -/
def eggRates (b : Breed) : List Nat :=
  match b with
  | Breed.BCM => [3, 4, 5]
  | Breed.RIR => [5, 6, 7]
  | Breed.LH  => [6, 7, 8]

/-- Represents the distribution of hens for each egg-laying rate -/
def henDistribution (b : Breed) : List Nat :=
  match b with
  | Breed.BCM => [32, 24, 25]
  | Breed.RIR => [22, 55, 33]
  | Breed.LH  => [26, 47, 32]

/-- Calculates the total eggs produced by a breed per week -/
def eggsByBreed (b : Breed) : Nat :=
  List.sum (List.zipWith (· * ·) (eggRates b) (henDistribution b))

/-- The main theorem: total eggs produced by all hens per week is 1729 -/
theorem total_eggs_per_week :
  (eggsByBreed Breed.BCM) + (eggsByBreed Breed.RIR) + (eggsByBreed Breed.LH) = 1729 := by
  sorry

#eval (eggsByBreed Breed.BCM) + (eggsByBreed Breed.RIR) + (eggsByBreed Breed.LH)

end NUMINAMATH_CALUDE_total_eggs_per_week_l2046_204641


namespace NUMINAMATH_CALUDE_work_hours_per_day_l2046_204675

/-- Proves that working 56 hours over 14 days results in 4 hours of work per day -/
theorem work_hours_per_day (total_hours : ℕ) (total_days : ℕ) (hours_per_day : ℕ) : 
  total_hours = 56 → total_days = 14 → total_hours = total_days * hours_per_day → hours_per_day = 4 := by
  sorry

#check work_hours_per_day

end NUMINAMATH_CALUDE_work_hours_per_day_l2046_204675


namespace NUMINAMATH_CALUDE_square_of_1008_l2046_204626

theorem square_of_1008 : (1008 : ℕ)^2 = 1016064 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1008_l2046_204626


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2046_204624

/-- Given a rectangular field with one side uncovered and three sides fenced, 
    calculate its area. -/
theorem rectangular_field_area 
  (L : ℝ) -- length of the uncovered side
  (fence_length : ℝ) -- total length of fencing for three sides
  (h1 : L = 25) -- the uncovered side is 25 feet
  (h2 : fence_length = 95.4) -- the total fencing required is 95.4 feet
  : L * ((fence_length - L) / 2) = 880 := by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_rectangular_field_area_l2046_204624


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_reciprocal_squares_l2046_204631

theorem cubic_roots_sum_of_reciprocal_squares :
  ∀ a b c : ℝ,
  (∀ x : ℝ, x^3 - 6*x^2 + 11*x - 6 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_reciprocal_squares_l2046_204631


namespace NUMINAMATH_CALUDE_table_tennis_equation_l2046_204653

/-- Represents a table tennis competition -/
structure TableTennisCompetition where
  teams : ℕ
  totalMatches : ℕ
  pairPlaysOneMatch : Bool

/-- The equation for the number of matches in a table tennis competition -/
def matchEquation (c : TableTennisCompetition) : Prop :=
  c.teams * (c.teams - 1) = c.totalMatches * 2

/-- Theorem stating the correct equation for the given competition conditions -/
theorem table_tennis_equation (c : TableTennisCompetition) 
  (h1 : c.pairPlaysOneMatch = true) 
  (h2 : c.totalMatches = 28) : 
  matchEquation c := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_equation_l2046_204653


namespace NUMINAMATH_CALUDE_room_length_l2046_204640

/-- Proves that a rectangular room with given volume, height, and width has a specific length -/
theorem room_length (volume : ℝ) (height : ℝ) (width : ℝ) (length : ℝ) 
  (h_volume : volume = 10000)
  (h_height : height = 10)
  (h_width : width = 10)
  (h_room_volume : volume = length * width * height) :
  length = 100 :=
by sorry

end NUMINAMATH_CALUDE_room_length_l2046_204640


namespace NUMINAMATH_CALUDE_sams_puppies_l2046_204635

theorem sams_puppies (initial_spotted : ℕ) (initial_nonspotted : ℕ) 
  (given_away_spotted : ℕ) (given_away_nonspotted : ℕ) 
  (remaining_spotted : ℕ) (remaining_nonspotted : ℕ) : ℕ :=
  by
  have h1 : initial_spotted = 8 := by sorry
  have h2 : initial_nonspotted = 5 := by sorry
  have h3 : given_away_spotted = 2 := by sorry
  have h4 : given_away_nonspotted = 3 := by sorry
  have h5 : remaining_spotted = 6 := by sorry
  have h6 : remaining_nonspotted = 2 := by sorry
  have h7 : initial_spotted - given_away_spotted = remaining_spotted := by sorry
  have h8 : initial_nonspotted - given_away_nonspotted = remaining_nonspotted := by sorry
  exact initial_spotted + initial_nonspotted

end NUMINAMATH_CALUDE_sams_puppies_l2046_204635


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2046_204671

/-- The perimeter of a triangle with vertices A(3,7), B(-5,2), and C(3,2) is √89 + 13. -/
theorem triangle_perimeter : 
  let A : ℝ × ℝ := (3, 7)
  let B : ℝ × ℝ := (-5, 2)
  let C : ℝ × ℝ := (3, 2)
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B + d B C + d C A = Real.sqrt 89 + 13 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2046_204671


namespace NUMINAMATH_CALUDE_units_digit_G_100_l2046_204666

-- Define the sequence G_n
def G (n : ℕ) : ℕ := 3 * 2^(2^n) + 2

-- Define a function to get the units digit
def units_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_G_100 : units_digit (G 100) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_100_l2046_204666


namespace NUMINAMATH_CALUDE_quadratic_solution_value_l2046_204609

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The solution set of the inequality f(x) < c -/
structure SolutionSet (f : ℝ → ℝ) (c : ℝ) where
  m : ℝ
  property : Set.Ioo m (m + 6) = {x | f x < c}

/-- The theorem stating that c = 9 given the conditions -/
theorem quadratic_solution_value
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h_f : f = QuadraticFunction a b)
  (h_range : Set.range f = Set.Ici 0)
  (h_solution : SolutionSet f c)
  : c = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_value_l2046_204609


namespace NUMINAMATH_CALUDE_jelly_bracelet_cost_l2046_204646

def friends : List String := ["Jessica", "Tori", "Lily", "Patrice"]

def total_spent : ℕ := 44

def total_bracelets : ℕ := (friends.map String.length).sum

theorem jelly_bracelet_cost :
  total_spent / total_bracelets = 2 := by sorry

end NUMINAMATH_CALUDE_jelly_bracelet_cost_l2046_204646


namespace NUMINAMATH_CALUDE_triple_base_quadruple_exponent_l2046_204649

theorem triple_base_quadruple_exponent 
  (a b : ℝ) (y : ℝ) (h1 : b ≠ 0) :
  let r := (3 * a) ^ (4 * b)
  r = a ^ b * y ^ b →
  y = 81 * a ^ 3 := by
sorry

end NUMINAMATH_CALUDE_triple_base_quadruple_exponent_l2046_204649


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2046_204667

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, a.1 = t * b.1 ∧ a.2 = t * b.2

theorem parallel_vectors_k_value :
  ∀ k : ℝ, 
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (k, 4)
  parallel a b → k = -2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2046_204667


namespace NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l2046_204699

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem eighth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_3 : a 3 = 3)
  (h_6 : a 6 = 24) :
  a 8 = 96 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l2046_204699


namespace NUMINAMATH_CALUDE_ellipse_equation_from_line_through_focus_and_vertex_l2046_204680

/-- Represents an ellipse in standard form -/
structure StandardEllipse where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: If a line with equation x - 2y + 2 = 0 passes through a focus and a vertex of an ellipse,
    then the standard equation of the ellipse is either x²/5 + y² = 1 or x²/4 + y²/5 = 1 -/
theorem ellipse_equation_from_line_through_focus_and_vertex 
  (l : Line) 
  (hl : l.a = 1 ∧ l.b = -2 ∧ l.c = 2) 
  (passes_through_focus_and_vertex : ∃ (e : StandardEllipse), 
    (∃ (x y : ℝ), x - 2*y + 2 = 0 ∧ 
      ((x = e.a ∧ y = 0) ∨ (x = 0 ∧ y = e.b) ∨ (x = -e.a ∧ y = 0) ∨ (x = 0 ∧ y = -e.b)) ∧
      ((x^2 / e.a^2 + y^2 / e.b^2 = 1) ∨ (y^2 / e.a^2 + x^2 / e.b^2 = 1)))) :
  ∃ (e : StandardEllipse), (e.a^2 = 5 ∧ e.b^2 = 1) ∨ (e.a^2 = 4 ∧ e.b^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_line_through_focus_and_vertex_l2046_204680


namespace NUMINAMATH_CALUDE_scarf_to_tie_belt_ratio_l2046_204684

-- Define the quantities given in the problem
def ties : ℕ := 34
def belts : ℕ := 40
def black_shirts : ℕ := 63
def white_shirts : ℕ := 42

-- Define the number of jeans based on the given condition
def jeans : ℕ := (2 * (black_shirts + white_shirts)) / 3

-- Define the number of scarves based on the given condition
def scarves : ℕ := jeans - 33

-- Theorem to prove
theorem scarf_to_tie_belt_ratio :
  scarves * 2 = ties + belts := by
  sorry


end NUMINAMATH_CALUDE_scarf_to_tie_belt_ratio_l2046_204684


namespace NUMINAMATH_CALUDE_complex_product_real_solutions_l2046_204677

theorem complex_product_real_solutions (x : ℝ) : 
  (Complex.I : ℂ) * ((x + Complex.I) * ((x + 3 : ℝ) + 2 * Complex.I) * ((x + 5 : ℝ) - Complex.I)).im = 0 ↔ 
  x = -1.5 ∨ x = -5 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_solutions_l2046_204677


namespace NUMINAMATH_CALUDE_quadratic_below_x_axis_iff_a_in_range_l2046_204665

/-- A quadratic function f(x) = ax^2 + 2ax - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x - 2

/-- The property that the graph of f is always below the x-axis -/
def always_below_x_axis (a : ℝ) : Prop :=
  ∀ x, f a x < 0

theorem quadratic_below_x_axis_iff_a_in_range :
  ∀ a : ℝ, always_below_x_axis a ↔ -2 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_below_x_axis_iff_a_in_range_l2046_204665


namespace NUMINAMATH_CALUDE_vector_equation_l2046_204655

-- Define the vector type
variable {V : Type*} [AddCommGroup V]

-- Define points in space
variable (A B C D : V)

-- Define vectors
def vec (X Y : V) : V := Y - X

-- Theorem statement
theorem vector_equation (A B C D : V) :
  vec D A + vec C D - vec C B = vec B A := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_l2046_204655


namespace NUMINAMATH_CALUDE_rachel_lunch_spending_l2046_204685

theorem rachel_lunch_spending (initial_amount : ℝ) 
  (h1 : initial_amount = 200)
  (h2 : ∃ dvd_amount : ℝ, dvd_amount = initial_amount / 2)
  (h3 : ∃ amount_left : ℝ, amount_left = 50) :
  ∃ lunch_fraction : ℝ, lunch_fraction = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rachel_lunch_spending_l2046_204685


namespace NUMINAMATH_CALUDE_sculpture_and_base_height_l2046_204695

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the total height of a sculpture and its base in inches -/
def total_height (sculpture_feet : ℕ) (sculpture_inches : ℕ) (base_inches : ℕ) : ℕ :=
  feet_to_inches sculpture_feet + sculpture_inches + base_inches

/-- Theorem stating that a sculpture of 2 feet 10 inches on an 8-inch base has a total height of 42 inches -/
theorem sculpture_and_base_height :
  total_height 2 10 8 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_and_base_height_l2046_204695


namespace NUMINAMATH_CALUDE_circle_regions_l2046_204673

/-- The number of regions into which n circles divide a plane, 
    where each pair of circles intersects and no three circles 
    intersect at the same point. -/
def f (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem stating the properties of the function f -/
theorem circle_regions (n : ℕ) : 
  n > 0 → 
  (f 3 = 8) ∧ 
  (f n = n^2 - n + 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_regions_l2046_204673


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l2046_204630

/-- Definition of the bow tie operation -/
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

/-- Theorem stating that if 5 ⋈ x = 12, then x = 42 -/
theorem bowtie_equation_solution :
  ∃ x : ℝ, bowtie 5 x = 12 → x = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l2046_204630


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2046_204610

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For any arithmetic sequence, if a₁ ≥ a₂, then a₂² ≥ a₁a₃ -/
theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  a 1 ≥ a 2 → a 2 ^ 2 ≥ a 1 * a 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2046_204610


namespace NUMINAMATH_CALUDE_max_cube_sum_on_unit_circle_l2046_204623

theorem max_cube_sum_on_unit_circle :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 → |x^3| + |y^3| ≤ M) ∧
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ |x^3| + |y^3| = M) := by
sorry

end NUMINAMATH_CALUDE_max_cube_sum_on_unit_circle_l2046_204623


namespace NUMINAMATH_CALUDE_intersection_P_Q_nonempty_intersection_P_R_nonempty_l2046_204629

-- Define the sets P, Q, and R
def P : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}
def Q (a : ℝ) : Set ℝ := {x | a * x^2 - 2*x + 2 > 0}
def R (a : ℝ) : Set ℝ := {x | a * x^2 - 2*x + 2 = 0}

-- Theorem for part 1
theorem intersection_P_Q_nonempty (a : ℝ) : 
  (P ∩ Q a).Nonempty → a > -1/2 :=
sorry

-- Theorem for part 2
theorem intersection_P_R_nonempty (a : ℝ) : 
  (P ∩ R a).Nonempty → a ≥ -1/2 ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_P_Q_nonempty_intersection_P_R_nonempty_l2046_204629


namespace NUMINAMATH_CALUDE_anna_bob_matches_l2046_204674

/-- The number of players in the chess tournament -/
def total_players : ℕ := 12

/-- The number of players in each match -/
def players_per_match : ℕ := 6

/-- The number of players to choose after Anna and Bob are selected -/
def players_to_choose : ℕ := players_per_match - 2

/-- The number of remaining players after Anna and Bob are selected -/
def remaining_players : ℕ := total_players - 2

/-- The number of matches where Anna and Bob play together -/
def matches_together : ℕ := Nat.choose remaining_players players_to_choose

theorem anna_bob_matches :
  matches_together = 210 := by sorry

end NUMINAMATH_CALUDE_anna_bob_matches_l2046_204674


namespace NUMINAMATH_CALUDE_star_value_l2046_204625

-- Define the * operation
def star (a b : ℤ) : ℚ :=
  1 / a + 1 / b

-- State the theorem
theorem star_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 7) (h4 : a * b = 12) :
  star a b = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l2046_204625


namespace NUMINAMATH_CALUDE_ratio_problem_l2046_204622

theorem ratio_problem (A B C D : ℝ) 
  (hA : A = 0.40 * B) 
  (hB : B = 0.25 * C) 
  (hD : D = 0.60 * C) : 
  A / D = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2046_204622


namespace NUMINAMATH_CALUDE_corner_circle_radius_l2046_204682

/-- The radius of a circle placed tangentially to four corner circles in a specific rectangle configuration -/
theorem corner_circle_radius (rectangle_width : ℝ) (rectangle_length : ℝ) 
  (h_width : rectangle_width = 3)
  (h_length : rectangle_length = 4)
  (large_circle_radius : ℝ)
  (h_large_radius : large_circle_radius = 2/3)
  (small_circle_radius : ℝ) :
  small_circle_radius = 1 :=
sorry

end NUMINAMATH_CALUDE_corner_circle_radius_l2046_204682


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2046_204650

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 5) ↔ x ≥ 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2046_204650


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2046_204605

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) (k : ℕ) :
  (∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence property
  a 1 = 0 →                        -- first term is 0
  d ≠ 0 →                          -- common difference is non-zero
  a k = (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7) →  -- sum condition
  k = 22 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2046_204605


namespace NUMINAMATH_CALUDE_sphere_radius_proof_l2046_204628

theorem sphere_radius_proof (a b c : ℝ) : 
  (a + b + c = 40) →
  (2 * a * b + 2 * b * c + 2 * c * a = 512) →
  (∃ r : ℝ, r^2 = 130 ∧ r^2 * 4 = a^2 + b^2 + c^2) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_proof_l2046_204628


namespace NUMINAMATH_CALUDE_pedestrians_collinear_at_most_twice_l2046_204642

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a pedestrian's motion in 2D space -/
structure Pedestrian where
  initial_pos : Point2D
  velocity : Point2D

/-- Three pedestrians walking in straight lines -/
def three_pedestrians (p1 p2 p3 : Pedestrian) : Prop :=
  -- Pedestrians have constant velocities
  ∀ t : ℝ, ∃ (pos1 pos2 pos3 : Point2D),
    pos1 = Point2D.mk (p1.initial_pos.x + p1.velocity.x * t) (p1.initial_pos.y + p1.velocity.y * t) ∧
    pos2 = Point2D.mk (p2.initial_pos.x + p2.velocity.x * t) (p2.initial_pos.y + p2.velocity.y * t) ∧
    pos3 = Point2D.mk (p3.initial_pos.x + p3.velocity.x * t) (p3.initial_pos.y + p3.velocity.y * t)

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- The main theorem -/
theorem pedestrians_collinear_at_most_twice
  (p1 p2 p3 : Pedestrian)
  (h_not_initially_collinear : ¬are_collinear p1.initial_pos p2.initial_pos p3.initial_pos)
  (h_walking : three_pedestrians p1 p2 p3) :
  ∃ (t1 t2 : ℝ), ∀ t : ℝ,
    are_collinear
      (Point2D.mk (p1.initial_pos.x + p1.velocity.x * t) (p1.initial_pos.y + p1.velocity.y * t))
      (Point2D.mk (p2.initial_pos.x + p2.velocity.x * t) (p2.initial_pos.y + p2.velocity.y * t))
      (Point2D.mk (p3.initial_pos.x + p3.velocity.x * t) (p3.initial_pos.y + p3.velocity.y * t))
    → t = t1 ∨ t = t2 :=
  sorry

end NUMINAMATH_CALUDE_pedestrians_collinear_at_most_twice_l2046_204642


namespace NUMINAMATH_CALUDE_equation_solution_l2046_204621

theorem equation_solution (x : ℚ) (h1 : x ≠ 0) (h2 : x ≠ -5) :
  (2 * x / (x + 5) - 1 = (x + 5) / x) ↔ (x = -5/3) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2046_204621


namespace NUMINAMATH_CALUDE_polynomial_equality_l2046_204602

-- Define the polynomial (x+y)^8
def polynomial (x y : ℝ) : ℝ := (x + y)^8

-- Define the third term of the expansion
def third_term (x y : ℝ) : ℝ := 28 * x^6 * y^2

-- Define the fourth term of the expansion
def fourth_term (x y : ℝ) : ℝ := 56 * x^5 * y^3

theorem polynomial_equality (p q : ℝ) :
  p > 0 ∧ q > 0 ∧ p + q = 1 ∧ third_term p q = fourth_term p q → p = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_equality_l2046_204602


namespace NUMINAMATH_CALUDE_bf_equals_ce_l2046_204664

-- Define the triangle ABC
variable (A B C : Point)

-- Define D as the foot of the angle bisector from A
def D : Point := sorry

-- Define E as the intersection of circumcircle ABD with AC
def E : Point := sorry

-- Define F as the intersection of circumcircle ADC with AB
def F : Point := sorry

-- Theorem statement
theorem bf_equals_ce : BF = CE := by sorry

end NUMINAMATH_CALUDE_bf_equals_ce_l2046_204664


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2046_204645

/-- Given a line passing through points (2, -3) and (5, 6), prove that m + b = -6 --/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b ↔ (x = 2 ∧ y = -3) ∨ (x = 5 ∧ y = 6)) →
  m + b = -6 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2046_204645


namespace NUMINAMATH_CALUDE_library_book_increase_l2046_204698

theorem library_book_increase (N : ℕ) : 
  N > 0 ∧ 
  (N * 1.004 * 1.008 : ℝ) < 50000 →
  ⌊(N * 1.004 * 1.008 - N * 1.004 : ℝ)⌋ = 251 :=
by sorry

end NUMINAMATH_CALUDE_library_book_increase_l2046_204698


namespace NUMINAMATH_CALUDE_correct_attitude_towards_superstitions_l2046_204606

/-- Represents different types of online superstitions -/
inductive OnlineSuperstition
  | AstrologicalFate
  | HoroscopeInterpretation
  | NorthStarBook
  | DreamInterpretation

/-- Represents possible attitudes towards online superstitions -/
inductive Attitude
  | Accept
  | StayAway
  | RespectDiversity
  | ImproveDiscernment

/-- Defines the correct attitude for teenage students -/
def correct_attitude : Attitude := Attitude.ImproveDiscernment

/-- Theorem stating the correct attitude towards online superstitions -/
theorem correct_attitude_towards_superstitions :
  ∀ (s : OnlineSuperstition), correct_attitude = Attitude.ImproveDiscernment :=
by sorry

end NUMINAMATH_CALUDE_correct_attitude_towards_superstitions_l2046_204606


namespace NUMINAMATH_CALUDE_greatest_four_digit_satisfying_conditions_l2046_204620

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def product_of_first_n (n : ℕ) : ℕ := Nat.factorial n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_satisfying_conditions :
  is_four_digit 9999 ∧
  ¬(product_of_first_n 9999 % sum_of_first_n 9999 = 0) ∧
  is_perfect_square (9999 + 1) ∧
  ∀ n : ℕ, is_four_digit n →
    n > 9999 ∨
    (product_of_first_n n % sum_of_first_n n = 0) ∨
    ¬(is_perfect_square (n + 1)) :=
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_satisfying_conditions_l2046_204620


namespace NUMINAMATH_CALUDE_negative_one_less_than_negative_two_thirds_l2046_204644

theorem negative_one_less_than_negative_two_thirds : -1 < -(2/3) := by
  sorry

end NUMINAMATH_CALUDE_negative_one_less_than_negative_two_thirds_l2046_204644


namespace NUMINAMATH_CALUDE_divisible_by_21_l2046_204607

theorem divisible_by_21 (N : Finset ℕ) 
  (h_card : N.card = 46)
  (h_div_3 : (N.filter (fun n => n % 3 = 0)).card = 35)
  (h_div_7 : (N.filter (fun n => n % 7 = 0)).card = 12) :
  ∃ n ∈ N, n % 21 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_21_l2046_204607


namespace NUMINAMATH_CALUDE_chips_cost_calculation_l2046_204654

/-- Given the original cost and discount of chips, calculate the actual amount spent -/
theorem chips_cost_calculation (original_cost discount : ℚ) 
  (h1 : original_cost = 35)
  (h2 : discount = 17) :
  original_cost - discount = 18 := by
  sorry

end NUMINAMATH_CALUDE_chips_cost_calculation_l2046_204654


namespace NUMINAMATH_CALUDE_largest_angle_convex_hexagon_l2046_204619

theorem largest_angle_convex_hexagon (x : ℝ) :
  (x + 2) + (2 * x + 3) + (3 * x + 4) + (4 * x + 5) + (5 * x + 6) + (6 * x + 7) = 720 →
  max (x + 2) (max (2 * x + 3) (max (3 * x + 4) (max (4 * x + 5) (max (5 * x + 6) (6 * x + 7))))) = 205 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_convex_hexagon_l2046_204619


namespace NUMINAMATH_CALUDE_fraction_condition_l2046_204613

theorem fraction_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, a / b > 1 → b / a < 1) ∧
  (∃ a b, b / a < 1 ∧ a / b ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_fraction_condition_l2046_204613


namespace NUMINAMATH_CALUDE_romeo_chocolate_bars_l2046_204656

theorem romeo_chocolate_bars : 
  ∀ (buy_cost sell_total packaging_cost profit num_bars : ℕ),
    buy_cost = 5 →
    sell_total = 90 →
    packaging_cost = 2 →
    profit = 55 →
    num_bars * (buy_cost + packaging_cost) + profit = sell_total →
    num_bars = 5 := by
  sorry

end NUMINAMATH_CALUDE_romeo_chocolate_bars_l2046_204656


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l2046_204686

theorem square_perimeter_sum (a b : ℝ) (h1 : a^2 + b^2 = 85) (h2 : a^2 - b^2 = 45) :
  4*a + 4*b = 4*(Real.sqrt 65 + 2*Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l2046_204686


namespace NUMINAMATH_CALUDE_smallest_circle_equation_l2046_204670

/-- The equation of the circle with the smallest area that is tangent to the line 3x + 4y + 3 = 0
    and has its center on the curve y = 3/x (x > 0) -/
theorem smallest_circle_equation (x y : ℝ) :
  (∀ a : ℝ, a > 0 → ∃ r : ℝ, r > 0 ∧
    (∀ x₀ y₀ : ℝ, (x₀ - a)^2 + (y₀ - 3/a)^2 = r^2 →
      (3*x₀ + 4*y₀ + 3 = 0 → False) ∧
      (3*x₀ + 4*y₀ + 3 ≠ 0 → (3*x₀ + 4*y₀ + 3)^2 > 25*r^2))) →
  (x - 2)^2 + (y - 3/2)^2 = 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_circle_equation_l2046_204670


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2046_204688

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    if the triangle formed by its left vertex, right vertex, and top vertex
    is an isosceles triangle with base angle 30°, then its eccentricity is √6/3 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (b / a = Real.sqrt 3 / 3) → 
  Real.sqrt (1 - (b / a)^2) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2046_204688


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2046_204662

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B : Set ℝ := {x : ℝ | x > -1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 4} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2046_204662


namespace NUMINAMATH_CALUDE_all_statements_imply_target_l2046_204637

theorem all_statements_imply_target (p q r : Prop) :
  (p ∧ ¬q ∧ ¬r → ((p → q) → ¬r)) ∧
  (¬p ∧ ¬q ∧ ¬r → ((p → q) → ¬r)) ∧
  (p ∧ q ∧ ¬r → ((p → q) → ¬r)) ∧
  (¬p ∧ q ∧ ¬r → ((p → q) → ¬r)) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_imply_target_l2046_204637


namespace NUMINAMATH_CALUDE_john_squat_increase_l2046_204687

/-- The additional weight John added to his squat after training -/
def additional_weight : ℝ := 265

/-- John's initial squat weight in pounds -/
def initial_weight : ℝ := 135

/-- The factor by which the magical bracer increases strength -/
def strength_increase_factor : ℝ := 7

/-- John's final squat weight in pounds -/
def final_weight : ℝ := 2800

theorem john_squat_increase :
  (initial_weight + additional_weight) * strength_increase_factor = final_weight :=
sorry

end NUMINAMATH_CALUDE_john_squat_increase_l2046_204687


namespace NUMINAMATH_CALUDE_russian_dolls_discount_l2046_204627

theorem russian_dolls_discount (original_price : ℝ) (original_quantity : ℕ) (discount_rate : ℝ) :
  original_price = 4 →
  original_quantity = 15 →
  discount_rate = 0.2 →
  ⌊(original_price * original_quantity) / (original_price * (1 - discount_rate))⌋ = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_russian_dolls_discount_l2046_204627


namespace NUMINAMATH_CALUDE_prime_product_sum_squared_sum_l2046_204683

theorem prime_product_sum_squared_sum (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a * b * c = 5 * (a + b + c) →
  a^2 + b^2 + c^2 = 78 :=
by sorry

end NUMINAMATH_CALUDE_prime_product_sum_squared_sum_l2046_204683


namespace NUMINAMATH_CALUDE_sum_to_target_l2046_204603

theorem sum_to_target : ∃ x : ℝ, 0.003 + 0.158 + x = 2.911 ∧ x = 2.750 := by
  sorry

end NUMINAMATH_CALUDE_sum_to_target_l2046_204603


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2046_204612

-- Problem 1
theorem problem_1 : 1/2 + (-2/3) + 4/5 + (-1/2) + (-1/3) = -1/5 := by sorry

-- Problem 2
theorem problem_2 : 2 * (-3)^3 - 4 * (-3) + 15 = -27 := by sorry

-- Problem 3
theorem problem_3 : (1/8 - 1/3 + 1 + 1/6) * (-48) = -46 := by sorry

-- Problem 4
theorem problem_4 : -2^4 - 32 / ((-2)^3 + 4) = -8 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2046_204612


namespace NUMINAMATH_CALUDE_extended_segment_endpoint_l2046_204615

/-- Given a segment AB with endpoints A(3, 3) and B(15, 9), extended through B to point C
    such that BC = 1/2 * AB, the coordinates of point C are (21, 12). -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (3, 3) → B = (15, 9) → 
  (C.1 - B.1, C.2 - B.2) = (1/2 * (B.1 - A.1), 1/2 * (B.2 - A.2)) →
  C = (21, 12) := by
  sorry

end NUMINAMATH_CALUDE_extended_segment_endpoint_l2046_204615


namespace NUMINAMATH_CALUDE_markers_problem_l2046_204694

theorem markers_problem (initial_markers : ℕ) (markers_per_box : ℕ) (final_markers : ℕ) :
  initial_markers = 32 →
  markers_per_box = 9 →
  final_markers = 86 →
  (final_markers - initial_markers) / markers_per_box = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_markers_problem_l2046_204694


namespace NUMINAMATH_CALUDE_complex_number_theorem_l2046_204638

theorem complex_number_theorem (z : ℂ) :
  (z^2).im = 0 ∧ Complex.abs (z - Complex.I) = 1 → z = 0 ∨ z = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l2046_204638


namespace NUMINAMATH_CALUDE_square_root_difference_l2046_204681

theorem square_root_difference : Real.sqrt (49 + 36) - Real.sqrt (36 - 0) = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_difference_l2046_204681


namespace NUMINAMATH_CALUDE_gcd_of_45_and_75_l2046_204632

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_and_75_l2046_204632


namespace NUMINAMATH_CALUDE_jack_afternoon_letters_l2046_204692

/-- The number of letters Jack received in the morning -/
def morning_letters : ℕ := 8

/-- The number of letters Jack received in the afternoon -/
def afternoon_letters : ℕ := morning_letters - 1

theorem jack_afternoon_letters : afternoon_letters = 7 := by
  sorry

end NUMINAMATH_CALUDE_jack_afternoon_letters_l2046_204692


namespace NUMINAMATH_CALUDE_opposite_equal_roots_iff_l2046_204672

/-- The equation has roots that are numerically equal but of opposite signs -/
def has_opposite_equal_roots (d e f n : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, y₁ ≠ 0 ∧ y₂ = -y₁ ∧
  (y₁^2 + 2*d*y₁) / (e*y₁ + f) = n / (n - 2) ∧
  (y₂^2 + 2*d*y₂) / (e*y₂ + f) = n / (n - 2)

/-- The main theorem -/
theorem opposite_equal_roots_iff (d e f : ℝ) :
  ∀ n : ℝ, has_opposite_equal_roots d e f n ↔ n = 4*d / (2*d - e) :=
sorry

end NUMINAMATH_CALUDE_opposite_equal_roots_iff_l2046_204672


namespace NUMINAMATH_CALUDE_congruence_in_range_l2046_204676

theorem congruence_in_range : 
  ∀ n : ℤ, 10 ≤ n ∧ n ≤ 20 ∧ n ≡ 12345 [ZMOD 7] → n = 11 ∨ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_congruence_in_range_l2046_204676


namespace NUMINAMATH_CALUDE_inequality_proof_l2046_204633

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2046_204633


namespace NUMINAMATH_CALUDE_bathtub_capacity_l2046_204652

/-- The capacity of a bathtub given tap flow rate, filling time, and drain leak rate -/
theorem bathtub_capacity 
  (tap_flow : ℝ)  -- Tap flow rate in liters per minute
  (fill_time : ℝ)  -- Filling time in minutes
  (leak_rate : ℝ)  -- Drain leak rate in liters per minute
  (h1 : tap_flow = 21 / 6)  -- Tap flow rate condition
  (h2 : fill_time = 22.5)  -- Filling time condition
  (h3 : leak_rate = 0.3)  -- Drain leak rate condition
  : tap_flow * fill_time - leak_rate * fill_time = 72 := by
  sorry


end NUMINAMATH_CALUDE_bathtub_capacity_l2046_204652
