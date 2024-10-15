import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3024_302410

/-- The value of a for which a line with equation ρsin(θ+ π/3)=a is tangent to a circle with equation ρ = 2sinθ in the polar coordinate system -/
theorem tangent_line_to_circle (a : ℝ) : 
  (∃ θ ρ, ρ = 2 * Real.sin θ ∧ ρ * Real.sin (θ + π/3) = a ∧ 
   ∀ θ' ρ', ρ' = 2 * Real.sin θ' → ρ' * Real.sin (θ' + π/3) ≠ a ∨ (θ' = θ ∧ ρ' = ρ)) →
  a = 3/2 ∨ a = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3024_302410


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3024_302462

theorem complex_number_quadrant (z : ℂ) (h : (2 - I) * z = 5) :
  0 < z.re ∧ 0 < z.im := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3024_302462


namespace NUMINAMATH_CALUDE_prime_equation_solution_l3024_302451

theorem prime_equation_solution (p q r : ℕ) (A : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
  (2 * p * q * r + 50 * p * q = A) ∧
  (7 * p * q * r + 55 * p * r = A) ∧
  (8 * p * q * r + 12 * q * r = A) →
  A = 1980 := by
sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l3024_302451


namespace NUMINAMATH_CALUDE_exists_m_with_infinite_solutions_l3024_302460

/-- The equation we're considering -/
def equation (m a b c : ℕ+) : Prop :=
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / (a * b * c) = m / (a + b + c)

/-- The existence of m with infinitely many solutions -/
theorem exists_m_with_infinite_solutions :
  ∃ m : ℕ+, ∀ n : ℕ, ∃ a b c : ℕ+, a > n ∧ b > n ∧ c > n ∧ equation m a b c :=
sorry

end NUMINAMATH_CALUDE_exists_m_with_infinite_solutions_l3024_302460


namespace NUMINAMATH_CALUDE_evaluate_expression_l3024_302453

theorem evaluate_expression : -(16 / 4 * 7 - 50 + 5 * 7) = -13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3024_302453


namespace NUMINAMATH_CALUDE_value_of_expression_l3024_302436

theorem value_of_expression (a b c d : ℝ) 
  (h1 : a - b = 3) 
  (h2 : c + d = 2) : 
  (a + c) - (b - d) = 5 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l3024_302436


namespace NUMINAMATH_CALUDE_intersection_M_N_l3024_302417

def M : Set ℝ := {x | |x| ≤ 1}
def N : Set ℝ := {x | x^2 - x < 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3024_302417


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l3024_302484

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2 * p.2, 2 * p.1 - p.2)

-- Theorem statement
theorem preimage_of_3_1 :
  ∃ (p : ℝ × ℝ), f p = (3, 1) ∧ p = (1, 1) :=
by
  sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l3024_302484


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l3024_302461

theorem power_fraction_simplification :
  (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l3024_302461


namespace NUMINAMATH_CALUDE_sqrt_360000_equals_600_l3024_302400

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_equals_600_l3024_302400


namespace NUMINAMATH_CALUDE_larger_cube_volume_l3024_302464

-- Define the volume of a smaller cube
def small_cube_volume : ℝ := 8

-- Define the number of smaller cubes
def num_small_cubes : ℕ := 2

-- Theorem statement
theorem larger_cube_volume :
  ∀ (small_edge : ℝ) (large_edge : ℝ),
  small_edge > 0 →
  large_edge > 0 →
  small_edge^3 = small_cube_volume →
  num_small_cubes * small_edge = large_edge →
  large_edge^3 = 64 := by
sorry

end NUMINAMATH_CALUDE_larger_cube_volume_l3024_302464


namespace NUMINAMATH_CALUDE_room_length_is_ten_l3024_302442

/-- Proves that the length of a rectangular room is 10 meters given specific conditions. -/
theorem room_length_is_ten (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 4.75 →
  total_cost = 42750 →
  paving_rate = 900 →
  total_cost / paving_rate / width = 10 := by
  sorry


end NUMINAMATH_CALUDE_room_length_is_ten_l3024_302442


namespace NUMINAMATH_CALUDE_keychain_manufacturing_cost_l3024_302437

theorem keychain_manufacturing_cost 
  (P : ℝ) -- Selling price
  (h1 : P > 0) -- Selling price is positive
  (h2 : P - 0.5 * P = 50) -- New manufacturing cost is $50
  : P - 0.4 * P = 60 := by
  sorry

end NUMINAMATH_CALUDE_keychain_manufacturing_cost_l3024_302437


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l3024_302487

/-- A dodecahedron is a polyhedron with 12 pentagonal faces and 20 vertices,
    where three faces meet at each vertex. -/
structure Dodecahedron where
  faces : Nat
  vertices : Nat
  faces_per_vertex : Nat
  faces_are_pentagonal : faces = 12
  vertex_count : vertices = 20
  three_faces_per_vertex : faces_per_vertex = 3

/-- An interior diagonal of a dodecahedron is a segment connecting two vertices
    which do not lie on the same face. -/
def interior_diagonal (d : Dodecahedron) := Unit

/-- The number of interior diagonals in a dodecahedron -/
def num_interior_diagonals (d : Dodecahedron) : Nat :=
  (d.vertices * (d.vertices - 1 - d.faces_per_vertex)) / 2

/-- Theorem stating that a dodecahedron has 160 interior diagonals -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  num_interior_diagonals d = 160 := by
  sorry

#check dodecahedron_interior_diagonals

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l3024_302487


namespace NUMINAMATH_CALUDE_polar_to_hyperbola_l3024_302495

/-- Theorem: The polar equation ρ² cos(2θ) = 1 represents a hyperbola in Cartesian coordinates -/
theorem polar_to_hyperbola (ρ θ x y : ℝ) : 
  (ρ^2 * (Real.cos (2 * θ)) = 1) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) → 
  (x^2 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_hyperbola_l3024_302495


namespace NUMINAMATH_CALUDE_zephyria_license_plates_l3024_302421

/-- The number of letters in the alphabet. -/
def num_letters : ℕ := 26

/-- The number of digits (0-9). -/
def num_digits : ℕ := 10

/-- The number of letters in a Zephyrian license plate. -/
def letters_in_plate : ℕ := 3

/-- The number of digits in a Zephyrian license plate. -/
def digits_in_plate : ℕ := 4

/-- The total number of valid license plates in Zephyria. -/
def total_license_plates : ℕ := num_letters ^ letters_in_plate * num_digits ^ digits_in_plate

theorem zephyria_license_plates :
  total_license_plates = 175760000 := by
  sorry

end NUMINAMATH_CALUDE_zephyria_license_plates_l3024_302421


namespace NUMINAMATH_CALUDE_smallest_square_division_smallest_square_division_is_two_total_squares_l3024_302412

theorem smallest_square_division (n : ℕ) : n > 0 ∧ 4*n - 4 = 2*n → n ≥ 2 :=
by sorry

theorem smallest_square_division_is_two : 
  ∃ (n : ℕ), n > 0 ∧ 4*n - 4 = 2*n ∧ ∀ (m : ℕ), (m > 0 ∧ 4*m - 4 = 2*m) → n ≤ m :=
by sorry

theorem total_squares (n : ℕ) : n > 0 ∧ 4*n - 4 = 2*n → n^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_division_smallest_square_division_is_two_total_squares_l3024_302412


namespace NUMINAMATH_CALUDE_ball_max_height_l3024_302416

/-- The height of the ball as a function of time -/
def h (t : ℝ) : ℝ := -16 * t^2 + 80 * t + 35

/-- The maximum height reached by the ball -/
theorem ball_max_height : ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 135 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l3024_302416


namespace NUMINAMATH_CALUDE_gloria_ticket_boxes_l3024_302414

/-- Given that Gloria has 45 tickets and each box holds 5 tickets,
    prove that the number of boxes Gloria has is 9. -/
theorem gloria_ticket_boxes : ∀ (total_tickets boxes_count tickets_per_box : ℕ),
  total_tickets = 45 →
  tickets_per_box = 5 →
  total_tickets = boxes_count * tickets_per_box →
  boxes_count = 9 := by
  sorry

end NUMINAMATH_CALUDE_gloria_ticket_boxes_l3024_302414


namespace NUMINAMATH_CALUDE_inheritance_solution_l3024_302475

/-- Represents the inheritance problem with given conditions --/
def inheritance_problem (total : ℝ) : Prop :=
  ∃ (x : ℝ),
    x > 0 ∧
    (total - x) > 0 ∧
    0.05 * x + 0.065 * (total - x) = 227 ∧
    total - x = 1800

/-- The theorem stating the solution to the inheritance problem --/
theorem inheritance_solution :
  ∃ (total : ℝ), inheritance_problem total ∧ total = 4000 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_solution_l3024_302475


namespace NUMINAMATH_CALUDE_gdp_2010_calculation_gdp_2010_l3024_302458

def gdp_2008 : ℝ := 1050
def growth_rate : ℝ := 0.132

theorem gdp_2010_calculation : 
  gdp_2008 * (1 + growth_rate)^2 = gdp_2008 * (1 + growth_rate) * (1 + growth_rate) :=
by sorry

theorem gdp_2010 : ℝ := gdp_2008 * (1 + growth_rate)^2

end NUMINAMATH_CALUDE_gdp_2010_calculation_gdp_2010_l3024_302458


namespace NUMINAMATH_CALUDE_footprint_calculation_l3024_302448

/-- Calculates the total number of footprints left by Pogo and Grimzi -/
def total_footprints (pogo_rate : ℚ) (grimzi_rate : ℚ) (distance : ℚ) : ℚ :=
  pogo_rate * distance + grimzi_rate * distance

/-- Theorem stating the total number of footprints left by Pogo and Grimzi -/
theorem footprint_calculation :
  let pogo_rate : ℚ := 4
  let grimzi_rate : ℚ := 1/2
  let distance : ℚ := 6000
  total_footprints pogo_rate grimzi_rate distance = 27000 := by
sorry

#eval total_footprints 4 (1/2) 6000

end NUMINAMATH_CALUDE_footprint_calculation_l3024_302448


namespace NUMINAMATH_CALUDE_sequence_product_l3024_302401

/-- Given that (-9, a, -1) is an arithmetic sequence and (-9, m, b, n, -1) is a geometric sequence,
    prove that ab = 5. -/
theorem sequence_product (a m b n : ℝ) : 
  ((-9 : ℝ) - a = a - (-1 : ℝ)) →  -- arithmetic sequence condition
  (m / (-9 : ℝ) = b / m) →         -- geometric sequence condition for first two terms
  (b / m = n / b) →                -- geometric sequence condition for middle terms
  (n / b = (-1 : ℝ) / n) →         -- geometric sequence condition for last two terms
  a * b = 5 := by
sorry

end NUMINAMATH_CALUDE_sequence_product_l3024_302401


namespace NUMINAMATH_CALUDE_smallest_divisible_fraction_l3024_302423

def fraction1 : Rat := 6 / 7
def fraction2 : Rat := 5 / 14
def fraction3 : Rat := 10 / 21

def smallest_fraction : Rat := 1 / 42

theorem smallest_divisible_fraction :
  (∀ r : Rat, (fraction1 ∣ r ∧ fraction2 ∣ r ∧ fraction3 ∣ r) → smallest_fraction ≤ r) ∧
  (fraction1 ∣ smallest_fraction ∧ fraction2 ∣ smallest_fraction ∧ fraction3 ∣ smallest_fraction) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_fraction_l3024_302423


namespace NUMINAMATH_CALUDE_probability_calculation_l3024_302443

/-- Represents a bag of bills -/
structure Bag where
  tens : ℕ
  fives : ℕ
  ones : ℕ

/-- Calculates the total value of bills in a bag -/
def bagValue (b : Bag) : ℕ := 10 * b.tens + 5 * b.fives + b.ones

/-- Calculates the number of ways to choose 2 bills from a bag -/
def chooseTwo (b : Bag) : ℕ := (b.tens + b.fives + b.ones) * (b.tens + b.fives + b.ones - 1) / 2

/-- Calculates the probability of the sum of remaining bills in bag A being greater than the sum of remaining bills in bag B -/
def probabilityAGreaterThanB (bagA bagB : Bag) : ℚ :=
  let totalOutcomes := chooseTwo bagA * chooseTwo bagB
  let favorableOutcomes := 3 * 18  -- This is a simplification based on the problem's specific conditions
  ↑favorableOutcomes / ↑totalOutcomes

theorem probability_calculation (bagA bagB : Bag) 
  (hA : bagA = ⟨2, 0, 3⟩) 
  (hB : bagB = ⟨0, 4, 3⟩) : 
  probabilityAGreaterThanB bagA bagB = 9/35 := by
  sorry

#eval probabilityAGreaterThanB ⟨2, 0, 3⟩ ⟨0, 4, 3⟩

end NUMINAMATH_CALUDE_probability_calculation_l3024_302443


namespace NUMINAMATH_CALUDE_vehicles_with_only_cd_player_l3024_302497

/-- Represents the percentage of vehicles with specific features -/
structure VehicleFeatures where
  power_windows : ℝ
  anti_lock_brakes : ℝ
  cd_player : ℝ
  power_windows_and_anti_lock : ℝ
  anti_lock_and_cd : ℝ
  power_windows_and_cd : ℝ

/-- The theorem stating the percentage of vehicles with only a CD player -/
theorem vehicles_with_only_cd_player (v : VehicleFeatures)
  (h1 : v.power_windows = 60)
  (h2 : v.anti_lock_brakes = 25)
  (h3 : v.cd_player = 75)
  (h4 : v.power_windows_and_anti_lock = 10)
  (h5 : v.anti_lock_and_cd = 15)
  (h6 : v.power_windows_and_cd = 22)
  (h7 : v.power_windows_and_anti_lock + v.anti_lock_and_cd + v.power_windows_and_cd ≤ v.cd_player) :
  v.cd_player - (v.power_windows_and_cd + v.anti_lock_and_cd) = 38 := by
  sorry

end NUMINAMATH_CALUDE_vehicles_with_only_cd_player_l3024_302497


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3024_302449

def f (n : ℕ) (a : ℝ) : ℝ := (-2) ^ (n - 1) * a ^ n

theorem tenth_term_of_sequence (a : ℝ) : f 10 a = -2^9 * a^10 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3024_302449


namespace NUMINAMATH_CALUDE_sequence_sum_l3024_302432

theorem sequence_sum (a b c d : ℕ+) : 
  (∃ r : ℚ, r > 1 ∧ b = a * r ∧ c = a * r^2) →  -- geometric progression
  (∃ k : ℤ, c - b = k ∧ d - c = k) →            -- arithmetic progression
  d = a + 40 →                                  -- difference between first and last term
  a + b + c + d = 104 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3024_302432


namespace NUMINAMATH_CALUDE_smoothie_proportion_l3024_302472

/-- Given that 13 smoothies can be made from 3 bananas, prove that 65 smoothies can be made from 15 bananas. -/
theorem smoothie_proportion (make_smoothie : ℕ → ℕ) 
    (h : make_smoothie 3 = 13) : make_smoothie 15 = 65 := by
  sorry

#check smoothie_proportion

end NUMINAMATH_CALUDE_smoothie_proportion_l3024_302472


namespace NUMINAMATH_CALUDE_triangle_with_altitudes_9_12_18_has_right_angle_l3024_302439

/-- A triangle with altitudes of lengths 9, 12, and 18 has a right angle as its largest angle. -/
theorem triangle_with_altitudes_9_12_18_has_right_angle :
  ∀ (a b c : ℝ) (α β γ : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c →
  α + β + γ = π →
  9 * a = 12 * b ∧ 12 * b = 18 * c →
  (∃ (h : ℝ), h * a = 9 ∧ h * b = 12 ∧ h * c = 18) →
  max α (max β γ) = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_altitudes_9_12_18_has_right_angle_l3024_302439


namespace NUMINAMATH_CALUDE_photographer_choices_l3024_302444

theorem photographer_choices (n : ℕ) (k₁ k₂ : ℕ) (h₁ : n = 7) (h₂ : k₁ = 4) (h₃ : k₂ = 5) :
  Nat.choose n k₁ + Nat.choose n k₂ = 56 := by
  sorry

end NUMINAMATH_CALUDE_photographer_choices_l3024_302444


namespace NUMINAMATH_CALUDE_weighted_inequality_l3024_302415

theorem weighted_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_one : a + b + c = 1) :
  (a + 2*a*b + 2*a*c + b*c)^a * (b + 2*b*c + 2*b*a + c*a)^b * (c + 2*c*a + 2*c*b + a*b)^c ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_weighted_inequality_l3024_302415


namespace NUMINAMATH_CALUDE_absolute_sum_sequence_minimum_sum_l3024_302467

/-- An absolute sum sequence with given initial term and absolute public sum. -/
def AbsoluteSumSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  fun n => if n = 1 then a₁ else sorry

/-- The sum of the first n terms of an absolute sum sequence. -/
def SequenceSum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem absolute_sum_sequence_minimum_sum :
  ∀ a : ℕ → ℝ,
  (a 1 = 2) →
  (∀ n : ℕ, |a (n + 1)| + |a n| = 3) →
  SequenceSum a 2019 ≥ -3025 ∧
  ∃ a : ℕ → ℝ, (a 1 = 2) ∧ (∀ n : ℕ, |a (n + 1)| + |a n| = 3) ∧ SequenceSum a 2019 = -3025 :=
by sorry

end NUMINAMATH_CALUDE_absolute_sum_sequence_minimum_sum_l3024_302467


namespace NUMINAMATH_CALUDE_football_players_count_l3024_302499

theorem football_players_count (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 38)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 9)
  (h5 : total = (football - both) + (tennis - both) + both + neither) :
  football = 26 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l3024_302499


namespace NUMINAMATH_CALUDE_opposite_of_negative_eleven_l3024_302411

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem statement
theorem opposite_of_negative_eleven : opposite (-11) = 11 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_eleven_l3024_302411


namespace NUMINAMATH_CALUDE_pet_ownership_problem_l3024_302492

/-- Represents the number of students in each section of the Venn diagram -/
structure PetOwnership where
  dogs_only : ℕ
  cats_only : ℕ
  other_only : ℕ
  dogs_cats : ℕ
  cats_other : ℕ
  dogs_other : ℕ
  all_three : ℕ

/-- The main theorem to prove -/
theorem pet_ownership_problem (po : PetOwnership) : po.all_three = 4 :=
  by
  have total_students : ℕ := 40
  have dog_fraction : Rat := 5 / 8
  have cat_fraction : Rat := 1 / 4
  have other_pet_count : ℕ := 8
  have no_pet_count : ℕ := 4

  have dogs_only : po.dogs_only = 15 := by sorry
  have cats_only : po.cats_only = 3 := by sorry
  have other_only : po.other_only = 2 := by sorry

  have dog_eq : po.dogs_only + po.dogs_cats + po.dogs_other + po.all_three = (total_students : ℚ) * dog_fraction := by sorry
  have cat_eq : po.cats_only + po.dogs_cats + po.cats_other + po.all_three = (total_students : ℚ) * cat_fraction := by sorry
  have other_eq : po.other_only + po.cats_other + po.dogs_other + po.all_three = other_pet_count := by sorry
  have total_eq : po.dogs_only + po.cats_only + po.other_only + po.dogs_cats + po.cats_other + po.dogs_other + po.all_three = total_students - no_pet_count := by sorry

  sorry

end NUMINAMATH_CALUDE_pet_ownership_problem_l3024_302492


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l3024_302483

theorem quadratic_roots_properties :
  let a : ℝ := 1
  let b : ℝ := 4
  let c : ℝ := -42
  let product_of_roots := c / a
  let sum_of_roots := -b / a
  product_of_roots = -42 ∧ sum_of_roots = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l3024_302483


namespace NUMINAMATH_CALUDE_percentage_problem_l3024_302425

theorem percentage_problem (total : ℝ) (part : ℝ) (h1 : total = 300) (h2 : part = 75) :
  (part / total) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3024_302425


namespace NUMINAMATH_CALUDE_min_value_z_l3024_302441

theorem min_value_z (x y : ℝ) (h1 : 2*x + 3*y - 3 ≤ 0) (h2 : 2*x - 3*y + 3 ≥ 0) (h3 : y + 3 ≥ 0) :
  ∀ z : ℝ, z = 2*x + y → z ≥ -3 ∧ ∃ x₀ y₀ : ℝ, 2*x₀ + 3*y₀ - 3 ≤ 0 ∧ 2*x₀ - 3*y₀ + 3 ≥ 0 ∧ y₀ + 3 ≥ 0 ∧ 2*x₀ + y₀ = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l3024_302441


namespace NUMINAMATH_CALUDE_healthcare_worker_identity_l3024_302486

/-- Represents the number of healthcare workers of each type -/
structure HealthcareWorkers where
  male_doctors : ℕ
  female_doctors : ℕ
  female_nurses : ℕ
  male_nurses : ℕ

/-- Checks if the given numbers satisfy all conditions -/
def satisfies_conditions (hw : HealthcareWorkers) : Prop :=
  hw.male_doctors + hw.female_doctors + hw.female_nurses + hw.male_nurses = 17 ∧
  hw.male_doctors + hw.female_doctors ≥ hw.female_nurses + hw.male_nurses ∧
  hw.female_nurses > hw.male_doctors ∧
  hw.male_doctors > hw.female_doctors ∧
  hw.male_nurses ≥ 2

/-- The unique solution that satisfies all conditions -/
def solution : HealthcareWorkers :=
  { male_doctors := 5
    female_doctors := 4
    female_nurses := 6
    male_nurses := 2 }

/-- The statement to be proved -/
theorem healthcare_worker_identity :
  satisfies_conditions solution ∧
  satisfies_conditions { male_doctors := solution.male_doctors,
                         female_doctors := solution.female_doctors - 1,
                         female_nurses := solution.female_nurses,
                         male_nurses := solution.male_nurses } ∧
  ∀ (hw : HealthcareWorkers), satisfies_conditions hw → hw = solution :=
sorry

end NUMINAMATH_CALUDE_healthcare_worker_identity_l3024_302486


namespace NUMINAMATH_CALUDE_bracket_calculation_l3024_302445

-- Define the single bracket operation
def single_bracket (a b c : ℚ) : ℚ := (a + b) / c

-- Define the double bracket operation
def double_bracket (a b c d e f : ℚ) : ℚ := single_bracket (a + b) (d + e) (c + f)

-- State the theorem
theorem bracket_calculation :
  let result := single_bracket
    (double_bracket 10 20 30 40 30 70)
    (double_bracket 8 4 12 18 9 27)
    1
  result = 0.04 + 4/39 := by sorry

end NUMINAMATH_CALUDE_bracket_calculation_l3024_302445


namespace NUMINAMATH_CALUDE_square_area_proof_l3024_302491

theorem square_area_proof (x : ℝ) : 
  (3 * x - 12 = 15 - 2 * x) → 
  ((3 * x - 12)^2 : ℝ) = 441 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l3024_302491


namespace NUMINAMATH_CALUDE_line_relation_in_plane_l3024_302427

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and subset relations
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the positional relationships between lines
inductive LineRelation : Type
  | parallel : LineRelation
  | skew : LineRelation
  | intersecting : LineRelation

-- Define the theorem
theorem line_relation_in_plane (a b : Line) (α : Plane) 
  (h1 : parallel a α) (h2 : subset b α) :
  (∃ r : LineRelation, r = LineRelation.parallel ∨ r = LineRelation.skew) ∧
  ¬(∃ r : LineRelation, r = LineRelation.intersecting) :=
sorry

end NUMINAMATH_CALUDE_line_relation_in_plane_l3024_302427


namespace NUMINAMATH_CALUDE_snake_count_l3024_302434

/-- The number of snakes counted at the zoo --/
def snakes : ℕ := sorry

/-- The number of arctic foxes counted at the zoo --/
def arctic_foxes : ℕ := 80

/-- The number of leopards counted at the zoo --/
def leopards : ℕ := 20

/-- The number of bee-eaters counted at the zoo --/
def bee_eaters : ℕ := 10 * leopards

/-- The number of cheetahs counted at the zoo --/
def cheetahs : ℕ := snakes / 2

/-- The number of alligators counted at the zoo --/
def alligators : ℕ := 2 * (arctic_foxes + leopards)

/-- The total number of animals counted at the zoo --/
def total_animals : ℕ := 670

/-- Theorem stating that the number of snakes counted is 113 --/
theorem snake_count : snakes = 113 := by sorry

end NUMINAMATH_CALUDE_snake_count_l3024_302434


namespace NUMINAMATH_CALUDE_cos_2A_plus_cos_2B_l3024_302480

theorem cos_2A_plus_cos_2B (A B : Real) 
  (h1 : Real.sin A + Real.sin B = 1)
  (h2 : Real.cos A + Real.cos B = 0) :
  12 * Real.cos (2 * A) + 4 * Real.cos (2 * B) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2A_plus_cos_2B_l3024_302480


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l3024_302474

theorem quadratic_roots_difference (p : ℝ) : 
  let a : ℝ := 1
  let b : ℝ := -(2*p + 1)
  let c : ℝ := p*(p + 1)
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  max root1 root2 - min root1 root2 = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l3024_302474


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3024_302408

def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | x < 3}

theorem intersection_of_A_and_B :
  A ∩ B = {x | -1 < x ∧ x < 3} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3024_302408


namespace NUMINAMATH_CALUDE_fraction_of_three_fourths_is_one_fifth_l3024_302476

theorem fraction_of_three_fourths_is_one_fifth (x : ℚ) : x * (3 / 4 : ℚ) = (1 / 5 : ℚ) → x = (4 / 15 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_three_fourths_is_one_fifth_l3024_302476


namespace NUMINAMATH_CALUDE_system_a_solution_l3024_302489

theorem system_a_solution (x y z t : ℝ) : 
  x - 3*y + 2*z - t = 3 ∧
  2*x + 4*y - 3*z + t = 5 ∧
  4*x - 2*y + z + t = 3 ∧
  3*x + y + z - 2*t = 10 →
  x = 2 ∧ y = -1 ∧ z = -3 ∧ t = -4 := by
sorry


end NUMINAMATH_CALUDE_system_a_solution_l3024_302489


namespace NUMINAMATH_CALUDE_brown_shirts_count_l3024_302435

def initial_blue_shirts : ℕ := 26

def remaining_blue_shirts : ℕ := initial_blue_shirts / 2

theorem brown_shirts_count (initial_brown_shirts : ℕ) : 
  remaining_blue_shirts + (initial_brown_shirts - initial_brown_shirts / 3) = 37 →
  initial_brown_shirts = 36 := by
  sorry

end NUMINAMATH_CALUDE_brown_shirts_count_l3024_302435


namespace NUMINAMATH_CALUDE_correct_num_pants_purchased_l3024_302418

/-- Represents the purchase and refund scenario at a clothing retailer -/
structure ClothingPurchase where
  shirtPrice : ℝ
  pantsPrice : ℝ
  totalCost : ℝ
  refundRate : ℝ
  numShirts : ℕ

/-- The number of pairs of pants purchased given the conditions -/
def numPantsPurchased (purchase : ClothingPurchase) : ℕ :=
  1

theorem correct_num_pants_purchased (purchase : ClothingPurchase) 
  (h1 : purchase.shirtPrice ≠ purchase.pantsPrice)
  (h2 : purchase.shirtPrice = 45)
  (h3 : purchase.numShirts = 2)
  (h4 : purchase.totalCost = 120)
  (h5 : purchase.refundRate = 0.25)
  : numPantsPurchased purchase = 1 := by
  sorry

#check correct_num_pants_purchased

end NUMINAMATH_CALUDE_correct_num_pants_purchased_l3024_302418


namespace NUMINAMATH_CALUDE_inverse_proportion_l3024_302469

theorem inverse_proportion (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : 1500 * 0.25 = k) :
  3000 * b = k → b = 0.125 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_l3024_302469


namespace NUMINAMATH_CALUDE_ponderosa_pine_price_l3024_302431

/-- The price of each ponderosa pine tree, given the total number of trees,
    number of Douglas fir trees, price of each Douglas fir, and total amount paid. -/
theorem ponderosa_pine_price
  (total_trees : ℕ)
  (douglas_fir_trees : ℕ)
  (douglas_fir_price : ℕ)
  (total_amount : ℕ)
  (h1 : total_trees = 850)
  (h2 : douglas_fir_trees = 350)
  (h3 : douglas_fir_price = 300)
  (h4 : total_amount = 217500) :
  (total_amount - douglas_fir_trees * douglas_fir_price) / (total_trees - douglas_fir_trees) = 225 := by
sorry


end NUMINAMATH_CALUDE_ponderosa_pine_price_l3024_302431


namespace NUMINAMATH_CALUDE_money_split_l3024_302430

theorem money_split (total : ℝ) (share : ℝ) (n : ℕ) :
  n = 2 →
  share = 32.5 →
  n * share = total →
  total = 65 := by
sorry

end NUMINAMATH_CALUDE_money_split_l3024_302430


namespace NUMINAMATH_CALUDE_parabola_vertex_on_negative_x_axis_l3024_302477

/-- Given a parabola y = x^2 - bx + 8, if its vertex lies on the negative half-axis of the x-axis, then b = -4√2 -/
theorem parabola_vertex_on_negative_x_axis (b : ℝ) :
  (∃ x, x < 0 ∧ x^2 - b*x + 8 = 0 ∧ ∀ y, y ≠ x → (y^2 - b*y + 8 > 0)) →
  b = -4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_negative_x_axis_l3024_302477


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3024_302488

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3024_302488


namespace NUMINAMATH_CALUDE_initial_price_is_four_l3024_302404

/-- Represents the sales data for a day --/
structure DaySales where
  price : ℝ
  quantity : ℝ

/-- Represents the sales data for three days --/
structure ThreeDaySales where
  day1 : DaySales
  day2 : DaySales
  day3 : DaySales

/-- Calculates the revenue for a given day --/
def revenue (day : DaySales) : ℝ :=
  day.price * day.quantity

/-- Checks if the sales data satisfies the problem conditions --/
def satisfiesConditions (sales : ThreeDaySales) : Prop :=
  sales.day2.price = sales.day1.price - 1 ∧
  sales.day2.quantity = sales.day1.quantity + 100 ∧
  sales.day3.price = sales.day2.price + 3 ∧
  sales.day3.quantity = sales.day2.quantity - 200 ∧
  revenue sales.day1 = revenue sales.day2 ∧
  revenue sales.day2 = revenue sales.day3

/-- The main theorem: if the sales data satisfies the conditions, the initial price was 4 yuan --/
theorem initial_price_is_four (sales : ThreeDaySales) :
  satisfiesConditions sales → sales.day1.price = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_price_is_four_l3024_302404


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l3024_302452

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 1

-- Part 1: Prove the solution set for f(x) + |2x-3| > 0 when a = 2
theorem solution_part1 : 
  {x : ℝ | f 2 x + |2*x - 3| > 0} = {x : ℝ | x ≥ 2 ∨ x ≤ 4/3} := by sorry

-- Part 2: Prove the range of a for which f(x) > |x-3| has solutions
theorem solution_part2 : 
  {a : ℝ | ∃ x, f a x > |x - 3|} = {a : ℝ | a < 2 ∨ a > 4} := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l3024_302452


namespace NUMINAMATH_CALUDE_max_volume_at_10cm_l3024_302473

/-- The side length of the original square sheet of metal in centimeters -/
def a : ℝ := 60

/-- The volume of the box as a function of the cut-out square's side length -/
def volume (x : ℝ) : ℝ := (a - 2*x)^2 * x

/-- The derivative of the volume function -/
def volume_derivative (x : ℝ) : ℝ := 3600 - 480*x + 12*x^2

theorem max_volume_at_10cm :
  ∃ (x : ℝ), x > 0 ∧ x < a/2 ∧
  volume_derivative x = 0 ∧
  (∀ y : ℝ, y > 0 → y < a/2 → volume y ≤ volume x) ∧
  x = 10 :=
sorry

end NUMINAMATH_CALUDE_max_volume_at_10cm_l3024_302473


namespace NUMINAMATH_CALUDE_square_area_difference_l3024_302413

def original_side_length : ℝ := 6
def increase_in_length : ℝ := 1

theorem square_area_difference :
  let new_side_length := original_side_length + increase_in_length
  let original_area := original_side_length ^ 2
  let new_area := new_side_length ^ 2
  new_area - original_area = 13 := by sorry

end NUMINAMATH_CALUDE_square_area_difference_l3024_302413


namespace NUMINAMATH_CALUDE_largest_fraction_l3024_302465

theorem largest_fraction : 
  let fractions : List ℚ := [2/3, 3/4, 2/5, 11/15]
  (3/4 : ℚ) = fractions.maximum := by sorry

end NUMINAMATH_CALUDE_largest_fraction_l3024_302465


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3024_302405

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3024_302405


namespace NUMINAMATH_CALUDE_expression_simplification_l3024_302455

theorem expression_simplification (x y : ℝ) :
  3 * y + 4 * y^2 + 2 - (7 - 3 * y - 4 * y^2 + 2 * x) = 8 * y^2 + 6 * y - 2 * x - 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3024_302455


namespace NUMINAMATH_CALUDE_xyz_inequality_l3024_302479

theorem xyz_inequality (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ 
  x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by sorry

end NUMINAMATH_CALUDE_xyz_inequality_l3024_302479


namespace NUMINAMATH_CALUDE_min_S_independent_of_P_l3024_302406

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y = x² + c -/
structure Parabola where
  c : ℝ

/-- Represents the area bounded by a line and a parabola -/
def boundedArea (p₁ p₂ : Point) (C : Parabola) : ℝ := sorry

/-- The sum of areas S as described in the problem -/
def S (P : Point) (C₁ C₂ : Parabola) (m : ℕ) : ℝ := sorry

/-- The minimum value of S -/
def minS (m : ℕ) : ℝ := sorry

theorem min_S_independent_of_P (m : ℕ) :
  ∀ P : Point, P.y = P.x^2 + m^2 → minS m = m^3 / 3 := by sorry

end NUMINAMATH_CALUDE_min_S_independent_of_P_l3024_302406


namespace NUMINAMATH_CALUDE_photo_arrangements_l3024_302471

def teacher : ℕ := 1
def boys : ℕ := 4
def girls : ℕ := 2
def total_people : ℕ := teacher + boys + girls

theorem photo_arrangements :
  (∃ (arrangements_girls_together : ℕ), arrangements_girls_together = 1440) ∧
  (∃ (arrangements_boys_apart : ℕ), arrangements_boys_apart = 144) :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l3024_302471


namespace NUMINAMATH_CALUDE_video_game_time_increase_l3024_302426

/-- Calculates the percentage increase in video game time given the original rate,
    total reading time, and additional time after raise. -/
theorem video_game_time_increase
  (original_rate : ℕ)  -- Original minutes of video game time per hour of reading
  (reading_time : ℕ)   -- Total hours of reading
  (additional_time : ℕ) -- Additional minutes of video game time after raise
  (h1 : original_rate = 30)
  (h2 : reading_time = 12)
  (h3 : additional_time = 72) :
  (additional_time : ℚ) / (original_rate * reading_time) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_video_game_time_increase_l3024_302426


namespace NUMINAMATH_CALUDE_expression_simplification_l3024_302407

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 2) : 
  ((m^2 - 9) / (m^2 - 6*m + 9) - 3 / (m - 3)) / (m^2 / (m - 3)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3024_302407


namespace NUMINAMATH_CALUDE_zero_point_existence_l3024_302438

def f (x : ℝ) : ℝ := -x^3 - 3*x + 5

theorem zero_point_existence :
  ∃ x₀ ∈ Set.Ioo 1 2, f x₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_point_existence_l3024_302438


namespace NUMINAMATH_CALUDE_horner_method_v₂_l3024_302446

def f (x : ℝ) : ℝ := x^5 + x^4 + 2*x^3 + 3*x^2 + 4*x + 1

def horner_v₀ (x : ℝ) : ℝ := 1

def horner_v₁ (x : ℝ) : ℝ := horner_v₀ x * x + 4

def horner_v₂ (x : ℝ) : ℝ := horner_v₁ x * x + 3

theorem horner_method_v₂ : horner_v₂ 2 = 15 := by sorry

end NUMINAMATH_CALUDE_horner_method_v₂_l3024_302446


namespace NUMINAMATH_CALUDE_equal_area_rectangles_length_l3024_302463

/-- Given two rectangles of equal area, where one rectangle has dimensions 12 inches by 10 inches,
    and the other has a width of 5 inches, prove that the length of the second rectangle is 24 inches. -/
theorem equal_area_rectangles_length (area jordan_length jordan_width carol_width : ℝ)
    (h1 : area = jordan_length * jordan_width)
    (h2 : jordan_length = 12)
    (h3 : jordan_width = 10)
    (h4 : carol_width = 5)
    (h5 : area = carol_width * (area / carol_width)) :
    area / carol_width = 24 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_length_l3024_302463


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3024_302429

/-- The quadratic function f(x) = ax² + mx + m - 1 -/
def f (a m x : ℝ) : ℝ := a * x^2 + m * x + m - 1

theorem quadratic_function_properties (a m : ℝ) (h_a : a ≠ 0) :
  /- Part 1: Number of zeros when f(-1) = 0 -/
  (f a m (-1) = 0 → (∃ x, f a m x = 0) ∧ (∃ x y, x ≠ y ∧ f a m x = 0 ∧ f a m y = 0)) ∧
  /- Part 2: Condition for always having two distinct zeros -/
  ((∀ m : ℝ, ∃ x y : ℝ, x ≠ y ∧ f a m x = 0 ∧ f a m y = 0) ↔ 0 < a ∧ a < 1) ∧
  /- Part 3: Existence of root between x₁ and x₂ -/
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a m x₁ ≠ f a m x₂ →
    ∃ x : ℝ, x₁ < x ∧ x < x₂ ∧ f a m x = (f a m x₁ + f a m x₂) / 2) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l3024_302429


namespace NUMINAMATH_CALUDE_combinations_with_repetition_l3024_302466

/-- F_n^r represents the number of r-combinatorial selections from [1, n] with repetition allowed -/
def F (n : ℕ) (r : ℕ) : ℕ := sorry

/-- C_n^r represents the binomial coefficient (n choose r) -/
def C (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The theorem states that F_n^r equals C_(n+r-1)^r -/
theorem combinations_with_repetition (n : ℕ) (r : ℕ) : F n r = C (n + r - 1) r := by
  sorry

end NUMINAMATH_CALUDE_combinations_with_repetition_l3024_302466


namespace NUMINAMATH_CALUDE_team_pays_seventy_percent_l3024_302490

/-- Represents the archer's arrow usage and costs --/
structure ArcherData where
  shots_per_day : ℕ
  days_per_week : ℕ
  recovery_rate : ℚ
  arrow_cost : ℚ
  weekly_spending : ℚ

/-- Calculates the percentage of arrow costs paid by the team --/
def team_payment_percentage (data : ArcherData) : ℚ :=
  let total_shots := data.shots_per_day * data.days_per_week
  let unrecovered_arrows := total_shots * (1 - data.recovery_rate)
  let total_cost := unrecovered_arrows * data.arrow_cost
  let team_contribution := total_cost - data.weekly_spending
  (team_contribution / total_cost) * 100

/-- Theorem stating that the team pays 70% of the archer's arrow costs --/
theorem team_pays_seventy_percent (data : ArcherData)
  (h1 : data.shots_per_day = 200)
  (h2 : data.days_per_week = 4)
  (h3 : data.recovery_rate = 1/5)
  (h4 : data.arrow_cost = 11/2)
  (h5 : data.weekly_spending = 1056) :
  team_payment_percentage data = 70 := by
  sorry

end NUMINAMATH_CALUDE_team_pays_seventy_percent_l3024_302490


namespace NUMINAMATH_CALUDE_watson_class_first_graders_l3024_302428

/-- The number of first graders in Ms. Watson's class -/
def first_graders (total : ℕ) (kindergartners : ℕ) (second_graders : ℕ) : ℕ :=
  total - (kindergartners + second_graders)

/-- Theorem stating the number of first graders in Ms. Watson's class -/
theorem watson_class_first_graders :
  first_graders 42 14 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_watson_class_first_graders_l3024_302428


namespace NUMINAMATH_CALUDE_markese_earnings_l3024_302470

/-- Given Evan's earnings E, Markese's earnings (E - 5), and their total earnings of 37,
    prove that Markese earned 16 dollars. -/
theorem markese_earnings (E : ℕ) : E + (E - 5) = 37 → E - 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_markese_earnings_l3024_302470


namespace NUMINAMATH_CALUDE_division_result_l3024_302456

theorem division_result : (64 : ℝ) / 0.08 = 800 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l3024_302456


namespace NUMINAMATH_CALUDE_height_difference_is_4b_minus_8_l3024_302485

/-- A circle inside a parabola y = 4x^2, tangent at two points -/
structure TangentCircle where
  /-- y-coordinate of the circle's center -/
  b : ℝ
  /-- x-coordinate of one tangent point (the other is -a) -/
  a : ℝ
  /-- The point (a, 4a^2) lies on the parabola -/
  tangent_on_parabola : 4 * a^2 = 4 * a^2
  /-- The point (a, 4a^2) lies on the circle -/
  tangent_on_circle : a^2 + (4 * a^2 - b)^2 = (b - 4 * a^2)^2 + a^2
  /-- Relation between a and b derived from tangency condition -/
  tangency_relation : 4 * b - a^2 = 8

/-- The difference in height between the circle's center and tangent points -/
def height_difference (c : TangentCircle) : ℝ := c.b - 4 * c.a^2

/-- Theorem: The height difference is always 4b - 8 -/
theorem height_difference_is_4b_minus_8 (c : TangentCircle) :
  height_difference c = 4 * c.b - 8 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_is_4b_minus_8_l3024_302485


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l3024_302440

/-- The number of trailing zeros in base 12 for the product 33 * 59 -/
def trailing_zeros_base_12 : ℕ := 2

/-- The product we're working with -/
def product : ℕ := 33 * 59

/-- Conversion to base 12 -/
def to_base_12 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 12) ((m % 12) :: acc)
    aux n []

/-- Count trailing zeros in a list of digits -/
def count_trailing_zeros (digits : List ℕ) : ℕ :=
  digits.reverse.takeWhile (· = 0) |>.length

theorem product_trailing_zeros :
  count_trailing_zeros (to_base_12 product) = trailing_zeros_base_12 := by
  sorry

#eval to_base_12 product
#eval count_trailing_zeros (to_base_12 product)

end NUMINAMATH_CALUDE_product_trailing_zeros_l3024_302440


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3024_302450

def A : Set ℝ := {1, 2, 3, 4, 5}
def B : Set ℝ := {x : ℝ | x * (4 - x) < 0}

theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3024_302450


namespace NUMINAMATH_CALUDE_sweater_a_markup_sweater_b_markup_l3024_302420

/-- Calculates the final price after applying a markup and two discounts -/
def final_price (wholesale : ℝ) (markup discount1 discount2 : ℝ) : ℝ :=
  wholesale * (1 + markup) * (1 - discount1) * (1 - discount2)

/-- Theorem for Sweater A -/
theorem sweater_a_markup (wholesale : ℝ) :
  final_price wholesale 3 0.2 0.5 = wholesale * 1.6 := by sorry

/-- Theorem for Sweater B -/
theorem sweater_b_markup (wholesale : ℝ) :
  ∃ ε > 0, ε < 0.0001 ∧ 
  |final_price wholesale 3.60606 0.25 0.45 - wholesale * 1.9| < ε := by sorry

end NUMINAMATH_CALUDE_sweater_a_markup_sweater_b_markup_l3024_302420


namespace NUMINAMATH_CALUDE_no_solutions_squared_l3024_302478

theorem no_solutions_squared (n : ℕ) (h : n > 2) :
  (∀ x y z : ℕ+, x^n + y^n ≠ z^n) →
  (∀ x y z : ℕ+, x^(2*n) + y^(2*n) ≠ z^2) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_squared_l3024_302478


namespace NUMINAMATH_CALUDE_oven_capacity_is_two_l3024_302459

/-- Represents the pizza-making process with given constraints -/
structure PizzaMaking where
  dough_time : ℕ  -- Time to make one batch of dough (in minutes)
  cook_time : ℕ   -- Time to cook pizzas in the oven (in minutes)
  pizzas_per_batch : ℕ  -- Number of pizzas one batch of dough can make
  total_time : ℕ  -- Total time to make all pizzas (in minutes)
  total_pizzas : ℕ  -- Total number of pizzas to be made

/-- Calculates the number of pizzas that can fit in the oven at once -/
def oven_capacity (pm : PizzaMaking) : ℕ :=
  let dough_making_time := (pm.total_pizzas / pm.pizzas_per_batch) * pm.dough_time
  let baking_time := pm.total_time - dough_making_time
  let baking_intervals := baking_time / pm.cook_time
  pm.total_pizzas / baking_intervals

/-- Theorem stating that given the conditions, the oven capacity is 2 pizzas -/
theorem oven_capacity_is_two (pm : PizzaMaking)
  (h1 : pm.dough_time = 30)
  (h2 : pm.cook_time = 30)
  (h3 : pm.pizzas_per_batch = 3)
  (h4 : pm.total_time = 300)  -- 5 hours = 300 minutes
  (h5 : pm.total_pizzas = 12) :
  oven_capacity pm = 2 := by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_oven_capacity_is_two_l3024_302459


namespace NUMINAMATH_CALUDE_first_month_sale_is_3435_l3024_302402

/-- Calculates the sale in the first month given the sales for the next five months and the average sale --/
def calculate_first_month_sale (sales_2_to_5 : List ℕ) (sale_6 : ℕ) (average_sale : ℕ) : ℕ :=
  6 * average_sale - (sales_2_to_5.sum + sale_6)

/-- Theorem stating that the sale in the first month is 3435 given the specified conditions --/
theorem first_month_sale_is_3435 :
  let sales_2_to_5 := [3920, 3855, 4230, 3560]
  let sale_6 := 2000
  let average_sale := 3500
  calculate_first_month_sale sales_2_to_5 sale_6 average_sale = 3435 := by
  sorry

#eval calculate_first_month_sale [3920, 3855, 4230, 3560] 2000 3500

end NUMINAMATH_CALUDE_first_month_sale_is_3435_l3024_302402


namespace NUMINAMATH_CALUDE_divisible_by_eight_expression_l3024_302424

theorem divisible_by_eight_expression :
  ∃ (A B C : ℕ), (A % 8 ≠ 0) ∧ (B % 8 ≠ 0) ∧ (C % 8 ≠ 0) ∧
    (∀ n : ℕ, (A * 5^n + B * 3^(n-1) + C) % 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_eight_expression_l3024_302424


namespace NUMINAMATH_CALUDE_simplify_fraction_l3024_302496

theorem simplify_fraction : (75 : ℚ) / 100 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3024_302496


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_l3024_302454

theorem consecutive_page_numbers (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 283 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_l3024_302454


namespace NUMINAMATH_CALUDE_shells_per_friend_eq_l3024_302468

/-- The number of shells each friend gets when Jillian, Savannah, and Clayton
    distribute their shells evenly among F friends. -/
def shellsPerFriend (F : ℕ+) : ℚ :=
  let J : ℕ := 29  -- Jillian's shells
  let S : ℕ := 17  -- Savannah's shells
  let C : ℕ := 8   -- Clayton's shells
  (J + S + C) / F

/-- Theorem stating that the number of shells each friend gets is 54 / F. -/
theorem shells_per_friend_eq (F : ℕ+) : shellsPerFriend F = 54 / F := by
  sorry

end NUMINAMATH_CALUDE_shells_per_friend_eq_l3024_302468


namespace NUMINAMATH_CALUDE_basic_computer_price_l3024_302433

/-- Proves that the price of a basic computer is $1500 given certain conditions. -/
theorem basic_computer_price (basic_price printer_price : ℕ) : 
  (basic_price + printer_price = 2500) →
  (printer_price = (basic_price + 500 + printer_price) / 3) →
  basic_price = 1500 := by
  sorry

#check basic_computer_price

end NUMINAMATH_CALUDE_basic_computer_price_l3024_302433


namespace NUMINAMATH_CALUDE_videocassette_recorder_fraction_l3024_302481

theorem videocassette_recorder_fraction 
  (cable_fraction : Real) 
  (cable_and_vcr_fraction : Real) 
  (neither_fraction : Real) :
  cable_fraction = 1/5 →
  cable_and_vcr_fraction = 1/3 * cable_fraction →
  neither_fraction = 0.7666666666666667 →
  ∃ (vcr_fraction : Real),
    vcr_fraction = 1/10 ∧
    vcr_fraction + cable_fraction - cable_and_vcr_fraction + neither_fraction = 1 :=
by sorry

end NUMINAMATH_CALUDE_videocassette_recorder_fraction_l3024_302481


namespace NUMINAMATH_CALUDE_perfect_squares_divisibility_l3024_302409

theorem perfect_squares_divisibility (a b : ℕ+) :
  (∃ (S : Set (ℕ+ × ℕ+)), Set.Infinite S ∧
    ∀ (p : ℕ+ × ℕ+), p ∈ S →
      ∃ (k l : ℕ+), (p.1.val ^ 2 + a.val * p.2.val + b.val = k.val ^ 2) ∧
                    (p.2.val ^ 2 + a.val * p.1.val + b.val = l.val ^ 2)) →
  a.val ∣ (2 * b.val) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_divisibility_l3024_302409


namespace NUMINAMATH_CALUDE_successive_discounts_equivalent_to_single_discount_l3024_302447

-- Define the original price and discount rates
def original_price : ℝ := 50
def discount1 : ℝ := 0.30
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.10

-- Define the equivalent single discount
def equivalent_discount : ℝ := 0.4645

-- Theorem statement
theorem successive_discounts_equivalent_to_single_discount :
  (1 - discount1) * (1 - discount2) * (1 - discount3) = 1 - equivalent_discount :=
by sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalent_to_single_discount_l3024_302447


namespace NUMINAMATH_CALUDE_prime_sequence_finite_l3024_302419

/-- A sequence of primes satisfying the given conditions -/
def PrimeSequence (p : ℕ → ℕ) : Prop :=
  (∀ n, Nat.Prime (p n)) ∧ 
  (∀ i ≥ 2, p i = 2 * p (i-1) - 1 ∨ p i = 2 * p (i-1) + 1)

/-- The theorem stating that any such sequence is finite -/
theorem prime_sequence_finite (p : ℕ → ℕ) (h : PrimeSequence p) : 
  ∃ N, ∀ n > N, ¬ Nat.Prime (p n) :=
sorry

end NUMINAMATH_CALUDE_prime_sequence_finite_l3024_302419


namespace NUMINAMATH_CALUDE_perimeter_of_remaining_figure_l3024_302493

/-- The perimeter of a rectangle after cutting out squares --/
def perimeter_after_cuts (length width num_cuts cut_size : ℕ) : ℕ :=
  2 * (length + width) + num_cuts * (4 * cut_size - 2 * cut_size)

/-- Theorem stating the perimeter of the remaining figure after cuts --/
theorem perimeter_of_remaining_figure :
  perimeter_after_cuts 40 30 10 5 = 240 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_remaining_figure_l3024_302493


namespace NUMINAMATH_CALUDE_triangle_larger_segment_l3024_302494

theorem triangle_larger_segment (a b c h x : ℝ) : 
  a = 35 → b = 65 → c = 85 → 
  a^2 = x^2 + h^2 → 
  b^2 = (c - x)^2 + h^2 → 
  c - x = 60 :=
by sorry

end NUMINAMATH_CALUDE_triangle_larger_segment_l3024_302494


namespace NUMINAMATH_CALUDE_fourth_smallest_is_six_probability_l3024_302482

def S : Finset ℕ := Finset.range 15

def probability_fourth_smallest_is_six (n : ℕ) : ℚ :=
  let total_combinations := Nat.choose 15 8
  let favorable_outcomes := Nat.choose 5 3 * Nat.choose 9 5
  (favorable_outcomes : ℚ) / total_combinations

theorem fourth_smallest_is_six_probability :
  probability_fourth_smallest_is_six 6 = 4 / 21 := by
  sorry

#eval probability_fourth_smallest_is_six 6

end NUMINAMATH_CALUDE_fourth_smallest_is_six_probability_l3024_302482


namespace NUMINAMATH_CALUDE_hall_ratio_l3024_302498

theorem hall_ratio (width length : ℝ) : 
  width > 0 →
  length > 0 →
  width * length = 288 →
  length - width = 12 →
  width / length = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_hall_ratio_l3024_302498


namespace NUMINAMATH_CALUDE_circle_passes_through_origin_l3024_302403

/-- Definition of the circle C with parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*(m-1)*x + 2*(m-1)*y + 2*m^2 - 6*m + 4 = 0

/-- Theorem stating that the circle passes through the origin when m = 2 -/
theorem circle_passes_through_origin :
  ∃ m : ℝ, circle_equation 0 0 m ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_origin_l3024_302403


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3024_302457

theorem arithmetic_equality : 4 * (8 - 3) - 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3024_302457


namespace NUMINAMATH_CALUDE_fraction_simplification_l3024_302422

theorem fraction_simplification (c d : ℝ) : 
  (5 + 4 * c - 3 * d) / 9 + 5 = (50 + 4 * c - 3 * d) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3024_302422
