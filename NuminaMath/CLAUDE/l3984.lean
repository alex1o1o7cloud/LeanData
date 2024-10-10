import Mathlib

namespace probability_second_green_given_first_green_l3984_398481

def total_balls : ℕ := 14
def green_balls : ℕ := 8
def red_balls : ℕ := 6

theorem probability_second_green_given_first_green :
  (green_balls : ℚ) / total_balls = 
  (green_balls : ℚ) / (green_balls + red_balls) :=
by sorry

end probability_second_green_given_first_green_l3984_398481


namespace cube_cuboid_volume_ratio_l3984_398485

theorem cube_cuboid_volume_ratio :
  let cube_side : ℝ := 1
  let cuboid_width : ℝ := 50 / 100
  let cuboid_length : ℝ := 50 / 100
  let cuboid_height : ℝ := 20 / 100
  let cube_volume := cube_side ^ 3
  let cuboid_volume := cuboid_width * cuboid_length * cuboid_height
  cube_volume / cuboid_volume = 20 := by
    sorry

end cube_cuboid_volume_ratio_l3984_398485


namespace divisibility_of_cube_difference_l3984_398454

theorem divisibility_of_cube_difference (a b c : ℕ) : 
  Nat.Prime a → Nat.Prime b → Nat.Prime c → 
  c ∣ (a + b) → c ∣ (a * b) → 
  c ∣ (a^3 - b^3) := by
  sorry

end divisibility_of_cube_difference_l3984_398454


namespace sector_central_angle_l3984_398416

/-- Given a sector with perimeter 4 and area 1, its central angle is 2 radians -/
theorem sector_central_angle (r l : ℝ) (h1 : l + 2*r = 4) (h2 : (1/2)*l*r = 1) :
  l / r = 2 := by sorry

end sector_central_angle_l3984_398416


namespace tetrahedron_sum_l3984_398458

/-- A regular tetrahedron is a three-dimensional shape with four congruent equilateral triangular faces. -/
structure RegularTetrahedron where
  -- We don't need to define any fields here, as we're only interested in its properties

/-- The number of edges in a regular tetrahedron -/
def num_edges (t : RegularTetrahedron) : ℕ := 6

/-- The number of vertices in a regular tetrahedron -/
def num_vertices (t : RegularTetrahedron) : ℕ := 4

/-- The number of faces in a regular tetrahedron -/
def num_faces (t : RegularTetrahedron) : ℕ := 4

/-- The theorem stating that the sum of edges, vertices, and faces of a regular tetrahedron is 14 -/
theorem tetrahedron_sum (t : RegularTetrahedron) : 
  num_edges t + num_vertices t + num_faces t = 14 := by
  sorry

end tetrahedron_sum_l3984_398458


namespace solve_business_partnership_l3984_398418

/-- Represents the problem of determining when Hari joined Praveen's business --/
def business_partnership_problem (praveen_investment : ℕ) (hari_investment : ℕ) (profit_ratio_praveen : ℕ) (profit_ratio_hari : ℕ) (total_months : ℕ) : Prop :=
  ∃ (x : ℕ), 
    x ≤ total_months ∧
    (praveen_investment * total_months) * profit_ratio_hari = 
    (hari_investment * (total_months - x)) * profit_ratio_praveen

/-- Theorem stating the solution to the business partnership problem --/
theorem solve_business_partnership : 
  business_partnership_problem 3360 8640 2 3 12 → 
  ∃ (x : ℕ), x = 5 := by
  sorry

end solve_business_partnership_l3984_398418


namespace profit_180_greater_than_170_l3984_398484

/-- Sales data for 20 days -/
def sales_data : List (ℕ × ℕ) := [(150, 3), (160, 4), (170, 6), (180, 5), (190, 1), (200, 1)]

/-- Total number of days -/
def total_days : ℕ := 20

/-- Purchase price in yuan per kg -/
def purchase_price : ℚ := 6

/-- Selling price in yuan per kg -/
def selling_price : ℚ := 10

/-- Return price in yuan per kg -/
def return_price : ℚ := 4

/-- Calculate expected profit for a given purchase amount -/
def expected_profit (purchase_amount : ℕ) : ℚ :=
  sorry

/-- Theorem: Expected profit from 180 kg purchase is greater than 170 kg purchase -/
theorem profit_180_greater_than_170 :
  expected_profit 180 > expected_profit 170 :=
sorry

end profit_180_greater_than_170_l3984_398484


namespace harkamal_purchase_amount_l3984_398474

/-- The total amount paid by Harkamal for grapes and mangoes -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 1135 for his purchase -/
theorem harkamal_purchase_amount :
  total_amount_paid 8 80 9 55 = 1135 := by
  sorry


end harkamal_purchase_amount_l3984_398474


namespace third_term_of_geometric_series_l3984_398480

/-- Given an infinite geometric series with common ratio 1/4 and sum 16, 
    the third term of the sequence is 3/4. -/
theorem third_term_of_geometric_series (a : ℝ) : 
  (∃ (S : ℝ), S = 16 ∧ S = a / (1 - (1/4))) →
  a * (1/4)^2 = 3/4 := by
sorry

end third_term_of_geometric_series_l3984_398480


namespace M_subset_N_l3984_398453

-- Define the sets M and N
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | x ≤ 1}

-- Theorem statement
theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l3984_398453


namespace star_six_three_l3984_398498

-- Define the ⭐ operation
def star (x y : ℝ) : ℝ := 4 * x - 2 * y

-- State the theorem
theorem star_six_three : star 6 3 = 18 := by
  sorry

end star_six_three_l3984_398498


namespace algebraic_expression_value_l3984_398432

theorem algebraic_expression_value (a b : ℝ) (h : a - 3 * b = 0) :
  (a - (2 * a * b - b^2) / a) / ((a^2 - b^2) / a) = 1 / 2 :=
by sorry

end algebraic_expression_value_l3984_398432


namespace g_of_3_equals_10_l3984_398431

def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem g_of_3_equals_10 : g 3 = 10 := by
  sorry

end g_of_3_equals_10_l3984_398431


namespace time_spent_on_other_subjects_l3984_398429

def total_time : ℝ := 150

def math_percent : ℝ := 0.20
def science_percent : ℝ := 0.25
def history_percent : ℝ := 0.10
def english_percent : ℝ := 0.15

def min_time_remaining_subject : ℝ := 30

theorem time_spent_on_other_subjects :
  let math_time := total_time * math_percent
  let science_time := total_time * science_percent
  let history_time := total_time * history_percent
  let english_time := total_time * english_percent
  let known_subjects_time := math_time + science_time + history_time + english_time
  let remaining_time := total_time - known_subjects_time
  remaining_time - min_time_remaining_subject = 15 := by
  sorry

end time_spent_on_other_subjects_l3984_398429


namespace equation_system_solution_l3984_398440

theorem equation_system_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : 1 / (x * y) = x / z + 1)
  (eq2 : 1 / (y * z) = y / x + 1)
  (eq3 : 1 / (z * x) = z / y + 1) :
  x = 1 / Real.sqrt 2 ∧ y = 1 / Real.sqrt 2 ∧ z = 1 / Real.sqrt 2 :=
by sorry

end equation_system_solution_l3984_398440


namespace max_profit_is_45_6_l3984_398424

-- Define the profit functions
def profit_A (t : ℕ) : ℚ := 5.06 * t - 0.15 * t^2
def profit_B (t : ℕ) : ℚ := 2 * t

-- Define the total profit function
def total_profit (x : ℕ) : ℚ := profit_A x + profit_B (15 - x)

-- Theorem statement
theorem max_profit_is_45_6 :
  ∃ (x : ℕ), x ≤ 15 ∧ total_profit x = 45.6 ∧
  ∀ (y : ℕ), y ≤ 15 → total_profit y ≤ 45.6 := by
  sorry


end max_profit_is_45_6_l3984_398424


namespace wicket_keeper_age_difference_l3984_398407

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  total_members : Nat
  average_age : Nat
  wicket_keeper_age : Nat
  haveProperty : (total_members - 2) * (average_age - 1) = (total_members * average_age - wicket_keeper_age - average_age)

/-- Theorem stating the age difference between the wicket keeper and the team average -/
theorem wicket_keeper_age_difference (team : CricketTeam)
  (h1 : team.total_members = 11)
  (h2 : team.average_age = 23) :
  team.wicket_keeper_age - team.average_age = 9 := by
  sorry

end wicket_keeper_age_difference_l3984_398407


namespace square_product_of_b_values_l3984_398457

theorem square_product_of_b_values : ∃ (b₁ b₂ : ℝ),
  (∀ (x y : ℝ), (y = 3 ∨ y = 8 ∨ x = 2 ∨ x = b₁ ∨ x = b₂) →
    ((x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 8) ∨ (x = b₁ ∧ y = 3) ∨ (x = b₁ ∧ y = 8) ∨
     (x = b₂ ∧ y = 3) ∨ (x = b₂ ∧ y = 8) ∨ (x = 2 ∧ 3 ≤ y ∧ y ≤ 8) ∨
     (x = b₁ ∧ 3 ≤ y ∧ y ≤ 8) ∨ (x = b₂ ∧ 3 ≤ y ∧ y ≤ 8) ∨
     (3 ≤ x ∧ x ≤ 8 ∧ y = 3) ∨ (3 ≤ x ∧ x ≤ 8 ∧ y = 8))) ∧
  b₁ * b₂ = -21 :=
by sorry

end square_product_of_b_values_l3984_398457


namespace inverse_of_A_cubed_l3984_398463

-- Define the matrix A⁻¹
def A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 0, 3]

-- State the theorem
theorem inverse_of_A_cubed :
  let A : Matrix (Fin 2) (Fin 2) ℝ := A_inv⁻¹
  (A^3)⁻¹ = !![8, -19; 0, 27] := by
  sorry

end inverse_of_A_cubed_l3984_398463


namespace all_hanging_pieces_equal_l3984_398451

/-- Represents a square table covered by a square tablecloth -/
structure TableWithCloth where
  table_side : ℝ
  cloth_side : ℝ
  hanging_piece : ℝ → ℝ → ℝ
  no_corner_covered : cloth_side > table_side
  no_overlap : cloth_side ≤ table_side + 2 * (hanging_piece 0 0)
  adjacent_equal : ∀ (i j : Fin 4), (i.val + 1) % 4 = j.val → 
    hanging_piece i.val 0 = hanging_piece j.val 0

/-- All four hanging pieces of the tablecloth are equal -/
theorem all_hanging_pieces_equal (t : TableWithCloth) : 
  ∀ (i j : Fin 4), t.hanging_piece i.val 0 = t.hanging_piece j.val 0 := by
  sorry

end all_hanging_pieces_equal_l3984_398451


namespace smallest_w_l3984_398430

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) (hw : w > 0) 
  (h1 : is_factor (2^5) (936 * w))
  (h2 : is_factor (3^3) (936 * w))
  (h3 : is_factor (13^2) (936 * w)) :
  w ≥ 156 ∧ ∃ w', w' = 156 ∧ w' > 0 ∧ 
    is_factor (2^5) (936 * w') ∧ 
    is_factor (3^3) (936 * w') ∧ 
    is_factor (13^2) (936 * w') :=
sorry

end smallest_w_l3984_398430


namespace no_valid_n_l3984_398425

theorem no_valid_n : ¬∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (y : ℤ), n^2 - 18*n + 80 = y^2) ∧ 
  (∃ (k : ℤ), 15 = n * k) := by
sorry

end no_valid_n_l3984_398425


namespace prism_volume_l3984_398486

/-- A right rectangular prism with face areas 15, 20, and 24 square inches has a volume of 60 cubic inches. -/
theorem prism_volume (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 20) (h3 : l * h = 24) :
  l * w * h = 60 := by
  sorry

end prism_volume_l3984_398486


namespace acme_vowel_soup_sequences_l3984_398405

/-- The number of vowels in the soup -/
def num_vowels : ℕ := 5

/-- The length of each sequence -/
def sequence_length : ℕ := 5

/-- The minimum number of times any vowel appears -/
def min_vowel_count : ℕ := 3

/-- The maximum number of times any vowel appears -/
def max_vowel_count : ℕ := 7

/-- The number of five-letter sequences that can be formed -/
def num_sequences : ℕ := num_vowels ^ sequence_length

theorem acme_vowel_soup_sequences :
  num_sequences = 3125 :=
sorry

end acme_vowel_soup_sequences_l3984_398405


namespace rotated_angle_measure_l3984_398433

/-- Given an initial angle of 60 degrees and a clockwise rotation of 600 degrees,
    the resulting new acute angle is 60 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 60 →
  rotation = 600 →
  let effective_rotation := rotation % 360
  let new_angle := (effective_rotation - initial_angle) % 180
  new_angle = 60 :=
by sorry

end rotated_angle_measure_l3984_398433


namespace north_movement_representation_l3984_398476

/-- Represents the direction of movement -/
inductive Direction
  | North
  | South

/-- Represents a movement with distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Converts a movement to its numerical representation -/
def movementToMeters (m : Movement) : ℝ :=
  match m.direction with
  | Direction.North => m.distance
  | Direction.South => -m.distance

theorem north_movement_representation (d : ℝ) (h : d > 0) :
  let southMovement : Movement := ⟨d, Direction.South⟩
  let northMovement : Movement := ⟨d, Direction.North⟩
  movementToMeters southMovement = -d →
  movementToMeters northMovement = d :=
by sorry

end north_movement_representation_l3984_398476


namespace complex_power_36_135_deg_l3984_398409

theorem complex_power_36_135_deg :
  (Complex.exp (Complex.I * Real.pi * (3 / 4)))^36 = Complex.ofReal (1 / 2) - Complex.I * Complex.ofReal (Real.sqrt 3 / 2) := by
  sorry

end complex_power_36_135_deg_l3984_398409


namespace derivative_of_exp_x_squared_minus_one_l3984_398452

theorem derivative_of_exp_x_squared_minus_one (x : ℝ) :
  deriv (λ x => Real.exp (x^2 - 1)) x = 2 * x * Real.exp (x^2 - 1) :=
by sorry

end derivative_of_exp_x_squared_minus_one_l3984_398452


namespace video_game_map_width_l3984_398438

/-- Represents the dimensions of a rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℝ :=
  prism.length * prism.width * prism.height

theorem video_game_map_width :
  ∀ (prism : RectangularPrism),
    volume prism = 50 →
    prism.length = 5 →
    prism.height = 2 →
    prism.width = 5 := by
  sorry

end video_game_map_width_l3984_398438


namespace log_eight_x_equals_three_halves_l3984_398420

theorem log_eight_x_equals_three_halves (x : ℝ) :
  Real.log x / Real.log 8 = 3/2 → x = 16 * Real.sqrt 2 := by
  sorry

end log_eight_x_equals_three_halves_l3984_398420


namespace cyrus_additional_bites_l3984_398408

/-- The number of mosquito bites Cyrus initially counted on his arms and legs -/
def initial_bites : ℕ := 14

/-- The number of people in Cyrus's family, excluding Cyrus -/
def family_members : ℕ := 6

/-- The number of additional mosquito bites on Cyrus's body -/
def additional_bites : ℕ := 14

/-- The total number of mosquito bites Cyrus got -/
def total_cyrus_bites : ℕ := initial_bites + additional_bites

/-- The total number of mosquito bites Cyrus's family got -/
def family_bites : ℕ := total_cyrus_bites / 2

/-- The number of mosquito bites each family member got -/
def bites_per_family_member : ℚ := family_bites / family_members

theorem cyrus_additional_bites :
  bites_per_family_member = additional_bites / family_members :=
by sorry

end cyrus_additional_bites_l3984_398408


namespace width_to_perimeter_ratio_l3984_398499

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular room -/
def perimeter (room : RoomDimensions) : ℝ :=
  2 * (room.length + room.width)

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplifyRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem width_to_perimeter_ratio (room : RoomDimensions)
    (h1 : room.length = 25)
    (h2 : room.width = 15) :
    simplifyRatio (Nat.floor room.width) (Nat.floor (perimeter room)) = (3, 16) := by
  sorry

end width_to_perimeter_ratio_l3984_398499


namespace classical_mechanics_not_incorrect_l3984_398442

/-- Represents a scientific theory -/
structure ScientificTheory where
  name : String
  hasLimitations : Bool
  isIncorrect : Bool

/-- Classical mechanics as a scientific theory -/
def classicalMechanics : ScientificTheory := {
  name := "Classical Mechanics"
  hasLimitations := true
  isIncorrect := false
}

/-- Truth has relativity -/
axiom truth_relativity : Prop

/-- Scientific exploration is endless -/
axiom endless_exploration : Prop

/-- Theorem stating that classical mechanics is not an incorrect scientific theory -/
theorem classical_mechanics_not_incorrect :
  classicalMechanics.hasLimitations ∧ truth_relativity ∧ endless_exploration →
  ¬classicalMechanics.isIncorrect := by
  sorry


end classical_mechanics_not_incorrect_l3984_398442


namespace max_arithmetic_mean_for_special_pair_l3984_398403

theorem max_arithmetic_mean_for_special_pair : ∃ (a b : ℕ), 
  10 ≤ a ∧ a ≤ 99 ∧ 
  10 ≤ b ∧ b ≤ 99 ∧ 
  a > b ∧
  (a + b) / 2 = (25 / 24) * Real.sqrt (a * b) ∧
  ∀ (c d : ℕ), 
    10 ≤ c ∧ c ≤ 99 ∧ 
    10 ≤ d ∧ d ≤ 99 ∧ 
    c > d ∧
    (c + d) / 2 = (25 / 24) * Real.sqrt (c * d) →
    (a + b) / 2 ≥ (c + d) / 2 ∧
  (a + b) / 2 = 75 := by
sorry

end max_arithmetic_mean_for_special_pair_l3984_398403


namespace parallelogram_side_lengths_l3984_398455

/-- A parallelogram with the given properties has sides of length 4 and 12 -/
theorem parallelogram_side_lengths 
  (perimeter : ℝ) 
  (triangle_perimeter_diff : ℝ) 
  (h_perimeter : perimeter = 32) 
  (h_diff : triangle_perimeter_diff = 8) :
  ∃ (a b : ℝ), a + b = perimeter / 2 ∧ b - a = triangle_perimeter_diff ∧ a = 4 ∧ b = 12 :=
by sorry

end parallelogram_side_lengths_l3984_398455


namespace sum_of_roots_quadratic_l3984_398482

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 4 = 0) → 
  (x₂^2 + 2*x₂ - 4 = 0) → 
  (x₁ + x₂ = -2) := by
sorry

end sum_of_roots_quadratic_l3984_398482


namespace regression_consistency_l3984_398497

/-- A structure representing a linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- A structure representing sample statistics -/
structure SampleStatistics where
  x_mean : ℝ
  y_mean : ℝ
  correlation : ℝ

/-- Checks if the given linear regression model is consistent with the sample statistics -/
def is_consistent_regression (stats : SampleStatistics) (model : LinearRegression) : Prop :=
  stats.correlation > 0 ∧ 
  stats.y_mean = model.slope * stats.x_mean + model.intercept

/-- The theorem stating that the given linear regression model is consistent with the sample statistics -/
theorem regression_consistency : 
  let stats : SampleStatistics := { x_mean := 3, y_mean := 3.5, correlation := 1 }
  let model : LinearRegression := { slope := 0.4, intercept := 2.3 }
  is_consistent_regression stats model := by
  sorry


end regression_consistency_l3984_398497


namespace bird_cost_problem_l3984_398449

/-- Calculates the cost per bird given the total money and number of birds -/
def cost_per_bird (total_money : ℚ) (num_birds : ℕ) : ℚ :=
  total_money / num_birds

/-- The problem statement -/
theorem bird_cost_problem :
  let total_money : ℚ := 4 * 50
  let total_wings : ℕ := 20
  let wings_per_bird : ℕ := 2
  let num_birds : ℕ := total_wings / wings_per_bird
  cost_per_bird total_money num_birds = 20 := by
  sorry

end bird_cost_problem_l3984_398449


namespace polynomial_coefficient_C_l3984_398435

theorem polynomial_coefficient_C (A B C D : ℤ) : 
  (∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ z : ℂ, z^6 - 15*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36 = 
      (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆)) ∧
    (r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 15)) → 
  C = -92 := by
sorry

end polynomial_coefficient_C_l3984_398435


namespace factorization_proof_l3984_398411

theorem factorization_proof (a : ℝ) : 74 * a^2 + 222 * a + 148 = 74 * (a + 2) * (a + 1) := by
  sorry

end factorization_proof_l3984_398411


namespace min_coach_handshakes_l3984_398471

/-- Represents the total number of handshakes -/
def total_handshakes : ℕ := 435

/-- Calculates the number of handshakes between players given the number of players -/
def player_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the number of handshakes the coach had -/
def coach_handshakes (n : ℕ) : ℕ := total_handshakes - player_handshakes n

theorem min_coach_handshakes :
  ∃ (n : ℕ), n > 1 ∧ player_handshakes n ≤ total_handshakes ∧ coach_handshakes n = 0 :=
sorry

end min_coach_handshakes_l3984_398471


namespace simplify_expression_l3984_398456

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (15 * x^2) * (6 * x) * (1 / (3 * x)^2) = 10 * x := by
  sorry

end simplify_expression_l3984_398456


namespace smallest_sum_of_reciprocals_l3984_398415

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 ∧ a + b = 49 ∧ ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12 → c + d ≥ 49 :=
by sorry

end smallest_sum_of_reciprocals_l3984_398415


namespace jane_earnings_l3984_398401

/-- Represents the number of flower bulbs planted for each type --/
structure FlowerBulbs where
  tulips : ℕ
  iris : ℕ
  daffodils : ℕ
  crocus : ℕ

/-- Calculates the total earnings from planting flower bulbs --/
def calculate_earnings (bulbs : FlowerBulbs) (price_per_bulb : ℚ) : ℚ :=
  price_per_bulb * (bulbs.tulips + bulbs.iris + bulbs.daffodils + bulbs.crocus)

/-- The main theorem stating Jane's earnings --/
theorem jane_earnings : ∃ (bulbs : FlowerBulbs),
  bulbs.tulips = 20 ∧
  bulbs.iris = bulbs.tulips / 2 ∧
  bulbs.daffodils = 30 ∧
  bulbs.crocus = 3 * bulbs.daffodils ∧
  calculate_earnings bulbs (1/2) = 75 := by
  sorry


end jane_earnings_l3984_398401


namespace tracy_candies_l3984_398478

theorem tracy_candies (x : ℕ) : 
  (∃ (y : ℕ), x = 4 * y) →  -- x is divisible by 4
  (∃ (z : ℕ), (3 * x) / 4 = 3 * z) →  -- (3/4)x is divisible by 3
  (7 ≤ x / 2 - 24) →  -- lower bound after brother takes candies
  (x / 2 - 24 ≤ 11) →  -- upper bound after brother takes candies
  (x = 72 ∨ x = 76) :=
by sorry

end tracy_candies_l3984_398478


namespace ellipse_foci_distance_l3984_398477

/-- The distance between foci of an ellipse with given semi-major and semi-minor axes -/
theorem ellipse_foci_distance (a b : ℝ) (ha : a = 7) (hb : b = 3) :
  2 * Real.sqrt (a^2 - b^2) = 4 * Real.sqrt 10 := by sorry

end ellipse_foci_distance_l3984_398477


namespace box_height_is_twelve_l3984_398441

-- Define the box dimensions and costs
def box_base_length : ℝ := 20
def box_base_width : ℝ := 20
def cost_per_box : ℝ := 0.50
def total_volume_needed : ℝ := 2160000
def min_spending : ℝ := 225

-- Theorem to prove
theorem box_height_is_twelve :
  ∃ (h : ℝ), h > 0 ∧ 
    (total_volume_needed / (box_base_length * box_base_width * h)) * cost_per_box ≥ min_spending ∧
    ∀ (h' : ℝ), h' > h → 
      (total_volume_needed / (box_base_length * box_base_width * h')) * cost_per_box < min_spending ∧
    h = 12 :=
by sorry

end box_height_is_twelve_l3984_398441


namespace parallel_resistors_combined_resistance_l3984_398460

/-- The combined resistance of two resistors connected in parallel -/
def combined_resistance (r1 r2 : ℚ) : ℚ :=
  1 / (1 / r1 + 1 / r2)

/-- Theorem: The combined resistance of two resistors with 8 ohms and 9 ohms connected in parallel is 72/17 ohms -/
theorem parallel_resistors_combined_resistance :
  combined_resistance 8 9 = 72 / 17 := by
  sorry

end parallel_resistors_combined_resistance_l3984_398460


namespace intersection_A_complement_B_l3984_398465

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define set A
def A : Set Nat := {2, 3, 5, 6}

-- Define set B
def B : Set Nat := {1, 3, 4, 6, 7}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 5} := by
  sorry

end intersection_A_complement_B_l3984_398465


namespace min_value_of_reciprocal_sum_l3984_398470

theorem min_value_of_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (2 * a * (-1) - b * 2 + 2 = 0) → 
  (∀ x y : ℝ, 2 * a * x - b * y + 2 = 0 → x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∀ c d : ℝ, c > 0 → d > 0 → (2 * c * (-1) - d * 2 + 2 = 0) → 1/a + 1/b ≤ 1/c + 1/d) →
  1/a + 1/b = 4 :=
sorry

end min_value_of_reciprocal_sum_l3984_398470


namespace students_in_front_of_yuna_l3984_398469

theorem students_in_front_of_yuna (total_students : ℕ) (students_behind_yuna : ℕ) : 
  total_students = 25 → students_behind_yuna = 9 → total_students - (students_behind_yuna + 1) = 15 := by
  sorry


end students_in_front_of_yuna_l3984_398469


namespace probability_divisible_by_15_l3984_398467

/-- The set of digits used to form the six-digit number -/
def digits : Finset Nat := {1, 2, 3, 4, 5, 9}

/-- The number of digits -/
def n : Nat := 6

/-- A permutation of the digits -/
def Permutation := Fin n → Fin n

/-- The set of all permutations -/
def allPermutations : Finset Permutation := sorry

/-- Predicate to check if a permutation results in a number divisible by 15 -/
def isDivisibleBy15 (p : Permutation) : Prop := sorry

/-- The number of permutations that result in a number divisible by 15 -/
def divisibleBy15Count : Nat := sorry

/-- The total number of permutations -/
def totalPermutations : Nat := Finset.card allPermutations

theorem probability_divisible_by_15 :
  (divisibleBy15Count : ℚ) / totalPermutations = 1 / 6 := by sorry

end probability_divisible_by_15_l3984_398467


namespace better_performance_criterion_l3984_398437

/-- Represents a shooter's performance statistics -/
structure ShooterStats where
  average_score : ℝ
  standard_deviation : ℝ

/-- Defines when a shooter has better performance than another -/
def better_performance (a b : ShooterStats) : Prop :=
  a.average_score > b.average_score ∧ a.standard_deviation < b.standard_deviation

/-- Theorem stating that a shooter with higher average score and lower standard deviation
    has better performance -/
theorem better_performance_criterion (shooter_a shooter_b : ShooterStats)
  (h1 : shooter_a.average_score > shooter_b.average_score)
  (h2 : shooter_a.standard_deviation < shooter_b.standard_deviation) :
  better_performance shooter_a shooter_b := by
  sorry

end better_performance_criterion_l3984_398437


namespace circumcenter_coordinates_l3984_398494

/-- A quadrilateral in 2D space -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The circumcenter of a quadrilateral -/
def circumcenter (q : Quadrilateral) : ℝ × ℝ := sorry

/-- A quadrilateral is inscribed in a circle if its circumcenter exists -/
def isInscribed (q : Quadrilateral) : Prop :=
  ∃ c : ℝ × ℝ, c = circumcenter q

theorem circumcenter_coordinates (q : Quadrilateral) (h : isInscribed q) :
  circumcenter q = (6, 1) := by sorry

end circumcenter_coordinates_l3984_398494


namespace triangle_area_increase_l3984_398400

theorem triangle_area_increase (a b θ : ℝ) (ha : a > 0) (hb : b > 0) (hθ : 0 < θ ∧ θ < π) :
  let original_area := (1/2) * a * b * Real.sin θ
  let new_area := (1/2) * (3*a) * (2*b) * Real.sin θ
  new_area = 6 * original_area := by
sorry

end triangle_area_increase_l3984_398400


namespace largest_integer_proof_l3984_398461

theorem largest_integer_proof (x : ℝ) (h : 20 * Real.sin x = 22 * Real.cos x) :
  ⌊(1 / (Real.sin x * Real.cos x) - 1)^7⌋ = 1 := by
  sorry

end largest_integer_proof_l3984_398461


namespace consecutive_product_111222_l3984_398493

theorem consecutive_product_111222 (b : ℕ) :
  b * (b + 1) = 111222 → b = 333 := by sorry

end consecutive_product_111222_l3984_398493


namespace infinitely_many_losing_positions_l3984_398464

/-- The set of numbers from which the first player loses -/
def losingSet : Set ℕ := sorry

/-- A number is a winning position if it's not in the losing set -/
def winningPosition (n : ℕ) : Prop := n ∉ losingSet

/-- A perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- The property that defines a losing position -/
def isLosingPosition (n : ℕ) : Prop :=
  ∀ k : ℕ, isPerfectSquare k → k ≤ n → winningPosition (n - k)

/-- The main theorem: there are infinitely many losing positions -/
theorem infinitely_many_losing_positions :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ isLosingPosition n :=
sorry

end infinitely_many_losing_positions_l3984_398464


namespace veranda_width_l3984_398412

/-- Veranda width problem -/
theorem veranda_width (room_length room_width veranda_area : ℝ) 
  (h1 : room_length = 17)
  (h2 : room_width = 12)
  (h3 : veranda_area = 132)
  (h4 : veranda_area = (room_length + 2 * w) * (room_width + 2 * w) - room_length * room_width)
  : w = 2 := by
  sorry

end veranda_width_l3984_398412


namespace line_mb_product_l3984_398473

/-- Given a line y = mx + b passing through points (0, -3) and (2, 3), prove that mb = -9 -/
theorem line_mb_product (m b : ℝ) : 
  (0 : ℝ) = m * 0 + b → -- The line passes through (0, -3)
  (-3 : ℝ) = m * 0 + b → -- The line passes through (0, -3)
  (3 : ℝ) = m * 2 + b → -- The line passes through (2, 3)
  m * b = -9 := by
sorry

end line_mb_product_l3984_398473


namespace liz_shopping_cost_l3984_398417

def problem (recipe_book_cost baking_dish_cost ingredient_cost apron_cost mixer_cost measuring_cups_cost spice_cost discount : ℝ) : Prop :=
  let total_cost := 
    recipe_book_cost + 
    baking_dish_cost + 
    (5 * ingredient_cost) + 
    apron_cost + 
    mixer_cost + 
    measuring_cups_cost + 
    (4 * spice_cost) - 
    discount
  total_cost = 84.5 ∧
  recipe_book_cost = 6 ∧
  baking_dish_cost = 2 * recipe_book_cost ∧
  ingredient_cost = 3 ∧
  apron_cost = recipe_book_cost + 1 ∧
  mixer_cost = 3 * baking_dish_cost ∧
  measuring_cups_cost = apron_cost / 2 ∧
  spice_cost = 2 ∧
  discount = 3

theorem liz_shopping_cost : ∃ (recipe_book_cost baking_dish_cost ingredient_cost apron_cost mixer_cost measuring_cups_cost spice_cost discount : ℝ),
  problem recipe_book_cost baking_dish_cost ingredient_cost apron_cost mixer_cost measuring_cups_cost spice_cost discount :=
by sorry

end liz_shopping_cost_l3984_398417


namespace existence_of_1000_consecutive_with_five_primes_l3984_398421

theorem existence_of_1000_consecutive_with_five_primes :
  (∃ n : ℕ, ∀ k ∈ Finset.range 1000, ¬ Nat.Prime (n + k + 2)) →
  (∃ m : ℕ, (Finset.filter (λ k => Nat.Prime (m + k + 1)) (Finset.range 1000)).card = 5) :=
by sorry

end existence_of_1000_consecutive_with_five_primes_l3984_398421


namespace sqrt_x_div_sqrt_y_l3984_398462

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (25*x)/(73*y)) :
  Real.sqrt x / Real.sqrt y = 5/2 := by
sorry

end sqrt_x_div_sqrt_y_l3984_398462


namespace descending_order_original_statement_l3984_398434

theorem descending_order : 0.38 > 0.373 ∧ 0.373 > 0.37 := by
  sorry

-- Define 37% as 0.37
def thirty_seven_percent : ℝ := 0.37

-- Prove that the original statement holds
theorem original_statement : 0.38 > 0.373 ∧ 0.373 > thirty_seven_percent := by
  sorry

end descending_order_original_statement_l3984_398434


namespace compound_has_six_hydrogen_atoms_l3984_398402

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The total molecular weight of the compound in g/mol -/
def total_weight : ℝ := 122

/-- The number of carbon atoms in the compound -/
def carbon_count : ℕ := 7

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 2

/-- Calculate the molecular weight of the compound given the number of hydrogen atoms -/
def molecular_weight (hydrogen_count : ℕ) : ℝ :=
  carbon_weight * carbon_count + oxygen_weight * oxygen_count + hydrogen_weight * hydrogen_count

/-- Theorem stating that the compound has 6 hydrogen atoms -/
theorem compound_has_six_hydrogen_atoms :
  ∃ (n : ℕ), molecular_weight n = total_weight ∧ n = 6 := by
  sorry

end compound_has_six_hydrogen_atoms_l3984_398402


namespace initial_amount_satisfies_equation_l3984_398427

/-- The initial amount of money the man has --/
def initial_amount : ℝ := 6.25

/-- The amount spent at each shop --/
def amount_spent : ℝ := 10

/-- The equation representing the man's transactions --/
def transaction_equation (x : ℝ) : Prop :=
  2 * (2 * (2 * x - amount_spent) - amount_spent) - amount_spent = 0

/-- Theorem stating that the initial amount satisfies the transaction equation --/
theorem initial_amount_satisfies_equation : 
  transaction_equation initial_amount := by sorry

end initial_amount_satisfies_equation_l3984_398427


namespace purchase_cost_l3984_398413

theorem purchase_cost (x y z : ℚ) 
  (eq1 : 4 * x + 9/2 * y + 12 * z = 6)
  (eq2 : 12 * x + 6 * y + 6 * z = 8) :
  4 * x + 3 * y + 6 * z = 4 := by
sorry

end purchase_cost_l3984_398413


namespace function_properties_l3984_398459

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 3^x else m - x^2

theorem function_properties :
  (∀ m < 0, ¬ ∃ x, f m x = 0) ∧
  f (1/9) (f (1/9) (-1)) = 0 := by sorry

end function_properties_l3984_398459


namespace average_rounds_played_l3984_398426

/-- Represents the distribution of golf rounds played by members -/
def golf_distribution : List (Nat × Nat) := [(1, 3), (2, 4), (3, 6), (4, 3), (5, 2)]

/-- Calculates the total number of rounds played -/
def total_rounds (dist : List (Nat × Nat)) : Nat :=
  dist.foldr (fun p acc => p.1 * p.2 + acc) 0

/-- Calculates the total number of golfers -/
def total_golfers (dist : List (Nat × Nat)) : Nat :=
  dist.foldr (fun p acc => p.2 + acc) 0

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : Rat) : Int :=
  if x - x.floor < 1/2 then x.floor else x.ceil

theorem average_rounds_played : 
  round_to_nearest ((total_rounds golf_distribution : Rat) / total_golfers golf_distribution) = 3 := by
  sorry

end average_rounds_played_l3984_398426


namespace chocolate_profit_l3984_398491

theorem chocolate_profit (num_bars : ℕ) (cost_per_bar : ℝ) (total_selling_price : ℝ) (packaging_cost_per_bar : ℝ) :
  num_bars = 5 →
  cost_per_bar = 5 →
  total_selling_price = 90 →
  packaging_cost_per_bar = 2 →
  total_selling_price - (num_bars * cost_per_bar + num_bars * packaging_cost_per_bar) = 55 := by
  sorry

end chocolate_profit_l3984_398491


namespace quadratic_root_in_unit_interval_l3984_398423

theorem quadratic_root_in_unit_interval (a b c : ℝ) (h : 2*a + 3*b + 6*c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end quadratic_root_in_unit_interval_l3984_398423


namespace mixed_gender_more_likely_l3984_398475

def child_gender := Bool

def prob_all_same_gender (n : ℕ) : ℚ :=
  (1 / 2) ^ n

def prob_mixed_gender (n : ℕ) : ℚ :=
  1 - prob_all_same_gender n

theorem mixed_gender_more_likely (n : ℕ) (h : n = 3) :
  prob_mixed_gender n > prob_all_same_gender n :=
sorry

end mixed_gender_more_likely_l3984_398475


namespace linear_equation_exponent_l3984_398446

theorem linear_equation_exponent (k : ℝ) : 
  (∀ x, ∃ a b : ℝ, x^(2*k - 1) + 2 = a*x + b) → k = 1 := by
  sorry

end linear_equation_exponent_l3984_398446


namespace f_has_zero_at_two_two_is_zero_point_of_f_l3984_398443

/-- A function that has a zero point at 2 -/
def f (x : ℝ) : ℝ := x - 2

/-- Theorem stating that f has a zero point at 2 -/
theorem f_has_zero_at_two : f 2 = 0 := by
  sorry

/-- Definition of a zero point -/
def is_zero_point (g : ℝ → ℝ) (x : ℝ) : Prop := g x = 0

/-- Theorem stating that 2 is a zero point of f -/
theorem two_is_zero_point_of_f : is_zero_point f 2 := by
  sorry

end f_has_zero_at_two_two_is_zero_point_of_f_l3984_398443


namespace car_speed_problem_l3984_398445

/-- The speed of Car A in km/h -/
def speed_A : ℝ := 80

/-- The time taken by Car A in hours -/
def time_A : ℝ := 5

/-- The speed of Car B in km/h -/
def speed_B : ℝ := 100

/-- The time taken by Car B in hours -/
def time_B : ℝ := 2

/-- The ratio of distances covered by Car A and Car B -/
def distance_ratio : ℝ := 2

theorem car_speed_problem :
  speed_A * time_A = distance_ratio * speed_B * time_B :=
sorry

end car_speed_problem_l3984_398445


namespace condition_for_inequality_l3984_398496

theorem condition_for_inequality (a b c : ℝ) :
  (¬ (∀ c, a > b → a * c^2 > b * c^2)) ∧
  ((a * c^2 > b * c^2) → a > b) :=
by sorry

end condition_for_inequality_l3984_398496


namespace shooting_range_problem_l3984_398448

theorem shooting_range_problem :
  ∀ (total_targets : ℕ) 
    (red_targets green_targets : ℕ) 
    (red_score green_score : ℚ)
    (hit_red_targets : ℕ),
  total_targets = 100 →
  total_targets = red_targets + green_targets →
  red_targets < green_targets / 3 →
  red_score = 10 →
  green_score = 8.5 →
  (green_score * green_targets + red_score * hit_red_targets : ℚ) = 
    (green_score * green_targets + red_score * red_targets : ℚ) →
  red_targets = 20 := by
sorry

end shooting_range_problem_l3984_398448


namespace min_perimeter_is_18_l3984_398410

/-- Represents a triangle with side lengths a and b, where a = AB = BC and b = AC -/
structure IsoscelesTriangle where
  a : ℕ
  b : ℕ

/-- Represents the incircle and excircles of the triangle -/
structure TriangleCircles (t : IsoscelesTriangle) where
  inradius : ℝ
  exradius_A : ℝ
  exradius_B : ℝ
  exradius_C : ℝ

/-- Represents the smaller circle φ -/
structure SmallerCircle (t : IsoscelesTriangle) (c : TriangleCircles t) where
  radius : ℝ

/-- Checks if the given triangle satisfies all the tangency conditions -/
def satisfiesTangencyConditions (t : IsoscelesTriangle) (c : TriangleCircles t) (φ : SmallerCircle t c) : Prop :=
  c.exradius_A = c.inradius + c.exradius_A ∧
  c.exradius_B = c.inradius + c.exradius_B ∧
  c.exradius_C = c.inradius + c.exradius_C ∧
  φ.radius = c.inradius - c.exradius_A

/-- The main theorem stating the minimum perimeter -/
theorem min_perimeter_is_18 :
  ∃ (t : IsoscelesTriangle) (c : TriangleCircles t) (φ : SmallerCircle t c),
    satisfiesTangencyConditions t c φ ∧
    ∀ (t' : IsoscelesTriangle) (c' : TriangleCircles t') (φ' : SmallerCircle t' c'),
      satisfiesTangencyConditions t' c' φ' →
      2 * t.a + t.b ≤ 2 * t'.a + t'.b ∧
      2 * t.a + t.b = 18 :=
sorry

end min_perimeter_is_18_l3984_398410


namespace cost_effective_plan_l3984_398419

/-- Represents the ticket purchasing scenario for a group of employees visiting a scenic spot. -/
structure TicketScenario where
  totalEmployees : ℕ
  regularPrice : ℕ
  groupDiscountRate : ℚ
  womenDiscountRate : ℚ
  minGroupSize : ℕ

/-- Calculates the cost of tickets with women's discount applied. -/
def womenDiscountCost (s : TicketScenario) (numWomen : ℕ) : ℚ :=
  s.regularPrice * s.womenDiscountRate * numWomen + s.regularPrice * (s.totalEmployees - numWomen)

/-- Calculates the cost of tickets with group discount applied. -/
def groupDiscountCost (s : TicketScenario) : ℚ :=
  s.totalEmployees * s.regularPrice * (1 - s.groupDiscountRate)

/-- Theorem stating the conditions for the most cost-effective ticket purchasing plan. -/
theorem cost_effective_plan (s : TicketScenario) (numWomen : ℕ) :
  s.totalEmployees = 30 ∧
  s.regularPrice = 80 ∧
  s.groupDiscountRate = 1/5 ∧
  s.womenDiscountRate = 1/2 ∧
  s.minGroupSize = 30 ∧
  numWomen ≤ s.totalEmployees →
  (numWomen < 12 → groupDiscountCost s < womenDiscountCost s numWomen) ∧
  (numWomen = 12 → groupDiscountCost s = womenDiscountCost s numWomen) ∧
  (numWomen > 12 → groupDiscountCost s > womenDiscountCost s numWomen) :=
by sorry

end cost_effective_plan_l3984_398419


namespace f_properties_l3984_398489

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

theorem f_properties :
  (∀ x > 0, f x ≥ 1) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ = f x₂ → x₁ + x₂ > 2) := by
  sorry

end f_properties_l3984_398489


namespace trailingZeros_2017_factorial_l3984_398404

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (Finset.range 4).sum fun i => (n / 5^(i + 1) : ℕ)

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => i + 1)

theorem trailingZeros_2017_factorial :
  trailingZeros 2017 = 502 :=
sorry

end trailingZeros_2017_factorial_l3984_398404


namespace boat_downstream_distance_l3984_398495

/-- Proof of downstream distance traveled by a boat given upstream travel time and distance, and stream speed. -/
theorem boat_downstream_distance
  (upstream_distance : ℝ)
  (upstream_time : ℝ)
  (stream_speed : ℝ)
  (downstream_time : ℝ)
  (h1 : upstream_distance = 75)
  (h2 : upstream_time = 15)
  (h3 : stream_speed = 3.75)
  (h4 : downstream_time = 8) :
  let upstream_speed := upstream_distance / upstream_time
  let boat_speed := upstream_speed + stream_speed
  let downstream_speed := boat_speed + stream_speed
  downstream_speed * downstream_time = 100 := by
  sorry


end boat_downstream_distance_l3984_398495


namespace fourth_term_is_8000_l3984_398466

/-- Geometric sequence with first term 1 and common ratio 20 -/
def geometric_sequence (n : ℕ) : ℕ :=
  1 * 20^(n - 1)

/-- The fourth term of the geometric sequence is 8000 -/
theorem fourth_term_is_8000 : geometric_sequence 4 = 8000 := by
  sorry

end fourth_term_is_8000_l3984_398466


namespace geometric_sum_eight_terms_l3984_398479

theorem geometric_sum_eight_terms : 
  let a : ℕ := 2
  let r : ℕ := 2
  let n : ℕ := 8
  a * (r^n - 1) / (r - 1) = 510 := by
  sorry

end geometric_sum_eight_terms_l3984_398479


namespace james_weekly_income_l3984_398483

/-- Calculates the weekly income from car rental given hourly rate, hours per day, and days per week. -/
def weekly_income (hourly_rate : ℝ) (hours_per_day : ℝ) (days_per_week : ℝ) : ℝ :=
  hourly_rate * hours_per_day * days_per_week

/-- Proves that James' weekly income from car rental is $640 given the specified conditions. -/
theorem james_weekly_income :
  let hourly_rate : ℝ := 20
  let hours_per_day : ℝ := 8
  let days_per_week : ℝ := 4
  weekly_income hourly_rate hours_per_day days_per_week = 640 := by
  sorry

#eval weekly_income 20 8 4

end james_weekly_income_l3984_398483


namespace hcf_problem_l3984_398492

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 1991) (h2 : Nat.lcm a b = 181) :
  Nat.gcd a b = 11 := by
  sorry

end hcf_problem_l3984_398492


namespace opposite_of_negative_fraction_l3984_398428

theorem opposite_of_negative_fraction :
  -(-(7 : ℚ) / 3) = 7 / 3 := by sorry

end opposite_of_negative_fraction_l3984_398428


namespace mary_income_proof_l3984_398450

/-- Calculates Mary's total income for a week --/
def maryIncome (maxHours regularhourlyRate overtimeRate1 overtimeRate2 bonus dues : ℝ) : ℝ :=
  let regularPay := regularhourlyRate * 20
  let overtimePay1 := overtimeRate1 * 20
  let overtimePay2 := overtimeRate2 * 20
  regularPay + overtimePay1 + overtimePay2 + bonus - dues

/-- Proves that Mary's total income is $650 given the specified conditions --/
theorem mary_income_proof :
  let maxHours : ℝ := 60
  let regularRate : ℝ := 8
  let overtimeRate1 : ℝ := regularRate * 1.25
  let overtimeRate2 : ℝ := regularRate * 1.5
  let bonus : ℝ := 100
  let dues : ℝ := 50
  maryIncome maxHours regularRate overtimeRate1 overtimeRate2 bonus dues = 650 := by
  sorry

#eval maryIncome 60 8 10 12 100 50

end mary_income_proof_l3984_398450


namespace wall_width_is_eight_l3984_398488

/-- Proves that the width of a wall with given proportions and volume is 8 meters -/
theorem wall_width_is_eight (w h l : ℝ) (h_height : h = 6 * w) (h_length : l = 7 * h) (h_volume : w * h * l = 129024) :
  w = 8 := by
  sorry

end wall_width_is_eight_l3984_398488


namespace exponent_multiplication_and_zero_power_l3984_398487

theorem exponent_multiplication_and_zero_power :
  (∀ x : ℝ, x^2 * x^4 = x^6) ∧ ((-5^2)^0 = 1) := by
  sorry

end exponent_multiplication_and_zero_power_l3984_398487


namespace triangular_coin_array_l3984_398444

theorem triangular_coin_array (N : ℕ) : (N * (N + 1)) / 2 = 3003 → N = 77 := by
  sorry

end triangular_coin_array_l3984_398444


namespace tax_saving_theorem_l3984_398447

theorem tax_saving_theorem (old_rate new_rate : ℝ) (saving : ℝ) (income : ℝ) : 
  old_rate = 0.45 → 
  new_rate = 0.30 → 
  saving = 7200 → 
  (old_rate - new_rate) * income = saving → 
  income = 48000 := by
sorry

end tax_saving_theorem_l3984_398447


namespace airplane_seats_l3984_398468

theorem airplane_seats (total_seats : ℕ) (first_class : ℕ) (coach_class : ℕ) : 
  total_seats = 387 →
  coach_class = 4 * first_class + 2 →
  first_class + coach_class = total_seats →
  coach_class = 310 := by
sorry

end airplane_seats_l3984_398468


namespace inequality_proof_l3984_398490

theorem inequality_proof (x y z : ℝ) (h : x^2 + y^2 + z^2 = 3) :
  x^3 - (y^2 + y*z + z^2)*x + y*z*(y + z) ≤ 3 * Real.sqrt 3 ∧
  (x^3 - (y^2 + y*z + z^2)*x + y*z*(y + z) = 3 * Real.sqrt 3 ↔ 
   x = Real.sqrt 3 ∧ y = 0 ∧ z = 0) :=
by sorry

end inequality_proof_l3984_398490


namespace disease_test_probability_l3984_398414

theorem disease_test_probability (disease_prevalence : ℝ) 
  (test_sensitivity : ℝ) (test_specificity : ℝ) : 
  disease_prevalence = 1/1000 →
  test_sensitivity = 1 →
  test_specificity = 0.95 →
  (disease_prevalence * test_sensitivity) / 
  (disease_prevalence * test_sensitivity + 
   (1 - disease_prevalence) * (1 - test_specificity)) = 100/5095 := by
sorry

end disease_test_probability_l3984_398414


namespace volume_of_specific_tetrahedron_l3984_398472

/-- A regular tetrahedron with specific properties -/
structure RegularTetrahedron where
  -- The distance from the midpoint of the height to a lateral face
  midpoint_to_face : ℝ
  -- The distance from the midpoint of the height to a lateral edge
  midpoint_to_edge : ℝ

/-- The volume of a regular tetrahedron -/
def volume (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the volume of a specific regular tetrahedron -/
theorem volume_of_specific_tetrahedron :
  ∃ (t : RegularTetrahedron),
    t.midpoint_to_face = 2 ∧
    t.midpoint_to_edge = Real.sqrt 10 ∧
    volume t = 80 * Real.sqrt 15 := by sorry

end volume_of_specific_tetrahedron_l3984_398472


namespace sqrt_x_minus_one_range_l3984_398406

-- Define the property of x that makes √(x-1) meaningful
def is_meaningful (x : ℝ) : Prop := x - 1 ≥ 0

-- Theorem stating the range of x where √(x-1) is meaningful
theorem sqrt_x_minus_one_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_range_l3984_398406


namespace regions_divisible_by_six_l3984_398439

/-- Represents a triangle with sides divided into congruent segments --/
structure DividedTriangle where
  segments : ℕ
  segments_pos : segments > 0

/-- Calculates the number of regions formed in a divided triangle --/
def num_regions (t : DividedTriangle) : ℕ :=
  t.segments^2 + (2*t.segments - 1) * (t.segments - 1) - r t
where
  /-- Number of points where three lines intersect (excluding vertices) --/
  r (t : DividedTriangle) : ℕ := sorry

/-- The main theorem stating that the number of regions is divisible by 6 --/
theorem regions_divisible_by_six (t : DividedTriangle) (h : t.segments = 2002) :
  6 ∣ num_regions t := by sorry

end regions_divisible_by_six_l3984_398439


namespace division_problem_l3984_398422

theorem division_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 95 →
  divisor = 15 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  quotient = 6 := by
  sorry

end division_problem_l3984_398422


namespace quadratic_form_equivalence_l3984_398436

theorem quadratic_form_equivalence (b : ℝ) (n : ℝ) :
  b < 0 →
  (∀ x, x^2 + b*x - 36 = (x + n)^2 - 20) →
  b = -8 := by
sorry

end quadratic_form_equivalence_l3984_398436
